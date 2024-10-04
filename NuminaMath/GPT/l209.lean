import Mathlib

namespace agnes_twice_jane_in_years_l209_209044

def agnes_age := 25
def jane_age := 6

theorem agnes_twice_jane_in_years (x : ℕ) : 
  25 + x = 2 * (6 + x) → x = 13 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry to skip them
  sorry

end agnes_twice_jane_in_years_l209_209044


namespace systematic_sampling_employee_l209_209874

theorem systematic_sampling_employee
    (n : ℕ)
    (employees : Finset ℕ)
    (sample : Finset ℕ)
    (h_n_52 : n = 52)
    (h_employees : employees = Finset.range 52)
    (h_sample_size : sample.card = 4)
    (h_systematic_sample : sample ⊆ employees)
    (h_in_sample : {6, 32, 45} ⊆ sample) :
    19 ∈ sample :=
by
  -- conditions 
  have h0 : 6 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h1 : 32 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h2 : 45 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h_arith : 6 + 45 = 32 + 19 :=
    by linarith
  sorry

end systematic_sampling_employee_l209_209874


namespace find_coordinates_with_respect_to_origin_l209_209303

-- Define the Cartesian coordinate system and properties
def Cartesian := Type

-- Given point
def point := (x : ℤ, y : ℤ)

-- Function to find the coordinates of a point with respect to the origin
def coordinates_with_respect_to_origin (p : point) : point :=
  (-p.1, -p.2)

-- Theorem statement
theorem find_coordinates_with_respect_to_origin :
  coordinates_with_respect_to_origin (-3, 2) = (3, -2) :=
by
  sorry

end find_coordinates_with_respect_to_origin_l209_209303


namespace probability_tile_in_TEST_l209_209911

def STATISTICS : List Char := ['S', 'T', 'A', 'T', 'I', 'S', 'T', 'I', 'C', 'S']
def TEST : List Char := ['T', 'E', 'S', 'T']

def count_occurrences : List Char → Char → Nat
| [], _ => 0
| (x :: xs), c => if x = c then 1 + count_occurrences xs c else count_occurrences xs c

def count_letters (letters : List Char) (word : List Char) : Nat :=
  letters.foldl (λ acc letter => acc + count_occurrences word letter) 0

def probability_in_TEST (word : List Char) (target : List Char) : Rat :=
  let match_count := count_letters target word
  let total_count := word.length
  match_count.toRat / total_count.toRat

theorem probability_tile_in_TEST :
  probability_in_TEST STATISTICS TEST = 3/5 := by
  sorry

end probability_tile_in_TEST_l209_209911


namespace derivative_of_composite_at_one_l209_209279

-- Define the differentiable function f and its derivative at the point 3
variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (hf' : deriv f 3 = 9)

-- Theorem statement
theorem derivative_of_composite_at_one : deriv (λ x, f (3 * x^2)) 1 = 54 := by
  sorry

end derivative_of_composite_at_one_l209_209279


namespace pythagorean_triple_divisible_by_12_l209_209740

open Nat

theorem pythagorean_triple_divisible_by_12 
  (m n : ℕ) 
  (h_rel_prime : gcd m n = 1) 
  (h_opposite_parity : (odd m ∧ even n) ∨ (even m ∧ odd n)) :
  let a := m^2 - n^2
  let b := 2 * m * n
  (a * b) % 12 = 0 :=
by
  sorry

end pythagorean_triple_divisible_by_12_l209_209740


namespace valid_triangles_count_l209_209667

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def count_valid_triangles_with_perimeter_12 : ℕ :=
  (List.range 12).filter (λ a => 
    (List.range 12).filter (λ b => 
      (List.range 12).filter (λ c => 
        a + b + c = 12 ∧ 
        is_triangle a b c
      ).length
    ).length
  ).length

theorem valid_triangles_count : count_valid_triangles_with_perimeter_12 = 6 := 
sorry

end valid_triangles_count_l209_209667


namespace v_2008_l209_209339

-- Defining the number of terms in the ith group.
def terms_in_group (i : ℕ) := i

-- Defining the starting number of each group and the modular condition.
def group_start_mod_condition (i : ℕ) : ℕ → Prop
| 1 := ∀ n, n = 1 + (n - 1) * 4
| j + 1 := ∀ n, n = i + (n - terms_in_group(j+1)) * 4

-- Defining the function g(n) from the solution based on the pattern.
noncomputable def g (n : ℕ) : ℕ := 2 * n^2 - (5 * n) / 2 + 4

-- Defining the problem statement.
theorem v_2008 : g 62 = 7297 :=
by
  have h : 62 * (62 + 1) / 2 = 2008 := by norm_num
  -- Based on the counting and g calculations already established
  exact h sorry

end v_2008_l209_209339


namespace count_triangles_with_positive_area_l209_209240

theorem count_triangles_with_positive_area : 
  let points : list (ℕ × ℕ) := [(x, y) | x ← [1, 2, 3, 4, 5], y ← [1, 2, 3, 4, 5]]
  let collinear (a b c : ℕ × ℕ) : Prop := (b.1 - a.1) * (c.2 - a.2) = (b.2 - a.2) * (c.1 - a.1)
  let non_degenerate (a b c : ℕ × ℕ) : Prop := ¬collinear a b c
  list.filter (λ t, non_degenerate t[0] t[1] t[2]) (points.combinations 3) = 2156 := by
  sorry

end count_triangles_with_positive_area_l209_209240


namespace trapezoid_area_sum_l209_209521

theorem trapezoid_area_sum (a b c d : ℝ) (ha : a = 4) (hb : b = 6) (hc : c = 8) (hd : d = 9) 
  (non_parallel : (a + b > c + d) ∧ (a + c > b + d) ∧ (a + d > b + c)) :
  3 * real.sqrt 15 := 
sorry

end trapezoid_area_sum_l209_209521


namespace domain_of_f_l209_209832

noncomputable def f : ℝ → ℝ :=
  λ x, log 3 (log 4 (log 5 x))

theorem domain_of_f :
  ∃ S : set ℝ, S = {x | x > 625} ∧ ∀ x ∈ S, f x = log 3 (log 4 (log 5 x)) :=
by
  sorry

end domain_of_f_l209_209832


namespace problem_U_complement_eq_l209_209990

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l209_209990


namespace compare_neg_sqrt_l209_209068

theorem compare_neg_sqrt :
  -5 > -Real.sqrt 26 := 
sorry

end compare_neg_sqrt_l209_209068


namespace sum_after_first_operation_sum_after_second_operation_sum_after_hundredth_operation_l209_209436

def initial_sequence : List ℤ := [3, 9, 8]

/-- After the first operation, the sum of all the new numbers added is 5 --/
theorem sum_after_first_operation : 
  ∑ (n : ℤ) in [9 - 3, 8 - 9], n = 5 :=
by
  calc
    (9 - 3) + (8 - 9) = 6 + (-1) : by simp
                   ... = 5       : by simp

/-- The sum of all the new numbers added in the second operation compared to the first operation is 5 --/
theorem sum_after_second_operation :
  ∑ (n : ℤ) in [3 - 3, 6 - 3, 9 - 6, (-1) - 9], n = 5 :=
by
  calc
    (3 - 3) + (6 - 3) + (9 - 6) + ((-1) - 9) = 0 + 3 + 3 + (-10) : by simp
                                           ... = -4            : by simp
                                           ... = 5             : by sorry

/-- The sum of all the new numbers added in the sequence obtained after the 100th operation compared to the sequence obtained after the 99th operation is 5 --/
theorem sum_after_hundredth_operation :
   (∑ (n : ℤ) in example_sequence 100, n ) - ( ∑ (n : ℤ) in example_sequence 99, n ) = 5 :=
by
  sorry

def example_sequence : ℕ → List ℤ
  | 0     => initial_sequence
  | (n+1) =>
    let lst := example_sequence n
    let diffs := let rec add_diffs (lst : List ℤ) : List ℤ :=
                   match lst with
                   | [] | _::[] => []
                   | x::(y::_ as t) => (y - x) :: add_diffs t
                 in add_diffs lst
    let interleave := diffs.zipWith (λ x y => [x, y]) lst diffs
    interleave.join

#print initial_seq
#print sum_after_fst_opr
#print sum_after_snd_opr
#print sum_after_100th_opr

end sum_after_first_operation_sum_after_second_operation_sum_after_hundredth_operation_l209_209436


namespace area_of_union_of_half_disks_l209_209601

noncomputable def union_area_of_half_disks : ℝ :=
  let D_y (y : ℝ) (H : 0 ≤ y ∧ y ≤ 2) : set (ℝ × ℝ) :=
    {p : ℝ × ℝ | let x := p.1, let x2 := p.2 in
      x^2 + (x2 - y)^2 ≤ 1 ∧ x ≥ 0 ∧ x + x2 ≤ 2}
  let D_union : set (ℝ × ℝ) :=
    ⋃ (y : ℝ) (H : 0 ≤ y ∧ y ≤ 2), D_y y H
  measure_theory.volume D_union

theorem area_of_union_of_half_disks : union_area_of_half_disks = π :=
by sorry

end area_of_union_of_half_disks_l209_209601


namespace contribution_split_l209_209484

theorem contribution_split (total : ℝ) (a_investment_months : ℝ) (b_investment_months : ℝ) (a_received : ℝ) (b_received : ℝ) :
  total = 1500 → a_investment_months = 3 → b_investment_months = 4 → 
  a_received = 690 → b_received = 1080 → 
  ∃ (a_contribution b_contribution : ℝ), a_contribution + b_contribution = total ∧ a_contribution = 600 ∧ b_contribution = 900 :=
by
  intros ht ha hb hra hrb
  use [600, 900]
  split
  · exact ht
  split
  · exact rfl
  · exact rfl

end contribution_split_l209_209484


namespace number_of_positive_area_triangles_l209_209243

def integer_points := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

def count_triangles_with_positive_area : Nat :=
  let total_points := 25 -- total number of integer points in the grid
  let total_combinations := Nat.choose total_points 3 -- total possible combinations
  let degenerate_cases := 136 -- total degenerate (collinear) cases
  total_combinations - degenerate_cases

theorem number_of_positive_area_triangles : count_triangles_with_positive_area = 2164 := by
  sorry

end number_of_positive_area_triangles_l209_209243


namespace probability_reaches_l209_209388

-- Definition of the conditions
def start_point := (0, 0)
def end_point := (3, 1)
def total_steps := 6
def directions := ['L', 'R', 'U', 'D']

-- Probability Calculation Proof Statement
theorem probability_reaches (start_point = (0, 0))
      (end_point = (3, 1))
      (total_steps = 6)
      (m = 15)
      (n = 256) :
      (p = 15/256) ∧ coprime 15 256 ∧ m + n = 271 :=
by
  -- Proof omitted
  sorry

end probability_reaches_l209_209388


namespace geometric_sequence_a7_l209_209308

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a1 : a 1 = 2) (h_a3 : a 3 = 4) : a 7 = 16 := 
sorry

end geometric_sequence_a7_l209_209308


namespace angle_DEF_EDF_proof_l209_209494

theorem angle_DEF_EDF_proof (angle_DOE : ℝ) (angle_EOD : ℝ) 
  (h1 : angle_DOE = 130) (h2 : angle_EOD = 90) :
  let angle_DEF := 45
  let angle_EDF := 45
  angle_DEF = 45 ∧ angle_EDF = 45 :=
by
  sorry

end angle_DEF_EDF_proof_l209_209494


namespace vectors_form_non_convex_quadrilateral_and_self_intersecting_line_l209_209953

def non_parallel (u v : ℝ × ℝ) := ∃ (a b c d : ℝ), u.1 ≠ v.1 ∧ u.2 ≠ v.2 ∧ ((a * u.1 + b * u.2 + c * v.1 + d * v.2) = 0)

theorem vectors_form_non_convex_quadrilateral_and_self_intersecting_line
    (u v w x : ℝ × ℝ) 
    (h_sum : u + v + w + x = (0, 0)) 
    (h_non_parallel_uv : non_parallel u v) 
    (h_non_parallel_uw : non_parallel u w) 
    (h_non_parallel_ux : non_parallel u x) 
    (h_non_parallel_vw : non_parallel v w) 
    (h_non_parallel_vx : non_parallel v x) 
    (h_non_parallel_wx : non_parallel w x) :
    ∃ (p : ℝ × ℝ) (q : ℝ × ℝ) (r : ℝ × ℝ) (s : ℝ × ℝ), -- existence of points forming the quadrilateral
    (p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ p) -- distinct vertices
    ∧ (non_convex_quadrilateral p q r s) -- non-convex condition
    ∧ (self_intersecting_four_segment p q r s) := -- self-intersecting condition
sorry

end vectors_form_non_convex_quadrilateral_and_self_intersecting_line_l209_209953


namespace shooter_hit_rate_l209_209870

noncomputable def shooter_prob := 2 / 3

theorem shooter_hit_rate:
  ∀ (x : ℚ), (1 - x)^4 = 1 / 81 → x = shooter_prob :=
by
  intro x h
  -- Proof is omitted
  sorry

end shooter_hit_rate_l209_209870


namespace number_of_sheets_fallen_out_l209_209510

-- Define the conditions
def first_page := 387
def last_page := 738  -- Since this is the even permutation mentioned in the solution

-- The theorem stating the number of sheets that fell out
theorem number_of_sheets_fallen_out : 
  (last_page - first_page + 1) / 2 = 176 :=
begin
  sorry
end

end number_of_sheets_fallen_out_l209_209510


namespace problem_U_complement_eq_l209_209986

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l209_209986


namespace regular_octagon_diagonal_ratio_l209_209689

theorem regular_octagon_diagonal_ratio
  (s : ℝ)  -- side length of the regular octagon
  (A B C D E F G H : Type) -- vertices of the octagon
  (interior_angle := 135 : ℝ)
  (ac_length := s * real.sqrt (2 - real.sqrt 2))
  (ad_length := s * real.sqrt (2 + real.sqrt 2)) :
  (ac_length / ad_length) = real.sqrt ((2 - real.sqrt 2) / (2 + real.sqrt 2)) :=
by sorry

end regular_octagon_diagonal_ratio_l209_209689


namespace isosceles_triangle_shaded_area_sum_l209_209878

theorem isosceles_triangle_shaded_area_sum :
  ∀ (r : ℝ) (a b c : ℕ),
    r = 10 → 
    (∀ (x y : ℝ), x = 14 → y = 14 → sqrt (x ^ 2 + y ^ 2) = 20) →
    (∀ (sector_area triangle_area : ℝ), sector_area = 25 * real.pi → triangle_area = 98 → (2 * (sector_area - triangle_area) = 50 * real.pi - 196)) →
    a = 50 → b = 196 → c = 1 → a + b + c = 247 :=
by
  intros r a b c
  intros hr hpyth hareas ha hb hc
  rw [hr, ha, hb, hc]
  exact 247

#check isosceles_triangle_shaded_area_sum

end isosceles_triangle_shaded_area_sum_l209_209878


namespace cos_pi_zero_l209_209362

theorem cos_pi_zero : ∃ f : ℝ → ℝ, (∀ x, f x = (Real.cos x) ^ 2 + Real.cos x) ∧ f Real.pi = 0 := by
  sorry

end cos_pi_zero_l209_209362


namespace emily_collected_8484_eggs_l209_209569

def number_of_baskets : ℕ := 303
def eggs_per_basket : ℕ := 28
def total_eggs : ℕ := number_of_baskets * eggs_per_basket

theorem emily_collected_8484_eggs : total_eggs = 8484 :=
by
  sorry

end emily_collected_8484_eggs_l209_209569


namespace symmetric_point_P_correct_l209_209313

-- Define point P with coordinates (4, -3, 7)
def P : ℝ × ℝ × ℝ := (4, -3, 7)

-- Define the symmetry with respect to the xOy plane
def symmetric_with_respect_to_xOy (point : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (point.1, point.2, -point.3)

-- Define the expected symmetric point of P
def symmetric_point_P := symmetric_with_respect_to_xOy P

-- The target theorem statement: symmetric point of P is (4, -3, -7)
theorem symmetric_point_P_correct : symmetric_point_P = (4, -3, -7) :=
by
  -- Proof is not provided
  sorry

end symmetric_point_P_correct_l209_209313


namespace binary_ternary_product_l209_209548

theorem binary_ternary_product :
  let b1 := 1 * (2^3) + 1 * (2^2) + 0 * (2^1) + 1 * (2^0),
      t1 := 1 * (3^2) + 0 * (3^1) + 2 * (3^0)
  in b1 * t1 = 143 :=
by
  let b1 := 1 * (2^3) + 1 * (2^2) + 0 * (2^1) + 1 * (2^0),
      t1 := 1 * (3^2) + 0 * (3^1) + 2 * (3^0)
  exact sorry

end binary_ternary_product_l209_209548


namespace total_hours_l209_209440

def initial_sleep_hours : ℝ := 6
def work_hours : ℝ := 7
def single_trip_travel_hours : ℝ := 1.5
def travel_hours : ℝ := 2 * single_trip_travel_hours
def sleep_increase_fraction : ℝ := 1 / 3

def new_sleep_hours := initial_sleep_hours + initial_sleep_hours * sleep_increase_fraction

theorem total_hours (h₁ : initial_sleep_hours = 6) 
                    (h₂ : work_hours = 7) 
                    (h₃ : single_trip_travel_hours = 1.5)
                    (h₄ : travel_hours = 2 * single_trip_travel_hours)
                    (h₅ : sleep_increase_fraction = 1 / 3) :
    work_hours + new_sleep_hours + travel_hours = 18 :=
by
  sorry

end total_hours_l209_209440


namespace overlap_area_of_sectors_l209_209447

/--
Given two sectors of a circle with radius 10, with centers at points P and R respectively, 
one having a central angle of 45 degrees and the other having a central angle of 90 degrees, 
prove that the area of the shaded region where they overlap is 12.5π.
-/
theorem overlap_area_of_sectors 
  (r : ℝ) (θ₁ θ₂ : ℝ) (A₁ A₂ : ℝ)
  (h₀ : r = 10)
  (h₁ : θ₁ = 45)
  (h₂ : θ₂ = 90)
  (hA₁ : A₁ = (θ₁ / 360) * π * r ^ 2)
  (hA₂ : A₂ = (θ₂ / 360) * π * r ^ 2)
  : A₁ = 12.5 * π := 
sorry

end overlap_area_of_sectors_l209_209447


namespace last_digit_of_powerset_powerset_l209_209730

-- Define the function that computes the cardinality of the set of non-empty subsets of a set
def numNonEmptySubsets (S : Finset ℕ) : ℕ := (2 ^ S.card) - 1

-- Define the set [n] = {1, 2, ..., n}
def numberedSet (n : ℕ) : Finset ℕ := Finset.range (n + 1)

theorem last_digit_of_powerset_powerset :
  let n := 2013
  let A := numberedSet n
  let B := Finset.powerset A \ {∅}
  let C := Finset.powerset B \ {∅}
  (numNonEmptySubsets C) % 10 = 7 := 
by
  sorry

end last_digit_of_powerset_powerset_l209_209730


namespace ratio_increases_l209_209294

open Real

noncomputable def OG (r h : ℝ) : ℝ := r - h
noncomputable def CG (r h : ℝ) : ℝ := sqrt (r^2 - (r - h)^2)
noncomputable def area_rectangle (r h d : ℝ) : ℝ := d * 2 * CG r h
noncomputable def area_trapezoid (r h d : ℝ) : ℝ := ½ * d * (2 * CG r h + 2 * CG r h)
noncomputable def ratio_areas (r ir im : ℝ) (h d : ℝ) : ℝ := area_trapezoid r h d / area_rectangle ir im d

theorem ratio_increases (r₁ r₂ h d : ℝ) (hr₁ : r₁ = 10) (hd : d = 4) (hinc : r₁ < r₂) :
  ratio_areas r₂ r₁ 10 h d > ratio_areas r₁ 10 10 h d := 
sorry

end ratio_increases_l209_209294


namespace triangle_side_lengths_approx_l209_209449

noncomputable def approx_side_lengths (AB : ℝ) (BAC ABC : ℝ) : ℝ × ℝ :=
  let α := BAC * Real.pi / 180
  let β := ABC * Real.pi / 180
  let c := AB
  let β1 := (90 - (BAC)) * Real.pi / 180
  let m := 2 * c * α * (β1 + 3) / (9 - α * β1)
  let c1 := 2 * c * β1 * (α + 3) / (9 - α * β1)
  let β2 := β1 - β
  let γ1 := α + β
  let a1 := β2 / γ1 * (γ1 + 3) / (β2 + 3) * m
  let a := (9 - β2 * γ1) / (2 * γ1 * (β2 + 3)) * m
  let b := c1 - a1
  (a, b)

theorem triangle_side_lengths_approx (AB : ℝ) (BAC ABC : ℝ) (hAB : AB = 441) (hBAC : BAC = 16.2) (hABC : ABC = 40.6) :
  approx_side_lengths AB BAC ABC = (147, 344) := by
  sorry

end triangle_side_lengths_approx_l209_209449


namespace cost_of_ruler_l209_209751

def total_spent : ℕ := 74
def notebook_cost : ℕ := 35
def pencil_cost : ℕ := 7
def pencil_count : ℕ := 3

theorem cost_of_ruler (total_spent notebook_cost pencil_cost pencil_count : ℕ) : 
  let pencil_total := pencil_cost * pencil_count in
  let total_books_pencils := notebook_cost + pencil_total in
  let ruler_cost := total_spent - total_books_pencils in
  ruler_cost = 18 :=
by
  sorry

end cost_of_ruler_l209_209751


namespace PQ_eq_QR_l209_209537

-- Definitions for the problem

variables (O1 O2 : Type) [circle O1] [circle O2]
variables (P Q A B : Type) [point P] [point Q] [point A] [point B]
variables (O : Type) [circumcircle O (triangle P A B)]
variables (R M N : Type) [point R] [point M] [point N]

-- Conditions
hypothesis h1 : P ≠ Q
hypothesis h2 : O1.intersection O2 = {P, Q}
hypothesis h3 : tangent O2 (line P A)
hypothesis h4 : tangent O1 (line P B)
hypothesis h5 : PQ.intersects O at R

-- Theorem we want to prove
theorem PQ_eq_QR : length PQ = length QR :=
sorry

end PQ_eq_QR_l209_209537


namespace pure_imaginary_implies_m_eq_1_l209_209679

noncomputable def z (m : ℝ) : ℂ := (1 + m * complex.I) / (1 - complex.I)

theorem pure_imaginary_implies_m_eq_1 (m : ℝ) (h : im (z m) = z m) : m = 1 :=
by
  sorry

end pure_imaginary_implies_m_eq_1_l209_209679


namespace longest_side_66_inches_l209_209321

/-!
The problem is to prove that the length of the longest side of a rectangular window
constructed with 6 equal-size panes of glass, arranged with 2 rows and 3 columns,
with a height-to-width ratio of 3:4 for each pane, where the border around the panes 
is 3 inches wide and the borders between the panes are 1.5 inches wide, is 66 inches.
-/

def longest_side_of_window (x : ℝ) : ℝ :=
  let width := 3 * (4 * x) + 2 * 1.5 + 2 * 3 in
  let height := 2 * (3 * x) + 1.5 + 2 * 3 in
  max width height

theorem longest_side_66_inches :
  ∃ x : ℝ, max (3 * (4 * x) + 2 * 1.5 + 2 * 3) (2 * (3 * x) + 1.5 + 2 * 3) = 66 :=
by
  use 4.75
  have width := 3 * (4 * 4.75) + 2 * 1.5 + 2 * 3
  have height := 2 * (3 * 4.75) + 1.5 + 2 * 3
  have : max width height = 66 := by
    rw [←rfl, ←rfl]
    simp [width, height]
  exact this

end longest_side_66_inches_l209_209321


namespace largest_possible_floor_D_eq_947_l209_209869

-- Define a sample of integers and a unique mode
variables (sample : Fin 121 → ℕ) (mode : ℕ)
variables (h_mode_bounds : mode ≥ 1 ∧ mode ≤ 1000)
variables (h_sample_bounds : ∀ i, 1 ≤ sample i ∧ sample i ≤ 1000)
variables (h_unique_mode : ∃ unique_count : ℕ, unique_count > 1 ∧ ∀ y, (∀ i, sample i = y → y = mode) ∧ unique_count = (sample.count mode))

-- Define arithmetic mean
noncomputable def arithmetic_mean (sample : Fin 121 → ℕ) : ℚ :=
  (∑ i, sample i) / 121

-- Define difference D
noncomputable def D (sample : Fin 121 → ℕ) (mode : ℕ) : ℚ :=
  | mode - arithmetic_mean sample |

-- The main theorem to prove the largest possible value of ⌊D⌋
theorem largest_possible_floor_D_eq_947 : 
  ∃ S : Fin 121 → ℕ, ∃ x : ℕ, 
    (x ≥ 1 ∧ x ≤ 1000) ∧
    (∀ i, 1 ≤ S i ∧ S i ≤ 1000) ∧
    (∃ unique_count : ℕ, unique_count > 1 ∧ ∀ y, (∀ i, S i = y → y = x) ∧ unique_count = (S.count x)) ∧
    ⌊(D S x)⌋ = 947 :=
by 
  sorry

end largest_possible_floor_D_eq_947_l209_209869


namespace count_odd_factors_of_360_l209_209221

-- Defining the prime factorization of 360
def prime_factorization_360 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

-- Defining the exponent range based on prime factorization for an odd factor
def exponent_ranges : List (ℕ × List ℕ) := [(3, [0, 1, 2]), (5, [0, 1])]

theorem count_odd_factors_of_360 : ∃ n : ℕ, n = 6 :=
by
  -- Conditions based on prime factorization
  have h₁ : prime_factorization_360 = [(2, 3), (3, 2), (5, 1)], from rfl
  have h₂ : exponent_ranges = [(3, [0, 1, 2]), (5, [0, 1])], from rfl
  
  -- Definitions should align with the conditions in the problem
  use 6
  sorry

end count_odd_factors_of_360_l209_209221


namespace inequality_solution_l209_209147

theorem inequality_solution (a : ℝ) : (∀ x : ℝ, (a + 1) * x > a + 1 → x < 1) ↔ a < -1 := by
  sorry

end inequality_solution_l209_209147


namespace number_of_positive_area_triangles_l209_209245

def integer_points := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

def count_triangles_with_positive_area : Nat :=
  let total_points := 25 -- total number of integer points in the grid
  let total_combinations := Nat.choose total_points 3 -- total possible combinations
  let degenerate_cases := 136 -- total degenerate (collinear) cases
  total_combinations - degenerate_cases

theorem number_of_positive_area_triangles : count_triangles_with_positive_area = 2164 := by
  sorry

end number_of_positive_area_triangles_l209_209245


namespace m_plus_n_sum_l209_209087

theorem m_plus_n_sum :
  let m := 271
  let n := 273
  m + n = 544 :=
by {
  -- sorry included to skip the proof steps
  sorry
}

end m_plus_n_sum_l209_209087


namespace find_f6_l209_209413

variable {R : Type*} [AddGroup R] [Semiring R]

def functional_equation (f : R → R) :=
∀ x y : R, f (x + y) = f x + f y

theorem find_f6 (f : ℝ → ℝ) (h1 : functional_equation f) (h2 : f 4 = 10) : f 6 = 10 :=
sorry

end find_f6_l209_209413


namespace maximal_cardinality_proof_l209_209327

open Nat

noncomputable def maximal_cardinality (n : ℕ) (h : 2 ≤ n) : ℕ :=
  if h : n % 2 = 0
  then (n / 2) * (n / 2)
  else (n / 2) * ((n / 2) + 1)

theorem maximal_cardinality_proof (n : ℕ) (h : 2 ≤ n) :
  ∃ M : Finset (ℕ × ℕ), (∀ j k m : ℕ, 
  (j, k) ∈ M →  1 ≤ j → j < k → k ≤ n → (k, m) ∉ M) ∧ 
  M.card = maximal_cardinality n h :=
sorry

end maximal_cardinality_proof_l209_209327


namespace problem_U_complement_eq_l209_209988

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l209_209988


namespace num_lattice_points_on_curve_l209_209864

theorem num_lattice_points_on_curve : 
  {p : ℤ × ℤ // (p.1^2 - p.2^2 = 47)}.card = 4 :=
sorry

end num_lattice_points_on_curve_l209_209864


namespace fleas_cannot_reach_opposite_vertex_l209_209692

theorem fleas_cannot_reach_opposite_vertex :
  ∀ (A B C : ℤ × ℤ),
  (A = (0,0) ∨ A = (1,0) ∨ A = (0,1)) ∧
  (B = (0,0) ∨ B = (1,0) ∨ B = (0,1)) ∧
  (C = (0,0) ∨ C = (1,0) ∨ C = (0,1)) ∧
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) →
  ¬ ∃ (A' B' C' : ℤ × ℤ),
  (A' = (1,1) ∨ B' = (1,1) ∨ C' = (1,1)) ∧
  ∀ (A₁ A₂ B₁ B₂ C₁ C₂ : ℤ × ℤ), 
  (A₁ - B₁ = A₂ - B₂) ∧
  (A₁ ≠ A₂ ∧ B₁ ≠ B₂) ∧
  (A₂ = C₁ ∨ A₂ = C₂ ∨ B₂ = C₁ ∨ B₂ = C₂ ∨ C₁ = A₂ ∨ C₂ = A₂) :=
begin
  sorry
end

end fleas_cannot_reach_opposite_vertex_l209_209692


namespace base_conversion_l209_209463

theorem base_conversion (A B : ℕ) (hA : A < 8) (hB : B < 6) (h : 7 * A = 5 * B) : 8 * A + B = 47 :=
by
  sorry

end base_conversion_l209_209463


namespace complete_square_sum_l209_209286

theorem complete_square_sum (a h k : ℤ) : 
  (6 * (x - 2) ^ 2 - 14 = a * (x - h) ^ 2 + k) → (a = 6 ∧ h = 2 ∧ k = -14) → (a + h + k = -6) := 
by
  intros h1 h2
  cases h2 with ha hk
  rw ha.left at h1
  rw ha.right at h1
  exact sorry

end complete_square_sum_l209_209286


namespace odd_function_implies_a_eq_one_l209_209281

theorem odd_function_implies_a_eq_one (a : ℝ) (h : ∀ x : ℝ, f x = sin x / ((x - a) * (x + 1)) ∧ f (-x) = -f x) : a = 1 :=
begin
  sorry
end

end odd_function_implies_a_eq_one_l209_209281


namespace determine_m_l209_209982

def setA_is_empty (m: ℝ) : Prop :=
  { x : ℝ | m * x = 1 } = ∅

theorem determine_m (m: ℝ) (h: setA_is_empty m) : m = 0 :=
by sorry

end determine_m_l209_209982


namespace sin_1090_eq_cos_80_l209_209467

theorem sin_1090_eq_cos_80 :
  sin (1090 : ℝ) = cos (80 : ℝ) :=
sorry

end sin_1090_eq_cos_80_l209_209467


namespace sum_of_squares_l209_209683

theorem sum_of_squares (n : ℕ) (h : n * (n + 1) * (n + 2) = 12 * (3 * n + 3)) :
  n^2 + (n + 1)^2 + (n + 2)^2 = 29 := 
sorry

end sum_of_squares_l209_209683


namespace simplify_fraction_l209_209892

def expr1 : ℚ := 3
def expr2 : ℚ := 2
def expr3 : ℚ := 3
def expr4 : ℚ := 4
def expected : ℚ := 12 / 5

theorem simplify_fraction : (expr1 / (expr2 - (expr3 / expr4))) = expected := by
  sorry

end simplify_fraction_l209_209892


namespace find_AF_l209_209914

theorem find_AF (ABC : Type) [euclidean_geometry ABC]
  (A B C D E F G : ABC) (side_length : ℝ) (h_equilateral : equilateral ABC A B C)
  (h_side : side_length = 840)
  (h_BD_perp_BC : perpendicular BD BC)
  (h_ell_parallel_BC : parallel (line_through D F) (line_through B C))
  (h_AF_eq_FG : AF = FG)
  (h_area_ratio : (area A F G) / (area B E D) = 8 / 9) :
  AF = 336 :=
sorry

end find_AF_l209_209914


namespace number_of_even_blue_faces_cubes_l209_209043

theorem number_of_even_blue_faces_cubes : 
  let length := 6
  let width := 4
  let height := 2
  let total_cubes := length * width * height
  let corner_cubes := 8
  let edge_cubes := (4 * (length - 2)) + (2 * (width - 2)) + (4 * 0)
  let face_center_cubes := (2 * ((length - 2) * (width - 2)))
  let interior_cubes := ((length - 2) * (width - 2) * (height - 2))
  let even_blue_faces_cubes := edge_cubes + interior_cubes
  in even_blue_faces_cubes = 20 :=
by 
  let length := 6
  let width := 4
  let height := 2
  let total_cubes := length * width * height
  let corner_cubes := 8
  let edge_cubes := (4 * (length - 2)) + (2 * (width - 2)) + (4 * 0)
  let face_center_cubes := (2 * ((length - 2) * (width - 2)))
  let interior_cubes := ((length - 2) * (width - 2) * (height - 2))
  let even_blue_faces_cubes := edge_cubes + interior_cubes
  show even_blue_faces_cubes = 20, from
    sorry

end number_of_even_blue_faces_cubes_l209_209043


namespace find_m_l209_209956

-- Define the sets A and B
def A : Set ℕ := {1, 2, m}
def B : Set ℕ := {4, 7, 13}

-- Define the function f
def f (x : ℕ) : ℕ := 3 * x + 1

-- Define the condition that f maps A to B
def maps_to (f : ℕ → ℕ) (A B : Set ℕ) : Prop :=
  ∀ x ∈ A, f x ∈ B

-- Given hypothesis that A maps to B via f
axiom h : maps_to f A B

-- Prove that m = 4
theorem find_m : (m : ℕ) → 3 * m + 1 = 13 → m = 4 := by
  intros m hm
  sorry

end find_m_l209_209956


namespace total_trip_length_is_95_miles_l209_209323

theorem total_trip_length_is_95_miles
  (d : ℝ) -- total length of the trip
  (h1 : ∀ (m : ℝ), m = 30 → 0) -- first 30 miles use no gasoline
  (h2 : ∀ (m : ℝ), m = 70 → 0.03 * 70) -- next 70 miles use gasoline at 0.03 gallons per mile
  (h3 : ∀ (m : ℝ), m = d - 100 → 0.04 * (d - 100)) -- remaining miles use gasoline at 0.04 gallons per mile
  (h4 : 50 = d / (0.03 * 70 + 0.04 * (d - 100))) -- overall trip averages 50 miles per gallon
  : d = 95 := sorry

end total_trip_length_is_95_miles_l209_209323


namespace exponential_inequality_l209_209769

-- Define the conditions
variables {n : ℤ} {x : ℝ}

theorem exponential_inequality 
  (h1 : n ≥ 2) 
  (h2 : |x| < 1) 
  : 2^n > (1 - x)^n + (1 + x)^n :=
sorry

end exponential_inequality_l209_209769


namespace find_x_l209_209263

theorem find_x (x : ℝ) : (3^(x - 4) = 9^3) → x = 10 :=
by
  intro h
  sorry

end find_x_l209_209263


namespace count_triangles_with_positive_area_l209_209239

theorem count_triangles_with_positive_area : 
  let points : list (ℕ × ℕ) := [(x, y) | x ← [1, 2, 3, 4, 5], y ← [1, 2, 3, 4, 5]]
  let collinear (a b c : ℕ × ℕ) : Prop := (b.1 - a.1) * (c.2 - a.2) = (b.2 - a.2) * (c.1 - a.1)
  let non_degenerate (a b c : ℕ × ℕ) : Prop := ¬collinear a b c
  list.filter (λ t, non_degenerate t[0] t[1] t[2]) (points.combinations 3) = 2156 := by
  sorry

end count_triangles_with_positive_area_l209_209239


namespace sum_of_sol_in_degrees_l209_209596

theorem sum_of_sol_in_degrees :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 360 ∧ (Real.cos (12 * x * Real.pi / 180) = 5 * Real.sin (3 * x * Real.pi / 180) + 9 * (Real.tan (x * Real.pi / 180))^2 + (Real.cot (x * Real.pi / 180))^2) → x = 210 ∨ x = 330) →
  ∑ x in {210, 330}, x = 540 :=
by
  sorry

end sum_of_sol_in_degrees_l209_209596


namespace triangle_circumradius_l209_209859

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 8) (h2 : b = 6) (h3 : c = 10) : 
  ∃ r : ℝ, r = 5 :=
by
  sorry

end triangle_circumradius_l209_209859


namespace train_speed_approx_l209_209013

noncomputable def man_speed_kmh : ℝ := 3
noncomputable def man_speed_ms : ℝ := (man_speed_kmh * 1000) / 3600
noncomputable def train_length : ℝ := 900
noncomputable def time_to_cross : ℝ := 53.99568034557235
noncomputable def train_speed_ms := (train_length / time_to_cross) + man_speed_ms
noncomputable def train_speed_kmh := (train_speed_ms * 3600) / 1000

theorem train_speed_approx :
  abs (train_speed_kmh - 63.009972) < 1e-5 := sorry

end train_speed_approx_l209_209013


namespace number_of_odd_factors_of_360_l209_209194

def is_odd_factor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ ∀ k, 2 ∣ k → ¬ (k ∣ d)

theorem number_of_odd_factors_of_360 : 
  {d : ℕ // is_odd_factor 360 d}.card = 6 := 
sorry

end number_of_odd_factors_of_360_l209_209194


namespace number_of_odd_factors_of_360_l209_209197

def is_odd_factor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ ∀ k, 2 ∣ k → ¬ (k ∣ d)

theorem number_of_odd_factors_of_360 : 
  {d : ℕ // is_odd_factor 360 d}.card = 6 := 
sorry

end number_of_odd_factors_of_360_l209_209197


namespace robert_arrival_time_l209_209778

def arrival_time (T : ℕ) : Prop :=
  ∃ D : ℕ, D = 10 * (12 - T) ∧ D = 15 * (13 - T)

theorem robert_arrival_time : arrival_time 15 :=
by
  sorry

end robert_arrival_time_l209_209778


namespace find_tangent_line_to_curve_l209_209101

noncomputable def tangent_line_at_point := 
  "the equation of the tangent line to the curve \( y = \frac{1}{2} \ln x \) at the point (1, 0) is \( y = \frac{1}{2} x - \frac{1}{2} \)"

noncomputable def curve (x : ℝ) : ℝ := (1 / 2) * log x

noncomputable def point := (1, 0)

theorem find_tangent_line_to_curve : 
  (y : ℝ) (x : ℝ) (h : curve 1 = 0) : y = (1 / 2) * x - (1 / 2) :=
  sorry

end find_tangent_line_to_curve_l209_209101


namespace hyperbola_min_value_l209_209622

def hyperbola_condition : Prop :=
  ∀ (m : ℝ), ∀ (x y : ℝ), (4 * x + 3 * y + m = 0 → (x^2 / 9 - y^2 / 16 = 1) → false)

noncomputable def minimum_value : ℝ :=
  2 * Real.sqrt 37 - 6

theorem hyperbola_min_value :
  hyperbola_condition → minimum_value =  2 * Real.sqrt 37 - 6 :=
by
  intro h
  sorry

end hyperbola_min_value_l209_209622


namespace problem1_l209_209975

noncomputable def f (x : ℝ) : ℝ := x^3 + (-3)*x^2 + (-3)*x + 2

theorem problem1 :
  (f(-1) = 1) ∧ (∀ x, f(x) = x^3 - 3*x^2 - 3*x + 2) :=
by
  sorry

# Output:
# Both parts of the theorem are structured as:
# 1. \( f(-1) = 1 \)
# 2. \( ∀ x, f(x) = x³ - 3x² -3x + 2 \)

end problem1_l209_209975


namespace geom_mean_commutative_only_l209_209356

-- Definitions and Conditions
def geom_mean (x y : ℝ) : ℝ := Real.sqrt (x * y)

def is_commutative (f : ℝ → ℝ → ℝ) : Prop :=
∀ x y : ℝ, f x y = f y x

-- Problem Statement
theorem geom_mean_commutative_only:
  (∀ f : ℝ → ℝ → ℝ, (f = geom_mean → is_commutative f)) ∧
  ¬ (∀ f : ℝ → ℝ → ℝ, (f = geom_mean → ∀ x y z : ℝ, f (f x y) z = f x (f y z))) ∧
  ¬ (∀ f : ℝ → ℝ → ℝ, (f = geom_mean → ∀ x y z : ℝ, f x (y * z) = (f x y) * (f x z))) ∧
  ¬ (∀ f : ℝ → ℝ → ℝ, (f = geom_mean → ∀ x y z : ℝ, x * (f y z) = f (x * y) (x * z))) ∧
  ¬ (∀ f : ℝ → ℝ → ℝ, (f = geom_mean → ∃ i : ℝ, ∀ x : ℝ, f x i = x)) :=
by
  sorry

end geom_mean_commutative_only_l209_209356


namespace helga_tried_on_66_pairs_of_shoes_l209_209663

variables 
  (n1 n2 n3 n4 n5 n6 : ℕ)
  (h1 : n1 = 7)
  (h2 : n2 = n1 + 2)
  (h3 : n3 = 0)
  (h4 : n4 = 2 * (n1 + n2 + n3))
  (h5 : n5 = n2 - 3)
  (h6 : n6 = n1 + 5)
  (total : ℕ := n1 + n2 + n3 + n4 + n5 + n6)

theorem helga_tried_on_66_pairs_of_shoes : total = 66 :=
by sorry

end helga_tried_on_66_pairs_of_shoes_l209_209663


namespace planks_needed_l209_209116

theorem planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) (h1 : total_nails = 4) (h2 : nails_per_plank = 2) : total_nails / nails_per_plank = 2 :=
by
  -- Prove that given the conditions, the required result is obtained
  sorry

end planks_needed_l209_209116


namespace problem_conditions_imply_options_l209_209117

theorem problem_conditions_imply_options (a b : ℝ) 
  (h1 : a + 1 > b) 
  (h2 : b > 2 / a) 
  (h3 : 2 / a > 0) : 
  (a = 2 ∧ a + 1 > 2 / a ∧ b > 2 / 2) ∨
  (a = 1 → a + 1 ≤ 2 / a) ∨
  (b = 1 → ∃ a, a > 1 ∧ a + 1 > 1 ∧ 1 > 2 / a) ∨
  (a * b = 1 → ab ≤ 2) := 
sorry

end problem_conditions_imply_options_l209_209117


namespace count_paths_from_A_to_F_l209_209305

-- Define the grid dimensions
def grid_width : ℕ := 12
def grid_height : ℕ := 2

-- Define the starting and ending points
def A : (ℕ × ℕ) := (0, 0)  -- Starting point at the bottom left corner
def F : (ℕ × ℕ) := (12, 1)  -- Ending point at the top right corner

-- Define the conditions for valid paths
def valid_path (path : List (ℕ × ℕ)) : Prop :=
  (path.head? = some A) ∧  -- Path must start at A
  (path.getLast (by simp) = F) ∧  -- Path must end at F
  (∀ i, i < path.length - 1 → -- Arrows must be perpendicular
    let p1 := path.get? i
    let p2 := path.get? (i + 1)
    match p1, p2 with
    | some (x1, y1), some (x2, y2) → (x1 = x2) ∨ (y1 = y2)
    | _, _ → False
  ) ∧ (∀ i, i < path.length - 2 → -- No two arrows can intersect at more than one point
    let p1 := path.get? i
    let p2 := path.get? (i + 2)
    match p1, p2 with
    | some (x1, y1), some (x2, y2) → ¬ ((x1 = x2) ∧ (y1 = y2))
    | _, _ → True
  ) ∧ (∀ i j, i < path.length → j < path.length → i ≠ j → -- All arrows have different lengths
    let p1 := path.get? i
    let p2 := path.get? j
    match p1, p2 with
    | some (x1, y1), some (x2, y2) → (x1^2 + y1^2 ≠ x2^2 + y2^2)
    | _, _ → True)

-- Define the theorem
theorem count_paths_from_A_to_F : ∃ n, n = 55 ∧ ∀ path, valid_path path → n = 55 :=
by sorry

end count_paths_from_A_to_F_l209_209305


namespace complex_conjugate_root_real_roots_parity_l209_209847

open Polynomial

-- Statement for the first part
theorem complex_conjugate_root (P : Polynomial ℂ) (a : ℂ) (hP : ∀ i, P.coeff i ∈ ℝ) (ha : P.eval a = 0) : P.eval (conj a) = 0 := 
sorry

-- Statement for the second part
theorem real_roots_parity (P : Polynomial ℝ) (hP : ∀ i, P.coeff i ∈ ℝ) : 
  (P.natDegree % 2 = (P.roots.filter (λ x, x.im = 0)).length % 2) := 
sorry

end complex_conjugate_root_real_roots_parity_l209_209847


namespace players_odd_sum_probability_l209_209568

theorem players_odd_sum_probability :
  let tiles := (1:ℕ) :: (2:ℕ) :: (3:ℕ) :: (4:ℕ) :: (5:ℕ) :: (6:ℕ) :: (7:ℕ) :: (8:ℕ) :: (9:ℕ) :: (10:ℕ) :: (11:ℕ) :: []
  let m := 1
  let n := 26
  m + n = 27 :=
by
  sorry

end players_odd_sum_probability_l209_209568


namespace min_value_of_expression_l209_209965

noncomputable def lineBisectsCircleCircumference (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 4*x + 2*y + 1 = 0 ∧ ax*x + by*y + 1 = 0

theorem min_value_of_expression (a b : ℝ) (h : lineBisectsCircleCircumference a b) :
  (a - 2)^2 + (b - 2)^2 = 5 := 
sorry

end min_value_of_expression_l209_209965


namespace shaded_area_l209_209882

noncomputable def area_of_shaded_region (AB : ℝ) (pi_approx : ℝ) : ℝ :=
  let R := AB / 2
  let r := R / 2
  let A_large := (1/2) * pi_approx * R^2
  let A_small := (1/2) * pi_approx * r^2
  2 * A_large - 4 * A_small

theorem shaded_area (h : area_of_shaded_region 40 3.14 = 628) : true :=
  sorry

end shaded_area_l209_209882


namespace cuboid_edge_lengths_l209_209802

theorem cuboid_edge_lengths (
  a b c : ℕ
) (h_volume : a * b * c + a * b + b * c + c * a + a + b + c = 2000) :
  (a = 28 ∧ b = 22 ∧ c = 2) ∨ 
  (a = 28 ∧ b = 2 ∧ c = 22) ∨
  (a = 22 ∧ b = 28 ∧ c = 2) ∨
  (a = 22 ∧ b = 2 ∧ c = 28) ∨
  (a = 2 ∧ b = 28 ∧ c = 22) ∨
  (a = 2 ∧ b = 22 ∧ c = 28) :=
sorry

end cuboid_edge_lengths_l209_209802


namespace convert_3206₈_to_base10_l209_209077

noncomputable def base8ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (λ (d : Nat) (acc : Nat × Nat), (d * acc.2 + acc.1, acc.2 * 8)) (0, 1)

theorem convert_3206₈_to_base10 : base8ToBase10 [3, 2, 0, 6] = 1670 := by
  sorry

end convert_3206₈_to_base10_l209_209077


namespace odd_factors_360_l209_209217

theorem odd_factors_360 : 
  let fac := (2^3) * (3^2) * (5^1) in
  fac = 360 → 
  num_factors (3^2 * 5^1) = 6 :=
sorry

end odd_factors_360_l209_209217


namespace cupcakes_difference_l209_209059

theorem cupcakes_difference (h : ℕ) (betty_rate : ℕ) (dora_rate : ℕ) (betty_break : ℕ) 
  (cupcakes_difference : ℕ) 
  (H₁ : betty_rate = 10) 
  (H₂ : dora_rate = 8) 
  (H₃ : betty_break = 2) 
  (H₄ : cupcakes_difference = 10) : 
  8 * h - 10 * (h - 2) = 10 → h = 5 :=
by
  intro H
  sorry

end cupcakes_difference_l209_209059


namespace solution_set_eq_l209_209901

noncomputable def f : ℝ → ℝ := sorry  -- Assumes existence of such a function

theorem solution_set_eq (f' : ℝ → ℝ) 
  (h1 : ∀ x, f' x > 1 - f x) 
  (h2 : f 0 = 6) :
  {x : ℝ | e^x * f x > e^x + 5} = set.Ioi 0 :=
by sorry

end solution_set_eq_l209_209901


namespace smallest_possible_difference_l209_209940

theorem smallest_possible_difference (p q r s : ℕ) (h0 : 0 < p) (h1 : 0 < q) (h2 : 0 < r) (h3 : 0 < s)
  (hp : p * q * r * s = nat.factorial 9) (hq : p < q) (hr : q < r) (hs : r < s) :
  s - p = 12 :=
sorry

end smallest_possible_difference_l209_209940


namespace wall_area_l209_209030

-- Define the conditions
variables (R J D : ℕ) (L W : ℝ)
variable (area_regular_tiles : ℝ)
variables (ratio_regular : ℕ) (ratio_jumbo : ℕ) (ratio_diamond : ℕ)
variables (length_ratio_jumbo : ℝ) (width_ratio_jumbo : ℝ)
variables (length_ratio_diamond : ℝ) (width_ratio_diamond : ℝ)
variable (total_area : ℝ)

-- Assign values to the conditions
axiom ratio : ratio_regular = 4 ∧ ratio_jumbo = 2 ∧ ratio_diamond = 1
axiom size_regular : area_regular_tiles = 80
axiom jumbo_tile_ratio : length_ratio_jumbo = 3 ∧ width_ratio_jumbo = 3
axiom diamond_tile_ratio : length_ratio_diamond = 2 ∧ width_ratio_diamond = 0.5

-- Define the statement
theorem wall_area (ratio : ratio_regular = 4 ∧ ratio_jumbo = 2 ∧ ratio_diamond = 1)
    (size_regular : area_regular_tiles = 80)
    (jumbo_tile_ratio : length_ratio_jumbo = 3 ∧ width_ratio_jumbo = 3)
    (diamond_tile_ratio : length_ratio_diamond = 2 ∧ width_ratio_diamond = 0.5):
    total_area = 140 := 
sorry

end wall_area_l209_209030


namespace complex_quadrant_proof_l209_209402

theorem complex_quadrant_proof :
  let z : ℂ := (1 / 2 + (real.sqrt 3) / 2 * complex.i) ^ 2
  ∃ (x y : ℝ), z = x + y * complex.i ∧ x < 0 ∧ y > 0 :=
by
  sorry

end complex_quadrant_proof_l209_209402


namespace simplify_f_max_min_f_interval_l209_209154

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x - sqrt 3 * cos (2 * x) + 1

theorem simplify_f :
  ∃ g : ℝ → ℝ, (∀ x, f(x) = g(x)) ∧ (∃ T > 0, ∀ x, g (x + T) = g x) ∧ T = π := by
sorry

theorem max_min_f_interval :
  (∀ x ∈ set.Icc (π/4) (π/2), 2 ≤ f x ∧ f x ≤ 3) ∧
  (∃ x ∈ set.Icc (π/4) (π/2), f x = 3) ∧
  (∃ x ∈ set.Icc (π/4) (π/2), f x = 2) := by
sorry

end simplify_f_max_min_f_interval_l209_209154


namespace valuable_files_proof_l209_209888

def valuable_files_after_all_rounds 
  (round1_downloads : ℕ) 
  (round1_deleted_percent : ℕ) 
  (round2_downloads : ℕ) 
  (round2_deleted_fraction : ℚ) 
  (round3_downloads : ℕ) 
  (round3_deleted_percent : ℕ) 
  : ℕ :=
  let round1_valuable := round1_downloads - (round1_deleted_percent * round1_downloads / 100)
  let round2_valuable := round2_downloads - (round2_deleted_fraction * round2_downloads).to_nat
  let interim_valuable := round1_valuable + round2_valuable
  let round3_valuable := round3_downloads - (round3_deleted_percent * round3_downloads / 100)
  interim_valuable + round3_valuable

theorem valuable_files_proof :
  ∀ (round1_downloads round1_deleted_percent round2_downloads round2_deleted_fraction round3_downloads round3_deleted_percent : ℕ),
  round1_downloads = 1200 →
  round1_deleted_percent = 80 →
  round2_downloads = 600 →
  round2_deleted_fraction = (4 / 5 : ℚ) →
  round3_downloads = 700 →
  round3_deleted_percent = 65 →
  valuable_files_after_all_rounds round1_downloads round1_deleted_percent round2_downloads round2_deleted_fraction round3_downloads round3_deleted_percent = 605 :=
by
  intros
  sorry

end valuable_files_proof_l209_209888


namespace sample_size_is_50_l209_209293

theorem sample_size_is_50 (n : ℕ) :
  (n > 0) → 
  (10 / n = 2 / (2 + 3 + 5)) → 
  n = 50 := 
by
  sorry

end sample_size_is_50_l209_209293


namespace cos_F_eq_15_over_17_l209_209699

noncomputable def DE : ℝ := 8
noncomputable def EF : ℝ := 17
noncomputable def angle_D : ℝ := 90

theorem cos_F_eq_15_over_17 :
  let DF := Real.sqrt (EF^2 - DE^2)
  ∧ DF = 15 
  ∧ (cos (Real.arccos (DE / EF))) = (15 / 17) :=
by
  sorry

end cos_F_eq_15_over_17_l209_209699


namespace checkerboard_disc_coverage_l209_209496

/-- A circular disc with a diameter of 5 units is placed on a 10 x 10 checkerboard with each square having a side length of 1 unit such that the centers of both the disc and the checkerboard coincide.
    Prove that the number of checkerboard squares that are completely covered by the disc is 36. -/
theorem checkerboard_disc_coverage :
  let diameter : ℝ := 5
  let radius : ℝ := diameter / 2
  let side_length : ℝ := 1
  let board_size : ℕ := 10
  let disc_center : ℝ × ℝ := (board_size / 2, board_size / 2)
  ∃ (count : ℕ), count = 36 := 
  sorry

end checkerboard_disc_coverage_l209_209496


namespace ratio_of_a_over_3_to_b_over_2_l209_209006

theorem ratio_of_a_over_3_to_b_over_2 (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : (a / 3) / (b / 2) = 1 :=
by
  sorry

end ratio_of_a_over_3_to_b_over_2_l209_209006


namespace find_x_l209_209252

theorem find_x (x : ℤ) (h : 3^(x - 4) = 9^3) : x = 10 := 
sorry

end find_x_l209_209252


namespace Zainab_hourly_wage_l209_209474

theorem Zainab_hourly_wage :
  ∀ (days_per_week hours_per_day weeks total_earnings : ℕ),
    days_per_week = 3 →
    hours_per_day = 4 →
    weeks = 4 →
    total_earnings = 96 →
    (total_earnings / (days_per_week * hours_per_day * weeks) : ℚ) = 2 :=
by
  intros days_per_week hours_per_day weeks total_earnings h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end Zainab_hourly_wage_l209_209474


namespace forty_percent_of_number_l209_209760

theorem forty_percent_of_number (N : ℝ) 
  (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 17) : 
  0.40 * N = 204 :=
sorry

end forty_percent_of_number_l209_209760


namespace q1_q2_l209_209614

noncomputable def parabola_equation (p : ℝ) : Prop :=
  ∀ x y : ℝ, (y^2 = 2 * p * x) → 
  (∃ p > 0, 
      let x1 := (- (5 * p) + real.sqrt ((5 * p)^2 - 4 * 4 * p * p)) / 8 in
      let x2 := (- (5 * p) - real.sqrt ((5 * p)^2 - 4 * 4 * p * p)) / 8 in
      let y1 := -4 * real.sqrt 2 in
      let y2 := 8 * real.sqrt 2 in
      let d := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) in
      d = 18 ∧ (y^2 = 16 * x)
  )

noncomputable def lambda_value (p : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, 
    (y1 = -4 * real.sqrt 2) →
    (y2 = 8 * real.sqrt 2) →
    let x3 := 2 + (8 * λ) in
    let y3 := -4 * real.sqrt 2 + 8 * λ * real.sqrt 2 in
    ((y3)^2 = 16 * (x3)) → (λ = 0 ∨ λ = 2)

theorem q1 (p : ℝ) (h : ∃ p > 0, parabola_equation p) : ∀ x y, y^2 = 16 * x :=
sorry

theorem q2 (p : ℝ) (h : ∃ p > 0, lambda_value p) : ∀ λ, λ = 0 ∨ λ = 2 :=
sorry

end q1_q2_l209_209614


namespace sandrine_washed_160_dishes_l209_209416

-- Define the number of pears picked by Charles
def charlesPears : ℕ := 50

-- Define the number of bananas cooked by Charles as 3 times the number of pears he picked
def charlesBananas : ℕ := 3 * charlesPears

-- Define the number of dishes washed by Sandrine as 10 more than the number of bananas Charles cooked
def sandrineDishes : ℕ := charlesBananas + 10

-- Prove that Sandrine washed 160 dishes
theorem sandrine_washed_160_dishes : sandrineDishes = 160 := by
  -- The proof is omitted
  sorry

end sandrine_washed_160_dishes_l209_209416


namespace find_x_l209_209251

theorem find_x (x : ℤ) (h : 3^(x - 4) = 9^3) : x = 10 := 
sorry

end find_x_l209_209251


namespace sequence_explicit_formula_l209_209949

theorem sequence_explicit_formula :
  ∃ (a : ℕ → ℕ), 
    prime (a 2) ∧ prime (a 3) ∧
    (∀ m n, 0 < m → m < n → a (m + n) = a m + a n + 31 ∧ 
      ((3 * n - 1) * (a m) < (3 * m - 1) * (a n) ∧ (a n) * (5 * m - 2) < (5 * n - 2) * (a m))) ∧
    (∀ n, a n = 90 * n - 31) :=
sorry

end sequence_explicit_formula_l209_209949


namespace solve_tetrahedron_side_length_l209_209480

noncomputable def side_length_of_circumscribing_tetrahedron (r : ℝ) (tangent_spheres : ℕ) (radius_spheres_equal : ℝ) : ℝ := 
  if h : r = 1 ∧ tangent_spheres = 4 then
    2 + 2 * Real.sqrt 6
  else
    0

theorem solve_tetrahedron_side_length :
  side_length_of_circumscribing_tetrahedron 1 4 1 = 2 + 2 * Real.sqrt 6 :=
by
  sorry

end solve_tetrahedron_side_length_l209_209480


namespace max_value_ratio_l209_209140

theorem max_value_ratio (a b c: ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) (h_eq: a * (a + b + c) = b * c) :
  (a / (b + c) ≤ (Real.sqrt 2 - 1) / 2) :=
sorry -- proof omitted

end max_value_ratio_l209_209140


namespace die_face_never_touches_board_l209_209360

theorem die_face_never_touches_board : 
  ∃ (cube : Type) (roll : cube → cube) (occupied : Fin 8 × Fin 8 → cube → Prop),
    (∀ p : Fin 8 × Fin 8, ∃ c : cube, occupied p c) ∧ 
    (∃ f : cube, ¬ (∃ p : Fin 8 × Fin 8, occupied p f)) :=
by sorry

end die_face_never_touches_board_l209_209360


namespace h_inv_f_neg3_does_not_exist_real_l209_209538

noncomputable def h : ℝ → ℝ := sorry
noncomputable def f : ℝ → ℝ := sorry

theorem h_inv_f_neg3_does_not_exist_real (h_inv : ℝ → ℝ)
  (h_cond : ∀ (x : ℝ), f (h_inv (h x)) = 7 * x ^ 2 + 4) :
  ¬ ∃ x : ℝ, h_inv (f (-3)) = x :=
by 
  sorry

end h_inv_f_neg3_does_not_exist_real_l209_209538


namespace distance_from_point_to_plane_is_7_l209_209099

open Real

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def plane_through_points (p1 p2 p3 : Point3D) : ℝ × ℝ × ℝ × ℝ :=
  let a := (p2.y - p1.y) * (p3.z - p1.z) - (p3.y - p1.y) * (p2.z - p1.z)
  let b := (p2.z - p1.z) * (p3.x - p1.x) - (p3.z - p1.z) * (p2.x - p1.x)
  let c := (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)
  let d := -(a * p1.x + b * p1.y + c * p1.z)
  (a, b, c, d)

def distance_point_to_plane (p : Point3D) (A B C D : ℝ) : ℝ :=
  abs (A * p.x + B * p.y + C * p.z + D) / sqrt (A ^ 2 + B ^ 2 + C ^ 2)

theorem distance_from_point_to_plane_is_7 :
  let M1 := Point3D.mk 1 5 (-7)
  let M2 := Point3D.mk (-3) 6 3
  let M3 := Point3D.mk (-2) 7 3
  let M0 := Point3D.mk 1 (-1) 2
  let (A, B, C, D) := plane_through_points M1 M2 M3
  distance_point_to_plane M0 A B C D = 7 := 
by
  -- Definitions for points and plane would go here.
  sorry

end distance_from_point_to_plane_is_7_l209_209099


namespace multiplications_in_three_hours_l209_209024

theorem multiplications_in_three_hours :
  let rate := 15000  -- multiplications per second
  let seconds_in_three_hours := 3 * 3600  -- seconds in three hours
  let total_multiplications := rate * seconds_in_three_hours
  total_multiplications = 162000000 :=
by
  let rate := 15000
  let seconds_in_three_hours := 3 * 3600
  let total_multiplications := rate * seconds_in_three_hours
  have h : total_multiplications = 162000000 := sorry
  exact h

end multiplications_in_three_hours_l209_209024


namespace relationship_among_abc_l209_209335

variable (f : ℝ → ℝ)

-- Define the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def f_increasing (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → y ≤ π / 2 → f x ≤ f y

-- Define the given function
noncomputable def fx : ℝ → ℝ := λ x, x^3 * sin x

-- Define the provided points
def a := f (sin (π / 3))
def b := f (sin 2)
def c := f (sin 3)

-- State the theorem
theorem relationship_among_abc 
  (hf_even : is_even f)
  (hf_def : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x = fx x)
  (h_inc : f_increasing f) :
  c < a ∧ a < b :=
sorry

end relationship_among_abc_l209_209335


namespace base_prime_representation_l209_209075

theorem base_prime_representation : 
  let n := 294 in
  let primes := [2, 3, 5, 7] in
  let exponents := [1, 1, 0, 2] in
  ∃ (repr : List ℕ), repr = exponents ∧ n = primes.prod.map_with_index (λ i x, x ^ repr.nth i.get_or_else 0).prod
  :=
by
  let n := 294
  let primes := [2, 3, 5, 7]
  let exponents := [1, 1, 0, 2]
  exists exponents
  split
  . refl
  . sorry

end base_prime_representation_l209_209075


namespace maximum_unique_numbers_in_circle_l209_209879

theorem maximum_unique_numbers_in_circle :
  ∀ (n : ℕ) (numbers : ℕ → ℤ), n = 2023 →
  (∀ i, numbers i = numbers ((i + 1) % n) * numbers ((i + n - 1) % n)) →
  ∀ i j, numbers i = numbers j :=
by
  sorry

end maximum_unique_numbers_in_circle_l209_209879


namespace compare_f_g_l209_209611

   variables (a b c α : ℝ)
   -- Assume a, b, c are positive real numbers
   assume (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0)

   def f_alpha (a b c α : ℝ) := a * b * c * (a^α + b^α + c^α)
   def g_alpha (a b c α : ℝ) := a^((a : ℝ) + 2) * (b + c - a) + b^((a : ℝ) + 2) * (a + c - b) + c^((a : ℝ) + 2) * (a + b - c)

   theorem compare_f_g : f_alpha a b c α > g_alpha a b c α :=
   sorry
   
end compare_f_g_l209_209611


namespace gcd_sequence_l209_209921

theorem gcd_sequence (n : ℕ) : ∃ d, d = Nat.gcd (2002 + 2) ((2002^2) + 2) ∧ d = 6 :=
by
  let a_n := λ (n: ℕ), 2002^n + 2
  let d := a_n 0
  let g := Nat.gcd a_n d
  have h1: g = 6 := sorry
  exact ⟨g, (by sorry : g = Nat.gcd (2002 + 2) ((2002^2) + 2)), h1⟩

end gcd_sequence_l209_209921


namespace dice_probability_l209_209487

theorem dice_probability :
  let one_digit_prob := 9 / 20
  let two_digit_prob := 11 / 20
  let number_of_dice := 5
  ∃ p : ℚ,
    (number_of_dice.choose 2) * (one_digit_prob ^ 2) * (two_digit_prob ^ 3) = p ∧
    p = 107811 / 320000 :=
by
  sorry

end dice_probability_l209_209487


namespace vectors_perpendicular_l209_209536

noncomputable def p : ℝ × ℝ × ℝ := (2, 0, -3)
noncomputable def q : ℝ × ℝ × ℝ := (3, 4, 2)

theorem vectors_perpendicular : euclidean_inner p q = 0 :=
by
  sorry

end vectors_perpendicular_l209_209536


namespace find_9a_value_l209_209418

theorem find_9a_value (a : ℚ) 
  (h : (4 - a) / (5 - a) = (4 / 5) ^ 2) : 9 * a = 20 :=
by
  sorry

end find_9a_value_l209_209418


namespace reflect_and_translate_F_l209_209824

theorem reflect_and_translate_F :
  ∃ F' : ℝ × ℝ, 
    F' = (-5, 4) ∧ 
    ∀ x y : ℝ, F = (5, 1) 
    → reflect_y (x, y) = (-x, y)
    → translate (x, y) 3 = (x, y+3) :=
by
  sorry

end reflect_and_translate_F_l209_209824


namespace difference_of_numbers_l209_209827

theorem difference_of_numbers {x y : ℕ} (h1 : x ≠ y) (h2 : x ∈ Finset.range 40) (h3 : y ∈ Finset.range 40)
  (h4 : 780 - x - y = 2 * x * y) : |y - x| = 2 := 
sorry

end difference_of_numbers_l209_209827


namespace equal_chords_isosceles_trapezoid_l209_209952

theorem equal_chords_isosceles_trapezoid
  (A B C D : Point) (circle1 circle2 : Circle)
  (isosceles_trapezoid : IsIsoscelesTrapezoid A B C D)
  (touch_circle1 : TouchesVertices circle1 A B (Base A B))
  (touch_circle2 : TouchesVertices circle2 C D (Base C D))
  (diagonal_bd_intersects_circles: IntersectsDiagonals (Diagonal B D) circle1 circle2) :
  ChordLength (ChordFromIntersection (Diagonal B D) circle1) =
  ChordLength (ChordFromIntersection (Diagonal B D) circle2) :=
sorry

end equal_chords_isosceles_trapezoid_l209_209952


namespace candy_problem_l209_209542

-- Define conditions and the statement
theorem candy_problem (K : ℕ) (h1 : 49 = K + 3 * K + 8 + 6 + 10 + 5) : K = 5 :=
sorry

end candy_problem_l209_209542


namespace construct_triangle_with_given_conditions_l209_209829

theorem construct_triangle_with_given_conditions
  {A B C : Type*} 
  (angle_smallest : ∠A < ∠B ∧ ∠A < ∠C)
  (d e : ℝ) 
  (h1 : d = (dist A B - dist B C))
  (h2 : e = (dist A C - dist C B)) :
  ∃ (A B C : Type*), 
    (angle_smallest ∈ {∠A, ∠B, ∠C}) ∧
    (d = (dist A B - dist B C)) ∧
    (e = (dist A C - dist C B)) := 
sorry

end construct_triangle_with_given_conditions_l209_209829


namespace complete_set_of_events_l209_209826

-- Define the range of numbers on a die
def die_range := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define what an outcome is
def outcome := { p : ℕ × ℕ | p.1 ∈ die_range ∧ p.2 ∈ die_range }

-- The theorem stating the complete set of outcomes
theorem complete_set_of_events : outcome = { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 6 ∧ 1 ≤ p.2 ∧ p.2 ≤ 6 } :=
by sorry

end complete_set_of_events_l209_209826


namespace expected_value_of_draws_before_stopping_l209_209489

noncomputable def totalBalls := 10
noncomputable def redBalls := 2
noncomputable def whiteBalls := 8

noncomputable def prob_one_draw_white : ℚ := whiteBalls / totalBalls
noncomputable def prob_two_draws_white : ℚ := (redBalls / totalBalls) * (whiteBalls / (totalBalls - 1))
noncomputable def prob_three_draws_white : ℚ := (redBalls / (totalBalls - redBalls + 1)) * ((redBalls - 1) / (totalBalls - 1)) * (whiteBalls / (totalBalls - 2))

noncomputable def expected_draws_before_white : ℚ :=
  1 * prob_one_draw_white + 2 * prob_two_draws_white + 3 * prob_three_draws_white

theorem expected_value_of_draws_before_stopping : expected_draws_before_white = 11 / 9 := by
  sorry

end expected_value_of_draws_before_stopping_l209_209489


namespace value_of_expression_l209_209460

theorem value_of_expression (x : ℝ) (h : x = real.sqrt 2 - 1) : x^2 + 2 * x = 1 :=
by
  sorry

end value_of_expression_l209_209460


namespace find_a_chi_square_test_l209_209358

noncomputable def a : ℕ := 10

theorem find_a (total_male total_female total_students_half_marathon total_students : ℕ) (ratio : ℕ × ℕ) (participating_male_half_marathon participating_male_mini_health_run participating_female_mini_health_run : ℕ) :
    total_male = 30 → total_female = 20 → total_students_half_marathon = 30 → total_students = 50 →
    ratio = (3, 2) → participating_male_half_marathon = 20 → participating_male_mini_health_run = 10 → participating_female_mini_health_run = 10 →
    a = 10 ∧ (30 / 50 : ℚ) = 3 / 5 :=
by
  intros
  sorry

theorem chi_square_test (total_male_total total_female_total total_students_total_half_marathon total_students_total : ℕ) (a b c d n : ℕ)
     (chi_square_critical_value : ℚ) :
    total_male_total = 30 → total_female_total = 20 → total_students_total_half_marathon = 30 → total_students_total = 50 →
    a = 20 → b = 10 → c = 10 → d = 10 → n = 50 →
    chi_square_critical_value = 2.706 →
    K_squared = (n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))) →
    K_squared ≈ 1.389 ∧ 1.389 < chi_square_critical_value :=
by
  intros
  sorry

end find_a_chi_square_test_l209_209358


namespace odd_factors_360_l209_209183

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l209_209183


namespace stickers_distribution_l209_209665

-- Definition of the problem
def num_ways_to_distribute_stickers : ℕ := 
  nat.choose (8 + 4 - 1) 3

theorem stickers_distribution (n_stickers : ℕ) (n_sheets : ℕ) (h_stickers : n_stickers = 8) (h_sheets : n_sheets = 4) : 
  num_ways_to_distribute_stickers = 165 :=
by
  -- Replace it with the actual proof
  sorry

end stickers_distribution_l209_209665


namespace perpendicular_P_M_passes_through_M_l209_209739

-- Definitions and assumptions
variables (A B C D P Q M : Point)
variables (h_quad_inscribable : inscribable_quadrilateral A B C D)
variables (h_AD_BC_intersect_P : ∃ P', line_intersection AD BC P)
variables (h_AB_CD_intersect_Q : ∃ Q', line_intersection AB CD Q)
variables (h_angle_APQ_90 : angle A P Q = 90)
variables (h_m_midpoint_BD : midpoint M B D)

-- Statement that we need to prove
theorem perpendicular_P_M_passes_through_M :
  perpendicular_from P to (line A B) passes_through M :=
by sorry

end perpendicular_P_M_passes_through_M_l209_209739


namespace simplification_part1_simplification_part2_l209_209369

theorem simplification_part1 : 
    (3 / Real.sqrt 6 = Real.sqrt 6 / 2) ∧ 
    (Real.sqrt (2 / 7) = Real.sqrt 14 / 7) ∧ 
    (2 / (Real.sqrt 5 - Real.sqrt 3) = Real.sqrt 5 + Real.sqrt 3) := by
  sorry

theorem simplification_part2 : 
    ∑ i in Finset.range 49, (1 / (Real.sqrt (2 * i + 1) + Real.sqrt (2 * i + 3))) = 
    (3 * Real.sqrt 11 - 1) / 2 := by
  sorry

end simplification_part1_simplification_part2_l209_209369


namespace baker_cakes_total_l209_209541

-- Conditions
def initial_cakes : ℕ := 121
def cakes_sold : ℕ := 105
def cakes_bought : ℕ := 170

-- Proof Problem
theorem baker_cakes_total :
  initial_cakes - cakes_sold + cakes_bought = 186 :=
by
  sorry

end baker_cakes_total_l209_209541


namespace correct_statement_l209_209997

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l209_209997


namespace sum_computation_l209_209553

theorem sum_computation :
  (∑ a b c, if 1 ≤ a ∧ a < b ∧ b < c then (1 / (3^a * 4^b * 6^c)) else 0) = 1 / 3975 :=
by
  sorry

end sum_computation_l209_209553


namespace player_B_wins_if_A_first_l209_209803

theorem player_B_wins_if_A_first :
  let numbers := {1..500}
  -- Initial sum of numbers is divisible by 3
  let S := 125250
  -- Player A starts first and removes a number i ∈ numbers
  -- Player B removes 501 - i whenever A removes i
  -- Game ends when two numbers remain
  -- Player B wins if the sum of the two remaining numbers is divisible by 3
  -- Strategy of Player B results in x + y being divisible by 3 
  ∀ (i : ℕ) (h : i ∈ numbers), 
    let B_removal := 501 - i in
    S % 3 = 0 → (125250 - (sum (A_removal : ι) : ι ∈ {1..500}')).sum_pairwise 501 % 3 = 0 :=
begin
  intros,
  sorry
end

end player_B_wins_if_A_first_l209_209803


namespace map_distance_1cm_map_distance_AB_l209_209359

-- Define the map scale
def scale := (1 : ℝ) / 400000

-- Define the actual distance between cities A and B
def d_actual := 80 * 1000 * 100 -- 80 kilometers converted to centimeters

-- Define the actual distance that 1 cm on the map represents
def d_map_1cm := 4 * 1000 * 100 -- 4 kilometers converted to centimeters

-- Define the map distance between cities A and B
def d_map_AB := d_actual / (4 * 1000 * 100)

-- The proof statements
theorem map_distance_1cm : scale * (4 * 1000 * 100) = 1 := sorry
theorem map_distance_AB : d_map_AB = 20 := sorry

end map_distance_1cm_map_distance_AB_l209_209359


namespace number_of_positive_area_triangles_l209_209247

theorem number_of_positive_area_triangles (s : Finset (ℕ × ℕ)) (h : s = {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}) :
  (s.powerset.filter λ t, t.card = 3 ∧ ¬∃ a b c, a = b ∨ b = c ∨ c = a).card = 2160 :=
by {
  sorry
}

end number_of_positive_area_triangles_l209_209247


namespace solve_for_x_l209_209565

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
    5 * y ^ 2 + 2 * y + 3 = 3 * (9 * x ^ 2 + y + 1) ↔ x = 0 ∨ x = 1 / 6 := 
by
  sorry

end solve_for_x_l209_209565


namespace bridge_length_l209_209479

noncomputable def length_of_bridge := 255

def train_length := 120
def train_speed_kmph := 45
def train_speed_mps := 12.5  -- Converted from km/hr to m/s
def crossing_time := 30

theorem bridge_length :
  let total_distance := train_speed_mps * crossing_time in
  let bridge_length := total_distance - train_length in
  bridge_length = length_of_bridge :=
by
  -- The details of the proof are not required, hence we use sorry
  sorry

end bridge_length_l209_209479


namespace odd_factors_360_l209_209181

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l209_209181


namespace sum_first_10_terms_l209_209796

def a_n (n: ℕ) : ℚ := n^2 + n

def summation (f : ℕ → ℚ) (m n : ℕ) : ℚ := 
  (Finset.range (n-m+1)).sum (λ k, f (m + k))

theorem sum_first_10_terms : 
  summation (λ n, 1 / a_n n) 1 10 = 10 / 11 := by
    sorry

end sum_first_10_terms_l209_209796


namespace area_of_efgh_l209_209821

def small_rectangle_shorter_side : ℝ := 7
def small_rectangle_longer_side : ℝ := 3 * small_rectangle_shorter_side
def larger_rectangle_width : ℝ := small_rectangle_longer_side
def larger_rectangle_length : ℝ := small_rectangle_longer_side + small_rectangle_shorter_side

theorem area_of_efgh :
  larger_rectangle_length * larger_rectangle_width = 588 := by
  sorry

end area_of_efgh_l209_209821


namespace remainder_23_to_2047_mod_17_l209_209834

theorem remainder_23_to_2047_mod_17 :
  23^2047 % 17 = 11 := 
by {
  sorry
}

end remainder_23_to_2047_mod_17_l209_209834


namespace f_sum_one_for_inverses_l209_209735

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem f_sum_one_for_inverses (n : ℝ) (hn : n > 0) : f(n) + f(1/n) = 1 :=
by
  -- proof goes here
  sorry

end f_sum_one_for_inverses_l209_209735


namespace number_of_good_points_l209_209325

-- Define a good point
def good_point (A B C P : Point) : Prop :=
  ∃ (rays : Fin 27 → LineSegment), (∀ i, 
    (rays i).endpoint1 = P ∧ (rays i).endpoint2 ∈ triangle Sides A B C) ∧ 
    divides_into_equal_area_triangles P (rays) 27

-- Define the theorem to prove
theorem number_of_good_points (A B C : Point) : 
  ∃! (P : Point), good_point A B C P :=
by sorry

end number_of_good_points_l209_209325


namespace find_z_l209_209945

theorem find_z (z : ℂ) (h : (Complex.I * z = 4 + 3 * Complex.I)) : z = 3 - 4 * Complex.I :=
by
  sorry

end find_z_l209_209945


namespace parabola_rotation_180_equivalent_l209_209373

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 - 2

-- Define the expected rotated parabola equation
def rotated_parabola (x : ℝ) : ℝ := -2 * (x - 3)^2 - 2

-- Prove that the rotated parabola is correctly transformed
theorem parabola_rotation_180_equivalent :
  ∀ x, rotated_parabola x = -2 * (x - 3)^2 - 2 := 
by
  intro x
  unfold rotated_parabola
  sorry

end parabola_rotation_180_equivalent_l209_209373


namespace number_of_multiples_of_31_in_array_l209_209042

/-- A triangular array of numbers has a first row consisting of odd integers 1, 3, 5,
    ..., up to 59 in increasing order. Each row below the first has one fewer entry than
    the row above it, and the bottom row has a single entry. Each entry in any row after the
    top row equals the sum of the two entries diagonally above it in the row immediately above it.
    We need to prove that there are 14 entries in the array which are multiples of 31. -/
theorem number_of_multiples_of_31_in_array :
  let first_row := (List.range 30).map (λ i => 1 + 2 * i) in
  let a : ℕ → ℕ → ℕ := λ n k, 2^(n-1) * (n + 2 * k - 2) in
  (∑ n in Finset.range 28, if n % 2 = 1 then 1 else 0) = 14 :=
by
  sorry

end number_of_multiples_of_31_in_array_l209_209042


namespace exists_zero_exists_term_to_S_div_N_l209_209129

-- Defining the sequence and its properties
variable (N : ℕ) (A_N : Fin N.succ → ℤ)
hypothesis (hN : N ≥ 2)
hypothesis (h_neg_pos : A_N 0 * A_N N < 0)
hypothesis (h_consecutive : ∀ i : Fin N, |A_N i - A_N (i.succ) | ≤ 1)

-- Define the sum of the sequence
def S (A_N : Fin N.succ → ℤ) : ℤ := ∑ i, A_N i

-- 1. Prove existence of a term equal to 0
theorem exists_zero : ∃ k : Fin N.succ, A_N k = 0 :=
by
  sorry

-- 2. Prove if S(A_N) is a multiple of N, there exists a term equal to S(A_N) / N
theorem exists_term_to_S_div_N (h_multiple : S A_N % N.succ = 0) : ∃ r : Fin N.succ, S A_N = N.succ * A_N r :=
by
  sorry

end exists_zero_exists_term_to_S_div_N_l209_209129


namespace circles_tangent_lengths_l209_209788

variables (R R' : ℝ)
variables (O O' A A' B B' : Type)
variables (CD' DC' : O → O' → Type)
variables (AA' : O → O' → Type)
variables (l_inner l_outer : ℝ)

theorem circles_tangent_lengths (h1 : ℝ = l_inner + 2 * R * classic.cot 0 - l_inner)
  (h2: ℝ = l_outer - 2 * R * classic.cot 0 - l_outer):
    (AA' B A' = 1) ∧ (AB * A'B = R * R') :=
    sorry

end circles_tangent_lengths_l209_209788


namespace maximum_value_expression_gt1_l209_209564

noncomputable def max_value_expression : ℝ :=
  let f (x : ℝ) := (x^4 - x^2) / (x^6 + 2 * x^3 - 1)
  sorry

theorem maximum_value_expression_gt1 (x : ℝ) (hx : 1 < x) :
  ∃ y, y = f x ∧ y ≤ 1 / 5 :=
sorry

end maximum_value_expression_gt1_l209_209564


namespace solve_for_x_l209_209674

theorem solve_for_x (x : ℤ) (h : 45 - (28 - (37 - (x - 16))) = 55) : x = 15 :=
begin
  sorry
end

end solve_for_x_l209_209674


namespace n_cubed_plus_5n_divisible_by_6_l209_209768

theorem n_cubed_plus_5n_divisible_by_6 (n : ℕ) : ∃ k : ℤ, n^3 + 5 * n = 6 * k :=
by
  sorry

end n_cubed_plus_5n_divisible_by_6_l209_209768


namespace intersection_set_l209_209985

def M : set ℝ := {x | log (1/2) (x-1) > -1}
def N : set ℝ := {x | 1 < 2^x ∧ 2^x < 4}

theorem intersection_set :
  M ∩ N = {x | 1 < x ∧ x < 2} :=
sorry

end intersection_set_l209_209985


namespace part1_part2_i_part2_ii_l209_209685

def equation1 (x : ℝ) : Prop := 3 * x - 2 = 0
def equation2 (x : ℝ) : Prop := 2 * x - 3 = 0
def equation3 (x : ℝ) : Prop := x - (3 * x + 1) = -7

def inequality1 (x : ℝ) : Prop := -x + 2 > x - 5
def inequality2 (x : ℝ) : Prop := 3 * x - 1 > -x + 2

def sys_ineq (x m : ℝ) : Prop := x + m < 2 * x ∧ x - 2 < m

def equation4 (x : ℝ) : Prop := (2 * x - 1) / 3 = -3

theorem part1 : 
  ∀ (x : ℝ), inequality1 x → inequality2 x → equation2 x → equation3 x :=
by sorry

theorem part2_i :
  ∀ (m : ℝ), (∃ (x : ℝ), equation4 x ∧ sys_ineq x m) → -6 < m ∧ m < -4 :=
by sorry

theorem part2_ii :
  ∀ (m : ℝ), ¬ (sys_ineq 1 m ∧ sys_ineq 2 m) → m ≥ 2 ∨ m ≤ -1 :=
by sorry

end part1_part2_i_part2_ii_l209_209685


namespace Paul_dig_days_alone_l209_209715

/-- Jake's daily work rate -/
def Jake_work_rate : ℚ := 1 / 16

/-- Hari's daily work rate -/
def Hari_work_rate : ℚ := 1 / 48

/-- Combined work rate of Jake, Paul, and Hari, when they work together they can dig the well in 8 days -/
def combined_work_rate (Paul_work_rate : ℚ) : Prop :=
  Jake_work_rate + Paul_work_rate + Hari_work_rate = 1 / 8

/-- Theorem stating that Paul can dig the well alone in 24 days -/
theorem Paul_dig_days_alone : ∃ (P : ℚ), combined_work_rate (1 / P) ∧ P = 24 :=
by
  use 24
  unfold combined_work_rate
  sorry

end Paul_dig_days_alone_l209_209715


namespace smallest_square_with_longest_sequence_of_identical_digits_l209_209456

theorem smallest_square_with_longest_sequence_of_identical_digits :
  ∃ n : ℕ, (n^2 = 1444 ∧ (∀ m : ℕ, m^2.ends_with_repeating_digit_sequence → m^2 ≥ 1444)) :=
sorry

end smallest_square_with_longest_sequence_of_identical_digits_l209_209456


namespace C_increases_as_n_increases_l209_209744

noncomputable def C (e R r n : ℝ) : ℝ := (e * n^2) / (R + n * r)

theorem C_increases_as_n_increases
  (e R r : ℝ) (e_pos : 0 < e) (R_pos : 0 < R) (r_pos : 0 < r) :
  ∀ n : ℝ, 0 < n → ∃ M : ℝ, ∀ N : ℝ, N > n → C e R r N > M :=
by
  sorry

end C_increases_as_n_increases_l209_209744


namespace odd_factors_of_360_l209_209173

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l209_209173


namespace polynomial_nonzero_coeffs_l209_209095

def has_n_nonzero_coeffs (p : Polynomial ℝ) (n : ℕ) : Prop :=
  (p.coefficients.filter (λ x, x ≠ 0)).length = n

theorem polynomial_nonzero_coeffs (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ n : ℕ, n = 0 ∨ n = 50 ∨ n = 100 ∨ n = 101 ∧ has_n_nonzero_coeffs ((Polynomial.C a * Polynomial.x + Polynomial.C b) ^ 100 - (Polynomial.C c * Polynomial.x + Polynomial.C d) ^ 100) n := 
sorry

end polynomial_nonzero_coeffs_l209_209095


namespace julia_cakes_remaining_l209_209698

/-- Formalizing the conditions of the problem --/
def cakes_per_day : ℕ := 5 - 1
def baking_days : ℕ := 6
def eaten_cakes_per_other_day : ℕ := 1
def total_days : ℕ := 6
def total_eaten_days : ℕ := total_days / 2

/-- The theorem to be proven --/
theorem julia_cakes_remaining : 
  let total_baked_cakes := cakes_per_day * baking_days in
  let total_eaten_cakes := eaten_cakes_per_other_day * total_eaten_days in
  total_baked_cakes - total_eaten_cakes = 21 := 
by
  sorry

end julia_cakes_remaining_l209_209698


namespace evaluate_fg_neg3_l209_209089

theorem evaluate_fg_neg3 :
  let f (x : ℝ) := 3 - Real.sqrt x
  let g (x : ℝ) := -x + 3 * x^2
  f (g (-3)) = 3 - Real.sqrt 30 :=
by
  sorry

end evaluate_fg_neg3_l209_209089


namespace count_triangles_with_positive_area_l209_209241

theorem count_triangles_with_positive_area : 
  let points : list (ℕ × ℕ) := [(x, y) | x ← [1, 2, 3, 4, 5], y ← [1, 2, 3, 4, 5]]
  let collinear (a b c : ℕ × ℕ) : Prop := (b.1 - a.1) * (c.2 - a.2) = (b.2 - a.2) * (c.1 - a.1)
  let non_degenerate (a b c : ℕ × ℕ) : Prop := ¬collinear a b c
  list.filter (λ t, non_degenerate t[0] t[1] t[2]) (points.combinations 3) = 2156 := by
  sorry

end count_triangles_with_positive_area_l209_209241


namespace probability_a_b_c_not_1_l209_209839

theorem probability_a_b_c_not_1 (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 8) (h2 : 1 ≤ b ∧ b ≤ 8) (h3 : 1 ≤ c ∧ c ≤ 8) :
  ((a - 1) * (b - 1) * (c - 1) ≠ 0) → 
  (real.of_rat (7 / 8) ^ 3 = real.of_rat (343 / 512)) :=
sorry

end probability_a_b_c_not_1_l209_209839


namespace dot_product_is_one_l209_209285

def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (-1, 2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

theorem dot_product_is_one : dot_product vec_a vec_b = 1 :=
by sorry

end dot_product_is_one_l209_209285


namespace evaluate_expression_l209_209603

/-- Definition of the greatest integer less than or equal to x (floor function) --/
def floor (x : ℝ) : ℤ := Int.toNat (x.floor)

/-- Assertion that evaluates the given expression when y = 7.2 --/
theorem evaluate_expression :
  let y := 7.2
  floor 6.5 * floor (2 / 3) + floor 2 * y + floor 8.4 - 6.2 = 16.2 :=
by
  sorry

end evaluate_expression_l209_209603


namespace max_value_of_expression_l209_209734

noncomputable def max_expression (c d : ℝ) (h1 : c > 0) (h2 : d > 0) : ℝ :=
  3 * (c - (1 / 2) * ((c^2 + d^2) / c)) * (sqrt ((c^2 + d^2)) + sqrt (((c^2 + d^2)^2 + d^2)))

theorem max_value_of_expression (c d : ℝ) (h1 : c > 0) (h2 : d > 0) :
  max_expression c d h1 h2 = (3 / 2) * (c^2 + d^2) := 
sorry

end max_value_of_expression_l209_209734


namespace odd_factors_of_360_l209_209231

theorem odd_factors_of_360 : ∃ n, 360 = 2^3 * 3^2 * 5^1 ∧ n = 6 :=
by 
  -- Define the prime factorization of 360
  let pf360 := (2^3 * 3^2 * 5^1)
  
  -- Define the count of odd factors by removing the contribution of factor 2.
  let odd_part := (3^2 * 5^1)
  
  -- Calculate the number of factors.
  have odd_factors_count_eq : (2 + 1) * (1 + 1) = 6 := by 
    calc (2 + 1) * (1 + 1) = 3 * 2 : by rfl
                         ... = 6 : by rfl
  
  -- Assert the existence of such n matching the count of odd factors.
  exact ⟨6, And.intro rfl odd_factors_count_eq⟩

end odd_factors_of_360_l209_209231


namespace number_of_odd_factors_of_360_l209_209207

theorem number_of_odd_factors_of_360 : 
  let factors (n : ℕ) := {d : ℕ | d ∣ n}
  let odd (d : ℕ) := d % 2 = 1
  let is_factor (n : ℕ) (d : ℕ) := d ∣ n
  let number_of_odd_factors (n : ℕ) := (factors n).count odd
  in number_of_odd_factors 360 = 6 :=
begin
  sorry
end

end number_of_odd_factors_of_360_l209_209207


namespace probability_roll_2_given_sum_3_l209_209861

open ProbabilityTheory

-- Define the structure of the unusual die
noncomputable def unusual_die : finset ℕ := {1, 2, 1, 1, 1, 2}

-- Define the probability space of rolling this die
noncomputable def die_probability_space := Finset.universalMeasure unusual_die

-- Define the event that the sum of rolled results is 3
def event_sum_3 (rolls : list ℕ) : Prop := rolls.sum = 3

-- Define the event that a roll resulted in a 2
def event_roll_2 (rolls : list ℕ) : Prop := 2 ∈ rolls

-- The theorem to prove the probability
theorem probability_roll_2_given_sum_3 (S : list ℕ) (hS : event_sum_3 S) :
  condProb (event_roll_2 S) (event_sum_3 S) die_probability_space = 0.6 := 
sorry

end probability_roll_2_given_sum_3_l209_209861


namespace exponent_multiplication_l209_209597

theorem exponent_multiplication :
  (625: ℝ) ^ 0.24 * (625: ℝ) ^ 0.06 = (5: ℝ) ^ (6 / 5) :=
by sorry

end exponent_multiplication_l209_209597


namespace sphere_radius_l209_209551

theorem sphere_radius (a b c d e f : ℝ) 
  (AP : 2 * a)
  (BP : 2 * b)
  (CP : 2 * c)
  (DP : 2 * d)
  (EP : 2 * e)
  (FP : 2 * f)
  (perpendicular_AB : (a * b) = (c * d) = (e * f)) : 
  let R := real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 - 2 * a * b - 2 * c * d - 2 * e * f) in
  true := 
sorry

end sphere_radius_l209_209551


namespace calculate_m_times_t_l209_209738

-- Define the function and conditions
def satisfies_functional_eq (g : ℝ → ℝ) :=
  ∀ x y z : ℝ, g(x^2 + y * g(z)) = x * g(x) + z * g(y)

-- Define the final proof goal
theorem calculate_m_times_t (g : ℝ → ℝ) (h : satisfies_functional_eq g) :
  (∃ m t : ℕ, m = 2 ∧ t = 3 ∧ m * t = 6) :=
begin
  sorry
end

end calculate_m_times_t_l209_209738


namespace unique_solution_implies_relation_l209_209481

theorem unique_solution_implies_relation (a b : ℝ)
    (h : ∃! (x y : ℝ), y = x^2 + a * x + b ∧ x = y^2 + a * y + b) : 
    a^2 = 2 * (a + 2 * b) - 1 :=
by
  sorry

end unique_solution_implies_relation_l209_209481


namespace number_of_odd_is_multiple_of_4_sum_of_squares_is_constant_l209_209331

noncomputable def E : Set ℕ := {n | 1 ≤ n ∧ n ≤ 200}
noncomputable def G : Set ℕ := {a | a ∈ E ∧ ∃ (i : ℕ), 1 ≤ i ∧ i ≤ 100}

axiom cond1 (G : Set ℕ) : ∀ (i j : ℕ), (i ≠ j ∧ i ∈ G ∧ j ∈ G) → i + j ≠ 201
axiom cond2 (G : Set ℕ) : ∑ a in G, a = 10080

theorem number_of_odd_is_multiple_of_4 (G : Set ℕ) : 
  (∃ n : ℕ, n % 4 = 0 ∧ (∑ a in G, if a % 2 = 1 then 1 else 0) = n) :=
sorry 

theorem sum_of_squares_is_constant (G : Set ℕ) : 
  (∃ c : ℕ, ∀ G' : Set ℕ, (G' = G) →
  (∑ a in G', a^2) = c) :=
sorry

end number_of_odd_is_multiple_of_4_sum_of_squares_is_constant_l209_209331


namespace resupply_percentage_l209_209505

def hiking_rate : Float := 2.5 -- miles per hour
def hiking_hours_per_day : Int := 8
def hiking_days : Int := 5
def supply_per_mile : Float := 0.5 -- pounds per mile
def initial_pack_weight : Float := 40 -- pounds

theorem resupply_percentage :
  let total_hours := hiking_hours_per_day * hiking_days
  let total_distance := hiking_rate * total_hours
  let total_supplies := total_distance * supply_per_mile
  let resupply_needed := total_supplies - initial_pack_weight
  let resupply_percentage := (resupply_needed / initial_pack_weight) * 100
  resupply_percentage = 25 :=
by {
  let total_hours := hiking_hours_per_day * hiking_days
  let total_distance := hiking_rate * ↑total_hours
  let total_supplies := total_distance * supply_per_mile
  let resupply_needed := total_supplies - initial_pack_weight
  let resupply_percentage := (resupply_needed / initial_pack_weight) * 100
  -- Here we need to calculate and prove resupply_percentage = 25
  sorry
}

end resupply_percentage_l209_209505


namespace path_sum_bounds_l209_209899

/--
Given a 4x4 multiplication table as follows:
| 1 | 2 | 3 | 4 |
|---|---|---|---|
| 2 | 4 | 6 | 8 |
| 3 | 6 | 9 | 12 |
| 4 | 8 | 12 | 16 |

and a path starting at the upper-left corner (1,1) to the lower-right corner (4,4) 
where movements are restricted to right or down, we want to prove:
1. The smallest possible sum of the path is 10.
2. The largest possible sum of the path is 30.
-/
theorem path_sum_bounds :
  ∃ min_sum max_sum : ℕ, 
  min_sum = 10 ∧
  max_sum = 30 :=
by
  -- Define the 4x4 multiplication table
  let mat := λ (i j : ℕ), (i + 1) * (j + 1)

  -- Consider paths that only move to the right or down
  -- Formal solution to prove desired properties (details omitted for brevity)
  sorry

end path_sum_bounds_l209_209899


namespace distance_center_to_point_l209_209451

def circle_center (h : ℝ) (k: ℝ) : Prop :=
  ∃ r: ℝ, ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r → x^2 + y^2 = 4*x + 6*y + 9

noncomputable def distance_between_points (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_center_to_point :
  ∃ h k : ℝ, circle_center h k ∧ distance_between_points h k 8 3 = 6 :=
by {
  sorry
}

end distance_center_to_point_l209_209451


namespace euler_quadrilateral_theorem_l209_209318

theorem euler_quadrilateral_theorem (A1 A2 A3 A4 P Q : ℝ) 
  (midpoint_P : P = (A1 + A3) / 2)
  (midpoint_Q : Q = (A2 + A4) / 2) 
  (length_A1A2 length_A2A3 length_A3A4 length_A4A1 length_A1A3 length_A2A4 length_PQ : ℝ)
  (h1 : length_A1A2 = A1A2) (h2 : length_A2A3 = A2A3)
  (h3 : length_A3A4 = A3A4) (h4 : length_A4A1 = A4A1)
  (h5 : length_A1A3 = A1A3) (h6 : length_A2A4 = A2A4)
  (h7 : length_PQ = PQ) :
  length_A1A2^2 + length_A2A3^2 + length_A3A4^2 + length_A4A1^2 = 
  length_A1A3^2 + length_A2A4^2 + 4 * length_PQ^2 := sorry

end euler_quadrilateral_theorem_l209_209318


namespace student_arrangement_l209_209849

theorem student_arrangement (n : ℕ) (A B : ℕ) (h1 : n = 6) (h2 : A = 1) (h3 : B = 1) : 
  let total_students := n,
      pair_AB := 2,
      factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1),
      unit_arrangements := (total_students - 1)!,
      pair_arrangement := pair_AB!,
      total_arrangements := unit_arrangements * pair_arrangement
  in total_arrangements = 240 :=
by 
  unfold factorial,
  intro total_students pair_AB factorial unit_arrangements pair_arrangement total_arrangements,
  simp [h1, h2, h3],
  unfold unit_arrangements pair_arrangement total_arrangements,
  simp,
  sorry

end student_arrangement_l209_209849


namespace find_x2017_l209_209626

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

-- Define that f is increasing
def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y
  
-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + n * d

-- Main theorem
theorem find_x2017
  (f : ℝ → ℝ) (x : ℕ → ℝ)
  (Hodd : is_odd_function f)
  (Hinc : is_increasing_function f)
  (Hseq : ∀ n, x (n + 1) = x n + 2)
  (H7_8 : f (x 7) + f (x 8) = 0) :
  x 2017 = 4019 := 
sorry

end find_x2017_l209_209626


namespace parabola_rotation_180_equivalent_l209_209374

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 - 2

-- Define the expected rotated parabola equation
def rotated_parabola (x : ℝ) : ℝ := -2 * (x - 3)^2 - 2

-- Prove that the rotated parabola is correctly transformed
theorem parabola_rotation_180_equivalent :
  ∀ x, rotated_parabola x = -2 * (x - 3)^2 - 2 := 
by
  intro x
  unfold rotated_parabola
  sorry

end parabola_rotation_180_equivalent_l209_209374


namespace solve_for_x_l209_209599

theorem solve_for_x :
  ∃ x : ℝ, 4^(x + 3) = 320 - 4^x ∧ x = (Real.log 64 - Real.log 13) / Real.log 4 :=
by
  use (Real.log 64 - Real.log 13) / Real.log 4
  split
  sorry

end solve_for_x_l209_209599


namespace true_propositions_l209_209557

section Propositions

variables {f : ℝ → ℝ} {a : ℝ}

-- Proposition 1
def prop1 (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x : ℝ, f(a + x) = f(a - x)

-- Proposition 2
def prop2 (f : ℝ → ℝ) :=
  ∀ x : ℝ, f(2 + x) = -f(x)

-- Proposition 3
def prop3 (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f(x1) - f(x2)) < 0

-- Proposition 4
def prop4 :=
  ∃ g : ℝ → ℝ, ∀ x : ℝ, g(x) = log (x + 3) - log 10

theorem true_propositions :
  (¬∀ f : ℝ → ℝ, ∀ a : ℝ, prop1 f a → (∀ x : ℝ, f(x) = f(-x))) ∧
  (∀ f : ℝ → ℝ, prop2 f → ∃ p : ℝ, ∀ x : ℝ, f(x + p) = f(x)) ∧
  (∀ f : ℝ → ℝ, prop3 f → ∀ x1 x2 : ℝ, x1 ≤ x2 → f(x1) ≥ f(x2)) ∧
  (prop4) :=
by
  admit -- Placeholder for the actual proof

end true_propositions_l209_209557


namespace number_of_odd_factors_of_360_l209_209196

def is_odd_factor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ ∀ k, 2 ∣ k → ¬ (k ∣ d)

theorem number_of_odd_factors_of_360 : 
  {d : ℕ // is_odd_factor 360 d}.card = 6 := 
sorry

end number_of_odd_factors_of_360_l209_209196


namespace circle_tangent_line_l209_209283

theorem circle_tangent_line (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, y = sqrt 2 * x → (x - a)^2 + y^2 = 2 → false) ↔ a = sqrt 3 := 
sorry

end circle_tangent_line_l209_209283


namespace odd_factors_360_l209_209180

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l209_209180


namespace problem1_problem2_l209_209894

theorem problem1 : -20 + 3 + 5 - 7 = -19 := by
  sorry

theorem problem2 : (-3)^2 * 5 + (-2)^3 / 4 - |-3| = 40 := by
  sorry

end problem1_problem2_l209_209894


namespace true_statements_l209_209733

variables (a b : Plane) (l m n : Line)

-- Given conditions
axiom planes_non_coincident : a ≠ b
axiom lines_non_coincident : l ≠ m ∧ m ≠ n ∧ n ≠ l

-- Statement 1 conditions
axiom plane_a_parallel_b : a ∥ b
axiom line_l_in_plane_a : l ∈ a

-- Statement 2 conditions
axiom line_m_in_plane_a : m ∈ a
axiom line_n_in_plane_a : n ∈ a
axiom lines_m_n_parallel_plane_b : m ∥ b ∧ n ∥ b

-- Statement 3 conditions
axiom line_l_parallel_plane_a : l ∥ a
axiom line_l_perpendicular_plane_b : l ⟂ b

-- Statement 4 conditions
axiom lines_m_n_skew : skew m n
axiom line_m_parallel_plane_a : m ∥ a
axiom line_n_parallel_plane_a : n ∥ a
axiom line_l_perpendicular_m_n : l ⟂ m ∧ l ⟂ n

-- Proof statement
theorem true_statements : ({1, 3, 4} : set ℕ) = 
                        ({if plane_a_parallel_b ∧ line_l_in_plane_a then l ∥ b else false,
                          if line_m_in_plane_a ∧ line_n_in_plane_a ∧ lines_m_n_parallel_plane_b then a ∥ b else false,
                          if line_l_parallel_plane_a ∧ line_l_perpendicular_plane_b then a ⟂ b else false,
                          if lines_m_n_skew ∧ line_m_parallel_plane_a ∧ line_n_parallel_plane_a ∧ line_l_perpendicular_m_n then l ⟂ a else false}: set ℕ) :=
by {
  -- Adding sorries since proof is not required
  sorry
}

end true_statements_l209_209733


namespace sqrt_two_irrational_l209_209833

theorem sqrt_two_irrational :
  ¬ ∃ (a b : ℕ), (a.gcd b = 1) ∧ (b ≠ 0) ∧ (a^2 = 2 * b^2) :=
sorry

end sqrt_two_irrational_l209_209833


namespace taxi_speed_l209_209516

theorem taxi_speed (v : ℝ) (hA : ∀ v : ℝ, 3 * v = 6 * (v - 30)) : v = 60 :=
by
  sorry

end taxi_speed_l209_209516


namespace parabola_axis_symmetry_value_p_l209_209159

theorem parabola_axis_symmetry_value_p (p : ℝ) (h_parabola : ∀ y x, y^2 = 2 * p * x) (h_axis_symmetry : ∀ (a: ℝ), a = -1 → a = -p / 2) : p = 2 :=
by 
  sorry

end parabola_axis_symmetry_value_p_l209_209159


namespace walking_speed_correct_l209_209507

-- Define the length of the bridge
def bridge_length : ℝ := 3000

-- Define the time taken to cross the bridge in minutes
def time_minutes : ℝ := 18

-- Define the distance in meters
def distance_meters : ℝ := bridge_length

-- Define the time in hours
def time_hours : ℝ := time_minutes / 60

-- Define the speed in km/hr
def speed_km_per_hr : ℝ := (distance_meters / 1000) / time_hours

-- Assert the correct speed in km/hr
theorem walking_speed_correct : speed_km_per_hr = 2.78 :=
by
  -- Proof is omitted
  sorry

end walking_speed_correct_l209_209507


namespace unique_solution_for_p_l209_209937

-- Define the function f(x) = ∛x + ∛(2 - x)
def f (x : ℝ) : ℝ := x^(1/3) + (2 - x)^(1/3)

-- Statement: The value of p for which the equation ∛x + ∛(2 - x) = p has exactly one solution is p = 2.
theorem unique_solution_for_p (p : ℝ) : 
(∃! x : ℝ, f x = p) ↔ p = 2 := 
begin
  sorry
end

end unique_solution_for_p_l209_209937


namespace budget_allocation_genetically_modified_microorganisms_l209_209017

theorem budget_allocation_genetically_modified_microorganisms :
  let microphotonics := 14
  let home_electronics := 19
  let food_additives := 10
  let industrial_lubricants := 8
  let total_percentage := 100
  let basic_astrophysics_percentage := 25
  let known_percentage := microphotonics + home_electronics + food_additives + industrial_lubricants + basic_astrophysics_percentage
  let genetically_modified_microorganisms := total_percentage - known_percentage
  genetically_modified_microorganisms = 24 := 
by
  sorry

end budget_allocation_genetically_modified_microorganisms_l209_209017


namespace problem_U_complement_eq_l209_209993

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l209_209993


namespace pow_gt_of_gt_l209_209476

variable {a x1 x2 : ℝ}

theorem pow_gt_of_gt (ha : a > 1) (hx : x1 > x2) : a^x1 > a^x2 :=
by sorry

end pow_gt_of_gt_l209_209476


namespace exists_non_intersecting_circle_l209_209014

noncomputable def circular_billiard_table (center : ℝ × ℝ) (radius : ℝ) : Prop :=
∃ (C : ℝ × ℝ) (r : ℝ), 
  C = center ∧ 0 < r ∧ r < radius ∧ 
  ∀ (ball_pos traj_point : ℝ × ℝ),
    ball_pos ≠ traj_point → 
    (∀ (p : ℝ × ℝ), 
      p ≠ C → 
      (euclidean_distance p center = r → ¬∃ t : ℝ, traj_point = ball_pos + t • p))

theorem exists_non_intersecting_circle 
  (center : ℝ × ℝ) 
  (radius : ℝ) 
  (ball_init_position : ℝ × ℝ) 
  (ball_trajectory : ℝ × ℝ → ℝ × ℝ) 
  (reflected_trajectory : ℝ × ℝ → ℝ × ℝ) : 
  (ball_init_position ≠ center ∧ euclidean_distance ball_init_position center < radius) → 
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    C = center ∧ 0 < r ∧ r < radius ∧ 
    (∀ (p : ℝ × ℝ), 
      p ≠ C → 
      euclidean_distance p C = r → ¬∃ t : ℝ, 
        ball_trajectory (reflected_trajectory ball_init_position) = ball_init_position + t • p) :=
sorry

end exists_non_intersecting_circle_l209_209014


namespace find_c_value_l209_209409

theorem find_c_value (x c : ℝ) (h1 : 3 * x + 9 = 0) (h2 : c * x ^ 2 - 7 = 6) : c = 13 / 9 := 
by 
  have hx : x = -3 := by linarith,
  rw [hx] at h2,
  have h_eq : c * 9 - 7 = 6, from h2,
  linarith

end find_c_value_l209_209409


namespace number_of_odd_factors_of_360_l209_209201

def is_odd_factor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ ∀ k, 2 ∣ k → ¬ (k ∣ d)

theorem number_of_odd_factors_of_360 : 
  {d : ℕ // is_odd_factor 360 d}.card = 6 := 
sorry

end number_of_odd_factors_of_360_l209_209201


namespace tens_digit_of_3_pow_2010_l209_209450

theorem tens_digit_of_3_pow_2010 : (3^2010 / 10) % 10 = 4 := by
  sorry

end tens_digit_of_3_pow_2010_l209_209450


namespace trajectory_C_range_PE_PF_angle_lambda_l209_209701

theorem trajectory_C (a : ℝ) (ha : 0 < a) (x y : ℝ)
  (A : ℝ × ℝ := (- real.sqrt 7 / 7 * a, 0)) 
  (B : ℝ × ℝ := (real.sqrt 7 / 7 * a, 0))
  (C : ℝ × ℝ := (x, y))
  (MA MB MC : ℝ × ℝ)
  (hM : (MA + MB + MC = (0,0))) 
  (hN : ∃ (N : ℝ × ℝ), |N - C| = real.sqrt 7 * |N - A| ∧ |N - C| = real.sqrt 7 * |N - B| 
         ∧ (N - (- real.sqrt 7 / 7 * a, 0)).fst = (N - (real.sqrt 7 / 7 * a, 0)).fst)
  : x^2 - y^2 / 3 = a^2 :=
sorry

theorem range_PE_PF (a : ℝ) (ha : 0 < a) (P E F : ℝ × ℝ)
  (hP : P = (0, a)) (hx2 : ∃ (k : ℝ), (3 - k^2) * E.fst^2 - 2 * a * k * E.fst - 4 * a^2 = 0)
  (hEF : E = (x1, y1) ∧ F = (x2, y2) 
         ∧ x1 + x2 = 2 * a * k / (3 - k^2) 
         ∧ x1 * x2 = - 4 * a^2 / (3 - k^2)
         ∧ (3 - k^2) * x1^2 - 2 * a * k * x1 - 4 * a^2 = 0)
  : ∃ x1 y1 x2 y2, -(∞ : ℝ) < (4 * a^2 * (1 + k^2) / (3 - k^2)) < 4 * a^2 ∨ 20 * a^2 < (4 * a^2 * (1 + k^2) / (3 - k^2)) < ∞ :=
sorry

theorem angle_lambda (a : ℝ) (ha : 0 < a) (x0 y0 : ℝ)
  (Q : ℝ × ℝ := (x0, y0)) 
  (G : ℝ × ℝ := (-a, 0)) 
  (H : ℝ × ℝ := (2 * a, 0))
  (hQ : x0^2 - y0^2 / 3 = a^2) 
  : ∃ λ : ℝ, λ = 2 ∧ 
    (Q.fst > 0 ∧ Q.snd > 0 
      → angle (H - Q) (Q - G) = λ * angle (Q - G) (H - Q)) :=
sorry

end trajectory_C_range_PE_PF_angle_lambda_l209_209701


namespace minimum_distance_AB_l209_209145

theorem minimum_distance_AB : 
  ∀ (A B : ℝ × ℝ), A.1 ^ 2 - A.2 + 1 = 0 → B.2 ^ 2 - B.1 + 1 = 0 → 
  dist A B = real.sqrt (9 / 8) → dist A B = 3 * real.sqrt 2 / 4 :=
begin
  intros A B hA hB hdist,
  rw hdist,
  sorry
end

end minimum_distance_AB_l209_209145


namespace problem_statement_l209_209973

def f (x : ℝ) : ℝ :=
  if x >= 0 then Real.cos (Real.pi * x)
  else 2 / x

theorem problem_statement : f (f (4 / 3)) = -4 := 
  sorry

end problem_statement_l209_209973


namespace area_quadrilateral_31_l209_209404

variables {A B C D O : Type} [metric_space A] [has_dist A]

-- Given lengths of sides
def length_AB : ℝ := 10
def length_BC : ℝ := 6
def length_CD : ℝ := 8
def length_DA : ℝ := 2
def angle_COB : real.angle := real.angle.of_deg 45

-- Diagonals intersecting at O
variables {AC BD : set (set A)} (intersect_at_O : AC ∩ BD = {O})

-- Define points in the space
variables (A B C D O : A)

-- Proposition statement
theorem area_quadrilateral_31 : 
  ∀ (A B C D O : A) [inhabited A],
    dist A B = length_AB ∧ 
    dist B C = length_BC ∧ 
    dist C D = length_CD ∧ 
    dist D A = length_DA ∧ 
    angle_cob = angle_COB ∧ 
    AC ∩ BD = {O} →
    quadrilateral_area A B C D = 31 := sorry

end area_quadrilateral_31_l209_209404


namespace even_m_n_l209_209119

variable {m n : ℕ}

theorem even_m_n
  (h_m : ∃ k : ℕ, m = 2 * k + 1)
  (h_n : ∃ k : ℕ, n = 2 * k + 1) :
  Even ((m - n) ^ 2) ∧ Even ((m - n - 4) ^ 2) ∧ Even (2 * m * n + 4) :=
by
  sorry

end even_m_n_l209_209119


namespace kids_go_to_camp_l209_209726

theorem kids_go_to_camp (total_kids stay_home kids_at_camp : ℕ) 
(h_total : total_kids = 313473) 
(h_stay : stay_home = 274865) 
(h_camp : kids_at_camp = total_kids - stay_home) : 
kids_at_camp = 38608 :=
by
  rw [h_total, h_stay]
  rw h_camp
  norm_num
  sorry

end kids_go_to_camp_l209_209726


namespace correct_option_is_B_l209_209468

theorem correct_option_is_B :
  (¬ ∃ (x : ℝ), x^2 - 1 < 0 ↔ ∀ (x : ℝ), x^2 - 1 ≥ 0) ∧
  ((∀ (x : ℝ), x = 3 → x^2 - 2 * x - 3 = 0) ↔ (∀ (x : ℝ), x ≠ 3 → x^2 - 2 * x - 3 ≠ 0)) ∧
  (∀ (α : ℝ), (∃ (k : ℤ), α = 2 * k * real.pi + real.pi / 3) → sane (real.sin (2 * α) = sqrt 3 / 2) ∧
               ¬(∀ (α : ℝ), real.sin (2 * α) = sqrt 3 / 2 → ∃ (k : ℤ), α = k * real.pi + real.pi / 6 ∨ α = k * real.pi + real.pi / 3)) ∧
  (¬ ∀ (x y : ℝ), real.cos x = real.cos y → x ≠ y → (¬ (real.cos x ≠ real.cos y) → ¬ (x ≠ y))) →
  true := sorry

end correct_option_is_B_l209_209468


namespace bob_clean_time_l209_209531

-- Definitions for the problem conditions
def alice_time : ℕ := 30
def bob_time := (1 / 3 : ℚ) * alice_time

-- The proof problem statement (only) in Lean 4
theorem bob_clean_time : bob_time = 10 := by
  sorry

end bob_clean_time_l209_209531


namespace constant_slope_of_line_EF_l209_209621

noncomputable def ellipse_equation (a b : ℝ) (h : a > b > 0) : Prop := a = 2 ∧ b = sqrt 3 ∧
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔
  x^2 / 4 + y^2 / 3 = 1

theorem constant_slope_of_line_EF
  (a b : ℝ)
  (h1 : a > b > 0)
  (h2 : ellipse_equation a b h1)
  (F1 F2 : ℝ × ℝ)
  (hF1 : F1 = (-1, 0))
  (hF2 : F2 = (1, 0))
  (P : ℝ × ℝ)
  (hP : P = (1, sqrt 3 / 2))
  (E F : ℝ × ℝ)
  (hEF_distinct : E ≠ F)
  (hE : E ∉ {P})
  (hF : F ∉ {P})
  (isComplementary : ∃ k : ℝ, (∃ E_line, E_line = (E.snd - P.snd) / (E.fst - P.fst) = k) ∧
    (∃ F_line, F_line = (F.snd - P.snd) / (F.fst - P.fst) = -1/k)) :
  slope_of (E, F) = 1/2 := sorry

end constant_slope_of_line_EF_l209_209621


namespace euler_formula_imag_part_l209_209570

theorem euler_formula_imag_part (θ : ℝ) (π : ℝ) (i : ℂ) [Preorder ℝ] [Ring ℂ] (h : θ = π / 6) :
  complex.exp(θ * i).im = 1 / 2 :=
by
  have h1 : exp(θ * I) = cos θ + I*sin θ := by sorry
  rw [h] at h1
  have h2 : cos(π / 6) = √3 / 2 := by sorry
  have h3 : sin(π / 6) = 1 / 2 := by sorry
  rw [h2, h3] at h1
  exact h1.2

end euler_formula_imag_part_l209_209570


namespace figure_representation_l209_209498

theorem figure_representation (x y : ℝ) : 
  |x| + |y| ≤ 2 * Real.sqrt (x^2 + y^2) ∧ 2 * Real.sqrt (x^2 + y^2) ≤ 3 * max (|x|) (|y|) → 
  Figure2 :=
sorry

end figure_representation_l209_209498


namespace proof_of_concurrency_l209_209332

open EuclideanGeometry 

noncomputable def concurrency_of_lines 
  (A B C O B1 C1 A2 C2 A3 B3 : Point) 
  (circumcircle_OBC : Circle) (circumcircle_OAC : Circle) (circumcircle_OAB : Circle) : Prop :=
  circumcenter A B C O ∧ 
  is_on_circle B1 circumcircle_OBC ∧ B1 ≠ B ∧ intersect AB B1 circumcircle_OBC ∧
  is_on_circle C1 circumcircle_OBC ∧ C1 ≠ C ∧ intersect AC C1 circumcircle_OBC ∧
  is_on_circle A2 circumcircle_OAC ∧ A2 ≠ A ∧ intersect BA A2 circumcircle_OAC ∧
  is_on_circle C2 circumcircle_OAC ∧ C2 ≠ C ∧ intersect BC C2 circumcircle_OAC ∧
  is_on_circle A3 circumcircle_OAB ∧ A3 ≠ A ∧ intersect CA A3 circumcircle_OAB ∧
  is_on_circle B3 circumcircle_OAB ∧ B3 ≠ B ∧ intersect CB B3 circumcircle_OAB → 
  concurrent A2 A3 B1 B3 C1 C2

-- Usage of this def would be as follows:
theorem proof_of_concurrency :
  ∀ (A B C O B1 C1 A2 C2 A3 B3 : Point) 
  (circumcircle_OBC circumcircle_OAC circumcircle_OAB : Circle),
  concurrency_of_lines A B C O B1 C1 A2 C2 A3 B3 circumcircle_OBC circumcircle_OAC circumcircle_OAB :=
begin
  -- Proof would go here
  sorry
end

end proof_of_concurrency_l209_209332


namespace find_x_l209_209259

theorem find_x (x : ℝ) : (3^(x - 4) = 9^3) → x = 10 :=
by
  sorry

end find_x_l209_209259


namespace binary_ternary_product_l209_209547

theorem binary_ternary_product :
  let b1 := 1 * (2^3) + 1 * (2^2) + 0 * (2^1) + 1 * (2^0),
      t1 := 1 * (3^2) + 0 * (3^1) + 2 * (3^0)
  in b1 * t1 = 143 :=
by
  let b1 := 1 * (2^3) + 1 * (2^2) + 0 * (2^1) + 1 * (2^0),
      t1 := 1 * (3^2) + 0 * (3^1) + 2 * (3^0)
  exact sorry

end binary_ternary_product_l209_209547


namespace problem1_problem2_l209_209153

noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / (4^x + 1)

theorem problem1 (x : ℝ) : f x < 1 / 3 ↔ x < 1 / 2 := 
by sorry

theorem problem2 : set.range f = set.Ioo (-1 : ℝ) (1 : ℝ) :=
by sorry

end problem1_problem2_l209_209153


namespace evaluate_series_l209_209917

-- Define the general term of the series
def term (n : ℕ) : ℝ := (n^3 + 2*n^2 - n + 1 : ℝ) / (n + 3)!

-- Define the infinite series sum function
noncomputable def infiniteSum (f : ℕ → ℝ) : ℝ :=
  ∑' n, f n

-- Define the theorem to evaluate the sum of the series
theorem evaluate_series : infiniteSum term = 1 / 3 :=
  sorry

end evaluate_series_l209_209917


namespace negation_of_p_l209_209349

-- Define the proposition p
def p : Prop := ∀ x : ℝ, Real.exp x > Real.log x

-- Define the negation of p
def neg_p : Prop := ∃ x : ℝ, Real.exp x ≤ Real.log x

-- The statement we want to prove
theorem negation_of_p : ¬p ↔ neg_p :=
by sorry

end negation_of_p_l209_209349


namespace problem_statement_l209_209062

noncomputable def expr : ℝ :=
  (1 - Real.sqrt 5)^0 + abs (-Real.sqrt 2) - 2 * Real.cos (Real.pi / 4) + (1 / 4 : ℝ)⁻¹

theorem problem_statement : expr = 5 := by
  sorry

end problem_statement_l209_209062


namespace number_of_valid_zeros_l209_209341

def f (x : ℝ) := Real.sin (Real.pi * x)
def is_zero_of_f (x : ℝ) := f x = 0
def is_valid_x0 (x0 : ℝ) := |x0| + f (x0 + 0.5) < 11

theorem number_of_valid_zeros : 
  {x0 : ℝ // is_zero_of_f x0 ∧ is_valid_x0 x0}.card = 21 :=
sorry

end number_of_valid_zeros_l209_209341


namespace all_squares_same_color_l209_209132

theorem all_squares_same_color (n : ℕ) : 
  (∃ sequence : List (ℕ × ℕ × ℕ × ℕ), (∀ (r : ℕ × ℕ × ℕ × ℕ), 
    r ∈ sequence → 
    (r.2.1 - r.1 + 1) > 1 ∧ 
    (r.4 - r.3 + 1) > 1 ∧ 
    ((r.2.1 - r.1 + 1) % 2 = 0 ∨ (r.2.1 - r.1 + 1) % 2 = 1) ∧ 
    ((r.4 - r.3 + 1) % 2 = 0 ∨ (r.4 - r.3 + 1) % 2 = 1) ∧ 
    (∀ i j, i ∈ List.range n → j ∈ List.range n → 
      ∃ c : Char, c = 'B' ∨ c = 'W' ∧ 
      (i % 2 = j % 2 → c = 'B') ∧ 
      (i % 2 ≠ j % 2 → c = 'W') ∧ 
      ∃ n_color : Char, sequence.foldl (λ color m, if (i, j) ∈ range(m.1, m.2.1) × range(m.3, m.4) 
        then if color = 'B' then 'W' else 'B' else color) c = n_color ∧
      n_color = 'B' ∨ n_color = 'W')) ↔ 
  n = 1 ∨ n ≥ 3 := 
sorry

end all_squares_same_color_l209_209132


namespace card_average_100_l209_209860

theorem card_average_100 (n : ℕ) (h1 : n > 0)
  (h2 : (∑ k in Finset.range n, k * (2 * k - 1)) / n ^ 2 = 100) : n = 10 :=
by
  sorry

end card_average_100_l209_209860


namespace real_part_of_z_l209_209121

-- Define the condition: z + 2conj(z) = 6 + i
def z_condition (z : ℂ) : Prop := z + 2 * conj(z) = 6 + complex.I

-- Define the proof problem: If z satisfies the condition, then the real part of z is 2
theorem real_part_of_z (z : ℂ) (h : z_condition z) : z.re = 2 := 
by
  sorry

end real_part_of_z_l209_209121


namespace sin_75_l209_209069

theorem sin_75 :
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end sin_75_l209_209069


namespace range_of_a_l209_209643

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a * x else a^2 * x - 7 * a + 14

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ a = f x₂ a) :
  a ∈ set.Ioo (-∞ : ℝ) 2 ∪ set.Ioo 3 5 :=
sorry

end range_of_a_l209_209643


namespace phi_value_unique_l209_209648

theorem phi_value_unique (omega : ℝ) (phi : ℝ) (k : ℤ)
  (h1 : omega > 0)
  (h2 : abs(phi) < real.pi / 2)
  (h3 : 2 * real.pi / omega = real.pi)
  (h4 : ∀ x, real.sin (omega * x - real.pi / 3 + phi) = real.sin (omega * x)) :
  phi = real.pi / 3 :=
by
  sorry

end phi_value_unique_l209_209648


namespace sum_first_n_terms_bn_l209_209797

theorem sum_first_n_terms_bn (n : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) :
  (∀ n, a n = 2 * n + 1) →
  (∀ n, b n = (∑ i in finset.range (n + 1), a i) / (n + 1)) →
  (∑ i in finset.range n, b (i + 1) ) = n * (n + 5) / 2 :=
by
  sorry

end sum_first_n_terms_bn_l209_209797


namespace hexagon_to_square_l209_209900

theorem hexagon_to_square (a : ℝ) : 
  ∃ (pieces : list (set (ℝ×ℝ))), 
  (is_regular_hexagon a ∧
   (hexagon_area a = (3 * Real.sqrt 3 / 2) * a^2) ∧
   (∃ (s : ℝ), s = Real.sqrt ((3 * Real.sqrt 3) / 2) * a ∧
     reassemble_as_square pieces s)) 
  :=
sorry

-- Definitions for regular hexagon and area calculations
def is_regular_hexagon (a : ℝ) : Prop :=
  -- Definition details go here, regarding vertices and angles

def hexagon_area (a : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * a^2

def reassemble_as_square (pieces : list (set (ℝ×ℝ))) (s : ℝ) : Prop :=
  -- Definition details go here about rearranging pieces to form square

end hexagon_to_square_l209_209900


namespace cost_of_each_notebook_is_3_l209_209560

noncomputable def notebooks_cost (total_spent : ℕ) (backpack_cost : ℕ) (pens_cost : ℕ) (pencils_cost : ℕ) (num_notebooks : ℕ) : ℕ :=
  (total_spent - (backpack_cost + pens_cost + pencils_cost)) / num_notebooks

theorem cost_of_each_notebook_is_3 :
  notebooks_cost 32 15 1 1 5 = 3 :=
by
  sorry

end cost_of_each_notebook_is_3_l209_209560


namespace parabola_equation_l209_209904

-- Definitions of the conditions
def parabola_passes_through (x y : ℝ) : Prop :=
  y^2 = -2 * (3 * x)

def focus_on_line (x y : ℝ) : Prop :=
  3 * x - 2 * y - 6 = 0

theorem parabola_equation (x y : ℝ) (hM : x = -6 ∧ y = 6) (hF : ∃ (x y : ℝ), focus_on_line x y) :
  parabola_passes_through x y = (y^2 = -6 * x) :=
by 
  sorry

end parabola_equation_l209_209904


namespace parity_odd_function_add_property_find_fa_l209_209642

noncomputable def f (x: ℝ) : ℝ := Real.log2 ((1 + x) / (1 - x))

-- Parity: Prove that f(x) is an odd function
theorem parity_odd_function (x : ℝ) (h : -1 < x) (h' : x < 1) : f(-x) = -f(x) := sorry

-- Addition property: Prove that f(x1) + f(x2) = f((x1 + x2) / (1 + x1 * x2))
theorem add_property (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < 1) (h3 : -1 < x2) (h4 : x2 < 1) :
  f(x1) + f(x2) = f((x1 + x2) / (1 + x1 * x2)) := sorry

-- Given conditions: Prove the specific value of f(a)
theorem find_fa (a b : ℝ) (h1 : -1 < a) (h2 : a < 1) (h3 : -1 < b) (h4 : b < 1) 
  (h5 : f((a + b) / (1 + a * b)) = 1) (h6 : f(-b) = 1/2) : f(a) = 3/2 := sorry

end parity_odd_function_add_property_find_fa_l209_209642


namespace maximum_guaranteed_money_l209_209012

theorem maximum_guaranteed_money (board_width board_height tromino_width tromino_height guaranteed_rubles : ℕ) 
  (h_board_width : board_width = 21) 
  (h_board_height : board_height = 20)
  (h_tromino_width : tromino_width = 3) 
  (h_tromino_height : tromino_height = 1)
  (h_guaranteed_rubles : guaranteed_rubles = 14) :
  true := by
  sorry

end maximum_guaranteed_money_l209_209012


namespace find_x_l209_209264

theorem find_x (x : ℝ) : (3^(x - 4) = 9^3) → x = 10 :=
by
  intro h
  sorry

end find_x_l209_209264


namespace odd_factors_of_360_l209_209177

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l209_209177


namespace number_smaller_than_neg2_l209_209532

theorem number_smaller_than_neg2 (a b c d : ℝ) (S : set ℝ) (hS : S = {-3, -1/2, 0, 2})
  (h1 : a ∈ S) (h2 : b ∈ S) (h3 : c ∈ S) (h4 : d ∈ S) :
  a < -2 → a = -3 :=
by sorry

end number_smaller_than_neg2_l209_209532


namespace find_counterfeit_coin_l209_209031

def is_counterfeit (coins : Fin 9 → ℝ) (i : Fin 9) : Prop :=
  ∀ j : Fin 9, j ≠ i → coins j = coins 0 ∧ coins i < coins 0

def algorithm_exists (coins : Fin 9 → ℝ) : Prop :=
  ∃ f : (Fin 9 → ℝ) → Fin 9, is_counterfeit coins (f coins)

theorem find_counterfeit_coin (coins : Fin 9 → ℝ) (h : ∃ i : Fin 9, is_counterfeit coins i) : algorithm_exists coins :=
by sorry

end find_counterfeit_coin_l209_209031


namespace periodic_even_function_value_l209_209680

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * (x - a)

-- Conditions: 
-- 1. f(x) is even 
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- 2. f(x) is periodic with period 6
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f x = f (x + p)

-- Main theorem
theorem periodic_even_function_value 
  (a : ℝ) 
  (f_def : ∀ x, -3 ≤ x ∧ x ≤ 3 → f x a = (x + 1) * (x - a))
  (h_even : is_even_function (f · a))
  (h_periodic : is_periodic_function (f · a) 6) : 
  f (-6) a = -1 := 
sorry

end periodic_even_function_value_l209_209680


namespace lines_perpendicular_to_same_plane_are_parallel_l209_209700

theorem lines_perpendicular_to_same_plane_are_parallel
  (l₁ l₂ : ℝ^3) (π : set ℝ^3) 
  (h₁ : ∀ p ∈ π, ∃ θ : ℝ, is_perpendicular l₁ (p - (l₁ / θ)))
  (h₂ : ∀ p ∈ π, ∃ θ : ℝ, is_perpendicular l₂ (p - (l₂ / θ))) :
  is_parallel l₁ l₂ :=
sorry

end lines_perpendicular_to_same_plane_are_parallel_l209_209700


namespace max_possible_a_l209_209503

theorem max_possible_a :
  ∃ (a : ℚ), ∀ (m : ℚ), (1/3 < m ∧ m < a) →
    (∀ x : ℤ, 0 < x ∧ x ≤ 150 → (∀ y : ℤ, y ≠ m * x + 3)) ∧ a = 50/149 :=
by
  sorry

end max_possible_a_l209_209503


namespace odd_factors_360_l209_209212

theorem odd_factors_360 : 
  let fac := (2^3) * (3^2) * (5^1) in
  fac = 360 → 
  num_factors (3^2 * 5^1) = 6 :=
sorry

end odd_factors_360_l209_209212


namespace price_of_horse_and_cow_l209_209703

theorem price_of_horse_and_cow (x y : ℝ) (h1 : 4 * x + 6 * y = 48) (h2 : 3 * x + 5 * y = 38) :
  (4 * x + 6 * y = 48) ∧ (3 * x + 5 * y = 38) := 
by
  exact ⟨h1, h2⟩

end price_of_horse_and_cow_l209_209703


namespace pascal_triangle_probability_l209_209877

theorem pascal_triangle_probability : 
  let rows := 20
  let total_elements := ∑ i in range rows, (i + 1)
  let occurrences_of_2 := 2 * (rows - 2)
  let probability := occurrences_of_2 / total_elements
  probability = 6 / 35 :=
by
  sorry

end pascal_triangle_probability_l209_209877


namespace upper_limit_of_arun_weight_l209_209292

variable (w : ℝ)

noncomputable def arun_opinion (w : ℝ) := 62 < w ∧ w < 72
noncomputable def brother_opinion (w : ℝ) := 60 < w ∧ w < 70
noncomputable def average_weight := 64

theorem upper_limit_of_arun_weight 
  (h1 : ∀ w, arun_opinion w → brother_opinion w → 64 = (62 + w) / 2 ) 
  : ∀ w, arun_opinion w ∧ brother_opinion w → w ≤ 66 :=
sorry

end upper_limit_of_arun_weight_l209_209292


namespace initial_notebooks_order_l209_209105

-- Define the initial conditions
def initial_notebooks : List String := ["blue", "gray", "brown", "red", "yellow"]

def first_stack_after_first_decomposition : List String := ["gray", "yellow", "red"]
def second_stack_after_first_decomposition : List String := ["blue", "brown"]

def first_stack_after_second_decomposition : List String := ["red", "brown"]
def second_stack_after_second_decomposition : List String := ["blue", "gray", "yellow"]

-- Define the theorem to prove the initial order of notebooks
theorem initial_notebooks_order :
  (first_stack_after_first_decomposition ++ second_stack_after_first_decomposition = initial_notebooks) ∧
  ((second_stack_after_first_decomposition.rev ++ first_stack_after_first_decomposition.rev).rev = first_stack_after_second_decomposition ++ second_stack_after_second_decomposition) →
  initial_notebooks = ["brown", "red", "yellow", "gray", "blue"] :=
  sorry

end initial_notebooks_order_l209_209105


namespace find_x_l209_209262

theorem find_x (x : ℝ) : (3^(x - 4) = 9^3) → x = 10 :=
by
  intro h
  sorry

end find_x_l209_209262


namespace intersection_complement_l209_209748

open Set

theorem intersection_complement (U A B : Set ℕ) (hU : U = {x | x ≤ 6}) (hA : A = {1, 3, 5}) (hB : B = {4, 5, 6}) :
  (U \ A) ∩ B = {4, 6} := by
  sorry

end intersection_complement_l209_209748


namespace measure_angle_GDA_l209_209073

-- Definitions of angles and geometric properties
variables (A B C D E F G : Type)

-- Conditions: $ABCD$ and $DEFG$ are squares, and $CDE$ is a right triangle with ∠CDE = 90°
def square (a b c d : Type) : Prop := 
  ∀ (α β γ δ : ℝ), α = β ∧ β = γ ∧ γ = δ ∧ δ = α ∧ α = 90

def right_triangle (a b c : Type) : Prop := 
  ∃ (α β γ : ℝ), α + β + γ = 180 ∧ α = 90

-- Given condition that angles ADC, CDE, and FDG are each 90 degrees
noncomputable def angle_CDE_90 := (90 : ℝ)

noncomputable def angle_ADC_90 := (90 : ℝ)

noncomputable def angle_FDG_90 := (90 : ℝ)

-- Problem statement in Lean:
theorem measure_angle_GDA :
  square A B C D ∧ square D E F G ∧ right_triangle C D E ∧ ∠CDE = 90 ∧ ∠ADC = 90 ∧ ∠FDG = 90
  → ∠GDA = 90 := by sorry

end measure_angle_GDA_l209_209073


namespace find_a_l209_209164

variables (U : Set ℝ) (M N : Set ℝ) (a : ℝ)

-- Conditions
def universal_set := U = {x | true}
def set_M := M = {x | x + 2 * a >= 0}
def set_N := N = {x | log 2 (x - 1) < 1}
def intersection_condition := M ∩ (U \ N) = {x | x = 1 ∨ x >= 3}

-- Theorem: prove that given the conditions, a = -1/2
theorem find_a (hU : universal_set U) (hM : set_M M a) (hN : set_N N) (hInt : intersection_condition M N a) : a = -1/2 := 
sorry

end find_a_l209_209164


namespace dice_problem_probability_l209_209931

theorem dice_problem_probability :
  (∀ (dice : Fin 5 → ℕ), (∀ i, 1 ≤ dice i ∧ dice i ≤ 6) → 
    (∃ n, ∃ s, (∃ i j k, n = dice i ∧ n = dice j ∧ n = dice k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧ s = (finset.univ.sum dice) ∧ s > 20) →
    (∃ p, p = (31 / 432))) := 
sorry

end dice_problem_probability_l209_209931


namespace farthest_from_origin_l209_209840

-- Definition of distance from the origin
def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  real.sqrt (p.1^2 + p.2^2)

-- The points to consider
def points : list (ℝ × ℝ) := [(2, 3), (4, -1), (-3, 4), (0, -7), (5, 0), (-6, 2)]

-- The point claimed to be farthest from the origin
def farthest_point : ℝ × ℝ := (0, -7)

-- Mathematical statement of the proof problem
theorem farthest_from_origin : ∀ p ∈ points, distance_from_origin p ≤ distance_from_origin farthest_point :=
by sorry

end farthest_from_origin_l209_209840


namespace num_repeating_decimals_l209_209113

theorem num_repeating_decimals : 
  let N := 10 in
  let integers_in_range := (1:ℤ) \u003c = n \u0026 \u0026 n \u003c = (15:ℤ) in
  ∀ n : ℤ, 
  (integers_in_range n) → 
  (nat.gcd (nat.abs n) (nat.abs 18) = 1 →
  (∃ (count : ℕ), 
    count = 15 - 5 ∧ count = N)) :=
sorry

end num_repeating_decimals_l209_209113


namespace arithmetic_seq_common_diff_l209_209049

theorem arithmetic_seq_common_diff (n : ℕ) (a : ℕ → ℤ)
  (h1 : ∑ i in (finset.range n).map finset.pred_multiset (λ i => a (2 * i + 1)) = 90)
  (h2 : ∑ i in (finset.range n).map finset.pred_multiset (λ i => a (2 * i + 2)) = 72)
  (h3 : a 1 - a (2 * n) = 33) :
  ∃ d : ℤ, d = -3 :=
begin
  have h4 : n * d = -18, from sorry,
  have h5 : (2 * n - 1) * d = -33, from sorry,
  use -3,
  have d_neg : d = -3, from sorry,
  exact ⟨d_neg⟩
end

end arithmetic_seq_common_diff_l209_209049


namespace smaller_number_is_5_l209_209813

theorem smaller_number_is_5 (x y : ℤ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 := by
  sorry

end smaller_number_is_5_l209_209813


namespace remainder_add_l209_209352

theorem remainder_add (a b : ℤ) (n m : ℤ) 
  (ha : a = 60 * n + 41) 
  (hb : b = 45 * m + 14) : 
  (a + b) % 15 = 10 := by 
  sorry

end remainder_add_l209_209352


namespace sqrt_sum_eq_fraction_l209_209274

-- Definitions as per conditions
def w : ℕ := 4
def x : ℕ := 9
def z : ℕ := 25

-- Main theorem statement
theorem sqrt_sum_eq_fraction : (Real.sqrt (w / x) + Real.sqrt (x / z) = 19 / 15) := by
  sorry

end sqrt_sum_eq_fraction_l209_209274


namespace godzilla_stitches_l209_209895

theorem godzilla_stitches :
  (∀ (Carolyn_time : ℕ) (unicorn_stitches flower_stitches time_required unicorns flowers : ℕ),
  unicorn_stitches = 180 →
  flower_stitches = 60 →
  time_required = 1085 →
  unicorns = 3 →
  flowers = 50 →
  Carolyn_time = 4 →
  let total_stitches := time_required * Carolyn_time in
  let total_unicorn_stitches := unicorns * unicorn_stitches in
  let total_flower_stitches := flowers * flower_stitches in
  let total_embroidery_stitches := total_unicorn_stitches + total_flower_stitches in
  total_stitches - total_embroidery_stitches = 800) :=
begin
  intros Carolyn_time unicorn_stitches flower_stitches time_required unicorns flowers,
  intros hunicorn hflower htime hunicorns hflowers ht,
  simp [hunicorn, hflower, htime, hunicorns, hflowers, ht],
  let total_stitches := 1085 * 4,
  let total_unicorn_stitches := 3 * 180,
  let total_flower_stitches := 50 * 60,
  let total_embroidery_stitches := total_unicorn_stitches + total_flower_stitches,
  have h_calculation: total_stitches - total_embroidery_stitches = 800 := by 
  {
    calc
      total_stitches = 4340 : rfl
      ... : - total_embroidery_stitches
      ... : - (total_unicorn_stitches + total_flower_stitches)
      ... : - (540 + 3000) 
      ... : = 800 : by simp,
  sorry
end

end godzilla_stitches_l209_209895


namespace ground_beef_total_cost_l209_209168

-- Define the conditions
def price_per_kg : ℝ := 5.00
def quantity_in_kg : ℝ := 12

-- The total cost calculation
def total_cost (price_per_kg quantity_in_kg : ℝ) : ℝ := price_per_kg * quantity_in_kg

-- Theorem statement
theorem ground_beef_total_cost :
  total_cost price_per_kg quantity_in_kg = 60.00 :=
sorry

end ground_beef_total_cost_l209_209168


namespace find_b_l209_209426

theorem find_b (a b : ℝ) (k : ℝ) (h1 : a = 4) (h2 : b = 16) (h3 : a^2 * real.sqrt b = k) (h4 : a + b = 20) : b = 16 :=
sorry -- Proof to be filled in later

end find_b_l209_209426


namespace hundredth_term_in_sequence_l209_209687

theorem hundredth_term_in_sequence : 
  ∃ (a : ℕ), (a * 3 + 1 - 1) + 99 * 3 = 298 := 
begin
  sorry
end

end hundredth_term_in_sequence_l209_209687


namespace inscribed_shapes_ratio_l209_209514

theorem inscribed_shapes_ratio {a b c : ℕ} (h : a^2 + b^2 = c^2) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) : 
  (∃ (x y : ℚ), x = b ∧ y = (a * b) / c ∧ x / y = 13 / 5) :=
by
  sorry

end inscribed_shapes_ratio_l209_209514


namespace count_odd_factors_of_360_l209_209224

-- Defining the prime factorization of 360
def prime_factorization_360 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

-- Defining the exponent range based on prime factorization for an odd factor
def exponent_ranges : List (ℕ × List ℕ) := [(3, [0, 1, 2]), (5, [0, 1])]

theorem count_odd_factors_of_360 : ∃ n : ℕ, n = 6 :=
by
  -- Conditions based on prime factorization
  have h₁ : prime_factorization_360 = [(2, 3), (3, 2), (5, 1)], from rfl
  have h₂ : exponent_ranges = [(3, [0, 1, 2]), (5, [0, 1])], from rfl
  
  -- Definitions should align with the conditions in the problem
  use 6
  sorry

end count_odd_factors_of_360_l209_209224


namespace total_time_to_fill_tank_l209_209437

theorem total_time_to_fill_tank :
  let rate1 := 1/20,
      rate2 := 1/30,
      rate3 := 1/40,
      combined_rate_before_leak := rate1 + rate2 + rate3,
      t1 := 0.8 / combined_rate_before_leak,
      reduced_rate := (3/4) * combined_rate_before_leak,
      t2 := 0.2 / reduced_rate,
      total_time := t1 + t2
  in total_time ≈ 9.84 :=
by
  let rate1 := 1/20
  let rate2 := 1/30
  let rate3 := 1/40
  let combined_rate_before_leak := rate1 + rate2 + rate3
  let t1 := 0.8 / combined_rate_before_leak
  let reduced_rate := (3/4) * combined_rate_before_leak
  let t2 := 0.2 / reduced_rate
  let total_time := t1 + t2
  have h : total_time ≈ 9.84 := sorry
  exact h

end total_time_to_fill_tank_l209_209437


namespace abc_over_ab_bc_ca_l209_209422

variable {a b c : ℝ}

theorem abc_over_ab_bc_ca (h1 : ab / (a + b) = 2)
                          (h2 : bc / (b + c) = 5)
                          (h3 : ca / (c + a) = 7) :
        abc / (ab + bc + ca) = 35 / 44 :=
by
  -- The proof would go here.
  sorry

end abc_over_ab_bc_ca_l209_209422


namespace sqrt_fraction_simplification_l209_209890

theorem sqrt_fraction_simplification : (real.sqrt 3) / (real.sqrt 3 + real.sqrt 12) = 1 / 3 := sorry

end sqrt_fraction_simplification_l209_209890


namespace find_a_given_integer_roots_l209_209791

-- Given polynomial equation and the condition of integer roots
theorem find_a_given_integer_roots (a : ℤ) :
    (∃ x y : ℤ, x ≠ y ∧ (x^2 - (a+8)*x + 8*a - 1 = 0) ∧ (y^2 - (a+8)*y + 8*a - 1 = 0)) → 
    a = 8 := 
by
  sorry

end find_a_given_integer_roots_l209_209791


namespace distinct_values_of_n_l209_209340

theorem distinct_values_of_n :
  let x1_x2_pairs := [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6),
                      (-1, -36), (-2, -18), (-3, -12), (-4, -9), (-6, -6)] in
  let n_values := x1_x2_pairs.map (λ p : Int × Int => p.1 + p.2) in
  n_values.toFinset.card = 10 :=
by
  sorry

end distinct_values_of_n_l209_209340


namespace odd_factors_of_360_l209_209232

theorem odd_factors_of_360 : ∃ n, 360 = 2^3 * 3^2 * 5^1 ∧ n = 6 :=
by 
  -- Define the prime factorization of 360
  let pf360 := (2^3 * 3^2 * 5^1)
  
  -- Define the count of odd factors by removing the contribution of factor 2.
  let odd_part := (3^2 * 5^1)
  
  -- Calculate the number of factors.
  have odd_factors_count_eq : (2 + 1) * (1 + 1) = 6 := by 
    calc (2 + 1) * (1 + 1) = 3 * 2 : by rfl
                         ... = 6 : by rfl
  
  -- Assert the existence of such n matching the count of odd factors.
  exact ⟨6, And.intro rfl odd_factors_count_eq⟩

end odd_factors_of_360_l209_209232


namespace matrix_addition_correct_l209_209889

def matrixA : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![0, 5]]
def matrixB : Matrix (Fin 2) (Fin 2) ℤ := ![![-6, 2], ![7, -10]]
def matrixC : Matrix (Fin 2) (Fin 2) ℤ := ![![-2, -1], ![7, -5]]

theorem matrix_addition_correct : matrixA + matrixB = matrixC := by
  sorry

end matrix_addition_correct_l209_209889


namespace cos_double_angle_l209_209268

theorem cos_double_angle (α : ℝ) (h : sin (π / 6 - α) = 1 / 3) : cos (2 * π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_double_angle_l209_209268


namespace max_value_expression_l209_209575

noncomputable def factorize_15000 := 2^3 * 3 * 5^4

theorem max_value_expression (x y : ℕ) (h1 : 6 * x^2 - 5 * x * y + y^2 = 0) (h2 : x ∣ factorize_15000) : 
  2 * x + 3 * y ≤ 60000 := sorry

end max_value_expression_l209_209575


namespace final_output_l209_209707

def input : ℕ := 15

def multiply_by_3 (x : ℕ) : ℕ := x * 3

def compare_with_20 (x : ℕ) : Bool := x > 20

def subtract_7 (x : ℕ) : ℕ := x - 7

theorem final_output : 
  let a := input in
  let b := multiply_by_3 a in
  if compare_with_20 b then
    subtract_7 b = 38
  else
    b = 38 := 
by
  sorry

end final_output_l209_209707


namespace solution_set_M_abs_ineq_l209_209157

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x - 2|

-- Define the set M
def M : Set ℝ := {x | 1 < x ∧ x < 4}

-- The first statement to prove the solution set M for the inequality
theorem solution_set_M : ∀ x, f x < 3 ↔ x ∈ M :=
by sorry

-- The second statement to prove the inequality when a, b ∈ M
theorem abs_ineq (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : |a + b| < |1 + ab| :=
by sorry

end solution_set_M_abs_ineq_l209_209157


namespace prop_p3_prop_p2_l209_209556

noncomputable section

-- Proposition p₃
theorem prop_p3 (x : ℝ) : x < -1 → log (1 / 2) (x^2 + 1) < -1 :=
by
  sorry

-- Proposition p₂
theorem prop_p2 (α β : ℝ) (h1 : 2 * sin (α - β) = 1) (h2 : 3 * sin (α + β) = 1) : 
  sin α * cos β = 5/12 :=
by
  sorry

end prop_p3_prop_p2_l209_209556


namespace mary_max_earnings_l209_209478

variable (hours : ℕ) (regular_hours : ℕ) (regular_rate : ℝ) (overtime_rate : ℝ)

def total_earnings (hours : ℕ) (regular_rate : ℝ) (overtime_rate : ℝ) : ℝ :=
  if hours ≤ regular_hours then
    hours * regular_rate
  else
    regular_hours * regular_rate + (hours - regular_hours) * overtime_rate

theorem mary_max_earnings :
  let hours := 80
  let regular_hours := 20
  let regular_rate := 8
  let overtime_rate := regular_rate * 1.25
  total_earnings hours regular_rate overtime_rate = 760 := by
  sorry

end mary_max_earnings_l209_209478


namespace water_added_to_mixture_l209_209848

theorem water_added_to_mixture :
  ∃ (W : ℝ), 
  (∃ (V : ℝ), V = 18 + W) ∧ 
  (∀ (A : ℝ), A = 0.20 * 18) ∧ 
  (∀ (P_new : ℝ), P_new = 17.14285714285715 / 100 → 
    A / V = P_new) →
  W = 3 :=
begin
  sorry
end

end water_added_to_mixture_l209_209848


namespace part_I_part_II_l209_209955

-- Definitions of the sets A, B, and C
def A : Set ℝ := { x | x ≤ -1 ∨ x ≥ 3 }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 6 }
def C (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m }

-- Proof statements
theorem part_I : A ∩ B = { x | 3 ≤ x ∧ x ≤ 6 } :=
by sorry

theorem part_II (m : ℝ) : (B ∪ C m = B) → (m ≤ 3) :=
by sorry

end part_I_part_II_l209_209955


namespace count_triangles_with_positive_area_l209_209242

theorem count_triangles_with_positive_area : 
  let points : list (ℕ × ℕ) := [(x, y) | x ← [1, 2, 3, 4, 5], y ← [1, 2, 3, 4, 5]]
  let collinear (a b c : ℕ × ℕ) : Prop := (b.1 - a.1) * (c.2 - a.2) = (b.2 - a.2) * (c.1 - a.1)
  let non_degenerate (a b c : ℕ × ℕ) : Prop := ¬collinear a b c
  list.filter (λ t, non_degenerate t[0] t[1] t[2]) (points.combinations 3) = 2156 := by
  sorry

end count_triangles_with_positive_area_l209_209242


namespace largest_triangle_perimeter_maximizes_l209_209524

theorem largest_triangle_perimeter_maximizes 
  (y : ℤ) 
  (h1 : 3 ≤ y) 
  (h2 : y < 16) : 
  (7 + 9 + y) = 31 ↔ y = 15 := 
by 
  sorry

end largest_triangle_perimeter_maximizes_l209_209524


namespace part1_part3_l209_209150

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - 1 / x

axiom f_deriv (a x : ℝ) (hx : 0 < x) : deriv (f a) x = a / x + 1 / (x^2)

theorem part1 (h : deriv (f a) 1 = 2) : a = 1 :=
by {
  rw [f_deriv a 1 (by norm_num)] at h,
  linarith,
}

theorem part3 (x : ℝ) (hx : 2 ≤ x) : 
  f 1 (x-1) ≤ 2 * x - 5 :=
by {
  let g : ℝ → ℝ := λ x, Real.log (x-1) - 1 / (x-1) - 2 * x + 5,
  have h_deriv : ∀ x, 2 < x → deriv g x < 0,
  {
    intros x hx,
    have der_g : deriv g x = (2*x-1)*(x-2) / ((x-1)^2),
    {
      simp [g],
      rw [deriv_sub, deriv_sub, deriv_const_mul, deriv_log, deriv_div'],
      field_simp [hx],
    },
    by_cases h_cases : x = 2,
    { exact false.elim (by linarith) },
    { rw [h_cases] at der_g,
      exact der_g,
      sorry
    }
  },
  specialize h_deriv x (by linarith),
  sorry,
}

end part1_part3_l209_209150


namespace odd_factors_of_360_l209_209230

theorem odd_factors_of_360 : ∃ n, 360 = 2^3 * 3^2 * 5^1 ∧ n = 6 :=
by 
  -- Define the prime factorization of 360
  let pf360 := (2^3 * 3^2 * 5^1)
  
  -- Define the count of odd factors by removing the contribution of factor 2.
  let odd_part := (3^2 * 5^1)
  
  -- Calculate the number of factors.
  have odd_factors_count_eq : (2 + 1) * (1 + 1) = 6 := by 
    calc (2 + 1) * (1 + 1) = 3 * 2 : by rfl
                         ... = 6 : by rfl
  
  -- Assert the existence of such n matching the count of odd factors.
  exact ⟨6, And.intro rfl odd_factors_count_eq⟩

end odd_factors_of_360_l209_209230


namespace relationship_ea_f0_fa_l209_209602

variable {ℝ : Type*} [Real ℝ]

-- Conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)
axiom f_deriv_exists : ∀ x : ℝ, ∃ f' : ℝ → ℝ, ∀ y : ℝ, differentiableAt ℝ) f' y   -- Existential quantifier for derivative
axiom f_prime_gt_f : ∀ x : ℝ, deriv f x > f(x) 
axiom a_gt_zero : a > 0

-- Statement to be proved
theorem relationship_ea_f0_fa (f_deriv_exists : ∀ x : ℝ, ∃ f' : ℝ → ℝ, ∀ y : ℝ, differentiableAt ℝ) f' y ) (f_prime_gt_f : ∀ x : ℝ, deriv f x > f(x)) (a_gt_zero : a > 0) : 
  e^a * f(0) < f(a) :=
sorry

end relationship_ea_f0_fa_l209_209602


namespace total_tagged_numbers_l209_209539

theorem total_tagged_numbers {W X Y Z : ℕ} 
  (hW : W = 200)
  (hX : X = W / 2)
  (hY : Y = W + X)
  (hZ : Z = 400) :
  W + X + Y + Z = 1000 := by
  sorry

end total_tagged_numbers_l209_209539


namespace passengers_with_round_trip_tickets_and_cars_l209_209758

theorem passengers_with_round_trip_tickets_and_cars (P : ℕ) :
  let percent_round_trip := 62.5 / 100
  let percent_no_cars := 60 / 100
  let percent_with_cars := 1 - percent_no_cars in
  percent_round_trip * percent_with_cars * 100 = 25 :=
by
  let percent_round_trip := 62.5 / 100
  let percent_no_cars := 60 / 100
  let percent_with_cars := 1 - percent_no_cars
  have h1 : percent_round_trip * percent_with_cars = 0.25, sorry
  exact h1

end passengers_with_round_trip_tickets_and_cars_l209_209758


namespace find_h_l209_209971

noncomputable def proof_h_value : ℝ → ℝ → ℝ → Prop :=
  λ p q h, (p + q = -h) ∧ (p * q = 8) ∧ ((p + 6) + (q + 6) = h)

theorem find_h (p q : ℝ) : ∃ h : ℝ, proof_h_value p q h ∧ h = 6 :=
by {
  sorry
}

end find_h_l209_209971


namespace transformation_of_vector_l209_209428

theorem transformation_of_vector (T : ℝ^3 → ℝ^3)
  (h_linear : ∀ (a b : ℝ) (v w : ℝ^3), T (a • v + b • w) = a • T v + b • T w)
  (h_cross : ∀ (v w : ℝ^3), T (v × w) = T v × T w)
  (h1 : T ⟨6, 6, 3⟩ = ⟨4, -1, 8⟩)
  (h2 : T ⟨-6, 3, 6⟩ = ⟨4, 8, -1⟩) : 
  T ⟨3, 9, 12⟩ = ⟨7, 8, 11⟩ := 
sorry

end transformation_of_vector_l209_209428


namespace samatha_route_count_l209_209779

theorem samatha_route_count :
  let house_sw_city_park := nat.choose 6 3,
  let diag_through_city_park := 1,
  let ne_city_park_school := nat.choose 6 3 in
  house_sw_city_park * diag_through_city_park * ne_city_park_school = 400 := by
sorry

end samatha_route_count_l209_209779


namespace line_plane_parallel_no_intersection_l209_209277

-- Definitions and conditions
variable (a : Type) (alpha : Type)
variable [Line a] [Plane alpha]

-- Parallelism definition
def parallel (a : a) (alpha : alpha) : Prop := 
  ∀ p : Point, (p ∈ a) → ¬(p ∈ alpha)

-- Theorem statement: If line 'a' is parallel to plane 'alpha', then line 'a' does not intersect with any line in plane 'alpha'.
theorem line_plane_parallel_no_intersection (a : a) (alpha : alpha) (H : parallel a alpha) :
  ∀ b : a, (b ∈ alpha) → ¬ ∃ p : Point, (p ∈ a) ∧ (p ∈ b) :=
sorry  -- Proof goes here

end line_plane_parallel_no_intersection_l209_209277


namespace S_3n_plus_1_l209_209136

noncomputable def S : ℕ → ℝ := sorry  -- S_n is the sum of the first n terms of the sequence {a_n}
noncomputable def a : ℕ → ℝ := sorry  -- Sequence {a_n}

-- Given conditions
axiom S3 : S 3 = 1
axiom S4 : S 4 = 11
axiom a_recurrence (n : ℕ) : a (n + 3) = 2 * a n

-- Define S_{3n+1} in terms of n
theorem S_3n_plus_1 (n : ℕ) : S (3 * n + 1) = 3 * 2^(n+1) - 1 :=
sorry

end S_3n_plus_1_l209_209136


namespace find_angle_between_plane_and_other_leg_l209_209298

-- Define the right triangle and the angles involved
variables (A B C D E : Type)
variables (α β γ : ℝ)

-- Define the conditions
axiom right_triangle (ABC : Triangle) : (∠ C = π / 2)
axiom hypotenuse (AB: Line) : (AB = hypotenuse ABC)
axiom plane_through_hypotenuse (π : Plane) : (π ∋ AB)
axiom angle_plane_triangle (α : ℝ) : (∠ (plane_of_triangle ABC) (π) = α)
axiom angle_plane_leg (β : ℝ) : (∠ (leg BC) (π) = β)

-- Define the goal
theorem find_angle_between_plane_and_other_leg :
  γ = arcsin (sqrt (sin (α + β) * sin (α - β))) :=
sorry

end find_angle_between_plane_and_other_leg_l209_209298


namespace largest_prime_factor_2525_l209_209452

theorem largest_prime_factor_2525 : 
  ∃ p : ℕ, prime p ∧ p ∣ 2525 ∧ ∀ q : ℕ, prime q ∧ q ∣ 2525 → q ≤ p :=
sorry

end largest_prime_factor_2525_l209_209452


namespace range_of_a_to_satisfy_l209_209746

def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else 2^x

theorem range_of_a_to_satisfy (a : ℝ) : 
  (f (f a) = 2^(f a)) → (a ≥ 2/3) :=
sorry

end range_of_a_to_satisfy_l209_209746


namespace solution_of_system_l209_209926

theorem solution_of_system :
  ∃ x y : ℝ, (x^4 + y^4 = 17) ∧ (x + y = 3) ∧ ((x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1)) :=
by
  sorry

end solution_of_system_l209_209926


namespace liquid_X_percentage_correct_l209_209060

noncomputable def percent_liquid_X_in_solution_A := 0.8 / 100
noncomputable def percent_liquid_X_in_solution_B := 1.8 / 100

noncomputable def weight_solution_A := 400.0
noncomputable def weight_solution_B := 700.0

noncomputable def weight_liquid_X_in_A := percent_liquid_X_in_solution_A * weight_solution_A
noncomputable def weight_liquid_X_in_B := percent_liquid_X_in_solution_B * weight_solution_B

noncomputable def total_weight_solution := weight_solution_A + weight_solution_B
noncomputable def total_weight_liquid_X := weight_liquid_X_in_A + weight_liquid_X_in_B

noncomputable def percent_liquid_X_in_mixed_solution := (total_weight_liquid_X / total_weight_solution) * 100

theorem liquid_X_percentage_correct :
  percent_liquid_X_in_mixed_solution = 1.44 :=
by
  sorry

end liquid_X_percentage_correct_l209_209060


namespace mike_total_investment_l209_209750

variable (T : ℝ)
variable (H1 : 0.09 * 1800 + 0.11 * (T - 1800) = 624)

theorem mike_total_investment : T = 6000 :=
by
  sorry

end mike_total_investment_l209_209750


namespace binomial_coeff_x5y3_in_expansion_eq_56_l209_209096

theorem binomial_coeff_x5y3_in_expansion_eq_56:
  let n := 8
  let k := 3
  let binom_coeff := Nat.choose n k
  binom_coeff = 56 := 
by sorry

end binomial_coeff_x5y3_in_expansion_eq_56_l209_209096


namespace minimize_quadratic_l209_209454

theorem minimize_quadratic (y : ℝ) : 
  ∃ m, m = 3 * y ^ 2 - 18 * y + 11 ∧ 
       (∀ z : ℝ, 3 * z ^ 2 - 18 * z + 11 ≥ m) ∧ 
       m = -16 := 
sorry

end minimize_quadratic_l209_209454


namespace prove_arithmetic_seq_find_a_formula_find_T_sum_l209_209131

variable {n : ℕ} (a : ℕ → ℝ)

-- Given conditions
def a_seq (n : ℕ) :=
  ∀ n, a 1 = 2 / 5 ∧ (2 * a n - 2 * a (n + 1) = 3 * a n * a (n + 1))

-- Define the arithmetic sequence
def arithmetic_seq : Prop :=
  ∀ n, (2 / a (n + 1)) - (2 / a n) = 3

-- General formula for a_n
def a_formula : Prop :=
  ∀ n, a n = 2 / (3 * n + 2)

-- Define c_n based on a_n
def c (n : ℕ) : ℝ :=
  2^n / (a n)

-- Sum of the first n terms of the sequence {c_n}
def T (n : ℕ) : ℝ :=
  (3 * n - 1) * 2^n + 1

theorem prove_arithmetic_seq
  (h : a_seq a) :
  arithmetic_seq a :=
sorry

theorem find_a_formula
  (h : a_seq a) :
  a_formula a :=
sorry

theorem find_T_sum
  (h : a_seq a) :
  ∀ n, (∑ i in finset.range n, c a i) = T n :=
sorry

end prove_arithmetic_seq_find_a_formula_find_T_sum_l209_209131


namespace boolean_function_structures_l209_209590

noncomputable def distinctBooleanStructures : Nat :=
  let transformations := [(λ (x1 x2 x3 : Bool), (x1, x2, x3)),
                         (λ (x1 x2 x3 : Bool), (x1 && x2 && x3)),
                         (λ (x1 x2 x3 : Bool), (x3, x2, x1)),
                         (λ (x1 x2 x3 : Bool), (x1, x2 && x3)),
                         (λ (x1 x2 x3 : Bool), (x2, x1 && x3)),
                         (λ (x1 x2 x3 : Bool), (x3, x1 && x2))]
  let states := [(false, false, false), (false, false, true),
                 (false, true, false), (false, true, true),
                 (true, false, false), (true, false, true),
                 (true, true, false), (true, true, true)]
  let permutations := transformations.map (λ h, states.map h)
  let cycles (p : List ((Bool × Bool × Bool) × (Bool × Bool × Bool))) : Nat := sorry
  let polya := (1 / 6 : ℚ) * (2 ^ 8 + 3 * 2 ^ 6 + 2 * 2 ^ 4)
  polya

theorem boolean_function_structures : distinctBooleanStructures = 80 :=
sorry

end boolean_function_structures_l209_209590


namespace probability_prime_or_odd_ball_l209_209912

def isPrime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def isOdd (n : ℕ) : Prop :=
  n % 2 = 1

def isPrimeOrOdd (n : ℕ) : Prop :=
  isPrime n ∨ isOdd n

theorem probability_prime_or_odd_ball :
  (1+2+3+5+7)/8 = 5/8 := by
  sorry

end probability_prime_or_odd_ball_l209_209912


namespace gcd_13924_27018_l209_209588

theorem gcd_13924_27018 : Int.gcd 13924 27018 = 2 := 
  by
    sorry

end gcd_13924_27018_l209_209588


namespace pirate_treasure_probability_l209_209512

theorem pirate_treasure_probability :
  let p_treasure_no_traps := 1 / 3
  let p_traps_no_treasure := 1 / 6
  let p_neither := 1 / 2
  let choose_4_out_of_8 := 70
  let p_4_treasure_no_traps := (1 / 3) ^ 4
  let p_4_neither := (1 / 2) ^ 4
  choose_4_out_of_8 * p_4_treasure_no_traps * p_4_neither = 35 / 648 :=
by
  sorry

end pirate_treasure_probability_l209_209512


namespace sum_f_positive_l209_209156

variable (a b c : ℝ)

def f (x : ℝ) := x^3 + x

theorem sum_f_positive (h1 : a + b > 0) (h2 : a + c > 0) (h3 : b + c > 0) :
  f a + f b + f c > 0 :=
sorry

end sum_f_positive_l209_209156


namespace max_radius_squared_l209_209435

/--
Given three congruent right circular cones each with a base radius of 4 and height 10, 
having their axes intersecting at right angles at a point 5 units from the base,
prove that the radius of a sphere that fits snugly within the cones has \(r^2 = \frac{100}{29}\),
and that the sum of the coprime integers \(m\) and \(n\) resulting in this fraction is 129.
-/
theorem max_radius_squared (r : ℝ) (m n : ℕ) 
  (rad_coprime : Int.gcd m n = 1) 
  (h_base_radius : ∀ (cone : ℕ), cone <= 3 → base_radius cone = 4)
  (h_height : ∀ (cone : ℕ), cone <= 3 → height cone = 10)
  (h_intersection : ∀ (cone : ℕ), cone <= 3 → axes_intersect_height cone = 5)
  (h_fits_snugly : fits_snugly_within_cones r): 
  r^2 = (100 : ℝ) / (29 : ℝ) ∧ m + n = 129 :=
by
  sorry

end max_radius_squared_l209_209435


namespace area_rescaling_ratio_l209_209284

theorem area_rescaling_ratio (s : ℝ) : 
  let A_original := s^2 in
  let A_resultant := (10 * s)^2 in
  A_resultant / A_original = 100 := 
by
  sorry

end area_rescaling_ratio_l209_209284


namespace total_amount_paid_l209_209721

def cost_of_nikes : ℝ := 150
def cost_of_work_boots : ℝ := 120
def tax_rate : ℝ := 0.1

theorem total_amount_paid :
  let total_cost := cost_of_nikes + cost_of_work_boots in
  let tax := total_cost * tax_rate in
  let total_paid := total_cost + tax in
  total_paid = 297 := by
  let total_cost := cost_of_nikes + cost_of_work_boots
  let tax := total_cost * tax_rate
  let total_paid := total_cost + tax
  sorry

end total_amount_paid_l209_209721


namespace angle_MAC_90_degrees_l209_209302

open EuclideanGeometry

noncomputable theory
open_locale classical

variables {A B C D E F G P Q M : Point}

/-- Convex cyclic quadrilateral ABCD with intersections as described, prove ∠MAC = 90°. -/
theorem angle_MAC_90_degrees
  (h1 : convex_cyclic_quadrilateral A B C D)
  (h2 : intersection_point AC BD E)
  (h3 : intersection_point AB CD F)
  (h4 : intersection_point BC DA G)
  (h5 : circumcircle_intersect A B E CB P)
  (h6 : circumcircle_intersect A D E CD Q)
  (h7 : collinear [C, B, P, G])
  (h8 : collinear [C, Q, D, F])
  (h9 : intersection_point FP GQ M) :
  ∠MAC = 90° :=
sorry

end angle_MAC_90_degrees_l209_209302


namespace max_lambda_inequality_l209_209922

theorem max_lambda_inequality :
  ∃ λ : ℝ, λ = 3/8 ∧ 
  (∀ (n : ℕ) (x : ℕ → ℝ), n > 0 → 
    (0 = x 0 ∧ (∀ k, 0 < k ∧ k ≤ n → x k ≤ x (k + 1)) ∧ x n = 1) → 
    ∑ k in finset.range n, (x k) ^ 3 * (x k - x (k - 1)) ≥ 1 / 4 + λ / n) :=
begin
  sorry
end

end max_lambda_inequality_l209_209922


namespace domain_of_function_l209_209407

section
variable (x : ℝ)

def condition_1 := x + 4 ≥ 0
def condition_2 := x + 2 ≠ 0
def domain := { x : ℝ | x ≥ -4 ∧ x ≠ -2 }

theorem domain_of_function : (condition_1 x ∧ condition_2 x) ↔ (x ∈ domain) :=
by
  sorry
end

end domain_of_function_l209_209407


namespace exists_x_satisfying_inequality_l209_209714

def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem exists_x_satisfying_inequality :
  ∃ x : ℝ, (x > 1) ∧ (¬(∃ n : ℤ, x = n)) ∧
  (finset.sum (finset.range 50) (λ k, fractional_part (x^(2 * k + 1))) < 1 / 2^99) :=
sorry

end exists_x_satisfying_inequality_l209_209714


namespace Nicki_runs_30_miles_per_week_in_second_half_l209_209753

/-
  Nicki ran 20 miles per week for the first half of the year.
  There are 26 weeks in each half of the year.
  She ran a total of 1300 miles for the year.
  Prove that Nicki ran 30 miles per week in the second half of the year.
-/

theorem Nicki_runs_30_miles_per_week_in_second_half (weekly_first_half : ℕ) (weeks_per_half : ℕ) (total_miles : ℕ) :
  weekly_first_half = 20 → weeks_per_half = 26 → total_miles = 1300 → 
  (total_miles - (weekly_first_half * weeks_per_half)) / weeks_per_half = 30 :=
by
  intros h1 h2 h3
  sorry

end Nicki_runs_30_miles_per_week_in_second_half_l209_209753


namespace find_length_of_second_tract_l209_209661

theorem find_length_of_second_tract :
  ∀ (length2 : ℕ), 
    let area1 := 300 * 500,
        combined_area := 307500,
        width2 := 630
    in length2 * width2 = combined_area - area1 → length2 = 250 :=
by
  intros length2 h
  sorry

end find_length_of_second_tract_l209_209661


namespace smallest_abs_diff_of_powers_l209_209595

open Nat

theorem smallest_abs_diff_of_powers :
  ∃ (m n : ℕ), abs (36 ^ m - 5 ^ n) = 11 :=
sorry

end smallest_abs_diff_of_powers_l209_209595


namespace problem_U_complement_eq_l209_209992

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l209_209992


namespace probability_exists_x0_l209_209324

-- Define the conditions as given
noncomputable def h (n : ℕ) (θ : fin n → ℝ) (x : ℝ) : ℝ :=
  (1 / n : ℝ) * ((finset.univ.filter (λ k, θ k < x)).card : ℝ)

-- Define the main Theorem
theorem probability_exists_x0 (n : ℕ) (θ : fin n → ℝ) 
  (h_uniform : ∀ i, 0 <= θ i ∧ θ i <= 1) 
  (h_independent : ∀ i j, i ≠ j → θ i ≠ θ j)
  : prob (∃ x0 ∈ (0, 1), h n θ x0 = x0) = 1 - (1 / n : ℝ) := 
sorry

end probability_exists_x0_l209_209324


namespace odd_factors_360_l209_209179

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l209_209179


namespace first_term_is_720_sqrt7_div_49_l209_209412

noncomputable def first_term_geometric_sequence (a r : ℝ) : Prop :=
  (a * r^3 = 720) ∧ (a * r^5 = 5040) ∧ (a = 720 * real.sqrt 7 / 49)

-- The theorem statement
theorem first_term_is_720_sqrt7_div_49 :
  ∃ a r : ℝ, first_term_geometric_sequence a r :=
begin
  use [720 * real.sqrt 7 / 49, real.sqrt 7],
  unfold first_term_geometric_sequence,
  split,
  { -- Prove a * r^3 = 720
    ring },
  split,
  { -- Prove a * r^5 = 5040
    ring },
  { -- Prove a = 720 * real.sqrt 7 / 49
    ring }
end

end first_term_is_720_sqrt7_div_49_l209_209412


namespace school_students_unique_l209_209299

theorem school_students_unique 
  (n : ℕ)
  (h1 : 70 < n) 
  (h2 : n < 130) 
  (h3 : n % 4 = 2) 
  (h4 : n % 5 = 2)
  (h5 : n % 6 = 2) : 
  (n = 92 ∨ n = 122) :=
  sorry

end school_students_unique_l209_209299


namespace prop1_prop2_correct_propositions_l209_209941

variables (α β : Set Line) (m n : Line)

-- Definition of perpendicular and parallel planes and lines.
def is_perpendicular (a b : Set Line) := sorry
def is_parallel (a b : Set Line) := sorry

-- Conditions:
axiom m_perp_alpha : is_perpendicular m α
axiom m_in_beta : m ⊆ β
axiom alpha_cap_beta_eq_m : α ∩ β = m
axiom n_parallel_m : is_parallel n m
axiom n_in_alpha : n ⊆ α
axiom n_in_beta : n ⊆ β

theorem prop1 : is_perpendicular m α ∧ m ⊆ β → is_perpendicular α β := sorry
theorem prop2 : (α ∩ β = m) ∧ is_parallel n m ∧ n ⊆ α ∧ n ⊆ β → is_parallel n α ∧ is_parallel n β := sorry

theorem correct_propositions : (prop1 ∧ prop2) := sorry

end prop1_prop2_correct_propositions_l209_209941


namespace odd_factors_360_l209_209185

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l209_209185


namespace distance_to_origin_l209_209304

open Real

theorem distance_to_origin :
  let P := (-sqrt 3, 1) in
  let O := (0, 0) in
  let d := sqrt ((O.1 - P.1)^2 + (O.2 - P.2)^2) in
  d = 2 := 
by
  let P := (-sqrt 3, 1)
  let O := (0, 0)
  let d := sqrt ((O.1 - P.1)^2 + (O.2 - P.2)^2)
  have h1 : (O.1 - P.1)^2 = (sqrt 3)^2 := by sorry
  have h2 : (O.2 - P.2)^2 = 1^2 := by sorry
  have h3 : sqrt ((sqrt 3)^2 + 1) = sqrt 4 := by sorry
  have h4 : sqrt 4 = 2 := by sorry
  sorry

end distance_to_origin_l209_209304


namespace alice_is_10_years_older_l209_209530

-- Problem definitions
variables (A B : ℕ)

-- Conditions of the problem
def condition1 := A + 5 = 19
def condition2 := A + 6 = 2 * (B + 6)

-- Question to prove
theorem alice_is_10_years_older (h1 : condition1 A) (h2 : condition2 A B) : A - B = 10 := 
by
  sorry

end alice_is_10_years_older_l209_209530


namespace tangent_lines_variety_count_l209_209297

-- Definition of the problem conditions
def radius1 := 4
def radius2 := 5

/-- Definition of the problem statement - 
    Prove that there are exactly 5 different values
    of the number of tangent lines (k) possible as 
    the distance (d) between the centers of the two circles changes --/
theorem tangent_lines_variety_count (d : ℝ) : 
    ∃ k_set : set ℕ, k_set = {0, 1, 2, 3, 4} ∧ (∀ d, d = 0 → 0 ∈ k_set) ∧ (∀ d, d = 1 → 1 ∈ k_set) ∧ 
    (∀ d, 1 < d ∧ d < 9 → 2 ∈ k_set) ∧ (∀ d, d = 9 → 3 ∈ k_set) ∧ (∀ d, d > 9 → 4 ∈ k_set) ∧ 
    fintype.card k_set = 5 :=
begin
  -- Skeleton of the proof
  sorry
end

end tangent_lines_variety_count_l209_209297


namespace hyperbola_focus_larger_y_l209_209800

open Real

theorem hyperbola_focus_larger_y :
  ∃ (F : ℝ × ℝ), ((F = (5, 10 + sqrt 58)) ∧
    (∀ (x y : ℝ), ((x - 5) ^ 2 / 49 - (y - 10) ^ 2 / 9 = 1) →
      y ≤ ((F.2)))) :=
begin
  -- Proof will be completed here
  sorry,
end

end hyperbola_focus_larger_y_l209_209800


namespace log_sum_geometric_l209_209932

variable {a : ℕ → ℝ}

-- Geometric sequence conditions
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given conditions
def pos_terms (a : ℕ → ℝ) := ∀ n, 0 < a n
def specific_condition (a : ℕ → ℝ) := a 5 * a 6 = 8

-- Proving the main statement
theorem log_sum_geometric (h1 : is_geometric_sequence a) (h2 : pos_terms a) (h3 : specific_condition a) :
  (∑ i in Finset.range 10, Real.log (a i) / Real.log 2) = 15 := 
sorry

end log_sum_geometric_l209_209932


namespace volume_tetrahedron_proof_l209_209871

noncomputable def volume_tetrahedron (s : ℝ) : ℝ :=
  let height := √((2 * s) ^ 2 - (2 * s / 3) ^ 2)
  let base_area := (√3 / 4) * s ^ 2
  (1 / 3) * base_area * height

theorem volume_tetrahedron_proof : volume_tetrahedron (3 * √3) = 81 * √2 := by
  sorry

end volume_tetrahedron_proof_l209_209871


namespace pencils_remaining_l209_209535

variable (initial_pencils : ℝ) (pencils_given : ℝ)

theorem pencils_remaining (h1 : initial_pencils = 56.0) 
                          (h2 : pencils_given = 9.5) 
                          : initial_pencils - pencils_given = 46.5 :=
by 
  sorry

end pencils_remaining_l209_209535


namespace proposition_equivalence_l209_209397

def f (x : ℝ) : ℝ :=
  if Rational x then 1 else 0

theorem proposition_equivalence : ¬ (∀ x, f (f x) = 0) ∧
  (∀ x, f x = f (-x)) ∧ 
  (∀ (x : ℝ) (T : ℚ), T ≠ 0 → f (x + T) = f x) ∧ 
  (∃ x1 x2 x3 : ℝ, 
    f x1 = 0 ∧ 
    f x2 = 1 ∧
    f x3 = 0 ∧ 
    (x1 = - (Real.sqrt 3 / 3)) ∧ 
    (x2 = 0) ∧ 
    (x3 = (Real.sqrt 3 / 3)) ∧
    (triangle_is_equilateral ⟨x1, f x1⟩ ⟨x2, f x2⟩ ⟨x3, f x3⟩))
    :=
by
  sorry

end proposition_equivalence_l209_209397


namespace range_of_k_tan_alpha_l209_209659

noncomputable def f (x k : Real) : Real := Real.sin x + k

theorem range_of_k (k : Real) : 
  (∃ x : Real, f x k = 1) ↔ (0 ≤ k ∧ k ≤ 2) :=
sorry

theorem tan_alpha (α k : Real) (h : α ∈ Set.Ioo (0 : Real) Real.pi) (hf : f α k = 1 / 3 + k) : 
  Real.tan α = Real.sqrt 2 / 4 :=
sorry

end range_of_k_tan_alpha_l209_209659


namespace curve_equation_and_max_distance_l209_209978

-- Define the polar equation and convert it to Cartesian form
def rho (θ : ℝ) : ℝ := 2 * Real.cos θ

def parametric_curve_C (θ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos θ, Real.sin θ)

def line_l (t : ℝ) : ℝ × ℝ :=
  (-2 + 4 * t, 3 * t)

noncomputable def point_to_line_distance (x y a b c : ℝ) : ℝ :=
  (|a * x + b * y + c|) / (Real.sqrt (a ^ 2 + b ^ 2))

theorem curve_equation_and_max_distance :
  (∀ θ, (parametric_curve_C θ).fst - 1)^2 + (parametric_curve_C θ).snd^2 = 1 ∧
  (∀ t, 3 * (line_l t).fst - 4 * (line_l t).snd + 6 = 0) ∧
  (∀ θ, point_to_line_distance (parametric_curve_C θ).fst (parametric_curve_C θ).snd 3 (-4) 6 ≤ 14 / 5) :=
  by
  -- The statement encapsulates the formal problem based on the identified conditions and correct answers.
  sorry

end curve_equation_and_max_distance_l209_209978


namespace range_F_l209_209455

noncomputable def F (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_F : set.range F = set.Ici 4 :=
by
  sorry

end range_F_l209_209455


namespace find_k_l209_209097

variable {α : ℝ}

theorem find_k (k : ℝ) (h : (sin α + csc α)^2 + (cos α + sec α)^2 + 4 * sin α * cos α = k + tan α^2 + cot α^2 + 2) :
  k = 7 + 2 * sin (2 * α) :=
by
  sorry

end find_k_l209_209097


namespace ant_reaches_bottom_vertex_probability_l209_209287

def ant_on_cube_probability : ℚ := 1 / 36

theorem ant_reaches_bottom_vertex_probability :
  let T := 1,
      A := 4,  -- 4 vertices adjacent to T
      B := 3,  -- Each Ai has 3 vertices (excluding T)
      C := 3,  -- Each Bi has 3 vertices (including the bottom one in total)
  ((1 / A) * (1 / B) * (1 / C) = ant_on_cube_probability) :=
by
  have A_prob : (1 / A) = 1 / 4 := by simp [A, four_div],
  have B_prob : (1 / B) = 1 / 3 := by simp [B, three_div],
  have C_prob : (1 / C) = 1 / 3 := by simp [C, three_div],
  rw [A_prob, B_prob, C_prob],
  norm_num

lemma four_div : 4 = (4:ℚ) := by norm_num

lemma three_div : 3 = (3:ℚ) := by norm_num


end ant_reaches_bottom_vertex_probability_l209_209287


namespace prove_parabola_equation_l209_209144

noncomputable def parabola_equation (p : ℝ) : Prop :=
  ∃ (A : ℝ×ℝ) (B C : ℝ×ℝ),
    (A.snd^2 = 2 * p * A.fst) ∧
    (dist (0, p / 2) A = dist B C) ∧
    (B = (0, -p / sqrt 3)) ∧
    (C = (0, p / sqrt 3)) ∧
    (dist B C = dist B (0, p / 2)) ∧
    (area_of_triangle (0, p / 2) B C = sqrt 3 / 4 * (dist B C)^2) ∧
    (1 / 2 * dist B C * (dist B C) = 128 / 3)

-- The theorem we are asked to prove
theorem prove_parabola_equation : 
  parabola_equation 8 :=
sorry

end prove_parabola_equation_l209_209144


namespace moving_point_trajectory_l209_209134

noncomputable theory

open_locale classical

variables {O A B C P : Type} [metric_space P]
variables {vec : P → P → Type}
variables [has_add (vec O B)] [has_div (vec O B)]
variables [has_mul ℝ (vec A B)] [has_mul ℝ (vec A C)]
variables [has_scalar ℝ (vec A B)] [has_sub (vec O B)]
variables [has_scalar ℝ (vec A C)] [inner_product_space ℝ (vec A B)]
variables [normalize_vec (vec A B)] [normalize_vec (vec A C)]
variables (λ : ℝ)

def is_perpendicular (u v : vec O B) : Prop := (u ⬝ v = 0)

def is_circumcenter (P : P) : Prop := ∀ (B C : P),
  dist P B = dist P C ∧
  ∀ (Q : P), (dist P B = dist Q B) ∧ (dist P C = dist Q C) → P = Q

theorem moving_point_trajectory 
  (O A B C : P) 
  (H_non_collinear : ¬ collinear O A B C) 
  (H_moving_point : ∀ (P : P),
    vector_between O P = 
    (vector_between O B + vector_between O C) / 2 + 
    λ * 
    ((normalize_vec (vector_between A B)) / cos_angle B + 
    (normalize_vec (vector_between A C)) / cos_angle C)) :
  ∃ (D : P), is_midpoint D B C ∧ 
  ∃ (P : P), is_on_perpendicular_bisector P B C → is_circumcenter P :=
sorry

end moving_point_trajectory_l209_209134


namespace family_satisfies_conditions_l209_209445

structure Person :=
(name : String)

structure Family :=
(G1 G2 H1 H2 W1 W2 : Person)
(relations :
  (G1 ≠ G2 ∧ H1 ≠ H2 ∧ W1 ≠ W2 ∧
  (∃ (S1 S2: Person),
    S1 ≠ S2 ∧
    (relationship G1 S1 ∧ relationship G2 S2 ∧
    relationship H1 W1 ∧ relationship H2 W2 ∧
    (∃ D1 D2: Person, D1 ≠ D2 ∧ relationship H1 D1 ∧ relationship H2 D2) ∧
    (relationship H1 H2 ∧ relationship W1 W2) ∧
    (relationship G1 W1 ∧ relationship G2 W2 ∧ 
    (∃ M1 M2 : Person, M1 ≠ M2 ∧ relationship W1 S1 ∧ relationship W1 S2) ∧
    (∃ YW1 YW2 : Person, YW1 ≠ YW2 ∧ relationship YW1 G1 ∧ relationship YW2 G2)))))

-- Relationship function to define how two persons relate to each other
def relationship (p1 p2 : Person) : Prop := sorry  -- The exact nature of relationships between persons

theorem family_satisfies_conditions :
  ∃ (f : Family),
    (∃ (S1 S2: Person),
      relationship f.G1 S1 ∧ relationship f.G2 S2 ∧
      relationship f.H1 f.W1 ∧ relationship f.H2 f.W2 ∧
      (∃ D1 D2: Person, D1 ≠ D2 ∧ relationship f.H1 D1 ∧ relationship f.H2 D2) ∧
      (relationship f.H1 f.H2 ∧ relationship f.W1 f.W2) ∧
      (relationship f.G1 f.W1 ∧ relationship f.G2 f.W2 ∧ 
      (∃ M1 M2 : Person, M1 ≠ M2 ∧ relationship f.W1 S1 ∧ relationship f.W2 S2) ∧
      (∃ YW1 YW2 : Person, YW1 ≠ YW2 ∧ relationship YW1 f.G1 ∧ relationship YW2 f.G2))) :=
sorry

end family_satisfies_conditions_l209_209445


namespace booklet_cost_l209_209392

theorem booklet_cost (b : ℝ) : 
  (10 * b < 15) ∧ (12 * b > 17) → b = 1.42 := by
  sorry

end booklet_cost_l209_209392


namespace math_olympiad_scores_l209_209784

theorem math_olympiad_scores 
  (correct_points : ℕ := 7)
  (incorrect_deduction : ℕ := 2)
  (unanswered_points : ℕ := 2)
  (total_problems : ℕ := 30)
  (attempted_problems : ℕ := 25)
  (unanswered_problems : ℕ := 5)
  (required_score : ℕ := 120) :
  ∃ x : ℕ, x ≥ 18 ∧ 
  let attempted_score := 7 * x - 2 * (25 - x) in 
  let total_score := attempted_score + 2 * unanswered_problems in
  total_score ≥ required_score := by
  sorry

end math_olympiad_scores_l209_209784


namespace no_sum_of_95_cents_l209_209930

-- Definitions based on the conditions
def is_valid_sum (cents : ℕ) : Bool :=
  let values := [1, 5, 10, 50]
  let possible_sums := list.sum <$> list.filter (λ l, l.length = 5) (list.permutations values)
  possible_sums.contains cents

-- Theorem based on the question and correct answer
theorem no_sum_of_95_cents : ¬ is_valid_sum 95 :=
by
  sorry

end no_sum_of_95_cents_l209_209930


namespace find_solution_l209_209581

theorem find_solution (x y z : ℝ) :
  (x * (y^2 + z) = z * (z + x * y)) ∧ 
  (y * (z^2 + x) = x * (x + y * z)) ∧ 
  (z * (x^2 + y) = y * (y + x * z)) → 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end find_solution_l209_209581


namespace weighted_average_fish_caught_l209_209567

-- Define the daily catches for each person
def AangCatches := [5, 7, 9]
def SokkaCatches := [8, 5, 6]
def TophCatches := [10, 12, 8]
def ZukoCatches := [6, 7, 10]

-- Define the group catches
def GroupCatches := AangCatches ++ SokkaCatches ++ TophCatches ++ ZukoCatches

-- Calculate the total number of fish caught by the group
def TotalFishCaught := List.sum GroupCatches

-- Calculate the total number of days fished by the group
def TotalDaysFished := 4 * 3

-- Calculate the weighted average
def WeightedAverage := TotalFishCaught.toFloat / TotalDaysFished.toFloat

-- Proof statement
theorem weighted_average_fish_caught :
  WeightedAverage = 7.75 := by
  sorry

end weighted_average_fish_caught_l209_209567


namespace correct_equation_B_l209_209045

theorem correct_equation_B (x : ℝ) : 
  ¬ (abs 4 = 2 ∨ abs 4 = -2) ∧ 
  (-(abs 2))^2 = 2 ∧ 
  (abs 3 / 2)^3 ≠ -2 ∧ 
  abs ((-2)^2) ≠ -2 := 
by sorry

end correct_equation_B_l209_209045


namespace inequality_always_holds_l209_209966

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 :=
sorry

end inequality_always_holds_l209_209966


namespace count_5_safe_7_safe_11_safe_le_5000_l209_209933

def p_safe (n p : ℕ) : Prop :=
  n % p ≠ 0 ∧ n % p ≠ 1 ∧ n % p ≠ 2 ∧ n % p ≠ p - 2 ∧ n % p ≠ p - 1

def safe_count (max_n : ℕ) : ℕ :=
  let safe_residues (p : ℕ) : Finset ℕ := (Finset.range p).filter (λ k, p_safe k p)
  let m := 5 * 7 * 11
  let total_safe_residues := (safe_residues 5).card * (safe_residues 7).card * (safe_residues 11).card
  let full_sets := max_n / m
  let remainder_safe := (Finset.range (max_n % m + 1)).filter (fun n => p_safe n 5 ∧ p_safe n 7 ∧ p_safe n 11)
  full_sets * total_safe_residues + remainder_safe.card

theorem count_5_safe_7_safe_11_safe_le_5000 : safe_count 5000 = 312 := by
  sorry

end count_5_safe_7_safe_11_safe_le_5000_l209_209933


namespace julia_cakes_remaining_l209_209695

namespace CakeProblem

def cakes_per_day : ℕ := 5 - 1
def days_baked : ℕ := 6
def total_cakes_baked : ℕ := cakes_per_day * days_baked
def days_clifford_eats : ℕ := days_baked / 2
def cakes_eaten_by_clifford : ℕ := days_clifford_eats

theorem julia_cakes_remaining : total_cakes_baked - cakes_eaten_by_clifford = 21 :=
by
  -- proof goes here
  sorry

end CakeProblem

end julia_cakes_remaining_l209_209695


namespace count_odd_factors_of_360_l209_209218

-- Defining the prime factorization of 360
def prime_factorization_360 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

-- Defining the exponent range based on prime factorization for an odd factor
def exponent_ranges : List (ℕ × List ℕ) := [(3, [0, 1, 2]), (5, [0, 1])]

theorem count_odd_factors_of_360 : ∃ n : ℕ, n = 6 :=
by
  -- Conditions based on prime factorization
  have h₁ : prime_factorization_360 = [(2, 3), (3, 2), (5, 1)], from rfl
  have h₂ : exponent_ranges = [(3, [0, 1, 2]), (5, [0, 1])], from rfl
  
  -- Definitions should align with the conditions in the problem
  use 6
  sorry

end count_odd_factors_of_360_l209_209218


namespace probability_second_yellow_l209_209058

theorem probability_second_yellow :
  let bagA_white := 3
  let bagA_black := 4
  let bagB_yellow := 6
  let bagB_blue := 4
  let bagC_yellow := 2
  let bagC_blue := 5
  (bagA_white + bagA_black = 7) →
  (bagB_yellow + bagB_blue = 10) →
  (bagC_yellow + bagC_blue = 7) →
  let P_WA := bagA_white / (bagA_white + bagA_black)
  let P_YB_WA := bagB_yellow / (bagB_yellow + bagB_blue)
  let P_WA_YB := P_WA * P_YB_WA
  let P_BA := bagA_black / (bagA_white + bagA_black)
  let P_YC_BA := bagC_yellow / (bagC_yellow + bagC_blue)
  let P_BA_YC := P_BA * P_YC_BA
  P_WA_YB + P_BA_YC = 103 / 245 := by
  intros,
  sorry

end probability_second_yellow_l209_209058


namespace total_spent_after_discount_and_tax_l209_209863

-- Define prices for each item
def price_bracelet := 4
def price_keychain := 5
def price_coloring_book := 3
def price_sticker := 1
def price_toy_car := 6

-- Define discounts and tax rates
def discount_bracelet := 0.10
def sales_tax := 0.05

-- Define the quantity of each item purchased by Paula, Olive, and Nathan
def quantity_paula_bracelets := 3
def quantity_paula_keychains := 2
def quantity_paula_coloring_books := 1
def quantity_paula_stickers := 4

def quantity_olive_coloring_books := 1
def quantity_olive_bracelets := 2
def quantity_olive_toy_cars := 1
def quantity_olive_stickers := 3

def quantity_nathan_toy_cars := 4
def quantity_nathan_stickers := 5
def quantity_nathan_keychains := 1

-- Function to calculate total cost before discount and tax
def total_cost_before_discount_and_tax (bracelets keychains coloring_books stickers toy_cars : Nat) : Float :=
  Float.ofNat (bracelets * price_bracelet) +
  Float.ofNat (keychains * price_keychain) +
  Float.ofNat (coloring_books * price_coloring_book) +
  Float.ofNat (stickers * price_sticker) +
  Float.ofNat (toy_cars * price_toy_car)

-- Function to calculate discount on bracelets
def bracelet_discount (bracelets : Nat) : Float :=
  Float.ofNat (bracelets * price_bracelet) * discount_bracelet

-- Function to calculate total cost after discount and before tax
def total_cost_after_discount (total_cost discount : Float) : Float :=
  total_cost - discount

-- Function to calculate total cost after tax
def total_cost_after_tax (total_cost : Float) (tax_rate : Float) : Float :=
  total_cost * (1 + tax_rate)

-- Proof statement (no proof provided, only the statement)
theorem total_spent_after_discount_and_tax : 
  total_cost_after_tax (
    total_cost_after_discount
      (total_cost_before_discount_and_tax quantity_paula_bracelets quantity_paula_keychains quantity_paula_coloring_books quantity_paula_stickers 0)
      (bracelet_discount quantity_paula_bracelets)
    +
    total_cost_after_discount
      (total_cost_before_discount_and_tax quantity_olive_bracelets 0 quantity_olive_coloring_books quantity_olive_stickers quantity_olive_toy_cars)
      (bracelet_discount quantity_olive_bracelets)
    +
    total_cost_before_discount_and_tax 0 quantity_nathan_keychains 0 quantity_nathan_stickers quantity_nathan_toy_cars
  ) sales_tax = 85.05 := 
sorry

end total_spent_after_discount_and_tax_l209_209863


namespace dog_food_bags_count_l209_209511

-- Define the constants based on the problem statement
def CatFoodBags := 327
def DogFoodMore := 273

-- Define the total number of dog food bags based on the given conditions
def DogFoodBags : ℤ := CatFoodBags + DogFoodMore

-- State the theorem we want to prove
theorem dog_food_bags_count : DogFoodBags = 600 := by
  sorry

end dog_food_bags_count_l209_209511


namespace eval_floor_4_7_l209_209572

theorem eval_floor_4_7 : Int.floor 4.7 = 4 := 
by 
    sorry

end eval_floor_4_7_l209_209572


namespace correct_statement_l209_209999

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l209_209999


namespace octahedron_coloring_probability_l209_209534

open Finset

/-- Prove the probability of positioning the octahedron so that all visible faces are the same color -/
theorem octahedron_coloring_probability : 
  let total_arrangements := 3^8 in
  let valid_arrangements := (3 + 3 * (choose 8 3) * 2) in
  (valid_arrangements : ℚ) / total_arrangements = 507 / 6561
:= by
  sorry

end octahedron_coloring_probability_l209_209534


namespace range_of_m_l209_209104

theorem range_of_m (m : ℝ) : 
  ((∀ x : ℝ, (m + 1) * x^2 - 2 * (m - 1) * x + 3 * (m - 1) < 0) ↔ (m < -1)) :=
sorry

end range_of_m_l209_209104


namespace totalCerealInThreeBoxes_l209_209818

def firstBox := 14
def secondBox := firstBox / 2
def thirdBox := secondBox + 5
def totalCereal := firstBox + secondBox + thirdBox

theorem totalCerealInThreeBoxes : totalCereal = 33 := 
by {
  sorry
}

end totalCerealInThreeBoxes_l209_209818


namespace centers_cyclic_l209_209126

-- Define the convex quadrilateral and the properties of the circles
variables {A B C D O₁ O₂ O₃ O₄ : Point}
variable (h_convex : ConvexQuadrilateral A B C D)
variable (h_circle_property : ∀ (i : Fin 4), Circle (Centers i) (Tangents i ∪ ExtensionAdjacentTangents i))

-- Main theorem statement
theorem centers_cyclic (hO₁ : Center O₁ O₂ O₃ O₄ (TangentSides A B C D))
  (hO₂ : Center O₁ O₂ O₃ O₄ (TangentSides B C D A))
  (hO₃ : Center O₁ O₂ O₃ O₄ (TangentSides C D A B))
  (hO₄ : Center O₁ O₂ O₃ O₄ (TangentSides D A B C)) :
  Cyclic O₁ O₂ O₃ O₄ := by
  sorry

end centers_cyclic_l209_209126


namespace distance_between_cities_l209_209405

variable (a b : Nat)

theorem distance_between_cities :
  (a = (10 * a + b) - (10 * b + a)) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → 10 * a + b = 98 := by
  sorry

end distance_between_cities_l209_209405


namespace cost_of_each_adult_meal_is_8_l209_209885

/- Define the basic parameters and conditions -/
def total_people : ℕ := 11
def kids : ℕ := 2
def total_cost : ℕ := 72
def kids_eat_free (k : ℕ) := k = 0

/- The number of adults is derived from the total people minus kids -/
def num_adults : ℕ := total_people - kids

/- The cost per adult meal can be defined and we need to prove it equals to $8 -/
def cost_per_adult (total_cost : ℕ) (num_adults : ℕ) : ℕ := total_cost / num_adults

/- The statement to prove that the cost per adult meal is $8 -/
theorem cost_of_each_adult_meal_is_8 : cost_per_adult total_cost num_adults = 8 := by
  sorry

end cost_of_each_adult_meal_is_8_l209_209885


namespace academic_academy_pass_criteria_l209_209055

theorem academic_academy_pass_criteria :
  ∀ (total_problems : ℕ) (passing_percentage : ℕ)
  (max_missed : ℕ),
  total_problems = 35 →
  passing_percentage = 80 →
  max_missed = total_problems - (passing_percentage * total_problems) / 100 →
  max_missed = 7 :=
by 
  intros total_problems passing_percentage max_missed
  intros h_total_problems h_passing_percentage h_calculation
  rw [h_total_problems, h_passing_percentage] at h_calculation
  sorry

end academic_academy_pass_criteria_l209_209055


namespace range_of_a_l209_209651

noncomputable def f (a x : ℝ) : ℝ := a * (x + a) * (x - a + 3)
noncomputable def g (x : ℝ) : ℝ := 2 ^ (x + 2) - 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f a x > 0 ∨ g x > 0) →
  1 < a ∧ a < 2 :=
by {
  have h_g : ∀ x : ℝ, x ≤ -2 → g x ≤ 0 := sorry,
  have h_f : ∀ x : ℝ, x ≤ -2 → f a x > 0 := sorry,
  have a_pos : 0 < a := sorry,
  have a_lt_2 : a < 2 := sorry,
  have a_gt_1 : 1 < a := sorry,
  exact ⟨a_gt_1, a_lt_2⟩,
}


end range_of_a_l209_209651


namespace number_of_odd_factors_of_360_l209_209200

def is_odd_factor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ ∀ k, 2 ∣ k → ¬ (k ∣ d)

theorem number_of_odd_factors_of_360 : 
  {d : ℕ // is_odd_factor 360 d}.card = 6 := 
sorry

end number_of_odd_factors_of_360_l209_209200


namespace incorrect_statement_A_l209_209472

-- Definitions of the statements
def statement_A : Prop := 
  ∀ (L1 L2 : ℝ^3 → ℝ^3), (L1 ∩ L2 = ∅) → (parallel L1 L2)

def statement_B : Prop := 
  ∀ (L : ℝ^3 → ℝ^3) (P : ℝ^3 → Prop), (L ∩ P = ∅) → (parallel L P)

def statement_C : Prop := 
  ∀ (P1 P2 : ℝ^3 → Prop), (P1 ∩ P2 = ∅) → (parallel P1 P2)

def statement_D : Prop := 
  ∀ (P1 P2 : ℝ^3 → Prop) (L : ℝ^3 → ℝ^3), (L ⊂ P1) ∧ (perp L P2) → (perp P1 P2)

-- The proof problem: Prove that statement A is incorrect given conditions
theorem incorrect_statement_A (hA : statement_A) (hB: statement_B) (hC: statement_C) (hD: statement_D) : false :=
  sorry

end incorrect_statement_A_l209_209472


namespace car_travelled_miles_l209_209857

def skips_digit_5 (n : ℕ) : ℕ :=
  let digits := list.filter (λ d, d ≠ 5) (n.digits 10)
  in digits.foldl (λ acc d, acc * 9 + if d < 5 then d else d - 1) 0

def odometer_to_actual_miles (odometer_reading : ℕ) : ℕ :=
  skips_digit_5 odometer_reading

theorem car_travelled_miles :
  odometer_to_actual_miles 3006 = 1541 :=
by
  sorry

end car_travelled_miles_l209_209857


namespace bus_cost_proof_l209_209873

-- Define conditions
def train_cost (bus_cost : ℚ) : ℚ := bus_cost + 6.85
def discount_rate : ℚ := 0.15
def service_fee : ℚ := 1.25
def combined_cost : ℚ := 10.50

-- Formula for the total cost after discount
def discounted_train_cost (bus_cost : ℚ) : ℚ := (train_cost bus_cost) * (1 - discount_rate)
def total_cost (bus_cost : ℚ) : ℚ := discounted_train_cost bus_cost + bus_cost + service_fee

-- Lean 4 statement asserting the cost of the bus ride before service fee
theorem bus_cost_proof : ∃ (B : ℚ), total_cost B = combined_cost ∧ B = 1.85 :=
sorry

end bus_cost_proof_l209_209873


namespace num_of_valid_m_vals_l209_209682

theorem num_of_valid_m_vals : 
  (∀ m x : ℤ, (x + m ≤ 4 ∧ (x / 2 - (x - 1) / 4 > 1 → x > 3 → ∃ (c : ℚ), (x + 1)/4 > 1 )) ∧
  (∃ (x : ℤ), (x + m ≤ 4 ∧ (x > 3) ∧ (m < 1 ∧ m > -4)) ∧ 
  ∃ a b : ℚ, x^2 + a * x + b = 0) → 
  (∃ (count m : ℤ), count = 2)) :=
sorry

end num_of_valid_m_vals_l209_209682


namespace part1_smallest_positive_period_and_increasing_interval_part2_max_value_b_plus_c_l209_209646

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + 2 * sqrt 3 * sin x * cos x - sin x ^ 2

theorem part1_smallest_positive_period_and_increasing_interval :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π) ∧
  (∀ k : ℤ, ∃ a b, a = k * π - π / 3 ∧ b = k * π + π / 6 ∧ ∀ x ∈ set.Icc a b, f x = 2 * sin (2 * x + π / 6)) :=
sorry

theorem part2_max_value_b_plus_c
  (A B C a b c : ℝ)
  (hA : f A = 1)
  (ha : a = sqrt 3)
  (hABC : a = sin A ∧ b = sin B ∧ c = sin C ∧ A + B + C = π) :
  b + c ≤ 2 * sqrt 3 :=
sorry

end part1_smallest_positive_period_and_increasing_interval_part2_max_value_b_plus_c_l209_209646


namespace geometric_sequence_problem_l209_209309

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) 
  (geom_seq : ∀ n m k, a m = (a (n + k)) * (a (n - k)))  -- Define the geometric sequence property
  (h : a 3 * a 5 * a 7 = (- real.sqrt 3) ^ 3) : 
  a 2 * a 8 = 3 := 
  sorry

end geometric_sequence_problem_l209_209309


namespace no_diophantine_solution_exists_l209_209563

theorem no_diophantine_solution_exists :
  ¬∃ (n : Fin 14 → ℕ), (∑ i, n i ^ 4) = 1599 :=
by 
  sorry

end no_diophantine_solution_exists_l209_209563


namespace find_x_l209_209265

theorem find_x (x : ℝ) : (3^(x - 4) = 9^3) → x = 10 :=
by
  intro h
  sorry

end find_x_l209_209265


namespace find_x_l209_209261

theorem find_x (x : ℝ) : (3^(x - 4) = 9^3) → x = 10 :=
by
  intro h
  sorry

end find_x_l209_209261


namespace simon_blueberry_count_l209_209384

theorem simon_blueberry_count (b₁ p b_per_pie : ℕ) (H₁ : b₁ = 100) (H₂ : p = 3) (H₃ : b_per_pie = 100) :
  let b_total := p * b_per_pie in
  let b_nearby := b_total - b₁ in
  b_nearby = 200 :=
by
  let b_total := p * b_per_pie
  let b_nearby := b_total - b₁
  have H_total : b_total = 300 := by
    rw [H₂, H₃]
    sorry
  have H_nearby : b_nearby = 200 := by
    rw [H₁, H_total]
    sorry
  exact H_nearby

end simon_blueberry_count_l209_209384


namespace enclosed_area_is_9_over_2_l209_209584

noncomputable def area_under_curve_and_line : ℝ :=
∫ x in -2..1, (9 - x^2) - (x + 7)

theorem enclosed_area_is_9_over_2 : area_under_curve_and_line = 9 / 2 :=
by
  sorry

end enclosed_area_is_9_over_2_l209_209584


namespace fraction_meaningful_condition_l209_209403

theorem fraction_meaningful_condition (x : ℝ) : 3 - x ≠ 0 ↔ x ≠ 3 :=
by sorry

end fraction_meaningful_condition_l209_209403


namespace selecting_n_cards_from_2n_deck_l209_209830

theorem selecting_n_cards_from_2n_deck (n : ℕ) (h : 0 < n) : 
  (2 * n).choose n = nat.factorial (2 * n) / (nat.factorial n * nat.factorial n) :=
by sorry

end selecting_n_cards_from_2n_deck_l209_209830


namespace correct_statement_l209_209995

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l209_209995


namespace find_g_l209_209576

theorem find_g (x : ℝ) (g : ℝ → ℝ) :
  2 * x^5 - 4 * x^3 + 3 * x^2 + g x = 7 * x^4 - 5 * x^3 + x^2 - 9 * x + 2 →
  g x = -2 * x^5 + 7 * x^4 - x^3 - 2 * x^2 - 9 * x + 2 :=
by
  intro h
  sorry

end find_g_l209_209576


namespace solve_equation_l209_209580

theorem solve_equation :
  (∃ x : ℝ, (Real.sqrt(Real.sqrt(59 - 3 * x)) + Real.sqrt(Real.sqrt(17 + 3 * x)) = 4)) ↔ (x = 20 ∨ x = -10) := sorry

end solve_equation_l209_209580


namespace pdf_correct_prob_interval_l209_209808

noncomputable def F (x : ℝ) : ℝ :=
if x ≤ 0 then 0
else if x ≤ π then (1 - Real.cos x) / 2
else 1

noncomputable def p (x : ℝ) : ℝ :=
if x ≤ 0 then 0
else if x ≤ π then (Real.sin x) / 2
else 0

theorem pdf_correct : ∀ x : ℝ, 
  p(x) = Deriv.deriv (F x) x :=
sorry

theorem prob_interval : 
  ∫ x in (π / 3)..(π / 2), p x = 1 / 4 :=
sorry

end pdf_correct_prob_interval_l209_209808


namespace pizza_topping_combinations_l209_209513

theorem pizza_topping_combinations (n k : ℕ) (h_n : n = 8) (h_k : k = 3) :
  nat.choose n k = 56 :=
by
  rw [h_n, h_k]
  dsimp
  norm_num
  -- The detailed proof steps would go here,
  -- but for the problem statement, we leave it as sorry
  sorry

end pizza_topping_combinations_l209_209513


namespace divisible_iff_condition_l209_209348

theorem divisible_iff_condition (a b : ℤ) : 
  (13 ∣ (2 * a + 3 * b)) ↔ (13 ∣ (2 * b - 3 * a)) :=
  sorry

end divisible_iff_condition_l209_209348


namespace no_broken_line_exists_l209_209947

-- Definitions for the conditions provided
def broken_line_intersects_each_segment_exactly_once (segments : ℕ) : Prop :=
  ∀ (line : set (ℝ × ℝ)), 
    -- assuming the segments are given, we define the intersection condition here
    (∃ (points : list (ℝ × ℝ)), 
      -- line is represented by a list of points in the plane
      (∀ segment ∈ segments, (∃ x y, x ≠ y ∧ (x, y) ∈ line → intersects segment (x, y))) ∧
      (∀ v : (ℝ × ℝ), v ∉ segments → v ∈ line)) →
    false

-- Given conditions based on the problem statement
def figure_conditions : Prop :=
  ∃ regions : ℕ, 
    regions = 6 ∧
    ∃ bounded_regions : ℕ, 
      bounded_regions = 5 ∧
      ∃ unbounded_region : ℕ,
        unbounded_region = 1 ∧
        ∃ even_border_regions : ℕ, 
          even_border_regions = 2 ∧
          ∃ odd_border_regions : ℕ, 
            odd_border_regions = 4

-- The mathematical proof problem: prove that such a broken line does not exist
theorem no_broken_line_exists (segments : list (ℝ × ℝ)) 
  (h_segments : length segments = 16) 
  (h_conditions : figure_conditions) : broken_line_intersects_each_segment_exactly_once 16 := 
by
  sorry -- Proof omitted as stated

end no_broken_line_exists_l209_209947


namespace scientific_notation_of_concentration_l209_209755

theorem scientific_notation_of_concentration :
  0.000042 = 4.2 * 10^(-5) :=
sorry

end scientific_notation_of_concentration_l209_209755


namespace high_school_ten_total_games_l209_209394

open Finset

-- Define the necessary conditions and the statement to be proved
theorem high_school_ten_total_games :
  let num_teams := 10,
      conference_games_per_team := (num_teams * (num_teams - 1)) / 2 * 2,
      non_conference_games_per_team := 6
  in conference_games_per_team + num_teams * non_conference_games_per_team = 150 :=
by
  sorry

end high_school_ten_total_games_l209_209394


namespace rate_of_work_l209_209842

theorem rate_of_work (A : ℝ) (h1: 0 < A) (h_eq : 1 / A + 1 / 6 = 1 / 2) : A = 3 := sorry

end rate_of_work_l209_209842


namespace triangle_equality_condition_l209_209363

-- Define the triangle and angles
variables {A B C : Point} -- Points in the triangle
variables {alpha beta gamma : ℝ} -- Angles at the vertices of the triangle
variables {BC XY : ℝ} -- Lengths of the sides

-- Conditions
def angle_tan_mul_condition := if (Real.tan beta * Real.tan gamma = 3 ∨ Real.tan beta * Real.tan gamma = -1) then True else False

-- The theorem to be proven
theorem triangle_equality_condition:
  angle_tan_mul_condition →
  BC = XY :=
sorry

end triangle_equality_condition_l209_209363


namespace min_books_l209_209604

theorem min_books (P1 P2 P3 P4 : Set ℕ)
  (hP1 : P1.card = 4)
  (hP2 : P2.card = 4)
  (hP3 : P3.card = 4)
  (hP4 : P4.card = 4)
  (hP1P2 : (P1 ∩ P2).card = 2)
  (hP1P3 : (P1 ∩ P3).card = 2)
  (hP1P4 : (P1 ∩ P4).card = 2)
  (hP2P3 : (P2 ∩ P3).card = 2)
  (hP2P4 : (P2 ∩ P4).card = 2)
  (hP3P4 : (P3 ∩ P4).card = 2) :
  (P1 ∪ P2 ∪ P3 ∪ P4).card = 7 := 
sorry

end min_books_l209_209604


namespace conner_needs_to_collect_l209_209390

theorem conner_needs_to_collect (sydney_start conner_start : ℕ) (sydney_day1 : ℕ)
  (conner_day1_mul : ℕ) (conner_day2 : ℕ) : ∃ (conner_day3 : ℕ),
  let 
    sydney_day1_total := sydney_start + sydney_day1,
    conner_day1 := sydney_day1 * conner_day1_mul,
    conner_day1_total := conner_start + conner_day1,
    sydney_day2_total := sydney_day1_total,
    conner_day2_total := conner_day1_total + conner_day2,
    sydney_day3 := conner_day1 * 2,
    sydney_day3_total := sydney_day2_total + sydney_day3
  in
  conner_day3_total = sydney_day3_total
  → conner_day3_total = conner_day2_total + conner_day3 :=
begin
  assume h,
  use sydney_day3_total - conner_day2_total,
  exact h,
end

end conner_needs_to_collect_l209_209390


namespace triangle_identity_l209_209771

variables {α : Type*} [LinearOrderedField α]

-- Definitions of the sides of the triangle
variables (a b c : α) (h_cb : c > b)

-- Definitions related to geometric points M₁ and F₁
variables (M₁ F₁ : α)

-- Hypotheses: M₁ is the foot of the altitude and F₁ is the midpoint of side a
hypothesis (h_midpoint : F₁ = a / 2)
hypothesis (h_alt_foot : ∃ (h : α), M₁ ^ 2 + (a / 2) ^ 2 + b ^ 2 = c ^ 2 ∧ M₁ ^ 2 + (a / 2) ^ 2 = h ^ 2)

theorem triangle_identity :
  c^2 - b^2 = 2 * a * M₁ :=
sorry

end triangle_identity_l209_209771


namespace strictly_decreasing_on_interval_l209_209082

noncomputable def f (x : ℝ) : ℝ := real.log (2 * x - x ^ 2)

theorem strictly_decreasing_on_interval : 
    ∀ x ∈ set.Ioo (1 : ℝ) 2, ∀ y ∈ set.Ioo (1 : ℝ) 2, (x < y → f x > f y) :=
by
  sorry

end strictly_decreasing_on_interval_l209_209082


namespace sequence_properties_l209_209638

theorem sequence_properties (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ, a (n + 1) = 2 * a n + 3) →
  (∀ n : ℕ, S n = ∑ i in finset.range n, a (i + 1)) →
  (a 3 = 13) ∧
  (∀ n : ℕ, a (n + 1) + 3 = 2 * (a n + 3)) ∧
  (∀ n : ℕ, S n = 2^(n + 2) - 3 * n - 4) :=
by
  intros h1 h2 h3
  -- Proof omitted
  sorry

end sequence_properties_l209_209638


namespace shaded_area_percentage_l209_209458

theorem shaded_area_percentage {E F G H : Type} 
  (area_total_square : ℝ)
  (area_shaded_first : ℝ)
  (area_shaded_second : ℝ)
  (area_shaded_third : ℝ) :
  area_total_square = 36 ∧
  area_shaded_first = 4 ∧
  area_shaded_second = 7 ∧
  area_shaded_third = 11 →
  (area_shaded_first + area_shaded_second + area_shaded_third) / area_total_square * 100 ≈ 61.11 :=
by
  -- Given conditions
  intros h,
  rcases h with ⟨htotal, hfirst, hsecond, hthird⟩,
  -- Deriving total shaded area and the required percentage
  sorry

end shaded_area_percentage_l209_209458


namespace problem_statement_l209_209130

def sequence_arithmetic (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a (n+1) / 2^(n+1) - a n / 2^n = 1)

theorem problem_statement : 
  ∃ a : ℕ → ℝ, a 1 = 2 ∧ a 2 = 8 ∧ (∀ n : ℕ, n ≥ 1 → a (n+1) - 2 * a n = 2^(n+1)) → sequence_arithmetic a :=
by
  sorry

end problem_statement_l209_209130


namespace smallest_positive_period_of_f_symmetry_axis_of_f_range_of_f_on_interval_l209_209647

def f (x : ℝ) : ℝ := sin (2 * x + π / 3) + tan (5 * π / 6) * cos (2 * x)

theorem smallest_positive_period_of_f :
  is_periodic f π ∧
  (∀ T > 0, is_periodic f T → π ≤ T) :=
sorry

theorem symmetry_axis_of_f :
  ∃ k : ℤ, (x = k * π / 2 + π / 6) :=
sorry

theorem range_of_f_on_interval :
  set.range (λ x, f x) = set.Icc (-(sqrt 3) / 6) (sqrt 3 / 3) :=
sorry

end smallest_positive_period_of_f_symmetry_axis_of_f_range_of_f_on_interval_l209_209647


namespace number_of_positive_area_triangles_l209_209248

theorem number_of_positive_area_triangles (s : Finset (ℕ × ℕ)) (h : s = {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}) :
  (s.powerset.filter λ t, t.card = 3 ∧ ¬∃ a b c, a = b ∨ b = c ∨ c = a).card = 2160 :=
by {
  sorry
}

end number_of_positive_area_triangles_l209_209248


namespace trapezium_area_l209_209585

theorem trapezium_area (a b h : ℝ) (ha : a = 10) (hb : b = 18) (hh : h = 15) : (1/2 * (a + b) * h) = 210 := 
by
  rw [ha, hb, hh]
  norm_num
  sorry

end trapezium_area_l209_209585


namespace carrie_profit_l209_209064

def total_hours_worked (hours_per_day: ℕ) (days: ℕ): ℕ := hours_per_day * days
def total_earnings (hours_worked: ℕ) (hourly_wage: ℕ): ℕ := hours_worked * hourly_wage
def profit (total_earnings: ℕ) (cost_of_supplies: ℕ): ℕ := total_earnings - cost_of_supplies

theorem carrie_profit (hours_per_day: ℕ) (days: ℕ) (hourly_wage: ℕ) (cost_of_supplies: ℕ): 
    hours_per_day = 2 → days = 4 → hourly_wage = 22 → cost_of_supplies = 54 → 
    profit (total_earnings (total_hours_worked hours_per_day days) hourly_wage) cost_of_supplies = 122 := 
by
    intros hpd d hw cos
    sorry

end carrie_profit_l209_209064


namespace reflection_matrix_values_l209_209909

noncomputable def matrix_reflection (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![[a, b], [-3/5, 4/5]]

noncomputable def identity_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1, 0], ![0, 1]]

theorem reflection_matrix_values (a b : ℚ) :
  matrix_reflection a b ⬝ matrix_reflection a b = identity_matrix ↔ a = -4/5 ∧ b = 3/5 := 
by
  sorry

end reflection_matrix_values_l209_209909


namespace work_together_l209_209856

theorem work_together (work : ℚ) (days_A days_B : ℚ) (rate_A rate_B : ℚ) (combined_rate : ℚ) (time : ℚ) :
  work = 1 → days_A = 6 → days_B = 12 → rate_A = 1 / days_A → rate_B = 1 / days_B →
  combined_rate = rate_A + rate_B → time = work / combined_rate → time = 4 :=
by
  intros h_work h_days_A h_days_B h_rate_A h_rate_B h_combined_rate h_time
  rw [h_work, h_days_A, h_days_B, h_rate_A, h_rate_B, h_combined_rate, h_time]
  sorry

end work_together_l209_209856


namespace compatibility_with_2003_l209_209948

def D (a : ℕ) : ℕ := 2 * a + 1
def T (a : ℕ) : ℕ := 3 * a + 2

def S (n : ℕ) : set ℕ :=
  {x | ∃ m k, (m + k > 0) ∧ ( (x = D^[m] (T^[k] n) ) ∨ (x = T^[m] (D^[k] n) ) )}

noncomputable def is_compatible (m n : ℕ) : Prop :=
  ↑∅ ⊂ (S m ∩ S n)

theorem compatibility_with_2003 :
  ∀ a ∈ {166, 333, 500, 667, 1001, 1335, 1502}, is_compatible 2003 a :=
by
  sorry

end compatibility_with_2003_l209_209948


namespace total_tagged_numbers_l209_209540

theorem total_tagged_numbers {W X Y Z : ℕ} 
  (hW : W = 200)
  (hX : X = W / 2)
  (hY : Y = W + X)
  (hZ : Z = 400) :
  W + X + Y + Z = 1000 := by
  sorry

end total_tagged_numbers_l209_209540


namespace circle_equation_l209_209636

theorem circle_equation (a : ℝ) (h : a < 0) 
  (tangent_line : 3 * a + 4 + 4 = 4 * 5) :
  ∃ x y, (x + a)^2 + y^2 = 16 ∧ 3 * x + 4 * y + 4 = 0 :=
by {
  use [-8, 0],
  split,
  { norm_num },
  { norm_num }
}

end circle_equation_l209_209636


namespace sum_areas_of_square_and_rectangle_l209_209811

theorem sum_areas_of_square_and_rectangle (s w l : ℝ) 
  (h1 : s^2 + w * l = 130)
  (h2 : 4 * s - 2 * (w + l) = 20)
  (h3 : l = 2 * w) : 
  s^2 + 2 * w^2 = 118 :=
by
  -- Provide space for proof
  sorry

end sum_areas_of_square_and_rectangle_l209_209811


namespace odd_factors_of_360_l209_209191

theorem odd_factors_of_360 : ∃ n, n = 6 ∧ 
  ∀ (x : ℕ), (∃ (a b : ℕ), 360 = 2^a * 3^b * x ∧ ¬ even x ∧ 0 < x) → (∀ (m : ℕ), m ∣ x ↔ (m = 1 ∨ m = 3 ∨ m = 5 ∨ m = 9 ∨ m = 15 ∨ m = 45)) :=
by
  -- Proof
  sorry

end odd_factors_of_360_l209_209191


namespace length_BD_l209_209442

noncomputable def triangle : Type := sorry

def is_isosceles {ABC : triangle} (A B C : triangle) : Prop := sorry
def midpoint (D : triangle) (BC AE : triangle) : Prop := sorry
def length (x y : triangle) : ℝ := sorry

theorem length_BD (ABC : triangle) (A B C D E : triangle) 
  (h_isosceles : is_isosceles ABC A B C)
  (h_midpoint_D : midpoint D BC AE)
  (h_length_CE : length C E = 15) 
  : length B D = 7.5 :=
sorry

end length_BD_l209_209442


namespace divide_estate_l209_209527

theorem divide_estate (total_estate : ℕ) (son_share : ℕ) (daughter_share : ℕ) (wife_share : ℕ) :
  total_estate = 210 →
  son_share = (4 / 7) * total_estate →
  daughter_share = (1 / 7) * total_estate →
  wife_share = (2 / 7) * total_estate →
  son_share + daughter_share + wife_share = total_estate :=
by
  intros
  sorry

end divide_estate_l209_209527


namespace odd_factors_of_360_l209_209171

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l209_209171


namespace liza_pages_per_hour_l209_209351

variable (L : ℕ)
variable (S : ℕ := 15)
variable (H1 : Liza reads L pages in an hour)
variable (H2 : Suzie reads S pages in an hour)
variable (H3 : Liza reads 15 more pages than Suzie in 3 hours)

theorem liza_pages_per_hour : L = 20 := by
  have suzie_pages_in_3_hours : S * 3 = 45 := by
    -- Suzie reads 15 pages per hour, so in 3 hours she reads 45 pages
    rw [S]
    norm_num
    
  have liza_pages_in_3_hours : Liza reads ( Suzies reads + 15 ) pages in 3 hours := by
    -- Liza reads 15 more pages than Suzie in 3 hours, so Liza reads 60 pages in 3 hours
    rw [suzie_pages_in_3_hours]
    norm_num
    
  have one_hour_reading : L = 60 / 3 := by
    sorry

  have final_reading : L = 20 := by
    sorry

  exact final_reading

end liza_pages_per_hour_l209_209351


namespace total_interest_proof_l209_209865

/-- 
  Define the initial conditions.
-/
def stock1_initial_rate : ℝ := 16
def stock2_initial_rate : ℝ := 12
def stock3_initial_rate : ℝ := 20

def stock1_purchase_price : ℝ := 128
def stock2_purchase_price : ℝ := 110
def stock3_purchase_price : ℝ := 136

def annual_increase : ℝ := 2
def face_value : ℝ := 100

/- 
  Calculate the interest for a given year.
-/
def interest (initial_rate : ℝ) (year : ℕ) : ℝ :=
  (initial_rate + (annual_increase * (year - 1))) * face_value / 100

/-- 
  Function to calculate total interest over 5 years.
-/
def total_interest_over_5_years : ℝ :=
  (List.sum (List.map (interest stock1_initial_rate) [1, 2, 3, 4, 5])) +
  (List.sum (List.map (interest stock2_initial_rate) [1, 2, 3, 4, 5])) +
  (List.sum (List.map (interest stock3_initial_rate) [1, 2, 3, 4, 5]))

/-- 
  Proof goal.
-/
theorem total_interest_proof : total_interest_over_5_years = 330 := by
  simp [stock1_initial_rate, stock2_initial_rate, stock3_initial_rate, face_value, annual_increase, interest, total_interest_over_5_years]
  sorry

end total_interest_proof_l209_209865


namespace no_real_roots_probability_l209_209165

theorem no_real_roots_probability :
  let a b : ℝ := sorry
  let condition := (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1) ∧ (a^2 < 4 * b^2)
  ∃ probability : ℝ, condition → probability = 3 / 4 := sorry

end no_real_roots_probability_l209_209165


namespace relationship_of_functions_l209_209459

theorem relationship_of_functions (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1) :
  let f := x^2
  let g := x ^ (1/2)
  let h := x^(-2)
in h > g ∧ g > f :=
by
  let f := x^2
  let g := x^(1/2)
  let h := x^(-2)
  sorry

end relationship_of_functions_l209_209459


namespace sum_of_denominators_at_least_l209_209391

theorem sum_of_denominators_at_least (n : ℕ) (h : n ≥ 2)
  (f : Π i, i < n → {x // 0 < x ∧ x < 1} → ℚ)
  (hf_different : ∀ i j (hi : i < n) (hj : j < n) (xi : {x // 0 < x ∧ x < 1}) (xj : {x // 0 < x ∧ x < 1}),
    f i hi xi = f j hj xj → (i = j ∧ xi = xj)) :
  ∑ i in finset.range n, (f i (by linarith) (⟨⟨i / (i + 1 : ℕ)⟩, by linarith, by linarith⟩)).denom
  ≥ ⌊(1 / 3 * (n : ℚ) ^ (1 / 2))⌋ :=
sorry

end sum_of_denominators_at_least_l209_209391


namespace part1_logarithm_identity_part2_logarithm_identity_l209_209573

theorem part1_logarithm_identity :
  (Real.log10 14 - 2 * Real.log10 (7 / 3) + Real.log10 7 - Real.log10 18) = 0 := 
sorry

theorem part2_logarithm_identity :
  (Real.logb 25 625 + Real.log10 0.01 + Real.log (Real.sqrt Real.exp) - 2^(1 + Real.log2 3)) = -11/2 := 
sorry

end part1_logarithm_identity_part2_logarithm_identity_l209_209573


namespace circular_card_covers_squares_l209_209021

-- Defining the radius of the circular card
def radius : ℝ := 1.5

-- Defining the side length of one square in the checkerboard
def square_side_length : ℝ := 1.0

-- Defining the total number of squares covered by the circular card
def max_squares_covered : ℕ := 12

-- The main theorem stating that a circular card with the given radius
-- can cover at least the given number of squares on the checkerboard
theorem circular_card_covers_squares 
    (r : ℝ) (a : ℝ) (n : ℕ)
    (h_r : r = radius) 
    (h_a : a = square_side_length)
    (h_n : n = max_squares_covered) :
    ∃ S : set (ℤ × ℤ), finite S ∧ S.card = max_squares_covered := 
begin
  sorry
end

end circular_card_covers_squares_l209_209021


namespace intersection_of_median_and_altitude_l209_209710

noncomputable section

open Real

structure Point where
  x : ℝ
  y : ℝ

/-- Given vertices A, B, and C in the plane -/
def A : Point := ⟨5, 1⟩
def B : Point := ⟨-1, -3⟩
def C : Point := ⟨4, 3⟩

/-- Function to compute midpoint of a line segment between two points -/
def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

/-- The main theorem stating the intersection point of median and altitude -/
theorem intersection_of_median_and_altitude :
  let M := midpoint A B
  let CM_eq := (2 * (C.x - M.x), (C.y - M.y))
  let AC_slope := (C.y - A.y) / (C.x - A.x)
  let BN_slope := -1 / AC_slope
  let BN_eq := (BN_slope * (B.x - A.x + B.y - A.y))
  (2 * (5 / 3) - (-5 / 3) - 5 = 0) ∧
  (5 / 3 - 2 * (-5 / 3) - 5 = 0) :=
by
  sorry

end intersection_of_median_and_altitude_l209_209710


namespace base16_to_base2_digits_l209_209465

theorem base16_to_base2_digits (n : ℕ) (hn : n = 0xB1234) : nat.log2 n + 1 = 20 :=
sorry

end base16_to_base2_digits_l209_209465


namespace odd_factors_of_360_l209_209175

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l209_209175


namespace probability_of_b_greater_than_a_l209_209776

theorem probability_of_b_greater_than_a :
  let outcomes := {(a, b) | a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3}} in
  let favorable := {(a, b) | a ∈ {1, 2, 3, 4, 5} ∧ b ∈ {1, 2, 3} ∧ b > a} in
  (favorable.card : ℚ) / (outcomes.card) = 1 / 5 :=
by
  sorry

end probability_of_b_greater_than_a_l209_209776


namespace sequence_formula_l209_209920

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n > 1, a n - a (n - 1) = 2^(n-1)) : a n = 2^n - 1 := 
sorry

end sequence_formula_l209_209920


namespace exponent_multiplication_l209_209610

theorem exponent_multiplication (m n : ℕ) (h : m + n = 3) : 2^m * 2^n = 8 := 
by
  sorry

end exponent_multiplication_l209_209610


namespace no_real_roots_for_Q_l209_209770

noncomputable def has_no_real_roots (P : polynomial ℝ) : Prop :=
∀ x : ℝ, P.eval x ≠ 0

theorem no_real_roots_for_Q (P : polynomial ℝ) (n : ℕ) (α : ℝ)
  (h_degree : P.natDegree = n) (h_no_real_roots : has_no_real_roots P) :
  has_no_real_roots (P + ∑ k in finset.range (n + 1), (monomial k α) * P.derivative ^ k) :=
sorry

end no_real_roots_for_Q_l209_209770


namespace limit_of_function_l209_209544

-- Define the function and its limit
noncomputable def f (x : ℝ) := (Real.tan (4 * x) / x) ^ (2 + x)

-- State the theorem to prove the limit
theorem limit_of_function : (Real.limit (λ x, f x) 0) = 16 :=
by
  sorry

end limit_of_function_l209_209544


namespace integral_cos_pi_over_6_l209_209915

theorem integral_cos_pi_over_6 :
  ∫ x in 0..(Real.pi / 6), Real.cos x = 1 / 2 := by
  sorry

end integral_cos_pi_over_6_l209_209915


namespace find_x_l209_209256

theorem find_x (x : ℝ) : (3^(x - 4) = 9^3) → x = 10 :=
by
  sorry

end find_x_l209_209256


namespace number_of_valid_triangles_l209_209236

-- Definition of the set of points in the 5x5 grid with integer coordinates
def gridPoints := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

-- Function to determine if three points are collinear
def collinear (a b c : ℕ × ℕ) : Prop :=
  (b.2 - a.2) * (c.1 - b.1) = (c.2 - b.2) * (b.1 - a.1)

-- The main theorem stating the number of triangles with positive area
theorem number_of_valid_triangles : 
  ∃ n, n = 2158 ∧ ∀ (a b c : ℕ × ℕ), a ∈ gridPoints → b ∈ gridPoints → c ∈ gridPoints → a ≠ b → b ≠ c → c ≠ a → ¬collinear a b c → n = 2158 :=
by
  sorry

end number_of_valid_triangles_l209_209236


namespace number_of_odd_factors_of_360_l209_209198

def is_odd_factor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ ∀ k, 2 ∣ k → ¬ (k ∣ d)

theorem number_of_odd_factors_of_360 : 
  {d : ℕ // is_odd_factor 360 d}.card = 6 := 
sorry

end number_of_odd_factors_of_360_l209_209198


namespace length_of_AB_perimeter_of_triangle_F2_AB_l209_209493

-- Define the hyperbola as a set of points (x, y) such that x^2 - y^2 / 3 = 1
def is_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 3 = 1

-- Define the focus points
def F1 := (-2 : ℝ, 0 : ℝ)
def F2 := (2 : ℝ, 0 : ℝ)

-- Define the line AB making an angle π/6 with the horizontal
def is_line_AB (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x + 2)

-- Length of a line segment between two points (x1, y1) and (x2, y2)
def length (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- The first part of the problem: the length of AB
theorem length_of_AB :
  ∃ x1 y1 x2 y2 : ℝ, is_hyperbola x1 y1 ∧ is_hyperbola x2 y2 ∧ is_line_AB x1 y1 ∧ is_line_AB x2 y2 ∧
  F1 = (x1, y1) ∨ F1 = (x2, y2) ∧
  F2 = (x1, y1) ∨ F2 = (x2, y2) ∧
  length x1 y1 x2 y2 = 3 :=
by sorry

-- The second part of the problem: the perimeter of triangle F2AB
theorem perimeter_of_triangle_F2_AB :
  ∃ x1 y1 x2 y2 : ℝ, is_hyperbola x1 y1 ∧ is_hyperbola x2 y2 ∧ is_line_AB x1 y1 ∧ is_line_AB x2 y2 ∧
  F1 = (x1, y1) ∨ F1 = (x2, y2) ∧
  F2 = (x1, y1) ∨ F2 = (x2, y2) ∧
  length F2.1 F2.2 x1 y1 + length F2.1 F2.2 x2 y2 + length x1 y1 x2 y2 = 3 + 3 * Real.sqrt 3 :=
by sorry

end length_of_AB_perimeter_of_triangle_F2_AB_l209_209493


namespace abc_sum_eq_50_l209_209389

theorem abc_sum_eq_50 (a b c : ℝ) :
  (∀ x : ℝ, f(x+5) = 5 * x^2 + 9 * x + 6) →
  (∀ x : ℝ, f(x) = a * x^2 + b * x + c) →
  a + b + c = 50 :=
by
  sorry

end abc_sum_eq_50_l209_209389


namespace centroid_of_quadrilateral_correct_l209_209401

noncomputable def centroid_of_quadrilateral (A B C D : Point)
  (α β γ δ : ℝ) (hα : α = (2 : ℝ) / 3)
  (hβ : β = (2 : ℝ) / 3)
  (hγ : γ = (2 : ℝ) / 3)
  (hδ : δ = (2 : ℝ) / 3) : Point :=
let A1 := (2 / 3 : ℝ) * (B - A) + A,
    B1 := (2 / 3 : ℝ) * (C - B) + B,
    C1 := (2 / 3 : ℝ) * (D - C) + C,
    D1 := (2 / 3 : ℝ) * (A - D) + D,
    A2 := (1 / 3 : ℝ) * (B - A) + A,
    B2 := (1 / 3 : ℝ) * (C - B) + B,
    C2 := (1 / 3 : ℝ) * (D - C) + C,
    D2 := (1 / 3 : ℝ) * (A - D) + D,
    K := line_intersection A2 B1,
    L := line_intersection B2 C1,
    M := line_intersection C2 D1,
    N := line_intersection D2 A1,
    S := line_intersection K M ∩ line_intersection L N in
S

theorem centroid_of_quadrilateral_correct (A B C D : Point)
  (α β γ δ : ℝ) (hα : α = (2 : ℝ) / 3)
  (hβ : β = (2 : ℝ) / 3)
  (hγ : γ = (2 : ℝ) / 3)
  (hδ : δ = (2 : ℝ) / 3) :
  let S := centroid_of_quadrilateral A B C D α β γ δ hα hβ hγ hδ in
  ∃ (S_D S_B S_C S_A : Point),
    centroid_triangle A B C = S_D ∧
    centroid_triangle A C D = S_B ∧
    centroid_triangle A B D = S_C ∧
    centroid_triangle C B D = S_A ∧
    segment (S_D S_B) ∩ segment (S_C S_A) = S := sorry

end centroid_of_quadrilateral_correct_l209_209401


namespace projection_magnitude_correct_l209_209166

-- Define the vectors a and b
def a : ℝ × ℝ := (0, 1)
def b : ℝ × ℝ := (1, real.sqrt 3)

-- Function to compute the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Function to compute the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Function to compute the magnitude of the projection of a onto b
def projection_magnitude (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude b)

-- Lean statement to prove the magnitude of the projection
theorem projection_magnitude_correct : projection_magnitude a b = real.sqrt 3 / 2 := by
  sorry

end projection_magnitude_correct_l209_209166


namespace emily_gardens_and_seeds_l209_209913

variables (total_seeds planted_big_garden tom_seeds lettuce_seeds pepper_seeds tom_gardens lettuce_gardens pepper_gardens : ℕ)

def seeds_left (total_seeds planted_big_garden : ℕ) : ℕ :=
  total_seeds - planted_big_garden

def seeds_used_tomatoes (tom_seeds tom_gardens : ℕ) : ℕ :=
  tom_seeds * tom_gardens

def seeds_used_lettuce (lettuce_seeds lettuce_gardens : ℕ) : ℕ :=
  lettuce_seeds * lettuce_gardens

def seeds_used_peppers (pepper_seeds pepper_gardens : ℕ) : ℕ :=
  pepper_seeds * pepper_gardens

def remaining_seeds (total_seeds planted_big_garden tom_seeds tom_gardens lettuce_seeds lettuce_gardens : ℕ) : ℕ :=
  seeds_left total_seeds planted_big_garden - (seeds_used_tomatoes tom_seeds tom_gardens + seeds_used_lettuce lettuce_seeds lettuce_gardens)

def total_small_gardens (tom_gardens lettuce_gardens pepper_gardens : ℕ) : ℕ :=
  tom_gardens + lettuce_gardens + pepper_gardens

theorem emily_gardens_and_seeds :
  total_seeds = 42 ∧
  planted_big_garden = 36 ∧
  tom_seeds = 4 ∧
  lettuce_seeds = 3 ∧
  pepper_seeds = 2 ∧
  tom_gardens = 3 ∧
  lettuce_gardens = 2 →
  seeds_used_peppers pepper_seeds pepper_gardens = 0 ∧
  total_small_gardens tom_gardens lettuce_gardens pepper_gardens = 5 :=
by
  sorry

end emily_gardens_and_seeds_l209_209913


namespace find_d_value_l209_209928

theorem find_d_value (a b : ℚ) (d : ℚ) (h1 : a = 2) (h2 : b = 11) 
  (h3 : ∀ x, 2 * x^2 + 11 * x + d = 0 ↔ x = (-11 + Real.sqrt 15) / 4 ∨ x = (-11 - Real.sqrt 15) / 4) : 
  d = 53 / 4 :=
sorry

end find_d_value_l209_209928


namespace odd_factors_360_l209_209211

theorem odd_factors_360 : 
  let fac := (2^3) * (3^2) * (5^1) in
  fac = 360 → 
  num_factors (3^2 * 5^1) = 6 :=
sorry

end odd_factors_360_l209_209211


namespace graph_shift_l209_209438

def f1 (x : ℝ) := sin (2 * x - π / 3)
def f2 (x : ℝ) := cos (2 * (π / 4 - x))

theorem graph_shift :
  (∀ x, f2 (x - π / 6) = f1 x) :=
by
  intros x
  have h1 : f1 x = sin (2 * (x - π / 6)), from sorry
  have h2 : f2 (x - π / 6) = sin 2 x, from sorry
  rw [←h1, h2]
  refl

end graph_shift_l209_209438


namespace problem_statement_l209_209618

theorem problem_statement (n : ℕ) (b : ℕ → ℝ) (h₀ : 2 ≤ n) (h₁ : ∀ i, 1 ≤ i → i ≤ n → 0 < b i) :
  ( ( (∑ j in finset.range (n + 1), (finset.prod (finset.range (n + 1)) (λ i, b i))^(1/(n + 1))) / (∑ j in finset.range (n + 1), b j) )^ (1/n) + 
  ( (finset.prod (finset.range (n + 1)) (λ i, b i))^(1/n) / (∑ j in finset.range (n + 1), (finset.prod (finset.range (j + 1)) (λ i, b i))^(1/(j + 1))) ) 
  ≤ (n + 1) / n) :=
sorry

end problem_statement_l209_209618


namespace john_total_payment_l209_209718

def cost_of_nikes : ℝ := 150
def cost_of_work_boots : ℝ := 120
def tax_rate : ℝ := 0.10

theorem john_total_payment :
  let total_cost_before_tax := cost_of_nikes + cost_of_work_boots in
  let tax := tax_rate * total_cost_before_tax in
  let total_payment := total_cost_before_tax + tax in
  total_payment = 297 :=
by
  sorry

end john_total_payment_l209_209718


namespace a_minus_b_l209_209681

theorem a_minus_b (a b : ℚ) :
  (∀ x y, (x = 3 → y = 7) ∨ (x = 10 → y = 19) → y = a * x + b) →
  a - b = -(1/7) :=
by
  sorry

end a_minus_b_l209_209681


namespace find_x_l209_209258

theorem find_x (x : ℝ) : (3^(x - 4) = 9^3) → x = 10 :=
by
  sorry

end find_x_l209_209258


namespace concatenated_natural_irrational_l209_209713

def concatenated_natural_decimal : ℝ := 0.1234567891011121314151617181920 -- and so on

theorem concatenated_natural_irrational :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ concatenated_natural_decimal = p / q :=
sorry

end concatenated_natural_irrational_l209_209713


namespace odd_factors_360_l209_209213

theorem odd_factors_360 : 
  let fac := (2^3) * (3^2) * (5^1) in
  fac = 360 → 
  num_factors (3^2 * 5^1) = 6 :=
sorry

end odd_factors_360_l209_209213


namespace ellipse_condition_l209_209934

variables (m n : ℝ)

-- Definition of the curve
def curve_eqn (x y : ℝ) := m * x^2 + n * y^2 = 1

-- Define the condition for being an ellipse
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

def mn_positive (m n : ℝ) : Prop := m * n > 0

-- Prove that mn > 0 is a necessary but not sufficient condition
theorem ellipse_condition (m n : ℝ) : mn_positive m n → is_ellipse m n → False := sorry

end ellipse_condition_l209_209934


namespace distinct_rectangular_arrays_l209_209023

theorem distinct_rectangular_arrays (chairs : ℕ) (h_chairs : chairs = 49) :
  ∃! (m n : ℕ), m * n = chairs ∧ 2 ≤ m ∧ 2 ≤ n :=
by
  use 7, 7
  split
  · exact h_chairs ▸ rfl
  · split
    · exact le_refl 7
    · exact le_refl 7
  · intros x y h
    rcases h with ⟨h, hx, hy⟩
    have : x = 7 := by
      rw [←Nat.mul_right_inj zero_lt_one] at h
      exact nat.eq_of_mul_eq_mul_right' h
    have : y = 7 := by
      rw [←Nat.mul_left_inj zero_lt_one] at h
      exact nat.eq_of_mul_eq_mul_left' h
    congr

end distinct_rectangular_arrays_l209_209023


namespace circle_equation_l209_209919

-- Definitions based on conditions
def center : ℝ × ℝ := (-3, 2)
def pointA : ℝ × ℝ := (1, -1)
def radius (c p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((c.1 - p.1)^2 + (c.2 - p.2)^2)

-- Problem statement rewritten as a Lean 4 theorem
theorem circle_equation : 
  let r := radius center pointA in
  r = 5 → 
  ∀ (x y : ℝ), (x + 3)^2 + (y - 2)^2 = 25 ↔
  (x, y) = center ∨ r = Real.sqrt ((x + 3)^2 + (y - 2)^2) :=
by
  intros
  intro hr
  split
  · intro h
    right
    rw [hr]
    sorry
  · intro h
    cases h with h1 h2
    · rw [h1]
      sorry
    · rwa [hr] at h2

end circle_equation_l209_209919


namespace odd_factors_of_360_l209_209192

theorem odd_factors_of_360 : ∃ n, n = 6 ∧ 
  ∀ (x : ℕ), (∃ (a b : ℕ), 360 = 2^a * 3^b * x ∧ ¬ even x ∧ 0 < x) → (∀ (m : ℕ), m ∣ x ↔ (m = 1 ∨ m = 3 ∨ m = 5 ∨ m = 9 ∨ m = 15 ∨ m = 45)) :=
by
  -- Proof
  sorry

end odd_factors_of_360_l209_209192


namespace route_y_saves_time_l209_209354

theorem route_y_saves_time (distance_X speed_X : ℕ)
                           (distance_Y_WOCZ distance_Y_CZ speed_Y speed_Y_CZ : ℕ)
                           (time_saved_in_minutes : ℚ) :
  distance_X = 8 → 
  speed_X = 40 → 
  distance_Y_WOCZ = 6 → 
  distance_Y_CZ = 1 → 
  speed_Y = 50 → 
  speed_Y_CZ = 25 → 
  time_saved_in_minutes = 2.4 →
  (distance_X / speed_X : ℚ) * 60 - 
  ((distance_Y_WOCZ / speed_Y + distance_Y_CZ / speed_Y_CZ) * 60) = time_saved_in_minutes :=
by
  intros
  sorry

end route_y_saves_time_l209_209354


namespace find_roots_of_star_eq_l209_209562

def star (a b : ℝ) : ℝ := a^2 - b^2

theorem find_roots_of_star_eq :
  (star (star 2 3) x = 9) ↔ (x = 4 ∨ x = -4) :=
by
  sorry

end find_roots_of_star_eq_l209_209562


namespace fraction_power_division_l209_209898

theorem fraction_power_division :
  let a := (9 : ℚ) / 5
  let b := (3 : ℚ) / 4
  (a^4 * a^(-4) / b^2) = (16 / 9 : ℚ) :=
by 
  sorry

end fraction_power_division_l209_209898


namespace determine_constants_l209_209109

-- Define the functions and transformations
variable (f : ℝ → ℝ)
variable (a b c : ℝ)

-- Define the function g in terms of f, a, b, c
def g (x : ℝ) : ℝ := a * f(b * x) + c

-- State the transformations
def transformation_hor_compression (x : ℝ) : ℝ := f(x / 3)
def transformation_v_shift (y : ℝ) : ℝ := y - 2

-- Main theorem statement
theorem determine_constants : 
  (∀ x, g x = transformation_v_shift (transformation_hor_compression x)) → 
  a = 1 ∧ b = 1/3 ∧ c = -2 :=
by
  sorry

end determine_constants_l209_209109


namespace acute_triangle_area_relation_l209_209950

noncomputable def area_triangle (A B C : EuclideanSpace ℝ^2) := 
  1/2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def is_acute (A B C : EuclideanSpace ℝ^2) : Prop := 
  ∠ A B C < π/2 ∧ ∠ B C A < π/2 ∧ ∠ C A B < π/2

def lies_on_semicircle (P A B : EuclideanSpace ℝ^2) : Prop :=
  (P.x - (A.x + B.x) / 2)^2 + P.y^2 = ((A.x - B.x) / 2)^2

theorem acute_triangle_area_relation (A B C A' B' C' : EuclideanSpace ℝ^2)
  (h₀ : is_acute A B C)
  (h₁ : ∃ D, orthogonal_projection_line (line_through B C) A = D ∧ lies_on_semicircle A' B C ∧ A' = D)
  (h₂ : ∃ E, orthogonal_projection_line (line_through C A) B = E ∧ lies_on_semicircle B' C A ∧ B' = E)
  (h₃ : ∃ F, orthogonal_projection_line (line_through A B) C = F ∧ lies_on_semicircle C' A B ∧ C' = F):
  area_triangle B C A' ^ 2 + area_triangle C A B' ^ 2 + area_triangle A B C' ^ 2 = area_triangle A B C ^ 2 :=
sorry

end acute_triangle_area_relation_l209_209950


namespace angle_Q_of_regular_decagon_l209_209382

noncomputable def decagon := { 
  sides : ℕ // sides = 10 
}

def interior_angle_sum (n : ℕ) : ℕ := 180 * (n - 2)

def regular_polygon_interior_angle (n : ℕ) : ℕ := (interior_angle_sum n) / n

theorem angle_Q_of_regular_decagon (d : decagon) : 
  let n := d.sides in
  let interior_angle := regular_polygon_interior_angle n in
  let exterior_angle := 180 - interior_angle in
  let angle_BFQ := 360 - 2 * interior_angle in
  let angle_Q := 360 - (2 * exterior_angle + angle_BFQ) in
  angle_Q = 216 := 
by {
  have h1 : n = 10 := d.2,
  rw h1,
  have H_inter : interior_angle = (180 * 8 / 10) := rfl,
  have H_ext : exterior_angle = 180 - (144) := by rw H_inter; norm_num,
  have H_BFQ : angle_BFQ = 360 - 2 * 144 := by rw H_inter; norm_num,
  norm_num,
  sorry
}

end angle_Q_of_regular_decagon_l209_209382


namespace smallest_n_with_digits_437_l209_209789

theorem smallest_n_with_digits_437 (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_m_lt_n : m < n)
  (h_digits : ∃ k : ℕ, 1000 * m = 437 * n + k ∧ k < n) : n = 1809 :=
sorry

end smallest_n_with_digits_437_l209_209789


namespace sum_possible_values_of_p_l209_209344

theorem sum_possible_values_of_p (p q : ℤ) (h1 : p + q = 2010)
  (h2 : ∃ (α β : ℕ), (10 * α * β = q) ∧ (10 * (α + β) = -p)) :
  p = -3100 :=
by
  sorry

end sum_possible_values_of_p_l209_209344


namespace product_of_roots_l209_209549

theorem product_of_roots :
  let a := 4
  let d := -36
  let pqr := -d / a
  pqr = 9 :=
by
  let a := 4
  let d := -36
  let pqr := -d / a
  show pqr = 9 from sorry

end product_of_roots_l209_209549


namespace first_term_is_720_sqrt7_div_49_l209_209411

noncomputable def first_term_geometric_sequence (a r : ℝ) : Prop :=
  (a * r^3 = 720) ∧ (a * r^5 = 5040) ∧ (a = 720 * real.sqrt 7 / 49)

-- The theorem statement
theorem first_term_is_720_sqrt7_div_49 :
  ∃ a r : ℝ, first_term_geometric_sequence a r :=
begin
  use [720 * real.sqrt 7 / 49, real.sqrt 7],
  unfold first_term_geometric_sequence,
  split,
  { -- Prove a * r^3 = 720
    ring },
  split,
  { -- Prove a * r^5 = 5040
    ring },
  { -- Prove a = 720 * real.sqrt 7 / 49
    ring }
end

end first_term_is_720_sqrt7_div_49_l209_209411


namespace polynomial_remainder_l209_209462

def polynomial := (x : ℤ) : ℤ := x^11 + 1

theorem polynomial_remainder (x : ℤ) : 
    polynomial (-1) = 0 := 
by 
    sorry

end polynomial_remainder_l209_209462


namespace volume_of_one_slice_l209_209038

noncomputable def radius_of_pizza (diameter : ℝ) := diameter / 2

noncomputable def volume_of_cylinder (r : ℝ) (h : ℝ) := π * r^2 * h

theorem volume_of_one_slice {diameter thickness : ℝ} 
  (h_diameter : diameter = 12)
  (h_thickness : thickness = 1 / 2)
  (num_slices : ℝ) (h_slices : num_slices = 8) : 
  let r := radius_of_pizza diameter in
  let V := volume_of_cylinder r thickness in
  V / num_slices = (9 * π) / 4 := 
by
  -- defining intermediate values
  let r := radius_of_pizza diameter
  let V := volume_of_cylinder r thickness
  -- using assumptions to derive the required proof
  have h_r : r = 6, by rw [radius_of_pizza, h_diameter]; norm_num
  have h_V : V = 18 * π, by rw [volume_of_cylinder, h_r, h_thickness]; norm_num
  -- showing the final calculated volume per slice
  show V / num_slices = (9 * π) / 4, by rw [h_V, h_slices]; norm_num

end volume_of_one_slice_l209_209038


namespace count_odd_factors_of_360_l209_209225

-- Defining the prime factorization of 360
def prime_factorization_360 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

-- Defining the exponent range based on prime factorization for an odd factor
def exponent_ranges : List (ℕ × List ℕ) := [(3, [0, 1, 2]), (5, [0, 1])]

theorem count_odd_factors_of_360 : ∃ n : ℕ, n = 6 :=
by
  -- Conditions based on prime factorization
  have h₁ : prime_factorization_360 = [(2, 3), (3, 2), (5, 1)], from rfl
  have h₂ : exponent_ranges = [(3, [0, 1, 2]), (5, [0, 1])], from rfl
  
  -- Definitions should align with the conditions in the problem
  use 6
  sorry

end count_odd_factors_of_360_l209_209225


namespace boys_collected_200_insects_l209_209400

theorem boys_collected_200_insects
  (girls_insects : ℕ)
  (groups : ℕ)
  (insects_per_group : ℕ)
  (total_insects : ℕ)
  (boys_insects : ℕ)
  (H1 : girls_insects = 300)
  (H2 : groups = 4)
  (H3 : insects_per_group = 125)
  (H4 : total_insects = groups * insects_per_group)
  (H5 : boys_insects = total_insects - girls_insects) :
  boys_insects = 200 :=
  by sorry

end boys_collected_200_insects_l209_209400


namespace length_BD_in_right_triangle_l209_209691

theorem length_BD_in_right_triangle 
  (a : ℝ) (h1 : a^2 - 7 ≥ 0) 
  (BC AC AD : ℝ) 
  (h_BC : BC = 3) 
  (h_AC : AC = a) 
  (h_AD : AD = 4) :
  ∃ BD : ℝ, BD = sqrt (a^2 - 7) :=
by
  sorry

end length_BD_in_right_triangle_l209_209691


namespace area_of_square_with_adjacent_points_l209_209757

theorem area_of_square_with_adjacent_points :
  ∀ (x1 y1 x2 y2 : ℝ), x1 = 1 → y1 = 2 → x2 = 5 → y2 = 6 →
  let side_length := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) in
  let area := side_length^2 in
  area = 32 :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2
  let side_length := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  let area := side_length^2
  sorry

end area_of_square_with_adjacent_points_l209_209757


namespace S_11_eq_zero_l209_209959

noncomputable def S (n : ℕ) : ℝ := sorry
variable (a_n : ℕ → ℝ) (d : ℝ)
variable (h1 : ∀ n, a_n (n+1) = a_n n + d) -- common difference d ≠ 0
variable (h2 : S 5 = S 6)

theorem S_11_eq_zero (h_nonzero : d ≠ 0) : S 11 = 0 := by
  sorry

end S_11_eq_zero_l209_209959


namespace initial_pencils_sold_l209_209506

theorem initial_pencils_sold (x : ℕ) (P : ℝ)
  (h1 : 1 = 0.9 * (x * P))
  (h2 : 1 = 1.2 * (8.25 * P))
  : x = 11 :=
by sorry

end initial_pencils_sold_l209_209506


namespace odd_factors_of_360_l209_209174

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l209_209174


namespace piecewise_function_value_l209_209641

theorem piecewise_function_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = (if x ≥ 0 then 2^(x-2) - 3 else 3 * x)) →
  (a = 4 → ∃ x, x ≥ real.log 3 / real.log 2 + 2 ∧ f x = a) ∧
  (a = -1 → ∃ x, x = -1/3 ∧ f x = a) := by
  intros hfa
  apply and.intro
  {
    intros ha
    use real.log 3 / real.log 2 + 2
    split
    {
      linarith
    },
    {
      rw ha
      exact hfa (real.log 3 / real.log 2 + 2)
    }
  },
  {
    intros ha
    use -1 / 3
    split
    {
      exact rfl
    },
    {
      rw ha
      exact hfa (-1 / 3)
    }
  }

end piecewise_function_value_l209_209641


namespace range_of_x_l209_209272

theorem range_of_x (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -2) : x > 1/3 :=
by
  sorry

end range_of_x_l209_209272


namespace vasya_more_ways_l209_209367

-- Define the board types
def board_petya : Type := fin 100 × fin 50
def board_vasya : Type := fin 100 × fin 100

-- Define the condition for no kings attacking each other on a board
def non_attacking (kings : set (fin n × fin m)) :=
  ∀ k1 k2 ∈ kings, k1 ≠ k2 → ¬(k1.fst = k2.fst ∨ k1.snd = k2.snd 
    ∨ (k1.fst - k2.fst).nat_abs = (k1.snd - k2.snd).nat_abs)

-- Define the condition that kings are placed on the white cells of a checkerboard
def is_white_square (square : fin 100 × fin 100) :=
  (square.fst.val + square.snd.val) % 2 = 0

-- The sets of possible kings placements
def valid_kings_petya := {k : set board_petya | k.card = 500 ∧ non_attacking k}
def valid_kings_vasya := {k : set board_vasya | k.card = 500 ∧ non_attacking k 
  ∧ ∀ square ∈ k, is_white_square square}

-- Statement of the problem as a Lean theorem
theorem vasya_more_ways : (valid_kings_petya.card < valid_kings_vasya.card) :=
sorry

end vasya_more_ways_l209_209367


namespace area_triangle_APD_l209_209774

variable {ABCD : Type} [parallelogram ABCD]
variable {A P D Q B C : Point ABCD}
variable (area_ABCD : ℝ) (midpoint_P : midpoint P A B) (midpoint_Q : midpoint Q C D)

theorem area_triangle_APD (h1 : parallelogram ABCD)
                          (h2 : area ABCD = 48)
                          (h3 : midpoint P A B)
                          (h4 : midpoint Q C D) :
                          area (triangle A P D) = 24 :=
by
  sorry

end area_triangle_APD_l209_209774


namespace decreasing_function_gt_zero_l209_209741

-- Declaration of a function being decreasing
def is_decreasing (f : ℝ → ℝ) : Prop := 
  ∀ (x y : ℝ), x < y → f x > f y

-- Main theorem statement
theorem decreasing_function_gt_zero
  (f : ℝ → ℝ)
  (h_decreasing : is_decreasing f)
  (h_condition : ∀ (x : ℝ), (f x / (deriv f x) + x < 1)) :
  ∀ (x : ℝ), x ∈ (1 : ℝ, +∞) → f x > 0 :=
by
  sorry

end decreasing_function_gt_zero_l209_209741


namespace range_of_a_l209_209976

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) ↔ (0 ≤ a ∧ a ≤ 1) := 
sorry

end range_of_a_l209_209976


namespace neg_two_is_negative_rational_l209_209485

theorem neg_two_is_negative_rational : 
  (-2 : ℚ) < 0 ∧ ∃ (r : ℚ), r = -2 := 
by
  sorry

end neg_two_is_negative_rational_l209_209485


namespace vector_plane_decomposition_linear_combination_zero_implies_zero_l209_209671

open Real

variables {a : Type*} [AddCommGroup a] [VectorSpace ℝ a]
variables (e1 e2 : a) (non_collinear : ¬ collinear ℝ ({e1, e2} : set a))

theorem vector_plane_decomposition
  (x : a) :
  ∃ (λ μ : ℝ), x = λ • e1 + μ • e2 := sorry

theorem linear_combination_zero_implies_zero
  {λ μ : ℝ} :
  λ • e1 + μ • e2 = (0 : a) → λ = 0 ∧ μ = 0 := sorry

end vector_plane_decomposition_linear_combination_zero_implies_zero_l209_209671


namespace inequality_and_equality_l209_209732

theorem inequality_and_equality (x : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ i, 1 ≤ x i ∧ x i ≤ 2) (h2 : n ≥ 2) :
  (∑ i in finset.range n, |x i - x ((i + 1) % n)|) ≤ (2/3) * (∑ i in finset.range n, x i) ∧
  ( (∑ i in finset.range n, |x i - x ((i + 1) % n)| = (2/3) * (∑ i in finset.range n, x i)) ↔ 
    (n % 2 = 0 ∧ (∀ i, x i = 1 ∨ x i = 2) ∧ ∀ i, x i = if (i % 2 = 0) then 1 else 2 ∨ x i = if (i % 2 = 0) then 2 else 1) ) :=
sorry

end inequality_and_equality_l209_209732


namespace number_of_soda_cans_daily_l209_209091

/- Definitions corresponding to given conditions -/
def daily_water : ℕ := 64
def weekly_fluid : ℕ := 868
def soda_per_can : ℕ := 12
def days_per_week : ℕ := 7

/- Main theorem statement -/
theorem number_of_soda_cans_daily : 
  let weekly_water := daily_water * days_per_week in
  let weekly_soda := weekly_fluid - weekly_water in
  let weekly_soda_cans := weekly_soda / soda_per_can in
  let daily_soda_cans := weekly_soda_cans / days_per_week in
  daily_soda_cans = 5 :=
by
  sorry

end number_of_soda_cans_daily_l209_209091


namespace smallest_base_conversion_l209_209047

def base3_to_decimal (n : ℕ) : ℕ :=
  2 + 0 * 3 + 0 * 9 + 1 * 27 -- 1002 base 3

def base6_to_decimal (n : ℕ) : ℕ :=
  0 + 1 * 6 + 2 * 36 -- 210 base 6

def base4_to_decimal (n : ℕ) : ℕ :=
  4 ^ 3 -- 1000 base 4

def base2_to_decimal (n : ℕ) : ℕ :=
  2 ^ 6 - 1 -- 111111 base 2

theorem smallest_base_conversion :
  ∃ n, n = 1002 ∧ base3_to_decimal n < base6_to_decimal n ∧
       base3_to_decimal n < base4_to_decimal n ∧
       base3_to_decimal n < base2_to_decimal n :=
begin
  sorry
end

end smallest_base_conversion_l209_209047


namespace odd_factors_360_l209_209214

theorem odd_factors_360 : 
  let fac := (2^3) * (3^2) * (5^1) in
  fac = 360 → 
  num_factors (3^2 * 5^1) = 6 :=
sorry

end odd_factors_360_l209_209214


namespace tangent_line_to_parabola_parallel_l209_209408

theorem tangent_line_to_parabola_parallel
  (x y : ℝ) :
  let parabola := λ x, x^2
  let parallel_line (x y : ℝ) := 2 * x - y + 4 = 0
  ∃ (l : ℝ → ℝ × ℝ → Prop), ∀ (tangent_point : ℝ × ℝ),
    (tangent_point = (1, 1) ∧ (∀ x, l x tangent_point = 2 * x - y - 1 = 0)) ∧
    ∀ (p1 p2: ℝ), p1 = p2 → (2 * x - y + 4 = 0) → (2 * x - tan(y, x^2, p2) + 4 = 0) :=
by
  sorry

end tangent_line_to_parabola_parallel_l209_209408


namespace selling_price_of_article_l209_209533

theorem selling_price_of_article (CP : ℝ) (L_percent : ℝ) (SP : ℝ) 
  (h1 : CP = 600) 
  (h2 : L_percent = 50) 
  : SP = 300 := 
by
  sorry

end selling_price_of_article_l209_209533


namespace max_passengers_l209_209430

theorem max_passengers (total_stops : ℕ) (bus_capacity : ℕ)
  (h_total_stops : total_stops = 12) 
  (h_bus_capacity : bus_capacity = 20) 
  (h_no_same_stop : ∀ (a b : ℕ), a ≠ b → (a < total_stops) → (b < total_stops) → 
    ∃ x y : ℕ, x ≠ y ∧ x < total_stops ∧ y < total_stops ∧ 
    ((x = a ∧ y ≠ a) ∨ (x ≠ b ∧ y = b))) :
  ∃ max_passengers : ℕ, max_passengers = 50 :=
  sorry

end max_passengers_l209_209430


namespace parallelogram_identity_l209_209822

variable {A B C D E : Type}

-- Given
variables [is_parallelogram ABCD] [incircle A B D K] [line_intersect E A C K]

theorem parallelogram_identity (h₁ : is_parallelogram ABCD) (h₂ : incircle A B D K) (h₃ : line_intersect E A C K) :
  (AB * AB + AD * AD) = (AC * AE) :=
sorry

end parallelogram_identity_l209_209822


namespace least_number_of_grapes_is_106_l209_209501

theorem least_number_of_grapes_is_106 :
  ∃ n : ℕ, n % 3 = 1 ∧ n % 5 = 1 ∧ n % 7 = 1 ∧ ∀ m : ℕ, (m % 3 = 1 ∧ m % 5 = 1 ∧ m % 7 = 1 → m ≥ n) :=
by
  use 106
  split; [refl, split; [refl, split; [refl, intro m, sorry]]]

end least_number_of_grapes_is_106_l209_209501


namespace count_positive_even_multiples_of_3_less_than_5000_perfect_squares_l209_209234

theorem count_positive_even_multiples_of_3_less_than_5000_perfect_squares :
  ∃ n : ℕ, (n = 11) ∧ ∀ k : ℕ, (k < 5000) → (k % 2 = 0) → (k % 3 = 0) → (∃ m : ℕ, k = m * m) → k ≤ 36 * 11 * 11 :=
by {
  sorry
}

end count_positive_even_multiples_of_3_less_than_5000_perfect_squares_l209_209234


namespace ball_probability_correct_l209_209851

noncomputable def ball_probability : ℚ := 
  let total_balls := 4 + 5 + 6 + 3 in
  let total_ways := Mathlib.Finset.card (Mathlib.Finset.powersetLen 4 (Finset.range total_balls)) in
  let green_ways := Mathlib.Finset.card (Mathlib.Finset.powersetLen 2 (Finset.range 6)) in
  let red_ways := Mathlib.Finset.card (Mathlib.Finset.powersetLen 1 (Finset.range 4)) in
  let blue_ways := Mathlib.Finset.card (Mathlib.Finset.powersetLen 1 (Finset.range 3)) in
  (green_ways * red_ways * blue_ways) / total_ways

theorem ball_probability_correct :
  ball_probability = 1 / 17 := sorry

end ball_probability_correct_l209_209851


namespace polynomial_degree_l209_209509

noncomputable def num_roots (n : ℕ) : ℕ :=
   if ∃ k : ℕ, k * k = n + 1 then 1 else 2

theorem polynomial_degree :
  ∀ (p : polynomial ℚ),
  (∀ n ∈ {1, 2, ..., 1000}, (n + (real.sqrt (n + 1)) : ℚ)) ∈ p.roots → 
  p.degree = 1970 :=
begin
  sorry
end

end polynomial_degree_l209_209509


namespace probability_of_a_plus_b_gt_5_l209_209777

noncomputable def all_events : Finset (ℕ × ℕ) := 
  { (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4) }

noncomputable def successful_events : Finset (ℕ × ℕ) :=
  { (2, 4), (3, 3), (3, 4) }

theorem probability_of_a_plus_b_gt_5 : 
  (successful_events.card : ℚ) / (all_events.card : ℚ) = 1 / 3 := by
  sorry

end probability_of_a_plus_b_gt_5_l209_209777


namespace number_of_positive_area_triangles_l209_209249

theorem number_of_positive_area_triangles (s : Finset (ℕ × ℕ)) (h : s = {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}) :
  (s.powerset.filter λ t, t.card = 3 ∧ ¬∃ a b c, a = b ∨ b = c ∨ c = a).card = 2160 :=
by {
  sorry
}

end number_of_positive_area_triangles_l209_209249


namespace piano_cost_is_500_l209_209320

noncomputable def piano_cost (lesson_cost_per_lesson : ℕ) (number_of_lessons : ℕ) (discount_rate : ℚ) (total_cost : ℕ) : ℕ :=
  let original_lesson_cost := lesson_cost_per_lesson * number_of_lessons
  let discount := discount_rate * original_lesson_cost
  let discounted_lesson_cost := original_lesson_cost - discount
  total_cost - discounted_lesson_cost

theorem piano_cost_is_500 :
  piano_cost 40 20 0.25 1100 = 500 :=
by
  sorry

end piano_cost_is_500_l209_209320


namespace bisectors_intersect_on_midpoint_line_l209_209345

structure Point := (x : ℝ) (y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)
  (isConvexInscribable : Prop)

def intersect (p1 p2 p3 p4 : Point) : Point := sorry

def midpoint (p1 p2 : Point) : Point := sorry

theorem bisectors_intersect_on_midpoint_line (Q : Quadrilateral)
  (H1 : Q.isConvexInscribable)
  (P : Point)
  (H2 : P = intersect Q.A Q.B Q.C Q.D)
  (Q_point : Point)
  (H3 : Q_point = intersect Q.B Q.C Q.D Q.A)
  : let M_AC := midpoint Q.A Q.C
    let M_BD := midpoint Q.B Q.D
    in sorry

end bisectors_intersect_on_midpoint_line_l209_209345


namespace sum_first_k_plus_2_terms_l209_209794

open_locale nat

theorem sum_first_k_plus_2_terms (k : ℕ) :
  let a₁ := k^2 - k + 1,
      d := 1,
      n := k + 2,
      a_n := a₁ + (n - 1) * d 
  in (n / 2) * (a₁ + a_n) = k^3 + 2k^2 + k + 2 := by
  sorry

end sum_first_k_plus_2_terms_l209_209794


namespace profit_percent_eq_20_l209_209050

-- Define cost price 'C' and original selling price 'S'
variable (C S : ℝ)

-- Hypothesis: selling at 2/3 of the original price results in a 20% loss 
def condition (C S : ℝ) : Prop :=
  (2 / 3) * S = 0.8 * C

-- Main theorem: profit percent when selling at the original price is 20%
theorem profit_percent_eq_20 (C S : ℝ) (h : condition C S) : (S - C) / C * 100 = 20 :=
by
  -- Proof steps would go here but we use sorry to indicate the proof is omitted
  sorry

end profit_percent_eq_20_l209_209050


namespace cover_circle_with_two_F_l209_209946

theorem cover_circle_with_two_F (F : set Point) (R : ℝ) 
  (h1 : convex F)
  (h2 : ¬ (∃ f : (set Point), f ≈ F ∧ ⊆ semicircle R)) :
  ∃ f1 f2 : set Point, f1 ≈ F ∧ f2 ≈ F ∧ (cover_circle R (f1 ∪ f2)) := 
sorry

end cover_circle_with_two_F_l209_209946


namespace sum_of_sequence_l209_209981

variable {n : ℕ} (hn : 0 < n)

def a (n : ℕ) : ℝ := (n * 2^n - 2^(n+1)) / ((n+1) * (n^2 + 2*n))

def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, a k.succ)

theorem sum_of_sequence (hn : 0 < n) : S n = (2^(n+1)) / ((n+1) * (n+2)) - 1 :=
sorry

end sum_of_sequence_l209_209981


namespace exists_triangle_with_properties_l209_209076

theorem exists_triangle_with_properties (c r r' : ℝ) :
  ∃ (A B C : Point), 
    dist A B = c ∧ 
    -- Let r and r' be the radii of the specified circles as described in the problem
    ∃ (incircle : Circle), 
      radius incircle = r ∧ 
      incircle tangent_to AB ∧
    ∃ (other_circle : Circle), 
      radius other_circle = r' ∧ 
      other_circle tangent_to AB ∧ 
      other_circle tangent_to_line BC ∧ 
      other_circle tangent_to_line CA :=
sorry

end exists_triangle_with_properties_l209_209076


namespace limit_n_b_n_l209_209078

-- Define the function M(x)
def M (x : ℝ) : ℝ := x - (x^2) / 2

-- Define b_n as the n-th iteration of M applied to (20/n)
def b_n (n : ℕ) : ℝ :=
  (nat.recOn n (20 / n) (λ k x, M x)).tail

-- Define the problem statement for proving the limit as n approaches infinity
theorem limit_n_b_n : 
  filter.tendsto (λ (n : ℕ), n * b_n n) filter.at_top (nhds (20 / 11)) :=
sorry

end limit_n_b_n_l209_209078


namespace count_odd_factors_of_360_l209_209223

-- Defining the prime factorization of 360
def prime_factorization_360 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

-- Defining the exponent range based on prime factorization for an odd factor
def exponent_ranges : List (ℕ × List ℕ) := [(3, [0, 1, 2]), (5, [0, 1])]

theorem count_odd_factors_of_360 : ∃ n : ℕ, n = 6 :=
by
  -- Conditions based on prime factorization
  have h₁ : prime_factorization_360 = [(2, 3), (3, 2), (5, 1)], from rfl
  have h₂ : exponent_ranges = [(3, [0, 1, 2]), (5, [0, 1])], from rfl
  
  -- Definitions should align with the conditions in the problem
  use 6
  sorry

end count_odd_factors_of_360_l209_209223


namespace evaluate_replacement_at_third_l209_209706

theorem evaluate_replacement_at_third :
  let x := (1 : ℝ) / 3
  in ( 
    ((x + 3) / (x - 3) + 3) / ((x + 3) / (x - 3) - 3)
  ) = -2 :=
by
  let x := (1 : ℝ) / 3
  show ((x + 3) / (x - 3) + 3) / ((x + 3) / (x - 3) - 3) = -2
  sorry

end evaluate_replacement_at_third_l209_209706


namespace probability_of_two_white_balls_probability_of_at_least_one_white_ball_l209_209490

noncomputable def probability_two_white_balls : ℚ :=
  let total_balls := 16
  let white_balls := 8
  let total_ways := (total_balls.choose 2)
  let white_ways := (white_balls.choose 2)
  (white_ways : ℚ) / total_ways

noncomputable def probability_at_least_one_white_ball : ℚ :=
  1 - probability_two_black_balls

noncomputable def probability_two_black_balls : ℚ :=
  let black_balls := 8
  let total_balls := 16
  let total_ways := (total_balls.choose 2)
  let black_ways := (black_balls.choose 2)
  (black_ways : ℚ) / total_ways

theorem probability_of_two_white_balls :
  (probability_two_white_balls = 7 / 30) := by
    sorry

theorem probability_of_at_least_one_white_ball :
  (probability_at_least_one_white_ball = 23 / 30) := by
    sorry

end probability_of_two_white_balls_probability_of_at_least_one_white_ball_l209_209490


namespace shaded_area_infinite_series_l209_209419

-- We are defining all necessary conditions from the problem statement.
def is_equilateral (a b c : ℝ) := a = b ∧ b = c

def equilateral_triangle_area (s : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * s^2

-- We'll interpret the infinite geometric series of the shaded areas in the Lean theorem.
theorem shaded_area_infinite_series 
  (base_area : ℝ)                -- condition: area of the largest triangle.
  (shaded_area_ratio : ℝ)        -- condition: ratio of areas of each set of shaded triangles to the previous one.
  (geo_series_sum : ℝ) :         -- conclusion: the infinite sum of the geometric series.

  base_area = (Real.sqrt 3 / 4) ∧         -- base of the largest triangle is 1
  shaded_area_ratio = 1/4 ∧               -- Each set of shaded triangles' side length is 1/2 of the previous set.
  geo_series_sum = (Real.sqrt 3 / 4 - 1) -- pattern continues infinitely with a common ratio 1/4.

  → geo_series_sum = (Real.sqrt 3 / 4) - base_area :=
sorry

end shaded_area_infinite_series_l209_209419


namespace find_original_number_l209_209102

theorem find_original_number (x : ℝ) (h : 0.5 * x = 30) : x = 60 :=
sorry

end find_original_number_l209_209102


namespace fraction_inequality_l209_209141

theorem fraction_inequality 
  (a b x y : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h1 : 1 / a > 1 / b)
  (h2 : x > y) : 
  x / (x + a) > y / (y + b) := 
  sorry

end fraction_inequality_l209_209141


namespace correlation_sign_switch_l209_209772

variable {X : ℝ → ℝ}
variable {R_x_dot_x : ℝ × ℝ → ℝ}
variable {k_x : ℝ → ℝ}

-- Given conditions
variable (cond1 : ∀ t1 t2 : ℝ, R_x_dot_x t1 t2 = k_x (t2 - t1))
variable (cond2 : ∀ τ : ℝ, differentiable ℝ (λ τ, k_x τ))

-- Prove that the mutual correlation function of a stationary stochastic process
-- X(t) and its derivative changes sign when switching the arguments t1 and t2.
theorem correlation_sign_switch (t1 t2 : ℝ) : R_x_dot_x t1 t2 = -R_x_dot_x t2 t1 :=
by
  sorry

end correlation_sign_switch_l209_209772


namespace speed_of_first_car_l209_209444

theorem speed_of_first_car (v : ℝ) 
  (h1 : ∀ v, v > 0 → (first_speed = 1.25 * v))
  (h2 : 720 = (v + 1.25 * v) * 4) : 
  first_speed = 100 := 
by
  sorry

end speed_of_first_car_l209_209444


namespace joe_average_score_l209_209717

theorem joe_average_score (A B C : ℕ) (lowest_score : ℕ) (final_average : ℕ) :
  lowest_score = 45 ∧ final_average = 65 ∧ (A + B + C) / 3 = final_average →
  (A + B + C + lowest_score) / 4 = 60 := by
  sorry

end joe_average_score_l209_209717


namespace union_sets_l209_209984

noncomputable def P (x : ℝ) : Set ℝ := { real.log 4 / real.log (2 * x), 3 }
def Q (x y : ℝ) : Set ℝ := { x, y }

theorem union_sets (x y : ℝ) (h₁ : P x ∩ Q x y = {2}) : 
  P x ∪ Q x y = {1, 2, 3} :=
by 
  sorry

end union_sets_l209_209984


namespace find_p_plus_q_l209_209291

/--
In \(\triangle{XYZ}\), \(XY = 12\), \(\angle{X} = 45^\circ\), and \(\angle{Y} = 60^\circ\).
Let \(G, E,\) and \(L\) be points on the line \(YZ\) such that \(XG \perp YZ\), 
\(\angle{XYE} = \angle{EYX}\), and \(YL = LY\). Point \(O\) is the midpoint of 
the segment \(GL\), and point \(Q\) is on ray \(XE\) such that \(QO \perp YZ\).
Prove that \(XQ^2 = \dfrac{81}{2}\) and thus \(p + q = 83\), where \(p\) and \(q\) 
are relatively prime positive integers.
-/
theorem find_p_plus_q :
  ∃ (p q : ℕ), gcd p q = 1 ∧ XQ^2 = 81 / 2 ∧ p + q = 83 :=
sorry

end find_p_plus_q_l209_209291


namespace centers_circumcircles_form_similar_triangle_l209_209767

theorem centers_circumcircles_form_similar_triangle
  (A B C P Q R : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] 
  [Inhabited P] [Inhabited Q] [Inhabited R]
  (triangle_ABC : Triangle A B C)
  (P_on_AB : P ∈ segment A B)
  (Q_on_BC : Q ∈ segment B C)
  (R_on_CA : R ∈ segment C A) :
  let circumcenter_APR := circumcenter (triangle_ABC.expand_to P R)
  let circumcenter_BPQ := circumcenter (triangle_ABC.expand_to Q P)
  let circumcenter_CQR := circumcenter (triangle_ABC.expand_to R Q)
  similar (triangle_ABC.circumcenters) (triangle_ABC) :=
by
  sorry

end centers_circumcircles_form_similar_triangle_l209_209767


namespace odd_factors_of_360_l209_209229

theorem odd_factors_of_360 : ∃ n, 360 = 2^3 * 3^2 * 5^1 ∧ n = 6 :=
by 
  -- Define the prime factorization of 360
  let pf360 := (2^3 * 3^2 * 5^1)
  
  -- Define the count of odd factors by removing the contribution of factor 2.
  let odd_part := (3^2 * 5^1)
  
  -- Calculate the number of factors.
  have odd_factors_count_eq : (2 + 1) * (1 + 1) = 6 := by 
    calc (2 + 1) * (1 + 1) = 3 * 2 : by rfl
                         ... = 6 : by rfl
  
  -- Assert the existence of such n matching the count of odd factors.
  exact ⟨6, And.intro rfl odd_factors_count_eq⟩

end odd_factors_of_360_l209_209229


namespace sergei_distance_from_andrei_l209_209052

theorem sergei_distance_from_andrei
  (a b c : ℝ) -- speeds of Andrei, Boris, and Sergei in meters per second
  (h1 : b * (100 / a) = 90) -- Boris was 10 meters behind when Andrei finished
  (h2 : c * (100 / b) = 90) -- Sergei was 62 meters behind when Boris finished
  : 100 - (c * (100 / a)) = 19 := -- Prove Sergei's distance from Andrei when Andrei finished is 19 meters
begin
  sorry
end

end sergei_distance_from_andrei_l209_209052


namespace complement_set_l209_209686

def U : Set ℝ := Set.univ
def M : Set ℝ := {y | ∃ x : ℝ, 0 < x ∧ x < 1 ∧ y = Real.log x / Real.log 2}

theorem complement_set :
  Set.compl M = {y : ℝ | y ≥ 0} :=
by
  sorry

end complement_set_l209_209686


namespace abs_eq_5_iff_l209_209677

theorem abs_eq_5_iff (a : ℝ) : |a| = 5 ↔ a = 5 ∨ a = -5 :=
by
  sorry

end abs_eq_5_iff_l209_209677


namespace solution_set_l209_209972

def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 + 1 else 3 - x

theorem solution_set :
  { x : ℝ | f x >= 2*x^2 - 3 } = set.Icc (-2 : ℝ) 2 :=
by
  sorry

end solution_set_l209_209972


namespace relationship_l209_209961

variable {a b c d : ℝ}

-- Defining our given conditions
def cond1 : Prop := a < b
def cond2 : Prop := d < c
def cond3 : Prop := (c - a) * (c - b) < 0
def cond4 : Prop := (d - a) * (d - b) > 0

-- The statement to prove
theorem relationship (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : d < a ∧ a < c ∧ c < b :=
sorry

end relationship_l209_209961


namespace value_of_m_explicit_formula_of_f_range_of_k_l209_209625

noncomputable def f (x : ℝ) :=
  if x >= 0 then 2^x + x - 1 else -2^(-x) + x + 1

theorem value_of_m (x : ℝ) :
  (f(0) = 0) -> (∃ m : ℝ, f(x) = 2^x + x - m ∀ x ∈ [0,∞)) ∧ f(-x) = -f(x) :=
sorry

theorem explicit_formula_of_f (x : ℝ) :
  (f(x) = if x >= 0 then 2^x + x - 1 else -2^(-x) + x + 1)
  ∧ (m = 1) :=
sorry

theorem range_of_k (k : ℝ) :
  (∀ x ∈ [-3, -2], f(k * 4^x) + f(1 - 2^(x+1)) > 0) -> k > -8 :=
sorry

end value_of_m_explicit_formula_of_f_range_of_k_l209_209625


namespace number_of_odd_factors_of_360_l209_209204

theorem number_of_odd_factors_of_360 : 
  let factors (n : ℕ) := {d : ℕ | d ∣ n}
  let odd (d : ℕ) := d % 2 = 1
  let is_factor (n : ℕ) (d : ℕ) := d ∣ n
  let number_of_odd_factors (n : ℕ) := (factors n).count odd
  in number_of_odd_factors 360 = 6 :=
begin
  sorry
end

end number_of_odd_factors_of_360_l209_209204


namespace sum_even_if_squares_even_l209_209271

theorem sum_even_if_squares_even (n m : ℤ) (h : even (n^2 + m^2)) : ¬ odd (n + m) :=
by sorry

end sum_even_if_squares_even_l209_209271


namespace prob_sum_is_10_prob_term_index_distance_even_l209_209379

open Finset

def set_A : Finset ℕ := {1, 2, 3, 4, 5}

def is_ordered_triple (a b c : ℕ) : Prop := a < b ∧ b < c

def sum_is_10 (a b c : ℕ) : Prop := a + b + c = 10

def term_index_distance (a b c : ℕ) : ℕ := |a - 1| + |b - 2| + |c - 3|

def even (n : ℕ) : Prop := n % 2 = 0

theorem prob_sum_is_10 : (∑ t in (set_A.product (set_A.product set_A)).filter (λ t, is_ordered_triple t.1 t.2.1 t.2.2 ∧ sum_is_10 t.1 t.2.1 t.2.2), 1) / (∑ t in (set_A.product (set_A.product set_A)).filter (λ t, is_ordered_triple t.1 t.2.1 t.2.2), 1)  = 1 / 5 := 
by sorry

theorem prob_term_index_distance_even : (∑ t in (set_A.product (set_A.product set_A)).filter (λ t, is_ordered_triple t.1 t.2.1 t.2.2 ∧ even (term_index_distance t.1 t.2.1 t.2.2)), 1) / (∑ t in (set_A.product (set_A.product set_A)).filter (λ t, is_ordered_triple t.1 t.2.1 t.2.2), 1) = 3 / 5 := 
by sorry

end prob_sum_is_10_prob_term_index_distance_even_l209_209379


namespace find_n_l209_209672

variable (n w : ℕ)
hypothesis (h1 : w > 0)
hypothesis (h2 : 2^5 ∣ n * w)
hypothesis (h3 : 3^3 ∣ n * w)
hypothesis (h4 : 13^2 ∣ n * w)
hypothesis (h5 : w = 156)

theorem find_n : n = 936 := sorry

end find_n_l209_209672


namespace larger_perimeter_perimeter_bounds_l209_209002

-- Conditions for Part (a)
variables {A B C A1 B1 C1 : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space A1] [metric_space B1] [metric_space C1]
variables (BC B1C1 : ℝ) (α : ℝ)
variables (tri_ABC tri_A1B1C1 : A × B × C)
variables (angle_ABC : triangle (tri_ABC))
variables (angle_A1B1C1 : triangle (tri_A1B1C1))

-- Mathematically equivalent statement for Part (a)
theorem larger_perimeter
    (h1 : BC = B1C1)
    (h2 : angle_ABC = α)
    (h3 : ∀ β: ℝ, |angle_ABC - angle_ABC.base| < |angle_A1B1C1 - angle_A1B1C1.base|) : 
    perimeter (tri_ABC) > perimeter (tri_A1B1C1) := 
by
  sorry

-- Conditions for Part (b)
variable (a : ℝ)
variable (α : ℝ)
variable (triangle_ABC : triangle (A × B × C))

-- Mathematically equivalent statement for Part (b)
theorem perimeter_bounds 
    (h1 : BC = a)
    (h2 : angle_ABC = α) :
    2a < 2 * perimeter triangle_ABC ∧ 2 * perimeter triangle_ABC ≤ a * (1 + csc (α / 2)) :=
by
  sorry

end larger_perimeter_perimeter_bounds_l209_209002


namespace speed_of_current_is_correct_l209_209504

noncomputable def speed_in_still_water : ℝ := 15 -- kmph
noncomputable def time_downstream : ℝ := 5.999520038396929 -- seconds
noncomputable def distance_downstream : ℝ := 30 -- meters
noncomputable def speed_of_current := (distance_downstream / time_downstream) - (speed_in_still_water * 1000 / 3600) -- meters/second

theorem speed_of_current_is_correct :
  (speed_of_current * 3600 / 1000) ≈ 2.99984 :=
by
  unfold speed_of_current
  have h : speed_in_still_water * 1000 / 3600 = 4.16666667 := sorry
  calc
    (distance_downstream / time_downstream - 4.16666667) * 3600 / 1000 = 2.99984 : sorry

end speed_of_current_is_correct_l209_209504


namespace train_speed_km_per_hr_l209_209850

theorem train_speed_km_per_hr
  (train_length : ℝ) 
  (platform_length : ℝ)
  (time_seconds : ℝ) 
  (h_train_length : train_length = 470) 
  (h_platform_length : platform_length = 520) 
  (h_time_seconds : time_seconds = 64.79481641468682) :
  (train_length + platform_length) / time_seconds * 3.6 = 54.975 := 
sorry

end train_speed_km_per_hr_l209_209850


namespace num_values_g_g_eq_4_l209_209558

def g (x : ℝ) : ℝ :=
if x ≥ -2 then (x + 1) ^ 2 - 3 else x + 2

theorem num_values_g_g_eq_4 : (finset.filter (λ x, g (g x) = 4) (finset.range 2000)).card = 2 :=
sorry

end num_values_g_g_eq_4_l209_209558


namespace smallest_value_between_0_and_1_l209_209275

theorem smallest_value_between_0_and_1 (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < y ∧ y^3 < 3 * y ∧ y^3 < y^(1/3 : ℝ) ∧ y^3 < 1 ∧ y^3 < 1 / y :=
by
  sorry

end smallest_value_between_0_and_1_l209_209275


namespace max_balloons_l209_209761

theorem max_balloons (p : ℕ) (h : 40 * p) : 
  ∃ n : ℕ, n = 52 :=
by
  sorry

end max_balloons_l209_209761


namespace inverse_expression_value_l209_209270

section
variable (i : ℂ)
hypothesis (hi_squared : i^2 = -1)

theorem inverse_expression_value :
  (i^3 - i^(-3))⁻¹ = (i / 2) :=
sorry
end

end inverse_expression_value_l209_209270


namespace cat_head_start_15_minutes_l209_209441

theorem cat_head_start_15_minutes :
  ∀ (t : ℕ), (25 : ℝ) = (20 : ℝ) * (1 + (t : ℝ) / 60) → t = 15 := by
  sorry

end cat_head_start_15_minutes_l209_209441


namespace a_n_eq_n_l209_209143

theorem a_n_eq_n (a : ℕ → ℝ) (h_pos : ∀ (n : ℕ), a n > 0) 
  (h_eq : ∀ (n : ℕ), ∑ i in Finset.range (n + 1), (a i) ^ 3 = (∑ i in Finset.range n, a (i + 1)) ^ 3) : 
  ∀ n : ℕ, a n = n :=
by
  sorry

end a_n_eq_n_l209_209143


namespace sequence_explicit_form_l209_209311

noncomputable def a_sequence : ℕ → ℝ
| 0       := 2 * Real.sqrt 3
| (n + 1) := (4 * a_sequence n) / (4 - (a_sequence n)^2)

theorem sequence_explicit_form (n : ℕ) :
  a_sequence n = 2 * Real.tan (Real.pi / (3 * 2^n)) :=
sorry

end sequence_explicit_form_l209_209311


namespace paths_ratio_l209_209361

theorem paths_ratio (m k : ℕ) (h : k > 0) : 
  let n := k * m in
  (nat.choose (n - 1 + m) m) = k * (nat.choose ((k * m) + m - 1) (m - 1)) :=
sorry

end paths_ratio_l209_209361


namespace area_of_triangle_proof_l209_209586

def area_of_triangle_sides_median (a b m : ℝ) : Prop :=
  ∃ (area : ℝ), a = 1 ∧ b = sqrt 15 ∧ m = 1 ∧ area = sqrt 15 / 2

theorem area_of_triangle_proof :
  area_of_triangle_sides_median 1 (sqrt 15) 1 := 
sorry

end area_of_triangle_proof_l209_209586


namespace sequence_explicit_form_l209_209310

noncomputable def a_sequence : ℕ → ℝ
| 0       := 2 * Real.sqrt 3
| (n + 1) := (4 * a_sequence n) / (4 - (a_sequence n)^2)

theorem sequence_explicit_form (n : ℕ) :
  a_sequence n = 2 * Real.tan (Real.pi / (3 * 2^n)) :=
sorry

end sequence_explicit_form_l209_209310


namespace number_of_odd_factors_of_360_l209_209199

def is_odd_factor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ ∀ k, 2 ∣ k → ¬ (k ∣ d)

theorem number_of_odd_factors_of_360 : 
  {d : ℕ // is_odd_factor 360 d}.card = 6 := 
sorry

end number_of_odd_factors_of_360_l209_209199


namespace AZ_perpendicular_BC_l209_209728

-- Define the required points and their properties
variable (A B C I D X Y Z : Point)
variable (DX: Line)
variable (DY: Line)
variable (XY: Line)
variable (AZ: Line)
variable (BC: Line)

-- Define triangles and angle properties
variable (triangle_ABC : Triangle A B C)

-- Define conditions
axiom incenter_I : incenter triangle_ABC I
axiom angle_A : measure (∠ A) = 60
axiom tangent_D : tangent (Incircle triangle_ABC) BC D
axiom DX_perpendicular_AB : Perpendicular (LineThrough D X) (LineThrough A B)
axiom DY_perpendicular_AC : Perpendicular (LineThrough D Y) (LineThrough A C)
axiom XYZ_equilateral : (Equilateral (Triangle X Y Z)) 
axiom same_half_plane_I_Z : same_half_plane (LineThrough X Y) I Z

-- Prove that AZ is perpendicular to BC
theorem AZ_perpendicular_BC : Perpendicular (AZ) (BC) :=
  sorry

end AZ_perpendicular_BC_l209_209728


namespace count_odd_factors_of_360_l209_209219

-- Defining the prime factorization of 360
def prime_factorization_360 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

-- Defining the exponent range based on prime factorization for an odd factor
def exponent_ranges : List (ℕ × List ℕ) := [(3, [0, 1, 2]), (5, [0, 1])]

theorem count_odd_factors_of_360 : ∃ n : ℕ, n = 6 :=
by
  -- Conditions based on prime factorization
  have h₁ : prime_factorization_360 = [(2, 3), (3, 2), (5, 1)], from rfl
  have h₂ : exponent_ranges = [(3, [0, 1, 2]), (5, [0, 1])], from rfl
  
  -- Definitions should align with the conditions in the problem
  use 6
  sorry

end count_odd_factors_of_360_l209_209219


namespace cylinder_heights_relationship_l209_209446

variables {r1 r2 h1 h2 : ℝ}

theorem cylinder_heights_relationship
    (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
    (radius_relation : r2 = 1.2 * r1) :
    h1 = 1.44 * h2 :=
by sorry

end cylinder_heights_relationship_l209_209446


namespace max_value_of_function_l209_209011

theorem max_value_of_function : ∀ x : ℝ, (0 < x ∧ x < 1) → x * (1 - x) ≤ 1 / 4 :=
sorry

end max_value_of_function_l209_209011


namespace odd_factors_360_l209_209182

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l209_209182


namespace distinct_roots_of_transformed_polynomial_l209_209317

theorem distinct_roots_of_transformed_polynomial
  (a b c : ℝ)
  (h : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
                    (a * x^5 + b * x^4 + c = 0) ∧ 
                    (a * y^5 + b * y^4 + c = 0) ∧ 
                    (a * z^5 + b * z^4 + c = 0)) :
  ∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
               (c * u^5 + b * u + a = 0) ∧ 
               (c * v^5 + b * v + a = 0) ∧ 
               (c * w^5 + b * w + a = 0) :=
  sorry

end distinct_roots_of_transformed_polynomial_l209_209317


namespace tan_theta_eq_neg_five_twelves_l209_209608

theorem tan_theta_eq_neg_five_twelves (m θ : ℝ) 
  (h1 : sin θ = (m - 3)/(m + 5)) 
  (h2 : cos θ = (4 - 2m)/(m + 5)) 
  (h3 : (π/2) < θ ∧ θ < π) : 
  tan θ = -5/12 :=
by sorry

end tan_theta_eq_neg_five_twelves_l209_209608


namespace points_concyclic_l209_209301

-- Define the structure of a triangle
structure Triangle (A B C : Type) :=
  (isosceles : A = B)
  (sides : A < C)
  (inequality : B < C)

-- Define the points D, P, Q with their properties
variables {A B C D P Q : Type}

-- Define the isosceles triangle ABC
axiom ABC : Triangle A B C

-- Define the condition DA = DB + DC
axiom condition_D : (D = A + (B + C))

-- Define the properties of perpendicular bisector and external angle bisector at point P
axiom perpendicular_bisector_AB : (P = (B + B) / 2)
axiom external_angle_bisector_ADB : (angle P = (angle ADB) + (angle ADB))

-- Define the properties of perpendicular bisector and external angle bisector at point Q
axiom perpendicular_bisector_AC : (Q = (C + C) / 2)
axiom external_angle_bisector_ADC : (angle Q = (angle ADC) + (angle ADC))

-- The main theorem
theorem points_concyclic : ∀ (B C P Q : Type), Triangle A B C → D = A + (B + C) → P = (B + B) / 2 → angle P = (angle ADB) + (angle ADB) → Q = (C + C) / 2 → angle Q = (angle ADC) + (angle ADC) → cyclic [B, C, P, Q] :=
by {
  sorry -- Proof to be filled in
}

end points_concyclic_l209_209301


namespace altitude_eq_point_slope_form_median_eq_gen_form_l209_209969

noncomputable def A : ℝ × ℝ := (-2, -1)
noncomputable def B : ℝ × ℝ := (2, 1)
noncomputable def C : ℝ × ℝ := (1, 3)

theorem altitude_eq_point_slope_form :
  let k_AB := (B.2 - A.2) / (B.1 - A.1),
      k_CH := -1 / k_AB,
      H := C.2 + k_CH * (C.1 - 1) in
  k_AB = 1 / 2 ∧ k_CH = -2 ∧ H = (y - 3 = -2 * (x - 1))

theorem median_eq_gen_form :
  let E := ((A.1 + B.1) / 2, (A.2 + B.2) / 2),
      k_EC := (C.2 - E.2) / (C.1 - E.1),
      G := y - 3 * x in
  E = (0, 0) ∧ k_EC = 3 ∧ G = (3 * x - y = 0)

end altitude_eq_point_slope_form_median_eq_gen_form_l209_209969


namespace hamburger_varieties_l209_209664

theorem hamburger_varieties : 
  let condiments := 10
  let condiment_combinations := 2 ^ condiments
  let meat_patties := 3
  let bun_types := 2
  condiment_combinations * meat_patties * bun_types = 6144 :=
by
  let condiments := 10
  let condiment_combinations := 2 ^ condiments
  let meat_patties := 3
  let bun_types := 2
  have h1 : condiment_combinations = 1024 := by sorry
  calc
    condiment_combinations * meat_patties * bun_types
        = 1024 * 3 * 2 : by rw h1
    ... = 6144 : by norm_num

end hamburger_varieties_l209_209664


namespace max_slope_no_lattice_points_l209_209074

theorem max_slope_no_lattice_points :
  (∃ b : ℚ, (∀ m : ℚ, 1 / 3 < m ∧ m < b → ∀ x : ℤ, 0 < x ∧ x ≤ 200 → ¬ ∃ y : ℤ, y = m * x + 3) ∧ b = 68 / 203) := 
sorry

end max_slope_no_lattice_points_l209_209074


namespace harmonic_mean_1985th_row_l209_209414

-- Define the harmonic table element function
def harmonicElement (n k : ℕ) : ℝ :=
if k = 1 then (1 : ℝ) / n else harmonicElement (n-1) (k-1) - harmonicElement n k

-- Calculate the harmonic mean of the nth row of the harmonic table
def harmonicMean (n : ℕ) : ℝ :=
n / (∑ k in Finset.range n, 1 / harmonicElement n (k+1))

-- Define the lean statement to prove the question == answer given conditions
theorem harmonic_mean_1985th_row :
  harmonicMean 1985 = (1 : ℝ) / (2 ^ 1984) :=
by
  -- Proof omitted
  sorry

end harmonic_mean_1985th_row_l209_209414


namespace ladder_height_l209_209502

theorem ladder_height (c a b : ℝ) (hc : c = 25) (ha : a = 15) (h : c^2 = a^2 + b^2) : b = 20 :=
by
  -- Given conditions
  have hc_squared : c^2 = 25^2, by rw [hc, pow_two]
  have ha_squared : a^2 = 15^2, by rw [ha, pow_two]
  -- Simplify the left hand side and right hand side of the Pythagorean theorem
  have lhs : c^2 = 625, by rw [hc_squared]; exact rfl
  have rhs : 15^2 = 225, by exact rfl
  have simplified_eq : c^2 = 225 + b^2, by rw [h, ha_squared]
  rw [lhs] at simplified_eq
  rw [rhs] at simplified_eq
  have b_squared : b^2 = 400, by linarith
  have b_eq : b = √400, from eq.symm (eq_of_mul_eq_mul_right _ (by exact sq_sqrt b_squared (by norm_num)))
  have b_final : b = 20, by rw [b_eq]; exact rfl
  exact b_final
  -- Sorry is used above to skip proof completion steps where necessary
  sorry

end ladder_height_l209_209502


namespace find_a_range_l209_209977

-- Definitions of the functions
def f (x : ℝ) : ℝ := if x < 0 then cos x + 2^x - 1/2 else 0
def g (x a : ℝ) : ℝ := cos x + log x (x + a)

-- Theorem statement
theorem find_a_range (a : ℝ) : 
  (∃ x < 0, f x = g (-x) a) ↔ a < sqrt 2 := sorry

end find_a_range_l209_209977


namespace odd_factors_of_360_l209_209227

theorem odd_factors_of_360 : ∃ n, 360 = 2^3 * 3^2 * 5^1 ∧ n = 6 :=
by 
  -- Define the prime factorization of 360
  let pf360 := (2^3 * 3^2 * 5^1)
  
  -- Define the count of odd factors by removing the contribution of factor 2.
  let odd_part := (3^2 * 5^1)
  
  -- Calculate the number of factors.
  have odd_factors_count_eq : (2 + 1) * (1 + 1) = 6 := by 
    calc (2 + 1) * (1 + 1) = 3 * 2 : by rfl
                         ... = 6 : by rfl
  
  -- Assert the existence of such n matching the count of odd factors.
  exact ⟨6, And.intro rfl odd_factors_count_eq⟩

end odd_factors_of_360_l209_209227


namespace parallel_lines_eq_a_l209_209658

theorem parallel_lines_eq_a (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a + 1) * x - a * y = 0) → (a = -3/2 ∨ a = 0) :=
by sorry

end parallel_lines_eq_a_l209_209658


namespace equilateral_triangle_area_decomposition_l209_209737

theorem equilateral_triangle_area_decomposition
  (a u v w : ℝ)
  (h_distances : u^2 + v^2 = w^2)
  (h : ∃ (ABC : Triangle) (P : Point),
    ABC.is_equilateral ∧ 
    ABC.side_length = a ∧ 
    (dist P ABC.A = u) ∧ 
    (dist P ABC.B = v) ∧ 
    (dist P ABC.C = w)) :
  w^2 + (Real.sqrt 3) * u * v = a^2 :=
begin
  sorry
end

end equilateral_triangle_area_decomposition_l209_209737


namespace parabola_equation_l209_209804

theorem parabola_equation (a b c : ℝ) (hA : (2:ℝ) = A) (hB : (-1:ℝ) = B)
(hCy_pos : C = (0, 2) → y = -x^2 + x + 2) 
(hCy_neg : C = (0, -2) → y = x^2 - x - 2):
(∀ (x y : ℝ), y = a*x^2 + b*x + c) ∧ (OC = 2) →  
(∃ (eq₁ eq₂ : (ℝ → ℝ)), eq₁ = (λ x, -x^2 + x + 2) ∧ eq₂ = (λ x, x^2 - x - 2)) :=
by sorry

end parabola_equation_l209_209804


namespace batsman_average_increase_l209_209853

def increase_in_average (runs_in_17th : ℕ) (average_after_17 : ℕ) (average_increase : ℕ) : Prop :=
  ∀ (A R : ℕ), 
    runs_in_17th = 88 → 
    average_after_17 = 40 →
    (R = 16 * A) →
    (16 * A + runs_in_17th = average_after_17 * 17) →
    average_increase = average_after_17 - A

theorem batsman_average_increase : increase_in_average 88 40 3 :=
by
  intros A R h_run h_avg_after h_R_eq h_total_eq
  have h1 : R = 16 * A := h_R_eq
  have h2 : 16 * A + 88 = 40 * 17 := h_total_eq
  have h3 : 40 * 17 = 680 := by norm_num
  have h4 : 16 * A + 88 = 680 := by rw [h2, h3]
  have h5 : 16 * A = 592 := by linarith
  have h6 : A = 37 := by norm_num1 : 592 / 16
  have h7 : 40 - 37 = 3 := by norm_num
  exact h7

end batsman_average_increase_l209_209853


namespace plane_angle_divides_cube_l209_209036

noncomputable def angle_between_planes (m n : ℕ) (h : m ≤ n) : ℝ :=
  Real.arctan (2 * m / (m + n))

theorem plane_angle_divides_cube (m n : ℕ) (h : m ≤ n) :
  ∃ α, α = angle_between_planes m n h :=
sorry

end plane_angle_divides_cube_l209_209036


namespace sum_of_squares_of_coeffs_l209_209891

   theorem sum_of_squares_of_coeffs :
     let expr := 3 * (X^3 - 4 * X^2 + X) - 5 * (X^3 + 2 * X^2 - 5 * X + 3)
     let simplified_expr := -2 * X^3 - 22 * X^2 + 28 * X - 15
     let coefficients := [-2, -22, 28, -15]
     (coefficients.map (λ a => a^2)).sum = 1497 := 
   by 
     -- expending, simplifying and summing up the coefficients 
     sorry
   
end sum_of_squares_of_coeffs_l209_209891


namespace correct_statement_l209_209996

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l209_209996


namespace odd_factors_of_360_l209_209176

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l209_209176


namespace sale_in_fourth_month_l209_209862

-- Definitions of the conditions
def sale_in_first_month : ℕ := 6435
def sale_in_second_month : ℕ := 6927
def sale_in_third_month : ℕ := 6855
def sale_in_fifth_month : ℕ := 6562
def sale_in_sixth_month : ℕ := 7991
def average_sale : ℕ := 7000

-- Prove the sale in the fourth month is Rs. 7230
theorem sale_in_fourth_month : 
  sale_in_first_month + sale_in_second_month + sale_in_third_month + 
  (some amount) + sale_in_fifth_month + sale_in_sixth_month = 6 * average_sale →
  (some amount) = 7230 :=
by
  sorry

end sale_in_fourth_month_l209_209862


namespace problem_l209_209627

-- Given the definition of the imaginary unit i with the property i^2 = -1
noncomputable def i : ℂ := complex.I

-- Define the expression for the given problem
def expr := (i ^ 2019) / (1 + i)

-- The theorem to be proved
theorem problem (hi : i * i = -1): expr = -1/2 - 1/2 * i :=
  sorry

end problem_l209_209627


namespace analytic_expression_f_l209_209624

def f (x : ℝ) : ℝ :=
  if x > 0 then -x^2 + x + 1 else if x = 0 then 0 else x^2 + x - 1

theorem analytic_expression_f (x : ℝ) (h_odd : ∀ x, f(-x) = -f(x)) (h_piecewise : ∀ x, x > 0 → f(x) = (-x^2 + x + 1)) : 
  f(x) = if x > 0 then -x^2 + x + 1 else if x = 0 then 0 else x^2 + x - 1 :=
by
  sorry

end analytic_expression_f_l209_209624


namespace radius_ratio_of_spheres_l209_209500

theorem radius_ratio_of_spheres (R r : ℝ) (V_big V_mini : ℝ) 
  (h1 : V_big = (4 / 3) * Real.pi * R^3)
  (h2 : V_big = 972 * Real.pi) 
  (h3 : V_mini = 0.64 * V_big)
  (h4 : V_mini = (4 / 3) * Real.pi * r^3) :
  r / R = 108 / 125 := 
begin
  sorry
end

end radius_ratio_of_spheres_l209_209500


namespace candies_equal_after_finite_rounds_l209_209034

theorem candies_equal_after_finite_rounds
  (n : ℕ) (candies : ℕ → ℕ) :
  (∀ i : ℕ, i < n → even (candies i))
  ∧ (∀ i : ℕ, i < n → 0 < candies i)
  ∧ (∀ k : ℕ,
        ∃ (next_candies : ℕ → ℕ),
          (∀ i : ℕ, i < n → next_candies i = (candies i / 2) + (candies (((i + n) - 1) % n) / 2) + if odd (candies i) then 1 else 0)
          →
            ∃ m : ℕ, ∀ i : ℕ, i < n → candies i = m) :=
begin
  sorry
end

end candies_equal_after_finite_rounds_l209_209034


namespace sum_modulo_9_l209_209905

theorem sum_modulo_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by
  -- Skipping the detailed proof steps
  sorry

end sum_modulo_9_l209_209905


namespace KarenParagraphCount_l209_209724

theorem KarenParagraphCount :
  ∀ (num_essays num_short_ans num_paragraphs total_time essay_time short_ans_time paragraph_time : ℕ),
    (num_essays = 2) →
    (num_short_ans = 15) →
    (total_time = 240) →
    (essay_time = 60) →
    (short_ans_time = 3) →
    (paragraph_time = 15) →
    (total_time = num_essays * essay_time + num_short_ans * short_ans_time + num_paragraphs * paragraph_time) →
    num_paragraphs = 5 :=
by
  sorry

end KarenParagraphCount_l209_209724


namespace eggs_in_second_tree_l209_209322

theorem eggs_in_second_tree
  (nests_in_first_tree : ℕ)
  (eggs_per_nest : ℕ)
  (eggs_in_front_yard : ℕ)
  (total_eggs : ℕ)
  (eggs_in_second_tree : ℕ)
  (h1 : nests_in_first_tree = 2)
  (h2 : eggs_per_nest = 5)
  (h3 : eggs_in_front_yard = 4)
  (h4 : total_eggs = 17)
  (h5 : nests_in_first_tree * eggs_per_nest + eggs_in_front_yard + eggs_in_second_tree = total_eggs) :
  eggs_in_second_tree = 3 :=
sorry

end eggs_in_second_tree_l209_209322


namespace find_C_l209_209429

theorem find_C (A B C : ℕ) (h1 : (19 + A + B) % 3 = 0) (h2 : (15 + A + B + C) % 3 = 0) : C = 1 := by
  sorry

end find_C_l209_209429


namespace find_s2_side_length_l209_209371

-- Define the variables involved
variables (r s : ℕ)

-- Conditions based on problem statement
def height_eq : Prop := 2 * r + s = 2160
def width_eq : Prop := 2 * r + 3 * s + 110 = 4020

-- The theorem stating that s = 875 given the conditions
theorem find_s2_side_length (h1 : height_eq r s) (h2 : width_eq r s) : s = 875 :=
by {
  sorry
}

end find_s2_side_length_l209_209371


namespace find_a12_l209_209133

variable {a : ℕ → ℝ}
variable (d : ℝ)

-- Definition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- The Lean statement for the problem
theorem find_a12 (h_seq : arithmetic_sequence a d)
  (h_cond1 : a 7 + a 9 = 16) (h_cond2 : a 4 = 1) : 
  a 12 = 15 :=
sorry

end find_a12_l209_209133


namespace max_sheep_pen_area_l209_209039

theorem max_sheep_pen_area :
  ∃ x y : ℝ, 15 * 2 = 30 ∧ (x + 2 * y = 30) ∧
  (x > 0 ∧ y > 0) ∧
  (x * y = 112) := by
  sorry

end max_sheep_pen_area_l209_209039


namespace prob_after_five_minutes_l209_209841

-- Define the recurrence relation p(n+1) = 1 - p(n)/2
def p : ℕ → ℚ
| 0     := 0
| (n+1) := 1 - p n / 2

-- Define the theorem we want to prove: p(5) = 11/16
theorem prob_after_five_minutes : p 5 = 11/16 :=
by {
  -- Placeholder for proof steps
  sorry
}

end prob_after_five_minutes_l209_209841


namespace julia_cakes_remaining_l209_209697

/-- Formalizing the conditions of the problem --/
def cakes_per_day : ℕ := 5 - 1
def baking_days : ℕ := 6
def eaten_cakes_per_other_day : ℕ := 1
def total_days : ℕ := 6
def total_eaten_days : ℕ := total_days / 2

/-- The theorem to be proven --/
theorem julia_cakes_remaining : 
  let total_baked_cakes := cakes_per_day * baking_days in
  let total_eaten_cakes := eaten_cakes_per_other_day * total_eaten_days in
  total_baked_cakes - total_eaten_cakes = 21 := 
by
  sorry

end julia_cakes_remaining_l209_209697


namespace real_part_of_z_l209_209120

-- Define the condition: z + 2conj(z) = 6 + i
def z_condition (z : ℂ) : Prop := z + 2 * conj(z) = 6 + complex.I

-- Define the proof problem: If z satisfies the condition, then the real part of z is 2
theorem real_part_of_z (z : ℂ) (h : z_condition z) : z.re = 2 := 
by
  sorry

end real_part_of_z_l209_209120


namespace vasya_more_ways_l209_209366

-- Define the board types
def board_petya : Type := fin 100 × fin 50
def board_vasya : Type := fin 100 × fin 100

-- Define the condition for no kings attacking each other on a board
def non_attacking (kings : set (fin n × fin m)) :=
  ∀ k1 k2 ∈ kings, k1 ≠ k2 → ¬(k1.fst = k2.fst ∨ k1.snd = k2.snd 
    ∨ (k1.fst - k2.fst).nat_abs = (k1.snd - k2.snd).nat_abs)

-- Define the condition that kings are placed on the white cells of a checkerboard
def is_white_square (square : fin 100 × fin 100) :=
  (square.fst.val + square.snd.val) % 2 = 0

-- The sets of possible kings placements
def valid_kings_petya := {k : set board_petya | k.card = 500 ∧ non_attacking k}
def valid_kings_vasya := {k : set board_vasya | k.card = 500 ∧ non_attacking k 
  ∧ ∀ square ∈ k, is_white_square square}

-- Statement of the problem as a Lean theorem
theorem vasya_more_ways : (valid_kings_petya.card < valid_kings_vasya.card) :=
sorry

end vasya_more_ways_l209_209366


namespace find_a7_l209_209936

theorem find_a7 (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : a 2 = 2) 
  (h3 : ∀ n, 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2) : 
  a 7 = real.sqrt 19 :=
sorry

end find_a7_l209_209936


namespace log_product_l209_209464

theorem log_product :
  (Real.log 100 / Real.log 10) * (Real.log (1 / 10) / Real.log 10) = -2 := by
  sorry

end log_product_l209_209464


namespace p_implies_q_but_q_not_implies_p_l209_209655

theorem p_implies_q_but_q_not_implies_p (x : ℝ) :
  (x = sqrt (3 * x + 4) → x^2 = 3 * x + 4) ∧ ¬ (x^2 = 3 * x + 4 → x = sqrt (3 * x + 4)) :=
begin
  sorry
end

end p_implies_q_but_q_not_implies_p_l209_209655


namespace cask_capacity_l209_209319

variable (C : ℝ)
variable (barrel_capacity : ℝ)
variable (total_capacity : ℝ)

def barrels_capacity (C : ℝ) : ℝ := 4 * (2 * C + 3)

theorem cask_capacity :
  (barrel_capacity = 2 * C + 3) →
  (total_capacity = barrels_capacity C + C) →
  (total_capacity = 172) →
  C ≈ 18 :=
by
  sorry

end cask_capacity_l209_209319


namespace ratio_of_areas_l209_209410

theorem ratio_of_areas (s : ℝ) :
  let area_small := (3 * (sqrt 3 / 4 * s^2))
  let area_large := (sqrt 3 / 4 * (3 * s)^2)
  area_small / area_large = 1 / 3 :=
by
  sorry

end ratio_of_areas_l209_209410


namespace total_problems_is_correct_l209_209316

/-- Definition of the number of pages of math homework. -/
def math_pages : ℕ := 2

/-- Definition of the number of pages of reading homework. -/
def reading_pages : ℕ := 4

/-- Definition that each page of homework contains 5 problems. -/
def problems_per_page : ℕ := 5

/-- The proof statement: given the number of pages of math and reading homework,
    and the number of problems per page, prove that the total number of problems is 30. -/
theorem total_problems_is_correct : (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end total_problems_is_correct_l209_209316


namespace percentage_saved_l209_209300

noncomputable def calculateSavedPercentage : ℚ :=
  let first_tier_free_tickets := 1
  let second_tier_free_tickets_per_ticket := 2
  let number_of_tickets_purchased := 10
  let total_free_tickets :=
    first_tier_free_tickets +
    (number_of_tickets_purchased - 5) * second_tier_free_tickets_per_ticket
  let total_tickets_received := number_of_tickets_purchased + total_free_tickets
  let free_tickets := total_tickets_received - number_of_tickets_purchased
  (free_tickets / total_tickets_received) * 100

theorem percentage_saved : calculateSavedPercentage = 52.38 :=
by
  sorry

end percentage_saved_l209_209300


namespace intersection_of_diagonals_quadrilateral_l209_209420

theorem intersection_of_diagonals_quadrilateral (k m b : ℝ) (h : k ≠ m) :
  let L1 (x : ℝ) := k * x + b,
      L2 (x : ℝ) := k * x - b,
      L3 (x : ℝ) := m * x + b,
      L4 (x : ℝ) := m * x - b,
      P1 := (0, b : ℝ),
      P2 := (0, -b : ℝ),
      P3 := (2 * b / (k - m), b : ℝ),
      P4 := (-2 * b / (k - m), -b : ℝ)
  in (0, 0) = ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2) := 
sorry

end intersection_of_diagonals_quadrilateral_l209_209420


namespace expression_as_polynomial_l209_209473

theorem expression_as_polynomial (x : ℝ) :
  (3 * x^3 + 2 * x^2 + 5 * x + 9) * (x - 2) -
  (x - 2) * (2 * x^3 + 5 * x^2 - 74) +
  (4 * x - 17) * (x - 2) * (x + 4) = 
  x^4 + 2 * x^3 - 5 * x^2 + 9 * x - 30 :=
sorry

end expression_as_polynomial_l209_209473


namespace find_missing_number_l209_209415

theorem find_missing_number
  (mean : ℝ)
  (n : ℕ)
  (nums : List ℝ)
  (total_sum : ℝ)
  (sum_known_numbers : ℝ)
  (missing_number : ℝ) :
  mean = 20 → 
  n = 8 →
  nums = [1, 22, 23, 24, 25, missing_number, 27, 2] →
  total_sum = mean * n →
  sum_known_numbers = 1 + 22 + 23 + 24 + 25 + 27 + 2 →
  missing_number = total_sum - sum_known_numbers :=
by
  intros
  sorry

end find_missing_number_l209_209415


namespace size_of_C_l209_209731

noncomputable def smallest_size_C (n : ℕ) (h1 : n > 0) : ℕ :=
  n * 2^n + 1

theorem size_of_C (n : ℕ) (h1 : n > 0)
  (C : set (set (fin (2^n))))
  (h2 : ∀ s : finset (fin (2^n)), s.card = 2^n - 1 → s.val ∈ C)
  (h3 : ∀ s : set (fin (2^n)), s ∈ C ∧ s ≠ ∅ → ∃ c : fin (2^n), s \ {c} ∈ C) :
  C.card ≥ smallest_size_C n h1 :=
sorry

end size_of_C_l209_209731


namespace find_angle_GFH_l209_209399

variables (Q G H F : Type) [HasArea Q G H F]

-- Defining the area of the triangles
axiom area_QGH : area Q G H = 4 * real.sqrt 2
axiom area_FGH_gt : area F G H > 16

-- The angle GFH that needs to be proven
theorem find_angle_GFH (Q G H F : Point) (h1 : Area (triangle Q G H) = 4 * sqrt 2)
  (h2 : Area (triangle F G H) > 16) : Angle F G H = 67.5 :=
sorry

end find_angle_GFH_l209_209399


namespace tangent_lines_perpendicular_and_area_l209_209633

noncomputable def curve : ℝ → ℝ := λ x, x^2 + x - 2

theorem tangent_lines_perpendicular_and_area :
  let l1 := (λ x, 3 * x - 3) in
  let l2 := (λ x, - (1 / 3) * x - (2 / 3)) in
  let p := (1 : ℝ, 0 : ℝ) in
  let slope (f : ℝ → ℝ) (x : ℝ) := (f (x + 1e-8) - f x) / 1e-8 in
  let point_on_curve (x : ℝ) := (x, curve x) in
  (slope curve 1 = 2 * 1 + 1) ∧ -- slope of the curve at x = 1
  (l1 p.1 = p.2) ∧              -- l1 passes through (1, 0)
  (slope curve (-2 / 3) = - (1 / 3)) ∧ -- the slope of the curve at the point -2/3 is -1/3
  -- proof for l1 ⊥ l2 i.e., product of slopes = -1
  ((slope curve 1) * (slope curve (-2 / 3)) = -1) ∧
  -- area of the triangle formed
  let x_intercept_l1 := (1 : ℝ) in
  let x_intercept_l2 := (-2 / 3 : ℝ) in
  let height := 1 in
  let base := x_intercept_l1 - x_intercept_l2 in
  (1 / 2) * base * height =
    (1 / 2) * ((1 - (- (2 / 3))) * 1)
:= by
  sorry

end tangent_lines_perpendicular_and_area_l209_209633


namespace incorrect_tripling_radius_l209_209357

-- Let r be the radius of a circle, and A be its area.
-- The claim is that tripling the radius quadruples the area.
-- We need to prove this claim is incorrect.

theorem incorrect_tripling_radius (r : ℝ) (A : ℝ) (π : ℝ) (hA : A = π * r^2) : 
    (π * (3 * r)^2) ≠ 4 * A :=
by
  sorry

end incorrect_tripling_radius_l209_209357


namespace odd_factors_of_360_l209_209186

theorem odd_factors_of_360 : ∃ n, n = 6 ∧ 
  ∀ (x : ℕ), (∃ (a b : ℕ), 360 = 2^a * 3^b * x ∧ ¬ even x ∧ 0 < x) → (∀ (m : ℕ), m ∣ x ↔ (m = 1 ∨ m = 3 ∨ m = 5 ∨ m = 9 ∨ m = 15 ∨ m = 45)) :=
by
  -- Proof
  sorry

end odd_factors_of_360_l209_209186


namespace harmonic_sum_inequality_l209_209167

def H (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), 1 / (i + 1 : ℝ)

theorem harmonic_sum_inequality :
  (∑ i in Finset.range 2012, 1 / (↑(i + 1) * (H (i + 1))^2)) < 2 :=
by
  sorry

end harmonic_sum_inequality_l209_209167


namespace no_real_roots_of_quadratic_l209_209669

theorem no_real_roots_of_quadratic (k : ℝ) (h : 12 - 3 * k < 0) : ∀ (x : ℝ), ¬ (x^2 + 4 * x + k = 0) := by
  sorry

end no_real_roots_of_quadratic_l209_209669


namespace find_x_l209_209254

theorem find_x (x : ℤ) (h : 3^(x - 4) = 9^3) : x = 10 := 
sorry

end find_x_l209_209254


namespace find_x_l209_209253

theorem find_x (x : ℤ) (h : 3^(x - 4) = 9^3) : x = 10 := 
sorry

end find_x_l209_209253


namespace number_of_valid_triangles_l209_209235

-- Definition of the set of points in the 5x5 grid with integer coordinates
def gridPoints := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

-- Function to determine if three points are collinear
def collinear (a b c : ℕ × ℕ) : Prop :=
  (b.2 - a.2) * (c.1 - b.1) = (c.2 - b.2) * (b.1 - a.1)

-- The main theorem stating the number of triangles with positive area
theorem number_of_valid_triangles : 
  ∃ n, n = 2158 ∧ ∀ (a b c : ℕ × ℕ), a ∈ gridPoints → b ∈ gridPoints → c ∈ gridPoints → a ≠ b → b ≠ c → c ≠ a → ¬collinear a b c → n = 2158 :=
by
  sorry

end number_of_valid_triangles_l209_209235


namespace analytical_expression_l209_209151

def f (x : ℝ) : ℝ := (1/3) * x^2 - 4 * x + 6

theorem analytical_expression (f : ℝ → ℝ) 
  (h : ∀ x, f x + 2 * f (3 - x) = x^2) : 
  ∀ x, f x = (1/3) * x^2 - 4 * x + 6 := 
by
  sorry

end analytical_expression_l209_209151


namespace seven_pow_l209_209477

theorem seven_pow (k : ℕ) (h : 7 ^ k = 2) : 7 ^ (4 * k + 2) = 784 :=
by 
  sorry

end seven_pow_l209_209477


namespace inclination_angle_of_y_axis_l209_209801

theorem inclination_angle_of_y_axis : 
  ∀ (l : ℝ), l = 90 :=
sorry

end inclination_angle_of_y_axis_l209_209801


namespace angle_MBC_45_degrees_l209_209346

noncomputable def center_of_square (A C D E : Point) : Point :=
(M) -- Detailed construction elided here, assumed given by the context of the problem
 
theorem angle_MBC_45_degrees 
  (A B C D E M : Point)
  (h1 : is_right_triangle A B C)
  (h2 : is_square A C D E)
  (h3 : center_of_square A C D E = M)
  : angle M B C = 45 :=
sorry

end angle_MBC_45_degrees_l209_209346


namespace problem_solution_inequality_l209_209582

theorem problem_solution_inequality (x : ℝ) (h : x > 9) :
    (sqrt (x - 9 * sqrt(x - 9)) + 3 = sqrt(x + 9 * sqrt(x - 9)) - 3) →
    (x ∈ set.Ici 40.5) := 
by
  sorry

end problem_solution_inequality_l209_209582


namespace common_difference_arith_seq_log_l209_209337

variable {p q r : ℝ}

def geom_seq (a b c : ℝ) : Prop := b^2 = a * c
def arith_seq (a b c : ℝ) (d : ℝ) : Prop := b - a = d ∧ c - b = d

theorem common_difference_arith_seq_log:
  ∀ (p q r : ℝ), 
  p ≠ q → p ≠ r → q ≠ r → 0 < p → 0 < q → 0 < r → 
  geom_seq p q r → 
  (∃ d, arith_seq (Real.log r p) (Real.log q r) (Real.log p q) d) → d = 3 / 2 :=
by 
  sorry

end common_difference_arith_seq_log_l209_209337


namespace number_of_odd_factors_of_360_l209_209208

theorem number_of_odd_factors_of_360 : 
  let factors (n : ℕ) := {d : ℕ | d ∣ n}
  let odd (d : ℕ) := d % 2 = 1
  let is_factor (n : ℕ) (d : ℕ) := d ∣ n
  let number_of_odd_factors (n : ℕ) := (factors n).count odd
  in number_of_odd_factors 360 = 6 :=
begin
  sorry
end

end number_of_odd_factors_of_360_l209_209208


namespace geometric_sequence_sum_l209_209276

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (a_pos : ∀ n, 0 < a n)
  (h_a2 : a 2 = 1) (h_a3a7_a5 : a 3 * a 7 - a 5 = 56)
  (S_eq : ∀ n, S n = (a 1 * (1 - (2 : ℝ) ^ n)) / (1 - 2)) :
  S 5 = 31 / 2 := by
  sorry

end geometric_sequence_sum_l209_209276


namespace binomial_20_13_l209_209958

theorem binomial_20_13 (h₁ : Nat.choose 21 13 = 203490) (h₂ : Nat.choose 21 14 = 116280) :
  Nat.choose 20 13 = 58140 :=
by
  sorry

end binomial_20_13_l209_209958


namespace polynomial_at_x_is_minus_80_l209_209448

def polynomial (x : ℤ) : ℤ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def x_value : ℤ := 2

theorem polynomial_at_x_is_minus_80 : polynomial x_value = -80 := 
by
  sorry

end polynomial_at_x_is_minus_80_l209_209448


namespace inequality_solution_l209_209781

-- Define the condition for the denominator being positive
def denom_positive (x : ℝ) : Prop :=
  x^2 + 2*x + 7 > 0

-- Statement of the problem
theorem inequality_solution (x : ℝ) (h : denom_positive x) :
  (x + 6) / (x^2 + 2*x + 7) ≥ 0 ↔ x ≥ -6 :=
sorry

end inequality_solution_l209_209781


namespace students_not_skating_nor_skiing_l209_209497

theorem students_not_skating_nor_skiing (total_students skating_students skiing_students both_students : ℕ)
  (h_total : total_students = 30)
  (h_skating : skating_students = 20)
  (h_skiing : skiing_students = 9)
  (h_both : both_students = 5) :
  total_students - (skating_students + skiing_students - both_students) = 6 :=
by
  sorry

end students_not_skating_nor_skiing_l209_209497


namespace sum_first_five_terms_geometric_seq_l209_209296

open_locale big_operators

noncomputable def geometric_seq (a q : ℕ → ℝ) (n : ℕ) : ℝ :=
a * q ^ n

theorem sum_first_five_terms_geometric_seq :
  ∀ (a q : ℕ → ℝ),
  a 1 = 2 →
  q (1:ℕ) > 0 →
  2 * q 2 3 + 4 = 2 * q 1 + 2 * q 4 →
  (∑ i in finset.range 5, geometric_seq a q i) = 62 :=
begin
  intros a q h1 h2 h3,
  sorry  -- proof placeholder
end

end sum_first_five_terms_geometric_seq_l209_209296


namespace intersections_form_rhombus_center_of_rhombus_on_EF_l209_209727

-- Definitions based on given conditions
variables {A B C D E F G H K L M N: Point}

-- Conditions of the problem
def isCyclicQuadrilateral (A B C D: Point): Prop := sorry
def isMidpoint (E F: Point) (A C B D: Point): Prop := sorry
def intersections (G H: Point) (A B C D: Point): Prop := sorry

-- Assertions to be proven
def formRhombus (K L M N: Point): Prop := sorry
def centerOnEF (E F: Point) (K L M N: Point): Prop := sorry

-- Final theorem statements
theorem intersections_form_rhombus 
    (h1: isCyclicQuadrilateral A B C D)
    (h2: isMidpoint E F (A, C) (B, D))
    (h3: intersections G H (A B C D)): 
    formRhombus K L M N := sorry

theorem center_of_rhombus_on_EF 
    (h1: isCyclicQuadrilateral A B C D)
    (h2: isMidpoint E F (A, C) (B, D))
    (h3: intersections G H (A B C D))
    (h4: formRhombus K L M N): 
    centerOnEF E F K L M N := sorry

end intersections_form_rhombus_center_of_rhombus_on_EF_l209_209727


namespace largest_three_digit_divisible_and_prime_sum_l209_209589

theorem largest_three_digit_divisible_and_prime_sum :
  ∃ n : ℕ, 900 ≤ n ∧ n < 1000 ∧
           (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ≠ 0 ∧ n % d = 0) ∧
           Prime (n / 100 + (n / 10) % 10 + n % 10) ∧
           n = 963 ∧
           ∀ m : ℕ, 900 ≤ m ∧ m < 1000 ∧
           (∀ d ∈ [m / 100, (m / 10) % 10, m % 10], d ≠ 0 ∧ m % d = 0) ∧
           Prime (m / 100 + (m / 10) % 10 + m % 10) →
           m ≤ 963 :=
by
  sorry

end largest_three_digit_divisible_and_prime_sum_l209_209589


namespace store_paid_price_l209_209424

-- Definition of the conditions
def selling_price : ℕ := 34
def difference_price : ℕ := 8

-- Statement that needs to be proven.
theorem store_paid_price : (selling_price - difference_price) = 26 :=
by
  sorry

end store_paid_price_l209_209424


namespace triangle_cotangent_identity_l209_209288

theorem triangle_cotangent_identity (a b c : ℝ) (A B C : ℝ) 
  (ha : a = 2 * real.sin A) 
  (hb : b = 2 * real.sin B) 
  (hc : c = 2 * real.sin C) 
  (sum_angle : A + B + C = real.pi) :
  (a^2 - b^2) * real.cot C + (b^2 - c^2) * real.cot A + (c^2 - a^2) * real.cot B = 0 := 
sorry -- Proof goes here

end triangle_cotangent_identity_l209_209288


namespace find_a4_l209_209747

variable {α : Type*} [Field α] [Inhabited α]

-- Definitions of the geometric sequence conditions
def geometric_sequence_condition1 (a₁ q : α) : Prop :=
  a₁ * (1 + q) = -1

def geometric_sequence_condition2 (a₁ q : α) : Prop :=
  a₁ * (1 - q^2) = -3

-- Definition of the geometric sequence
def geometric_sequence (a₁ q : α) (n : ℕ) : α :=
  a₁ * q^n

-- The theorem to be proven
theorem find_a4 (a₁ q : α) (h₁ : geometric_sequence_condition1 a₁ q) (h₂ : geometric_sequence_condition2 a₁ q) :
  geometric_sequence a₁ q 3 = -8 :=
  sorry

end find_a4_l209_209747


namespace solve_system_for_x_l209_209673

theorem solve_system_for_x :
  ∃ x y : ℝ, (2 * x + y = 4) ∧ (x + 2 * y = 5) ∧ (x = 1) :=
by
  sorry

end solve_system_for_x_l209_209673


namespace max_possible_value_l209_209764

theorem max_possible_value (a : Fin 9 → ℕ) (h_diff : Function.Injective a) (h_mean : (∑ i, a i) = 144) : 
  ∃ i, a i = 108 :=
by {
  sorry
}

end max_possible_value_l209_209764


namespace number_of_positive_area_triangles_l209_209244

def integer_points := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

def count_triangles_with_positive_area : Nat :=
  let total_points := 25 -- total number of integer points in the grid
  let total_combinations := Nat.choose total_points 3 -- total possible combinations
  let degenerate_cases := 136 -- total degenerate (collinear) cases
  total_combinations - degenerate_cases

theorem number_of_positive_area_triangles : count_triangles_with_positive_area = 2164 := by
  sorry

end number_of_positive_area_triangles_l209_209244


namespace first_year_after_2022_with_digit_sum_5_l209_209709

def sum_of_digits (n : ℕ) : ℕ :=
  (toString n).foldl (λ acc c => acc + c.toNat - '0'.toNat) 0

theorem first_year_after_2022_with_digit_sum_5 :
  ∃ y : ℕ, y > 2022 ∧ sum_of_digits y = 5 ∧ ∀ z : ℕ, z > 2022 ∧ z < y → sum_of_digits z ≠ 5 :=
sorry

end first_year_after_2022_with_digit_sum_5_l209_209709


namespace triangle_angles_l209_209035

-- Definitions of the points L, K, and the fact that they comply with the perpendicular bisector condition
-- Define the isosceles triangle ABC with AB = AC
-- Define the equality of areas for triangles ALC and KBL

theorem triangle_angles (α β γ : ℝ) (A B C L K : ℂ)
  (h_isosceles_tri: angle A B C = β ∧ angle A C B = β ∧ angle B A C = α)
  (h_perpendicular_bisector : ∃ L K, K ≠ L ∧ M = midpoint A C ∧ K in line (perpendicular_bisector A C) ∧ L ∈ segment A B )
  (h_area_eq : 2 * area A L C = area K B L) :
  α = 36 ∧ β = 72 ∧ γ = 72 :=
begin
  sorry,
end

end triangle_angles_l209_209035


namespace binary_ternary_product_l209_209546

-- Definitions for numbers in binary and ternary
def binary_to_decimal : Nat := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0
def ternary_to_decimal : Nat := 1 * 3^2 + 0 * 3^1 + 2 * 3^0

-- Main theorem to prove
theorem binary_ternary_product : binary_to_decimal * ternary_to_decimal = 143 := by
  -- converting binary and ternary to decimal
  have h1 : binary_to_decimal = 13 := by
    simp [binary_to_decimal]
  have h2 : ternary_to_decimal = 11 := by
    simp [ternary_to_decimal]
  
  -- Calculation of the product
  calc
    binary_to_decimal * ternary_to_decimal = 13 * 11     := by rw [h1, h2]
                                        ... = 143 := by norm_num


end binary_ternary_product_l209_209546


namespace B_subset_A_l209_209957

variable {m : ℝ}

def A (m : ℝ) : Set ℝ := {x | x^2 + (m - 2) * x - 2 * m = 0}
def B (m : ℝ) : Set ℝ := {x | m * x = 2 * x + 1}

theorem B_subset_A (m : ℝ) : B m ⊆ A m ↔ (m = 1 ∨ m = 2 ∨ m = 5 / 2) := sorry

end B_subset_A_l209_209957


namespace sum_of_mean_and_median_of_b_eq_2exp289_l209_209983

def a : List ℕ := [17, 27, 31, 53, 61]
def b := [Real.exp(17 * 17), 2 * 27 * Real.sin 27, Real.sqrt 31, Real.log 53, sorry]

theorem sum_of_mean_and_median_of_b_eq_2exp289 :
  let b := [Real.exp(17 * 17), 2 * 27 * Real.sin 27, Real.sqrt 31, Real.log 53]  -- (excluding b[4] since it's not valid)
  (mean b + median b) = 2 * Real.exp(289) :=
sorry

end sum_of_mean_and_median_of_b_eq_2exp289_l209_209983


namespace sum_of_pairs_l209_209846

theorem sum_of_pairs (a : ℕ → ℝ) (h1 : ∀ n, a n ≠ 0)
  (h2 : ∀ n, a n * a (n + 3) = a (n + 2) * a (n + 5))
  (h3 : a 1 * a 2 + a 3 * a 4 + a 5 * a 6 = 6) :
  a 1 * a 2 + a 3 * a 4 + a 5 * a 6 + a 7 * a 8 + a 9 * a 10 + a 11 * a 12 + 
  a 13 * a 14 + a 15 * a 16 + a 17 * a 18 + a 19 * a 20 + a 21 * a 22 + 
  a 23 * a 24 + a 25 * a 26 + a 27 * a 28 + a 29 * a 30 + a 31 * a 32 + 
  a 33 * a 34 + a 35 * a 36 + a 37 * a 38 + a 39 * a 40 + a 41 * a 42 = 42 := 
sorry

end sum_of_pairs_l209_209846


namespace quadratic_no_real_roots_l209_209469

theorem quadratic_no_real_roots : ∀ (a b c : ℝ), a ≠ 0 → Δ = (b*b - 4*a*c) → x^2 + 3 = 0 → Δ < 0 := by
  sorry

end quadratic_no_real_roots_l209_209469


namespace find_z_l209_209670

theorem find_z (x y z : ℝ) (h : 1 / (x + 1) + 1 / (y + 1) = 1 / z) :
  z = (x + 1) * (y + 1) / (x + y + 2) :=
sorry

end find_z_l209_209670


namespace number_of_valid_triangles_l209_209237

-- Definition of the set of points in the 5x5 grid with integer coordinates
def gridPoints := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

-- Function to determine if three points are collinear
def collinear (a b c : ℕ × ℕ) : Prop :=
  (b.2 - a.2) * (c.1 - b.1) = (c.2 - b.2) * (b.1 - a.1)

-- The main theorem stating the number of triangles with positive area
theorem number_of_valid_triangles : 
  ∃ n, n = 2158 ∧ ∀ (a b c : ℕ × ℕ), a ∈ gridPoints → b ∈ gridPoints → c ∈ gridPoints → a ≠ b → b ≠ c → c ≠ a → ¬collinear a b c → n = 2158 :=
by
  sorry

end number_of_valid_triangles_l209_209237


namespace biased_coin_heads_divisible_by_3_l209_209488

open ProbabilityTheory

-- Define the probability of heads for the biased coin
def probability_of_heads : ℝ := 3 / 4

-- Define the number of tosses
def num_of_tosses : ℕ := 60

-- Define the event that number of heads is divisible by 3
def event_heads_div_by_3 (num_heads : ℕ) : Prop := num_heads % 3 = 0

-- Main theorem statement
theorem biased_coin_heads_divisible_by_3 :
  P (λ ω => event_heads_div_by_3 (sum (λ i:nat, if bernoulli probability_of_heads ω i then 1 else 0))) = 1 / 3 :=
sorry

end biased_coin_heads_divisible_by_3_l209_209488


namespace triangle_area_common_focus_l209_209640

theorem triangle_area_common_focus
  (m n : ℝ) (hx : m > 1) (hy : n > 0)
  (h : m^2 - n^2 = 2)
  (ellipse : ∃ P : ℝ × ℝ, (P.1^2 / m^2 + P.2^2 = 1))
  (hyperbola : ∃ P : ℝ × ℝ, (P.1^2 / n^2 - P.2^2 = 1)) :
  ∃ F1 F2 P : ℝ × ℝ, is_right_angle_triangle F1 P F2 ∧ area_triangle F1 P F2 = 1 :=
sorry

end triangle_area_common_focus_l209_209640


namespace intersecting_lines_implies_a_eq_c_l209_209482

theorem intersecting_lines_implies_a_eq_c
  (k b a c : ℝ)
  (h_kb : k ≠ b)
  (exists_point : ∃ (x y : ℝ), (y = k * x + k) ∧ (y = b * x + b) ∧ (y = a * x + c)) :
  a = c := 
sorry

end intersecting_lines_implies_a_eq_c_l209_209482


namespace minimum_value_a_l209_209649

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x

theorem minimum_value_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → f' a x ≥ 0) → a ≥ -3 :=
by {
  -- The proof will be placed here
  sorry
}

end minimum_value_a_l209_209649


namespace odd_factors_of_360_l209_209233

theorem odd_factors_of_360 : ∃ n, 360 = 2^3 * 3^2 * 5^1 ∧ n = 6 :=
by 
  -- Define the prime factorization of 360
  let pf360 := (2^3 * 3^2 * 5^1)
  
  -- Define the count of odd factors by removing the contribution of factor 2.
  let odd_part := (3^2 * 5^1)
  
  -- Calculate the number of factors.
  have odd_factors_count_eq : (2 + 1) * (1 + 1) = 6 := by 
    calc (2 + 1) * (1 + 1) = 3 * 2 : by rfl
                         ... = 6 : by rfl
  
  -- Assert the existence of such n matching the count of odd factors.
  exact ⟨6, And.intro rfl odd_factors_count_eq⟩

end odd_factors_of_360_l209_209233


namespace integral_of_f_l209_209943

noncomputable def f : ℝ → ℝ :=
λ x, if x ∈ set.Icc (-1:ℝ) (1:ℝ) then real.sqrt (1 - x^2) else if x ∈ set.Ioo (1:ℝ) (2:ℝ) then x^2 - 1 else 0

theorem integral_of_f :
  ∫ x in -1..2, f x = (real.pi / 2) + (4 / 3) :=
by 
  sorry

end integral_of_f_l209_209943


namespace shaded_region_ratio_l209_209554

theorem shaded_region_ratio (n : ℕ) (h_n : n = 5) :
  let total_squares := n * n,
      shaded_area := 2.5,
      total_area := total_squares in
  (shaded_area / total_area) = 1 / 10 :=
by
  sorry

end shaded_region_ratio_l209_209554


namespace max_elves_without_caps_proof_max_elves_with_caps_proof_l209_209295

-- Defining the conditions and the problem statement
open Nat

-- We model the problem with the following:
axiom truth_teller : Type
axiom liar_with_caps : Type
axiom dwarf_with_caps : Type
axiom dwarf_without_caps : Type

noncomputable def max_elves_without_caps : ℕ :=
  59

noncomputable def max_elves_with_caps : ℕ :=
  30

-- Part (a): Given the conditions, we show that the maximum number of elves without caps is 59
theorem max_elves_without_caps_proof : max_elves_without_caps = 59 :=
by
  sorry

-- Part (b): Given the conditions, we show that the maximum number of elves with caps is 30
theorem max_elves_with_caps_proof : max_elves_with_caps = 30 :=
by
  sorry

end max_elves_without_caps_proof_max_elves_with_caps_proof_l209_209295


namespace rainfall_second_week_l209_209004

theorem rainfall_second_week (x : ℝ) (h1 : x + 1.5 * x = 20) : 1.5 * x = 12 := 
by {
  sorry
}

end rainfall_second_week_l209_209004


namespace find_real_part_of_a_l209_209278

noncomputable def pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_real_part_of_a (a : ℝ) (h : pure_imaginary ((a + complex.I) / (1 - complex.I))) : a = 1 :=
by
  sorry

end find_real_part_of_a_l209_209278


namespace sin_squared_minus_sin_cos_eq_2_div_5_l209_209623

theorem sin_squared_minus_sin_cos_eq_2_div_5 
    (α : ℝ) 
    (h : (sin α + 3 * cos α) / (3 * cos α - sin α) = 5) : 
    sin α * sin α - sin α * cos α = 2 / 5 := 
by
  sorry

end sin_squared_minus_sin_cos_eq_2_div_5_l209_209623


namespace odd_factors_360_l209_209210

theorem odd_factors_360 : 
  let fac := (2^3) * (3^2) * (5^1) in
  fac = 360 → 
  num_factors (3^2 * 5^1) = 6 :=
sorry

end odd_factors_360_l209_209210


namespace inscribed_quad_equal_circles_is_square_l209_209617

-- Definitions to support the conditions
structure Quadrilateral (A B C D : Point) :=
(inscribed : ∃ O : Point, is_circumcircle O A B C D)

structure TangentCircles (A B C D P : Point) :=
(equal_circles : ∀ (circ : Circle), touches_diagonals_and_circumcircle circ A B C D P → equal_radii circ)

theorem inscribed_quad_equal_circles_is_square
  {A B C D P : Point}
  (h1 : Quadrilateral A B C D)
  (h2 : TangentCircles A B C D P) :
  is_square A B C D :=
sorry

end inscribed_quad_equal_circles_is_square_l209_209617


namespace maximum_x5_l209_209620

noncomputable def max_x5_condition : ℕ :=
  let x1 := 1
  let x2 := 1
  let x3 := 1
  let x4 := 2
  let x5 := 5
  in x5

theorem maximum_x5 (x1 x2 x3 x4 x5 : ℕ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) (h_eq : x1 + x2 + x3 + x4 + x5 = x1 * x2 * x3 * x4 * x5) : x5 ≤ 5 :=
by {
  have positive_integers : 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧ 0 < x5 := ⟨h1, h2, h3, h4, h5⟩,
  sorry
}

end maximum_x5_l209_209620


namespace intersection_of_sets_l209_209657

open Set

def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}

def B : Set ℝ := Ico 0 4  -- Ico stands for interval [0, 4)

theorem intersection_of_sets : A ∩ B = Ico 2 4 :=
by 
  sorry

end intersection_of_sets_l209_209657


namespace DE_minimal_length_in_triangle_l209_209711

noncomputable def min_length_DE (BC AC : ℝ) (angle_B : ℝ) : ℝ :=
  if BC = 5 ∧ AC = 12 ∧ angle_B = 13 then 2 * Real.sqrt 3 else sorry

theorem DE_minimal_length_in_triangle :
  min_length_DE 5 12 13 = 2 * Real.sqrt 3 :=
sorry

end DE_minimal_length_in_triangle_l209_209711


namespace neg_q_necessary_not_sufficient_for_neg_p_l209_209743

-- Proposition p: |x + 2| > 2
def p (x : ℝ) : Prop := abs (x + 2) > 2

-- Proposition q: 1 / (3 - x) > 1
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Negation of p and q
def neg_p (x : ℝ) : Prop := -4 ≤ x ∧ x ≤ 0
def neg_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

-- Theorem: negation of q is a necessary but not sufficient condition for negation of p
theorem neg_q_necessary_not_sufficient_for_neg_p :
  (∀ x : ℝ, neg_p x → neg_q x) ∧ (∃ x : ℝ, neg_q x ∧ ¬neg_p x) :=
by
  sorry

end neg_q_necessary_not_sufficient_for_neg_p_l209_209743


namespace solitaire_game_end_with_one_piece_l209_209759

theorem solitaire_game_end_with_one_piece (n : ℕ) : 
  ∃ (remaining_pieces : ℕ), 
  remaining_pieces = 1 ↔ n % 3 ≠ 0 :=
sorry

end solitaire_game_end_with_one_piece_l209_209759


namespace sampling_method_correct_l209_209566

-- Define the conditions
def selection1 :=
  { n : ℕ // n = 2 ∧ 10 }

def selection2 :=
  { n : ℕ // n = 50 ∧ 1000 }

-- Define the result as true/false proposition
def isOptC (selection1 : {n : ℕ // n = 2 ∧ 10}) 
           (selection2 : {n : ℕ // n = 50 ∧ 1000}) : Prop :=
  (selection1 = 2 ∧ 10) ∧ (selection2 = 50 ∧ 1000)

-- The theorem we need to prove:
theorem sampling_method_correct : ∀ (s1 : {n : ℕ // n = 2 ∧ 10}) 
                                    (s2 : {n : ℕ // n = 50 ∧ 1000}), 
                                    isOptC s1 s2 := 
by
  sorry

end sampling_method_correct_l209_209566


namespace circle_center_l209_209020

theorem circle_center (a b : ℝ)
  (passes_through_point : (a - 0)^2 + (b - 9)^2 = r^2)
  (is_tangent : (a - 3)^2 + (b - 9)^2 = r^2 ∧ b = 6 * (a - 3) + 9 ∧ (b - 9) / (a - 3) = -1/6) :
  a = 3/2 ∧ b = 37/4 := 
by 
  sorry

end circle_center_l209_209020


namespace rotate_parabola_180_l209_209376

theorem rotate_parabola_180 (x: ℝ) : 
  let original_parabola := λ x, 2 * (x - 3)^2 - 2,
      rotated_parabola := λ x, -2 * (x - 3)^2 - 2 in
  original_parabola x = rotated_parabola x :=
sorry

end rotate_parabola_180_l209_209376


namespace day_after_7k_days_is_friday_day_before_7k_days_is_thursday_day_after_100_days_is_sunday_l209_209823

-- Definitions based on problem conditions and questions
def day_of_week_after (n : ℤ) (current_day : String) : String :=
  if n % 7 = 0 then current_day else
    if n % 7 = 1 then "Saturday" else
    if n % 7 = 2 then "Sunday" else
    if n % 7 = 3 then "Monday" else
    if n % 7 = 4 then "Tuesday" else
    if n % 7 = 5 then "Wednesday" else
    "Thursday"

def day_of_week_before (n : ℤ) (current_day : String) : String :=
  day_of_week_after (-n) current_day

-- Conditions
def today : String := "Friday"

-- Prove the following
theorem day_after_7k_days_is_friday (k : ℤ) : day_of_week_after (7 * k) today = "Friday" :=
by sorry

theorem day_before_7k_days_is_thursday (k : ℤ) : day_of_week_before (7 * k) today = "Thursday" :=
by sorry

theorem day_after_100_days_is_sunday : day_of_week_after 100 today = "Sunday" :=
by sorry

end day_after_7k_days_is_friday_day_before_7k_days_is_thursday_day_after_100_days_is_sunday_l209_209823


namespace no_purchase_count_l209_209486

def total_people : ℕ := 15
def people_bought_tvs : ℕ := 9
def people_bought_computers : ℕ := 7
def people_bought_both : ℕ := 3

theorem no_purchase_count : total_people - (people_bought_tvs - people_bought_both) - (people_bought_computers - people_bought_both) - people_bought_both = 2 := by
  sorry

end no_purchase_count_l209_209486


namespace sales_profit_l209_209868

theorem sales_profit 
  (n C n_swap p_swap n_dept p_dept Profit : ℕ)
  (sale_price : ℕ -> ℕ)
  (h1 : n = 48)
  (h2 : C = 576)
  (h3 : n_swap = 17)
  (h4 : p_swap = 18)
  (h5 : n_dept = 10)
  (h6 : p_dept = 25)
  (h7 : Profit = 442)
  (h_sale : sale_price = λ remaining, 22):
  let remaining_bag_count := n - (n_swap + n_dept) in
  let total_cost := C in
  let total_revenue := (n_swap * p_swap + n_dept * p_dept + remaining_bag_count * sale_price remaining_bag_count) in
  (total_revenue - total_cost = Profit) :=
by
  have remaining_bag_count := n - (n_swap + n_dept)
  have total_cost := C
  have total_revenue := (n_swap * p_swap + n_dept * p_dept + remaining_bag_count * sale_price remaining_bag_count)
  show (total_revenue - total_cost = Profit)
  sorry

end sales_profit_l209_209868


namespace binomial_coeff_sum_l209_209668

theorem binomial_coeff_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ),
    (∀ x : ℝ, (1 - 2*x)^10 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 + a_6*x^6 + a_7*x^7 + a_8*x^8 + a_9*x^9 + a_10*x^10) →
    a_0 + a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 + 6 * a_6 + 7 * a_7 + 8 * a_8 + 9 * a_9 + 10 * a_10 = 21 :=
begin
  intro a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10,
  intro H,
  sorry
end

end binomial_coeff_sum_l209_209668


namespace real_part_of_complex_eq_2_l209_209124

-- Definition of the problem
def complex_satisfying_equation (z : ℂ) : Prop :=
  z + 2 * conj z = 6 + complex.i

-- Proving the real part of z in given conditions
theorem real_part_of_complex_eq_2 (z : ℂ) (h : complex_satisfying_equation z) : z.re = 2 :=
by {
  sorry
}

end real_part_of_complex_eq_2_l209_209124


namespace log_a2016_eq_2_l209_209306

noncomputable def f (x : ℝ) : ℝ := (1/3)*x^3 - 4*x^2 + 6*x - 1

theorem log_a2016_eq_2 :
  ∀ (a : ℕ → ℝ), 
  (a 2) ∈ {x | f' x = 0} ∧ (a 4030) ∈ {x | f' x = 0} → 
  log 2 (a 2016) = 2 :=
by
  sorry

end log_a2016_eq_2_l209_209306


namespace find_x_l209_209577

def fractional_part (x : ℝ) : ℝ := x - x.floor

theorem find_x
  (x : ℝ)
  (hx0 : x ≠ 0)
  (h_arith_seq : x.floor - fractional_part x = 2 * fractional_part x)
  (h_floor_frac : x.floor = 3 * fractional_part x) :
  x = 5 / 3 :=
by
  sorry

end find_x_l209_209577


namespace number_of_integers_P_leq_zero_l209_209561

def P (x : ℝ) : ℝ := ∏ i in finset.range(1, 51), (x - (i^2 : ℝ))

theorem number_of_integers_P_leq_zero 
  : ∃ n : ℕ, P(n) ≤ 0 ∧ n = 1300 := 
by
  sorry

end number_of_integers_P_leq_zero_l209_209561


namespace min_deliveries_to_cover_cost_l209_209350

theorem min_deliveries_to_cover_cost (cost_per_van earnings_per_delivery gasoline_cost_per_delivery : ℕ) (h1 : cost_per_van = 4500) (h2 : earnings_per_delivery = 15 ) (h3 : gasoline_cost_per_delivery = 5) : 
  ∃ d : ℕ, 10 * d ≥ cost_per_van ∧ ∀ x : ℕ, x < d → 10 * x < cost_per_van :=
by
  use 450
  sorry

end min_deliveries_to_cover_cost_l209_209350


namespace reflection_matrix_values_l209_209907

theorem reflection_matrix_values (a b : ℚ) :
  let R : Matrix (Fin 2) (Fin 2) ℚ := ![![a, b], ![-(3/5), 4/5]] in
  R ⬝ R = (1 : Matrix (Fin 2) (Fin 2) ℚ) ↔ (a = -4/5 ∧ b = -3/5) :=
sorry

end reflection_matrix_values_l209_209907


namespace angle_ABC_is_60_deg_l209_209314

-- Definitions of the conditions and problem statement
variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables (length : A → A → ℝ) (angle : A → A → A → ℝ)

axiom eq_AB_BC : length A B = length B C
axiom point_on_AC : ∃ D : A, True
axiom BD_bisects_BAC : ∀ B D C, angle B D C = angle D C B
axiom BD_equal_CD : length B D = length C D

-- The statement which needs to be proven: ∠ABC = 60°
theorem angle_ABC_is_60_deg : 
  ∀ A B C D, (length A B = length B C) → (BD_bisects_BAC B D C) → (length B D = length C D) → angle A B C = 60 := 
by
  sorry

end angle_ABC_is_60_deg_l209_209314


namespace cos_double_angle_at_origin_l209_209967

noncomputable def vertex : ℝ × ℝ := (0, 0)
noncomputable def initial_side : ℝ × ℝ := (1, 0)
noncomputable def terminal_side : ℝ × ℝ := (-1, 3)
noncomputable def cos2alpha (v i t : ℝ × ℝ) : ℝ :=
  2 * ((t.1) / (Real.sqrt (t.1 ^ 2 + t.2 ^ 2))) ^ 2 - 1

theorem cos_double_angle_at_origin :
  cos2alpha vertex initial_side terminal_side = -4 / 5 :=
by
  sorry

end cos_double_angle_at_origin_l209_209967


namespace inequality_proof_l209_209954

theorem inequality_proof (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z ≥ 1) : 
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by
  sorry

end inequality_proof_l209_209954


namespace largest_possible_b_l209_209816

theorem largest_possible_b 
  (V : ℕ)
  (a b c : ℤ)
  (hV : V = 360)
  (h1 : 1 < c)
  (h2 : c < b)
  (h3 : b < a)
  (h4 : a * b * c = V) 
  : b = 12 := 
  sorry

end largest_possible_b_l209_209816


namespace mangoes_rate_l209_209053

theorem mangoes_rate (grapes_weight mangoes_weight total_amount grapes_rate mango_rate : ℕ)
  (h1 : grapes_weight = 7)
  (h2 : grapes_rate = 68)
  (h3 : total_amount = 908)
  (h4 : mangoes_weight = 9)
  (h5 : total_amount - grapes_weight * grapes_rate = mangoes_weight * mango_rate) :
  mango_rate = 48 :=
by
  sorry

end mangoes_rate_l209_209053


namespace evaluate_expression_l209_209090

theorem evaluate_expression : 27^(- (2 / 3 : ℝ)) + Real.log 4 / Real.log 8 = 7 / 9 :=
by
  sorry

end evaluate_expression_l209_209090


namespace solve_a_b_l209_209578

theorem solve_a_b (a b : ℕ) (h₀ : 2 * a^2 = 3 * b^3) : ∃ k : ℕ, a = 18 * k^3 ∧ b = 6 * k^2 := 
sorry

end solve_a_b_l209_209578


namespace base8_product_is_zero_l209_209084

-- Define the base 10 number.
def n : ℕ := 8056

-- Define a theorem to state the product of the digits in the base 8 representation of n (8056) is 0.
theorem base8_product_is_zero : 
  let digits := [1, 7, 5, 7, 0] in
  digits.product = 0 := 
by 
  sorry

end base8_product_is_zero_l209_209084


namespace sequences_count_equals_fibonacci_n_21_l209_209923

noncomputable def increasing_sequences_count (n: ℕ) : ℕ := 
  -- Function to count the number of valid increasing sequences
  sorry

def fibonacci : ℕ → ℕ 
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem sequences_count_equals_fibonacci_n_21 :
  increasing_sequences_count 20 = fibonacci 21 :=
sorry

end sequences_count_equals_fibonacci_n_21_l209_209923


namespace prob_5_lt_X_lt_6_l209_209637

noncomputable def normal_cdf (μ σ : ℝ) (a b : ℝ) : ℝ :=
  let cdf_x := fun x => 1/2 * (1 + Real.erf ((x - μ) / (σ * Real.sqrt 2)))
  cdf_x b - cdf_x a

theorem prob_5_lt_X_lt_6 (μ σ : ℝ) (hμ : μ = 4) (hσ : σ = 1) :
    normal_cdf μ σ 5 6 = 0.1359 :=
by
  rw [hμ, hσ]
  have : normal_cdf 4 1 5 6 = 0.1359
  sorry
  exact this

end prob_5_lt_X_lt_6_l209_209637


namespace wendy_pictures_in_one_album_l209_209831

theorem wendy_pictures_in_one_album 
  (total_pictures : ℕ) (pictures_per_album : ℕ) (num_other_albums : ℕ)
  (h_total : total_pictures = 45) (h_pictures_per_album : pictures_per_album = 2) 
  (h_num_other_albums : num_other_albums = 9) : 
  ∃ (pictures_in_one_album : ℕ), pictures_in_one_album = 27 :=
by {
  sorry
}

end wendy_pictures_in_one_album_l209_209831


namespace _l209_209693

variables {A B C I H O X Y M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
[MetricSpace I] [MetricSpace H] [MetricSpace O] [MetricSpace X] [MetricSpace Y] [MetricSpace M]
[M ∈ midpoint (BC)] {perpendicular : B ⟶ C}

noncomputable def main_theorem :
  ∀ (I H O X Y A B C : Point) (m M : Point),
    I = circumcenter A B C ∧ H = orthocenter A B C ∧
    (AB < AC) ∧
    (exists m : Point, M = midpoint B C ∧ Line_perpendicular_through_point M BC ∧ Intersects M AB = X ∧ Tangent_circumcircle B = Y) →
    ∠ X O Y = ∠ A O B :=
by
  intros I H O X Y A B C M
  sorry

end _l209_209693


namespace second_rooster_weight_l209_209088

theorem second_rooster_weight (cost_per_kg : ℝ) (weight_1 : ℝ) (total_earnings : ℝ) (weight_2 : ℝ) :
  cost_per_kg = 0.5 →
  weight_1 = 30 →
  total_earnings = 35 →
  total_earnings = weight_1 * cost_per_kg + weight_2 * cost_per_kg →
  weight_2 = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end second_rooster_weight_l209_209088


namespace circumcircle_radius_l209_209289

-- Define the triangle and given conditions
variables (A B C : Type) [triangle A B C]
variables {a b c : ℝ}
variables (a_eq : a = 2) (b_eq : b = 3) (C_eq : C = π / 3)

-- State the goal
theorem circumcircle_radius (a b : ℝ) (C : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : C = π / 3) :
  ∃ R : ℝ, R = sqrt(21) / 3 :=
by
  sorry

end circumcircle_radius_l209_209289


namespace pyramid_volume_l209_209370

theorem pyramid_volume (AB BC PA : ℝ) 
  (h1 : AB = 10) 
  (h2 : BC = 5) 
  (h3 : PA = 8) 
  (h4 : ∀P A B : ℝ, P ≠ A → P ≠ B → A ≠ B → (⊥ P A B)) :
  (1/3) * (AB * BC) * PA = 400 / 3 := 
by
  subst h1
  subst h2
  subst h3
  sorry

end pyramid_volume_l209_209370


namespace eval_expression_l209_209916

theorem eval_expression : -30 + 12 * (8 / 4)^2 = 18 :=
by
  sorry

end eval_expression_l209_209916


namespace choose_5_starters_with_twins_l209_209762

theorem choose_5_starters_with_twins (n : ℕ) (k : ℕ) (twin1 twin2 : ℕ) :
  n = 12 ∧ k = 5 ∧ twin1 = 2 ∧ twin2 = (n - twin1) → 
  let team := (finset.range (twin2 + 1)).erase 0 -- Here twin1 is represented by 1 and twin2 by 2, rest start from 3 to 12
  in (team.card - twin1 = 10) ∧ (k - twin1 = 3) ∧ ((team.card - twin1).choose (k - twin1) = 120) :=
by
  sorry

end choose_5_starters_with_twins_l209_209762


namespace find_domain_of_f_l209_209790

def domain_f (f : ℝ → ℝ) (dom : ℝ → Prop) : Prop :=
  ∀ x, dom x → f x = sqrt (log x - 2)

theorem find_domain_of_f :
  domain_f (λ x => sqrt (log x - 2)) (λ x => e^2 ≤ x) := sorry

end find_domain_of_f_l209_209790


namespace optimized_cylinder_lateral_surface_area_l209_209654

theorem optimized_cylinder_lateral_surface_area :
  ∃ a : ℝ, (perimeter : ℝ -> ℝ -> ℝ := λ a b => 2 * (a + b))
∧ (lateral_surface_area : ℝ -> ℝ -> ℝ := λ a b => 2 * ℝ.pi * a * b)
∧ perimeter a (15 - a) = 30 
∧ lateral_surface_area a (15 - a)  
∧ ∀ b c, lateral_surface_area a (15 - a) ≥ lateral_surface_area b (15-b) 
∧ a = 7.5
∧ lateral_surface_area a (15 - a)= 112.5 * ℝ.pi := sorry

end optimized_cylinder_lateral_surface_area_l209_209654


namespace slope_arithmetic_sequence_of_focal_chord_angle_relation_when_perpendicular_l209_209653

variables {p : ℝ} (hp : p > 0)
def parabola (p : ℝ) (x y : ℝ) := y^2 = 2 * p * x

noncomputable def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def is_focal_chord (p : ℝ) (x1 y1 x2 y2 : ℝ) :=
  parabola p x1 y1 ∧ parabola p x2 y2

def is_on_directrix (p : ℝ) (M : ℝ × ℝ) := M.1 = -p / 2

theorem slope_arithmetic_sequence_of_focal_chord 
  {x_A y_A x_B y_B : ℝ} (M : ℝ × ℝ) (hM : is_on_directrix p M)
  (hA : parabola p x_A y_A) (hB : parabola p x_B y_B) :
  let k1 := (y_A - M.2) / (x_A + p / 2),
      k2 := (y_B - M.2) / (x_B + p / 2),
      k := M.2 / p in
  k1 + k2 = 2 * k := sorry

theorem angle_relation_when_perpendicular
  {x_A y_A x_B y_B : ℝ} (M : ℝ × ℝ) (hM : is_on_directrix p M)
  (hA : parabola p x_A y_A) (hB : parabola p x_B y_B)
  (h_perp : ((y_A - M.2) / (x_A + p / 2)) * ((y_B - M.2) / (x_B + p / 2)) = -1) :
  let θ_AMF := arctan ((y_A - M.2) / (x_A + p / 2)),
      θ_BMF := arctan ((y_B - M.2) / (x_B + p / 2)),
      θ_MFO := arctan (M.2 / p) in
  θ_MFO = abs (θ_BMF - θ_AMF) := sorry

end slope_arithmetic_sequence_of_focal_chord_angle_relation_when_perpendicular_l209_209653


namespace find_y_l209_209598

theorem find_y (y : ℝ) : (|y - 25| + |y - 23| = |2y - 46|) → y = 24 :=
by
  sorry

end find_y_l209_209598


namespace sum_of_squares_inequality_l209_209754

theorem sum_of_squares_inequality (n : ℕ) (hn : n ≥ 1) : 
  1 + (∑ k in finset.range n, 1 / (k + 2) ^ 2) < (2 * n + 1) / (n + 1) :=
sorry

end sum_of_squares_inequality_l209_209754


namespace total_goals_in_5_matches_proof_l209_209028

-- Given conditions
variables (A : ℝ) -- Let A be the average number of goals before the fifth match
variables (fourth_total_goals : ℝ) -- The total number of goals after 4 matches
variables (fifth_goals : ℝ := 4) -- The number of goals in the fifth match
variables (new_average : ℝ := A + 0.2) -- The new average after the fifth match
variables (total_matches : ℝ := 5) -- The total number of matches

-- Total goals after 5 matches should be 4A + 4
def total_goals_after_fifth : ℝ := fourth_total_goals + fifth_goals

-- The new average in terms of total goals after the fifth match
def new_avg_calc : ℝ := (total_goals_after_fifth) / total_matches

theorem total_goals_in_5_matches_proof (h : fourth_total_goals = 4 * A) (h_avg : new_avg_calc = new_average) : 
  total_goals_after_fifth = 16 :=
by
  -- Applying the conditions from problem's solution steps
  rw [h, new_avg_calc, total_goals_after_fifth] at h_avg
  have : (4 * A + 4) / 5 = A + 0.2 := h_avg
  linarith

end total_goals_in_5_matches_proof_l209_209028


namespace geom_sequence_f_sum_values_eq_l209_209629

def f (k : ℝ) (x : ℝ) : ℝ := k * x + 1

theorem geom_sequence_f (k : ℝ) :
  (f k 1, f k 4, f k 13).2^2 = (f k 1 * f k 13) :=
by
  sorry

theorem sum_values_eq (k : ℝ) (n : ℕ) (h : k ≠ 0) (g_seq : (f k 1, f k 4, f k 13).2^2 = (f k 1 * f k 13)) :
  (finset.range (n + 1)).sum (λ i, f k (2 * i)) = 3 * n + 2 * n^2 :=
by
  sorry

end geom_sequence_f_sum_values_eq_l209_209629


namespace arithmetic_sequence_S5_l209_209951

noncomputable def arithmetic_sum (a : ℕ → ℝ) : ℕ → ℝ 
| 0 := 0
| n := n * ((2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)

variables (a : ℕ → ℝ)
variables (S4 : arithmetic_sum a 4 = -4) 
variables (S6 : arithmetic_sum a 6 = 6)

theorem arithmetic_sequence_S5 : arithmetic_sum a 5 = 0 :=
by sorry

end arithmetic_sequence_S5_l209_209951


namespace additional_cost_per_kg_l209_209881

theorem additional_cost_per_kg (l m : ℝ) 
  (h1 : 168 = 30 * l + 3 * m) 
  (h2 : 186 = 30 * l + 6 * m) 
  (h3 : 20 * l = 100) : 
  m = 6 := 
by
  sorry

end additional_cost_per_kg_l209_209881


namespace reflection_matrix_values_l209_209906

theorem reflection_matrix_values (a b : ℚ) :
  let R : Matrix (Fin 2) (Fin 2) ℚ := ![![a, b], ![-(3/5), 4/5]] in
  R ⬝ R = (1 : Matrix (Fin 2) (Fin 2) ℚ) ↔ (a = -4/5 ∧ b = -3/5) :=
sorry

end reflection_matrix_values_l209_209906


namespace roots_sum_eq_product_l209_209970

theorem roots_sum_eq_product (m : ℝ) :
  (∀ x : ℝ, 2 * (x - 1) * (x - 3 * m) = x * (m - 4)) →
  (∀ a b : ℝ, 2 * a * b = 2 * (5 * m + 6) / -2 ∧ 2 * a * b = 6 * m / 2) →
  m = -2 / 3 :=
by
  sorry

end roots_sum_eq_product_l209_209970


namespace hyperbola_eccentricity_theorem_l209_209652

noncomputable def hyperbola_eccentricity_range (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : Prop :=
∃ c e : ℝ, 
  c = Real.sqrt (a^2 + b^2) ∧ 
  ∃ P : ℝ × ℝ, 
    let F1 := (-c, 0), F2 := (c, 0), O := (0, 0)
    in let M := ((c + P.1) / 2, P.2 / 2)
    in abs (Real.sqrt (M.1^2 + M.2^2)) = (1/5)*c ∧
       e = c / a ∧
       1 < e ∧ e ≤ 5 / 3

theorem hyperbola_eccentricity_theorem {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b) :
  hyperbola_eccentricity_range a b h1 h2 :=
sorry

end hyperbola_eccentricity_theorem_l209_209652


namespace normal_probability_interval_l209_209607

noncomputable theory
open real

theorem normal_probability_interval :
  ∀ (X : ℝ → ℝ),
  (∀ x, PDF_normal X x 0 1) →
  P_X (-1 < X < 2) = 0.8185 :=
by sorry

end normal_probability_interval_l209_209607


namespace odd_factors_of_360_l209_209188

theorem odd_factors_of_360 : ∃ n, n = 6 ∧ 
  ∀ (x : ℕ), (∃ (a b : ℕ), 360 = 2^a * 3^b * x ∧ ¬ even x ∧ 0 < x) → (∀ (m : ℕ), m ∣ x ↔ (m = 1 ∨ m = 3 ∨ m = 5 ∨ m = 9 ∨ m = 15 ∨ m = 45)) :=
by
  -- Proof
  sorry

end odd_factors_of_360_l209_209188


namespace sum_of_distances_to_R_l209_209938

variables (A B C D P Q R : Point)
variables (rA rB rC rD : ℝ)
variables (hAC : dist A C = 50)
variables (hBD : dist B D = 50)
variables (hPQ : dist P Q = 52)
variables (hR : midpoint R P Q)
variables (h1 : dist A P = rA)
variables (h2 : dist C P = rC)
variables (h3 : dist B Q = rB)
variables (h4 : dist D Q = rD)
variables (h5 : rA = (5/8 : ℝ) * rC)
variables (h6 : rB = (5/8 : ℝ) * rD)

theorem sum_of_distances_to_R : dist A R + dist B R + dist C R + dist D R = 100 :=
sorry

end sum_of_distances_to_R_l209_209938


namespace arithmetic_operations_l209_209461

theorem arithmetic_operations (n : ℕ) (h : n = 8000) :
  let x := (1 / 20) / 100 * n,
      y := (1 / 10) * n,
      z := y - x,
      w := z * (2 / 5),
      result := w / 4
  in result = 79.6 :=
by
  sorry

end arithmetic_operations_l209_209461


namespace hypotenuse_length_triangle_l209_209866

theorem hypotenuse_length_triangle (a b c : ℝ) (h1 : a + b + c = 40) (h2 : (1/2) * a * b = 30) 
  (h3 : a = b) : c = 2 * Real.sqrt 30 :=
by
  sorry

end hypotenuse_length_triangle_l209_209866


namespace smallest_n_divides_2016_l209_209594

theorem smallest_n_divides_2016 (n : ℕ) :
  2016 ∣ 20^n - 16^n ↔ n = 6 :=
by {
  have h2016 : 2016 = 2^5 * 3^2 * 7,
  have h_expr : 20^n - 16^n = 2^{2n} * (5^n - 4^n),
  -- Conditions decomposed from the problem
  have h_cond1 : 2^{2n} ≥ 2^5,
  have h_cond2 : 5^n ≡ 4^n [MOD 3^2],
  have h_cond3 : 5^n ≡ 4^n [MOD 7],
  -- We need to prove n = 6
  sorry
}

end smallest_n_divides_2016_l209_209594


namespace vector_subtraction_l209_209137

variables (e1 e2 : Type) [AddCommGroup e1] [AddCommGroup e2]
variables (a b : e1)
variables (s1 s2 t1 t2 r1 r2 : ℤ)

-- Define the vector definitions
def a := e1 + 2 • e2
def b := 3 • e1 - e2

-- State the theorem
theorem vector_subtraction :
  3 • a - 2 • b = -3 • e1 + 8 • e2 :=
by sorry

end vector_subtraction_l209_209137


namespace steve_travel_time_l209_209406

noncomputable def total_travel_time (distance: ℕ) (speed_to_work: ℕ) (speed_back: ℕ) : ℕ :=
  (distance / speed_to_work) + (distance / speed_back)

theorem steve_travel_time : 
  ∀ (distance speed_back speed_to_work : ℕ), 
  (speed_to_work = speed_back / 2) → 
  speed_back = 15 → 
  distance = 30 → 
  total_travel_time distance speed_to_work speed_back = 6 := 
by
  intros
  rw [total_travel_time]
  sorry

end steve_travel_time_l209_209406


namespace prob_less_than_8_prob_at_least_7_l209_209639

def prob_9_or_above : ℝ := 0.56
def prob_8 : ℝ := 0.22
def prob_7 : ℝ := 0.12

theorem prob_less_than_8 : prob_7 + (1 - prob_9_or_above - prob_8) = 0.22 := 
sorry

theorem prob_at_least_7 : prob_9_or_above + prob_8 + prob_7 = 0.9 := 
sorry

end prob_less_than_8_prob_at_least_7_l209_209639


namespace find_sin_cos_relation_l209_209630

noncomputable def condition (x : ℝ) : Prop :=
  (π / 2 < x ∧ x < π) ∧ (tan x)^2 + 3 * tan x - 4 = 0

theorem find_sin_cos_relation (x : ℝ) (h : condition x) :
  (sin x + cos x) / (2 * sin x - cos x) = 1 / 3 :=
sorry

end find_sin_cos_relation_l209_209630


namespace calculate_expression_l209_209893

theorem calculate_expression : 
  (3^2 - 2 * 3) - (5^2 - 2 * 5) + (7^2 - 2 * 7) = 23 := 
by sorry

end calculate_expression_l209_209893


namespace find_z_l209_209944

theorem find_z (z : ℂ) (h : (Complex.I * z = 4 + 3 * Complex.I)) : z = 3 - 4 * Complex.I :=
by
  sorry

end find_z_l209_209944


namespace curve_equation_and_inclination_angle_PA_PB_sum_l209_209702

open Real

theorem curve_equation_and_inclination_angle
  (C_param : ∃ α : ℝ, ∀ α, (x = 2 * cos α) ∧ (y = sin α))
  (l_polar : ∃ θ ρ : ℝ, ∀ θ ρ, (ρ * sin (θ - π / 4) = sqrt 2 / 2)):
  (∃ x y : ℝ, ∀ x y, (x^2 / 4 + y^2 = 1)) ∧ (atan 1 = π / 4) :=
sorry

theorem PA_PB_sum
  (P : ∃ (x y : ℝ), x = 0 ∧ y = 1)
  (curve_eq : ∃ x y : ℝ, x^2 / 4 + y^2 = 1)
  (line_eq : ∃ x y: ℝ, x - y + 1 = 0)
  (t_roots : ∃ t_A t_B : ℝ, (t_A + t_B = -8 * sqrt 2 / 5)):
  abs 0 -(x t_A) + abs 0 -(y t_A) + abs (x t_B) + abs (y t_B) = 8 * sqrt 2 / 5 :=
sorry

end curve_equation_and_inclination_angle_PA_PB_sum_l209_209702


namespace decrease_percent_in_revenue_l209_209008

theorem decrease_percent_in_revenue 
  (T C : ℝ) 
  (original_revenue : ℝ := T * C)
  (new_tax : ℝ := 0.80 * T)
  (new_consumption : ℝ := 1.15 * C)
  (new_revenue : ℝ := new_tax * new_consumption) :
  ((original_revenue - new_revenue) / original_revenue) * 100 = 8 := 
sorry

end decrease_percent_in_revenue_l209_209008


namespace angle_between_hands_at_3_40_l209_209883

def degrees_per_minute_minute_hand := 360 / 60
def minutes_passed := 40
def degrees_minute_hand := degrees_per_minute_minute_hand * minutes_passed -- 240 degrees

def degrees_per_hour_hour_hand := 360 / 12
def hours_passed := 3
def degrees_hour_hand_at_hour := degrees_per_hour_hour_hand * hours_passed -- 90 degrees

def degrees_per_minute_hour_hand := degrees_per_hour_hour_hand / 60
def degrees_hour_hand_additional := degrees_per_minute_hour_hand * minutes_passed -- 20 degrees

def total_degrees_hour_hand := degrees_hour_hand_at_hour + degrees_hour_hand_additional -- 110 degrees

def expected_angle_between_hands := 130

theorem angle_between_hands_at_3_40
  (h1: degrees_minute_hand = 240)
  (h2: total_degrees_hour_hand = 110):
  (degrees_minute_hand - total_degrees_hour_hand = expected_angle_between_hands) :=
by
  sorry

end angle_between_hands_at_3_40_l209_209883


namespace hallie_read_pages_third_day_more_than_second_day_l209_209662

theorem hallie_read_pages_third_day_more_than_second_day :
  ∀ (d1 d2 d3 d4 : ℕ),
  d1 = 63 →
  d2 = 2 * d1 →
  d4 = 29 →
  d1 + d2 + d3 + d4 = 354 →
  (d3 - d2) = 10 :=
by
  intros d1 d2 d3 d4 h1 h2 h4 h_sum
  sorry

end hallie_read_pages_third_day_more_than_second_day_l209_209662


namespace remaining_surface_area_l209_209330

theorem remaining_surface_area (room_edge_length : ℕ) (hole_edge_length : ℕ) :
  room_edge_length = 10 → hole_edge_length = 2 → 
  let total_surface_area := 6 * room_edge_length^2,
      removed_surface_area := 3 * hole_edge_length^2,
      remaining_surface_area := total_surface_area - removed_surface_area
  in remaining_surface_area = 588 :=
by
  intros h1 h2
  let total_surface_area := 6 * room_edge_length^2
  let removed_surface_area := 3 * hole_edge_length^2
  let remaining_surface_area := total_surface_area - removed_surface_area
  sorry

end remaining_surface_area_l209_209330


namespace triangle_area_inequality_l209_209326

theorem triangle_area_inequality {A B C P A₁ B₁ C₁ : Type*}
(ABC : Triangle A B C)
(hP : P \in interior ABC)
(hA₁ : A₁ = line_intersection (line_through A P) (line_through B C))
(hB₁ : B₁ = line_intersection (line_through B P) (line_through A C))
(hC₁ : C₁ = line_intersection (line_through C P) (line_through A B))
(S₁ S₂ S₃ : ℝ)
(h_order : S₁ ≤ S₂ ∧ S₂ ≤ S₃)
(S : ℝ)
(h_triangle_areas : S = area_triangle A₁ B₁ C₁)
(h_areas : S₁ = area_triangle A B₁ C₁ ∧ S₂ = area_triangle A₁ B C₁ ∧ S₃ = area_triangle A₁ B₁ C) :
sqrt(S₁ * S₂) ≤ S ∧ S ≤ sqrt(S₂ * S₃) :=
sorry

end triangle_area_inequality_l209_209326


namespace correct_statement_l209_209998

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l209_209998


namespace max_S_n_l209_209960

-- Define the arithmetic sequence and the conditions
variables {a : ℕ → ℤ} 
noncomputable def d : ℤ := -2
noncomputable def a₁ : ℤ := 39

-- Arithmetic sequence definition
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom cond1 : a 0 = a₁
axiom cond2 : a 2 = a 0 + 2*d
axiom cond3 : a 4 = a 0 + 4*d
axiom cond4 : a 1 = a 0 + d
axiom cond5 : a 3 = a 0 + 3*d
axiom cond6 : a 5 = a 0 + 5*d
axiom sum1 : a₁ + a₂ + a₃ = 105
axiom sum2 : a₄ + a₅ + a₆ = 99

-- Sum of the first n terms of the sequence
noncomputable def S_n (n : ℕ) : ℤ := (n * (a₁ * 2 + (n - 1) * d)) / 2

-- Maximizing S_n at n = 20
theorem max_S_n : ∀ n : ℕ, S_n n ≤ S_n 20 :=
begin
  sorry
end

end max_S_n_l209_209960


namespace solve_equation_l209_209782

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x^2 - 1 ≠ 0) : (x / (x - 1) = 2 / (x^2 - 1)) → (x = -2) :=
by
  intro h
  sorry

end solve_equation_l209_209782


namespace max_students_is_273_l209_209051

section AuditoriumSeating

def seats_in_row (i : ℕ) : ℕ := 14 + i

def max_students_in_row (i : ℕ) : ℕ := (seats_in_row i + 1) / 2

def total_max_students : ℕ :=
  ∑ i in Finset.range 20, max_students_in_row (i + 1)

theorem max_students_is_273 : total_max_students = 273 := by
  sorry

end AuditoriumSeating

end max_students_is_273_l209_209051


namespace number_of_odd_factors_of_360_l209_209205

theorem number_of_odd_factors_of_360 : 
  let factors (n : ℕ) := {d : ℕ | d ∣ n}
  let odd (d : ℕ) := d % 2 = 1
  let is_factor (n : ℕ) (d : ℕ) := d ∣ n
  let number_of_odd_factors (n : ℕ) := (factors n).count odd
  in number_of_odd_factors 360 = 6 :=
begin
  sorry
end

end number_of_odd_factors_of_360_l209_209205


namespace glassware_damage_l209_209520

theorem glassware_damage
  (total_glassware : ℕ)
  (fee_per_undamaged : ℚ)
  (compensation_per_damaged : ℚ)
  (total_fee_received : ℚ)
  (damaged_glassware : ℕ) :
  total_glassware = 1500 →
  fee_per_undamaged = 2.5 →
  compensation_per_damaged = 3 →
  total_fee_received = 3618 →
  2.5 * (1500 - damaged_glassware) - 3 * damaged_glassware = 3618 →
  damaged_glassware = 24 :=
by {
  intros h_total h_fee h_comp h_received h_eq,
  sorry
 }

end glassware_damage_l209_209520


namespace correct_statement_l209_209994

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l209_209994


namespace secant_length_problem_l209_209605

theorem secant_length_problem (tangent_length : ℝ) (internal_segment_length : ℝ) (external_segment_length : ℝ) 
    (h1 : tangent_length = 18) (h2 : internal_segment_length = 27) : external_segment_length = 9 :=
by
  sorry

end secant_length_problem_l209_209605


namespace jessy_extra_minutes_per_day_l209_209716

theorem jessy_extra_minutes_per_day :
  let total_pages := 140
  let days_in_week := 7
  let initial_speed := 10  -- pages per hour
  let initial_time_per_session := 30 / 60  -- hours (since 30 minutes)
  let sessions_per_day := 2
  let mid_week_days := days_in_week / 2
  let new_speed := 15  -- pages per hour
  let remaining_days := days_in_week - mid_week_days
  let initial_daily_reading := initial_speed * initial_time_per_session * sessions_per_day
  let pages_read_by_mid_week := initial_daily_reading * mid_week_days
  let remaining_pages := total_pages - pages_read_by_mid_week
  let new_daily_reading := remaining_pages / remaining_days
  let extra_daily_reading := new_daily_reading - initial_daily_reading
  in extra_daily_reading * 60 = 60 :=
by
  sorry

end jessy_extra_minutes_per_day_l209_209716


namespace simplify_and_evaluate_expression_l209_209385

theorem simplify_and_evaluate_expression (a b : ℝ) (h : (a + 1)^2 + |b + 1| = 0) : 
  1 - (a^2 + 2 * a * b + b^2) / (a^2 - a * b) / ((a + b) / (a - b)) = -1 := 
sorry

end simplify_and_evaluate_expression_l209_209385


namespace flagpole_height_proof_l209_209499

-- Definitions based on given conditions
variables {A B C D E : Type}
variables (dist_AC : ℝ) (dist_AD : ℝ) (height_Joe : ℝ)
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]

-- Given conditions
def flagpole_conditions (dist_AC : ℝ) (dist_AD : ℝ) (height_Joe : ℝ) : Prop :=
  dist_AC = 4 ∧ dist_AD = 3 ∧ height_Joe = 1.8

-- Function to denote height of flagpole
def height_flagpole (dist_AC dist_AD height_Joe : ℝ) : ℝ :=
  let dist_DC := dist_AC - dist_AD in
  let ratio := height_Joe / dist_DC in
  ratio * dist_AC

-- The theorem to be proved
theorem flagpole_height_proof (h : flagpole_conditions dist_AC dist_AD height_Joe) :
  height_flagpole dist_AC dist_AD height_Joe = 7.2 :=
by {
  sorry -- Proof to be filled in
}

end flagpole_height_proof_l209_209499


namespace decimal_150th_place_7_div_11_l209_209836

theorem decimal_150th_place_7_div_11 :
  let decimal_rep : ℕ → ℕ := λ n, ite (n % 2 = 0) 3 6
  (decimal_rep 150) = 3 :=
by
  -- Proving the statement directly here
  simp [decimal_rep]
  sorry

end decimal_150th_place_7_div_11_l209_209836


namespace minimum_a_for_decreasing_function_range_of_a_for_condition_l209_209644

noncomputable def f (x a : ℝ) : ℝ := x / Real.log x - a * x
noncomputable def f_prime (x a : ℝ) : ℝ := (Real.log x - 1) / (Real.log x) ^ 2 - a

theorem minimum_a_for_decreasing_function (a : ℝ) (h : a > 0) :
  (∀ x > 1, f_prime x a ≤ 0) ↔ a ≥ 1 / 4 := sorry

theorem range_of_a_for_condition (a : ℝ) (h : a > 0) :
  (∃ x1 x2 ∈ set.Icc (Real.exp 1) (Real.exp 2), f x1 a ≤ f_prime x2 a + a) ↔ a ≥ 1 / 2 - 1 / (4 * Real.exp 2 ^ 2) := sorry

end minimum_a_for_decreasing_function_range_of_a_for_condition_l209_209644


namespace compare_abc_l209_209333
-- Necessary import to include all required mathematical libraries

-- Definitions of the variables based on provided conditions
def a := (1 / 2) ^ (1 / 3)
def b := (1 / 3) ^ (1 / 2)
def c := Real.log (3 / Real.pi)

-- Statement of the theorem to be proven
theorem compare_abc : c < b ∧ b < a :=
by
  sorry

end compare_abc_l209_209333


namespace condition_for_s_eq_2CP2_l209_209329

open Real

variables {A B C P : Point} (AC BC AB AP PB CP : ℝ)
variables (k : ℝ)
noncomputable def right_triangle (A B C : Point) : Prop :=
  AC = 3 ∧ BC = 4 ∧ AB = 5 ∧ hypotenuse AB

noncomputable def s (A B P: Point) : ℝ :=
  (dist A P)^2 + (dist P B)^2

noncomputable def CP_distance (C P : Point) : ℝ :=
  ∣k∣ / 5

theorem condition_for_s_eq_2CP2 {A B C P : Point} (h : right_triangle A B C)
  (line_parallel : ∀ P : Point, is_on_line P (-4/3) k)
  (hP : dist A P = dist P B): s A B P = 2 * (CP_distance C P) ^ 2 ↔ k = 5 ∨ k = -5 := by
  sorry

end condition_for_s_eq_2CP2_l209_209329


namespace solution_sets_differ_l209_209470

theorem solution_sets_differ (x : ℝ) :
  (∃ x, (5 * x > 10) ∧ (3 * x > 6)) ↔ (x > 2) ∧ 
  (∃ x, (6 * x - 9 < 3 * x + 6) ∧ (x < 5)) ↔ (x < 5) ∧ 
  (∃ x, (x < -2) ∧ (-14 * x > 28)) ↔ (x < -2) ∧ 
  (¬(∃ x, (x - 7 < 2 * x + 8) ∧ (x > 15)) ↔ (x > -15) ∧ (x > 15)) := 
by
  tidy
  sorry

end solution_sets_differ_l209_209470


namespace pointed_star_interior_angles_sum_l209_209033

/-
  A new "n-pointed star" is formed by extending every third side of an n-sided convex polygon, where n ≥ 6.
  The sides of the polygon are numbered consecutively from 1 to n.
  The extensions of sides k and k+3, considering indices modulo n, intersect to form vertices of the star.
  We aim to prove that the sum of the interior angles at these vertices is 180°(n-2).
-/
theorem pointed_star_interior_angles_sum (n : ℕ) (hn : n ≥ 6) : 
  let S := 180 * (n - 2) in
  S = 180 * (n - 2) := by
  sorry

end pointed_star_interior_angles_sum_l209_209033


namespace number_of_spinsters_l209_209007

-- Given conditions
variables (S C : ℕ)
axiom ratio_condition : S / C = 2 / 9
axiom difference_condition : C = S + 63

-- Theorem to prove
theorem number_of_spinsters : S = 18 :=
sorry

end number_of_spinsters_l209_209007


namespace math_problems_l209_209942

variable {a b : ℝ}

theorem math_problems (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b = 1) :
  (1 / 2 < 2^(a - sqrt b) ∧ 2^(a - sqrt b) < 2) ∧ (a + sqrt b ≤ sqrt 2) :=
by
  sorry

end math_problems_l209_209942


namespace triangle_right_l209_209815

theorem triangle_right (a b c : ℝ) (h₀ : a ≠ c) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : ∃ x₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + 2 * a * x₀ + b^2 = 0 ∧ x₀^2 + 2 * c * x₀ - b^2 = 0) :
  a^2 = b^2 + c^2 := 
sorry

end triangle_right_l209_209815


namespace limit_of_function_l209_209543

-- Define the function and its limit
noncomputable def f (x : ℝ) := (Real.tan (4 * x) / x) ^ (2 + x)

-- State the theorem to prove the limit
theorem limit_of_function : (Real.limit (λ x, f x) 0) = 16 :=
by
  sorry

end limit_of_function_l209_209543


namespace angelina_speed_l209_209003

theorem angelina_speed (v : ℝ) (h1 : 200 / v - 50 = 300 / (2 * v)) : 2 * v = 2 := 
by
  sorry

end angelina_speed_l209_209003


namespace leg_equals_midsegment_l209_209019

theorem leg_equals_midsegment
  (a b c : ℝ)
  (h1 : a + b = 2 * c)
  (h2 : ∀ (a b : ℝ), midsegment a b = (a + b) / 2) : 
  c = (a + b) / 2 := by
  sorry

end leg_equals_midsegment_l209_209019


namespace number_of_solutions_l209_209591

theorem number_of_solutions (sin_exp_eq : ∀ x, sin x = (1 / 3) ^ x):
  ∃ n, 0 ≤ n < 150 ∧ n = 150 :=
begin
  sorry
end

end number_of_solutions_l209_209591


namespace area_of_triangle_DEF_l209_209587

-- Definitions based on conditions
def angle_D : ℝ := 45
def side_DE : ℝ := 8
def is_right_triangle : Prop := true

-- Mathematically equivalent proof problem
theorem area_of_triangle_DEF : 
  (angle_D = 45) → 
  (side_DE = 8) → 
  (is_right_triangle) → 
  ∃ (A : ℝ), A = 32 :=
by 
  intro h_angle_D h_side_DE h_right_triangle
  use 32
  sorry

end area_of_triangle_DEF_l209_209587


namespace obtuse_triangle_values_l209_209810

theorem obtuse_triangle_values :
  (∃ (k : ℕ), k > 0 ∧
    (let a := 12 in let b := 16 in
    ((b > a ∧ ∀ (k : ℕ), k > 4 ∧ k < 11 → b > a ∧ b^2 > a^2 + k^2) ∨
     (k > 16 ∧ k < 28 ∧ k * k > 400))
  )) = 13 := sorry

end obtuse_triangle_values_l209_209810


namespace trigonometric_identity_l209_209138

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin (π / 2 + θ) - Real.cos (π - θ)) / (Real.cos (3 * π / 2 - θ) - Real.sin (π - θ)) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l209_209138


namespace ball_bounces_l209_209852

theorem ball_bounces (k : ℕ) :
  1500 * (2 / 3 : ℝ)^k < 2 ↔ k ≥ 19 :=
sorry

end ball_bounces_l209_209852


namespace die_volume_l209_209785

theorem die_volume (area_of_side : ℝ) (h_area : area_of_side = 64) : 
  ∃ (volume : ℝ), volume = 512 :=
by
  let side_length := real.sqrt area_of_side
  have h_side_length : side_length = 8 := by
    rw [h_area, real.sqrt_eq_rpow, real.rpow_nat_cast, real.rpow_self]
    norm_num
  let volume := side_length ^ 3
  use volume
  rw [h_side_length, pow_succ, pow_one, pow_one]
  norm_num

end die_volume_l209_209785


namespace segment_length_after_reflection_l209_209443

noncomputable def length_of_reflected_segment 
  (D E F : (ℝ × ℝ)) (Dy Ey Fy : ℝ) 
  (reflect_y : (-D.1, D.2) → (D'.1, D.2) ) : ℝ := 
  let F := (-4, 3)
  let F' := (4, 3)
  real.dist F.1 F'.1

theorem segment_length_after_reflection :
  ∀ (F : ℝ × ℝ), 
  F = (-4, 3) → 
  let F' := (4, 3)
  length_of_reflected_segment D E F Fy F' = 8 :=
λ F hF, by
  sorry

end segment_length_after_reflection_l209_209443


namespace compound_propositions_l209_209656

def divides (a b : Nat) : Prop := ∃ k : Nat, b = k * a

-- Define the propositions p and q
def p : Prop := divides 6 12
def q : Prop := divides 6 24

-- Prove the compound propositions
theorem compound_propositions :
  (p ∨ q) ∧ (p ∧ q) ∧ ¬¬p :=
by
  -- We are proving three statements:
  -- 1. "p or q" is true.
  -- 2. "p and q" is true.
  -- 3. "not p" is false (which is equivalent to "¬¬p" being true).
  -- The actual proof will be constructed here.
  sorry

end compound_propositions_l209_209656


namespace inequality_holds_l209_209635

def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def monotonic_on (f : ℝ → ℝ) (I : set ℝ) := 
  (∀ a b, a ∈ I ∧ b ∈ I ∧ a < b → f a ≤ f b) ∨
  (∀ a b, a ∈ I ∧ b ∈ I ∧ a < b → f a ≥ f b)

variable f : ℝ → ℝ
variable h_even : even_function f
variable h_mono : monotonic_on f (set.Ico 0 6)
variable h_lt : f (-2) < f 1

theorem inequality_holds : f 5 < f (-3) ∧ f (-3) < f (-1) := 
by sorry

end inequality_holds_l209_209635


namespace water_volume_to_sea_per_minute_l209_209867

theorem water_volume_to_sea_per_minute (depth width : ℝ) (flow_rate_kmph : ℝ) 
  (h_depth : depth = 2) 
  (h_width : width = 45) 
  (h_flow_rate_kmph : flow_rate_kmph = 5) : 
  ∃ (volume_per_minute : ℝ), volume_per_minute ≈ 7499.7 := by
  -- Definitions using given conditions
  let area := depth * width     -- cross-sectional area
  let flow_rate_mpm := flow_rate_kmph * 1000 / 60  -- convert kmph to m/min
  let volume_per_minute := area * flow_rate_mpm  -- volume per minute
  -- Asserting the desired result
  use volume_per_minute
  have h_area : area = 90 := by 
    calc area = depth * width : rfl
    ... = 2 * 45 : by rw [h_depth, h_width]
    ... = 90 : rfl
  have h_flow_rate_mpm : flow_rate_mpm = 83.33 := by 
    calc flow_rate_mpm = flow_rate_kmph * 1000 / 60 : rfl
    ... = 5 * 1000 / 60 : by rw h_flow_rate_kmph
    ... = 5000 / 60 : rfl
    ... ≈ 83.33 : by norm_num
  have h_volume_per_minute : volume_per_minute = area * flow_rate_mpm := rfl
  calc volume_per_minute = area * flow_rate_mpm : rfl
  ... = 90 * 83.33 : by rw [h_area, h_flow_rate_mpm]
  ... = 7499.7 : by norm_num
  -- Assert the volume is approximately 7499.7
  exact rat.approx_eq_approx volume_per_minute 7499.7 sorry

end water_volume_to_sea_per_minute_l209_209867


namespace relationship_among_abc_l209_209269

noncomputable def a : ℝ := Real.ln 2
noncomputable def b : ℝ := 5^(-1/2 : ℝ)
noncomputable def c : ℝ := (1 / 4) * ∫ x in 0..Real.pi, Real.sin x

theorem relationship_among_abc : b < c ∧ c < a :=
by
  sorry

end relationship_among_abc_l209_209269


namespace carrie_profit_l209_209066

def hours_per_day : ℕ := 2
def days_worked : ℕ := 4
def hourly_rate : ℕ := 22
def cost_of_supplies : ℕ := 54
def total_hours_worked : ℕ := hours_per_day * days_worked
def total_payment : ℕ := hourly_rate * total_hours_worked
def profit : ℕ := total_payment - cost_of_supplies

theorem carrie_profit : profit = 122 := by
  sorry

end carrie_profit_l209_209066


namespace area_of_gray_region_l209_209552

/-- 
Let Circle C be centered at (5, 4) with radius 5.
Let Circle D be centered at (15, 4) with radius 5.
The area of the gray region bound by the circles and the x-axis is 40 - 25π square units.
-/
theorem area_of_gray_region : 
  let C_center := (5, 4) in
  let C_radius := 5 in
  let D_center := (15, 4) in
  let D_radius := 5 in
  let rectangle_area := 10 * 4 in
  let sector_area := (1 / 2) * Real.pi * (5 ^ 2) in
  rectangle_area - 2 * sector_area = 40 - 25 * Real.pi := 
sorry

end area_of_gray_region_l209_209552


namespace problem_U_complement_eq_l209_209989

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l209_209989


namespace find_CD_l209_209705

noncomputable theory

-- define the lengths of sides and the ang
variable (CD FG DE GH : ℝ)
variable (angle_C angle_F : ℝ)

-- Conditions are that triangles are similar by SAS with the given sides and angle.
def similar_TRIANGLES_CDE_AND_FGH : Prop :=
  angle_C = 120 ∧
  angle_F = 120 ∧ 
  DE = 21 ∧ 
  GH = 7.5 ∧ 
  FG = 4.5

-- The theorem statement to prove
theorem find_CD (h : similar_TRIANGLES_CDE_AND_FGH CD FG DE GH angle_C angle_F) : CD = 12.6 := 
sorry

end find_CD_l209_209705


namespace angle_Q_is_72_degrees_l209_209381

-- Definitions and conditions based on the given problem
variables {A B C D E F G H I J Q : Type} 
variables [Decagon : RegularDecagon A B C D E F G H I J]
variables [ExtendAH : AQ = Line(A, H)]
variables [ExtendEF : EQ = Line(E, F)]
variables [ParallelABGH : ∥ A B ∥ G H]
variables [ParallelEFCD : ∥ E F ∥ C D]

-- Main theorem
theorem angle_Q_is_72_degrees :
  ∠ Q = 72 :=
sorry

end angle_Q_is_72_degrees_l209_209381


namespace max_possible_M_l209_209688

variable (n : ℕ)

/-- In a 2n x 2n grid where each cell is filled with either 1 or -1, 
there are exactly 2n^2 cells filled with 1 and 2n^2 cells filled with -1.
Let M be the minimum value of the maximum absolute value of the sum of numbers
in any row or column. The maximum possible value of M is n. -/
theorem max_possible_M (n : ℕ) (grid : Fin 2n → Fin 2n → ℤ)
  (h1 : ∀ i j, grid i j = 1 ∨ grid i j = -1)
  (h2 : ∑ i, ∑ j, grid i j = 0)
  (h3 : ∃ p : Fin 2n → Fin 2n → Bool, 
           (∑ i j, ite (p i j) 1 0 = 2n^2) ∧ 
           (∑ i j, ite (¬p i j) 1 0 = 2n^2)) :
  ∃ M : ℕ, M = n :=
begin
  sorry
end

end max_possible_M_l209_209688


namespace rectangle_area_l209_209805

-- Define the length and width of the rectangle based on given ratio
def length (k: ℝ) := 5 * k
def width (k: ℝ) := 2 * k

-- The perimeter condition
def perimeter (k: ℝ) := 2 * (length k) + 2 * (width k) = 280

-- The diagonal condition
def diagonal_condition (k: ℝ) := (width k) * Real.sqrt 2 = (length k) / 2

-- The area of the rectangle
def area (k: ℝ) := (length k) * (width k)

-- The main theorem to be proven
theorem rectangle_area : ∃ k: ℝ, perimeter k ∧ diagonal_condition k ∧ area k = 4000 :=
by
  sorry

end rectangle_area_l209_209805


namespace simplest_root_is_optionB_l209_209048

-- Define the conditions
def optionA := Real.sqrt 4
def optionB := Real.sqrt 5
def optionC := Real.sqrt 8
def optionD := Real.sqrt (1 / 2)

-- Define the correct answer
def simplest_square_root := optionB

-- Prove that the simplest square root is optionB
theorem simplest_root_is_optionB : 
  (Real.sqrt 4 = 2) →
  (Real.sqrt 5 = Real.sqrt 5) → -- cannot be simplified further
  (Real.sqrt 8 = 2 * Real.sqrt 2) →
  (Real.sqrt (1 / 2) = Real.sqrt 2 / 2) →
  simplest_square_root = Real.sqrt 5 :=
by
  intros hA hB hC hD
  -- Use provided facts to conclude
  sorry

end simplest_root_is_optionB_l209_209048


namespace expression_correct_l209_209929

def E (x : ℕ) : ℕ := x - 1

theorem expression_correct (x : ℕ) (h : x = 4) : 7 * E x = 21 :=
by 
  rw [h, E]
  exact rfl

end expression_correct_l209_209929


namespace quadratic_no_roots_c_positive_l209_209807

theorem quadratic_no_roots_c_positive
  (a b c : ℝ)
  (h_no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0)
  (h_positive : a + b + c > 0) :
  c > 0 :=
sorry

end quadratic_no_roots_c_positive_l209_209807


namespace sarah_overall_score_l209_209377

/-- Given Sarah's scores on three tests, we can determine her overall score on a combined 100-problem test. -/
theorem sarah_overall_score :
  let score1 := 0.60 * 15,
      score2 := 0.75 * 40,
      score3 := (0.85 * 45).floor in
  (score1 + score2 + score3) / 100 = 0.77 :=
by 
  sorry

end sarah_overall_score_l209_209377


namespace Vasya_has_more_ways_l209_209364

def king (pos: ℕ × ℕ) (positions: Finset (ℕ × ℕ)) : Prop :=
  ∀ (x y: ℕ × ℕ), x ≠ y ∧ x ∈ positions ∧ y ∈ positions → 
  (abs (x.1 - y.1) > 1 ∨ abs (x.2 - y.2) > 1)

def PetyaBoard : Finset (ℕ × ℕ) := 
{p | p.1 < 100 ∧ p.2 < 50}

def VasyaBoard : Finset (ℕ × ℕ) := 
{p | p.1 < 100 ∧ p.2 < 100 ∧ (p.1 + p.2) % 2 = 0}

theorem Vasya_has_more_ways :
  (∃ ps : Finset (ℕ × ℕ), ps.card = 500 ∧ king ps) → 
  (∃ ps : Finset (ℕ × ℕ), ps.card = 500 ∧ king ps) :=
sorry

end Vasya_has_more_ways_l209_209364


namespace coin_flip_probability_l209_209600

open ProbabilityTheory

theorem coin_flip_probability :
  let S := { outcomes | List.length (List.filter (λ x => x = true) outcomes) > List.length (List.filter (λ x => x = false) outcomes) }
  Pr (S : Set (List (Bool))) = 1 / 2 :=
by
  sorry

end coin_flip_probability_l209_209600


namespace derivative_of_y_l209_209098

noncomputable def y (x : ℝ) : ℝ := (1 / Real.sqrt 2) * Real.log (Real.sqrt 2 * Real.tan x + Real.sqrt (1 + 2 * (Real.tan x)^2))

theorem derivative_of_y (x : ℝ) (h : x ≠ (π / 2 + k * π) for integer k) : 
  deriv (λ x, y x) x = 1 / ((Real.cos x)^2 * Real.sqrt (1 + 2 * (Real.tan x)^2)) :=
sorry

end derivative_of_y_l209_209098


namespace number_of_positive_area_triangles_l209_209250

theorem number_of_positive_area_triangles (s : Finset (ℕ × ℕ)) (h : s = {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}) :
  (s.powerset.filter λ t, t.card = 3 ∧ ¬∃ a b c, a = b ∨ b = c ∨ c = a).card = 2160 :=
by {
  sorry
}

end number_of_positive_area_triangles_l209_209250


namespace rotate_parabola_180_l209_209375

theorem rotate_parabola_180 (x: ℝ) : 
  let original_parabola := λ x, 2 * (x - 3)^2 - 2,
      rotated_parabola := λ x, -2 * (x - 3)^2 - 2 in
  original_parabola x = rotated_parabola x :=
sorry

end rotate_parabola_180_l209_209375


namespace rajans_share_calculation_l209_209368

-- Define the investment amounts and durations
def rajans_investment {α : Type} [CommRing α] := 20000
def rakeshs_investment {α : Type} [CommRing α] := 25000
def mukeshs_investment {α : Type} [CommRing α] := 15000

def rajans_duration {α : Type} [CommRing α] := 12
def rakeshs_duration {α : Type} [CommRing α] := 4
def mukeshs_duration {α : Type} [CommRing α] := 8

-- Define the total profit
def total_profit {α : Type} [CommRing α] := 4600

-- Define the investment ratios
def rajans_investment_ratio {α : Type} [CommRing α] :=
  rajans_investment * rajans_duration

def rakeshs_investment_ratio {α : Type} [CommRing α] :=
  rakeshs_investment * rakeshs_duration

def mukeshs_investment_ratio {α : Type} [CommRing α] :=
  mukeshs_investment * mukeshs_duration

def total_investment_ratio {α : Type} [CommRing α] :=
  rajans_investment_ratio + rakeshs_investment_ratio + mukeshs_investment_ratio

-- Define Rajan's share of the profit
def rajans_share_of_profit {α : Type} [CommRing α] :=
  rajans_investment_ratio / total_investment_ratio * total_profit

-- The theorem stating the main question to be proven
theorem rajans_share_calculation {α : Type} [CommRing α] :
  rajans_share_of_profit = (240000 / 460000) * 4600 :=
sorry

end rajans_share_calculation_l209_209368


namespace geometric_sequence_third_term_l209_209634

theorem geometric_sequence_third_term (a : ℕ → ℕ) (x : ℕ) (h1 : a 1 = 4) (h2 : a 2 = 6) (h3 : a 3 = x) (h_geom : ∀ n, a (n + 1) = a n * r) :
  x = 9 := 
sorry

end geometric_sequence_third_term_l209_209634


namespace number_of_odd_factors_of_360_l209_209209

theorem number_of_odd_factors_of_360 : 
  let factors (n : ℕ) := {d : ℕ | d ∣ n}
  let odd (d : ℕ) := d % 2 = 1
  let is_factor (n : ℕ) (d : ℕ) := d ∣ n
  let number_of_odd_factors (n : ℕ) := (factors n).count odd
  in number_of_odd_factors 360 = 6 :=
begin
  sorry
end

end number_of_odd_factors_of_360_l209_209209


namespace eight_queens_problem_l209_209063

def chessboard := Fin 8 → Option (Fin 8)
def safe_queen_placement (placement : chessboard) := 
  ∀ (i j : Fin 8), 
    i ≠ j → 
    placement i ≠ placement j ∧ 
    placement i ≠ placement j ± (i - j) ∧ 
    placement i ≠ placement j ± (j - i)

theorem eight_queens_problem : ∃ placement : chessboard, safe_queen_placement placement :=
sorry

end eight_queens_problem_l209_209063


namespace projection_of_vector_on_line_l209_209593

open Real EuclideanSpace

theorem projection_of_vector_on_line :
  let v := ⟨3, -3, -2⟩ : ℝ^3
  let d := ⟨1, -4, 2⟩ : ℝ^3
  let proj := (⟨11 / 21, -44 / 21, 22 / 21⟩ : ℝ^3)
  (v•d / d•d) • d = proj :=
by
  let v : ℝ^3 := ⟨3, -3, -2⟩
  let d : ℝ^3 := ⟨1, -4, 2⟩
  let proj : ℝ^3 := ⟨11 / 21, -44 / 21, 22 / 21⟩
  sorry

end projection_of_vector_on_line_l209_209593


namespace percentage_increase_pizza_l209_209398

-- Definitions and conditions
def radius1 : ℝ := 6
def radius2 : ℝ := 4
def area (r : ℝ) : ℝ := Real.pi * r^2
def percentage_increase (A1 A2 : ℝ) : ℝ :=
  ((A1 - A2) / A2) * 100

-- The target theorem (proof not included) proving the integer closest to N
theorem percentage_increase_pizza :
  Int.round (percentage_increase (area radius1) (area radius2)) = 125 :=
by
  sorry

end percentage_increase_pizza_l209_209398


namespace count_four_digit_numbers_less_than_1239_with_distinct_digits_l209_209583

theorem count_four_digit_numbers_less_than_1239_with_distinct_digits : 
  (number_of_valid_numbers : Nat) (four_digit_numbers : Finset Nat): 
  (∀ n ∈ four_digit_numbers, (1000 ≤ n ∧ n < 1239) ∧ (∀ i j : Nat, i ≠ j → Nat.digits n i ≠ Nat.digits n j)) ∧ 
  number_of_valid_numbers = four_digit_numbers.card :=
begin
  let four_digit_numbers := (Finset.range 2239).filter (λ n, 1000 ≤ n ∧ distinct (nat.digits 10 n)),
  exact sorry,
end

end count_four_digit_numbers_less_than_1239_with_distinct_digits_l209_209583


namespace is_even_function_with_period_pi_over_2_l209_209046

def f1 (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x)
def f2 (x : ℝ) : ℝ := sin (2 * x) * cos (2 * x)
def f3 (x : ℝ) : ℝ := sin (x) ^ 2 + cos (2 * x)
def f4 (x : ℝ) : ℝ := sin (2 * x) ^ 2 - cos (2 * x) ^ 2

theorem is_even_function_with_period_pi_over_2 : 
  ∀ x : ℝ, f4 x = f4 (-x) ∧ (f4 (x + π / 2) = f4 x) := 
sorry

end is_even_function_with_period_pi_over_2_l209_209046


namespace total_amount_paid_l209_209720

def cost_of_nikes : ℝ := 150
def cost_of_work_boots : ℝ := 120
def tax_rate : ℝ := 0.1

theorem total_amount_paid :
  let total_cost := cost_of_nikes + cost_of_work_boots in
  let tax := total_cost * tax_rate in
  let total_paid := total_cost + tax in
  total_paid = 297 := by
  let total_cost := cost_of_nikes + cost_of_work_boots
  let tax := total_cost * tax_rate
  let total_paid := total_cost + tax
  sorry

end total_amount_paid_l209_209720


namespace acceptable_colorings_correct_l209_209115

def acceptableColorings (n : ℕ) : ℕ :=
  (3^(n + 1) + (-1:ℤ)^(n + 1)).natAbs / 2

theorem acceptable_colorings_correct (n : ℕ) :
  acceptableColorings n = (3^(n + 1) + (-1:ℤ)^(n + 1)).natAbs / 2 :=
by
  sorry

end acceptable_colorings_correct_l209_209115


namespace adjacent_zero_point_range_l209_209619

def f (x : ℝ) : ℝ := x - 1
def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a*x - a + 3

theorem adjacent_zero_point_range (a : ℝ) :
  (∀ β, (∃ x, g x a = 0) → (|1 - β| ≤ 1 → (∃ x, f x = 0 → |x - β| ≤ 1))) →
  (2 ≤ a ∧ a ≤ 7 / 3) :=
sorry

end adjacent_zero_point_range_l209_209619


namespace sufficient_condition_l209_209266

theorem sufficient_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + a = 0 → a < 1) ↔ 
  (∀ c : ℝ, x^2 - 2 * x + c = 0 ↔ 4 - 4 * c ≥ 0 ∧ c < 1 → ¬ (∀ d : ℝ, d ≤ 1 → d < 1)) := 
by 
sorry

end sufficient_condition_l209_209266


namespace odd_factors_of_360_l209_209187

theorem odd_factors_of_360 : ∃ n, n = 6 ∧ 
  ∀ (x : ℕ), (∃ (a b : ℕ), 360 = 2^a * 3^b * x ∧ ¬ even x ∧ 0 < x) → (∀ (m : ℕ), m ∣ x ↔ (m = 1 ∨ m = 3 ∨ m = 5 ∨ m = 9 ∨ m = 15 ∨ m = 45)) :=
by
  -- Proof
  sorry

end odd_factors_of_360_l209_209187


namespace part_a_part_b_l209_209729

variables {A B C D E F P Q : Point}

-- Condition declarations
def is_parallelogram (A B C D : Point) : Prop := sorry -- Define this as necessary
def midpoint (M X Y : Point) : Prop := sorry -- Define midpoint of X and Y is M
def on_diagonal (P Q B D : Point) : Prop := sorry -- Define P, Q are on diagonal BD such that BP = PQ = QD

-- First part:
theorem part_a (h_parallelogram : is_parallelogram A B C D)
              (h_on_diagonal : on_diagonal P Q B D)
              (h_intersections : (line A P).intersect (line B C) = E 
                              ∧ (line A Q).intersect (line D C) = F) :
  midpoint E B C ∧ midpoint F D C :=
sorry

-- Second part:
theorem part_b (h_midpoint_E : midpoint E B C)
              (h_midpoint_F : midpoint F D C)
              (h_on_diagonal : on_diagonal P Q B D)
              (h_intersections : (line A P).intersect (line B C) = E 
                              ∧ (line A Q).intersect (line D C) = F) :
  is_parallelogram A B C D :=
sorry

end part_a_part_b_l209_209729


namespace fraction_of_total_calls_l209_209855

-- Definitions based on conditions
variable (B : ℚ) -- Calls processed by each member of Team B
variable (N : ℚ) -- Number of members in Team B

-- The fraction of calls processed by each member of Team A
def team_A_call_fraction : ℚ := 1 / 5

-- The fraction of calls processed by each member of Team C
def team_C_call_fraction : ℚ := 7 / 8

-- The fraction of agents in Team A relative to Team B
def team_A_agents_fraction : ℚ := 5 / 8

-- The fraction of agents in Team C relative to Team B
def team_C_agents_fraction : ℚ := 3 / 4

-- Total calls processed by Team A, Team B, and Team C
def total_calls_team_A : ℚ := (B * team_A_call_fraction) * (N * team_A_agents_fraction)
def total_calls_team_B : ℚ := B * N
def total_calls_team_C : ℚ := (B * team_C_call_fraction) * (N * team_C_agents_fraction)

-- Sum of total calls processed by all teams
def total_calls_all_teams : ℚ := total_calls_team_A B N + total_calls_team_B B N + total_calls_team_C B N

-- Potential total calls if all teams were as efficient as Team B
def potential_total_calls : ℚ := 3 * (B * N)

-- Fraction of total calls processed by all teams combined
def processed_fraction : ℚ := total_calls_all_teams B N / potential_total_calls B N

theorem fraction_of_total_calls : processed_fraction B N = 19 / 32 :=
by
  sorry -- Proof omitted

end fraction_of_total_calls_l209_209855


namespace chloe_minimum_dimes_l209_209896

variables (n : ℕ)

def chloe_has_enough_money (n : ℕ) : Prop :=
  let total_money := 40 + 2.5 + 3 + 0.10 * n
  in total_money ≥ 45.50

theorem chloe_minimum_dimes : ∃ n : ℕ, chloe_has_enough_money n ∧ n = 0 :=
  by {
    use 0,
    unfold chloe_has_enough_money,
    norm_num,
    sorry
  }

end chloe_minimum_dimes_l209_209896


namespace range_of_k_l209_209152

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 4 * x else x * Real.log x

def g (x k : ℝ) : ℝ := k * x - 1

theorem range_of_k (k : ℝ) :
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ (-2:ℝ, 2) ∧ b ∈ (-2:ℝ, 2) ∧ c ∈ (-2:ℝ, 2) ∧
   f a - g a k = 0 ∧ f b - g b k = 0 ∧ f c - g c k = 0) ↔
  (1 < k ∧ k < Real.log (2 * Real.sqrt Real.exp 1) ∨ 3 / 2 < k ∧ k < 2) :=
sorry

end range_of_k_l209_209152


namespace optimalPaths_count_l209_209508

-- Definition of the vertices of the cube and paths
inductive Vertex
| A | B | C | D | A₁ | B₁ | C₁ | D₁

open Vertex

-- Defining the conditions
def isOptimalPath (path : List Vertex) : Prop :=
  path.head = A ∧ path.last = A ∧ path.contains C₁ ∧
  (path.nodup ∧ path.length = 7) -- 7 vertices mean 6 edges since it starts and ends at A

-- The main theorem statement asserting that the number of optimal paths is 18
theorem optimalPaths_count : (List Vertex).count (paths : List (List Vertex)) (isOptimalPath paths) = 18 :=
sorry -- Proof is omitted as per instructions

end optimalPaths_count_l209_209508


namespace train_speed_72_kmph_l209_209519

theorem train_speed_72_kmph
  (L V : ℝ) -- L: Length of the train in meters, V: Speed of the train in m/s
  (h1 : L = V * 15) -- Condition when train crosses the man.
  (h2 : L + 300 = V * 30) -- Condition when train crosses the platform.
  : V * 3.6 = 72 := -- Speed of the train in km/h.
begin
  sorry
end

end train_speed_72_kmph_l209_209519


namespace reflection_matrix_values_l209_209908

noncomputable def matrix_reflection (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![[a, b], [-3/5, 4/5]]

noncomputable def identity_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1, 0], ![0, 1]]

theorem reflection_matrix_values (a b : ℚ) :
  matrix_reflection a b ⬝ matrix_reflection a b = identity_matrix ↔ a = -4/5 ∧ b = 3/5 := 
by
  sorry

end reflection_matrix_values_l209_209908


namespace ratio_of_volumes_of_spheres_l209_209684

theorem ratio_of_volumes_of_spheres (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a / b = 1 / 2 ∧ b / c = 2 / 3) : a^3 / b^3 = 1 / 8 ∧ b^3 / c^3 = 8 / 27 :=
by
  sorry

end ratio_of_volumes_of_spheres_l209_209684


namespace part1_part2_part3_l209_209155

-- Part (1):
theorem part1 (a : ℝ) : (∀ x : ℝ, x = 0 → f(x) = e^x - 1 - a * (Real.sin x) → f'(x) = -1) → a = 2 :=
by
  sorry

-- Part (2):
theorem part2 : (∀ x ∈ Set.Icc (0 : ℝ) π, f(x) = e^x - 1 - 2 * (Real.sin x) → 
   f(x) ≤ e^π - 1) :=
by
  sorry

-- Part (3):
theorem part3 (a : ℝ) : (∀ x ∈ Set.Icc (0 : ℝ) π, f(x) = e^x - 1 - a * (Real.sin x) → 
   f(x) ≥ 0) → a ∈ Set.Iic (1 : ℝ) :=
by
  sorry

end part1_part2_part3_l209_209155


namespace elena_hike_total_miles_l209_209725

theorem elena_hike_total_miles (x1 x2 x3 x4 x5 : ℕ)
  (h1 : x1 + x2 = 36)
  (h2 : x2 + x3 = 40)
  (h3 : x3 + x4 + x5 = 45)
  (h4 : x1 + x4 = 38) : 
  x1 + x2 + x3 + x4 + x5 = 81 := 
sorry

end elena_hike_total_miles_l209_209725


namespace tangent_line_at_2_l209_209793

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem tangent_line_at_2 :
  ∀ (x y : ℝ),
    (y = f 2 + (f' 2) * (x - 2)) → (x + 4 * y - 4 = 0) :=
by sorry

def f' (x : ℝ) : ℝ := -1 / x^2

example : f' 2 = -1 / 4 := by
  rw [f']
  norm_num

end tangent_line_at_2_l209_209793


namespace count_repeating_decimals_l209_209110

def is_repeating_decimal (a b : ℕ) :=
  let d := Nat.factorization b
  d ≠ 0 ∧ (¬ d.find (λ p => p = 2) = some 0) ∧ (¬ d.find (λ p => p = 5) = some 0)

theorem count_repeating_decimals :
  let nums := (List.range 15).map (λ n => n + 1)
  let count := List.countp (λ n => is_repeating_decimal n 18) nums 
  count = 10 :=
by
  sorry

end count_repeating_decimals_l209_209110


namespace percentage_reduction_in_price_l209_209029

-- Definitions based on conditions
def original_price (P : ℝ) (X : ℝ) := P * X
def reduced_price (R : ℝ) (X : ℝ) := R * (X + 5)

-- Theorem statement based on the problem to prove
theorem percentage_reduction_in_price
  (R : ℝ) (H1 : R = 55)
  (H2 : original_price P X = 1100)
  (H3 : reduced_price R X = 1100) :
  ((P - R) / P) * 100 = 25 :=
by
  sorry

end percentage_reduction_in_price_l209_209029


namespace shaded_area_to_circle_ratio_l209_209555

noncomputable def ab : ℝ := 9
noncomputable def ac : ℝ := 6
noncomputable def cb : ℝ := 3

-- Calculating radius and areas of semicircles
noncomputable def r_ab : ℝ := ab / 2
noncomputable def r_ac : ℝ := ac / 2
noncomputable def r_cb : ℝ := cb / 2

noncomputable def area_semicircle (r : ℝ) : ℝ := (↑(1/2) * real.pi * r^2)

noncomputable def area_ab : ℝ := area_semicircle r_ab
noncomputable def area_ac : ℝ := area_semicircle r_ac
noncomputable def area_cb : ℝ := area_semicircle r_cb

noncomputable def shaded_area : ℝ := area_ab - (area_ac + area_cb)

-- CD is the radius of the circle on AB (r_ab == 4.5)
noncomputable def cd : ℝ := r_ab

noncomputable def area_cd_circle : ℝ := real.pi * cd^2

noncomputable def ratio : ℝ := shaded_area / area_cd_circle

theorem shaded_area_to_circle_ratio :
  ratio = 2 / 9 :=
sorry

end shaded_area_to_circle_ratio_l209_209555


namespace convert_to_polar_coords_l209_209559

noncomputable def polar_coords (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x*x + y*y)
  let θ := if x = 0 then if y > 0 then Real.pi / 2 else if y < 0 then 3 * Real.pi / 2 else 0
           else Real.atan (y / x) + if x < 0 then Real.pi else 0
  (r, θ)

theorem convert_to_polar_coords : polar_coords (-2) (2 * Real.sqrt 3) = (4, 2 * Real.pi / 3) := 
  sorry

end convert_to_polar_coords_l209_209559


namespace no_two_consecutive_or_even_adjacent_l209_209417

def cube_faces := {1, 2, 3, 4, 5, 6}

def consecutive (a b : ℕ) : Prop :=
(a = 1 ∧ b = 6) ∨ (a = 6 ∧ b = 1) ∨ (a = b + 1) ∨ (b = a + 1)

def even (n : ℕ) : Prop := n % 2 = 0

def adjacent (f1 f2 : ℕ) : Prop := -- Assuming some definition for adjacent faces

noncomputable def arrangements (cube : ℕ → ℕ) : Prop :=
(∀ i ∈ cube_faces, cube i ≠ cube (i + 1)) ∧ -- No two consecutive numbers adjacent
(∀ (i : ℕ), even (cube i → ∀ (i' : ℕ), adjacent i i' → ¬even (cube i'))) -- No adjacent even numbers

theorem no_two_consecutive_or_even_adjacent :
  let p : ℚ := 1/30 in ∃ m n : ℕ, m.gcd n = 1 ∧ m / (m + n) = p ∧ (m + n) = 31 :=
by {
   sorry
}

end no_two_consecutive_or_even_adjacent_l209_209417


namespace distinct_values_exists_l209_209935

def P (x : ℝ) : ℂ := 2 + complex.exp (complex.I * x) - 2 * complex.exp (2 * complex.I * x) + complex.exp (3 * complex.I * x)

theorem distinct_values_exists (h : ∀ x, P x = 0 → 0 ≤ x ∧ x < 2 * real.pi) :
  (∃! x : ℝ, 0 ≤ x ∧ x < 2 * real.pi ∧ P x = 0) :=
sorry

end distinct_values_exists_l209_209935


namespace fewest_four_dollar_frisbees_l209_209844

theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : x + y = 64) (h2 : 3 * x + 4 * y = 196) : y = 4 :=
by
  sorry

end fewest_four_dollar_frisbees_l209_209844


namespace trig_inequalities_l209_209425

theorem trig_inequalities :
  (cos 8.5) < (sin 3) ∧ (sin 3) < (sin 1.5) := 
  by
    sorry

end trig_inequalities_l209_209425


namespace diagonals_in_polygon_l209_209070

-- Define the number of sides of the polygon
def n : ℕ := 30

-- Define the formula for the total number of diagonals in an n-sided polygon
def total_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Define the number of excluded diagonals for being parallel to one given side
def excluded_diagonals : ℕ := 1

-- Define the final count of valid diagonals after exclusion
def valid_diagonals : ℕ := total_diagonals n - excluded_diagonals

-- State the theorem to prove
theorem diagonals_in_polygon : valid_diagonals = 404 := by
  sorry


end diagonals_in_polygon_l209_209070


namespace poly_div_remainder_l209_209925

-- Define the polynomials and their conditions
def poly_div (f g : ℝ[X]) : (ℝ[X] × ℝ[X]) := f.div_mod g

def f := 2 * (X^6 : ℝ[X]) - (X^4 : ℝ[X]) + 4 * (X^2 : ℝ[X]) - 7
def g := (X^2 : ℝ[X]) + 4 * X + 3
def remainder := -704 * X - 706

-- The theorem to prove
theorem poly_div_remainder : (poly_div f g).2 = remainder := sorry

end poly_div_remainder_l209_209925


namespace trig_expression_value_l209_209085

theorem trig_expression_value : 
  (2 * (Real.sin (25 * Real.pi / 180))^2 - 1) / 
  (Real.sin (20 * Real.pi / 180) * Real.cos (20 * Real.pi / 180)) = -2 := 
by
  -- Proof goes here
  sorry

end trig_expression_value_l209_209085


namespace trajectory_of_circle_center_is_ellipse_l209_209613

theorem trajectory_of_circle_center_is_ellipse 
    (a b : ℝ) (θ : ℝ) 
    (h1 : a ≠ b)
    (h2 : 0 < a)
    (h3 : 0 < b)
    : ∃ (x y : ℝ), 
    (x, y) = (a * Real.cos θ, b * Real.sin θ) ∧ 
    (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end trajectory_of_circle_center_is_ellipse_l209_209613


namespace expand_product_l209_209092

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5 * x - 36 :=
by
  sorry

end expand_product_l209_209092


namespace math_problem_l209_209579

noncomputable def is_solution (x : ℝ) : Prop :=
  1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12

theorem math_problem :
  (is_solution ((7 + Real.sqrt 153) / 2)) ∧ (is_solution ((7 - Real.sqrt 153) / 2)) := 
by
  sorry

end math_problem_l209_209579


namespace number_of_odd_factors_of_360_l209_209195

def is_odd_factor (n : ℕ) (d : ℕ) : Prop := d ∣ n ∧ ∀ k, 2 ∣ k → ¬ (k ∣ d)

theorem number_of_odd_factors_of_360 : 
  {d : ℕ // is_odd_factor 360 d}.card = 6 := 
sorry

end number_of_odd_factors_of_360_l209_209195


namespace f_decreasing_f_odd_f_range_l209_209612

section

variable {f : ℝ → ℝ}

-- Condition: For any a, b ∈ ℝ, f(a+b) = f(a) + f(b)
axiom additivity (a b : ℝ) : f(a + b) = f(a) + f(b)

-- Condition: When x > 0, f(x) < 0
axiom positivity (x : ℝ) (h : x > 0) : f(x) < 0

-- Question 1: Prove f is decreasing on ℝ
theorem f_decreasing (x₁ x₂ : ℝ) (h : x₁ > x₂) : f(x₁) < f(x₂) := sorry

-- Question 2: Prove f is odd
theorem f_odd (x : ℝ) : f(-x) = -f(x) := sorry

-- Question 3: Given f(x^2 - 2) + f(x) < 0, find the range of x.
theorem f_range (x : ℝ) (h : f(x^2 - 2) + f(x) < 0) : x ∈ Set.Ioo (-∞) (-2) ∪ Set.Ioo 1 ∞ := sorry

end

end f_decreasing_f_odd_f_range_l209_209612


namespace cube_edge_probability_l209_209897

-- Definitions of cube vertices and edges
def V : Type := fin 8  -- 8 vertices of a cube
def E (v1 v2 : V) : Prop := -- Predicate that holds if v1 and v2 are endpoints of an edge
  (v1 ≠ v2) ∧ (
     (v1 = 0 ∧ v2 = 1) ∨
     (v1 = 2 ∧ v2 = 3) ∨
     -- add all 12 edge connections considering uniqueness
     true -- simplified to meet the condition of having 12 unique edges
  )

-- Theorem stating the probability that two randomly selected vertices form an edge.
theorem cube_edge_probability : 
  probability (∃ (v1 v2 : V), E v1 v2) = 3 / 7 :=
by
  sorry

end cube_edge_probability_l209_209897


namespace relation_between_x_and_y_l209_209675

variable (t : ℝ)
variable (x : ℝ := t ^ (2 / (t - 1))) (y : ℝ := t ^ ((t + 1) / (t - 1)))

theorem relation_between_x_and_y (h1 : t > 0) (h2 : t ≠ 1) : y ^ (1 / x) = x ^ y :=
by sorry

end relation_between_x_and_y_l209_209675


namespace ratio_of_distances_equal_l209_209434

theorem ratio_of_distances_equal
  {O₁ O₂ O₃ C : Point}
  {A₁ A₂ A₃ : Point}
  (h₁ : onCircleThrough O₁ C O₂ A₁)
  (h₂ : onCircleThrough O₂ C O₁ A₂)
  (h₃ : onCircleThrough O₃ C O₁ A₃)
  (line_through_point : ∃ l : Line, onLine l C ∧ intersects l A₁ ∧ intersects l A₂ ∧ intersects l A₃) :
  dist A₁ A₂ / dist A₂ A₃ = dist O₁ O₂ / dist O₂ O₃ :=
by
  sorry

end ratio_of_distances_equal_l209_209434


namespace num_repeating_decimals_l209_209112

theorem num_repeating_decimals : 
  let N := 10 in
  let integers_in_range := (1:ℤ) \u003c = n \u0026 \u0026 n \u003c = (15:ℤ) in
  ∀ n : ℤ, 
  (integers_in_range n) → 
  (nat.gcd (nat.abs n) (nat.abs 18) = 1 →
  (∃ (count : ℕ), 
    count = 15 - 5 ∧ count = N)) :=
sorry

end num_repeating_decimals_l209_209112


namespace doughnut_price_l209_209666

theorem doughnut_price
  (K C B : ℕ)
  (h1: K = 4 * C + 5)
  (h2: K = 5 * C - 6)
  (h3: K = 2 * C + 3 * B) :
  B = 9 := 
sorry

end doughnut_price_l209_209666


namespace log_2_16_is_4_log_5_25_is_not_5_log_3_81_is_4_l209_209876

theorem log_2_16_is_4 : log 2 16 = 4 :=
by 
  -- Add proof here 
  sorry

theorem log_5_25_is_not_5 : log 5 25 ≠ 5 :=
by 
  -- Add proof here 
  sorry

theorem log_3_81_is_4 : log 3 81 = 4 :=
by 
  -- Add proof here 
  sorry

end log_2_16_is_4_log_5_25_is_not_5_log_3_81_is_4_l209_209876


namespace trig_simplification_l209_209386

theorem trig_simplification (α : ℝ) :
  (cos (α - real.pi) / sin (real.pi - α)) * sin (α - real.pi / 2) * cos (3 * real.pi / 2 - α)
  = - cos α ^ 2 :=
by
  -- We use trigonometric identities to rewrite the theorem:
  -- cos(α - π) = - cos(α)
  -- sin(π - α) = sin(α)
  -- sin(α - π/2) = - cos(α)
  -- cos(3π/2 - α) = - sin(α)
  sorry

end trig_simplification_l209_209386


namespace power_function_value_l209_209146

theorem power_function_value (a : ℝ) (f : ℝ → ℝ) (h₁ : f = λ x, x^a) (h₂ : f 3 = 1/9) : f 2 = 1/4 :=
by
  sorry

end power_function_value_l209_209146


namespace sin_double_angle_l209_209631

theorem sin_double_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : tan α = 3 / 4) : sin (2 * α) = 24 / 25 := by
  sorry

end sin_double_angle_l209_209631


namespace number_of_subsets_of_setA_eq_2_l209_209162

noncomputable def setA (a : ℝ) : Set ℝ := {x | x^2 = a}

theorem number_of_subsets_of_setA_eq_2 (a : ℝ) (h : Finset.card {x : ℝ | x^2 = a} = 1) : 
  a = 0 := 
begin
  -- Proof goes here
  sorry
end

end number_of_subsets_of_setA_eq_2_l209_209162


namespace sebastian_orchestra_trombone_count_l209_209378

def number_of_trombone_players (orchestra_size : ℕ)
  (drummer : ℕ) (trumpet_players : ℕ) (french_horn_players : ℕ)
  (violinists : ℕ) (cellists : ℕ) (contrabassists : ℕ)
  (clarinet_players : ℕ) (flute_players : ℕ) (maestro : ℕ) : ℕ :=
  orchestra_size - (drummer + trumpet_players + french_horn_players + violinists +
                   cellists + contrabassists + clarinet_players + flute_players + maestro)

theorem sebastian_orchestra_trombone_count :
  number_of_trombone_players 21 1 2 1 3 1 1 3 4 1 = 4 :=
by
  have cond := number_of_trombone_players 21 1 2 1 3 1 1 3 4 1
  show cond = 4
  sorry

end sebastian_orchestra_trombone_count_l209_209378


namespace numeral_150th_decimal_place_l209_209837

theorem numeral_150th_decimal_place (n : ℕ) (h : n = 150) : 
  let decimal_representation := "63".cycle;
  let index := n % 2 in
  decimal_representation.get index = '3' :=
by
  sorry

end numeral_150th_decimal_place_l209_209837


namespace difference_of_squares_l209_209814

theorem difference_of_squares (a b : ℕ) (h₁ : a + b = 60) (h₂ : a - b = 14) : a^2 - b^2 = 840 := by
  sorry

end difference_of_squares_l209_209814


namespace palindrome_count_200_to_800_l209_209169

theorem palindrome_count_200_to_800 : 
  let valid_palindrome (n : ℕ) := 
    let d3 := n / 100 in
    let d2 := (n / 10) % 10 in
    let d1 := n % 10 in
    (200 <= n) ∧ (n <= 800) ∧ (d3 = d1) ∧ (d3 ≥ 2) ∧ (d3 ≤ 7)

  (∃ n, valid_palindrome n) → 
  ∃ count : ℕ, (count = 60) := 
by
  sorry

end palindrome_count_200_to_800_l209_209169


namespace number_of_solutions_l209_209114

-- Define the conditions
namespace MathProblem

def is_divisor (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

def satisfies_condition (m : ℕ) : Prop :=
  is_divisor 2520 (m^2 - 2)

-- Define the main theorem: Prove there are exactly 5 values
theorem number_of_solutions : 
  { m : ℕ // m > 0 ∧ satisfies_condition m }.to_finset.card = 5 :=
by 
  sorry

end MathProblem

end number_of_solutions_l209_209114


namespace hyperbola_eccentricity_l209_209127

def hyperbola_asymptotes (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x ∨ y = -(b / a) * x

theorem hyperbola_eccentricity :
  ∀ (a b c e : ℝ),
  (0 < a) →
  (0 < b) →
  b = (sqrt 2 / 2) * a →
  c = sqrt (a^2 + b^2) →
  e = c / a →
  e = sqrt 6 / 2 :=
by
  intros a b c e ha hb hb_eq hc he
  sorry

end hyperbola_eccentricity_l209_209127


namespace area_of_rectangle_l209_209280

theorem area_of_rectangle (a b : ℝ) : 
  ∃ x y, x ^ 2 + y ^ 2 = (2 * a + b) ^ 2 ∧ x * y = 2 * a * b :=
begin
  sorry
end

end area_of_rectangle_l209_209280


namespace earthquake_magnitude_amplitude_ratio_l209_209787

def richter_magnitude (A A_0 : ℝ) : ℝ := log A - log A_0

theorem earthquake_magnitude (A A_0 : ℝ) (hA : A = 1000) (hA0 : A_0 = 0.001) : richter_magnitude A A_0 = 6 :=
by
  rw [hA, hA0]
  -- We would finish the proof here using the properties of logarithms.
  sorry

theorem amplitude_ratio (M9 M5 : ℝ) (hM9 : M9 = 9) (hM5 : M5 = 5) (A_0 : ℝ) (hA0 : A_0 = 0.001) :
    let x := 10^(6)
    let y := 10^(2)
    x / y = 10000 :=
by
  rw [←hM9, ←hM5, hA0]
  -- We would finish the proof here using the properties of logarithms and the given values.
  sorry

end earthquake_magnitude_amplitude_ratio_l209_209787


namespace maximum_ratio_of_t_l209_209809

-- Definitions from the conditions
variable (r x : ℝ)
def t := x / r

-- Given the constraints as conditions
axiom cone_properties : r > 0 ∧ x > 0 

-- Statement of the proof problem
theorem maximum_ratio_of_t (h_cone: cone_properties r x) : 
  t = (7 - Real.sqrt 22) / 3 := 
sorry

end maximum_ratio_of_t_l209_209809


namespace prove_equivalence_l209_209108

variable (x : ℝ)

def operation1 (x : ℝ) : ℝ := 8 - x

def operation2 (x : ℝ) : ℝ := x - 8

theorem prove_equivalence : operation2 (operation1 14) = -14 := by
  sorry

end prove_equivalence_l209_209108


namespace faucets_fill_time_l209_209939

theorem faucets_fill_time (fill_time_4faucets_200gallons_12min : 4 * 12 * faucet_rate = 200) 
    (fill_time_m_50gallons_seconds : ∃ (rate: ℚ), 8 * t_to_seconds * rate = 50) : 
    8 * t_to_seconds / 33.33 = 90 :=
by sorry


end faucets_fill_time_l209_209939


namespace part1_part2_l209_209016

-- Definitions and conditions
def a : ℕ := 60
def b : ℕ := 40
def c : ℕ := 80
def d : ℕ := 20
def n : ℕ := a + b + c + d

-- Given critical value for 99% certainty
def critical_value_99 : ℝ := 6.635

-- Calculate K^2 using the given formula
noncomputable def K_squared : ℝ := (n * ((a * d - b * c) ^ 2)) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Calculation of probability of selecting 2 qualified products from 5 before renovation
def total_sampled : ℕ := 5
def qualified_before_renovation : ℕ := 3
def total_combinations (n k : ℕ) : ℕ := Nat.choose n k
def prob_selecting_2_qualified : ℚ := (total_combinations qualified_before_renovation 2 : ℚ) / 
                                      (total_combinations total_sampled 2 : ℚ)

-- Proof statements
theorem part1 : K_squared > critical_value_99 := by
  sorry

theorem part2 : prob_selecting_2_qualified = 3 / 10 := by
  sorry

end part1_part2_l209_209016


namespace Martha_finished_two_problems_l209_209880

-- Definitions for given conditions
variables {M : ℕ}
def Jenna_problems : ℕ := 4 * M - 2
def Mark_problems : ℕ := (4 * M - 2) / 2
def Angela_problems : ℕ := 9
def Total_problems : ℕ := 20

-- Statement of the proposition
theorem Martha_finished_two_problems
  (h : M + Jenna_problems + Mark_problems + Angela_problems = Total_problems) :
  M = 2 :=
by
  sorry

end Martha_finished_two_problems_l209_209880


namespace log_a_b_eq_pi_l209_209018

theorem log_a_b_eq_pi (a b : ℝ) (r C : ℝ)
  (h1 : r = 3 * Real.log10 a)
  (h2 : C = 6 * Real.log10 b)
  (h3 : C = 2 * Real.pi * r) : Real.log a b = Real.pi :=
by
  sorry

end log_a_b_eq_pi_l209_209018


namespace odd_factors_of_360_l209_209228

theorem odd_factors_of_360 : ∃ n, 360 = 2^3 * 3^2 * 5^1 ∧ n = 6 :=
by 
  -- Define the prime factorization of 360
  let pf360 := (2^3 * 3^2 * 5^1)
  
  -- Define the count of odd factors by removing the contribution of factor 2.
  let odd_part := (3^2 * 5^1)
  
  -- Calculate the number of factors.
  have odd_factors_count_eq : (2 + 1) * (1 + 1) = 6 := by 
    calc (2 + 1) * (1 + 1) = 3 * 2 : by rfl
                         ... = 6 : by rfl
  
  -- Assert the existence of such n matching the count of odd factors.
  exact ⟨6, And.intro rfl odd_factors_count_eq⟩

end odd_factors_of_360_l209_209228


namespace triangle_inequality_l209_209964

theorem triangle_inequality 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : A + B + C = π) 
  (h5 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  (h6 : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B)
  (h7 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) :
  3 / 2 ≤ a^2 / (b^2 + c^2) + b^2 / (c^2 + a^2) + c^2 / (a^2 + b^2) ∧
  (a^2 / (b^2 + c^2) + b^2 / (c^2 + a^2) + c^2 / (a^2 + b^2) ≤ 
     2 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2)) :=
sorry

end triangle_inequality_l209_209964


namespace time_elapsed_in_minutes_l209_209372

-- Definitions for conditions
def speed_riya := 24 -- in kmph
def speed_priya := 35 -- in kmph
def distance := 44.25 -- in km

-- Definition for the proof goal
theorem time_elapsed_in_minutes : (distance / (speed_riya + speed_priya) * 60) = 45 :=
by
  sorry

end time_elapsed_in_minutes_l209_209372


namespace range_of_b_l209_209974

noncomputable def f (b x : ℝ) : ℝ := Real.exp x * (x^2 - b*x)

def is_monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem range_of_b (b : ℝ) :
  (∃ I : Set ℝ, I ⊆ Set.Icc (1/2 : ℝ) 2 ∧ is_monotonic_increasing (f b) (Inf I) (Sup I)) →
  b < 8/3 :=
sorry

end range_of_b_l209_209974


namespace suitable_survey_method_l209_209427

-- Definitions of conditions
def high_precision_needed : Prop := true
def no_errors_allowed : Prop := true

-- Survey methods
inductive SurveyMethod
| comprehensive_survey
| sampling_survey

-- Theorem statement
theorem suitable_survey_method 
    (c1: high_precision_needed) 
    (c2: no_errors_allowed) : 
    SurveyMethod := 
SurveyMethod.comprehensive_survey

-- Include sorry to skip the proof
sorry

end suitable_survey_method_l209_209427


namespace new_students_correct_l209_209694

variable 
  (students_start_year : Nat)
  (students_left : Nat)
  (students_end_year : Nat)

def new_students (students_start_year students_left students_end_year : Nat) : Nat :=
  students_end_year - (students_start_year - students_left)

theorem new_students_correct :
  ∀ (students_start_year students_left students_end_year : Nat),
  students_start_year = 10 →
  students_left = 4 →
  students_end_year = 48 →
  new_students students_start_year students_left students_end_year = 42 :=
by
  intros students_start_year students_left students_end_year h1 h2 h3
  rw [h1, h2, h3]
  unfold new_students
  norm_num

end new_students_correct_l209_209694


namespace distance_B_to_plane_EFG_l209_209963

-- Definitions and setup based on the conditions of the problem
def point := (ℝ × ℝ × ℝ)

def A : point := (0, 0, 0)
def B : point := (4, 0, 0)
def C : point := (4, 4, 0)
def D : point := (0, 4, 0)

def E : point := ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)
def F : point := ((A.1 + D.1) / 2, (A.2 + D.2) / 2, (A.3 + D.3) / 2)

def G : point := (C.1, C.2, 2)

noncomputable def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

-- We are asked to prove that distance from point B to the plane EFG is sqrt(6)/3
theorem distance_B_to_plane_EFG : sorry :=
begin
  let B_to_plane_EFG := distance B (sorry), -- placeholder for the correct mathematical function for distance
  exact B_to_plane_EFG = real.sqrt(2) / real.sqrt(3), -- i.e., sqrt(6)/3
end

end distance_B_to_plane_EFG_l209_209963


namespace baker_cake_count_l209_209886

theorem baker_cake_count :
  let initial_cakes := 62
  let additional_cakes := 149
  let sold_cakes := 144
  initial_cakes + additional_cakes - sold_cakes = 67 :=
by
  sorry

end baker_cake_count_l209_209886


namespace product_divisible_by_condition_l209_209343

theorem product_divisible_by_condition 
  (m : List ℕ) (n : List ℕ)
  (h : ∀ d > 1, (m.count (λ x, x % d = 0)) ≥ (n.count (λ x, x % d = 0))) :
  (∏ i in m, i) % (∏ i in n, i) = 0 :=
by sorry

end product_divisible_by_condition_l209_209343


namespace odd_factors_360_l209_209184

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l209_209184


namespace SunshineOrchard_has_54_pumpkins_l209_209057

noncomputable def SunshineOrchard_pumpkin_count (MoonglowOrchard: ℕ) : ℕ :=
  3 * MoonglowOrchard + 12

theorem SunshineOrchard_has_54_pumpkins (Moonglow_pumpkins: ℕ) (h: Moonglow_pumpkins = 14) : 
  SunshineOrchard_pumpkin_count Moonglow_pumpkins = 54 := by
  rw [h]
  simp [SunshineOrchard_pumpkin_count]
  -- At this point we would proceed to show the calculation explicitly
  sorry

end SunshineOrchard_has_54_pumpkins_l209_209057


namespace logs_in_stack_l209_209515

def num_rows (start end : ℕ) : ℕ :=
  start - end + 1

def sum_arithmetic_series (n a l : ℕ) : ℕ :=
  (n * (a + l)) / 2

theorem logs_in_stack : 
  ∀ (start bottom_logs top_logs : ℕ), 
    start = 15 → bottom_logs = 15 → top_logs = 4 → 
    let n := num_rows bottom_logs top_logs in
    let total_logs := sum_arithmetic_series n bottom_logs top_logs in
    total_logs = 114 :=
by
  intros start bottom_logs top_logs h_start h_bottom_logs h_top_logs,
  subst h_start, subst h_bottom_logs, subst h_top_logs,
  let n := num_rows 15 4,
  have : n = 12 := rfl,
  let total_logs := sum_arithmetic_series n 15 4,
  have : total_logs = (12 * (15 + 4)) / 2 := rfl,
  have : total_logs = 114 := rfl,
  assumption

end logs_in_stack_l209_209515


namespace quadrilateral_area_proof_l209_209005

def area_of_quadrilateral (d h1 h2 : ℝ) : ℝ := (1/2) * d * (h1 + h2)

theorem quadrilateral_area_proof :
  area_of_quadrilateral 20 5 4 = 90 := by
  sorry

end quadrilateral_area_proof_l209_209005


namespace function_properties_l209_209650

theorem function_properties (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 25 ≠ 0 ∧ x^2 - (k - 6) * x + 16 ≠ 0) → 
  (-2 < k ∧ k < 10) :=
by
  intros h
  sorry

end function_properties_l209_209650


namespace population_of_seventh_village_l209_209806

def village_populations : List ℕ := [803, 900, 1100, 1023, 945, 980]

def average_population : ℕ := 1000

theorem population_of_seventh_village 
  (h1 : List.length village_populations = 6)
  (h2 : 1000 * 7 = 7000)
  (h3 : village_populations.sum = 5751) : 
  7000 - village_populations.sum = 1249 := 
by {
  -- h1 ensures there's exactly 6 villages in the list
  -- h2 calculates the total population of 7 villages assuming the average population
  -- h3 calculates the sum of populations in the given list of 6 villages
  -- our goal is to show that 7000 - village_populations.sum = 1249
  -- this will be simplified in the proof
  sorry
}

end population_of_seventh_village_l209_209806


namespace translated_sine_symmetric_min_m_l209_209799

noncomputable def smallest_m : ℝ :=
  let m := π / 6
  in m

theorem translated_sine_symmetric_min_m (m : ℝ) (h₁ : m > 0) :
  (∃ k : ℤ, m + π / 3 = k * π + π / 2) ↔ m = smallest_m :=
by
  sorry

end translated_sine_symmetric_min_m_l209_209799


namespace find_a_l209_209139

open Real

noncomputable def valid_solutions (a b : ℝ) : Prop :=
  a + 2 / b = 17 ∧ b + 2 / a = 1 / 3

theorem find_a (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : valid_solutions a b) :
  a = 6 ∨ a = 17 :=
by sorry

end find_a_l209_209139


namespace identify_non_convex_function_l209_209149

noncomputable def f1 (x : Real) : Real := Real.sin x + Real.cos x
noncomputable def f2 (x : Real) : Real := -x * Real.exp (-x)
noncomputable def f3 (x : Real) : Real := -(x^3) + 2*x - 1
noncomputable def f4 (x : Real) : Real := Real.log x - 2*x

def is_convex (f : Real → Real) (interval : Set Real) : Prop :=
  ∀ x ∈ interval, ∀ y ∈ interval, ∀ λ ∈ Set.Icc (0 : Real) 1, 
    f (λ*x + (1 - λ)*y) ≤ λ*f x + (1 - λ)*y

theorem identify_non_convex_function : 
  ¬ is_convex f2 { x | 0 < x ∧ x < Real.pi / 2 } ∧ 
  is_convex f1 { x | 0 < x ∧ x < Real.pi / 2 } ∧ 
  is_convex f3 { x | 0 < x ∧ x < Real.pi / 2 } ∧ 
  is_convex f4 { x | 0 < x ∧ x < Real.pi / 2 } := 
sorry

end identify_non_convex_function_l209_209149


namespace train_length_l209_209872

def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (5 / 18)

theorem train_length :
  let train_speed := 72.99376049916008
  let man_speed := 5
  let time_to_cross := 6
  let relative_speed_kmph := train_speed + man_speed
  let relative_speed_mps := kmph_to_mps relative_speed_kmph
  (relative_speed_mps * time_to_cross) = 129.98626749860016 :=
by
  let train_speed := 72.99376049916008
  let man_speed := 5
  let time_to_cross := 6
  let relative_speed_kmph := train_speed + man_speed
  let relative_speed_mps := kmph_to_mps relative_speed_kmph
  have relative_speed_mps_correct : relative_speed_mps = 21.66437791643336 := sorry
  sorry

end train_length_l209_209872


namespace infinite_unlucky_numbers_l209_209032

def is_unlucky (n : ℕ) : Prop :=
  ¬(∃ x y : ℕ, x > 1 ∧ y > 1 ∧ (n = x^2 - 1 ∨ n = y^2 - 1))

theorem infinite_unlucky_numbers : ∀ᶠ n in at_top, is_unlucky n := sorry

end infinite_unlucky_numbers_l209_209032


namespace grapefruits_orchards_proof_l209_209022

/-- 
Given the following conditions:
1. There are 40 orchards in total.
2. 15 orchards are dedicated to lemons.
3. The number of orchards for oranges is two-thirds of the number of orchards for lemons.
4. Limes and grapefruits have an equal number of orchards.
5. Mandarins have half as many orchards as limes or grapefruits.
Prove that the number of citrus orchards growing grapefruits is 6.
-/
def num_grapefruit_orchards (TotalOrchards Lemons Oranges L G M : ℕ) : Prop :=
  TotalOrchards = 40 ∧
  Lemons = 15 ∧
  Oranges = 2 * Lemons / 3 ∧
  L = G ∧
  M = G / 2 ∧
  L + G + M = TotalOrchards - (Lemons + Oranges) ∧
  G = 6

theorem grapefruits_orchards_proof : ∃ (TotalOrchards Lemons Oranges L G M : ℕ), num_grapefruit_orchards TotalOrchards Lemons Oranges L G M :=
by
  sorry

end grapefruits_orchards_proof_l209_209022


namespace part1_part1_general_part2_l209_209010

-- Define the sequence {a_n} with initial conditions and recurrence relation
def a : ℕ → ℤ
| 0       := 1
| 1       := 3
| (n + 2) := 3 * a (n + 1) - 2 * a n

-- Define the sequence {b_n} in terms of {a_n}
def b (n : ℕ) := 2 * Int.log2 (a n + 1) - 1

-- Define the sequence {c_n} in terms of {a_n} and {b_n}
def c (n : ℕ) := ((a n + 1) * (3 - 2 * n)) / (b n * b (n + 1))

-- Define the sum of the first n terms of {c_n}
def S (n : ℕ) := (Finset.range n).sum (λ k, c k)

-- The first part of the proof problem
theorem part1 : ∀ n, a (n + 1) - a n = 2^n :=
by sorry

-- The general term of the sequence {a_n}
theorem part1_general : ∀ n, a n = 2^n - 1 :=
by sorry

-- The second part of the proof problem
theorem part2 : ∀ n, S n = 2 - (2^n * 2) / (2 * n + 1) :=
by sorry

end part1_part1_general_part2_l209_209010


namespace sequence_exists_l209_209315

theorem sequence_exists (n : ℕ) (h_n : 2 ≤ n) (ℓ : fin n → fin n → ℝ)
  (ℓ_nonneg : ∀ i j : fin n, i < j → 0 ≤ ℓ i j) :
  ∃ (a : fin n → ℝ), 
    (∀ i j : fin n, i < j → |a i - a j| ≥ ℓ i j) ∧
    (∑ i : fin n, a i ≤ ∑ i j : fin n, if i < j then ℓ i j else 0) := 
sorry

end sequence_exists_l209_209315


namespace q_simplification_l209_209342

noncomputable def q (x a b c D : ℝ) : ℝ :=
  (x + a)^2 / ((a - b) * (a - c)) + 
  (x + b)^2 / ((b - a) * (b - c)) + 
  (x + c)^2 / ((c - a) * (c - b))

theorem q_simplification (a b c D x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  q x a b c D = a + b + c + 2 * x + 3 * D / (a + b + c) :=
by
  sorry

end q_simplification_l209_209342


namespace line_equation_min_ratio_l209_209160

-- Define the parabola and points
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Assuming a point F at (1, 0)
def F : ℝ × ℝ := (1, 0)

-- Define line l passing through F
structure Line (m : ℝ) :=
(x y : ℝ)
(eqn : x = m * y + 1)

-- Definitions of coordinates and conditions
def A (x y : ℝ) : Prop := parabola x y ∧ Line (-1) x y
def B (x y : ℝ) : Prop := parabola x y ∧ Line (-1) x y
def C (y : ℝ) : Prop := y = -1 / 1

-- Statement (1): Find equation of line l
theorem line_equation : ∃ m, (- 4 * m = 4) ∧ (∀ l : Line m, l.eqn = (λ x y, x + y = 1)) :=
sorry

-- Definitions of segments and areas
def NDC_area (x y : ℝ) : ℝ := 
1 / 2 * abs ((2 * -1 ^ 3 + 3 * -1 + 1) * (2 * (-1)^2 + 1))
def FDM_area (x : ℝ) : ℝ := 
1 / 2 * abs ((2 * (-1)^2 + 2) * 2 * abs (-1))
def ratio (x : ℝ) : ℝ := 
NDC_area x / FDM_area x

-- Statement (2): Find minimum value of the ratio
theorem min_ratio : ∃ (m: ℝ), m^2 = 1/2 → (∀ r : ratio m, r ≥ 2) :=
sorry

end line_equation_min_ratio_l209_209160


namespace shaded_area_l209_209704

theorem shaded_area 
  (R r : ℝ) 
  (h_area_larger_circle : π * R ^ 2 = 100 * π) 
  (h_shaded_larger_fraction : 2 / 3 = (area_shaded_larger / (π * R ^ 2))) 
  (h_relationship_radius : r = R / 2) 
  (h_area_smaller_circle : π * r ^ 2 = 25 * π)
  (h_shaded_smaller_fraction : 1 / 3 = (area_shaded_smaller / (π * r ^ 2))) : 
  (area_shaded_larger + area_shaded_smaller = 75 * π) := 
sorry

end shaded_area_l209_209704


namespace total_population_l209_209752

variables (NY NE PA MD NJ : ℕ)
variable (population_NE : NE = 2100000)
variable (population_NY : NY = (2 * NE) / 3)
variable (population_PA : PA = (3 * NE) / 2)
variable (population_MD_NJ : MD + NJ = NE + NE / 5)

theorem total_population
  (total := NY + NE + PA + MD + NJ) :
  total = 9170000 :=
by
  rw [population_NE, population_NY, population_PA, population_MD_NJ]
  sorry

end total_population_l209_209752


namespace tunnel_length_is_correct_l209_209517

-- Define the conditions given in the problem
def length_of_train : ℕ := 90
def speed_of_train : ℕ := 160
def time_to_pass_tunnel : ℕ := 3

-- Define the length of the tunnel to be proven
def length_of_tunnel : ℕ := 480 - length_of_train

-- Define the statement to be proven
theorem tunnel_length_is_correct : length_of_tunnel = 390 := by
  sorry

end tunnel_length_is_correct_l209_209517


namespace count_ways_to_place_balls_l209_209765

def Ball : Type := {A, B, C, D}
def Box : Type := {box1, box2, box3}

variables (place : Ball → Box)
def at_least_one (b : Box) : Prop := ∃ a b : Ball, place a = b ∧ place b = b
def distinct_boxes (b : Box) : Prop := ∀ b1 b2 : Ball, b1 ≠ b2 → place b1 ≠ place b2
def separate_AB : Prop := place 'A' ≠ place 'B'

theorem count_ways_to_place_balls :
  (∀ b : Box, at_least_one b) ∧ (separate_AB) → 
  (∃ (n : ℕ), n = 30) := by
  sorry

end count_ways_to_place_balls_l209_209765


namespace josh_quadrilateral_rod_count_l209_209723

theorem josh_quadrilateral_rod_count :
  let rods := {n : ℕ | 1 ≤ n ∧ n ≤ 40} 
  let placed_rods := {5, 12, 20}
  let valid_rods := {d : ℕ | 
    d ∈ rods ∧ 
    d ≠ 5 ∧ d ≠ 12 ∧ d ≠ 20 ∧ 
    3 < d ∧ d < 37 
  }
  (valid_rods.card = 30) := 
by
  sorry

end josh_quadrilateral_rod_count_l209_209723


namespace conjugate_of_complex_number_l209_209118

theorem conjugate_of_complex_number : 
  ∃ z : ℂ, z = (2 + Complex.i)^2 ∧ Complex.conj z = 3 - 4 * Complex.i :=
by
  sorry

end conjugate_of_complex_number_l209_209118


namespace sqrt_sum_simplification_l209_209571

theorem sqrt_sum_simplification : ∀ (x y : ℝ), (sqrt 27 + sqrt 75 = 8 * sqrt 3) :=
by
  intros,
  sorry

end sqrt_sum_simplification_l209_209571


namespace john_total_payment_l209_209719

def cost_of_nikes : ℝ := 150
def cost_of_work_boots : ℝ := 120
def tax_rate : ℝ := 0.10

theorem john_total_payment :
  let total_cost_before_tax := cost_of_nikes + cost_of_work_boots in
  let tax := tax_rate * total_cost_before_tax in
  let total_payment := total_cost_before_tax + tax in
  total_payment = 297 :=
by
  sorry

end john_total_payment_l209_209719


namespace odd_factors_of_360_l209_209193

theorem odd_factors_of_360 : ∃ n, n = 6 ∧ 
  ∀ (x : ℕ), (∃ (a b : ℕ), 360 = 2^a * 3^b * x ∧ ¬ even x ∧ 0 < x) → (∀ (m : ℕ), m ∣ x ↔ (m = 1 ∨ m = 3 ∨ m = 5 ∨ m = 9 ∨ m = 15 ∨ m = 45)) :=
by
  -- Proof
  sorry

end odd_factors_of_360_l209_209193


namespace domain_of_sqrt_function_l209_209100

theorem domain_of_sqrt_function : 
  ∀ x, (7 + 6 * x - x^2) ≥ 0 ↔ -1 ≤ x ∧ x ≤ 7 :=
by {
  intro x,
  sorry
}

end domain_of_sqrt_function_l209_209100


namespace number_of_regions_l209_209712

-- Define the necessary conditions
def general_position (lines : List (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (l1 l2 : ℝ × ℝ × ℝ), l1 ≠ l2 → -- No two lines are the same
    let (a1, b1, c1) := l1 in
    let (a2, b2, c2) := l2 in
    (a1 * b2 ≠ a2 * b1) ∧ -- No two lines are parallel
    ∀ (l3 : ℝ × ℝ × ℝ), l3 ≠ l1 → l3 ≠ l2 → -- No three lines are concurrent
      let (a3, b3, c3) := l3 in
      (a1 * b2 * c3 + a2 * b3 * c1 + a3 * b1 * c2 ≠ 
       c1 * b2 * a3 + c2 * b3 * a1 + c3 * b1 * a2)

-- Formulate the theorem
theorem number_of_regions (lines : List (ℝ × ℝ × ℝ)) (n : ℕ) :
  length lines = n →
  general_position lines →
  number_of_regions lines = 1 + (n * (n + 1)) / 2 :=
by
  sorry

end number_of_regions_l209_209712


namespace v3_at_x2_l209_209061

open polynomial

-- Define the polynomial
def f (x : ℝ) : ℝ := x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

-- Define v_3 using Horner's method
def v_3 (x : ℝ) : ℝ :=
  let v₀ := 1 in
  let v₁ := x - 12 in
  let v₂ := v₁ * x + 60 in
  v₂ * x - 160

-- State the theorem
theorem v3_at_x2 : v_3 2 = -80 :=
by
  rw [v_3, f]
  sorry

end v3_at_x2_l209_209061


namespace sequence_a7_l209_209979

/-- 
  Given a sequence {a_n} such that a_1 + a_{2n-1} = 4n - 6, 
  prove that a_7 = 11 
-/
theorem sequence_a7 (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : a 7 = 11 :=
by
  sorry

end sequence_a7_l209_209979


namespace additional_workers_needed_l209_209475

theorem additional_workers_needed
  (total_days : ℕ) (initial_workers : ℕ) (days_elapsed : ℕ)
  (work_done : ℚ) : ℕ :=
  let remaining_work := 1 - work_done in
  let remaining_days := total_days - days_elapsed in
  let initial_work_rate := work_done / days_elapsed in
  let required_work_rate := remaining_work / remaining_days in
  let workforce_ratio := required_work_rate / initial_work_rate in
  let new_workforce := workforce_ratio * initial_workers in
  new_workforce.to_nat - initial_workers

-- Given conditions
example : additional_workers_needed 50 20 25 0.4 = 10 :=
by sorry

end additional_workers_needed_l209_209475


namespace arithmetic_sequence_seventh_term_l209_209812

theorem arithmetic_sequence_seventh_term (a d : ℚ) 
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 15)
  (h2 : a + 5 * d = 8) :
  a + 6 * d = 29 / 3 := 
sorry

end arithmetic_sequence_seventh_term_l209_209812


namespace ground_beef_sold_ratio_l209_209756

variable (beef_sold_Thursday : ℕ) (beef_sold_Saturday : ℕ) (avg_sold_per_day : ℕ) (days : ℕ)

theorem ground_beef_sold_ratio (h₁ : beef_sold_Thursday = 210)
                             (h₂ : beef_sold_Saturday = 150)
                             (h₃ : avg_sold_per_day = 260)
                             (h₄ : days = 3) :
  let total_sold := avg_sold_per_day * days
  let beef_sold_Friday := total_sold - beef_sold_Thursday - beef_sold_Saturday
  (beef_sold_Friday : ℕ) / (beef_sold_Thursday : ℕ) = 2 := by
  sorry

end ground_beef_sold_ratio_l209_209756


namespace determine_day_of_the_week_l209_209395

-- Define the given conditions
def leap_years (start_year: ℕ) (end_year: ℕ) : ℕ :=
  filter (λ year, (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)) (list.range' start_year (end_year - start_year + 1)).length

def regular_years (start_year: ℕ) (end_year: ℕ) : ℕ :=
  end_year - start_year + 1 - leap_years start_year end_year

def total_days_back (start_year: ℕ) (end_year: ℕ) : ℤ :=
  regular_years start_year end_year + 2 * leap_years start_year end_year

def day_of_the_week (days_back: ℤ) (start_day: ℕ) : ℕ :=
  (start_day + (7 - (days_back % 7))) % 7

-- Definition of the initial condition stating that the 100th anniversary of the event is on a Tuesday
def event_day (anniversary_year: ℕ) : ℕ :=
  -- Tuesday is represented as 2 (starting from Sunday as 0)
  2

-- The main theorem proving the original day of the week of the event
theorem determine_day_of_the_week :
  day_of_the_week
    (total_days_back 1912 2012)
    (event_day 2012) = 3 := 
-- We will prove that the start day is Wednesday, represented by 3 (Sunday=0)
sorry

end determine_day_of_the_week_l209_209395


namespace least_number_to_add_to_4499_is_1_l209_209009

theorem least_number_to_add_to_4499_is_1 (x : ℕ) : (4499 + x) % 9 = 0 → x = 1 := sorry

end least_number_to_add_to_4499_is_1_l209_209009


namespace horse_bags_problem_l209_209528

theorem horse_bags_problem (x y : ℤ) 
  (h1 : x - 1 = y + 1) : 
  x + 1 = 2 * (y - 1) :=
sorry

end horse_bags_problem_l209_209528


namespace find_a_min_value_of_f_l209_209158

theorem find_a (a : ℕ) (h1 : 3 / 2 < 2 + a) (h2 : 1 / 2 ≥ 2 - a) : a = 1 := by
  sorry

theorem min_value_of_f (a x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) : 
    (a = 1) → ∃ m : ℝ, m = 3 ∧ ∀ x : ℝ, |x + a| + |x - 2| ≥ m := by
  sorry

end find_a_min_value_of_f_l209_209158


namespace sin_expression_value_l209_209267

theorem sin_expression_value (α : ℝ) (h : Real.cos (α + π / 5) = 4 / 5) :
  Real.sin (2 * α + 9 * π / 10) = 7 / 25 :=
sorry

end sin_expression_value_l209_209267


namespace number_of_positive_area_triangles_l209_209246

def integer_points := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

def count_triangles_with_positive_area : Nat :=
  let total_points := 25 -- total number of integer points in the grid
  let total_combinations := Nat.choose total_points 3 -- total possible combinations
  let degenerate_cases := 136 -- total degenerate (collinear) cases
  total_combinations - degenerate_cases

theorem number_of_positive_area_triangles : count_triangles_with_positive_area = 2164 := by
  sorry

end number_of_positive_area_triangles_l209_209246


namespace number_of_valid_sets_l209_209328

theorem number_of_valid_sets (A : Set ℤ) (B : Set ℤ) (f : ℤ → ℤ) 
  (hB : B = {1, 2}) 
  (hf : ∀ x ∈ A, f x ∈ B)
  (hf_def : ∀ x, f x = x^2)
  (hA_nonempty : A.nonempty) :
  { S : Set ℤ // S ⊆ A }.filter (λ S, S.nonempty) |>.toFinset.card = 3 := by
  sorry

end number_of_valid_sets_l209_209328


namespace bounded_region_area_is_1800_l209_209798

open Real

def region := {p : ℝ × ℝ | let (x, y) := p in 
  (x >= 0 → y^2 + 2*x*y + 60*x = 900) ∧ 
  (x < 0 → y^2 + 2*x*y - 60*x = 900)}

noncomputable def area_bounded_region : ℝ :=
  let vertices := [(0, 30), (0, -30), (30, -30), (-30, 30)] in
  let base := 30 - 0 in
  let height := 30 - (-30) in
  base * height

theorem bounded_region_area_is_1800 :
  ∃ (area : ℝ), area = area_bounded_region ∧ area = 1800 := by
  sorry

end bounded_region_area_is_1800_l209_209798


namespace count_odd_factors_of_360_l209_209220

-- Defining the prime factorization of 360
def prime_factorization_360 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

-- Defining the exponent range based on prime factorization for an odd factor
def exponent_ranges : List (ℕ × List ℕ) := [(3, [0, 1, 2]), (5, [0, 1])]

theorem count_odd_factors_of_360 : ∃ n : ℕ, n = 6 :=
by
  -- Conditions based on prime factorization
  have h₁ : prime_factorization_360 = [(2, 3), (3, 2), (5, 1)], from rfl
  have h₂ : exponent_ranges = [(3, [0, 1, 2]), (5, [0, 1])], from rfl
  
  -- Definitions should align with the conditions in the problem
  use 6
  sorry

end count_odd_factors_of_360_l209_209220


namespace evaluate_expression_at_third_l209_209307

theorem evaluate_expression_at_third :
  (let f : ℚ → ℚ := λ x, ((x + 2) / (x - 2))
   in ((f (1/3) + 2) / (f (1/3) - 2))) = -31 / 37 := 
by
  let f : ℚ → ℚ := λ x, (x + 2) / (x - 2)
  have h : f (1/3) = -7 / 5, by sorry
  rw [h]
  sorry

end evaluate_expression_at_third_l209_209307


namespace sum_of_last_three_coefficients_l209_209574

theorem sum_of_last_three_coefficients (a : ℝ) (h : a ≠ 0) : 
  let expansion := (1 - (1 / a)) ^ 8,
      terms := (finset.range 9).map (λ k, (binomial 8 k) * (-1)^k * a^(8 - k))
  in (terms.get 0 + terms.get 1 + terms.get 2) / a^(8 - k) + 21 :=
  sorry

end sum_of_last_three_coefficients_l209_209574


namespace range_cos_sq_alpha_cos_sq_beta_l209_209273

variable (α β : Real) 

theorem range_cos_sq_alpha_cos_sq_beta (h : 3 * sin α ^ 2 + 2 * sin β ^ 2 - 2 * sin α = 0) : 
  ∃ c, c = cos α ^ 2 + cos β ^ 2 ∧ c ∈ Set.Icc (14 / 9) 2 := 
begin
  sorry
end

end range_cos_sq_alpha_cos_sq_beta_l209_209273


namespace math_problem_l209_209773

theorem math_problem (a b c : ℝ) :
  let A := b + c - 2 * a
  let B := c + a - 2 * b
  let C := a + b - 2 * c
  in A^3 + B^3 + C^3 = A * B * C :=
by
  let A := b + c - 2 * a
  let B := c + a - 2 * b
  let C := a + b - 2 * c
  have h : A + B + C = 0 := by
    calc A + B + C
        = (b + c - 2 * a) + (c + a - 2 * b) + (a + b - 2 * c) : by refl
    ... = b + c - 2 * a + c + a - 2 * b + a + b - 2 * c   : by refl
    ... = (b - 2 * a + a + b + c -2 * c + c + a) : by ring
    ... = b - b - a + a + c - c - a + a           : by ring
    ... = 0                                       : by ring
  sorry

end math_problem_l209_209773


namespace find_m_plus_n_l209_209347

-- Defining the conditions of the problem
variables (a b : ℂ)

-- Conditions given in the problem
axiom h1 : a^2 + b^2 = 7
axiom h2 : a^3 + b^3 = 10

-- The goal is to prove that the sum of the maximum and minimum real values of (a + b) is -1.
theorem find_m_plus_n : 
  let s := {re (a + b) | a b : ℂ, a^2 + b^2 = 7, a^3 + b^3 = 10} in
  (Sup s) + (Inf s) = -1 :=
sorry

end find_m_plus_n_l209_209347


namespace number_of_odd_factors_of_360_l209_209206

theorem number_of_odd_factors_of_360 : 
  let factors (n : ℕ) := {d : ℕ | d ∣ n}
  let odd (d : ℕ) := d % 2 = 1
  let is_factor (n : ℕ) (d : ℕ) := d ∣ n
  let number_of_odd_factors (n : ℕ) := (factors n).count odd
  in number_of_odd_factors 360 = 6 :=
begin
  sorry
end

end number_of_odd_factors_of_360_l209_209206


namespace nigel_money_ratio_l209_209355

theorem nigel_money_ratio (x y : ℝ) (hx : y = 25) :
  let after_won := x + 45,
      after_gave_some := after_won - y,
      after_mother := after_gave_some + 80,
      after_gave_final := after_mother - 25,
      certain_amount := 25 in
  after_gave_final = 2 * x + certain_amount →
  after_gave_final / x = 3 :=
by
  let after_won := x + 45
  let after_gave_some := after_won - y
  let after_mother := after_gave_some + 80
  let after_gave_final := after_mother - 25
  let certain_amount := 25
  intro h
  sorry

end nigel_money_ratio_l209_209355


namespace perp_line_in_plane_l209_209742

variables {m n l : Type} {α : Type}
  [plane α] [line m] [line n] [line l]

-- Predicate indicating if a line is perpendicular to another line or a plane
def perp (p q : Type) [plane q] [line p] := sorry 
def subset (x y : Type) [plane y] [line x] := sorry

theorem perp_line_in_plane (m n : Type) [line m] [line n] [plane α] :
  perp m α → subset n α → perp m n :=
sorry

end perp_line_in_plane_l209_209742


namespace shorter_piece_length_l209_209843

noncomputable def total_length : ℝ := 140
noncomputable def ratio : ℝ := 2 / 5

theorem shorter_piece_length (x : ℝ) (y : ℝ) (h1 : x + y = total_length) (h2 : x = ratio * y) : x = 40 :=
by
  sorry

end shorter_piece_length_l209_209843


namespace sequence_a7_l209_209980

/-- 
  Given a sequence {a_n} such that a_1 + a_{2n-1} = 4n - 6, 
  prove that a_7 = 11 
-/
theorem sequence_a7 (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : a 7 = 11 :=
by
  sorry

end sequence_a7_l209_209980


namespace triangle_acuteness_l209_209041

-- Define the sides of the triangle with a common multiple x
theorem triangle_acuteness (x : ℝ) (h : 0 < x) :
  let a := 6 * x,
      b := 8 * x,
      c := 9 * x in
  a^2 + b^2 > c^2 →
  (a < b + c ∧ b < a + c ∧ c < a + b) →
  a^2 + c^2 > b^2 →
  b^2 + c^2 > a^2 →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  ∀ α β γ : ℝ, α + β + γ = π ∧ α > 0 ∧ β > 0 ∧ γ > 0 →
  0 < α ∧ α < π →
  0 < β ∧ β < π →
  0 < γ ∧ γ < π →
  α = (angle b c a) →
  β = (angle a c b) →
  γ = (angle a b c) →
  (triangle_is_acute a b c) :=
by
  -- Proof goes here
  sorry

end triangle_acuteness_l209_209041


namespace bridge_length_l209_209518

theorem bridge_length
  (train_length : ℕ) (train_speed_kmh : ℕ) (time_to_pass_bridge_sec : ℕ)
  (train_length = 510) (train_speed_kmh = 45) (time_to_pass_bridge_sec = 52) :
  bridge_length = 140 := 
by
  sorry

end bridge_length_l209_209518


namespace julia_cakes_remaining_l209_209696

namespace CakeProblem

def cakes_per_day : ℕ := 5 - 1
def days_baked : ℕ := 6
def total_cakes_baked : ℕ := cakes_per_day * days_baked
def days_clifford_eats : ℕ := days_baked / 2
def cakes_eaten_by_clifford : ℕ := days_clifford_eats

theorem julia_cakes_remaining : total_cakes_baked - cakes_eaten_by_clifford = 21 :=
by
  -- proof goes here
  sorry

end CakeProblem

end julia_cakes_remaining_l209_209696


namespace find_x_l209_209645

noncomputable def f (x : ℝ) : ℝ := 5 * x^3 - 3

theorem find_x (x : ℝ) : (f⁻¹ (-2) = x) → x = -43 := by
  sorry

end find_x_l209_209645


namespace incorrect_statement_C_l209_209471

theorem incorrect_statement_C (a b : ℤ) (h : |a| = |b|) : (a ≠ b ∧ a = -b) :=
by
  sorry

end incorrect_statement_C_l209_209471


namespace average_speed_last_40_minutes_l209_209054

noncomputable def average_speed_last_segment : ℕ → ℕ → ℕ → ℕ :=
  λ total_distance total_minutes avg_speed1 avg_speed2,
  let total_time_hours := (total_minutes : ℕ) / 60 in
  let avg_speed_overall := total_distance / total_time_hours in
  let x := (avg_speed_overall * 3) - (avg_speed1 + avg_speed2) in
  x

theorem average_speed_last_40_minutes (total_distance total_minutes avg_speed1 avg_speed2 x : ℕ) :
  total_distance = 120 →
  total_minutes = 120 →
  avg_speed1 = 50 →
  avg_speed2 = 60 →
  x = average_speed_last_segment 120 120 50 60 →
  x = 70 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [average_speed_last_segment, h1, h2, h3, h4] at h5,
  simp at h5,
  exact h5,
end

end average_speed_last_40_minutes_l209_209054


namespace find_f_2015_l209_209079

noncomputable def f (x : ℝ) : ℝ := sorry

axiom condition1 : ∀ x : ℝ, f(x + 4) = -f(x) + 2
axiom condition2 : f(-3) = 3

theorem find_f_2015 : f(2015) = -1 :=
by
  sorry

end find_f_2015_l209_209079


namespace members_left_for_treasurer_l209_209491

-- Conditions
def total_members : ℕ := 10
def probability_jarry_is_secretary_or_treasurer : ℝ := 0.2

-- Question and proof goal
theorem members_left_for_treasurer :
  ∃ (members_left : ℕ), 
    members_left = total_members - 2 :=
sorry

end members_left_for_treasurer_l209_209491


namespace find_x_l209_209255

theorem find_x (x : ℤ) (h : 3^(x - 4) = 9^3) : x = 10 := 
sorry

end find_x_l209_209255


namespace john_mary_reading_time_l209_209722

noncomputable def book_reading_time (time_Luke : ℕ) (speed_John_factor speed_Mary_factor: ℚ) : ℕ × ℕ :=
  let time_John := (time_Luke : ℚ) / speed_John_factor
  let time_Mary := (time_Luke : ℚ) / speed_Mary_factor
  (time_John.to_nat, time_Mary.to_nat)

-- Conditions given in the problem
def time_Luke := 180
def speed_John_factor := (1/2 : ℚ)
def speed_Mary_factor := (2 : ℚ)

-- Proving the time taken by John and Mary
theorem john_mary_reading_time :
  book_reading_time time_Luke speed_John_factor speed_Mary_factor = (360, 90) :=
by
  sorry

end john_mary_reading_time_l209_209722


namespace odd_factors_of_360_l209_209226

theorem odd_factors_of_360 : ∃ n, 360 = 2^3 * 3^2 * 5^1 ∧ n = 6 :=
by 
  -- Define the prime factorization of 360
  let pf360 := (2^3 * 3^2 * 5^1)
  
  -- Define the count of odd factors by removing the contribution of factor 2.
  let odd_part := (3^2 * 5^1)
  
  -- Calculate the number of factors.
  have odd_factors_count_eq : (2 + 1) * (1 + 1) = 6 := by 
    calc (2 + 1) * (1 + 1) = 3 * 2 : by rfl
                         ... = 6 : by rfl
  
  -- Assert the existence of such n matching the count of odd factors.
  exact ⟨6, And.intro rfl odd_factors_count_eq⟩

end odd_factors_of_360_l209_209226


namespace count_valid_ks_l209_209628

theorem count_valid_ks : 
  ∃ (ks : Finset ℕ), (∀ k ∈ ks, k > 0 ∧ k ≤ 50 ∧ 
    ∀ n : ℕ, n > 0 → 7 ∣ (2 * 3^(6 * n) + k * 2^(3 * n + 1) - 1)) ∧ ks.card = 7 :=
sorry

end count_valid_ks_l209_209628


namespace max_distance_to_line_l209_209763

-- Define the parametric equations of the line l
def line_parametric_eq (t : ℝ) : ℝ × ℝ :=
  (-real.sqrt(3) * t, 1 + t)

-- Define the polar equation of the curve C
def polar_eq_curve_C (rho theta : ℝ) : Prop :=
  rho^2 * (real.cos theta)^2 + 3 * rho^2 * (real.sin theta)^2 = 3

-- Define the maximum distance function for a point on curve C to line l
def max_distance (rho theta : ℝ) (M : ℝ × ℝ) :=
  let x := real.sqrt(3) * real.cos theta in
  let y := real.sin theta in
  let d := abs (x + real.sqrt(3) * y - real.sqrt(3)) / 2 in
  (rho^2 * (real.cos theta)^2 + 3 * rho^2 * (real.sin theta)^2 = 3) ∧
  M = (x, y) ∧
  d = (real.sqrt(3) * abs (real.sqrt(2) * real.sin (theta + real.pi / 4) - 1)) / 2

theorem max_distance_to_line :
  ∃ (rho theta : ℝ) (M : ℝ × ℝ), max_distance rho theta M ∧
  M = (-real.sqrt(6) / 2, -real.sqrt(2) / 2) :=
by
  sorry

end max_distance_to_line_l209_209763


namespace simplify_expression_l209_209550

theorem simplify_expression (a b : ℝ) :
  ((3 * a^3 * b - 12 * a^2 * b^2 - 6 * a * b^3) / (-3 * a * b) - 4 * a * b) = (-a^2 + 2 * b^2) :=
by
  sorry

end simplify_expression_l209_209550


namespace parametric_graph_intersections_l209_209072

noncomputable def parametric_x (t : ℝ) : ℝ := (cos t) + (t / 3)
noncomputable def parametric_y (t : ℝ) : ℝ := sin t

theorem parametric_graph_intersections :
  ∃ n_inters : ℕ, n_inters = 12 ∧
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ t₁ ≥ 0 ∧ t₂ ≥ 0 ∧ t₁ ≤ 60 ∧ t₂ ≤ 60 ∧ 
  parametric_x t₁ = parametric_x t₂ :=
sorry

end parametric_graph_intersections_l209_209072


namespace right_triangle_area_l209_209040

theorem right_triangle_area {DE DF : ℝ} (hDE : DE = 8) (hDF : DF = 3) (hRightAngle : ∠D = 90) :
  (1/2) * DE * DF = 12 := 
by
  sorry

end right_triangle_area_l209_209040


namespace sapling_planting_methods_correct_l209_209086

-- Definition of the problem conditions and result
def sapling_planting_methods (A B C : Type) : Nat :=
if A ≠ B ∧ B ≠ C ∧ A ≠ C then 6 else 0

-- Theorem statement - the number of planting methods is 6
theorem sapling_planting_methods_correct (A B C : Type) (h : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  sapling_planting_methods A B C = 6 :=
by
  sorry

end sapling_planting_methods_correct_l209_209086


namespace tetrahedron_distance_sum_l209_209148

theorem tetrahedron_distance_sum (S₁ S₂ S₃ S₄ H₁ H₂ H₃ H₄ V k : ℝ) 
  (h1 : S₁ = k) (h2 : S₂ = 2 * k) (h3 : S₃ = 3 * k) (h4 : S₄ = 4 * k)
  (V_eq : (1 / 3) * S₁ * H₁ + (1 / 3) * S₂ * H₂ + (1 / 3) * S₃ * H₃ + (1 / 3) * S₄ * H₄ = V) :
  1 * H₁ + 2 * H₂ + 3 * H₃ + 4 * H₄ = (3 * V) / k :=
by
  sorry

end tetrahedron_distance_sum_l209_209148


namespace custom_modular_home_cost_l209_209902

noncomputable theory

-- Definitions for square footage and costs per module
def kitchen_sf : ℕ := 500
def kitchen_cost : ℕ := 35000

def bathroom_sf : ℕ := 250
def bathroom_cost : ℕ := 15000
def num_bathrooms : ℕ := 3

def bedroom_sf : ℕ := 350
def bedroom_cost : ℕ := 21000
def num_bedrooms : ℕ := 4

def living_area_sf : ℕ := 600
def living_area_cost_per_sf : ℕ := 100

def upgraded_cost_per_sf : ℕ := 150

-- Total area of the modular home
def total_sf : ℕ := 3500

-- Calculate the remaining square footage for office and media room
def remaining_sf := total_sf - (kitchen_sf + num_bathrooms * bathroom_sf + num_bedrooms * bedroom_sf + living_area_sf)
def each_upgraded_sf := remaining_sf / 2

-- Calculate the total cost
def total_cost : ℕ :=
  kitchen_cost +
  (num_bathrooms * bathroom_cost) +
  (num_bedrooms * bedroom_cost) +
  (living_area_sf * living_area_cost_per_sf) +
  (each_upgraded_sf * upgraded_cost_per_sf * 2)

theorem custom_modular_home_cost :
  total_cost = 261500 := by
  sorry

end custom_modular_home_cost_l209_209902


namespace sum_of_squared_projections_l209_209616

theorem sum_of_squared_projections (a l m n : ℝ) (l_proj m_proj n_proj : ℝ)
  (h : l_proj = a * Real.cos θ)
  (h1 : m_proj = a * Real.cos (Real.pi / 3 - θ))
  (h2 : n_proj = a * Real.cos (Real.pi / 3 + θ)) :
  l_proj ^ 2 + m_proj ^ 2 + n_proj ^ 2 = 3 / 2 * a ^ 2 :=
by sorry

end sum_of_squared_projections_l209_209616


namespace odd_factors_of_360_l209_209170

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l209_209170


namespace intensity_of_replacement_paint_l209_209387

-- Define the given conditions
def initial_intensity : ℝ := 50 / 100
def fraction_replaced : ℝ := 0.6
def resulting_intensity : ℝ := 35 / 100

-- Define the variable for the replacement intensity
def replacement_intensity : ℝ := 0.25

-- The theorem statement representing the problem
theorem intensity_of_replacement_paint :
  let I := (resulting_intensity - (1 - fraction_replaced) * initial_intensity) / fraction_replaced in
  I = replacement_intensity :=
by
  sorry

end intensity_of_replacement_paint_l209_209387


namespace books_sold_correct_l209_209854

-- Define the initial number of books, number of books added, and the final number of books.
def initial_books : ℕ := 41
def added_books : ℕ := 2
def final_books : ℕ := 10

-- Define the number of books sold.
def sold_books : ℕ := initial_books + added_books - final_books

-- The theorem we need to prove: the number of books sold is 33.
theorem books_sold_correct : sold_books = 33 := by
  sorry

end books_sold_correct_l209_209854


namespace count_repeating_decimals_l209_209111

def is_repeating_decimal (a b : ℕ) :=
  let d := Nat.factorization b
  d ≠ 0 ∧ (¬ d.find (λ p => p = 2) = some 0) ∧ (¬ d.find (λ p => p = 5) = some 0)

theorem count_repeating_decimals :
  let nums := (List.range 15).map (λ n => n + 1)
  let count := List.countp (λ n => is_repeating_decimal n 18) nums 
  count = 10 :=
by
  sorry

end count_repeating_decimals_l209_209111


namespace problem_U_complement_eq_l209_209991

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l209_209991


namespace divisor_add_sub_l209_209001

variable {R : Type*} [CommRing R] {x : R}

-- Defining polynomials D, P, Q as variables
variables (D P Q : R[X])

-- Given conditions
def D_divides_PQ : Prop :=
  D ∣ P ∧ D ∣ Q

-- Theorem statement
theorem divisor_add_sub (h : D_divides_PQ D P Q) : D ∣ (P + Q) ∧ D ∣ (P - Q) :=
by sorry

end divisor_add_sub_l209_209001


namespace power_addition_prime_l209_209783

theorem power_addition_prime (p a n : ℕ) (hp : p.prime) (ha : a > 0) (hn : n > 0) (h : 2^p + 3^p = a^n) : n = 1 :=
by
  sorry

end power_addition_prime_l209_209783


namespace sec_330_eq_2_div_sqrt_3_l209_209093

theorem sec_330_eq_2_div_sqrt_3 :
  real.sec (330 * real.pi / 180) = 2 / real.sqrt 3 := by
  -- Since we are given conditions to use, we can assume specific properties directly
  -- such as the period of the cosine function and specific angle values.
  sorry

end sec_330_eq_2_div_sqrt_3_l209_209093


namespace odd_factors_of_360_l209_209172

theorem odd_factors_of_360 : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, k ∣ 360 → k % 2 = 1 ↔ k ∣ (3^2 * 5^1) := 
by
  sorry

end odd_factors_of_360_l209_209172


namespace find_AE_length_l209_209495

def parallelogram (A B C D : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ 
  (∃ l1 l2 : Line, l1 ≠ l2 ∧ points_on_line l1 {A, B} ∧ points_on_line l2 {A, D})

def circle_through (circle : Circle) (A B C : Point) : Prop := 
  points_on_circle circle {A, B, C}

def tangent (line : Line) (circle : Circle) (P : Point) : Prop :=
  point_on_line P line ∧ point_on_circle P circle ∧ ∀ Q, Q ≠ P ∧ point_on_line Q line -> ¬ point_on_circle Q circle

def intersects (line : Line) (circle : Circle) (A B : Point) : Prop :=
  point_on_line A line ∧ point_on_circle A circle ∧ point_on_line B line ∧ point_on_circle B circle

-- Definitions based on problem conditions
noncomputable def AE_length (A B C D E : Point) (AD CE AE : ℝ) : Prop :=
  parallelogram A B C D ∧ 
  circle_through (circle B C D) B C D ∧ 
  tangent (line AD) (circle B C D) D ∧ 
  intersects (line AB) (circle B C D) B E ∧ 
  AD = 4 ∧ 
  CE = 5 ∧ 
  AE = 16/5

-- Statement of the problem in Lean 4
theorem find_AE_length (A B C D E : Point) (circle : Circle) (AD CE AE : ℝ) :
  AE_length A B C D E AD CE AE → AE = 16/5 :=
by
  intros h,
  exact h.6.2

end find_AE_length_l209_209495


namespace product_of_all_real_values_of_r_l209_209592

theorem product_of_all_real_values_of_r :
  ∀ (r : ℝ), (∃ x : ℝ, x ≠ 0 ∧ (1 / (3 * x) = (r - 2 * x) / 10))
  ∧ (b^2 - 4 * a * c = 0) → ∏ (r : set ℝ), r = -80 / 3 :=
begin
  sorry
end

end product_of_all_real_values_of_r_l209_209592


namespace triangle_circle_intersection_l209_209522

noncomputable def triangle : Type := sorry
noncomputable def circle : Type := sorry
noncomputable def area (t : Type) : ℝ := sorry

theorem triangle_circle_intersection (t : triangle) (c : circle) :
  area (t ∩ c) ≤ (1/3) * area t + (1/2) * area c :=
sorry

end triangle_circle_intersection_l209_209522


namespace odd_factors_360_l209_209178

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l209_209178


namespace find_m_for_decreasing_power_fun_l209_209161

theorem find_m_for_decreasing_power_fun (m : ℝ) : 
  (∀ x : ℝ, (0 < x) → 
  (∃ m : ℝ, y = (m^2 - 5*m - 5)*x^(2*m+1) → 
  is_decreasing (λ x, (m^2 - 5*m - 5)*x^(2*m+1)) (0, ⊤))) ↔ m = -1 := 
begin
  sorry
end

end find_m_for_decreasing_power_fun_l209_209161


namespace probability_of_half_top_grade_parts_l209_209421

noncomputable def probability_top_grade_parts 
  (n : ℕ) (m : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - p in
  let np := n * p in
  let sigma := Real.sqrt (np * q) in
  let x := (m - np) / sigma in
  let phi := Mathlib.Probability.StandardNormal.cdf x in
  phi / sigma

theorem probability_of_half_top_grade_parts :
  probability_top_grade_parts 26 13 0.4 ≈ 0.093 :=
by sorry

end probability_of_half_top_grade_parts_l209_209421


namespace number_of_valid_triangles_l209_209238

-- Definition of the set of points in the 5x5 grid with integer coordinates
def gridPoints := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

-- Function to determine if three points are collinear
def collinear (a b c : ℕ × ℕ) : Prop :=
  (b.2 - a.2) * (c.1 - b.1) = (c.2 - b.2) * (b.1 - a.1)

-- The main theorem stating the number of triangles with positive area
theorem number_of_valid_triangles : 
  ∃ n, n = 2158 ∧ ∀ (a b c : ℕ × ℕ), a ∈ gridPoints → b ∈ gridPoints → c ∈ gridPoints → a ≠ b → b ≠ c → c ≠ a → ¬collinear a b c → n = 2158 :=
by
  sorry

end number_of_valid_triangles_l209_209238


namespace problem1_problem2_l209_209780

-- First problem: Simplify and evaluate 4(m+1)^2 - (2m+5)(2m-5) given m = -3
theorem problem1 (m : ℤ) (h : m = -3) : 4 * (m + 1)^2 - (2 * m + 5) * (2 * m - 5) = 5 :=
by {
  rw h,
  -- detailed proof would go here
  sorry
}

-- Second problem: Simplify and evaluate (x^2-1) / (x^2+2x) ÷ (x - 1) / x given x = 2
theorem problem2 (x : ℚ) (h : x = 2) :
  (x^2 - 1) / (x^2 + 2 * x) / ((x - 1) / x) = 3 / 4 :=
by {
  rw h,
  -- detailed proof would go here
  sorry
}

end problem1_problem2_l209_209780


namespace new_person_weight_l209_209786

variable (W : ℝ)
variable (avg_weight_increase : ℝ) (group_size : ℕ) (replaced_weight : ℝ)

-- All the given conditions:
def average_weight_condition := avg_weight_increase = 4
def group_size_condition := group_size = 12
def replaced_weight_condition := replaced_weight = 65

-- The proof problem:
theorem new_person_weight 
  (avg_weight_increase_condition : average_weight_condition)
  (group_size_condition_given : group_size_condition)
  (replaced_weight_condition_given : replaced_weight_condition) :
  W = 113 := by
  sorry

end new_person_weight_l209_209786


namespace total_houses_in_lincoln_county_l209_209432

theorem total_houses_in_lincoln_county 
  (original_houses : ℕ) 
  (houses_built : ℕ) 
  (h_original : original_houses = 20817) 
  (h_built : houses_built = 97741) : 
  original_houses + houses_built = 118558 := 
by 
  -- Proof steps or tactics would go here
  sorry

end total_houses_in_lincoln_county_l209_209432


namespace relationship_between_a_and_b_l209_209606

theorem relationship_between_a_and_b (a b : ℝ) (h1 : 3^a = 15) (h2 : 5^b = 15) :
  (1 / a + 1 / b = 1) ∧ (a * b > 4) ∧ ((a + 1)^2 + (b + 1)^2 > 16) :=
by
  sorry

end relationship_between_a_and_b_l209_209606


namespace real_part_of_z_l209_209122

-- Define the condition: z + 2conj(z) = 6 + i
def z_condition (z : ℂ) : Prop := z + 2 * conj(z) = 6 + complex.I

-- Define the proof problem: If z satisfies the condition, then the real part of z is 2
theorem real_part_of_z (z : ℂ) (h : z_condition z) : z.re = 2 := 
by
  sorry

end real_part_of_z_l209_209122


namespace mode_of_data_set_l209_209615

theorem mode_of_data_set (x : ℤ) (h_median: (x + 4) / 2 = 7) : 
    mode [-8, -1, 4, x, 10, 13] = 10 := 
begin
  sorry
end

end mode_of_data_set_l209_209615


namespace complement_of_A_is_correct_l209_209163

open Set

variable (U : Set ℝ) (A : Set ℝ)

def complement_of_A (U : Set ℝ) (A : Set ℝ) :=
  {x : ℝ | x ∉ A}

theorem complement_of_A_is_correct :
  (U = univ) →
  (A = {x : ℝ | x^2 - 2 * x > 0}) →
  (complement_of_A U A = {x : ℝ | 0 ≤ x ∧ x ≤ 2}) :=
by
  intros hU hA
  simp [hU, hA, complement_of_A]
  sorry

end complement_of_A_is_correct_l209_209163


namespace common_difference_arith_seq_log_l209_209338

variable {p q r : ℝ}

def geom_seq (a b c : ℝ) : Prop := b^2 = a * c
def arith_seq (a b c : ℝ) (d : ℝ) : Prop := b - a = d ∧ c - b = d

theorem common_difference_arith_seq_log:
  ∀ (p q r : ℝ), 
  p ≠ q → p ≠ r → q ≠ r → 0 < p → 0 < q → 0 < r → 
  geom_seq p q r → 
  (∃ d, arith_seq (Real.log r p) (Real.log q r) (Real.log p q) d) → d = 3 / 2 :=
by 
  sorry

end common_difference_arith_seq_log_l209_209338


namespace largest_possible_perimeter_l209_209525

theorem largest_possible_perimeter (y : ℤ) (hy1 : 3 ≤ y) (hy2 : y < 16) : 7 + 9 + y ≤ 31 := 
by
  sorry

end largest_possible_perimeter_l209_209525


namespace monthly_rent_l209_209492

-- Definitions based on the given conditions
def length_ft : ℕ := 360
def width_ft : ℕ := 1210
def sq_feet_per_acre : ℕ := 43560
def cost_per_acre_per_month : ℕ := 60

-- Statement of the problem
theorem monthly_rent : (length_ft * width_ft / sq_feet_per_acre) * cost_per_acre_per_month = 600 := sorry

end monthly_rent_l209_209492


namespace odd_factors_of_360_l209_209190

theorem odd_factors_of_360 : ∃ n, n = 6 ∧ 
  ∀ (x : ℕ), (∃ (a b : ℕ), 360 = 2^a * 3^b * x ∧ ¬ even x ∧ 0 < x) → (∀ (m : ℕ), m ∣ x ↔ (m = 1 ∨ m = 3 ∨ m = 5 ∨ m = 9 ∨ m = 15 ∨ m = 45)) :=
by
  -- Proof
  sorry

end odd_factors_of_360_l209_209190


namespace real_part_of_complex_eq_2_l209_209125

-- Definition of the problem
def complex_satisfying_equation (z : ℂ) : Prop :=
  z + 2 * conj z = 6 + complex.i

-- Proving the real part of z in given conditions
theorem real_part_of_complex_eq_2 (z : ℂ) (h : complex_satisfying_equation z) : z.re = 2 :=
by {
  sorry
}

end real_part_of_complex_eq_2_l209_209125


namespace no_j_satisfies_condition_l209_209080

-- Define f(j) as the sum of all positive divisors of j
noncomputable def f (j : ℕ) : ℕ := ∑ d in (finset.range (j + 1)).filter (λ d, j % d = 0), d

-- Main theorem statement
theorem no_j_satisfies_condition :
  ¬ ∃ j : ℕ, 1 ≤ j ∧ j ≤ 5000 ∧ f(j) = 1 + j + 2 * nat.sqrt j :=
by
  sorry

end no_j_satisfies_condition_l209_209080


namespace count_odd_factors_of_360_l209_209222

-- Defining the prime factorization of 360
def prime_factorization_360 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 1)]

-- Defining the exponent range based on prime factorization for an odd factor
def exponent_ranges : List (ℕ × List ℕ) := [(3, [0, 1, 2]), (5, [0, 1])]

theorem count_odd_factors_of_360 : ∃ n : ℕ, n = 6 :=
by
  -- Conditions based on prime factorization
  have h₁ : prime_factorization_360 = [(2, 3), (3, 2), (5, 1)], from rfl
  have h₂ : exponent_ranges = [(3, [0, 1, 2]), (5, [0, 1])], from rfl
  
  -- Definitions should align with the conditions in the problem
  use 6
  sorry

end count_odd_factors_of_360_l209_209222


namespace sum_b_sequence_eq_l209_209135

noncomputable def a_sequence : ℕ → ℝ 
| 0     := 1/2
| (n+1) := (1/2) * a_sequence n

def b_sequence (n : ℕ) : ℝ := Real.log2 (a_sequence n)

noncomputable def sum_b_sequence (n : ℕ) : ℝ :=
∑ k in Finset.range n, 1 / (b_sequence k * b_sequence (k + 1))

theorem sum_b_sequence_eq (n : ℕ) (n_pos : 0 < n) :
  sum_b_sequence n = n / (n + 1) := sorry

end sum_b_sequence_eq_l209_209135


namespace visitors_that_day_l209_209529

theorem visitors_that_day (total_visitors : ℕ) (previous_day_visitors : ℕ) 
  (h_total : total_visitors = 406) (h_previous : previous_day_visitors = 274) : 
  total_visitors - previous_day_visitors = 132 :=
by
  sorry

end visitors_that_day_l209_209529


namespace odd_factors_360_l209_209215

theorem odd_factors_360 : 
  let fac := (2^3) * (3^2) * (5^1) in
  fac = 360 → 
  num_factors (3^2 * 5^1) = 6 :=
sorry

end odd_factors_360_l209_209215


namespace bugs_meet_at_point_P_l209_209825

theorem bugs_meet_at_point_P (r1 r2 v1 v2 t : ℝ) (h1 : r1 = 7) (h2 : r2 = 3) (h3 : v1 = 4 * Real.pi) (h4 : v2 = 3 * Real.pi) :
  t = 14 :=
by
  repeat { sorry }

end bugs_meet_at_point_P_l209_209825


namespace find_x_solution_l209_209457

theorem find_x_solution :
    ∃ x : ℕ, x + (3 * 14 + 3 * 15 + 3 * 18) = 152 ∧ x = 11 :=
begin
  sorry
end

end find_x_solution_l209_209457


namespace ducks_born_per_year_l209_209918

theorem ducks_born_per_year :
  ∃ (x : ℕ), 
    (100 - 20 * 1 + x = 100 - 20 + x) ∧
    (100 - 20 * 2 + 2 * x = 100 - 40 + 2 * x) ∧ 
    (100 - 20 * 3 + 3 * x = 100 - 60 + 3 * x) ∧ 
    (100 - 20 * 4 + 4 * x = 100 - 80 + 4 * x) ∧ 
    (100 - 20 * 5 + 5 * x = 100 - 100 + 5 * x) ∧ 
    (5 * x + 150 = 300) ∧
    x = 30 :=
begin
  use 30,
  split, 
  { linarith, },
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { linarith, },
  split,
  { linarith, },
  linarith,
sorry,

end

end ducks_born_per_year_l209_209918


namespace find_f_x2_minus_1_l209_209736

def f (x : ℝ) := sorry

theorem find_f_x2_minus_1 (x : ℝ)
  (h : f (x^2 + 1) = x^4 + 5 * x^3 + 4 * x^2) :
  f (x^2 - 1) = x^4 - 4 * x^2 + 4 + 5 * ((x^2 - 2) ^ (3 / 2)) + 4 * (x^2 - 2) :=
sorry

end find_f_x2_minus_1_l209_209736


namespace arithmetic_sequence_num_terms_l209_209903

theorem arithmetic_sequence_num_terms : 
  ∀ (a1 d : ℤ) (a_last : ℤ) (n : ℕ), a1 = -48 → d = 8 → a_last = 80 → 
  a_last = a1 + (↑n - 1) * d → n = 17 :=
begin
  intros a1 d a_last n h1 h2 h3 h4,
  sorry
end

end arithmetic_sequence_num_terms_l209_209903


namespace distinct_students_count_l209_209884

-- Definition of the initial parameters
def num_gauss : Nat := 12
def num_euler : Nat := 10
def num_fibonnaci : Nat := 7
def overlap : Nat := 1

-- The main theorem to prove
theorem distinct_students_count : num_gauss + num_euler + num_fibonnaci - overlap = 28 := by
  sorry

end distinct_students_count_l209_209884


namespace largest_triangle_perimeter_maximizes_l209_209523

theorem largest_triangle_perimeter_maximizes 
  (y : ℤ) 
  (h1 : 3 ≤ y) 
  (h2 : y < 16) : 
  (7 + 9 + y) = 31 ↔ y = 15 := 
by 
  sorry

end largest_triangle_perimeter_maximizes_l209_209523


namespace regression_analysis_statements_correct_l209_209466

theorem regression_analysis_statements_correct 
    (r : ℝ)
    (R2 : ℝ)
    (residual_points : Set (ℝ × ℝ))
    (model_appropriate : Prop)
    (prediction_accuracy : Prop)
    (forecast_values_approximate : Prop) :
  (|r| ≤ 1 ∧ (|r| = 1 → strong_correlation) ∧ (|r| = 0 → weak_correlation)) →
  (R2 ≥ 0 ∧ R2 ≤ 1 ∧ (R2 = 1 → perfect_fit) ∧ (R2 = 0 → no_fit)) →
  (model_appropriate → evenly_distributed residual_points) →
  (forecast_values_approximate → prediction_accuracy) →
  (1 ∧ 3 ∧ 4) = (correct_statement) :=
by 
  sorry

end regression_analysis_statements_correct_l209_209466


namespace determine_white_balls_l209_209819

theorem determine_white_balls (R B W : ℕ) 
  (h1 : R = 80) 
  (h2 : B = 40) 
  (h3 : R = B + W - 12) : 
  W = 52 := 
by 
  rw [h1, h2] at h3
  linarith

end determine_white_balls_l209_209819


namespace find_x_l209_209260

theorem find_x (x : ℝ) : (3^(x - 4) = 9^3) → x = 10 :=
by
  sorry

end find_x_l209_209260


namespace g_increasing_in_interval_l209_209962

theorem g_increasing_in_interval (a : ℝ) 
  (h : ∀ x : ℝ, x < 1 → (x^2 - 2*a*x + a) ≥ (a^2 - 2*a*a + a)) :
  ∀ x : ℝ, 1 < x → (g'(x) > 0) :=
by
  let f : ℝ → ℝ := λ x, (1/3)*x^3 - a*x^2 + a*x + 2
  let f' : ℝ → ℝ := λ x, x^2 - 2*a*x + a
  let g : ℝ → ℝ := λ x, (f'(x)) / x
  let g' : ℝ → ℝ := λ x, 1 - a/(x^2)
  assume x : ℝ
  assume hx : 1 < x
  have ha : a < 0, from sorry
  have h_deriv : g'(x) = 1 - a/(x^2), from sorry
  show g'(x) > 0, from sorry

end g_increasing_in_interval_l209_209962


namespace magnitude_z_minus_one_l209_209336

open Complex

theorem magnitude_z_minus_one (z : ℂ) (h : I * z = 1 + 2 * I) : abs (z - 1) = real.sqrt 2 := 
sorry

end magnitude_z_minus_one_l209_209336


namespace polygon_with_20_diagonals_is_octagon_l209_209858

theorem polygon_with_20_diagonals_is_octagon :
  ∃ (n : ℕ), n ≥ 3 ∧ (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end polygon_with_20_diagonals_is_octagon_l209_209858


namespace max_incorrect_questions_l209_209056

def max_incorrect_to_pass (total_questions : ℕ) (percentage_to_pass : ℕ) : ℕ :=
  let incorrect_percentage := 100 - percentage_to_pass
  let incorrect_questions := total_questions * incorrect_percentage / 100
  incorrect_questions

theorem max_incorrect_questions (total_questions : ℕ) (percentage_to_pass : ℕ) :
  total_questions = 50 → percentage_to_pass = 85 → max_incorrect_to_pass total_questions percentage_to_pass = 7 :=
by
  intros h₁ h₂
  simp [max_incorrect_to_pass]
  rw [h₁, h₂]
  norm_num
  sorry

end max_incorrect_questions_l209_209056


namespace supplementary_angle_l209_209678

theorem supplementary_angle {α : ℝ} (h : 90 - α = 125) : 180 - α = 125 := by
  sorry

end supplementary_angle_l209_209678


namespace sheets_per_ream_l209_209015

-- defining necessary variables and conditions
variables (cost_per_ream : ℕ) (total_sheets_needed : ℕ) (total_cost : ℕ)

-- conditions
def condition_1 := cost_per_ream = 27
def condition_2 := total_sheets_needed = 5000
def condition_3 := total_cost = 270

-- theorem stating the number of sheets in one ream
theorem sheets_per_ream (S : ℕ) (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : S = 500 :=
by
  -- the proof steps will be written here, but we skip them now
  sorry

end sheets_per_ream_l209_209015


namespace sum_of_external_angles_of_cyclic_quadrilateral_l209_209027

variable {α : Type*} [linear_ordered_field α]

/-- A cyclic quadrilateral is one that can be inscribed in a circle. -/
structure cyclic_quadrilateral (A B C D : α) : Prop :=
(is_cyclic : ∃ (Ω : Type*) [circle Ω], inscribed Ω A B C D)

theorem sum_of_external_angles_of_cyclic_quadrilateral (A B C D : α) 
(h : cyclic_quadrilateral A B C D) :
  let θ₁ := 360 - (angle A B C)
  let θ₂ := 360 - (angle B C D)
  let θ₃ := 360 - (angle C D A)
  let θ₄ := 360 - (angle D A B) 
  θ₁ + θ₂ + θ₃ + θ₄ = 360 :=
sorry

end sum_of_external_angles_of_cyclic_quadrilateral_l209_209027


namespace likelihood_of_white_crows_at_birch_unchanged_l209_209433

theorem likelihood_of_white_crows_at_birch_unchanged 
  (a b c d : ℕ) 
  (h1 : a + b = 50) 
  (h2 : c + d = 50) 
  (h3 : b ≥ a) 
  (h4 : d ≥ c - 1) : 
  (bd + ac + a + b : ℝ) / 2550 > (bc + ad : ℝ) / 2550 := by 
  sorry

end likelihood_of_white_crows_at_birch_unchanged_l209_209433


namespace find_pairs_l209_209094

theorem find_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (2 * m^2 + n^2) ∣ (3 * m * n + 3 * m) ↔ (m, n) = (1, 1) ∨ (m, n) = (4, 2) ∨ (m, n) = (4, 10) :=
sorry

end find_pairs_l209_209094


namespace equation_of_perpendicular_line_l209_209792

theorem equation_of_perpendicular_line (a b c : ℝ) (p q : ℝ) (hx : a ≠ 0) (hy : b ≠ 0)
  (h_perpendicular : a * 2 + b * 1 = 0) (h_point : (-1) * a + 2 * b + c = 0)
  : a = 1 ∧ b = -2 ∧ c = -5 → (x:ℝ) * 1 + (y:ℝ) * (-2) + (-5) = 0 :=
by sorry

end equation_of_perpendicular_line_l209_209792


namespace decimal_150th_place_7_div_11_l209_209835

theorem decimal_150th_place_7_div_11 :
  let decimal_rep : ℕ → ℕ := λ n, ite (n % 2 = 0) 3 6
  (decimal_rep 150) = 3 :=
by
  -- Proving the statement directly here
  simp [decimal_rep]
  sorry

end decimal_150th_place_7_div_11_l209_209835


namespace find_x_l209_209910

theorem find_x (x : ℝ) (h : (x / 2) + 6 = 2 * x - 6) : x = 8 :=
by
  sorry

end find_x_l209_209910


namespace Vasya_has_more_ways_l209_209365

def king (pos: ℕ × ℕ) (positions: Finset (ℕ × ℕ)) : Prop :=
  ∀ (x y: ℕ × ℕ), x ≠ y ∧ x ∈ positions ∧ y ∈ positions → 
  (abs (x.1 - y.1) > 1 ∨ abs (x.2 - y.2) > 1)

def PetyaBoard : Finset (ℕ × ℕ) := 
{p | p.1 < 100 ∧ p.2 < 50}

def VasyaBoard : Finset (ℕ × ℕ) := 
{p | p.1 < 100 ∧ p.2 < 100 ∧ (p.1 + p.2) % 2 = 0}

theorem Vasya_has_more_ways :
  (∃ ps : Finset (ℕ × ℕ), ps.card = 500 ∧ king ps) → 
  (∃ ps : Finset (ℕ × ℕ), ps.card = 500 ∧ king ps) :=
sorry

end Vasya_has_more_ways_l209_209365


namespace max_distinct_integer_squares_sum_2500_l209_209453

theorem max_distinct_integer_squares_sum_2500 :
  ∃ (a : Fin 18 → ℕ), (∀ i j, i ≠ j → a i ≠ a j) ∧ (∀ i, 0 < a i) ∧ (∑ i, (a i)^2 = 2500) := sorry

end max_distinct_integer_squares_sum_2500_l209_209453


namespace find_a_l209_209745

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 9 * Real.log x

def monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

def valid_interval (a : ℝ) : Prop :=
  monotonically_decreasing f (Set.Icc (a-1) (a+1))

theorem find_a :
  {a : ℝ | valid_interval a} = {a : ℝ | 1 < a ∧ a ≤ 2} :=
by
  sorry

end find_a_l209_209745


namespace a_8_value_l209_209312

noncomputable def a_seq (n : ℕ) : ℝ :=
  sorry

def b_seq (n : ℕ) : ℝ :=
  1 / (a_seq n + 1)

theorem a_8_value (q : ℝ) (h2 : a_seq 2 = 1) (h5 : a_seq 5 = 3) (hg : ∀ n m, b_seq (n + m) = (b_seq n) * (b_seq m)) : a_seq 8 = 7 :=
by
  sorry

end a_8_value_l209_209312


namespace range_of_m_l209_209282

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → -3 ≤ m ∧ m ≤ 5 :=
by
  sorry

end range_of_m_l209_209282


namespace number_of_odd_factors_of_360_l209_209203

theorem number_of_odd_factors_of_360 : 
  let factors (n : ℕ) := {d : ℕ | d ∣ n}
  let odd (d : ℕ) := d % 2 = 1
  let is_factor (n : ℕ) (d : ℕ) := d ∣ n
  let number_of_odd_factors (n : ℕ) := (factors n).count odd
  in number_of_odd_factors 360 = 6 :=
begin
  sorry
end

end number_of_odd_factors_of_360_l209_209203


namespace probability_no_adjacent_standing_l209_209393

-- Conditions
def fair_coin : Type := {
  outcome : ℕ → Prop,
  total_outcomes : ℕ
}

def initial_conditions : ℕ → ℕ 
| 2     := 3
| 3     := 4
| (n+1) := initial_conditions n + initial_conditions (n-1)

def num_people : ℕ := 10

-- Desired proof
theorem probability_no_adjacent_standing (p : fair_coin) :
  initial_conditions num_people = 123 → 
  (123 : ℚ) / (2 ^ 10 : ℚ) = 123 / 1024 := sorry

end probability_no_adjacent_standing_l209_209393


namespace regular_polygon_properties_l209_209037

theorem regular_polygon_properties 
  (perimeter : ℝ) (side_length : ℝ) (n : ℕ)
  (h_perimeter : perimeter = 180) 
  (h_side_length : side_length = 15) 
  (h_n : n = perimeter / side_length) : 
  n = 12 ∧ (180 * (n - 2)) / n = 150 :=
by
  -- Given conditions
  have h1 : perimeter = 180 := h_perimeter,
  have h2 : side_length = 15 := h_side_length,
  have h3 : n = perimeter / side_length := h_n,
  
  -- Statement: the polygon has 12 sides
  have h_sides : n = 12 := 
    by 
      rw [h1, h2] at h3;
      rw [h3];
      exact rfl,

  -- Statement: the measure of each interior angle
  have h_angle : (180 * (n - 2)) / n = 150 := 
    by 
      rw [h_sides];
      simp;
      exact rfl,
  
  -- Conclusion
  exact ⟨h_sides, h_angle⟩

end regular_polygon_properties_l209_209037


namespace omega_possible_values_eq_two_l209_209795

theorem omega_possible_values_eq_two
  (f : ℝ → ℝ) (ω : ℝ) (φ : ℝ)
  (h1 : ω > 0)
  (h2 : f = (λ x, Real.sin (ω * x + φ)))
  (h3 : f (π / 6) = 1)
  (h4 : f (π / 3) = 0)
  (h5 : ∀ x y, (π / 6 < x ∧ x < π / 4) → (π / 6 < y ∧ y < π / 4) → (x < y → f x ≤ f y)) :
  ∃ n : ℕ, n = 2 := 
sorry

end omega_possible_values_eq_two_l209_209795


namespace a_20_is_46_l209_209708

def sequence (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, a(2 * n) = a(2 * n - 1) + (-1)^n) ∧
  (∀ n : ℕ, a(2 * n + 1) = a(2 * n) + n) ∧
  (a 1 = 1)

theorem a_20_is_46 (a : ℕ → ℕ) (h : sequence a) : a 20 = 46 :=
by
  sorry

end a_20_is_46_l209_209708


namespace tangent_from_point_to_circle_l209_209081

theorem tangent_from_point_to_circle :
  ∀ (x y : ℝ),
  (x - 6)^2 + (y - 3)^2 = 4 →
  (x = 10 → y = 0 →
    4 * x - 3 * y = 19) :=
by
  sorry

end tangent_from_point_to_circle_l209_209081


namespace calculate_profit_l209_209026

noncomputable def profit_calculation (SP : ℝ) (profit_percentage : ℝ) : ℝ :=
  let CP := SP / (1 + profit_percentage / 100)
  in (profit_percentage / 100) * CP

theorem calculate_profit :
  profit_calculation 850 33.85826771653544 ≈ 215.29 := 
sorry

end calculate_profit_l209_209026


namespace angle_Q_is_72_degrees_l209_209380

-- Definitions and conditions based on the given problem
variables {A B C D E F G H I J Q : Type} 
variables [Decagon : RegularDecagon A B C D E F G H I J]
variables [ExtendAH : AQ = Line(A, H)]
variables [ExtendEF : EQ = Line(E, F)]
variables [ParallelABGH : ∥ A B ∥ G H]
variables [ParallelEFCD : ∥ E F ∥ C D]

-- Main theorem
theorem angle_Q_is_72_degrees :
  ∠ Q = 72 :=
sorry

end angle_Q_is_72_degrees_l209_209380


namespace number_of_odd_factors_of_360_l209_209202

theorem number_of_odd_factors_of_360 : 
  let factors (n : ℕ) := {d : ℕ | d ∣ n}
  let odd (d : ℕ) := d % 2 = 1
  let is_factor (n : ℕ) (d : ℕ) := d ∣ n
  let number_of_odd_factors (n : ℕ) := (factors n).count odd
  in number_of_odd_factors 360 = 6 :=
begin
  sorry
end

end number_of_odd_factors_of_360_l209_209202


namespace odd_factors_of_360_l209_209189

theorem odd_factors_of_360 : ∃ n, n = 6 ∧ 
  ∀ (x : ℕ), (∃ (a b : ℕ), 360 = 2^a * 3^b * x ∧ ¬ even x ∧ 0 < x) → (∀ (m : ℕ), m ∣ x ↔ (m = 1 ∨ m = 3 ∨ m = 5 ∨ m = 9 ∨ m = 15 ∨ m = 45)) :=
by
  -- Proof
  sorry

end odd_factors_of_360_l209_209189


namespace coordinates_of_projection_l209_209142

theorem coordinates_of_projection (start_position : ℝ × ℝ)
  (circle_radius : ℝ) (center : ℝ × ℝ) (angle_AOB : ℝ) :
  start_position = (4, 0) →
  circle_radius = 4 →
  center = (0, 0) →
  ∃ B : ℝ × ℝ, B = (4 * Real.cos angle_AOB, 4 * Real.sin angle_AOB) ∧
  ∃ C : ℝ × ℝ, C = (B.1, 0) → C = (4 * Real.cos angle_AOB, 0) :=
begin
  sorry
end

end coordinates_of_projection_l209_209142


namespace zero_point_in_interval_l209_209817

def f (x : ℝ) : ℝ := Real.pi * x + Real.log x / Real.log 2

theorem zero_point_in_interval :
  ∃ c ∈ Set.Icc (1 / 4) (1 / 2), f c = 0 :=
sorry

end zero_point_in_interval_l209_209817


namespace complex_problem_l209_209968

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem complex_problem :
  (1 - z) * (1 - z^2) * (1 - z^3) * (1 - z^4) = 5 :=
by
  sorry

end complex_problem_l209_209968


namespace race_head_start_l209_209000

variable (vA vB L h : ℝ)
variable (hva_vb : vA = (16 / 15) * vB)

theorem race_head_start (hL_pos : L > 0) (hvB_pos : vB > 0) 
    (h_times_eq : (L / vA) = ((L - h) / vB)) : h = L / 16 :=
by
  sorry

end race_head_start_l209_209000


namespace sin_and_tan_of_angle_A_l209_209690

noncomputable def right_triangle_data (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
(A B : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] (angleC : Angle) := 
⟪AB, AC⟫ -> 5 ∈ ℝ -> 12 ∈ ℝ

theorem sin_and_tan_of_angle_A
  (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (angleC : A → B → C → ℝ) (AB AC BC : ℝ)
  (h1 : AB = 5)
  (h2 : AC = 12)
  (h3 : angleC A B C = π / 2) :
  sin (angle A) = 12 / 13 ∧ tan (angle A) = 12 / 5 :=
by
  sorry

end sin_and_tan_of_angle_A_l209_209690


namespace odd_factors_360_l209_209216

theorem odd_factors_360 : 
  let fac := (2^3) * (3^2) * (5^1) in
  fac = 360 → 
  num_factors (3^2 * 5^1) = 6 :=
sorry

end odd_factors_360_l209_209216


namespace fraction_of_AD_eq_BC_l209_209766

theorem fraction_of_AD_eq_BC (x y : ℝ) (B C D A : ℝ) 
  (h1 : B < C) 
  (h2 : C < D)
  (h3 : D < A) 
  (hBD : B < D)
  (hCD : C < D)
  (hAD : A = D)
  (hAB : A - B = 3 * (D - B)) 
  (hAC : A - C = 7 * (D - C))
  (hx_eq : x = 2 * y) 
  (hADx : A - D = 4 * x)
  (hADy : A - D = 8 * y)
  : (C - B) = 1/8 * (A - D) := 
sorry

end fraction_of_AD_eq_BC_l209_209766


namespace find_k_l209_209749

theorem find_k (k : ℝ) : 
  (∀ x y, (y = 4 * x + 2) → (y = k * x + 3) → (y = 6 ∧ x = 1)) → k = 3 :=
by
  intro h
  have h1 := h 1 6 rfl rfl
  sorry

end find_k_l209_209749


namespace find_z_l209_209845

theorem find_z
  (z : ℂ)
  (h1 : complex.arg (z^2 - 4) = 5 * Real.pi / 6)
  (h2 : complex.arg (z^2 + 4) = Real.pi / 3) :
  z = complex.i + (complex.sqrt 3) / 2 ∨ z = -(complex.i + (complex.sqrt 3) / 2) :=
by
  sorry

end find_z_l209_209845


namespace largest_possible_perimeter_l209_209526

theorem largest_possible_perimeter (y : ℤ) (hy1 : 3 ≤ y) (hy2 : y < 16) : 7 + 9 + y ≤ 31 := 
by
  sorry

end largest_possible_perimeter_l209_209526


namespace sum_of_reciprocal_transformed_roots_l209_209071

theorem sum_of_reciprocal_transformed_roots :
  ∀ (a b c : ℝ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    -1 < a ∧ a < 1 ∧
    -1 < b ∧ b < 1 ∧
    -1 < c ∧ c < 1 ∧
    (45 * a ^ 3 - 70 * a ^ 2 + 28 * a - 2 = 0) ∧
    (45 * b ^ 3 - 70 * b ^ 2 + 28 * b - 2 = 0) ∧
    (45 * c ^ 3 - 70 * c ^ 2 + 28 * c - 2 = 0)
  → (1 - a)⁻¹ + (1 - b)⁻¹ + (1 - c)⁻¹ = 13 / 9 := 
by 
  sorry

end sum_of_reciprocal_transformed_roots_l209_209071


namespace min_C_over_D_l209_209676

theorem min_C_over_D (x C D : ℝ) (h1 : x^2 + 1/x^2 = C) (h2 : x + 1/x = D) (hC_pos : 0 < C) (hD_pos : 0 < D) : 
  (∃ m : ℝ, m = 2 * Real.sqrt 2 ∧ ∀ y : ℝ, y = C / D → y ≥ m) :=
  sorry

end min_C_over_D_l209_209676


namespace binary_ternary_product_l209_209545

-- Definitions for numbers in binary and ternary
def binary_to_decimal : Nat := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0
def ternary_to_decimal : Nat := 1 * 3^2 + 0 * 3^1 + 2 * 3^0

-- Main theorem to prove
theorem binary_ternary_product : binary_to_decimal * ternary_to_decimal = 143 := by
  -- converting binary and ternary to decimal
  have h1 : binary_to_decimal = 13 := by
    simp [binary_to_decimal]
  have h2 : ternary_to_decimal = 11 := by
    simp [ternary_to_decimal]
  
  -- Calculation of the product
  calc
    binary_to_decimal * ternary_to_decimal = 13 * 11     := by rw [h1, h2]
                                        ... = 143 := by norm_num


end binary_ternary_product_l209_209545


namespace least_perimeter_of_triangle_l209_209290

/-- Given a triangle ∆XYZ with integer side lengths, cosines of angles X, Y, Z are given by
    cos X = 3/5, cos Y = 1/2, cos Z = -1/3. The task is to prove that the least possible perimeter
    of this triangle is 78. -/
theorem least_perimeter_of_triangle 
  (X Y Z : ℝ) 
  (hx : cos X = 3/5)
  (hy : cos Y = 1/2)
  (hz : cos Z = -1/3) : 
  ∃ (a b c : ℕ), a + b + c = 78 ∧ 
  sin X = 4/5 ∧ sin Y = real.sqrt 3 / 2 ∧ sin Z = 2 * real.sqrt 2 / 3 := 
sorry

end least_perimeter_of_triangle_l209_209290


namespace maria_apple_probability_l209_209353

/-- Maria has 10 apples: 6 are red and 4 are green. The probability that, if she chooses 
3 apples at random, exactly two of them are red is 1/2. -/
theorem maria_apple_probability :
  let total_apples := 10 in
  let red_apples := 6 in
  let green_apples := 4 in
  let total_ways := Nat.choose total_apples 3 in
  let successful_ways := Nat.choose red_apples 2 * Nat.choose green_apples 1 in
  (successful_ways : ℚ) / total_ways = 1 / 2 :=
by
  sorry

end maria_apple_probability_l209_209353


namespace problem_solution_l209_209927

def p (x n : ℕ) : ℕ := ∑ k in finset.range (n + 1), x^k * ∏ (i : ℕ) in finset.range (k + 1, n + 1), (1 - x^i)

theorem problem_solution :
  p 1993 1993 = 1 := by
  sorry

end problem_solution_l209_209927


namespace carrie_profit_l209_209065

def total_hours_worked (hours_per_day: ℕ) (days: ℕ): ℕ := hours_per_day * days
def total_earnings (hours_worked: ℕ) (hourly_wage: ℕ): ℕ := hours_worked * hourly_wage
def profit (total_earnings: ℕ) (cost_of_supplies: ℕ): ℕ := total_earnings - cost_of_supplies

theorem carrie_profit (hours_per_day: ℕ) (days: ℕ) (hourly_wage: ℕ) (cost_of_supplies: ℕ): 
    hours_per_day = 2 → days = 4 → hourly_wage = 22 → cost_of_supplies = 54 → 
    profit (total_earnings (total_hours_worked hours_per_day days) hourly_wage) cost_of_supplies = 122 := 
by
    intros hpd d hw cos
    sorry

end carrie_profit_l209_209065


namespace dot_product_parallel_vectors_l209_209660

variable (x : ℝ)
def a := (1, 2 : ℝ × ℝ)
def b := (x, -4 : ℝ × ℝ)
def parallel (u v : ℝ × ℝ) : Prop := ∃ (λ : ℝ), v = (λ * u.1, λ * u.2)

theorem dot_product_parallel_vectors (h : parallel a b) : a.1 * b.1 + a.2 * b.2 = -10 := by
  sorry

end dot_product_parallel_vectors_l209_209660


namespace parabola_distance_relation_l209_209128

theorem parabola_distance_relation {n : ℝ} {x₁ x₂ y₁ y₂ : ℝ}
  (h₁ : y₁ = x₁^2 - 4 * x₁ + n)
  (h₂ : y₂ = x₂^2 - 4 * x₂ + n)
  (h : y₁ > y₂) :
  |x₁ - 2| > |x₂ - 2| := 
sorry

end parabola_distance_relation_l209_209128


namespace problem_U_complement_eq_l209_209987

universe u
variable {α : Type u} [DecidableEq α]
variable (U M : Set α)

theorem problem_U_complement_eq
  (hU : U = ({1, 2, 3, 4, 5} : Set ℕ))
  (hM : U \ M = ({1, 3} : Set ℕ)) : 
  (2 : ℕ) ∈ M := 
by
  sorry

end problem_U_complement_eq_l209_209987


namespace q_to_the_fourth_l209_209632

noncomputable def a_n (n : ℕ) := sorry -- Placeholder for the geometric sequence

variables (q : ℝ) -- Common ratio of the geometric sequence
variables (a_3 a_4 a_6 a_7 : ℝ)

-- Conditions from the problem
axiom h1 : ∀ n : ℕ, a_n (n + 1) = q * a_n n -- Definition of a geometric sequence
axiom h2 : a_n 3 + a_n 7 = 5 -- Given condition a_3 + a_7 = 5
axiom h3 : a_n 6 * a_n 4 = 6 -- Given condition a_6 * a_4 = 6

-- The theorem to be proven
theorem q_to_the_fourth : q^4 = (3/2) := by
  sorry

end q_to_the_fourth_l209_209632


namespace probability_of_selecting_Zhang_l209_209106

-- Define the set of surgeons
inductive Surgeon
| Zhang | Wang | Li | Liu

open Surgeon

-- Define a pair of surgeons as a set of two elements
def pair (a b: Surgeon) : Set Surgeon := {a, b}

-- List all possible pairs of selecting 2 out of 4 surgeons
def all_pairs : List (Set Surgeon) :=
  [pair Zhang Wang, pair Zhang Li, pair Zhang Liu, pair Wang Li, pair Wang Liu, pair Li Liu]

-- Filter pairs containing Dr. Zhang
def pairs_with_Zhang : List (Set Surgeon) :=
  all_pairs.filter (λ s, Zhang ∈ s)

-- Calculate the probability
def prob_Zhang_selected : ℚ :=
  pairs_with_Zhang.length / all_pairs.length

theorem probability_of_selecting_Zhang :
  prob_Zhang_selected = 1 / 2 :=
by
  -- Simplifying the proof
  sorry

end probability_of_selecting_Zhang_l209_209106


namespace Karlson_max_candies_l209_209431

theorem Karlson_max_candies :
  let n := 27
  in ∑ i in Finset.range (n - 1), (i + 1) = 351 :=
by
  sorry

end Karlson_max_candies_l209_209431


namespace scientific_notation_of_3395000_l209_209483

theorem scientific_notation_of_3395000 :
  3395000 = 3.395 * 10^6 :=
sorry

end scientific_notation_of_3395000_l209_209483


namespace phase_shift_of_cosine_l209_209924

theorem phase_shift_of_cosine (x : ℝ) : 
  (\exists x' : ℝ, 2 * x' - π / 3 = 0 ∧ x' = x) → x = π / 6 :=
by
  intro h
  cases h with x' hx
  cases hx with h₁ h₂
  rw [h₂] at h₁
  linarith

end phase_shift_of_cosine_l209_209924


namespace algebraic_expression_transition_l209_209828

theorem algebraic_expression_transition (n : ℕ) (h1 : n > 2) :
  let lhs_n := ∑ i in finset.range(n+1) \ finset.range(n+1), (1 / (n + 1 + i ))
  let lhs_n1 := ∑ i in finset.range(n+2) \ finset.range(n+1), (1 / (n + 1 + 1 + i))
  lhs_n1 - lhs_n = (1 / (2 * n + 1) + 1 / (2 * (n + 1) - 1 / (n + 1))) ∧ (lhs_n > 13 / 24) :=
sorry

end algebraic_expression_transition_l209_209828


namespace cos_six_theta_constants_squared_sum_l209_209820

theorem cos_six_theta_constants_squared_sum :
  ∃ (b_1 b_2 b_3 b_4 b_5 b_6 : ℝ),
  (∀ (θ : ℝ), cos θ ^ 6 = b_1 * cos θ + b_2 * cos (2 * θ) + b_3 * cos (3 * θ) + b_4 * cos (4 * θ) + b_5 * cos (5 * θ) + b_6 * cos (6 * θ)) ∧
  b_1^2 + b_2^2 + b_3^2 + b_4^2 + b_5^2 + b_6^2 = 131 / 512 :=
sorry

end cos_six_theta_constants_squared_sum_l209_209820


namespace wall_area_l209_209887

theorem wall_area (width : ℝ) (height : ℝ) (h1 : width = 2) (h2 : height = 4) : width * height = 8 := by
  sorry

end wall_area_l209_209887


namespace compound_interest_l209_209396

theorem compound_interest 
  (P : ℝ) (r : ℝ) (t : ℕ) : P = 500 → r = 0.02 → t = 3 → (P * (1 + r)^t) - P = 30.60 :=
by
  intros P_invest rate years
  simp [P_invest, rate, years]
  sorry

end compound_interest_l209_209396


namespace seq_geom_prog_l209_209334

theorem seq_geom_prog (a : ℕ → ℝ) (b : ℝ) (h_pos_b : 0 < b)
  (h_pos_a : ∀ n, 0 < a n)
  (h_recurrence : ∀ n, a (n + 2) = (b + 1) * a n * a (n + 1)) :
  (∃ r, ∀ n, a (n + 1) = r * a n) ↔ a 0 = a 1 :=
sorry

end seq_geom_prog_l209_209334


namespace number_of_incorrect_propositions_l209_209083

-- Definitions based on conditions
def prop1 := ∀ x : ℝ, x^2 - x + 1 ≤ 0
def prop2 := ∀ p q : Prop, ¬(p ∨ q) → ¬p ∧ ¬q
def prop3 (m n : ℝ) := mn > 0 → (∃ m n : ℝ, mn > 0 → mx^2 + ny^2 = 1)

-- The main statement
theorem number_of_incorrect_propositions : 
  (¬ (prop1 ↔ ¬(∃ x : ℝ, x^2 - x + 1 > 0))) + 
  (¬ (prop2 _ _ (iff.mpr by exact _))) +
  (∃ m n : ℝ, ¬(prop3 m n)) = 1 :=
sorry

end number_of_incorrect_propositions_l209_209083


namespace minimum_set_A_size_l209_209107

open Set Function

noncomputable def prime_diff_function (f : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → Prime (nat.abs (i - j)) → f i ≠ f j

theorem minimum_set_A_size : ∃ (A : Finset ℕ), ∀ (f : ℕ → ℕ),
  prime_diff_function f → ∀ (x : ℕ), x ∈ A → (x < 4) ∧ (∀ y ∈ A, x ≠ y → Prime (nat.abs (x - y))) :=
begin
  sorry
end

end minimum_set_A_size_l209_209107


namespace transform_f_l209_209609

variables (f : ℝ → ℝ)

theorem transform_f (H : ∀ x, f(x - 1) = x^2 + 4 * x - 5) :
  ∀ x, f(x + 1) = x^2 + 8 * x + 7 :=
sorry

end transform_f_l209_209609


namespace real_part_of_complex_eq_2_l209_209123

-- Definition of the problem
def complex_satisfying_equation (z : ℂ) : Prop :=
  z + 2 * conj z = 6 + complex.i

-- Proving the real part of z in given conditions
theorem real_part_of_complex_eq_2 (z : ℂ) (h : complex_satisfying_equation z) : z.re = 2 :=
by {
  sorry
}

end real_part_of_complex_eq_2_l209_209123


namespace angle_Q_of_regular_decagon_l209_209383

noncomputable def decagon := { 
  sides : ℕ // sides = 10 
}

def interior_angle_sum (n : ℕ) : ℕ := 180 * (n - 2)

def regular_polygon_interior_angle (n : ℕ) : ℕ := (interior_angle_sum n) / n

theorem angle_Q_of_regular_decagon (d : decagon) : 
  let n := d.sides in
  let interior_angle := regular_polygon_interior_angle n in
  let exterior_angle := 180 - interior_angle in
  let angle_BFQ := 360 - 2 * interior_angle in
  let angle_Q := 360 - (2 * exterior_angle + angle_BFQ) in
  angle_Q = 216 := 
by {
  have h1 : n = 10 := d.2,
  rw h1,
  have H_inter : interior_angle = (180 * 8 / 10) := rfl,
  have H_ext : exterior_angle = 180 - (144) := by rw H_inter; norm_num,
  have H_BFQ : angle_BFQ = 360 - 2 * 144 := by rw H_inter; norm_num,
  norm_num,
  sorry
}

end angle_Q_of_regular_decagon_l209_209383


namespace find_x_l209_209257

theorem find_x (x : ℝ) : (3^(x - 4) = 9^3) → x = 10 :=
by
  sorry

end find_x_l209_209257


namespace net_change_in_salary_l209_209423

variable (S : ℝ)

theorem net_change_in_salary : 
  let increased_salary := S + (0.1 * S)
  let final_salary := increased_salary - (0.1 * increased_salary)
  final_salary - S = -0.01 * S :=
by
  sorry

end net_change_in_salary_l209_209423


namespace inscribed_circle_radius_l209_209103

-- Given definitions
def side_PQ : ℝ := 26
def side_PR : ℝ := 10
def side_QR : ℝ := 18

-- Prove statement
theorem inscribed_circle_radius :
  let s := (side_PQ + side_PR + side_QR) / 2 in
  let K := Real.sqrt (s * (s - side_PQ) * (s - side_PR) * (s - side_QR)) in
  let r := K / s in
  r = Real.sqrt 17 :=
by
  sorry

end inscribed_circle_radius_l209_209103


namespace perpendicular_lines_l209_209025

-- Definitions for problem
variables {A B C D O : Type} [ConvexQuadrilateral A B C D O]
variables {H_a H_b K_a K_b : Point}
-- Assume given conditions:
variables (H_a_orth_center_TRIANGLE_AOB : Orthocenter A O B H_a)
variables (H_b_orth_center_TRIANGLE_COD : Orthocenter C O D H_b)
variables (K_a_mid_point_BC : Midpoint B C K_a)
variables (K_b_mid_point_AD : Midpoint A D K_b)

-- We need to prove that H_aH_b is perpendicular to K_aK_b
theorem perpendicular_lines :
  Perpendicular (Line_segment H_a H_b) (Line_segment K_a K_b) :=
  sorry

end perpendicular_lines_l209_209025


namespace jelly_bean_matching_probability_l209_209875

noncomputable def abe_jelly_beans := [2, 2, 1] -- green, red, blue
noncomputable def clara_jelly_beans := [3, 2, 2, 1] -- green, yellow, red, blue

theorem jelly_bean_matching_probability :
  (2 / 5) * (3 / 8) + (2 / 5) * (2 / 8) + (1 / 5) * (1 / 8) = 11 / 40 :=
by
  have abe_total : ℚ := 2 + 2 + 1 -- total jelly beans Abe holds
  have clara_total : ℚ := 3 + 2 + 2 + 1 -- total jelly beans Clara holds
  have prob_green : ℚ := (abe_jelly_beans.head! / abe_total) * (clara_jelly_beans.head! / clara_total)
  have prob_red : ℚ := (abe_jelly_beans.nth! 1 / abe_total) * (clara_jelly_beans.nth! 2 / clara_total )
  have prob_blue : ℚ := (abe_jelly_beans.tail!.tail!.head! / abe_total) * (clara_jelly_beans.tail!.tail!.tail!.head! / clara_total)
  have h1 : prob_green = (2 / 5) * (3 / 8) := rfl
  have h2 : prob_red = (2 / 5) * (2 / 8) := rfl
  have h3 : prob_blue = (1 / 5) * (1 / 8) := rfl
  have sum_of_probs : prob_green + prob_red + prob_blue = 11 / 40 :=
    by sorry -- Detailed steps for rational addition
  exact sum_of_probs
sorry

end jelly_bean_matching_probability_l209_209875


namespace how_many_years_later_will_tom_be_twice_tim_l209_209439

-- Conditions
def toms_age := 15
def total_age := 21
def tims_age := total_age - toms_age

-- Define the problem statement
theorem how_many_years_later_will_tom_be_twice_tim (x : ℕ) 
  (h1 : toms_age + tims_age = total_age) 
  (h2 : toms_age = 15) 
  (h3 : ∀ y : ℕ, toms_age + y = 2 * (tims_age + y) ↔ y = x) : 
  x = 3 
:= sorry

end how_many_years_later_will_tom_be_twice_tim_l209_209439


namespace Radhika_total_games_l209_209775

theorem Radhika_total_games :
  let christmas_games := 12
  let birthday_games := 8
  let family_gathering_games := 5
  let total_gifted_games := christmas_games + birthday_games + family_gathering_games
  let initial_games := floor (2 / 3 * total_gifted_games)
  let games_after_gifts := initial_games + total_gifted_games
  let additional_purchased_games := 6
  let total_games_after_purchase := games_after_gifts + additional_purchased_games
  total_games_after_purchase = 47 :=
by
  sorry

end Radhika_total_games_l209_209775


namespace numeral_150th_decimal_place_l209_209838

theorem numeral_150th_decimal_place (n : ℕ) (h : n = 150) : 
  let decimal_representation := "63".cycle;
  let index := n % 2 in
  decimal_representation.get index = '3' :=
by
  sorry

end numeral_150th_decimal_place_l209_209838


namespace carrie_profit_l209_209067

def hours_per_day : ℕ := 2
def days_worked : ℕ := 4
def hourly_rate : ℕ := 22
def cost_of_supplies : ℕ := 54
def total_hours_worked : ℕ := hours_per_day * days_worked
def total_payment : ℕ := hourly_rate * total_hours_worked
def profit : ℕ := total_payment - cost_of_supplies

theorem carrie_profit : profit = 122 := by
  sorry

end carrie_profit_l209_209067
