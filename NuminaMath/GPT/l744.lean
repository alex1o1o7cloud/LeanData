import Mathlib

namespace find_bc_find_area_l744_744240

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Condition 1: From the given problem
variable (h1 : (b^2 + c^2 - a^2) / (cos A) = 2)

-- Condition 2: From the given problem
variable (h2 : (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1)

theorem find_bc : bc = 1 :=
sorry

theorem find_area (area_abc : ℝ) : area_abc = (sqrt 3) / 4 :=
sorry

end find_bc_find_area_l744_744240


namespace more_knights_than_liars_l744_744295

-- Define the two tribes
inductive Tribe
| Knight : Tribe
| Liar : Tribe

-- Number of Knights and Liars
variables (knights liars : ℕ)

-- Friendship relation
variable (friends : ℕ → ℕ → Bool)
-- friends i j indicates whether i-th and j-th islander are friends

-- Predicate for "more than half of my friends are fellow tribesmen"
def more_than_half_friends_fellow_tribesmen (i : ℕ) : Bool :=
  let friends_count := (List.range (knights + liars)).filter (friends i)
  let fellow_tribesmen_count := friends_count.filter (λ j => j < knights)
  if i < knights then
    fellow_tribesmen_count.length > friends_count.length / 2
  else
    (friends_count.length - fellow_tribesmen_count.length) > friends_count.length / 2

theorem more_knights_than_liars (H : ∀ (i : ℕ), i < knights + liars → more_than_half_friends_fellow_tribesmen i) : 
  knights > liars :=
sorry

end more_knights_than_liars_l744_744295


namespace smallest_pos_int_b_for_factorization_l744_744863

theorem smallest_pos_int_b_for_factorization :
  ∃ b : ℤ, 0 < b ∧ ∀ (x : ℤ), ∃ r s : ℤ, r * s = 4032 ∧ r + s = b ∧ x^2 + b * x + 4032 = (x + r) * (x + s) ∧
    (∀ b' : ℤ, 0 < b' → b' ≠ b → ∃ rr ss : ℤ, rr * ss = 4032 ∧ rr + ss = b' ∧ x^2 + b' * x + 4032 = (x + rr) * (x + ss) → b < b') := 
sorry

end smallest_pos_int_b_for_factorization_l744_744863


namespace green_balls_l744_744706

variable (B G : ℕ)

theorem green_balls (h1 : B = 20) (h2 : B / G = 5 / 3) : G = 12 :=
by
  -- Proof goes here
  sorry

end green_balls_l744_744706


namespace area_smaller_octagon_l744_744590

theorem area_smaller_octagon (apothem : ℝ) (h : apothem = 3) : ∃ A, 
  A = 36 / real.sqrt (2 + real.sqrt 2) ∧ 
  is_regular_octagon (P : ℕ → point) ∧
  apothem_of_octagon P = apothem → 
  is_regular_octagon (Q : ℕ → point) ∧
  side_length_of_octagon Q = (side_length_of_octagon P) / 2 ∧
  area_of_octagon Q = A :=
begin
  sorry
end

end area_smaller_octagon_l744_744590


namespace minimal_period_f_intervals_of_increase_f_max_value_f_min_value_f_l744_744137

-- Definition of the function f
def f (x : ℝ) : ℝ := sqrt 2 * sin (2 * x + π / 4) + 2

-- Proving the minimal positive period of f(x) is π
theorem minimal_period_f : ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = π := sorry

-- Proving the intervals of monotonic increase of f(x) are [-3π/8 + kπ, π/8 + kπ] where k is an integer
theorem intervals_of_increase_f : ∀ k : ℤ, ∀ x ∈ Icc (-3 * π / 8 + k * π) (π / 8 + k * π), ∀ y ∈ Icc (-3 * π / 8 + k * π) (π / 8 + k * π), x < y → f x < f y := sorry

-- Proving the maximum value of f(x) on [0, π/2] is 2 + sqrt(2)
theorem max_value_f : ∃ x ∈ Icc 0 (π / 2), f x = 2 + sqrt 2 := sorry

-- Proving the minimum value of f(x) on [0, π/2] is 1
theorem min_value_f : ∃ x ∈ Icc 0 (π / 2), f x = 1 := sorry

end minimal_period_f_intervals_of_increase_f_max_value_f_min_value_f_l744_744137


namespace integer_solutions_equation_l744_744158

theorem integer_solutions_equation : 
  (∃ x y : ℤ, (1 / (2022 : ℚ) = 1 / (x : ℚ) + 1 / (y : ℚ))) → 
  ∃! (n : ℕ), n = 53 :=
by
  sorry

end integer_solutions_equation_l744_744158


namespace anna_tower_construction_l744_744818

theorem anna_tower_construction :
  let discs := {1, 2, 3, 4, 5}
  (∃ S ⊆ discs, S.card = 3 ∧ (∀ x ∈ S, ∀ y ∈ S, x < y → (∃ f : ℕ → ℕ, 
  (f 0 = x ∧ f 1 = y ∧ S =↑ (finset.image f (finset.range 3)) ∧ 0 < 1))))
  → (@finset.card _ (finset.powerset_len 3 discs) = 10) :=
begin
  intro h,
  -- Since this is just a problem statement, we end with sorry to skip the proof.
  sorry
end

end anna_tower_construction_l744_744818


namespace remainder_17_pow_45_div_5_l744_744747

theorem remainder_17_pow_45_div_5 : (17 ^ 45) % 5 = 2 :=
by
  -- proof goes here
  sorry

end remainder_17_pow_45_div_5_l744_744747


namespace cubes_sum_l744_744682

theorem cubes_sum (a b c : ℝ) (h1 : a + b + c = 6) (h2 : a * b + a * c + b * c = 5) (h3 : a * b * c = -12) : 
  a^3 + b^3 + c^3 = 90 :=
begin
  sorry
end

end cubes_sum_l744_744682


namespace find_f_expression_l744_744527

theorem find_f_expression (f : ℝ → ℝ) :
  (∀ x : ℝ, f (x - 1) = x^2 + 1) →
  (∀ x : ℝ, f(x) = (x + 1)^2 + 1) :=
by
  intro h
  sorry

end find_f_expression_l744_744527


namespace altitude_inequality_l744_744983

open Classical
noncomputable theory

variables {A B C P : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]

def altitude_length (h_a h_b h_c : ℝ) (ABC : Triangle A B C) := ∀ 
  (h_a > 0) 
  (h_b > 0) 
  (h_c > 0)

def point_inside_triangle (P : A → B → C → Type) (ABC : Triangle A B C) := ∃ 
  (P_inside : P ABC)

theorem altitude_inequality (h_a h_b h_c : ℝ) (ABC : Triangle A B C) (P : A → B → C → Type) 
  (ha : altitude_length h_a h_b h_c ABC)
  (P_in_triangle : point_inside_triangle P ABC) :
  (PA / (h_b + h_c)) + (PB / (h_a + h_c)) + (PC / (h_a + h_b)) ≥ 1 :=
sorry

end altitude_inequality_l744_744983


namespace kylie_beads_used_l744_744633

noncomputable def total_beads (necklaces : ℕ) (bracelets : ℕ) (earrings : ℕ) (anklets : ℕ) (rings : ℕ) : ℕ :=
  20 * necklaces + 10 * bracelets + 5 * earrings + 8 * anklets + 7 * rings

theorem kylie_beads_used :
  let necklaces := 10 + 2 in
  let bracelets := 5 in
  let earrings := 3 in
  let anklets := 4 in
  let rings := 6 in
  total_beads necklaces bracelets earrings anklets rings = 379 :=
begin
  sorry
end

end kylie_beads_used_l744_744633


namespace sum_second_left_right_l744_744813

theorem sum_second_left_right : 
  let odds := [1, 3, 5, 7, 9, 11, 13, 15] in
  odds[1] + odds[6] = 16 :=
by
  sorry

end sum_second_left_right_l744_744813


namespace volume_of_solid_l744_744805

-- Define the basic properties of the solid
def solid (T : Type) [convex T] : Prop :=
  (∃ (tri_faces : Finset {t : Triangle // t.side_length = 1}),
    tri_faces.card = 8) ∧
  (∃ (sq_faces : Finset {s : Square}),
    sq_faces.card = 2 ∧
    ∀ s1 s2 ∈ sq_faces, s1.plane ∥ s2.plane)

theorem volume_of_solid (T : Type) [solid T] : 
  volume T = real.sqrt (real.sqrt 2) * (1 + real.sqrt 2) / 3 :=
sorry

end volume_of_solid_l744_744805


namespace smallest_n_with_251_in_decimal_l744_744691

theorem smallest_n_with_251_in_decimal :
  ∃ (n m : ℕ), n > 0 ∧ m > 0 ∧ Nat.coprime m n ∧ m < n ∧
  (∃ (d : ℚ), (m : ℚ) / n = d ∧
    (d.to_digits 10).indexOf [2, 5, 1] ≠ none) ∧
  n = 127 :=
by 
  sorry

end smallest_n_with_251_in_decimal_l744_744691


namespace total_area_is_16pi_over_3_l744_744852

noncomputable def area_of_circles (n : ℕ) : ℝ :=
  let rn := 2^(2 - n)
  π * (rn)^2

noncomputable def total_area_of_circles : ℝ := 
  ∑' n, area_of_circles n

theorem total_area_is_16pi_over_3 : total_area_of_circles = (16 * π) / 3 := 
  sorry

end total_area_is_16pi_over_3_l744_744852


namespace bc_is_one_area_of_triangle_l744_744235

-- Define a triangle with sides a, b, and c and corresponding angles A, B, and C.
structure Triangle :=
(a b c A B C : ℝ)

-- Assume the Law of Cosines condition is given.
axiom Law_of_Cosines_condition (T : Triangle) : (T.b^2 + T.c^2 - T.a^2) / (Real.cos T.A) = 2

-- Prove that bc = 1.
theorem bc_is_one (T : Triangle) (h : Law_of_Cosines_condition T) : T.b * T.c = 1 := 
by {
    sorry
}

-- Assume the given ratio condition.
axiom ratio_condition (T : Triangle) :
  (T.a * Real.cos T.B - T.b * Real.cos T.A) / (T.a * Real.cos T.B + T.b * Real.cos T.A)
  - (T.b / T.c) = 1

-- Prove that the area of the triangle is sqrt(3)/4.
theorem area_of_triangle (T : Triangle) (h1 : Law_of_Cosines_condition T) (h2 : ratio_condition T) :
  let S := 0.5 * T.b * T.c * Real.sin T.A
  in S = Real.sqrt 3 / 4 :=
by {
    sorry
}

end bc_is_one_area_of_triangle_l744_744235


namespace proper_subsets_count_of_setA_l744_744717

open Set

def setA : Set ℤ := {-1, 0, 1}

theorem proper_subsets_count_of_setA : (card {s : Set ℤ | s ⊆ setA ∧ s ≠ setA} = 7) := by
  sorry

end proper_subsets_count_of_setA_l744_744717


namespace triangle_angle_problems_l744_744340

theorem triangle_angle_problems 
  (D : Point)
  (A B C : Triangle)
  (h1 : AB = CD)
  (h2 : ∠ ABC = 100)
  (h3 : ∠ DCB = 40) :
  ∠ BDC = 40 ∧ ∠ ACD = 10 :=
begin
  sorry
end

end triangle_angle_problems_l744_744340


namespace probability_stack_36ft_tall_l744_744854

-- Definitions of the conditions
def crateDimensions : List ℕ := [2, 5, 7]
def numCrates : ℕ := 8
def targetHeight : ℕ := 36

-- Theorem that needs to be proved
theorem probability_stack_36ft_tall : 
  (let totalWays := 3^numCrates in 
   let validWays := 98 in
   validWays / totalWays = (98:ℕ) / 6561) :=
by sorry

end probability_stack_36ft_tall_l744_744854


namespace line_intersects_circle_l744_744754

open Real

theorem line_intersects_circle (a : ℝ) (h : a > 0) :
  let line := λ (x y : ℝ), x + a^2 * y - a = 0
  let circle := λ (x y : ℝ), (x - a)^2 + (y - (1 / a))^2 = 1
  ∃ x y : ℝ, line x y ∧ circle x y :=
by
  sorry

end line_intersects_circle_l744_744754


namespace smallest_y_l744_744749

theorem smallest_y (y : ℤ) (h : y < 3 * y - 15) : y = 8 :=
  sorry

end smallest_y_l744_744749


namespace coupon1_better_than_coupon2_and_coupon3_at_219_95_l744_744033
noncomputable def coupon_discounts (x : ℝ) : ℝ × ℝ × ℝ :=
  (0.1 * x, 20, 0.18 * (x - 100))

theorem coupon1_better_than_coupon2_and_coupon3_at_219_95 :
  let x := 219.95 in
  let (discount1, discount2, discount3) := coupon_discounts x in
  discount1 > discount2 ∧ discount1 > discount3 :=
by
  sorry

end coupon1_better_than_coupon2_and_coupon3_at_219_95_l744_744033


namespace sum_of_products_eq_35910_l744_744406

theorem sum_of_products_eq_35910 :
  (∑ n in Finset.range 18, (n + 1) * (n + 2) * (n + 3)) = 35910 :=
by
  sorry

end sum_of_products_eq_35910_l744_744406


namespace sum_of_roots_l744_744971

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - 2016 * x + 2015

theorem sum_of_roots (a b c : ℝ) (h1 : f a = c) (h2 : f b = c) (h3 : a ≠ b) :
  a + b = 2016 :=
by
  sorry

end sum_of_roots_l744_744971


namespace frequency_of_zero_in_3021004201_l744_744620

def digit_frequency (n : Nat) (d : Nat) :  Rat :=
  let digits := n.digits 10
  let count_d := digits.count d
  (count_d : Rat) / digits.length

theorem frequency_of_zero_in_3021004201 : 
  digit_frequency 3021004201 0 = 0.4 := 
by 
  sorry

end frequency_of_zero_in_3021004201_l744_744620


namespace manager_salary_is_3600_l744_744010

noncomputable def manager_salary (M : ℕ) : ℕ :=
  let total_salary_20 := 20 * 1500
  let new_average_salary := 1600
  let total_salary_21 := 21 * new_average_salary
  total_salary_21 - total_salary_20

theorem manager_salary_is_3600 : manager_salary 3600 = 3600 := by
  sorry

end manager_salary_is_3600_l744_744010


namespace inscribed_sphere_radius_l744_744859

theorem inscribed_sphere_radius (a α : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) :
  ∃ (ρ : ℝ), ρ = a * (1 - Real.cos α) / (2 * Real.sqrt (1 + Real.cos α) * (1 + Real.sqrt (- Real.cos α))) :=
  sorry

end inscribed_sphere_radius_l744_744859


namespace total_distance_between_foci_l744_744816

-- Definition of the problem's given conditions
def ellipse_center : (ℝ × ℝ) := (2, 3)
def semi_major_axis : ℝ := 8
def semi_minor_axis : ℝ := 5

-- Distance calculation between the foci
theorem total_distance_between_foci : 2 * real.sqrt (semi_major_axis^2 - semi_minor_axis^2) = 2 * real.sqrt 39 :=
by
  sorry

end total_distance_between_foci_l744_744816


namespace relationship_between_p_and_q_l744_744925

variable (x y : ℝ)

def p := x * y ≥ 0
def q := |x + y| = |x| + |y|

theorem relationship_between_p_and_q : (p x y ↔ q x y) :=
sorry

end relationship_between_p_and_q_l744_744925


namespace count_odd_quadruples_l744_744636

theorem count_odd_quadruples :
  let S := {0, 1, 2, 3, 4}
  ∃ n, n = 16 ∧ (n = Finset.card (Finset.filter (λ (t : ℤ × ℤ × ℤ × ℤ),
    let (a, b, c, d) := t in
    (a * d - b * c + a) % 2 = 1) (Set.toFinset (Set.univ.prod (Set.univ.prod (Set.univ.prod S S) S))))) :=
by
  sorry

end count_odd_quadruples_l744_744636


namespace area_of_tangency_triangle_l744_744836

-- Define the radii of the circles
def r1 : ℝ := 2
def r2 : ℝ := 3
def r3 : ℝ := 4

-- Define the distances between centers based on mutual tangency
def d12 : ℝ := r1 + r2
def d23 : ℝ := r2 + r3
def d13 : ℝ := r1 + r3

-- Calculate the semi-perimeter and area using Heron's formula
def s := (d12 + d23 + d13) / 2
def area_triangle := Real.sqrt (s * (s - d12) * (s - d23) * (s - d13))

-- Prove the area of the triangle
theorem area_of_tangency_triangle : area_triangle = 6 * Real.sqrt 6 :=
sorry

end area_of_tangency_triangle_l744_744836


namespace largest_square_l744_744098

def sticks_side1 : List ℕ := [4, 4, 2, 3]
def sticks_side2 : List ℕ := [4, 4, 3, 1, 1]
def sticks_side3 : List ℕ := [4, 3, 3, 2, 1]
def sticks_side4 : List ℕ := [3, 3, 3, 2, 2]

def sum_of_sticks (sticks : List ℕ) : ℕ := sticks.foldl (· + ·) 0

theorem largest_square (h1 : sum_of_sticks sticks_side1 = 13)
                      (h2 : sum_of_sticks sticks_side2 = 13)
                      (h3 : sum_of_sticks sticks_side3 = 13)
                      (h4 : sum_of_sticks sticks_side4 = 13) :
  13 = 13 := by
  sorry

end largest_square_l744_744098


namespace pure_imaginary_number_solution_real_quotient_solution_l744_744125

-- Define the problem for the first part
theorem pure_imaginary_number_solution :
  ∀ (z : ℂ), (∃ a : ℝ, z = a * complex.I) → |z - 1| = |z - 1 + complex.I| → z = complex.I ∨ z = -complex.I :=
by
  sorry

-- Define the problem for the second part
theorem real_quotient_solution :
  ∀ (z : ℂ), (z + 10 / z).im = 0 → 1 ≤ (z + 10 / z).re ∧ (z + 10 / z).re ≤ 6 →
  z = 1 + 3 * complex.I ∨ z = 3 + complex.I ∨ z = 3 - complex.I :=
by
  sorry

end pure_imaginary_number_solution_real_quotient_solution_l744_744125


namespace greatest_divisor_with_sum_of_digits_four_l744_744177

/-- Define the given numbers -/
def a := 4665
def b := 6905

/-- Define the sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Define the greatest number n that divides both a and b, leaving the same remainder and having a sum of digits equal to 4 -/
theorem greatest_divisor_with_sum_of_digits_four :
  ∃ (n : ℕ), (∀ (d : ℕ), (d ∣ a - b ∧ sum_of_digits d = 4) → d ≤ n) ∧ (n ∣ a - b) ∧ (sum_of_digits n = 4) ∧ n = 40 := sorry

end greatest_divisor_with_sum_of_digits_four_l744_744177


namespace planes_perpendicular_from_skew_lines_l744_744888

universe u

variables {V : Type u} [InnerProductSpace ℝ V]
variables {m n : V} {α β : Submodule ℝ V}

-- Two vectors are perpendicular skew lines if they are perpendicular and not parallel
def perpendicular_skew (m n : V) : Prop :=
  InnerProductSpace.orthogonal m n ∧ (∀ k : ℝ, m ≠ k • n)

-- m and n are perpendicular to planes α and β respectively
def line_perp_to_plane (m : V) (α : Submodule ℝ V) : Prop :=
  ∀ x ∈ α, InnerProductSpace.orthogonal m x

-- α and β are perpendicular planes
def planes_perpendicular (α β : Submodule ℝ V) : Prop :=
  ∀ x ∈ α, ∀ y ∈ β, InnerProductSpace.orthogonal x y

-- Main theorem statement
theorem planes_perpendicular_from_skew_lines (h_skew : perpendicular_skew m n)
    (hm_perp : line_perp_to_plane m α) (hn_perp : line_perp_to_plane n β) :
  planes_perpendicular α β :=
sorry

end planes_perpendicular_from_skew_lines_l744_744888


namespace range_of_a_l744_744941

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 9 * x

-- Define the conditions and the statement to prove
theorem range_of_a (a : ℝ) (ha : a ≥ sqrt 2 ∧ a ≤ 4)
  (hmax : ∃ x ∈ set.Ioo (3 - a^2) a, ∀ y ∈ set.Ioo (3 - a^2) a, f x ≥ f y) : (sqrt 2 < a ∧ a ≤ 4) :=
sorry

end range_of_a_l744_744941


namespace discount_percentage_is_40_l744_744794

-- Definitions based on the conditions
def CP : ℝ := 100  -- given for simplicity
def markup : ℝ := 0.75
def profit : ℝ := 0.05

-- Derived quantities
def MP : ℝ := CP * (1 + markup)
def SP : ℝ := CP * (1 + profit)
def discount : ℝ := MP - SP
def discount_pct : ℝ := (discount / MP) * 100

-- Statement to prove
theorem discount_percentage_is_40 :
  discount_pct = 40 :=
by
  -- We are assuming all conditions are correct and just focusing on the structure
  -- Proof is not given as per instruction, only the statement is required.
  sorry

end discount_percentage_is_40_l744_744794


namespace minimum_value_of_f_l744_744220

noncomputable def a : ℕ+ → ℕ
| 1 := 2
| (n+1) := a n + 2

noncomputable def S : ℕ+ → ℕ
| 1 := a 1
| (n+1) := S n + a (n + 1)

noncomputable def f (n : ℕ+) : ℚ :=
(S n + 60) / (n + 1)

theorem minimum_value_of_f :
  (∃ (n : ℕ+), f n = 29 / 2) := sorry

end minimum_value_of_f_l744_744220


namespace percentage_decrease_in_denominator_l744_744944

theorem percentage_decrease_in_denominator :
  ∀ (N D : ℝ), (N / D = 5 / 7) →
  (let N' := 1.20 * N in
   ∀ (x : ℝ), (N' / (D * (1 - x / 100)) = 20 / 21) → x = 10) :=
by
  intros N D h1
  let N' := 1.20 * N
  intros x h2
  sorry

end percentage_decrease_in_denominator_l744_744944


namespace smallest_integer_y_l744_744751

theorem smallest_integer_y (y : ℤ) (h: y < 3 * y - 15) : y ≥ 8 :=
by sorry

end smallest_integer_y_l744_744751


namespace geologists_probability_l744_744615

theorem geologists_probability :
  let r := 4 -- speed of each geologist in km/h
  let d := 6 -- distance in km
  let sectors := 8 -- number of sectors (roads)
  let total_outcomes := sectors * sectors
  let favorable_outcomes := sectors * 3 -- when distance > 6 km

  -- Calculating probability
  let P := (favorable_outcomes: ℝ) / (total_outcomes: ℝ)

  P = 0.375 :=
by
  sorry

end geologists_probability_l744_744615


namespace prove_bc_prove_area_l744_744230

variable {a b c A B C : ℝ}

-- Given conditions as Lean definitions
def condition_cos_A : Prop := (b^2 + c^2 - a^2 = 2 * cos A)
def condition_bc : Prop := b * c = 1
def condition_trig_identity : Prop := (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1

-- Definitions for the questions translated to Lean
def find_bc : Prop :=
  condition_cos_A → b * c = 1

def find_area (bc_value : Prop) : Prop :=
  bc_value →
  condition_trig_identity →
  1/2 * b * c * sin A = (Real.sqrt 3) / 4

-- Theorems to be proven
theorem prove_bc : find_bc :=
by
  intro h1
  sorry

theorem prove_area : find_area condition_bc :=
by
  intro h1 h2
  sorry

end prove_bc_prove_area_l744_744230


namespace lines_are_skew_iff_a_ne_5_l744_744091

def parametric_line (p d : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (p.1 + t * d.1, p.2 + t * d.2, p.3 + t * d.3)

theorem lines_are_skew_iff_a_ne_5 (a : ℝ) :
  let l1 := parametric_line (2, 3, a) (1, 4, 5),
      l2 := parametric_line (7, 0, 1) (3, 1, 2) in
  (∀ t u, l1 t ≠ l2 u) ↔ a ≠ 5 :=
by
  sorry

end lines_are_skew_iff_a_ne_5_l744_744091


namespace semicircle_chord_length_l744_744391

-- Assuming a semicircle with a certain radius, let's define the main conditions
def semicircle_area (r : ℝ) : ℝ :=
  (1 / 2) * Real.pi * r^2

def remaining_area (r : ℝ) : ℝ :=
  semicircle_area r - 2 * semicircle_area (r / 2)

theorem semicircle_chord_length (r : ℝ) (h : remaining_area r = 16 * Real.pi^3) : 
  2 * (r / 2) * Real.sqrt 2 = 32 :=
by
  sorry

end semicircle_chord_length_l744_744391


namespace simplify_sin_expression_l744_744309

theorem simplify_sin_expression : 
  (sin 58 - sin 28 * cos 30) / cos 28 = 1 / 2 :=
by
  sorry

end simplify_sin_expression_l744_744309


namespace loan_difference_calculation_l744_744445

noncomputable def compounded_amount (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r * t)

theorem loan_difference_calculation 
(P : ℝ)
(r1 r2 : ℝ)
(n : ℕ)
(t1 t2: ℝ)
(three_year_payment_fraction : ℝ)
: 
let compounded_initial := compounded_amount P r1 n t1 in
let payment := three_year_payment_fraction * compounded_initial in
let remaining := compounded_initial - payment in
let compounded_final := compounded_amount remaining r1 n t2 in
let total_compounded := payment + compounded_final in
let simple_total := simple_interest_amount P r2 (t1 + t2) in
(abs (simple_total - total_compounded) ≈ 2834) := 
  sorry

end loan_difference_calculation_l744_744445


namespace Euler_lines_coincide_l744_744980

open EuclideanGeometry

theorem Euler_lines_coincide
  (ABC : Triangle)
  (acute : is_acuted_triangle ABC)
  (A1 B1 C1 : Point)
  (feet_of_altitudes : are_feet_of_altitudes ABC A1 B1 C1)
  (A2 B2 C2 : Point)
  (incircle_touches : incircle_touches_sides A1 B1 C1 A2 B2 C2) :
  euler_line ABC = euler_line (Triangle.mk A2 B2 C2) := sorry

end Euler_lines_coincide_l744_744980


namespace calculate_total_driving_time_l744_744423

/--
A rancher needs to transport 400 head of cattle to higher ground 60 miles away.
His truck holds 20 head of cattle and travels at 60 miles per hour.
Prove that the total driving time to transport all cattle is 40 hours.
-/
theorem calculate_total_driving_time
  (total_cattle : Nat)
  (cattle_per_trip : Nat)
  (distance_one_way : Nat)
  (speed : Nat)
  (round_trip_miles : Nat)
  (total_miles : Nat)
  (total_time_hours : Nat)
  (h1 : total_cattle = 400)
  (h2 : cattle_per_trip = 20)
  (h3 : distance_one_way = 60)
  (h4 : speed = 60)
  (h5 : round_trip_miles = 2 * distance_one_way)
  (h6 : total_miles = (total_cattle / cattle_per_trip) * round_trip_miles)
  (h7 : total_time_hours = total_miles / speed) :
  total_time_hours = 40 :=
by
  sorry

end calculate_total_driving_time_l744_744423


namespace find_vertex_P_l744_744623

-- Define the midpoints as given conditions
def midpoint_QR := (2, 3, 1) : ℝ × ℝ × ℝ
def midpoint_PR := (-1, 2, 3) : ℝ × ℝ × ℝ
def midpoint_PQ := (4, 1, -2) : ℝ × ℝ × ℝ

-- Define the coordinates of point P that we aim to prove
def vertex_P := (7, 2, -4) : ℝ × ℝ × ℝ

-- The theorem statement that we need to prove
theorem find_vertex_P :
  ∃ P : ℝ × ℝ × ℝ, 
    P = vertex_P ∧
    (2 * (fst P + snd P) / 2, 
     2 * (snd P) / 2, 
     2 * (snd P - sixth P) / 2) = midpoint_PR ∧ 
    (P, (2, 3, 1))
      ∧ ((4 + fst (2, 3, 1)) / 2, 
          (1 + snd (2, 3, 1)) / 2, 
          (-2 + trd (2, 3, 1)) / 2) = (3, 2, -1/2) := 
sorry

end find_vertex_P_l744_744623


namespace a_9_value_l744_744621

-- Define the sequence and conditions
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Add the given conditions
def a1 := a 1 = 2
def S_def (n : ℕ) := S n = (Finset.range (n + 1)).sum (λ i, a (i + 1))
def point_on_line (n : ℕ) := (S n) / n.succ = 2 * (S n) / n - 2

-- The goal is to prove a_9 = 1281
theorem a_9_value : 
  a1 ∧ (∀ n, S_def n) ∧ (∀ n, point_on_line n) → a 9 = 1281 :=
by simp [a1, S_def, point_on_line]; sorry

end a_9_value_l744_744621


namespace range_of_f_when_a_eq_2_sufficient_but_not_necessary_condition_for_q_l744_744143

-- Define the function
def f (x a : ℝ) : ℝ := x^2 - a * x + 4 - a^2

-- Problem (1): Range of the function when a = 2
theorem range_of_f_when_a_eq_2 :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x 2 = (x - 1)^2 - 1) →
  Set.image (f 2) (Set.Icc (-2 : ℝ) 3) = Set.Icc (-1 : ℝ) 8 := sorry

-- Problem (2): Sufficient but not necessary condition
theorem sufficient_but_not_necessary_condition_for_q :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x 4 ≤ 0) →
  (Set.Icc (-2 : ℝ) 2 → (∃ (M : Set ℝ), Set.singleton 4 ⊆ M ∧ 
    (∀ a ∈ M, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 0) ∧
    (∀ a ∈ Set.Icc (-2 : ℝ) 2, ∃ a' ∉ M, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a' ≤ 0))) := sorry

end range_of_f_when_a_eq_2_sufficient_but_not_necessary_condition_for_q_l744_744143


namespace total_loaves_served_l744_744407

-- Definitions based on the conditions provided
def wheat_bread_loaf : ℝ := 0.2
def white_bread_loaf : ℝ := 0.4

-- Statement that needs to be proven
theorem total_loaves_served : wheat_bread_loaf + white_bread_loaf = 0.6 := 
by
  sorry

end total_loaves_served_l744_744407


namespace sequence_sum_l744_744919

theorem sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n : ℕ, S (n + 1) = (n + 1) * (n + 1) - 1)
  (ha : ∀ n : ℕ, a (n + 1) = S (n + 1) - S n) :
  a 1 + a 3 + a 5 + a 7 + a 9 = 44 :=
by
  sorry

end sequence_sum_l744_744919


namespace compute_A_3_2_l744_744470

def A : ℕ → ℕ → ℕ
| 0 n := n + 1
| (m + 1) 0 := A m 1
| (m + 1) (n + 1) := A m (A (m + 1) n)

theorem compute_A_3_2 : A 3 2 = 19 := 
by {
    -- skipped proof
    sorry
}

end compute_A_3_2_l744_744470


namespace prob_not_less_l744_744021

open ProbabilityTheory

-- Define the experiment
def example_space := {1, 2, 3, 4}

-- Define the event corresponding to drawing a ball
def draw_ball (s : example_space) : Prop := s ∈ example_space

-- Define the probability of drawing each numbered ball
noncomputable def prob_draw_ball (n : ℕ) : ℚ :=
  if n ∈ example_space then 1 / 4 else 0

-- Define the event where the second draw is not less than the first draw
def event (n₁ n₂ : ℕ) : Prop :=
  n₂ >= n₁

-- Define the probability space
noncomputable def probability_space : MeasureTheory.ProbabilitySpace (example_space × example_space)  :=
  ⟨
    λ (s : example_space × example_space), prob_draw_ball s.1 * prob_draw_ball s.2,
    sorry  -- skip the proof
  ⟩

-- Define the event probability in the probability space
noncomputable def event_prob_not_less : ℚ :=
  probability_space.prob (λ ω, event ω.1 ω.2)

-- The theorem we want to prove
theorem prob_not_less : event_prob_not_less = 5 / 8 :=
  sorry

end prob_not_less_l744_744021


namespace guests_calculation_l744_744064

theorem guests_calculation (total_cookies : ℕ) (cookies_per_guest : ℕ) (h1 : total_cookies = 38) (h2 : cookies_per_guest = 19) :
  total_cookies / cookies_per_guest = 2 :=
by
  rw [h1, h2]
  sorry

end guests_calculation_l744_744064


namespace antonio_meatballs_l744_744822

-- Define the conditions
def meat_per_meatball : ℝ := 1 / 8
def family_members : ℕ := 8
def total_hamburger : ℝ := 4

-- Assertion to prove
theorem antonio_meatballs : 
  (total_hamburger / meat_per_meatball) / family_members = 4 :=
by sorry

end antonio_meatballs_l744_744822


namespace angle_BCD_is_80_l744_744966

theorem angle_BCD_is_80 
  (A B C D : Type) 
  (h₁: CD ∥ AB)
  (h₂: m∠ DCB = 50) : 
  m∠ BCD = 80 := 
by 
  -- proof
  sorry

end angle_BCD_is_80_l744_744966


namespace points_E_F_existence_BE_expression_l744_744947

variable (A B C D E F : Type)
variable (a b c : ℝ)
variable [LT b c]
variable [Angle A B C : ℝ]
variable [Angle B E D : ℝ]
variable [Angle C F D : ℝ]

-- Define the necessary condition
def necessary_condition (A B : Angle) : Prop :=
  2 * B > A

-- Prove the necessary condition for the existence of points E and F
theorem points_E_F_existence (hAD: is_angle_bisector A B C D) :
  necessary_condition A B ↔ (∃ E F, E ∈ (AB \ {A, B}) ∧ F ∈ (AC \ {A, C}) ∧ (B * E = C * F) ∧ (Angle B E D = Angle C F D)) :=
sorry

-- Prove the expression for BE in terms of a, b, and c
theorem BE_expression (h_exist: ∃ E F, E ∈ (AB \ {A, B}) ∧ F ∈ (AC \ {A, C}) ∧ (B * E = C * F) ∧ (Angle B E D = Angle C F D)) :
  B * E = a^2 / (b + c) :=
sorry

end points_E_F_existence_BE_expression_l744_744947


namespace exists_ten_positive_integers_l744_744478

theorem exists_ten_positive_integers :
  ∃ (a : ℕ → ℕ), (∀ i j, i ≠ j → ¬ (a i ∣ a j))
  ∧ (∀ i j, (a i)^2 ∣ a j) :=
sorry

end exists_ten_positive_integers_l744_744478


namespace prove_relationships_l744_744284

variables {p0 p1 p2 p3 : ℝ}
variable (h_p0_pos : p0 > 0)
variable (h_gas_car : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
variable (h_hybrid_car : 10^(5 / 2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)
variable (h_electric_car : p3 = 100 * p0)

theorem prove_relationships : 
  (p1 ≥ p2) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) := by
  split
  sorry

end prove_relationships_l744_744284


namespace share_difference_l744_744059

theorem share_difference (x : ℤ) (hx : 5 * x = 4000) : 4 * x = 3200 :=
by
  -- Define the shares
  let p := 3 * x
  let q := 7 * x
  let r := 12 * x
  -- Validate the condition
  have h_diff_qr : r - q = 4000, from by
    rw [r, q]
    rw [← hx]
    exact Eq.refl 4000
  -- Conclude the difference between p and q
  show 4 * x = 3200, from by
    rw [← hx]
    exact Eq.refl 3200

end share_difference_l744_744059


namespace simplify_abs_expression_l744_744133

theorem simplify_abs_expression (a b c : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : c = 0) :
  |a - c| + |c - b| - |a - b| = 0 := 
by
  sorry

end simplify_abs_expression_l744_744133


namespace value_of_g_at_3_l744_744694

def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_of_g_at_3 : g 3 = 3 := by
  sorry

end value_of_g_at_3_l744_744694


namespace area_triangle_ACE_given_pentagon_area_ABCDE_l744_744826

variables (A B C D E F : Type) [euclidean_geometry A] [segment A C] [segment C E]

theorem area_triangle_ACE_given_pentagon_area_ABCDE 
  (area_pentagon_ABCDE : ℝ)
  (perpendicular_BC_CE : ⊥ (segment B C) (segment C E))
  (perpendicular_EF_CE : ⊥ (segment E F) (segment C E))
  (square_ABDF : is_square (quadrilateral A B D F))
  (ratio_CD_ED : ∃ a : ℝ, length (segment C D) = 3 * a ∧ length (segment D E) = 2 * a)
  (hyp : area_pentagon_ABCDE = 2014) :
  (∃ a : ℝ, a ^ 2 = 106) →
  (area (triangle A C E) = 1325) :=
by
  sorry

end area_triangle_ACE_given_pentagon_area_ABCDE_l744_744826


namespace maximize_area_CDFE_l744_744219

noncomputable def area_CDFE (AF AE : ℝ) : ℝ :=
  let x : ℝ := AF in
  let E : ℝ×ℝ := (AE, 2) in
  let F : ℝ×ℝ := (0, x) in
  let triangle_CDF_area := x in
  let triangle_DFE_area := (1 / 2) * (2 - x) * (2 - x) in
  triangle_CDF_area + triangle_DFE_area

theorem maximize_area_CDFE
  (ABCD_is_square : ∀(A B C D : ℝ×ℝ), dist A B = 2 ∧ dist B C = 2 ∧ dist C D = 2 ∧ dist D A = 2 ∧ dist A C = dist B D)
  (E_on_AB : ∃(AE : ℝ) (AP : ℝ×ℝ), AE = 2 * AF ∧ AP = E)
  (F_on_AD : ∃(AF : ℝ) (FP : ℝ×ℝ), FP = F)
  : ∃ (AF : ℝ), area_CDFE AF (2 * AF) = 1.5 :=
sorry

end maximize_area_CDFE_l744_744219


namespace rectangle_problem_l744_744061

noncomputable def rectangle_fold (AB AD AED DE AE BC F) : Prop :=
  ∃ BC : ℝ,
  AB = 10 ∧
  (∃ AE : ℝ, AE = BC) ∧
  (∃ BF : ℝ, BF = 2 / 5) ∧
  (∃ x : ℝ, BC = x) ∧
  (∃ area_ABF : ℝ, area_ABF = (1 / 2) * 10 * (2 / 5)) ∧
  area_ABF = 2 ∧
  BC = 5.2

theorem rectangle_problem: rectangle_fold 10 AD AED DE AE 5.2 F := sorry

end rectangle_problem_l744_744061


namespace kite_area_l744_744018

-- Definitions for points P and Q, the square, and the kite-shaped region
def side_length : ℝ := 10
def midpoints (a b : ℝ) : ℝ := (a + b) / 2
def area_of_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height
def total_area_of_kite (triangle_area : ℝ) : ℝ := 2 * triangle_area

-- Given the conditions
def square_side_length := side_length
def point_P := midpoints 0 side_length  -- midpoint of the bottom side (0 to side_length)
def point_Q := midpoints 12 (12 + side_length)  -- midpoint of the top side (12 to 12+side_length)
def line_PQ_length := side_length

-- Proposition to prove the desired area
theorem kite_area :
  ∃ (PA : ℝ) (RAS : ℝ), PA = area_of_triangle (side_length / 2) (side_length / 2) ∧ RAS = 25 :=
by
  use area_of_triangle (side_length / 2) (side_length / 2), 25
  sorry

end kite_area_l744_744018


namespace third_segment_lt_quarter_AC_l744_744968

theorem third_segment_lt_quarter_AC 
  (A B C : Type) [triangle A B C] 
  (angleA angleB angleC : ℝ) 
  (h1 : 3 * angleA - angleC < π)
  (h2 : ∀ K L M : Type, segment_divides B AC into_four_equal_parts K L M) :
  segment_length A (third_segment AC K L M) < segment_length AC / 4 :=
sorry

end third_segment_lt_quarter_AC_l744_744968


namespace largest_sample_number_l744_744786

theorem largest_sample_number (total_employees dismissed_employees : ℕ) (remaining_employees smallest_number interval : ℕ) :
   total_employees = 624 →
   dismissed_employees = 4 →
   remaining_employees = total_employees - dismissed_employees →
   smallest_number = 7 →
   interval = remaining_employees / 62 →
   ∃ largest_sample_number, largest_sample_number = smallest_number + (interval * 61) ∧ largest_sample_number = 617 :=
begin
  sorry
end

end largest_sample_number_l744_744786


namespace probability_car_Y_win_l744_744952

-- Define the probabilities
def P_X := 1 / 4
def P_Z := 1 / 12
def P_total := 0.4583333333333333

-- Define the requirement for mutually exclusive events
def mutually_exclusive (A B C : Prop) : Prop :=
  (A → ¬B) ∧ (A → ¬C) ∧ (B → ¬C)

-- The proof statement
theorem probability_car_Y_win :
  let P_Y := P_total - (P_X + P_Z) in
  (mutually_exclusive (P_X ≠ 0) (P_Y ≠ 0) (P_Z ≠ 0)) →
  P_Y = 1 / 8 :=
by
  sorry

end probability_car_Y_win_l744_744952


namespace variance_of_dataset_l744_744686

theorem variance_of_dataset (x : ℝ) (h : (8 + 9 + x + 11 + 12) / 5 = 10) : 
  (1/5) * ((8 - 10)^2 + (9 - 10)^2 + (x - 10)^2 + (11 - 10)^2 + (12 - 10)^2) = 2 :=
by 
  have hx : x = 10 := sorry
  rw hx
  sorry

end variance_of_dataset_l744_744686


namespace fraction_equality_l744_744469

def at (a b : ℤ) : ℤ := a * b - b ^ 2
def hash (a b : ℤ) : ℤ := a + b - a * b ^ 2

theorem fraction_equality : (at 6 2 : ℚ) / hash 6 2 = -1 / 2 := by
  sorry

end fraction_equality_l744_744469


namespace roots_sum_and_product_l744_744096

theorem roots_sum_and_product (k p : ℝ) (hk : (k / 3) = 9) (hp : (p / 3) = 10) : k + p = 57 := by
  sorry

end roots_sum_and_product_l744_744096


namespace cos_double_angle_l744_744581

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l744_744581


namespace pen_shorter_than_pencil_l744_744796

-- Definitions of the given conditions
def P (R : ℕ) := R + 3
def L : ℕ := 12
def total_length (R : ℕ) := R + P R + L

-- The theorem to be proven
theorem pen_shorter_than_pencil (R : ℕ) (h : total_length R = 29) : L - P R = 2 :=
by
  sorry

end pen_shorter_than_pencil_l744_744796


namespace number_of_real_roots_l744_744500

theorem number_of_real_roots (a : ℝ) :
  (|a| < (2 * Real.sqrt 3 / 9) → ∃ x y z : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ z^3 - z - a = 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  (|a| = (2 * Real.sqrt 3 / 9) → ∃ x y : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ x = y) ∧
  (|a| > (2 * Real.sqrt 3 / 9) → ∃ x : ℝ, x^3 - x - a = 0 ∧ ∀ y : ℝ, y^3 - y - a ≠ 0 ∨ y = x) :=
sorry

end number_of_real_roots_l744_744500


namespace molecular_weight_of_N2O5_is_correct_l744_744845

-- Definitions for atomic weights
def atomic_weight_N : ℚ := 14.01
def atomic_weight_O : ℚ := 16.00

-- Define the molecular weight calculation for N2O5
def molecular_weight_N2O5 : ℚ := (2 * atomic_weight_N) + (5 * atomic_weight_O)

-- The theorem to prove
theorem molecular_weight_of_N2O5_is_correct : molecular_weight_N2O5 = 108.02 := by
  -- Proof here
  sorry

end molecular_weight_of_N2O5_is_correct_l744_744845


namespace cos_double_angle_l744_744576

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l744_744576


namespace more_knights_than_liars_l744_744294

-- Define the two tribes
inductive Tribe
| Knight : Tribe
| Liar : Tribe

-- Number of Knights and Liars
variables (knights liars : ℕ)

-- Friendship relation
variable (friends : ℕ → ℕ → Bool)
-- friends i j indicates whether i-th and j-th islander are friends

-- Predicate for "more than half of my friends are fellow tribesmen"
def more_than_half_friends_fellow_tribesmen (i : ℕ) : Bool :=
  let friends_count := (List.range (knights + liars)).filter (friends i)
  let fellow_tribesmen_count := friends_count.filter (λ j => j < knights)
  if i < knights then
    fellow_tribesmen_count.length > friends_count.length / 2
  else
    (friends_count.length - fellow_tribesmen_count.length) > friends_count.length / 2

theorem more_knights_than_liars (H : ∀ (i : ℕ), i < knights + liars → more_than_half_friends_fellow_tribesmen i) : 
  knights > liars :=
sorry

end more_knights_than_liars_l744_744294


namespace julian_comic_pages_l744_744631

-- Definitions from conditions
def frames_per_page : ℝ := 143.0
def total_frames : ℝ := 1573.0

-- The theorem stating the proof problem
theorem julian_comic_pages : total_frames / frames_per_page = 11 :=
by
  sorry

end julian_comic_pages_l744_744631


namespace equal_probability_of_sectors_probability_of_consecutive_sectors_l744_744695

-- Conditions
constant numOfSectors : ℕ := 13
constant numOfPlayedSectors : ℕ := 6

-- Part (a)
theorem equal_probability_of_sectors : 
  (6 / 13 = 6 / 13) :=
sorry

-- Part (b)
theorem probability_of_consecutive_sectors :
  (7 ^ 5 / 13 ^ 6) = (7 ^ 5 / 13 ^ 6) :=
sorry

end equal_probability_of_sectors_probability_of_consecutive_sectors_l744_744695


namespace base_of_logarithm_l744_744333

theorem base_of_logarithm (x b : ℝ) (hx : 5^(3*x + 4) = 12^x) (hx_exp : x = Real.logb b (5^4)) : b = (12 / 5) :=
by
  sorry

end base_of_logarithm_l744_744333


namespace greatest_m_l744_744037

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_reversed (m n : ℕ) : Prop :=
  let rev := λ (x : ℕ), x % 10 * 1000 + (x % 100) / 10 * 100 + (x % 1000) / 100 * 10 + x / 1000 in
  rev m = n

def is_divisible_by (x y : ℕ) : Prop :=
  y ≠ 0 ∧ x % y = 0

theorem greatest_m (m : ℕ) :
  is_four_digit m →
  (∃ n, is_four_digit n ∧ is_reversed m n) →
  is_divisible_by m 45 →
  is_divisible_by m 7 →
  m = 5985 :=
by
  intros _ _ _ _
  sorry

end greatest_m_l744_744037


namespace simplify_expression_l744_744308

theorem simplify_expression (n : ℕ) : 
  (3 ^ (n + 5) - 3 * 3 ^ n) / (3 * 3 ^ (n + 4)) = 80 / 27 :=
by sorry

end simplify_expression_l744_744308


namespace number_of_sides_is_5_l744_744184

-- Definitions based on conditions
def is_polygon (n : ℕ) : Prop := n ≥ 3
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

-- Given conditions
def has_2_diagonals (n : ℕ) : Prop := diagonals_from_vertex(n) = 2

-- Theorem to prove
theorem number_of_sides_is_5 (n : ℕ) (h1 : is_polygon n) (h2 : has_2_diagonals n) : n = 5 :=
by
  sorry

end number_of_sides_is_5_l744_744184


namespace parallelogram_area_Lean_l744_744225

open Real EuclideanSpace

noncomputable def parallelogram_area (p q : EuclideanSpace ℝ (Fin 3))
  (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (hθ : ∃ θ, θ = π / 4 ∧ cos θ = (p ⬝ q)) : ℝ :=
  ‖ ((3 : ℝ) • q - p) × ((3 : ℝ) • p + (3 : ℝ) • q) ‖

theorem parallelogram_area_Lean (p q : EuclideanSpace ℝ (Fin 3))
  (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (hθ : ∃ θ, θ = π / 4 ∧ cos θ = (p ⬝ q)) :
  parallelogram_area p q hp hq hθ = 9 * Real.sqrt 2 / 4 :=
sorry

end parallelogram_area_Lean_l744_744225


namespace sum_of_coefficients_at_1_l744_744865

def P (x : ℝ) := 2 * (4 * x^8 - 3 * x^5 + 9)
def Q (x : ℝ) := 9 * (x^6 + 2 * x^3 - 8)
def R (x : ℝ) := P x + Q x

theorem sum_of_coefficients_at_1 : R 1 = -25 := by
  sorry

end sum_of_coefficients_at_1_l744_744865


namespace probability_X_eq_1_l744_744166

noncomputable def X : Type :=
  ℕ

def binomial_pmf (n : ℕ) (p : ℚ) : pmf ℕ :=
  pmf.of_finset (finset.range (n + 1))
    (λ k, (n.choose k : ℚ) * (p^k) * ((1 - p)^(n - k)))
    (by sorry) -- This lemma ensures the sum to 1

theorem probability_X_eq_1 (X : ℕ) (n : ℕ) (p : ℚ) (h1 : X ∼ binomial_pmf n p)
  (h2 : X.mean = 6) (h3 : X.variance = 3) : P(X = 1) = 3 * (2 : ℚ) ^ (-10) :=
sorry

end probability_X_eq_1_l744_744166


namespace inequality_solution_set_max_min_values_range_of_m_l744_744545

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a*Real.exp x - x
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := a*Real.exp x - 1

-- Condition (I) - Solution set of the inequality f'(x)(x-1) > 0 for different values of a.
theorem inequality_solution_set (a x: ℝ) : ((a*Real.exp x - 1) * (x - 1) > 0) ↔ 
  ((a ≤ 0 ∧ x < 1) ∨ (0 < a ∧ a < 1/Real.exp 1 ∧ (x < 1 ∨ x > Real.ln (1/a))) ∨ 
  (a = 1/Real.exp 1 ∧ x ≠ 1) ∨ (a > 1/Real.exp 1 ∧ (x < Real.ln (1/a) ∨ x > 1))) :=
sorry

-- Condition (II)(a) - Find maximum and minimum values of f(x) on x ∈ [-m, m] for a = 1
theorem max_min_values (m : ℝ) (h : 0 < m) :
  let a := 1 in 
  f(0, a) = 1 ∧ (f(m, a) = Real.exp m - m) :=
by
  simp [f]
  sorry

-- Condition (II)(b) - Determine the range of m such that f(x) < e^2 - 2 holds for all x ∈ [-m, m]
theorem range_of_m (m : ℝ) : 
  let a := 1 in 
  (0 < m ∧ m < 2) ↔ (f(m, a) < Real.exp 2 - 2 ∧ f(-m, a) < Real.exp 2 - 2) :=
by
  simp [f]
  sorry

end inequality_solution_set_max_min_values_range_of_m_l744_744545


namespace math_problem_l744_744376

theorem math_problem : (((((3 + 2)⁻¹ - 1)⁻¹ - 1)⁻¹) - 1) = -13 / 9 :=
by
  sorry

end math_problem_l744_744376


namespace nonempty_solution_iff_a_gt_one_l744_744599

theorem nonempty_solution_iff_a_gt_one (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 :=
sorry

end nonempty_solution_iff_a_gt_one_l744_744599


namespace sum_a_eq_T_n_l744_744868

noncomputable def sum_sequence_a (n : ℕ) : ℤ :=
  if n = 1 then -4 else (λ (a_n : ℕ → ℤ) (n : ℕ), a_n n) 
  (λ (a_n : ℕ → ℤ)
    (n : ℕ),
      if n = 1 then -4
      else (∑ i in finset.range(n + 1), -4 * (i : ℤ))
  ) n

theorem sum_a_eq_T_n (n : ℕ) (h : 0 < n) : 
  sum_sequence_a n = -2 * n * (n + 1) :=
sorry

end sum_a_eq_T_n_l744_744868


namespace Tom_earns_per_week_l744_744353

-- Definitions based on conditions
def crab_buckets_per_day := 8
def crabs_per_bucket := 12
def price_per_crab := 5
def days_per_week := 7

-- The proof goal
theorem Tom_earns_per_week :
  (crab_buckets_per_day * crabs_per_bucket * price_per_crab * days_per_week) = 3360 := by
  sorry

end Tom_earns_per_week_l744_744353


namespace plane_points_distance_bound_l744_744290

theorem plane_points_distance_bound (N n : ℕ) (hN : N ≥ 3)
  (h_distinct : ∀ (A : Fin N → ℝ × ℝ), ∃ (distances : Finset ℝ), 
  distances.card ≤ n ∧ ∀ i j, i ≠ j → (∃ d ∈ distances, d = dist (A i) (A j))) : 
  N ≤ (n + 1) ^ 2 :=
sorry

def dist (a b : ℝ × ℝ) : ℝ := 
  let (ax, ay) := a
  let (bx, by) := b
  (sqrt ((ax - bx)^2 + (ay - by)^2))

end plane_points_distance_bound_l744_744290


namespace evaluate_expression_l744_744856

-- Define the base and the exponents
def base : ℝ := 11
def exp1 : ℝ := 1/5
def exp2 : ℝ := 1/4

-- Define the expression in question
def expression : ℝ := base^(exp1) / base^(exp2)

-- Statement of the theorem
theorem evaluate_expression : expression = base^(-(1/20)) := by
  sorry

end evaluate_expression_l744_744856


namespace probability_four_even_four_odd_l744_744484

-- Definition of the problem conditions: Rolling eight 8-sided dice.
def eight_8_sided_dice (dices : Fin 8 → Fin 8) : Prop :=
  ∀ i, dices i < 8

-- Definition of even numbers condition
def is_even (n : ℕ) : Prop :=
  n % 2 = 0

-- Definition of odd numbers condition
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

-- Each die has 4 even and 4 odd outcomes.
def dice_has_even_and_odd : Prop :=
  ∀ (n : Fin 8), is_even (n : ℕ) → n.val < 8 / 2 ∧ is_odd (n : ℕ) → n.val < 8 / 2

-- The main theorem to prove.
theorem probability_four_even_four_odd :
  eight_8_sided_dice dices →
  dice_has_even_and_odd →
  (probability (exactly_four_even dices) = 35 / 128) :=
by
  sorry

end probability_four_even_four_odd_l744_744484


namespace sum_of_rel_primes_l744_744989

theorem sum_of_rel_primes (c d : ℕ) (hc : Nat.gcd c d = 1)
  (h : (c : ℚ) / d = ∑ n in (Finset.range 100), if n % 2 = 0 then (n/2 + 1) / 2^(n/2 + 2) else (n/2 + 1) / 3^(n/2 + 3)):
  c + d = 169 := by
  sorry

end sum_of_rel_primes_l744_744989


namespace find_k_l744_744213

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^2 + 2 * x + 1) / (k * x - 1)

theorem find_k (k : ℝ) : (∀ x : ℝ, f k (f k x) = x) ↔ k = -2 :=
  sorry

end find_k_l744_744213


namespace find_x_l744_744764

theorem find_x (m n k : ℝ) (x z : ℝ) (h1 : x = m * (n / (Real.sqrt z))^3)
  (h2 : x = 3 ∧ z = 12 ∧ 3 * 12 * Real.sqrt 12 = k) :
  (z = 75) → x = 24 / 125 :=
by
  -- Placeholder for proof, these assumptions and conditions would form the basis of the proof.
  sorry

end find_x_l744_744764


namespace hyperbola_circle_intersection_l744_744128

open Real

theorem hyperbola_circle_intersection (a r : ℝ) (P Q R S : ℝ × ℝ) 
  (hP : P.1^2 - P.2^2 = a^2) (hQ : Q.1^2 - Q.2^2 = a^2) (hR : R.1^2 - R.2^2 = a^2) (hS : S.1^2 - S.2^2 = a^2)
  (hO : r ≥ 0)
  (hPQRS : (P.1 - 0)^2 + (P.2 - 0)^2 = r^2 ∧
            (Q.1 - 0)^2 + (Q.2 - 0)^2 = r^2 ∧
            (R.1 - 0)^2 + (R.2 - 0)^2 = r^2 ∧
            (S.1 - 0)^2 + (S.2 - 0)^2 = r^2) : 
  (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) + (R.1^2 + R.2^2) + (S.1^2 + S.2^2) = 4 * r^2 :=
by
  sorry

end hyperbola_circle_intersection_l744_744128


namespace minimum_correct_questions_l744_744052

variable (x : ℕ)

def conditions : Prop := 
  let total_questions := 25
  let points_correct := 4
  let points_incorrect := -1
  ∀ x, total_questions - x ≥ 0 → (points_correct * x + points_incorrect * (total_questions - x)) > 70

def question : Prop := x ≥ 19

theorem minimum_correct_questions (x : ℕ) (h : conditions x) : question x := sorry

end minimum_correct_questions_l744_744052


namespace average_chore_time_l744_744958

theorem average_chore_time 
  (times : List ℕ := [4, 3, 2, 1, 0])
  (counts : List ℕ := [2, 4, 2, 1, 1]) 
  (total_students : ℕ := 10)
  (total_time : ℕ := List.sum (List.zipWith (λ t c => t * c) times counts)) :
  (total_time : ℚ) / total_students = 2.5 := by
  sorry

end average_chore_time_l744_744958


namespace probability_of_seventh_head_l744_744264

theorem probability_of_seventh_head (fair_coin : ∀ flip, (flip = "H" ∨ flip = "T") ∧ flip = "H" → (1/2)) 
  (independent_flips : ∀ flip1 flip2, flip1 ≠ flip2) : 
  (∀ n, n = 7 → (1/2)) := 
by
  sorry

end probability_of_seventh_head_l744_744264


namespace monotonicity_when_a_eq_0_range_of_a_l744_744651

-- Define the function f(x) = ln(x) - (a+1)x
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (a + 1) * x

-- Monotonicity when a = 0
theorem monotonicity_when_a_eq_0 : 
  ∀ x : ℝ, 0 < x → 
  ((0 < x ∧ x < 1 → 0 < Real.log x - x - 1) ∧ 
  (1 < x → Real.log x - x - 1 < 0)) := 
sorry

-- Range of a when a > -1 and max value of f(x) > -2
theorem range_of_a (a : ℝ) : 
  (-1 < a ∧ a < Real.exp 1 - 1) ↔ (a > -1 ∧ (Real.log (a + 1) - 1 > -2)) := 
sorry

end monotonicity_when_a_eq_0_range_of_a_l744_744651


namespace balloon_height_l744_744688

-- Define the problem given in the question
def angle_elevation_A: ℝ := 45
def angle_elevation_B: ℝ := 22.5
def distance_AB: ℝ := 1600
def angle_directions: ℝ := 135

-- Prove that the height of the balloon above the ground is approximately 500 meters given the above conditions
theorem balloon_height : 
  ∀ (m : ℝ) (α β δ d : ℝ),
  α = angle_elevation_A → β = angle_elevation_B → δ = angle_directions → d = distance_AB →
  (∃ m : ℝ, abs(m - 500) < 0.001) :=
by
  sorry

end balloon_height_l744_744688


namespace Antonio_eats_meatballs_l744_744821

def meatballs_per_member (total_hamburger : ℝ) (hamburger_per_meatball : ℝ) (num_family_members : ℕ) : ℝ :=
  (total_hamburger / hamburger_per_meatball) / num_family_members

theorem Antonio_eats_meatballs :
  meatballs_per_member 4 (1 / 8) 8 = 4 := 
by
  sorry

end Antonio_eats_meatballs_l744_744821


namespace ellipse_eq_fixed_points_l744_744515

noncomputable def sqrt7 : ℝ := real.sqrt 7
noncomputable def sqrt3 : ℝ := real.sqrt 3

structure Ellipse :=
(a b : ℝ)
(a_pos : a > 0)
(b_pos : b > 0)

instance : Inhabited Ellipse :=
⟨{a := 2, b := sqrt3, a_pos := by norm_num, b_pos := real.sqrt_pos.mpr zero_lt_three}⟩

theorem ellipse_eq (e : Ellipse) (h : e.a > e.b) (h_ab : sqrt7 = real.sqrt (e.a ^ 2 + e.b ^ 2)) (h_slope : e.b / e.a = sqrt3):
  e.a = 2 ∧ e.b = sqrt3 ∧ (∀ x y, (x^2 / e.a^2) + (y^2 / e.b^2) = 1 ↔ (x^2 / 4) + (y^2 / 3) = 1) :=
begin
  sorry
end

noncomputable def M (m : ℝ) : ℝ × ℝ :=
(4 - 3 * m ^ 2) / (3 * m ^ 2 + 4), -6 * m / (3 * m ^ 2 + 4)

theorem fixed_points {m : ℝ} (elm : (3 * m ^ 2 + 4) ≠ 0) :
  (∀ P Q F, F = (1, 0) → P ≠ Q → ∃ M : ℝ × ℝ, M = M m (P, Q) →
   (dists : ℝ := real.dist M (-1 / 2, 0) + real.dist M (1 / 2, 0)) → dists = 2) :=
begin
  sorry
end

end ellipse_eq_fixed_points_l744_744515


namespace cross_section_area_l744_744517

variables {S A B C O O' P : Type} [metric_space S] [metric_space A] [metric_space B] [metric_space C] [metric_space O] [metric_space O'] [metric_space P]
open_locale big_operators

-- Condition definitions
variables (height_SO : ℝ) (base_edge : ℝ) (AO_O' : ℝ) (AP_ratio : ℝ)
variables (O'_foot_perpendicular: Type) (prism_equilateral: Type) (P_on_AO':  Type)

-- Given conditions
def conditions : Prop :=
  height_SO = 3 ∧
  base_edge = 6 ∧
  AO_O' = 3 * sqrt 3 / 2 + 3 * sqrt 3 ∧
  AP_ratio = 8 

-- Proof statement
theorem cross_section_area (S A B C O O' P : Type) [metric_space S] [metric_space A]
  [metric_space B] [metric_space C] [metric_space O] [metric_space O'] [metric_space P]
  (h: conditions height_SO base_edge AO_O' AP_ratio O'_foot_perpendicular prism_equilateral P_on_AO') :
  ∃ (area : ℝ), area = sqrt 3 :=
begin
  sorry
end

end cross_section_area_l744_744517


namespace total_amount_spent_l744_744630

-- Definitions based on conditions
def price_basketball_game : ℝ := 5.20
def price_racing_game : ℝ := 4.23
def sales_tax_rate : ℝ := 0.065

-- Theorem to prove
theorem total_amount_spent :
  let sales_tax_basketball := (price_basketball_game * sales_tax_rate).round(2)
  let sales_tax_racing := (price_racing_game * sales_tax_rate).round(2)
  let total_cost_basketball := price_basketball_game + sales_tax_basketball
  let total_cost_racing := price_racing_game + sales_tax_racing
  total_cost_basketball + total_cost_racing = 10.04 :=
by 
  sorry

end total_amount_spent_l744_744630


namespace min_value_proof_l744_744894

noncomputable def min_value (a b : ℝ) : ℝ := (1 : ℝ)/a + (1 : ℝ)/b

theorem min_value_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + 2 * b = 2) :
  min_value a b = (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_proof_l744_744894


namespace perimeter_division_l744_744709

-- Define the given conditions
def is_pentagon (n : ℕ) : Prop := n = 5
def side_length (s : ℕ) : Prop := s = 25
def perimeter (P : ℕ) (n s : ℕ) : Prop := P = n * s

-- Define the Lean statement to prove
theorem perimeter_division (n s P x : ℕ) 
  (h1 : is_pentagon n) 
  (h2 : side_length s) 
  (h3 : perimeter P n s) 
  (h4 : P = 125) 
  (h5 : s = 25) : 
  P / x = s → x = 5 := 
by
  sorry

end perimeter_division_l744_744709


namespace analytical_expression_and_period_perimeter_of_triangle_ABC_l744_744889

open Real

variables (x A b c : ℝ)

def f (x : ℝ) : ℝ := 4 - 2 * sin (x + pi / 3)
def area_ABC (b c : ℝ) : ℝ := (1 / 2) * b * c * sin (2 * pi / 3)

theorem analytical_expression_and_period :
  (∀ x, f x = 4 - 2 * sin (x + pi / 3)) ∧ (∀ x, f (x + 2 * pi) = f x) :=
by
  sorry

theorem perimeter_of_triangle_ABC
  (A : ℝ) (h1 : f A = 4) (h2 : 0 < A ∧ A < pi)
  (h3 : b * c = 3) (h4 : area_ABC b c = 3 * sqrt 3 / 4) :
  b + c + sqrt ((b + c)^2 + b * c) = 3 + 2 * sqrt 3 :=
by
  sorry

end analytical_expression_and_period_perimeter_of_triangle_ABC_l744_744889


namespace ratio_a_c_l744_744767

theorem ratio_a_c (a b c : ℕ) (h1 : a / b = 8 / 3) (h2 : b / c = 1 / 5) : a / c = 8 / 15 := 
by
  sorry

end ratio_a_c_l744_744767


namespace linear_function_decreases_l744_744298

theorem linear_function_decreases (m b x : ℝ) (h : m < 0) : 
  ∃ y : ℝ, y = m * x + b ∧ ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m * x₁ + b) > (m * x₂ + b) :=
by 
  sorry

end linear_function_decreases_l744_744298


namespace Tom_earns_per_week_l744_744352

-- Definitions based on conditions
def crab_buckets_per_day := 8
def crabs_per_bucket := 12
def price_per_crab := 5
def days_per_week := 7

-- The proof goal
theorem Tom_earns_per_week :
  (crab_buckets_per_day * crabs_per_bucket * price_per_crab * days_per_week) = 3360 := by
  sorry

end Tom_earns_per_week_l744_744352


namespace smallest_four_digit_in_pascals_triangle_l744_744748

theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, (binomial n k = 1000 ∧ 999 < 1000 ∧ 1000 < 10000) :=
sorry

end smallest_four_digit_in_pascals_triangle_l744_744748


namespace f_pos_for_all_x_g_le_ax_plus_1_for_a_eq_1_l744_744548

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (x + 1)^2 / 2
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log (x + 1) + Real.exp (-x)

theorem f_pos_for_all_x (x : ℝ) (hx : x > -1) : f x > 0 := by
  sorry

theorem g_le_ax_plus_1_for_a_eq_1 (a : ℝ) (ha : a > 0) : (∀ x : ℝ, -1 < x → g x ≤ a * x + 1) ↔ a = 1 := by
  sorry

end f_pos_for_all_x_g_le_ax_plus_1_for_a_eq_1_l744_744548


namespace convert_to_cylindrical_l744_744841

theorem convert_to_cylindrical :
  ∀ (x y z : ℝ), x = 4 → y = 4 * real.sqrt 3 → z = -3 →
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * real.pi ∧ r = real.sqrt (x^2 + y^2) ∧ tan θ = y / x ∧ (r, θ, z) = (8, real.pi / 3, -3) :=
by
  intros x y z hx hy hz
  subst hx hy hz 
  use (real.sqrt (4^2 + (4 * real.sqrt 3)^2)), (real.pi / 3)
  split
  { norm_num }  -- r > 0
  split
  { norm_num }  -- 0 ≤ θ
  split
  { norm_num [real.pi] }  -- θ < 2π
  split
  { norm_num [real.sqrt] }, -- r = real.sqrt (x^2 + y^2)
  split
  { have : (4 * real.sqrt 3) / 4 = real.sqrt 3 : by norm_num,
  rw [this, real.tan_pi_div_three] }, -- tan θ = y / x
  norm_num
  admit

end convert_to_cylindrical_l744_744841


namespace linear_eq_solution_l744_744940

theorem linear_eq_solution (m : ℤ) (x : ℝ) (h1 : |m| = 1) (h2 : 1 - m ≠ 0) : x = -1/2 :=
by
  sorry

end linear_eq_solution_l744_744940


namespace prove_relationships_l744_744287

variables {p0 p1 p2 p3 : ℝ}
variable (h_p0_pos : p0 > 0)
variable (h_gas_car : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
variable (h_hybrid_car : 10^(5 / 2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)
variable (h_electric_car : p3 = 100 * p0)

theorem prove_relationships : 
  (p1 ≥ p2) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) := by
  split
  sorry

end prove_relationships_l744_744287


namespace dive_score_is_correct_l744_744397

def calculate_dive_score (scores : List ℝ) (difficulty : ℝ) : ℝ :=
  let sorted_scores := scores.sorted
  let trimmed_scores := sorted_scores.drop 1 -- drop the lowest score
  let remaining_scores := trimmed_scores.dropLast 1 -- drop the highest score
  let sum_of_remaining_scores := remaining_scores.sum
  sum_of_remaining_scores * difficulty

theorem dive_score_is_correct :
  calculate_dive_score [7.5, 8.1, 9.0, 6.0, 8.5] 3.2 = 77.12 :=
by
  sorry

end dive_score_is_correct_l744_744397


namespace no_solution_x3_y3_eq_z3_plus_4_mod_9_l744_744366

theorem no_solution_x3_y3_eq_z3_plus_4_mod_9 (x y z : ℤ) : ¬ (x^3 + y^3 ≡ z^3 + 4 [MOD 9]) :=
by
  sorry

end no_solution_x3_y3_eq_z3_plus_4_mod_9_l744_744366


namespace sound_pressures_relationships_l744_744275

variables (p p0 p1 p2 p3 : ℝ)
  (Lp Lpg Lph Lpe : ℝ)

-- The definitions based on the conditions
def sound_pressure_level (p : ℝ) (p0 : ℝ) : ℝ := 20 * (Real.log10 (p / p0))

-- Given conditions
axiom p0_gt_zero : p0 > 0

axiom gasoline_car_levels : 60 ≤ Lpg ∧ Lpg ≤ 90
axiom hybrid_car_levels : 50 ≤ Lph ∧ Lph ≤ 60
axiom electric_car_level : Lpe = 40

axiom gasoline_car_sound_pressure : Lpg = sound_pressure_level p1 p0
axiom hybrid_car_sound_pressure : Lph = sound_pressure_level p2 p0
axiom electric_car_sound_pressure : Lpe = sound_pressure_level p3 p0

-- The proof to be derived
theorem sound_pressures_relationships (p0_gt_zero : p0 > 0)
  (gasoline_car_levels : 60 ≤ Lpg ∧ Lpg ≤ 90)
  (hybrid_car_levels : 50 ≤ Lph ∧ Lph ≤ 60)
  (electric_car_level : Lpe = 40)
  (gasoline_car_sound_pressure : Lpg = sound_pressure_level p1 p0)
  (hybrid_car_sound_pressure : Lph = sound_pressure_level p2 p0)
  (electric_car_sound_pressure : Lpe = sound_pressure_level p3 p0) :
  p1 ≥ p2 ∧ p3 = 100 * p0 ∧ p1 ≤ 100 * p2 :=
by
  sorry

end sound_pressures_relationships_l744_744275


namespace divide_area_into_squares_l744_744084

theorem divide_area_into_squares :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x / y = 4 / 3 ∧ (x^2 + y^2 = 100) ∧ x = 8 ∧ y = 6) := 
by {
  sorry
}

end divide_area_into_squares_l744_744084


namespace donation_B_correct_l744_744315

-- Defining the donations
noncomputable def donation_A : ℕ := 240000
noncomputable def donation_B : ℕ := 250000
noncomputable def donation_C : ℕ := 260000

-- Defining the possible donations
def donations : List ℕ := [donation_A, donation_B, donation_C]

-- Defining the logical statements of A, B, and C
def statement_A (A B C : ℕ) := B ≠ min donation_A (min donation_B donation_C)
def statement_B (A B C : ℕ) := A > C
def statement_C (A B C : ℕ) := (C = min donation_A (min donation_B donation_C)) → A ≠ max donation_A (max donation_B donation_C)

-- Problem statement in Lean
theorem donation_B_correct :
  ∃ (A B C : ℕ),
  A ∈ donations ∧ B ∈ donations ∧ C ∈ donations ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  statement_A A B C ∧ statement_B A B C ∧ statement_C A B C ∧
  B = 260000 :=
by
  sorry

end donation_B_correct_l744_744315


namespace product_even_probability_l744_744678

def C := {1, 2, 3, 4, 5}
def D := {1, 2, 3, 4}

def equally_likely (s : finset ℕ) : ℚ :=
  1 / s.card

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def even_product_probability : ℚ :=
  let outcomes := (C.product D).card
  let even_outcomes := finset.filter (λ (p : ℕ × ℕ), is_even (p.1 * p.2)) (C.product D)
  (even_outcomes.card : ℚ) / outcomes

theorem product_even_probability : 
  even_product_probability = 7 / 10 :=
  sorry

end product_even_probability_l744_744678


namespace imag_unit_binomial_sum_l744_744776

open Real

theorem imag_unit_binomial_sum :
  let i : ℂ := complex.I in
  1 + (6.choose 1) * i + (6.choose 2) * i^2 + (6.choose 3) * i^3 + (6.choose 4) * i^4 + (6.choose 5) * i^5 + (6.choose 6) * i^6 = -8 * i :=
by
  sorry

end imag_unit_binomial_sum_l744_744776


namespace conditional_probability_l744_744519

-- Define the given conditions
variable (A B : Type) [ProbabilitySpace A] [ProbabilitySpace B]
variable (P_B : B → ℝ) (P_AB : A → B → ℝ)

-- Assume the given probabilities
axiom prob_B : P_B B = 1/4
axiom prob_AB : P_AB A B = 1/8

-- Define the conditional probability
def P_A_given_B (P_AB : A → B → ℝ) (P_B : B → ℝ) : ℝ :=
  P_AB A B / P_B B

-- The theorem to prove
theorem conditional_probability : P_A_given_B P_AB P_B = 1/2 :=
by sorry

end conditional_probability_l744_744519


namespace milk_production_l744_744680

theorem milk_production (a b c d e f : ℝ) (h_c: c ≠ 0) (h_ac: a * c ≠ 0) :
  (∀ t : ℝ, t = d * e * (b / (a * c)) + d * f * (b / (2 * a * c))) →
  t = \frac{dbe}{ac} + \frac{dbf}{2ac} :=
by
  intro h
  sorry

end milk_production_l744_744680


namespace enchilada_cost_l744_744763

theorem enchilada_cost : ∃ T E : ℝ, 2 * T + 3 * E = 7.80 ∧ 3 * T + 5 * E = 12.70 ∧ E = 2.00 :=
by
  sorry

end enchilada_cost_l744_744763


namespace sum_of_arguments_of_fifth_roots_l744_744847

-- Define the angle calculation
noncomputable def sum_of_angles : ℤ := 765

theorem sum_of_arguments_of_fifth_roots :
  let z := Complex.ofReal 81 * Complex.I + 1 in
  let fifth_roots := {k : ℤ | 0 ≤ k ∧ k < 5} in
  let angles := {θ | ∃ k ∈ fifth_roots, θ = ((180 / 5) * (1 + 2 * k))} in
  angles.sum = sum_of_angles := by
  sorry

end sum_of_arguments_of_fifth_roots_l744_744847


namespace problem_one_problem_two_problem_three_l744_744612

-- Problem 1: 
-- Prove that if M is a vertex on the short axis of the ellipse, and 
-- triangle MF1F2 is a right-angled triangle, then a = sqrt(2) or a = sqrt(2)/2.
theorem problem_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (M F1 F2 : ℝ × ℝ)
  (h3 : (1:ℝ) = dist M F1) (h4 : (1:ℝ) = dist M F2) 
  (h5 : 2 * (dist F1 F2)^2 = (dist M F1)^2) : a = real.sqrt 2 ∨ a = real.sqrt 2 / 2 :=
sorry

-- Problem 2: 
-- Prove that when k = 1 and triangle OAB is a right-angled triangle with O as the 
-- right-angle vertex, then the relationship between a and m is m^2(a^2 + 1) = 2a^2.
theorem problem_two (a m : ℝ) (h1 : k = 1) (O A B : ℝ × ℝ)
  (h2 : a > 0) (h3 : a ≠ 1) 
  (h4 : is_right_angle O A B) : m^2 * (a^2 + 1) = 2 * a^2 :=
sorry

-- Problem 3:
-- Prove that given a = 2 and k_OA * k_OB = -1/4, the area of triangle OAB 
-- is constant and equals 1.
theorem problem_three (k x1 x2 y1 y2 m : ℝ) (O A B : ℝ × ℝ)
  (h1 : a = 2) (h2 : k_OA * k_OB = -1/4) 
  (h3 : is_right_angle O A B) : 
  let area := ↑1 / 2 :=
area O A B = 1 :=
sorry

end problem_one_problem_two_problem_three_l744_744612


namespace count_three_digit_integers_with_remainder_3_div_7_l744_744161

theorem count_three_digit_integers_with_remainder_3_div_7 :
  ∃ n, (100 ≤ 7 * n + 3 ∧ 7 * n + 3 < 1000) ∧
  ∀ m, (100 ≤ 7 * m + 3 ∧ 7 * m + 3 < 1000) → m - n < 142 - 14 + 1 :=
by
  sorry

end count_three_digit_integers_with_remainder_3_div_7_l744_744161


namespace janet_sculpture_rate_l744_744974

-- Definitions of conditions
def hourly_rate_exterminator : ℝ := 70
def hours_worked_exterminator : ℝ := 20
def sculpture_weight1 : ℝ := 5
def sculpture_weight2 : ℝ := 7
def total_earnings : ℝ := 1640

-- Definition to be proven
theorem janet_sculpture_rate :
  let earnings_exterminator := hourly_rate_exterminator * hours_worked_exterminator in
  let earnings_sculptures := total_earnings - earnings_exterminator in
  let total_sculpture_weight := sculpture_weight1 + sculpture_weight2 in
  earnings_sculptures / total_sculpture_weight = 20 :=
by
  sorry

end janet_sculpture_rate_l744_744974


namespace unique_valid_configuration_l744_744431

-- Define the conditions: a rectangular array of chairs organized in rows and columns such that
-- each row contains the same number of chairs as every other row, each column contains the
-- same number of chairs as every other column, with at least two chairs in every row and column.
def valid_array_configuration (rows cols : ℕ) : Prop :=
  2 ≤ rows ∧ 2 ≤ cols ∧ rows * cols = 49

-- The theorem statement: determine how many valid arrays are possible given the conditions.
theorem unique_valid_configuration : ∃! (rows cols : ℕ), valid_array_configuration rows cols :=
sorry

end unique_valid_configuration_l744_744431


namespace angle_B_triangle_area_proof_l744_744187

noncomputable def triangle_angle 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : (2 * c - a) * Real.cos B = b * Real.cos A) 
  : Prop := B = Real.pi / 3

noncomputable def triangle_area 
  (a b c : ℝ) 
  (B : ℝ := Real.pi / 3)
  (h2 : b = 6) 
  (h3 : c = 2 * a)
  : Prop := 
  let area := 0.5 * a * c * Real.sin B in 
  area = 6 * Real.sqrt 3

-- Statements
theorem angle_B (A B C a b c : ℝ) (h1 : (2 * c - a) * Real.cos B = b * Real.cos A) : triangle_angle A B C a b c h1 := 
  sorry

theorem triangle_area_proof (a b c : ℝ) (h2 : b = 6) (h3 : c = 2 * a) : triangle_area a b c := 
  sorry

end angle_B_triangle_area_proof_l744_744187


namespace contaminated_data_is_6_4_l744_744605

noncomputable def find_contaminated_profit : ℕ → ℕ → List ℝ → ℝ
  | _, _, [] => 0
  | mean_x, mean_y, profit_data =>
    let total_profit := List.sum profit_data
    let expected_total_profit := mean_y * profit_data.length
    expected_total_profit - total_profit

theorem contaminated_data_is_6_4 :
  let profits := [6.0, 6.1, 6.2, 6.0, 6.9, 6.8, 7.1, 7.0]
  let expected_mean_x := 5
  let expected_mean_y := 6.5
  let missing_profit := find_contaminated_profit expected_mean_x expected_mean_y profits
  missing_profit = 6.4 :=
by
  let profits : List ℝ := [6.0, 6.1, 6.2, 6.0, 6.9, 6.8, 7.1, 7.0]
  let expected_mean_x : ℕ := 5
  let expected_mean_y : ℕ := 6.5
  let expected_total_profit := expected_mean_y * (profits.length + 1)
  let total_known_profit := List.sum profits
  let missing_profit := expected_total_profit - total_known_profit
  show missing_profit = 6.4
  sorry

end contaminated_data_is_6_4_l744_744605


namespace probability_dart_in_center_square_l744_744036

noncomputable def hexagon_to_square_probability (a : ℝ) : ℝ :=
  let b := (a * real.sqrt 3) / 2
  let area_square := b ^ 2
  let area_hexagon := (3 * real.sqrt 3 / 2) * (a ^ 2)
  area_square / area_hexagon

theorem probability_dart_in_center_square (a : ℝ) :
  hexagon_to_square_probability a = 1 / (2 * real.sqrt 3) :=
by
  sorry

end probability_dart_in_center_square_l744_744036


namespace number_whose_square_is_64_l744_744707

theorem number_whose_square_is_64 (x : ℝ) (h : x^2 = 64) : x = 8 ∨ x = -8 :=
sorry

end number_whose_square_is_64_l744_744707


namespace highest_time_l744_744321

def freshness_loss (t : ℝ) : ℝ :=
if 0 ≤ t ∧ t < 10 then (t^2) / 1000
else if 10 ≤ t ∧ t ≤ 100 then (1 / 20) * 2^((20 + t) / 30)
else 0

theorem highest_time (t : ℝ) :
  freshness_loss t ≤ 3 / 20 → t ≤ 28 :=
sorry

end highest_time_l744_744321


namespace find_m_l744_744116

noncomputable def arithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ m : ℕ, m ≥ 1 → a m = a 1 + (m - 1) * d

def sumOfFirstNTerms (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  ∀ n : ℕ, S n = n * a 1 + (n * (n - 1))/2 * (a 2 - a 1)

theorem find_m :
  ∃ m : ℕ, m ∈ {2, 3, 4, 5, 6, 7} ∧
  ∀ (a S : ℕ → ℕ),
  (arithmeticSequence a) →
  (sumOfFirstNTerms a S) →
  (S (m-1) = 16) →
  (S m = 25) →
  (a 1 = 1) →
   m = 5 := by
  sorry

end find_m_l744_744116


namespace problem_l744_744634

theorem problem (x y : ℕ) (hy : y > 3) (h : x^2 + y^4 = 2 * ((x-6)^2 + (y+1)^2)) : x^2 + y^4 = 1994 := by
  sorry

end problem_l744_744634


namespace encryption_cycle_length_l744_744312

noncomputable def smallest_cycle_lcm : Nat :=
  Nat.lcm (List.range' 1 33).foldr Nat.lcm 1 -- calculate LCM from 1 to 33

theorem encryption_cycle_length (N : Nat) :
  (∀ (σ : Fin 33 → Fin 33) (x : Fin 33), 
    (∀ i, ∃ k, (∀ m, (Nat.iterate σ (k * m)) x = x)) → N = smallest_cycle_lcm) :=
  sorry

end encryption_cycle_length_l744_744312


namespace line_passes_through_fixed_point_intersection_condition_l744_744550

theorem line_passes_through_fixed_point (k : ℝ) :
  ∃ P : ℝ × ℝ, P = (2, -1) ∧ (∀ x y, k * x - y + 1 - 2 * k = 0 → P = (x, y)) :=
by
  use (2, -1)
  intros x y h
  split
  sorry

theorem intersection_condition (k : ℝ) :
  (∃ A B : ℝ × ℝ, (A.fst > 0 ∧ A.snd = 0) ∧ (B.fst = 0 ∧ B.snd > 0) ∧ ((A.fst ^ 2 + A.snd ^ 2) = (B.fst ^ 2 + B.snd ^ 2)) ∧
  (∀ x y, k * x - y + 1 - 2 * k = 0 → A = (x, 0) → B = (0, -2 * k - 1))) → k = -1 :=
by
  intro h
  sorry

end line_passes_through_fixed_point_intersection_condition_l744_744550


namespace prove_relationships_l744_744285

variables {p0 p1 p2 p3 : ℝ}
variable (h_p0_pos : p0 > 0)
variable (h_gas_car : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
variable (h_hybrid_car : 10^(5 / 2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)
variable (h_electric_car : p3 = 100 * p0)

theorem prove_relationships : 
  (p1 ≥ p2) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) := by
  split
  sorry

end prove_relationships_l744_744285


namespace value_of_x_l744_744934

theorem value_of_x (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := 
by 
  sorry

end value_of_x_l744_744934


namespace check_error_difference_l744_744418

-- Let us define x and y as two-digit natural numbers
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem check_error_difference
    (x y : ℕ)
    (hx : isTwoDigit x)
    (hy : isTwoDigit y)
    (hxy : x > y)
    (h_difference : (100 * y + x) - (100 * x + y) = 2187)
    : x - y = 22 :=
by
  sorry

end check_error_difference_l744_744418


namespace intersection_complement_l744_744147

open Set

noncomputable def U := ℝ
noncomputable def A := {x : ℝ | x^2 + 2 * x < 3}
noncomputable def B := {x : ℝ | x - 2 ≤ 0 ∧ x ≠ 0}

theorem intersection_complement :
  A ∩ -B = {x : ℝ | -3 < x ∧ x ≤ 0} :=
sorry

end intersection_complement_l744_744147


namespace gasoline_car_p_ge_hybrid_electric_car_p_gasoline_car_p_le_100_hybrid_car_p_l744_744281

noncomputable def sound_pressure_level (p p0 : ℝ) : ℝ :=
20 * real.log10 (p / p0)

variables {p0 p1 p2 p3 : ℝ} (h_p0 : p0 > 0)
(h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
(h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)
(h_p3 : p3 = 100 * p0)

theorem gasoline_car_p_ge_hybrid (h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
                                (h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)  : p1 ≥ p2 :=
sorry

theorem electric_car_p (h_p3 : p3 = 100 * p0) : p3 = 100 * p0 :=
sorry

theorem gasoline_car_p_le_100_hybrid_car_p (h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
                                           (h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0) : p1 ≤ 100 * p2 :=
sorry

#check gasoline_car_p_ge_hybrid
#check electric_car_p
#check gasoline_car_p_le_100_hybrid_car_p

end gasoline_car_p_ge_hybrid_electric_car_p_gasoline_car_p_le_100_hybrid_car_p_l744_744281


namespace largest_divisor_of_even_n_cube_difference_l744_744097

theorem largest_divisor_of_even_n_cube_difference (n : ℤ) (h : Even n) : 6 ∣ (n^3 - n) := by
  sorry

end largest_divisor_of_even_n_cube_difference_l744_744097


namespace circus_tent_sections_l744_744729

noncomputable def sections_in_circus_tent (total_capacity : ℕ) (section_capacity : ℕ) : ℕ :=
  total_capacity / section_capacity

theorem circus_tent_sections : sections_in_circus_tent 984 246 = 4 := 
  by 
  sorry

end circus_tent_sections_l744_744729


namespace inverse_proposition_right_triangle_l744_744702

open Real

theorem inverse_proposition_right_triangle {a b c : ℝ} :
  (a^2 + b^2 = c^2) ∧ (a = c / 2) ↔ (b = c * sqrt 3 / 2) ∧ (atan (sqrt 3 / 3) = π / 6) :=
by
  sorry

end inverse_proposition_right_triangle_l744_744702


namespace arnold_numbers_solution_l744_744447

variables (x1 x2 x3 x4 x5 : ℚ)
variables (sums : multiset ℚ)

/-- Given that the set of sums of all pairs of some five numbers is {6, 7, 8, 8, 9, 9, 10, 10, 11, 12}, 
we need to prove that the numbers are {5/2, 7/2, 9/2, 11/2, 13/2}. -/
def arnold_numbers_problem :=
  sums = {6, 7, 8, 8, 9, 9, 10, 10, 11, 12}.to_finset

theorem arnold_numbers_solution (h : arnold_numbers_problem sums) :
  {x1, x2, x3, x4, x5} = {5/2, 7/2, 9/2, 11/2, 13/2} :=
sorry

end arnold_numbers_solution_l744_744447


namespace sin_x_value_sin_2x_plus_pi_over_3_value_l744_744524

-- Definition of the problem with the given condition
def problem1_given := 
  ∃ x : ℝ, 
    cos (x - π / 4) = sqrt 2 / 10 ∧ 
    x > π / 2 ∧ 
    x < 3 * π / 4

-- Proof statement for part (1)
theorem sin_x_value (hx : ∃ x : ℝ, cos (x - π / 4) = sqrt 2 / 10 ∧ x > π / 2 ∧ x < 3 * π / 4) :
  ∃ x : ℝ, sin x = 4 / 5 := 
sorry

-- Additional definition to include solution of part (1)
def problem2_given :=
  ∃ x : ℝ, 
    cos (x - π / 4) = sqrt 2 / 10 ∧ 
    x > π / 2 ∧ 
    x < 3 * π / 4 ∧ 
    sin x = 4 / 5

-- Proof statement for part (2)
theorem sin_2x_plus_pi_over_3_value (hx : ∃ x : ℝ, 
                                      cos (x - π / 4) = sqrt 2 / 10 ∧ 
                                      x > π / 2 ∧ 
                                      x < 3 * π / 4 ∧ 
                                      sin x = 4 / 5 ) :
   ∃ x : ℝ, sin (2 * x + π / 3) = - (24 + 7 * sqrt 3) / 50 := 
sorry

end sin_x_value_sin_2x_plus_pi_over_3_value_l744_744524


namespace tank_capacity_l744_744808

theorem tank_capacity (V : ℝ) (initial_fraction final_fraction : ℝ) (added_water : ℝ)
  (h1 : initial_fraction = 1 / 4)
  (h2 : final_fraction = 3 / 4)
  (h3 : added_water = 208)
  (h4 : final_fraction - initial_fraction = 1 / 2)
  (h5 : (1 / 2) * V = added_water) :
  V = 416 :=
by
  -- Given: initial_fraction = 1/4, final_fraction = 3/4, added_water = 208
  -- Difference in fullness: 1/2
  -- Equation for volume: 1/2 * V = 208
  -- Hence, V = 416
  sorry

end tank_capacity_l744_744808


namespace crayons_difference_l744_744661

def initial_crayons : ℕ := 8597
def crayons_given : ℕ := 7255
def crayons_lost : ℕ := 3689

theorem crayons_difference : crayons_given - crayons_lost = 3566 := by
  sorry

end crayons_difference_l744_744661


namespace find_bc_find_area_l744_744241

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Condition 1: From the given problem
variable (h1 : (b^2 + c^2 - a^2) / (cos A) = 2)

-- Condition 2: From the given problem
variable (h2 : (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1)

theorem find_bc : bc = 1 :=
sorry

theorem find_area (area_abc : ℝ) : area_abc = (sqrt 3) / 4 :=
sorry

end find_bc_find_area_l744_744241


namespace geom_seq_limit_l744_744883

theorem geom_seq_limit (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) 
  (h1 : ∀ n, a_n n = a_n 1 * (1 / 2)^(n - 1))
  (h2 : a_n 2 = 1)
  (h3 : S_n 2 = 3) :
  (∃ L, tendsto S_n at_top (𝓝 4)) :=
sorry

end geom_seq_limit_l744_744883


namespace chord_length_l744_744389

theorem chord_length (d : ℝ) (M : ℝ) : 
  (∃ (A B : ℝ), A < M ∧ M < B ∧ A + B = d) ∧
  (let remaining_area := 16 * (Real.pi ^ 3),
       total_area := (Real.pi * d ^ 2) / 8,
       cutout_area := 2 * (Real.pi * (d / 4) ^ 2 / 2),
       remaining_area_result := total_area - cutout_area
   in remaining_area = remaining_area_result) 
  → (2 * (d / 2 / sqrt(2)) = d * sqrt(2)) := 
sorry

end chord_length_l744_744389


namespace number_of_solutions_l744_744492

def sign (a : ℝ) : ℝ :=
if a > 0 then 1 else
if a = 0 then 0 else
-1

theorem number_of_solutions : 
  { 
    (x, y, z : ℝ) |
    x = 2023 - 2024 * sign (y^2 - z^2) ∧
    y = 2023 - 2024 * sign (x^2 - z^2) ∧
    z = 2023 - 2024 * sign (x^2 - y^2)
  }.card = 3 := 
sorry

end number_of_solutions_l744_744492


namespace baguettes_sold_after_second_batch_l744_744687

theorem baguettes_sold_after_second_batch
  (total_baguettes : ℕ)
  (sold_after_first_batch : ℕ)
  (sold_after_third_batch : ℕ)
  (baguettes_left : ℕ)
  (total_baguettes = 144)
  (sold_after_first_batch = 37)
  (sold_after_third_batch = 49)
  (baguettes_left = 6) :
  total_baguettes - baguettes_left - sold_after_first_batch - sold_after_third_batch = 52 :=
by
  sorry

end baguettes_sold_after_second_batch_l744_744687


namespace ratio_problem_l744_744554

theorem ratio_problem
  (w x y z : ℝ)
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 2 / 3)
  (h3 : w / z = 3 / 5) :
  (x + y) / z = 27 / 10 :=
by
  sorry

end ratio_problem_l744_744554


namespace ratio_of_inscribed_squares_l744_744806

theorem ratio_of_inscribed_squares (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (hx : x = 60 / 17) (hy : y = 3) :
  x / y = 20 / 17 :=
by
  sorry

end ratio_of_inscribed_squares_l744_744806


namespace find_b_l744_744583

theorem find_b (b : ℚ) (h : ∃ c : ℚ, (3 * x + c)^2 = 9 * x^2 + 27 * x + b) : b = 81 / 4 := 
sorry

end find_b_l744_744583


namespace cos_double_angle_example_l744_744570

def cos_double_angle_identity (x : ℝ) : Prop :=
  cos (2 * x) = 1 - 2 * (sin x) ^ 2

theorem cos_double_angle_example : cos_double_angle_identity (x : ℝ) 
  (h : sin x = - 2 / 3) : cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_example_l744_744570


namespace intersection_A_B_l744_744119

def A := { x : ℝ | -5 < x ∧ x < 2 }
def B := { x : ℝ | x^2 - 9 < 0 }
def AB := { x : ℝ | -3 < x ∧ x < 2 }

theorem intersection_A_B : A ∩ B = AB := by
  sorry

end intersection_A_B_l744_744119


namespace ceil_x_squared_values_count_l744_744173

theorem ceil_x_squared_values_count (x : ℝ) (h : ⌈x⌉ = 14) : 
  (finset.card ((finset.range (196 + 1)).filter (λ n, 170 ≤ n))) = 27 := 
  sorry

end ceil_x_squared_values_count_l744_744173


namespace range_of_m_l744_744900

variable {f : ℝ → ℝ}
variable {a : ℝ}
variable {m : ℝ}

def is_even_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x ∈ I, f(x) = f(-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x ≤ y → f(x) ≤ f(y)

theorem range_of_m
  (h1 : is_even_on f (set.Icc (2 - a) 3))
  (h2 : is_monotonically_increasing_on f (set.Icc 0 3))
  (h3 : f(-m^2 - a / 5) > f(-m^2 + 2 * m - 2)) :
  (1 / 2) < m ∧ m ≤ real.sqrt 2 :=
by
  sorry

end range_of_m_l744_744900


namespace range_of_lambda_l744_744122

variable {i j : ℝ → ℝ} -- Define vector space variables

noncomputable def is_unit_vector (v : ℝ → ℝ) : Prop := 
  ∥v∥ = 1

noncomputable def is_perpendicular (v w : ℝ → ℝ) : Prop := 
  v ⬝ w = 0

noncomputable def acute_angle (a b : ℝ → ℝ) : Prop := 
  a ⬝ b > 0

variable (a b : ℝ → ℝ)
variable (λ : ℝ)

theorem range_of_lambda 
    (h1 : is_unit_vector i) 
    (h2 : is_unit_vector j) 
    (h3 : is_perpendicular i j)
    (h4 : a = i - 2 • j)
    (h5 : b = i + λ • j)
    (h6 : acute_angle a b) :
    λ ∈ set.Iio (-2) ∪ set.Ioo (-2) (1 / 2) := sorry

end range_of_lambda_l744_744122


namespace equilateral_triangle_l744_744624

namespace TriangleProof

variables {A B C F E : Type} [euclidean_geometry A B C F E]

-- Definitions of the triangle and medians
def is_median (A B C F E : Type) [euclidean_geometry A B C F E] := sorry

-- Given conditions in the problem
def given_conditions (A B C F E : Type) [euclidean_geometry A B C F E] : Prop :=
  is_median A B C F E ∧
  angle BAF = 30 ∧
  angle BCE = 30

-- Statement to prove the triangle is equilateral
theorem equilateral_triangle (A B C F E : Type) [euclidean_geometry A B C F E]
  (h : given_conditions A B C F E) : equilateral A B C :=
sorry

end TriangleProof

end equilateral_triangle_l744_744624


namespace equation_solution_l744_744402

theorem equation_solution (x y : ℕ) :
  (x^2 + 1)^y - (x^2 - 1)^y = 2 * x^y ↔ 
  (x = 1 ∧ y = 1) ∨ (x = 0 ∧ ∃ k : ℕ, y = 2 * k ∧ k > 0) :=
by sorry

end equation_solution_l744_744402


namespace angle_between_vectors_perpendicular_vector_l744_744150

-- Define vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 4)

-- The angle between vectors a + b and a - b
theorem angle_between_vectors : 
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  let a_minus_b := (a.1 - b.1, a.2 - b.2)
  ∀ θ : ℝ, cos θ = ((a_plus_b.1 * a_minus_b.1 + a_plus_b.2 * a_minus_b.2) / 
  (real.sqrt (a_plus_b.1^2 + a_plus_b.2^2) * real.sqrt (a_minus_b.1^2 + a_minus_b.2^2)))
  → θ = 3 * real.pi / 4 :=
by sorry

-- If vector a is perpendicular to (a + λb), then λ = -1
theorem perpendicular_vector (λ : ℝ) : 
  ∀ λ : ℝ, a.1 * (a.1 + λ * b.1) + a.2 * (a.2 + λ * b.2) = 0 
  → λ = -1 :=
by sorry

end angle_between_vectors_perpendicular_vector_l744_744150


namespace cos_double_angle_example_l744_744571

def cos_double_angle_identity (x : ℝ) : Prop :=
  cos (2 * x) = 1 - 2 * (sin x) ^ 2

theorem cos_double_angle_example : cos_double_angle_identity (x : ℝ) 
  (h : sin x = - 2 / 3) : cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_example_l744_744571


namespace bc_is_one_area_of_triangle_l744_744236

-- Define a triangle with sides a, b, and c and corresponding angles A, B, and C.
structure Triangle :=
(a b c A B C : ℝ)

-- Assume the Law of Cosines condition is given.
axiom Law_of_Cosines_condition (T : Triangle) : (T.b^2 + T.c^2 - T.a^2) / (Real.cos T.A) = 2

-- Prove that bc = 1.
theorem bc_is_one (T : Triangle) (h : Law_of_Cosines_condition T) : T.b * T.c = 1 := 
by {
    sorry
}

-- Assume the given ratio condition.
axiom ratio_condition (T : Triangle) :
  (T.a * Real.cos T.B - T.b * Real.cos T.A) / (T.a * Real.cos T.B + T.b * Real.cos T.A)
  - (T.b / T.c) = 1

-- Prove that the area of the triangle is sqrt(3)/4.
theorem area_of_triangle (T : Triangle) (h1 : Law_of_Cosines_condition T) (h2 : ratio_condition T) :
  let S := 0.5 * T.b * T.c * Real.sin T.A
  in S = Real.sqrt 3 / 4 :=
by {
    sorry
}

end bc_is_one_area_of_triangle_l744_744236


namespace choose_officers_l744_744784

-- Definitions from conditions
parameter Members : Type
parameter (Alice Bob : Members)
parameter (isMember : Members → Prop)
axiom all_members : ∀ m, isMember m
axiom num_members : (finset.univ.filter isMember).card = 30
-- Officer Positions
inductive Position
| president : Position
| vice_president : Position
| secretary : Position
| treasurer : Position

-- (Question, Conditions, Correct Answer)
theorem choose_officers (h : Alice ≠ Bob) : 
  let N := (num_members - 2) * (num_members - 3) * (num_members - 4) * (num_members - 5) + 
  (choose 4 2) * 2 * (num_members - 2) * (num_members - 3) 
in N = 500472 :=
begin
  sorry
end

end choose_officers_l744_744784


namespace intersection_complement_l744_744121

open Set

variable (R : Set ℝ) (M N : Set ℝ)

noncomputable definition set_M : Set ℝ := {x | x^2 - 2*x < 0}
noncomputable definition set_N : Set ℝ := {x | 1 ≤ x}

theorem intersection_complement (R : Set ℝ) :
  (M = {x | x^2 - 2 * x < 0}) → (N = {x | 1 ≤ x}) →
  M ∩ (R \ N) = {x | 0 < x ∧ x < 1} :=
begin
  sorry
end

end intersection_complement_l744_744121


namespace ravi_nickels_l744_744671

variables (n q d : ℕ)

-- Defining the conditions
def quarters (n : ℕ) : ℕ := n + 2
def dimes (q : ℕ) : ℕ := q + 4

-- Using these definitions to form the Lean theorem
theorem ravi_nickels : 
  ∃ n, q = quarters n ∧ d = dimes q ∧ 
  (0.05 * n + 0.25 * q + 0.10 * d : ℝ) = 3.50 ∧ n = 6 :=
sorry

end ravi_nickels_l744_744671


namespace students_interested_in_both_l744_744189

theorem students_interested_in_both (A B C Total : ℕ) (hA : A = 35) (hB : B = 45) (hC : C = 4) (hTotal : Total = 55) :
  A + B - 29 + C = Total :=
by
  -- Assuming the correct answer directly while skipping the proof.
  sorry

end students_interested_in_both_l744_744189


namespace sum_seven_consecutive_integers_l744_744719

theorem sum_seven_consecutive_integers (n : ℕ) : 
  ∃ k : ℕ, (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) = 7 * k := 
by 
  -- Use sum of integers and factor to demonstrate that the sum is multiple of 7
  sorry

end sum_seven_consecutive_integers_l744_744719


namespace even_function_has_specific_m_l744_744595

theorem even_function_has_specific_m (m : ℝ) (f : ℝ → ℝ) (h_def : ∀ x : ℝ, f x = x^2 + (m - 1) * x - 3) (h_even : ∀ x : ℝ, f x = f (-x)) :
  m = 1 :=
by
  sorry

end even_function_has_specific_m_l744_744595


namespace symmetric_axis_cos_transformed_l744_744700

theorem symmetric_axis_cos_transformed :
  ∀ (x : ℝ), (∃ k : ℤ, 4 * x - π / 10 = k * π) → x = π / 40 :=
by
  intro x h
  cases h with k hk
  have : x = (k * π + π / 10) / 4 := by linarith
  have k_eq_zero : k = 0 := sorry -- Assume we have a proof here
  rw [k_eq_zero, zero_mul, add_zero, div_eq_mul_one_div, π, one_mul, div_self, div_div_eq_div_mul, mul_one] at this
  exact this

end symmetric_axis_cos_transformed_l744_744700


namespace equation_of_curve_C_equations_of_line_l_l744_744107

def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ → Prop := λ P, (P.1 + 1)^2 + P.2^2 = 2 * ((P.1 - 1)^2 + P.2^2)

theorem equation_of_curve_C (x y : ℝ) (h : C (x, y)) :
  x^2 + y^2 - 6*x + 1 = 0 :=
sorry

theorem equations_of_line_l (x y : ℝ) (h1 : x = 1 ∨ y = 2) (h2 : (x-3)^2 + y^2 = 8) 
  (h3 : (x - 1)^2 + (y - 2)^2 = 16) :
  (x = 1 ∨ y = 2) :=
sorry

end equation_of_curve_C_equations_of_line_l_l744_744107


namespace find_a_l744_744205

theorem find_a (a : ℝ) (ha : 0 < a) :
  (∀ (ρ θ : ℝ), ρ * cos θ + sqrt 3 * ρ * sin θ + 1 = 0 →
    ρ = 2 * a * cos θ) →
  a = 1 :=
by
  sorry

end find_a_l744_744205


namespace sound_pressure_proof_l744_744270

noncomputable theory

def sound_pressure_level (p p0 : ℝ) : ℝ :=
  20 * real.log10 (p / p0)

variables (p0 : ℝ) (p0_pos : 0 < p0)
variables (p1 p2 p3 : ℝ)

def gasoline_car (Lp : ℝ) : Prop :=
  60 <= Lp ∧ Lp <= 90

def hybrid_car (Lp : ℝ) : Prop :=
  50 <= Lp ∧ Lp <= 60

def electric_car (Lp : ℝ) : Prop :=
  Lp = 40

theorem sound_pressure_proof :
  gasoline_car (sound_pressure_level p1 p0) ∧
  hybrid_car (sound_pressure_level p2 p0) ∧
  electric_car (sound_pressure_level p3 p0) →
  (p1 ≥ p2) ∧ (¬ (p2 > 10 * p3)) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) :=
begin
  sorry
end

end sound_pressure_proof_l744_744270


namespace ratio_doubled_to_original_l744_744429

theorem ratio_doubled_to_original (x y : ℕ) (h1 : y = 2 * x + 9) (h2 : 3 * y = 57) : 2 * x = 2 * (x / 1) := 
by sorry

end ratio_doubled_to_original_l744_744429


namespace double_root_possible_values_l744_744043

theorem double_root_possible_values (b_3 b_2 b_1 : ℤ) (s : ℤ)
  (h : (Polynomial.X - Polynomial.C s) ^ 2 ∣
    Polynomial.C 24 + Polynomial.C b_1 * Polynomial.X + Polynomial.C b_2 * Polynomial.X ^ 2 + Polynomial.C b_3 * Polynomial.X ^ 3 + Polynomial.X ^ 4) :
  s = -2 ∨ s = -1 ∨ s = 1 ∨ s = 2 :=
sorry

end double_root_possible_values_l744_744043


namespace trajectory_of_M_line_and_area_of_POM_l744_744521

def point := (ℝ × ℝ)
def line := ℝ → ℝ

-- Conditions
def P : point := (2, 2)
def O : point := (0, 0)
def C : point → Prop := λ (x, y), x^2 + y^2 - 8*y = 0

-- Proof for (1): Trajectory of M
theorem trajectory_of_M :
  ∀ M : point, 
  (∃ A B : point, line (λ x, y) ∧ point_on_circle A ∧ point_on_circle B ∧ midpoint A B = M) →
  (x - 1)^2 + (y - 3)^2 = 2 :=
sorry

-- Proof for (2): Equation of line and area of triangle POM
theorem line_and_area_of_POM :
  ∀ M : point, 
  ( (x - 1)^2 + (y - 3)^2 = 2 ∧ |distance O P = distance O M| ) →
  (line_eq : ∃ l : line, ∀ x, l x = - (1/3) * (x - 2) + 2 ∧ area O P M = 16 / 5) :=
sorry

end trajectory_of_M_line_and_area_of_POM_l744_744521


namespace compute_m_n_sum_l744_744774

theorem compute_m_n_sum :
  let AB := 10
  let BC := 15
  let height := 30
  let volume_ratio := 9
  let smaller_base_AB := AB / 3
  let smaller_base_BC := BC / 3
  let diagonal_AC := Real.sqrt (AB^2 + BC^2)
  let smaller_diagonal_A'C' := Real.sqrt ((smaller_base_AB)^2 + (smaller_base_BC)^2)
  let y_length := 145 / 9   -- derived from geometric considerations
  let YU := 20 + y_length
  let m := 325
  let n := 9
  YU = m / n ∧ Nat.gcd m n = 1 ∧ m + n = 334 :=
  by
  sorry

end compute_m_n_sum_l744_744774


namespace cubic_sum_l744_744897

theorem cubic_sum :
  let f := λ x : ℝ, -x^3 + 3*x^2 in
  (∑ i in (Finset.range 4045).filter (λ i, i % 2023 ≠ 0), f (i / 2023)) = 8090 :=
by sorry

end cubic_sum_l744_744897


namespace max_value_of_quadratic_l744_744844

theorem max_value_of_quadratic :
  ∀ s : ℝ, ∃ (M : ℝ), is_maximum M (λ s, -7 * s^2 + 56 * s - 18) M := 
begin
  intro s,
  use 94,
  sorry
end

end max_value_of_quadratic_l744_744844


namespace cos_double_angle_l744_744579

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l744_744579


namespace prime_quadratic_residues_l744_744081

theorem prime_quadratic_residues (p : ℕ) [prime p] (h : ∀ k, k > 0 ∧ k ≤ p → is_quadratic_residue (2 * (p / k) - 1) p) : p = 2 := 
sorry

end prime_quadratic_residues_l744_744081


namespace power_function_value_at_fixed_point_l744_744699

theorem power_function_value_at_fixed_point 
  (a : ℝ) (h_positive : a > 0) (h_not_one : a ≠ 1) 
  (P : ℝ × ℝ) (h_P : P = (2, sqrt 2 / 2)) :
  ∃ α : ℝ, 
    (∀ x : ℝ, f x = x ^ α) → 
    (f 9 = 1 / 3) :=
by
  sorry

end power_function_value_at_fixed_point_l744_744699


namespace toms_weekly_revenue_l744_744348

def crabs_per_bucket : Nat := 12
def number_of_buckets : Nat := 8
def price_per_crab : Nat := 5
def days_per_week : Nat := 7

theorem toms_weekly_revenue :
  (crabs_per_bucket * number_of_buckets * price_per_crab * days_per_week) = 3360 :=
by
  sorry

end toms_weekly_revenue_l744_744348


namespace isosceles_triangle_count_l744_744192

-- Definition of the vertices and division points
noncomputable def A : ℝ² := (0, 0)
noncomputable def B : ℝ² := (1, 0)
noncomputable def C : ℝ² := (0.5, (3.sqrt / 2))

-- Points dividing AB, AC, BC into 4 equal segments
noncomputable def M : ℝ² := (0.25, 0)
noncomputable def N : ℝ² := (0.75, 0)
noncomputable def P : ℝ² := (0.125, (3.sqrt / 8))
noncomputable def Q : ℝ² := (0.375, (3.sqrt / 8))
noncomputable def K : ℝ² := (0.625, (3.sqrt / 4))
noncomputable def L : ℝ² := (0.875, (3.sqrt / 4))

-- The proof problem
theorem isosceles_triangle_count :
  count_isosceles_triangles {A, B, C, M, N, P, Q, K, L} = 18 :=
sorry

end isosceles_triangle_count_l744_744192


namespace amanda_final_notebooks_l744_744057

noncomputable def initial_notebooks : ℕ := 65
noncomputable def ordered_notebooks : ℕ := 23
noncomputable def loss_percentage : ℝ := 0.15
noncomputable def total_notebooks_before_loss := initial_notebooks + ordered_notebooks
noncomputable def notebooks_lost := (loss_percentage * total_notebooks_before_loss.to_real).floor.to_nat
noncomputable def final_notebooks := total_notebooks_before_loss - notebooks_lost

theorem amanda_final_notebooks : final_notebooks = 75 :=
by
  rw [initial_notebooks, ordered_notebooks, loss_percentage, total_notebooks_before_loss, notebooks_lost, final_notebooks]
  sorry

end amanda_final_notebooks_l744_744057


namespace proof_problem_l744_744329

noncomputable def otimes (a b : ℝ) : ℝ := a^3 / b^2

theorem proof_problem : ((otimes (otimes 2 3) 4) - otimes 2 (otimes 3 4)) = -224/81 :=
by
  sorry

end proof_problem_l744_744329


namespace quadratic_inequality_solution_l744_744683

theorem quadratic_inequality_solution (a b: ℝ) (h1: ∀ x: ℝ, 1 < x ∧ x < 2 → ax^2 + bx - 4 > 0) (h2: ∀ x: ℝ, x ≤ 1 ∨ x ≥ 2 → ax^2 + bx - 4 ≤ 0) : a + b = 4 :=
sorry

end quadratic_inequality_solution_l744_744683


namespace find_ratio_of_sides_l744_744208

variable {A B : ℝ}
variable {a b : ℝ}

-- Given condition
axiom given_condition : a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = a * Real.sqrt 3

-- Theorem we need to prove
theorem find_ratio_of_sides (h : a ≠ 0) : b / a = Real.sqrt 3 / 3 :=
by
  sorry

end find_ratio_of_sides_l744_744208


namespace courtyard_length_l744_744733

/-- Given the following conditions:
  1. The width of the courtyard is 16.5 meters.
  2. 66 paving stones are required.
  3. Each paving stone measures 2.5 meters by 2 meters.
  Prove that the length of the rectangular courtyard is 20 meters. -/
theorem courtyard_length :
  ∃ L : ℝ, L = 20 ∧ 
           (∃ W : ℝ, W = 16.5) ∧ 
           (∃ n : ℕ, n = 66) ∧ 
           (∃ A : ℝ, A = 2.5 * 2) ∧
           n * A = L * W :=
by
  sorry

end courtyard_length_l744_744733


namespace find_ages_l744_744267

def product_of_digits (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def family_ages (father mother daughter son : ℕ) : Prop :=
  (sqrt father) * (sqrt father) = father ∧         -- father's age is a perfect square
  product_of_digits father = mother ∧              -- product of father's digits equals mother's age
  sum_of_digits father = daughter ∧                -- sum of father's digits equals daughter's age
  sum_of_digits mother = son                       -- sum of mother's digits equals son's age

theorem find_ages : ∃ father mother daughter son,
  family_ages father mother daughter son ∧
  father = 49 ∧ mother = 36 ∧ daughter = 13 ∧ son = 9 :=
by {
  use 49,
  use 36,
  use 13,
  use 9,
  split,
  {
    unfold family_ages,
    split,
    {
      sorry, -- prove 49 is a perfect square
    },
    split,
    {
      sorry, -- prove product of digits of 49 equals 36
    },
    split,
    {
      sorry, -- prove sum of digits of 49 equals 13
    },
    {
      sorry, -- prove sum of digits of 36 equals 9
    }
  },
  split,
  refl,
  split,
  refl,
  split,
  refl,
  refl
}

end find_ages_l744_744267


namespace target_more_tools_l744_744739

theorem target_more_tools (walmart_tools : ℕ) (target_tools : ℕ) (walmart_tools_is_6 : walmart_tools = 6) (target_tools_is_11 : target_tools = 11) :
  target_tools - walmart_tools = 5 :=
by
  rw [walmart_tools_is_6, target_tools_is_11]
  exact rfl

end target_more_tools_l744_744739


namespace sum_of_inradii_l744_744603

-- Definitions of the lengths in the triangle ABC
def AB := 7
def AC := 9
def BC := 12

-- Midpoint D of BC
def D_is_midpoint := BC / 2

-- Inradii of triangles ADB and ADC
def r_inradius_ADB := sorry -- radius of the inscribed circle in triangle ADB
def r_inradius_ADC := sorry -- radius of the inscribed circle in triangle ADC

-- Sum of the inradii is 4.25
theorem sum_of_inradii : r_inradius_ADB + r_inradius_ADC = 4.25 := sorry

end sum_of_inradii_l744_744603


namespace problem_statement_l744_744905

noncomputable def sum_a (n : ℕ) : ℝ := 
  if h : n > 0 then ∑ i in Finset.range n, (Real.log10 i - Real.log10 (i + 1)) else 0

theorem problem_statement : sum_a 1000 = -3 := 
by sorry

end problem_statement_l744_744905


namespace students_at_end_of_year_l744_744957

def n_start := 10
def n_left := 4
def n_new := 42

theorem students_at_end_of_year : n_start - n_left + n_new = 48 := by
  sorry

end students_at_end_of_year_l744_744957


namespace sunflowers_count_l744_744305

theorem sunflowers_count (n d : ℕ) (hnd : n = 12) (hdaises : d = 2) :
  let remaining := n - d in
  let tulips := (3 * remaining) / 5 in
  let sunflowers := remaining - tulips in
  sunflowers = 4 :=
by
  sorry

end sunflowers_count_l744_744305


namespace polar_line_eq_l744_744204

theorem polar_line_eq (rho theta : ℝ) (h₁ : (rho = 2 ∧ theta = π / 2)) (h₂ : parallel_polar_axis (line_through_point rho theta)) :
  polar_equation_of_line rho theta = ρ * sin θ := 
sorry

end polar_line_eq_l744_744204


namespace min_operations_to_all_ones_l744_744804

theorem min_operations_to_all_ones (p : ℕ) (seq : Fin p → ℤ) (h : ∀ i, seq i = 1 ∨ seq i = -1) :
  seq → ℕ :=
best_op_count : ℕ := ⌈(p + 1)/2⌉
begin
  sorry
end

end min_operations_to_all_ones_l744_744804


namespace like_terms_solutions_l744_744568

theorem like_terms_solutions (x y : ℤ) (h1 : 5 = 4 * x + 1) (h2 : 3 * y = 6) :
  x = 1 ∧ y = 2 := 
by 
  -- proof goes here
  sorry

end like_terms_solutions_l744_744568


namespace linear_relation_is_correct_maximum_profit_l744_744056

-- Define the given data points
structure DataPoints where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

-- Define the given conditions
def conditions : DataPoints := ⟨50, 100, 60, 90⟩

-- Define the cost and sell price range conditions
def cost_per_kg : ℝ := 20
def max_selling_price : ℝ := 90

-- Define the linear relationship function
def linear_relationship (k b x : ℝ) : ℝ := k * x + b

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - cost_per_kg) * (linear_relationship (-1) 150 x)

-- Statements to Prove
theorem linear_relation_is_correct (k b : ℝ) :
  linear_relationship k b 50 = 100 ∧
  linear_relationship k b 60 = 90 →
  (b = 150 ∧ k = -1) := by
  intros h
  sorry

theorem maximum_profit :
  ∃ x : ℝ, 20 ≤ x ∧ x ≤ max_selling_price ∧ profit_function x = 4225 := by
  use 85
  sorry

end linear_relation_is_correct_maximum_profit_l744_744056


namespace find_admission_score_l744_744608

noncomputable def admission_score : ℝ := 87

theorem find_admission_score :
  ∀ (total_students admitted_students not_admitted_students : ℝ) 
    (admission_score admitted_avg not_admitted_avg overall_avg : ℝ),
    admitted_students = total_students / 4 →
    not_admitted_students = 3 * admitted_students →
    admitted_avg = admission_score + 10 →
    not_admitted_avg = admission_score - 26 →
    overall_avg = 70 →
    total_students * overall_avg = 
    (admitted_students * admitted_avg + not_admitted_students * not_admitted_avg) →
    admission_score = 87 :=
by
  intros total_students admitted_students not_admitted_students 
         admission_score admitted_avg not_admitted_avg overall_avg
         h1 h2 h3 h4 h5 h6
  sorry

end find_admission_score_l744_744608


namespace number_of_real_roots_l744_744499

theorem number_of_real_roots (a : ℝ) :
  (|a| < (2 * Real.sqrt 3 / 9) → ∃ x y z : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ z^3 - z - a = 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  (|a| = (2 * Real.sqrt 3 / 9) → ∃ x y : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ x = y) ∧
  (|a| > (2 * Real.sqrt 3 / 9) → ∃ x : ℝ, x^3 - x - a = 0 ∧ ∀ y : ℝ, y^3 - y - a ≠ 0 ∨ y = x) :=
sorry

end number_of_real_roots_l744_744499


namespace final_building_height_l744_744472

noncomputable def height_of_final_building 
    (Crane1_height : ℝ)
    (Building1_height : ℝ)
    (Crane2_height : ℝ)
    (Building2_height : ℝ)
    (Crane3_height : ℝ)
    (Average_difference : ℝ) : ℝ :=
    Crane3_height / (1 + Average_difference)

theorem final_building_height
    (Crane1_height : ℝ := 228)
    (Building1_height : ℝ := 200)
    (Crane2_height : ℝ := 120)
    (Building2_height : ℝ := 100)
    (Crane3_height : ℝ := 147)
    (Average_difference : ℝ := 0.13)
    (HCrane1 : 1 + (Crane1_height - Building1_height) / Building1_height = 1.14)
    (HCrane2 : 1 + (Crane2_height - Building2_height) / Building2_height = 1.20)
    (HAvg : (1.14 + 1.20) / 2 = 1.13) :
    height_of_final_building Crane1_height Building1_height Crane2_height Building2_height Crane3_height Average_difference = 130 := 
sorry

end final_building_height_l744_744472


namespace gcf_150_225_300_l744_744368

def prime_factors_150 := {2, 3, 5}
def prime_factors_225 := {3, 5}
def prime_factors_300 := {2, 3, 5}

def gcf (a b c : Nat) : Nat :=
  have gcd_ab := Nat.gcd a b
  Nat.gcd gcd_ab c

theorem gcf_150_225_300 : gcf 150 225 300 = 75 := by
  sorry

end gcf_150_225_300_l744_744368


namespace train_speed_l744_744356

theorem train_speed (L1 L2: ℕ) (V2: ℕ) (T: ℕ) (V1: ℕ) : 
  L1 = 120 -> 
  L2 = 280 -> 
  V2 = 30 -> 
  T = 20 -> 
  (L1 + L2) * 18 = (V1 + V2) * T * 100 -> 
  V1 = 42 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end train_speed_l744_744356


namespace determine_word_meaning_determine_humanity_l744_744658

-- Definitions of conditions
def asked_question : Prop := "Does 'бал' mean 'yes'?"
def received_answer : Prop := "бал"

-- Problem statement: proving the conclusions given the conditions
theorem determine_word_meaning : 
    (asked_question → received_answer) → 
    (¬(∀ (meaning : string), meaning = "yes" ∨ meaning = "no")) :=
by
  intro h
  sorry

theorem determine_humanity :
    (asked_question → received_answer) →
    (¬ ∀ (type : string), type = "zombie") :=
by
  intro h
  sorry

end determine_word_meaning_determine_humanity_l744_744658


namespace sam_total_spent_l744_744288

-- Define the values of a penny and a dime in dollars
def penny_value : ℝ := 0.01
def dime_value : ℝ := 0.10

-- Define what Sam spent
def friday_spent : ℝ := 2 * penny_value
def saturday_spent : ℝ := 12 * dime_value

-- Define total spent
def total_spent : ℝ := friday_spent + saturday_spent

theorem sam_total_spent : total_spent = 1.22 := 
by
  -- The following is a placeholder for the actual proof
  sorry

end sam_total_spent_l744_744288


namespace problem_1_problem_2_l744_744922

def coords_of_vector_c (a : ℝ × ℝ) (c : ℝ × ℝ) : Prop :=
  let norm_c := (c.1^2 + c.2^2) ^ (1/2)
  ∧ norm_c = 2 * Real.sqrt 5
  ∧ c = (-2, -4)

def proj_of_a_on_b (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_b := (b.1^2 + b.2^2) ^ (1/2)
  ∧ norm_b = Real.sqrt (5 / 4)
  ∧ (a.1 + 2 * b.1) * (2 * a.1 - b.1) + (a.2 + 2 * b.2) * (2 * a.2 - b.2) = 15 / 4
  ∧ dot_product = -5 / 12
  ∧ (dot_product / norm_b) = -Real.sqrt 5 / 3

theorem problem_1 (a : ℝ × ℝ) (c : ℝ × ℝ) : a = (1, 2) ∧ coords_of_vector_c a c := do sorry

theorem problem_2 (a : ℝ × ℝ) (b : ℝ × ℝ) : a = (1, 2) ∧ proj_of_a_on_b a b := do sorry

end problem_1_problem_2_l744_744922


namespace variance_decreases_l744_744332

def scores_initial := [5, 9, 7, 10, 9] -- Initial 5 shot scores
def additional_shot := 8 -- Additional shot score

-- Given variance of initial scores
def variance_initial : ℝ := 3.2

-- Placeholder function to calculate variance of a list of scores
noncomputable def variance (scores : List ℝ) : ℝ := sorry

-- Definition of the new scores list
def scores_new := scores_initial ++ [additional_shot]

-- Define the proof problem
theorem variance_decreases :
  variance scores_new < variance_initial :=
sorry

end variance_decreases_l744_744332


namespace students_participated_l744_744310

theorem students_participated (like_dislike_sum : 383 + 431 = 814) : 
  383 + 431 = 814 := 
by exact like_dislike_sum

end students_participated_l744_744310


namespace members_playing_both_badminton_and_tennis_l744_744009

-- Definitions based on conditions
def N : ℕ := 35  -- Total number of members in the sports club
def B : ℕ := 15  -- Number of people who play badminton
def T : ℕ := 18  -- Number of people who play tennis
def Neither : ℕ := 5  -- Number of people who do not play either sport

-- The theorem based on the inclusion-exclusion principle
theorem members_playing_both_badminton_and_tennis :
  (B + T - (N - Neither) = 3) :=
by
  sorry

end members_playing_both_badminton_and_tennis_l744_744009


namespace sequence_terms_l744_744946

theorem sequence_terms (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n^2 + 2n + 1) →
  (a 1 = S 1) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  (∀ n, a n = if n = 1 then 4 else 2 * n + 1) := 
by 
  intro hSn ha1 h2
  funext n
  cases n
  case zero => sorry
  case succ 
  cases n
  case zero => exact ha1
  case succ => exact h2 (n + 2) (n + 1).succ_pos

end sequence_terms_l744_744946


namespace model_volume_correct_l744_744028

/-- Hemisphere definition with given volume -/
def volume_hemisphere (R : ℝ) := (2 / 3) * Real.pi * (R ^ 3)

/-- Frustum definition with given dimensions -/
def volume_frustum (r R h : ℝ) := (1 / 3) * Real.pi * h * (r ^ 2 + r * R + R ^ 2)

/-- Total volume definition -/
def volume_model (R r h : ℝ) := volume_hemisphere R + volume_frustum r R h

theorem model_volume_correct :
  ∃ (R r h : ℝ),
    volume_hemisphere R = 144 * Real.pi ∧
    r = R / 2 ∧
    h = R / 2 ∧
    volume_model R r h = 648 * Real.pi :=
by
  sorry

end model_volume_correct_l744_744028


namespace more_knights_than_liars_l744_744292

-- Define the context of the problem: knights and liars on an island.
def islander : Type := {x : Bool // x = true ∨ x = false} -- True for knights, False for liars

-- Define the number of each type of islander
def number_of_knights (n : islander → Bool) : Nat :=
  (Finset.filter (fun i => n i = true) Finset.univ).card

def number_of_liars (n : islander → Bool) : Nat :=
  (Finset.filter (fun i => n i = false) Finset.univ).card

-- Define the friendship relation
def friends_with (n : islander → Bool) (i j : islander) : Bool :=
  if n i = n j then true else arbitrary Bool -- Arbitrary logic for heterogenous friendships

-- Define the statement made by every islander
def statement (n : islander → Bool) (i : islander) : Prop :=
  let friends := (Finset.filter (fun j => friends_with n i j) Finset.univ).card
  if n i = true then
    (Finset.filter (fun j => n j = true && friends_with n i j) Finset.univ).card > friends / 2
  else
    (Finset.filter (fun j => n j = false && friends_with n i j) Finset.univ).card > friends / 2

-- Define the theorem to prove more knights exist than liars
theorem more_knights_than_liars (n : islander → Bool) :
  (∀ i, statement n i) →
  number_of_knights n > number_of_liars n :=
  sorry

end more_knights_than_liars_l744_744292


namespace a_n_formula_T_n_formula_l744_744108

-- Definitions based on the provided conditions
def S (n : ℕ) : ℕ := (finset.range (n+1)).sum (λ k, a (k + 1)) -- S_n is sum of first n terms

axiom a_seq_pos : ∀ n, 0 < a n  -- Positive sequence
axiom a_1 : a 1 = 1  -- Initial condition
axiom a_n_condition : ∀ n, a (n + 1)^2 = S (n + 1) + S n  -- Given condition

-- Prove a_n = n
theorem a_n_formula (n : ℕ) : a n = n := 
sorry

-- Auxiliary definition for b_n
def b (n : ℕ) : ℕ := a (2 * n - 1) * 2^(a n)

-- Sum of first n terms of b_n
def T (n : ℕ) : ℕ := (finset.range n).sum (λ k, b (k + 1))

-- Prove T_n formula
theorem T_n_formula (n : ℕ) : T n = (2 * n - 3) * 2^(n + 1) + 6 := 
sorry

end a_n_formula_T_n_formula_l744_744108


namespace probability_at_least_eight_sixes_l744_744789

-- Define the binomial probability function
noncomputable def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ := 
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- Given conditions: fair die and ten rolls
def fair_die := 1 / 6
def trials := 10

-- Theorem: the probability of getting at least 8 sixes in 10 rolls
theorem probability_at_least_eight_sixes : 
  binomial_prob trials 8 fair_die + binomial_prob trials 9 fair_die + binomial_prob trials 10 fair_die = 3 / 15504 := 
  sorry

end probability_at_least_eight_sixes_l744_744789


namespace chairs_stools_legs_l744_744673

theorem chairs_stools_legs (x : ℕ) (h1 : 4 * x + 3 * (16 - x) = 60) : 4 * x + 3 * (16 - x) = 60 :=
by
  exact h1

end chairs_stools_legs_l744_744673


namespace problem1_problem2_problem3_l744_744103

theorem problem1 (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + a + 3 = 0) → (a ≤ -2 ∨ a ≥ 6) :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a + 3 ≥ 4) → 
    (if a > 2 then 
      ∀ x : ℝ, ((x ≤ 1) ∨ (x ≥ a-1)) 
    else if a = 2 then 
      ∀ x : ℝ, true
    else 
      ∀ x : ℝ, ((x ≤ a - 1) ∨ (x ≥ 1))) :=
sorry

theorem problem3 (a : ℝ) :
  (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ x^2 - a*x + a + 3 = 0) → (6 ≤ a ∧ a ≤ 7) :=
sorry

end problem1_problem2_problem3_l744_744103


namespace sum_rk_equals_l744_744647

def binom (n k : ℕ) : ℕ := nat.choose n k

def r_k (k : ℕ) : ℕ :=
  if k % 2 = 0 then
    if k ≤ 31 then 1 else 5
  else
    if k ≤ 31 then 7 else 3

theorem sum_rk_equals :
  (∑ k in Finset.range 64, k * r_k k) = 8096 := sorry

end sum_rk_equals_l744_744647


namespace minimum_sum_of_products_l744_744948

def is_valid_grid (grid : Array (Array Nat)) : Prop :=
  ∀ i j, i < 10 → j < 10 → grid[i][j] = i * 10 + j + 1

def product_of_rectangles (grid : Array (Array Nat)) (rectangles : List (Nat × Nat × Nat × Nat)) : Nat :=
  rectangles.foldl (λ acc ⟨x1, y1, x2, y2⟩ => acc + grid[x1][y1] * grid[x2][y2]) 0

def are_vertical_rectangles (rectangles : List (Nat × Nat × Nat × Nat)) : Prop :=
  ∀ ⟨x1, y1, x2, y2⟩ ∈ rectangles, y1 = y2 ∧ x2 = x1 + 1

theorem minimum_sum_of_products (grid : Array (Array Nat)) (rectangles : List (Nat × Nat × Nat × Nat)) :
  is_valid_grid grid →
  (∃ vertical_rectangles : List (Nat × Nat × Nat × Nat),
    are_vertical_rectangles vertical_rectangles ∧
    product_of_rectangles grid vertical_rectangles ≤ product_of_rectangles grid rectangles) :=
sorry

end minimum_sum_of_products_l744_744948


namespace exists_odd_integers_l744_744307

theorem exists_odd_integers (n : ℕ) (hn : n ≥ 3) : 
  ∃ x y : ℤ, x % 2 = 1 ∧ y % 2 = 1 ∧ x^2 + 7 * y^2 = 2^n :=
sorry

end exists_odd_integers_l744_744307


namespace num_sets_A_l744_744165

theorem num_sets_A : ∃ (n : ℕ), (∀ A : set ℤ, A ∪ ({-1, 1} : set ℤ) = {-1, 1} ↔ A ⊆ {-1, 1})
                      ∧ n = 4 :=
by
  sorry

end num_sets_A_l744_744165


namespace earned_points_l744_744191

def points_per_enemy := 3
def total_enemies := 6
def enemies_undefeated := 2
def enemies_defeated := total_enemies - enemies_undefeated

theorem earned_points : enemies_defeated * points_per_enemy = 12 :=
by sorry

end earned_points_l744_744191


namespace ideal_point_distance_y_axis_exists_ideal_point_linear_range_of_t_l744_744196

variable (a b : ℝ)
variable (m x : ℝ)
variable (t : ℝ)
variable (A B C : ℝ)

-- Define ideal points
def is_ideal_point (p : ℝ × ℝ) := p.snd = 2 * p.fst

-- Define the conditions for question 1
def distance_from_y_axis (a : ℝ) := abs a = 2

-- Question 1: Prove that M(2, 4) or M(-2, -4)
theorem ideal_point_distance_y_axis (a b : ℝ) (h1 : is_ideal_point (a, b)) (h2 : distance_from_y_axis a) :
  (a = 2 ∧ b = 4) ∨ (a = -2 ∧ b = -4) := sorry

-- Define the linear function
def linear_func (m x : ℝ) : ℝ := 3 * m * x - 1

-- Question 2: Prove or disprove the existence of ideal points in y = 3mx - 1
theorem exists_ideal_point_linear (m x : ℝ) (hx : is_ideal_point (x, linear_func m x)) :
  (m ≠ 2/3 → ∃ x, linear_func m x = 2 * x) ∧ (m = 2/3 → ¬ ∃ x, linear_func m x = 2 * x) := sorry

-- Question 3 conditions
def quadratic_func (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def quadratic_conditions (a b c : ℝ) : Prop :=
  (quadratic_func a b c 0 = 5 * a + 1) ∧ (quadratic_func a b c (-2) = 5 * a + 1)

-- Question 3: Prove the range of t = a^2 + a + 1 given the quadratic conditions
theorem range_of_t (a b c t : ℝ) (h1 : is_ideal_point (x, quadratic_func a b c x))
  (h2 : quadratic_conditions a b c) (ht : t = a^2 + a + 1) :
    3 / 4 ≤ t ∧ t ≤ 21 / 16 ∧ t ≠ 1 := sorry

end ideal_point_distance_y_axis_exists_ideal_point_linear_range_of_t_l744_744196


namespace lambda_range_l744_744148

noncomputable def range_of_lambda (m λ θ : ℝ) : Prop :=
  let z1 := m + (4 - m^2) * complex.I
  let z2 := 2 * real.cos θ + (λ + 3 * real.sin θ) * complex.I
  z1 = z2 → -9 / 16 ≤ λ ∧ λ ≤ 7

theorem lambda_range (m λ θ : ℝ) (h : z1 = z2) : -9 / 16 ≤ λ ∧ λ ≤ 7 := 
by {
  sorry
}

end lambda_range_l744_744148


namespace problem_statement_l744_744991

theorem problem_statement (m : ℤ) (h1 : 0 ≤ m ∧ m < 41) (h2 : 4 * m ≡ 1 [ZMOD 41]) :
  ((3 ^ m) ^ 4 - 3) % 41 = 36 :=
sorry

end problem_statement_l744_744991


namespace ball_bounce_height_l744_744415

theorem ball_bounce_height (b : ℕ) : 
  ∃ b : ℕ, 400 * (3 / 4 : ℝ)^b < 50 ∧ ∀ b' < b, 400 * (3 / 4 : ℝ)^b' ≥ 50 :=
sorry

end ball_bounce_height_l744_744415


namespace semi_circle_radius_l744_744710

theorem semi_circle_radius (P : ℝ) (pi_approx : ℝ) (hP : P = 144) (h_pi : pi_approx = 3.14159) :
  let r := 144 / (pi_approx + 2) in r ≈ 28.01 :=
by
  let r := 144 / (pi_approx + 2)
  have hr : r = 144 / (pi_approx + 2) := rfl
  rw [hP, h_pi] at hr
  have approx_r : r ≈ 28.01 := by sorry
  exact approx_r

end semi_circle_radius_l744_744710


namespace tan_half_theta_positive_l744_744174

noncomputable def theta_in_second_quadrant (k : ℤ) (θ : ℝ) : Prop :=
  (real.pi / 2 + 2 * k * real.pi < θ) ∧ (θ < real.pi + 2 * k * real.pi)

theorem tan_half_theta_positive (k : ℤ) (θ : ℝ)
  (h_second_quadrant : theta_in_second_quadrant k θ) :
  real.tan (θ / 2) > 0 :=
begin
  sorry
end

end tan_half_theta_positive_l744_744174


namespace original_length_of_ribbon_l744_744046

theorem original_length_of_ribbon (n : ℕ) (cm_per_piece : ℝ) (remaining_meters : ℝ) 
  (pieces_cm_to_m : cm_per_piece / 100 = 0.15) (remaining_ribbon : remaining_meters = 36) 
  (pieces_cut : n = 100) : n * (cm_per_piece / 100) + remaining_meters = 51 := 
by 
  sorry

end original_length_of_ribbon_l744_744046


namespace finiteM_eq_zero_exists_positive_C_l744_744072

/-
Define the problem parameters:
- M(n, m) = |n * sqrt (n^2 + a) - b * m|
- Given n, m are positive integers
- a is a fixed odd positive integer
- b is a rational number with an odd denominator
-/

def M (n m : ℕ) (a : ℕ) (b : ℚ) : ℚ :=
  abs (n * real.sqrt (n^2 + a) - b * m)

theorem finiteM_eq_zero (a : ℕ) (b : ℚ) (h_odd_a : a % 2 = 1) (h_b_odd_denom : b.denom % 2 = 1) :
  {p : ℕ × ℕ | M p.fst p.snd a b = 0}.finite :=
sorry

theorem exists_positive_C (a : ℕ) (b : ℚ) (h_odd_a : a % 2 = 1) (h_b_odd_denom : b.denom % 2 = 1) :
  ∃ C > 0, ∀ (n m : ℕ), M n m a b ≠ 0 → M n m a b ≥ C :=
sorry

end finiteM_eq_zero_exists_positive_C_l744_744072


namespace q_is_contrapositive_of_r_l744_744943

variable {A B : Prop}

def inverse (p : Prop) : Prop := 
  B → A

def negation (p : Prop) : Prop := 
  ¬A → ¬B

theorem q_is_contrapositive_of_r 
  (p : Prop) (q : Prop) (r : Prop) 
  (hp : p = (A → B)) 
  (hq : q = inverse p) 
  (hr : r = negation p) 
  : q = (¬B → ¬A) := 
  sorry

end q_is_contrapositive_of_r_l744_744943


namespace convert_decimal_to_fraction_l744_744394

theorem convert_decimal_to_fraction : (0.38 : ℚ) = 19 / 50 :=
by
  sorry

end convert_decimal_to_fraction_l744_744394


namespace polar_equation_of_line_l744_744330

theorem polar_equation_of_line (ρ θ : ℝ) :
  (∃ P : ℝ × ℝ, P = (4,0) ∧ ∀ (x y : ℝ), x = 4 -> y ⟨0 <= y⟩) →
  ρ = 4 / cos θ :=
sorry

end polar_equation_of_line_l744_744330


namespace management_personnel_to_draw_l744_744783

-- Define the conditions and the question, then state the theorem
variables (total_employees : ℕ) (salespeople : ℕ) (management_personnel : ℕ) (logistics_personnel : ℕ) (sample_size : ℕ)
  (sampling_ratio : ℚ) (management_personnel_drawn : ℕ)

-- Populate the given conditions
def company_conditions : Prop := total_employees = 150 ∧ 
                                  salespeople = 100 ∧ 
                                  management_personnel = 15 ∧ 
                                  logistics_personnel = 35 ∧ 
                                  sample_size = 30 ∧ 
                                  sampling_ratio = sample_size / total_employees ∧ 
                                  management_personnel_drawn = management_personnel * sampling_ratio

-- State the theorem we need to prove
theorem management_personnel_to_draw (h : company_conditions) : management_personnel_drawn = 3 :=
  sorry  -- proof to be filled in by the theorem prover

end management_personnel_to_draw_l744_744783


namespace max_value_of_f_l744_744092

def f (x : ℝ) : ℝ := -x + 1/x

theorem max_value_of_f : 
  ∃ x ∈ set.Icc (-2 : ℝ) (-1/3), ∀ y ∈ set.Icc (-2 : ℝ) (-1/3), f y ≤ f x :=
begin
  use -2,
  split,
  { norm_num, }, -- -2 is within the interval [-2, -1/3]
  intros y hy,
  -- Now we need to show that for all y in the interval [-2, -1/3], f y ≤ f (-2)
  have neg_y_le_neg_two: -y ≤ -(-2),
  { linarith [hy.1, hy.2] },
  have one_over_y_le_neg_half: 1/y ≤ 1/(-2),
  { apply one_div_le_one_div_of_nonpos_of_le;
    linarith [hy.1, hy.2] },
  calc
    f y = -y + 1/y : rfl
    ... ≤ 2 - 1/2   : by linarith [neg_y_le_neg_two, one_over_y_le_neg_half]
    ... = 3/2       : by norm_num,
end

end max_value_of_f_l744_744092


namespace bc_eq_one_area_of_triangle_l744_744245

variable (a b c A B : ℝ)

-- Conditions
def condition_1 : Prop := (b^2 + c^2 - a^2) / (Real.cos A) = 2
def condition_2 : Prop := (a * (Real.cos B) - b * (Real.cos A)) / (a * (Real.cos B) + b * (Real.cos A)) - b / c = 1

-- Equivalent proof problems
theorem bc_eq_one (h1 : condition_1 a b c A) : b * c = 1 := 
by 
  sorry

theorem area_of_triangle (h2 : condition_2 a b c A B) : (1/2) * b * c * Real.sin A = (Real.sqrt 3) / 4 := 
by 
  sorry

end bc_eq_one_area_of_triangle_l744_744245


namespace james_total_payment_is_45_l744_744210

noncomputable def cost_of_first_shoe := 40
noncomputable def cost_of_second_shoe := 60
noncomputable def discount_on_cheaper_pair := cost_of_first_shoe / 2
noncomputable def discounted_price_of_second_shoe := cost_of_first_shoe - discount_on_cheaper_pair
noncomputable def initial_total := cost_of_first_shoe + discounted_price_of_second_shoe
noncomputable def extra_discount := initial_total / 4
noncomputable def total_payment := initial_total - extra_discount

theorem james_total_payment_is_45 : total_payment = 45 := 
begin
  sorry
end

end james_total_payment_is_45_l744_744210


namespace ellipse_equation_and_triangle_area_l744_744199

noncomputable def ellipse : Type := sorry
variables {a b c : ℝ} {x y : ℝ} {P A B : ellipse} (OP : {x : ℝ × ℝ // x ≠ (0, 0)})

-- Conditions
def eccentricity (e : ℝ) : Prop := e = (Real.sqrt 2) / 2
def on_ellipse (P : ℝ × ℝ) : Prop := (P.fst ^ 2) / 6 + (P.snd ^ 2) / 3 = 1
def midpoint_on_line (M P : ℝ × ℝ) : Prop := M.snd = (1 / 2) * M.fst

-- Lean 4 statement
theorem ellipse_equation_and_triangle_area :
  (∃ a b : ℝ, a = Real.sqrt 6 ∧ b = Real.sqrt 3 ∧
    ∀ x y : ℝ, (x ^ 2) / 6 + (y ^ 2) / 3 = 1) ∧
  (max_area_of_triangle_AOB : ℝ, max_area_of_triangle_AOB = (3 * Real.sqrt 2) / 2)
:=
by
  sorry

end ellipse_equation_and_triangle_area_l744_744199


namespace length_PQ_l744_744071

-- Conditions as definitions
def radius_k1 : ℝ := 6
def radius_k2 : ℝ := 3
def radius_k : ℝ := 9

-- Main conjecture
theorem length_PQ : 
  ∃ k_1 k_2 k : Type,
  ∃ r1 r2 r3 : ℝ,
  ∃ H1 : r1 = radius_k1,
  ∃ H2 : r2 = radius_k2,
  ∃ H3 : r3 = radius_k,
  ∃ P Q : {x // dist x k_1 = r1} × {y // dist y k_2 = r2} × {z // dist z k = r3}, 
  let length_PQ := dist P Q in 
  length_PQ = 4 * real.sqrt 14 :=
sorry

end length_PQ_l744_744071


namespace first_player_always_wins_l744_744360

structure Table :=
  (width : ℕ)
  (height : ℕ)

def free_spot (table: Table) (x y : ℕ) : Prop :=
  x < table.width ∧ y < table.height

def first_player_can_move (table : Table) (occupied: set (ℕ × ℕ)) : Prop :=
  ∃ x y, free_spot table x y ∧ (x, y) ∉ occupied

def second_player_can_move (table : Table) (occupied: set (ℕ × ℕ)) : Prop :=
  ∃ x y, free_spot table x y ∧ (x, y) ∉ occupied

def first_player_wins (table : Table) (occupied: set (ℕ × ℕ)) : Prop :=
  ¬ second_player_can_move table occupied

theorem first_player_always_wins (table : Table) (occupied: set (ℕ × ℕ)) :
  (first_player_can_move table occupied) → first_player_wins table occupied :=
by
  sorry

end first_player_always_wins_l744_744360


namespace toms_weekly_income_l744_744349

variable (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def daily_crabs := num_buckets * crabs_per_bucket
def daily_income := daily_crabs * price_per_crab
def weekly_income := daily_income * days_per_week

theorem toms_weekly_income 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  weekly_income num_buckets crabs_per_bucket price_per_crab days_per_week = 3360 :=
by
  sorry

end toms_weekly_income_l744_744349


namespace function_value_l744_744542

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem function_value (a b : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log_base a (2 + b) = 1) (h₃ : log_base a (8 + b) = 2) : a + b = 4 :=
by
  sorry

end function_value_l744_744542


namespace length_of_AG_in_isosceles_right_triangle_l744_744193

theorem length_of_AG_in_isosceles_right_triangle
  {α β γ δ ε ζ η θ η κ λ μ ν ξ ο π ρ σ ψ ω} -- triangle vertices
  (h1 : is_isosceles_right_triangle ABC)
  (h2 : angle ABC = 45° ∧ angle BAC = 90°)
  (h3 : angle ACB = 45°)
  (h4 : circle_inscribed_in_triangle ABC touching AB AC BC at D E F)
  (h5 : meets_line_extension DF AC at G)
  (h6 : length DF = 6) :
  length AG = 3 * √ 2 
:=
sorry

end length_of_AG_in_isosceles_right_triangle_l744_744193


namespace total_difference_is_18_l744_744953

-- Define variables for Mike, Joe, and Anna's bills
variables (m j a : ℝ)

-- Define the conditions given in the problem
def MikeTipped := (0.15 * m = 3)
def JoeTipped := (0.25 * j = 3)
def AnnaTipped := (0.10 * a = 3)

-- Prove the total amount of money that was different between the highest and lowest bill is 18
theorem total_difference_is_18 (MikeTipped : 0.15 * m = 3) (JoeTipped : 0.25 * j = 3) (AnnaTipped : 0.10 * a = 3) :
  |a - j| = 18 := 
sorry

end total_difference_is_18_l744_744953


namespace find_factor_l744_744335

-- Define the conditions
def number : ℕ := 9
def expr1 (f : ℝ) : ℝ := (number + 2) * f
def expr2 : ℝ := 24 + number

-- The proof problem statement
theorem find_factor (f : ℝ) : expr1 f = expr2 → f = 3 := by
  sorry

end find_factor_l744_744335


namespace AIMN_cyclic_l744_744255

open EuclideanGeometry

-- Given a triangle ABC and the incenter I,
variables {α : Type*} [normed_group α] [normed_space ℝ α] [inner_product_space ℝ α]

-- Points A, B, C are distinct and form a triangle
variables {A B C I D E F M N : α} 

-- Define triangle geometry
noncomputable def triangle_ABC : affine_plane ℝ α :=
{ A := A,
  B := B,
  C := C,
  I := I,
  D := line_intersection (line_through A I) (perpendicular_bisector (segment A D)),
  E := line_intersection (line_through B I) (perpendicular_bisector (segment B E)),
  F := line_intersection (line_through C I) (perpendicular_bisector (segment C F)),
  M := line_intersection (perpendicular_bisector (segment A D)) (line_through B I),
  N := line_intersection (perpendicular_bisector (segment A D)) (line_through C I) }

-- Prove that A, I, M, and N are concyclic
theorem AIMN_cyclic : are_concyclic {A, I, M, N} :=
sorry

end AIMN_cyclic_l744_744255


namespace numbers_from_1_to_100_present_l744_744677

theorem numbers_from_1_to_100_present (a : Fin 200 → ℕ) (is_blue : Fin 200 → Prop) :
  (∀ n : ℕ, n ≤ 100 → n ∈ { i | is_blue i ∧ a i = n }) →
  (∀ n : ℕ, 100 < n → n ∈ { i | ¬is_blue i ∧ a i = (200 - n) }) →
  {a i | i < 100} = {n : ℕ | n ≤ 100} :=
by
  intros h_blue h_red
  sorry

end numbers_from_1_to_100_present_l744_744677


namespace probability_each_wins_one_game_l744_744665

theorem probability_each_wins_one_game : 
  let p_A_black_first := 1 / 3
  let p_B_black_first := 1 / 2
  let p_B_wins_given_A_black := (1 - p_A_black_first)
  let p_A_wins_given_B_black := (1 - p_B_black_first)
  let p_case1 := (1 / 2) * p_A_black_first * p_B_black_first + (1 / 2) * p_A_wins_given_B_black * p_B_black_first
  let p_case2 := (1 / 2) * p_B_wins_given_A_black * p_A_black_first + (1 / 2) * p_B_black_first * p_A_black_first
  let p_total := p_case1 + p_case2
  in
  p_total = 29 / 72 :=
  by sorry

end probability_each_wins_one_game_l744_744665


namespace domain_of_f_l744_744079

noncomputable def f (x : ℝ) : ℝ := (x - 1).sqrt / (x - 2)

theorem domain_of_f : ∀ x, (x - 1 >= 0) ∧ (x ≠ 2) ↔ (x ∈ Icc 1 2 ∪ Ioi 2) :=
by sorry

end domain_of_f_l744_744079


namespace gasoline_car_p_ge_hybrid_electric_car_p_gasoline_car_p_le_100_hybrid_car_p_l744_744279

noncomputable def sound_pressure_level (p p0 : ℝ) : ℝ :=
20 * real.log10 (p / p0)

variables {p0 p1 p2 p3 : ℝ} (h_p0 : p0 > 0)
(h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
(h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)
(h_p3 : p3 = 100 * p0)

theorem gasoline_car_p_ge_hybrid (h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
                                (h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)  : p1 ≥ p2 :=
sorry

theorem electric_car_p (h_p3 : p3 = 100 * p0) : p3 = 100 * p0 :=
sorry

theorem gasoline_car_p_le_100_hybrid_car_p (h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
                                           (h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0) : p1 ≤ 100 * p2 :=
sorry

#check gasoline_car_p_ge_hybrid
#check electric_car_p
#check gasoline_car_p_le_100_hybrid_car_p

end gasoline_car_p_ge_hybrid_electric_car_p_gasoline_car_p_le_100_hybrid_car_p_l744_744279


namespace arianna_lost_pieces_l744_744824

noncomputable def arianna_pieces_lost 
  (samantha_lost : ℕ)
  (total_pieces_left : ℕ)
  (initial_pieces_per_player : ℕ)
  : ℕ :=
  let initial_total_pieces := initial_pieces_per_player * 2
  let samantha_pieces_left := initial_pieces_per_player - samantha_lost
  let arianna_pieces_left := total_pieces_left - samantha_pieces_left
  initial_pieces_per_player - arianna_pieces_left

theorem arianna_lost_pieces 
  (samantha_lost: ℕ)
  (total_pieces_left : ℕ)
  (initial_pieces_per_player : ℕ)
  (h1 : samantha_lost = 9)
  (h2 : total_pieces_left = 20)
  (h3 : initial_pieces_per_player = 16)
  : arianna_pieces_lost samantha_lost total_pieces_left initial_pieces_per_player = 3 :=
by
  rw [h1, h2, h3]
  simp [arianna_pieces_lost]
  sorry

end arianna_lost_pieces_l744_744824


namespace smallest_integer_y_l744_744752

theorem smallest_integer_y (y : ℤ) (h: y < 3 * y - 15) : y ≥ 8 :=
by sorry

end smallest_integer_y_l744_744752


namespace eval_g_five_l744_744169

def g (x : ℝ) : ℝ := 4 * x - 2

theorem eval_g_five : g 5 = 18 := by
  sorry

end eval_g_five_l744_744169


namespace find_a_l744_744911

theorem find_a (a : ℝ) (h₀ : 0 < a) (h₁ : ∀ x, f x = sin (a * x + π / 3))
    (h₂ : ∃ d, d = 4 ∧ ∀ x₁ x₂, f x₁ = f x₂ → abs (x₁ - x₂) = d ∨ abs (x₁ - x₂) = 2 * d) :
  a = π / 4 :=
by
  sorry

end find_a_l744_744911


namespace inequality_true_l744_744582

theorem inequality_true (a b : ℝ) (h : a > b) (x : ℝ) : 
  (a > b) → (x ≥ 0) → (a / ((2^x) + 1) > b / ((2^x) + 1)) :=
by 
  sorry

end inequality_true_l744_744582


namespace sound_pressures_relationships_l744_744274

variables (p p0 p1 p2 p3 : ℝ)
  (Lp Lpg Lph Lpe : ℝ)

-- The definitions based on the conditions
def sound_pressure_level (p : ℝ) (p0 : ℝ) : ℝ := 20 * (Real.log10 (p / p0))

-- Given conditions
axiom p0_gt_zero : p0 > 0

axiom gasoline_car_levels : 60 ≤ Lpg ∧ Lpg ≤ 90
axiom hybrid_car_levels : 50 ≤ Lph ∧ Lph ≤ 60
axiom electric_car_level : Lpe = 40

axiom gasoline_car_sound_pressure : Lpg = sound_pressure_level p1 p0
axiom hybrid_car_sound_pressure : Lph = sound_pressure_level p2 p0
axiom electric_car_sound_pressure : Lpe = sound_pressure_level p3 p0

-- The proof to be derived
theorem sound_pressures_relationships (p0_gt_zero : p0 > 0)
  (gasoline_car_levels : 60 ≤ Lpg ∧ Lpg ≤ 90)
  (hybrid_car_levels : 50 ≤ Lph ∧ Lph ≤ 60)
  (electric_car_level : Lpe = 40)
  (gasoline_car_sound_pressure : Lpg = sound_pressure_level p1 p0)
  (hybrid_car_sound_pressure : Lph = sound_pressure_level p2 p0)
  (electric_car_sound_pressure : Lpe = sound_pressure_level p3 p0) :
  p1 ≥ p2 ∧ p3 = 100 * p0 ∧ p1 ≤ 100 * p2 :=
by
  sorry

end sound_pressures_relationships_l744_744274


namespace solution_correct_l744_744066

open Real

noncomputable def problem_expression : ℝ :=
  log10 2 + log10 5 - 42 * 8^(1/4) - 2017^0

theorem solution_correct : problem_expression = -2 :=
by
  -- The proof will be filled in later
  sorry

end solution_correct_l744_744066


namespace acute_triangle_side_length_range_l744_744113

theorem acute_triangle_side_length_range (a : ℝ) (h1 : a > 0) (h2 : ∀ (α β γ : ℝ), 
  (1 + 3 > a) ∧ (1 + a > 3) ∧ (3 + a > 1) ∧ 
  (cos α > 0) ∧ (cos β > 0) ∧ (cos γ > 0)) : 
  2 * Real.sqrt 2 < a ∧ a < Real.sqrt 10 :=
by
  sorry

end acute_triangle_side_length_range_l744_744113


namespace polynomial_inequality_l744_744981

theorem polynomial_inequality (P : ℝ[X]) (n : ℕ) (a : ℝ) (h_deg : P.degree = n) (h_real : ∀ x, P.eval x ∈ set.Icc (-∞) (∞)) (h_a_ge : a ≥ 3) :
 ∃ j ∈ set.Icc 0 (n + 1), ∥a^j - P.eval j∥ ≥ 1 := 
sorry

end polynomial_inequality_l744_744981


namespace solve_years_later_twice_age_l744_744793

-- Define the variables and the given conditions
def man_age (S: ℕ) := S + 25
def years_later_twice_age (S M: ℕ) (Y: ℕ) := (M + Y = 2 * (S + Y))

-- Given conditions
def present_age_son := 23
def present_age_man := man_age present_age_son

theorem solve_years_later_twice_age :
  ∃ Y, years_later_twice_age present_age_son present_age_man Y ∧ Y = 2 := by
  sorry

end solve_years_later_twice_age_l744_744793


namespace snowfall_difference_l744_744452

theorem snowfall_difference :
  let BaldMountain_snow := 1.5 * 100 -- in cm
  let BillyMountain_snow := 350.3 -- in cm
  let MountPilot_snow := 126.2 -- in cm
  let RockstonePeak_snow := 5257 / 10 -- in cm
  let SunsetRidge_snow := 224.75 -- in cm
  let RiverHill_snow := 1.75 * 100 -- in cm
  BaldMountain_snow <|
  let combined_snow := BillyMountain_snow + MountPilot_snow + RockstonePeak_snow + SunsetRidge_snow + RiverHill_snow
  combined_snow - BaldMountain_snow = 1251.95 := sorry

end snowfall_difference_l744_744452


namespace sphere_wall_thickness_l744_744791

noncomputable def radius (diameter : ℝ) : ℝ := diameter / 2

noncomputable def weight_of_displaced_water (R : ℝ) : ℝ := 
  (2 / 3) * Real.pi * R^3

noncomputable def weight_of_silver_shell (s : ℝ) (R r : ℝ) : ℝ := 
  (4 / 3) * Real.pi * s * (R^3 - r^3)

noncomputable def inner_radius_cube (R s : ℝ) : ℝ := 
  (R^3 * (2 * s - 1)) / (2 * s)

noncomputable def thickness (R r : ℝ) : ℝ := R - r

theorem sphere_wall_thickness :
  ∀ (diameter : ℝ) (s : ℝ), 
  diameter = 1 → s = 10.5 → 
  thickness (radius diameter) (Real.cbrt (inner_radius_cube (radius diameter) s)) = 0.008 :=
by
  intros diameter s h_diameter h_s
  have R : ℝ := radius diameter
  rw [h_diameter] at R
  have r : ℝ := Real.cbrt (inner_radius_cube R s)
  have thickness_val : ℝ := thickness R r
  rw [h_s] at thickness_val
  -- We now need to show that thickness_val is 0.008
  have h1 : R = 0.5 := by
    simp [radius, h_diameter]
  have h2 : inner_radius_cube R s = 0.119 := by
    simp [inner_radius_cube, h_diameter, h_s, h1]
  have h3 : r = Real.cbrt 0.119 := by
    simp [Real.cbrt, h2]
  -- Finally
  simp [thickness, h1, h3]
  -- Final expression simplification is not straightforward, so we skip the proof here
  sorry

end sphere_wall_thickness_l744_744791


namespace ratio_equal_one_of_log_conditions_l744_744994

noncomputable def logBase (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem ratio_equal_one_of_log_conditions
  (p q : ℝ)
  (hp : 0 < p)
  (hq : 0 < q)
  (h : logBase 8 p = logBase 18 q ∧ logBase 18 q = logBase 24 (p + 2 * q)) :
  q / p = 1 :=
by
  sorry

end ratio_equal_one_of_log_conditions_l744_744994


namespace choose_stick_l744_744387

-- Define the lengths of the sticks Xiaoming has
def xm_stick1 : ℝ := 4
def xm_stick2 : ℝ := 7

-- Define the lengths of the sticks Xiaohong has
def stick2 : ℝ := 2
def stick3 : ℝ := 3
def stick8 : ℝ := 8
def stick12 : ℝ := 12

-- Define the condition for a valid stick choice from Xiaohong's sticks
def valid_stick (x : ℝ) : Prop := 3 < x ∧ x < 11

-- State the problem as a theorem to be proved
theorem choose_stick : valid_stick stick8 := by
  sorry

end choose_stick_l744_744387


namespace ducks_snails_l744_744039

noncomputable def total_snails_found (
    n_ducklings : ℕ,
    snails_first_group : ℕ,
    snails_second_group : ℕ,
    snails_mother : ℕ,
    snails_remaining_ducklings : ℕ): ℕ :=
  snails_first_group + snails_second_group + snails_mother + snails_remaining_ducklings

theorem ducks_snails (
    n_ducklings : ℕ,
    snails_per_first_group_duckling : ℕ,
    snails_per_second_group_duckling : ℕ,
    first_group_ducklings : ℕ,
    second_group_ducklings : ℕ,
    remaining_ducklings : ℕ,
    mother_duck_snails_mult : ℕ,
    half_mother_duck_snails : ℕ
) :
  n_ducklings = 8 →
  first_group_ducklings = 3 →
  second_group_ducklings = 3 →
  remaining_ducklings = 2 →
  snails_per_first_group_duckling = 5 →
  snails_per_second_group_duckling = 9 →
  mother_duck_snails_mult = 3 →
  ∀ mother_snails snails_per_remaining_duckling snails_first_group snails_second_group total_snails, 
    mother_snails = mother_duck_snails_mult * (first_group_ducklings * snails_per_first_group_duckling + second_group_ducklings * snails_per_second_group_duckling) →
    snails_per_remaining_duckling = mother_snails / 2 →
    snails_first_group = first_group_ducklings * snails_per_first_group_duckling →
    snails_second_group = second_group_ducklings * snails_per_second_group_duckling →
    total_snails = total_snails_found (
      n_ducklings,
      snails_first_group,
      snails_second_group,
      mother_snails,
      remaining_ducklings * snails_per_remaining_duckling
    ) →
    total_snails = 294 :=
by {
  intros,
  sorry
}

end ducks_snails_l744_744039


namespace distance_travelled_is_correct_l744_744769

def speed_of_boat_still_water : ℝ := 15
def rate_of_current : ℝ := 3
def time_in_hours : ℝ := 24 / 60
def effective_speed_downstream : ℝ := speed_of_boat_still_water + rate_of_current
def distance_travelled_downstream : ℝ := effective_speed_downstream * time_in_hours

theorem distance_travelled_is_correct :
  distance_travelled_downstream = 7.2 :=
by
  sorry

end distance_travelled_is_correct_l744_744769


namespace range_of_u_l744_744104

variable (a b u : ℝ)

theorem range_of_u (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∀ x : ℝ, x > 0 → a^2 + b^2 ≥ x ↔ x ≤ 16) :=
sorry

end range_of_u_l744_744104


namespace angles_of_triangle_l744_744635

-- Define the circumradius and angles
def circumradius := 1
def angle_A := Real.pi / 6
def max_m_X := Real.sqrt 3 / 3

-- Define the statement of the proof problem
theorem angles_of_triangle (ABC : Triangle) (X : Point_in_triangle_or_boundary ABC)
  (m : Point -> ℝ) (notA : X ≠ A) (notB : X ≠ B) (notC : X ≠ C)
  (AX : X ≠ A) (BX : X ≠ B) (CX : X ≠ C)
  (h1 : ∠BAC = angle_A)
  (h2 : circumradius_of_triangle ABC = circumradius)
  (h3 : m(X) = min (dist X A) (min (dist X B) (dist X C))) 
  (h4 : max (m := fun X => min (dist X A) (min (dist X B) (dist X C))) = max_m_X) : 
  ∠A = angle_A ∧ ∠B = 2 * angle_A ∧ ∠C = angle_A := sorry

end angles_of_triangle_l744_744635


namespace coeff_a3b3_l744_744744

theorem coeff_a3b3 (a b c : ℚ) : 
  coeff_of_term (a^3 * b^3) ((a+b)^6 * (c + 1/c)^4) = 240 :=
sorry

end coeff_a3b3_l744_744744


namespace square_must_rotate_at_least_5_turns_l744_744560

-- Define the square and pentagon as having equal side lengths
def square_sides : Nat := 4
def pentagon_sides : Nat := 5

-- The problem requires us to prove that the square needs to rotate at least 5 full turns
theorem square_must_rotate_at_least_5_turns :
  let lcm := Nat.lcm square_sides pentagon_sides
  lcm / square_sides = 5 :=
by
  -- Proof to be provided
  sorry

end square_must_rotate_at_least_5_turns_l744_744560


namespace coupon1_better_than_coupon2_and_coupon3_at_219_95_l744_744032
noncomputable def coupon_discounts (x : ℝ) : ℝ × ℝ × ℝ :=
  (0.1 * x, 20, 0.18 * (x - 100))

theorem coupon1_better_than_coupon2_and_coupon3_at_219_95 :
  let x := 219.95 in
  let (discount1, discount2, discount3) := coupon_discounts x in
  discount1 > discount2 ∧ discount1 > discount3 :=
by
  sorry

end coupon1_better_than_coupon2_and_coupon3_at_219_95_l744_744032


namespace mining_company_percentage_nickel_l744_744828

theorem mining_company_percentage_nickel (nickel_daily : ℕ) (copper_daily : ℕ) (total_daily : ℕ) (percentage : ℚ) :
  nickel_daily = 720 → copper_daily = 360 → total_daily = nickel_daily + copper_daily → percentage = (720 / 1080) * 100 → percentage = 66.67 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h3
  have : 720 + 360 = 1080 := rfl
  rw this at h3
  rw h3 at h4
  exact h4

end mining_company_percentage_nickel_l744_744828


namespace lesson_duration_on_monday_l744_744444

theorem lesson_duration_on_monday 
  (goes_to_school_every_day : ∀ d, true) 
  (monday_lessons : ℕ := 6) 
  (tuesday_lessons : ℕ := 3) 
  (tuesday_lesson_duration : ℕ := 60) -- lesson duration in minutes
  (wednesday_duration_factor : ℕ := 2) 
  (total_school_time : ℕ := 12 * 60) -- total time in minutes
  :
  (monday_lesson_duration : ℕ) :
  monday_lesson_duration = 30 :=
by sorry

end lesson_duration_on_monday_l744_744444


namespace bc_eq_one_area_of_triangle_l744_744247

variable (a b c A B : ℝ)

-- Conditions
def condition_1 : Prop := (b^2 + c^2 - a^2) / (Real.cos A) = 2
def condition_2 : Prop := (a * (Real.cos B) - b * (Real.cos A)) / (a * (Real.cos B) + b * (Real.cos A)) - b / c = 1

-- Equivalent proof problems
theorem bc_eq_one (h1 : condition_1 a b c A) : b * c = 1 := 
by 
  sorry

theorem area_of_triangle (h2 : condition_2 a b c A B) : (1/2) * b * c * Real.sin A = (Real.sqrt 3) / 4 := 
by 
  sorry

end bc_eq_one_area_of_triangle_l744_744247


namespace sqrt_infinite_nested_problem_l744_744316

theorem sqrt_infinite_nested_problem :
  ∃ m : ℝ, m = Real.sqrt (6 + m) ∧ m = 3 :=
by
  sorry

end sqrt_infinite_nested_problem_l744_744316


namespace sum_of_squares_formula_l744_744738

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

theorem sum_of_squares_formula (n : ℕ) (h : 0 < n) :
  ∑ i in finset.range (n + 1), i ^ 2 = sum_of_squares n :=
by
  sorry

end sum_of_squares_formula_l744_744738


namespace f_neg_two_value_l744_744124

-- Proving f(-2) given f(x) is an odd function and f(x) = 2^x - 1 for x > 0

-- Define that f is odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

-- Define f based on the given condition f(x) = 2^x - 1 for x > 0
def f (x : ℝ) : ℝ :=
if x > 0 then 2 ^ x - 1 else 1 -- temporary placeholder for non-positive x

-- Declare the ultimate goal
theorem f_neg_two_value : 
  is_odd_function f ∧ (∀ x, x > 0 → f x = 2 ^ x - 1) → f (-2) = -3 :=
by 
  -- we shall provide the proof here
  sorry

end f_neg_two_value_l744_744124


namespace running_time_around_pentagon_l744_744565

theorem running_time_around_pentagon :
  let l₁ := 40
  let l₂ := 50
  let l₃ := 60
  let l₄ := 45
  let l₅ := 55
  let v₁ := 9 * 1000 / 60
  let v₂ := 8 * 1000 / 60
  let v₃ := 7 * 1000 / 60
  let v₄ := 6 * 1000 / 60
  let v₅ := 5 * 1000 / 60
  let t₁ := l₁ / v₁
  let t₂ := l₂ / v₂
  let t₃ := l₃ / v₃
  let t₄ := l₄ / v₄
  let t₅ := l₅ / v₅
  t₁ + t₂ + t₃ + t₄ + t₅ = 2.266 := by
    sorry

end running_time_around_pentagon_l744_744565


namespace lines_parallel_or_skew_l744_744559

-- Define a rectangular solid
structure RectangularSolid (α : Type*) :=
(top_plane : set α)
(bottom_plane : set α)
(parallel_planes : ∀ x ∈ top_plane, ∀ y ∈ bottom_plane, x ≠ y)

-- Define lines a and b
structure Line (α : Type*) :=
(points : set α)

def in_plane {α : Type*} (l : Line α) (p : set α) : Prop :=
∀ x ∈ l.points, x ∈ p

-- Given conditions
variables {α : Type*} (R : RectangularSolid α)
variables (a b : Line α)
variables (h₁ : in_plane a R.top_plane)
variables (h₂ : in_plane b R.bottom_plane)

-- The proposition to prove: lines a and b are either parallel or skew
theorem lines_parallel_or_skew : (∀ x ∈ a.points, ∀ y ∈ b.points, x ≠ y) ∨
  (∀ x1 ∈ a.points, ∀ x2 ∈ a.points, ∀ y1 ∈ b.points, ∀ y2 ∈ b.points, x1 ≠ x2 ∧ y1 ≠ y2 ∧ x1 ≠ y1 ∧ x2 ≠ y2) :=
sorry

end lines_parallel_or_skew_l744_744559


namespace bill_experience_l744_744063

noncomputable def solution : ℕ :=
  let B := 10  -- Bill's current years of experience accounting for the break
  B

theorem bill_experience:
  ∀ (current_bill_age current_joan_age bill_experience_before_break joan_experience_now : ℕ),
    current_bill_age = 40 →
    current_joan_age = 50 →
    ∃ (B J : ℕ), 5 * J = 3 * (B - 5) →
    2 * B = J →
    bill_experience_before_break = B - 5 →
    -- Bill's current experience considering his 5-year break
    B = 10 :=
by
  intros current_bill_age current_joan_age bill_experience_before_break joan_experience_now h1 h2 h3 h4 h5
  sorry

end bill_experience_l744_744063


namespace integer_points_on_segment_l744_744200

noncomputable def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

theorem integer_points_on_segment (n : ℕ) (h : 0 < n) :
  f n = if n % 3 = 0 then 2 else 0 :=
by
  sorry

end integer_points_on_segment_l744_744200


namespace find_a_l744_744552

-- Define what it means for P(X = k) to be given by a particular function
def P (X : ℕ) (a : ℕ) := X / (2 * a)

-- Define the condition on the probabilities
def sum_of_probabilities_is_one (a : ℕ) :=
  (1 / (2 * a) + 2 / (2 * a) + 3 / (2 * a) + 4 / (2 * a)) = 1

-- The theorem to prove
theorem find_a (a : ℕ) (h : sum_of_probabilities_is_one a) : a = 5 :=
by sorry

end find_a_l744_744552


namespace base7_to_base10_conversion_l744_744077

theorem base7_to_base10_conversion :
  let n := 5213
  in let base7_to_base10 (n : ℕ) : ℕ :=
    ((n / 1000) % 10) * 7^3 + ((n / 100) % 10) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0
  in base7_to_base10 n = 1823 := 
by
  sorry

end base7_to_base10_conversion_l744_744077


namespace carson_clawed_39_times_l744_744457

def wombats_count := 9
def wombat_claws_per := 4
def rheas_count := 3
def rhea_claws_per := 1

def wombat_total_claws := wombats_count * wombat_claws_per
def rhea_total_claws := rheas_count * rhea_claws_per
def total_claws := wombat_total_claws + rhea_total_claws

theorem carson_clawed_39_times : total_claws = 39 :=
  by sorry

end carson_clawed_39_times_l744_744457


namespace chord_length_intersection_l744_744915

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 4 * x
noncomputable def line (x y b : ℝ) : Prop := y = 2 * x + b

theorem chord_length_intersection (b : ℝ) :
  (∃ (x1 x2 y1 y2 : ℝ),
    parabola x1 y1 ∧ line x1 y1 b ∧
    parabola x2 y2 ∧ line x2 y2 b ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    sqrt(5) * sqrt((x1 + x2)^2 - 4 * x1 * x2) = 3 * sqrt(5)) →
  b = -4 :=
sorry

end chord_length_intersection_l744_744915


namespace geom_seq_not_necessary_sufficient_l744_744618

theorem geom_seq_not_necessary_sufficient (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) (h2 : q > 1) :
  ¬(∀ n, a n > a (n + 1) → false) ∨ ¬(∀ n, a (n + 1) > a n) :=
sorry

end geom_seq_not_necessary_sufficient_l744_744618


namespace coupon_1_best_for_219_95_l744_744034

def discount_coupon_1 (x : ℝ) : ℝ := 0.1 * x
def discount_coupon_2 (x : ℝ) : ℝ := if 100 ≤ x then 20 else 0
def discount_coupon_3 (x : ℝ) : ℝ := if 100 < x then 0.18 * (x - 100) else 0

theorem coupon_1_best_for_219_95 :
  (200 < 219.95) ∧ (219.95 < 225) →
  (discount_coupon_1 219.95 > discount_coupon_2 219.95) ∧
  (discount_coupon_1 219.95 > discount_coupon_3 219.95) :=
by sorry

end coupon_1_best_for_219_95_l744_744034


namespace Mysoon_ornament_collection_l744_744657

theorem Mysoon_ornament_collection :
  ∃ O : ℕ, 
    (10 + (1 / 6 : ℚ) * O) = (2 * (1 / 3 : ℚ) * O) ∧ 
    (1 / 2 * (10 + (1 / 6 : ℚ) * O) = (1 / 3 : ℚ) * O) ∧
    O = 20 :=
begin
  sorry
end

end Mysoon_ornament_collection_l744_744657


namespace car_mpg_difference_l744_744023

theorem car_mpg_difference 
  (T : ℤ) -- tank capacity in gallons
  (mileage_highway : ℤ) -- mileage per tankful on the highway
  (mileage_city : ℤ) -- mileage per tankful in the city
  (mpg_city : ℤ) -- miles per gallon in the city
  (h1 : mileage_highway = 462) -- condition 1: 462 miles per tankful on the highway
  (h2 : mileage_city = 336) -- condition 2: 336 miles per tankful in the city
  (h3 : mpg_city = 48) : -- condition 3: 48 miles per gallon in the city
  T = mileage_city / mpg_city := (18 : ℤ) :=
by
  have T := mileage_city / mpg_city  -- Calculate the tank capacity
  have mpg_highway := mileage_highway / T -- Calculate the highway mpg
  have answer := mpg_highway - mpg_city -- Calculate the fewer miles per gallon
  sorry -- proof steps to be inserted

end car_mpg_difference_l744_744023


namespace probability_of_divisibility_l744_744986

theorem probability_of_divisibility (S : Set ℕ) (a1 a2 a3 a4 : ℕ) (hS : S = {d | d ∣ 15^7}) 
(h1 : a1 ∈ S) (h2 : a2 ∈ S) (h3 : a3 ∈ S) (h4 : a4 ∈ S) 
(prob : (∃ (m n : ℕ), RelativelyPrime m n ∧ n ≠ 0 ∧ (m : ℚ) * (n : ℚ)⁻¹ = 14400 / 16777216)) :
  ∃ (m : ℕ), m = 225 := sorry

end probability_of_divisibility_l744_744986


namespace r_expansion_l744_744935

theorem r_expansion (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by
  sorry

end r_expansion_l744_744935


namespace chess_group_games_l744_744723

theorem chess_group_games (n : ℕ) (h : n = 15) : (n.choose 2) = 105 :=
by
  rw [h]
  exact Nat.choose_symmetric 15 2
  rw [Nat.choose_eq_factorial_div_factorial]
  sorry

end chess_group_games_l744_744723


namespace half_sum_neg_l744_744701

-- Define the conditions as Lean definitions
def x : ℝ := sorry
def sum_x3 := x + 3
def half_sum := (1/2) * sum_x3

-- Define the proposition that needs to be proven
theorem half_sum_neg : half_sum < 0 :=
sorry

end half_sum_neg_l744_744701


namespace math_proof_problem_l744_744386

noncomputable def statement_B : Prop :=
  ∀ a : ℝ, l₁ a = line1 a ∧ l₂ a = line2 a →
  slope l₁ = slope l₂ →
  l₁.parallel l₂ →

noncomputable def statement_D : Prop :=
  let M := (1, 2)
      N := (4, 6)
      line := { P : Point | P.y = P.x - 1 }
  in ∃ P : Point, P ∈ line ∧ (dist P M + dist P N) 

theorem math_proof_problem : statement_B ∧ statement_D := sorry

end math_proof_problem_l744_744386


namespace problem_statement_l744_744990

theorem problem_statement (m : ℤ) (h1 : 0 ≤ m ∧ m < 41) (h2 : 4 * m ≡ 1 [ZMOD 41]) :
  ((3 ^ m) ^ 4 - 3) % 41 = 36 :=
sorry

end problem_statement_l744_744990


namespace same_graph_l744_744759

-- Definitions of the functions in the problem
def fA1 (x : ℝ) := x
def fA2 (x : ℝ) := Real.sqrt (x^2)

def fB1 (x : ℝ) := x - 1
def fB2 (x : ℝ) := (x^2 - 1) / (x + 1)

def fC1 (x : ℝ) := x^2
def fC2 (x : ℝ) := 2 * x^2

def fD1 (x : ℝ) := x^2 - 4 * x + 6
def fD2 (x : ℝ) := (x - 2)^2 + 2

-- The statement we need to prove
theorem same_graph : (∀ x, fA1 x = fA2 x) = false ∧ 
                     (∀ x, fB1 x = fB2 x) = false ∧ 
                     (∀ x, fC1 x = fC2 x) = false ∧ 
                     (∀ x, fD1 x = fD2 x) = true :=
by sorry

end same_graph_l744_744759


namespace percent_difference_l744_744163

theorem percent_difference:
  let percent_value1 := (55 / 100) * 40
  let fraction_value2 := (4 / 5) * 25
  percent_value1 - fraction_value2 = 2 :=
by
  sorry

end percent_difference_l744_744163


namespace certain_number_proof_l744_744379

-- Definitions as per the conditions in the problem
variables (x y : ℕ)

def original_ratio := (2 : ℕ) / (3 : ℕ)
def desired_ratio := (x : ℕ) / (5 : ℕ)

-- Problem statement: Prove that x = 4 given the conditions
theorem certain_number_proof (h1 : 3 + y = 5) (h2 : 2 + y = x) : x = 4 := by
  sorry

end certain_number_proof_l744_744379


namespace geometric_locus_circle_l744_744735

noncomputable def locus_midpoints (l m : set (E : Type) [Euclidean_space E]) [skew_perpendicular l m] (d h : ℝ) : set (E) :=
  { M : E | ∃ C D, C ∈ l ∧ D ∈ m ∧ dist C D = d ∧ dist M ((C + D) / 2) = 1 / 2 * sqrt (d ^ 2 - h ^ 2) }

theorem geometric_locus_circle (l m : set (E : Type) [Euclidean_space E]) [skew_perpendicular l m] (d h : ℝ) (σ : set E) :
  let locus := locus_midpoints l m d h in
  is_plane σ ∧ equidistant_and_parallel σ l m ∧ (∀ M ∈ locus, center (σ) ∈ locus ∧ ∀ R, dist (center σ) R = 1 / 2 * sqrt (d ^ 2 - h ^ 2)) :=
sorry

end geometric_locus_circle_l744_744735


namespace find_lambda_l744_744924

variables {V : Type} [AddCommGroup V] [Module ℝ V]
variables (e1 e2 : V) (a b : V) (lambda : ℝ)

-- Assume the given conditions
def non_collinear (u v : V) : Prop := ¬ ∃ (c : ℝ), c • u = v

def collinear (u v : V) : Prop := ∃ (c : ℝ), c • u = v

-- Given conditions
hypothesis h1 : non_collinear e1 e2
hypothesis h2 : a = 2 • e1 - e2
hypothesis h3 : b = e1 + lambda • e2
hypothesis h4 : collinear a b

-- Statement to prove
theorem find_lambda : lambda = -1 / 2 :=
sorry

end find_lambda_l744_744924


namespace scale_drawing_represents_line_segment_l744_744435

-- Define the given conditions
def scale_factor : ℝ := 800
def line_segment_length_inch : ℝ := 4.75

-- Prove the length in feet
theorem scale_drawing_represents_line_segment :
  line_segment_length_inch * scale_factor = 3800 :=
by
  sorry

end scale_drawing_represents_line_segment_l744_744435


namespace abs_div_nonzero_l744_744167

theorem abs_div_nonzero (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  ¬ (|a| / a + |b| / b = 1) :=
by
  sorry

end abs_div_nonzero_l744_744167


namespace determine_range_of_x_l744_744588

theorem determine_range_of_x (x : ℝ) (h₁ : 1/x < 3) (h₂ : 1/x > -2) : x > 1/3 ∨ x < -1/2 :=
sorry

end determine_range_of_x_l744_744588


namespace minimize_distance_between_curves_l744_744377

theorem minimize_distance_between_curves (t : ℝ) (ht : t > 0)
  (A : ℝ → ℝ) (B : ℝ → ℝ)
  (hA : ∀ x, A x = x^2 + 1)
  (hB : ∀ x, B x = log x) :
  t = sqrt 2 / 2 :=
by
  -- Definitions
  let y := A t - B t
  let dy := deriv (λ x, x^2 + 1) t - deriv (λ x, log x) t
  -- We need to show that this dy is equal to 0 at t = sqrt 2 / 2
  -- and that the second derivative test confirms it is a minimum.
  sorry

end minimize_distance_between_curves_l744_744377


namespace problem_l744_744257

variable (n : ℕ) (x : Fin n → ℝ)

theorem problem (h1: n > 1) (h2 : ∀ i : Fin n, 0 < x i) (h3 : ∑ i, x i = 1) :
  ∑ i, 1 / (x i - (x i) ^ 3) ≥ n ^ 4 / (n ^ 2 - 1) := 
sorry

end problem_l744_744257


namespace f_g_neg2_eq_729_div_64_l744_744641

def f (x : ℝ) : ℝ := x^3
def g (x : ℝ) : ℝ := 2^x + 2

theorem f_g_neg2_eq_729_div_64 : f (g (-2)) = 729 / 64 :=
by
  sorry

end f_g_neg2_eq_729_div_64_l744_744641


namespace partial_fraction_sum_l744_744065

theorem partial_fraction_sum :
  (let A := (1:ℚ) / 30
   let B := -(1:ℚ) / 8
   let C := (1:ℚ) / 6
   let D := (1:ℚ) /12
   let E := (1:ℚ) / 120
   in A + B + C + D + E = 1 / 6) :=
by {
  sorry
}

end partial_fraction_sum_l744_744065


namespace probability_of_shaded_triangle_l744_744964

theorem probability_of_shaded_triangle :
  let A B C D E F : Type
  let triangles := {("AEC"), ("AEB"), ("BEC"), ("BED"), ("BDC"), ("BDF"), ("BFC"), ("DFC")}
  let shaded_triangles := {("BDF"), ("BFC"), ("DFC")}
  let total_triangles : ℕ := triangles.card
  let shaded_count : ℕ := shaded_triangles.card
  total_triangles = 8 ∧ shaded_count = 3 → 
  (shaded_count : ℚ) / (total_triangles : ℚ) = 3 / 8 :=
by
  let A B C D E F := ℕ
  let triangles := {("AEC"), ("AEB"), ("BEC"), ("BED"), ("BDC"), ("BDF"), ("BFC"), ("DFC")}
  let shaded_triangles := {("BDF"), ("BFC"), ("DFC")}
  let total_triangles := triangles.card
  let shaded_count := shaded_triangles.card
  have hex_triangles : total_triangles = 8 := sorry
  have hex_shaded : shaded_count = 3 := sorry
  show (shaded_count : ℚ) / (total_triangles : ℚ) = 3 / 8
  from sorry

end probability_of_shaded_triangle_l744_744964


namespace problem_part_I_problem_part_II_l744_744130

variable (a : ℕ → ℕ) (a_n : ℕ → ℕ)
variable (b : ℕ → ℕ) (S : ℕ → ℕ)

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), S n = ∑ i in finset.range (n + 1), a i

theorem problem_part_I 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_condition1 : a 5 = 3 * a 2)
  (h_condition2 : S 7 = 14 * a 2 + 7) :
  ∀ n : ℕ, a n = 2 * n - 1 :=
sorry

theorem problem_part_II 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_condition1 : a 5 = 3 * a 2)
  (h_condition2 : S 7 = 14 * a 2 + 7)
  (h_geom : ∀ (b : ℕ → ℕ), (∀ n, a n + b n = 2^(n-1)) ∧ b n = 2^(n-1) - a n) :
  ∀ n : ℕ, (∑ i in finset.range (n + 1), b i * (a i + b i)) = ((4^n - 10) / 3 - (2*n - 3) * 2^n) :=
sorry

end problem_part_I_problem_part_II_l744_744130


namespace binomial_coefficient_proof_l744_744965

open Real BigOperators

-- Definition of region D and the point P(x, y)
def isInRegionD (x y : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Definition of the inequality y ≤ x^2
def satisfiesIneq (x y : ℝ) : Prop := y ≤ x^2

-- The statement we need to prove
theorem binomial_coefficient_proof :
  let a := (∫ x in -1..1, x^2) / 2 in
  a = 1 / 3 ∧
  (∑ r in finset.range 6, if 5 - (3 / 2 : ℝ) * r = 2 then 
    nat.choose 5 r * (-1) ^ r * 3 ^ (5 - r) else 0) = 270 := 
by {
  set a := (\int (x in -1..1), x^2) / 2,
  split,
  { simp [a], 
    sorry },
  { 
    simp [a],
    sorry 
  }
}

end binomial_coefficient_proof_l744_744965


namespace exponential_function_value_l744_744698

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem exponential_function_value :
  f (f 2) = 16 := by
  simp only [f]
  sorry

end exponential_function_value_l744_744698


namespace display_stands_arrangements_l744_744725

theorem display_stands_arrangements :
  ∃ n : ℕ, n = 48 ∧
    let stands := finset.range 9 in
    let valid_stands := stands \ {0, 8} in
    ∃ (s1 s2 s3 : ℕ) (h1 : s1 ∈ valid_stands) (h2 : s2 ∈ valid_stands) (h3 : s3 ∈ valid_stands),
      nat.abs (s1 - s2) > 1 ∧ nat.abs (s1 - s2) <= 2 ∧
      nat.abs (s2 - s3) > 1 ∧ nat.abs (s2 - s3) <= 2 ∧
      nat.abs (s1 - s3) > 1 ∧ nat.abs (s1 - s3) <= 2 ∧
      finset.card {s1, s2, s3} = 3 := sorry

end display_stands_arrangements_l744_744725


namespace perpendicular_lines_b_eq_neg_six_l744_744477

theorem perpendicular_lines_b_eq_neg_six
    (b : ℝ) :
    (∀ x y : ℝ, 3 * y + 2 * x - 4 = 0 → y = (-2/3) * x + 4/3) →
    (∀ x y : ℝ, 4 * y + b * x - 6 = 0 → y = (-b/4) * x + 3/2) →
    - (2/3) * (-b/4) = -1 →
    b = -6 := 
sorry

end perpendicular_lines_b_eq_neg_six_l744_744477


namespace ratio_of_areas_l744_744323

theorem ratio_of_areas (s : ℝ) (hs : 0 < s) :
  let longer_side_R := 1.2 * s
  let shorter_side_R := 0.85 * s
  let area_R := longer_side_R * shorter_side_R
  let area_S := s^2
  area_R / area_S = 51 / 50 :=
by
  let longer_side_R := 1.2 * s
  let shorter_side_R := 0.85 * s
  let area_R := longer_side_R * shorter_side_R
  let area_S := s^2
  calc
    area_R / area_S = (1.2 * s * 0.85 * s) / (s * s) : by rw [area_R, area_S]
    ... = 1.02 : by { field_simp [ne_of_gt hs], ring }
    ... = 51 / 50 : by norm_num

end ratio_of_areas_l744_744323


namespace mode_is_1_75_l744_744451

/-- Definition of the dataset containing heights and their frequencies -/
def height_frequencies : list (ℝ × ℕ) := [
  (1.55, 1),
  (1.60, 4),
  (1.65, 3),
  (1.70, 4),
  (1.75, 6),
  (1.80, 2)
]

/-- A function to find the mode in the provided dataset -/
def mode (dataset : list (ℝ × ℕ)) : ℝ :=
  dataset.foldr (λ x y, if x.2 > y.2 then x else y) (0, 0).1

/-- The theorem stating the mode of the dataset is 1.75 -/
theorem mode_is_1_75 : mode height_frequencies = 1.75 := sorry

end mode_is_1_75_l744_744451


namespace proof1_proof2_proof3_proof4_l744_744662

noncomputable def calc1 : ℝ := 3.21 - 1.05 - 1.95
noncomputable def calc2 : ℝ := 15 - (2.95 + 8.37)
noncomputable def calc3 : ℝ := 14.6 * 2 - 0.6 * 2
noncomputable def calc4 : ℝ := 0.25 * 1.25 * 32

theorem proof1 : calc1 = 0.21 := by
  sorry

theorem proof2 : calc2 = 3.68 := by
  sorry

theorem proof3 : calc3 = 28 := by
  sorry

theorem proof4 : calc4 = 10 := by
  sorry

end proof1_proof2_proof3_proof4_l744_744662


namespace prop_2_prop_3_l744_744073

variables {a b c : ℝ}

-- Proposition 2: a > |b| -> a^2 > b^2
theorem prop_2 (h : a > |b|) : a^2 > b^2 := sorry

-- Proposition 3: a > b -> a^3 > b^3
theorem prop_3 (h : a > b) : a^3 > b^3 := sorry

end prop_2_prop_3_l744_744073


namespace probability_of_all_selected_l744_744731

theorem probability_of_all_selected :
  let p_x := 1 / 7
  let p_y := 2 / 9
  let p_z := 3 / 11
  p_x * p_y * p_z = 1 / 115.5 :=
by
  let p_x := 1 / 7
  let p_y := 2 / 9
  let p_z := 3 / 11
  sorry

end probability_of_all_selected_l744_744731


namespace solve_for_m_l744_744179

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (2 / (2^x + 1)) + m

theorem solve_for_m (m : ℝ) : (∀ x : ℝ, f m (-x) = - (f m x)) ↔ m = -1 := by
sorry

end solve_for_m_l744_744179


namespace cos_double_angle_example_l744_744573

def cos_double_angle_identity (x : ℝ) : Prop :=
  cos (2 * x) = 1 - 2 * (sin x) ^ 2

theorem cos_double_angle_example : cos_double_angle_identity (x : ℝ) 
  (h : sin x = - 2 / 3) : cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_example_l744_744573


namespace lambda_equal_one_sufficient_unnecessary_l744_744151

noncomputable def vector_a (λ : ℝ) : ℝ × ℝ :=
(λ, -2)

noncomputable def vector_b (λ : ℝ) : ℝ × ℝ :=
(1 + λ, 1)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2

def is_orthogonal (v1 v2 : ℝ × ℝ) : Prop :=
dot_product v1 v2 = 0

theorem lambda_equal_one_sufficient_unnecessary :
  ∀ (λ : ℝ),
    is_orthogonal (vector_a λ) (vector_b λ) ↔ (λ = 1 → is_orthogonal (vector_a λ) (vector_b λ)) ∧ (∀ λ, is_orthogonal (vector_a λ) (vector_b λ) ↔ λ = 1 ∨ λ = -2) :=
by
  sorry

end lambda_equal_one_sufficient_unnecessary_l744_744151


namespace tic_tac_toe_tie_fraction_l744_744951

theorem tic_tac_toe_tie_fraction 
  (fraction_james_wins : ℚ)
  (fraction_mary_wins : ℚ) :
  fraction_james_wins = 4/9 → 
  fraction_mary_wins = 5/18 → 
  (1 - (fraction_james_wins + fraction_mary_wins)) = 5/18 :=
by 
  intros h1 h2
  rw [h1, h2]
  have := 4/9 + 5/18
  norm_num at this
  exact this

-- Summary:
-- Given:
-- 1. James wins 4/9 of the games.
-- 2. Mary wins 5/18 of the games.
-- Prove:
-- The fraction of games that result in a tie (in months other than February) is 5/18.

end tic_tac_toe_tie_fraction_l744_744951


namespace incorrect_reasoning_D_l744_744002

variables {A B C : Type} {l α β : set Type} [linear_order Type]

-- Conditions
variable (h1 : A ∈ l ∧ A ∈ α ∧ B ∈ l ∧ B ∈ α)
variable (h2 : A ∈ α ∧ A ∈ β ∧ B ∈ α ∧ B ∈ β)
variable (h3 : C ∈ α ∧ A ∈ β ∧ B ∈ β ∧ C ∈ β ∧ ¬collinear A B C)
variable (h4 : ¬ (l ⊆ α) ∧ A ∈ l)

-- Lean 4 statement for the equivalent proof problem
theorem incorrect_reasoning_D : 
  (¬(l ⊆ α) ∧ A ∈ l → ¬(A ∈ α)) :=
  sorry

end incorrect_reasoning_D_l744_744002


namespace ratio_area_rectangle_to_square_l744_744324

variable (s : ℝ)
variable (area_square : ℝ := s^2)
variable (longer_side_rectangle : ℝ := 1.2 * s)
variable (shorter_side_rectangle : ℝ := 0.85 * s)
variable (area_rectangle : ℝ := longer_side_rectangle * shorter_side_rectangle)

theorem ratio_area_rectangle_to_square :
  area_rectangle / area_square = 51 / 50 := by
  sorry

end ratio_area_rectangle_to_square_l744_744324


namespace number_of_ways_to_choose_committee_l744_744266

-- Definitions of the conditions
def eligible_members : ℕ := 30
def new_members : ℕ := 3
def committee_size : ℕ := 5
def eligible_pool : ℕ := eligible_members - new_members

-- Problem statement to prove
theorem number_of_ways_to_choose_committee : (Nat.choose eligible_pool committee_size) = 80730 := by
  -- This space is reserved for the proof which is not required per instructions.
  sorry

end number_of_ways_to_choose_committee_l744_744266


namespace prove_bc_prove_area_l744_744227

variable {a b c A B C : ℝ}

-- Given conditions as Lean definitions
def condition_cos_A : Prop := (b^2 + c^2 - a^2 = 2 * cos A)
def condition_bc : Prop := b * c = 1
def condition_trig_identity : Prop := (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1

-- Definitions for the questions translated to Lean
def find_bc : Prop :=
  condition_cos_A → b * c = 1

def find_area (bc_value : Prop) : Prop :=
  bc_value →
  condition_trig_identity →
  1/2 * b * c * sin A = (Real.sqrt 3) / 4

-- Theorems to be proven
theorem prove_bc : find_bc :=
by
  intro h1
  sorry

theorem prove_area : find_area condition_bc :=
by
  intro h1 h2
  sorry

end prove_bc_prove_area_l744_744227


namespace cos_double_angle_l744_744577

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l744_744577


namespace problem_1_problem_2_problem_3_l744_744910

noncomputable def f (x a : ℝ) : ℝ := (Real.log (x + 1)) + a * x

theorem problem_1 (a : ℝ) (h : f 0 a = @Real.max ℝ (f 0  a)) : a = -1 :=
sorry

theorem problem_2 (a : ℝ) (h : ∃ x : ℝ, x ∈ set.Icc 1 2 ∧ (1 / (x + 1) + a) ≥ 2 * x) : a ≥ 3 / 2 :=
sorry

theorem problem_3 (a : ℝ) : 
  (a ≥ 0 → (∀ x ∈ set.Ioo -1 (1 : ℝ), f x a > f 0 a) ∧ (∀ x ∈ set.Ioo 1 (0 : ℝ), f x a < f 0 a))
  ∧
  (a < 0 → ∃ b : ℝ, b = -(1 / a) - 1 ∧ 
    ((∀ x ∈ set.Ioo -1 b, f x a > f 0 a) ∧ 
     (∀ x ∈ set.Ioo b 0, f x a < f 0 a))) :=
sorry

end problem_1_problem_2_problem_3_l744_744910


namespace chocolates_sold_in_fourth_week_l744_744364

theorem chocolates_sold_in_fourth_week :
  ∀ (week1 week2 week3 week5 totalWeeks : ℕ) (mean : ℕ),
  week1 = 75 →
  week2 = 67 →
  week3 = 75 →
  week5 = 68 →
  totalWeeks = 5 →
  mean = 71 →
  let totalChocolates := mean * totalWeeks,
      knownWeeksChocolates := week1 + week2 + week3 + week5,
      week4 := totalChocolates - knownWeeksChocolates
  in week4 = 70 :=
by
  intros week1 week2 week3 week5 totalWeeks mean h1 h2 h3 h4 h5 h6
  let totalChocolates := mean * totalWeeks
  let knownWeeksChocolates := week1 + week2 + week3 + week5
  let week4 := totalChocolates - knownWeeksChocolates
  simp [h1, h2, h3, h4, h5, h6]
  exact (71 * 5 - (75 + 67 + 75 + 68) = 70) sorry

end chocolates_sold_in_fourth_week_l744_744364


namespace sum_sequence_ratio_l744_744903

theorem sum_sequence_ratio :
  (∀ n : ℕ, S n = 2^n - 1) → (S 4 / a 3 = 15 / 4) :=
by
  intro h
  sorry

end sum_sequence_ratio_l744_744903


namespace prob_twins_prob_twins_in_three_children_expected_number_pairs_twins_l744_744714

variables (p : ℝ) (N : ℝ)

-- Condition: 0 ≤ p ≤ 1
axiom probability_nonnegative_and_limited (hp : 0 ≤ p ∧ p ≤ 1) : True

-- Part (a)
theorem prob_twins (hp : 0 ≤ p ∧ p ≤ 1) : (2 * p) / (p + 1) = \frac{2p}{p + 1} := 
by {
  sorry
}

-- Part (b)
theorem prob_twins_in_three_children (hp : 0 ≤ p ∧ p ≤ 1) : (2 * p) / (2 * p + (1 - p)^2) = \frac{2p}{2p + (1 - p)^2} := 
by {
  sorry
}

-- Part (c)
theorem expected_number_pairs_twins (hp : 0 ≤ p ∧ p ≤ 1) : (N * p) / (p + 1) = \frac{Np}{p + 1} := 
by {
  sorry
}

end prob_twins_prob_twins_in_three_children_expected_number_pairs_twins_l744_744714


namespace carson_clawed_total_l744_744460

theorem carson_clawed_total :
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  wombats * wombat_claws + rheas * rhea_claws = 39 := by
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  show wombats * wombat_claws + rheas * rhea_claws = 39
  sorry

end carson_clawed_total_l744_744460


namespace triangle_segments_probability_l744_744803

noncomputable def probability_triangle_segments : ℝ :=
let L : ℝ := 1 in
(∫ x in (L/4)..(L/2), 1) / (∫ x in 0..(L/2), 1)

theorem triangle_segments_probability : probability_triangle_segments = 1 / 2 :=
by
  sorry

end triangle_segments_probability_l744_744803


namespace derivative_at_pi_over_3_l744_744543

/-- Given the function f(x) = sin(2x + π/3), prove that the derivative evaluated at x = π/3 is -2. -/
theorem derivative_at_pi_over_3 : 
  let f := λ x : ℝ, Real.sin (2 * x + Real.pi / 3) in
  let f' := λ x : ℝ, Deriv (λ x, f x) in
  f' (Real.pi / 3) = -2 := 
by 
  sorry

end derivative_at_pi_over_3_l744_744543


namespace sufficient_but_not_necessary_condition_l744_744901

-- Definition of a point in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Definition of a line passing through a point with a certain slope
def Line_through (p : Point) (k : ℝ) := ∃ b : ℝ, ∀ x : ℝ, p.y = k * p.x + b

-- Definition of a circle with center and radius
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Definitions of the given conditions
def point := Point.mk 0 5
def circle := Circle.mk (Point.mk 3 0) 3

-- Main theorem
theorem sufficient_but_not_necessary_condition :
  ∀ l_slope : ℝ, l_slope = -8/15 → (let line := Line_through point l_slope in
  ∀ (circ : Circle), circ = circle →
  ∀ tangent : (Line_through point (-8/15)) → tangent ∧ ¬ ∀ (slope : ℝ), slope = l_slope) :=
by
  intro l_slope hl_slope
  intro line circ hc tangent
  sorry

end sufficient_but_not_necessary_condition_l744_744901


namespace area_of_midpoints_of_segments_l744_744313

noncomputable def side_length : ℝ := 4
noncomputable def segment_length : ℝ := Real.sqrt 18
noncomputable def area_enclosed_by_midpoints : ℝ := (9 * Real.pi) / 2
noncomputable def area_enclosed_by_midpoints_approx : ℝ := 14.14

theorem area_of_midpoints_of_segments : 100 * area_enclosed_by_midpoints_approx = 1414 := by
  have h1 : (side_length = 4) := rfl
  have h2 : (segment_length = Real.sqrt 18) := rfl
  have h3 : (area_enclosed_by_midpoints = (9 * Real.pi) / 2) := rfl
  have h4 : (area_enclosed_by_midpoints ≈ 14.14) := rfl
  sorry

end area_of_midpoints_of_segments_l744_744313


namespace star_3_5_l744_744874

def star (a b : ℕ) : ℕ := a^2 + 3 * a * b + b^2

theorem star_3_5 : star 3 5 = 79 := 
by
  sorry

end star_3_5_l744_744874


namespace probability_even_product_is_5_over_6_l744_744381

noncomputable def probability_even_product : ℚ :=
  let s := ({1, 2, 3, 4} : Finset ℕ)
  let pairs := (s.product s).filter (λ (p : ℕ × ℕ), p.1 < p.2)
  let even_pairs := pairs.filter (λ (p : ℕ × ℕ), (p.1 * p.2) % 2 = 0)
  (even_pairs.card : ℚ) / pairs.card

theorem probability_even_product_is_5_over_6 : probability_even_product = 5 / 6 := sorry

end probability_even_product_is_5_over_6_l744_744381


namespace solve_exponent_l744_744017

theorem solve_exponent :
  ∃ (n : ℕ), 112 * 5^n = 70000 ∧ n = 4 :=
by
  use 4
  split
  {
    calc
    112 * 5^4 = 112 * 625 : by rw pow_succ
    ... = 70000 : by norm_num,
    done}

end solve_exponent_l744_744017


namespace image_of_center_after_transformations_l744_744070

-- Define the initial center of circle C
def initial_center : ℝ × ℝ := (3, -4)

-- Define a function to reflect a point across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define a function to translate a point by some units left
def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

-- Define the final coordinates after transformations
def final_center : ℝ × ℝ :=
  translate_left (reflect_x_axis initial_center) 5

-- The theorem to prove
theorem image_of_center_after_transformations :
  final_center = (-2, 4) :=
by
  sorry

end image_of_center_after_transformations_l744_744070


namespace construct_line_through_points_l744_744111

-- Definitions of the conditions
def points_on_sheet (A B : ℝ × ℝ) : Prop := A ≠ B
def tool_constraints (ruler_length compass_max_opening distance_A_B : ℝ) : Prop :=
  distance_A_B > 2 * ruler_length ∧ distance_A_B > 2 * compass_max_opening

-- The main theorem statement
theorem construct_line_through_points (A B : ℝ × ℝ) (ruler_length compass_max_opening : ℝ) 
  (h_points : points_on_sheet A B) 
  (h_constraints : tool_constraints ruler_length compass_max_opening (dist A B)) : 
  ∃ line : ℝ × ℝ → Prop, line A ∧ line B :=
sorry

end construct_line_through_points_l744_744111


namespace passing_probability_l744_744675

theorem passing_probability :
  let num_students := 6
  let probability :=
    1 - (2/6) * (2/5) * (2/4) * (2/3) * (2/2)
  probability = 44 / 45 :=
by
  let num_students := 6
  let probability :=
    1 - (2/6) * (2/5) * (2/4) * (2/3) * (2/2)
  have p_eq : probability = 44 / 45 := sorry
  exact p_eq

end passing_probability_l744_744675


namespace bc_is_one_area_of_triangle_l744_744234

-- Define a triangle with sides a, b, and c and corresponding angles A, B, and C.
structure Triangle :=
(a b c A B C : ℝ)

-- Assume the Law of Cosines condition is given.
axiom Law_of_Cosines_condition (T : Triangle) : (T.b^2 + T.c^2 - T.a^2) / (Real.cos T.A) = 2

-- Prove that bc = 1.
theorem bc_is_one (T : Triangle) (h : Law_of_Cosines_condition T) : T.b * T.c = 1 := 
by {
    sorry
}

-- Assume the given ratio condition.
axiom ratio_condition (T : Triangle) :
  (T.a * Real.cos T.B - T.b * Real.cos T.A) / (T.a * Real.cos T.B + T.b * Real.cos T.A)
  - (T.b / T.c) = 1

-- Prove that the area of the triangle is sqrt(3)/4.
theorem area_of_triangle (T : Triangle) (h1 : Law_of_Cosines_condition T) (h2 : ratio_condition T) :
  let S := 0.5 * T.b * T.c * Real.sin T.A
  in S = Real.sqrt 3 / 4 :=
by {
    sorry
}

end bc_is_one_area_of_triangle_l744_744234


namespace solve_for_x_l744_744569

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 = -2 * x + 10) : x = 3 := 
sorry

end solve_for_x_l744_744569


namespace solve_for_n_l744_744311

theorem solve_for_n (n : ℤ) : (3 : ℝ)^(2 * n + 2) = 1 / 9 ↔ n = -2 := by
  sorry

end solve_for_n_l744_744311


namespace lattice_points_count_l744_744190

def is_lattice_point (p: ℤ × ℤ) : Prop := 
  ∃ x y : ℤ, p = (x, y)

def on_line (x y : ℤ) : Prop := 
  7 * x + 11 * y = 77

def on_coordinate_axes (p: ℤ × ℤ) : Prop := 
  p.1 = 0 ∨ p.2 = 0

def in_triangle (p: ℤ × ℤ) : Prop :=
  (on_coordinate_axes p ∨ on_line p.fst p.snd) ∧
  0 ≤ p.fst ∧ p.fst ≤ 11 ∧
  0 ≤ p.snd ∧ p.snd ≤ 7 

def lattice_points_in_triangle : ℤ :=
  (set_of (λ p, is_lattice_point p ∧ in_triangle p)).to_finset.card

theorem lattice_points_count : lattice_points_in_triangle = 49 := 
  sorry

end lattice_points_count_l744_744190


namespace jacqueline_bob_sugar_l744_744627

variables (L J B : ℝ)

theorem jacqueline_bob_sugar (h1 : J = 0.60 * L) (h2 : B = 0.70 * L) : (J / B) = 0.857142857142857 {
  -- proof here
  sorry
}

end jacqueline_bob_sugar_l744_744627


namespace speed_conversion_l744_744842

theorem speed_conversion (s : ℚ) (h : s = 13 / 48) : 
  ((13 / 48) * 3.6 = 0.975) :=
by
  sorry

end speed_conversion_l744_744842


namespace find_bc_find_area_l744_744242

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Condition 1: From the given problem
variable (h1 : (b^2 + c^2 - a^2) / (cos A) = 2)

-- Condition 2: From the given problem
variable (h2 : (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1)

theorem find_bc : bc = 1 :=
sorry

theorem find_area (area_abc : ℝ) : area_abc = (sqrt 3) / 4 :=
sorry

end find_bc_find_area_l744_744242


namespace exist_two_digit_pairs_l744_744008

theorem exist_two_digit_pairs :
  (∃ x y : ℕ, 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧
    10 ≤ x + 15 ∧ x + 15 < 100 ∧
    10 ≤ y - 20 ∧ y - 20 < 100 ∧
    (x + 15) * (y - 20) = x * y) ↔
  (16) :=
sorry

end exist_two_digit_pairs_l744_744008


namespace find_k_b_l744_744251

-- Define the sets A and B
def A : Set (ℝ × ℝ) := { p | ∃ x y: ℝ, p = (x, y) }
def B : Set (ℝ × ℝ) := { p | ∃ x y: ℝ, p = (x, y) }

-- Define the mapping f
def f (p : ℝ × ℝ) (k b : ℝ) : ℝ × ℝ := (k * p.1, p.2 + b)

-- Define the conditions
def condition (f : (ℝ × ℝ) → ℝ × ℝ) :=
  f (3,1) = (6,2)

-- Statement: Prove that the values of k and b are 2 and 1 respectively
theorem find_k_b : ∃ (k b : ℝ), f (3, 1) k b = (6, 2) ∧ k = 2 ∧ b = 1 :=
by
  sorry

end find_k_b_l744_744251


namespace sufficient_condition_for_perpendicular_line_plane_l744_744895

noncomputable def perpendicular_planes_implies_perpendicular_line_plane
  (α β : Plane) (m : Line) : Prop :=
  (parallel α β) ∧ (perpendicular m β) → perpendicular m α

-- Proof
theorem sufficient_condition_for_perpendicular_line_plane
  (α β : Plane) (m : Line)
  (h1 : parallel α β) (h2 : perpendicular m β) : perpendicular m α :=
by
  sorry

end sufficient_condition_for_perpendicular_line_plane_l744_744895


namespace find_bc_find_area_l744_744239

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Condition 1: From the given problem
variable (h1 : (b^2 + c^2 - a^2) / (cos A) = 2)

-- Condition 2: From the given problem
variable (h2 : (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1)

theorem find_bc : bc = 1 :=
sorry

theorem find_area (area_abc : ℝ) : area_abc = (sqrt 3) / 4 :=
sorry

end find_bc_find_area_l744_744239


namespace mary_balloons_l744_744099

-- Definitions for the conditions.
def fred_balloons : ℕ := 5
def sam_balloons : ℕ := 6
def total_balloons : ℕ := 18

-- The theorem stating that Mary has 7 balloons given the above conditions.
theorem mary_balloons : ℕ :=
  total_balloons - fred_balloons - sam_balloons = 7 :=
by
  sorry

end mary_balloons_l744_744099


namespace square_root_of_4m_minus_5n_is_pm7_l744_744601

theorem square_root_of_4m_minus_5n_is_pm7 (m n : ℤ) 
  (h₁ : Real.sqrt (m - 2) = 3)
  (h₂ : Real.cbrt (-64) = 7 * n + 3) : 
  Real.sqrt (4 * m - 5 * n) = 7 ∨ Real.sqrt (4 * m - 5 * n) = -7 := 
sorry

end square_root_of_4m_minus_5n_is_pm7_l744_744601


namespace movies_watched_l744_744341

theorem movies_watched (t u : ℕ) (h_t : t = 17) (h_u : u = 10) : t - u = 7 :=
by
  rw [h_t, h_u]
  norm_num

end movies_watched_l744_744341


namespace modulo_problem_l744_744993

theorem modulo_problem (m : ℤ) (hm : 0 ≤ m ∧ m < 41) (hmod : 4 * m ≡ 1 [MOD 41]) :
  ((3 ^ m) ^ 4 - 3) % 41 = 37 := 
by
  sorry

end modulo_problem_l744_744993


namespace grill_cost_difference_l744_744339

theorem grill_cost_difference:
  let in_store_price : Float := 129.99
  let payment_per_installment : Float := 32.49
  let number_of_installments : Float := 4
  let shipping_handling : Float := 9.99
  let total_tv_cost : Float := (number_of_installments * payment_per_installment) + shipping_handling
  let cost_difference : Float := in_store_price - total_tv_cost
  cost_difference * 100 = -996 := by
    sorry

end grill_cost_difference_l744_744339


namespace three_tails_probability_l744_744101

noncomputable def prob_three_tails : ℚ := 
  let p_first_5_heads := (1 : ℚ) / 4
  let p_first_5_tails := (3 : ℚ) / 4
  let p_last_5_heads := (1 : ℚ) / 2
  let p_last_5_tails := (1 : ℚ) / 2
  let binom := λ n k : ℕ, (Nat.choose n k : ℚ)
  (finset.range 4).sum (λ k : ℕ,
    binom 5 k * (p_first_5_tails ^ k) * (p_first_5_heads ^ (5 - k)) * 
    binom 5 (3 - k) * (p_last_5_tails ^ (3 - k)) * (p_last_5_heads ^ (2 + k)))

theorem three_tails_probability :
  prob_three_tails = 55 / 2048 := 
sorry

end three_tails_probability_l744_744101


namespace g_symmetric_about_x_equals_1_l744_744074

def floor_function (x : ℝ) : ℤ := int.floor x

def g (x : ℝ) : ℝ := abs (floor_function (2 * x)) - abs (floor_function (2 - 2 * x))

theorem g_symmetric_about_x_equals_1 :
  ∀ x : ℝ, g (1 + x) = g (1 - x) :=
begin
  sorry
end

end g_symmetric_about_x_equals_1_l744_744074


namespace integer_solutions_equation_l744_744157

theorem integer_solutions_equation : 
  (∃ x y : ℤ, (1 / (2022 : ℚ) = 1 / (x : ℚ) + 1 / (y : ℚ))) → 
  ∃! (n : ℕ), n = 53 :=
by
  sorry

end integer_solutions_equation_l744_744157


namespace possible_values_of_m_l744_744907

def F1 := (-3, 0)
def F2 := (3, 0)
def possible_vals := [2, -1, 4, -3, 1/2]

noncomputable def is_valid_m (m : ℝ) : Prop :=
  abs (2 * m - 1) < 6 ∧ m ≠ 1/2

theorem possible_values_of_m : {m ∈ possible_vals | is_valid_m m} = {2, -1} := by
  sorry

end possible_values_of_m_l744_744907


namespace Tom_earns_per_week_l744_744354

-- Definitions based on conditions
def crab_buckets_per_day := 8
def crabs_per_bucket := 12
def price_per_crab := 5
def days_per_week := 7

-- The proof goal
theorem Tom_earns_per_week :
  (crab_buckets_per_day * crabs_per_bucket * price_per_crab * days_per_week) = 3360 := by
  sorry

end Tom_earns_per_week_l744_744354


namespace parallelogram_area_Lean_l744_744224

open Real EuclideanSpace

noncomputable def parallelogram_area (p q : EuclideanSpace ℝ (Fin 3))
  (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (hθ : ∃ θ, θ = π / 4 ∧ cos θ = (p ⬝ q)) : ℝ :=
  ‖ ((3 : ℝ) • q - p) × ((3 : ℝ) • p + (3 : ℝ) • q) ‖

theorem parallelogram_area_Lean (p q : EuclideanSpace ℝ (Fin 3))
  (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (hθ : ∃ θ, θ = π / 4 ∧ cos θ = (p ⬝ q)) :
  parallelogram_area p q hp hq hθ = 9 * Real.sqrt 2 / 4 :=
sorry

end parallelogram_area_Lean_l744_744224


namespace proving_sequencing_statements_l744_744512

def seq (n : ℕ) : ℤ :=
  match n with
  | 1 => 1
  | 2 => 3
  | n + 2 => seq (n + 1) - seq n

def S (n : ℕ) : ℤ :=
  (Finset.range n).sum (λ i => seq (i + 1))

theorem proving_sequencing_statements :
  seq 100 = -1 ∧ S 100 = 5 :=
by
  sorry

end proving_sequencing_statements_l744_744512


namespace negation_exists_to_forall_l744_744703

theorem negation_exists_to_forall :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by
  sorry

end negation_exists_to_forall_l744_744703


namespace find_r_l744_744164

variable (m r : ℝ)

theorem find_r (h1 : 5 = m * 3^r) (h2 : 45 = m * 9^(2 * r)) : r = 2 / 3 := by
  sorry

end find_r_l744_744164


namespace power_comparison_l744_744757

theorem power_comparison (A B : ℝ) (h1 : A = 1997 ^ (1998 ^ 1999)) (h2 : B = 1999 ^ (1998 ^ 1997)) (h3 : 1997 < 1999) :
  A > B :=
by
  sorry

end power_comparison_l744_744757


namespace box_length_is_12_l744_744426

-- Definitions of given conditions
def width : ℝ := 16
def height : ℝ := 6
def cube_vol : ℝ := 3
def num_cubes : ℝ := 384

-- The volume of the box should be num_cubes * cube_vol
def box_volume : ℝ := num_cubes * cube_vol

-- Statement that specifies what needs to be proved
theorem box_length_is_12 :
  ∃ (length : ℝ), length * width * height = box_volume ∧ length = 12 := 
by 
  let length := 12
  use length
  split
  have vol_calc : box_volume = 1152 := rfl
  simp [box_volume, length, width, height],
  simp [length],
  sorry

end box_length_is_12_l744_744426


namespace acres_used_for_corn_l744_744393

theorem acres_used_for_corn (total_land : ℕ) (ratio_beans : ℕ) (ratio_wheat : ℕ) (ratio_corn : ℕ)
  (total_ratio_parts : ℕ) (one_part_size : ℕ) :
  total_land = 1034 →
  ratio_beans = 5 →
  ratio_wheat = 2 →
  ratio_corn = 4 →
  total_ratio_parts = ratio_beans + ratio_wheat + ratio_corn →
  one_part_size = total_land / total_ratio_parts →
  ratio_corn * one_part_size = 376 :=
by
  intros
  sorry

end acres_used_for_corn_l744_744393


namespace pencils_total_l744_744342

theorem pencils_total (original_pencils : ℕ) (added_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : original_pencils = 41) 
  (h2 : added_pencils = 30) 
  (h3 : total_pencils = original_pencils + added_pencils) : 
  total_pencils = 71 := 
by
  sorry

end pencils_total_l744_744342


namespace cost_price_per_meter_l744_744438

namespace ClothCost

theorem cost_price_per_meter (selling_price_total : ℝ) (meters_sold : ℕ) (loss_per_meter : ℝ) : 
  selling_price_total = 18000 → 
  meters_sold = 300 → 
  loss_per_meter = 5 →
  (selling_price_total / meters_sold) + loss_per_meter = 65 := 
by
  intros hsp hms hloss
  sorry

end ClothCost

end cost_price_per_meter_l744_744438


namespace sum_after_operations_l744_744337

theorem sum_after_operations (a b S : ℝ) (h : a + b = S) : 
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := 
by 
  sorry

end sum_after_operations_l744_744337


namespace overall_pass_percentage_is_correct_l744_744410

def total_students1 := 40
def pass_percentage1 := 1.0
def total_students2 := 50
def pass_percentage2 := 0.9
def total_students3 := 60
def pass_percentage3 := 0.8

def total_passed := total_students1 * pass_percentage1 + total_students2 * pass_percentage2 + total_students3 * pass_percentage3
def total_appeared := total_students1 + total_students2 + total_students3
def overall_pass_percentage := (total_passed / total_appeared) * 100

theorem overall_pass_percentage_is_correct : overall_pass_percentage = 88.67 := by
  -- calculation details, but omitted as we were instructed to skip the proof
  sorry

end overall_pass_percentage_is_correct_l744_744410


namespace convert_38_to_binary_l744_744839

theorem convert_38_to_binary :
  let option_a_in_decimal : ℕ := 2 + 8 + 32,
      option_b_in_decimal : ℕ := 2 + 4 + 32,
      option_c_in_decimal : ℕ := 4 + 16 + 32,
      option_d_in_decimal : ℕ := 2 + 8 + 32
  in
    option_b_in_decimal = 38 :=
by {
  let option_a_in_decimal := 2 + 8 + 32,
  let option_b_in_decimal := 2 + 4 + 32,
  let option_c_in_decimal := 4 + 16 + 32,
  let option_d_in_decimal := 2 + 8 + 32,
  have h_b : option_b_in_decimal = 38 := by norm_num,
  show option_b_in_decimal = 38 from h_b
}

end convert_38_to_binary_l744_744839


namespace roots_squared_sum_l744_744645

theorem roots_squared_sum : 
  ∀ r s : ℝ, (r ≠ s) → (r ^ 2 - 5 * r + 6 = 0) → (s ^ 2 - 5 * s + 6 = 0) → r^2 + s^2 = 13 := 
by 
  intros r s hr hs hs' 
  have h1 : r + s = 5 := by { have := hs, ring_nf } 
  have h2 : r * s = 6 := by { have := hs', ring_nf } 
  rw [←h1, ←h2] 
  ring 
  sorry

end roots_squared_sum_l744_744645


namespace find_f_8_l744_744860

def f (n : ℕ) : ℕ := n^2 - 3 * n + 20

theorem find_f_8 : f 8 = 60 := 
by 
sorry

end find_f_8_l744_744860


namespace intervals_of_monotonicity_range_of_a_inequality_for_m_n_l744_744544

def f (a x : ℝ) := a * log x + (1/2) * x^2 - (1 + a) * x

theorem intervals_of_monotonicity (a : ℝ) :
  (a ≤ 0 → ∀ x, (0 < x ∧ x < 1 → f' a x < 0) ∧ (x > 1 → f' a x > 0)) ∧
  (0 < a ∧ a < 1 → ∀ x, (a < x ∧ x < 1 → f' a x < 0) ∧ ((0 < x ∧ x < a) ∨ (x > 1) → f' a x > 0)) ∧
  (a = 1 → ∀ x, x > 0 → f' a x ≥ 0) ∧
  (a > 1 → ∀ x, (1 < x ∧ x < a → f' a x < 0) ∧ ((0 < x ∧ x < 1) ∨ (x > a) → f' a x > 0)) :=
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x, f a x ≥ 0) → a ≤ - (1 / 2) :=
  sorry

theorem inequality_for_m_n (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  ∑ k in range (m + 1) (m + n + 1), 1 / log k > n / (m * (m + n)) :=
  sorry

end intervals_of_monotonicity_range_of_a_inequality_for_m_n_l744_744544


namespace perfect_set_conclusions_l744_744176

def is_perfect_set (A : Set ℚ) : Prop :=
  (0 ∈ A ∧ 1 ∈ A) ∧
  (∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A) ∧
  (∀ x, x ∈ A → x ≠ 0 → (1 / x) ∈ A)

theorem perfect_set_conclusions :
  let B := {-1, 0, 1}
  let Q := {q : ℚ | True}
  ∀ A : Set ℚ,
    is_perfect_set A →
    (¬ is_perfect_set B ∧ is_perfect_set Q ∧
     (∀ x y, x ∈ A → y ∈ A → (x + y) ∈ A) ∧
     (∀ x y, x ∈ A → y ∈ A → (x * y) ∈ A) ∧
     (∀ x y, x ∈ A → y ∈ A → x ≠ 0 → (y / x) ∈ A)) :=
sorry

end perfect_set_conclusions_l744_744176


namespace find_platform_length_l744_744068

noncomputable def platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600 in
  let total_distance := train_speed_mps * crossing_time in
  total_distance - train_length

theorem find_platform_length :
  platform_length 100 60 14.998800095992321 = 150.0133344 := 
by
  apply rfl

end find_platform_length_l744_744068


namespace initial_gasohol_quantity_l744_744422

-- Definitions for the conditions
def initial_gasohol (x : ℝ) := 0.05 * x
def added_ethanol := 2.5
def final_ethanol (x : ℝ) := (initial_gasohol x) + added_ethanol
def final_volume (x : ℝ) := x + added_ethanol

-- Condition stating the desired ethanol percentage in the final mixture
def optimal_ethanol_mixture (x : ℝ) := final_ethanol x = 0.10 * final_volume x

-- Theorem stating that the initial amount of gasohol added was 45 liters based on given conditions
theorem initial_gasohol_quantity : ∃ x : ℝ, optimal_ethanol_mixture x ∧ x = 45 :=
by
  sorry

end initial_gasohol_quantity_l744_744422


namespace proof_problem_l744_744877

-- Define the function f using conditional cases
def f (x : ℝ) : ℝ :=
  if x >= 0 then
    x^2 + x
  else
    g x

-- Define the function g for x < 0
def g (x : ℝ) : ℝ :=
  -x^2 + x

-- f is an odd function
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- Proving the desired properties
theorem proof_problem : 
  f(-1) = -2 ∧ 
  f(f(1)) = 6 ∧ 
  (∀ x : ℝ, x < 0 → g(x) = -x^2 + x) :=
by {
  sorry
}

end proof_problem_l744_744877


namespace det_dilation_matrix_l744_744223

section DilationMatrixProof

def E : Matrix (Fin 3) (Fin 3) ℝ := !![5, 0, 0; 0, 5, 0; 0, 0, 5]

theorem det_dilation_matrix :
  Matrix.det E = 125 :=
by {
  sorry
}

end DilationMatrixProof

end det_dilation_matrix_l744_744223


namespace range_of_x_l744_744138

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)

theorem range_of_x (a b x : ℝ) (h : a ≠ 0) (h₁ : abs (a + b) + abs (a - b) ≥ abs a * f x) :
  0 ≤ x ∧ x ≤ 4 :=
sorry

end range_of_x_l744_744138


namespace number_of_pieces_of_tape_l744_744000

variable (length_of_tape : ℝ := 8.8)
variable (overlap : ℝ := 0.5)
variable (total_length : ℝ := 282.7)

theorem number_of_pieces_of_tape : 
  ∃ (N : ℕ), total_length = length_of_tape + (N - 1) * (length_of_tape - overlap) ∧ N = 34 :=
sorry

end number_of_pieces_of_tape_l744_744000


namespace estimate_total_fish_l744_744042

theorem estimate_total_fish (m n k : ℕ) (hk : k ≠ 0) (hm : m ≠ 0) (hn : n ≠ 0):
  ∃ x : ℕ, x = (m * n) / k :=
by
  sorry

end estimate_total_fish_l744_744042


namespace lucky_ticket_exists_l744_744417

def is_lucky_ticket (n : ℕ) : Prop :=
  let digits := (n / 100000 % 10, n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10)
  let (a, b, c, x, y, z) := digits in 
  a + b + c = x + y + z

theorem lucky_ticket_exists : ∀ (start : ℕ), ∃ (n : ℕ), start ≤ n ∧ n < start + 1001 ∧ is_lucky_ticket n := by
  sorry

end lucky_ticket_exists_l744_744417


namespace domain_of_expression_l744_744489

theorem domain_of_expression (x : ℝ) :
  (∃ y : ℝ, y = (sqrt (x - 4)) / (sqrt (7 - x) + 1)) ↔ (4 ≤ x ∧ x ≤ 7) :=
by sorry

end domain_of_expression_l744_744489


namespace range_of_a_l744_744538

theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, a * x^2 + a * x + 1 > 0) : a ∈ Set.Icc 0 4 :=
sorry

end range_of_a_l744_744538


namespace closest_approximation_of_w_l744_744399

def approx_w : Real := ((69.28 * 0.004) / 0.03)

theorem closest_approximation_of_w :
  Real.round (approx_w * 100) / 100 = 9.24 :=
by
  sorry

end closest_approximation_of_w_l744_744399


namespace max_min_value_l744_744303

theorem max_min_value (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2 + 3) :
  let M := max (fun V => ∃ A B C : ℝ, A = x + y + z ∧ B = x^2 + y^2 + z^2 ∧ C = xy + xz + yz ∧ 5 * A = B + 3 ∧ V = C) in
  let m := min (fun V => ∃ A B C : ℝ, A = x + y + z ∧ B = x^2 + y^2 + z^2 ∧ C = xy + xz + yz ∧ 5 * A = B + 3 ∧ V = C) in
  M + 10 * m = 2 :=
by
  sorry

end max_min_value_l744_744303


namespace find_BD_distance_l744_744622

noncomputable def triangle_sides (A B C D : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D] :=
AC_distance : dist A C = 10
BC_distance : dist B C = 10
AD_distance : dist A D = 13
CD_distance : dist C D = 5

theorem find_BD_distance {A B C D : Type} [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D] 
  (AC_distance : dist A C = 10) (BC_distance : dist B C = 10) (AD_distance : dist A D = 13) (CD_distance : dist C D = 5) : 
  ∃ BD, dist B D ≈ 5.8595 :=
sorry

end find_BD_distance_l744_744622


namespace r_expansion_l744_744936

theorem r_expansion (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by
  sorry

end r_expansion_l744_744936


namespace range_of_m_l744_744540

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x - π / 6) - 1/2
noncomputable def g (x : ℝ) : ℝ := sqrt 3 * sin (2 * x - π / 3) - 1/2

theorem range_of_m :
  (∀ x ∈ set.Icc 0 (π / 3), ∃ m, g x - 3 ≤ m ∧ m ≤ g x + 3) ↔ (∃ m, m ∈ set.Icc (-2:ℝ) 1) :=
sorry

end range_of_m_l744_744540


namespace angle_bisectors_concurrent_iff_eq_segments_l744_744511

variable {A B C D P Q R : Point}

-- Assume quadrilateral ABCD is inscribed in a circle
axiom inscribed (ABCD : Quadrilateral) : inscribed ABCD

-- Definitions for perpendicular bases from D
axiom perpendicular_base (D : Point) (BC CA AB : Line) (P Q R : Point) :
  is_perpendicular D BC P ∧ is_perpendicular D CA Q ∧ is_perpendicular D AB R

-- Definitions of angle bisectors intersection property
axiom angle_bisectors_intersect_at (α β θ : Angle) (AC : Line) :
  bisects_angle α (at_points B C) ∧ bisects_angle β (at_points D C)
  ∧ bisects_diagonal θ (at_point A C)

-- Angles definitions
noncomputable def angle_CAB : Angle := angle A C B
noncomputable def angle_ACD : Angle := angle A C D

-- Proving the equivalence condition for PQ = QR iff AB * DC = BC * AD
theorem angle_bisectors_concurrent_iff_eq_segments
  (ABCD : Quadrilateral) (P Q R : Point) :
  inscribed ABCD →
  perpendicular_base D BC CA AB P Q R →
  angle_bisectors_intersect_at (angle B) (angle D) (angle θ) AC →
  |PQ| = |QR| ↔ AB * DC = BC * AD :=
by sorry

end angle_bisectors_concurrent_iff_eq_segments_l744_744511


namespace martin_ratio_of_fruits_eaten_l744_744263

theorem martin_ratio_of_fruits_eaten
    (initial_fruits : ℕ)
    (current_oranges : ℕ)
    (current_oranges_twice_limes : current_oranges = 2 * (current_oranges / 2))
    (initial_fruits_count : initial_fruits = 150)
    (current_oranges_count : current_oranges = 50) :
    (initial_fruits - (current_oranges + (current_oranges / 2))) / initial_fruits = 1 / 2 := 
by
    sorry

end martin_ratio_of_fruits_eaten_l744_744263


namespace ratio_of_areas_l744_744482

theorem ratio_of_areas
  (s: ℝ) (h₁: s > 0)
  (large_square_area: ℝ)
  (inscribed_square_area: ℝ)
  (harea₁: large_square_area = s * s)
  (harea₂: inscribed_square_area = (s / 2) * (s / 2)) :
  inscribed_square_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l744_744482


namespace brocard_angle_of_A1B1C1_circles_l744_744100

-- Define the Brocard angle condition for triangles
def brocard_condition (S1 : ℝ) (varphi : ℝ) (a1 b1 c1 : ℝ) : Prop :=
  4 * S1 * cot varphi = a1^2 + b1^2 + c1^2

-- Establish the geometric conditions relating points M to the circumcircle of ABC
def points_M_circles {A B C : Point} (circumcircle : Circle) (varphi : ℝ) : Set Point :=
  { M | 
    let A1 := foot_of_perpendicular M B C,
    let B1 := foot_of_perpendicular M C A,
    let C1 := foot_of_perpendicular M A B,
    brocard_condition (area A1 B1 C1) varphi (distance B1 C1) (distance C1 A1) (distance A1 B1) }
                  
theorem brocard_angle_of_A1B1C1_circles (A B C : Point) (circumcircle : Circle) (varphi : ℝ) :
  ∃ circle1 circle2 : Circle,
    (∀ M, M ∈ points_M_circles circumcircle varphi ↔ M ∈ circle1.toSet ∨ M ∈ circle2.toSet) ∧
    circle1 ⊆ inside circumcircle ∧
    circle2 ⊆ outside circumcircle :=
sorry

end brocard_angle_of_A1B1C1_circles_l744_744100


namespace lcm_12_18_30_l744_744746

def lcm_three_numbers (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem lcm_12_18_30 : lcm_three_numbers 12 18 30 = 180 := by
  -- Conditions
  have h1 : 12 = 2^2 * 3 := rfl
  have h2 : 18 = 2 * 3^2 := rfl
  have h3 : 30 = 2 * 3 * 5 := rfl
  -- sorry is used as placeholder, proof is not required in the task.
  sorry

end lcm_12_18_30_l744_744746


namespace parabola_shifted_l744_744319

-- Definitions from conditions
def original_function (x : ℝ) : ℝ := x^2 - 1
def shift_up (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := λ x, f x + c
def shift_right (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := λ x, f (x - c)

-- Theorem statement
theorem parabola_shifted :
  (shift_right (shift_up original_function 2) 1) = (λ x, (x - 1)^2 + 1) :=
by
  sorry

end parabola_shifted_l744_744319


namespace smallest_y_l744_744750

theorem smallest_y (y : ℤ) (h : y < 3 * y - 15) : y = 8 :=
  sorry

end smallest_y_l744_744750


namespace coefficient_of_x_neg3_in_expansion_l744_744875

noncomputable def a : ℝ := ∫ x in (1/e : ℝ)..e, 1/x

theorem coefficient_of_x_neg3_in_expansion : 
  (∫ x in (1/e : ℝ)..e, 1/x = 2) →
  (a = ∫ x in (1/e : ℝ)..e, 1/x) →
  (5.choose 2 * (-2) ^ 3 = -80) :=
by
  intro ha ha'
  rw [ha] at ha'
  field_simp
  sorry

end coefficient_of_x_neg3_in_expansion_l744_744875


namespace cyclic_quadrilateral_intersection_l744_744674

noncomputable theory
open_locale big_operators

def cyclicPoints (α : Type*) [add_group α] (A B C D E F : α) : Prop :=
  ∃O R, (∀ P ∈ {A, B, C, D, E, F}, P ∈ circle O R)

def collinear (α : Type*) [add_group α] (U V W : α) : Prop :=
  ∃(ℓ : line α), U ∈ℓ ∧ V ∈ℓ ∧ W ∈ℓ

def meet (α : Type*) (ℓ₁ ℓ₂ : line α) : α :=
  classical.some (line_intersects_line ℓ₁ ℓ₂)

variables {α : Type*} [add_group α]
variables {A B C D E F Z X Y P Q R O : α}

theorem cyclic_quadrilateral_intersection
  (hcyclic : cyclicPoints α A B C D E F)
  (hAB : ¬ diameter A B) (hCD : ¬ diameter C D) (hEF : ¬ diameter E F)
  (hZ : Z = meet (line_ext A B) (line_ext C D))
  (hX : X = meet (line_ext C D) (line_ext E F))
  (hY : Y = meet (line_ext E F) (line_ext B A))
  (hP : P = meet (line_ext A C) (line_ext B F))
  (hQ : Q = meet (line_ext C E) (line_ext B D))
  (hR : R = meet (line_ext A E) (line_ext D F))
  (hO : O = meet (line_ext Y Q) (line_ext Z R)) :
  ∠ X O P = 90 :=
sorry

end cyclic_quadrilateral_intersection_l744_744674


namespace find_f_of_f_0_l744_744105

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 + 1 
  else -2 * x

theorem find_f_of_f_0 : f (f 0) = -2 := 
  sorry

end find_f_of_f_0_l744_744105


namespace max_product_OP_OQ_l744_744198

section CircleProof

-- Define parametric equations for circles C1 and C2
def C1_parametric (φ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos φ, 2 * Real.sin φ)
def C2_parametric (β : ℝ) : ℝ × ℝ := (Real.cos β, 1 + Real.sin β)

-- Polar coordinate equations derived from the parametric equations
def C1_polar (θ : ℝ) : ℝ := 4 * Real.cos θ
def C2_polar (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Function to compute |OP| * |OQ| based on the polar coordinate equations and the angle α
def product_OP_OQ (α : ℝ) : ℝ := 
  let OP := 4 * Real.cos α
  let OQ := 2 * Real.sin α
  OP * OQ

-- Theorem statement: proving the maximum value of |OP| * |OQ|
theorem max_product_OP_OQ : 
  ∃ (α : ℝ), 0 < α ∧ α < Real.pi / 2 ∧ product_OP_OQ α = 4 :=
sorry

end CircleProof

end max_product_OP_OQ_l744_744198


namespace amount_distributed_l744_744400

theorem amount_distributed (A : ℕ) (h : A / 14 = A / 18 + 80) : A = 5040 :=
sorry

end amount_distributed_l744_744400


namespace restaurant_bill_l744_744466

theorem restaurant_bill 
  (salisbury_steak : ℝ := 16.00)
  (chicken_fried_steak : ℝ := 18.00)
  (mozzarella_sticks : ℝ := 8.00)
  (caesar_salad : ℝ := 6.00)
  (bowl_chili : ℝ := 7.00)
  (chocolate_lava_cake : ℝ := 7.50)
  (cheesecake : ℝ := 6.50)
  (iced_tea : ℝ := 3.00)
  (soda : ℝ := 3.50)
  (half_off_meal : ℝ := 0.5)
  (dessert_discount : ℝ := 0.1)
  (tip_percent : ℝ := 0.2)
  (sales_tax : ℝ := 0.085) :
  let total : ℝ :=
    (salisbury_steak * half_off_meal) +
    (chicken_fried_steak * half_off_meal) +
    mozzarella_sticks +
    caesar_salad +
    bowl_chili +
    (chocolate_lava_cake * (1 - dessert_discount)) +
    (cheesecake * (1 - dessert_discount)) +
    iced_tea +
    soda
  let total_with_tax : ℝ := total * (1 + sales_tax)
  let final_total : ℝ := total_with_tax * (1 + tip_percent)
  final_total = 73.04 :=
by
  sorry

end restaurant_bill_l744_744466


namespace determine_digit_x_l744_744777

/--
Given a 2023-digit number \( N = 2000\ldots0x0\ldots00023 \), where \( x \) is the 23rd digit from the right,
prove that if \( N \) is divisible by \( 13 \), then \( x \) must equal 3.
-/
theorem determine_digit_x (x : ℤ) (h : ∃ (N : ℤ), (digits N = 2023 ∧ digit_at 23 N = x) ∧ N % 13 = 0) : x = 3 :=
sorry

end determine_digit_x_l744_744777


namespace rate_of_simple_interest_l744_744007

theorem rate_of_simple_interest (P : ℝ) (R : ℝ) (T : ℝ) (P_nonzero : P ≠ 0) : 
  (P * R * T = P / 6) → R = 1 / 42 :=
by
  intro h
  sorry

end rate_of_simple_interest_l744_744007


namespace largest_angle_is_120_degrees_l744_744520

-- Definition of angles
variables {A B C : Real} -- angles
-- Condition: ratio of sines
axiom sin_ratio : Real.sin A / Real.sin B = 1 / 1 ∧ Real.sin B / Real.sin C = 1 / sqrt 3

-- Goal: The largest interior angle is 120 degrees
theorem largest_angle_is_120_degrees (h : sin_ratio) : 
  ∃ C_angle, C_angle = 120 ∧ (C = C_angle) := 
sorry

end largest_angle_is_120_degrees_l744_744520


namespace exists_int_pow_prod_l744_744882

-- Defining necessary mathematical constructs
open Nat Int

theorem exists_int_pow_prod (n : ℕ) (h : n > 2) :
  ∃ K : ℕ, ∀ k : ℕ, k ≥ K → 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (k^n < a ∧ a < (k+1)^n) ∧ 
  (k^n < b ∧ b < (k+1)^n) ∧ 
  (k^n < c ∧ c < (k+1)^n) ∧ 
  (∏ i in ([a, b, c] : List ℕ).toFinset, i) = d ^ n :=
by sorry

end exists_int_pow_prod_l744_744882


namespace find_constant_m_l744_744134

noncomputable def equation_has_one_positive_and_one_negative_real_root (x n m : ℝ) : Prop :=
  (sqrt (m - x^2) = log2 (x + n)) ∧ 
  ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ x1 + x2 = 0
  
theorem find_constant_m (n : ℝ) (h : 3 ≤ n ∧ n < 4) : 
  ∃ m : ℝ, (∀ x : ℝ, equation_has_one_positive_and_one_negative_real_root x n m) ∧ m = 9 :=
begin
  sorry
end

end find_constant_m_l744_744134


namespace largest_number_of_acute_angles_in_convex_octagon_l744_744369

-- Definitions for convex octagon, acute angle, interior angle sum
def acute_angle (θ : ℝ) : Prop := θ < 90

def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

def is_convex_octagon (angles : Fin 8 → ℝ) : Prop :=
  (∀ i, 0 < angles i ∧ angles i < 180) ∧ (sum angles = sum_of_interior_angles 8)

-- The statement to prove
theorem largest_number_of_acute_angles_in_convex_octagon :
  ∀ angles : Fin 8 → ℝ, is_convex_octagon angles → 
    (∀ i, acute_angle (angles i) → ∃ k ≤ 4, ∀ j < k, acute_angle (angles j)) := 
by
  intros
  sorry

end largest_number_of_acute_angles_in_convex_octagon_l744_744369


namespace vector_addition_sin_identity_l744_744561

noncomputable def a (θ : ℝ) : ℝ × ℝ := (1, Real.sin θ)
def b : ℝ × ℝ := (3, 1)

-- Proof that when θ = π/6, 2a + b = (5, 2)
theorem vector_addition (θ : ℝ) (hθ : θ = Real.pi / 6) :
  (2 * a θ + b) = (5, 2) :=
by
  sorry

-- Proof that when a is parallel to b and θ ∈ (0, π/2), sin(2θ + π/4) = (8 + 7√2) / 18
theorem sin_identity (θ : ℝ) (h_parallel : (1, Real.sin θ) = α * (3, 1) for some α : ℝ) (hθ : 0 < θ ∧ θ < Real.pi / 2) :
  Real.sin (2 * θ + Real.pi / 4) = (8 + 7 * Real.sqrt 2) / 18 :=
by
  sorry

end vector_addition_sin_identity_l744_744561


namespace imaginary_part_conjugate_eq_neg_one_l744_744535

noncomputable def z : ℂ := (1 + complex.I) * complex.I

theorem imaginary_part_conjugate_eq_neg_one :
  complex.im (conj z) = -1 := 
sorry

end imaginary_part_conjugate_eq_neg_one_l744_744535


namespace sqrt_form_sum_l744_744016

theorem sqrt_form_sum (a b c : ℤ) (h1 : c = 3)
    (h2 : 88 + 42 * Real.sqrt 3 = (a + b * Real.sqrt c)^2)
    (h3 : ∀ k : ℕ, nat_prime k → k ∣ c → k = 3): a + b + c = 13 :=
by
  sorry

end sqrt_form_sum_l744_744016


namespace books_bought_on_third_day_l744_744869

theorem books_bought_on_third_day : 
  ∃ x : ℕ, (1 + 2 + x + (4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 15 + 16 + 17 + 18 + 19) = 190 ∧ x = 3) :=
by
  have arithmetic_series_sum : ∑ i in range (16) + 4 ∣ i = (16 * (2 * 4 + (16 - 1) * 1)) / 2 := by sorry
  have total_books_sum : 1 + 2 + 3 + arithmetic_series_sum = 190 := by sorry
  refine ⟨3, ⟨total_books_sum, rfl⟩⟩

end books_bought_on_third_day_l744_744869


namespace greatest_singleton_in_partition_l744_744209

theorem greatest_singleton_in_partition (n : ℕ) (h : n > 0) :
  ∃ (A : fin n → set ℕ) (j : fin n), 
  (∀ i : fin n, ∀ r s ∈ A i, r ≠ s → r + s ∈ A i) ∧
  (A j).card = 1 ∧
  ∀ k ∈ A j, k = n - 1 :=
by
  sorry

end greatest_singleton_in_partition_l744_744209


namespace binomial_sum_eq_l744_744997

theorem binomial_sum_eq (m n : ℕ) (h : m ≤ n) : 
  (∑ k in Finset.range (m + 1), Nat.choose m k * Nat.choose n k) = Nat.choose (m + n) m := 
by sorry

end binomial_sum_eq_l744_744997


namespace chord_vs_arc_length_l744_744106

theorem chord_vs_arc_length {R : ℝ} (hR : 0 < R) :
  1.05 * R > (real.pi * R) / 3 := by
sorry

end chord_vs_arc_length_l744_744106


namespace fraction_flower_beds_l744_744800

theorem fraction_flower_beds (length1 length2 height triangle_area yard_area : ℝ) (h1 : length1 = 18) (h2 : length2 = 30) (h3 : height = 10) (h4 : triangle_area = 2 * (1 / 2 * (6 ^ 2))) (h5 : yard_area = ((length1 + length2) / 2) * height) : 
  (triangle_area / yard_area) = 3 / 20 :=
by 
  sorry

end fraction_flower_beds_l744_744800


namespace bc_eq_one_area_of_triangle_l744_744246

variable (a b c A B : ℝ)

-- Conditions
def condition_1 : Prop := (b^2 + c^2 - a^2) / (Real.cos A) = 2
def condition_2 : Prop := (a * (Real.cos B) - b * (Real.cos A)) / (a * (Real.cos B) + b * (Real.cos A)) - b / c = 1

-- Equivalent proof problems
theorem bc_eq_one (h1 : condition_1 a b c A) : b * c = 1 := 
by 
  sorry

theorem area_of_triangle (h2 : condition_2 a b c A B) : (1/2) * b * c * Real.sin A = (Real.sqrt 3) / 4 := 
by 
  sorry

end bc_eq_one_area_of_triangle_l744_744246


namespace sqrt_neg_nine_is_pm_3i_l744_744334

theorem sqrt_neg_nine_is_pm_3i {z : ℂ} (h : z^2 = -9) : z = 3 * complex.I ∨ z = -3 * complex.I := sorry

end sqrt_neg_nine_is_pm_3i_l744_744334


namespace remaining_pair_parallel_l744_744811

theorem remaining_pair_parallel (n : ℕ) (h1 : 1 < n) (h2 : ∀ (k : ℕ), k < n - 1 → parallel (opposite_side k) (opposite_side_of_polygon (k+1))) : 
  n % 2 = 1 := 
sorry

end remaining_pair_parallel_l744_744811


namespace problem_solution_l744_744871

-- Define the angle for 30 degrees in radians
def theta := Real.pi / 6

-- Length of the broken line B_0 B_1 ... B_6
def broken_line_length := (1/2) * (1 + (Real.sqrt 3) / 2 + ((Real.sqrt 3) / 2)^2 + ((Real.sqrt 3) / 2)^3 + ((Real.sqrt 3) / 2)^4 + ((Real.sqrt 3) / 2)^5)

-- Area of the polygon B_0 B_1 ... B_6
def polygon_area := (1/4) * (Real.sqrt 3) * (1 + ((Real.sqrt 3) / 2)^2 + ((Real.sqrt 3) / 2)^4 + ((Real.sqrt 3) / 2)^6 + ((Real.sqrt 3) / 2)^8 + ((Real.sqrt 3) / 2)^10)

-- Theorem to prove
theorem problem_solution :
  broken_line_length = 37 * (2 + Real.sqrt 3) / 64 ∧
  polygon_area = 3367 * (Real.sqrt 3) / 8192 :=
by 
  sorry

end problem_solution_l744_744871


namespace remove_terms_for_desired_sum_l744_744830

theorem remove_terms_for_desired_sum :
  let series_sum := (1/3) + (1/5) + (1/7) + (1/9) + (1/11) + (1/13)
  series_sum - (1/11 + 1/13) = 11/20 :=
by
  sorry

end remove_terms_for_desired_sum_l744_744830


namespace sequence_property_l744_744614

variable {a : ℕ → ℝ} {a_1 d : ℝ}

axiom arithmetic_sequence (n : ℕ) : a (n + 1) = a_n + n * d

theorem sequence_property (h : (a 3 + a 5 + a 7 + a 9 + a 11) = 100) :
  3 * a 9 - a 13 = 2 * (a 1 + 6 * d) :=
sorry

end sequence_property_l744_744614


namespace determinant_identity_l744_744640

variables {R : Type*} [Comm_ring R]

-- Define the polynomial and roots conditions
variables (a b c p q : R)

-- The polynomial equation
def poly_eq : Prop := a + b + c = 0 ∧ ab + ac + bc = p ∧ abc = -q

-- The determinant expression
def matrix_det : R := 
  matrix.det ![![1 + a, 1, 1], ![1, 1 + b, 1], ![1, 1, 1 + c]]

-- The theorem we need to prove
theorem determinant_identity (h : poly_eq a b c p q) : matrix_det a b c = p - q :=
sorry

end determinant_identity_l744_744640


namespace triangle_angle_condition_for_circumcircle_l744_744732

-- Define the base structure and necessary classes
open Classical

noncomputable def circumcircle_of_squares_equals_circumcircle_of_triangle (A B C : ℝ) : Prop :=
let angle_A := 60
let angle_B := 60
let angle_C := 60 in
(∀ A B C : ℝ, angle_A + angle_B + angle_C = 180) ∧ 
(∃ (tri : Type) [Inhabited tri],
  ∃ (squares : tri → tri) (on_single_circle : Prop),
    (on_single_circle ↔ (angle_A = 60 ∧ angle_B = 60 ∧ angle_C = 60) ∨ 
                         ((angle_A = 45 ∧ angle_B = 45 ∧ angle_C = 90) ∨ 
                          (angle_A = 45 ∧ angle_B = 90 ∧ angle_C = 45))))

-- The theorem states the equivalence condition
theorem triangle_angle_condition_for_circumcircle :
  ∀ {A B C : ℝ}, circumcircle_of_squares_equals_circumcircle_of_triangle A B C :=
by
  intros
  sorry

end triangle_angle_condition_for_circumcircle_l744_744732


namespace min_value_of_frac_l744_744654

theorem min_value_of_frac (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (collinear : ∃ λ : ℝ, (a - 1, 1) = λ * (-b - 1, 2)) :
  (∃ x : ℝ, x = 1 / a + 2 / b ∧ x = 8) := sorry

end min_value_of_frac_l744_744654


namespace ratio_of_jars_to_pots_l744_744420

theorem ratio_of_jars_to_pots 
  (jars : ℕ)
  (pots : ℕ)
  (k : ℕ)
  (marbles_total : ℕ)
  (h1 : jars = 16)
  (h2 : jars = k * pots)
  (h3 : ∀ j, j = 5)
  (h4 : ∀ p, p = 15)
  (h5 : marbles_total = 200) :
  (jars / pots = 2) :=
by
  sorry

end ratio_of_jars_to_pots_l744_744420


namespace sound_pressure_proof_l744_744272

noncomputable theory

def sound_pressure_level (p p0 : ℝ) : ℝ :=
  20 * real.log10 (p / p0)

variables (p0 : ℝ) (p0_pos : 0 < p0)
variables (p1 p2 p3 : ℝ)

def gasoline_car (Lp : ℝ) : Prop :=
  60 <= Lp ∧ Lp <= 90

def hybrid_car (Lp : ℝ) : Prop :=
  50 <= Lp ∧ Lp <= 60

def electric_car (Lp : ℝ) : Prop :=
  Lp = 40

theorem sound_pressure_proof :
  gasoline_car (sound_pressure_level p1 p0) ∧
  hybrid_car (sound_pressure_level p2 p0) ∧
  electric_car (sound_pressure_level p3 p0) →
  (p1 ≥ p2) ∧ (¬ (p2 > 10 * p3)) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) :=
begin
  sorry
end

end sound_pressure_proof_l744_744272


namespace exist_epsilon_bounded_sum_l744_744872

open Real

variables (n : ℕ)
variables (v : Fin n → EuclideanSpace ℝ (Fin n))
variables (ε : Fin n → ℤ)

noncomputable def vector_length_one (i : Fin n) : Prop := ∥v i∥ = 1

theorem exist_epsilon_bounded_sum :
  (∀ i, vector_length_one n v i) →
  ∃ ε : Fin n → ℤ, (∀ i, ε i = 1 ∨ ε i = -1) ∧
  ∥∑ i, (ε i) • (v i)∥ ≤ sqrt n :=
begin
  sorry
end

end exist_epsilon_bounded_sum_l744_744872


namespace trajectory_of_midpoint_l744_744666

-- Define the conditions
variables {A : ℝ × ℝ} (B C : ℝ × ℝ)
def fixed_point_A : ℝ × ℝ := (0, 2)
def on_circle (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 16
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
def perpendicular (p1 p2 : ℝ × ℝ) : Prop := (p2.1 - p1.1) * (p2.2 - p1.2) = -1

-- State to prove the equation of the trajectory of M
theorem trajectory_of_midpoint
  (hA : A = fixed_point_A)
  (hB : on_circle B)
  (hC : on_circle C)
  (h_perpendicular : perpendicular B A ∧ perpendicular C A) :
  ∃ (x y : ℝ), (midpoint B C) = (x, y) ∧ (x^2 + (y - 1)^2 = 7) :=
sorry

end trajectory_of_midpoint_l744_744666


namespace problem_1_problem_2_l744_744141

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

theorem problem_1 (a : ℝ) (h1 : 0 < a) :
  (f a).derivative 3 = (f a).derivative (3 / 2) → a = 3.5 := 
by sorry

theorem problem_2 (a : ℝ) (h1 : 0 < a) :
  (∃ x : ℝ, f a x = 0) → a ≤ 1 := 
by sorry

end problem_1_problem_2_l744_744141


namespace range_of_k_l744_744908

noncomputable def f (k : ℝ) (x : ℝ) := (Real.exp x) / (x^2) + 2 * k * Real.log x - k * x

theorem range_of_k (k : ℝ) (h₁ : ∀ x > 0, (deriv (f k) x = 0) → x = 2) : k < Real.exp 2 / 4 :=
by
  sorry

end range_of_k_l744_744908


namespace right_triangle_inequality_l744_744667

-- Definition of a right-angled triangle with given legs a, b, hypotenuse c, and altitude h_c to the hypotenuse
variables {a b c h_c : ℝ}

-- Right-angled triangle condition definition with angle at C is right
def right_angled_triangle (a b c : ℝ) : Prop :=
  ∃ (a b c : ℝ), c^2 = a^2 + b^2

-- Definition of the altitude to the hypotenuse
def altitude_to_hypotenuse (a b c h_c : ℝ) : Prop :=
  h_c = (a * b) / c

-- Theorem statement to prove the inequality for any right-angled triangle
theorem right_triangle_inequality (a b c h_c : ℝ) (h1 : right_angled_triangle a b c) (h2 : altitude_to_hypotenuse a b c h_c) : 
  a + b < c + h_c :=
by
  sorry

end right_triangle_inequality_l744_744667


namespace grid_sum_condition_l744_744188

theorem grid_sum_condition {n : ℕ} (a b : Fin n → ℝ) (h : ∀ i, 0 < a i ∧ 0 < b i ∧ a i + b i = 1) :
  ∃ (S : Fin n → Bool), (∑ i in Finset.univ.filter (λ i, S i), a i) ≤ (n + 1) / 4 ∧ 
                         (∑ i in Finset.univ.filter (λ i, ¬ S i), b i) ≤ (n + 1) / 4 :=
by
  sorry

end grid_sum_condition_l744_744188


namespace chord_length_l744_744388

theorem chord_length (d : ℝ) (M : ℝ) : 
  (∃ (A B : ℝ), A < M ∧ M < B ∧ A + B = d) ∧
  (let remaining_area := 16 * (Real.pi ^ 3),
       total_area := (Real.pi * d ^ 2) / 8,
       cutout_area := 2 * (Real.pi * (d / 4) ^ 2 / 2),
       remaining_area_result := total_area - cutout_area
   in remaining_area = remaining_area_result) 
  → (2 * (d / 2 / sqrt(2)) = d * sqrt(2)) := 
sorry

end chord_length_l744_744388


namespace distance_CP_l744_744551

noncomputable theory

def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

def circle_center (r : ℝ) : ℝ × ℝ :=
  (r / 2, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_CP :
  let C := circle_center 4,
      P := polar_to_cartesian 4 (Real.pi / 3) in
  distance C P = 2 * Real.sqrt 3 :=
by {
  sorry
}

end distance_CP_l744_744551


namespace fixed_point_coordinates_l744_744180

theorem fixed_point_coordinates (a : ℝ) (P : ℝ × ℝ) (H : ∀ x : ℝ, f x = a^(x - 2) + 3) :
  P = (2, 4) :=
sorry

end fixed_point_coordinates_l744_744180


namespace estimated_opposed_l744_744029

theorem estimated_opposed (total_population : ℕ) (sample_size : ℕ) 
  (approved : ℕ) (opposed_in_sample : ℕ) : 
  opposed_in_sample = sample_size - approved → 
  (opposed_in_area : ℕ) = total_population * opposed_in_sample / sample_size → 
  opposed_in_area = 6912 :=
by
  intros h1 h2
  have total_population_def : total_population = 9600 := rfl
  have sample_size_def : sample_size = 50 := rfl
  have approved_def : approved = 14 := rfl
  have opposed_in_sample_def : opposed_in_sample = 36 := by rw [sample_size_def, approved_def]; norm_num
  have opposed_in_area_def : opposed_in_area = 6912 := by 
    rw [total_population_def, opposed_in_sample_def, sample_size_def]
    norm_num
  exact opposed_in_area_def

end estimated_opposed_l744_744029


namespace complex_number_on_line_l744_744938

theorem complex_number_on_line (a : ℝ) (h : (3 : ℝ) = (a - 1) + 2) : a = 2 :=
by
  sorry

end complex_number_on_line_l744_744938


namespace magnitude_of_z_l744_744939

noncomputable def z (a : ℝ) : ℂ :=
  (1 + a * complex.i) / complex.i

theorem magnitude_of_z (a : ℝ) (h : (z a).re = (z a).im): |z a| = real.sqrt 2 :=
by
  sorry

end magnitude_of_z_l744_744939


namespace maximum_profit_within_90_days_team_advances_successfully_l744_744202

def daily_sales_volume (x : ℕ) : ℤ :=
  if x = 1 then 198 else
  if x = 3 then 194 else 
  if x = 6 then 188 else 
  if x = 10 then 180 else 0

def selling_price (x : ℕ) : ℤ :=
  if 1 ≤ x ∧ x < 50 then x + 60 else 
  if x ≥ 50 then 100 else 0

def cost_per_item : ℤ := 40

def daily_profit (x : ℕ) : ℤ :=
  if 1 ≤ x ∧ x < 50 then -2 * (x ^ 2) + 160 * x + 4000 else
  if x ≥ 50 then -120 * x + 12000 else 0

theorem maximum_profit_within_90_days : 
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 90 ∧ daily_profit x = 7200 ∧ (∀ y : ℕ, 1 ≤ y ∧ y ≤ 90 → daily_profit y ≤ daily_profit x) :=
sorry

theorem team_advances_successfully :
  (∃ days_set : set ℕ, (|days_set| = 25 ∧ ∀ d ∈ days_set, daily_profit d ≥ 5400)) :=
sorry

end maximum_profit_within_90_days_team_advances_successfully_l744_744202


namespace sqrt_inequalities_l744_744218

theorem sqrt_inequalities
  (a b c d e : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 1)
  (hb : 0 ≤ b ∧ b ≤ 1)
  (hc : 0 ≤ c ∧ c ≤ 1)
  (hd : 0 ≤ d ∧ d ≤ 1)
  (he : 0 ≤ e ∧ e ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ∧
  Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ≤ 5 * Real.sqrt 2 :=
by {
  sorry
}

end sqrt_inequalities_l744_744218


namespace minimum_checks_l744_744663

-- Define the conditions
def sticks_sorted (sticks : List ℝ) : Prop :=
  sticks.sorted (· ≤ ·)

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c

-- Using the conditions to formulate the proof problem
theorem minimum_checks (sticks : List ℝ) (h_sorted : sticks_sorted sticks) :
  ∃ (a b c : ℝ), 
    sticks = a :: b :: c :: List.nil ∧ sticks.length = 3 →
    (triangle_inequality a b c ∨ ¬ triangle_inequality a b c) := 
sorry

end minimum_checks_l744_744663


namespace largest_angle_in_triangle_is_120_degrees_l744_744967

theorem largest_angle_in_triangle_is_120_degrees
  {A B C : ℝ}
  (h1 : A + B + C = 180) -- angles of a triangle
  (h2 : ∀ a b c : ℝ, sin(A) + sin(B) = a * 4 /\ sin(B) + sin(C) = b * 5 /\ sin(C) + sin(A) = c * 6) :
  max A (max B C) = 120 :=
by
  sorry

end largest_angle_in_triangle_is_120_degrees_l744_744967


namespace max_min_diff_iterative_avg_l744_744817

-- Definition of the iterative averaging function.
def iterative_avg : List ℚ → ℚ
  | [] => 0
  | [x] => x
  | (x :: y :: xs) => iterative_avg ((x + y) / 2 :: xs)

-- The initial list of numbers.
noncomputable def initial_nums : List ℚ := [1, 3, 5, 7, 9, 11]

-- Define the statement asserting the difference between the maximum and minimum possible iterative averages.
theorem max_min_diff_iterative_avg :
  ∃ (max_avg min_avg : ℚ), 
    (iterative_avg [1, 3, 5, 7, 9, 11] = max_avg ∧
    iterative_avg [11, 9, 7, 5, 3, 1] = min_avg ∧
    max_avg - min_avg = 6.125) :=
sorry

end max_min_diff_iterative_avg_l744_744817


namespace gasoline_car_p_ge_hybrid_electric_car_p_gasoline_car_p_le_100_hybrid_car_p_l744_744278

noncomputable def sound_pressure_level (p p0 : ℝ) : ℝ :=
20 * real.log10 (p / p0)

variables {p0 p1 p2 p3 : ℝ} (h_p0 : p0 > 0)
(h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
(h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)
(h_p3 : p3 = 100 * p0)

theorem gasoline_car_p_ge_hybrid (h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
                                (h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)  : p1 ≥ p2 :=
sorry

theorem electric_car_p (h_p3 : p3 = 100 * p0) : p3 = 100 * p0 :=
sorry

theorem gasoline_car_p_le_100_hybrid_car_p (h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
                                           (h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0) : p1 ≤ 100 * p2 :=
sorry

#check gasoline_car_p_ge_hybrid
#check electric_car_p
#check gasoline_car_p_le_100_hybrid_car_p

end gasoline_car_p_ge_hybrid_electric_car_p_gasoline_car_p_le_100_hybrid_car_p_l744_744278


namespace ball_bouncing_height_l744_744414

theorem ball_bouncing_height : ∃ (b : ℕ), 400 * (3/4 : ℝ)^b < 50 ∧ ∀ n < b, 400 * (3/4 : ℝ)^n ≥ 50 :=
by
  use 8
  sorry

end ball_bouncing_height_l744_744414


namespace gcd_le_sqrt_l744_744302

theorem gcd_le_sqrt (a b : ℕ) : nat.gcd a b ≤ nat.sqrt (a + b) :=
sorry

end gcd_le_sqrt_l744_744302


namespace find_slower_speed_l744_744175

-- Variables and conditions definitions
variable (v : ℝ)

def slower_speed (v : ℝ) : Prop :=
  (20 / v = 2) ∧ (v = 10)

-- The statement to be proven
theorem find_slower_speed : slower_speed 10 :=
by
  sorry

end find_slower_speed_l744_744175


namespace parallelogram_area_l744_744531

variables {V : Type*} [inner_product_space ℝ V]

-- Condition given in problem
axiom norm_cross_product : ∀ (a b : V), ∥a ×ᵥ b∥ = 12

-- Prove that the area of the new parallelogram is 180
theorem parallelogram_area (a b : V) : ∥(3 • a + 4 • b) ×ᵥ (2 • a - 3 • b)∥ = 180 :=
by
  sorry

end parallelogram_area_l744_744531


namespace cube_volume_and_side_length_l744_744378

theorem cube_volume_and_side_length (V1 V2 : Real) (s1 s2 : Real) :
  (V1 = 8) →
  (6 * s2^2 = 3 * 6 * s1^2) →
  (s1^3 = 8) →
  (s2 = 2 * sqrt 3) ∧ (V2 = 24 * sqrt 3) :=
by
  intros h1 h2 h3
  sorry

end cube_volume_and_side_length_l744_744378


namespace different_hundreds_digit_count_l744_744567

theorem different_hundreds_digit_count :
  (finset.filter (λ n : ℕ, 10 ≤ n ∧ n ≤ 500 ∧ ((17 * n) % 100 >= 83) ∧ ((17 * n % 100) + 17) % 100 < 83) 
     (finset.range 501)).card = 84 :=
by sorry

end different_hundreds_digit_count_l744_744567


namespace James_total_tabs_l744_744973

theorem James_total_tabs (browsers windows tabs additional_tabs : ℕ) 
  (h_browsers : browsers = 4)
  (h_windows : windows = 5)
  (h_tabs : tabs = 12)
  (h_additional_tabs : additional_tabs = 3) : 
  browsers * (windows * (tabs + additional_tabs)) = 300 := by
  -- Proof goes here
  sorry

end James_total_tabs_l744_744973


namespace question_f_m_minus_16_l744_744518

def f (m : ℝ) (x : ℝ) : ℝ := if x >= 0 then log (x + m) / log 2 else -f m (-x)

theorem question_f_m_minus_16 (m : ℝ) (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →
  (∀ x, 0 ≤ x → f x = log (x + m) / log 2) →
  f 0 = 0 →
  f (m - 16) = -4 :=
by
  sorry

end question_f_m_minus_16_l744_744518


namespace largest_k_l744_744490

noncomputable def mean_abs (n : ℕ) (xs : Fin n → ℝ) : ℝ :=
  (∑ i, |xs i|) / n

theorem largest_k (xs : Fin 101 → ℝ) (h_sum : ∑ i, xs i = 0) :
    ∃ k, k = 101 ∧ (∑ i, (xs i) ^ 2) ≥ k * (mean_abs 101 xs) ^ 2 := 
by
  use 101
  sorry

end largest_k_l744_744490


namespace q_can_be_true_or_false_l744_744596

-- Define the propositions p and q
variables (p q : Prop)

-- The assumptions given in the problem
axiom h1 : ¬ (p ∧ q)
axiom h2 : ¬ p

-- The statement we want to prove
theorem q_can_be_true_or_false : ∀ q, q ∨ ¬ q :=
by
  intro q
  exact em q -- Use the principle of excluded middle

end q_can_be_true_or_false_l744_744596


namespace construct_plane_l744_744076

open Real

noncomputable theory

variables {V : Type*} [inner_product_space ℝ V]

def plane_condition (d1 d2 c : V) : Prop :=
  let n := d1 × c in
  abs (1 - ((n ⬝ d2) / (∥n∥ * ∥d2∥)).abs) < 0.001

-- The mathematical problem as a Lean 4 statement:
theorem construct_plane (L1 L2 : V) (d1 d2 : V)
  (hL1 : direction_vector L1 = d1)
  (hL2 : direction_vector L2 = d2) :
  ∃ c : V, plane_condition d1 d2 c :=
sorry

end construct_plane_l744_744076


namespace modulus_of_1_ai_l744_744260

theorem modulus_of_1_ai (a : ℝ) (h : (1 - a) - (a + 1) * complex.i = 0) : 
  complex.abs (1 - a * complex.i) = real.sqrt 2 := by
sorry

end modulus_of_1_ai_l744_744260


namespace distance_midpoint_AA1_to_plane_P_l744_744963

-- Given cube defined with vertices and side length of 2
-- Midpoints of specific edges
structure Point3D (α : Type _) := (x y z : α)

def A : Point3D ℝ := { x := 0, y := 0, z := 0 }
def B : Point3D ℝ := { x := 2, y := 0, z := 0 }
def C : Point3D ℝ := { x := 2, y := 2, z := 0 }
def D : Point3D ℝ := { x := 0, y := 2, z := 0 }
def A₁ : Point3D ℝ := { x := 0, y := 0, z := 2 }
def B₁ : Point3D ℝ := { x := 2, y := 0, z := 2 }
def C₁ : Point3D ℝ := { x := 2, y := 2, z := 2 }
def D₁ : Point3D ℝ := { x := 0, y := 2, z := 2 }

-- Midpoints
def M : Point3D ℝ := { x := 1, y := 1, z := 0 } -- Midpoint of A₁D₁
def N : Point3D ℝ := { x := 0, y := 1, z := 0 } -- Midpoint of C₁D₁
def K : Point3D ℝ := { x := 0, y := 0, z := 1 } -- Midpoint of AA₁

-- The plane P defined by points D, M, N
-- Equation of plane P: z = 0
-- Finding distance from point K to plane P

def distance_from_point_to_plane (p : Point3D ℝ) (n : Point3D ℝ) (d : ℝ) :=
  (abs (n.x * p.x + n.y * p.y + n.z * p.z + d)) / (Real.sqrt (n.x ^ 2 + n.y ^ 2 + n.z ^ 2))

def distance_K_to_P : ℝ :=
  distance_from_point_to_plane K { x := 0, y := 0, z := 1 } 0

theorem distance_midpoint_AA1_to_plane_P : distance_K_to_P = 2 := 
  sorry

end distance_midpoint_AA1_to_plane_P_l744_744963


namespace deductive_reasoning_is_option_C_l744_744003

theorem deductive_reasoning_is_option_C :
  (Option C: The area S of a circle with radius r is S = π r^2, which implies the area S of a unit circle is S = π) = "deductive_reasoning" :=
by
  sorry

end deductive_reasoning_is_option_C_l744_744003


namespace chocolate_chip_difference_l744_744365

noncomputable def V_v : ℕ := 20 -- Viviana's vanilla chips
noncomputable def S_c : ℕ := 25 -- Susana's chocolate chips
noncomputable def S_v : ℕ := 3 * V_v / 4 -- Susana's vanilla chips

theorem chocolate_chip_difference (V_c : ℕ) (h1 : V_c + V_v + S_c + S_v = 90) :
  V_c - S_c = 5 := by sorry

end chocolate_chip_difference_l744_744365


namespace sin_cos_function_identity_l744_744078

variable {α : Type} [Field α]

def f (x : α) :=
  sorry -- actual definition of the function will be filled here

theorem sin_cos_function_identity (x : α) (f : α → α) (h : cos (17 * x) = f (cos x)) :
  sin (17 * x) = f (sin x) :=
sorry

end sin_cos_function_identity_l744_744078


namespace vectors_parallel_trig_identity_l744_744152

theorem vectors_parallel_trig_identity :
  ∀ x : ℝ, (sin x - 2 * cos x = 0) →
    2 * sin (x + π / 4) / (sin x - cos x) = 3 * sqrt 2 :=
by 
  intro x h_parallel,
  sorry

end vectors_parallel_trig_identity_l744_744152


namespace shaded_area_of_square_is_correct_l744_744411

open Real

theorem shaded_area_of_square_is_correct :
  let A := (5 : ℝ, 10 : ℝ)
  let B := (5 : ℝ,  0 : ℝ)
  -- Suppose the square corners are (0,0), (10,0), (10,10), and (0,10)
  let square := [(0,0), (10,0), (10,10), (0,10)]
  -- Suppose the shaded region's area is twice the area of one kite
  -- with diagonals spanning from A to B (10) and the midpoints (5)
  1/2 * 10 * 5 * 2 = 50 :=
sorry

end shaded_area_of_square_is_correct_l744_744411


namespace length_OP_l744_744711
noncomputable def P : Type := ℝ
def A : P := sorry
def M : P := sorry
def O : P := sorry

def PM := 10 * Real.sqrt 5
def AM := 20 * Real.sqrt 5

def perimeter (x y z : ℝ) := x + y + z
def length (x y : P) := Real.sqrt ((x - y) * (x - y))

axiom perimeter_APM : perimeter (length A P) (length P M) (length A M) = 180
axiom angle_PAM_right : ∃ x y z, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
  (angle_in_radians x y z = Real.pi / 2)
axiom center_O_on_AP : ∃ x, O = x
axiom tangents_O : ∀ x y : P, distance O x = 20 ∧ distance O y = 20
axiom AM_eq_2PM : AM = 2 * PM

theorem length_OP : length O P = 20 := sorry

end length_OP_l744_744711


namespace chandra_monsters_l744_744069

theorem chandra_monsters :
  ∃ x : ℕ, (x + 2 * x + 4 * x + 8 * x + 16 * x = 62) ∧ (x = 2) :=
by
  use 2
  split
  · sorry  -- Here you'd prove that the left-hand side equals 62 when x is 2.
  · rfl   -- This simply states that 2 = 2.

end chandra_monsters_l744_744069


namespace proper_subsets_A_count_l744_744183

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3}

-- A is the complement of {2} in U
def A : Set ℕ := U \ {2}

-- Define proper subsets of set A
def proper_subsets (s : Set ℕ) := {t : Set ℕ | t ⊂ s}

-- Prove that the number of proper subsets of set A is 7.
theorem proper_subsets_A_count : ∃ n : ℕ, (n = 7 ∧ (∀ t : Set ℕ, t ∈ proper_subsets A → t.card < A.card)) :=
by
  have h : A = {0, 1, 3} := rfl
  sorry

end proper_subsets_A_count_l744_744183


namespace smallest_a_l744_744396

theorem smallest_a (a : ℤ) : 
  (112 ∣ (a * 43 * 62 * 1311)) ∧ (33 ∣ (a * 43 * 62 * 1311)) ↔ a = 1848 := 
sorry

end smallest_a_l744_744396


namespace smallest_n_exists_l744_744503

theorem smallest_n_exists : 
  ∃ (n : ℕ), 
  (∀ (x : ℕ → ℝ), (∀ i, 1 ≤ i ∧ i ≤ n → x i ∈ Ioo (-1 : ℝ) (1 : ℝ)) ∧ 
                 (∑ i in finset.range n, x i = 0) ∧ 
                 (∑ i in finset.range n, (x i) ^ 2 = 40)) ∧ 
  n = 42 :=
by
  sorry

end smallest_n_exists_l744_744503


namespace negation_example_l744_744327

theorem negation_example : ¬ (∃ x : ℤ, x^2 + 2 * x + 1 ≤ 0) ↔ ∀ x : ℤ, x^2 + 2 * x + 1 > 0 := 
by 
  sorry

end negation_example_l744_744327


namespace max_sum_of_ten_consecutive_in_hundred_l744_744475

theorem max_sum_of_ten_consecutive_in_hundred :
  ∀ (s : Fin 100 → ℕ), (∀ i : Fin 100, 1 ≤ s i ∧ s i ≤ 100) → 
  (∃ i : Fin 91, (s i + s (i + 1) + s (i + 2) + s (i + 3) +
  s (i + 4) + s (i + 5) + s (i + 6) + s (i + 7) + s (i + 8) + s (i + 9)) ≥ 505) :=
by
  intro s hs
  sorry

end max_sum_of_ten_consecutive_in_hundred_l744_744475


namespace purely_imaginary_implies_a_eq_one_l744_744643

theorem purely_imaginary_implies_a_eq_one (a : ℝ) (h : ((1 : ℂ) + complex.I) * ((1 : ℂ) + a * complex.I)).re = 0) : a = 1 :=
sorry

end purely_imaginary_implies_a_eq_one_l744_744643


namespace abs_neg_gt_neg_intro_l744_744933

theorem abs_neg_gt_neg_intro {a : ℝ} (h : | -a | > -a) : a > 0 :=
sorry

end abs_neg_gt_neg_intro_l744_744933


namespace infinitely_many_n_with_perfect_square_average_l744_744668

theorem infinitely_many_n_with_perfect_square_average :
  ∃∞ (n : ℕ), ∃ (m : ℕ), (n + 1) * (2 * n + 1) = 6 * m ^ 2 :=
sorry

end infinitely_many_n_with_perfect_square_average_l744_744668


namespace prove_bc_prove_area_l744_744229

variable {a b c A B C : ℝ}

-- Given conditions as Lean definitions
def condition_cos_A : Prop := (b^2 + c^2 - a^2 = 2 * cos A)
def condition_bc : Prop := b * c = 1
def condition_trig_identity : Prop := (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1

-- Definitions for the questions translated to Lean
def find_bc : Prop :=
  condition_cos_A → b * c = 1

def find_area (bc_value : Prop) : Prop :=
  bc_value →
  condition_trig_identity →
  1/2 * b * c * sin A = (Real.sqrt 3) / 4

-- Theorems to be proven
theorem prove_bc : find_bc :=
by
  intro h1
  sorry

theorem prove_area : find_area condition_bc :=
by
  intro h1 h2
  sorry

end prove_bc_prove_area_l744_744229


namespace product_of_x_y_l744_744959

-- Assume the given conditions
variables (EF GH FG HE : ℝ)
variables (x y : ℝ)
variable (EFGH : Type)

-- Conditions given
axiom h1 : EF = 58
axiom h2 : GH = 3 * x + 1
axiom h3 : FG = 2 * y^2
axiom h4 : HE = 36
-- It is given that EFGH forms a parallelogram
axiom h5 : EF = GH
axiom h6 : FG = HE

-- The product of x and y is determined by the conditions
theorem product_of_x_y : x * y = 57 * Real.sqrt 2 :=
by
  sorry

end product_of_x_y_l744_744959


namespace distances_sum_geq_three_times_inradius_l744_744653

variable (A B C : Type _) [AcuteTriangle A B C]
variable (O : Circumcenter A B C)
variable (d_a d_b d_c : DistancesFromCircumcenter O)
variable (r : Inradius A B C)

-- The semiperimeter of the triangle A, B, C
def semiperimeter (a b c : ℝ) := (a + b + c) / 2

-- The total area of the triangle in multiple representations
def area (t r s : ℝ) := rs

def double_area (t : ℝ) (a b c : ℝ) (d_a d_b d_c : ℝ) := a * d_a + b * d_b + c * d_c

theorem distances_sum_geq_three_times_inradius 
  (A B C : Type _) [AcuteTriangle A B C] 
  (O : Circumcenter A B C)
  (r : Inradius A B C)
  (d_a d_b d_c : ℝ) 
  (a b c : ℝ)
  (s : ℝ) 
  (t : ℝ) :
  d_a + d_b + d_c ≥ 3 * r :=
by
  have h1 : area = r * s := sorry
  have h2 : double_area = a * d_a + b * d_b + c * d_c := sorry
  have h3 : 2 * t = 2 * r * s := sorry
  have h4 : 3 * (a * d_a + b * d_b + c * d_c) ≤ (a + b + c) * (d_a + d_b + d_c) := sorry
  have h5 : 3 * r * (a + b + c) ≤ (a + b + c) * (d_a + d_b + d_c) := sorry
  have h6 : d_a + d_b + d_c ≥ 3 * r := sorry
  exact h6

end distances_sum_geq_three_times_inradius_l744_744653


namespace value_range_of_expression_l744_744587

noncomputable def value_range (x y : ℝ) (h : x + y^2 = 4) : set ℝ :=
  {z : ℝ | z = x * y / (x + y^2)}

theorem value_range_of_expression (x y : ℝ) (h : x + y^2 = 4) :
  value_range x y h = {z : ℝ | 1 - real.sqrt 2 ≤ z ∧ z ≤ 1 + real.sqrt 2} :=
sorry

end value_range_of_expression_l744_744587


namespace integral_abs_polynomial_l744_744873

theorem integral_abs_polynomial :
  ∫ x in 0..3, abs (x^2 - 1) = 22 / 3 :=
by
  sorry

end integral_abs_polynomial_l744_744873


namespace sequence_periodicity_and_value_2023_l744_744513

theorem sequence_periodicity_and_value_2023 (a : ℕ → ℚ) :
  a 1 = 2 →
  (∀ n, a (n + 1) = (1 + a n) / (1 - a n)) →
  a 2023 = -1 / 2 :=
begin
  sorry
end

end sequence_periodicity_and_value_2023_l744_744513


namespace solution_set_l744_744642

-- Declare the conditions:

variables {f : ℝ → ℝ}
variables [f_odd : ∀ x : ℝ, f(-x) = -f(x)]
variables [f_increasing : ∀ x y : ℝ, 0 < x → x < y → f(x) < f(y)]
variables {f_at_3 : f 3 = 0}

-- Declare the theorem that needs to be proved:
theorem solution_set :
  ∀ x : ℝ, (x + 3) * (f x - f (-x)) < 0 ↔ 0 < x ∧ x < 3 :=
by
  sorry

end solution_set_l744_744642


namespace perimeter_of_triangle_ABF2_max_min_PF1_PF2_l744_744906

-- Problems as Lean statements

-- Theorem for Part (1)
theorem perimeter_of_triangle_ABF2 (n : ℝ) (F1 F2 : ℝ × ℝ) (slope : ℝ) (A B : ℝ × ℝ) :
  n = -1 ∧ slope = sqrt 3 ∧ (A.1 ^ 2 - A.2 ^ 2 = 1) ∧ (B.1 ^ 2 - B.2 ^ 2 = 1) →
  dist A B + dist A F2 + dist B F2 = 12 :=
sorry

-- Theorem for Part (2)
theorem max_min_PF1_PF2 (n : ℝ) (F1 F2 P : ℝ × ℝ) :
  n = 4 ∧ (P.1 ^ 2 + P.2 ^ 2 / 4 = 1) →
  ∃ (x : ℝ), 2 - sqrt 3 ≤ x ∧ x ≤ 2 + sqrt 3 ∧ 
  PF1 * PF2 = x * (4 - x) ∧
  ∃ max_value min_value, max_value = 4 ∧ min_value = 1 :=
sorry

end perimeter_of_triangle_ABF2_max_min_PF1_PF2_l744_744906


namespace inverse_at_8_l744_744876

def f (x : ℝ) := 10^(x - 1) - 2

theorem inverse_at_8 : f⁻¹ 8 = 2 := 
by
  sorry

end inverse_at_8_l744_744876


namespace coeffs_sum_of_binomial_expansion_l744_744505

theorem coeffs_sum_of_binomial_expansion :
  (3 * x - 2) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_0 = 64 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = -63 :=
by
  sorry

end coeffs_sum_of_binomial_expansion_l744_744505


namespace min_value_expression_l744_744253

theorem min_value_expression (w : ℂ) (h : complex.abs (w - (3 - 3 * complex.I)) = 6) :
  (complex.abs (w + 2 - complex.I))^2 + (complex.abs (w - 7 + 5 * complex.I))^2 = 120 :=
sorry

end min_value_expression_l744_744253


namespace not_polynomial_D_l744_744382

inductive AlgebraicExpression
| A : AlgebraicExpression
| B : AlgebraicExpression
| C : AlgebraicExpression
| D : AlgebraicExpression

def isPolynomial : AlgebraicExpression → Prop
| AlgebraicExpression.A := True
| AlgebraicExpression.B := True
| AlgebraicExpression.C := True
| AlgebraicExpression.D := False

theorem not_polynomial_D :
  ¬ isPolynomial AlgebraicExpression.D :=
by
  -- Insert proof here
  sorry

end not_polynomial_D_l744_744382


namespace find_bc_find_area_l744_744238

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Condition 1: From the given problem
variable (h1 : (b^2 + c^2 - a^2) / (cos A) = 2)

-- Condition 2: From the given problem
variable (h2 : (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1)

theorem find_bc : bc = 1 :=
sorry

theorem find_area (area_abc : ℝ) : area_abc = (sqrt 3) / 4 :=
sorry

end find_bc_find_area_l744_744238


namespace pipe_length_l744_744004

theorem pipe_length (L x : ℝ) 
  (h1 : 20 = L - x)
  (h2 : 140 = L + 7 * x) : 
  L = 35 := by
  sorry

end pipe_length_l744_744004


namespace four_digit_numbers_divisible_by_11_and_5_with_sum_12_l744_744566

theorem four_digit_numbers_divisible_by_11_and_5_with_sum_12:
  ∀ a b c d : ℕ, (a + b + c + d = 12) ∧ ((a + c) - (b + d)) % 11 = 0 ∧ (d = 0 ∨ d = 5) →
  false :=
by
  intro a b c d
  intro h
  sorry

end four_digit_numbers_divisible_by_11_and_5_with_sum_12_l744_744566


namespace rationalize_denominator_l744_744670

theorem rationalize_denominator :
  (1 / (Real.cbrt 2 + Real.cbrt 32 + 1) = 2 / 11) :=
by
  have h1 : Real.cbrt 32 = 2 * Real.cbrt 4 :=
    by sorry
  have h2 : Real.cbrt 4 = Real.cbrt (2^2) :=
    by sorry
  sorry

end rationalize_denominator_l744_744670


namespace sqrt_meaningful_real_domain_l744_744600

theorem sqrt_meaningful_real_domain (x : ℝ) (h : 6 - 4 * x ≥ 0) : x ≤ 3 / 2 :=
by sorry

end sqrt_meaningful_real_domain_l744_744600


namespace correct_equation_l744_744384

theorem correct_equation : \(\sqrt{6} \cdot \sqrt{2} = 2\sqrt{3}\) :=
by
  sorry

end correct_equation_l744_744384


namespace ordered_triples_unique_solution_l744_744929

theorem ordered_triples_unique_solution :
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a ∧ ca = b ∧ a + b + c = 2 :=
sorry

end ordered_triples_unique_solution_l744_744929


namespace number_of_true_propositions_l744_744304

variables {m n : Type} {α β γ : Type}

-- Establishing the conditions as hypotheses
hypothesis h1 : (∀ m α β, (m ∥ α ∧ n ∥ β ∧ α ∥ β) → m ∥ n) ↔ False
hypothesis h2 : (∀ α β γ, (α ∩ β = m ∧ α ⊥ γ ∧ β ⊥ γ) → m ⊥ γ) ↔ True
hypothesis h3 : (∀ m α β, (m ⊥ α ∧ n ⊥ β ∧ α ⊥ β) → m ⊥ n) ↔ True

-- Statement of the proof problem
theorem number_of_true_propositions : (∑ (b : bool) in [(h1 = True), (h2 = True), (h3 = True)], ite b 1 0) = 2 :=
by
  sorry

end number_of_true_propositions_l744_744304


namespace simple_interest_l744_744598

/--
Given:
- r = 0.05
- t = 2
- CI = 51.25

Prove:
The simple interest (SI) for P = 500 is 50

We begin with:
CI = P * [(1 + r)^t - 1]
51.25 = P * [(1.05)^2 - 1]
51.25 = P * [1.1025 - 1]
51.25 = P * 0.1025

Determine P:
P = 51.25 / 0.1025
P = 500

Calculate SI:
SI = P * r * t
SI = 500 * 0.05 * 2
SI = 50
-/
theorem simple_interest (r : ℝ) (t : ℕ) (CI : ℝ) (P : ℝ) (SI : ℝ) :
  r = 0.05 →
  t = 2 →
  CI = 51.25 →
  P = (51.25 / 0.1025) →
  SI = 500 * 0.05 * 2 →
  SI = 50 :=
by
  intros hr ht hCI hP hSI
  rw [hr, ht, hCI, hP, hSI]
  norm_num
  sorry

end simple_interest_l744_744598


namespace coefficient_comparison_l744_744625

theorem coefficient_comparison : 
  let P := (1 - X^2 + X^3) ^ 1000
  let Q := (1 + X^2 - X^3) ^ 1000
  coeff (X^20) Q > coeff (X^20) P :=
by sorry

end coefficient_comparison_l744_744625


namespace rain_in_august_probability_l744_744331

noncomputable def rain_probability (n : ℕ) (p : ℚ) : ℚ :=
(let q := 1 - p in
  (((q) ^ 7) + (7 * (p) * (q) ^ 6) +
   (21 * (p ^ 2) * (q ^ 5)) + (35 * (p ^ 3) * (q ^ 4))))

theorem rain_in_august_probability :
  rain_probability 7 (1/5) = 0.813 :=
by
  sorry

end rain_in_august_probability_l744_744331


namespace expression_simplified_l744_744857

noncomputable def expression : ℚ := 1 + 3 / (4 + 5 / 6)

theorem expression_simplified : expression = 47 / 29 :=
by
  sorry

end expression_simplified_l744_744857


namespace find_AB_squared_l744_744689

-- Define the conditions of the problem
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 5)^2 = 9

-- Define the intersection points A and B
def is_intersection (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- Define the distance between the points A and B
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Given points A and B are the intersection points of the first two circles
axiom A is_intersection : ∃ x y, is_intersection x y
axiom B is_intersection : ∃ x y, is_intersection x y

-- Lean statement to prove (AB)^2 = 224/9
theorem find_AB_squared :
  let A := classical.some A_is_intersection in
  let B := classical.some B_is_intersection in
  distance_squared (A.1, A.2) (B.1, B.2) = 224 / 9 :=
sorry

end find_AB_squared_l744_744689


namespace cos_double_angle_example_l744_744572

def cos_double_angle_identity (x : ℝ) : Prop :=
  cos (2 * x) = 1 - 2 * (sin x) ^ 2

theorem cos_double_angle_example : cos_double_angle_identity (x : ℝ) 
  (h : sin x = - 2 / 3) : cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_example_l744_744572


namespace find_tangent_points_l744_744338

-- Step a: Define the curve and the condition for the tangent line parallel to y = 4x.
def curve (x : ℝ) : ℝ := x^3 + x - 2
def tangent_slope : ℝ := 4

-- Step d: Provide the statement that the coordinates of P₀ are (1, 0) and (-1, -4).
theorem find_tangent_points : 
  ∃ (P₀ : ℝ × ℝ), (curve P₀.1 = P₀.2) ∧ 
                 ((P₀ = (1, 0)) ∨ (P₀ = (-1, -4))) := 
by
  sorry

end find_tangent_points_l744_744338


namespace box_inequalities_l744_744044

variables {x y z : ℝ} (hxyz: x < y ∧ y < z)

def perimeter (x y z : ℝ) : ℝ := 4 * (x + y + z)

def surface_area (x y z : ℝ) : ℝ := 2 * (x * y + y * z + z * x)

def diagonal (x y z : ℝ) : ℝ := real.sqrt (x^2 + y^2 + z^2)

theorem box_inequalities (x y z : ℝ) (hxyz: x < y ∧ y < z) :
  let p := perimeter x y z,
      s := surface_area x y z,
      d := diagonal x y z in
  3 * x < p / 4 - real.sqrt (d^2 - s / 2) ∧
  3 * z > p / 4 + real.sqrt (d^2 - s / 2) :=
by {
  let p := perimeter x y z,
  let s := surface_area x y z,
  let d := diagonal x y z,
  sorry
}

end box_inequalities_l744_744044


namespace unpainted_area_of_boards_l744_744355

theorem unpainted_area_of_boards
  (width1 width2 : ℝ)
  (angle : ℝ)
  (h_width1 : width1 = 5)
  (h_width2 : width2 = 7)
  (h_angle : angle = π / 4) :
  let height := width2 * Real.sin angle,
      area := width1 * height in
  area = 35 * Real.sqrt 2 / 2 :=
by
  sorry

end unpainted_area_of_boards_l744_744355


namespace number_of_subsets_l744_744920

theorem number_of_subsets :
  ∀ (a : ℝ), let M := {x : ℝ | x^2 - 3 * x - a^2 + 2 = 0} in
  (∀ x y : ℝ, x ∈ M ∧ y ∈ M → x = y ∨ x ≠ y) →
  ∃ n : ℕ, 2 ^ n = 4 :=
begin
  intro a,
  let M := {x : ℝ | x^2 - 3 * x - a^2 + 2 = 0},
  intros h,
  use 2,
  norm_num,
end

end number_of_subsets_l744_744920


namespace work_completion_days_l744_744012

theorem work_completion_days 
  (x_days : ℕ) (y_days : ℕ) (z_days : ℕ) 
  (y_worked_days : ℕ) 
  (x_work_rate : ℚ := 1 / x_days) 
  (y_work_rate : ℚ := 1 / y_days)
  (z_work_rate : ℚ := 1 / z_days)
  (combined_work_rate : ℚ := x_work_rate + z_work_rate)
  (remaining_work : ℚ := 1 - (y_work_rate * y_worked_days))
  (d : ℚ := remaining_work / combined_work_rate)
  (days_needed : ℕ := d.ceil) :
  x_days = 20 ∧ y_days = 15 ∧ z_days = 25 ∧ y_worked_days = 9 → 
  days_needed = 5 :=
by
  sorry

end work_completion_days_l744_744012


namespace inscribed_cube_side_length_l744_744802

theorem inscribed_cube_side_length :
  ∀ (r h : ℝ), r = 1 → h = 3 → ∃ (s : ℝ), s = 6 / (2 + 3 * Real.sqrt 2) :=
by
  assume r h hr hh
  use (6 / (2 + 3 * Real.sqrt 2))
  sorry

end inscribed_cube_side_length_l744_744802


namespace rook_placement_l744_744931

def rook_placement_count : ℕ := 18816

theorem rook_placement (r c : ℕ) (h_r : r = 3) (h_c : c = 8) :
  ∃ n, (n = (Nat.choose h_c h_r) * (h_c - 0) * (h_c - 1) * (h_c - 2)) ∧ n = rook_placement_count :=
begin
  use (Nat.choose 8 3) * 8 * 7 * 6,
  split,
  {
    rw [Nat.choose_eq_factorial_div_factorial (8 - 3)],
    have fact5_cancel := Nat.factorial (8 - 3),
    rw fact5_cancel,
    ring,
  },
  {
    refl,
  }
end

end rook_placement_l744_744931


namespace log_seq_arithmetic_l744_744597

/-- Definition of the geometric sequence {a_n} with a_1 = 2 and common ratio 4 -/
def geom_seq (n : ℕ) : ℕ := 2 * 4^(n-1)

/-- Definition of the sequence log_2 a_n, where a_n is defined by the geometric sequence -/
def log_seq (n : ℕ) : ℤ := Int.log 2 (2 * 4^(n-1))

/-- Theorem stating that the sequence {log_2 a_n} is an arithmetic sequence with a common difference of 2 -/
theorem log_seq_arithmetic : ∀ n : ℕ, log_seq (n + 1) - log_seq n = 2 :=
by
  sorry

end log_seq_arithmetic_l744_744597


namespace positive_number_square_roots_l744_744902

theorem positive_number_square_roots (x : ℕ) :
  ∃ n : ℕ, (sqrt n = x + 1 ∧ sqrt n = x - 5) → n = 9 :=
by
  sorry

end positive_number_square_roots_l744_744902


namespace bound_on_quadratic_congruences_l744_744522

theorem bound_on_quadratic_congruences (m : ℕ) (a : ℤ) (c : ℝ) 
  (hm : m > 0 ∧ m % 2 = 1) : 
  ∃ S : finset ℤ, (∀ x ∈ S, x ∈ set.Icc (⌊c⌋) (⌊c + real.sqrt m⌋) ∧ x^2 % m = a % m) → 
  S.card ≤ 2 + real.log2 m :=
by
  sorry

end bound_on_quadratic_congruences_l744_744522


namespace conjugate_quadrant_l744_744708

noncomputable def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem conjugate_quadrant (z : ℂ) (hz : determinant z (-complex.I) (1 - complex.I) (-2 * complex.I) = 0) :
    (z.conjugate.re > 0) ∧ (z.conjugate.im > 0) :=
by {
  sorry
}

end conjugate_quadrant_l744_744708


namespace intersection_point_l744_744214

noncomputable def g (x : ℝ) : ℝ := x^3 + 3 * x^2 + 9 * x + 15

theorem intersection_point :
  ∃ a : ℝ, g a = a ∧ a = -3 :=
by
  sorry

end intersection_point_l744_744214


namespace sound_pressure_proof_l744_744269

noncomputable theory

def sound_pressure_level (p p0 : ℝ) : ℝ :=
  20 * real.log10 (p / p0)

variables (p0 : ℝ) (p0_pos : 0 < p0)
variables (p1 p2 p3 : ℝ)

def gasoline_car (Lp : ℝ) : Prop :=
  60 <= Lp ∧ Lp <= 90

def hybrid_car (Lp : ℝ) : Prop :=
  50 <= Lp ∧ Lp <= 60

def electric_car (Lp : ℝ) : Prop :=
  Lp = 40

theorem sound_pressure_proof :
  gasoline_car (sound_pressure_level p1 p0) ∧
  hybrid_car (sound_pressure_level p2 p0) ∧
  electric_car (sound_pressure_level p3 p0) →
  (p1 ≥ p2) ∧ (¬ (p2 > 10 * p3)) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) :=
begin
  sorry
end

end sound_pressure_proof_l744_744269


namespace max_area_YZQ_l744_744207

-- Define the given sides of triangle XYZ
def XY : ℝ := 12
def YZ : ℝ := 18
def ZX : ℝ := 20

-- Define a point E on the line segment YZ
axiom E : exists (u : ℝ), 0 ≤ u ∧ u ≤ 1 -- assuming E is parameterized by u on YZ

-- Define points J_Y and J_Z as the incenters of triangles XYE and XZE respectively
axiom J_Y : ∀ (E : ℝ), is_incenter (triangle XY E)
axiom J_Z : ∀ (E : ℝ), is_incenter (triangle XZ E)

-- Define points of intersection Q and E
axiom Q : point

-- The problem is to prove the maximum possible area of triangle YZQ 
-- and express it in the form p - q√r and sum p + q + r = 781
theorem max_area_YZQ (p q r : ℕ) (hp : p = 303) (hq : q = 303) (hr : r = 175):
  let area_YZQ := 303.75 - 303.75 * (Real.sqrt 175) in
  area_YZQ = cast p - cast q * (Real.sqrt r) →
  p + q + r = 781 :=
by 
  sorry

end max_area_YZQ_l744_744207


namespace card_selection_ways_l744_744162

theorem card_selection_ways (deck_size : ℕ) (suits : ℕ) (cards_per_suit : ℕ) (total_cards_chosen : ℕ)
  (repeated_suit_count : ℕ) (distinct_suits_count : ℕ) (distinct_ranks_count : ℕ) 
  (correct_answer : ℕ) :
  deck_size = 52 ∧ suits = 4 ∧ cards_per_suit = 13 ∧ total_cards_chosen = 5 ∧ 
  repeated_suit_count = 2 ∧ distinct_suits_count = 3 ∧ distinct_ranks_count = 11 ∧ 
  correct_answer = 414384 :=
by 
  -- Sorry is used to skip actual proof steps, according to the instructions.
  sorry

end card_selection_ways_l744_744162


namespace sound_pressures_relationships_l744_744273

variables (p p0 p1 p2 p3 : ℝ)
  (Lp Lpg Lph Lpe : ℝ)

-- The definitions based on the conditions
def sound_pressure_level (p : ℝ) (p0 : ℝ) : ℝ := 20 * (Real.log10 (p / p0))

-- Given conditions
axiom p0_gt_zero : p0 > 0

axiom gasoline_car_levels : 60 ≤ Lpg ∧ Lpg ≤ 90
axiom hybrid_car_levels : 50 ≤ Lph ∧ Lph ≤ 60
axiom electric_car_level : Lpe = 40

axiom gasoline_car_sound_pressure : Lpg = sound_pressure_level p1 p0
axiom hybrid_car_sound_pressure : Lph = sound_pressure_level p2 p0
axiom electric_car_sound_pressure : Lpe = sound_pressure_level p3 p0

-- The proof to be derived
theorem sound_pressures_relationships (p0_gt_zero : p0 > 0)
  (gasoline_car_levels : 60 ≤ Lpg ∧ Lpg ≤ 90)
  (hybrid_car_levels : 50 ≤ Lph ∧ Lph ≤ 60)
  (electric_car_level : Lpe = 40)
  (gasoline_car_sound_pressure : Lpg = sound_pressure_level p1 p0)
  (hybrid_car_sound_pressure : Lph = sound_pressure_level p2 p0)
  (electric_car_sound_pressure : Lpe = sound_pressure_level p3 p0) :
  p1 ≥ p2 ∧ p3 = 100 * p0 ∧ p1 ≤ 100 * p2 :=
by
  sorry

end sound_pressures_relationships_l744_744273


namespace train_speed_including_stoppages_l744_744088

-- Define the conditions
def speed_excluding_stoppages : ℝ := 54
def stoppage_time_minutes : ℝ := 15.56
def minutes_to_hours (minutes : ℝ) : ℝ := minutes / 60

-- Define the question and its answer
theorem train_speed_including_stoppages :
  let stoppage_time_hours := minutes_to_hours stoppage_time_minutes
  let time_moving_in_hour := 1 - stoppage_time_hours
  let distance_covered_moving := speed_excluding_stoppages * time_moving_in_hour
  distance_covered_moving / 1 = 40 :=
by {
  sorry
}

end train_speed_including_stoppages_l744_744088


namespace max_intersection_points_l744_744371

theorem max_intersection_points (circle_triangle : ∀ sides : ℕ, sides = 3 → ∃ points : ℕ, points ≤ 6)
  (circle_rectangle : ∀ sides : ℕ, sides = 4 → ∃ points : ℕ, points ≤ 8)
  (triangle_rectangle : ∀ (triangle_sides rectangle_sides : ℕ), triangle_sides = 3 → rectangle_sides = 4 → ∃ points : ℕ, points ≤ 24) :
  ∃ total_points : ℕ, total_points = 38 :=
begin
  sorry
end

end max_intersection_points_l744_744371


namespace ellipse_equation_midpoint_trajectory_l744_744516

theorem ellipse_equation (x y : ℝ)
  (h1 : x = 2 ∨ x = -2)
  (h2 : y = 0 ∨ y = 0)
  (h3 : ∃ (F : ℝ × ℝ), F = (-√3, 0))
  (h4 : ∃ (D : ℝ × ℝ), D = (2, 0)) :
  ((x^2 / 4) + y^2 = 1) := sorry

theorem midpoint_trajectory (x y : ℝ)
  (A : ℝ × ℝ)
  (hA : A = (1, 1/2))
  (P M : ℝ × ℝ)
  (hP : ∃ x0 y0, P = (x0, y0) ∧ ((x0^2 / 4) + y0^2 = 1))
  (hM : M = ((P.1 + A.1) / 2, (P.2 + A.2) / 2)) :
  ((2 * M.1 - 1)^2 / 4 + (2 * M.2 - 1/2)^2 = 1) := sorry

end ellipse_equation_midpoint_trajectory_l744_744516


namespace area_of_triangle_BQW_l744_744203

-- Define the problem conditions and proof statement in Lean 4

open real

theorem area_of_triangle_BQW 
  (AZ WC AB : ℝ)
  (trapezoid_area : ℝ)
  (rect_area : ℝ)
  (AZ_eq : AZ = 8)
  (WC_eq : WC = 8)
  (AB_eq : AB = 16)
  (trapezoid_area_eq : trapezoid_area = 160)
  (rect_area_eq : rect_area = 16 * (16 + AZ))
  : (1 / 2) * (rect_area - trapezoid_area) = 112 :=
by
  sorry

end area_of_triangle_BQW_l744_744203


namespace sum_ratios_geq_n_l744_744770

variable (n : ℕ) (x : Fin n → ℝ) (y : Fin n → ℝ)

def isPermutation {n : ℕ} (x y : Fin n → ℝ) : Prop :=
  ∃ p : Fin n → Fin n, Function.Bijective p ∧ ∀ i, y (p i) = x i

theorem sum_ratios_geq_n (hx : ∀ i, 0 < x i) (hy : isPermutation x y) :
  ∑ i in Finset.univ, x i / y i ≥ n :=
sorry

end sum_ratios_geq_n_l744_744770


namespace Julio_total_earnings_l744_744977

theorem Julio_total_earnings :
  let commission_per_customer := 1
  let customers_first_week := 35
  let customers_second_week := 2 * customers_first_week
  let customers_third_week := 3 * customers_first_week
  let salary := 500
  let bonus := 50
  in let total_customers := customers_first_week + customers_second_week + customers_third_week
     let total_commission := total_customers * commission_per_customer
     let total_earnings := total_commission + salary + bonus
     total_earnings = 760 :=
by
  sorry

end Julio_total_earnings_l744_744977


namespace infinite_sum_eq_l744_744858

noncomputable def infinite_sum : ℝ :=
  ∑' (n : ℕ), (n : ℝ) / ((n : ℝ)^4 + 16)

theorem infinite_sum_eq : infinite_sum = 3 / 40 := 
by sorry

end infinite_sum_eq_l744_744858


namespace mod_inv_81_l744_744120

theorem mod_inv_81 (h : (9 : ℤ)⁻¹ ≡ 43 [MOD 97]) : (81 : ℤ)⁻¹ ≡ 6 [MOD 97] :=
sorry

end mod_inv_81_l744_744120


namespace modulo_problem_l744_744992

theorem modulo_problem (m : ℤ) (hm : 0 ≤ m ∧ m < 41) (hmod : 4 * m ≡ 1 [MOD 41]) :
  ((3 ^ m) ^ 4 - 3) % 41 = 37 := 
by
  sorry

end modulo_problem_l744_744992


namespace product_of_series_l744_744831

theorem product_of_series :
  (1 - 1/2^2) * (1 - 1/3^2) * (1 - 1/4^2) * (1 - 1/5^2) * (1 - 1/6^2) *
  (1 - 1/7^2) * (1 - 1/8^2) * (1 - 1/9^2) * (1 - 1/10^2) = 11 / 20 :=
by 
  sorry

end product_of_series_l744_744831


namespace second_less_than_first_third_less_than_first_l744_744730

variable (X : ℝ)

def first_number : ℝ := 0.70 * X
def second_number : ℝ := 0.63 * X
def third_number : ℝ := 0.59 * X

theorem second_less_than_first : 
  ((first_number X - second_number X) / first_number X * 100) = 10 :=
by
  sorry

theorem third_less_than_first : 
  ((third_number X - first_number X) / first_number X * 100) = -15.71 :=
by
  sorry

end second_less_than_first_third_less_than_first_l744_744730


namespace find_f7_l744_744693

theorem find_f7 (f : ℝ → ℝ) (h_add : ∀ x y, f(x + y) = f(x) + f(y)) (h_f6 : f(6) = 3) :
  f(7) = 7 / 2 :=
sorry

end find_f7_l744_744693


namespace distance_to_lightning_l744_744291

theorem distance_to_lightning (time_interval : ℕ) (speed_of_sound : ℕ) (feet_per_mile : ℕ) :
  time_interval = 8 →
  speed_of_sound = 1100 →
  feet_per_mile = 5280 →
  (round ((speed_of_sound * time_interval : ℕ) / feet_per_mile.toFloat / 0.5) * 0.5) = 1.5 :=
by
  intros h_time h_speed h_feet
  -- Convert inputs to floats for calculation
  let distance_in_feet := (speed_of_sound : ℕ) * time_interval
  let distance_in_miles := distance_in_feet.toFloat / feet_per_mile.toFloat
  have dist_calc : distance_in_miles = 8800 / 5280 := by
    -- allowing the proof engine to figure out the exact value chains
    sorry
  have rounding : (round (distance_in_miles / 0.5) * 0.5) = 1.5 := by
    -- proving the rounding
    sorry
  exact rounding

end distance_to_lightning_l744_744291


namespace problem_1_part_1_problem_1_part_2_problem_2_part_1_problem_2_part_2_l744_744093

noncomputable def partial_derivative_u (u v : ℝ) : ℝ :=
  (2 * v * u^2 * (cos v)^2 * (v * cos u + cos v)) / (1 + (u * v * sin u * cos v)^2)

noncomputable def partial_derivative_v (u v : ℝ) : ℝ :=
  (2 * u * v^2 * (sin u)^2 * (sin u - u * sin v)) / (1 + (u * v * sin u * cos v)^2)

noncomputable def partial_derivative_x (f : ℝ × ℝ → ℝ) (u v x y : ℝ) : ℝ :=
  ∂(λ u => f (u, v)) x + ∂(λ v => 2 * x * y^2) x

noncomputable def partial_derivative_y (f : ℝ × ℝ → ℝ) (u v x y : ℝ) : ℝ :=
  ∂(λ u => f (u, v)) y + ∂(λ v => 2 * x^2 * y) y

theorem problem_1_part_1 (u v : ℝ) :
  let z := arctan ((v * sin u) ^ 2 * (u * cos v) ^ 2)
  ∂ z u = partial_derivative_u u v := by
  sorry

theorem problem_1_part_2 (u v : ℝ) :
  let z := arctan ((v * sin u) ^ 2 * (u * cos v) ^ 2)
  ∂ z v = partial_derivative_v u v := by
  sorry

theorem problem_2_part_1 (f : ℝ × ℝ → ℝ) (x y : ℝ) :
  let z := f (x ^ 2 + y ^ 2, x ^ 2 * y ^ 2)
  ∂ z x = partial_derivative_x f (x ^ 2 + y ^ 2) (x ^ 2 * y ^ 2) x y := by
  sorry

theorem problem_2_part_2 (f : ℝ × ℝ → ℝ) (x y : ℝ) :
  let z := f (x ^ 2 + y ^ 2, x ^ 2 * y ^ 2)
  ∂ z y = partial_derivative_y f (x ^ 2 + y ^ 2) (x ^ 2 * y ^ 2) x y := by
  sorry

end problem_1_part_1_problem_1_part_2_problem_2_part_1_problem_2_part_2_l744_744093


namespace zoey_finished_on_wednesday_l744_744005

theorem zoey_finished_on_wednesday :
  let total_days := ∑ n in finset.range 18, (2 * n + 1)
  let remainder := total_days % 7
  total_days = 324 → remainder = 2 → remainder = 2 := by
  sorry

end zoey_finished_on_wednesday_l744_744005


namespace bc_eq_one_area_of_triangle_l744_744244

variable (a b c A B : ℝ)

-- Conditions
def condition_1 : Prop := (b^2 + c^2 - a^2) / (Real.cos A) = 2
def condition_2 : Prop := (a * (Real.cos B) - b * (Real.cos A)) / (a * (Real.cos B) + b * (Real.cos A)) - b / c = 1

-- Equivalent proof problems
theorem bc_eq_one (h1 : condition_1 a b c A) : b * c = 1 := 
by 
  sorry

theorem area_of_triangle (h2 : condition_2 a b c A B) : (1/2) * b * c * Real.sin A = (Real.sqrt 3) / 4 := 
by 
  sorry

end bc_eq_one_area_of_triangle_l744_744244


namespace tan_alpha_plus_pi_over_4_l744_744893

theorem tan_alpha_plus_pi_over_4 
  {α β : ℝ} 
  (h1 : Real.tan (α + β) = 2/5) 
  (h2 : Real.tan β = 1/3) : 
  Real.tan (α + π/4) = 9/8 := 
by 
  sorry

end tan_alpha_plus_pi_over_4_l744_744893


namespace midpoint_meeting_coord_l744_744684

open_locale real

def susan_start : ℝ × ℝ × ℝ := (1.5, -3.5, 2)
def bob_start : ℝ × ℝ × ℝ := (-2, 4.5, -3)

theorem midpoint_meeting_coord {x1 y1 z1 x2 y2 z2 : ℝ} :
  (x1, y1, z1) = susan_start →
  (x2, y2, z2) = bob_start →
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2) = (-0.25, 0.5, -0.5) :=
by
  intros h1 h2
  rw [h1, h2]
  -- Calculation steps can be detailed here, but skipped for brevity.
  sorry

end midpoint_meeting_coord_l744_744684


namespace quadrilateral_area_l744_744798

-- Define the conditions
variables {A B C D : Type*}
variables {α c : ℝ}
variables [InCircle A B C D] -- A condition specifying that A, B, C, D are inscribed in a circle
variables {h1 : SegmentLen B C = SegmentLen C D} -- BC = CD
variables {h2 : SegmentLen A C = c} -- AC = c
variables {h3 : Angle A B D = 2 * α} -- ∠BAD = 2α

-- Define the goal
theorem quadrilateral_area :
  quadrilateral_area A B C D = c^2 * sin(α) * cos(α) :=
sorry

end quadrilateral_area_l744_744798


namespace simultaneous_eq_solution_l744_744504

theorem simultaneous_eq_solution (n : ℝ) (hn : n ≠ 1 / 2) : 
  ∃ (x y : ℝ), (y = (3 * n + 1) * x + 2) ∧ (y = (5 * n - 2) * x + 5) := 
sorry

end simultaneous_eq_solution_l744_744504


namespace number_of_permutations_satisfying_conditions_l744_744984

def is_permutation (l : List ℕ) : Prop :=
  l.perm (List.range 1 11)

def satisfies_conditions (l : List ℕ) : Prop :=
  l.length = 10 ∧ 
  (l.nth 0 > l.nth 1 ∧ l.nth 1 > l.nth 2 ∧ l.nth 2 > l.nth 3 ∧ l.nth 3 > l.nth 4) ∧
  (l.nth 4 < l.nth 5 ∧ l.nth 5 < l.nth 6 ∧ l.nth 6 < l.nth 7 ∧ l.nth 7 < l.nth 8 ∧ l.nth 8 < l.nth 9)

theorem number_of_permutations_satisfying_conditions : 
  ∃ (l : List ℕ), is_permutation l ∧ satisfies_conditions l ↔ ∃! n : ℕ, n = 126 :=
by
  sorry

end number_of_permutations_satisfying_conditions_l744_744984


namespace bc_is_one_area_of_triangle_l744_744232

-- Define a triangle with sides a, b, and c and corresponding angles A, B, and C.
structure Triangle :=
(a b c A B C : ℝ)

-- Assume the Law of Cosines condition is given.
axiom Law_of_Cosines_condition (T : Triangle) : (T.b^2 + T.c^2 - T.a^2) / (Real.cos T.A) = 2

-- Prove that bc = 1.
theorem bc_is_one (T : Triangle) (h : Law_of_Cosines_condition T) : T.b * T.c = 1 := 
by {
    sorry
}

-- Assume the given ratio condition.
axiom ratio_condition (T : Triangle) :
  (T.a * Real.cos T.B - T.b * Real.cos T.A) / (T.a * Real.cos T.B + T.b * Real.cos T.A)
  - (T.b / T.c) = 1

-- Prove that the area of the triangle is sqrt(3)/4.
theorem area_of_triangle (T : Triangle) (h1 : Law_of_Cosines_condition T) (h2 : ratio_condition T) :
  let S := 0.5 * T.b * T.c * Real.sin T.A
  in S = Real.sqrt 3 / 4 :=
by {
    sorry
}

end bc_is_one_area_of_triangle_l744_744232


namespace derivative_at_pi_div_two_l744_744526

def f (x : Real) : Real := x * Real.sin x

def f' (x : Real) : Real := 
  Real.sin x + x * Real.cos x
  
theorem derivative_at_pi_div_two :
  f' (Real.pi / 2) = 1 := by
  unfold f'
  rw [Real.sin_pi_div_two, Real.cos_pi_div_two]
  norm_num
  sorry

end derivative_at_pi_div_two_l744_744526


namespace ball_bouncing_height_l744_744413

theorem ball_bouncing_height : ∃ (b : ℕ), 400 * (3/4 : ℝ)^b < 50 ∧ ∀ n < b, 400 * (3/4 : ℝ)^n ≥ 50 :=
by
  use 8
  sorry

end ball_bouncing_height_l744_744413


namespace log_base_eq_l744_744760

theorem log_base_eq (x : ℝ) (h₁ : x > 0) (h₂ : x ≠ 1) : 
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 := 
by 
  sorry

end log_base_eq_l744_744760


namespace work_completion_time_l744_744022

theorem work_completion_time (A_work_rate B_work_rate C_work_rate : ℝ) 
  (hA : A_work_rate = 1 / 8) 
  (hB : B_work_rate = 1 / 16) 
  (hC : C_work_rate = 1 / 16) : 
  1 / (A_work_rate + B_work_rate + C_work_rate) = 4 :=
by
  -- Proof goes here
  sorry

end work_completion_time_l744_744022


namespace cube_has_empty_after_jumps_l744_744788

theorem cube_has_empty_after_jumps :
  ∀ (cube : fin 5 × fin 5 × fin 5) (initial_pos : fin 5 × fin 5 × fin 5 → Prop),
  (∀ p : fin 5 × fin 5 × fin 5, initial_pos p) →
  (∀ p : fin 5 × fin 5 × fin 5, ∃ q : fin 5 × fin 5 × fin 5,
    adjacent p q ∧ initial_pos q ∧ ¬ initial_pos p) →
  ∃ r, ¬ initial_pos r :=
by
  -- Placeholder proof
  sorry

-- Adjacent definition can be added for completeness
def adjacent (p q : fin 5 × fin 5 × fin 5) : Prop :=
  ∑ i, abs (p.1 - q.1) + abs (p.2 - q.2) + abs (p.3 - q.3) = 1

end cube_has_empty_after_jumps_l744_744788


namespace maximum_value_of_N_l744_744217

def S_pi (n : ℕ) (a : Fin n → ℕ) (pi : Equiv.Perm (Fin n)) : Finset (Fin n) :=
  {i : Fin n | (a i) % (pi i).val = 0}

def max_N (n : ℕ) : ℕ :=
  2^n - n

theorem maximum_value_of_N (n : ℕ) (a : Fin n → ℕ) :
  n > 0 → (∃pi : Equiv.Perm (Fin n), ∀ S ∈ (Fin n).powerset, 
    (∃ pi' : Equiv.Perm (Fin n), S_pi n a pi' = S) ↔
    S.card = n ∨ (Fin (n - 1) → False)) →
  ∃ (N : ℕ), N = max_N n :=
by sorry

end maximum_value_of_N_l744_744217


namespace partI_inequality_partII_inequality_l744_744547

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Part (Ⅰ): Prove f(x) ≤ x + 1 for 1 ≤ x ≤ 5
theorem partI_inequality (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 5) : f x ≤ x + 1 := by
  sorry

-- Part (Ⅱ): Prove (a^2)/(a+1) + (b^2)/(b+1) ≥ 1 when a + b = 2 and a > 0, b > 0
theorem partII_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) : 
    (a^2) / (a + 1) + (b^2) / (b + 1) ≥ 1 := by
  sorry

end partI_inequality_partII_inequality_l744_744547


namespace prove_bc_prove_area_l744_744226

variable {a b c A B C : ℝ}

-- Given conditions as Lean definitions
def condition_cos_A : Prop := (b^2 + c^2 - a^2 = 2 * cos A)
def condition_bc : Prop := b * c = 1
def condition_trig_identity : Prop := (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1

-- Definitions for the questions translated to Lean
def find_bc : Prop :=
  condition_cos_A → b * c = 1

def find_area (bc_value : Prop) : Prop :=
  bc_value →
  condition_trig_identity →
  1/2 * b * c * sin A = (Real.sqrt 3) / 4

-- Theorems to be proven
theorem prove_bc : find_bc :=
by
  intro h1
  sorry

theorem prove_area : find_area condition_bc :=
by
  intro h1 h2
  sorry

end prove_bc_prove_area_l744_744226


namespace most_likely_units_digit_l744_744853

open Fin

theorem most_likely_units_digit :
  ∃ d : Fin 10, (∀ n : Fin 10, n ≠ d → count_unit_digit_occurrences n < count_unit_digit_occurrences d) ∧ d = 0 :=
by
  -- Define picking integers from the set {1, 2, ..., 9}
  let J := Fin 9
  let K := Fin 9

  -- Define the sum of J and K and its units digit
  let S := J.val + K.val
  let units_digit := S % 10

  -- Define a function to count occurrences of each units digit
  noncomputable def count_unit_digit_occurrences (n : Fin 10) : ℕ :=
    (Finset.univ.filter (λ (pair : Fin 9 × Fin 9), (pair.1.val + pair.2.val) % 10 = n)).card

  -- The proof statement
  sorry

end most_likely_units_digit_l744_744853


namespace find_integer_pairs_l744_744473

theorem find_integer_pairs (a b : ℕ) (h_positive_a : a > 0) (h_positive_b : b > 0) :
  (ab2_plus_b_plus_7_divides_asb_plus_a_plus_b : (ab2 : ℕ := a * b * b + b + 7) ∣ (a2b := a * a * b + a + b)) ↔ 
  (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ ∃ k : ℕ, k > 0 ∧ a = 7 * k * k ∧ b = 7 * k :=
by {
  have h_ab2 : ℕ := a * b * b + b + 7,
  have h_a2b : ℕ := a * a * b + a + b,
  sorry
}

end find_integer_pairs_l744_744473


namespace bc_eq_one_area_of_triangle_l744_744248

variable (a b c A B : ℝ)

-- Conditions
def condition_1 : Prop := (b^2 + c^2 - a^2) / (Real.cos A) = 2
def condition_2 : Prop := (a * (Real.cos B) - b * (Real.cos A)) / (a * (Real.cos B) + b * (Real.cos A)) - b / c = 1

-- Equivalent proof problems
theorem bc_eq_one (h1 : condition_1 a b c A) : b * c = 1 := 
by 
  sorry

theorem area_of_triangle (h2 : condition_2 a b c A B) : (1/2) * b * c * Real.sin A = (Real.sqrt 3) / 4 := 
by 
  sorry

end bc_eq_one_area_of_triangle_l744_744248


namespace gasoline_car_p_ge_hybrid_electric_car_p_gasoline_car_p_le_100_hybrid_car_p_l744_744282

noncomputable def sound_pressure_level (p p0 : ℝ) : ℝ :=
20 * real.log10 (p / p0)

variables {p0 p1 p2 p3 : ℝ} (h_p0 : p0 > 0)
(h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
(h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)
(h_p3 : p3 = 100 * p0)

theorem gasoline_car_p_ge_hybrid (h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
                                (h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)  : p1 ≥ p2 :=
sorry

theorem electric_car_p (h_p3 : p3 = 100 * p0) : p3 = 100 * p0 :=
sorry

theorem gasoline_car_p_le_100_hybrid_car_p (h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
                                           (h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0) : p1 ≤ 100 * p2 :=
sorry

#check gasoline_car_p_ge_hybrid
#check electric_car_p
#check gasoline_car_p_le_100_hybrid_car_p

end gasoline_car_p_ge_hybrid_electric_car_p_gasoline_car_p_le_100_hybrid_car_p_l744_744282


namespace toms_weekly_revenue_l744_744346

def crabs_per_bucket : Nat := 12
def number_of_buckets : Nat := 8
def price_per_crab : Nat := 5
def days_per_week : Nat := 7

theorem toms_weekly_revenue :
  (crabs_per_bucket * number_of_buckets * price_per_crab * days_per_week) = 3360 :=
by
  sorry

end toms_weekly_revenue_l744_744346


namespace area_between_chords_l744_744357

theorem area_between_chords (r : ℝ) (d : ℝ) (θ : ℝ) : 
  r = 10 → 
  d = 6 → 
  θ = 2 * real.arccos(0.3) → 
  (r^2 * θ - d * real.sqrt(r^2 - (d / 2)^2)) = 100 * θ - 6 * real.sqrt(91) :=
by
  intros hr hd hθ
  rw [hr, hd, hθ]
  sorry

end area_between_chords_l744_744357


namespace parabola_properties_l744_744146

theorem parabola_properties 
  (p : ℝ) (h_pos : 0 < p) (m : ℝ) 
  (A B : ℝ × ℝ)
  (h_AB_on_parabola : ∀ (P : ℝ × ℝ), P = A ∨ P = B → (P.snd)^2 = 2 * p * P.fst) 
  (h_line_intersection : ∀ (P : ℝ × ℝ), P = A ∨ P = B → P.fst = m * P.snd + 3)
  (h_dot_product : (A.fst * B.fst + A.snd * B.snd) = 6)
  : (exists C : ℝ × ℝ, C = (-3, 0)) ∧
    (∃ k1 k2 : ℝ, 
        k1 = A.snd / (A.fst + 3) ∧ 
        k2 = B.snd / (B.fst + 3) ∧ 
        (1 / k1^2 + 1 / k2^2 - 2 * m^2) = 24) :=
by
  sorry

end parabola_properties_l744_744146


namespace math_problem_eq_l744_744455

theorem math_problem_eq :
  log 3 (sqrt 27) + (8 / 125) ^ (- 1 / 3) - (- 3 / 5) ^ 0 + 416 ^ 3 = 11 := by
  sorry

end math_problem_eq_l744_744455


namespace number_of_real_roots_l744_744498

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem number_of_real_roots (a : ℝ) :
    ((|a| < (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ x₃ : ℝ, f x₁ = a ∧ f x₂ = a ∧ f x₃ = a ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ∧
    ((|a| > (2 * Real.sqrt 3) / 9) → (∃ x : ℝ, f x = a ∧ ∀ y : ℝ, f y = a → y = x)) ∧
    ((|a| = (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ : ℝ, f x₁ = a ∧ f x₂ = a ∧ x₁ ≠ x₂ ∧ ∀ y : ℝ, (f y = a → (y = x₁ ∨ y = x₂)) ∧ (x₁ = x₂ ∨ ∀ z : ℝ, (f z = a → z = x₁ ∨ z = x₂)))) := sorry

end number_of_real_roots_l744_744498


namespace Zia_club_l744_744765

theorem Zia_club (degrees_per_sector : Fin 6 → ℕ)
  (h₁ : degrees_per_sector 0 = 35)
  (h₂ : ∀ i, degrees_per_sector (i + 1) = degrees_per_sector i + 10)
  (h₃ : (∑ i, degrees_per_sector i) = 360)
  (h₄ : ∃! i, degrees_per_sector i = 35)
  (seven_people_sector_angle : degrees_per_sector (Fin.ofNat 0) = 35) :
  (360 / (35 / 7)) = 72 :=
by
  sorry

end Zia_club_l744_744765


namespace sum_x_y_z_l744_744250

noncomputable def a : ℝ := -Real.sqrt (9/27)
noncomputable def b : ℝ := Real.sqrt ((3 + Real.sqrt 7)^2 / 9)

theorem sum_x_y_z (ha : a = -Real.sqrt (9 / 27)) (hb : b = Real.sqrt ((3 + Real.sqrt 7) ^ 2 / 9)) (h_neg_a : a < 0) (h_pos_b : b > 0) :
  ∃ x y z : ℕ, (a + b)^3 = (x * Real.sqrt y) / z ∧ x + y + z = 718 := 
sorry

end sum_x_y_z_l744_744250


namespace Leroy_min_bail_rate_l744_744761

noncomputable def min_bailing_rate
    (distance_to_shore : ℝ)
    (leak_rate : ℝ)
    (max_tolerable_water : ℝ)
    (rowing_speed : ℝ)
    : ℝ :=
  let time_to_shore := distance_to_shore / rowing_speed * 60
  let total_water_intake := leak_rate * time_to_shore
  let required_bailing := total_water_intake - max_tolerable_water
  required_bailing / time_to_shore

theorem Leroy_min_bail_rate
    (distance_to_shore : ℝ := 2)
    (leak_rate : ℝ := 15)
    (max_tolerable_water : ℝ := 60)
    (rowing_speed : ℝ := 4)
    : min_bailing_rate 2 15 60 4 = 13 := 
by
  simp [min_bailing_rate]
  sorry

end Leroy_min_bail_rate_l744_744761


namespace share_of_B_profit_l744_744809

theorem share_of_B_profit (b d : ℝ) (total_profit : ℝ) 
  (condition1 : ∃ a, a = (1/2) * b)
  (condition2 : b = 2 * (b/2))
  (condition3 : ∃ c, c = (1/3) * d)
  (a' : ℝ := 2 * (1/2) * b)
  (c' : ℝ := (1/2) * (b/2))
  (a_total : ℝ := ((1/2) * b) + a')
  (b_total : ℝ := 2 * b)
  (c_total : ℝ := (b/2) + c')
  (d_total : ℝ := 2 * d)
  (total : ℝ := a_total + b_total + c_total + d_total) :
  total_profit = 36000 → (8 / 29) * total_profit = 9600 := 
by 
  sorry

-- assumptions from the problem in simplified form
axiom b_is_2c (c : ℝ) : b = 2 * c 
axiom d_is_3b (b : ℝ) : d = 3 * (b / 2)

end share_of_B_profit_l744_744809


namespace shifted_linear_func_is_2x_l744_744181

-- Define the initial linear function
def linear_func (x : ℝ) : ℝ := 2 * x - 3

-- Define the shifted linear function
def shifted_linear_func (x : ℝ) : ℝ := linear_func x + 3

theorem shifted_linear_func_is_2x (x : ℝ) : shifted_linear_func x = 2 * x := by
  -- Proof would go here, but we use sorry to skip it
  sorry

end shifted_linear_func_is_2x_l744_744181


namespace boundary_internal_circles_l744_744787

theorem boundary_internal_circles (n : ℕ) 
  (convex_n_gon : Π (i : ℕ) (hi : 1 ≤ i ∧ i ≤ n), Type) 
  (no_four_vertices_same_circle : ∀ (a b c d : ℕ) 
    (hac : a ≠ c ∧ a ≠ b ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d),
    ¬ collinear_convex convex_n_gon {a, b, c, d})
  (described_circle_boundary : Π (a b c: ℕ) 
    (habc : a = b + 1 ∧ b = c + 1 ∨ b = a + 1 ∧ a = c + 1 ∨ c = a + 1 ∧ b = c + 1),
    circle_description convex_n_gon {a, b, c})
  (described_circle_internal : Π (a b c: ℕ) 
    (hnot_adj : a ≠ b + 1 ∧ a ≠ c + 1 ∧ b ≠ a + 1 ∧ b ≠ c + 1 ∧ c ≠ a + 1 ∧ c ≠ b + 1) 
    (htri : set.convex.convex_hull {convex_n_gon a, convex_n_gon b, convex_n_gon c} 
      (λ v, ∃ p1 p2 p3, p1 ≠ p3 ∧ v ∈ convex_hull ℝ {p1, p2, p3})),
    circle_description convex_n_gon {a, b, c}) 
    (Γ B : ℕ) :
  n > 3 → Γ - B = 2 := 
begin 
  sorry 
end

end boundary_internal_circles_l744_744787


namespace initial_goats_l744_744289

theorem initial_goats (G : ℕ) (h1 : 2 + 3 + G + 3 + 5 + 2 = 21) : G = 4 :=
by
  sorry

end initial_goats_l744_744289


namespace lloyd_hourly_rate_l744_744262

-- Define the given conditions
def normal_hours := 7.5
def overtime_multiplier := 2.0
def total_hours_worked := 10.5
def total_earnings := 60.75

-- State the problem
theorem lloyd_hourly_rate :
  ∃ R, 
    (7.5 * R) + (total_hours_worked - normal_hours) * (overtime_multiplier * R) = total_earnings 
    ∧ R = 4.5 :=
begin
  sorry
end

end lloyd_hourly_rate_l744_744262


namespace find_b_l744_744584

theorem find_b (b : ℚ) (h : ∃ c : ℚ, (3 * x + c)^2 = 9 * x^2 + 27 * x + b) : b = 81 / 4 := 
sorry

end find_b_l744_744584


namespace two_variables_with_scatter_plot_l744_744812

-- Definition of two variables for statistical data
def variable1 : Type := sorry
def variable2 : Type := sorry

-- Condition 1: Analysis of relationship between two variables
def analysis_of_relationship (v1 v2 : Type) : Prop := sorry

-- Condition 2: Representation by a scatter plot
def can_be_represented_with_scatter_plot (v1 v2 : Type) : Prop := sorry

-- Theorem statement: Both variables can be represented with a scatter plot
theorem two_variables_with_scatter_plot (v1 v2 : Type) 
  (h1 : analysis_of_relationship v1 v2)
  (h2 : can_be_represented_with_scatter_plot v1 v2) : 
  can_be_represented_with_scatter_plot v1 v2 :=
begin
  sorry
end

end two_variables_with_scatter_plot_l744_744812


namespace probability_of_technic_l744_744481

-- Define the set of letters in "MATHEMATICS"
def mathematics_letters : Multiset Char := ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S']

-- Define the set of letters in "TECHNIC"
def technic_letters : Set Char := {'T', 'E', 'C', 'H', 'N', 'I'}

-- Define the condition of counting relevant letters from "TECHNIC" in "MATHEMATICS"
def count_relevant_letters (s : Multiset Char) (set_char : Set Char) : ℕ :=
  s.countp (λ c, c ∈ set_char)

-- Define the mathematical proof statement
theorem probability_of_technic : 
  count_relevant_letters mathematics_letters technic_letters = 6 →
  (6 / 12 : Rat) = 1 / 2 :=
by
  intros h,
  have : 6 / 12 = (1 / 2 : Rat) := sorry,
  exact this

end probability_of_technic_l744_744481


namespace students_more_than_rabbits_l744_744086

/- Define constants for the problem. -/
def students_per_class : ℕ := 20
def rabbits_per_class : ℕ := 3
def num_classes : ℕ := 5

/- Define total counts based on given conditions. -/
def total_students : ℕ := students_per_class * num_classes
def total_rabbits : ℕ := rabbits_per_class * num_classes

/- The theorem we need to prove: The difference between total students and total rabbits is 85. -/
theorem students_more_than_rabbits : total_students - total_rabbits = 85 := by
  sorry

end students_more_than_rabbits_l744_744086


namespace alice_winning_equivalence_l744_744810

noncomputable def alice_winning (k n : ℕ) : Prop := sorry

theorem alice_winning_equivalence (k l l' : ℕ) (h : k > 2)
  (h1 : ∀ p : ℕ, p.prime → p ≤ k → (p ∣ l ↔ p ∣ l')) :
  (alice_winning k l ↔ alice_winning k l') :=
sorry

end alice_winning_equivalence_l744_744810


namespace disc_partitioning_l744_744216

-- Define the problem with relevant conditions
theorem disc_partitioning {n k : ℕ} (hn : n > 0) (hk : k > 0)
  (h : ∀ (D : Fin n.succ → Set (ℝ × ℝ)), (∀ (i : Fin n.succ), is_closed (D i)) → 
    (∀ (s : Fin (n.succ - 1) → Fin n.succ), (pairwise (λ i j, Disjoint (D (s i)) (D (s j))))) -> 
      pairwise (λ i j, Disjoint (D i) (D j))) :
  ∃ (C : Fin n.succ → Fin (10 * k).succ), 
    pairwise (λ i j, C i = C j → Disjoint (D i) (D j)) := 
sorry

end disc_partitioning_l744_744216


namespace binomial_distribution_n_value_l744_744109

theorem binomial_distribution_n_value 
  (n : ℕ) 
  (h1 : ∃ X : ℝ, X ∼ B(n, 0.8)) 
  (h2 : (n : ℝ) * 0.8 * 0.2 = 1.6) : 
  n = 10 := 
by 
  sorry

end binomial_distribution_n_value_l744_744109


namespace ratio_of_p_to_q_l744_744592

theorem ratio_of_p_to_q (p q r : ℚ) (h1: p = r * q) (h2: 18 / 7 + (2 * q - p) / (2 * q + p) = 3) : r = 29 / 10 :=
by
  sorry

end ratio_of_p_to_q_l744_744592


namespace log_probability_l744_744734

open Nat

/-- 
Given c and d are distinct numbers randomly chosen 
from the set {3, 3^2, 3^3, ..., 3^20}, 
the probability that log_c(d) is an integer is 32/95.
-/
theorem log_probability (c d : ℕ) (h_c_in_set : ∃ m : ℕ, m ∈ range 1 21 ∧ c = 3 ^ m)
(h_d_in_set : ∃ n : ℕ, n ∈ range 1 21 ∧ d = 3 ^ n) (h_distinct : c ≠ d) :
  (∃ k : ℕ, log c d = k) ↔ 32 / 95 :=
sorry

end log_probability_l744_744734


namespace toms_weekly_income_l744_744350

variable (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def daily_crabs := num_buckets * crabs_per_bucket
def daily_income := daily_crabs * price_per_crab
def weekly_income := daily_income * days_per_week

theorem toms_weekly_income 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  weekly_income num_buckets crabs_per_bucket price_per_crab days_per_week = 3360 :=
by
  sorry

end toms_weekly_income_l744_744350


namespace first_player_wins_l744_744359

-- Given conditions
structure Game :=
(table : ℝ × ℝ) -- rectangular table
(coins : set (ℝ × ℝ)) -- positions of coins (free spots)
(turns : ℕ) -- number of turns taken

def valid_move (g : Game) (pos : ℝ × ℝ) : Prop :=
  pos ∉ g.coins∧ pos.1 ≥ 0 ∧ pos.2 ≥ 0 ∧ pos.1 ≤ g.table.1 ∧ pos.2 ≤ g.table.2

-- Goal: Prove that the first player can always win
theorem first_player_wins (g : Game) : ∃ strategy : ℕ → ℝ × ℝ, (∀ n, valid_move (place_coin g (strategy n)) (strategy n)) → 
  loser (final_position g strategy) = second_player :=
sorry

end first_player_wins_l744_744359


namespace Antonio_eats_meatballs_l744_744820

def meatballs_per_member (total_hamburger : ℝ) (hamburger_per_meatball : ℝ) (num_family_members : ℕ) : ℝ :=
  (total_hamburger / hamburger_per_meatball) / num_family_members

theorem Antonio_eats_meatballs :
  meatballs_per_member 4 (1 / 8) 8 = 4 := 
by
  sorry

end Antonio_eats_meatballs_l744_744820


namespace bs_sequence_bounded_iff_f_null_l744_744437

def is_bs_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = abs (a (n + 1) - a (n + 2))

def f_null (a : ℕ → ℝ) : Prop :=
  ∀ n k, a n * a k * (a n - a k) = 0

def bs_bounded (a : ℕ → ℝ) : Prop :=
  ∃ M, ∀ n, abs (a n) ≤ M

theorem bs_sequence_bounded_iff_f_null (a : ℕ → ℝ) :
  is_bs_sequence a →
  (bs_bounded a ↔ f_null a) := by
  sorry

end bs_sequence_bounded_iff_f_null_l744_744437


namespace count_leap_years_in_range_l744_744040

-- Define the conditions given in the problem.
def is_leap_year (y : ℕ) : Prop :=
  y % 1200 = 300

def in_range (y : ℕ) : Prop :=
  1500 ≤ y ∧ y ≤ 4500

-- Putting the question and the correct answer together in the Lean statement.
theorem count_leap_years_in_range : 
  (finset.filter (λ y, is_leap_year y ∧ in_range y) (finset.range 4501)).card = 2 :=
sorry

end count_leap_years_in_range_l744_744040


namespace kaleb_first_load_pieces_l744_744212

-- Definitions of given conditions
def total_pieces : ℕ := 39
def num_equal_loads : ℕ := 5
def pieces_per_load : ℕ := 4

-- Definition for calculation of pieces in equal loads
def pieces_in_equal_loads : ℕ := num_equal_loads * pieces_per_load

-- Definition for pieces in the first load
def pieces_in_first_load : ℕ := total_pieces - pieces_in_equal_loads

-- Statement to prove that the pieces in the first load is 19
theorem kaleb_first_load_pieces : pieces_in_first_load = 19 := 
by
  -- The proof is skipped
  sorry

end kaleb_first_load_pieces_l744_744212


namespace proportion_of_angular_speeds_l744_744955

-- Define the inputs for the problem
variables (p q r k : ℝ)

-- Conditions from the problem
def omega_A := 2 * (k / p)
def omega_B := k / q
def omega_C := k / r

-- The proof problem statement
theorem proportion_of_angular_speeds : omega_A / omega_B = 2 * (r / q) ∧ omega_A / omega_C = 2 * (q / p) := by
  sorry

end proportion_of_angular_speeds_l744_744955


namespace cos_alpha_second_quadrant_l744_744529

theorem cos_alpha_second_quadrant (α : ℝ) 
  (h1 : (π / 2 < α ∧ α < π))
  (h2 : cos (π / 2 - α) = 4 / 5) : 
  cos α = -3 / 5 := by
  sorry

end cos_alpha_second_quadrant_l744_744529


namespace angle_between_vectors_l744_744558

variables {φ θ : ℝ}

def vector_a (φ : ℝ) : ℝ × ℝ := (2 * real.cos φ, 2 * real.sin φ)
def vector_b : ℝ × ℝ := (0, -1)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem angle_between_vectors (hφ : φ ∈ set.Ioo (real.pi / 2) real.pi) :
  θ = 3 * real.pi / 2 - φ ↔ real.cos θ = -real.sin φ := by
  sorry

end angle_between_vectors_l744_744558


namespace first_player_wins_l744_744358

-- Given conditions
structure Game :=
(table : ℝ × ℝ) -- rectangular table
(coins : set (ℝ × ℝ)) -- positions of coins (free spots)
(turns : ℕ) -- number of turns taken

def valid_move (g : Game) (pos : ℝ × ℝ) : Prop :=
  pos ∉ g.coins∧ pos.1 ≥ 0 ∧ pos.2 ≥ 0 ∧ pos.1 ≤ g.table.1 ∧ pos.2 ≤ g.table.2

-- Goal: Prove that the first player can always win
theorem first_player_wins (g : Game) : ∃ strategy : ℕ → ℝ × ℝ, (∀ n, valid_move (place_coin g (strategy n)) (strategy n)) → 
  loser (final_position g strategy) = second_player :=
sorry

end first_player_wins_l744_744358


namespace sum_of_roots_is_zero_l744_744593

variables {R : Type*} [Field R] {a b c p q : R}

theorem sum_of_roots_is_zero (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a^3 + p * a + q = 0) (h₅ : b^3 + p * b + q = 0) (h₆ : c^3 + p * c + q = 0) :
  a + b + c = 0 :=
by
  sorry

end sum_of_roots_is_zero_l744_744593


namespace integer_solution_count_l744_744155

theorem integer_solution_count : ∃ (n : ℕ), n = 53 ∧ 
  (∀ (x y : ℤ), x ≠ 0 → y ≠ 0 → (1 : ℚ) / 2022 = (1 : ℚ) / x + (1 : ℚ) / y → 
  (∃ (a b : ℤ), 2022 * (a - 2022) * (b - 2022) = 2022^2) :=
begin
  use 53,
  split, {
    refl,
  },
  intros x y hx hy hxy,
  sorry
end

end integer_solution_count_l744_744155


namespace perpendicular_k_value_l744_744563

variable (k : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 1)
def c (k : ℝ) : ℝ × ℝ := (1 + k, 2 + k)

theorem perpendicular_k_value : k = -3 / 2 →
  let c := c k in
  (b.1 * c.1 + b.2 * c.2 = 0) :=
by
  sorry

end perpendicular_k_value_l744_744563


namespace find_k_l744_744926

theorem find_k (a b : ℤ × ℤ) (k : ℤ) 
  (h₁ : a = (2, 1)) 
  (h₂ : a.1 + b.1 = 1 ∧ a.2 + b.2 = k)
  (h₃ : a.1 * b.1 + a.2 * b.2 = 0) : k = 3 :=
sorry

end find_k_l744_744926


namespace min_dist_PQ_l744_744300

-- Defining the circles
def C1 (P : ℝ × ℝ) : Prop := (P.1 - 4) ^ 2 + (P.2 - 2) ^ 2 = 9
def C2 (Q : ℝ × ℝ) : Prop := (Q.1 + 2) ^ 2 + (Q.2 + 1) ^ 2 = 6

-- Distance function
def dist (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Centers of the circles
def center_C1 : ℝ × ℝ := (4, 2)
def center_C2 : ℝ × ℝ := (-2, -1)

-- Radii of the circles
def radius_C1 : ℝ := 3
def radius_C2 : ℝ := Real.sqrt 6

-- Distance between centers
def center_dist : ℝ := dist center_C1 center_C2

-- Minimum distance between points P and Q on the two circles
theorem min_dist_PQ : ∀ (P Q : ℝ × ℝ), 
  (C1 P) → (C2 Q) → dist P Q = (3 * Real.sqrt 5 - 3 - Real.sqrt 6) :=
by
  sorry

end min_dist_PQ_l744_744300


namespace quadrilateral_circumcenter_sum_l744_744646

noncomputable theory

-- Definitions of circumcenter and circumradius
variables {A : Type} [metric_space A] (A1 A2 A3 A4 : A)

def circumcenter (A B C : A) : A := sorry
def circumradius (A B C : A) : ℝ := sorry

-- Conditions: A non-cyclic quadrilateral
def non_cyclic (A1 A2 A3 A4 : A) : Prop := sorry

-- Main theorem to be proved
theorem quadrilateral_circumcenter_sum (A1 A2 A3 A4 O1 O2 O3 O4 : A)
  (r1 r2 r3 r4 : ℝ)
  (h_noncyclic : non_cyclic A1 A2 A3 A4)
  (h1 : O1 = circumcenter A2 A3 A4)
  (h2 : O2 = circumcenter A1 A3 A4)
  (h3 : O3 = circumcenter A1 A2 A4)
  (h4 : O4 = circumcenter A1 A2 A3)
  (r1_def : r1 = circumradius A2 A3 A4)
  (r2_def : r2 = circumradius A1 A3 A4)
  (r3_def : r3 = circumradius A1 A2 A4)
  (r4_def : r4 = circumradius A1 A2 A3) :
  (1 / (dist O1 A1 ^ 2 - r1 ^ 2)) +
  (1 / (dist O2 A2 ^ 2 - r2 ^ 2)) +
  (1 / (dist O3 A3 ^ 2 - r3 ^ 2)) +
  (1 / (dist O4 A4 ^ 2 - r4 ^ 2)) = 0 :=
sorry

end quadrilateral_circumcenter_sum_l744_744646


namespace combined_work_days_l744_744392

theorem combined_work_days (W D : ℕ) (h1: ∀ a b : ℕ, a + b = 4) (h2: (1/6:ℝ) = (1/6:ℝ)) :
  D = 4 :=
by
  sorry

end combined_work_days_l744_744392


namespace ratio_of_sequence_terms_l744_744923

theorem ratio_of_sequence_terms 
  (a b : ℕ → ℕ)
  (h : ∀ n : ℕ, n > 0 → (∑ i in finset.range n, a i) / (∑ i in finset.range n, b i) = (7 * n + 1) / (4 * n + 27)) :
  (a 10 / b 10) = 4 / 3 :=
  by
    sorry

end ratio_of_sequence_terms_l744_744923


namespace area_AMDN_eq_area_ABC_l744_744613

open Real Plane Geometry

/- Declare types representing points and triangles -/
variables {A B C D E F M N : Point}

/- Declare conditions -/
axiom h_acute_ABC : acute_triangle A B C
axiom h_E_and_F_on_BC : on_line_segment B C E ∧ on_line_segment B C F
axiom h_angle_BAE_eq_CAF : ∠BAE = ∠CAF
axiom h_FM_perp_AB : perp_line_through_point AB F M
axiom h_FN_perp_AC : perp_line_through_point AC F N
axiom h_AE_intersects_circumcircle_ABC_at_D : intersects_circumcircle A E (circumcircle A B C) D

/- Proof goal -/
theorem area_AMDN_eq_area_ABC :
  area_quadrilateral A M D N = area_triangle A B C :=
sorry

end area_AMDN_eq_area_ABC_l744_744613


namespace unique_identity_element_l744_744737

variable {G : Type*} [Group G]

theorem unique_identity_element (e e' : G) (h1 : ∀ g : G, e * g = g ∧ g * e = g) (h2 : ∀ g : G, e' * g = g ∧ g * e' = g) : e = e' :=
by 
sorry

end unique_identity_element_l744_744737


namespace smallest_four_digit_2_mod_11_l744_744375

theorem smallest_four_digit_2_mod_11 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 11 = 2 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 11 = 2 → n ≤ m) := 
by 
  use 1003
  sorry

end smallest_four_digit_2_mod_11_l744_744375


namespace number_count_l744_744317

theorem number_count (N : ℕ)
  (avg_all : (∑ i in finRange N, i) / N = 3.9)
  (avg_first_two : (3.4 + 3.4) / 2 = 3.4)
  (avg_second_two : (3.85 + 3.85) / 2 = 3.85)
  (avg_remaining_two : (4.45 + 4.45) / 2 = 4.45) :
  N = 6 :=
by
  sorry

end number_count_l744_744317


namespace log_prop_1_log_prop_2_log_prop_3_log_prop_4_log_prop_5_l744_744849

variable {a M N x y : ℝ}

-- 1st statement
theorem log_prop_1 (h : M > 0) (h' : N > 0) : log a (M*N) = log a M + log a N → false := sorry

-- 2nd statement
theorem log_prop_2 (h₁ : x > 0) (h₂ : y > 0) : log a x * log a y = log a (x + y) → false := sorry

-- 3rd statement
theorem log_prop_3 : ¬(is_logarithm (fun x => log 2 x) ∧ is_logarithm (fun x => log (1/3) (3*x))) := sorry

-- 4th statement
theorem log_prop_4 (ha : a > 0) (ha' : a ≠ 1) : (∀ x > 0, log a x > 0) → false := sorry

-- 5th statement
theorem log_prop_5 (ha : a > 0) (ha' : a ≠ 1) : (∀ x, x > 0 → 
  (log a 1 = 0 ∧ log a a = 1 ∧ log a (1/a) = -1 ∧ 
   (log a x > 0 ∨ log a x < 0))) := sorry

end log_prop_1_log_prop_2_log_prop_3_log_prop_4_log_prop_5_l744_744849


namespace geometric_sequence_property_l744_744607

theorem geometric_sequence_property (n : ℕ) (n_pos : 0 < n) (q : ℝ) (q_pos : 0 < q)
  (a : ℕ → ℝ) (h1 : a 1 = 2) 
  (h2 : 2 * a 1, a 3, 3 * a 2 form an arithmetic sequence) :
  a n = 2 ^ n :=
sorry

end geometric_sequence_property_l744_744607


namespace part1_part2_l744_744195

open Classical
noncomputable theory

-- Define the initial state of Box A
def boxA : Type := {q // q < 2} ∪ {q // 2 ≤ q ∧ q < 4}

-- Define the initial state of Box B
def boxB : Type := {q // q < 2} ∪ {q // 2 ≤ q ∧ q < 5}

-- Define the events related to student A drawing two questions from Box A
def event_A_1 (q : boxA) : Prop := q.val < 2
def event_A_2 (first_q second_q : boxA) : Prop := second_q.val < 2

-- Define the events related to student B drawing two questions from Box B after incorrect replacement
def event_B_1 (q : boxB) : Prop := q.val < 2

-- Probability function
def probability (event : Set boxA) : ℝ := 
  Set.toFinset event.card.toReal / Set.toFinset (@Set.univ boxA).card.toReal

theorem part1 :
  ∀ first_q second_q : boxA, 
  probability {q | event_A_2 first_q q} = 1 / 2 :=
sorry

theorem part2 :
  ∀ first_q second_q : boxB, 
  probability {q | event_B_1 q} = 3 / 7 :=
sorry

end part1_part2_l744_744195


namespace target_has_more_tools_l744_744741

-- Define the number of tools in the Walmart multitool
def walmart_screwdriver : ℕ := 1
def walmart_knives : ℕ := 3
def walmart_other_tools : ℕ := 2
def walmart_total_tools : ℕ := walmart_screwdriver + walmart_knives + walmart_other_tools

-- Define the number of tools in the Target multitool
def target_screwdriver : ℕ := 1
def target_knives : ℕ := 2 * walmart_knives
def target_files_scissors : ℕ := 3 + 1
def target_total_tools : ℕ := target_screwdriver + target_knives + target_files_scissors

-- The theorem stating the difference in the number of tools
theorem target_has_more_tools : (target_total_tools - walmart_total_tools) = 6 := by
  sorry

end target_has_more_tools_l744_744741


namespace find_ab_range_k_nat_ineq_l744_744546

-- Problem 1: Proving the values of a and b
theorem find_ab (a b : ℝ) (f : ℝ → ℝ) (h₁ : f x = a * x + b - log x) (h₂ : f 2 = 2 * a + b - log 2) (h₃ : (∀ x, diff f x = a - 1 / x)) :
    a = 1 ∧ b = -1 := sorry

-- Problem 2: Proving the range of k
theorem range_k (kx : ℝ) (hx : f x ≥ kx - 2 ∀ x (0 < x) : (kx ≤ 1 - 1 / exp 2)) := sorry

-- Problem 3: Proving the inequality for natural numbers
theorem nat_ineq (n : ℕ⁺) : n * (n + 1) ≤ 2 * (exp n - 1) / (exp 1 - 1) := sorry

end find_ab_range_k_nat_ineq_l744_744546


namespace Hannah_cut_strands_l744_744153

variable (H : ℕ)

theorem Hannah_cut_strands (h : 2 * (H + 3) = 22) : H = 8 :=
by
  sorry

end Hannah_cut_strands_l744_744153


namespace find_number_l744_744850

theorem find_number : ∃ x : ℝ, (6 * ((x / 8 + 8) - 30) = 12) ∧ x = 192 :=
by sorry

end find_number_l744_744850


namespace yoongi_has_smallest_points_l744_744632

def points_jungkook : ℕ := 6 + 3
def points_yoongi : ℕ := 4
def points_yuna : ℕ := 5

theorem yoongi_has_smallest_points : points_yoongi < points_jungkook ∧ points_yoongi < points_yuna :=
by
  sorry

end yoongi_has_smallest_points_l744_744632


namespace sally_picks_correct_number_of_peaches_l744_744672

theorem sally_picks_correct_number_of_peaches
  (initial_peaches : ℕ)
  (total_peaches : ℕ)
  (picked_peaches : ℕ)
  (h1 : initial_peaches = 13)
  (h2 : total_peaches = 55) :
  picked_peaches = total_peaches - initial_peaches → picked_peaches = 42 :=
by
  intros h
  simp [h1, h2] at h
  rw h
  sorry

end sally_picks_correct_number_of_peaches_l744_744672


namespace value_of_a_l744_744589

theorem value_of_a (a : ℝ) (x : ℝ) (h : 2 * x + 3 * a = -1) (hx : x = 1) : a = -1 :=
by
  sorry

end value_of_a_l744_744589


namespace inequality_two_vars_generalized_inequality_l744_744258

theorem inequality_two_vars {n : ℕ} {p q : ℝ} (hpq : p + q = 1) (hp : p ≥ 0) (hq : q ≥ 0) 
  {x y : Fin n → ℝ} (hx : ∀ i, x i > 0) (hy : ∀ i, y i > 0) :
  (∑ i, (x i) ^ p * (y i) ^ q) ≤ (∑ i, x i) ^ p * (∑ i, y i) ^ q := 
sorry

theorem generalized_inequality {k n : ℕ} {p : Fin k → ℝ} (hp_sum : (∑ i, p i) = 1) 
  (hp : ∀ i, p i ≥ 0)
  {x : Fin k → Fin n → ℝ} (hx : ∀ j i, x j i > 0) :
  (∑ i, (∏ j, (x j i) ^ (p j))) ≤ (∏ j, (∑ i, x j i) ^ (p j)) := 
sorry

end inequality_two_vars_generalized_inequality_l744_744258


namespace july_has_five_fridays_l744_744681
open Nat

theorem july_has_five_fridays (N : ℕ) (june_has_five_tuesdays : ∃ d1 d2 d3 d4 d5, {d1, d2, d3, d4, d5} = {2, 9, 16, 23, 30}) :
  ∃ f1 f2 f3 f4 f5, {f1, f2, f3, f4, f5} = {3, 10, 17, 24, 31} :=
by
  sorry

end july_has_five_fridays_l744_744681


namespace edward_friend_score_l744_744483

theorem edward_friend_score (total_points edward_points : ℕ) (h_total : total_points = 13) (h_edward : edward_points = 7) : total_points - edward_points = 6 :=
by
  simp [h_total, h_edward]
  sorry

end edward_friend_score_l744_744483


namespace average_production_n_days_l744_744870

theorem average_production_n_days (n : ℕ) (P : ℕ) 
  (hP : P = 80 * n)
  (h_new_avg : (P + 220) / (n + 1) = 95) : 
  n = 8 := 
by
  sorry -- Proof of the theorem

end average_production_n_days_l744_744870


namespace bisecting_lines_through_inner_point_l744_744031

theorem bisecting_lines_through_inner_point {S : Type} [ConvexPolygon S] (P : Point) (n : ℕ) 
    (h1 : has_n_sides S n) 
    (h2 : no_parallel_sides S)
    (h3 : is_inner_point P S) : 
    ∃ m, (m ≤ n) ∧ (∀ e : Line, (passes_through P e) → (bisects_area S e)) → (m = n) :=
by
  sorry

end bisecting_lines_through_inner_point_l744_744031


namespace antonio_meatballs_l744_744823

-- Define the conditions
def meat_per_meatball : ℝ := 1 / 8
def family_members : ℕ := 8
def total_hamburger : ℝ := 4

-- Assertion to prove
theorem antonio_meatballs : 
  (total_hamburger / meat_per_meatball) / family_members = 4 :=
by sorry

end antonio_meatballs_l744_744823


namespace count_integers_abs_le_5_l744_744328

theorem count_integers_abs_le_5 : (Set.toFinset {x : ℤ | |x| ≤ 5}).card = 11 :=
by
  sorry

end count_integers_abs_le_5_l744_744328


namespace subset_bound_l744_744112

theorem subset_bound (n k m : ℕ) (S : Finset (Finset ℕ)) 
  (hS_card : S.card = m)
  (hS_size : ∀ A B ∈ S, A ≠ B → (A ∩ B).card < k) :
  m ≤ ∑ i in Finset.range (k + 1), Nat.choose n i :=
sorry

end subset_bound_l744_744112


namespace logarithmic_relationship_l744_744891

theorem logarithmic_relationship
  (a b c : ℝ) (m n r : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c)
  (h5 : m = Real.log c / Real.log a)
  (h6 : n = Real.log c / Real.log b)
  (h7 : r = a ^ c) :
  n < m ∧ m < r :=
sorry

end logarithmic_relationship_l744_744891


namespace symmetric_function_value_a_l744_744942

theorem symmetric_function_value_a :
  ∃ (a : ℝ), (∀ x : ℝ, ∃ f : ℝ → ℝ, f x = (1/2) * real.sin (2 * x) + a * real.cos (2 * x) ∧ f (π / 12 - x) = f (π / 12 + x)) → a = real.sqrt 3 / 2 :=
begin
  sorry
end

end symmetric_function_value_a_l744_744942


namespace cot_identity_l744_744988

theorem cot_identity
  (a b c : ℝ)
  (α β γ : ℝ)
  (h1 : a^2 + b^2 = 2001 * c^2)
  (h2 : a > 0 ∧ b > 0 ∧ c > 0)
  (h3 : α + β + γ = Real.pi)
  (h4 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos γ)
  (h5 : Real.sin α ≠ 0 ∧ Real.sin β ≠ 0 ∧ Real.sin γ ≠ 0) :
  Real.cot γ / (Real.cot α + Real.cot β) = 1000 :=
sorry

end cot_identity_l744_744988


namespace problem_curves_l744_744690

theorem problem_curves (x y : ℝ) : 
  ((x * (x^2 + y^2 - 4) = 0 → (x = 0 ∨ x^2 + y^2 = 4)) ∧
  (x^2 + (x^2 + y^2 - 4)^2 = 0 → ((x = 0 ∧ y = -2) ∨ (x = 0 ∧ y = 2)))) :=
by
  sorry -- proof to be filled in later

end problem_curves_l744_744690


namespace find_unknown_number_l744_744626

-- Definitions

-- Declaring that we have an inserted number 'a' between 3 and unknown number 'b'
variable (a b : ℕ)

-- Conditions provided in the problem
def arithmetic_sequence_condition (a b : ℕ) : Prop := 
  a - 3 = b - a

def geometric_sequence_condition (a b : ℕ) : Prop :=
  (a - 6) / 3 = b / (a - 6)

-- The theorem statement equivalent to the problem
theorem find_unknown_number (h1 : arithmetic_sequence_condition a b) (h2 : geometric_sequence_condition a b) : b = 27 :=
sorry

end find_unknown_number_l744_744626


namespace how_many_positive_integers_divide_l744_744502

theorem how_many_positive_integers_divide (n : ℕ) :
  ∃ B : ℕ, B = 7 ∧
  (∀ k > 0, 
    ((1 + 2 + ... + k = k * (k + 1) / 2) → 
    (12 * k % (k * (k + 1) / 2) = 0)) → 
    (k + 1 ∣ 24)) :=
sorry

end how_many_positive_integers_divide_l744_744502


namespace carol_first_six_probability_l744_744446

theorem carol_first_six_probability :
  let p := 1 / 6
  let q := 5 / 6
  let prob_cycle := q^4
  (p * q^3) / (1 - prob_cycle) = 125 / 671 :=
by
  sorry

end carol_first_six_probability_l744_744446


namespace sin_B_value_l744_744969

theorem sin_B_value 
  (a b : ℝ) 
  (sinA sinB : ℝ) 
  (h₁ : a = 1) 
  (h₂ : b = real.sqrt 2) 
  (h₃ : sinA = 1 / 3) 
  (h₄ : a / sinA = b / sinB) :
  sinB = real.sqrt 2 / 3 :=
sorry

end sin_B_value_l744_744969


namespace polygon_inequality_l744_744998

noncomputable def perimeter_A {n : ℕ} (h : n ≥ 3) (α : Fin n → ℝ) : ℝ :=
  2 * (Finset.univ.sum (λ i, Real.tan (α i)))

noncomputable def perimeter_B {n : ℕ} (h : n ≥ 3) (α : Fin n → ℝ) : ℝ :=
  2 * (Finset.univ.sum (λ i, Real.sin (α i)))

theorem polygon_inequality 
  {n : ℕ} (h : n ≥ 3) (α : Fin n → ℝ)
  (h_sum : (Finset.univ.sum (λ i, α i)) = Real.pi) :
  let p_A := perimeter_A h α,
      p_B := perimeter_B h α in
  p_A * p_B^2 > 8 * Real.pi^3 :=
by sorry

end polygon_inequality_l744_744998


namespace roots_sum_condition_l744_744639

theorem roots_sum_condition (a b : ℝ) 
  (h1 : ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 9) 
    ∧ (x * y + y * z + x * z = a) ∧ (x * y * z = b)) :
  a + b = 38 := 
sorry

end roots_sum_condition_l744_744639


namespace minimum_value_2x_4y_l744_744129

theorem minimum_value_2x_4y (x y : ℝ) (h : x + 2 * y = 3) : 
  ∃ (min_val : ℝ), min_val = 2 ^ (5/2) ∧ (2 ^ x + 4 ^ y = min_val) :=
by
  sorry

end minimum_value_2x_4y_l744_744129


namespace boys_in_school_l744_744949

-- Definitions for the problem conditions
variables (B G : ℕ) -- Define the number of boys and girls as natural numbers

-- The conditions given in the problem
abbreviation ratio_condition := (13 * B) = (5 * G)
abbreviation difference_condition := G = B + 128

-- The proof goal (combining the question and the correct answer)
theorem boys_in_school : ratio_condition B G ∧ difference_condition B G → B = 80 :=
by
  sorry

end boys_in_school_l744_744949


namespace regular_octagon_arc_length_l744_744801

theorem regular_octagon_arc_length (r : ℝ) (h : r = 4) :
  let C := 2 * Real.pi * r,
      arc_length := (45 / 360) * C
  in arc_length = Real.pi := by
  intros
  suffices : (45 / 360) * 2 * Real.pi * r = Real.pi
  { rw [← this], ring },
  field_simp,
  linarith

end regular_octagon_arc_length_l744_744801


namespace angle_ACP_is_equal_l744_744306

-- Definitions according to the given conditions
def segment_AB_length : ℝ := 6
def AC_length : ℝ := 4
def CB_length : ℝ := 2
def radius1 : ℝ := segment_AB_length / 2
def radius2 : ℝ := AC_length / 2

-- Problem to prove the angle ACP in degrees
theorem angle_ACP_is_equal :
  ∀ (AB AC CB radius1 radius2 : ℝ),
    AB = 6 →
    AC = 4 →
    CB = 2 →
    radius1 = AB / 2 →
    radius2 = AC / 2 →
    ∃ θ : ℝ, θ = 117.3 := 
by
  intros AB AC CB radius1 radius2 hAB hAC hCB hr1 hr2
  use 117.3
  sorry

end angle_ACP_is_equal_l744_744306


namespace cos_double_angle_l744_744580

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l744_744580


namespace solid_cannot_be_triangular_pyramid_if_views_are_same_l744_744602

theorem solid_cannot_be_triangular_pyramid_if_views_are_same :
  (∀ solid, (solid.front_view = solid.side_view) ∧ (solid.side_view = solid.top_view)
   → ¬ (solid = TriangularPyramid)) :=
by
  intros solid h
  sorry

end solid_cannot_be_triangular_pyramid_if_views_are_same_l744_744602


namespace Owen_spending_on_burgers_in_June_l744_744501

theorem Owen_spending_on_burgers_in_June (daily_burgers : ℕ) (cost_per_burger : ℕ) (days_in_June : ℕ) :
  daily_burgers = 2 → 
  cost_per_burger = 12 → 
  days_in_June = 30 → 
  daily_burgers * cost_per_burger * days_in_June = 720 :=
by
  intros
  sorry

end Owen_spending_on_burgers_in_June_l744_744501


namespace problem_subsets_removal_mean_l744_744160

theorem problem_subsets_removal_mean :
  let S := {i | 1 ≤ i ∧ i ≤ 15} in
  let total_sum := 120 in -- Sum of numbers from 1 to 15
  let target_sum := 104 in -- target sum to get mean 8 after removing 2 elements
  let pairs := {(i, j) | i < j ∧ i ∈ S ∧ j ∈ S ∧ i + j = 16} in
  Finset.card pairs = 7 :=
by
  sorry

end problem_subsets_removal_mean_l744_744160


namespace B_shorter_than_A_l744_744409

-- Let's define the problem using Lean 4 notation.

constant students : matrix (Fin 10) (Fin 20) ℝ  -- 200 students in a 10 by 20 matrix of heights
constant A : ℝ  -- shortest among the tallest students in each column
constant B : ℝ  -- tallest among the shortest students in each row

-- Define the conditions
axiom cond1 : ∀ j : Fin 20, ∃ i : Fin 10, students i j ≥ A
axiom cond2 : ∀ i : Fin 10, ∃ j : Fin 20, students i j ≤ B

-- Given proof problem statement
theorem B_shorter_than_A : B < A := by
  sorry

end B_shorter_than_A_l744_744409


namespace cylinder_height_l744_744664

theorem cylinder_height (r : ℝ) (h : ℝ) (n : ℕ) 
  (sphere_radius : ∀ i:ℕ, 0 < i → i ≤ 8 → r = 1 ) 
  (spheres_tangent_four_neighbors : ∀ i : ℕ, 0 < i → i ≤ 8 → ∃ j k l m : ℕ, 0 < j ∧ j ≤ 8 ∧ 0 < k ∧ k ≤ 8 ∧ 
    0 < l ∧ l ≤ 8 ∧ 0 < m ∧ m ≤ 8 ∧ (i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m) ∧ 
    dist (sphere_center i) (sphere_center j) = 2 * r ∧ 
    dist (sphere_center i) (sphere_center k) = 2 * r ∧ 
    dist (sphere_center i) (sphere_center l) = 2 * r ∧ 
    dist (sphere_center i) (sphere_center m) = 2 * r ) 
  (spheres_tangent_base : ∀ i : ℕ, 0 < i → i ≤ 8 → tangent_to_base (sphere_center i) ) 
  (spheres_tangent_side : ∀ i : ℕ, 0 < i → i ≤ 8 → tangent_to_side (sphere_center i) ) : 
  h = (√8)^(1/4) + 2 := 
sorry

end cylinder_height_l744_744664


namespace value_of_x_is_7_l744_744049

-- Define the data set
def data_set : List ℤ := [2, 1, 4, 6] -- x will be added later

-- Define the average condition
def average_condition (x : ℤ) : Prop :=
  (2 + 1 + 4 + x + 6) / 5 = 4

-- The main theorem we want to prove
theorem value_of_x_is_7 : ∃ x, average_condition x ∧ x = 7 :=
by
  use 7
  unfold average_condition
  split
  calc
    (2 + 1 + 4 + 7 + 6) / 5 = 20 / 5 := by norm_num
                     ... = 4       := by norm_num
  sorry

end value_of_x_is_7_l744_744049


namespace solve_for_a_l744_744172

theorem solve_for_a (x y a : ℤ) (h1 : 3 * x + y = 40) (h2 : a * x - y = 20) (h3 : 3 * y^2 = 48) (hx : x ∈ ℤ) (hy : y ∈ ℤ) :
  a = 2 :=
by
  /- Proof goes here -/
  sorry

end solve_for_a_l744_744172


namespace count_nonnegative_balanced_ternary_l744_744159

theorem count_nonnegative_balanced_ternary :
  let f := λ (a : Fin 10 → {-1, 0, 1}), ∑ i : Fin 10, a i * 3^i
  in  ∑ x in Finset.filter (λ x, 0 ≤ f x) (Finset.univ : Finset (Fin 10 → {-1, 0, 1})),
      1 = (3^10 + 1) / 2 :=
by
  sorry

end count_nonnegative_balanced_ternary_l744_744159


namespace ordered_pair_of_ratios_l744_744318

def p_q_ratios (x : ℝ) : Prop := 
  let y_curve := 2 * Real.sin x
  let y_line := 2 * Real.sin (Real.pi / 180 * 80)
  ∀ (n : ℤ), 
  let x1 := 80 * Real.pi / 180 + 360 * Real.pi / 180 * n
  let x2 := 100 * Real.pi / 180 + 360 * Real.pi / 180 * n
  y_curve = y_line ↔ (∃ (p q : ℕ), p < q ∧ Nat.coprime p q ∧ p = 1 ∧ q = 8)

theorem ordered_pair_of_ratios (x : ℝ) (h : p_q_ratios x) : ∃ (p q : ℕ), p = 1 ∧ q = 8 := by 
  sorry

end ordered_pair_of_ratios_l744_744318


namespace tangent_slope_at_1_1_l744_744718

noncomputable def f (x : ℝ) : ℝ := x^2

theorem tangent_slope_at_1_1 : 
  let f_prime := λ x, 2 * x in 
  f_prime 1 = 2 :=
by
  sorry

end tangent_slope_at_1_1_l744_744718


namespace max_S_n_of_arithmetic_seq_l744_744896

theorem max_S_n_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a 1 + n * d)
  (h2 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h3 : a 1 + a 3 + a 5 = 15)
  (h4 : a 2 + a 4 + a 6 = 0) : 
  ∃ n : ℕ, S n = 40 ∧ (∀ m : ℕ, S m ≤ 40) :=
sorry

end max_S_n_of_arithmetic_seq_l744_744896


namespace train_cross_platform_time_l744_744019

variables (train_length platform_length time_cross_tree : ℝ)

def speed (length : ℝ) (time : ℝ) : ℝ := length / time
def time_to_cross_platform (total_length : ℝ) (train_speed : ℝ) : ℝ := total_length / train_speed

theorem train_cross_platform_time :
  ∀ (train_length platform_length time_cross_tree : ℝ), 
    train_length = 1200 → 
    platform_length = 1200 →
    time_cross_tree = 120 →
    time_to_cross_platform (train_length + platform_length) (speed train_length time_cross_tree) = 240 :=
by
  intros
  simp [train_length, platform_length, time_cross_tree, speed, time_to_cross_platform]
  sorry

end train_cross_platform_time_l744_744019


namespace total_animals_peppersprayed_l744_744628

-- Define the conditions
def number_of_raccoons : ℕ := 12
def squirrels_vs_raccoons : ℕ := 6
def number_of_squirrels (raccoons : ℕ) (factor : ℕ) : ℕ := raccoons * factor

-- Define the proof statement
theorem total_animals_peppersprayed : 
  number_of_squirrels number_of_raccoons squirrels_vs_raccoons + number_of_raccoons = 84 :=
by
  -- The proof would go here
  sorry

end total_animals_peppersprayed_l744_744628


namespace patrick_total_money_l744_744660

-- Define the constants and conditions
def bicycle_cost : ℝ := 150
def saved_amount : ℝ := bicycle_cost / 2
def loan_principal_alice : ℝ := 50
def interest_rate_alice : ℝ := 0.05
def loan_period_alice_years : ℝ := 8 / 12
def loan_principal_bob : ℝ := 30
def interest_rate_bob : ℝ := 0.07
def loan_period_bob_years : ℝ := 6 / 12

-- Calculate interests
def interest_alice : ℝ := loan_principal_alice * interest_rate_alice * loan_period_alice_years
def interest_bob : ℝ := loan_principal_bob * interest_rate_bob * loan_period_bob_years

-- Calculate the total money Patrick has
def total_money_patrick : ℝ :=
  saved_amount + loan_principal_alice + interest_alice +
  loan_principal_bob + interest_bob

-- Prove that total_money_patrick = $157.72
theorem patrick_total_money : total_money_patrick = 157.72 := by
  sorry

end patrick_total_money_l744_744660


namespace function_property_l744_744982

-- Define the set of positive integers
def Z_plus := {n : ℕ // n > 0}

-- Define the function f satisfying the condition
theorem function_property (f : Z_plus → Z_plus) :
  (∀ a b : Z_plus, (f a + f b : ℕ) ∣ (a + b : ℕ)^2) →
  (∀ n : Z_plus, f n = n) :=
by
  sorry

end function_property_l744_744982


namespace complex_number_properties_l744_744879

theorem complex_number_properties (z : ℂ) (A B C : ℂ)
  (h₀ : |z| = real.sqrt 2)
  (h₁ : complex.im (z^2) = 2)
  (hA : A = z)
  (hB : B = z^2)
  (hC : C = z - z^2) :
  (z = 1 + complex.i ∨ z = -1 - complex.i) ∧
  ((z = 1 + complex.i → (A+B) • C = -2) ∧ (z = -1 - complex.i → (A+B) • C = 8)) :=
sorry

end complex_number_properties_l744_744879


namespace car_and_bus_speeds_l744_744781

-- Definitions of given conditions
def car_speed : ℕ := 44
def bus_speed : ℕ := 52

-- Definition of total distance after 4 hours
def total_distance (car_speed bus_speed : ℕ) := 4 * car_speed + 4 * bus_speed

-- Definition of fact that cars started from the same point and traveled in opposite directions
def cars_from_same_point (car_speed bus_speed : ℕ) := car_speed + bus_speed

theorem car_and_bus_speeds :
  total_distance car_speed (car_speed + 8) = 384 :=
by
  -- Proof constructed based on the conditions given
  sorry

end car_and_bus_speeds_l744_744781


namespace range_of_m_l744_744912

noncomputable def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m (m: ℝ) (θ : ℝ) (h1 : 0 < θ) (h2 : θ < real.pi / 2) 
(h3 : f (m * real.sin θ) + f (1 - m) > 0) : m ≤ 1 := 
sorry

end range_of_m_l744_744912


namespace willows_in_the_park_l744_744343

theorem willows_in_the_park (W O : ℕ) 
  (h1 : W + O = 83) 
  (h2 : O = W + 11) : 
  W = 36 := 
by 
  sorry

end willows_in_the_park_l744_744343


namespace construct_triangle_l744_744054

def parabola_focus_directrix (F : Point) (d : Line) : Prop :=
sorry -- Define what it means for a point to be on a parabola with given focus and directrix.

def directions_of_sides (v1 v2 v3 : Point) (d1 d2 d3 : Vector) : Prop := 
sorry -- Define what it means for sides to have given directions.

def no_side_perpendicular (d : Line) (s1 s2 s3 : Line) : Prop := 
sorry -- Define what it means for no side to be perpendicular to the directrix.

theorem construct_triangle (F : Point) (d : Line) (d1 d2 d3 : Vector) :
  ∃ (A B C : Point), 
    parabola_focus_directrix A F d ∧ 
    parabola_focus_directrix B F d ∧ 
    parabola_focus_directrix C F d ∧ 
    directions_of_sides A B C d1 d2 d3 ∧ 
    no_side_perpendicular d (line_from_points A B) (line_from_points B C) (line_from_points C A) :=
sorry

end construct_triangle_l744_744054


namespace tory_needs_to_sell_more_packs_l744_744496

theorem tory_needs_to_sell_more_packs 
  (total_goal : ℤ) (packs_grandmother : ℤ) (packs_uncle : ℤ) (packs_neighbor : ℤ) 
  (total_goal_eq : total_goal = 50)
  (packs_grandmother_eq : packs_grandmother = 12)
  (packs_uncle_eq : packs_uncle = 7)
  (packs_neighbor_eq : packs_neighbor = 5) :
  total_goal - (packs_grandmother + packs_uncle + packs_neighbor) = 26 :=
by
  rw [total_goal_eq, packs_grandmother_eq, packs_uncle_eq, packs_neighbor_eq]
  norm_num

end tory_needs_to_sell_more_packs_l744_744496


namespace count_three_digit_values_satisfying_condition_l744_744987

-- Define the sum of digits function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main statement
theorem count_three_digit_values_satisfying_condition : 
  { x : ℕ // 100 ≤ x ∧ x ≤ 999 ∧ digit_sum (digit_sum x) = 4 }.to_finset.card = 48 := by 
  sorry

end count_three_digit_values_satisfying_condition_l744_744987


namespace find_p_l744_744848

theorem find_p (p : ℝ) :
  (∀ x : ℝ, x^2 + p * x + p - 1 = 0) →
  ((exists x1 x2 : ℝ, x^2 + p * x + p - 1 = 0 ∧ x1^2 + x1^3 = - (x2^2 + x2^3) ) → (p = 1 ∨ p = 2)) :=
by
  intro h
  sorry

end find_p_l744_744848


namespace ball_bounce_height_l744_744416

theorem ball_bounce_height (b : ℕ) : 
  ∃ b : ℕ, 400 * (3 / 4 : ℝ)^b < 50 ∧ ∀ b' < b, 400 * (3 / 4 : ℝ)^b' ≥ 50 :=
sorry

end ball_bounce_height_l744_744416


namespace probability_not_hearing_fav_song_l744_744461

-- Define the properties within the given conditions
def song_lengths := (list.range 12).map (λ i, 40 * (i + 1))
def favorite_song_length := 280
def total_songs := 12
def first_six_minutes := 360

-- Define the theorem to prove
theorem probability_not_hearing_fav_song :
  ∃ p : ℚ, p = 11 / 12 ∧ 
  ∀ (songs_order : list ℕ), 
    songs_order.length = total_songs →
    (∀ i, i < total_songs → songs_order.nth i ∈ song_lengths) →
    (∃ k, ((songs_order.takeWhile (λ t, t ≠ favorite_song_length)).sum ≤ first_six_minutes) → 
    ((songs_order.drop k).sum > first_six_minutes) → 
    k ≠ total_songs) :=
sorry

end probability_not_hearing_fav_song_l744_744461


namespace general_term_seq_a_l744_744886

noncomputable def seq_a (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 1
  | (n+2) => (have h : ℕ from sqrt(a n * a (n+2-2)) - sqrt(a (n+2-1) * a (n+2-2)), 
              have h2 : ℕ from 2 * a (n+2-1), 
              if h == h2 then (2^ (n+2-1) - 1)^2 * a (n+2-1) else 0)

theorem general_term_seq_a (n : ℕ) : a n = ∏ k in range n, (2^k - 1)^2 :=
begin
  sorry
end

end general_term_seq_a_l744_744886


namespace unique_function_for_conditions_l744_744089

theorem unique_function_for_conditions
  (f : ℝ → ℤ)
  (h1 : ∀ x y : ℝ, f(x + y) < f(x) + f(y))
  (h2 : ∀ x : ℝ, f(f(x)) = ⌊x⌋ + 2) :
  (∀ x : ℤ, f(x) = x + 1) :=
begin
  sorry
end

end unique_function_for_conditions_l744_744089


namespace min_balls_to_draw_l744_744722

noncomputable def balls (c : String) : ℕ :=
  if c = "red" then 10
  else if c = "yellow" then 10
  else if c = "white" then 10
  else 0

theorem min_balls_to_draw (ball_red ball_yellow ball_white : ℕ) (h_red : ball_red = 10) (h_yellow : ball_yellow = 10) (h_white : ball_white = 10) :
  ∃ (n : ℕ), (∀ (b : String), b ∈ {"red", "yellow", "white"} → (balls b + n) ≥ 5) → n = 13 :=
by
  sorry

end min_balls_to_draw_l744_744722


namespace intersection_Y_distance_to_BC_l744_744970

open Real

noncomputable def distance_Y_to_BC (s : ℝ) : ℝ :=
  let A := (0, 0)
  let B := (s, 0)
  let C := (s, s)
  let D := (0, s)
  let M := (s / 2, 0)
  let N := (s / 2, s)
  let equation₁ := λ (x y : ℝ), (x - s / 2)^2 + y^2 = (s / 2)^2
  let equation₂ := λ (x y : ℝ), (x - s / 2)^2 + (y - s)^2 = (s / 2)^2
  let Y := (s / 2, s / 2)
  abs (s - (Y.1))

theorem intersection_Y_distance_to_BC (s : ℝ) : distance_Y_to_BC s = s / 2 :=
  sorry

end intersection_Y_distance_to_BC_l744_744970


namespace can_divide_2007_triangles_can_divide_2008_triangles_l744_744834

theorem can_divide_2007_triangles :
  ∃ k : ℕ, 2007 = 9 + 3 * k :=
by
  sorry

theorem can_divide_2008_triangles :
  ∃ m : ℕ, 2008 = 4 + 3 * m :=
by
  sorry

end can_divide_2007_triangles_can_divide_2008_triangles_l744_744834


namespace angle_between_vectors_is_120_degrees_l744_744149

-- Defining vectors a and b
def vector_a : ℝ × ℝ := (real.sqrt 3, 1)
def vector_b : ℝ × ℝ := (-2 * real.sqrt 3, 2)

-- Function to calculate the dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Function to calculate the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Prove that the angle between vector a and vector b is 120 degrees
theorem angle_between_vectors_is_120_degrees : 
  let a := vector_a in
  let b := vector_b in
  let cos_theta := dot_product a b / (magnitude a * magnitude b) in
  real.acos cos_theta = real.pi * 2 / 3 :=
by
  sorry

end angle_between_vectors_is_120_degrees_l744_744149


namespace total_papers_calculation_l744_744427

-- Define the original side length and the conditions
variables (n : ℕ)

-- Define the stated equations based on the problem description
def original_total_papers := n^2 + 20
def new_total_papers := (n + 1)^2 - 9

-- The proof statement
theorem total_papers_calculation :
  (original_total_papers n = new_total_papers n) →
  (n = 14) →
  original_total_papers n = 216 :=
by {
  intro h1 h2,
  rw h2 at *,
  unfold original_total_papers new_total_papers at *,
  sorry
}

end total_papers_calculation_l744_744427


namespace geometric_sequence_sixth_term_l744_744336

variable (a r : ℝ) 

theorem geometric_sequence_sixth_term (h1 : a * (1 + r + r^2 + r^3) = 40)
                                    (h2 : a * r^4 = 32) :
  a * r^5 = 1280 / 15 :=
by sorry

end geometric_sequence_sixth_term_l744_744336


namespace age_difference_l744_744715

theorem age_difference (father son : ℕ) (h : father * son = 2015) : abs (father - son) = 34 :=
by
  sorry

end age_difference_l744_744715


namespace expected_number_of_rounds_l744_744412

-- Define the game and its conditions
structure game :=
  (wins_A_odd : ℚ := 3 / 4)  -- Winning probability of Player A in odd rounds
  (wins_B_even : ℚ := 3 / 4) -- Winning probability of Player B in even rounds
  (no_ties : ∀ (n : ℕ), ¬(wins_A_odd = 1/2 ∧ wins_B_even = 1/2)) -- No ties in any round
  (end_condition : ∀ (a_wins b_wins : ℕ), abs (a_wins - b_wins) = 2 → game_terminated)

-- Define the expected number of rounds 
noncomputable def expected_rounds (g : game) : ℚ :=
sorry

-- Expected number of rounds statement
theorem expected_number_of_rounds (g : game) : expected_rounds g = 16 / 3 :=
sorry

end expected_number_of_rounds_l744_744412


namespace flight_duration_NY_to_CT_l744_744835

theorem flight_duration_NY_to_CT :
  let departure_London_to_NY : Nat := 6 -- time in ET on Monday
  let arrival_NY_later_hours : Nat := 18 -- hours after departure
  let arrival_NY : Nat := (departure_London_to_NY + arrival_NY_later_hours) % 24 -- time in ET on Tuesday
  let arrival_CapeTown : Nat := 10 -- time in ET on Tuesday
  let duration_flight_NY_to_CT := (arrival_CapeTown + 24 - arrival_NY) % 24 -- duration calculation
  duration_flight_NY_to_CT = 10 :=
by
  let departure_London_to_NY := 6
  let arrival_NY_later_hours := 18
  let arrival_NY := (departure_London_to_NY + arrival_NY_later_hours) % 24
  let arrival_CapeTown := 10
  let duration_flight_NY_to_CT := (arrival_CapeTown + 24 - arrival_NY) % 24
  show duration_flight_NY_to_CT = 10
  sorry

end flight_duration_NY_to_CT_l744_744835


namespace number_of_clients_l744_744436

theorem number_of_clients (num_cars num_selections_per_car selections_per_client : ℕ)
  (h1 : num_cars = 15)
  (h2 : num_selections_per_car = 3)
  (h3 : selections_per_client = 3)
  (h4 : ∀ car, car < num_cars → ∃ clients, card clients = num_selections_per_car ∧ ∀ c ∈ clients, c < (num_cars * num_selections_per_car / selections_per_client)) :
  (num_cars * num_selections_per_car / selections_per_client) = 15 :=
by
  rw [h1, h2, h3]
  sorry

end number_of_clients_l744_744436


namespace toms_weekly_revenue_l744_744347

def crabs_per_bucket : Nat := 12
def number_of_buckets : Nat := 8
def price_per_crab : Nat := 5
def days_per_week : Nat := 7

theorem toms_weekly_revenue :
  (crabs_per_bucket * number_of_buckets * price_per_crab * days_per_week) = 3360 :=
by
  sorry

end toms_weekly_revenue_l744_744347


namespace count_intersection_ways_l744_744363

theorem count_intersection_ways :
  ∃ (A B C D E F : ℕ),
  A ≠ D ∧ B ≠ E ∧ C ≠ F ∧
  {A, B, C, D, E, F} ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (A-D) ≠ 0 ∧ (B-E) ≠ 0 ∧ (C-F) ≠ 0 ∧
  (A+D) ≠ 0 ∧
  9! / (3! * 3!) = 60480 :=
by
  sorry

end count_intersection_ways_l744_744363


namespace sum_largest_and_smallest_l744_744728

-- Define the three-digit number properties
def hundreds_digit := 4
def tens_digit := 8
def A : ℕ := sorry  -- Placeholder for the digit A

-- Define the number based on the digits
def number (A : ℕ) : ℕ := 100 * hundreds_digit + 10 * tens_digit + A

-- Hypotheses
axiom A_range : 0 ≤ A ∧ A ≤ 9

-- Largest and smallest possible numbers
def largest_number := number 9
def smallest_number := number 0

-- Prove the sum
theorem sum_largest_and_smallest : largest_number + smallest_number = 969 :=
by
  sorry

end sum_largest_and_smallest_l744_744728


namespace find_x_plus_y_l744_744648

noncomputable def det3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

noncomputable def det2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem find_x_plus_y (x y : ℝ) (h1 : x ≠ y)
  (h2 : det3x3 2 5 10 4 x y 4 y x = 0)
  (h3 : det2x2 x y y x = 16) : x + y = 30 := by
  sorry

end find_x_plus_y_l744_744648


namespace smallest_c_inv_l744_744256

def f (x : ℝ) : ℝ := (x + 3)^2 - 7

theorem smallest_c_inv (c : ℝ) : (∀ x1 x2 : ℝ, c ≤ x1 → c ≤ x2 → f x1 = f x2 → x1 = x2) →
  c = -3 :=
sorry

end smallest_c_inv_l744_744256


namespace find_m_value_l744_744913

theorem find_m_value (m : ℝ) (x : ℝ) (y : ℝ) :
  (∀ (x y : ℝ), x + m * y + 3 - 2 * m = 0) →
  (∃ (y : ℝ), x = 0 ∧ y = -1) →
  m = 1 :=
by
  sorry

end find_m_value_l744_744913


namespace movie_theater_open_hours_l744_744795

theorem movie_theater_open_hours (screens movies hours_per_movie : ℕ) 
  (screens_eq : screens = 6) 
  (movies_eq : movies = 24) 
  (hours_per_movie_eq : hours_per_movie = 2) : 
  (movies * hours_per_movie) / screens = 8 := 
by 
  rw [screens_eq, movies_eq, hours_per_movie_eq]
  norm_num
  sorry

end movie_theater_open_hours_l744_744795


namespace train_length_correct_l744_744053

noncomputable def speed_km_hr : ℝ := 6
noncomputable def time_sec : ℝ := 2
noncomputable def conversion_factor : ℝ := 1000 / 3600

noncomputable def speed_m_s : ℝ := speed_km_hr * conversion_factor
noncomputable def length_of_train : ℝ := speed_m_s * time_sec

theorem train_length_correct : length_of_train = 10 / 3 :=
by
  have h_speed_m_s : speed_m_s = 5 / 3 := by 
    simp [speed_m_s, speed_km_hr, conversion_factor]
  simp [length_of_train, h_speed_m_s]
  norm_num

#eval 10 / 3 -- This will evaluate to 3.33 because 10/3 ≈ 3.33

end train_length_correct_l744_744053


namespace repeated_root_condition_l744_744380

theorem repeated_root_condition (m : ℝ) : m = 10 → ∃ x, (5 * x) / (x - 2) + 1 = m / (x - 2) ∧ x = 2 :=
by
  sorry

end repeated_root_condition_l744_744380


namespace number_of_valid_menus_l744_744421

/-- There are 7 days in the week, represented as Sunday (0) to Saturday (6) --/
def days_of_week := Fin 7

/-- Desserts options available: cake, pie, ice cream, pudding, cookies --/
inductive Dessert
| cake
| pie
| ice_cream
| pudding
| cookies

open Dessert

/-- Function indicating the choice of dessert for each day of the week --/
def menu : days_of_week → Dessert

/-- Conditions derived from the problem statement --/
def valid_menu (menu : days_of_week → Dessert) : Prop :=
  -- The dessert each day cannot repeat from the previous day
  (∀ i : Fin 6, menu i ≠ menu (i + 1)) ∧
  -- Pie on Monday (1st day)
  menu ⟨1, by decide⟩ = pie ∧
  -- Cake on Friday (5th day)
  menu ⟨5, by decide⟩ = cake

/-- Prove that given the conditions, the number of valid dessert menus for the week is 1024. --/
theorem number_of_valid_menus : (Finset.univ.filter valid_menu).card = 1024 := sorry

end number_of_valid_menus_l744_744421


namespace eclipse_time_coincide_eclipse_start_time_eclipse_end_time_l744_744453

noncomputable def relative_speed_moon_sun := (17/16 : ℝ) - (1/12 : ℝ)
noncomputable def initial_distance := (47/10 : ℝ)
noncomputable def time_coincide := initial_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_time_coincide : 
  (time_coincide - 12 : ℝ) = (2 + 1/60 : ℝ) :=
sorry

noncomputable def start_distance := (37/10 : ℝ)
noncomputable def time_start := start_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_start_time : 
  (time_start - 12 : ℝ) = (1 + 59/60 : ℝ) :=
sorry

noncomputable def end_distance := (57/10 : ℝ)
noncomputable def time_end := end_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_end_time : 
  (time_end - 12 : ℝ) = (3 + 2/60 : ℝ) :=
sorry

end eclipse_time_coincide_eclipse_start_time_eclipse_end_time_l744_744453


namespace sum_of_x_and_y_l744_744259

theorem sum_of_x_and_y : ∀ x y : ℝ, 
  (x - 1) ^ 3 + 2015 * (x - 1) = -1 ∧ 
  (y - 1) ^ 3 + 2015 * (y - 1) = 1 → 
  x + y = 2 := 
by 
  intros x y h,
  sorry

end sum_of_x_and_y_l744_744259


namespace quadratic_distinct_real_roots_l744_744145

theorem quadratic_distinct_real_roots (k : ℝ) :
  (k > -2 ∧ k ≠ 0) ↔ ( ∃ (a b c : ℝ), a = k ∧ b = -4 ∧ c = -2 ∧ (b^2 - 4 * a * c) > 0) :=
by
  sorry

end quadratic_distinct_real_roots_l744_744145


namespace number_of_divisors_of_n_l744_744252

theorem number_of_divisors_of_n :
  ∃ n : ℕ, (∀ k : ℕ, 0 < k ∧ k < n → ¬ (149^k - 2^k) % (3^3 * 5^5 * 7^7) = 0) 
  ∧ (149^n - 2^n) % (3^3 * 5^5 * 7^7) = 0 
  ∧ (number_of_divisors n = 270) :=
sorry

end number_of_divisors_of_n_l744_744252


namespace find_bc_find_area_l744_744243

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Condition 1: From the given problem
variable (h1 : (b^2 + c^2 - a^2) / (cos A) = 2)

-- Condition 2: From the given problem
variable (h2 : (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1)

theorem find_bc : bc = 1 :=
sorry

theorem find_area (area_abc : ℝ) : area_abc = (sqrt 3) / 4 :=
sorry

end find_bc_find_area_l744_744243


namespace surface_area_of_solid_prism_l744_744439

noncomputable def side_length := 10
noncomputable def height := 20

structure Prism (α : Type) :=
  (A B C D E F : α)

structure Point (α : Type) :=
  (x y z : ℝ)

noncomputable def midpoint (p1 p2 : Point ℝ) : Point ℝ :=
  Point.mk ((p1.x + p2.x) / 2) ((p1.y + p2.y) / 2) ((p1.z + p2.z) / 2)

noncomputable def surface_area_of_solid (L M N : Point ℝ) : ℝ :=
  -- Areas of triangles RNL and RNM
  let area_RNL := 25 in
  let area_RNM := 25 in
  -- Area of triangle RLM
  let area_RLM := 25 * Real.sqrt(3) / 4 in
  -- Area of triangle LMN
  let area_LMN := 5 * Real.sqrt(118.75) / 2 in
  area_RNL + area_RNM + area_RLM + area_LMN

theorem surface_area_of_solid_prism :
  ∀ (P Q R S T U : Point ℝ)
    (L M N : Point ℝ),
  midpoint P R = L →
  midpoint R Q = M →
  midpoint Q T = N →
  surface_area_of_solid L M N = 50 + 25 * Real.sqrt(3) / 4 + 5 * Real.sqrt(118.75) / 2 :=
sorry

end surface_area_of_solid_prism_l744_744439


namespace multiply_polynomials_l744_744265

theorem multiply_polynomials (x : ℝ) : 
  (x^6 + 64 * x^3 + 4096) * (x^3 - 64) = x^9 - 262144 :=
by
  sorry

end multiply_polynomials_l744_744265


namespace smallest_number_divisibility_l744_744374

def is_divisible (a b : ℕ) : Prop := a % b = 0

theorem smallest_number_divisibility :
  ∃ n : ℕ, n = 3153 ∧
           (is_divisible (n - 3) 18) ∧
           (is_divisible (n - 3) 70) ∧
           (is_divisible (n - 3) 25) ∧
           (is_divisible (n - 3) 21) :=
by
  exists 3153
  split
  rfl
  split
  sorry -- proof that 3150 is divisible by 18
  split
  sorry -- proof that 3150 is divisible by 70
  split
  sorry -- proof that 3150 is divisible by 25
  sorry -- proof that 3150 is divisible by 21

end smallest_number_divisibility_l744_744374


namespace closest_to_5_cm_l744_744385

def length_school_bus := 900 -- in cm
def length_school_bus_upper := 1200 -- in cm
def height_picnic_table := 75 -- in cm
def height_picnic_table_upper := 80 -- in cm
def height_elephant := 250 -- in cm
def height_elephant_upper := 400 -- in cm
def length_foot := 24 -- in cm
def length_foot_upper := 30 -- in cm
def length_thumb := 5 -- in cm

theorem closest_to_5_cm : closest_to length_thumb :=
by
  sorry

end closest_to_5_cm_l744_744385


namespace overall_percent_change_l744_744829

theorem overall_percent_change (x : ℝ) :
  let day1_value := 0.9 * x in
  let day2_value := 1.5 * day1_value in
  let day3_value := 0.8 * day2_value in
  (day3_value - x) / x = 0.08 :=
by
  let day1_value := 0.9 * x
  let day2_value := 1.5 * day1_value
  let day3_value := 0.8 * day2_value
  sorry

end overall_percent_change_l744_744829


namespace equation_of_parabola_equation_of_line_l744_744916

-- Definitions and conditions
def parabola (p : ℝ) (h : p > 0) := {pt : ℝ × ℝ // (pt.snd)^2 = 2 * p * pt.fst}
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)
def is_point_on_parabola (pt : ℝ × ℝ) (p : ℝ) : Prop := (pt.snd)^2 = 2 * p * pt.fst
def equidistant_to_focus_directrix (pt : ℝ × ℝ) (focus : ℝ × ℝ) (directrix : ℝ → ℝ) : Prop :=
  dist pt focus = abs (pt.fst - directrix 0)
def y_midpoint (A B : ℝ × ℝ) : ℝ := (A.snd + B.snd) / 2
def line_through_points (m : ℝ) (P Q : ℝ × ℝ) : ℝ × ℝ → Prop := λ (X : ℝ × ℝ), (X.snd - P.snd) = m * (X.fst - P.fst)

-- Part 1: Prove the equation of the parabola
theorem equation_of_parabola :
  ∃ p, ∀ (m : ℝ), ∀ (pt : ℝ × ℝ), is_point_on_parabola pt p →
  dist pt (focus p) = 5 → (pt.fst = 3) → y^2 = 8x :=
sorry

-- Part 2: Prove the equation of the line
theorem equation_of_line :
  ∀ p, ∀ (F : ℝ × ℝ), F = (2, 0) → ∀ (A B : ℝ × ℝ), (is_point_on_parabola A p) →
  (is_point_on_parabola B p) → y_midpoint A B = -1 →
  (line_through_points (-4) F) (4x + y - 8 = 0) :=
sorry

end equation_of_parabola_equation_of_line_l744_744916


namespace intersection_lines_of_three_planes_l744_744961

theorem intersection_lines_of_three_planes (P1 P2 P3 : Plane) 
  (h12 : ∃ l1 : Line, ∀ x : Point, x ∈ l1 ↔ (x ∈ P1 ∧ x ∈ P2))
  (h13 : ∃ l2 : Line, ∀ x : Point, x ∈ l2 ↔ (x ∈ P1 ∧ x ∈ P3))
  (h23 : ∃ l3 : Line, ∀ x : Point, x ∈ l3 ↔ (x ∈ P2 ∧ x ∈ P3)) :
  ∃! n : ℕ, n = 3 := 
sorry

end intersection_lines_of_three_planes_l744_744961


namespace solve_inequality_system_simplify_expression_l744_744015

-- Part 1: System of Inequalities

theorem solve_inequality_system : 
  ∀ (x : ℝ), (x + 2) / 5 < 1 ∧ 3 * x - 1 ≥ 2 * x → 1 ≤ x ∧ x < 3 :=  by
  sorry

-- Part 2: Expression Simplification

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) : 
  (m - 1 / m) * ((m^2 - m) / (m^2 - 2 * m + 1)) = m + 1 :=
  by
  sorry

end solve_inequality_system_simplify_expression_l744_744015


namespace lines_intersect_at_l744_744792

def line1 (s : ℝ) : ℝ × ℝ :=
  (2 + 3 * s, 3 + 4 * s)

def line2 (v : ℝ) : ℝ × ℝ :=
  (-1 + v, 2 - v)

theorem lines_intersect_at :
  ∃ s v : ℝ, line1 s = line2 v ∧ line1 s = (-2/7 : ℝ, 5/7 : ℝ) := sorry

end lines_intersect_at_l744_744792


namespace average_output_is_approx_45_59_l744_744060

def assembly_line_production (initialOrder nextOrder remainingOrder : ℝ) 
(rate1 rate2 rate3 : ℝ) : ℝ :=
  let time1 := initialOrder / rate1
  let time2 := nextOrder / rate2
  let time3 := remainingOrder / rate3
  let totalTime := time1 + time2 + time3
  let totalCogs := initialOrder + nextOrder + remainingOrder
  totalCogs / totalTime

theorem average_output_is_approx_45_59 :
  assembly_line_production 100 100 100 30 50 80 ≈ 45.59 :=
by
  sorry

end average_output_is_approx_45_59_l744_744060


namespace relationship_among_a_b_c_l744_744778

-- Definitions of a, b, and c as given in the problem conditions.
def a : ℝ := 2^0.1
def b : ℝ := Real.log10 (5 / 2)
def c : ℝ := Real.logBase 3 (9 / 10)

-- Statement of the theorem to be proven.
theorem relationship_among_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_among_a_b_c_l744_744778


namespace solve_for_a_l744_744465

def g (x : ℝ) : ℝ := 5 * x - 6

theorem solve_for_a (a : ℝ) : g a = 4 → a = 2 := by
  sorry

end solve_for_a_l744_744465


namespace no_infinite_prime_sequence_l744_744851

theorem no_infinite_prime_sequence (p : ℕ) (h_prime : Nat.Prime p) :
  ¬(∃ (p_seq : ℕ → ℕ), (∀ n, Nat.Prime (p_seq n)) ∧ (∀ n, p_seq (n + 1) = 2 * p_seq n + 1)) :=
by
  sorry

end no_infinite_prime_sequence_l744_744851


namespace cos_double_angle_example_l744_744575

def cos_double_angle_identity (x : ℝ) : Prop :=
  cos (2 * x) = 1 - 2 * (sin x) ^ 2

theorem cos_double_angle_example : cos_double_angle_identity (x : ℝ) 
  (h : sin x = - 2 / 3) : cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_example_l744_744575


namespace range_of_x_in_function_l744_744617

theorem range_of_x_in_function :
  ∀ x : ℝ, (∃ y : ℝ, y = (sqrt (1 - 2 * x)) / x) →
  x ≤ 1 / 2 ∧ x ≠ 0 :=
by
  intros x hx
  split
  -- First condition: x ≤ 1 / 2
  -- Second condition: x ≠ 0
  sorry

end range_of_x_in_function_l744_744617


namespace initial_water_percentage_88_l744_744020

-- Definitions based on conditions
variables (initial_volume : ℕ) (added_sugar added_water added_kola : ℝ)
variable (final_sugar_percentage : ℝ)

-- Given initial conditions
def initial_kola_percentage : ℝ := 5 / 100
def initial_volume_liters : ℝ := 340
def final_volume_liters : ℝ := 340 + 3.2 + 10 + 6.8
def sugar_of_new_solution : ℝ := 0.075 * (initial_volume + 3.2 + 10 + 6.8)
def added_sugar_liters : ℝ := 3.2

-- Calculate initial sugar percentage based on given equation
def initial_sugar_percentage (s : ℝ) : Prop :=
  (s / 100) * initial_volume + added_sugar = sugar_of_new_solution

-- Use initial_sugar_percentage to calculate the initial water percentage
def initial_water_percentage (s w : ℝ) : Prop :=
  s + w = 95

-- Statement of the problem in Lean 
theorem initial_water_percentage_88 (w : ℝ) (s : ℝ)
  (h1 : initial_sugar_percentage s) 
  (h2 : initial_water_percentage s w) : 
  w = 88 :=
sorry

end initial_water_percentage_88_l744_744020


namespace more_knights_than_liars_l744_744293

-- Define the context of the problem: knights and liars on an island.
def islander : Type := {x : Bool // x = true ∨ x = false} -- True for knights, False for liars

-- Define the number of each type of islander
def number_of_knights (n : islander → Bool) : Nat :=
  (Finset.filter (fun i => n i = true) Finset.univ).card

def number_of_liars (n : islander → Bool) : Nat :=
  (Finset.filter (fun i => n i = false) Finset.univ).card

-- Define the friendship relation
def friends_with (n : islander → Bool) (i j : islander) : Bool :=
  if n i = n j then true else arbitrary Bool -- Arbitrary logic for heterogenous friendships

-- Define the statement made by every islander
def statement (n : islander → Bool) (i : islander) : Prop :=
  let friends := (Finset.filter (fun j => friends_with n i j) Finset.univ).card
  if n i = true then
    (Finset.filter (fun j => n j = true && friends_with n i j) Finset.univ).card > friends / 2
  else
    (Finset.filter (fun j => n j = false && friends_with n i j) Finset.univ).card > friends / 2

-- Define the theorem to prove more knights exist than liars
theorem more_knights_than_liars (n : islander → Bool) :
  (∀ i, statement n i) →
  number_of_knights n > number_of_liars n :=
  sorry

end more_knights_than_liars_l744_744293


namespace minimum_solutions_in_interval_l744_744775

open Function Real

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define what it means for a function to be periodic
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

-- Main theorem statement
theorem minimum_solutions_in_interval :
  ∀ (f : ℝ → ℝ),
  is_even f → is_periodic f 3 → f 2 = 0 →
  (∃ x1 x2 x3 x4 : ℝ, 0 < x1 ∧ x1 < 6 ∧ f x1 = 0 ∧
                     0 < x2 ∧ x2 < 6 ∧ f x2 = 0 ∧
                     0 < x3 ∧ x3 < 6 ∧ f x3 = 0 ∧
                     0 < x4 ∧ x4 < 6 ∧ f x4 = 0 ∧
                     x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧
                     x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :=
by
  sorry

end minimum_solutions_in_interval_l744_744775


namespace area_of_PQRST_l744_744448

-- Definitions for geometrical structurings
variables {Point : Type} [Geometry Point]
variables (A B C D E S R T P Q : Point)

-- Definition of areas being equal to 1
def area_equality : Prop :=
  (area A S R = 1) ∧ (area B T T = 1) ∧ (area C D T = 1) ∧ (area D Q P = 1) ∧ (area E R Q = 1)

-- Definition of the convex pentagon and its intersections
def convex_pentagon_intersections : Prop :=
  on_same_line [A, C, S, R] ∧ on_same_line [A, D, R] ∧ 
  on_same_line [C, A, T, P] ∧ on_same_line [C, E, P] ∧ 
  on_same_line [C, E, Q] ∧ on_same_line [A, D, Q]

-- The mathematical proof problem to be translated in Lean 4
theorem area_of_PQRST (h1 : area_equality A B C D E S R T P Q)
                      (h2 : convex_pentagon_intersections A B C D E S R T P Q) :
  area P Q R S T = sqrt(5) :=
sorry

end area_of_PQRST_l744_744448


namespace car_rental_cost_eq_800_l744_744656

-- Define the number of people
def num_people : ℕ := 8

-- Define the cost of the Airbnb rental
def airbnb_cost : ℕ := 3200

-- Define each person's share
def share_per_person : ℕ := 500

-- Define the total contribution of all people
def total_contribution : ℕ := num_people * share_per_person

-- Define the car rental cost
def car_rental_cost : ℕ := total_contribution - airbnb_cost

-- State the theorem to be proved
theorem car_rental_cost_eq_800 : car_rental_cost = 800 :=
  by sorry

end car_rental_cost_eq_800_l744_744656


namespace quadrilateral_cyclic_l744_744222

-- Definitions and conditions
variable {Γ : Type} [circle Γ] (B C : point Γ)
variable (A : point Γ) (MidpointArcBC : midpoint_arc Γ B C A)
variable (D E : point Γ) (AD AE : chord Γ A D) (chord_AE : chord Γ A E)
variable (F G : point Γ) (intersect_AD_BC : intersection_point AD B C F) (intersect_AE_BC : intersection_point AE B C G)

-- Theorem statement
theorem quadrilateral_cyclic : cyclic_quadrilateral Γ D F G E :=
sorry

end quadrilateral_cyclic_l744_744222


namespace exists_parallel_intersecting_line_l744_744509

noncomputable def construct_parallel_line
  (Γ : Type) [metric_space Γ]
  (D : Type) [line D]
  (a : ℝ) : Prop :=
∀ (circle Γ : circle) (line D : line) (a : ℝ),
  ∃ Δ : line, (parallel Δ D) ∧
              ∃ P Q : Γ, (intersects Δ P) ∧ (intersects Δ Q) ∧ (dist P Q = a)

theorem exists_parallel_intersecting_line
  (Γ : Type) [metric_space Γ]
  (D : Type) [line D]
  (a : ℝ) :
  construct_parallel_line Γ D a :=
sorry

end exists_parallel_intersecting_line_l744_744509


namespace road_unrepaired_is_42_percent_statement_is_false_l744_744779

def road_length : ℝ := 1
def phase1_completion : ℝ := 0.40
def phase2_remaining_factor : ℝ := 0.30

def remaining_road (road : ℝ) (phase1 : ℝ) (phase2_factor : ℝ) : ℝ :=
  road - phase1 - (road - phase1) * phase2_factor

theorem road_unrepaired_is_42_percent (road_length : ℝ) (phase1_completion : ℝ) (phase2_remaining_factor : ℝ) :
  remaining_road road_length phase1_completion phase2_remaining_factor = 0.42 :=
by
  sorry

theorem statement_is_false : ¬(remaining_road road_length phase1_completion phase2_remaining_factor = 0.30) :=
by
  sorry

end road_unrepaired_is_42_percent_statement_is_false_l744_744779


namespace sum_sequence_l744_744206

theorem sum_sequence (a b : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = 4)
  (h3 : ∀ n, 2 * a (n + 1) = a n + a (n + 2))
  (h4 : b 1 = 2)
  (h5 : b 2 = 4)
  (h6 : ∀ n, b (n + 1) - b n < 2^n + 1/2)
  (h7 : ∀ n, b (n + 2) - b n > 3 * 2^n - 1)
  (h8 : ∀ n, b n ∈ ℤ) :
  (∑ k in finset.range n, n * b n / a n) = 2^n - 1 := by
  sorry

end sum_sequence_l744_744206


namespace floor_of_root_l744_744525

def floor_function (x : ℝ) : ℤ := Int.floor x

def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem floor_of_root :
  ∃ x₀, f x₀ = 0 ∧ floor_function x₀ = 2 :=
by
  sorry

end floor_of_root_l744_744525


namespace sound_pressure_proof_l744_744268

noncomputable theory

def sound_pressure_level (p p0 : ℝ) : ℝ :=
  20 * real.log10 (p / p0)

variables (p0 : ℝ) (p0_pos : 0 < p0)
variables (p1 p2 p3 : ℝ)

def gasoline_car (Lp : ℝ) : Prop :=
  60 <= Lp ∧ Lp <= 90

def hybrid_car (Lp : ℝ) : Prop :=
  50 <= Lp ∧ Lp <= 60

def electric_car (Lp : ℝ) : Prop :=
  Lp = 40

theorem sound_pressure_proof :
  gasoline_car (sound_pressure_level p1 p0) ∧
  hybrid_car (sound_pressure_level p2 p0) ∧
  electric_car (sound_pressure_level p3 p0) →
  (p1 ≥ p2) ∧ (¬ (p2 > 10 * p3)) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) :=
begin
  sorry
end

end sound_pressure_proof_l744_744268


namespace increasing_interval_l744_744541

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * x^2 + 2 * a * x - Real.log x

def f_prime (a : ℝ) (x : ℝ) : ℝ := x + 2 * a - (1 / x)

theorem increasing_interval (a : ℝ) :
  (∀ x ∈ set.Icc (1/3 : ℝ) 2, f_prime a x ≥ 0) ↔ a ≥ 4 / 3 :=
by
  sorry

end increasing_interval_l744_744541


namespace f_at_one_f_increasing_f_range_for_ineq_l744_744881

-- Define the function f with its properties
noncomputable def f : ℝ → ℝ := sorry

-- Properties of f
axiom f_domain : ∀ x, 0 < x → f x ≠ 0 
axiom f_property_additive : ∀ x y, f (x * y) = f x + f y
axiom f_property_positive : ∀ x, (1 < x) → (0 < f x)
axiom f_property_fract : f (1/3) = -1

-- Proofs to be completed
theorem f_at_one : f 1 = 0 :=
sorry

theorem f_increasing : ∀ (x₁ x₂ : ℝ), (0 < x₁) → (0 < x₂) → (x₁ < x₂) → (f x₁ < f x₂) :=
sorry

theorem f_range_for_ineq : {x : ℝ | 2 < x ∧ x ≤ 9/4} = {x : ℝ | f x - f (x - 2) ≥ 2} :=
sorry

end f_at_one_f_increasing_f_range_for_ineq_l744_744881


namespace prime_divides_sum_of_squares_l744_744766

theorem prime_divides_sum_of_squares (p a b : ℤ) (hp : p.prime) (hmod : p % 4 = 3)
  (hdiv : p ∣ (a^2 + b^2)) : p ∣ a ∧ p ∣ b :=
sorry

end prime_divides_sum_of_squares_l744_744766


namespace car_mpg_difference_l744_744024

theorem car_mpg_difference (h_miles : Int) (c_miles : Int) (c_mpg : Int) : Int :=
  let tank_size := c_miles / c_mpg
  let h_mpg := h_miles / tank_size
  h_mpg - c_mpg
by 
  have h_miles := 462
  have c_miles := 336
  have c_mpg := 8
  show car_mpg_difference h_miles c_miles c_mpg = 3 from sorry

end car_mpg_difference_l744_744024


namespace g_five_eq_thirteen_sevenths_l744_744170

def g (x : ℚ) : ℚ := (3 * x - 2) / (x + 2)

theorem g_five_eq_thirteen_sevenths : g 5 = 13 / 7 := by
  sorry

end g_five_eq_thirteen_sevenths_l744_744170


namespace kite_diagonals_pass_through_center_l744_744030

noncomputable theory
open EuclideanGeometry
open_locale real

-- Lean statement for the problem
theorem kite_diagonals_pass_through_center (A B C D O : Point) (k : is_kite A B C D)
  (hO : incircle O A B C D) :
  let E := intersection (line_perp_at O A) (line_perp_at O B),
      F := intersection (line_perp_at O B) (line_perp_at O C),
      G := intersection (line_perp_at O C) (line_perp_at O D),
      H := intersection (line_perp_at O D) (line_perp_at O A) in
  collinear {E, O, G} ∧ collinear {F, O, H} :=
sorry

end kite_diagonals_pass_through_center_l744_744030


namespace total_time_taken_l744_744025

theorem total_time_taken (b km : ℝ) : 
  (b / 50 + km / 80) = (8 * b + 5 * km) / 400 := 
sorry

end total_time_taken_l744_744025


namespace days_after_sunday_marie_birthday_l744_744655

theorem days_after_sunday (n : ℕ) : (n % 7 = 125 % 7) → (n = 6) :=
by
  sorry

theorem marie_birthday : (let start_day := 0 in let days := 125 in (days + start_day) % 7 = 6) :=
by
  sorry

end days_after_sunday_marie_birthday_l744_744655


namespace calculate_s_at_2_l744_744995

-- Given definitions
def t (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 1
def s (p : ℝ) : ℝ := p^3 - 4 * p^2 + p + 6

-- The target statement
theorem calculate_s_at_2 : s 2 = ((5 + Real.sqrt 33) / 4)^3 - 4 * ((5 + Real.sqrt 33) / 4)^2 + ((5 + Real.sqrt 33) / 4) + 6 := 
by 
  sorry

end calculate_s_at_2_l744_744995


namespace weight_of_new_student_l744_744398

theorem weight_of_new_student (avg_decrease_per_student : ℝ) (num_students : ℕ) (weight_replaced_student : ℝ) (total_reduction : ℝ) 
    (h1 : avg_decrease_per_student = 5) (h2 : num_students = 8) (h3 : weight_replaced_student = 86) (h4 : total_reduction = num_students * avg_decrease_per_student) :
    ∃ (x : ℝ), x = weight_replaced_student - total_reduction ∧ x = 46 :=
by
  use 46
  simp [h1, h2, h3, h4]
  sorry

end weight_of_new_student_l744_744398


namespace min_value_x2_y2_z2_l744_744878

noncomputable def min_value (x y z : ℝ) (h : x - 2*y - 3*z = 4) : ℝ :=
 min (x^2 + y^2 + z^2) sorry

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x - 2*y - 3*z = 4) :
  min_value x y z h = (8:ℝ) / (7:ℝ) :=
begin
  sorry
end

end min_value_x2_y2_z2_l744_744878


namespace general_term_of_sequence_l744_744885

theorem general_term_of_sequence (a : ℕ → ℝ) 
  (h0 : a 0 = 1) 
  (h1 : a 1 = 1) 
  (hn : ∀ n ≥ 2, Real.sqrt (a n * a (n-2)) - Real.sqrt (a (n-1) * a (n-2)) = 2 * a (n-1)) :
  ∀ n : ℕ, n ≥ 2 → a n = ∏ k in finset.range(n+1).filter (λ k, k ≥ 1), (2 * k - 1) ^ 2 :=
by
  sorry

end general_term_of_sequence_l744_744885


namespace square_side_length_range_l744_744178

theorem square_side_length_range (a : ℝ) (h : a^2 = 30) : 5.4 < a ∧ a < 5.5 :=
sorry

end square_side_length_range_l744_744178


namespace equation_correct_l744_744557

variable (x y : ℝ)

-- Define the conditions
def condition1 : Prop := (x + y) / 3 = 1.888888888888889
def condition2 : Prop := 2 * x + y = 7

-- Prove the required equation under given conditions
theorem equation_correct : condition1 x y → condition2 x y → (x + y) = 5.666666666666667 := by
  intros _ _
  sorry

end equation_correct_l744_744557


namespace shortest_chord_length_l744_744117

theorem shortest_chord_length 
  (C : ℝ → ℝ → Prop) 
  (l : ℝ → ℝ → ℝ → Prop) 
  (radius : ℝ) 
  (center_x center_y : ℝ) 
  (cx cy : ℝ) 
  (m : ℝ) :
  (∀ x y, C x y ↔ (x - 1)^2 + (y - 2)^2 = 25) →
  (∀ x y m, l x y m ↔ (2*m+1)*x + (m+1)*y - 7*m - 4 = 0) →
  center_x = 1 →
  center_y = 2 →
  radius = 5 →
  cx = 3 →
  cy = 1 →
  ∃ shortest_chord_length : ℝ, shortest_chord_length = 4 * Real.sqrt 5 := sorry

end shortest_chord_length_l744_744117


namespace geometric_sequence_general_formula_l744_744126

noncomputable def a_n (n : ℕ) : ℝ := 2^n

theorem geometric_sequence_general_formula :
  (∀ n : ℕ, 2 * (a_n n + a_n (n + 2)) = 5 * a_n (n + 1)) →
  (a_n 5 ^ 2 = a_n 10) →
  ∀ n : ℕ, a_n n = 2 ^ n := 
by 
  sorry

end geometric_sequence_general_formula_l744_744126


namespace volume_of_prism_in_cubic_feet_l744_744928

theorem volume_of_prism_in_cubic_feet:
  let length_yd := 1
  let width_yd := 2
  let height_yd := 3
  let yard_to_feet := 3
  let length_ft := length_yd * yard_to_feet
  let width_ft := width_yd * yard_to_feet
  let height_ft := height_yd * yard_to_feet
  let volume := length_ft * width_ft * height_ft
  volume = 162 := by
  sorry

end volume_of_prism_in_cubic_feet_l744_744928


namespace Debby_eats_l744_744468

theorem Debby_eats (candy_initial : ℕ) (friend_gave : ℕ) (eat_fraction : ℚ) (share_percent : ℚ) : 
  candy_initial = 12 ∧ friend_gave = 5 ∧ eat_fraction = 2/3 ∧ share_percent = 25/100 →
  (let total_candy := candy_initial + friend_gave in
   let candy_eaten := floor (eat_fraction * total_candy) in
   candy_eaten = 11) :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4⟩
  let total_candy := h1 + h2
  let candy_eaten := floor (h3 * total_candy)
  have : candy_eaten = 11 := sorry
  exact this

end Debby_eats_l744_744468


namespace angle_BCD_is_36_degrees_l744_744606

theorem angle_BCD_is_36_degrees
  (E B D C A : Type)
  (diameter_EB : ∀ t : Type, t ∥ C)
  (angles_ratio : ∀ x : ℝ, double_angle x 2 / triple_angle x 3 = 2/3)
  (angle_AEB : ∀ t : angle_90 t)
  :
  ∠ B C D = 36 :=
  
begin
  sorry
end

end angle_BCD_is_36_degrees_l744_744606


namespace residual_for_individual_l744_744314

theorem residual_for_individual 
  (x_i y_i : ℝ) (b a : ℝ) 
  (reg_eq : b = 0.85 ∧ a = -85.71) 
  (data: x_i = 170 ∧ y_i = 58) : 
  let e := y_i - (b * x_i + a) in 
  e = -0.79 := by
  sorry

end residual_for_individual_l744_744314


namespace necessary_but_not_sufficient_for_q_l744_744523

variable (x : ℝ) (m : ℝ)

def p : Prop := (x + 2 ≥ 10) ∧ (x - 10 ≤ 0)
def q : Prop := (-m ≤ x ∧ x ≤ 1 + m)

theorem necessary_but_not_sufficient_for_q (h : ¬p → ¬q) : m ≥ 9 :=
by
  sorry

end necessary_but_not_sufficient_for_q_l744_744523


namespace probability_draw_eq_l744_744755

variable (P : String → Prop)

variable (P_A_win P_A_not_lose P_A_draw : Prop)

-- Given conditions
axiom P_A_win_value : P_A_win ↔ (0.3 : ℝ)
axiom P_A_not_lose_value : P_A_not_lose ↔ (0.8 : ℝ)

-- Proof problem statement
theorem probability_draw_eq :
  P_A_draw ↔ (0.5 : ℝ) :=
by
  sorry

end probability_draw_eq_l744_744755


namespace square_area_l744_744051

theorem square_area (x : ℝ) (s : ℝ) 
  (h1 : s^2 + s^2 = (2 * x)^2) 
  (h2 : 4 * s = 16 * x) : s^2 = 16 * x^2 :=
by {
  sorry -- Proof not required
}

end square_area_l744_744051


namespace cos_double_angle_example_l744_744574

def cos_double_angle_identity (x : ℝ) : Prop :=
  cos (2 * x) = 1 - 2 * (sin x) ^ 2

theorem cos_double_angle_example : cos_double_angle_identity (x : ℝ) 
  (h : sin x = - 2 / 3) : cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_example_l744_744574


namespace sum_of_integers_with_given_product_and_difference_l744_744716

theorem sum_of_integers_with_given_product_and_difference :
  ∃ (a b : ℕ), a * b = 18 ∧ a ≠ b ∧ (a > b ∨ b > a) ∧ abs (a - b) = 3 ∧ a + b = 9 :=
begin
  sorry
end

end sum_of_integers_with_given_product_and_difference_l744_744716


namespace chosen_number_is_reconstructed_l744_744403

theorem chosen_number_is_reconstructed (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 26) :
  ∃ (a0 a1 a2 : ℤ), (a0 = 0 ∨ a0 = 1 ∨ a0 = 2) ∧ 
                     (a1 = 0 ∨ a1 = 1 ∨ a1 = 2) ∧ 
                     (a2 = 0 ∨ a2 = 1 ∨ a2 = 2) ∧ 
                     n = a0 * 3^0 + a1 * 3^1 + a2 * 3^2 ∧ 
                     n = (if a0 = 1 then 1 else 0) + (if a0 = 2 then 2 else 0) +
                         (if a1 = 1 then 3 else 0) + (if a1 = 2 then 6 else 0) +
                         (if a2 = 1 then 9 else 0) + (if a2 = 2 then 18 else 0) := 
sorry

end chosen_number_is_reconstructed_l744_744403


namespace journey_time_l744_744430

-- Define the conditions
def total_distance : ℝ := 225
def first_half_distance : ℝ := total_distance / 2
def second_half_distance : ℝ := total_distance / 2
def first_speed : ℝ := 21
def second_speed : ℝ := 24

-- Prove that the total time taken for the journey is approximately 10.0445 hours
theorem journey_time (T : ℝ) (h : T = first_half_distance / first_speed + second_half_distance / second_speed) : T ≈ 10.0445 := 
by 
  sorry

end journey_time_l744_744430


namespace part1_part2_l744_744880

noncomputable def z (m : ℝ) : ℂ := 1 + m * complex.I
noncomputable def condition1 (m : ℝ) : Prop := (z m + 2) / (1 - complex.I) ∈ ℝ

theorem part1 (m : ℝ) (h : condition1 m) : m = -3 := sorry

noncomputable def z0 (m : ℝ) : ℂ := -3 - complex.I + z m
noncomputable def is_root (p : ℂ → Prop) (r : ℂ) := p r = 0

theorem part2 {b c : ℝ} (m : ℝ) (h : m = -3) (h1 : is_root (λ x => x^2 + b * x + c) (z0 m)) :
  b = 4 ∧ c = 20 := sorry

end part1_part2_l744_744880


namespace check_random_event_l744_744758

def random_event (A B C D : Prop) : Prop := ∃ E, D = E

def event_A : Prop :=
  ∀ (probability : ℝ), probability = 0

def event_B : Prop :=
  ∀ (probability : ℝ), probability = 0

def event_C : Prop :=
  ∀ (probability : ℝ), probability = 1

def event_D : Prop :=
  ∀ (probability : ℝ), 0 < probability ∧ probability < 1

theorem check_random_event :
  random_event event_A event_B event_C event_D :=
sorry

end check_random_event_l744_744758


namespace prove_relationships_l744_744286

variables {p0 p1 p2 p3 : ℝ}
variable (h_p0_pos : p0 > 0)
variable (h_gas_car : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
variable (h_hybrid_car : 10^(5 / 2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)
variable (h_electric_car : p3 = 100 * p0)

theorem prove_relationships : 
  (p1 ≥ p2) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) := by
  split
  sorry

end prove_relationships_l744_744286


namespace determine_c_l744_744909

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem determine_c (a b : ℝ) (m c : ℝ) 
  (h1 : ∀ x, 0 ≤ x → f x a b = x^2 + a * x + b)
  (h2 : ∃ m : ℝ, ∀ x : ℝ, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
sorry

end determine_c_l744_744909


namespace seat_arrangements_not_adjacent_l744_744194

theorem seat_arrangements_not_adjacent (n : ℕ) (Anne John : ℕ) : 
  n = 8 ∧ Anne ≠ John →
  (factorial 8 - (factorial 7 * 2)) = 30240 :=
by
  intro h
  sorry

end seat_arrangements_not_adjacent_l744_744194


namespace max_inner_product_l744_744562

variables {a b e : ℝ → ℝ} -- defining a, b, and e as vector functions

-- conditions: vector norms and inequality
axiom a_norm : ∥a∥ = 1
axiom b_norm : ∥b∥ = 2
axiom unit_vector (e : ℝ → ℝ) : ∥e∥ = 1
axiom inequality (e : ℝ → ℝ) : ∥a • e∥ + ∥b • e∥ ≤ sqrt 6

-- theorem to prove the maximum value of inner product
theorem max_inner_product : (a • b) ≤ 1 / 2 :=
sorry

end max_inner_product_l744_744562


namespace find_x_l744_744140

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then |x - 1| else 3^x

theorem find_x (x : ℝ) : f x = 3 ↔ x = -2 :=
by
  sorry

end find_x_l744_744140


namespace S_12_l744_744201

variable {S : ℕ → ℕ}

-- Given conditions
axiom S_4 : S 4 = 4
axiom S_8 : S 8 = 12

-- Goal: Prove S_12
theorem S_12 : S 12 = 24 :=
by
  sorry

end S_12_l744_744201


namespace divide_into_arithmetic_progressions_l744_744833

def is_nice_triple (a b c : ℝ) : Prop :=
  a = (b + c) / 2 ∨ b = (a + c) / 2 ∨ c = (a + b) / 2

theorem divide_into_arithmetic_progressions (k : ℕ) (a : ℝ → ℝ)
  (h_len : ∀ n m : ℤ, n ≠ m → a n ≠ a m)
  (h_nice : ∃ l : list (ℤ × ℤ × ℤ), l.length = k^2 ∧ (∀ (i j m : ℤ), (i, j, m) ∈ l → is_nice_triple (a i) (a j) (a m))) :
  ∃ (b₁ b₂ r₁ r₂ : ℝ), r₁ = r₂ ∧ 
  (∀ i : ℤ, (i < 0 → a i = b₁ + i * r₁) ∧ (i ≥ 0 → a i = b₂ + i * r₂)) :=
sorry

end divide_into_arithmetic_progressions_l744_744833


namespace max_value_is_one_l744_744080

noncomputable def max_value_fraction (x : ℝ) : ℝ :=
  (1 + Real.cos x) / (Real.sin x + Real.cos x + 2)

theorem max_value_is_one : ∃ x : ℝ, max_value_fraction x = 1 := by
  sorry

end max_value_is_one_l744_744080


namespace perimeter_of_triangle_is_eight_l744_744899

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci
def focus1 : ℝ × ℝ := (±2, 0)  -- Assuming standard position of foci for simplicity
def focus2 : ℝ × ℝ := (-2, 0)  -- The other focus

-- Define the chord
def chord (A B : ℝ × ℝ) : Prop := 
  A.1 + A.2 = focus1.1 + focus1.2 ∧
  B.1 + B.2 = focus1.1 + focus1.2 

-- Define the points A and B on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define the perimeter of the triangle ABF_2
def perimeter_triangle (A B F2 : ℝ × ℝ) : ℝ := 
  (real.sqrt ((A.1 - F2.1)^2 + (A.2 - F2.2)^2) + 
   real.sqrt ((B.1 - F2.1)^2 + (B.2 - F2.2)^2) + 
   real.sqrt ((F1.1 - A.1)^2 + (F1.2 - A.2)^2) + 
   real.sqrt ((F1.1 - B.1)^2 + (F1.2 - B.2)^2))

theorem perimeter_of_triangle_is_eight 
  (A B : ℝ × ℝ) 
  (hA : point_on_ellipse A)
  (hB : point_on_ellipse B)
  (hChord : chord A B) : 
  perimeter_triangle A B focus2 = 8 := sorry

end perimeter_of_triangle_is_eight_l744_744899


namespace length_of_chord_AB_equation_of_circle_c2_line_l1_intersects_and_shortest_chord_l744_744508

open Real

noncomputable def circle_c1 := (x y : ℝ) → x^2 + y^2 - 2*x - 4*y + 4 = 0
noncomputable def line_l := (x y : ℝ) → x + 2*y - 4 = 0
noncomputable def circle_c2 := (x y : ℝ) → ∃ (a b c : ℝ), x^2 + y^2 + a*x + b*y + c = 0

theorem length_of_chord_AB :
  (∃ (x y : ℝ), circle_c1 x y ∧ line_l x y) →
  ∃ (length : ℝ), length = (4 * sqrt 5) / 5 :=
sorry

theorem equation_of_circle_c2 :
  (∃ (E F : ℝ × ℝ), E = (1, -3) ∧ F = (0, 4) ∧
    (∀ (P : ℝ × ℝ), ∃ (x y : ℝ), (P = (x, y) ∧ circle_c1 x y ∧ line_l x y)) ∧
    ∃ (line : ℝ → ℝ → Prop), 
      (∀ (x y : ℝ), (2*x + y + 1 = 0) → circle_c1 x y ∧ line x y)) →
  (∃ (a b c : ℝ), circle_c2 a b c ∧ a = 6 ∧ b = 0 ∧ c = -16) :=
sorry

theorem line_l1_intersects_and_shortest_chord :
  (∀ (λ : ℝ), ∃ (x y : ℝ), ¬ circle_c1 x y → 2*λ*x - 2*y + 3 - λ = 0) →
  (∃ (l : ℝ → ℝ → Prop), l = (λ (x y : ℝ) → x + y - 2 = 0)) :=
sorry

end length_of_chord_AB_equation_of_circle_c2_line_l1_intersects_and_shortest_chord_l744_744508


namespace bernardo_vs_silvia_probability_l744_744454

theorem bernardo_vs_silvia_probability :
  let bernardo_set := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let silvia_set := {1, 2, 3, 4, 5, 6, 7, 8}
  let bernardo_choices := (bernardo_set.to_finset.powerset.filter (λ s, s.card = 3)).to_finset
  let silvia_choices := (silvia_set.to_finset.powerset.filter (λ s, s.card = 3)).to_finset
  let total_bernardo := bernardo_choices.card
  let total_silvia := silvia_choices.card
  let favorable_cases := (bernardo_choices.to_list.product silvia_choices.to_list).countp
    (λ (p : finset ℕ × finset ℕ), p.1.to_list.sort (· > ·) > p.2.to_list.sort (· > ·))
  let total_cases := total_bernardo * total_silvia
  (favorable_cases / total_cases : ℚ) = 37 / 56 :=
by sorry

end bernardo_vs_silvia_probability_l744_744454


namespace net_change_correct_l744_744440
-- Import the necessary library

-- Price calculation function
def price_after_changes (initial_price: ℝ) (changes: List (ℝ -> ℝ)): ℝ :=
  changes.foldl (fun price change => change price) initial_price

-- Define each model's price changes
def modelA_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.9, 
  fun price => price * 1.3, 
  fun price => price * 0.85
]

def modelB_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.85, 
  fun price => price * 1.25, 
  fun price => price * 0.80
]

def modelC_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.80, 
  fun price => price * 1.20, 
  fun price => price * 0.95
]

-- Calculate final prices
def final_price_modelA := price_after_changes 1000 modelA_changes
def final_price_modelB := price_after_changes 1500 modelB_changes
def final_price_modelC := price_after_changes 2000 modelC_changes

-- Calculate net changes
def net_change_modelA := final_price_modelA - 1000
def net_change_modelB := final_price_modelB - 1500
def net_change_modelC := final_price_modelC - 2000

-- Set up theorem
theorem net_change_correct:
  net_change_modelA = -5.5 ∧ net_change_modelB = -225 ∧ net_change_modelC = -176 := by
  -- Proof is skipped
  sorry

end net_change_correct_l744_744440


namespace exp_is_convex_log_is_concave_l744_744843

-- Part 1: Prove f(x) = a^x is convex for a > 0.
theorem exp_is_convex {a : ℝ} (h : a > 0) : convex_on ℝ (λ x, a^x) :=
sorry

-- Part 2: Prove f(x) = log x is concave for x > 0.
theorem log_is_concave : concave_on {x : ℝ | x > 0} log :=
sorry

end exp_is_convex_log_is_concave_l744_744843


namespace tan_A_max_area_l744_744514

open Real

variable {a b c A B C : ℝ}
variable [Triangle ABC]

-- First part: Prove that tan(A) = 2sqrt(2) given the condition
theorem tan_A (h : 3 * b * cos A = c * cos A + a * cos C) : tan A = 2 * sqrt 2 :=
by
  sorry

-- Second part: Prove the maximum area of the triangle is 8sqrt(2) when a = 4sqrt(2)
theorem max_area (ha : a = 4 * sqrt 2) : area ABC ≤ 8 * sqrt 2 :=
by
  sorry

end tan_A_max_area_l744_744514


namespace probability_of_picking_dime_l744_744425

def value_of_quarters : ℝ := 15.00
def value_of_nickels : ℝ := 5.00
def value_of_pennies : ℝ := 2.00
def value_of_dimes : ℝ := 12.00
def value_per_quarter : ℝ := 0.25
def value_per_nickel : ℝ := 0.05
def value_per_penny : ℝ := 0.01
def value_per_dime : ℝ := 0.10

theorem probability_of_picking_dime
  (h_quarters : value_of_quarters = 15.00)
  (h_nickels : value_of_nickels = 5.00)
  (h_pennies : value_of_pennies = 2.00)
  (h_dimes : value_of_dimes = 12.00)
  (h_value_per_quarter : value_per_quarter = 0.25)
  (h_value_per_nickel : value_per_nickel = 0.05)
  (h_value_per_penny : value_per_penny = 0.01)
  (h_value_per_dime : value_per_dime = 0.10) :
  let num_quarters := value_of_quarters / value_per_quarter,
      num_nickels := value_of_nickels / value_per_nickel,
      num_pennies := value_of_pennies / value_per_penny,
      num_dimes := value_of_dimes / value_per_dime,
      total_coins := num_quarters + num_nickels + num_pennies + num_dimes in
  (num_dimes / total_coins) = 1 / 4 :=
by sorry

end probability_of_picking_dime_l744_744425


namespace calculate_binomial_sum_l744_744721

theorem calculate_binomial_sum (n : ℕ) :
  let m := (n - 1) / 4 in 
  ∑ i in finset.range (m + 1), nat.choose n (4 * i + 1) = 
  1 / 2 * (2^(n - 1) + real.sqrt (2^n) * real.cos (n * real.pi / 4)) :=
by
  sorry

end calculate_binomial_sum_l744_744721


namespace y_intercept_implies_value_of_m_l744_744611

theorem y_intercept_implies_value_of_m 
  (m : ℝ)
  (h_line : ∀ x y : ℝ, x - 2 * y + m - 1 = 0)
  (h_y_intercept : ∀ y : ℝ, h_line 0 y → y = 1 / 2) : 
  m = 2 :=
sorry

end y_intercept_implies_value_of_m_l744_744611


namespace sum_of_extreme_values_of_x_l744_744254

open Real

theorem sum_of_extreme_values_of_x 
  (x y z : ℝ)
  (h1 : x + y + z = 6)
  (h2 : x^2 + y^2 + z^2 = 14) : 
  (min x + max x) = (10 / 3) :=
sorry

end sum_of_extreme_values_of_x_l744_744254


namespace intersection_distance_correct_l744_744956

noncomputable def distance_between_intersections : ℝ :=
  let p1 := (1, 1, 2 : ℝ × ℝ × ℝ)
  let p2 := (-2, -5, -3 : ℝ × ℝ × ℝ)
  let line_eq (t : ℝ) : ℝ × ℝ × ℝ :=
    (1 - 3 * t, 1 - 6 * t, 2 - 5 * t)
  let sphere_intersection eq : Polynomials.pdata :=
    (line_eq t).1^2 + (line_eq t).2^2 + (line_eq t).3^2 - 1
  let t_vals := quadratic_formula 70 -38 5 in
  let (t1, t2) := (t_vals^2 - t_vals^2) in
  let intersection1 := line_eq t1
  let intersection2 := line_eq t2
  distance intersection1 intersection2

theorem intersection_distance_correct : 
  distance_between_intersections = (√154) / 10 :=
by
  sorry

end intersection_distance_correct_l744_744956


namespace combined_perimeter_two_right_triangles_l744_744362

theorem combined_perimeter_two_right_triangles :
  ∀ (h1 h2 : ℝ),
    (h1^2 = 15^2 + 20^2) ∧
    (h2^2 = 9^2 + 12^2) ∧
    (h1 = h2) →
    (15 + 20 + h1) + (9 + 12 + h2) = 106 := by
  sorry

end combined_perimeter_two_right_triangles_l744_744362


namespace polynomial_binomial_square_l744_744585

theorem polynomial_binomial_square (b : ℚ) :
  (∃ c : ℚ, (3 * polynomial.X + polynomial.C c)^2 = 9 * polynomial.X^2 + 27 * polynomial.X + polynomial.C b) →
  b = 81 / 4 :=
by
  intro h
  rcases h with ⟨c, hc⟩
  have : 6 * c = 27 := by sorry -- This corresponds to solving 6c = 27
  have : c = 9 / 2 := by sorry -- This follows from the above
  have : b = (9 / 2)^2 := by sorry -- This follows from substituting back c and expanding
  simp [this]

end polynomial_binomial_square_l744_744585


namespace closest_point_on_given_line_l744_744493

noncomputable def closest_point_on_line 
  (line_point direction_vector : ℝ × ℝ × ℝ) 
  (t : ℝ) : ℝ × ℝ × ℝ :=
  (line_point.1 + t * direction_vector.1, 
   line_point.2 + t * direction_vector.2, 
   line_point.3 + t * direction_vector.3)

theorem closest_point_on_given_line :
  let point := (1, 2, 3)
  let line_point := (4, 0, 1)
  let direction_vector := (-1, 7, -2)
  let t := 17 / 54
  closest_point_on_line line_point direction_vector t = (199 / 54, 119 / 54, 10 / 27) :=
by
  sorry

end closest_point_on_given_line_l744_744493


namespace solve_for_x_l744_744676

theorem solve_for_x : 
  ∃ x: ℚ, 5 + 3.2 * x = 4.4 * x - 30 ∧ x = 175 / 6 :=
by
  use 175 / 6
  linarith

end solve_for_x_l744_744676


namespace bc_is_one_area_of_triangle_l744_744233

-- Define a triangle with sides a, b, and c and corresponding angles A, B, and C.
structure Triangle :=
(a b c A B C : ℝ)

-- Assume the Law of Cosines condition is given.
axiom Law_of_Cosines_condition (T : Triangle) : (T.b^2 + T.c^2 - T.a^2) / (Real.cos T.A) = 2

-- Prove that bc = 1.
theorem bc_is_one (T : Triangle) (h : Law_of_Cosines_condition T) : T.b * T.c = 1 := 
by {
    sorry
}

-- Assume the given ratio condition.
axiom ratio_condition (T : Triangle) :
  (T.a * Real.cos T.B - T.b * Real.cos T.A) / (T.a * Real.cos T.B + T.b * Real.cos T.A)
  - (T.b / T.c) = 1

-- Prove that the area of the triangle is sqrt(3)/4.
theorem area_of_triangle (T : Triangle) (h1 : Law_of_Cosines_condition T) (h2 : ratio_condition T) :
  let S := 0.5 * T.b * T.c * Real.sin T.A
  in S = Real.sqrt 3 / 4 :=
by {
    sorry
}

end bc_is_one_area_of_triangle_l744_744233


namespace integer_solution_count_l744_744156

theorem integer_solution_count : ∃ (n : ℕ), n = 53 ∧ 
  (∀ (x y : ℤ), x ≠ 0 → y ≠ 0 → (1 : ℚ) / 2022 = (1 : ℚ) / x + (1 : ℚ) / y → 
  (∃ (a b : ℤ), 2022 * (a - 2022) * (b - 2022) = 2022^2) :=
begin
  use 53,
  split, {
    refl,
  },
  intros x y hx hy hxy,
  sorry
end

end integer_solution_count_l744_744156


namespace elderly_people_not_set_l744_744058

def is_well_defined (S : Set α) : Prop := Nonempty S

def all_positive_numbers : Set ℝ := {x : ℝ | 0 < x}
def real_numbers_non_zero : Set ℝ := {x : ℝ | x ≠ 0}
def four_great_inventions : Set String := {"compass", "gunpowder", "papermaking", "printing"}

def elderly_people_description : String := "elderly people"

theorem elderly_people_not_set :
  ¬ (∃ S : Set α, elderly_people_description = "elderly people" ∧ is_well_defined S) :=
sorry

end elderly_people_not_set_l744_744058


namespace exists_indecomposable_multiple_ways_l744_744644

def is_indecomposable_in_vn (n m : ℕ) (Vn : set ℕ) : Prop :=
∃ (m ∈ Vn), ¬ ∃ p q ∈ Vn, p * q = m

theorem exists_indecomposable_multiple_ways (n : ℕ) (hn : n > 2) :
  ∃ (r : ℕ) (Vn : set ℕ), (Vn = {k + 1 * n | k : ℕ ∧ 0 < k}) ∧ 
  r ∈ Vn ∧ 
  (∃ (f1 f2 : list ℕ), 
    (∀ x ∈ f1, x ∈ Vn ∧ is_indecomposable_in_vn n x Vn) ∧ 
    (∀ x ∈ f2, x ∈ Vn ∧ is_indecomposable_in_vn n x Vn) ∧ 
    list.prod f1 = r ∧ 
    list.prod f2 = r ∧ 
    f1 ≠ f2) :=
by 
  sorry

end exists_indecomposable_multiple_ways_l744_744644


namespace collinear_A_l744_744650

variables {A B C A' B' C' : Type*}
variables (h1 : ∀ (P Q R : set.point2d ℝ), ¬ collinear P Q R)
variables (ℋ : set.hyperbola ℝ)
variables (H_tangent_A : line ℝ)
variables (H_tangent_B : line ℝ)
variables (H_tangent_C : line ℝ)
variables (H_orthic_triangle : set.point2d ℝ)
variables (circumcircle_A'B'C' : set.point2d ℝ)
variables (nine_point_circle : set.point2d ℝ)
variables (Simson_line_A' : line ℝ)
variables (A* B* C* : set.point2d ℝ)
variables (H_tangent_property : ∀ {P Q R : set.point2d ℝ}, is_tangent P Q R ℋ)
variables (H_circles : ∃ (circ : circle ℝ), ∀ P ∈ circumcircle_A'B'C', P ∈ nine_point_circle) 

theorem collinear_A*_B*_C* 
: collinear A* B* C* :=
sorry

end collinear_A_l744_744650


namespace four_letter_arrangements_count_l744_744154

theorem four_letter_arrangements_count :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
  let first_letter := 'D';
  ∀ arrangement : List Char, arrangement.length = 4 →
  arrangement.head = first_letter →
  ('E' ∈ arrangement.tail) →
  (∀ x, x ∈ arrangement → arrangement.count x = 1) →
  (arrangement.all (λ x, x ∈ letters))
  ∑ (p : Finset (List Char)), (∃ arrangement,
    p = arrangement ∧
    arrangement.head = first_letter ∧
    ('E' ∈ arrangement.tail) ∧
    (∀ x, x ∈ arrangement → arrangement.count x = 1) ∧
    arrangement.all (λ x, x ∈ letters)) = 60 :=
sorry

end four_letter_arrangements_count_l744_744154


namespace count_valid_quadruples_l744_744815

def validQuadruple (a₀ a₁ a₂ a₃ : ℕ) : Prop :=
  a₀ ≥ 1 ∧ a₁ ≥ 2 * a₀ ∧ a₂ ≥ 2 * a₁ ∧ a₃ ≥ 2 * a₂ ∧ a₃ ≤ 60

theorem count_valid_quadruples : 
  (finset.card (finset.filter (λ quad : Fin 100 × Fin 100 × Fin 100 × Fin 100,
    let (a₀, a₁, a₂, a₃) := quad in
    validQuadruple a₀.val a₁.val a₂.val a₃.val)
    (finset.product (finset.range 61)
      (finset.product (finset.range 31)
        (finset.product (finset.range 16)
          (finset.range 8))))))
  = 27 := 
by
  sorry

end count_valid_quadruples_l744_744815


namespace false_proposition_l744_744001

-- Define propositions as conditions
def proposition_A : Prop := ∀ (a b : ℝ), a ≠ b → ¬(vertical_angle a b)
def proposition_B : Prop := ∀ (p1 p2 : ℝ × ℝ), ∃ (l : set (ℝ × ℝ)), is_line_segment p1 p2 l
def proposition_C : Prop := ∃ (x y : ℝ), irrational x ∧ irrational y ∧ ¬irrational (x + y)
def proposition_D : Prop := ∀ (L1 L2 l : set (ℝ × ℝ)), line L1 ∧ line L2 ∧ intersect l L1 ∧ intersect l L2 → corresponding_angles l L1 L2

-- Prove that the false proposition is D
theorem false_proposition : ¬ proposition_D := 
sorry -- proof goes here

end false_proposition_l744_744001


namespace location_of_points_l744_744918

noncomputable def distance_sq (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 - x2)^2 + (y1 - y2)^2

def Q : (ℝ × ℝ) := (1, -3)

def point_A (x y : ℝ) : Prop := (x, y) = (0, 0)
def point_B (x y : ℝ) : Prop := (x, y) = (-2, 1)
def point_C (x y : ℝ) : Prop := (x, y) = (3, 3)
def point_D (x y : ℝ) : Prop := (x, y) = (2, -1)

def circle_center : (ℝ × ℝ) := Q
def circle_radius_sq : ℝ := 25

def on_circle (x y : ℝ) : Prop :=
  distance_sq x y (circle_center.1) (circle_center.2) = circle_radius_sq

def inside_circle (x y : ℝ) : Prop :=
  distance_sq x y (circle_center.1) (circle_center.2) < circle_radius_sq

def outside_circle (x y : ℝ) : Prop :=
  distance_sq x y (circle_center.1) (circle_center.2) > circle_radius_sq

theorem location_of_points :
  inside_circle 0 0 ∧
  on_circle (-2) 1 ∧
  outside_circle 3 3 ∧
  inside_circle 2 -1 :=
by {
  sorry
}

end location_of_points_l744_744918


namespace correct_answer_l744_744727

-- Define the conditions for the first scenario
def area_function1 (x : ℝ) : ℝ := 5 * (10 - x)

-- Prove that the area function of the first scenario is linear
lemma scenario1_linear : ∀ x : ℝ, ∃ m b : ℝ, area_function1 x = m * x + b :=
by sorry -- Skipping the actual proof

-- Define the conditions for the second scenario
def area_function2 (x : ℝ) : ℝ := (30 + x) * (20 + x)

-- Prove that the area function of the second scenario is quadratic
lemma scenario2_quadratic : ∃ a b c : ℝ, ∀ x : ℝ, area_function2 x = a * x^2 + b * x + c :=
by sorry -- Skipping the actual proof

-- Proving that the correct function types are as identified: Linear for scenario 1 and Quadratic for scenario 2
theorem correct_answer : 
    (∀ x : ℝ, ∃ m b : ℝ, area_function1 x = m * x + b) ∧ 
    (∃ a b c : ℝ, ∀ x : ℝ, area_function2 x = a * x^2 + b * x + c) := 
by 
    split;
    { sorry }

end correct_answer_l744_744727


namespace sum_of_possible_club_members_l744_744785

theorem sum_of_possible_club_members :
  (∑ m in { n | 300 ≤ n ∧ n ≤ 400 ∧ (n - 2) % 8 = 0 }, m) = 4200 :=
by
  sorry

end sum_of_possible_club_members_l744_744785


namespace upper_limit_of_people_l744_744487

theorem upper_limit_of_people (T : ℕ) (h1 : (3/7) * T = 24) (h2 : T > 50) : T ≤ 56 :=
by
  -- The steps to solve this proof would go here.
  sorry

end upper_limit_of_people_l744_744487


namespace points_triangle_area_leq_one_fourth_l744_744985

theorem points_triangle_area_leq_one_fourth :
  ∃ (A B C : (ℝ × ℝ)), ∃ (D : (ℝ × ℝ)),
    let P := (0, 1) in
    let Q := (1, 0) in
    let R := (0, 0) in
    (A = (a, 1 - a) ∧ B = (b, 1 - b) ∧ C = (0, y) ∧ D = (x, 0)) ∧
    ∃ (X Y Z : (ℝ × ℝ)), (X = A ∨ X = B ∨ X = C ∨ X = D) ∧
                         (Y = A ∨ Y = B ∨ Y = C ∨ Y = D) ∧
                         (Z = A ∨ Z = B ∨ Z = C ∨ Z = D) ∧
                         X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X ∧
    (1 / 2) * |(X.1 * (Y.2 - Z.2) + Y.1 * (Z.2 - X.2) + Z.1 * (X.2 - Y.2))| > (1 / 8) :=
sorry

end points_triangle_area_leq_one_fourth_l744_744985


namespace pyarelal_loss_l744_744449

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (h1 : total_loss = 670) (h2 : 1 / 9 * P + P = 10 / 9 * P):
  (9 / (1 + 9)) * total_loss = 603 :=
by
  sorry

end pyarelal_loss_l744_744449


namespace exp_min_value_l744_744182

noncomputable def f (a x : ℝ) : ℝ := a^x

theorem exp_min_value (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : ∀ x ∈ Set.Icc (-2 : ℝ) (1 : ℝ), f a x ≤ 4)
  (h₄ : ∃ x ∈ Set.Icc (-2 : ℝ) (1 : ℝ), f a x = 4) :
  ∃ m ∈ {f a (-2), f a 1}, m = f a (-2) ∨ m = f a 1 :=
by
  sorry

end exp_min_value_l744_744182


namespace geometric_series_sum_l744_744837

def first_term : ℤ := 3
def common_ratio : ℤ := -2
def last_term : ℤ := -1536
def num_terms : ℕ := 10
def sum_of_series (a r : ℤ) (n : ℕ) : ℤ := a * ((r ^ n - 1) / (r - 1))

theorem geometric_series_sum :
  sum_of_series first_term common_ratio num_terms = -1023 := by
  sorry

end geometric_series_sum_l744_744837


namespace min_partitions_not_perfect_square_l744_744297

def S : set ℕ := {n | 1 ≤ n ∧ n ≤ 36}

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def valid_partition (P : finset (finset ℕ)) : Prop :=
  (∀ A ∈ P, A ≠ ∅) ∧
  (∀ A₁ A₂ ∈ P, A₁ ≠ A₂ → disjoint A₁ A₂) ∧
  (∀ A ∈ P, ∀ x y ∈ A, x ≠ y → ¬ is_perfect_square (x + y))

theorem min_partitions_not_perfect_square : ∃ P : finset (finset ℕ), valid_partition P ∧ S.card = 36 ∧ P.card = 3 :=
sorry

end min_partitions_not_perfect_square_l744_744297


namespace rate_of_mangoes_is_60_l744_744564

-- Define the conditions
def kg_grapes : ℕ := 8
def rate_per_kg_grapes : ℕ := 70
def kg_mangoes : ℕ := 9
def total_paid : ℕ := 1100

-- Define the cost of grapes and total cost
def cost_of_grapes : ℕ := kg_grapes * rate_per_kg_grapes
def cost_of_mangoes : ℕ := total_paid - cost_of_grapes
def rate_per_kg_mangoes : ℕ := cost_of_mangoes / kg_mangoes

-- Prove that the rate of mangoes per kg is 60
theorem rate_of_mangoes_is_60 : rate_per_kg_mangoes = 60 := by
  -- Here we would provide the proof
  sorry

end rate_of_mangoes_is_60_l744_744564


namespace volume_of_snow_l744_744629

def length : ℝ := 40
def width : ℝ := 3
def depth : ℝ := 3 / 4

def volume : ℝ := length * width * depth

theorem volume_of_snow : volume = 90 :=
by
  show (40 : ℝ) * (3 : ℝ) * (3 / 4 : ℝ) = 90
  sorry

end volume_of_snow_l744_744629


namespace part1_part2_l744_744555

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1 (a : ℝ) : (set_A ∪ set_B a = set_A ∩ set_B a) → a = 1 :=
sorry

theorem part2 (a : ℝ) : (set_A ∪ set_B a = set_A) → (a ≤ -1 ∨ a = 1) :=
sorry

end part1_part2_l744_744555


namespace smallest_n_for_Tn_integer_l744_744221

def K : ℚ := (1/1) + (1/2) + (1/3) + (1/4) + (1/5) + (1/6) + (1/7) + (1/8) + (1/9)

noncomputable def T_n (n : ℕ) : ℚ := (n * 10^(n-1) + 13) * K + 1

theorem smallest_n_for_Tn_integer : ∃ (n : ℕ), T_n n ∈ ℤ ∧ ∀ (m : ℕ), m < n → T_n m ∉ ℤ :=
begin
  use 63,
  split,
  { sorry },  -- Proof that T_63 is an integer
  { intro m,
    intro h,
    sorry },  -- Proof that for all m<63, T_m is not an integer
end

end smallest_n_for_Tn_integer_l744_744221


namespace log_sum_sq_eq_l744_744136

-- Define the conditions
variables {a : ℝ} {x : ℝ} {x_1 x_2 : fin 2017 → ℝ}
noncomputable def f (x : ℝ) := real.logb a x

-- Provide assumptions
axiom ha_pos : a > 0
axiom ha_ne_one : a ≠ 1
axiom h_prod : f (x_1 0 * x_2 1 * ... * x_{2017} 2016) = 8

-- The main statement to prove
theorem log_sum_sq_eq : f (x_1 0 ^ 2) + f (x_2 1 ^ 2) + ... + f (x_{2017} 2016 ^ 2) = 16 :=
sorry -- proof goes here

end log_sum_sq_eq_l744_744136


namespace length_of_WH_l744_744299

noncomputable def WH_length : ℝ := 12 * Real.sqrt 2

theorem length_of_WH :
  ∀ (W X Y Z G H : Type) (a b c d e : ℝ),
    WXYZ_is_square W X Y Z →
    square_area W X Y Z = 144 →
    point_G_on_AD W X Y Z G →
    perpendicular_at_Z_to_ZG W X Y Z G H →
    triangle_ZGH_area W X Y Z G H = 72 →
    WH = WH_length := by
  intros W X Y Z G H a b c d e hsq hsquare hpointG hperpZ htriangle
  sorry

end length_of_WH_l744_744299


namespace smallest_t_for_circle_covered_l744_744697

theorem smallest_t_for_circle_covered:
  ∃ t, (∀ θ, 0 ≤ θ → θ ≤ t → (∃ r, r = Real.sin θ)) ∧
         (∀ t', (∀ θ, 0 ≤ θ → θ ≤ t' → (∃ r, r = Real.sin θ)) → t' ≥ t) :=
sorry

end smallest_t_for_circle_covered_l744_744697


namespace AK_plus_BM_eq_CM_l744_744960

noncomputable theory

variables {A B C D K M : Point}

-- Defining the rectangle ABCD
def is_rectangle (A B C D : Point) : Prop :=
  dist A B = dist C D ∧ dist B C = dist D A ∧
  angle A B C = π/2 ∧ angle B C D = π/2

-- Point K is on diagonal AC such that CK = BC
def point_K_condition (A B C K : Point) : Prop :=
  collinear A C K ∧ dist C K = dist B C

-- Point M is on side BC such that KM = CM
def point_M_condition (B C M K : Point) : Prop :=
  collinear B C M ∧ dist K M = dist C M

-- Proof that AK + BM = CM in rectangle ABCD
theorem AK_plus_BM_eq_CM
  (Hrect : is_rectangle A B C D)
  (HK : point_K_condition A B C K)
  (HM : point_M_condition B C M K) :
  dist A K + dist B M = dist C M :=
sorry

end AK_plus_BM_eq_CM_l744_744960


namespace number_of_solutions_256_l744_744930

theorem number_of_solutions_256 :
  ∃ (x y : ℕ), Nat.gcd x y = Nat.factorial 20 ∧ Nat.lcm x y = Nat.factorial 30 ∧ (number_of_solutions x y = 256) := 
sorry

end number_of_solutions_256_l744_744930


namespace blocks_left_l744_744669

theorem blocks_left (total_blocks used_blocks : ℕ) (h1 : total_blocks = 78) (h2 : used_blocks = 19) : total_blocks - used_blocks = 59 :=
by
  rwa [h1, h2]

end blocks_left_l744_744669


namespace games_played_l744_744085

theorem games_played (x : ℕ) (h1 : x * 26 + 42 * (20 - x) = 600) : x = 15 :=
by {
  sorry
}

end games_played_l744_744085


namespace problem_solution_l744_744428

noncomputable def a_b_sum : ℤ :=
  let k := (10 : ℝ) + real.sqrt 11 in -- As derived k ≈ 10 + √11
  let intersection1 := (k, real.log_base 2 k) in
  let intersection2 := (k, real.log_base 2 (k + 6)) in
  let dist := real.abs (real.log_base 2 k - real.log_base 2 (k + 6)) in
  if dist = 0.6 then 10 + 11 else 0

theorem problem_solution : a_b_sum = 21 := 
by {
  sorry
}

end problem_solution_l744_744428


namespace fraction_of_teachers_ate_pizza_l744_744450

variables (T : ℚ) (teachers staff members : ℕ) 
          (non_pizza_eaters pizza_fraction_staff : ℚ)

-- Conditions
def conditions : Prop :=
  teachers = 30 ∧
  staff_members = 45 ∧
  pizza_fraction_staff = 4/5 ∧
  non_pizza_eaters = 19

-- Equation: Proving the fraction of teachers who ate pizza
theorem fraction_of_teachers_ate_pizza (h : conditions) : T = 2/3 :=
by
  cases h
  simp only [teachers, staff_members, pizza_fraction_staff, non_pizza_eaters] at *
  sorry

end fraction_of_teachers_ate_pizza_l744_744450


namespace area_ratio_l744_744649

/-
  Let point O be a point inside triangle ABC that satisfies the equation:
  vector_OA + 2 * vector_OB + 3 * vector_OC = 
    3 * vector_AB + 2 * vector_BC + vector_CA.
  Then, prove that the value of 
  (area_△AOB + 2 * area_△BOC + 3 * area_△COA) / area_△ABC 
  is equal to 11 / 6.
-/

variables {A B C O : Type} 
          [AffineSpace ℝ A] 
          [AffineSpace ℝ B] 
          [AffineSpace ℝ C] 
          [AffineSpace ℝ O]

variables (vector_OA vector_OB vector_OC vector_AB vector_BC vector_CA : ℝ → AffineSpace ℝ ℝ)

/-- The given condition: 
    vector_OA + 2 * vector_OB + 3 * vector_OC = 3 * vector_AB + 2 * vector_BC + vector_CA. -/
def given_condition : Prop :=
  vector_OA + 2 * vector_OB + 3 * vector_OC = 3 * vector_AB + 2 * vector_BC + vector_CA

/-- The areas concerned. -/
variables (S_△AOB S_△BOC S_△COA S_△ABC : ℝ) 

/-- The mathematical proof problem statement, proving the correct answer is 11/6. -/
theorem area_ratio : given_condition vector_OA vector_OB vector_OC vector_AB vector_BC vector_CA → 
  (S_△AOB + 2 * S_△BOC + 3 * S_△COA) / S_△ABC = 11 / 6 :=
sorry

end area_ratio_l744_744649


namespace prove_bc_prove_area_l744_744228

variable {a b c A B C : ℝ}

-- Given conditions as Lean definitions
def condition_cos_A : Prop := (b^2 + c^2 - a^2 = 2 * cos A)
def condition_bc : Prop := b * c = 1
def condition_trig_identity : Prop := (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1

-- Definitions for the questions translated to Lean
def find_bc : Prop :=
  condition_cos_A → b * c = 1

def find_area (bc_value : Prop) : Prop :=
  bc_value →
  condition_trig_identity →
  1/2 * b * c * sin A = (Real.sqrt 3) / 4

-- Theorems to be proven
theorem prove_bc : find_bc :=
by
  intro h1
  sorry

theorem prove_area : find_area condition_bc :=
by
  intro h1 h2
  sorry

end prove_bc_prove_area_l744_744228


namespace tory_needs_to_sell_more_packs_l744_744495

theorem tory_needs_to_sell_more_packs 
  (total_goal : ℤ) (packs_grandmother : ℤ) (packs_uncle : ℤ) (packs_neighbor : ℤ) 
  (total_goal_eq : total_goal = 50)
  (packs_grandmother_eq : packs_grandmother = 12)
  (packs_uncle_eq : packs_uncle = 7)
  (packs_neighbor_eq : packs_neighbor = 5) :
  total_goal - (packs_grandmother + packs_uncle + packs_neighbor) = 26 :=
by
  rw [total_goal_eq, packs_grandmother_eq, packs_uncle_eq, packs_neighbor_eq]
  norm_num

end tory_needs_to_sell_more_packs_l744_744495


namespace train_length_l744_744442

-- Define conditions

-- Condition 1: The train takes a specific time to cross the man.
def cross_time : ℝ := 10.999120070394369

-- Condition 2: Speed of the man in km/hr and converting to m/s.
def speed_man_km_per_hr : ℝ := 8
def speed_man_m_per_s : ℝ := speed_man_km_per_hr * 1000 / 3600

-- Condition 3: Speed of the train in km/hr and converting to m/s.
def speed_train_km_per_hr : ℝ := 80
def speed_train_m_per_s : ℝ := speed_train_km_per_hr * 1000 / 3600

-- Relative speed of the train and the man in m/s
def relative_speed : ℝ := speed_train_m_per_s - speed_man_m_per_s

-- Prove that the length of the train is the product of relative speed and the time taken to cross
theorem train_length :
  (relative_speed * cross_time) = 219.98 :=
by sorry

end train_length_l744_744442


namespace solve_fractional_equation_l744_744864

theorem solve_fractional_equation (x : ℝ) :
  (\frac{15 * x - x^2}{x + 2} * (x + \frac{15 - x}{x + 2}) = 36) ↔ (x = 4 ∨ x = 3) :=
sorry

end solve_fractional_equation_l744_744864


namespace time_remaining_correct_total_time_correct_t1_gt_t2_l744_744782

variable (x a b : ℝ)
variable (hx : x ≠ 0)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hab_neq : a ≠ b)

def time_remaining (x : ℝ) : ℝ :=
  (360 - 2 * x) / (3 * x)

def total_time (x : ℝ) : ℝ :=
  180 / x - 2 / 3

def t1 (a b : ℝ) : ℝ :=
  90 * (a + b) / (a * b)

def t2 (a b : ℝ) : ℝ :=
  360 / (a + b)

theorem time_remaining_correct : time_remaining x = (360 - 2 * x) / (3 * x) :=
by sorry

theorem total_time_correct (hx : x = 60) : total_time 60 = 7 / 3 :=
by sorry

theorem t1_gt_t2 : t1 a b > t2 a b :=
by 
  have h : (90 * (a - b) ^ 2) / (a * b * (a + b)) > 0,
    from by sorry,
  exact h

end time_remaining_correct_total_time_correct_t1_gt_t2_l744_744782


namespace dan_violet_marbles_l744_744467

def InitMarbles : ℕ := 128
def MarblesGivenMary : ℕ := 24
def MarblesGivenPeter : ℕ := 16
def MarblesReceived : ℕ := 10

def FinalMarbles : ℕ := InitMarbles - MarblesGivenMary - MarblesGivenPeter + MarblesReceived

theorem dan_violet_marbles : FinalMarbles = 98 := 
by 
  sorry

end dan_violet_marbles_l744_744467


namespace length_of_integer_is_five_l744_744480

theorem length_of_integer_is_five :
  ∀ (digits : Finset ℕ),
    (∀ x, x ∈ digits → x ∈ (Finset.range 6 \ {0})) →
    (∀ p : List ℕ, (∀ x, x ∈ p → x ∈ digits) → (p.length = 5) →
      (¬ (∃ i, p.nth i = some 3 ∧ p.nth (i + 1) = some 4) ∧
       ¬ (∃ i, p.nth i = some 4 ∧ p.nth (i + 1) = some 3))) → 
    72 = ((Finset.permutations (digits)).card) →
    digits.card = 5 := by
  sorry

end length_of_integer_is_five_l744_744480


namespace carson_clawed_39_times_l744_744458

def wombats_count := 9
def wombat_claws_per := 4
def rheas_count := 3
def rhea_claws_per := 1

def wombat_total_claws := wombats_count * wombat_claws_per
def rhea_total_claws := rheas_count * rhea_claws_per
def total_claws := wombat_total_claws + rhea_total_claws

theorem carson_clawed_39_times : total_claws = 39 :=
  by sorry

end carson_clawed_39_times_l744_744458


namespace km_to_miles_approx_l744_744840

-- Definition of the Fibonacci sequence
def Fibonacci : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => Fibonacci n + Fibonacci (n + 1)

-- Representation of a number in the Fibonacci numeral system
-- and the transformation/shift operation
def to_fib (n : ℕ) : list ℕ := sorry
def shift_fib (fib_rep : list ℕ) : list ℕ := sorry

-- Conversion from Fibonacci numeral system to base 10
def from_fib (fib_rep : list ℕ) : ℕ := sorry

theorem km_to_miles_approx (n : ℕ) (h_bounds : n ≤ 100) : 
  let conversion_factor : ℝ := 1 / 1.609,
  let in_fib := to_fib n,
  let shifted_fib := shift_fib in_fib,
  let approx_miles := from_fib shifted_fib
  abs (approx_miles - (n * conversion_factor)) < 2 / 3 :=
sorry

end km_to_miles_approx_l744_744840


namespace polynomial_root_expression_l744_744866

theorem polynomial_root_expression (a b : ℂ) 
  (h₁ : a + b = 5) (h₂ : a * b = 6) : 
  a^4 + a^5 * b^3 + a^3 * b^5 + b^4 = 2905 := by
  sorry

end polynomial_root_expression_l744_744866


namespace find_n_l744_744696

noncomputable def a_n (n : ℕ) : ℚ := (2^n - 1) / 2^n

noncomputable def S_n (n : ℕ) : ℚ := (list.range n).sum (λ i, a_n (i + 1))

theorem find_n : ∃ n : ℕ, S_n n = 321 / 64 ∧ n = 6 :=
by sorry

end find_n_l744_744696


namespace min_students_l744_744441

theorem min_students (n questions folders per_folder : ℕ) 
  (solve_all : ℕ → Set ℕ → Prop)
  (cond1 : n = 6)
  (cond2 : questions = 2010)
  (cond3 : folders = 3)
  (cond4 : per_folder = 670)
  (cond5 : ∀ q, card {s | ¬ solve_all s q} ≤ 2) :
  cond1 :=
by
  sorry

end min_students_l744_744441


namespace complex_power_identity_l744_744528

theorem complex_power_identity (i : ℂ) (hi : i^2 = -1) :
  ( (1 + i) / (1 - i) ) ^ 2013 = i :=
by sorry

end complex_power_identity_l744_744528


namespace number_of_special_divisors_l744_744890

theorem number_of_special_divisors (p q : ℕ) (hp : p > 0) (hq : q > 0) :
  let n := 2^p * 3^q in
  (finset.filter (λ d, d < n ∧ ¬ (d ∣ n)) (finset.divisors (n * n))).card = p * q :=
by
  let n := 2^p * 3^q
  sorry

end number_of_special_divisors_l744_744890


namespace evaluate_powers_of_i_mod_4_l744_744855

theorem evaluate_powers_of_i_mod_4 :
  (Complex.I ^ 48 + Complex.I ^ 96 + Complex.I ^ 144) = 3 := by
  sorry

end evaluate_powers_of_i_mod_4_l744_744855


namespace distance_from_Beijing_to_Lanzhou_l744_744745

-- Conditions
def distance_Beijing_Lanzhou_Lhasa : ℕ := 3985
def distance_Lanzhou_Lhasa : ℕ := 2054

-- Define the distance from Beijing to Lanzhou
def distance_Beijing_Lanzhou : ℕ := distance_Beijing_Lanzhou_Lhasa - distance_Lanzhou_Lhasa

-- Proof statement that given conditions imply the correct answer
theorem distance_from_Beijing_to_Lanzhou :
  distance_Beijing_Lanzhou = 1931 :=
by
  -- conditions and definitions are already given
  sorry

end distance_from_Beijing_to_Lanzhou_l744_744745


namespace triangle_side_length_l744_744186

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ)
  (h : A + C = 2 * B) (ha : a = 1) (hb : b = √3)
  (angle_sum : A + B + C = π) :
  c = √3 :=
by
  sorry

end triangle_side_length_l744_744186


namespace problem1_proof_problem2_proof_l744_744067

noncomputable def problem1_statement : Prop :=
  (2 * Real.sin (Real.pi / 6) - Real.sin (Real.pi / 4) * Real.cos (Real.pi / 4) = 1 / 2)

noncomputable def problem2_statement : Prop :=
  ((-1)^2023 + 2 * Real.sin (Real.pi / 4) - Real.cos (Real.pi / 6) + Real.sin (Real.pi / 3) + Real.tan (Real.pi / 3)^2 = 2 + Real.sqrt 2)

theorem problem1_proof : problem1_statement :=
by
  sorry

theorem problem2_proof : problem2_statement :=
by
  sorry

end problem1_proof_problem2_proof_l744_744067


namespace quadratic_eq_D_l744_744383

variable (a b c x y : ℝ)

-- Defining the given conditions
def A := a * x^2 + b * x + c = 0
def B := (1 / x^2) + (1 / x) = 2
def C := x^2 + 2 * x = y^2 - 1
def D := 3 * (x + 1)^2 = 2 * (x + 1)

-- The main theorem to prove
theorem quadratic_eq_D :
  ¬A ∧ ¬B ∧ ¬C ∧ D ∧ (∃ a b c, a ≠ 0 ∧ 3 * x^2 + 4 * x + 1 = a * x^2 + b * x + c) :=
by
  sorry

end quadratic_eq_D_l744_744383


namespace k_value_five_l744_744118

theorem k_value_five (a b k : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (a^2 + b^2) / (a * b - 1) = k) : k = 5 := 
sorry

end k_value_five_l744_744118


namespace triangle_side_ratio_sqrt2_l744_744014

variables (A B C A1 B1 C1 X Y : Point)
variable (triangle : IsAcuteAngledTriangle A B C)
variable (altitudes : AreAltitudes A B C A1 B1 C1)
variable (midpoints : X = Midpoint A C1 ∧ Y = Midpoint A1 C)
variable (equality : Distance X Y = Distance B B1)

theorem triangle_side_ratio_sqrt2 :
  ∃ (AC AB : ℝ), (AC / AB = Real.sqrt 2) := sorry

end triangle_side_ratio_sqrt2_l744_744014


namespace ratio_of_areas_l744_744322

theorem ratio_of_areas (s : ℝ) (hs : 0 < s) :
  let longer_side_R := 1.2 * s
  let shorter_side_R := 0.85 * s
  let area_R := longer_side_R * shorter_side_R
  let area_S := s^2
  area_R / area_S = 51 / 50 :=
by
  let longer_side_R := 1.2 * s
  let shorter_side_R := 0.85 * s
  let area_R := longer_side_R * shorter_side_R
  let area_S := s^2
  calc
    area_R / area_S = (1.2 * s * 0.85 * s) / (s * s) : by rw [area_R, area_S]
    ... = 1.02 : by { field_simp [ne_of_gt hs], ring }
    ... = 51 / 50 : by norm_num

end ratio_of_areas_l744_744322


namespace power_function_value_l744_744127

theorem power_function_value (a : ℝ) (ha : 3^a = 1/9) : (2 : ℝ)^a = 1/4 :=
sorry

end power_function_value_l744_744127


namespace pure_imaginary_a_zero_l744_744131

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_a_zero (a : ℝ) (h : is_pure_imaginary (i / (1 + a * i))) : a = 0 :=
sorry

end pure_imaginary_a_zero_l744_744131


namespace triangle_area_5_3_4_l744_744367

theorem triangle_area_5_3_4 : let a := 5
                               let b := 3
                               let c := 4
                               let s := (a + b + c) / 2
                               let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
                               in area = 6 := by
  let a := 5
  let b := 3
  let c := 4
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  show area = 6
  by sorry

end triangle_area_5_3_4_l744_744367


namespace range_of_f_prime_over_f_l744_744532

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f_prime (x : ℝ) : ℝ := sorry

axiom f_equation : ∀ x : ℝ, f_prime x + f x = 2 * x * real.exp (-x)
axiom f_initial : f 0 = 1

theorem range_of_f_prime_over_f : set.range (λ x : ℝ, f_prime x / f x) = set.Icc (-2 : ℝ) 0 :=
sorry

end range_of_f_prime_over_f_l744_744532


namespace replacement_in_june_l744_744486

-- Define the months in a year
inductive Month
| January | February | March | April | May | June | July | August | September | October | November | December

-- Define the next month function to account for the cyclic nature of months
def next_month : Month → Month
| Month.January   := Month.February
| Month.February  := Month.March
| Month.March     := Month.April
| Month.April     := Month.May
| Month.May       := Month.June
| Month.June      := Month.July
| Month.July      := Month.August
| Month.August    := Month.September
| Month.September := Month.October
| Month.October   := Month.November
| Month.November  := Month.December
| Month.December  := Month.January

-- Define a function to advance n months from a given month
def advance_months (start: Month) (n : Nat) : Month :=
  (List.iterate next_month n start) 

-- Problem Statement: Prove that the 12th replacement occurs in June
theorem replacement_in_june : advance_months Month.January (7 * 11) = Month.June := 
sorry

end replacement_in_june_l744_744486


namespace solution_problem_l744_744533

noncomputable def proof_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : Prop :=
  (-1 < (x - y)) ∧ ((x - y) < 1) ∧ (∀ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (x + y = 1) → (min ((1/x) + (x/y)) = 3))

theorem solution_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) :
  proof_problem x y hx hy h := 
sorry

end solution_problem_l744_744533


namespace part_a_part_b_l744_744904

noncomputable def correlationFunction_X (ω : ℝ) (t1 t2 : ℝ) : ℝ :=
  cos (ω * t1) * cos (ω * t2)

noncomputable def process_Y (X : ℝ → ℝ) (t : ℝ) : ℝ :=
∫ (u : ℝ) in (0..t), X u

noncomputable def correlationFunction_Y (ω : ℝ) (t1 t2 : ℝ) : ℝ :=
(∫ (u : ℝ) in (0..t1), cos (ω * u)) * (∫ (v : ℝ) in (0..t2), cos (ω * v)) / ω^2

noncomputable def variance_Y (ω : ℝ) (t : ℝ) : ℝ :=
(∫ (u : ℝ) in (0..t), cos (ω * u)) ^ 2 / ω^2

theorem part_a (ω t1 t2 : ℝ) :
  correlationFunction_Y ω t1 t2 = (sin (ω * t1) * sin (ω * t2)) / ω^2 :=
sorry

theorem part_b (ω t : ℝ) :
  variance_Y ω t = (sin (ω * t)) ^ 2 / ω^2 :=
sorry

end part_a_part_b_l744_744904


namespace tan_a4_a12_eq_neg_sqrt3_l744_744534

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

variables {a : ℕ → ℝ} (h_arith : is_arithmetic_sequence a)
          (h_sum : a 1 + a 8 + a 15 = Real.pi)

-- The main statement to prove
theorem tan_a4_a12_eq_neg_sqrt3 : 
  Real.tan (a 4 + a 12) = -Real.sqrt 3 :=
sorry

end tan_a4_a12_eq_neg_sqrt3_l744_744534


namespace range_of_x_when_a_is_1_and_p_and_q_range_of_a_when_q_is_sufficient_not_necessary_for_p_l744_744887

variable (x a : ℝ)

-- Part 1
theorem range_of_x_when_a_is_1_and_p_and_q:
  a = 1 ∧ (x - a) * (x - 3 * a) < 0 ∧ 8 < 2 ^ (x + 1) ∧ 2 ^ (x + 1) ≤ 16 →
  2 < x ∧ x < 3 := sorry

-- Part 2
theorem range_of_a_when_q_is_sufficient_not_necessary_for_p:
  (∀ x, 8 < 2 ^ (x + 1) ∧ 2 ^ (x + 1) ≤ 16 → (x - a) * (x - 3 * a) < 0) →
  (∀ x, (x - a) * (x - 3 * a) < 0 → 8 < 2 ^ (x + 1) ∧ 2 ^ (x + 1) ≤ 16 → False) →
  (1 < a ∧ a ≤ 2) := sorry

end range_of_x_when_a_is_1_and_p_and_q_range_of_a_when_q_is_sufficient_not_necessary_for_p_l744_744887


namespace joe_paint_problem_l744_744976

theorem joe_paint_problem (f : ℝ) (h₁ : 360 * f + (1 / 6) * (360 - 360 * f) = 135) : f = 1 / 4 := 
by
  sorry

end joe_paint_problem_l744_744976


namespace find_principal_l744_744094

noncomputable def principal (P : ℝ) : ℝ :=
  let r1 := 0.07
  let r2 := 0.09
  let r3 := 0.11 * (2 / 5)
  let i := 200
  let A1 := P * (1 + r1) + i
  let A2 := A1 * (1 + r2) + i
  let A3 := A2 * (1 + r3)
  A3

theorem find_principal (P : ℝ) : principal P = 1120 → P ≈ 556.25 := 
by
  sorry

end find_principal_l744_744094


namespace find_max_min_y_l744_744491

theorem find_max_min_y (x : ℝ) : 
  let y := 2 * Real.cos x - 1
  ∃ y_max y_min, y_max = 1 ∧ y_min = -3 ∧ (∀ x : ℝ, y ≤ y_max ∧ y ≥ y_min) :=
by
  -- Define y according to the given function
  let y := 2 * Real.cos x - 1
  
  -- Introduce the maximum and minimum values for y
  use 1, -3

  -- Prove the maximum value
  have h1 : y = 1 ↔ 2 * Real.cos x - 1 = 1 := by sorry
  have hmax : y ≤ 1 := by sorry

  -- Prove the minimum value
  have h2 : y = -3 ↔ 2 * Real.cos x - 1 = -3 := by sorry
  have hmin : y ≥ -3 := by sorry

  exact ⟨1, -3, hmax, hmin⟩

end find_max_min_y_l744_744491


namespace polynomial_value_at_2018_l744_744215

theorem polynomial_value_at_2018 (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f (-x^2 - x - 1) = x^4 + 2*x^3 + 2022*x^2 + 2021*x + 2019) : 
  f 2018 = -2019 :=
sorry

end polynomial_value_at_2018_l744_744215


namespace number_of_real_roots_l744_744497

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem number_of_real_roots (a : ℝ) :
    ((|a| < (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ x₃ : ℝ, f x₁ = a ∧ f x₂ = a ∧ f x₃ = a ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ∧
    ((|a| > (2 * Real.sqrt 3) / 9) → (∃ x : ℝ, f x = a ∧ ∀ y : ℝ, f y = a → y = x)) ∧
    ((|a| = (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ : ℝ, f x₁ = a ∧ f x₂ = a ∧ x₁ ≠ x₂ ∧ ∀ y : ℝ, (f y = a → (y = x₁ ∨ y = x₂)) ∧ (x₁ = x₂ ∨ ∀ z : ℝ, (f z = a → z = x₁ ∨ z = x₂)))) := sorry

end number_of_real_roots_l744_744497


namespace compressor_distances_distances_when_a_15_l744_744013

theorem compressor_distances (a : ℝ) (x y z : ℝ) (h1 : x + y = 2 * z) (h2 : x + z = y + a) (h3 : x + z = 75) :
  0 < a ∧ a < 100 → 
  let x := (75 + a) / 3;
  let y := 75 - a;
  let z := 75 - x;
  x + y = 2 * z ∧ x + z = y + a ∧ x + z = 75 :=
sorry

theorem distances_when_a_15 (x y z : ℝ) (h : 15 = 15) :
  let x := (75 + 15) / 3;
  let y := 75 - 15;
  let z := 75 - x;
  x = 30 ∧ y = 60 ∧ z = 45 :=
sorry

end compressor_distances_distances_when_a_15_l744_744013


namespace part_I_part_II_l744_744139

noncomputable def f (x : ℝ) : ℝ := abs x

theorem part_I (x : ℝ) : f (x-1) > 2 ↔ x < -1 ∨ x > 3 := 
by sorry

theorem part_II (x y z : ℝ) (h : f x ^ 2 + y ^ 2 + z ^ 2 = 9) : ∃ (min_val : ℝ), min_val = -9 ∧ ∀ (a b c : ℝ), f a ^ 2 + b ^ 2 + c ^ 2 = 9 → (a + 2 * b + 2 * c) ≥ min_val := 
by sorry

end part_I_part_II_l744_744139


namespace major_premise_incorrect_l744_744797

theorem major_premise_incorrect (a : ℝ) (ha1 : a > 1 ∨ (0 < a ∧ a < 1)) :
  ¬(∀ x : ℝ, (0 < x → (log a x) < log a (x + 1))) :=
begin
  sorry
end

end major_premise_incorrect_l744_744797


namespace sum_of_first_five_terms_l744_744114

variable (a b : ℕ → ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (d : ℝ) (a1 : ℝ) : ℕ → ℝ
| 0     => a1
| (n+1) => a1 + n * d

-- Given conditions
variable (d : ℝ) (a1 : ℝ) (h : arithmetic_sequence d a1 1 + arithmetic_sequence d a1 3 = 6)

-- Proof problem statement
theorem sum_of_first_five_terms (a d : ℝ) (h : arithmetic_sequence d a 1 + arithmetic_sequence d a 3 = 6) :
  ∑ i in finset.range 5, arithmetic_sequence d a i = 15 :=
sorry

end sum_of_first_five_terms_l744_744114


namespace simplify_f_f_value_in_third_quadrant_f_value_for_specific_alpha_l744_744506

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (Real.pi + α) * Real.cos (2 * Real.pi - α) * Real.tan (-α)) 
  / 
  (Real.tan (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem simplify_f (α : ℝ) : f α = -Real.cos α := 
sorry

theorem f_value_in_third_quadrant (α : ℝ) 
  (h1 : α ∈ set.Icc (Real.pi) (3 * Real.pi / 2)) 
  (h2 : Real.sin (α - Real.pi) = 1 / 5) : 
  f α = 2 * Real.sqrt 6 / 5 := 
sorry

theorem f_value_for_specific_alpha : f (-31 * Real.pi / 5) = -Real.cos (Real.pi / 5) := 
sorry

end simplify_f_f_value_in_third_quadrant_f_value_for_specific_alpha_l744_744506


namespace max_distance_P_to_C_D_l744_744917

theorem max_distance_P_to_C_D 
  (θ : ℝ)
  (0 ≤ θ) (θ < 2 * Real.pi) : 
  let x := sqrt 2 * Real.cos θ
  let y := sqrt 2 * Real.sin θ
  let P := (x, y)
  let A := (1, 1)
  let B := (-1, -1)
  let C := (1/2, 1/2)
  let D := (-1/2, -1/2) in
  (dist P C + dist P D) ≤ sqrt 10 :=
begin
  sorry
end

end max_distance_P_to_C_D_l744_744917


namespace periodicity_of_f_l744_744476

noncomputable def f (x : ℝ) : ℝ := x - floor x

theorem periodicity_of_f : ∀ x : ℝ, f (x + 1) = f x :=
by
  intro x
  have h1 : f (x + 1) = x + 1 - floor (x + 1), from rfl
  have h2 : floor (x + 1) = floor x + 1, from Int.floor_add_one x
  rw [h2] at h1
  rw [h1]
  rw [Int.cast_add]
  rw [Int.cast_one]
  linarith

lemma no_smaller_positive_period :
  ¬∃ T : ℝ, 0 < T ∧ T < 1 ∧ (∀ x : ℝ, f(x + T) = f x) :=
by
  intro h
  rcases h with ⟨T, ⟨hT0, hT1, hTF⟩⟩
  have hT_f0 : f T = f 0 := hTF 0
  have h_f0 : f 0 = 0 := by
    show f (0 : ℝ) = 0
    simp [f]
  have hI : ∀ x : ℝ, 0 ≤ x - floor x ∧ x - floor x < 1 by
    intro x
    exact Int.fract_divmod x
  have hT_correct : T = f T := rfl
  have : T = f 0 := by
    rw [hT_f0]
    rw [h_f0] 
  contradiction
 
example : (∀ x : ℝ, f (x + 1) = f x) ∧
           (¬∃ T : ℝ, 0 < T ∧ T < 1 ∧ (∀ x : ℝ, f(x + T) = f x)) :=
by 
  constructor
  apply periodicity_of_f
  apply no_smaller_positive_period

end periodicity_of_f_l744_744476


namespace ratio_sphere_locus_l744_744370

noncomputable def sphere_locus_ratio (r : ℝ) : ℝ :=
  let F1 := 2 * Real.pi * r^2 * (1 - Real.sqrt (2 / 3))
  let F2 := Real.pi * r^2 * (2 * Real.sqrt 3 / 3)
  F1 / F2

theorem ratio_sphere_locus (r : ℝ) (h : r > 0) : sphere_locus_ratio r = Real.sqrt 3 - 1 :=
by
  sorry

end ratio_sphere_locus_l744_744370


namespace find_digit_P_l744_744692

theorem find_digit_P (P Q R S T : ℕ) (digits : Finset ℕ) (h1 : digits = {1, 2, 3, 6, 8}) 
(h2 : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T)
(h3 : P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ T ∈ digits)
(hPQR_div_6 : (100 * P + 10 * Q + R) % 6 = 0)
(hQRS_div_8 : (100 * Q + 10 * R + S) % 8 = 0)
(hRST_div_3 : (100 * R + 10 * S + T) % 3 = 0) : 
P = 2 := 
sorry

end find_digit_P_l744_744692


namespace initial_glass_bowls_l744_744041

noncomputable def percent_gain := 8.050847457627118 / 100
noncomputable def cost_per_bowl := 12
noncomputable def selling_price_per_bowl := 15
noncomputable def bowls_sold := 102

theorem initial_glass_bowls (x : ℕ) : 
  ([(bowls_sold * selling_price_per_bowl) - (bowls_sold * cost_per_bowl)] / (cost_per_bowl * x) = percent_gain) → 
  x = 316 :=
by
  sorry

end initial_glass_bowls_l744_744041


namespace combined_time_third_attempt_l744_744978

noncomputable def first_lock_initial : ℕ := 5
noncomputable def second_lock_initial : ℕ := 3 * first_lock_initial - 3
noncomputable def combined_initial : ℕ := 5 * second_lock_initial

noncomputable def first_lock_second_attempt : ℝ := first_lock_initial - 0.1 * first_lock_initial
noncomputable def first_lock_third_attempt : ℝ := first_lock_second_attempt - 0.1 * first_lock_second_attempt

noncomputable def second_lock_second_attempt : ℝ := second_lock_initial - 0.15 * second_lock_initial
noncomputable def second_lock_third_attempt : ℝ := second_lock_second_attempt - 0.15 * second_lock_second_attempt

noncomputable def combined_third_attempt : ℝ := 5 * second_lock_third_attempt

theorem combined_time_third_attempt : combined_third_attempt = 43.35 :=
by
  sorry

end combined_time_third_attempt_l744_744978


namespace triangle_side_length_l744_744604

theorem triangle_side_length (a : ℝ) (B : ℝ) (C : ℝ) (c : ℝ) 
  (h₀ : a = 10) (h₁ : B = 60) (h₂ : C = 45) : 
  c = 10 * (Real.sqrt 3 - 1) :=
sorry

end triangle_side_length_l744_744604


namespace unique_perpendicular_line_through_point_l744_744530

-- Definitions of the geometric entities and their relationships
structure Point := (x : ℝ) (y : ℝ)

structure Line := (m : ℝ) (b : ℝ)

-- A function to check if a point lies on a given line
def point_on_line (P : Point) (l : Line) : Prop := P.y = l.m * P.x + l.b

-- A function to represent that a line is perpendicular to another line at a given point
def perpendicular_lines_at_point (P : Point) (l1 l2 : Line) : Prop :=
  l1.m = -(1 / l2.m) ∧ point_on_line P l1 ∧ point_on_line P l2

-- The statement to be proved
theorem unique_perpendicular_line_through_point (P : Point) (l : Line) (h : point_on_line P l) :
  ∃! l' : Line, perpendicular_lines_at_point P l' l :=
by
  sorry

end unique_perpendicular_line_through_point_l744_744530


namespace minimum_absolute_value_expression_l744_744372

theorem minimum_absolute_value_expression :
  ∃ x ∈ set.Icc 0 10, 
    |x - 4| + |x + 2| + |x - 5| + |3 * x - 1| + |2 * x + 6| = 17.333 :=
begin
  use 1/3,
  split,
  -- proving 1/3 is in the interval [0, 10]
  { split; norm_num },
  -- showing that the minimum value is 17.333
  { norm_num,
    sorry }
end

end minimum_absolute_value_expression_l744_744372


namespace derivative_of_y_l744_744861

noncomputable def y (x : ℝ) : ℝ := 
  real.cbrt (cos (real.sqrt 2)) - (1 / 52) * (cos (26 * x))^2 / sin (52 * x)

theorem derivative_of_y (x : ℝ) : 
  deriv y x = 1 / (2 * (sin (26 * x))^2) :=
by 
  sorry

end derivative_of_y_l744_744861


namespace sum_x_coordinates_mod11_l744_744772

theorem sum_x_coordinates_mod11 :
    ∀ x y : ℕ, (y ≡ 3 * x + 1 [MOD 11]) → (y ≡ 7 * x + 5 [MOD 11]) → (0 ≤ x ∧ x < 11) → x = 10 := 
by
  intros x y h1 h2 bounds
  have : 3 * x + 1 ≡ 7 * x + 5 [MOD 11] := by rw [Nat.ModEq.symm h1, h2]
  have : 4 * x + 4 ≡ 0 [MOD 11] := by linarith 
  have : 4 * (x + 1) ≡ 0 [MOD 11] := by linarith
  have : x + 1 ≡ 0 [MOD 11] := sorry -- because 4 and 11 are coprime
  have : x ≡ 10 [MOD 11] := by linarith
  exact sorry -- constraints on x show it must be 10

end sum_x_coordinates_mod11_l744_744772


namespace triangle_area_correct_l744_744055

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_correct :
  let A : point := (3, -1)
  let B : point := (3, 6)
  let C : point := (8, 6)
  triangle_area A B C = 17.5 :=
by
  sorry

end triangle_area_correct_l744_744055


namespace log_product_eq_l744_744485

def log_product : ℝ :=
  let x := (list.range 36).map (λ n, real.log (n + 5) / real.log (n + 4)) |>.prod in
  x

theorem log_product_eq : log_product = (3 * real.log 2 + real.log 5) / (2 * real.log 2) :=
  sorry

end log_product_eq_l744_744485


namespace simplify_expr_l744_744462

theorem simplify_expr : ∀ (y : ℤ), y = 3 → (y^6 + 8*y^3 + 16) / (y^3 + 4) = 31 :=
by
  assume y hy
  have h : y = 3 := hy
  sorry

end simplify_expr_l744_744462


namespace first_player_always_wins_l744_744361

structure Table :=
  (width : ℕ)
  (height : ℕ)

def free_spot (table: Table) (x y : ℕ) : Prop :=
  x < table.width ∧ y < table.height

def first_player_can_move (table : Table) (occupied: set (ℕ × ℕ)) : Prop :=
  ∃ x y, free_spot table x y ∧ (x, y) ∉ occupied

def second_player_can_move (table : Table) (occupied: set (ℕ × ℕ)) : Prop :=
  ∃ x y, free_spot table x y ∧ (x, y) ∉ occupied

def first_player_wins (table : Table) (occupied: set (ℕ × ℕ)) : Prop :=
  ¬ second_player_can_move table occupied

theorem first_player_always_wins (table : Table) (occupied: set (ℕ × ℕ)) :
  (first_player_can_move table occupied) → first_player_wins table occupied :=
by
  sorry

end first_player_always_wins_l744_744361


namespace toms_weekly_income_l744_744351

variable (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def daily_crabs := num_buckets * crabs_per_bucket
def daily_income := daily_crabs * price_per_crab
def weekly_income := daily_income * days_per_week

theorem toms_weekly_income 
  (h1 : num_buckets = 8)
  (h2 : crabs_per_bucket = 12)
  (h3 : price_per_crab = 5)
  (h4 : days_per_week = 7) :
  weekly_income num_buckets crabs_per_bucket price_per_crab days_per_week = 3360 :=
by
  sorry

end toms_weekly_income_l744_744351


namespace range_of_m_in_third_quadrant_l744_744132

noncomputable def z (m : ℝ) := m * (3 + complex.I) - (2 + complex.I)

theorem range_of_m_in_third_quadrant : 
  ∀ (m : ℝ), (im (z m) < 0 ∧ re (z m) < 0) → m < (2 / 3) :=
by
  intro m
  sorry

end range_of_m_in_third_quadrant_l744_744132


namespace exists_natural_number_a_l744_744261

open Set

noncomputable def solution_set_a (a : ℕ) : Set ℝ := {x : ℝ | a * x^2 + 2 * abs (x - a) - 20 < 0}
def solution_set_ineq1 : Set ℝ := {x : ℝ | x^2 + x - 2 < 0}
def solution_set_ineq2 : Set ℝ := {x : ℝ | abs(2 * x - 1) < x + 2}

theorem exists_natural_number_a (a : ℕ) :
  a ∈ {1, 2, 3, 4, 5, 6, 7} ∧ 
  solution_set_ineq1 ⊆ solution_set_a a ∧ 
  solution_set_ineq2 ⊆ solution_set_a a :=
begin
  sorry,
end

end exists_natural_number_a_l744_744261


namespace scalene_triangle_minimum_altitude_l744_744048

theorem scalene_triangle_minimum_altitude (a b c : ℕ) (h : ℕ) 
  (h₁ : a ≠ b ∧ b ≠ c ∧ c ≠ a) -- scalene condition
  (h₂ : ∃ k : ℕ, ∃ m : ℕ, k * m = a ∧ m = 6) -- first altitude condition
  (h₃ : ∃ k : ℕ, ∃ n : ℕ, k * n = b ∧ n = 8) -- second altitude condition
  (h₄ : c = (7 : ℕ) * b / (3 : ℕ)) -- third side condition given inequalities and area relations
  : h = 2 := 
sorry

end scalene_triangle_minimum_altitude_l744_744048


namespace three_digit_integer_count_l744_744556

-- Lean equivalent of defining the problem conditions
def digits : Set Nat := {1, 5, 8}
def num_three_digit_integers : Nat := 
  (2 * (digits.card - 1))!

-- Lean statement expressing the proof problem
theorem three_digit_integer_count (h : digits.card = 3) : num_three_digit_integers = 6 :=
sorry

end three_digit_integer_count_l744_744556


namespace find_x_l744_744616

-- Definitions based on the given conditions
variables {B C D : Type} (A : Type)

-- Angles in degrees
variables (angle_ACD : ℝ := 100)
variables (angle_ADB : ℝ)
variables (angle_ABD : ℝ := 2 * angle_ADB)
variables (angle_DAC : ℝ)
variables (angle_BAC : ℝ := angle_DAC)
variables (angle_ACB : ℝ := 180 - angle_ACD)
variables (y : ℝ := angle_DAC)
variables (x : ℝ := angle_ADB)

-- The proof statement
theorem find_x (h1 : B = C) (h2 : C = D) 
    (h3: angle_ACD = 100) 
    (h4: angle_ADB = x) 
    (h5: angle_ABD = 2 * x) 
    (h6: angle_DAC = angle_BAC) 
    (h7: angle_DAC = y)
    : x = 20 :=
sorry

end find_x_l744_744616


namespace question_one_question_two_l744_744553

noncomputable theory

def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x

def S (n : ℕ) : ℝ := 3 * (n:ℝ)^2 - 2 * (n:ℝ)

def a (n : ℕ) : ℝ := if n = 1 then 1 else S n - S (n - 1)

def b (n : ℕ) : ℝ := 3 / (a n * a (n + 1))

def T (n : ℕ) : ℝ := 1 / 2 * (1 - 1 / (6 * (n:ℝ) + 1))

theorem question_one (n : ℕ) (h : n ≠ 0) : a n = 6 * (n:ℝ) - 5 :=
sorry

theorem question_two (m : ℕ) (h : m ≥ 9) : ∀ n : ℕ, n ≠ 0 → T n < m / 20 :=
sorry

end question_one_question_two_l744_744553


namespace find_divisor_l744_744862

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) 
  (h1 : dividend = 62976) 
  (h2 : quotient = 123) 
  (h3 : dividend = divisor * quotient) 
  : divisor = 512 := 
by
  sorry

end find_divisor_l744_744862


namespace mean_score_l744_744395

theorem mean_score (M SD : ℝ) (h₁ : 58 = M - 2 * SD) (h₂ : 98 = M + 3 * SD) : M = 74 :=
by
  sorry

end mean_score_l744_744395


namespace polygon_area_value_l744_744432

theorem polygon_area_value (R : ℝ) (hR : R > 0) :
  ∃ n : ℕ, (n > 2) ∧ (n * (Float.sin (360 / n) + Float.cos (180 / n)) = 8) ∧ n = 24 :=
by
  sorry

end polygon_area_value_l744_744432


namespace area_between_concentric_circles_l744_744720

theorem area_between_concentric_circles :
  (∀ C B : Point, ∀ r R : ℝ,
    0 < r ∧ 0 < R ∧ r < R ∧
    tangent C B r AD ∧
    chord AD = 24 ∧
    distance C A = 15)
  → (π * (15^2) - π * (9^2)) = 144 * π :=
by
  -- conditions
  intro C B r R h
  have hR: R = 15 := by sorry
  have hCB : distance C B = 9 := by sorry
  exact sorry

end area_between_concentric_circles_l744_744720


namespace two_pow_a_add_three_pow_b_eq_square_l744_744488

theorem two_pow_a_add_three_pow_b_eq_square (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) 
(h : 2 ^ a + 3 ^ b = n ^ 2) : (a = 4 ∧ b = 2) :=
sorry

end two_pow_a_add_three_pow_b_eq_square_l744_744488


namespace elementary_school_coats_l744_744679

theorem elementary_school_coats (total_coats : Nat) (high_school_coats : Nat) : 
  total_coats = 9437 → high_school_coats = 6922 → total_coats - high_school_coats = 2515 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end elementary_school_coats_l744_744679


namespace eliza_says_500_l744_744494

def skips_every_4th (start : ℕ) : ℕ → bool
| n := (n % 4 ≠ start % 4)

def alice_says (n : ℕ) : bool :=
n ≤ 500 ∧ skips_every_4th 4 n

def barbara_says (n : ℕ) : bool :=
n ≤ 500 ∧ ¬alice_says n ∧ skips_every_4th 1 n

def candice_says (n : ℕ) : bool :=
n ≤ 500 ∧ ¬alice_says n ∧ ¬barbara_says n ∧ skips_every_4th 2 n

def debbie_says (n : ℕ) : bool :=
n ≤ 500 ∧ ¬alice_says n ∧ ¬barbara_says n ∧ ¬candice_says n ∧ skips_every_4th 3 n

def eliza_says (n : ℕ) : bool :=
n ≤ 500 ∧ ¬alice_says n ∧ ¬barbara_says n ∧ ¬candice_says n ∧ ¬debbie_says n

theorem eliza_says_500 : eliza_says 500 :=
by {
  simp only [eliza_says, alice_says, barbara_says, candice_says, debbie_says, skips_every_4th],
  sorry
}

end eliza_says_500_l744_744494


namespace symmetry_of_graphs_l744_744320

def f (x : ℝ) : ℝ := 2 ^ x
def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem symmetry_of_graphs : ∀ x: ℝ, ∃ y : ℝ, f x = y ∧ g y = x :=
by
  sorry

end symmetry_of_graphs_l744_744320


namespace find_k_for_binomial_square_l744_744753

-- The goal is to show that there exists a value of k such that the given quadratic is a perfect square.
theorem find_k_for_binomial_square (k : ℝ) : (∃ b : ℝ, (x^2 - 10 * x + k = (x + b)^2)) ↔ (k = 25) := 
begin
  sorry
end

end find_k_for_binomial_square_l744_744753


namespace find_value_of_expression_l744_744892

open Polynomial

theorem find_value_of_expression
  {α β : ℝ}
  (h1 : α^2 - 5 * α + 6 = 0)
  (h2 : β^2 - 5 * β + 6 = 0) :
  3 * α^3 + 10 * β^4 = 2305 :=
by
  sorry

end find_value_of_expression_l744_744892


namespace sum_of_roots_l744_744095

theorem sum_of_roots (x : ℝ) (h : x + 49 / x = 14) : x + x = 14 :=
sorry

end sum_of_roots_l744_744095


namespace rope_rounds_proof_l744_744434

def number_of_rounds (radius1 radius2 : ℝ) (rounds2 : ℕ) : ℕ :=
  let length_of_rope := 2 * Real.pi * radius2 * rounds2
  let circumference1 := 2 * Real.pi * radius1
  nat.floor (length_of_rope / circumference1)

theorem rope_rounds_proof (radius1 radius2 : ℝ) (rounds2 : ℕ) (h1 : radius1 = 14)
  (h2 : radius2 = 20) (h3 : rounds2 = 49) : number_of_rounds radius1 radius2 rounds2 = 70 :=
by
  rw [h1, h2, h3]
  sorry

end rope_rounds_proof_l744_744434


namespace remainder_of_number_when_divided_by_89_l744_744768

theorem remainder_of_number_when_divided_by_89 : 
  ∃ (n : ℕ), (n = 347 * 101) ∧ (n % 89 = 70) :=
begin
  let n := 347 * 101,
  use n,
  split,
  { refl },
  { sorry },
end

end remainder_of_number_when_divided_by_89_l744_744768


namespace calculate_volume_of_prism_l744_744685

noncomputable def volume_of_prism (M N P l : ℝ) : ℝ :=
  (1 / (4 * l)) * real.sqrt ((N + M + P) * (N + P - M) * (N + M - P) * (M + P - N))

theorem calculate_volume_of_prism (M N P l : ℝ) (h_l : l ≠ 0) :
  volume_of_prism M N P l = (1 / (4 * l)) * real.sqrt ((N + M + P) * (N + P - M) * (N + M - P) * (M + P - N)) :=
begin
  sorry
end

end calculate_volume_of_prism_l744_744685


namespace meeting_point_closest_to_C_l744_744211

-- Define the problem as a theorem
theorem meeting_point_closest_to_C
  (s : ℝ) -- Hector's speed
  (d : ℝ := 27) -- Total loop distance
  (jane_speed_ratio : ℝ := 3) -- Jane’s speed is 3 times Hector’s speed
  (meeting_point : ℝ := d / 4) -- Distance where they meet from the start point
  (start_point : ℝ := 0) -- They start together at point 0 (assuming A)
  (points : list (ℝ × string) := [(0, "A"), (d / 4, "B"), (d / 2, "C"), (3 * d / 4, "D"), (d, "E")]) -- Points on the loop
  (closest_point : string := "C") : 
  (∃ t : ℝ, s * t + jane_speed_ratio * s * t = d) ∧
  (s * (d / (4 * s)) + jane_speed_ratio * s * (d / (4 * s)) = d) ∧
  points.nth_le 2 sorry = (meeting_point, closest_point) := 
sorry

end meeting_point_closest_to_C_l744_744211


namespace convex_polygon_area_lt_pi_div_4_l744_744479

theorem convex_polygon_area_lt_pi_div_4
  (P : Type*) [polygon P] (a : P → ℝ → Prop)
  (h_a :
    ∀ (vertex : P),
      ∃ (line : Π t, a vertex t),
      let segment := {x ∈ line | is_inside_polygon x}
      in measure length segment ≤ 1)
  :
  measure area P < π / 4 :=
sorry

end convex_polygon_area_lt_pi_div_4_l744_744479


namespace probability_same_heads_l744_744979

theorem probability_same_heads (k e : Fin 2) : 
  let outcomes := [(k, e)];
  (count (λ oe, oe.1 = oe.2) outcomes) / (card Fin 2 * card Fin 8) = 1 / 4 := sorry

end probability_same_heads_l744_744979


namespace carson_clawed_total_l744_744459

theorem carson_clawed_total :
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  wombats * wombat_claws + rheas * rhea_claws = 39 := by
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  show wombats * wombat_claws + rheas * rhea_claws = 39
  sorry

end carson_clawed_total_l744_744459


namespace tyrone_gave_15_marbles_l744_744736

variables (x : ℕ)

-- Define initial conditions for Tyrone and Eric
def initial_tyrone := 120
def initial_eric := 20

-- Define the condition after giving marbles
def condition_after_giving (x : ℕ) := 120 - x = 3 * (20 + x)

theorem tyrone_gave_15_marbles (x : ℕ) : condition_after_giving x → x = 15 :=
by
  intro h
  sorry

end tyrone_gave_15_marbles_l744_744736


namespace ratio_of_sides_l744_744807

theorem ratio_of_sides (a b : ℝ) 
  (h1 : a < b)
  (h2 : a + b - sqrt (a^2 + b^2) = (1/3) * b)
  (h3 : ∀ k : ℝ, k > 1 → a / b = (a * k) / (b * k)) : 
  a / b = 5 / 12 :=
sorry

end ratio_of_sides_l744_744807


namespace Problem_statement_l744_744637

open EuclideanGeometry

noncomputable def Problem : Prop :=
  ∀ (A B C D E F : Point) (l1 l2 : Line),
  Parallelogram ABCD ∧
  ∠ABC = 100 ∧
  dist A B = 20 ∧
  dist B C = 14 ∧
  D ∈ l1 ∧
  E ∈ l1 ∧
  F ∈ l2 ∧
  D ≠ E ∧
  dist D E = 6 ∧
  Meq (LineSegment B E) l2 ∧ 
  Meq (LineSegment A D) l2 → 
  dist F D = 4.2

theorem Problem_statement : Problem := by
  sorry

end Problem_statement_l744_744637


namespace sound_pressures_relationships_l744_744276

variables (p p0 p1 p2 p3 : ℝ)
  (Lp Lpg Lph Lpe : ℝ)

-- The definitions based on the conditions
def sound_pressure_level (p : ℝ) (p0 : ℝ) : ℝ := 20 * (Real.log10 (p / p0))

-- Given conditions
axiom p0_gt_zero : p0 > 0

axiom gasoline_car_levels : 60 ≤ Lpg ∧ Lpg ≤ 90
axiom hybrid_car_levels : 50 ≤ Lph ∧ Lph ≤ 60
axiom electric_car_level : Lpe = 40

axiom gasoline_car_sound_pressure : Lpg = sound_pressure_level p1 p0
axiom hybrid_car_sound_pressure : Lph = sound_pressure_level p2 p0
axiom electric_car_sound_pressure : Lpe = sound_pressure_level p3 p0

-- The proof to be derived
theorem sound_pressures_relationships (p0_gt_zero : p0 > 0)
  (gasoline_car_levels : 60 ≤ Lpg ∧ Lpg ≤ 90)
  (hybrid_car_levels : 50 ≤ Lph ∧ Lph ≤ 60)
  (electric_car_level : Lpe = 40)
  (gasoline_car_sound_pressure : Lpg = sound_pressure_level p1 p0)
  (hybrid_car_sound_pressure : Lph = sound_pressure_level p2 p0)
  (electric_car_sound_pressure : Lpe = sound_pressure_level p3 p0) :
  p1 ≥ p2 ∧ p3 = 100 * p0 ∧ p1 ≤ 100 * p2 :=
by
  sorry

end sound_pressures_relationships_l744_744276


namespace solve_system_evaluate_expression_l744_744405

-- Definitions to set up the problem
def system_of_equations (x y : ℝ) : Prop := 
  (2 * x + y = 4) ∧ (x + 2 * y = 5)

def expression (x : ℝ) : ℝ := 
  let term1 := 1 / x
  let term2 := (x ^ 2 - 2 * x + 1) / (x ^ 2 - 1)
  let divisor := 1 / (x + 1)
  (term1 + term2) / divisor

-- Theorem statements
theorem solve_system :
  ∃ x y : ℝ, system_of_equations x y ∧ x = 1 ∧ y = 2 :=
by { use [1, 2], simp [system_of_equations] }

theorem evaluate_expression : expression (-2) = -5 / 2 :=
by { simp [expression], norm_num }

end solve_system_evaluate_expression_l744_744405


namespace exists_N_with_term_containing_digit_5_l744_744884

noncomputable def sequence : ℕ → ℕ := λ n, ⌊ n^(2018 / 2017) ⌋

theorem exists_N_with_term_containing_digit_5 :
  ∃ N : ℕ, ∀ k : ℕ, ∃ i : ℕ, i < N ∧ (sequence (k + i)).toString.contains '5' :=
sorry

end exists_N_with_term_containing_digit_5_l744_744884


namespace find_a2_l744_744115

variable (a : ℕ → ℤ)

-- Conditions
def is_arithmetic_sequence (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (x y z : ℤ) : Prop :=
  y * y = x * z

-- Specific condition for the problem
axiom h_arithmetic : is_arithmetic_sequence a 2
axiom h_geometric : is_geometric_sequence (a 1 + 2) (a 3 + 6) (a 4 + 8)

-- Theorem to prove
theorem find_a2 : a 1 + 2 = -8 := 
sorry

-- We assert that the value of a_2 must satisfy the given conditions

end find_a2_l744_744115


namespace problem1_problem2_l744_744197

theorem problem1 (c : ℝ) (hc : 0 < c) :
  ∀ {A B : ℝ × ℝ}, (A.snd = A.fst^2) ∧ (B.snd = B.fst^2) →
  let E := (0, c) in
  let P := ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2) in
  let Q := (P.fst, -c) in
  (∃ k : ℝ, (E.snd = k * E.fst + c) ∧
             A.fst * B.fst = -c ∧
             A.fst + B.fst = k) →
  (∃ l : ℝ, (Q.snd = 2 * l * Q.fst - l^2)) :=
begin
  sorry
end

theorem problem2 (c : ℝ) (hc : c = 1/4) :
  ∀ {A B : ℝ × ℝ},
  let E := (0, c) in
  let line := λ y : ℝ, -y in
  let O := (0, 0) in
  let C := (-(1/(4*A.fst)), -c) in
  let D := (-(1/(4*B.fst)), -c) in
  let M := (0, c / 2) in
  let N := (0, -3*c / 2) in
  A.snd = A.fst^2 ∧ B.snd = B.fst^2 →
  (∃ M : ℝ × ℝ, M = (0, 1/4) ∨ M = (0, -3/4)) :=
begin
  sorry
end

end problem1_problem2_l744_744197


namespace hannah_trip_time_ratio_l744_744927

theorem hannah_trip_time_ratio 
  (u : ℝ) -- Speed on the first trip in miles per hour.
  (u_pos : u > 0) -- Speed should be positive.
  (t1 t2 : ℝ) -- Time taken for the first and second trip respectively.
  (h_t1 : t1 = 30 / u) -- Time for the first trip.
  (h_t2 : t2 = 150 / (4 * u)) -- Time for the second trip.
  : t2 / t1 = 1.25 := by
  sorry

end hannah_trip_time_ratio_l744_744927


namespace semicircle_inequality_l744_744083

-- Define the points on the semicircle
variables (A B C D E : ℝ)
-- Define the length function
def length (X Y : ℝ) : ℝ := abs (X - Y)

-- This is the main theorem statement
theorem semicircle_inequality {A B C D E : ℝ} :
  length A B ^ 2 + length B C ^ 2 + length C D ^ 2 + length D E ^ 2 +
  length A B * length B C * length C D + length B C * length C D * length D E < 4 :=
sorry

end semicircle_inequality_l744_744083


namespace incenter_coordinates_l744_744536

theorem incenter_coordinates (x1 y1 x2 y2 x3 y3 a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let I_x := (a * x1 + b * x2 + c * x3) / (a + b + c)
  let I_y := (a * y1 + b * y2 + c * y3) / (a + b + c)
  in (I_x, I_y) = (a * x1 + b * x2 + c * x3) / (a + b + c), (a * y1 + b * y2 + c * y3) / (a + b + c) := by
  sorry

end incenter_coordinates_l744_744536


namespace function_range_2sin_l744_744082

theorem function_range_2sin (x : ℝ) (h : -π/6 < x ∧ x < π/6) :
  0 < 2 * sin (2 * x + π/3) ∧ 2 * sin (2 * x + π/3) ≤ 2 :=
by
  sorry

end function_range_2sin_l744_744082


namespace sound_pressure_proof_l744_744271

noncomputable theory

def sound_pressure_level (p p0 : ℝ) : ℝ :=
  20 * real.log10 (p / p0)

variables (p0 : ℝ) (p0_pos : 0 < p0)
variables (p1 p2 p3 : ℝ)

def gasoline_car (Lp : ℝ) : Prop :=
  60 <= Lp ∧ Lp <= 90

def hybrid_car (Lp : ℝ) : Prop :=
  50 <= Lp ∧ Lp <= 60

def electric_car (Lp : ℝ) : Prop :=
  Lp = 40

theorem sound_pressure_proof :
  gasoline_car (sound_pressure_level p1 p0) ∧
  hybrid_car (sound_pressure_level p2 p0) ∧
  electric_car (sound_pressure_level p3 p0) →
  (p1 ≥ p2) ∧ (¬ (p2 > 10 * p3)) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) :=
begin
  sorry
end

end sound_pressure_proof_l744_744271


namespace number_line_problem_l744_744296

theorem number_line_problem (A : ℝ) (h : A = 2) (B : ℝ) (hB : abs (B - A) = 3) : B = -1 ∨ B = 5 :=
by
  rw h at hB
  sorry

end number_line_problem_l744_744296


namespace find_prices_optimal_plan_l744_744027

/-
Part 1: Find the price per plant for types $A$ and $B$.
-/

def typeA_price : ℕ := 30
def typeB_price : ℕ := 15
def total_cost1 : ℕ := 675
def total_cost2 : ℕ := 265

theorem find_prices (x y : ℕ) : 30 * x + 15 * y = 675 ∧ 12 * x + 5 * y = 265 → x = 20 ∧ y = 5 :=
by
  intros h,
  sorry

/-
Part 2: Design the most cost-effective plan with given constraints.
-/

def total_plants : ℕ := 31
def cost_per_A : ℕ := 20
def cost_per_B : ℕ := 5
def plan_cost (m : ℕ) : ℕ := 15 * m + 155

theorem optimal_plan (m : ℕ) : 31 - m < 2 * m → m > 31 / 3 → plan_cost 11 = 320 :=
by
  intros h1 h2,
  sorry

end find_prices_optimal_plan_l744_744027


namespace percent_defective_shipped_l744_744619

theorem percent_defective_shipped
  (P_d : ℝ) (P_s : ℝ)
  (hP_d : P_d = 0.1)
  (hP_s : P_s = 0.05) :
  P_d * P_s = 0.005 :=
by
  sorry

end percent_defective_shipped_l744_744619


namespace num_valid_n_values_l744_744609

theorem num_valid_n_values :
  let angles := (D, E, F) // def of angles in triangle
  let side_lengths := (3*n + 10, 2*n + 8, n + 15) // side lengths given in conditions
  ∃ n : ℕ, angles.D > angles.E > angles.F → 
           (3*n + 10) > (2*n + 8) → 
           (2*n + 8) > (n + 15) → 
           ∀ (8 ≤ n) (n ≤ 100),  
           (set.Icc 8 100).card := 93 :=
begin
  -- Proof goes here
  sorry,
end

end num_valid_n_values_l744_744609


namespace find_original_price_l744_744345

-- Define the original price P
variable (P : ℝ)

-- Define the conditions as per the given problem
def revenue_equation (P : ℝ) : Prop :=
  820 = (10 * 0.60 * P) + (20 * 0.85 * P) + (18 * P)

-- Prove that the revenue equation implies P = 20
theorem find_original_price (P : ℝ) (h : revenue_equation P) : P = 20 :=
  by sorry

end find_original_price_l744_744345


namespace ratio_area_rectangle_to_square_l744_744325

variable (s : ℝ)
variable (area_square : ℝ := s^2)
variable (longer_side_rectangle : ℝ := 1.2 * s)
variable (shorter_side_rectangle : ℝ := 0.85 * s)
variable (area_rectangle : ℝ := longer_side_rectangle * shorter_side_rectangle)

theorem ratio_area_rectangle_to_square :
  area_rectangle / area_square = 51 / 50 := by
  sorry

end ratio_area_rectangle_to_square_l744_744325


namespace trains_clear_time_l744_744011

-- Given conditions
def length_of_train1 : ℕ := 120
def length_of_train2 : ℕ := 165
def speed_of_train1_kmh : ℕ := 80
def speed_of_train2_kmh : ℕ := 65

-- Derived parameters
def total_distance : ℕ := length_of_train1 + length_of_train2
def relative_speed_kmh : ℕ := speed_of_train1_kmh + speed_of_train2_kmh
def relative_speed_ms : ℝ := relative_speed_kmh * 1000 / 3600

-- Prove the time to be clear
theorem trains_clear_time : total_distance / relative_speed_ms ≈ 7.075 := 
  by sorry

end trains_clear_time_l744_744011


namespace max_value_l744_744401

open BigOperators

def max_sum (a : ℝ) (k : ℕ) (r : ℕ) (k_i : Fin r → ℕ) : ℝ :=
  ∑ i, a ^ k_i i

theorem max_value (k : ℕ) (a : ℝ) 
  (h₀ : a > 0) (h₁ : ∃ (r : ℕ) (k_i : Fin r → ℕ), (∑ i, k_i i = k) ∧ (r ≥ 1) ∧ (r ≤ k))
  : ∃ r (k_i : Fin r → ℕ), max_sum a k r k_i = max (k * a) (a ^ k) :=
sorry

end max_value_l744_744401


namespace estimated_black_pieces_eq_twelve_l744_744610

/-
In an opaque bag, there are a total of 20 chess pieces, including white and black ones. These chess pieces are identical except for their colors. After mixing the chess pieces in the bag, one piece is randomly drawn, the color is noted, and then the piece is put back into the bag. This process is repeated 100 times, and it is found that 60 times a black chess piece was drawn. Estimate the number of black chess pieces in the bag.
-/

-- Define the conditions
def total_chess_pieces : ℕ := 20
def total_draws : ℕ := 100
def black_chess_draws : ℕ := 60

-- Calculate the frequency of drawing a black piece
def frequency_black : ℚ := black_chess_draws / total_draws

-- Calculate the estimated number of black chess pieces
def estimated_black_chess_pieces : ℕ := (total_chess_pieces * frequency_black).toNat

-- Prove the estimation
theorem estimated_black_pieces_eq_twelve :
  estimated_black_chess_pieces = 12 := by
  unfold total_chess_pieces total_draws black_chess_draws frequency_black estimated_black_chess_pieces
  norm_num
  sorry

end estimated_black_pieces_eq_twelve_l744_744610


namespace period_of_function_is_pi_l744_744712

noncomputable def function_period : ℝ := sorry

theorem period_of_function_is_pi :
  (∀ x ∈ ℝ, function_period = (π : ℝ)) :=
sorry

end period_of_function_is_pi_l744_744712


namespace inequality_a_b_c_l744_744996

theorem inequality_a_b_c 
  (a b c : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_c : c > 0) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3 / 2 :=
by 
  sorry

end inequality_a_b_c_l744_744996


namespace part_I_part_II_l744_744638

-- Definitions
def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def a_seq (n : ℕ) : ℝ := 3 * n + 1
def b_seq (n : ℕ) : ℝ := 3 / ((3 * n + 1) * (3 * (n + 1) + 1))
def T_n (n : ℕ) : ℝ := ∑ i in finset.range n, b_seq i

-- Theorem 1
theorem part_I (n : ℕ) :
  0 < a_seq n ∧ (a_seq n)^2 + 3 * a_seq n = 6 * S_n a_seq n + 4 :=
sorry

-- Theorem 2
theorem part_II (n : ℕ) :
  T_n n = (1/4) - (1/(3 * n + 4)) :=
sorry

end part_I_part_II_l744_744638


namespace area_closed_figure_l744_744962

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin(a * x) + Real.cos(a * x)

theorem area_closed_figure (a : ℝ) (h : a > 0) :
  (∫ x in 0 .. (2 * Real.pi / a), f a x) = (∫ x in 0 .. (2 * Real.pi / a), a * Real.sin(a * x) + Real.cos(a * x)) :=
by
  sorry

end area_closed_figure_l744_744962


namespace robin_gum_pieces_l744_744404

-- Defining the conditions
def packages : ℕ := 9
def pieces_per_package : ℕ := 15
def total_pieces : ℕ := 135

-- Theorem statement
theorem robin_gum_pieces (h1 : packages = 9) (h2 : pieces_per_package = 15) : packages * pieces_per_package = total_pieces := by
  -- According to the problem, the correct answer is 135 pieces
  have h: 9 * 15 = 135 := by norm_num
  rw [h1, h2]
  exact h

end robin_gum_pieces_l744_744404


namespace fair_hair_percentage_gender_ratio_fair_hair_l744_744659

variable (E : ℝ) -- Total number of employees
variable (h_men_fair : ℝ := 0.05) -- 5% of men have fair hair
variable (h_women_fair : ℝ := 0.20) -- 20% of women have fair hair
variable (percent_women : ℝ := 0.35) -- 35% of employees are women
variable (percent_fair_hair_women : ℝ := 0.40) -- 40% of fair-haired employees are women

theorem fair_hair_percentage (E : ℝ) :
  let fair_hair_men := h_men_fair * (1 - percent_women) * E,
      fair_hair_women := h_women_fair * percent_women * E,
      total_fair_hair := fair_hair_men + fair_hair_women in
  total_fair_hair = 0.1025 * E := sorry

theorem gender_ratio_fair_hair (E : ℝ) :
  let fair_hair_men := h_men_fair * (1 - percent_women) * E,
      fair_hair_women := h_women_fair * percent_women * E,
      total_fair_hair := fair_hair_men + fair_hair_women in
  (fair_hair_men / total_fair_hair) / (fair_hair_women / total_fair_hair) = 3 / 2 := sorry


end fair_hair_percentage_gender_ratio_fair_hair_l744_744659


namespace time_for_a_and_b_together_l744_744591

variable (R_a R_b : ℝ)
variable (T_ab : ℝ)

-- Given conditions
def condition_1 : Prop := R_a = 3 * R_b
def condition_2 : Prop := R_a * 28 = 1  -- '1' denotes the entire work

-- Proof goal
theorem time_for_a_and_b_together (h1 : condition_1 R_a R_b) (h2 : condition_2 R_a) : T_ab = 21 := 
by
  sorry

end time_for_a_and_b_together_l744_744591


namespace solution_unique_l744_744090

def is_solution (x : ℝ) : Prop :=
  ⌊x * ⌊x⌋⌋ = 48

theorem solution_unique (x : ℝ) : is_solution x → x = -48 / 7 :=
by
  intro h
  -- Proof goes here
  sorry

end solution_unique_l744_744090


namespace two_colonies_same_time_l744_744006

def doubles_in_size_every_day (P : ℕ → ℕ) : Prop :=
∀ n, P (n + 1) = 2 * P n

def reaches_habitat_limit_in (f : ℕ → ℕ) (days limit : ℕ) : Prop :=
f days = limit

theorem two_colonies_same_time (P : ℕ → ℕ) (Q : ℕ → ℕ) (limit : ℕ) (days : ℕ)
  (h1 : doubles_in_size_every_day P)
  (h2 : reaches_habitat_limit_in P days limit)
  (h3 : ∀ n, Q n = 2 * P n) :
  reaches_habitat_limit_in Q days limit :=
sorry

end two_colonies_same_time_l744_744006


namespace prove_relationships_l744_744283

variables {p0 p1 p2 p3 : ℝ}
variable (h_p0_pos : p0 > 0)
variable (h_gas_car : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
variable (h_hybrid_car : 10^(5 / 2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)
variable (h_electric_car : p3 = 100 * p0)

theorem prove_relationships : 
  (p1 ≥ p2) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) := by
  split
  sorry

end prove_relationships_l744_744283


namespace area_of_triangle_PBQ_l744_744825

theorem area_of_triangle_PBQ : 
  ∀ (A B C D P Q : Point) (PD AP AQ QC : ℝ),
  is_square A B C D ∧
  P ∈ line_segment A D ∧
  Q ∈ line_segment A C ∧
  PD / AP = 4 / 1 ∧
  QC / AQ = 2 / 3 ∧
  (area_of_square A B C D = 25) →
  (area_of_triangle P B Q = 6.5) := 
begin
  sorry
end

end area_of_triangle_PBQ_l744_744825


namespace bc_is_one_area_of_triangle_l744_744237

-- Define a triangle with sides a, b, and c and corresponding angles A, B, and C.
structure Triangle :=
(a b c A B C : ℝ)

-- Assume the Law of Cosines condition is given.
axiom Law_of_Cosines_condition (T : Triangle) : (T.b^2 + T.c^2 - T.a^2) / (Real.cos T.A) = 2

-- Prove that bc = 1.
theorem bc_is_one (T : Triangle) (h : Law_of_Cosines_condition T) : T.b * T.c = 1 := 
by {
    sorry
}

-- Assume the given ratio condition.
axiom ratio_condition (T : Triangle) :
  (T.a * Real.cos T.B - T.b * Real.cos T.A) / (T.a * Real.cos T.B + T.b * Real.cos T.A)
  - (T.b / T.c) = 1

-- Prove that the area of the triangle is sqrt(3)/4.
theorem area_of_triangle (T : Triangle) (h1 : Law_of_Cosines_condition T) (h2 : ratio_condition T) :
  let S := 0.5 * T.b * T.c * Real.sin T.A
  in S = Real.sqrt 3 / 4 :=
by {
    sorry
}

end bc_is_one_area_of_triangle_l744_744237


namespace pocket_money_calculation_l744_744726

theorem pocket_money_calculation
  (a b c d e : ℝ)
  (h1 : (a + b + c + d + e) / 5 = 2300)
  (h2 : (a + b) / 2 = 3000)
  (h3 : (b + c) / 2 = 2100)
  (h4 : (c + d) / 2 = 2750)
  (h5 : a = b + 800) :
  d = 3900 :=
by
  sorry

end pocket_money_calculation_l744_744726


namespace prove_bc_prove_area_l744_744231

variable {a b c A B C : ℝ}

-- Given conditions as Lean definitions
def condition_cos_A : Prop := (b^2 + c^2 - a^2 = 2 * cos A)
def condition_bc : Prop := b * c = 1
def condition_trig_identity : Prop := (a * cos B - b * cos A) / (a * cos B + b * cos A) - b / c = 1

-- Definitions for the questions translated to Lean
def find_bc : Prop :=
  condition_cos_A → b * c = 1

def find_area (bc_value : Prop) : Prop :=
  bc_value →
  condition_trig_identity →
  1/2 * b * c * sin A = (Real.sqrt 3) / 4

-- Theorems to be proven
theorem prove_bc : find_bc :=
by
  intro h1
  sorry

theorem prove_area : find_area condition_bc :=
by
  intro h1 h2
  sorry

end prove_bc_prove_area_l744_744231


namespace business_school_student_count_l744_744026

noncomputable def number_of_students_in_business_school 
  (LawSchoolStudents : ℕ) 
  (SiblingPairs : ℕ) 
  (Probability : ℝ) : ℕ :=
  let B := 30 / (Probability * 800) in 
  if B = 5000 then 5000 else 0

theorem business_school_student_count :
  number_of_students_in_business_school 800 30 (7.5 * 10^(-5)) = 5000 := 
by
  -- proof skipped
  sorry

end business_school_student_count_l744_744026


namespace area_of_circle_l744_744950

open Real

/-- In a circle with center O, AB and CD are diameters such that AB ⊥ CD. 
Chord GH intersects AB at I. If GI = 1 and IH = 7, then the area of the circle equals 7π. -/
theorem area_of_circle
  (O A B C D G H I : Point)
  (r : ℝ)
  (circle_center : circle O r)
  (diam_AB : diameter O A B)
  (diam_CD : diameter O C D)
  (perp_AB_CD : perp AB CD)
  (chord_GH : chord G H)
  (intersect_I : intersect chord_GH AB I)
  (GI : ℝ)
  (IH : ℝ)
  (GI_eq : GI = 1)
  (IH_eq : IH = 7) :
  area circle_center = 7 * π :=
by
  -- Proof should be provided here
  sorry

end area_of_circle_l744_744950


namespace prove_hyperbola_eccentricity_l744_744102

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : Prop :=
  let c := real.sqrt (a^2 + b^2)
  let F1 := (-c, 0)
  let F2 := (c, 0)
  let asymptote := λ x, (b / a) * x
  let distance_F1_to_asymptote := (b * c) / real.sqrt (a^2 + b^2)
  let M := (-c, 2 * b)
  let MF1 := 2 * b
  let F1F2 := 2 * c
  let MF2 := real.sqrt (MF1^2 + F1F2^2)
  let OA_parallel_to_F2M := true
  let right_angle_triangle := (MF1^2 + F1F2^2 = MF2^2)
  let c_square_relation := 3 * c^2 = 4 * b^2
  let hyperbola_relation := c^2 = a^2 + b^2
  let simplified_relation := 3 * a^2 = b^2
  let e := c / a
  e = 2

theorem prove_hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  hyperbola_eccentricity a b h_a h_b :=
by
  sorry -- Proof omitted

end prove_hyperbola_eccentricity_l744_744102


namespace negation_equivalence_l744_744539

variable (Man Woman GoodDriver BadDriver : Type)

variable (isMan : Man → Prop) (isWoman : Woman → Prop)
variable (isGoodDriver : GoodDriver → Prop) (isBadDriver : BadDriver → Prop)

-- Conditions
axiom axiom1 : ∀ w : Woman, isGoodDriver w
axiom axiom2 : ∀ d : GoodDriver, isWoman d
axiom axiom3 : ∀ m : Man, isMan m → ¬isGoodDriver m
axiom axiom4 : ∀ m : Man, isMan m → isBadDriver m
axiom axiom5 : ∃ m : Man, isMan m ∧ isBadDriver m
axiom axiom6 : ∀ m : Man, isMan m → isGoodDriver m

-- Negation of the 6th statement
theorem negation_equivalence : ¬(∀ m : Man, isMan m → isGoodDriver m) ↔ ∃ m : Man, isMan m ∧ isBadDriver m := 
sorry

end negation_equivalence_l744_744539


namespace converse_even_power_divisible_l744_744171

theorem converse_even_power_divisible (n : ℕ) (h_even : ∀ (k : ℕ), n = 2 * k → (3^n + 63) % 72 = 0) :
  (3^n + 63) % 72 = 0 → ∃ (k : ℕ), n = 2 * k :=
by sorry

end converse_even_power_divisible_l744_744171


namespace monotonic_increasing_interval_l744_744142

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * Real.pi * x - Real.pi / 5)

theorem monotonic_increasing_interval (x : ℝ)
  (h1 : -3 / 20 < x) (h2 : x < 7 / 20) :
  f(x) is_strict_mono_in [-3 / 20, 7 / 20] ∧
  ¬(f(x) is_strict_mono_in (-1 / 5, 3 / 10)) ∧
  ¬(f(x) is_strict_mono_in (3 / 10, 4 / 5)) ∧
  ¬(f(x) is_strict_mono_in (3 / 20, 13 / 20)) :=
sorry

end monotonic_increasing_interval_l744_744142


namespace derivative_at_neg_one_l744_744594

noncomputable def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

theorem derivative_at_neg_one (a b c : ℝ) (h : (4 * a * 1^3 + 2 * b * 1) = 2) : 
  (4 * a * (-1)^3 + 2 * b * (-1)) = -2 := 
sorry

end derivative_at_neg_one_l744_744594


namespace tile_board_remainder_l744_744814

def num_ways_to_tile_8x1_board : ℕ := 14100 -- Total number of valid configurations for the problem.

theorem tile_board_remainder (N : ℕ) (modulus : ℕ) :
  let N := num_ways_to_tile_8x1_board,
  let modulus := 1000,
  N % modulus = 100 :=
by
  let N := 14100
  let modulus := 1000
  have answer : N % modulus = 100 := by sorry
  exact answer

end tile_board_remainder_l744_744814


namespace positive_difference_A_B_is_72_l744_744075

-- Define sequences A and B using given conditions
noncomputable def A : ℕ := (∑ n in finset.range 9, (2 * n + 2) * (2 * n + 3)) + 18
noncomputable def B : ℕ := ∑ n in finset.range 9, (2 * n + 1) * (2 * n + 2)

-- Define the positive difference between integers A and B
noncomputable def positive_difference_A_B : ℕ := if A ≥ B then A - B else B - A

-- Proof statement: The positive difference between A and B is 72
theorem positive_difference_A_B_is_72 : positive_difference_A_B = 72 :=
sorry

end positive_difference_A_B_is_72_l744_744075


namespace find_x_in_sample_data_l744_744050

theorem find_x_in_sample_data :
  ∃ x : ℕ, 
  let data := [13, 14, 19, x, 23, 27, 28, 32], 
      sorted_data := data.sorted in
  sorted_data[3] + sorted_data[4] = 44 ∧ sorted_data[3] = x ∧ x = 21 := 
by
  sorry

end find_x_in_sample_data_l744_744050


namespace volume_of_cylinder_in_pyramid_l744_744045

theorem volume_of_cylinder_in_pyramid
  (a α : ℝ)
  (sin_alpha : ℝ := Real.sin α)
  (tan_alpha : ℝ := Real.tan α)
  (sin_pi_four_alpha : ℝ := Real.sin (Real.pi / 4 + α))
  (sqrt_two : ℝ := Real.sqrt 2) :
  (π * a^3 * sqrt_two * (Real.sin (2 * α))^3) / (128 * sin_pi_four_alpha^3) =
  (π * a^3 * sqrt_two * (Real.sin (2 * α))^3 / (128 * sin_pi_four_alpha^3)) :=
by
  sorry

end volume_of_cylinder_in_pyramid_l744_744045


namespace investment_rate_l744_744443

theorem investment_rate (total_investment : ℝ) (invest1 : ℝ) (rate1 : ℝ) (invest2 : ℝ) (rate2 : ℝ) (desired_income : ℝ) (remaining_investment : ℝ) (remaining_rate : ℝ) : 
( total_investment = 12000 ∧ invest1 = 5000 ∧ rate1 = 0.06 ∧ invest2 = 4000 ∧ rate2 = 0.035 ∧ desired_income = 700 ∧ remaining_investment = 3000 ) → remaining_rate = 0.0867 :=
by
  sorry

end investment_rate_l744_744443


namespace sum_of_roots_l744_744972

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - 2016 * x + 2015

theorem sum_of_roots (a b c : ℝ) (h1 : f a = c) (h2 : f b = c) (h3 : a ≠ b) :
  a + b = 2016 :=
by
  sorry

end sum_of_roots_l744_744972


namespace close_class_and_students_still_distinct_l744_744954

theorem close_class_and_students_still_distinct 
(n : ℕ) 
(classes : Fin n → Fin n → Prop) 
(h_diff : ∀ i j : Fin n, i ≠ j → ∃ k : Fin n, classes i k ≠ classes j k) : 
∃ k : Fin n, ∀ i j : Fin n, i ≠ j → ∃ m : Fin (n-1), classes i (Fin.cast m) ≠ classes j (Fin.cast m) :=
by
  sorry

end close_class_and_students_still_distinct_l744_744954


namespace general_terms_and_sum_l744_744123

theorem general_terms_and_sum (a_n b_n : ℕ → ℤ) (n : ℕ) :
  (a_n 1 = 3) → (a_n 4 = 24) →
  (b_n 1 = 1) → (b_n 4 = -8) →
  (∀ n, ∃ d : ℤ, a_n + b_n = a_n 1 + b_n 1 + d * (n - 1)) →
  (∀ n, a_n = 3 * 2^(n - 1)) ∧
  (∀ n, b_n = 4*n - 3 * 2^(n - 1)) ∧
  (∀ n, ∑ i in finset.range (n + 1), b_n i = 2*n^2 + 2*n + 3 - 3*2^n) :=
by
  sorry

end general_terms_and_sum_l744_744123


namespace pseudo_prime_looking_count_l744_744832

def pseudo_prime_looking (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Prime n ∧ ¬ (2 ∣ n) ∧ ¬ (3 ∣ n) ∧ ¬ (7 ∣ n) ∧ ¬ (11 ∣ n)

def count_pseudo_prime_looking_below (limit : ℕ) : ℕ :=
  (Finset.range limit).filter pseudo_prime_looking |>.card

theorem pseudo_prime_looking_count :
  count_pseudo_prime_looking_below 500 = 34 := 
sorry

end pseudo_prime_looking_count_l744_744832


namespace P_over_P_neg_one_l744_744168

theorem P_over_P_neg_one :
  let f (x : ℂ) := x ^ 2007 + 29 * x ^ 2006 + 1,
  ∃ (r : Fin 2007 → ℂ), (∀ j, f (r j) = 0) ∧ Function.Injective r →
  ∃ P : ℂ → ℂ, (∀ j, P (r j + (r j)⁻¹) = 0) ∧ P.degree = 2007 →
  P 1 / P (-1) = 1 := by
  sorry

end P_over_P_neg_one_l744_744168


namespace largest_power_2010_divides_factorial_l744_744463

theorem largest_power_2010_divides_factorial : 
  let k := ∑ i in Nat.range (Nat.log 67 2010 + 1), Nat.div (2010) (67^i)
  k = 30 := 
by
  sorry

end largest_power_2010_divides_factorial_l744_744463


namespace gasoline_car_p_ge_hybrid_electric_car_p_gasoline_car_p_le_100_hybrid_car_p_l744_744280

noncomputable def sound_pressure_level (p p0 : ℝ) : ℝ :=
20 * real.log10 (p / p0)

variables {p0 p1 p2 p3 : ℝ} (h_p0 : p0 > 0)
(h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
(h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)
(h_p3 : p3 = 100 * p0)

theorem gasoline_car_p_ge_hybrid (h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
                                (h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0)  : p1 ≥ p2 :=
sorry

theorem electric_car_p (h_p3 : p3 = 100 * p0) : p3 = 100 * p0 :=
sorry

theorem gasoline_car_p_le_100_hybrid_car_p (h_p1 : 1000 * p0 ≤ p1 ∧ p1 ≤ 10^(9/2) * p0)
                                           (h_p2 : 10^(5/2) * p0 ≤ p2 ∧ p2 ≤ 1000 * p0) : p1 ≤ 100 * p2 :=
sorry

#check gasoline_car_p_ge_hybrid
#check electric_car_p
#check gasoline_car_p_le_100_hybrid_car_p

end gasoline_car_p_ge_hybrid_electric_car_p_gasoline_car_p_le_100_hybrid_car_p_l744_744280


namespace total_weight_of_ripe_fruits_correct_l744_744087

-- Definitions based on conditions
def total_apples : ℕ := 14
def total_pears : ℕ := 10
def total_lemons : ℕ := 5

def ripe_apple_weight : ℕ := 150
def ripe_pear_weight : ℕ := 200
def ripe_lemon_weight : ℕ := 100

def unripe_apples : ℕ := 6
def unripe_pears : ℕ := 4
def unripe_lemons : ℕ := 2

def total_weight_of_ripe_fruits : ℕ :=
  (total_apples - unripe_apples) * ripe_apple_weight +
  (total_pears - unripe_pears) * ripe_pear_weight +
  (total_lemons - unripe_lemons) * ripe_lemon_weight

theorem total_weight_of_ripe_fruits_correct :
  total_weight_of_ripe_fruits = 2700 :=
by
  -- proof goes here (use sorry to skip the actual proof)
  sorry

end total_weight_of_ripe_fruits_correct_l744_744087


namespace integer_conditions_satisfy_eq_l744_744474

theorem integer_conditions_satisfy_eq (
  a b c : ℤ 
) : (a > b ∧ b = c → (a * (a - b) + b * (b - c) + c * (c - a) = 2)) ∧
    (¬(a = b - 1 ∧ b = c - 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a = c + 1 ∧ b = a + 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a = c ∧ b - 2 = c) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a + b + c = 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) :=
by
sorry

end integer_conditions_satisfy_eq_l744_744474


namespace semicircle_chord_length_l744_744390

-- Assuming a semicircle with a certain radius, let's define the main conditions
def semicircle_area (r : ℝ) : ℝ :=
  (1 / 2) * Real.pi * r^2

def remaining_area (r : ℝ) : ℝ :=
  semicircle_area r - 2 * semicircle_area (r / 2)

theorem semicircle_chord_length (r : ℝ) (h : remaining_area r = 16 * Real.pi^3) : 
  2 * (r / 2) * Real.sqrt 2 = 32 :=
by
  sorry

end semicircle_chord_length_l744_744390


namespace value_of_f_at_7_l744_744932

def f (x : ℝ) : ℝ := (5 * x + 1) / (x - 1)

theorem value_of_f_at_7 : f 7 = 6 := by
  sorry

end value_of_f_at_7_l744_744932


namespace factorial_base_a5_l744_744705

theorem factorial_base_a5 (a : ℕ → ℕ) (n : ℕ) (h0 : 0 ≤ a n ∧ a n ≤ n)
  (h1 : 1050 = ∑ i in Finset.range (n + 1), a i * Nat.factorial i) : a 5 = 2 :=
sorry

end factorial_base_a5_l744_744705


namespace f_1994_of_4_l744_744135

noncomputable def f (x : ℚ) : ℚ := (2 + x) / (2 - 2 * x)

def f_n (n : ℕ) (x : ℚ) : ℚ := 
  Nat.recOn n x (λ _ y => f y)

theorem f_1994_of_4 : f_n 1994 4 = 1 / 4 := by
  sorry

end f_1994_of_4_l744_744135


namespace negation_of_existence_l744_744704

variable (x : ℝ)

theorem negation_of_existence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2 * x₀ - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by sorry

end negation_of_existence_l744_744704


namespace probability_miss_at_least_once_l744_744713
-- Importing the entirety of Mathlib

-- Defining the conditions and question
variable (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1)

-- The main statement for the proof problem
theorem probability_miss_at_least_once (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1) : P ≤ 1 → 0 ≤ P ∧ 1 - P^3 ≥ 0 := 
by
  sorry

end probability_miss_at_least_once_l744_744713


namespace find_f_value_l744_744144

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x + 1

theorem find_f_value : f 2019 + f (-2019) = 2 :=
by
  sorry

end find_f_value_l744_744144


namespace sum_of_first_13_terms_of_sequence_l744_744110

theorem sum_of_first_13_terms_of_sequence 
  (a : ℕ → ℝ)
  (h1 : a 1 = -13)
  (h2 : a 6 + a 8 = -2)
  (h3 : ∀ n ≥ 2, a (n - 1) = 2 * a n - a (n + 1)) :
  (Finset.range 13).sum (λ n, 1 / (a n * a (n + 1))) = -1 / 13 := 
sorry

end sum_of_first_13_terms_of_sequence_l744_744110


namespace area_of_right_triangle_l744_744047

theorem area_of_right_triangle (m k : ℝ) (hm : 0 < m) (hk : 0 < k) : 
  ∃ A : ℝ, A = (k^2) / (2 * m) :=
by
  sorry

end area_of_right_triangle_l744_744047


namespace min_C2_minus_D2_l744_744999

theorem min_C2_minus_D2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  let C := sqrt (x + 3) + sqrt (y + 6) + sqrt (z + 12),
      D := sqrt (x + 2) + sqrt (y + 2) + sqrt (z + 2)
  in C^2 - D^2 ≥ 36 :=
sorry

end min_C2_minus_D2_l744_744999


namespace new_median_l744_744419

def initial_collection : Multiset ℕ := {3, 3, 4, 5, 8}
def new_element : ℕ := 9

theorem new_median (initial_mode_unique : multiset.mode initial_collection = some 3)
  (initial_median : multiset.median initial_collection = 4)
  (initial_mean : multiset.mean initial_collection = 4.6) :
  multiset.median (initial_collection.add new_element) = 4.5 :=
by
  -- Placeholder for proof
  sorry

end new_median_l744_744419


namespace target_more_tools_l744_744740

theorem target_more_tools (walmart_tools : ℕ) (target_tools : ℕ) (walmart_tools_is_6 : walmart_tools = 6) (target_tools_is_11 : target_tools = 11) :
  target_tools - walmart_tools = 5 :=
by
  rw [walmart_tools_is_6, target_tools_is_11]
  exact rfl

end target_more_tools_l744_744740


namespace interesting_quadruples_count_l744_744471

-- Define the conditions and the final proof statement
theorem interesting_quadruples_count : 
    ∃ (quadruples : Finset (ℤ × ℤ × ℤ × ℤ)), 
    (∀ q ∈ quadruples, let (a, b, c, d) := q in 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 12 ∧ a + d > b + c) ∧
    quadruples.card = 200 :=
begin
  sorry
end

end interesting_quadruples_count_l744_744471


namespace bc_eq_one_area_of_triangle_l744_744249

variable (a b c A B : ℝ)

-- Conditions
def condition_1 : Prop := (b^2 + c^2 - a^2) / (Real.cos A) = 2
def condition_2 : Prop := (a * (Real.cos B) - b * (Real.cos A)) / (a * (Real.cos B) + b * (Real.cos A)) - b / c = 1

-- Equivalent proof problems
theorem bc_eq_one (h1 : condition_1 a b c A) : b * c = 1 := 
by 
  sorry

theorem area_of_triangle (h2 : condition_2 a b c A B) : (1/2) * b * c * Real.sin A = (Real.sqrt 3) / 4 := 
by 
  sorry

end bc_eq_one_area_of_triangle_l744_744249


namespace boundary_of_bolded_figure_l744_744799

noncomputable def boundary_length (width length : ℝ) : ℝ :=
  let w := width in
  let l := length in
  let segment_length := l / 3 in
  let arc_circumference := 2 * Real.pi * segment_length / 4 in
  let total_arc_length := 4 * arc_circumference in
  let total_segment_length := 4 * w in
  total_segment_length + total_arc_length

theorem boundary_of_bolded_figure (w l : ℝ) 
  (h_area : w * l = 96) 
  (h_ratio : l = 4 * w) : 
  boundary_length w l = Real.sqrt 6 * (8 + 16 * Real.pi / 3) :=
by
  sorry

end boundary_of_bolded_figure_l744_744799


namespace card_distribution_l744_744762

-- Definitions of the total cards and distribution rules
def total_cards : ℕ := 363

def ratio_xiaoming_xiaohua (k : ℕ) : Prop := ∃ x y, x = 7 * k ∧ y = 6 * k
def ratio_xiaogang_xiaoming (m : ℕ) : Prop := ∃ x z, z = 8 * m ∧ x = 5 * m

-- Final values to prove
def xiaoming_cards : ℕ := 105
def xiaohua_cards : ℕ := 90
def xiaogang_cards : ℕ := 168

-- The proof statement
theorem card_distribution (x y z k m : ℕ) 
  (hk : total_cards = 7 * k + 6 * k + 8 * m)
  (hx : ratio_xiaoming_xiaohua k)
  (hz : ratio_xiaogang_xiaoming m) :
  x = xiaoming_cards ∧ y = xiaohua_cards ∧ z = xiaogang_cards :=
by
  -- Placeholder for the proof
  sorry

end card_distribution_l744_744762


namespace minimum_waves_to_21_l744_744326

def initial_flowers : ℕ := 3
def target_flowers : ℕ := 21

-- The operations: upward decreases open flowers by 1, downward doubles open flowers.
def upward_wave (n : ℕ) : ℕ := n - 1
def downward_wave (n : ℕ) : ℕ := n * 2

theorem minimum_waves_to_21 (waves : ℕ) (h1 : waves ≥ 0) : 
  (∃ seq : list (ℕ → ℕ), seq.length = waves ∧ 
    list.foldl (λ n f, f n) initial_flowers seq = target_flowers) :=
begin
  sorry
end

end minimum_waves_to_21_l744_744326


namespace polynomial_binomial_square_l744_744586

theorem polynomial_binomial_square (b : ℚ) :
  (∃ c : ℚ, (3 * polynomial.X + polynomial.C c)^2 = 9 * polynomial.X^2 + 27 * polynomial.X + polynomial.C b) →
  b = 81 / 4 :=
by
  intro h
  rcases h with ⟨c, hc⟩
  have : 6 * c = 27 := by sorry -- This corresponds to solving 6c = 27
  have : c = 9 / 2 := by sorry -- This follows from the above
  have : b = (9 / 2)^2 := by sorry -- This follows from substituting back c and expanding
  simp [this]

end polynomial_binomial_square_l744_744586


namespace total_pencils_correct_l744_744819

def initial_pencils : ℕ := 245
def added_pencils : ℕ := 758
def total_pencils : ℕ := initial_pencils + added_pencils

theorem total_pencils_correct : total_pencils = 1003 := 
by
  sorry

end total_pencils_correct_l744_744819


namespace postage_cost_for_5p3_ounces_l744_744433

def postage_cost (weight : ℝ) : ℝ :=
  if weight <= 1 then 35 else 35 + (weight - 1) * 25

theorem postage_cost_for_5p3_ounces :
  postage_cost 5.3 = 142.5 :=
by
  sorry

end postage_cost_for_5p3_ounces_l744_744433


namespace husband_weekly_saving_l744_744424

variable (H : ℕ)

-- conditions
def weekly_wife : ℕ := 225
def months : ℕ := 6
def weeks_per_month : ℕ := 4
def weeks := months * weeks_per_month
def amount_per_child : ℕ := 1680
def num_children : ℕ := 4

-- total savings calculation
def total_saving : ℕ := weeks * H + weeks * weekly_wife

-- half of total savings divided among children
def half_savings_div_by_children : ℕ := num_children * amount_per_child

-- proof statement
theorem husband_weekly_saving : H = 335 :=
by
  let total_children_saving := half_savings_div_by_children
  have half_saving : ℕ := total_children_saving 
  have total_saving_eq : total_saving = 2 * total_children_saving := sorry
  have total_saving_eq_simplified : weeks * H + weeks * weekly_wife = 13440 := sorry
  have H_eq : H = 335 := sorry
  exact H_eq

end husband_weekly_saving_l744_744424


namespace ducks_snails_l744_744038

noncomputable def total_snails_found (
    n_ducklings : ℕ,
    snails_first_group : ℕ,
    snails_second_group : ℕ,
    snails_mother : ℕ,
    snails_remaining_ducklings : ℕ): ℕ :=
  snails_first_group + snails_second_group + snails_mother + snails_remaining_ducklings

theorem ducks_snails (
    n_ducklings : ℕ,
    snails_per_first_group_duckling : ℕ,
    snails_per_second_group_duckling : ℕ,
    first_group_ducklings : ℕ,
    second_group_ducklings : ℕ,
    remaining_ducklings : ℕ,
    mother_duck_snails_mult : ℕ,
    half_mother_duck_snails : ℕ
) :
  n_ducklings = 8 →
  first_group_ducklings = 3 →
  second_group_ducklings = 3 →
  remaining_ducklings = 2 →
  snails_per_first_group_duckling = 5 →
  snails_per_second_group_duckling = 9 →
  mother_duck_snails_mult = 3 →
  ∀ mother_snails snails_per_remaining_duckling snails_first_group snails_second_group total_snails, 
    mother_snails = mother_duck_snails_mult * (first_group_ducklings * snails_per_first_group_duckling + second_group_ducklings * snails_per_second_group_duckling) →
    snails_per_remaining_duckling = mother_snails / 2 →
    snails_first_group = first_group_ducklings * snails_per_first_group_duckling →
    snails_second_group = second_group_ducklings * snails_per_second_group_duckling →
    total_snails = total_snails_found (
      n_ducklings,
      snails_first_group,
      snails_second_group,
      mother_snails,
      remaining_ducklings * snails_per_remaining_duckling
    ) →
    total_snails = 294 :=
by {
  intros,
  sorry
}

end ducks_snails_l744_744038


namespace fraction_of_phone_numbers_l744_744827

-- Definitions from conditions
def total_valid_phone_numbers : ℕ := 7 * 10^6
def phone_numbers_start_with_9_end_with_1 : ℕ := 10^5

-- Theorem statement, no proof required
theorem fraction_of_phone_numbers : (phone_numbers_start_with_9_end_with_1 / total_valid_phone_numbers : ℚ) = 1 / 70 :=
begin
  sorry
end

end fraction_of_phone_numbers_l744_744827


namespace circle_tangent_proof_l744_744937

noncomputable def circle_tangent_range : Set ℝ :=
  { k : ℝ | k > 0 ∧ ((3 - 2 * k)^2 + (1 - k)^2 > k) }

theorem circle_tangent_proof :
  ∀ k > 0, ((3 - 2 * k)^2 + (1 - k)^2 > k) ↔ (k ∈ (Set.Ioo 0 1 ∪ Set.Ioi 2)) :=
by
  sorry

end circle_tangent_proof_l744_744937


namespace men_complete_units_per_day_l744_744724

noncomputable def UnitsCompletedByMen (total_units : ℕ) (units_by_women : ℕ) : ℕ :=
  total_units - units_by_women

theorem men_complete_units_per_day :
  UnitsCompletedByMen 12 3 = 9 := by
  -- Proof skipped
  sorry

end men_complete_units_per_day_l744_744724


namespace range_of_a_l744_744537

theorem range_of_a 
  (e : ℝ) (h_e_pos : 0 < e) 
  (a : ℝ) 
  (h_equation : ∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ ≠ x₂ ∧ (1 / e ^ x₁ - a / x₁ = 0) ∧ (1 / e ^ x₂ - a / x₂ = 0)) :
  0 < a ∧ a < 1 / e :=
by
  sorry

end range_of_a_l744_744537


namespace percent_of_200_is_400_when_whole_is_50_l744_744408

theorem percent_of_200_is_400_when_whole_is_50 (Part Whole : ℕ) (hPart : Part = 200) (hWhole : Whole = 50) :
  (Part / Whole) * 100 = 400 :=
by {
  -- Proof steps go here.
  sorry
}

end percent_of_200_is_400_when_whole_is_50_l744_744408


namespace sound_pressures_relationships_l744_744277

variables (p p0 p1 p2 p3 : ℝ)
  (Lp Lpg Lph Lpe : ℝ)

-- The definitions based on the conditions
def sound_pressure_level (p : ℝ) (p0 : ℝ) : ℝ := 20 * (Real.log10 (p / p0))

-- Given conditions
axiom p0_gt_zero : p0 > 0

axiom gasoline_car_levels : 60 ≤ Lpg ∧ Lpg ≤ 90
axiom hybrid_car_levels : 50 ≤ Lph ∧ Lph ≤ 60
axiom electric_car_level : Lpe = 40

axiom gasoline_car_sound_pressure : Lpg = sound_pressure_level p1 p0
axiom hybrid_car_sound_pressure : Lph = sound_pressure_level p2 p0
axiom electric_car_sound_pressure : Lpe = sound_pressure_level p3 p0

-- The proof to be derived
theorem sound_pressures_relationships (p0_gt_zero : p0 > 0)
  (gasoline_car_levels : 60 ≤ Lpg ∧ Lpg ≤ 90)
  (hybrid_car_levels : 50 ≤ Lph ∧ Lph ≤ 60)
  (electric_car_level : Lpe = 40)
  (gasoline_car_sound_pressure : Lpg = sound_pressure_level p1 p0)
  (hybrid_car_sound_pressure : Lph = sound_pressure_level p2 p0)
  (electric_car_sound_pressure : Lpe = sound_pressure_level p3 p0) :
  p1 ≥ p2 ∧ p3 = 100 * p0 ∧ p1 ≤ 100 * p2 :=
by
  sorry

end sound_pressures_relationships_l744_744277


namespace number_of_values_of_z_l744_744790

open Complex

noncomputable def f (z : ℂ) : ℂ := I * conj z

theorem number_of_values_of_z :
  (set_of (λ z : ℂ, abs z = 3 ∧ f z = z)).to_finset.card = 2 :=
begin
  sorry
end

end number_of_values_of_z_l744_744790


namespace contradiction_assumption_l744_744756

theorem contradiction_assumption (l : set (ℝ × ℝ)) (h : set (ℝ × ℝ)) (H : ∀ p : ℝ × ℝ, p ∈ l ∧ p ∈ h → ¬ (∃ q r : ℝ × ℝ, q ∈ l ∧ q ∈ h ∧ q ≠ r ∧ r ∈ l ∧ r ∈ h)) :
  ∃ p1 p2 : ℝ × ℝ, p1 ∈ l ∧ p1 ∈ h ∧ p2 ∈ l ∧ p2 ∈ h ∧ p1 ≠ p2 :=
sorry

end contradiction_assumption_l744_744756


namespace counterexample_to_prime_statement_l744_744838

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ is_prime n

theorem counterexample_to_prime_statement 
  (n : ℕ) 
  (h_n_composite : is_composite n) 
  (h_n_minus_3_not_prime : ¬ is_prime (n - 3)) : 
  n = 18 ∨ n = 24 :=
by 
  sorry

end counterexample_to_prime_statement_l744_744838


namespace conjugate_of_z_l744_744510

theorem conjugate_of_z (z : ℂ) (h : z * (1 - complex.I) = 4 * complex.I) : 
  complex.conj z = -2 - 2 * complex.I :=
sorry

end conjugate_of_z_l744_744510


namespace right_triangle_sin_b_l744_744185

theorem right_triangle_sin_b (a b c : ℝ) (h : a^2 + b^2 = c^2) (θ : ℝ) (hc90 : θ = π / 2) (hB : a = c * sin θ) :
  b = c * sin θ :=
by
  sorry

end right_triangle_sin_b_l744_744185


namespace coupon_1_best_for_219_95_l744_744035

def discount_coupon_1 (x : ℝ) : ℝ := 0.1 * x
def discount_coupon_2 (x : ℝ) : ℝ := if 100 ≤ x then 20 else 0
def discount_coupon_3 (x : ℝ) : ℝ := if 100 < x then 0.18 * (x - 100) else 0

theorem coupon_1_best_for_219_95 :
  (200 < 219.95) ∧ (219.95 < 225) →
  (discount_coupon_1 219.95 > discount_coupon_2 219.95) ∧
  (discount_coupon_1 219.95 > discount_coupon_3 219.95) :=
by sorry

end coupon_1_best_for_219_95_l744_744035


namespace problem_1_problem_2_l744_744549

noncomputable def f (x : ℝ) := Real.cos x ^ 2
noncomputable def g (x : ℝ) := 1 + 1 / 2 * Real.sin (2 * x)

theorem problem_1 (x₀ k : ℝ) (hk : 2 * x₀ = k * Real.pi) : g (2 * x₀) = 1 :=
  by
  rw [hk]
  simp [g]
  simp [Real.sin_mul]

noncomputable def h (x : ℝ) := f x + g x

theorem problem_2 : set.Icc (2 : ℝ) ((3 + Real.sqrt 2) / 2) = 
  ⋂  (x : ℝ), h x :=
  sorry

end problem_1_problem_2_l744_744549


namespace find_constants_and_extrema_l744_744507

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem find_constants_and_extrema (a b c : ℝ) (h : a ≠ 0) 
    (ext1 : deriv (f a b c) 1 = 0) (ext2 : deriv (f a b c) (-1) = 0) (value1 : f a b c 1 = -1) :
    a = -1/2 ∧ b = 0 ∧ c = 1/2 ∧ 
    (∃ x : ℝ, x = 1 ∧ deriv (deriv (f a b c)) x < 0) ∧
    (∃ x : ℝ, x = -1 ∧ deriv (deriv (f a b c)) x > 0) :=
sorry

end find_constants_and_extrema_l744_744507


namespace find_greatest_integer_l744_744464

-- Define the conditions
variables (AB BC CA : ℝ) (R NE BE : ℝ)
def is_triangle (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b 

variables (p sqrt_q : ℝ)
def calculates_be (p sqrt_q : ℝ) : ℝ := p + sqrt_q

noncomputable def problem_statement : Prop :=
  let a := (65 / 2 : ℝ) in
  let b := real.sqrt 17 in
  floor (a + b) = 32

-- Main statement to prove
theorem find_greatest_integer :
  is_triangle 26 15 37 ∧
  R = 65 / (2 * real.sqrt 17) ∧ 
  BE = R ∧
  calculates_be (65 / 2) (real.sqrt 17) = 32 → problem_statement := by
  sorry

end find_greatest_integer_l744_744464


namespace problem1_problem2_l744_744456

-- Define variables needed for the proof
noncomputable def question1 := sqrt 48 / sqrt 3 - 4 * sqrt (1/5) * sqrt 30 + (2 * sqrt 2 + sqrt 3)^2
noncomputable def question2 := sqrt 27 + abs (1 - sqrt 3) + (1/3)^(-1) - (pi - 3)^0

-- State the theorem for Question 1
theorem problem1 : question1 = 15 := by
  sorry

-- State the theorem for Question 2
theorem problem2 : question2 = 4 * sqrt 3 + 2 := by
  sorry

end problem1_problem2_l744_744456


namespace moose_arrangements_l744_744846

theorem moose_arrangements : 
  let m := 1 in
  let o := 2 in
  let s := 1 in
  let e := 1 in
  let total_letters := m + o + s + e in
  total_letters = 5 ∧ ∃ r_n : ℕ, (∀ c : ℕ, c ∈ [m, o, s, e] → c ≤ total_letters) → 
  r_n = m! * e! * s! * o! / (m! * e! * s! * o!) → r_n = 60 := sorry

end moose_arrangements_l744_744846


namespace range_k_l744_744652

variable {R : Type} [LinearOrderedField R]
open Real

def f (x : R) : R :=
  (exp 2 * x^2 + 1) / x

def g (x : R) : R :=
  (exp 2 * x) / (exp x)

theorem range_k (k : R) (hk : 0 < k) :
  (∀ (x1 x2 : R), 0 < x1 → 0 < x2 → x1 ∈ Set.Ioi 0 → x2 ∈ Set.Ioi 0 → g x1 / k ≤ f x2 / (k + 1)) →
  1 ≤ k :=
by
  sorry

end range_k_l744_744652


namespace polygon_properties_l744_744773

-- Assume n is the number of sides of the polygon
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_of_exterior_angles : ℝ := 360

-- Given the condition
def given_condition (n : ℕ) : Prop := sum_of_interior_angles n = 5 * sum_of_exterior_angles

theorem polygon_properties (n : ℕ) (h1 : given_condition n) :
  n = 12 ∧ (n * (n - 3)) / 2 = 54 :=
by
  sorry

end polygon_properties_l744_744773


namespace covering_inequality1_covering_inequality2_covering_inequality3_l744_744945

-- Problem (1)
theorem covering_inequality1 {x : ℝ} : x < -3 → x < -1 := by
  intro h
  exact lt_trans h (by linarith)

-- Problem (2)
theorem covering_inequality2 (m : ℝ) : (∀ x : ℝ, x < 4 * m → x < -2) → m ≤ -1/2 := by
  intro h
  have h1 := h (-2)
  linarith

-- Problem (3)
theorem covering_inequality3 (a : ℝ) : (∀ x : ℝ, (x < 2 * a - 1 ∧ x > (3 * a - 5)/2) → (1 ≤ x ∧ x ≤ 6)) → 7/3 ≤ a ∧ a ≤ 7/2 := by
  intro h
  have h1 := h ((3 * a - 5)/2 + 1)
  linarith

end covering_inequality1_covering_inequality2_covering_inequality3_l744_744945


namespace not_exactly_one_nice_cell_l744_744344

-- Defining the board size.
def BoardSize : ℕ := 100

-- Defining what it means for a cell to be nice: a cell is nice if it has an even number of counters in adjacent cells.
def is_nice (board : Matrix BoardSize BoardSize ℕ) (pos : Fin BoardSize × Fin BoardSize) : Prop :=
  (∑ (i: Fin BoardSize) in adjacent_cells pos, board i) % 2 = 0

-- Function to get adjacent cells' positions.
def adjacent_cells (pos : Fin BoardSize × Fin BoardSize) : Finset (Fin BoardSize × Fin BoardSize) := 
  sorry -- Definition of adjacent cells would go here.

-- Hypothesis: cell (i, j) can be nice based on the given conditions.
variable (board : Matrix BoardSize BoardSize ℕ)

-- The main theorem: it is impossible to have exactly one nice cell.
theorem not_exactly_one_nice_cell : ¬ ∃! (pos : Fin BoardSize × Fin BoardSize), is_nice board pos :=
by
  sorry

end not_exactly_one_nice_cell_l744_744344


namespace jenny_last_page_stamps_l744_744975

theorem jenny_last_page_stamps :
  ∀ (books pages: ℕ) (stamps_per_page_initial stamps_per_page_new: ℕ)
  (full_books new_full_books partial_new_full_books: ℕ),
  books = 10 →
  pages = 50 →
  stamps_per_page_initial = 6 →
  stamps_per_page_new = 8 →
  full_books = 6 →
  new_full_books = 6 →
  partial_new_full_books = 45 →
  let total_pages_in_7th_book := 50
  let total_stamps := books * pages * stamps_per_page_initial,
  let new_total_pages := total_stamps / stamps_per_page_new,
  let stamps_on_last_page := total_stamps - (new_total_pages - 1) * stamps_per_page_new,
  stamps_on_last_page = 8 :=
begin
  sorry
end

end jenny_last_page_stamps_l744_744975


namespace probability_A_inter_B_l744_744921

def set_A (x : ℝ) : Prop := -1 < x ∧ x < 5
def set_B (x : ℝ) : Prop := (x-2)/(3-x) > 0

def A_inter_B (x : ℝ) : Prop := set_A x ∧ set_B x

theorem probability_A_inter_B :
  let length_A := 5 - (-1)
  let length_A_inter_B := 3 - 2 
  length_A > 0 ∧ length_A_inter_B > 0 →
  length_A_inter_B / length_A = 1 / 6 :=
by
  intro h
  sorry

end probability_A_inter_B_l744_744921


namespace hyperbola_eccentricity_l744_744914

theorem hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_asymptote : ∃ x y : ℝ, (y = (a / b) * x) ∧ (x^2 = 8 * y)) 
  (h_distance : ∃ x, |((8 * a^2 / b^2) - (-2))| = 4) :
  (b = 2 * a) → (Real.sqrt (a^2 + b^2) / a = Real.sqrt 5) :=
sorry

end hyperbola_eccentricity_l744_744914


namespace sum_of_weighted_t_is_negative_l744_744771

theorem sum_of_weighted_t_is_negative {x y : ℕ → ℝ} {n : ℕ}
  (h1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → -1 < x i ∧ x i < x j ∧ x j < 1)
  (h2 : ∑ i in Finset.range n, (x i) ^ 13 = ∑ i in Finset.range n, (x i))
  (h3 : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → y i < y j) :
  ∑ i in Finset.range n, (x i ^ 13 - x i) * y i < 0 :=
by
  sorry

end sum_of_weighted_t_is_negative_l744_744771


namespace limit_of_sequence_l744_744301

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) :
  (∀ n : ℕ, a_n n = (2 * (n ^ 3)) / ((n ^ 3) - 2)) →
  a = 2 →
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) :=
by
  intros h1 h2 ε hε
  sorry

end limit_of_sequence_l744_744301


namespace contractor_pays_child_worker_per_day_l744_744780

theorem contractor_pays_child_worker_per_day
  (number_of_males : ℕ)
  (number_of_females : ℕ)
  (number_of_children : ℕ)
  (wage_male : ℕ)
  (wage_female : ℕ)
  (average_wage : ℕ)
  (total_workers : ℕ := number_of_males + number_of_females + number_of_children)
  (total_amount_paid : ℕ := total_workers * average_wage) :
  (number_of_males * wage_male + number_of_females * wage_female + number_of_children * 8 = total_amount_paid) →
  (average_wage = 21) →
  (number_of_males = 20) →
  (number_of_females = 15) →
  (number_of_children = 5) →
  (wage_male = 25) →
  (wage_female = 20) →
  8 = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  exact eq.refl 8

end contractor_pays_child_worker_per_day_l744_744780


namespace number_splits_is_power_of_two_l744_744062

-- Define the conditions of the problem
variables (Mathematician : Type) [Fintype Mathematician]

-- Define friendships as a relation on Mathematicians
variable (friend : Mathematician → Mathematician → Prop)

-- Define the property that each participant needs an even number of friends in the same room
def even_friend_count (r1 r2 : Set Mathematician) : Prop :=
  ∀ m : Mathematician, (m ∈ r1 ∧ (Fintype.card {m' | friend m m' ∧ m' ∈ r1} % 2 = 0)) ∨ (m ∈ r2 ∧ (Fintype.card {m' | friend m m' ∧ m' ∈ r2} % 2 = 0))

-- Definition of the main problem
theorem number_splits_is_power_of_two
  (h : ∃ r1 r2 : Set Mathematician, r1 ∪ r2 = Finset.univ ∧ r1 ∩ r2 = ∅ ∧ even_friend_count friend r1 r2) :
  ∃ k : ℕ, Fintype.card {⟨r1, r2⟩ : Set Mathematician × Set Mathematician // r1 ∪ r2 = Finset.univ ∧ r1 ∩ r2 = ∅ ∧ even_friend_count friend r1 r2} = 2^k := sorry

end number_splits_is_power_of_two_l744_744062


namespace f_3x_equals_2011_l744_744898

noncomputable def f : ℝ → ℝ := sorry

theorem f_3x_equals_2011 (h_domain : ∀ x : ℝ, x ∈ set.univ)
                         (h_constant : ∀ x : ℝ, f (2011 * x) = 2011) :
  f (3 * x) = 2011 :=
by
  sorry

end f_3x_equals_2011_l744_744898


namespace cos_double_angle_l744_744578

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l744_744578


namespace next_term_in_geom_sequence_l744_744373

   /- Define the given geometric sequence as a function in Lean -/

   def geom_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ n

   theorem next_term_in_geom_sequence (x : ℤ) (n : ℕ) 
     (h₁ : geom_sequence 3 (-3*x) 0 = 3)
     (h₂ : geom_sequence 3 (-3*x) 1 = -9*x)
     (h₃ : geom_sequence 3 (-3*x) 2 = 27*(x^2))
     (h₄ : geom_sequence 3 (-3*x) 3 = -81*(x^3)) :
     geom_sequence 3 (-3*x) 4 = 243*(x^4) := 
   sorry
   
end next_term_in_geom_sequence_l744_744373


namespace volume_prism_l744_744867

variables (a : ℝ) (V : ℝ)

theorem volume_prism (a : ℝ) (inclined_angle : ℝ) (h_inclined_angle : inclined_angle = 60) : 
  let base_area := (a^2 * (Real.sqrt 3)) / 4 in
  let height := a * (Real.sin (inclined_angle * Real.pi / 180)) in
  let V := base_area * height in
  V = 3 * a^3 / 8 := 
by
  sorry

end volume_prism_l744_744867


namespace replacement_inequality_l744_744743

theorem replacement_inequality (n : ℕ) (h : n > 0) :
  (∃ x : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → (∃ m : ℕ, m = n - 1 ∧ ∀ t, t ∈ {1 | t > 0}) ∧
    ((∀ k : ℕ, k < n → (1 : ℝ/n)) ∧ 
    (∀ l : ℕ, l = n - 1 → x ≥ 1 / n)))) :=
sorry

end replacement_inequality_l744_744743


namespace target_has_more_tools_l744_744742

-- Define the number of tools in the Walmart multitool
def walmart_screwdriver : ℕ := 1
def walmart_knives : ℕ := 3
def walmart_other_tools : ℕ := 2
def walmart_total_tools : ℕ := walmart_screwdriver + walmart_knives + walmart_other_tools

-- Define the number of tools in the Target multitool
def target_screwdriver : ℕ := 1
def target_knives : ℕ := 2 * walmart_knives
def target_files_scissors : ℕ := 3 + 1
def target_total_tools : ℕ := target_screwdriver + target_knives + target_files_scissors

-- The theorem stating the difference in the number of tools
theorem target_has_more_tools : (target_total_tools - walmart_total_tools) = 6 := by
  sorry

end target_has_more_tools_l744_744742
