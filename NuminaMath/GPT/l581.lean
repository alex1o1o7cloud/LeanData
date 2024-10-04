import Mathlib

namespace perfect_squares_and_cubes_l581_581775

def is_perfect_square (n: ℕ) : Prop := ∃ k: ℕ, k ^ 2 = n
def is_perfect_cube (n: ℕ) : Prop := ∃ k: ℕ, k ^ 3 = n
def is_perfect_sixth_power (n: ℕ) : Prop := ∃ k: ℕ, k ^ 6 = n

def num_perfect_squares (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_square n).length
def num_perfect_cubes (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_cube n).length
def num_perfect_sixth_powers (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_sixth_power n).length

theorem perfect_squares_and_cubes (bound: ℕ) (hbound: bound = 1000) :
  num_perfect_squares bound + num_perfect_cubes bound - num_perfect_sixth_powers bound = 37 :=
by
  -- conditions:
  have ps := num_perfect_squares bound
  have pc := num_perfect_cubes bound
  have p6 := num_perfect_sixth_powers bound
  sorry

end perfect_squares_and_cubes_l581_581775


namespace find_decreasing_function_l581_581176

def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f y < f x

theorem find_decreasing_function :
  ∃! f : ℝ → ℝ,
    (f = (λ x, Real.log x / Real.log (4 / 5))) ∧
    is_decreasing_on f (set.Ioi 0) ∧
    ¬ is_decreasing_on (λ x, Real.log x / Real.log (5 / 4)) (set.Ioi 0) ∧
    ¬ is_decreasing_on (λ x, Real.exp x) (set.Ioi 0) ∧
    ¬ is_decreasing_on Real.log (set.Ioi 0) :=
sorry

end find_decreasing_function_l581_581176


namespace units_digit_A_is_1_l581_581733

def units_digit (n : ℕ) : ℕ := n % 10

noncomputable def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

theorem units_digit_A_is_1 : units_digit A = 1 := by
  sorry

end units_digit_A_is_1_l581_581733


namespace sum_of_odd_divisors_252_l581_581534

noncomputable def sum_of_odd_divisors (n : ℕ) : ℕ :=
  finset.sum (finset.filter (λ d, d % 2 = 1) (finset.divisors n)) id

theorem sum_of_odd_divisors_252 : sum_of_odd_divisors 252 = 104 :=
by
  sorry

end sum_of_odd_divisors_252_l581_581534


namespace real_solutions_l581_581665

noncomputable def solveEquation (x : ℝ) : Prop :=
  (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10

theorem real_solutions :
  {x : ℝ | solveEquation x} = {x : ℝ | x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15} :=
by
  sorry

end real_solutions_l581_581665


namespace four_by_four_grid_no_all_pluses_l581_581969

theorem four_by_four_grid_no_all_pluses :
  let initial_grid := 
    matrix.vecCons (1 %% 2 * -1 %% 2 + 0 %% 2) 
    (matrix.vecCons (-1 %% 2) 
    (matrix.vecCons (1 %% 2 * -1 %% 2 + 0 %% 2) (/////) 
  ∀ operations, 
    (initial_grid === grid_with_all_pluses) → False := {
  sorry
}

end four_by_four_grid_no_all_pluses_l581_581969


namespace problem_statement_l581_581028

noncomputable def a (n : ℕ) : ℤ := 3 * n - 14
noncomputable def b (n : ℕ) : ℤ := (2 : ℤ)^(n - 1)
noncomputable def c (n : ℕ) : ℤ := abs (a n) + b n

def S (n : ℕ) : ℤ := (Finset.range n).sum (λ i, b (i + 1))

theorem problem_statement : 
  a 3 = -5 ∧ 
  a 8 = 10 ∧ 
  b 1 = 1 ∧ 
  (4 * S 1 + 3 * S 2 + 2 * S 3) = t3 -> -- Equating the arithmetic sequence formation as per condition
  let T := (Finset.range 10).sum (λ n, c (n + 1))
  in T = 1100 :=
by 
  sorry

end problem_statement_l581_581028


namespace count_perfect_squares_and_cubes_l581_581750

theorem count_perfect_squares_and_cubes (n : ℕ) (h : n = 1000) : 
  let squares := {x : ℕ | (x ^ 2) < n}
  let cubes := {x : ℕ | (x ^ 3) < n}
  let sixth_powers := {x : ℕ | (x ^ 6) < n}
  (squares.card + cubes.card - sixth_powers.card) = 39 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581750


namespace line_intersects_circle_l581_581975

theorem line_intersects_circle
  (R d : ℝ)
  (hR : R = 4)
  (hd : d = 3) :
  d < R :=
by {
  rw [hR, hd],
  exact DecTrivial.le_refl 3 4 }.

end line_intersects_circle_l581_581975


namespace crayon_selection_l581_581934

/-- The number of ways Tina can select six crayons out of 15 
    such that the selection includes exactly one red and one blue crayon 
    and the order of selection does not matter is 715. -/
theorem crayon_selection : 
  let total_crayons := 15,
      remaining_crayons := total_crayons - 2, -- subtracting the red and blue crayons
      choose_k := 4 in
  nat.choose remaining_crayons choose_k = 715 :=
by
  let total_crayons := 15,
      remaining_crayons := total_crayons - 2, -- subtracting the red and blue crayons
      choose_k := 4 
  exact sorry

end crayon_selection_l581_581934


namespace alice_wins_probability_is_5_over_6_l581_581224

noncomputable def probability_Alice_wins : ℚ := 
  let total_pairs := 36
  let losing_pairs := 6
  1 - (losing_pairs / total_pairs)

theorem alice_wins_probability_is_5_over_6 : 
  let winning_probability := probability_Alice_wins
  winning_probability = 5 / 6 :=
by
  sorry

end alice_wins_probability_is_5_over_6_l581_581224


namespace parabola_shifting_produces_k_l581_581713

theorem parabola_shifting_produces_k
  (k : ℝ)
  (h1 : -k/2 > 0)
  (h2 : (0 : ℝ) = (((0 : ℝ) - 3) + k/2)^2 - (5*k^2)/4 + 1)
  :
  k = -5 :=
sorry

end parabola_shifting_produces_k_l581_581713


namespace fraction_of_oil_sent_to_production_l581_581610

-- Definitions based on the problem's conditions
def initial_concentration : ℝ := 0.02
def replacement_concentration1 : ℝ := 0.03
def replacement_concentration2 : ℝ := 0.015
def final_concentration : ℝ := 0.02

-- Main theorem stating the fraction x is 1/2
theorem fraction_of_oil_sent_to_production (x : ℝ) (hx : x > 0) :
  (initial_concentration + (replacement_concentration1 - initial_concentration) * x) * (1 - x) +
  replacement_concentration2 * x = final_concentration →
  x = 0.5 :=
  sorry

end fraction_of_oil_sent_to_production_l581_581610


namespace number_of_perfect_squares_and_cubes_l581_581766

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end number_of_perfect_squares_and_cubes_l581_581766


namespace division_number_l581_581007

open Real

theorem division_number (x : ℝ) (h1 : 3.242 * 15 = 48.63) (h2 : 48.63 / x = 0.04863) : x ≈ 999.793 :=
by
  /- Proof omitted -/
  sorry

end division_number_l581_581007


namespace problem_statement_l581_581949

theorem problem_statement (a x m : ℝ) (h₀ : |a| ≤ 1) (h₁ : |x| ≤ 1) :
  (∀ x a, |x^2 - a * x - a^2| ≤ m) ↔ m ≥ 5/4 :=
sorry

end problem_statement_l581_581949


namespace find_a_l581_581063

noncomputable def f (a x : ℝ) : ℝ :=
  a / (x - 1) + 1 / (x - 2) + 1 / (x - 6)

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_max : ∀ x ∈ Ioo 3 5, f a x ≤ f a 4) : a = -9 / 2 :=
  sorry

end find_a_l581_581063


namespace numPerfectSquaresOrCubesLessThan1000_l581_581772

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def isSixthPower (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

theorem numPerfectSquaresOrCubesLessThan1000 : 
  ∃ n : ℕ, n = 38 ∧ ∀ k < 1000, (isPerfectSquare k ∨ isPerfectCube k → k) ↔ n :=
by
  sorry

end numPerfectSquaresOrCubesLessThan1000_l581_581772


namespace transformation_count_l581_581158

-- Define the vertices of the triangle XYZ
def X := (1, 0)
def Y := (0, 1)
def Z := (-1, 0)

-- Define the transformations as permutations
def L := [Y, Z, X]
def R := [Z, Y, X]
def H := [Z, Y, X]
def V := [X, Z, Y]
def T := [X, Z, Y]

-- Define the identity transformation
def I := [X, Y, Z]

-- Define the function to count sequences of 24 transformations
def count_transformations(L R H V T : List (Int × Int)) : ℕ :=
  let transformations := [L, R, H, V, T]
  let half_count := 12
  Nat.choose (half_count + (transformations.length - 1)) (transformations.length - 1)

-- Prove the claim
theorem transformation_count :
  count_transformations L R H V T = 1820 := by
  sorry

end transformation_count_l581_581158


namespace product_digit_count_l581_581262

noncomputable def num_digits (n : ℕ) : ℕ :=
  nat.floor (real.log10 n) + 1

theorem product_digit_count : num_digits (6^4 * 7^8) = 10 :=
  sorry

end product_digit_count_l581_581262


namespace matrix_determinant_is_zero_l581_581247

variable (a b : ℝ)

theorem matrix_determinant_is_zero :
  Matrix.det ![
    ![1, Real.cos (a - b), Real.cos a], 
    ![Real.cos (a - b), 1, Real.cos b], 
    ![Real.cos a, Real.cos b, 1]
  ] = 0 := 
sorry

end matrix_determinant_is_zero_l581_581247


namespace limit_one_minus_reciprocal_l581_581617

theorem limit_one_minus_reciprocal (h : Filter.Tendsto (fun (n : ℕ) => 1 / n) Filter.atTop (nhds 0)) :
  Filter.Tendsto (fun (n : ℕ) => 1 - 1 / n) Filter.atTop (nhds 1) :=
sorry

end limit_one_minus_reciprocal_l581_581617


namespace convex_ngon_iff_m_eq_l581_581968

noncomputable def binom (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

variable {n : ℕ} (S : finset (ℝ × ℝ))

def no_three_collinear (S : finset (ℝ × ℝ)) : Prop :=
  ∀ (P1 P2 P3 : (ℝ × ℝ)), P1 ∈ S → P2 ∈ S → P3 ∈ S → P1 ≠ P2 → P2 ≠ P3 → P1 ≠ P3 → 
  ¬ collinear P1 P2 P3

def no_four_concyclic (S : finset (ℝ × ℝ)) : Prop :=
  ∀ (P1 P2 P3 P4 : (ℝ × ℝ)), P1 ∈ S → P2 ∈ S → P3 ∈ S → P4 ∈ S → P1 ≠ P2 → P2 ≠ P3 → P3 ≠ P4 → P4 ≠ P1 →
  ¬ concyclic P1 P2 P3 P4

def a_t (S : finset (ℝ × ℝ)) (t : ℝ × ℝ) : ℕ :=
  (S.to_finset.powerset.filter (λ T, T.card = 3 ∧ t ∈ T)).card

def m (S : finset (ℝ × ℝ)) : ℕ :=
  ∑ t in S, a_t S t

def is_convex_ngon (S : finset (ℝ × ℝ)) : Prop :=
  convex_hull ℝ ↑S = ↑S

theorem convex_ngon_iff_m_eq (S : finset (ℝ × ℝ)) (hn : 4 ≤ S.card) (h_no_three_collinear : no_three_collinear S) (h_no_four_concyclic : no_four_concyclic S) :
  (m S = 2 * binom S.card 4) ↔ is_convex_ngon S :=
sorry

end convex_ngon_iff_m_eq_l581_581968


namespace seating_arrangements_l581_581461

theorem seating_arrangements (n : ℕ) (k : ℕ) (total_seats available_gaps people_permutations : ℕ) :
  total_seats = 9 →
  k = 3 →
  available_gaps = 5 →
  (people_permutations : ℕ) = (Nat.factorial k) →
  C(available_gaps, k) * people_permutations = 60 := by
  intro h1 h2 h3 h4
  have h5 : C(available_gaps, k) = Nat.choose available_gaps k := by sorry
  rw [h1, h2, h3, h4]
  calc
  Nat.choose available_gaps k * people_permutations
        = (5.choose 3) * 6 : by simp [h5, Nat.choose]
    ... = 10 * 6 : by norm_num
    ... = 60 : by norm_num

end seating_arrangements_l581_581461


namespace count_perfect_squares_and_cubes_l581_581735

theorem count_perfect_squares_and_cubes : 
  let num_squares := 31 in
  let num_cubes := 9 in
  let num_sixth_powers := 3 in
  num_squares + num_cubes - num_sixth_powers = 37 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581735


namespace find_a_l581_581113

theorem find_a (a : ℝ) : 
  (let chord_length := (x : ℝ) → (x - a)^2 + x^2 = 4 ∧ x = 2 ∧ x = 2)
  (chord_length =  2 * sqrt(3)) → a = 1 ∨  a = 3 :=
by
  sorry

end find_a_l581_581113


namespace island_has_treasures_l581_581135

-- Definitions
def is_knight (person : Type) : Prop := sorry -- A knight always tells the truth
def is_liar (person : Type) : Prop := sorry -- A liar always lies

-- Inhabitants and their statements
variables (A B C : Type)
def statement_A : Prop := sorry -- A: The number of liars on this island is even
def statement_B : Prop := sorry -- B: There is an odd number of people on our island right now
def statement_C : Prop := (is_knight C ↔ (is_knight A ↔ is_knight B))

-- Condition of the island
def has_even_knights (island : Type) : Prop := sorry -- The island has an even number of knights
def has_odd_knights (island : Type) : Prop := sorry -- The island has an odd number of knights

-- Treasures condition
def has_treasure (island : Type) : Prop := (has_even_knights island)

-- Main theorem
theorem island_has_treasures (island : Type) [inhabitants : island -> Prop] :
  has_treasure island :=
by
  -- Provided the conditions and given statements, prove there are treasures on the island
  sorry

end island_has_treasures_l581_581135


namespace distance_between_points_l581_581652

-- Define the points
def point1 : ℝ × ℝ × ℝ := (2, 3, 1)
def point2 : ℝ × ℝ × ℝ := (5, 9, 4)

-- Define the distance function for two points in 3D space
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

-- State the theorem
theorem distance_between_points :
  distance point1 point2 = 3 * real.sqrt 6 :=
by
  sorry

end distance_between_points_l581_581652


namespace probability_y_leq_1_div_x_l581_581707

open Real
open Interval

noncomputable def probability := (1 + 2 * ln 2) / 4

theorem probability_y_leq_1_div_x :
  let S := Icc 0 2 ×ˢ Icc 0 2 in
  volume { p : ℝ × ℝ | p ∈ S ∧ p.snd ≤ 1 / p.fst } / volume S = probability :=
by
  sorry

end probability_y_leq_1_div_x_l581_581707


namespace truncated_pyramid_properties_l581_581121

noncomputable def truncatedPyramidSurfaceArea
  (a b c : ℝ) (theta m : ℝ) : ℝ :=
sorry -- Definition of the surface area function

noncomputable def truncatedPyramidVolume
  (a b c : ℝ) (theta m : ℝ) : ℝ :=
sorry -- Definition of the volume function

theorem truncated_pyramid_properties
  (a b c : ℝ) (theta m : ℝ)
  (h₀ : a = 148) 
  (h₁ : b = 156) 
  (h₂ : c = 208) 
  (h₃ : theta = 112.62) 
  (h₄ : m = 27) :
  (truncatedPyramidSurfaceArea a b c theta m = 74352) ∧
  (truncatedPyramidVolume a b c theta m = 395280) :=
by
  sorry -- The actual proof will go here

end truncated_pyramid_properties_l581_581121


namespace volume_of_quadrilateral_pyramid_l581_581476

theorem volume_of_quadrilateral_pyramid (m α : ℝ) : 
  ∃ (V : ℝ), V = (2 / 3) * m^3 * (Real.cos α) * (Real.sin (2 * α)) :=
by
  sorry

end volume_of_quadrilateral_pyramid_l581_581476


namespace probability_divisible_by_10_l581_581234

theorem probability_divisible_by_10 :
  ∀ (balls : list ℕ) (draws : list ℕ),
    (balls = [1, 2, 3, 4, 5]) →
    (length draws = 3) →
    (∀ d, d ∈ draws → d ∈ balls) →
    (let p : ℚ := (42 : ℚ) / 125 in
    p = (↑(draws.filter (λ(e : list ℕ), (∃ n ∈ [2, 4], e.prod % 10 = 0)).length) / (5 ^ 3 : ℚ))) :=
by {
  intros balls draws h_balls h_len h_draws,
  let p : ℚ := (42 : ℚ) / 125,
  have : p = (↑(draws.filter (λ(e : list ℕ), (∃ n ∈ [2, 4], e.prod % 10 = 0)).length) / (5 ^ 3 : ℚ)),
  sorry,
}

end probability_divisible_by_10_l581_581234


namespace min_Box_value_l581_581351

/-- The conditions are given as:
  1. (ax + b)(bx + a) = 24x^2 + Box * x + 24
  2. a, b, Box are distinct integers
  The task is to find the minimum possible value of Box.
-/
theorem min_Box_value :
  ∃ (a b Box : ℤ), a ≠ b ∧ a ≠ Box ∧ b ≠ Box ∧ (∀ x : ℤ, (a * x + b) * (b * x + a) = 24 * x^2 + Box * x + 24) ∧ Box = 52 := sorry

end min_Box_value_l581_581351


namespace find_m_n_l581_581339

theorem find_m_n (m n : ℝ) (A B : set ℝ)
  (hA : A = { x : ℝ | |x + 2| < 3 })
  (hB : B = { x : ℝ | (x - m) * (x - 2) < 0 })
  (h_intersection : A ∩ B = set.Ioo (-1 : ℝ) n) :
  m + n = 0 :=
sorry

end find_m_n_l581_581339


namespace find_x_for_f_eq_one_fourth_l581_581429

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x) else Real.log x / Real.log 81

theorem find_x_for_f_eq_one_fourth : ∃ x : ℝ, f x = 1 / 4 ↔ x = 3 := 
by
  sorry

end find_x_for_f_eq_one_fourth_l581_581429


namespace blue_tickets_needed_l581_581937

theorem blue_tickets_needed
  (yellow_required : Nat := 20)
  (yellow_have : Nat := 12)
  (red_per_yellow : Nat := 15)
  (red_have : Nat := 8)
  (green_per_red : Nat := 12)
  (green_have : Nat := 14)
  (blue_per_green : Nat := 10)
  (blue_have : Nat := 27) :
  (let
    additional_yellow_needed := yellow_required - yellow_have,
    total_additional_red_needed := additional_yellow_needed * red_per_yellow - red_have,
    total_additional_green_needed := total_additional_red_needed * green_per_red - green_have,
    total_additional_blue_needed := total_additional_green_needed * blue_per_green - blue_have
  in total_additional_blue_needed = 13273) :=
  sorry

end blue_tickets_needed_l581_581937


namespace geometric_sequence_ninth_term_l581_581495

theorem geometric_sequence_ninth_term (a3 a6 : ℕ) (r : ℕ) (h1 : a3 = 16) (h2 : a6 = 144) (h3 : a6 = a3 * r^3) : (a6 * r^3 = 1296) :=
by
  rw [h1, h2, h3]
  -- Proof would go here
  sorry

end geometric_sequence_ninth_term_l581_581495


namespace number_subtraction_l581_581590

theorem number_subtraction
  (x : ℕ) (y : ℕ)
  (h1 : x = 30)
  (h2 : 8 * x - y = 102) : y = 138 :=
by 
  sorry

end number_subtraction_l581_581590


namespace math_problem_l581_581723

theorem math_problem 
  (f : ℝ → ℝ)
  (phi : ℝ)
  (h_def : ∀ x, f x = 2 * Real.sin (2 * x + phi) + 1)
  (h_point : f 0 = 0)
  (h_phi_range : -Real.pi / 2 < phi ∧ phi < 0) : 
  (phi = -Real.pi / 6) ∧ (∃ k : ℤ, ∀ x, f x = 3 ↔ x = k * Real.pi + 2 * Real.pi / 3) :=
sorry

end math_problem_l581_581723


namespace order_of_numbers_l581_581562

noncomputable def q := 70.3
noncomputable def r := 0.37
noncomputable def s := Real.log 0.3

theorem order_of_numbers : q > r ∧ r > s :=
by
  -- Given
  have h₁ : q = 70.3 := rfl
  have h₂ : r = 0.37 := rfl
  have h₃ : s = Real.log 0.3 := rfl
  -- Proof of conditions
  have h₄ : q > 1 := by sorry
  have h₅ : 0 < r := by sorry
  have h₆ : r < 1 := by sorry
  have h₇ : s < 0 := by sorry
  -- Combine to finish the proof
  sorry

end order_of_numbers_l581_581562


namespace extreme_value_when_a_is_e_no_positive_a_such_that_f_always_greater_than_a_l581_581330

noncomputable def f (x : ℝ) (a : ℝ) := x + a / Real.exp x

theorem extreme_value_when_a_is_e : 
  ∃ x : ℝ, f x Real.exp 1 = 2 ∧ ∀ y : ℝ, f y Real.exp 1 ≥ f x Real.exp 1 := 
  sorry

theorem no_positive_a_such_that_f_always_greater_than_a (a : ℝ) :
  (0 < a) → ¬ (∀ x : ℝ, f x a > a) := 
  sorry

end extreme_value_when_a_is_e_no_positive_a_such_that_f_always_greater_than_a_l581_581330


namespace increasing_interval_of_decreasing_l581_581319

variable {α : Type*} [LinearOrder α] {f : α → α} 

theorem increasing_interval_of_decreasing (h : ∀ x₁ x₂, 2 < x₁ → x₁ < x₂ → x₂ < 8 → f x₁ > f x₂) :
  ∀ x₃ x₄, -4 < x₃ → x₃ < x₄ → x₄ < 2 → f (4 - x₃) < f (4 - x₄) :=
by
  intro x₃ x₄ hx₃ hx x₄_upper
  -- proof here
  sorry

end increasing_interval_of_decreasing_l581_581319


namespace subtraction_division_l581_581565

theorem subtraction_division : 3550 - (1002 / 20.04) = 3499.9501 := by
  sorry

end subtraction_division_l581_581565


namespace emir_needs_extra_money_l581_581640

def cost_dictionary : ℕ := 5
def cost_dinosaur_book : ℕ := 11
def cost_cookbook : ℕ := 5
def amount_saved : ℕ := 19

def total_cost : ℕ := cost_dictionary + cost_dinosaur_book + cost_cookbook
def amount_needed : ℕ := total_cost - amount_saved

theorem emir_needs_extra_money : amount_needed = 2 := by
  rfl -- actual proof that amount_needed equals 2 goes here
  -- Sorry can be used to skip if the proof needs additional steps.
  sorry

end emir_needs_extra_money_l581_581640


namespace find_original_number_l581_581160

theorem find_original_number : ∃ (n : ℕ), 8453719 > 3 * n ∧ (swap_digits n) = 8453719 ∧ n = 1453789 :=
sorry

-- Helper function to swap digits at two specified positions
-- This is a stub and should be implemented appropriately
def swap_digits (n : ℕ) : ℕ := sorry

end find_original_number_l581_581160


namespace correct_answer_is_B_l581_581130

def lack_of_eco_friendly_habits : Prop := true
def major_global_climate_change_cause (s : String) : Prop :=
  s = "cause"

theorem correct_answer_is_B :
  major_global_climate_change_cause "cause" ∧ lack_of_eco_friendly_habits → "B" = "cause" :=
by
  sorry

end correct_answer_is_B_l581_581130


namespace g_is_odd_l581_581842

def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_is_odd_l581_581842


namespace length_major_axis_of_ellipse_l581_581250

-- Definitions based on the problem statement
def foci_1 : ℝ × ℝ := (3, -4 + 2 * Real.sqrt 2)
def foci_2 : ℝ × ℝ := (3, -4 - 2 * Real.sqrt 2)
def center : ℝ × ℝ := ((foci_1.1 + foci_2.1) / 2, (foci_1.2 + foci_2.2) / 2)
def tangent_to_x_axis := center.2 + 4 -- because center is at (3, -4) and topmost point is at (3, 0)
def tangent_to_y_axis := center.1 == 3 -- peripheral condition indicating tangency to y-axis doesn't affect calculation

theorem length_major_axis_of_ellipse : 
  tangent_to_x_axis = 4 → 
  tangent_to_y_axis → 
  ∃ length : ℝ, length = 8 := 
by 
  intros _ _ 
  use 8 
  sorry

end length_major_axis_of_ellipse_l581_581250


namespace max_sum_abs_diff_l581_581064

theorem max_sum_abs_diff (a : list ℕ) (h1 : perm a (list.range (n + 1))) :
  ∃ s, s = (list.sum (list.map (λ (i : ℕ), |a[i] - (i + 1)|) (list.range n))) ∧
  s ≤ (n * (n + 1)) / 2 := 
sorry

end max_sum_abs_diff_l581_581064


namespace james_total_points_l581_581040

theorem james_total_points (field_goals field_goal_points shots shot_points : ℕ) :
  field_goals = 13 → field_goal_points = 3 → shots = 20 → shot_points = 2 →
  (field_goals * field_goal_points + shots * shot_points) = 79 :=
by
  intros hfg hfgp hs hsp
  rw [hfg, hfgp, hs, hsp]
  calc
    13 * 3 + 20 * 2 = 39 + 40 := by sorry
    ... = 79 := by sorry

end james_total_points_l581_581040


namespace line_intersects_at_least_one_of_skew_lines_l581_581316

variable (α β : Plane)
variable (m n l : Line)

-- Definitions based on conditions
/- m and n are skew lines means they are neither parallel nor intersect. -/
def skew_lines (m n : Line) : Prop :=
  ¬ (m ∥ n) ∧ ¬ (∃ P : Point, P ∈ m ∧ P ∈ n)

/- Intersection of planes α and β is line l. -/
def plane_intersection (α β : Plane) (l : Line) : Prop :=
  l ⊆ α ∧ l ⊆ β

-- Problem statement
theorem line_intersects_at_least_one_of_skew_lines
  (m_skew_n : skew_lines m n)
  (m_in_plane_α : m ⊆ α)
  (n_in_plane_β : n ⊆ β)
  (l_is_intersection : plane_intersection α β l) :
  ∃ P : Point, (P ∈ l ∧ P ∈ m) ∨ (P ∈ l ∧ P ∈ n) :=
sorry

end line_intersects_at_least_one_of_skew_lines_l581_581316


namespace area_product_equal_l581_581233

theorem area_product_equal {A B C D O : Type*} 
  (S1 S2 S3 S4 : ℝ)
  (hS1 : S1 = 0.5 * (dist A O) * (dist B O) * sin (angle A O B))
  (hS2 : S2 = 0.5 * (dist B O) * (dist C O) * sin (angle B O C))
  (hS3 : S3 = 0.5 * (dist C O) * (dist D O) * sin (angle C O D))
  (hS4 : S4 = 0.5 * (dist D O) * (dist A O) * sin (angle D O A))
  (hcyclic : true -- assuming quadrilateral is cyclic since this ensures angle sum relations):
  S1 * S3 = S2 * S4 :=
by sorry

end area_product_equal_l581_581233


namespace olivia_total_time_spent_l581_581878

theorem olivia_total_time_spent :
  ∀ (num_problems : ℕ) (time_per_problem : ℕ) (time_checking : ℕ),
    num_problems = 7 →
    time_per_problem = 4 →
    time_checking = 3 →
    num_problems * time_per_problem + time_checking = 31 :=
by
  intros num_problems time_per_problem time_checking h_num_problems h_time_per_problem h_time_checking
  rw [h_num_problems, h_time_per_problem, h_time_checking]
  sorry

end olivia_total_time_spent_l581_581878


namespace distinct_order_factorizations_identical_order_factorizations_l581_581559

-- Prime factorization of 1,000,000 is fixed
def prime_factors : ℕ := 1000000

-- Given functions to calculate the combinations
def combinations_with_repetition (n k : ℕ) : ℕ :=
  nat.choose (n + k - 1) (k - 1)

-- Number of ways to represent 1,000,000 as product of three factors with distinct order
theorem distinct_order_factorizations : nat :=
  let distribution_twos := combinations_with_repetition 6 3,
      distribution_fives := combinations_with_repetition 6 3
  in distribution_twos * distribution_fives -- Should be 28 * 28

-- Total ways considering distinct order
example : distinct_order_factorizations = 784 := by sorry

-- Number of ways to represent 1,000,000 as product of three factors with identical order
theorem identical_order_factorizations : nat := 139

-- Prove that the calculated identical order factorization is correct
example : identical_order_factorizations = 139 := by sorry

end distinct_order_factorizations_identical_order_factorizations_l581_581559


namespace average_revenue_per_item_l581_581454

def price_strawberry_smoothie : ℕ := 4
def quantity_strawberry_smoothie : ℕ := 20
def price_blueberry_smoothie : ℕ := 3
def quantity_blueberry_smoothie : ℕ := 30
def price_chocolate_cake : ℕ := 5
def quantity_chocolate_cake : ℕ := 15
def price_vanilla_cake : ℕ := 6
def quantity_vanilla_cake : ℕ := 10

theorem average_revenue_per_item :
  let total_sales := price_strawberry_smoothie * quantity_strawberry_smoothie +
                     price_blueberry_smoothie * quantity_blueberry_smoothie +
                     price_chocolate_cake * quantity_chocolate_cake +
                     price_vanilla_cake * quantity_vanilla_cake,
      total_quantity := quantity_strawberry_smoothie + quantity_blueberry_smoothie +
                        quantity_chocolate_cake + quantity_vanilla_cake
  in (total_sales : ℚ) / total_quantity = 4.07 := by
  sorry

end average_revenue_per_item_l581_581454


namespace part1_f0_f1_part1_f_neg1_f2_part1_f_neg2_f3_part2_conjecture_l581_581427

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem part1_f0_f1 : f 0 + f 1 = Real.sqrt 3 / 3 := sorry

theorem part1_f_neg1_f2 : f (-1) + f 2 = Real.sqrt 3 / 3 := sorry

theorem part1_f_neg2_f3 : f (-2) + f 3 = Real.sqrt 3 / 3 := sorry

theorem part2_conjecture (x1 x2 : ℝ) (h : x1 + x2 = 1) : f x1 + f x2 = Real.sqrt 3 / 3 := sorry

end part1_f0_f1_part1_f_neg1_f2_part1_f_neg2_f3_part2_conjecture_l581_581427


namespace _l581_581391

noncomputable def problem1 (a b : ℝ) (A B : ℝ) (cosB : ℝ) (hcos : cosB = 4/5) (hB : b = 2) (hA : A = 30) : ℝ :=
  a

noncomputable theorem problem1_answer : ∃ a : ℝ, ∀ (cosB : ℝ) (hcos : cosB = 4/5) (b : ℝ) (hb : b = 2) (A : ℝ) (hA : A = 30),
  problem1 a b A cosB hcos hb hA = 5/3 :=
begin
  sorry
end

noncomputable def problem2 (b : ℝ) (cosB : ℝ) (hcos : cosB = 4/5) (hB : b = 2) : ℝ :=
  let a := 5/3 in -- this is derived from the first solution step
  let c := a in   -- symmetry of triangle when maximizing area
  (1/2) * a * c * sin (acos cosB)

noncomputable theorem problem2_answer : ∃ area : ℝ, ∀ (cosB : ℝ) (hcos : cosB = 4/5) (b : ℝ) (hb : b = 2),
  problem2 b cosB hcos hb = 3 :=
begin
  sorry
end

end _l581_581391


namespace hyperbolas_asymptotes_and_focal_lengths_l581_581891

/-- Definitions for the hyperbolas C1 and C2 -/
def C1 (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1
def C2 (x y : ℝ) : Prop := y^2 / 2 - x^2 / 3 = 1

/-- Proof that asymptotes of C1 and C2 are the same and their focal lengths are the same -/
theorem hyperbolas_asymptotes_and_focal_lengths :
  (∀ (x y : ℝ), C1 x y ↔ C2 x y) ∧ 
  ((∃ c : ℝ, c = sqrt (3 + 2)) ∧ (∃ c' : ℝ, c' = sqrt (2 + 3)) ∧ ∀ c₁ c₂ : ℝ, c₁ = c₂) :=
begin
  sorry -- proof left as exercise
end

end hyperbolas_asymptotes_and_focal_lengths_l581_581891


namespace find_2023rd_term_mod_7_l581_581624

/-- 
In the sequence where each positive integer n appears n times, 
prove that the 2023rd term modulo 7 is 1.
-/
theorem find_2023rd_term_mod_7 :
  let seq := λ n : ℕ, (∑ i in Finset.range (n+1), i : ℕ)
  ∃ n : ℕ, 
  seq n ≥ 2023 ∧ seq n - seq (n - 1) = 2023 ∧ (n ≡ 1 [MOD 7]) :=
begin
  sorry
end

end find_2023rd_term_mod_7_l581_581624


namespace initial_percentage_reduction_l581_581484

theorem initial_percentage_reduction (x : ℝ) :
  (1 - x / 100) * 1.17649 = 1 → x = 15 :=
by
  sorry

end initial_percentage_reduction_l581_581484


namespace solution_set_of_inequality_l581_581411

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^2 - x else Real.log (x + 1) / Real.log 2

theorem solution_set_of_inequality :
  { x : ℝ | f x ≥ 2 } = { x : ℝ | x ∈ Set.Iic (-1) } ∪ { x : ℝ | x ∈ Set.Ici 3 } :=
by
  sorry

end solution_set_of_inequality_l581_581411


namespace point_A_l581_581882

theorem point_A'_in_second_quadrant
  (A : ℝ × ℝ)
  (hA : A = (-3, -1))
  (A' : ℝ × ℝ)
  (hA' : A' = (A.1 - 2, A.2 + 4)) :
  A'.1 < 0 ∧ A'.2 > 0 :=
by
  rw [hA, hA']
  show (-3 - 2) < 0 ∧ (-1 + 4) > 0
  simp
  exact ⟨by linarith, by linarith⟩

end point_A_l581_581882


namespace number_of_passed_boys_l581_581185

theorem number_of_passed_boys 
  (total_boys: ℕ)
  (avg_marks_all: ℕ)
  (avg_marks_passed: ℕ)
  (avg_marks_failed: ℕ)
  (h_total: total_boys = 120)
  (h_avg_all: avg_marks_all = 37)
  (h_avg_passed: avg_marks_passed = 39)
  (h_avg_failed: avg_marks_failed = 15) :
  ∃ P: ℕ, 
    (λ F: ℕ, total_boys = P + F ∧ 
             37 * total_boys = 39 * P + 15 * F) (120 - P) ∧ 
    P = 110 :=
by
  sorry

end number_of_passed_boys_l581_581185


namespace sum_of_odd_divisors_252_l581_581536

noncomputable def sum_of_odd_divisors (n : ℕ) : ℕ :=
  finset.sum (finset.filter (λ d, d % 2 = 1) (finset.divisors n)) id

theorem sum_of_odd_divisors_252 : sum_of_odd_divisors 252 = 104 :=
by
  sorry

end sum_of_odd_divisors_252_l581_581536


namespace unique_arrangements_in_grid_l581_581727

-- Define the type for the letters
inductive Letter
| a : Letter
| b : Letter
| c : Letter

open Letter

-- The grid will be a matrix of 3 rows and 2 columns
def Grid := Array (Array (Option Letter))

-- Function to check if a grid is valid (distinct letters in each row and column)
def valid_grid (g : Grid) : Bool :=
  (all_distinct g[0]) &&
  (all_distinct g[1]) &&
  (all_distinct g[2]) &&
  (distinct_columns g)

-- Helper function to check if an array has all distinct values
def all_distinct (arr : Array (Option Letter)) : Bool :=
  arr.toList.nodup

-- Helper function to check if all columns in the grid are distinct
def distinct_columns (g : Grid) : Bool :=
  (all_distinct (g.map (λ row => row[0]))) &&
  (all_distinct (g.map (λ row => row[1])))

-- Main theorem
theorem unique_arrangements_in_grid : 
  ∃ (grids : Finset Grid), (∀ g ∈ grids, valid_grid g) ∧ (grids.card = 12) :=
sorry

end unique_arrangements_in_grid_l581_581727


namespace log_base_2_eq_3_implies_x_eq_8_l581_581802

theorem log_base_2_eq_3_implies_x_eq_8 (x : ℝ) (h : Real.log 2 x = 3) : x = 8 :=
sorry

end log_base_2_eq_3_implies_x_eq_8_l581_581802


namespace count_perfect_squares_and_cubes_l581_581736

theorem count_perfect_squares_and_cubes : 
  let num_squares := 31 in
  let num_cubes := 9 in
  let num_sixth_powers := 3 in
  num_squares + num_cubes - num_sixth_powers = 37 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581736


namespace count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581793

theorem count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes :
  let num_perfect_squares := 31
  let num_perfect_cubes := 9
  let num_perfect_sixth_powers := 2
  let num_either_squares_or_cubes := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers
  in num_either_squares_or_cubes = 38 :=
by
  sorry

end count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581793


namespace count_same_remainder_11_12_l581_581656

theorem count_same_remainder_11_12 : 
  (∑ a in Finset.filter (λ a : ℕ, a % 11 = a % 12) (Finset.Icc 1 1000), 1) = 87 := by
  sorry

end count_same_remainder_11_12_l581_581656


namespace largest_number_of_primes_is_five_l581_581698

def sequence (a b : ℕ) (h₁ : a % 5 ≠ 0) (h₂ : b % 5 ≠ 0) : ℕ → ℕ
| 0     := 5
| (n+1) := a * sequence a b n + b

theorem largest_number_of_primes_is_five
  (a b : ℕ) (h₁ : a % 5 ≠ 0) (h₂ : b % 5 ≠ 0) :
  ∃ n, ∀ m < n, Nat.Prime (sequence a b m) ∧ Nat.NotPrime (sequence a b n) :=
sorry

end largest_number_of_primes_is_five_l581_581698


namespace tan_half_angle_of_sin_alpha_is_given_l581_581317

theorem tan_half_angle_of_sin_alpha_is_given 
  (α : Real) 
  (h1 : Real.sin α = 4 / 5)
  (h2 : π / 2 < α ∧ α < π) : Real.tan (α / 2) = 2 := 
by 
  sorry

end tan_half_angle_of_sin_alpha_is_given_l581_581317


namespace sqrt_fraction_identity_l581_581189

theorem sqrt_fraction_identity (n : ℕ) : 
    sqrt (n + 1 / (n + 2)) = ((n + 1) * sqrt (n + 2)) / (n + 2) := by 
    sorry

end sqrt_fraction_identity_l581_581189


namespace length_AD_l581_581377

/-- Given a quadrilateral ABCD with the following properties:
    - AB = 6 units
    - BC = 10 units
    - CD = 18 units
    - Angle B is a right angle
    - Angle C is a right angle
    - D is directly above C
    Prove that the length of segment AD is 2 * sqrt 41 units. -/
theorem length_AD {A B C D : Type} [inner_product_space ℝ B] [inner_product_space ℝ C] 
  (h1 : dist A B = 6)
  (h2 : dist B C = 10)
  (h3 : dist C D = 18)
  (h4 : ∠ B = π / 2)
  (h5 : ∠ C = π / 2)
  (h6 : vertical_above D C) : dist A D = 2 * real.sqrt 41 :=
sorry

end length_AD_l581_581377


namespace pedal_triangle_b1c1_length_l581_581058

open Function

variables {A B C P A1 B1 C1 : Type} {BC AP R : ℝ}

-- Assume the definitions given by the conditions
def is_perpendicular (p q : Type) : Prop := sorry
def is_pedal_triangle (P : Type) (A B C : Type) (A1 B1 C1 : Type) : Prop := sorry
def circumradius (A B C : Type) : ℝ := R -- assuming the value of circumradius is given as R

-- The statement we need to prove
theorem pedal_triangle_b1c1_length 
  (h1 : is_perpendicular P BC) 
  (h2 : is_pedal_triangle P A B C A1 B1 C1)
  (h3 : circumradius A B C = R) :
  B1C1 = BC * AP / (2 * R) :=
sorry

end pedal_triangle_b1c1_length_l581_581058


namespace stmtA_stmtB_stmtC_stmtD_l581_581177

-- Definitions from the conditions
variable {R : Type*} [LinearOrderedField R]

noncomputable def P : Prop := ∃ x₀ : R, x₀^2 + 2*x₀ + 2 < 0
noncomputable def not_P : Prop := ∀ x : R, x^2 + 2*x + 2 ≥ 0

noncomputable def has_opposite_roots (m : R) : Prop := 
  ∃ x₁ x₂ : R, x₁ * x₂ < 0 ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0

def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x
def even_function (g : R → R) : Prop := ∀ x : R, g (-x) = g x
def even_h (f g : R → R) : Prop := odd_function f → even_function g → ∀ x : R, f (g x) = f (g (-x))

-- Required proof statements
theorem stmtA : P → not_P := by sorry
theorem stmtB (m : R) : has_opposite_roots m → m < 0 := by sorry
theorem stmtC (f g : R → R) : even_h f g := by sorry
theorem stmtD (x y : R) : x > y → ∃ x y, sqrt x > sqrt y := by sorry

end stmtA_stmtB_stmtC_stmtD_l581_581177


namespace f_f_2_eq_2_l581_581426

def f (x : ℝ) : ℝ := 
  if x < 2 then 
    2 * Real.exp (x - 1) 
  else 
    Real.log (x ^ 2 - 1) / Real.log 3

theorem f_f_2_eq_2 : f (f 2) = 2 := 
by 
  sorry

end f_f_2_eq_2_l581_581426


namespace total_bill_cost_l581_581103

-- Definitions of costs and conditions
def curtis_meal_cost : ℝ := 16.00
def rob_meal_cost : ℝ := 18.00
def total_cost_before_discount : ℝ := curtis_meal_cost + rob_meal_cost
def discount_rate : ℝ := 0.5
def time_of_meal : ℝ := 3.0

-- Condition for discount applicability
def discount_applicable : Prop := 2.0 ≤ time_of_meal ∧ time_of_meal ≤ 4.0

-- Total cost with discount applied
def cost_with_discount (total_cost : ℝ) (rate : ℝ) : ℝ := total_cost * rate

-- Theorem statement we need to prove
theorem total_bill_cost :
  discount_applicable →
  cost_with_discount total_cost_before_discount discount_rate = 17.00 :=
by
  sorry

end total_bill_cost_l581_581103


namespace local_minimum_point_of_f_l581_581710

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem local_minimum_point_of_f :
  ∃ x₀ : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x₀| < δ → f(x₀) ≤ f(x)) ∧ x₀ = 2 :=
sorry

end local_minimum_point_of_f_l581_581710


namespace iso_triangle_parallel_angles_l581_581009

theorem iso_triangle_parallel_angles (X Y Z F G : Type*) 
  (XY_eq_YZ : XY = YZ) 
  (m_angle_FZY : ∠FZY = 30) 
  (FZ_parallel_XY : FZ ∥ XY) : ∠GFZ = 30 :=
sorry

end iso_triangle_parallel_angles_l581_581009


namespace minimum_side_length_of_wrapping_paper_l581_581936

theorem minimum_side_length_of_wrapping_paper (a : ℝ) : 
    ∃ l : ℝ, l = (sqrt 2 + sqrt 6) / 2 * a :=
begin
  sorry
end

end minimum_side_length_of_wrapping_paper_l581_581936


namespace son_l581_581203

theorem son's_present_age
  (S F : ℤ)
  (h1 : F = S + 45)
  (h2 : F + 10 = 4 * (S + 10))
  (h3 : S + 15 = 2 * S) :
  S = 15 :=
by
  sorry

end son_l581_581203


namespace HB_HC_K_collinear_l581_581561

theorem HB_HC_K_collinear 
  (A B C I H_B H_C K : Type)
  [IsTriangle A B C] 
  [Incenter I A B C]
  [Orthocenter H_B A C I]
  [Orthocenter H_C A B I]
  [IncircleContactPoint K A B C] 
  : Collinear H_B H_C K :=
  sorry

end HB_HC_K_collinear_l581_581561


namespace Q_value_l581_581418

noncomputable theory

def is_arithmetic_sequence (a d: Int) (P: Int): Prop := 
  ∃ k: Int, P = a + k * d

def log_base_sum (base log_arg: Int) (terms: List Int) : Real := 
  terms.foldl (λ sum k => sum + Real.logBase base (log_arg^k)) 0

theorem Q_value : ∀ Q : Real, 
  (Q = log_base_sum 128 2 [3, 5, 7, ..., 95]) → Q = 329 :=
sorry

end Q_value_l581_581418


namespace num_ways_select_at_least_one_defective_l581_581006

theorem num_ways_select_at_least_one_defective :
  let total_products := 100
  let defective_products := 6
  let selected_products := 3
  finset.card (finset.range (total_products + 1).choose selected_products) - 
  finset.card (finset.range (total_products - defective_products + 1).choose selected_products) 
  = finset.card (finset.range (total_products + 1)).choose selected_products - 
    finset.card (finset.range (total_products - defective_products + 1)).choose selected_products :=
sorry

end num_ways_select_at_least_one_defective_l581_581006


namespace total_cups_sold_is_46_l581_581893

-- Define the number of cups sold last week
def cups_sold_last_week : ℕ := 20

-- Define the percentage increase
def percentage_increase : ℕ := 30

-- Calculate the number of cups sold this week
def cups_sold_this_week : ℕ := cups_sold_last_week + (cups_sold_last_week * percentage_increase / 100)

-- Calculate the total number of cups sold over both weeks
def total_cups_sold : ℕ := cups_sold_last_week + cups_sold_this_week

-- State the theorem to prove the total number of cups sold
theorem total_cups_sold_is_46 : total_cups_sold = 46 := sorry

end total_cups_sold_is_46_l581_581893


namespace cows_number_l581_581368

theorem cows_number (D C : ℕ) (L H : ℕ) 
  (h1 : L = 2 * D + 4 * C)
  (h2 : H = D + C)
  (h3 : L = 2 * H + 12) 
  : C = 6 := 
by
  sorry

end cows_number_l581_581368


namespace colored_pencils_count_l581_581501

-- Given conditions
def bundles := 7
def pencils_per_bundle := 10
def extra_colored_pencils := 3

-- Calculations based on conditions
def total_pencils : ℕ := bundles * pencils_per_bundle
def total_colored_pencils : ℕ := total_pencils + extra_colored_pencils

-- Statement to be proved
theorem colored_pencils_count : total_colored_pencils = 73 := by
  sorry

end colored_pencils_count_l581_581501


namespace ryegrass_percentage_Y_is_25_l581_581455

-- Definitions based on the given problem
def seed_mixture_X_ryegrass_percentage := 0.4
def final_mixture_ryegrass_percentage := 0.27
def seed_mixture_X_percentage_in_final_mixture := 0.13333333333333332

-- Variables
variable (R : ℝ)

-- Conditions
def seed_mixture_Y_percentage_in_final_mixture := 0.8666666666666667
def prerequisite_mixture_eq := (final_mixture_ryegrass_percentage = 
  seed_mixture_X_percentage_in_final_mixture * seed_mixture_X_ryegrass_percentage +
  seed_mixture_Y_percentage_in_final_mixture * R)

-- The theorem to prove that the percentage of ryegrass in seed mixture Y is 25%
theorem ryegrass_percentage_Y_is_25 (h : prerequisite_mixture_eq) : R = 0.25 :=
sorry

end ryegrass_percentage_Y_is_25_l581_581455


namespace arithmetic_expression_evaluation_l581_581248

theorem arithmetic_expression_evaluation :
  (3 + 9) ^ 2 + (3 ^ 2) * (9 ^ 2) = 873 :=
by
  -- Proof is skipped, using sorry for now.
  sorry

end arithmetic_expression_evaluation_l581_581248


namespace line_through_point_parallel_l581_581126

theorem line_through_point_parallel (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) (hA : A = (2, 3)) (hl : ∀ x y, l x y ↔ 2 * x - 4 * y + 7 = 0) :
  ∃ m, (∀ x y, (2 * x - 4 * y + m = 0) ↔ (x - 2 * y + 4 = 0)) ∧ (2 * (A.1) - 4 * (A.2) + m = 0) := 
sorry

end line_through_point_parallel_l581_581126


namespace largest_perfect_square_factor_of_882_l581_581170

theorem largest_perfect_square_factor_of_882 : ∃ n, n * n = 441 ∧ ∀ m, m * m ∣ 882 → m * m ≤ 441 := 
by 
 sorry

end largest_perfect_square_factor_of_882_l581_581170


namespace area_of_triangle_ABC_l581_581908

theorem area_of_triangle_ABC
  (A B C P: Point)
  (h1 : RightTriangle A B C)
  (h2 : OnHypotenuse P A C)
  (h3 : ∠ A B P = 45)
  (h4 : dist A P = 3)
  (h5 : dist P C = 6) :

  area A B C = 81 / 5 :=
sorry

end area_of_triangle_ABC_l581_581908


namespace option_B_is_valid_distribution_l581_581951

def is_probability_distribution (p : List ℚ) : Prop :=
  p.sum = 1 ∧ ∀ x ∈ p, 0 < x ∧ x ≤ 1

theorem option_B_is_valid_distribution : is_probability_distribution [1/2, 1/3, 1/6] :=
by
  sorry

end option_B_is_valid_distribution_l581_581951


namespace ellipse_and_circle_proof_l581_581719

-- Definition of points M and N
def M := (2 : ℝ, real.sqrt 2)
def N := (real.sqrt 6, 1 : ℝ)

-- Definition of the ellipse E with certain a and b
def Ellipse (a b : ℝ) (x y : ℝ) :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Mathematical equivalent proof problem
theorem ellipse_and_circle_proof :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ Ellipse a b M.1 M.2 ∧ Ellipse a b N.1 N.2 ∧
  (a^2 = 8) ∧ (b^2 = 4) ∧ ∃ r : ℝ, (r = 2 * real.sqrt 2 / 3) ∧ (r^2 = 8 / 3) ∧
  ∀ (x y : ℝ), (x^2 + y^2 = r^2) → (∃ (k m : ℝ), (m^2 > 2) ∧ (3 * m^2 > 8)) :=
sorry

end ellipse_and_circle_proof_l581_581719


namespace walter_exceptional_days_l581_581944

theorem walter_exceptional_days :
  ∃ (b w : ℕ), b + w = 13 ∧ 4 * b + 7 * w = 65 ∧ w = 9 :=
by
  existsi (5 : ℕ)
  existsi (9 : ℕ)
  split
  . exact rfl
  split
  . exact rfl
  .   exact rfl

end walter_exceptional_days_l581_581944


namespace negation_of_p_l581_581337

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, x ≥ 2

-- State the proof problem as a Lean theorem
theorem negation_of_p : (∀ x : ℝ, x ≥ 2) → ∃ x₀ : ℝ, x₀ < 2 :=
by
  intro h
  -- Define how the proof would generally proceed
  -- as the negation of a universal statement is an existential statement.
  sorry

end negation_of_p_l581_581337


namespace exists_number_square_contains_digit_sequence_2018_l581_581266

-- State the assertion that a specific number n has a square containing the sequence "2018"
theorem exists_number_square_contains_digit_sequence_2018 : 
  ∃ n : ℕ, (5002018 = n) ∧ (decimal_representation(n^2).contains("2018")) :=
sorry

end exists_number_square_contains_digit_sequence_2018_l581_581266


namespace geometric_sequence_fraction_l581_581494

section geometric_sequence
variables (a : ℕ → ℝ) (q : ℝ)
hypothesis h_seq : ∀ n : ℕ, 0 < a n
hypothesis h_geo : ∀ n : ℕ, a (n + 1) = a n * q
hypothesis h_arth_seq : 2 * a 1, (a 3) / 2, a 2 form an arithmetic sequence

theorem geometric_sequence_fraction (h_seq : ∀ n, 0 < a n) (h_geo : ∀ n, a (n + 1) = a n * q)
(h_arth_seq : (a 3 / 2) = (2 * a 1 + a 2) / 2) :
(a 2017 + a 2016) / (a 2015 + a 2014) = 4 :=
by sorry

end geometric_sequence

end geometric_sequence_fraction_l581_581494


namespace projection_z_component_l581_581660

/-- Define the input vector u and the plane parameters. -/
def u : ℝ × ℝ × ℝ := (2, 3, 4)

def normal_vector : ℝ × ℝ × ℝ := (2, 1, -3)

def plane (x y z : ℝ) : Prop := 2 * x + y - 3 * z = 0

/-- Prove that the z-component of the projection of vector u onto the plane is 41/14. -/
theorem projection_z_component :
  let n : ℝ × ℝ × ℝ := normal_vector in
  let (u₁, u₂, u₃) := u in
  let (n₁, n₂, n₃) := n in
  let dot_product_u_n := u₁ * n₁ + u₂ * n₂ + u₃ * n₃ in
  let n_square := n₁ * n₁ + n₂ * n₂ + n₃ * n₃ in
  let proj_u_n := (dot_product_u_n / n_square) * n in
  let projection := (u₁ - proj_u_n.1, u₂ - proj_u_n.2, u₃ - proj_u_n.3) in
  projection.2 = 41 / 14 :=
by sorry

end projection_z_component_l581_581660


namespace prob_ab_divisible_by_4_l581_581950

/-- Define the event that a number on an 8-sided die is divisible by 4 -/
def is_divisible_by_4 (n : ℕ) : Prop := n = 4 ∨ n = 8

/-- Define the probability of an event occurring for a fair 8-sided die -/
def probability (p : Prop) : ℚ :=
  if p then 1 / 8 else 0

/-- Define the probability that both a and b (results of rolling two fair 8-sided dice) are divisible by 4 -/
noncomputable def prob_both_divisible_by_4 : ℚ :=
  probability (is_divisible_by_4 4) * probability (is_divisible_by_4 8)

/-- The main theorem statement -/
theorem prob_ab_divisible_by_4 : prob_both_divisible_by_4 = 1 / 16 :=
sorry

end prob_ab_divisible_by_4_l581_581950


namespace distance_between_parallel_lines_l581_581506

theorem distance_between_parallel_lines (r : ℝ) (d : ℝ) (h1 : ∃ (C D E F : ℝ), CD = 38 ∧ EF = 38 ∧ DE = 34) :
  d = 6 :=
begin
  sorry
end

end distance_between_parallel_lines_l581_581506


namespace exists_function_f_l581_581959

/-- Define the divisor function -/
def d (m : ℕ) : ℕ :=
  (finset.Ico 1 (m + 1)).filter (λ x, m % x = 0).card

/-- Prove the existence of a function f : ℕ → ℕ
    such that (1) there exists n ∈ ℕ such that f(n) ≠ n
             (2) the number of divisors of m is f(n) ↔ the number of divisors of f(m) is n -/
theorem exists_function_f :
  ∃ f : ℕ → ℕ, (∃ n : ℕ, f n ≠ n) ∧ ∀ m n : ℕ, d m = f n ↔ d (f m) = n :=
sorry

end exists_function_f_l581_581959


namespace sum_first_10_terms_l581_581338

-- Define the sequence as per the given conditions
def seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) + 2 * a n = 0

def a_2_value : ℤ := -6

-- Statement to be proved: The sum of the first 10 terms is -1023
theorem sum_first_10_terms (a : ℕ → ℤ) 
  (h_seq : seq a) (h_a2 : a 2 = a_2_value) : 
  (∑ i in Finset.range 10, a i) = -1023 := 
  sorry

end sum_first_10_terms_l581_581338


namespace simplify_and_evaluate_expression_l581_581899

theorem simplify_and_evaluate_expression (a : ℕ) (h : a = 2023) :
  a * (1 - 2 * a) + 2 * (a + 1) * (a - 1) = 2021 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l581_581899


namespace ratio_of_shaded_to_white_areas_l581_581529

-- Define the conditions as given
def vertices_are_in_middle_squares (largest_square : Type) (small_squares : list largest_square) : Prop := 
  ∀ small_square ∈ small_squares, 
  vertices_of_small_square_at_middle_of_largest_square_vertices largest_square small_square

-- Define the theorem with the given question and correct answer
theorem ratio_of_shaded_to_white_areas
  (largest_square : Type) 
  (small_squares : list largest_square)
  (h: vertices_are_in_middle_squares largest_square small_squares) :
  ratio_of_areas_of_shaded_to_white largest_square small_squares = 5/3 :=
sorry

end ratio_of_shaded_to_white_areas_l581_581529


namespace complex_fraction_eq_l581_581974

theorem complex_fraction_eq (i : ℂ) (h : i^2 = -1) : (5 * (-h * i) / (2 - (-h * i))) = -1 + 2 * (-h * i) := by
  sorry

end complex_fraction_eq_l581_581974


namespace mul_powers_same_base_l581_581551

theorem mul_powers_same_base : 2^2 * 2^3 = 2^5 :=
by sorry

end mul_powers_same_base_l581_581551


namespace base_conversion_zero_l581_581111

theorem base_conversion_zero (A B : ℕ) (hA : A < 8) (hB : B < 6) (h : 8 * A + B = 6 * B + A) : 8 * A + B = 0 :=
by
  sorry

end base_conversion_zero_l581_581111


namespace ratio_of_w_to_y_l581_581137

variables (w x y z : ℚ)

theorem ratio_of_w_to_y:
  (w / x = 5 / 4) →
  (y / z = 5 / 3) →
  (z / x = 1 / 5) →
  (w / y = 15 / 4) :=
by
  intros hwx hyz hzx
  sorry

end ratio_of_w_to_y_l581_581137


namespace determine_weight_two_weighings_l581_581201

theorem determine_weight_two_weighings :
  ∃ (x : ℝ), (∃ n : ℕ, n ≤ 2 ∧ ∀ b1 b2 : list ℝ, perform_weighings b1 b2 x n)
  → (∃ w ∈ {7, 8, 9, 10, 11, 12, 13}, balance_weighted_bag b1 b2 w) :=
sorry

end determine_weight_two_weighings_l581_581201


namespace swapped_digits_greater_by_18_l581_581594

theorem swapped_digits_greater_by_18 (x : ℕ) : 
  (10 * x + 1) - (10 + x) = 18 :=
  sorry

end swapped_digits_greater_by_18_l581_581594


namespace average_power_heater_l581_581820

structure Conditions where
  (M : ℝ)    -- mass of the piston
  (tau : ℝ)  -- time period τ
  (a : ℝ)    -- constant acceleration
  (c : ℝ)    -- specific heat at constant volume
  (R : ℝ)    -- universal gas constant

theorem average_power_heater (cond : Conditions) : 
  let P := cond.M * cond.a^2 * cond.tau / 2 * (1 + cond.c / cond.R)
  P = (cond.M * cond.a^2 * cond.tau / 2) * (1 + cond.c / cond.R) :=
by
  sorry

end average_power_heater_l581_581820


namespace solution_exists_real_solution_31_l581_581663

theorem solution_exists_real_solution_31 :
  ∃ x : ℝ, (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10 ∧ 
            (x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15) :=
sorry

end solution_exists_real_solution_31_l581_581663


namespace eggs_in_box_and_weight_l581_581345

def initial_eggs : ℕ := 47

def eggs_added_improper_fraction : ℚ := 17 / 3

def total_eggs (init : ℕ) (added : ℚ) : ℚ :=
  init + added

def weight_in_ounces : ℚ := 143.5

def conversion_factor : ℚ := 28.3495

def weight_in_grams (ounces : ℚ) (conv_factor : ℚ) : ℚ :=
  ounces * conv_factor

theorem eggs_in_box_and_weight :
  total_eggs initial_eggs eggs_added_improper_fraction = (52 : ℚ) ∧
  weight_in_grams weight_in_ounces conversion_factor ≈ 4067.86 :=
by sorry

end eggs_in_box_and_weight_l581_581345


namespace total_leaves_on_farm_l581_581222

noncomputable def number_of_branches : ℕ := 10
noncomputable def sub_branches_per_branch : ℕ := 40
noncomputable def leaves_per_sub_branch : ℕ := 60
noncomputable def number_of_trees : ℕ := 4

theorem total_leaves_on_farm :
  number_of_branches * sub_branches_per_branch * leaves_per_sub_branch * number_of_trees = 96000 :=
by
  sorry

end total_leaves_on_farm_l581_581222


namespace power_function_through_point_l581_581715

theorem power_function_through_point :
  ∃ (a : ℝ), (∀ (x : ℝ), f x = x^a) ∧ f (1/2) = (1/√2)/2 :=
by
  let f (x : ℝ) := x^(1/2)
  use 1/2
  split
  sorry
  sorry

end power_function_through_point_l581_581715


namespace circle_line_intersection_angle_eq_l581_581303

theorem circle_line_intersection_angle_eq (a : ℝ) :
  let C := { p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * a * p.1 - 2 * p.2 + 2 = 0 },
      line := { p : ℝ × ℝ | p.2 = p.1 },
      A : ℝ × ℝ := sorry, -- Intersection point A
      B : ℝ × ℝ := sorry, -- Intersection point B
      C_center : ℝ × ℝ := (a, 1),
      r := real.sqrt (a^2 - 1),
      d := abs (a - 1) / real.sqrt 2 in
  (∠ACB = π / 3) ->
  a = -5 :=
sorry

end circle_line_intersection_angle_eq_l581_581303


namespace sum_of_all_n_for_postage_l581_581667

noncomputable def solve_sum_of_n (n : ℕ) : ℕ :=
  if ∃ m : ℕ, m ≤ 70 ∧
  (∀ (a b c : ℕ), (70 < (a * 3 + b * n + c * (n + 1)))) ∧
  (∀ x : ℕ, x ≠ 70 → (70 < (a * 3 + b * n + c * (n + 1)) ∨ x < 70)) then
    37
  else 0

theorem sum_of_all_n_for_postage (n : ℕ) (h1 : 3 > 0) (h2 : n > 0) (h3 : (n + 1) > 0) :
  solve_sum_of_n n = 37 := by
  sorry

end sum_of_all_n_for_postage_l581_581667


namespace ladybugs_calculation_l581_581460

def total_ladybugs : ℕ := 67082
def ladybugs_with_spots : ℕ := 12170
def ladybugs_without_spots : ℕ := 54912

theorem ladybugs_calculation :
  total_ladybugs - ladybugs_with_spots = ladybugs_without_spots :=
by
  sorry

end ladybugs_calculation_l581_581460


namespace common_difference_of_arithmetic_seq_l581_581691

-- Definition of arithmetic sequence sum and general term
def arithmetic_sum(n a1 d : ℕ) : ℕ := n * a1 + (n * (n - 1) / 2) * d
def arithmetic_term(n a1 d : ℕ) : ℕ := a1 + (n - 1) * d

theorem common_difference_of_arithmetic_seq (a1 d : ℕ) :
  arithmetic_sum 13 a1 d = 104 ∧ arithmetic_term 6 a1 d = 5 → d = 3 :=
by {
  sorry
}

end common_difference_of_arithmetic_seq_l581_581691


namespace error_in_reasoning_l581_581493

-- Definitions based on given conditions
def rhombus_is_parallelogram : Prop := ∀ R : Type, is_rhombus R → is_parallelogram R
def quadrilateral_ABCD_is_parallelogram : Prop := is_parallelogram quadrilateral_ABCD

-- The theorem to be proved
theorem error_in_reasoning (h1 : rhombus_is_parallelogram) (h2 : quadrilateral_ABCD_is_parallelogram):
  ¬(quadrilateral_ABCD_is_rhombus) :=
sorry

end error_in_reasoning_l581_581493


namespace part_a_number_of_9s_correct_part_b_sum_of_digits_correct_l581_581079

noncomputable def number_of_nines : ℕ := 
  -- The number of nines that appear in the sequence
  List.sum (List.range (100 + 1))

theorem part_a_number_of_9s_correct : 
  -- Prove that the calculated number of nines is 5050
  number_of_nines = 5050 := 
by 
  sorry

open Finset

noncomputable def S : ℤ := 
  -- Sum of the sequence: (10^2 - 3) + (10^3 - 3) + ... + (10^101 - 3)
  Finset.sum (range 100) (λ k, 10 ^ (k + 2) - 3)

noncomputable def sum_of_digits_of (n : ℤ) : ℕ := 
  -- Calculate sum of digits of n
  natDigits 10 n |>.sum

theorem part_b_sum_of_digits_correct :
  -- Prove that sum of the digits of S is 106
  sum_of_digits_of S = 106 := 
by
  sorry

end part_a_number_of_9s_correct_part_b_sum_of_digits_correct_l581_581079


namespace shaded_area_of_equilateral_triangle_l581_581832

noncomputable def radius_of_circle (l : ℕ) := 
  (5 * Real.sqrt 3) / 2

noncomputable def area_of_one_shaded_region (l : ℕ) := 
  (4 * Real.pi - 3 * Real.sqrt 3) / 12 * l^2

theorem shaded_area_of_equilateral_triangle (AB : ℝ) 
  (h : AB = 10): 
  ∃ (A : ℝ), A = (100 * Real.pi - 75 * Real.sqrt 3) / 8 := 
by
  let radius := radius_of_circle 10
  let area := area_of_one_shaded_region radius
  use 2 * area
  sorry

end shaded_area_of_equilateral_triangle_l581_581832


namespace transformed_sine_function_l581_581474

theorem transformed_sine_function :
  (∀ x : ℝ, y = sin x) →
  (∀ x : ℝ, y = sin (x + (π / 3))) →
  (∀ x : ℝ, y = sin ((x / 2) + (π / 3))) :=
sorry

end transformed_sine_function_l581_581474


namespace real_solutions_l581_581664

noncomputable def solveEquation (x : ℝ) : Prop :=
  (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10

theorem real_solutions :
  {x : ℝ | solveEquation x} = {x : ℝ | x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15} :=
by
  sorry

end real_solutions_l581_581664


namespace angle_A_eq_60_degree_l581_581034

theorem angle_A_eq_60_degree
  {α β γ : ℝ}
  (A B C B1 C1 X : Type)
  [Triangle ABC]
  [angle_bisector B B1 A C]
  [angle_bisector C C1 A B]
  (h1 : X ∈ BC)
  (h2 : cyclic_quadrilateral (ABB1)) 
  (h3 : cyclic_quadrilateral (ACC1)) 
  (hXYB : X ∈ [ABB1])
  (hXYC : X ∈ [ACC1]) 
  (hA : ∠ A = 60) :
  ∀ A B C : Type, ∠ A = 60 :=
by
  -- Proof goes here
  sorry

end angle_A_eq_60_degree_l581_581034


namespace probability_of_triangle_l581_581502

/-- There are 12 figures in total: 4 squares, 5 triangles, and 3 rectangles.
    Prove that the probability of choosing a triangle is 5/12. -/
theorem probability_of_triangle (total_figures : ℕ) (num_squares : ℕ) (num_triangles : ℕ) (num_rectangles : ℕ)
  (h1 : total_figures = 12)
  (h2 : num_squares = 4)
  (h3 : num_triangles = 5)
  (h4 : num_rectangles = 3) :
  num_triangles / total_figures = 5 / 12 :=
sorry

end probability_of_triangle_l581_581502


namespace tangent_line_FE_circumcircle_EGH_l581_581014

-- Definitions for cyclic quadrilateral, intersection points, and midpoints
variables {A B C D E F G H : Point}
variables (cyclic_ABCD : CyclicQuadrilateral A B C D)
variables (intersect_AC_BD : IntersectPoint A C B D E)
variables (intersect_AD_BC : IntersectPoint A D B C F)
variables (midpoint_G : Midpoint G A B)
variables (midpoint_H : Midpoint H C D)

-- Main theorem statement
theorem tangent_line_FE_circumcircle_EGH :
  tangent_line_FE_circumcircle E G H F :=
by {
  sorry
}

end tangent_line_FE_circumcircle_EGH_l581_581014


namespace and_or_distrib_left_or_and_distrib_right_l581_581188

theorem and_or_distrib_left (A B C : Prop) : A ∧ (B ∨ C) ↔ (A ∧ B) ∨ (A ∧ C) :=
sorry

theorem or_and_distrib_right (A B C : Prop) : A ∨ (B ∧ C) ↔ (A ∨ B) ∧ (A ∨ C) :=
sorry

end and_or_distrib_left_or_and_distrib_right_l581_581188


namespace vertices_form_parabola_l581_581422

noncomputable def parabola_vertices (a c : ℝ) (t : ℝ) : ℝ × ℝ :=
let x_t := -((4 + t) / (2 * a)) in
let y_t := a * x_t ^ 2 + (4 + t) * x_t + c in
(x_t, y_t)

noncomputable def set_of_vertices (a c : ℝ) : set (ℝ × ℝ) :=
{p | ∃ t, p = parabola_vertices a c t}

theorem vertices_form_parabola (a c : ℝ) (h₁ : a = 2) (h₂ : c = 3) :
  set_of_vertices a c = {p | ∃ x, p = (x, -2 * x^2 + 2)} :=
by
  sorry

end vertices_form_parabola_l581_581422


namespace rectangle_right_triangle_max_area_and_hypotenuse_l581_581480

theorem rectangle_right_triangle_max_area_and_hypotenuse (x y h : ℝ) (h_triangle : h^2 = x^2 + y^2) (h_perimeter : 2 * (x + y) = 60) :
  (x * y ≤ 225) ∧ (x = 15) ∧ (y = 15) ∧ (h = 15 * Real.sqrt 2) :=
by
  sorry

end rectangle_right_triangle_max_area_and_hypotenuse_l581_581480


namespace tangent_line_eval_l581_581471

variable {f : ℝ → ℝ}

theorem tangent_line_eval (h_tangent : ∀ x, x = 1 → 3 * x + (λ x, -3 * x + 4) x - 4 = 0) : 
  f(1) + (derivative f) 1 = -2 :=
begin
  sorry
end

end tangent_line_eval_l581_581471


namespace simple_interest_principal_l581_581659

theorem simple_interest_principal (A r t : ℝ) (ht_pos : t > 0) (hr_pos : r > 0) (hA_pos : A > 0) :
  (A = 1120) → (r = 0.08) → (t = 2.4) → ∃ (P : ℝ), abs (P - 939.60) < 0.01 :=
by
  intros hA hr ht
  -- Proof would go here
  sorry

end simple_interest_principal_l581_581659


namespace real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_zero_condition_l581_581324

variable (m : ℝ) -- Considering m to be a real number.

theorem real_number_condition : (m^2 - m - 6 = 0) → (m = 3 ∨ m = -2) :=
by sorry

theorem imaginary_number_condition : (m^2 - m - 6 ≠ 0) → (m ≠ 3 ∧ m ≠ -2) :=
by sorry

theorem pure_imaginary_number_condition : (m^2 - 3m = 0 ∧ m^2 - m - 6 ≠ 0) → m = 0 :=
by sorry

theorem zero_condition : (m^2 - 3m = 0 ∧ m^2 - m - 6 = 0) → m = 3 :=
by sorry

end real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_zero_condition_l581_581324


namespace concurrency_of_lines_l581_581854

theorem concurrency_of_lines 
  (ABCD : Rectangle)
  (P Q : Point)
  (K M N X Y Z : Point)
  (ℓ₁ ℓ₂ ℓ₃ : Line) 
  (hP : P.midpoint (ABCD.B) (ABCD.C))
  (hQ : Q.midpoint (ABCD.C) (ABCD.D))
  (hK : K.intersection (Line_through_point (ABCD.D) P) (Line_through_point Q (ABCD.B)))
  (hM : M.intersection (Line_through_point (ABCD.D) P) (Line_through_point Q (ABCD.A)))
  (hN : N.intersection (Line_through_point P (ABCD.A)) (Line_through_point Q (ABCD.B)))
  (hX : X.midpoint (ABCD.A) N)
  (hY : Y.midpoint K N)
  (hZ : Z.midpoint (ABCD.A) M)
  (hℓ₁ : ℓ₁.is_perpendicular_to (Line_through_point K M) (Line_through_point X P))
  (hℓ₂ : ℓ₂.is_perpendicular_to (Line_through_point (ABCD.A) M) (Line_through_point Y P))
  (hℓ₃ : ℓ₃.is_perpendicular_to (Line_through_point K N) (Line_through_point Z P)) : 
  ∃ (O : Point), O ∈ ℓ₁ ∧ O ∈ ℓ₂ ∧ O ∈ ℓ₃ := 
sorry

end concurrency_of_lines_l581_581854


namespace count_perfect_squares_cubes_under_1000_l581_581751

-- Condition definitions
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, m * m = n
def isPerfectCube (n : Nat) : Prop := ∃ m : Nat, m * m * m = n

-- Main proof problem
theorem count_perfect_squares_cubes_under_1000 : 
  let count (p : Nat -> Prop) := Finset.card (Finset.filter p (Finset.range 1000))
  count (λ n, isPerfectSquare n ∨ isPerfectCube n) = 37 := 
sorry

end count_perfect_squares_cubes_under_1000_l581_581751


namespace binomial_sum_id_l581_581173

open Nat

theorem binomial_sum_id (n1 n2 n3 n4 : ℕ) (H1: binomial n1 n2 + binomial n1 n3 = binomial n4 n2) : 
  (n2 + n3 = 26) :=
by
  sorry

end binomial_sum_id_l581_581173


namespace yearly_water_consumption_correct_l581_581545

def monthly_water_consumption : ℝ := 182.88
def months_in_a_year : ℕ := 12
def yearly_water_consumption : ℝ := monthly_water_consumption * (months_in_a_year : ℝ)

theorem yearly_water_consumption_correct :
  yearly_water_consumption = 2194.56 :=
by
  sorry

end yearly_water_consumption_correct_l581_581545


namespace sufficient_but_not_necessary_condition_l581_581410

variables (a b : ℝ)

def p : Prop := a > b ∧ b > 1
def q : Prop := a - b < a^2 - b^2

theorem sufficient_but_not_necessary_condition (h : p a b) : q a b :=
  sorry

end sufficient_but_not_necessary_condition_l581_581410


namespace intersections_collinear_a_intersections_collinear_b_l581_581932

--Define trapezoid and intersection conditions
variable {Point : Type}

structure Trapezoid (P1 P2 P3 P4 : Point) :=
(base1 : P1 ≠ P2)
(base2 : P3 ≠ P4)
(distinct_lengths : P1 ≠ P3 ∧ P2 ≠ P4)

def intersection (l1 l2 : list Point) : Point := sorry

axiom trapezoid_ABCD_APQD (A B C D P Q : Point) (trapezoid1 : Trapezoid A B C D) 
    (trapezoid2 : Trapezoid A P Q D) : 
    (distinct_intersections : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (A ≠ P) ∧ (P ≠ Q) ∧ (Q ≠ D) ∧ (B ≠ P) ∧ (C ≠ Q))

-- The first collinearity proof for intersections
theorem intersections_collinear_a (A B C D P Q K L M : Point)
    (trapezoid1 : Trapezoid A B C D) (trapezoid2 : Trapezoid A P Q D)
    (distinct_lengths : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (A ≠ P) ∧ (P ≠ Q) ∧ (Q ≠ D) ∧ (B ≠ P) ∧ (C ≠ Q))
    (K_def : K = intersection [A, B] [C, D])
    (L_def : L = intersection [A, P] [D, Q])
    (M_def : M = intersection [B, P] [C, Q]) :
    collinear [K, L, M] := sorry

-- The second collinearity proof for intersections
theorem intersections_collinear_b (A B C D P Q K L M : Point)
    (trapezoid1 : Trapezoid A B C D) (trapezoid2 : Trapezoid A P Q D)
    (distinct_lengths : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (A ≠ P) ∧ (P ≠ Q) ∧ (Q ≠ D) ∧ (B ≠ P) ∧ (C ≠ Q))
    (K_def : K = intersection [A, B] [C, D])
    (L_def : L = intersection [A, Q] [D, P])
    (M_def : M = intersection [B, Q] [C, P]) :
    collinear [K, L, M] := sorry
    

end intersections_collinear_a_intersections_collinear_b_l581_581932


namespace points_on_same_side_of_line_l581_581716

theorem points_on_same_side_of_line (a : ℝ) :
  (-a - 7) * (24 - a) > 0 ↔ a ∈ Set.Ioo (-∞ : ℝ) (-7) ∪ Set.Ioo (24 : ℝ) (+∞) :=
sorry

end points_on_same_side_of_line_l581_581716


namespace sphere_radius_is_five_l581_581960

theorem sphere_radius_is_five
    (π : ℝ)
    (r r_cylinder h : ℝ)
    (A_sphere A_cylinder : ℝ)
    (h1 : A_sphere = 4 * π * r ^ 2)
    (h2 : A_cylinder = 2 * π * r_cylinder * h)
    (h3 : h = 10)
    (h4 : r_cylinder = 5)
    (h5 : A_sphere = A_cylinder) :
    r = 5 :=
by
  sorry

end sphere_radius_is_five_l581_581960


namespace number_times_half_squared_is_eight_l581_581962

noncomputable def num : ℝ := 32

theorem number_times_half_squared_is_eight :
  (num * (1 / 2) ^ 2 = 2 ^ 3) :=
by
  sorry

end number_times_half_squared_is_eight_l581_581962


namespace area_CDGHF_pentagon_l581_581235

noncomputable def side_length := 6
noncomputable def midpoint (a b : ℝ) : ℝ := (a + b) / 2

namespace Geometry

-- Assume the coordinates of points based on given information
noncomputable def A := (0, 6)
noncomputable def B := (6, 6)
noncomputable def E := (0, 0)
noncomputable def F := (6, 0)
noncomputable def C := (6, -6)
noncomputable def D := (0, -6)
noncomputable def G := (0, midpoint (-6) (0))  -- Midpoint of DE

-- Intersection of line BG with EF
noncomputable def H := (2, 0)

def area_pentagon : ℝ := (6*6) - (1/2 * 3 * 2)

theorem area_CDGHF_pentagon : area_pentagon = 33 :=
by
    sorry

end Geometry

end area_CDGHF_pentagon_l581_581235


namespace total_pens_left_l581_581996

def initial_blue_pens := 9
def removed_blue_pens := 4
def initial_black_pens := 21
def removed_black_pens := 7
def initial_red_pens := 6

def remaining_blue_pens := initial_blue_pens - removed_blue_pens
def remaining_black_pens := initial_black_pens - removed_black_pens
def remaining_red_pens := initial_red_pens

def total_remaining_pens := remaining_blue_pens + remaining_black_pens + remaining_red_pens

theorem total_pens_left : total_remaining_pens = 25 :=
by
  -- Proof will be provided here
  sorry

end total_pens_left_l581_581996


namespace inequality_proof_l581_581425

theorem inequality_proof (a b : ℝ) (c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : c < 0) :
  (c / a > c / b) ∧ (a^c < b^c) ∧ (Real.log (a - c) / Real.log b > Real.log (b - c) / Real.log a) := 
sorry

end inequality_proof_l581_581425


namespace proof_problem_l581_581344

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1 / 2)
noncomputable def f (x : ℝ) : ℝ := (vector_a x + vector_b x) • vector_a x - 2

-- Defining the conditions
def side_a : ℝ := 2 * Real.sqrt 3
def side_c : ℝ := 4
def angle_A (A : ℝ) : Prop :=
  0 < A ∧ A < Real.pi / 2 ∧ f A = 1

open Real

theorem proof_problem (A : ℝ) (b : ℝ) (S : ℝ) :
  (2 * A - pi / 6 = pi / 2) ∧
  (side_a ^ 2 = b ^ 2 + side_c ^ 2 - 2 * b * side_c * cos A) ∧
  (S = 1 / 2 * b * side_c * sin (pi / 3)) →
  f.period = pi ∧
  A = pi / 3 ∧
  b = 2 ∧
  S = 2 * sqrt 3 :=
sorry

end proof_problem_l581_581344


namespace count_perfect_squares_and_cubes_l581_581744

theorem count_perfect_squares_and_cubes (n : ℕ) (h : n = 1000) : 
  let squares := {x : ℕ | (x ^ 2) < n}
  let cubes := {x : ℕ | (x ^ 3) < n}
  let sixth_powers := {x : ℕ | (x ^ 6) < n}
  (squares.card + cubes.card - sixth_powers.card) = 39 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581744


namespace harmony_numbers_with_first_digit_two_l581_581521

def is_harmony_number (n : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  (d1 + d2 + d3 + d4 = 6)

def first_digit_is_two (n : ℕ) : Prop :=
  n / 1000 = 2

theorem harmony_numbers_with_first_digit_two :
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ is_harmony_number n ∧ first_digit_is_two n}.card = 15 :=
sorry

end harmony_numbers_with_first_digit_two_l581_581521


namespace sector_central_angle_l581_581322

noncomputable def sector_radius (r l : ℝ) : Prop :=
2 * r + l = 10

noncomputable def sector_area (r l : ℝ) : Prop :=
(1 / 2) * l * r = 4

noncomputable def central_angle (α r l : ℝ) : Prop :=
α = l / r

theorem sector_central_angle (r l α : ℝ) 
  (h1 : sector_radius r l) 
  (h2 : sector_area r l) 
  (h3 : central_angle α r l) : 
  α = 1 / 2 := 
by
  sorry

end sector_central_angle_l581_581322


namespace marbles_leftover_l581_581452

theorem marbles_leftover (r p j : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) (hj : j % 8 = 2) : (r + p + j) % 8 = 6 := 
sorry

end marbles_leftover_l581_581452


namespace problem_l581_581628

def op (x y : ℝ) : ℝ := x^2 + y^3

theorem problem (k : ℝ) : op k (op k k) = k^2 + k^6 + 6*k^7 + k^9 :=
by
  sorry

end problem_l581_581628


namespace water_height_correct_l581_581498

theorem water_height_correct :
  ∀ (r h : ℕ) (p : ℕ),
  r = 16 →
  h = 120 →
  p = 30 →
  let a := 60 in
  let b := 6 in
  a + b = 66 →
  ∃ x, (x = a * (b : ℝ)^(1/3)) :=
begin
  sorry
end

end water_height_correct_l581_581498


namespace calculate_remainder_product_l581_581243

theorem calculate_remainder_product :
  let n := 2_345_678
  let d := 128
  let remainder := n % d
  remainder * 3 = 33 :=
by
  sorry

end calculate_remainder_product_l581_581243


namespace numPerfectSquaresOrCubesLessThan1000_l581_581773

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def isSixthPower (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

theorem numPerfectSquaresOrCubesLessThan1000 : 
  ∃ n : ℕ, n = 38 ∧ ∀ k < 1000, (isPerfectSquare k ∨ isPerfectCube k → k) ↔ n :=
by
  sorry

end numPerfectSquaresOrCubesLessThan1000_l581_581773


namespace jordan_birth_year_l581_581127

theorem jordan_birth_year 
  (first_amc8 : ℕ)
  (annual : ℕ → ℕ)
  (jordan_age_at_tenth_amc8 : ℕ)
  (year_of_first_amc8 := 1985)
  (annual := λ n, 1985 + n - 1)
  (age_at_tenth := 14)
  (year_of_tenth_amc8 := annual 10) : 
  (year_of_tenth_amc8 - jordan_age_at_tenth_amc8 = 1980) :=
by
  sorry

end jordan_birth_year_l581_581127


namespace proof_problem_l581_581359

variables {z y x w : ℝ}
-- Conditions
def condition1 := 0.45 * z = 1.20 * y
def condition2 := y = 0.75 * (x - 0.5 * w)
def condition3 := z = 1.20 * w

-- Statement we need to prove
noncomputable def percent_proof :=
  (0.45 * w ^ 2 - 0.40 * w) / (1.1 * w) * 100 = 4.55

theorem proof_problem (z y x w : ℝ)
  (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  percent_proof :=
begin
  sorry
end

end proof_problem_l581_581359


namespace simple_interest_rate_l581_581237

-- Define variables
def principal := 850
def amount := 950
def time := 5
def simple_interest := amount - principal

-- Define the rate of interest that needs to be proven
def rate := (simple_interest * 100) / (principal * time)

-- The theorem statement to be proven
theorem simple_interest_rate :
  rate = 2.35 := 
by
  -- Step through calculations (skipped in this example)
  sorry

end simple_interest_rate_l581_581237


namespace odd_prime_condition_l581_581274

/-- Given a positive integer n, if for every positive divisor d of n,
    d + 1 divides n + 1, then n must be an odd prime -/
theorem odd_prime_condition (n : ℕ) (h : ∀ d : ℕ, d ∣ n → (d + 1) ∣ (n + 1)) : 
  nat.prime n ∧ n % 2 = 1 :=
by 
  sorry

end odd_prime_condition_l581_581274


namespace quadruple_solution_l581_581630

theorem quadruple_solution (x y z w : ℝ) (h1: x + y + z + w = 0) (h2: x^7 + y^7 + z^7 + w^7 = 0) :
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨ (x = -y ∧ z = -w) ∨ (x = -z ∧ y = -w) ∨ (x = -w ∧ y = -z) :=
by
  sorry

end quadruple_solution_l581_581630


namespace evaluate_expression_l581_581268

theorem evaluate_expression : (3 / (1 - (2 / 5))) = 5 := by
  sorry

end evaluate_expression_l581_581268


namespace total_pets_l581_581606

theorem total_pets (a_pets : ℕ) (a_cats_fraction : ℕ → ℕ) (l_cats_fraction : ℕ → ℕ) (l_dogs_additional : ℕ → ℕ) 
    (h_a_pets : a_pets = 12)
    (h_a_cats_fraction : a_cats_fraction a_pets = 2/3 * a_pets)
    (h_l_cats_fraction : l_cats_fraction (a_cats_fraction a_pets) = 1/2 * a_cats_fraction a_pets)
    (h_l_dogs_additional : l_dogs_additional 4 = 4 + 7) :
    a_pets + (l_cats_fraction (a_cats_fraction a_pets) + l_dogs_additional 4) = 27 :=
begin
    sorry
end

end total_pets_l581_581606


namespace integer_solution_exists_l581_581273

theorem integer_solution_exists : ∃ n : ℤ, (⌊(n^2 : ℚ) / 3⌋ - ⌊(n : ℚ) / 2⌋^2 = 3) ∧ n = 6 := by
  sorry

end integer_solution_exists_l581_581273


namespace condition_for_all_real_solutions_l581_581466

theorem condition_for_all_real_solutions (c : ℝ) :
  (∀ x : ℝ, x^2 + x + c > 0) ↔ c > 1 / 4 :=
sorry

end condition_for_all_real_solutions_l581_581466


namespace Laplace_original_1_Laplace_original_2_Laplace_original_3_l581_581558

noncomputable def inverse_laplace_transform_1 (p : ℂ) : ℂ :=
  e ^ p * (cos (2 * p) + 1/2 * sin (2 * p))

noncomputable def inverse_laplace_transform_2 (p : ℂ) : ℂ :=
  1 / 12 * (e ^ (2 * p) - e ^ (-p) * cos (real.sqrt 3 * p) - real.sqrt 3 * e ^ (-p) * sin (real.sqrt 3 * p))

noncomputable def inverse_laplace_transform_3 (p : ℂ) : ℂ :=
  -1 / 6 + e ^ p - 3 / 2 * e ^ (2 * p) + 2 / 3 * e ^ (3 * p)

theorem Laplace_original_1 (p : ℂ) : 
  ∀ t, Mathlib.LaplaceTransform.inverseLaplace (λ p, p / (p^2 - 2*p + 5)) t = inverse_laplace_transform_1 t := 
    sorry
    
theorem Laplace_original_2 (p : ℂ) : 
  ∀ t, Mathlib.LaplaceTransform.inverseLaplace (λ p, 1 / (p^3 - 8)) t = inverse_laplace_transform_2 t := 
    sorry

theorem Laplace_original_3 (p : ℂ) : 
  ∀ t, Mathlib.LaplaceTransform.inverseLaplace (λ p, (p + 1) / (p * (p - 1) * (p - 2) * (p - 3))) t = inverse_laplace_transform_3 t := 
    sorry

end Laplace_original_1_Laplace_original_2_Laplace_original_3_l581_581558


namespace total_leaves_on_farm_l581_581221

noncomputable def number_of_branches : ℕ := 10
noncomputable def sub_branches_per_branch : ℕ := 40
noncomputable def leaves_per_sub_branch : ℕ := 60
noncomputable def number_of_trees : ℕ := 4

theorem total_leaves_on_farm :
  number_of_branches * sub_branches_per_branch * leaves_per_sub_branch * number_of_trees = 96000 :=
by
  sorry

end total_leaves_on_farm_l581_581221


namespace fraction_eq_zero_iff_x_eq_2_l581_581363

theorem fraction_eq_zero_iff_x_eq_2 (x : ℝ) : (x - 2) / (x + 2) = 0 ↔ x = 2 := by sorry

end fraction_eq_zero_iff_x_eq_2_l581_581363


namespace ratio_of_seg_lengths_l581_581851

def point := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2) ^ (1 / 2)

theorem ratio_of_seg_lengths (A B C D : point) (a b : ℝ) (hA : A ≠ B ∧ A ≠ C ∧ A ≠ D)
                            (hB : B ≠ C ∧ B ≠ D ∧ C ≠ D)
                            (h_eq : {distance A B, distance A C, distance A D, distance B C, distance B D, distance C D} = {a, b})
                            (h_gt : a > b) :
  (∃ l, l ∈ {Real.sqrt 2, Real.sqrt 3, (1 + Real.sqrt 5) / 2, 1 / Real.sqrt (2 - Real.sqrt 3)} ∧ a = l * b) := 
sorry

end ratio_of_seg_lengths_l581_581851


namespace rate_of_pipe_B_l581_581444

-- Definitions based on conditions
def tank_capacity : ℕ := 850
def pipe_A_rate : ℕ := 40
def pipe_C_rate : ℕ := 20
def cycle_time : ℕ := 3
def full_time : ℕ := 51

-- Prove that the rate of pipe B is 30 liters per minute
theorem rate_of_pipe_B (B : ℕ) : 
  (17 * (B + 20) = 850) → B = 30 := 
by 
  introv h1
  sorry

end rate_of_pipe_B_l581_581444


namespace perfect_squares_and_cubes_count_lt_1000_l581_581790

theorem perfect_squares_and_cubes_count_lt_1000 : 
  let perfect_square := λ n : ℕ, ∃ k : ℕ, n = k^2
  let perfect_cube := λ n : ℕ, ∃ k : ℕ, n = k^3
  let sixth_power := λ n : ℕ, ∃ k : ℕ, n = k^6
  (count (λ n, n < 1000 ∧ (perfect_square n ∨ perfect_cube n)) (Finset.range 1000)) = 38 :=
by {
  sorry
}

end perfect_squares_and_cubes_count_lt_1000_l581_581790


namespace find_friends_in_classes_l581_581456

def friends_in_classes (C : ℕ) (A : ℕ) (total_thread : ℕ) : Prop :=
  A = 0.5 * C ∧
  12 * (C + A) = total_thread

theorem find_friends_in_classes (C A total_thread : ℕ) (h : friends_in_classes C A total_thread) : C = 6 :=
by
  sorry

end find_friends_in_classes_l581_581456


namespace tangents_of_intersection_l581_581884

theorem tangents_of_intersection (O A B C : Point) 
  (h₀ : O ≠ A)
  (h₁ : dist O B = dist O C) -- Assuming B and C lie at the same distance from O
  (h₂ : ∠BOA = 90)  -- Angle BOA is 90 degrees 
  (h₃ : ∠COA = 90)  -- Angle COA is 90 degrees
  (h₄ : B ≠ C) -- Distinct intersection points B and C 
  (h₅ : lies_on_circle B centered_at O and 
        lies_on_circle C centered_at O)  
  (hX: diameter O A)
  : tangent AB to circle_centered_at O ∧ tangent AC to circle_centered_at O := sorry

end tangents_of_intersection_l581_581884


namespace correct_sum_rounded_l581_581872

-- Define the conditions: sum and rounding
def sum_58_46 : ℕ := 58 + 46
def round_to_nearest_hundred (n : ℕ) : ℕ :=
  if n % 100 >= 50 then ((n / 100) + 1) * 100 else (n / 100) * 100

-- state the theorem
theorem correct_sum_rounded :
  round_to_nearest_hundred sum_58_46 = 100 :=
by
  sorry

end correct_sum_rounded_l581_581872


namespace sally_earnings_per_day_l581_581453

variable (S : ℝ) -- the amount of money Sally makes per day in dollars
variable (daily_savings : ℝ)
variable (total_savings : ℝ)

-- Conditions
def sally_savings_per_day (S : ℝ) : ℝ :=
  S / 2

def bob_savings_per_day : ℝ :=
  4 / 2

def total_savings_per_day (S : ℝ) : ℝ :=
  sally_savings_per_day S + bob_savings_per_day

def yearly_savings (d : ℝ) : ℝ :=
  365 * d

-- Problem statement
theorem sally_earnings_per_day :
  total_savings_per_day S = daily_savings →
  yearly_savings daily_savings = total_savings →
  total_savings = 1825 →
  S = 6 := 
by
  sorry

end sally_earnings_per_day_l581_581453


namespace second_smallest_of_set_l581_581149

theorem second_smallest_of_set {s : set ℕ} (h : s = {10, 11, 12, 13}) : 
  ∃ n ∈ s, ∀ m ∈ s, n > m → ∃ k ∈ s, k < n ∧ ∀ l ∈ s, l < k → l = m :=
begin
  sorry
end

end second_smallest_of_set_l581_581149


namespace minimum_scripts_for_similarity_l581_581978

theorem minimum_scripts_for_similarity :
  ∀ (students : ℕ) (questions : ℕ), students = 132009 → questions = 10 →
  ∃ min_scripts : ℕ, min_scripts = 513 ∧
    ∀ scripts : fin students → vector bool questions,
    (∃ s1 s2 : fin students, s1 ≠ s2 ∧
      (∃ similar_count : ℕ, similar_count ≥ 9 ∧
        (¬ ∃ i, i < questions ∧ (scripts s1).nth i ≠ (scripts s2).nth i))) → scripts.card ≥ 513 := sorry

end minimum_scripts_for_similarity_l581_581978


namespace total_cartons_accepted_l581_581402

theorem total_cartons_accepted (total_cartons : ℕ) (customers : ℕ) (damaged_cartons_per_customer : ℕ) (initial_cartons_per_customer accepted_cartons_per_customer total_accepted_cartons : ℕ) :
    total_cartons = 400 →
    customers = 4 →
    damaged_cartons_per_customer = 60 →
    initial_cartons_per_customer = total_cartons / customers →
    accepted_cartons_per_customer = initial_cartons_per_customer - damaged_cartons_per_customer →
    total_accepted_cartons = accepted_cartons_per_customer * customers →
    total_accepted_cartons = 160 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_cartons_accepted_l581_581402


namespace evaluate_expression_simplified_l581_581458

theorem evaluate_expression_simplified (x : ℝ) (h : x = Real.sqrt 2) : 
  (x + 3) ^ 2 + (x + 2) * (x - 2) - x * (x + 6) = 7 := by
  rw [h]
  sorry

end evaluate_expression_simplified_l581_581458


namespace train_cross_time_l581_581214

-- Define the given conditions
def train_length : ℕ := 100
def train_speed_kmph : ℕ := 45
def total_length : ℕ := 275
def seconds_in_hour : ℕ := 3600
def meters_in_km : ℕ := 1000

-- Convert the speed from km/hr to m/s
noncomputable def train_speed_mps : ℚ := (train_speed_kmph * meters_in_km) / seconds_in_hour

-- The time to cross the bridge
noncomputable def time_to_cross (train_length total_length : ℕ) (train_speed_mps : ℚ) : ℚ :=
  total_length / train_speed_mps

-- The statement we want to prove
theorem train_cross_time : time_to_cross train_length total_length train_speed_mps = 30 :=
by
  sorry

end train_cross_time_l581_581214


namespace solve_ineq_l581_581674

theorem solve_ineq (x : ℝ) : (x > 0 ∧ x < 3 ∨ x > 8) → x^3 - 9 * x^2 + 24 * x > 0 :=
by
  sorry

end solve_ineq_l581_581674


namespace exists_martinien_congruent_l581_581670

-- Definition of a_martinien
def is_a_martinien (a : ℕ) (n : ℕ) : Prop :=
  ∃ (k : ℕ) (t : Fin k → ℕ), n = (∑ i, (t i)^a) ∧ 
    n.digits 10 = t.foldr (λ i acc, acc ++ (to_digits i 10)) []

-- Main theorem statement
theorem exists_martinien_congruent (ell : ℕ) (r : ℤ) (a : Fin ell → ℕ) 
  (h : ∀ j, 2 ≤ a j) : 
  ∃ (n : ℤ), (∀ j, is_a_martinien (a j) n) ∧ n % 2021 = r % 2021 := 
by
  sorry

end exists_martinien_congruent_l581_581670


namespace total_students_l581_581979

theorem total_students (N : ℕ) (avg1 : ℕ := 80) (avg2 : ℕ := 90) (avg_total : ℕ := 84)
  (s1 : ℕ := 15) (s2 : ℕ := 10) (total_points : ℕ := 1200 + 900) : N = 25 := 
by
  -- The given conditions
  have h1 : 15 * 80 = 1200, from sorry
  have h2 : 10 * 90 = 900, from sorry
  have h3 : total_points = 84 * N, from sorry
  -- Proving the total number of students
  have h4 : 2100 = 84 * N, by rw [total_points, h1, h2]
  have h5 : N = 2100 / 84, from sorry
  exact (show N = 25, from sorry)


end total_students_l581_581979


namespace complex_trig_form_l581_581552

noncomputable def sin (x : Real) : Real := sorry
noncomputable def cos (x : Real) : Real := sorry

theorem complex_trig_form (sin_36 cos_54 : Real)
  (h1 : sin 36 = sin_36)
  (h2 : cos 54 = cos_54)
  : z = sin 36 + i * cos 54 
    -> z = sqrt 2 * sin 36 * (cos 45 + i * sin 45) :=
sorry

end complex_trig_form_l581_581552


namespace loss_percentage_l581_581187

theorem loss_percentage (x y : ℝ) (h : 30 * x = 40 * y) : ((x - y) / x) * 100 = 25 := by
  have y_eq : y = (3 / 4) * x := by
    sorry
  have loss := x - y
  have loss_eq : loss = (1 / 4) * x := by
    sorry
  calc
    ((x - y) / x) * 100 = ((1 / 4) * x / x) * 100 := by
      rw [loss_eq]
    ... = (1 / 4) * 100 := by
      ring
    ... = 25 := by
      norm_num

end loss_percentage_l581_581187


namespace monic_quadratic_real_coeff_root_l581_581655

theorem monic_quadratic_real_coeff_root (p : ℝ[X]) :
  monic p ∧ degree p = 2 ∧ ∀ x : ℂ, (x - (1 - 3 * complex.I)) * (x - (1 + 3 * complex.I)) = 0 → p.eval x = 0 :=
by
  have h : p = polynomial.C1 * X ^ 2 - 2 * X + 10 := sorry
  exact h

example : ∃ p : ℝ[X], monic p ∧ degree p = 2 ∧ ∀ x : ℂ, (x = 1 - 3 * complex.I → p.eval x = 0) ∧ (x = 1 + 3 * complex.I → p.eval x = 0) := 
by
  use polynomial.X ^ 2 - polynomial.C 2 * polynomial.X + polynomial.C 10
  split
  · exact polynomial.monic_X_pow
  split
  · simp [polynomial.degree_add_eq_of_pos_degree] 
  · sorry

end monic_quadratic_real_coeff_root_l581_581655


namespace find_triple_sum_l581_581358

theorem find_triple_sum (x y z : ℝ) 
  (h1 : y + z = 20 - 4 * x)
  (h2 : x + z = 1 - 4 * y)
  (h3 : x + y = -12 - 4 * z) :
  3 * x + 3 * y + 3 * z = 9 / 2 := 
sorry

end find_triple_sum_l581_581358


namespace distinct_polynomial_roots_l581_581302

theorem distinct_polynomial_roots (n : ℕ) (h : n > 1)
    (a b : Fin n → ℝ) 
    (h_distinct : Function.Injective a ∧ Function.Injective b ∧ Disjoint (Set.range a) (Set.range b)) :
    ¬(∀ x, ∃ i : Fin n, Polynomial.aeval x (Polynomial.X ^ 2 - Polynomial.C (a i) * Polynomial.X + Polynomial.C (b i)) = 0) :=
by
  sorry

end distinct_polynomial_roots_l581_581302


namespace determinant_is_one_minus_k_squared_l581_581249

open Matrix

noncomputable def matrix_example (k a b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, k * sin (a - b), sin a],
    ![k * sin (a - b), 1, k * sin b],
    ![sin a, k * sin b, 1]]

theorem determinant_is_one_minus_k_squared (k a b : ℝ) : 
  det (matrix_example k a b) = 1 - k^2 :=
by 
  -- proof goes here
  sorry

end determinant_is_one_minus_k_squared_l581_581249


namespace karen_grooming_l581_581398

theorem karen_grooming : 
  let (x minutes to groom a border collie) := x
  let rottweiler_time := 20
  let chihuahua_time := 45
  let total_time := 255
  (6 * rottweiler_time + 9 * x + chihuahua_time = total_time) → x = 10 :=
by {
  sorry
}

end karen_grooming_l581_581398


namespace find_side_AC_l581_581033

-- Define the given problem's conditions.
variables {A B C : Type}
variables (triangle : Triangle A B C)
variables (right_triangle : IsRightTriangleAtC triangle)
variables (tan_A : tan_angle triangle A = 4 / 3)
variables (side_AB : side_length triangle A B = 3)

-- State the theorem.
theorem find_side_AC : side_length triangle A C = 5 :=
sorry

end find_side_AC_l581_581033


namespace angle_410_in_first_quadrant_l581_581147

theorem angle_410_in_first_quadrant:
  (θ : ℝ) (Hθ : θ = 410) → (0 ≤ θ % 360 ∧ θ % 360 < 90) :=
by
  intro θ Hθ
  rw Hθ
  simp
  split
  . norm_num
  . norm_num
  sorry

end angle_410_in_first_quadrant_l581_581147


namespace population_of_missing_village_l581_581482

theorem population_of_missing_village 
  (pop1 pop2 pop3 pop4 pop5 pop6 : ℕ) 
  (avg_pop : ℕ) 
  (h1 : pop1 = 803)
  (h2 : pop2 = 900)
  (h3 : pop3 = 1023)
  (h4 : pop4 = 945)
  (h5 : pop5 = 980)
  (h6 : pop6 = 1249)
  (h_avg : avg_pop = 1000) :
  ∃ (pop_missing : ℕ), pop_missing = 1100 := 
by
  -- Placeholder for proof
  sorry

end population_of_missing_village_l581_581482


namespace true_propositions_l581_581472

noncomputable theory
open_locale classical

-- Define each proposition
def prop1 : Prop :=
∀ (p : Point) (l : Line), ¬(p ∈ l) → ∃! m : Line, m ∥ l ∧ p ∈ m

def prop2 : Prop :=
∀ (p : Point) (l : Line), ∃! m : Line, m ⟂ l ∧ p ∈ m

def prop3 : Prop :=
∀ (∠α ∠β : Angle), supplementary (∠α, ∠β) → 
∃ (θα θβ : Line), angle_bisector ∠α θα ∧ angle_bisector ∠β θβ ∧ θα ⟂ θβ

def prop4 : Prop :=
∀ (a b c : Line), (a ∥ b) ∧ (b ⟂ c) → a ⟂ c

-- Mathematical equivalent proof problem
def math_proof_problem : Prop :=
prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4

-- Lean statement
theorem true_propositions :
math_proof_problem :=
by
  sorry

end true_propositions_l581_581472


namespace count_perfect_squares_and_cubes_l581_581741

theorem count_perfect_squares_and_cubes : 
  let num_squares := 31 in
  let num_cubes := 9 in
  let num_sixth_powers := 3 in
  num_squares + num_cubes - num_sixth_powers = 37 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581741


namespace power_series_solution_l581_581280

noncomputable def y_series (x : ℝ) : ℝ :=
  2 + x + x^2 + x^3 / 2 + x^4 / 3

theorem power_series_solution (y : ℝ → ℝ) (y' : ℝ → ℝ) (y'' : ℝ → ℝ) :
  (∀ x, y'' x - y x * Real.exp x = 0) →
  y 0 = 2 →
  y' 0 = 1 →
  (∀ x ∈ set.Icc 0 1, y x = y_series x) :=
by
  intros h_diff_eq h_y0 h_y'0
  -- Placeholder for the actual proof
  sorry

end power_series_solution_l581_581280


namespace fractional_eq_extraneous_k_l581_581812

theorem fractional_eq_extraneous_k (k : ℝ) :
  (∃ x : ℝ, x = 5 ∧ (∀ y, y ≠ 5 → (x - 6) / (x - 5) = k / (5 - x)) ∧ (x - 5 = 0)) → k = 1 :=
by
  intro h
  cases h with x hx
  cases hx with h1 hx2
  cases hx2 with h2 h3
  have : x = 5 := h3
  rw [h1, this]
  rw [h1]
  sorry

end fractional_eq_extraneous_k_l581_581812


namespace quadrilateral_construction_part_a_quadrilateral_construction_part_b_l581_581556

-- Define the necessary entities for the problem conditions: points, quadrilateral, bisector, inscribed circle, etc.

-- Problem (a) statement
theorem quadrilateral_construction_part_a
  (A B C D : Type)
  (AB AC AD BC CD DB : ℝ)
  (h1 : AB ≠ AD) :
  ∃ (A B C D : Type), AC bisects ∠A ∧ length AB = AB ∧ length AC = AC ∧ length AD = AD ∧ length BC = BC ∧ length CD = CD ∧ length DB = DB :=
sorry

-- Problem (b) statement
theorem quadrilateral_construction_part_b
  (A B C D : Type)
  (AB AD : ℝ)
  (angleB angleD : ℝ)
  (h2 : angle ADC ≠ angle ABC) :
  ∃ (A B C D : Type), (inscribed_circle A B C D) ∧ length AB = AB ∧ length AD = AD ∧ angle B = angleB ∧ angle D = angleD :=
sorry

end quadrilateral_construction_part_a_quadrilateral_construction_part_b_l581_581556


namespace heptagon_non_special_after_perturbation_l581_581430

-- Declare the heptagon and the property of being special
structure Heptagon where
  vertices : Fin 7 → Point
  is_special : ∃ P : Point, ∃ i j k l m n: Fin 7, i ≠ k ∧ j ≠ l ∧ ((vertices i).line (vertices k)).inter ((vertices j).line (vertices l)) = P ∧ ((vertices i).line (vertices k)).inter ((vertices m).line (vertices n)) = P

-- Helper function to slightly move a vertex
noncomputable def perturbation (p : Point) : Point := sorry

-- Define the theorem
theorem heptagon_non_special_after_perturbation (h : Heptagon) (i : Fin 7) : ¬(Heptagon.mk (Function.update h.vertices i (perturbation (h.vertices i))) (h.is_special)) := 
sorry

end heptagon_non_special_after_perturbation_l581_581430


namespace count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581794

theorem count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes :
  let num_perfect_squares := 31
  let num_perfect_cubes := 9
  let num_perfect_sixth_powers := 2
  let num_either_squares_or_cubes := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers
  in num_either_squares_or_cubes = 38 :=
by
  sorry

end count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581794


namespace num_ways_distribute_plants_correct_l581_581089

def num_ways_to_distribute_plants : Nat :=
  let basil := 2
  let aloe := 1
  let cactus := 1
  let white_lamps := 2
  let red_lamp := 1
  let blue_lamp := 1
  let plants := basil + aloe + cactus
  let lamps := white_lamps + red_lamp + blue_lamp
  4
  
theorem num_ways_distribute_plants_correct :
  num_ways_to_distribute_plants = 4 :=
by
  sorry -- Proof of the correctness of the distribution

end num_ways_distribute_plants_correct_l581_581089


namespace lambda_one_sufficient_not_necessary_l581_581801

theorem lambda_one_sufficient_not_necessary (λ : ℝ) :
  (3 * x + (λ - 1) * y = 1 ∧ λ * x + (1 - λ) * y = 2) →
  parallel_lines (λ = 1) ↔ (λ = 1 ∨ λ = 3) :=
sorry

end lambda_one_sufficient_not_necessary_l581_581801


namespace coefficient_x3_y9_l581_581166

theorem coefficient_x3_y9 :
  let f := (2 / 3 : ℚ) * x - (3 / 4 : ℚ) * y
  in coeff (expand f 12) (3, 9) = - (315 / 1024 : ℚ) :=
sorry

end coefficient_x3_y9_l581_581166


namespace sqrt_prime_irrational_l581_581412

theorem sqrt_prime_irrational (p : ℕ) (hp : Nat.Prime p) : Irrational (Real.sqrt p) :=
by
  sorry

end sqrt_prime_irrational_l581_581412


namespace find_a_from_parabola_l581_581468

noncomputable def a_value (y0 : ℝ) (a : ℝ) : Prop :=
  let P := (3 / 2 : ℝ, y0)
  let F := (a / 4 : ℝ, 0)
  (y0^2 = a * (3 / 2)) ∧ (∥P.1 - F.1∥ + ∥P.2 - F.2∥ = 2)

theorem find_a_from_parabola (y0 : ℝ) (a > 0) :
  a_value y0 a → a = 2 := by
  sorry

end find_a_from_parabola_l581_581468


namespace emir_needs_more_money_l581_581637

def dictionary_cost : ℕ := 5
def dinosaur_book_cost : ℕ := 11
def cookbook_cost : ℕ := 5
def saved_money : ℕ := 19
def total_cost : ℕ := dictionary_cost + dinosaur_book_cost + cookbook_cost
def additional_money_needed : ℕ := total_cost - saved_money

theorem emir_needs_more_money : additional_money_needed = 2 := by
  sorry

end emir_needs_more_money_l581_581637


namespace sum_first_eight_terms_arithmetic_sequence_l581_581857

theorem sum_first_eight_terms_arithmetic_sequence (a : ℕ → ℕ) (h1 : a 2 = 3) (h2 : a 7 = 13) :
  (∑ i in finset.range 8, a (i + 1)) = 64 :=
sorry

end sum_first_eight_terms_arithmetic_sequence_l581_581857


namespace area_difference_l581_581607

-- Definition of given conditions
def AE : ℕ := 15
def DE : ℕ := 20
def AD : ℕ := Int.sqrt (AE^2 + DE^2)

def square_area (side: ℕ) : ℕ := side * side
def triangle_area (b h: ℕ) : ℕ := b * h / 2
def trapezoid_area (a b h: ℕ) : ℕ := (a + b) * h / 2

-- The lengths used in the calculation
def AG : ℕ := Int.sqrt (AE^2 - ((AE * DE) / AD)^2)  -- Using similar triangles proportionally
def GD : ℕ := AD - AG
def GF : ℕ := AD - ((AE * DE) / AD)
def FG : ℕ := GF

-- Calculate required areas
def area_ADC : ℕ := triangle_area AD AD
def area_CDGF : ℕ := trapezoid_area GF AD GD

-- The final statement to prove
theorem area_difference : area_ADC - area_CDGF = -23.5 := 
by
  sorry

end area_difference_l581_581607


namespace count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581792

theorem count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes :
  let num_perfect_squares := 31
  let num_perfect_cubes := 9
  let num_perfect_sixth_powers := 2
  let num_either_squares_or_cubes := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers
  in num_either_squares_or_cubes = 38 :=
by
  sorry

end count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581792


namespace numPerfectSquaresOrCubesLessThan1000_l581_581767

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def isSixthPower (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

theorem numPerfectSquaresOrCubesLessThan1000 : 
  ∃ n : ℕ, n = 38 ∧ ∀ k < 1000, (isPerfectSquare k ∨ isPerfectCube k → k) ↔ n :=
by
  sorry

end numPerfectSquaresOrCubesLessThan1000_l581_581767


namespace total_surface_area_is_correct_l581_581477

noncomputable def lateral_surface_6pi_4pi : Prop :=
  ∃ (r h : ℝ), (
    (6 * π = 2 * π * r ∨ 4 * π = 2 * π * r) ∧
    (6 * π = h ∨ 4 * π = h) ∧
    (h = 4 * π → 2 * π * r * h + 2 * π * r^2 = 24 * π^2 + 18 * π) ∧ 
    (h = 6 * π → 2 * π * r * h + 2 * π * r^2 = 24 * π^2 + 8 * π)
  )

theorem total_surface_area_is_correct : lateral_surface_6pi_4pi :=
sorry

end total_surface_area_is_correct_l581_581477


namespace non_zero_digits_count_l581_581353

noncomputable def fraction : ℚ := 120 / (2^5 * 5^10)

noncomputable def decimal : ℚ := fraction

noncomputable def result : ℚ := decimal - 1 / 10^6

theorem non_zero_digits_count : 
  (number_of_non_zero_digits (result) = 3) :=
sorry

end non_zero_digits_count_l581_581353


namespace arithmetic_sequence_ratio_l581_581671

theorem arithmetic_sequence_ratio :
  (∀ n : ℕ, S n = (n / 2) * (2 * a₁ + (n - 1) * d) ∧ T n = (n / 2) * (2 * b₁ + (n - 1) * e)) →
  (∀ n : ℕ, S n / T n = 2 * n / (3 * n + 1)) →
  (a₁ + 3 * d = a₄ ∧ a₁ + 5 * d = a₆ ∧ b₁ + 2 * e = b₃ ∧ b₁ + 6 * e = b₇) →
  (2 * (a₁ + 4 * d) = a₄ + a₆ ∧ 2 * (b₁ + 4 * e) = b₃ + b₇) →
  (a₄ + a₆) / (b₃ + b₇) = 9 / 14 :=
by
  sorry

end arithmetic_sequence_ratio_l581_581671


namespace sum_remainders_l581_581153

theorem sum_remainders (a b c : ℕ) (h₁ : a % 30 = 7) (h₂ : b % 30 = 11) (h₃ : c % 30 = 23) : 
  (a + b + c) % 30 = 11 := 
by
  sorry

end sum_remainders_l581_581153


namespace total_accepted_cartons_l581_581399

theorem total_accepted_cartons 
  (total_cartons : ℕ) 
  (customers : ℕ) 
  (damaged_cartons : ℕ)
  (h1 : total_cartons = 400)
  (h2 : customers = 4)
  (h3 : damaged_cartons = 60)
  : total_cartons / customers * (customers - (damaged_cartons / (total_cartons / customers))) = 160 := by
  sorry

end total_accepted_cartons_l581_581399


namespace area_of_region_is_correct_l581_581914

noncomputable def area_enclosed_by_curves : ℝ :=
  ∫ x in 0..1, (Real.exp x - Real.exp (-x))

theorem area_of_region_is_correct :
  area_enclosed_by_curves = Real.exp 1 + Real.exp (-1) - 2 := by
  sorry

end area_of_region_is_correct_l581_581914


namespace taxi_fare_60_miles_l581_581598

theorem taxi_fare_60_miles (m fare_40 fix_charged fare_60 : ℝ)
  (h1 : fare_40 = 95)
  (h2 : fix_charged = 15)
  (h3 : fare_40 = fix_charged + 40 * m)
  (h4 : fare_60 = fix_charged + 60 * m) :
  fare_60 = 135 :=
by {
  have m_calc : m = 2,
  { linarith, },
  have fare_60_calc : fare_60 = 15 + 120,
  { linarith, },
  linarith
}

end taxi_fare_60_miles_l581_581598


namespace rectangle_area_l581_581929

theorem rectangle_area (l : ℝ) (w : ℝ) (h_l : l = 15) (h_ratio : (2 * l + 2 * w) / w = 5) : (l * w) = 150 :=
by
  sorry

end rectangle_area_l581_581929


namespace ellipse_eccentricity_l581_581325

variables {a b c e : ℝ}
variables (a_pos : a > 0) (b_pos : b > 0) (ab_cond : a > b)

/-- Given an ellipse with equation x²/a² + y²/b² = 1 (a > b > 0),
    right focus F(c, 0), and Q is the symmetric point of F with respect to
    the line y = (b/c)x, and Q lies on the ellipse, prove that the eccentricity 
    e = c/a is sqrt(2)/2. -/
theorem ellipse_eccentricity (ellipse_eq : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (focus : F : ℝ × ℝ := (c, 0))
  (sym_Q : Q : ℝ × ℝ := (2*c - F.1, (b/c) * (2*c - F.1)))
  (Q_on_ellipse : sym_Q.1^2 / a^2 + sym_Q.2^2 / b^2 = 1)
  (eccentricity_eq : e = c / a) :
  e = sqrt 2 / 2 :=
sorry

end ellipse_eccentricity_l581_581325


namespace area_of_R_eq_correct_l581_581045

open Int Real

def region_R (x y : ℝ) : Prop := 
  (⌊x^2⌋ = ⌊y⌋) ∧ (⌊y^2⌋ = ⌊x⌋)

noncomputable def area_of_region_R : ℝ :=
  let square_area (a b : ℝ) := (b - a) * (b - a)
  in square_area 0 1 + square_area 1 (sqrt 2)

theorem area_of_R_eq_correct :
  let R := {p : ℝ × ℝ | region_R p.1 p.2}
  let calculated_area := area_of_region_R
  in calculated_area = 4 - 2 * sqrt 2 := by sorry

end area_of_R_eq_correct_l581_581045


namespace problem1_problem2_l581_581304

def f (x : ℕ) : ℝ := sorry

axiom f_add : ∀ (x y : ℕ), f (x + y) = f x * f y
axiom f_one : f 1 = 1 / 2

theorem problem1 (n : ℕ) (h : 0 < n) : f n = (1 / 2) ^ n := sorry

def b (n : ℕ) := (9 - n) * (f (n + 1) / f n)
def S (n : ℕ) := ∑ i in finset.range n, b (i + 1)

theorem problem2 : ∀ (n : ℕ), n = 8 ∨ n = 9 → is_maximizing (S n) := sorry

end problem1_problem2_l581_581304


namespace max_value_of_vectors_l581_581421

def vectors_max_value_problem (a b c : ℝ) : ℝ :=
  ∥a - 3 * b∥^2 + ∥b - 3 * c∥^2 + ∥c - 3 * a∥^2

theorem max_value_of_vectors (a b c : ℝ)
  (h₁ : ∥a∥ = 2)
  (h₂ : ∥b∥ = 1)
  (h₃ : ∥c∥ = 1) : (vectors_max_value_problem a b c) ≤ 42 :=
sorry

end max_value_of_vectors_l581_581421


namespace represent_2015_as_sum_of_palindromes_l581_581580

def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr.to_list in
  s = s.reverse

theorem represent_2015_as_sum_of_palindromes :
  ∃ a b : ℕ, is_palindrome a ∧ is_palindrome b ∧ a + b = 2015 :=
begin
  use [1551, 464],
  split,
  { unfold is_palindrome,
    simp [Nat.repr, List.reverse],
    -- You would normally provide a proof here, but we use sorry
    sorry, 
  },
  split,
  { unfold is_palindrome,
    simp [Nat.repr, List.reverse],
    -- You would normally provide a proof here, but we use sorry
    sorry,
  },
  { -- Verify the sum
    norm_num,
  }
end

end represent_2015_as_sum_of_palindromes_l581_581580


namespace count_perfect_squares_and_cubes_l581_581749

theorem count_perfect_squares_and_cubes (n : ℕ) (h : n = 1000) : 
  let squares := {x : ℕ | (x ^ 2) < n}
  let cubes := {x : ℕ | (x ^ 3) < n}
  let sixth_powers := {x : ℕ | (x ^ 6) < n}
  (squares.card + cubes.card - sixth_powers.card) = 39 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581749


namespace igor_number_proof_l581_581597

noncomputable def igor_number (init_lineup : List ℕ) (igor_num : ℕ) : Prop :=
  let after_first_command := [9, 11, 10, 6, 8, 7] -- Results after first command 
  let after_second_command := [9, 11, 10, 8] -- Results after second command
  let after_third_command := [11, 10, 8] -- Results after third command
  ∃ (idx : ℕ), init_lineup.get? idx = some igor_num ∧
    (∀ new_lineup, 
       (new_lineup = after_first_command ∨ new_lineup = after_second_command ∨ new_lineup = after_third_command) →
       igor_num ∉ new_lineup) ∧ 
    after_third_command.length = 3

theorem igor_number_proof : igor_number [9, 1, 11, 2, 10, 3, 6, 4, 8, 5, 7] 5 :=
  sorry 

end igor_number_proof_l581_581597


namespace product_198_times_2_approx_400_nine_times_number_whose_double_is_56_l581_581928

-- Conditions
def twice_a_number_is_56 (x : ℕ) : Prop :=
  2 * x = 56

-- Statements to prove
theorem product_198_times_2_approx_400 : 
  abs (198 * 2 - 400) < 10 := sorry

theorem nine_times_number_whose_double_is_56 (x : ℕ) 
  (h : twice_a_number_is_56 x) : 
  9 * x = 252 :=
by
  rw [twice_a_number_is_56, nat.mul_left_inj]; sorry

end product_198_times_2_approx_400_nine_times_number_whose_double_is_56_l581_581928


namespace range_of_m_l581_581699

noncomputable def range_m (a b : ℝ) (m : ℝ) : Prop :=
  (3 * a + 4 / b = 1) ∧ a > 0 ∧ b > 0 → (1 / a + 3 * b > m)

theorem range_of_m (m : ℝ) : (∀ a b : ℝ, (range_m a b m)) ↔ m < 27 :=
by
  sorry

end range_of_m_l581_581699


namespace fraction_lost_l581_581245

-- Definitions of the given conditions
def initial_pencils : ℕ := 30
def lost_pencils_initially : ℕ := 6
def current_pencils : ℕ := 16

-- Statement of the proof problem
theorem fraction_lost (initial_pencils lost_pencils_initially current_pencils : ℕ) :
  let remaining_pencils := initial_pencils - lost_pencils_initially
  let lost_remaining_pencils := remaining_pencils - current_pencils
  (lost_remaining_pencils : ℚ) / remaining_pencils = 1 / 3 :=
by
  let remaining_pencils := initial_pencils - lost_pencils_initially
  let lost_remaining_pencils := remaining_pencils - current_pencils
  sorry

end fraction_lost_l581_581245


namespace motorist_routes_birmingham_to_sheffield_l581_581206

-- Definitions for the conditions
def routes_bristol_to_birmingham : ℕ := 6
def routes_sheffield_to_carlisle : ℕ := 2
def total_routes_bristol_to_carlisle : ℕ := 36

-- The proposition that should be proven
theorem motorist_routes_birmingham_to_sheffield : 
  ∃ x : ℕ, routes_bristol_to_birmingham * x * routes_sheffield_to_carlisle = total_routes_bristol_to_carlisle ∧ x = 3 :=
sorry

end motorist_routes_birmingham_to_sheffield_l581_581206


namespace initial_birds_l581_581903

-- Given conditions
def number_birds_initial (x : ℕ) : Prop :=
  ∃ (y : ℕ), y = 4 ∧ (x + y = 6)

-- Proof statement
theorem initial_birds : ∃ x : ℕ, number_birds_initial x ↔ x = 2 :=
by {
  sorry
}

end initial_birds_l581_581903


namespace red_folder_stickers_l581_581881

theorem red_folder_stickers (R : ℕ) :
  let red_folder := 10 * R,
      green_folder := 10 * 2,
      blue_folder := 10 * 1,
      total_stickers := red_folder + green_folder + blue_folder
  in total_stickers = 60 → R = 3 :=
by
  intro h
  simp [red_folder, green_folder, blue_folder, total_stickers] at h
  exact Nat.eq_of_mul_eq_mul_left (by norm_num) h

end red_folder_stickers_l581_581881


namespace alfred_forces_win_l581_581134

-- Define the players and the polynomial conditions
def players := ℕ
def polynomial (n : ℕ) := polynomial ℤ

-- Condition: Alfred wins if the polynomial has an integer root.
def alfred_wins (P : polynomial) := ∃ r : ℤ, polynomial.eval r P = 0

-- Proof statement: Alfred can force a win if and only if n is odd.
theorem alfred_forces_win (n : ℕ) (h : 2 ≤ n) :
  (∃ strategy : ℕ → ℤ, ∀ choices : ℕ → ℤ, alfred_wins (polynomial.mk (λ j, strategy j))) ↔ n % 2 = 1 :=
sorry

end alfred_forces_win_l581_581134


namespace find_f_6_l581_581361

def f : ℕ → ℕ := sorry

lemma f_equality (x : ℕ) : f (x + 1) = x := sorry

theorem find_f_6 : f 6 = 5 :=
by
  -- the proof would go here
  sorry

end find_f_6_l581_581361


namespace correct_statements_l581_581013

-- Define the coordinates of fixed points A and B
def A := (1 : ℝ, 0 : ℝ)
def B := (-1 : ℝ, 0 : ℝ)

-- Define the equation of the curve C
def equation_of_C (x y : ℝ) : Prop :=
  ((x + 1)^2 + y^2) * ((x - 1)^2 + y^2) = (sqrt 2)^2

-- Statements to be proven
def statement1 (x y : ℝ) : Prop := 
  equation_of_C x y ↔ equation_of_C (-x) y

def statement2 (x y : ℝ) : Prop :=
  equation_of_C x y ↔ equation_of_C (-x) (-y)

theorem correct_statements : 
  (∀ x y : ℝ, statement1 x y) ∧ (∀ x y : ℝ, statement2 x y) :=
by
  sorry

end correct_statements_l581_581013


namespace area_trapezoid_def_l581_581585

noncomputable def area_trapezoid (a : ℝ) (h : a ≠ 0) : ℝ :=
  let b := 108 / a
  let DE := a / 2
  let FG := b / 3
  let height := b / 2
  (DE + FG) * height / 2

theorem area_trapezoid_def (a : ℝ) (h : a ≠ 0) :
  area_trapezoid a h = 18 + 18 / a :=
by
  sorry

end area_trapezoid_def_l581_581585


namespace trips_and_weights_l581_581239

theorem trips_and_weights (x : ℕ) (w : ℕ) (trips_Bill Jean_total limit_total: ℕ)
  (h1 : x + (x + 6) = 40)
  (h2 : trips_Bill = x)
  (h3 : Jean_total = x + 6)
  (h4 : w = 7850)
  (h5 : limit_total = 8000)
  : 
  trips_Bill = 17 ∧ 
  Jean_total = 23 ∧ 
  (w : ℝ) / 40 = 196.25 := 
by 
  sorry

end trips_and_weights_l581_581239


namespace extreme_point_of_f_l581_581920

open Real

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * x^2 - log x

theorem extreme_point_of_f : ∃ x₀ > 0, f x₀ = f (sqrt 3 / 3) ∧ 
  (∀ x < sqrt 3 / 3, f x > f (sqrt 3 / 3)) ∧
  (∀ x > sqrt 3 / 3, f x > f (sqrt 3 / 3)) :=
sorry

end extreme_point_of_f_l581_581920


namespace man_speed_approx_l581_581592

theorem man_speed_approx (
  (train_length : ℝ) (train_speed_km_hr : ℝ) (train_pass_time_s : ℝ)
  (h_train_length: train_length = 330)
  (h_train_speed_km_hr: train_speed_km_hr = 60)
  (h_train_pass_time_s: train_pass_time_s = 17.998560115190788)
  ) : 
  ∃ (man_speed_km_hr : ℝ), man_speed_km_hr ≈ 6.0132 :=
by
  sorry

end man_speed_approx_l581_581592


namespace borrowed_amount_l581_581236

noncomputable def calculate_principal (I : ℝ) : ℝ :=
  I / 0.95

theorem borrowed_amount (I : ℝ) (h : I = 11400) : calculate_principal I = 12000 :=
by {
  rw [h, calculate_principal],
  norm_num,
}

end borrowed_amount_l581_581236


namespace largest_integer_solution_l581_581525

theorem largest_integer_solution (x : ℤ) : (8 - 5 * x > 25) -> (x ≤ -4) :=
sorry

noncomputable def largest_integer_value : ℤ :=
begin
  use -4,
  sorry
end

end largest_integer_solution_l581_581525


namespace standard_hyperbola_equation_l581_581306

-- Given conditions
def center_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0

def symmetry_axes_along_coordinates (h : ∀ (x y : ℝ), ℝ) : Prop :=
  ∀ (x y : ℝ), h x y = h (-x) y ∧ h x y = h x (-y)

def passes_through_P (P : ℝ × ℝ) (x y : ℝ) : Prop := P = (1, 3) ∧ (x, y) = P

def eccentricity (e : ℝ) : Prop := e = real.sqrt 2

def hyperbola_equation (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (y^2 / b) - (x^2 / a) = 1

-- Theorem to prove
theorem standard_hyperbola_equation :
  ∃ a b : ℝ, center_origin 0 0 ∧ symmetry_axes_along_coordinates (λ x y, x^2 / a - y^2 / b) ∧
  passes_through_P (1, 3) 1 3 ∧ eccentricity (real.sqrt 2) ∧ hyperbola_equation 8 8 :=
by sorry

end standard_hyperbola_equation_l581_581306


namespace AJHSMETL_19892_reappears_on_line_40_l581_581478
-- Import the entire Mathlib library

-- Define the conditions
def cycleLengthLetters : ℕ := 8
def cycleLengthDigits : ℕ := 5
def lcm_cycles : ℕ := Nat.lcm cycleLengthLetters cycleLengthDigits

-- Problem statement with proof to be filled in later
theorem AJHSMETL_19892_reappears_on_line_40 :
  lcm_cycles = 40 := 
by
  sorry

end AJHSMETL_19892_reappears_on_line_40_l581_581478


namespace complement_of_A_l581_581341

open Set

theorem complement_of_A (U : Set ℕ) (A : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {3, 4, 5}) :
  (U \ A) = {1, 2, 6} :=
by
  sorry

end complement_of_A_l581_581341


namespace line_through_point_parallel_to_l_l581_581123

theorem line_through_point_parallel_to_l {x y : ℝ} (l : ℝ → ℝ → Prop) (A : ℝ × ℝ) :
  l = (λ x y, 2 * x - 4 * y + 7 = 0) → A = (2, 3) →
  (∃ m, (λ x y, 2 * x - 4 * y + m = 0) = (λ x y, x - 2 * y + 4 = 0)) :=
by
  intros hl hA
  use 8
  -- Further proof steps would go here
  sorry

end line_through_point_parallel_to_l_l581_581123


namespace find_d_l581_581486

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + 2 * c^2 + 4 = 2 * d + Real.sqrt (a^2 + b^2 + c - d)) :
  d = 1/2 :=
sorry

end find_d_l581_581486


namespace bob_day3_miles_l581_581240

variable (total_miles day1_percent day2_percent : ℝ)
-- Total miles Bob runs in three days
variable (H_total_miles : total_miles = 70)
-- Bob runs 20 percent of the total miles on day one
variable (H_day1_percent : day1_percent = 0.20)
-- Bob runs 50 percent of the remaining miles on day two
variable (H_day2_percent : day2_percent = 0.50)
-- Miles Bob runs on day one
def day1_miles := total_miles * day1_percent
-- Remaining miles after day one
def remaining_after_day1 := total_miles - day1_miles
-- Miles Bob runs on day two
def day2_miles := remaining_after_day1 * day2_percent
-- Remaining miles after day two
def remaining_after_day2 := remaining_after_day1 - day2_miles

-- Bob's miles on day three
theorem bob_day3_miles : remaining_after_day2 = 28 := by
  sorry

end bob_day3_miles_l581_581240


namespace increase_by_percentage_l581_581192

theorem increase_by_percentage (initial_number : ℝ) (percentage : ℝ) :
  initial_number = 450 → percentage = 0.75 → initial_number + initial_number * percentage = 787.5 :=
by
  assume h₁ h₂
  rw [h₁, h₂]
  sorry

end increase_by_percentage_l581_581192


namespace tangent_line_eq_intersection_range_m_l581_581327

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x - x^2 + a * x

-- Part (I)
theorem tangent_line_eq {a : ℝ} (ha : a = 2) :
  let f_tangent := λ x : ℝ, 2 * x - 1
  ∀ x : ℝ, f x a = 2 * x - x^2 + a * x →
    (f 1 2 = 1) ∧ (derivative (λ x, f x 2) 1 = 2) →
    (∀ t : ℝ, f_tangent t = 2 * t - 1) := sorry

-- Part (II)
theorem intersection_range_m (m : ℝ) (a : ℝ) (ha : a = 2) :
  (∀ x : ℝ, x ∈ set.Icc (1 / Real.exp 1) (Real.exp 1)) →
  ∃ (x1 x2 : ℝ), x1 ∈ set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ 
    x2 ∈ set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ 
    x1 ≠ x2 ∧ 
    (f x1 a = ax - m) ∧ (f x2 a = ax - m) →
  1 < m ∧ m ≤ 2 + 1 / (Real.exp 2) := sorry

end tangent_line_eq_intersection_range_m_l581_581327


namespace impossible_d_values_count_l581_581927

def triangle_rectangle_difference (d : ℕ) : Prop :=
  ∃ (l w : ℕ),
  l = 2 * w ∧
  6 * w > 0 ∧
  (6 * w + 2 * d) - 6 * w = 1236 ∧
  d > 0

theorem impossible_d_values_count : ∀ d : ℕ, d ≠ 618 → ¬triangle_rectangle_difference d :=
by
  sorry

end impossible_d_values_count_l581_581927


namespace number_of_perfect_squares_and_cubes_l581_581764

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end number_of_perfect_squares_and_cubes_l581_581764


namespace probability_blocks_l581_581510

theorem probability_blocks (A B C : Fin 5 → Fin 5 → Fin 5)
    (distinct_colors : Fin 5) :
    let m := 71,
        n := 400 in
    m + n = 471 := by
-- conditions defining the problem
    -- Each person has five blocks of distinct colors
    have h1 : ∀ (i j : Fin 5), A i j ≠ B i j ∧ A i j ≠ C i j ∧ B i j ≠ C i j, from sorry,
    -- Each box is chosen randomly and independently by each person
    have h2 : ∀ (i : Fin 5), ∃ j : Fin 5, A i j = B i j ∧ B i j = C i j, from sorry,
    -- Probability calculations ensure exactly m and n
    have prob : ∃ p : ℚ, p = 71 / 400, from sorry,
    -- Result:
    show 71 + 400 = 471, by rfl

#check probability_blocks

end probability_blocks_l581_581510


namespace sum_of_odd_divisors_252_l581_581535

noncomputable def sum_of_odd_divisors (n : ℕ) : ℕ :=
  finset.sum (finset.filter (λ d, d % 2 = 1) (finset.divisors n)) id

theorem sum_of_odd_divisors_252 : sum_of_odd_divisors 252 = 104 :=
by
  sorry

end sum_of_odd_divisors_252_l581_581535


namespace Iggy_Tuesday_Run_l581_581008

def IggyRunsOnTuesday (total_miles : ℕ) (monday_miles : ℕ) (wednesday_miles : ℕ) (thursday_miles : ℕ) (friday_miles : ℕ) : ℕ :=
  total_miles - (monday_miles + wednesday_miles + thursday_miles + friday_miles)

theorem Iggy_Tuesday_Run :
  let monday_miles := 3
  let wednesday_miles := 6
  let thursday_miles := 8
  let friday_miles := 3
  let total_miles := 240 / 10
  IggyRunsOnTuesday total_miles monday_miles wednesday_miles thursday_miles friday_miles = 4 :=
by
  sorry

end Iggy_Tuesday_Run_l581_581008


namespace correct_choice_is_C_l581_581603

def first_quadrant_positive_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

def right_angle_is_axial (θ : ℝ) : Prop :=
  θ = 90

def obtuse_angle_second_quadrant (θ : ℝ) : Prop :=
  90 < θ ∧ θ < 180

def terminal_side_initial_side_same (θ : ℝ) : Prop :=
  θ = 0 ∨ θ = 360

theorem correct_choice_is_C : obtuse_angle_second_quadrant 120 :=
by
  sorry

end correct_choice_is_C_l581_581603


namespace modulo_500_of_M_l581_581407

def M : ℕ := 561

theorem modulo_500_of_M : M % 500 = 61 := by
  -- Since we are given M = 561
  have h : M = 561 := rfl
  -- Therefore, we need to show 561 % 500 = 61
  rw h
  exact Nat.mod_eq_of_lt (by norm_num : 61 < 500)

end modulo_500_of_M_l581_581407


namespace count_isosceles_triangles_l581_581821

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2).sqrt

def is_isosceles (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (a, b, c) := triangle
  distance a b = distance b c ∨ distance b c = distance c a ∨ distance c a = distance a b

def triangle1 := ((0, 7) : ℝ × ℝ, (3, 7) : ℝ × ℝ, (1, 5) : ℝ × ℝ)
def triangle2 := ((4, 5) : ℝ × ℝ, (4, 7) : ℝ × ℝ, (6, 5) : ℝ × ℝ)
def triangle3 := ((0, 2) : ℝ × ℝ, (3, 3) : ℝ × ℝ, (7, 2) : ℝ × ℝ)
def triangle4 := ((11, 5) : ℝ × ℝ, (10, 7) : ℝ × ℝ, (12, 5) : ℝ × ℝ)

theorem count_isosceles_triangles :
  (List.countp is_isosceles [triangle1, triangle2, triangle3, triangle4] = 1) :=
by
  sorry

end count_isosceles_triangles_l581_581821


namespace intersection_of_circle_and_line_l581_581334

theorem intersection_of_circle_and_line 
  (α : ℝ) 
  (x y : ℝ)
  (h1 : x = Real.cos α) 
  (h2 : y = 1 + Real.sin α) 
  (h3 : y = 1) :
  (x, y) = (1, 1) :=
by
  sorry

end intersection_of_circle_and_line_l581_581334


namespace goose_eggs_count_l581_581877

theorem goose_eggs_count (E : ℕ) 
  (hatch_ratio : ℝ := 1/4)
  (survival_first_month_ratio : ℝ := 4/5)
  (survival_first_year_ratio : ℝ := 3/5)
  (survived_first_year : ℕ := 120) :
  ((survival_first_year_ratio * (survival_first_month_ratio * hatch_ratio * E)) = survived_first_year) → E = 1000 :=
by
  intro h
  sorry

end goose_eggs_count_l581_581877


namespace transistors_in_1995_l581_581875

axiom moores_law (P : ℕ → ℕ → Prop) (t : ℕ) : ∀ n, P t (n+2) = 2 * P t n

def transistors_in_1985 : ℕ := 500000

theorem transistors_in_1995 : ∀ P, ∃ n, moores_law P t ∧ P t = 16000000 := 
by  
  sorry

end transistors_in_1995_l581_581875


namespace total_bd_correct_l581_581675

noncomputable def banker's_discount (A TD : ℝ) : ℝ :=
  (A * TD) / (A - TD)

def bd1 := banker's_discount 2260 360
def bd2 := banker's_discount 3280 520
def bd3 := banker's_discount 4510 710
def bd4 := banker's_discount 6240 980

def total_bankers_discount := bd1 + bd2 + bd3 + bd4

theorem total_bd_correct : abs (total_bankers_discount - 3050.96) < 0.01 :=
by
  sorry

end total_bd_correct_l581_581675


namespace perfect_squares_and_cubes_l581_581781

def is_perfect_square (n: ℕ) : Prop := ∃ k: ℕ, k ^ 2 = n
def is_perfect_cube (n: ℕ) : Prop := ∃ k: ℕ, k ^ 3 = n
def is_perfect_sixth_power (n: ℕ) : Prop := ∃ k: ℕ, k ^ 6 = n

def num_perfect_squares (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_square n).length
def num_perfect_cubes (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_cube n).length
def num_perfect_sixth_powers (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_sixth_power n).length

theorem perfect_squares_and_cubes (bound: ℕ) (hbound: bound = 1000) :
  num_perfect_squares bound + num_perfect_cubes bound - num_perfect_sixth_powers bound = 37 :=
by
  -- conditions:
  have ps := num_perfect_squares bound
  have pc := num_perfect_cubes bound
  have p6 := num_perfect_sixth_powers bound
  sorry

end perfect_squares_and_cubes_l581_581781


namespace determine_weight_two_weighings_l581_581202

theorem determine_weight_two_weighings :
  ∃ (x : ℝ), (∃ n : ℕ, n ≤ 2 ∧ ∀ b1 b2 : list ℝ, perform_weighings b1 b2 x n)
  → (∃ w ∈ {7, 8, 9, 10, 11, 12, 13}, balance_weighted_bag b1 b2 w) :=
sorry

end determine_weight_two_weighings_l581_581202


namespace intersection_of_M_and_N_l581_581360

def M := {x : ℝ | abs x ≤ 2}
def N := {x : ℝ | x^2 - 3 * x = 0}

theorem intersection_of_M_and_N : M ∩ N = {0} :=
by
  sorry

end intersection_of_M_and_N_l581_581360


namespace place_value_difference_l581_581523

theorem place_value_difference :
  let sum_27242 := 2 + 200
  let sum_7232062 := 20 + 2000000
  2000020 - 202 = 1999818 := by
  have h1 : sum_27242 = 202 := by rfl
  have h2 : sum_7232062 = 2000020 := by rfl
  calc
    2000020 - 202 = 1999818 : by norm_num
#align place_value_difference

end place_value_difference_l581_581523


namespace total_bill_cost_l581_581102

-- Definitions of costs and conditions
def curtis_meal_cost : ℝ := 16.00
def rob_meal_cost : ℝ := 18.00
def total_cost_before_discount : ℝ := curtis_meal_cost + rob_meal_cost
def discount_rate : ℝ := 0.5
def time_of_meal : ℝ := 3.0

-- Condition for discount applicability
def discount_applicable : Prop := 2.0 ≤ time_of_meal ∧ time_of_meal ≤ 4.0

-- Total cost with discount applied
def cost_with_discount (total_cost : ℝ) (rate : ℝ) : ℝ := total_cost * rate

-- Theorem statement we need to prove
theorem total_bill_cost :
  discount_applicable →
  cost_with_discount total_cost_before_discount discount_rate = 17.00 :=
by
  sorry

end total_bill_cost_l581_581102


namespace locus_of_M_is_circle_l581_581180

variable {A B A₁ B₁ M : Point}
variable {circle : Circle}

-- Assuming points A and B are fixed on the circle
axiom A_on_circle : A ∈ circle
axiom B_on_circle : B ∈ circle

-- Points A₁ and B₁ move along the circle with constant arc length
axiom A₁_on_circle : A₁ ∈ circle
axiom B₁_on_circle : B₁ ∈ circle
axiom constant_arc_length : arc_length circle A₁ B₁ = 2 * varphi

-- M is the intersection point of lines AA₁ and BB₁
axiom intersection_AA₁_BB₁ : intersects (line_through A A₁) (line_through B B₁) M

-- Conclusion: the locus of M is a circle containing points A and B
theorem locus_of_M_is_circle (v : Real) : 
  (∃ circle₂ : Circle, A ∈ circle₂ ∧ B ∈ circle₂ ∧ M ∈ circle₂) :=
sorry

end locus_of_M_is_circle_l581_581180


namespace sequence_arithmetic_sum_first_n_terms_l581_581052

-- Question 1 statement in Lean 4
theorem sequence_arithmetic (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h₁ : ∀ n, 2 * S n = n * a n)
  (h₂ : a 2 = 1) : 
  (∀ n, a n = n - 1) ∧ ∀ n, (2 * S n = n * (n - 1)) :=
begin
  sorry
end

-- Question 2 statement for condition (a) in Lean 4
theorem sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h₁ : ∀ n, 2 * S n = n * a n)
  (h₂ : a 2 = 1)
  (h₃ : ∀ n, a n = n - 1)
  (h₄ : ∀ n, b n = a n * 2 ^ n) :
  ∀ n, T n = (n-2) * 2^(n+1) + 4 :=
begin
  sorry
end

end sequence_arithmetic_sum_first_n_terms_l581_581052


namespace div_expr_l581_581946

namespace Proof

theorem div_expr (x : ℝ) (h : x = 3.242 * 10) : x / 100 = 0.3242 := by
  sorry

end Proof

end div_expr_l581_581946


namespace determine_integers_l581_581261

theorem determine_integers (m n k : ℕ)
  (h1 : m ≥ 2)
  (h2 : n ≥ 2)
  (h3 : k ≥ 3)
  (h4 : (∃ (d : Fin k → ℕ), m.divisors = multiset.map d.toFun (finset.range k)) ∧
        (∃ (d' : Fin k → ℕ), n.divisors = multiset.map d'.toFun (finset.range k)))
  (h5 : ∀ i : Fin (k - 2), (multiset.nthLe (m.divisors.sorted_lt) (i + 1).val (by linarith)) + 1 =
                          (multiset.nthLe (n.divisors.sorted_lt) (i + 1).val (by linarith))) :
  (m = 4 ∧ n = 9 ∧ k = 3) ∨ (m = 8 ∧ n = 15 ∧ k = 4) := sorry

end determine_integers_l581_581261


namespace functions_intersect_at_one_one_l581_581923

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log2 x

noncomputable def g (x : ℝ) : ℝ := 2 ^ (-x + 1)

theorem functions_intersect_at_one_one :
  f 1 = g 1 :=
by
  sorry

end functions_intersect_at_one_one_l581_581923


namespace no_partition_square_isosceles_10deg_l581_581886

theorem no_partition_square_isosceles_10deg :
  ¬ ∃ (P : ℝ → ℝ → Prop), 
    (∀ x y, P x y → ((x = y) ∨ ((10 * x + 10 * y + 160 * (180 - x - y)) = 9 * 10))) ∧
    (∀ x y, P x 90 → P x y) ∧
    (P 90 90 → False) :=
by
  sorry

end no_partition_square_isosceles_10deg_l581_581886


namespace find_AX_l581_581271

noncomputable theory

-- Define the problem statement
theorem find_AX :
  ∀ (A B C X : Type)
    [triangle ABC]
    (CX_bisects_ACB: bisects CX ∠ACB)
    (AB : length A B = 45)
    (BC : length B C = 55)
    (AC : length A C = 33)
    (angle_ACB : angle A C B = 60),
  length A X = 16.875 := 
sorry

end find_AX_l581_581271


namespace part1_monotonic_intervals_part2_three_zeros_l581_581686

open Function

noncomputable def f (a : ℝ) (x : ℝ) := x^2 + 2 * x - 4 + a / x

theorem part1_monotonic_intervals :
  (∀ x ∈ (Set.Ioo (-∞ : ℝ) 0), deriv (f 4) x < 0) ∧
  (∀ x ∈ (Set.Ioo (0 : ℝ) 1), deriv (f 4) x < 0) ∧
  (∀ x ∈ (Set.Ioo (1 : ℝ) ∞), deriv (f 4) x > 0) :=
sorry

theorem part2_three_zeros (a : ℝ) :
  (∀ b : ℝ, ∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0) ↔ (a ∈ Set.Ioo (-8 : ℝ) 0 ∪ Set.Ioo 0 (40 / 27)) :=
sorry

end part1_monotonic_intervals_part2_three_zeros_l581_581686


namespace equivalent_proof_l581_581435

noncomputable def find_m_n : ℕ :=
  let boxes := [4, 5, 6] in
  let totalWays := Nat.choose 15 4 * Nat.choose (15 - 4) 5 * Nat.choose (15 - 4 - 5) 6 in
  let mathIn4 := Nat.choose 11 5 * Nat.choose (11 - 5) 6 in
  let mathIn5 := Nat.choose 11 1 * Nat.choose 10 4 * Nat.choose 6 6 in
  let mathIn6 := Nat.choose 11 2 * Nat.choose 9 4 * Nat.choose 5 5 in
  let totalMathWays := mathIn4 + mathIn5 + mathIn6 in
  let prob := totalMathWays * totalWays in
  let gcd_val := Nat.gcd (totalMathWays * totalWays) totalWays in
  let m := (totalMathWays * totalWays) / gcd_val in
  let n := totalWays / gcd_val in
  m + n

theorem equivalent_proof : 
  ∃ m n : ℕ, 
    (Nat.gcd m n = 1) ∧
    (m / Nat.gcd m n + n / Nat.gcd m n = find_m_n) :=
by
  sorry

end equivalent_proof_l581_581435


namespace complex_number_propositions_correct_count_l581_581602

theorem complex_number_propositions_correct_count :
  ∃ (x y : ℂ), (¬ ((x^2 + y^2 = 0) → (x = 0 ∧ y = 0))) ∧ 
     (¬ (∀ a b : ℝ, a > b → (a + complex.I > b + complex.I))) ∧
     (¬ (∀ x y : ℝ, (x + y * complex.I = 1 + complex.I) → (x = 1 ∧ y = 1))) → 0 = 0 :=
by
  sorry

end complex_number_propositions_correct_count_l581_581602


namespace survey_population_l581_581608

-- Definitions based on conditions
def number_of_packages := 10
def dozens_per_package := 10
def sets_per_dozen := 12

-- Derived from conditions
def total_sets := number_of_packages * dozens_per_package * sets_per_dozen

-- Populations for the proof
def population_quality : ℕ := total_sets
def population_satisfaction : ℕ := total_sets

-- Proof statement
theorem survey_population:
  (population_quality = 1200) ∧ (population_satisfaction = 1200) := by
  sorry

end survey_population_l581_581608


namespace find_Q_l581_581415

theorem find_Q : 
  let P := 95 in 
  let Q := (Σ k in finset.range (P - 3) \ 2, log 128 (2^k)) in
  Q = 329 := 
by
  sorry

end find_Q_l581_581415


namespace volume_of_rotated_square_area_4_l581_581589

noncomputable def volume_of_rotated_square (a : ℝ) := 
  let side := real.sqrt a;
  let r := real.sqrt 2;
  let h := real.sqrt 2;
  2 * (1 / 3 * real.pi * r ^ 2 * h)

theorem volume_of_rotated_square_area_4 :
  volume_of_rotated_square 4 = 4 * real.sqrt 2 * real.pi / 3 := 
by
  -- Proof omitted
  sorry

end volume_of_rotated_square_area_4_l581_581589


namespace area_of_triangle_ABC_l581_581909

theorem area_of_triangle_ABC
  (A B C P: Point)
  (h1 : RightTriangle A B C)
  (h2 : OnHypotenuse P A C)
  (h3 : ∠ A B P = 45)
  (h4 : dist A P = 3)
  (h5 : dist P C = 6) :

  area A B C = 81 / 5 :=
sorry

end area_of_triangle_ABC_l581_581909


namespace sum_odd_divisors_252_l581_581538

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : list ℕ := (list.range (n + 1)).filter (λ d, n % d = 0)

def odd_divisors_sum (n : ℕ) : ℕ :=
(list.filter is_odd (divisors n)).sum

theorem sum_odd_divisors_252 : odd_divisors_sum 252 = 104 :=
by
  -- Proof goes here
  sorry

end sum_odd_divisors_252_l581_581538


namespace exists_independent_subsets_l581_581059

variables {V : Type*} [Fintype V] (G : Type*) [Graph G] [Bipartite G]

-- Given the conditions of the problem
variables (A B : Finset V) (a b c d : ℕ)
(hA : Fintype.card A = a)
(hB : Fintype.card B = b)
(hc : c ≤ a)
(hd : d ≤ b)
(h_edges : Graph.edges G ≤ (a - c) * (b - d) / d)

-- Formulate the theorem
theorem exists_independent_subsets (G : Type*) [Graph G] [Bipartite G] : 
  ∃ (C : Finset V) (D : Finset V), 
    C ⊆ A ∧ D ⊆ B ∧ Fintype.card C = c ∧ Fintype.card D = d ∧ 
    (∀ v1 ∈ C, ∀ v2 ∈ D, ¬Graph.adj G v1 v2) :=
begin
  sorry
end

end exists_independent_subsets_l581_581059


namespace angle_x_value_l581_581834

theorem angle_x_value :
  ∀ (Q R S T P : Type) 
    (lies_on : Q → P → Prop)
    (angle_QST angle_TSP angle_RQS x : ℝ),
  (lies_on Q R) → 
  (lies_on S (QT)) → 
  (angle_QST = 180) → 
  (angle_TSP = 50) → 
  (angle_RQS = 150) → 
  (angle_RQS = angle_QST - angle_TSP + x) → 
  x = 20 :=
by
  intros,
  sorry

end angle_x_value_l581_581834


namespace geom_seq_prop_l581_581290

variable (b : ℕ → ℝ) (r : ℝ) (s t : ℕ)
variable (h : s ≠ t)
variable (h1 : s > 0) (h2 : t > 0)
variable (h3 : b 1 = 1)
variable (h4 : ∀ n, b (n + 1) = b n * r)

theorem geom_seq_prop : s ≠ t → s > 0 → t > 0 → b 1 = 1 → (∀ n, b (n + 1) = b n * r) → (b t)^(s - 1) / (b s)^(t - 1) = 1 :=
by
  intros h h1 h2 h3 h4
  sorry

end geom_seq_prop_l581_581290


namespace divisor_remainder_7_dividing_59_l581_581259

theorem divisor_remainder_7_dividing_59 :
  {d : ℕ // d > 7 ∧ d ∣ 52}.card = 3 :=
sorry

end divisor_remainder_7_dividing_59_l581_581259


namespace find_a_l581_581340

noncomputable def A : Set ℝ := {1, 2, 3}
noncomputable def B (a : ℝ) : Set ℝ := { x | x^2 - (a + 1) * x + a = 0 }

theorem find_a (a : ℝ) (h : A ∪ B a = A) : a = 1 ∨ a = 2 ∨ a = 3 :=
by
  sorry

end find_a_l581_581340


namespace largest_consecutive_multiple_l581_581143

theorem largest_consecutive_multiple (n : ℕ) (h : 3 * n + 3 * (n + 1) + 3 * (n + 2) = 117) : 3 * (n + 2) = 42 :=
sorry

end largest_consecutive_multiple_l581_581143


namespace value_of_x2_plus_9y2_l581_581356

theorem value_of_x2_plus_9y2 (x y : ℝ) (h1 : x + 3 * y = 9) (h2 : x * y = -15) : x^2 + 9 * y^2 = 171 :=
sorry

end value_of_x2_plus_9y2_l581_581356


namespace transformation_matrix_l581_581526

theorem transformation_matrix (θ : ℝ) (a b : ℝ) (hθ : θ = Real.pi / 2) (ha : a = 2) (hb : b = 1) :
  let M := a * Matrix.fromBlocks (Matrix.scalar 0 (Real.sin (b * θ))) (Matrix.scalar 0 (-Real.cos (b * θ)))
                (Matrix.scalar 0 (Real.cos (b * θ))) (Matrix.scalar 0 (Real.sin (b * θ))) in
  M = Matrix.fromBlocks (Matrix.scalar 0 0) (Matrix.scalar 0 (-2 : ℝ)) (Matrix.scalar 0 (2 : ℝ)) (Matrix.scalar 0 0) :=
begin
  -- No proof necessary, 'sorry' is used as a placeholder.
  sorry
end

end transformation_matrix_l581_581526


namespace problem_propositions_l581_581230

/-- The propositions given in the problem. We need to show that only the fourth and the fifth are true. -/
theorem problem_propositions (p q : Prop) (m : ℝ) : 
  (∀ (F₁ F₂ : ℝ × ℝ) (c : ℝ), c > dist F₁ F₂ → ( ∀ (P : ℝ × ℝ), dist P F₁ + dist P F₂ = c ↔ False )) ∧ 
  (∀ (e₁ e₂ e₃ a : ℝ × ℝ), (¬ (collinear e₁ e₂ e₃) ∧ ∃! (λ₁ λ₂ λ₃ : ℝ), a = λ₁ • e₁ + λ₂ • e₂ + λ₃ • e₃)) ∧ 
  (∀ (x y : ℝ), (y = real.sqrt x ↔ x = y^2) = False) ∧ 
  (∀ (p q : Prop), (p → q) ∧ ¬(q → p) → ((¬ p → ¬ q) ∧ ¬(¬ q → ¬ p))) ∧ 
  (∀ (x y : ℝ) (m : ℝ), (2 < m ∧ m < 5) ↔ (x^2 / (5 - m) + y^2 / (2 - m) = 1)) := 
begin
  sorry
end

end problem_propositions_l581_581230


namespace altitudes_concurrent_l581_581447

noncomputable def orthocenter_of_triangle (A B C : Point) : Point := 
  sorry

theorem altitudes_concurrent (A B C : Point) : 
  let H := orthocenter_of_triangle A B C in
  (is_altitude A B C H) ∧ (is_altitude B C A H) ∧ (is_altitude C A B H) :=
sorry

axiom is_altitude (A B C H : Point) : Prop :=
  sorry

end altitudes_concurrent_l581_581447


namespace max_sum_of_squares_eq_7_l581_581499

theorem max_sum_of_squares_eq_7 :
  ∃ (x y : ℤ), (x^2 + y^2 = 25 ∧ x + y = 7) ∧
  (∀ x' y' : ℤ, (x'^2 + y'^2 = 25 → x' + y' ≤ 7)) := by
sorry

end max_sum_of_squares_eq_7_l581_581499


namespace rectangle_dimensions_l581_581346

theorem rectangle_dimensions (l w : ℝ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 2880) :
  (l = 86.833 ∧ w = 33.167) ∨ (l = 33.167 ∧ w = 86.833) :=
by
  sorry

end rectangle_dimensions_l581_581346


namespace paperback_to_hardback_ratio_l581_581505

theorem paperback_to_hardback_ratio
    (H_B : ℕ) (P : ℕ) (T : ℕ) (T_H : ℕ)
    (h_paperback : P = 363600)
    (h_total : T = 440000)
    (h_hardback_initial : H_B = 36000)
    (h_total_hardback : T_H = T - P) :
    let H := T_H - H_B in
    P / H = 9 := by
  -- Introduce variables
  let H := T_H - H_B
  -- Compute the ratio
  have h1 : H = T_H - H_B, by sorry
  have h2 : P / H = 9, by sorry
  exact h2

#eval paperback_to_hardback_ratio 36000 363600 440000 (440000 - 363600) (363600) (440000) (36000) (440000 - 363600)

end paperback_to_hardback_ratio_l581_581505


namespace pizza_eaten_after_six_trips_l581_581954

noncomputable def fraction_eaten : ℚ :=
  let first_trip := 1 / 3
  let second_trip := 1 / (3 ^ 2)
  let third_trip := 1 / (3 ^ 3)
  let fourth_trip := 1 / (3 ^ 4)
  let fifth_trip := 1 / (3 ^ 5)
  let sixth_trip := 1 / (3 ^ 6)
  first_trip + second_trip + third_trip + fourth_trip + fifth_trip + sixth_trip

theorem pizza_eaten_after_six_trips : fraction_eaten = 364 / 729 :=
by sorry

end pizza_eaten_after_six_trips_l581_581954


namespace twelfth_term_of_arithmetic_sequence_l581_581542

/-- Condition: a_1 = 1/2 -/
def a1 : ℚ := 1 / 2

/-- Condition: common difference d = 1/3 -/
def d : ℚ := 1 / 3

/-- Prove that the 12th term in the arithmetic sequence is 25/6 given the conditions. -/
theorem twelfth_term_of_arithmetic_sequence : a1 + 11 * d = 25 / 6 := by
  sorry

end twelfth_term_of_arithmetic_sequence_l581_581542


namespace inequality_proof_l581_581709

open Real

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_product : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by {
  sorry
}

end inequality_proof_l581_581709


namespace uphill_integers_div_by_9_count_l581_581620

def is_uphill_integer (n : ℕ) : Prop :=
  let digits := to_digits 10 n in
  ∀ i j, i < j → digits[i] < digits[j]

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def num_uphill_integers_divisible_by_9 : ℕ :=
  Finset.filter (λ n, is_uphill_integer n ∧ is_divisible_by_9 n)
                (Finset.range 10^10)
  .card

theorem uphill_integers_div_by_9_count : num_uphill_integers_divisible_by_9 = 34 := sorry

end uphill_integers_div_by_9_count_l581_581620


namespace divide_figure_into_8_identical_parts_l581_581255

/-- A cube or rectangle can be divided into 8 identical parts. -/
theorem divide_figure_into_8_identical_parts (figure : Type) 
  (is_cube : figure -> Prop) 
  (is_rectangle : figure -> Prop) 
  (cut : figure -> vector figure 8) 
  : (is_cube figure ∨ is_rectangle figure) -> 
    ∃ parts : vector figure 8, ∀ i j, parts[i] = parts[j] :=
by sorry

end divide_figure_into_8_identical_parts_l581_581255


namespace distance_focus_parabola_asymptotes_hyperbola_l581_581118

theorem distance_focus_parabola_asymptotes_hyperbola :
  let focus := (2 : ℝ, 0 : ℝ)
  let asymptote1 := λ x y : ℝ, x - (√3 / 3) * y = 0
  let asymptote2 := λ x y : ℝ, x + (√3 / 3) * y = 0
  let distance := λ (p : ℝ × ℝ) (a b c : ℝ), (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)
  distance focus 1 (-(√3 / 3)) 0 = √3 :=
by
  sorry

end distance_focus_parabola_asymptotes_hyperbola_l581_581118


namespace first_player_win_probability_l581_581518

theorem first_player_win_probability : 
  (λ P, let P_win_in_one_roll := 7 / 12,
            P_not_win_in_one_roll := 5 / 12,
            P_total_win := P_win_in_one_roll / (1 - P_not_win_in_one_roll * P_not_win_in_one_roll) in
        P_total_win) = 12 / 17 :=
by
  sorry

end first_player_win_probability_l581_581518


namespace find_5_tuples_l581_581076

theorem find_5_tuples :
  {t : ℕ × ℕ × ℕ × ℕ × ℕ // 
    let a := t.1, b := t.2.1, c := t.2.2.1, d := t.2.2.2.1, n := t.2.2.2.2 in
    a + b + c + d = 100 ∧   
    a + n = b - n ∧
    c * n = d / n ∧
    d % n = 0
  } =
 [{(24, 26, 25, 25, 1), (12, 20, 4, 64, 4), (0, 18, 1, 81, 9)}] := 
begin
  sorry
end

end find_5_tuples_l581_581076


namespace coordinates_P_wrt_origin_l581_581916

/-- Define a point P with coordinates we are given. -/
def P : ℝ × ℝ := (-1, 2)

/-- State that the coordinates of P with respect to the origin O are (-1, 2). -/
theorem coordinates_P_wrt_origin : P = (-1, 2) :=
by
  -- Proof would go here
  sorry

end coordinates_P_wrt_origin_l581_581916


namespace cos_x_eq_half_has_two_solutions_l581_581799

theorem cos_x_eq_half_has_two_solutions: 
  (setOf (λ x: ℝ, cos (x * real.pi / 180) = 0.5 ∧ 0 ≤ x ∧ x ≤ 360)).card = 2 := 
by sorry

end cos_x_eq_half_has_two_solutions_l581_581799


namespace g_value_18_l581_581860

noncomputable def g : ℕ+ → ℕ+
axiom g_increasing : ∀ n : ℕ+, g (n + 1) > g n
axiom g_multiplicative : ∀ (m n : ℕ+), g (m * n) = g m * g n
axiom g_equal_powers_condition : ∀ (m n : ℕ+), m ≠ n → m ^ n = n ^ m → (g m = n ^ 2 ∨ g n = m ^ 2)

theorem g_value_18 : g 18 = 104976 := sorry

end g_value_18_l581_581860


namespace largest_n_l581_581044

/-- Let K and N > K be fixed positive integers. Let n 
    be a positive integer and let a_1, a_2, ..., a_n 
    be distinct integers such that for integers 
    m_1, m_2, ..., m_n, not all equal to 0, where 
    |m_i| ≤ K, the sum (m_1 * a_1 + m_2 * a_2 + ... + m_n * a_n) 
    is not divisible by N. What is the largest possible 
    value of n? -/
theorem largest_n (K N : ℕ) (hKN : N > K) :
  ∃ n, ∀ (a : Fin n → ℤ), (Function.Injective a) →
    (∀ (m : Fin n → ℤ), (∃ i, m i ≠ 0) → (∀ i, |m i| ≤ K) → ∑ i, m i * a i % N ≠ 0) →
    n = Nat.floor (Real.log N / Real.log (K + 1)) :=
by
  sorry

end largest_n_l581_581044


namespace proof_A_minus_2B_eq_11_l581_581677

theorem proof_A_minus_2B_eq_11 
  (a b : ℤ)
  (hA : ∀ a b, A = 3*b^2 - 2*a^2)
  (hB : ∀ a b, B = ab - 2*b^2 - a^2) 
  (ha : a = 2) 
  (hb : b = -1) : 
  (A - 2*B = 11) :=
by
  sorry

end proof_A_minus_2B_eq_11_l581_581677


namespace range_f_l581_581285

noncomputable def f (x : ℝ) : ℝ := x / (x^2 - x + 1)

theorem range_f : set.Icc (-1/3 : ℝ) (1 : ℝ) = 
  {y : ℝ | ∃ x : ℝ, y = f x} := sorry

end range_f_l581_581285


namespace find_angle_between_intersections_l581_581829

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  ((t * Real.cos (8 * Real.pi / 3)), (-4 + t * Real.sin (8 * Real.pi / 3)))

def polar_curve (ρ : ℝ) : Prop :=
  ρ^2 - 3*ρ - 4 = 0 ∧ ρ ≥ 0

def cartesian_curve (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

def general_eq_line (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y = 4

theorem find_angle_between_intersections : 
  (∀ t, let (x, y) := parametric_line t in general_eq_line x y) →
  polar_curve 4 →
  (x y : ℝ) → cartesian_curve x y →
  ∃ A B : ℝ × ℝ, -- Assume A and B are points of intersection
  let d := 2 in -- center to line distance is 2
  let cos_half_angle := d / 4 in 
  cos_half_angle = 1/2 → 
  let half_angle := Real.arccos cos_half_angle in 
  half_angle = Real.pi / 3 →
  2 * half_angle = 2 * Real.pi / 3 :=
sorry

end find_angle_between_intersections_l581_581829


namespace intersection_coords_perpendicular_line_l581_581697

def line1 (x y : ℝ) := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) := x + y - 2 = 0

theorem intersection_coords : ∃ P : ℝ × ℝ, line1 P.1 P.2 ∧ line2 P.1 P.2 ∧ P = (1, 1) := by
  sorry

theorem perpendicular_line (x y : ℝ) (P : ℝ × ℝ) (hP: P = (1, 1)) : 
  (line2 P.1 P.2) → x - y = 0 := by
  sorry

end intersection_coords_perpendicular_line_l581_581697


namespace alice_and_bob_pies_l581_581600

theorem alice_and_bob_pies (T : ℝ) : (T / 5 = T / 6 + 2) → T = 60 := by
  sorry

end alice_and_bob_pies_l581_581600


namespace f_increasing_domain_h_has_min_value_g_zeros_and_derivative_l581_581331

def f (x : ℝ) := (x - 2) / (x + 2) * Real.exp x
def g (x a : ℝ) := 2 * Real.log x - a * x
def h (x b : ℝ) := (Real.exp x - b * x - b) / (x^2)

theorem f_increasing_domain :
  ∀ x, (x < -2 ∨ x > -2) → f' x ≥ 0 :=
sorry

theorem h_has_min_value (b : ℝ) (hb : 0 ≤ b ∧ b < 1) :
  ∃ t > 0, h t b = (Real.exp t) / (t + 2) ∧ 
    (1 / 2 < h t b) ∧ (h t b ≤ Real.exp 2 / 4) :=
sorry

theorem g_zeros_and_derivative (x1 x2 a : ℝ) (hx1x2 : x1 < x2) (hx1 : g x1 a = 0) (hx2 : g x2 a = 0) :
  (0 < a ∧ a < 2 / Real.exp 1) ∧ g' ((x1 + 2 * x2) / 3) a < 0 :=
sorry

end f_increasing_domain_h_has_min_value_g_zeros_and_derivative_l581_581331


namespace marathon_yards_l581_581205

theorem marathon_yards (mile_in_yards : ℕ)
  (marathon_miles : ℕ) (marathon_extra_yards : ℕ)
  (num_marathons : ℕ)
  (marathon_yards : marathon_miles * mile_in_yards + marathon_extra_yards)
  (total_yards : num_marathons * marathon_yards)
  (miles_covered : ℕ := total_yards / mile_in_yards)
  (yards_covered : ℕ := total_yards % mile_in_yards) :
  (mile_in_yards = 1760) → (marathon_miles = 26) → (marathon_extra_yards = 395) → 
  (num_marathons = 15) → (0 ≤ yards_covered) → (yards_covered < mile_in_yards) → 
  yards_covered = 645 := 
by
  intros
  sorry

end marathon_yards_l581_581205


namespace greatest_possible_length_l581_581092

-- Define the lengths of the ropes
def rope_lengths : List ℕ := [72, 48, 120, 96]

-- Define the gcd function to find the greatest common divisor of a list of numbers
def list_gcd (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- Define the target problem statement
theorem greatest_possible_length 
  (h : list_gcd rope_lengths = 24) : 
  ∀ length ∈ rope_lengths, length % 24 = 0 :=
by
  intros length h_length
  sorry

end greatest_possible_length_l581_581092


namespace track_width_l581_581210

theorem track_width (r : ℝ) (h1 : 4 * π * r - 2 * π * r = 16 * π) (h2 : 2 * r = r + r) : 2 * r - r = 8 :=
by
  sorry

end track_width_l581_581210


namespace fraction_sum_is_one_l581_581409

theorem fraction_sum_is_one
    (a b c d w x y z : ℝ)
    (h1 : 17 * w + b * x + c * y + d * z = 0)
    (h2 : a * w + 29 * x + c * y + d * z = 0)
    (h3 : a * w + b * x + 37 * y + d * z = 0)
    (h4 : a * w + b * x + c * y + 53 * z = 0)
    (a_ne_17 : a ≠ 17)
    (b_ne_29 : b ≠ 29)
    (c_ne_37 : c ≠ 37)
    (wxyz_nonzero : w ≠ 0 ∨ x ≠ 0 ∨ y ≠ 0) :
    (a / (a - 17)) + (b / (b - 29)) + (c / (c - 37)) + (d / (d - 53)) = 1 := 
sorry

end fraction_sum_is_one_l581_581409


namespace calculate_value_l581_581993

theorem calculate_value :
  let A := (6 * 1000) + (36 * 100) in
  let B := 876 - 197 - 197 in
  A - B = 9118 :=
by
  sorry

end calculate_value_l581_581993


namespace arithmetic_sequence_seventh_term_l581_581827

/-- In an arithmetic sequence, the sum of the first three terms is 9 and the third term is 8. 
    Prove that the seventh term is 28. -/
theorem arithmetic_sequence_seventh_term :
  ∃ (a d : ℤ), (a + (a + d) + (a + 2 * d) = 9) ∧ (a + 2 * d = 8) ∧ (a + 6 * d = 28) :=
by
  sorry

end arithmetic_sequence_seventh_term_l581_581827


namespace polynomial_evaluation_l581_581357

theorem polynomial_evaluation (x : ℤ) (h : x = 2) : 3 * x^2 + 5 * x - 2 = 20 := by
  sorry

end polynomial_evaluation_l581_581357


namespace one_third_point_coordinates_l581_581278

-- Definitions for the conditions given in the problem
def A : ℝ × ℝ := (2, 6)
def B : ℝ × ℝ := (8, -2)
def w1 : ℝ := 2 / 3
def w2 : ℝ := 1 / 3

-- The statement of the proof problem in Lean 4
theorem one_third_point_coordinates (x y : ℝ) :
  (x, y) = (w1 * A.1 + w2 * B.1, w1 * A.2 + w2 * B.2) → (x, y) = (4, 10 / 3) :=
by
  sorry

end one_third_point_coordinates_l581_581278


namespace polynomial_divisibility_l581_581645

theorem polynomial_divisibility (n : ℕ) (h : n > 2) : 
    (∀ k : ℕ, n = 3 * k + 1) ↔ ∃ (k : ℕ), n = 3 * k + 1 := 
sorry

end polynomial_divisibility_l581_581645


namespace geom_seq_l581_581814

noncomputable def a_n : ℕ → ℤ
| n := 2 * n - 1

noncomputable def first_order_derivative (a : ℕ → ℤ) : ℕ → ℤ
| n := n * a n - (n - 1) * a (n - 1)

noncomputable def second_order_derivative (a : ℕ → ℤ) : ℕ → ℤ
| n := n * first_order_derivative a n - (n - 1) * first_order_derivative a (n - 1)

def first_four_terms : List ℤ :=
  [first_order_derivative (first_order_derivative a_n) 2 - 1,
   first_order_derivative (first_order_derivative a_n) 3 - 1,
   first_order_derivative (first_order_derivative a_n) 4 - 1,
   first_order_derivative (first_order_derivative a_n) 5 - 1]

theorem geom_seq (m : ℕ) (hm : 2 ≤ m) : 
  ∀ n : ℕ, 1 ≤ n → ∃ r : ℤ, r = 2 ∧ second_order_derivative a_n m = 2 ^ (n+2) * (m - 1) :=
begin
  sorry
end

-- Verify the first-term theorem
#eval first_four_terms  -- Should return [8, 16, 32, 64]

end geom_seq_l581_581814


namespace divide_cube_into_8_smaller_cubes_l581_581253

-- Defining a cube and a function to divide it into 8 identical smaller cubes
structure Cube where
  side_length : ℝ
  volume : ℝ := side_length ^ 3

-- Defining a function that checks if a cube is divided into 8 identical smaller cubes
def divide_cube (c : Cube) : Prop :=
  let smaller_side_length := c.side_length / 2
  let smaller_volume := smaller_side_length ^ 3
  (smaller_volume * 8 = c.volume)

-- The theorem statement: A cube can be divided into 8 identical smaller cubes
theorem divide_cube_into_8_smaller_cubes (c : Cube) : divide_cube c :=
begin
  -- Implementation proof is skipped
  sorry
end

end divide_cube_into_8_smaller_cubes_l581_581253


namespace rectangle_midpoint_PQ_square_l581_581023

theorem rectangle_midpoint_PQ_square {A B C D P Q : Type} [EuclideanGeometry A B C D P Q] 
  (rectangle_ABCD : rectangle A B C D)
  (midpoint_BC : midpoint B P C)
  (midpoint_DA : midpoint D Q A)
  (AB_eq_10 : length A B = 10)
  (BC_eq_26 : length B C = 26) :
  (length P Q) ^ 2 = 100 :=
sorry

end rectangle_midpoint_PQ_square_l581_581023


namespace simplify_fraction_product_l581_581095

theorem simplify_fraction_product : 
  (256 / 20 : ℚ) * (10 / 160) * ((16 / 6) ^ 2) = 256 / 45 :=
by norm_num

end simplify_fraction_product_l581_581095


namespace longest_side_AB_l581_581370

-- Definitions of angles in the quadrilateral
def angle_ABC := 65
def angle_BCD := 70
def angle_CDA := 60

/-- In a quadrilateral ABCD with angles as specified, prove that AB is the longest side. -/
theorem longest_side_AB (AB BC CD DA : ℝ) : 
  (angle_ABC = 65 ∧ angle_BCD = 70 ∧ angle_CDA = 60) → 
  AB > DA ∧ AB > BC ∧ AB > CD :=
by
  intros h
  sorry

end longest_side_AB_l581_581370


namespace sum_a_b_l581_581005

theorem sum_a_b (a b : ℝ) (h : set_of (λ x, (x - a) / (x - b) > 0) = set_of (λ x, x < 1) ∪ set_of (λ x, x > 4)) : a + b = 5 :=
sorry

end sum_a_b_l581_581005


namespace closure_properties_of_v_l581_581071

def v := {n : ℕ | ∃ k : ℕ, n = k^3}

theorem closure_properties_of_v :
  ¬ (∀ a b ∈ v, a + b ∈ v) ∧
  (∀ a b ∈ v, a * b ∈ v) ∧
  (∀ a b ∈ v, a ≠ 0 → b ≠ 0 → a / b ∈ v) ∧
  ¬ (∀ a ∈ v, 1 / a ∈ v) :=
by
  sorry

end closure_properties_of_v_l581_581071


namespace number_of_even_factors_of_M_l581_581060

def M : ℕ := 2^5 * 3^4 * 5^3 * 7^3 * 11^1

theorem number_of_even_factors_of_M : 
  let even_factors := ∑ a in Finset.range 6, 
                      ∑ b in Finset.range 5, 
                      ∑ c in Finset.range 4,
                      ∑ d in Finset.range 4,
                      ∑ e in Finset.range 2,
                      1
  in even_factors = 800 := by
  let even_factors := ∑ a in Finset.range 6, 
                      ∑ b in Finset.range 5, 
                      ∑ c in Finset.range 4,
                      ∑ d in Finset.range 4,
                      ∑ e in Finset.range 2,
                      1
  show even_factors = 800
  sorry

end number_of_even_factors_of_M_l581_581060


namespace kaiden_cans_in_third_week_l581_581397

-- Kaiden collects 158 cans in the first week
def first_week_collected : ℕ := 158
-- Kaiden increases his collection rate by 25% in the second week
def increase_rate : ℝ := 0.25
-- Kaiden's goal is to collect 500 cans in total over three weeks
def goal : ℕ := 500

-- Declare the main theorem
theorem kaiden_cans_in_third_week (cans_in_first_week cans_in_second_week : ℕ) 
    (total_cans : ℕ) (cans_needed : ℕ) :
  (cans_in_first_week = first_week_collected) →
  (cans_in_second_week = first_week_collected + (first_week_collected * increase_rate).toInt) →
  (total_cans = cans_in_first_week + cans_in_second_week) →
  (cans_needed = goal - total_cans) →
  cans_needed = 145 :=
  by
  intros h1 h2 h3 h4
  sorry

end kaiden_cans_in_third_week_l581_581397


namespace base_7_divisibility_l581_581632

theorem base_7_divisibility (y : ℕ) :
  (934 + 7 * y) % 19 = 0 ↔ y = 3 :=
by
  sorry

end base_7_divisibility_l581_581632


namespace trig_identity_l581_581295

theorem trig_identity (α : ℝ) (h : sin (α + π / 3) = 1 / 3) : cos (π / 6 - α) = 1 / 3 :=
by
  sorry

end trig_identity_l581_581295


namespace domain_of_sqrt_2_cos_x_minus_1_l581_581279

theorem domain_of_sqrt_2_cos_x_minus_1 :
  {x : ℝ | ∃ k : ℤ, - (Real.pi / 3) + 2 * k * Real.pi ≤ x ∧ x ≤ (Real.pi / 3) + 2 * k * Real.pi } =
  {x : ℝ | 2 * Real.cos x - 1 ≥ 0 } :=
sorry

end domain_of_sqrt_2_cos_x_minus_1_l581_581279


namespace sixth_term_sequence_l581_581270

noncomputable def a_seq : ℕ → ℕ
| 0 := 0
| (n+1) := a_seq n + 2 ^ (n + 1)

theorem sixth_term_sequence : a_seq 5 = 62 :=
by
  have h1 : a_seq 0 = 0 := rfl
  have h2 : a_seq 1 = 2 := by rw [a_seq, h1]; norm_num
  have h3 : a_seq 2 = 6 := by rw [a_seq, h2]; norm_num
  have h4 : a_seq 3 = 14 := by rw [a_seq, h3]; norm_num
  have h5 : a_seq 4 = 30 := by rw [a_seq, h4]; norm_num
  have h6 : a_seq 5 = 62 := by rw [a_seq, h5]; norm_num
  exact h6

end sixth_term_sequence_l581_581270


namespace real_roots_range_l581_581470

theorem real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k*x^2 - 6*x + 9 = 0) ↔ k ≤ 1 :=
sorry

end real_roots_range_l581_581470


namespace milburg_population_correct_l581_581496

-- Definitions for the initial population counts
def grown_ups := 5256
def children := 2987
def senior_citizens := 840
def infants := 1224

-- Definition for the population growth rate
def growth_rate := 0.06

-- Calculate the total population before growth
def total_population_before_growth := grown_ups + children + senior_citizens + infants

-- Calculate the population growth count (rounded to nearest whole number)
def population_growth := Int.floor (total_population_before_growth * growth_rate)

-- Calculate the current population after growth
def current_population := total_population_before_growth + population_growth

-- Statement to prove the final population
theorem milburg_population_correct : current_population = 11985 := by
  sorry

end milburg_population_correct_l581_581496


namespace tan_value_of_point_on_graph_l581_581813

theorem tan_value_of_point_on_graph (a : ℝ) (h : (4 : ℝ) ^ (1/2) = a) : 
  Real.tan ((a / 6) * Real.pi) = Real.sqrt 3 :=
by 
  sorry

end tan_value_of_point_on_graph_l581_581813


namespace perfect_squares_and_cubes_l581_581776

def is_perfect_square (n: ℕ) : Prop := ∃ k: ℕ, k ^ 2 = n
def is_perfect_cube (n: ℕ) : Prop := ∃ k: ℕ, k ^ 3 = n
def is_perfect_sixth_power (n: ℕ) : Prop := ∃ k: ℕ, k ^ 6 = n

def num_perfect_squares (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_square n).length
def num_perfect_cubes (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_cube n).length
def num_perfect_sixth_powers (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_sixth_power n).length

theorem perfect_squares_and_cubes (bound: ℕ) (hbound: bound = 1000) :
  num_perfect_squares bound + num_perfect_cubes bound - num_perfect_sixth_powers bound = 37 :=
by
  -- conditions:
  have ps := num_perfect_squares bound
  have pc := num_perfect_cubes bound
  have p6 := num_perfect_sixth_powers bound
  sorry

end perfect_squares_and_cubes_l581_581776


namespace sachin_age_l581_581093

variable {S R : ℕ}

theorem sachin_age : R = S + 4 ∧ S * 9 = R * 7 → S = 14 :=
by
  intro h
  cases h with h1 h2
  sorry

end sachin_age_l581_581093


namespace solution_set_of_inequality_system_l581_581490

theorem solution_set_of_inequality_system (x : ℝ) :
  (2 - x < 0 ∧ -2x < 6) → x > 2 :=
by
  sorry

end solution_set_of_inequality_system_l581_581490


namespace total_population_after_births_l581_581985

theorem total_population_after_births:
  let initial_population := 300000
  let immigrants := 50000
  let emigrants := 30000
  let pregnancies_fraction := 1 / 8
  let twins_fraction := 1 / 4
  let net_population := initial_population + immigrants - emigrants
  let pregnancies := net_population * pregnancies_fraction
  let twin_pregnancies := pregnancies * twins_fraction
  let twin_children := twin_pregnancies * 2
  let single_births := pregnancies - twin_pregnancies
  net_population + single_births + twin_children = 370000 := by
  sorry

end total_population_after_births_l581_581985


namespace count_perfect_squares_and_cubes_l581_581738

theorem count_perfect_squares_and_cubes : 
  let num_squares := 31 in
  let num_cubes := 9 in
  let num_sixth_powers := 3 in
  num_squares + num_cubes - num_sixth_powers = 37 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581738


namespace earnings_proof_l581_581181

theorem earnings_proof (A B C : ℕ) (h1 : A + B + C = 600) (h2 : B + C = 300) (h3 : C = 100) : A + C = 400 :=
sorry

end earnings_proof_l581_581181


namespace unique_bijective_function_l581_581864

noncomputable def find_bijective_function {n : ℕ}
  (hn : n ≥ 3) (hodd : n % 2 = 1)
  (x : Fin n → ℝ)
  (f : Fin n → ℝ) : Prop :=
∀ i : Fin n, f i = x i

theorem unique_bijective_function (n : ℕ) (hn : n ≥ 3) (hodd : n % 2 = 1)
  (x : Fin n → ℝ) (f : Fin n → ℝ)
  (hf_bij : Function.Bijective f)
  (h_abs_diff : ∀ i, |f i - x i| = 0) : find_bijective_function hn hodd x f :=
by
  sorry

end unique_bijective_function_l581_581864


namespace angle_ADC_eq_135_degrees_l581_581828

variables (ABCD : Type) [parallelogram ABCD]
variables (angle_ABCD : ∀ {a b c d}, parallelogram ABCD → int → int → int → int → Prop)
variables {A B C D : ABCD}
variables {α β γ δ : int}

-- Given ABCD is parallelogram and angle ABC is 3 times angle BCD
axiom parallelogram_ABCD : parallelogram ABCD
axiom measure_angle_ABC_eq_3_times_angle_BCD :
  angle_ABCD parallelogram_ABCD α β γ δ → α = 3 * β

-- Prove that measure of angle ADC is 135 degrees
theorem angle_ADC_eq_135_degrees :
  angle_ABCD parallelogram_ABCD 135 γ β δ → γ = 135 :=
sorry

end angle_ADC_eq_135_degrees_l581_581828


namespace ratio_of_divided_height_equals_tan_product_l581_581294

variables {ABC : Type*} [euclidean_geometry ABC]
variables {A B C H3 M N : Point}
variables {AC BC CH3 MN : Line}
variables (h_perp1 : is_perpendicular H3 M AC)
variables (h_perp2 : is_perpendicular H3 N BC)
variables (h_height : is_height C H3 ABC)
variables (h_intersect : MN ∩ CH3 ≠ ∅ )

theorem ratio_of_divided_height_equals_tan_product :
  divides_height_ratio MN CH3 = |tan_angle A * tan_angle B| :=
sorry

end ratio_of_divided_height_equals_tan_product_l581_581294


namespace smith_family_seating_l581_581106

-- Define the problem as a Lean statement without proof
theorem smith_family_seating :
  ∀ (sons daughters : ℕ), sons = 4 → daughters = 3 → 
  let total_seats := sons + daughters in
  let total_arrangements := Nat.factorial total_seats in
  let no_adjacent_boys := Nat.factorial sons * Nat.factorial daughters in
  ((total_arrangements - no_adjacent_boys) = 4896) := 
by {
  assume sons daughters,
  assume h_sons : sons = 4,
  assume h_daughters : daughters = 3,
  let total_seats := sons + daughters,
  let total_arrangements := Nat.factorial total_seats,
  let no_adjacent_boys := Nat.factorial sons * Nat.factorial daughters,
  have h1 : total_arrangements = 5040, sorry,
  have h2 : no_adjacent_boys = 144, sorry,
  calc
  total_arrangements - no_adjacent_boys
      = 5040 - 144 : by rw [h1, h2]
  ... = 4896 : by norm_num,
}

end smith_family_seating_l581_581106


namespace sum_first_2500_terms_l581_581586

noncomputable def problem_sequence (b : ℕ → ℤ) : Prop :=
  (∀ n ≥ 3, b n = b (n - 1) - b (n - 2)) ∧
  (∑ i in Finset.range 1600, b i = 2023) ∧
  (∑ i in Finset.range 2023, b i = 1600) ∧
  (∀ k, ∑ i in Finset.range (6 * k + 6), b (i + k * 6) = 0)

theorem sum_first_2500_terms :
  ∃ b : ℕ → ℤ, problem_sequence b → ∑ i in Finset.range 2500, b i = -754 := 
sorry

end sum_first_2500_terms_l581_581586


namespace derivative_at_zero_l581_581298

noncomputable def f (x : ℝ) : ℝ := x^2 - x * (deriv f 2)

theorem derivative_at_zero : (deriv f 0) = -2 := sorry

end derivative_at_zero_l581_581298


namespace geometry_problem_l581_581022

-- Define the geometry problem in Lean

theorem geometry_problem
  (A B C D P Q R S T : Type)
  [Line P Q R S T]
  (hCAB_BCD : angle C A B = angle B C D)
  (hAP_PC : dist A P = dist P C)
  (hAC_DQ_parallel : parallel C A D Q)
  (hR : intersect_line A B = intersect_line C D)
  (hS : intersect_line C A = intersect_line Q R)
  (hCircumcircle_T : on_circumcircle T (triangle A Q S) (intersect_line A D))
  : parallel A B Q T := 
sorry

end geometry_problem_l581_581022


namespace part_one_part_two_l581_581724

open Real

def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

theorem part_one (h1 : ∀ x, f (x + π) = f x)
                 (h2 : ∀ x, f (x + π / 6) = f (-x + π / 6))
                 (h3 : f 0 = 1) :
  f = λ x, 2 * sin (2 * x + π / 6) :=
by sorry

theorem part_two (α β : ℝ)
                 (hαβ1 : α ∈ Ioc 0 (π / 4))
                 (hαβ2 : β ∈ Ioc 0 (π / 4))
                 (h4 : f (α - π / 3) = -10 / 13)
                 (h5 : f (β + π / 6) = 6 / 5) :
  cos (2 * α - 2 * β) = 63 / 65 :=
by sorry

end part_one_part_two_l581_581724


namespace slope_range_PA2_l581_581291

-- Define the given conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

def A1 : ℝ × ℝ := (-2, 0)
def A2 : ℝ × ℝ := (2, 0)
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.fst P.snd

-- Define the range of the slope of line PA1
def slope_range_PA1 (k_PA1 : ℝ) : Prop := -2 ≤ k_PA1 ∧ k_PA1 ≤ -1

-- Main theorem
theorem slope_range_PA2 (x0 y0 k_PA1 k_PA2 : ℝ) (h1 : on_ellipse (x0, y0)) (h2 : slope_range_PA1 k_PA1) :
  k_PA1 = (y0 / (x0 + 2)) →
  k_PA2 = (y0 / (x0 - 2)) →
  - (3 / 4) = k_PA1 * k_PA2 →
  (3 / 8) ≤ k_PA2 ∧ k_PA2 ≤ (3 / 4) :=
by
  sorry

end slope_range_PA2_l581_581291


namespace find_second_part_sum_l581_581958

-- Definitions: The total sum, interest rates, and durations
def total_sum : ℝ := 2665
def interest_rate_first_part : ℝ := 3 / 100
def duration_first_part : ℝ := 8
def interest_rate_second_part : ℝ := 5 / 100
def duration_second_part : ℝ := 3

-- Given the conditions:
def interest_first_part (x : ℝ) : ℝ := x * interest_rate_first_part * duration_first_part
def interest_second_part (x : ℝ) : ℝ := (total_sum - x) * interest_rate_second_part * duration_second_part
def condition (x : ℝ) : Prop := interest_first_part x = interest_second_part x

-- The main theorem to prove
theorem find_second_part_sum : ∃ x : ℝ, condition x ∧ (total_sum - x = 1640) :=
by
  -- Proof will go here
  sorry

end find_second_part_sum_l581_581958


namespace fractional_identity_l581_581312

theorem fractional_identity (m n r t : ℚ) 
  (h₁ : m / n = 5 / 2) 
  (h₂ : r / t = 8 / 5) : 
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -5 / 11 :=
by 
  sorry

end fractional_identity_l581_581312


namespace equivalent_proof_l581_581434

noncomputable def find_m_n : ℕ :=
  let boxes := [4, 5, 6] in
  let totalWays := Nat.choose 15 4 * Nat.choose (15 - 4) 5 * Nat.choose (15 - 4 - 5) 6 in
  let mathIn4 := Nat.choose 11 5 * Nat.choose (11 - 5) 6 in
  let mathIn5 := Nat.choose 11 1 * Nat.choose 10 4 * Nat.choose 6 6 in
  let mathIn6 := Nat.choose 11 2 * Nat.choose 9 4 * Nat.choose 5 5 in
  let totalMathWays := mathIn4 + mathIn5 + mathIn6 in
  let prob := totalMathWays * totalWays in
  let gcd_val := Nat.gcd (totalMathWays * totalWays) totalWays in
  let m := (totalMathWays * totalWays) / gcd_val in
  let n := totalWays / gcd_val in
  m + n

theorem equivalent_proof : 
  ∃ m n : ℕ, 
    (Nat.gcd m n = 1) ∧
    (m / Nat.gcd m n + n / Nat.gcd m n = find_m_n) :=
by
  sorry

end equivalent_proof_l581_581434


namespace quadratic_polynomial_exists_l581_581661

theorem quadratic_polynomial_exists (p : ℚ → ℚ) :
  (p(-6) = 0) →
  (p(3) = 0) →
  (p(-3) = -40) →
  (∀ x, p x = (20 / 9) * x^2 + (20 / 3) * x - 40) :=
by
  intro h₁ h₂ h₃
  sorry

end quadratic_polynomial_exists_l581_581661


namespace three_secretaries_project_l581_581462

theorem three_secretaries_project (t1 t2 t3 : ℕ) 
  (h1 : t1 / t2 = 1 / 2) 
  (h2 : t1 / t3 = 1 / 5) 
  (h3 : t3 = 75) : 
  t1 + t2 + t3 = 120 := 
  by 
    sorry

end three_secretaries_project_l581_581462


namespace area_eq_85pi_l581_581649

variable (P Q : Point) (r : ℝ)

def circle_center (P : Point) : Bool :=
  P = (2, -1)

def circle_passes_through (Q : Point) : Bool :=
  Q = (-4, 6)

noncomputable def radius : ℝ :=
  Real.sqrt ((-4 - 2)^2 + (6 - (-1))^2)

noncomputable def area_of_circle : ℝ :=
  π * (radius) ^ 2

theorem area_eq_85pi (hP : circle_center P) (hQ : circle_passes_through Q) : 
  area_of_circle = 85 * π :=
by
  sorry

end area_eq_85pi_l581_581649


namespace perfect_squares_and_cubes_count_lt_1000_l581_581784

theorem perfect_squares_and_cubes_count_lt_1000 : 
  let perfect_square := λ n : ℕ, ∃ k : ℕ, n = k^2
  let perfect_cube := λ n : ℕ, ∃ k : ℕ, n = k^3
  let sixth_power := λ n : ℕ, ∃ k : ℕ, n = k^6
  (count (λ n, n < 1000 ∧ (perfect_square n ∨ perfect_cube n)) (Finset.range 1000)) = 38 :=
by {
  sorry
}

end perfect_squares_and_cubes_count_lt_1000_l581_581784


namespace increasing_intervals_triangle_side_length_l581_581329

noncomputable def f(x : ℝ) : ℝ := sin x ^ 2 - cos x ^ 2 + 2 * sqrt 3 * sin x * cos x

theorem increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, -π/6 + k*π ≤ x ∧ x ≤ π/3 + k*π ↔ 0 < (λ y, f y) x.to_real_deriv :=
sorry

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (k : ℤ) (hA : f A = 2) (hc : c = 5) (hcosB : cos B = 1/7) :
  a = 7 :=
sorry

end increasing_intervals_triangle_side_length_l581_581329


namespace barney_extra_weight_l581_581611

-- Define the weight of a regular dinosaur
def regular_dinosaur_weight : ℕ := 800

-- Define the combined weight of five regular dinosaurs
def five_regular_dinosaurs_weight : ℕ := 5 * regular_dinosaur_weight

-- Define the total weight of Barney and the five regular dinosaurs together
def total_combined_weight : ℕ := 9500

-- Define the weight of Barney
def barney_weight : ℕ := total_combined_weight - five_regular_dinosaurs_weight

-- The proof statement
theorem barney_extra_weight : barney_weight - five_regular_dinosaurs_weight = 1500 :=
by sorry

end barney_extra_weight_l581_581611


namespace max_difference_in_set_l581_581169

theorem max_difference_in_set : 
  let S := {-12, -6, 0, 3, 7, 15}
  in ∃ x y ∈ S, (∀ a b ∈ S, a - b ≤ x - y) ∧ x - y = 27 :=
by
  sorry

end max_difference_in_set_l581_581169


namespace collinear_DGX_l581_581413

-- Let ABC be an acute triangle and γ its circumcircle.
variables {A B C : Point}
variables [acute ∠A ∠B ∠C] -- assuming it's possible to define acute angles in this way

-- Let B' be the midpoint of AC and C' the midpoint of AB.
variables {B' C' : Point}
def midpoint_AC (B' : Point) : Prop := dist(A, B') = dist(B', C)
def midpoint_AB (C' : Point) : Prop := dist(A, C') = dist(C', B)

-- Let D be the foot of the altitude from A.
variable {D : Point}
def foot_altitude_A (D : Point) : Prop := ∃ (H : Line), perpendicular H (Line A B) ∧ foot D A H

-- Let G be the centroid of ΔABC.
variable {G : Point}
def centroid_ABC (G : Point) : Prop := ∃ (M N P : Point), each M, N, P ∈ {midpoints of sides of ΔABC},
                                      ∃ (H1 H2 H3 : Line), G ∈ H1 ∧ G ∈ H2 ∧ G ∈ H3

-- Let ω be the circle passing through B' and C' and tangent to γ at a point X ≠ A.
variables {ω γ : Circle} {X : Point}
def circle_tangent (ω γ : Circle) (X : Point) (h : X ≠ A) : Prop := ∃ (T : TangentPoint), circle_tangent_at ω γ T ∧ T = X

-- Prove that D, G, and X are collinear.
theorem collinear_DGX : 
  acute ∠A ∠B ∠C → 
  midpoint_AC B' → 
  midpoint_AB C' → 
  foot_altitude_A D → 
  centroid_ABC G → 
  circle_tangent ω γ X X_ne_A → 
  collinear {D, G, X} :=
by
  sorry

end collinear_DGX_l581_581413


namespace simplify_expr1_simplify_expr2_l581_581097

variable {a b : ℝ}

theorem simplify_expr1 : 3 * a - (4 * b - 2 * a + 1) = 5 * a - 4 * b - 1 :=
by
  sorry

theorem simplify_expr2 : 2 * (5 * a - 3 * b) - 3 * (a ^ 2 - 2 * b) = 10 * a - 3 * a ^ 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l581_581097


namespace ramesh_discount_percentage_l581_581890

theorem ramesh_discount_percentage :
  ∃ (P : ℝ), 
    P + 0.1 * P = 20350 ∧
    P - 14500 = 4000 ∧
    ((P - 14500) / P) * 100 ≈ 21.62 := 
by {
  sorry
}

end ramesh_discount_percentage_l581_581890


namespace min_value_l581_581711

theorem min_value (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_sum : a + b = 1) : 
  ∃ x : ℝ, (x = 25) ∧ x ≤ (4 / a + 9 / b) :=
by
  sorry

end min_value_l581_581711


namespace relationship_abc_l581_581297

noncomputable def a : ℝ := 5 ^ (Real.log 3.4 / Real.log 3)
noncomputable def b : ℝ := 5 ^ (Real.log 3.6 / Real.log 3)
noncomputable def c : ℝ := (1 / 5) ^ (Real.log 0.5 / Real.log 3)

theorem relationship_abc : b > a ∧ a > c := by 
  sorry

end relationship_abc_l581_581297


namespace max_seq_len_1967_1_l581_581213

-- Define the sequence
def seq_fun (a : ℕ) (b : ℕ) : ℕ :=
  abs (a - b)

-- Define the maximum function
noncomputable def sequence_max_length (a₀ a₁ : ℕ) : ℕ :=
  let rec aux (a₀ a₁ : ℕ) (count: ℕ): ℕ :=
    let a₂ := seq_fun a₀ a₁ in
    if a₂ = 0 then count + 1 else aux a₁ a₂ (count + 1)
  in aux a₀ a₁ 2

-- State the theorem
theorem max_seq_len_1967_1 : sequence_max_length 1967 1 = 2951 :=
  sorry

end max_seq_len_1967_1_l581_581213


namespace probability_abs_diff_less_than_one_third_l581_581090

def coin_flip_distribution : ProbabilityMeasure ℝ :=
  sorry -- This represents the probability distribution described in the problem

noncomputable def prob_diff_lt_one_third (x y : ℝ) : ℝ :=
  coin_flip_distribution.prob {z : ℝ × ℝ | abs (z.1 - z.2) < 1/3}

theorem probability_abs_diff_less_than_one_third :
  let x := coin_flip_distribution,
      y := coin_flip_distribution in
  prob_diff_lt_one_third x y = 1/8 :=
by
  sorry

end probability_abs_diff_less_than_one_third_l581_581090


namespace hours_of_work_l581_581161

variables (M W X : ℝ)

noncomputable def work_rate := 
  (2 * M + 3 * W) * X * 5 = 1 ∧ 
  (4 * M + 4 * W) * 3 * 7 = 1 ∧ 
  7 * M * 4 * 5.000000000000001 = 1

theorem hours_of_work (M W : ℝ) (h : work_rate M W 7) : X = 7 :=
sorry

end hours_of_work_l581_581161


namespace isosceles_triangles_in_regular_hexagon_l581_581973

theorem isosceles_triangles_in_regular_hexagon
  (A B C D E F O : Point)
  (h_hexagon : regular_hexagon A B C D E F)
  (h_center : center O A B C D E F)
  : number_of_isosceles_triangles_with_vertices_in A B C D E F O = 20 :=
sorry

end isosceles_triangles_in_regular_hexagon_l581_581973


namespace perfect_squares_and_cubes_l581_581779

def is_perfect_square (n: ℕ) : Prop := ∃ k: ℕ, k ^ 2 = n
def is_perfect_cube (n: ℕ) : Prop := ∃ k: ℕ, k ^ 3 = n
def is_perfect_sixth_power (n: ℕ) : Prop := ∃ k: ℕ, k ^ 6 = n

def num_perfect_squares (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_square n).length
def num_perfect_cubes (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_cube n).length
def num_perfect_sixth_powers (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_sixth_power n).length

theorem perfect_squares_and_cubes (bound: ℕ) (hbound: bound = 1000) :
  num_perfect_squares bound + num_perfect_cubes bound - num_perfect_sixth_powers bound = 37 :=
by
  -- conditions:
  have ps := num_perfect_squares bound
  have pc := num_perfect_cubes bound
  have p6 := num_perfect_sixth_powers bound
  sorry

end perfect_squares_and_cubes_l581_581779


namespace prime_gt_3_div_24_num_form_6n_plus_minus_1_div_24_l581_581087

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_gt_3_div_24 (p : ℕ) (hp : is_prime p) (h : p > 3) : 
  24 ∣ (p^2 - 1) :=
sorry

theorem num_form_6n_plus_minus_1_div_24 (n : ℕ) : 
  24 ∣ (6 * n + 1)^2 - 1 ∧ 24 ∣ (6 * n - 1)^2 - 1 :=
sorry

end prime_gt_3_div_24_num_form_6n_plus_minus_1_div_24_l581_581087


namespace garbage_company_charge_per_trash_bin_l581_581433

theorem garbage_company_charge_per_trash_bin :
  ∃ T : ℝ, 
    let weekly_trash_bin_cost := 2 * T in
    let weekly_recycling_bin_cost := 5 in
    let monthly_trash_bin_cost := 4 * weekly_trash_bin_cost in
    let monthly_recycling_bin_cost := 4 * weekly_recycling_bin_cost in
    let total_monthly_cost_before_discount_and_fine := monthly_trash_bin_cost + monthly_recycling_bin_cost in
    let discount := 0.18 * total_monthly_cost_before_discount_and_fine in
    let fine := 20 in
    let total_cost_before_fine := total_monthly_cost_before_discount_and_fine - discount in
    let final_cost := total_cost_before_fine + fine in
    final_cost = 102 ∧ T = 10 := by
  -- proof should follow
  sorry

end garbage_company_charge_per_trash_bin_l581_581433


namespace copies_of_2019_in_row_2019_l581_581383

def pattern (n : ℕ) : list ℕ :=
  match n with
  | 0 => [1, 1]
  | n+1 => 
    let prev_row := pattern (n) in
    (prev_row.zip (prev_row.tail!)).foldr (λ (ab : ℕ×ℕ) acc, ab.1 :: (ab.1 + ab.2) :: acc) []

def euler_totient (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (Finset.range (n + 1)).filter (λ m => Nat.gcd n m = 1).card

theorem copies_of_2019_in_row_2019 : (pattern 2019).count 2019 = euler_totient 2019 :=
  sorry

end copies_of_2019_in_row_2019_l581_581383


namespace original_square_side_length_l581_581450

-- Defining the variables and conditions
variables (x : ℝ) (h₁ : 1.2 * x * (x - 2) = x * x)

-- Theorem statement to prove the side length of the original square is 12 cm
theorem original_square_side_length : x = 12 :=
by
  sorry

end original_square_side_length_l581_581450


namespace count_perfect_squares_cubes_under_1000_l581_581754

-- Condition definitions
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, m * m = n
def isPerfectCube (n : Nat) : Prop := ∃ m : Nat, m * m * m = n

-- Main proof problem
theorem count_perfect_squares_cubes_under_1000 : 
  let count (p : Nat -> Prop) := Finset.card (Finset.filter p (Finset.range 1000))
  count (λ n, isPerfectSquare n ∨ isPerfectCube n) = 37 := 
sorry

end count_perfect_squares_cubes_under_1000_l581_581754


namespace prob_A_exactly_once_l581_581387

theorem prob_A_exactly_once (P : ℚ) (h : 1 - (1 - P)^3 = 63 / 64) : 
  (3 * P * (1 - P)^2 = 9 / 64) :=
by
  sorry

end prob_A_exactly_once_l581_581387


namespace necessary_but_not_sufficient_l581_581972

theorem necessary_but_not_sufficient (x : ℝ) : 
  (0 < x ∧ x < 2) → (x^2 - x - 6 < 0) ∧ ¬ ((x^2 - x - 6 < 0) → (0 < x ∧ x < 2)) :=
by
  sorry

end necessary_but_not_sufficient_l581_581972


namespace ratio_of_volumes_l581_581599

/-- Define the volume of a cylinder given radius and height -/
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

/-- Define the conditions of the problem as constants -/
def alex_can : ℝ := cylinder_volume 2 10          -- Alex's can volume
def felicia_can : ℝ := cylinder_volume 5 4        -- Felicia's can volume
def alex_total_volume : ℝ := 10 * alex_can        -- Total volume Alex uses in a week
def felicia_total_volume : ℝ := 5 * felicia_can   -- Total volume Felicia uses in a week

/-- The theorem to be proved -/
theorem ratio_of_volumes : alex_total_volume / felicia_total_volume = 4 / 5 :=
by
  sorry

end ratio_of_volumes_l581_581599


namespace exists_integer_with_properties_l581_581036

theorem exists_integer_with_properties :
  ∃ (N : ℕ), 
    (∃ (m : ℕ), N = (1990 * m + (1990 * (1990 - 1)) / 2)) ∧
    ∃ (count : ℕ), count = 1990 ∧ 
    ∀ (n k : ℕ), k ≥ 2 → (N = k * n + (k * (k - 1)) / 2) → ∃ (cnt : ℕ), cnt = count :=
begin
  sorry
end

end exists_integer_with_properties_l581_581036


namespace divisible_by_five_l581_581446

theorem divisible_by_five {x y z : ℤ} (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  5 ∣ ((x - y)^5 + (y - z)^5 + (z - x)^5) :=
sorry

end divisible_by_five_l581_581446


namespace initial_velocity_is_three_l581_581605

noncomputable def displacement (t : ℝ) : ℝ :=
  3 * t - t^2

theorem initial_velocity_is_three : 
  (deriv displacement 0) = 3 :=
by
  sorry

end initial_velocity_is_three_l581_581605


namespace point_A_in_Quadrant_IV_l581_581083

-- Define the coordinates of point A
def A : ℝ × ℝ := (5, -4)

-- Define the quadrants based on x and y signs
def in_Quadrant_I (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def in_Quadrant_II (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def in_Quadrant_III (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def in_Quadrant_IV (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Statement to prove that point A lies in Quadrant IV
theorem point_A_in_Quadrant_IV : in_Quadrant_IV A :=
by
  sorry

end point_A_in_Quadrant_IV_l581_581083


namespace FruitKeptForNextWeek_l581_581431

/-- Define the variables and conditions -/
def total_fruit : ℕ := 10
def fruit_eaten : ℕ := 5
def fruit_brought_on_friday : ℕ := 3

/-- Define what we need to prove -/
theorem FruitKeptForNextWeek : 
  ∃ k, total_fruit - fruit_eaten - fruit_brought_on_friday = k ∧ k = 2 :=
by
  sorry

end FruitKeptForNextWeek_l581_581431


namespace james_needs_more_marbles_l581_581845

def number_of_additional_marbles (friends marbles : Nat) : Nat :=
  let required_marbles := (friends * (friends + 1)) / 2
  (if marbles < required_marbles then required_marbles - marbles else 0)

theorem james_needs_more_marbles :
  number_of_additional_marbles 15 80 = 40 := by
  sorry

end james_needs_more_marbles_l581_581845


namespace hyperbola_properties_l581_581654

theorem hyperbola_properties :
  let a := 3
  let b := (3 / 2 : Real)
  let c := sqrt (9 + 9/4)
  let e := c / a
  (a, b, c) = (3, 3/2, 3 * sqrt 5 / 2) ∧
  2 * a = 6 ∧
  e = sqrt 5 / 2 ∧
  (± (3 * sqrt 5 / 2), 0) = ((3 * sqrt 5 / 2), 0) ∧
  (± (3), 0) = (3, 0) :=
by
  sorry

end hyperbola_properties_l581_581654


namespace inscribed_quadrilateral_property_l581_581930

open EuclideanGeometry

-- Define the setup for the problem
variables {A B C D K L M N P Q : Point} [InscribedQuadrilateral A B C D]

-- Incircle touches sides AB, BC, CD, DA at K, L, M, N respectively
axiom touch_points : IncircleTouchPoints A B C D K L M N

-- Line through C is parallel to diagonal BD
axiom parallel_line : ∀ α β γ δ : Point, Parallel α γ β δ → Line (C α) → ParallelLine 

-- Line intersects NL and KM at points P and Q respectively
axiom intersections_PQ : ∀ α β γ δ ε ζ : Point, LineParallelα ψ γ δ ε ζ, LineIntersect α γ ζ β

-- Prove CP = CQ
theorem inscribed_quadrilateral_property :
  CP = CQ :=
by
  sorry

end inscribed_quadrilateral_property_l581_581930


namespace find_angle_of_pentagon_inscribed_in_circle_l581_581208

-- Definitions for the problem
def isPentinCircle (p : List ℝ) : Prop := p.length = 5 ∧ (∀ angle ∈ p, 0 < angle ∧ angle < 180)
def sumOfAngles (angles : List ℝ) : ℝ := List.foldr (+) 0 angles
def allAnglesEqual (angles : List ℝ) : Prop := ∀ a ∈ angles, a = angles.head!

-- Statement of the problem in Lean 4
theorem find_angle_of_pentagon_inscribed_in_circle (angles : List ℝ) 
  (h1 : isPentinCircle angles) 
  (h2 : sumOfAngles angles = 540) 
  (h3 : allAnglesEqual angles) : 
  angles.head! = 108 := 
sorry

end find_angle_of_pentagon_inscribed_in_circle_l581_581208


namespace slower_train_speed_l581_581519

noncomputable def speed_of_slower_train (v : ℝ) : Prop :=
  let speed_faster := 45.0
  let length_faster := 270.0216
  let time := 12.0
  let relative_speed := (v + speed_faster) * (5.0 / 18.0)
  length_faster = relative_speed * time

theorem slower_train_speed : ∃ v : ℝ, speed_of_slower_train v ∧ v = 117.01296 := by
  exists 117.01296
  dsimp [speed_of_slower_train]
  rw [mul_add, mul_comm, ←mul_assoc, mul_comm (5.0 / 18.0) 12.0, mul_div_cancel' _ (by norm_num : (18.0:ℝ) ≠ 0)]
  norm_num
  linarith
  sorry

end slower_train_speed_l581_581519


namespace product_plus_one_eq_216_l581_581965

variable (a b c : ℝ)

theorem product_plus_one_eq_216 
  (h1 : a * b + a + b = 35)
  (h2 : b * c + b + c = 35)
  (h3 : c * a + c + a = 35)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) :
  (a + 1) * (b + 1) * (c + 1) = 216 := 
sorry

end product_plus_one_eq_216_l581_581965


namespace age_difference_l581_581159

theorem age_difference (b e : ℕ) (h1 : b + e = 28) (h2 : b = 16) : b - e = 4 := 
by 
  have he : e = 28 - b := Nat.add_sub_cancel_left (e) (b)
  rw [h2] at he
  rw he
  sorry

end age_difference_l581_581159


namespace first_discount_percentage_l581_581570

theorem first_discount_percentage
  (actual_price final_price : ℝ)
  (second_discount third_discount : ℕ)
  (h_actual_price : actual_price = 9356.725146198829)
  (h_final_price : final_price = 6400)
  (h_second_discount : second_discount = 10)
  (h_third_discount : third_discount = 5) :
  ∃ x : ℝ, (actual_price * (1 - x / 100) * (1 - second_discount / 100) * (1 - third_discount / 100) = final_price) ∧ (x ≈ 19.95) := 
sorry

end first_discount_percentage_l581_581570


namespace find_m_l581_581310

variables {V : Type*} [normed_add_comm_group V] [inner_product_space ℝ V]
variables (a b c : V)
variables (m : ℝ)

-- Definitions from the conditions
def c_eq_a_add_m_b : Prop := c = a + m • b
def a_perp_c : Prop := inner_product_space.is_orthogonal ℝ a c
def b_dot_c_eq_neg2 : Prop := inner_product_space.dot_product b c = -2
def norm_c_eq_2 : Prop := ∥c∥ = 2

-- The statement that puts it all together
theorem find_m 
  (h1 : c_eq_a_add_m_b a b c m)
  (h2 : a_perp_c a c)
  (h3 : b_dot_c_eq_neg2 b c)
  (h4 : norm_c_eq_2 c) : m = -2 := 
sorry

end find_m_l581_581310


namespace hotel_ticket_ratio_l581_581072

theorem hotel_ticket_ratio (initial_amount : ℕ) (remaining_amount : ℕ) (ticket_cost : ℕ) (hotel_cost : ℕ) :
  initial_amount = 760 →
  remaining_amount = 310 →
  ticket_cost = 300 →
  initial_amount - remaining_amount - ticket_cost = hotel_cost →
  (hotel_cost : ℚ) / (ticket_cost : ℚ) = 1 / 2 :=
by
  intros h_initial h_remaining h_ticket h_hotel
  sorry

end hotel_ticket_ratio_l581_581072


namespace complex_conjugate_correct_l581_581705

noncomputable def complex_conjugate_test : Prop :=
  let i := Complex.I in
  let z := (1 + 2 * i) / (2 + i) in
  let z_conjugate := IsROrC.conj z in
  z_conjugate = (4 / 5 : ℂ) - (3 / 5) * i

theorem complex_conjugate_correct : complex_conjugate_test :=
by
  sorry

end complex_conjugate_correct_l581_581705


namespace total_leaves_on_farm_l581_581220

noncomputable def number_of_branches : ℕ := 10
noncomputable def sub_branches_per_branch : ℕ := 40
noncomputable def leaves_per_sub_branch : ℕ := 60
noncomputable def number_of_trees : ℕ := 4

theorem total_leaves_on_farm :
  number_of_branches * sub_branches_per_branch * leaves_per_sub_branch * number_of_trees = 96000 :=
by
  sorry

end total_leaves_on_farm_l581_581220


namespace original_number_l581_581207

theorem original_number (x y : ℝ) (h1 : 10 * x + 22 * y = 780) (h2 : y = 34) : x + y = 37.2 :=
sorry

end original_number_l581_581207


namespace increasing_function_range_of_f_l581_581257

noncomputable def f : ℝ → ℝ :=
sorry  -- assume such f exists, based on the conditions

theorem increasing_function (f_is_function : ∀ x y : ℝ, f(x + y) = f(x) + f(y) - 1)
  (f_neg : ∀ x : ℝ, x < 0 → f(x) < 1) :
  ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) :=
sorry

theorem range_of_f (f_is_function : ∀ x y : ℝ, f(x + y) = f(x) + f(y) - 1)
  (f_3 : f(3) = 4)
  (f_is_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2))
  (f_neg : ∀ x : ℝ, x < 0 → f(x) < 1) :
  set.range (f ∘ Icc 1 3) = set.Icc 2 4 :=
sorry

end increasing_function_range_of_f_l581_581257


namespace sum_first_4n_terms_l581_581731

noncomputable def a_sequence : ℕ → ℝ 
| 0       := 0
| (n + 1) := -((-1)^n * a_sequence n) + (2 * n - 1)

def S (n : ℕ) : ℝ := ∑ i in finset.range (4 * n), a_sequence i

theorem sum_first_4n_terms (n : ℕ) : S n = 16 * n^2 + 12 * n :=
by sorry

end sum_first_4n_terms_l581_581731


namespace min_y_value_l581_581376

noncomputable def y (x : ℝ) : ℝ :=
  (x - 6.5)^2 + (x - 5.9)^2 + (x - 6.0)^2 + (x - 6.7)^2 + (x - 4.5)^2

theorem min_y_value : 
  ∃ x : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ x' : ℝ, abs (x' - 5.92) < δ → abs (y x' - y 5.92) < ε) :=
sorry

end min_y_value_l581_581376


namespace correct_prism_property_l581_581822

namespace Prism

-- Conditions as definitions using Lean's Defnitions
def is_parallelogram (f : Set ℝ^3) : Prop := sorry
def are_parallel (f1 f2 : Set ℝ^3) : Prop := sorry
def is_prism (p : Set (Set ℝ^3)) : Prop :=
  ∃ (base1 base2 : Set ℝ^3) (faces : Set (Set ℝ^3)),
    are_parallel base1 base2 ∧
    (∀ f ∈ faces, is_parallelogram f) ∧
    (∀ (f1 f2 ∈ faces), are_parallel f1 f2) ∧ 
    (sorry additional conditions about faces forming a prism)

-- Problem statement rewrite in Lean 4 type declaration
theorem correct_prism_property (p : Set (Set ℝ^3)) (h : is_prism p) :
  ∃ base1 base2 lateral_edges,
    are_parallel base1 base2 ∧ 
    are_parallel lateral_edges :=
sorry

end Prism

end correct_prism_property_l581_581822


namespace digit_permutation_of_prime_multiple_insertion_l581_581406

theorem digit_permutation_of_prime_multiple_insertion
  (p : ℕ) (k : ℕ) (A B : ℕ)
  (hprime : p.prime) (hpk : p > 10^k)
  (hA : 10^(k-1) ≤ A ∧ A < 10^k) (hB : 10^(k-1) ≤ B ∧ B < 10^k)
  (multiple_p : ∃ C : ℕ, C % p = 0)
  (ins1 : ∃ C : ℕ, C % p = 0 ∧ C = insert_digits_multiple p A k)
  (ins2 : ∃ C' : ℕ, C' % p = 0 ∧ C' = insert_digits_multiple' p C B k) :
  B.digits_permutation_of A.digits :=
sorry

end digit_permutation_of_prime_multiple_insertion_l581_581406


namespace simplify_expression_l581_581424

variable (c d : ℝ)
variable (hc : 0 < c)
variable (hd : 0 < d)
variable (h : c^3 + d^3 = 3 * (c + d))

theorem simplify_expression : (c / d) + (d / c) - (3 / (c * d)) = 1 := by
  sorry

end simplify_expression_l581_581424


namespace sum_and_product_roots_l581_581140

structure quadratic_data where
  m : ℝ
  n : ℝ

def roots_sum_eq (qd : quadratic_data) : Prop :=
  qd.m / 3 = 9

def roots_product_eq (qd : quadratic_data) : Prop :=
  qd.n / 3 = 20

theorem sum_and_product_roots (qd : quadratic_data) :
  roots_sum_eq qd → roots_product_eq qd → qd.m + qd.n = 87 := by
  sorry

end sum_and_product_roots_l581_581140


namespace slope_of_parallel_line_l581_581172

theorem slope_of_parallel_line (x y : ℝ) :
  (3 * x + 6 * y = -21) → (∃ m : ℝ, m = -1/2 ∧ parallel_slope 3 6 m) :=
begin
  intros h,
  use -1/2,
  split,
  { refl },
  { sorry }
end

end slope_of_parallel_line_l581_581172


namespace problem_statement_l581_581918

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let p := (a + b + c) / 2
  real.sqrt (p * (p - a) * (p - b) * (p - c))

theorem problem_statement :
  ∀ a b c : ℝ,
  (a^3 - 5*a^2 + 6*a - 3/5 = 0) →
  (b^3 - 5*b^2 + 6*b - 3/5 = 0) →
  (c^3 - 5*c^2 + 6*c - 3/5 = 0) →
  triangle_area a b c = 2*real.sqrt 21 / 5 :=
by
  sorry

end problem_statement_l581_581918


namespace q_at_14_l581_581861

noncomputable def q (x : ℝ) : ℝ := - (1 / 2) * x^2 + x + 2

theorem q_at_14 : q 14 = -82 := by
  sorry

end q_at_14_l581_581861


namespace perfect_squares_and_cubes_l581_581782

def is_perfect_square (n: ℕ) : Prop := ∃ k: ℕ, k ^ 2 = n
def is_perfect_cube (n: ℕ) : Prop := ∃ k: ℕ, k ^ 3 = n
def is_perfect_sixth_power (n: ℕ) : Prop := ∃ k: ℕ, k ^ 6 = n

def num_perfect_squares (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_square n).length
def num_perfect_cubes (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_cube n).length
def num_perfect_sixth_powers (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_sixth_power n).length

theorem perfect_squares_and_cubes (bound: ℕ) (hbound: bound = 1000) :
  num_perfect_squares bound + num_perfect_cubes bound - num_perfect_sixth_powers bound = 37 :=
by
  -- conditions:
  have ps := num_perfect_squares bound
  have pc := num_perfect_cubes bound
  have p6 := num_perfect_sixth_powers bound
  sorry

end perfect_squares_and_cubes_l581_581782


namespace function_constant_if_averaged_l581_581963

-- Definition of the function and its property.
def is_constant_function (f : ℤ × ℤ → ℝ) (c : ℝ) : Prop :=
  ∀ (x y : ℤ), f (x, y) = c

-- The main theorem.
theorem function_constant_if_averaged (f : ℤ × ℤ → ℝ) (h : ∀ (x y : ℤ), f (x, y) = (f (x - 1, y) + f (x, y - 1)) / 2) :
  ∃ c ∈ Icc (0 : ℝ) 1, is_constant_function f c :=
sorry

end function_constant_if_averaged_l581_581963


namespace chord_intersection_points_l581_581182

theorem chord_intersection_points (n : ℕ) (h : n ≥ 4) : 
  let num_intersections := (n.choose 4) in
  num_intersections = n! / (4! * (n - 4)!) :=
by
  sorry

end chord_intersection_points_l581_581182


namespace emir_needs_extra_money_l581_581639

def cost_dictionary : ℕ := 5
def cost_dinosaur_book : ℕ := 11
def cost_cookbook : ℕ := 5
def amount_saved : ℕ := 19

def total_cost : ℕ := cost_dictionary + cost_dinosaur_book + cost_cookbook
def amount_needed : ℕ := total_cost - amount_saved

theorem emir_needs_extra_money : amount_needed = 2 := by
  rfl -- actual proof that amount_needed equals 2 goes here
  -- Sorry can be used to skip if the proof needs additional steps.
  sorry

end emir_needs_extra_money_l581_581639


namespace maximum_ratio_of_segments_l581_581595

noncomputable theory
open_locale classical

-- Define the lattice points and their conditions
variables (A B C X : ℤ × ℤ) -- Lattice points in the plane
variables (E : ℤ × ℤ) -- Intersection point on BC

-- Establish the conditions
def is_lattice_point (p : ℤ × ℤ) : Prop := true -- as all integers pairs are lattice points

def triangle_contains_point (A B C X : ℤ × ℤ) : Prop :=
  -- A placeholder condition that denotes triangle ABC contains point X in its interior
  true

def line_intersects_segment (A X B C E : ℤ × ℤ) : Prop :=
  -- A placeholder condition that denotes line AX meets BC at point E
  true

-- Statement of the problem in Lean 4
theorem maximum_ratio_of_segments
  (h1 : is_lattice_point A) 
  (h2 : is_lattice_point B)
  (h3 : is_lattice_point C)
  (h4 : triangle_contains_point A B C X)
  (h5 : line_intersects_segment A X B C E) :
  ∃ (k : ℕ), k = 5 ∧ (∀ (AX XE : ℕ), ∃ (ratio : ℚ), ratio = AX / XE ∧ ratio ≤ k) :=
sorry

end maximum_ratio_of_segments_l581_581595


namespace red_card_distribution_l581_581231

theorem red_card_distribution :
  let A := {card ∈ ["red", "orange", "yellow", "green"] | card = "red"}
  let B := A
  mutually_exclusive A B :=
by
  -- Assume red, orange, yellow, and green playing cards are distributed to A, B, C, and D
  let A_distribute := "red"
  let B_distribute := "red"

  -- Events cannot occur simultaneously
  have hA : A_distribute ≠ B_distribute := by sorry
  show mutually_exclusive A_distribute B_distribute := by sorry

end red_card_distribution_l581_581231


namespace xy_zero_iff_x_zero_necessary_not_sufficient_l581_581683

theorem xy_zero_iff_x_zero_necessary_not_sufficient {x y : ℝ} : 
  (x * y = 0) → ((x = 0) ∨ (y = 0)) ∧ ¬((x = 0) → (x * y ≠ 0)) := 
sorry

end xy_zero_iff_x_zero_necessary_not_sufficient_l581_581683


namespace color_points_circle_l581_581933

noncomputable def largest_k_colorable_points (n m : ℕ) : Prop :=
  ∀ (k : ℕ), (k ≤ n / 2) → ∀ (coloring : Fin n → Bool),
    ∃ (remaining_coloring : Fin n → Bool), 
      (∀ i j : Fin n, (coloring i = coloring j) → 
        (∃ p : List (Fin n × Fin n), p.length = m ∧ (p.Nodup ∧ ∀ q ∈ p, (Set.Pair q.fst q.snd ∈ intervals i j))))

theorem color_points_circle : largest_k_colorable_points 100 50 := sorry

end color_points_circle_l581_581933


namespace ratio_of_shaded_to_white_areas_l581_581530

-- Define the conditions as given
def vertices_are_in_middle_squares (largest_square : Type) (small_squares : list largest_square) : Prop := 
  ∀ small_square ∈ small_squares, 
  vertices_of_small_square_at_middle_of_largest_square_vertices largest_square small_square

-- Define the theorem with the given question and correct answer
theorem ratio_of_shaded_to_white_areas
  (largest_square : Type) 
  (small_squares : list largest_square)
  (h: vertices_are_in_middle_squares largest_square small_squares) :
  ratio_of_areas_of_shaded_to_white largest_square small_squares = 5/3 :=
sorry

end ratio_of_shaded_to_white_areas_l581_581530


namespace election_total_votes_l581_581372

theorem election_total_votes (V : ℝ)
  (h_invalid : 0.3 * V)
  (h_valid_second_candidate : 0.4 * 0.7 * V = 2519.9999999999995) :
  V ≈ 9000 :=
by
  -- The proof will go here
  sorry

end election_total_votes_l581_581372


namespace problem_statement_l581_581990

theorem problem_statement : ¬ (487.5 * 10^(-10) = 0.0000004875) :=
by
  sorry

end problem_statement_l581_581990


namespace max_value_of_z_l581_581282

-- Given the problem conditions
def conditions (x y : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (2 * x + y ≤ 2)

-- Objective function
def z (x y : ℝ) : ℝ := x - 2 * y

-- Problem statement
theorem max_value_of_z : ∃ (x y : ℝ), conditions x y ∧ z x y = 1 ∧
  (∀ (x' y' : ℝ), conditions x' y' → z x' y' ≤ 1) :=
begin
  sorry -- Proof is omitted
end

end max_value_of_z_l581_581282


namespace polygon_sides_l581_581905

theorem polygon_sides (triangles : ℕ) (not_vertex : triangles = 2023) : 
  let sides := triangles + 1 in sides = 2024 :=
by
  sorry

end polygon_sides_l581_581905


namespace rate_of_interest_per_annum_l581_581578

theorem rate_of_interest_per_annum (R : ℝ) : 
  (5000 * R * 2 / 100) + (3000 * R * 4 / 100) = 1540 → 
  R = 7 := 
by {
  sorry
}

end rate_of_interest_per_annum_l581_581578


namespace numPerfectSquaresOrCubesLessThan1000_l581_581774

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def isSixthPower (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

theorem numPerfectSquaresOrCubesLessThan1000 : 
  ∃ n : ℕ, n = 38 ∧ ∀ k < 1000, (isPerfectSquare k ∨ isPerfectCube k → k) ↔ n :=
by
  sorry

end numPerfectSquaresOrCubesLessThan1000_l581_581774


namespace necessary_but_not_sufficient_condition_l581_581808

/-- If x and y are real numbers, x = 0 is a necessary but not sufficient condition 
    for x + yi being a pure imaginary number. -/
theorem necessary_but_not_sufficient_condition (x y : ℝ) :
  (x = 0 → ∃ y, x + y * complex.I ∈ {z : ℂ | z.im = y ∧ z.re = 0}) ∧
  (∀ y, x = 0 → x + y * complex.I ∈ {z : ℂ | z.im = y ∧ z.re = 0} → y ≠ 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l581_581808


namespace remainder_of_special_numbers_l581_581049

noncomputable def num_special_base2_numbers (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).count (λ x, bit0 (x.bitsSum x) < bit0 x.bit_count)

theorem remainder_of_special_numbers (M : ℕ) : (M = num_special_base2_numbers 3000) → M % 1000 = 590 :=
by
  intros hM
  rw hM
  sorry

end remainder_of_special_numbers_l581_581049


namespace restaurant_profit_l581_581371

theorem restaurant_profit (C : ℝ) :
  let profit := 1.6 * C,
      S := C + profit,  -- S = 2.6C
      C_new := 1.12 * C,
      P_new := S - C_new in
  (P_new / S) * 100 ≈ 56.92 :=
by {
  sorry
}

end restaurant_profit_l581_581371


namespace smallest_value_w3_z3_l581_581300

noncomputable section

open Complex

def complex_example (w z : ℂ) (h₁ : |w + z| = 2) (h₂ : |w^2 + z^2| = 16) : ℝ :=
  if h : |w^3 + z^3| = 22 then |w^3 + z^3| else 22

theorem smallest_value_w3_z3 (w z : ℂ) (h₁ : |w + z| = 2) (h₂ : |w^2 + z^2| = 16) : |w^3 + z^3| = 22 :=
by {
  sorry
}

end smallest_value_w3_z3_l581_581300


namespace wolf_notices_red_riding_hood_l581_581870

def wolf_smell_distance_exceeds : Prop :=
  let distance (t: ℕ) := (100 * t^2 - 1280 * t + 6400)
  in ∀ t: ℕ, distance t / t ≥ 48 → (distance t / t) > 45

theorem wolf_notices_red_riding_hood (speed_rh speed_wolf distance_to_path smell_distance: ℝ) :
  speed_rh = 6 ∧ speed_wolf = 8 ∧ distance_to_path = 80 ∧ smell_distance ≤ 45 →
  ¬ wolf_smell_distance_exceeds :=
by sorry

end wolf_notices_red_riding_hood_l581_581870


namespace simplify_and_evaluate_expression_l581_581898

variable (a b : ℝ)

theorem simplify_and_evaluate_expression
  (h1 : a = sqrt 3 + 2)
  (h2 : b = sqrt 3 - 2) :
  ( (a^2 / (a^2 + 2*a*b + b^2) - a / (a+b)) / (a^2 / (a^2 - b^2) - b / (a - b) - 1) ) = 2 * sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_expression_l581_581898


namespace tangent_line_ln_x_xsq_l581_581919

theorem tangent_line_ln_x_xsq (x y : ℝ) (h_curve : y = Real.log x + x^2) (h_point : (x, y) = (1, 1)) :
  3 * x - y - 2 = 0 :=
sorry

end tangent_line_ln_x_xsq_l581_581919


namespace part_i_part_ii_l581_581687

-- Define the conditions for the circle C
def circle_radius : ℝ := 2

def circle_center {x : ℝ} (hx : x > 0) := (x, 0)

def tangent_line (x y : ℝ) := 3 * x - 4 * y + 4 = 0

-- Define the conditions for the line l
def passes_through_Q (x y : ℝ) := y = x - 3

-- Statement for Part (I): Finding the equation of the circle
theorem part_i (x : ℝ) (hx : x > 0) (htangent : ∀ x y : ℝ, tangent_line x y → ((x - x)^2 + y^2 = 4)) :
  ((x - 2) ^ 2 + y ^ 2 = 4) :=
sorry

-- Statement for Part (II): Finding the equation of the line l
theorem part_ii (k : ℝ) (x1 x2 y1 y2 : ℝ) (hx1 : ∃ y, (x1 - 2)^2 + y^2 = 4) 
  (hx2 : ∃ y, (x2 - 2)^2 + y^2 = 4) 
  (hq : y = k * 0 - 3) 
  (hOAOB : x1 * x2 + y1 * y2 = 3) :
  (k > 5/12 → y = x - 3 ∨ y = -5x - 3) :=
sorry


end part_i_part_ii_l581_581687


namespace phase_shift_sin_l581_581658

theorem phase_shift_sin (B C : ℝ) (hB : B = 4) (hC : C = 2 * Real.pi) : C / B = Real.pi / 2 := by
  rw [hB, hC]
  norm_num
  exact Real.pi_div_two


end phase_shift_sin_l581_581658


namespace melindas_math_textbooks_probability_l581_581437

def total_ways_to_arrange_textbooks : ℕ :=
  (Nat.choose 15 4) * (Nat.choose 11 5) * (Nat.choose 6 6)

def favorable_ways (b : ℕ) : ℕ :=
  match b with
  | 4 => (Nat.choose 11 0) * (Nat.choose 11 5) * (Nat.choose 6 6)
  | 5 => (Nat.choose 11 1) * (Nat.choose 10 4) * (Nat.choose 6 6)
  | 6 => (Nat.choose 11 2) * (Nat.choose 9 4) * (Nat.choose 5 5)
  | _ => 0

def total_favorable_ways : ℕ :=
  favorable_ways 4 + favorable_ways 5 + favorable_ways 6

theorem melindas_math_textbooks_probability :
  let m := 1
  let n := 143
  Nat.Gcd m n = 1 ∧ total_ways_to_arrange_textbooks = 1387386 ∧ total_favorable_ways = 9702
  → m + n = 144 := by
sory

end melindas_math_textbooks_probability_l581_581437


namespace apples_left_l581_581074

theorem apples_left (Mike_apples Keith_apples Nancy_eaten : ℝ) (Keith_pears : ℝ) :
    Mike_apples = 7.0 → Keith_apples = 6.0 → Nancy_eaten = 3.0 → Keith_pears = 4.0 → 
    Mike_apples + Keith_apples - Nancy_eaten = 10.0 :=
by
  intros H1 H2 H3 H4
  have H5 : Mike_apples + Keith_apples = 13.0 := by rw [H1, H2]; norm_num
  have H6 : 13.0 - Nancy_eaten = 10.0 := by rw [H3]; norm_num
  rw [H5] at H6
  exact H6

end apples_left_l581_581074


namespace cost_of_notebook_l581_581817

theorem cost_of_notebook (s n c : ℕ) 
    (h1 : s > 18) 
    (h2 : n ≥ 2) 
    (h3 : c > n) 
    (h4 : s * c * n = 2376) : 
    c = 11 := 
  sorry

end cost_of_notebook_l581_581817


namespace pipe_Q_fill_time_l581_581082

theorem pipe_Q_fill_time (x : ℝ) (h1 : 6 > 0)
    (h2 : 24 > 0)
    (h3 : 3.4285714285714284 > 0)
    (h4 : (1 / 6) + (1 / x) + (1 / 24) = 1 / 3.4285714285714284) :
    x = 8 := by
  sorry

end pipe_Q_fill_time_l581_581082


namespace odds_against_event_is_five_l581_581926

-- odds_in_favor, odds_against and probability_occur are given
def odds_in_favor : ℝ := 3
axiom probability_occur : ℝ := 0.375

-- x represents the unknown ratio in the odds
variable (x : ℝ)

-- The theorem states that given the above conditions, x must be equal to 5
theorem odds_against_event_is_five (h : probability_occur = odds_in_favor / (odds_in_favor + x)) : x = 5 :=
  sorry

end odds_against_event_is_five_l581_581926


namespace minimal_area_when_midpoints_l581_581086

variables {n : ℕ} (hn : n > 3) (P : RegularNGon n) 

-- Define an inscribed regular n-gon Q
variables (Q : InscribedRegularNGon P)

-- Define the condition that Q's vertices are at the midpoints of P's sides
def vertices_at_midpoints (Q P) : Prop :=
  ∀ i, Q.vertex i = midpoints (P.side i)

-- Define the areas of P and Q
variables (S_P S_Q : ℝ)

-- State the minimal area condition
theorem minimal_area_when_midpoints (Q P) (S_Q, S_P : ℝ) :
  vertices_at_midpoints Q P →
  S_Q ≤ S_P :=
sorry

end minimal_area_when_midpoints_l581_581086


namespace P_has_common_root_l581_581475

def P (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^2 + p * x + q

theorem P_has_common_root (p q : ℝ) (t : ℝ) (h : P t p q = 0) :
  P 0 p q * P 1 p q = 0 :=
by
  sorry

end P_has_common_root_l581_581475


namespace solve_complex_equation_l581_581000

-- Define the condition: z = i * (2 - z) for complex number z
def satisfies_condition (z : ℂ) : Prop := z = complex.I * (2 - z)

-- The theorem to prove: if z satisfies the condition, then z = 1 + i
theorem solve_complex_equation (z : ℂ) (h : satisfies_condition z) : z = 1 + complex.I :=
by 
  sorry

end solve_complex_equation_l581_581000


namespace factors_of_n_multiples_of_300_l581_581806

noncomputable def n : ℕ := 2^12 * 3^15 * 5^9
def factor_300 : ℕ := 2^2 * 3^1 * 5^2

theorem factors_of_n_multiples_of_300 : 
  ∃ k : ℕ, k = 1320 ∧ (∀ d : ℕ, (d ∣ n → d % factor_300 = 0 → (2^2 ≤ d ∧ d ≤ 2^12) ∧ (3^1 ≤ d ∧ d ≤ 3^15) ∧ (5^2 ≤ d ∧ d ≤ 5^9))) :=
begin
  sorry
end

end factors_of_n_multiples_of_300_l581_581806


namespace factorial_17_digits_sum_l581_581464

theorem factorial_17_digits_sum (X Y Z : ℕ) (hX : X < 10) (hY : Y < 10) (hZ : Z < 10)
  (h_repr : 17.factorial = 355600000 + 10000 * X + 100 * Y + Z) : 
  X + Y + Z = 7 :=
sorry

end factorial_17_digits_sum_l581_581464


namespace line_CD_area_triangle_equality_line_CD_midpoint_l581_581026

theorem line_CD_area_triangle_equality :
  ∃ k : ℝ, 4 * k - 1 = 1 - k := sorry

theorem line_CD_midpoint :
  ∃ k : ℝ, 9 * k - 2 = 1 := sorry

end line_CD_area_triangle_equality_line_CD_midpoint_l581_581026


namespace marked_price_each_article_l581_581956

noncomputable def pair_price : ℝ := 50
noncomputable def discount : ℝ := 0.60
noncomputable def marked_price_pair : ℝ := 50 / 0.40
noncomputable def marked_price_each : ℝ := marked_price_pair / 2

theorem marked_price_each_article : 
  marked_price_each = 62.50 := by
  sorry

end marked_price_each_article_l581_581956


namespace exists_polynomials_and_min_k_l581_581067

theorem exists_polynomials_and_min_k (n : ℕ) (h_even : n % 2 = 0) :
  ∃ (f g : polynomial ℤ) (k : ℕ), 
    k = f * (X + 1) ^ n + g * (X ^ n + 1) ∧ 
    k = 2 ^ (n / 2^nat.find (nat.exists_pow_of_n bigness (n/2^nat.find(nat.exists_embig:get_embig_onbig)))

end exists_polynomials_and_min_k_l581_581067


namespace age_of_new_boy_l581_581564

open Nat

theorem age_of_new_boy : 
  (avg_age_3_years_ago : ℕ) (members_3_years_ago : ℕ) (members_today : ℕ) (avg_age_today : ℕ)
  (h1 : avg_age_3_years_ago = 19)  (h2 : members_3_years_ago = 6)  (h3 : members_today = 7)
  (h4 : avg_age_today = avg_age_3_years_ago) : 
  let total_age_3_yrs_ago := members_3_years_ago * avg_age_3_years_ago
  let total_increase := members_3_years_ago * 3
  let total_age_today := total_age_3_yrs_ago + total_increase
  let total_age_with_boy := members_today * avg_age_today
  total_age_with_boy - total_age_today = 1 := 
by
  sorry

end age_of_new_boy_l581_581564


namespace probability_one_white_one_black_l581_581365

-- Define the basic setup of the problem
def setup : Type :=
  { red : ℕ // red = 1 } ×
  { white : ℕ // white = 2 } ×
  { black : ℕ // black = 3 }

-- Statement of the problem
theorem probability_one_white_one_black (s : setup) :
  let total_balls := 6
      total_combinations := (total_balls.choose 2)
      favourable_combinations := (s.2.1.val * s.2.2.1.val)
  in (favourable_combinations : ℚ) / (total_combinations : ℚ) = 2/5 :=
by
  sorry

end probability_one_white_one_black_l581_581365


namespace find_distance_l581_581066

noncomputable def distance_skew_lines (l m : ℝ) (A B C D E F : ℝ) : Prop :=
  let AB := B - A in
  let BC := C - B in
  let AD := D - A in
  let BE := E - B in
  let CF := F - C in
  AB = BC ∧
  AD = Real.sqrt 15 ∧
  BE = 7 / 2 ∧
  CF = Real.sqrt 10 ∧
  ∃ x : ℝ, x = Real.sqrt 6

theorem find_distance : ∀ (l m A B C D E F : ℝ),
  distance_skew_lines l m A B C D E F →
  ∃ x : ℝ, x = Real.sqrt 6 :=
begin
  intros l m A B C D E F hlm,
  sorry
end

end find_distance_l581_581066


namespace sum_of_areas_eq_2021_l581_581289

noncomputable def f (x y : ℝ) : ℝ :=
  let M := max (floor x) (floor y)
  let m := min (floor x) (floor y)
  sqrt (M * (M + 1)) * (abs (x - m) + abs (y - m))

theorem sum_of_areas_eq_2021 :
  let s := ∑ m in finset.range 2021 \ finset.range 2, (2 / (m * (m + 1)))
  let area := (1 / 2 : ℝ) * ((2 : ℝ) / sqrt (m * (m + 1))) ^ 2
  s = 1010 / 1011 ∧ 1010 + 1011 = 2021 :=
by
  sorry

end sum_of_areas_eq_2021_l581_581289


namespace percentage_increase_of_wattage_l581_581579

theorem percentage_increase_of_wattage (original_wattage new_wattage : ℕ) 
    (h_orig : original_wattage = 80) (h_new : new_wattage = 100) : 
    ((new_wattage - original_wattage) * 100) / original_wattage = 25 := 
by
  -- definition of percentage increase
  have increase : ℕ := new_wattage - original_wattage,
  have percentage_increase : ℕ := (increase * 100) / original_wattage,
  rw [h_orig, h_new] at *,
  simp only [Nat.sub_eq_iff_eq_add, Nat.add_sub_cancel] at *,
  exact sorry

end percentage_increase_of_wattage_l581_581579


namespace tan_ratio_l581_581681

variable (α β : ℝ)

def condition1 : Prop := cos (π / 4 - α) = 3 / 5
def condition2 : Prop := sin (5 * π / 4 + β) = -12 / 13
def condition3 : Prop := α ∈ Ioo (π / 4) (3 * π / 4)
def condition4 : Prop := β ∈ Ioo 0 (π / 4)

theorem tan_ratio (h1 : condition1 α) (h2 : condition2 β) 
                  (h3 : condition3 α) (h4 : condition4 β) :
  (Real.tan α / Real.tan β) = -17 :=
sorry

end tan_ratio_l581_581681


namespace jan_clean_car_water_l581_581846

def jan_water_problem
  (initial_water : ℕ)
  (car_water : ℕ)
  (plant_additional : ℕ)
  (plate_clothes_water : ℕ)
  (remaining_water : ℕ)
  (used_water : ℕ)
  (car_cleaning_water : ℕ) : Prop :=
  initial_water = 65 ∧
  plate_clothes_water = 24 ∧
  plant_additional = 11 ∧
  remaining_water = 2 * plate_clothes_water ∧
  used_water = initial_water - remaining_water ∧
  car_water = used_water + plant_additional ∧
  car_cleaning_water = car_water / 4

theorem jan_clean_car_water : jan_water_problem 65 17 11 24 48 17 7 :=
by {
  sorry
}

end jan_clean_car_water_l581_581846


namespace sum_of_odd_divisors_of_252_l581_581531

theorem sum_of_odd_divisors_of_252 : 
  let prime_factors := (2^2, 3^2, 7)
  let factors := {d : ℕ | d ∣ 252 ∧ d % 2 = 1}
  ∑ d in factors, d = 104 := sorry

end sum_of_odd_divisors_of_252_l581_581531


namespace rotating_right_triangle_results_in_cone_l581_581547

theorem rotating_right_triangle_results_in_cone (T : Triangle) (h : isRightTriangle T) (leg : Side T) :
  ¬(isHypotenuse leg) → 
  resultingShapeFromRotation T leg = Shape.cone :=
by
  sorry

end rotating_right_triangle_results_in_cone_l581_581547


namespace warehouse_rental_comparison_purchase_vs_rent_comparison_l581_581986

-- Define the necessary constants and conditions
def monthly_cost_first : ℕ := 50000
def monthly_cost_second : ℕ := 10000
def moving_cost : ℕ := 70000
def months_in_year : ℕ := 12
def purchase_cost : ℕ := 2000000
def duration_installments : ℕ := 3 * 12 -- 3 years in months
def worst_case_prob : ℕ := 50

-- Question (a)
theorem warehouse_rental_comparison
  (annual_cost_first : ℕ := monthly_cost_first * months_in_year)
  (cost_second_4months : ℕ := monthly_cost_second * 4)
  (cost_switching : ℕ := moving_cost)
  (cost_first_8months : ℕ := monthly_cost_first * 8)
  (worst_case_cost_second : ℕ := cost_second_4months + cost_first_8months + cost_switching) :
  annual_cost_first > worst_case_cost_second :=
by
  sorry

-- Question (b)
theorem purchase_vs_rent_comparison
  (total_rent_cost_4years : ℕ := 4 * annual_cost_first + worst_case_cost_second)
  (total_purchase_cost : ℕ := purchase_cost) :
  total_rent_cost_4years > total_purchase_cost :=
by
  sorry

end warehouse_rental_comparison_purchase_vs_rent_comparison_l581_581986


namespace dice_prime_probability_l581_581612

/-- Prove the probability of exactly three out of six 12-sided dice showing a prime number -/
theorem dice_prime_probability :
  let d12 := finset.range 12 in
  let primes := {n ∈ d12 | nat.prime n} in
  let pPrime := (primes.card : ℚ) / d12.card in
  let pNonPrime := 1 - pPrime in
  (choose 6 3 : ℚ) * (pPrime ^ 3) * (pNonPrime ^ 3) = 857500 / 2985984 :=
by
  sorry

end dice_prime_probability_l581_581612


namespace perfect_squares_and_cubes_count_lt_1000_l581_581788

theorem perfect_squares_and_cubes_count_lt_1000 : 
  let perfect_square := λ n : ℕ, ∃ k : ℕ, n = k^2
  let perfect_cube := λ n : ℕ, ∃ k : ℕ, n = k^3
  let sixth_power := λ n : ℕ, ∃ k : ℕ, n = k^6
  (count (λ n, n < 1000 ∧ (perfect_square n ∨ perfect_cube n)) (Finset.range 1000)) = 38 :=
by {
  sorry
}

end perfect_squares_and_cubes_count_lt_1000_l581_581788


namespace joe_selects_SIMPLER_l581_581849

noncomputable def probability_SIMPLER : ℚ := (3 / 10) * (1 / 3) * (2 / 3)

theorem joe_selects_SIMPLER :
  let THINK := {'T', 'H', 'I', 'N', 'K'} in
  let STREAM := {'S', 'T', 'R', 'E', 'A', 'M'} in
  let PLACES := {'P', 'L', 'A', 'C', 'E', 'S'} in
  let needed_letters := {'S', 'I', 'M', 'P', 'L', 'E', 'R'} in
  (think_subprob : THINK.card = 5) ∧ (stream_subprob : STREAM.card = 6) ∧ (places_subprob : PLACES.card = 6) →
  (p_needed_think : (THINK.choose 3).filter (λ subset, 'I' ∈ subset ∧ 'N' ∈ subset).card / (THINK.choose 3).card = (3 / 10)) →
  (p_needed_stream : (STREAM.choose 5).filter (λ subset, 'S' ∈ subset ∧ 'M' ∈ subset ∧ 'E' ∈ subset ∧ 'R' ∈ subset).card / (STREAM.choose 5).card = (1 / 3)) →
  (p_needed_places : (PLACES.choose 4).filter (λ subset, 'L' ∈ subset).card / (PLACES.choose 4).card = (2 / 3)) →
  probability_SIMPLER = 1 / 15 :=
by
  sorry

end joe_selects_SIMPLER_l581_581849


namespace centers_form_regular_iff_affinely_regular_l581_581443
noncomputable theory

def is_affinely_regular (n : ℕ) (A : ℕ → ℂ) : Prop :=
  ∀ j, A j = A (j - 1) * complex.exp (2 * real.pi * complex.I / n) + A (j + 1) * complex.exp (- 2 * real.pi * complex.I / n)

def centers_form_regular_polygon (n : ℕ) (B : ℕ → ℂ) : Prop :=
  ∀ j, B j = (B (j - 1)) * complex.exp (2 * real.pi * complex.I / n)

theorem centers_form_regular_iff_affinely_regular (n : ℕ) (A : ℕ → ℂ) (B : ℕ → ℂ) :
  (∀ j, B j = (A (j) + A (j + 1)) / 2) ∧ is_convex_n_gon n A ↔ is_affinely_regular n A :=
sorry

end centers_form_regular_iff_affinely_regular_l581_581443


namespace problem1_part1_problem1_part2_problem2_problem3_l581_581258

-- Definition of Convex Function
def isConvex (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  differentiable_on ℝ f (Ioo a b) ∧
  ∀ x ∈ Ioo a b, ∀ y ∈ Ioo a b, x < y → fderiv ℝ f x < fderiv ℝ f y

-- Problem 1: Determine convexity of specific functions
theorem problem1_part1 : ¬ isConvex (λ x : ℝ, x^3) 0 1 := sorry

theorem problem1_part2 : isConvex (λ x : ℝ, log (1 / x)) 0 1 := sorry

-- Problem 2: Weighted sum inequality for convex functions
theorem problem2 (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) (λ : fin n → ℝ) (x : fin n → ℝ) :
  isConvex f a b →
  (∀ i, 0 < λ i) →
  finset.univ.sum λ = 1 →
  (∀ i, x i ∈ Ioo a b) →
  finset.univ.sum (λ i, λ i * f (x i)) ≥ f (finset.univ.sum (λ i, λ i * x i)) := sorry

-- Problem 3: Inequality involving exponents
theorem problem3 (a b c : ℝ) (n : ℕ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : n ≥ b):
  a^n + b^n + c^n ≥ a^(n-5) * b^3 * c^2 + b^(n-5) * c^3 * a^2 + c^(n-5) * a^3 * b^2 := sorry

end problem1_part1_problem1_part2_problem2_problem3_l581_581258


namespace cos_a_correct_sin_beta_correct_l581_581680

noncomputable def cos_a (a : ℝ) (h1 : a ∈ Set.Ioo (π / 2) π) (h2 : (Real.sin (a / 2) + Real.cos (a / 2)) = (2 * Real.sqrt 3) / 3) : Real :=
-((2 * Real.sqrt 2) / 3)

noncomputable def sin_beta (a β : ℝ) (h1 : a ∈ Set.Ioo (π / 2) π) (h2 : Real.sin ((α + β) = -(3 / 5)) (h3 : β ∈ Set.Ioo 0 (π / 2)) : Real :=
(6 * Real.sqrt 2 + 4) / 15

theorem cos_a_correct (a : ℝ) (h1 : a ∈ Set.Ioo (π / 2) π) (h2 : Real.sin (a / 2) + Real.cos (a / 2) = (2 * Real.sqrt 3) / 3) : 
  cos_a a h1 h2 = -(2 * Real.sqrt 2) / 3 :=
sorry

theorem sin_beta_correct (a β : ℝ) (h1 : a ∈ Set.Ioo (π / 2) π) (h2 : Real.sin (α + β) = -(3 / 5)) (h3 : β ∈ Set.Ioo 0 (π / 2) : 
  sin_beta a β h1 h2 h3 = (6 * Real.sqrt 2 + 4) / 15 :=
sorry

end cos_a_correct_sin_beta_correct_l581_581680


namespace part_1_part_2_l581_581313

open Real

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

def sum_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def set_A (a₁ d : ℝ) : set (ℝ × ℝ) :=
  { p | ∃ (n : ℕ), n > 0 ∧ p = (arithmetic_sequence a₁ d n, (sum_sequence a₁ d n) / n) }

def set_B : set (ℝ × ℝ) :=
  { p | ∃ (x y : ℝ), p = (x, y) ∧ (1 / 4) * x^2 - y^2 = 1 }

theorem part_1 (a₁ d : ℝ) (h1 : d ≠ 0) : ∀ (x y : ℝ), (x, y) ∈ set_A a₁ d ∩ set_B → set_A a₁ d ∩ set_B = {(x, y)} :=
sorry

theorem part_2 (a₁ d : ℝ) (h1 : d ≠ 0) (h2 : a₁ ≠ 0) : set_A a₁ d ∩ set_B ≠ ∅ :=
sorry

end part_1_part_2_l581_581313


namespace probability_event_condition_l581_581550

def is_valid_die_num (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

def event_condition (x y : ℕ) : Prop := x + y + x * y = 12 ∧ is_valid_die_num x ∧ is_valid_die_num y

theorem probability_event_condition :
  let outcomes := (finset.range 6).image (λ i, i + 1)
  let valid_pairs := outcomes.product outcomes
  let event := valid_pairs.filter (λ ⟨x, y⟩, event_condition x y)
  event.card / (outcomes.card * outcomes.card : ℚ) = 1/36 :=
begin
  sorry
end

end probability_event_condition_l581_581550


namespace second_pipe_filling_time_l581_581162

noncomputable def fill_time_second_pipe (combined_time : ℝ) : ℝ :=
  let combined_rate := 1 / combined_time
  let first_pipe_rate := 1 / 10
  let third_pipe_rate := -1 / 25
  have H : combined_rate = first_pipe_rate + (1 / x) + third_pipe_rate := sorry
  x

-- State the main problem with conditions and the conclusion
theorem second_pipe_filling_time :
  ∀ (x : ℝ), 
  (first_pipe_rate : ℝ := 1 / 10) →
  (third_pipe_rate : ℝ := -1 / 25) →
  (combined_rate : ℝ := 1 / 6.976744186046512) →
  combined_rate = first_pipe_rate + (1 / x) + third_pipe_rate →
  x ≈ 11.994 :=
by {
  intros,
  sorry,
}

end second_pipe_filling_time_l581_581162


namespace machine_present_value_l581_581497

theorem machine_present_value
  (depreciation_rate : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (dep_years : ℕ)
  (value_after_depreciation : ℝ)
  (present_value : ℝ) :

  depreciation_rate = 0.8 →
  selling_price = 118000.00000000001 →
  profit = 22000 →
  dep_years = 2 →
  value_after_depreciation = (selling_price - profit) →
  value_after_depreciation = 96000.00000000001 →
  present_value * (depreciation_rate ^ dep_years) = value_after_depreciation →
  present_value = 150000.00000000002 :=
by sorry

end machine_present_value_l581_581497


namespace correct_operation_l581_581953

theorem correct_operation :
  (∀ a : ℕ, a ^ 3 * a ^ 2 = a ^ 5) ∧
  (∀ a : ℕ, a + a ^ 2 ≠ a ^ 3) ∧
  (∀ a : ℕ, 6 * a ^ 2 / (2 * a ^ 2) = 3) ∧
  (∀ a : ℕ, (3 * a ^ 2) ^ 3 ≠ 9 * a ^ 6) :=
by
  sorry

end correct_operation_l581_581953


namespace trace_vertices_of_triangle_l581_581223

-- Define a structure representing a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the centroid of a triangle given three points
def centroid (A B C : Point3D) : Point3D :=
  { x := (A.x + B.x + C.x) / 3,
    y := (A.y + B.y + C.y) / 3,
    z := (A.z + B.z + C.z) / 3 }

-- Define the main theorem statement
theorem trace_vertices_of_triangle (A_path : ℝ → Point3D) (B C : Point3D) :
  -- Condition 1: Centroid of triangle is stationary
  let G := centroid (A_path 0) B C in 
  G = centroid (A_path t) B C ∀ t : ℝ →
  -- Condition 2: Vertex A traces a circular path
  (∃ r ω : ℝ, ∀ t : ℝ, (A_path t).x = (A_path 0).x + r * cos (ω * t) ∧ (A_path t).y = (A_path 0).y + r * sin (ω * t)) →
  -- Condition 3: Plane of the triangle remains perpendicular to the plane of the circle traced by A
  --- This needs to be defined more mathematically, but for now assuming it's implied

  -- Conclusion: Paths traced by all vertices are circles
  ∃ r_B r_C ω_B ω_C : ℝ, ∀ t : ℝ, 
    (B.x, B.y, B.z) = ((G.x + r_B * cos (ω_B * t)), (G.y + r_B * sin (ω_B * t)), B.z) ∧
    (C.x, C.y, C.z) = ((G.x + r_C * cos (ω_C * t)), (G.y + r_C * sin (ω_C * t)), C.z) := 
sorry

end trace_vertices_of_triangle_l581_581223


namespace det_le_one_l581_581404

open Matrix

-- Define the problem: Given A is an n x n real matrix, and the condition holds for all positive integers m.

variable {n : Type*} [Fintype n] [DecidableEq n]

def satisfies_condition (A : Matrix n n ℝ) : Prop :=
  ∀ (m : ℕ) (hm : 0 < m), ∃ (B : Matrix n n ℝ), Symmetric B ∧ 2021 • B = A ^ m + B ^ 2

-- Formalize the theorem statement

theorem det_le_one
  (A : Matrix n n ℝ)
  (h : satisfies_condition A) :
  |det A| ≤ 1 :=
sorry

end det_le_one_l581_581404


namespace find_m_f_monotonicity_l581_581721

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 / x - x ^ m

theorem find_m : ∃ (m : ℝ), f 4 m = -7 / 2 := sorry

noncomputable def g (x : ℝ) : ℝ := 2 / x - x

theorem f_monotonicity : ∀ x1 x2 : ℝ, (0 < x2 ∧ x2 < x1) → f x1 1 < f x2 1 := sorry

end find_m_f_monotonicity_l581_581721


namespace total_cantaloupes_l581_581403

theorem total_cantaloupes (Keith_cantaloupes : ℕ) (Fred_cantaloupes : ℕ) (Jason_cantaloupes : ℕ)
  (h1 : Keith_cantaloupes = 29) (h2 : Fred_cantaloupes = 16) (h3 : Jason_cantaloupes = 20) :
  Keith_cantaloupes + Fred_cantaloupes + Jason_cantaloupes = 65 :=
by
  rw [h1, h2, h3]
  exact rfl

end total_cantaloupes_l581_581403


namespace sin_alpha_minus_3pi_l581_581702

theorem sin_alpha_minus_3pi (α : ℝ) (h : Real.sin α = 3/5) : Real.sin (α - 3 * Real.pi) = -3/5 :=
by
  sorry

end sin_alpha_minus_3pi_l581_581702


namespace limit_exists_all_possible_values_of_L_l581_581672

def a_seq (r : ℝ) (n : ℕ) : ℕ :=
  Nat.recOn n 1 (λ n' a_n, ⌊r * a_n⌋)

def L (r : ℝ) : ℝ :=
  lim (λ n, (a_seq r n) / (r ^ n))

theorem limit_exists (r : ℝ) (hr : r > 0) : ∃ L, ∀ ε > 0, ∃ N, ∀ n ≥ N, |a_seq r n / r^n - L| < ε :=
sorry

theorem all_possible_values_of_L : 
  {L | ∃ r > 0, L = lim (λ n, (a_seq r n) / (r ^ n))} = {0} ∪ (1/2, 1] :=
sorry

end limit_exists_all_possible_values_of_L_l581_581672


namespace minimum_a2_plus_4b2_l581_581296

theorem minimum_a2_plus_4b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 1) : 
  a^2 + 4 * b^2 ≥ 32 :=
sorry

end minimum_a2_plus_4b2_l581_581296


namespace stack_trays_height_l581_581587

theorem stack_trays_height
  (thickness : ℕ)
  (top_diameter : ℕ)
  (bottom_diameter : ℕ)
  (decrement_step : ℕ)
  (base_height : ℕ)
  (cond1 : thickness = 2)
  (cond2 : top_diameter = 30)
  (cond3 : bottom_diameter = 8)
  (cond4 : decrement_step = 2)
  (cond5 : base_height = 2) :
  (bottom_diameter + decrement_step * (top_diameter - bottom_diameter) / decrement_step * thickness + base_height) = 26 :=
by
  sorry

end stack_trays_height_l581_581587


namespace jim_gas_gallons_l581_581012

theorem jim_gas_gallons (G : ℕ) (C_NC C_VA : ℕ → ℕ) 
  (h₁ : ∀ G, C_NC G = 2 * G)
  (h₂ : ∀ G, C_VA G = 3 * G)
  (h₃ : C_NC G + C_VA G = 50) :
  G = 10 := 
sorry

end jim_gas_gallons_l581_581012


namespace largest_equal_cost_under_options_l581_581515

def cost_option_1 (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d, d * d).sum

def cost_option_2 (n : ℕ) : ℕ :=
  (n.digits 2).map (λ d, d * d).sum

theorem largest_equal_cost_under_options : ∃ (n : ℕ), n < 5000 ∧ cost_option_1 n = cost_option_2 n ∧ ∀ m : ℕ, m < 5000 → cost_option_1 m = cost_option_2 m → m ≤ n :=
by
  use 3999
  sorry

end largest_equal_cost_under_options_l581_581515


namespace find_C_plus_D_l581_581836

theorem find_C_plus_D
  (C D : ℕ)
  (h1 : D = C + 2)
  (h2 : 2 * C^2 + 5 * C + 3 - (7 * D + 5) = (C + D)^2 + 6 * (C + D) + 8)
  (hC_pos : 0 < C)
  (hD_pos : 0 < D) :
  C + D = 26 := by
  sorry

end find_C_plus_D_l581_581836


namespace darkCubeValidPositions_l581_581838

-- Conditions:
-- 1. The structure is made up of twelve identical cubes.
-- 2. The dark cube must be relocated to a position where the surface area remains unchanged.
-- 3. The cubes must touch each other with their entire faces.
-- 4. The positions of the light cubes cannot be changed.

-- Let's define the structure and the conditions in Lean.

structure Cube :=
  (id : ℕ) -- unique identifier for each cube

structure Position :=
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)

structure Configuration :=
  (cubes : List Cube)
  (positions : Cube → Position)

def initialCondition (config : Configuration) : Prop :=
  config.cubes.length = 12

def surfaceAreaUnchanged (config : Configuration) (darkCube : Cube) (newPos : Position) : Prop :=
  sorry -- This predicate should capture the logic that the surface area remains unchanged

def validPositions (config : Configuration) (darkCube : Cube) : List Position :=
  sorry -- This function should return the list of valid positions for the dark cube

-- Main theorem: The number of valid positions for the dark cube to maintain the surface area.
theorem darkCubeValidPositions (config : Configuration) (darkCube : Cube) :
    initialCondition config →
    (validPositions config darkCube).length = 3 :=
  by
  sorry

end darkCubeValidPositions_l581_581838


namespace cost_of_each_soda_l581_581541

theorem cost_of_each_soda (total_cost sandwiches_cost : ℝ) (number_of_sodas : ℕ)
  (h_total_cost : total_cost = 6.46)
  (h_sandwiches_cost : sandwiches_cost = 2 * 1.49) :
  total_cost - sandwiches_cost = 4 * 0.87 := by
  sorry

end cost_of_each_soda_l581_581541


namespace difference_between_means_is_2700_l581_581463

variable (S : ℕ) (n : ℕ := 500) (actual_highest incorrect_highest : ℕ := 150000, 1500000)

def mean_diff : ℕ :=
  (incorrect_highest - actual_highest) / n

theorem difference_between_means_is_2700 :
  mean_diff S n actual_highest incorrect_highest = 2700 := by
sorry

end difference_between_means_is_2700_l581_581463


namespace number_of_perfect_squares_and_cubes_l581_581759

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end number_of_perfect_squares_and_cubes_l581_581759


namespace find_angle_EAB_l581_581389

-- Definitions based on given conditions
variable (A B C E : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited E]

-- Angles are measured in degrees
variable (angle : Type) [AddGroup angle] [HasScalar ℝ angle]
variable (angle_ABC angle_AEC angle_BAE : angle)

-- Given conditions
axiom angle_BAE_value : (angle_BAE : ℝ) = 25 -- ∠BAE = 25°
axiom angle_AEC_value : (angle_AEC : ℝ) = 65 -- ∠AEC = 65°
axiom E_on_BC : E ∈ LineSegment B C -- E is on segment BC

-- The statement we need to prove
theorem find_angle_EAB : (∠EAB : ℝ) = 65 :=
by
  -- Proof omitted, sorry to skip
  sorry

end find_angle_EAB_l581_581389


namespace g_even_function_l581_581260

def g (x : ℝ) : ℝ := 5 / (3 * x ^ 8 - 7)

theorem g_even_function : ∀ x : ℝ, g (-x) = g x :=
by
  intros x
  unfold g
  -- Lean should simplify g(-x) to g(x)
  simp
  sorry

end g_even_function_l581_581260


namespace GI_div_HJ_eq_three_l581_581030

theorem GI_div_HJ_eq_three
  (ABC_eqt : equilateral_triangle ABC)
  (HDE_on_BC : B ≠ C ∧ D ≠ E ∧ collinear [B, D, C] ∧ collinear [B, E, C])
  (HBC_3_DE : dist B C = 3 * dist D E)
  (DEF_eqt : equilateral_triangle DEF)
  (H_AF : collinear [A, F])
  (DG_parallel_AF : parallel D G A F)
  (EH_parallel_AF : parallel E H A F ∧ collinear [E, H, C])
  (GI_perp_AF : perpendicular GI A F)
  (HJ_perp_AF : perpendicular HJ A F)
  (area_BDF : area (triangle B D F) = 45)
  (area_DEF : area (triangle D E F) = 30) :
  dist G I / dist H J = 3 :=
begin
  sorry
end

end GI_div_HJ_eq_three_l581_581030


namespace non_neg_scalar_product_l581_581684

theorem non_neg_scalar_product (a b c d e f g h : ℝ) : 
  (0 ≤ ac + bd) ∨ (0 ≤ ae + bf) ∨ (0 ≤ ag + bh) ∨ (0 ≤ ce + df) ∨ (0 ≤ cg + dh) ∨ (0 ≤ eg + fh) :=
  sorry

end non_neg_scalar_product_l581_581684


namespace true_converse_l581_581229

-- Definitions of the propositions
def vertical_angles_equal (α β : ℝ) : Prop := α = β
def squares_of_equal_numbers (a b : ℝ) : Prop := a = b → a^2 = b^2
def corresponding_angles_of_congruent_triangles (α β : ℝ) : Prop := α = β
def pythagorean_theorem (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Definitions of the converses
def converse_vertical_angles_equal (α β : ℝ) : Prop := α = β → vertical_angles_equal α β
def converse_squares_of_equal_numbers (a b : ℝ) : Prop := a^2 = b^2 → a = b
def converse_corresponding_angles_of_congruent_triangles (α β : ℝ) : Prop := α = β → corresponding_angles_of_congruent_triangles α β
def converse_pythagorean_theorem (a b c : ℝ) : Prop := a^2 + b^2 = c^2 → pythagorean_theorem a b c

-- The proof problem statement
theorem true_converse :
    converse_pythagorean_theorem ∧
    ¬ converse_vertical_angles_equal ∧
    ¬ converse_squares_of_equal_numbers ∧
    ¬ converse_corresponding_angles_of_congruent_triangles :=
by
  constructor; sorry
  constructor; sorry
  constructor; sorry
  sorry

end true_converse_l581_581229


namespace num_cans_in_pack_l581_581900

theorem num_cans_in_pack (pack_cost individual_can_cost : ℝ) (pack_cost_eq : pack_cost = 2.99) (individual_can_cost_eq : individual_can_cost = 0.25) : 
  floor (pack_cost / individual_can_cost) = 11 :=
by
  rw [pack_cost_eq, individual_can_cost_eq]
  norm_num
  sorry

end num_cans_in_pack_l581_581900


namespace arc_length_correct_l581_581242

open IntervalIntegrable

-- Define the function and the interval
def f (x : ℝ) := (√(1 - x^2)) + Real.arcsin x

-- Define the derivative of the function
def df (x : ℝ) := (1 - x) / √(1 - x^2)

-- Define the interval
def a := (0 : ℝ)
def b := (7 / 9 : ℝ)

-- Define the arc length calculation
def arc_length := ∫ x in a..b, √(1 + (df x)^2)

theorem arc_length_correct :
  arc_length = (2 * √2) / 3 :=
by {
  -- skip the proof for now
  sorry
}

end arc_length_correct_l581_581242


namespace triangle_side_height_inequality_l581_581131

theorem triangle_side_height_inequality (a b h_a h_b S : ℝ) (h1 : a > b) 
  (h2: h_a = 2 * S / a) (h3: h_b = 2 * S / b) :
  a + h_a ≥ b + h_b :=
by sorry

end triangle_side_height_inequality_l581_581131


namespace distance_between_parallel_lines_l581_581507

theorem distance_between_parallel_lines (r : ℝ) (d : ℝ) (h1 : ∃ (C D E F : ℝ), CD = 38 ∧ EF = 38 ∧ DE = 34) :
  d = 6 :=
begin
  sorry
end

end distance_between_parallel_lines_l581_581507


namespace distance_from_origin_to_line_l581_581119

noncomputable def distance_from_point_to_line (A B C x0 y0 : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / (sqrt (A^2 + B^2))

theorem distance_from_origin_to_line : 
  distance_from_point_to_line 1 2 (-5) 0 0 = sqrt 5 :=
by
  sorry

end distance_from_origin_to_line_l581_581119


namespace recliner_gross_revenue_increase_l581_581992

/-
  Prove that the percentage increase of the gross revenue is 36% given
  the conditions below:
  1. The price of recliners is dropped by 20%.
  2. The number of recliners sold increases by 70%.
-/

theorem recliner_gross_revenue_increase
  (P R : ℝ)
  (price_drop_percent : ℝ := 20)
  (sales_increase_percent : ℝ := 70) :
  let original_gross_revenue := P * R,
      new_price := P * (1 - price_drop_percent / 100),
      new_quantity_sold := R * (1 + sales_increase_percent / 100),
      new_gross_revenue := new_price * new_quantity_sold,
      percentage_increase := ((new_gross_revenue - original_gross_revenue) / original_gross_revenue) * 100 in
  percentage_increase = 36 :=
by
  sorry

end recliner_gross_revenue_increase_l581_581992


namespace multiples_3_not_5_l581_581133

theorem multiples_3_not_5 (n : ℕ) : 
  n = 2009 → 
  (finset.filter (λ x, x % 3 = 0 ∧ x % 5 ≠ 0) (finset.range (n + 1))).card = 536 :=
by sorry

end multiples_3_not_5_l581_581133


namespace length_AD_reciprocal_l581_581048

noncomputable def triangle_AB_angle120_bisector_D (A B C D : Type) [metric_space A]
  [metric_space B] [metric_space C] [metric_space D] : Prop :=
∃ (a b c d : ℝ) (α : ℝ), -- α to represent the angle in radians
triangle A B C ∧
α = 2/3 * π ∧ -- 120 degrees in radians is 2/3 * π 
is_angle_bisector A B C D

theorem length_AD_reciprocal (A B C D : Type) [metric_space A]
  [metric_space B] [metric_space C] [metric_space D]
  (h : triangle_AB_angle120_bisector_D A B C D) :
  ∃ (a b c d : ℝ), 
  ∀ (AD AB AC : ℝ), 
  (1 / AD = 1 / AB + 1 / AC) := sorry

end length_AD_reciprocal_l581_581048


namespace triangles_equal_area_opposite_pairs_l581_581583

-- Define the structure of an equilateral triangle
structure EquilateralTriangle (V : Type) :=
(A B C : V)
(equilateral : dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B)

-- Define the point inside the triangle that divides sides in a 2:1 ratio
structure PointInsideTriangle (V : Type) (T : EquilateralTriangle V) :=
(P : V)
(divides_AB : ∃ D : V, on_line_segment T.A T.B D ∧ dist T.A D / dist D T.B = 2)
(divides_BC : ∃ E : V, on_line_segment T.B T.C E ∧ dist T.B E / dist E T.C = 2)
(divides_CA : ∃ F : V, on_line_segment T.C T.A F ∧ dist T.C F / dist F T.A = 2)

-- Prove that the triangles determined by P are equal in area in opposite pairs
theorem triangles_equal_area_opposite_pairs 
  (V : Type) [MetricSpace V] 
  (T : EquilateralTriangle V) 
  (P_in : PointInsideTriangle V T) :
∃ (pairs : list (V × V × V × V)), 
    ∀ (pair : V × V × V × V), pair ∈ pairs → 
    let (A, B, C, D) := pair in 
    (triangle_area A B P_in.P = triangle_area C D P_in.P) ∧ 
    (triangle_area A P_in.P B = triangle_area C P_in.P D) ∧ 
    (triangle_area B P_in.P A = triangle_area D P_in.P C) := 
sorry

end triangles_equal_area_opposite_pairs_l581_581583


namespace find_length_of_AB_l581_581024

theorem find_length_of_AB
  (A B C D P Q : Type)
  [h1 : is_rectangle A B C D]
  [h2 : on_side P C B]
  (BP : ℝ) (CP : ℝ)
  (PQ_AB : ℝ)
  (tan_APD : ℝ) :
  BP = 24 →
  CP = 6 →
  PQ_AB = AB →
  tan_APD = 4 →
  AB = 13.5 := 
by 
  sorry

end find_length_of_AB_l581_581024


namespace fraction_value_l581_581175

theorem fraction_value : (2 * 0.24) / (20 * 2.4) = 0.01 := by
  sorry

end fraction_value_l581_581175


namespace fraction_proof_l581_581952

-- Define the fractions as constants
def a := 1 / 3
def b := 1 / 4
def c := 1 / 2
def d := 1 / 3

-- Prove the main statement
theorem fraction_proof : (a - b) / (c - d) = 1 / 2 := by
  sorry

end fraction_proof_l581_581952


namespace intervals_of_monotonicity_sin_intervals_of_monotonicity_cos_l581_581653

theorem intervals_of_monotonicity_sin:
  (∀ k : ℤ, 
    ∀ x : ℝ,
      ((2 * k * π - π / 2 ≤ x ∧ x ≤ 2 * k * π + π / 2) → 
        (1 + sin x >= 1 + sin (x - 1e-6))) ∧
      ((2 * k * π + π / 2 < x ∧ x ≤ 2 * k * π + 3 * π / 2) → 
        (1 + sin x ≤ 1 + sin (x - 1e-6)))) :=
begin
  sorry
end

theorem intervals_of_monotonicity_cos:
  (∀ k : ℤ,
    ∀ x : ℝ,
      ((2 * k * π - π ≤ x ∧ x < 2 * k * π) → 
        (-cos x <= -cos (x - 1e-6))) ∧
      ((2 * k * π ≤ x ∧ x ≤ 2 * k * π + π) →
        (-cos x >= -cos (x - 1e-6)))) :=
begin
  sorry
end

end intervals_of_monotonicity_sin_intervals_of_monotonicity_cos_l581_581653


namespace part_a_part_b_l581_581980

noncomputable def is_arithmetic_sequence (P : ℕ → polynomial ℤ) (Q : polynomial ℤ) : Prop :=
∀ n, P (n + 1) = P n + Q

variables (P Q : polynomial ℤ)
variables (sequence : ℕ → polynomial ℤ)
variables (h1 : is_arithmetic_sequence sequence Q)
variables (h2 : monic P)
variables (h3 : monic Q)
variables (h4 : ∀ n, (∃ x : ℤ, polynomial.eval x (sequence n) = 0))
variables (h5 : ¬(∃ x : ℤ, polynomial.eval x P = 0 ∧ polynomial.eval x Q = 0))

theorem part_a : P ∣ Q :=
sorry

theorem part_b : (P / Q).degree = 1 :=
sorry

end part_a_part_b_l581_581980


namespace hyperbola_eccentricity_asymptotic_lines_l581_581718

-- Define the conditions and the proof goal:

theorem hyperbola_eccentricity_asymptotic_lines {a b c e : ℝ} 
  (h_asym : ∀ x y : ℝ, (y = x ∨ y = -x) ↔ (a = b)) 
  (h_c : c = Real.sqrt (a ^ 2 + b ^ 2))
  (h_e : e = c / a) : e = Real.sqrt 2 := sorry

end hyperbola_eccentricity_asymptotic_lines_l581_581718


namespace Oates_reunion_l581_581517

-- Declare the conditions as variables
variables (total_guests both_reunions yellow_reunion : ℕ)
variables (H1 : total_guests = 100)
variables (H2 : both_reunions = 7)
variables (H3 : yellow_reunion = 65)

-- The proof problem statement
theorem Oates_reunion (O : ℕ) (H4 : total_guests = O + yellow_reunion - both_reunions) : O = 42 :=
sorry

end Oates_reunion_l581_581517


namespace part_a_part_b_part_c_l581_581191

-- Define the conditions as assumptions.
def num_scientists : ℕ := 18
def initial_knowers : ℕ := 10
def initial_non_knowers : ℕ := num_scientists - initial_knowers

-- a) Probability that the number of scientists who know the news is 13.
theorem part_a : 
  let probability_13 := 0 in
  probability_13 = 0 := by
  sorry

-- b) Probability that the number of scientists who know the news is 14.
theorem part_b : 
  let probability_14 := 20160 / 43758 in
  probability_14 = 0.461 := by
  sorry

-- c) Expected number of scientists who know the news.
theorem part_c : 
  let expected_value := 10 + 8 * (10 / 17 : ℚ) in
  expected_value = 14.7 := by
  sorry

end part_a_part_b_part_c_l581_581191


namespace intersection_points_length_l581_581004

noncomputable def polar_to_rectangular_curve1 (ρ θ : ℝ) : Prop :=
  ρ = 1 → x^2 + y^2 = 1

noncomputable def polar_to_rectangular_curve2 (ρ θ : ℝ) : Prop :=
  ρ = 2 * cos (θ + π / 3) → x^2 + y^2 - x + sqrt 3 * y = 0

theorem intersection_points_length :
  ∀ (A B : ℝ × ℝ),
  (polar_to_rectangular_curve1 1 0) ∧ (polar_to_rectangular_curve2 1 0) →
  (polar_to_rectangular_curve1 0 (π / 3)) ∧ (polar_to_rectangular_curve2 0 (π / 3)) →
  A = (1, 0) ∧ B = (-1 / 2, -sqrt 3 / 2) →
  (dist A B = sqrt 3) :=
by
  sorry

end intersection_points_length_l581_581004


namespace perfect_squares_and_cubes_count_lt_1000_l581_581786

theorem perfect_squares_and_cubes_count_lt_1000 : 
  let perfect_square := λ n : ℕ, ∃ k : ℕ, n = k^2
  let perfect_cube := λ n : ℕ, ∃ k : ℕ, n = k^3
  let sixth_power := λ n : ℕ, ∃ k : ℕ, n = k^6
  (count (λ n, n < 1000 ∧ (perfect_square n ∨ perfect_cube n)) (Finset.range 1000)) = 38 :=
by {
  sorry
}

end perfect_squares_and_cubes_count_lt_1000_l581_581786


namespace count_perfect_squares_cubes_under_1000_l581_581756

-- Condition definitions
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, m * m = n
def isPerfectCube (n : Nat) : Prop := ∃ m : Nat, m * m * m = n

-- Main proof problem
theorem count_perfect_squares_cubes_under_1000 : 
  let count (p : Nat -> Prop) := Finset.card (Finset.filter p (Finset.range 1000))
  count (λ n, isPerfectSquare n ∨ isPerfectCube n) = 37 := 
sorry

end count_perfect_squares_cubes_under_1000_l581_581756


namespace sum_is_18_l581_581380

/-- Define the distinct non-zero digits, Hen, Xin, Chun, satisfying the given equation. -/
theorem sum_is_18 (Hen Xin Chun : ℕ) (h1 : Hen ≠ Xin) (h2 : Xin ≠ Chun) (h3 : Hen ≠ Chun)
  (h4 : 1 ≤ Hen ∧ Hen ≤ 9) (h5 : 1 ≤ Xin ∧ Xin ≤ 9) (h6 : 1 ≤ Chun ∧ Chun ≤ 9) :
  Hen + Xin + Chun = 18 :=
sorry

end sum_is_18_l581_581380


namespace shaded_square_percentage_l581_581544

theorem shaded_square_percentage (total_squares shaded_squares : ℕ) (h_total: total_squares = 25) (h_shaded: shaded_squares = 13) : 
(shaded_squares * 100) / total_squares = 52 := 
by
  sorry

end shaded_square_percentage_l581_581544


namespace greatest_divisor_of_620_and_180_l581_581168

/-- This theorem asserts that the greatest divisor of 620 that 
    is smaller than 100 and also a factor of 180 is 20. -/
theorem greatest_divisor_of_620_and_180 (d : ℕ) (h1 : d ∣ 620) (h2 : d ∣ 180) (h3 : d < 100) : d ≤ 20 :=
by
  sorry

end greatest_divisor_of_620_and_180_l581_581168


namespace find_XY_squared_l581_581046

-- Definitions drawn from conditions
variables (A B C T X Y : Type) [metric_space T]
variables [normed_group B] [normed_group C]
variables [normed_group X] [normed_group Y]
variables (ω : Type) [has_circumcircle ABC ω]

-- Projections and distances given
variables (BT CT : ℝ) (BC : ℝ)
variables (TX TY XY : ℝ)
variables (AB AC : line_segment A := B == line A B) (B == C)

-- Given conditions
variables (H1 : BT = 16) (H2 : CT = 16) (H3 : BC = 22) 
variables (H4 : TX ^ 2 + TY ^ 2 + XY ^ 2 = 1143)

-- Theorem to prove XY^2 = 717
theorem find_XY_squared : XY ^ 2 = 717 :=
sorry

end find_XY_squared_l581_581046


namespace find_p_l581_581332

noncomputable def parabola_focus (p : ℝ) (hp : 0 < p) : ℝ × ℝ := (p / 2, 0)

noncomputable def line_through_focus (m p y : ℝ) : ℝ := m * y + p / 2

axiom line_intersects_parabola (p m y1 y2 x1 x2 : ℝ) (h : 0 < p) :
  y1 ^ 2 = 2 * p * x1 ∧ y2 ^ 2 = 2 * p * x2 ∧
  x1 = (m * y1 + p / 2) ∧ x2 = (m * y2 + p / 2) ∧
  (y1 + y2) / 2 = sqrt 2 ∧
  abs (x2 - x1) = 5 * sqrt 2 → y1 ≠ y2

theorem find_p (p : ℝ) (hp : 0 < p) :
  (∃ (m y1 y2 x1 x2 : ℝ),
    y1 ^ 2 = 2 * p * x1 ∧ y2 ^ 2 = 2 * p * x2 ∧
    x1 = m * y1 + p / 2 ∧ x2 = m * y2 + p / 2 ∧
    (y1 + y2) / 2 = sqrt 2 ∧
    abs (x2 - x1) = 5 * sqrt 2) →
  (p = 2 * sqrt 2 ∨ p = sqrt 2 / 2) :=
by sorry

end find_p_l581_581332


namespace geometric_sequence_value_of_m_l581_581288

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_value_of_m (r : ℝ) (hr : r ≠ 1) 
    (h1 : is_geometric_sequence a r)
    (h2 : a 5 * a 6 + a 4 * a 7 = 18) 
    (h3 : a 1 * a m = 9) :
  m = 10 :=
by
  sorry

end geometric_sequence_value_of_m_l581_581288


namespace max_value_of_y_l581_581800

theorem max_value_of_y (x : ℝ) (h : 0 < x ∧ x < 1 / 2) : (∃ y, y = x^2 * (1 - 2*x) ∧ y ≤ 1 / 27) :=
sorry

end max_value_of_y_l581_581800


namespace number_of_blocks_differing_in_two_ways_l581_581194

-- Definitions based on conditions
inductive Material
| plastic
| wood

inductive Size
| small
| medium
| large

inductive Color
| blue
| green
| red
| yellow

inductive Shape
| circle
| hexagon
| square
| triangle

-- The proof statement
theorem number_of_blocks_differing_in_two_ways :
  let s := 96
  let material_choices := 2
  let size_choices := 3
  let color_choices := 4
  let shape_choices := 4
  let differing_in_two_ways := 
    (1 + x) * (1 + 2 * x) * (1 + 3 * x) ^ 2
  in coefficient_of_x_2 differing_in_two_ways = 29 :=
by
  sorry

end number_of_blocks_differing_in_two_ways_l581_581194


namespace ka_eq_nc_l581_581405

theorem ka_eq_nc
  (A B C Z Y M N L K : Point)
  (circ : Triangle ABC → Circle)
  (incircle_tangent_1 : Tangent (circ (Triangle.mk A B C)) Z A B)
  (incircle_tangent_2 : Tangent (circ (Triangle.mk A B C)) Y C A)
  (YZ_parallel_MN : Parallel (Line.mk Z Y) (Line.mk M N))
  (M_midpoint_BC : Midpoint M B C)
  (N_CA : Line.mk N A = Line.mk N C)
  (NL_eq_AB : Length (Segment.mk N L) = Length (Segment.mk A B))
  (L_on_CA_same_side : OnSameSide L N A)
  (ML_inter_AB_K : intersects (Line.mk M L) (Line.mk A B) K) :
  Length (Segment.mk K A) = Length (Segment.mk N C) :=
by
  sorry

end ka_eq_nc_l581_581405


namespace count_perfect_squares_and_cubes_l581_581739

theorem count_perfect_squares_and_cubes : 
  let num_squares := 31 in
  let num_cubes := 9 in
  let num_sixth_powers := 3 in
  num_squares + num_cubes - num_sixth_powers = 37 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581739


namespace area_BOC_l581_581728

open Real

noncomputable def point (x y : ℝ) : Type := ℝ × ℝ

def line_parametric (t : ℝ) : point ℝ :=
  let angle := real.sin (π / 6) in
  (2 - t * angle, -1 + t * angle)

def circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 8

def intersection (p : point ℝ) : Prop :=
  ∃ t : ℝ, p = line_parametric t ∧ circle p.1 p.2

def points (B C : point ℝ) : Prop :=
  intersection B ∧ intersection C

def area_triangle (A B C : point ℝ) : ℝ :=
  let (x1, y1) := B in
  let (x2, y2) := C in
  abs (0.5 * (x1 * y2 - x2 * y1))

theorem area_BOC {B C : point ℝ} (hB : intersection B) (hC : intersection C) :
  area_triangle (0, 0) B C = sqrt 15 / 2 :=
sorry

end area_BOC_l581_581728


namespace determine_abc_l581_581633

def f (a b c : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then a*x + 2
  else if x = 0 then 4*b
  else 2*b*x + c

theorem determine_abc :
  ∃ (a b c : ℕ), 
    f a b c 1 = 3 ∧ 
    f a b c 0 = 8 ∧ 
    f a b c (-1) = -4 := 
    by
      use 1, 2, 0
      simp [f]
      split
      · intros
        simp
        linarith
      split
      · intros
        simp
        linarith
      sorry

end determine_abc_l581_581633


namespace total_cups_sold_is_46_l581_581894

-- Define the number of cups sold last week
def cups_sold_last_week : ℕ := 20

-- Define the percentage increase
def percentage_increase : ℕ := 30

-- Calculate the number of cups sold this week
def cups_sold_this_week : ℕ := cups_sold_last_week + (cups_sold_last_week * percentage_increase / 100)

-- Calculate the total number of cups sold over both weeks
def total_cups_sold : ℕ := cups_sold_last_week + cups_sold_this_week

-- State the theorem to prove the total number of cups sold
theorem total_cups_sold_is_46 : total_cups_sold = 46 := sorry

end total_cups_sold_is_46_l581_581894


namespace min_pieces_pie_l581_581577

theorem min_pieces_pie (p q : ℕ) (h_coprime : Nat.gcd p q = 1) : 
  ∃ n : ℕ, n = p + q - 1 ∧ 
    (∀ m, m < n → ¬ (∀ k : ℕ, (k < p → n % p = 0) ∧ (k < q → n % q = 0))) :=
sorry

end min_pieces_pie_l581_581577


namespace distance_skew_lines_l581_581386

def point (ℝ : Type) := ℝ × ℝ × ℝ

variables (A B C D : point ℝ)
variables (AD BD CD : ℝ)
variables (angle_ADB angle_BDC angle_ADC : ℝ)

-- Given the conditions
def conditions : Prop :=
  -- Triangle ADB is isosceles right triangle
  (AD = 1) ∧
  (angle_ADB = 90) ∧
  (angle_BDC = 60) ∧
  (angle_ADC = 60)

-- Distance between skew lines AB and CD
noncomputable def distance_between_skew_AB_CD : ℝ :=
  1 / 2

-- Prove the distance between skew lines AB and CD is 1/2
theorem distance_skew_lines (h : conditions AD angle_ADB angle_BDC angle_ADC) :
  distance_between_skew_AB_CD = 1 / 2 :=
sorry

end distance_skew_lines_l581_581386


namespace resulting_shape_is_cone_l581_581548

-- Assume we have a right triangle
structure right_triangle (α β γ : ℝ) : Prop :=
  (is_right : γ = π / 2)
  (sum_of_angles : α + β + γ = π)
  (acute_angles : α < π / 2 ∧ β < π / 2)

-- Assume we are rotating around one of the legs
def rotate_around_leg (α β : ℝ) : Prop := sorry

theorem resulting_shape_is_cone (α β γ : ℝ) (h : right_triangle α β γ) :
  ∃ (shape : Type), rotate_around_leg α β → shape = cone :=
by
  sorry

end resulting_shape_is_cone_l581_581548


namespace max_area_rectangular_playground_l581_581513

theorem max_area_rectangular_playground (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 360) 
  (h_length : l ≥ 90) 
  (h_width : w ≥ 50) : 
  (l * w) ≤ 8100 :=
by
  sorry

end max_area_rectangular_playground_l581_581513


namespace count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581791

theorem count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes :
  let num_perfect_squares := 31
  let num_perfect_cubes := 9
  let num_perfect_sixth_powers := 2
  let num_either_squares_or_cubes := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers
  in num_either_squares_or_cubes = 38 :=
by
  sorry

end count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581791


namespace perpendicular_centers_l581_581865

variables {A B C D O1 O2 O3 O4 : Type} 
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [metric_space O1] [metric_space O2] [metric_space O3] [metric_space O4]

def is_convex (quad : quadrilateral A B C D) : Prop :=
  convex A B C quad ∧ convex B C D quad ∧ convex C D A quad ∧ convex D A B quad 

def equilateral_triangle_center (A B : Point) : Point := sorry
def midpoint (A B : Point) : Point := sorry

theorem perpendicular_centers 
  (ABCD : quadrilateral A B C D)
  (h1 : is_convex ABCD)
  (h2 : dist A C = dist B D)
  (O1 := equilateral_triangle_center A B)
  (O2 := equilateral_triangle_center B C)
  (O3 := equilateral_triangle_center C D)
  (O4 := equilateral_triangle_center D A) :
  is_perpendicular (line_through O1 O3) (line_through O2 O4) :=
sorry

end perpendicular_centers_l581_581865


namespace a_n_formula_l581_581031

noncomputable def a : ℕ+ → ℕ
| ⟨1, _⟩ => 1
| ⟨n+1, h⟩ => 2^n * a ⟨n, Nat.succ_pos n⟩

theorem a_n_formula (n : ℕ+) : a n = 2^((n - 1) * n / 2) := sorry

end a_n_formula_l581_581031


namespace no_distinct_natural_numbers_eq_sum_and_cubes_eq_l581_581457

theorem no_distinct_natural_numbers_eq_sum_and_cubes_eq:
  ∀ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
  → a^3 + b^3 = c^3 + d^3
  → a + b = c + d
  → false := 
by
  intros
  sorry

end no_distinct_natural_numbers_eq_sum_and_cubes_eq_l581_581457


namespace problem_l581_581053

noncomputable def f : ℕ → ℕ := sorry

axiom h : ∀ a b : ℕ, 2 * f (a^2 + b^2) = (f a)^2 + (f b)^2

theorem problem : let possible_values := {f 49 | f 49 = 0 ∨ f 49 = 1 ∨ f 49 = 16}
in
  let m := possible_values.to_finset.card
  let t := possible_values.sum (λ x, x)
  in m * t = 51 :=
by
  sorry

end problem_l581_581053


namespace eq_line_QF2_min_lambda_n_l581_581693

-- Ellipse equation and conditions
def ellipse_eq (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def passes_through (a b : ℝ) (D : ℝ × ℝ) : Prop :=
  ellipse_eq a b (D.1) (D.2)

def equation_of_ellipse (a b : ℝ) (x y: ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

-- Eccentricity condition
def eccentricity (a e : ℝ) : Prop :=
  e = (Real.sqrt 2) / 2

-- Line equation with slope constraint
def line_eq (m x y : ℝ) : Prop :=
  x ± Real.sqrt (Real.sqrt 2) * y - 1 = 0

-- Proof placeholders
theorem eq_line_QF2 (a b : ℝ) (e : ℝ) (D : ℝ × ℝ) (m : ℝ) :
  passes_through a b D →
  eccentricity a e →
  ellipse_eq a b (D.1) (D.2) →
  line_eq m (D.1) (D.2) :=
  by sorry

theorem min_lambda_n (a b m n x1 y1 : ℝ) :
  equation_of_ellipse a b x1 y1 →
  ∃ (λ n : ℝ), λ * n = 1 / 9 :=
  by sorry

end eq_line_QF2_min_lambda_n_l581_581693


namespace evaluate_complex_fraction_l581_581619

theorem evaluate_complex_fraction : 
  (1 / (2 + (1 / (3 + 1 / 4)))) = (13 / 30) :=
by
  sorry

end evaluate_complex_fraction_l581_581619


namespace find_angle_C_in_triangle_l581_581392

theorem find_angle_C_in_triangle
  (A B C : ℝ)
  (angle_A : A = 45)
  (AB : ℝ)
  (BC : ℝ)
  (h_AB : AB = real.sqrt 2)
  (h_BC : BC = 2) :
  A = 45 → AB = real.sqrt 2 → BC = 2 → C = 30 :=
by
  intros h1 h2 h3
  sorry

end find_angle_C_in_triangle_l581_581392


namespace numPerfectSquaresOrCubesLessThan1000_l581_581771

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def isSixthPower (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

theorem numPerfectSquaresOrCubesLessThan1000 : 
  ∃ n : ℕ, n = 38 ∧ ∀ k < 1000, (isPerfectSquare k ∨ isPerfectCube k → k) ↔ n :=
by
  sorry

end numPerfectSquaresOrCubesLessThan1000_l581_581771


namespace configuration_circles_points_exists_l581_581440

/-- There exists a configuration on a plane with several circles and marked points such that
    each circle passes through exactly five marked points,
    and each marked point lies on exactly five circles. -/
theorem configuration_circles_points_exists :
  ∃ (circles : Set (Set Point)) (points : Set Point),
    (∀ c ∈ circles, ∃! ps ⊆ points, ps.card = 5 ∧ ∀ p ∈ ps, p ∈ c) ∧
    (∀ p ∈ points, ∃! cs ⊆ circles, cs.card = 5 ∧ ∀ c ∈ cs, p ∈ c) :=
sorry

end configuration_circles_points_exists_l581_581440


namespace sufficient_but_not_necessary_condition_l581_581315

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a > 2) : (a^2 > 2 * a) ∧ (∀ a ∈ ℝ, a^2 > 2 * a → (a > 2 ∨ a < 0)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l581_581315


namespace sin_angle_PQS_l581_581025

variable (P Q R S : Type)
variable [innerProductSpace ℝ (P : Type)] [innerProductSpace ℝ (Q : Type)] 
variable [innerProductSpace ℝ (R : Type)] [innerProductSpace ℝ (S : Type)]
variable (a b : ℝ)

-- Given conditions
variable (anglePRS anglePSQ angleQSR : ℝ)
variable (h1 : anglePRS = 90)
variable (h2 : anglePSQ = 90)
variable (h3 : angleQSR = 90)
variable (h4 : a = Real.cos anglePSQ)
variable (h5 : b = Real.cos angleQSR)

-- Proof statement
theorem sin_angle_PQS (h1 : anglePRS = 90) (h2 : anglePSQ = 90) (h3 : angleQSR = 90)
  (h4 : a = Real.cos anglePSQ) (h5 : b = Real.cos angleQSR) : 
  Real.sin anglePSQ = Real.sqrt (1 - a^2) :=
sorry

end sin_angle_PQS_l581_581025


namespace partition_administrators_l581_581379

-- Define administrators and reporting relation as a type
variable {Admin : Type} (A B C : Admin)

-- Define the reporting relation
variable (reports_to : Admin → Admin → Prop)

-- Condition 1: If A reports to B and B reports to C, then C reports to A
axiom transitive_reporting : ∀ (A B C : Admin), reports_to A B → reports_to B C → reports_to C A 

-- Condition 2: No administrator reports to themselves
axiom no_self_reporting : ∀ (A : Admin), ¬ reports_to A A

-- Theorem: Existence of partition into three disjoint sets X, Y, Z with the required properties
theorem partition_administrators (H : ∀ (A B : Admin), reports_to A B → 
            (∃ (X Y Z : set Admin), (A ∈ X ∧ B ∈ Y) ∨ (A ∈ Y ∧ B ∈ Z) ∨ (A ∈ Z ∧ B ∈ X)) 
          ) : 
    ∃ (X Y Z : set Admin), ∀ A B, reports_to A B → 
                           (A ∈ X ∧ B ∈ Y) ∨ (A ∈ Y ∧ B ∈ Z) ∨ (A ∈ Z ∧ B ∈ X) :=
sorry

end partition_administrators_l581_581379


namespace inequality_solution_exists_l581_581648

theorem inequality_solution_exists (x : ℝ) :
  x ∈ set.Ioo (neg_infty : ℝ) (-2 : ℝ) ∪ set.Ioo (0 : ℝ) (1 : ℝ) ∪ set.Ioo (1 : ℝ) (pos_infty : ℝ) →
  ∃ a ∈ set.Icc (-1 : ℝ) (2 : ℝ), 
  (2 - a) * x^3 + (1 - 2 * a) * x^2 - 6 * x + 5 + 4 * a - a^2 < 0 :=
by 
  intro h
  -- Proof omitted
  sorry

end inequality_solution_exists_l581_581648


namespace sequence_general_term_l581_581385

def recurrence_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n / (1 + a n)

theorem sequence_general_term :
  ∀ a : ℕ → ℚ, recurrence_sequence a → ∀ n : ℕ, n ≥ 1 → a n = 2 / (2 * n - 1) :=
by
  intro a h n hn
  sorry

end sequence_general_term_l581_581385


namespace aluminum_carbonate_weight_l581_581171

-- Define the atomic weights
def Al : ℝ := 26.98
def C : ℝ := 12.01
def O : ℝ := 16.00

-- Define the molecular weight of aluminum carbonate
def molecularWeightAl2CO3 : ℝ := (2 * Al) + (3 * C) + (9 * O)

-- Define the number of moles
def moles : ℝ := 5

-- Calculate the total weight of 5 moles of aluminum carbonate
def totalWeight : ℝ := moles * molecularWeightAl2CO3

-- Statement to prove
theorem aluminum_carbonate_weight : totalWeight = 1169.95 :=
by {
  sorry
}

end aluminum_carbonate_weight_l581_581171


namespace concurrent_segments_unique_solution_l581_581852

theorem concurrent_segments_unique_solution (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  4^c - 1 = (2^a - 1) * (2^b - 1) ↔ (a = 1 ∧ b = 2 * c) ∨ (a = 2 * c ∧ b = 1) :=
by
  sorry

end concurrent_segments_unique_solution_l581_581852


namespace tiffany_lives_problem_l581_581514

/-- Tiffany's lives problem -/
theorem tiffany_lives_problem (L : ℤ) (h1 : 43 - L + 27 = 56) : L = 14 :=
by {
  sorry
}

end tiffany_lives_problem_l581_581514


namespace count_perfect_squares_and_cubes_l581_581747

theorem count_perfect_squares_and_cubes (n : ℕ) (h : n = 1000) : 
  let squares := {x : ℕ | (x ^ 2) < n}
  let cubes := {x : ℕ | (x ^ 3) < n}
  let sixth_powers := {x : ℕ | (x ^ 6) < n}
  (squares.card + cubes.card - sixth_powers.card) = 39 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581747


namespace particle_midpoint_area_ratio_l581_581941

-- Definitions of points A, B, and C
variables (A B C : ℝ × ℝ)
-- Conditions on points A, B, and C forming an equilateral triangle
axiom eq_triangle : A = (0, 0) ∧ B = (1, 0) ∧ C = (1/2, Math.sqrt 3 / 2)

-- Starting points of particles
variables (startA startC : ℝ × ℝ)
-- Conditions on starting points
axiom start_points : startA = A ∧ startC = C

-- Speed of particles where particle at A moves at twice the speed of particle at C
variables (speedA speedC : ℝ)
axiom speed_ratio : speedA = 2 * speedC

-- Function defining the midpoint of the line segment joining two particles
def midpoint (t : ℝ) : ℝ × ℝ := 
  let P1 := ((t / speedA, 0) : ℝ × ℝ) in
  let P2 := ((1 / 2 + t / (2 * speedC), Math.sqrt 3 / 2 - (Math.sqrt 3) * t / (2 * speedC)) : ℝ × ℝ) in
  ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

-- The main goal to prove
theorem particle_midpoint_area_ratio :
  let area_ratio : ℝ := 1 / 16
  let enclosed_area := (midpoint 1).1 * (midpoint 2).2 - (midpoint 2).1 * (midpoint 1).2 / 2
  enclosed_area / (Math.sqrt 3 / 4) = area_ratio := sorry

end particle_midpoint_area_ratio_l581_581941


namespace polar_eq_cartesian_l581_581335

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

theorem polar_eq_cartesian {ρ θ : ℝ} (h : ρ = 6 * cos θ) :
  let (x, y) := polar_to_cartesian ρ θ in x^2 + y^2 = 6 * x := by
  sorry

end polar_eq_cartesian_l581_581335


namespace find_a100_l581_581323

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a n = a 0 + n * d

noncomputable def a : ℕ → ℤ := sorry

lemma sum_nine_terms_eq : (∑ i in finset.range 9, a i) = 27 := sorry
lemma term_ten_eq : a 9 = 8 := sorry

theorem find_a100 : a 99 = 98 := by
  have hs : (∑ i in finset.range 9, a i) = 27 := sum_nine_terms_eq
  have ha : a 9 = 8 := term_ten_eq
  sorry

end find_a100_l581_581323


namespace area_ratio_l581_581027

theorem area_ratio (s : ℝ) (A B C D M N O : ℝ × ℝ) (h_square : A = (0, 0) ∧ B = (s, 0) ∧ C = (s, s) ∧ D = (0, s))
  (h_midpoints : M = (s / 2, 0) ∧ N = (s, s / 2))
  (h_intersection : ∃ x y, AN eq line_eq AN, CM eq line_eq CM) :
  let AO := ∃ ox oy : ℝ,
    ((ox = 2 * s / 3) ∧ (oy = s / 3) ∧ from intersection_points AN CM ANeq CMeq) in
   ((area AOCD AO C D) / (area ABCD A B C D)) = (2 / 3) :=
by
  sorry

end area_ratio_l581_581027


namespace smallest_sum_of_consecutive_primes_divisible_by_5_l581_581666

-- Define a predicate for consecutive prime numbers
def is_consecutive_primes (a b c d : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d ∧
  (b = a + 1 ∨ b = a + 2) ∧
  (c = b + 1 ∨ c = b + 2) ∧
  (d = c + 1 ∨ d = c + 2)

-- Define the main problem statement
theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ (a b c d : ℕ), is_consecutive_primes a b c d ∧ (a + b + c + d) % 5 = 0 ∧ ∀ (w x y z : ℕ), is_consecutive_primes w x y z ∧ (w + x + y + z) % 5 = 0 → a + b + c + d ≤ w + x + y + z :=
sorry

end smallest_sum_of_consecutive_primes_divisible_by_5_l581_581666


namespace ratio_q_t_l581_581604

variable (t q : ℝ)
variable (hc1 : ∀ t q, t = sqrt 3 / 2)
variable (hc2 : ∀ t q, q = 3 - sqrt 3)

theorem ratio_q_t (t q : ℝ) (hc1 : t = sqrt 3 / 2) (hc2 : q = 3 - sqrt 3) : q / t = 2 * sqrt 3 - 2 := 
  sorry

end ratio_q_t_l581_581604


namespace count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581795

theorem count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes :
  let num_perfect_squares := 31
  let num_perfect_cubes := 9
  let num_perfect_sixth_powers := 2
  let num_either_squares_or_cubes := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers
  in num_either_squares_or_cubes = 38 :=
by
  sorry

end count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581795


namespace find_r_plus_s_l581_581925

theorem find_r_plus_s :
  ∃ (r s : ℝ), 
  (∃ Q P : ℝ × ℝ, 
     Q = (0, 6) ∧ 
     P = (12, 0) ∧ 
     s ∈ {s | s = 1.5} ∧ 
     r ∈ {r | r ∈ ℝ ∧ 1.5 = -1/2 * r + 6}) ∧
  r + s = 10.5 :=
by sorry

end find_r_plus_s_l581_581925


namespace maximize_binomial_pmf_l581_581678

open scoped BigOperators

def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem maximize_binomial_pmf :
  ∃ k : ℕ, ∀ (p : ℝ), p = 1/2 → k = 10 → binomial_pmf 20 k p = binomial_pmf 20 10 (1 / 2) := 
sorry

end maximize_binomial_pmf_l581_581678


namespace simplify_S_n_l581_581867

variables (a : Nat → ℤ) (d : ℤ) (n : Nat)
variables (x : ℤ)
variable (h_arith_seq : ∀ k : Nat, k ≤ n → a (k + 1) = a k + d)
variable (h_distinct : ∀ i j : Nat, i ≠ j → a i ≠ a j)

noncomputable def S_n : ℤ := 
  ∑ k in Finset.range n, x ^ (a (k + 1)^2 - a k^2)

theorem simplify_S_n (h_x_ne_one : x ≠ 1) :
  S_n a d n x = (x ^ (2 * a 1 * d + 2 * n * d^2) - x ^ (2 * a 1 * d + d^2)) / (x ^ (2 * d^2) - 1) := by 
  sorry

end simplify_S_n_l581_581867


namespace solution_set_of_inequality_l581_581489

theorem solution_set_of_inequality (x : ℝ) (h : 2 * x + 3 ≤ 1) : x ≤ -1 :=
sorry

end solution_set_of_inequality_l581_581489


namespace warehouse_rental_comparison_purchase_vs_rent_comparison_l581_581987

-- Define the necessary constants and conditions
def monthly_cost_first : ℕ := 50000
def monthly_cost_second : ℕ := 10000
def moving_cost : ℕ := 70000
def months_in_year : ℕ := 12
def purchase_cost : ℕ := 2000000
def duration_installments : ℕ := 3 * 12 -- 3 years in months
def worst_case_prob : ℕ := 50

-- Question (a)
theorem warehouse_rental_comparison
  (annual_cost_first : ℕ := monthly_cost_first * months_in_year)
  (cost_second_4months : ℕ := monthly_cost_second * 4)
  (cost_switching : ℕ := moving_cost)
  (cost_first_8months : ℕ := monthly_cost_first * 8)
  (worst_case_cost_second : ℕ := cost_second_4months + cost_first_8months + cost_switching) :
  annual_cost_first > worst_case_cost_second :=
by
  sorry

-- Question (b)
theorem purchase_vs_rent_comparison
  (total_rent_cost_4years : ℕ := 4 * annual_cost_first + worst_case_cost_second)
  (total_purchase_cost : ℕ := purchase_cost) :
  total_rent_cost_4years > total_purchase_cost :=
by
  sorry

end warehouse_rental_comparison_purchase_vs_rent_comparison_l581_581987


namespace range_of_a_l581_581703

noncomputable def e : ℝ := real.exp 1

theorem range_of_a :
  (∀ x₁ ∈ set.Icc (0:ℝ) 1, ∃! x₂ ∈ set.Icc (-1:ℝ) 1, x₁ + x₂^2 * exp x₂ = a) →
  a ∈ set.Ioc (1 + 1/e:ℝ) e :=
begin
  sorry
end

end range_of_a_l581_581703


namespace sum_possible_values_n_l581_581939

theorem sum_possible_values_n (n : ℕ) (h : 0 < n)
    (h_meeting_point : (∀ m : ℕ, m = 25 ∨ m = 3 → ∃ k : ℕ, m % n = k % n)) :
    ∑ m in {1, 2, 4, 7, 11, 14, 22, 28}.to_finset, m = 89 := sorry

end sum_possible_values_n_l581_581939


namespace taehyung_walks_more_than_minyoung_l581_581101

def taehyung_distance_per_minute : ℕ := 114
def minyoung_distance_per_minute : ℕ := 79
def minutes_per_hour : ℕ := 60

theorem taehyung_walks_more_than_minyoung :
  (taehyung_distance_per_minute * minutes_per_hour) -
  (minyoung_distance_per_minute * minutes_per_hour) = 2100 := by
  sorry

end taehyung_walks_more_than_minyoung_l581_581101


namespace polygon_interior_angle_sum_l581_581115

theorem polygon_interior_angle_sum (n : ℕ) (h : 180 * (n - 2) = 2340) :
  180 * (n - 2 + 3) = 2880 := by
  sorry

end polygon_interior_angle_sum_l581_581115


namespace series_convergence_l581_581190

open_locale big_operators

theorem series_convergence (a : ℕ → ℝ) (h₀ : ∀ n, a n ≥ 0)
  (h₁ : ∀ n ≥ 1, (1 / n) * (∑ k in finset.range n, a (k + 1)) ≥ ∑ k in finset.range n, a (n + k + 1)) :
  ∃ l ≤ 2 * real.exp (1:ℝ) * a 1, ∑' n, a n = l :=
by
  sorry

end series_convergence_l581_581190


namespace at_least_one_angle_le_30_l581_581862
open Real

variables (A B C M : ℝ × ℝ)

def angle (P Q R : ℝ × ℝ) : ℝ := 
  let m := - (((Q.2 - R.2) / (Q.1 - R.1)))
  let n := (((P.2 - Q.2) / (P.1 - Q.1)))
  atan ((m - n) / (1 + (m * n)))

-- Given M inside triangle ABC
axiom M_inside_ABC : is_in_triangle M A B C

theorem at_least_one_angle_le_30 (hM : M_inside_ABC):
  (∃ θ₁ ∈ { angle M A B, angle M B C, angle M C A }, θ₁ ≤ 30) ∧ 
  (∃ θ₂ ∈ { angle M A C, angle M C B, angle M B A }, θ₂ ≤ 30) :=
sorry

end at_least_one_angle_le_30_l581_581862


namespace k_range_l581_581725

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  (Real.log x) - x - x * Real.exp (-x) - k

theorem k_range (k : ℝ) : (∀ x > 0, ∃ x > 0, f x k = 0) ↔ k ≤ -1 - (1 / Real.exp 1) :=
sorry

end k_range_l581_581725


namespace area_bounded_by_arccos_cos_l581_581276

-- Definition of the function and the interval
def arccos_cos (x : ℝ) : ℝ := Real.arccos (Real.cos x)
def interval_a : ℝ := -Real.pi / 2
def interval_b : ℝ := 3 * Real.pi / 2

-- The area under the curve arccos(cos x) on the given interval
noncomputable def area_under_curve : ℝ :=
  (interval_b - interval_a) * (Real.pi / 2) / 2

theorem area_bounded_by_arccos_cos :
  ∫ x in interval_a..interval_b, arccos_cos x = (3 * Real.pi^2) / 4 :=
by
  sorry

end area_bounded_by_arccos_cos_l581_581276


namespace count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581798

theorem count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes :
  let num_perfect_squares := 31
  let num_perfect_cubes := 9
  let num_perfect_sixth_powers := 2
  let num_either_squares_or_cubes := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers
  in num_either_squares_or_cubes = 38 :=
by
  sorry

end count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581798


namespace constant_term_in_expansion_l581_581651

noncomputable def binomialCoefficient : ℕ → ℕ → ℕ
| n, 0 := 1
| 0, k := 0
| n+1, k+1 := binomialCoefficient n k + binomialCoefficient (n+1) k

theorem constant_term_in_expansion :
  let general_term (r : ℕ) := ( -2 : ℝ ) ^ r * (binomialCoefficient 5 r) * ( x : ℝ) ^ (10 - 5 * r)
  (x : ℝ) :
  general_term 2 = 40 :=
by 
  sorry

end constant_term_in_expansion_l581_581651


namespace smallest_n_for_divisibility_l581_581062

theorem smallest_n_for_divisibility (a : ℕ) (h_odd : a % 2 = 1) : ∃ n : ℕ, (a^n - 1) % 2^2007 = 0 ∧ n = 2^(2007 - k) :=
by
  let k := Nat.findGreatestZeros a,
  have h2 : is_power_of_two 2 := sorry,
  use 2^(2007 - k),
  have h3 : (a^2^(2007 - k) - 1) % 2^2007 = 0 := sorry,
  exact ⟨2^(2007 - k), h3⟩,
  sorry

end smallest_n_for_divisibility_l581_581062


namespace part1_part2_part3_l581_581696

/-- Part 1 -/
theorem part1 (x : ℝ) : x ≥ 0 → f(x) = x - sin x → a = 1 → f(x) ≥ 0 :=
sorry

/-- Part 2 -/
theorem part2 : ∀ x ∈ Icc (1/2: ℝ) 2, g(x) = x^2 - log x → g'(1) = 1 →
  g(x) ∈ Icc (1/2 * (1 + log 2), 4 - log 2) :=
sorry

/-- Part 3 -/
theorem part3 (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f(x1) + sin x1 = log x1 ∧ f(x2) + sin x2 = log x2) →
  0 < a ∧ a < 1 / Real.e :=
sorry

end part1_part2_part3_l581_581696


namespace isabella_book_total_pages_l581_581843

theorem isabella_book_total_pages : 
  (∀ (d : ℕ), (d = 8)) → 
  (∀ (p1 : ℕ), (p1 = 4 * 28)) → 
  (∀ (p2 : ℕ), (p2 = 3 * 52)) → 
  (∀ (p3 : ℕ), (p3 = 20)) → 
  (p1 + p2 + p3 = 288) :=
by
  intros hd dp1 dp2 dp3
  rw dp1
  rw dp2
  rw dp3
  exact rfl

end isabella_book_total_pages_l581_581843


namespace integral_solution_l581_581459

theorem integral_solution (φ : ℝ → ℝ) (h : ∀ x > 0, ∫ (t : ℝ) in (0 : ℝ)..Real.top, φ t * sin (x * t) = Real.exp (-x)) :
  ∀ t : ℝ, φ t = (2 / Real.pi) * (t / (1 + t ^ 2)) :=
by
  assume t : ℝ
  existsi (λ t, (2 / Real.pi) * (t / (1 + t ^ 2)))
  sorry

end integral_solution_l581_581459


namespace inequality_proof_l581_581423

theorem inequality_proof (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l581_581423


namespace symmetry_proof_l581_581228

def Shape := Type
def originalShape : Shape
def shapes : List Shape := [shape1, shape2, shape3, shape4, shape5]
def is_symmetric_wrt_diagonal_dashed_line (s1 s2 : Shape) : Prop := sorry

theorem symmetry_proof :
  ∃ shape3 ∈ shapes, is_symmetric_wrt_diagonal_dashed_line originalShape shape3 :=
sorry

end symmetry_proof_l581_581228


namespace remainder_of_max_6_multiple_no_repeated_digits_l581_581856

theorem remainder_of_max_6_multiple_no_repeated_digits (M : ℕ) 
  (hM : ∃ n, M = 6 * n) 
  (h_unique_digits : ∀ (d : ℕ), d ∈ (M.digits 10) → (M.digits 10).count d = 1) 
  (h_max_M : ∀ (k : ℕ), (∃ n, k = 6 * n) ∧ (∀ (d : ℕ), d ∈ (k.digits 10) → (k.digits 10).count d = 1) → k ≤ M) :
  M % 100 = 78 := 
sorry

end remainder_of_max_6_multiple_no_repeated_digits_l581_581856


namespace Bs_share_l581_581555

theorem Bs_share (total_money : ℕ) (ratio_a ratio_b ratio_c : ℕ) 
  (h_total_money : total_money = 1800)
  (h_ratio : (ratio_a, ratio_b, ratio_c) = (2, 3, 4)) :
  let total_parts := ratio_a + ratio_b + ratio_c
  let value_per_part := total_money / total_parts
  let B_share := value_per_part * ratio_b
  B_share = 600 :=
by
  have h1 : total_parts = 2 + 3 + 4 := by rw h_ratio; refl
  have h2 : total_parts = 9 := by rw h1; norm_num
  have h3 : value_per_part = 1800 / 9 := by rw [h_total_money, h2]
  have h4 : value_per_part = 200 := by norm_num
  have h5 : B_share = 200 * 3 := by rw [h3, h_ratio]; refl
  have h6 : B_share = 600 := by rw h5; norm_num
  exact h6

end Bs_share_l581_581555


namespace count_perfect_squares_and_cubes_l581_581737

theorem count_perfect_squares_and_cubes : 
  let num_squares := 31 in
  let num_cubes := 9 in
  let num_sixth_powers := 3 in
  num_squares + num_cubes - num_sixth_powers = 37 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581737


namespace circle_through_BD_of_parallelogram_ABCD_l581_581573

theorem circle_through_BD_of_parallelogram_ABCD :
  ∀ (A B C D M N P K : Point) (circle : Circle),
    Parallelogram A B C D →
    PointsOnCircle circle [B, D] →
    IntersectCircleAndSides circle A B C D M N P K →
    Parallelogram.AB_parallel_MK circle A B C D M K →
    Parallelogram.CD_parallel_NP circle A B C D N P →
    Parallel MK NP :=
by
  sorry

end circle_through_BD_of_parallelogram_ABCD_l581_581573


namespace proportional_segments_l581_581511

variables (α β γ : Plane)
variables (A B C A' B' C' : Point)
variables (l l' : Line)

-- Conditions
axiom parallel_planes : α ∥ β ∧ β ∥ γ ∧ α ∥ γ
axiom intersections_l : l ∩ α = {A} ∧ l ∩ β = {B} ∧ l ∩ γ = {C}
axiom intersections_l' : l' ∩ α = {A'} ∧ l' ∩ β = {B'} ∧ l' ∩ γ = {C'}

-- Prove the final statement
theorem proportional_segments : (|A' - B'| / |B' - C'|) = (|A - B| / |B - C|) :=
sorry

end proportional_segments_l581_581511


namespace prove_f_of_increasing_l581_581321

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def strictly_increasing_on_positives (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

theorem prove_f_of_increasing {f : ℝ → ℝ}
  (h_odd : odd_function f)
  (h_incr : strictly_increasing_on_positives f) :
  f (-3) > f (-5) :=
by
  sorry

end prove_f_of_increasing_l581_581321


namespace find_m_l581_581805

theorem find_m (m : ℝ) (h1 : m > 0) (h2 : (4 - m) / (m - 2) = 2 * m) : 
  m = (3 + Real.sqrt 41) / 4 := by
  sorry

end find_m_l581_581805


namespace term_2023_is_4_l581_581921

noncomputable def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d, d^2).sum

noncomputable def sequence (n : ℕ) : ℕ :=
  Nat.iterate sum_of_squares_of_digits n 2023

theorem term_2023_is_4 : sequence 2023 = 4 := by
  sorry

end term_2023_is_4_l581_581921


namespace sum_S_odd_up_to_2017_eq_l581_581051

-- Define the sequence a_n and the sum S_n
noncomputable def a (n : ℕ) : ℝ := sorry 
noncomputable def S (n : ℕ) : ℝ := (-1)^n * (a n) - (1 / (2^n))

-- Define the specific sum S_1 + S_3 + ... + S_2017
noncomputable def sum_S_odd (n : ℕ) : ℝ :=
  if odd n then S n else 0

-- Calculate the sum for the required limits
noncomputable def sum_S_odd_limited (m : ℕ) : ℝ :=
  ∑ i in (range m).filter (λ i, odd i), S i

theorem sum_S_odd_up_to_2017_eq :
  sum_S_odd_limited 2018 = (1/3) * ((1 / (2^2018)) - 1) :=
begin
  sorry
end

end sum_S_odd_up_to_2017_eq_l581_581051


namespace count_perfect_squares_and_cubes_l581_581745

theorem count_perfect_squares_and_cubes (n : ℕ) (h : n = 1000) : 
  let squares := {x : ℕ | (x ^ 2) < n}
  let cubes := {x : ℕ | (x ^ 3) < n}
  let sixth_powers := {x : ℕ | (x ^ 6) < n}
  (squares.card + cubes.card - sixth_powers.card) = 39 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581745


namespace count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581797

theorem count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes :
  let num_perfect_squares := 31
  let num_perfect_cubes := 9
  let num_perfect_sixth_powers := 2
  let num_either_squares_or_cubes := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers
  in num_either_squares_or_cubes = 38 :=
by
  sorry

end count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581797


namespace maria_distance_from_start_l581_581871

theorem maria_distance_from_start :
  let initial_position := (0 : ℝ, 0 : ℝ),
      movement_north := (0, 10),
      movement_east := (30, 0),
      movement_south := (0, -40),
      final_position := initial_position + movement_north + movement_east + movement_south
  in
  dist initial_position final_position = 30 * Real.sqrt 2 :=
by
  -- All the movements are added as vectors
  let initial_position := (0 : ℝ, 0 : ℝ)
  let movement_north := (0, 10)
  let movement_east := (30, 0)
  let movement_south := (0, -40)
  --
  let final_position := initial_position + movement_north + movement_east + movement_south
  --
  sorry

end maria_distance_from_start_l581_581871


namespace area_triangle_PQU_l581_581029

-- Given areas
variables (R S T V : Type)
variables (area_RST area_RTV area_RSV area_PQU : ℝ)
variables (T_intersection : S → R → T) (U_intersection : T → V → S)

-- Assume given conditions
axiom area_RST_given : area_RST = 55
axiom area_RTV_given : area_RTV = 66
axiom area_RSV_given : area_RSV = 77
axiom configurations : ∀ S R V T, ∃ P Q U, 
  -- Intersections and specific triangle areas as configurations
  T_intersection S R = T ∧ U_intersection T V = U

-- Statement of the problem
theorem area_triangle_PQU :
  area_PQU = 840 :=
sorry

end area_triangle_PQU_l581_581029


namespace isolate_trees_with_fences_l581_581078

-- Given the following conditions: a square field with 16 oak trees arranged in a 4x4 grid.
def square_field : Set (ℕ × ℕ) := { (i, j) | i < 4 ∧ j < 4 }

-- Each tree is represented as a point in the 4x4 grid
def oak_trees : Set (ℕ × ℕ) := square_field

-- We want to prove that it is possible to place exactly 5 straight fences such that each tree is in its own section.
theorem isolate_trees_with_fences :
  ∃ fences : Set (ℕ → ℕ), 
  (∀ tree1 tree2 ∈ oak_trees, tree1 ≠ tree2 → ¬∃ fence ∈ fences, fence tree1 = fence tree2) ∧
  fences.card = 5 := 
sorry

end isolate_trees_with_fences_l581_581078


namespace second_year_growth_rate_l581_581848

variable (initial_investment : ℝ) (first_year_growth : ℝ) (additional_investment : ℝ) (final_value : ℝ) (second_year_growth : ℝ)

def calculate_portfolio_value_after_first_year (initial_investment first_year_growth : ℝ) : ℝ :=
  initial_investment * (1 + first_year_growth)

def calculate_new_value_after_addition (value_after_first_year additional_investment : ℝ) : ℝ :=
  value_after_first_year + additional_investment

def calculate_final_value_after_second_year (new_value second_year_growth : ℝ) : ℝ :=
  new_value * (1 + second_year_growth)

theorem second_year_growth_rate 
  (h1 : initial_investment = 80) 
  (h2 : first_year_growth = 0.15) 
  (h3 : additional_investment = 28) 
  (h4 : final_value = 132) : 
  calculate_final_value_after_second_year
    (calculate_new_value_after_addition
      (calculate_portfolio_value_after_first_year initial_investment first_year_growth)
      additional_investment)
    0.1 = final_value := 
  by
  sorry

end second_year_growth_rate_l581_581848


namespace problem_l581_581150

theorem problem 
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2010)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2006)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2010)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2006)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2010)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2006) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 996 / 1005 :=
sorry

end problem_l581_581150


namespace distribute_money_correct_l581_581075

def distribute_money : Prop :=
  ∃ (a b c d e f g h : ℕ),
    a = 823543 ∧
    b = 117649 ∧
    c = 16807 ∧
    d = 16807 ∧
    e = 16807 ∧
    f = 2401 ∧
    g = 2401 ∧
    h = 2401 ∧
    i = 343 ∧
    j = 343 ∧
    k = 343 ∧
    l = 49 ∧
    m = 49 ∧
    n = 49 ∧
    o = 7 ∧
    p = 1 ∧
    a + b + 3 * c + 3 * f + 3 * i + 3 * l + o + p = 1000000

theorem distribute_money_correct : distribute_money :=
by sorry

end distribute_money_correct_l581_581075


namespace intersection_point_count_l581_581634

def eq1 := λ x y : ℝ => x - 2*y + 3 = 0
def eq2 := λ x y : ℝ => 4*x + y - 5 = 0
def eq3 := λ x y : ℝ => x + 2*y - 3 = 0
def eq4 := λ x y : ℝ => 3*x - 4*y + 6 = 0

theorem intersection_point_count :
  (∃ x y : ℝ, eq1 x y ∧ eq3 x y) ∨ 
  (∃ x y : ℝ, eq1 x y ∧ eq4 x y) ∨ 
  (∃ x y : ℝ, eq2 x y ∧ eq3 x y) ∨ 
  (∃ x y : ℝ, eq2 x y ∧ eq4 x y) = 3 :=
sorry

end intersection_point_count_l581_581634


namespace hexagon_label_count_l581_581451

def hexagon_label (s : Finset ℕ) (a b c d e f g : ℕ) : Prop :=
  s = Finset.range 8 ∧ 
  (a ∈ s) ∧ (b ∈ s) ∧ (c ∈ s) ∧ (d ∈ s) ∧ (e ∈ s) ∧ (f ∈ s) ∧ (g ∈ s) ∧
  a + b + c + d + e + f + g = 28 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ 
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g ∧
  a + g + d = b + g + e ∧ b + g + e = c + g + f

theorem hexagon_label_count : ∃ s a b c d e f g, hexagon_label s a b c d e f g ∧ 
  (s.card = 8) ∧ (a + g + d = 10) ∧ (b + g + e = 10) ∧ (c + g + f = 10) ∧ 
  144 = 3 * 48 :=
sorry

end hexagon_label_count_l581_581451


namespace correct_first_coupon_day_l581_581037

def is_redemption_valid (start_day : ℕ) (interval : ℕ) (num_coupons : ℕ) (closed_day : ℕ) : Prop :=
  ∀ n : ℕ, n < num_coupons → (start_day + n * interval) % 7 ≠ closed_day

def wednesday : ℕ := 3  -- Assuming Sunday = 0, Monday = 1, ..., Saturday = 6

theorem correct_first_coupon_day : 
  is_redemption_valid wednesday 10 6 0 :=
by {
  -- Proof goes here
  sorry
}

end correct_first_coupon_day_l581_581037


namespace yellow_red_chair_ratio_l581_581100

variable (Y B : ℕ)
variable (red_chairs : ℕ := 5)
variable (total_chairs : ℕ := 43)

-- Condition: There are 2 fewer blue chairs than yellow chairs
def blue_chairs_condition : Prop := B = Y - 2

-- Condition: Total number of chairs
def total_chairs_condition : Prop := red_chairs + Y + B = total_chairs

-- Prove the ratio of yellow chairs to red chairs is 4:1
theorem yellow_red_chair_ratio (h1 : blue_chairs_condition Y B) (h2 : total_chairs_condition Y B) :
  (Y / red_chairs) = 4 := 
sorry

end yellow_red_chair_ratio_l581_581100


namespace cos_2A_value_find_a_l581_581011

variable (A B C a b c : ℝ)

-- Given conditions
hypothesis (h1 : A > B)
hypothesis (h2 : real.cos C = 5/13)
hypothesis (h3 : real.cos (A - B) = 3/5)
hypothesis (h4 : c = 15)

-- Questions rewritten as goals
theorem cos_2A_value :
  real.cos 2A = -63/65 := sorry

theorem find_a :
  a = 2 * real.sqrt 65 := sorry

end cos_2A_value_find_a_l581_581011


namespace zero_in_interval_l581_581129

def f (x : ℝ) := x^3 + x - 3

theorem zero_in_interval :
  (∃ c ∈ Ioo 1 2, f c = 0) :=
begin
  have h1 : f 1 < 0,
  { calc
    f 1 = 1^3 + 1 - 3 : by simp [f]
    ... = -1 : by norm_num },

  have h2 : f 2 > 0,
  { calc
    f 2 = 2^3 + 2 - 3 : by simp [f]
    ... = 7 : by norm_num },

  have h_cont : continuous f := by continuity,
  have h_incr : ∀ x y, x < y → f x < f y,
  { intros x y hxy, dsimp [f],
    refine lt_of_add_lt_add_right (add_lt_add_left _ 2),
    exact lt_of_pow_lt_pow 2 (by linarith [hxy]) (le_refl 2) },

  exact (intermediate_value_theorem_Ioo (-∞) ∞).mpr ⟨h_cont, h1, h2⟩
end

end zero_in_interval_l581_581129


namespace first_car_gas_consumed_l581_581015

theorem first_car_gas_consumed 
    (sum_avg_mpg : ℝ) (g2_gallons : ℝ) (total_miles : ℝ) 
    (avg_mpg_car1 : ℝ) (avg_mpg_car2 : ℝ) (g1_gallons : ℝ) :
    sum_avg_mpg = avg_mpg_car1 + avg_mpg_car2 →
    g2_gallons = 35 →
    total_miles = 2275 →
    avg_mpg_car1 = 40 →
    avg_mpg_car2 = 35 →
    g1_gallons = (total_miles - (avg_mpg_car2 * g2_gallons)) / avg_mpg_car1 →
    g1_gallons = 26.25 :=
by
  intros h_sum_avg_mpg h_g2_gallons h_total_miles h_avg_mpg_car1 h_avg_mpg_car2 h_g1_gallons
  sorry

end first_car_gas_consumed_l581_581015


namespace minimum_value_of_quadratic_expression_l581_581955

noncomputable def quadraticMinimum (a : ℝ) : ℝ :=
  let (x, y) := quadRoots (1 : ℝ) (-2 * a) (a + 6)
  (x - 1)^2 + (y - 1)^2

theorem minimum_value_of_quadratic_expression (a : ℝ) (h : quadraticDiscriminant 1 (-2 * a) (a + 6) ≥ 0) :
  quadraticMinimum a = 8 :=
sorry

end minimum_value_of_quadratic_expression_l581_581955


namespace long_furred_brown_dogs_l581_581366

-- Definitions based on given conditions
def T : ℕ := 45
def L : ℕ := 36
def B : ℕ := 27
def N : ℕ := 8

-- The number of long-furred brown dogs (LB) that needs to be proved
def LB : ℕ := 26

-- Lean 4 statement to prove LB
theorem long_furred_brown_dogs :
  L + B - LB = T - N :=
by 
  unfold T L B N LB -- we unfold definitions to simplify the theorem
  sorry

end long_furred_brown_dogs_l581_581366


namespace ram_pairs_sold_correct_l581_581576

-- Define the costs
def graphics_card_cost := 600
def hard_drive_cost := 80
def cpu_cost := 200
def ram_pair_cost := 60

-- Define the number of items sold
def graphics_cards_sold := 10
def hard_drives_sold := 14
def cpus_sold := 8
def total_earnings := 8960

-- Calculate earnings from individual items
def earnings_graphics_cards := graphics_cards_sold * graphics_card_cost
def earnings_hard_drives := hard_drives_sold * hard_drive_cost
def earnings_cpus := cpus_sold * cpu_cost

-- Calculate total earnings from graphics cards, hard drives, and CPUs
def earnings_other_items := earnings_graphics_cards + earnings_hard_drives + earnings_cpus

-- Calculate earnings from RAM
def earnings_from_ram := total_earnings - earnings_other_items

-- Calculate number of RAM pairs sold
def ram_pairs_sold := earnings_from_ram / ram_pair_cost

-- The theorem to be proven
theorem ram_pairs_sold_correct : ram_pairs_sold = 4 :=
by
  sorry

end ram_pairs_sold_correct_l581_581576


namespace upstream_distance_correct_l581_581204

-- Define the conditions
def downstream_distance : ℝ := 16
def downstream_time : ℝ := 2
def upstream_time : ℝ := 2
def still_water_speed : ℝ := 6.5

-- Define the speed of the stream
noncomputable def stream_speed (v : ℝ) : Prop := downstream_distance = (still_water_speed + v) * downstream_time

-- Define the upstream distance based on the speed of the stream
noncomputable def upstream_distance (d : ℝ) (v : ℝ) : Prop := d = (still_water_speed - v) * upstream_time

-- The main statement to prove
theorem upstream_distance_correct (v : ℝ) (d : ℝ) :
  stream_speed v →
  upstream_distance d v →
  d = 10 :=
begin
  intros h_stream_speed h_upstream_distance,
  sorry
end

end upstream_distance_correct_l581_581204


namespace count_perfect_squares_cubes_under_1000_l581_581758

-- Condition definitions
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, m * m = n
def isPerfectCube (n : Nat) : Prop := ∃ m : Nat, m * m * m = n

-- Main proof problem
theorem count_perfect_squares_cubes_under_1000 : 
  let count (p : Nat -> Prop) := Finset.card (Finset.filter p (Finset.range 1000))
  count (λ n, isPerfectSquare n ∨ isPerfectCube n) = 37 := 
sorry

end count_perfect_squares_cubes_under_1000_l581_581758


namespace midpoint_PQ_on_line_OT_range_TF_PQ_l581_581692

variable {a b : ℝ}
variable {F P Q T : ℝ × ℝ}
variable {O : ℝ × ℝ := (0, 0)}
variable C : ℝ → ℝ → ℝ

noncomputable def ellipse : (ℝ × ℝ) → Prop :=
  λ (p : ℝ × ℝ), (p.1^2 / a^2) + (p.2^2 / b^2) = 1

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def on_line_OT (G T : ℝ × ℝ) : Prop :=
  (T.1 - O.1) * (G.2 - O.2) = (T.2 - O.2) * (G.1 - O.1)

theorem midpoint_PQ_on_line_OT {a b : ℝ} (h : 0 < b ∧ b < a ∧ a > 0) (h₁ : 2 * real.sqrt (a^2 - b^2) = 2)
  (h₂a : midpoint P Q = G) (h₂b : on_line_OT G T) :
  true :=
by
	sorry

theorem range_TF_PQ {F P Q T : ℝ × ℝ} (h : 0 < b ∧ b < a ∧ a > 0) :
  set.range (λ x : ℝ, abs (real.sqrt ((4 - 1)^2 + (x - 0)^2) / (real.sqrt (1 + x^2) * real.sqrt ((-6*x / (3*x^2 + 4))^2 - 4 * (-9 / (3*x^2 + 4)))))) = set.Ici 1 :=
 by
	sorry

end midpoint_PQ_on_line_OT_range_TF_PQ_l581_581692


namespace total_leaves_on_farm_l581_581219

theorem total_leaves_on_farm : 
  (branches_per_tree subbranches_per_branch leaves_per_subbranch trees_on_farm : ℕ)
  (h1 : branches_per_tree = 10)
  (h2 : subbranches_per_branch = 40)
  (h3 : leaves_per_subbranch = 60)
  (h4 : trees_on_farm = 4) :
  (trees_on_farm * branches_per_tree * subbranches_per_branch * leaves_per_subbranch = 96000) :=
by
  sorry

end total_leaves_on_farm_l581_581219


namespace number_of_valid_sequences_l581_581688

-- Define the sequence and conditions
def is_valid_sequence (a : Fin 9 → ℝ) : Prop :=
  a 0 = 1 ∧ a 8 = 1 ∧
  ∀ i : Fin 8, (a (i + 1) / a i) ∈ ({2, 1, -1/2} : Set ℝ)

-- The main problem statement
theorem number_of_valid_sequences : ∃ n, n = 491 ∧ ∀ a : Fin 9 → ℝ, is_valid_sequence a ↔ n = 491 := 
sorry

end number_of_valid_sequences_l581_581688


namespace minimum_moves_to_win_l581_581690

-- Define the problem conditions
noncomputable def has_no_equal_angles (triangle : Type) : Prop := sorry
noncomputable def game_rule (point : Type) (color : Type) : Prop := sorry
noncomputable def forms_monochromatic_similar_triangle (points : set Type) (coloring : Type → Type → Prop) : Prop := sorry

-- Theorem statement for the equivalent proof problem
theorem minimum_moves_to_win (triangle : Type) [has_no_equal_angles triangle] (point : Type) (color : Type)
  (mark : ℕ → point) (coloring : point → color) (game_rule point color)
  (forms_monochromatic_similar_triangle : set point → (point → color) → Prop) :
  (∀ points : set point, forms_monochromatic_similar_triangle points coloring → points.card ≥ 3) →
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ mark : ℕ → point, ∃ (coloring : point → color),
    forms_monochromatic_similar_triangle ({mark i | i < n}.to_set) coloring) := 
sorry

end minimum_moves_to_win_l581_581690


namespace sally_baseball_cards_l581_581892

theorem sally_baseball_cards (initial_cards sold_cards : ℕ) (h1 : initial_cards = 39) (h2 : sold_cards = 24) :
  (initial_cards - sold_cards = 15) :=
by
  -- Proof needed
  sorry

end sally_baseball_cards_l581_581892


namespace incorrect_statement_d_l581_581057

noncomputable def x := Complex.mk (-1/2) (Real.sqrt 3 / 2)
noncomputable def y := Complex.mk (-1/2) (-Real.sqrt 3 / 2)

theorem incorrect_statement_d : (x^12 + y^12) ≠ 1 := by
  sorry

end incorrect_statement_d_l581_581057


namespace angle_ADB_is_90_degrees_l581_581572

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Given conditions
def is_circle (r : ℝ) (center : C) (points : Set C) : Prop := 
  ∀ p ∈ points, dist center p = r

def is_isosceles_triangle (A B C : C) : Prop :=
  dist A C = dist B C

def passes_through (circle : Set C) (point : C) : Prop :=
  point ∈ circle

def extension_intersects_circle (A C center : C) (circle : Set C) (extension_point : C) : Prop :=
  line A C ∈ extension_point ∧ extension_point ∈ circle

noncomputable def angle_ADB_degrees (A B D : C) : ℝ := sorry

-- Main theorem statement
theorem angle_ADB_is_90_degrees 
  (circle : Set C) (center : C) (r : ℝ) (A B D : C)
  (h_circle : is_circle r center circle)
  (h_radius : r = 8)
  (h_isosceles : is_isosceles_triangle A B center)
  (h_AB_not_equal_AC : dist A B ≠ dist A center)
  (h_passes_B : passes_through circle B)
  (h_extension_D : extension_intersects_circle A center center circle D) :
  angle_ADB_degrees A B D = 90 := sorry

end angle_ADB_is_90_degrees_l581_581572


namespace number_of_digits_with_10_sticks_l581_581165

-- Definitions based on conditions
def sticks_required (d : ℕ) : ℕ :=
  if d = 1 then 2 else 
  if d = 7 then 3 else
  if d = 4 then 4 else
  if d ∈ {2, 3, 5} then 5 else
  if d ∈ {0, 6, 9} then 6 else
  if d = 8 then 7 else 0

-- Lean statement for the proof problem
theorem number_of_digits_with_10_sticks (n : ℕ) : n = 59 := by
  sorry

end number_of_digits_with_10_sticks_l581_581165


namespace original_number_of_employees_l581_581179

theorem original_number_of_employees (x : ℕ) (reduction : ℝ) (retained_employees : ℕ) :
  reduction = 0.13 ∧ retained_employees = 195 ∧ (1 - reduction) * x = retained_employees → x = 224 :=
by
  intro h
  cases h with h_reduction h_rest
  cases h_rest with h_retained_employees h_equation
  -- Further proof steps are omitted
  sorry

end original_number_of_employees_l581_581179


namespace line_through_point_parallel_to_l_l581_581124

theorem line_through_point_parallel_to_l {x y : ℝ} (l : ℝ → ℝ → Prop) (A : ℝ × ℝ) :
  l = (λ x y, 2 * x - 4 * y + 7 = 0) → A = (2, 3) →
  (∃ m, (λ x y, 2 * x - 4 * y + m = 0) = (λ x y, x - 2 * y + 4 = 0)) :=
by
  intros hl hA
  use 8
  -- Further proof steps would go here
  sorry

end line_through_point_parallel_to_l_l581_581124


namespace average_speed_correct_l581_581982

-- Define the conditions
def part1_distance : ℚ := 10
def part1_speed : ℚ := 12
def part2_distance : ℚ := 12
def part2_speed : ℚ := 10

-- Total distance
def total_distance : ℚ := part1_distance + part2_distance

-- Time computations
def time1 : ℚ := part1_distance / part1_speed
def time2 : ℚ := part2_distance / part2_speed
def total_time : ℚ := time1 + time2

-- Average speed computation
def average_speed : ℚ := total_distance / total_time

theorem average_speed_correct :
  average_speed = 660 / 61 := sorry

end average_speed_correct_l581_581982


namespace number_of_perfect_squares_and_cubes_l581_581762

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end number_of_perfect_squares_and_cubes_l581_581762


namespace curve_equation_max_OPQ_dot_product_l581_581308

def moving_point_satisfies_slope_condition (x y : ℝ) : Prop :=
  (y / (x + 2)) * (y / (x - 2)) = -1 / 4

def curve_C (x y : ℝ) : Prop :=
  (x ≠ 2) ∧ (x ≠ -2) ∧ (x^2 / 4 + y^2 = 1)

theorem curve_equation (x y : ℝ) (h : moving_point_satisfies_slope_condition x y) : curve_C x y :=
sorry

theorem max_OPQ_dot_product :
  ∀ l : Line, line_passes_through l (1, 0) →
  (∃ P Q : ℝ × ℝ, (on_curve P ∧ on_curve Q ∧ line_intersects l P Q)) →
  max_dot_product (O P) (O Q) = 1 / 4 :=
sorry

end curve_equation_max_OPQ_dot_product_l581_581308


namespace bhanu_petrol_expense_l581_581613

variable (X : ℝ) -- Bhanu's total income
variable (petrol_expense house_rent : ℝ) -- expenses on petrol and house rent

-- Define the conditions
def conditions (X : ℝ) (house_rent : ℝ) : Prop :=
  petrol_expense = 0.30 * X ∧
  house_rent = 0.30 * (X - petrol_expense) ∧
  house_rent = 210

-- The proof problem stating that Bhanu's petrol expense is Rs. 300
theorem bhanu_petrol_expense (X : ℝ) (house_rent : ℝ) (h : conditions X house_rent) :
  petrol_expense = 300 :=
by
  sorry

end bhanu_petrol_expense_l581_581613


namespace emir_needs_more_money_l581_581638

def dictionary_cost : ℕ := 5
def dinosaur_book_cost : ℕ := 11
def cookbook_cost : ℕ := 5
def saved_money : ℕ := 19
def total_cost : ℕ := dictionary_cost + dinosaur_book_cost + cookbook_cost
def additional_money_needed : ℕ := total_cost - saved_money

theorem emir_needs_more_money : additional_money_needed = 2 := by
  sorry

end emir_needs_more_money_l581_581638


namespace polygon_intersection_l581_581967

variable (ℝ : Type _) [LinearOrderedField ℝ]

-- Defining a point in the plane
structure Point (ℝ : Type _) where
  x : ℝ
  y : ℝ

-- Defining a line in the plane by a slope and intercept
structure Line (ℝ : Type _) where
  slope : ℝ
  intercept : ℝ

-- Defining a closed polygonal chain with a finite number of segments
structure Polygon (ℝ : Type _) where
  vertices : List (Point ℝ)
  is_closed : vertices.head? = vertices.last?

-- A function to count the number of intersection points between a line and a polygon
def count_intersections (l : Line ℝ) (P : Polygon ℝ) : ℕ := sorry

-- Given conditions
axiom given_conditions 
  (P : Polygon ℝ) 
  (l : Line ℝ) 
  (h_intersections : count_intersections l P = 1985) : 

-- The theorem to prove
theorem polygon_intersection (P : Polygon ℝ) (l : Line ℝ) 
  (h_intersections : count_intersections l P = 1985) : 
  ∃ l₁ : Line ℝ, count_intersections l₁ P > 1985 := 
sorry

end polygon_intersection_l581_581967


namespace length_BD_l581_581084

/-- Points A, B, C, and D lie on a line in that order. We are given:
  AB = 2 cm,
  AC = 5 cm, and
  CD = 3 cm.
Then, we need to show that the length of BD is 6 cm. -/
theorem length_BD :
  ∀ (A B C D : ℕ),
  A + B = 2 → A + C = 5 → C + D = 3 →
  D - B = 6 :=
by
  intros A B C D h1 h2 h3
  -- Proof steps to be filled in
  sorry

end length_BD_l581_581084


namespace probability_same_color_set_l581_581850

theorem probability_same_color_set 
  (black_pairs blue_pairs : ℕ)
  (green_pairs : {g : Finset (ℕ × ℕ) // g.card = 3})
  (total_pairs := 15)
  (total_shoes := total_pairs * 2) :
  2 * black_pairs + 2 * blue_pairs + green_pairs.val.card * 2 = total_shoes →
  ∃ probability : ℚ, 
    probability = 89 / 435 :=
by
  intro h_total_shoes
  let black_shoes := black_pairs * 2
  let blue_shoes := blue_pairs * 2
  let green_shoes := green_pairs.val.card * 2
  
  have h_black_probability : ℚ := (black_shoes / total_shoes) * (black_pairs / (total_shoes - 1))
  have h_blue_probability : ℚ := (blue_shoes / total_shoes) * (blue_pairs / (total_shoes - 1))
  have h_green_probability : ℚ := (green_shoes / total_shoes) * (green_pairs.val.card / (total_shoes - 1))
  
  have h_total_probability : ℚ := h_black_probability + h_blue_probability + h_green_probability
  
  use h_total_probability
  sorry

end probability_same_color_set_l581_581850


namespace probability_X_equals_Y_l581_581227

noncomputable def prob_X_equals_Y : ℚ :=
  let count_intersections : ℚ := 15
  let total_possibilities : ℚ := 15 * 15
  count_intersections / total_possibilities

theorem probability_X_equals_Y :
  (∀ (x y : ℝ), -15 * Real.pi ≤ x ∧ x ≤ 15 * Real.pi ∧ -15 * Real.pi ≤ y ∧ y ≤ 15 * Real.pi →
    (Real.cos (Real.cos x) = Real.cos (Real.cos y)) →
    prob_X_equals_Y = 1/15) :=
sorry

end probability_X_equals_Y_l581_581227


namespace softball_team_total_players_l581_581500

theorem softball_team_total_players (M W : ℕ) (h1 : W = M + 5) (h2 : M = 0.5 * W) : M + W = 15 := by
  sorry

end softball_team_total_players_l581_581500


namespace count_perfect_squares_and_cubes_l581_581748

theorem count_perfect_squares_and_cubes (n : ℕ) (h : n = 1000) : 
  let squares := {x : ℕ | (x ^ 2) < n}
  let cubes := {x : ℕ | (x ^ 3) < n}
  let sixth_powers := {x : ℕ | (x ^ 6) < n}
  (squares.card + cubes.card - sixth_powers.card) = 39 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581748


namespace no_even_and_increasing_function_l581_581835

-- Definition of a function being even
def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Definition of a function being increasing
def is_increasing_function (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x ≤ f y

-- Theorem stating the non-existence of a function that is both even and increasing
theorem no_even_and_increasing_function : ¬ ∃ f : ℝ → ℝ, is_even_function f ∧ is_increasing_function f :=
by
  sorry

end no_even_and_increasing_function_l581_581835


namespace propositions_correct_l581_581601

def vertical_angles (α β : ℝ) : Prop := ∃ γ, α = γ ∧ β = γ

def problem_statement : Prop :=
  (∀ α β, vertical_angles α β → α = β) ∧
  ¬(∀ α β, α = β → vertical_angles α β) ∧
  ¬(∀ α β, ¬vertical_angles α β → ¬(α = β)) ∧
  (∀ α β, ¬(α = β) → ¬vertical_angles α β)

theorem propositions_correct :
  problem_statement :=
by
  sorry

end propositions_correct_l581_581601


namespace arcs_independent_of_red_ordering_l581_581408

open Set

noncomputable def unit_circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }

variables {n : ℕ} (P : Fin 2n → (ℝ × ℝ)) (R B : Fin n → (ℝ × ℝ))

-- Conditions
axiom h1 : ∀ i j : Fin 2n, i ≠ j → P i ≠ P j
axiom h2 : ∀ i : Fin 2n, P i ∈ unit_circle ∧ P i ≠ (1, 0)
axiom h3 : ∀ i : Fin n, P (Fin.of_nat i) = R i
axiom h4 : ∀ i : Fin n, P (Fin.add_nat i n) = B i
axiom h5 : Function.Surjective (λ x : Fin 2n, if x.val < n then R (Fin.of_nat x.val) else B (Fin.of_nat (x.val - n)))

-- Define nearest blue point traveling counterclockwise
def nearest_blue_counterclockwise (ri : (ℝ × ℝ)) (blues : Fin n → (ℝ × ℝ)) : (ℝ × ℝ) :=
  let distances := λ bi : (ℝ × ℝ), (Math.atan2 bi.2 bi.1 - Math.atan2 ri.2 ri.1 + 2 * π) % (2 * π)
  let nearest := Fin.argmin distances blues
  blues nearest

axiom blue_points_assignment : ∀ (i : Fin n), B i = nearest_blue_counterclockwise (R i) B

-- The proof statement
theorem arcs_independent_of_red_ordering :
  ∀ (ordering : Fin n → (ℝ × ℝ)), 
  (∃ ! i : Fin n, (1, 0) ∈ (unit_circle : set ℝ × ℝ) ∧ Arc (λ i, (R i, B i)) i) ↔
  (∃ ! i : Fin n, (1, 0) ∈ (unit_circle : set ℝ × ℝ) ∧ Arc (λ i, (ordering i, nearest_blue_counterclockwise (ordering i) B)) i)
:= 
sorry

end arcs_independent_of_red_ordering_l581_581408


namespace segment_length_l581_581281

theorem segment_length :
  let
    curve_x (t : ℝ) := t + 1 / t,
    curve_y (t : ℝ) := t - 1 / t,
    line_x (s : ℝ) := -3 + (√3 / 2) * s,
    line_y (s : ℝ) := (1 / 2) * s,
    cartesian_x_y (x y : ℝ) := x^2 - y^2 = 4,
    quadratic_eq (s : ℝ) := s^2 - 6 * √3 * s + 10
  in
  (∃ (s₁ s₂ : ℝ), quadratic_eq s₁ = 0 ∧ quadratic_eq s₂ = 0 ∧
                   abs (s₁ - s₂) = 2 * √17) :=
by
  sorry

end segment_length_l581_581281


namespace min_value_g_range_of_m_l581_581299

section
variable (x : ℝ)
noncomputable def g (x : ℝ) := Real.exp x - x

theorem min_value_g :
  (∀ x : ℝ, g x ≥ g 0) ∧ g 0 = 1 := 
by 
  sorry

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2 * x - m) / g x > x) → m < Real.log 2 ^ 2 := 
by 
  sorry
end

end min_value_g_range_of_m_l581_581299


namespace length_of_b_l581_581841

-- Defining the given conditions in the problem
variable {A B C : Type} [triangle : triangle A B C] -- Representing triangle ABC
variable (angle_A angle_C : ℝ)                      -- Angles in the triangle A, C
variable (a c : ℝ)                                  -- Sides a, c in the triangle

-- Specifying the conditions
def condition1 := angle_C = 3 * angle_A
def condition2 := a = 27
def condition3 := c = 48

-- Statement of the proof problem
theorem length_of_b (A B C : Type) [triangle : triangle A B C] 
  (angle_A angle_C : ℝ) (a c : ℝ) 
  (h1 : angle_C = 3 * angle_A) (h2 : a = 27) (h3 : c = 48) : ∃ b : ℝ, b = 35 := 
sorry

end length_of_b_l581_581841


namespace find_smallest_n_l581_581730

-- State the conditions.
def sequence : ℕ → ℝ 
| n := if n = 0 then 9 else
  let a_n := sequence n - 1 in
  (4 - a_n) / 3 

noncomputable def Sn (n : ℕ) : ℝ := ∑ i in finset.range (n+1), sequence i

-- State the problem
theorem find_smallest_n : ∃ (n : ℕ), (|Sn n - n - 6| < 1 / 125) ∧ (∀ m < n, |Sn m - m - 6| ≥ 1 / 125) :=
sorry

end find_smallest_n_l581_581730


namespace find_n_l581_581809

-- Given conditions
variable (x y : ℝ) 
variable (h : 2 * x - y = 4)

-- To prove: n = 3 in the equation 6x - ny = 12
theorem find_n : 6 * x - 3 * y = 12 ≈ (6 * x - n * y = 12 → n = 3) :=
by
  sorry

end find_n_l581_581809


namespace maximize_S_n_l581_581019

variable {a : ℕ → ℝ} -- Sequence term definition
variable {S : ℕ → ℝ} -- Sum of first n terms

-- Definitions based on conditions
noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  n * a 1 + (n * (n - 1) / 2) * ((a 2) - (a 1))

axiom a1_positive (a1 : ℝ) : 0 < a1 -- given a1 > 0
axiom S3_eq_S16 (a1 d : ℝ) : sum_of_first_n_terms a 3 = sum_of_first_n_terms a 16

-- Problem Statement
theorem maximize_S_n (a : ℕ → ℝ) (d : ℝ) : is_arithmetic_sequence a d →
  a 1 > 0 →
  sum_of_first_n_terms a 3 = sum_of_first_n_terms a 16 →
  (∀ n, sum_of_first_n_terms a n = sum_of_first_n_terms a 9 ∨ sum_of_first_n_terms a n = sum_of_first_n_terms a 10) :=
by
  sorry

end maximize_S_n_l581_581019


namespace doors_keys_99_attempts_doors_keys_75_attempts_doors_keys_74_attempts_not_possible_l581_581504

-- Definitions based on conditions
structure DoorKeyConfiguration (n : ℕ) :=
  (doors : Fin n)
  (keys : Fin n)
  (possible_mismatches : Fin n → Fin n → Bool) -- represents whether a key can open a door

-- Hypotheses based on conditions
axiom keys_match_or_diff_one : ∀ {n : ℕ} (config : DoorKeyConfiguration n), 
  ∀ (k d : Fin n), 
  config.possible_mismatches k d = (k.1 = d.1 ∨ k.1 = d.1 + 1 ∨ k.1 + 1 = d.1)

-- Statement of the problems
theorem doors_keys_99_attempts (config : DoorKeyConfiguration 100) : 
  ∃ attempt_seq : ℕ → Fin 100 × Fin 100, 
  (∀ i : Fin 100, ∃ j : ℕ, attempt_seq j = (i, i) ∨ attempt_seq j = (i, ⟨i + 1, dec_trivial⟩) ∨ attempt_seq j = (⟨i + 1, dec_trivial⟩, i)) ∧
  attempts ≤ 99 := sorry

theorem doors_keys_75_attempts (config : DoorKeyConfiguration 100) : 
  ∃ attempt_seq : ℕ → Fin 100 × Fin 100, 
  (∀ i : Fin 100, ∃ j : ℕ, attempt_seq j = (i, i) ∨ attempt_seq j = (i, ⟨i + 1, dec_trivial⟩) ∨ attempt_seq j = (⟨i + 1, dec_trivial⟩, i)) ∧
  attempts ≤ 75 := sorry

theorem doors_keys_74_attempts_not_possible (config : DoorKeyConfiguration 100) : 
  ¬ ( ∃ attempt_seq : ℕ → Fin 100 × Fin 100, 
  (∀ i : Fin 100, ∃ j : ℕ, attempt_seq j = (i, i) ∨ attempt_seq j = (i, ⟨i + 1, dec_trivial⟩) ∨ attempt_seq j = (⟨i + 1, dec_trivial⟩, i)) ∧
  attempts ≤ 74 ) := sorry

end doors_keys_99_attempts_doors_keys_75_attempts_doors_keys_74_attempts_not_possible_l581_581504


namespace intersect_at_point_m_eq_1_3_n_eq_neg_73_9_lines_parallel_pass_through_lines_perpendicular_y_intercept_l581_581732

theorem intersect_at_point_m_eq_1_3_n_eq_neg_73_9 
  (m : ℚ) (n : ℚ) : 
  (m^2 + 8 + n = 0) ∧ (3*m - 1 = 0) → 
  (m = 1/3 ∧ n = -73/9) := 
by 
  sorry

theorem lines_parallel_pass_through 
  (m : ℚ) (n : ℚ) :
  (m ≠ 0) → (m^2 = 16) ∧ (3*m - 8 + n = 0) → 
  (m = 4 ∧ n = -4) ∨ (m = -4 ∧ n = 20) :=
by 
  sorry

theorem lines_perpendicular_y_intercept 
  (m : ℚ) (n : ℚ) :
  (m = 0 ∧ 8*(-1) + n = 0) → 
  (m = 0 ∧ n = 8) :=
by 
  sorry

end intersect_at_point_m_eq_1_3_n_eq_neg_73_9_lines_parallel_pass_through_lines_perpendicular_y_intercept_l581_581732


namespace exists_wily_l581_581016

-- Define the concepts of Knight, Liar, and Wily
inductive Individual
| knight
| liar
| wily

-- Statements made by individuals
def statement_A (x y z : Individual) : Prop :=
  x = Individual.liar ∨ y = Individual.liar ∨ z = Individual.liar

def statement_B (x y z : Individual) : Prop :=
  (x = Individual.liar ∨ y = Individual.liar) ∧ (x = Individual.liar ∨ z = Individual.liar) ∧ (y = Individual.liar ∨ z = Individual.liar)

def statement_C (x y z : Individual) : Prop :=
  x = Individual.liar ∧ y = Individual.liar ∧ z = Individual.liar

-- Main theorem statement
theorem exists_wily (x y z : Individual) : 
  (statement_A x y z) → 
  (statement_B x y z) → 
  (statement_C x y z) → 
  x = Individual.wily ∨ y = Individual.wily ∨ z = Individual.wily :=
begin
  intros hA hB hC,
  sorry
end

end exists_wily_l581_581016


namespace green_apples_ordered_l581_581141

-- Definitions based on the conditions
variable (red_apples : Nat := 25)
variable (students : Nat := 10)
variable (extra_apples : Nat := 32)
variable (G : Nat)

-- The mathematical problem to prove
theorem green_apples_ordered :
  red_apples + G - students = extra_apples → G = 17 := by
  sorry

end green_apples_ordered_l581_581141


namespace count_subsets_containing_123_l581_581616

open Set

theorem count_subsets_containing_123 : 
  {X : Set ℕ // {1, 2, 3} ⊆ X ∧ X ⊆ {1, 2, 3, 4, 5, 6}}.toFinset.card = 8 := 
by 
  sorry

end count_subsets_containing_123_l581_581616


namespace model2_best_fit_l581_581375
-- Import necessary tools from Mathlib

-- Define the coefficients of determination for the four models
def R2_model1 : ℝ := 0.75
def R2_model2 : ℝ := 0.90
def R2_model3 : ℝ := 0.28
def R2_model4 : ℝ := 0.55

-- Define the best fitting model
def best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ) : Prop :=
  R2_2 > R2_1 ∧ R2_2 > R2_3 ∧ R2_2 > R2_4

-- Statement to prove
theorem model2_best_fit : best_fitting_model R2_model1 R2_model2 R2_model3 R2_model4 :=
  by
  -- Proof goes here
  sorry

end model2_best_fit_l581_581375


namespace coefficient_x7_in_expansion_l581_581167

theorem coefficient_x7_in_expansion : 
  let n := 10
  let k := 7
  let binom := Nat.choose n k
  let coeff := 1
  coeff * binom = 120 :=
by
  sorry

end coefficient_x7_in_expansion_l581_581167


namespace intersection_P_Q_l581_581070

-- Define set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define set Q
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the problem statement as a theorem
theorem intersection_P_Q : P ∩ Q = {1, 2} :=
by
  sorry

end intersection_P_Q_l581_581070


namespace count_perfect_squares_and_cubes_l581_581746

theorem count_perfect_squares_and_cubes (n : ℕ) (h : n = 1000) : 
  let squares := {x : ℕ | (x ^ 2) < n}
  let cubes := {x : ℕ | (x ^ 3) < n}
  let sixth_powers := {x : ℕ | (x ^ 6) < n}
  (squares.card + cubes.card - sixth_powers.card) = 39 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581746


namespace power_function_value_l581_581336

theorem power_function_value (a : ℝ) (f : ℝ → ℝ) 
    (h₀ : ∀ x : ℝ, f x = x ^ a)
    (h₁ : f 2 = (real.sqrt 2) / 2) : 
  f 4 = 1 / 2 := 
sorry

end power_function_value_l581_581336


namespace optimal_rental_decision_optimal_purchase_decision_l581_581989

-- Definitions of conditions
def monthly_fee_first : ℕ := 50000
def monthly_fee_second : ℕ := 10000
def probability_seizure : ℚ := 0.5
def moving_cost : ℕ := 70000
def months_first_year : ℕ := 12
def months_seizure : ℕ := 4
def months_after_seizure : ℕ := months_first_year - months_seizure
def purchase_cost : ℕ := 2000000
def installment_period : ℕ := 36

-- Proving initial rental decision
theorem optimal_rental_decision :
  let annual_cost_first := monthly_fee_first * months_first_year
  let annual_cost_second := (monthly_fee_second * months_seizure) + (monthly_fee_first * months_after_seizure) + moving_cost
  annual_cost_second < annual_cost_first := 
by
  sorry

-- Proving purchasing decision
theorem optimal_purchase_decision :
  let total_rent_cost_after_seizure := (monthly_fee_second * months_seizure) + moving_cost + (monthly_fee_first * (4 * months_first_year - months_seizure))
  let total_purchase_cost := purchase_cost
  total_purchase_cost < total_rent_cost_after_seizure :=
by
  sorry

end optimal_rental_decision_optimal_purchase_decision_l581_581989


namespace price_per_pound_l581_581844

variable (P : ℝ)

-- Conditions as definitions
def weight_vest_cost : ℝ := 250
def weight_plate_pounds : ℝ := 200
def vest_and_plate_total_cost : ℝ := weight_vest_cost + (weight_plate_pounds * P)
def discounted_vest_cost : ℝ := 700 - 100
def savings : ℝ := 110
def expected_total_cost : ℝ := discounted_vest_cost - savings

-- Statement to prove that price per pound for the weight plates is 1.2
theorem price_per_pound (h : vest_and_plate_total_cost = expected_total_cost) : P = 1.2 :=
by
  sorry

end price_per_pound_l581_581844


namespace train_length_l581_581163

theorem train_length (speed_fast speed_slow : ℝ) (time_pass : ℝ)
  (L : ℝ)
  (hf : speed_fast = 46 * (1000/3600))
  (hs : speed_slow = 36 * (1000/3600))
  (ht : time_pass = 36)
  (hL : (2 * L = (speed_fast - speed_slow) * time_pass)) :
  L = 50 := by
  sorry

end train_length_l581_581163


namespace sum_odd_divisors_252_l581_581539

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : list ℕ := (list.range (n + 1)).filter (λ d, n % d = 0)

def odd_divisors_sum (n : ℕ) : ℕ :=
(list.filter is_odd (divisors n)).sum

theorem sum_odd_divisors_252 : odd_divisors_sum 252 = 104 :=
by
  -- Proof goes here
  sorry

end sum_odd_divisors_252_l581_581539


namespace sufficient_but_not_necessary_l581_581354

theorem sufficient_but_not_necessary (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) : 
  (a > b > 0) ↔ (a² > b²) ∧ (a² > b²) ↔ ¬(a > b ∧ a > 0 ∧ b > 0) := 
sorry

end sufficient_but_not_necessary_l581_581354


namespace different_lists_possible_l581_581977

-- Define the problem and state the main theorem.
theorem different_lists_possible : 
  let num_balls := 12
  let total_lists := 12 * 12 * 12
  total_lists = 1728 :=
by
  let num_balls := 12
  have h1 : total_lists = num_balls ^ 3 := sorry
  show total_lists = 1728 from sorry

end different_lists_possible_l581_581977


namespace area_of_given_ellipse_l581_581522

noncomputable def ellipse_area {a b : ℝ} (h : a ≠ 0 ∧ b ≠ 0) : ℝ := 
  Real.pi * a * b

theorem area_of_given_ellipse : 
  ellipse_area 4 3 ⟨ by norm_num, by norm_num ⟩ = 12 * Real.pi :=
sorry

end area_of_given_ellipse_l581_581522


namespace largest_diff_with_rearranged_digits_l581_581117

theorem largest_diff_with_rearranged_digits :
  ∀ (a b : ℕ), (a ≤ 3000 ∧ a ≥ 1000) ∧ (b ≤ 3000 ∧ b ≥ 1000) ∧
  (a ∈ {n | ∃ (x y z w : ℕ), [x, y, z, w] ~ [2, 0, 2, 1] ∧ n = x * 1000 + y * 100 + z * 10 + w}) ∧
  (b ∈ {n | ∃ (x y z w : ℕ), [x, y, z, w] ~ [2, 0, 2, 1] ∧ n = x * 1000 + y * 100 + z * 10 + w}) →
  abs (a - b) ≤ 1188 := by
  sorry

end largest_diff_with_rearranged_digits_l581_581117


namespace train_speed_approx_l581_581215

noncomputable def distance_in_kilometers (d : ℝ) : ℝ :=
d / 1000

noncomputable def time_in_hours (t : ℝ) : ℝ :=
t / 3600

noncomputable def speed_in_kmh (d : ℝ) (t : ℝ) : ℝ :=
distance_in_kilometers d / time_in_hours t

theorem train_speed_approx (d t : ℝ) (h_d : d = 200) (h_t : t = 5.80598713393251) :
  abs (speed_in_kmh d t - 124.019) < 1e-3 :=
by
  rw [h_d, h_t]
  simp only [distance_in_kilometers, time_in_hours, speed_in_kmh]
  norm_num
  -- We're using norm_num to deal with numerical approximations and constants
  -- The actual calculations can be verified through manual checks or external tools but in Lean we skip this step.
  sorry

end train_speed_approx_l581_581215


namespace total_profit_amount_l581_581584

noncomputable def total_profit : ℕ :=
let P_X := 3 * 200 in    -- P_X = 3k, k = 200
let P_Y := 2 * 200 in    -- P_Y = 2k
5 * 200 -- Total Profit = 3k + 2k = 5k

theorem total_profit_amount :
  (P_X - P_Y = 200 ∧ P_X / P_Y = 3 / 2) →
  total_profit = 1000 :=
by
  sorry

end total_profit_amount_l581_581584


namespace volume_of_sphere_in_cone_l581_581209

theorem volume_of_sphere_in_cone :
  let diameter := 16
  let radius := (diameter : ℝ) / 2
  let hypotenuse := radius * Real.sqrt 2
  let radius_of_sphere := radius * (Real.sqrt 2 - 1) / Real.sqrt 2
  volume (sphere radius_of_sphere) = (1024 * (2 * Real.sqrt 2 - 3)^3 * real.pi) / 9 :=
by 
  sorry

end volume_of_sphere_in_cone_l581_581209


namespace total_leaves_on_farm_l581_581217

theorem total_leaves_on_farm : 
  (branches_per_tree subbranches_per_branch leaves_per_subbranch trees_on_farm : ℕ)
  (h1 : branches_per_tree = 10)
  (h2 : subbranches_per_branch = 40)
  (h3 : leaves_per_subbranch = 60)
  (h4 : trees_on_farm = 4) :
  (trees_on_farm * branches_per_tree * subbranches_per_branch * leaves_per_subbranch = 96000) :=
by
  sorry

end total_leaves_on_farm_l581_581217


namespace max_expression_value_l581_581641

-- Define each expression as a real number term in Lean
def expr1 : ℝ := 3 + 1 + 2 + 4
def expr2 : ℝ := 3 * 1 + 2 + 4
def expr3 : ℝ := 3 + 1 * 2 + 4
def expr4 : ℝ := 3 + 1 + 2 * 4
def expr5 : ℝ := 3 * 1 * 2 * 4

-- Define the correctness of the evaluated values of each expression
lemma expr1_correct : expr1 = 10 := by norm_num
lemma expr2_correct : expr2 = 9 := by norm_num
lemma expr3_correct : expr3 = 9 := by norm_num
lemma expr4_correct : expr4 = 12 := by norm_num
lemma expr5_correct : expr5 = 24 := by norm_num

-- Prove that expr5 is the largest value among the given expressions
lemma expr5_is_largest : expr5 > expr1 ∧ expr5 > expr2 ∧ expr5 > expr3 ∧ expr5 > expr4 :=
  by {
    split; norm_num,
    split; norm_num,
    split; norm_num,
    norm_num
  }

-- Final statement to prove that the maximum value is indeed from expr5
theorem max_expression_value : expr5 = (max (max (max (max expr1 expr2) expr3) expr4) expr5) := by {
  rw [expr1_correct, expr2_correct, expr3_correct, expr4_correct, expr5_correct],
  norm_num
}

end max_expression_value_l581_581641


namespace minimize_average_annual_cost_l581_581581

theorem minimize_average_annual_cost :
  ∃ x : ℕ, (x = 11) ∧ 
            let c := 121 in
            let a := 10 in
            let m := (λ (n : ℕ), if n = 1 then 0 else 2 * (n - 1)) in
            let total_cost := c + x * a + (∑ i in range x, m (i + 1)) in
            let average_annual_cost := total_cost / x in
            (∀ y : ℕ, y ≠ x → let total_cost_y := c + y * a + (∑ i in range y, m (i + 1)) in
                              let average_annual_cost_y := total_cost_y / y in
                              average_annual_cost < average_annual_cost_y) := 
begin
  sorry
end

end minimize_average_annual_cost_l581_581581


namespace probability_of_negative_product_is_3_over_7_l581_581940

noncomputable def probability_of_negative_product : ℚ :=
  let set_of_integers := {-5, 0, -8, 7, 4, -2, -3, 1}
  let total_combinations := 28 -- (binom 8 2)
  let negative_product_combinations := 12 -- (4 negatives * 3 positives)
  in negative_product_combinations / total_combinations

theorem probability_of_negative_product_is_3_over_7 : 
  probability_of_negative_product = 3/7 :=
sorry

end probability_of_negative_product_is_3_over_7_l581_581940


namespace distance_between_parallel_lines_l581_581509

theorem distance_between_parallel_lines 
  (r : ℝ) (d : ℝ) 
  (h1 : 3 * (2 * r^2) = 722 + (19 / 4) * d^2) 
  (h2 : 3 * (2 * r^2) = 578 + (153 / 4) * d^2) : 
  d = 6 :=
by
  sorry

end distance_between_parallel_lines_l581_581509


namespace matchsticks_problem_l581_581622

theorem matchsticks_problem (total_matchsticks : ℕ) (frac_removed : ℚ)
  (initial_word : String) (remaining_word : String) :
  total_matchsticks = 10 ∧ frac_removed = 7/10 ∧ initial_word = "FIVE" ∧ remaining_word = "IV" →
  (⟦ total_matchsticks * (1 - frac_removed) = 3 ⟧) := 
by
  intros h
  sorry

end matchsticks_problem_l581_581622


namespace lemonade_sales_l581_581896

theorem lemonade_sales (cups_last_week : ℕ) (percent_more : ℕ) 
  (h_last_week : cups_last_week = 20)
  (h_percent_more : percent_more = 30) : 
  let cups_this_week := cups_last_week + (percent_more * cups_last_week / 100)
  in cups_last_week + cups_this_week = 46 := 
by
  -- Definitions and calculation
  let cups_this_week := cups_last_week + (percent_more * cups_last_week / 100)
  have h_this_week : cups_this_week = 26, from calc
    cups_this_week = 20 + (30 * 20 / 100) : by rw [h_last_week, h_percent_more]
    ... = 20 + 6 : by norm_num
    ... = 26 : by norm_num,
  show cups_last_week + cups_this_week = 46, from calc
    20 + 26 = 46 : by norm_num

end lemonade_sales_l581_581896


namespace total_cost_of_bill_l581_581104

def original_price_curtis := 16.00
def original_price_rob := 18.00
def time_of_meal := 3

def is_early_bird_discount_applicable (time : ℕ) : Prop :=
  2 ≤ time ∧ time ≤ 4

theorem total_cost_of_bill :
  is_early_bird_discount_applicable time_of_meal →
  original_price_curtis / 2 + original_price_rob / 2 = 17.00 :=
by
  sorry

end total_cost_of_bill_l581_581104


namespace gwen_books_total_l581_581734

theorem gwen_books_total
  (mystery_shelves : Nat) (picture_shelves : Nat) (books_per_shelf : Nat)
  (hm : mystery_shelves = 5) (hp : picture_shelves = 3) (hbooks : books_per_shelf = 4) :
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 32 := 
by
  rw [hm, hp, hbooks]
  norm_num
  sorry

end gwen_books_total_l581_581734


namespace geometric_sequence_divisibility_l581_581428

theorem geometric_sequence_divisibility 
  (a1 : ℚ) (h1 : a1 = 1 / 2) 
  (a2 : ℚ) (h2 : a2 = 10) 
  (n : ℕ) :
  ∃ (n : ℕ), a_n = (a1 * 20^(n - 1)) ∧ (n ≥ 4) ∧ (5000 ∣ a_n) :=
by
  sorry

end geometric_sequence_divisibility_l581_581428


namespace positive_number_sum_square_eq_210_l581_581931

theorem positive_number_sum_square_eq_210 (x : ℕ) (h1 : x^2 + x = 210) (h2 : 0 < x) (h3 : x < 15) : x = 14 :=
by
  sorry

end positive_number_sum_square_eq_210_l581_581931


namespace sin_double_angle_eq_half_l581_581708

theorem sin_double_angle_eq_half (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : Real.sin (π / 2 + 2 * α) = Real.cos (π / 4 - α)) : 
  Real.sin (2 * α) = 1 / 2 :=
by
  sorry

end sin_double_angle_eq_half_l581_581708


namespace product_decreased_by_3_increases_by_15_l581_581839

theorem product_decreased_by_3_increases_by_15 :
  ∃ (a1 a2 a3 a4 a5 : ℕ), 
    let Poriginal := a1 * a2 * a3 * a4 * a5 in
    let Pnew := (a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) in
    Pnew = 15 * Poriginal :=
by {
  sorry
}

end product_decreased_by_3_increases_by_15_l581_581839


namespace transformed_sample_properties_l581_581132

noncomputable def mean (xs : List ℝ) : ℝ :=
  (xs.foldl (+) 0) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.foldl (λ sum x => sum + (x - m) ^ 2) 0) / xs.length

variables (xs : List ℝ)

theorem transformed_sample_properties
  (hmean : mean xs = 5) 
  (hvariance : variance xs = 7) :
  mean (List.map (λ x => 2 * x - 1) xs) = 9 ∧
  variance (List.map (λ x => 2 * x - 1) xs) = 28 := by
  sorry

end transformed_sample_properties_l581_581132


namespace trapezoid_rotation_volume_l581_581465

theorem trapezoid_rotation_volume (AB CD : ℝ) (angle : ℝ) (h_AB : AB = 8) (h_CD : CD = 2) (h_angle : angle = real.pi / 4) : 
  let DD1 := (AB - CD) / 2,
      height := DD1,
      D1C1 := CD in
  let V1 := real.pi * height^2 * D1C1,
      V2 := (1 / 3) * real.pi * height^2 * DD1,
      V := V1 + 2 * V2 in
  V = 36 * real.pi :=
by
  sorry

end trapezoid_rotation_volume_l581_581465


namespace count_perfect_squares_cubes_under_1000_l581_581755

-- Condition definitions
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, m * m = n
def isPerfectCube (n : Nat) : Prop := ∃ m : Nat, m * m * m = n

-- Main proof problem
theorem count_perfect_squares_cubes_under_1000 : 
  let count (p : Nat -> Prop) := Finset.card (Finset.filter p (Finset.range 1000))
  count (λ n, isPerfectSquare n ∨ isPerfectCube n) = 37 := 
sorry

end count_perfect_squares_cubes_under_1000_l581_581755


namespace train_speed_l581_581593

/--
Given:
- The speed of the first person \(V_p\) is 4 km/h.
- The train takes 9 seconds to pass the first person completely.
- The length of the train is approximately 50 meters (49.999999999999986 meters).

Prove:
- The speed of the train \(V_t\) is 24 km/h.
-/
theorem train_speed (V_p : ℝ) (t : ℝ) (L : ℝ) (V_t : ℝ) 
  (hV_p : V_p = 4) 
  (ht : t = 9)
  (hL : L = 49.999999999999986)
  (hrel_speed : (L / t) * 3.6 = V_t - V_p) :
  V_t = 24 :=
by
  sorry

end train_speed_l581_581593


namespace part1_Exi_part2_range_alpha_l581_581575

noncomputable def ProjectA : DiscreteFiniteDist ℝ :=
  discrete_fin { | (1, 1/2), (0, 1/4), (-1, 1/4) }

def E_xi : ℝ := ∑ x in ProjectA.support, x * ProjectA.prob x

theorem part1_Exi : E_xi = 0.25 :=
by
  -- unfolding definition of E_xi and ProjectA
  simp only [E_xi, ProjectA, discrete_fin]
  sorry

noncomputable def ProjectB (α β : ℝ) (h : α + β = 1) : DiscreteFiniteDist ℝ :=
  discrete_fin { | (2, α), (-2, β) }

def E_eta (α β : ℝ) (h : α + β = 1) : ℝ :=
  ∑ x in (ProjectB α β h).support, x * (ProjectB α β h).prob x

theorem part2_range_alpha (α β : ℝ) (h : α + β = 1) : (E_eta α β h ≥ 0.25) ↔ (α ≥ 9/16) :=
by
  -- unfolding definition of E_eta and ProjectB
  simp only [E_eta, ProjectB, discrete_fin]
  sorry

end part1_Exi_part2_range_alpha_l581_581575


namespace probability_nearsighted_light_phone_users_l581_581085

theorem probability_nearsighted_light_phone_users :
    (let total_students := 100
         nearsighted_students := 0.4 * total_students
         heavy_phone_users := 0.2 * total_students
         nearsighted_heavy_phone_users := 0.5 * heavy_phone_users
         light_phone_users := total_students - heavy_phone_users
         nearsighted_light_phone_users := nearsighted_students - nearsighted_heavy_phone_users
         probability := nearsighted_light_phone_users / light_phone_users
     in probability = 0.375) :=
by
    sorry

end probability_nearsighted_light_phone_users_l581_581085


namespace proof_PF_l581_581571

/-- Proof problem to solve for PF in terms of sin(φ).
Given:
- a circle centered at P with radius 2 containing point D,
- a segment DE tangent to the circle at D,
- an angle DPE = φ,
- a point F lying on the segment PD such that EF bisects the angle DPE,
we aim to prove: PF = 2 / (1 + u) where u = sin(φ). -/
theorem proof_PF (P D E F : Point) (φ u v : ℝ) (circle : Set Point)
    (center_P : ∀ Q, Q ∉ circle ↔ dist P Q ≠ 2)
    (on_circle_D : D ∈ circle)
    (tangent_DE : ∀ Q, tangent circle D Q ↔ Q = E)
    (angle_DPE_phi : ∠DPE = φ)
    (lies_on_PD_F : lies_on F (segment P D))
    (bisects_angle_EF : bisects_segment_angle E F (∠DPE))
    (u_def : u = sin φ)
    (v_def : v = cos φ) : 
    dist P F = 2 / (1 + u) := 
sorry

end proof_PF_l581_581571


namespace terry_lunch_options_l581_581912

/-- 
Terry is having lunch at a salad bar. There are four types of lettuce to choose from, 
as well as 5 types of tomatoes, 6 types of olives, 3 types of bread, and 4 types of fruit. 
He must also decide whether or not to have one of the three types of soup on the side.
This function proves the total number of options for his lunch combo.
-/
theorem terry_lunch_options
  (num_lettuce : ℕ := 4)
  (num_tomatoes : ℕ := 5)
  (num_olives : ℕ := 6)
  (num_bread : ℕ := 3)
  (num_fruit : ℕ := 4)
  (num_soup : ℕ := 3) :
  num_lettuce * num_tomatoes * num_olives * num_bread * num_fruit * num_soup = 4320 :=
by
  calc
    4 * 5 * 6 * 3 * 4 * 3 = 4320 := sorry

end terry_lunch_options_l581_581912


namespace ac_over_ap_l581_581032

noncomputable def AC_AP_ratio (AC AP : ℝ) : Prop :=
  AC / AP = 350 / 117

theorem ac_over_ap (A B C M N P : Type)
  (AM AB BC AC AP : ℝ)
  (h1 : AM / AB = 3 / 25)
  (h2 : 3 / 14 = BN / BC)
  (h3 : P = Intersection (Line AC) (Line MN)) :
  AC_AP_ratio AC AP := 
sorry

end ac_over_ap_l581_581032


namespace find_length_XZ_l581_581833

noncomputable def length_XZ (O C B Z X : ℝ) (h1 : ∠COB = 90) (h2 : OZ ⊥ CB) 
  (radius : ℝ) (h3 : OC = radius) (h4 : CB = radius) : ℝ :=
15 - (radius / 2)

theorem find_length_XZ (O C B Z X : ℝ) (h1 : ∠COB = 90) (h2 : OZ ⊥ CB) 
  (radius : ℝ) (h3 : OC = radius) (h4 : CB = radius) : 
  length_XZ O C B Z X h1 h2 radius h3 h4 = 7.5 := 
sorry

end find_length_XZ_l581_581833


namespace permutation_count_l581_581910

theorem permutation_count:
  (finset.filter (λ (l : list ℕ), 
      list.perm l [1, 2, 3, 4] ∧ 
        |(l.nth_le 0 _) - 1| + 
        |(l.nth_le 1 _) - 2| + 
        |(l.nth_le 2 _) - 3| + 
        |(l.nth_le 3 _) - 4| = 6) 
      (finset.univ : finset (list ℕ))).card = 9 := 
by { sorry }

end permutation_count_l581_581910


namespace conditional_probability_l581_581676

-- Conditions
def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def evens := {n | n ∈ numbers ∧ (n % 2 = 0)}
def odds := {n | n ∈ numbers ∧ (n % 2 = 1)}

def P_A : ℚ := 4 / 9
def P_AB : ℚ := (4 / 9) * (5 / 8)

-- Proof Problem
theorem conditional_probability : P_AB / P_A = 5 / 8 :=
by
  sorry -- Proof not required as per given instructions

end conditional_probability_l581_581676


namespace decision_to_go_to_sea_l581_581991

def expected_gain_go_to_sea (p_good_weather: ℝ) (gain_good_weather: ℝ) (loss_bad_weather: ℝ): ℝ :=
  p_good_weather * gain_good_weather + (1 - p_good_weather) * loss_bad_weather

def loss_not_going_to_sea : ℝ := -1000

theorem decision_to_go_to_sea : 
  let p_good_weather := 0.6 in
  let gain_good_weather := 6000 in
  let loss_bad_weather := -8000 in
  expected_gain_go_to_sea p_good_weather gain_good_weather loss_bad_weather > loss_not_going_to_sea :=
by
  sorry

end decision_to_go_to_sea_l581_581991


namespace men_wages_l581_581566

-- Define wages for men, women, and boys respectively
variables (wage_men wage_women wage_boys : ℝ)

-- Define the total earnings
constant total_earnings : ℝ := 60

-- Define the equality conditions
constant equal_men_women : 5 * wage_men = wage_women
constant equal_women_boys : wage_women = 8 * wage_boys

-- Defining the complete earning equation
constant combined_earnings : 3 * (5 * wage_men) = total_earnings

-- Proving the wage of one man
theorem men_wages : wage_men = 4 := sorry

end men_wages_l581_581566


namespace triangleABC_obtuse_l581_581390

noncomputable def triangleProblem (A B C : ℝ) (a b c : ℝ) (sin cos : ℝ → ℝ) : Prop :=
  a * sin A = 4 * b * sin B ∧
  a * c = sqrt 5 * (a^2 - b^2 - c^2) ∧
  (a = 2 * b ∨ (cos A = -sqrt 5 / 5) ∨ (sin B = sqrt 5 / 5)) ∧
  (cos A < 0 → A > π / 2)

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (sin cos : ℝ → ℝ)

theorem triangleABC_obtuse 
  (h1 : a * sin A = 4 * b * sin B)
  (h2 : a * c = sqrt 5 * (a^2 - b^2 - c^2)) :
  triangleProblem A B C a b c sin cos :=
by
  sorry

end triangleABC_obtuse_l581_581390


namespace winning_strategy_l581_581917

theorem winning_strategy (n m : ℕ) : 
  (even n ∨ even m → ∃ first_player_wins : Π (turn : ℕ) (king_pos : ℕ × ℕ), bool, ∀ (k : ℕ × ℕ), (king_pos = k) → turn % 2 = 0 → first_player_wins turn king_pos = tt) ∧ 
  (odd n ∧ odd m → ∃ second_player_wins : Π (turn : ℕ) (king_pos : ℕ × ℕ), bool, ∀ (k : ℕ × ℕ), (king_pos = k) → turn % 2 = 1 → second_player_wins turn king_pos = tt) := 
sorry

end winning_strategy_l581_581917


namespace concrete_slab_height_l581_581043

theorem concrete_slab_height 
  (num_homes : ℕ) (home_length home_width : ℝ)
  (density price_per_pound foundation_cost : ℝ) :
  num_homes = 3 →
  home_length = 100 →
  home_width = 100 →
  density = 150 →
  price_per_pound = 0.02 →
  foundation_cost = 45000 →
  ∃ height : ℝ, height = 0.5 :=
by {
  intro h1 h2 h3 h4 h5 h6,
  use 0.5,
  sorry
}

end concrete_slab_height_l581_581043


namespace odd_function_value_neg_range_l581_581301

-- Define that f(x) is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

theorem odd_function_value_neg_range :
  (is_odd_function f) →
  (∀ x, 0 ≤ x ∧ x ≤ 4 → f(x) = x^2 - 2*x) →
  (∀ x, -4 ≤ x ∧ x ≤ 0 → f(x) = -x^2 - 2*x) :=
by {
  intros h_odd h_range x hx,
  sorry
}

end odd_function_value_neg_range_l581_581301


namespace speed_of_boat_l581_581569

-- Given conditions
variables (V_b : ℝ) (V_s : ℝ) (T : ℝ) (D : ℝ)

-- Problem statement in Lean
theorem speed_of_boat (h1 : V_s = 5) (h2 : T = 1) (h3 : D = 45) :
  D = T * (V_b + V_s) → V_b = 40 := 
by
  intro h4
  rw [h1, h2, h3] at h4
  linarith

end speed_of_boat_l581_581569


namespace initial_iron_cars_l581_581216

noncomputable def total_delivery_time : ℕ := 100
noncomputable def travel_time_per_station : ℕ := 25
noncomputable def stops : ℕ := total_delivery_time / travel_time_per_station

noncomputable def coal_initial : ℕ := 6
noncomputable def wood_initial : ℕ := 2
noncomputable def iron_per_station : ℕ := 3

theorem initial_iron_cars : Σ (iron : ℕ), iron = iron_per_station * stops := 
  sorry

end initial_iron_cars_l581_581216


namespace arithmetic_mean_l581_581650

-- Variables and conditions
variables (x a : ℝ)
hypothesis (h₀ : x ≠ 0)
def c : ℝ := 1 - (a / (2 * x))

-- The statement to be proved
theorem arithmetic_mean (h₀ : x ≠ 0) : 
  (1 / 2) * ((x + 2 * a) / x + (x - 3 * a) / x) = 1 - (a / (2 * x)) := 
by {
  sorry
}

end arithmetic_mean_l581_581650


namespace quadratic_solutions_l581_581901

theorem quadratic_solutions (x : ℝ) : (2 * x^2 + 5 * x + 3 = 0) → (x = -1 ∨ x = -3 / 2) :=
by {
  sorry
}

end quadratic_solutions_l581_581901


namespace cos_sum_eq_l581_581810

noncomputable def calc_cos_sum (gamma delta : ℝ) : ℝ :=
  let exp_i_gamma := complex.exp (complex.I * gamma)
  let exp_i_delta := complex.exp (complex.I * delta)
  let exp_i_sum := exp_i_gamma * exp_i_delta
  exp_i_sum.re

theorem cos_sum_eq : ∀ γ δ : ℝ,
    complex.exp (complex.I * γ) = (8 / 17) + (15 / 17) * complex.I →
    complex.exp (complex.I * δ) = (3 / 5) - (4 / 5) * complex.I →
    calc_cos_sum γ δ = 84 / 85 :=
by
  intros γ δ hγ hδ
  -- Proof will be provided here
  sorry

end cos_sum_eq_l581_581810


namespace area_transformed_function_l581_581913

-- Given: The area under the graph of y = g(x) over a certain range is equal to 15 square units
def area_g (range : Set ℝ) (g : ℝ → ℝ) : ℝ := sorry

-- Define the new function
def new_function (x : ℝ) (g : ℝ → ℝ) : ℝ := 4 * g(2 * (x - 1))

-- Statement to prove
theorem area_transformed_function (range : Set ℝ) (g : ℝ → ℝ)
  (h : area_g range g = 15) :
  area_g range (new_function _ g) = 60 :=
sorry

end area_transformed_function_l581_581913


namespace winning_votes_cast_l581_581152

theorem winning_votes_cast (V : ℝ) (h1 : 0.40 * V = 280) : 0.70 * V = 490 :=
by
  sorry

end winning_votes_cast_l581_581152


namespace exists_integer_root_l581_581863

theorem exists_integer_root (a b c d : ℤ) (ha : a ≠ 0)
  (h : ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ x * (a * x^3 + b * x^2 + c * x + d) = y * (a * y^3 + b * y^2 + c * y + d)) :
  ∃ z : ℤ, a * z^3 + b * z^2 + c * z + d = 0 :=
by
  sorry

end exists_integer_root_l581_581863


namespace find_height_of_parallelepiped_l581_581885

variables {a : ℝ} {A A1 B C D M D1 : ℝ} {BD BC A1C : ℝ}

-- Given conditions
def midpoint (M A A1 : ℝ) : Prop := 2 * M = A + A1
def pairwise_perpendicular (BD MD1 A1C : ℝ) : Prop :=
  BD * MD1 = 0 ∧ BD * A1C = 0 ∧ MD1 * A1C = 0

-- Given variables
def BD_value : BD = 2 * a := sorry
def BC_value : BC = (3 / 2) * a := sorry
def A1C_value : A1C = 4 * a := sorry

-- Prove the height is 2a√2
def height_value : ℝ :=
  let h := 2 * a * Real.sqrt 2
  h

-- Statement of what to prove
theorem find_height_of_parallelepiped (h : ℝ) :
  midpoint M A A1 ∧
  pairwise_perpendicular BD MD1 A1C ∧
  BD_value ∧
  BC_value ∧
  A1C_value →
  h = 2 * a * Real.sqrt 2 :=
sorry

end find_height_of_parallelepiped_l581_581885


namespace length_of_GH_l581_581091

theorem length_of_GH
  (AB BC : ℝ)
  (hAB : AB = 6) (hBC : BC = 8) : 
  ∃ AC GH : ℝ, let AC := real.sqrt (AB^2 + BC^2) in
                let GA := (AC * AB) / BC in
                let AH := (AC * 6) / 8 in
                GH = GA + AH ∧ GH = 15 :=
by
  sorry

end length_of_GH_l581_581091


namespace first_brother_is_treljalya_l581_581837

noncomputable section

def first_brother_statement1 := "My name is Treljalya."
def second_brother_statement1 := "My name is Treljalya."
def first_brother_statement2 := "My brother has an orange-suit card."
def same_suit_implication := "If both brothers have cards of the same suit, one brother tells the truth and the other lies."
def different_suit_implication := "If the cards are of different suits, either both brothers tell the truth or both lie."

theorem first_brother_is_treljalya
  (fb_st1 : first_brother_statement1)
  (sb_st1 : second_brother_statement1)
  (fb_st2 : first_brother_statement2)
  (same_suit : same_suit_implication)
  (diff_suit : different_suit_implication)
  : "The first brother is Treljalya." :=
sorry

end first_brother_is_treljalya_l581_581837


namespace largest_sum_ABC_l581_581826

noncomputable def max_sum_ABC (A B C : ℕ) : ℕ :=
if A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2310 then
  A + B + C
else
  0

theorem largest_sum_ABC : ∃ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2310 ∧ max_sum_ABC A B C = 52 :=
sorry

end largest_sum_ABC_l581_581826


namespace brokerage_percentage_l581_581112

/--
The cash realized on selling a 14% stock is Rs. 101.25.
The total amount before brokerage is Rs. 101.
What is the percentage of the brokerage?
-/
theorem brokerage_percentage (cash_realized : ℝ) (total_before_brokerage : ℝ) (brokerage_percentage : ℝ) :
  cash_realized = 101.25 →
  total_before_brokerage = 101 →
  brokerage_percentage = ((total_before_brokerage - cash_realized) / total_before_brokerage) * 100 →
  brokerage_percentage = 0.25 :=
by
  intros h_cash_realized h_total_before h_brokerage_percentage
  rw [h_cash_realized, h_total_before, h_brokerage_percentage]
  sorry

end brokerage_percentage_l581_581112


namespace prob_at_least_one_palindrome_correct_l581_581369

-- Define a function to represent the probability calculation.
def probability_at_least_one_palindrome : ℚ :=
  let prob_digit_palindrome : ℚ := 1 / 100
  let prob_letter_palindrome : ℚ := 1 / 676
  let prob_both_palindromes : ℚ := (1 / 100) * (1 / 676)
  (prob_digit_palindrome + prob_letter_palindrome - prob_both_palindromes)

-- The theorem we are stating based on the given problem and solution:
theorem prob_at_least_one_palindrome_correct : probability_at_least_one_palindrome = 427 / 2704 :=
by
  -- We assume this step for now as we are just stating the theorem
  sorry

end prob_at_least_one_palindrome_correct_l581_581369


namespace divide_cube_into_8_smaller_cubes_l581_581254

-- Defining a cube and a function to divide it into 8 identical smaller cubes
structure Cube where
  side_length : ℝ
  volume : ℝ := side_length ^ 3

-- Defining a function that checks if a cube is divided into 8 identical smaller cubes
def divide_cube (c : Cube) : Prop :=
  let smaller_side_length := c.side_length / 2
  let smaller_volume := smaller_side_length ^ 3
  (smaller_volume * 8 = c.volume)

-- The theorem statement: A cube can be divided into 8 identical smaller cubes
theorem divide_cube_into_8_smaller_cubes (c : Cube) : divide_cube c :=
begin
  -- Implementation proof is skipped
  sorry
end

end divide_cube_into_8_smaller_cubes_l581_581254


namespace inverse_square_l581_581614

theorem inverse_square : (3⁻¹) ^ 2 = 1 / 9 :=
by sorry

end inverse_square_l581_581614


namespace dirk_sells_amulets_for_days_l581_581264

/--  
Dirk sells amulets at a Ren Faire. He sells each amulet at $40, and the cost to make each amulet is $30. 
He has to give 10% of the revenue per amulet to the faire. He sells 25 amulets each day.
If Dirk made a total profit of $300, prove that the number of days he sold amulets is 2.
-/
theorem dirk_sells_amulets_for_days
  (selling_price_per_amulet : ℤ := 40)
  (cost_price_per_amulet : ℤ := 30)
  (faire_percentage : ℤ := 10)
  (amulents_sold_per_day : ℤ := 25)
  (total_profit : ℤ := 300) :
  let profit_per_amulet := selling_price_per_amulet - cost_price_per_amulet
      faire_cut_per_amulet := (faire_percentage * selling_price_per_amulet) / 100
      net_profit_per_amulet := profit_per_amulet - faire_cut_per_amulet
      total_net_profit_per_day := net_profit_per_amulet * amulents_sold_per_day in
  (total_profit / total_net_profit_per_day) = 2 :=
by
  sorry

end dirk_sells_amulets_for_days_l581_581264


namespace solve_trig_identity_l581_581679

noncomputable def trig_identity (α : ℝ) : Prop :=
  tan α = 1 / 2 → (sin α - 3 * cos α) / (sin α + cos α) = -5 / 3

theorem solve_trig_identity (α : ℝ) (h : tan α = 1 / 2) : 
  (sin α - 3 * cos α) / (sin α + cos α) = -5 / 3 :=
by
  exact trig_identity α h
  sorry

end solve_trig_identity_l581_581679


namespace probability_of_drawing_blue_ball_additional_blue_balls_needed_l581_581020

-- Definitions based on the provided conditions
def total_balls : Nat := 30
def red_balls : Nat := 6
def yellow_balls (blue_balls : Nat) : Nat := 2 * blue_balls
def probability_of_blue_ball (blue_balls : Nat) : ℚ := blue_balls / total_balls

-- 1. Prove the probability of drawing a blue ball is 4 / 15
theorem probability_of_drawing_blue_ball (blue_balls : Nat) (h1 : red_balls + yellow_balls blue_balls + blue_balls = total_balls) 
: probability_of_blue_ball blue_balls = 4 / 15 :=
begin
  -- Proof goes here
  sorry
end

-- 2. Prove that 14 additional blue balls are needed to make the probability 1 / 2
theorem additional_blue_balls_needed (blue_balls : Nat) (additional_blue_balls : Nat) (h1 : red_balls + yellow_balls blue_balls + blue_balls = total_balls) 
: (additional_blue_balls = 14) ↔ ((blue_balls + additional_blue_balls) / (total_balls + additional_blue_balls) = 1 / 2) :=
begin
  -- Proof goes here
  sorry
end

end probability_of_drawing_blue_ball_additional_blue_balls_needed_l581_581020


namespace unique_digits_from_eight_l581_581116

-- Definitions
inductive Digit
| zero
| one
| two
| three
| four
| five
| six
| seven
| eight
| nine
| Omega

open Digit

-- Relation indicating a matchstick configuration can transform to another digit
def transforms_to : Digit -> Digit -> Prop
| eight, Omega := true
| eight, six := true
| eight, zero := true
| eight, nine := true
| eight, three := true
| eight, five := true
| _, _ := false

-- Digit Reduction Axiom
axiom digit_removal : 
  ∀ (d1 d2 : Digit), (d2 = Omega ∨ d2 = six ∨ d2 = zero ∨ d2 = nine ∨ d2 = three ∨ d2 = five) 
                     → d1 = eight 
                     → transforms_to d1 d2  

-- Statement: We can get 6 unique digits by transforming the digit 8.
theorem unique_digits_from_eight : 
  ∃ l : List Digit, l.length = 6 ∧ (∀ d ∈ l, transforms_to eight d) ∧ l.nodup := 
by
  sorry

end unique_digits_from_eight_l581_581116


namespace true_proposition_l581_581311

open Real

theorem true_proposition :
  (∃ x0 : ℝ, log x0 ≥ x0 - 1) ∧ (¬ ∀ θ : ℝ, sin θ + cos θ < 1) :=
by
  have p : ∃ x0 : ℝ, log 1 ≥ 1 - 1 := ⟨1, by norm_num [log_one]⟩
  have q : ¬ ∀ θ : ℝ, sin θ + cos θ < 1 :=
    begin
      intro h,
      specialize h (π / 4),
      rw [sin_pi_div_four, cos_pi_div_four],
      simp only [one_div, mul_self_sqrt, le_refl] at h,
      linarith,
    end
  exact ⟨p, q⟩

end true_proposition_l581_581311


namespace p_implies_q_q_not_implies_p_p_sufficient_not_necessary_l581_581695

def p (f : ℝ → ℝ) : Prop := ∀ x, f (x + Real.pi) = -f x
def q (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2 * Real.pi) = f x

theorem p_implies_q {f : ℝ → ℝ} (hp : p f) : q f :=
by sorry

theorem q_not_implies_p {f : ℝ → ℝ} (hq : q f) (hf : f = Mathlib.Real.tan) : ¬ p f :=
by sorry

theorem p_sufficient_not_necessary (f : ℝ → ℝ) : (p f → q f) ∧ (q f → ¬ p f) :=
by sorry

end p_implies_q_q_not_implies_p_p_sufficient_not_necessary_l581_581695


namespace perfect_squares_and_cubes_count_lt_1000_l581_581789

theorem perfect_squares_and_cubes_count_lt_1000 : 
  let perfect_square := λ n : ℕ, ∃ k : ℕ, n = k^2
  let perfect_cube := λ n : ℕ, ∃ k : ℕ, n = k^3
  let sixth_power := λ n : ℕ, ∃ k : ℕ, n = k^6
  (count (λ n, n < 1000 ∧ (perfect_square n ∨ perfect_cube n)) (Finset.range 1000)) = 38 :=
by {
  sorry
}

end perfect_squares_and_cubes_count_lt_1000_l581_581789


namespace angle_AOD_equals_144_l581_581381

variables (OA OD OB OC : Type) [inner_product_space ℝ OA] [inner_product_space ℝ OD] [inner_product_space ℝ OB] [inner_product_space ℝ OC]

-- Assuming vectors OA, OD, OB, and OC
variable AOD : ℝ
variable BOC : ℝ

-- Given conditions
axiom perpendicular1 : OA ⊥ OD
axiom perpendicular2 : OB ⊥ OC
axiom angle_condition : AOD = 4 * BOC

-- Goal to prove
theorem angle_AOD_equals_144 : AOD = 144 := 
sorry

end angle_AOD_equals_144_l581_581381


namespace trajectory_of_M_l581_581307

theorem trajectory_of_M (x y : ℝ) (A B M : ℝ × ℝ)
  (h1 : A = (m, 0))
  (h2 : B = (0, b))
  (h3 : AB = 2)
  (h4 : abs (AM / MB) = 1/2)
  : 9 * x^2 + 36 * y^2 = 16 :=
sorry

end trajectory_of_M_l581_581307


namespace number_of_goats_l581_581503

-- Mathematical definitions based on the conditions
def number_of_hens : ℕ := 10
def total_cost : ℤ := 2500
def price_per_hen : ℤ := 50
def price_per_goat : ℤ := 400

-- Prove the number of goats
theorem number_of_goats (G : ℕ) : 
  number_of_hens * price_per_hen + G * price_per_goat = total_cost ↔ G = 5 := 
by
  sorry

end number_of_goats_l581_581503


namespace polynomial_divisibility_l581_581866

theorem polynomial_divisibility (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
    (h3 : (a + b^3) % (a^2 + 3 * a * b + 3 * b^2 - 1) = 0) : 
    ∃ k : ℕ, k ≥ 1 ∧ (a^2 + 3 * a * b + 3 * b^2 - 1) % k^3 = 0 :=
    sorry

end polynomial_divisibility_l581_581866


namespace tangent_to_circumcircle_of_BDF_l581_581853

-- Definitions for points and cyclic quadrilateral
variable (A B C D E F B' : Type) [Field A]
variable [AddGroup A] [VectorSpace ℝ A]
variables [AddGroup B] [VectorSpace ℝ B]
variables [AddGroup C] [VectorSpace ℝ C]
variables [AddGroup D] [VectorSpace ℝ D]
variables [AddGroup E] [VectorSpace ℝ E]
variables [AddGroup F] [VectorSpace ℝ F]
variables [AddGroup B'] [VectorSpace ℝ B']

noncomputable def is_cyclic_quadrilateral (A B C D : A) := sorry
noncomputable def midpoint (A C : A) : A := sorry
noncomputable def reflect_across (B F : A) : A := sorry
noncomputable def circumcircle_of_triangle (C D E : A) : A := sorry
noncomputable def intersect (circ : A) (BC : A) : A := sorry
noncomputable def tangent (E F : A) (circ : A) : Prop := sorry

-- Problem Statement
theorem tangent_to_circumcircle_of_BDF
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_midpoint : E = midpoint A C)
  (F_ne_C : F ≠ C)
  (h_intersect : F = intersect (circumcircle_of_triangle C D E) B)
  (h_reflect: B' = reflect_across B F) :
  tangent E F (circumcircle_of_triangle B' D F) :=
sorry

end tangent_to_circumcircle_of_BDF_l581_581853


namespace cell_divisions_3_hours_20_minutes_l581_581984

theorem cell_divisions_3_hours_20_minutes :
  (n div 20 = 10) ∧ (cells = 2^(n div 20)) ∧ (hours = 3 + 20):
  cells = 1024 := 
sorry

end cell_divisions_3_hours_20_minutes_l581_581984


namespace vector_linear_dependence_l581_581647

noncomputable theory

open Function

theorem vector_linear_dependence (k : ℝ) :
    (∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a • (2 : ℝ) + b • (1 : ℝ) = 0) ∧ (a • (4 : ℝ) + b • k = 0))
    → k = 2 :=
by
  intro h
  sorry

end vector_linear_dependence_l581_581647


namespace parallel_chords_of_parabola_l581_581970

-- Define the conditions
variables {P : Type*} [preorder P] -- P represents the space
variables {A B C D E F : P}         -- Points A, B, C, D, E, F
variables {S1 S2 : set P}          -- Circles S1 and S2

-- Define the main condition: two parallel chords AB and CD of a parabola
variable (is_parallel : ∀ (x y z: P), (x = A ∧ y = B) ∨ (x = C ∧ y = D) → z is_parallel_to x y)

-- Circles through points A, B, and C, D intersect at points E and F
variable (circle_intersects: (E ∈ S1 ∧ E ∈ S2) ∧ (F ∈ S1 ∧ F ∈ S2))

-- Given that E is on the parabola, prove that F is on the parabola
theorem parallel_chords_of_parabola (hE: E ∈ parabola P):
  F ∈ parabola P :=
sorry

end parallel_chords_of_parabola_l581_581970


namespace find_function_satisfying_property_l581_581643

noncomputable def example_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + y^2) = f (x^2 - y^2) + f (2 * x * y)

theorem find_function_satisfying_property (f : ℝ → ℝ) (h : ∀ x, 0 ≤ f x) (hf : example_function f) :
  ∃ a : ℝ, 0 ≤ a ∧ ∀ x : ℝ, f x = a * x^2 :=
sorry

end find_function_satisfying_property_l581_581643


namespace zoo_total_animals_l581_581225

def zoo_enclosures (tiger_enclosures zebra_enclosures elephant_enclosures giraffe_enclosures rhino_enclosures : ℕ)
    (animals_in_tiger_enclosure animals_in_zebra_enclosure animals_in_elephant_enclosure animals_in_giraffe_enclosure animals_in_rhino_enclosure : ℕ) : ℕ :=
  (tiger_enclosures * animals_in_tiger_enclosure) +
  (zebra_enclosures * animals_in_zebra_enclosure) +
  (elephant_enclosures * animals_in_elephant_enclosure) +
  (giraffe_enclosures * animals_in_giraffe_enclosure) +
  (rhino_enclosures * animals_in_rhino_enclosure)

theorem zoo_total_animals : 
  let tiger_enclosures := 4,
      zebra_enclosures := 4 * 2,
      elephant_enclosures := zebra_enclosures + 1,
      giraffe_enclosures := elephant_enclosures * 3,
      rhino_enclosures := 4,
      animals_in_tiger_enclosure := 4,
      animals_in_zebra_enclosure := 10,
      animals_in_elephant_enclosure := 3,
      animals_in_giraffe_enclosure := 2,
      animals_in_rhino_enclosure := 1 in
  zoo_enclosures tiger_enclosures zebra_enclosures elephant_enclosures giraffe_enclosures rhino_enclosures 
                 animals_in_tiger_enclosure animals_in_zebra_enclosure animals_in_elephant_enclosure animals_in_giraffe_enclosure animals_in_rhino_enclosure = 181 :=
by
  sorry

end zoo_total_animals_l581_581225


namespace count_perfect_squares_and_cubes_l581_581740

theorem count_perfect_squares_and_cubes : 
  let num_squares := 31 in
  let num_cubes := 9 in
  let num_sixth_powers := 3 in
  num_squares + num_cubes - num_sixth_powers = 37 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581740


namespace domain_of_sqrt_f_l581_581469

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * x - x ^ 2

theorem domain_of_sqrt_f : 
  {x : ℝ | f x ≥ 0} = set.Icc (-3 : ℝ) 1 :=
sorry

end domain_of_sqrt_f_l581_581469


namespace melindas_math_textbooks_probability_l581_581436

def total_ways_to_arrange_textbooks : ℕ :=
  (Nat.choose 15 4) * (Nat.choose 11 5) * (Nat.choose 6 6)

def favorable_ways (b : ℕ) : ℕ :=
  match b with
  | 4 => (Nat.choose 11 0) * (Nat.choose 11 5) * (Nat.choose 6 6)
  | 5 => (Nat.choose 11 1) * (Nat.choose 10 4) * (Nat.choose 6 6)
  | 6 => (Nat.choose 11 2) * (Nat.choose 9 4) * (Nat.choose 5 5)
  | _ => 0

def total_favorable_ways : ℕ :=
  favorable_ways 4 + favorable_ways 5 + favorable_ways 6

theorem melindas_math_textbooks_probability :
  let m := 1
  let n := 143
  Nat.Gcd m n = 1 ∧ total_ways_to_arrange_textbooks = 1387386 ∧ total_favorable_ways = 9702
  → m + n = 144 := by
sory

end melindas_math_textbooks_probability_l581_581436


namespace arithmetic_sqrt_25_l581_581109

-- Define the arithmetic square root condition
def is_arithmetic_sqrt (x a : ℝ) : Prop :=
  0 ≤ x ∧ x^2 = a

-- Lean statement to prove the arithmetic square root of 25 is 5
theorem arithmetic_sqrt_25 : is_arithmetic_sqrt 5 25 :=
by 
  sorry

end arithmetic_sqrt_25_l581_581109


namespace probability_M_in_triangle_FA_l581_581333

-- Given conditions
variables (p : ℝ) (p_pos : p > 0)
def parabola (y : ℝ) : set (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }

-- Focus and directrix
def focus : ℝ × ℝ := (p / 2, 0)
def directrix : ℝ × ℝ := (-(p / 2), 0)

-- Points A and B on the parabola
variables (A B : ℝ × ℝ)
hypothesis (HA : A ∈ parabola p) (HB : B ∈ parabola p)
hypothesis (Hline : ∃ m b : ℝ, ∀ x y, y = m*x + b → (A = (x, y) ∨ B = (x, y)) ∧ (focus p) = (x, y))

-- Distance AB
hypothesis (HAB : real.dist A B = 3 * p)

-- Projections A' and B' on the directrix
def proj (P : ℝ × ℝ) : ℝ × ℝ := (- (p / 2), P.2)
def A' := proj A
def B' := proj B

-- Write the Lean theorem statement
theorem probability_M_in_triangle_FA'B' :
  let quadrilateral_area := sorry -- Area of quadrilateral AA'B'B
  let triangle_area := sorry -- Area of triangle FA'B'
  let P := triangle_area / quadrilateral_area in
  quadrilateral_area ≠ 0 → P = 1 / 3 :=
sorry

end probability_M_in_triangle_FA_l581_581333


namespace greatest_consecutive_integers_sum_55_l581_581524

theorem greatest_consecutive_integers_sum_55 :
  ∃ N a : ℤ, (N * (2 * a + N - 1)) = 110 ∧ (∀ M a' : ℤ, (M * (2 * a' + M - 1)) = 110 → N ≥ M) :=
sorry

end greatest_consecutive_integers_sum_55_l581_581524


namespace swimming_pool_time_l581_581591

theorem swimming_pool_time 
  (empty_rate : ℕ) (fill_rate : ℕ) (capacity : ℕ) (final_volume : ℕ) (t : ℕ)
  (h_empty : empty_rate = 120 / 4) 
  (h_fill : fill_rate = 120 / 6) 
  (h_capacity : capacity = 120) 
  (h_final : final_volume = 90) 
  (h_eq : capacity - (empty_rate - fill_rate) * t = final_volume) :
  t = 3 := 
sorry

end swimming_pool_time_l581_581591


namespace six_digit_numbers_divisible_by_six_l581_581164

theorem six_digit_numbers_divisible_by_six : 
  let digits := [2, 3, 9] in
  let last_digit := 2 in
  let is_valid (n : ℕ) := 
    let digits_list := [2, 2, 2, 2, 2, 2] ++ 
                       [0, 2^3 - 1,  2, 5^2 - 1] in
    last_digit = 2 ∧ 
    (digits_list.sum + last_digit) % 3 = 0 ∧ 
    last_digit ∈ digits ∧
    n = 81 
  in is_valid 81 := sorry

end six_digit_numbers_divisible_by_six_l581_581164


namespace valid_paths_A_to_B_l581_581438

theorem valid_paths_A_to_B : 
  let total_paths : ℕ := Nat.choose 12 3 in
  let forbidden_paths : ℕ := 2 * (Nat.choose 5 2 * Nat.choose 7 1) in
  total_paths - forbidden_paths = 80 :=
by {
  let total_paths := Nat.choose 12 3,
  let forbidden_paths := 2 * (Nat.choose 5 2 * Nat.choose 7 1),
  have total_paths_correct : total_paths = 220 := by sorry, -- Proof of total paths calculation
  have forbidden_paths_correct : forbidden_paths = 140 := by sorry, -- Proof of forbidden paths calculation
  calc
    total_paths - forbidden_paths
        = 220 - 140 : by rw [total_paths_correct, forbidden_paths_correct]
    ... = 80         : by norm_num
}

end valid_paths_A_to_B_l581_581438


namespace simplify_expression_l581_581244

theorem simplify_expression (a b : ℚ) (h : a ≠ b) : 
  a^2 / (a - b) + (2 * a * b - b^2) / (b - a) = a - b :=
by
  sorry

end simplify_expression_l581_581244


namespace log_5_125_l581_581804

theorem log_5_125 : ∀ (a x N : ℝ), a^x = N ∧ a > 0 ∧ a ≠ 1 → (5^3 = 125 → log 5 125 = 3) :=
begin
  intros a x N h,
  assume h₁,
  -- proof goes here
  sorry
end

end log_5_125_l581_581804


namespace sin_double_angle_identity_l581_581701

theorem sin_double_angle_identity 
  (α : ℝ) 
  (h₁ : α ∈ Set.Ioo (Real.pi / 2) Real.pi) 
  (h₂ : Real.sin α = 1 / 5) : 
  Real.sin (2 * α) = - (4 * Real.sqrt 6) / 25 :=
by
  sorry

end sin_double_angle_identity_l581_581701


namespace minimum_tournament_cost_l581_581065

theorem minimum_tournament_cost (k : ℕ) (h : 0 < k) : 
  let Σ := (2 * k : ℕ) in
  let matches := (Σ * (Σ - 1)) / 2 in
  let cost := ∑ i in finset.range (2 * k), (matches - 2 * (i * (i - 1) / 2)) + (if i > k then (((2 * k - i) ^ 2) - (2 * (i * i - 1) / 2)) else 0) in
  cost = (k * (4 * k^2 + k - 1)) / 2 :=
sorry

end minimum_tournament_cost_l581_581065


namespace total_pens_left_l581_581995

def initial_blue_pens := 9
def removed_blue_pens := 4
def initial_black_pens := 21
def removed_black_pens := 7
def initial_red_pens := 6

def remaining_blue_pens := initial_blue_pens - removed_blue_pens
def remaining_black_pens := initial_black_pens - removed_black_pens
def remaining_red_pens := initial_red_pens

def total_remaining_pens := remaining_blue_pens + remaining_black_pens + remaining_red_pens

theorem total_pens_left : total_remaining_pens = 25 :=
by
  -- Proof will be provided here
  sorry

end total_pens_left_l581_581995


namespace calc_nabla_l581_581803

-- Define the new operation ∇
def nabla (a b : ℝ) : ℝ :=
  (a + b) / (1 + a * b)

-- Main theorem to prove
theorem calc_nabla : nabla 1 (nabla 2 (nabla 3 4)) = 1 := 
by
  sorry  -- The proof will go here

end calc_nabla_l581_581803


namespace allocation_methods_count_l581_581563

theorem allocation_methods_count :
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  ∃ (allocation_methods : ℕ), allocation_methods = 12 := 
by
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  use doctors * Nat.choose nurses 2
  sorry

end allocation_methods_count_l581_581563


namespace melissa_points_per_game_l581_581073

variable (t g p : ℕ)

theorem melissa_points_per_game (ht : t = 36) (hg : g = 3) : p = t / g → p = 12 :=
by
  intro h
  sorry

end melissa_points_per_game_l581_581073


namespace find_linear_function_point_not_on_function_l581_581320

-- Definitions of given points
structure Point where
  x : ℝ
  y : ℝ

def M : Point := { x := 1, y := 3 }
def N : Point := { x := -2, y := 12 }

-- Linear function passing through M and N
def linear_function (k b : ℝ) := λ x : ℝ, k * x + b

-- Proving the function y = -3x + 6
theorem find_linear_function : 
  ∃ (k b : ℝ), (linear_function k b M.x = M.y) ∧ (linear_function k b N.x = N.y) ∧ (k = -3) ∧ (b = 6) := 
by
  sorry

-- Given a point P(2a, -6a + 8)
def P (a : ℝ) : Point := { x := 2 * a, y := -6 * a + 8 }

-- Proving P is not on the function y = -3x + 6 for all a
theorem point_not_on_function (a : ℝ) : 
  P a.y ≠ linear_function (-3) 6 (P a).x := 
by
  sorry

end find_linear_function_point_not_on_function_l581_581320


namespace ratio_of_girls_to_boys_l581_581621

-- Define conditions
def num_boys : ℕ := 40
def children_per_counselor : ℕ := 8
def num_counselors : ℕ := 20

-- Total number of children
def total_children : ℕ := num_counselors * children_per_counselor

-- Number of girls
def num_girls : ℕ := total_children - num_boys

-- The ratio of girls to boys
def girls_to_boys_ratio : ℚ := num_girls / num_boys

-- The theorem we need to prove
theorem ratio_of_girls_to_boys : girls_to_boys_ratio = 3 := by
  sorry

end ratio_of_girls_to_boys_l581_581621


namespace sum_of_odd_divisors_of_252_l581_581532

theorem sum_of_odd_divisors_of_252 : 
  let prime_factors := (2^2, 3^2, 7)
  let factors := {d : ℕ | d ∣ 252 ∧ d % 2 = 1}
  ∑ d in factors, d = 104 := sorry

end sum_of_odd_divisors_of_252_l581_581532


namespace uniform_distribution_of_random_simulation_l581_581449

/-- 
Given that random simulation methods generate real numbers on an interval, 
prove that these numbers are uniformly distributed.
-/
theorem uniform_distribution_of_random_simulation (interval : Set ℝ) (random_simulation : ℝ → Prop) :
  (∀ (x : ℝ), x ∈ interval → random_simulation x) → 
  ∃ (dist : MeasureTheory.ProbabilityMeasure ℝ), dist.Uniform interval := 
sorry

end uniform_distribution_of_random_simulation_l581_581449


namespace find_x_l581_581002

def operation (x y : ℝ) : ℝ := 2 * x * y

theorem find_x (x : ℝ) :
  operation 6 (operation 4 x) = 480 ↔ x = 5 := 
by
  sorry

end find_x_l581_581002


namespace factor_cubic_expression_l581_581487

-- The goal is to prove the factored form of the given polynomial
theorem factor_cubic_expression (a : ℝ) : a^3 - 4a = a * (a + 2) * (a - 2) :=
by
  sorry

end factor_cubic_expression_l581_581487


namespace ker_is_normal_l581_581069

open GroupTheory

variable {G H : Type}
variable [Group G] [Group H]

def Ker (φ : G →* H) : Subgroup G :=
{ carrier := { g | φ g = 1 },
  one_mem' := by simp,
  mul_mem' := by intros a b ha hb; simp [ha, hb],
  inv_mem' := by intro a ha; simp [ha] }

theorem ker_is_normal (φ : G →* H) : (Ker φ).Normal :=
begin
  sorry
end

end ker_is_normal_l581_581069


namespace tetrahedron_insphere_radii_sum_l581_581924

theorem tetrahedron_insphere_radii_sum (r r1 r2 r3 r4 : ℝ) 
    (cond1 : ∀ (A B C D : ℝ), insphere_radius A B C D = r)
    (cond2 : ∀ (r₁ r₂ r₃ r₄ : ℝ), smaller_tetrahedrons_radii r₁ r₂ r₃ r₄) :
    r1 + r2 + r3 + r4 = 2 * r := 
sorry

end tetrahedron_insphere_radii_sum_l581_581924


namespace area_of_triangle_F₁PF₂_coordinates_of_P_l581_581050

-- Definitions for the conditions
def ellipse (x y : ℝ) : Prop := (x ^ 2) / 25 + (y ^ 2) / 9 = 1

def foci_F₁ := (-4, 0 : ℝ)
def foci_F₂ := (4, 0 : ℝ)
def angle_F₁PF₂ (P : ℝ × ℝ) := 60 * (Real.pi / 180)

-- Main propositions
theorem area_of_triangle_F₁PF₂ (P : ℝ × ℝ) (hP : ellipse P.1 P.2) (hAngle : angle_F₁PF₂ P = Real.pi / 3) :
  let F₁ := foci_F₁;
      F₂ := foci_F₂;
      m := Real.dist P F₁;
      n := Real.dist P F₂
  in m + n = 10 ∧ 3 * Real.sqrt 3 = 3 * Real.sqrt 3 :=
sorry

theorem coordinates_of_P (P : ℝ × ℝ) (hP : ellipse P.1 P.2) (hAngle : angle_F₁PF₂ P = Real.pi / 3) :
  P = (5 * Real.sqrt 13 / 4, Real.sqrt 3) ∨
  P = (-5 * Real.sqrt 13 / 4, Real.sqrt 3) ∨
  P = (5 * Real.sqrt 13 / 4, -Real.sqrt 3) ∨
  P = (-5 * Real.sqrt 13 / 4, -Real.sqrt 3) :=
sorry

end area_of_triangle_F₁PF₂_coordinates_of_P_l581_581050


namespace how_many_eyes_do_I_see_l581_581350

def boys : ℕ := 23
def eyes_per_boy : ℕ := 2
def total_eyes : ℕ := boys * eyes_per_boy

theorem how_many_eyes_do_I_see : total_eyes = 46 := by
  sorry

end how_many_eyes_do_I_see_l581_581350


namespace problem_proof_l581_581054

-- Defining the functions f and g as provided in the conditions
def f (x : ℚ) : ℚ := (2 * x^2 + 4 * x + 9) / (x^2 + x + 3)
def g (x : ℚ) : ℚ := x^2 - 1

-- The goal is to prove that f(g(3)) + g(f(3)) = 601 / 75
theorem problem_proof : f(g(3)) + g(f(3)) = 601 / 75 := sorry

end problem_proof_l581_581054


namespace time_to_cross_l581_581961

-- Definitions based on the conditions
def speed_faster_train := 150 -- km/hr
def speed_slower_train := 90 -- km/hr
def length_faster_train := 1.10 -- km
def length_slower_train := 0.9 -- km

-- Proof statement
theorem time_to_cross : 
  let relative_speed := (speed_faster_train + speed_slower_train) * 1000 / 3600 in 
  let combined_length := (length_faster_train + length_slower_train) * 1000 in 
  (combined_length / relative_speed) = 30 := 
by
  sorry

end time_to_cross_l581_581961


namespace cat_food_weight_l581_581439

theorem cat_food_weight (x : ℝ) :
  let bags_of_cat_food := 2
  let bags_of_dog_food := 2
  let ounces_per_pound := 16
  let total_ounces_of_pet_food := 256
  let dog_food_extra_weight := 2
  (ounces_per_pound * (bags_of_cat_food * x + bags_of_dog_food * (x + dog_food_extra_weight))) = total_ounces_of_pet_food
  → x = 3 :=
by
  sorry

end cat_food_weight_l581_581439


namespace total_bottles_needed_l581_581041

-- Definitions from conditions
def large_bottle_capacity : ℕ := 450
def small_bottle_capacity : ℕ := 45
def extra_large_bottle_capacity : ℕ := 900

-- Theorem statement
theorem total_bottles_needed :
  ∃ (num_large_bottles num_small_bottles : ℕ), 
    num_large_bottles * large_bottle_capacity + num_small_bottles * small_bottle_capacity = extra_large_bottle_capacity ∧ 
    num_large_bottles + num_small_bottles = 2 :=
by
  sorry

end total_bottles_needed_l581_581041


namespace perfect_squares_and_cubes_count_lt_1000_l581_581783

theorem perfect_squares_and_cubes_count_lt_1000 : 
  let perfect_square := λ n : ℕ, ∃ k : ℕ, n = k^2
  let perfect_cube := λ n : ℕ, ∃ k : ℕ, n = k^3
  let sixth_power := λ n : ℕ, ∃ k : ℕ, n = k^6
  (count (λ n, n < 1000 ∧ (perfect_square n ∨ perfect_cube n)) (Finset.range 1000)) = 38 :=
by {
  sorry
}

end perfect_squares_and_cubes_count_lt_1000_l581_581783


namespace compute_f_neg3_plus_f_0_l581_581694

-- Define the odd function f with the given properties
noncomputable def f : ℝ → ℝ := λ x, if 0 < x ∧ x ≤ 1 then |x| + 1 / x else if x ≥ 0 then 0 else -f (-x)

-- Prove that f(-3) + f(0) = -2 under the given conditions
theorem compute_f_neg3_plus_f_0 :
  (λ f : ℝ → ℝ, (f(-3) + f(0) = -2)) f := 
by
  sorry

end compute_f_neg3_plus_f_0_l581_581694


namespace cornelia_european_countries_l581_581252

def total_countries : Nat := 42
def south_american_countries : Nat := 10
def asian_countries : Nat := 6

def non_european_countries : Nat :=
  south_american_countries + 2 * asian_countries

def european_countries : Nat :=
  total_countries - non_european_countries

theorem cornelia_european_countries :
  european_countries = 20 := by
  sorry

end cornelia_european_countries_l581_581252


namespace find_certain_number_l581_581567

theorem find_certain_number (x : ℝ) (h : 34 = (4/5) * x + 14) : x = 25 :=
by
  sorry

end find_certain_number_l581_581567


namespace sin_plus_cos_value_l581_581700

theorem sin_plus_cos_value (x : ℝ) (hx₁ : 0 < x) (hx₂ : x < π / 2) (h : tan (x - π / 4) = -1 / 7) :
  sin x + cos x = 7 / 5 :=
sorry

end sin_plus_cos_value_l581_581700


namespace smallest_n_for_roots_of_unity_l581_581947

theorem smallest_n_for_roots_of_unity :
  ∃ n : ℕ, (n > 0) ∧ (∀ z : ℂ, (z ^ 6 - z ^ 3 + 1 = 0) → (∃ k : ℤ, z = complex.exp (2 * real.pi * complex.I * k / n))) ∧ n = 18 :=
sorry

end smallest_n_for_roots_of_unity_l581_581947


namespace sequence_not_periodic_digit_1000_is_one_position_10000th_one_position_formula_ones_position_formula_zeros_l581_581629

-- Definitions based on the conditions of the problem
inductive Seq : Type
| Zero : Seq
| Expand : Seq → Seq

def start_seq := Seq.Zero

def expand : Seq → Seq
| Seq.Zero := Seq.Expand Seq.Zero
| Seq.Expand s := Seq.expand s

-- 1. The sequence is not periodic
theorem sequence_not_periodic 
  (s : Seq) (start_seq = Seq.Zero) (H : ∀ s, s ≠ expand (expand s)) : 
  ¬ periodic s := sorry

-- 2. The 1000th digit is 1
theorem digit_1000_is_one : 
  (digit_nth start_seq 1000) = 1 := sorry

-- 3. The position of the 10,000th '1' is 34142
theorem position_10000th_one : 
  (nth_one_position start_seq 10000) = 34142 := sorry

-- 4. Formulas for positions of '1' and '0' in the sequence
theorem position_formula_ones :
  ∀ n, position_nth_one n = floor ((2 + sqrt 2) * n) := sorry 

theorem position_formula_zeros :
  ∀ n, position_nth_zero n = floor (sqrt 2 * n) := sorry

end sequence_not_periodic_digit_1000_is_one_position_10000th_one_position_formula_ones_position_formula_zeros_l581_581629


namespace proof_problem_l581_581858

-- Given conditions
variables {a b c : ℕ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variables (h4 : a > b) (h5 : a^2 - a * b - a * c + b * c = 7)

-- Statement to prove
theorem proof_problem : a - c = 1 ∨ a - c = 7 :=
sorry

end proof_problem_l581_581858


namespace functional_equation_solution_l581_581644

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (y * f (x + y) + f x) = 4 * x + 2 * y * f (x + y)) →
  (∀ x : ℝ, f x = 2 * x) :=
sorry

end functional_equation_solution_l581_581644


namespace volunteers_to_communities_l581_581286

theorem volunteers_to_communities :
  let volunteers := 5 in
  let communities := 3 in
  let at_least_in_A := 2 in
  let at_least_in_B := 1 in
  let at_least_in_C := 1 in
  ∃ (arrangements : ℕ),
    arrangements = 80 ∧ arrangements_int satisfies_conditions volunteers at_least_in_A at_least_in_B at_least_in_C :=
sorry

end volunteers_to_communities_l581_581286


namespace perimeter_of_square_l581_581108

theorem perimeter_of_square (area : ℝ) (h : area = 500 / 3) : 
  let s := sqrt (500 / 3)
  let perimeter := 4 * s
  perimeter = 40 * sqrt 15 / 3 :=
by
  sorry

end perimeter_of_square_l581_581108


namespace total_pens_left_l581_581994

def initial_blue_pens := 9
def removed_blue_pens := 4
def initial_black_pens := 21
def removed_black_pens := 7
def initial_red_pens := 6

def remaining_blue_pens := initial_blue_pens - removed_blue_pens
def remaining_black_pens := initial_black_pens - removed_black_pens
def remaining_red_pens := initial_red_pens

def total_remaining_pens := remaining_blue_pens + remaining_black_pens + remaining_red_pens

theorem total_pens_left : total_remaining_pens = 25 :=
by
  -- Proof will be provided here
  sorry

end total_pens_left_l581_581994


namespace sum_of_odd_divisors_of_252_l581_581533

theorem sum_of_odd_divisors_of_252 : 
  let prime_factors := (2^2, 3^2, 7)
  let factors := {d : ℕ | d ∣ 252 ∧ d % 2 = 1}
  ∑ d in factors, d = 104 := sorry

end sum_of_odd_divisors_of_252_l581_581533


namespace perimeter_decrease_l581_581488

-- Define original lengths
variable (a b : Real)

-- Define the initial conditions
def initial_length_reduction (a : Real) : Real := 0.9 * a
def initial_width_reduction (b : Real) : Real := 0.8 * b

-- Equation representing the initial reduction by 12%
def initial_perimeter_reduction (a b : Real) : Prop :=
    2 * (0.9 * a + 0.8 * b) = 0.88 * 2 * (a + b)

-- Derived relationship a = 4b
axiom relation_a_4b (a b : Real) : a = 4 * b

-- Define new reductions
def new_length_reduction (a : Real) : Real := 0.8 * a
def new_width_reduction (b : Real) : Real := 0.9 * b

-- Define the original perimeter
def original_perimeter (a b : Real) : Real := 2 * (a + b)

-- Define the new perimeter after new reductions
def new_perimeter (a b : Real) : Real := 2 * (new_length_reduction a + new_width_reduction b)

-- Prove the percentage decrease in perimeter
theorem perimeter_decrease (a b : Real) 
    (H : initial_perimeter_reduction a b)
    (Hr : relation_a_4b a b) : 
    (new_perimeter a b) = 0.82 * (original_perimeter a b) :=
sorry

end perimeter_decrease_l581_581488


namespace area_transformation_l581_581061

-- Define the matrix
def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, -2], ![4, 1]]

-- Define the original area
def area_S : ℝ := 5

-- Define the area of the transformed region
def area_S' : ℝ := 55

-- State the theorem/proof problem
theorem area_transformation :
  let det_A := Matrix.det matrix_A in
  let scaled_area := det_A * area_S in
  abs det_A = 11 → scaled_area = area_S' := 
by
  let det_A := Matrix.det matrix_A
  let scaled_area := det_A * area_S
  sorry

end area_transformation_l581_581061


namespace expression_value_l581_581543

-- Define the expression based on the given arithmetic sequence and operations
def expr : ℝ :=
  List.foldl (*) 1 ((List.range' 2 52 2).map (λ n, 
    if n % 4 = 2 then n / (n + 2)
    else (n - 2) / n))

-- The theorem states that the computed expression equals 1 / 26
theorem expression_value : 
  expr = 1 / 26 :=
  sorry

end expression_value_l581_581543


namespace vector_projection_equiv_l581_581343

variables {ℝ : Type*} [inner_product_space ℝ ℝ]

-- Condition definitions
def vector_a : ℝ := 2
def vector_b : ℝ := 3
def theta : ℝ := real.rad_of_deg 120

-- Calculate the dot product
noncomputable def dot_product_ab := vector_a * vector_b * real.cos theta

-- Prove the projection formula
theorem vector_projection_equiv :
  let va := ∥vector_a∥,
      vb := ∥vector_b∥,
      proj := (2*vector_a + 3*vector_b),
      dir := (2*vector_a + vector_b).
  let dot_prod := (2*va + 3*vb) ∙ (2*va + vb),
  let norm_dir := ∥2*va + vb∥,
  let projection := (dot_prod / norm_dir) in
  projection = (19 * real.sqrt 13) / 13 := 
by
  sorry

end vector_projection_equiv_l581_581343


namespace krishan_money_l581_581138

theorem krishan_money 
  (R G K : ℝ)
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (h3 : R = 490) : K = 2890 :=
sorry

end krishan_money_l581_581138


namespace find_height_squared_of_trapezoid_l581_581388

theorem find_height_squared_of_trapezoid 
  (PQ RS QR PS : ℝ)
  (PQ_val : PQ = Real.sqrt 17)
  (PS_val : PS = Real.sqrt 1369)
  (QR_perp_PQ : QR ⊥ PQ)
  (QR_perp_RS : QR ⊥ RS)
  (PS_perp_QR : PS ⊥ QR) :
  QR^2 = 272 :=
by 
  sorry

end find_height_squared_of_trapezoid_l581_581388


namespace jane_final_score_l581_581018

theorem jane_final_score :
  ∀ (correct incorrect unanswered total : ℕ)
    (points_correct points_incorrect points_unanswered : ℤ),
  correct = 15 →
  incorrect = 10 →
  unanswered = 5 →
  total = 30 →
  points_correct = 2 →
  points_incorrect = -1 →
  points_unanswered = 0 →
  correct + incorrect + unanswered = total →
  correct * points_correct + incorrect * points_incorrect + unanswered * points_unanswered = 20 :=
by
  -- Import the required Lean 4 definitions and add the theorem statement
  intro correct incorrect unanswered total points_correct points_incorrect points_unanswered
  intros h_correct h_incorrect h_unanswered h_total h_points_correct h_points_incorrect h_points_unanswered h_sum_questions
  rw [h_correct, h_incorrect, h_unanswered, h_total, h_points_correct, h_points_incorrect, h_points_unanswered, h_sum_questions]
  simp
  sorry

end jane_final_score_l581_581018


namespace q_one_eq_five_l581_581238

variable (q : ℝ → ℝ)
variable (h : q 1 = 5)

theorem q_one_eq_five : q 1 = 5 :=
by sorry

end q_one_eq_five_l581_581238


namespace probability_of_perfect_square_or_multiple_of_5_l581_581154

noncomputable def perfect_square_or_multiple_of_5_probability : ℚ :=
  let dice := {1, 2, 3, 4, 5, 6}
  let sum_count := (dice.product dice).product dice.count
    (λ t, let (d1, (d2, d3)) := t in
          let s := d1 + d2 + d3 in
          s = 4 ∨ s = 5 ∨ s = 9 ∨ s = 10 ∨ s = 15 ∨ s = 16)
  sum_count / 216

theorem probability_of_perfect_square_or_multiple_of_5 :
  perfect_square_or_multiple_of_5_probability = 77 / 216 := by
  sorry

end probability_of_perfect_square_or_multiple_of_5_l581_581154


namespace james_paid_110_l581_581038

theorem james_paid_110 :
  ∀ (packs : ℕ) (weight_per_pack : ℕ) (price_per_pound : ℚ),
  packs = 5 → 
  weight_per_pack = 4 → 
  price_per_pound = 5.50 → 
  (packs * weight_per_pack) * price_per_pound = 110 := 
by 
  intros packs weight_per_pack price_per_pound hpacks hweight hprice
  rw [hpacks, hweight, hprice]
  norm_num
  sorry

end james_paid_110_l581_581038


namespace longest_and_shortest_chords_sum_l581_581326

noncomputable def circle_eq : ∀ (x y : ℝ), x^2 + y^2 + 6*x - 8*y = 0 :=
begin
  intros,
  sorry
end

noncomputable def point_M : ℝ × ℝ := (-3, 5)

noncomputable def center_and_radius : ∃ (h : ℝ × ℝ) (r : ℝ),
  (∀ (x y : ℝ), (x - h.1)^2 + (y - h.2)^2 = r^2 ↔ x^2 + y^2 + 6*x - 8*y = 0) :=
begin
  use (-3, 4),
  use 5,
  intros,
  sorry
end

lemma chord_lengths : ∀ AC BD : ℝ, AC = 10 → BD = 4 * Real.sqrt 6 → AC + BD = 10 + 4 * Real.sqrt 6 := 
begin
  intros,
  sorry
end

theorem longest_and_shortest_chords_sum : 
  (AC BD : ℝ) → AC = 10 → BD = 4 * Real.sqrt 6 → |AC| + |BD| = 10 + 4 * Real.sqrt 6 :=
by exact chord_lengths

end longest_and_shortest_chords_sum_l581_581326


namespace correlation_is_1_3_4_l581_581922

def relationship1 := "The relationship between a person's age and their wealth"
def relationship2 := "The relationship between a point on a curve and its coordinates"
def relationship3 := "The relationship between apple production and climate"
def relationship4 := "The relationship between the diameter of the cross-section and the height of the same type of tree in a forest"

def isCorrelation (rel: String) : Bool :=
  if rel == relationship1 ∨ rel == relationship3 ∨ rel == relationship4 then true else false

theorem correlation_is_1_3_4 :
  {relationship1, relationship3, relationship4} = {r | isCorrelation r = true} := 
by
  sorry

end correlation_is_1_3_4_l581_581922


namespace pond_length_l581_581374

theorem pond_length (Width Depth Volume : ℝ) 
  (h1 : Width = 10) (h2 : Depth = 5) (h3 : Volume = 1000) : 
  ∃ Length, Length = 20 :=
by
  have Length := Volume / (Width * Depth) 
  have h4 : Length = 20 := by
    rw [h3, h1, h2]
    norm_num
  exact ⟨Length, h4⟩

end pond_length_l581_581374


namespace correct_average_l581_581184

theorem correct_average 
(n : ℕ) (avg1 avg2 avg3 : ℝ): 
  n = 10 
  → avg1 = 40.2 
  → avg2 = avg1
  → avg3 = avg1
  → avg1 = avg3 :=
by 
  intros hn h_avg1 h_avg2 h_avg3
  sorry

end correct_average_l581_581184


namespace tan_value_l581_581717

open Real

noncomputable def geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

noncomputable def arithmetic_seq (b : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem tan_value
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : geometric_seq a)
  (hb : arithmetic_seq b)
  (h_geom : a 0 * a 5 * a 10 = -3 * sqrt 3)
  (h_arith : b 0 + b 5 + b 10 = 7 * π) :
  tan ((b 2 + b 8) / (1 - a 3 * a 7)) = -sqrt 3 :=
sorry

end tan_value_l581_581717


namespace total_cost_of_bill_l581_581105

def original_price_curtis := 16.00
def original_price_rob := 18.00
def time_of_meal := 3

def is_early_bird_discount_applicable (time : ℕ) : Prop :=
  2 ≤ time ∧ time ≤ 4

theorem total_cost_of_bill :
  is_early_bird_discount_applicable time_of_meal →
  original_price_curtis / 2 + original_price_rob / 2 = 17.00 :=
by
  sorry

end total_cost_of_bill_l581_581105


namespace solve_valid_a_range_l581_581362

noncomputable def valid_a_range : Set ℝ :=
  {a | ∀ x : ℝ, x^2 - 2 * a * x + 1 < 0 → (-∞ < a ∧ a < -1) ∨ (1 < a ∧ a ≤ ∞)}

theorem solve_valid_a_range :
  ∀ (a : ℝ), (∀ x : ℝ, x^2 - 2 * a * x + 1 < 0 ↔ a ∈ valid_a_range) :=
by
  sorry

end solve_valid_a_range_l581_581362


namespace number_of_perfect_squares_and_cubes_l581_581760

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end number_of_perfect_squares_and_cubes_l581_581760


namespace geometric_sequence_term_l581_581382

/-
Prove that the 303rd term in a geometric sequence with the first term a1 = 5 and the second term a2 = -10 is 5 * 2^302.
-/

theorem geometric_sequence_term :
  let a1 := 5
  let a2 := -10
  let r := a2 / a1
  let n := 303
  let a_n := a1 * r^(n-1)
  a_n = 5 * 2^302 :=
by
  let a1 := 5
  let a2 := -10
  let r := a2 / a1
  let n := 303
  have h1 : a1 * r^(n-1) = 5 * 2^302 := sorry
  exact h1

end geometric_sequence_term_l581_581382


namespace exists_excircle_radius_at_least_three_times_incircle_radius_l581_581094

variable (a b c s T r ra rb rc : ℝ)
variable (ha : ra = T / (s - a))
variable (hb : rb = T / (s - b))
variable (hc : rc = T / (s - c))
variable (hincircle : r = T / s)

theorem exists_excircle_radius_at_least_three_times_incircle_radius
  (ha : ra = T / (s - a)) (hb : rb = T / (s - b)) (hc : rc = T / (s - c)) (hincircle : r = T / s) :
  ∃ rc, rc ≥ 3 * r :=
by {
  use rc,
  sorry
}

end exists_excircle_radius_at_least_three_times_incircle_radius_l581_581094


namespace primes_eq_seven_and_three_l581_581275

open Nat

-- Define the problem statement in Lean
theorem primes_eq_seven_and_three (p q : ℕ) (h₀ : prime p) (h₁ : prime q) (h₂ : p^3 - q^5 = (p + q)^2) 
    : p = 7 ∧ q = 3 := by 
  sorry

end primes_eq_seven_and_three_l581_581275


namespace num_divisors_18800_divisible_by_235_l581_581635

open Nat

theorem num_divisors_18800_divisible_by_235 :
  let num_divisors := (fun (n : ℕ) (p : ℕ) =>
    let prime_factorization_n := (2^4) * (5^2) * 47;
    let prime_factorization_p := 5 * 47;
    if n = prime_factorization_n ∧ p = prime_factorization_p then 10 else 0)
  in num_divisors 18800 235 = 10 :=
by 
  sorry

end num_divisors_18800_divisible_by_235_l581_581635


namespace min_blue_edges_l581_581636

def tetrahedron_min_blue_edges : ℕ := sorry

theorem min_blue_edges (edges_colored : ℕ → Bool) (face_has_blue_edge : ℕ → Bool) 
    (H1 : ∀ face, face_has_blue_edge face)
    (H2 : ∀ edge, face_has_blue_edge edge = True → edges_colored edge = True) : 
    tetrahedron_min_blue_edges = 2 := 
sorry

end min_blue_edges_l581_581636


namespace sum_of_distinct_divisors_l581_581897

theorem sum_of_distinct_divisors (n : ℕ) (x : ℕ) (h : x < nat.factorial n) :
  ∃ (k : ℕ) (d : fin k → ℕ), k ≤ n ∧ (∀ i j, i ≠ j → d i ≠ d j) ∧ (∀ i, d i ∣ nat.factorial n) ∧ (finset.univ.sum d = x) :=
sorry

end sum_of_distinct_divisors_l581_581897


namespace product_of_81_numbers_is_negative_l581_581017

theorem product_of_81_numbers_is_negative
  (a : Fin 81 → ℝ) 
  (h_nonzero : ∀ i, a i ≠ 0)
  (h_sum_adj : ∀ i : Fin 80, a i + a (⟨i + 1, by linarith⟩) > 0) 
  (h_sum_neg : ∑ i, a i < 0) : 
  ∏ i, a i < 0 := 
sorry

end product_of_81_numbers_is_negative_l581_581017


namespace sum_of_first_twelve_nice_numbers_l581_581241

def is_proper_divisor (n d : ℕ) : Prop := d > 1 ∧ d < n ∧ n % d = 0

def nice_number (n : ℕ) : Prop :=
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n = p * q) ∨ 
  (∃ p : ℕ, Prime p ∧ n = p ^ 3) ∨
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ n = p * q * r)

def first_twelve_nice_numbers : List ℕ :=
  [6, 8, 10, 14, 15, 21, 22, 26, 27, 30, 33, 34]

def is_first_twelve_nice_numbers (l : List ℕ) : Prop :=
  l = first_twelve_nice_numbers

theorem sum_of_first_twelve_nice_numbers :
  (∑ n in first_twelve_nice_numbers, n) = 246 :=
by
  sorry

end sum_of_first_twelve_nice_numbers_l581_581241


namespace numPerfectSquaresOrCubesLessThan1000_l581_581768

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def isSixthPower (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

theorem numPerfectSquaresOrCubesLessThan1000 : 
  ∃ n : ℕ, n = 38 ∧ ∀ k < 1000, (isPerfectSquare k ∨ isPerfectCube k → k) ↔ n :=
by
  sorry

end numPerfectSquaresOrCubesLessThan1000_l581_581768


namespace measure_of_angle_E_l581_581021

theorem measure_of_angle_E
    (A B C D E F : ℝ)
    (h1 : A = B)
    (h2 : B = C)
    (h3 : C = D)
    (h4 : E = F)
    (h5 : A = E - 30)
    (h6 : A + B + C + D + E + F = 720) :
  E = 140 :=
by
  -- Proof goes here
  sorry

end measure_of_angle_E_l581_581021


namespace perfect_squares_and_cubes_l581_581780

def is_perfect_square (n: ℕ) : Prop := ∃ k: ℕ, k ^ 2 = n
def is_perfect_cube (n: ℕ) : Prop := ∃ k: ℕ, k ^ 3 = n
def is_perfect_sixth_power (n: ℕ) : Prop := ∃ k: ℕ, k ^ 6 = n

def num_perfect_squares (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_square n).length
def num_perfect_cubes (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_cube n).length
def num_perfect_sixth_powers (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_sixth_power n).length

theorem perfect_squares_and_cubes (bound: ℕ) (hbound: bound = 1000) :
  num_perfect_squares bound + num_perfect_cubes bound - num_perfect_sixth_powers bound = 37 :=
by
  -- conditions:
  have ps := num_perfect_squares bound
  have pc := num_perfect_cubes bound
  have p6 := num_perfect_sixth_powers bound
  sorry

end perfect_squares_and_cubes_l581_581780


namespace amy_basket_count_l581_581232

theorem amy_basket_count :
  ∀ (chocolate_bars mms marshmallows total_candies baskets_per_basket total_baskets : ℕ),
    chocolate_bars = 5 →
    mms = 7 * chocolate_bars →
    marshmallows = 6 * mms →
    total_candies = chocolate_bars + mms + marshmallows →
    baskets_per_basket = 10 →
    total_baskets = total_candies / baskets_per_basket →
    total_baskets = 25 :=
begin
  intros chocolate_bars mms marshmallows total_candies baskets_per_basket total_baskets,
  assume h1 h2 h3 h4 h5 h6,
  sorry,
end

end amy_basket_count_l581_581232


namespace count_perfect_squares_and_cubes_l581_581743

theorem count_perfect_squares_and_cubes (n : ℕ) (h : n = 1000) : 
  let squares := {x : ℕ | (x ^ 2) < n}
  let cubes := {x : ℕ | (x ^ 3) < n}
  let sixth_powers := {x : ℕ | (x ^ 6) < n}
  (squares.card + cubes.card - sixth_powers.card) = 39 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581743


namespace area_of_remaining_figure_l581_581293

theorem area_of_remaining_figure (side_length: ℝ) (cells_per_side : ℕ) : 
  side_length = 1 → cells_per_side = 6 → 
  let total_area := (side_length * cells_per_side) * (side_length * cells_per_side) in
  let dark_gray_area := (side_length * 1) * (side_length * 3) in
  let light_gray_area := (side_length * 2) * (side_length * 3) in
  let total_gray_area := dark_gray_area + light_gray_area in
  let remaining_area := total_area - total_gray_area in
  remaining_area = 27 := 
by
  intros h_side_length h_cells_per_side
  let total_area := (side_length * cells_per_side) * (side_length * cells_per_side)
  let dark_gray_area := (side_length * 1) * (side_length * 3)
  let light_gray_area := (side_length * 2) * (side_length * 3)
  let total_gray_area := dark_gray_area + light_gray_area
  let remaining_area := total_area - total_gray_area
  have h1 : total_area = 36, by 
    rw [h_side_length, h_cells_per_side]
    norm_num
  have h2 : dark_gray_area = 3, by 
    rw [h_side_length]
    norm_num
  have h3 : light_gray_area = 6, by 
    rw [h_side_length]
    norm_num
  have h4 : total_gray_area = 9, by 
    rw [h2, h3]
    norm_num
  have h5 : remaining_area = 27, by 
    rw [h1, h4]
    norm_num
  exact h5

end area_of_remaining_figure_l581_581293


namespace appropriate_term_for_assessment_l581_581720

-- Definitions
def price : Type := String
def value : Type := String
def cost : Type := String
def expense : Type := String

-- Context for assessment of the project
def assessment_context : Type := Π (word : String), word ∈ ["price", "value", "cost", "expense"] → Prop

-- Main Lean statement
theorem appropriate_term_for_assessment (word : String) (h : word ∈ ["price", "value", "cost", "expense"]) :
  word = "value" :=
sorry

end appropriate_term_for_assessment_l581_581720


namespace smallest_number_of_ten_consecutive_natural_numbers_l581_581976

theorem smallest_number_of_ten_consecutive_natural_numbers 
  (x : ℕ) 
  (h : 6 * x + 39 = 2 * (4 * x + 6) + 15) : 
  x = 6 := 
by 
  sorry

end smallest_number_of_ten_consecutive_natural_numbers_l581_581976


namespace extension_AC_passes_through_D_l581_581904

/-- Given segment AB is 8 cm, point F bisects AB creating two segments AF and FB of 4 cm each.
    Isosceles triangles AFC and FBD are constructed with legs 3 cm and 7 cm respectively.
    Prove that if extension of AC passes through D then CD is 6 cm. -/
theorem extension_AC_passes_through_D 
  (A B F C D : ℝ)
  (h1 : A + B = 8)
  (h2 : F = 4)
  (h3 : dist A F = 4)
  (h4 : dist F B = 4)
  (h5 : dist A C = 3)
  (h6 : dist F D = 7)
  (h7 : line_through A C D)
  : dist C D = 6 :=
sorry

end extension_AC_passes_through_D_l581_581904


namespace find_a_plus_b_l581_581704

noncomputable def even_function_condition {a b : ℝ} (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f(x) = a * x^2 + b * x ∧ f(-x) = f(x)

theorem find_a_plus_b (a b : ℝ) (f : ℝ → ℝ) (h1 : even_function_condition f) (h2 : ∀ x, a - 1 ≤ x ∧ x ≤ 2 * a):
  a + b = 1 / 3 :=
begin
  sorry
end

end find_a_plus_b_l581_581704


namespace increasing_function_range_l581_581722

theorem increasing_function_range (m : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y) ↔ m ∈ Icc (-6 : ℝ) (∞ : ℝ) :=
by
  have f : ℝ → ℝ := λ x, 3 * x^2 + m * x + 2
  sorry

end increasing_function_range_l581_581722


namespace parent_combinations_for_O_l581_581596

-- Define the blood types
inductive BloodType
| A
| B
| O
| AB

open BloodType

-- Define the conditions given in the problem
def parent_not_AB (p : BloodType) : Prop :=
  p ≠ AB

def possible_parent_types : List BloodType :=
  [A, B, O]

-- The math proof problem
theorem parent_combinations_for_O :
  ∀ (mother father : BloodType),
    parent_not_AB mother →
    parent_not_AB father →
    mother ∈ possible_parent_types →
    father ∈ possible_parent_types →
    (possible_parent_types.length * possible_parent_types.length) = 9 := 
by
  intro mother father h1 h2 h3 h4
  sorry

end parent_combinations_for_O_l581_581596


namespace complement_intersection_l581_581352

def A : Set ℝ := { x | log 7 (x - 2) < 1 }

def B : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }

theorem complement_intersection (x : ℝ) :
  x ∈ (A ∩ B)ᶜ ↔ x ∈ (-∞, 2] ∪ [3, +∞) := sorry

end complement_intersection_l581_581352


namespace certain_number_is_1_l581_581483

theorem certain_number_is_1 (z : ℕ) (hz : z % 4 = 0) :
  ∃ n : ℕ, (z * (6 + z) + n) % 2 = 1 ∧ n = 1 :=
by
  sorry

end certain_number_is_1_l581_581483


namespace complement_A_eq_interval_l581_581342

-- Define the universal set U as the set of all real numbers.
def U : Set ℝ := Set.univ

-- Define the set A using the condition x^2 - 2x - 3 > 0.
def A : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the complement of A with respect to U.
def A_complement : Set ℝ := { x | -1 <= x ∧ x <= 3 }

theorem complement_A_eq_interval : A_complement = { x | -1 <= x ∧ x <= 3 } :=
by
  sorry

end complement_A_eq_interval_l581_581342


namespace geometric_sequence_log_sum_l581_581287

variable {α : Type*}

noncomputable def geometric_sequence (a : α → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (r : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : geometric_sequence a r)
  (h3 : a 10 * a 11 = Real.exp 5) :
  (Finset.range 20).sum (λ n, Real.log (a n + 1)) = 50 :=
sorry

end geometric_sequence_log_sum_l581_581287


namespace lcm_of_numbers_with_ratio_and_hcf_l581_581139

theorem lcm_of_numbers_with_ratio_and_hcf (a b : ℕ) (h1 : a = 3 * x) (h2 : b = 4 * x) (h3 : Nat.gcd a b = 3) : Nat.lcm a b = 36 := 
  sorry

end lcm_of_numbers_with_ratio_and_hcf_l581_581139


namespace vectors_length_sum_greater_than_one_l581_581318

noncomputable def resultant_vector_length (S : Finset ℝ → ℝ) : ℝ := 
  if S = ∅ then 0 else
    let v := (Finset.sum S) in Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vectors_length_sum_greater_than_one {v : ℕ → ℝ × ℝ} (hv : ∀ i, |v i| ≥ 0) (h : Finset.sum (Finset.univ.map v) = 4) :
  ∃ S : Finset ℝ, resultant_vector_length S > 1 :=
sorry

end vectors_length_sum_greater_than_one_l581_581318


namespace polynomial_degree_is_eight_l581_581625

noncomputable def resulting_polynomial_deg (x : ℂ) : ℕ :=
  (x^4 - 2 + x^(-4)) * x^4 * (1 - 3 * x^(-1) + 3 * x^(-2))

theorem polynomial_degree_is_eight  (x : ℤ) : 
  (resulting_polynomial_deg (x : ℂ)).degree = 8 := 
begin
  sorry
end

end polynomial_degree_is_eight_l581_581625


namespace min_value_of_d_plus_PQ_l581_581883

noncomputable def parabola := { P : ℝ × ℝ // ∃ x y, y^2 = 4 * x ∧ P = (x, y) }
noncomputable def line := { Q : ℝ × ℝ // ∃ x y, x - y + 5 = 0 ∧ Q = (x, y) }

def parabola_directrix_distance (P : ℝ × ℝ) : ℝ :=
  let directrix := -1 in -- Directrix of y^2 = 4x is x = -1
  abs (P.1 + 1) / √1

def point_to_line_distance (F Q : ℝ × ℝ) : ℝ :=
  abs (Q.1 - Q.2 + 5) / √2

def P_to_F_distance (P : ℝ × ℝ) : ℝ :=
  let F := (1, 0) in
  real.sqrt ((P.1 - 1)^2 + (P.2 - 0)^2)

def PQ_distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem min_value_of_d_plus_PQ :
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ parabola ∧ Q ∈ line ∧ 
  parabola_directrix_distance P + PQ_distance P Q = 3 * real.sqrt 2 :=
sorry

end min_value_of_d_plus_PQ_l581_581883


namespace find_ordered_pair_l581_581657

theorem find_ordered_pair (x y : ℚ) 
  (h1 : 7 * x - 30 * y = 3) 
  (h2 : 3 * y - x = 5) : 
  x = -53 / 3 ∧ y = -38 / 9 :=
sorry

end find_ordered_pair_l581_581657


namespace container_initial_fill_l581_581193

def initial_fill_percentage (P : ℕ) : Prop :=
  ∃ (C : ℕ) (added_water : ℕ) (final_fill_fraction : ℚ),
  C = 60 ∧ added_water = 27 ∧ final_fill_fraction = 3/4 ∧
  ((P : ℚ) / 100 * C + added_water = final_fill_fraction * C)

theorem container_initial_fill (P : ℕ) : initial_fill_percentage P → P = 30 :=
by
  intro h
  cases h with C hC
  cases hC with added_water hC
  cases hC with final_fill_fraction hC
  cases' hC with hC1 hC'
  cases' hC' with hA hC'
  cases' hC' with hF hEq
  have hc : C = 60 := by assumption
  have ha : added_water = 27 := by assumption
  have hf : final_fill_fraction = 3/4 := by assumption
  have heq : (P : ℚ) / 100 * 60 + 27 = 3/4 * 60 := by assumption
  sorry

end container_initial_fill_l581_581193


namespace xiaofang_final_score_l581_581553

def removeHighestLowestScores (scores : List ℕ) : List ℕ :=
  let max_score := scores.maximum.getD 0
  let min_score := scores.minimum.getD 0
  scores.erase max_score |>.erase min_score

def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem xiaofang_final_score :
  let scores := [95, 94, 91, 88, 91, 90, 94, 93, 91, 92]
  average (removeHighestLowestScores scores) = 92 := by
  sorry

end xiaofang_final_score_l581_581553


namespace petya_total_rides_l581_581819

theorem petya_total_rides (spending1 spending2 total spending1_cost spending2_cost: ℕ) :
  spending1 = 345 → spending2 = 365 → total = spending1 + spending2 → spending1_cost = 345 → spending2_cost = 365 → (spending1 + spending2 = 710):
  ∃ (n : ℕ), n = 15 :=
begin
  sorry
end

end petya_total_rides_l581_581819


namespace shaded_to_white_ratio_l581_581527

theorem shaded_to_white_ratio (largest_square_area : ℝ) 
  (vertices_condition : ∀ (n : ℕ), n > 1 → 
    (let side_length := (1:ℝ) / (2 ^ (n - 1)) 
     in side_length = side_length / 2 + side_length / 2)) :
  let shaded_area := 5 * (largest_square_area / 8) / (2 ^ 2) in
  let white_area := 3 * (largest_square_area / 8) / (2 ^ 2) in
  shaded_area / white_area = 5 / 3 :=
by
  sorry

end shaded_to_white_ratio_l581_581527


namespace salary_of_thomas_l581_581186

variable (R Ro T : ℕ)

theorem salary_of_thomas 
  (h1 : R + Ro = 8000) 
  (h2 : R + Ro + T = 15000) : 
  T = 7000 := by
  sorry

end salary_of_thomas_l581_581186


namespace triangular_region_adjacent_to_each_line_l581_581441

theorem triangular_region_adjacent_to_each_line 
  (n : ℕ) (h_n : 3 ≤ n) (lines : fin n → line ℝ)
  (no_parallel : ∀ i j, i ≠ j → ¬ parallel (lines i) (lines j))
  (no_three_concurrent : ∀ i j k, i ≠ j ∧ j ≠ k → ¬ concurrent (lines i) (lines j) (lines k)) :
  ∀ i, ∃ Δ : set (point ℝ), is_triangle Δ ∧ Δ ⊆ adjacent_regions (lines i) :=
sorry

end triangular_region_adjacent_to_each_line_l581_581441


namespace silverware_probability_l581_581349

-- Definitions based on the problem conditions
def total_silverware : ℕ := 8 + 10 + 7
def total_combinations : ℕ := Nat.choose total_silverware 4

def fork_combinations : ℕ := Nat.choose 8 2
def spoon_combinations : ℕ := Nat.choose 10 1
def knife_combinations : ℕ := Nat.choose 7 1

def favorable_combinations : ℕ := fork_combinations * spoon_combinations * knife_combinations
def specific_combination_probability : ℚ := favorable_combinations / total_combinations

-- The statement to prove the given probability
theorem silverware_probability :
  specific_combination_probability = 392 / 2530 :=
by
  sorry

end silverware_probability_l581_581349


namespace mean_height_is_approx_correct_l581_581491

def heights : List ℕ := [120, 123, 127, 132, 133, 135, 140, 142, 145, 148, 152, 155, 158, 160]

def mean_height : ℚ := heights.sum / heights.length

theorem mean_height_is_approx_correct : 
  abs (mean_height - 140.71) < 0.01 := 
by
  sorry

end mean_height_is_approx_correct_l581_581491


namespace f_zero_is_two_l581_581911

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x1 x2 x3 x4 x5 : ℝ) : 
  f (x1 + x2 + x3 + x4 + x5) = f x1 + f x2 + f x3 + f x4 + f x5 - 8

theorem f_zero_is_two : f 0 = 2 := 
by
  sorry

end f_zero_is_two_l581_581911


namespace sage_can_determine_weight_l581_581200

theorem sage_can_determine_weight : 
  ∃ (wei : ℕ → ℕ), 
    (∀ i, i ∈ {0, 1, 2, 3, 4, 5, 6} → wei i ∈ {7, 8, 9, 10, 11, 12, 13}) →  -- weights per coin in each bag
    (∀ B : ℕ, B ∈ {0, 1, 2, 3, 4, 5, 6} →  -- B indicates the specified bag
      ∃ f : ℕ → {0, 1, ..., 6} → Prop,
        f 10 = (λ i, 7 + 8 + 9 + 10 + 11 + 12 + 13 ≤ 70 * wei B) ∧
        ( (70 * wei (f 10) = 70 * wei B) ∨
          ((70 * wei (f 10) < 70 * wei B ∧ 70 * wei B ≤ 80 * wei (f 1)) ∨
           (80 * wei (f 1) < 70 * wei B ∧ 70 * wei B ≤ 90 * wei (f 2)) ∨
           (90 * wei (f 2) < 70 * wei B ∧ 70 * wei B ≤ 100 * wei (f 3)) ∨
           (100 * wei (f 3) < 70 * wei B ∧ 70 * wei B ≤ 110 * wei (f 4)) ∨
           (110 * wei (f 4) < 70 * wei B ∧ 70 * wei B ≤ 120 * wei (f 5)) ∨
           (120 * wei (f 5) < 70 * wei B ∧ 70 * wei B ≤ 130 * wei (f 6))))) :=
sorry

end sage_can_determine_weight_l581_581200


namespace nonneg_int_lin_comb_covers_z_n_l581_581047

-- Definitions based on given conditions
variables {n m : ℕ} (a : fin m → ℤ ^ n)

-- Nonnegative integer linear combinations and its implications
open Set
def nonneg_int_lin_comb (a : fin m → ℤ ^ n) : fin m →₀ ℕ → ℤ ^ n :=
  λ c, ∑ i in c.support, (c i) • (a i)

-- Conditions as hypotheses
def cond1 (a : fin m → ℤ ^ n) : Prop :=
  ∀ w : ℤ ^ n, ∃ i j, i ≠ j ∧ (a i - a j) ≠ 0

def cond2 (a : fin m → ℤ ^ n) : Prop :=
  gcd_dvd (determinant_submatrix a) = 1

-- The theorem statement
theorem nonneg_int_lin_comb_covers_z_n {a : fin m → ℤ ^ n} (h₁ : m ≥ n) (h₂ : cond1 a) (h₃ : cond2 a) :
  ∀ z : ℤ ^ n, ∃ c : fin m →₀ ℕ, nonneg_int_lin_comb a c = z :=
  sorry

end nonneg_int_lin_comb_covers_z_n_l581_581047


namespace existence_of_two_balanced_lines_l581_581099

structure Point :=
(x : ℝ)
(y : ℝ)

def are_not_collinear (a b c : Point) : Prop :=
(a.y - b.y) * (b.x - c.x) ≠ (b.y - c.y) * (a.x - b.x)

def balanced_line (blue_points red_points : list Point) (l : Point × Point) : Prop :=
∃ (b_points r_points : list Point) (b₁ r₁ : Point),
b₁ ∈ blue_points ∧ r₁ ∈ red_points ∧ 
l = (b₁, r₁) ∧
b_points.count l.1 > 0 ∧ r_points.count l.2 > 0 ∧
b_points.count l.1 = r_points.count l.2

theorem existence_of_two_balanced_lines
  (n : ℕ) (h1 : n > 1) (points : list Point)
  (h2 : points.length = 2 * n)
  (blue_points red_points : list Point)
  (h3 : blue_points.length = n ∧ red_points.length = n)
  (h4 : ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → are_not_collinear p1 p2 p3)
  (h5 : ∀ p, p ∈ points → (p ∈ blue_points ∨ p ∈ red_points)): 
  ∃ (l1 l2 : Point × Point), balanced_line blue_points red_points l1 ∧ balanced_line blue_points red_points l2 ∧ l1 ≠ l2 :=
by
  sorry

end existence_of_two_balanced_lines_l581_581099


namespace find_sum_of_numbers_l581_581183

-- Define the problem using the given conditions
def sum_of_three_numbers (a b c : ℝ) : ℝ :=
  a + b + c

-- The main theorem we want to prove
theorem find_sum_of_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 222) (h2 : a * b + b * c + c * a = 131) :
  sum_of_three_numbers a b c = 22 :=
by
  sorry

end find_sum_of_numbers_l581_581183


namespace sum_is_maximized_at_7th_term_l581_581128

noncomputable def arithmetic_sequence (a d : ℝ) : ℕ → ℝ
| n => a + n * d

theorem sum_is_maximized_at_7th_term 
  (a d : ℝ) 
  (h1 : a > 0) 
  (h2 : (∑ i in finset.range 3, arithmetic_sequence a d i) = (∑ i in finset.range 11, arithmetic_sequence a d i)) :
  (∑ i in finset.range 7, arithmetic_sequence a d i) > (∑ i in finset.range (nat.succ 7), arithmetic_sequence a d i) :=
sorry

end sum_is_maximized_at_7th_term_l581_581128


namespace ratio_EF_BC_l581_581966

variable {A B C D M N E F : Type}
variable [Point A] [Point B] [Point C] [Point D]
variable [Point M] [Point N]
variable [Point E] [Point F]

noncomputable def are_medians 
    (A : Point) (B : Point) (C : Point) (D : Point) 
    (M : Point) (N : Point) : Prop := 
    isMedian B C A M ∧ isMedian C D A N 

noncomputable def EF_parallel_BC 
    (E : Point) (F : Point) (B : Point) (C : Point) : Prop := 
    isOnMedian F D N ∧ isParallel (lineThroughPoints E F) (lineThroughPoints B C)

theorem ratio_EF_BC 
    (A B C D M N E F : Point) 
    (h_medians : are_medians A B C D M N) 
    (h_parallel : EF_parallel_BC E F B C) : 
    ratio (length E F) (length B C) = 1 / 3 :=
sorry

end ratio_EF_BC_l581_581966


namespace area_of_triangle_l581_581907

-- Definitions based on the given conditions
-- Let x be AB and y be BC
variables (x y : ℝ)
noncomputable def AC : ℝ := 9

-- The conditions
def is_right_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] 
  [has_angle A B C 90] := sorry

def satisfies_angle_bisector_theorem (A B C P : Type) [metric_space A] 
  [metric_space B] [metric_space C] [metric_space P] (AB BC AP CP : ℝ) :=
  AP = 3 ∧ CP = 6 ∧ AP / CP = AB / BC

-- The statement of the problem to prove
theorem area_of_triangle (A B C P : Type) [metric_space A] [metric_space B] [metric_space C] 
  [metric_space P]
  (h_right_triangle : is_right_triangle A B C)
  (h_angle_bisector_theorem : satisfies_angle_bisector_theorem A B C P x y 3 6)
  (AC_eq : AC = 9) :
  (1 / 2) * x * y = 81 / 5 :=
sorry

end area_of_triangle_l581_581907


namespace cost_of_two_burritos_and_five_quesadillas_l581_581396

theorem cost_of_two_burritos_and_five_quesadillas
  (b q : ℝ)
  (h1 : b + 4 * q = 3.50)
  (h2 : 4 * b + q = 4.10) :
  2 * b + 5 * q = 5.02 := 
sorry

end cost_of_two_burritos_and_five_quesadillas_l581_581396


namespace integral_proof_l581_581964

open Real

noncomputable def integrand (x : ℝ) : ℝ :=
  (real.sqrt $ (1 + real.cbrt (x^2))^3)^0.25 / (x^2 * (x^1/6))

noncomputable def integral (x C : ℝ) : ℝ :=
  -(6/7) * (real.sqrt $\frac {real.sqrt[((1 + real.cbrt (x^2))^7) }^{ 1 / 4 }}) ^7 / (real.sqrt[$\frac {x^7} ^{ 1 / 6 }}) + C

theorem integral_proof (C : ℝ) : ∀ x > 0, 
  ∫ integrand x = integral x C := 
begin 
  sorry
end

end integral_proof_l581_581964


namespace count_perfect_squares_cubes_under_1000_l581_581753

-- Condition definitions
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, m * m = n
def isPerfectCube (n : Nat) : Prop := ∃ m : Nat, m * m * m = n

-- Main proof problem
theorem count_perfect_squares_cubes_under_1000 : 
  let count (p : Nat -> Prop) := Finset.card (Finset.filter p (Finset.range 1000))
  count (λ n, isPerfectSquare n ∨ isPerfectCube n) = 37 := 
sorry

end count_perfect_squares_cubes_under_1000_l581_581753


namespace parametric_to_general_l581_581626

theorem parametric_to_general (θ : ℝ) :
  let x := 2 + Real.sin θ ^ 2,
      y := -1 + 2 * Real.cos θ ^ 2
  in 2 * x + y - 5 = 0 ∧ 2 ≤ x ∧ x ≤ 3 :=
by
  sorry

end parametric_to_general_l581_581626


namespace correct_answer_l581_581156

theorem correct_answer (A B C D : String) (sentence : String)
  (h1 : A = "us")
  (h2 : B = "we")
  (h3 : C = "our")
  (h4 : D = "ours")
  (h_sentence : sentence = "To save class time, our teacher has _ students do half of the exercise in class and complete the other half for homework.") :
  sentence = "To save class time, our teacher has " ++ A ++ " students do half of the exercise in class and complete the other half for homework." :=
by
  sorry

end correct_answer_l581_581156


namespace count_integer_values_b_for_ineq_l581_581631

theorem count_integer_values_b_for_ineq (b : Int) : 
  (∃ x₁ x₂ x₃ : ℤ, (b : ℚ) ≥ 0 ∧ 
    (x₁^2 + (b : ℚ) * x₁ + 6 ≤ 0) ∧ 
    (x₂^2 + (b : ℚ) * x₂ + 6 ≤ 0) ∧ 
    (x₃^2 + (b : ℚ) * x₃ + 6 ≤ 0)) → 
    ((b = -6 ∨ b = 6) → 
    (∃ n : ℤ, n = 2)).
sorry

end count_integer_values_b_for_ineq_l581_581631


namespace cylinder_volume_rotation_l581_581811

theorem cylinder_volume_rotation (length width : ℝ) (π : ℝ) (h : length = 4) (w : width = 2) (V : ℝ) :
  (V = π * (4^2) * width ∨ V = π * (2^2) * length) :=
by
  sorry

end cylinder_volume_rotation_l581_581811


namespace trajectory_and_range_of_k_l581_581445

noncomputable theory

-- Definition for the trajectory of M satisfying the given equation
def trajectory_of_M (x y : ℝ) : Prop :=
  real.sqrt ((x - 2 * real.sqrt 2)^2 + y^2) + real.sqrt ((x + 2 * real.sqrt 2)^2 + y^2) = 6

-- Standard equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 9) + y^2 = 1

-- Definition for the given line
def line_eq (k x y : ℝ) : Prop :=
  y = k * x - 2 * real.sqrt 2 * k

-- Definition that the line intersects the ellipse at points A and B
def intersects_at_A_B (x1 y1 x2 y2 k : ℝ) : Prop :=
  line_eq k x1 y1 ∧ line_eq k x2 y2 ∧ ellipse_equation x1 y1 ∧ ellipse_equation x2 y2

-- Definition for the vector relation between points D, A, and B
def vector_relation (x1 y1 x2 y2 lambda : ℝ) : Prop :=
  (2 * real.sqrt 2 - x1, -y1) = λ • (x2 - (2 * real.sqrt 2), y2)

-- The final statement combining all the conditions to prove the required result
theorem trajectory_and_range_of_k (x y x1 y1 x2 y2 k λ : ℝ) (hM : trajectory_of_M x y)
  (hLine : intersects_at_A_B x1 y1 x2 y2 k) (hAD_DB : vector_relation x1 y1 x2 y2 λ)
  (hLambda : 1 < λ ∧ λ < 2) : 
  ellipse_equation x y ∧ (k ∈ set.Iio (-real.sqrt 7) ∪ set.Ioi (real.sqrt 7)) :=
sorry

end trajectory_and_range_of_k_l581_581445


namespace max_candies_value_l581_581943

theorem max_candies_value (n : ℕ) (h_n : n ≥ 145) 
  (h_condition : ∀ (S : finset ℕ), 145 ≤ S.card ∧ S ⊆ finset.range n → ∃ t : ℕ, S.count t = 10) : 
  n ≤ 160 :=
begin
  sorry
end

end max_candies_value_l581_581943


namespace sum_of_roots_l581_581540

theorem sum_of_roots (a b c : ℝ) (h : a = 1 ∧ b = -5 ∧ c = 6) :
  (-b / a) = 5 :=
by {
    cases h with ha hb,
    cases hb with hb hc,
    rw [ha, hb],
    have h1 : a ≠ 0, by { intro h, linarith },
    simp [h1],
    linarith
}

#check sum_of_roots

end sum_of_roots_l581_581540


namespace total_accepted_cartons_l581_581400

theorem total_accepted_cartons 
  (total_cartons : ℕ) 
  (customers : ℕ) 
  (damaged_cartons : ℕ)
  (h1 : total_cartons = 400)
  (h2 : customers = 4)
  (h3 : damaged_cartons = 60)
  : total_cartons / customers * (customers - (damaged_cartons / (total_cartons / customers))) = 160 := by
  sorry

end total_accepted_cartons_l581_581400


namespace distance_between_parallel_lines_l581_581508

theorem distance_between_parallel_lines 
  (r : ℝ) (d : ℝ) 
  (h1 : 3 * (2 * r^2) = 722 + (19 / 4) * d^2) 
  (h2 : 3 * (2 * r^2) = 578 + (153 / 4) * d^2) : 
  d = 6 :=
by
  sorry

end distance_between_parallel_lines_l581_581508


namespace total_cartons_accepted_l581_581401

theorem total_cartons_accepted (total_cartons : ℕ) (customers : ℕ) (damaged_cartons_per_customer : ℕ) (initial_cartons_per_customer accepted_cartons_per_customer total_accepted_cartons : ℕ) :
    total_cartons = 400 →
    customers = 4 →
    damaged_cartons_per_customer = 60 →
    initial_cartons_per_customer = total_cartons / customers →
    accepted_cartons_per_customer = initial_cartons_per_customer - damaged_cartons_per_customer →
    total_accepted_cartons = accepted_cartons_per_customer * customers →
    total_accepted_cartons = 160 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_cartons_accepted_l581_581401


namespace total_leaves_on_farm_l581_581218

theorem total_leaves_on_farm : 
  (branches_per_tree subbranches_per_branch leaves_per_subbranch trees_on_farm : ℕ)
  (h1 : branches_per_tree = 10)
  (h2 : subbranches_per_branch = 40)
  (h3 : leaves_per_subbranch = 60)
  (h4 : trees_on_farm = 4) :
  (trees_on_farm * branches_per_tree * subbranches_per_branch * leaves_per_subbranch = 96000) :=
by
  sorry

end total_leaves_on_farm_l581_581218


namespace percentage_decrease_in_speed_ascending_l581_581246

theorem percentage_decrease_in_speed_ascending (x : ℝ) : 
  (30 : ℝ) - 0.01 * x * 30 > 0 → 
  60 / (30 - 0.01 * x * 30) + 2 = 6 → 
  x = 50 :=
by
  intro hpos htime
  sorry

end percentage_decrease_in_speed_ascending_l581_581246


namespace evaluate_expression_l581_581267

theorem evaluate_expression : ((3 ^ 2) ^ 3) - ((2 ^ 3) ^ 2) = 665 := by
  sorry

end evaluate_expression_l581_581267


namespace classes_diff_correct_l581_581211

def t (enrollments : List ℝ) : ℝ :=
  enrollments.sum / enrollments.length

def s (students : ℝ) (enrollments : List ℝ) : ℝ :=
  enrollments.sum $ λ e, e * (e / students)

def classes_diff (students : ℝ) (enrollments : List ℝ) : ℝ :=
  t enrollments - s students enrollments

theorem classes_diff_correct :
  let students := 120
  let enrollments := [60, 30, 20, 10]
  classes_diff students enrollments = -11.67 :=
by
  sorry

end classes_diff_correct_l581_581211


namespace find_Q_l581_581416

theorem find_Q : 
  let P := 95 in 
  let Q := (Σ k in finset.range (P - 3) \ 2, log 128 (2^k)) in
  Q = 329 := 
by
  sorry

end find_Q_l581_581416


namespace angle_bisector_centroid_l581_581394

theorem angle_bisector_centroid (a b c : ℝ) (h : 1 / a = 1 / b + 1 / c) :
  let A := (0, 0)
  let B := (a, 0)
  let C := (0, b)
  let D := (c * A.1 + a * C.1) / (a + c)
  let E := (b * A.1 + a * B.1) / (a + b)
  let centroid := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  in collinear {A, B, C} centroid D E :=
sorry

end angle_bisector_centroid_l581_581394


namespace moles_of_AgOH_formed_l581_581284

theorem moles_of_AgOH_formed (moles_AgNO3 : ℕ) (moles_NaOH : ℕ) 
  (reaction : moles_AgNO3 + moles_NaOH = 2) : moles_AgNO3 + 2 = 2 :=
by
  sorry

end moles_of_AgOH_formed_l581_581284


namespace exists_smallest_x_l581_581198

def f : ℝ → ℝ :=
λ x, if 2 ≤ x ∧ x ≤ 4 then 1 - |x - 2| else sorry

theorem exists_smallest_x (x : ℝ) : 
  (∀ x > 0, f(3 * x) = 3 * f(x)) →
  (f(2003) = 2793003) →
  ∃ y : ℝ, f(y) = 2793003 ∧ ∀ z : ℝ, f(z) = 2793003 → y ≤ z := 
sorry

end exists_smallest_x_l581_581198


namespace initial_tax_rate_l581_581003

theorem initial_tax_rate 
  (income : ℝ)
  (differential_savings : ℝ)
  (final_tax_rate : ℝ)
  (initial_tax_rate : ℝ) 
  (h1 : income = 42400) 
  (h2 : differential_savings = 4240) 
  (h3 : final_tax_rate = 32)
  (h4 : differential_savings = (initial_tax_rate / 100) * income - (final_tax_rate / 100) * income) :
  initial_tax_rate = 42 :=
sorry

end initial_tax_rate_l581_581003


namespace total_balloons_is_18_l581_581292

variable (fredBalloons : Nat) (samBalloons : Nat) (maryBalloons : Nat)
variable (cost : Nat) -- We include cost because it's mentioned in conditions

theorem total_balloons_is_18 (h₁ : fredBalloons = 5)
                           (h₂ : samBalloons = 6)
                           (h₃ : maryBalloons = 7)
                           (h₄ : cost = 9) : 
                           fredBalloons + samBalloons + maryBalloons = 18 := 
by 
  rw [h₁, h₂, h₃]
  rfl

end total_balloons_is_18_l581_581292


namespace children_tickets_l581_581981

theorem children_tickets (A C : ℝ) (h1 : A + C = 200) (h2 : 3 * A + 1.5 * C = 510) : C = 60 := by
  sorry

end children_tickets_l581_581981


namespace initialPersonsCount_l581_581110

noncomputable def numberOfPersonsInitially (increaseInAverageWeight kg_diff : ℝ) : ℝ :=
  kg_diff / increaseInAverageWeight

theorem initialPersonsCount :
  numberOfPersonsInitially 2.5 20 = 8 := by
  sorry

end initialPersonsCount_l581_581110


namespace min_angle_inclination_l581_581726

def f (x : ℝ) := (x^3 / 3) - (x^2) + 1

noncomputable def f' (x : ℝ) := deriv f x

-- Statement: Given the function f for 0 < x < 2, prove the minimum angle of inclination alpha is 3π/4
theorem min_angle_inclination : (∀ x, 0 < x ∧ x < 2 → f' x = x^2 - 2*x) →
  ∃ α, (∀ x, 0 < x ∧ x < 2 → tan α = f' x) ∧ α = 3 * Real.pi / 4 :=
by
  sorry

end min_angle_inclination_l581_581726


namespace pascal_even_rows_count_l581_581348

open Nat

theorem pascal_even_rows_count : ∃ n, n = 4 ∧ ∀ r, 2 ≤ r ∧ r ≤ 30 → 
  (∀ k, 1 ≤ k ∧ k ≤ r - 1 → (binomial r k) % 2 = 0) :=
sorry

end pascal_even_rows_count_l581_581348


namespace sleep_time_comparison_l581_581157

def xiaoyu_sleep_times : List ℝ := [8, 9, 9, 9, 10, 9, 9]
def xiaozhong_sleep_times : List ℝ := [10, 10, 9, 9, 8, 8, 9]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.foldr (+) 0) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.foldr (λ x acc => acc + (x - m) ^ 2) 0) / l.length

theorem sleep_time_comparison :
  mean xiaoyu_sleep_times = mean xiaozhong_sleep_times ∧
  variance xiaoyu_sleep_times ≠ variance xiaozhong_sleep_times := by
    sorry

end sleep_time_comparison_l581_581157


namespace micah_water_l581_581873

theorem micah_water (x : ℝ) (h1 : 3 * x + x = 6) : x = 1.5 :=
sorry

end micah_water_l581_581873


namespace Q_value_l581_581417

noncomputable theory

def is_arithmetic_sequence (a d: Int) (P: Int): Prop := 
  ∃ k: Int, P = a + k * d

def log_base_sum (base log_arg: Int) (terms: List Int) : Real := 
  terms.foldl (λ sum k => sum + Real.logBase base (log_arg^k)) 0

theorem Q_value : ∀ Q : Real, 
  (Q = log_base_sum 128 2 [3, 5, 7, ..., 95]) → Q = 329 :=
sorry

end Q_value_l581_581417


namespace probability_area_or_circumference_gt_150_l581_581512

noncomputable def sum_of_dice (d1 d2 d3 : ℕ) : ℕ :=
  d1 + d2 + d3

noncomputable def area_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

noncomputable def four_times_circumference (r : ℝ) : ℝ :=
  8 * Real.pi * r

theorem probability_area_or_circumference_gt_150 :
  let favorable_outcomes := {s ∈ (list.Ico 3 19) | s > 6.91}.card in 
  let total_outcomes := 216 in
  (favorable_outcomes / total_outcomes : ℝ) = (101 / 108 : ℝ) :=
by
  sorry

end probability_area_or_circumference_gt_150_l581_581512


namespace angle_EDF_60_degrees_l581_581481

theorem angle_EDF_60_degrees (O D E F : Type) [metric_space O] [topological_space O] [order_topology O] 
  (circumscribed : O → O → O → Prop) 
  (center : circumscribed O D E F)
  (angle_DOF : ∠ D O F = 110)
  (angle_EOF : ∠ E O F = 130) : 
  ∠ E D F = 60 :=
by
  sorry

end angle_EDF_60_degrees_l581_581481


namespace length_of_chord_l581_581479

def line_equation (x y : ℝ) : Prop := x + y = 2
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

theorem length_of_chord (A B : ℝ × ℝ) (hA : circle_equation A.1 A.2) (hB : circle_equation B.1 B.2) 
  (hlineA : line_equation A.1 A.2) (hlineB : line_equation B.1 B.2) :
  dist A B = sqrt 2 := 
sorry

end length_of_chord_l581_581479


namespace perpendicular_condition_l581_581869

-- Definitions of planes and line
variable {α β : Plane} (b : Line)

-- Conditions: α and β are different planes, and b is a line such that b is in plane β
axiom different_planes (h : α ≠ β)
axiom line_in_plane (hb : b ⊂ β)

-- Question: Prove that "line b is perpendicular to plane α" is neither a sufficient 
-- nor a necessary condition for "plane α is perpendicular to plane β".
theorem perpendicular_condition (h₁ : b ⊥ α) : ¬ ((α ⊥ β) ↔ (b ⊥ α)) ∧ ¬ ((b ⊥ α) ↔ (α ⊥ β)) :=
sorry

end perpendicular_condition_l581_581869


namespace find_positive_x_l581_581646

theorem find_positive_x (x : ℝ) (h1 : x > 0) (h2 : x * (real.sqrt (16 - x^2)) + real.sqrt (16 * x - x^4) ≥ 16) : 
  x = 2 * real.sqrt 2 :=
by 
  sorry

end find_positive_x_l581_581646


namespace root_value_l581_581706

theorem root_value (m : ℝ) (h : 2 * m^2 - 7 * m + 1 = 0) : m * (2 * m - 7) + 5 = 4 := by
  sorry

end root_value_l581_581706


namespace marvin_birthday_friday_l581_581432

open Nat

def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0)

noncomputable def day_of_week (year month day : ℕ) : ℕ :=
  Date.civil_to_gregorian_transformed (mk_civil year month day) mod 7

theorem marvin_birthday_friday (year : ℕ) (by2013 : day_of_week 2013 5 27 = 1) :
  ∃ year, year > 2013 ∧ day_of_week year 5 27 = 5 :=
by
  have day_increment : ∀ n, day_of_week (2013 + n) 5 27 = (day_of_week 2013 5 27 + finset.range (n).sum (λ i, if is_leap_year (2013 + i) then 2 else 1)) % 7 :=
    λ n, sorry  -- method to calculate each year's increment

  existsi 2016
  split
  · linarith
  · simp [day_of_week, by2013, day_increment]
    sorry -- proof that the specific day of week for May 27, 2016 is a Friday

end marvin_birthday_friday_l581_581432


namespace find_F_and_ellipse_l581_581712

-- Definitions for the given conditions
def ellipse (x y a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def parabola (x y p : ℝ) := (y^2 = 2 * p * x)
def distance_from_point_to_line (x y a b c : ℝ) := abs (a * x + b * y + c) / real.sqrt (a^2 + b^2)
def common_chord_length (¥ common_chord_length: ℝ) := 2 * real.sqrt(6)

axiom a_gt_b_gt_0 (a b : ℝ) : a > b ∧ b > 0
axiom p_gt_0 (p : ℝ) : p > 0
axiom distance_condition (x y : ℝ) (p : ℝ) : ∃ F : ℝ × ℝ, distance_from_point_to_line F.1 F.2 1 (-1) 1 = real.sqrt(2) ∧ F = (p / 2, 0)

open Function

-- The proof problem
theorem find_F_and_ellipse (a b p : ℝ) :
  a > b ∧ b > 0 ∧ p > 0 ∧
  (∃ F : ℝ × ℝ, distance_from_point_to_line F.1 F.2 1 (-1) 1 = real.sqrt(2) ∧ F = (1, 0)) ∧ 
  (∃ F : ℝ × ℝ, F = (1, 0) ∧ ellipse (3 / 2) (real.sqrt(6)) a b = 1) →
  ellipse (x y 3 2) a b = 1 :=
by
  sorry

end find_F_and_ellipse_l581_581712


namespace initial_roses_in_vase_l581_581151

theorem initial_roses_in_vase (added_roses current_roses : ℕ) (h1 : added_roses = 8) (h2 : current_roses = 18) : 
  current_roses - added_roses = 10 :=
by
  sorry

end initial_roses_in_vase_l581_581151


namespace nancy_museum_pictures_l581_581178

theorem nancy_museum_pictures (zoo_pictures museum_pictures deleted_pictures remaining_pictures total_pictures : ℕ)
  (h1 : zoo_pictures = 49)
  (h2 : deleted_pictures = 38)
  (h3 : remaining_pictures = 19)
  (h4 : total_pictures = remaining_pictures + deleted_pictures) :
  museum_pictures = total_pictures - zoo_pictures :=
by 
  have h_total : total_pictures = 57, from calc
    total_pictures = remaining_pictures + deleted_pictures : h4
                ... = 19 + 38                       : by rw [h3, h2]
                ... = 57                            : by norm_num,
  have h_museum : museum_pictures = total_pictures - zoo_pictures, from calc
    museum_pictures = total_pictures - zoo_pictures : by sorry,
  exact h_museum

end nancy_museum_pictures_l581_581178


namespace injective_function_inequality_unique_l581_581272

theorem injective_function_inequality_unique (f : ℝ → ℝ) 
  (H_injective : Function.Injective f) 
  (H_condition : ∀ x ∈ ℝ, ∀ n ∈ ℕ, 
    |∑ i in Finset.range n, (i+1) * (f(x + i + 1) - f(f(x + i)))| ≤ 2019) :
  ∀ x : ℝ, f(x) = x + 1 :=
by 
  sorry

end injective_function_inequality_unique_l581_581272


namespace inequality_solution_l581_581098

theorem inequality_solution (x : ℝ) :
  (6 * x^2 + 18 * x - 64) / ((3 * x - 2) * (x + 5)) < 2 ↔ x ∈ set.Ioo (-5 : ℝ) (2 / 3 : ℝ) :=
sorry

end inequality_solution_l581_581098


namespace count_perfect_squares_cubes_under_1000_l581_581752

-- Condition definitions
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, m * m = n
def isPerfectCube (n : Nat) : Prop := ∃ m : Nat, m * m * m = n

-- Main proof problem
theorem count_perfect_squares_cubes_under_1000 : 
  let count (p : Nat -> Prop) := Finset.card (Finset.filter p (Finset.range 1000))
  count (λ n, isPerfectSquare n ∨ isPerfectCube n) = 37 := 
sorry

end count_perfect_squares_cubes_under_1000_l581_581752


namespace smallest_sum_l581_581948

-- Define the 7 digits available
def seven_digits : list ℕ := [0, 4, 5, 6, 7, 8, 9]

-- Define the function to calculate the sum of two 3-digit numbers
def digit_sum {a b c d e f : ℕ} (h1 : a ∈ seven_digits) (h2 : b ∈ seven_digits) (h3 : c ∈ seven_digits)
  (h4 : d ∈ seven_digits) (h5 : e ∈ seven_digits) (h6 : f ∈ seven_digits) : ℕ :=
  100 * a + 10 * b + c + (100 * d + 10 * e + f)

-- Prove that the minimal sum achievable is 534
theorem smallest_sum (a b c d e f : ℕ) (h : ∀ x ∈ seven_digits, x ∉ [a, b, c, d, e, f]) :
  digit_sum (by simp) (by simp) (by simp) (by simp) (by simp) (by simp) = 534 :=
sorry

end smallest_sum_l581_581948


namespace edy_phone_number_probability_l581_581642

def total_phone_numbers : ℕ := 10^7

def favorable_phone_numbers : ℕ :=
  (∑ k in finset.range 5, 
    ((nat.choose 9 k) * 
     (∑ i in finset.range (k+1), 
      ((-1)^(k - i) * (nat.choose k i) * (i + 1)^7))))

theorem edy_phone_number_probability :
  (favorable_phone_numbers : ℚ) / total_phone_numbers = 0.41032 :=
sorry

end edy_phone_number_probability_l581_581642


namespace find_third_number_l581_581983

theorem find_third_number : ∃ (x : ℝ), 0.3 * 0.8 + x * 0.5 = 0.29 ∧ x = 0.1 :=
by
  use 0.1
  sorry

end find_third_number_l581_581983


namespace solution_exists_real_solution_31_l581_581662

theorem solution_exists_real_solution_31 :
  ∃ x : ℝ, (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10 ∧ 
            (x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15) :=
sorry

end solution_exists_real_solution_31_l581_581662


namespace smallest_number_of_coins_l581_581520

theorem smallest_number_of_coins :
  ∃ (n : ℕ), (∀ (a : ℕ), 5 ≤ a ∧ a < 100 → 
    ∃ (c : ℕ → ℕ), (a = 5 * c 0 + 10 * c 1 + 25 * c 2) ∧ 
    (c 0 + c 1 + c 2 = n)) ∧ n = 9 :=
by
  sorry

end smallest_number_of_coins_l581_581520


namespace perfect_squares_and_cubes_l581_581777

def is_perfect_square (n: ℕ) : Prop := ∃ k: ℕ, k ^ 2 = n
def is_perfect_cube (n: ℕ) : Prop := ∃ k: ℕ, k ^ 3 = n
def is_perfect_sixth_power (n: ℕ) : Prop := ∃ k: ℕ, k ^ 6 = n

def num_perfect_squares (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_square n).length
def num_perfect_cubes (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_cube n).length
def num_perfect_sixth_powers (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_sixth_power n).length

theorem perfect_squares_and_cubes (bound: ℕ) (hbound: bound = 1000) :
  num_perfect_squares bound + num_perfect_cubes bound - num_perfect_sixth_powers bound = 37 :=
by
  -- conditions:
  have ps := num_perfect_squares bound
  have pc := num_perfect_cubes bound
  have p6 := num_perfect_sixth_powers bound
  sorry

end perfect_squares_and_cubes_l581_581777


namespace count_perfect_squares_cubes_under_1000_l581_581757

-- Condition definitions
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, m * m = n
def isPerfectCube (n : Nat) : Prop := ∃ m : Nat, m * m * m = n

-- Main proof problem
theorem count_perfect_squares_cubes_under_1000 : 
  let count (p : Nat -> Prop) := Finset.card (Finset.filter p (Finset.range 1000))
  count (λ n, isPerfectSquare n ∨ isPerfectCube n) = 37 := 
sorry

end count_perfect_squares_cubes_under_1000_l581_581757


namespace numPerfectSquaresOrCubesLessThan1000_l581_581770

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def isSixthPower (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

theorem numPerfectSquaresOrCubesLessThan1000 : 
  ∃ n : ℕ, n = 38 ∧ ∀ k < 1000, (isPerfectSquare k ∨ isPerfectCube k → k) ↔ n :=
by
  sorry

end numPerfectSquaresOrCubesLessThan1000_l581_581770


namespace domain_of_function_l581_581120

-- Definition of the conditions
def sin_nonneg (x : ℝ) : Prop := sin x ≥ 0
def cos_le_half (x : ℝ) : Prop := cos x ≤ 1/2

-- Definition of the domain of the function
noncomputable def function_domain : Set ℝ :=
  {x | ∃ k : ℤ, (π/3 + 2 * k * π ≤ x ∧ x ≤ π + 2 * k * π)}

-- Proposition stating that the domain of the function y = sqrt(sin x) + sqrt(1/2 - cos x)
-- is equivalent to the defined domain function_domain
theorem domain_of_function (x : ℝ) :
  (sin_nonneg x ∧ cos_le_half x) ↔ x ∈ function_domain :=
by
  sorry

end domain_of_function_l581_581120


namespace clock_hands_angle_3_15_l581_581615

def hour_hand_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let base_angle := hours * 30
  let additional_angle := minutes * 0.5
  base_angle + additional_angle

def minute_hand_angle (minutes : ℕ) : ℝ :=
  minutes * 6

theorem clock_hands_angle_3_15 : abs (hour_hand_angle 3 15 - minute_hand_angle 15) = 7.5 :=
  sorry

end clock_hands_angle_3_15_l581_581615


namespace work_completion_days_l581_581557

theorem work_completion_days (x : ℕ) (h_ratio : 5 * 18 = 3 * 30) : 30 = 30 :=
by {
    sorry
}

end work_completion_days_l581_581557


namespace perfect_squares_and_cubes_l581_581778

def is_perfect_square (n: ℕ) : Prop := ∃ k: ℕ, k ^ 2 = n
def is_perfect_cube (n: ℕ) : Prop := ∃ k: ℕ, k ^ 3 = n
def is_perfect_sixth_power (n: ℕ) : Prop := ∃ k: ℕ, k ^ 6 = n

def num_perfect_squares (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_square n).length
def num_perfect_cubes (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_cube n).length
def num_perfect_sixth_powers (bound: ℕ) : ℕ := (List.range bound).filter (λ n, is_perfect_sixth_power n).length

theorem perfect_squares_and_cubes (bound: ℕ) (hbound: bound = 1000) :
  num_perfect_squares bound + num_perfect_cubes bound - num_perfect_sixth_powers bound = 37 :=
by
  -- conditions:
  have ps := num_perfect_squares bound
  have pc := num_perfect_cubes bound
  have p6 := num_perfect_sixth_powers bound
  sorry

end perfect_squares_and_cubes_l581_581778


namespace numPerfectSquaresOrCubesLessThan1000_l581_581769

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

def isPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def isSixthPower (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

theorem numPerfectSquaresOrCubesLessThan1000 : 
  ∃ n : ℕ, n = 38 ∧ ∀ k < 1000, (isPerfectSquare k ∨ isPerfectCube k → k) ↔ n :=
by
  sorry

end numPerfectSquaresOrCubesLessThan1000_l581_581769


namespace pens_left_in_jar_l581_581999

theorem pens_left_in_jar : 
  ∀ (initial_blue_pens initial_black_pens initial_red_pens removed_blue_pens removed_black_pens : ℕ),
  initial_blue_pens = 9 →
  initial_black_pens = 21 →
  initial_red_pens = 6 →
  removed_blue_pens = 4 →
  removed_black_pens = 7 →
  (initial_blue_pens - removed_blue_pens) + (initial_black_pens - removed_black_pens) + initial_red_pens = 25 :=
begin
  intros initial_blue_pens initial_black_pens initial_red_pens removed_blue_pens removed_black_pens,
  intros h1 h2 h3 h4 h5,
  simp [h1, h2, h3, h4, h5],
  norm_num,
end

end pens_left_in_jar_l581_581999


namespace number_of_perfect_squares_and_cubes_l581_581763

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end number_of_perfect_squares_and_cubes_l581_581763


namespace at_least_3_out_of_4_cured_is_0_9477_l581_581136

open ProbabilityTheory

noncomputable def probability_cured := 0.9

def prob_at_least_3_out_of_4_cured : ℝ :=
  (4.choose 3) * (probability_cured ^ 3) * ((1 - probability_cured) ^ 1) +
  (probability_cured ^ 4)

theorem at_least_3_out_of_4_cured_is_0_9477 :
  prob_at_least_3_out_of_4_cured = 0.9477 :=
by
  sorry

end at_least_3_out_of_4_cured_is_0_9477_l581_581136


namespace nigella_base_salary_is_3000_l581_581876

noncomputable def nigella_base_salary : ℝ :=
  let house_A_cost := 60000
  let house_B_cost := 3 * house_A_cost
  let house_C_cost := (2 * house_A_cost) - 110000
  let commission_A := 0.02 * house_A_cost
  let commission_B := 0.02 * house_B_cost
  let commission_C := 0.02 * house_C_cost
  let total_earnings := 8000
  let total_commission := commission_A + commission_B + commission_C
  total_earnings - total_commission

theorem nigella_base_salary_is_3000 : 
  nigella_base_salary = 3000 :=
by sorry

end nigella_base_salary_is_3000_l581_581876


namespace find_n_with_divisors_sum_l581_581420

theorem find_n_with_divisors_sum (n : ℕ) (d1 d2 d3 d4 : ℕ)
  (h1 : d1 = 1) (h2 : d2 = 2) (h3 : d3 = 5) (h4 : d4 = 10) 
  (hd : n = 130) : d1^2 + d2^2 + d3^2 + d4^2 = n :=
sorry

end find_n_with_divisors_sum_l581_581420


namespace minimum_value_of_function_l581_581283

open Real

/-- The minimum value of the function f(x) = sin^2(x) + sin(x) cos(x) + 1 is (3 - sqrt(2)) / 2. -/
theorem minimum_value_of_function : 
  ∃ x : ℝ, 
    ∀ y : ℝ, (sin y^2 + sin y * cos y + 1) ≥ (3 - sqrt 2) / 2 :=
begin
  sorry
end

end minimum_value_of_function_l581_581283


namespace triangle_angle_opposite_c_l581_581815

theorem triangle_angle_opposite_c (a b c : ℝ) (x : ℝ) 
  (ha : a = 2) (hb : b = 2) (hc : c = 4) : x = 180 :=
by 
  -- proof steps are not required as per the instruction
  sorry

end triangle_angle_opposite_c_l581_581815


namespace part1_find_angle_A_part2_find_perimeter_l581_581816

-- Part 1: Prove A = π/3 given conditions of the problem.
theorem part1_find_angle_A (a b c : ℝ) (A B C : ℝ) 
  (h1 : b = c)  -- since 'bc = 4' and 'b = c' would be necessary for later solution
  (h_triangle : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h2 : a = 2)
  (h_cos_relation : c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A) :
  A = π / 3 :=
by
  sorry

-- Part 2: Prove the perimeter is 6 given area and certain sides.
theorem part2_find_perimeter (a b c : ℝ) (A : ℝ)
  (h1 : b = c)  -- since 'bc = 4' and 'b = c'
  (h_triangle : 0 < A ∧ A < π)
  (h_area : 0.5 * b * c * Real.sin A = sqrt 3)
  (h_a_value : a = 2)
  (h_cos_relation : c * Real.cos B + b *Real. cos C = 2 * a * Real.cos A) :
  a + b + c = 6 := 
by
  sorry

end part1_find_angle_A_part2_find_perimeter_l581_581816


namespace savings_proportional_l581_581107

theorem savings_proportional (s : ℝ) (h : ℝ) :
  (∀ x y : ℝ, x / y = 150 / 25 → x = (150 * y) / 25) →
  h = 35 →
  s = (150 * 35) / 25 :=
by 
  intros H h_eq
  rw h_eq
  exact H 150 25
  sorry

end savings_proportional_l581_581107


namespace range_of_log_func_l581_581485

noncomputable def quadratic_expr (x : ℝ) : ℝ := -x^2 + 3 * x + 4

noncomputable def log_func (a : ℝ) : ℝ := Real.logBase 0.4 a

theorem range_of_log_func :
  (∀ x : ℝ, 0 < quadratic_expr x) →
  ∀ y : ℝ, ∃ x : ℝ, y = log_func (quadratic_expr x) ↔ y ∈ [-2, ∞) :=
by 
  intros h y
  have := λ x, log_func (quadratic_expr x)
  sorry

end range_of_log_func_l581_581485


namespace A_E_not_third_l581_581823

-- Define the runners and their respective positions.
inductive Runner
| A : Runner
| B : Runner
| C : Runner
| D : Runner
| E : Runner
open Runner

variable (position : Runner → Nat)

-- Conditions
axiom A_beats_B : position A < position B
axiom C_beats_D : position C < position D
axiom B_beats_E : position B < position E
axiom D_after_A_before_B : position A < position D ∧ position D < position B

-- Prove that A and E cannot be in third place.
theorem A_E_not_third : position A ≠ 3 ∧ position E ≠ 3 :=
sorry

end A_E_not_third_l581_581823


namespace inequality_proof_l581_581685

variable {n : ℕ}
variable {x y : Fin n → ℝ}

-- Defining the conditions
def x_i_bounds (i : Fin n) : Prop := 0 < x i ∧ x i ≤ 1
def y_i_bounds (i : Fin n) : Prop := 0 < y i ∧ y i ≤ 1
def x_y_sum (i : Fin n) : Prop := x i + y i = 1

-- Main theorem
theorem inequality_proof (m : ℕ) (hx : ∀ i, x_i_bounds i) (hy : ∀ i, y_i_bounds i) (hxy : ∀ i, x_y_sum i) :
  (1 - ∏ i in Finset.univ, x i)^m + ∏ i in Finset.univ, (1 - (y i)^m) ≥ 1 :=
sorry

end inequality_proof_l581_581685


namespace most_reasonable_sampling_method_is_stratified_l581_581212

def population_has_significant_differences 
    (grades : List String)
    (understanding : String → ℕ)
    : Prop := sorry -- This would be defined based on the details of "significant differences"

theorem most_reasonable_sampling_method_is_stratified
    (grades : List String)
    (understanding : String → ℕ)
    (h : population_has_significant_differences grades understanding)
    : (method : String) → (method = "Stratified sampling") :=
sorry

end most_reasonable_sampling_method_is_stratified_l581_581212


namespace sage_can_determine_weight_l581_581199

theorem sage_can_determine_weight : 
  ∃ (wei : ℕ → ℕ), 
    (∀ i, i ∈ {0, 1, 2, 3, 4, 5, 6} → wei i ∈ {7, 8, 9, 10, 11, 12, 13}) →  -- weights per coin in each bag
    (∀ B : ℕ, B ∈ {0, 1, 2, 3, 4, 5, 6} →  -- B indicates the specified bag
      ∃ f : ℕ → {0, 1, ..., 6} → Prop,
        f 10 = (λ i, 7 + 8 + 9 + 10 + 11 + 12 + 13 ≤ 70 * wei B) ∧
        ( (70 * wei (f 10) = 70 * wei B) ∨
          ((70 * wei (f 10) < 70 * wei B ∧ 70 * wei B ≤ 80 * wei (f 1)) ∨
           (80 * wei (f 1) < 70 * wei B ∧ 70 * wei B ≤ 90 * wei (f 2)) ∨
           (90 * wei (f 2) < 70 * wei B ∧ 70 * wei B ≤ 100 * wei (f 3)) ∨
           (100 * wei (f 3) < 70 * wei B ∧ 70 * wei B ≤ 110 * wei (f 4)) ∨
           (110 * wei (f 4) < 70 * wei B ∧ 70 * wei B ≤ 120 * wei (f 5)) ∨
           (120 * wei (f 5) < 70 * wei B ∧ 70 * wei B ≤ 130 * wei (f 6))))) :=
sorry

end sage_can_determine_weight_l581_581199


namespace perfect_squares_and_cubes_count_lt_1000_l581_581785

theorem perfect_squares_and_cubes_count_lt_1000 : 
  let perfect_square := λ n : ℕ, ∃ k : ℕ, n = k^2
  let perfect_cube := λ n : ℕ, ∃ k : ℕ, n = k^3
  let sixth_power := λ n : ℕ, ∃ k : ℕ, n = k^6
  (count (λ n, n < 1000 ∧ (perfect_square n ∨ perfect_cube n)) (Finset.range 1000)) = 38 :=
by {
  sorry
}

end perfect_squares_and_cubes_count_lt_1000_l581_581785


namespace length_of_second_train_l581_581554

-- Definitions of the conditions
def length_first_train : ℝ := 220
def speed_first_train_kmh : ℝ := 120
def speed_second_train_kmh : ℝ := 80
def crossing_time : ℝ := 9
def kmh_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

-- Prove the length of the second train
theorem length_of_second_train :
  let S_1 := kmh_to_mps speed_first_train_kmh in
  let S_2 := kmh_to_mps speed_second_train_kmh in
  let relative_speed := S_1 + S_2 in
  let total_distance := relative_speed * crossing_time in
  let L_1 := length_first_train in
  ∃ L_2 : ℝ, abs (total_distance - (L_1 + L_2)) < 0.01 :=
sorry

end length_of_second_train_l581_581554


namespace coordinates_of_A_l581_581378

-- Definition of the point A with coordinates (-1, 3)
def point_A : ℝ × ℝ := (-1, 3)

-- Statement that the coordinates of point A with respect to the origin are (-1, 3)
theorem coordinates_of_A : point_A = (-1, 3) := by
  sorry

end coordinates_of_A_l581_581378


namespace q_q_2_neg2_q_neg3_neg1_l581_581868

def q (x y : ℝ) : ℝ :=
if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
else if x < 0 ∧ y < 0 then x - 3 * y
else 2 * x + y

theorem q_q_2_neg2_q_neg3_neg1 : q (q 2 (-2)) (q (-3) (-1)) = 2 :=
by
  sorry

end q_q_2_neg2_q_neg3_neg1_l581_581868


namespace best_approximation_is_mean_l581_581384

def best_approximation (a : ℝ) (data : list ℝ) : ℝ :=
  sum (data.map (λ x, (x - a)^2))

def mean (data : list ℝ) : ℝ :=
  (sum data) / (data.length)

theorem best_approximation_is_mean (n : ℕ) (a1 a2 … an : ℝ) (h : list ℝ) :
  best_approximation (mean h) h ≤ best_approximation a h :=
sorry

end best_approximation_is_mean_l581_581384


namespace group_division_l581_581935

theorem group_division (total_students groups_per_group : ℕ) (h1 : total_students = 30) (h2 : groups_per_group = 5) : 
  (total_students / groups_per_group) = 6 := 
by 
  sorry

end group_division_l581_581935


namespace james_bought_boxes_l581_581039

theorem james_bought_boxes (pouches_per_box : ℕ) (total_spent : ℕ) (cost_per_pouch : ℚ) : 
  pouches_per_box = 6 →
  total_spent = 1200 →
  cost_per_pouch = 20 →
  (total_spent.to_rat / cost_per_pouch : ℚ) / pouches_per_box = 10 :=
by
  intros h1 h2 h3
  have h2' : total_spent.to_rat = 12 := by linarith [h2]
  have h3' : cost_per_pouch / 100 = 0.20 := by norm_num [h3]
  sorry

end james_bought_boxes_l581_581039


namespace horner_rule_example_l581_581942

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_rule_example : f 2 = 62 := by
  sorry

end horner_rule_example_l581_581942


namespace car_speed_first_hour_l581_581142

theorem car_speed_first_hour (x : ℕ) (hx : x = 65) : 
  let speed_second_hour := 45 
  let average_speed := 55
  (x + 45) / 2 = 55 
  :=
  by
  sorry

end car_speed_first_hour_l581_581142


namespace unique_extreme_value_range_l581_581001

noncomputable def f (a x : ℝ) : ℝ := x^2 + (a+3)*x + Real.log x

theorem unique_extreme_value_range (a : ℝ) :
  (∃! c ∈ Set.Ioo 1 2, IsLocalExtreme (f a) c) ↔ -15/2 < a ∧ a < -6 := sorry

end unique_extreme_value_range_l581_581001


namespace range_of_a_l581_581328

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.exp x - 2 * a * x - a ^ 2 + 3

theorem range_of_a (h : ∀ x, x ≥ 0 → f x a - x ^ 2 ≥ 0) :
  -Real.sqrt 5 ≤ a ∧ a ≤ 3 - Real.log 3 := sorry

end range_of_a_l581_581328


namespace vector_proof_l581_581729

def vector_op (a b : ℤ × ℤ) : ℤ × ℤ :=
  (a.1 * b.1 - a.2 * b.2, a.2 * b.1 + a.1 * b.2)

theorem vector_proof (m n p q : ℤ) :
  let a := (m, n)
  let b := (p, q)
  vector_op (1, 2) (2, 1) = (0, 5) → 
  vector_op a b = (5, 0) →
  (m^2 + n^2 < 25) →
  (p^2 + q^2 < 25) →
  a = (2, 1) ∧ b = (2, -1) := 
by {
  intros h1 h2 h3 h4,
  sorry
}

end vector_proof_l581_581729


namespace symmetric_circle_l581_581122

variable (x y : ℝ)

def original_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5
def line_of_symmetry (x y : ℝ) : Prop := y = x

theorem symmetric_circle :
  (∃ x y, original_circle x y) → (x^2 + (y + 2)^2 = 5) :=
sorry

end symmetric_circle_l581_581122


namespace integer_average_problem_l581_581148

theorem integer_average_problem (a b c d : ℤ) (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
(h_max : max a (max b (max c d)) = 90) (h_min : min a (min b (min c d)) = 29) : 
(a + b + c + d) / 4 = 45 := 
sorry

end integer_average_problem_l581_581148


namespace necessary_but_not_sufficient_l581_581114

-- Definitions extracted from the problem conditions
def isEllipse (k : ℝ) : Prop := (9 - k > 0) ∧ (k - 7 > 0) ∧ (9 - k ≠ k - 7)

-- The necessary but not sufficient condition for the ellipse equation
theorem necessary_but_not_sufficient : 
  (7 < k ∧ k < 9) → isEllipse k → (isEllipse k ↔ (7 < k ∧ k < 9)) := 
by 
  sorry

end necessary_but_not_sufficient_l581_581114


namespace deck_width_l581_581957

theorem deck_width (w : ℝ) : 
  let pool_length := 10 
  let pool_width := 12 
  let total_area := 360 
  (pool_length + 2 * w) * (pool_width + 2 * w) = total_area → w = 4 := 
by
  intros
  let pool_area := pool_length * pool_width
  let pool_and_deck_area := total_area
  have h_eq : (pool_length + 2 * w) * (pool_width + 2 * w) = pool_and_deck_area := by assumption
  sorry

end deck_width_l581_581957


namespace last_score_is_100_l581_581874

-- Define the scores and the condition of integer averages
def scores : List ℕ := [60, 65, 70, 75, 85, 90, 100]

-- Total sum of the scores
def total_sum (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Define the condition that the average must be an integer after each score entry
def average_is_integer (l : List ℕ) : Prop :=
  ∀ n : ℕ, n < l.length → total_sum (l.take (n + 1)) % (n + 1) = 0

-- Prove that under the given conditions, the last score must be 100
theorem last_score_is_100 (h : average_is_integer scores) : List.get scores 6 = 100 := 
  sorry

end last_score_is_100_l581_581874


namespace problem_statement_l581_581055

noncomputable def f (x : ℝ) : ℝ := 2 ^ (x - 1)

theorem problem_statement 
    (h_even : ∀ x : ℝ, f(x) = f(-x))
    (h_period : ∀ x : ℝ, f(x + 1) = f(x - 1))
    (h_def : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x) = 2 ^ (x - 1)) :
    (∀ x : ℝ, f(x + 2) = f(x)) ∧  -- (1)
    (∀ x : ℝ, 2 < x ∧ x < 3 → f(x) > f(x - 0.1)) ∧  -- (2)
    (∀ x : ℝ, f(2 + x) = f(2 - x)) :=  -- (4)
sorry

end problem_statement_l581_581055


namespace number_of_perfect_squares_and_cubes_l581_581761

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end number_of_perfect_squares_and_cubes_l581_581761


namespace area_of_square_field_l581_581945

theorem area_of_square_field (side_length : ℕ) (h : side_length = 25) :
  side_length * side_length = 625 := by
  sorry

end area_of_square_field_l581_581945


namespace distinct_collections_BIOLOGY_l581_581442

theorem distinct_collections_BIOLOGY :
  let vowels := ['I', 'O', 'O']
  let consonants := ['B', 'L', 'G', 'Y']
  let choose_vowels := 2
  let choose_consonants := 4
  -- Calculate the number of ways to choose 2 vowels from {I, O, O}
  let ways_vowels := (nat.choose 1 2) + (nat.choose 1 1 * nat.choose 1 1) + (nat.choose 2 2)
  -- Calculate the number of ways to choose 4 consonants from {B, L, G, Y}
  let ways_consonants := (nat.choose 4 4)
  -- The total number of distinct collections
  ways_vowels * ways_consonants = 2 := by 
  let vowels := ['I', 'O', 'O']
  let consonants := ['B', 'L', 'G', 'Y']
  let choose_vowels := 2
  let ways_vowels := (nat.choose 1 2) + (nat.choose 1 1 * nat.choose 1 1) + (nat.choose 2 2)
  let ways_consonants := (nat.choose 4 4)
  show ways_vowels * ways_consonants = 2 from sorry

end distinct_collections_BIOLOGY_l581_581442


namespace rotating_right_triangle_results_in_cone_l581_581546

theorem rotating_right_triangle_results_in_cone (T : Triangle) (h : isRightTriangle T) (leg : Side T) :
  ¬(isHypotenuse leg) → 
  resultingShapeFromRotation T leg = Shape.cone :=
by
  sorry

end rotating_right_triangle_results_in_cone_l581_581546


namespace fraction_comparison_l581_581473

noncomputable def one_seventh : ℚ := 1 / 7
noncomputable def decimal_0_point_14285714285 : ℚ := 14285714285 / 10^11
noncomputable def eps_1 : ℚ := 1 / (7 * 10^11)
noncomputable def eps_2 : ℚ := 1 / (7 * 10^12)

theorem fraction_comparison :
  one_seventh = decimal_0_point_14285714285 + eps_1 :=
sorry

end fraction_comparison_l581_581473


namespace circle_rotation_maps_to_circle_l581_581889

noncomputable theory
open_locale classical

-- Define the initial conditions: circle S with center Q and radius R,
-- and a rotation by angle θ about point O.

def center : Type := ℝ × ℝ -- using a 2D real coordinate system for center
def radius (r : ℝ) : Prop := r > 0

structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

def rotate_point (O Q : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ :=
-- dummy definition, as actual implementation is not required
sorry 

theorem circle_rotation_maps_to_circle (O Q : ℝ × ℝ) (R θ : ℝ) (hR : radius R) :
  ∃ Q₁, ∃ S₁ : Circle, S₁.center = rotate_point O Q θ ∧ S₁.radius = R :=
sorry

end circle_rotation_maps_to_circle_l581_581889


namespace presidency_meeting_arrangements_l581_581574

theorem presidency_meeting_arrangements :
  (∃ (schools : Fin 4), 
    ∃ (host_representatives : Fin 5 → ℕ), 
    ∃ (other_representatives : Fin 3 → Fin 5 → ℕ),
    (∀ i, host_representatives i ∈ {0, 1, 2, 3}) ∧
    (∀ j k, other_representatives j k ∈ {0, 1})) →
  ∃ (ways_to_arrange_meeting : ℕ), ways_to_arrange_meeting = 5000 :=
by
  sorry

end presidency_meeting_arrangements_l581_581574


namespace sum_arith_alternating_series_l581_581269

theorem sum_arith_alternating_series : 
  let seq := (λ n : ℕ, if n % 2 = 0 then -2 + 3 * n else -2 + 3 * n + 7)
  let sum := (λ N : ℕ, (∑ n in (Finset.range N), seq n))
  in sum 1669 = 7503 :=
by
  sorry

end sum_arith_alternating_series_l581_581269


namespace ellipse_equation_trajectory_of_N_l581_581714

-- Setup for Question 1
def center := (0, 0)
def right_focus := (3, 0)
def line_L := fun x y => x + 2*y - 2 = 0
def midpoint_AB := (1, 1 / 2)

-- Theorem for Question 1
theorem ellipse_equation (h1 : true) : 
  ∃ (m n : ℝ), (m > n ∧ n > 0 ∧ (3 ^ 2 = m - n) ∧
  ∀ x y, (x + 2 * y - 2 = 0) → 
  ((f : ℝ) = y^2 / n ∧ (f : ℝ) = x^2 / m) ∧
  x + 2 * y = 2 / n ∧ y_1 + y_2 = 1 ) ∧
  (m = 12) ∧ (n = 3) ∧ (true → (∃ k l : ℝ, k^2/m + l^2/n = 1)) := sorry

-- Theorem for Question 2
theorem trajectory_of_N:
  ∀ (A B : (ℝ × ℝ)), 
    (A + B = (2, 1) ∧ A ≠ B ∧ 
    ∀ (N : (ℝ × ℝ)), (N - A) ⊢/ (N - B) = (x - 1)^2 + (y - 1/2)^2 = 25/4) :=
  sorry

end ellipse_equation_trajectory_of_N_l581_581714


namespace arithmetic_sequence_sum_l581_581314

theorem arithmetic_sequence_sum (a : ℕ → Int) (a1 a2017 : Int)
  (h1 : a 1 = a1) 
  (h2017 : a 2017 = a2017)
  (roots_eq : ∀ x, x^2 - 10 * x + 16 = 0 → (x = a1 ∨ x = a2017))
  (arith_seq : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) :
  a 2 + a 1009 + a 2016 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l581_581314


namespace Menelaus_theorem_l581_581879

noncomputable def collinear (P Q R : Point) : Prop :=
∃ l : Line, P ∈ l ∧ Q ∈ l ∧ R ∈ l

variable {ABC : Triangle}
variable {A1 B1 C1 : Point}

/-- Menelaus's theorem applied to triangle ABC and points A1, B1, C1 on sides BC, CA, AB respectively, stating the collinearity of A1, B1, C1 -/
theorem Menelaus_theorem 
  (hA1_on_BC : A1 ∈ line_through ABC.B ABC.C) 
  (hB1_on_CA : B1 ∈ line_through ABC.C ABC.A) 
  (hC1_on_AB : C1 ∈ line_through ABC.A ABC.B):
  collinear A1 B1 C1 ↔ 
  (dist ABC.B A1 / dist A1 ABC.C) * (dist ABC.C B1 / dist B1 ABC.A) * (dist ABC.A C1 / dist C1 ABC.B) = 1 :=
sorry

end Menelaus_theorem_l581_581879


namespace find_angle_AMC_l581_581309

noncomputable def angle_AMC (A B C M : Point) : ℝ :=
  -- Assumptions
  let AB := dist A B in
  let AC := dist A C in
  let ∠BAC := 110 in
  let ∠MBC := 30 in
  let ∠MCB := 25 in

  -- Proof goal
  let ∠AMC := 85 in
  ∠AMC

theorem find_angle_AMC (A B C M : Point)
  (h1 : dist A B = dist A C)
  (h2 : angle A = 110)
  (h3 : 30: ℝ)
  (h4 : 25: ℝ) :
  angle_AMC A B C M = 85 :=
sorry

end find_angle_AMC_l581_581309


namespace proof_angle_ADO_eq_angle_HAN_l581_581414

open EuclideanGeometry

variable (A B C O H M D N : Point)
variable (circumcircle_ABC : Circle ABC.center)
variable (orthocenter_H : ∃ H, Orthocenter_of_triangle A B C H)
variable (M_midpoint_BC : Midpoint M B C)
variable (D_on_BC : ∃ D, ∠BAD = ∠CAD)
variable (MO_intersects_circumcircle_BHC : ∃ N, Line MO meets Circle B H C at N)

noncomputable def angle_ADO_eq_angle_HAN : Prop :=
  ∠ A D O = ∠ H A N

-- Statement of the theorem
theorem proof_angle_ADO_eq_angle_HAN :
  circocenter_O O A B C →
  orthocenter_H H A B C →
  midpoint M B C →
  is_on_BC D A B C →
  intersects_circumcircle N M O B H C →
  angle_ADO_eq_angle_HAN A D O H N :=
by
  intros
  sorry

end proof_angle_ADO_eq_angle_HAN_l581_581414


namespace least_k_even_sum_divisible_by_n_l581_581971

theorem least_k_even_sum_divisible_by_n (n : ℕ) : 
  let k := if n % 2 = 1 then 2 * n else n + 1 in
  ∀ A : Finset ℕ, (A.card = k) → ∃ S : Finset ℕ, (S ⊆ A) ∧ (S.card % 2 = 0) ∧ (S.sum id % n = 0) :=
by
  sorry

end least_k_even_sum_divisible_by_n_l581_581971


namespace Penelope_daily_savings_l581_581080

theorem Penelope_daily_savings
  (total_savings : ℝ)
  (days_in_year : ℕ)
  (h1 : total_savings = 8760)
  (h2 : days_in_year = 365) :
  total_savings / days_in_year = 24 :=
by
  sorry

end Penelope_daily_savings_l581_581080


namespace find_f3_l581_581305

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f3 (h1 : ∀ x : ℝ, f (x + 1) = f (-x - 1))
                (h2 : ∀ x : ℝ, f (2 - x) = -f x) :
  f 3 = 0 := 
sorry

end find_f3_l581_581305


namespace garden_area_l581_581847

variable (L W A : ℕ)
variable (H1 : 3000 = 50 * L)
variable (H2 : 3000 = 15 * (2*L + 2*W))

theorem garden_area : A = 2400 :=
by
  sorry

end garden_area_l581_581847


namespace no_2012_integers_with_product_2_and_sum_0_l581_581393

theorem no_2012_integers_with_product_2_and_sum_0 :
  ¬ ∃ (a : Fin 2012 → Int), (∏ i, a i = 2) ∧ (∑ i, a i = 0) :=
begin
  sorry
end

end no_2012_integers_with_product_2_and_sum_0_l581_581393


namespace count_integers_divisible_by_2_5_7_l581_581347

theorem count_integers_divisible_by_2_5_7 : 
  {n : ℕ | n < 300 ∧ n % 2 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0}.card = 4 := 
by
  sorry

end count_integers_divisible_by_2_5_7_l581_581347


namespace solve_system_of_equations_solve_system_of_inequalities_l581_581902

-- Proof for the system of equations
theorem solve_system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 32) 
  (h2 : 2 * x - y = 0) :
  x = 8 ∧ y = 16 :=
by
  sorry

-- Proof for the system of inequalities
theorem solve_system_of_inequalities (x : ℝ)
  (h3 : 3 * x - 1 < 5 - 2 * x)
  (h4 : 5 * x + 1 ≥ 2 * x + 3) :
  (2 / 3 : ℝ) ≤ x ∧ x < (6 / 5 : ℝ) :=
by
  sorry

end solve_system_of_equations_solve_system_of_inequalities_l581_581902


namespace calculate_a3_l581_581689

theorem calculate_a3 (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, S n = 2^n - 1) (h2 : ∀ n, a n = S n - S (n-1)) : 
  a 3 = 4 :=
by
  sorry

end calculate_a3_l581_581689


namespace sum_of_integers_2004_l581_581088

theorem sum_of_integers_2004 (N : ℕ) (n : ℕ) :
  (∀ n ≥ N, ∃ a : Fin 2004 → ℕ, 
    (n = ∑ i, a i) ∧ 
    ∀ i j : Fin 2004, i < j → a i < a j ∧ 
    ∀ i : Fin 2003, (a i) ∣ (a ⟨i.val + 1, by linarith⟩)) :=
sorry

end sum_of_integers_2004_l581_581088


namespace inequality_proof_l581_581068

variables {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 1/x + 1/y + 1/z = 1) :
  sqrt (x + y * z) + sqrt (y + z * x) + sqrt (z + x * y) ≥ sqrt (x * y * z) + sqrt x + sqrt y + sqrt z := 
sorry

end inequality_proof_l581_581068


namespace age_of_youngest_child_l581_581492

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 60) : 
  x = 6 :=
sorry

end age_of_youngest_child_l581_581492


namespace Tile_in_rectangle_R_l581_581155

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

def X : Tile := ⟨5, 3, 6, 2⟩
def Y : Tile := ⟨3, 6, 2, 5⟩
def Z : Tile := ⟨6, 0, 1, 5⟩
def W : Tile := ⟨2, 5, 3, 0⟩

theorem Tile_in_rectangle_R : 
  X.top = 5 ∧ X.right = 3 ∧ X.bottom = 6 ∧ X.left = 2 ∧ 
  Y.top = 3 ∧ Y.right = 6 ∧ Y.bottom = 2 ∧ Y.left = 5 ∧ 
  Z.top = 6 ∧ Z.right = 0 ∧ Z.bottom = 1 ∧ Z.left = 5 ∧ 
  W.top = 2 ∧ W.right = 5 ∧ W.bottom = 3 ∧ W.left = 0 → 
  (∀ rectangle_R : Tile, rectangle_R = W) :=
by sorry

end Tile_in_rectangle_R_l581_581155


namespace max_levels_passable_prob_level_2_higher_l581_581568
open Classical

-- Definition of passing a level in this game
def passes_level (n : ℕ) : Prop :=
  let die_faces := {1, 2, 3, 4, 5, 6}
  let sums := {s | ∃ rolls : List ℕ, rolls.length = n ∧ (∀ r ∈ rolls, r ∈ die_faces) ∧ s = List.sum rolls}
  ∃ s ∈ sums, s > 2^n

-- Proving the maximum number of levels that can be passed is 4
theorem max_levels_passable : ∃ n : ℕ, (passes_level n ∧ n ≤ 4) ∧ ∀ m > 4, ¬passes_level m :=
sorry

-- Probability of passing the first level is greater than the second
def prob_pass_level_1 := 2 / 3
def prob_pass_level_2 := 5 / 6

theorem prob_level_2_higher : prob_pass_level_2 > prob_pass_level_1 :=
sorry

end max_levels_passable_prob_level_2_higher_l581_581568


namespace total_customers_is_40_l581_581560

-- The number of tables the waiter is attending
def num_tables : ℕ := 5

-- The number of women at each table
def women_per_table : ℕ := 5

-- The number of men at each table
def men_per_table : ℕ := 3

-- The total number of customers at each table
def customers_per_table : ℕ := women_per_table + men_per_table

-- The total number of customers the waiter has
def total_customers : ℕ := num_tables * customers_per_table

theorem total_customers_is_40 : total_customers = 40 :=
by
  -- Proof goes here
  sorry

end total_customers_is_40_l581_581560


namespace number_of_perfect_squares_and_cubes_l581_581765

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end number_of_perfect_squares_and_cubes_l581_581765


namespace point_transformation_l581_581582

theorem point_transformation (a b : ℝ) :
  let P := (a, b)
  let P₁ := (2 * 2 - a, 2 * 3 - b) -- Rotate P 180° counterclockwise around (2, 3)
  let P₂ := (P₁.2, P₁.1)           -- Reflect P₁ about the line y = x
  P₂ = (5, -4) → a - b = 7 :=
by
  intros
  sorry

end point_transformation_l581_581582


namespace triangle_base_increase_l581_581938

def triangleAreas (p : ℝ) (b h : ℝ) : Prop :=
  let areaA := (1/2) * b * (1 + p/100) * 0.80 * h
  let areaB := (1/2) * b * h
  areaA = areaB * 0.9975

theorem triangle_base_increase :
  ∃ p : ℝ, p ≈ 24.69 ∧ ∀ (b h : ℝ), triangleAreas p b h :=
by
  intro b h
  use 24.69
  sorry

end triangle_base_increase_l581_581938


namespace difference_between_mean_and_median_l581_581824

theorem difference_between_mean_and_median :
  let students : ℕ := 20
  let pct_65 : ℝ := 0.15
  let pct_75 : ℝ := 0.20
  let pct_88 : ℝ := 0.25
  let pct_92 : ℝ := 0.10
  let pct_100 : ℝ := 0.30
  let count_65 : ℕ := (pct_65 * students).to_nat
  let count_75 : ℕ := (pct_75 * students).to_nat
  let count_88 : ℕ := (pct_88 * students).to_nat
  let count_92 : ℕ := (pct_92 * students).to_nat
  let count_100 : ℕ := (pct_100 * students).to_nat
  let sorted_scores := list.repeat 65 count_65 ++ list.repeat 75 count_75 ++ list.repeat 88 count_88 ++ list.repeat 92 count_92 ++ list.repeat 100 count_100
  let median : ℝ := (
    let mid := sorted_scores.length / 2
    if sorted_scores.length % 2 = 0 then
      (sorted_scores.nth_le (mid - 1) sorry + sorted_scores.nth_le mid sorry) / 2
    else
      sorted_scores.nth_le mid sorry
  )
  let mean : ℝ := (65 * count_65 + 75 * count_75 + 88 * count_88 + 92 * count_92 + 100 * count_100) / students
  mean - median = (-2) := sorry

end difference_between_mean_and_median_l581_581824


namespace raviraj_cycle_distance_l581_581880

theorem raviraj_cycle_distance :
  ∃ (d : ℝ), d = Real.sqrt ((425: ℝ)^2 + (200: ℝ)^2) ∧ d = 470 := 
by
  sorry

end raviraj_cycle_distance_l581_581880


namespace optimal_rental_decision_optimal_purchase_decision_l581_581988

-- Definitions of conditions
def monthly_fee_first : ℕ := 50000
def monthly_fee_second : ℕ := 10000
def probability_seizure : ℚ := 0.5
def moving_cost : ℕ := 70000
def months_first_year : ℕ := 12
def months_seizure : ℕ := 4
def months_after_seizure : ℕ := months_first_year - months_seizure
def purchase_cost : ℕ := 2000000
def installment_period : ℕ := 36

-- Proving initial rental decision
theorem optimal_rental_decision :
  let annual_cost_first := monthly_fee_first * months_first_year
  let annual_cost_second := (monthly_fee_second * months_seizure) + (monthly_fee_first * months_after_seizure) + moving_cost
  annual_cost_second < annual_cost_first := 
by
  sorry

-- Proving purchasing decision
theorem optimal_purchase_decision :
  let total_rent_cost_after_seizure := (monthly_fee_second * months_seizure) + moving_cost + (monthly_fee_first * (4 * months_first_year - months_seizure))
  let total_purchase_cost := purchase_cost
  total_purchase_cost < total_rent_cost_after_seizure :=
by
  sorry

end optimal_rental_decision_optimal_purchase_decision_l581_581988


namespace boat_separation_one_minute_before_collision_l581_581516

theorem boat_separation_one_minute_before_collision :
  ∀ (speed1 speed2 : ℝ) (initial_distance : ℝ),
  speed1 = 5 → speed2 = 21 → initial_distance = 20 →
  let combined_speed := (speed1 + speed2) / 60 in
  let time_to_collide := initial_distance / combined_speed in
  let one_min_before_collide := time_to_collide - 1 in
  (one_min_before_collide * combined_speed) = 0.4333 :=
by
  intros speed1 speed2 initial_distance
  intros h_speed1 h_speed2 h_initial_distance
  rw [h_speed1, h_speed2, h_initial_distance]
  let combined_speed := (5 + 21) / 60
  let time_to_collide := 20 / combined_speed
  let one_min_before_collide := time_to_collide - 1
  show (one_min_before_collide * combined_speed) = 0.4333
  sorry

end boat_separation_one_minute_before_collision_l581_581516


namespace area_of_triangle_l581_581906

-- Definitions based on the given conditions
-- Let x be AB and y be BC
variables (x y : ℝ)
noncomputable def AC : ℝ := 9

-- The conditions
def is_right_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] 
  [has_angle A B C 90] := sorry

def satisfies_angle_bisector_theorem (A B C P : Type) [metric_space A] 
  [metric_space B] [metric_space C] [metric_space P] (AB BC AP CP : ℝ) :=
  AP = 3 ∧ CP = 6 ∧ AP / CP = AB / BC

-- The statement of the problem to prove
theorem area_of_triangle (A B C P : Type) [metric_space A] [metric_space B] [metric_space C] 
  [metric_space P]
  (h_right_triangle : is_right_triangle A B C)
  (h_angle_bisector_theorem : satisfies_angle_bisector_theorem A B C P x y 3 6)
  (AC_eq : AC = 9) :
  (1 / 2) * x * y = 81 / 5 :=
sorry

end area_of_triangle_l581_581906


namespace sum_of_x_main_l581_581668

theorem sum_of_x (x : ℝ) (h : 3^(x^2 - 4 * x - 4) = 9^(x - 5)) : x = 3 + sqrt 3 ∨ x = 3 - sqrt 3 :=
sorry

lemma sum_of_values: (3 + sqrt 3) + (3 - sqrt 3) = 6 :=
by ring

theorem main : ∑ x in {x | 3^(x^2 - 4 * x - 4) = 9^(x - 5)}, x = 6 :=
begin
  have hx1 : 3^(3 + sqrt 3 - 4 * (3 + sqrt 3) - 4) = 9^(3 + sqrt 3 - 5) := by sorry,
  have hx2 : 3^(3 - sqrt 3 - 4 * (3 - sqrt 3) - 4) = 9^(3 - sqrt 3 - 5) := by sorry,
  have key : {x | 3^(x^2 - 4 * x - 4) = 9^(x - 5)} = {3 + sqrt 3, 3 - sqrt 3} := by sorry,
  rw key,
  exact sum_of_values,
end

end sum_of_x_main_l581_581668


namespace minimum_sequence_length_l581_581419

def S : Set ℕ := {1, 2, 3, 4}

def isValidSequence (a : List ℕ) : Prop :=
  ∀ B : Finset ℕ, B ⊆ S → B.Nonempty → 
  ∃ (l : List ℕ), l.length = B.card ∧ l.toFinset = B ∧ l ⊆ a

theorem minimum_sequence_length :
  ∃ a : List ℕ, isValidSequence a ∧ a.length = 8 :=
sorry

end minimum_sequence_length_l581_581419


namespace rectangular_to_polar_correct_l581_581627

-- Define the point in rectangular coordinates.
def point_rectangular : (ℝ × ℝ) := (8, 4 * Real.sqrt 2)

-- Define the polar coordinates conversion function.
noncomputable def to_polar (x y : ℝ) : ℝ × ℝ :=
let r := Real.sqrt (x^2 + y^2)
let θ := Real.atan (y / x)
(r, θ)

-- Define the expected result for the given point (8, 4sqrt(2)).
def expected_polar := (4 * Real.sqrt 6, Real.pi / 8)

-- Theorem stating that converting (8, 4sqrt(2)) to polar coordinates gives (4sqrt(6), pi/8).
theorem rectangular_to_polar_correct :
  to_polar 8 (4 * Real.sqrt 2) = expected_polar :=
by
  sorry

end rectangular_to_polar_correct_l581_581627


namespace prob_red_on_fourth_draw_expectation_xi_l581_581818

-- Definitions for the problem conditions
def total_balls : ℕ := 8
def red_balls : ℕ := 5
def white_balls : ℕ := 3
def draws (n : ℕ) : ℕ := n

-- Probability calculation for drawing a red ball on the fourth draw
theorem prob_red_on_fourth_draw : 
  (probability_of_drawing_red_ball_on_nth_draw 4 total_balls red_balls white_balls) = 5 / 14 :=
by
  sorry

-- Let ξ be the number of red balls drawn in the first three draws
def xi : ℕ := number_of_red_balls_in_first_n_draws 3 total_balls red_balls white_balls

-- Expectation calculation of ξ
theorem expectation_xi : expectation_of_xi xi = -- provide the expected value here :=
by
  sorry

end prob_red_on_fourth_draw_expectation_xi_l581_581818


namespace distribution_schemes_count_l581_581265

def students : Finset (Fin 4) := {0, 1, 2, 3}
def villages : Finset (Fin 3) := {0, 1, 2}

theorem distribution_schemes_count (h : ∀ village, 1 ≤ (students.filter (λ student, ∃ v ∈ villages, true)).card) :
  students.card = 4 ∧ villages.card = 3 →
  ∃ n, n = 36 :=
by
  intro h_card
  have h_proof : ∀ village, 1 ≤ (students.filter (λ student, ∃ v, v ∈ villages)).card, from sorry
  exact ⟨36, rfl⟩

end distribution_schemes_count_l581_581265


namespace find_other_person_weight_l581_581915

noncomputable def other_person_weight (n avg new_avg W1 : ℕ) : ℕ :=
  let total_initial := n * avg
  let new_n := n + 2
  let total_new := new_n * new_avg
  total_new - total_initial - W1

theorem find_other_person_weight:
  other_person_weight 23 48 51 78 = 93 := by
  sorry

end find_other_person_weight_l581_581915


namespace find_number_l581_581077

theorem find_number (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) (number : ℕ) :
  quotient = 9 ∧ remainder = 1 ∧ divisor = 30 → number = 271 := by
  intro h
  sorry

end find_number_l581_581077


namespace tan_ratio_l581_581682

variable (α β : ℝ)

def condition1 : Prop := cos (π / 4 - α) = 3 / 5
def condition2 : Prop := sin (5 * π / 4 + β) = -12 / 13
def condition3 : Prop := α ∈ Ioo (π / 4) (3 * π / 4)
def condition4 : Prop := β ∈ Ioo 0 (π / 4)

theorem tan_ratio (h1 : condition1 α) (h2 : condition2 β) 
                  (h3 : condition3 α) (h4 : condition4 β) :
  (Real.tan α / Real.tan β) = -17 :=
sorry

end tan_ratio_l581_581682


namespace penelope_food_intake_l581_581081

theorem penelope_food_intake
(G P M E : ℕ) -- Representing amount of food each animal eats per day
(h1 : P = 10 * G) -- Penelope eats 10 times Greta's food
(h2 : M = G / 100) -- Milton eats 1/100 of Greta's food
(h3 : E = 4000 * M) -- Elmer eats 4000 times what Milton eats
(h4 : E = P + 60) -- Elmer eats 60 pounds more than Penelope
(G_val : G = 2) -- Greta eats 2 pounds per day
: P = 20 := -- Prove Penelope eats 20 pounds per day
by
  rw [G_val] at h1 -- Replace G with 2 in h1
  norm_num at h1 -- Evaluate the expression in h1
  exact h1 -- Conclude P = 20

end penelope_food_intake_l581_581081


namespace resulting_shape_is_cone_l581_581549

-- Assume we have a right triangle
structure right_triangle (α β γ : ℝ) : Prop :=
  (is_right : γ = π / 2)
  (sum_of_angles : α + β + γ = π)
  (acute_angles : α < π / 2 ∧ β < π / 2)

-- Assume we are rotating around one of the legs
def rotate_around_leg (α β : ℝ) : Prop := sorry

theorem resulting_shape_is_cone (α β γ : ℝ) (h : right_triangle α β γ) :
  ∃ (shape : Type), rotate_around_leg α β → shape = cone :=
by
  sorry

end resulting_shape_is_cone_l581_581549


namespace find_k_series_sum_l581_581669

theorem find_k_series_sum (k : ℝ) :
  (2 + ∑' n : ℕ, (2 + (n + 1) * k) / 2 ^ (n + 1)) = 6 -> k = 1 :=
by 
  sorry

end find_k_series_sum_l581_581669


namespace expression_independent_of_alpha_l581_581887

theorem expression_independent_of_alpha
  (α : Real) (n : ℤ) (h : α ≠ (n * (π / 2)) + (π / 12)) :
  (1 - 2 * Real.sin (α - (3 * π / 2))^2 + (Real.sqrt 3) * Real.cos (2 * α + (3 * π / 2))) /
  (Real.sin (π / 6 - 2 * α)) = -2 := 
sorry

end expression_independent_of_alpha_l581_581887


namespace brenda_age_correct_l581_581226

open Nat

noncomputable def brenda_age_proof : Prop :=
  ∃ (A B J : ℚ), 
  (A = 4 * B) ∧ 
  (J = B + 8) ∧ 
  (A = J) ∧ 
  (B = 8 / 3)

theorem brenda_age_correct : brenda_age_proof := 
  sorry

end brenda_age_correct_l581_581226


namespace scientific_notation_l581_581467

theorem scientific_notation (x y : ℝ) (h1 : x = 8.1) (h2 : y = -8) :
  0.000000081 = x * 10 ^ y :=
by {
  rw [h1, h2],
  sorry,
}

end scientific_notation_l581_581467


namespace sum_of_coefficients_l581_581618

theorem sum_of_coefficients (P : polynomial ℝ) :
  (20 * X^27 + 2 * X^2 + 1) * P = 2001 * X^2001 → 
  P.eval 1 = 87 :=
by
  sorry

end sum_of_coefficients_l581_581618


namespace prove_smallest_positive_angle_l581_581263

noncomputable def smallest_positive_angle (θ : ℝ) : Prop :=
  10 * sin θ * (cos θ) ^ 3 - 10 * (sin θ) ^ 3 * cos θ = sqrt 2

noncomputable def correct_answer (θ : ℝ) : Prop :=
  θ = (1 / 4) * asin ((2 * sqrt 2) / 5)

theorem prove_smallest_positive_angle θ : smallest_positive_angle θ → correct_answer θ := sorry

end prove_smallest_positive_angle_l581_581263


namespace twin_primes_iff_non_expressible_l581_581888

def is_twin_prime_pair (p q : ℕ) : Prop := 
  nat.prime p ∧ nat.prime q ∧ q = p + 2

def infinitude_twin_primes : Prop := 
  ∀ n : ℕ, ∃ p q : ℕ, p > n ∧ is_twin_prime_pair p q

def cannot_be_written_in_forms (k : ℕ) : Prop :=
  ∀ u v : ℕ, u > 0 → v > 0 →
    (k ≠ 6 * u * v + u + v ∧
     k ≠ 6 * u * v + u - v ∧
     k ≠ 6 * u * v - u + v ∧
     k ≠ 6 * u * v - u - v)

def infinitude_non_expressible_in_forms : Prop :=
  ∀ n : ℕ, ∃ k : ℕ, k > n ∧ cannot_be_written_in_forms k

theorem twin_primes_iff_non_expressible :
  infinitude_twin_primes ↔ infinitude_non_expressible_in_forms :=
sorry

end twin_primes_iff_non_expressible_l581_581888


namespace avg_amount_lost_per_loot_box_l581_581042

-- Define the conditions
def cost_per_loot_box : ℝ := 5
def avg_value_of_items : ℝ := 3.5
def total_amount_spent : ℝ := 40

-- Define the goal
theorem avg_amount_lost_per_loot_box : 
  (total_amount_spent / cost_per_loot_box) * (cost_per_loot_box - avg_value_of_items) / (total_amount_spent / cost_per_loot_box) = 1.5 := 
by 
  sorry

end avg_amount_lost_per_loot_box_l581_581042


namespace compute_star_difference_l581_581807

def star (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem compute_star_difference : (star 6 3) - (star 3 6) = 45 := by
  sorry

end compute_star_difference_l581_581807


namespace triangle_angle_relation_l581_581825
open EuclideanGeometry

variables {A B C P : Point}

-- Conditions
-- This condition assumes a planar geometry with triangular points and ratios.
-- BC = CA + 1/2 AB
-- BP : PA = 1 : 3

theorem triangle_angle_relation :
  (BC = CA + (1/2) * AB) ∧ ((BP / PA) = (1 / 3)) →
  (∠A C P = 2 * ∠C P A) :=
by
  sorry

end triangle_angle_relation_l581_581825


namespace fraction_dislike_interested_l581_581609

-- Defining the conditions
def total_students := 200
def interested_in_art_percent := 75
def interested_in_art := 0.75 * total_students
def not_interested_in_art := total_students - interested_in_art

def say_dislike_interested_percent := 30
def say_dislike_not_interested_percent := 80

def say_dislike_interested := 0.30 * interested_in_art
def say_dislike_not_interested := 0.80 * not_interested_in_art

def total_say_dislike := say_dislike_interested + say_dislike_not_interested

-- Proving the fraction of students who claim they dislike art but are actually interested
theorem fraction_dislike_interested :
  let fraction := say_dislike_interested / total_say_dislike in
  fraction = 9 / 17 :=
by
  sorry

end fraction_dislike_interested_l581_581609


namespace inner_cube_surface_area_l581_581588

-- The statement of the theorem
theorem inner_cube_surface_area (S : ℝ) (surface_area_outer_cube : S = 24) : 
  let side_length_outer_cube := real.sqrt (S / 6) in
  let diameter_of_sphere := side_length_outer_cube in
  let side_length_inner_cube := diameter_of_sphere / real.sqrt 3 in
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2) in
  surface_area_inner_cube = 8 :=
by
  sorry

end inner_cube_surface_area_l581_581588


namespace sum_odd_divisors_252_l581_581537

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : list ℕ := (list.range (n + 1)).filter (λ d, n % d = 0)

def odd_divisors_sum (n : ℕ) : ℕ :=
(list.filter is_odd (divisors n)).sum

theorem sum_odd_divisors_252 : odd_divisors_sum 252 = 104 :=
by
  -- Proof goes here
  sorry

end sum_odd_divisors_252_l581_581537


namespace day_50th_day_year_N_minus_1_l581_581035

-- Introduction for necessary assumptions
universe u

-- Defining the problem
def day_of_week (year : ℕ) (day : ℕ) : string := sorry

-- Given conditions
axiom condition1 : day_of_week N 250 = "Wednesday"
axiom condition2 : day_of_week (N + 1) 150 = "Wednesday"

-- Goal
theorem day_50th_day_year_N_minus_1 :
  day_of_week (N - 1) 50 = "Monday" := sorry

end day_50th_day_year_N_minus_1_l581_581035


namespace square_side_length_tangent_circle_l581_581195

theorem square_side_length_tangent_circle (r s : ℝ) :
  (∃ (O : ℝ × ℝ) (A : ℝ × ℝ) (AB : ℝ) (AD : ℝ),
    AB = AD ∧
    O = (r, r) ∧
    A = (0, 0) ∧
    dist O A = r * Real.sqrt 2 ∧
    s = dist (O.fst, 0) A ∧
    s = dist (0, O.snd) A ∧
    ∀ x y, (O = (x, y) → x = r ∧ y = r)) → s = 2 * r :=
by
  sorry

end square_side_length_tangent_circle_l581_581195


namespace part1_part2_l581_581855

-- Definitions based on the conditions
def a_i (i : ℕ) : ℕ := sorry -- Define ai's values based on the given conditions
def f (n : ℕ) : ℕ := sorry  -- Define f(n) as the number of n-digit wave numbers satisfying the given conditions

-- Prove the first part: f(10) = 3704
theorem part1 : f 10 = 3704 := sorry

-- Prove the second part: f(2008) % 13 = 10
theorem part2 : (f 2008) % 13 = 10 := sorry

end part1_part2_l581_581855


namespace pens_left_in_jar_l581_581997

theorem pens_left_in_jar : 
  ∀ (initial_blue_pens initial_black_pens initial_red_pens removed_blue_pens removed_black_pens : ℕ),
  initial_blue_pens = 9 →
  initial_black_pens = 21 →
  initial_red_pens = 6 →
  removed_blue_pens = 4 →
  removed_black_pens = 7 →
  (initial_blue_pens - removed_blue_pens) + (initial_black_pens - removed_black_pens) + initial_red_pens = 25 :=
begin
  intros initial_blue_pens initial_black_pens initial_red_pens removed_blue_pens removed_black_pens,
  intros h1 h2 h3 h4 h5,
  simp [h1, h2, h3, h4, h5],
  norm_num,
end

end pens_left_in_jar_l581_581997


namespace find_C_value_l581_581830

theorem find_C_value (A B C : ℕ) 
  (cond1 : A + B + C = 10) 
  (cond2 : B + A = 9)
  (cond3 : A + 1 = 3) :
  C = 1 :=
by
  sorry

end find_C_value_l581_581830


namespace worth_of_5_inch_cube_l581_581197

noncomputable def volume_of_cube (side_length : ℕ) : ℕ :=
  side_length ^ 3

def worth_of_cube (side_length : ℕ) (base_side_length : ℕ) (base_worth : ℝ) : ℝ :=
  let volume_ratio := volume_of_cube side_length / volume_of_cube base_side_length
  base_worth * volume_ratio

theorem worth_of_5_inch_cube :
  worth_of_cube 5 4 800 = 1563 := by
  sorry

end worth_of_5_inch_cube_l581_581197


namespace shaded_to_white_ratio_l581_581528

theorem shaded_to_white_ratio (largest_square_area : ℝ) 
  (vertices_condition : ∀ (n : ℕ), n > 1 → 
    (let side_length := (1:ℝ) / (2 ^ (n - 1)) 
     in side_length = side_length / 2 + side_length / 2)) :
  let shaded_area := 5 * (largest_square_area / 8) / (2 ^ 2) in
  let white_area := 3 * (largest_square_area / 8) / (2 ^ 2) in
  shaded_area / white_area = 5 / 3 :=
by
  sorry

end shaded_to_white_ratio_l581_581528


namespace complex_quadrant_l581_581056

theorem complex_quadrant : 
  let i : ℂ := complex.I in 
  let z : ℂ := i / (2 + i) in 
  0 < z.re ∧ 0 < z.im :=
by
  sorry

end complex_quadrant_l581_581056


namespace simplify_expression_l581_581096

-- Define the condition
def x := (Real.sqrt 3) / 2 + 1 / 2

-- Define the algebraic expression
def algebraic_expr (x : ℝ) : ℝ := 
  (((1 / x) + ((x + 1) / x)) / ((x + 2) / (x^2 + x)))

-- State the theorem
theorem simplify_expression : algebraic_expr x = (Real.sqrt 3 + 3) / 2 := 
by
  sorry

end simplify_expression_l581_581096


namespace inverse_sum_l581_581859

def f (x : ℝ) : ℝ := x * abs x

theorem inverse_sum : (∃ y : ℝ, f y = 9 ∧ y > 0) ∧ (∃ z : ℝ, f z = -81 ∧ z < 0) → 
  let y := classical.some (exists_inverse 9 false) in 
  let z := classical.some (exists_inverse (-81) true) in
  y + z = -6 :=
by
  sorry

noncomputable def exists_inverse (a : ℝ) (neg : bool) : ∃ x : ℝ, f x = a ∧ (if neg then x < 0 else x > 0) :=
by
  sorry

end inverse_sum_l581_581859


namespace pens_left_in_jar_l581_581998

theorem pens_left_in_jar : 
  ∀ (initial_blue_pens initial_black_pens initial_red_pens removed_blue_pens removed_black_pens : ℕ),
  initial_blue_pens = 9 →
  initial_black_pens = 21 →
  initial_red_pens = 6 →
  removed_blue_pens = 4 →
  removed_black_pens = 7 →
  (initial_blue_pens - removed_blue_pens) + (initial_black_pens - removed_black_pens) + initial_red_pens = 25 :=
begin
  intros initial_blue_pens initial_black_pens initial_red_pens removed_blue_pens removed_black_pens,
  intros h1 h2 h3 h4 h5,
  simp [h1, h2, h3, h4, h5],
  norm_num,
end

end pens_left_in_jar_l581_581998


namespace constant_term_in_expansion_l581_581277

theorem constant_term_in_expansion :
  let expr := (x + 3) * (1 - 2 / Real.sqrt x) ^ 5 in
  -- Extract the coefficients for specific powers of x
  ∃ c : ℝ, constant_term expr = 121 :=
sorry

end constant_term_in_expansion_l581_581277


namespace volume_of_extended_set_l581_581251

def volume_parallelepiped (a b c : ℝ) : ℝ := a * b * c
def volume_external_parallelepipeds (a b c r : ℝ) : ℝ := 2 * (a * b * r) + 2 * (a * c * r) + 2 * (b * c * r)
def volume_semi_spherical_caps (r : ℝ) : ℝ := 4 * (2 / 3 * Real.pi * r^3)

theorem volume_of_extended_set (m n p : ℕ) (a b c r : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 6) (h4 : r = 1)
  (hm : m = 324) (hn : n = 8) (hp : p = 3) :
  (volume_parallelepiped a b c + volume_external_parallelepipeds a b c r + volume_semi_spherical_caps r) = (324 + 8 * Real.pi) / 3 :=
by
  sorry

end volume_of_extended_set_l581_581251


namespace count_perfect_squares_and_cubes_l581_581742

theorem count_perfect_squares_and_cubes : 
  let num_squares := 31 in
  let num_cubes := 9 in
  let num_sixth_powers := 3 in
  num_squares + num_cubes - num_sixth_powers = 37 :=
by
  sorry

end count_perfect_squares_and_cubes_l581_581742


namespace sum_of_reciprocals_l581_581145

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 8 * x * y) : 
  (1 / x) + (1 / y) = 8 := 
by 
  sorry

end sum_of_reciprocals_l581_581145


namespace equilateral_triangle_perimeter_l581_581373

-- Given conditions:
variables (a b : ℝ)
def radius := 2
def offset := 1
def side (a b : ℝ) := a + b

-- Define the perimeter calculation function
def perimeter (a : ℝ) := 3 * a

-- Define the key properties of the side length under the given conditions
axiom side_property : side (2 * √3) 4 = 2√3 + 4

theorem equilateral_triangle_perimeter : perimeter (2√3 + 4) = 6√3 + 12 :=
by
-- Axiom utilizes side_property
apply side_property; sorry -- Placeholder to indicate where further proof steps would go.

#eval equilateral_triangle_perimeter

end equilateral_triangle_perimeter_l581_581373


namespace real_roots_exist_l581_581448

noncomputable def cubic_equation (x : ℝ) := x^3 - x^2 - 2*x + 1

theorem real_roots_exist : ∃ (a b : ℝ), 
  cubic_equation a = 0 ∧ cubic_equation b = 0 ∧ a - a * b = 1 := 
by
  sorry

end real_roots_exist_l581_581448


namespace product_and_difference_of_squares_l581_581146

theorem product_and_difference_of_squares :
  ∃ (a b : ℕ), (a + b = 40) ∧ (a - b = 8) ∧ (a * b = 384) ∧ ((a^2 - b^2) = 320) :=
begin
  sorry
end

end product_and_difference_of_squares_l581_581146


namespace sides_imply_angles_l581_581364

noncomputable theory

open Real

theorem sides_imply_angles {a b A B C : ℝ}
  (h_triangle: a > 0 ∧ b > 0 ∧ C > 0)
  (h_sides_opposite: ∠ABC = A ∧ ∠BAC = B)
  (h_law_of_sines: a / sin A = b / sin B) :
  (a > b ↔ sin A > sin B) :=
  sorry

end sides_imply_angles_l581_581364


namespace jennifer_total_discount_is_28_l581_581395

-- Define the conditions in the Lean context

def initial_whole_milk_cans : ℕ := 40 
def mark_whole_milk_cans : ℕ := 30 
def mark_skim_milk_cans : ℕ := 15 
def almond_milk_per_3_whole_milk : ℕ := 2 
def whole_milk_per_5_skim_milk : ℕ := 4 
def discount_per_10_whole_milk : ℕ := 4 
def discount_per_7_almond_milk : ℕ := 3 
def discount_per_3_almond_milk : ℕ := 1

def jennifer_additional_almond_milk := (mark_whole_milk_cans / 3) * almond_milk_per_3_whole_milk
def jennifer_additional_whole_milk := (mark_skim_milk_cans / 5) * whole_milk_per_5_skim_milk

def jennifer_whole_milk_cans := initial_whole_milk_cans + jennifer_additional_whole_milk
def jennifer_almond_milk_cans := jennifer_additional_almond_milk

def jennifer_whole_milk_discount := (jennifer_whole_milk_cans / 10) * discount_per_10_whole_milk
def jennifer_almond_milk_discount := 
  (jennifer_almond_milk_cans / 7) * discount_per_7_almond_milk + 
  ((jennifer_almond_milk_cans % 7) / 3) * discount_per_3_almond_milk

def total_jennifer_discount := jennifer_whole_milk_discount + jennifer_almond_milk_discount

-- Theorem stating the total discount 
theorem jennifer_total_discount_is_28 : total_jennifer_discount = 28 := by
  sorry

end jennifer_total_discount_is_28_l581_581395


namespace parallel_segment_length_l581_581010

/-- In \( \triangle ABC \), given side lengths AB = 500, BC = 550, and AC = 650,
there exists an interior point P such that each segment drawn parallel to the
sides of the triangle and passing through P splits the sides into segments proportional
to the overall sides of the triangle. Prove that the length \( d \) of each segment
parallel to the sides is 28.25 -/
theorem parallel_segment_length
  (A B C P : Type)
  (d AB BC AC : ℝ)
  (ha : AB = 500)
  (hb : BC = 550)
  (hc : AC = 650)
  (hp : AB * BC = AC * 550) -- This condition ensures proportionality of segments
  : d = 28.25 :=
sorry

end parallel_segment_length_l581_581010


namespace sum_T_mod_1000_l581_581673

open Nat

def T (a b : ℕ) : ℕ :=
  if h : a + b ≤ 6 then Nat.choose 6 a * Nat.choose 6 b * Nat.choose 6 (a + b) else 0

def sum_T : ℕ :=
  (Finset.range 7).sum (λ a => (Finset.range (7 - a)).sum (λ b => T a b))

theorem sum_T_mod_1000 : sum_T % 1000 = 564 := by
  sorry

end sum_T_mod_1000_l581_581673


namespace line_through_point_parallel_l581_581125

theorem line_through_point_parallel (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) (hA : A = (2, 3)) (hl : ∀ x y, l x y ↔ 2 * x - 4 * y + 7 = 0) :
  ∃ m, (∀ x y, (2 * x - 4 * y + m = 0) ↔ (x - 2 * y + 4 = 0)) ∧ (2 * (A.1) - 4 * (A.2) + m = 0) := 
sorry

end line_through_point_parallel_l581_581125


namespace count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581796

theorem count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes :
  let num_perfect_squares := 31
  let num_perfect_cubes := 9
  let num_perfect_sixth_powers := 2
  let num_either_squares_or_cubes := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers
  in num_either_squares_or_cubes = 38 :=
by
  sorry

end count_of_integers_less_than_1000_that_are_either_perfect_squares_or_cubes_l581_581796


namespace find_a1_l581_581144

noncomputable def sum_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
if h : q = 1 then n * a 0 else a 0 * (1 - q ^ n) / (1 - q)

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Definitions from conditions
def S_3_eq_a2_plus_10a1 (a_1 a_2 S_3 : ℝ) : Prop :=
S_3 = a_2 + 10 * a_1

def a_5_eq_9 (a_5 : ℝ) : Prop :=
a_5 = 9

-- Main theorem statement
theorem find_a1 (h1 : S_3_eq_a2_plus_10a1 (a 1) (a 2) (sum_of_geometric_sequence a q 3))
                (h2 : a_5_eq_9 (a 5))
                (h3 : q ≠ 0 ∧ q ≠ 1) :
    a 1 = 1 / 9 :=
sorry

end find_a1_l581_581144


namespace long_furred_brown_dogs_l581_581367

variable (L B N T LB : ℕ)

theorem long_furred_brown_dogs :
  L = 26 →
  B = 22 →
  N = 8 →
  T = 45 →
  LB = L + B - (T - N) →
  LB = 11 :=
by
  intros hL hB hN hT hLB
  rw [hL, hB, hN, hT] at hLB
  exact hLB

# To use this theorem in Lean 4, we need to set the values for L, B, N, and T to correspond to the conditions given in the original problem:
noncomputable def kennel_problem :=
  long_furred_brown_dogs 26 22 8 45 11

end long_furred_brown_dogs_l581_581367


namespace width_of_domain_g_l581_581355

-- Definitions from the condition
variable (h : ℝ → ℝ) -- Assume h is a function from ℝ to ℝ.
variable (domain_h : ∀ x, x ∈ Icc (-15 : ℝ) 15 → ∃ y, h y = x) -- h's domain is [-15, 15]

-- Definition of g from the condition
noncomputable def g (x : ℝ) := h (x / 3)

-- The theorem we need to prove
theorem width_of_domain_g : 
  (∀ x, x ∈ Icc (-15 : ℝ) 15 → ∃ y, h y = x) → 
  ∀ x, x ∈ Icc (-45 : ℝ) 45 ↔ g x = h (x / 3) → 
  (45 - (-45) = 90) :=
by
  intro h_dom g_def
  have domain_g : ∀ x, x ∈ Icc (-45 : ℝ) 45 ↔ x ∈ Icc (-45 : ℝ) 45 :=
    sorry
  let width := 45 - (-45) 
  show width = 90
  rfl

end width_of_domain_g_l581_581355


namespace divide_figure_into_8_identical_parts_l581_581256

/-- A cube or rectangle can be divided into 8 identical parts. -/
theorem divide_figure_into_8_identical_parts (figure : Type) 
  (is_cube : figure -> Prop) 
  (is_rectangle : figure -> Prop) 
  (cut : figure -> vector figure 8) 
  : (is_cube figure ∨ is_rectangle figure) -> 
    ∃ parts : vector figure 8, ∀ i j, parts[i] = parts[j] :=
by sorry

end divide_figure_into_8_identical_parts_l581_581256


namespace pyramid_top_block_minimum_l581_581623

theorem pyramid_top_block_minimum : 
  ∀ (a b c d : ℕ),
  a >= 1 ∧ a <= 4 →
  b >= 1 ∧ b <= 4 →
  c >= 1 ∧ c <= 4 →
  d >= 1 ∧ d <= 4 →
  let x := a + b in
  let y := b + c in
  let z := c + d in
  let w := x + y in
  let v := y + z in
  let top := w + v in
  top >= 20 :=
begin
  sorry
end

end pyramid_top_block_minimum_l581_581623


namespace lemonade_sales_l581_581895

theorem lemonade_sales (cups_last_week : ℕ) (percent_more : ℕ) 
  (h_last_week : cups_last_week = 20)
  (h_percent_more : percent_more = 30) : 
  let cups_this_week := cups_last_week + (percent_more * cups_last_week / 100)
  in cups_last_week + cups_this_week = 46 := 
by
  -- Definitions and calculation
  let cups_this_week := cups_last_week + (percent_more * cups_last_week / 100)
  have h_this_week : cups_this_week = 26, from calc
    cups_this_week = 20 + (30 * 20 / 100) : by rw [h_last_week, h_percent_more]
    ... = 20 + 6 : by norm_num
    ... = 26 : by norm_num,
  show cups_last_week + cups_this_week = 46, from calc
    20 + 26 = 46 : by norm_num

end lemonade_sales_l581_581895


namespace sandwiches_bought_l581_581174

theorem sandwiches_bought (sandwich_cost soda_cost total_cost_sodas total_cost : ℝ)
  (h1 : sandwich_cost = 2.45)
  (h2 : soda_cost = 0.87)
  (h3 : total_cost_sodas = 4 * soda_cost)
  (h4 : total_cost = 8.38) :
  ∃ (S : ℕ), sandwich_cost * S + total_cost_sodas = total_cost ∧ S = 2 :=
by
  use 2
  simp [h1, h2, h3, h4]
  sorry

end sandwiches_bought_l581_581174


namespace cube_tetrahedron_circumscribed_sphere_diameter_l581_581831

noncomputable def cube_dihedral_angle_diameter (A B C D A1 B1 C1 D1 E : ℝ) (AB : ℝ) : Prop :=
  let a := sqrt(1 + 9 + 9) in
  let tan_dihedral_angle := 3 * sqrt 2 / 2 in
  (tan_dihedral_angle = 3 * sqrt 2 / 2) →
  (E ∈ segment A1 B1) →
  (a = sqrt 19) →
  (AB = 3) →
  (∃ a, a = sqrt 19 ∧ (a / AB = sqrt 19 / 3))

theorem cube_tetrahedron_circumscribed_sphere_diameter 
  {A B C D A1 B1 C1 D1 E AB : ℝ} 
  (h1 : E ∈ segment A1 B1)
  (h2 : tan (dihedral_angle (plane E B D) (plane A B C D)) = 3 * sqrt 2 / 2)
  (h3: ∃ a, a = sqrt 19 ∧ (a / AB = sqrt 19 / 3)) : Prop :=
  ∃ a, a = sqrt 19 ∧ (a / AB = sqrt 19 / 3)

end cube_tetrahedron_circumscribed_sphere_diameter_l581_581831


namespace find_length_BG_l581_581840

-- Define the lengths of the sides of the triangle
variables (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
variable [noncomputable_instance] (AB AC BC : ℕ) [fact (AB = 12)] [fact (AC = 13)] [fact (BC = 15)]

-- Define the concept of centroid in a triangle and the separation property
noncomputable def centroid (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : Type

-- Define the problem condition, length of segments, and calculation of BG
theorem find_length_BG (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (AB AC BC : ℕ) (h₁ : AB = 12) (h₂ : AC = 13) (h₃ : BC = 15) :  
  ∃ G : Type, (centroid A B C = G) ∧ BG = (2 * real.sqrt 526) / 3 :=
sorry

end find_length_BG_l581_581840


namespace perfect_squares_and_cubes_count_lt_1000_l581_581787

theorem perfect_squares_and_cubes_count_lt_1000 : 
  let perfect_square := λ n : ℕ, ∃ k : ℕ, n = k^2
  let perfect_cube := λ n : ℕ, ∃ k : ℕ, n = k^3
  let sixth_power := λ n : ℕ, ∃ k : ℕ, n = k^6
  (count (λ n, n < 1000 ∧ (perfect_square n ∨ perfect_cube n)) (Finset.range 1000)) = 38 :=
by {
  sorry
}

end perfect_squares_and_cubes_count_lt_1000_l581_581787


namespace radius_of_base_of_cone_l581_581196

theorem radius_of_base_of_cone {θ r : ℝ} (hθ : θ = 150) (hr : r = 12) : 
  let arc_length := (θ / 360) * (2 * Real.pi * r),
      radius_of_base := (arc_length / (2 * Real.pi))
  in radius_of_base = 5 :=
by
  sorry

end radius_of_base_of_cone_l581_581196
