import Mathlib

namespace centroid_quadrilateral_area_ratio_l317_317303

theorem centroid_quadrilateral_area_ratio {A B C D : Point} (h_convex: ConvexQuadrilateral A B C D) :
  let M := intersection (line_through A C) (line_through B D) in
  let S_A := centroid (Triangle.make M A B) in
  let S_B := centroid (Triangle.make M B C) in
  let S_C := centroid (Triangle.make M C D) in
  let S_D := centroid (Triangle.make M D A) in
  area (Quadrilateral.make S_A S_B S_C S_D) = (2/9) * area (Quadrilateral.make A B C D) :=
sorry

end centroid_quadrilateral_area_ratio_l317_317303


namespace mean_big_integers_l317_317216

def isPermutation {n : ℕ} (a : List ℕ) : Prop :=
  a.perm (List.ofFn fun i => i + 1)

def isBig (a : List ℕ) (i : ℕ) : Prop :=
  ∀ j : ℕ, j < i → i < a.length → a.get!! i > a.get!! j

def meanNumberOfBigIntegers (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (1 / (k + 1 : ℝ))

theorem mean_big_integers (n : ℕ) (a : List ℕ) (h : isPermutation a) :
  (∑ i in Finset.range a.length, if isBig a i then 1 else 0 : ℝ) / ↑a.length = meanNumberOfBigIntegers n := by
  sorry

end mean_big_integers_l317_317216


namespace non_matching_outfits_l317_317468

theorem non_matching_outfits {shirts pants hats colors : ℕ} 
  (h_shirts : shirts = 8)
  (h_pants : pants = 8) 
  (h_hats : hats = 8) 
  (h_colors : colors = 8) : 
  let total_outfits := shirts * pants * hats in
  let mono_color_outfits := colors in
  let non_matching_outfits := total_outfits - mono_color_outfits in
  non_matching_outfits = 504 :=
by 
  have h_total : total_outfits = 512 := by rw [h_shirts, h_pants, h_hats]; norm_num
  have h_mono : mono_color_outfits = 8 := by rw [h_colors]
  have h_non_matching : non_matching_outfits = 504 := by rw [h_total, h_mono]; norm_num
  exact h_non_matching

end non_matching_outfits_l317_317468


namespace three_teams_no_match_l317_317171

theorem three_teams_no_match {teams : Finset ℕ} (h_teams_count : teams.card = 18) 
  (played_against_each_other : ℕ → Finset ℕ) 
  (h_played : ∀ t ∈ teams, (played_against_each_other t).card = 8) : 
  ∃ t1 t2 t3 ∈ teams, ∀ (x y : ℕ), x ≠ y → x ∈ ({t1, t2, t3} : Finset ℕ) → y ∈ ({t1, t2, t3} : Finset ℕ) → (x ∉ played_against_each_other y) :=
sorry

end three_teams_no_match_l317_317171


namespace compute_diff_squares_l317_317795

theorem compute_diff_squares (a b : ℤ) (ha : a = 153) (hb : b = 147) :
  a ^ 2 - b ^ 2 = 1800 :=
by
  rw [ha, hb]
  sorry

end compute_diff_squares_l317_317795


namespace arrangement_count_of_multiple_of_5_l317_317177

-- Define the digits and the condition that the number must be a five-digit multiple of 5
def digits := [1, 1, 2, 5, 0]
def is_multiple_of_5 (n : Nat) : Prop := n % 5 = 0

theorem arrangement_count_of_multiple_of_5 :
  ∃ (count : Nat), count = 21 ∧
  (∀ (num : List Nat), num.perm digits → is_multiple_of_5 (Nat.of_digits 10 num) → true) :=
begin
  use 21,
  split,
  { refl },
  { intros num h_perm h_multiple_of_5,
    sorry
  }
end

end arrangement_count_of_multiple_of_5_l317_317177


namespace not_exists_k_eq_one_l317_317425

theorem not_exists_k_eq_one (k : ℝ) : (∃ x y : ℝ, y = k * x + 2 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 :=
by sorry

end not_exists_k_eq_one_l317_317425


namespace sum_fractions_l317_317223

noncomputable def f (x : ℝ) : ℝ := 3 / (4^x + 3)

theorem sum_fractions : 
  (∑ k in Finset.range 2001, f (k.succ / 2002)) = 1000 := by
  sorry

end sum_fractions_l317_317223


namespace no_primes_between_fac_and_fac_plus_n_l317_317423

theorem no_primes_between_fac_and_fac_plus_n (n : ℕ) (h : n > 2) : 
  ∃ (k : ℕ), k = 0 ∧ 
  ∀ p, nat.prime p → (n! + 2 < p) → (p < n! + n + 1) → false :=
by
  sorry

end no_primes_between_fac_and_fac_plus_n_l317_317423


namespace find_third_side_of_triangle_l317_317301

theorem find_third_side_of_triangle (a b : ℝ) (A : ℝ) (h1 : a = 6) (h2 : b = 10) (h3 : A = 18) (h4 : ∃ C, 0 < C ∧ C < π / 2 ∧ A = 0.5 * a * b * Real.sin C) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 22 :=
by
  sorry

end find_third_side_of_triangle_l317_317301


namespace circle_chords_tangent_ratio_l317_317074

theorem circle_chords_tangent_ratio {
  (A B C D E M F G: Point)
  (h_circle : circle (A, B, C, D, E))
  (h_chords : (chord A B E) ∧ (chord C D E))
  (h_M_on_EB : M ∈ segment E B)
  (h_tangent_line : tangent_line_through (circle (A, B, C, D, E)) D E M intersects_at BC F)
  (h_tangent_line : tangent_line_through (circle (A, B, C, D, E)) D E M intersects_at AC G)
  (t : ℝ)
  (h_AM_AB : (AM / AB) = t) :
  (EG / EF) = (t / (1 - t)) := 
sorry

end circle_chords_tangent_ratio_l317_317074


namespace original_gain_percentage_is_5_l317_317734

variable (CP SP SP' CP' : ℝ)
variable (discount percentage profit reduction : ℝ)

noncomputable def cost_price := 400
noncomputable def new_cost_price := cost_price - (0.05 * cost_price)
noncomputable def new_selling_price := (1.10 * new_cost_price)
noncomputable def selling_price_after_reduction := new_selling_price + 2
noncomputable def gain := selling_price_after_reduction - cost_price
noncomputable def gain_percentage := (gain / cost_price) * 100

theorem original_gain_percentage_is_5 :
  cost_price = 400 →
  new_cost_price = 380 →
  new_selling_price = 418 →
  selling_price_after_reduction = 420 →
  gain_percentage = 5 := sorry

end original_gain_percentage_is_5_l317_317734


namespace simplify_expression_l317_317592

-- Define the constants.
def a : ℚ := 8
def b : ℚ := 27

-- Assuming cube root function is available and behaves as expected for rationals.
def cube_root (x : ℚ) : ℚ := x^(1/3 : ℚ)

-- Assume the necessary property of cube root of 27.
axiom cube_root_27_is_3 : cube_root 27 = 3

-- The main statement to prove.
theorem simplify_expression : cube_root (a + b) * cube_root (a + cube_root b) = cube_root 385 :=
by
  sorry

end simplify_expression_l317_317592


namespace both_complementary_angles_acute_is_certain_event_l317_317336

def complementary_angles (A B : ℝ) : Prop :=
  A + B = 90

def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

theorem both_complementary_angles_acute_is_certain_event (A B : ℝ) (h1 : complementary_angles A B) (h2 : acute_angle A) (h3 : acute_angle B) : (A < 90) ∧ (B < 90) :=
by
  sorry

end both_complementary_angles_acute_is_certain_event_l317_317336


namespace slope_of_line_through_midpoints_l317_317678

theorem slope_of_line_through_midpoints (A B C D : ℝ × ℝ) (hA : A = (1, 1)) (hB : B = (3, 4)) (hC : C = (4, 1)) (hD : D = (7, 4)) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  (N.2 - M.2) / (N.1 - M.1) = 0 := by
  sorry

end slope_of_line_through_midpoints_l317_317678


namespace above_theorem_l317_317105

noncomputable def reflect_point (p: (ℝ × ℝ)) (y_line: ℝ) : (ℝ × ℝ) :=
  (p.1, 2 * y_line - p.2)

def point_O : ℝ × ℝ := (0, 0)
def point_A : ℝ × ℝ := (12, 2)
def point_B : ℝ × ℝ := (0, 8)

def line_y : ℝ := 6

def point_P : ℝ × ℝ := reflect_point point_O line_y
def point_Q : ℝ × ℝ := reflect_point point_A line_y
def point_R : ℝ × ℝ := reflect_point point_B line_y

theorem above_theorem :
  let M := (4, 6)
  (area (point_B, M, point_R) = 8) :=
by
  sorry

end above_theorem_l317_317105


namespace jett_profit_l317_317210

def initial_cost : ℕ := 600
def vaccination_cost : ℕ := 500
def daily_food_cost : ℕ := 20
def number_of_days : ℕ := 40
def selling_price : ℕ := 2500

def total_expenses : ℕ := initial_cost + vaccination_cost + daily_food_cost * number_of_days
def profit : ℕ := selling_price - total_expenses

theorem jett_profit : profit = 600 :=
by
  -- Completed proof steps
  sorry

end jett_profit_l317_317210


namespace problem_intersection_empty_l317_317067

open Set

noncomputable def A (m : ℝ) : Set ℝ := {x | x^2 + 2*x + m = 0}
def B : Set ℝ := {x | x > 0}

theorem problem_intersection_empty (m : ℝ) : (A m ∩ B = ∅) ↔ (0 ≤ m) :=
sorry

end problem_intersection_empty_l317_317067


namespace prove_problem_statement_l317_317845

noncomputable def problem_statement (x y : ℝ) :=
  x - y = 1/2 ∧ xy = 4/3 → xy^2 - x^2 y = -2/3

theorem prove_problem_statement (x y : ℝ) :
  problem_statement x y :=
  by
    sorry

end prove_problem_statement_l317_317845


namespace valid_orders_fraction_l317_317395

-- Define the conditions of the problem
variables (n : ℕ) (m : ℕ) (price : ℕ)

-- Initial conditions
constant eight_people : n = 8
constant four_100Ft : m = 4
constant four_200Ft : n - m = 4
constant ticket_price : price = 100
constant register_empty : ticket_price * 0 = 0

-- Define the main proof problem
theorem valid_orders_fraction : 
  ∀ (valid_orders : ℕ) (total_orders : ℕ),
  valid_orders = 14 → total_orders = 70 → 
  valid_orders * 5 = total_orders :=
by 
  intros ;
  simp only [valid_orders, total_orders] ;
  sorry

end valid_orders_fraction_l317_317395


namespace directrix_of_parabola_l317_317408

theorem directrix_of_parabola (h : ∀ x : ℝ, y = -3 * x ^ 2 + 6 * x - 5) : ∃ y : ℝ, y = -25 / 12 :=
by
  sorry

end directrix_of_parabola_l317_317408


namespace no_perfect_squares_in_sequence_l317_317139

theorem no_perfect_squares_in_sequence : 
  ∀ N ∈ ({20142015, 201402015, 2014002015, 20140002015, 201400002015} : set ℕ), 
  ¬ (∃ k : ℕ, N = k^2) := 
by
  sorry

end no_perfect_squares_in_sequence_l317_317139


namespace cristina_gave_nicky_a_head_start_l317_317987

-- Define the constants and conditions from the problem
def cristina_speed : ℝ := 5
def nicky_speed : ℝ := 3
def time_nicky_runs : ℝ := 30

-- Define the time it takes for Cristina to cover the distance Nicky runs in 30 seconds
def time_cristina_takes : ℝ := (nicky_speed * time_nicky_runs) / cristina_speed

-- Define the problem statement in Lean 4
theorem cristina_gave_nicky_a_head_start : time_nicky_runs - time_cristina_takes = 12 :=
by
  sorry

end cristina_gave_nicky_a_head_start_l317_317987


namespace problem_l317_317949

theorem problem (
    α : ℝ
    (x : ℝ) (hx : x = 1 + (2 * (Real.sin α)^2) / ((Real.cos α)^2 - (Real.sin α)^2))
    (y : ℝ) (hy : y = (Real.sqrt 2 * Real.sin α * Real.cos α) / ((Real.cos α)^2 - (Real.sin α)^2))
    (θ : ℝ) (hθ : θ = π / 6)
) :
  (∀ x y, x^2 - 2 * y^2 = 1 → y = Real.sqrt 3 / 3 * x) ∧
  (∀ x y, (x = Real.sqrt 3 ∧ y = 1) ∨ (x = -Real.sqrt 3 ∧ y = -1)) ∧ 
  (∀ x y, (x^2 - 2 * y^2 = 1) ∧ (y = Real.sqrt 3 / 3 * x) → (sqrt (x ^ 2 + y ^ 2) = 2 ∧ atan (y / x) = π / 6) ∨ (sqrt (x ^ 2 + y ^ 2) = 2 ∧ atan (y / x) = 7 * π / 6)) :=
by
  sorry

end problem_l317_317949


namespace max_amount_spent_l317_317334

-- Definitions for banknotes and coins
variable (bills : Set ℚ) (coins : Set ℚ)

-- Conditions as hypotheses
hypothesis h1 : ∀ x ∈ bills, x > 1
hypothesis h2 : ∀ x ∈ coins, x < 1
hypothesis h3 : bills.size = 4
hypothesis h4 : coins.size = 4
hypothesis h5 : (bills.sum : ℚ) % 3 = 0
hypothesis h6 : (coins.sum * 100 : ℚ) % 7 = 0
hypothesis h7 : ∀ x ∈ bills, x ∈ {1, 5, 10, 20, 50, 100}
hypothesis h8 : ∀ x ∈ coins, x ∈ {0.5, 0.1, 0.05, 0.02, 0.01}

-- Statement
theorem max_amount_spent : (100 - (bills.sum + coins.sum) = 63.37) := 
by 
  sorry

end max_amount_spent_l317_317334


namespace select_graduates_condition_l317_317837

theorem select_graduates_condition :
  let graduates : Finset ℕ := (Finset.range 10) in
  let A : ℕ := 0 in  -- Graduate A
  let B : ℕ := 1 in  -- Graduate B
  let C : ℕ := 2 in  -- Graduate C
  let subsets := graduates.powerset.filter (λ s, s.card = 3) in
  let valid_sets := subsets.filter (λ s, (A ∈ s ∨ B ∈ s) ∧ C ∉ s) in
  valid_sets.card = 49 :=
by 
  sorry

end select_graduates_condition_l317_317837


namespace difference_between_avg_weight_and_joe_l317_317702

variables (n : ℕ) (weight_joe : ℝ) (initial_avg : ℝ) (new_avg : ℝ) (final_avg : ℝ)
variables (num_students_leave : ℕ) (diff : ℝ)

def initial_total_weight (n : ℕ) (initial_avg : ℝ) : ℝ :=
  n * initial_avg

def total_weight_with_joe (n : ℕ) (initial_avg weight_joe : ℝ) : ℝ :=
  initial_total_weight n initial_avg + weight_joe

def new_avg_weight (n : ℕ) (initial_avg weight_joe : ℝ) : ℝ :=
  total_weight_with_joe n initial_avg weight_joe / (n + 1)

def final_total_weight (n : ℕ) (initial_avg weight_joe : ℝ) (num_students_leave : ℕ) (final_avg : ℝ) : ℝ :=
  (n + 1 - num_students_leave) * final_avg

def weight_two_students_left (n : ℕ) (initial_avg new_avg weight_joe final_avg : ℝ) (num_students_leave : ℕ) : ℝ :=
  total_weight_with_joe n initial_avg weight_joe - final_total_weight n initial_avg weight_joe num_students_leave final_avg

def avg_weight_two_students_left (n : ℕ) (initial_avg new_avg weight_joe final_avg : ℝ) (num_students_leave : ℕ) : ℝ :=
  weight_two_students_left n initial_avg new_avg weight_joe final_avg num_students_leave / num_students_leave

theorem difference_between_avg_weight_and_joe 
  (h1 : weight_joe = 42)
  (h2 : initial_avg = 30)
  (h3 : new_avg = 31)
  (h4 : final_avg = 30)
  (h5 : num_students_leave = 2)
  (h6 : new_avg_weight n initial_avg weight_joe = new_avg) :
  abs (avg_weight_two_students_left n initial_avg new_avg weight_joe final_avg num_students_leave - weight_joe) = 6 :=
by
  sorry

end difference_between_avg_weight_and_joe_l317_317702


namespace integral_solution_l317_317387

noncomputable def integral_problem : Prop :=
  let L := {z : ℂ | complex.abs z = 1} in
  ∫ (z : ℂ) in L, (60 * complex.exp z / (z * (z + 3) * (z + 4) * (z + 5))) = 2 * real.pi * complex.I

theorem integral_solution : integral_problem :=
  sorry

end integral_solution_l317_317387


namespace correct_answer_l317_317765

-- Problem statement definitions
def fA (x : ℝ) : ℝ := x + 1
def fB (x : ℝ) : ℝ := -1 / x
def fC (x : ℝ) : ℝ := x^2
def fD (x : ℝ) : ℝ := x^3

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Definition of a monotonically increasing function
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Mathematical equivalence: The correct answer is the function fD which is both odd and increasing
theorem correct_answer :
  is_odd fD ∧ is_increasing fD := by
  -- Proof is intentionally left out
  sorry

end correct_answer_l317_317765


namespace isosceles_right_triangle_area_and_perimeter_l317_317276

theorem isosceles_right_triangle_area_and_perimeter (h : ℝ) (l : ℝ) (hyp : h = 6 * Real.sqrt 2) (leg_eq : h = Real.sqrt 2 * l) :
  let A := (1 / 2) * l * l,
      P := 2 * l + h in
  A = 18 ∧ P = 12 + 6 * Real.sqrt 2 := 
by
  sorry

end isosceles_right_triangle_area_and_perimeter_l317_317276


namespace polynomial_roots_l317_317403

noncomputable def P (x : ℂ) : ℂ := x^4 + 4 * x^3 - 2 * x^2 - 20 * x + 24

theorem polynomial_roots :
  ∀ x : ℂ, P x = 0 ↔ x = 2 ∨ x = -2 ∨ x = 2 * complex.I ∨ x = -2 * complex.I :=
by 
  sorry

end polynomial_roots_l317_317403


namespace inequality_proof_l317_317564

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (c + a)) * (1 + 4 * c / (a + b)) > 25 := by
  sorry

end inequality_proof_l317_317564


namespace chess_tournament_games_l317_317934

theorem chess_tournament_games (n : ℕ) (k : ℕ) 
  (h_n : n = 24) (h_k : k = 2) : 
  (nat.choose n k = 552) :=
by
  rw [h_n, h_k]
  -- Proof would typically go here 
  sorry

end chess_tournament_games_l317_317934


namespace fn_lt_gn_for_all_n_l317_317474

def f (n : ℕ) : ℝ :=
1 + ∑ i in finset.range(n), 1 / real.sqrt(i.succ)

def g (n : ℕ) : ℝ :=
2 * real.sqrt(n)

theorem fn_lt_gn_for_all_n (n : ℕ) (hn : 2 < n) : f(n) < g(n) :=
sorry

end fn_lt_gn_for_all_n_l317_317474


namespace part_1_part_2_l317_317453

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / x
noncomputable def g (x m : ℝ) : ℝ := m * Real.cos x - x

theorem part_1 :
  (∀ x ∈ Ioo (-Real.pi) 0, deriv f x > 0) ∧ (∀ x ∈ Ioo 0 Real.pi, deriv f x < 0) :=
sorry

theorem part_2 (m : ℝ) :
  (∃! x ∈ Ioo 0 (3 * Real.pi / 2), m * f x = g x m) → m > (9 * Real.pi ^ 2) / 4 :=
sorry

end part_1_part_2_l317_317453


namespace smallest_perfect_square_divisible_by_5_and_6_l317_317684

-- 1. Define the gcd and lcm functionality
def lcm (a b : ℕ) : ℕ :=
  (a * b) / Nat.gcd a b

-- 2. Define the condition that a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- 3. State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 : ∃ n : ℕ, is_perfect_square n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (is_perfect_square m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m :=
  sorry

end smallest_perfect_square_divisible_by_5_and_6_l317_317684


namespace find_valid_n_l317_317046

theorem find_valid_n :
  ∀ n : ℕ, 
    (0 < n) → 
    (∃ (x : Fin n → ℝ), 
        (∀ i : Fin n, -1 < x i ∧ x i < 1) ∧ 
        (Finset.univ.sum x = 0) ∧ 
        (Finset.univ.sum (λ i, Real.sqrt (1 - (x i) ^ 2)) = 1)
      ) ↔ (n = 1 ∨ ∃ k : ℕ, n = 2 * k) :=
by
  sorry

end find_valid_n_l317_317046


namespace triangle_area_l317_317048

theorem triangle_area (A B C : ℝ) (R : ℝ) 
  (hA : A = π / 7) 
  (hB : B = 2 * π / 7) 
  (hC : C = 4 * π / 7) 
  (hR : R = 1) :
  2 * R^2 * (Real.sin A) * (Real.sin B) * (Real.sin C) = sqrt 7 / 4 := 
by
  rw [hA, hB, hC, hR]
  sorry

end triangle_area_l317_317048


namespace avg_annual_growth_rate_2010_2012_annual_growth_rate_2012_2013_l317_317381

noncomputable def averageAnnualGrowthRate2010_2012 (C2010 C2012 : ℝ) (x : ℝ) := C2012 = C2010 * (1 + x)^2

theorem avg_annual_growth_rate_2010_2012 :
  averageAnnualGrowthRate2010_2012 1 1.44 0.2 :=
by
  unfold averageAnnualGrowthRate2010_2012
  norm_num
  sorry

noncomputable def growthRateConstraint2012_2013 (C2012 C2013_max : ℝ) (y : ℝ) := C2012 * (1 + y) * 0.9 ≤ C2013_max

theorem annual_growth_rate_2012_2013 :
  growthRateConstraint2012_2013 1.44 1.5552 0.18 :=
by
  unfold growthRateConstraint2012_2013
  norm_num
  sorry

end avg_annual_growth_rate_2010_2012_annual_growth_rate_2012_2013_l317_317381


namespace square_difference_l317_317790

theorem square_difference :
  153^2 - 147^2 = 1800 :=
by
  sorry

end square_difference_l317_317790


namespace final_price_correct_l317_317753

def list_price : ℝ := 150
def first_discount_percentage : ℝ := 19.954259576901087
def second_discount_percentage : ℝ := 12.55

def first_discount_amount : ℝ := list_price * (first_discount_percentage / 100)
def price_after_first_discount : ℝ := list_price - first_discount_amount
def second_discount_amount : ℝ := price_after_first_discount * (second_discount_percentage / 100)
def final_price : ℝ := price_after_first_discount - second_discount_amount

theorem final_price_correct : final_price = 105 := by
  sorry

end final_price_correct_l317_317753


namespace final_price_correct_l317_317752

noncomputable def final_price_after_discounts (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let first_discount_amount := p * d1 / 100
  let price_after_first_discount := p - first_discount_amount
  let second_discount_amount := price_after_first_discount * d2 / 100
  price_after_first_discount - second_discount_amount

theorem final_price_correct :
  final_price_after_discounts 150 19.954259576901087 12.55 ≈ 105.00000063464838 :=
by 
  -- Approximate calculations to assert the correctness
  sorry

end final_price_correct_l317_317752


namespace time_difference_l317_317762

noncomputable def hour_angle (n : ℝ) : ℝ :=
  150 + (n / 2)

noncomputable def minute_angle (n : ℝ) : ℝ :=
  6 * n

theorem time_difference (n1 n2 : ℝ)
  (h1 : |(hour_angle n1) - (minute_angle n1)| = 120)
  (h2 : |(hour_angle n2) - (minute_angle n2)| = 120) :
  n2 - n1 = 43.64 := 
sorry

end time_difference_l317_317762


namespace equilateral_triangle_EFG_l317_317193

-- Define the setup of the geometrical points and their relations.
variables {O A B C D E F G : Type*}
variables {dist : O → O → ℝ}
variables [decidable_eq O]

-- Given conditions
def is_midpoint (M X Y : O) := dist M X = dist M Y
def angle_measure (X Y Z : O) := dist X Y = dist Y Z
def is_convex (S : set O) := ∀ {x y z w : O}, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ w ∈ S → true

-- Conditions from the problem
condition1 : dist A O = dist B O
condition2 : dist C O = dist D O
condition3 : angle_measure A O B = 120
condition4 : angle_measure C O D = 120
condition5 : is_midpoint E A B
condition6 : is_midpoint F B C
condition7 : is_midpoint G C D

-- The goal is to prove that triangle EFG is equilateral
theorem equilateral_triangle_EFG : dist E F = dist F G ∧ dist F G = dist G E :=
sorry

end equilateral_triangle_EFG_l317_317193


namespace value_of_k_for_binomial_square_l317_317311

theorem value_of_k_for_binomial_square (k : ℝ) : (∃ (b : ℝ), x^2 - 18 * x + k = (x + b)^2) → k = 81 :=
by
  intro h
  cases h with b hb
  -- We will use these to directly infer things without needing the proof here
  sorry

end value_of_k_for_binomial_square_l317_317311


namespace number_of_prime_factors_of_N_l317_317913

theorem number_of_prime_factors_of_N (N : ℕ) (h : log 2 (log 3 (log 5 (log 11 N))) = 7) : 
  nat.factors N = [11] :=
sorry

end number_of_prime_factors_of_N_l317_317913


namespace checkerboard_pattern_exists_l317_317042

/-- 
A theorem stating that on a 100x100 board where every cell is painted black or white,
all boundary cells are black, and there are no monochromatic 2x2 squares, 
there exists a 2x2 square with cells colored in a checkerboard pattern.
-/
theorem checkerboard_pattern_exists 
  (board : Fin 100 → Fin 100 → bool)
  (boundary_black : ∀ (i : Fin 100), board 0 i = tt ∧ board 99 i = tt ∧ board i 0 = tt ∧ board i 99 = tt)
  (no_monochromatic_2x2 : ∀ (i j : Fin 99), ¬ ((board i j = board (i + 1) j) ∧ (board i j = board i (j + 1)) ∧ (board i j = board (i + 1) (j + 1)))) :
  ∃ (i j : Fin 99), (board i j ≠ board (i + 1) j) ∧ (board i (j + 1) ≠ board (i + 1) (j + 1)) :=
by
  sorry

end checkerboard_pattern_exists_l317_317042


namespace triangle_area_correct_l317_317305

-- Definitions based on the conditions
def vertex1 : ℝ × ℝ := (0, 0)
def vertex2 : ℝ × ℝ := (0, 6)
def vertex3 : ℝ × ℝ := (8, 10)

-- Compute the base and height of the triangle
def base (a b : ℝ × ℝ) : ℝ := real.dist (a, b).snd
def height (a : ℝ × ℝ) : ℝ := real.dist ((a.fst, 0), (x.fst, y.snd)).fst

-- Calculate the area of the triangle
def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  (1 / 2) * base b a * height c

-- Statement to be proved
theorem triangle_area_correct : 
  triangle_area vertex1 vertex2 vertex3 = 24 :=
sorry

end triangle_area_correct_l317_317305


namespace ratio_proof_l317_317247

theorem ratio_proof (a b x : ℝ) (h : a > b) (h_b_pos : b > 0)
  (h_x : x = 0.5 * Real.sqrt (a / b) + 0.5 * Real.sqrt (b / a)) :
  2 * b * Real.sqrt (x^2 - 1) / (x - Real.sqrt (x^2 - 1)) = a - b := 
sorry

end ratio_proof_l317_317247


namespace team_selection_l317_317511

theorem team_selection : 
  ∀ (male female : ℕ), male = 3 → female = 3 → (number_of_ways male female 3) = 18 :=
by sorry

-- Required definition of number_of_ways
noncomputable def number_of_ways (male female team_size : ℕ) : ℕ :=
  (Nat.choose male 2) * (Nat.choose female 1) + 
  (Nat.choose male 1) * (Nat.choose female 2)

end team_selection_l317_317511


namespace number_of_dogs_l317_317808

-- Define the number of legs humans have
def human_legs : ℕ := 2

-- Define the total number of legs/paws in the pool
def total_legs_paws : ℕ := 24

-- Define the number of paws per dog
def paws_per_dog : ℕ := 4

-- Prove that the number of dogs is 5
theorem number_of_dogs : ∃ (dogs : ℕ), (2 * human_legs) + (dogs * paws_per_dog) = total_legs_paws ∧ dogs = 5 :=
by
  use 5
  split
  sorry

end number_of_dogs_l317_317808


namespace simplify_expression_l317_317603

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem simplify_expression :
  cube_root (8 + 27) * cube_root (8 + cube_root 27) = cube_root 385 :=
by
  sorry

end simplify_expression_l317_317603


namespace sample_stat_properties_l317_317866

/-- Given a set of sample data, and another set obtained by adding a non-zero constant to each element of the original set,
prove some properties about their statistical measures. -/
theorem sample_stat_properties (n : ℕ) (c : ℝ) (h₀ : c ≠ 0) 
  (x : ℕ → ℝ) :
  let y := λ i, x i + c 
  in (∑ i in finset.range n, x i) / n ≠ (∑ i in finset.range n, y i) / n ∧ 
     (∃ (median_x : ℝ) (median_y : ℝ), median_y = median_x + c ∧ median_y ≠ median_x) ∧
     (stddev (finset.range n).map x = stddev (finset.range n).map y) ∧
     (range (finset.range n).map x = range (finset.range n).map y) := 
by
  sorry

-- Helper functions for median, stddev, and range could be defined if missing from Mathlib.

end sample_stat_properties_l317_317866


namespace general_term_and_sum_of_b_n_l317_317018

structure ArithmeticSequence (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  (sum_first_n_terms : ∀ n, S n = ∑ i in Finset.range n, a (i + 1))
  (a2_plus_a4 : a 2 + a 4 = 6)
  (S4_equals_10 : S 4 = 10)

def b_n (a : ℕ → ℕ) (n : ℕ) : ℕ := a n * 2^n

def T_n (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in Finset.range n, b_n a (i + 1)

theorem general_term_and_sum_of_b_n (a : ℕ → ℕ) (S : ℕ → ℕ) (seq : ArithmeticSequence a S) :
  (∀ n, a n = n) ∧ (∀ n, T_n a n = (n - 1) * 2^(n+1) + 2) :=
by
  sorry

end general_term_and_sum_of_b_n_l317_317018


namespace tetrahedron_plane_intersects_interior_l317_317759

theorem tetrahedron_plane_intersects_interior (V : Finset ℝ) (hV : V.card = 4) :
  let planes := {s : Finset ℝ | s.card = 3 ∧ s ⊆ V} in
  ∀ p ∈ planes, (∃ x ∈ tetrahedron, x ∈ p) → probability planes = 1 :=
by sorry

end tetrahedron_plane_intersects_interior_l317_317759


namespace draw_probability_l317_317345

variable (P_lose_a win_a : ℝ)
variable (not_lose_a : ℝ := 0.8)
variable (win_prob_a : ℝ := 0.6)

-- Given conditions
def A_not_losing : Prop := not_lose_a = win_prob_a + win_a

-- Main theorem to prove
theorem draw_probability : P_lose_a = 0.2 :=
by
  sorry

end draw_probability_l317_317345


namespace area_bulldozer_of_triangle_l317_317667

variables {A B C P Q X Y Z : Point}
variable {PA PB PC QA QB QC : ℝ}
variable {BC CA AB : Line}
variable {S : Set Point}

noncomputable def condition1 : Prop := 
  PA * length BC = PB * length CA ∧ PA * length BC = PC * length AB

noncomputable def condition2 : Prop := 
  QA * length BC = QB * length CA ∧ QA * length BC = QC * length AB

noncomputable def condition3 : Prop :=
  length (↑X - ↑Y) = 1

noncomputable def condition4 : Set Point := 
  {Z | length (↑X - ↑Z) = 2 * length (↑Y - ↑Z)}

theorem area_bulldozer_of_triangle :
  condition1 ∧ condition2 ∧ condition3 →
  let S := {Z : Point | Z ∈ plane ∧ Z = (p₁ + t * p₂) ∧ t ≥ 0}
  in floor (100 * area S) = 129 :=
by
  sorry

end area_bulldozer_of_triangle_l317_317667


namespace sequence_15th_term_is_5_l317_317616

def sequence := List.recOn Nat (fun n => n)

-- condition: The sequence is constructed by repeating each natural number n times.
def repeat_n_times (n : ℕ) : List ℕ := List.replicate n n

-- condition: Append all these repeated sequences to form the global sequence.
def global_sequence (m : ℕ) : List ℕ :=
  List.join (List.map repeat_n_times (List.range m))

-- problem: Prove that the 15th term is 5.
theorem sequence_15th_term_is_5 : (global_sequence 7).get? 14 = some 5 :=
by
  sorry

end sequence_15th_term_is_5_l317_317616


namespace circumcenter_of_triangle_ABC_l317_317545

open EuclideanGeometry

noncomputable def point_A := sorry
noncomputable def point_X := sorry
noncomputable def point_Y := sorry
noncomputable def point_B := sorry
noncomputable def point_C := sorry
noncomputable def point_P := sorry

axiom point_of_intersection (c1 c2 : Circle) : ∃ A, A ∈ (Circle.points c1) ∧ A ∈ (Circle.points c2)
axiom tangent_condition (c1 c2 : Circle) (A : Point) : ∃ B, ∃ C, is_tangent c1 A B ∧ is_tangent c2 A C
axiom parallelogram_condition (P X A Y : Point) : parallelogram P X A Y

theorem circumcenter_of_triangle_ABC :
  is_circumcenter P (triangle B A C) :=
begin
  sorry
end

end circumcenter_of_triangle_ABC_l317_317545


namespace rearrange_numbers_diff_3_or_5_l317_317200

noncomputable def is_valid_permutation (n : ℕ) (σ : list ℕ) : Prop :=
  (σ.nodup ∧ ∀ i < σ.length - 1, |σ.nth_le i sorry - σ.nth_le (i + 1) sorry| = 3 ∨ |σ.nth_le i sorry - σ.nth_le (i + 1) sorry| = 5)

theorem rearrange_numbers_diff_3_or_5 (n : ℕ) :
  (n = 25 ∨ n = 1000) → ∃ σ : list ℕ, (σ = (list.range n).map (+1)) ∧ is_valid_permutation n σ :=
by
  sorry

end rearrange_numbers_diff_3_or_5_l317_317200


namespace find_integer_n_l317_317052

theorem find_integer_n :
  ∃ (n : ℤ), -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.cos (675 * Real.pi / 180) ∧ n = 45 :=
sorry

end find_integer_n_l317_317052


namespace percentage_of_boys_passed_examination_l317_317175

theorem percentage_of_boys_passed_examination (total_candidates : ℕ)
  (girls : ℕ) (boys : ℕ) (girls_passed_percentage : ℝ)
  (failed_percentage : ℝ) :
  total_candidates = 2000 →
  girls = 900 →
  boys = total_candidates - girls →
  girls_passed_percentage = 0.32 →
  failed_percentage = 0.702 →
  let total_passed_percentage := 1 - failed_percentage in
  let girls_passed := girls_passed_percentage * girls in
  let total_passed := total_passed_percentage * total_candidates in
  let boys_passed := total_passed - girls_passed in
  let boys_passed_percentage := (boys_passed / boys) * 100 in
  boys_passed_percentage = 28 :=
by
  intros
  sorry

end percentage_of_boys_passed_examination_l317_317175


namespace x_eq_1_sufficient_not_necessary_for_x_sq_eq_1_l317_317707

theorem x_eq_1_sufficient_not_necessary_for_x_sq_eq_1 (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ ((x^2 = 1) → (x = 1 ∨ x = -1)) :=
by 
  sorry

end x_eq_1_sufficient_not_necessary_for_x_sq_eq_1_l317_317707


namespace find_X_l317_317161

variable (X : ℝ)  -- Threshold income level for the lower tax rate
variable (I : ℝ)  -- Income of the citizen
variable (T : ℝ)  -- Total tax amount

-- Conditions
def income : Prop := I = 50000
def tax_amount : Prop := T = 8000
def tax_formula : Prop := T = 0.15 * X + 0.20 * (I - X)

theorem find_X (h1 : income I) (h2 : tax_amount T) (h3 : tax_formula T I X) : X = 40000 :=
by
  sorry

end find_X_l317_317161


namespace inequality_solution_l317_317262

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + x^3

theorem inequality_solution (x : ℝ) (hx : 0 < x ∧ x ≤ 1) : 
  Real.arcsin x ^ 2 + Real.arcsin x + x ^ 6 + x ^ 3 > 0 :=
by {
  have h_g : StrictMono g := sorry, -- Proof that g is strictly increasing on [-1, 1]
  have h_odd : ∀ x, g (-x) = -g x := sorry, -- Proof that g is an odd function
  have h_g_pos : g x^2 > g (-x) := sorry, -- Derived condition g(x^2) > g(-x)
  sorry
}

end inequality_solution_l317_317262


namespace regions_divided_by_n_lines_l317_317036

theorem regions_divided_by_n_lines (n : ℕ) : 
  ∀ (u_n : ℕ), (u_n 0 = 1) ∧ (u_n 1 = 2) ∧ (∀ k : ℕ, u_n (k + 1) = u_n k + (k + 1)) → 
  u_n n = (n * (n + 1)) / 2 + 1 :=
by
  sorry

end regions_divided_by_n_lines_l317_317036


namespace not_possible_to_network_1987_computers_l317_317538

theorem not_possible_to_network_1987_computers :
  ¬(∃ (G : SimpleGraph (Fin 1987)), ∀ v : Fin 1987, G.degree v = 5) :=
by
  sorry

end not_possible_to_network_1987_computers_l317_317538


namespace hyperbola_equation_line_equation_through_midpoint_l317_317078

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (asymptote : ∀ x, y = 2*x) (focus_distance_to_asymptote : ℝ) (h3 : focus_distance_to_asymptote = 1) :
  let C := { p : ℝ × ℝ | (p.snd ^ 2) / (a ^ 2) - (p.fst ^ 2) / (b ^ 2) = 1 } in
  C = { p : ℝ × ℝ | (p.snd ^ 2) / 4 - p.fst ^ 2 = 1 } :=
sorry

theorem line_equation_through_midpoint (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (asymptote_y_eq_2x : ∀ x, y = 2 * x) (focus_distance_to_asymptote : ℝ) (h3 : focus_distance_to_asymptote = 1)
  (M : ℝ × ℝ) (h4 : M = (1, 4)) :
  let C := { p : ℝ × ℝ | (p.snd ^ 2) / 4 - p.fst ^ 2 = 1 } in
  ∃ l : ℝ × ℝ → Prop, l = { p : ℝ × ℝ | p.snd - p.fst + 3 = 0 } ∧ ∀ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2) →
    ∃ k : ℝ, l (k * A + (1 - k) * B) :=
sorry

end hyperbola_equation_line_equation_through_midpoint_l317_317078


namespace f_2007_l317_317485

noncomputable def f : ℕ → ℝ :=
  sorry

axiom functional_eq (x y : ℕ) : f (x + y) = f x * f y

axiom f_one : f 1 = 2

theorem f_2007 : f 2007 = 2 ^ 2007 :=
by
  sorry

end f_2007_l317_317485


namespace smallest_a_plus_b_l317_317443

theorem smallest_a_plus_b (a b : ℕ) (h1: 0 < a) (h2: 0 < b) (h3 : 2^10 * 7^3 = a^b) : a + b = 31 :=
sorry

end smallest_a_plus_b_l317_317443


namespace salary_increase_l317_317588

variable (S : ℝ) -- Robert's original salary
variable (P : ℝ) -- Percentage increase after decrease in decimal form

theorem salary_increase (h1 : 0.5 * S * (1 + P) = 0.75 * S) : P = 0.5 := 
by 
  sorry

end salary_increase_l317_317588


namespace zero_of_f_when_a_zero_range_of_f_when_a_one_solution_set_of_f_gt_zero_l317_317117

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + 2

theorem zero_of_f_when_a_zero : f 0 2 = 0 :=
by sorry

theorem range_of_f_when_a_one (m : ℝ) : 
  (∃ x ∈ Icc 1 3, f 1 x = m) ↔ m ∈ Icc (-1 / 4) 2 :=
by sorry

theorem solution_set_of_f_gt_zero (a : ℝ) (ha : a > 0) (x : ℝ) : 
  f a x > 0 ↔ 
    (a = 1 / 2 ∧ x ≠ 2) ∨
    (0 < a ∧ a < 1 / 2 ∧ (x < 2 ∨ x > 1 / a)) ∨
    (a > 1 / 2 ∧ (x < 1 / a ∨ x > 2)) :=
by sorry

end zero_of_f_when_a_zero_range_of_f_when_a_one_solution_set_of_f_gt_zero_l317_317117


namespace coordinates_of_Q_l317_317948

variables (x y : ℝ)

def point_P := (real.sqrt 3, 1)

def orthogonality_condition := (real.sqrt 3 * x + y = 0)

def magnitude_condition := (x^2 + y^2 = 4)

theorem coordinates_of_Q:
  (∃ x y : ℝ, (real.sqrt 3 * x + y = 0) ∧ (x^2 + y^2 = 4) ∧ (x = -1) ∧ (y = real.sqrt 3)) :=
begin
  sorry
end

end coordinates_of_Q_l317_317948


namespace algebraic_expression_value_l317_317318

theorem algebraic_expression_value (p q : ℤ) 
  (h : 8 * p + 2 * q = -2023) : 
  (p * (-2) ^ 3 + q * (-2) + 1 = 2024) :=
by
  sorry

end algebraic_expression_value_l317_317318


namespace picked_balls_correct_l317_317655

-- Conditions
def initial_balls := 6
def final_balls := 24

-- The task is to find the number of picked balls
def picked_balls : Nat := final_balls - initial_balls

-- The proof goal
theorem picked_balls_correct : picked_balls = 18 :=
by
  -- We declare, but the proof is not required
  sorry

end picked_balls_correct_l317_317655


namespace range_of_m_l317_317106

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 / x) + (3 / y) = 1)
  (h4 : 3 * x + 2 * y > m^2 + 2 * m) :
  -6 < m ∧ m < 4 :=
sorry

end range_of_m_l317_317106


namespace correct_algebraic_notation_l317_317302

variable (A B C D : String)

def lowerCase (str : String) : String :=
  String.map str (fun c => Char.imp (c >= 'A' && c <= 'Z') (Char.toLower c) c)

theorem correct_algebraic_notation :
  (A = "ax ÷ 4" ∧ B = "-1a" ∧ C = "-3xy" ∧ D = "1⅔m") →
  lowerCase C = "-3xy" := sorry

end correct_algebraic_notation_l317_317302


namespace mike_runs_more_l317_317504

theorem mike_runs_more (street_width : ℝ) (block_side : ℝ) : 
    street_width = 30 → block_side = 500 → 
    let matt_perimeter := 4 * block_side in
    let mike_perimeter := 4 * (block_side + 2 * street_width) in
    mike_perimeter - matt_perimeter = 240 :=
by
  intros hw hs
  rw [hw, hs]
  let matt_perimeter := 4 * 500
  let mike_perimeter := 4 * (500 + 2 * 30)
  have hmatt : matt_perimeter = 2000 := by norm_num
  have hmike : mike_perimeter = 2240 := by norm_num
  rw [hmatt, hmike]
  norm_num
  sorry

end mike_runs_more_l317_317504


namespace lattice_point_count_l317_317733

def is_lattice_point (p : ℤ × ℤ) : Prop :=
  ∃ x y, p = (x, y)

def curve1 (x : ℝ) : ℝ := abs x
def curve2 (x : ℝ) : ℝ := -x^2 + 5

-- Region bounded by y = |x| and y = -x^2 + 5
def in_region (p : ℝ × ℝ) : Prop :=
  ∃ x y, p = (x, y) ∧ (curve1 x ≤ y ∧ y ≤ curve2 x)

theorem lattice_point_count : 
  {p : ℤ × ℤ | is_lattice_point p ∧ in_region (p.1, p.2)}.toFinset.card = 14 :=
sorry

end lattice_point_count_l317_317733


namespace triangle_equilateral_l317_317668

variable {a b c : ℝ}

theorem triangle_equilateral (h : a^2 + 2 * b^2 = 2 * b * (a + c) - c^2) : a = b ∧ b = c := by
  sorry

end triangle_equilateral_l317_317668


namespace back_wheel_revolutions_calculation_l317_317575

def front_wheel_radius : ℝ := 3
def back_wheel_radius : ℝ := 0.5
def gear_ratio : ℝ := 2
def front_wheel_revolutions : ℕ := 50

noncomputable def back_wheel_revolutions (front_wheel_radius back_wheel_radius gear_ratio : ℝ) (front_wheel_revolutions : ℕ) : ℝ :=
  let front_circumference := 2 * Real.pi * front_wheel_radius
  let distance_traveled := front_circumference * front_wheel_revolutions
  let back_circumference := 2 * Real.pi * back_wheel_radius
  distance_traveled / back_circumference * gear_ratio

theorem back_wheel_revolutions_calculation :
  back_wheel_revolutions front_wheel_radius back_wheel_radius gear_ratio front_wheel_revolutions = 600 :=
sorry

end back_wheel_revolutions_calculation_l317_317575


namespace range_of_m_for_false_p_and_q_l317_317567

theorem range_of_m_for_false_p_and_q (m : ℝ) :
  (¬ (∀ x y : ℝ, (x^2 / (1 - m) + y^2 / (m + 2) = 1) ∧ ∀ x y : ℝ, (x^2 / (2 * m) + y^2 / (2 - m) = 1))) →
  (m ≤ 1 ∨ m ≥ 2) :=
sorry

end range_of_m_for_false_p_and_q_l317_317567


namespace assignment_of_teachers_l317_317426

noncomputable def number_of_ways_to_assign_teachers : ℕ :=
  36

theorem assignment_of_teachers :
  let teachers := {1, 2, 3, 4}
      classes := {A, B, C}
      partitions := {p : finset classes → finset teachers // 
                        ∀ c, ∃ t, t ∈ p c ∧ p c ≠ ∅ ∧ ∀ c₁ c₂, c₁ ≠ c₂ → p c₁ ∩ p c₂ = ∅}
  ∑ p in partitions, 1 = 
  number_of_ways_to_assign_teachers :=
by sorry

end assignment_of_teachers_l317_317426


namespace find_AC_given_triangle_properties_and_circle_l317_317427

variable (A B C D M N P : Type) [LinearOrderedField A]
variable (AD c m n AC : A)
variable (h1 : AB = c)
variable (h2 : AM = m)
variable (h3 : AN = n)
variable (h4 : AD ⊥ BC)
variable (circle_intersects_AB_at_M : circle(A, D, AD).intersects(AB, M))
variable (circle_intersects_AC_at_N : circle(A, D, AD).intersects(AC, N))

theorem find_AC_given_triangle_properties_and_circle 
  (A B C D M N P) (c m n k: A) (h1 : AB = c) (h2 : AM = m) (h3 : AN = n) :
  k = \frac{m \cdot c}{n}
 := sorry

end find_AC_given_triangle_properties_and_circle_l317_317427


namespace no_integer_roots_l317_317901

theorem no_integer_roots (a b : ℤ) : ¬∃ u : ℤ, u^2 + 3 * a * u + 3 * (2 - b^2) = 0 :=
by
  sorry

end no_integer_roots_l317_317901


namespace a_value_for_point_on_x_axis_l317_317437

theorem a_value_for_point_on_x_axis (a : ℝ) : (∃ y : ℝ, P = ⟨4, y⟩ ∧ y = 2 * a + 10 ∧ y = 0) → a = -5 :=
by
  sorry

-- Here ⟨4, y⟩ represents the point (4, y).

end a_value_for_point_on_x_axis_l317_317437


namespace problem_statement_l317_317431

def f (x : ℝ) : ℝ :=
  if x > 2 then x^2 - 3 else abs(x - 2) + 1

theorem problem_statement : f(f(real.sqrt 5)) = 1 := by
  sorry

end problem_statement_l317_317431


namespace heptagon_triangle_count_l317_317828

noncomputable def number_of_triangles_in_heptagon : ℕ :=
  ∑ k in ({6, 5, 4, 3} : Finset ℕ), match k with
  | 6 => 1 * (Nat.choose 7 6)
  | 5 => 5 * (Nat.choose 7 5)
  | 4 => 4 * (Nat.choose 7 4)
  | 3 => 1 * (Nat.choose 7 3)
  | _ => 0

theorem heptagon_triangle_count : number_of_triangles_in_heptagon = 287 := by
  sorry

end heptagon_triangle_count_l317_317828


namespace find_m_plus_n_l317_317953

theorem find_m_plus_n (A B C D : ℕ) (a b c : ℕ)
  (h₀ : right_angle C)
  (h₁ : altitude C meets_line A B at D)
  (h₂ : side_lengths_integers A B C)
  (h₃ : BD = 29^2)
  (h₄ : cos_B = m / n) :
  m + n = 30 :=
sorry

end find_m_plus_n_l317_317953


namespace metallic_sheet_length_l317_317352

theorem metallic_sheet_length (w : ℝ) (s : ℝ) (v : ℝ) (L : ℝ) 
  (h_w : w = 38) 
  (h_s : s = 8) 
  (h_v : v = 5632) 
  (h_volume : (L - 2 * s) * (w - 2 * s) * s = v) : 
  L = 48 :=
by
  -- To complete the proof, follow the mathematical steps:
  -- (L - 2 * s) * (w - 2 * s) * s = v
  -- (L - 2 * 8) * (38 - 2 * 8) * 8 = 5632
  -- Simplify and solve for L
  sorry

end metallic_sheet_length_l317_317352


namespace words_to_score_A_l317_317469

-- Define the total number of words
def total_words : ℕ := 600

-- Define the target percentage
def target_percentage : ℚ := 90 / 100

-- Define the minimum number of words to learn
def min_words_to_learn : ℕ := 540

-- Define the condition for scoring at least 90%
def meets_requirement (learned_words : ℕ) : Prop :=
  learned_words / total_words ≥ target_percentage

-- The goal is to prove that learning 540 words meets the requirement
theorem words_to_score_A : meets_requirement min_words_to_learn :=
by
  sorry

end words_to_score_A_l317_317469


namespace paul_total_spent_l317_317239

noncomputable def total_cost (shirts pants suit sweaters ties shoes tax_rate coupon_rate: ℝ) : ℝ :=
  let initial_cost := 4 * shirts + 2 * pants + suit + 2 * sweaters + 3 * ties + shoes
  let discounted_shirts_cost := (4 * shirts) - (0.20 * (4 * shirts))
  let discounted_pants_cost := (2 * pants) - (0.30 * (2 * pants))
  let discounted_suit_cost := suit - (0.15 * suit)
  let discounted_total := discounted_shirts_cost + discounted_pants_cost + discounted_suit_cost + (2 * sweaters) + (3 * ties) + shoes
  let coupon_discount := discounted_total * coupon_rate
  let subtotal := discounted_total - coupon_discount
  let final_cost := subtotal + (subtotal * tax_rate)
  final_cost

theorem paul_total_spent :
  total_cost 15 40 150 30 20 80 0.05 0.10 = 407.77 :=
by
  compute_simulate
  simp
  sorry

end paul_total_spent_l317_317239


namespace tournament_divisibility_l317_317617

theorem tournament_divisibility :
  ∃ (n : ℕ), n ≥ 43 ∧ ∀ k, k ∈ {1, 2, 23, 43, 46, 86, 506, 989, 1978} \ {43} → 1978 + k = k * m for some m : ℕ :=
by
  sorry

end tournament_divisibility_l317_317617


namespace sample_stat_properties_l317_317868

/-- Given a set of sample data, and another set obtained by adding a non-zero constant to each element of the original set,
prove some properties about their statistical measures. -/
theorem sample_stat_properties (n : ℕ) (c : ℝ) (h₀ : c ≠ 0) 
  (x : ℕ → ℝ) :
  let y := λ i, x i + c 
  in (∑ i in finset.range n, x i) / n ≠ (∑ i in finset.range n, y i) / n ∧ 
     (∃ (median_x : ℝ) (median_y : ℝ), median_y = median_x + c ∧ median_y ≠ median_x) ∧
     (stddev (finset.range n).map x = stddev (finset.range n).map y) ∧
     (range (finset.range n).map x = range (finset.range n).map y) := 
by
  sorry

-- Helper functions for median, stddev, and range could be defined if missing from Mathlib.

end sample_stat_properties_l317_317868


namespace regression_lines_intersect_l317_317658

variables {R : Type*} [LinearOrderedField R] {x y : Type*}

-- Definitions of regression lines l1 and l2.
def regression_line (observations : list (R × R)) : R × R := sorry

 -- Average observed data
variables 
  {s t : R} 
  {data1 : list (R × R)}
  {data2 : list (R × R)}
  (avg_x1 : (∑ xy in data1, xy.1) / data1.length = s)
  (avg_y1 : (∑ xy in data1, xy.2) / data1.length = t)
  (avg_x2 : (∑ xy in data2, xy.1) / data2.length = s)
  (avg_y2 : (∑ xy in data2, xy.2) / data2.length = t)

-- Definition of regression lines l1 and l2 based on data points
def l1 := regression_line data1
def l2 := regression_line data2

-- Proving that the two regression lines intersect at (s, t)
theorem regression_lines_intersect
  (h1 : l1 = (s, t))
  (h2 : l1 ≠ l2 → l2 = (s, t)) :
  ∃ p : R × R, p = (s, t) ∧ (l1 = p ∨ l2 = p) := 
by {
  -- Use the sample center point must lie on linear regression lines
  use (s, t),
  split,
  -- Point of intersection
  refl,
  intro h,
  left,
  assumption,
}

end regression_lines_intersect_l317_317658


namespace inner_polygon_perimeter_le_outer_l317_317997

section geometric_inequality

-- Define the convex polygons
variables {P_in P_out : Type} [convex P_in] [convex P_out] -- Assuming convex is properly defined somewhere

-- Assume P_in is inside P_out
variable (inside : P_in ⊆ P_out)

-- Assume projections of polygons onto any line L
variables {L : Type} (Proj_in Proj_out : L → ℝ)

-- Assume the projection inequality condition
variable (proj_inequality : ∀ l : L, Proj_in l ≤ Proj_out l)

-- Define perimeter as a sum of projection
noncomputable def Perimeter (P : Type) [convex P] := ∑ l : L, (L → ℝ)

-- The theorem to state
theorem inner_polygon_perimeter_le_outer :
  Perimeter P_in ≤ Perimeter P_out :=
sorry

end geometric_inequality

end inner_polygon_perimeter_le_outer_l317_317997


namespace simplify_expression_l317_317259

theorem simplify_expression {x a : ℝ} (h1 : x > a) (h2 : x ≠ 0) (h3 : a ≠ 0) :
  (x * (x^2 - a^2)⁻¹ + 1) / (a * (x - a)⁻¹ + (x - a)^(1 / 2))
  / ((a^2 * (x + a)^(1 / 2)) / (x - (x^2 - a^2)^(1 / 2)) + 1 / (x^2 - a * x))
  = 2 / (x^2 - a^2) :=
by sorry

end simplify_expression_l317_317259


namespace g_50_eq_392_l317_317024
noncomputable theory

def g : ℕ → ℕ
| x := if (∃ (k : ℤ), x = 3^k) then (log 3 x).to_nat
       else 1 + g (x + 2)

theorem g_50_eq_392 : g 50 = 392 := by
  sorry

end g_50_eq_392_l317_317024


namespace sum_of_integers_abs_less_than_five_l317_317287

theorem sum_of_integers_abs_less_than_five : (∑ i in (Finset.filter (λ x : ℤ, |x| < 5) (Finset.Icc (-4) 4)), i) = 0 := by
  sorry

end sum_of_integers_abs_less_than_five_l317_317287


namespace star_m_value_l317_317971

def sum_of_digits (x : ℕ) : ℕ :=
  (x.digits 10).sum

def S : Set ℕ := {n | sum_of_digits n = 15 ∧ n < 10^7}

theorem star_m_value : sum_of_digits (S.to_finset.card) = 9 :=
by
  sorry

end star_m_value_l317_317971


namespace average_class_weight_l317_317270

theorem average_class_weight
  (n_boys n_girls n_total : ℕ)
  (avg_weight_boys avg_weight_girls total_students : ℕ)
  (h1 : n_boys = 15)
  (h2 : n_girls = 10)
  (h3 : n_total = 25)
  (h4 : avg_weight_boys = 48)
  (h5 : avg_weight_girls = 405 / 10) 
  (h6 : total_students = 25) :
  (48 * 15 + 40.5 * 10) / 25 = 45 := 
sorry

end average_class_weight_l317_317270


namespace arrangement_count_of_multiple_of_5_l317_317176

-- Define the digits and the condition that the number must be a five-digit multiple of 5
def digits := [1, 1, 2, 5, 0]
def is_multiple_of_5 (n : Nat) : Prop := n % 5 = 0

theorem arrangement_count_of_multiple_of_5 :
  ∃ (count : Nat), count = 21 ∧
  (∀ (num : List Nat), num.perm digits → is_multiple_of_5 (Nat.of_digits 10 num) → true) :=
begin
  use 21,
  split,
  { refl },
  { intros num h_perm h_multiple_of_5,
    sorry
  }
end

end arrangement_count_of_multiple_of_5_l317_317176


namespace find_s_l317_317738

noncomputable def area_of_parallelogram (s : ℝ) : ℝ :=
  (3 * s) * (s * Real.sin (Real.pi / 3))

theorem find_s (s : ℝ) (h1 : area_of_parallelogram s = 27 * Real.sqrt 3) : s = 3 * Real.sqrt 2 := 
  sorry

end find_s_l317_317738


namespace range_of_a_l317_317973

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) :
  (p := ∀ x, |4 * x - 1| ≤ 1) →
  (q := ∀ x, x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) →
  (¬ (∀ x, |4 * x - 1| ≤ 1) → ¬ (∀ x, x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) ∧ 
  ¬ (¬ (∀ x, |4 * x - 1| ≤ 1) → ¬ (∀ x, x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0))) →
  (-1/2 : ℝ) ≤ a ∧ a ≤ (0 : ℝ) :=
by
  sorry

end range_of_a_l317_317973


namespace sequence_term_25_l317_317938

theorem sequence_term_25 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, n ≥ 2 → a n = (a (n - 1) + a (n + 1)) / 4)
  (h2 : a 1 = 1)
  (h3 : a 9 = 40545) : 
  a 25 = 57424611447841 := 
sorry

end sequence_term_25_l317_317938


namespace find_k_l317_317073

   theorem find_k (m n : ℝ) (k : ℝ) (hm : m > 0) (hn : n > 0)
     (h1 : k = Real.log m / Real.log 2)
     (h2 : k = Real.log n / (Real.log 4))
     (h3 : k = Real.log (4 * m + 3 * n) / (Real.log 8)) :
     k = 2 :=
   by
     sorry
   
end find_k_l317_317073


namespace total_new_cans_l317_317062

-- Define the condition
def initial_cans : ℕ := 256
def first_term : ℕ := 64
def ratio : ℚ := 1 / 4
def terms : ℕ := 4

-- Define the sum of the geometric series
noncomputable def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * ((1 - r ^ n) / (1 - r))

-- Problem statement in Lean 4
theorem total_new_cans : geometric_series_sum first_term ratio terms = 85 := by
  sorry

end total_new_cans_l317_317062


namespace sum_of_reciprocals_l317_317928

variables {a b : ℕ}

def HCF (m n : ℕ) : ℕ := m.gcd n
def LCM (m n : ℕ) : ℕ := m.lcm n

theorem sum_of_reciprocals (h_sum : a + b = 55)
                           (h_hcf : HCF a b = 5)
                           (h_lcm : LCM a b = 120) :
  (1 / a : ℚ) + (1 / b) = 11 / 120 :=
sorry

end sum_of_reciprocals_l317_317928


namespace number_of_rings_l317_317703

def is_number_ring (A : Set ℝ) : Prop :=
  ∀ (a b : ℝ), a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A ∧ (a * b) ∈ A

def Z := { n : ℝ | ∃ k : ℤ, n = k }
def N := { n : ℝ | ∃ k : ℕ, n = k }
def Q := { n : ℝ | ∃ (a b : ℤ), b ≠ 0 ∧ n = a / b }
def R := { n : ℝ | True }
def M := { x : ℝ | ∃ (n m : ℤ), x = n + m * Real.sqrt 2 }
def P := { x : ℝ | ∃ (m n : ℕ), n ≠ 0 ∧ x = m / (2 * n) }

theorem number_of_rings :
  (is_number_ring Z) ∧ ¬(is_number_ring N) ∧ (is_number_ring Q) ∧ 
  (is_number_ring R) ∧ (is_number_ring M) ∧ ¬(is_number_ring P) :=
by sorry

end number_of_rings_l317_317703


namespace perpendicular_vector_dot_product_l317_317500

theorem perpendicular_vector_dot_product (x : ℝ) :
  let a : ℝ × ℝ × ℝ := (2, 4, -4)
  let b : ℝ × ℝ × ℝ := (-2, x, 2)
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0 → x = 3 :=
by
  let a : ℝ × ℝ × ℝ := (2, 4, -4)
  let b : ℝ × ℝ × ℝ := (-2, x, 2)
  sorry

end perpendicular_vector_dot_product_l317_317500


namespace plane_existence_possibilities_l317_317740

-- Definition of the conditions
def plane_parallel_to_line_through_points
  (line : Set Point)
  (p1 p2 : Point) : Prop :=
  ∃ plane : Set Point,
    (∀ l ∈ line, ∀ p ∈ plane, ((l - p) × p1) = 0 ∧ ((l - p) × p2) = 0)

-- The theorem to be proved
theorem plane_existence_possibilities
  (line : Set Point)
  (p1 p2 : Point)
  (parallel_condition : ∃ l ∈ line, ∀ plane : Set Point, ((l - p1) × p1 = 0) ∧ ((l - p2) × p2 = 0)) :
  (∃! plane, plane_parallel_to_line_through_points line p1 p2) ∨ 
  (∃ planes, ∀ plane ∈ planes, plane_parallel_to_line_through_points line p1 p2) ∨ 
  (¬ ∃ plane, plane_parallel_to_line_through_points line p1 p2) :=
sorry

end plane_existence_possibilities_l317_317740


namespace no_perfect_square_in_sequence_l317_317133

def sequence_term (i : ℕ) : ℕ :=
  let baseDigits : List ℕ := [2, 0, 1, 4, 2, 0, 1, 5]
  let termDigits := baseDigits.foldr (fun d acc => acc + d * 10 ^ (baseDigits.length - acc.length - 1)) 0
  termDigits + 10 ^ (i + 5)

theorem no_perfect_square_in_sequence : ¬ ∃ i : ℕ, ∃ k : ℕ, k * k = sequence_term i := 
sorry

end no_perfect_square_in_sequence_l317_317133


namespace car_B_travel_time_correct_graph_D_l317_317385

-- Definitions of the constants and conditions
variables {v d t : ℝ} -- speed of car A, distance, time taken by car A

-- Car A travels at a constant speed
axiom car_A_speed : ∀ t', 0 ≤ t' → t' ≤ t → v * t' = d * (t' / t)

-- Car B travels at 1.5 times the speed of Car A
def car_B_speed : ℝ := 1.5 * v

-- Car B travels the same distance in 2/3 of the time of Car A
theorem car_B_travel_time : (2/3 : ℝ) * t = d / car_B_speed := 
by sorry

-- Both cars start their journeys at the same time
axiom same_start : ∀ t', 0 ≤ t' → 0 ≤ t'

-- The graph representing the distance versus time for car A
def car_A_graph (t' : ℝ) : ℝ := v * t'

-- The graph representing the distance versus time for car B
def car_B_graph : ∀ t', 0 ≤ t' → t' ≤ (2/3) * t → ℝ := 
λ t', (1.5 * v) * t'

-- Prove that the correct graph shows Car B reaches the distance in 2/3 the time of Car A
theorem correct_graph_D : 
    ∀ t', 0 ≤ t' → t' ≤ t → (car_A_graph t' = car_B_graph t' ↔ t' = (2/3 : ℝ) * t) :=
by 
  intro t' 
  intro h1
  intro h2
  sorry

end car_B_travel_time_correct_graph_D_l317_317385


namespace quadratic_square_binomial_l317_317312

theorem quadratic_square_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 18 * x + k = (x + b) ^ 2) ↔ k = 81 := by
  sorry

end quadratic_square_binomial_l317_317312


namespace log_x_125_l317_317481

theorem log_x_125 {x : ℝ} (h : log 8 (5 * x) = 3) : 
  log x 125 = 3 * (1 / (9 * log 5 2 - 1)) :=
sorry

end log_x_125_l317_317481


namespace product_of_roots_l317_317818

theorem product_of_roots : ∀ (x : ℝ), (x + 3) * (x - 4) = 2 * (x + 1) → 
  let a := 1
  let b := -3
  let c := -14
  let product_of_roots := c / a
  product_of_roots = -14 :=
by
  intros x h
  let a := 1
  let b := -3
  let c := -14
  let product_of_roots := c / a
  sorry

end product_of_roots_l317_317818


namespace exists_x_y_m_l317_317980

theorem exists_x_y_m (p : ℕ) (hp : p.prime) (hodd : odd p) :
  ∃ (x y m : ℤ), 1 + x^2 + y^2 = m * p ∧ 0 < m ∧ ↑m < ↑p :=
by
  sorry

end exists_x_y_m_l317_317980


namespace ratio_of_x_to_y_l317_317158

variable {x y : ℝ}

theorem ratio_of_x_to_y (h1 : (3 * x - 2 * y) / (2 * x + 3 * y) = 5 / 4) (h2 : x + y = 5) : x / y = 23 / 2 := 
by {
  sorry
}

end ratio_of_x_to_y_l317_317158


namespace last_locker_opens_l317_317757

theorem last_locker_opens (n : ℕ) (h₀ : n = 511) 
  (h₁ : ∀ k ≤ n, k.even → locker k has been opened by the first pass)
  (h₂ : ∀ k ≤ n, k.odd → k≠ 511 → k has been opened before locker 511):
  locker 511 has been opened last :=
sorry

-- Auxiliary definitions for understanding.

def locker (k : ℕ) : Prop := 0 ≤ k ∧ k ≤ 511
def even (k : ℕ) : Prop := k % 2 = 0
def odd (k : ℕ) : Prop := k % 2 = 1

-- States that locker is now open. 
-- Representing the action could be elaborated more based on system setup.
def is_open (k : ℕ) : Prop := true -- Simplistic representation for state.

end last_locker_opens_l317_317757


namespace points_concyclic_l317_317974

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def foot_of_altitude (A B C : Triangle) : Point := sorry
noncomputable def incircle_touch (Δ : Triangle) (side : Line) : Point := sorry
noncomputable def excircle_touch (Δ : Triangle) (side : Line) : Point := sorry
noncomputable def common_tangent_intersection (C₁ C₂ : Circle) (line : Line) : Point := sorry
noncomputable def circle_through_two_points (P Q : Point) : Circle := sorry

theorem points_concyclic 
(ABC : Triangle) 
(K : Point) (M : Point) (D : Point) (E : Point) 
(F : Point) (G : Point) 
(hK : K = midpoint (ABC.ABC_sides.BC.A.B))
(hM : M = foot_of_altitude ABC.ABC_sides.ABC (ABC.ABC_sides.BC.A.B))
(hD : D = incircle_touch ABC ABC.ABC_sides.BC)
(hE : E = excircle_touch ABC ABC.ABC_sides.BC)
(hF : ∃ C₁ C₂ : Circle, F = (common_tangent_intersection C₁ C₂ ABC.ABC_sides.ABC))
(hG : ∃ C₁ C₂ : Circle, G = (common_tangent_intersection C₁ C₂ ABC.ABC_sides.ABC)) :
is_cyclic_four_points D E F G := 
sorry

end points_concyclic_l317_317974


namespace angle_A_measure_bc_range_l317_317518

noncomputable def acute_triangle :=
∀ (A B C a b c : ℝ),
  (∀ (x : ℝ), 0 < x) →
  (a = 1 ∧ b = 1 ∧ c = 1) →
  2 * a * sin A * (b^2 + c^2 - a^2) = c * sin B * (a^2 + b^2 - c^2) + b * sin C * (a^2 + c^2 - b^2)

theorem angle_A_measure : ∀ (A B C a b c : ℝ),
  acute_triangle A B C a b c →
  A = π / 3 := 
sorry

theorem bc_range : ∀ (b c : ℝ),
  2 * b * sin (π - B - C) = 1 →
  2 * b * c  ≠ 0 →
  1 ≥ 0 ∧ 1 ≤ 4 :=
sorry

end angle_A_measure_bc_range_l317_317518


namespace volume_of_water_l317_317470

theorem volume_of_water (h w l : ℕ) (h_eq : h = 5) (w_eq : w = 10) (l_eq : l = 12) : (h * w * l) = 600 := by
  rw [h_eq, w_eq, l_eq]
  exact by norm_num

end volume_of_water_l317_317470


namespace sum_of_n_values_l317_317688

theorem sum_of_n_values : 
  (∑ n in {n : ℤ | ∃ k : ℤ, 35 = k * (2 * n - 1)}.to_finset, n) = 26 := 
by
  sorry

end sum_of_n_values_l317_317688


namespace sequence_a2017_l317_317080

theorem sequence_a2017 (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 3)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 2017 = 2 :=
sorry

end sequence_a2017_l317_317080


namespace number_of_integer_x_satisfying_abs_x_minus_1_lt_4pi_l317_317467

theorem number_of_integer_x_satisfying_abs_x_minus_1_lt_4pi : 
  {x : ℤ | |(x : ℝ) - 1| < 4 * Real.pi}.finite ∧ 
  {x : ℤ | |(x : ℝ) - 1| < 4 * Real.pi}.to_finset.card = 25 := 
by
  sorry

end number_of_integer_x_satisfying_abs_x_minus_1_lt_4pi_l317_317467


namespace find_white_towels_l317_317568

variable (W : ℕ) -- Let W be the number of white towels Maria bought

def green_towels : ℕ := 40
def towels_given : ℕ := 65
def towels_left : ℕ := 19

theorem find_white_towels :
  green_towels + W - towels_given = towels_left →
  W = 44 :=
by
  intro h
  sorry

end find_white_towels_l317_317568


namespace triangle_DEF_angles_l317_317909

-- Definitions
variables {α β γ : ℝ}
variables (A B C D E F : Type*)

-- Conditions
def is_triangle (A B C : Type*) : Prop := true  -- Placeholder definition
def is_tangent (circle_triangle : Type*) (D : Type*) (BC : Type*) : Prop := true  -- Placeholder definition
def extension_tangent (circle_triangle : Type*) (F : Type*) (AB : Type*) (A : Type*) : Prop := true  -- Placeholder definition
def extension_tangent2 (circle_triangle : Type*) (E : Type*) (AC : Type*) (C : Type*) : Prop := true  -- Placeholder definition

-- Statement to be proved
theorem triangle_DEF_angles : 
  is_triangle A B C →
  is_tangent (escribed_circle A B C) D (segment B C) →
  extension_tangent (escribed_circle A B C) F (extension A B) A →
  extension_tangent2 (escribed_circle A B C) E (extension A C) C →
  ∃ ΔDEF : Triangle, angles_are DEF (one_obtuse_and_two_acute ∧ angles_are_unequal) :=
by {
  sorry
}

end triangle_DEF_angles_l317_317909


namespace measure_of_angle_A_lengths_of_sides_l317_317881

-- Define the setup of the triangle and the given condition
def triangle_given_conditions (a b c : ℝ) (A B C : ℝ) :=
  ∠A + ∠B + ∠C = π ∧
  a = 7 ∧
  sin C = 2 * sin B ∧
  a / (2 * b + c) = -cos A / cos C

-- Define the Lean statement for proving the measure of angle A
theorem measure_of_angle_A 
  (a b c A B C : ℝ) :
  triangle_given_conditions a b c A B C →
  cos A = -1 / 2 :=
sorry

-- Define the Lean statement for proving the sides given the other conditions
theorem lengths_of_sides
  (a b c A B C : ℝ) :
  triangle_given_conditions a b c A B C →
  b = real.sqrt 7 ∧ c = 2 * real.sqrt 7 :=
sorry

end measure_of_angle_A_lengths_of_sides_l317_317881


namespace even_function_f_l317_317265

noncomputable def f (x : ℝ) : ℝ := if 0 < x ∧ x < 10 then Real.log x else 0

theorem even_function_f (x : ℝ) (h : f (-x) = f x) (h1 : ∀ x, 0 < x ∧ x < 10 → f x = Real.log x) :
  f (-Real.exp 1) + f (Real.exp 2) = 3 := by
  sorry

end even_function_f_l317_317265


namespace problem_statement_l317_317190

noncomputable def acute_triangle := sorry -- Placeholder for acute triangle definition
noncomputable def circumcenter (A B C : Point) := sorry -- Placeholder for circumcenter definition
noncomputable def incenter (A B C : Point) := sorry -- Placeholder for incenter definition
noncomputable def touches (circle : Circle) (side : Segment) (point : Point) := sorry -- Placeholder for touch condition
noncomputable def intersect (line1 line2 : Line) := sorry -- Placeholder for intersection point
noncomputable def altitude (point1 point2 point3 : Point) := sorry -- Placeholder for altitude definition
noncomputable def tangent_intersection (circle1 circle2 : Circle) := sorry -- Placeholder for tangent intersection
noncomputable def diameter (circle : Circle) := sorry -- Placeholder for diameter definition
noncomputable def collinear (point1 point2 point3 : Point) := sorry -- Placeholder for collinear definition
noncomputable def concyclic (points : List Point) := sorry -- Placeholder for concyclic definition

theorem problem_statement (A B C O I D X Y L P Q : Point)
  (h_acute_triangle : acute_triangle A B C)
  (h_circumcenter : O = circumcenter A B C)
  (h_incenter : I = incenter A B C)
  (h_touches : touches (incenter A B C) (BC_segment) D)
  (h_intersect_AO_BC : X = intersect (line_through A O) (BC_segment))
  (h_altitude_AY_BC : Y = altitude A B C)
  (h_tangent_intersection : L = tangent_intersection (circumcircle A B C) (circumcircle B C O))
  (h_diameter_PQ : PQ = diameter (circumcircle A B C))
  (h_P_through_I : I ∈ (line_through P Q)) :
  collinear A D L ↔ concyclic [P, X, Y, Q] :=
begin
  sorry
end

end problem_statement_l317_317190


namespace jake_present_weight_l317_317328

variables (J S : ℕ)

theorem jake_present_weight :
  (J - 33 = 2 * S) ∧ (J + S = 153) → J = 113 :=
by
  sorry

end jake_present_weight_l317_317328


namespace directrix_of_parabola_l317_317406

-- Define the given parabola equation
def parabola_eq (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 5

-- Define the expected result for the directrix
def directrix_eq : ℝ := -23 / 12

-- State the problem in Lean
theorem directrix_of_parabola : 
  (∃ d : ℝ, (∀ x y : ℝ, y = parabola_eq x → y = d) → d = directrix_eq) :=
by
  sorry

end directrix_of_parabola_l317_317406


namespace problem_equivalence_l317_317497

variables (P Q : Prop)

theorem problem_equivalence :
  (P ↔ Q) ↔ ((P → Q) ∧ (Q → P) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q)) :=
by sorry

end problem_equivalence_l317_317497


namespace simplify_expression_l317_317590

-- Define the constants.
def a : ℚ := 8
def b : ℚ := 27

-- Assuming cube root function is available and behaves as expected for rationals.
def cube_root (x : ℚ) : ℚ := x^(1/3 : ℚ)

-- Assume the necessary property of cube root of 27.
axiom cube_root_27_is_3 : cube_root 27 = 3

-- The main statement to prove.
theorem simplify_expression : cube_root (a + b) * cube_root (a + cube_root b) = cube_root 385 :=
by
  sorry

end simplify_expression_l317_317590


namespace increasing_interval_of_f_l317_317637

def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval_of_f :
  ∀ x, 2 < x → ∀ y, x < y → f x ≤ f y := sorry

end increasing_interval_of_f_l317_317637


namespace sally_weekly_bread_l317_317993

-- Define the conditions
def monday_bread : Nat := 3
def tuesday_bread : Nat := 2
def wednesday_bread : Nat := 4
def thursday_bread : Nat := 2
def friday_bread : Nat := 1
def saturday_bread : Nat := 2 * 2  -- 2 sandwiches, 2 pieces each
def sunday_bread : Nat := 2

-- Define the total bread count
def total_bread : Nat := 
  monday_bread + 
  tuesday_bread + 
  wednesday_bread + 
  thursday_bread + 
  friday_bread + 
  saturday_bread + 
  sunday_bread

-- The proof statement
theorem sally_weekly_bread : total_bread = 18 := by
  sorry

end sally_weekly_bread_l317_317993


namespace find_jack_euros_l317_317539

theorem find_jack_euros (E : ℕ) (h1 : 45 + 2 * E = 117) : E = 36 :=
by
  sorry

end find_jack_euros_l317_317539


namespace blue_label_multiple_of_three_l317_317574

theorem blue_label_multiple_of_three :
  let marked_points := 4038
  let chords := 2019
  let total_endpoints := 4038
  let yellow_values := ∀ (n : ℕ), 0 ≤ n ∧ n ≤ chords
  ∃ (a b : ℕ), (0 ≤ a ∧ a < marked_points) ∧ (0 ≤ b ∧ b < marked_points) ∧ a ≠ b ∧ (a - b) % 3 = 0 

-- Assumptions based on conditions
  (∀ (points : ℕ), points = total_endpoints): ∃ (u v : ℕ), (0 ≤ u ∧ u < marked_points) ∧ (0 ≤ v ∧ v < marked_points) ∧ u ≠ v ∧ abs(u - v) % 3 = 0 := 
sorry

end blue_label_multiple_of_three_l317_317574


namespace compute_diff_squares_l317_317794

theorem compute_diff_squares (a b : ℤ) (ha : a = 153) (hb : b = 147) :
  a ^ 2 - b ^ 2 = 1800 :=
by
  rw [ha, hb]
  sorry

end compute_diff_squares_l317_317794


namespace sequence_not_perfect_square_l317_317142

theorem sequence_not_perfect_square (n : ℕ) (N : ℕ → ℕ) :
  (∀ i, let d := [2, 0, 0, 1, 4, 0, 2, 0, 1, 5]
  in (N i = 2014 * 10^(n - i) + list.sum d) ∧ list.sum d = 15) →
  (∀ i, ¬ is_square (N i)) :=
by
  sorry

end sequence_not_perfect_square_l317_317142


namespace primes_between_2_and_100_l317_317230

open Nat

theorem primes_between_2_and_100 :
  { p : ℕ | 2 ≤ p ∧ p ≤ 100 ∧ Nat.Prime p } = 
  {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97} :=
by
  sorry

end primes_between_2_and_100_l317_317230


namespace at_least_one_worker_must_wait_l317_317831

/-- 
Given five workers who collectively have a salary of 1500 rubles, 
and each tape recorder costs 320 rubles, we need to prove that 
at least one worker will not be able to buy a tape recorder immediately. 
-/
theorem at_least_one_worker_must_wait 
  (num_workers : ℕ) 
  (total_salary : ℕ) 
  (tape_recorder_cost : ℕ) 
  (h_workers : num_workers = 5) 
  (h_salary : total_salary = 1500) 
  (h_cost : tape_recorder_cost = 320) :
  ∀ (tape_recorders_required : ℕ), 
    tape_recorders_required = num_workers → total_salary < tape_recorder_cost * tape_recorders_required → ∃ (k : ℕ), 1 ≤ k ∧ k ≤ num_workers ∧ total_salary < k * tape_recorder_cost :=
by 
  intros tape_recorders_required h_required h_insufficient
  sorry

end at_least_one_worker_must_wait_l317_317831


namespace range_of_m_l317_317499

noncomputable def proof_problem (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1/x + 2/y = 1) : Prop :=
  ∃ x y : ℝ, (0 < x) ∧ (0 < y) ∧ (1/x + 2/y = 1) ∧ (x + y / 2 < m^2 + 3 * m) ↔ (m < -4 ∨ m > 1)

theorem range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1/x + 2/y = 1) :
  proof_problem x y m hx hy hxy :=
sorry

end range_of_m_l317_317499


namespace evaluate_star_l317_317813

-- Define the custom operation
def star (x y : ℝ) : ℝ := (x + y) / 3

-- State the theorem that we need to prove
theorem evaluate_star : star (star 3 15) 9 = 5 := sorry

end evaluate_star_l317_317813


namespace circles_intersect_l317_317129

-- Definitions from the conditions
def circle1_equation (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 6 * y = 0
def circle2_equation (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 6 * y - 26 = 0

-- The theorem statement based on the question and conditions
theorem circles_intersect : 
  (∃ x y : ℝ, circle1_equation x y ∧ circle2_equation x y) :=
begin
  sorry
end

end circles_intersect_l317_317129


namespace triangle_angles_sum_eq_30_degrees_l317_317869

variables {A B C D E F : Type*}

-- Given a triangle ABC with points D, E, F on BC, AC, AB respectively
-- and the conditions BC = 3BD, BA = 3BF, EA = (1/3)AC,
-- prove that the sum of angles ∠ADE + ∠FEB = 30°.
theorem triangle_angles_sum_eq_30_degrees
  (h1 : BC = 3 * BD)
  (h2 : BA = 3 * BF)
  (h3 : EA = 1 / 3 * AC) :
  ∠ADE + ∠FEB = 30 := 
  sorry

end triangle_angles_sum_eq_30_degrees_l317_317869


namespace train_speed_l317_317367

theorem train_speed (train_length bridge_length time_seconds : ℕ) 
  (h1 : train_length = 160) 
  (h2 : bridge_length = 215) 
  (h3 : time_seconds = 30) : 
  (train_length + bridge_length) / time_seconds * 3.6 = 45 :=
by sorry

end train_speed_l317_317367


namespace sum_free_image_l317_317814

open Set

def is_sum_free (A : Set ℕ) : Prop :=
  ∀ x y ∈ A, x + y ∉ A

theorem sum_free_image (f : ℕ → ℕ) (hf : surjective f) :
  (∀ (A : Set ℕ), is_sum_free A → is_sum_free (f '' A)) ↔ (∀ n, f n = n) :=
begin
  sorry
end

end sum_free_image_l317_317814


namespace non_congruent_rectangles_l317_317851

theorem non_congruent_rectangles (a b : ℝ) (h : a < b) :
  ∀ (x y : ℝ), x < a → y < a → x + y = (a + b) / 4 → x * y = (a * b) / 4 → x = y :=
by
  assume (x y : ℝ) (hx : x < a) (hy : y < a) (hsum : x + y = (a + b) / 4) (hprod : x * y = (a * b) / 4)
  sorry

end non_congruent_rectangles_l317_317851


namespace polynomial_roots_integer_l317_317825

theorem polynomial_roots_integer {P : Polynomial ℤ} :
  (∀ n : ℕ, ∀ x : ℤ, is_root (Polynomial.eval (P.eval)^[n] x)) →
  (∃ (a : ℤ) (b : ℕ), (a ≠ 0 ∧ b ≥ 2 ∧ P = Polynomial.C a * Polynomial.X ^ b) ∨
   ∃ c : ℤ, (P = Polynomial.X + Polynomial.C c) ∨ (P = -Polynomial.X + Polynomial.C c)) :=
by sorry

end polynomial_roots_integer_l317_317825


namespace sum_of_cubes_correct_l317_317876

noncomputable def expression_for_sum_of_cubes (x y z w a b c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (hxy : x * y = a) (hxz : x * z = b) (hyz : y * z = c) (hxw : x * w = d) : Prop :=
  x^3 + y^3 + z^3 + w^3 = (a^3 * d^3 + a^3 * c^3 + b^3 * d^3 + b^3 * d^3) / (a * b * c * d)

theorem sum_of_cubes_correct (x y z w a b c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (hxy : x * y = a) (hxz : x * z = b) (hyz : y * z = c) (hxw : x * w = d) :
  expression_for_sum_of_cubes x y z w a b c d hx hy hz hw ha hb hc hd hxy hxz hyz hxw :=
sorry

end sum_of_cubes_correct_l317_317876


namespace find_eg_dot_fh_l317_317079

variables {V : Type*} [inner_product_space ℝ V]

-- Definitions and conditions given in the problem
variable P A B C E F G H : V

noncomputable def edge_length := 1
def tetrahedron_regular : Prop := 
  dist P A = edge_length ∧
  dist P B = edge_length ∧
  dist P C = edge_length ∧
  dist A B = edge_length ∧
  dist A C = edge_length ∧
  dist B C = edge_length

def is_edge_point (u v : V) (t : ℝ) : Prop := 
  0 ≤ t ∧ t ≤ 1 ∧ E = t • u + (1 - t) • v

def points_on_edges : Prop := 
  is_edge_point P A E ∧
  is_edge_point P B F ∧
  is_edge_point C A G ∧
  is_edge_point C B H

def vector_condition1 : Prop := 
  E + F = B

def vector_condition2 : Prop := 
  inner_product (E - H) (F - G) = 1 / 18

theorem find_eg_dot_fh :
  tetrahedron_regular →
  points_on_edges →
  vector_condition1 →
  vector_condition2 →
  inner_product (E - G) (F - H) = 5 / 18 :=
sorry

end find_eg_dot_fh_l317_317079


namespace ladder_length_l317_317168

theorem ladder_length (w k : ℝ) (a : ℝ) (angle1 angle2 : ℝ)
  (hangle1 : angle1 = 30 * real.pi / 180)
  (hangle2 : angle2 = 60 * real.pi / 180)
  (hw : w = 10)
  (hk : k = 5)
  (h_ladder1 : k = a * real.sin angle1)
  (h_ladder2 : real.sin angle2 = (w / 2) / a) :
  a = 10 :=
by sorry

end ladder_length_l317_317168


namespace cost_of_5_pound_bag_l317_317731

variable (x : ℝ)
variable (w_min w_max : ℝ)
variable (p10 p25 cost_min : ℝ)

-- Given conditions
def conditions : Prop :=
  w_min = 65 ∧ w_max = 80 ∧
  p10 = 20.43 ∧ p25 = 32.25 ∧
  cost_min = 98.75

-- Proof problem: Prove that the cost of the 5-pound bag is $13.82
theorem cost_of_5_pound_bag (h : conditions)
  (h_least_cost : 2 * p25 + p10 + x = cost_min) : x = 13.82 :=
sorry

end cost_of_5_pound_bag_l317_317731


namespace price_increase_to_restore_original_price_l317_317332

theorem price_increase_to_restore_original_price :
  let original_price : ℝ := 100
  let first_reduction : ℝ := original_price * 0.25
  let second_reduction_price : ℝ := (original_price - first_reduction) * 0.3
  let final_price : ℝ := (original_price - first_reduction) - second_reduction_price
  ∃ x : ℝ, final_price * (1 + x / 100) = original_price :=
begin
  sorry
end

end price_increase_to_restore_original_price_l317_317332


namespace smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l317_317680

theorem smallest_positive_perfect_square_divisible_by_5_and_6_is_900 :
  ∃ n : ℕ, 0 < n ∧ (n ^ 2) % 5 = 0 ∧ (n ^ 2) % 6 = 0 ∧ (n ^ 2 = 900) := by
  sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l317_317680


namespace g_50_eq_392_l317_317025
noncomputable theory

def g : ℕ → ℕ
| x := if (∃ (k : ℤ), x = 3^k) then (log 3 x).to_nat
       else 1 + g (x + 2)

theorem g_50_eq_392 : g 50 = 392 := by
  sorry

end g_50_eq_392_l317_317025


namespace angles_NAK_eq_NCK_l317_317355

-- Let the statements and definitions for the conditions be assumed
variables (A B C D K N M T : Type) 
variables [EuclideanGeometry A B C D]
variables (midpoint_AD : Midpoint A D M)
variables (midpoint_CD : Midpoint C D T)
variables (midpoint_BK : Midpoint B K N)
variables (equidistant_M : dist M K = dist M C)
variables (equidistant_T : dist T K = dist T A)

-- The theorem to prove: angles NAK and NCK are equal
theorem angles_NAK_eq_NCK (A B C D K N M T : Type) 
  [EuclideanGeometry A B C D] 
  (midpoint_AD : Midpoint A D M)
  (midpoint_CD : Midpoint C D T)
  (midpoint_BK : Midpoint B K N)
  (equidistant_M : dist M K = dist M C)
  (equidistant_T : dist T K = dist T A) :
  ∠NAK = ∠NCK :=
sorry

end angles_NAK_eq_NCK_l317_317355


namespace balloon_difference_l317_317324

def num_balloons_you := 7
def num_balloons_friend := 5

theorem balloon_difference : (num_balloons_you - num_balloons_friend) = 2 := by
  sorry

end balloon_difference_l317_317324


namespace product_is_correct_l317_317237

noncomputable def IKS := 521
noncomputable def KSI := 215
def product := 112015

theorem product_is_correct : IKS * KSI = product :=
by
  -- Proof yet to be constructed
  sorry

end product_is_correct_l317_317237


namespace system_equations_solution_exists_l317_317061

theorem system_equations_solution_exists (m : ℝ) :
  (∃ x y : ℝ, y = 3 * m * x + 6 ∧ y = (4 * m - 3) * x + 9) ↔ m ≠ 3 :=
by
  sorry

end system_equations_solution_exists_l317_317061


namespace range_of_a_l317_317341

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = abs (x - 2) + abs (x + a) ∧ f x ≥ 3) : a ≤ -5 ∨ a ≥ 1 :=
sorry

end range_of_a_l317_317341


namespace inequality_proof_l317_317439

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    sqrt ((a + c) ^ 2 + (b + d) ^ 2) + (2 * |a * d - b * c|) / sqrt ((a + c) ^ 2 + (b + d) ^ 2) 
    ≥ sqrt (a ^ 2 + b ^ 2) + sqrt (c ^ 2 + d ^ 2) ∧ sqrt (a ^ 2 + b ^ 2) + sqrt (c ^ 2 + d ^ 2) 
    ≥ sqrt ((a + c) ^ 2 + (b + d) ^ 2) :=
begin
    sorry
end

end inequality_proof_l317_317439


namespace square_of_85_l317_317015

-- Define the given variables and values
def a := 80
def b := 5
def c := a + b

theorem square_of_85:
  c = 85 → (c * c) = 7225 :=
by
  intros h
  rw h
  sorry

end square_of_85_l317_317015


namespace original_gift_card_value_l317_317587

def gift_card_cost_per_pound : ℝ := 8.58
def coffee_pounds_bought : ℕ := 4
def remaining_balance_after_purchase : ℝ := 35.68

theorem original_gift_card_value :
  (remaining_balance_after_purchase + coffee_pounds_bought * gift_card_cost_per_pound) = 70.00 :=
by
  -- Proof goes here
  sorry

end original_gift_card_value_l317_317587


namespace find_pairs_l317_317393

theorem find_pairs (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) :
  y ∣ x^2 + 1 ∧ x^2 ∣ y^3 + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 2) :=
by
  sorry

end find_pairs_l317_317393


namespace triangles_to_square_l317_317991

theorem triangles_to_square (figure : set (ℝ × ℝ)) 
  (h_divided : figure.divided_into 5 triangles) : 
  ∃ square : set (ℝ × ℝ), (∀ triangle ∈ triangles, triangle ⊆ square) ∧ figure.area = square.area :=
begin
  sorry
end

end triangles_to_square_l317_317991


namespace solve_for_x_l317_317801

noncomputable def dimensions := λ x : ℝ, (x + 1, 3 * x - 4)
noncomputable def area := λ x : ℝ, (x + 1) * (3 * x - 4)

theorem solve_for_x (x : ℝ) : 
  area x = 12 * x - 19 → x = (13 + Real.sqrt 349) / 6 :=
by
  sorry

end solve_for_x_l317_317801


namespace collinear_points_eq_l317_317890

variables (p q : ℝ)
def A : ℝ × ℝ × ℝ := (1, 5, -2)
def B : ℝ × ℝ × ℝ := (2, 4, 1)
def C : ℝ × ℝ × ℝ := (p, 3, q + 2)

def vector_sub (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2, Q.3 - P.3)

def are_collinear (v1 v2 : ℝ × ℝ × ℝ) :=
  ∃ λ : ℝ, v1 = (λ * v2.1, λ * v2.2, λ * v2.3)

theorem collinear_points_eq :
  are_collinear (vector_sub A B) (vector_sub A C) →
  p = 3 ∧ q = 2 :=
by
  sorry

end collinear_points_eq_l317_317890


namespace domain_of_f_l317_317621

-- The domain of the function is the set of all x such that the function is defined.
theorem domain_of_f:
  {x : ℝ | x > 3 ∧ x ≠ 4} = (Set.Ioo 3 4 ∪ Set.Ioi 4) := 
sorry

end domain_of_f_l317_317621


namespace population_meets_capacity_l317_317234

-- Define the initial conditions and parameters
def initial_year : ℕ := 1998
def initial_population : ℕ := 100
def population_growth_rate : ℕ := 4  -- quadruples every 20 years
def years_per_growth_period : ℕ := 20
def land_area_hectares : ℕ := 15000
def hectares_per_person : ℕ := 2
def maximum_capacity : ℕ := land_area_hectares / hectares_per_person

-- Define the statement
theorem population_meets_capacity :
  ∃ (years_from_initial : ℕ), years_from_initial = 60 ∧
  initial_population * population_growth_rate ^ (years_from_initial / years_per_growth_period) ≥ maximum_capacity :=
by
  sorry

end population_meets_capacity_l317_317234


namespace directrix_of_parabola_l317_317407

-- Define the given parabola equation
def parabola_eq (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 5

-- Define the expected result for the directrix
def directrix_eq : ℝ := -23 / 12

-- State the problem in Lean
theorem directrix_of_parabola : 
  (∃ d : ℝ, (∀ x y : ℝ, y = parabola_eq x → y = d) → d = directrix_eq) :=
by
  sorry

end directrix_of_parabola_l317_317407


namespace bisection_method_iterations_l317_317072

noncomputable def f (x : ℝ) : ℝ := -real.log x

theorem bisection_method_iterations :
  (∃ x₀ ∈ Ioo 1 2, f x₀ = 0) →
  ∃ n : ℕ, n = 4 ∧ ∀ a b : ℝ, 1 < a ∧ b < 2 ∧ f a * f b < 0 →
    let ε := 0.1 in
    let I₀ := (a, b) in
    let I := (a, (a + b) / 2) in
    abs ((a + b) / 2 - I₀.1) < ε :=
begin
  sorry
end

end bisection_method_iterations_l317_317072


namespace car_speed_in_mph_l317_317346

-- Defining the given conditions
def fuel_efficiency : ℚ := 56 -- kilometers per liter
def gallons_to_liters : ℚ := 3.8 -- liters per gallon
def kilometers_to_miles : ℚ := 1 / 1.6 -- miles per kilometer
def fuel_decrease_gallons : ℚ := 3.9 -- gallons
def time_hours : ℚ := 5.7 -- hours

-- Using definitions to compute the speed
theorem car_speed_in_mph :
  (fuel_decrease_gallons * gallons_to_liters * fuel_efficiency * kilometers_to_miles) / time_hours = 91 :=
sorry

end car_speed_in_mph_l317_317346


namespace lisa_takes_72_more_minutes_than_ken_l317_317213

theorem lisa_takes_72_more_minutes_than_ken
  (ken_speed : ℕ) (lisa_speed : ℕ) (book_pages : ℕ)
  (h_ken_speed: ken_speed = 75)
  (h_lisa_speed: lisa_speed = 60)
  (h_book_pages: book_pages = 360) :
  ((book_pages / lisa_speed:ℚ) - (book_pages / ken_speed:ℚ)) * 60 = 72 :=
by
  sorry

end lisa_takes_72_more_minutes_than_ken_l317_317213


namespace sample_statistics_comparison_l317_317859

variable {α : Type*} [LinearOrder α] [Field α]

def sample_data_transformed (x : list α) (c : α) : list α :=
  x.map (λ xi, xi + c)

def sample_mean (x : list α) : α :=
  x.sum / (x.length : α)

def sample_median (x : list α) : α :=
  if x.length % 2 = 1 then
    x.sort.nth_le (x.length / 2) (by sorry)
  else
    (x.sort.nth_le (x.length / 2 - 1) (by sorry) + x.sort.nth_le (x.length / 2) (by sorry)) / 2

def sample_variance (x : list α) : α :=
  let m := sample_mean x
  in (x.map (λ xi, (xi - m)^2)).sum / (x.length : α)

def sample_standard_deviation (x : list α) : α :=
  (sample_variance x).sqrt

def sample_range (x : list α) : α :=
  x.maximum (by sorry) - x.minimum (by sorry)

theorem sample_statistics_comparison (x : list α) (c : α) (hc : c ≠ 0) :
  (sample_mean (sample_data_transformed x c) ≠ sample_mean x) ∧
  (sample_median (sample_data_transformed x c) ≠ sample_median x) ∧
  (sample_standard_deviation (sample_data_transformed x c) = sample_standard_deviation x) ∧
  (sample_range (sample_data_transformed x c) = sample_range x) :=
  sorry

end sample_statistics_comparison_l317_317859


namespace sham_completes_task_l317_317251

theorem sham_completes_task (W : ℝ) (RahulSham_rate : ℝ) (Rahul_rate : ℝ) (Sham_rate : ℝ)
  (H1 : RahulSham_rate = W / 35) (H2 : Rahul_rate = W / 60) (H3 : Rahul_rate + Sham_rate = RahulSham_rate) : 
  Sham_rate = W / 84 :=
by
  have H4 : W / 60 + Sham_rate = W / 35 := H3
  have H5 : Sham_rate = W / 35 - W / 60 := by linarith
  have H6 : Sham_rate = (12 * W - 7 * W) / 420 := by { field_simp [W], linarith }
  have H7 : Sham_rate = 5 * W / 420 := by linarith
  have H8 : Sham_rate = W / 84 := by { field_simp [W], linarith }
  exact H8

end sham_completes_task_l317_317251


namespace total_projection_length_eq_l317_317954

-- Given conditions
variables (D E F J S T U : Type) [has_dist D] [has_dist E] [has_dist F] [has_dist J] [has_dist S] [has_dist T] [has_dist U]
variables (DE DF EF : ℝ) (JS JT JU : ℝ) 
variables (DG EH FI : line D E)

-- Suppose the lengths of the sides of triangle DEF
def DE := 4
def DF := 6
def EF := 5

-- DG, EH, and FI are medians intersecting at centroid J
def is_centroid (J : Type) [has_centroid DG J] [has_centroid EH J] [has_centroid FI J] : Prop :=
  centroid D E F J

-- S, T, U are the projections of J onto EF, DF, and DE
def is_projection (S T U : Type) (J : Type) [has_projection J EF S] [has_projection J DF T] [has_projection J DE U] : Prop :=
  projection J EF S ∧ projection J DF T ∧ projection J DE U

-- Prove that the total length of the projections is 4.07888
theorem total_projection_length_eq :
  DE = 4 ∧ DF = 6 ∧ EF = 5 ∧ is_centroid J ∧ is_projection S T U J → JS + JT + JU = 4.07888 :=
sorry

end total_projection_length_eq_l317_317954


namespace rational_number_addition_l317_317694

theorem rational_number_addition :
  (-206 : ℚ) + (401 + 3 / 4) + (-(204 + 2 / 3)) + (-(1 + 1 / 2)) = -10 - 5 / 12 :=
by
  sorry

end rational_number_addition_l317_317694


namespace min_value_expr_min_value_achieved_l317_317053

theorem min_value_expr (x : ℝ) (hx : x > 0) : 4*x + 1/x^4 ≥ 5 :=
by
  sorry

theorem min_value_achieved (x : ℝ) : x = 1 → 4*x + 1/x^4 = 5 :=
by
  sorry

end min_value_expr_min_value_achieved_l317_317053


namespace evaluate_expression_l317_317217

theorem evaluate_expression : 
  let x := 3 in
  let y := 1 in
  (1 / 3) ^ (x - y) = 1 / 9 := 
by 
  -- Proof omitted
  sorry

end evaluate_expression_l317_317217


namespace total_cost_is_63_l317_317779

-- Define the original price, markdown percentage, and sales tax percentage
def original_price : ℝ := 120
def markdown_percentage : ℝ := 0.50
def sales_tax_percentage : ℝ := 0.05

-- Calculate the reduced price
def reduced_price : ℝ := original_price * (1 - markdown_percentage)

-- Calculate the sales tax on the reduced price
def sales_tax : ℝ := reduced_price * sales_tax_percentage

-- Calculate the total cost
noncomputable def total_cost : ℝ := reduced_price + sales_tax

-- Theorem stating that the total cost of the aquarium is $63
theorem total_cost_is_63 : total_cost = 63 := by
  sorry

end total_cost_is_63_l317_317779


namespace three_digit_numbers_count_no_repeats_l317_317066

theorem three_digit_numbers_count_no_repeats : 
  (∃ S : Finset ℕ, S = {0, 1, 2, 3, 4, 5} ∧
  (∀ (a b c : ℕ), 
    (a ∈ S ∧ b ∈ S ∧ c ∈ S) ∧ 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
    a ≠ 0) → 
  { 100 }) :=
by
  use Finset.range 6
  sorry

end three_digit_numbers_count_no_repeats_l317_317066


namespace last_digit_of_prime_l317_317370

theorem last_digit_of_prime (n : ℕ) (h1 : 859433 = 214858 * 4 + 1) : (2 ^ 859433 - 1) % 10 = 1 := by
  sorry

end last_digit_of_prime_l317_317370


namespace volume_of_tetrahedron_OABC_is_zero_l317_317295

-- Define the conditions
def condition1 (a b : ℝ) : Prop := a^2 + b^2 = 25
def condition2 (b c : ℝ) : Prop := b^2 + c^2 = 144
def condition3 (a c : ℝ) : Prop := c^2 + a^2 = 169

-- Formulate the problem
theorem volume_of_tetrahedron_OABC_is_zero (a b c : ℝ) (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 a c) : 
  volume_of_tetrahedron_0 (((a, 0, 0) : ℝ×ℝ×ℝ),((0, b, 0) : ℝ×ℝ×ℝ),((0, 0, c) : ℝ×ℝ×ℝ)) 
 Proof
calc
   (1/6* a* b* c)
   (a=5, b=0, c= 12) 
 : 0 := sorry

-- noncomputable to aid Lean in handling real numbers
noncomputable def volume_of_tetrahedron_0 : ℝ×ℝ×ℝ
  calc 0 := sorry

end volume_of_tetrahedron_OABC_is_zero_l317_317295


namespace equal_distribution_l317_317930

theorem equal_distribution (total_cookies bags : ℕ) (h_total : total_cookies = 14) (h_bags : bags = 7) : total_cookies / bags = 2 := by
  sorry

end equal_distribution_l317_317930


namespace calculation_result_l317_317308

theorem calculation_result :
  let a := 0.0088
  let b := 4.5
  let c := 0.05
  let d := 0.1
  let e := 0.008
  (a * b) / (c * d * e) = 990 :=
by
  sorry

end calculation_result_l317_317308


namespace university_admissions_l317_317035

def students : Fin 5 := ⟦0, 1, 2, 3, 4⟧ -- Representing students A, B, and others via indices
def universities : Fin 3 := ⟦0, 1, 2⟧ -- Representing universities via indices (0, 1, 2)

theorem university_admissions :
  (number_of_distributions : 
    ∃ f : Fin 5 → Fin 3,
      (∀ u : Fin 3, ∃ s : Finset (Fin 5), ∃ x : Fin 5, x ∈ s ∧ f x = u)
  ) = 150 := 
sorry

end university_admissions_l317_317035


namespace no_integer_roots_quadratic_l317_317898

theorem no_integer_roots_quadratic (a b : ℤ) : 
  ∀ u : ℤ, ¬(u^2 + 3*a*u + 3*(2 - b^2) = 0) := 
by
  sorry

end no_integer_roots_quadratic_l317_317898


namespace problem_statement_b_problem_statement_c_l317_317023

def clubsuit (x y : ℝ) : ℝ := |x - y + 3|

theorem problem_statement_b :
  ∃ x y : ℝ, 3 * (clubsuit x y) ≠ clubsuit (3 * x + 3) (3 * y + 3) := by
  sorry

theorem problem_statement_c :
  ∃ x : ℝ, clubsuit x (-3) ≠ x := by
  sorry

end problem_statement_b_problem_statement_c_l317_317023


namespace sample_stat_properties_l317_317865

/-- Given a set of sample data, and another set obtained by adding a non-zero constant to each element of the original set,
prove some properties about their statistical measures. -/
theorem sample_stat_properties (n : ℕ) (c : ℝ) (h₀ : c ≠ 0) 
  (x : ℕ → ℝ) :
  let y := λ i, x i + c 
  in (∑ i in finset.range n, x i) / n ≠ (∑ i in finset.range n, y i) / n ∧ 
     (∃ (median_x : ℝ) (median_y : ℝ), median_y = median_x + c ∧ median_y ≠ median_x) ∧
     (stddev (finset.range n).map x = stddev (finset.range n).map y) ∧
     (range (finset.range n).map x = range (finset.range n).map y) := 
by
  sorry

-- Helper functions for median, stddev, and range could be defined if missing from Mathlib.

end sample_stat_properties_l317_317865


namespace probability_same_number_on_four_dice_l317_317675

open Probability

/-- The probability that the same number will be facing up on each of four eight-sided dice that are tossed simultaneously. -/
def prob_same_number_each_four_dice : ℚ := 1 / 512

/-- Given that the dice are eight-sided, four dice are tossed,
and the results on the four dice are independent of each other.
Prove that the probability that the same number will be facing up on each of these four dice is 1/512. -/
theorem probability_same_number_on_four_dice (n : ℕ) (s : Fin n → Fin 8) :
  n = 4 → (∀ i, ∃ k, s i = k) → prob_same_number_each_four_dice = 1 / 512 :=
by intros; simp; sorry

end probability_same_number_on_four_dice_l317_317675


namespace solution_to_power_tower_l317_317623

noncomputable def infinite_power_tower (x : ℝ) : ℝ := sorry

theorem solution_to_power_tower : ∃ x : ℝ, infinite_power_tower x = 4 ∧ x = Real.sqrt 2 := sorry

end solution_to_power_tower_l317_317623


namespace area_of_triangle_ABC_l317_317532

-- Definitions and conditions
variables (A B C D E : Type)
variables [IsTriangle A B C]
variables [IsMedian B D]
variables [OnExtension E A C]
variables [Distance E B 12]
variables [GeometricProgression (tan (angle C B E)) (tan (angle D B E)) (tan (angle A B E))]
variables [ArithmeticProgression (cot (angle D B E)) (cot (angle C B E)) (cot (angle D B C))]

-- Goal
theorem area_of_triangle_ABC : area (triangle A B C) = 12 := 
sorry

end area_of_triangle_ABC_l317_317532


namespace num_real_values_of_k_for_magnitude_l317_317058

theorem num_real_values_of_k_for_magnitude (k : ℝ) : 
  (|1 - k * Complex.I| = 1.5) ↔ (k = Real.sqrt 1.25 ∨ k = -Real.sqrt 1.25) :=
by 
  sorry

end num_real_values_of_k_for_magnitude_l317_317058


namespace area_CDE_l317_317244

variables (ABC : Type) [triangle ABC]
variables (A B C D E F : ABC)
variables (AD AC : Line ABC) (BD BE : Line ABC) (AE : Line ABC)

-- Areas of triangles
def area (t : Triangle ABC) : ℝ := sorry

-- Given conditions
axiom h1 : area (triangle A D F) = 1 / 2
axiom h2 : area (triangle A B F) = 1
axiom h3 : area (triangle B E F) = 1 / 4

-- Target statement: Area of CDE
theorem area_CDE : area (triangle C D E) = 15 / 56 := sorry

end area_CDE_l317_317244


namespace smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l317_317679

theorem smallest_positive_perfect_square_divisible_by_5_and_6_is_900 :
  ∃ n : ℕ, 0 < n ∧ (n ^ 2) % 5 = 0 ∧ (n ^ 2) % 6 = 0 ∧ (n ^ 2 = 900) := by
  sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l317_317679


namespace radius_of_inscribed_circle_l317_317269

theorem radius_of_inscribed_circle (S : ℝ) (h_height_leg : ∀ (a b c : ℝ), a + b = 2 * c ∧ ((a + b) / 4) = c / 2 ∧ (S = (1 / 2) * (a + b) * (c / 2))) : 
  ∃ R : ℝ, R = (Real.sqrt(2 * S) / 4) :=
by
  sorry

end radius_of_inscribed_circle_l317_317269


namespace ratio_area_HIJK_ABC_l317_317660

-- Define the context and conditions
variables {A B C D E F G H I J K : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (triangle_ABC : Triangle A B C) (equilateral_ABC : triangle_ABC.isEquilateral)
variables [ParallelLines DE BC] [ParallelLines FG BC] [ParallelLines HI BC] [ParallelLines JK BC]
variables (AD_eq_DF : AD = DF) (DF_eq_FH : DF = FH) (FH_eq_HJ : FH = HJ)

-- Define the areas of triangles and trapezoids
variables (area_ABC : Real)
variables (area_ADJ : Real)
variables (area_HIJK : Real)
variables (ratio_ADJ_ABC : area_ADJ / area_ABC = (4/5)^2)

-- State the theorem to prove
theorem ratio_area_HIJK_ABC 
  (h_area_ADJ_ABC : area_ADJ / area_ABC = 16/25)
  (h_area_ADJ_HIJK_ABC : area_ADJ + area_HIJK = area_ABC) :
  area_HIJK / area_ABC = 9/25 :=
sorry

end ratio_area_HIJK_ABC_l317_317660


namespace no_perfect_square_in_sequence_l317_317135

def sequence_term (i : ℕ) : ℕ :=
  let baseDigits : List ℕ := [2, 0, 1, 4, 2, 0, 1, 5]
  let termDigits := baseDigits.foldr (fun d acc => acc + d * 10 ^ (baseDigits.length - acc.length - 1)) 0
  termDigits + 10 ^ (i + 5)

theorem no_perfect_square_in_sequence : ¬ ∃ i : ℕ, ∃ k : ℕ, k * k = sequence_term i := 
sorry

end no_perfect_square_in_sequence_l317_317135


namespace find_K_l317_317669

theorem find_K 
  (Z K : ℤ) 
  (hZ_range : 1000 < Z ∧ Z < 2000)
  (hZ_eq : Z = K^4)
  (hK_pos : K > 0) :
  K = 6 :=
by {
  sorry -- Proof to be filled in
}

end find_K_l317_317669


namespace original_savings_l317_317985

-- Define the problem conditions
def savings_fraction_on_tv (total_savings : ℝ) : ℝ := 1/4 * total_savings
def tv_cost : ℝ := 450

-- Define the theorem statement
theorem original_savings (S : ℝ) (h : savings_fraction_on_tv S = tv_cost) : S = 1800 :=
by
  sorry

end original_savings_l317_317985


namespace pink_roses_at_Mrs_Dawson_l317_317773

def rows : ℕ := 10
def roses_per_row : ℕ := 20
def red_ratio : ℚ := 1 / 2
def white_ratio : ℚ := 3 / 5

theorem pink_roses_at_Mrs_Dawson (R : ℕ) (T : ℕ) (red_ratio : ℚ) (white_ratio : ℚ) (hR : R = 10) (hT : T = 20) 
  (h_red_ratio : red_ratio = 1 / 2) (h_white_ratio : white_ratio = 3 / 5) : 
  (T - (red_ratio * T).toNat - ((white_ratio * (T - (red_ratio * T).toNat)).toNat)) * R = 40 := 
  sorry

end pink_roses_at_Mrs_Dawson_l317_317773


namespace find_parameters_l317_317627

noncomputable def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + a^2

theorem find_parameters 
  (a b : ℝ) 
  (h_deriv : 3 - 2 * a + b = 0) 
  (h_func : -1 - a - b + a^2 = 8) : 
  a = 2 ∧ b = -7 :=
begin
  sorry
end

end find_parameters_l317_317627


namespace periodicity_of_m_arith_fibonacci_l317_317712

def m_arith_fibonacci (m : ℕ) (v : ℕ → ℕ) : Prop :=
∀ n : ℕ, v (n + 2) = (v n + v (n + 1)) % m

theorem periodicity_of_m_arith_fibonacci (m : ℕ) (v : ℕ → ℕ) 
  (hv : m_arith_fibonacci m v) : 
  ∃ r : ℕ, r ≤ m^2 ∧ ∀ n : ℕ, v (n + r) = v n := 
by
  sorry

end periodicity_of_m_arith_fibonacci_l317_317712


namespace no_perfect_squares_in_sequence_l317_317137

theorem no_perfect_squares_in_sequence : 
  ∀ N ∈ ({20142015, 201402015, 2014002015, 20140002015, 201400002015} : set ℕ), 
  ¬ (∃ k : ℕ, N = k^2) := 
by
  sorry

end no_perfect_squares_in_sequence_l317_317137


namespace isosceles_triangle_angle_B_l317_317709

theorem isosceles_triangle_angle_B {A B C D : Type}
  [Point A][Point B][Point C][Point D]
  (isosceles_ABC : is_isosceles_triangle A B C)
  (base_AC : base A C)
  (bisector_CD : is_bisector C D)
  (angle_ADC : angle A D C = 150) :
  angle B = 140 :=
by
  sorry

end isosceles_triangle_angle_B_l317_317709


namespace difference_of_squares_153_147_l317_317796

theorem difference_of_squares_153_147 : (153^2 - 147^2) = 1800 := by
  sorry

end difference_of_squares_153_147_l317_317796


namespace mode_of_dataSet_l317_317633

def dataSet : List ℕ := [5, 4, 4, 3, 6, 2]

def mode (l : List ℕ) : ℕ :=
  (l.foldl (λ m x, m.insert x (m.findD x 0 + 1)) ∅).maxByOption (λ k v, v).iget.key

theorem mode_of_dataSet : mode dataSet = 4 :=
by
  sorry

end mode_of_dataSet_l317_317633


namespace joan_missed_games_l317_317960

theorem joan_missed_games :
  ∀ (total_games attended_games missed_games : ℕ),
  total_games = 864 →
  attended_games = 395 →
  missed_games = total_games - attended_games →
  missed_games = 469 :=
by
  intros total_games attended_games missed_games H1 H2 H3
  rw [H1, H2] at H3
  exact H3

end joan_missed_games_l317_317960


namespace simplify_expression_l317_317604

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem simplify_expression :
  cube_root (8 + 27) * cube_root (8 + cube_root 27) = cube_root 385 :=
by
  sorry

end simplify_expression_l317_317604


namespace rearrange_numbers_l317_317199

theorem rearrange_numbers (n : ℕ) (h : n = 25 ∨ n = 1000) :
  ∃ (l : List ℕ), l.nodup ∧ l.perm (List.range (n + 1))
  ∧ (∀ (i : ℕ), i < n → ((l.get? i.succ = some (l.get! i + 3)) ∨ (l.get? i.succ = some (l.get! i + 5)) ∨ (l.get? i.succ = some (l.get! i - 3)) ∨ (l.get? i.succ = some (l.get! i - 5)))) :=
by
  sorry

end rearrange_numbers_l317_317199


namespace value_of_expression_l317_317490

theorem value_of_expression (x : ℤ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  exact rfl

end value_of_expression_l317_317490


namespace initial_winning_margin_percentage_l317_317173

-- Define the conditions given in the problem
def total_votes : ℕ := 15000
def vote_change : ℕ := 3000
def margin_percentage : ℝ := 0.2

-- The initial and modified votes for winner and loser
def initial_w_votes (W L : ℕ) : Prop := W + L = total_votes
def modified_l_votes (L : ℕ) : Prop := L + vote_change = 0.6 * total_votes
def modified_w_votes (W : ℕ) : Prop := W - vote_change = 0.4 * total_votes

-- The initial margin percentage definition
def initial_margin_percentage (W L : ℕ) : ℝ := (W - L : ℝ) / total_votes * 100

-- The main theorem
theorem initial_winning_margin_percentage (W L : ℕ)
  (h1 : initial_w_votes W L)
  (h2 : modified_l_votes L)
  (h3 : modified_w_votes W) :
  initial_margin_percentage W L = 20 :=
by
  sorry

end initial_winning_margin_percentage_l317_317173


namespace mean_visits_between_200_and_300_l317_317781

def monday_visits := 300
def tuesday_visits := 400
def wednesday_visits := 300
def thursday_visits := 200
def friday_visits := 200

def total_visits := monday_visits + tuesday_visits + wednesday_visits + thursday_visits + friday_visits
def number_of_days := 5
def mean_visits_per_day := total_visits / number_of_days

theorem mean_visits_between_200_and_300 : 200 ≤ mean_visits_per_day ∧ mean_visits_per_day ≤ 300 :=
by sorry

end mean_visits_between_200_and_300_l317_317781


namespace sample_statistics_comparison_l317_317860

variable {α : Type*} [LinearOrder α] [Field α]

def sample_data_transformed (x : list α) (c : α) : list α :=
  x.map (λ xi, xi + c)

def sample_mean (x : list α) : α :=
  x.sum / (x.length : α)

def sample_median (x : list α) : α :=
  if x.length % 2 = 1 then
    x.sort.nth_le (x.length / 2) (by sorry)
  else
    (x.sort.nth_le (x.length / 2 - 1) (by sorry) + x.sort.nth_le (x.length / 2) (by sorry)) / 2

def sample_variance (x : list α) : α :=
  let m := sample_mean x
  in (x.map (λ xi, (xi - m)^2)).sum / (x.length : α)

def sample_standard_deviation (x : list α) : α :=
  (sample_variance x).sqrt

def sample_range (x : list α) : α :=
  x.maximum (by sorry) - x.minimum (by sorry)

theorem sample_statistics_comparison (x : list α) (c : α) (hc : c ≠ 0) :
  (sample_mean (sample_data_transformed x c) ≠ sample_mean x) ∧
  (sample_median (sample_data_transformed x c) ≠ sample_median x) ∧
  (sample_standard_deviation (sample_data_transformed x c) = sample_standard_deviation x) ∧
  (sample_range (sample_data_transformed x c) = sample_range x) :=
  sorry

end sample_statistics_comparison_l317_317860


namespace cuddly_numbers_count_eq_one_l317_317761

def is_cuddly (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 10 * a + b = a + b^2

theorem cuddly_numbers_count_eq_one :
  (finset.filter is_cuddly (finset.Icc 10 99)).card = 1 :=
by
  sorry

end cuddly_numbers_count_eq_one_l317_317761


namespace smallest_positive_perfect_square_divisible_by_5_and_6_l317_317686

theorem smallest_positive_perfect_square_divisible_by_5_and_6 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m * m) ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (∀ k : ℕ, (∃ p : ℕ, k = p * p) ∧ k % 5 = 0 ∧ k % 6 = 0 → n ≤ k) := 
sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_l317_317686


namespace range_of_x1_x2_minus_a_l317_317114

noncomputable def f (x a : ℝ) := 2 * Real.sin (2 * x + π / 6) + a - 1

theorem range_of_x1_x2_minus_a (x1 x2 a : ℝ) 
  (hx1 : 0 ≤ x1 ∧ x1 ≤ π/2) 
  (hx2 : 0 ≤ x2 ∧ x2 ≤ π/2) 
  (hzeros : f x1 a = 0 ∧ f x2 a = 0 ∧ x1 ≠ x2) :
  x1 + x2 - a ∈ Set.Ico (π / 3) (π / 3 + 1) :=
sorry

end range_of_x1_x2_minus_a_l317_317114


namespace correct_statements_l317_317375

variable (A B C D : Prop)

-- Corresponding statements
def statement1 : Prop := ∀ r : ℝ, ∃ p : ℝ, r = p
def statement2 : Prop := ∀ x : ℝ, (sqrt x = x) ↔ (x = 0 ∨ x = 1)
def statement3 : Prop := sqrt 6 = real.sqrt 6
def statement4 : Prop := ¬ ∃ x : ℝ, 1 < x ∧ x < 3 ∧ irrational x ∧ x ≠ sqrt 2 ∧ x ≠ sqrt 3 ∧ x ≠ sqrt 5 ∧ x ≠ sqrt 7

-- Correct conclusion
def correct_conclusion : Prop := (statement1 ∧ statement3) ∧ ¬ statement2 ∧ ¬ statement4

theorem correct_statements (A : statement1) (B : ¬ statement2) (C : statement3) (D : ¬ statement4) :
  correct_conclusion :=
by {
  unfold correct_conclusion,
  apply and.intro,
  { apply and.intro,
    { exact A, },
    { exact C, }, },
  apply and.intro,
  { exact B, },
  { exact D, },
}

end correct_statements_l317_317375


namespace partial_fraction_l317_317555

noncomputable def roots := 
  {p q r : ℝ // is_root (X^3 - 24*X^2 + 98*X - 75) p ∧ 
                  is_root (X^3 - 24*X^2 + 98*X - 75) q ∧ 
                  is_root (X^3 - 24*X^2 + 98*X - 75) r}

noncomputable def A (p q r s : ℝ) := 
  (5 / ((s - p) * (s - q) * (s - r)))

noncomputable def B (p q r s : ℝ) := 
  (5 / ((s - p) * (s - q) * (s - r)))

noncomputable def C (p q r s : ℝ) :=
  (5 / ((s - p) * (s - q) * (s - r)))

theorem partial_fraction (p q r A B C : ℝ) (hpqrs : roots p q r) 
  (hA : A (p q r s) = (dfrac{A}{s-p})
  (hB : B (p q r s) = (dfrac{B}{s-q})
  (hC : C (p q r s) = (dfrac{C}{s-r})
  : 1 / A + 1 / B + 1 / C = 256 :=
sorry

end partial_fraction_l317_317555


namespace chess_group_games_l317_317653

theorem chess_group_games (n : ℕ) (h_n : n = 10) : 
  (∃ (g : ℕ), g = (n * (n - 1)) / 2) → g = 45 :=
by
  intros h
  cases h with games games_def
  rw [games_def, h_n]
  norm_num
  sorry

end chess_group_games_l317_317653


namespace probability_of_boys_and_girls_l317_317723

def total_outcomes := Nat.choose 7 4
def only_boys_outcomes := Nat.choose 4 4
def both_boys_and_girls_outcomes := total_outcomes - only_boys_outcomes
def probability := both_boys_and_girls_outcomes / total_outcomes

theorem probability_of_boys_and_girls :
  probability = 34 / 35 :=
by
  sorry

end probability_of_boys_and_girls_l317_317723


namespace no_such_function_l317_317206

theorem no_such_function : ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2 := 
sorry

end no_such_function_l317_317206


namespace number_of_points_l317_317092

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {1, 5}

theorem number_of_points : (M.product N).card + (N.product M).card - 1 = 11 := by
sorry

end number_of_points_l317_317092


namespace product_of_solutions_l317_317677

theorem product_of_solutions (a b c x : ℝ) (p : a * x^2 + b * x + c = 0) :
    (∃ α β : ℝ, (α * β = c / a) ∧ (a * α^2 + b * α + c = 0) ∧ (a * β^2 + b * β + c = 0)) :=
begin
  sorry,
end

end product_of_solutions_l317_317677


namespace number_of_ordered_pairs_reciprocal_l317_317466

theorem number_of_ordered_pairs_reciprocal
  (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  (1 / m.toReal + 1 / n.toReal = 1 / 3) → 
  (∃ (m n : ℕ), (m > 0 ∧ n > 0) ∧ (1 / m.toReal + 1 / n.toReal = 1 / 3) ∧
    ((m = 4 ∧ n = 12) ∨ (m = 6 ∧ n = 6) ∨ (m = 12 ∧ n = 4)).length = 3) :=
sorry

end number_of_ordered_pairs_reciprocal_l317_317466


namespace problem_statement_l317_317188

noncomputable def first_order_lattice_points_function (f : ℤ → ℤ) : Prop :=
  {p : ℤ × ℤ // f p.1 = p.2}.card = 1

theorem problem_statement :
  first_order_lattice_points_function (λ x, Int.sin x) ∧
  ¬ first_order_lattice_points_function (λ x, Int.cos (x + (Int.pi / 6))) ∧
  ¬ first_order_lattice_points_function (λ x, Int.log10 x) ∧
  ¬ first_order_lattice_points_function (λ x, x^2) :=
by
  sorry

end problem_statement_l317_317188


namespace compute_diff_squares_l317_317793

theorem compute_diff_squares (a b : ℤ) (ha : a = 153) (hb : b = 147) :
  a ^ 2 - b ^ 2 = 1800 :=
by
  rw [ha, hb]
  sorry

end compute_diff_squares_l317_317793


namespace negation_of_proposition_l317_317280

theorem negation_of_proposition :
  (¬ ∃ α β : ℝ, (sin (α + β) * sin (α - β) ≥ sin α ^ 2 - sin β ^ 2)) ↔ 
  (∀ α β : ℝ, sin (α + β) * sin (α - β) < sin α ^ 2 - sin β ^ 2) := by 
  sorry

end negation_of_proposition_l317_317280


namespace area_of_triangle_ABC_l317_317531

-- Definitions and conditions
variables (A B C D E : Type)
variables [IsTriangle A B C]
variables [IsMedian B D]
variables [OnExtension E A C]
variables [Distance E B 12]
variables [GeometricProgression (tan (angle C B E)) (tan (angle D B E)) (tan (angle A B E))]
variables [ArithmeticProgression (cot (angle D B E)) (cot (angle C B E)) (cot (angle D B C))]

-- Goal
theorem area_of_triangle_ABC : area (triangle A B C) = 12 := 
sorry

end area_of_triangle_ABC_l317_317531


namespace angle_at_4oclock_l317_317783

theorem angle_at_4oclock : 
  -- Conditions
  (hours : ℕ) (hours = 12) → 
  (total_degrees : ℕ) (total_degrees = 360) → 
  (hr_hand_position : ℕ) (hr_hand_position = 4) → 
  (min_hand_position : ℕ) (min_hand_position = 12) →
  -- Prove the angle
  let degrees_per_hour := (total_degrees : ℕ) / hours in
  let angle := (hr_hand_position * degrees_per_hour) % total_degrees in
  angle = 120 :=
by {
  intros hours h_hours total_degrees h_degrees hr_hand_position h_hr min_hand_position h_min,
  rw [h_hours, h_degrees, h_hr, h_min],
  rw [Nat.div_eq_of_eq_mul_right (by norm_num : 0 < 12) (by norm_num : 360 = 12 * 30)],
  have h_degrees_per_hour : degrees_per_hour = 30 := 
    by { rw [h_hours, h_degrees], norm_num },
  rw [h_degrees_per_hour],
  have h_angle : angle = (4 * 30) % 360 := rfl,
  rw [mul_comm 4 30, Nat.mul_mod 120 360],
  norm_num,
}

end angle_at_4oclock_l317_317783


namespace orange_juice_fraction_in_mixture_l317_317300

theorem orange_juice_fraction_in_mixture :
  let capacity1 := 800
  let capacity2 := 700
  let fraction1 := (1 : ℚ) / 4
  let fraction2 := (3 : ℚ) / 7
  let orange_juice1 := capacity1 * fraction1
  let orange_juice2 := capacity2 * fraction2
  let total_orange_juice := orange_juice1 + orange_juice2
  let total_volume := capacity1 + capacity2
  let fraction := total_orange_juice / total_volume
  fraction = (1 : ℚ) / 3 := by
  sorry

end orange_juice_fraction_in_mixture_l317_317300


namespace exists_person_knows_everyone_l317_317166

-- Definitions of conditions
def company (n : ℕ) := fin (2 * n + 1)
def is_acquainted (p q : company n) : Prop := sorry  -- Replace this with the actual acquaintance relation

-- The main theorem statement
theorem exists_person_knows_everyone {n : ℕ} 
  (h : ∀ (S : finset (company n)), S.card = n → ∃ (p : company n), ∀ q ∈ S, is_acquainted p q) :
  ∃ p : company n, ∀ q : company n, is_acquainted p q :=
sorry

end exists_person_knows_everyone_l317_317166


namespace chordal_graph_cliques_connected_components_eq_l317_317803

-- Define the involved terms and properties
variables (G : Type) [Graph G] [Chordal G]
noncomputable def f (G : Type) [Graph G] : ℤ :=
  ∑ i in finset.range (max_clique_size G + 1), if odd i then cliques_size G i else -cliques_size G i

-- Define c_i as the number of cliques with i vertices
def cliques_size (G : Type) [Graph G] (i : ℕ) : ℕ := 
  finset.card { S : finset G | is_clique S ∧ S.card = i }

-- Define k(G) as the number of connected components
def k (G : Type) [Graph G] : ℕ := 
  connected_components G

-- State the theorem
theorem chordal_graph_cliques_connected_components_eq (G : Type) [Graph G] [Chordal G] :
  f G = k G :=
sorry

end chordal_graph_cliques_connected_components_eq_l317_317803


namespace minimum_value_of_f_on_interval_l317_317629

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := sin (2 * x + φ)

theorem minimum_value_of_f_on_interval : 
  ∀ (φ : ℝ), 
    abs φ < π/2 →
    φ = -π/3 →
    ∀ (x : ℝ), 0 ≤ x ∧ x ≤ π/2 →
    has_inf ((f x φ) ' interval 0 (π/2)) (-sqrt 3 / 2) := 
  by
    intros φ hφ1 hφ2 x hx
    -- Proof would go here
    sorry


end minimum_value_of_f_on_interval_l317_317629


namespace earnings_per_day_correct_l317_317643

-- Given conditions
variable (total_earned : ℕ) (days : ℕ) (earnings_per_day : ℕ)

-- Specify the given values from the conditions
def given_conditions : Prop :=
  total_earned = 165 ∧ days = 5 ∧ total_earned = days * earnings_per_day

-- Statement of the problem: proving the earnings per day
theorem earnings_per_day_correct (h : given_conditions total_earned days earnings_per_day) : 
  earnings_per_day = 33 :=
by
  sorry

end earnings_per_day_correct_l317_317643


namespace neg_p_l317_317124

variable (x : ℝ)

def p : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

theorem neg_p : ¬p ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by
  sorry

end neg_p_l317_317124


namespace inverse_h_l317_317827

def h (x : ℝ) : ℝ := 7 + 3 * x

def j (x : ℝ) : ℝ := (x - 7) / 3

theorem inverse_h : ∀ x, h (j x) = x ∧ j (h x) = x := by
  sorry

end inverse_h_l317_317827


namespace simplify_expression_l317_317606

-- Define the original expression and the simplified version
def original_expr (x y : ℤ) : ℤ := 7 * x + 3 - 2 * x + 15 + y
def simplified_expr (x y : ℤ) : ℤ := 5 * x + y + 18

-- The equivalence to be proved
theorem simplify_expression (x y : ℤ) : original_expr x y = simplified_expr x y :=
by sorry

end simplify_expression_l317_317606


namespace proof_solution_l317_317981

noncomputable def proof_problem (x y z : ℝ) : Prop :=
  x > 4 ∧ y > 4 ∧ z > 4 ∧
  ( (x + 3)^2 / (y + z - 3) + 
    (y + 5)^2 / (z + x - 5) + 
    (z + 7)^2 / (x + y - 7) = 45 ) ∧
  (x, y, z) = (11, 10, 9)

theorem proof_solution : ∃ x y z : ℝ, proof_problem x y z :=
by {use 11, use 10, use 9, sorry}

end proof_solution_l317_317981


namespace kerosene_cost_l317_317329

/-- In a market, a dozen eggs cost as much as a pound of rice, and a half-liter of kerosene 
costs as much as 8 eggs. If the cost of each pound of rice is $0.33, then a liter of kerosene costs 44 cents. --/
theorem kerosene_cost : 
  let egg_cost := 0.33 / 12
  let rice_cost := 0.33
  let half_liter_kerosene_cost := 8 * egg_cost
  rice_cost = 0.33 → 1 * ((2 * half_liter_kerosene_cost) * 100) = 44 := 
by
  intros egg_cost rice_cost half_liter_kerosene_cost h_rice_cost
  sorry

end kerosene_cost_l317_317329


namespace problem_I_problem_II_l317_317164

def events := {1, 2, 3, 4}

def event_A (draws : Finset ℕ) : Prop :=
  draws.sum > 7

def event_B : Finset (ℕ × ℕ) :=
  { (a, b) | a = 3 ∨ b = 3 }

noncomputable def probability (s : Finset ℕ) (event : Finset ℕ → Prop) : ℚ :=
  (s.filter event).card / s.card

theorem problem_I : probability (events.choose 3) event_A = 1 / 2 :=
by sorry

theorem problem_II : probability (Finset.cartesian_product events events) event_B.card = 7 / 16 :=
by sorry

end problem_I_problem_II_l317_317164


namespace simplify_cube_root_expression_l317_317594

-- statement of the problem
theorem simplify_cube_root_expression :
  ∛(8 + 27) * ∛(8 + ∛(27)) = ∛(385) :=
  sorry

end simplify_cube_root_expression_l317_317594


namespace complement_intersection_l317_317907

open Set

noncomputable def I := {1, 2, 3, 4, 5, 6}
noncomputable def A := {1, 2, 3, 4}
noncomputable def B := {3, 4, 5, 6}

theorem complement_intersection :
  ∁ (A ∩ B : Set ℕ) = {1, 2, 5, 6} :=
by
  sorry

end complement_intersection_l317_317907


namespace triangle_ABC_area_is_six_l317_317533

noncomputable def area_of_triangle_ABC : ℝ :=
  let AB : ℝ := 6
  let BD : ℝ := 6
  let AC : ℝ := 12
  let a : ℝ := 2
  let k : ℝ := 3
  let CBE_α := π / 4
  let DBC_β := π / 4
  let area := 1 / 2 * 6 * 2 in
  if (tan CBE_α = 1 ∧ (tan (CBE_α + DBC_β) = (1 + a / BD) / (1 - a / BD))
    ∧ (k = 3) ∧ (BD = 6) ∧ (a = 2))
  then area else 0

theorem triangle_ABC_area_is_six :
    let α := area_of_triangle_ABC
    in α = 6 :=
by
  sorry

end triangle_ABC_area_is_six_l317_317533


namespace mark_baking_time_l317_317986

def total_baking_time (
  prepare_time : ℕ := 30,
  first_rise_time : ℕ := 120, 
  fold_time : ℕ := 20,
  second_rise_time : ℕ := 120,
  knead_time : ℕ := 10,
  rest_time : ℕ := 30,
  bake_time : ℕ := 30,
  cool_time : ℕ := 15
) : ℕ :=
  prepare_time + first_rise_time + fold_time + second_rise_time + knead_time + rest_time + bake_time + cool_time

theorem mark_baking_time :
  total_baking_time = 375 :=
by
  sorry

end mark_baking_time_l317_317986


namespace square_difference_l317_317789

theorem square_difference :
  153^2 - 147^2 = 1800 :=
by
  sorry

end square_difference_l317_317789


namespace midpoint_sum_coords_l317_317548

theorem midpoint_sum_coords (x y : ℝ) :
  let A := (2, 7 : ℝ × ℝ)
  let B := (x, y : ℝ × ℝ)
  let C := (4, 3 : ℝ × ℝ)
  (C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →
  x + y = 5 :=
by
  sorry

end midpoint_sum_coords_l317_317548


namespace project_completion_time_l317_317696

def work_rate_A : ℚ := 1 / 20
def work_rate_B : ℚ := 1 / 30
def total_project_days (x : ℚ) : Prop := (work_rate_A * (x - 10) + work_rate_B * x = 1)

theorem project_completion_time (x : ℚ) (h : total_project_days x) : x = 13 := 
sorry

end project_completion_time_l317_317696


namespace range_f_l317_317032

def f (x : ℝ) : ℝ := x - sqrt (1 - 2 * x)

theorem range_f : set.range f = set.Iic (1 / 2) := by
  sorry

end range_f_l317_317032


namespace train_passes_bridge_in_time_l317_317368

def train_length := 360 -- in meters
def bridge_length := 140 -- in meters
def train_speed_kmh := 75 -- in km per hour

def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

def total_distance (train_length bridge_length : ℝ) : ℝ :=
  train_length + bridge_length

def time_to_pass (distance speed : ℝ) : ℝ :=
  distance / speed

theorem train_passes_bridge_in_time :
  (time_to_pass (total_distance train_length bridge_length) (kmh_to_ms train_speed_kmh)) ≈ 24 :=
by
  sorry

end train_passes_bridge_in_time_l317_317368


namespace integral_sin_cos_sq_find_complex_numbers_l317_317784

-- Part 1: Integral Calculation
theorem integral_sin_cos_sq : 
  ∫ x in 0 .. Real.pi / 2, (Real.sin (x / 2) + Real.cos (x / 2)) ^ 2 = Real.pi / 2 + 1 :=
by
  sorry

-- Part 2: Complex Number Equation
theorem find_complex_numbers (z : ℂ) : 
  z * comjugate(z) - complex.I * (conjugate(3 * z)) = 1 - (conjugate(3 * complex.I)) ↔ z = -1 ∨ z = -1 + 3 * complex.I :=
by
  sorry

end integral_sin_cos_sq_find_complex_numbers_l317_317784


namespace geometric_locus_of_equilateral_triangle_centers_l317_317051

noncomputable theory
open_locale classical
open_locale big_operators

universe u

variables {α : Type u} [linear_ordered_field α] [char_zero α]

/-- 
The geometric locus of the centers of equilateral triangles circumscribed around a given triangle ABC 
is the circumcircle of the equilateral triangle formed by the centers of equilateral triangles on 
the sides of ∆ABC.
-/
theorem geometric_locus_of_equilateral_triangle_centers (A B C : euclidean_space α (fin 2)) :
  ∃ (O : euclidean_space α (fin 2)),
  is_circumcenter O 
    (mk_triangle 
      (center_of_equilateral_triangle (mk_triangle A B C))
      (center_of_equilateral_triangle (mk_triangle B C A))
      (center_of_equilateral_triangle (mk_triangle C A B))) := sorry

end geometric_locus_of_equilateral_triangle_centers_l317_317051


namespace max_det_sqrt_377_l317_317549

noncomputable def uvec : ℝ^3 := sorry -- Assume this is a unit vector.

def v : ℝ^3 := ⟨3, 2, -4⟩
def w3 : ℝ^3 := ⟨2, -3, 0⟩

def maxDet (u : ℝ^3) : ℝ :=
  let cross := λ (a b : ℝ^3), (a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x) in
  let v_cross_w3 := ⟨cross v w3⟩ in
  (u.x * v_cross_w3.1 + u.y * v_cross_w3.2 + u.z * v_cross_w3.3)

theorem max_det_sqrt_377 (u : ℝ^3) (hu : ∥u∥ = 1) : ∃ u : ℝ^3, maxDet u = Real.sqrt 377 := by
  sorry

end max_det_sqrt_377_l317_317549


namespace maximum_cos_sum_l317_317373

theorem maximum_cos_sum (A B C : ℝ) (hA : A = 60) (hABC : A + B + C = 180) :
  ∃ x, x = cos A + cos B * cos C ∧ 
  (∀ y, y = cos A + cos B * cos C → y ≤ 5 / 4) :=
sorry

end maximum_cos_sum_l317_317373


namespace no_power_of_2_ends_with_four_identical_digits_l317_317249

theorem no_power_of_2_ends_with_four_identical_digits :
  ∀ n : ℕ, let last_four_digits := (2^n % 10000)
  in ¬(last_four_digits = 2222 ∨ last_four_digits = 4444 ∨ last_four_digits = 6666 ∨ last_four_digits = 8888) :=
by
  sorry

end no_power_of_2_ends_with_four_identical_digits_l317_317249


namespace double_inequality_solution_l317_317608

open Set

theorem double_inequality_solution (x : ℝ) :
  -1 < (x^2 - 16 * x + 24) / (x^2 - 4 * x + 8) ∧
  (x^2 - 16 * x + 24) / (x^2 - 4 * x + 8) < 1 ↔
  x ∈ Ioo (3 / 2) 4 ∪ Ioi 8 :=
by
  sorry

end double_inequality_solution_l317_317608


namespace find_b_vals_l317_317847

theorem find_b_vals (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) (q : ℝ) 
  (h_q_pos : 0 < q) (h_q_lt_1 : q < 1)
  (h_cond1 : ∀ k, 1 ≤ k ∧ k ≤ n → a k < b k)
  (h_cond2 : ∀ k, 1 ≤ k ∧ k < n → q < b (k + 1) / b k ∧ b (k + 1) / b k < 1 / q)
  (h_cond3 : (∑ k in finset.range n, b k) < ((1 + q) / (1 - q)) * (∑ k in finset.range n, a k)) :
  ∃ b : ℕ → ℝ, 
    (∀ k, 1 ≤ k ∧ k ≤ n → a k < b k) ∧ 
    (∀ k, 1 ≤ k ∧ k < n → q < b (k + 1) / b k ∧ b (k + 1) / b k < 1 / q) ∧ 
    (∑ k in finset.range n, b k < ((1 + q) / (1 - q)) * (∑ k in finset.range n, a k)) :=
begin
  -- proof will be provided here
  sorry
end

end find_b_vals_l317_317847


namespace angle_x_l317_317946

-- Conditions
variable (ABC BAC CDE DCE : ℝ)
variable (h1 : ABC = 70)
variable (h2 : BAC = 50)
variable (h3 : CDE = 90)
variable (h4 : ∃ BCA : ℝ, DCE = BCA ∧ ABC + BAC + BCA = 180)

-- The statement to prove
theorem angle_x (x : ℝ) (h : ∃ BCA : ℝ, (ABC = 70) ∧ (BAC = 50) ∧ (CDE = 90) ∧ (DCE = BCA ∧ ABC + BAC + BCA = 180) ∧ (DCE + x = 90)) :
  x = 30 := by
  sorry

end angle_x_l317_317946


namespace problem_statement_l317_317537

noncomputable def collinear_points (A B C P R Q I D G M N S T : Point) : Prop :=
  collinear {M, B, T}

theorem problem_statement
  (A B C P R Q I D G M N S T : Point)
  (hABC : Triangle ABC) 
  (hAB_AC : AB < AC)
  (hTangents : is_tangent P B (circumcircle A B C) ∧ is_tangent P C (circumcircle A B C))
  (hR_on_arc : on_arc R (arc A C) (circumcircle A B C))
  (hPR_intersect_circumQ : Q ∈ meet PR (circumcircle A B C) ∧ Q ≠ R)
  (hIncenter : is_incenter I (Triangle ABC))
  (hID_perp_BC : ID ⊥ BC ∧ footpoint D ID BC)
  (hQD_intersect_circumG : G ∈ meet QD (circumcircle A B C) ∧ G ≠ D)
  (hLine_I_perp_AI : perpendicular_line_through I (line AI) (line AG) M ∧ perpendicular_line_through I (line AI) (line AC) N)
  (hMidpoint_S : midpoint S (arc AR) (circumcircle A B C))
  (hSN_intersect_circumT : T ∈ meet SN (circumcircle A B C) ∧ T ≠ N)
  (hAR_parallel_BC : parallel AR BC) :
  collinear_points A B C P R Q I D G M N S T := 
sorry

end problem_statement_l317_317537


namespace percentage_error_in_area_is_36_89_percent_l317_317376

-- Define the actual side of the square
variable (x : ℝ)

-- Define the measurement error percentage
def measurement_error : ℝ := 0.17

-- Define the measured side with error
def measured_side (x : ℝ) : ℝ := (1 + measurement_error) * x

-- Define the actual area of the square
def actual_area (x : ℝ) : ℝ := x^2

-- Define the calculated (erroneous) area of the square
def calculated_area (x : ℝ) : ℝ := (measured_side x)^2

-- Define the error in the area
def area_error (x : ℝ) : ℝ := calculated_area x - actual_area x

-- Define the percentage error in the area
def percentage_area_error (x : ℝ) : ℝ := (area_error x / actual_area x) * 100

-- Theorem stating the percentage of error in the calculated area is 36.89%
theorem percentage_error_in_area_is_36_89_percent (x : ℝ) : percentage_area_error x = 36.89 := by
  sorry

end percentage_error_in_area_is_36_89_percent_l317_317376


namespace directrix_of_parabola_l317_317409

theorem directrix_of_parabola (h : ∀ x : ℝ, y = -3 * x ^ 2 + 6 * x - 5) : ∃ y : ℝ, y = -25 / 12 :=
by
  sorry

end directrix_of_parabola_l317_317409


namespace three_x_plus_four_l317_317492

theorem three_x_plus_four (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  sorry

end three_x_plus_four_l317_317492


namespace AK_perp_AB_l317_317172

noncomputable theory
open_locale classical

variables (A B C K H : Type*) [inhabited A] [inhabited B] [inhabited C] [inhabited H] [inhabited K]
variables [geometry E, triangle A B C] (h_acute : ∀ a b c : E, acute ∠ A B C) (h_altitude : altitude B H) (h_ABeqCH : AB = CH)
variables (hK : ∠ B K C = ∠ B C K ∧ ∠ A B K = ∠ A C B)

theorem AK_perp_AB : ∠ BAK = 90° :=
sorry

end AK_perp_AB_l317_317172


namespace pseudo_even_cos_l317_317422

def pseudo_even_function (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f x = f (2*a - x)

theorem pseudo_even_cos : pseudo_even_function (λ x, Real.cos (x + 1)) (-1/2) := 
by {
  assume x : ℝ,
  -- proof steps would go here
  sorry
}

end pseudo_even_cos_l317_317422


namespace log_2_a2013_l317_317522

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4 * x^2 + 6 * x - 1

noncomputable def f_prime (x : ℝ) : ℝ := x^2 - 8 * x + 6

theorem log_2_a2013 :
  let a : ℕ → ℝ := λ n, a₁ + (n - 1) * d,
  (extreme_points x f x (a₁ + a₄₀₂₅) = 8) → 
  (a₁ + a₄₀₂₅ = 2 * a₂₀₁₃) → 
  (a₂₀₁₃ = 4) → 
  log 2 a₂₀₁₃ = 2 :=
by
  sorry

end log_2_a2013_l317_317522


namespace no_perfect_square_in_sequence_l317_317136

def sequence_term (i : ℕ) : ℕ :=
  let baseDigits : List ℕ := [2, 0, 1, 4, 2, 0, 1, 5]
  let termDigits := baseDigits.foldr (fun d acc => acc + d * 10 ^ (baseDigits.length - acc.length - 1)) 0
  termDigits + 10 ^ (i + 5)

theorem no_perfect_square_in_sequence : ¬ ∃ i : ℕ, ∃ k : ℕ, k * k = sequence_term i := 
sorry

end no_perfect_square_in_sequence_l317_317136


namespace solve_quadratic_l317_317609

theorem solve_quadratic {x : ℝ} (h : 2 * (x - 1)^2 = x - 1) : x = 1 ∨ x = 3 / 2 :=
sorry

end solve_quadratic_l317_317609


namespace median_shows_increase_l317_317647

noncomputable def median_increase (scores : List ℕ) (new_score : ℕ) : Prop :=
  let original_sorted := scores.sorted
  let new_sorted := (new_score :: scores).sorted
  let original_median := if scores.length % 2 = 1 then
                          original_sorted.nth (scores.length / 2)
                        else
                          (original_sorted.nth (scores.length / 2 - 1) + original_sorted.nth (scores.length / 2)) / 2
  let new_median := if (scores.length + 1) % 2 = 1 then
                          new_sorted.nth ((scores.length + 1) / 2)
                        else
                          (new_sorted.nth ((scores.length + 1) / 2 - 1) + new_sorted.nth ((scores.length + 1) / 2)) / 2
  new_median > original_median

def scores_first_11 := [45, 52, 56, 56, 60, 61, 61, 64, 67, 68, 74]
def twelfth_game_score := 55

theorem median_shows_increase : median_increase scores_first_11 twelfth_game_score :=
by
  sorry

end median_shows_increase_l317_317647


namespace tennis_players_l317_317515

theorem tennis_players (total_members badminton_players neither_sport both_sports : ℕ)
  (h_total : total_members = 30)
  (h_badminton : badminton_players = 17)
  (h_neither : neither_sport = 2)
  (h_both : both_sports = 8) :
  let members_at_least_one := total_members - neither_sport,
      only_badminton := badminton_players - both_sports,
      only_tennis := members_at_least_one - only_badminton - both_sports
  in only_tennis + both_sports = 19 :=
by
  sorry

end tennis_players_l317_317515


namespace square_of_85_l317_317014

-- Define the given variables and values
def a := 80
def b := 5
def c := a + b

theorem square_of_85:
  c = 85 → (c * c) = 7225 :=
by
  intros h
  rw h
  sorry

end square_of_85_l317_317014


namespace roots_of_modified_quadratic_l317_317019

theorem roots_of_modified_quadratic 
  (k : ℝ) (hk : 0 < k) :
  (∃ z₁ z₂ : ℂ, (12 * z₁^2 - 4 * I * z₁ - k = 0) ∧ (12 * z₂^2 - 4 * I * z₂ - k = 0) ∧ (z₁ ≠ z₂) ∧ (z₁.im = 0) ∧ (z₂.im ≠ 0)) ↔ (k = 1/4) :=
by
  sorry

end roots_of_modified_quadratic_l317_317019


namespace directrix_of_parabola_l317_317405

-- Define the given parabola equation
def parabola_eq (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 5

-- Define the expected result for the directrix
def directrix_eq : ℝ := -23 / 12

-- State the problem in Lean
theorem directrix_of_parabola : 
  (∃ d : ℝ, (∀ x y : ℝ, y = parabola_eq x → y = d) → d = directrix_eq) :=
by
  sorry

end directrix_of_parabola_l317_317405


namespace three_colored_flag_l317_317916

theorem three_colored_flag (colors : Finset ℕ) (h : colors.card = 6) : 
  (∃ top middle bottom : ℕ, top ≠ middle ∧ top ≠ bottom ∧ middle ≠ bottom ∧ 
                            top ∈ colors ∧ middle ∈ colors ∧ bottom ∈ colors) → 
  colors.card * (colors.card - 1) * (colors.card - 2) = 120 :=
by 
  intro h_exists
  exact sorry

end three_colored_flag_l317_317916


namespace max_and_liz_problem_l317_317577

theorem max_and_liz_problem :
  ∀ (x y : ℕ), x + y = 23 → 2 * x + 2 = y → y = 16 :=
by
  intros x y h1 h2
  have h3 : x + (2 * x + 2) = 23, from by rw [h2, h1]
  have h4 : 3 * x + 2 = 23, from by rw [add_assoc, ← add_assoc (x + 2)]
  have h5 : 3 * x = 21, from by rwa [add_comm 2 21, add_comm 2 23]
  have h6 : x = 7, from Nat.eq_of_mul_eq_mul_right (by norm_num) h5
  show y = 16, from by rw [h2, h6, mul_two, add_assoc]
  sorry

end max_and_liz_problem_l317_317577


namespace walking_robot_distance_5th_minute_l317_317519

-- Define the initial condition of the problem
def initial_distance : ℕ → ℕ
| 1 := 2
| (n+1) := 2 * initial_distance n

-- State the goal to prove the distance on the 5th minute is 32 meters
theorem walking_robot_distance_5th_minute :
  initial_distance 5 = 32 :=
by
  sorry

end walking_robot_distance_5th_minute_l317_317519


namespace cupcakes_per_tray_l317_317772

theorem cupcakes_per_tray (x : ℕ) 
  (four_trays : 4 * x) 
  (three_fifths_sold : (3 / 5 : ℚ)) 
  (price_per_cupcake : 2 : ℚ) 
  (total_earnings : 4 * x * (3 / 5) * 2 = 96)
  : x = 20 := by sorry

end cupcakes_per_tray_l317_317772


namespace value_of_expression_l317_317486

theorem value_of_expression (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by {
   rw h,
   norm_num,
   sorry
}

end value_of_expression_l317_317486


namespace exists_equal_sum_disjoint_subsets_l317_317584

-- Define the set and conditions
def is_valid_set (S : Finset ℕ) : Prop :=
  S.card = 15 ∧ ∀ x ∈ S, x ≤ 2020

-- Define the problem statement
theorem exists_equal_sum_disjoint_subsets (S : Finset ℕ) (h : is_valid_set S) :
  ∃ (A B : Finset ℕ), A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

end exists_equal_sum_disjoint_subsets_l317_317584


namespace unique_nonnegative_triple_solution_l317_317029

theorem unique_nonnegative_triple_solution :
  ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z →
    (x^2 - y = (z - 1)^2) →
    (y^2 - z = (x - 1)^2) →
    (z^2 - x = (y - 1)^2) →
    (x = 1 ∧ y = 1 ∧ z = 1) := 
by
  intros x y z ngon [hx, hy, hz] hx_eq hy_eq hz_eq
  split
  -- Proof Needed
  sorry

end unique_nonnegative_triple_solution_l317_317029


namespace Amy_current_age_l317_317570

def Mark_age_in_5_years : ℕ := 27
def years_in_future : ℕ := 5
def age_difference : ℕ := 7

theorem Amy_current_age : ∃ (Amy_age : ℕ), Amy_age = 15 :=
  by
    let Mark_current_age := Mark_age_in_5_years - years_in_future
    let Amy_age := Mark_current_age - age_difference
    use Amy_age
    sorry

end Amy_current_age_l317_317570


namespace find_angle_A_find_area_l317_317160

-- Definitions
variables {a b c : ℝ} (A : ℝ)

-- Given conditions for the triangle
axiom triangle_conditions : b^2 + c^2 = a^2 + b * c

-- Required to prove angle A
theorem find_angle_A (h : triangle_conditions) : A = Real.pi / 3 :=
sorry

-- Given specific values for a, b, and A to find the area
variables (a_val : ℝ := Real.sqrt 7) (b_val : ℝ := 2) (c_val : ℝ := 3) (A_val : ℝ := Real.pi / 3)

-- Area of the triangle with given specific values
theorem find_area (ha : a = a_val) (hb : b = b_val) (hc : c = c_val) (hA : A = A_val) :
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 :=
sorry

end find_angle_A_find_area_l317_317160


namespace tangent_line_addition_l317_317157

theorem tangent_line_addition (a b : ℝ) : 
  (∀ x, f x = exp x + x - a) ∧ (∀ x, y = 2 * x + b) →
  (∃ x_0, f'(x_0) = 2 ∧ f(x_0) = 2 * x_0 + b) →
  a + b = 1 := by
  sorry

end tangent_line_addition_l317_317157


namespace fourth_number_is_83_l317_317645

def sequence := [11, 23, 47, 83, 131, 191, 263, 347, 443, 551, 671] -- Define the sequence

theorem fourth_number_is_83 : sequence.nth 3 = some 83 := 
by 
  sorry -- Proof omitted

end fourth_number_is_83_l317_317645


namespace rearrange_possible_l317_317205

theorem rearrange_possible (n : ℕ) (h : n = 25 ∨ n = 1000) :
  ∃ (f : ℕ → ℕ), (∀ i < n, f i + 1 < n → (f (i + 1) - f i = 3 ∨ f (i + 1) - f i = 5)) :=
  sorry

end rearrange_possible_l317_317205


namespace new_profit_percentage_l317_317384

theorem new_profit_percentage (P SP NSP NP : ℝ) 
  (h1 : SP = 1.10 * P) 
  (h2 : SP = 879.9999999999993) 
  (h3 : NP = 0.90 * P) 
  (h4 : NSP = SP + 56) : 
  (NSP - NP) / NP * 100 = 30 := 
by
  sorry

end new_profit_percentage_l317_317384


namespace tracing_arc_center_concyclic_l317_317786

variable {k : Type*} [EuclideanSpace k] (A C1 C2 B P Q R I S : k)
variable (C1_center C2_center : k) (circ1 circ2 : Circle k)
variable (incenter_tri : Triangle k → k) (circumcenter_tri : Triangle k → k)

-- Defining the circles and their properties
def intersecting_circles (circ1 circ2 : Circle k) (A B : k) : Prop :=
  circ1.center = C1_center ∧ circ2.center = C2_center ∧ circ1.has_point A ∧ circ1.has_point B ∧ circ2.has_point A ∧ circ2.has_point B

-- Points P and Q collinear with B and B between them
def collinear_with_B (P Q B : k) : Prop :=
  collinear P Q B ∧ B ∈ seg P Q

-- Intersecting lines and their properties
def intersecting_lines_at_R (P Q R : k) : Prop :=
  line_through P C1_center ∩ line_through Q C2_center = {R}

-- The incenter and circumcenter definitions
def incenter_PQR (P Q R : k) : k := incenter_tri (triangle P Q R)
def circumcenter_PIQ (P Q I : k) : k := circumcenter_tri (triangle P I Q)

-- Final theorem statement
theorem tracing_arc_center_concyclic (circ1 circ2 : Circle k) (A B P Q R I S : k)
  (C1_center C2_center : k)
  (hyp1 : intersecting_circles circ1 circ2 A B)
  (hyp2 : collinear_with_B P Q B)
  (hyp3 : intersecting_lines_at_R P Q R)
  (hyp4 : I = incenter_PQR P Q R)
  (hyp5 : S = circumcenter_PIQ P I Q)
  : ∀ (P Q : k), (center_of_circle S).is_concyclic A C1_center C2_center := 
  sorry

end tracing_arc_center_concyclic_l317_317786


namespace points_on_line_relationship_l317_317882

theorem points_on_line_relationship :
  let m := 2 * Real.sqrt 2 + 1
  let n := 4
  m < n :=
by
  sorry

end points_on_line_relationship_l317_317882


namespace log_product_eq_one_l317_317151

theorem log_product_eq_one {x y : ℝ} (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_ne_one : x ≠ 1) (hy_ne_one : y ≠ 1)
  : log x y * log y x = 1 := 
sorry

end log_product_eq_one_l317_317151


namespace kevins_speed_second_half_hour_l317_317964

theorem kevins_speed_second_half_hour :
  (∀ Distance_1 Distance_3 Distance_2 Total_distance_1_and_3 Total_distance Time_2,
    (Distance_1 = 10 * 0.5) →
    (Distance_3 = 8 * 0.25) →
    (Total_distance_1_and_3 = Distance_1 + Distance_3) →
    (Total_distance = 17) →
    (Distance_2 = Total_distance - Total_distance_1_and_3) →
    (Time_2 = 0.5) →
    (Distance_2 / Time_2 = 20)) :=
by
  intro Distance_1 Distance_3 Distance_2 Total_distance_1_and_3 Total_distance Time_2
  intro h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end kevins_speed_second_half_hour_l317_317964


namespace sixth_power_sum_l317_317288

theorem sixth_power_sum (a b c d e f : ℤ) :
  a^6 + b^6 + c^6 + d^6 + e^6 + f^6 = 6 * a * b * c * d * e * f + 1 → 
  (a = 1 ∨ a = -1 ∨ b = 1 ∨ b = -1 ∨ c = 1 ∨ c = -1 ∨ 
   d = 1 ∨ d = -1 ∨ e = 1 ∨ e = -1 ∨ f = 1 ∨ f = -1) ∧
  ((a = 1 ∨ a = -1 ∨ a = 0) ∧ 
   (b = 1 ∨ b = -1 ∨ b = 0) ∧ 
   (c = 1 ∨ c = -1 ∨ c = 0) ∧ 
   (d = 1 ∨ d = -1 ∨ d = 0) ∧ 
   (e = 1 ∨ e = -1 ∨ e = 0) ∧ 
   (f = 1 ∨ f = -1 ∨ f = 0)) ∧
  (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∨ e ≠ 0 ∨ f ≠ 0) ∧
  (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0 ∨ e = 0 ∨ f = 0) := 
sorry

end sixth_power_sum_l317_317288


namespace range_of_eccentricity_l317_317089

-- Given conditions
variables (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Foci of the hyperbola
def F1 := (-a, 0)
def F2 := (a, 0)

-- Eccentricity of the hyperbola
def e (c : ℝ) := c / a

-- Acute triangle condition
def acute_triangle_condition (x : ℝ) : Prop := 
  let c := sqrt (a^2 + b^2) in 1 < x ∧ x < 1 + sqrt 2 ∧ x = e c

-- Main theorem
theorem range_of_eccentricity :
  (∀ c, (∃ y, hyperbola (-c) y) → acute_triangle_condition (e c)) :=
by
  sorry

end range_of_eccentricity_l317_317089


namespace order_of_abc_l317_317070

def a := Real.sqrt 3
def b := Real.logb (1 / 3) (1 / 2)
def c := Real.logb 2 (1 / 3)

theorem order_of_abc : a > b ∧ b > c := by
  sorry

end order_of_abc_l317_317070


namespace problem_correctness_l317_317471

theorem problem_correctness :
  let octal_to_decimal := 3 * 8^2 + 2 * 8 + 6
  let base5_to_decimal := 1 * 5^3 + 3 * 5^2 + 2 * 5 + 4
  let horner_eval := ((((((7 * 3 + 0) * 3 + 0) * 3 + 4) * 3 + 3) * 3 + 2) * 3 + 1) * 3 + 0
  let stratified_sampling := 16 / (2 / (2 + 3 + 4))
  let systematic_sampling := 240 / (840 / 42)
  
  (octal_to_decimal = 214) ∧
  (base5_to_decimal = 214) ∧
  (horner_eval = 21610) ∧
  (stratified_sampling = 72) ∧
  (systematic_sampling = 12)  -> 
  ∀ (p : ℕ), (p = 1 ∨ p = 2 ∨ p = 3 ∨ p = 4 ∨ p = 5)
  sorry

end problem_correctness_l317_317471


namespace decimal_representation_of_fraction_l317_317816

theorem decimal_representation_of_fraction :
  (3 / 40 : ℝ) = 0.075 :=
sorry

end decimal_representation_of_fraction_l317_317816


namespace order_of_variables_l317_317874

noncomputable def a : ℝ := 0.5 ^ 0.6
noncomputable def b : ℝ := 0.6 ^ 0.5
noncomputable def c : ℝ := Real.logBase 2 0.5
noncomputable def d : ℝ := Real.logBase 2 0.6

theorem order_of_variables : b > a ∧ a > d ∧ d > c :=
by sorry

end order_of_variables_l317_317874


namespace interval_of_monotonic_increase_l317_317632

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (1 / (x - 2))

theorem interval_of_monotonic_increase :
  ∀ x, x ≠ 2 → (∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂) ↔ (x ∈ Ico (-∞:ℝ) 2) ∨ (x ∈ Ioo (2:ℝ) ∞) :=
by
  sorry

end interval_of_monotonic_increase_l317_317632


namespace determine_a_range_l317_317449

theorem determine_a_range (a : ℝ) : 
  (∃ x : ℝ, sin x ^ 2 - (2 * a + 1) * cos x - a ^ 2 = 0) ↔ -5/4 ≤ a ∧ a ≤ 1 + real.sqrt 2 := by
sory

end determine_a_range_l317_317449


namespace window_dimensions_l317_317820

-- Given conditions
def panes := 12
def rows := 3
def columns := 4
def height_to_width_ratio := 3
def border_width := 2

-- Definitions based on given conditions
def width_per_pane (x : ℝ) := x
def height_per_pane (x : ℝ) := 3 * x

def total_width (x : ℝ) := columns * width_per_pane x + (columns + 1) * border_width
def total_height (x : ℝ) := rows * height_per_pane x + (rows + 1) * border_width

-- Theorem statement: width and height of the window
theorem window_dimensions (x : ℝ) : 
  total_width x = 4 * x + 10 ∧ 
  total_height x = 9 * x + 8 := by
  sorry

end window_dimensions_l317_317820


namespace max_log_value_l317_317071

noncomputable def max_log_product (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ a * b = 8 then (Real.logb 2 a) * (Real.logb 2 (2 * b)) else 0

theorem max_log_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b = 8) :
  max_log_product a b ≤ 4 :=
sorry

end max_log_value_l317_317071


namespace period_of_f_max_value_of_f_and_values_l317_317113

noncomputable def f (x : ℝ) : ℝ := (1 - Real.sin (2 * x)) / (Real.sin x - Real.cos x)

-- Statement 1: The period of f(x) is 2π
theorem period_of_f : ∀ x, f (x + 2 * Real.pi) = f x := by
  sorry

-- Statement 2: The maximum value of f(x) is √2 and it is attained at x = 2kπ + 3π/4, k ∈ ℤ
theorem max_value_of_f_and_values :
  (∀ x, f x ≤ Real.sqrt 2) ∧
  (∃ k : ℤ, f (2 * k * Real.pi + 3 * Real.pi / 4) = Real.sqrt 2) := by
  sorry

end period_of_f_max_value_of_f_and_values_l317_317113


namespace calculate_3_diamond_4_l317_317391

-- Define the operations
def op (a b : ℝ) : ℝ := a^2 + 2 * a * b
def diamond (a b : ℝ) : ℝ := 4 * a + 6 * b - op a b

-- State the theorem
theorem calculate_3_diamond_4 : diamond 3 4 = 3 := by
  sorry

end calculate_3_diamond_4_l317_317391


namespace inequality_g_l317_317895

noncomputable def g (x : ℝ) : ℝ := (x^2 - real.cos x) * real.sin (π / 6)

theorem inequality_g
  (x1 x2 : ℝ)
  (h1 : -π / 2 ≤ x1 ∧ x1 ≤ π / 2)
  (h2 : -π / 2 ≤ x2 ∧ x2 ≤ π / 2)
  (h3 : x1 > |x2|)
  (h4 : x1^2 > x2^2) :
  g x1 > g x2 :=
sorry

end inequality_g_l317_317895


namespace ellipse_standard_equation_and_delta_y_l317_317085

noncomputable def eccentricity := 3 / 5
noncomputable def minor_axis_length := 8
noncomputable def minor_axis_half := 4

def ellipse_equation (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

def foci (a b : ℝ) (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = - sqrt(a^2 - b^2) ∧ y1 = 0 ∧ x2 = sqrt(a^2 - b^2) ∧ y2 = 0

def inscribed_circle_radius := 1 / 2
def inscribed_circle_circumference := 2 * inscribed_circle_radius * real.pi

theorem ellipse_standard_equation_and_delta_y
  (a b : ℝ)
  (h1 : a > b > 0)
  (h2 : b = minor_axis_half)
  (h3 : eccentricity = sqrt(1 - (b^2) / (a^2)))
  : ellipse_equation 25 16 a b ∧ ∃ (x1 y1 x2 y2 : ℝ), 
    let f1 := (- sqrt(a^2 - b^2), 0) 
    let f2 := (sqrt(a^2 - b^2), 0)
    (foci a b f1.1 f1.2 f2.1 f2.2) ∧
    (⟨x1, y1⟩, ⟨x2, y2⟩) ∈ line l ∧ inscribed_circle_radius = inscribed_circle_radius ∧
    3 * |y1 - y2| = 5 :=
begin
    sorry
end

end ellipse_standard_equation_and_delta_y_l317_317085


namespace continuous_necessity_l317_317421

-- Define a predicate that states a function is defined at a specific point
def is_defined_at (f : ℝ → ℝ) (x : ℝ) := ∃ y : ℝ, f x = y

-- Define a predicate that states a function is continuous at a specific point
def is_continuous_at (f : ℝ → ℝ) (x : ℝ) := ∀ ε > 0, ∃ δ > 0, ∀ x', abs (x' - x) < δ → abs (f x' - f x) < ε

-- State the main theorem that having a definition at a point implies necessity but not sufficiency for continuity at that point
theorem continuous_necessity (f : ℝ → ℝ) (x : ℝ) :
  is_defined_at f x → (¬ is_defined_at f x ∨ ¬ is_continuous_at f x) :=
by
  intro h_defined_at
  apply or.intro_right
  intro h_continuous_at
  sorry

end continuous_necessity_l317_317421


namespace arrangement_of_digits_11250_l317_317180

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then
    1
  else
    n * factorial (n - 1)

def number_of_arrangements (digits : List ℕ) : ℕ :=
  let number_ends_in_0 := factorial 4 / factorial 2
  let number_ends_in_5 := 3 * (factorial 3 / factorial 2)
  number_ends_in_0 + number_ends_in_5

theorem arrangement_of_digits_11250 :
  number_of_arrangements [1, 1, 2, 5, 0] = 21 :=
by
  sorry

end arrangement_of_digits_11250_l317_317180


namespace combinations_of_7_choose_3_l317_317589

theorem combinations_of_7_choose_3 : nat.choose 7 3 = 35 :=
by
  sorry

end combinations_of_7_choose_3_l317_317589


namespace triangular_pyramid_radius_l317_317082

noncomputable def radius_circumscribed_sphere (P A B C : Type*) [euclidean_geometry P A B C]
  (hPA : ∥P - A∥ = 6)
  (hAB : ∥A - B ∥ = 6)
  (hAC : ∥A - C∥ = 8)
  (hBC : ∥B - C∥ = 10)
  (h_perpendicular : is_perpendicular (plane A B C) P) : ℝ :=
5

theorem triangular_pyramid_radius (P A B C : Type*) [euclidean_geometry P A B C]
  (hPA : ∥P - A∥ = 6)
  (hAB : ∥A - B∥ = 6)
  (hAC : ∥A - C∥ = 8)
  (hBC : ∥B - C∥ = 10)
  (h_perpendicular : is_perpendicular (plane A B C) P) :
  radius_circumscribed_sphere P A B C hPA hAB hAC hBC h_perpendicular = 5 :=
sorry

end triangular_pyramid_radius_l317_317082


namespace rearrange_numbers_l317_317197

theorem rearrange_numbers (n : ℕ) (h : n = 25 ∨ n = 1000) :
  ∃ (l : List ℕ), l.nodup ∧ l.perm (List.range (n + 1))
  ∧ (∀ (i : ℕ), i < n → ((l.get? i.succ = some (l.get! i + 3)) ∨ (l.get? i.succ = some (l.get! i + 5)) ∨ (l.get? i.succ = some (l.get! i - 3)) ∨ (l.get? i.succ = some (l.get! i - 5)))) :=
by
  sorry

end rearrange_numbers_l317_317197


namespace sample_stddev_and_range_unchanged_l317_317862

noncomputable def sample_data (n : ℕ) : Type :=
  fin n → ℝ

variables (n : ℕ) (x y : sample_data n) (c : ℝ)

-- Condition: creating new sample data by adding a constant c
axiom data_transform : ∀ i : fin n, y i = x i + c 

-- Condition: c is non-zero
axiom c_nonzero : c ≠ 0

-- The theorem that states sample standard deviations and sample ranges of x and y are the same
theorem sample_stddev_and_range_unchanged :
  (sample_standard_deviation y = sample_standard_deviation x) ∧ 
  (sample_range y = sample_range x) :=
sorry

end sample_stddev_and_range_unchanged_l317_317862


namespace hypotenuse_not_5_cm_l317_317651

theorem hypotenuse_not_5_cm (a b c : ℝ) (h₀ : a + b = 8) (h₁ : a^2 + b^2 = c^2) : c ≠ 5 := by
  sorry

end hypotenuse_not_5_cm_l317_317651


namespace tangency_points_incircle_l317_317992

variable (A B C A1 B1 C1 : Type)

variables (a b c x y z : ℝ) (triangle : Type)
variables [Inhabited triangle]

def on_side (A B P : triangle) : Prop := sorry
def incircle_tangent (triangle : Type) (A1 B1 C1 A B C : triangle) : Prop := sorry

variable [∀ (A B C A1 B1 C1 : triangle), on_side A B A1 → on_side B C B1 → on_side C A C1 → incircle_tangent triangle A1 B1 C1 A B C]

theorem tangency_points_incircle
  (h1 : on_side A B C)
  (h2 : on_side B C A1)
  (h3 : on_side C A B1)
  (h4 : A1 = midpoint B C)
  (h5 : B1 = midpoint A C)
  (h6 : C1 = midpoint A B)
  (hx : AC1 = AB1)
  (hy : BA1 = BC1)
  (hz : CA1 = CB1) :
  incircle_tangent triangle A1 B1 C1 A B C :=
sorry

end tangency_points_incircle_l317_317992


namespace ellipse_properties_l317_317434

theorem ellipse_properties (b : ℝ) (P Q : ℝ × ℝ)
  (hb : 0 < b ∧ b < 2)
  (hP_inside : (P.1 = sqrt 2 ∧ P.2 = 1) ∧ (P.1^2 / 4 + P.2^2 / b^2 < 1))
  (hQ_on : Q.1^2 / 4 + Q.2^2 / b^2 = 1) :
  (∃ e : ℝ, e ∈ (0, sqrt 2 / 2)) ∧
  (∀ Q : ℝ × ℝ, Q.1^2 / 4 + Q.2^2 / b^2 = 1 →
    dot_product (Q.1, Q.2) (some_f1_function, some_f2_function) ≠ 0) :=
sorry

end ellipse_properties_l317_317434


namespace gcd_A_B_l317_317641

theorem gcd_A_B (a b : ℕ) (h1 : Nat.gcd a b = 1) (h2 : a > 0) (h3 : b > 0) : 
  Nat.gcd (8 * a + 3 * b) (3 * a + 2 * b) ≠ 1 → Nat.gcd (8 * a + 3 * b) (3 * a + 2 * b) = 7 :=
by
  sorry

end gcd_A_B_l317_317641


namespace sample_stat_properties_l317_317867

/-- Given a set of sample data, and another set obtained by adding a non-zero constant to each element of the original set,
prove some properties about their statistical measures. -/
theorem sample_stat_properties (n : ℕ) (c : ℝ) (h₀ : c ≠ 0) 
  (x : ℕ → ℝ) :
  let y := λ i, x i + c 
  in (∑ i in finset.range n, x i) / n ≠ (∑ i in finset.range n, y i) / n ∧ 
     (∃ (median_x : ℝ) (median_y : ℝ), median_y = median_x + c ∧ median_y ≠ median_x) ∧
     (stddev (finset.range n).map x = stddev (finset.range n).map y) ∧
     (range (finset.range n).map x = range (finset.range n).map y) := 
by
  sorry

-- Helper functions for median, stddev, and range could be defined if missing from Mathlib.

end sample_stat_properties_l317_317867


namespace algae_population_growth_l317_317154

theorem algae_population_growth :
  ∀ (triple_duration hours : ℕ) (initial_population target_population: ℕ), 
    triple_duration = 3 →
    initial_population = 200 →
    target_population = 145800 →
    hours = 18 
    →
    target_population = initial_population * 3^((hours / triple_duration)) :=
by
  intros triple_duration hours initial_population target_population H1 H2 H3 H4
  rw [H1, H2, H3, H4]
  norm_num
  exact eq.refl _

end algae_population_growth_l317_317154


namespace similar_triangles_l317_317218

-- Definitions of points and circles
variables {Ω₁ Ω₂ : Circle} {O₁ O₂ A B C D : Point}

-- Assumptions based on the conditions
axiom intersecting_circles : ∃ (Ω₁ Ω₂ : Circle), Ω₁.center = O₁ ∧ Ω₂.center = O₂ ∧ ∃ (A B : Point), A ∈ Ω₁ ∧ A ∈ Ω₂ ∧ B ∈ Ω₁ ∧ B ∈ Ω₂
axiom line_through_B : ∃ (C D : Point), line_through B intersects Ω₁ at C ∧ line_through B intersects Ω₂ at D

-- The statement to be proved
theorem similar_triangles : similar_triangle (triangle A O₁ O₂) (triangle A C D) :=
sorry

end similar_triangles_l317_317218


namespace arrangement_count_of_multiple_of_5_l317_317178

-- Define the digits and the condition that the number must be a five-digit multiple of 5
def digits := [1, 1, 2, 5, 0]
def is_multiple_of_5 (n : Nat) : Prop := n % 5 = 0

theorem arrangement_count_of_multiple_of_5 :
  ∃ (count : Nat), count = 21 ∧
  (∀ (num : List Nat), num.perm digits → is_multiple_of_5 (Nat.of_digits 10 num) → true) :=
begin
  use 21,
  split,
  { refl },
  { intros num h_perm h_multiple_of_5,
    sorry
  }
end

end arrangement_count_of_multiple_of_5_l317_317178


namespace function_decreasing_interval_l317_317631

noncomputable def f (x : ℝ) : ℝ := (x - 4) * Real.exp x  -- Define the function

-- State the theorem
theorem function_decreasing_interval : ∀ x : ℝ, x < 3 → f' x < 0 :=
by
  -- Define the derivative of the function
  let f' (x : ℝ) : ℝ := Real.exp x * (x - 3)
  -- State the fact that we are assuming and setting up
  assume x (hx : x < 3)
  -- Include a sorry for the proof part
  sorry

end function_decreasing_interval_l317_317631


namespace ron_sold_tickets_l317_317252

theorem ron_sold_tickets (R K : ℕ) (h1 : R + K = 20) (h2 : 2 * R + 4.5 * K = 60) : R = 12 :=
by
  sorry

end ron_sold_tickets_l317_317252


namespace distance_between_intersections_l317_317031

noncomputable def intersection_distance (x y : ℝ) : ℝ :=
  if h : x = y^4 ∧ x - y^2 = 1 then 
    let x_val := y^4 in
    let y1 := (1 + Real.sqrt 5) / 2 in
    let y2 := -( (1 + Real.sqrt 5) / 2 ) in
    let dist := Real.sqrt (1 + Real.sqrt 5) in
    dist
  else 0

theorem distance_between_intersections :
  ∃ u v : ℝ, intersection_distance (y^4) (sqrt((1 + sqrt 5)/2)) = Real.sqrt (u + v * Real.sqrt 5) ∧
    (u, v) = (1 : ℝ, 1 : ℝ) :=
by
  sorry

end distance_between_intersections_l317_317031


namespace squares_partition_l317_317513

open Set

variable {α : Type*} [TopologicalSpace α]

/-- Given a finite set of equal, parallel squares in a plane such that among any 
k+1 squares, there exist two that intersect,
prove that this set can be partitioned into no more than 2k-1 non-empty subsets
such that in each subset, all squares share a common point. -/
theorem squares_partition (k : ℕ) (squares : Finset (Set α))
  (h1 : ∀ s ∈ squares, ∃ a : α, is_square s a)
  (h2 : ∀ t : Finset (Set α), t ⊆ squares → (t.card > k) → ∃ s1 s2 ∈ t, intersect s1 s2) :
  ∃ partitions : Finset (Finset (Set α)), 
    partitions.card ≤ 2 * k - 1 ∧ 
    (∀ part ∈ partitions, ∃ p : α, ∀ s ∈ part, p ∈ s) ∧
    (∀ part ∈ partitions, part.nonempty) ∧
    (⋃₀ partitions.to_set = squares.to_set) :=
sorry

end squares_partition_l317_317513


namespace parabola_line_intersection_l317_317094

theorem parabola_line_intersection (F : ℝ × ℝ)
  (A B : ℝ × ℝ)
  (k : ℝ)
  (p : ℝ)
  (hxAF : A.1 ^ 2 = 4 * A.2)
  (hxBF : B.1 ^ 2 = 4 * B.2)
  (hLineA : A.2 = k * A.1 - 1)
  (hLineB : B.2 = k * B.1 - 1)
  (hF : F = (0, 1))
  (hDistAF : abs (A.2 + 1) = 3 * abs (B.2 + 1)) :
  k = 2 * real.sqrt 3 / 3 :=
by
  sorry

end parabola_line_intersection_l317_317094


namespace cost_calculation_proof_l317_317165

theorem cost_calculation_proof (x : ℕ) :
  let ticket_price := 40
  let teachers := 5
  let students := x
  let cost_A := 20 * x + 200
  let cost_B := 24 * x + 120 in
  (cost_A = 20 * x + 200) ∧ (cost_B = 24 * x + 120) ∧ (∃ x50 : ℕ, x50 = 50 ∧ 20 * x50 + 200 = 1200 ∧ 24 * x50 + 120 = 1320 ∧ 20 * x50 + 200 < 24 * x50 + 120) :=
by {
  intro,
  let ticket_price := 40,
  let teachers := 5,
  let students := x,
  let cost_A := 20 * x + 200,
  let cost_B := 24 * x + 120,
  split,
  { refl },
  split,
  { refl },
  use 50,
  split,
  { refl },
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
}

end cost_calculation_proof_l317_317165


namespace matrix_mult_7_l317_317414

theorem matrix_mult_7 (M : Matrix (Fin 3) (Fin 3) ℝ) (v : Fin 3 → ℝ) : 
  (∀ v, M.mulVec v = (7 : ℝ) • v) ↔ M = 7 • 1 :=
by
  sorry

end matrix_mult_7_l317_317414


namespace central_angle_radian_measure_l317_317889

-- Definitions for the conditions
def circumference (r l : ℝ) : Prop := 2 * r + l = 8
def area (r l : ℝ) : Prop := (1/2) * l * r = 4
def radian_measure (l r θ : ℝ) : Prop := θ = l / r

-- Prove the radian measure of the central angle of the sector is 2
theorem central_angle_radian_measure (r l θ : ℝ) : 
  circumference r l → 
  area r l → 
  radian_measure l r θ → 
  θ = 2 :=
by
  sorry

end central_angle_radian_measure_l317_317889


namespace rectangular_solid_diagonal_l317_317289

theorem rectangular_solid_diagonal
  (a b c : ℝ)
  (h1 : 2 * (a * b + b * c + c * a) = 24)
  (h2 : 4 * (a + b + c) = 28) :
  sqrt (a^2 + b^2 + c^2) = 5 := 
by
  sorry

end rectangular_solid_diagonal_l317_317289


namespace prism_faces_l317_317744

-- Define a structure for a prism with a given number of edges
def is_prism (edges : ℕ) := 
  ∃ (n : ℕ), 3 * n = edges

-- Define the theorem to prove the number of faces in a prism given it has 21 edges
theorem prism_faces (h : is_prism 21) : ∃ (faces : ℕ), faces = 9 :=
by
  sorry

end prism_faces_l317_317744


namespace smallest_a_b_sum_l317_317033

theorem smallest_a_b_sum :
∀ (a b : ℕ), 
  (5 * a + 6 = 6 * b + 5) ∧ 
  (∀ d : ℕ, d < 10 → d < a) ∧ 
  (∀ d : ℕ, d < 10 → d < b) ∧ 
  (0 < a) ∧ 
  (0 < b) 
  → a + b = 13 :=
by
  sorry

end smallest_a_b_sum_l317_317033


namespace x_equals_neg_one_l317_317153

theorem x_equals_neg_one
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : (a + b - c) / c = (a - b + c) / b ∧ (a + b - c) / c = (-a + b + c) / a)
  (x : ℝ)
  (h5 : x = (a + b) * (b + c) * (c + a) / (a * b * c))
  (h6 : x < 0) :
  x = -1 := 
sorry

end x_equals_neg_one_l317_317153


namespace sum_of_powers_eq_one_l317_317000

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem sum_of_powers_eq_one : (∑ k in Finset.range 1 17, ω ^ k) = 1 :=
by {
  have h : ω ^ 17 = 1 := by {
    rw [ω, Complex.exp_nat_mul, Complex.mul_div_cancel' (2 * Real.pi * Complex.I) (show (17 : ℂ) ≠ 0, by norm_cast; norm_num)],
    rw Complex.exp_cycle,
  },
  sorry
}

end sum_of_powers_eq_one_l317_317000


namespace max_distinct_sum_max_distinct_sum_max_l317_317323

def sum_first_n_integers (N : ℕ) : ℕ := N * (N + 1) / 2

theorem max_distinct_sum (N : ℕ) : sum_first_n_integers N ≤ 2012 → N ≤ 62 := 
begin
  sorry
end

theorem max_distinct_sum_max (N : ℕ) : sum_first_n_integers N = 2012 → N = 62:= 
begin
  sorry
end

end max_distinct_sum_max_distinct_sum_max_l317_317323


namespace prop1_prop3_l317_317221

def custom_op (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

theorem prop1 (x y : ℝ) : custom_op x y = custom_op y x :=
by sorry

theorem prop3 (x : ℝ) : custom_op (x + 1) (x - 1) = custom_op x x - 1 :=
by sorry

end prop1_prop3_l317_317221


namespace eight_disks_area_sum_final_result_l317_317822

theorem eight_disks_area_sum (r : ℝ) (C : ℝ) :
  C = 1 ∧ r = (Real.sqrt 2 + 1) / 2 → 
  8 * (π * (r ^ 2)) = 2 * π * (3 + 2 * Real.sqrt 2) :=
by
  intros h
  sorry

theorem final_result :
  let a := 6
  let b := 4
  let c := 2
  a + b + c = 12 :=
by
  intros
  norm_num

end eight_disks_area_sum_final_result_l317_317822


namespace subset_card_ge_l317_317560

variable (S : Finset (ℤ × ℤ))

def A (S : Finset (ℤ × ℤ)) : Finset (ℤ × ℤ) :=
  {p ∈ S | ∀ q ∈ S, p ≠ q → (p.1 = q.1 ∧ p.2 ≠ q.2) ∨ (p.1 ≠ q.1 ∧ p.2 = q.2)}

def B (S : Finset (ℤ × ℤ)) : Finset ℤ :=
  Nat.find (λ n, ∃ B : Finset ℤ, B.card = n ∧ ∀ (x, y) ∈ S, x ∈ B ∨ y ∈ B)

theorem subset_card_ge {S : Finset (ℤ × ℤ)} : (A S).card ≥ (B S).card := sorry

end subset_card_ge_l317_317560


namespace binary_to_decimal_10_ones_l317_317800

theorem binary_to_decimal_10_ones :
  let binary_sum := (finset.range 10).sum (λ i, 2^i)
  in binary_sum = 2^10 - 1 :=
by
  sorry

end binary_to_decimal_10_ones_l317_317800


namespace problem_statement_l317_317123

noncomputable def f (x : ℝ) : ℝ := 3^x - 4/x

def p := ∃ x ∈ Ioo (1 : ℝ) (3/2 : ℝ), f x = 0

def q := ∀ x₀, f'' x₀ = 0 → is_extreme_point f x₀

theorem problem_statement : (p ∨ q) := sorry

end problem_statement_l317_317123


namespace find_min_value_l317_317875

theorem find_min_value (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 2) : 
  (∃ (c : ℝ), ∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x + y = 2 → c = 1 / 2 ∧ (8 / ((x + 2) * (y + 4))) ≥ c) :=
  sorry

end find_min_value_l317_317875


namespace sum_ends_with_zero_l317_317576

theorem sum_ends_with_zero :
  ∃ (a b c d : ℕ), a = 111 ∧ b = 22 ∧ c = 2 ∧ d = 555 ∧ (a + b + c + d) % 10 = 0 :=
by {
  use [111, 22, 2, 555],
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  suffices : 690 % 10 = 0, from this,
  norm_num,
  sorry
}

end sum_ends_with_zero_l317_317576


namespace triangle_area_is_3_max_f_l317_317196

noncomputable def triangle_area :=
  let a : ℝ := 2
  let b : ℝ := 2 * Real.sqrt 3
  let c : ℝ := 2
  let A : ℝ := Real.pi / 3
  (1 / 2) * b * c * Real.sin A

theorem triangle_area_is_3 :
  triangle_area = 3 := by
  sorry

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.cos x * (Real.sin x * Real.cos (Real.pi / 3) + Real.cos x * Real.sin (Real.pi / 3))

theorem max_f :
  ∃ x ∈ Set.Icc 0 (Real.pi / 3), f x = 2 + Real.sqrt 3 ∧ x = Real.pi / 12 := by
  sorry

end triangle_area_is_3_max_f_l317_317196


namespace value_of_expression_l317_317489

theorem value_of_expression (x : ℤ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  exact rfl

end value_of_expression_l317_317489


namespace area_region_inside_hexagon_outside_circle_l317_317739

-- Definitions from the conditions
def cube_side_length : ℝ := 2
def inscribed_sphere_radius : ℝ := 1

-- Required theorem statement
theorem area_region_inside_hexagon_outside_circle (sₛ := cube_side_length) (rₛ := inscribed_sphere_radius)
  (h₁ : sₛ = 2) (h₂ : rₛ = 1) : 
  (3 * Real.sqrt 3 - Real.pi) = 
  let a : ℝ := Real.sqrt 2
  let hex_area : ℝ := (3 * Real.sqrt 3 / 2) * (a ^ 2)
  let circle_area : ℝ := Real.pi * (rₛ ^ 2)
  hex_area - circle_area := 
  by 
    have ha : a = Real.sqrt 2 := rfl
    have hex_area_eq : hex_area = 3 * Real.sqrt 3 := 
    by 
      calc
      hex_area = (3 * Real.sqrt 3 / 2) * (a ^ 2) : by rfl
      ... = (3 * Real.sqrt 3 / 2) * (Real.sqrt 2 ^ 2) : by rw ha
      ... = (3 * Real.sqrt 3 / 2) * 2 : by norm_num
      ... = 3 * Real.sqrt 3 : by norm_num
    have circle_area_eq : circle_area = Real.pi := 
    by 
      calc
      circle_area = Real.pi * (rₛ ^ 2) : by rfl
      ... = Real.pi * (1 ^ 2) : by rw h₂
      ... = Real.pi * 1 : by norm_num
      ... = Real.pi : by norm_num
    calc
    hex_area - circle_area = 3 * Real.sqrt 3 - Real.pi : by rw [hex_area_eq, circle_area_eq]

end area_region_inside_hexagon_outside_circle_l317_317739


namespace terminal_side_second_or_third_quadrant_l317_317476

-- Definitions and conditions directly from part a)
def sin (x : ℝ) : ℝ := sorry
def tan (x : ℝ) : ℝ := sorry
def terminal_side_in_quadrant (x : ℝ) (q : ℕ) : Prop := sorry

-- Proving the mathematically equivalent proof
theorem terminal_side_second_or_third_quadrant (x : ℝ) :
  sin x * tan x < 0 →
  (terminal_side_in_quadrant x 2 ∨ terminal_side_in_quadrant x 3) :=
by
  sorry

end terminal_side_second_or_third_quadrant_l317_317476


namespace random_events_l317_317374

-- Define the concept of events given specific conditions
def event1 : Prop := ∃ (selected : set ℕ), selected ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧ selected.card = 3 ∧ ∀ x ∈ selected, x ≤ 8
def event2 : Prop := ∃ (digit : ℕ), digit ∈ finset.range 10 ∧ digit = correct_digit
def event3 : Prop := ∀ (charge1 charge2 : ℕ), opposite_charges charge1 charge2 → attract charge1 charge2
def event4 : Prop := ∃ (win : bool), win = true ∧ participated_lottery

-- Prove that event3 is certain and event1, event2, and event4 are random
theorem random_events : 
  (event1 ∧ event2 ∧ event4) ∧ event3 :=
by
  have event1_random : event1 := sorry
  have event2_random : event2 := sorry
  have event4_random : event4 := sorry
  have event3_certain : event3 := sorry
  exact ⟨⟨event1_random, event2_random, event4_random⟩, event3_certain⟩

end random_events_l317_317374


namespace solution_set_ineq_l317_317648

theorem solution_set_ineq (x : ℝ) :
  4^x - 2^(x+2) + 3 ≥ 0 ↔ x ∈ (Set.Iic 0 ∪ Set.Ici (Real.log 3 / Real.log 2)) := by
  sorry

end solution_set_ineq_l317_317648


namespace sequence_sum_l317_317527

-- Defining the sequence terms
variables (J K L M N O P Q R S : ℤ)
-- Condition N = 7
def N_value : Prop := N = 7
-- Condition sum of any four consecutive terms is 40
def sum_of_consecutive : Prop := 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40

-- The main theorem stating J + S = 40 given the conditions
theorem sequence_sum (N_value : N = 7) (sum_of_consecutive : 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40) : 
  J + S = 40 := sorry

end sequence_sum_l317_317527


namespace simplify_expression_l317_317258

theorem simplify_expression (a b c d x y : ℝ) (h : cx ≠ -dy) :
  (cx * (b^2 * x^2 + 3 * b^2 * y^2 + a^2 * y^2) + dy * (b^2 * x^2 + 3 * a^2 * x^2 + a^2 * y^2)) / (cx + dy)
  = (b^2 + 3 * a^2) * x^2 + (a^2 + 3 * b^2) * y^2 := by
  sorry

end simplify_expression_l317_317258


namespace bisect_BC_l317_317095

theorem bisect_BC
  (A B C H F: Point)
  (h_orthocenter: orthocenter H A B C)
  (h_circle_intersect: ∃ circle1, diameter circle1 A H ∧ intersects_circumcircle circle1 (circumcircle A B C) F)
  (h_circle_diameter: diameter (circumcircle A B C) A H)
  : bisects F H B C := by
  sorry

end bisect_BC_l317_317095


namespace find_angle_ACB_l317_317159

noncomputable def angle_ACB (A B C D : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  (angle_ABC : ℝ) (angle_DAB : ℝ) (BD : ℝ) (CD : ℝ) (h1 : angle_ABC = π / 3) (h2 : 3 * BD = CD) (h3 : angle_DAB = π / 18) : ℝ :=
  sorry

theorem find_angle_ACB (A B C D : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  (angle_ABC : ℝ) (angle_DAB : ℝ) (BD : ℝ) (CD : ℝ) 
  (h1 : angle_ABC = π / 3) (h2 : 3 * BD = CD) (h3 : angle_DAB = π / 18) :
  angle_ACB A B C D angle_ABC angle_DAB BD CD h1 h2 h3 = 4 * π / 9 :=
  sorry

end find_angle_ACB_l317_317159


namespace find_BI_length_l317_317530

noncomputable def length_BI (a b c : ℝ) (AB AC BC : ℝ) (K s r : ℝ) (I D E F BI : Point) :=
  ∀ (a ≠ b) (a ≠ c) (b ≠ c)
    (incenter I a b c)
    (incircle_touches I D E F a b c)
    (AB = 15) (AC = 17) (BC = 16)
    (heron_area a b c K)
    (semi_perimeter a b c s)
    (r = K / s)
    (BD_length D B = 7)
    (radius_incircle r = 6),
  BI_length B I = sqrt 85

-- Here we define points and functions needed to comply with our conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def incenter : Point → Point → Point → Point → Prop := sorry
def incircle_touches : Point → Point → Point → Point → Point → Point → Prop := sorry
def heron_area : Point → Point → Point → ℝ → Prop := sorry
def semi_perimeter : Point → Point → Point → ℝ → Prop := sorry
def radius_incircle : ℝ → ℝ := sorry
def BD_length : Point → Point → ℝ := sorry
def BI_length : Point → Point → ℝ := sorry

-- Assert the final desired length of BI 
theorem find_BI_length : length_BI a b c AB AC BC K s r I D E F (BI : Point) :=
  begin
    sorry -- proof to be completed
  end

end find_BI_length_l317_317530


namespace distance_between_points_l317_317049

theorem distance_between_points : 
  ∀ (x1 y1 x2 y2 : ℤ), (x1, y1) = (0, 10) → (x2, y2) = (24, 0) → 
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 26 :=  
by
  sorry

end distance_between_points_l317_317049


namespace pradeep_max_marks_l317_317245

theorem pradeep_max_marks (M : ℝ) 
  (pass_condition : 0.35 * M = 210) : M = 600 :=
sorry

end pradeep_max_marks_l317_317245


namespace value_of_b_l317_317929

-- Definitions based on conditions
def curve (x : ℝ) : ℝ := Real.log x
def tangent_line (x : ℝ) (b : ℝ) : ℝ := Real.exp 1 * x + b

-- The point of tangency
def point_of_tangency (x₀ : ℝ) : Prop := curve x₀ = Real.log x₀

-- The slope of the curve at the point of tangency
def slope_at_tangency (x₀ : ℝ) : ℝ := (curve.deriv) x₀

-- The equation of the tangent line at the point of tangency
def eq_tangent_at_tangency (x₀ : ℝ) (x : ℝ) (b : ℝ) : ℝ := slope_at_tangency x₀ * (x - x₀) + curve x₀ - b

-- Prove b = -2
theorem value_of_b : ∃ (x₀ : ℝ), point_of_tangency x₀ → tangent_line x₀ (-2) = eq_tangent_at_tangency x₀ x₀ (-2) :=
by
  sorry

end value_of_b_l317_317929


namespace sum_first_100_terms_of_sum_seq_l317_317878

theorem sum_first_100_terms_of_sum_seq :
  ∀ (a b : ℕ → ℕ), 
    (a 1 = 25) →
    (b 1 = 75) →
    (a 100 + b 100 = 100) →
    (∑ i in Finset.range 100, (a (i + 1) + b (i + 1))) = 10000 :=
by sorry

end sum_first_100_terms_of_sum_seq_l317_317878


namespace sequence_periodicity_l317_317428

theorem sequence_periodicity (a : Real) (h : a > 0) : 
  (let S : ℕ → Real :=
    λ n, match n with
      | 1 => 1 / a
      | n + 1 => if (n + 1) % 2 = 0 then -S n - 1 else 1 / S n
    in S 2023 = 1 / 2) :=
by 
  have a_gt_0 := h
  let S : ℕ → Real := λ n, match n with
    | 1 => 1 / a
    | n + 1 => if (n + 1) % 2 = 0 then -S n - 1 else 1 / S n
  have s1 := S 1
  have s7 := S 7
  sorry

end sequence_periodicity_l317_317428


namespace probability_two_dice_sum_gt_8_l317_317297

def num_ways_to_get_sum_at_most_8 := 
  1 + 2 + 3 + 4 + 5 + 6 + 5

def total_outcomes := 36

def probability_sum_greater_than_8 : ℚ := 1 - (num_ways_to_get_sum_at_most_8 / total_outcomes)

theorem probability_two_dice_sum_gt_8 :
  probability_sum_greater_than_8 = 5 / 18 :=
by
  sorry

end probability_two_dice_sum_gt_8_l317_317297


namespace avg_monthly_bill_over_6_months_l317_317726

theorem avg_monthly_bill_over_6_months :
  ∀ (avg_first_4_months avg_last_2_months : ℝ), 
  avg_first_4_months = 30 → 
  avg_last_2_months = 24 → 
  (4 * avg_first_4_months + 2 * avg_last_2_months) / 6 = 28 :=
by
  intros
  sorry

end avg_monthly_bill_over_6_months_l317_317726


namespace difference_of_squares_153_147_l317_317797

theorem difference_of_squares_153_147 : (153^2 - 147^2) = 1800 := by
  sorry

end difference_of_squares_153_147_l317_317797


namespace exists_triangle_with_small_area_l317_317747

theorem exists_triangle_with_small_area (m : ℕ) :
  ∃ T : set ℝ, T ⊆ ({p | p ∈ set.Icc (0 : ℝ) 1}) ∧
  area T ≤ (3 * Real.sqrt 3) / (4 * (m + 2)) := by
  sorry

end exists_triangle_with_small_area_l317_317747


namespace trivia_team_students_l317_317292

def total_students (not_picked groups students_per_group: ℕ) :=
  not_picked + groups * students_per_group

theorem trivia_team_students (not_picked groups students_per_group: ℕ) (h_not_picked: not_picked = 10) (h_groups: groups = 8) (h_students_per_group: students_per_group = 6) :
  total_students not_picked groups students_per_group = 58 :=
by
  sorry

end trivia_team_students_l317_317292


namespace x_squared_minus_one_l317_317444

theorem x_squared_minus_one (x : ℤ) (h : 3^(x+1) + 3^(x+1) = 486) : x^2 - 1 = 15 :=
by 
  -- Proof would go here
  sorry

end x_squared_minus_one_l317_317444


namespace sum_of_squares_l317_317152

variable {x y z a b c : Real}
variable (h₁ : x * y = a) (h₂ : x * z = b) (h₃ : y * z = c)
variable (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)

theorem sum_of_squares : x^2 + y^2 + z^2 = (a * b)^2 / (a * b * c) + (a * c)^2 / (a * b * c) + (b * c)^2 / (a * b * c) := 
sorry

end sum_of_squares_l317_317152


namespace total_cost_is_63_l317_317780

-- Define the original price, markdown percentage, and sales tax percentage
def original_price : ℝ := 120
def markdown_percentage : ℝ := 0.50
def sales_tax_percentage : ℝ := 0.05

-- Calculate the reduced price
def reduced_price : ℝ := original_price * (1 - markdown_percentage)

-- Calculate the sales tax on the reduced price
def sales_tax : ℝ := reduced_price * sales_tax_percentage

-- Calculate the total cost
noncomputable def total_cost : ℝ := reduced_price + sales_tax

-- Theorem stating that the total cost of the aquarium is $63
theorem total_cost_is_63 : total_cost = 63 := by
  sorry

end total_cost_is_63_l317_317780


namespace no_identical_sums_of_larger_cube_l317_317821

-- Define the conditions as Lean definitions
def face_numbers (cuboids : ℕ) := ∀ i ∈ finset.range cuboids, 1 ≤ i ∧ i ≤ 6

def touching_faces_differ_by_1 (cube : ℕ × ℕ × ℕ) :=
∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x →
  (x = y + 1 ∨ x = y - 1) ∧ (y = z + 1 ∨ y = z - 1)

def sum_faces_identical (sums : list ℕ) (n : ℕ) :=
∀ sum ∈ sums, sum = n

-- Proposition we want to prove
theorem no_identical_sums_of_larger_cube 
  (cuboids : ℕ) (cube : ℕ × ℕ × ℕ) (total_sum : ℕ) : ¬ 
  (face_numbers cuboids ∧ touching_faces_differ_by_1 cube ∧ 
  ∃ sums : list ℕ, sum_faces_identical sums n) :=
begin
  sorry
end

end no_identical_sums_of_larger_cube_l317_317821


namespace investment_C_120000_l317_317369

noncomputable def investment_C (P_B P_A_difference : ℕ) (investment_A investment_B : ℕ) : ℕ :=
  let P_A := (P_B * investment_A) / investment_B
  let P_C := P_A + P_A_difference
  (P_C * investment_B) / P_B

theorem investment_C_120000
  (investment_A investment_B P_B P_A_difference : ℕ)
  (hA : investment_A = 8000)
  (hB : investment_B = 10000)
  (hPB : P_B = 1400)
  (hPA_difference : P_A_difference = 560) :
  investment_C P_B P_A_difference investment_A investment_B = 120000 :=
by
  sorry

end investment_C_120000_l317_317369


namespace domain_f_parity_f_monotonicity_f_l317_317892

open set real

-- Define the function
def f (x : ℝ) : ℝ := 1 / (x^2 - 1)

-- 1. Prove that the domain of f(x) is {x | x ≠ ±1}
theorem domain_f : ∀ x : ℝ, (x ≠ 1 ∧ x ≠ -1) ↔ x ∈ {x | f x = f x} :=
by sorry

-- 2. Prove that f(x) is an even function
theorem parity_f : ∀ x : ℝ, f (-x) = f x :=
by sorry

-- 3. Prove that f(x) is decreasing on the interval (1, +∞)
theorem monotonicity_f : ∀ {x₁ x₂ : ℝ}, 1 < x₁ → x₁ < x₂ → 1 < x₂ → f x₁ > f x₂ :=
by sorry

end domain_f_parity_f_monotonicity_f_l317_317892


namespace binomial_expansion_constant_term_l317_317192

theorem binomial_expansion_constant_term (x : ℝ) (n : ℕ) (A B : ℝ)
  (h_sum_coeffs : A = 4^n) 
  (h_binom_coeffs : B = 2^n)
  (h_equation : A + B = 72) :
  (√x + (3/x))^n = (√x + (3/x))^3 → T = 9 := 
by
  sorry

end binomial_expansion_constant_term_l317_317192


namespace value_of_k_for_binomial_square_l317_317309

theorem value_of_k_for_binomial_square (k : ℝ) : (∃ (b : ℝ), x^2 - 18 * x + k = (x + b)^2) → k = 81 :=
by
  intro h
  cases h with b hb
  -- We will use these to directly infer things without needing the proof here
  sorry

end value_of_k_for_binomial_square_l317_317309


namespace transformations_map_onto_itself_l317_317802

noncomputable def recurring_pattern_map_count (s : ℝ) : ℕ := sorry

theorem transformations_map_onto_itself (s : ℝ) :
  recurring_pattern_map_count s = 2 := sorry

end transformations_map_onto_itself_l317_317802


namespace total_birds_in_marsh_l317_317335

theorem total_birds_in_marsh (geese ducks : ℕ) (h_geese : geese = 58) (h_ducks : ducks = 37) : geese + ducks = 95 := 
by
  rw [h_geese, h_ducks]
  simp
  norm_num

end total_birds_in_marsh_l317_317335


namespace slope_of_line_through_focus_l317_317902

def parabola (x y : ℝ) := y^2 = 4 * x

def focus : ℝ × ℝ := (1, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem slope_of_line_through_focus (A B : ℝ × ℝ)
  (m n : ℝ) (k : ℝ)
  (h_focusA : distance focus A = m)
  (h_focusB : distance focus B = n)
  (h_slope_min : (abs (distance A focus) + 4 * abs (distance B focus)) = min (abs (distance A focus) + 4 * abs (distance B focus))) : 
  (k = 2 * Real.sqrt 2) ∨ (k = -2 * Real.sqrt 2) := 
sorry

end slope_of_line_through_focus_l317_317902


namespace imaginary_part_of_z_magnitude_of_omega_raised_to_2012_l317_317108

-- Define the given condition
def complex_condition (z : ℂ) : Prop := i * (z + 1) = -2 + 2 * i

-- Define the proof problem for the imaginary part
theorem imaginary_part_of_z (z : ℂ) (hz : complex_condition z) : z.im = 2 := 
sorry

-- Define the proof problem for the magnitude of omega raised to 2012
theorem magnitude_of_omega_raised_to_2012 (z ω : ℂ) (hz : complex_condition z) (hω : ω = z / (1-2*i)) : abs ω ^ 2012 = 1 :=
sorry

end imaginary_part_of_z_magnitude_of_omega_raised_to_2012_l317_317108


namespace smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l317_317681

theorem smallest_positive_perfect_square_divisible_by_5_and_6_is_900 :
  ∃ n : ℕ, 0 < n ∧ (n ^ 2) % 5 = 0 ∧ (n ^ 2) % 6 = 0 ∧ (n ^ 2 = 900) := by
  sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_is_900_l317_317681


namespace prism_faces_l317_317743

-- Define a structure for a prism with a given number of edges
def is_prism (edges : ℕ) := 
  ∃ (n : ℕ), 3 * n = edges

-- Define the theorem to prove the number of faces in a prism given it has 21 edges
theorem prism_faces (h : is_prism 21) : ∃ (faces : ℕ), faces = 9 :=
by
  sorry

end prism_faces_l317_317743


namespace average_monthly_bill_l317_317727

-- Definitions based on conditions
def first_4_months_average := 30
def last_2_months_average := 24
def first_4_months_total := 4 * first_4_months_average
def last_2_months_total := 2 * last_2_months_average
def total_spent := first_4_months_total + last_2_months_total
def total_months := 6

-- The theorem statement
theorem average_monthly_bill : total_spent / total_months = 28 := by
  sorry

end average_monthly_bill_l317_317727


namespace sequence_not_perfect_square_l317_317143

theorem sequence_not_perfect_square (n : ℕ) (N : ℕ → ℕ) :
  (∀ i, let d := [2, 0, 0, 1, 4, 0, 2, 0, 1, 5]
  in (N i = 2014 * 10^(n - i) + list.sum d) ∧ list.sum d = 15) →
  (∀ i, ¬ is_square (N i)) :=
by
  sorry

end sequence_not_perfect_square_l317_317143


namespace vasya_divisibility_l317_317717

theorem vasya_divisibility (a b : ℕ) (ha : 0 < a ∧ a ≤ 12) (hb : 0 ≤ b ∧ b < 60) :
  ¬ ∃ k : ℤ, 100 * a + b = k * (60 * a + b) :=
by {
  intro h,
  cases h with k hk,
  sorry,
}

end vasya_divisibility_l317_317717


namespace pencils_used_l317_317578

theorem pencils_used (initial_pencils now_pencils : ℕ) (h1 : initial_pencils = 94) (h2 : now_pencils = 91) : initial_pencils - now_pencils = 3 :=
by
  rw [h1, h2]
  norm_num

end pencils_used_l317_317578


namespace sample_statistics_comparison_l317_317858

variable {α : Type*} [LinearOrder α] [Field α]

def sample_data_transformed (x : list α) (c : α) : list α :=
  x.map (λ xi, xi + c)

def sample_mean (x : list α) : α :=
  x.sum / (x.length : α)

def sample_median (x : list α) : α :=
  if x.length % 2 = 1 then
    x.sort.nth_le (x.length / 2) (by sorry)
  else
    (x.sort.nth_le (x.length / 2 - 1) (by sorry) + x.sort.nth_le (x.length / 2) (by sorry)) / 2

def sample_variance (x : list α) : α :=
  let m := sample_mean x
  in (x.map (λ xi, (xi - m)^2)).sum / (x.length : α)

def sample_standard_deviation (x : list α) : α :=
  (sample_variance x).sqrt

def sample_range (x : list α) : α :=
  x.maximum (by sorry) - x.minimum (by sorry)

theorem sample_statistics_comparison (x : list α) (c : α) (hc : c ≠ 0) :
  (sample_mean (sample_data_transformed x c) ≠ sample_mean x) ∧
  (sample_median (sample_data_transformed x c) ≠ sample_median x) ∧
  (sample_standard_deviation (sample_data_transformed x c) = sample_standard_deviation x) ∧
  (sample_range (sample_data_transformed x c) = sample_range x) :=
  sorry

end sample_statistics_comparison_l317_317858


namespace percentage_of_students_owning_cats_l317_317517

theorem percentage_of_students_owning_cats (N C : ℕ) (hN : N = 500) (hC : C = 75) :
  (C / N : ℚ) * 100 = 15 := by
  sorry

end percentage_of_students_owning_cats_l317_317517


namespace triangle_probability_l317_317349

theorem triangle_probability :
  (∑ a in Finset.range 6, ∑ b in Finset.range 6, if a + 1 + b + 1 > 5 then 1 else 0) / 36 = 7 / 18 :=
by
  sorry

end triangle_probability_l317_317349


namespace max_value_y_l317_317445

-- Define the problem conditions
variables (a b c d x : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hbd : b < d)

-- Define the function y
def y := a * real.sqrt (x - b) + c * real.sqrt (d - x)

-- Statement of the theorem
theorem max_value_y : (a * real.sqrt (x - b) + c * real.sqrt (d - x)) ≤ real.sqrt((d - b) * (a^2 + c^2)) :=
sorry

end max_value_y_l317_317445


namespace ratio_of_areas_l317_317722

theorem ratio_of_areas (r : ℝ) (h : r > 0) :
  let R1 := r
  let R2 := 3 * r
  let S1 := 6 * R1
  let S2 := 6 * r
  let area_smaller_circle := π * R2 ^ 2
  let area_larger_square := S2 ^ 2
  (area_smaller_circle / area_larger_square) = π / 4 :=
by
  sorry

end ratio_of_areas_l317_317722


namespace grape_sweets_divisible_by_four_l317_317294

theorem grape_sweets_divisible_by_four (G : ℕ) 
  (h1 : ∃ n : ℕ, 36 = 4 * n) 
  (h2 : ∃ m : ℕ, G = 4 * m) : 
  ∃ k : ℕ, G = 4 * k :=
begin
  -- since we have h2 stating G = 4 * m, the theorem statement is already proved
  exact h2,
end

end grape_sweets_divisible_by_four_l317_317294


namespace correct_operation_l317_317322

variable (a b : ℝ)

theorem correct_operation :
  -a^6 / a^3 = -a^3 := by
  sorry

end correct_operation_l317_317322


namespace no_n_ge_1_such_that_sum_is_perfect_square_l317_317045

theorem no_n_ge_1_such_that_sum_is_perfect_square :
  ¬ ∃ n : ℕ, n ≥ 1 ∧ ∃ k : ℕ, 2^n + 12^n + 2014^n = k^2 :=
by
  sorry

end no_n_ge_1_such_that_sum_is_perfect_square_l317_317045


namespace cos_double_angle_l317_317150

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.cos (2 * α) = -3 / 5 := by
  sorry

end cos_double_angle_l317_317150


namespace trigonometric_identity_l317_317843

theorem trigonometric_identity
  (α β : Real)
  (h : Real.cos α * Real.cos β - Real.sin α * Real.sin β = 0) :
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = 1 ∨
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = -1 := by
  sorry

end trigonometric_identity_l317_317843


namespace newer_model_distance_l317_317354

theorem newer_model_distance (d_old : ℝ) (p_increase : ℝ) (d_new : ℝ) (h1 : d_old = 300) (h2 : p_increase = 0.30) (h3 : d_new = d_old * (1 + p_increase)) : d_new = 390 :=
by
  sorry

end newer_model_distance_l317_317354


namespace min_val_AP_plus_DQ_l317_317501

noncomputable def minSegmentSum
  (A B C D P Q : Point)
  (AC_midpoint_D: Midpoint A C D)
  (triangle_angles : Angle B = π / 4 ∧ Angle C = 5 * π / 12)
  (AC_length : Distance A C = 2 * sqrt 6)
  (PQ_slides_BC : Segment PQ)
  (PQ_length : Distance P Q = 3)
  : ℝ :=
  min (Distance A P + Distance D Q) (3 * sqrt 10 / 2 + sqrt 30 / 2)
  
theorem min_val_AP_plus_DQ 
  (A B C D P Q : Point)
  (AC_midpoint_D : Midpoint A C D)
  (triangle_angles : Angle B = π / 4 ∧ Angle C = 5 * π / 12)
  (AC_length : Distance A C = 2 * sqrt 6)
  (PQ_slides_BC : Segment PQ)
  (PQ_length : Distance P Q = 3)
  : minSegmentSum A B C D P Q AC_midpoint_D triangle_angles AC_length PQ_slides_BC PQ_length =  3 * sqrt 10 / 2 + sqrt 30 / 2
 := by
  sorry

end min_val_AP_plus_DQ_l317_317501


namespace max_area_cross_section_prism_cut_plane_l317_317362

theorem max_area_cross_section_prism_cut_plane :
  let prism_base_side := 8 in
  let plane := λ x y z => 3 * x - 5 * y + 2 * z = 20 in
  let cross_section_area := 160 in
  prism_with_square_base_max_area prism_base_side plane = cross_section_area :=
by
  sorry

end max_area_cross_section_prism_cut_plane_l317_317362


namespace find_m_from_equation_l317_317109

theorem find_m_from_equation :
  ∀ (x m : ℝ), (x^2 + 2 * x - 1 = 0) → ((x + m)^2 = 2) → m = 1 :=
by
  intros x m h1 h2
  sorry

end find_m_from_equation_l317_317109


namespace statisticalProperties_l317_317855

def sampleStandardDeviation (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  let variance := (data.map (λ x, (x - mean) ^ 2)).sum / data.length
  Real.sqrt variance

def sampleRange (data : List ℝ) : ℝ := data.maximum - data.minimum

theorem statisticalProperties (n : ℕ) (x y : Fin n → ℝ) (c : ℝ) (h : ∀ i : Fin n, y i = x i + c) (hn : n > 0) (hc : c ≠ 0) :
  sampleStandardDeviation (List.ofFn y) = sampleStandardDeviation (List.ofFn x) ∧ sampleRange (List.ofFn y) = sampleRange (List.ofFn x) := 
by
  sorry

end statisticalProperties_l317_317855


namespace larger_sphere_radius_l317_317299

theorem larger_sphere_radius (r : ℝ) (π : ℝ) (h : r^3 = 2) :
  r = 2^(1/3) :=
by
  sorry

end larger_sphere_radius_l317_317299


namespace number_of_groups_of_sets_l317_317127

section ProofProblem

-- Define the universal set
def I : set ℕ := {0, 1, 2}

-- Define the set complement operation
def complement (universal : set ℕ) (s : set ℕ) : set ℕ := universal \ s

-- State the main problem
theorem number_of_groups_of_sets (A B : set ℕ) (h_cond : complement I (A ∪ B) = {2}) : 
    (A = ∅ ∧ B = {0, 1}) ∨ 
    (A = {0} ∧ B = {0, 1}) ∨ 
    (A = {0} ∧ B = {1}) ∨ 
    (A = {1} ∧ B = {0}) ∨ 
    (A = {1} ∧ B = {0, 1}) ∨ 
    (A = {0, 1} ∧ B = {0, 1}) ∨ 
    (A = {0, 1} ∧ B = {0}) ∨ 
    (A = {0, 1} ∧ B = {1}) ∨ 
    (A = {0, 1} ∧ B = ∅) :=
sorry

end ProofProblem

end number_of_groups_of_sets_l317_317127


namespace monotonic_intervals_find_a_l317_317087

-- Problem 1: Monotonic intervals of the function f
theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2)) ∧
  (a > 0 → (∀ x1 x2 : ℝ, x1 < x2 → (x2 < log a → f x1 > f x2) ∧ (x1 > log a → f x1 < f x2))) :=
sorry

-- Problem 2: Finding the set of real numbers for a
theorem find_a (a : ℝ) :
  (∀ x : ℝ, -π / 2 ≤ x ∧ x ≤ π / 4 → f x - g x ≥ 0) → a = 2 :=
sorry

end monotonic_intervals_find_a_l317_317087


namespace blocksDifferByTwoWaysIs35_l317_317721

-- Define the characteristics of a block
inductive Material | plastic | wood | metal
inductive Size | small | large
inductive Color | blue | green | red | yellow | orange
inductive Shape | circle | square | triangle | hexagon

-- Define a block as a tuple of its characteristics
structure Block where
  mat : Material
  size : Size
  col : Color
  shape : Shape

-- The set of all blocks
def allBlocks : List Block := List.product (List.product (List.product Material Size) Color) Shape

-- The reference block
def referenceBlock : Block := { mat := Material.metal, size := Size.large, col := Color.green, shape := Shape.hexagon }

-- Count the number of blocks that differ from the reference block in exactly 2 ways
def numDifferByTwoWays (blocks : List Block) (ref : Block) : Nat :=
  blocks.count (λ b =>
    let diff_material := b.mat ≠ ref.mat
    let diff_size := b.size ≠ ref.size
    let diff_color := b.col ≠ ref.col
    let diff_shape := b.shape ≠ ref.shape
    [diff_material, diff_size, diff_color, diff_shape].count id = 2)

-- The theorem to be proven
theorem blocksDifferByTwoWaysIs35 : numDifferByTwoWays allBlocks referenceBlock = 35 := by
  -- Proof would go here
  sorry

end blocksDifferByTwoWaysIs35_l317_317721


namespace total_heads_l317_317351

variables (H C : ℕ)

theorem total_heads (h_hens: H = 22) (h_feet: 2 * H + 4 * C = 140) : H + C = 46 :=
by
  sorry

end total_heads_l317_317351


namespace count_two_digit_prime_numbers_with_units_digit_7_l317_317915

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_units_digit_7 (n : ℕ) : Prop := n % 10 = 7

def tens_digit_is_not_2_5_or_8 (n : ℕ) : Prop :=
  let tens_digit := (n / 10) % 10 in tens_digit ≠ 2 ∧ tens_digit ≠ 5 ∧ tens_digit ≠ 8

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem count_two_digit_prime_numbers_with_units_digit_7 : 
  ∃ (count : ℕ), count = 5 ∧
  ((∀ n, is_two_digit n → is_units_digit_7 n → tens_digit_is_not_2_5_or_8 n → is_prime n) ↔ count = 5) := by
 sorry

end count_two_digit_prime_numbers_with_units_digit_7_l317_317915


namespace Q_not_in_set_of_3_l317_317547

theorem Q_not_in_set_of_3 (a b c d : ℝ) (h : a * d - b * c = 1) :
  let Q := a^2 + b^2 + c^2 + d^2 + a * c + b * d in
  Q ≠ 0 ∧ Q ≠ 1 ∧ Q ≠ -1 := 
by {
  let Q := a^2 + b^2 + c^2 + d^2 + a * c + b * d,
  sorry
}

end Q_not_in_set_of_3_l317_317547


namespace polly_average_move_time_l317_317581

theorem polly_average_move_time :
  ∀ (x : ℕ), 
  (∀ peter_moves polly_moves move_time total_moves total_time, 
   peter_moves = 15 ∧ polly_moves = 15 ∧ move_time = 40 ∧ total_moves = 30 ∧ total_time = 1020 ∧ 
   total_time = peter_moves * move_time + polly_moves * x 
  ) → 
  x = 28 :=
begin
  intros x h,
  obtain ⟨peter_moves, polly_moves, move_time, total_moves, total_time, h1, h2, h3, h4, h5, h6⟩ := h,
  simp [peter_moves, polly_moves, move_time, total_moves, total_time] at h6,
  have : 1020 = 15 * 40 + 15 * x,
  { rw [h6], },
  have : 15 * x = 1020 - 15 * 40,
  { rw [this], },
  norm_num at this,
  exact Eq.symm this,
end

end polly_average_move_time_l317_317581


namespace max_volume_of_rectangular_solid_l317_317665

theorem max_volume_of_rectangular_solid (x : ℝ) (h₁ : 0 < x) (h₂ : x < 3 / 2) :
  let length := 2 * x,
      height := (18 - 2 * x) / 4,
      volume := x * length * height
  in (-6 * x^3 + 9 * x^2 ≤ 3) :=
sorry

end max_volume_of_rectangular_solid_l317_317665


namespace shop_owner_percentage_profit_l317_317698

theorem shop_owner_percentage_profit
  (cost_price_per_kg : ℝ)
  (selling_price_per_kg : ℝ)
  (buying_cheat_percentage : ℝ)
  (selling_cheat_percentage : ℝ)
  (h_buying_cheat : buying_cheat_percentage = 20)
  (h_selling_cheat : selling_cheat_percentage = 20)
  (h_cost_price_when_buying : cost_price_per_kg = 100 / 1.2)
  (h_selling_price_when_selling : selling_price_per_kg = 100 / 0.8) : 
  ((selling_price_per_kg - cost_price_per_kg) / cost_price_per_kg) * 100 = 50 := 
by 
  have cost_price_per_kg_value : cost_price_per_kg = 83.33 := by sorry
  have selling_price_per_kg_value : selling_price_per_kg = 125 := by sorry
  calc
    ((125 - 83.33) / 83.33) * 100 = 50 := by sorry

end shop_owner_percentage_profit_l317_317698


namespace probability_all_dice_same_l317_317673

/--
Given four eight-sided dice, each numbered from 1 to 8 and each die landing independently,
prove that the probability of all four dice showing the same number is 1/512.
-/
theorem probability_all_dice_same :
  let n := 8 in       -- Number of sides on each dice
  let total_outcomes := n * n * n * n in  -- Total possible outcomes for four dice
  let favorable_outcomes := n in          -- Favorable outcomes (one same number for all dice)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 512 :=
by
  sorry

end probability_all_dice_same_l317_317673


namespace july_16_2010_is_wednesday_l317_317963

-- Define necessary concepts for the problem

def is_tuesday (d : ℕ) : Prop := (d % 7 = 2)
def day_after_n_days (d n : ℕ) : ℕ := (d + n) % 7

-- The statement we want to prove
theorem july_16_2010_is_wednesday (h : is_tuesday 1) : day_after_n_days 1 15 = 3 := 
sorry

end july_16_2010_is_wednesday_l317_317963


namespace sum_of_first_20_terms_l317_317447

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x
noncomputable def f' (x : ℝ) : ℝ := 2*x + 2
noncomputable def a_n (n : ℕ) : ℝ := 2 * n + 2
noncomputable def b_n (n : ℕ) : ℝ := 3^n
noncomputable def c_n (n : ℕ) : ℝ := n + 1 / ((n + 1) * (n + 2))

noncomputable def T_20 : ℝ := ∑ i in Finset.range 20, c_n (i + 1)

theorem sum_of_first_20_terms : T_20 = 2315 / 11 :=
  by sorry

end sum_of_first_20_terms_l317_317447


namespace mary_total_spent_on_clothing_l317_317571

variable {dollars : Type} [AddGroup dollars] [AddMonoidWithOne dollars] [Inhabited dollars]

def shirt_cost : dollars := 13.04
def jacket_cost : dollars := 12.27
def total_cost : dollars := 25.31

theorem mary_total_spent_on_clothing :
  shirt_cost + jacket_cost = total_cost :=
sorry

end mary_total_spent_on_clothing_l317_317571


namespace tangent_line_at_1_l317_317624

def f (x : ℝ) : ℝ := sorry

theorem tangent_line_at_1 (f' : ℝ → ℝ) (h1 : ∀ x, deriv f x = f' x) (h2 : ∀ y, 2 * 1 + y - 3 = 0) :
  f' 1 + f 1 = -1 :=
by
  sorry

end tangent_line_at_1_l317_317624


namespace sequence_sum_l317_317528

-- Defining the sequence terms
variables (J K L M N O P Q R S : ℤ)
-- Condition N = 7
def N_value : Prop := N = 7
-- Condition sum of any four consecutive terms is 40
def sum_of_consecutive : Prop := 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40

-- The main theorem stating J + S = 40 given the conditions
theorem sequence_sum (N_value : N = 7) (sum_of_consecutive : 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40) : 
  J + S = 40 := sorry

end sequence_sum_l317_317528


namespace max_good_elements_l317_317228

/-- A definition of a set with 2012 elements where the ratio of any two elements is never an integer,
and where "good elements" in the set are defined by the given property. We need to show that the
maximum number of good elements is 2010. -/

def SetWithGoodElements (S : Set ℤ) :=
  S.card = 2012 ∧
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y → ¬ (∃ k : ℤ, y = k * x)) ∧
  ∀ x ∈ S, (∃ y ∈ S, ∃ z ∈ S, y ≠ z ∧ x^2 ∣ (y * z))

theorem max_good_elements (S : Set ℤ) (hS : SetWithGoodElements S) : 
  ∃ T ⊆ S, T.card = 2010 ∧ ∀ x ∈ T, ∃ y ∈ T, z ∈ T, y ≠ z ∧ x^2 ∣ (y * z) :=
sorry

end max_good_elements_l317_317228


namespace lawn_mowing_rate_l317_317022

-- Definitions based on conditions
def total_hours_mowed : ℕ := 2 * 7
def money_left_after_expenses (R : ℕ) : ℕ := (14 * R) / 4

-- The problem statement
theorem lawn_mowing_rate (h : money_left_after_expenses R = 49) : R = 14 := 
sorry

end lawn_mowing_rate_l317_317022


namespace cart_distance_at_mn_l317_317990

theorem cart_distance_at_mn (m n : ℕ) (h : m > n) :
  let dist := (m - n) * n in ∀ k ∈ List.range (m * n),
   ( ( k % n = 0 → (k / n : ℤ) > 0 → k / n - k / m) = dist) :=
sorry

end cart_distance_at_mn_l317_317990


namespace sum_of_powers_eq_one_l317_317001

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 17)

theorem sum_of_powers_eq_one : (∑ k in Finset.range 1 17, ω ^ k) = 1 :=
by {
  have h : ω ^ 17 = 1 := by {
    rw [ω, Complex.exp_nat_mul, Complex.mul_div_cancel' (2 * Real.pi * Complex.I) (show (17 : ℂ) ≠ 0, by norm_cast; norm_num)],
    rw Complex.exp_cycle,
  },
  sorry
}

end sum_of_powers_eq_one_l317_317001


namespace palindrome_divisible_by_11_probability_l317_317737

-- Define a four-digit palindrome
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ let a := n / 1000 in let b := (n / 10) % 10 in n = 1001 * a + 110 * b

-- Define the divisibility by 11
def divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Define the probability function to determine if all valid palindromes are divisible by 11
noncomputable def probability_divisible_by_11 : ℚ :=
  let total_palindromes := 90 in
  let valid_palindromes := 90 in
  valid_palindromes / total_palindromes

-- Proof statement
theorem palindrome_divisible_by_11_probability :
  (∀ n, is_four_digit_palindrome n → divisible_by_11 n) → probability_divisible_by_11 = 1 :=
by
  intros h
  sorry

end palindrome_divisible_by_11_probability_l317_317737


namespace units_digit_and_sum_of_digits_l317_317689

-- Define the first five positive composite numbers
def composites : List ℕ := [4, 6, 8, 9, 10]

-- Define the product of the list of composite numbers
def product_composites : ℕ := List.foldl (· * ·) 1 composites

-- Define a function that computes the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Define a function that computes the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.data.foldr (λ c sum => sum + c.toNat - '0'.toNat) 0)

-- The theorem that needs to be proven
theorem units_digit_and_sum_of_digits :
  units_digit product_composites = 0 ∧ sum_of_digits product_composites = 18 :=
by 
  sorry

end units_digit_and_sum_of_digits_l317_317689


namespace volume_of_circumscribed_sphere_of_cube_l317_317448

theorem volume_of_circumscribed_sphere_of_cube (a : ℝ) (h : a = 1) : 
  (4 / 3) * Real.pi * ((Real.sqrt 3 / 2) ^ 3) = (Real.sqrt 3 / 2) * Real.pi :=
by sorry

end volume_of_circumscribed_sphere_of_cube_l317_317448


namespace minimum_convoy_time_l317_317038

noncomputable def convoy_time (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 12) then
  3480 / x
else if h : (12 < x ∧ x ≤ 20) then
  5 * x + (2880 / x) + 10
else
  0

theorem minimum_convoy_time : ∃ (x : ℝ), x = 20 ∧ convoy_time x = 254 :=
begin
  use 20,
  split,
  { refl },
  { norm_num,
    sorry }
end

end minimum_convoy_time_l317_317038


namespace minimum_w_value_l317_317416

theorem minimum_w_value : 
  (∀ x y : ℝ, w = 2*x^2 + 3*y^2 - 12*x + 9*y + 35) → 
  ∃ w_min : ℝ, w_min = 41 / 4 ∧ 
  (∀ x y : ℝ, 2*x^2 + 3*y^2 - 12*x + 9*y + 35 ≥ w_min) :=
by
  sorry

end minimum_w_value_l317_317416


namespace system_of_equations_solution_l317_317261

theorem system_of_equations_solution (x y : ℤ) (h1 : 2 * x + 5 * y = 26) (h2 : 4 * x - 2 * y = 4) : 
    x = 3 ∧ y = 4 :=
by
  sorry

end system_of_equations_solution_l317_317261


namespace smallest_k_odd_rightmost_digit_l317_317832

def factorial_ratio (n : ℕ) : ℕ := (n + 9)! / (n - 1)!

def rightmost_nonzero_digit (n : ℕ) : ℕ :=
  let significant_digit := (factorial_ratio n % 10) in
  if significant_digit = 0 then rightmost_nonzero_digit (n / 10) else significant_digit

theorem smallest_k_odd_rightmost_digit :
  let k := 78117 in 
  rightmost_nonzero_digit k = 9 :=
by
  sorry

end smallest_k_odd_rightmost_digit_l317_317832


namespace rectangular_garden_shorter_side_length_l317_317360

theorem rectangular_garden_shorter_side_length
  (a b : ℕ)
  (h1 : 2 * a + 2 * b = 46)
  (h2 : a * b = 108) :
  b = 9 :=
by 
  sorry

end rectangular_garden_shorter_side_length_l317_317360


namespace arrangement_of_11250_l317_317184

theorem arrangement_of_11250 : 
  let digits := [1, 1, 2, 5, 0]
  let total_count := 21
  let valid_arrangement (num : ℕ) : Prop := ∃ (perm : List ℕ), List.perm perm digits ∧ (num % 5 = 0) ∧ (num / 10000 ≥ 1)
  ∃ (count : ℕ), count = total_count ∧ 
  count = Nat.card {n // valid_arrangement n} := 
by 
  sorry

end arrangement_of_11250_l317_317184


namespace common_difference_arithmetic_sequence_l317_317451

theorem common_difference_arithmetic_sequence {d : ℝ} (a : ℕ → ℝ) 
  (h₁ : a 1 = 5) 
  (h₂ : a 6 + a 8 = 58)
  (h₃ : ∃ d, ∀ n, a (n + 1) = a n + d):
  d = 4 := 
begin
  sorry
end

end common_difference_arithmetic_sequence_l317_317451


namespace sequence_not_perfect_square_l317_317144

theorem sequence_not_perfect_square (n : ℕ) (N : ℕ → ℕ) :
  (∀ i, let d := [2, 0, 0, 1, 4, 0, 2, 0, 1, 5]
  in (N i = 2014 * 10^(n - i) + list.sum d) ∧ list.sum d = 15) →
  (∀ i, ¬ is_square (N i)) :=
by
  sorry

end sequence_not_perfect_square_l317_317144


namespace solve_quadratic_expr_l317_317149

theorem solve_quadratic_expr (x : ℝ) (h : 2 * x^2 - 5 = 11) : 
  4 * x^2 + 4 * x + 1 = 33 + 8 * Real.sqrt 2 ∨ 4 * x^2 + 4 * x + 1 = 33 - 8 * Real.sqrt 2 := 
by 
  sorry

end solve_quadratic_expr_l317_317149


namespace trigonometric_inequalities_equivalence_l317_317298

def cos_sq (θ : ℝ) := (Real.cos θ) ^ 2
def sin_sq (θ : ℝ) := (Real.sin θ) ^ 2

theorem trigonometric_inequalities_equivalence (θ1 θ2 θ3 θ4 x : ℝ) : 
  (cos_sq θ1 * cos_sq θ2 - (Real.sin θ1 * Real.sin θ2 - x) ^ 2 ≥ 0) ∧ 
  (cos_sq θ3 * cos_sq θ4 - (Real.sin θ3 * Real.sin θ4 - x) ^ 2 ≥ 0) ↔ 
  (sin_sq θ1 + sin_sq θ2 + sin_sq θ3 + sin_sq θ4 ≤ 2 * 
    (1 + (Real.sin θ1 * Real.sin θ2 * Real.sin θ3 * Real.sin θ4) + 
    (Real.cos θ1 * Real.cos θ2 * Real.cos θ3 * Real.cos θ4))) := 
by 
  sorry

end trigonometric_inequalities_equivalence_l317_317298


namespace range_of_f_on_0_4_range_of_a_l317_317982

section
  variable (f : ℝ → ℝ) (t : ℝ)

  -- Definition of function f and its behavior when t = 1
  def fx_when_t_is_one := f = fun x => x^2 - 2 * x + 2

  -- Problem 1: Range of values of f(x) on [0, 4] when t = 1
  theorem range_of_f_on_0_4 { f : ℝ → ℝ } (hx : fx_when_t_is_one f 1) :
    set.image f (set.Icc 0 4) = set.Icc 1 10 := sorry

  -- Problem 2: Range of a such that f(x) <= 5 for x ∈ [a, a+2]
  theorem range_of_a { f : ℝ → ℝ } (hx : fx_when_t_is_one f 1) 
                    (h : ∀ x, x ∈ set.Icc a (a + 2) → f x ≤ 5) :
    a ∈ set.Icc (-1) 1 := sorry
end

end range_of_f_on_0_4_range_of_a_l317_317982


namespace triangle_equilateral_l317_317502

-- Assume we are given side lengths a, b, and c of a triangle and angles A, B, and C in radians.
variables {a b c : ℝ} {A B C : ℝ}

-- We'll use the assumption that (a + b + c) * (b + c - a) = 3 * b * c and sin A = 2 * sin B * cos C.
axiom triangle_condition1 : (a + b + c) * (b + c - a) = 3 * b * c
axiom triangle_condition2 : Real.sin A = 2 * Real.sin B * Real.cos C

-- We need to prove that the triangle is equilateral.
theorem triangle_equilateral : (a = b) ∧ (b = c) ∧ (c = a) := by
  sorry

end triangle_equilateral_l317_317502


namespace product_is_correct_l317_317238

noncomputable def IKS := 521
noncomputable def KSI := 215
def product := 112015

theorem product_is_correct : IKS * KSI = product :=
by
  -- Proof yet to be constructed
  sorry

end product_is_correct_l317_317238


namespace optimal_response_l317_317377

theorem optimal_response (n : ℕ) (m : ℕ) (s : ℕ) (a_1 : ℕ) (a_2 : ℕ -> ℕ) (a_opt : ℕ):
  n = 100 → 
  m = 107 →
  (∀ i, i ≥ 1 ∧ i ≤ 99 → a_2 i = a_opt) →
  a_1 = 7 :=
by
  sorry

end optimal_response_l317_317377


namespace prob_AIE19_l317_317525

theorem prob_AIE19 :
  let vowels := 5, 
      non_vowels := 21, 
      digits := 10, 
      total_plates := vowels * vowels * non_vowels * (non_vowels - 1) * digits,
      favorable_outcomes := 1 in
  probability "AIE19" total_plates favorable_outcomes = 1 / 105000 :=
by
  let vowels := 5
  let non_vowels := 21
  let digits := 10
  let total_plates := vowels * vowels * non_vowels * (non_vowels - 1) * digits
  let favorable_outcomes := 1
  have total_plates_is_105000 : total_plates = 105000 :=
    by simp only [vowels, non_vowels, digits]; norm_num
  have probability "AIE19" total_plates favorable_outcomes = favorable_outcomes / total_plates := rfl
  simp only [favorable_outcomes, total_plates_is_105000]
  norm_num
  sorry -- Finish by providing the steps matching the proof in Lean

end prob_AIE19_l317_317525


namespace measure_of_smallest_angle_l317_317933

noncomputable def smallest_angle_in_parallelogram (max_angle : ℝ) (d : ℝ) : ℝ :=
  let a := (max_angle - 90) in a

theorem measure_of_smallest_angle (d : ℝ) (h : d ≠ 0) (max_angle : ℝ) (h_max : max_angle = 130) :
  smallest_angle_in_parallelogram max_angle d = 40 := 
by
  sorry

end measure_of_smallest_angle_l317_317933


namespace cos_theta_value_l317_317877

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (θ : ℝ)

def magnitude (v : V) : ℝ := real.sqrt (⟪v, v⟫)

-- Given conditions
noncomputable def conditions (a b : V) : Prop :=
  magnitude a = 4 ∧ magnitude b = 1 ∧ ∃ θ, magnitude (a - (2 • b)) = 4

theorem cos_theta_value (a b : V) (θ : ℝ) (h : conditions a b) :
  real.cos θ = 1 / 4 :=
sorry

end cos_theta_value_l317_317877


namespace simplify_cube_root_expression_l317_317597

-- statement of the problem
theorem simplify_cube_root_expression :
  ∛(8 + 27) * ∛(8 + ∛(27)) = ∛(385) :=
  sorry

end simplify_cube_root_expression_l317_317597


namespace prove_Problem_l317_317887

def sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, S n = ∑ i in finset.range (n + 1), a i

def given_condition (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, 3 * a n - 2 * S n = 1

def b_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = a n * real.logb 3 (a (n + 1))

def b_sum (b : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T n = ∑ i in finset.range (n + 1), b i

def a_sequence_correct (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = 3 ^ (n - 1)

def T_n_correct (T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T n = ((2 * n - 1) * 3 ^ n + 1) / 4

theorem prove_Problem (a S b T : ℕ → ℝ) :
  sequence_sum a S →
  given_condition a S →
  b_sequence a b →
  b_sum b T →
  a_sequence_correct a ∧ T_n_correct T :=
by
  intros h1 h2 h3 h4
  sorry

end prove_Problem_l317_317887


namespace mode_of_data_set_is_4_l317_317635

-- Definition of the data set
def data_set : List ℕ := [5, 4, 4, 3, 6, 2]

-- Mode calculation predicate
def mode (l : List ℕ) (m : ℕ) : Prop :=
  ∀ n ∈ l, count l n ≤ count l m

theorem mode_of_data_set_is_4 : mode data_set 4 :=
by
  -- Add your proof directives here if necessary, otherwise use "sorry" for skipping the proof.
  sorry

end mode_of_data_set_is_4_l317_317635


namespace possible_remainder_degrees_l317_317693

def degree (p : Polynomial ℝ) : ℕ :=
p.natDegree

theorem possible_remainder_degrees (q : Polynomial ℝ) :
  degree (-3 * X ^ 5 + 10 * X - 11) = 5 → ∃ d, d < 5 ∧ (degree (q % (-3 * X ^ 5 + 10 * X - 11)) = d ∧ d ∈ {0, 1, 2, 3, 4}) :=
by
  sorry 

end possible_remainder_degrees_l317_317693


namespace log_x_125_equal_approx_l317_317483

-- Defining the condition
def condition (x : ℝ) : Prop :=
  log 8 (5 * x) = 3

-- Stating the problem
theorem log_x_125_equal_approx : ∃ x : ℝ, condition x → log x 125 = (3 / 1.7227) :=
begin
  sorry
end

end log_x_125_equal_approx_l317_317483


namespace probability_second_three_star_probability_first_white_second_three_star_probability_first_white_conditional_probability_B_given_A_independence_A_B_X_probability_distribution_expectation_X_l317_317267

noncomputable theory
open probability

-- Define the problem's conditions
def balls : List (String × String) := [
  ("three-star", "white"), ("three-star", "white"),
  ("one-star", "white"), ("one-star", "white"),
  ("one-star", "white"), ("one-star", "white"),
  ("three-star", "yellow"), 
  ("one-star", "yellow"), ("one-star", "yellow")
]

def first_ball_is_white (b1 b2 : String × String) := b1.2 = "white"
def second_ball_is_three_star (b1 b2 : String × String) := b2.1 = "three-star"

-- Definitions of the events A and B
def A (b1 b2 : String × String) := first_ball_is_white b1 b2
def B (b1 b2 : String × String) := second_ball_is_three_star b1 b2
def AB (b1 b2 : String × String) := A b1 b2 ∧ B b1 b2

-- Main theorem statements

theorem probability_second_three_star :
  P(B) = 1/3 := sorry

theorem probability_first_white_second_three_star :
  P(AB) = 2/9 := sorry

theorem probability_first_white :
  P(A) = 2/3 := sorry

theorem conditional_probability_B_given_A :
  P(B|A) = 1/3 := sorry

theorem independence_A_B :
  P(AB) = P(A) * P(B) := sorry

-- Probability distribution and expectation of X

def X_distribution : string → ℚ :=
λ s, match s with
| "0" => 4/7
| "1" => 8/21
| "2" => 1/21
| _ => 0
end

theorem X_probability_distribution :
  ∀ x, P(X = x) = X_distribution x := sorry

theorem expectation_X :
  E(X) = 10/21 := sorry

end probability_second_three_star_probability_first_white_second_three_star_probability_first_white_conditional_probability_B_given_A_independence_A_B_X_probability_distribution_expectation_X_l317_317267


namespace total_cost_of_aquarium_l317_317778

variable (original_price discount_rate sales_tax_rate : ℝ)
variable (original_cost : original_price = 120)
variable (discount : discount_rate = 0.5)
variable (tax : sales_tax_rate = 0.05)

theorem total_cost_of_aquarium : 
  (original_price * (1 - discount_rate) * (1 + sales_tax_rate) = 63) :=
by
  rw [original_cost, discount, tax]
  sorry

end total_cost_of_aquarium_l317_317778


namespace min_weight_count_l317_317169

theorem min_weight_count (weights : Set ℝ) (h_distinct: ∀ (x y : ℝ), x ∈ weights → y ∈ weights → x ≠ y) 
                         (h_balance: ∀ (x y : ℝ), x ∈ weights → y ∈ weights → 
                                    (∃ z, z ∈ weights ∧ z ≠ x ∧ z ≠ y))
                         : weights.size ≥ 6 := 
sorry

end min_weight_count_l317_317169


namespace simplify_cube_root_expression_l317_317596

-- statement of the problem
theorem simplify_cube_root_expression :
  ∛(8 + 27) * ∛(8 + ∛(27)) = ∛(385) :=
  sorry

end simplify_cube_root_expression_l317_317596


namespace carpooling_plans_l317_317541

def last_digits (jia : ℕ) (friend1 : ℕ) (friend2 : ℕ) (friend3 : ℕ) (friend4 : ℕ) : Prop :=
  jia = 0 ∧ friend1 = 0 ∧ friend2 = 2 ∧ friend3 = 1 ∧ friend4 = 5

def total_car_plans : Prop :=
  ∀ (jia friend1 friend2 friend3 friend4 : ℕ),
    last_digits jia friend1 friend2 friend3 friend4 →
    (∃ num_ways : ℕ, num_ways = 64)

theorem carpooling_plans : total_car_plans :=
sorry

end carpooling_plans_l317_317541


namespace pink_roses_at_Mrs_Dawson_l317_317774

def rows : ℕ := 10
def roses_per_row : ℕ := 20
def red_ratio : ℚ := 1 / 2
def white_ratio : ℚ := 3 / 5

theorem pink_roses_at_Mrs_Dawson (R : ℕ) (T : ℕ) (red_ratio : ℚ) (white_ratio : ℚ) (hR : R = 10) (hT : T = 20) 
  (h_red_ratio : red_ratio = 1 / 2) (h_white_ratio : white_ratio = 3 / 5) : 
  (T - (red_ratio * T).toNat - ((white_ratio * (T - (red_ratio * T).toNat)).toNat)) * R = 40 := 
  sorry

end pink_roses_at_Mrs_Dawson_l317_317774


namespace coprime_exist_m_n_l317_317976

theorem coprime_exist_m_n (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_a : a ≥ 1) (h_b : b ≥ 1) :
  ∃ (m n : ℕ), m ≥ 1 ∧ n ≥ 1 ∧ a^m + b^n ≡ 1 [MOD a * b] :=
by
  use Nat.totient b, Nat.totient a
  sorry

end coprime_exist_m_n_l317_317976


namespace binomial_coefficient_x_term_l317_317524

/-- 
In the binomial expansion of (x - 2 / sqrt x)^7, the coefficient of the x term is 560.
-/
theorem binomial_coefficient_x_term :
  let expr := (x - 2 / real.sqrt x) in
  let n := 7 in
  let k := 4 in
  let general_term := (λ k: ℕ, (-2)^k * (nat.choose n k) * x^((14 - 3 * k) / 2)) in
  general_term k = 560 := 
by
  sorry

end binomial_coefficient_x_term_l317_317524


namespace common_area_of_translated_triangles_l317_317296

theorem common_area_of_translated_triangles :
  ∀ (h : ℝ), 
    (∀ a b c : ℝ, a^2 +  b^2 = c^2  ∧ a/c = 1/2 ∧ b/c = sqrt(3)/2 →
    a = 5 ∧ b = 5*sqrt(3)) →
    h = 10 →
    let y := 5*sqrt(3) in
    let overlap := y - 2 in
    let area := 1/2 * h * overlap in
    area = 25*sqrt(3) - 10 :=
by
  intros h hyp_cond hyp_val y
  calc area : ℭ := sorry   -- This is a placeholder for the actual proof


end common_area_of_translated_triangles_l317_317296


namespace graph_shift_right_pi_over_4_l317_317659

noncomputable def g : ℝ → ℝ := λ x, 2 * (Real.cos x)^2 - 1
noncomputable def f : ℝ → ℝ := λ x, 2 * Real.sin x * Real.cos x
def shift (h : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, h (x - a)

theorem graph_shift_right_pi_over_4 :
  (shift g (π / 4)) = f :=
by sorry

end graph_shift_right_pi_over_4_l317_317659


namespace inscribed_circle_radius_l317_317081

variables {α β c : ℝ}

noncomputable def radius_inscribed_circle (c : ℝ) (α β : ℝ) : ℝ :=
  c * (Real.sin (α / 2) * Real.sin (β / 2)) / Real.sin ((α + β) / 2)

theorem inscribed_circle_radius (h1 : 0 < c) (h2 : 0 < α) (h3 : 0 < β) (h4 : α + β < π) :
  ∃ r : ℝ, r = c * (Real.sin (α / 2) * Real.sin (β / 2)) / Real.sin ((α + β) / 2) :=
begin
  use radius_inscribed_circle c α β,
  simp [radius_inscribed_circle],
  sorry -- Proof steps will go here.
end

end inscribed_circle_radius_l317_317081


namespace problem_statement_l317_317782

noncomputable def larger_root_exceeds_smaller_root_difference : ℝ :=
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := -12
  let discriminant : ℝ := b * b - 4 * a * c
  let root1 : ℝ := (-b + Real.sqrt discriminant) / (2 * a)
  let root2 : ℝ := (-b - Real.sqrt discriminant) / (2 * a)
  max root1 root2 - min root1 root2

theorem problem_statement : larger_root_exceeds_smaller_root_difference = 5.5 := 
by 
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := -12
  let discriminant : ℝ := b * b - 4 * a * c
  have h_discriminant : discriminant = 121 := by simp [discriminant]
  let root1 : ℝ := (-b + Real.sqrt discriminant) / (2 * a)
  let root2 : ℝ := (-b - Real.sqrt discriminant) / (2 * a)
  have h_root1 : root1 = 1.5 := by simp [root1, Real.sqrt_def]
  have h_root2 : root2 = -4 := by simp [root2, Real.sqrt_def]
  show larger_root_exceeds_smaller_root_difference = 5.5 by
    unfold larger_root_exceeds_smaller_root_difference
    simp [h_root1, h_root2]
    exact rfl

end problem_statement_l317_317782


namespace solve_abs_inequality_l317_317417

theorem solve_abs_inequality (x : ℝ) : abs ((7 - 2 * x) / 4) < 3 ↔ -2.5 < x ∧ x < 9.5 := by
  sorry

end solve_abs_inequality_l317_317417


namespace sin_cos_sum_eq_one_or_neg_one_l317_317920

theorem sin_cos_sum_eq_one_or_neg_one (α : ℝ) (h : (Real.sin α)^4 + (Real.cos α)^4 = 1) : (Real.sin α + Real.cos α) = 1 ∨ (Real.sin α + Real.cos α) = -1 :=
sorry

end sin_cos_sum_eq_one_or_neg_one_l317_317920


namespace julie_hours_per_week_during_school_year_l317_317962

theorem julie_hours_per_week_during_school_year
  (summer_hours_per_week : ℕ)
  (summer_weeks : ℕ)
  (summer_earnings : ℕ)
  (school_weeks : ℕ)
  (school_earnings_needed : ℕ)
  (reduction_rate : ℚ) :
  summer_hours_per_week = 60 →
  summer_weeks = 10 →
  summer_earnings = 6000 →
  school_weeks = 40 →
  school_earnings_needed = 8000 →
  reduction_rate = 0.20 →
  let summer_hourly_wage := (summer_earnings : ℚ) / (summer_hours_per_week * summer_weeks : ℕ)
  in let school_hourly_wage := summer_hourly_wage * (1 - reduction_rate)
  in let total_school_hours := (school_earnings_needed : ℚ) / school_hourly_wage
  in let hours_per_week := total_school_hours / (school_weeks : ℚ)
  in hours_per_week = 25 :=
by
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  -- The following steps are the logical simplifications to reach the conclusion.
  sorry

end julie_hours_per_week_during_school_year_l317_317962


namespace log_f_of_2_eq_half_l317_317885

theorem log_f_of_2_eq_half :
  (∃ α : ℝ, f : ℝ → ℝ, (∀ x, f x = x ^ α) ∧ f (1/2) = (Real.sqrt 2) / 2) →
  Real.logb 2 (f 2) = 1/2 :=
by
  sorry

end log_f_of_2_eq_half_l317_317885


namespace pizza_pieces_remaining_l317_317167

-- Definitions based on conditions
def total_people : ℕ := 25
def fraction_eating_pizza : ℚ := 7 / 10
def total_pizza_pieces : ℕ := 80
def pieces_per_person : ℕ := 5

-- Theorem stating the output
theorem pizza_pieces_remaining :
  let people_eating_pizza := (total_people * fraction_eating_pizza).floor.to_nat,
      people_fully_served := min people_eating_pizza (total_pizza_pieces / pieces_per_person)
  in total_pizza_pieces - people_fully_served * pieces_per_person = 0 :=
by
  sorry  -- Proof to be filled in

end pizza_pieces_remaining_l317_317167


namespace find_angle_QPR_l317_317170

-- Define the given conditions
variables (A B C I P Q R : Point)
variables (θ : ℝ) (triangle_ABC : Triangle A B C) 
variables (I_incenter : IsIncenter I triangle_ABC)
variables (incircle : Incircle triangle_ABC)
variables (AI_bisects_angle : Bisects_angle (segment A I) (Angle A B C))
variables (BI_bisects_angle : Bisects_angle (segment B I) (Angle B A C))
variables (CI_bisects_angle : Bisects_angle (segment C I) (Angle C A B))

-- Define the intersections of AI, BI, and CI with the incircle
variables (intersection_AI : Intersects (segment A I) incircle P)
variables (intersection_BI : Intersects (segment B I) incircle Q)
variables (intersection_CI : Intersects (segment C I) incircle R)
variables (angle_BAC_condition : θ = 40)

-- Define the main theorem statement
theorem find_angle_QPR : ∀ (triangle_ABC : Triangle A B C), IsIncenter I triangle_ABC →
    ∀ (P Q R : Point), Intersects (segment A I) incircle P →
    Intersects (segment B I) incircle Q →
    Intersects (segment C I) incircle R →
    ∀ (θ : ℝ), θ = 40 →
    ∠ Q P R = 20 := by 
  sorry

end find_angle_QPR_l317_317170


namespace no_integer_roots_quadratic_l317_317899

theorem no_integer_roots_quadratic (a b : ℤ) : 
  ∀ u : ℤ, ¬(u^2 + 3*a*u + 3*(2 - b^2) = 0) := 
by
  sorry

end no_integer_roots_quadratic_l317_317899


namespace directrix_of_parabola_l317_317412

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = -3 * x^2 + 6 * x - 5 → y = -35 / 18 := by
  sorry

end directrix_of_parabola_l317_317412


namespace max_sn_at_16_l317_317644

variable {a : ℕ → ℝ} -- the sequence a_n is represented by a

-- Conditions given in the problem
def isArithmetic (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def bn (a : ℕ → ℝ) (n : ℕ) : ℝ := a n * a (n + 1) * a (n + 2)

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := (Finset.range n).sum (bn a)

-- Condition: a_{12} = 3/8 * a_5 and a_12 > 0
def specificCondition (a : ℕ → ℝ) : Prop := a 12 = (3 / 8) * a 5 ∧ a 12 > 0

-- The theorem to prove that for S n, the maximum value is reached at n = 16
theorem max_sn_at_16 (a : ℕ → ℝ) (h_arithmetic : isArithmetic a) (h_condition : specificCondition a) :
  ∀ n : ℕ, Sn a n ≤ Sn a 16 := sorry

end max_sn_at_16_l317_317644


namespace samuel_remaining_money_l317_317708

-- Define the total amount
def total_amount : ℕ := 240

-- Define Samuel's share as three-fourths of the total amount
def samuel_share : ℕ := (3 * total_amount) / 4

-- Define the amount Samuel spent on drinks as one-fifth of the total amount
def amount_spent_on_drinks : ℕ := total_amount / 5

-- Prove the remaining amount Samuel has
theorem samuel_remaining_money : samuel_share - amount_spent_on_drinks = 132 := 
by
  -- Definitions
  have h1 : total_amount = 240 := rfl
  have h2 : samuel_share = 180 := by calc
    samuel_share = (3 * 240) / 4 := by { rw h1 }
    ... = 180 := by norm_num
  have h3 : amount_spent_on_drinks = 48 := by calc
    amount_spent_on_drinks = 240 / 5 := by { rw h1 }
    ... = 48 := by norm_num

  -- Proof
  calc
    samuel_share - amount_spent_on_drinks = 180 - 48 := by { rw [h2, h3] }
    ... = 132 := by norm_num

end samuel_remaining_money_l317_317708


namespace part_a_part_b_l317_317390

-- Part (a): Prove that for N = a^2 + 2, the equation has positive integral solutions for infinitely many a.
theorem part_a (N : ℕ) (a : ℕ) (x y z t : ℕ) (hx : x = a * (a^2 + 2)) (hy : y = a) (hz : z = 1) (ht : t = 1) :
  (∃ (N : ℕ), ∀ (a : ℕ), ∃ (x y z t : ℕ),
    x^2 + y^2 + z^2 + t^2 = N * x * y * z * t + N ∧
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0) :=
sorry

-- Part (b): Prove that for N = 4^k(8m + 7), the equation has no positive integral solutions.
theorem part_b (N : ℕ) (k m : ℕ) (x y z t : ℕ) (hN : N = 4^k * (8 * m + 7)) :
  ¬ (x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ x^2 + y^2 + z^2 + t^2 = N * x * y * z * t + N) :=
sorry

end part_a_part_b_l317_317390


namespace test_scores_l317_317939

noncomputable def expected_score (n : ℕ) (points : ℕ) (p : ℝ) : ℝ :=
  n * (points * p)

noncomputable def variance_score (n : ℕ) (points : ℕ) (p : ℝ) : ℝ :=
  n * ((points^2 * p) - (points * p)^2)

theorem test_scores (n : ℕ) (points : ℕ) (p : ℝ) (max_score : ℕ) :
  n = 25 ∧ points = 4 ∧ p = 0.8 ∧ max_score = 100 →
  expected_score n points p = 80 ∧ variance_score n points p = 64 :=
by
  intros h
  cases h with hn hrest
  cases hrest with hpqr hmax
  cases hpqr with hpq hp
  split
  {
    simp [expected_score, hn, hpq, hp],
    norm_num,
  },
  {
    simp [variance_score, hn, hpq, hp],
    norm_num,
  }

end test_scores_l317_317939


namespace total_pink_roses_l317_317776

theorem total_pink_roses (rows : ℕ) (roses_per_row : ℕ) (half : ℚ) (three_fifths : ℚ) :
  rows = 10 →
  roses_per_row = 20 →
  half = 1/2 →
  three_fifths = 3/5 →
  (rows * (roses_per_row - roses_per_row * half - (roses_per_row * (1 - half) * three_fifths))) = 40 :=
begin
  sorry
end

end total_pink_roses_l317_317776


namespace intersection_A_complement_is_2_4_l317_317128

-- Declare the universal set U, set A, and set B
def U : Set ℕ := { 1, 2, 3, 4, 5, 6, 7 }
def A : Set ℕ := { 2, 4, 5 }
def B : Set ℕ := { 1, 3, 5, 7 }

-- Define the complement of set B with respect to U
def complement_U_B : Set ℕ := { x ∈ U | x ∉ B }

-- Define the intersection of set A and the complement of set B
def intersection_A_complement_U_B : Set ℕ := { x ∈ A | x ∈ complement_U_B }

-- State the theorem
theorem intersection_A_complement_is_2_4 : 
  intersection_A_complement_U_B = { 2, 4 } := 
by
  sorry

end intersection_A_complement_is_2_4_l317_317128


namespace range_of_a_l317_317452

noncomputable def f (x a : ℝ) : ℝ := (Real.sqrt x) / (x^3 - 3 * x + a)

theorem range_of_a (a : ℝ) :
    (∀ x, 0 ≤ x → x^3 - 3 * x + a ≠ 0) ↔ 2 < a := 
by 
  sorry

end range_of_a_l317_317452


namespace probability_of_multiple_2_3_7_l317_317371

open Nat

def count_multiples (n m : ℕ) : ℕ :=
  m / n

def inclusion_exclusion (a b c : ℕ) (ab ac bc abc : ℕ) : ℕ :=
  a + b + c - ab - ac - bc + abc

def probability_multiple_2_3_7 : ℚ :=
  let total_cards := 150
  let multiples_2 := count_multiples 2 total_cards
  let multiples_3 := count_multiples 3 total_cards
  let multiples_7 := count_multiples 7 total_cards
  let multiples_6 := count_multiples 6 total_cards
  let multiples_14 := count_multiples 14 total_cards
  let multiples_21 := count_multiples 21 total_cards
  let multiples_42 := count_multiples 42 total_cards
  let total_multiples := inclusion_exclusion multiples_2 multiples_3 multiples_7 multiples_6 multiples_14 multiples_21 multiples_42
  (total_multiples : ℚ) / (total_cards : ℚ)

theorem probability_of_multiple_2_3_7 :
  probability_multiple_2_3_7 = 107 / 150 :=
  sorry

end probability_of_multiple_2_3_7_l317_317371


namespace sum_fourth_powers_l317_317917

theorem sum_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2)
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a^4 + b^4 + c^4 = 25 / 6 :=
by sorry

end sum_fourth_powers_l317_317917


namespace sequence_not_perfect_square_l317_317145

-- Define the sequence of numbers.
def N (i : ℕ) : ℕ := 
  let zeros := String.mk $ List.replicate (i - 1) '0'
  ("2014" ++ zeros ++ "2015").to_nat

-- Main theorem stating that none of the numbers in the sequence are perfect squares.
theorem sequence_not_perfect_square (i : ℕ) : ¬ ∃ x : ℕ, x * x = N i := by
  sorry

end sequence_not_perfect_square_l317_317145


namespace bond_selling_price_l317_317700

def bond_face_value : ℝ := 5000
def bond_interest_rate : ℝ := 0.06
def interest_approx : ℝ := bond_face_value * bond_interest_rate
def selling_price_interest_rate : ℝ := 0.065
def approximate_selling_price : ℝ := 4615.38

theorem bond_selling_price :
  interest_approx = selling_price_interest_rate * approximate_selling_price :=
sorry

end bond_selling_price_l317_317700


namespace part_a_part_b_l317_317342

-- Part (a)
theorem part_a (a b c : ℚ) (z : ℚ) (h : a * z^2 + b * z + c = 0) (n : ℕ) (hn : n > 0) :
  ∃ f : ℚ → ℚ, z = f (z^n) :=
sorry

-- Part (b)
theorem part_b (x : ℚ) (h : x ≠ 0) :
  x = (x^3 + (x + 1/x)) / ((x + 1/x)^2 - 1) :=
sorry

end part_a_part_b_l317_317342


namespace min_value_is_four_l317_317766

theorem min_value_is_four {x : ℝ} : 
  ∃ x, 3^x + 4 * 3^(-x) = 4 ∧ x = Real.log 2 / Real.log 3 :=
by 
  sorry

end min_value_is_four_l317_317766


namespace unique_cube_division_into_tetrahedra_l317_317255

theorem unique_cube_division_into_tetrahedra
    (A B C D A' B' C' D' : ℝ³)
    (is_cube : is_cube (A B C D A' B' C' D'))
    (volume_cube : volume (A B C D A' B' C' D') = 1)
    (divided_into_tetrahedra : divided_into_tetrahedra (A B C D A' B' C' D') [Δ₁, Δ₂, Δ₃, Δ₄, Δ₅]) :
    unique_division (A B C D A' B' C' D') [Δ₁, Δ₂, Δ₃, Δ₄, Δ₅] := sorry

end unique_cube_division_into_tetrahedra_l317_317255


namespace shapes_form_symmetric_figure_l317_317037

structure Shape := (id : ℕ)
structure Figure := (shapes : list Shape) (axis_of_symmetry : bool)

axiom shapes : list Shape -- There are three given shapes
axiom shape1 : Shape
axiom shape2 : Shape
axiom shape3 : Shape

def use_each_shape_once (fig : Figure) : Prop := 
  fig.shapes.length = 3 ∧ 
  (∀ s, s ∈ fig.shapes → s = shape1 ∨ s = shape2 ∨ s = shape3)

def has_axis_of_symmetry (fig : Figure) : Prop := fig.axis_of_symmetry

theorem shapes_form_symmetric_figure : 
  ∃ fig : Figure, use_each_shape_once fig ∧ has_axis_of_symmetry fig := 
sorry

end shapes_form_symmetric_figure_l317_317037


namespace find_lambda_l317_317438

theorem find_lambda (lambda : ℝ) (h1 : λ ≠ 0) (h2 : 5 + 3 * lambda = 4 + 5 * lambda) : lambda = 1 / 2 := by
  sorry

end find_lambda_l317_317438


namespace problem_proof_l317_317966

noncomputable def irreducible_polynomial_exists (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ b : ℕ → ℕ, (∀ i : ℕ, 0 ≤ i ∧ i ≤ n → a i ≤ b i ∧ b i ≤ 2 * a i) ∧ 
                irreducible (polynomial.C (b 0) + 
                             polynomial.C (b 1) * polynomial.X + 
                             ⋯ +  (polynomial.C (b n) * polynomial.X ^ n))

theorem problem_proof (a : ℕ → ℕ) (n : ℕ) (h0 : ∀ i : ℕ, 0 ≤ i ∧ i ≤ n → 0 ≤ a i) : 
  irreducible_polynomial_exists a n :=
sorry

end problem_proof_l317_317966


namespace find_a1_a7_l317_317433

variable {a n : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k n, a (k + n) = a k + n * d

theorem find_a1_a7 
  (a1 : ℝ) (d : ℝ)
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h1 : a 3 + a 5 = 14)
  (h2 : a 2 * a 6 = 33) :
  a 1 * a 7 = 13 := 
sorry

end find_a1_a7_l317_317433


namespace area_CDE_l317_317243

variables (ABC : Type) [triangle ABC]
variables (A B C D E F : ABC)
variables (AD AC : Line ABC) (BD BE : Line ABC) (AE : Line ABC)

-- Areas of triangles
def area (t : Triangle ABC) : ℝ := sorry

-- Given conditions
axiom h1 : area (triangle A D F) = 1 / 2
axiom h2 : area (triangle A B F) = 1
axiom h3 : area (triangle B E F) = 1 / 4

-- Target statement: Area of CDE
theorem area_CDE : area (triangle C D E) = 15 / 56 := sorry

end area_CDE_l317_317243


namespace max_happy_times_l317_317232

theorem max_happy_times (weights : Fin 2021 → ℝ) (unique_mass : Function.Injective weights) : 
  ∃ max_happy : Nat, max_happy = 673 :=
by
  sorry

end max_happy_times_l317_317232


namespace equally_spaced_complex_numbers_l317_317914

theorem equally_spaced_complex_numbers (n : ℕ) (hz : 2 ≤ n) (z : ℕ → ℂ) 
  (h_mag : ∀ k, |z k| = 1) (h_sum : ∑ i in Finset.range n, z i = 2) 
  : (∃ m, n = 2 ∧ m = 1) :=
by
  -- The statement that the count of such n's is 1.
  existsi 1
  split
  . exact eq.refl 2
  . exact eq.refl 1

-- Note: This theorem statement asserts the existence of m = 1 when n = 2, proving that only such n = 2 exists.

end equally_spaced_complex_numbers_l317_317914


namespace area_ratio_triangle_trapezoid_l317_317195

theorem area_ratio_triangle_trapezoid
  (A B C D E : Type)
  [plane_geometry A B C D E]
  (ABCD_is_trapezoid : is_trapezoid A B C D)
  (AB_length : length A B = 10)
  (CD_length : length C D = 20)
  (E_is_intersection : intersects_extended_legs A B C D E) :
  area_ratio (triangle DEC A B C D E) (trapezoid ABCD A B C D) = 2 / 3 :=
sorry

end area_ratio_triangle_trapezoid_l317_317195


namespace area_ratio_OAB_OBC_l317_317558

variables {O A B C : Type*}
variables [AddCommGroup O] [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables [Module ℝ O] [Module ℝ A] [Module ℝ B] [Module ℝ C]

theorem area_ratio_OAB_OBC
  (O : point) (A B C : triangle)
  (h : (O ∈ A ∧ B ∧ C) ∧ (vector_OA O A = 1/3 * vector_AB A B + 1/4 * vector_AC A C))
  :
  area(O, A, B) / area(O, B, C) = 3 / 5 := sorry

end area_ratio_OAB_OBC_l317_317558


namespace evaluate_expression_eq_l317_317824

theorem evaluate_expression_eq : 3 + (-3)^(3 - (-2)) = -240 := sorry

end evaluate_expression_eq_l317_317824


namespace speed_conversion_l317_317806

theorem speed_conversion (v : ℚ) (h : v = 9/36) : v * 3.6 = 0.9 := by
  sorry

end speed_conversion_l317_317806


namespace find_x1971_l317_317227

noncomputable def sequence_x (x : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 2 → 3 * x n - x (n - 1) = n

noncomputable def initial_condition (x : ℕ → ℝ) : Prop :=
| x 1 | < 1971

theorem find_x1971 (x : ℕ → ℝ) (h_seq : sequence_x x) (h_init : initial_condition x) :
  x 1971 = 985.250000 :=
sorry

end find_x1971_l317_317227


namespace largest_number_of_hcf_lcm_l317_317701

theorem largest_number_of_hcf_lcm (HCF : ℕ) (factor1 factor2 : ℕ) (n1 n2 : ℕ) (largest : ℕ) 
  (h1 : HCF = 52) 
  (h2 : factor1 = 11) 
  (h3 : factor2 = 12) 
  (h4 : n1 = HCF * factor1) 
  (h5 : n2 = HCF * factor2) 
  (h6 : largest = max n1 n2) : 
  largest = 624 := 
by 
  sorry

end largest_number_of_hcf_lcm_l317_317701


namespace probability_square_product_correct_l317_317663

noncomputable def probability_square_product : ℚ := 23 / 96

theorem probability_square_product_correct :
  let T := ({1, 2, ..., 12} : finset ℕ),
      D := ({1, 2, ..., 8} : finset ℕ),
      event (t d : ℕ) := (t * d) ∈ {n | ∃ m, m * m = n} in
  (T.product D).card = 96 ∧
  (T.product D).filter (λ ⟨t, d⟩, event t d).card = 23 →
  (23 / 96 : ℚ) = probability_square_product :=
begin
  intros T D event total_count favorable_count,
  rw [total_count, favorable_count],
  exact rfl
end

end probability_square_product_correct_l317_317663


namespace pyramid_properties_l317_317748

theorem pyramid_properties (l w h : ℝ) 
  (base_length : l = 12) 
  (base_width : w = 8) 
  (height : h = 15) :
  let d := real.sqrt (l^2 + w^2) in
  let half_d := d / 2 in
  let s := real.sqrt (h^2 + half_d^2) in
  let total_edge_length := 2 * (l + w) + 4 * s in
  let volume := (1/3:ℝ) * (l * w) * h in
  total_edge_length = 40 + 4 * real.sqrt 277 ∧ 
  volume = 480 :=
by 
  sorry

end pyramid_properties_l317_317748


namespace prism_faces_l317_317742

theorem prism_faces (E L : ℕ) (hE : E = 21) (hFormula : E = 3 * L) : 2 + L = 9 :=
by
  rw [hE, hFormula]
  have hL : L = 7 := sorry 
  rw [hL]
  rfl

end prism_faces_l317_317742


namespace compute_a_l317_317441

theorem compute_a (a b : ℚ) 
  (h_root1 : (-1:ℚ) - 5 * (Real.sqrt 3) = -1 - 5 * (Real.sqrt 3))
  (h_rational1 : (-1:ℚ) + 5 * (Real.sqrt 3) = -1 + 5 * (Real.sqrt 3))
  (h_poly : ∀ x, x^3 + a*x^2 + b*x + 48 = 0) :
  a = 50 / 37 :=
by
  sorry

end compute_a_l317_317441


namespace angles_with_same_terminal_side_pi_div_3_l317_317646

noncomputable def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + 2 * k * Real.pi

theorem angles_with_same_terminal_side_pi_div_3 :
  { α : ℝ | same_terminal_side α (Real.pi / 3) } =
  { α : ℝ | ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3 } :=
by
  sorry

end angles_with_same_terminal_side_pi_div_3_l317_317646


namespace exist_floors_eq_l317_317967

theorem exist_floors_eq (a : FinSeq ℕ) (m : ℕ) (h_pos : ∀ i, i ∈ a → 0 < i) :
  ∃ (b c N : ℕ), ∀ n : ℕ, n > N →
    ⌊∑ i in a, real.sqrt (n + i)⌉ = ⌊real.sqrt (m^2 * n + (sum a) * m - 1)⌉ := by
  sorry

end exist_floors_eq_l317_317967


namespace option_B_correct_l317_317430

noncomputable def f : ℝ → ℝ := sorry

theorem option_B_correct (a b : ℝ)
  (h1 : ∀ x : ℝ, f(x) ≥ abs(x))
  (h2 : ∀ x : ℝ, f(x) ≥ 2^x)
  (h3 : f(a) ≤ 2^b) : a ≤ b := by
  sorry

end option_B_correct_l317_317430


namespace find_money_of_Kent_l317_317764

variable (Alison Brittany Brooke Kent : ℝ)

def money_relations (h1 : Alison = 4000)
    (h2 : Alison = Brittany / 2)
    (h3 : Brittany = 4 * Brooke)
    (h4 : Brooke = 2 * Kent) : Prop :=
  Kent = 1000

theorem find_money_of_Kent
  {Alison Brittany Brooke Kent : ℝ}
  (h1 : Alison = 4000)
  (h2 : Alison = Brittany / 2)
  (h3 : Brittany = 4 * Brooke)
  (h4 : Brooke = 2 * Kent) :
  money_relations Alison Brittany Brooke Kent h1 h2 h3 h4 :=
by 
  sorry

end find_money_of_Kent_l317_317764


namespace no_perfect_squares_in_sequence_l317_317140

theorem no_perfect_squares_in_sequence : 
  ∀ N ∈ ({20142015, 201402015, 2014002015, 20140002015, 201400002015} : set ℕ), 
  ¬ (∃ k : ℕ, N = k^2) := 
by
  sorry

end no_perfect_squares_in_sequence_l317_317140


namespace intersection_points_count_l317_317388

open Real

def fraction_part (x : ℝ) : ℝ := x - floor x

def f (x y : ℝ) : Prop := (fraction_part x)^2 + y^2 = fraction_part x + 1

def g (x y : ℝ) : Prop := y = (1.0 / 3.0) * x + 1

theorem intersection_points_count : 
  (∀ p : ℕ, f (↑p) (1.0 / 3.0 * (↑p: ℝ) + 1) → g (↑p) (1.0 / 3.0 * (↑p: ℝ) + 1)) →
  (∀ q : ℕ, f (-↑q) (1.0 / 3.0 * (-↑q: ℝ) + 1) → g (-↑q) (1.0 / 3.0 * (-↑q: ℝ) + 1)) →
  (2 * (5 + 5 + 1) - 1) = 21 :=
by
  sorry

end intersection_points_count_l317_317388


namespace g_50_eq_119_l317_317026

noncomputable def g : ℕ → ℕ 
| x := if (∃ n : ℕ, real.logBase 3 x = n) then (real.logBase 3 x).to_nat
       else 1 + g (x + 2)

theorem g_50_eq_119 : g 50 = 119 := by
  sorry

end g_50_eq_119_l317_317026


namespace arrangement_of_11250_l317_317183

theorem arrangement_of_11250 : 
  let digits := [1, 1, 2, 5, 0]
  let total_count := 21
  let valid_arrangement (num : ℕ) : Prop := ∃ (perm : List ℕ), List.perm perm digits ∧ (num % 5 = 0) ∧ (num / 10000 ≥ 1)
  ∃ (count : ℕ), count = total_count ∧ 
  count = Nat.card {n // valid_arrangement n} := 
by 
  sorry

end arrangement_of_11250_l317_317183


namespace smallest_positive_perfect_square_divisible_by_5_and_6_l317_317687

theorem smallest_positive_perfect_square_divisible_by_5_and_6 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m * m) ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (∀ k : ℕ, (∃ p : ℕ, k = p * p) ∧ k % 5 = 0 ∧ k % 6 = 0 → n ≤ k) := 
sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_l317_317687


namespace rectangle_invalid_perimeter_l317_317358

-- Define conditions
def positive_integer (n : ℕ) : Prop := n > 0

-- Define the rectangle with given area
def area_24 (length width : ℕ) : Prop := length * width = 24

-- Define the function to calculate perimeter for given length and width
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- The theorem to prove
theorem rectangle_invalid_perimeter (length width : ℕ) (h₁ : positive_integer length) (h₂ : positive_integer width) (h₃ : area_24 length width) : 
  (perimeter length width) ≠ 36 :=
sorry

end rectangle_invalid_perimeter_l317_317358


namespace crayons_per_person_l317_317836

theorem crayons_per_person (total_crayons : ℕ) (total_people : ℕ) (h1 : total_crayons = 24) (h2 : total_people = 3) :
  total_crayons / total_people = 8 := by
  rw [h1, h2]
  norm_num
  sorry

end crayons_per_person_l317_317836


namespace number_of_players_l317_317613
-- Importing the necessary library

-- Define the number of games formula for the tournament
def number_of_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- The theorem to prove the number of players given the conditions
theorem number_of_players (n : ℕ) (h : number_of_games n = 306) : n = 18 :=
by
  sorry

end number_of_players_l317_317613


namespace arrangement_of_digits_11250_l317_317181

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then
    1
  else
    n * factorial (n - 1)

def number_of_arrangements (digits : List ℕ) : ℕ :=
  let number_ends_in_0 := factorial 4 / factorial 2
  let number_ends_in_5 := 3 * (factorial 3 / factorial 2)
  number_ends_in_0 + number_ends_in_5

theorem arrangement_of_digits_11250 :
  number_of_arrangements [1, 1, 2, 5, 0] = 21 :=
by
  sorry

end arrangement_of_digits_11250_l317_317181


namespace find_x_l317_317561

def star (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_x (x : ℝ) (h : star 4 x = 52) : x = 8 :=
by
  sorry

end find_x_l317_317561


namespace exists_diagonal_cut_triangle_l317_317585

open Set

def is_convex_hexagon (H : Set (ℝ × ℝ)) : Prop :=
  ∃ A B C D E F : ℝ × ℝ,
    Convex (convexHull ℝ ({A, B, C, D, E, F} : Set (ℝ × ℝ))) ∧
    H = convexHull ℝ ({A, B, C, D, E, F} : Set (ℝ × ℝ))

noncomputable def area_hexagon (H : Set (ℝ × ℝ)) : ℝ := sorry

theorem exists_diagonal_cut_triangle
  (H : Set (ℝ × ℝ))
  (h_convex : is_convex_hexagon H)
  (S : ℝ)
  (h_area : S = area_hexagon H) :
  ∃ diag : Set (ℝ × ℝ), ∃ T : Set (ℝ × ℝ),
    is_triangle T ∧
    (∀ A B C : ℝ × ℝ, A ∈ diag ∧ B ∈ diag ∧ C ∈ T → A ≠ B) ∧
    area_triangle T ≤ S / 6 := sorry

end exists_diagonal_cut_triangle_l317_317585


namespace three_x_plus_four_l317_317493

theorem three_x_plus_four (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  sorry

end three_x_plus_four_l317_317493


namespace right_triangle_hypotenuse_length_l317_317514

theorem right_triangle_hypotenuse_length (P Q R S T : ℝ) (y : ℝ) (hPQ : P ≠ Q) 
  (hT : ∃ PS SQ PT TQ : ℝ, PS = SQ ∧ PT = TQ / 3 ∧ PS + SQ = PQ ∧ PT + TQ = PQ)
  (hRS_RT : R = S + sin y ∧ R = T + cos y) (hy : 0 < y ∧ y < π / 2) :
    PQ = 8 * sqrt 5 / 15 :=
by 
  sorry

end right_triangle_hypotenuse_length_l317_317514


namespace ratio_both_basketball_volleyball_l317_317508

variable (total_students : ℕ) (play_basketball : ℕ) (play_volleyball : ℕ) (play_neither : ℕ) (play_both : ℕ)

theorem ratio_both_basketball_volleyball (h1 : total_students = 20)
    (h2 : play_basketball = 20 / 2)
    (h3 : play_volleyball = (2 * 20) / 5)
    (h4 : play_neither = 4)
    (h5 : total_students - play_neither = play_basketball + play_volleyball - play_both) :
    play_both / total_students = 1 / 10 :=
by
  sorry

end ratio_both_basketball_volleyball_l317_317508


namespace solve_system_l317_317612

def system_of_equations (x y : ℝ) : Prop :=
  (4 * (x - y) = 8 - 3 * y) ∧ (x / 2 + y / 3 = 1)

theorem solve_system : ∃ x y : ℝ, system_of_equations x y ∧ x = 2 ∧ y = 0 := 
  by
  sorry

end solve_system_l317_317612


namespace construct_square_on_parallel_lines_l317_317804

theorem construct_square_on_parallel_lines 
  (a b c : Set (ℝ × ℝ)) 
  (ha : ∃ p : ℝ × ℝ, p ∈ a)
  (hb : ∃ p : ℝ × ℝ, p ∈ b)
  (hc : ∃ p : ℝ × ℝ, p ∈ c)
  (parallel_a : ∀ x y ∈ a, x ≠ y → (y.2 - x.2) / (y.1 - x.1) = (hd.2 - ha.2) / (hd.1 - ha.1))
  (parallel_b : ∀ x y ∈ b, x ≠ y → (y.2 - x.2) / (y.1 - x.1) = (he.2 - hb.2) / (he.1 - hb.1))
  (parallel_c : ∀ x y ∈ c, x ≠ y → (y.2 - x.2) / (y.1 - x.1) = (hf.2 - hc.2) / (hf.1 - hc.1))
  (between : ∀ p ∈ a, ∀ q ∈ b, ∀ r ∈ c, p.2 < q.2 ∧ q.2 < r.2):
  ∃ (A B C D : ℝ × ℝ), A ∈ a ∧ B ∈ b ∧ C ∈ c ∧ 
  (dist A B) = (dist B C) ∧ (dist B C) = (dist C D) ∧ (dist C D) = (dist D A) ∧ 
  ∠ A B C = 90 ∧ ∠ B C D = 90 ∧ ∠ C D A = 90 ∧ ∠ D A B = 90 :=
sorry

end construct_square_on_parallel_lines_l317_317804


namespace general_formula_compare_Tn_l317_317850

open scoped BigOperators

-- Define the sequence {a_n} and its sum S_n
noncomputable def aSeq (n : ℕ) : ℕ := n + 1
noncomputable def S (n : ℕ) : ℕ := ∑ k in Finset.range n, aSeq (k + 1)

-- Given condition
axiom given_condition (n : ℕ) : 2 * S n = (aSeq n - 1) * (aSeq n + 2)

-- Prove the general formula of the sequence
theorem general_formula (n : ℕ) : aSeq n = n + 1 :=
by
  sorry  -- proof

-- Define T_n sequence
noncomputable def T (n : ℕ) : ℕ := ∑ k in Finset.range n, (k - 1) * 2^k / (k * aSeq k)

-- Compare T_n with the given expression
theorem compare_Tn (n : ℕ) : 
  if n < 17 then T n < (2^(n+1)*(18-n)-2*n-2)/(n+1)
  else if n = 17 then T n = (2^(n+1)*(18-n)-2*n-2)/(n+1)
  else T n > (2^(n+1)*(18-n)-2*n-2)/(n+1) :=
by
  sorry  -- proof

end general_formula_compare_Tn_l317_317850


namespace susan_flower_vase_l317_317266

def initial_count (dozens : ℕ) := dozens * 12

lemma flower_count_after_giveaway (F : Type) [HasAdd F] [HasSub F] [HasDiv F] [Mul F ℚ] 
    (initial : F) : 
    initial - initial * (3 / 4) = initial / 4 := sorry

lemma flower_count_after_purchase (F : Type) [HasAdd F]
    (remaining : F) (added : F) :
    remaining + added = remaining + added := sorry

lemma flower_count_after_wilting (F : Type) [HasAdd F] [HasSub F] [HasDiv F] [Mul F ℚ]
    (remaining : F) (fraction : ℚ) :
    remaining - remaining * fraction = remaining * (1 - fraction) := sorry

theorem susan_flower_vase : 
    let initial_roses := initial_count 3 in
    let initial_tulips := initial_count 2 in
    let initial_daisies := initial_count 4 in
    let initial_lilies := initial_count 5 in
    let initial_sunflowers := initial_count 3 in

    let remaining_roses := flower_count_after_giveaway ℝ initial_roses in
    let remaining_tulips := flower_count_after_giveaway ℝ initial_tulips in
    let remaining_daisies := flower_count_after_giveaway ℝ initial_daisies in
    let remaining_lilies := flower_count_after_giveaway ℝ initial_lilies in
    let remaining_sunflowers := flower_count_after_giveaway ℝ initial_sunflowers in

    let remaining_roses := flower_count_after_purchase ℝ remaining_roses 12 in
    let remaining_tulips := flower_count_after_purchase ℝ remaining_tulips 0 in
    let remaining_daisies := flower_count_after_purchase ℝ remaining_daisies 24 in
    let remaining_lilies := flower_count_after_purchase ℝ remaining_lilies 0 in
    let remaining_sunflowers := flower_count_after_purchase ℝ remaining_sunflowers 6 in

    let final_roses := flower_count_after_wilting ℝ remaining_roses (1 / 3) in
    let final_tulips := flower_count_after_wilting ℝ remaining_tulips (1 / 4) in
    let final_daisies := flower_count_after_wilting ℝ remaining_daisies (1 / 2) in
    let final_lilies := flower_count_after_wilting ℝ remaining_lilies (2 / 5) in
    let final_sunflowers := flower_count_after_wilting ℝ remaining_sunflowers (1 / 6) in

    final_roses = 14 ∧ final_tulips = 5 ∧ final_daisies = 18 ∧ final_lilies = 9 ∧ final_sunflowers = 13 := sorry

end susan_flower_vase_l317_317266


namespace divisible_by_55_l317_317060

theorem divisible_by_55 (n : ℤ) : 
  (55 ∣ (n^2 + 3 * n + 1)) ↔ (n % 55 = 46 ∨ n % 55 = 6) := 
by 
  sorry

end divisible_by_55_l317_317060


namespace simplify_cube_root_expression_l317_317595

-- statement of the problem
theorem simplify_cube_root_expression :
  ∛(8 + 27) * ∛(8 + ∛(27)) = ∛(385) :=
  sorry

end simplify_cube_root_expression_l317_317595


namespace value_of_m_l317_317918

theorem value_of_m : 
  (2 ^ 1999 - 2 ^ 1998 - 2 ^ 1997 + 2 ^ 1996 - 2 ^ 1995 = m * 2 ^ 1995) -> m = 5 :=
by 
  sorry

end value_of_m_l317_317918


namespace prove_polynomial_has_no_positive_roots_l317_317903

noncomputable def polynomial_has_no_positive_roots
  (a : List ℕ) (k M : ℕ) (h1 : ∑ i in a, 1 / (i : ℝ) = k)
  (h2 : a.prod = M) (h3 : M > 1) : Prop :=
  ∀ x : ℝ, x > 0 → 
    let P := M * (x + 1) ^ k - (a.map (λ ai, x + ai)).prod
    in P < 0

theorem prove_polynomial_has_no_positive_roots
  (a : List ℕ) (k M : ℕ) (h1 : ∑ i in a, 1 / (i : ℝ) = k)
  (h2 : a.prod = M) (h3 : M > 1) :
  polynomial_has_no_positive_roots a k M h1 h2 h3 :=
by sorry

end prove_polynomial_has_no_positive_roots_l317_317903


namespace statisticalProperties_l317_317856

def sampleStandardDeviation (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  let variance := (data.map (λ x, (x - mean) ^ 2)).sum / data.length
  Real.sqrt variance

def sampleRange (data : List ℝ) : ℝ := data.maximum - data.minimum

theorem statisticalProperties (n : ℕ) (x y : Fin n → ℝ) (c : ℝ) (h : ∀ i : Fin n, y i = x i + c) (hn : n > 0) (hc : c ≠ 0) :
  sampleStandardDeviation (List.ofFn y) = sampleStandardDeviation (List.ofFn x) ∧ sampleRange (List.ofFn y) = sampleRange (List.ofFn x) := 
by
  sorry

end statisticalProperties_l317_317856


namespace min_distance_PQ_l317_317222

noncomputable def find_min_distance : ℝ :=
  let center := (8 : ℝ, 0 : ℝ)
  let radius := real.sqrt 2
  let distance_from_center_to_line := |8 / (real.sqrt 2)| -- 4 * real.sqrt 2
  distance_from_center_to_line - radius -- 3 * real.sqrt 2

theorem min_distance_PQ : find_min_distance = 3 * real.sqrt 2 := 
by 
    sorry

end min_distance_PQ_l317_317222


namespace conjugate_of_z_l317_317107

noncomputable def z : ℂ := (1 + 3 * Complex.I) / (1 + Complex.I)

theorem conjugate_of_z :
  Complex.conj z = 2 - Complex.I := by
  sorry

end conjugate_of_z_l317_317107


namespace alloy_chromium_l317_317185

variable (x : ℝ)

theorem alloy_chromium (h : 0.15 * 15 + 0.08 * x = 0.101 * (15 + x)) : x = 35 := by
  sorry

end alloy_chromium_l317_317185


namespace eighty_five_squared_l317_317007

theorem eighty_five_squared : 85^2 = 7225 := by
  sorry

end eighty_five_squared_l317_317007


namespace compute_cubic_sum_l317_317093

theorem compute_cubic_sum (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : x * y + x ^ 2 + y ^ 2 = 17) : x ^ 3 + y ^ 3 = 52 :=
sorry

end compute_cubic_sum_l317_317093


namespace pastor_prayer_l317_317994

theorem pastor_prayer (P : ℕ) (h1 : ∀ d ≠ Sunday, PaulPrays d = P)
  (h2 : PaulPrays Sunday = 2 * P) (h3 : ∀ d ≠ Sunday, BrucePrays d = P / 2) 
  (h4 : BrucePrays Sunday = 4 * P)
  (h5 : ∑ d in Weekdays, PaulPrays d + PaulPrays Sunday = ∑ d in Weekdays, BrucePrays d + BrucePrays Sunday + 20) :
  P = 20 :=
by sorry

end pastor_prayer_l317_317994


namespace max_m_value_l317_317473

theorem max_m_value : ∀ (x : ℝ), x ∈ set.Icc (-Real.pi / 4) (Real.pi / 4) → ∃ (m : ℝ), m ≤ Real.tan x + 1 ∧ m = 2 :=
by
  sorry

end max_m_value_l317_317473


namespace exponential_function_point_l317_317927

theorem exponential_function_point :
  (∃ (f : ℝ → ℝ), ((∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ f = (λ x, a ^ x)) ∧ f 1 = 2)) →
  (∃ (f : ℝ → ℝ), f 2 = 4) :=
by
  sorry

end exponential_function_point_l317_317927


namespace quadratic_root_implies_coefficients_l317_317219

theorem quadratic_root_implies_coefficients (b c : ℝ) : 
  (2 - complex.i) ∈ complex.roots (polynomial.X^2 + polynomial.C b * polynomial.X + polynomial.C c) →
  b = -4 ∧ c = 5 :=
by
  sorry

end quadratic_root_implies_coefficients_l317_317219


namespace range_of_a_l317_317897

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2 * a * (Real.sin x + Real.cos x) - 4 * a * x

noncomputable def g'' (a : ℝ) (x : ℝ) : ℝ := -2 * a * (Real.sin x + Real.cos x)

theorem range_of_a (a : ℝ) : (∃ x ∈ Icc 0 (Real.pi / 2), f x ≥ g'' a x) ↔ a ≥ 1 / 2 :=
by
  sorry

end range_of_a_l317_317897


namespace east_west_exaggeration_factor_l317_317736

noncomputable def exaggeration_factor (R : ℝ) : ℝ :=
  let actual_ns_distance := (2 * Real.pi * R) / 3
  let globe_ew_circumference := Real.pi * R * Real.sqrt 3
  let globe_ratio := globe_ew_circumference / actual_ns_distance
  let map_ratio := 2 * Real.pi
  map_ratio / globe_ratio

theorem east_west_exaggeration_factor :
  ∀ (R : ℝ), exaggeration_factor R ≈ 2.42 :=
by
  sorry

end east_west_exaggeration_factor_l317_317736


namespace log_x_125_equal_approx_l317_317484

-- Defining the condition
def condition (x : ℝ) : Prop :=
  log 8 (5 * x) = 3

-- Stating the problem
theorem log_x_125_equal_approx : ∃ x : ℝ, condition x → log x 125 = (3 / 1.7227) :=
begin
  sorry
end

end log_x_125_equal_approx_l317_317484


namespace percent_quarter_value_correct_l317_317325

-- Defining the conditions
def dimes_count : ℕ := 70
def quarters_count : ℕ := 30
def nickels_count : ℕ := 15

def dime_value : ℕ := 10 -- value in cents
def quarter_value : ℕ := 25 -- value in cents
def nickel_value : ℕ := 5 -- value in cents

-- Calculate total values
def total_dime_value : ℕ := dimes_count * dime_value
def total_quarter_value : ℕ := quarters_count * quarter_value
def total_nickel_value : ℕ := nickels_count * nickel_value

def total_value : ℕ := total_dime_value + total_quarter_value + total_nickel_value

-- Statement to prove
def percent_quarter_value : ℝ := (total_quarter_value : ℝ) / (total_value : ℝ)

theorem percent_quarter_value_correct : percent_quarter_value ≈ 0.4902 :=
by 
  -- Lean code to introduce sorry here since proof steps are not required.
  sorry

end percent_quarter_value_correct_l317_317325


namespace problem_a_b_c_relation_l317_317068

noncomputable def a : ℝ := 2 ^ (-1/3 : ℝ)
noncomputable def b : ℝ := (3 : ℝ) ^ (-1/2)
noncomputable def c : ℝ := 1 / 2

theorem problem_a_b_c_relation : a > b ∧ b > c := 
by 
  sorry

end problem_a_b_c_relation_l317_317068


namespace journey_time_comparison_l317_317691

variable {S x : ℝ} (S_pos : S > 0) (x_pos : x > 0)

-- Xiao Ming's usual cycling speed
noncomputable def cycling_time := (S + S) / (2 * x)

-- Xiao Ming's journey this time
noncomputable def bus_speed := 4 * x
noncomputable def walking_speed := x / 2
noncomputable def bus_time := S / bus_speed
noncomputable def walking_time := S / walking_speed
noncomputable def total_time_this_journey := bus_time + walking_time

theorem journey_time_comparison (S_pos : S > 0) (x_pos : x > 0) :
  total_time_this_journey S x > cycling_time S x :=
by
  sorry

end journey_time_comparison_l317_317691


namespace parallel_lines_cond_l317_317088

def line (a b c : ℝ) := λ x y : ℝ, a * x + b * y + c = 0

theorem parallel_lines_cond (a : ℝ) :
  (a ≠ 0) → (a = -3 → (∀ x y : ℝ, line a (a + 2) 1 x y = 0 → line a (-1) 2 x y = 0)) → 
  (a = 0 ∨ a ≠ -2 ∧ -a / (a + 2) = a) → 
  (sufficient_but_not_necessary : a = -3) :=
by
  intro ha hn heq
  sorry

end parallel_lines_cond_l317_317088


namespace sufficient_condition_for_parallel_planes_l317_317462

variables {m n : Line} {α β : Plane}

-- Definitions of perpendicular and parallel relations between lines and planes
def perpendicular (l : Line) (π : Plane) : Prop := 
∀ p₁ p₂ ∈ π, p₁ ≠ p₂ → line_through p₁ p₂ ⊥ l

def parallel (l : Line) (π : Plane) : Prop := 
∀ p₁ p₂ ∈ π, p₁ ≠ p₂ → line_through p₁ p₂ ∥ l

def parallel_planes (π₁ π₂ : Plane) : Prop := 
∀ l₁ l₂ ∈ π₁, l₂ ∈ π₂ → l₁ ∥ l₂

variables h₁ : perpendicular m α
variables h₂ : perpendicular n β
variables h₃ : m ∥ n

-- Theorem statement
theorem sufficient_condition_for_parallel_planes : parallel_planes α β :=
by {
  sorry,
}

end sufficient_condition_for_parallel_planes_l317_317462


namespace quoted_value_of_stock_l317_317344

theorem quoted_value_of_stock (F P : ℝ) (h1 : F > 0) (h2 : P = 1.25 * F) : 
  (0.10 * F) / P = 0.08 := 
sorry

end quoted_value_of_stock_l317_317344


namespace incorrect_statements_l317_317988

-- Definitions
def double_square_side_area (s : ℝ) : Prop :=
  let original_area := s ^ 2
  let new_area := (2 * s) ^ 2
  new_area = 4 * original_area

def double_cylinder_height_volume (r h : ℝ) : Prop :=
  let original_volume := π * r ^ 2 * h
  let new_volume := π * r ^ 2 * (2 * h)
  new_volume = 2 * original_volume

def double_cube_edge_volume (e : ℝ) : Prop :=
  let original_volume := e ^ 3
  let new_volume := (2 * e) ^ 3
  new_volume = 8 * original_volume

def modify_fraction (a b : ℝ) (hb : b ≠ 0) : Prop :=
  let original_fraction := a / b
  let new_fraction := (2 * a) / (b / 2)
  new_fraction = 4 * original_fraction

def add_zero (x : ℝ) : Prop :=
  x + 0 = x

-- Problem statement
theorem incorrect_statements :
  ∃ (double_cylinder_height_volume_incorrect : Prop) (add_zero_incorrect : Prop),
    (double_cylinder_height_volume_incorrect = ¬ double_cylinder_height_volume) ∧
    (add_zero_incorrect = ¬ add_zero) ∧
    double_cylinder_height_volume_incorrect ∧
    add_zero_incorrect :=
sorry

end incorrect_statements_l317_317988


namespace quadratic_axis_of_symmetry_l317_317496

theorem quadratic_axis_of_symmetry (b c : ℝ) (h : -b / 2 = 3) : b = 6 :=
by
  sorry

end quadratic_axis_of_symmetry_l317_317496


namespace cos_eq_iff_angle_eq_l317_317529

variables {A B : ℝ}

theorem cos_eq_iff_angle_eq (hA : 0 < A ∧ A < real.pi) (hB : 0 < B ∧ B < real.pi) :
  (real.cos A = real.cos B) ↔ (A = B) :=
begin
  sorry
end

end cos_eq_iff_angle_eq_l317_317529


namespace min_value_of_y_l317_317279

theorem min_value_of_y (a : ℝ) (x : ℝ) (h : ∀ x, a * sin x + 1 ≤ 3) : ∃ x, a * sin x + 1 = -1 := by
  sorry

end min_value_of_y_l317_317279


namespace find_orthonormal_system_l317_317215

variable {α : Type*}
variable [InnerProductSpace ℝ α] [InfiniteDimensional ℝ α]

def exists_orthonormal_system (d : ℝ) (S : Set α) (h_dist : ∀ x y ∈ S, x ≠ y → dist x y = d) (hd_pos : d > 0) : Prop :=
  ∃ y ∈ α, Orthonormal ℝ (λ x : S, (√2 / d) • (↑x - y))

theorem find_orthonormal_system 
  (𝓗 : Type*) [InnerProductSpace ℝ 𝓗] [InfiniteDimensional ℝ 𝓗]  
  (d : ℝ) (S : Set 𝓗) 
  (hd_pos : d > 0) 
  (h_dist : ∀ x y ∈ S, x ≠ y → dist x y = d) : 
  exists_orthonormal_system d S h_dist hd_pos :=
sorry

end find_orthonormal_system_l317_317215


namespace cost_to_fill_sandbox_l317_317211

structure Section :=
(length : ℝ)
(width : ℝ)
(depth : ℝ)

def volume (s : Section) : ℝ :=
  s.length * s.width * s.depth

def cost_per_cubic_foot : ℝ := 3.0
def discount_threshold : ℝ := 20.0
def discount_rate : ℝ := 0.10

def total_volume (s1 s2 : Section) : ℝ :=
  volume s1 + volume s2

def raw_cost (total_volume : ℝ) : ℝ :=
  total_volume * cost_per_cubic_foot

def discount (raw_cost : ℝ) : ℝ :=
  if raw_cost / cost_per_cubic_foot > discount_threshold then 
    raw_cost * discount_rate 
  else 
    0

def final_cost (raw_cost : ℝ) (discount : ℝ) : ℝ :=
  raw_cost - discount

def sandbox_cost (s1 s2 : Section) : ℝ :=
  let total_vol := total_volume s1 s2
  let raw_c := raw_cost total_vol
  final_cost raw_c (discount raw_c)

theorem cost_to_fill_sandbox : sandbox_cost ⟨3, 2, 2⟩ ⟨5, 2, 2⟩ = 86.40 := by
  sorry

end cost_to_fill_sandbox_l317_317211


namespace RS_eq_5_l317_317361

-- Definitions for the tetrahedron, vertices, and edges:

def vertices := {P Q R S : ℕ} -- The tetrahedron has four vertices P, Q, R, S
def edges := {PQ PR PS QR QS RS : ℕ} -- The tetrahedron has six edges PQ, PR, PS, QR, QS, RS

-- The ten numbers used in vertices and edges
def numbers : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 11}

-- Conditions for the vertices and edges
def valid_vertex_number (n : ℕ) : Prop := n ∈ numbers
def valid_edge_number (edge : ℕ) : Prop := edge ∈ {1 + 2, 1 + 3, 1 + 4, 1 + 5, 1 + 6, 1 + 7, 1 + 8, 1 + 9, 1 + 11,
  2 + 3, 2 + 4, 2 + 5, 2 + 6, 2 + 7, 2 + 8, 2 + 9, 2 + 11,
  3 + 4, 3 + 5, 3 + 6, 3 + 7, 3 + 8, 3 + 9, 3 + 11,
  4 + 5, 4 + 6, 4 + 7, 4 + 8, 4 + 9, 4 + 11,
  5 + 6, 5 + 7, 5 + 8, 5 + 9, 5 + 11,
  6 + 7, 6 + 8, 6 + 9, 6 + 11,
  7 + 8, 7 + 9, 7 + 11,
  8 + 9, 8 + 11,
  9 + 11} -- Possible sums representing edges 

-- Edge PQ equals 9
axiom PQ_eq_9 : (P + Q) = 9

-- Prove edge RS equals 5
theorem RS_eq_5 : (R + S) = 5 := 
sorry

end RS_eq_5_l317_317361


namespace number_of_dogs_l317_317811

theorem number_of_dogs (h1 : 24 = 2 * 2 + 4 * n) : n = 5 :=
by
  sorry

end number_of_dogs_l317_317811


namespace square_difference_l317_317788

theorem square_difference :
  153^2 - 147^2 = 1800 :=
by
  sorry

end square_difference_l317_317788


namespace square_difference_l317_317791

theorem square_difference :
  153^2 - 147^2 = 1800 :=
by
  sorry

end square_difference_l317_317791


namespace value_of_k_for_binomial_square_l317_317310

theorem value_of_k_for_binomial_square (k : ℝ) : (∃ (b : ℝ), x^2 - 18 * x + k = (x + b)^2) → k = 81 :=
by
  intro h
  cases h with b hb
  -- We will use these to directly infer things without needing the proof here
  sorry

end value_of_k_for_binomial_square_l317_317310


namespace policeman_always_reaches_same_side_l317_317705

theorem policeman_always_reaches_same_side
    (s : ℝ) (v : ℝ) (policeman_speed : ℝ)
    (gangster_speed : ℝ)
    (center O : ℝ)
    (vertex A : ℝ)
    (gangster_at_A : gangster_speed = 2.9 * policeman_speed)
    (policeman_in_center : O = s / √2)
    (gangster_on_edge : ∀ t, gangster_at_A ↔ (t ≥ 0 ∧ t ≤ s)) :
  ∃ (t : ℝ), policeman_in_center + t * policeman_speed = gangster_on_edge :=
sorry

end policeman_always_reaches_same_side_l317_317705


namespace point_on_graph_l317_317273

theorem point_on_graph (a : ℝ) (x y : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (hx : x = 2) (hy : y = a^(x - 2) - 1) : (x, y) = (2, 0) :=
by
    subst hx
    rw hy
    calc a^(2-2) - 1 = a^0 - 1 : by rw sub_self
                      ... = 1 - 1 : by rw pow_zero
                      ... = 0 : by rw sub_self
    exact Eq.refl (2, 0)

end point_on_graph_l317_317273


namespace minimum_value_expr_pos_reals_l317_317975

noncomputable def expr (a b : ℝ) := a^2 + b^2 + 2 * a * b + 1 / (a + b)^2

theorem minimum_value_expr_pos_reals (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : 
  (expr a b) ≥ 2 :=
sorry

end minimum_value_expr_pos_reals_l317_317975


namespace serena_mother_years_l317_317253

theorem serena_mother_years (x : ℕ) (serena_age mother's_age : ℕ)
    (h1 : serena_age = 9)
    (h2 : mother's_age = 39) 
    (h3 : mother's_age + x = 3 * (serena_age + x)) : x = 6 :=
by {
  subtle proof steps
  sorry
}

end serena_mother_years_l317_317253


namespace hyperbola_standard_equation_l317_317884

theorem hyperbola_standard_equation 
  (M : Type) 
  (eccentricity : ℝ) 
  (focus_to_asymptote_dist : ℝ)
  (h_ecc : eccentricity = sqrt 3)
  (h_dist : focus_to_asymptote_dist = 2) :
  (exists (a b : ℝ), a^2 = 2 ∧ b^2 = 4 ∧ (   (∀ x y : ℝ, (x^2 / 2 - y^2 / 4 = 1)   ∨   (y^2 / 2 - x^2 / 4 = 1)   ))) :=
by
  sorry

end hyperbola_standard_equation_l317_317884


namespace repeating_decimal_to_fraction_l317_317695

theorem repeating_decimal_to_fraction : (let a := (0.28282828 : ℚ); a = 28/99) := sorry

end repeating_decimal_to_fraction_l317_317695


namespace find_angle_between_vectors_l317_317563

open Real EuclideanSpace

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ := 
  let len_a := ‖a‖
  let len_b := ‖b‖
  let diff_ab := ‖a - b‖
  let dot_ab := dot_product a b 
  real.arccos (dot_ab / (len_a * len_b))

theorem find_angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) 
  (h_a_nonzero : a ≠ 0) 
  (h_b_nonzero : b ≠ 0)
  (h_norm_a : ‖a‖ = 2)
  (h_norm_b : ‖b‖ = 2)
  (h_norm_diff : ‖a - b‖ = 2) : 
  angle_between_vectors a b = 60 :=
  sorry

end find_angle_between_vectors_l317_317563


namespace simplify_cub_root_multiplication_l317_317598

theorem simplify_cub_root_multiplication (a b : ℝ) (ha : a = 8) (hb : b = 27) :
  (real.cbrt (a + b) * real.cbrt (a + real.cbrt b)) = real.cbrt ((a + b) * (a + real.cbrt b)) := 
by
  sorry

end simplify_cub_root_multiplication_l317_317598


namespace sum_log_geom_seq_l317_317098

theorem sum_log_geom_seq (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_geom : a 5 * a 6 = 4) :
    ∑ i in Finset.range 10, Real.log (a i) / Real.log 2 = 10 := 
by
  sorry

end sum_log_geom_seq_l317_317098


namespace exists_triangle_construction_l317_317805

noncomputable def construct_triangle (a A bc : ℝ) : Prop :=
∀ (A B C : Type), ∃ B C : Type,
-- Given sides and angle conditions
angle A b c bc triangle ABC,
  -- Base BC of triangle
  BC = a ∧
  -- Given angle ∠BAC
  angle BAC A ∧
  -- Sum of sides b + c
  (side b side c = bc)
  -- Ensure that the triangle has vertices A, B, and C with the described properties

theorem exists_triangle_construction (a A bc : ℝ) 
  (h1 : a > 0) 
  (h2 : A > 0 ∧ A < π) 
  (h3 : bc > a):
  construct_triangle a A bc := 
begin
  -- Construction process described in the solution
  -- Base construction and midpoint positioning
  let B := (_ : Type),    -- A potential type for point B
  let C := (_ : Type),    -- A potential type for point C
  let D := (_ : Type),    -- Midpoint type
  let K := (_ : Type),    -- Point on the circle
  have BC_construction : BC = a := sorry,
  have circle_property : ∀ (DC : Line) (K : Point), on_circle K Kcircle DC := sorry,
  have point_A_conditions : ∀ (b c : side), (extend_seq K AC = D) := sorry,
  use B, use C,  
  exact ⟨BC_construction, circle_property, point_A_conditions⟩,
  sorry
end

end exists_triangle_construction_l317_317805


namespace unique_solution_c_min_l317_317419

theorem unique_solution_c_min (x y : ℝ) (c : ℝ)
  (h1 : 2 * (x+7)^2 + (y-4)^2 = c)
  (h2 : (x+4)^2 + 2 * (y-7)^2 = c) :
  c = 6 :=
sorry

end unique_solution_c_min_l317_317419


namespace avg_monthly_bill_over_6_months_l317_317725

theorem avg_monthly_bill_over_6_months :
  ∀ (avg_first_4_months avg_last_2_months : ℝ), 
  avg_first_4_months = 30 → 
  avg_last_2_months = 24 → 
  (4 * avg_first_4_months + 2 * avg_last_2_months) / 6 = 28 :=
by
  intros
  sorry

end avg_monthly_bill_over_6_months_l317_317725


namespace quadratic_binomial_square_l317_317315

theorem quadratic_binomial_square (k : ℝ) : (∃ b : ℝ, x^2 - 18 * x + k = (x - b)^2) → k = 81 :=
begin
  sorry
end

end quadratic_binomial_square_l317_317315


namespace range_of_m_l317_317090

-- Definitions based on the conditions
def p (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 > 0

theorem range_of_m (m : ℝ) (hm : m > 0) :
  (∀ x : ℝ, p x → q x m) ∧ (∃ x : ℝ, ¬(p x) ∧ q x m) ↔ 0 < m ∧ m ≤ 2 := sorry

end range_of_m_l317_317090


namespace sequence_not_perfect_square_l317_317148

-- Define the sequence of numbers.
def N (i : ℕ) : ℕ := 
  let zeros := String.mk $ List.replicate (i - 1) '0'
  ("2014" ++ zeros ++ "2015").to_nat

-- Main theorem stating that none of the numbers in the sequence are perfect squares.
theorem sequence_not_perfect_square (i : ℕ) : ¬ ∃ x : ℕ, x * x = N i := by
  sorry

end sequence_not_perfect_square_l317_317148


namespace value_of_expression_l317_317491

theorem value_of_expression (x : ℤ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  exact rfl

end value_of_expression_l317_317491


namespace petya_wins_with_optimal_play_l317_317944
-- prelude and import

-- theorem statement
theorem petya_wins_with_optimal_play :
  ∀ (contacts : ℕ) 
    (initial_wires : contacts * (contacts - 1) / 2)
    (turns : ℕ → ℕ → Prop)
    (vasya_starts : turns = 1)
    (petya_moves : turns → Prop)
    (last_wire_rule : ∀ last_turn wires, 
      (turns last_turn = 1 ∧ wires = 0) → false),
    contacts = 2000 →
    (∀ (moves : ℕ) (v_turn : bool), v_turn = tt → (moves = 1)) →
    (∀ (moves : ℕ) (p_turn : bool), p_turn = ff → (moves = 1 ∨ moves = 3)) →
    Petya_wins :=
begin
  sorry
end

end petya_wins_with_optimal_play_l317_317944


namespace find_a_min_value_l317_317894

theorem find_a_min_value :
  ∀ (a : ℝ),
  (∀ (x : ℝ),
  0 < x ∧ x ≤ π / 4 →
  f(x, a) = sin (2 * x) + a * cos (2 * x) ∧
  (∀ y : ℝ, 0 < y ∧ y ≤ π /4 → f(y, a) ≥ a)) →
  a = 1 :=
by
  sorry

end find_a_min_value_l317_317894


namespace range_of_k_for_monotonicity_l317_317450

noncomputable def f (x k : ℝ) : ℝ := 4 * x^2 + k * x - 1

def is_monotonic_on_interval (f' : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ (set.Icc a b), f' x ≥ 0) ∨ (∀ x ∈ (set.Icc a b), f' x ≤ 0)

theorem range_of_k_for_monotonicity :
  let f' (x : ℝ) (k : ℝ) := 8 * x + k in
  (∀ x ∈ (set.Icc 1 2), f' x k ≥ 0) ∨ (∀ x ∈ (set.Icc 1 2), f' x k ≤ 0) ↔
  (k ∈ set.Icc (-∞ : ℝ, -16]) ∨ (k ∈ set.Icc [-8 : ℝ, ∞]) := 
sorry

end range_of_k_for_monotonicity_l317_317450


namespace total_weight_full_bucket_l317_317692

theorem total_weight_full_bucket (x y c d : ℝ) 
(h1 : x + 3/4 * y = c) 
(h2 : x + 1/3 * y = d) :
x + y = (8 * c - 3 * d) / 5 :=
sorry

end total_weight_full_bucket_l317_317692


namespace rearrange_numbers_diff_3_or_5_l317_317202

noncomputable def is_valid_permutation (n : ℕ) (σ : list ℕ) : Prop :=
  (σ.nodup ∧ ∀ i < σ.length - 1, |σ.nth_le i sorry - σ.nth_le (i + 1) sorry| = 3 ∨ |σ.nth_le i sorry - σ.nth_le (i + 1) sorry| = 5)

theorem rearrange_numbers_diff_3_or_5 (n : ℕ) :
  (n = 25 ∨ n = 1000) → ∃ σ : list ℕ, (σ = (list.range n).map (+1)) ∧ is_valid_permutation n σ :=
by
  sorry

end rearrange_numbers_diff_3_or_5_l317_317202


namespace sequence_not_perfect_square_l317_317141

theorem sequence_not_perfect_square (n : ℕ) (N : ℕ → ℕ) :
  (∀ i, let d := [2, 0, 0, 1, 4, 0, 2, 0, 1, 5]
  in (N i = 2014 * 10^(n - i) + list.sum d) ∧ list.sum d = 15) →
  (∀ i, ¬ is_square (N i)) :=
by
  sorry

end sequence_not_perfect_square_l317_317141


namespace no_integer_roots_l317_317900

theorem no_integer_roots (a b : ℤ) : ¬∃ u : ℤ, u^2 + 3 * a * u + 3 * (2 - b^2) = 0 :=
by
  sorry

end no_integer_roots_l317_317900


namespace part1_impossible_2008_2009_2010_part1_possible_2009_2010_2011_part2_possible_values_of_x_l317_317666

noncomputable def transform1 (A B C : ℤ) : ℤ × ℤ × ℤ :=
  (2 * B + 2 * C - A, B, C)

noncomputable def invariant_mod2 (t : ℤ × ℤ × ℤ) : ℤ :=
  let (A, B, C) := t
  (A + B + C) % 2

noncomputable def invariant_quadratic (t : ℤ × ℤ × ℤ) : ℤ :=
  let (A, B, C) := t
  A^2 + B^2 + C^2 - 2 * (A * B + B * C + C * A)

def initial_triplet : ℤ × ℤ × ℤ := (-1, 0, 1)

def target1 : ℤ × ℤ × ℤ := (2008, 2009, 2010)
def target2 : ℤ × ℤ × ℤ := (2009, 2010, 2011)
def final_triplet (x : ℤ) : ℤ × ℤ × ℤ := (1, 2024, x)

theorem part1_impossible_2008_2009_2010 :
  ¬ ∃ n : ℕ, ∃ t : ℤ × ℤ × ℤ, 
    (transform1^[n]) initial_triplet = t ∧ t = target1 := sorry

theorem part1_possible_2009_2010_2011 : 
  ∃ n : ℕ, ∃ t : ℤ × ℤ × ℤ, 
    (transform1^[n]) initial_triplet = t ∧ t = target2 := sorry

theorem part2_possible_values_of_x (x : ℤ) :
  (∃ n : ℕ, ∃ t : ℤ × ℤ × ℤ, 
    (transform1^[n]) initial_triplet = t ∧ t = final_triplet x) ↔ 
    x = 2034 ∨ x = 2016 := sorry

end part1_impossible_2008_2009_2010_part1_possible_2009_2010_2011_part2_possible_values_of_x_l317_317666


namespace zero_in_interval_2_3_l317_317290

def f (x : ℝ) : ℝ := log x + 2 * x - 5

theorem zero_in_interval_2_3 : ∃ c ∈ (2, 3), f c = 0 :=
by
  sorry

end zero_in_interval_2_3_l317_317290


namespace point_in_first_quadrant_l317_317189

/-- In the Cartesian coordinate system, if a point P has x-coordinate 2 and y-coordinate 4, it lies in the first quadrant. -/
theorem point_in_first_quadrant (x y : ℝ) (h1 : x = 2) (h2 : y = 4) : 
  x > 0 ∧ y > 0 → 
  (x, y).1 = 2 ∧ (x, y).2 = 4 → 
  (x > 0 ∧ y > 0) := 
by
  intros
  sorry

end point_in_first_quadrant_l317_317189


namespace inequality_am_gm_l317_317979

theorem inequality_am_gm (n : ℕ) (h : 1 ≤ n) (x : Fin (2 * n + 1) → ℝ)
    (hx : ∀ i, 0 < x i) :
    (∑ i in Finset.range (2 * n + 1), (x i * x (i + 1) / x (i + 2))) ≥ ∑ i in Finset.range (2 * n + 1), x i 
    ↔ ∀ i j, x i = x j := 
sorry

end inequality_am_gm_l317_317979


namespace rearrange_possible_l317_317203

theorem rearrange_possible (n : ℕ) (h : n = 25 ∨ n = 1000) :
  ∃ (f : ℕ → ℕ), (∀ i < n, f i + 1 < n → (f (i + 1) - f i = 3 ∨ f (i + 1) - f i = 5)) :=
  sorry

end rearrange_possible_l317_317203


namespace lauren_annual_income_l317_317526

open Real

theorem lauren_annual_income (p : ℝ) (A : ℝ) (T : ℝ) :
  (T = (p + 0.45)/100 * A) →
  (T = (p/100) * 20000 + ((p + 1)/100) * 15000 + ((p + 3)/100) * (A - 35000)) →
  A = 36000 :=
by
  intros
  sorry

end lauren_annual_income_l317_317526


namespace range_of_a_l317_317435

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x + 1) ^ 2 > 4 → x > a) → a ≥ 1 := sorry

end range_of_a_l317_317435


namespace symmetric_line_eq_l317_317710

theorem symmetric_line_eq (x y : ℝ) (h : 3 * x + 4 * y + 5 = 0) : 3 * x - 4 * y + 5 = 0 :=
sorry

end symmetric_line_eq_l317_317710


namespace find_c_cos_min_l317_317383

theorem find_c_cos_min (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∃ x : ℝ, x = π / (2 * b) ∧ y = a * cos (b * x + c) ∧ bx + c = π) →
  c = π / 2 :=
by
  sorry

end find_c_cos_min_l317_317383


namespace cos_equivalent_l317_317096

open Real

theorem cos_equivalent (alpha : ℝ) (h : sin (π / 3 + alpha) = 1 / 3) : 
  cos (5 * π / 6 + alpha) = -1 / 3 :=
sorry

end cos_equivalent_l317_317096


namespace age_difference_l317_317932

noncomputable def B_years : ℕ := 37
noncomputable def A_years : ℕ := 2 * (B_years - 10) - 10

theorem age_difference :
  A_years - B_years = 7 :=
by
  have h1 : B_years = 37 := rfl
  have h2 : A_years = 2 * (37 - 10) - 10 := rfl
  have h3 : A_years = 54 - 10 := by rw [h2]
  have h4 : A_years - 37 = 44 - 37 := by rw [←h3, h1]
  show 44 - 37 = 7 by rfl

end age_difference_l317_317932


namespace polynomial_reciprocal_derivative_sum_zero_l317_317996

noncomputable def polynomial_recip_sum_zero (P : Polynomial ℝ) (n : ℕ) (roots : Fin n → ℝ) (h_deg : P.degree = (n : with_bot ℕ)) (h_distinct : ∀ i j, i ≠ j → roots i ≠ roots j) (h_roots : ∀ i, P.eval (roots i) = 0) : Prop :=
  ∑ i, 1 / (P.derivative.eval (roots i)) = 0

theorem polynomial_reciprocal_derivative_sum_zero :
  ∀ (P : Polynomial ℝ) (n : ℕ), n > 1 →
  ∀ (roots : Fin n → ℝ),
  (P.degree = (n : with_bot ℕ)) →
  (∀ i j, i ≠ j → roots i ≠ roots j) →
  (∀ i, P.eval (roots i) = 0) →
  ∑ i, 1 / (P.derivative.eval (roots i)) = 0 :=
sorry

end polynomial_reciprocal_derivative_sum_zero_l317_317996


namespace gina_snake_mice_in_decade_l317_317842

-- Definitions based on the conditions in a)
def weeks_per_mouse : ℕ := 4
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10

-- The final statement to prove
theorem gina_snake_mice_in_decade : 
  (weeks_per_year / weeks_per_mouse) * years_per_decade = 130 :=
by
  sorry

end gina_snake_mice_in_decade_l317_317842


namespace difference_of_squares_153_147_l317_317798

theorem difference_of_squares_153_147 : (153^2 - 147^2) = 1800 := by
  sorry

end difference_of_squares_153_147_l317_317798


namespace area_of_quadrilateral_ADEC_l317_317521

-- We define the conditions and the goal in Lean 4
theorem area_of_quadrilateral_ADEC
  (A B C D M : Point)
  (hM_midpoint : midpoint A B M)
  (hCM : distance C M = 5)
  (hAC : distance A C = 8)
  (hABC_perpendicular : angle A C B = 90) :
  area_of_quadrilateral A D E C = 48 :=
sorry

end area_of_quadrilateral_ADEC_l317_317521


namespace eighty_five_squared_l317_317010

theorem eighty_five_squared :
  (85:ℕ)^2 = 7225 := 
by
  let a := 80
  let b := 5
  have h1 : (a + b) = 85 := rfl
  have h2 : (a^2 + 2 * a * b + b^2) = 7225 := by norm_num
  rw [←h1, ←h1]
  rw [ sq (a + b)]
  rw [ mul_add, add_mul, add_mul, mul_comm 2 b]
  rw [←mul_assoc, ←mul_assoc, add_assoc, add_assoc, nat.add_right_comm ]
  exact h2

end eighty_five_squared_l317_317010


namespace dot_product_is_correct_l317_317911

variables (a b : Real × Real)
def a := (2, -1)
def b := (-1, 2)

theorem dot_product_is_correct : (a.1 * b.1 + a.2 * b.2) = -4 :=
by {
  -- Proof is omitted
  sorry
}

end dot_product_is_correct_l317_317911


namespace triangle_inequality_l317_317097

variables (a b c : ℝ) (S_triangle : ℝ)

def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_inequality (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : S_triangle = area_of_triangle a b c) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S_triangle :=
sorry

end triangle_inequality_l317_317097


namespace min_value_of_exponential_expr_l317_317440

variable {x y : ℝ}

theorem min_value_of_exponential_expr (h : 2 * x - y = 4) : 4^x + (1 / 2)^y = 8 :=
by
  sorry

end min_value_of_exponential_expr_l317_317440


namespace sufficient_but_not_necessary_condition_l317_317642

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, x > 4 → (x > 3 ∨ x < -1)) ∧ ¬ (∀ x : ℝ, (x > 3 ∨ x < -1) → x > 4) :=
by
  sorry

end sufficient_but_not_necessary_condition_l317_317642


namespace dacid_chemistry_marks_l317_317807

theorem dacid_chemistry_marks (marks_english marks_math marks_physics marks_biology : ℕ)
  (average_marks total_subjects : ℕ) 
  (h1 : marks_english = 73) 
  (h2 : marks_math = 69) 
  (h3 : marks_physics = 92) 
  (h4 : marks_biology = 82)
  (h5 : average_marks = 76)
  (h6 : total_subjects = 5) :
  let total_marks := average_marks * total_subjects in
  let total_marks_in_known := marks_english + marks_math + marks_physics + marks_biology in
  let marks_chemistry := total_marks - total_marks_in_known in
  marks_chemistry = 64 :=
by {
  admit
}

end dacid_chemistry_marks_l317_317807


namespace can_draw_13_Ts_on_grid_l317_317394

/-- Define the grid with 9 horizontal and 9 vertical lines. -/
def grid : Finset (ℕ × ℕ) :=
  Finset.univ.image (λ (x : Fin 9) (y : Fin 9), (x, y))

/-- Each letter "T" occupies 5 intersections. -/
def T_points : ℕ := 5

theorem can_draw_13_Ts_on_grid :
  let total_intersections := 81
  let points_per_T := 5
  let required_points_for_13_Ts := 13 * points_per_T
  required_points_for_13_Ts ≤ total_intersections :=
by
  let total_intersections := grid.card
  let points_per_T := T_points
  let required_points_for_13_Ts := 13 * points_per_T
  calc required_points_for_13_Ts ≤ 81 : sorry -- we require points <= 81

end can_draw_13_Ts_on_grid_l317_317394


namespace find_k_l317_317873

theorem find_k (k : ℝ) (h : 2 * k = real.sqrt (k * (k + 3))) : k = 1 := by
  sorry

end find_k_l317_317873


namespace gina_snake_mice_in_decade_l317_317841

-- Definitions based on the conditions in a)
def weeks_per_mouse : ℕ := 4
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10

-- The final statement to prove
theorem gina_snake_mice_in_decade : 
  (weeks_per_year / weeks_per_mouse) * years_per_decade = 130 :=
by
  sorry

end gina_snake_mice_in_decade_l317_317841


namespace am_plus_bm_leq_four_div_m_l317_317958

variables {n : ℕ} (a b : ℕ → ℚ) (m : ℕ)
hypothesis (h_unique_a : ∀ i j, i ≠ j -> a i ≠ a j)
hypothesis (h_unique_b : ∀ i j, i ≠ j -> b i ≠ b j)
hypothesis (h_range_a : ∀ i, a i ∈ {k | ∃ j, k = 1 / j} ∧ 1 ≤ j ∧ j ≤ n)
hypothesis (h_range_b : ∀ i, b i ∈ {k | ∃ j, k = 1 / j} ∧ 1 ≤ j ∧ j ≤ n)
hypothesis (h_sorted : ∀ i j, i ≤ j -> a i + b i ≥ a j + b j)
hypothesis (h_bounds : 1 ≤ m ∧ m ≤ n)

theorem am_plus_bm_leq_four_div_m : a m + b m ≤ 4 / m :=
sorry

end am_plus_bm_leq_four_div_m_l317_317958


namespace difference_students_pets_in_all_classrooms_l317_317040

-- Definitions of the conditions
def students_per_classroom : ℕ := 24
def rabbits_per_classroom : ℕ := 3
def guinea_pigs_per_classroom : ℕ := 2
def number_of_classrooms : ℕ := 5

-- Proof problem statement
theorem difference_students_pets_in_all_classrooms :
  (students_per_classroom * number_of_classrooms) - 
  ((rabbits_per_classroom + guinea_pigs_per_classroom) * number_of_classrooms) = 95 := by
  sorry

end difference_students_pets_in_all_classrooms_l317_317040


namespace reading_comprehension_application_method_1_application_method_2_l317_317240

-- Reading Comprehension Problem in Lean 4
theorem reading_comprehension (x : ℝ) (h : x^2 + x + 5 = 8) : 2 * x^2 + 2 * x - 4 = 2 :=
by sorry

-- Application of Methods Problem (1) in Lean 4
theorem application_method_1 (x : ℝ) (h : x^2 + x + 2 = 9) : -2 * x^2 - 2 * x + 3 = -11 :=
by sorry

-- Application of Methods Problem (2) in Lean 4
theorem application_method_2 (a b : ℝ) (h : 8 * a + 2 * b = 5) : a * (-2)^3 + b * (-2) + 3 = -2 :=
by sorry

end reading_comprehension_application_method_1_application_method_2_l317_317240


namespace parity_expression_l317_317565

theorem parity_expression
  (a b c : ℕ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_odd : a % 2 = 1)
  (h_b_odd : b % 2 = 1) :
  (5^a + (b + 1)^2 * c) % 2 = 1 :=
by
  sorry

end parity_expression_l317_317565


namespace meet_point_QS_distance_l317_317661

noncomputable def Q_on_QR (P Q R S : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S] :=
(PQ_distance : ℝ)
(QR_distance : ℝ)
(RP_distance : ℝ)
(bug1_distance : ℝ)
(bug2_distance : ℝ)
(PQ_distance = 7) ∧ 
(QR_distance = 8) ∧ 
(RP_distance = 9) ∧ 
(bug1_distance = 10) ∧ 
(bug2_distance = 10) ∧
(QS : ℝ) ∧
(meet_point_S : ∃ S, true) ∧
(QS = QR_distance - 4)
-- statement to prove
theorem meet_point_QS_distance (P Q R S : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]
  (PQ_distance QR_distance RP_distance bug1_distance bug2_distance QS : ℝ)
  (h : Q_on_QR P Q R S PQ_distance QR_distance RP_distance bug1_distance bug2_distance):
  QS = 3 := sorry

end meet_point_QS_distance_l317_317661


namespace max_value_product_geometric_sequence_l317_317947

theorem max_value_product_geometric_sequence :
  ∀ n : ℕ, 
    let a_n := (1536 : ℤ) * (-1 / 2) ^ (n - 1) in
    let π_n := ∏ i in finset.range n, a_n in
    |π_n| ≤ |∏ i in finset.range 12, a_n| := 
sorry

end max_value_product_geometric_sequence_l317_317947


namespace one_over_a_plus_one_over_b_eq_neg_one_l317_317056

theorem one_over_a_plus_one_over_b_eq_neg_one
  (a b : ℝ) (h_distinct : a ≠ b)
  (h_eq : a / b + a = b / a + b) :
  1 / a + 1 / b = -1 :=
by
  sorry

end one_over_a_plus_one_over_b_eq_neg_one_l317_317056


namespace problem_equivalence_l317_317523

noncomputable def a_n (n : ℕ) : ℝ := 1/3 + (n-1) * 1/3
noncomputable def b_n (n : ℕ) : ℝ := 3^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := n * (1/3) + (n * (n-1)/2) * (1/3)
noncomputable def c_n (n : ℕ) : ℝ := a_n n * b_n n
noncomputable def T_n (n : ℕ) : ℝ := (finset.range n).sum (λ k, c_n (k+1))

theorem problem_equivalence (b_2 : ℝ) (q : ℝ)
  (h1 : q + S_n 2 = 4)
  (h2 : q = b_2 * S_n 2) :
  (∀ n, a_n n = n/3) ∧
  (∀ n, b_n n = 3^(n-1)) ∧
  (∀ n, T_n n = (2*n - 1)/4 * 3^(n-1) + 1/12) :=
by sorry

end problem_equivalence_l317_317523


namespace sample_statistics_comparison_l317_317857

variable {α : Type*} [LinearOrder α] [Field α]

def sample_data_transformed (x : list α) (c : α) : list α :=
  x.map (λ xi, xi + c)

def sample_mean (x : list α) : α :=
  x.sum / (x.length : α)

def sample_median (x : list α) : α :=
  if x.length % 2 = 1 then
    x.sort.nth_le (x.length / 2) (by sorry)
  else
    (x.sort.nth_le (x.length / 2 - 1) (by sorry) + x.sort.nth_le (x.length / 2) (by sorry)) / 2

def sample_variance (x : list α) : α :=
  let m := sample_mean x
  in (x.map (λ xi, (xi - m)^2)).sum / (x.length : α)

def sample_standard_deviation (x : list α) : α :=
  (sample_variance x).sqrt

def sample_range (x : list α) : α :=
  x.maximum (by sorry) - x.minimum (by sorry)

theorem sample_statistics_comparison (x : list α) (c : α) (hc : c ≠ 0) :
  (sample_mean (sample_data_transformed x c) ≠ sample_mean x) ∧
  (sample_median (sample_data_transformed x c) ≠ sample_median x) ∧
  (sample_standard_deviation (sample_data_transformed x c) = sample_standard_deviation x) ∧
  (sample_range (sample_data_transformed x c) = sample_range x) :=
  sorry

end sample_statistics_comparison_l317_317857


namespace black_white_ratio_extended_pattern_l317_317364

theorem black_white_ratio_extended_pattern
  (original_black : ℕ) (original_white : ℕ) (added_black : ℕ)
  (h1 : original_black = 10)
  (h2 : original_white = 26)
  (h3 : added_black = 20) :
  (original_black + added_black) / original_white = 30 / 26 :=
by sorry

end black_white_ratio_extended_pattern_l317_317364


namespace range_of_m_l317_317456

theorem range_of_m (a m : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  m * (a + 1/a) / Real.sqrt 2 > 1 → m ≥ Real.sqrt 2 / 2 := by
  sorry

end range_of_m_l317_317456


namespace AE_eq_inradius_l317_317378

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the geometric configuration of the triangle, incenter, altitude, etc.
variables {A B C I M E H : ℝ × ℝ}
variables {r : ℝ} -- the inradius

-- Conditions
axiom midpoint_BC : M = midpoint B C
axiom incenter_def : I -- the incenter of triangle ABC
axiom altitude_AH : H.1 = A.1 ∧ H.2 = C.2 -- H is vertically aligned with A and on the same line as BC
axiom intersection_IM_AH : E -- intersection of IM and AH

-- Proof goal
theorem AE_eq_inradius :
  let AE := dist A E
  in AE = r := 
sorry

end AE_eq_inradius_l317_317378


namespace consecutive_product_last_digit_l317_317343

theorem consecutive_product_last_digit (n : ℤ) : ∃ d ∈ {0, 2, 6}, (n * (n + 1)) % 10 = d := sorry

end consecutive_product_last_digit_l317_317343


namespace trajectory_eq_range_of_k_l317_317429

-- definitions based on the conditions:
def fixed_circle (x y : ℝ) := (x + 1)^2 + y^2 = 16
def moving_circle_passing_through_B (M : ℝ × ℝ) (B : ℝ × ℝ) := 
    B = (1, 0) ∧ M.1^2 / 4 + M.2^2 / 3 = 1 -- the ellipse trajectory equation

-- question 1: prove the equation of the ellipse
theorem trajectory_eq :
    ∀ M : ℝ × ℝ, (∃ B : ℝ × ℝ, moving_circle_passing_through_B M B)
    → (M.1^2 / 4 + M.2^2 / 3 = 1) :=
sorry

-- question 2: find the range of k which satisfies given area condition
theorem range_of_k (k : ℝ) :
    (∃ M : ℝ × ℝ, ∃ B : ℝ × ℝ, moving_circle_passing_through_B M B) → 
    (0 < k) → (¬ (k = 0)) →
    ((∃ m : ℝ, (4 * k^2 + 3 - m^2 > 0) ∧ 
    (1 / 2) * (|k| * m^2 / (4 * k^2 + 3)^2) = 1 / 14) → (3 / 4 < k ∧ k < 1) 
    ∨ (-1 < k ∧ k < -3 / 4)) :=
sorry

end trajectory_eq_range_of_k_l317_317429


namespace calculate_change_l317_317959

def book_price : ℝ := 25
def pen_price : ℝ := 4
def ruler_price : ℝ := 1
def notebook_price : ℝ := 8
def pencil_case_price : ℝ := 6
def book_discount : ℝ := 0.10
def pen_discount : ℝ := 0.05
def sales_tax_rate : ℝ := 0.06
def payment : ℝ := 100

def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_cost (book pen ruler notebook pencil_case : ℝ) : ℝ :=
  book + pen + ruler + notebook + pencil_case

def apply_sales_tax (subtotal : ℝ) (tax_rate : ℝ) : ℝ :=
  subtotal * (1 + tax_rate)

theorem calculate_change :
  let book := discounted_price book_price book_discount
  let pen := discounted_price pen_price pen_discount
  let ruler := ruler_price
  let notebook := notebook_price
  let pencil_case := pencil_case_price
  let subtotal := total_cost book pen ruler notebook pencil_case
  let total_with_tax := apply_sales_tax subtotal sales_tax_rate in
  payment - total_with_tax = 56.22 := sorry

end calculate_change_l317_317959


namespace problem1_problem2_l317_317401

theorem problem1 (n : ℕ) : 2^n + 3 = k * k → n = 0 :=
by
  intros
  sorry 

theorem problem2 (n : ℕ) : 2^n + 1 = x * x → n = 3 :=
by
  intros
  sorry 

end problem1_problem2_l317_317401


namespace max_value_p_plus_q_l317_317711

theorem max_value_p_plus_q :
  let p := (-1) ^ 3 * Nat.choose 8 3 * (1/2) ^ 3 in
  let q := (-1) * Nat.choose 3 2 * 2^2 * (1/7) in
  p + q = -4 * Real.sqrt 3 :=
by sorry

end max_value_p_plus_q_l317_317711


namespace correct_equation_system_l317_317337

-- Define the conditions
def condition1 (x y : ℤ) : Prop := 8 * x - y = 3
def condition2 (x y : ℤ) : Prop := y - 7 * x = 4

-- Define the hypothesis for the problem
noncomputable def problem (x y : ℤ) : Prop :=
  condition1 x y ∧ condition2 x y

-- Prove that the system of equations is correct
theorem correct_equation_system (x y : ℤ) (h : problem x y) :
  (8 * x - y = 3 ∧ y - 7 * x = 4) :=
by
  cases h with h1 h2
  exact ⟨h1, h2⟩

end correct_equation_system_l317_317337


namespace max_sides_polygon_l317_317076

theorem max_sides_polygon (n : ℕ) 
  (angles : Fin n → ℕ)
  (distinct_angles : Function.Injective angles)
  (angle_sum : ∑ i, angles i = 180 * (n - 2))
  (largest_three_times_smallest : angles 0 = 3 * angles (n - 1))
  (angles_lt_180 : ∀ i, angles i < 180) :
  n ≤ 20 :=
by {
  sorry -- proof goes here
}

end max_sides_polygon_l317_317076


namespace find_x_coord_l317_317356

theorem find_x_coord : 
  ∃ x y : ℝ, (abs y = abs x) ∧ (abs (x + y - 4) / √2 = abs x) → x = 2 :=
by
  sorry

end find_x_coord_l317_317356


namespace greatest_gcd_of_natural_sum_l317_317573

theorem greatest_gcd_of_natural_sum (a : Fin 49 → ℕ) (h_sum : ∑ i, a i = 540) : gcd (λ i, a i) = 10 := 
sorry

end greatest_gcd_of_natural_sum_l317_317573


namespace valid_options_are_three_l317_317077

def symmetric_sequence (a : ℕ → ℕ) (n : ℕ) :=
  ∀ i, 1 ≤ i ∧ i ≤ n → a i = a (n - i + 1)

def first_m_terms_sequence (b : ℕ → ℕ) (m : ℕ) :=
  ∀ i, 1 ≤ i ∧ i ≤ m → b i = 2^(i-1)

def sum_of_first_2009_terms (S : ℕ) : Prop :=
  S = 2^(2009) - 1 ∨
  S = 2 * (2^(2009) - 1) ∨
  S = 3 * 2^(m - 1) - 2^(2*m - 2010) - 1 ∨
  S = 2^(m + 1) - 2^(2*m - 2009) - 1

theorem valid_options_are_three
  (b : ℕ → ℕ) (m : ℕ) (h_symm : symmetric_sequence b (2*m))
  (h_first_terms : first_m_terms_sequence b m) (h_m_gt_one : m > 1) :
  (sum_of_first_2009_terms (S 2009)) :=
sorry

end valid_options_are_three_l317_317077


namespace no_four_points_within_two_cm_l317_317706

theorem no_four_points_within_two_cm (P : Finset (EuclideanSpace ℝ (Fin 2))) 
  (hP_card : P.card = 12) 
  (hP_dist : ∀ p₁ p₂ ∈ P, dist p₁ p₂ ≤ 3) : 
  ¬ ∃ Q : Finset (EuclideanSpace ℝ (Fin 2)), Q.card = 4 ∧ ∀ q₁ q₂ ∈ Q, dist q₁ q₂ ≤ 2 :=
by
  sorry

end no_four_points_within_two_cm_l317_317706


namespace sum_of_squares_gt_10_l317_317846

-- Define the condition for the problem
def condition (a : List ℝ) : Prop :=
  a.length = 5 ∧ ∀ i j, i ≠ j → |a[i] - a[j]| > 1

-- Define the sum of squares function
def sum_of_squares (a : List ℝ) : ℝ :=
  a.map (λ x => x * x).sum

-- The main statement translating the problem to Lean 4
theorem sum_of_squares_gt_10 (a : List ℝ) (h : condition a) : sum_of_squares a > 10 :=
begin
  sorry -- Proof omitted as instructed
end

end sum_of_squares_gt_10_l317_317846


namespace triangle_ABC_area_is_six_l317_317534

noncomputable def area_of_triangle_ABC : ℝ :=
  let AB : ℝ := 6
  let BD : ℝ := 6
  let AC : ℝ := 12
  let a : ℝ := 2
  let k : ℝ := 3
  let CBE_α := π / 4
  let DBC_β := π / 4
  let area := 1 / 2 * 6 * 2 in
  if (tan CBE_α = 1 ∧ (tan (CBE_α + DBC_β) = (1 + a / BD) / (1 - a / BD))
    ∧ (k = 3) ∧ (BD = 6) ∧ (a = 2))
  then area else 0

theorem triangle_ABC_area_is_six :
    let α := area_of_triangle_ABC
    in α = 6 :=
by
  sorry

end triangle_ABC_area_is_six_l317_317534


namespace original_paint_intensity_l317_317719

theorem original_paint_intensity (I : ℝ) (h1 : 0.5 * I + 0.5 * 20 = 15) : I = 10 :=
sorry

end original_paint_intensity_l317_317719


namespace area_of_CDE_l317_317241

-- Define the areas of the given triangles
def AreaADF := 1 / 2
def AreaABF := 1
def AreaBEF := 1 / 4

-- The area of triangle CDE that we need to prove
def AreaCDE : ℝ := 15 / 56

-- Prove the area of triangle CDE is as claimed
theorem area_of_CDE (AreaADF AreaABF AreaBEF : ℝ) :
  AreaADF = 1 / 2 →
  AreaABF = 1 →
  AreaBEF = 1 / 4 →
  AreaCDE = 15 / 56 :=
by
  intros h1 h2 h3
  sorry

end area_of_CDE_l317_317241


namespace ratio_V_W_zero_l317_317507

theorem ratio_V_W_zero
    (O : Point)
    (r : ℝ) 
    (AB CD : Chord)
    (P Q S T U : Point)
    (h_center : O.is_center_of_circle(O, r))
    (h_radius : circle_radius(O, r))
    (h_AB_mid : P.is_midpoint(AB))
    (h_CD_mid : Q.is_midpoint(CD))
    (h_collinear : collinear({O, S, T, U}))
    (h_CD_midpoint : Q.is_midpoint(CD))
    (h_AB_fixed : is_fixed(AB))
    (h_CD_translate : can_translate_vertically(CD))
    (h_distance_OS : OS_vertical_movement(OS -> r))
    (V W : Area)
    (h_V : V = area_quad(S, AB, T))
    (h_W : W = area_quad(O, CD, U)) :
  limit (OS -> r) (V / W) = 0 :=
sorry

end ratio_V_W_zero_l317_317507


namespace matrix_projection_l317_317415

variable {α : Type*} [CommRing α] [Module α (Matrix (Fin 3) (Fin 1) α)]
open Matrix

def v  : Fin 3 → α := ![x, y, z]
def u  : Fin 3 → α := ![1, -1, 2]

def P : Matrix (Fin 3) (Fin 3) α :=
  ![
    [1 / 6, -1 / 6, 1 / 3],
    [-1 / 6,  1 / 6 , -1 / 3],
    [1 / 3, -1 / 3,  2 / 3]
  ]

theorem matrix_projection (v : Fin 3 → α) : P.mulVec v = ((x - y + 2*z) / 6) • u :=
sorry

end matrix_projection_l317_317415


namespace sin_angle_A_when_c_equals_5_range_of_c_when_angle_A_is_obtuse_l317_317908

-- Define the coordinates of the points
def A := (3, 4)
def B := (0, 0)
def C (c : ℝ) := (c, 0)

-- Problem 1: Prove sin(angle A) when c = 5
theorem sin_angle_A_when_c_equals_5 : 
  sin (∠ ABC A B (C 5)) = 2 * real.sqrt 5 / 5 :=
by 
  sorry

-- Problem 2: Prove range of c when angle A is obtuse
theorem range_of_c_when_angle_A_is_obtuse : 
  (∠ ABC A B (C c) > π / 2) → c > 25 / 3 :=
by 
  sorry

end sin_angle_A_when_c_equals_5_range_of_c_when_angle_A_is_obtuse_l317_317908


namespace intersection_of_complement_l317_317984

open Set

variable (U : Set ℤ) (A B : Set ℤ)

def complement (U A : Set ℤ) : Set ℤ := U \ A

theorem intersection_of_complement (hU : U = {-1, 0, 1, 2, 3, 4})
  (hA : A = {1, 2, 3, 4}) (hB : B = {0, 2}) :
  (complement U A) ∩ B = {0} :=
by
  sorry

end intersection_of_complement_l317_317984


namespace log_condition_l317_317272

theorem log_condition (x : ℝ) : x > 2 → log 2 x > 0 ∧ (∃ (x : ℝ), log 2 x > 0 ∧ ¬(x > 2)) :=
by 
  sorry

end log_condition_l317_317272


namespace find_bc_l317_317505

theorem find_bc (A : ℝ) (a : ℝ) (area : ℝ) (b c : ℝ) :
  A = 60 * (π / 180) → a = Real.sqrt 7 → area = (3 * Real.sqrt 3) / 2 →
  ((b = 3 ∧ c = 2) ∨ (b = 2 ∧ c = 3)) :=
by
  intros hA ha harea
  -- From the given area condition, derive bc = 6
  have h1 : b * c = 6 := sorry
  -- From the given conditions, derive b + c = 5
  have h2 : b + c = 5 := sorry
  -- Solve the system of equations to find possible values for b and c
  -- Using x² - S⋅x + P = 0 where x are roots, S = b + c, P = b⋅c
  have h3 : (b = 3 ∧ c = 2) ∨ (b = 2 ∧ c = 3) := sorry
  exact h3

end find_bc_l317_317505


namespace correct_propositions_l317_317110

/-
Given the following four propositions:

① The necessary and sufficient condition for "line a is parallel to line b" is "a is parallel to the plane where b lies".

② The necessary and sufficient condition for "line l is perpendicular to all lines in plane α" is "l is perpendicular to plane α".

③ A sufficient but not necessary condition for "lines a and b are skew lines" is "lines a and b do not intersect".

④ A necessary but not sufficient condition for "plane α is parallel to plane β" is "there exist three non-collinear points in α that are equidistant from β".

We need to prove that propositions ② and ④ are correct.
-/

variables (a b l : Type) (α β : Type) [Plane α] [Plane β] [Line a] [Line b] [Line l]

/-- Propositions --/
def prop1 : Prop := (∀ a b : Line, (a ∥ b) ↔ (a ∥ plane_of b))
def prop2 : Prop := (∀ l : Line, ∀ α : Plane, (∀ l', l' ∈ α → l ⟂ l') ↔ l ⟂ α)
def prop3 : Prop := (∀ a b : Line, (a ∩ b = ∅) → (a and b are skew lines))
def prop4 : Prop := (∀ α β : Plane, (∃ (A B C : Point), non-collinear A B C ∧ distance_to_plane A β = distance_to_plane B β ∧ distance_to_plane C β = distance_to_plane B β) → α ∥ β)

/-- The correct propositions are ② and ④ --/
theorem correct_propositions : prop2 ∧ prop4 :=
by {
  -- placeholder for the proof, not required for the task
  sorry
}

end correct_propositions_l317_317110


namespace min_value_expr_l317_317442

theorem min_value_expr (a : ℝ) (h₁ : 0 < a) (h₂ : a < 3) : 
  ∃ m : ℝ, (∀ x : ℝ, 0 < x → x < 3 → (1/x + 9/(3 - x)) ≥ m) ∧ m = 16 / 3 :=
sorry

end min_value_expr_l317_317442


namespace percent_square_area_in_rectangle_l317_317746

theorem percent_square_area_in_rectangle (s : ℝ) : 
  let width_rectangle := 3 * s,
  let length_rectangle := (3 / 2) * width_rectangle,
  let area_square := s^2,
  let area_rectangle := length_rectangle * width_rectangle
  in (area_square / area_rectangle) * 100 = 7.41 := 
by
  let width_rectangle := 3 * s,
  let length_rectangle := (3 / 2) * width_rectangle,
  let area_square := s^2,
  let area_rectangle := length_rectangle * width_rectangle
  calc
    (area_square / area_rectangle) * 100 
        = (s^2 / (4.5 * 3 * s^2)) * 100 : by sorry
    ... = (1 / 13.5) * 100 : by sorry
    ... = 7.41 : by sorry

end percent_square_area_in_rectangle_l317_317746


namespace max_four_digit_integer_satisfying_conditions_l317_317472

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 1000 + (n / 10 % 10) * 100 + (n / 100 % 10) * 10 + (n / 1000)

theorem max_four_digit_integer_satisfying_conditions :
  ∃ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧
           (m % 36 = 0) ∧
           (m % 11 = 0) ∧
           (let n := reverse_digits m in n % 36 = 0) ∧
           ∀ x : ℕ, (1000 ≤ x ∧ x ≤ 9999) ∧
                    (x % 36 = 0) ∧
                    (x % 11 = 0) ∧
                    (let y := reverse_digits x in y % 36 = 0) → x ≤ m :=
  ∃ m : ℕ, (m = 9504)
sorry

end max_four_digit_integer_satisfying_conditions_l317_317472


namespace number_of_ways_l317_317715

-- Define the advertisements
inductive Advertisement
| Commercial (id : ℕ)
| WorldExpo (id : ℕ)

-- There are 5 advertisements, 3 commercials and 2 World Expo
def ads := [Advertisement.Commercial 1, Advertisement.Commercial 2, Advertisement.Commercial 3,
            Advertisement.WorldExpo 1, Advertisement.WorldExpo 2]

-- Total advertisements should be 5
axiom total_ads : ads.length = 5

-- Number of commercials should be 3
axiom num_commercials : (ads.filter (λ ad, match ad with
                                       | Advertisement.Commercial _ => true
                                       | _ => false end)).length = 3

-- Number of World Expo ads should be 2
axiom num_worldexpo : (ads.filter (λ ad, match ad with
                                    | Advertisement.WorldExpo _ => true
                                    | _ => false end)).length = 2

-- The last advertisement should be a World Expo ad
axiom last_ad_worldexpo : ∃ ad, ad = ads.last' none ∧ match ad with
                                       | some (Advertisement.WorldExpo _) => true
                                       | _ => false end

-- The World Expo ads should not be broadcasted consecutively
axiom not_consecutive : ∀ a b c d e, ads = [a, b, c, d, e] →
                        ¬ ((a = Advertisement.WorldExpo 1 ∧ b = Advertisement.WorldExpo 2) ∨
                           (b = Advertisement.WorldExpo 1 ∧ c = Advertisement.WorldExpo 2) ∨
                           (c = Advertisement.WorldExpo 1 ∧ d = Advertisement.WorldExpo 2) ∨
                           (d = Advertisement.WorldExpo 1 ∧ e = Advertisement.WorldExpo 2))

-- The number of valid arrangements should be 36
theorem number_of_ways : ∃ n, n = 36 ∧ number_of_ways_to_broadcast ads = n := sorry

end number_of_ways_l317_317715


namespace find_ellipse_equation_l317_317084

noncomputable section

def ellipse_equation (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def line_equation (x : ℝ) : ℝ := x + 3

def intersects_at_one_point (a b : ℝ) : Prop :=
  let discriminant := (a^2 + b^2) * 1 + 6 * a^2 * 1 + 9 * a^2 - a^2 * b^2 in
  discriminant = 0

def eccentricity (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ √(1 - b^2 / a^2) = √(5) / 5

def ellipse_is (a b : ℝ) (a' b' : ℝ) : Prop :=
  a' = 5 ∧ b' = 4

theorem find_ellipse_equation (a b : ℝ)
 (h1: ∀ x y, ellipse_equation x y a b)
 (h2: intersects_at_one_point a b)
 (h3: eccentricity a b) :
 ellipse_is a b 5 4 :=
sorry

end find_ellipse_equation_l317_317084


namespace sequence_third_term_l317_317125

theorem sequence_third_term (a m : ℤ) (h_a_neg : a < 0) (h_a1 : a + m = 2) (h_a2 : a^2 + m = 4) :
  (a^3 + m = 2) :=
by
  sorry

end sequence_third_term_l317_317125


namespace cartesian_equation_of_curve_C_perpendicular_OA_OB_value_l317_317122

-- Definitions
def polar_equation (ρ θ : ℝ) := ρ^2 = 9 / (cos θ^2 + 9 * sin θ^2)
def cartesian_coordinates (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)
def perpendicular (α : ℝ) (ρ1 ρ2 : ℝ) := (ρ1, α) ≠ (ρ2, α ± π/2)

-- The lean statement includes the two main points needed to prove
theorem cartesian_equation_of_curve_C (ρ θ : ℝ) (h : polar_equation ρ θ) :
  let (x, y) := cartesian_coordinates ρ θ in
  (x^2 / 9 + y^2 = 1) := sorry

theorem perpendicular_OA_OB_value (ρ1 ρ2 α : ℝ) (h1 : polar_equation ρ1 α) (h2 : polar_equation ρ2 (α + π/2) ∨ polar_equation ρ2 (α - π/2)) :
  (1 / ρ1^2 + 1 / ρ2^2 = 10 / 9) := sorry

end cartesian_equation_of_curve_C_perpendicular_OA_OB_value_l317_317122


namespace difference_between_m_and_n_l317_317282

theorem difference_between_m_and_n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 10 * 2^m = 2^n + 2^(n + 2)) :
  n - m = 1 :=
sorry

end difference_between_m_and_n_l317_317282


namespace no_real_pairs_exist_l317_317817

theorem no_real_pairs_exist : ∀ (a b : ℝ), 
  (8, a, b, a * b) = λ i : ℕ, 8 + i * ((a - 8) / 2) ∧ a * (1 + b) = 3 * b 
  → False :=
by sorry

end no_real_pairs_exist_l317_317817


namespace minimize_abs_z_l317_317557

noncomputable def minimumModulus (z : ℂ) : ℝ :=
  if h : |z - 8| + |z - 7 * complex.I| = 17 then
    min (abs z) (17 / real.sqrt 113)
  else
    0

theorem minimize_abs_z (z : ℂ) (h : |z - 8| + |z - 7 * complex.I| = 17) : 
  abs z = 7 / real.sqrt 113 :=
by
  sorry

end minimize_abs_z_l317_317557


namespace complex_sum_exp_l317_317003

open Complex

theorem complex_sum_exp : (∑ k in finset.range 16, exp (2 * π * I * (k + 1) / 17)) = -1 := 
  sorry

end complex_sum_exp_l317_317003


namespace quadratic_inequality_solution_l317_317649

theorem quadratic_inequality_solution (b c : ℝ) 
    (h1 : ∀ x : ℝ, (1 < x ∧ x < 2) → x^2 + b * x + c < 0) :
    b + c = -1 :=
sorry

end quadratic_inequality_solution_l317_317649


namespace find_a_l317_317904

-- Definitions
def isInInterval (x : ℝ) := 1 < x ∧ x < 2

def expInInterval (x : ℝ) (a : ℝ) := exp x - a > 0

-- Lean 4 statement
theorem find_a : (∀ x, isInInterval x → exp x - a ≤ 0) → a ≥ exp 2 :=
by
  intro h
  sorry

end find_a_l317_317904


namespace decagon_OP_eq_AB_l317_317432

noncomputable def OP_eq_AB (r x : ℝ) (O A B P : Point) : Prop :=
  OA = r ∧ OB = r ∧ AB = x ∧ angle O A B = 36 ∧ angle O B A = 72 ∧ OP^2 = OB * PB → OP = x

theorem decagon_OP_eq_AB {r x : ℝ} {O A B P : Point}
  (h_OA : distance O A = r) 
  (h_OB : distance O B = r)
  (h_AB : distance A B = x)
  (h_angle_OAB : angle O A B = 36)
  (h_angle_OBA : angle O B A = 72)
  (h_OP_condition : distance O P ^ 2 = distance O B * distance P B) : distance O P = x := sorry

end decagon_OP_eq_AB_l317_317432


namespace midpoint_locus_l317_317826

noncomputable theory
open_locale classical

-- Define the point P
def P : ℝ × ℝ := (4, -2)

-- Define the condition for Q being on the circle
def is_on_circle (Q : ℝ × ℝ) : Prop := Q.1 ^ 2 + Q.2 ^ 2 = 4

-- Define the midpoint formula M for points P and Q
def M (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the condition for the locus of M as (x - 2)^2 + (y + 1)^2 = 1
def is_on_locus (M : ℝ × ℝ) : Prop := (M.1 - 2) ^ 2 + (M.2 + 1) ^ 2 = 1

-- Theorem statement
theorem midpoint_locus (Q : ℝ × ℝ) (hQ : is_on_circle Q) :
  is_on_locus (M P Q) :=
sorry  -- Proof is omitted.

end midpoint_locus_l317_317826


namespace gina_snake_mice_eaten_in_decade_l317_317839

-- Define the constants and conditions
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10
def weeks_per_decade : ℕ := years_per_decade * weeks_per_year
def mouse_eating_period : ℕ := 4

-- The problem to prove
theorem gina_snake_mice_eaten_in_decade : (weeks_per_decade / mouse_eating_period) = 130 := 
by
  -- The proof would typically go here, but we skip it
  sorry

end gina_snake_mice_eaten_in_decade_l317_317839


namespace simplify_cub_root_multiplication_l317_317601

theorem simplify_cub_root_multiplication (a b : ℝ) (ha : a = 8) (hb : b = 27) :
  (real.cbrt (a + b) * real.cbrt (a + real.cbrt b)) = real.cbrt ((a + b) * (a + real.cbrt b)) := 
by
  sorry

end simplify_cub_root_multiplication_l317_317601


namespace general_term_formula_l317_317275

theorem general_term_formula :
  ∀ n : ℕ, n > 0 → sequence_term n = (n + 2) / (3 * n + 2) :=
by
  sorry

end general_term_formula_l317_317275


namespace range_of_k_l317_317156

noncomputable def f (x : ℝ) : ℝ := x^3 - 12*x
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 12

theorem range_of_k (k : ℝ) : ¬ monotone_on f (set.Ioo (k-1) (k+1)) ↔ (k > 1 ∧ k < 3) ∨ (k > -3 ∧ k < -1) := by
  sorry

end range_of_k_l317_317156


namespace base2_to_base4_conversion_l317_317670

def base2_to_nat (s : String) : Nat :=
  s.foldl (λ n d => 2 * n + (if d = '1' then 1 else 0)) 0

def nat_to_base4 (n : Nat) : String := 
  if n = 0 then "0"
  else
    let rec loop (n : Nat) (acc : List Char) : List Char :=
      if n = 0 then acc
      else
        let remainder := n % 4
        let digit := Char.ofNat (remainder + '0'.val)
        loop (n / 4) (digit :: acc)
    (loop n []).asString

theorem base2_to_base4_conversion :
  nat_to_base4 (base2_to_nat "10101010") = "2212" :=
by
  sorry

end base2_to_base4_conversion_l317_317670


namespace sequence_not_perfect_square_l317_317147

-- Define the sequence of numbers.
def N (i : ℕ) : ℕ := 
  let zeros := String.mk $ List.replicate (i - 1) '0'
  ("2014" ++ zeros ++ "2015").to_nat

-- Main theorem stating that none of the numbers in the sequence are perfect squares.
theorem sequence_not_perfect_square (i : ℕ) : ¬ ∃ x : ℕ, x * x = N i := by
  sorry

end sequence_not_perfect_square_l317_317147


namespace parabola_focus_line_circle_intersection_hyperbola_equation_function_extreme_value_l317_317340

-- Problem 1
theorem parabola_focus (p : ℝ) (h : p = 2 * 2): (∀ x y : ℝ, y^2 = 8 * x ↔ (x,y) = (2,0)) := by
sory 

-- Problem 2
theorem line_circle_intersection (k : ℝ) : |(1 - k) / sqrt 2| < sqrt 2 → -1 < k ∧ k < 3 := by
sorry

-- Problem 3
theorem hyperbola_equation (a b c : ℝ) (h : c = 4 * sqrt 3 ∧ a / b = 1 ∧ a = b ∧ 2 * a^2 = 48) : 
∀ x y : ℝ, y^2 = x^2 + 24 := by
sorry

-- Problem 4
theorem function_extreme_value (a x : ℝ) (h : ∀ y, 3*y^2 + 2*a*y + a + 6 ≠ 0) : 
-3 ≤ a ∧ a ≤ 6 := by
sorry

end parabola_focus_line_circle_intersection_hyperbola_equation_function_extreme_value_l317_317340


namespace probability_same_number_on_four_dice_l317_317676

open Probability

/-- The probability that the same number will be facing up on each of four eight-sided dice that are tossed simultaneously. -/
def prob_same_number_each_four_dice : ℚ := 1 / 512

/-- Given that the dice are eight-sided, four dice are tossed,
and the results on the four dice are independent of each other.
Prove that the probability that the same number will be facing up on each of these four dice is 1/512. -/
theorem probability_same_number_on_four_dice (n : ℕ) (s : Fin n → Fin 8) :
  n = 4 → (∀ i, ∃ k, s i = k) → prob_same_number_each_four_dice = 1 / 512 :=
by intros; simp; sorry

end probability_same_number_on_four_dice_l317_317676


namespace probability_intersection_inside_nonagon_correct_l317_317713

def nonagon_vertices : ℕ := 9

def total_pairs_of_points := Nat.choose nonagon_vertices 2

def sides_of_nonagon : ℕ := nonagon_vertices

def diagonals_of_nonagon := total_pairs_of_points - sides_of_nonagon

def pairs_of_diagonals := Nat.choose diagonals_of_nonagon 2

def sets_of_intersecting_diagonals := Nat.choose nonagon_vertices 4

noncomputable def probability_intersection_inside_nonagon : ℚ :=
  sets_of_intersecting_diagonals / pairs_of_diagonals

theorem probability_intersection_inside_nonagon_correct :
  probability_intersection_inside_nonagon = 14 / 39 := 
  sorry

end probability_intersection_inside_nonagon_correct_l317_317713


namespace percent_absent_is_correct_l317_317463

def total_students : ℕ := 180
def boys : ℕ := 80
def girls : ℕ := 100
def fraction_boys_absent : ℚ := 1 / 8
def fraction_girls_absent : ℚ := 2 / 5

def num_boys_absent : ℕ := fraction_boys_absent * boys
def num_girls_absent : ℕ := fraction_girls_absent * girls
def total_absent_students : ℕ := num_boys_absent + num_girls_absent
def absent_fraction : ℚ := total_absent_students / total_students

theorem percent_absent_is_correct : 
  (absent_fraction * 100 : ℚ).to_real ≈ 27.78 :=
sorry

end percent_absent_is_correct_l317_317463


namespace area_of_CDE_l317_317242

-- Define the areas of the given triangles
def AreaADF := 1 / 2
def AreaABF := 1
def AreaBEF := 1 / 4

-- The area of triangle CDE that we need to prove
def AreaCDE : ℝ := 15 / 56

-- Prove the area of triangle CDE is as claimed
theorem area_of_CDE (AreaADF AreaABF AreaBEF : ℝ) :
  AreaADF = 1 / 2 →
  AreaABF = 1 →
  AreaBEF = 1 / 4 →
  AreaCDE = 15 / 56 :=
by
  intros h1 h2 h3
  sorry

end area_of_CDE_l317_317242


namespace parallelogram_area_l317_317699

theorem parallelogram_area (base height : ℝ) (h_base : base = 20) (h_height : height = 16) :
  base * height = 320 :=
by
  sorry

end parallelogram_area_l317_317699


namespace optimal_game_distinct_rows_l317_317580

theorem optimal_game_distinct_rows :
  ∀ (A B : ℕ), A = 2^100 ∧ B = 100 ∧
  (∀ i, (i ∈ (fin B))),
  players_use_optimal_strategy → 
  distinct_rows A B = 100 :=
by
  assume A B h1 h2,
  sorry

end optimal_game_distinct_rows_l317_317580


namespace square_of_85_l317_317013

-- Define the given variables and values
def a := 80
def b := 5
def c := a + b

theorem square_of_85:
  c = 85 → (c * c) = 7225 :=
by
  intros h
  rw h
  sorry

end square_of_85_l317_317013


namespace find_c_l317_317922

def is_square_of_binomial (p : ℚ[X]) : Prop :=
  ∃ (a b : ℚ), p = (a * X + b)^2

theorem find_c : 
  ∀ (c : ℚ), is_square_of_binomial (9 * X^2 - 27 * X + c) → c = 20.25 := 
by
  sorry

end find_c_l317_317922


namespace perimeter_triangle_ABC_l317_317191

noncomputable def PerimeterTriangle (A B C : Point) : ℝ := 
  dist A B + dist B C + dist C A

structure Circle (center : Point) (radius : ℝ) :=
  (center := center)
  (radius := radius)

def centers_ABC_tangent (A B C P Q R S T : Point) :=
  dist P Q = 4 ∧ dist P R = 4 ∧ dist P S = 4 ∧ dist Q R = 4 ∧ dist Q S = 4 ∧ dist R S = 4 ∧
  dist P T = 4 ∧ dist Q T = 4 ∧ T ∈ line_segment A B

theorem perimeter_triangle_ABC :
  ∃ (A B C P Q R S T : Point),
    (Circle P 2 ∧ Circle Q 2 ∧ Circle R 2 ∧ Circle S 2 ∧ Circle T 2) ∧
    centers_ABC_tangent A B C P Q R S T →
    PerimeterTriangle A B C = 20 :=
by
  sorry

end perimeter_triangle_ABC_l317_317191


namespace find_roots_l317_317121

-- Given the conditions:
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Given points (x, y)
def points := [(-5, 6), (-4, 0), (-2, -6), (0, -4), (2, 6)] 

-- Prove that the roots of the quadratic equation are -4 and 1
theorem find_roots (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : quadratic_function a b c (-5) = 6)
  (h₂ : quadratic_function a b c (-4) = 0)
  (h₃ : quadratic_function a b c (-2) = -6)
  (h₄ : quadratic_function a b c (0) = -4)
  (h₅ : quadratic_function a b c (2) = 6) :
  ∃ x₁ x₂ : ℝ, quadratic_function a b c x₁ = 0 ∧ quadratic_function a b c x₂ = 0 ∧ x₁ = -4 ∧ x₂ = 1 := 
sorry

end find_roots_l317_317121


namespace extra_yellow_balls_dispatched_l317_317756

/-- A sports retailer originally ordered 288 tennis balls in equal numbers of white and yellow.
    Due to a mistake, the ratio of white balls to yellow balls changed to 8/13. -/
theorem extra_yellow_balls_dispatched :
  ∃ (original_white original_yellow dispatched_white dispatched_yellow n x : ℕ),
    original_white = 144 ∧
    original_yellow = 144 ∧
    original_white + original_yellow = 288 ∧
    dispatched_white = 8 * x ∧
    dispatched_yellow = 13 * x ∧
    dispatched_white + dispatched_yellow = 288 ∧
    n = dispatched_yellow - original_yellow ∧
    n = 25 :=
begin
  sorry
end

end extra_yellow_balls_dispatched_l317_317756


namespace f_6_eq_3_l317_317155

def f : ℤ → ℤ
| n := if n = 4 then 14 else f (n - 1) - n

theorem f_6_eq_3 : f 6 = 3 := by
  sorry

end f_6_eq_3_l317_317155


namespace quadratic_square_binomial_l317_317313

theorem quadratic_square_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 18 * x + k = (x + b) ^ 2) ↔ k = 81 := by
  sorry

end quadratic_square_binomial_l317_317313


namespace polygon_inequality_l317_317999

variable {n : ℕ} (n_geq_3 : n ≥ 3)
variable (A : Fin n → Fin n → ℝ)
variable (convex : ∀ p q, 0 ≤ p < n ∧ 0 ≤ q < n → A p q > 0)
variable (triangle_inequality : ∀ p q, |p - q| ≥ 2 → A p (p + 1) + A q (q + 1) < A p q + A (p + 1) (q + 1))

theorem polygon_inequality (hp : 0 ≤ p) (hq : 0 ≤ q) : 
    (1 / n) * ∑ i in Finset.range n, A i (i + 1) < (2 / (n * (n - 3))) * ∑ i j in Finset.range n, if |i - j| ≥ 2 then A i j else 0 := 
    sorry

end polygon_inequality_l317_317999


namespace largest_share_of_partner_l317_317626

theorem largest_share_of_partner 
    (ratios : List ℕ := [2, 3, 4, 4, 6])
    (total_profit : ℕ := 38000) :
    let total_parts := ratios.sum
    let part_value := total_profit / total_parts
    let largest_share := List.maximum ratios * part_value
    largest_share = 12000 :=
by
    let total_parts := ratios.sum
    let part_value := total_profit / total_parts
    let largest_share := List.maximum ratios * part_value
    have h1 : total_parts = 19 := by
        sorry
    have h2 : part_value = 2000 := by
        sorry
    have h3 : List.maximum ratios = 6 := by
        sorry
    have h4 : largest_share = 12000 := by
        sorry
    exact h4


end largest_share_of_partner_l317_317626


namespace sum_of_products_not_zero_l317_317162

def grid25 : Type := Fin 25 × Fin 25 → ℤ

noncomputable def valid_grid (g : grid25) : Prop := 
  ∀ i, g i = 1 ∨ g i = -1

noncomputable def row_product (g : grid25) (i : Fin 25) : ℤ :=
  ∏ j, g (i, j)

noncomputable def column_product (g : grid25) (j : Fin 25) : ℤ :=
  ∏ i, g (i, j)

theorem sum_of_products_not_zero (g : grid25) (h : valid_grid g) : 
  (∑ i, row_product g i + ∑ j, column_product g j) ≠ 0 :=
sorry

end sum_of_products_not_zero_l317_317162


namespace area_bounded_region_l317_317628

theorem area_bounded_region (x y : ℝ) (h : y^2 + 2*x*y + 30*|x| = 300) : 
  ∃ A, A = 900 := 
sorry

end area_bounded_region_l317_317628


namespace directrix_of_parabola_l317_317410

theorem directrix_of_parabola (h : ∀ x : ℝ, y = -3 * x ^ 2 + 6 * x - 5) : ∃ y : ℝ, y = -25 / 12 :=
by
  sorry

end directrix_of_parabola_l317_317410


namespace smallest_perfect_square_divisible_by_5_and_6_l317_317682

-- 1. Define the gcd and lcm functionality
def lcm (a b : ℕ) : ℕ :=
  (a * b) / Nat.gcd a b

-- 2. Define the condition that a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- 3. State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 : ∃ n : ℕ, is_perfect_square n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (is_perfect_square m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m :=
  sorry

end smallest_perfect_square_divisible_by_5_and_6_l317_317682


namespace range_of_a_for_critical_point_l317_317893

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 9 * Real.log x

theorem range_of_a_for_critical_point :
  ∀ a : ℝ, (∃ x ∈ Set.Icc (a - 1) (a + 1), deriv f x = 0) ↔ 2 < a ∧ a < 4 :=
by
  sorry

end range_of_a_for_critical_point_l317_317893


namespace simplify_expression_l317_317591

-- Define the constants.
def a : ℚ := 8
def b : ℚ := 27

-- Assuming cube root function is available and behaves as expected for rationals.
def cube_root (x : ℚ) : ℚ := x^(1/3 : ℚ)

-- Assume the necessary property of cube root of 27.
axiom cube_root_27_is_3 : cube_root 27 = 3

-- The main statement to prove.
theorem simplify_expression : cube_root (a + b) * cube_root (a + cube_root b) = cube_root 385 :=
by
  sorry

end simplify_expression_l317_317591


namespace max_a_for_minimum_value_l317_317553

def f (a : ℝ) (x : ℝ) : ℝ :=
if x < a then -a * x + 3 else (x - 3) * Real.exp x + Real.exp 2

def has_minimum_value (f : ℝ → ℝ) : Prop :=
∃ m : ℝ, ∀ x : ℝ, f x ≥ m

theorem max_a_for_minimum_value :
  ∀ a : ℝ, (has_minimum_value (f a)) → a ≤ Real.sqrt 3 := 
by
  sorry

end max_a_for_minimum_value_l317_317553


namespace valid_third_side_length_l317_317102

theorem valid_third_side_length (x : ℝ) : 4 < x ∧ x < 14 ↔ (((5 : ℝ) + 9 > x) ∧ (x + 5 > 9) ∧ (x + 9 > 5)) :=
by 
  sorry

end valid_third_side_length_l317_317102


namespace xy_product_l317_317919

variable (x y: ℝ)

-- Define the conditions
def cond1 : Prop := 8^x / 4^(x + y) = 32
def cond2 : Prop := 16^(x + y) / 4^(5 * y) = 1024

-- Prove that the product xy = 25 given the conditions
theorem xy_product (h1 : cond1 x y) (h2 : cond2 x y) : x * y = 25 := 
by sorry

end xy_product_l317_317919


namespace num_pairs_reals_l317_317016

/-- 
The number of ordered pairs of integers (x, y) with 1 ≤ x < y ≤ 100 
such that i^x + i^y is a real number is 1850.
-/
theorem num_pairs_reals : 
  (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 1850 ∧
    ∀ p ∈ pairs, let ⟨x,y⟩ := p in 
    1 ≤ x ∧ x < y ∧ y ≤ 100 ∧ (Complex.I ^ x + Complex.I ^ y).im = 0) := 
sorry

end num_pairs_reals_l317_317016


namespace unpainted_area_on_five_inch_board_l317_317664

theorem unpainted_area_on_five_inch_board
    (width_five_inch_board : ℝ)
    (width_seven_inch_board : ℝ)
    (angle_between_boards : ℝ) :
    width_five_inch_board = 5 →
    width_seven_inch_board = 7 →
    angle_between_boards = 45 →
    let base := width_seven_inch_board / Real.sin (Real.pi / 4)
    let height := width_five_inch_board in
    width_five_inch_board * base * height = 35 * Real.sqrt 2 :=
by
    intros
    sorry

end unpainted_area_on_five_inch_board_l317_317664


namespace carla_math_textbooks_probability_l317_317785

def boxes : Type := Fin 3

def textbooks : Type := Fin 15
def math_textbooks : {t : textbooks // ∃ i : Fin 4, t = (⟨i, by simp [Nat.le_add_left]⟩ : textbooks)}

def binom (n k : ℕ) : ℕ := Nat.choose n k

def prob_all_math_textbooks_in_same_box : ℚ :=
  let total_ways := binom 15 4 * binom 11 5 * binom 6 6
  let favourable_ways :=
    binom 11 5 +        -- all four math textbooks in the first box
    binom 11 1 * binom 10 4 +  -- all four math textbooks in the second box
    binom 11 2 * binom 9 4     -- all four math textbooks in the third box
  (favourable_ways : ℚ) / total_ways

theorem carla_math_textbooks_probability :
  let p := prob_all_math_textbooks_in_same_box
  let m := 18
  let n := 1173
  p = (18 / 1173) ∧ m + n = 1191 :=
by
  sorry

end carla_math_textbooks_probability_l317_317785


namespace value_of_a_l317_317020

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := x^2 + 2

theorem value_of_a (a : ℝ) (ha : a > 1) (h : f (g a) = 12) : 
  a = Real.sqrt (Real.sqrt 10 - 2) :=
by sorry

end value_of_a_l317_317020


namespace log_product_eq_one_l317_317339

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_product_eq_one :
  log_base 5 2 * log_base 4 25 = 1 := 
by
  sorry

end log_product_eq_one_l317_317339


namespace three_consecutive_arithmetic_sequence_exists_arithmetic_sequence_r_s_no_four_terms_arithmetic_sequence_l317_317852

noncomputable def a (n : ℕ) : ℤ := 2^n - (-1)^n

theorem three_consecutive_arithmetic_sequence :
  ∃ k : ℕ, 1 ≤ k ∧ 2 a (k + 1) = a k + a (k + 2) :=
exists.intro 2 (by triv; sorry)

theorem exists_arithmetic_sequence_r_s :
  ∃ (r s : ℕ), 1 < r ∧ r < s ∧ 2 a r = 3 + a s ∧ s = r + 1 :=
exists.intro 2 (exists.intro 3 (by triv; sorry))

theorem no_four_terms_arithmetic_sequence :
  ∀ (q r s t : ℕ), q < r ∧ r < s ∧ s < t →
  a q + a t ≠ a r + a s :=
by
  intros q r s t h
  sorry

end three_consecutive_arithmetic_sequence_exists_arithmetic_sequence_r_s_no_four_terms_arithmetic_sequence_l317_317852


namespace sin_sum_eq_4_sin_product_cot_sum_eq_cot_product_l317_317246

theorem sin_sum_eq_4_sin_product (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 4 * Real.sin α * Real.sin β * Real.sin γ :=
sorry

theorem cot_sum_eq_cot_product (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  Real.cot (α / 2) + Real.cot (β / 2) + Real.cot (γ / 2) = Real.cot (α / 2) * Real.cot (β / 2) * Real.cot (γ / 2) :=
sorry

end sin_sum_eq_4_sin_product_cot_sum_eq_cot_product_l317_317246


namespace real_solution_unique_l317_317402

variable (x : ℝ)

theorem real_solution_unique :
  (x ≠ 2 ∧ (x^3 - 3 * x^2) / (x^2 - 4 * x + 4) + x = 3) ↔ x = 1 := 
by 
  sorry

end real_solution_unique_l317_317402


namespace mode_of_dataSet_l317_317634

def dataSet : List ℕ := [5, 4, 4, 3, 6, 2]

def mode (l : List ℕ) : ℕ :=
  (l.foldl (λ m x, m.insert x (m.findD x 0 + 1)) ∅).maxByOption (λ k v, v).iget.key

theorem mode_of_dataSet : mode dataSet = 4 :=
by
  sorry

end mode_of_dataSet_l317_317634


namespace omega_sum_zero_l317_317550

theorem omega_sum_zero {ω : ℂ} (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^{21} + ω^{24} + ω^{27} + ω^{30} + ω^{33} + ω^{36} + ω^{39} + ω^{42} + ω^{45} + ω^{48} + ω^{51} + ω^{54} + ω^{57} + ω^{60} + ω^{63} = 0 := 
sorry

end omega_sum_zero_l317_317550


namespace smallest_positive_perfect_square_divisible_by_5_and_6_l317_317685

theorem smallest_positive_perfect_square_divisible_by_5_and_6 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m * m) ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (∀ k : ℕ, (∃ p : ℕ, k = p * p) ∧ k % 5 = 0 ∧ k % 6 = 0 → n ≤ k) := 
sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_l317_317685


namespace probability_satisfies_condition_l317_317225

-- Define the sets for x and y
def X : Finset ℤ := { -1, 1 }
def Y : Finset ℤ := { -2, 0, 2 }

-- Define the condition to check
def satisfies_condition (x y : ℤ) : Prop :=
  x + 2 * y ≥ 1

-- Calculate the probability
theorem probability_satisfies_condition : (Finset.card ((X.product Y).filter (λ p, satisfies_condition p.1 p.2))).toRat / (X.card * Y.card) = 1 / 2 :=
by
  sorry

end probability_satisfies_condition_l317_317225


namespace bead_2000_is_white_l317_317163

def bead_pattern (n : Nat) : String :=
  let pattern := ["red", "red", "red", "red", "yellow", "yellow", "yellow", "green", "green", "white"]
  pattern[(n - 1) % 10]

theorem bead_2000_is_white : bead_pattern 2000 = "white" :=
  by
    -- proof
    sorry

end bead_2000_is_white_l317_317163


namespace fraction_expression_l317_317844

theorem fraction_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + 3 * Real.sin α) = 3 / 8 := by
  sorry

end fraction_expression_l317_317844


namespace circle_area_l317_317306

-- Given the diameter of the circle
def diameter : ℝ := 10
-- Define the radius as half of the diameter
def radius : ℝ := diameter / 2

theorem circle_area :
  let area := π * radius^2
  area = 25 * π := by
  -- This is where the proof would go
  sorry

end circle_area_l317_317306


namespace gina_snake_mice_eaten_in_decade_l317_317840

-- Define the constants and conditions
def weeks_per_year : ℕ := 52
def years_per_decade : ℕ := 10
def weeks_per_decade : ℕ := years_per_decade * weeks_per_year
def mouse_eating_period : ℕ := 4

-- The problem to prove
theorem gina_snake_mice_eaten_in_decade : (weeks_per_decade / mouse_eating_period) = 130 := 
by
  -- The proof would typically go here, but we skip it
  sorry

end gina_snake_mice_eaten_in_decade_l317_317840


namespace rahul_work_days_l317_317586

theorem rahul_work_days
  (R : ℕ)
  (Rajesh_days : ℕ := 2)
  (total_payment : ℕ := 170)
  (rahul_share : ℕ := 68)
  (combined_work_rate : ℚ := 1) :
  (∃ R : ℕ, (1 / (R : ℚ) + 1 / (Rajesh_days : ℚ) = combined_work_rate) ∧ (68 / (total_payment - rahul_share) = 2 / R) ∧ R = 3) :=
sorry

end rahul_work_days_l317_317586


namespace volume_correctness_l317_317055

noncomputable def volume_of_regular_triangular_pyramid (d : ℝ) : ℝ :=
  1/3 * d^2 * d * Real.sqrt 2

theorem volume_correctness (d : ℝ) : 
  volume_of_regular_triangular_pyramid d = 1/3 * d^3 * Real.sqrt 2 :=
by
  sorry

end volume_correctness_l317_317055


namespace triangle_perimeter_of_ellipse_l317_317871

theorem triangle_perimeter_of_ellipse
  (a : ℝ) (h_a : a > 5)
  (x y : ℝ) 
  (elliptic_eq : (x^2)/(a^2) + (y^2)/25 = 1)
  (f1 f2 : ℝ × ℝ)
  (distance_f1f2 : abs (f1.1 - f2.1) = 8)
  (chord_AB_passes_f1 : ∃ A B : ℝ × ℝ, (A ∈ elliptic_eq ∧ B ∈ elliptic_eq) ∧ (A - B) = f1) :
  let b := 5 in
  let c := 4 in
  let perimeter := 4 * sqrt (b^2 + c^2) in
  perimeter = 4 * sqrt 41 := sorry

end triangle_perimeter_of_ellipse_l317_317871


namespace arithmetic_sequence_products_l317_317566

-- Definitions for points and the circle
variables {A B C D B1 C1 D1 : Point}
variables (circle_O : Circle)

-- Conditions
axiom passes_through_A : circle_O.passes_through A
axiom intersects_AB_at_B1 : circle_O.intersects (line_through A B) B1
axiom intersects_AC_at_C1 : circle_O.intersects (line_through A C) C1
axiom intersects_AD_at_D1 : circle_O.intersects (median A B C) D1

-- Median condition
axiom AD_is_median : is_median A D (B, C)

-- Statement to prove
theorem arithmetic_sequence_products :
  AB1 * AB = 2 * AD1 * AD - AC1 * AC := 
sorry

end arithmetic_sequence_products_l317_317566


namespace unique_divisor_d_l317_317283

theorem unique_divisor_d
  (p : Nat) (hp : Nat.prime p) (hp_gt_two: p > 2)
  (n : Int) :
  ∃! d : Nat, d ∣ (p * n^2) ∧ ∃ m : Nat, m^2 = n^2 + d :=
  sorry

end unique_divisor_d_l317_317283


namespace LinesNotParallel_l317_317059

variable {m n : Line}
variable {α β : Plane}

-- Definitions from conditions in the problem
def line_in_plane (l : Line) (p : Plane) : Prop := ∀ (x : Point), x ∈ l → x ∈ p
def lines_not_parallel (l₁ l₂ : Line) : Prop := ¬parallel l₁ l₂
def lines_coplanar (l₁ l₂ : Line) (p : Plane) : Prop := ∀ x ∈ l₁, x ∈ p ∧ ∀ x ∈ l₂, x ∈ p

-- The proof problem statement
theorem LinesNotParallel 
  (h₁ : line_in_plane m α)
  (h₂ : lines_not_parallel n α)
  (h₃ : lines_coplanar m n β) : 
  lines_not_parallel m n := 
sorry

end LinesNotParallel_l317_317059


namespace square_area_l317_317989

/-- Given points (1,1) and (1,5) are adjacent points on a square.
    The area of the square formed by these points is 16. -/
theorem square_area (A B : ℝ × ℝ) 
  (hA : A = (1, 1)) 
  (hB : B = (1, 5))
  (adjacent : dist A B = 4) :
  ∃ s : ℝ, s ^ 2 = 16 :=
begin
  use 4,
  norm_num,
end

end square_area_l317_317989


namespace neg_p_suff_not_nec_neg_q_l317_317480

theorem neg_p_suff_not_nec_neg_q (x : ℝ) (p : Prop) (q : Prop) :
  (|x + 1| > 2 ↔ p) → (x > 2 ↔ q) → (¬ p → ¬ q) ∧ (¬ q → ¬ p) → False :=
by
  intros h1 h2 h3
  have h4 : ∀ x, ¬(|x + 1| > 2) → x ≤ 2 := by sorry
  have h5 : ∀ x, x ≤ 2 → ¬(|x + 1| > 2) := by sorry
  subst_vars
  apply h4
  sorry

end neg_p_suff_not_nec_neg_q_l317_317480


namespace value_of_leftover_coins_l317_317749

def quarters_per_roll : ℕ := 30
def dimes_per_roll : ℕ := 40

def ana_quarters : ℕ := 95
def ana_dimes : ℕ := 183

def ben_quarters : ℕ := 104
def ben_dimes : ℕ := 219

def leftover_quarters : ℕ := (ana_quarters + ben_quarters) % quarters_per_roll
def leftover_dimes : ℕ := (ana_dimes + ben_dimes) % dimes_per_roll

def dollar_value (quarters dimes : ℕ) : ℝ := quarters * 0.25 + dimes * 0.10

theorem value_of_leftover_coins : 
  dollar_value leftover_quarters leftover_dimes = 6.95 := 
  sorry

end value_of_leftover_coins_l317_317749


namespace family_gathering_handshakes_l317_317379

theorem family_gathering_handshakes :
  -- conditions
  let twins := 12 * 2 in
  let triplets := 4 * 3 in
  let twin_handshakes := (twins * (twins - 2)) / 2 in
  let triplet_handshakes := (triplets * (triplets - 3)) / 2 in 
  let cross_handshakes := twins * (3 / 4 * triplets) + triplets * (3 / 4 * twins) in
  -- question == correct answer
  twin_handshakes + triplet_handshakes + cross_handshakes = 750 :=
begin
  sorry
end

end family_gathering_handshakes_l317_317379


namespace statisticalProperties_l317_317853

def sampleStandardDeviation (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  let variance := (data.map (λ x, (x - mean) ^ 2)).sum / data.length
  Real.sqrt variance

def sampleRange (data : List ℝ) : ℝ := data.maximum - data.minimum

theorem statisticalProperties (n : ℕ) (x y : Fin n → ℝ) (c : ℝ) (h : ∀ i : Fin n, y i = x i + c) (hn : n > 0) (hc : c ≠ 0) :
  sampleStandardDeviation (List.ofFn y) = sampleStandardDeviation (List.ofFn x) ∧ sampleRange (List.ofFn y) = sampleRange (List.ofFn x) := 
by
  sorry

end statisticalProperties_l317_317853


namespace triangle_problem_proof_l317_317535

noncomputable def angle_A_and_area_triangle (a b: ℝ) (cos_diff: ℝ) (sin_sum: ℝ) 
  (cos_A: ℝ) (sin_A: ℝ) (sin_C: ℝ) : Prop :=
  ((cos_diff - 2 * sin_sum = -1 / 2) ∧ (cos_A = 1 / 2) →
    ((a = 5) ∧ (b = 4)) →
    (A = (π / 3)) ∧ (S_triangle = 2 * sqrt 3 + sqrt 39))

theorem triangle_problem_proof : 
  angle_A_and_area_triangle 5 4 
    (cos (B - C)) (sin B + sin C) 
    (cos (π / 3)) (sin (π / 3)) (sin 60) := 
  sorry

end triangle_problem_proof_l317_317535


namespace radius_sum_l317_317654

-- Define a right triangle with vertices A, B, and C and right angle at A
variable {A B C : Type} [triangle A B C]

-- Define the incircle radius and excircle radii
variable {r r_A r_B r_C : ℝ}

-- Assume the given conditions in the problem
variable (hr : ∀ t : ℝ, incircle_radius A B C = r)
variable (hra : ∀ t : ℝ, excircle_radius A B C = r_A)
variable (hrb : ∀ t : ℝ, excircle_radius B A C = r_B)
variable (hrc : ∀ t : ℝ, excircle_radius C A B = r_C)

-- Statement to prove
theorem radius_sum (h_triangle : right_triangle A B C)
 (h_inc : incircle_radius A B C = r)
 (h_exa : excircle_radius A B C = r_A)
 (h_exb : excircle_radius B A C = r_B)
 (h_exc : excircle_radius C A B = r_C)
 : r_A = r + r_B + r_C :=
by
  sorry

end radius_sum_l317_317654


namespace statisticalProperties_l317_317854

def sampleStandardDeviation (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  let variance := (data.map (λ x, (x - mean) ^ 2)).sum / data.length
  Real.sqrt variance

def sampleRange (data : List ℝ) : ℝ := data.maximum - data.minimum

theorem statisticalProperties (n : ℕ) (x y : Fin n → ℝ) (c : ℝ) (h : ∀ i : Fin n, y i = x i + c) (hn : n > 0) (hc : c ≠ 0) :
  sampleStandardDeviation (List.ofFn y) = sampleStandardDeviation (List.ofFn x) ∧ sampleRange (List.ofFn y) = sampleRange (List.ofFn x) := 
by
  sorry

end statisticalProperties_l317_317854


namespace correct_derivative_operation_l317_317321

theorem correct_derivative_operation :
  (differentiable (λ x, log 2 x) → deriv (λ x, log 2 x) = (λ x, 1 / (x * log base.e 2))) :=
by
  sorry

end correct_derivative_operation_l317_317321


namespace find_value_l317_317069

noncomputable def a : ℝ := 5 - 2 * Real.sqrt 6

theorem find_value :
  a^2 - 10 * a + 1 = 0 :=
by
  -- Since we are only required to write the statement, add sorry to skip the proof.
  sorry

end find_value_l317_317069


namespace sin_lt_x_when_pos_l317_317583

theorem sin_lt_x_when_pos (x : ℝ) (h : x > 0) : sin x < x :=
sorry

end sin_lt_x_when_pos_l317_317583


namespace difference_of_squares_153_147_l317_317799

theorem difference_of_squares_153_147 : (153^2 - 147^2) = 1800 := by
  sorry

end difference_of_squares_153_147_l317_317799


namespace determine_house_numbers_l317_317495

-- Definitions based on the conditions given
def even_numbered_side (n : ℕ) : Prop :=
  n % 2 = 0

def sum_balanced (n : ℕ) (house_numbers : List ℕ) : Prop :=
  let left_sum := house_numbers.take n |>.sum
  let right_sum := house_numbers.drop (n + 1) |>.sum
  left_sum = right_sum

def house_constraints (n : ℕ) : Prop :=
  50 < n ∧ n < 500

-- Main theorem statement
theorem determine_house_numbers : 
  ∃ (n : ℕ) (house_numbers : List ℕ), 
    even_numbered_side n ∧ 
    house_constraints n ∧ 
    sum_balanced n house_numbers :=
  sorry

end determine_house_numbers_l317_317495


namespace problem_1_problem_2_problem_3_l317_317116

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

def condition_1 (A ω : ℝ) (φ : ℝ) : Prop := 
  A = 2 ∧ ω = 2 ∧ φ = π / 6 ∧ 0 < A ∧ 0 < φ ∧ φ < π / 2

theorem problem_1 (A ω : ℝ) (φ : ℝ) (h : condition_1 A ω φ) : 
  f x = 2 * sin (2 * x + φ) := by
  sorry

theorem problem_2 (k : ℤ) (x : ℝ) 
  (h1 : k * π + π / 6 ≤ x) 
  (h2 : x ≤ k * π + 2 * π / 3) : 
  monotone_decreasing_on (f : ℝ → ℝ) (Icc (k * π + π / 6) (k * π + 2 * π / 3)) := by
  sorry

theorem problem_3 (x : ℝ) 
  (h1 : π / 12 ≤ x) 
  (h2 : x ≤ π / 2) : 
  range (λ x, f x) (Icc (π / 12) (π / 2)) = Icc (-1) 2 := by
  sorry

end problem_1_problem_2_problem_3_l317_317116


namespace cannot_factorize_using_difference_of_squares_l317_317520

theorem cannot_factorize_using_difference_of_squares (x y : ℝ) :
  ¬ ∃ a b : ℝ, -x^2 - y^2 = a^2 - b^2 :=
sorry

end cannot_factorize_using_difference_of_squares_l317_317520


namespace g_triple_composition_l317_317028

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n + 3

theorem g_triple_composition : g (g (g 3)) = 49 :=
by
  sorry

end g_triple_composition_l317_317028


namespace problem_continuous_function_l317_317224

variable {ℝ : Type} [LinearOrder ℝ] [TopologicalSpace ℝ] [OrderTopology ℝ]

theorem problem_continuous_function (f : ℝ → ℝ) 
  (hf_cont : Continuous f) 
  (hf_deriv_neg : ∀ x : ℝ, (x - 1) * (deriv f x) < 0) :
  f 0 + f 2 < 2 * f 1 := 
sorry

end problem_continuous_function_l317_317224


namespace base3_addition_l317_317763

theorem base3_addition :
  (2 + 1 * 3 + 2 * 9 + 1 * 27 + 2 * 81) + (1 + 1 * 3 + 2 * 9 + 2 * 27) + (2 * 9 + 1 * 27 + 0 * 81 + 2 * 243) + (1 + 1 * 3 + 1 * 9 + 2 * 27 + 2 * 81) = 
  2 + 1 * 3 + 1 * 9 + 2 * 27 + 2 * 81 + 1 * 243 + 1 * 729 := sorry

end base3_addition_l317_317763


namespace overall_gain_percentage_l317_317769

variable (n : ℕ)
variable (C_A : ℝ)
variable (C_B : ℝ := 2 * C_A)
variable (S_A : ℝ := (2 / 3) * C_A)
variable (S_B : ℝ := 1.2 * C_B)

-- Condition: Selling type A gadgets at 2/3 of its price results in a 10% loss =>
-- we need S_A to be equal to 90% of its cost price.
axiom selling_price_A : S_A = 0.9 * C_A

-- Prove that the overall gain or loss percentage is 10%
theorem overall_gain_percentage (h : C_B = 2 * C_A) (h1 : S_A = (2 / 3) * C_A) (h2 : S_B = 1.2 * C_B) (n_pos : 0 < n) :
  let total_cost_price := (n * C_A + n * C_B),
      total_selling_price := (n * S_A + n * S_B),
      gain_or_loss_percentage := ((total_selling_price - total_cost_price) / total_cost_price) * 100
  in gain_or_loss_percentage = 10 := 
by {
  sorry
}

end overall_gain_percentage_l317_317769


namespace eighty_five_squared_l317_317008

theorem eighty_five_squared : 85^2 = 7225 := by
  sorry

end eighty_five_squared_l317_317008


namespace num_of_pipes_needed_l317_317760

-- Defining volumes of cylindrical pipes given radius and height
def volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- The main theorem we want to prove
theorem num_of_pipes_needed (h : ℝ) :
  let V_12 := volume 6 h in
  let V_3 := volume 1.5 h in
  (V_12 / V_3) = 16 := by
  sorry

end num_of_pipes_needed_l317_317760


namespace product_even_for_any_permutation_l317_317546

theorem product_even_for_any_permutation (A : Finset ℕ) (h : A.card = 5) : 
  ∀ σ : Equiv.Perm (Fin 5), 
  even ((A.toList.nthLe (σ 0) (by simp)).val - A.toList.nthLe 0 (by simp)) *
       ((A.toList.nthLe (σ 1) (by simp)).val - A.toList.nthLe 1 (by simp)) *
       ((A.toList.nthLe (σ 2) (by simp)).val - A.toList.nthLe 2 (by simp)) *
       ((A.toList.nthLe (σ 3) (by simp)).val - A.toList.nthLe 3 (by simp)) *
       ((A.toList.nthLe (σ 4) (by simp)).val - A.toList.nthLe 4 (by simp)) := 
by
  sorry

end product_even_for_any_permutation_l317_317546


namespace find_highway_miles_l317_317347

def highway_miles_used (distance_miles : ℝ) : ℝ := distance_miles / 36
def city_miles_used (distance_miles : ℝ) : ℝ := distance_miles / 20

def total_gasoline_first_scenario : ℝ := highway_miles_used 4 + city_miles_used 4

theorem find_highway_miles :
  let x := (14 / 45) / 1.4000000000000001 in
  let highway_distance := x * 36 in
  highway_distance ≈ 5.714285714285714 := sorry

end find_highway_miles_l317_317347


namespace tens_digit_23_pow_2045_l317_317034

theorem tens_digit_23_pow_2045 : 
  let a: ℤ := 23
  let m: ℤ := 100
  let k: ℤ := 2045
  (a ^ 20) % m = 1 → k % 20 = 5 → (a ^ 5) % m = 43 → 
  (((a ^ k) % m) / 10) % 10 = 4 :=
by
  intros h1 h2 h3
  sorry

end tens_digit_23_pow_2045_l317_317034


namespace product_probability_probability_one_l317_317254

def S : Set Int := {13, 57}

theorem product_probability (a b : Int) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : a ≠ b) : 
  (a * b > 15) := 
by 
  sorry

theorem probability_one : 
  (∃ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a * b > 15) ∧ 
  (∀ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b → a * b > 15) :=
by 
  sorry

end product_probability_probability_one_l317_317254


namespace volume_of_water_flow_per_min_l317_317697

-- Definitions as per conditions:
def river_depth : ℝ := 2
def river_width : ℝ := 45
def flow_rate_kmph : ℝ := 5

-- Translate the flow rate from kmph to m/min:
def flow_rate_m_per_min : ℝ := (flow_rate_kmph * 1000) / 60

-- Calculate the cross-sectional area of the river:
def cross_sectional_area : ℝ := river_depth * river_width

-- Question/statement to prove: the volume of water flowing per minute
theorem volume_of_water_flow_per_min :
  (cross_sectional_area * flow_rate_m_per_min) ≈ 7499.7 :=
by sorry

end volume_of_water_flow_per_min_l317_317697


namespace kaeli_problems_per_day_l317_317569

-- Definitions based on conditions
def problems_solved_per_day_marie_pascale : ℕ := 4
def total_problems_marie_pascale : ℕ := 72
def total_problems_kaeli : ℕ := 126

-- Number of days both took should be the same
def number_of_days : ℕ := total_problems_marie_pascale / problems_solved_per_day_marie_pascale

-- Kaeli solves 54 more problems than Marie-Pascale
def extra_problems_kaeli : ℕ := 54

-- Definition that Kaeli's total problems solved is that of Marie-Pascale plus 54
axiom kaeli_total_problems (h : total_problems_marie_pascale + extra_problems_kaeli = total_problems_kaeli) : True

-- Now to find x, the problems solved per day by Kaeli
def x : ℕ := total_problems_kaeli / number_of_days

-- Prove that x = 7
theorem kaeli_problems_per_day (h : total_problems_marie_pascale + extra_problems_kaeli = total_problems_kaeli) : x = 7 := by
  sorry

end kaeli_problems_per_day_l317_317569


namespace number_of_dogs_l317_317809

-- Define the number of legs humans have
def human_legs : ℕ := 2

-- Define the total number of legs/paws in the pool
def total_legs_paws : ℕ := 24

-- Define the number of paws per dog
def paws_per_dog : ℕ := 4

-- Prove that the number of dogs is 5
theorem number_of_dogs : ∃ (dogs : ℕ), (2 * human_legs) + (dogs * paws_per_dog) = total_legs_paws ∧ dogs = 5 :=
by
  use 5
  split
  sorry

end number_of_dogs_l317_317809


namespace continuous_curve_chord_length_l317_317977

theorem continuous_curve_chord_length (A B: ℝ × ℝ) (hAB : dist A B = 1) (a : ℝ) :
  (∃ n : ℕ, n > 0 ∧ a = 1 / n → ∃ K : ℝ → ℝ × ℝ, continuous K ∧ K 0 = A ∧ K 1 = B 
    ∧ ∃ t₁ t₂ : ℝ, t₁ < t₂ ∧ dist (K t₁) (K t₂) = a) ∧
  ((∀ n : ℕ, a ≠ 1 / n) → ∃ K : ℝ → ℝ × ℝ, continuous K ∧ K 0 = A ∧ K 1 = B 
    ∧ ∀ t₁ t₂ : ℝ, t₁ < t₂ → dist (K t₁) (K t₂) ≠ a) :=
by
  sorry

end continuous_curve_chord_length_l317_317977


namespace wrappers_found_at_park_l317_317021

noncomputable def bottle_caps_initial := 6
noncomputable def bottle_caps_found := 22
noncomputable def bottle_caps_now := 28
noncomputable def wrappers_now := 63

theorem wrappers_found_at_park 
  (bottle_caps_initial = 6)
  (bottle_caps_found = 22)
  (bottle_caps_now = 28)
  (wrappers_now = 63) :
  (63 - wrappers_now) = 22 := 
by sorry

end wrappers_found_at_park_l317_317021


namespace illumination_overlap_exists_l317_317503

structure City :=
  (radius : ℝ)
  (radius_property : radius = 10000)

structure StreetLight :=
  (initial_radius : ℝ)
  (initial_radius_property : initial_radius = 200)
  (diminish_rate : ℝ)
  (diminish_rate_property : diminish_rate = 10)

def batteries_per_day := 18000

theorem illumination_overlap_exists 
  (c : City)
  (s : StreetLight)
  (b_per_day : ℕ)
  (b_per_day_property : b_per_day = batteries_per_day) : 
  ∃ p : ℝ × ℝ, 
    ∃ t : ℝ, 
    ∃ s₁ s₂ : StreetLight, s₁ ≠ s₂ ∧ illuminates s₁ p t ∧ illuminates s₂ p t := 
by 
  sorry

end illumination_overlap_exists_l317_317503


namespace determine_phi_l317_317896

theorem determine_phi
  (A ω : ℝ) (φ : ℝ) (x : ℝ)
  (hA : 0 < A)
  (hω : 0 < ω)
  (hφ : abs φ < Real.pi / 2)
  (h_symm : ∃ f : ℝ → ℝ, f (-Real.pi / 4) = A ∨ f (-Real.pi / 4) = -A)
  (h_zero : ∃ x₀ : ℝ, A * Real.sin (ω * x₀ + φ) = 0 ∧ abs (x₀ + Real.pi / 4) = Real.pi / 2) :
  φ = -Real.pi / 4 :=
sorry

end determine_phi_l317_317896


namespace find_sum_of_cubes_l317_317551

noncomputable def roots_of_polynomial := 
  ∃ a b c : ℝ, 
    (6 * a^3 + 500 * a + 1001 = 0) ∧ 
    (6 * b^3 + 500 * b + 1001 = 0) ∧ 
    (6 * c^3 + 500 * c + 1001 = 0)

theorem find_sum_of_cubes (a b c : ℝ) 
  (h : roots_of_polynomial) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 500.5 := 
sorry

end find_sum_of_cubes_l317_317551


namespace angle_between_slant_and_face_is_30_l317_317278

-- Defining the regular quadrilateral pyramid with the specified angles
structure RegularQuadrilateralPyramid where
  a : ℝ -- the side length of the square base
  h : ℝ -- the height from the base to the apex
  lateral_face_angle : ℝ := 45 -- the angle between the lateral face and the base

-- Given conditions
variable (P : RegularQuadrilateralPyramid) [Fact (P.lateral_face_angle = 45)]

-- Define the angle \(\phi\) between the slant height and the plane of an adjacent face
def angle_between_slant_height_and_adjacent_face := 30

theorem angle_between_slant_and_face_is_30 : 
  angle_between_slant_height_and_adjacent_face P = 30 := 
sorry

end angle_between_slant_and_face_is_30_l317_317278


namespace binomial_expansion_l317_317888

theorem binomial_expansion :
  let a_0 : ℝ := -1
  let a_1 : ℝ := 0
  let a_2 : ℝ := -24
  let a_3 : ℝ := 8 * real.sqrt 5
  (a_0 + a_2)^2 - (a_1 + a_3)^2 = -64 :=
by
  sorry

end binomial_expansion_l317_317888


namespace series_convergence_l317_317256

theorem series_convergence (a : ℕ → ℝ) (h : Summable a) : Summable (λ n, a n / (n + 1)) := sorry

end series_convergence_l317_317256


namespace total_pink_roses_l317_317775

theorem total_pink_roses (rows : ℕ) (roses_per_row : ℕ) (half : ℚ) (three_fifths : ℚ) :
  rows = 10 →
  roses_per_row = 20 →
  half = 1/2 →
  three_fifths = 3/5 →
  (rows * (roses_per_row - roses_per_row * half - (roses_per_row * (1 - half) * three_fifths))) = 40 :=
begin
  sorry
end

end total_pink_roses_l317_317775


namespace smallest_perfect_square_divisible_by_5_and_6_l317_317683

-- 1. Define the gcd and lcm functionality
def lcm (a b : ℕ) : ℕ :=
  (a * b) / Nat.gcd a b

-- 2. Define the condition that a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- 3. State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 : ∃ n : ℕ, is_perfect_square n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (is_perfect_square m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m :=
  sorry

end smallest_perfect_square_divisible_by_5_and_6_l317_317683


namespace eighty_five_squared_l317_317012

theorem eighty_five_squared :
  (85:ℕ)^2 = 7225 := 
by
  let a := 80
  let b := 5
  have h1 : (a + b) = 85 := rfl
  have h2 : (a^2 + 2 * a * b + b^2) = 7225 := by norm_num
  rw [←h1, ←h1]
  rw [ sq (a + b)]
  rw [ mul_add, add_mul, add_mul, mul_comm 2 b]
  rw [←mul_assoc, ←mul_assoc, add_assoc, add_assoc, nat.add_right_comm ]
  exact h2

end eighty_five_squared_l317_317012


namespace rearrange_numbers_diff_3_or_5_l317_317201

noncomputable def is_valid_permutation (n : ℕ) (σ : list ℕ) : Prop :=
  (σ.nodup ∧ ∀ i < σ.length - 1, |σ.nth_le i sorry - σ.nth_le (i + 1) sorry| = 3 ∨ |σ.nth_le i sorry - σ.nth_le (i + 1) sorry| = 5)

theorem rearrange_numbers_diff_3_or_5 (n : ℕ) :
  (n = 25 ∨ n = 1000) → ∃ σ : list ℕ, (σ = (list.range n).map (+1)) ∧ is_valid_permutation n σ :=
by
  sorry

end rearrange_numbers_diff_3_or_5_l317_317201


namespace solve_inequality_l317_317610

theorem solve_inequality (x : ℝ) : 2 * x ^ 2 - 7 * x - 30 < 0 ↔ - (5 / 2) < x ∧ x < 6 := 
sorry

end solve_inequality_l317_317610


namespace arrangement_of_digits_11250_l317_317179

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then
    1
  else
    n * factorial (n - 1)

def number_of_arrangements (digits : List ℕ) : ℕ :=
  let number_ends_in_0 := factorial 4 / factorial 2
  let number_ends_in_5 := 3 * (factorial 3 / factorial 2)
  number_ends_in_0 + number_ends_in_5

theorem arrangement_of_digits_11250 :
  number_of_arrangements [1, 1, 2, 5, 0] = 21 :=
by
  sorry

end arrangement_of_digits_11250_l317_317179


namespace collinear_vectors_value_m_l317_317460

theorem collinear_vectors_value_m (m : ℝ) : 
  (∃ k : ℝ, (2*m = k * (m - 1)) ∧ (3 = k)) → m = 3 :=
by
  sorry

end collinear_vectors_value_m_l317_317460


namespace compute_diff_squares_l317_317792

theorem compute_diff_squares (a b : ℤ) (ha : a = 153) (hb : b = 147) :
  a ^ 2 - b ^ 2 = 1800 :=
by
  rw [ha, hb]
  sorry

end compute_diff_squares_l317_317792


namespace def_integral_abs_x2_minus_2x_l317_317397

theorem def_integral_abs_x2_minus_2x :
  (∫ x in -2..2, |x^2 - 2*x|) = 8 :=
by
  -- Conditions specified in the text
  have h1 : ∀ x, x ∈ set.Icc (-2:ℝ) 0 → (x^2 - 2*x) ≥ 0 := by
    intros x hx
    linarith [hx.1, hx.2, pow_two_nonneg x]
  have h2 : ∀ x, x ∈ set.Ioc (0:ℝ) 2 → (x^2 - 2*x) < 0 := by
    intros x hx
    linarith [hx.1, hx.2, pow_two_nonneg x]
  sorry

end def_integral_abs_x2_minus_2x_l317_317397


namespace locus_not_always_parabola_l317_317950

noncomputable theory

-- Condition from the problem
def is_locus_parabola (F : point) (l : line) : Prop :=
  ∀ P : point, dist P F = dist P l → is_parabola P F l

-- Theorem to be proved
theorem locus_not_always_parabola (F : point) (l : line) :
  ¬ is_locus_parabola F l :=
sorry

end locus_not_always_parabola_l317_317950


namespace students_per_class_l317_317912

theorem students_per_class :
  let buns_per_package := 8
  let packages := 30
  let buns_per_student := 2
  let classes := 4
  (packages * buns_per_package) / (buns_per_student * classes) = 30 :=
by
  sorry

end students_per_class_l317_317912


namespace total_corresponding_angles_l317_317063

-- Define the conditions given in the problem.
def lines_intersect_pairwise (l : Fin 4 → Set (Point ℝ)) : Prop :=
  ∀ i j, i ≠ j → (l i ∩ l j).Nonempty

def no_three_lines_intersect_at_same_point (l : Fin 4 → Set (Point ℝ)) : Prop :=
  ∀ i j k, i ≠ j → j ≠ k → k ≠ i → (l i ∩ l j ∩ l k).Empty

theorem total_corresponding_angles (l : Fin 4 → Set (Point ℝ)) :
  lines_intersect_pairwise l → no_three_lines_intersect_at_same_point l → 
  total_corresponding_angles l = 48 :=
  by
  intros
  sorry

end total_corresponding_angles_l317_317063


namespace triangle_congruence_by_two_angles_and_included_side_l317_317998

theorem triangle_congruence_by_two_angles_and_included_side
  (A B C A₁ B₁ C₁ : Type)
  [DecidableEq A] [DecidableEq B] [DecidableEq C]
  [DecidableEq A₁] [DecidableEq B₁] [DecidableEq C₁]
  (angle_A : A) (angle_B : B) (side_BC : C)
  (angle_A₁ : A₁) (angle_B₁ : B₁) (side_B₁C₁ : C₁) :
  (angle_A = angle_A₁) →
  (angle_B = angle_B₁) →
  (side_BC = side_B₁C₁) →
  (triangle ABC ≃ triangle A₁B₁C₁) :=
by
  intro h1 h2 h3
  sorry

end triangle_congruence_by_two_angles_and_included_side_l317_317998


namespace smallest_x_inequality_l317_317418

theorem smallest_x_inequality (a : ℝ) (h : a ≥ 0) : 
  ∃ x : ℝ, (∀ a : ℝ, a ≥ 0 → a ≥ 14 * real.sqrt(a) - x) ∧ (x = 49) :=
sorry

end smallest_x_inequality_l317_317418


namespace simplify_expression_l317_317602

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem simplify_expression :
  cube_root (8 + 27) * cube_root (8 + cube_root 27) = cube_root 385 :=
by
  sorry

end simplify_expression_l317_317602


namespace max_value_of_g_l317_317544

noncomputable def g (x : ℝ) : ℝ := real.sqrt (x * (60 - x)) + real.sqrt (x * (4 - x))

theorem max_value_of_g : (∃ x1 N, (∀ x, 0 ≤ x ∧ x ≤ 4 → g x ≤ N) ∧ g x1 = N ∧ N = 16 ∧ x1 = 15 / 4) :=
by
  sorry

end max_value_of_g_l317_317544


namespace digits_right_of_decimal_l317_317132

theorem digits_right_of_decimal (h : \(\frac{5^8}{10^6 \cdot 16}\) = 0.025) : (nat.digits 10 (1 / 0.025)).length = 3 :=
sorry

end digits_right_of_decimal_l317_317132


namespace line_through_bisecting_point_l317_317891

theorem line_through_bisecting_point (x y : ℝ) 
  (ellipse : x^2 / 2 + y^2 = 1)
  (P : ℝ × ℝ := (1/2, 1/2)) 
  (line : ∃ a b c : ℝ, line_eq := 2 * x + 4 * y - 3 = 0)
: ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧ c ≠ 0 :=
  sorry

end line_through_bisecting_point_l317_317891


namespace solve_probability_l317_317357

/-- 
Given two points chosen independently and uniformly at random from the interval [0, 1], 
let x be the red point and y be the blue point. 

We want to show that the probability that x < y < 3 * x is 5 / 6.
-/
noncomputable def probability_blue_point_condition : ℝ := 
  let region1_area := (1/2) * (1/3) * 1 in  -- Area of triangle
  let region2_area := (2/3) * 1 in          -- Area of rectangle
  region1_area + region2_area               -- Total area

theorem solve_probability : probability_blue_point_condition = 5 / 6 := 
by
  sorry

end solve_probability_l317_317357


namespace neg_p_is_correct_l317_317638

def is_positive_integer (x : ℕ) : Prop := x > 0

def proposition_p (x : ℕ) : Prop := (1 / 2 : ℝ) ^ x ≤ 1 / 2

def negation_of_p : Prop := ∃ x : ℕ, is_positive_integer x ∧ ¬ proposition_p x

theorem neg_p_is_correct : negation_of_p :=
sorry

end neg_p_is_correct_l317_317638


namespace integer_solutions_of_equation_l317_317630

def satisfies_equation (x y : ℤ) : Prop :=
  x * y - 2 * x - 2 * y + 7 = 0

theorem integer_solutions_of_equation :
  { (x, y) : ℤ × ℤ | satisfies_equation x y } = { (5, 1), (-1, 3), (3, -1), (1, 5) } :=
by sorry

end integer_solutions_of_equation_l317_317630


namespace student_marks_l317_317365

def marks_obtained (total_marks : ℕ) (passing_percentage : ℝ) (failed_by : ℕ) : ℕ :=
  let passing_marks := (passing_percentage / 100) * total_marks
  (passing_marks : ℕ) - failed_by

theorem student_marks :
  marks_obtained 400 45 30 = 150 := by
  sorry

end student_marks_l317_317365


namespace range_of_a_l317_317091

noncomputable def A : Set ℤ := { x | ∃ (n : ℤ), x = n ∧ sqrt (n - 1) ≤ 2 }
noncomputable def B (a : ℝ) : Set ℤ := { x | ∃ (n : ℤ), x = n ∧ n ≤ a }

theorem range_of_a (a : ℝ) : (A ∩ B a = A) ↔ (a ≥ 5) :=
by
  sorry

end range_of_a_l317_317091


namespace div_by_7_l317_317815

theorem div_by_7 (n : ℕ) : (3 ^ (12 * n + 1) + 2 ^ (6 * n + 2)) % 7 = 0 := by
  sorry

end div_by_7_l317_317815


namespace g_f_neg7_eq_neg2_l317_317552

def f (x : ℝ) : ℝ :=
if x >= 0 then real.logb 2 (x+1) else g (x)

def g (x : ℝ) : ℝ :=
- real.logb 2 (-x + 1)

theorem g_f_neg7_eq_neg2 :
  g (f (-7)) = -2 :=
by sorry

end g_f_neg7_eq_neg2_l317_317552


namespace average_consecutive_and_spaced_integers_l317_317830

theorem average_consecutive_and_spaced_integers (a : ℕ) (h : 0 < a) :
  let b := (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5 in
  let c := b + 10 in
  let d := b + 20 in
  (b + c + d) / 3 = a + 12 :=
by {
  -- Problem setup and transformations based on conditions and known results
  let b := (a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) / 5,
  -- Using the definition of new sequence values
  let c := b + 10,
  let d := b + 20,
  -- Placeholder for actual proof
  sorry
}

end average_consecutive_and_spaced_integers_l317_317830


namespace tangent_line_intersections_l317_317848

open Set

structure Circle (ℝ : Type*) [MetricSpace ℝ] :=
(center : ℝ)
(radius : ℝ)

def on_circle (ℝ : Type*) [MetricSpace ℝ] (C : Circle ℝ) (P : ℝ) : Prop :=
dist P C.center = C.radius

def tangent_line (ℝ : Type*) [MetricSpace ℝ] (C : Circle ℝ) (B : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
on_circle ℝ C B ∧ ∃ T, dist B T = dist B C.center ∧ ∀ P, l T P ↔ dist T P = dist T B

def line_perpendicular_to (ℝ : Type*) [MetricSpace ℝ] (A B : ℝ) (O : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
∃ M, l O M ∧ dot_product (B - A) (O - M) = 0

def intersection_point (ℝ : Type*) [MetricSpace ℝ] (l1 l2 : ℝ → ℝ → Prop) (T : ℝ) : Prop :=
l1 L T ∧ l2 L T

noncomputable def polar_line (ℝ : Type*) [MetricSpace ℝ] (C : Circle ℝ) (A : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
∃ x y ∈ Affine.orthogonalProjectionBeginC.bounds (O, C.center - A, l)

theorem tangent_line_intersections (ℝ : Type*) [MetricSpace ℝ] 
  (C : Circle ℝ) (O A : ℝ)
  (hA : ¬ on_circle ℝ C A) 
  (B : ℝ) (hB : on_circle ℝ C B) 
  (l_perpendicular : ∃ l, line_perpendicular_to ℝ A B O l)
  (T : ℝ) 
  (hT1 : tangent_line ℝ C B (λ x y, ∃ P, P = T))
  (hT2 : intersection_point ℝ (λ x y, ∃ P, P = T) (λ x y, ∃ P, P = O)) : 
  polar_line ℝ C A (λ x y, ∃ P, P = T) := 
begin
  sorry,
end

end tangent_line_intersections_l317_317848


namespace select_positive_numbers_l317_317516

open Matrix

theorem select_positive_numbers (N : ℕ) (A : Matrix (Fin N) (Fin N) ℝ) 
  (h_row_sum : ∀ i, (∑ j, A i j) = 1) 
  (h_col_sum : ∀ j, (∑ i, A i j) = 1) : 
  ∃ (indices : Fin N → Fin N), 
  (∀ i, A i (indices i) > 0) ∧ Function.Injective indices := 
sorry

end select_positive_numbers_l317_317516


namespace external_circle_radius_l317_317662

theorem external_circle_radius 
  (A B C : Type) [MetricSpace A B C] [Triangle A B C]
  (right_angle_C : ∠ A B C = π / 2) 
  (angle_A_eq_45 : ∠ A = π / 4)
  (AC_eq_12 : distance A C = 12) :
  let AB := distance A B in
  radius_external_circle_tangent_to_AB_extending_beyond_C 
    (triangle_45_45_90 (distance A B)) = 6 * sqrt 2 :=
by 
  sorry

end external_circle_radius_l317_317662


namespace monotonic_intervals_max_min_values_log_inequality_l317_317454

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) / (a * x) - Real.log x

theorem monotonic_intervals (a : ℝ) (h : a ≠ 0) :
  (if a < 0 then ∀ x, 0 < x → differentiable_at ℝ (f x a) ∧ deriv (f x a) < 0
   else if a > 0 then (∀ x, 0 < x ∧ x < 1/a → differentiable_at ℝ (f x a) ∧ deriv (f x a) > 0)
   ∧ (∀ x, 1/a < x → differentiable_at ℝ (f x a) ∧ deriv (f x a) < 0)
   else false) :=
sorry

theorem max_min_values (h_log2 : 0.69 < Real.log 2 ∧ Real.log 2 < 0.70):
  let f1 (x : ℝ) := (x - 1) / x - Real.log x in
  (∀ x ∈ set.Icc (1 / 2) 2, f1 x ≤ 0) ∧
  f1 (1 / 2) = -1 + Real.log 2 ∧
  f1 2 > -1 + Real.log 2 :=
sorry

theorem log_inequality (x : ℝ) (hx : 0 < x) : Real.log (Real.exp 2 / x) ≤ (1 + x) / x :=
sorry

end monotonic_intervals_max_min_values_log_inequality_l317_317454


namespace folding_not_possible_l317_317284

theorem  folding_not_possible :
  let triangles := [
    (A1, B1, A2), (B1, A2, B2), (A2, B2, A3), (B2, A3, B3),
    (A3, B3, A4), (B3, A4, B4), (A4, B4, A5), (B4, A5, B5),
    (A5, B5, A6), (B5, A6, B6), (A6, B6, A7), (B6, A7, B7),
    (A7, B7, A8), (B7, A8, B8), (A8, B8, A9), (B8, A9, B9),
    (A9, B9, A10), (B9, A10, B10), (A10, B10, A11), (B10, A11, B11),
    (A11, B11, A12), (B11, A12, B12), (A12, B12, A13), (B12, A13, B13),
    (A13, B13, A14), (B13, A14, B14), (A14, B14, A1), (B14, A1, B1)
  ],
  let edges := [
    (A1, B1), (B1, A2), (A2, B2), (B2, A3), (A3, B3), (B3, A4),
    (A4, B4), (B4, A5), (A5, B5), (B5, A6), (A6, B6), (B6, A7),
    (A7, B7), (B7, A8), (A8, B8), (B8, A9), (A9, B9), (B9, A10),
    (A10, B10), (B10, A11), (A11, B11), (B11, A12), (A12, B12), (B12, A13),
    (A13, B13), (B13, A14), (A14, B14), (B14, A1)
  ]
  in ¬ (∃ (f : list (ℝ × ℝ) → list (ℝ × ℝ)), ∀ (p ∈ triangles), ∀ (q ∈ triangles), f p ≠ f q ∧ Collinear f p q) := sorry

end folding_not_possible_l317_317284


namespace a_10_value_l317_317194

noncomputable def a : ℕ → ℚ
| 0       := 2 -- Since Lean uses zero-based indexing by default
| (n + 1) := a n / (1 + 3 * a n)

theorem a_10_value : a 9 = 2 / 55 :=
by
  sorry

end a_10_value_l317_317194


namespace initial_girls_count_l317_317732

variable (p : ℕ) -- total number of people initially in the group
variable (initial_girls : ℕ) -- number of girls initially

-- Condition 1: Initially, 50% of the group are girls
def initially_fifty_percent_girls (p : ℕ) (initial_girls : ℕ) : Prop := initial_girls = p / 2

-- Condition 2: Three girls leave and three boys arrive
def after_girls_leave_and_boys_arrive (initial_girls : ℕ) : ℕ := initial_girls - 3

-- Condition 3: After the change, 40% of the group are girls
def after_the_change_forty_percent_girls (p : ℕ) (initial_girls : ℕ) : Prop :=
  (after_girls_leave_and_boys_arrive initial_girls) = 2 * (p / 5)

theorem initial_girls_count (p : ℕ) (initial_girls : ℕ) :
  initially_fifty_percent_girls p initial_girls →
  after_the_change_forty_percent_girls p initial_girls →
  initial_girls = 15 := by
  sorry

end initial_girls_count_l317_317732


namespace find_coordinates_of_M_l317_317872

-- Definitions of the points A, B, C
def A : (ℝ × ℝ) := (2, -4)
def B : (ℝ × ℝ) := (-1, 3)
def C : (ℝ × ℝ) := (3, 4)

-- Definitions of vectors CA and CB
def vector_CA : (ℝ × ℝ) := (A.1 - C.1, A.2 - C.2)
def vector_CB : (ℝ × ℝ) := (B.1 - C.1, B.2 - C.2)

-- Definition of the point M
def M : (ℝ × ℝ) := (-11, -15)

-- Definition of vector CM
def vector_CM : (ℝ × ℝ) := (M.1 - C.1, M.2 - C.2)

-- The condition to prove
theorem find_coordinates_of_M : vector_CM = (2 * vector_CA.1 + 3 * vector_CB.1, 2 * vector_CA.2 + 3 * vector_CB.2) :=
by
  sorry

end find_coordinates_of_M_l317_317872


namespace probability_t_bone_boxes_selected_l317_317263

-- Define the events and conditions
def total_boxes : ℕ := 100
def t_bone_boxes : ℕ := 30
def non_t_bone_boxes : ℕ := 70
def proportion_sampled : ℕ := 10
def t_bone_sampled : ℕ := 3
def non_t_bone_sampled : ℕ := 7
def total_selected : ℕ := 4
def t_bone_selected : ℕ := 2

-- Probability calculation using combinations
theorem probability_t_bone_boxes_selected :
  (comb t_bone_sampled t_bone_selected) * (comb non_t_bone_sampled (total_selected - t_bone_selected)) /
  (comb proportion_sampled total_selected) = 3 / 10 := by
  sorry

end probability_t_bone_boxes_selected_l317_317263


namespace average_monthly_bill_l317_317728

-- Definitions based on conditions
def first_4_months_average := 30
def last_2_months_average := 24
def first_4_months_total := 4 * first_4_months_average
def last_2_months_total := 2 * last_2_months_average
def total_spent := first_4_months_total + last_2_months_total
def total_months := 6

-- The theorem statement
theorem average_monthly_bill : total_spent / total_months = 28 := by
  sorry

end average_monthly_bill_l317_317728


namespace blue_pill_cost_is_correct_l317_317372

-- Define the conditions as Lean definitions
def total_days : ℕ := 21
def total_cost : ℝ := 945
def daily_cost : ℝ := total_cost / total_days
def red_pill_cost (blue_pill_cost : ℝ) : ℝ := blue_pill_cost - 2

-- Prove that the cost of one blue pill is 23.5 dollars
theorem blue_pill_cost_is_correct : ∃ (y : ℝ), (y + (red_pill_cost y) = daily_cost) ∧ y = 23.5 := 
by 
  sorry

end blue_pill_cost_is_correct_l317_317372


namespace rational_solution_condition_l317_317829

theorem rational_solution_condition (x y a b : ℚ) :
  x + y = a → x^2 + y^2 = b → 
  ∃ (x y : ℚ) (h1 : x + y = a) (h2 : x^2 + y^2 = b), 
    ∃ (n : ℚ) (hn : 2 * b - a^2 = n^2) (hpos : 2 * b - a^2 > 0), 
      (x > 0) ∧ (y > 0) ↔ x and y are rational numbers :=
sorry

end rational_solution_condition_l317_317829


namespace intersection_of_A_and_B_union_of_CR_and_B_range_of_m_l317_317458

def A : Set ℝ := { x | 3 < x ∧ x < 10 }
def B : Set ℝ := { x | x^2 - 9*x + 14 < 0 }
def C (m : ℝ) : Set ℝ := { x | 5 - m < x ∧ x < 2*m }
def CR (A : Set ℝ) : Set ℝ := { x | x ≤ 3 ∨ x ≥ 10 }  -- Complement of A in reals

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | 3 < x ∧ x < 7 } := sorry

theorem union_of_CR_and_B : (CR A) ∪ B = { x : ℝ | x < 7 ∨ 10 ≤ x } := sorry

theorem range_of_m {m : ℝ} (suff_not_necess : ∀ x, x ∈ C m → x ∈ A ∩ B) : 
  m ∈ Iic (2 : ℝ) :=
begin
  have h1 : m ≤ 5 / 3 → C m = ∅,
  { sorry }, 
  have h2 : 5 / 3 < m ∧ m ≤ 2 → C m ⊆ A ∩ B,
  { sorry },
  exact sorry
end

end intersection_of_A_and_B_union_of_CR_and_B_range_of_m_l317_317458


namespace describe_shape_as_plane_l317_317941

def cylindrical_coords (r θ z : ℝ) : ℝ × ℝ × ℝ := (r, θ, z)

-- Define the condition θ = 2c in cylindrical coordinates
def constant_angle_plane (c : ℝ) : set (ℝ × ℝ × ℝ) :=
  {p | ∃ r z, p = cylindrical_coords r (2 * c) z}

-- The proof goal is to show that this set represents a plane.
theorem describe_shape_as_plane (c : ℝ) :
  ∃ (plane : set (ℝ × ℝ × ℝ)),
    plane = {p | ∃ r z, p = cylindrical_coords r (2 * c) z}
:= sorry

end describe_shape_as_plane_l317_317941


namespace value_of_expression_l317_317487

theorem value_of_expression (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by {
   rw h,
   norm_num,
   sorry
}

end value_of_expression_l317_317487


namespace suitable_sampling_method_l317_317380

def students_population :=
  (500 : ℕ, 400 : ℕ)

def sample_taken :=
  (25 : ℕ, 20 : ℕ)

theorem suitable_sampling_method :
  sample_taken = (25, 20) →
  students_population = (500, 400) →
  (sample_taken.fst / sample_taken.snd = students_population.fst / students_population.snd) →
  "stratified sampling" :=
by
  intros
  sorry

end suitable_sampling_method_l317_317380


namespace derivative_of_f_at_alpha_l317_317479

variable (α x : ℝ)

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 * Real.cos α - Real.sin x

-- Theorem stating that the derivative of f at α is -cos α
theorem derivative_of_f_at_alpha :
  deriv (f α) = -Real.cos α := sorry

end derivative_of_f_at_alpha_l317_317479


namespace circle_labeling_l317_317579

theorem circle_labeling (n : ℕ) (white_positions black_positions : Fin n → ℕ) :
  (∀ (i : Fin n), 1 ≤ white_positions i ∧ white_positions i ≤ n ∧
                  1 ≤ black_positions i ∧ black_positions i ≤ n) →
  (∀ (i j : Fin n), i ≠ j → white_positions i ≠ white_positions j ∧
                                  black_positions i ≠ black_positions j) →
  ∃ (seq : Fin n → ℕ), (∀ (i : Fin n), 1 ≤ seq i ∧ seq i ≤ n) ∧
                               (∃ (start : ℕ), (∀ (j : Fin n), seq (⟨(start + j) % n, sorry⟩) ∈ set.univ)) ∧
                               {x | ∃ (k : Fin n), seq k = x} = {1, 2, ..., n} :=
by
  sorry

end circle_labeling_l317_317579


namespace symmetry_center_l317_317115

theorem symmetry_center {φ : ℝ} (hφ : |φ| < Real.pi / 2) (h : 2 * Real.sin φ = Real.sqrt 3) : 
  ∃ x : ℝ, 2 * Real.sin (2 * x + φ) = 2 * Real.sin (- (2 * x + φ)) ∧ x = -Real.pi / 6 :=
by
  sorry

end symmetry_center_l317_317115


namespace directrix_of_parabola_l317_317413

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = -3 * x^2 + 6 * x - 5 → y = -35 / 18 := by
  sorry

end directrix_of_parabola_l317_317413


namespace line_AB_fixed_point_minimum_triangle_area_optimal_point_P_l317_317559

-- Definitions for the conditions
variables (x₀ x₁ x₂ : ℝ) (y : ℝ) (P : ℝ × ℝ)

-- Given P on the line y = x - 2
def on_line (P : ℝ × ℝ) := P = (x₀, x₀ - 2)

-- Tangent line equations to the parabola y = 1/2 x^2 passing through the points A(x₁, 1/2 x₁^2) and B(x₂, 1/2 x₂^2)
def tangent_eq (A B : ℝ × ℝ) := 
  A = (x₁, 1/2 * x₁^2) ∧ B = (x₂, 1/2 * x₂^2)

-- Equations derived from the conditions
def tangent_conditions (x₀ x₁ x₂ : ℝ) :=
  x₀ - 2 = x₁ * x₀ - 1 / 2 * x₁^2 ∧ 
  x₀ - 2 = x₂ * x₀ - 1 / 2 * x₂^2 ∧
  x₁ + x₂ = 2 * x₀ ∧ 
  x₁ * x₂ = 2 * x₀ - 4

-- Prove line AB passes through a fixed point (1,2)
theorem line_AB_fixed_point :
  ∀ x₀ x₁ x₂ : ℝ,
    on_line (x₀, x₀ - 2) →
    tangent_eq (x₁, 1/2 * x₁^2) (x₂, 1/2 * x₂^2) →
    tangent_conditions x₀ x₁ x₂ →
    (1, 2) ∈ ℝ × ℝ :=
by
  sorry

-- Minimum area of triangle PAB when point P is moving on the line y = x - 2
theorem minimum_triangle_area :
  ∃ x₀ : ℝ, on_line (x₀, x₀ - 2) ∧
    ∀ S : ℝ, 
    S = (2 * sqrt (x₀^2 - 2 * x₀ + 4) * (x₀^2 - 2 * x₀ + 4)) / (sqrt (x₀^2 + 1)) → 
    S = 3 * sqrt 3 :=
by
  sorry
   
-- Optimal point P when the minimum area is achieved
theorem optimal_point_P :
  ∃ x₀ : ℝ, on_line (x₀, x₀ - 2) ∧
    ∀ P : ℝ × ℝ, P = (1, -1) :=
by
  sorry

end line_AB_fixed_point_minimum_triangle_area_optimal_point_P_l317_317559


namespace simplify_cub_root_multiplication_l317_317600

theorem simplify_cub_root_multiplication (a b : ℝ) (ha : a = 8) (hb : b = 27) :
  (real.cbrt (a + b) * real.cbrt (a + real.cbrt b)) = real.cbrt ((a + b) * (a + real.cbrt b)) := 
by
  sorry

end simplify_cub_root_multiplication_l317_317600


namespace solve_log_equation_l317_317399

theorem solve_log_equation :
  ∃ x : ℝ, log 16 (3 * x - 2) = -1 / 4 ∧ x = 5 / 6 :=
by
  sorry

end solve_log_equation_l317_317399


namespace find_speeds_of_bus_and_cyclist_l317_317838

noncomputable def speeds_of_bus_and_cyclist (v1 v2 : ℝ) : Prop :=
  40 * v1 + 30 * v2 = 37 ∧ 31 * v1 + 51 * v2 = 37 ∧ 
  v1 * 60 = 42 ∧ v2 * 60 = 18

theorem find_speeds_of_bus_and_cyclist :
  ∃ v1 v2 : ℝ, speeds_of_bus_and_cyclist v1 v2 :=
begin
  use [0.7, 0.3],
  repeat { split };
  try { norm_num },
  -- you may refer to solving exact conditions here if needed
  sorry
end

end find_speeds_of_bus_and_cyclist_l317_317838


namespace coeff_x20_greater_in_Q_l317_317956

noncomputable def coeff (f : ℕ → ℕ → ℤ) (p x : ℤ) : ℤ :=
(x ^ 20) * p

noncomputable def P (x : ℤ) := (1 - x^2 + x^3) ^ 1000
noncomputable def Q (x : ℤ) := (1 + x^2 - x^3) ^ 1000

theorem coeff_x20_greater_in_Q :
  coeff 20 (Q x) x > coeff 20 (P x) x :=
  sorry

end coeff_x20_greater_in_Q_l317_317956


namespace ellipse_equation_max_triangle_area_S_value_l317_317226

-- Definitions based on conditions
def ellipse_M (a b : ℝ) : set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ (x^2 / a^2) + (y^2 / b^2) = 1}

def point_A (a : ℝ) : ℝ × ℝ :=
  (a, 0)

def point_B (b : ℝ) : ℝ × ℝ :=
  (0, -b)

def eccentricity_e : ℝ := real.sqrt 2 / 2

def distance_from_origin_to_line_AB (a b : ℝ) : ℝ := 
  2 * real.sqrt 3 / 3

def line_l (m : ℝ) : ℝ × ℝ :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ y = 2 * x + m}

noncomputable def conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (1 - (b^2 / a^2) = eccentricity_e^2) ∧
  (distance_from_origin_to_line_AB a b = 2 * real.sqrt 3 / 3)

noncomputable def max_area_S : ℝ :=
  10 * real.sqrt 2 / 9

-- Prove the equation of the ellipse M
theorem ellipse_equation (a b : ℝ) (h : conditions a b) : 
  ∀ x y, (x^2 / 4) + (y^2 / 2) = 1 :=
sorry

-- Prove the maximum value of the area S
noncomputable def max_triangle_area_S (a b : ℝ) (h : conditions a b) : ℝ :=
  if (line_l m ∈ intersects (ellipse_M a b)) then max_area_S else 0

theorem max_triangle_area_S_value (a b : ℝ) (h : conditions a b): 
  max_triangle_area_S a b h = 10 * real.sqrt 2 / 9 :=
sorry

end ellipse_equation_max_triangle_area_S_value_l317_317226


namespace total_hens_and_cows_l317_317735

theorem total_hens_and_cows (H C : ℕ) (hH : H = 28) (h_feet : 2 * H + 4 * C = 136) : H + C = 48 :=
by
  -- Proof goes here 
  sorry

end total_hens_and_cows_l317_317735


namespace school_badminton_rackets_l317_317720

theorem school_badminton_rackets :
  ∃ (x y : ℕ), x + y = 30 ∧ 50 * x + 40 * y = 1360 ∧ x = 16 ∧ y = 14 :=
by
  sorry

end school_badminton_rackets_l317_317720


namespace ab_ac_ad_bc_bd_cd_value_set_l317_317972

noncomputable def ab_ac_ad_bc_bd_cd_set (a b c d : ℝ) (h : a + b + c + d = 1) : set ℝ :=
  {x | x = ab + ac + ad + bc + bd + cd ∧ 
        a + b + c + d = 1 }

theorem ab_ac_ad_bc_bd_cd_value_set :
  ∃a b c d : ℝ, a + b + c + d = 1 → {x | x = ab + ac + ad + bc + bd + cd} = [0, 3/8] :=
sorry

end ab_ac_ad_bc_bd_cd_value_set_l317_317972


namespace equilateral_triangle_arithmetic_seq_angles_l317_317536

-- Define the problem conditions and the goal to prove
theorem equilateral_triangle_arithmetic_seq_angles
  (A B C : ℝ)
  (a b c : ℝ)
  (AB AC BC : EuclideanGeometry.Vector ℝ)
  (angle_sum : A + B + C = 180)
  (arithmetic_seq : 3 * B = 180)
  (B_eq_60 : B = 60)
  (vector_eq : (AB + AC) • BC = 0)
: ∃ α, α = 60 ∧ angle_sum 
        ∧ arithmetic_seq
        ∧ vector_eq
        ∧ (AB + AC = 2 * EuclideanGeometry.Vector.midpoint BC)
        ∧ (AB.length = AC.length) := sorry

end equilateral_triangle_arithmetic_seq_angles_l317_317536


namespace product_is_112015_l317_317236

-- Definitions and conditions
def is_valid_digit (d : ℕ) : Prop := d ≠ 0 ∧ d < 10

def valid_product (a b : ℕ) (p : ℕ) : Prop :=
  p = a * b ∧
  (∀ d, d ∈ [a / 100, (a % 100) / 10, a % 10, b / 100, (b % 100) / 10, b % 10] → is_valid_digit d) ∧
  (∃ C I K S, C ≠ I ∧ C ≠ K ∧ C ≠ S ∧ I ≠ K ∧ I ≠ S ∧ K ≠ S ∧ is_valid_digit C ∧ is_valid_digit I ∧ is_valid_digit K ∧ is_valid_digit S ∧
    (C ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p % 10]) ∧
    I ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p % 10] ∧
    K ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p %10] ∧
    S ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p % 10]) ∧
  p / 100000 = C

-- Theorem statement
theorem product_is_112015 (a b : ℕ) (h1: a = 521) (h2: b = 215) : valid_product a b 112015 := 
by sorry

end product_is_112015_l317_317236


namespace sum_f_n_is_21_l317_317424

def f (n : ℕ) : ℝ :=
  if (∃ k : ℤ, n = 4^k) then real.log n / real.log 4 else 0

theorem sum_f_n_is_21 : ∑ n in finset.range 4095, f n = 21 := 
sorry

end sum_f_n_is_21_l317_317424


namespace yvette_money_remained_l317_317326

noncomputable def yvette_remaining_money
    (budget : ℝ)
    (new_frame_increase : ℝ)
    (discount : ℝ)
    (sales_tax : ℝ)
    (shipping_fee : ℝ)
    (smaller_frame_ratio : ℝ) : ℝ :=
let initial_new_price := budget * (1 + new_frame_increase / 100) in
let discounted_price := initial_new_price * (1 - discount / 100) in
let taxed_price := discounted_price * (1 + sales_tax / 100) in
let total_price := taxed_price + shipping_fee in
let smaller_frame_price := initial_new_price * smaller_frame_ratio in
let smaller_discounted_price := smaller_frame_price * (1 - discount / 100) in
let smaller_taxed_price := smaller_discounted_price * (1 + sales_tax / 100) in
let smaller_total_price := smaller_taxed_price + shipping_fee in
budget - smaller_total_price

theorem yvette_money_remained :
  yvette_remaining_money 60 20 10 5 5 (3 / 4) = 3.97 :=
by
  sorry

end yvette_money_remained_l317_317326


namespace quadratic_binomial_square_l317_317316

theorem quadratic_binomial_square (k : ℝ) : (∃ b : ℝ, x^2 - 18 * x + k = (x - b)^2) → k = 81 :=
begin
  sorry
end

end quadratic_binomial_square_l317_317316


namespace no_perfect_squares_in_sequence_l317_317138

theorem no_perfect_squares_in_sequence : 
  ∀ N ∈ ({20142015, 201402015, 2014002015, 20140002015, 201400002015} : set ℕ), 
  ¬ (∃ k : ℕ, N = k^2) := 
by
  sorry

end no_perfect_squares_in_sequence_l317_317138


namespace largest_angle_is_176_l317_317724

-- Define the angles of the pentagon
def angle1 (y : ℚ) : ℚ := y
def angle2 (y : ℚ) : ℚ := 2 * y + 2
def angle3 (y : ℚ) : ℚ := 3 * y - 3
def angle4 (y : ℚ) : ℚ := 4 * y + 4
def angle5 (y : ℚ) : ℚ := 5 * y - 5

-- Define the function to calculate the largest angle
def largest_angle (y : ℚ) : ℚ := 5 * y - 5

-- Problem statement: Prove that the largest angle in the pentagon is 176 degrees
theorem largest_angle_is_176 (y : ℚ) (h : angle1 y + angle2 y + angle3 y + angle4 y + angle5 y = 540) :
  largest_angle y = 176 :=
by sorry

end largest_angle_is_176_l317_317724


namespace area_of_circle_l317_317047

noncomputable def calculate_area (d : ℝ) : ℝ :=
  let r := d / 2
  π * r^2

theorem area_of_circle (d : ℝ) (h : d = 13) : calculate_area d ≈ 132.732 :=
  by
  sorry

end area_of_circle_l317_317047


namespace complex_sum_exp_l317_317004

open Complex

theorem complex_sum_exp : (∑ k in finset.range 16, exp (2 * π * I * (k + 1) / 17)) = -1 := 
  sorry

end complex_sum_exp_l317_317004


namespace tomatoes_picked_yesterday_l317_317729

/-
Given:
1. The farmer initially had 171 tomatoes.
2. The farmer picked some tomatoes yesterday (Y).
3. The farmer picked 30 tomatoes today.
4. The farmer will have 7 tomatoes left after today.

Prove:
The number of tomatoes the farmer picked yesterday (Y) is 134.
-/

theorem tomatoes_picked_yesterday (Y : ℕ) (h : 171 - Y - 30 = 7) : Y = 134 :=
sorry

end tomatoes_picked_yesterday_l317_317729


namespace single_elimination_matches_l317_317652

theorem single_elimination_matches (players byes : ℕ)
  (h1 : players = 100)
  (h2 : byes = 28) :
  (players - 1) = 99 :=
by
  -- The proof would go here if it were needed
  sorry

end single_elimination_matches_l317_317652


namespace fraction_of_shaded_triangles_in_eighth_step_is_315_over_4096_l317_317771

theorem fraction_of_shaded_triangles_in_eighth_step_is_315_over_4096 :
  let shaded_triangles (n : ℕ) := Nat.factorial n in
  let total_triangles (n : ℕ) := Finset.range (n + 1) |>.sum (λ k, 8^k) in
  shaded_triangles 8 / total_triangles 8 = (315 : ℝ) / 4096 := by
sorry

end fraction_of_shaded_triangles_in_eighth_step_is_315_over_4096_l317_317771


namespace measurement_error_probability_l317_317572

noncomputable def normal_distribution_cdf (z : ℝ) : ℝ :=
  sorry -- Assume there is an existing function for the CDF of normal distribution

theorem measurement_error_probability :
  let σ := 10
  let δ := 15
  let phi := normal_distribution_cdf
  2 * phi (δ / σ) = 0.8664 :=
by
  intros
  -- The proof is omitted as per instruction
  sorry

end measurement_error_probability_l317_317572


namespace repeating_decimal_sum_l317_317320

theorem repeating_decimal_sum (x : ℚ) (hx : x = 145 / 999) : x.num + x.denom = 1144 :=
sorry

end repeating_decimal_sum_l317_317320


namespace tetrahedron_volume_lower_bound_l317_317248

noncomputable def volume_tetrahedron (d1 d2 d3 : ℝ) : ℝ := sorry

theorem tetrahedron_volume_lower_bound {d1 d2 d3 : ℝ} (h1 : d1 > 0) (h2 : d2 > 0) (h3 : d3 > 0) :
  volume_tetrahedron d1 d2 d3 ≥ (1 / 3) * d1 * d2 * d3 :=
sorry

end tetrahedron_volume_lower_bound_l317_317248


namespace correct_number_of_bananas_l317_317965

variable (B : ℕ)

def fruits_consumed_last_night : ℕ := 3 + B + 4

def fruits_consumed_today : ℕ := 7 + 10 * B + 14

def total_fruits (B : ℕ) : ℕ := fruits_consumed_last_night B + fruits_consumed_today B

theorem correct_number_of_bananas (B : ℕ) : total_fruits B = 39 → B = 1 :=
by
  intro h
  have : total_fruits B = 24 + 11 * B := calc
    total_fruits B
      = (3 + B + 4) + (7 + 10 * B + 14) : by rfl
      = 24 + 11 * B : by ring
  rw [this] at h
  have : 24 + 11 * B = 39 := h
  sorry

end correct_number_of_bananas_l317_317965


namespace estimated_percentage_negative_attitude_l317_317186

-- Define the conditions
def total_parents := 2500
def sample_size := 400
def negative_attitude := 360

-- Prove the estimated percentage of parents with a negative attitude is 90%
theorem estimated_percentage_negative_attitude : 
  (negative_attitude: ℝ) / (sample_size: ℝ) * 100 = 90 := by
  sorry

end estimated_percentage_negative_attitude_l317_317186


namespace product_is_112015_l317_317235

-- Definitions and conditions
def is_valid_digit (d : ℕ) : Prop := d ≠ 0 ∧ d < 10

def valid_product (a b : ℕ) (p : ℕ) : Prop :=
  p = a * b ∧
  (∀ d, d ∈ [a / 100, (a % 100) / 10, a % 10, b / 100, (b % 100) / 10, b % 10] → is_valid_digit d) ∧
  (∃ C I K S, C ≠ I ∧ C ≠ K ∧ C ≠ S ∧ I ≠ K ∧ I ≠ S ∧ K ≠ S ∧ is_valid_digit C ∧ is_valid_digit I ∧ is_valid_digit K ∧ is_valid_digit S ∧
    (C ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p % 10]) ∧
    I ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p % 10] ∧
    K ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p %10] ∧
    S ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p % 10]) ∧
  p / 100000 = C

-- Theorem statement
theorem product_is_112015 (a b : ℕ) (h1: a = 521) (h2: b = 215) : valid_product a b 112015 := 
by sorry

end product_is_112015_l317_317235


namespace value_of_a_sub_b_l317_317457

def f (m x : ℝ) : ℝ := m * x + 2
def g (m x : ℝ) : ℝ := x^2 + 2 * x + m

theorem value_of_a_sub_b (m a b : ℤ) (h : ∀ x : ℤ, a ≤ f m x - g m x ∧ f m x - g m x ≤ b → x ∈ set.Icc a b) : a - b = -2 :=
  sorry

end value_of_a_sub_b_l317_317457


namespace carlos_goals_product_l317_317942

theorem carlos_goals_product :
  ∃ (g11 g12 : ℕ), g11 < 8 ∧ g12 < 8 ∧ 
  (33 + g11) % 11 = 0 ∧ 
  (33 + g11 + g12) % 12 = 0 ∧ 
  g11 * g12 = 49 := 
by
  sorry

end carlos_goals_product_l317_317942


namespace simplify_expression_l317_317593

-- Define the constants.
def a : ℚ := 8
def b : ℚ := 27

-- Assuming cube root function is available and behaves as expected for rationals.
def cube_root (x : ℚ) : ℚ := x^(1/3 : ℚ)

-- Assume the necessary property of cube root of 27.
axiom cube_root_27_is_3 : cube_root 27 = 3

-- The main statement to prove.
theorem simplify_expression : cube_root (a + b) * cube_root (a + cube_root b) = cube_root 385 :=
by
  sorry

end simplify_expression_l317_317593


namespace max_value_is_4_l317_317461

open Real

variables {θ : ℝ}

def a : ℝ × ℝ × ℝ := (cos θ, sin θ, 1)
def b : ℝ × ℝ × ℝ := (sqrt 3, -1, 2)

def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def max_value_of_expression (θ : ℝ) : ℝ :=
  let a := (cos θ, sin θ, 1)
  let b := (sqrt 3, -1, 2)
  let ab := 2 * a.1 - b.1, 2 * a.2 - b.2, 2 * a.3 - b.3
  vector_magnitude ab

theorem max_value_is_4 : max_value_of_expression θ = 4 :=
sorry

end max_value_is_4_l317_317461


namespace time_to_fill_tank_with_leak_l317_317995

-- Definitions based on the given conditions:
def rate_of_pipe_A := 1 / 6 -- Pipe A fills the tank in 6 hours
def rate_of_leak := 1 / 12 -- The leak empties the tank in 12 hours
def combined_rate := rate_of_pipe_A - rate_of_leak -- Combined rate with leak

-- The proof problem: Prove the time taken to fill the tank with the leak present is 12 hours.
theorem time_to_fill_tank_with_leak : 
  (1 / combined_rate) = 12 := by
    -- Proof goes here...
    sorry

end time_to_fill_tank_with_leak_l317_317995


namespace projectile_reaches_100ft_at_time_l317_317745

theorem projectile_reaches_100ft_at_time :
  ∃ t : ℝ, (y t = 100) ∧ t = 3 - (real.sqrt 11) / 2 ∧ round (t * 10) / 10 = 1.3 :=
begin
  let h : ℝ → ℝ := λ t, -16 * t^2 + 96 * t,
  use 3 - real.sqrt 11 / 2,
  split,
  { dsimp [h], sorry },
  split,
  { refl },
  { norm_num }
end

end projectile_reaches_100ft_at_time_l317_317745


namespace liangs_speed_equation_l317_317657

-- Define the speed before training
variable (x : ℝ) (h : x > 0) -- x must be positive for the equation to make sense

-- Define the speeds and times before and after training
def speed_before_training := x
def speed_after_training := 1.25 * x

def time_before_training := 3000 / x
def time_after_training := 3000 / (1.25 * x)

-- State the main theorem to be proved
theorem liangs_speed_equation : 
  time_before_training x h - time_after_training x h = 3 :=
sorry  -- Proof not required for this task

end liangs_speed_equation_l317_317657


namespace range_of_a_l317_317562

noncomputable def odd_function_periodic_real (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ -- odd function condition
  (∀ x, f (x + 5) = f x) ∧ -- periodic function condition
  (f 1 < -1) ∧ -- given condition
  (f 4 = Real.log a / Real.log 2) -- condition using log base 2

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h : odd_function_periodic_real f a) : a > 2 :=
by sorry 

end range_of_a_l317_317562


namespace first_term_exceeding_10000_l317_317625

theorem first_term_exceeding_10000 :
  ∃ (n : ℕ), (2^(n-1) > 10000) ∧ (2^(n-1) = 16384) :=
by
  sorry

end first_term_exceeding_10000_l317_317625


namespace breadth_of_plot_l317_317268

variable (b l : ℝ)

-- Conditions
def area_condition : Prop := l * b = 21 * b
def diff_condition : Prop := l - b = 10

-- The statement to prove
theorem breadth_of_plot : area_condition l b ∧ diff_condition l b → b = 11 :=
by
  intros h,
  cases h with h_area h_diff,
  sorry

end breadth_of_plot_l317_317268


namespace g_50_eq_119_l317_317027

noncomputable def g : ℕ → ℕ 
| x := if (∃ n : ℕ, real.logBase 3 x = n) then (real.logBase 3 x).to_nat
       else 1 + g (x + 2)

theorem g_50_eq_119 : g 50 = 119 := by
  sorry

end g_50_eq_119_l317_317027


namespace remainder_of_8_pow_2002_mod_9_l317_317307

theorem remainder_of_8_pow_2002_mod_9 : ∀ (n : ℕ), n = 2002 → (8 ^ n) % 9 = 1 := 
by
  assume n hn,
  sorry

end remainder_of_8_pow_2002_mod_9_l317_317307


namespace soccer_match_final_score_l317_317382

theorem soccer_match_final_score
  (no_draw : Prop)
  (south_concede_goals : Prop)
  (north_win : Prop)
  (north_not_lose : Prop)
  (three_goals : Prop)
  (exactly_three_correct : ([no_draw, south_concede_goals, north_win, north_not_lose, three_goals].count (λ P, P)) = 3)
  : (north_win ∧ south_concede_goals ∧ three_goals) →
    (¬no_draw ∧ ¬north_not_lose) →
    ∃ final_score : (ℕ × ℕ), final_score = (2, 1) :=
by
  intro h_true_predictions
  intro h_false_predictions
  use (2, 1)
  sorry

end soccer_match_final_score_l317_317382


namespace total_tickets_proof_l317_317656

def total_tickets_sold (A C : ℕ) (adult_cost child_cost total_receipts : ℕ) : ℕ :=
  A + C

theorem total_tickets_proof (A C : ℕ) (adult_cost child_cost total_receipts : ℕ) (hC : C = 90) (h1 : adult_cost = 12) (h2 : child_cost = 4) (h3 : total_receipts = 840) (h4 : adult_cost * A + child_cost * C = total_receipts) : total_tickets_sold A C adult_cost child_cost total_receipts = 130 :=
by
  simp [total_tickets_sold]
  have hA : A = 40 := by
    linarith
  simp [hA, hC]
  done

end total_tickets_proof_l317_317656


namespace A_Q_R_collinear_l317_317436

noncomputable theory

variables {A B C D P Q R : Point} (k₁ k₂ k₃ : Circle)

-- Given conditions translated into Lean:
axiom points_in_plane : ∀ {P Q R : Point}, ¬ collinear P Q R
axiom circle_k₁ : ∀ (A B C D : Point), k₁ ∋ B ∧ k₁ ∋ C ∧ k₁ ∋ D
axiom circle_k₂ : ∀ (A B C D : Point), k₂ ∋ C ∧ k₂ ∋ D ∧ k₂ ∋ A
axiom circle_k₃ : ∀ (A B C D : Point), k₃ ∋ D ∧ k₃ ∋ A ∧ k₃ ∋ B
axiom P_on_k₁ : ∀ (P : Point), k₁ ∋ P
axiom Q_second_intersection : ∀ (P C A B D: Point), k₂ ∋ Q ∧ collinear P Q C
axiom R_second_intersection : ∀ (P B A C D: Point), k₃ ∋ R ∧ collinear P R B

-- Theorem to prove collinearity of A, Q, and R:
theorem A_Q_R_collinear (h₁ : k₁ ∋ B ∧ k₁ ∋ C ∧ k₁ ∋ D)
  (h₂ : k₂ ∋ C ∧ k₂ ∋ D ∧ k₂ ∋ A) 
  (h₃ : k₃ ∋ D ∧ k₃ ∋ A ∧ k₃ ∋ B)
  (h₄ : k₁ ∋ P) 
  (h₅ : k₂ ∋ Q ∧ collinear P Q C) 
  (h₆ : k₃ ∋ R ∧ collinear P R B) : collinear A Q R := 
sorry

end A_Q_R_collinear_l317_317436


namespace ratio_tina_betsy_l317_317386

theorem ratio_tina_betsy :
  ∀ (t_cindy t_betsy t_tina : ℕ),
  t_cindy = 12 →
  t_betsy = t_cindy / 2 →
  t_tina = t_cindy + 6 →
  t_tina / t_betsy = 3 :=
by
  intros t_cindy t_betsy t_tina h_cindy h_betsy h_tina
  sorry

end ratio_tina_betsy_l317_317386


namespace problem1_problem2_problem3_l317_317044

-- Define conditions used in our proof
def cond_1 (a b : ℝ) : Prop := -0.1 < -0.01
def cond_2 (a b : ℝ) : Prop := -(-1) = | -1 |
def cond_3 (a b : ℝ) : Prop := -| -(7/8) | < - (5/6)

-- Theorem statements for the proof problems
theorem problem1 : cond_1 (-0.1) (-0.01) := by
  sorry

theorem problem2 : cond_2 (-(-1)) (| -1 |) := by
  sorry

theorem problem3 : cond_3 (-| -(7/8) |) (- (5/6)) := by
  sorry

end problem1_problem2_problem3_l317_317044


namespace exponent_computation_l317_317017

theorem exponent_computation (x y : Real) :
  ((x ^ -3 * y ^ -4) ^ -1 * (x ^ 2 * y ^ -1) ^ 2) = x ^ 7 * y ^ 2 := 
by sorry

end exponent_computation_l317_317017


namespace evaluate_complex_magnitude_product_l317_317823

noncomputable def complex_magnitude (z : Complex) : ℝ :=
  Complex.abs z

theorem evaluate_complex_magnitude_product :
  let z1 := 5 * Real.sqrt 2 - Complex.i * (5 : ℂ)
  let z2 := 2 * Real.sqrt 3 + Complex.i * (4 : ℂ)
  complex_magnitude (z1 * z2) = 10 * Real.sqrt 21 :=
by
  sorry

end evaluate_complex_magnitude_product_l317_317823


namespace percentage_gain_is_20_l317_317755

noncomputable def costPrice (SP_loss : ℝ) (loss_percent : ℝ) : ℝ :=
  SP_loss / (1 - loss_percent)

noncomputable def gainPercent (SP_gain : ℝ) (CP : ℝ) : ℝ :=
  ((SP_gain - CP) / CP) * 100

theorem percentage_gain_is_20 (SP_loss SP_gain : ℝ) (loss_percent : ℝ) (expected_gain : ℝ) (real_CP : ℝ) :
  SP_loss = 136 → loss_percent = 0.15 → SP_gain = 192 → real_CP = costPrice SP_loss loss_percent → 
  gainPercent SP_gain real_CP = 20 :=
by
  intros h1 h2 h3 h4
  rw h4 -- Use the provided cost price
  rw [h1, h2] -- Replace the SP_loss and loss_percent in the cost price formula
  simp -- Simplify the expressions
  sorry -- Omitted proof steps

end percentage_gain_is_20_l317_317755


namespace cross_section_area_l317_317271

variables (V α : ℝ)

noncomputable def area_cross_section (V : ℝ) (α : ℝ) : ℝ :=
  Real.cbrt (3 * Real.sqrt 3 * V^2 / (Real.sin α)^2 * Real.cos α)

theorem cross_section_area (hV : V > 0) (hα : 0 < α ∧ α < Real.pi / 2) :
  area_cross_section V α = Real.cbrt (3 * Real.sqrt 3 * V^2 / (Real.sin α)^2 * Real.cos α) :=
sorry

end cross_section_area_l317_317271


namespace distance_between_A_and_B_l317_317319

theorem distance_between_A_and_B 
  (d : ℝ)
  (h1 : ∀ (t : ℝ), (t = 2 * (t / 2)) → t = 200) 
  (h2 : ∀ (t : ℝ), 100 = d - (t / 2 + 50))
  (h3 : ∀ (t : ℝ), d = 2 * (d - 60)): 
  d = 300 :=
sorry

end distance_between_A_and_B_l317_317319


namespace probability_CMWMC_block_l317_317714

theorem probability_CMWMC_block : 
  let total_permutations := nat.factorial 9 / (nat.factorial 3 * nat.factorial 3 * nat.factorial 3) in
  let favorable_permutations := nat.factorial 5 / (nat.factorial 1 * nat.factorial 1 * nat.factorial 2 * nat.factorial 1) in
  let probability := favorable_permutations / total_permutations in
  probability = 1 / 28 :=
by
  sorry

end probability_CMWMC_block_l317_317714


namespace rearrange_numbers_l317_317198

theorem rearrange_numbers (n : ℕ) (h : n = 25 ∨ n = 1000) :
  ∃ (l : List ℕ), l.nodup ∧ l.perm (List.range (n + 1))
  ∧ (∀ (i : ℕ), i < n → ((l.get? i.succ = some (l.get! i + 3)) ∨ (l.get? i.succ = some (l.get! i + 5)) ∨ (l.get? i.succ = some (l.get! i - 3)) ∨ (l.get? i.succ = some (l.get! i - 5)))) :=
by
  sorry

end rearrange_numbers_l317_317198


namespace three_digit_number_ends_in_square_abc_equals_864_l317_317400

theorem three_digit_number_ends_in_square_abc_equals_864 :
  ∃ (a b c : ℕ), a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  (10 * b + c = a ^ 2) ∧ ((100 * a + 10 * b + c) % 9 = 4) ∧ (100 * a + 10 * b + c = 864) :=
by
  sorry

end three_digit_number_ends_in_square_abc_equals_864_l317_317400


namespace hyperbola_eccentricity_correct_l317_317099

noncomputable def hyperbola_eccentricity (a b : ℝ) (a_pos : a > b) (b_pos : b > 0) : ℝ :=
  let c := (a^2 + b^2) ^ (1/2) in c / a

theorem hyperbola_eccentricity_correct (a b : ℝ) (h : a > b) (hb : b > 0)
  (h_intersects : ∀ P Q : ℝ → ℝ, (P ≠ Q) ∧
    (∃ x y : ℝ, \frac{x^{2}}{5} + y^{2} = 1 ∧ P(x, y) ∧ Q(x, y))
  ∧ (F : ℝ → ℝ, PF ⊥ QF)) :
  hyperbola_eccentricity a b h hb = \frac{4}{15} * sqrt(15) := by
  sorry

end hyperbola_eccentricity_correct_l317_317099


namespace unique_denominators_count_l317_317264

theorem unique_denominators_count :
  ∀ (c d : ℕ), (c ≤ 9) → (d ≤ 9) → (c ≠ 0 ∨ d ≠ 0) → (c ≠ 9 ∨ d ≠ 9) →
  let n := 10 * c + d in
  let factors := {1, 13, 17, 221}.erase 1 in
  let denominators := factors.filter (λ k, k ∣ 221 ∧ ¬(∃ m, n = k * m)) in
  denominators.card = 3 := 
by
  sorry

end unique_denominators_count_l317_317264


namespace count_three_digit_numbers_with_sum_20_l317_317835

theorem count_three_digit_numbers_with_sum_20 : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ a + b + c = 20) → (Finset.card ((Finset.range 10).product (Finset.range 10).product (Finset.range 10) | p in λ p : ℕ × ℕ × ℕ, 
    let ⟨a, b, c⟩ := p in 
    (1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ a + b + c = 20)) = 4 :=
begin
  sorry
end

end count_three_digit_numbers_with_sum_20_l317_317835


namespace translation_coordinates_l317_317187

theorem translation_coordinates :
  ∀ (x y : ℤ) (a : ℤ), 
  (x, y) = (3, -4) → a = 5 → (x - a, y) = (-2, -4) :=
by
  sorry

end translation_coordinates_l317_317187


namespace equation_of_circle_correct_equation_of_line_correct_l317_317880

noncomputable def equation_of_circle (M N P : Real × Real) : Real × Real × Real :=
  let a := 0 in
  let b := 0 in
  let r := 1 in
  ⟨a, b, r⟩

theorem equation_of_circle_correct :
  let M := (-1, 0) in
  let N := (0, 1) in
  let P := (1/2, -Real.sqrt 3 / 2) in
  equation_of_circle M N P = (0, 0, 1) := 
by
  intros
  sorry

def equation_of_line (C M : Real × Real) (r : Real) : Real × Real :=
  let xC := 2 in
  let yC := 2 in
  let A := xC - 1/sqrt 2 in
  let B := yC - 1/sqrt 2 in
  ⟨A, B⟩

theorem equation_of_line_correct :
  let M := (-1, 0) in
  let N := (0, 1) in
  let P := (1/2, -Real.sqrt 3 / 2) in
  let C := (2, 2) in
  let line_AB := (0.5, 0.5) in
  equation_of_line C (0, 0) 1 = line_AB :=
by
  intros
  sorry

end equation_of_circle_correct_equation_of_line_correct_l317_317880


namespace infinite_sets_of_seven_numbers_l317_317465

theorem infinite_sets_of_seven_numbers
    (S : Set ℝ)
    (h : ∀ a ∈ S, ∃ b c ∈ S, a = b * c)
    (card_S : S.card = 7) :
    ∃ (S' : Set ℝ), S ≠ S' ∧ ∀ S'' ∈ (@Powerset ℝ), ∃ (S' : Set ℝ), S ≠ S' ∧ ∀ a ∈ S', ∃ b c ∈ S', a = b * c :=
sorry

end infinite_sets_of_seven_numbers_l317_317465


namespace OQ_perpendicular_PQ_l317_317969

open EuclideanGeometry
open Real

theorem OQ_perpendicular_PQ
  {A B C D O P Q : Point} 
  (hABC: ConvexQuadrilateral A B C D)
  (hO: Circle O A B C D)
  (hP: Intersection AC BD P)
  (hQ: CircleIntersect ABP CD Q) :
    perp OQ PQ := 
by 
    sorry

end OQ_perpendicular_PQ_l317_317969


namespace sample_stddev_and_range_unchanged_l317_317864

noncomputable def sample_data (n : ℕ) : Type :=
  fin n → ℝ

variables (n : ℕ) (x y : sample_data n) (c : ℝ)

-- Condition: creating new sample data by adding a constant c
axiom data_transform : ∀ i : fin n, y i = x i + c 

-- Condition: c is non-zero
axiom c_nonzero : c ≠ 0

-- The theorem that states sample standard deviations and sample ranges of x and y are the same
theorem sample_stddev_and_range_unchanged :
  (sample_standard_deviation y = sample_standard_deviation x) ∧ 
  (sample_range y = sample_range x) :=
sorry

end sample_stddev_and_range_unchanged_l317_317864


namespace area_of_square_inscribed_in_circle_l317_317285

theorem area_of_square_inscribed_in_circle (a : ℝ) :
  ∃ S : ℝ, S = (2 * a^2) / 3 :=
sorry

end area_of_square_inscribed_in_circle_l317_317285


namespace number_of_sheep_l317_317510

theorem number_of_sheep (s d : ℕ) 
  (h1 : s + d = 15)
  (h2 : 4 * s + 2 * d = 22 + 2 * (s + d)) : 
  s = 11 :=
by
  sorry

end number_of_sheep_l317_317510


namespace eighty_five_squared_l317_317011

theorem eighty_five_squared :
  (85:ℕ)^2 = 7225 := 
by
  let a := 80
  let b := 5
  have h1 : (a + b) = 85 := rfl
  have h2 : (a^2 + 2 * a * b + b^2) = 7225 := by norm_num
  rw [←h1, ←h1]
  rw [ sq (a + b)]
  rw [ mul_add, add_mul, add_mul, mul_comm 2 b]
  rw [←mul_assoc, ←mul_assoc, add_assoc, add_assoc, nat.add_right_comm ]
  exact h2

end eighty_five_squared_l317_317011


namespace cos_double_angle_l317_317104

theorem cos_double_angle (α : ℝ) 
  (h : ∀ x y : ℝ, (x = -√5 / 5) ∧ (y = 2 * √5 / 5) → x^2 + y^2 = 1) :
  cos (2 * α) = -3 / 5 := 
begin
  sorry
end

end cos_double_angle_l317_317104


namespace chebyshevs_inequality_for_nonneg_reals_l317_317978

theorem chebyshevs_inequality_for_nonneg_reals (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 ≤ a i) :
    n * (∑ i, (a i)^3) ≥ (∑ i, a i) * (∑ i, (a i)^2) := 
by
  sorry

end chebyshevs_inequality_for_nonneg_reals_l317_317978


namespace range_of_a_no_solution_inequality_l317_317475

theorem range_of_a_no_solution_inequality (a : ℝ) :
  (∀ x : ℝ, x + 2 > 3 → x < a) ↔ a ≤ 1 :=
by {
  sorry
}

end range_of_a_no_solution_inequality_l317_317475


namespace parabola_focus_coordinates_l317_317030

theorem parabola_focus_coordinates :
  ∃ h k : ℝ, (y = -1/8 * x^2 + 2 * x - 1) ∧ (h = 8 ∧ k = 5) :=
sorry

end parabola_focus_coordinates_l317_317030


namespace true_statements_l317_317220

variables {m n : Line} {α β : Plane}

-- Define the necessary conditions and questions

def non_coincident_lines (m n : Line) : Prop := ¬ (m = n)
def non_coincident_planes (α β : Plane) : Prop := ¬ (α = β)

def perpendicular_line_plane (m : Line) (α : Plane) : Prop := -- definition of line perpendicular to a plane
sorry

def parallel_lines (m n : Line) : Prop := -- definition of parallel lines
sorry

def line_in_plane (m : Line) (α : Plane) : Prop := -- definition of a line lying within a plane
sorry

def perpendicular_planes (α β : Plane) : Prop := -- definition of perpendicular planes
sorry

-- Prove that statements 1 and 3 are true

theorem true_statements
  (h_lines_non_coincident : non_coincident_lines m n)
  (h_planes_non_coincident : non_coincident_planes α β)
  (h1 : perpendicular_line_plane m α)
  (h2 : perpendicular_line_plane n α)
  (h3 : parallel_lines m α)
  (h4 : perpendicular_line_plane n β)
  (h5 : line_in_plane m α)
  (h6 : parallel_lines n β)
  (h7 : perpendicular_planes α β) :
  (parallel_lines m n) ∧ (perpendicular_line_plane m n) :=
by {
  -- proof steps would be here
  sorry
}

end true_statements_l317_317220


namespace ellipse_equation_and_fixed_point_l317_317883

theorem ellipse_equation_and_fixed_point :
  ∃ (a b : ℝ) (k m : ℝ),
  a > b ∧ b > 0 ∧
  -- Conditions for the ellipse and tangent line
  ∃ c : ℝ, (c / a = 1 / 2) ∧ ((sqrt 3 * c) / 2 = b / a) ∧ (a^2 = b^2 + c^2) ∧
  -- Form of the ellipse equation
  (a = 2) ∧ (c = 1) ∧ (b = sqrt 3) ∧
  -- Equation of the ellipse
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  -- Conditions for line passing through fixed point
  let Mx My (kx m : ℝ) := (Mx, My) in
  let Nx Ny (kx m : ℝ) := (Nx, Ny) in
  (3 + 4 * k^2 > m^2) ∧
  ((1 + k^2) * (Mx * Nx) + ((m * k) - 2) * (Mx + Nx) + m^2 + 4 = 0) ↔
  -- Fixed point condition
  (∃ px py : ℝ, px = 2 / 7 ∧ py = 0)
:= sorry

end ellipse_equation_and_fixed_point_l317_317883


namespace final_price_correct_l317_317754

def list_price : ℝ := 150
def first_discount_percentage : ℝ := 19.954259576901087
def second_discount_percentage : ℝ := 12.55

def first_discount_amount : ℝ := list_price * (first_discount_percentage / 100)
def price_after_first_discount : ℝ := list_price - first_discount_amount
def second_discount_amount : ℝ := price_after_first_discount * (second_discount_percentage / 100)
def final_price : ℝ := price_after_first_discount - second_discount_amount

theorem final_price_correct : final_price = 105 := by
  sorry

end final_price_correct_l317_317754


namespace convex_polygon_partition_l317_317957

open Set

-- Define a convex polygon and let it contain the circles
variables {Poly : Set (ℝ × ℝ)} (Poly_convex : IsConvex Poly)
variables {Circles : Fin (n : ℕ) → Circle} (Circles_disjoint : ∀(i j : Fin n), i ≠ j → ((Circles i).set ∩ (Circles j).set) = ∅)
variables {Circles_radii_distinct : ∀ (i j : Fin n), i ≠ j → (Circles i).radius ≠ (Circles j).radius}

theorem convex_polygon_partition :
  ∃ (parts : Fin n → Set (ℝ × ℝ)),
    (∀ i, IsConvex (parts i)) ∧ 
    (∀ i, (Circles i).set ⊆ parts i) ∧ 
    (⋃ i, parts i) ⊆ Poly ∧
    (∀ i j, i ≠ j → Disjoint (parts i) (parts j)) :=
  sorry

end convex_polygon_partition_l317_317957


namespace new_container_volume_l317_317730

theorem new_container_volume (original_volume : ℕ) (factor : ℕ) (new_volume : ℕ) 
    (h1 : original_volume = 5) (h2 : factor = 4 * 4 * 4) : new_volume = 320 :=
by
  sorry

end new_container_volume_l317_317730


namespace value_of_a_l317_317870

-- Define sets as given in the conditions
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {0, 1, 2}

-- Define the element a and the inclusion/exclusion conditions
variable (a : ℕ)
hypothesis (h1 : a ∈ A)
hypothesis (h2 : a ∉ B)

-- State the theorem we need to prove
theorem value_of_a : a = 3 := sorry

end value_of_a_l317_317870


namespace find_100th_negative_index_l317_317392

noncomputable def sequence_b (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), Real.cos k

theorem find_100th_negative_index :
  ∃ n, ∃ count < 100, sequence_b n < 0 ∧
    (count = 99 → (n = 628)) :=
sorry

end find_100th_negative_index_l317_317392


namespace range_of_a_l317_317119

open Real

-- Define the inequality condition
def inequality (a x : ℝ) : Prop := x^2 + a*x - 2 > 0

-- The theorem we want to prove given the conditions
theorem range_of_a (a : ℝ) (h : ∃ x ∈ Icc (1 : ℝ) 2, inequality a x) : a > -1 := 
sorry

end range_of_a_l317_317119


namespace simplify_expression_l317_317605

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem simplify_expression :
  cube_root (8 + 27) * cube_root (8 + cube_root 27) = cube_root 385 :=
by
  sorry

end simplify_expression_l317_317605


namespace calc_delta_l317_317921

noncomputable def delta (a b : ℝ) : ℝ :=
  (a^2 + b^2) / (1 + a * b)

-- Definition of the main problem as a Lean 4 statement
theorem calc_delta (h1 : 2 > 0) (h2 : 3 > 0) (h3 : 4 > 0) :
  delta (delta 2 3) 4 = 6661 / 2891 :=
by
  sorry

end calc_delta_l317_317921


namespace circles_intersect_common_chord_equation_common_chord_length_l317_317130

noncomputable def circle_1_center := (3 : ℝ, 0 : ℝ)
noncomputable def circle_1_radius := real.sqrt 15
noncomputable def circle_2_center := (0 : ℝ, 2 : ℝ)
noncomputable def circle_2_radius := real.sqrt 10

def distance_between_centers :=
  real.sqrt ((circle_1_center.fst - circle_2_center.fst) ^ 2 + (circle_1_center.snd - circle_2_center.snd) ^ 2)

theorem circles_intersect :
  real.sqrt 15 - real.sqrt 10 < distance_between_centers ∧ distance_between_centers < real.sqrt 15 + real.sqrt 10 := 
sorry

theorem common_chord_equation :
  ∃ a b c : ℝ, a * 3 + b * (-2) = c ∧ a * 0 + b * 0 = c ∧ a * 3 + b * 0 = c ∧
  (∀ (x y : ℝ), x^2 + y^2 - 6 * x - 6 = 0 ∧ x^2 + y^2 - 4 * y - 6 = 0 → 3 * x - 2 * y = 0) := 
sorry

theorem common_chord_length :
  let d := (abs (3 * circle_1_center.fst + 0 - 2 * circle_1_center.snd) / real.sqrt (3 ^ 2 + (-2) ^ 2)) in
  2 * real.sqrt (15 - (d * d)/13) = (2 * real.sqrt 1182) / 13 := 
sorry

end circles_intersect_common_chord_equation_common_chord_length_l317_317130


namespace no_values_of_n_le_40_such_that_g20_eq_9_l317_317833

def divisors_count (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).card

def g1 (n : ℕ) : ℕ := 3 * divisors_count n

def gj : ℕ → ℕ → ℕ
| 0, n := n
| (j + 1), n := g1 (gj j n)

theorem no_values_of_n_le_40_such_that_g20_eq_9 :
  ∀ n : ℕ, n ≤ 40 → gj 20 n ≠ 9 :=
by sorry

end no_values_of_n_le_40_such_that_g20_eq_9_l317_317833


namespace isosceles_inscribed_equilateral_l317_317940

theorem isosceles_inscribed_equilateral {α β γ : ℝ} 
  (isosceles : ∀ {A B C : ℝ}, A = B → B = C)
  (equilateral_inscribed : ∀ {A B C : ℝ}, (A + B + C = 180) 
  (angles_BFD_ADE_FEC : α = β ∧ β = γ ∧ γ = α) : α = (β + γ) / 2 :=
begin
  sorry
end

end isosceles_inscribed_equilateral_l317_317940


namespace solution_set_correct_l317_317886

theorem solution_set_correct (a b c : ℝ) (h : a < 0) (h1 : ∀ x, (ax^2 + bx + c < 0) ↔ ((x < 1) ∨ (x > 3))) :
  ∀ x, (cx^2 + bx + a > 0) ↔ (1 / 3 < x ∧ x < 1) :=
sorry

end solution_set_correct_l317_317886


namespace modulo_remainder_l317_317054

theorem modulo_remainder :
  (7 * 10^24 + 2^24) % 13 = 8 := 
by
  sorry

end modulo_remainder_l317_317054


namespace directrix_of_parabola_l317_317411

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = -3 * x^2 + 6 * x - 5 → y = -35 / 18 := by
  sorry

end directrix_of_parabola_l317_317411


namespace sum_fractions_l317_317398

theorem sum_fractions :
  (1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)) = (2 / 9) :=
by
  sorry

end sum_fractions_l317_317398


namespace cost_per_pouch_in_cents_l317_317540

-- Define the conditions
def boxes : ℕ := 10
def pouches_per_box : ℕ := 6
def total_dollars : ℝ := 12

-- Define the theorem to prove the cost per pouch in cents
theorem cost_per_pouch_in_cents : 
  let total_pouches := boxes * pouches_per_box in
  let total_cents := total_dollars * 100 in
  total_cents / total_pouches = 20 := 
by
  sorry

end cost_per_pouch_in_cents_l317_317540


namespace fifteenth_odd_multiple_of_5_l317_317304

theorem fifteenth_odd_multiple_of_5 : 
  let a_n (n : ℕ) := 10 * n - 5 in
  a_n 15 = 145 :=
by
  sorry

end fifteenth_odd_multiple_of_5_l317_317304


namespace time_to_cross_bridge_l317_317366

noncomputable def train_crossing_time
  (length_train : ℕ)
  (length_bridge : ℕ)
  (speed_kmph : ℕ)
  (conversion_factor : ℚ) : ℚ :=
  (length_train + length_bridge) / (speed_kmph * conversion_factor)

theorem time_to_cross_bridge :
  train_crossing_time 135 240 45 (5 / 18) = 30 := by
  sorry

end time_to_cross_bridge_l317_317366


namespace larger_model_ratio_smaller_model_ratio_l317_317286

-- Definitions for conditions
def statue_height := 305 -- The height of the actual statue in feet
def larger_model_height := 10 -- The height of the larger model in inches
def smaller_model_height := 5 -- The height of the smaller model in inches

-- The ratio calculation for larger model
theorem larger_model_ratio : 
  (statue_height : ℝ) / (larger_model_height : ℝ) = 30.5 := by
  sorry

-- The ratio calculation for smaller model
theorem smaller_model_ratio : 
  (statue_height : ℝ) / (smaller_model_height : ℝ) = 61 := by
  sorry

end larger_model_ratio_smaller_model_ratio_l317_317286


namespace angle_BCA_correct_l317_317935

def convex_quadrilateral (A B C D M O : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited M] [Inhabited O] :=
  let angle (P Q R : Type) := ℝ -- angles in degrees
  let midpoint (P Q R : Type) := sorry -- you can define this as needed
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ midpoint A D M ∧
  (BM = sorry intersect AC at O) ∧
  (angle A B M = 55) ∧ (angle A M B = 70) ∧
  (angle B O C = 80) ∧ (angle A D C = 60)

theorem angle_BCA_correct {A B C D M O : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited M] [Inhabited O] :
  convex_quadrilateral A B C D M O → angle B C A = 35 :=
begin
  intro h,
  sorry -- Proof goes here
end

end angle_BCA_correct_l317_317935


namespace find_circumference_semi_circle_l317_317331

noncomputable def semi_circumference (π : Real) (side : Real) : Real := (π * side / 2) + side

def perimeter_rectangle (length breadth : ℝ) : ℝ := 2 * (length + breadth)
def perimeter_square (side : ℝ) : ℝ := 4 * side
def side_of_square (perimeter : ℝ) : ℝ := perimeter / 4

theorem find_circumference_semi_circle :
  let length := 16
  let breadth := 14
  let perm_rec := perimeter_rectangle length breadth
  let perm_sq := perm_rec
  let side := side_of_square perm_sq
  let diameter := side
  let π := 3.14
  semi_circumference π diameter = 38.55 :=
by
  let length := 16
  let breadth := 14
  let perm_rec := perimeter_rectangle length breadth
  let perm_sq := perm_rec
  let side := side_of_square perm_sq
  let diameter := side
  let π := 3.14
  have h1 : perm_rec = 60 := by sorry
  have h2 : perm_sq = 60 := by sorry
  have h3 : side = 15 := by sorry
  have h4 : diameter = 15 := by sorry
  calc
    semi_circumference π diameter
      = (π * diameter / 2 + diameter) : by rfl
    ... = (3.14 * 15 / 2 + 15) : by sorry
    ... = 38.55 : by sorry

end find_circumference_semi_circle_l317_317331


namespace summation_inequality_l317_317333

theorem summation_inequality 
    (n : ℕ) (n_ge_2 : 2 ≤ n) 
    (a : Fin n → ℝ) (a_ascending : ∀ (i j : Fin n), i < j → a i < a j) 
    (a_pos : ∀ i, 0 < a i)
    (sum_reciprocal_le_1 : (∑ i, 1 / a i) ≤ 1)
    (x : ℝ) :
    ( (∑ i, 1 / (a i ^ 2 + x ^ 2)) ^ 2 <= 1 / 2 * (1 / (a 0 * (a 0 - 1) + x ^ 2)) ) :=
by
  sorry

end summation_inequality_l317_317333


namespace sample_stddev_and_range_unchanged_l317_317861

noncomputable def sample_data (n : ℕ) : Type :=
  fin n → ℝ

variables (n : ℕ) (x y : sample_data n) (c : ℝ)

-- Condition: creating new sample data by adding a constant c
axiom data_transform : ∀ i : fin n, y i = x i + c 

-- Condition: c is non-zero
axiom c_nonzero : c ≠ 0

-- The theorem that states sample standard deviations and sample ranges of x and y are the same
theorem sample_stddev_and_range_unchanged :
  (sample_standard_deviation y = sample_standard_deviation x) ∧ 
  (sample_range y = sample_range x) :=
sorry

end sample_stddev_and_range_unchanged_l317_317861


namespace molecular_weight_of_compound_l317_317672

theorem molecular_weight_of_compound (C H O n : ℕ) 
    (atomic_weight_C : ℝ) (atomic_weight_H : ℝ) (atomic_weight_O : ℝ) 
    (total_weight : ℝ) 
    (h_C : C = 2) (h_H : H = 4) 
    (h_atomic_weight_C : atomic_weight_C = 12.01) 
    (h_atomic_weight_H : atomic_weight_H = 1.008) 
    (h_atomic_weight_O : atomic_weight_O = 16.00) 
    (h_total_weight : total_weight = 60) : 
    C * atomic_weight_C + H * atomic_weight_H + n * atomic_weight_O = total_weight → 
    n = 2 := 
sorry

end molecular_weight_of_compound_l317_317672


namespace sum_at_2_eq_10_l317_317214

noncomputable def p1 : Polynomial ℤ := Polynomial.X^2 + 1
noncomputable def p2 : Polynomial ℤ := Polynomial.X^3 - Polynomial.X - 1

lemma p1_is_monic : p1.monic := by sorry
lemma p2_is_monic : p2.monic := by sorry
lemma p1_irreducible : irreducible p1 := by sorry
lemma p2_irreducible : irreducible p2 := by sorry
lemma factorization_correct : Polynomial.X^5 - Polynomial.X^2 - Polynomial.X - 1 = p1 * p2 := by sorry

theorem sum_at_2_eq_10 : p1.eval 2 + p2.eval 2 = 10 := by sorry

end sum_at_2_eq_10_l317_317214


namespace intersection_points_distance_l317_317050

-- Define the conditions
def quadratic_eq (x y : ℝ) : Prop := x^2 + y = 11
def linear_eq (x y : ℝ) : Prop := x + y = 11

-- State the theorem
theorem intersection_points_distance :
  ∃ x1 y1 x2 y2 : ℝ,
    quadratic_eq x1 y1 ∧
    linear_eq x1 y1 ∧
    quadratic_eq x2 y2 ∧
    linear_eq x2 y2 ∧
    ((x1, y1) ≠ (x2, y2)) ∧
    real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = real.sqrt 2 :=
by {
  -- skip the proof
  sorry,
}

end intersection_points_distance_l317_317050


namespace max_beach_volleyball_players_l317_317039

theorem max_beach_volleyball_players (n : ℕ) :
  (∀ (g : ℕ), g < n → ℕ (/4) players in each game) →  -- Each game has 4 players
  (∀ (p : ℕ), p < n → ℕ (p) participates in n games) →  -- Each player plays in n games
  (∀ (p1 p2 : ℕ), p1 < n → p2 < n → (p1 ≠ p2 → ∃ (g : ℕ), g < n ∧ p1 ∈ game(g) ∧ p2 ∈ game(g))) →  -- Any two players have played together in at least one game
  n ≤ 13 := -- We need to prove that n is no more than 13
sorry

end max_beach_volleyball_players_l317_317039


namespace problem_I4_1_l317_317879

theorem problem_I4_1 
  (x y : ℝ)
  (h : (10 * x - 3 * y) / (x + 2 * y) = 2) :
  (y + x) / (y - x) = 15 :=
sorry

end problem_I4_1_l317_317879


namespace three_x_plus_four_l317_317494

theorem three_x_plus_four (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  sorry

end three_x_plus_four_l317_317494


namespace final_price_correct_l317_317751

noncomputable def final_price_after_discounts (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let first_discount_amount := p * d1 / 100
  let price_after_first_discount := p - first_discount_amount
  let second_discount_amount := price_after_first_discount * d2 / 100
  price_after_first_discount - second_discount_amount

theorem final_price_correct :
  final_price_after_discounts 150 19.954259576901087 12.55 ≈ 105.00000063464838 :=
by 
  -- Approximate calculations to assert the correctness
  sorry

end final_price_correct_l317_317751


namespace sum_of_exterior_angles_of_hexagon_l317_317650

theorem sum_of_exterior_angles_of_hexagon {α : Type} [linear_ordered_semiring α] :
  ∀ (hexagon : fin 6 → α × α), ∑ i in finset.fin_range 6, exterior_angle (hexagon i) = 360 :=
by
  sorry

end sum_of_exterior_angles_of_hexagon_l317_317650


namespace factorization_l317_317043

theorem factorization (a b : ℝ) : a^2 * b - 6 * a * b + 9 * b = b * (a - 3)^2 :=
by sorry

end factorization_l317_317043


namespace divisible_by_10001_l317_317768

def is_formed_by_repeating (n : ℕ) : Prop :=
  ∃ (abcd : ℕ), abcd < 10000 ∧ n = 10001 * abcd

theorem divisible_by_10001 (n : ℕ) :
  is_formed_by_repeating n → 10001 ∣ n :=
by { intro h, cases h with abcd h_abcd, rw h_abcd.2, use abcd, }

-- This theorem indicates that any eight-digit number that is formed by repeating a four-digit
-- number is divisible by 10001.

end divisible_by_10001_l317_317768


namespace find_ax6_by6_l317_317478

variable {a b x y : ℝ}

theorem find_ax6_by6
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 12)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 80) :
  a * x^6 + b * y^6 = 1531.25 :=
sorry

end find_ax6_by6_l317_317478


namespace max_sum_of_squares_of_roots_l317_317671

theorem max_sum_of_squares_of_roots {a b c d : ℤ} (h : {a, b, c, d} = {2, 0, 1, 7})
  : let r := 49 in
  (a^2 - 2 * b) ≤ r :=
sorry

end max_sum_of_squares_of_roots_l317_317671


namespace det_E_eq_81_l317_317970

-- Define the matrix E corresponding to the dilation with scale factor 9 centered at the origin.
def E : Matrix (Fin 2) (Fin 2) ℝ := ![![9, 0], ![0, 9]]

-- State the theorem that the determinant of E is 81.
theorem det_E_eq_81 : det E = 81 := by
  sorry

end det_E_eq_81_l317_317970


namespace david_average_marks_is_93_l317_317812

def marks : List ℕ := [96, 95, 82, 97, 95]

def total_marks : ℕ := marks.sum

def number_of_subjects : ℕ := marks.length

def average_marks : ℕ := total_marks / number_of_subjects

theorem david_average_marks_is_93 : average_marks = 93 := 
by 
  unfold average_marks
  unfold total_marks
  unfold number_of_subjects
  unfold marks
  simp
  sorry

end david_average_marks_is_93_l317_317812


namespace quadratic_square_binomial_l317_317314

theorem quadratic_square_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 18 * x + k = (x + b) ^ 2) ↔ k = 81 := by
  sorry

end quadratic_square_binomial_l317_317314


namespace min_area_ABM_l317_317120

variables {A B F M : Type*}
variables (x1 y1 x2 y2 x0 y0 : ℝ)
variable (lambda : ℝ)

noncomputable def parabola (x y : ℝ) : Prop := x ^ 2 = 4 * y
def focus : (ℝ × ℝ) := (0, 1)

def on_parabola_A : Prop := parabola x1 y1
def on_parabola_B : Prop := parabola x2 y2

def tangent_at (p : ℝ × ℝ) (x : ℝ) : ℝ :=
  let (px, py) := p in (1 / 2) * px * (x - px) + py

def intersect_tangents : ℝ × ℝ := (2 * (x1 + x2) / 2, -1)
def vector_AF : ℝ × ℝ := (-x1, 1 - y1)
def vector_FB : ℝ × ℝ := (x2, y2 - 1)
def vector_FM : ℝ × ℝ := (2 * (x1 + x2) / 2, -2)
def vector_AB : ℝ × ℝ := ((x2 - x1), (y2 - y1))

lemma FM_dot_AB_const : vector_FM x1 x2 • vector_AB x1 y1 x2 y2 = 0 :=
sorry

noncomputable def area_of_triangle (AB FM : ℝ) : ℝ := (1 / 2) * AB * FM

noncomputable def AB_len : ℝ :=
  let x1_sq := x1 ^ 2 in
  let x2_sq := x2 ^ 2 in
  let y1_val := x1_sq / 4 in
  let y2_val := x2_sq / 4 in
  (sqrt lambda + 1 / sqrt lambda) ^ 2

noncomputable def FM_len : ℝ := sqrt (lambda + 1 / lambda + 2)

noncomputable def area_ABM : ℝ :=
  area_of_triangle (AB_len lambda) (FM_len lambda)

theorem min_area_ABM : ∀ λ > 0, (sqrt λ + 1 / sqrt λ) ^ 3 / 2 ≥ 4 :=
sorry

end min_area_ABM_l317_317120


namespace find_number_of_girls_l317_317509

-- Define the ratio of boys to girls as 8:4.
def ratio_boys_to_girls : ℕ × ℕ := (8, 4)

-- Define the total number of students.
def total_students : ℕ := 600

-- Define what it means for the number of girls given a ratio and total students.
def number_of_girls (ratio : ℕ × ℕ) (total : ℕ) : ℕ :=
  let total_parts := (ratio.1 + ratio.2)
  let part_value := total / total_parts
  ratio.2 * part_value

-- State the goal to prove the number of girls is 200 given the conditions.
theorem find_number_of_girls :
  number_of_girls ratio_boys_to_girls total_students = 200 :=
sorry

end find_number_of_girls_l317_317509


namespace quadratic_binomial_square_l317_317317

theorem quadratic_binomial_square (k : ℝ) : (∃ b : ℝ, x^2 - 18 * x + k = (x - b)^2) → k = 81 :=
begin
  sorry
end

end quadratic_binomial_square_l317_317317


namespace log_x_125_l317_317482

theorem log_x_125 {x : ℝ} (h : log 8 (5 * x) = 3) : 
  log x 125 = 3 * (1 / (9 * log 5 2 - 1)) :=
sorry

end log_x_125_l317_317482


namespace terminating_decimal_count_l317_317834

theorem terminating_decimal_count:
  (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 500), ∃ k, k = 23 ∧ ∀ x, (1 ≤ x ∧ x ≤ 500 ∧ terminating_decimal (x / 2100)) ↔ (x % 21 = 0) := 
sorry

end terminating_decimal_count_l317_317834


namespace identify_solutions_l317_317704

-- Define the solutions
def solution_1 := "CaCl₂"
def solution_2 := "NaCl"
def solution_3 := "AlCl₃"
def solution_4 := "NaOH"
def solution_5 := "AgNO₃"
def solution_6 := "CuSO₄"
def solution_7 := "NH₄OH"
def solution_8 := "BaCl₂"
def solution_9 := "HCl"

-- Define the conditions
def condition1 := 
  (forms_precipitate_with solution_5 [solution_1, solution_2, solution_3, solution_4, solution_6, solution_7, solution_8, solution_9]) ∧
  (forms_precipitate_with solution_5 [solution_1, solution_2, solution_3, solution_4, solution_7, solution_8, solution_9] on_strong_dilution) ∧
  (dissolves_in_excess_of solution_5 solution_7)

def condition2 := 
  (forms_precipitate_with solution_3 [solution_4, solution_7]) ∧
  (dissolves_in_excess_of solution_3 solution_4)

def condition3 := 
  (forms_precipitate_with solution_6 [solution_1, solution_4, solution_7, solution_8]) ∧
  (forms_precipitate_with solution_6 [solution_4, solution_7, solution_8] on_strong_dilution)

def condition4 := 
  (solubility_decreases_with_heating solution_6 solution_1)

def condition5 := 
  (litmus_changes_color solution_9) ∧
  (color_reverts_with solution_9 solution_7)

theorem identify_solutions : 
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 → 
  solution_1 = "CaCl₂" ∧ 
  solution_2 = "NaCl" ∧ 
  solution_3 = "AlCl₃" ∧ 
  solution_4 = "NaOH" ∧ 
  solution_5 = "AgNO₃" ∧ 
  solution_6 = "CuSO₄" ∧ 
  solution_7 = "NH₄OH" ∧ 
  solution_8 = "BaCl₂" ∧ 
  solution_9 = "HCl" := by 
    sorry

end identify_solutions_l317_317704


namespace find_f_value_l317_317101

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ set.Icc (-3) (-2) then (x + 2) ^ 2 else 0 -- This defines f only on the given interval

lemma symmetric_about_x_neg2 (x : ℝ) : f (-4 - x) = f x := 
sorry -- Symmetry about the line x = -2 (to be proved)

lemma periodicity (x : ℝ) : f (x + 2) = f x :=
sorry -- Periodicity with period 2 (to be proved)

theorem find_f_value : f (5 / 2) = 1 / 4 :=
by
  have h1 : f (5 / 2) = f (1 / 2), 
  { -- Because of the symmetry and periodicity: f is symmetric about x = -2
    rw ←symmetric_about_x_neg2 (5 / 2),
    exact periodicity (1 / 2) },
  have h2 : f (1 / 2) = f (-3 / 2),
  { -- Because of the periodicity: f(x) = f(x mod 2)
    exact periodicity (1 / 2) },
  have h3 : f (-3 / 2) = (1 / 2 + 2) ^ 2,
  { -- Given the definition of f on the interval [-3, -2]
    rw [←if_pos (show -3 / 2 ∈ set.Icc (-3) (-2), by norm_num)],
    rw f,
    exact if_pos (show -3 / 2 ∈ set.Icc (-3) (-2), by norm_num) },
  -- Show that it is equal to 1 / 4
  calc
    f (5 / 2) = f (1 / 2) : by rw h1
    ... = f (-3 / 2) : by rw h2
    ... = (1 / 2 + 2) ^ 2 : by rw h3
    ... = 1 / 4 : by norm_num
  sorry

end find_f_value_l317_317101


namespace comp_function_value_l317_317111

def f (x : ℝ) : ℝ := 
  if x < 2 then 
    -2 * x - 3 
  else 
    2^(-x)

theorem comp_function_value : f (f (-3)) = 1 / 8 := 
by 
  sorry

end comp_function_value_l317_317111


namespace probability_hugo_roll_7_given_win_l317_317937

/-- Definition of the game conditions --/
def players := 5
def sides_first_round := 8
def sided_tie_round := 10
def hugo_first_roll := 7
def hugo_wins_event := true /- in the context of probabilities -/

/-- The probability that Hugo's first roll was 7 given that he won the game --/
theorem probability_hugo_roll_7_given_win :
  ∀ (Hugo_roll : ℕ, wins : bool), 
    (Hugo_roll = hugo_first_roll ∧ wins = hugo_wins_event) →
    (1/5 : ℚ) → 
    P(Hugo_roll = 7 | wins = true) = 961 / 2048 :=
sorry

end probability_hugo_roll_7_given_win_l317_317937


namespace find_a2_b2_c2_l317_317923

-- Define the roots, sum of the roots, sum of the product of the roots taken two at a time, and product of the roots
variables {a b c : ℝ}
variable (h_roots : a = b ∧ b = c)
variable (h_sum : a + b + c = 12)
variable (h_sum_products : a * b + b * c + a * c = 47)
variable (h_product : a * b * c = 30)

-- State the theorem
theorem find_a2_b2_c2 : (a^2 + b^2 + c^2) = 50 :=
by {
  sorry
}

end find_a2_b2_c2_l317_317923


namespace count_num_valid_pairs_l317_317359

namespace MathProof

noncomputable def num_valid_pairs : ℕ :=
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ b > a ∧ a * b = 3 * (a - 4) * (b - 4)

theorem count_num_valid_pairs : num_valid_pairs = 4 := by sorry

end MathProof

end count_num_valid_pairs_l317_317359


namespace sum_final_term_and_series_l317_317389

-- Definition of arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ → ℝ
| 0     := 15
| (k+1) := arithmetic_sequence k + 0.2

-- Definition of geometric sequence
def geometric_sequence (n : ℕ) : ℕ → ℝ
| 0     := 15
| (k+1) := geometric_sequence k * 2

-- Sum of arithmetic sequence up to n
def sum_arithmetic_sequence (n : ℕ) : ℝ :=
  (n / 2) * (2 * 15 + (n - 1) * 0.2)

-- Sum of geometric sequence up to n
def sum_geometric_sequence (n : ℕ) : ℝ :=
  15 * ((2^n - 1) / (2 - 1))

-- Main theorem to prove
theorem sum_final_term_and_series :
  let n := 101 in
  sum_arithmetic_sequence (n: ℕ) + sum_geometric_sequence (n: ℕ) = 15 * (2^101 - 1) + 2525 := 
by sorry

end sum_final_term_and_series_l317_317389


namespace problem_l317_317787

noncomputable def sum_A : ℝ := ∑' n, (n^2 : ℝ) / 3^n
noncomputable def sum_B : ℝ := ∑' n, (n : ℝ) / 3^n
noncomputable def sum_C : ℝ := ∑' n, 1 / 3^n

theorem problem : (∑' n, (4 * (n^2 : ℝ) - 2 * n + 1) / 3^n) = 5 :=
by
  let A := sum_A
  let B := sum_B
  let C := sum_C
  have hA : A = 3 / 2 := by sorry
  have hB : B = 3 / 4 := by sorry
  have hC : C = 1 / 2 := by sorry
  calc
    (∑' n, (4 * (n^2 : ℝ) - 2 * n + 1) / 3^n)
        = 4 * A - 2 * B + C : by sorry
    ... = 4 * (3 / 2) - 2 * (3 / 4) + (1 / 2) : by rw [hA, hB, hC]
    ... = 5 : by norm_num

end problem_l317_317787


namespace sisters_gift_amount_l317_317208

def jacoby_needs_total : ℕ := 5000
def jacoby_job_rate : ℕ := 20
def jacoby_job_hours : ℕ := 10
def jacoby_cookie_price : ℕ := 4
def jacoby_cookies_sold : ℕ := 24
def lottery_ticket_cost : ℕ := 10
def lottery_winning : ℕ := 500
def jacoby_needs_more : ℕ := 3214

theorem sisters_gift_amount :
  let job_earnings := jacoby_job_rate * jacoby_job_hours 
  let cookie_earnings := jacoby_cookie_price * jacoby_cookies_sold 
  let total_before_lottery := job_earnings + cookie_earnings
  let total_after_lottery_ticket := total_before_lottery - lottery_ticket_cost 
  let total_with_lottery_win := total_after_lottery_ticket + lottery_winning
  let total_with_sisters := jacoby_needs_total - jacoby_needs_more in
  total_with_sisters - total_with_lottery_win = 1000 :=
by {
  let job_earnings := jacoby_job_rate * jacoby_job_hours;
  let cookie_earnings := jacoby_cookie_price * jacoby_cookies_sold;
  let total_before_lottery := job_earnings + cookie_earnings;
  let total_after_lottery_ticket := total_before_lottery - lottery_ticket_cost;
  let total_with_lottery_win := total_after_lottery_ticket + lottery_winning;
  let total_with_sisters := jacoby_needs_total - jacoby_needs_more;
  calc
    total_with_sisters - total_with_lottery_win
        = 1000 : by linarith
}

end sisters_gift_amount_l317_317208


namespace problem1_union_problem2_intersection_problem3_subset_l317_317906

def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x + m^2 - 4 ≤ 0}

theorem problem1_union (m : ℝ) (hm : m = 2) : A ∪ B m = {x | -1 ≤ x ∧ x ≤ 4} :=
sorry

theorem problem2_intersection (m : ℝ) (h : A ∩ B m = {x | 1 ≤ x ∧ x ≤ 3}) : m = 3 :=
sorry

theorem problem3_subset (m : ℝ) (h : A ⊆ {x | ¬ (x ∈ B m)}) : m > 5 ∨ m < -3 :=
sorry

end problem1_union_problem2_intersection_problem3_subset_l317_317906


namespace mode_of_data_set_is_4_l317_317636

-- Definition of the data set
def data_set : List ℕ := [5, 4, 4, 3, 6, 2]

-- Mode calculation predicate
def mode (l : List ℕ) (m : ℕ) : Prop :=
  ∀ n ∈ l, count l n ≤ count l m

theorem mode_of_data_set_is_4 : mode data_set 4 :=
by
  -- Add your proof directives here if necessary, otherwise use "sorry" for skipping the proof.
  sorry

end mode_of_data_set_is_4_l317_317636


namespace old_geometry_book_pages_l317_317353

def old_pages := 340
def new_pages := 450
def deluxe_pages := 915

theorem old_geometry_book_pages : 
  (new_pages = 2 * old_pages - 230) ∧ 
  (deluxe_pages = new_pages + old_pages + 125) ∧ 
  (deluxe_pages ≥ old_pages + old_pages / 10) 
  → old_pages = 340 := by
  sorry

end old_geometry_book_pages_l317_317353


namespace factorial_expr_value_l317_317690

theorem factorial_expr_value : (13! - 12!) / 10! = 1584 := 
sorry

end factorial_expr_value_l317_317690


namespace a2_eq_1_l317_317849

-- Define the geometric sequence and the conditions
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Given conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom a1_eq_2 : a 1 = 2
axiom condition1 : geometric_sequence a q
axiom condition2 : 16 * a 3 * a 5 = 8 * a 4 - 1

-- Prove that a_2 = 1
theorem a2_eq_1 : a 2 = 1 :=
by
  -- This is where the proof would go
  sorry

end a2_eq_1_l317_317849


namespace range_of_k_find_roots_when_k_is_1_l317_317640

-- Given conditions
def quadratic_equation (k : ℝ) (x : ℝ) : Prop := x^2 - 2*x + 2*k - 3 = 0
def has_distinct_real_roots (k : ℝ) : Prop := ( ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic_equation k x1 ∧ quadratic_equation k x2)

-- Proving the range for k
theorem range_of_k (k : ℝ) : has_distinct_real_roots k → k < 2 :=
begin
  sorry  -- proof to be provided
end

-- Given k is a positive integer and k < 2, find the roots
theorem find_roots_when_k_is_1 : 
  ∀ (k : ℕ), (has_distinct_real_roots k ∧ k < 2 ∧ k > 0) → 
  ( ∃ (x1 x2 : ℝ), x1 = 1 + Real.sqrt 2 ∧ x2 = 1 - Real.sqrt 2 ∧ quadratic_equation k x1 ∧ quadratic_equation k x2) :=
begin
  sorry  -- proof to be provided
end

end range_of_k_find_roots_when_k_is_1_l317_317640


namespace fifty_third_card_is_A_s_l317_317041

def sequence_position (n : ℕ) : String :=
  let cycle_length := 26
  let pos_in_cycle := (n - 1) % cycle_length + 1
  if pos_in_cycle <= 13 then
    "A_s"
  else
    "A_h"

theorem fifty_third_card_is_A_s : sequence_position 53 = "A_s" := by
  sorry  -- proof placeholder

end fifty_third_card_is_A_s_l317_317041


namespace find_a_b_l317_317118

noncomputable def f (x : ℝ) : ℝ :=
real.abs (real.log (x + 1))

theorem find_a_b (a b : ℝ) (ha : a < b)
  (h1 : f a = f (-((b + 1) / (b + 2))))
  (h2 : f (10 * a + 6 * b + 21) = 4 * real.log 2) :
  a = -2 / 5 ∧ b = -1 / 3 :=
sorry

end find_a_b_l317_317118


namespace number_of_dogs_l317_317810

theorem number_of_dogs (h1 : 24 = 2 * 2 + 4 * n) : n = 5 :=
by
  sorry

end number_of_dogs_l317_317810


namespace inner_rectangle_length_l317_317750

def inner_rect_width : ℕ := 2

def second_rect_area (x : ℕ) : ℕ := 6 * (x + 4)

def largest_rect_area (x : ℕ) : ℕ := 10 * (x + 8)

def shaded_area_1 (x : ℕ) : ℕ := second_rect_area x - 2 * x

def shaded_area_2 (x : ℕ) : ℕ := largest_rect_area x - second_rect_area x

def in_arithmetic_progression (a b c : ℕ) : Prop := b - a = c - b

theorem inner_rectangle_length (x : ℕ) :
  in_arithmetic_progression (2 * x) (shaded_area_1 x) (shaded_area_2 x) → x = 4 := by
  intros
  sorry

end inner_rectangle_length_l317_317750


namespace range_of_a_l317_317925

noncomputable def f (x : ℝ) : ℝ := x^2 - (1 / 2) * Real.log x + 1

def f_deriv (x : ℝ) : ℝ := 2 * x - (1 / (2 * x))

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ (set.Ioo 0 2 ∪ set.Ioo (2.5 : ℝ) +∞), f_deriv x < 0) 
  → (∀ x ∈ set.Ioo (2 : ℝ) 2.5, f_deriv x > 0)
  → ¬ monotone_on f (set.Ioo (a - 2 : ℝ) (a + 2 : ℝ)) 
  → 2 ≤ a ∧ a < 2.5 :=
sorry

end range_of_a_l317_317925


namespace probability_perm_divisible_by_8_l317_317498

theorem probability_perm_divisible_by_8 :
  let digits := [1, 2, 3, 4, 5, 6, 7, 8],
      total_permutations := (Nat.factorial 8),
      valid_permutations := 42
  in (valid_permutations : ℚ) / (total_permutations : ℚ) = 1/8 :=
by
  let digits := [1, 2, 3, 4, 5, 6, 7, 8]
  let total_permutations := (Nat.factorial 8)
  let valid_permutations := 42
  have h1 : (valid_permutations : ℚ) = 42 := rfl
  have h2 : (total_permutations : ℚ) = 40320 := by norm_num
  show (valid_permutations : ℚ) / (total_permutations : ℚ) = 1/8
  calc 
    (valid_permutations : ℚ) / (total_permutations : ℚ) 
    _ = 42 / 40320 : by rw [←h1, ←h2]
    ... = 1 / 8 : by norm_num

end probability_perm_divisible_by_8_l317_317498


namespace identify_symmetric_star_l317_317277

-- Defining the conditions as structures or constants
structure Tablecloth (α : Type) :=
  (is_silk_patches : α) 
  (is_sewn_together : α)
  (triangular_patches : α)
  (symmetrical_star_fits : α)

-- Given conditions assumed to hold
constant silk_patches : Prop
constant sewn_together : Prop
constant triangular_patches : Prop
constant symmetrical_star_fits : Prop

theorem identify_symmetric_star (cloth: Tablecloth Prop) :
  silk_patches ∧ sewn_together ∧ triangular_patches ∧ symmetrical_star_fits →
  ∃ star, star = "Symmetric star pattern identified" := 
begin
  intros h,
  sorry
end

end identify_symmetric_star_l317_317277


namespace smallest_n_value_l317_317952

open Real

theorem smallest_n_value {ABC : Triangle} (AB BC CA : ℝ) (hAB : AB = 52) 
  (hBC : BC = 34) (hCA : CA = 50) (n : ℕ) :
  (∃ (split_points : Finset (SegmentPoint (Subsegment BC) n)),
      (∃ (D : SegmentPoint (Subsegment BC) spl), True) ∧
      (∃ (M : SegmentPoint (Subsegment BC) (Fin.mk (BC/2) (by linarith))), True) ∧
      (∃ (X : SegmentPoint (Subsegment BC) (Fin.mk ((ℝ.divsup (ℝ.add (51/51) 26)) 25) (by linarith))), True)
  ) → n = 102 :=
begin
  sorry
end

end smallest_n_value_l317_317952


namespace fully_transport_one_team_l317_317945

theorem fully_transport_one_team 
  (m t : ℕ) 
  (teams : Fin m → Fin 11 → Prop)
  (flights : Fin 10 → Fin t → {p // ∃ (i : Fin m) (j : Fin 11), teams i j})
  (helicopter : {p // ∃ (i : Fin m) (j : Fin 11), teams i j})
  (h_flights_helicopter_count : 10 * t + 1 >= 11 * m) :
  ∃ (i : Fin m), ∀ (j : Fin 11), teams i j :=
by
  sorry

end fully_transport_one_team_l317_317945


namespace angle_between_vectors_45_degrees_parallelogram_area_half_l317_317910

-- Define the vectors and their properties
variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
def conditions : Prop :=
  (∥a∥ = 1) ∧ 
  (inner_product_space.inner a b = 1/2) ∧ 
  (inner_product_space.inner (a - b) (a + b) = 1/2)

-- Prove the angle between the vectors
theorem angle_between_vectors_45_degrees (ha : ∥a∥ = 1) (hab : inner_product_space.inner a b = 1/2) (hab' : inner_product_space.inner (a - b) (a + b) = 1/2) :
  real.angle.cos (inner_product_space.angle a b) = (real.sqrt 2) / 2 :=
sorry

-- Prove the area of the parallelogram
theorem parallelogram_area_half (ha : ∥a∥ = 1) (hab : inner_product_space.inner a b = 1/2) (hab' : inner_product_space.inner (a - b) (a + b) = 1/2) :
  let area := ∥a∥ * ∥b∥ * real.sin(real.angle.to_real_angle(real.angle a b)) in
  area / (real.sqrt 2) = 1 / 2 :=
sorry

end angle_between_vectors_45_degrees_parallelogram_area_half_l317_317910


namespace johns_overall_average_speed_l317_317542

theorem johns_overall_average_speed :
  (let time_driving := 45 / 60.0 in
   let speed_driving := 20 in
   let distance_driving := speed_driving * time_driving in
   let time_riding := 30 / 60.0 in
   let speed_riding := 6 in
   let distance_riding := speed_riding * time_riding in
   let total_distance := distance_driving + distance_riding in
   let total_time := time_driving + time_riding in
   let average_speed := total_distance / total_time in
   average_speed) = 14.4 :=
by
  sorry

end johns_overall_average_speed_l317_317542


namespace monotonically_decreasing_a_ge_3_l317_317926

theorem monotonically_decreasing_a_ge_3 (a : ℝ) :
  (∀ x ∈ Ioo (0 : ℝ) 2, deriv (λ x, x^3 - a*x^2 + 1) x ≤ 0) → a ≥ 3 :=
sorry

end monotonically_decreasing_a_ge_3_l317_317926


namespace sum_combinatoric_problem_l317_317006

theorem sum_combinatoric_problem :
  (1 / 2 ^ 2000) * (∑ n in Finset.range 1001, (-3 : ℤ)^n * Nat.choose 2000 (2 * n)) = -1 / 2 :=
by
  sorry

end sum_combinatoric_problem_l317_317006


namespace eighty_five_squared_l317_317009

theorem eighty_five_squared : 85^2 = 7225 := by
  sorry

end eighty_five_squared_l317_317009


namespace part_a_part_b_l317_317327

-- Part (a): Prove it is impossible for all numbers to become zeros if there are 13 of them
theorem part_a (n : ℕ) (h : n = 13) :
  ∀ (s : zmod n → bool),
    (∀ i, s i = if s (i - 1) = s (i + 1) then ff else tt) → 
    ¬(∀ i, s i = ff) :=
by sorry

-- Part (b): Prove it is impossible for all numbers to become ones if there are 14 of them
theorem part_b (n : ℕ) (h : n = 14) :
  ∀ (s : zmod n → bool),
    (∀ i, s i = if s (i - 1) = s (i + 1) then ff else tt) → 
    ¬(∀ i, s i = tt) :=
by sorry

end part_a_part_b_l317_317327


namespace simplification_evaluation_l317_317257

-- Define the variables x and y
def x : ℕ := 2
def y : ℕ := 3

-- Define the expression
def expr := 5 * (3 * x^2 * y - x * y^2) - (x * y^2 + 3 * x^2 * y)

-- Lean 4 statement to prove the equivalence
theorem simplification_evaluation : expr = 36 :=
by
  -- Place the proof here when needed
  sorry

end simplification_evaluation_l317_317257


namespace inequality_proof_l317_317582

theorem inequality_proof
  (k : ℕ)
  (a : Fin k → ℝ) (b : Fin k → ℝ)
  (ha : ∀ i, 0 < a i) (hb : ∀ i, 0 < b i) :
  let A := ∑ i in Finset.finRange k, a i
      B := ∑ i in Finset.finRange k, b i
  in ( (∑ i in Finset.finRange k, (a i * b i) / (a i * B + b i * A) - 1) ^ 2
       ≥ (∑ i in Finset.finRange k, (a i ^ 2) / (a i * B + b i * A))
       * (∑ i in Finset.finRange k, (b i ^ 2) / (a i * B + b i * A)) ) := 
sorry

end inequality_proof_l317_317582


namespace find_triangle_angles_l317_317618

-- Define the condition where the sum of interior angles of a triangle is 180 degrees
axiom sum_of_triangle_angles (α β γ : ℝ) : α + β + γ = 180

-- Given the angles formed by the angle bisectors of the triangle
-- We denote the ratios as 37:41:42 which should be scaled to sum to 180 degrees
axiom angle_bisectors_ratios (A B C : ℝ) : A / 37 = B / 41 ∧ A / 37 = C / 42

-- Define the relationship between the original angles and the bisected angles
axiom bisected_angle_relation (α β γ A B C : ℝ) :
  A = α / 2 + β / 2 ∧ B = α / 2 + γ / 2 ∧ C = β / 2 + γ / 2

-- The theorem to state the final proof problem
theorem find_triangle_angles (α β γ : ℝ) (A B C : ℝ) 
   (h1 : sum_of_triangle_angles α β γ)
   (h2 : angle_bisectors_ratios A B C)
   (h3 : bisected_angle_relation α β γ A B C) :
   α = 72 ∧ β = 66 ∧ γ = 42 :=
sorry -- Proof to be provided

end find_triangle_angles_l317_317618


namespace joan_paid_230_l317_317961

theorem joan_paid_230 (J K : ℝ) (h1 : J + K = 600) (h2 : 2 * J = K + 90) : J = 230 :=
sorry

end joan_paid_230_l317_317961


namespace sum_dot_products_of_cube_vectors_l317_317348

-- Define the side length of the cube based on the given sphere radius
def cube_side_length (r : ℝ) : ℝ := 2 * r / Real.sqrt 3

-- Define vectors from one face center to other face centers and vertices
structure Vec3 := (x : ℝ) (y : ℝ) (z : ℝ)

def vectors_from_face_center (a : ℝ) : List Vec3 :=
  [⟨a / 2, 0, 0⟩, ⟨-a / 2, 0, 0⟩, ⟨0, a / 2, 0⟩, ⟨0, -a / 2, 0⟩, ⟨0, 0, a / 2⟩, ⟨0, 0, -a / 2⟩,
   ⟨a / 2, a / 2, a / 2⟩, ⟨-a / 2, a / 2, a / 2⟩, ⟨a / 2, -a / 2, a / 2⟩, ⟨a / 2, a / 2, -a / 2⟩,
   ⟨-a / 2, -a / 2, a / 2⟩, ⟨-a / 2, a / 2, -a / 2⟩, ⟨a / 2, -a / 2, -a / 2⟩, ⟨-a / 2, -a / 2, -a / 2⟩]

-- Define the dot product of two vectors
def dot_product (v1 v2 : Vec3) : ℝ := v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Sum of dot products of all distinct pairs of vectors
noncomputable def sum_of_dot_products (a : ℝ) : ℝ :=
  let vectors := vectors_from_face_center a
  vectors.pairs.sum (fun (p : Vec3 × Vec3) => dot_product p.1 p.2)

-- The main theorem to be proven
theorem sum_dot_products_of_cube_vectors (r : ℝ) (r_pos : 0 < r) :
  sum_of_dot_products (cube_side_length r) = 76 :=
by
  sorry

end sum_dot_products_of_cube_vectors_l317_317348


namespace amy_points_per_treasure_l317_317767

theorem amy_points_per_treasure (treasures_first_level treasures_second_level total_score : ℕ) (h1 : treasures_first_level = 6) (h2 : treasures_second_level = 2) (h3 : total_score = 32) :
  total_score / (treasures_first_level + treasures_second_level) = 4 := by
  sorry

end amy_points_per_treasure_l317_317767


namespace sum_a_n_eq_2014_l317_317455

def f (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n : ℤ)^2 else - (n : ℤ)^2

def a (n : ℕ) : ℤ :=
  f n + f (n + 1)

theorem sum_a_n_eq_2014 : (Finset.range 2014).sum a = 2014 :=
by
  sorry

end sum_a_n_eq_2014_l317_317455


namespace measure_of_angle_B_l317_317338

-- The given conditions as a definition in Lean 4
def is_equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def on_segment (P A B : Point) : Prop :=
  -- Define what it means for a point to lie on a given segment
  sorry

def equilateral_segment_partition (A B P Q : Point) : Prop :=
  on_segment P A B ∧ on_segment Q B C ∧ dist A P = dist P Q ∧ dist P Q = dist Q B

-- The theorem we need to prove
theorem measure_of_angle_B {A B C P Q : Point} :
  is_equilateral_triangle A B C →
  equilateral_segment_partition A B P Q →
  angle A B C = 30 :=
begin
  sorry
end

end measure_of_angle_B_l317_317338


namespace prism_faces_l317_317741

theorem prism_faces (E L : ℕ) (hE : E = 21) (hFormula : E = 3 * L) : 2 + L = 9 :=
by
  rw [hE, hFormula]
  have hL : L = 7 := sorry 
  rw [hL]
  rfl

end prism_faces_l317_317741


namespace geometry_problem_l317_317086

noncomputable def line (a b c : ℝ) : ℝ × ℝ → Prop := 
λ p, a * p.1 + b * p.2 + c = 0

theorem geometry_problem (A B C D : ℝ × ℝ)
  (H_iso : ∀ x, (2*x + (A.snd + 1 - 13)*x - 4 = 0) ∨ (x + 2 * (A.snd - 5/2)*x - 5 = 0))
  (H_AB : line 2 1 -4 A)
  (H_AD : line 1 (-1) 1 D)
  (H_D : D = (4, 5))
  (H_mid_MD : A.fst = D.fst ∨ C = (2 * D.fst - B.fst, 2 * D.snd - B.snd)) :
  (line 1 1 -9 B ∧ B = (-5, 14)) ∧
  (line 1 1 -9 C ∧ C = (13, -4)) ∧
  line 1 2 -5 C := 
by
  sorry

end geometry_problem_l317_317086


namespace tan_pi_plus_a6_l317_317477

open Int Real

variable {a : ℕ → ℝ} (S : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in Finset.range n, a (i + 1)

theorem tan_pi_plus_a6 {a : ℕ → ℝ} (S : ℕ → ℝ)
  (ha : is_arithmetic_sequence a)
  (hS : sum_of_first_n_terms a S)
  (hS11 : S 11 = 22 * π / 3) :
  tan (π + a 6) = -√3 :=
by sorry

end tan_pi_plus_a6_l317_317477


namespace value_of_expression_l317_317488

theorem value_of_expression (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by {
   rw h,
   norm_num,
   sorry
}

end value_of_expression_l317_317488


namespace prob_enter_A_and_exit_F_l317_317770

-- Define the problem description
def entrances : ℕ := 2
def exits : ℕ := 3

-- Define the probabilities
def prob_enter_A : ℚ := 1 / entrances
def prob_exit_F : ℚ := 1 / exits

-- Statement that encapsulates the proof problem
theorem prob_enter_A_and_exit_F : prob_enter_A * prob_exit_F = 1 / 6 := 
by sorry

end prob_enter_A_and_exit_F_l317_317770


namespace smallest_number_of_locks_and_keys_l317_317291

open Finset Nat

-- Definitions based on conditions
def committee : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

def can_open_safe (members : Finset ℕ) : Prop :=
  ∀ (subset : Finset ℕ), subset.card = 6 → members ⊆ subset

def cannot_open_safe (members : Finset ℕ) : Prop :=
  ∃ (subset : Finset ℕ), subset.card = 5 ∧ members ⊆ subset

-- Proof statement
theorem smallest_number_of_locks_and_keys :
  ∃ (locks keys : ℕ), locks = 462 ∧ keys = 2772 ∧
  (∀ (subset : Finset ℕ), subset.card = 6 → can_open_safe subset) ∧
  (∀ (subset : Finset ℕ), subset.card = 5 → ¬can_open_safe subset) :=
sorry

end smallest_number_of_locks_and_keys_l317_317291


namespace total_cost_of_aquarium_l317_317777

variable (original_price discount_rate sales_tax_rate : ℝ)
variable (original_cost : original_price = 120)
variable (discount : discount_rate = 0.5)
variable (tax : sales_tax_rate = 0.05)

theorem total_cost_of_aquarium : 
  (original_price * (1 - discount_rate) * (1 + sales_tax_rate) = 63) :=
by
  rw [original_cost, discount, tax]
  sorry

end total_cost_of_aquarium_l317_317777


namespace resistor_value_l317_317174

theorem resistor_value (x r y : ℝ) (hx : x = 3) (hr : r = 1.875) 
  (h_eq : 1/r = 1/x + 1/y) : y = 5 :=
by
  have : 1/1.875 = 0.5333333333333333 := rfl
  have : 1/3 = 0.3333333333333333 := rfl
  have : 0.5333333333333333 - 0.3333333333333333 = 0.2 := by norm_num
  have : 1 / y = 0.2 := by norm_num
  have : y = 1 / 0.2 := sorry
  have : y = 5 := by sorry
  exact this

end resistor_value_l317_317174


namespace coterminal_angle_l317_317951

theorem coterminal_angle :
  ∃ θ : ℝ, θ ∈ set.Ico 0 360 ∧ (∃ k : ℤ, θ = -510 + k * 360) :=
sorry

end coterminal_angle_l317_317951


namespace probability_all_dice_same_l317_317674

/--
Given four eight-sided dice, each numbered from 1 to 8 and each die landing independently,
prove that the probability of all four dice showing the same number is 1/512.
-/
theorem probability_all_dice_same :
  let n := 8 in       -- Number of sides on each dice
  let total_outcomes := n * n * n * n in  -- Total possible outcomes for four dice
  let favorable_outcomes := n in          -- Favorable outcomes (one same number for all dice)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 512 :=
by
  sorry

end probability_all_dice_same_l317_317674


namespace corrected_mean_l317_317330

/-- The original mean of 20 observations is 36, an observation of 25 was wrongly recorded as 40.
    The correct mean is 35.25. -/
theorem corrected_mean 
  (Mean : ℝ)
  (Observations : ℕ)
  (IncorrectObservation : ℝ)
  (CorrectObservation : ℝ)
  (h1 : Mean = 36)
  (h2 : Observations = 20)
  (h3 : IncorrectObservation = 40)
  (h4 : CorrectObservation = 25) :
  (Mean * Observations - (IncorrectObservation - CorrectObservation)) / Observations = 35.25 :=
sorry

end corrected_mean_l317_317330


namespace rearrange_possible_l317_317204

theorem rearrange_possible (n : ℕ) (h : n = 25 ∨ n = 1000) :
  ∃ (f : ℕ → ℕ), (∀ i < n, f i + 1 < n → (f (i + 1) - f i = 3 ∨ f (i + 1) - f i = 5)) :=
  sorry

end rearrange_possible_l317_317204


namespace find_side_length_l317_317931

theorem find_side_length (a : ℝ) (b : ℝ) (A B : ℝ) (ha : a = 4) (hA : A = 45) (hB : B = 60) :
    b = 2 * Real.sqrt 6 := by
  sorry

end find_side_length_l317_317931


namespace a_2016_eq_neg_1_l317_317905

noncomputable def a : ℕ → ℝ
| 1       := 2
| (n + 1) := 1 - (1 / a n)

theorem a_2016_eq_neg_1 : a 2016 = -1 := by
  sorry

end a_2016_eq_neg_1_l317_317905


namespace solve_system_of_equations_l317_317611

def sys_eq1 (x y : ℝ) : Prop := 6 * (1 - x) ^ 2 = 1 / y
def sys_eq2 (x y : ℝ) : Prop := 6 * (1 - y) ^ 2 = 1 / x

theorem solve_system_of_equations (x y : ℝ) :
  sys_eq1 x y ∧ sys_eq2 x y ↔
  ((x = 3 / 2 ∧ y = 2 / 3) ∨
   (x = 2 / 3 ∧ y = 3 / 2) ∨
   (x = 1 / 6 * (4 + 2 ^ (2 / 3) + 2 ^ (4 / 3)) ∧ y = 1 / 6 * (4 + 2 ^ (2 / 3) + 2 ^ (4 / 3)))) :=
sorry

end solve_system_of_equations_l317_317611


namespace smallest_possible_degree_of_polynomial_l317_317614

/--
Given that the following numbers are roots of the same nonzero polynomial with rational coefficients:
2 - 3*sqrt(3), -2 - 3*sqrt(3), 2 + sqrt(11), 2 - sqrt(11),
the smallest possible degree of such a polynomial is 6.
-/
theorem smallest_possible_degree_of_polynomial :
  ∃ (p : Polynomial ℚ), p ≠ 0 ∧
  p.eval (2 - 3*Real.sqrt 3) = 0 ∧ p.eval (-2 - 3*Real.sqrt 3) = 0 ∧
  p.eval (2 + Real.sqrt 11) = 0 ∧ p.eval (2 - Real.sqrt 11) = 0 ∧
  Polynomial.degree p = 6 :=
by
  sorry

end smallest_possible_degree_of_polynomial_l317_317614


namespace minimal_hall_number_l317_317293

-- Definitions of the conditions
def friends (G : Type*) (t : G → ℤ) (X' X'' : G) : Prop := X' ≠ X'' ∧ t X' ≠ t X''

def consecutive_scores (G : Type*) (t : G → ℤ) (X : G) (friends : G → G → Prop) : Prop :=
  ∃ (Ys : Finset ℤ), (∀ Y, friends X Y → Y ∈ Ys) ∧ (Finset.card Ys = Finset.card (Finset.range (Finset.sup Ys - Finset.inf Ys + 1)))

-- The main problem statement
theorem minimal_hall_number (G : Type*) [Fintype G] (t : G → ℤ) (friends : G → G → Prop)
  (h_friends : ∀ (X' X'' : G), friends X' X'' → t X' ≠ t X'')
  (h_consecutive : ∀ X : G, consecutive_scores G t X friends)
  : ∃ (partition : G → Bool), ∀ (X X' : G), friends X X' → partition X ≠ partition X' :=
sorry

end minimal_hall_number_l317_317293


namespace no_such_function_l317_317207

theorem no_such_function : ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2 := 
sorry

end no_such_function_l317_317207


namespace tablecloth_radius_l317_317064

theorem tablecloth_radius (diameter : ℝ) (h : diameter = 10) : diameter / 2 = 5 :=
by {
  -- Outline the proof structure to ensure the statement is correct
  sorry
}

end tablecloth_radius_l317_317064


namespace multiple_of_shorter_piece_l317_317716

theorem multiple_of_shorter_piece :
  ∃ (m : ℕ), 
  (35 + (m * 35 + 15) = 120) ∧
  (m = 2) :=
by
  sorry

end multiple_of_shorter_piece_l317_317716


namespace range_of_m_for_hyperbola_l317_317622

theorem range_of_m_for_hyperbola (m : ℝ) :
  (∃ (x y : ℝ), (m+2) ≠ 0 ∧ (m-2) ≠ 0 ∧ (x^2)/(m+2) + (y^2)/(m-2) = 1) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end range_of_m_for_hyperbola_l317_317622


namespace find_Susan_cats_l317_317615

variable (S : ℕ)       -- The number of cats Susan has initially.
def Bob_cats : ℕ := 3  -- Bob initially has 3 cats.

-- Susan gives 4 of her cats to Bob.
def Susan_after_giving : ℕ := S - 4
def Bob_after_receiving : ℕ := Bob_cats + 4

-- After giving 4 cats to Bob, Susan has 14 more cats than Bob.
def condition : Prop := Susan_after_giving = Bob_after_receiving + 14

theorem find_Susan_cats (h : condition S) : S = 25 := 
by 
  sorry

end find_Susan_cats_l317_317615


namespace Nicole_cards_l317_317233

variables (N : ℕ)

-- Conditions from step A
def Cindy_collected (N : ℕ) : ℕ := 2 * N
def Nicole_and_Cindy_combined (N : ℕ) : ℕ := N + Cindy_collected N
def Rex_collected (N : ℕ) : ℕ := (Nicole_and_Cindy_combined N) / 2
def Rex_cards_each (N : ℕ) : ℕ := Rex_collected N / 4

-- Question: How many cards did Nicole collect? Answer: N = 400
theorem Nicole_cards (N : ℕ) (h : Rex_cards_each N = 150) : N = 400 :=
sorry

end Nicole_cards_l317_317233


namespace local_maximum_condition_l317_317983

noncomputable def f (a b x : ℝ) : ℝ := log x + a * x^2 + b * x

theorem local_maximum_condition (a : ℝ) :
  (∃ b : ℝ, ∀ x : ℝ, x > 0 → diff3 f a b 1 = 0 → diff3 f a b (x - 1) ≤ 0) → a < 1/2 :=
begin
  sorry
end

end local_maximum_condition_l317_317983


namespace solve_for_y_l317_317607

theorem solve_for_y (y : ℤ) : 3^(y-4) = 9^(y+2) → y = -8 := by
  intro h
  have base_eq : 9 = 3^2 := by norm_num
  rw [base_eq] at h
  rw [pow_mul] at h
  have h1 : 3^(y - 4) = 3^(2 * (y + 2)) := by
    rw [←h]
  have exp_eq : (y - 4) = 2 * (y + 2) := by 
    apply pow_injective
    norm_num
    exact h1
  linarith only [exp_eq]
  sorry

end solve_for_y_l317_317607


namespace find_z_value_l317_317229

def x : ℝ := 88 + (4 / 3) * 88

def y : ℝ := x + (3 / 5) * x

def z : ℝ := (1 / 2) * (x + y)

theorem find_z_value : z = 266.9325 := 
  by
  sorry

end find_z_value_l317_317229


namespace top_four_players_points_lost_l317_317506

theorem top_four_players_points_lost
  (scores : List ℝ)
  (h_size : scores.length = 8)
  (h_sorted : scores = [7, 6, 4, 4, 3, 2, 1.5, 0.5]) :
  let top_four := scores.take 4;
  let bottom_four := scores.drop 4;
  (top_four.sum - (4 * 4)) = 1 :=
by
  let top_four := scores.take 4
  let bottom_four := scores.drop 4
  have h_top_four_sum : top_four.sum = 21 := by
    simp [top_four, scores]
  have h_actual_points : 4 * 4 = 16 := by
    norm_num
  have h_points_lost := h_top_four_sum - h_actual_points
  show h_points_lost = 1
  sorry

end top_four_players_points_lost_l317_317506


namespace cardinality_of_A_l317_317126

def A : set (ℕ × ℕ) := { p | p.1 ^ 2 + p.2 ^ 2 ≤ 3 }

theorem cardinality_of_A : set.card A = 4 := 
by
  sorry

end cardinality_of_A_l317_317126


namespace cost_of_baseball_deck_l317_317212

def cost_of_pack : ℝ := 4.45
def num_packs : ℕ := 4
def total_spent : ℝ := 23.86

theorem cost_of_baseball_deck : 
  let cost_of_digimon_cards := num_packs * cost_of_pack in
  total_spent - cost_of_digimon_cards = 6.06 := 
by
  sorry

end cost_of_baseball_deck_l317_317212


namespace library_average_visitors_l317_317350

theorem library_average_visitors (V : ℝ) (h1 : (4 * 1000 + 26 * V = 750 * 30)) : V = 18500 / 26 := 
by 
  -- The actual proof is omitted and replaced by sorry.
  sorry

end library_average_visitors_l317_317350


namespace sequence_not_perfect_square_l317_317146

-- Define the sequence of numbers.
def N (i : ℕ) : ℕ := 
  let zeros := String.mk $ List.replicate (i - 1) '0'
  ("2014" ++ zeros ++ "2015").to_nat

-- Main theorem stating that none of the numbers in the sequence are perfect squares.
theorem sequence_not_perfect_square (i : ℕ) : ¬ ∃ x : ℕ, x * x = N i := by
  sorry

end sequence_not_perfect_square_l317_317146


namespace arrangement_of_11250_l317_317182

theorem arrangement_of_11250 : 
  let digits := [1, 1, 2, 5, 0]
  let total_count := 21
  let valid_arrangement (num : ℕ) : Prop := ∃ (perm : List ℕ), List.perm perm digits ∧ (num % 5 = 0) ∧ (num / 10000 ≥ 1)
  ∃ (count : ℕ), count = total_count ∧ 
  count = Nat.card {n // valid_arrangement n} := 
by 
  sorry

end arrangement_of_11250_l317_317182


namespace minimum_value_sum_l317_317556

theorem minimum_value_sum (x : Fin 50 → ℝ) (h_pos : ∀ i, 0 < x i) (h_sum_squares : ∑ i, (x i) ^ 2 = 50) :
  (∑ i, (x i) / (2 - (x i)^2)) ≥ 10 :=
by
  sorry

end minimum_value_sum_l317_317556


namespace sum_S13_l317_317083

variable {a : ℕ → ℝ}

-- Define the arithmetic sequence property
axiom arithmetic_sequence (a : ℕ → ℝ) : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions: The arithmetic sequence and the condition on terms
axiom h1 : a 5 + a 9 - a 7 = 10

-- Define the sum sequence S_n
def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

-- The theorem stating that S_13 = 130 given the conditions
theorem sum_S13 : S 13 = 130 :=
by
  sorry

end sum_S13_l317_317083


namespace count_n_condition_l317_317057

open Real

theorem count_n_condition : (Finset.card {n ∈ Finset.range 1001 | ∀ t : ℝ, (sin t + complex.i * cos t)^n = sin (n * t) + complex.i * cos (n * t) }) = 250 := 
sorry

end count_n_condition_l317_317057


namespace correct_answer_l317_317274

noncomputable def C (F : ℤ) : ℤ := ((5 * (F - 32)) / 9) 
noncomputable def C' (F : ℤ) : ℤ := ((5 * (F - 32) + 4) / 9) -- rounding implemented by + 4 and div 9
noncomputable def F' (F : ℤ) : ℤ := ((9 * C' F + 32 * 5 + 4) / 5) -- rounding to nearest integer implemented

theorem correct_answer : 
  (Finset.Icc 32 2000).filter (λ F, F = F' F).card = 1094 :=
by {
  sorry
}

end correct_answer_l317_317274


namespace number_of_paths_A_to_D_via_B_C_l317_317464

theorem number_of_paths_A_to_D_via_B_C :
  let paths (steps_right steps_down : ℕ) := Nat.choose (steps_right + steps_down) steps_right in
  let paths_A_to_B := paths 2 2 in
  let paths_B_to_C := paths 1 3 in
  let paths_C_to_D := paths 3 1 in
  paths_A_to_B * paths_B_to_C * paths_C_to_D = 96 :=
by
  let paths (steps_right steps_down : ℕ) := Nat.choose (steps_right + steps_down) steps_right
  let paths_A_to_B := paths 2 2
  let paths_B_to_C := paths 1 3
  let paths_C_to_D := paths 3 1
  have paths_A_to_B_eq : paths_A_to_B = Nat.choose 4 2 := Nat.choose_eq_comb 4 2
  have paths_B_to_C_eq : paths_B_to_C = Nat.choose 4 1 := Nat.choose_eq_comb 4 1
  have paths_C_to_D_eq : paths_C_to_D = Nat.choose 4 1 := Nat.choose_eq_comb 4 1
  calc
    paths_A_to_B * paths_B_to_C * paths_C_to_D
        = Nat.choose 4 2 * Nat.choose 4 1 * Nat.choose 4 1 : by rw [paths_A_to_B_eq, paths_B_to_C_eq, paths_C_to_D_eq]
    ... = 6 * 4 * 4 : by norm_num
    ... = 96 : by norm_num

end number_of_paths_A_to_D_via_B_C_l317_317464


namespace domain_of_f_l317_317404

def f (x: ℝ) : ℝ := (x^2 + 5 * x + 6) / (abs (x - 2) + abs (x + 3) - 5)

theorem domain_of_f : 
  {x : ℝ | abs (x - 2) + abs (x + 3) - 5 ≠ 0} = 
    (Set.Iio (-3)) ∪ (Set.Ioi 2) :=
by
  sorry

end domain_of_f_l317_317404


namespace curve_is_rhombus_not_square_l317_317943

theorem curve_is_rhombus_not_square : (∃ x y : ℝ, (abs (x + y) / 2 + abs (x - y) = 1)) ↔ 
  (∃ a b c d : ℝ, x = a ∧ y = b ∧ (√(a^2 + b^2) = 1) ∧ (a^2 + b^2) ≠ (c^2 + d^2)) :=
  sorry

end curve_is_rhombus_not_square_l317_317943


namespace luke_payment_difference_l317_317231

noncomputable def plan1_balance_after_3_years (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def plan1_remaining_balance_after_7_years (balance_3yrs : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  balance_3yrs * (1 + r / n) ^ (n * t)

noncomputable def total_payments_plan1 (initial_balance : ℝ) (r : ℝ) (n : ℝ) : ℝ :=
  let balance_3yrs := plan1_balance_after_3_years initial_balance r n 3
  let half_payment := balance_3yrs / 2
  let remaining_balance := plan1_remaining_balance_after_7_years half_payment r n 7
  half_payment + remaining_balance

noncomputable def total_payments_plan2 (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r) ^ t

theorem luke_payment_difference (P : ℝ) (r_monthly r_annual : ℝ) : ℝ :=
  let T1 := total_payments_plan1 P r_monthly 12
  let T2 := total_payments_plan2 P r_annual 10
  T2 - T1 = 6705 :=
by
  sorry

end luke_payment_difference_l317_317231


namespace no_valid_p_e_and_f_l317_317446

variable (e f : Prop)

-- Given conditions
axiom p_e : ℝ := 25
axiom p_f : ℝ := 75
variable (p_e_and_f : ℝ) -- unknown value
axiom p_e_given_f : ℝ := 3
axiom p_f_given_e : ℝ := 3

-- Definitions of conditional probability
def cond_prob_e_given_f : ℝ := p_e_and_f / p_f
def cond_prob_f_given_e : ℝ := p_e_and_f / p_e

theorem no_valid_p_e_and_f : cond_prob_e_given_f = p_e_given_f → cond_prob_f_given_e = p_f_given_e → false := by
  sorry

end no_valid_p_e_and_f_l317_317446


namespace area_ratio_l317_317968

-- Definitions based on conditions
def Triangle := Type
def EquilateralTriangle (ABC : Triangle) := sorry -- assume definition of equilateral triangle
def Extend (A B' : ℝ) (k : ℝ) := B' = k * A

variable (ABC : Triangle) [EquilateralTriangle ABC]
def A₀ B₀ C₀ : ℝ := sorry -- assume side lengths

-- Extensions with given conditions
def BB' := A₀ * 3
def CC' := B₀ * 3
def AA' := C₀ * 3

-- Problem statement to prove
theorem area_ratio (A₀ B₀ C₀ : ℝ) (h₁ : BB' = 3 * A₀) (h₂ : CC' = 3 * B₀) (h₃ : AA' = 3 * C₀) :
  let area_ABC := sorry -- assume some area calculation for triangle ABC
  let area_A'B'C' := sorry -- assume some area calculation for triangle A'B'C'
  area_A'B'C' = 9 * area_ABC :=
sorry

end area_ratio_l317_317968


namespace probability_is_one_third_l317_317260

noncomputable def probability_four_of_a_kind_or_full_house : ℚ :=
  let total_outcomes := 6
  let probability_triplet_match := 1 / total_outcomes
  let probability_pair_match := 1 / total_outcomes
  probability_triplet_match + probability_pair_match

theorem probability_is_one_third :
  probability_four_of_a_kind_or_full_house = 1 / 3 :=
by
  -- sorry
  trivial

end probability_is_one_third_l317_317260


namespace no_perfect_square_in_sequence_l317_317134

def sequence_term (i : ℕ) : ℕ :=
  let baseDigits : List ℕ := [2, 0, 1, 4, 2, 0, 1, 5]
  let termDigits := baseDigits.foldr (fun d acc => acc + d * 10 ^ (baseDigits.length - acc.length - 1)) 0
  termDigits + 10 ^ (i + 5)

theorem no_perfect_square_in_sequence : ¬ ∃ i : ℕ, ∃ k : ℕ, k * k = sequence_term i := 
sorry

end no_perfect_square_in_sequence_l317_317134


namespace sample_stddev_and_range_unchanged_l317_317863

noncomputable def sample_data (n : ℕ) : Type :=
  fin n → ℝ

variables (n : ℕ) (x y : sample_data n) (c : ℝ)

-- Condition: creating new sample data by adding a constant c
axiom data_transform : ∀ i : fin n, y i = x i + c 

-- Condition: c is non-zero
axiom c_nonzero : c ≠ 0

-- The theorem that states sample standard deviations and sample ranges of x and y are the same
theorem sample_stddev_and_range_unchanged :
  (sample_standard_deviation y = sample_standard_deviation x) ∧ 
  (sample_range y = sample_range x) :=
sorry

end sample_stddev_and_range_unchanged_l317_317863


namespace area_PQR_l317_317955

variables {P Q R S T : Type}
 [Order P] [Order Q] [Order R] [Order S] [Order T]

variable (triangle : Π {P Q R : Type} (PQR : Set (P × Q × R)), Prop)
variable (area : Π {PQR : Set (P × Q × R)} (T : P × Q × R), ℝ)

def PQ {P Q R : Type} (PQR : Set (P × Q × R)) := { PS : P × Q ⊆ PQ | ∃ x, x / 3 = 2 }
def SR {S R : Type} (PQR : Set (S × R)) := { TS : S ⊆ SR | ∃ h, h = 20 ∧ h = 18 }

theorem area_PQR (PQR : Set (P × Q × R)) 
(PQ_INIT : PQ PQR = { r | ∃ r1, r1 / 3 = 2 })
(SR_INIT : SR PQR = { s1 | ∃ s2, area SR = 20 ∧ area QT = 18 }) :
  area (triangle PQR) = 80 :=
sorry

end area_PQR_l317_317955


namespace count_4_digit_even_digits_div_by_4_l317_317131

theorem count_4_digit_even_digits_div_by_4 : 
  let even_digits := {0, 2, 4, 6, 8}
  let num_valid_combinations := 4 * 5 * 15
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 2 = 0 ∧ n % 4 = 0 ∧ 
          (∀ d ∈ to_digits n, d ∈ even_digits) → 
          num_valid_combinations = 300 := 
by
  sorry

end count_4_digit_even_digits_div_by_4_l317_317131


namespace at_least_2020_distinct_n_l317_317250

theorem at_least_2020_distinct_n : 
  ∃ (N : Nat), N ≥ 2020 ∧ ∃ (a : Fin N → ℕ), 
  Function.Injective a ∧ ∀ i, ∃ k : ℚ, (a i : ℚ) + 0.25 = (k + 1/2)^2 := 
sorry

end at_least_2020_distinct_n_l317_317250


namespace problem_l317_317112

noncomputable def f (x : ℝ) := x / (1 + x^2)

open Set

theorem problem
  (h1 : ∀ x : ℝ, f(-x) = -f(x))
  (h2 : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f(x1) < f(x2))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 1/2 → f(x-1) + f(x) < 0) :
  True :=
  sorry

end problem_l317_317112


namespace complex_sum_exp_l317_317002

open Complex

theorem complex_sum_exp : (∑ k in finset.range 16, exp (2 * π * I * (k + 1) / 17)) = -1 := 
  sorry

end complex_sum_exp_l317_317002


namespace trio_all_three_games_l317_317512

theorem trio_all_three_games (n : ℕ) :
  ∀ (G : SimpleGraph (Fin (3 * n + 1))),
    (∀ v : Fin (3 * n + 1), ∃ w1 w2 w3 : Fin (3 * n + 1),
      w1 ≠ v ∧ w2 ≠ v ∧ w3 ≠ v ∧
      G.Adj v w1 ∧ G.Adj v w2 ∧ G.Adj v w3 ∧ 
      ((G.Adj w1 w2 ∧ G.Adj w2 w3 ∧ G.Adj w3 w1) ∨ 
      (G.Adj w1 w3 ∧ G.Adj v w2 ∧ G.Adj v w1))) →
    ∃ (v1 v2 v3 : Fin (3 * n + 1)),
      v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
      G.Adj v1 v2 ∧ G.Adj v2 v3 ∧ G.Adj v3 v1 :=
sorry

end trio_all_three_games_l317_317512


namespace complex_sum_exp_l317_317005

open Complex

theorem complex_sum_exp : (∑ k in finset.range 16, exp (2 * π * I * (k + 1) / 17)) = -1 := 
  sorry

end complex_sum_exp_l317_317005


namespace seven_digit_numbers_count_l317_317065

/-- Given a six-digit phone number represented by six digits A, B, C, D, E, F:
- There are 7 positions where a new digit can be inserted: before A, between each pair of consecutive digits, and after F.
- Each of these positions can be occupied by any of the 10 digits (0 through 9).
The number of seven-digit numbers that can be formed by adding one digit to the six-digit phone number is 70. -/
theorem seven_digit_numbers_count (A B C D E F : ℕ) (hA : 0 ≤ A ∧ A < 10) (hB : 0 ≤ B ∧ B < 10) 
  (hC : 0 ≤ C ∧ C < 10) (hD : 0 ≤ D ∧ D < 10) (hE : 0 ≤ E ∧ E < 10) (hF : 0 ≤ F ∧ F < 10) : 
  ∃ n : ℕ, n = 70 :=
sorry

end seven_digit_numbers_count_l317_317065


namespace Jones_travel_time_l317_317543

/-- 
Jones traveled 100 miles on his first trip. On his second trip, he traveled 500 miles while going 
four times as fast as on his first trip. Prove that his new time was 1.25 times his old time.
-/
theorem Jones_travel_time (v : ℝ) (h1 : v > 0) :
  let t1 := 100 / v in
  let t2 := 125 / v in
  t2 = 1.25 * t1 :=
by
  sorry

end Jones_travel_time_l317_317543


namespace inequality_has_one_solution_l317_317639

def posIntSolutionsToInequality : ℕ :=
  {x : ℕ // x > 0 ∧ 4 * (x - 1) < 3 * x - 2}.card

theorem inequality_has_one_solution : posIntSolutionsToInequality = 1 := by
  sorry

end inequality_has_one_solution_l317_317639


namespace fish_population_estimation_l317_317936

def tagged_fish_day1 := (30, 25, 25) -- (Species A, Species B, Species C)
def tagged_fish_day2 := (40, 35, 25) -- (Species A, Species B, Species C)
def caught_fish_day3 := (60, 50, 30) -- (Species A, Species B, Species C)
def tagged_fish_day3 := (4, 6, 2)    -- (Species A, Species B, Species C)
def caught_fish_day4 := (70, 40, 50) -- (Species A, Species B, Species C)
def tagged_fish_day4 := (5, 7, 3)    -- (Species A, Species B, Species C)

def total_tagged_fish (day1 : (ℕ × ℕ × ℕ)) (day2 : (ℕ × ℕ × ℕ)) :=
  let (a1, b1, c1) := day1
  let (a2, b2, c2) := day2
  (a1 + a2, b1 + b2, c1 + c2)

def average_proportion_tagged (caught3 tagged3 caught4 tagged4 : (ℕ × ℕ × ℕ)) :=
  let (c3a, c3b, c3c) := caught3
  let (t3a, t3b, t3c) := tagged3
  let (c4a, c4b, c4c) := caught4
  let (t4a, t4b, t4c) := tagged4
  ((t3a / c3a + t4a / c4a) / 2,
   (t3b / c3b + t4b / c4b) / 2,
   (t3c / c3c + t4c / c4c) / 2)

def estimate_population (total_tagged average_proportion : (ℕ × ℕ × ℕ)) :=
  let (ta, tb, tc) := total_tagged
  let (pa, pb, pc) := average_proportion
  (ta / pa, tb / pb, tc / pc)

theorem fish_population_estimation :
  let total_tagged := total_tagged_fish tagged_fish_day1 tagged_fish_day2
  let avg_prop := average_proportion_tagged caught_fish_day3 tagged_fish_day3 caught_fish_day4 tagged_fish_day4
  estimate_population total_tagged avg_prop = (1014, 407, 790) :=
by
  sorry

end fish_population_estimation_l317_317936


namespace jett_profit_l317_317209

def initial_cost : ℕ := 600
def vaccination_cost : ℕ := 500
def daily_food_cost : ℕ := 20
def number_of_days : ℕ := 40
def selling_price : ℕ := 2500

def total_expenses : ℕ := initial_cost + vaccination_cost + daily_food_cost * number_of_days
def profit : ℕ := selling_price - total_expenses

theorem jett_profit : profit = 600 :=
by
  -- Completed proof steps
  sorry

end jett_profit_l317_317209


namespace simplify_cub_root_multiplication_l317_317599

theorem simplify_cub_root_multiplication (a b : ℝ) (ha : a = 8) (hb : b = 27) :
  (real.cbrt (a + b) * real.cbrt (a + real.cbrt b)) = real.cbrt ((a + b) * (a + real.cbrt b)) := 
by
  sorry

end simplify_cub_root_multiplication_l317_317599


namespace complex_number_quadrant_l317_317075

def quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else if z.re > 0 ∧ z.im < 0 then 4
  else 0  -- This handles the cases where the point lies on the axis, which is not relevant here.

theorem complex_number_quadrant :
  ∀ z : ℂ, (z * (complex.I)) = (2 + complex.I) → quadrant z = 4 :=
by
  intro z
  intro h
  sorry

end complex_number_quadrant_l317_317075


namespace original_ratio_white_yellow_l317_317363

-- Define the given conditions
variables (W Y : ℕ)
axiom total_balls : W + Y = 64
axiom erroneous_dispatch : W = 8 * (Y + 20) / 13

-- The theorem we need to prove
theorem original_ratio_white_yellow (W Y : ℕ) (h1 : W + Y = 64) (h2 : W = 8 * (Y + 20) / 13) : W = Y :=
by sorry

end original_ratio_white_yellow_l317_317363


namespace wilsons_theorem_l317_317554

theorem wilsons_theorem (p : ℕ) (hp : p ≥ 2) : Nat.Prime p ↔ (Nat.factorial (p - 1) + 1) % p = 0 := 
sorry

end wilsons_theorem_l317_317554


namespace complex_pure_imaginary_solution_l317_317924

theorem complex_pure_imaginary_solution (a : ℝ) :
  (a^2 - 4 * a + 3 = 0) → (a = 1 ∨ a = 3) :=
by
  intro h
  have h_eq : (a - 1) * (a - 3) = 0 := sorry
  cases h_eq with h1 h3
  · left
    exact h1
  · right
    exact h3

end complex_pure_imaginary_solution_l317_317924


namespace part1_part2_part3_l317_317103

-- Definitions for conditions used in the proof problems
def eq1 (a b : ℝ) : Prop := 2 * a + b = 0
def eq2 (a x : ℝ) : Prop := x = a ^ 2

-- Part 1: Prove b = 4 and x = 4 given a = -2
theorem part1 (a b x : ℝ) (h1 : a = -2) (h2 : eq1 a b) (h3 : eq2 a x) : b = 4 ∧ x = 4 :=
by sorry

-- Part 2: Prove a = -3 and x = 9 given b = 6
theorem part2 (a b x : ℝ) (h1 : b = 6) (h2 : eq1 a b) (h3 : eq2 a x) : a = -3 ∧ x = 9 :=
by sorry

-- Part 3: Prove x = 2 given a^2*x + (a + b)^2*x = 8
theorem part3 (a b x : ℝ) (h : a^2 * x + (a + b)^2 * x = 8) : x = 2 :=
by sorry

end part1_part2_part3_l317_317103


namespace arc_length_l317_317620

theorem arc_length (circumference : ℝ) (angle_degrees : ℝ) (h : circumference = 90) (θ : angle_degrees = 45) :
  (angle_degrees / 360) * circumference = 11.25 := 
  by 
    sorry

end arc_length_l317_317620


namespace minimum_distance_l317_317100

/-
  Given that P is on the parabola y^2 = 8x, 
  with F being the focus of the parabola, 
  and given that point A has coordinates (3, 2),
  the minimum value of |PA| + |PF| is equal to 5.
-/

open Real

def parabola (x y : ℝ) : Prop := y^2 = 8 * x
def point_A := (3 : ℝ, 2 : ℝ)
def focus_F := (2 : ℝ, 0 : ℝ)

theorem minimum_distance (P : ℝ × ℝ) 
  (hP : parabola P.1 P.2) :
  |(real.dist P point_A) + (real.dist P focus_F)| = 5 := 
sorry

end minimum_distance_l317_317100


namespace arithmetic_mean_solution_l317_317619

theorem arithmetic_mean_solution (x : ℚ) :
  (x + 10 + 20 + 3*x + 18 + 3*x + 6) / 5 = 30 → x = 96 / 7 :=
by
  intros h
  sorry

end arithmetic_mean_solution_l317_317619


namespace evaluate_exponentiation_l317_317396

theorem evaluate_exponentiation : (81 : ℝ) ^ (2 ^ (-2 : ℝ)) = 3 := by
  sorry

end evaluate_exponentiation_l317_317396


namespace poly_factor_l317_317819

theorem poly_factor (c p : ℚ) (h1 : 3x^3 + c * x + 8 = (x^2 + p * x - 1) * (3x + 8)) : 
c = -73 / 3 :=
by
  sorry

end poly_factor_l317_317819


namespace sum_of_positive_integers_l317_317420

theorem sum_of_positive_integers (n : ℕ) (h : 2.5 * n - 6.5 < 10) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 → List.sum [1, 2, 3, 4, 5, 6] = 21 := by sorry

end sum_of_positive_integers_l317_317420


namespace bug_returns_to_start_probability_l317_317718

-- Define the structure of the cube
structure Cube :=
(vertices : Fin 8)
(edges : vertices → List vertices)

-- Define the probability calculation for the problem
def cube_paths (c : Cube) (start : c.vertices) : Nat :=
  3 ^ 8

def valid_cube_paths (c : Cube) (start : c.vertices) : Nat :=
  192

def probability_return_to_start (c : Cube) (start : c.vertices) : ℚ :=
  (valid_cube_paths c start : ℚ) / (cube_paths c start : ℚ)

-- The main theorem to prove the probability
theorem bug_returns_to_start_probability :
  ∀ (c : Cube) (start : c.vertices),
  probability_return_to_start c start = 64 / 2187 :=
by
  intro c start
  sorry

end bug_returns_to_start_probability_l317_317718


namespace point_in_first_quadrant_l317_317281

-- Introducing the coordinates (2, 1)
def point := (2, 1)

-- Defining conditions for the first quadrant
def is_first_quadrant (p : ℝ × ℝ) : Prop := p.fst > 0 ∧ p.snd > 0

-- Stating the theorem
theorem point_in_first_quadrant : is_first_quadrant point :=
by
  -- Proof goes here
  sorry

end point_in_first_quadrant_l317_317281


namespace variance_of_data_l317_317758

noncomputable def data : List ℝ := [10, 6, 8, 5, 6]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length : ℝ)

noncomputable def variance (l : List ℝ) : ℝ :=
  (l.map (λ x, (x - mean l)^2)).sum / (l.length : ℝ)

theorem variance_of_data : variance data = 3.2 := 
  by
  sorry

end variance_of_data_l317_317758


namespace math_proof_l317_317459

-- Definitions
def U := Set ℝ
def A : Set ℝ := {x | x ≥ 3}
def B : Set ℝ := {x | x^2 - 8*x + 7 ≤ 0}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem
theorem math_proof (a : ℝ) :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x | x ≥ 1}) ∧
  (C a ∪ A = A → a ≥ 4) :=
by
  sorry

end math_proof_l317_317459
