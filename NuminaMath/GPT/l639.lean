import Mathlib
import Mathlib.
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomMean
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.InnerProductSpace.Projection
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry
import Mathlib.Meta.Default
import Mathlib.NumberTheory.Moduli.Basic
import Mathlib.Probability
import Mathlib.Set.Basic
import Mathlib.SetTheory.Cardinal.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Topology.Continuity

namespace count_integers_in_range_with_increasing_digits_l639_639406

theorem count_integers_in_range_with_increasing_digits : 
  let integers_in_range := { n | 200 ≤ n ∧ n < 250 ∧ ((n % 10), (n / 10 % 10), (n / 100)) = (d₀, d₁, d₂) ∧ d₀ < d₁ ∧ d₁ < d₂ } in
  integers_in_range.card = 34 :=
sorry

end count_integers_in_range_with_increasing_digits_l639_639406


namespace exists_solution_for_lambda_9_l639_639333

theorem exists_solution_for_lambda_9 :
  ∃ x y : ℝ, (x^2 + y^2 = 8 * x + 6 * y) ∧ (9 * x^2 + y^2 = 6 * y) ∧ (y^2 + 9 = 9 * x + 6 * y + 9) :=
by
  sorry

end exists_solution_for_lambda_9_l639_639333


namespace reflection_squared_is_identity_l639_639955

noncomputable def reflection_matrix (v : ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let (a, b) := v in
  let norm := real.sqrt (a * a + b * b) in
  ![(a * a - b * b) / norm^2, (2 * a * b) / norm^2; (2 * a * b) / norm^2, (b * b - a * a) / norm^2]

theorem reflection_squared_is_identity :
  let S := reflection_matrix (4, -2) in S * S = 1 :=
sorry

end reflection_squared_is_identity_l639_639955


namespace intersection_point_property_l639_639909

noncomputable def P : ℝ × ℝ := (0, 1)
noncomputable def Q : ℝ × ℝ := (-3, 0)

def line_l (a : ℝ) : ℝ × ℝ → Prop := λ M, a * M.1 + M.2 - 1 = 0
def line_m (a : ℝ) : ℝ × ℝ → Prop := λ M, M.1 - a * M.2 + 3 = 0

theorem intersection_point_property (a : ℝ) (M : ℝ × ℝ)
  (hl : line_l a M) (hm : line_m a M) :
  (dist M P)^2 + (dist M Q)^2 = 10 := sorry

end intersection_point_property_l639_639909


namespace sphere_volume_of_circumscribed_rect_prism_l639_639348

theorem sphere_volume_of_circumscribed_rect_prism
  (l1 l2 l3 : ℝ) (hl1 : l1 = 1) (hl2 : l2 = sqrt 10) (hl3 : l3 = 5) :
  let d := sqrt (l1^2 + l2^2 + l3^2) in
  let R := d / 2 in
  let V := (4 / 3) * π * R^3 in
  V = 36 * π := 
by 
  sorry

end sphere_volume_of_circumscribed_rect_prism_l639_639348


namespace min_value_expr_l639_639782

open Real

theorem min_value_expr (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) :
  3 * cos θ + 2 / sin θ + 2 * sqrt 2 * tan θ ≥ (3 : ℝ) * (12 * sqrt 2)^((1 : ℝ) / (3 : ℝ)) := sorry

end min_value_expr_l639_639782


namespace tangent_point_ratio_l639_639689

-- Define the vertices of the right triangle ABC
variables (A B C K : Point)

-- Midpoints M and N
variables (M N : Point)

-- The conditions
variables
  (h1 : midpoint A B M) -- M is midpoint of AB
  (h2 : midpoint B C N) -- N is midpoint of BC
  (circle_passing_through_midpoints : circle_through_points M N)
  (tangent_circle_touch_point : circle_tangent_at_point K A C)

-- The length of line segment
variables (len_AC : ℝ)

-- Definitions and the final ratio condition
def AC_split_ratio : AC_ratio := sorry

-- Statement of the theorem
theorem tangent_point_ratio :
  AC_split_ratio K A C M N len_AC = 1 / 3 := 
  sorry

end tangent_point_ratio_l639_639689


namespace cost_of_each_book_l639_639546

theorem cost_of_each_book
  (initial_money : ℝ)
  (books_leftover_money : ℝ)
  (number_of_books : ℕ)
  (total_cost_of_books : ℝ) :
  initial_money = 85 → books_leftover_money = 35 → number_of_books = 10 → total_cost_of_books = initial_money - books_leftover_money → 
  (total_cost_of_books / number_of_books) = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have : total_cost_of_books = 50 := by linarith
  rw this
  norm_num

end cost_of_each_book_l639_639546


namespace count_integers_with_increasing_digits_200_to_250_l639_639424

theorem count_integers_with_increasing_digits_200_to_250 :
  {n : ℕ | 200 ≤ n ∧ n ≤ 250 ∧ 
          (let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 in 
           d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
           d1 < d2 ∧ d2 < d3)}.card = 11 :=
by
  sorry

end count_integers_with_increasing_digits_200_to_250_l639_639424


namespace paint_price_max_boxes_paint_A_l639_639569

theorem paint_price :
  ∃ x y : ℕ, (x + 2 * y = 56 ∧ 2 * x + y = 64) ∧ x = 24 ∧ y = 16 := by
  sorry

theorem max_boxes_paint_A (m : ℕ) :
  24 * m + 16 * (200 - m) ≤ 3920 → m ≤ 90 := by
  sorry

end paint_price_max_boxes_paint_A_l639_639569


namespace a_speed_calculation_l639_639168

def meeting_time : ℝ := 179.98560115190787
def track_length : ℝ := 900
def b_speed_kmph : ℝ := 54
def b_speed_mps : ℝ := b_speed_kmph * (1000 / 3600)
def b_distance_covered : ℝ := b_speed_mps * meeting_time
def b_laps : ℝ := b_distance_covered / track_length

theorem a_speed_calculation
  (meeting_time: ℝ)
  (track_length: ℝ)
  (b_speed_kmph: ℝ)
  (a_speed: ℝ)
  (h_meet_time: meeting_time = 179.98560115190787)
  (h_track_length: track_length = 900)
  (h_b_speed: b_speed_kmph = 54)
  : a_speed = 54 := by
  sorry

end a_speed_calculation_l639_639168


namespace range_of_a_l639_639051

noncomputable def f (a x : ℝ) : ℝ := x + a^2 / x
noncomputable def g (x : ℝ) : ℝ := x - Real.log x

theorem range_of_a (a : ℝ) :
  (a > 0) ∧ (∀ x2 ∈ set.Icc (1/Real.exp 1) 1, ∃ x1 ∈ set.Icc (1/Real.exp 1) 1, f a x1 ≥ g x2) ↔ 
  a ∈ set.Ici (1/2) ∪ set.Icc (Real.sqrt (Real.exp 2 - 1) / Real.exp 1) (1 / Real.exp 1) := 
  sorry

end range_of_a_l639_639051


namespace petya_wins_last_l639_639401

--- Definitions and conditions
variables {a b c : ℝ}
variable h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a
variable h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
variable discriminant : ℝ × ℝ × ℝ → ℝ := λ (coeffs : ℝ × ℝ × ℝ), coeffs.2^2 - 4 * coeffs.1 * coeffs.3

noncomputable def petya_turn (coeffs : ℝ × ℝ × ℝ) : Prop :=
discriminant coeffs ≥ 0

noncomputable def vasya_turn (coeffs : ℝ × ℝ × ℝ) : Prop :=
discriminant coeffs < 0

variable order : {coeffs // petya_turn coeffs} ⊕ {coeffs // vasya_turn coeffs}  -- permutation results assuming order in such way 5 already settled
variable sequence : list ({coeffs // petya_turn coeffs} ⊕ {coeffs // vasya_turn coeffs})

axiom petya_three_first : sequence.take 3 = [sum.inl _, sum.inl _, sum.inl _]
axiom vasya_two_next : sequence.drop 3 = [sum.inr _, sum.inr _]

theorem petya_wins_last [inhabited sequence] :
  sequence.length = 5 → sum.inl (_ : {coeffs // petya_turn coeffs}) ∈ (sequence.take 6).nth_le 5 sorry :=
sorry

end petya_wins_last_l639_639401


namespace sqrt_expression_l639_639750

theorem sqrt_expression : Real.sqrt ((4^2) * (5^6)) = 500 := by
  sorry

end sqrt_expression_l639_639750


namespace sum_of_slopes_correct_l639_639672

noncomputable def sum_of_slopes : ℚ :=
  let Γ1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}
  let Γ2 := {p : ℝ × ℝ | (p.1 - 10)^2 + (p.2 - 11)^2 = 1}
  let l := {k : ℝ | ∃ p1 ∈ Γ1, ∃ p2 ∈ Γ1, ∃ p3 ∈ Γ2, ∃ p4 ∈ Γ2, p1 ≠ p2 ∧ p3 ≠ p4 ∧ p1.2 = k * p1.1 ∧ p3.2 = k * p3.1}
  let valid_slopes := {k | k ∈ l ∧ (k = 11/10 ∨ k = 1 ∨ k = 5/4)}
  (11 / 10) + 1 + (5 / 4)

theorem sum_of_slopes_correct : sum_of_slopes = 67 / 20 := 
  by sorry

end sum_of_slopes_correct_l639_639672


namespace intersection_of_sets_l639_639862

def set_M := { y : ℝ | y ≥ 0 }
def set_N := { y : ℝ | ∃ x : ℝ, y = -x^2 + 1 }

theorem intersection_of_sets : set_M ∩ set_N = { y : ℝ | 0 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_of_sets_l639_639862


namespace log_sqrt_12_eq_7_l639_639300

theorem log_sqrt_12_eq_7 : log (1728 * sqrt 12) (sqrt 12) = 7 :=
by
  sorry

end log_sqrt_12_eq_7_l639_639300


namespace athena_total_spent_l639_639913

noncomputable def cost_sandwiches := 4 * 3.25
noncomputable def cost_fruit_drinks := 3 * 2.75
noncomputable def cost_cookies := 6 * 1.50
noncomputable def cost_chips := 2 * 1.85

noncomputable def total_cost := cost_sandwiches + cost_fruit_drinks + cost_cookies + cost_chips

theorem athena_total_spent : total_cost = 33.95 := 
by 
  simp [cost_sandwiches, cost_fruit_drinks, cost_cookies, cost_chips, total_cost]
  sorry

end athena_total_spent_l639_639913


namespace range_of_f_max_value_of_f_l639_639851

noncomputable def f (x : ℝ) (φ : ℝ) := (√3 / 2) * Real.cos (2 * x + φ) + Real.sin x ^ 2

-- Part 1
theorem range_of_f (φ : ℝ) (h : φ = π / 6) : (λ x, f x φ) '' (Ici 0 ∩ Iio π) = set.Icc 0 1 := by 
sorry

-- Part 2
theorem max_value_of_f (φ : ℝ) (h : ∀ x, f x φ ≤ 3 / 2) : φ = π / 2 := by 
sorry

end range_of_f_max_value_of_f_l639_639851


namespace rectangle_area_l639_639613

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l639_639613


namespace area_of_triangle_ACG_l639_639700

-- Define a regular octagon and its properties
structure RegularOctagon :=
  (side_length : ℝ) (vertex_A : Type) (vertex_C : Type) (vertex_G : Type) (center_O : Type)

def octagon := RegularOctagon.mk 4 ℕ ℕ ℕ ℕ

-- Define the area of triangle ACG in the octagon
noncomputable def area_triangle_ACG (o : RegularOctagon) : ℝ :=
  128 * real.sqrt (3 - 2 * real.sqrt 2)

-- Assume o is our specific octagon
def specific_octagon : RegularOctagon := octagon

-- Prove the area of triangle ACG in the specific octagon is 128 * sqrt(3 - 2 * sqrt(2))
theorem area_of_triangle_ACG : 
  area_triangle_ACG specific_octagon = 128 * real.sqrt (3 - 2 * real.sqrt 2) := 
sorry

end area_of_triangle_ACG_l639_639700


namespace mutually_exclusive_necessary_but_not_sufficient_for_complementary_l639_639821

variable {Ω : Type}
variable {A1 A2 : Set Ω}

def mutually_exclusive (A1 A2 : Set Ω) : Prop :=
  A1 ∩ A2 = ∅

def complementary (A1 A2 : Set Ω) : Prop :=
  A1 ∪ A2 = set.univ ∧ A1 ∩ A2 = ∅

theorem mutually_exclusive_necessary_but_not_sufficient_for_complementary :
  (mutually_exclusive A1 A2 → complementary A1 A2) ∧
  (complementary A1 A2 → mutually_exclusive A1 A2) :=
by
  sorry

end mutually_exclusive_necessary_but_not_sufficient_for_complementary_l639_639821


namespace routes_in_3x3_grid_l639_639873

theorem routes_in_3x3_grid :
  let n := 3 in
  let total_moves := 2 * n in
  let k := n in
  let combinations := Nat.choose total_moves k in
  combinations = 20 :=
by
  let n := 3
  sorry

end routes_in_3x3_grid_l639_639873


namespace constant_term_expansion_l639_639366

noncomputable def integral_value : ℝ := ∫ x in 1..(Real.exp 1), 6 / x

theorem constant_term_expansion (n : ℕ) (h : n = integral_value) : 
  let t := (x^2 - 1 / x)^n
  in true :=
by
  have integral_result : n = 6 := by
    -- Proof for n = 6
    sorry
  have constant_term : C(6, 2) = 15 := by
    -- Proof using binomial theorem
    sorry
  exact trivial

end constant_term_expansion_l639_639366


namespace count_integers_with_increasing_digits_l639_639419

theorem count_integers_with_increasing_digits :
  let count_integers := 
    ∑ second_digit in ({3, 4, 5} : Finset ℕ), 
      ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ second_digit}, 
        1 in

  count_integers = 15 :=
by
  have step1 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 3}, 1 = 6,
  { -- Explanation: If second digit is 3, third can be 4, 5, 6, 7, 8, 9 -> 6 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step2 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 4}, 1 = 5,
  { -- Explanation: If second digit is 4, third can be 5, 6, 7, 8, 9 -> 5 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step3 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 5}, 1 = 4,
  { -- Explanation: If second digit is 5, third can be 6, 7, 8, 9 -> 4 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have count_integers := step1 + step2 + step3,
  simp only [count_integers, add_comm],
  exact eq.refl 15

end count_integers_with_increasing_digits_l639_639419


namespace rectangle_area_l639_639618

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l639_639618


namespace tan_X_value_l639_639925

open real

-- Definitions of the conditions
def right_triangle (a b c : ℝ) (a_sq_plus_b_sq_eq_c_sq : a^2 + b^2 = c^2) : Prop :=
  a^2 + b^2 = c^2

-- Given conditions
def Y := 90 * (π / 180) -- Angle Y in radians corresponding to 90 degrees
def YZ := 4 : ℝ
def XY := real.sqrt 34

-- Definition to compute XZ based on Pythagorean theorem
def XZ := real.sqrt (XY^2 - YZ^2)

-- Tangent of angle X in the right triangle
def tan_X := YZ / XZ

-- Theorem statement
theorem tan_X_value : 
  right_triangle YZ XY XZ (YZ^2 + XZ^2 = XY^2) →
  tan_X = (2 * real.sqrt 2) / 3 :=
begin
  sorry
end

end tan_X_value_l639_639925


namespace count_integers_in_range_with_increasing_digits_l639_639409

theorem count_integers_in_range_with_increasing_digits : 
  let integers_in_range := { n | 200 ≤ n ∧ n < 250 ∧ ((n % 10), (n / 10 % 10), (n / 100)) = (d₀, d₁, d₂) ∧ d₀ < d₁ ∧ d₁ < d₂ } in
  integers_in_range.card = 34 :=
sorry

end count_integers_in_range_with_increasing_digits_l639_639409


namespace Jennifer_future_age_Jordana_future_age_Jordana_current_age_l639_639933

variable (Jennifer_age_now Jordana_age_now : ℕ)

-- Conditions
def age_in_ten_years (current_age : ℕ) : ℕ := current_age + 10
theorem Jennifer_future_age : age_in_ten_years Jennifer_age_now = 30 := sorry
theorem Jordana_future_age : age_in_ten_years Jordana_age_now = 3 * age_in_ten_years Jennifer_age_now := sorry

-- Question to prove
theorem Jordana_current_age : Jordana_age_now = 80 := sorry

end Jennifer_future_age_Jordana_future_age_Jordana_current_age_l639_639933


namespace find_two_numbers_l639_639142

noncomputable def x := 5 + 2 * Real.sqrt 5
noncomputable def y := 5 - 2 * Real.sqrt 5

theorem find_two_numbers :
  (x * y = 5) ∧ (x + y = 10) :=
by {
  sorry
}

end find_two_numbers_l639_639142


namespace wicket_keeper_older_by_l639_639901

-- Given conditions
def captain_age : ℕ := 25
def team_size : ℕ := 11
def average_team_age : ℕ := 22
def remaining_players_age_after_exclusion : ℕ := 21

theorem wicket_keeper_older_by :
  let W := captain_age + x 
  ∧ total_age := average_team_age * team_size
  ∧ remaining_players := team_size - 2
  ∧ remaining_average_age := 21
  ∧ remaining_total_age := remaining_players_age_after_exclusion * remaining_players in
  total_age = remaining_total_age + captain_age + W -> x = 3 :=
by
  sorry

end wicket_keeper_older_by_l639_639901


namespace total_population_l639_639456

theorem total_population (b g t : ℕ) (h₁ : b = 6 * g) (h₂ : g = 5 * t) :
  b + g + t = 36 * t :=
by
  sorry

end total_population_l639_639456


namespace compute_largest_possible_sum_l639_639945

universe u

def number_of_ordered_pairs (A B : set ℕ) : ℕ :=
  if (A ∪ B).size = 999 ∧ ((A ∩ B) ∩ {1, 2}) = {1} 
  then 2 * 3 ^ 997 * choose 1995 997 + 3 ^ 998 * choose 1995 998 else 0

def largest_possible_sum (A B : set ℕ) : ℕ :=
  if h : number_of_ordered_pairs A B > 0 then 1004 else 0

theorem compute_largest_possible_sum (A B : set ℕ) :
  largest_possible_sum A B = 1004 := by
  sorry

end compute_largest_possible_sum_l639_639945


namespace find_ratio_squares_l639_639823

variables (x y z a b c : ℝ)

theorem find_ratio_squares 
  (h1 : x / a + y / b + z / c = 5) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := 
sorry

end find_ratio_squares_l639_639823


namespace total_bills_proof_l639_639768

variable (a : ℝ) (total_may : ℝ) (total_june_may_june : ℝ)

-- The total bill in May is 140 yuan.
def total_bill_may (a : ℝ) := 140

-- The water bill increases by 10% in June.
def water_bill_june (a : ℝ) := 1.1 * a

-- The electricity bill in May.
def electricity_bill_may (a : ℝ) := 140 - a

-- The electricity bill increases by 20% in June.
def electricity_bill_june (a : ℝ) := (140 - a) * 1.2

-- Total electricity bills in June.
def total_electricity_june (a : ℝ) := (140 - a) + 0.2 * (140 - a)

-- Total water and electricity bills in June.
def total_water_electricity_june (a : ℝ) := 1.1 * a + 168 - 1.2 * a

-- Total water and electricity bills for May and June.
def total_water_electricity_may_june (a : ℝ) := a + (1.1 * a) + (140 - a) + ((140 - a) * 1.2)

-- When a = 40, the total water and electricity bills for May and June.
theorem total_bills_proof : ∀ a : ℝ, a = 40 → total_water_electricity_may_june a = 304 := 
by
  intros a ha
  rw [ha]
  sorry

end total_bills_proof_l639_639768


namespace winning_strategy_l639_639650

theorem winning_strategy (n : ℕ) (h : 1 < n) : 
  (even n → first_player_wins) ∧ (odd n → second_player_wins) := 
sorry

end winning_strategy_l639_639650


namespace find_n_l639_639313

def is_permutation (σ : List ℕ) (n : ℕ) : Prop :=
  σ.perm (List.range (n + 1))

def nested_sqrt (σ : List ℕ) : ℤ :=
  let rec aux (l : List ℕ) : ℚ :=
    match l with
    | [] => 0
    | (h::t) => Real.sqrt (h + aux t)
  aux σ

theorem find_n {n : ℕ} (h : 0 < n) :
  (∃ (σ : List ℕ), is_permutation σ n ∧ nested_sqrt σ ∈ ℚ) ↔ (n = 1 ∨ n = 3) :=
by
  sorry

end find_n_l639_639313


namespace value_of_a_eq_6_l639_639815

theorem value_of_a_eq_6 (a : ℝ) (A : set ℝ) (hA : A = {a^2, 2 - a, 4}) (h_card : A.to_finset.card = 3) : a = 6 :=
by
  sorry

end value_of_a_eq_6_l639_639815


namespace maximal_positive_integers_l639_639554

theorem maximal_positive_integers (n : ℕ) (h : n = 2018) (a : Fin n → ℤ)
  (h_condition : ∀ i : Fin n, a i > a (Fin.prev i) + a (Fin.prev (Fin.prev i))) :
  ∃ k : ℕ, k = 1008 ∧ (∃ S : Finset (Fin n), S.card = k ∧ ∀ i ∈ S, 0 < a i) :=
sorry

end maximal_positive_integers_l639_639554


namespace min_cyclical_subsets_zero_max_cyclical_subsets_l639_639899

-- Given the number of teams is 2n + 1
variable (n : ℕ)

-- Cyclical 3-subset definition
def cyclical_subset (A B C : ℕ) (beats : ℕ → ℕ → Prop) : Prop :=
  beats A B ∧ beats B C ∧ beats C A

-- Theorem for minimum number of cyclical 3-subsets
theorem min_cyclical_subsets_zero (beats : ℕ → ℕ → Prop) (h : ∀ i, ∃ j, beats (2 * n + 1) i j ∨ beats j i)
  : ∃ (sets : list ℕ), sets.length = 0 :=
begin
  sorry
end

-- Theorem for maximum number of cyclical 3-subsets
theorem max_cyclical_subsets (beats : ℕ → ℕ → Prop)
  : ∃ (sets : list ℕ), sets.length = (2 * n + 1) * n * (n + 1) / 6 :=
begin
  sorry
end

end min_cyclical_subsets_zero_max_cyclical_subsets_l639_639899


namespace range_of_a_non_monotonic_l639_639012

noncomputable def f (x a : ℝ) : ℝ := 2 * x ^ 2 + (x - a) * |x - a|

def is_not_monotonic_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∃ x₁ x₂ x₃ ∈ I, x₁ < x₂ < x₃ ∧ ¬ ((f x₁ ≤ f x₂ ∧ f x₂ ≤ f x₃) ∨ (f x₁ ≥ f x₂ ∧ f x₂ ≥ f x₃))

theorem range_of_a_non_monotonic {a : ℝ} :
  ¬ is_not_monotonic_on (λ x, f x a) (Set.Icc (-3 : ℝ) (0 : ℝ)) ↔ a ∈ Set.Ioo (-9 : ℝ) (0 : ℝ) ∨ Set.Ioo (0 : ℝ) (3 : ℝ) := 
sorry

end range_of_a_non_monotonic_l639_639012


namespace sum_of_solutions_eq_zero_l639_639323

theorem sum_of_solutions_eq_zero :
  ∀ x : ℝ, (-π ≤ x ∧ x ≤ 3 * π ∧ (1 / Real.sin x + 1 / Real.cos x = 4))
  → x = 0 := sorry

end sum_of_solutions_eq_zero_l639_639323


namespace eighteen_women_time_l639_639884

theorem eighteen_women_time (h : ∀ (n : ℕ), n = 6 → ∀ (t : ℕ), t = 60 → true) : ∀ (n : ℕ), n = 18 → ∀ (t : ℕ), t = 20 → true :=
by
  sorry

end eighteen_women_time_l639_639884


namespace maximum_value_of_f_is_5_l639_639357

noncomputable def f : ℝ → ℝ := sorry

def g (x : ℝ) : ℝ := f x - Real.logb 2 x

-- Conditions
axiom odd_f : ∀ (x : ℝ), f (-x) = -f x
axiom periodic_f : ∀ (x : ℝ), f (2 - x) = f x
axiom increasing_f : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 1 → f x₁ ≤ f x₂
axiom g_has_two_zeros : ∃ (a b : ℝ), a ≠ b ∧ g a = 0 ∧ g b = 0

-- The theorem to establish
theorem maximum_value_of_f_is_5 : ∃ M, ∀ x : ℝ, f x ≤ M ∧ M = Real.logb 2 (2^5) :=
  sorry

end maximum_value_of_f_is_5_l639_639357


namespace unique_perpendicular_line_l639_639643

noncomputable def line_perpendicular_to_plane (P : Point) (pl : Plane) : Prop :=
  ∃! l : Line, (passes_through P l ∧ perpendicular_to l pl)

theorem unique_perpendicular_line (P : Point) (pl : Plane) : line_perpendicular_to_plane P pl :=
by
  sorry

end unique_perpendicular_line_l639_639643


namespace triangle_inequality_l639_639880

theorem triangle_inequality (a b c : ℝ) (h : a < b + c) : a^2 - b^2 - c^2 - 2*b*c < 0 := by
  sorry

end triangle_inequality_l639_639880


namespace neg_p_neither_sufficient_nor_necessary_for_q_l639_639818

variable (x : ℝ)

def p := x ≥ 1
def q := (1 / x) < 1

theorem neg_p_neither_sufficient_nor_necessary_for_q :
  (¬ p) ↔ (¬ q) → false :=
by sorry

end neg_p_neither_sufficient_nor_necessary_for_q_l639_639818


namespace remainder_h_x_10_div_h_x_l639_639515

noncomputable def h (x : ℤ) : ℤ := x^5 - x^4 + x^3 - x^2 + x - 1

theorem remainder_h_x_10_div_h_x (x : ℤ) : polynomial.div_mod_by_monic (h (x)) (h (x)) (h (x^{10})) = (x, 5) :=
by
  -- Proof omitted.
  sorry

end remainder_h_x_10_div_h_x_l639_639515


namespace emily_total_spent_l639_639772

def total_cost (art_supplies_cost skirt_cost : ℕ) (number_of_skirts : ℕ) : ℕ :=
  art_supplies_cost + (skirt_cost * number_of_skirts)

theorem emily_total_spent :
  total_cost 20 15 2 = 50 :=
by
  sorry

end emily_total_spent_l639_639772


namespace find_b_l639_639832

noncomputable def ellipse_foci (a b : ℝ) (hb : b > 0) (hab : a > b) : Prop :=
∃ (F1 F2 P : ℝ×ℝ), 
    (∃ (h : a > b), (2 * b^2 + 9 = a^2)) ∧ 
    (dist P F1 + dist P F2 = 2 * a) ∧ 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧ 
    (2 * 4 * (a^2 - b^2) = 36)

theorem find_b (a b : ℝ) (hb : b > 0) (hab : a > b) : 
    ellipse_foci a b hb hab → b = 3 :=
by
  sorry

end find_b_l639_639832


namespace area_of_larger_hexagon_l639_639296

theorem area_of_larger_hexagon {A B C D E F G H I : Type} 
  [triangle ABC] 
  (side_length_triangle : ∀ (a b : ℝ), side_length_triangle(a) = 2 ∧ side_length_triangle(b) = 2)
  (equilateral : eq_triangle ABC)
  (side_length_hexagon : ∀ (s : ℝ), side_length_hexagon(s) = 2) :
  area_of_larger_hexagon DEFGHI = 36 * Real.sqrt 3 := 
by 
  sorry

end area_of_larger_hexagon_l639_639296


namespace area_of_grid_cells_within_circle_l639_639175

-- Definition of the grid and the circle properties
def circle (radius : ℝ) (center : ℝ × ℝ) := 
  { point : ℝ × ℝ | (point.1 - center.1) ^ 2 + (point.2 - center.2) ^ 2 ≤ radius ^ 2 }

def grid_cell (side_length : ℝ) (bottom_left : ℝ × ℝ) := 
  { point : ℝ × ℝ | bottom_left.1 ≤ point.1 ∧ point.1 < bottom_left.1 + side_length ∧ 
                    bottom_left.2 ≤ point.2 ∧ point.2 < bottom_left.2 + side_length }

-- Main theorem statement
theorem area_of_grid_cells_within_circle :
  let circle_radius := 1000
  let circle_center := (0, 0)
  let side_length := 1
  let circle_area := π * (1000:ℝ) ^ 2
  ∃ grid_cells : set (ℝ × ℝ),
    (∀ cell ∈ grid_cells, ∃ bottom_left : ℝ × ℝ, 
      (grid_cell side_length bottom_left ⊆ circle circle_radius circle_center)) ∧
    (set.sum (λ cell, set.measure (grid_cell side_length cell)) grid_cells) ≥ 0.99 * circle_area :=
sorry

end area_of_grid_cells_within_circle_l639_639175


namespace back_seat_capacity_l639_639022

theorem back_seat_capacity :
  let l_seats := 15 in
  let r_seats := l_seats - 3 in
  let ppl_per_seat := 3 in
  let total_capacity := 91 in
  let total_side_ppl := (l_seats + r_seats) * ppl_per_seat in
  total_capacity - total_side_ppl = 10 :=
by
  sorry

end back_seat_capacity_l639_639022


namespace solution_set_of_inequality_min_value_of_expression_l639_639853

def f (x : ℝ) : ℝ := |x + 1| - |2 * x - 2|

-- (I) Prove that the solution set of the inequality f(x) ≥ x - 1 is [0, 2]
theorem solution_set_of_inequality 
  (x : ℝ) : f x ≥ x - 1 ↔ 0 ≤ x ∧ x ≤ 2 := 
sorry

-- (II) Given the maximum value m of f(x) is 2 and a + b + c = 2, prove the minimum value of b^2/a + c^2/b + a^2/c is 2
theorem min_value_of_expression
  (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 2) :
  b^2 / a + c^2 / b + a^2 / c ≥ 2 :=
sorry

end solution_set_of_inequality_min_value_of_expression_l639_639853


namespace A_beats_B_by_20_seconds_l639_639459

-- Definitions and conditions
variables (A_time B_time A_speed B_speed t_50 : ℝ)

-- Given conditions
def condition_1 : Prop := A_time = 380
def condition_2 : Prop := ∃ T_B : ℝ, T_B / 380 = 1000 / 950

-- Statement to prove
theorem A_beats_B_by_20_seconds (h1 : condition_1) (h2 : condition_2) : B_time - A_time = 20 :=
by
  sorry

end A_beats_B_by_20_seconds_l639_639459


namespace geom_sequence_2010th_term_l639_639107

theorem geom_sequence_2010th_term (p q r a_n : ℕ) (h1 : ∀ p q r, 3p / q = 3 * p * q)
  (h2: p * r = 9) (h3: 9 * r = 3 * p / q) (h4: r = 9 / p):
  a_n = 9 :=
by
  sorry

end geom_sequence_2010th_term_l639_639107


namespace max_value_of_f_at_0_min_value_of_f_on_neg_inf_to_0_range_of_a_for_ineq_l639_639381

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 5*x + 5) / Real.exp x

theorem max_value_of_f_at_0 :
  f 0 = 5 := by
  sorry

theorem min_value_of_f_on_neg_inf_to_0 :
  f (-3) = -Real.exp 3 := by
  sorry

theorem range_of_a_for_ineq :
  ∀ x : ℝ, x^2 + 5*x + 5 - a * Real.exp x ≥ 0 ↔ a ≤ -Real.exp 3 := by
  sorry

end max_value_of_f_at_0_min_value_of_f_on_neg_inf_to_0_range_of_a_for_ineq_l639_639381


namespace truck_driver_speed_l639_639211

theorem truck_driver_speed (spends_per_gallon : ℝ) (miles_per_gallon : ℕ) (paid_per_mile : ℝ) (earned : ℝ) (hours_driven : ℕ)
  (h_spends : spends_per_gallon = 2) 
  (h_miles : miles_per_gallon = 10)
  (h_paid : paid_per_mile = 0.5)
  (h_earned : earned = 90)
  (h_hours : hours_driven = 10) : (earned / paid_per_mile) / hours_driven = 18 :=
by
  rw [h_spends, h_miles, h_paid, h_earned, h_hours]
  simp
  norm_num
  sorry

end truck_driver_speed_l639_639211


namespace num_three_digit_numbers_l639_639784

-- Definition of the problem
def digits := {0, 1, 3, 5, 6}
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def greaterThan200 (n : ℕ) : Prop := n > 200
def noRepeatedDigits (n : ℕ) : Prop := 
  let digitsList := List.repeat n.toString.toList.nth 
  digitsList.nodup

-- The three-digit numbers must use these digits (no repetitions) and be greater than 200
theorem num_three_digit_numbers : 
  ∃ (ns : Finset ℕ), (∀ n ∈ ns, isThreeDigit n ∧ greaterThan200 n ∧ noRepeatedDigits n) ∧ 
  ns.card = 36 := 
sorry

end num_three_digit_numbers_l639_639784


namespace count_integers_with_increasing_digits_l639_639422

theorem count_integers_with_increasing_digits :
  let count_integers := 
    ∑ second_digit in ({3, 4, 5} : Finset ℕ), 
      ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ second_digit}, 
        1 in

  count_integers = 15 :=
by
  have step1 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 3}, 1 = 6,
  { -- Explanation: If second digit is 3, third can be 4, 5, 6, 7, 8, 9 -> 6 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step2 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 4}, 1 = 5,
  { -- Explanation: If second digit is 4, third can be 5, 6, 7, 8, 9 -> 5 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step3 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 5}, 1 = 4,
  { -- Explanation: If second digit is 5, third can be 6, 7, 8, 9 -> 4 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have count_integers := step1 + step2 + step3,
  simp only [count_integers, add_comm],
  exact eq.refl 15

end count_integers_with_increasing_digits_l639_639422


namespace max_points_congruent_triangles_l639_639657

-- Definition of points and congruency
def Point := ℝ × ℝ
def Triangle (A B C : Point) := (A, B, C)

-- Definition of congruence between triangles
def congruent (t1 t2 : Triangle) : Prop := sorry
-- Placeholder for the proper definition of congruence of triangles 

def distinct (a b : Point) : Prop := a ≠ b

-- The proof statement
theorem max_points_congruent_triangles :
  ∃ (n : ℕ), n = 4 ∧
    ∃ (A B C D : Point) (X : Fin n → Point),
      distinct A B ∧ distinct C D ∧ 
      (∀ i, congruent (Triangle A B (X i)) (Triangle C D (X i))) :=
begin
  -- This is the place where the formal proof would go, for now, it is omitted with sorry.
  sorry
end

end max_points_congruent_triangles_l639_639657


namespace calculate_fraction_square_mul_l639_639263

theorem calculate_fraction_square_mul :
  ((8 / 9) ^ 2) * ((1 / 3) ^ 2) = 64 / 729 :=
by
  sorry

end calculate_fraction_square_mul_l639_639263


namespace max_frac_a_c_squared_l639_639508

theorem max_frac_a_c_squared 
  (a b c : ℝ) (y z : ℝ)
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order: a ≥ b ∧ b ≥ c)
  (h_system: a^2 + z^2 = c^2 + y^2 ∧ c^2 + y^2 = (a - y)^2 + (c - z)^2)
  (h_bounds: 0 ≤ y ∧ y < a ∧ 0 ≤ z ∧ z < c) :
  (a/c)^2 ≤ 4/3 :=
sorry

end max_frac_a_c_squared_l639_639508


namespace jason_plums_l639_639727

theorem jason_plums (total_plums alyssa_plums jason_plums : ℕ) (h1 : total_plums = 27) (h2 : alyssa_plums = 17) : jason_plums = total_plums - alyssa_plums :=
by
  have : jason_plums = 27 - 17, from congr_arg (λ x, x - 17) h1,
  sorry

end jason_plums_l639_639727


namespace find_integer_n_l639_639780

theorem find_integer_n 
  (h₀ : -180 ≤ (-72 : ℤ)) 
  (h₁ : (-72 : ℤ) ≤ 180) 
  (h₂ : ∀ n : ℤ, -180 ≤ n ∧ n ≤ 180 → (sin (n : ℝ)) = cos 522) 
  : ∃ n : ℤ, sin (n : ℝ) = cos 522 ∧ n = -72 :=
begin
  use -72,
  split,
  { exact h₂ (-72) ⟨h₀, h₁⟩ },
  { refl }
end

end find_integer_n_l639_639780


namespace sum_first_10_terms_arith_seq_l639_639951

theorem sum_first_10_terms_arith_seq (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 3 = 5)
  (h2 : a 7 = 13)
  (h3 : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h4 : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  S 10 = 100 :=
sorry

end sum_first_10_terms_arith_seq_l639_639951


namespace Petya_receives_last_wrapper_l639_639398

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem Petya_receives_last_wrapper
  (h1 : discriminant a b c ≥ 0)
  (h2 : discriminant a c b ≥ 0)
  (h3 : discriminant b a c ≥ 0)
  (h4 : discriminant c a b < 0)
  (h5 : discriminant b c a < 0) :
  discriminant c b a ≥ 0 :=
sorry

end Petya_receives_last_wrapper_l639_639398


namespace recipe_calls_for_cups_of_sugar_l639_639547

variable (totalSugar : ℕ)

def cups_of_sugar (alreadyPut : ℕ) (needsToAdd : ℕ) := alreadyPut + needsToAdd

theorem recipe_calls_for_cups_of_sugar
  (alreadyPut : ℕ) 
  (needsToAdd : ℕ) 
  (recipeTotal : ℕ) :
  cups_of_sugar alreadyPut needsToAdd = recipeTotal :=
by
  sorry

-- Instantiation
example : recipe_calls_for_cups_of_sugar 4 3 7 := by sorry

end recipe_calls_for_cups_of_sugar_l639_639547


namespace chromium_percentage_in_second_alloy_l639_639906

theorem chromium_percentage_in_second_alloy
  (x : ℝ)
  (h1 : chromium_percentage_in_first_alloy = 15)
  (h2 : weight_first_alloy = 15)
  (h3 : weight_second_alloy = 35)
  (h4 : chromium_percentage_in_new_alloy = 10.1)
  (h5 : total_weight = weight_first_alloy + weight_second_alloy)
  (h6 : chromium_in_new_alloy = chromium_percentage_in_new_alloy / 100 * total_weight)
  (h7 : chromium_in_first_alloy = chromium_percentage_in_first_alloy / 100 * weight_first_alloy)
  (h8 : chromium_in_second_alloy = x / 100 * weight_second_alloy)
  (h9 : chromium_in_new_alloy = chromium_in_first_alloy + chromium_in_second_alloy) :
  x = 8 := by
  sorry

end chromium_percentage_in_second_alloy_l639_639906


namespace total_milk_production_l639_639100

theorem total_milk_production
  (a b c d e f : ℝ)
  (h_ac_pos : a * c > 0)
  (h_f_pos : f > 0) :
  let initial_rate := b / (a * c),
      productivity_decline := (1 - f / 100),
      total_production := d * initial_rate * (100 / f) * (1 - productivity_decline^e) in
  total_production = d * (b / (a * c)) * (100 / f) * (1 - (1 - f / 100)^e) :=
by
  sorry

end total_milk_production_l639_639100


namespace math_problem_l639_639394

noncomputable def proof_problem : Prop := 
  let x : ℤ := 2
  let y : ℤ := 1
  (2*x + y = 5 ∧ x - y = 1) →
  ∃ a b : ℤ, (a = -2) ∧ (b = 3) ∧ (a*x + 3*y = -1) ∧ (4*x + b*y = 11)

-- The problem we need to prove
theorem math_problem : proof_problem :=
begin
  sorry
end

end math_problem_l639_639394


namespace pascal_ratio_2_3_4_exists_row_l639_639898

theorem pascal_ratio_2_3_4_exists_row :
  ∃ n k : ℕ, (choose n k * 3 = choose n (k + 1) * 2) ∧ (choose n (k + 1) * 4 = choose n (k + 2) * 3) ∧ n = 34 :=
by
  sorry

end pascal_ratio_2_3_4_exists_row_l639_639898


namespace bus_waiting_probability_l639_639187

-- Definitions
def arrival_time_range := (0, 90)  -- minutes from 1:00 to 2:30
def bus_wait_time := 20             -- bus waits for 20 minutes

noncomputable def probability_bus_there_when_Laura_arrives : ℚ :=
  let total_area := 90 * 90
  let trapezoid_area := 1400
  let triangle_area := 200
  (trapezoid_area + triangle_area) / total_area

-- Theorem statement
theorem bus_waiting_probability : probability_bus_there_when_Laura_arrives = 16 / 81 := by
  sorry

end bus_waiting_probability_l639_639187


namespace speed_of_train_A_is_90_kmph_l639_639146

-- Definitions based on the conditions
def train_length_A := 225 -- in meters
def train_length_B := 150 -- in meters
def crossing_time := 15 -- in seconds

-- The total distance covered by train A to cross train B
def total_distance := train_length_A + train_length_B

-- The speed of train A in m/s
def speed_in_mps := total_distance / crossing_time

-- Conversion factor from m/s to km/hr
def mps_to_kmph (mps: ℕ) := mps * 36 / 10

-- The speed of train A in km/hr
def speed_in_kmph := mps_to_kmph speed_in_mps

-- The theorem to be proved
theorem speed_of_train_A_is_90_kmph : speed_in_kmph = 90 := by
  -- Proof steps go here
  sorry

end speed_of_train_A_is_90_kmph_l639_639146


namespace pages_per_hour_l639_639125

-- Definitions corresponding to conditions
def lunch_time : ℕ := 4 -- time taken to grab lunch and come back (in hours)
def total_pages : ℕ := 4000 -- total pages in the book
def book_time := 2 * lunch_time  -- time taken to read the book is twice the lunch_time

-- Statement of the problem to be proved
theorem pages_per_hour : (total_pages / book_time = 500) := 
  by
    -- We assume the definitions and want to show the desired property
    sorry

end pages_per_hour_l639_639125


namespace divisibility_equiv_l639_639819

open Nat

theorem divisibility_equiv (m n : ℕ) : 
  (2^n - 1) % ((2^m - 1)^2) = 0 ↔ n % (m * (2^m - 1)) = 0 := 
sorry

end divisibility_equiv_l639_639819


namespace abs_quadratic_bound_l639_639833

theorem abs_quadratic_bound (a b : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + a * x + b) :
  (|f 1| ≥ (1 / 2)) ∨ (|f 2| ≥ (1 / 2)) ∨ (|f 3| ≥ (1 / 2)) :=
by
  sorry

end abs_quadratic_bound_l639_639833


namespace cost_of_fruits_l639_639340

-- Define variables and functions
variables (a b c d e : ℕ)

-- Define the conditions
def condition1 : Prop := a + b + c + d + e = 30
def condition2 : Prop := d = 2 * a
def condition3 : Prop := c = a - b
def condition4 : Prop := e = a + b

-- Define the goal
def goal : Prop := b + c + e = 12

-- Create the theorem statement
theorem cost_of_fruits (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : goal :=
by sorry

end cost_of_fruits_l639_639340


namespace final_song_count_l639_639724

theorem final_song_count {init_songs added_songs removed_songs doubled_songs final_songs : ℕ} 
    (h1 : init_songs = 500)
    (h2 : added_songs = 500)
    (h3 : doubled_songs = (init_songs + added_songs) * 2)
    (h4 : removed_songs = 50)
    (h_final : final_songs = doubled_songs - removed_songs) : 
    final_songs = 2950 :=
by
  sorry

end final_song_count_l639_639724


namespace sum_f_1_to_2014_eq_sqrt_3_l639_639807

def f (x : ℕ) : ℝ :=
  Math.sin ((Real.pi / 3) * (x + 1)) - (Real.sqrt 3) * Math.cos ((Real.pi / 3) * (x + 1))

theorem sum_f_1_to_2014_eq_sqrt_3 : 
  ∑ k in Finset.range 2014, f (k + 1) = Real.sqrt 3 :=
by 
  -- The proof steps would go here.
  sorry

end sum_f_1_to_2014_eq_sqrt_3_l639_639807


namespace rational_unique_representation_l639_639991

theorem rational_unique_representation (p q : ℤ) (h_rational : q ≠ 0) (h_nonzero : p ≠ 0):
  ∃ (n : ℕ) (x : ℕ → ℤ), (0 < n) ∧ (x n ≠ 0) ∧ (∀ k, 2 ≤ k → k ≤ n → 0 ≤ x k ∧ x k < k) ∧
  (p/q : ℚ) = (x 1 + ∑ k in finset.range (n+1), (x k / (k!))) := 
sorry

end rational_unique_representation_l639_639991


namespace jordon_machine_input_l639_639937

theorem jordon_machine_input (x : ℝ) : (3 * x - 6) / 2 + 9 = 27 → x = 14 := 
by
  sorry

end jordon_machine_input_l639_639937


namespace problem_statement_l639_639761

theorem problem_statement :
  ∀ k : Nat, (∃ r s : Nat, r > 0 ∧ s > 0 ∧ (k^2 - 6 * k + 11)^(r - 1) = (2 * k - 7)^s) ↔ (k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 8) :=
by
  sorry

end problem_statement_l639_639761


namespace rowan_distance_downstream_l639_639084

-- Conditions
def speed_still : ℝ := 9.75
def downstream_time : ℝ := 2
def upstream_time : ℝ := 4

-- Statement to prove
theorem rowan_distance_downstream : ∃ (d : ℝ) (c : ℝ), 
  d / (speed_still + c) = downstream_time ∧
  d / (speed_still - c) = upstream_time ∧
  d = 26 := by
    sorry

end rowan_distance_downstream_l639_639084


namespace rectangle_area_l639_639621

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l639_639621


namespace distribute_cookies_l639_639470

theorem distribute_cookies :
  (∑ i in Finset.range 5, (3 + 0)) ≤ 30 →
  (∑ i in Finset.range 5, y i) + 15 = 30 →
  (∑ i in Finset.range 5, y i) = 15 →
  nat.choose (15 + 5 - 1) (5 - 1) = 3876 :=
by {
  sorry
}

end distribute_cookies_l639_639470


namespace digit_inequality_l639_639583

theorem digit_inequality :
  {d : ℕ | d < 10 ∧ 3.1 + 0.01 * d + 0.003 > 3.123}.card = 7 :=
by
  sorry

end digit_inequality_l639_639583


namespace complex_number_imaginary_l639_639371

theorem complex_number_imaginary (z : ℂ) (hz : Im z ≠ 0) (hz1: Im ((z + 2)^2 + 5) = 0) : z = 3 * complex.I ∨ z = -3 * complex.I :=
sorry

end complex_number_imaginary_l639_639371


namespace hyperbola_imaginary_axis_length_l639_639857

open Real

-- Definitions for the problem
def distance_from_focus_to_asymptote (b : ℝ) : ℝ := 
  b * sqrt (4 + b^2)

-- The condition that the distance from foci to asymptote is 3
axiom distance_condition (b : ℝ) (hb : b > 0) : distance_from_focus_to_asymptote b = 3

-- The proof statement for the length of the imaginary axis being 6
theorem hyperbola_imaginary_axis_length (b : ℝ) (hb : b > 0) 
  (h_dist : distance_from_focus_to_asymptote b = 3) : 
  2 * b = 6 :=
sorry

end hyperbola_imaginary_axis_length_l639_639857


namespace sqrt_of_product_of_powers_l639_639752

theorem sqrt_of_product_of_powers :
  (Real.sqrt (4^2 * 5^6) = 500) :=
by
  sorry

end sqrt_of_product_of_powers_l639_639752


namespace sum_of_valid_b_values_of_quadratic_with_rational_roots_l639_639763

theorem sum_of_valid_b_values_of_quadratic_with_rational_roots :
  let b_set := {b : ℕ | ∃ k : ℕ, 49 - 12 * b = k^2 ∧ 49 > k^2}
  ∑ b in b_set, b = 6 :=
begin
  sorry
end

end sum_of_valid_b_values_of_quadratic_with_rational_roots_l639_639763


namespace primes_with_ones_digit_three_l639_639444

theorem primes_with_ones_digit_three (n : ℕ) (prime : ℕ → Prop) (h1 : n < 200) (h2 : prime n) (h3 : n % 10 = 3) : 
  ∃ p_list, p_list.length = 12 ∧ ∀ p ∈ p_list, p < 200 ∧ prime p ∧ p % 10 = 3 := 
sorry

end primes_with_ones_digit_three_l639_639444


namespace max_profit_l639_639208

-- Define the given conditions
def cost_price : ℝ := 80
def sales_relationship (x : ℝ) : ℝ := -0.5 * x + 160
def selling_price_range (x : ℝ) : Prop := 120 ≤ x ∧ x ≤ 180

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_relationship x

-- The goal: prove the maximum profit and the selling price that achieves it
theorem max_profit : ∃ (x : ℝ), selling_price_range x ∧ profit x = 7000 := 
  sorry

end max_profit_l639_639208


namespace tan_beta_eq_one_seventh_l639_639827

theorem tan_beta_eq_one_seventh
  (α β : Real)
  (h1 : π / 2 < α ∧ α < π)
  (h2 : sin α = 3 / Real.sqrt 10)
  (h3 : tan (α + β) = -2) :
  tan β = 1 / 7 := sorry

end tan_beta_eq_one_seventh_l639_639827


namespace zero_not_nec_nor_suff_l639_639121

variable {α : Type*} [LinearOrderedField α]

def is_extreme_value (f : α → α) (x₀ : α) : Prop :=
(∀ δ : α, δ > 0 → f(x₀) ≤ f(x₀ + δ)) ∧ (∀ δ : α, δ > 0 → f(x₀) ≤ f(x₀ - δ)) ∨ 
(∀ δ : α, δ > 0 → f(x₀) ≥ f(x₀ + δ)) ∧ (∀ δ : α, δ > 0 → f(x₀) ≥ f(x₀ - δ))

theorem zero_not_nec_nor_suff {f : α → α} {x₀ : α}
  (hf : Differentiable α f) (hf_ext: is_extreme_value f x₀) :
  ¬(Necessary (f x₀ = 0) (is_extreme_value f x₀)) ∧ ¬(Sufficient (f x₀ = 0) (is_extreme_value f x₀)) :=
by
  sorry

end zero_not_nec_nor_suff_l639_639121


namespace max_remainder_square_mod_5_l639_639870

theorem max_remainder_square_mod_5 (n : ℤ) : ∃ m ∈ {0, 1, 4}, m ≤ n^2 % 5 := sorry

end max_remainder_square_mod_5_l639_639870


namespace bounces_count_l639_639684

-- Define the conditions
def α : ℝ := 19.94
def β : ℝ := α / 10
def AB_eq_BC : Prop := True  -- Since AB = BC is given and doesn't need further proof

-- Define the correct answer
def number_of_bounces : ℕ := 71

-- The proof statement
theorem bounces_count (AB_eq_BC : AB_eq_BC) (hα : α = 19.94) (hβ : β = α / 10) : 
    number_of_bounces = 71 := 
sorry

end bounces_count_l639_639684


namespace problem1_problem2_l639_639360

-- Define the points A, B, and C
def A := (3, 0)
def B := (0, 3)
def C (α : ℝ) := (Real.cos α, Real.sin α)

-- Define the interval for α
def in_interval (α : ℝ) : Prop := α > Real.pi / 2 ∧ α < 3 * Real.pi / 2

-- Problem (1) to prove α = 5π / 4
theorem problem1 (α : ℝ) (h_interval : in_interval α)
  (h_condition1 : Real.sqrt ((3 - Real.cos α)^2 + (0 - Real.sin α)^2) = Real.sqrt ((0 - Real.cos α)^2 + (3 - Real.sin α)^2)) :
  α = 5 * Real.pi / 4 :=
sorry

-- Problem (2) to prove 2sin²α + sin 2α / 1 + tan α = -5 / 9
theorem problem2 (α : ℝ) (h_interval : in_interval α)
  (h_condition2 : (Real.cos α - 3, Real.sin α) ⋅ (Real.cos α, Real.sin α - 3) = -1) :
  (2 * (Real.sin α)^2 + Real.sin (2 * α)) / (1 + Real.tan α) = -5 / 9 :=
sorry

end problem1_problem2_l639_639360


namespace number_of_solutions_l639_639443

def satisfied_conditions (a b : ℤ) : Prop :=
  a^2 + b^2 < 25 ∧ a^2 + b^2 < 10 * a ∧ a^2 + b^2 < 10 * b

theorem number_of_solutions : 
  (finset.univ.filter (λ (pair : ℤ × ℤ), satisfied_conditions pair.1 pair.2)).card = 9 :=
by {
  sorry
}

end number_of_solutions_l639_639443


namespace part_I_part_II_part_III_l639_639846

-- Define the function f and its derivative
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x^2

def f' (x : ℝ) (a b : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

-- Proof of existence of tangent line at point (2, f(2))
theorem part_I (a b : ℝ) :
  (f 2 a b = 6 * 2 + 3 * f 2 a b - 10) ∧
  (f' 2 a b = -2) →
  a = -1 / 3 ∧ b = 1 / 2 :=
sorry

-- Proof of minimum value of k
theorem part_II (a b k : ℝ) 
  (h1 : f' x a b ≤ k * log(x + 1) for all x ∈ [0, ∞)) :
  ∃ k_min, k_min = 1 :=
sorry

-- Proof of the summation inequality
theorem part_III (n : ℕ) 
  (h2 : n > 0) :
  ∑ i in range (n + 1), 1 / i.to_real < log (n + 1) + 2 :=
sorry

end part_I_part_II_part_III_l639_639846


namespace product_of_conjugates_l639_639539

theorem product_of_conjugates (z1 z2 : ℂ) (h1 : z1 = 1 + √3 * Complex.i) (h2 : z2 = Complex.conj z1) : z1 * z2 = 4 :=
by
  sorry

end product_of_conjugates_l639_639539


namespace harmonic_mean_pairs_l639_639600

theorem harmonic_mean_pairs : 
  {p : (ℕ × ℕ) // p.1 ≠ p.2 ∧ (2 * p.1 * p.2) / (p.1 + p.2) = 4 ^ 15}.card = 29 :=
by sorry

end harmonic_mean_pairs_l639_639600


namespace problem_2021_CCA_Bonanza_11_l639_639737

noncomputable def areaRatio := (x_1 x_2 x_3 y_1 y_2 y_3 : ℝ) 
  (hx_sum : x_1 + x_2 + x_3 = 3)
  (hy_sum : y_1 + y_2 + y_3 = 3)
  (hx_cubic_rel : x_1^3 + x_2^3 + x_3^3 = 3 * x_1 * x_2 * x_3 + 20)
  (hy_cubic_rel : y_1^3 + y_2^3 + y_3^3 = 3 * y_1 * y_2 * y_3 + 21)
  (h_ratio : 3/4/5 ∈ ({x_1, x_2, x_3}, {y_1, y_2, y_3})) : ℝ :=
    (82, 25)

theorem problem_2021_CCA_Bonanza_11 
  (x_1 x_2 x_3 y_1 y_2 y_3 : ℝ)
  (hx_sum : x_1 + x_2 + x_3 = 3)
  (hy_sum : y_1 + y_2 + y_3 = 3)
  (hx_cubic_rel : x_1^3 + x_2^3 + x_3^3 = 3 * x_1 * x_2 * x_3 + 20)
  (hy_cubic_rel : y_1^3 + y_2^3 + y_3^3 = 3 * y_1 * y_2 * y_3 + 21)
  (h_ratio : 3 / 4 / 5 ∈ ({x_1, x_2, x_3}, {y_1, y_2, y_3})) :
  m + n = 107 :=
begin
  have area := areaRatio x_1 x_2 x_3 y_1 y_2 y_3 hx_sum hy_sum hx_cubic_rel hy_cubic_rel h_ratio,
  sorry
end

end problem_2021_CCA_Bonanza_11_l639_639737


namespace transportation_inverse_proportion_l639_639588

theorem transportation_inverse_proportion (V t : ℝ) (h: V * t = 10^5) : V = 10^5 / t :=
by
  sorry

end transportation_inverse_proportion_l639_639588


namespace coordinate_of_B_l639_639024

noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem coordinate_of_B (O A : ℝ × ℝ) (C : ℝ × ℝ) :
  O = (0, 0) →
  A = (4, 3) →
  C.1 > 0 ∧ C.2 < 0 →
  side_length O A = 5 ∧
  side_length A C = 5 ∧
  side_length O C = 5 →
  ∃ B : ℝ × ℝ, B = (7, -1) :=
  by
  intros hO hA hC_eq h_len
  use (7, -1)
  sorry

end coordinate_of_B_l639_639024


namespace inequality_of_positive_reals_l639_639522

variable {a b c : ℝ}

theorem inequality_of_positive_reals (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ 3 / 2 :=
sorry

end inequality_of_positive_reals_l639_639522


namespace find_n_l639_639317

theorem find_n (n : ℤ) (h : 0 ≤ n ∧ n ≤ 180) : 
  (cos (n : ℝ) * (real.pi / 180)) = cos (1010 * (real.pi / 180)) → n = 70 := 
sorry

end find_n_l639_639317


namespace acute_isosceles_triangle_k_l639_639249

theorem acute_isosceles_triangle_k (ABC : Triangle) (circ : Circle)
  (D : Point)
  (h1 : ABC.angles.B = ABC.angles.C) -- Isosceles property
  (h2 : ∀ P ∈ circ, is_tangent B P circ) -- Tangent property through B
  (h3 : ∀ Q ∈ circ, is_tangent C Q circ) -- Tangent property through C
  (h4 : angle ABC.angles.B = 3 * angle D )
  (h5 : ∃ k, angle ABC.angles.A = k * π ) :
  ∃ k, k = 5 / 11 :=
by
  sorry

end acute_isosceles_triangle_k_l639_639249


namespace triangle_base_angles_eq_l639_639067

theorem triangle_base_angles_eq
  (A B C C1 C2 : ℝ)
  (h1 : A > B)
  (h2 : C1 = 2 * C2)
  (h3 : A + B + C = 180)
  (h4 : B + C2 = 90)
  (h5 : C = C1 + C2) :
  A = B := by
  sorry

end triangle_base_angles_eq_l639_639067


namespace find_G_in_terms_of_F_l639_639502

def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

def G (x : ℝ) : ℝ := log ((1 + x * (1 + x^2) / (1 + x^4)) / (1 - x * (1 + x^2) / (1 + x^4)))

theorem find_G_in_terms_of_F (x : ℝ) : G x = 2 * F x := 
by sorry

end find_G_in_terms_of_F_l639_639502


namespace wendy_furniture_time_l639_639150

theorem wendy_furniture_time (chairs tables minutes_per_piece : ℕ) 
    (h_chairs : chairs = 4) 
    (h_tables : tables = 4) 
    (h_minutes_per_piece : minutes_per_piece = 6) : 
    chairs + tables * minutes_per_piece = 48 := 
by 
    sorry

end wendy_furniture_time_l639_639150


namespace cone_projection_trig_lemma_l639_639143

section cone_angle_proof

variables {α β varphi x : ℝ} (h_alpha : 0 < α)
(h_beta : α < β) (h_angle : β < 180)

theorem cone_projection_trig_lemma :
  sin varphi = sin (α / 2) / sin (β / 2) ∧ 
  sin (x / 2) = sin (β / 2) * sin (varphi / 2) := 
by 
  sorry

end cone_angle_proof

end cone_projection_trig_lemma_l639_639143


namespace arc_length_BA_l639_639476

-- Define a circle with center O and radius OB.
variable {O B A : Point} (r : ℝ) (circ : Circle O r)

-- Assume the given conditions.
axiom angle_BOA_measures_45 : measure_angle O B A = 45
axiom OB_is_12 : dist O B = 12

-- The proof statement.
theorem arc_length_BA : (arc_length circ B A) = 6 * π :=
by
  -- Use conditions and axioms to show the proof
  sorry

end arc_length_BA_l639_639476


namespace polynomial_coefficients_l639_639860

theorem polynomial_coefficients (a_0 a_1 a_2 a_3 a_4 : ℤ) :
  (∀ x : ℤ, (x + 2)^5 = (x + 1)^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_0 = 31 ∧ a_1 = 75 :=
by
  sorry

end polynomial_coefficients_l639_639860


namespace car_speed_ratio_l639_639722

-- Assuming the bridge length as L, pedestrian's speed as v_p, and car's speed as v_c.
variables (L v_p v_c : ℝ)

-- Mathematically equivalent proof problem statement in Lean 4.
theorem car_speed_ratio (h1 : 2/5 * L = 2/5 * L)
                       (h2 : (L - 2/5 * L) / v_p = L / v_c) :
    v_c = 5 * v_p := 
  sorry

end car_speed_ratio_l639_639722


namespace k_value_correct_l639_639241

-- Let k be the value such that ∠BAC = k * π
def k : ℝ := 5 / 11

-- Define the isosceles triangle ABC inscribed in a circle
variables (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
variables [tangents_through B C D] (angle_BAC ABC ACB D : ℝ)

# Define the angles:
-- ∠ABC = ∠ACB = 3 * ∠D
axiom angle_equivalence : (ABC = 3 * D) ∧ (ACB = 3 * D)
-- ∠BAC = k * π
axiom angle_BAC_def : angle_BAC = k * real.pi

-- The proof assertion:
theorem k_value_correct (h : k = 5/11) : 
  (∀ (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
     [tangents_through B C D] (angle_BAC ABC ACB D : ℝ),
  angle_equivalence → angle_BAC_def) → 
  angle_BAC = (5 / 11 : ℝ) * real.pi :=
by
  sorry

end k_value_correct_l639_639241


namespace k_value_correct_l639_639242

-- Let k be the value such that ∠BAC = k * π
def k : ℝ := 5 / 11

-- Define the isosceles triangle ABC inscribed in a circle
variables (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
variables [tangents_through B C D] (angle_BAC ABC ACB D : ℝ)

# Define the angles:
-- ∠ABC = ∠ACB = 3 * ∠D
axiom angle_equivalence : (ABC = 3 * D) ∧ (ACB = 3 * D)
-- ∠BAC = k * π
axiom angle_BAC_def : angle_BAC = k * real.pi

-- The proof assertion:
theorem k_value_correct (h : k = 5/11) : 
  (∀ (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
     [tangents_through B C D] (angle_BAC ABC ACB D : ℝ),
  angle_equivalence → angle_BAC_def) → 
  angle_BAC = (5 / 11 : ℝ) * real.pi :=
by
  sorry

end k_value_correct_l639_639242


namespace count_valid_palindromic_tables_correct_l639_639184

def palindromic_table (table : Array (Array Char)) : Prop :=
  table.size = 3 ∧ (∀ i, i < 3 → table[i].size = 3) ∧
  ∀ i j, i < 3 → j < 3 → table[i][j] = table[i][2-j] ∧ table[i][j] = table[2-i][j]

def is_valid_letter (c : Char) : Prop :=
  c = 'O' ∨ c = 'M'

def valid_table (table : Array (Array Char)) : Prop :=
  (∀ i j, (i < 3 ∧ j < 3) → is_valid_letter (table[i][j])) ∧ palindromic_table table

noncomputable def count_valid_palindromic_tables : Nat :=
  (Array.foldr (. + .) 0 (Array.map (λ a => 
    Array.foldr (. + .) 0 (Array.map (λ b => 
      Array.foldr (. + .) 0 (Array.map (λ d => 
        Array.foldr (. + .) 0 (Array.map (λ e => 
          if valid_table #[#[a, b, a], #[d, e, d], #[a, b, a]] then 1 else 0) ['O', 'M'])) ['O', 'M'])) ['O', 'M'])) ['O', 'M']))

theorem count_valid_palindromic_tables_correct :
  count_valid_palindromic_tables = 16 := sorry

end count_valid_palindromic_tables_correct_l639_639184


namespace correct_propositions_B_and_D_l639_639380

-- Definitions and Conditions
def is_centroid (G : Triangle → Point) : Prop :=
  ∀ △ABC, let G := centroid △ABC in
  dist G A * 2 = dist G (midpoint B C)

def is_incenter_centroid_circumcenter_on_altitude (T : Triangle) : Prop :=
  is_isosceles T → 
  ∃ I G O : Point, incenter T = I ∧ centroid T = G ∧ circumcenter T = O ∧ 
  collinear [I, G, O] ∧ altitude T

def right_triangle_area (T : Triangle) (a b c R : ℝ) : Prop :=
  a^2 + b^2 = c^2 → circumradius T = R → 
  R = c / 2 → 
  a = 3 → b = 4 → c = 5 → 
  area T = 24

def incenter_ratio_area (T : Triangle) (A B C : Angle) (I : Point) : Prop :=
  ∠A = 30 → ∠B = 60 → ∠C = 90 → incenter T = I → 
  (S △ IAB) : (S △ IBC) : (S △ IAC) = 2 : 1 : sqrt 3

-- Proof problem as Lean 4 statement
theorem correct_propositions_B_and_D 
  (T : Triangle) (A B C : Angle) (I G O : Point) : 
  is_incenter_centroid_circumcenter_on_altitude T ∧
  incenter_ratio_area T A B C I →
  true := 
by 
  intro,
  sorry

end correct_propositions_B_and_D_l639_639380


namespace angle_comparison_l639_639969

theorem angle_comparison (A B C M : Point) (MC_perp_plane_ABC : MC ⊥ plane ABC) : 
  ∠AMB > ∠ACB :=
sorry

end angle_comparison_l639_639969


namespace acute_isosceles_triangle_k_l639_639250

theorem acute_isosceles_triangle_k (ABC : Triangle) (circ : Circle)
  (D : Point)
  (h1 : ABC.angles.B = ABC.angles.C) -- Isosceles property
  (h2 : ∀ P ∈ circ, is_tangent B P circ) -- Tangent property through B
  (h3 : ∀ Q ∈ circ, is_tangent C Q circ) -- Tangent property through C
  (h4 : angle ABC.angles.B = 3 * angle D )
  (h5 : ∃ k, angle ABC.angles.A = k * π ) :
  ∃ k, k = 5 / 11 :=
by
  sorry

end acute_isosceles_triangle_k_l639_639250


namespace jill_marathon_time_l639_639488

def jack_marathon_distance : ℝ := 42
def jack_marathon_time : ℝ := 6
def speed_ratio : ℝ := 0.7

theorem jill_marathon_time :
  ∃ t_jill : ℝ, (t_jill = jack_marathon_distance / (jack_marathon_distance / jack_marathon_time / speed_ratio)) ∧
  t_jill = 4.2 :=
by
  -- The proof goes here
  sorry

end jill_marathon_time_l639_639488


namespace find_k_l639_639217

-- Definitions based on conditions in step a)
def acute_isosceles_triangle_inscribed (A B C : Type) : Prop := sorry -- Formal definition of the triangle being acute isosceles and inscribed in a circle
def tangents_meeting_at_point (A B C D : Type) : Prop := sorry -- Formal definition of tangents through B and C meeting at D
def angle_relation (ABC D : Type) (theta : ℝ) : Prop := 3 * theta = sorry -- Formal definition of \(\angle ABC = \angle ACB = 3 \angle D\)
def angle_BAC (k : ℝ) (theta : ℝ) : Prop := theta = k * real.pi -- Formal definition of \(\angle BAC = k \pi\)

-- Theorem statement for our proof problem
theorem find_k
  (A B C D : Type)
  (h1 : acute_isosceles_triangle_inscribed A B C)
  (h2 : tangents_meeting_at_point A B C D)
  (theta : ℝ)
  (h3 : angle_relation ABC D theta)
  (k : ℝ)
  (h4 : angle_BAC k theta) :
  k = 1 / 13 := by
  sorry

end find_k_l639_639217


namespace a_n_expression_l639_639814

open Nat

-- Definitions and assumptions
def a (n : ℕ) : ℝ := sorry -- Sequence a_n
def S (n : ℕ) : ℝ := ∑ i in range n, a i -- Sum of the first n terms, S_n
def b (n : ℕ) : ℝ := S n + n * a n -- The arithmetic sequence S_n + n * a_n
axiom a_one_half : a 0 = 1 / 2 -- Given a = 1 / 2
axiom arithmetic_sequence : ∀ n, b n - b (n - 1) = b (n + 1) - b n

noncomputable def desired_a (n : ℕ) : ℝ :=
  1 / (n * (n + 1))

-- The theorem
theorem a_n_expression (n : ℕ) : a n = desired_a n := sorry

end a_n_expression_l639_639814


namespace similar_triangles_legs_l639_639702

theorem similar_triangles_legs (y : ℝ) (h : 12 / y = 9 / 7) : y = 84 / 9 := by
  sorry

end similar_triangles_legs_l639_639702


namespace car_speed_ratio_to_pedestrian_speed_l639_639721

noncomputable def ratio_of_speeds (length_pedestrian_car: ℝ) (length_bridge: ℝ) (speed_pedestrian: ℝ) (speed_car: ℝ) : ℝ :=
  speed_car / speed_pedestrian

theorem car_speed_ratio_to_pedestrian_speed
  (L : ℝ)  -- length of the bridge
  (vc vp : ℝ)  -- speeds of car and pedestrian respectively
  (h1 : (2 / 5) * L + (3 / 5) * vp = L)
  (h2 : (3 / 5) * L = vc * (L / (5 * vp))) :
  ratio_of_speeds L L vp vc = 5 :=
by
  sorry

end car_speed_ratio_to_pedestrian_speed_l639_639721


namespace perpendicular_vectors_l639_639868

variable {t : ℝ}

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

theorem perpendicular_vectors (ht : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) : t = -5 :=
sorry

end perpendicular_vectors_l639_639868


namespace total_number_of_people_l639_639087

theorem total_number_of_people : ∃ (x : ℕ), (12 * x + 3 = 13 * x - 12) ∧ x = 15 :=
by
  existsi 15
  split
  sorry
  rfl

end total_number_of_people_l639_639087


namespace price_of_shirt_l639_639134

theorem price_of_shirt (T S : ℝ) 
  (h1 : T + S = 80.34) 
  (h2 : T = S - 7.43) : 
  T = 36.455 :=
by
  sorry

end price_of_shirt_l639_639134


namespace piece_50_squares_union_pieces_1_to_50_squares_sum_first_50_even_numbers_sum_first_100_numbers_l639_639278

theorem piece_50_squares : 
  (∀ n, number_of_squares_in_piece n = 1 + 2 * (n - 1)) → 
  number_of_squares_in_piece 50 = 99 :=
by
  intro h
  simp [h]
  sorry

theorem union_pieces_1_to_50_squares : 
  (∀ n, number_of_squares_in_piece n = 1 + 2 * (n - 1)) → 
  (∀ m, total_squares_in_first_m_pieces m = m * m) → 
  total_squares_in_first_m_pieces 50 = 2500 :=
by
  intros h1 h2
  simp [h2]
  sorry

theorem sum_first_50_even_numbers (h : ∑ i in Finset.range 50, 2 * (i + 1) = 2550) : 
  ∑ i in Finset.range 50, 2 * (i + 1) = 2550 :=
by
  exact h

theorem sum_first_100_numbers : 
  ∑ i in Finset.range 100, (i + 1) = 5050 :=
by
  sorry

end piece_50_squares_union_pieces_1_to_50_squares_sum_first_50_even_numbers_sum_first_100_numbers_l639_639278


namespace paint_price_and_max_boxes_l639_639572

theorem paint_price_and_max_boxes (x y m : ℕ) 
  (h1 : x + 2 * y = 56) 
  (h2 : 2 * x + y = 64) 
  (h3 : 24 * m + 16 * (200 - m) ≤ 3920) : 
  x = 24 ∧ y = 16 ∧ m ≤ 90 := 
  by 
    sorry

end paint_price_and_max_boxes_l639_639572


namespace sixth_number_of_11_consecutive_odd_sum_1991_is_181_l639_639892

theorem sixth_number_of_11_consecutive_odd_sum_1991_is_181 :
  (∃ (n : ℤ), (2 * n + 1) + (2 * n + 3) + (2 * n + 5) + (2 * n + 7) + (2 * n + 9) + (2 * n + 11) + (2 * n + 13) + (2 * n + 15) + (2 * n + 17) + (2 * n + 19) + (2 * n + 21) = 1991) →
  2 * 85 + 11 = 181 := 
by
  sorry

end sixth_number_of_11_consecutive_odd_sum_1991_is_181_l639_639892


namespace angle_between_vectors_l639_639957

noncomputable theory

def p : ℝ × ℝ × ℝ := (2, -3, 4)
def q : ℝ × ℝ × ℝ := (real.sqrt 3, 5, -2)
def r : ℝ × ℝ × ℝ := (-7, 3, 10)

-- Function to compute dot product
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Function to perform scalar multiplication
def scalar_mul (a : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a * v.1, a * v.2, a * v.3)

-- Function to perform vector subtraction
def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

-- Define the target vector
def target_vector : ℝ × ℝ × ℝ :=
  vector_sub (scalar_mul (dot_product p r) q) (scalar_mul (dot_product p q) r)

-- Proof that the dot product of p and target_vector is zero
theorem angle_between_vectors : dot_product p target_vector = 0 :=
by
  sorry

end angle_between_vectors_l639_639957


namespace rectangle_area_l639_639627

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l639_639627


namespace f_minus_g_odd_f_minus_g_pos_set_l639_639856

open Real

-- Definitions of the functions
noncomputable def f (a x : ℝ) : ℝ := log a (x + 1)
noncomputable def g (a x : ℝ) : ℝ := log a (1 - x)

-- Theorem stating f(x) - g(x) is odd
theorem f_minus_g_odd (a : ℝ) (x : ℝ) (h_a : a > 0 ∧ a ≠ 1 ∧ -1 < x ∧ x < 1) : 
  f a x - g a x = - (f a (-x) - g a (-x)) :=
sorry

-- Theorem for the solution set of f(x) - g(x) > 0
theorem f_minus_g_pos_set (a : ℝ) (x : ℝ) (h_a : a > 0 ∧ a ≠ 1 ∧ -1 < x ∧ x < 1) : 
  (0 < a ∧ a < 1 ∧ -1 < x ∧ x < 0) ∨ (1 < a ∧ 0 < x ∧ x < 1) :=
sorry

end f_minus_g_odd_f_minus_g_pos_set_l639_639856


namespace max_value_of_6_f_x_plus_2012_l639_639794

noncomputable def f (x : ℝ) : ℝ :=
  min (min (4*x + 1) (x + 2)) (-2*x + 4)

theorem max_value_of_6_f_x_plus_2012 : ∃ x : ℝ, 6 * f x + 2012 = 2028 :=
sorry

end max_value_of_6_f_x_plus_2012_l639_639794


namespace unique_control_l639_639455

theorem unique_control (n : ℕ)
  (lights switches : fin n → bool)
  (control : fin n → fin n)
  (h_control : ∀ i : fin n, ∀ s : fin n → bool,
                   let s' := function.update s i (¬ s i) in
                   lights i ≠ lights (control i) → lights (control i)^= ¬ lights (control i)) :
  ∀ j : fin n, ∃ i : fin n, ∀ s : fin n → bool, 
    let s' := function.update s i (¬ s i) in 
    lights j s' = ¬ lights j s  :=
begin
  sorry
end

end unique_control_l639_639455


namespace a_100_eq_7_pow_99_l639_639759

-- Define the sequence as per the conditions
def a : ℕ → ℝ
| 0     := 1
| (n+1) := 7 * a n

-- State the theorem to prove
theorem a_100_eq_7_pow_99: a 99 = 7 ^ 99 := sorry

end a_100_eq_7_pow_99_l639_639759


namespace suitable_option_is_D_not_suitable_option_A_not_suitable_option_B_not_suitable_option_C_l639_639122

-- Definitions based on provided conditions
inductive SurveyOption
| A | B | C | D

-- Defining the property of a suitable option for a comprehensive survey
def suitable_for_comprehensive_survey (opt : SurveyOption) : Prop :=
  match opt with
  | SurveyOption.A => False
  | SurveyOption.B => False
  | SurveyOption.C => False
  | SurveyOption.D => True 

-- The theorem to state that Option D is the suitable one
theorem suitable_option_is_D : suitable_for_comprehensive_survey SurveyOption.D :=
by
  -- Explicitly showing the truth value based on the property definition
  unfold suitable_for_comprehensive_survey
  exact true.intro

-- Using the properties to show others are unsuitable
theorem not_suitable_option_A : ¬ suitable_for_comprehensive_survey SurveyOption.A :=
by
  unfold suitable_for_comprehensive_survey
  exact id

theorem not_suitable_option_B : ¬ suitable_for_comprehensive_survey SurveyOption.B :=
by
  unfold suitable_for_comprehensive_survey
  exact id

theorem not_suitable_option_C : ¬ suitable_for_comprehensive_survey SurveyOption.C :=
by
  unfold suitable_for_comprehensive_survey
  exact id

end suitable_option_is_D_not_suitable_option_A_not_suitable_option_B_not_suitable_option_C_l639_639122


namespace groceries_spent_l639_639718

/-- Defining parameters from the conditions provided -/
def rent : ℝ := 5000
def milk : ℝ := 1500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 700
def savings_rate : ℝ := 0.10
def savings : ℝ := 1800

/-- Adding an assertion for the total spent on groceries -/
def groceries : ℝ := 4500

theorem groceries_spent (total_salary total_expenses : ℝ) :
  total_salary = savings / savings_rate →
  total_expenses = rent + milk + education + petrol + miscellaneous →
  groceries = total_salary - (total_expenses + savings) :=
by
  intros h_salary h_expenses
  sorry

end groceries_spent_l639_639718


namespace percentage_increase_l639_639212

theorem percentage_increase (new_wage original_wage : ℝ) (h₁ : new_wage = 42) (h₂ : original_wage = 28) :
  ((new_wage - original_wage) / original_wage) * 100 = 50 := by
  sorry

end percentage_increase_l639_639212


namespace number_of_rows_l639_639209

theorem number_of_rows (S a d : ℕ) (h1 : a = 3) (h2 : d = 4) (h3 : S = 361) : ℕ :=
let n := (2*n^2 + n = S) in
n

end number_of_rows_l639_639209


namespace rectangle_area_eq_16sqrt3_l639_639767

theorem rectangle_area_eq_16sqrt3
  (M N P Q : Point)
  (R S : Line)
  (PQ_len : dist P Q = 8)
  (PT TU UQ : ℝ)
  (H_PT : PT = 2)
  (H_TU : TU = 2)
  (H_UQ : UQ = 2)
  (H_R_perpendicular : is_perpendicular R PQ)
  (H_S_perpendicular : is_perpendicular S PQ)
  (H_M_lies_on_R : lies_on M R)
  (H_Q_lies_on_S : lies_on Q S) :
  area (rectangle MNPQ) = 16 * sqrt 3 :=
sorry

end rectangle_area_eq_16sqrt3_l639_639767


namespace price_of_paint_models_max_boxes_of_paint_A_l639_639576

-- Define the conditions as hypotheses
def price_paint_model_A_B (x y : ℕ) : Prop :=
  x + 2 * y = 56 ∧ 2 * x + y = 64

def total_cost_constraint (x y m : ℕ) : Prop :=
  24 * m + 16 * (200 - m) ≤ 3920

-- Prove the prices of the paint models given the conditions
theorem price_of_paint_models :
  ∃ (x y : ℕ), price_paint_model_A_B x y ∧ x = 24 ∧ y = 16 :=
by
  sorry

-- Prove the maximum number of boxes of paint model A the school can purchase given the total cost constraint
theorem max_boxes_of_paint_A :
  ∃ (m : ℕ), total_cost_constraint 24 16 m ∧ m = 90 :=
by
  sorry

end price_of_paint_models_max_boxes_of_paint_A_l639_639576


namespace theater_seat_count_l639_639696

theorem theater_seat_count :
  let odd_rows := 6
  let even_rows := 5
  let seats_in_odd_rows := 15
  let seats_in_even_rows := 16
  (odd_rows * seats_in_odd_rows) + (even_rows * seats_in_even_rows) = 170 :=
by
  let odd_rows := 6
  let even_rows := 5
  let seats_in_odd_rows := 15
  let seats_in_even_rows := 16
  have odd_seats := odd_rows * seats_in_odd_rows
  have even_seats := even_rows * seats_in_even_rows
  have total_seats := odd_seats + even_seats
  show total_seats = 170
  sorry

end theater_seat_count_l639_639696


namespace domain_of_f_2x_minus_1_l639_639386

noncomputable def domain_of_shifted_function (f : ℝ → ℝ) (x : ℝ) : set ℝ :=
{x | ∃ y ∈ [-2, 3], y = x + 1}

noncomputable def domain_of_scaled_shifted_function (f : ℝ → ℝ) (x : ℝ) : set ℝ :=
{x | ∃ y ∈ [-1, 4], y = 2*x - 1}

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x ∈ [-2, 3], f(x + 1) ≠ none) →
  (domain_of_scaled_shifted_function f = set.Icc 0 (5/2)) :=
by
  sorry

end domain_of_f_2x_minus_1_l639_639386


namespace find_x_plus_y_l639_639902

-- Define the segments and their lengths
def segment_AB_length := 5 : ℝ
def segment_A'B'_length := 8 : ℝ

-- Define the midpoints
def midpoint_AB := segment_AB_length / 2
def midpoint_A'B' := segment_A'B'_length / 2

-- Define the distances x and y
variables (x y : ℝ)

-- Define the total path constraint
def total_path_condition := midpoint_AB + x + y + midpoint_A'B' = segment_AB_length + segment_A'B'_length

-- Prove that x + y is 6.5 given the conditions
theorem find_x_plus_y (h : total_path_condition x y) : x + y = 6.5 := by
  have h1 : midpoint_AB = 2.5 := by norm_cast; refl
  have h2 : midpoint_A'B' = 4 := by norm_cast; refl
  rw [h1, h2] at h
  linarith

#print axioms find_x_plus_y

end find_x_plus_y_l639_639902


namespace p_polynomial_l639_639786

noncomputable def p : ℝ → ℝ := sorry

theorem p_polynomial (x y : ℝ) :
  p 3 = 10 ∧ (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) → p = λ x, x^2 + 1 := sorry

end p_polynomial_l639_639786


namespace first_five_consecutive_even_digits_position_l639_639739

-- Definition of the concatenated sequence of natural numbers as one string.
def natural_concatenated_string : String := 
String.join $ List.map Nat.toDigits (List.range' 1 1000)

-- Definition to check if a digit is even
def is_even (n : Char) : Bool := 
n == '0' ∨ n == '2' ∨ n == '4' ∨ n == '6' ∨ n == '8'

-- Find the position of the first occurrence of five consecutive even digits.
noncomputable def first_even_sequence_position : Option ℕ :=
let seq := natural_concatenated_string.toList in
seq.indexOfList (List.replicate 5 ('2' : Char) ++ List.replicate 5 ('0' : Char))

-- The main theorem statement proving the position is 490 
theorem first_five_consecutive_even_digits_position : first_even_sequence_position = some 490 := 
sorry

end first_five_consecutive_even_digits_position_l639_639739


namespace solve_ineqs_l639_639096

theorem solve_ineqs (a x : ℝ) (h1 : |x - 2 * a| ≤ 3) (h2 : 0 < x + a ∧ x + a ≤ 4) 
  (ha : a = 3) (hx : x = 1) : 
  (|x - 2 * a| ≤ 3) ∧ (0 < x + a ∧ x + a ≤ 4) :=
by
  sorry

end solve_ineqs_l639_639096


namespace count_integer_points_l639_639988

theorem count_integer_points (T : ℕ) :
  T = { (x, y) | x^2 + y^2 < 10 ∧ x ∈ ℤ ∧ y ∈ ℤ }.to_finset.card :=
by
  sorry

end count_integer_points_l639_639988


namespace arithmetic_sequence_common_difference_l639_639911

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h₁ : a 2 = 9) (h₂ : a 5 = 33) :
  ∀ d : ℤ, (∀ n : ℕ, a n = a 1 + (n - 1) * d) → d = 8 :=
by
  -- We state the theorem and provide a "sorry" proof placeholder
  sorry

end arithmetic_sequence_common_difference_l639_639911


namespace min_value_of_expression_l639_639866

theorem min_value_of_expression 
  (a b : ℝ)
  (C1 : ∀ (x y : ℝ), x^2 + y^2 = 4)
  (C2 : ∀ (x y : ℝ), (x - 1)^2 + (y - 3)^2 = 4)
  (h : ∀ (P : ℝ × ℝ), P ∈ {P | ∃ (M N : ℝ × ℝ), |dist P M = dist P N|}) :
  ∃ P : ℝ × ℝ, (a, b) ∈ {P | a^2 + b^2 - 6a - 4b + 13 ≥ 0} ∧ min (a^2 + b^2 - 6a - 4b + 13) = 8 / 5 := sorry

end min_value_of_expression_l639_639866


namespace jerome_speed_l639_639487

theorem jerome_speed (t_J t_N: ℝ) (N J: ℝ) (h1: t_J = 6) 
  (h2: t_N = 3) (h3: N = 8) (h4: J * t_J = N * t_N) : 
  J = 4 :=
by
  -- Definitions based on conditions
  rw [h1, h2, h3] at h4
  -- Simplify the equation
  have h5: J * 6 = 8 * 3, from h4
  have h6: J * 6 = 24, by rw h5
  have h7: J = 24 / 6, by linarith
  show J = 4, by linarith

end jerome_speed_l639_639487


namespace count_integers_in_range_with_increasing_digits_l639_639410

theorem count_integers_in_range_with_increasing_digits : 
  let integers_in_range := { n | 200 ≤ n ∧ n < 250 ∧ ((n % 10), (n / 10 % 10), (n / 100)) = (d₀, d₁, d₂) ∧ d₀ < d₁ ∧ d₁ < d₂ } in
  integers_in_range.card = 34 :=
sorry

end count_integers_in_range_with_increasing_digits_l639_639410


namespace find_area_of_triangle_ABQ_l639_639033

noncomputable def area_triangle_ABQ {A B C P Q R : Type*}
  (AP PB : ℝ) (area_ABC area_ABQ : ℝ) (h_areas_equal : area_ABQ = 15 / 2)
  (h_triangle_area : area_ABC = 15) (h_AP : AP = 3) (h_PB : PB = 2)
  (h_AB : AB = AP + PB) : Prop := area_ABQ = 15

theorem find_area_of_triangle_ABQ
  (A B C P Q R : Type*) (AP PB : ℝ)
  (h_triangle_area : area_ABC = 15)
  (h_AP : AP = 3) (h_PB : PB = 2)
  (h_AB : AB = AP + PB) (h_areas_equal : area_ABQ = 15 / 2) :
  area_ABQ = 15 := sorry

end find_area_of_triangle_ABQ_l639_639033


namespace triangle_square_side_length_ratio_l639_639735

theorem triangle_square_side_length_ratio (t s : ℝ) (ht : 3 * t = 12) (hs : 4 * s = 12) : 
  t / s = 4 / 3 :=
by
  sorry

end triangle_square_side_length_ratio_l639_639735


namespace simplify_trig_expression_l639_639578

theorem simplify_trig_expression (x y : ℝ) :
  sin x ^ 2 + sin (x + y) ^ 2 - 2 * sin x * sin y * sin (x + y) = cos y ^ 2 := 
by sorry

end simplify_trig_expression_l639_639578


namespace log_strictly_increasing_on_neg_infty_to_neg3_l639_639286

def log_base_one_third (t : ℝ) := real.log t / real.log (1/3)

theorem log_strictly_increasing_on_neg_infty_to_neg3 :
  ∀ x : ℝ, x < -3 → log_base_one_third (x^2 - 9) is strictly increasing :=
sorry

end log_strictly_increasing_on_neg_infty_to_neg3_l639_639286


namespace count_three_digit_values_with_double_sum_eq_six_l639_639952

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_three_digit (x : ℕ) : Prop := 
  100 ≤ x ∧ x < 1000

theorem count_three_digit_values_with_double_sum_eq_six :
  ∃ count : ℕ, is_three_digit count ∧ (
    (∀ x, is_three_digit x → sum_of_digits (sum_of_digits x) = 6) ↔ count = 30
  ) :=
sorry

end count_three_digit_values_with_double_sum_eq_six_l639_639952


namespace base8_perfect_square_c_eq_one_l639_639008

theorem base8_perfect_square_c_eq_one (a b c : ℕ) (h1 : a ≠ 0) (h2 : a < 8) (h3 : b < 8) (h4 : c < 8) :
  let n := 512 * a + 64 * b + 24 + c in
  ∃ k : ℕ, n = k^2 → c = 1 :=
by
  sorry

end base8_perfect_square_c_eq_one_l639_639008


namespace max_profit_l639_639633

-- Define the conditions
def profit (m : ℝ) := (m - 8) * (900 - 15 * m)

-- State the theorem
theorem max_profit (m : ℝ) : 
  ∃ M, M = profit 34 ∧ ∀ x, profit x ≤ M :=
by
  -- the proof goes here
  sorry

end max_profit_l639_639633


namespace count_integers_with_increasing_digits_200_to_250_l639_639429

theorem count_integers_with_increasing_digits_200_to_250 :
  {n : ℕ | 200 ≤ n ∧ n ≤ 250 ∧ 
          (let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 in 
           d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
           d1 < d2 ∧ d2 < d3)}.card = 11 :=
by
  sorry

end count_integers_with_increasing_digits_200_to_250_l639_639429


namespace coeff_b_l639_639599

noncomputable def g (a b c d e : ℝ) (x : ℝ) :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem coeff_b (a b c d e : ℝ):
  -- The function g(x) has roots at x = -1, 0, 1, 2
  (g a b c d e (-1) = 0) →
  (g a b c d e 0 = 0) →
  (g a b c d e 1 = 0) →
  (g a b c d e 2 = 0) →
  -- The function passes through the point (0, 3)
  (g a b c d e 0 = 3) →
  -- Assuming a = 1
  (a = 1) →
  -- Prove that b = -2
  b = -2 :=
by
  intros _ _ _ _ _ a_eq_1
  -- Proof omitted
  sorry

end coeff_b_l639_639599


namespace k_value_correct_l639_639246

-- Let k be the value such that ∠BAC = k * π
def k : ℝ := 5 / 11

-- Define the isosceles triangle ABC inscribed in a circle
variables (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
variables [tangents_through B C D] (angle_BAC ABC ACB D : ℝ)

# Define the angles:
-- ∠ABC = ∠ACB = 3 * ∠D
axiom angle_equivalence : (ABC = 3 * D) ∧ (ACB = 3 * D)
-- ∠BAC = k * π
axiom angle_BAC_def : angle_BAC = k * real.pi

-- The proof assertion:
theorem k_value_correct (h : k = 5/11) : 
  (∀ (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
     [tangents_through B C D] (angle_BAC ABC ACB D : ℝ),
  angle_equivalence → angle_BAC_def) → 
  angle_BAC = (5 / 11 : ℝ) * real.pi :=
by
  sorry

end k_value_correct_l639_639246


namespace least_non_lucky_multiple_of_12_l639_639695

/- Defines what it means for a number to be a lucky integer -/
def isLucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

/- Proves the least positive multiple of 12 that is not a lucky integer is 96 -/
theorem least_non_lucky_multiple_of_12 : ∃ n, n % 12 = 0 ∧ ¬isLucky n ∧ ∀ m, m % 12 = 0 ∧ ¬isLucky m → n ≤ m :=
  by
  sorry

end least_non_lucky_multiple_of_12_l639_639695


namespace quadratic_specific_a_l639_639010

noncomputable def quadratic_root_condition (a : ℝ) : Prop :=
  ∃ x : ℝ, (a + 2) * x^2 + 2 * a * x + 1 = 0

theorem quadratic_specific_a (a : ℝ) (h : quadratic_root_condition a) :
  a = 2 ∨ a = -1 :=
sorry

end quadratic_specific_a_l639_639010


namespace find_negative_number_l639_639215

theorem find_negative_number :
  let a := -(-2)
  let b := -| -2 |
  let c := (-2)^2
  let d := 1 - (-2)
  b < 0 ∧ a ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 :=
by
  let a := -(-2)
  let b := -| -2 |
  let c := (-2)^2
  let d := 1 - (-2)
  split
  . exact b_neg
  . split
    . exact a_nonneg
    . split
      . exact c_nonneg
      . exact d_nonneg
  sorry

end find_negative_number_l639_639215


namespace count_integers_in_range_with_increasing_digits_l639_639408

theorem count_integers_in_range_with_increasing_digits : 
  let integers_in_range := { n | 200 ≤ n ∧ n < 250 ∧ ((n % 10), (n / 10 % 10), (n / 100)) = (d₀, d₁, d₂) ∧ d₀ < d₁ ∧ d₁ < d₂ } in
  integers_in_range.card = 34 :=
sorry

end count_integers_in_range_with_increasing_digits_l639_639408


namespace percentage_deficit_for_second_side_l639_639027

-- Defining the given conditions and the problem statement
def side1_excess : ℚ := 0.14
def area_error : ℚ := 0.083
def original_length (L : ℚ) := L
def original_width (W : ℚ) := W
def measured_length_side1 (L : ℚ) := (1 + side1_excess) * L
def measured_width_side2 (W : ℚ) (x : ℚ) := W * (1 - 0.01 * x)
def original_area (L W : ℚ) := L * W
def calculated_area (L W x : ℚ) := 
  measured_length_side1 L * measured_width_side2 W x

theorem percentage_deficit_for_second_side (L W : ℚ) :
  (calculated_area L W 5) / (original_area L W) = 1 + area_error :=
by
  sorry

end percentage_deficit_for_second_side_l639_639027


namespace cube_face_area_l639_639636

-- Definition for the condition of the cube's surface area
def cube_surface_area (s : ℝ) : Prop := s = 36

-- Definition stating a cube has 6 faces
def cube_faces : ℝ := 6

-- The target proposition to prove
theorem cube_face_area (s : ℝ) (area_of_one_face : ℝ) (h1 : cube_surface_area s) (h2 : cube_faces = 6) : area_of_one_face = s / 6 :=
by
  sorry

end cube_face_area_l639_639636


namespace find_neg_a_l639_639099

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 0 then -x else 3 * x - 50

theorem find_neg_a :
  ∃ a : ℝ, a < 0 ∧ g (g (g 15)) = g (g (g a)) ∧ a = -55 / 3 :=
by
  have h1 : g 15 = -5 := by sorry
  have h2 : g (-5) = 5 := by sorry
  have h3 : g 5 = -35 := by sorry
  have h4 : g (g (g 15)) = -35 := by sorry
  use -55 / 3
  split
  · sorry -- prove that -55/3 is negative
  split
  · sorry -- sub-goal to show g(g(g(-55/3))) = -35
  · sorry -- sub-goal to establish a = -55/3

end find_neg_a_l639_639099


namespace fraction_traditionalists_l639_639169

theorem fraction_traditionalists (P T : ℝ) 
  (H_divided : ∀ (T : ℝ), T = P / 15) 
  (H_provinces : ∃ (n : ℝ), n = 5) :
  (5 * T) / ((P + 5 * T) / 3) = 1 / 4 :=
by
  intro P T H_divided H_provinces
  sorry

end fraction_traditionalists_l639_639169


namespace base_9_subtract_base_8_add_base_7_l639_639262

theorem base_9_subtract_base_8_add_base_7 :
  let n9 := 321
  let n8 := 256
  let n7 := 134
  (9^2 * 3 + 9^1 * 2 + 9^0 * 1) - (8^2 * 2 + 8^1 * 5 + 8^0 * 6) + (7^2 * 1 + 7^1 * 3 + 7^0 * 4) = 162 := 
by
  let n9 := 3 * 9^2 + 2 * 9^1 + 1 * 9^0
  let n8 := 2 * 8^2 + 5 * 8^1 + 6 * 8^0
  let n7 := 1 * 7^2 + 3 * 7^1 + 4 * 7^0
  calc
  (n9 - n8 + n7) = 262 - 174 + 74 := sorry
                ... = 162 := sorry

end base_9_subtract_base_8_add_base_7_l639_639262


namespace count_increasing_order_digits_l639_639417

def count_valid_numbers : ℕ :=
  let numbers := [234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249] in
  numbers.length

theorem count_increasing_order_digits : count_valid_numbers = 11 :=
by
  sorry

end count_increasing_order_digits_l639_639417


namespace max_profit_l639_639634

theorem max_profit (m : ℝ) :
  (m - 8) * (900 - 15 * m) = -15 * (m - 34) ^ 2 + 10140 :=
by
  sorry

end max_profit_l639_639634


namespace math_problem_l639_639156

open Nat

def gcd' (a b : ℕ) := gcd a b
def lcm' (a b : ℕ) := lcm a b
def sum_gcd_lcm (a b : ℕ) := gcd' a b + lcm' a b
def product_gcd_lcm (a b : ℕ) := gcd' a b * lcm' a b

theorem math_problem (a b : ℕ) (h₀ : gcd' a b = 84) (h₁ : lcm' a b = 3780) : 
  sum_gcd_lcm a b * product_gcd_lcm a b = 1227194880 :=
by
  sorry

end math_problem_l639_639156


namespace sum_squares_bound_l639_639532

theorem sum_squares_bound (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) : 
  ab + bc + cd + da ≤ 4 :=
sorry

end sum_squares_bound_l639_639532


namespace sqrt_product_l639_639753

-- Define the radicals and their product
def rad1 := Real.sqrt 72
def rad2 := Real.sqrt 27
def rad3 := Real.sqrt 8

-- Theorem statement for the given problem
theorem sqrt_product : rad1 * rad2 * rad3 = 72 * Real.sqrt 3 :=
by
  -- With given definitions, you can prove this theorem
  sorry

end sqrt_product_l639_639753


namespace collinear_D_E_F_iff_OH_eq_2R_l639_639968

-- Definition of the geometric entities involved
variables (A B C H O : Point) (R : ℝ)
variables (BC CA AB : Line)
variables (D E F : Point)

-- Conditions
axiom orthocenter_def (ABC : Triangle) : H = orthocenter ABC
axiom circumcenter_def (ABC : Triangle) : O = circumcenter ABC
axiom circumradius_def (ABC : Triangle) : R = circumradius ABC
axiom reflection_D (A BC : Point) : D = reflection A BC
axiom reflection_E (B CA : Point) : E = reflection B CA
axiom reflection_F (C AB : Point) : F = reflection C AB

-- Proof Theorem Statement
theorem collinear_D_E_F_iff_OH_eq_2R
  (ABC : Triangle) (H = orthocenter ABC)
  (O = circumcenter ABC) (R = circumradius ABC)
  (D = reflection A (side BC ABC))
  (E = reflection B (side CA ABC))
  (F = reflection C (side AB ABC))
  : collinear D E F ↔ dist O H = 2 * R :=
by sorry

end collinear_D_E_F_iff_OH_eq_2R_l639_639968


namespace solve_x_minus_y_l639_639141

theorem solve_x_minus_y :
  (2 = 0.25 * x) → (2 = 0.1 * y) → (x - y = -12) :=
by
  sorry

end solve_x_minus_y_l639_639141


namespace find_k_in_isosceles_triangle_l639_639224

theorem find_k_in_isosceles_triangle 
  (A B C D : Type)
  (h_triangle : triangle ABC)
  (h_acute : acute ABC)
  (h_isosceles : isosceles ABC)
  (h_circumscribed : circumscribed ABC)
  (h_tangents : tangents B C D)
  (h_angle_relation : ∠ABC = ∠ACB ∧ ∠ABC = 3 * ∠D)
  : ∠BAC = (5 * π / 11) := 
by 
  sorry

end find_k_in_isosceles_triangle_l639_639224


namespace divisors_of_fact8_divisible_by_12_l639_639405

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

-- noncomputable because calculating big factorials explicitly
noncomputable def fact8 : ℕ := factorial 8

theorem divisors_of_fact8_divisible_by_12 : 
  let divisors_count_by_12 := 
    (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1) / (1 + 1) in
  divisors_count_by_12 = 48 :=
by
  sorry

end divisors_of_fact8_divisible_by_12_l639_639405


namespace johns_profit_l639_639935

noncomputable def selling_price : ℝ := 2
noncomputable def num_newspapers : ℕ := 500
noncomputable def sell_fraction : ℝ := 0.80
noncomputable def buy_discount : ℝ := 0.75

def buying_price_per_newspaper : ℝ := selling_price * (1 - buy_discount)
def total_cost : ℝ := num_newspapers * buying_price_per_newspaper
def num_sold_newspapers : ℕ := (sell_fraction * num_newspapers.to_real).to_nat
def revenue : ℝ := num_sold_newspapers * selling_price
def profit : ℝ := revenue - total_cost

theorem johns_profit :
  profit = 550 := by
  sorry

end johns_profit_l639_639935


namespace problem_statement_l639_639050

noncomputable def a := Real.tan (9 * Real.pi / 8)
noncomputable def b := 2^(1 / 3 : ℝ)
noncomputable def c := Real.log 3 / Real.log 2

theorem problem_statement : a < b ∧ b < c :=
by
  sorry

end problem_statement_l639_639050


namespace count_two_digit_prime_numbers_l639_639876

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def valid_digits : List ℕ := [1, 3, 7, 9]

def two_digit_numbers (tens units : ℕ) : ℕ :=
  10 * tens + units

def valid_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ is_prime n

noncomputable def count_valid_two_digit_primes : ℕ :=
  (valid_digits.product valid_digits).filter (λ (x : ℕ × ℕ), x.1 ≠ x.2 ∧ valid_two_digit_prime (two_digit_numbers x.1 x.2)).length

theorem count_two_digit_prime_numbers :
  count_valid_two_digit_primes = 9 :=
by sorry

end count_two_digit_prime_numbers_l639_639876


namespace general_term_a_general_term_b_sum_of_first_n_c_l639_639349

-- Definition of the sequence a_n
def sequence_a (a : ℕ → ℕ) : Prop :=
∀ m n : ℕ, a (n + m) = a n + a m

-- Proof that a_n = n
theorem general_term_a (a : ℕ → ℕ) (h : sequence_a a) : ∀ n : ℕ, a n = n :=
sorry

-- Definition of the sequence b_n
def sequence_b (b a : ℕ → ℕ) : Prop :=
b 1 = a 1 ∧ b 2 = 2 * a 1 ∧ b 3 = 3 * a 1 + 1

-- Proof that b_n = 2^(n-1)
theorem general_term_b (a b : ℕ → ℕ) (h1 : sequence_a a) (h2 : sequence_b b a) : ∀ n : ℕ, b n = 2^(n - 1) :=
sorry

-- Definition of the sequence c_n
def sequence_c (a b : ℕ → ℕ) (c : ℕ → ℕ) : Prop :=
∀ n, c n = a n / b n

-- Proof that T_n = (n-1)*2^n + 1 is the sum of the first n terms of c_n
theorem sum_of_first_n_c (a b c : ℕ → ℕ) (h1 : sequence_a a) (h2 : sequence_b b a) (h3 : sequence_c a b c) :
 ∀ n, (∑ k in Finset.range n + 1, c (k + 1)) = (n - 1) * 2^n + 1 :=
sorry

end general_term_a_general_term_b_sum_of_first_n_c_l639_639349


namespace find_m_plus_n_l639_639311

-- Conditions in Lean definitions
def num_adults : ℕ := 15

def valid_pairing_cycles (pairing : List (List ℕ)) : Prop :=
  ∀ k < 7, ¬∃ S : Finset (Fin num_adults), S.card = k ∧ 
    ∀ i ∈ S, (pairing.nth i) = some [i]

def total_pairings := (fact num_adults)

def bad_pairings := (fact (num_adults - 1)) + (Nat.choose num_adults 7 * (fact 6) * (fact 7)) / 2

def probability_valid_pairing : QP := 
  (total_pairings - bad_pairings) / total_pairings

-- Equivalent statement with the question
theorem find_m_plus_n (m n : ℕ) (h1 : probability_valid_pairing = (m / n)) (h2 : Nat.coprime m n) :
  m + n = 16 :=
sorry

end find_m_plus_n_l639_639311


namespace not_square_or_cube_l639_639516

theorem not_square_or_cube (n : ℕ) (h : n > 1) : 
  ¬ (∃ a : ℕ, 2^n - 1 = a^2) ∧ ¬ (∃ a : ℕ, 2^n - 1 = a^3) :=
by
  sorry

end not_square_or_cube_l639_639516


namespace line_intersects_curve_ap_aq_product_l639_639908

-- Definitions based on the conditions
def line_l_param (t : ℝ) : ℝ × ℝ := (1 + sqrt 2 * t, sqrt 2 * t)

def curve_C_polar (ρ θ : ℝ) : Prop :=
  3 * ρ^2 * (cos θ)^2 + 4 * ρ^2 * (sin θ)^2 = 12

def curve_C_cartesian (x y : ℝ) : Prop :=
  3 * x^2 + 4 * y^2 = 12

def line_l_general (x y : ℝ) : Prop := 
  x - y - 1 = 0

def point_A (A : ℝ × ℝ) : Prop :=
  A = (1, 0)

-- The final proof statement
theorem line_intersects_curve_ap_aq_product : 
  (∃ P Q : ℝ × ℝ, 
    curve_C_cartesian (P.1) (P.2) ∧ line_l_general (P.1) (P.2) ∧
    curve_C_cartesian (Q.1) (Q.2) ∧ line_l_general (Q.1) (Q.2)) →
  (∃ A : ℝ × ℝ, point_A A) →
  (|1 - 1|) * (sqrt 2 * (P.1 - 1)) * (sqrt 2 * (Q.1 - 1)) = 18 / 7 := 
by
  sorry

end line_intersects_curve_ap_aq_product_l639_639908


namespace median_of_scores_is_85_l639_639900

theorem median_of_scores_is_85 (scores : List ℕ) (h : scores = [65, 78, 86, 91, 85]) : 
  List.median [65, 78, 86, 91, 85] = 85 :=
by 
  -- Skipping the proof by using sorry
  sorry

end median_of_scores_is_85_l639_639900


namespace angle_of_inclination_l639_639837

/--
Given the direction vector of line l as (-sqrt(3), 3),
prove that the angle of inclination α of line l is 120 degrees.
-/
theorem angle_of_inclination (α : ℝ) :
  let direction_vector : Real × Real := (-Real.sqrt 3, 3)
  let slope := direction_vector.2 / direction_vector.1
  slope = -Real.sqrt 3 → α = 120 :=
by
  sorry

end angle_of_inclination_l639_639837


namespace area_of_PUTS_l639_639995

-- Definitions based on conditions
def length_QR := 10
def width_QR  := 5
def area_PQRS := length_QR * width_QR

def point_T_mid_QR := length_QR / 2
def point_U_mid_RS := length_QR / 2

def triangle_area (base height : ℕ) := (base * height) / 2

def area_PQT := triangle_area length_QR width_QR
def area_PSU := triangle_area (length_QR / 2) width_QR

-- The theorem to prove
theorem area_of_PUTS : area_PQRS - area_PQT - area_PSU = 12.5 := 
by 
  sorry

end area_of_PUTS_l639_639995


namespace repeating_decmials_sum_is_fraction_l639_639307

noncomputable def x : ℚ := 2/9
noncomputable def y : ℚ := 2/99
noncomputable def z : ℚ := 2/9999

theorem repeating_decmials_sum_is_fraction :
  (x + y + z) = 2426 / 9999 := by
  sorry

end repeating_decmials_sum_is_fraction_l639_639307


namespace shark_population_change_l639_639280

def initialSharkPopulation := 65
def newportSharks := 22
def danaPointSharks := 4 * newportSharks
def huntingtonSharks := 1 / 2 * danaPointSharks
def totalCurrentSharkPopulation := newportSharks + danaPointSharks + huntingtonSharks
def percentageChange := ((totalCurrentSharkPopulation - initialSharkPopulation) / initialSharkPopulation) * 100

theorem shark_population_change :
  percentageChange = 136.92 := 
sorry

end shark_population_change_l639_639280


namespace sum_proper_divisors_432_l639_639691

/-- A proper divisor of a number n is a divisor that is not equal to n --/
def is_proper_divisor (n d : ℕ) : Prop :=
  d ∣ n ∧ d ≠ n

/-- We define the sum of proper divisors given the divisors --/
def sum_of_proper_divisors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ d, is_proper_divisor n d) (Finset.range (n + 1))), d

/-- Prime factorization of 432 is 2^4 * 3^3 --/
def prime_factors_432 : ℕ := 2 ^ 4 * 3 ^ 3

theorem sum_proper_divisors_432 : sum_of_proper_divisors 432 = 808 := sorry

end sum_proper_divisors_432_l639_639691


namespace percentage_both_correct_l639_639004

variable (A B : Type) 

noncomputable def percentage_of_test_takers_correct_first : ℝ := 0.85
noncomputable def percentage_of_test_takers_correct_second : ℝ := 0.70
noncomputable def percentage_of_test_takers_neither_correct : ℝ := 0.05

theorem percentage_both_correct :
  percentage_of_test_takers_correct_first + 
  percentage_of_test_takers_correct_second - 
  (1 - percentage_of_test_takers_neither_correct) = 0.60 := by
  sorry

end percentage_both_correct_l639_639004


namespace find_a_plus_b_eq_102_l639_639098

theorem find_a_plus_b_eq_102 :
  ∃ (a b : ℕ), (1600^(1 / 2) - 24 = (a^(1 / 2) - b)^2) ∧ (a + b = 102) :=
by {
  sorry
}

end find_a_plus_b_eq_102_l639_639098


namespace num_of_solutions_l639_639319

noncomputable def satisfies_equation (x y : ℤ) : Prop := 
  x^2 + 7 * x * y + 6 * y^2 = 15^50

theorem num_of_solutions : 
  ∃ n : ℕ, n = 4998 ∧ ∀ (x y : ℤ), satisfies_equation x y ↔ (x, y) ∈ (finset.univ.filter (λ p, satisfies_equation p.1 p.2)).val :=
sorry

end num_of_solutions_l639_639319


namespace Cristina_pace_problem_statement_l639_639985

variables (Nicky_pace : ℝ) (head_start_time : ℝ) (catch_up_time : ℝ)

def Nicky_initial_distance := Nicky_pace * head_start_time
def Nicky_additional_distance := Nicky_pace * catch_up_time
def Nicky_total_distance := Nicky_initial_distance + Nicky_additional_distance
def Cristina_catchup_distance := Nicky_total_distance - Nicky_initial_distance

theorem Cristina_pace :
  Cristina_catchup_distance / catch_up_time = Nicky_pace := by
  sorry

-- Given conditions
def given_conditions : Prop :=
  Nicky_pace = 3 ∧ head_start_time = 12 ∧ catch_up_time = 30

-- Theorem encapsulating the problem
theorem problem_statement (h : given_conditions) : Cristina_pace Nicky_pace head_start_time catch_up_time = 3 := by
  rcases h with ⟨h1, h2, h3⟩
  unfold Cristina_pace
  rw [h1, h2, h3]
  sorry

end Cristina_pace_problem_statement_l639_639985


namespace poly_difference_form_poly_s_square_l639_639499

noncomputable def poly_sum_coeffs_eq (p q : Polynomial ℝ) (s : ℝ) : Prop :=
  (Polynomial.eval 1 p) = s ∧ (Polynomial.eval 1 q) = s

def poly_equation (p q : Polynomial ℝ) : Prop :=
  p^3 - q^3 = Polynomial.eval 3 p - Polynomial.eval 3 q

theorem poly_difference_form
  (p q : Polynomial ℝ) (s : ℝ)
  (h_sum : poly_sum_coeffs_eq p q s)
  (h_eq : poly_equation p q) :
  ∃ a : ℕ, a ≥ 1 ∧ ∃ r : Polynomial ℝ, (p - q) = (Polynomial.X - 1)^a * r ∧ r.eval 1 ≠ 0 := sorry

theorem poly_s_square
  (p q : Polynomial ℝ) (s : ℝ)
  (h_sum : poly_sum_coeffs_eq p q s)
  (h_eq : poly_equation p q)
  (h_diff : ∃ a : ℕ, a ≥ 1 ∧ ∃ r : Polynomial ℝ, (p - q) = (Polynomial.X - 1)^a * r ∧ r.eval 1 ≠ 0) :
  s^2 = 3^(nat.pred (nat.succ (classical.some (classical.some h_diff)))) := sorry

end poly_difference_form_poly_s_square_l639_639499


namespace frank_original_money_l639_639338

theorem frank_original_money (X : ℝ) :
  (X - (1 / 5) * X - (1 / 4) * (X - (1 / 5) * X) = 360) → (X = 600) :=
by
  sorry

end frank_original_money_l639_639338


namespace find_k_l639_639232

noncomputable theory
open_locale classical

variables (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)

-- Condition 1: ABC is an acute isosceles triangle inscribed in a circle
-- Condition 2: Tangents from B and C meet at D
-- Condition 3: angle ABC = angle ACB = 3 * angle D
    
def tangents_intersect_at_D (ABC : triangle) (B C D : point ABC) : Prop :=
  tangent_to_circle_at B D ∧ tangent_to_circle_at C D ∧ B ≠ C

def isosceles_triangle_condition (angle : point ABC → point ABC → point ABC → real) (B C : point ABC) : Prop :=
  angle ABC B C = angle ABC C B

def angle_triple_D (angle : point ABC → point ABC → point ABC → real) (B C D : point ABC) : Prop :=
  ∃ k, angle ABC B C = 3 * angle ABC D C ∧ angle ABC A B = k * π

theorem find_k (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)
  (h1 : tangents_intersect_at_D ABC B C D)
  (h2 : isosceles_triangle_condition angle B C)
  (h3 : angle_triple_D angle B C D) :
  ∃ k : real, angle ABC A B = (5 / 11) * π ∧ k = 5 / 11 :=
by sorry

end find_k_l639_639232


namespace cars_travel_same_distance_l639_639637

-- Define all the variables and conditions
def TimeR : ℝ := sorry -- the time taken by car R
def TimeP : ℝ := TimeR - 2
def SpeedR : ℝ := 58.4428877022476
def SpeedP : ℝ := SpeedR + 10

-- state the distance travelled by both cars
def DistanceR : ℝ := SpeedR * TimeR
def DistanceP : ℝ := SpeedP * TimeP

-- Prove that both distances are the same and equal to 800
theorem cars_travel_same_distance : DistanceR = 800 := by
  sorry

end cars_travel_same_distance_l639_639637


namespace circle_moving_process_terminates_l639_639639

noncomputable def barycenter (points : List (ℝ × ℝ)) : ℝ × ℝ :=
  let sum_x := points.map Prod.fst |>.sum
  let sum_y := points.map Prod.snd |>.sum
  let n := points.length
  (sum_x / n, sum_y / n)

theorem circle_moving_process_terminates (n : ℕ) (points : Fin n → ℝ × ℝ)
    (circle : ℝ × ℝ × ℝ) (h_in_circle : ∃ p : Fin n, (p.2 - circle.1.2)^2 + (p.1 - circle.1.1)^2 < circle.2^2) : 
  ∃ k : ℕ, ∀ m ≥ k, step_moved_center m = step_moved_center k := 
sorry

end circle_moving_process_terminates_l639_639639


namespace reflection_twice_is_identity_l639_639953

theorem reflection_twice_is_identity :
  let a := ![4, -2]
  let S := reflection_matrix a
  S * S = (1 : Matrix (Fin 2) (Fin 2) ℝ) := 
by
  let a := ![4, -2]
  let S := reflection_matrix a
  sorry

end reflection_twice_is_identity_l639_639953


namespace curve_transformation_l639_639841

theorem curve_transformation (A : Matrix (Fin 2) (Fin 2) ℝ)
  (C1_eq : ∀ (x₀ y₀ : ℝ), (A.mul_vec ![x₀, y₀]) = ![x₀, sqrt 2 * y₀] → (x₀^2 / 4 + y₀^2 / 2 = 1)) :
  ∀ (x y : ℝ), ((x^2 / 4 + y^2 = 1) ↔ C1_eq (A.mul_vec ![x, y])) :=
by
  sorry

end curve_transformation_l639_639841


namespace cryptarithm_problem_l639_639903

theorem cryptarithm_problem (F E D : ℤ) (h1 : F - E = D - 1) (h2 : D + E + F = 16) (h3 : F - E = D) : 
    F - E = 5 :=
by sorry

end cryptarithm_problem_l639_639903


namespace meadow_income_is_960000_l639_639981

theorem meadow_income_is_960000 :
  let boxes := 30
  let packs_per_box := 40
  let diapers_per_pack := 160
  let price_per_diaper := 5
  (boxes * packs_per_box * diapers_per_pack * price_per_diaper) = 960000 := 
by
  sorry

end meadow_income_is_960000_l639_639981


namespace average_speed_bike_l639_639297

-- Defining the conditions
def swim_distance := 0.5
def swim_speed := 2
def run_distance := 5
def run_speed := 10
def kayak_distance := 1
def kayak_speed := 3
def bike_distance := 20
def total_time := 3

-- Calculated times based on conditions
def swim_time := swim_distance / swim_speed
def run_time := run_distance / run_speed
def kayak_time := kayak_distance / kayak_speed
def non_bike_time := swim_time + run_time + kayak_time
def bike_time := total_time - non_bike_time

-- Proving the average speed required for the bicycle ride
def bike_required_speed := bike_distance / bike_time

theorem average_speed_bike : bike_required_speed = (240 / 23) := by
  sorry

end average_speed_bike_l639_639297


namespace gcd_lcm_sum_l639_639058

-- Define the necessary components: \( A \) as the greatest common factor and \( B \) as the least common multiple of 16, 32, and 48
def A := Int.gcd (Int.gcd 16 32) 48
def B := Int.lcm (Int.lcm 16 32) 48

-- Statement that needs to be proved
theorem gcd_lcm_sum : A + B = 112 := by
  sorry

end gcd_lcm_sum_l639_639058


namespace binom_lt_pow2_binom_divisible_by_primes_pi_inequality1_pi_inequality2_pi_deduction_l639_639676

-- Part (a)
theorem binom_lt_pow2 (n : ℕ) : 
  nat.choose (2 * n) n < 2 ^ (2 * n) :=
sorry

theorem binom_divisible_by_primes (n p : ℕ) (hp : n < p ∧ p < 2 * n) : 
  p ∣ nat.choose (2 * n) n :=
sorry

-- Part (b)
def pi : ℕ → ℕ := sorry -- assuming pi is number of primes ≤ x

theorem pi_inequality1 (n : ℕ) (hn : n > 2) :
  pi (2 * n) < pi n + 2 * n / Math.log2 n :=
sorry

theorem pi_inequality2 (n : ℕ) (hn : n > 1) :
  pi (2^n) < 2^(n+1) / (n * Math.log2 (n - 1)) :=
sorry

theorem pi_deduction (x : ℕ) (hx : x ≥ 8) :
  pi x < 4 * x / Math.log2 x * Math.log2 (Math.log2 x) :=
sorry

end binom_lt_pow2_binom_divisible_by_primes_pi_inequality1_pi_inequality2_pi_deduction_l639_639676


namespace count_multiples_of_72_l639_639875

noncomputable def lcm (m n : ℕ) : ℕ := Nat.lcm m n

theorem count_multiples_of_72 (count : ℕ) :
  let lcm_12_18_24 := lcm 12 (lcm 18 24)
  500 ≤ 2500 →
  ∀ (x : ℕ), 
    (500 ≤ x ∧ x ≤ 2500) → 
    (lcm_12_18_24 ∣ x) ↔ 
    count = 28 :=
by
  let lcm := lcm 12 (lcm 18 24)
  have lcm_eq_72 : lcm = 72 := by sorry
  sorry

end count_multiples_of_72_l639_639875


namespace lights_at_top_of_tower_l639_639029

theorem lights_at_top_of_tower :
  ∃ a₁ : ℕ, (a₁ * ∑ i in finset.range 7, 2^i) = 381 ∧ 
  a₁ = 3 :=
by
  sorry

end lights_at_top_of_tower_l639_639029


namespace min_digits_to_decimal_l639_639287

theorem min_digits_to_decimal (n d : ℕ) (h : n = 987654321) (h2 : d = 2^30 * 5^6) : 
  minimum_digits_to_decimal n d = 30 :=
sorry

end min_digits_to_decimal_l639_639287


namespace count_7_digit_palindromes_l639_639871

-- Define the problem conditions
def digits : Multiset ℕ := {1, 1, 1, 4, 4, 6, 6}

-- Define a 7-digit palindrome condition
def is_palindrome (n : List ℕ) : Prop := n = n.reverse

-- Define the main problem statement
theorem count_7_digit_palindromes : 
  countp (λ n, is_palindrome n ∧ n.length = 7) (perm (digits.to_list)) = 6 :=
sorry

end count_7_digit_palindromes_l639_639871


namespace frank_money_l639_639336

theorem frank_money (X : ℝ) (h1 : (3/4) * (4/5) * X = 360) : X = 600 :=
sorry

end frank_money_l639_639336


namespace numerator_divisible_by_p_l639_639993

open Nat

theorem numerator_divisible_by_p (p : ℕ) (hp : prime p) (hp2 : p > 2) :
  (∃ k : ℤ, (1 + ∑ i in finset.range (p - 1), (1 : ℚ) / (i + 1)).num = p * k) :=
sorry

end numerator_divisible_by_p_l639_639993


namespace polynomial_roots_r_l639_639585

theorem polynomial_roots_r :
  ∀ (α β γ : ℝ),
    (polynomial.root_set (polynomial.C (-14) + polynomial.C 5 * polynomial.X 
                          + polynomial.C 4 * polynomial.X^2 + polynomial.X^3) ℝ = 
     {α, β, γ}) →
    ∃ p q r : ℝ, 
      (polynomial.root_set (polynomial.C r + polynomial.C q * polynomial.X 
                            + polynomial.C p * polynomial.X^2 + polynomial.X^3) ℝ = 
       {α + β, β + γ, γ + α}) ∧ 
      r = 34 :=
begin
  sorry
end

end polynomial_roots_r_l639_639585


namespace find_a_l639_639745

-- Define the function and the conditions
def y (a b x : ℝ) : ℝ := a * Real.cos (b * x)

theorem find_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (hmax : ∀ x : ℝ, y a b x ≤ 3) :
  a = 3 :=
sorry

end find_a_l639_639745


namespace cucumbers_for_24_apples_l639_639896

theorem cucumbers_for_24_apples :
  (∀ (a b c d apples bananas cucumbers : ℕ), (12 * a = 6 * b) → (3 * b = 4 * d) → (24 * a = n * cucumbers)) :=
by
  intros a b c d apples bananas cucumbers h1 h2
  have h3 : 2 * (12 * a) = 2 * (6 * b) := sorry
  have h4 : 24 * a = 12 * b := by
    rw [←h3]
    exact sorry
  have h5 : 4 * (3 * b) = 4 * (4 * d) := sorry
  have h6 : 12 * b = 16 * d := by
    rw [←h5]
    exact sorry
  have h7 : 24 * a = 16 * d := by
    rw [h4, h6]
    exact sorry
  exact eq.symm h7

end cucumbers_for_24_apples_l639_639896


namespace max_profit_l639_639635

theorem max_profit (m : ℝ) :
  (m - 8) * (900 - 15 * m) = -15 * (m - 34) ^ 2 + 10140 :=
by
  sorry

end max_profit_l639_639635


namespace product_of_roots_eq_one_l639_639806

noncomputable def product_of_real_roots (a : ℝ) (h : a > 1) : ℝ :=
  let x := λ y : ℝ, 10 ^ y in
  let lg_a := Real.log10 a in
  let sqrt_lg_a := Real.sqrt lg_a in
  x sqrt_lg_a * x (-sqrt_lg_a)

theorem product_of_roots_eq_one (a : ℝ) (h : a > 1) : product_of_real_roots a h = 1 :=
  sorry

end product_of_roots_eq_one_l639_639806


namespace problem_1_problem_2_l639_639848

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x - 3|

theorem problem_1 (a x : ℝ) (h1 : a < 3) (h2 : (∀ x, f x a >= 4 ↔ x ≤ 1/2 ∨ x ≥ 9/2)) : 
  a = 2 :=
sorry

theorem problem_2 (a : ℝ) (h1 : ∀ x : ℝ, f x a + |x - 3| ≥ 1) : 
  a ≤ 2 :=
sorry

end problem_1_problem_2_l639_639848


namespace triangles_have_same_ratio_of_area_and_circumradius_l639_639544

-- Conditions
variables {A B C A1 B1 C1 A2 B2 C2 : Type}
variables (a1 b1 c1 a2 b2 c2 : A → B → C → Prop) -- lines passing through vertices A, B, C

-- Definitions of reflections and intersections
def reflects_on_bisector (l1 l2 : A → B → C → Prop) : Prop := 
  -- Assuming a definition for line reflection
  sorry

def intersection (l1 l2 : A → B → C → Prop) : Type := 
  -- Assuming a definition for the intersection of two lines
  sorry

def circumradius (a b c : Type) : ℝ := 
  -- Assuming a definition for the circumradius of triangle
  sorry

def area (a b c : Type) : ℝ := 
  -- Assuming a definition for the area of triangle
  sorry

-- Theorem statement
theorem triangles_have_same_ratio_of_area_and_circumradius :
  (reflects_on_bisector a1 a2) →
  (reflects_on_bisector b1 b2) →
  (reflects_on_bisector c1 c2) →
  (intersection b1 c1 = A1) →
  (intersection a1 c1 = B1) →
  (intersection a1 b1 = C1) →
  (intersection b2 c2 = A2) →
  (intersection a2 c2 = B2) →
  (intersection a2 b2 = C2) →
  (area A1 B1 C1 / circumradius A1 B1 C1) = (area A2 B2 C2 / circumradius A2 B2 C2) :=
begin
  sorry
end

end triangles_have_same_ratio_of_area_and_circumradius_l639_639544


namespace bucket_capacity_l639_639685

-- Given Conditions
variable (C : ℝ)
variable (h : (2 / 3) * C = 9)

-- Goal
theorem bucket_capacity : C = 13.5 := by
  sorry

end bucket_capacity_l639_639685


namespace quadratic_polynomial_solution_l639_639321

def quadratic_polynomial (x : ℝ) : ℝ :=
  (10 / 3) * x^2 + (50 / 3) * x - 80

theorem quadratic_polynomial_solution : 
  (∀ x, quadratic_polynomial x = (10 / 3) * x^2 + (50 / 3) * x - 80) ∧ 
  quadratic_polynomial (-8) = 0 ∧ 
  quadratic_polynomial (3) = 0 ∧ 
  quadratic_polynomial (4) = 40 :=
by 
  sorry

end quadratic_polynomial_solution_l639_639321


namespace domain_of_f_l639_639153

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x - 5) + real.sqrt (6 - x) + real.cbrt (x + 4)

theorem domain_of_f :
  { x : ℝ | 5 ≤ x ∧ x ≤ 6 } = set_of (λ x, x ∈ dom f) :=
by
  sorry

end domain_of_f_l639_639153


namespace lamps_remain_off_after_operations_l639_639640

def is_multiple_of (n k : ℕ) : Prop :=
  n % k = 0

def ends_in_5 (n : ℕ) : Prop :=
  n % 10 = 5

def count_multiples_up_to (k m : ℕ) : ℕ :=
  m / k

def count_numbers_ending_in_5_up_to (m : ℕ) : ℕ :=
  (if m < 5 then 0 else (m - 5) / 10 + 1)

theorem lamps_remain_off_after_operations (m : ℕ) (h : m = 200) : 
    ∃ (off_lamps : ℕ), off_lamps = 73 :=
by
  have multiples_of_3 := count_multiples_up_to 3 200
  have numbers_ending_in_5 := count_numbers_ending_in_5_up_to 200
  have multiples_of_15 := count_multiples_up_to 15 200
  have toggled_once := multiples_of_3 + numbers_ending_in_5 - multiples_of_15
  existsi 73
  have h1 : multiples_of_3 = 66 := sorry
  have h2 : numbers_ending_in_5 = 20 := sorry
  have h3 : multiples_of_15 = 13 := sorry
  calc
    toggled_once = 66 + 20 - 13 : by rw [h1, h2, h3]
              ... = 73 : by norm_num

end lamps_remain_off_after_operations_l639_639640


namespace raise_percentage_is_correct_l639_639715

-- Define the initial wage and conditions.
def initial_wage (W : ℝ) := W
def wage_after_cut (W : ℝ) := 0.7 * W
def wage_after_deduction (W : ℝ) := 0.7 * W - 100
def required_raise_percentage (W : ℝ) (new_wage : ℝ) := (W / new_wage - 1) * 100

-- Assert that the required raise percentage is 66.67%.
theorem raise_percentage_is_correct (W : ℝ) (hW : W > 0) : 
  required_raise_percentage W (wage_after_deduction W) = 66.67 :=
by
  -- The proof will be done in another step.
  sorry

end raise_percentage_is_correct_l639_639715


namespace insurance_coverage_is_80_percent_l639_639644

-- Definitions and conditions
def MRI_cost : ℕ := 1200
def doctor_hourly_fee : ℕ := 300
def doctor_examination_time : ℕ := 30  -- in minutes
def seen_fee : ℕ := 150
def amount_paid_by_tim : ℕ := 300

-- The total cost calculation
def total_cost : ℕ := MRI_cost + (doctor_hourly_fee * doctor_examination_time / 60) + seen_fee

-- The amount covered by insurance
def amount_covered_by_insurance : ℕ := total_cost - amount_paid_by_tim

-- The percentage of coverage by insurance
def insurance_coverage_percentage : ℕ := (amount_covered_by_insurance * 100) / total_cost

theorem insurance_coverage_is_80_percent : insurance_coverage_percentage = 80 := by
  sorry

end insurance_coverage_is_80_percent_l639_639644


namespace sum_value_l639_639135

variable (T R S PV : ℝ)
variable (TD SI : ℝ) (h_td : TD = 80) (h_si : SI = 88)
variable (h1 : SI = TD + (TD * R * T) / 100)
variable (h2 : (PV * R * T) / 100 = TD)
variable (h3 : PV = S - TD)
variable (h4 : R * T = 10)

theorem sum_value : S = 880 := by
  sorry

end sum_value_l639_639135


namespace probability_B_second_shot_probability_A_ith_shot_expected_A_shots_first_n_l639_639990

def player_A_accuracy : ℚ := 3/5
def player_B_accuracy : ℚ := 4/5
def initial_probability : ℚ := 1/2

theorem probability_B_second_shot :
  initial_probability * (1 - player_A_accuracy) + initial_probability * player_B_accuracy = 3/5 :=
by sorry

theorem probability_A_ith_shot (i : ℕ) : 
  (1/3 : ℚ) + (1/6) * (bit0 (1/5) : ℚ)^(i-1) = 
    if i = 0 then 1/2 else
      let p := player_A_accuracy * (1/2 : ℚ) 
              + (1/5 : ℚ) * (initial_probability) 
      in p :=
by sorry

theorem expected_A_shots_first_n (n : ℕ) : 
  (5/18 : ℚ) * (1 - ((bit0 (1/5 : ℚ))^n)) + (n / 3 : ℚ) = 
    if n = 0 then 0 else
      (5/18 * (1 - (2/5)^n)) + (n/3) :=
by sorry

end probability_B_second_shot_probability_A_ith_shot_expected_A_shots_first_n_l639_639990


namespace find_n_for_even_function_l639_639843

noncomputable def determinant (a1 a2 a3 a4 : ℝ) : ℝ :=
  a1 * a4 - a2 * a3

noncomputable def f (x : ℝ) : ℝ :=
  determinant (sqrt 3) 1 (sin x) (cos x)

theorem find_n_for_even_function (n : ℝ) (h1 : ∀ (a1 a2 a3 a4 : ℝ), determinant a1 a2 a3 a4 = a1 * a4 - a2 * a3)
  (h2 : ∀ (x : ℝ), f x = sqrt 3 * cos x - sin x)
  (h3 : ∀ (x : ℝ), (λ y, f (x + n)) y = 2 * cos (x + n + π / 6))
  (h4 : 0 < n)
  : n = 5 * π / 6 := sorry

end find_n_for_even_function_l639_639843


namespace rotation_center_circumcenter_l639_639967

-- Definitions of required points and triangles
variables {A B C A' B' C' : Type*}

-- Conditions: A'B'C' is similar to triangle ABC and points A', B', C' lie on sides BC, AC, AB respectively.
def similar_triangles (A B C A' B' C' : Type*) : Prop :=
  sorry -- Assume it defines similarity between triangles

def on_side (P1 P2 P : Type*) : Prop :=
  sorry -- Assume it asserts point P is on the line segment joining P1 and P2

-- Stating the problem
theorem rotation_center_circumcenter (A B C A' B' C' : Type*)
  (similar : similar_triangles A B C A' B' C')
  (onA'BC : on_side B C A')
  (onB'AC : on_side A C B')
  (onC'AB : on_side A B C') :
  ∃ O : Type*, (is_rotation_center O A B C A' B' C') ∧ (is_circumcenter O A B C) := 
sorry -- The proof will be filled in.

end rotation_center_circumcenter_l639_639967


namespace polar_line_circle_intersection_count_l639_639482

noncomputable def polar_to_cartesian := sorry

theorem polar_line_circle_intersection_count :
  let line_cartesian := λ (x y : ℝ), (√2 / 2) * x + (√2 / 2) * y = √2,
      circle_cartesian := λ (x y : ℝ), x^2 + y^2 = 2 in
  ∃! (p : ℝ × ℝ), line_cartesian p.1 p.2 ∧ circle_cartesian p.1 p.2 :=
begin
  have h1 : ∀ (x y : ℝ), line_cartesian x y ↔ x + y - 2 = 0, sorry,
  have h2 : ∀ (x y : ℝ), circle_cartesian x y ↔ x^2 + y^2 = 2, sorry,
  have h_dist : ∀ (x y : ℝ), (x = 0 ∧ y = 0) → 
    (λ (x y : ℝ), (abs (0+0-2) / sqrt (2))) = sqrt (2), sorry,
  have h_tangent : sqrt (2) = 2 → ∃! (p : ℝ × ℝ), x + y - 2 = 0 ∧ x^2 + y^2 = 2, sorry,
  exact h_tangent,
end

end polar_line_circle_intersection_count_l639_639482


namespace integer_coeff_squares_sum_eq_2180_l639_639879

theorem integer_coeff_squares_sum_eq_2180
  (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 216 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 2180 := 
begin
  sorry
end

end integer_coeff_squares_sum_eq_2180_l639_639879


namespace max_points_on_circle_l639_639949

noncomputable def I : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ irrational x ∧ irrational y}
def R : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ ∃ qx qy : ℚ, x = qx ∧ y = qy}
def circle (C : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ (x - C.1)^2 + (y - C.2)^2 = r^2}

theorem max_points_on_circle (C : ℝ × ℝ) (r : ℝ) (hC : C ∈ I) (hr : irrational r) : 
  ∃ S ⊆ R, S.finite ∧ S.card = 2 ∧ ∀ P ∈ S, P ∈ circle C r :=
sorry

end max_points_on_circle_l639_639949


namespace ratio_SX_SP_is_one_l639_639473

-- Definitions based on the problem conditions
def is_square (PQRS : set (ℝ × ℝ)) (PQ_side_length : ℝ) : Prop :=
∃ (P Q R S : ℝ × ℝ),
  PQRS = {P, Q, R, S} ∧
  PQ = PQ_side_length ∧
  -- Other conditions defining a square are omitted for brevity
  sorry

-- Problem specifics defining the square with side length 6 cm
def square_PQRS : set (ℝ × ℝ) := sorry
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry
def R : ℝ × ℝ := sorry
def S : ℝ × ℝ := sorry
def N : ℝ × ℝ := midpoint R S
def X : ℝ × ℝ := intersection PQ_line SN_line

-- Prove that the ratio of SX to SP is 1
theorem ratio_SX_SP_is_one : 
  is_square square_PQRS 6 → 
  N = midpoint R S →
  X = intersection PQ_line SN_line →
  ratio (dist S X) (dist S P) = 1 := 
begin
  intros,
  sorry 
end

end ratio_SX_SP_is_one_l639_639473


namespace root_sum_value_l639_639042

noncomputable def polynomial_roots := {a b c d : ℂ // (λ x : ℂ, x^4 + 2*x + 4) x = 0}

theorem root_sum_value (r : polynomial_roots) :
  let ⟨a, b, c, d, _⟩ := r in
  (a^2 / (a^3 + 2) + b^2 / (b^3 + 2) + c^2 / (c^3 + 2) + d^2 / (d^3 + 2)) = -4 := by
  sorry

end root_sum_value_l639_639042


namespace acute_isosceles_triangle_inscribed_circle_l639_639240

variables {A B C D : Type} [EuclideanGeometry A B C D]

theorem acute_isosceles_triangle_inscribed_circle (h1 : is_acute A B C)
    (h2 : is_isosceles A B C) (h3 : inscribed_in_circle A B C)
    (h4 : are_tangents B C D) (h5 : ∠ ABC = ∠ ACB = 3 * ∠ D)
    (h6 : tan_from_circle_point B C D) :
    (∠ BAC = 5 * π / 11) :=
sorry

end acute_isosceles_triangle_inscribed_circle_l639_639240


namespace area_of_diamond_l639_639009

theorem area_of_diamond (x y : ℝ) : (|x / 2| + |y / 2| = 1) → 
∃ (area : ℝ), area = 8 :=
by sorry

end area_of_diamond_l639_639009


namespace series_diverges_l639_639773

def series (n : ℕ) : ℝ := (nat.factorial n) / (2 : ℝ)^n

theorem series_diverges : ¬ (summable series) := 
sorry

end series_diverges_l639_639773


namespace meadow_total_money_l639_639982

/-
  Meadow orders 30 boxes of diapers weekly.
  Each box contains 40 packs.
  Each pack contains 160 diapers.
  Meadow sells each diaper for $5.
  Prove that the total money Meadow makes from selling all her diapers is $960000.
-/

theorem meadow_total_money :
  let number_of_boxes := 30 in
  let packs_per_box := 40 in
  let diapers_per_pack := 160 in
  let price_per_diaper := 5 in
  let total_packs := number_of_boxes * packs_per_box in
  let total_diapers := total_packs * diapers_per_pack in
  let total_money := total_diapers * price_per_diaper in
  total_money = 960000 :=
by
  sorry

end meadow_total_money_l639_639982


namespace b_2015_value_l639_639803

-- Conditions
def f1 (x : ℝ) := (x^2 + 2*x + 1) * Real.exp x
def f_n (n : ℕ) (x : ℝ) : ℝ := 
  let f_aux : ℕ → (ℝ → ℝ) := 
    λ k, if k = 0 then f1 else f_aux (k - 1)^(1)
  f_aux n (x)

-- Define b_n
def b_n (n : ℕ) : ℕ := 2 * n

-- Theorem to prove b_{2015} = 4030
theorem b_2015_value : b_n 2015 = 4030 := by {
  -- This is where the proof would go
  sorry
}

end b_2015_value_l639_639803


namespace tan_J_right_triangle_l639_639464

theorem tan_J_right_triangle (IK IJ : ℝ) (K : ℕ) (h_right : angle I J K = 90) 
  (h_IK : IK = 20) (h_IJ : IJ = 29) : 
  tan (angle I J K) = 21 / 20 := 
sorry

end tan_J_right_triangle_l639_639464


namespace loss_percentage_is_15_l639_639204

-- A noncomputable definition to represent the cost price (CP)
noncomputable def CP : ℝ := 120.0

-- Definitions of selling prices at given conditions
def SP_gain_20 : ℝ := 144.0
def SP_loss : ℝ := 102.0

-- Condition for percentage gain
def condition_gain_20 (CP SP_gain_20 : ℝ) : Prop :=
  SP_gain_20 = CP * 1.20

-- Condition for the percentage loss to be proven
def percentage_loss (SP_loss CP : ℝ) : ℝ :=
  ((CP - SP_loss) / CP) * 100

-- The theorem statement to prove
theorem loss_percentage_is_15 (h : condition_gain_20 CP SP_gain_20) :
  percentage_loss SP_loss CP = 15 :=
sorry

end loss_percentage_is_15_l639_639204


namespace x_varies_as_sin_squared_l639_639056

variable {k j z : ℝ}
variable (x y : ℝ)

-- condition: x is proportional to y^2
def proportional_xy_square (x y : ℝ) (k : ℝ) : Prop :=
  x = k * y ^ 2

-- condition: y is proportional to sin(z)
def proportional_y_sin (y : ℝ) (j z : ℝ) : Prop :=
  y = j * Real.sin z

-- statement to prove: x is proportional to (sin(z))^2
theorem x_varies_as_sin_squared (k j z : ℝ) (x y : ℝ)
  (h1 : proportional_xy_square x y k)
  (h2 : proportional_y_sin y j z) :
  ∃ m, x = m * (Real.sin z) ^ 2 :=
by
  sorry

end x_varies_as_sin_squared_l639_639056


namespace kernels_needed_for_movie_night_l639_639492

structure PopcornPreferences where
  caramel_popcorn: ℝ
  butter_popcorn: ℝ
  cheese_popcorn: ℝ
  kettle_corn_popcorn: ℝ

noncomputable def total_kernels_needed (preferences: PopcornPreferences) : ℝ :=
  (preferences.caramel_popcorn / 6) * 3 +
  (preferences.butter_popcorn / 4) * 2 +
  (preferences.cheese_popcorn / 8) * 4 +
  (preferences.kettle_corn_popcorn / 3) * 1

theorem kernels_needed_for_movie_night :
  let preferences := PopcornPreferences.mk 3 4 6 3
  total_kernels_needed preferences = 7.5 :=
sorry

end kernels_needed_for_movie_night_l639_639492


namespace collinear_R_S_T_l639_639538

open Real
open Geometry

-- Define the cyclic quadrilateral, diagonals and their intersection
variables {A B C D E T R S : Point}
variables (circumcircle : Circle)
variables (Γ : Circle)

-- Assume the conditions specified
axiom cyclic_quad (h : CyclicQuadrilateral A B C D) : True
axiom diagonal_intersection (h : LineIntersection (Line A C) (Line B D) E) : True
axiom circle_tangent_arc (Γ.internalTangent (circumcircle.arc B C) (T) (not_contains (D))) : True
axiom circle_tangent_BE_CE (Γ.tangent_to_B_and_C_lines (Line B E) (Line C E)) : True
axiom R_bisectors_intersection (R_is_intersection_of_angle_bisectors ∠A B C ∠B C D) : True
axiom S_incenter (S_is_incenter_of_triangle B C E) : True

-- Formalize the main theorem statement
theorem collinear_R_S_T (h : CyclicQuadrilateral A B C D) (h : LineIntersection (Line A C) (Line B D) E )
  (h : Γ.internalTangent (circumcircle.arc B C) (T) (not_contains (D)))
  (h : Γ.tangent_to_B_and_C_lines (Line B E) (Line C E))
  (h : R_is_intersection_of_angle_bisectors ∠A B C ∠B C D)
  (h : S_is_incenter_of_triangle B C E) :
  Collinear R S T := 
sorry

end collinear_R_S_T_l639_639538


namespace acute_isosceles_triangle_k_l639_639252

theorem acute_isosceles_triangle_k (ABC : Triangle) (circ : Circle)
  (D : Point)
  (h1 : ABC.angles.B = ABC.angles.C) -- Isosceles property
  (h2 : ∀ P ∈ circ, is_tangent B P circ) -- Tangent property through B
  (h3 : ∀ Q ∈ circ, is_tangent C Q circ) -- Tangent property through C
  (h4 : angle ABC.angles.B = 3 * angle D )
  (h5 : ∃ k, angle ABC.angles.A = k * π ) :
  ∃ k, k = 5 / 11 :=
by
  sorry

end acute_isosceles_triangle_k_l639_639252


namespace calories_per_orange_is_correct_l639_639489

noncomputable def calories_per_orange
  (oranges pieces_per_orange num_people calories_per_person : ℕ)
  (h_oranges : oranges = 5)
  (h_pieces_per_orange : pieces_per_orange = 8)
  (h_num_people : num_people = 4)
  (h_calories_per_person : calories_per_person = 100) : ℕ :=
by
  -- Definitions derived from conditions
  let total_pieces := oranges * pieces_per_orange
  let pieces_per_person := total_pieces / num_people
  let total_calories := calories_per_person
  have calories_per_piece := total_calories / pieces_per_person

  -- Conclusion
  have calories_per_orange := pieces_per_orange * calories_per_piece
  exact calories_per_orange

theorem calories_per_orange_is_correct
  (oranges pieces_per_orange num_people calories_per_person : ℕ)
  (h_oranges : oranges = 5)
  (h_pieces_per_orange : pieces_per_orange = 8)
  (h_num_people : num_people = 4)
  (h_calories_per_person : calories_per_person = 100) :
  calories_per_orange oranges pieces_per_orange num_people calories_per_person
    h_oranges h_pieces_per_orange h_num_people h_calories_per_person = 100 :=
by
  simp [calories_per_orange]
  sorry  -- Proof omitted

end calories_per_orange_is_correct_l639_639489


namespace rectangle_area_l639_639616

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l639_639616


namespace johns_profit_l639_639934

noncomputable def selling_price : ℝ := 2
noncomputable def num_newspapers : ℕ := 500
noncomputable def sell_fraction : ℝ := 0.80
noncomputable def buy_discount : ℝ := 0.75

def buying_price_per_newspaper : ℝ := selling_price * (1 - buy_discount)
def total_cost : ℝ := num_newspapers * buying_price_per_newspaper
def num_sold_newspapers : ℕ := (sell_fraction * num_newspapers.to_real).to_nat
def revenue : ℝ := num_sold_newspapers * selling_price
def profit : ℝ := revenue - total_cost

theorem johns_profit :
  profit = 550 := by
  sorry

end johns_profit_l639_639934


namespace concurrency_equivalence_l639_639526

-- Definitions of the conditions given
variables (A B C : Type) [triangle A B C]

-- Conditions: concurrency of angle bisector of A, perpendicular bisector of AB, and altitude from B
variables (AM : line) (perp_bisector_AB : line) (altitude_B : line)
variables (concur1 : concurrent AM perp_bisector_AB altitude_B)

-- Desired condition: concurrency of angle bisector of A, perpendicular bisector of AC, and altitude from C
variables (perp_bisector_AC : line) (altitude_C : line)
variables (concur2 : concurrent AM perp_bisector_AC altitude_C)

-- The angle condition that needs to be proven
theorem concurrency_equivalence : ∠BAC = 60 → concur2 :=
sorry

end concurrency_equivalence_l639_639526


namespace reading_rate_l639_639131

-- Definitions based on conditions
def one_way_trip_time : ℕ := 4
def round_trip_time : ℕ := 2 * one_way_trip_time
def read_book_time : ℕ := 2 * round_trip_time
def book_pages : ℕ := 4000

-- The theorem to prove Juan's reading rate is 250 pages per hour.
theorem reading_rate : book_pages / read_book_time = 250 := by
  sorry

end reading_rate_l639_639131


namespace length_vector_P_l639_639820

-- Define the points P and Q in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the projection function to the xy-plane
def projection_xy_plane (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := 0 }

-- Define the distance function between two points in 3D space
def distance (p q : Point3D) : ℝ :=
  real.sqrt ((q.x - p.x) ^ 2 + (q.y - p.y) ^ 2 + (q.z - p.z) ^ 2)

-- Define the points P and Q
def P : Point3D := { x := 1, y := 2, z := 3 }
def Q : Point3D := { x := -3, y := 5, z := real.sqrt 2 }

-- Define the projections P' and Q' onto the xy-plane
def P' : Point3D := projection_xy_plane P
def Q' : Point3D := projection_xy_plane Q

-- Prove the length of the vector from P' to Q' is 5
theorem length_vector_P'Q' : distance P' Q' = 5 :=
  by
    sorry

end length_vector_P_l639_639820


namespace limit_at_1_l639_639273

noncomputable def limit_expression (x : ℝ) : ℝ := (49 / (x^2 - x + 0.07)) ^ 3

theorem limit_at_1 : filter.tendsto limit_expression (𝓝 1) (𝓝 343000000) :=
sorry

end limit_at_1_l639_639273


namespace maxOddFactor_sum_l639_639159

def maxOddFactor (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  let rec go (k : ℕ) (d : ℕ) :=
    if k = 1 then d
    else if k % 2 = 0 then go (k / 2) d
    else go (k - 2) k
  go n n

def sumOddFactors (n : ℕ) : ℕ :=
  ∑ k in finset.range (2^n + 1), maxOddFactor k

theorem maxOddFactor_sum (n : ℕ) (hn : 0 < n) : sumOddFactors n = (4^n + 2) / 3 := by
  sorry

end maxOddFactor_sum_l639_639159


namespace max_m_value_l639_639036

def b_series (n : ℕ) : ℝ := 2 ^ (n - 1)

def a_series (n : ℕ) : ℝ := real.log (b_series n) / real.log 2 + n + 2

theorem max_m_value (m : ℕ) (hm : ∑ i in finset.range m, a_series (i + 1) ≤ 63) : m ≤ 7 :=
begin
  sorry
end

end max_m_value_l639_639036


namespace arithmetic_sequence_first_term_and_difference_l639_639115

theorem arithmetic_sequence_first_term_and_difference
  (a1 d : ℤ)
  (h1 : (a1 + 2 * d) * (a1 + 5 * d) = 406)
  (h2 : a1 + 8 * d = 2 * (a1 + 3 * d) + 6) : 
  a1 = 4 ∧ d = 5 :=
by 
  sorry

end arithmetic_sequence_first_term_and_difference_l639_639115


namespace triangle_sides_length_l639_639560

theorem triangle_sides_length (a : ℝ) :
  let x := a; y := - (1 / 3) * a^2 in
  let OA := real.sqrt(x^2 + y^2);
  let OB := real.sqrt(x^2 + y^2);
  let AB := real.sqrt((2*x)^2 + (2*y)^2) in
  OA = a * real.sqrt(1 + (1 / 9) * a^2) ∧
  OB = a * real.sqrt(1 + (1 / 9) * a^2) ∧
  AB = 2 * a * real.sqrt(1 + (1 / 9) * a^2) := by
  sorry

end triangle_sides_length_l639_639560


namespace num_ways_to_choose_officers_same_gender_l639_639558

-- Definitions based on conditions
def num_members : Nat := 24
def num_boys : Nat := 12
def num_girls : Nat := 12
def num_officers : Nat := 3

-- Theorem statement using these definitions
theorem num_ways_to_choose_officers_same_gender :
  (num_boys * (num_boys-1) * (num_boys-2) * 2) = 2640 :=
by
  sorry

end num_ways_to_choose_officers_same_gender_l639_639558


namespace ln_power_eq_r_ln_l639_639079

theorem ln_power_eq_r_ln (f : ℝ) (r : ℚ) (h_pos : 0 < f) :
  (Real.ln (f^(r:ℝ)) = (r : ℝ) * Real.ln f) :=
sorry

end ln_power_eq_r_ln_l639_639079


namespace distance_squared_willy_sammy_l639_639028

theorem distance_squared_willy_sammy :
  ∀ (r_inner r_outer : ℝ),
  r_inner = 11 ∧ r_outer = 12 →
  let distance_squared := r_inner^2 + r_outer^2 - 2 * r_inner * r_outer * real.cos (real.pi / 6)
  in distance_squared = 265 - 132 * real.sqrt 3 :=
by
  intros r_inner r_outer h
  cases h with h_inner h_outer
  dsimp [distance_squared]
  rw [←h_inner, ←h_outer]
  norm_num
  sorry

end distance_squared_willy_sammy_l639_639028


namespace ones_digit_of_7_pow_53_l639_639659

theorem ones_digit_of_7_pow_53 : (7^53 % 10) = 7 := by
  sorry

end ones_digit_of_7_pow_53_l639_639659


namespace even_number_selection_l639_639568

-- Define the set and the condition of selecting 2 elements
def my_set : Finset ℕ := {1, 2, 3, 4, 5}

def even_numbers : Finset ℕ := {x ∈ my_set | x % 2 = 0}

-- Define the random variable that counts the number of even numbers in a subset
def count_even_numbers (s : Finset ℕ) : ℕ :=
  s.filter (λ x, x % 2 = 0).card

-- The possible values of this random variable when selecting 2 elements
def possible_values : Finset ℕ := {0, 1, 2}

-- Our statement to prove
theorem even_number_selection :
  ∀ (s : Finset ℕ), s ⊆ my_set → s.card = 2 →
  count_even_numbers s ∈ possible_values :=
by
  intro s h_subset h_card
  sorry

end even_number_selection_l639_639568


namespace count_integers_with_increasing_digits_l639_639420

theorem count_integers_with_increasing_digits :
  let count_integers := 
    ∑ second_digit in ({3, 4, 5} : Finset ℕ), 
      ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ second_digit}, 
        1 in

  count_integers = 15 :=
by
  have step1 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 3}, 1 = 6,
  { -- Explanation: If second digit is 3, third can be 4, 5, 6, 7, 8, 9 -> 6 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step2 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 4}, 1 = 5,
  { -- Explanation: If second digit is 4, third can be 5, 6, 7, 8, 9 -> 5 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step3 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 5}, 1 = 4,
  { -- Explanation: If second digit is 5, third can be 6, 7, 8, 9 -> 4 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have count_integers := step1 + step2 + step3,
  simp only [count_integers, add_comm],
  exact eq.refl 15

end count_integers_with_increasing_digits_l639_639420


namespace polar_circle_eqn_l639_639481

theorem polar_circle_eqn 
  (OM_diam : ∀ θ : ℝ, ∃ p : ℝ, p * cos θ = 2) :
    ∃ (ρ : ℝ), ρ = 2 * cos θ := sorry

end polar_circle_eqn_l639_639481


namespace set_intersection_complement_l639_639975

def setA : Set ℝ := {-2, -1, 0, 1, 2}
def setB : Set ℝ := { x : ℝ | x^2 + 2*x < 0 }
def complementB : Set ℝ := { x : ℝ | x ≥ 0 ∨ x ≤ -2 }

theorem set_intersection_complement :
  setA ∩ complementB = {-2, 0, 1, 2} :=
by
  sorry

end set_intersection_complement_l639_639975


namespace sqrt_inequality_l639_639181

theorem sqrt_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b + c = 1) :
  sqrt (3 * a + 1) + sqrt (3 * b + 1) + sqrt (3 * c + 1) ≤ 3 * sqrt 2 := 
sorry

end sqrt_inequality_l639_639181


namespace pages_read_per_hour_l639_639129

theorem pages_read_per_hour (lunch_time : ℕ) (book_pages : ℕ) (round_trip_time : ℕ)
  (h1 : lunch_time = round_trip_time)
  (h2 : book_pages = 4000)
  (h3 : round_trip_time = 2 * 4) :
  book_pages / (2 * lunch_time) = 250 :=
by
  sorry

end pages_read_per_hour_l639_639129


namespace area_proof_l639_639611

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l639_639611


namespace sequence_nth_term_l639_639844

theorem sequence_nth_term (a : ℕ → ℚ) (h : a 1 = 3 / 2 ∧ a 2 = 1 ∧ a 3 = 5 / 8 ∧ a 4 = 3 / 8) :
  ∀ n : ℕ, a n = (n^2 - 11*n + 34) / 16 := by
  sorry

end sequence_nth_term_l639_639844


namespace students_in_class_l639_639551

theorem students_in_class (y : ℕ) (H : 2 * y^2 + 6 * y + 9 = 490) : 
  y + (y + 3) = 31 := by
  sorry

end students_in_class_l639_639551


namespace hyperbola_touching_circle_l639_639389

noncomputable def hyperbola_equation_of_tangency (a b : ℝ) : Prop :=
  (∑ := ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∃ f : ℝ, f = 3 ∧ (b * 3 = 2 * sqrt (a^2 + b^2))) →
    (a = sqrt 5 ∧ b = 2))

theorem hyperbola_touching_circle :
  ∀ (a b : ℝ),
  (hyperbola_equation_of_tangency a b) ∧
  (∀ (x y : ℝ), hyperbola_equation_of_tangency a b) →
  (∃ (x y : ℝ),
    (∃ (k : ℝ), x^2 + y^2 - 6 * y + 5 = 0 ∧ k = 3 ∧ ∀ c: ℝ, c = 3 → c = k) →
    (∃ (y1 : ℝ),
      (y1^2 / 5 - x^2 / 4 = 1))) := sorry

end hyperbola_touching_circle_l639_639389


namespace rectangle_area_l639_639605

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_area_l639_639605


namespace abs_inequality_solution_rational_inequality_solution_l639_639582

theorem abs_inequality_solution (x : ℝ) : (|x - 2| + |2 * x - 3| < 4) ↔ (1 / 3 < x ∧ x < 3) :=
sorry

theorem rational_inequality_solution (x : ℝ) : 
  (x^2 - 3 * x) / (x^2 - x - 2) ≤ x ↔ (x ∈ Set.Icc (-1) 0 ∪ {1} ∪ Set.Ioi 2) := 
sorry

#check abs_inequality_solution
#check rational_inequality_solution

end abs_inequality_solution_rational_inequality_solution_l639_639582


namespace Chris_buys_48_golf_balls_l639_639590

theorem Chris_buys_48_golf_balls (total_golf_balls : ℕ) (dozen_to_balls : ℕ → ℕ)
  (dan_buys : ℕ) (gus_buys : ℕ) (chris_buys : ℕ) :
  dozen_to_balls 1 = 12 →
  dan_buys = 5 →
  gus_buys = 2 →
  total_golf_balls = 132 →
  (chris_buys * 12) + (dan_buys * 12) + (gus_buys * 12) = total_golf_balls →
  chris_buys * 12 = 48 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Chris_buys_48_golf_balls_l639_639590


namespace count_possible_classmates_l639_639002

theorem count_possible_classmates :
  let divisors := {d | d ∣ 100 ∧ d > 1 ∧ d ≤ 50} in
  divisors.card = 7 :=
by
  let divisors := {d | d ∣ 100 ∧ d > 1 ∧ d ≤ 50}
  show divisors.card = 7
  sorry

end count_possible_classmates_l639_639002


namespace max_tan_A_in_triangle_l639_639922

theorem max_tan_A_in_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] 
(hAB : dist A B = 24) (hBC : dist B C = 12) : 
  ∃ (angleA : ℝ), tan angleA = 1 / sqrt 3 :=
sorry

end max_tan_A_in_triangle_l639_639922


namespace h_value_at_1_l639_639494

def f (x : ℝ) : ℝ := x^2 + 3*x + 2

def g (x : ℝ) : ℝ := Real.sqrt (2 * f x) + 3

def h (x : ℝ) : ℝ := f (g x)

theorem h_value_at_1 : h 1 = 32 + 9 * Real.sqrt 12 :=
by
  unfold h g f
  sorry

end h_value_at_1_l639_639494


namespace quadratic_roots_real_and_values_l639_639858

theorem quadratic_roots_real_and_values (m : ℝ) (x : ℝ) :
  (x ^ 2 - x + 2 * m - 2 = 0) → (m ≤ 9 / 8) ∧ (m = 1 → (x = 0 ∨ x = 1)) :=
by
  sorry

end quadratic_roots_real_and_values_l639_639858


namespace largest_non_sum_l639_639524

open Set

def A_n (n : ℕ) (k : ℕ) : ℕ :=
  if n ≥ 2 ∧ 0 ≤ k ∧ k < n + 1 then 2^n - 2^k else 0

theorem largest_non_sum (n : ℕ) (h : n ≥ 2) :
  let S := {x | ∃ k : ℕ, 0 ≤ k ∧ k < n ∧ x = A_n n k} in
  ∀ m : ℕ, (¬∃ xs ∈ powerset S, xs.sum = m) ↔ m = (n-2) * 2^n + 1 :=
begin
  sorry
end

end largest_non_sum_l639_639524


namespace meadow_income_is_960000_l639_639980

theorem meadow_income_is_960000 :
  let boxes := 30
  let packs_per_box := 40
  let diapers_per_pack := 160
  let price_per_diaper := 5
  (boxes * packs_per_box * diapers_per_pack * price_per_diaper) = 960000 := 
by
  sorry

end meadow_income_is_960000_l639_639980


namespace calculate_land_tax_l639_639747

def plot_size : ℕ := 15
def cadastral_value_per_sotka : ℕ := 100000
def tax_rate : ℝ := 0.003

theorem calculate_land_tax :
  plot_size * cadastral_value_per_sotka * tax_rate = 4500 := 
by 
  sorry

end calculate_land_tax_l639_639747


namespace max_arithmetic_subsequences_l639_639997

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d c : ℤ), ∀ n : ℕ, a n = d * n + c

-- Condition that the sum of the indices is even
def sum_indices_even (n m : ℕ) : Prop :=
  (n % 2 = 0 ∧ m % 2 = 0) ∨ (n % 2 = 1 ∧ m % 2 = 1)

-- Maximum count of 3-term arithmetic sequences in a sequence of 20 terms
theorem max_arithmetic_subsequences (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) :
  ∃ n : ℕ, n = 180 :=
by
  sorry

end max_arithmetic_subsequences_l639_639997


namespace integral_sqrt_quarter_circle_l639_639306

noncomputable def integral_sqrt : ℝ :=
  ∫ x in 2..3, real.sqrt (1 - (x - 3) ^ 2)

theorem integral_sqrt_quarter_circle :
  integral_sqrt = π / 4 :=
sorry

end integral_sqrt_quarter_circle_l639_639306


namespace cards_distribution_l639_639817

def cards_distribution_possible (n : ℕ) (hn : n ≥ 3)
  (c : Fin (n + 1) → ℕ) : Prop :=
  let total_cards : ℕ := (Finset.univ : Finset (Fin (n + 1))).sum c in
  total_cards ≥ n^2 + 3 * n + 1 →
    ∃ c' : Fin (n + 1) → ℕ, (∀ i : Fin (n + 1), c' i ≥ n + 1)

theorem cards_distribution (n : ℕ) (hn : n ≥ 3)
  (c : Fin (n + 1) → ℕ)
  (htotal : (Finset.univ : Finset (Fin (n + 1))).sum c ≥ n^2 + 3 * n + 1) :
  ∃ c' : Fin (n + 1) → ℕ, (∀ i : Fin (n + 1), c' i ≥ n + 1) :=
sorry

end cards_distribution_l639_639817


namespace ten_sided_regular_polygon_has_10_equilateral_l639_639395

-- Define the vertices of the regular decagon
def vertices : fin 10 → Type := sorry

-- Define the condition of a regular polygon
def is_regular_polygon (sides : ℕ) (vertices : fin sides → Type) : Prop := sorry

-- Define the condition to check if a triangle is equilateral using vertices
def is_equilateral (v1 v2 v3 : Type) : Prop := sorry

-- Define the total number of distinct equilateral triangles
def distinct_equilateral_triangles (vertices : fin 10 → Type) : ℕ :=
  10

theorem ten_sided_regular_polygon_has_10_equilateral :
  is_regular_polygon 10 vertices →
  (∃ triangles : finite_set (finset (fin 10 × fin 10 × fin 10)),
    (∀ t ∈ triangles, is_equilateral (t.1) (t.2) (t.3)) ∧
    triangles.card = distinct_equilateral_triangles vertices
  ) :=
by
  intro h
  use sorry
  split; sorry

end ten_sided_regular_polygon_has_10_equilateral_l639_639395


namespace count_integers_in_range_with_increasing_digits_l639_639407

theorem count_integers_in_range_with_increasing_digits : 
  let integers_in_range := { n | 200 ≤ n ∧ n < 250 ∧ ((n % 10), (n / 10 % 10), (n / 100)) = (d₀, d₁, d₂) ∧ d₀ < d₁ ∧ d₁ < d₂ } in
  integers_in_range.card = 34 :=
sorry

end count_integers_in_range_with_increasing_digits_l639_639407


namespace non_empty_subsets_of_set_l639_639483

theorem non_empty_subsets_of_set (A : Set ℕ) (hA : A = {1, 2, 3, 4, 5}) :
  (2 ^ A.card - 1) = 31 :=
by
  have h : A.card = 5,
  { rw hA,
    sorry },
  rw [← h],
  sorry

end non_empty_subsets_of_set_l639_639483


namespace school_club_profit_l639_639202

def calculate_profit (bars_bought : ℕ) (cost_per_3_bars : ℚ) (bars_sold : ℕ) (price_per_4_bars : ℚ) : ℚ :=
  let cost_per_bar := cost_per_3_bars / 3
  let total_cost := bars_bought * cost_per_bar
  let price_per_bar := price_per_4_bars / 4
  let total_revenue := bars_sold * price_per_bar
  total_revenue - total_cost

theorem school_club_profit :
  calculate_profit 1200 1.50 1200 2.40 = 120 :=
by sorry

end school_club_profit_l639_639202


namespace d_minus_b_equals_757_l639_639528

theorem d_minus_b_equals_757 
  (a b c d : ℕ) 
  (h1 : a^5 = b^4) 
  (h2 : c^3 = d^2) 
  (h3 : c - a = 19) : 
  d - b = 757 := 
by 
  sorry

end d_minus_b_equals_757_l639_639528


namespace XF_XG_product_l639_639081

noncomputable def conditions (AB BC CD DA : ℝ) (X Y : ℝ) (AX AD CX BC : ℝ) (XG : ℝ) : Prop :=
  let BD := sqrt ((AB * CD + BC * DA) / AX) in
  direct_sum_uniform_space.ring X (XF * XG = (5 / 18) * BD^2)

theorem XF_XG_product :
  ∀ (AB BC CD DA AX AD CX BC XG : ℝ),
    AB = 4 →
    BC = 3 →
    CD = 7 →
    DA = 9 →
    ∃ X Y,
      ∃ E F,
        E = point.inter AX (point.parallel Y AD) →
        F = point.inter CX (point.parallel E BC) →
        let XF := point.length X F in
        let XG := point.length X G in
        XF * XG = (5 / 18) * (BD := sqrt ((AB * CD + BC * DA) / AX))^2 :=
  sorry

end XF_XG_product_l639_639081


namespace problem_l639_639755

-- Define the function v
def v (x : ℝ) : ℝ :=
  -x + 4 * Real.cos (Real.pi * x / 2)

-- State the theorem to prove the given problem
theorem problem :
  v (-2.67) + v (-1.21) + v (1.21) + v (2.67) = 0 :=
by
  sorry

end problem_l639_639755


namespace greatest_even_perfect_square_below_200_l639_639984

theorem greatest_even_perfect_square_below_200 : 
    ∃ (n : ℕ), (∃ (k : ℕ), n = k ^ 2 ∧ even n) ∧ n < 200 ∧ 
    (∀ (m : ℕ), (∃ (l : ℕ), m = l ^ 2 ∧ even m) ∧ m < 200 → m ≤ n) ↔ n = 196 :=
begin
  sorry
end

end greatest_even_perfect_square_below_200_l639_639984


namespace rock_game_probability_identity_l639_639725

/-- 
  Alice Czarina's rock game: we start with 2015 rocks, and at each round she removes k rocks uniformly.
  Let p be the probability that the number of rocks left after each round is a multiple of 5.
  If p = 5^a * 31^b * c/d, where a, b are integers and c, d are relatively prime to 5 * 31,
  then a + b is -501. This is a Lean statement for proving this result.
--/
theorem rock_game_probability_identity :
  let p := (1 / (2015 * (2015 - 1) * ... * 1 : ℚ)) * ∏ n in (range 2015).filter (λ x, x % 5 = 0), (x - 4) / x,
      a := -502,
      b := 1 in
  a + b = -501 :=
by sorry

end rock_game_probability_identity_l639_639725


namespace fraction_evaluation_l639_639305

theorem fraction_evaluation :
  (1/5 - 1/7) / (3/8 + 2/9) = 144/1505 := 
  by 
    sorry

end fraction_evaluation_l639_639305


namespace find_value_a_prove_inequality_l639_639350

noncomputable def arithmetic_sequence (a : ℕ) (S : ℕ → ℕ) (a_n : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 2 → S n * S n = 3 * n ^ 2 * a_n n + S (n - 1) * S (n - 1) ∧ a_n n ≠ 0

theorem find_value_a {S : ℕ → ℕ} {a_n : ℕ → ℕ} :
  (∃ (a : ℕ), arithmetic_sequence a S a_n) → a = 3 :=
sorry

noncomputable def sequence_bn (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) :=
  ∀ n : ℕ, b_n n = 1 / ((a_n n - 1) * (a_n n + 2))

theorem prove_inequality {S : ℕ → ℕ} {a_n : ℕ → ℕ} {b_n : ℕ → ℕ} {T : ℕ → ℕ} :
  (∃ (a : ℕ), arithmetic_sequence a S a_n) →
  (sequence_bn a_n b_n) →
  ∀ n : ℕ, T n < 1 / 6 :=
sorry

end find_value_a_prove_inequality_l639_639350


namespace count_integers_with_increasing_digits_200_to_250_l639_639425

theorem count_integers_with_increasing_digits_200_to_250 :
  {n : ℕ | 200 ≤ n ∧ n ≤ 250 ∧ 
          (let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 in 
           d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
           d1 < d2 ∧ d2 < d3)}.card = 11 :=
by
  sorry

end count_integers_with_increasing_digits_200_to_250_l639_639425


namespace acute_angle_sufficient_but_not_necessary_l639_639378

open Real InnerProductSpace

variables {E : Type*} [InnerProductSpace ℝ E] (a b : E)
hypothesis (unit_vectors : ∥a∥ = 1 ∧ ∥b∥ = 1)

def cosine_angle (x y : E) := ⟪x, y⟫ / (∥x∥ * ∥y∥)

theorem acute_angle_sufficient_but_not_necessary :
  (cosine_angle a b > 0) ↔ (∥a∥ = 1 ∧ ∥b∥ = 1) :=
by
  sorry

end acute_angle_sufficient_but_not_necessary_l639_639378


namespace remainder_of_h_x10_div_h_x_l639_639512

noncomputable def h (x : ℤ) : ℤ := x^5 - x^4 + x^3 - x^2 + x - 1

theorem remainder_of_h_x10_div_h_x (x : ℤ) : (h (x ^ 10)) % (h x) = -6 :=
by
  sorry

end remainder_of_h_x10_div_h_x_l639_639512


namespace sum_and_product_of_rational_roots_l639_639322

def h (x : ℚ) : ℚ := x^3 - 9 * x^2 + 27 * x - 27

theorem sum_and_product_of_rational_roots :
  (∑ r in (finset.filter (λ r, h r = 0) (finset.range 28)), r) = 3 ∧ 
  (∏ r in (finset.filter (λ r, h r = 0) (finset.range 28)), r) = 27 :=
by
  -- Conditions: 
  -- h(x) = x^3 - 9x^2 + 27x - 27
  -- Questions:
  -- Sum of rational roots = 3
  -- Product of rational roots = 27
  sorry

end sum_and_product_of_rational_roots_l639_639322


namespace paint_price_and_max_boxes_l639_639573

theorem paint_price_and_max_boxes (x y m : ℕ) 
  (h1 : x + 2 * y = 56) 
  (h2 : 2 * x + y = 64) 
  (h3 : 24 * m + 16 * (200 - m) ≤ 3920) : 
  x = 24 ∧ y = 16 ∧ m ≤ 90 := 
  by 
    sorry

end paint_price_and_max_boxes_l639_639573


namespace connie_mistake_l639_639754

theorem connie_mistake (x : ℝ) (h : x^2 = 144) : sqrt x = 2 * sqrt 3 :=
sorry

end connie_mistake_l639_639754


namespace petya_wins_last_l639_639400

--- Definitions and conditions
variables {a b c : ℝ}
variable h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a
variable h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
variable discriminant : ℝ × ℝ × ℝ → ℝ := λ (coeffs : ℝ × ℝ × ℝ), coeffs.2^2 - 4 * coeffs.1 * coeffs.3

noncomputable def petya_turn (coeffs : ℝ × ℝ × ℝ) : Prop :=
discriminant coeffs ≥ 0

noncomputable def vasya_turn (coeffs : ℝ × ℝ × ℝ) : Prop :=
discriminant coeffs < 0

variable order : {coeffs // petya_turn coeffs} ⊕ {coeffs // vasya_turn coeffs}  -- permutation results assuming order in such way 5 already settled
variable sequence : list ({coeffs // petya_turn coeffs} ⊕ {coeffs // vasya_turn coeffs})

axiom petya_three_first : sequence.take 3 = [sum.inl _, sum.inl _, sum.inl _]
axiom vasya_two_next : sequence.drop 3 = [sum.inr _, sum.inr _]

theorem petya_wins_last [inhabited sequence] :
  sequence.length = 5 → sum.inl (_ : {coeffs // petya_turn coeffs}) ∈ (sequence.take 6).nth_le 5 sorry :=
sorry

end petya_wins_last_l639_639400


namespace batsman_average_after_17th_inning_l639_639682

variable (A : ℝ) (total_runs : ℝ) (new_average : ℝ)
hypothesis h1 : total_runs = 16 * A + 87
hypothesis h2 : new_average =  (total_runs / 17)
hypothesis h3 : new_average = A + 3

theorem batsman_average_after_17th_inning :
  new_average = 39 :=
by
  sorry

end batsman_average_after_17th_inning_l639_639682


namespace derivative_of_y_l639_639671

noncomputable def y (x : ℝ) : ℝ := (3^x * (4 * sin(4 * x) + (Real.log 3) * cos(4 * x))) / (16 + (Real.log 3)^2)

theorem derivative_of_y (x : ℝ) : deriv y x = 3^x * cos(4 * x) :=
by
  sorry

end derivative_of_y_l639_639671


namespace Petya_receives_last_wrapper_l639_639399

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem Petya_receives_last_wrapper
  (h1 : discriminant a b c ≥ 0)
  (h2 : discriminant a c b ≥ 0)
  (h3 : discriminant b a c ≥ 0)
  (h4 : discriminant c a b < 0)
  (h5 : discriminant b c a < 0) :
  discriminant c b a ≥ 0 :=
sorry

end Petya_receives_last_wrapper_l639_639399


namespace compute_sum_l639_639266

open BigOperators

theorem compute_sum : 
  (1 / 2 ^ 2010 : ℝ) * ∑ n in Finset.range 1006, (-3 : ℝ) ^ n * (Nat.choose 2010 (2 * n)) = -1 / 2 :=
by
  sorry

end compute_sum_l639_639266


namespace driveways_shoveled_l639_639646

-- Define the constants and variables based on conditions.
def allowancePerMonth : ℕ := 5
def months : ℕ := 3
def allowanceTotal : ℕ := allowancePerMonth * months

def lawnsMowed : ℕ := 4
def earningsPerLawn : ℕ := 15
def mowingTotal : ℕ := lawnsMowed * earningsPerLawn

def earningsPerDriveway : ℕ := 7
-- Define D as the number of driveways shoveled
variable {D : ℕ}

def shoeCost : ℕ := 95
def changeAfterShoes : ℕ := 15

-- Total money Tobias had initially
def totalAmount (D : ℕ) : ℕ := allowanceTotal + mowingTotal + earningsPerDriveway * D + changeAfterShoes

-- The equation setup from problem statement
def totalAfterShoes : ℕ := shoeCost + changeAfterShoes

-- The final goal is proving the number of driveways shoveled
theorem driveways_shoveled (D : ℕ) : totalAmount D = totalAfterShoes → D = 2 := by
  let h : 110 = 110 := rfl
  have h1 : allowanceTotal + mowingTotal + changeAfterShoes = 90 := rfl
  have h2 : totalAfterShoes = 110 := h
  intro h_eq
  have h_driveways : 7 * D = 20 := by
    calc
      90 + 7 * D = totalAfterShoes := h_eq
      _ = 110 := h
      _ - 90 = 20 := by linarith
  have D_eq : D = 20 / 7 := by linarith
  have : 20 / 7 = 2 := rfl
  exact this ▸ D_eq.symm ▸ rfl


end driveways_shoveled_l639_639646


namespace count_integers_with_increasing_digits_200_to_250_l639_639428

theorem count_integers_with_increasing_digits_200_to_250 :
  {n : ℕ | 200 ≤ n ∧ n ≤ 250 ∧ 
          (let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 in 
           d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
           d1 < d2 ∧ d2 < d3)}.card = 11 :=
by
  sorry

end count_integers_with_increasing_digits_200_to_250_l639_639428


namespace total_amount_18423_53_l639_639687

noncomputable theory
open Real

def total_amount_invested 
  (T : ℝ) (H : ℝ) (interest_total : ℝ) (fraction_high_rate : ℝ) : Prop :=
  H = fraction_high_rate * T ∧ 
  interest_total = H * 0.09 + (T - H) * 0.06 ∧ 
  interest_total = 1440 ∧
  fraction_high_rate = 0.55 

theorem total_amount_18423_53 (T : ℝ) (H : ℝ) :
  total_amount_invested T H 1440 0.55 → T = 18823.53 :=
begin
  sorry
end

end total_amount_18423_53_l639_639687


namespace factor_polynomial_l639_639288

noncomputable def polynomial (x y n : ℤ) : ℤ := x^2 + 4 * x * y + 2 * x + n * y - n

theorem factor_polynomial (n : ℤ) :
  (∃ A B C D E F : ℤ, polynomial A B C = (A * x + B * y + C) * (D * x + E * y + F)) ↔ n = 0 :=
sorry

end factor_polynomial_l639_639288


namespace count_4_digit_numbers_with_conditions_l639_639872

def valid_4_digit_numbers : Type := {n : ℕ // 1000 ≤ n ∧ n < 10000}

def count_valid_numbers : ℕ :=
  ∑ x in finset.range 10, ∑ y in finset.range x, ite (x ≠ 1 ∧ y ≠ 1) 1 0

theorem count_4_digit_numbers_with_conditions :
  count_valid_numbers = 108 := sorry

end count_4_digit_numbers_with_conditions_l639_639872


namespace max_min_product_leq_neg_inv_2019_l639_639525

theorem max_min_product_leq_neg_inv_2019 
  (u : Fin 2019 → ℝ)
  (h1 : ∑ i, u i = 0)
  (h2 : ∑ i, (u i)^2 = 1) :
  let a := Finset.sup Finset.univ (λ i, u i),
      b := Finset.inf Finset.univ (λ i, u i)
  in a * b ≤ - (1 / 2019) :=
by
  sorry

end max_min_product_leq_neg_inv_2019_l639_639525


namespace sqrt_meaningful_range_l639_639645

theorem sqrt_meaningful_range (x : ℝ) : (√(x - 9)).isReal → x ≥ 9 :=
by
  sorry

end sqrt_meaningful_range_l639_639645


namespace not_raining_probability_l639_639114

theorem not_raining_probability (P_rain : ℚ) (h : P_rain = 4/9) : 1 - P_rain = 5 / 9 :=
by {
  rw h,
  norm_num,
  sorry
}

end not_raining_probability_l639_639114


namespace count_increasing_order_digits_l639_639415

def count_valid_numbers : ℕ :=
  let numbers := [234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249] in
  numbers.length

theorem count_increasing_order_digits : count_valid_numbers = 11 :=
by
  sorry

end count_increasing_order_digits_l639_639415


namespace greatest_face_area_of_box_l639_639450

theorem greatest_face_area_of_box (l w h : ℕ) (hl : l = 5) (hw : w = 2) (hh : h = 3) : (max (l * w) (max (l * h) (w * h))) = 15 := 
by
  rw [hl, hw, hh]
  simp
  sorry

end greatest_face_area_of_box_l639_639450


namespace solve_problem_1_solve_problem_2_l639_639580

-- Problem statement 1: Prove that the solutions to x(x-2) = x-2 are x = 1 and x = 2.
theorem solve_problem_1 (x : ℝ) : (x * (x - 2) = x - 2) ↔ (x = 1 ∨ x = 2) :=
  sorry

-- Problem statement 2: Prove that the solutions to 2x^2 + 3x - 5 = 0 are x = 1 and x = -5/2.
theorem solve_problem_2 (x : ℝ) : (2 * x^2 + 3 * x - 5 = 0) ↔ (x = 1 ∨ x = -5 / 2) :=
  sorry

end solve_problem_1_solve_problem_2_l639_639580


namespace binomial_sum_real_part_l639_639269

theorem binomial_sum_real_part :
  (1 / (2:ℝ)^(2010)) * ∑ n in Finset.range (1005 + 1), (-3:ℝ)^n * Nat.choose 2010 (2 * n) = -1 / 2 := 
sorry

end binomial_sum_real_part_l639_639269


namespace log_sqrt_12_eq_7_l639_639301

theorem log_sqrt_12_eq_7 : log (1728 * sqrt 12) (sqrt 12) = 7 :=
by
  sorry

end log_sqrt_12_eq_7_l639_639301


namespace sum_squares_bound_l639_639533

theorem sum_squares_bound (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) : 
  ab + bc + cd + da ≤ 4 :=
sorry

end sum_squares_bound_l639_639533


namespace volume_of_extended_region_of_cube_equals_1059_l639_639274

theorem volume_of_extended_region_of_cube_equals_1059 :
  let side_length := 4
  let extended_length := side_length + 2 * 2
  let original_volume := side_length ^ 3
  let extended_volume := extended_length ^ 3
  let extending_region_volume := extended_volume - original_volume
  let corner_contributions := (8 : ℝ) * (4 / 3 * π * (2 ^ 3 / 8))
  let edge_contributions := (12 : ℝ) * (π * 2 ^ 2 * 4)
  let total_volume := extending_region_volume + corner_contributions + edge_contributions
  in total_volume = 448 + 608 / 3 * π ∧ 448 + 608 + 3 = 1059 := by
  sorry

end volume_of_extended_region_of_cube_equals_1059_l639_639274


namespace area_proof_l639_639612

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l639_639612


namespace sum_of_distances_l639_639362

theorem sum_of_distances (A B C P : ℝ → ℝ) (h_eq_triangle : A - B = 2 ∧ B - C = 2 ∧ C - A = 2)
  (P_on_BC : ∃ β : ℝ, 0 ≤ β ∧ β ≤ 1 ∧ (1- β) * B + β * C = P) :
  P - A + P - B + P - C = 6 := 
sorry

end sum_of_distances_l639_639362


namespace average_is_correct_l639_639151

def nums : List ℝ := [13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_is_correct :
  (nums.sum / nums.length) = 125830.8 :=
by sorry

end average_is_correct_l639_639151


namespace count_integers_between_200_250_l639_639439

theorem count_integers_between_200_250 :
  {n : ℕ // 200 ≤ n ∧ n < 250 ∧ 
            (let d2 := (n / 10) % 10, d3 := n % 10 in
             (n / 100 = 2) ∧ (d2 ≠ d3) ∧ (2 < d2) ∧ (d2 < d3)
            )}.to_finset.card = 11 :=
by
  -- Start the proof process here
  sorry

end count_integers_between_200_250_l639_639439


namespace prob_in_sync_l639_639649

-- Define what it means for two people to guess numbers in sync
def in_sync (a b : ℕ) : Prop := |a - b| ≤ 1

-- Prove the probability of being in sync is 4/9
theorem prob_in_sync : 
  let outcomes := (finset.range 6).product (finset.range 6),
      favorable := outcomes.filter (λ ab, in_sync (ab.1 + 1) (ab.2 + 1))
  in (favorable.card : ℚ) / outcomes.card = 4 / 9 :=
by 
  sorry

end prob_in_sync_l639_639649


namespace rectangle_area_l639_639617

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l639_639617


namespace trigonometric_identity_l639_639341

theorem trigonometric_identity 
  (a b x : ℝ) (n : ℕ) (h : n > 0) 
  (h1 : (sin x) ^ 4 / a + (cos x) ^ 4 / b = 1 / (a + b)) :
  (sin x) ^ (4 * n) / (a ^ (2 * n - 1)) + (cos x) ^ (4 * n) / (b ^ (2 * n - 1)) = 
  1 / ((a + b) ^ (2 * n - 1)) := 
by
  sorry

end trigonometric_identity_l639_639341


namespace cost_per_pumpkin_pie_l639_639259

theorem cost_per_pumpkin_pie
  (pumpkin_pies : ℕ)
  (cherry_pies : ℕ)
  (cost_cherry_pie : ℕ)
  (total_profit : ℕ)
  (selling_price : ℕ)
  (total_revenue : ℕ)
  (total_cost : ℕ)
  (cost_pumpkin_pie : ℕ)
  (H1 : pumpkin_pies = 10)
  (H2 : cherry_pies = 12)
  (H3 : cost_cherry_pie = 5)
  (H4 : total_profit = 20)
  (H5 : selling_price = 5)
  (H6 : total_revenue = (pumpkin_pies + cherry_pies) * selling_price)
  (H7 : total_cost = total_revenue - total_profit)
  (H8 : total_cost = pumpkin_pies * cost_pumpkin_pie + cherry_pies * cost_cherry_pie) :
  cost_pumpkin_pie = 3 :=
by
  -- Placeholder for proof
  sorry

end cost_per_pumpkin_pie_l639_639259


namespace find_k_in_isosceles_triangle_l639_639223

theorem find_k_in_isosceles_triangle 
  (A B C D : Type)
  (h_triangle : triangle ABC)
  (h_acute : acute ABC)
  (h_isosceles : isosceles ABC)
  (h_circumscribed : circumscribed ABC)
  (h_tangents : tangents B C D)
  (h_angle_relation : ∠ABC = ∠ACB ∧ ∠ABC = 3 * ∠D)
  : ∠BAC = (5 * π / 11) := 
by 
  sorry

end find_k_in_isosceles_triangle_l639_639223


namespace distinct_elements_triangle_not_isosceles_l639_639392

theorem distinct_elements_triangle_not_isosceles
  {a b c : ℝ} (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  ¬(a = b ∨ b = c ∨ a = c) := by
  sorry

end distinct_elements_triangle_not_isosceles_l639_639392


namespace possible_values_of_a_l639_639802

-- Define A and B
def A (a : ℝ) : set ℝ := {x | a * x + 2 = 0}
def B : set ℝ := {x | x^2 - 3 * x + 2 = 0}

-- State the proof problem
theorem possible_values_of_a :
  ∀ a : ℝ, (A a ⊆ B) ↔ (a = -2 ∨ a = -1) :=
by
  sorry

end possible_values_of_a_l639_639802


namespace price_of_paint_models_max_boxes_of_paint_A_l639_639577

-- Define the conditions as hypotheses
def price_paint_model_A_B (x y : ℕ) : Prop :=
  x + 2 * y = 56 ∧ 2 * x + y = 64

def total_cost_constraint (x y m : ℕ) : Prop :=
  24 * m + 16 * (200 - m) ≤ 3920

-- Prove the prices of the paint models given the conditions
theorem price_of_paint_models :
  ∃ (x y : ℕ), price_paint_model_A_B x y ∧ x = 24 ∧ y = 16 :=
by
  sorry

-- Prove the maximum number of boxes of paint model A the school can purchase given the total cost constraint
theorem max_boxes_of_paint_A :
  ∃ (m : ℕ), total_cost_constraint 24 16 m ∧ m = 90 :=
by
  sorry

end price_of_paint_models_max_boxes_of_paint_A_l639_639577


namespace find_missing_number_l639_639889

theorem find_missing_number :
  ∃ n : ℕ, let x := 755 in
  let known_numbers := [744, 745, 747, 749, 752, 752, 753, 755, x] in
  let total_sum := 750 * 10 in
  let sum_known_numbers := list.sum known_numbers in
  total_sum - sum_known_numbers = n ∧ n = 1748 :=
begin
  use 1748,
  let x := 755,
  let known_numbers := [744, 745, 747, 749, 752, 752, 753, 755, x],
  let total_sum := 750 * 10,
  let sum_known_numbers := list.sum known_numbers,
  split,
  { exact total_sum - sum_known_numbers },
  { exact 1748 }
end

end find_missing_number_l639_639889


namespace flea_reaches_1_0_l639_639005

def can_flea_reach (m n : ℕ) : Prop :=
  (m % 2 = 0 ∨ n % 2 = 0) ∧ Nat.gcd m n = 1

theorem flea_reaches_1_0 (m n : ℕ) : can_flea_reach m n ↔ 
  (∃ k : ℕ, (0, 0) →*{(x, y) | (x + m, y + n) ∨ (x + n, y + m) ∨ (x - m, y - n) ∨ (x - n, y - m)} (1, 0)) :=
sorry

end flea_reaches_1_0_l639_639005


namespace num_possible_multisets_l639_639279

theorem num_possible_multisets (a0 a1 a2 a3 a4 a5 a6 a7 a8 : ℤ) 
  (r : ℤ) 
  (h1 : a8 * r^8 + a7 * r^7 + a6 * r^6 + a5 * r^5 + a4 * r^4 + a3 * r^3 + a2 * r^2 + a1 * r + a0 = 0)
  (h2 : a0 * (1 / r)^8 + a1 * (1 / r)^7 + a2 * (1 / r)^6 + a3 * (1 / r)^5 + a4 * (1 / r)^4 + a5 * (1 / r)^3 + a6 * (1 / r)^2 + a7 * (1 / r) + a8 = 0) 
  (h3 : r ≠ 0) :
  ∃ (S : multiset ℤ), S.card = 8 ∧ ∀ (x ∈ S), x = 1 ∨ x = -1 ∧ multiset.nodup S ∧ multiset.length S = 8 ∧ multiset.count 1 S ≤ 8 ∧ multiset.count (-1) S ≤ 8 :=
sorry

end num_possible_multisets_l639_639279


namespace garden_ratio_l639_639201

theorem garden_ratio (L W : ℝ) (h1 : 2 * L + 2 * W = 180) (h2 : L = 60) : L / W = 2 :=
by
  -- this is where you would put the proof
  sorry

end garden_ratio_l639_639201


namespace sequence_is_arithmetic_sum_of_reciprocal_sequence_l639_639921

section problem_statement

variable {a : ℕ → ℕ}

-- Conditions
axiom cond1 : a 1 = 4
axiom cond2 : ∀ n : ℕ, n > 0 → n * a (n + 1) - (n + 1) * a n = 2 * n^2 + 2 * n

-- Part (Ⅰ): Prove that {a_n / n} is an arithmetic sequence.
theorem sequence_is_arithmetic (h1 : cond1) (h2 : cond2) (n : ℕ) :
  (a (n + 1) / (n + 1) - a n / n) = 2 :=
sorry

-- Part (Ⅱ): Find the sum of the first n terms of {1 / a_n}.
theorem sum_of_reciprocal_sequence (h1 : cond1) (h2 : cond2) (n : ℕ) :
  let S_n := (∑ i in range n, 1 / a (i + 1)) in
  S_n = n / (2 * (n + 1)) :=
sorry

end problem_statement

end sequence_is_arithmetic_sum_of_reciprocal_sequence_l639_639921


namespace area_of_MBCN_l639_639822

-- Define the points and proportional conditions
variables (A B C D P M N : Point)
variable  [affine.Points Point]

-- Define the quadrilateral ABCD with area 45
def quadrilateral_ABCD := 45

-- Define the conditions given
def MB_ratio := (MB : ℝ) = (1/3) * (AB : ℝ)
def NC_ratio := (NC : ℝ) = (2/3) * (DC : ℝ)
def BP_ratio := (BP : ℝ) = (3/5) * (BD : ℝ)
def PC_ratio := (PC : ℝ) = (2/3) * (AC : ℝ)

-- Define the quadrilateral MBCN with the correct area to prove
theorem area_of_MBCN (A B C D P M N quadrilateral_ABCD : ℝ) (MB_ratio NC_ratio BP_ratio PC_ratio : ℝ) :
  area_ABC M B C N = 79 / 3 :=
by
    sorry

end area_of_MBCN_l639_639822


namespace car_speed_ratio_l639_639723

-- Assuming the bridge length as L, pedestrian's speed as v_p, and car's speed as v_c.
variables (L v_p v_c : ℝ)

-- Mathematically equivalent proof problem statement in Lean 4.
theorem car_speed_ratio (h1 : 2/5 * L = 2/5 * L)
                       (h2 : (L - 2/5 * L) / v_p = L / v_c) :
    v_c = 5 * v_p := 
  sorry

end car_speed_ratio_l639_639723


namespace quadratic_has_negative_root_sufficiency_quadratic_has_negative_root_necessity_l639_639824

theorem quadratic_has_negative_root_sufficiency 
  (a : ℝ) : (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) → (a < 0) :=
sorry

theorem quadratic_has_negative_root_necessity 
  (a : ℝ) : (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (a < 0) :=
sorry

end quadratic_has_negative_root_sufficiency_quadratic_has_negative_root_necessity_l639_639824


namespace seven_lines_divide_into_29_regions_l639_639086

open Function

theorem seven_lines_divide_into_29_regions : 
  ∀ n : ℕ, (∀ l m : ℕ, l ≠ m → l < n ∧ m < n) → 1 + n + (n.choose 2) = 29 :=
by
  sorry

end seven_lines_divide_into_29_regions_l639_639086


namespace range_a_decreasing_function_l639_639109

def f (a x : ℝ) : ℝ :=
  if x < 0 then -x + 3 * a else -(x + 1)^2 + 2

theorem range_a_decreasing_function :
  ∀ a : ℝ, (∀ x y : ℝ, x ≤ y → f a x ≥ f a y) ↔ a ≥ 1/3 :=
by
  sorry

end range_a_decreasing_function_l639_639109


namespace find_prime_coeffs_l639_639594

open Real

def polynomial (p q : ℤ) : ℝ → ℝ := λ x, x^2 + (p:ℝ) * x + (q:ℝ)

def prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

def roots_distance_eq_1 (p q : ℤ) : Prop :=
  let Δ := (p:ℝ)^2 - 4 * (q:ℝ)
  (Δ^0.5)^2 = 1

theorem find_prime_coeffs (p q : ℤ) (hp : prime p) (hq : prime q) (hd : roots_distance_eq_1 p q) : 
  p = 3 ∧ q = 2 := by
  sorry

end find_prime_coeffs_l639_639594


namespace answered_both_correctly_l639_639170

variable (A B : Prop)
variable (P_A P_B P_not_A_and_not_B P_A_and_B : ℝ)

axiom P_A_eq : P_A = 0.75
axiom P_B_eq : P_B = 0.35
axiom P_not_A_and_not_B_eq : P_not_A_and_not_B = 0.20

theorem answered_both_correctly (h1 : P_A = 0.75) (h2 : P_B = 0.35) (h3 : P_not_A_and_not_B = 0.20) : 
  P_A_and_B = 0.30 :=
by
  sorry

end answered_both_correctly_l639_639170


namespace circumcircle_tangents_fixed_circles_l639_639791

/-- Given:
  - a circle Γ,
  - a line ℓ tangent to Γ,
  - another circle Ω disjoint from ℓ such that Γ and Ω lie on opposite sides of ℓ,
  - tangents to Γ from a variable point X on Ω meet ℓ at Y and Z.

Prove:
  - That as X varies over Ω, the circumcircle of triangle XYZ is tangent to two fixed circles.
-/
theorem circumcircle_tangents_fixed_circles
  (Γ : set Point)
  (Ω : set Point)
  (ℓ : set Point)
  (tangent_Γ_ℓ : tangent ℓ Γ)
  (disjoint_Ω_ℓ : disjoint Ω ℓ)
  (opposite_sides_Γ_Ω : opposite_sides ℓ Γ Ω)
  (variable_point_X : Point)
  (X_on_Ω : X ∈ Ω)
  (tangent_points_Y_Z : ∃ Y Z, Y ∈ ℓ ∧ Z ∈ ℓ ∧ tangent (X, Y) Γ ∧ tangent (X, Z) Γ) :
  ∃ (fixed_circle1 fixed_circle2 : set Point),
    ∀ (X ∈ Ω) (Y Z : Point),
      Y ∈ ℓ ∧ Z ∈ ℓ ∧ tangent (X, Y) Γ ∧ tangent (X, Z) Γ →
      tangent (circumcircle (triangle X Y Z)) fixed_circle1 ∧
      tangent (circumcircle (triangle X Y Z)) fixed_circle2 :=
sorry

end circumcircle_tangents_fixed_circles_l639_639791


namespace circumcircles_are_tangent_l639_639566

-- Define the given conditions and main problem statement in Lean 4
variable {A B C D O : Type} [Field A] [Field B] [Field C] [Field D] [Field O]

-- Assumptions
variable (circumcircle : ∀ {X Y Z : Type}, Type)  -- Circumcircle definition
variable (perpendicular : ∀ {X Y : Type}, Type)  -- Perpendicular definition
variable (tangent : ∀ {X : Type}, Type)  -- Tangent definition
variable (inscribed : ∀ {X : Type}, Type)  -- Inscribed quadrilateral
variable (circle_center : ∀ {X : Type}, Type)  -- Circle's center
variable (intersect : ∀ {X Y : Type}, Type)  -- Intersection definition

-- Given conditions rewritten from the mathematical problem
def quadrilateral_inscribed_with_perpendicular_diagonals (A B C D O : Type) :
  inscribed A B C D ∧
  perpendicular A O ∧
  perpendicular C O ∧
  circle_center O := sorry

-- The tangents at points A and C and the intersecting line BD form triangle Δ
def triangle_formed_by_tangents_and_line (A B C D O Δ : Type) :
  tangent A ∧
  tangent C ∧
  intersect B D Δ := sorry

-- Main statement to prove
theorem circumcircles_are_tangent (A B C D O Δ : Type) :
  quadrilateral_inscribed_with_perpendicular_diagonals A B C D O →
  triangle_formed_by_tangents_and_line A B C D O Δ →
  tangent (circumcircle B O D) (circumcircle Δ) := sorry

end circumcircles_are_tangent_l639_639566


namespace sum_of_elements_in_A_otimes_B_proper_subsets_of_A_otimes_B_l639_639281

def A : Set ℝ := {0, 2}
def B : Set ℝ := {1, 2}
def A_otimes_B := {z | ∃ x ∈ A, ∃ y ∈ B, z = x * y + x / y}

theorem sum_of_elements_in_A_otimes_B : ∑ z in A_otimes_B, z = 9 := by
  admit  

theorem proper_subsets_of_A_otimes_B :
  {∅, {0}, {4}, {5}, {0, 4}, {0, 5}, {4, 5}} = {s | s ⊆ A_otimes_B ∧ s ≠ A_otimes_B} := by
  admit

end sum_of_elements_in_A_otimes_B_proper_subsets_of_A_otimes_B_l639_639281


namespace log_base_sqrt12_of_1728_sqrt12_eq_seven_l639_639303

theorem log_base_sqrt12_of_1728_sqrt12_eq_seven : 
  log (sqrt 12) (1728 * sqrt 12) = 7 := 
by 
  sorry

end log_base_sqrt12_of_1728_sqrt12_eq_seven_l639_639303


namespace pages_per_hour_l639_639126

-- Definitions corresponding to conditions
def lunch_time : ℕ := 4 -- time taken to grab lunch and come back (in hours)
def total_pages : ℕ := 4000 -- total pages in the book
def book_time := 2 * lunch_time  -- time taken to read the book is twice the lunch_time

-- Statement of the problem to be proved
theorem pages_per_hour : (total_pages / book_time = 500) := 
  by
    -- We assume the definitions and want to show the desired property
    sorry

end pages_per_hour_l639_639126


namespace count_odd_integers_in_range_l639_639324

theorem count_odd_integers_in_range : (set_of (λ n : ℤ, 25 < n^2 ∧ n^2 < 144 ∧ odd n)).card = 6 :=
by
  sorry

end count_odd_integers_in_range_l639_639324


namespace k_value_correct_l639_639244

-- Let k be the value such that ∠BAC = k * π
def k : ℝ := 5 / 11

-- Define the isosceles triangle ABC inscribed in a circle
variables (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
variables [tangents_through B C D] (angle_BAC ABC ACB D : ℝ)

# Define the angles:
-- ∠ABC = ∠ACB = 3 * ∠D
axiom angle_equivalence : (ABC = 3 * D) ∧ (ACB = 3 * D)
-- ∠BAC = k * π
axiom angle_BAC_def : angle_BAC = k * real.pi

-- The proof assertion:
theorem k_value_correct (h : k = 5/11) : 
  (∀ (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
     [tangents_through B C D] (angle_BAC ABC ACB D : ℝ),
  angle_equivalence → angle_BAC_def) → 
  angle_BAC = (5 / 11 : ℝ) * real.pi :=
by
  sorry

end k_value_correct_l639_639244


namespace arithmetic_sequence_nine_to_twelve_l639_639469

def arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (n + 1) * a 0 + (n * (n + 1) / 2) * a 1

theorem arithmetic_sequence_nine_to_twelve {a : ℕ → ℝ} 
  (S : ℕ → ℝ)
  (h1 : S 4 = 4)
  (h2 : S 8 = 12) :
  (S 12 - S 8) = 12 :=
by
  sorry

end arithmetic_sequence_nine_to_twelve_l639_639469


namespace tangent_circumcircle_of_triangle_TSH_l639_639914

namespace Geometry

open EuclideanGeometry

def Quadrilateral (A B C D : Point) := Convex A B C D

def foot_of_perpendicular (A H : Point) (BD : Line) : Prop :=
  perpendicular A H BD

def inside_triangle (H S C T : Point) : Prop :=
  in_interior_of_triangle H S C T

def angle_condition1 (C H S B : Point) : Prop :=
  angle_diff_eq C H S C S B 90

def angle_condition2 (H T C D : Point) : Prop :=
  angle_diff_eq H T C D T C 90

theorem tangent_circumcircle_of_triangle_TSH
  (A B C D H T S : Point)
  (BD : Line) :
  Quadrilateral A B C D ∧
  (∠ B A C) = 90 ∧
  (∠ C D A) = 90 ∧
  foot_of_perpendicular A H BD ∧
  S ∈ Line AB ∧
  T ∈ Line CD ∧
  inside_triangle H S C T ∧
  angle_condition1 C H S B ∧
  angle_condition2 H T C D
  → tangent_line (circumcircle T S H) BD :=
begin
  sorry
end

end Geometry

end tangent_circumcircle_of_triangle_TSH_l639_639914


namespace perimeter_is_11_or_13_l639_639016

-- Define the basic length properties of the isosceles triangle
structure IsoscelesTriangle where
  a b c : ℝ
  a_eq_b_or_a_eq_c : a = b ∨ a = c

-- Conditions of the problem
def sides : IsoscelesTriangle where
  a := 3
  b := 5
  c := 5
  a_eq_b_or_a_eq_c := Or.inr rfl

def sides' : IsoscelesTriangle where
  a := 3
  b := 3
  c := 5
  a_eq_b_or_a_eq_c := Or.inl rfl

-- Prove the perimeters
theorem perimeter_is_11_or_13 : (sides.a + sides.b + sides.c = 13) ∨ (sides'.a + sides'.b + sides'.c = 11) :=
by
  sorry

end perimeter_is_11_or_13_l639_639016


namespace interesting_numbers_count_500_l639_639282

noncomputable def g1 (n : ℕ) : ℕ :=
  if n = 1 then 1
  else
    let prime_factors := unique_factorization_monoid.factors n
    prime_factors.foldr (λ p acc, acc * (p + 2)) 1 * prime_factors.foldr (λ p acc, acc / p) 1

noncomputable def g (m n : ℕ) : ℕ :=
  if m = 1 then g1 n else g1 (g (m - 1) n)

def is_unbounded_sequence (n : ℕ) : Prop :=
  ∀ K : ℕ, ∃ k : ℕ, k ≥ K ∧ g k n ≥ K

def interesting_sequence_count (N : ℕ) : ℕ :=
  (finset.range (N + 1)).filter (λ n, is_unbounded_sequence n).card

theorem interesting_numbers_count_500 : interesting_sequence_count 500 = 38 := sorry

end interesting_numbers_count_500_l639_639282


namespace karl_savings_proof_l639_639040

-- Definitions based on the conditions
def original_price_per_notebook : ℝ := 3.00
def sale_discount : ℝ := 0.25
def extra_discount_threshold : ℝ := 10
def extra_discount_rate : ℝ := 0.05

-- The number of notebooks Karl could have purchased instead
def notebooks_purchased : ℝ := 12

-- The total savings calculation
noncomputable def total_savings : ℝ := 
  let original_total := notebooks_purchased * original_price_per_notebook
  let discounted_price_per_notebook := original_price_per_notebook * (1 - sale_discount)
  let extra_discount := if notebooks_purchased > extra_discount_threshold then discounted_price_per_notebook * extra_discount_rate else 0
  let total_price_after_discounts := notebooks_purchased * discounted_price_per_notebook - notebooks_purchased * extra_discount
  original_total - total_price_after_discounts

-- Formal statement to prove
theorem karl_savings_proof : total_savings = 10.35 := 
  sorry

end karl_savings_proof_l639_639040


namespace number_of_intersections_l639_639655

def values := {1, 2, 3, 4, 5, 6}

theorem number_of_intersections : ∃ (A B C D : ℕ), A ∈ values ∧ B ∈ values ∧ C ∈ values ∧ D ∈ values ∧
  A ≠ C ∧ (((D - B) % (A - C) = 0) ∨ ((D - B) % (A - C) = 0)) ∧ 2 * 2 * 2 * 2 = 90 := sorry

end number_of_intersections_l639_639655


namespace not_divisible_a1a2_l639_639948

theorem not_divisible_a1a2 (a1 a2 b1 b2 : ℕ) (h1 : 1 < b1) (h2 : b1 < a1) (h3 : 1 < b2) (h4 : b2 < a2) (h5 : b1 ∣ a1) (h6 : b2 ∣ a2) :
  ¬ (a1 * a2 ∣ a1 * b1 + a2 * b2 - 1) :=
by
  sorry

end not_divisible_a1a2_l639_639948


namespace find_a_l639_639970

theorem find_a (a : ℝ) (ha_nonzero : a ≠ 0) 
  (h_eq : (∑ k in finset.range 6, (nat.choose 5 k) * (a * (λ x : ℝ, x))^k * 1^(5 - k) = 
            ∑ k in finset.range 6, (nat.choose 5 k) * (a * (λ x : ℝ, x))^(k + 1) * 1^(5 - k - 1))) : 
  a = 2 := 
by 
  sorry

end find_a_l639_639970


namespace total_area_of_L_shaped_figure_l639_639904

-- Define the specific lengths for each segment
def bottom_rect_length : ℕ := 10
def bottom_rect_width : ℕ := 6
def central_rect_length : ℕ := 4
def central_rect_width : ℕ := 4
def top_rect_length : ℕ := 5
def top_rect_width : ℕ := 1

-- Calculate the area of each rectangle
def bottom_rect_area : ℕ := bottom_rect_length * bottom_rect_width
def central_rect_area : ℕ := central_rect_length * central_rect_width
def top_rect_area : ℕ := top_rect_length * top_rect_width

-- Given the length and width of the rectangles, calculate the total area of the L-shaped figure
theorem total_area_of_L_shaped_figure : 
  bottom_rect_area + central_rect_area + top_rect_area = 81 := by
  sorry

end total_area_of_L_shaped_figure_l639_639904


namespace sarah_ninth_finger_written_value_l639_639598

def g : ℕ → ℕ
| 0 := 0
| 1 := 8
| 2 := 5
| 3 := 0
| 4 := 7
| 5 := 3
| 6 := 9
| 7 := 2
| 8 := 1
| 9 := 4
| _ := 0  -- Default case to cover full ℕ domain

def sarah_written_value (n : ℕ) : ℕ :=
  if n = 1 then 5
  else g (sarah_written_value (n - 1))

theorem sarah_ninth_finger_written_value : sarah_written_value 9 = 0 :=
by sorry

end sarah_ninth_finger_written_value_l639_639598


namespace count_integers_with_increasing_digits_200_to_250_l639_639426

theorem count_integers_with_increasing_digits_200_to_250 :
  {n : ℕ | 200 ≤ n ∧ n ≤ 250 ∧ 
          (let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 in 
           d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
           d1 < d2 ∧ d2 < d3)}.card = 11 :=
by
  sorry

end count_integers_with_increasing_digits_200_to_250_l639_639426


namespace compute_g_i_l639_639523

def g (x : Complex) : Complex := (x^5 + 3 * x^3 + 2 * x) / (x^2 + 2 * x + 2)

theorem compute_g_i : g Complex.I = -4 - 2 * Complex.I := by
  -- The proof would go here
  sorry

end compute_g_i_l639_639523


namespace positive_projection_l639_639367

variables {V : Type*} [inner_product_space ℝ V]
variables {a b : V}
variables ha_norm : ‖a‖ = 2
variables hb_norm : ‖b‖ = 2
variables angle_ab : real.angle (∠ (a, b)) = real.pi / 3

theorem positive_projection (ha_norm : ‖a‖ = 2) (hb_norm : ‖b‖ = 2) (angle_ab : real.angle (∠ (a, b)) = real.pi / 3) :
  ‖(a + b)‖ * real.cos (real.pi / 6) = 3 :=
sorry

end positive_projection_l639_639367


namespace probability_one_head_three_tosses_l639_639629

theorem probability_one_head_three_tosses : 
  let p := 1 / 2
  in (3.choose 1) * (p^1) * ((1 - p)^2) = 3 / 8 := 
by sorry

end probability_one_head_three_tosses_l639_639629


namespace area_proof_l639_639610

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l639_639610


namespace rectangle_area_l639_639625

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l639_639625


namespace find_a_value_l639_639373

theorem find_a_value
  (a : ℝ)
  (h : ∀ x, 0 ≤ x ∧ x ≤ (π / 2) → a * Real.sin x + Real.cos x ≤ 2)
  (h_max : ∃ x, 0 ≤ x ∧ x ≤ (π / 2) ∧ a * Real.sin x + Real.cos x = 2) :
  a = Real.sqrt 3 :=
sorry

end find_a_value_l639_639373


namespace num_subsets_of_a_l639_639501

noncomputable def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | a*x = 1}

theorem num_subsets_of_a :
  let possible_a := {a | ∃ x ∈ A, B a = {x} ∨ B a = ∅};
  finite possible_a ∧ possible_a.to_finset.card = 3
  → 2^possible_a.to_finset.card = 8 := by
    sorry

end num_subsets_of_a_l639_639501


namespace sqrt_expression_l639_639749

theorem sqrt_expression : Real.sqrt ((4^2) * (5^6)) = 500 := by
  sorry

end sqrt_expression_l639_639749


namespace least_n_l639_639828

theorem least_n (n : ℕ) (h : 1 ≤ n) : 
  (1 / n.to_real) - (1 / (n + 1).to_real) < 1 / 20.to_real ↔ n = 5 := 
begin
  sorry
end

end least_n_l639_639828


namespace time_for_r_alone_l639_639669

theorem time_for_r_alone
  (r_s_together: ℕ) (s_alone: ℕ) (r_alone: ℕ) 
  (h1: r_s_together = 10)
  (h2: s_alone = 20)
  (h3: ∀ t_r, (1 / t_r) + (1 / s_alone) = (1 / r_s_together) → r_alone = t_r) :
  r_alone = 20 :=
by
  have t_r_eq : ∃ t_r, (1 / t_r) = (1 / r_s_together) - (1 / s_alone), 
  {
    use 20,
    calc
      (1 / r_s_together) - (1 / s_alone)
      = (1 / 10) - (1 / 20) : by rw [h1, h2]
      = 1 / 20 : by norm_num
  },
  cases t_r_eq with t_r ht_r,
  apply h3 t_r,
  exact ht_r,
  sorry -- Skipping the full proof

end time_for_r_alone_l639_639669


namespace train_length_approx_l639_639210

noncomputable def relative_speed_in_m_per_s (v_t v_m : ℕ) : ℚ :=
let relative_speed_km_per_hr := v_t - v_m in
(relative_speed_km_per_hr : ℚ) * (5 / 18)

noncomputable def length_of_train (t : ℚ) (v_t v_m : ℕ) : ℚ :=
(relative_speed_in_m_per_s v_t v_m) * t

theorem train_length_approx (t : ℚ) (v_m v_t : ℕ) (h_t : t = 47.99616030717543) (h_vm : v_m = 5) (h_vt : v_t = 65) :
  length_of_train t v_t v_m ≈ 799.94 :=
by
  sorry

end train_length_approx_l639_639210


namespace selling_price_correct_l639_639174

-- Define the conditions
def purchase_price : ℝ := 12000
def repair_costs : ℝ := 5000
def transportation_charges : ℝ := 1000
def profit_percentage : ℝ := 0.50

-- Calculate total cost
def total_cost : ℝ := purchase_price + repair_costs + transportation_charges

-- Define the selling price and the proof goal
def selling_price : ℝ := total_cost + (profit_percentage * total_cost)

-- Prove that the selling price equals Rs 27000
theorem selling_price_correct : selling_price = 27000 := 
by 
  -- Proof is not required, so we use sorry
  sorry

end selling_price_correct_l639_639174


namespace binomial_sum_real_part_l639_639268

theorem binomial_sum_real_part :
  (1 / (2:ℝ)^(2010)) * ∑ n in Finset.range (1005 + 1), (-3:ℝ)^n * Nat.choose 2010 (2 * n) = -1 / 2 := 
sorry

end binomial_sum_real_part_l639_639268


namespace maxwell_age_l639_639897

theorem maxwell_age (M : ℕ) (h1 : ∃ n : ℕ, n = M + 2) (h2 : ∃ k : ℕ, k = 4) (h3 : (M + 2) = 2 * 4) : M = 6 :=
sorry

end maxwell_age_l639_639897


namespace dot_product_MA_MB_l639_639474

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4) / x

theorem dot_product_MA_MB (t : ℝ) (ht : t > 0) :
  let M := (t, f t) in
  let A := ((t + (f t)) / 2, (t + (f t)) / 2) in
  let B := (t, 0) in
  (let distance (p1 p2 : ℝ × ℝ) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) in
  let MA := distance M A in
  let MB := distance M B in
  let cos135 := - real.sqrt(2) / 2 in
  MA * MB * cos135) = -2 :=
by
  sorry

end dot_product_MA_MB_l639_639474


namespace number_of_subsets_of_A_l639_639018

open Finset

theorem number_of_subsets_of_A (U A : Finset ℕ) (hU : U = {0, 1, 2}) (hA_complement : U \ A = {2}) :
  card (powerset A) = 4 :=
sorry

end number_of_subsets_of_A_l639_639018


namespace mean_power_inequality_l639_639564

variable {α β : ℝ}
variable {n : ℕ}
variable {x : ℝ}
variable {xs : Fin n → ℝ}

noncomputable def S_alpha (α : ℝ) (xs : Fin n → ℝ) : ℝ :=
  ( ( finset.univ.sum (λ i, (xs i) ^ α) / n ) ) ^ (1 / α)

theorem mean_power_inequality
  (h1 : α < β)
  (h2 : α * β ≠ 0) :
  S_alpha α xs ≤ S_alpha β xs :=
sorry

end mean_power_inequality_l639_639564


namespace product_of_nonreal_roots_l639_639787

def polynomial_eqn := x^6 - 6*x^5 + 15*x^4 - 20*x^3 + 15*x^2 - 6*x - 1996 = 0

theorem product_of_nonreal_roots : 
  (∀x, polynomial_eqn x) → 
  (∏ (r : ℂ) in { r : ℂ | polynomial_eqn r ∧ r ≠ 1 }, r) = 
  1 + complex.sqrt 1997 := 
sorry

end product_of_nonreal_roots_l639_639787


namespace find_sachins_age_l639_639173

variable (S R : ℕ)

theorem find_sachins_age (h1 : R = S + 8) (h2 : S * 9 = R * 7) : S = 28 := by
  sorry

end find_sachins_age_l639_639173


namespace cos_angle_tangent_circle_l639_639344

theorem cos_angle_tangent_circle (x y : ℝ) (h1: (x-1)^2 + (y-1)^2 = 1) :
  ∃ (A B : ℝ × ℝ), 
  let P := (3, 2)
  let C := (1, 1)
  let dist := √((3 - 1) ^ 2 + (2 - 1) ^ 2) 
  let cos_APC := 2 / dist
  let cos_APB := 2 * (cos_APC)^2 - 1
  cos_APB = 3 / 5 := sorry

end cos_angle_tangent_circle_l639_639344


namespace find_k_l639_639221

-- Definitions based on conditions in step a)
def acute_isosceles_triangle_inscribed (A B C : Type) : Prop := sorry -- Formal definition of the triangle being acute isosceles and inscribed in a circle
def tangents_meeting_at_point (A B C D : Type) : Prop := sorry -- Formal definition of tangents through B and C meeting at D
def angle_relation (ABC D : Type) (theta : ℝ) : Prop := 3 * theta = sorry -- Formal definition of \(\angle ABC = \angle ACB = 3 \angle D\)
def angle_BAC (k : ℝ) (theta : ℝ) : Prop := theta = k * real.pi -- Formal definition of \(\angle BAC = k \pi\)

-- Theorem statement for our proof problem
theorem find_k
  (A B C D : Type)
  (h1 : acute_isosceles_triangle_inscribed A B C)
  (h2 : tangents_meeting_at_point A B C D)
  (theta : ℝ)
  (h3 : angle_relation ABC D theta)
  (k : ℝ)
  (h4 : angle_BAC k theta) :
  k = 1 / 13 := by
  sorry

end find_k_l639_639221


namespace solve_for_A_l639_639160

theorem solve_for_A (A : ℕ) (h1 : 3 + 68 * A = 691) (h2 : 68 * A < 1000) (h3 : 68 * A ≥ 100) : A = 8 :=
by
  sorry

end solve_for_A_l639_639160


namespace multiples_of_4_count_100_350_l639_639442

theorem multiples_of_4_count_100_350 : 
  let count_multiples_of_4 := (λ (x₀ x₁ n : ℕ), 
    let first := if h : x₀ % n = 0 then x₀ else x₀ + (n - x₀ % n) in 
    let last := if h : x₁ % n = 0 then x₁ else x₁ - (x₁ % n) in 
    (last - first) / n + 1) in
  count_multiples_of_4 100 350 4 = 62 := 
by 
  let count_multiples_of_4 := (λ (x₀ x₁ n : ℕ), 
    let first := if h : x₀ % n = 0 then x₀ else x₀ + (n - x₀ % n) in 
    let last := if h : x₁ % n = 0 then x₁ else x₁ - (x₁ % n) in 
    (last - first) / n + 1)
  show count_multiples_of_4 100 350 4 = 62
  sorry

end multiples_of_4_count_100_350_l639_639442


namespace find_lambda_range_l639_639849

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 - (2 * a + 2) * x + (2 * a + 1) * log x

theorem find_lambda_range :
  ∀ (a : ℝ) (x1 x2 : ℝ), (3/2) ≤ a ∧ a ≤ 5/2 ∧ 1 ≤ x1 ∧ x1 ≤ 2 ∧ 1 ≤ x2 ∧ x2 ≤ 2 →
  abs (f x1 a - f x2 a) ≤ λ * abs (1/x1 - 1/x2) ↔ λ ∈ set.Ici 8 :=
sorry

end find_lambda_range_l639_639849


namespace quadratic_coefficient_is_one_l639_639347

theorem quadratic_coefficient_is_one :
  ∀ (x : ℝ), (x^2 - 2 * x + 1 = 0) → ∃ a b c : ℝ, (a = 1) ∧ (b = -2) ∧ (c = 1) := by
  intro x
  intro h
  use 1, -2, 1
  split
  repeat { sorry }

end quadratic_coefficient_is_one_l639_639347


namespace fraction_of_students_with_buddy_l639_639465

variable (s n : ℕ)

theorem fraction_of_students_with_buddy (h : n = 4 * s / 3) : 
  (n / 4 + s / 3) / (n + s) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l639_639465


namespace paint_price_max_boxes_paint_A_l639_639571

theorem paint_price :
  ∃ x y : ℕ, (x + 2 * y = 56 ∧ 2 * x + y = 64) ∧ x = 24 ∧ y = 16 := by
  sorry

theorem max_boxes_paint_A (m : ℕ) :
  24 * m + 16 * (200 - m) ≤ 3920 → m ≤ 90 := by
  sorry

end paint_price_max_boxes_paint_A_l639_639571


namespace max_profit_at_one_device_l639_639192

noncomputable def revenue (x : ℕ) : ℝ := 30 * x - 0.2 * x^2

def fixed_monthly_cost : ℝ := 40

def material_cost_per_device : ℝ := 5

noncomputable def cost (x : ℕ) : ℝ := fixed_monthly_cost + material_cost_per_device * x

noncomputable def profit_function (x : ℕ) : ℝ := (revenue x) - (cost x)

noncomputable def marginal_profit_function (x : ℕ) : ℝ :=
  profit_function (x + 1) - profit_function x

theorem max_profit_at_one_device :
  marginal_profit_function 1 = 24.4 ∧
  ∀ x : ℕ, marginal_profit_function x ≤ 24.4 := sorry

end max_profit_at_one_device_l639_639192


namespace rational_count_correct_l639_639729

theorem rational_count_correct :
  (∃ (l : List ℚ), l = [-1, 0, 22/7, 3.14]) ∧
  (∀ x ∈ [-1, 0, 22/7, 3.14], x ∈ [-1, 0, 22/7, 3.14] → x ≠ 4.11213415 ∧ x ≠ π/2) ∧
  (∀ y ∈ [-1,0,π/2,4.11213415,22/7,3.14],
    x ∈ [-1, 0, 22/7, 4.11213415, π/2, 3.14] → x ∉ [-1, 0, 22/7, 3.14] 
  → ∃ t : ℚ, t ≠ π/2 ∧ t ≠ 4.11213415) ∧
  (#[-1, 0, 22/7, 3.14] = 4) :=
sorry


end rational_count_correct_l639_639729


namespace largest_quantity_l639_639666

theorem largest_quantity : 
  let D := (2007 / 2006) + (2007 / 2008) in
  let E := (2008 / 2007) + (2010 / 2007) in
  let F := (2009 / 2008) + (2009 / 2010) in
  E > D ∧ E > F := 
sorry

end largest_quantity_l639_639666


namespace count_increasing_order_digits_l639_639416

def count_valid_numbers : ℕ :=
  let numbers := [234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249] in
  numbers.length

theorem count_increasing_order_digits : count_valid_numbers = 11 :=
by
  sorry

end count_increasing_order_digits_l639_639416


namespace quadratic_solution_range_l639_639298

theorem quadratic_solution_range :
  ∃ x : ℝ, x^2 + 12 * x - 15 = 0 ∧ 1.1 < x ∧ x < 1.2 :=
sorry

end quadratic_solution_range_l639_639298


namespace k_value_correct_l639_639243

-- Let k be the value such that ∠BAC = k * π
def k : ℝ := 5 / 11

-- Define the isosceles triangle ABC inscribed in a circle
variables (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
variables [tangents_through B C D] (angle_BAC ABC ACB D : ℝ)

# Define the angles:
-- ∠ABC = ∠ACB = 3 * ∠D
axiom angle_equivalence : (ABC = 3 * D) ∧ (ACB = 3 * D)
-- ∠BAC = k * π
axiom angle_BAC_def : angle_BAC = k * real.pi

-- The proof assertion:
theorem k_value_correct (h : k = 5/11) : 
  (∀ (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
     [tangents_through B C D] (angle_BAC ABC ACB D : ℝ),
  angle_equivalence → angle_BAC_def) → 
  angle_BAC = (5 / 11 : ℝ) * real.pi :=
by
  sorry

end k_value_correct_l639_639243


namespace simplify_complex_expression_l639_639579

theorem simplify_complex_expression (i : ℂ) (h : i^2 = -1) : 
  7 * (4 - 2 * i) - 2 * i * (3 - 4 * i) = 20 - 20 * i := 
by
  sorry

end simplify_complex_expression_l639_639579


namespace average_after_17th_inning_l639_639680

-- Define the conditions.
variable (A : ℚ) -- The initial average after 16 innings

-- Define the score in the 17th inning and the increment in the average.
def runs_in_17th_inning : ℚ := 87
def increment_in_average : ℚ := 3

-- Define the equation derived from the conditions.
theorem average_after_17th_inning :
  (16 * A + runs_in_17th_inning) / 17 = A + increment_in_average →
  A + increment_in_average = 39 :=
sorry

end average_after_17th_inning_l639_639680


namespace rectangle_area_l639_639603

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_area_l639_639603


namespace minimize_barrel_cost_l639_639760

theorem minimize_barrel_cost :
  ∃ (r h : ℝ), 
  (π * r^2 * h = π / 2) ∧
  (3π * r^2 + 40π * r * (1 / (2 * r^2))) = 30 * (3 ^ (1 / 3)) * π ∧
  r = (1 / 3) ^ (1 / 3) ∧
  h = (3 ^ (1 / 3)) / 2 :=
sorry

end minimize_barrel_cost_l639_639760


namespace original_number_of_people_l639_639035

/-- Initially, one-third of the people in a room left.
Then, one-fourth of those remaining started to dance.
There were then 18 people who were not dancing.
What was the original number of people in the room? -/
theorem original_number_of_people (x : ℕ) 
  (h_one_third_left : ∀ y : ℕ, 2 * y / 3 = x) 
  (h_one_fourth_dancing : ∀ y : ℕ, y / 4 = x) 
  (h_non_dancers : x / 2 = 18) : 
  x = 36 :=
sorry

end original_number_of_people_l639_639035


namespace quadrilateral_area_correct_l639_639567

open Real

variables (A B C D E : Point) (x y : ℝ)
variables (angle_ABC angle_ACD : Angle)

noncomputable def quadrilateral_area :=
  let AE := 6
  let EC := 12
  let AC := 18
  let CD := 24
  -- Areas
  let area_ACD := (1 / 2) * AC * CD
  let area_ABC := 9 * x
  -- Pythagorean relationship
  in if angle_ABC = 90 ∧ angle_ACD = 90 ∧ x^2 + y^2 = AC^2
     then area_ABC + area_ACD
     else 0

-- Statement of the theorem
theorem quadrilateral_area_correct :
  angle_ABC = 90 ∧ angle_ACD = 90 ∧ 18 = 6 * 3 ∧ EC = 6 * 2 ∧ AC = 18 ∧ CD = 24 ∧ (x^2 + y^2 = 18^2) →
  quadrilateral_area A B C D E x y angle_ABC angle_ACD = 216 + (36 * sqrt 5) :=
by sorry

end quadrilateral_area_correct_l639_639567


namespace sum_of_b_values_with_rational_roots_l639_639765

theorem sum_of_b_values_with_rational_roots : 
  let quadratic_eq_has_rational_roots (b : ℕ) := ∃ (x y : ℚ), y ≠ 0 ∧ (3 * x^2 + 7 * x + b = 0)
  let discriminant_is_perfect_square (b : ℕ) := ∃ (k : ℕ), 49 - 12 * b = k^2
  (∑ b in {b : ℕ | (b > 0) ∧ quadratic_eq_has_rational_roots b ∧ discriminant_is_perfect_square b}, b) = 6 :=
by sorry

end sum_of_b_values_with_rational_roots_l639_639765


namespace simon_can_make_blueberry_pies_l639_639089

theorem simon_can_make_blueberry_pies (bush1 bush2 blueberries_per_pie : ℕ) (h1 : bush1 = 100) (h2 : bush2 = 200) (h3 : blueberries_per_pie = 100) : 
  (bush1 + bush2) / blueberries_per_pie = 3 :=
by
  -- Proof goes here
  sorry

end simon_can_make_blueberry_pies_l639_639089


namespace find_common_ratio_l639_639123

noncomputable def geometric_series (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a_1 * (1 - q^n) / (1 - q)

theorem find_common_ratio (a_1 : ℝ) (q : ℝ) (n : ℕ) (S_n : ℕ → ℝ)
  (h1 : ∀ n, S_n n = geometric_series a_1 q n)
  (h2 : S_n 3 = (2 * a_1 + a_1 * q) / 2)
  : q = -1/2 :=
  sorry

end find_common_ratio_l639_639123


namespace evaluate_expression_l639_639304

theorem evaluate_expression : (2^2010 * 3^2012 * 25) / 6^2011 = 37.5 := by
  sorry

end evaluate_expression_l639_639304


namespace profit_percentage_is_20_l639_639717

-- Definitions
def CP : ℝ := 180
def discount : ℝ := 45
def markup_percentage : ℝ := 0.45

-- Markup calculation
def markup := markup_percentage * CP

-- Marked price calculation
def MP := CP + markup

-- Selling price calculation
def SP := MP - discount

-- Profit calculation
def profit := SP - CP

-- Profit percentage calculation
def profit_percentage := (profit / CP) * 100

-- Theorem to prove
theorem profit_percentage_is_20 :
  profit_percentage = 20 := 
by
  sorry

end profit_percentage_is_20_l639_639717


namespace find_15th_term_l639_639293

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 2 = 4 ∧ ∀ n, a (n + 2) = a n

theorem find_15th_term :
  ∃ a : ℕ → ℕ, seq a ∧ a 15 = 3 :=
by
  sorry

end find_15th_term_l639_639293


namespace find_a_find_b_plus_c_l639_639485

variable (a b c : ℝ)
variable (A B C : ℝ) -- Angles in radians

-- Condition: Given that 2a / cos A = (3c - 2b) / cos B
axiom condition1 : 2 * a / (Real.cos A) = (3 * c - 2 * b) / (Real.cos B)

-- Condition 1: b = sqrt(5) * sin B
axiom condition2 : b = Real.sqrt 5 * (Real.sin B)

-- Proof problem for finding a
theorem find_a : a = 5 / 3 := by
  sorry

-- Condition 2: a = sqrt(6) and the area is sqrt(5) / 2
axiom condition3 : a = Real.sqrt 6
axiom condition4 : 1 / 2 * b * c * (Real.sin A) = Real.sqrt 5 / 2

-- Proof problem for finding b + c
theorem find_b_plus_c : b + c = 4 := by
  sorry

end find_a_find_b_plus_c_l639_639485


namespace salad_dressing_percentage_l639_639996

variable (P Q : ℝ) -- P and Q are the amounts of dressings P and Q in grams

-- Conditions
variable (h1 : 0.3 * P + 0.1 * Q = 12) -- The combined vinegar percentage condition
variable (h2 : P + Q = 100)            -- The total weight condition

-- Statement to prove
theorem salad_dressing_percentage (P_percent : ℝ) 
    (h1 : 0.3 * P + 0.1 * Q = 12) (h2 : P + Q = 100) : 
    P / (P + Q) * 100 = 10 :=
sorry

end salad_dressing_percentage_l639_639996


namespace fifteenth_term_is_three_l639_639295

noncomputable def sequence (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 else 4

theorem fifteenth_term_is_three : sequence 15 = 3 :=
  by sorry

end fifteenth_term_is_three_l639_639295


namespace price_of_paint_models_max_boxes_of_paint_A_l639_639575

-- Define the conditions as hypotheses
def price_paint_model_A_B (x y : ℕ) : Prop :=
  x + 2 * y = 56 ∧ 2 * x + y = 64

def total_cost_constraint (x y m : ℕ) : Prop :=
  24 * m + 16 * (200 - m) ≤ 3920

-- Prove the prices of the paint models given the conditions
theorem price_of_paint_models :
  ∃ (x y : ℕ), price_paint_model_A_B x y ∧ x = 24 ∧ y = 16 :=
by
  sorry

-- Prove the maximum number of boxes of paint model A the school can purchase given the total cost constraint
theorem max_boxes_of_paint_A :
  ∃ (m : ℕ), total_cost_constraint 24 16 m ∧ m = 90 :=
by
  sorry

end price_of_paint_models_max_boxes_of_paint_A_l639_639575


namespace sum_squares_bound_l639_639534

theorem sum_squares_bound (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) : 
  ab + bc + cd + da ≤ 4 :=
sorry

end sum_squares_bound_l639_639534


namespace part1_part2_l639_639675

noncomputable def is_monotonically_increasing (f' : ℝ → ℝ) := ∀ x, f' x ≥ 0

noncomputable def is_monotonically_decreasing (f' : ℝ → ℝ) (I : Set ℝ) := ∀ x ∈ I, f' x ≤ 0

def f' (a x : ℝ) : ℝ := 3 * x ^ 2 - a

theorem part1 (a : ℝ) : 
  is_monotonically_increasing (f' a) ↔ a ≤ 0 := sorry

theorem part2 (a : ℝ) : 
  is_monotonically_decreasing (f' a) (Set.Ioo (-1 : ℝ) (1 : ℝ)) ↔ a ≥ 3 := sorry

end part1_part2_l639_639675


namespace probability_each_delegate_next_to_diff_country_l639_639648

open BigOperators

-- Definitions corresponding to the given conditions
def total_delegates := 12
def countries := 3
def delegates_per_country := 4

-- Statement of the proof problem in Lean 4
theorem probability_each_delegate_next_to_diff_country (p q : ℕ) 
  (h : Nat.gcd 16 17 = 1)
  (hpq : 16 + 17 = 33) :
  p = 16 ∧ q = 17 ∧ p + q = 33 :=
by
  use 16
  use 17
  simp [Nat.gcd_eq_one_iff_coprime.mpr (Nat.coprime_of_prime_prime (Nat.prime_of_eq_prime 2 8).symm (Nat.prime_of_eq_prime 2 9).symm)]
  exact h
  exact hpq
sorry

end probability_each_delegate_next_to_diff_country_l639_639648


namespace poisson_inequalities_poisson_inequalities_central_limit_poisson_l639_639674

noncomputable theory

open ProbabilityTheory

variable {λ : ℝ} (hλ : 0 < λ)

/-- Lemma: Poisson distribution probabilities inequalities --/
theorem poisson_inequalities (n : ℕ) (hn1 : 0 ≤ n ∧ n < λ) :
  P(X_λ ≤ n) ≤ (λ / (λ - n)) * P(X_λ = n) :=
sorry

theorem poisson_inequalities (n : ℕ) (hn2 : n > λ - 1) :
  P(X_λ ≤ n) ≤ ((n + 1) / (n + 1 - λ)) * P(X_λ = n) :=
sorry

/-- Theorem: Central limit theorem for Poisson distribution --/
theorem central_limit_poisson (n : ℕ) :
  tendsto (λ n, P(X_λ ≥ n) / P(λ.sqrt * ξ ≥ n - λ)) at_top (𝓝 ∞) :=
sorry

end poisson_inequalities_poisson_inequalities_central_limit_poisson_l639_639674


namespace power_function_no_origin_l639_639013

theorem power_function_no_origin (m : ℝ) :
  (m^2 - 3 * m + 3 ≠ 0) → ((m^2 - 3 * m + 3 = 1) ∧ (m^2 - m - 2 ≤ 0)) → (m = 1 ∨ m = 2) :=
by
  intro h1 h2
  cases h2 with h_eq h_le
  sorry

end power_function_no_origin_l639_639013


namespace probability_floor_eq_l639_639200

theorem probability_floor_eq (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1000) :
    let y := x / 2.5
    let b := y % 5
    let a := (y - b) / 5
    let floor_eq := ⌊(⌊y⌋ / 2.5)⌋ = ⌊y / 2.5⌋
    (∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1000) → 
    let b := x % 5
    let a := (x - b) / 2.5
    ((floor_eq = true) ↔ (b < 2.5 ∨ b ≥ 3))) →
    ∫ (x in 0..1000), if ⌊ (⌊(x / 2.5)⌋ / 2.5) ⌋ = ⌊(x / 6.25)⌋ then 1 else 0 / 1000 = 9 / 10 := sorry

end probability_floor_eq_l639_639200


namespace combined_towel_weight_l639_639979

/-
Given:
1. Mary has 5 times as many towels as Frances.
2. Mary has 3 times as many towels as John.
3. The total weight of their towels is 145 pounds.
4. Mary has 60 towels.

To prove: 
The combined weight of Frances's and John's towels is 22.863 kilograms.
-/

theorem combined_towel_weight (total_weight_pounds : ℝ) (mary_towels frances_towels john_towels : ℕ) 
  (conversion_factor : ℝ) (combined_weight_kilograms : ℝ) :
  mary_towels = 60 →
  mary_towels = 5 * frances_towels →
  mary_towels = 3 * john_towels →
  total_weight_pounds = 145 →
  conversion_factor = 0.453592 →
  combined_weight_kilograms = 22.863 :=
by
  sorry

end combined_towel_weight_l639_639979


namespace bus_speed_including_stoppages_l639_639774

theorem bus_speed_including_stoppages :
  ∀ (s t : ℝ), s = 75 → t = 24 → (s * ((60 - t) / 60)) = 45 :=
by
  intros s t hs ht
  rw [hs, ht]
  sorry

end bus_speed_including_stoppages_l639_639774


namespace fn_factorial_l639_639804

open Nat

noncomputable def C (n k : ℕ) : ℕ :=
  if h : k ≤ n then (factorial n) / ((factorial k) * (factorial (n - k)))
  else 0

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in Finset.range (n + 1), (-1) ^ k * (C n k) * (x - k) ^ n

theorem fn_factorial (n : ℕ) (x : ℝ) : f n x = n! :=
by
  sorry

end fn_factorial_l639_639804


namespace triangle_area_l639_639486

theorem triangle_area (A B C: Point) (Q: Point)
  (area_ΔABQ: ℝ) (area_ΔBCQ: ℝ) (area_ΔCAQ: ℝ)
  (h1: area_ΔABQ = 16) (h2: area_ΔBCQ = 25) (h3: area_ΔCAQ = 36)
  (h4: lines_parallel_to_sides_through_Q (Δ A B C) Q): 
  area (Δ A B C) = 77 :=
by 
  sorry

end triangle_area_l639_639486


namespace quintuplets_babies_l639_639910

theorem quintuplets_babies (t r q : ℕ) (h1 : r = 6 * q)
  (h2 : t = 2 * r)
  (h3 : 2 * t + 3 * r + 5 * q = 1500) :
  5 * q = 160 :=
by
  sorry

end quintuplets_babies_l639_639910


namespace area_of_triangle_l639_639673

theorem area_of_triangle (A B C M P O : Type) [Triangle ABC] 
  (h1 : angle_bisector A ∩ side BC = M) 
  (h2 : angle_bisector B ∩ side AC = P) 
  (h3 : angle_bisectors_intersect AM BP O)
  (h4 : similar_triangles BOM AOP)
  (h5 : BO = (1 + Real.sqrt 3) * OP)
  (h6 : BC = 1) :
  area ABC = Real.sqrt 3 / 4 := 
by 
  sorry

end area_of_triangle_l639_639673


namespace OT_bisects_PQ_minimize_ratio_TF_PQ_l639_639354

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | let (x, y) := p in x^2 / 6 + y^2 / 2 = 1 }

def focus_left : (ℝ × ℝ) := (-2, 0)

def point_T (m : ℝ) : (ℝ × ℝ) := (-3, m)

def intersects_ellipse (F : ℝ × ℝ) (T : ℝ × ℝ) : Set (ℝ × ℝ) :=
  { p | let (x, y) := p in p ∈ ellipse ∧ (y - F.2) * (T.1 - F.1) = (y - T.2) * (F.1 - T.1) }

theorem OT_bisects_PQ (m : ℝ) : 
  let T := point_T m,
      F := focus_left,
      P := (⊤, ⊤), -- Placeholder for intersection point 1
      Q := (⊤, ⊤)  -- Placeholder for intersection point 2
  midpoint (
    let (px₁, py₁) := P, (px₂, py₂) := Q in
    (px₁ + px₂) / 2, (py₁ + py₂) / 2
  ) lies_on_line (0, 0) T := sorry

theorem minimize_ratio_TF_PQ : 
  ∃ (m : ℝ), m = 1 ∨ m = -1 ∧ 
  let T := point_T m,
      F := focus_left,
      len_TF := dist T F,
      P := (⊤, ⊤), -- Placeholder for intersection point 1
      Q := (⊤, ⊤), -- Placeholder for intersection point 2
      len_PQ := dist P Q
  in (len_TF / len_PQ) = sqrt 3 / 3 := sorry

end OT_bisects_PQ_minimize_ratio_TF_PQ_l639_639354


namespace compute_expression_l639_639510

def f (x : ℕ) : ℕ := x + 3
def g (x : ℕ) : ℕ := x / 4
def h (x : ℕ) : ℕ := x + 1

def f_inv (x : ℕ) : ℕ := x - 3
def g_inv (x : ℕ) : ℕ := 4 * x
def h_inv (x : ℕ) : ℕ := x - 1

theorem compute_expression : f (h (g_inv (h_inv (f_inv (h (g (f 20)))))))) = 0 :=
by
  sorry

end compute_expression_l639_639510


namespace count_increasing_order_digits_l639_639412

def count_valid_numbers : ℕ :=
  let numbers := [234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249] in
  numbers.length

theorem count_increasing_order_digits : count_valid_numbers = 11 :=
by
  sorry

end count_increasing_order_digits_l639_639412


namespace Larry_tenth_finger_value_l639_639940

def f : ℕ → ℕ
| 4 := 3
| 1 := 8
| 7 := 2
| _ := 0  -- default case not listed in given conditions

def g : ℕ → ℕ
| 3 := 1
| 8 := 7
| 2 := 1
| _ := 0  -- default case not listed in given conditions

theorem Larry_tenth_finger_value :
  let seq := list.foldl (λ acc i, if i % 2 == 0 then f acc else g acc) 4 [1,2,3,4,5,6,7,8,9]
  in seq = 2 :=
by 
  sorry  -- proof goes here

end Larry_tenth_finger_value_l639_639940


namespace soap_box_width_l639_639195

-- Conditions from the problem
def carton_length : ℝ := 30
def carton_width : ℝ := 42
def carton_height : ℝ := 60
def soap_length : ℝ := 7
def soap_height : ℝ := 5
def max_soap_boxes : ℝ := 360

-- The volume of the carton
def carton_volume : ℝ := carton_length * carton_width * carton_height

-- The volume of one soap box
def soap_volume (W : ℝ) : ℝ := soap_length * W * soap_height

-- The equation given the maximum number of soap boxes
def soap_boxes_volume_eq (W : ℝ) : ℝ := max_soap_boxes * soap_volume(W)

-- The actual proof statement we need to show
theorem soap_box_width :
  ∃ W : ℝ, soap_boxes_volume_eq(W) = carton_volume :=
begin
  -- We're skipping the proof here with sorry to just write the statement
  use 6,
  sorry,
end

end soap_box_width_l639_639195


namespace count_integers_between_200_250_l639_639436

theorem count_integers_between_200_250 :
  {n : ℕ // 200 ≤ n ∧ n < 250 ∧ 
            (let d2 := (n / 10) % 10, d3 := n % 10 in
             (n / 100 = 2) ∧ (d2 ≠ d3) ∧ (2 < d2) ∧ (d2 < d3)
            )}.to_finset.card = 11 :=
by
  -- Start the proof process here
  sorry

end count_integers_between_200_250_l639_639436


namespace james_vegetable_consumption_l639_639929

theorem james_vegetable_consumption : 
  (1/4 + 1/4) * 2 * 7 + 3 = 10 := 
by
  sorry

end james_vegetable_consumption_l639_639929


namespace coefficient_x2_l639_639916

/-- Given the expansion of (\frac{1}{x} + 2)(1 - x)^4, the coefficient of x^2 is 8. -/
theorem coefficient_x2 (x : ℝ) (h : x ≠ 0) : 
  ((1/x + 2) * (1 - x)^4).coeff 2 = 8 :=
sorry

end coefficient_x2_l639_639916


namespace middle_lines_bisect_each_other_l639_639069

theorem middle_lines_bisect_each_other
  (A B C D E F G H : Point)
  (hE : midpoint E A B)
  (hF : midpoint F B C)
  (hG : midpoint G C D)
  (hH : midpoint H D A) :
  ∃ M : Point, midpoint M E G ∧ midpoint M F H := 
sorry

end middle_lines_bisect_each_other_l639_639069


namespace find_password_in_45_operations_l639_639203

theorem find_password_in_45_operations :
  ∃ plan : (Π (inputs : list (list ℕ)), list (list ℕ)), ∀ password : list ℕ,
  (∀ group : list ℕ, |group| = 8 → ∃ order, plan [group] = [order]) →
  (length (plan (generate_inputs password)) ≤ 45) :=
sorry

end find_password_in_45_operations_l639_639203


namespace max_value_of_expression_l639_639055

theorem max_value_of_expression (x y : ℝ) (h : x + y = 4) :
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 ≤ 7225 / 28 :=
sorry

end max_value_of_expression_l639_639055


namespace reading_rate_l639_639133

-- Definitions based on conditions
def one_way_trip_time : ℕ := 4
def round_trip_time : ℕ := 2 * one_way_trip_time
def read_book_time : ℕ := 2 * round_trip_time
def book_pages : ℕ := 4000

-- The theorem to prove Juan's reading rate is 250 pages per hour.
theorem reading_rate : book_pages / read_book_time = 250 := by
  sorry

end reading_rate_l639_639133


namespace parametric_function_f_l639_639110

theorem parametric_function_f (f : ℚ → ℚ)
  (x y : ℝ) (t : ℚ) :
  y = 20 * t - 10 →
  y = (3 / 4 : ℝ) * x - 15 →
  x = f t →
  f t = (80 / 3) * t + 20 / 3 :=
by
  sorry

end parametric_function_f_l639_639110


namespace log_base_sqrt12_of_1728_sqrt12_eq_seven_l639_639302

theorem log_base_sqrt12_of_1728_sqrt12_eq_seven : 
  log (sqrt 12) (1728 * sqrt 12) = 7 := 
by 
  sorry

end log_base_sqrt12_of_1728_sqrt12_eq_seven_l639_639302


namespace AQ_length_l639_639966

-- Definitions of lengths and angle bisector
def AD : ℝ := 7 / 4
def AP : ℝ := 21 / 2
def AB : ℝ := 14 / 11

-- Main theorem statement
theorem AQ_length (AQ : ℝ) (bisects_bad : ∀ {A B C D P Q : Type}, (AC bisects ∠BAD)) :
  AQ = 42 / 13 :=
sorry

end AQ_length_l639_639966


namespace initials_count_is_276_l639_639550

-- Given conditions as Lean definitions
def first_initial_fixed : Char := 'B'
def valid_middle_initials : Finset Char := Finset.filter (λ c, 'C' ≤ c ∧ c ≤ 'Z') (Finset.range (Char.ofNat 26))
def valid_last_initials (m : Char) : Finset Char := Finset.filter (λ c, m < c ∧ c ≠ 'A' ∧ 'C' ≤ c ∧ c ≤ 'Z') (Finset.range (Char.ofNat 26))

-- Number of valid sets of initials
def count_valid_initial_sets : ℕ :=
  valid_middle_initials.sum (λ m, (valid_last_initials m).card)

-- Proof statement
theorem initials_count_is_276 : count_valid_initial_sets = 276 :=
sorry

end initials_count_is_276_l639_639550


namespace reading_rate_l639_639132

-- Definitions based on conditions
def one_way_trip_time : ℕ := 4
def round_trip_time : ℕ := 2 * one_way_trip_time
def read_book_time : ℕ := 2 * round_trip_time
def book_pages : ℕ := 4000

-- The theorem to prove Juan's reading rate is 250 pages per hour.
theorem reading_rate : book_pages / read_book_time = 250 := by
  sorry

end reading_rate_l639_639132


namespace ellipse_equation_tangent_line_constant_value_l639_639352

-- Definitions for the Ellipse problem.
variables (a b : ℝ)

def ellipse_eq := (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def passes_through_M := ellipse_eq 1 (√2 / 2)
def isosceles_right_triangle := a = √2 * b
def tangent_line_of_circle (k m : ℝ) := 
  ∃ y x, y = k * x + m ∧ x^2 + y^2 = 2 / 3 ∧ ellipse_eq x y

-- Main theorems to prove.
theorem ellipse_equation : 
  a > b ∧ b > 0 ∧ passes_through_M ∧ isosceles_right_triangle → 
  ∃ y x, ellipse_eq x y := 
sorry

theorem tangent_line_constant_value :
  ∀ k m, tangent_line_of_circle k m → 
  ∃ P Q : ℝ × ℝ, 
  (P.x = Q.x ∧ ∀ O : ℝ × ℝ, (O.x * P.x + O.y * P.y + O.x * Q.x + O.y * Q.y) = 0) :=
sorry

end ellipse_equation_tangent_line_constant_value_l639_639352


namespace kaleb_total_score_l639_639468

def first_half_score := 43
def bonus_percentage := 0.20
def second_half_score := 23
def penalty_percentage := 0.10

theorem kaleb_total_score :
  let first_half_adjusted := first_half_score + first_half_score * bonus_percentage
  let second_half_adjusted := second_half_score - second_half_score * penalty_percentage
  let first_half_rounded := Int.ceil first_half_adjusted
  let second_half_rounded := Int.floor second_half_adjusted
  first_half_rounded + second_half_rounded = 73 := by
  sorry

end kaleb_total_score_l639_639468


namespace correct_expression_l639_639165

theorem correct_expression (a : ℝ) :
  (a^3 * a^2 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ ¬((a - 1)^2 = a^2 - 1) :=
by
  sorry

end correct_expression_l639_639165


namespace smallest_possible_value_l639_639518

open Complex

noncomputable def min_modulus_w (w : ℂ) (h : ∥w^2 - 3∥ = ∥w * (2 * w + 3 * I)∥) : ℝ :=
  min (abs ((sqrt 3 - sqrt 6) / 3)) (abs ((sqrt 3 + sqrt 6) / 3))

theorem smallest_possible_value (w : ℂ) (h : ∥w^2 - 3∥ = ∥w * (2 * w + 3 * I)∥) : 
  min_modulus_w w h = abs ((sqrt 3 - sqrt 6) / 3) := 
sorry

end smallest_possible_value_l639_639518


namespace p_necessary_not_sufficient_q_l639_639053

theorem p_necessary_not_sufficient_q (x : ℝ) : (|x| = 2) → (x = 2) → (|x| = 2 ∧ (x ≠ 2 ∨ x = -2)) := by
  intros h_p h_q
  sorry

end p_necessary_not_sufficient_q_l639_639053


namespace unique_zero_condition_l639_639890

def f (x m : ℝ) : ℝ := x^2 - m * (Real.cos x) + m^2 + 3 * m - 8

theorem unique_zero_condition {m : ℝ} : (∀ x : ℝ, f x m = 0 → x = 0) ↔ m = 2 :=
  sorry

end unique_zero_condition_l639_639890


namespace length_of_real_axis_l639_639119

noncomputable def parabola (y: ℝ) : ℝ := (y^2) / 16

noncomputable def hyperbola (x y : ℝ) (a: ℝ) : Bool := x^2 - y^2 = a^2

theorem length_of_real_axis
  (yA yB : ℝ)
  (a : ℝ)
  (h_parabola : parabola yA = -4 ∧ yA = 2 * Real.sqrt 3 ∧ yB = -(2 * Real.sqrt 3))
  (h_distance : abs (yA - yB) = 4 * Real.sqrt 3)
  (h_hyperbola : hyperbola (-4) (2 * Real.sqrt 3) a) :
  2 * a = 4 :=
by
  -- proof steps are required here
  sorry

end length_of_real_axis_l639_639119


namespace perimeter_of_triangle_l639_639320

def point (x y : ℝ) := (x, y)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def perimeter_triangle (a b c : ℝ × ℝ) : ℝ :=
  distance a b + distance b c + distance c a

theorem perimeter_of_triangle :
  let A := point 1 2
  let B := point 6 8
  let C := point 1 5
  perimeter_triangle A B C = Real.sqrt 61 + Real.sqrt 34 + 3 :=
by
  -- proof steps can be provided here
  sorry

end perimeter_of_triangle_l639_639320


namespace cubes_sum_l639_639999

theorem cubes_sum (a b c : ℝ) (h1 : a + b + c = 8) (h2 : a * b + a * c + b * c = 9) (h3 : a * b * c = -18) :
  a^3 + b^3 + c^3 = 242 :=
by
  sorry

end cubes_sum_l639_639999


namespace tan_alpha_eq_sqrt3_div_3_l639_639000

theorem tan_alpha_eq_sqrt3_div_3 
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : sin α ^ 2 + cos (2 * α) = 3 / 4) : 
  tan α = sqrt 3 / 3 :=
by
  sorry

end tan_alpha_eq_sqrt3_div_3_l639_639000


namespace sin_and_tan_combined_condition_l639_639829

noncomputable def theta : ℝ := sorry

axiom theta_in_fourth_quadrant : 0 < theta ∧ theta < 2 * π ∧ θ > 3 * π / 2 ∧ θ < 2 * π

axiom sin_theta_plus_pi_over_4 : Real.sin (theta + π / 4) = 3 / 5

theorem sin_and_tan_combined_condition :
  Real.sin theta = -√2 / 10 ∧ 
  Real.tan (theta - π / 4) = -4 / 3 :=
by
  sorry

end sin_and_tan_combined_condition_l639_639829


namespace marked_nodes_on_circle_l639_639699

noncomputable def regular_hexagon_nodes (s : ℕ) : ℕ := 3 * s * s + 3 * s + 1

theorem marked_nodes_on_circle
  (s : ℕ)
  (h : s = 5)
  (n : ℕ)
  (hnodes : n = regular_hexagon_nodes s)
  (h_marked : 2 * (hnodes/2) < hnodes)
  : ∃ (c : list ℕ) (hc : c.length >= 5), true :=
  sorry

end marked_nodes_on_circle_l639_639699


namespace find_q_l639_639006

theorem find_q (q : ℤ) (h1 : lcm (lcm 12 16) (lcm 18 q) = 144) : q = 1 := sorry

end find_q_l639_639006


namespace lines_perpendicular_if_one_perpendicular_and_one_parallel_l639_639370

def Line : Type := sorry  -- Define the type representing lines
def Plane : Type := sorry  -- Define the type representing planes

def is_perpendicular_to_plane (a : Line) (α : Plane) : Prop := sorry  -- Definition for a line being perpendicular to a plane
def is_parallel_to_plane (b : Line) (α : Plane) : Prop := sorry  -- Definition for a line being parallel to a plane
def is_perpendicular (a b : Line) : Prop := sorry  -- Definition for a line being perpendicular to another line

theorem lines_perpendicular_if_one_perpendicular_and_one_parallel 
  (a b : Line) (α : Plane) 
  (h1 : is_perpendicular_to_plane a α) 
  (h2 : is_parallel_to_plane b α) : 
  is_perpendicular a b := 
sorry

end lines_perpendicular_if_one_perpendicular_and_one_parallel_l639_639370


namespace find_m_n_l639_639836

-- Define the vectors OA, OB, OC
def vector_oa (m : ℝ) : ℝ × ℝ := (-2, m)
def vector_ob (n : ℝ) : ℝ × ℝ := (n, 1)
def vector_oc : ℝ × ℝ := (5, -1)

-- Define the condition that OA is perpendicular to OB
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Define the condition that points A, B, and C are collinear.
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (A.1 - B.1) * (C.2 - A.2) = k * ((C.1 - A.1) * (A.2 - B.2))

theorem find_m_n (m n : ℝ) :
  collinear (-2, m) (n, 1) (5, -1) ∧ perpendicular (-2, m) (n, 1) → m = 3 ∧ n = 3 / 2 := by
  intro h
  sorry

end find_m_n_l639_639836


namespace find_b_l639_639375

theorem find_b (k a b : ℝ) (h1 : 1 + a + b = 3) (h2 : k = 3 + a) :
  b = 3 := 
sorry

end find_b_l639_639375


namespace geometric_sequence_first_term_l639_639840

theorem geometric_sequence_first_term
  (a : ℕ → ℝ) -- sequence a_n
  (r : ℝ) -- common ratio
  (h1 : r = 2) -- given common ratio
  (h2 : a 4 = 16) -- given a_4 = 16
  (h3 : ∀ n, a n = a 1 * r^(n-1)) -- definition of geometric sequence
  : a 1 = 2 := 
sorry

end geometric_sequence_first_term_l639_639840


namespace power_inequality_l639_639365

theorem power_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  a^5 + b^5 > a^2 * b^3 + a^3 * b^2 :=
sorry

end power_inequality_l639_639365


namespace problem_solution_l639_639679

-- Definitions of events and conditions from the problem
def balls : Finset ℕ := {1, 2, 3}

def draws : ℕ → List ℕ
| 0 => []
| (n + 1) => List.append balls.to_list (draws n)  -- generates draw sequence

def event_A (draws: List ℕ) : Prop := draws.sum = 6

def event_B (draws: List ℕ) : Prop := draws = [2, 2, 2]

noncomputable def P (event: List ℕ → Prop) : ℚ :=
  (Finset.card ((Finset.univ : Finset (List ℕ)).filter event)) / ((finset.card balls) ^ 3)

noncomputable def P_cond (A B: List ℕ → Prop) : ℚ :=
  P (λ draws => A draws ∧ B draws) / P A

theorem problem_solution :
  P_cond event_A event_B = 1/7 :=
sorry

end problem_solution_l639_639679


namespace xy_value_l639_639019

theorem xy_value (x y : ℝ) (h1 : (x + y) / 3 = 1.222222222222222) : x + y = 3.666666666666666 :=
by
  sorry

end xy_value_l639_639019


namespace remainder_h_x_10_div_h_x_l639_639514

noncomputable def h (x : ℤ) : ℤ := x^5 - x^4 + x^3 - x^2 + x - 1

theorem remainder_h_x_10_div_h_x (x : ℤ) : polynomial.div_mod_by_monic (h (x)) (h (x)) (h (x^{10})) = (x, 5) :=
by
  -- Proof omitted.
  sorry

end remainder_h_x_10_div_h_x_l639_639514


namespace repeating_decimal_to_fraction_l639_639308

theorem repeating_decimal_to_fraction :
  let x : ℚ := 5312 / 999 in
  let y : ℚ := (Dec.fromReal 5.317m) in  -- this is an approximation since Lean doesn't have a primitive for repeating decimals
  y = x := sorry

end repeating_decimal_to_fraction_l639_639308


namespace solve_for_x_l639_639886

theorem solve_for_x (x : ℝ) (hx_pos : 0 < x) (hx : 18 / real.sqrt x = 2) : x = 81 := 
by
  sorry

end solve_for_x_l639_639886


namespace locus_of_M_l639_639077

theorem locus_of_M (k : ℝ) (A B M : ℝ × ℝ) (hA : A.1 ≥ 0 ∧ A.2 = 0) (hB : B.2 ≥ 0 ∧ B.1 = 0) (h_sum : A.1 + B.2 = k) :
    ∃ (M : ℝ × ℝ), (M.1 - k / 2)^2 + (M.2 - k / 2)^2 = k^2 / 2 :=
by
  sorry

end locus_of_M_l639_639077


namespace find_x_l639_639842

open Real

def vector (a b : ℝ) : ℝ × ℝ := (a, b)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

def problem_statement (x : ℝ) : Prop :=
  let m := vector 2 x
  let n := vector 4 (-2)
  let m_minus_n := vector (2 - 4) (x - (-2))
  perpendicular m m_minus_n → x = -1 + sqrt 5 ∨ x = -1 - sqrt 5

-- We assert the theorem based on the problem statement
theorem find_x (x : ℝ) : problem_statement x :=
  sorry

end find_x_l639_639842


namespace parallelogram_angle_equation_l639_639471

theorem parallelogram_angle_equation 
    (EFGH : Parallelogram) (P : Point)
    (hP : P = intersection (diagonal EFGH) (diagonal FHE))
    (phi : ℝ)
    (hFEG : ∠FEG = 3 * phi)
    (hHFG : ∠HFG = 3 * phi)
    (hSinLawFGP : sin (3 * phi) / sin (4 * phi) = dist P G / dist F G)
    (hSinLawEFG : sin (4 * phi) / sin (3 * phi) = dist E G / dist F G)
    (hLengths : dist E G = 2 * dist P G) :
    (∠EGH = t * ∠EHP) := 
begin
    sorry,
end

end parallelogram_angle_equation_l639_639471


namespace bob_constant_term_is_3_l639_639726

-- Definitions of conditions
def alice_poly (p : ℕ → ℝ) : Prop :=
  p 5 = 1 ∧ p 0 > 0 ∧ p 0 = p 1

def bob_poly (q : ℕ → ℝ) : Prop :=
  q 5 = 1 ∧ q 0 > 0 ∧ q 0 = q 1

def same_constant_term (p q : ℕ → ℝ) : Prop :=
  p 0 = q 0

def polynomials_product (p q : ℕ → ℝ) : (ℕ → ℝ) :=
  λ n, (list.range (n + 1)).sum (λ i, p i * q (n - i))

-- The given product polynomial
def product_poly (n : ℕ) : ℝ :=
  [
    9, 4, 4, 6, 5, 10, 5, 4, 6, 4, 1
  ].nth n.get_or_else 0

-- The problem statement to prove
theorem bob_constant_term_is_3 (p q : ℕ → ℝ)
  (h_alice : alice_poly p)
  (h_bob : bob_poly q)
  (h_same : same_constant_term p q)
  (h_prod : ∀ n, polynomials_product p q n = product_poly n) :
  q 0 = 3 :=
by
  sorry

end bob_constant_term_is_3_l639_639726


namespace smallest_n_satisfying_sum_l639_639972

def f (x : ℕ) : ℚ := 1 / (x * (x + 1) * (x + 2))

theorem smallest_n_satisfying_sum :
  ∃ n : ℕ, (∀ m : ℕ, m < n → ∑ i in Finset.range m, f (i + 1) ≤ 503 / 2014) ∧ ∑ i in Finset.range n, f (i + 1) > 503 / 2014 :=
sorry

end smallest_n_satisfying_sum_l639_639972


namespace termination_constant_exists_l639_639043

open Real

variables {n m : ℕ}
variables (vecs : Fin m → Fin n → ℝ)

-- Define the condition that each vector has a strictly positive first coordinate
def positive_first_coordinate (v : Fin n → ℝ) : Prop := v 0 > 0

-- Define the condition for choosing index i
def choose_index (w : Fin n → ℝ) (i : Fin m) : Prop := dot_product w (vecs i) ≤ 0

-- Main theorem statement
theorem termination_constant_exists
  (h₁ : ∀ i, positive_first_coordinate (vecs i))
  : ∃ C : ℝ, ∀ (w : Fin n → ℝ), w = 0 →
    ∃ R : ℕ, R ≤ C ∧ (∀ i, choose_index w i → w + vecs i ≠ w) → 
    ∀ (R' : ℕ), R' > R → (∀ w' : Fin n → ℝ, w' ≠ 0 → choose_index w' i → 
    (w' := w' + vecs i) → False) :=
sorry

end termination_constant_exists_l639_639043


namespace stephan_writing_time_l639_639097

theorem stephan_writing_time :
  let rearrangements := 7.factorial
  let rate := 12
  let time_in_minutes := rearrangements / rate
  let time_in_hours := time_in_minutes / 60
  stephan_name_length = 7 →
  stephan_rate_per_minute = 12 →
  time_in_hours = 7 := by
  sorry

end stephan_writing_time_l639_639097


namespace hexagon_angles_sum_l639_639918

theorem hexagon_angles_sum (mA mB mC : ℤ) (x y : ℤ)
  (hA : mA = 35) (hB : mB = 80) (hC : mC = 30)
  (hSum : (6 - 2) * 180 = 720)
  (hAdjacentA : 90 + 90 = 180)
  (hAdjacentC : 90 - mC = 60) :
  x + y = 95 := by
  sorry

end hexagon_angles_sum_l639_639918


namespace sqrt_of_product_of_powers_l639_639751

theorem sqrt_of_product_of_powers :
  (Real.sqrt (4^2 * 5^6) = 500) :=
by
  sorry

end sqrt_of_product_of_powers_l639_639751


namespace sum_x_coords_above_line_eq_zero_l639_639859

theorem sum_x_coords_above_line_eq_zero :
  (∀ (p : ℕ × ℕ), p ∈ [(3, 10), (6, 20), (12, 35), (18, 40), (20, 50)] → 
                    p.snd > 3 * p.fst + 5 → 0 = 0) :=
by
  intro p h_points h_above
  cases h_points with
  | inl h3 => cases h3 with
              | refl => exact zero_add 0
  | inr p_rest =>
    cases p_rest with
    | inl h6 => cases h6 with
                | refl => exact zero_add 0
    | inr p_rest2 =>
      cases p_rest2 with
      | inl h12 => cases h12 with
                   | refl => exact zero_add 0
      | inr p_rest3 =>
        cases p_rest3 with
        | inl h18 => cases h18 with
                     | refl => exact zero_add 0
        | inr p_rest4 =>
          cases p_rest4 with
          | inl h20 => cases h20 with
                       | refl => exact zero_add 0
          | inr h_nil =>
            contradiction
  sorry

end sum_x_coords_above_line_eq_zero_l639_639859


namespace turtle_ratio_l639_639712

theorem turtle_ratio (total_turtles swept_to_sea remaining_turtles : ℕ) 
  (h1 : total_turtles = 42) 
  (h2 : remaining_turtles = 28)
  (h3 : swept_to_sea = total_turtles - remaining_turtles) :
  swept_to_sea : total_turtles = 1 : 3 :=
by
  sorry

end turtle_ratio_l639_639712


namespace find_even_periodic_function_l639_639728

/-- Among the following functions, the even function with a period of π/2 is proven. --/
theorem find_even_periodic_function :
  let fA := λ x : ℝ, sin (4 * x),
      fB := λ x : ℝ, cos (2 * x) ^ 2 - sin (2 * x) ^ 2,
      fC := λ x : ℝ, tan (2 * x),
      fD := λ x : ℝ, cos (2 * x) in
  (even_function fB ∧ periodic fB (π / 2)) :=
begin
  let fA := λ x : ℝ, sin (4 * x),
  let fB := λ x : ℝ, cos (2 * x) ^ 2 - sin (2 * x) ^ 2,
  let fC := λ x : ℝ, tan (2 * x),
  let fD := λ x : ℝ, cos (2 * x),
  sorry
end

end find_even_periodic_function_l639_639728


namespace external_angle_theorem_l639_639020

theorem external_angle_theorem
  (u v w x : ℝ) -- angles as real numbers
  (triangle_ABC : Type) -- type for triangle
  (P : Type) -- type for point outside the triangle
  (AP BP CP : triangle_ABC → P → Prop) -- lines forming angles at A, B, C with P
  (external_angles : ∀ (A B C : triangle_ABC), AP A P → BP B P → CP C P → Prop)
  (internal_opposite_angle : ∀ (A B C : triangle_ABC) (P : P), AP A P → ¬ (CP C P) → Prop)
  : x = u + v := 
begin
  sorry -- proof to be provided
end

end external_angle_theorem_l639_639020


namespace circle_tangent_ratio_l639_639060

theorem circle_tangent_ratio
  (P Q A B C D : ℝ)
  (l ω Ω : set ℝ)
  (PQ_eq_12 : dist P Q = 12)
  (ω_tangent_P : ω ∋ P)
  (Ω_tangent_P : Ω ∋ P)
  (ω_Ω_tangent : disjoint ω Ω)
  (line_through_Q_intersects_ω : ∃ A B, l ∋ A ∧ l ∋ B ∧ dist A B = 10)
  (line_through_Q_intersects_Ω : ∃ C D, l ∋ C ∧ l ∋ D ∧ dist C D = 7)
  (exists_ratio : ∃ A B C D, P A Q B C D : ℝ) :
  dist (A - D) / dist (B - C) = 8 / 9 :=
sorry

end circle_tangent_ratio_l639_639060


namespace cost_of_5_spoons_l639_639705

theorem cost_of_5_spoons (cost_per_set : ℕ) (num_spoons_per_set : ℕ) (num_spoons_needed : ℕ)
  (h1 : cost_per_set = 21) (h2 : num_spoons_per_set = 7) (h3 : num_spoons_needed = 5) :
  (cost_per_set / num_spoons_per_set) * num_spoons_needed = 15 :=
by
  sorry

end cost_of_5_spoons_l639_639705


namespace debby_total_tickets_l639_639758

theorem debby_total_tickets : 
  let first_trip := 2 + 10 + 2,
      second_trip := 3 + 7 + 5,
      third_trip := 8 + 15 + 4,
      total_tickets := first_trip + second_trip + third_trip
  in total_tickets = 56 :=
by
  let first_trip := 2 + 10 + 2;
  let second_trip := 3 + 7 + 5;
  let third_trip := 8 + 15 + 4;
  let total_tickets := first_trip + second_trip + third_trip;
  have h1 : first_trip = 14 := by rfl;
  have h2 : second_trip = 15 := by rfl;
  have h3 : third_trip = 27 := by rfl;
  have h4 : total_tickets = 14 + 15 + 27 := by rfl;
  exact h4.symm.trans (by norm_num);
  sorry

end debby_total_tickets_l639_639758


namespace good_triangles_upper_bound_l639_639345

section convex_ngon

variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α] -- This will handle the geometry

/-- A polygon is convex if every line segment between two vertices lies inside the polygon -/
def is_convex_polygon (P : set α) (n : ℕ) : Prop :=
  ∃ (V : fin n → α), (∀ i j, dist (V i) (V j) ≤ dist (V i) (V j)) ∧ P = {p | ∃ (u : fin n → ℝ), (∀ i, 0 ≤ u i) ∧ (finset.univ.sum u = 1) ∧ p = finset.univ.sum (λ i, u i • (V i))}

/-- A triangle is good if all its sides are unit length -/
def is_good_triangle (Δ : fin 3 → α) : Prop :=
  ∀ i j, i ≠ j → dist (Δ i) (Δ j) = 1

/-- The function that defines the number of good triangles in a convex n-gon -/
def good_triangles_count (P : set α) [fintype P] : ℕ :=
  fintype.card {Δ : fin 3 → α // is_good_triangle Δ ∧ (∀ i, Δ i ∈ P)}

theorem good_triangles_upper_bound (P : set α) (n : ℕ) 
  (h_convex : is_convex_polygon P n) 
  [fintype P] : 
  good_triangles_count P ≤ 2 * n / 3 :=
begin
  sorry
end

end convex_ngon

end good_triangles_upper_bound_l639_639345


namespace routes_from_A_to_B_l639_639339

theorem routes_from_A_to_B (roads_AC roads_CB : ℕ) (h1 : roads_AC = 4) (h2 : roads_CB = 2) : 
  roads_AC * roads_CB = 8 :=
by 
  simp [h1, h2]; sorry

end routes_from_A_to_B_l639_639339


namespace line_passes_through_fixed_point_min_AB_segment_length_radius_range_min_OA_OB_dot_product_l639_639391

-- Defining the line by its equation and a point (2, 1)
def line_eq (m x y : ℝ) : Prop := m * x + y - 1 - 2 * m = 0

-- Defining the equation of the circle
def circle_eq (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Given the conditions
variables (m x y r : ℝ)
axiom distinct_points (h : x ≠ y)

-- Proving the necessary results
theorem line_passes_through_fixed_point : line_eq m 2 1 := sorry

theorem min_AB_segment_length (h : r = 4) : ∃ AB_min : ℝ, AB_min = 2 * sqrt (11) := sorry

theorem radius_range : ¬ ∃ r, 0 < r ∧ r ≤ sqrt 5 ∧ circle_eq x y r ∧ line_eq m x y := sorry

theorem min_OA_OB_dot_product (h : r = 4) : ∃ dot_min : ℝ, dot_min = -16 := sorry

end line_passes_through_fixed_point_min_AB_segment_length_radius_range_min_OA_OB_dot_product_l639_639391


namespace probability_multiple_of_3_l639_639770

-- Definitions to represent the conditions in Lean 4
def boxes : list ℕ := [1, 2, 3]
def total_outcomes : ℕ := (boxes.length) * (boxes.length)
def favorable_outcomes : ℕ := ((boxes.filter (λ x => x == 3)).length) * (boxes.length) + 
                              ((boxes.length) - 1) -- accounting for duplicated (3, 3)

-- The Lean 4 statement for the proof problem
theorem probability_multiple_of_3 : 
  (favorable_outcomes.toRat / total_outcomes.toRat) = (5 / 9) := 
by
  sorry

end probability_multiple_of_3_l639_639770


namespace count_integers_with_increasing_digits_l639_639418

theorem count_integers_with_increasing_digits :
  let count_integers := 
    ∑ second_digit in ({3, 4, 5} : Finset ℕ), 
      ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ second_digit}, 
        1 in

  count_integers = 15 :=
by
  have step1 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 3}, 1 = 6,
  { -- Explanation: If second digit is 3, third can be 4, 5, 6, 7, 8, 9 -> 6 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step2 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 4}, 1 = 5,
  { -- Explanation: If second digit is 4, third can be 5, 6, 7, 8, 9 -> 5 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step3 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 5}, 1 = 4,
  { -- Explanation: If second digit is 5, third can be 6, 7, 8, 9 -> 4 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have count_integers := step1 + step2 + step3,
  simp only [count_integers, add_comm],
  exact eq.refl 15

end count_integers_with_increasing_digits_l639_639418


namespace min_value_quadratic_l639_639878

theorem min_value_quadratic (a : ℝ) : ∃ m, (∀ x : ℝ, x^2 - 4 * x + 9 ≥ m) ∧ m = 5 :=
begin
  sorry
end

end min_value_quadratic_l639_639878


namespace count_integers_between_200_and_250_with_increasing_digits_l639_639433

theorem count_integers_between_200_and_250_with_increasing_digits :
  ∃ n, n = 11 ∧ ∀ x, 200 ≤ x ∧ x ≤ 250 ∧ 
  (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k) → 
  n = ∑ x in {x | 200 ≤ x ∧ x ≤ 250 ∧ (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k)}, 1
:= 
sorry

end count_integers_between_200_and_250_with_increasing_digits_l639_639433


namespace num_int_values_N_l639_639329

theorem num_int_values_N (N : ℕ) : 
  (∃ M, M ∣ 72 ∧ M > 3 ∧ N = M - 3) ↔ N ∈ ({1, 3, 5, 6, 9, 15, 21, 33, 69} : Finset ℕ) :=
by
  sorry

end num_int_values_N_l639_639329


namespace decreasing_interval_of_f_l639_639602

def f (x : ℝ) : ℝ := x * Real.log x
def f' (x : ℝ) : ℝ := 1 + Real.log x

theorem decreasing_interval_of_f :
  { x : ℝ | 0 < x ∧ f'(x) ≤ 0 } = { x : ℝ | 0 < x ∧ x ≤ 1 / Real.exp 1 } :=
by
  sorry

end decreasing_interval_of_f_l639_639602


namespace coeff_x3_f_mul_g_l639_639152

def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3 + 2 * x^2 - 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 5

theorem coeff_x3_f_mul_g : polynomial.coeff (polynomial.of_fn (λ n, if n = 3 then polynomial.eval (n) (f(n) * g(n)) else 0)) 3 = -32 :=
by
  sorry

end coeff_x3_f_mul_g_l639_639152


namespace count_increasing_order_digits_l639_639413

def count_valid_numbers : ℕ :=
  let numbers := [234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249] in
  numbers.length

theorem count_increasing_order_digits : count_valid_numbers = 11 :=
by
  sorry

end count_increasing_order_digits_l639_639413


namespace isosceles_triangle_perimeter_l639_639014

theorem isosceles_triangle_perimeter 
    (a b : ℕ) (h_iso : a = 3 ∨ a = 5) (h_other : b = 3 ∨ b = 5) 
    (h_distinct : a ≠ b) : 
    ∃ p : ℕ, p = (3 + 3 + 5) ∨ p = (5 + 5 + 3) :=
by
  sorry

end isosceles_triangle_perimeter_l639_639014


namespace least_possible_value_of_N_l639_639498

theorem least_possible_value_of_N :
  ∀ (N : ℕ), (∀ x : ℝ, x > 0 → x < 1 →
    (∃ (color : ℕ → ℕ → ℕ), (∀ n : ℕ, ∃ i : ℕ, color n i = n ∧ 
      (∀ m : ℕ, m < N → 
        (∃ r : ℚ, ∀ k : ℕ, 
          (color x.floor (k + m) = m) ↔ r.floor = x.floor (k + m)))))
    → N = 10 :=
sorry

end least_possible_value_of_N_l639_639498


namespace min_k_value_l639_639177

theorem min_k_value (a : Fin 21 → ℕ) (k : ℕ) :
  (∑ i in Finset.range 21, a i) = 2012 →
  (∑ i in Finset.range 21, i * a i) = k →
  (∃ m in Finset.range 1 (20 + 1), (∑ i in Finset.range m 21, a i) ≥ Nat.ceil (100 / m)) →
  k ≥ 349 :=
by
  intros h₁ h₂ h₃
  sorry

end min_k_value_l639_639177


namespace area_triangle_ABC_l639_639923

theorem area_triangle_ABC (AB BC BD AC: ℝ) :
  AB = BC ∧ BD = sqrt 2 * 6 ∧ BE = 12 ∧ ∠ DBE = 45 ∧
  ∃ (α β : ℝ) , α = 45 ∧ β = 0 ∧ 
  tan (α - β) * tan (α + β) = 1 ∧ 
  cot β = ∞ ∧ cot (α - β) = 1 ∧ cot (α + β) = 1 → 
  area_of_triangle ABC = 12 :=
by
  sorry

end area_triangle_ABC_l639_639923


namespace vector_magnitude_is_five_l639_639869

-- Define the given vectors and conditions
variables {x y : ℝ}

def a : ℝ × ℝ := (x, y)
def b : ℝ × ℝ := (-1, 2)
def add_vectors (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def a_plus_b := add_vectors a b
def a_minus_2b := add_vectors a (scalar_mult (-2) b)

-- State the theorem
theorem vector_magnitude_is_five (h1 : a_plus_b = (1, 3)) : magnitude a_minus_2b = 5 :=
by
  sorry

end vector_magnitude_is_five_l639_639869


namespace number_of_negative_x_l639_639795

theorem number_of_negative_x (n : ℤ) (hn : 1 ≤ n ∧ n * n < 200) : 
  ∃ m ≥ 1, m = 14 := sorry

end number_of_negative_x_l639_639795


namespace tangent_circle_angles_equal_l639_639505

open EuclideanGeometry

/-- Let Γ and Γ' be two circles tangent at point A, with Γ' located inside Γ. 
    Let D be a point on Γ' other than A. Denote M and N as the points of
    intersection of the tangent to Γ' at D with Γ.
    Then, show that the angle ∠NAD equals the angle ∠MAD. -/
theorem tangent_circle_angles_equal
  (Γ Γ' : Circle)
  (A : Point)
  (tangent_condition : Tangent Γ Γ' A)
  (inside_condition : Inside Γ' Γ)
  (D : Point)
  (D_on_Γ' : OnCircle D Γ')
  (D_ne_A : D ≠ A)
  (M N : Point)
  (M_N_on_Γ : OnCircle M Γ ∧ OnCircle N Γ)
  (M_N_tangent_intersections : TangentAtLineIntersection Γ' D M N) :
  angle N A D = angle M A D :=
begin
  sorry
end

end tangent_circle_angles_equal_l639_639505


namespace distance_between_trees_l639_639461

theorem distance_between_trees
  (n : ℕ) (L : ℕ) (x : ℕ)
  (h₁ : n = 11)
  (h₂ : L = 151)
  (h₃ : L = n + 10 * x) :
  x = 14 :=
by {
  subst h₁,
  subst h₂,
  rw add_comm at h₃,
  linarith,
}

end distance_between_trees_l639_639461


namespace pages_read_per_hour_l639_639128

theorem pages_read_per_hour (lunch_time : ℕ) (book_pages : ℕ) (round_trip_time : ℕ)
  (h1 : lunch_time = round_trip_time)
  (h2 : book_pages = 4000)
  (h3 : round_trip_time = 2 * 4) :
  book_pages / (2 * lunch_time) = 250 :=
by
  sorry

end pages_read_per_hour_l639_639128


namespace chess_club_officer_ways_l639_639704

theorem chess_club_officer_ways :
  let members := 24
  let officers := 3
  let conditions (P Q R : Prop) : Prop := 
    P ∧ Q ∨ P ∧ R ∨ Q ∧ R

  ∃ (ways : ℕ),
    ways = (3.choose 2 * 6 * 21) + 6 :=
by
  sorry

end chess_club_officer_ways_l639_639704


namespace range_of_t_l639_639451

theorem range_of_t (t : ℝ) (x : ℝ) : (1 < x ∧ x ≤ 4) → (|x - t| < 1 ↔ 2 ≤ t ∧ t ≤ 3) :=
by
  sorry

end range_of_t_l639_639451


namespace author_of_sea_island_arithmetic_l639_639912

theorem author_of_sea_island_arithmetic :
  (exists (x : Type) (h1 : "The Sea Island Arithmetic" brought Chinese surveying to the peak), 
   h2 : "The Sea Island Arithmetic" preceded Europe by 1300 to 1500 years) -> 
  x = LiuHui :=
by
  sorry

end author_of_sea_island_arithmetic_l639_639912


namespace tan_identity_given_condition_l639_639363

variable (α : Real)

theorem tan_identity_given_condition :
  (Real.tan α + 1 / Real.tan α = 9 / 4) →
  (Real.tan α ^ 2 + 1 / (Real.sin α * Real.cos α) + 1 / Real.tan α ^ 2 = 85 / 16) := 
by
  sorry

end tan_identity_given_condition_l639_639363


namespace pages_read_per_hour_l639_639130

theorem pages_read_per_hour (lunch_time : ℕ) (book_pages : ℕ) (round_trip_time : ℕ)
  (h1 : lunch_time = round_trip_time)
  (h2 : book_pages = 4000)
  (h3 : round_trip_time = 2 * 4) :
  book_pages / (2 * lunch_time) = 250 :=
by
  sorry

end pages_read_per_hour_l639_639130


namespace correct_transformations_count_l639_639730

open Classical

theorem correct_transformations_count : 
  let t1 := (3 + x = 5) -> (x = 5 + 3)
  let t2 := (7 * x = -4) -> (x = -4 / 7)
  let t3 := ((1/2) * y = 0) -> (y = 2)
  let t4 := (3 = x - 2) -> (x = -2 - 3)
  in count (λ t, t) [t1, t2, t3, t4] = 1 := 
by 
  sorry

end correct_transformations_count_l639_639730


namespace correct_times_between_7_and_8_l639_639260

theorem correct_times_between_7_and_8 (x : ℕ) :
  let hour_angle := 210 + (x / 2) in
  let minute_angle := 6 * x in
  abs (hour_angle - minute_angle) = 84 ↔ x = 23 ∨ x = 53 :=
by {
  sorry
}

end correct_times_between_7_and_8_l639_639260


namespace check_right_triangle_sets_l639_639667

theorem check_right_triangle_sets :
  ¬ (∃ (a b c : ℕ), a = 2 ∧ b = 3 ∧ c = 4 ∧ a^2 + b^2 = c^2) :=
by {
  intro h,
  rcases h with ⟨a, b, c, ha, hb, hc, hpythagorean⟩,
  rw [ha, hb, hc] at hpythagorean,
  norm_num at hpythagorean,
}

end check_right_triangle_sets_l639_639667


namespace james_vegetable_intake_l639_639932

theorem james_vegetable_intake :
  let daily_asparagus := 0.25
  let daily_broccoli := 0.25
  let daily_intake := daily_asparagus + daily_broccoli
  let doubled_daily_intake := daily_intake * 2
  let weekly_intake_asparagus_broccoli := doubled_daily_intake * 7
  let weekly_kale := 3
  let total_weekly_intake := weekly_intake_asparagus_broccoli + weekly_kale
  total_weekly_intake = 10 := 
by
  sorry

end james_vegetable_intake_l639_639932


namespace geometric_sequence_thm_l639_639458

theorem geometric_sequence_thm :
  (∀ (n : ℕ), a n = 6 * 2^(n-1)) ∧ (∑ i in finset.range n, a (i+1) = 6 * (2^n - 1) / (2 - 1)) :=
by
  let a := 3 * 2^n
  let S := 3 * 2^(n+1) - 6
  sorry

end geometric_sequence_thm_l639_639458


namespace find_n_l639_639369

theorem find_n (n : ℕ) (h : 0 < 37.5 ^ n + 26.5 ^ n) : n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 :=
sorry

end find_n_l639_639369


namespace total_cost_of_job_l639_639189

theorem total_cost_of_job (hourly_rate materials_cost hours_needed : ℕ) (h1 : hourly_rate = 28) (h2 : materials_cost = 560) (h3 : hours_needed = 15) : 
    (hourly_rate * hours_needed + materials_cost = 980) := 
by
  simp [h1, h2, h3]
  sorry

end total_cost_of_job_l639_639189


namespace book_cost_l639_639003

-- Definitions from conditions
def priceA : ℝ := 340
def priceB : ℝ := 350
def gain_percent_more : ℝ := 0.05

-- proof problem
theorem book_cost (C : ℝ) (G : ℝ) :
  (priceA - C = G) →
  (priceB - C = (1 + gain_percent_more) * G) →
  C = 140 :=
by
  intros
  sorry

end book_cost_l639_639003


namespace december_sales_fraction_l639_639941

noncomputable def average_sales (A : ℝ) := 11 * A
noncomputable def december_sales (A : ℝ) := 3 * A
noncomputable def total_sales (A : ℝ) := average_sales A + december_sales A

theorem december_sales_fraction (A : ℝ) (h1 : december_sales A = 3 * A)
  (h2 : average_sales A = 11 * A) :
  december_sales A / total_sales A = 3 / 14 :=
by
  sorry

end december_sales_fraction_l639_639941


namespace lucy_breaks_bill_probability_l639_639587

-- Definitions for the problem setup
def toy_prices : List ℝ := [0.50, 1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00]
def lucy_initial_coins : ℝ := 2.50
def lucy_favorite_toy_price : ℝ := 3.50
def total_toys : ℕ := 10
def toy_combinations : ℕ := 10!

-- Preliminary calculation fields for valid outcomes (Favorable to set proof conventionally)
def favorable_outcomes (total: ℕ) (toy_price: ℝ) : ℕ :=
  if toy_price == 3.50 then
     1 * (9! + (9 ᾰ 1) * 8! + (9 ᾰ 2) * 7! + (9 ᾰ 3) * 6! + (9 ᾰ 4) * 5! + (9 ᾰ 5) * 4!)
  else
     0

-- Function that returns probability that Lucy uses her $20 bill
noncomputable def probability_use_20_bill : ℝ :=
  1 - ((favorable_outcomes total_toys lucy_favorite_toy_price) / toy_combinations)

-- The statement asserting the proof we aim to check
theorem lucy_breaks_bill_probability :
  ∃ p : ℝ, p = probability_use_20_bill := 
  sorry

end lucy_breaks_bill_probability_l639_639587


namespace A_50_correct_l639_639942

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := 
  ![![3, 2], 
    ![-8, -5]]

-- The theorem to prove
theorem A_50_correct : A^50 = ![![(-199 : ℤ), -100], 
                                 ![400, 201]] := 
by
  sorry

end A_50_correct_l639_639942


namespace karen_start_late_l639_639039

noncomputable def karen_speed := 60 -- mph
noncomputable def tom_speed := 45 -- mph
noncomputable def tom_distance := 24 -- miles
noncomputable def karen_lead := 4 -- miles
noncomputable def hour_to_minutes := 60 -- conversion factor

def tom_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed -- Time = Distance / Speed

def karen_distance (time : ℝ) (speed : ℝ) : ℝ :=
  speed * time -- Distance = Speed * Time

theorem karen_start_late : 
  (tom_time tom_distance tom_speed) * hour_to_minutes - 
  (tom_time (tom_distance + karen_lead) karen_speed) * hour_to_minutes = 4 :=
by 
  sorry

end karen_start_late_l639_639039


namespace smallest_n_obtuse_l639_639049

-- Definitions based on given conditions
def initial_triangle_angles : ℝ × ℝ × ℝ := (59, 61, 60)

def altitude_angle (angle : ℝ) : ℝ := 90 - angle

def midpoint_angle (angle1 angle2 : ℝ) : ℝ := (angle1 + angle2) / 2

-- Main theorem statement
theorem smallest_n_obtuse :
  ∃ n : ℕ, 
    (n > 0 ∧ 
      let rec transform_angles (angles : ℝ × ℝ × ℝ) (k : ℕ) : ℝ × ℝ × ℝ :=
        if k = 0 then angles
        else let (α, β, γ) := transform_angles angles (k - 1) in
             (midpoint_angle (α + β) 90, midpoint_angle (β + γ) 90, midpoint_angle (γ + α) 90)
      in
      let (α', β', γ') := transform_angles initial_triangle_angles n in
      α' > 90 ∨ β' > 90 ∨ γ' > 90
    ) ∧ n = 11 :=
begin
  -- Proof would go here
  sorry
end

end smallest_n_obtuse_l639_639049


namespace trig_identities_of_angle_l639_639835

-- Define the conditions
variables (m : ℝ) (α : ℝ)
hypothesis h_m : m < 0
hypothesis h_P : (3 * m, -2 * m) -- point P

-- State the propositions to prove
theorem trig_identities_of_angle :
  sin α = (2 * Real.sqrt 13) / 13 ∧
  cos α = - (3 * Real.sqrt 13) / 13 ∧
  tan α = -2 / 3 :=
sorry

end trig_identities_of_angle_l639_639835


namespace rectangle_area_l639_639620

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l639_639620


namespace f_x_minus_2_odd_l639_639808

variables (f : ℝ → ℝ)

-- Conditions
def condition1 : Prop := ∀ x, f(x - 1) + f(1 - x) = 0
def condition2 : Prop := ∀ x, f(x + 1) = f(-x + 1)
def condition3 : Prop := f (-3 / 2) = 1

-- Conclusion to be proved based on conditions
theorem f_x_minus_2_odd :
  (condition1 f) →
  (condition2 f) →
  (condition3 f) →
  ∀ x, f(x - 2) = -f(-x - 2) :=
by
  intros h1 h2 h3
  sorry

end f_x_minus_2_odd_l639_639808


namespace find_k_l639_639233

noncomputable theory
open_locale classical

variables (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)

-- Condition 1: ABC is an acute isosceles triangle inscribed in a circle
-- Condition 2: Tangents from B and C meet at D
-- Condition 3: angle ABC = angle ACB = 3 * angle D
    
def tangents_intersect_at_D (ABC : triangle) (B C D : point ABC) : Prop :=
  tangent_to_circle_at B D ∧ tangent_to_circle_at C D ∧ B ≠ C

def isosceles_triangle_condition (angle : point ABC → point ABC → point ABC → real) (B C : point ABC) : Prop :=
  angle ABC B C = angle ABC C B

def angle_triple_D (angle : point ABC → point ABC → point ABC → real) (B C D : point ABC) : Prop :=
  ∃ k, angle ABC B C = 3 * angle ABC D C ∧ angle ABC A B = k * π

theorem find_k (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)
  (h1 : tangents_intersect_at_D ABC B C D)
  (h2 : isosceles_triangle_condition angle B C)
  (h3 : angle_triple_D angle B C D) :
  ∃ k : real, angle ABC A B = (5 / 11) * π ∧ k = 5 / 11 :=
by sorry

end find_k_l639_639233


namespace relationship_abc_l639_639881

theorem relationship_abc (a b c : ℝ) (ha : a = Real.log 2) (hb : b = 5^(-1/2)) (hc : c = ∫ x in 0..1, x) :
  b < c ∧ c < a :=
by {
  rw [ha, hb, hc],
  simp, 
  sorry
}

end relationship_abc_l639_639881


namespace exists_n_element_set_l639_639561

open Set

theorem exists_n_element_set (n : ℕ) (hn : n ≥ 2) :
  ∃ S : Set ℤ, S.card = n ∧ ∀ a b ∈ S, a ≠ b → (a - b)^2 ∣ a * b :=
sorry

end exists_n_element_set_l639_639561


namespace medians_intersect_cyclic_implies_equal_sides_l639_639112

-- Definitions of the geometrical constructs and conditions
noncomputable theory

variables {A B C A1 C1 M : Type} 
variables [Medians A B C A1 C1 M] -- A, B, C, A1, and C1 are vertices, and M is the centroid.

-- Conditions
def medians_intersect_at_M (A B C A1 C1 M : Type) [Medians A B C A1 C1 M] : Prop :=
  true

def quadrilateral_cyclic (A1 B C1 M : Type) [Quadrilateral A1 B C1 M] : Prop :=
  true

-- Theorem to prove
theorem medians_intersect_cyclic_implies_equal_sides 
  (A B C A1 C1 M : Type)
  [Medians A B C A1 C1 M]
  [Quadrilateral A1 B C1 M] 
  (h_medians : medians_intersect_at_M A B C A1 C1 M)
  (h_cyclic : quadrilateral_cyclic A1 B C1 M) :
  AB = BC :=
  sorry

end medians_intersect_cyclic_implies_equal_sides_l639_639112


namespace acute_isosceles_triangle_k_l639_639247

theorem acute_isosceles_triangle_k (ABC : Triangle) (circ : Circle)
  (D : Point)
  (h1 : ABC.angles.B = ABC.angles.C) -- Isosceles property
  (h2 : ∀ P ∈ circ, is_tangent B P circ) -- Tangent property through B
  (h3 : ∀ Q ∈ circ, is_tangent C Q circ) -- Tangent property through C
  (h4 : angle ABC.angles.B = 3 * angle D )
  (h5 : ∃ k, angle ABC.angles.A = k * π ) :
  ∃ k, k = 5 / 11 :=
by
  sorry

end acute_isosceles_triangle_k_l639_639247


namespace construct_triangle_l639_639798

variables {P : Type*} [normed_add_comm_group P] [normed_space ℝ P]
variables (a b c : ray ℝ P) (X Y Z : P)

def lies_in_angular_region (r1 r2 : ray ℝ P) (q : P) : Prop :=
  ∃ d1 d2 : P, 
    d1 ∈ r1 ∧ d2 ∈ r2 ∧ 
    q = d1 + d2

-- Define rays a, b, c emanating from P
-- Define X in the angular region b ∩ c
-- Define Y in the angular region c ∩ a
-- Define Z in the angular region a ∩ b

axiom X_in_bc : lies_in_angular_region b c X
axiom Y_in_ca : lies_in_angular_region c a Y
axiom Z_in_ab : lies_in_angular_region a b Z

-- Proof problem statement
theorem construct_triangle :
  ∃ A B C : P, 
    (A ∈ a) ∧ (B ∈ b) ∧ (C ∈ c) ∧ 
    line_passes_through A B X ∧ 
    line_passes_through B C Y ∧ 
    line_passes_through C A Z :=
sorry

end construct_triangle_l639_639798


namespace center_of_tangent_circle_lies_on_hyperbola_l639_639591

open Real

def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 6*y + 24 = 0

noncomputable def locus_of_center : Set (ℝ × ℝ) :=
  {P | ∃ (r : ℝ), ∀ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x2 y2 → 
    dist P (x1, y1) = r + 2 ∧ dist P (x2, y2) = r + 1}

theorem center_of_tangent_circle_lies_on_hyperbola :
  ∀ P : ℝ × ℝ, P ∈ locus_of_center → ∃ (a b : ℝ) (F1 F2 : ℝ × ℝ), ∀ Q : ℝ × ℝ,
    dist Q F1 - dist Q F2 = 1 ∧ 
    dist F1 F2 = 5 ∧
    P ∈ {Q | dist Q F1 - dist Q F2 = 1} :=
sorry

end center_of_tangent_circle_lies_on_hyperbola_l639_639591


namespace fountain_area_l639_639140

theorem fountain_area (AB DC: ℝ) (D_midpoint: Prop):
  AB = 20 → DC = 12 → 
  (D_midpoint ∧ (∀ D, midpoint D A B)) →
  (∀ (R: ℝ), A = 244 * π) :=
by sorry

end fountain_area_l639_639140


namespace count_integers_in_range_with_increasing_digits_l639_639411

theorem count_integers_in_range_with_increasing_digits : 
  let integers_in_range := { n | 200 ≤ n ∧ n < 250 ∧ ((n % 10), (n / 10 % 10), (n / 100)) = (d₀, d₁, d₂) ∧ d₀ < d₁ ∧ d₁ < d₂ } in
  integers_in_range.card = 34 :=
sorry

end count_integers_in_range_with_increasing_digits_l639_639411


namespace factorization1_factorization2_factorization3_l639_639310

-- Problem 1
theorem factorization1 (a x : ℝ) : a * x^2 - 4 * a = a * (x + 2) * (x - 2) :=
sorry

-- Problem 2
theorem factorization2 (m x y : ℝ) : m * x^2 + 2 * m * x * y + m * y^2 = m * (x + y)^2 :=
sorry

-- Problem 3
theorem factorization3 (a b : ℝ) : (1 / 2) * a^2 - a * b + (1 / 2) * b^2 = (1 / 2) * (a - b)^2 :=
sorry

end factorization1_factorization2_factorization3_l639_639310


namespace find_missing_digits_l639_639480

theorem find_missing_digits :
  let x := 1251
  let y := 371
  -- Digits have been replaced by asterisks in the multiplication problem
  -- Find the multiplicands x and y such that their product holds true
  -- Given: x * y = 1251 * 371 = 1282121
  (∃ x y : ℕ, 
     x = 1251 ∧ 
     y = 371 ∧ 
     x * y = 1282121) :=
by
  use 1251,
  use 371,
  sorry

end find_missing_digits_l639_639480


namespace bottle_height_difference_l639_639557

/-- Mathematical definitions and conditions --/
def normal_circumference_bottle : ℝ := 27.5
def waist_circumference_bottle : ℝ := 21.6
def waist_height : ℝ := 1
def cone_heights : ℝ := 2
def bottle_height_diff : ℝ := 1.18

/-- Prove that the height difference between the narrowed bottle and the regular bottle is 1.18 cm --/
theorem bottle_height_difference : 
    ∀ (R r V_1 V_2 m : ℝ),
    2 * π * R = normal_circumference_bottle →
    2 * π * r = waist_circumference_bottle →
    V_1 = (4 / 3) * π * (R^2 + R * r + r^2) + r^2 * π →
    V_2 = R^2 * π * m →
    V_1 = V_2 →
    regular_bottle_height : ℝ := 2 * cone_heights + waist_height + regular_full_bottle_section →
    m = (4 / 3) + (4 / 3) * (r / R) + (7 / 3) * (r / R)^2 →
    height_difference := regular_bottle_height - m →
    height_difference = bottle_height_diff := 
    begin
      sorry -- Proof goes here
    end

end bottle_height_difference_l639_639557


namespace continuity_at_2_l639_639063

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 4 * x^2 + 5 else b * x + 3

theorem continuity_at_2 (b : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) → b = 9 :=
by
  sorry  

end continuity_at_2_l639_639063


namespace g_one_third_value_l639_639998

noncomputable def g : ℚ → ℚ := sorry

theorem g_one_third_value : (∀ (x : ℚ), x ≠ 0 → (4 * g (1 / x) + 3 * g x / x^2 = x^3)) → g (1 / 3) = 21 / 44 := by
  intro h
  sorry

end g_one_third_value_l639_639998


namespace task1_task2_task3_task4_l639_639706

-- Definitions of the given conditions
def cost_price : ℝ := 16
def selling_price_range (x : ℝ) : Prop := 16 ≤ x ∧ x ≤ 48
def init_selling_price : ℝ := 20
def init_sales_volume : ℝ := 360
def decreasing_sales_rate : ℝ := 10
def daily_sales_vol (x : ℝ) : ℝ := 360 - 10 * (x - 20)
def daily_total_profit (x : ℝ) (y : ℝ) : ℝ := y * (x - cost_price)

-- Proof task (1)
theorem task1 : daily_sales_vol 25 = 310 ∧ daily_total_profit 25 (daily_sales_vol 25) = 2790 := 
by 
    -- Your code here
    sorry

-- Proof task (2)
theorem task2 : ∀ x, daily_sales_vol x = -10 * x + 560 := 
by 
    -- Your code here
    sorry

-- Proof task (3)
theorem task3 : ∀ x, 
    W = (x - 16) * (daily_sales_vol x) 
    ∧ W = -10 * x ^ 2 + 720 * x - 8960 
    ∧ (∃ x, -10 * x ^ 2 + 720 * x - 8960 = 4000 ∧ selling_price_range x) := 
by 
    -- Your code here 
    sorry

-- Proof task (4)
theorem task4 : ∃ x, 
    -10 * (x - 36) ^ 2 + 4000 = 3000 
    ∧ selling_price_range x 
    ∧ (x = 26 ∨ x = 46) := 
by 
    -- Your code here 
    sorry

end task1_task2_task3_task4_l639_639706


namespace sum_of_b_values_with_rational_roots_l639_639766

theorem sum_of_b_values_with_rational_roots : 
  let quadratic_eq_has_rational_roots (b : ℕ) := ∃ (x y : ℚ), y ≠ 0 ∧ (3 * x^2 + 7 * x + b = 0)
  let discriminant_is_perfect_square (b : ℕ) := ∃ (k : ℕ), 49 - 12 * b = k^2
  (∑ b in {b : ℕ | (b > 0) ∧ quadratic_eq_has_rational_roots b ∧ discriminant_is_perfect_square b}, b) = 6 :=
by sorry

end sum_of_b_values_with_rational_roots_l639_639766


namespace island_knight_majority_villages_l639_639556

def NumVillages := 1000
def NumInhabitants := 99
def TotalKnights := 54054
def AnswersPerVillage : ℕ := 66 -- Number of villagers who answered "more knights"
def RemainingAnswersPerVillage : ℕ := 33 -- Number of villagers who answered "more liars"

theorem island_knight_majority_villages : 
  ∃ n : ℕ, n = 638 ∧ (66 * n + 33 * (NumVillages - n) = TotalKnights) :=
by -- Begin the proof
  sorry -- Proof to be filled in later

end island_knight_majority_villages_l639_639556


namespace certain_number_is_3500_l639_639688

theorem certain_number_is_3500 :
  ∃ x : ℝ, x - (1000 / 20.50) = 3451.2195121951218 ∧ x = 3500 :=
by
  sorry

end certain_number_is_3500_l639_639688


namespace rate_per_kg_of_mangoes_l639_639403

theorem rate_per_kg_of_mangoes (kg_grapes cost_grapes paid total_kg_mangoes : ℕ) (rate_grapes : ℝ) :
  kg_grapes = 10 ∧ cost_grapes = 70 ∧ paid = 1195 ∧ total_kg_mangoes = 9 ∧ rate_grapes = 70 →
  total_kg_mangoes * (paid - kg_grapes * cost_grapes) / total_kg_mangoes = 55 :=
by 
  intros h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  cases h6 with h7 h8,
  sorry

end rate_per_kg_of_mangoes_l639_639403


namespace solve_trig_equation_l639_639094

theorem solve_trig_equation (x : ℝ) : 
  (∃ (k : ℤ), x = (Real.pi / 16) * (4 * k + 1)) ↔ 2 * (Real.cos (4 * x) - Real.sin x * Real.cos (3 * x)) = Real.sin (4 * x) + Real.sin (2 * x) :=
by
  -- The full proof detail goes here.
  sorry

end solve_trig_equation_l639_639094


namespace greatest_candies_to_office_l639_639709

-- Problem statement: Prove that the greatest possible number of candies given to the office is 7 when distributing candies among 8 students.

theorem greatest_candies_to_office (n : ℕ) : 
  ∃ k : ℕ, k = n % 8 ∧ k ≤ 7 ∧ k = 7 :=
by
  sorry

end greatest_candies_to_office_l639_639709


namespace isosceles_triangle_ratio_l639_639057

noncomputable def isosceles_triangle := 
  ∀ (A B C F E D : Type) [inner product space ℝ A] [inner product space ℝ B] 
  [inner product space ℝ C] [inner product space ℝ F] 
  [inner product space ℝ E] [inner product space ℝ D]

def midpoint {A : Type} [inner product space ℝ A] :=
  ∀ (X Y : A), (X + Y) / 2

def symmetric_point {A : Type} [inner product space ℝ A] :=
  ∀ (X F : A), 2 * F - X

theorem isosceles_triangle_ratio 
  (A B C F E D : Type) [inner product space ℝ A] 
  [inner product space ℝ B] [inner product space ℝ C] 
  [inner product space ℝ F] [inner product space ℝ E] 
  [inner product space ℝ D]
  (isosceles : ∀ (B : B), angle A B = angle C B)
  (angle_bisector : ∀ (F : F), bisect_angle ABC F)
  (parallel : ∀ (AF BC : parallel_space), parallel AF BC)
  (midpoint_E : ∀ (E : E), midpoint B C = E)
  (symmetric_D : ∀ (D : D), symmetric_point A F = D)
  (dist_EF_BD : dist E F = 1 / 2 * dist B D) :
  ∃ (EF : A B C F E D), (dist_EF_BD) :=
sorry

end isosceles_triangle_ratio_l639_639057


namespace odd_terms_in_binomial_expansion_l639_639448

theorem odd_terms_in_binomial_expansion (m n : ℤ) (h1 : m % 2 = 1) (h2 : n % 2 = 1) : 
  (λ (count odd_terms : ℕ), 
    count = 4 → 
    ∀ (k : ℕ), k ≤ 6 → 
    (binom 6 k % 2 = 1 → ((m ^ (6 - k)) * (n ^ k) % 2 = 1)) → 
    odd_terms = count) :=
by sorry

end odd_terms_in_binomial_expansion_l639_639448


namespace problem_part_i_problem_part_ii_problem_part_iii_problem_part_iv_l639_639497

open EuclideanGeometry

variables {O A B C D E F K : Point} {R : ℝ}

-- Given conditions
def circle (O R : ℝ) (P : Point) : Prop := dist O P = R
def tangent (e : Line) (C : Circle) : Prop := ∃ (A : Point), tangent_at A C e
def antipodal (C : Circle) (P E : Point) : Prop := dist_center C P = dist_center C E 
    ∧ ∠(C.center, P, E) = π -- E is the antipodal point of C

-- Problem: Prove (question, conditions, correct answer)
theorem problem_part_i
  (circle_K : circle O R K)
  (tangent_e : tangent e K)
  (parallel_OA_BC : parallel (line_through O A) (line_through B C))
  (intersects_BC_K : (B ∈ K) ∧ (C ∈ K) ∧ (C.between B D))
  (e_intersects_D : D ∈ e)
  (antipodal_C_E : antipodal K C E)
  (EA_intersects_BD : (line_through E A) ∩ (line_through B D) = some F) :
  is_isosceles_triangle C E F := sorry

theorem problem_part_ii
  (circle_K : circle O R K)
  (tangent_e : tangent e K)
  (parallel_OA_BC : parallel (line_through O A) (line_through B C))
  (intersects_BC_K : (B ∈ K) ∧ (C ∈ K) ∧ (C.between B D))
  (e_intersects_D : D ∈ e)
  (antipodal_C_E : antipodal K C E)
  (EA_intersects_BD : (line_through E A) ∩ (line_through B D) = some F) :
  2 * dist A D = dist E B := sorry 

theorem problem_part_iii
  (circle_K : circle O R K)
  (tangent_e : tangent e K)
  (parallel_OA_BC : parallel (line_through O A) (line_through B C))
  (intersects_BC_K : (B ∈ K) ∧ (C ∈ K) ∧ (C.between B D))
  (e_intersects_D : D ∈ e)
  (antipodal_C_E : antipodal K C E)
  (EA_intersects_BD : (line_through E A) ∩ (line_through B D) = some F)
  (midpoint_CF_K : midpoint C F K) :
  dist A B = dist K O := sorry 

theorem problem_part_iv
  (circle_K : circle O (5/2) K)
  (tangent_e : tangent e K)
  (parallel_OA_BC : parallel (line_through O A) (line_through B C))
  (intersects_BC_K : (B ∈ K) ∧ (C ∈ K) ∧ (C.between B D))
  (e_intersects_D : D ∈ e)
  (antipodal_C_E : antipodal K C E)
  (EA_intersects_BD : (line_through E A) ∩ (line_through B D) = some F)
  (AD_eq_3_div_2 : dist A D = 3/2) :
  area (triangle E B F) = 15/4 := sorry 

end problem_part_i_problem_part_ii_problem_part_iii_problem_part_iv_l639_639497


namespace costForFirstKgs_l639_639254

noncomputable def applePrice (l : ℝ) (q : ℝ) (x : ℝ) (totalWeight : ℝ) : ℝ :=
  if totalWeight <= x then l * totalWeight else l * x + q * (totalWeight - x)

theorem costForFirstKgs (l q x : ℝ) :
  l = 10 ∧ q = 11 ∧ (applePrice l q x 33 = 333) ∧ (applePrice l q x 36 = 366) ∧ (applePrice l q 15 15 = 150) → x = 30 := 
by
  sorry

end costForFirstKgs_l639_639254


namespace triangle_angle_equality_l639_639503

theorem triangle_angle_equality 
  (A B C M N P Q : Type) [has_angle A] [has_angle B] [has_angle C] 
  [has_angle M] [has_angle N] [has_angle P] [has_angle Q]
  (h_triangle : is_triangle A B C)
  (h_BC : lies_on M B C) (h_CN : lies_on N B C)
  (h_BM : BM = CN) (h_M_between_BN : lies_between M B N)
  (h_ANC : lies_on P A N) (h_AMC : lies_on Q A M)
  (h_angle_PRM : ∠PMC = ∠MAB) (h_angle_QNB : ∠QNB = ∠NAC) :
  ∠QBC = ∠PCB :=
by 
  -- Proof will go here
  sorry

end triangle_angle_equality_l639_639503


namespace train_speed_l639_639713

theorem train_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 400) (h_time : time = 40) : distance / time = 10 := by
  rw [h_distance, h_time]
  norm_num

end train_speed_l639_639713


namespace retailer_profit_percentage_l639_639701

theorem retailer_profit_percentage :
  ∀ (mp cp sp d n : ℝ),
  (mp = 1) →
  (cp = 36) →
  (d = 1/100) →
  (n = 120) →
  (sp = (1 - d) * mp * n) →
  ((sp - cp) / cp) * 100 = 230 := 
begin
  intros,
  sorry
end

end retailer_profit_percentage_l639_639701


namespace second_number_pascal_triangle_with_n_plus_two_equals_43_l639_639157

theorem second_number_pascal_triangle_with_n_plus_two_equals_43 :
  (Nat.choose 42 1) = 42 :=
by
  -- Pascal's triangle's (n)th row has (n+1) numbers
  -- We solved k + 1 = 43 to get k = 42
  -- The second number in 43-numbered row is C(42, 1) = 42
  sorry 

end second_number_pascal_triangle_with_n_plus_two_equals_43_l639_639157


namespace ab_bc_cd_da_le_four_l639_639531

theorem ab_bc_cd_da_le_four (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 :=
by
  sorry

end ab_bc_cd_da_le_four_l639_639531


namespace radius_of_circle_l639_639023

-- Definitions based on conditions in the problem
variables {O A A' B B' P D C : Type} 
variables [InCircle : O ∈ set [A, A', B, B']]
variables [A, A' ⟂ B, B']
variables [ant : ∀x ∈ [A, B, A', B'], anticlockwise_order x]
variables [minor_arc : P ∈ minor_arc(A', B')]
variables [interA : AP ∩ BB' = {D}]
variables [interB : BP ∩ AA' = {C}]
variables [quadrilateral_area : area (quadrilateral A B C D) = 100]

-- The main theorem to prove
theorem radius_of_circle (ω : Type) (r : ℝ) [is_circle ω] 
  (H : circle_radius_eq_radius (O, A, B, A', B', P, D, C) r = 10) : 
  radius ω = 10 :=
sorry

end radius_of_circle_l639_639023


namespace acute_isosceles_triangle_inscribed_circle_l639_639238

variables {A B C D : Type} [EuclideanGeometry A B C D]

theorem acute_isosceles_triangle_inscribed_circle (h1 : is_acute A B C)
    (h2 : is_isosceles A B C) (h3 : inscribed_in_circle A B C)
    (h4 : are_tangents B C D) (h5 : ∠ ABC = ∠ ACB = 3 * ∠ D)
    (h6 : tan_from_circle_point B C D) :
    (∠ BAC = 5 * π / 11) :=
sorry

end acute_isosceles_triangle_inscribed_circle_l639_639238


namespace five_digit_number_count_l639_639318

theorem five_digit_number_count : ∃ n, n = 1134 ∧ ∀ (a b c d e : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧ 
  (a < b ∧ b < c ∧ c > d ∧ d > e) → n = 1134 :=
by 
  sorry

end five_digit_number_count_l639_639318


namespace annika_return_time_l639_639738

variables (rate : ℝ) (d1 d2 : ℝ)

-- Given conditions
def constant_rate := rate = 12
def initial_distance := d1 = 2.75
def total_distance_east := d2 = 3.5

-- Question and proof of the correct answer
theorem annika_return_time
  (h1 : constant_rate rate)
  (h2 : initial_distance d1)
  (h3 : total_distance_east d2) : 
  rate * (d2 - d1 + d2) = 51 :=
begin
  rw [constant_rate, initial_distance, total_distance_east] at *,
  sorry,
end

end annika_return_time_l639_639738


namespace largest_N_l639_639762

def number_ordered (row : Fin 100 → ℕ) : Prop :=
  (∃ p : Fin 100 → Fin 100, ∀ i, row i = (p i).succ)

def valid_table (T : Fin (fact $ nat.factorial 100 / 2^50) → Fin 100 → ℕ) : Prop :=
  (∀ r, number_ordered (T r)) ∧
  (∀ r s, r ≠ s → ∃ c, abs (T r c - T s c) ≥ 2)

theorem largest_N :
  (∃ (T : Fin (fact $ nat.factorial 100 / 2^50) → Fin 100 → ℕ), valid_table T) ∧ 
  ¬ (∃ (T : Fin (fact $ nat.factorial 100 / 2^50 + 1) → Fin 100 → ℕ), valid_table T) :=
by sorry

end largest_N_l639_639762


namespace binomial_expansion_sum_coeffs_eq_zero_binomial_term_with_largest_coeff_binomial_rational_terms_l639_639917

noncomputable def binomial_terms_equal (n : ℕ) : Prop :=
  Binomial.coeff n 1 = Binomial.coeff n 7

theorem binomial_expansion_sum_coeffs_eq_zero (x : ℝ) (h : binomial_terms_equal 8) :
  (∑ k in Finset.range (8 + 1), (Binomial.coeff 8 k) * ((x ^ - (1/3)) ^ (8 - k)) * (-1)^k) = 0 :=
by
  have h : 8 = 8 := rfl
  simp [h]
  sorry

theorem binomial_term_with_largest_coeff (x : ℝ) (h : binomial_terms_equal 8) :
  ∃ k : ℕ, k = 4 ∧ Binomial.coeff 8 k * ((x ^ - (1/3)) ^ (8 - k)) * (-1)^k = 70 * x^(-4/3) :=
by
  have h : 8 = 8 := rfl
  simp [h]
  sorry

theorem binomial_rational_terms (x : ℝ) (h : binomial_terms_equal 8) :
  { t : ℝ | ∃ k : ℕ, k ∈ {2, 5, 8} ∧ t = Binomial.coeff 8 k * ((x ^ - (1/3)) ^ (8 - k)) * (-1)^k } =
    { 28 * x^(-2), -56 * x^(-1), 1 } :=
by
  have h : 8 = 8 := rfl
  simp [h]
  sorry

end binomial_expansion_sum_coeffs_eq_zero_binomial_term_with_largest_coeff_binomial_rational_terms_l639_639917


namespace log_condition_nec_not_suff_l639_639883

theorem log_condition_nec_not_suff (x y : ℝ) :
  (log 2 (xy + 4*x - 2*y) = 3 → x^2 + y^2 - 6*x + 8*y + 25 = 0) ∧ 
  (x^2 + y^2 - 6*x + 8*y + 25 = 0 → log 2 (xy + 4*x - 2*y) = 3) ∧
  ¬(log 2 (xy + 4*x - 2*y) = 3 ↔ x^2 + y^2 - 6*x + 8*y + 25 = 0) := 
by {
  sorry
}

end log_condition_nec_not_suff_l639_639883


namespace gcd_2024_1728_l639_639154

theorem gcd_2024_1728 : Int.gcd 2024 1728 = 8 := 
by
  sorry

end gcd_2024_1728_l639_639154


namespace rectangle_area_l639_639623

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l639_639623


namespace Z_is_divisible_by_10001_l639_639048

theorem Z_is_divisible_by_10001
    (Z : ℕ) (a b c d : ℕ) (ha : a ≠ 0)
    (hZ : Z = 1000 * 10001 * a + 100 * 10001 * b + 10 * 10001 * c + 10001 * d)
    : 10001 ∣ Z :=
by {
    -- Proof omitted
    sorry
}

end Z_is_divisible_by_10001_l639_639048


namespace smoking_lung_disease_statements_incorrect_l639_639213

theorem smoking_lung_disease_statements_incorrect :
  let k : ℝ := 6.635
  let confidence_99 := 0.99
  let confidence_95 := 0.95
  (K2 : ℝ) (confidence : ℝ) 
  (h₀ : K2 = k → confidence = confidence_99)
  (h₁ : confidence = confidence_95 → false) 
  (h₂ : confidence = confidence_99 → false) :
  (¬ (K2 = k ∧ confidence = confidence_99 → ∃ n, n = 100 → ∃ m, m = 99 → m / n = confidence_99)) ∧
  (¬ (confidence = confidence_95 → confidence = 0.95 ∧ confidence_99 ≠ 0.99)) ∧
  (¬ (confidence = confidence_99 → ∃ m, m = 99 → m / 100 = confidence_99)) :=
by
  intros
  sorry

end smoking_lung_disease_statements_incorrect_l639_639213


namespace lamps_on_bridge_l639_639186

theorem lamps_on_bridge (bridge_length : ℕ) (lamp_spacing : ℕ) (num_intervals : ℕ) (num_lamps : ℕ) 
  (h1 : bridge_length = 30) 
  (h2 : lamp_spacing = 5)
  (h3 : num_intervals = bridge_length / lamp_spacing)
  (h4 : num_lamps = num_intervals + 1) :
  num_lamps = 7 := 
by
  sorry

end lamps_on_bridge_l639_639186


namespace expression_eval_l639_639596

theorem expression_eval : 2 * 3 + 2 * 3 = 12 := by
  sorry

end expression_eval_l639_639596


namespace rectangle_area_l639_639615

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l639_639615


namespace Jeff_probability_multiple_of_4_l639_639490

theorem Jeff_probability_multiple_of_4 :
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 12 ∧ (
   let moves := [1, -1, -1, -1] in
   let positions := (1 + moves) in 
    ∃ (m : ℕ), ∃ (p : ℕ), 
    m ∈ moves ∧
    p ∈ moves ∧ 
    (n + m) + (n + m + p) % 4 = 0)) = 1 / 4 := by
  sorry

end Jeff_probability_multiple_of_4_l639_639490


namespace sum_of_leading_digits_l639_639059

def N : Nat := 88888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

def leading_digit (x : Real) : Nat := 
  let integer_part := floor (x / 10 ^ (floor (log10 x)))
  Nat.ofInt integer_part

def f (r : Nat) : Nat := leading_digit (Real.cbrt N)

-- Main theorem 
theorem sum_of_leading_digits : f 3 + f 4 + f 5 + f 6 + f 7 = 7 := 
by
  -- The actual proof will go here
  sorry

end sum_of_leading_digits_l639_639059


namespace PR_eq_QS_l639_639740

-- Definitions of geometrical concepts, points, lines, and circles
structure Point :=
(x : ℝ)
(y : ℝ)

structure Circle :=
(center : Point)
(radius : ℝ)

structure Line :=
(a : Point)
(b : Point)

-- Conditions
variables (A B C P Q R S : Point)
variables (O1 O2 O : Circle)
variables (line_AC : Line)
variables (line_B : Line)

-- Conditions based on the problem statement
axiom A_on_AC : A ∈ line_AC.a ∧ C ∈ line_AC.b
axiom B_on_AC : B ∈ Line
axiom Circle_O1_diameter_AB : O1.center = midpoint A B ∧ O1.radius = distance A B / 2
axiom Circle_O2_diameter_BC : O2.center = midpoint B C ∧ O2.radius = distance B C / 2
axiom Circle_O_diameter_AC : O.center = midpoint A C ∧ O.radius = distance A C / 2
axiom points_PQ_on_O : P ∈ Circle_O ∧ Q ∈ Circle_O
axiom points_R_on_O1 : R ∈ Circle_O1
axiom points_S_on_O2 : S ∈ Circle_O2

-- The theorem we need to prove
theorem PR_eq_QS : distance P R = distance Q S :=
sorry

end PR_eq_QS_l639_639740


namespace infinite_stars_not_smaller_l639_639138

theorem infinite_stars_not_smaller (stars : ℕ → ℕ × ℕ) (h_infinite : set.infinite (set.range stars)) (h_diff : ∀ i j, i ≠ j → stars i ≠ stars j) :
  ∃ i j, (stars i).fst ≥ (stars j).fst ∧ (stars i).snd ≥ (stars j).snd :=
sorry

end infinite_stars_not_smaller_l639_639138


namespace multiply_58_62_l639_639270

theorem multiply_58_62 : 58 * 62 = 3596 :=
by
  let a : ℕ := 60
  let b : ℕ := 2
  have h1 : (a - b) * (a + b) = a^2 - b^2 := by exact Nat.mul_sub_mul_add_mul_eq a b 
  have h2 : a^2 = 3600 := rfl
  have h3 : b^2 = 4 := rfl
  calc
    58 * 62 = (60 - 2) * (60 + 2) : by rfl
    ... = 60^2 - 2^2 : h1
    ... = 3600 - 4  : by rw [h2, h3]
    ... = 3596 : by rfl

end multiply_58_62_l639_639270


namespace find_points_attempts_l639_639555

theorem find_points_attempts (n : ℕ) : ∃ k ≤ (n + 1) ^ 2, ∀ m ∈ (finset.range n).map (fun i => Misha_selects i), Kolya_discovers m :=
by
  sorry

end find_points_attempts_l639_639555


namespace martin_distance_l639_639071

def speed : ℝ := 12.0  -- Speed in miles per hour
def time : ℝ := 6.0    -- Time in hours

theorem martin_distance : (speed * time) = 72.0 :=
by
  sorry

end martin_distance_l639_639071


namespace factor_difference_of_cubes_l639_639309

theorem factor_difference_of_cubes (t : ℝ) : t^3 - 8 = (t - 2) * (t^2 + 2*t + 4) := 
begin
  sorry
end

end factor_difference_of_cubes_l639_639309


namespace count_sets_satisfy_condition_l639_639396

theorem count_sets_satisfy_condition :
  let U := {1, 2, 3, 4, 5}
  let A : Finset ℕ := {a_1, a_2, a_3}
  ∃ (n : ℕ), n = 10 ∧
    ∀ (a_1 a_2 a_3 : ℕ) (hU1 : a_1 ∈ U) (hU2 : a_2 ∈ U) (hU3 : a_3 ∈ U),
      a_3 ≥ a_2 + 1 ∧ a_2 ≥ a_1 + 2 → 
      A.count (λ {a_1 b_3 a_3}, a_3 >= a_2 + 1 ∧ a_2 >= a_1 + 2) = n :=
begin
  sorry -- proof is omitted
end

end count_sets_satisfy_condition_l639_639396


namespace find_k_l639_639234

noncomputable theory
open_locale classical

variables (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)

-- Condition 1: ABC is an acute isosceles triangle inscribed in a circle
-- Condition 2: Tangents from B and C meet at D
-- Condition 3: angle ABC = angle ACB = 3 * angle D
    
def tangents_intersect_at_D (ABC : triangle) (B C D : point ABC) : Prop :=
  tangent_to_circle_at B D ∧ tangent_to_circle_at C D ∧ B ≠ C

def isosceles_triangle_condition (angle : point ABC → point ABC → point ABC → real) (B C : point ABC) : Prop :=
  angle ABC B C = angle ABC C B

def angle_triple_D (angle : point ABC → point ABC → point ABC → real) (B C D : point ABC) : Prop :=
  ∃ k, angle ABC B C = 3 * angle ABC D C ∧ angle ABC A B = k * π

theorem find_k (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)
  (h1 : tangents_intersect_at_D ABC B C D)
  (h2 : isosceles_triangle_condition angle B C)
  (h3 : angle_triple_D angle B C D) :
  ∃ k : real, angle ABC A B = (5 / 11) * π ∧ k = 5 / 11 :=
by sorry

end find_k_l639_639234


namespace floor_alpha_n_squared_even_infinite_l639_639080

theorem floor_alpha_n_squared_even_infinite (α : ℝ) (hα : 0 < α) :
  ∃ᶠ n in at_top, ∃ k : ℤ, 2 * k = int.floor (α * n^2) :=
sorry

end floor_alpha_n_squared_even_infinite_l639_639080


namespace a_when_b_is_16_l639_639101

noncomputable def a_varies_inversely_with_c (a c : ℝ) : Prop :=
  ∃ (k : ℝ), a * c = k

theorem a_when_b_is_16 (a b c : ℝ)
  (h1 : a_varies_inversely_with_c a c)
  (h2 : c = Real.sqrt b)
  (h3 : ∀ (a₀ b₀ : ℝ), b₀ = 9 → a₀ = 15 → a_varies_inversely_with_c a₀ (Real.sqrt b₀)) : 
  b = 16 → a = 11.25 :=
by
  intro hb_eq_16
  rw [hb_eq_16, Real.sqrt, h2] at h1
  sorry

end a_when_b_is_16_l639_639101


namespace sum_a1_to_a12_l639_639025

variable {a : ℕ → ℕ}

axiom geom_seq (n : ℕ) : a n * a (n + 1) * a (n + 2) = 8
axiom a_1 : a 1 = 1
axiom a_2 : a 2 = 2

theorem sum_a1_to_a12 : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12) = 28 :=
by
  sorry

end sum_a1_to_a12_l639_639025


namespace DF_EG_intersect_on_circumcircle_ABC_l639_639520

-- Definitions for the given conditions
variables {A B C I P D E F G : Type*} [Point A B C I P D E F G]

-- Hypotheses and conditions
hypotheses (h_ABC_isosceles : isosceles_triangle A B C)
           (h_AC_eq_BC : AC = BC)
           (h_incenter_I : incenter I A B C)
           (h_P_on_circumcircle_AIB : on_circumcircle P A I B)
           (h_P_inside_ABC : inside_triangle P A B C)
           (h_D_on_AB : on_line D A B ∧ parallel P D CA)
           (h_E_on_AB : on_line E A B ∧ parallel P E CB)
           (h_F_on_CA : on_line F C A ∧ parallel P F AB)
           (h_G_on_CB : on_line G C B ∧ parallel P G AB)

-- Proof goal
theorem DF_EG_intersect_on_circumcircle_ABC :
  intersect (line_through D F) (line_through E G) (circumcircle A B C) :=
sorry

end DF_EG_intersect_on_circumcircle_ABC_l639_639520


namespace ab_bc_cd_da_le_four_l639_639529

theorem ab_bc_cd_da_le_four (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 :=
by
  sorry

end ab_bc_cd_da_le_four_l639_639529


namespace distinct_digits_sum_l639_639540

theorem distinct_digits_sum (A B C D G : ℕ) (AB CD GGG : ℕ)
  (h1: AB = 10 * A + B)
  (h2: CD = 10 * C + D)
  (h3: GGG = 111 * G)
  (h4: AB * CD = GGG)
  (h5: A ≠ B)
  (h6: A ≠ C)
  (h7: A ≠ D)
  (h8: A ≠ G)
  (h9: B ≠ C)
  (h10: B ≠ D)
  (h11: B ≠ G)
  (h12: C ≠ D)
  (h13: C ≠ G)
  (h14: D ≠ G)
  (hA: A < 10)
  (hB: B < 10)
  (hC: C < 10)
  (hD: D < 10)
  (hG: G < 10)
  : A + B + C + D + G = 17 := sorry

end distinct_digits_sum_l639_639540


namespace find_k_l639_639231

noncomputable theory
open_locale classical

variables (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)

-- Condition 1: ABC is an acute isosceles triangle inscribed in a circle
-- Condition 2: Tangents from B and C meet at D
-- Condition 3: angle ABC = angle ACB = 3 * angle D
    
def tangents_intersect_at_D (ABC : triangle) (B C D : point ABC) : Prop :=
  tangent_to_circle_at B D ∧ tangent_to_circle_at C D ∧ B ≠ C

def isosceles_triangle_condition (angle : point ABC → point ABC → point ABC → real) (B C : point ABC) : Prop :=
  angle ABC B C = angle ABC C B

def angle_triple_D (angle : point ABC → point ABC → point ABC → real) (B C D : point ABC) : Prop :=
  ∃ k, angle ABC B C = 3 * angle ABC D C ∧ angle ABC A B = k * π

theorem find_k (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)
  (h1 : tangents_intersect_at_D ABC B C D)
  (h2 : isosceles_triangle_condition angle B C)
  (h3 : angle_triple_D angle B C D) :
  ∃ k : real, angle ABC A B = (5 / 11) * π ∧ k = 5 / 11 :=
by sorry

end find_k_l639_639231


namespace EDTA_Ca2_complex_weight_l639_639748

-- Definitions of atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_Ca : ℝ := 40.08

-- Number of atoms in EDTA
def num_atoms_C : ℝ := 10
def num_atoms_H : ℝ := 16
def num_atoms_N : ℝ := 2
def num_atoms_O : ℝ := 8

-- Molecular weight of EDTA
def molecular_weight_EDTA : ℝ :=
  num_atoms_C * atomic_weight_C +
  num_atoms_H * atomic_weight_H +
  num_atoms_N * atomic_weight_N +
  num_atoms_O * atomic_weight_O

-- Proof that the molecular weight of the complex is 332.328 g/mol
theorem EDTA_Ca2_complex_weight : molecular_weight_EDTA + atomic_weight_Ca = 332.328 := by
  sorry

end EDTA_Ca2_complex_weight_l639_639748


namespace DE_parallel_BC_l639_639495

theorem DE_parallel_BC (A B C D E : Type) [triangle ABC]
  (HD : is_orthogonal_projection_of_onto_internal_bisector A B D)
  (HE : is_orthogonal_projection_of_onto_internal_bisector A C E) : 
  parallel DE BC :=
sorry

end DE_parallel_BC_l639_639495


namespace quadratic_roots_sum_l639_639963

theorem quadratic_roots_sum (m n : ℝ) (i : ℂ) (h_imaginary_unit : i^2 = -1)
  (h_root : is_root (λ x : ℂ, x^2 + m * x + n) (1 - complex.sqrt 3 * i)) :
  m + n = 2 :=
by
  sorry

end quadratic_roots_sum_l639_639963


namespace batsman_average_after_17th_inning_l639_639683

variable (A : ℝ) (total_runs : ℝ) (new_average : ℝ)
hypothesis h1 : total_runs = 16 * A + 87
hypothesis h2 : new_average =  (total_runs / 17)
hypothesis h3 : new_average = A + 3

theorem batsman_average_after_17th_inning :
  new_average = 39 :=
by
  sorry

end batsman_average_after_17th_inning_l639_639683


namespace problem_statement_l639_639812

noncomputable def sequence_an (t : ℝ) (n : ℕ) : ℝ := 2 * t^n

def Sn (t : ℝ) (n : ℕ) : ℝ := (2 * (1 - t^n)) / (3 * (1 - t))

def bn (an : ℕ → ℝ) (Sn : ℕ → ℝ) (n : ℕ) : ℝ := (-an n) * Real.log (3 - Sn n)

def Tn (n : ℕ) : ℝ := (∑ i in finset.range n, bn (sequence_an (1/3)) (Sn (1/3)) i)

theorem problem_statement (t : ℝ) (h : t ≠ 0) (h2 : t ≠ 1) :
  (∀ n, (t - 1) * Sn t n = t * (sequence_an t n - 2)) →
  sequence_an t = λ n, 2 * t^n ∧
  Tn n = (3 / 2) - (2 * (n+1) / (2 * 3 ^ n)) :=
by
  intros,
  sorry

end problem_statement_l639_639812


namespace average_minutes_heard_l639_639196

theorem average_minutes_heard (total_people attendees : ℕ)'
  (percentage_full percentage_inattentive : ℕ) (lecture_duration : ℝ) : 
  60 * 90 + 0 * 30 + 44 * 45 + 66 * 67.5 = 11835 :=
by {
  have h1 : attendees = 200 := rfl,
  have h2 : 30 * percentage_full = 60 := rfl,
  have h3 : 15 * percentage_inattentive = 30 := rfl,
  have h4 : total_people - (60 + 30) = 110 := by simp,
  have h5 : 40 * 110 / 100 = 44 := rfl,
  have h6 : 60 * 110 / 100 = 66 := rfl,
  have h7 : 60 * 90 + 0 * 30 + 44 * 45 + 66 * 67.5 = 11835 := by simp,
  exact h7,
}

end average_minutes_heard_l639_639196


namespace max_area_l639_639915

structure Points :=
  (A B C P : ℝ × ℝ)
  (PA PB PC BC : ℕ)

def condition (pts : Points) : Prop :=
  pts.PA = dist pts.P pts.A ∧
  pts.PB = dist pts.P pts.B ∧
  pts.PC = dist pts.P pts.C ∧
  pts.BC = dist pts.B pts.C

theorem max_area {pts : Points} (h : condition pts) : 
  ∃ hA, hC,
  h = 3 ∧
  hB = 4 ∧
  hC = 5 ∧
  hBC = 6 →
  max_area_triangle pts.A pts.B pts.C = 19 :=
begin
  sorry
end

end max_area_l639_639915


namespace angles_in_interval_l639_639775

-- Define the main statement we need to prove
theorem angles_in_interval (theta : ℝ) (h1 : 0 ≤ theta) (h2 : theta ≤ 2 * Real.pi) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos theta - x * (1 - x) + (1-x)^2 * Real.sin theta < 0) →
  (Real.pi / 2 < theta ∧ theta < 3 * Real.pi / 2) :=
by
  sorry

end angles_in_interval_l639_639775


namespace find_lambda_l639_639402

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 0)
def c : ℝ × ℝ := (3, 4)

-- Given the condition that (λ * a + b) is perpendicular to c
def lambda_perpendicular (λ : ℝ) : Prop :=
  let v := (λ * a.1 + b.1, λ * a.2 + b.2) in
  (v.1 * c.1 + v.2 * c.2) = 0

-- Prove that λ == -3/11 given the above condition
theorem find_lambda (λ : ℝ) (h : lambda_perpendicular λ) : λ = -3 / 11 :=
  sorry

end find_lambda_l639_639402


namespace sum_p_bound_l639_639854

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 - x

def b_n (n : ℕ) : ℝ := Real.exp (1 / n) - 1 - (1 / n)

def a_n (n : ℕ) : ℝ := 1 / n

def p_k (k : ℕ) (n : ℕ) : ℝ := (∏ i in Finset.range k, a_n (2 * (i + 1))) / 
                                (∏ i in Finset.range k, a_n (2 * i + 1))

theorem sum_p_bound (n : ℕ) (hn : 0 < n) :
  (∑ k in Finset.range n, p_k k n) < Real.sqrt ((2 / a_n n) + 1) - 1 :=
sorry

end sum_p_bound_l639_639854


namespace even_function_f_at_neg4_l639_639838

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then log x / log 2 + 1 else log (-x) / log 2 + 1

theorem even_function_f_at_neg4 : f (-4) = 3 := 
by
  sorry

end even_function_f_at_neg4_l639_639838


namespace find_k_in_isosceles_triangle_l639_639228

theorem find_k_in_isosceles_triangle 
  (A B C D : Type)
  (h_triangle : triangle ABC)
  (h_acute : acute ABC)
  (h_isosceles : isosceles ABC)
  (h_circumscribed : circumscribed ABC)
  (h_tangents : tangents B C D)
  (h_angle_relation : ∠ABC = ∠ACB ∧ ∠ABC = 3 * ∠D)
  : ∠BAC = (5 * π / 11) := 
by 
  sorry

end find_k_in_isosceles_triangle_l639_639228


namespace continuous_iff_k_n_continuous_l639_639387

def k_n (n : ℝ) (x : ℝ) : ℝ :=
  if x ≤ -n then -n
  else if x < n then x
  else n

noncomputable def f_continuous_iff_k_n_f_continuous (f : ℝ → ℝ) : Prop :=
  (Continuous f) ↔ ∀ n : ℝ, Continuous (λ x, k_n n (f x))

theorem continuous_iff_k_n_continuous (f : ℝ → ℝ) : f_continuous_iff_k_n_f_continuous f :=
sorry

end continuous_iff_k_n_continuous_l639_639387


namespace acute_isosceles_triangle_inscribed_circle_l639_639239

variables {A B C D : Type} [EuclideanGeometry A B C D]

theorem acute_isosceles_triangle_inscribed_circle (h1 : is_acute A B C)
    (h2 : is_isosceles A B C) (h3 : inscribed_in_circle A B C)
    (h4 : are_tangents B C D) (h5 : ∠ ABC = ∠ ACB = 3 * ∠ D)
    (h6 : tan_from_circle_point B C D) :
    (∠ BAC = 5 * π / 11) :=
sorry

end acute_isosceles_triangle_inscribed_circle_l639_639239


namespace triangle_areas_equal_l639_639958

open Real EuclideanGeometry

/-
Let ω be a circle with diameter AB. Circle γ, with center C lying on ω, is tangent to AB at D and intersects ω at E and F. Prove triangles CEF and DEF have the same area.
-/

variables {A B C D E F : Point}
variable {ω : circle}
variable {γ : circle}
variable [hab : ω.diameter A B]
variable [hc: C ∈ ω]
variable [hγ : γ.center C ∧ γ.tangent AB D]
variable [h_intersect : E ∈ γ.points ∧ F ∈ γ.points]

theorem triangle_areas_equal : triangle_area C E F = triangle_area D E F :=
sorry

end triangle_areas_equal_l639_639958


namespace total_ants_correct_l639_639698

-- Define the conditions
def park_width_ft : ℕ := 450
def park_length_ft : ℕ := 600
def ants_per_sq_inch_first_half : ℕ := 2
def ants_per_sq_inch_second_half : ℕ := 4

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Convert width and length from feet to inches
def park_width_inch : ℕ := park_width_ft * feet_to_inches
def park_length_inch : ℕ := park_length_ft * feet_to_inches

-- Define the area of each half of the park in square inches
def half_length_inch : ℕ := park_length_inch / 2
def area_first_half_sq_inch : ℕ := park_width_inch * half_length_inch
def area_second_half_sq_inch : ℕ := park_width_inch * half_length_inch

-- Define the number of ants in each half
def ants_first_half : ℕ := ants_per_sq_inch_first_half * area_first_half_sq_inch
def ants_second_half : ℕ := ants_per_sq_inch_second_half * area_second_half_sq_inch

-- Define the total number of ants
def total_ants : ℕ := ants_first_half + ants_second_half

-- The proof problem
theorem total_ants_correct : total_ants = 116640000 := by
  sorry

end total_ants_correct_l639_639698


namespace tan_120_deg_l639_639272

theorem tan_120_deg : Real.tan (120 * Real.pi / 180) = -Real.sqrt 3 := by
  sorry

end tan_120_deg_l639_639272


namespace range_of_varphi_l639_639852

-- Define the function f(x)
noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ) + 1

-- Define the given conditions
theorem range_of_varphi (ω : ℝ) (f_gt_one : ∀ (x : ℝ), x ∈ Ioo (-π / 12) (π / 3) → f 2 φ x > 1) : 
  abs φ ≤ π / 2 → 
  (∀ (x1 x2 : ℝ), 
    ((f ω φ x1 = -1) ∧ (f ω φ x2 = -1) ∧ (x2 - x1 = π)) → 
    (ω = 2 ∧ φ ∈ Icc (π / 6) (π / 3))) :=
begin
  sorry
end

end range_of_varphi_l639_639852


namespace combination_12_6_binomial_expansion_12_l639_639271

theorem combination_12_6 : nat.choose 12 6 = 924 := 
  sorry

theorem binomial_expansion_12 (x y : ℕ) (hx : x = 1) (hy : y = 1): (x + y) ^ 12 = 4096 := 
  by
    rw [hx, hy]
    norm_num

end combination_12_6_binomial_expansion_12_l639_639271


namespace number_of_roses_cut_l639_639642

theorem number_of_roses_cut (initial_roses final_roses roses_cut : ℕ) (h1 : initial_roses = 3) (h2 : final_roses = 14) : 
  roses_cut = final_roses - initial_roses := 
by
  rw [h1, h2]
  exact rfl

end number_of_roses_cut_l639_639642


namespace sum_of_reciprocals_of_polynomial_l639_639961

noncomputable def sum_of_reciprocals (a b c d k : ℝ) (hk : k ≠ 0) (roots_on_circle : ∀ z, ↑(z ^ 4) + a * ↑(z ^ 3) + b * ↑(z ^ 2) + c * z + d = 0 → complex.abs z = 2) : ℝ :=
(-a) / (4 * k)

theorem sum_of_reciprocals_of_polynomial :
  ∀ a b c d k : ℝ, k ≠ 0 → (∀ z : ℂ, z ^ 4 * ↑k + ↑a * z ^ 3 + ↑b * z ^ 2 + ↑c * z + ↑d = 0 → complex.abs z = 2) →
  sum_of_reciprocals a b c d k = -a / (4 * k) :=
by {
  intros a b c d k hk roots_on_circle,
  sorry
}

end sum_of_reciprocals_of_polynomial_l639_639961


namespace simplify_fraction_l639_639090

def z1 : ℂ := 5 - 3 * complex.i
def z2 : ℂ := 2 - 3 * complex.i

theorem simplify_fraction :
  z1 / z2 = - (19 / 5 : ℂ) - (9 / 5 : ℂ) * complex.i := 
by
  sorry

end simplify_fraction_l639_639090


namespace log_comparison_l639_639216

theorem log_comparison :
  (∀ x y : ℝ, 3 > 1 → x > y → log 3 x > log 3 y) →
  (∀ x y : ℝ, (1 / 3) < 1 → x > y → log (1 / 3) x < log (1 / 3) y) →
  (\left (\frac{1}{5}\right)^{0} = 1) →
  log 3 4 > 1 ∧ 1 > log (1 / 3) 10 :=
sorry

end log_comparison_l639_639216


namespace triangle_altitudes_ineq_triangle_exradii_ineq_l639_639034

variables {ABC : Type} [triangle: Triangle ABC]
variables {a b c s Δ r r_a r_b r_c h_a h_b h_c : Real}
-- Defining conditions
variable (A1: ⟦ triangle ⟧(ABC, a, b, c))
variable (H1: semi_perimeter(ABC, s))
variable (H2: area(ABC, Δ))
variable (H3: inradius(ABC, r))
variable (H4: exradius(ABC, r_a, r_b, r_c))
variable (H5: altitudes(ABC, h_a, h_b, h_c))

-- Problem 1: Prove h_a h_b h_c ≥ 27 r^3
theorem triangle_altitudes_ineq : h_a * h_b * h_c ≥ 27 * r^3 := sorry

-- Problem 2: Prove r_a r_b r_c ≥ 27 r^3
theorem triangle_exradii_ineq : r_a * r_b * r_c ≥ 27 * r^3 := sorry

end triangle_altitudes_ineq_triangle_exradii_ineq_l639_639034


namespace count_integers_between_200_250_l639_639438

theorem count_integers_between_200_250 :
  {n : ℕ // 200 ≤ n ∧ n < 250 ∧ 
            (let d2 := (n / 10) % 10, d3 := n % 10 in
             (n / 100 = 2) ∧ (d2 ≠ d3) ∧ (2 < d2) ∧ (d2 < d3)
            )}.to_finset.card = 11 :=
by
  -- Start the proof process here
  sorry

end count_integers_between_200_250_l639_639438


namespace max_possible_trailing_zeros_l639_639654

theorem max_possible_trailing_zeros :
  ∃ n : ℕ, n ≤ 9 ∧ 
            (∀ digits : Set (Fin 10), digits = {1, 2, 3, 4, 5, 6, 7, 8, 9} →
            ∃ nine_digit_nums : Fin 9 → ℕ, 
            (∀ i, nine_digit_nums i ∈ perms digits) ∧
            ∃ sum_nine_digits : ℕ, sum_nine_digits = ∑ i, nine_digit_nums i ∧
            sum_nine_digits % 10^n = 0) →
            n = 8 := 
by
  -- Proof omitted
  sorry

end max_possible_trailing_zeros_l639_639654


namespace area_rectangle_BGEF_l639_639919

-- Define Points and Rectangle
variables {Point : Type} (A B C D E F G : Point)

-- Define the lengths of AB and AD
def AB_length : ℝ := 8
def AD_length : ℝ := 10

-- Define the area of triangle DFG
def area_triangle_DFG : ℝ := 30

-- Calculate areas
def area_rectangle_ABCD : ℝ := AB_length * AD_length := 80
def area_triangle_ABC : ℝ := (1 / 2) * AB_length * AD_length := 40
def area_triangle_BFG : ℝ := area_triangle_ABC - area_triangle_DFG := 10

-- The Lean 4 statement to prove that the area of rectangle BGEF is 20
theorem area_rectangle_BGEF : area_rectangle_BGEF = 2 * area_triangle_BFG := 20 :=
by
  sorry

end area_rectangle_BGEF_l639_639919


namespace range_of_a_for_monotonicity_l639_639850

noncomputable def f (x a : ℝ) : ℝ := (3 * x / a) - 2 * x^2 + Real.log x

theorem range_of_a_for_monotonicity :
  (∀ x ∈ set.Icc 1 2, deriv (λ x => f x a) x ≥ 0) 
  ∨ (∀ x ∈ set.Icc 1 2, deriv (λ x => f x a) x ≤ 0) 
  ↔ (0 < a ∧ a ≤ 2 / 5) ∨ a ≥ 1 :=
sorry

end range_of_a_for_monotonicity_l639_639850


namespace angle_FCE_eq_angle_ADE_angle_FEC_eq_angle_BDC_l639_639176

-- Define the geometric setup
variables {A B C D E F : Type} [LinearOrderedField A]
-- Let ABCDE be a convex pentagon such that DC = DE
variable (ABCDE : ConvexPentagon A B C D E) (h1 : D-C = E-D)
-- with angles ∠C = ∠E = 90°
variable (hc : angle (C.1 - D.1) (C.2 - D.2) = 90) (he : angle (E.1 - D.1) (E.2 - D.2) = 90)
-- Let F be on AB such that AF/BF = AE/BC
variable (F : Point A B) (h2 : ratio (A.1-F.1)/(B.1-F.1) = ratio (A.1-E.1)/(B.1-C.1))

-- Prove ∠FCE = ∠ADE
theorem angle_FCE_eq_angle_ADE :
  angle (F.1 - C.1) (E.1 - C.1) = angle (A.1 - D.1) (E.1 - D.1) := sorry

-- Prove ∠FEC = ∠BDC
theorem angle_FEC_eq_angle_BDC :
  angle (F.1 - E.1) (C.1 - E.1) = angle (B.1 - D.1) (C.1 - D.1) := sorry

end angle_FCE_eq_angle_ADE_angle_FEC_eq_angle_BDC_l639_639176


namespace sequence_nonzero_l639_639368

def sequence (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if (sequence (n - 2)) * (sequence (n - 1)) % 2 = 0 then
    5 * (sequence (n - 1)) - 3 * (sequence (n - 2))
  else
    (sequence (n - 1)) - (sequence (n - 2))

theorem sequence_nonzero : ∀ (n : ℕ), sequence n ≠ 0 :=
by
  sorry

end sequence_nonzero_l639_639368


namespace six_points_intersect_l639_639361

-- Define the condition that no four points are coplanar
def no_four_coplanar (points : List (ℝ^3)) : Prop :=
  ∀ (S : Finset (ℝ^3)), S.card = 4 → ¬ ∃ (a b c d : ℝ^3), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    affine_independent ℝ ![a,b,c,d]

-- Define that the triangles formed by two groups intersect
def triangles_intersect (tri1 tri2 : set (ℝ^3)) : Prop :=
  ∃ p, p ∈ tri1 ∧ p ∈ tri2

-- Define the main theorem based on the conditions and correct answer
theorem six_points_intersect :
  ∀ (A1 A2 A3 A4 A5 A6 : ℝ^3), no_four_coplanar [A1, A2, A3, A4, A5, A6] →
  ∃ (G1 G2 G3 G4 G5 G6 : ℝ^3),
    ((G1 = A1 ∨ G1 = A2 ∨ G1 = A3 ∨ G1 = A4 ∨ G1 = A5 ∨ G1 = A6) ∧
    (G2 = A1 ∨ G2 = A2 ∨ G2 = A3 ∨ G2 = A4 ∨ G2 = A5 ∨ G2 = A6) ∧
    (G3 = A1 ∨ G3 = A2 ∨ G3 = A3 ∨ G3 = A4 ∨ G3 = A5 ∨ G3 = A6) ∧
    (G4 = A1 ∨ G4 = A2 ∨ G4 = A3 ∨ G4 = A4 ∨ G4 = A5 ∨ G4 = A6) ∧
    (G5 = A1 ∨ G5 = A2 ∨ G5 = A3 ∨ G5 = A4 ∨ G5 = A5 ∨ G5 = A6) ∧
    (G6 = A1 ∨ G6 = A2 ∨ G6 = A3 ∨ G6 = A4 ∨ G6 = A5 ∨ G6 = A6)) ∧
    let tri1 := {[G1, G2, G3]} in
    let tri2 := {[G4, G5, G6]} in
    triangles_intersect tri1 tri2 :=
sorry

end six_points_intersect_l639_639361


namespace james_vegetable_intake_l639_639931

theorem james_vegetable_intake :
  let daily_asparagus := 0.25
  let daily_broccoli := 0.25
  let daily_intake := daily_asparagus + daily_broccoli
  let doubled_daily_intake := daily_intake * 2
  let weekly_intake_asparagus_broccoli := doubled_daily_intake * 7
  let weekly_kale := 3
  let total_weekly_intake := weekly_intake_asparagus_broccoli + weekly_kale
  total_weekly_intake = 10 := 
by
  sorry

end james_vegetable_intake_l639_639931


namespace rectangle_area_l639_639624

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l639_639624


namespace product_equality_l639_639261

theorem product_equality : 
  (∏ n in finset.range 15, (n + 1) * (n + 3) / (n + 5)^2) = 113 / 4 := 
by
  sorry

end product_equality_l639_639261


namespace geometric_common_ratio_l639_639597

theorem geometric_common_ratio (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 5 * d)^2 = a1 * (a1 + 20 * d)) : 
  (a1 + 5 * d) / a1 = 3 :=
by
  sorry

end geometric_common_ratio_l639_639597


namespace lcm_at_least_ten_a1_l639_639343

theorem lcm_at_least_ten_a1 (a : ℕ → ℕ) (h_distinct: (strict_mono a)) (h_range: ∀ i : ℕ, 0 < i → i ≤ 10 → a i = i → 1 ≤ a 1) (h_lt: a 1 < a 2 ∧ a 2 < a 3 ∧ a 3 < a 4 ∧ a 4 < a 5 ∧ a 5 < a 6 ∧ a 6 < a 7 ∧ a 7 < a 8 ∧ a 8 < a 9 ∧ a 9 < a 10) : nat.lcm (finset.range 10).map a.val ≥ 10 * a 1 := 
by
  sorry

end lcm_at_least_ten_a1_l639_639343


namespace polynomial_evaluation_l639_639158

theorem polynomial_evaluation :
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 101^5 :=
by
  sorry

end polynomial_evaluation_l639_639158


namespace intersection_points_l639_639083

noncomputable def sides_set : set ℕ := {4, 5, 7, 9}

def condition_no_shared_vertex (sides_set : set ℕ) : Prop := 
  ∀ (polygon1 ∈ sides_set) (polygon2 ∈ sides_set),
  polygon1 ≠ polygon2 →
  ¬ ∃ (v : ℝ × ℝ), v ∈ (vertices polygon1) ∩ (vertices polygon2)

def condition_no_three_intersect (sides_set : set ℕ) : Prop := 
  ∀ (polygon1 polygon2 polygon3 ∈ sides_set),
  polygon1 ≠ polygon2 ∧ polygon2 ≠ polygon3 ∧ polygon1 ≠ polygon3 →
  ¬ ∃ (p : ℝ × ℝ), p ∈ (sides polygon1) ∩ (sides polygon2) ∩ (sides polygon3)

theorem intersection_points
  (h1 : condition_no_shared_vertex sides_set)
  (h2 : condition_no_three_intersect sides_set) :
  ∑ polygon1 ∈ sides_set, ∑ polygon2 ∈ sides_set, if polygon1 < polygon2 then 2 * polygon1 else 0 = 58 := 
sorry

end intersection_points_l639_639083


namespace hurricanes_valid_lineups_l639_639103

/-- Define the set of Hurricanes' players indexed from 1 to 15. -/
def players := Finset.range 15

/-- Anne is player 0, Rick is player 1, Sam is player 2, John is player 3. -/
def Anne := 0
def Rick := 1
def Sam := 2
def John := 3

/-- The number of valid starting lineups (of 5 players) for which Anne and Rick cannot both play together and Sam and John cannot both play together. -/
def valid_starting_lineups : ℕ :=
  let total_lineups := players.card.choose 5 in
  let invalid_AR := ((players.erase Anne).erase Rick).card.choose 4 in
  let invalid_SJ := ((players.erase Sam).erase John).card.choose 4 in
  let intersection_AR_SJ := ((players.erase Anne).erase Rick).erase Sam).erase John).card.choose 3 in
  total_lineups - invalid_AR - invalid_SJ + intersection_AR_SJ

theorem hurricanes_valid_lineups : valid_starting_lineups = 
    -- Final desired count, combining cases 
    sorry

end hurricanes_valid_lineups_l639_639103


namespace count_integers_between_200_and_250_with_increasing_digits_l639_639431

theorem count_integers_between_200_and_250_with_increasing_digits :
  ∃ n, n = 11 ∧ ∀ x, 200 ≤ x ∧ x ≤ 250 ∧ 
  (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k) → 
  n = ∑ x in {x | 200 ≤ x ∧ x ≤ 250 ∧ (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k)}, 1
:= 
sorry

end count_integers_between_200_and_250_with_increasing_digits_l639_639431


namespace acute_isosceles_triangle_inscribed_circle_l639_639235

variables {A B C D : Type} [EuclideanGeometry A B C D]

theorem acute_isosceles_triangle_inscribed_circle (h1 : is_acute A B C)
    (h2 : is_isosceles A B C) (h3 : inscribed_in_circle A B C)
    (h4 : are_tangents B C D) (h5 : ∠ ABC = ∠ ACB = 3 * ∠ D)
    (h6 : tan_from_circle_point B C D) :
    (∠ BAC = 5 * π / 11) :=
sorry

end acute_isosceles_triangle_inscribed_circle_l639_639235


namespace number_of_valid_routes_l639_639703

-- Given conditions
def initial_position := (0, 0)
def final_position := (5, 5)
def grid_size := 5
def valid_moves : set ((ℕ × ℕ) × (ℕ × ℕ)) :=
  { ((a, b), (a+1, b)) | 0 ≤ a ∧ a < grid_size ∧ 0 ≤ b ∧ b ≤ grid_size } ∪
  { ((a, b), (a, b+1)) | 0 ≤ a ∧ a ≤ grid_size ∧ 0 ≤ b ∧ b < grid_size } ∪
  { ((a, b), (a-1, b+1)) | 1 ≤ a ∧ a ≤ grid_size ∧ 0 ≤ b ∧ b < grid_size }

-- The proof problem statement
theorem number_of_valid_routes :
  number_of_routes initial_position final_position valid_moves = 1650 :=
sorry

end number_of_valid_routes_l639_639703


namespace probability_of_at_least_one_six_in_six_rolls_l639_639661

theorem probability_of_at_least_one_six_in_six_rolls :
  let p := 1 - (5 / 6)^6 in
  p = 1 - (5 / 6)^6 := sorry

end probability_of_at_least_one_six_in_six_rolls_l639_639661


namespace no_negatives_possible_l639_639885

theorem no_negatives_possible
  (a b c d : ℤ)
  (h : 4^a + 4^b = 5^c + 5^d + 1) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 :=
sorry

end no_negatives_possible_l639_639885


namespace car_speed_ratio_to_pedestrian_speed_l639_639720

noncomputable def ratio_of_speeds (length_pedestrian_car: ℝ) (length_bridge: ℝ) (speed_pedestrian: ℝ) (speed_car: ℝ) : ℝ :=
  speed_car / speed_pedestrian

theorem car_speed_ratio_to_pedestrian_speed
  (L : ℝ)  -- length of the bridge
  (vc vp : ℝ)  -- speeds of car and pedestrian respectively
  (h1 : (2 / 5) * L + (3 / 5) * vp = L)
  (h2 : (3 / 5) * L = vc * (L / (5 * vp))) :
  ratio_of_speeds L L vp vc = 5 :=
by
  sorry

end car_speed_ratio_to_pedestrian_speed_l639_639720


namespace tangent_line_equation_l639_639316

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem tangent_line_equation (x : ℝ) (hx : x ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2))
  (h_deriv : deriv f x = 1) : ∃ (y : ℝ), f 0 = -1 ∧ (y + 1 = x) → (x - y - 1 = 0) :=
by
  -- conditions and definitions used in the Lean statement
  have t : 1 = 1 := rfl
  existsi -1
  exact ⟨rfl, λh, by
    have hy : y = x - 1 := eq_sub_iff_add_eq.mpr h
    simp only [hy, sub_self]⟩

end tangent_line_equation_l639_639316


namespace positive_difference_largest_prime_factors_l639_639660

theorem positive_difference_largest_prime_factors :
  let p1 := 139
  let p2 := 29
  p1 - p2 = 110 := sorry

end positive_difference_largest_prime_factors_l639_639660


namespace each_student_practice_time_l639_639790

/-- Given that there are 5 students and only 2 can practice at a time, with a total practice time 
of 90 minutes and each student practices for an equal amount of time, prove that each student 
practices for 36 minutes. -/
theorem each_student_practice_time (total_students : ℕ) (pairs_per_time : ℕ) (total_minutes : ℕ)
  (equal_practice_time : ℕ) (h_total_students : total_students = 5) (h_pairs_per_time : pairs_per_time = 2)
  (h_total_minutes : total_minutes = 90) (h_equal_practice_time : ∀ t : ℕ, t = (total_minutes * pairs_per_time) / total_students) :
  equal_practice_time = 36 :=
by
  -- Conditions provided
  have h1 : total_students = 5 := h_total_students,
  have h2 : pairs_per_time = 2 := h_pairs_per_time,
  have h3 : total_minutes = 90 := h_total_minutes,
  -- Calculation using conditions
  have total_practice_time := total_minutes * pairs_per_time,
  have each_time := total_practice_time / total_students,
  have h4 : equal_practice_time = each_time := h_equal_practice_time each_time,
  -- Setting the required practice time as 36
  exact h4

end each_student_practice_time_l639_639790


namespace collinear_P_center_midEG_l639_639346

structure Circle (Point : Type) :=
(center : Point)
(radius : ℝ)

structure Quadrilateral (Point : Type) :=
(A B C D : Point)

variables {Point : Type} [MetricSpace Point] [NormedAddGroup Point] [NormedSpace ℝ Point]

def is_concyclic {Point : Type} [EuclideanGeometry Point] (a b c d : Point) : Prop :=
∃ (Ω : Circle Point), Ω.center = Circumcenter a b c d

def collinear (a b c : Point) : Prop :=
∃ (l : Line Point), a ∈ l ∧ b ∈ l ∧ c ∈ l

variables (A B C D P E F G H : Point)
variable (Γ : Circle Point)
variable [inscribed : InscribedCircle Γ (Quadrilateral.mk A B C D)]

def midpoint (a b : Point) : Point := (a + b) / 2

theorem collinear_P_center_midEG :
  collinear P Γ.center (midpoint E G) := sorry

end collinear_P_center_midEG_l639_639346


namespace part1_part2_l639_639809

-- Define the conditions
variables {R : Type*} [linear_ordered_field R]
variables (f : R → R)
variable (a : R)

-- Conditions from the problem
def condition1 := ∀ x y, f(x) + f(y) = f(x + y) + 2
def condition2 := ∀ x, (0 < x) → (2 < f(x))

-- Part 1: Prove f(x) is increasing
theorem part1 (cond1 : condition1 f) (cond2 : condition2 f) : ∀ x y, x < y → f(x) < f(y) :=
by sorry

-- Part 2: Solve the inequality
theorem part2 (cond1 : condition1 f) (cond2 : condition2 f) (f_3 : f 3 = 5) 
  (ineq : f(a^2 - 2*a - 2) < 3) : -1 < a ∧ a < 3 :=
by sorry

end part1_part2_l639_639809


namespace count_integers_between_200_250_l639_639441

theorem count_integers_between_200_250 :
  {n : ℕ // 200 ≤ n ∧ n < 250 ∧ 
            (let d2 := (n / 10) % 10, d3 := n % 10 in
             (n / 100 = 2) ∧ (d2 ≠ d3) ∧ (2 < d2) ∧ (d2 < d3)
            )}.to_finset.card = 11 :=
by
  -- Start the proof process here
  sorry

end count_integers_between_200_250_l639_639441


namespace incorrect_statement_D_l639_639839

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + x
else -(x^2 + x)

theorem incorrect_statement_D : ¬(∀ x : ℝ, x ≤ 0 → f x = x^2 + x) :=
by
  sorry

end incorrect_statement_D_l639_639839


namespace probability_volleyball_is_one_third_l639_639631

-- Define the total number of test items
def total_test_items : ℕ := 3

-- Define the number of favorable outcomes for hitting the wall with a volleyball
def favorable_outcomes_volleyball : ℕ := 1

-- Define the probability calculation
def probability_hitting_wall_with_volleyball : ℚ :=
  favorable_outcomes_volleyball / total_test_items

-- Prove the probability is 1/3
theorem probability_volleyball_is_one_third :
  probability_hitting_wall_with_volleyball = 1 / 3 := 
sorry

end probability_volleyball_is_one_third_l639_639631


namespace two_lines_perpendicular_to_same_line_relationship_l639_639472

theorem two_lines_perpendicular_to_same_line_relationship (L1 L2 L3 : Type) 
  [line L1] [line L2] [line L3]
  (h1 : L1 ⊥ L3) (h2 : L2 ⊥ L3) :
  (L1 ∥ L2) ∨ (∃ P : Type, [point P] (P ∈ L1) ∧ (P ∈ L2)) ∨ (skew L1 L2) :=
sorry

end two_lines_perpendicular_to_same_line_relationship_l639_639472


namespace remove_6_maximizes_probability_l639_639149

def original_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

-- Define what it means to maximize the probability of pairs summing to 12
def maximize_probability (l : List Int) : Prop :=
  ∀ x y, x ≠ y → x ∈ l → y ∈ l → x + y = 12

-- Prove that removing 6 maximizes the probability that the sum of the two chosen numbers is 12
theorem remove_6_maximizes_probability :
  maximize_probability (original_list.erase 6) :=
sorry

end remove_6_maximizes_probability_l639_639149


namespace frank_original_money_l639_639337

theorem frank_original_money (X : ℝ) :
  (X - (1 / 5) * X - (1 / 4) * (X - (1 / 5) * X) = 360) → (X = 600) :=
by
  sorry

end frank_original_money_l639_639337


namespace count_integer_solutions_on_circle_and_line_l639_639793

theorem count_integer_solutions_on_circle_and_line : 
  (∀ (a b : ℝ), (∃ (x y : ℤ ), a * x + b * y = 3 ∧ x^2 + y^2 = 29) → (a, b) ∈ {(a, b) | a ∈ ℝ ∧ b ∈ ℝ}) ↔ 36 :=
begin
  sorry
end

end count_integer_solutions_on_circle_and_line_l639_639793


namespace rotated_parabola_eq_l639_639452

theorem rotated_parabola_eq (x : ℝ) :
  let original_parabola := λ x : ℝ, (1/2) * x^2 + 1
  let rotated_parabola := λ x : ℝ, -(1/2) * x^2 + 1
  (∀ x', original_parabola x' = (1/2) * x'^2 + 1) →
  rotated_parabola x = -(1/2) * x^2 + 1 :=
by
  intro h
  sorry

end rotated_parabola_eq_l639_639452


namespace problem1_problem2_l639_639383

def f (x a : ℝ) := x^2 + 2 * a * x + 2

theorem problem1 (a : ℝ) (h : a = -1) : 
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a ≤ 37) ∧ (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x a = 37) ∧
  (∀ x : ℝ, -5 ≤ x ∧ x ≤ 5 → f x a ≥ 1) ∧ (∃ x : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ f x a = 1) :=
by
  sorry

theorem problem2 (a : ℝ) : 
  (∀ x1 x2 : ℝ, -5 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 5 → f x1 a > f x2 a) ↔ a ≤ -5 :=
by
  sorry

end problem1_problem2_l639_639383


namespace compare_a_b_c_l639_639064

noncomputable def a := Real.sin (Real.pi / 5)
noncomputable def b := Real.logb (Real.sqrt 2) (Real.sqrt 3)
noncomputable def c := (1 / 4)^(2 / 3)

theorem compare_a_b_c : c < a ∧ a < b := by
  sorry

end compare_a_b_c_l639_639064


namespace annika_hiking_rate_l639_639253

theorem annika_hiking_rate :
  ∀ (d1 d2 : ℝ) (t : ℝ), d1 = 2.75 → d2 = 3.625 → t = 45 → 
  (d2 - d1) * 2 + d1 = 4.5 → t / 4.5 = 10 :=
by
  intros d1 d2 t h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end annika_hiking_rate_l639_639253


namespace prod_sum_leq_four_l639_639537

theorem prod_sum_leq_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 4) :
  ab + bc + cd + da ≤ 4 :=
sorry

end prod_sum_leq_four_l639_639537


namespace area_proof_l639_639609

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l639_639609


namespace max_a_plus_b_max_a_plus_b_is_achieved_l639_639507

theorem max_a_plus_b (a b : ℤ) (h : a * b = -72) : a + b ≤ 71 := 
sorry

theorem max_a_plus_b_is_achieved : ∃ a b : ℤ, a * b = -72 ∧ a + b = 71 :=
by {
  use [-1, 72],
  split,
  {
    norm_num,
  },
  {
    refl, 
  },
}

end max_a_plus_b_max_a_plus_b_is_achieved_l639_639507


namespace find_special_n_l639_639677

def is_perfect (n : ℕ) : Prop :=
  (∑ i in (list.divisors n).nodup, i) = 2 * n

theorem find_special_n :
  ∀ n : ℕ, (is_perfect (n-1) ∧ is_perfect (n * (n + 1) / 2)) → n = 7 :=
begin
  intros n h,
  sorry
end

end find_special_n_l639_639677


namespace area_proof_l639_639608

-- Define the problem conditions
variables (l w : ℕ)
def length_is_four_times_width : Prop := l = 4 * w
def perimeter_is_200 : Prop := 2 * l + 2 * w = 200

-- Define the target to prove
def area_of_rectangle : Prop := (l * w = 1600)


-- Lean 4 statement to prove the area given the conditions
theorem area_proof (h1 : length_is_four_times_width l w) (h2 : perimeter_is_200 l w) : area_of_rectangle l w := 
  sorry

end area_proof_l639_639608


namespace sum_of_roots_eq_h_over_4_l639_639519

theorem sum_of_roots_eq_h_over_4 (x1 x2 h b : ℝ) (h_ne : x1 ≠ x2)
  (hx1 : 4 * x1 ^ 2 - h * x1 = b) (hx2 : 4 * x2 ^ 2 - h * x2 = b) : x1 + x2 = h / 4 :=
sorry

end sum_of_roots_eq_h_over_4_l639_639519


namespace electricity_fee_l639_639457

theorem electricity_fee (a b : ℝ) : 
  let base_usage := 100
  let additional_usage := 160 - base_usage
  let base_cost := base_usage * a
  let additional_cost := additional_usage * b
  base_cost + additional_cost = 100 * a + 60 * b :=
by
  sorry

end electricity_fee_l639_639457


namespace fill_tank_in_12_minutes_l639_639670

theorem fill_tank_in_12_minutes (rate1 rate2 rate_out : ℝ) 
  (h1 : rate1 = 1 / 18) (h2 : rate2 = 1 / 20) (h_out : rate_out = 1 / 45) : 
  12 = 1 / (rate1 + rate2 - rate_out) :=
by
  -- sorry will be replaced with the actual proof.
  sorry

end fill_tank_in_12_minutes_l639_639670


namespace range_of_a_l639_639385

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (-x)

theorem range_of_a (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) (h_ineq : f a (-2) > f a (-3)) : 0 < a ∧ a < 1 :=
by {
  sorry
}

end range_of_a_l639_639385


namespace largest_divided_sub_l639_639061

def binary_strings (n : ℕ) : Finset (Fin n → Bool) := 
  Finset.univ

def num_blocks (s : Fin n → Bool) : ℕ := 
  let change_points := 
    Finset.image (λ i, (s i) ≠ (s ⟨i+1 % n, sorry⟩)) (Finset.range (n-1))
  Finset.card change_points + 1

def twist (s : Fin n → Bool) : Fin n → Bool := 
  let b := num_blocks s
  λ i, if i.val = b then !s i else s i

def is_descendant (a b : Fin n → Bool) : Prop := 
  ∃ k, (Function.iterate twist k b = a)

def is_divided (S : Finset (Fin n → Bool)) : Prop := 
  ∀ a b ∈ S, ¬(is_descendant a b ∨ is_descendant b a)

theorem largest_divided_sub (n : ℕ) (hn : n > 0) 
  : ∃ S : Finset (Fin n → Bool), is_divided S ∧ S.card = 2^(n-2) := sorry

end largest_divided_sub_l639_639061


namespace arithmetic_sequence_a7_l639_639376

/-- Given conditions:
  - The sum of the first 10 terms of an arithmetic sequence is 165.
  - The 4th term in this sequence is 12.
  Prove that the 7th term is 21. -/
theorem arithmetic_sequence_a7 (S10 : ℕ → ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : S10 10 165) (h2 : a 4 = 12) : a 7 = 21 :=
sorry

end arithmetic_sequence_a7_l639_639376


namespace find_ellipse_slope_l639_639372

variables (a b x y : ℝ)
def P := (2 : ℝ, -1 : ℝ)

noncomputable def ellipse_eq := 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧ ((x^2 / a^2) + (y^2 / b^2) = 1) ∧ ((2^2 / a^2) + ((-1)^2 / b^2) = 1))

noncomputable def ellipse_eccentricity := 
  (∃ c : ℝ, c = sqrt(a^2 - b^2) ∧ c / a = sqrt(3) / 2)

theorem find_ellipse_slope (h : ellipse_eq ∧ ellipse_eccentricity ) :
  ((x^2 / 8) + (y^2 / 2) = 1) ∧ (∃ k : ℝ, k = 1 / 2) :=
sorry

end find_ellipse_slope_l639_639372


namespace joaozinho_card_mariazinha_card_pedrinho_error_l639_639183

-- Define the card transformation function
def transform_card (number : ℕ) (color_adjustment : ℕ) : ℕ :=
  (number * 2 + 3) * 5 + color_adjustment

-- The proof problems
theorem joaozinho_card : transform_card 3 4 = 49 :=
by
  sorry

theorem mariazinha_card : ∃ number, ∃ color_adjustment, transform_card number color_adjustment = 76 :=
by
  sorry

theorem pedrinho_error : ∀ number color_adjustment, ¬ transform_card number color_adjustment = 61 :=
by
  sorry

end joaozinho_card_mariazinha_card_pedrinho_error_l639_639183


namespace angle_EMZ_eq_angle_EYF_l639_639275

theorem angle_EMZ_eq_angle_EYF (A B C D E F M X Y Z : Type) 
  [regular_octahedron A B C D E F]
  [middle_cross_section A B C D]
  [is_midpoint M A B C D]
  [circumscribed_sphere k A B C D E F]
  [point_on_face X A B F]
  (hEX_k : EX ∩ k = {E, Z})
  (hEX_plane : EX ∩ plane A B C D = {Y})
  : angle E M Z = angle E Y F :=
sorry

end angle_EMZ_eq_angle_EYF_l639_639275


namespace shuttle_speeds_l639_639206

def speed_at_altitude (speed_per_sec : ℕ) : ℕ :=
  speed_per_sec * 3600

theorem shuttle_speeds (speed_300 speed_800 avg_speed : ℕ) :
  speed_at_altitude 7 = 25200 ∧ 
  speed_at_altitude 6 = 21600 ∧ 
  avg_speed = (25200 + 21600) / 2 ∧ 
  avg_speed = 23400 := 
by
  sorry

end shuttle_speeds_l639_639206


namespace at_least_two_pass_theory_all_three_pass_entire_course_l639_639641

-- Define probabilities of passing theory assessment
def P_A1 := 0.9
def P_A2 := 0.8
def P_A3 := 0.7

-- Define probabilities of passing experimental assessment
def P_B1 := 0.8
def P_B2 := 0.7
def P_B3 := 0.9

-- Complement probabilities for theory assessment
def P_not_A3 := 1 - P_A3
def P_not_A2 := 1 - P_A2
def P_not_A1 := 1 - P_A1

/- Question 1: Prove the probability that at least two among A, B, and C pass the theory assessment is 0.902 -/
theorem at_least_two_pass_theory : 
  (P_A1 * P_A2 * P_not_A3 + P_A1 * P_not_A2 * P_A3 + P_not_A1 * P_A2 * P_A3 + P_A1 * P_A2 * P_A3) = 0.902 :=
by sorry

/- Question 2: Prove the probability that all three of them pass the entire course assessment is 0.254 -/
theorem all_three_pass_entire_course : 
  (P_A1 * P_B1 * P_A2 * P_B2 * P_A3 * P_B3) ≈ 0.254 :=
by sorry

end at_least_two_pass_theory_all_three_pass_entire_course_l639_639641


namespace budget_given_for_year_l639_639710

theorem budget_given_for_year
  (cost1 cost2 last_year_remaining budget_remaining : ℕ)
  (h1 : cost1 = 13)
  (h2 : cost2 = 24)
  (h3 : last_year_remaining = 6)
  (h4 : budget_remaining = 19) :
  let total_spent := cost1 + cost2 in
  let total_available := total_spent + budget_remaining in
  let budget_for_this_year := total_available - last_year_remaining in
  budget_for_this_year = 50 := 
by {
  sorry
}

end budget_given_for_year_l639_639710


namespace student_average_correct_l639_639276

theorem student_average_correct (w x y z : ℤ) (h : w < x ∧ x < y ∧ y < z) :
  (let A := (w + x + y + z) / 4 in
   let B := ((w + x) / 2 + (y + z) / 2) / 2 in
   A = B) :=
by
  sorry

end student_average_correct_l639_639276


namespace rectangle_area_l639_639614

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l639_639614


namespace prob_qualified_prod_by_A_l639_639479

variable (p_A : ℝ) (p_not_A : ℝ) (p_B_given_A : ℝ) (p_B_given_not_A : ℝ)
variable (p_A_and_B : ℝ)

axiom prob_A : p_A = 0.7
axiom prob_not_A : p_not_A = 0.3
axiom prob_B_given_A : p_B_given_A = 0.95
axiom prob_B_given_not_A : p_B_given_not_A = 0.8

theorem prob_qualified_prod_by_A :
  p_A_and_B = p_A * p_B_given_A :=
  sorry

end prob_qualified_prod_by_A_l639_639479


namespace digit_d_is_six_l639_639332

theorem digit_d_is_six (d : ℕ) (h_even : d % 2 = 0) (h_digits_sum : 7 + 4 + 8 + 2 + d % 9 = 0) : d = 6 :=
by 
  sorry

end digit_d_is_six_l639_639332


namespace count_integers_between_200_250_l639_639437

theorem count_integers_between_200_250 :
  {n : ℕ // 200 ≤ n ∧ n < 250 ∧ 
            (let d2 := (n / 10) % 10, d3 := n % 10 in
             (n / 100 = 2) ∧ (d2 ≠ d3) ∧ (2 < d2) ∧ (d2 < d3)
            )}.to_finset.card = 11 :=
by
  -- Start the proof process here
  sorry

end count_integers_between_200_250_l639_639437


namespace casper_initial_candies_l639_639074

theorem casper_initial_candies 
  (x : ℚ)
  (h1 : ∃ y : ℚ, y = x - (1/4) * x - 3) 
  (h2 : ∃ z : ℚ, z = y - (1/5) * y - 5) 
  (h3 : z - 10 = 10) : x = 224 / 3 :=
by
  sorry

end casper_initial_candies_l639_639074


namespace fraction_females_local_handball_league_l639_639989

theorem fraction_females_local_handball_league
  (last_year_males : ℕ)
  (last_year_females : ℕ)
  (this_year_participants : ℕ)
  : 
  last_year_males = 25 →
  this_year_participants = 30 + 1.20 * last_year_females →
  this_year_participants = 28 + 1.3 * last_year_females →
  (26 / this_year_participants) = 13 / 27 :=
by
  sorry

end fraction_females_local_handball_league_l639_639989


namespace find_number_of_children_l639_639668

theorem find_number_of_children (C B : ℕ) (H1 : B = 2 * C) (H2 : B = 4 * (C - 360)) : C = 720 := 
by
  sorry

end find_number_of_children_l639_639668


namespace james_vegetable_consumption_l639_639930

theorem james_vegetable_consumption : 
  (1/4 + 1/4) * 2 * 7 + 3 = 10 := 
by
  sorry

end james_vegetable_consumption_l639_639930


namespace vitya_wins_with_optimal_play_l639_639939

-- Definitions for the infinite grid and nodes on the grid
def infinite_grid := ℤ × ℤ
def is_valid_move (nodes : set infinite_grid) (new_node : infinite_grid) : Prop :=
  ∀ n ∈ nodes, ∃ x y : infinite_grid, -- condition to check nodes form vertices of a convex polygon
  new_node ≠ x ∧ x ≠ y ∧ y ≠ new_node ∧
  convex_polygon (insert new_node nodes)

-- Main statement: Vitya wins the game with optimal play
theorem vitya_wins_with_optimal_play :
  ∃ O : infinite_grid, ∀ moves : list infinite_grid,
  ∀ m ∈ moves,
  (is_valid_move (initial_move :: take_moves moves) m) ∧ 
  (∃ k : ℕ, moves.length = 2 * k + 1 → ¬is_valid_move (initial_move :: take_moves moves) m) →
  (∃ v : infinite_grid, is_valid_move (initial_move :: take_moves moves) v) :=
sorry

end vitya_wins_with_optimal_play_l639_639939


namespace find_k_in_isosceles_triangle_l639_639225

theorem find_k_in_isosceles_triangle 
  (A B C D : Type)
  (h_triangle : triangle ABC)
  (h_acute : acute ABC)
  (h_isosceles : isosceles ABC)
  (h_circumscribed : circumscribed ABC)
  (h_tangents : tangents B C D)
  (h_angle_relation : ∠ABC = ∠ACB ∧ ∠ABC = 3 * ∠D)
  : ∠BAC = (5 * π / 11) := 
by 
  sorry

end find_k_in_isosceles_triangle_l639_639225


namespace find_k_l639_639218

-- Definitions based on conditions in step a)
def acute_isosceles_triangle_inscribed (A B C : Type) : Prop := sorry -- Formal definition of the triangle being acute isosceles and inscribed in a circle
def tangents_meeting_at_point (A B C D : Type) : Prop := sorry -- Formal definition of tangents through B and C meeting at D
def angle_relation (ABC D : Type) (theta : ℝ) : Prop := 3 * theta = sorry -- Formal definition of \(\angle ABC = \angle ACB = 3 \angle D\)
def angle_BAC (k : ℝ) (theta : ℝ) : Prop := theta = k * real.pi -- Formal definition of \(\angle BAC = k \pi\)

-- Theorem statement for our proof problem
theorem find_k
  (A B C D : Type)
  (h1 : acute_isosceles_triangle_inscribed A B C)
  (h2 : tangents_meeting_at_point A B C D)
  (theta : ℝ)
  (h3 : angle_relation ABC D theta)
  (k : ℝ)
  (h4 : angle_BAC k theta) :
  k = 1 / 13 := by
  sorry

end find_k_l639_639218


namespace original_polygon_sides_l639_639716

theorem original_polygon_sides {n : ℕ} 
  (h : (n - 2) * 180 = 1620) : n = 10 ∨ n = 11 ∨ n = 12 :=
sorry

end original_polygon_sides_l639_639716


namespace conditional_probability_A_given_B_l639_639139

noncomputable def conditional_probability (A B : Set (Fin 6 × Fin 6 × Fin 6)) : ℚ :=
  (A ∩ B).card / B.card

def event_A : Set (Fin 6 × Fin 6 × Fin 6) :=
  { (a, b, c) | a ≠ b ∧ b ≠ c ∧ a ≠ c }

def event_B : Set (Fin 6 × Fin 6 × Fin 6) :=
  { (a, b, c) | a = 3 ∨ b = 3 ∨ c = 3 }

theorem conditional_probability_A_given_B :
  conditional_probability event_A event_B = 60 / 91 :=
by
  sorry

end conditional_probability_A_given_B_l639_639139


namespace dealership_sedan_sales_l639_639694

-- Definitions based on conditions:
def sports_cars_ratio : ℕ := 3
def sedans_ratio : ℕ := 5
def anticipated_sports_cars : ℕ := 36

-- Proof problem statement
theorem dealership_sedan_sales :
    (anticipated_sports_cars * sedans_ratio) / sports_cars_ratio = 60 :=
by
  -- Proof goes here
  sorry

end dealership_sedan_sales_l639_639694


namespace remaining_amount_to_be_paid_l639_639887

theorem remaining_amount_to_be_paid (p : ℝ) (deposit : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) (final_payment : ℝ) :
  deposit = 80 ∧ tax_rate = 0.07 ∧ discount_rate = 0.05 ∧ deposit = 0.1 * p ∧ 
  final_payment = (p - (discount_rate * p)) * (1 + tax_rate) - deposit → 
  final_payment = 733.20 :=
by
  sorry

end remaining_amount_to_be_paid_l639_639887


namespace vector_triangle_sum_zero_eq_zero_l639_639926

variables {V : Type*} [add_group V] [module ℝ V]
variables (A B C : V)

def vector_triangle_sum_zero (A B C : V) : V :=
  (B - A) + (C - B) + (A - C)

theorem vector_triangle_sum_zero_eq_zero (A B C : V) :
  vector_triangle_sum_zero A B C = 0 := by
  sorry

end vector_triangle_sum_zero_eq_zero_l639_639926


namespace remainder_of_h_x10_div_h_x_l639_639513

noncomputable def h (x : ℤ) : ℤ := x^5 - x^4 + x^3 - x^2 + x - 1

theorem remainder_of_h_x10_div_h_x (x : ℤ) : (h (x ^ 10)) % (h x) = -6 :=
by
  sorry

end remainder_of_h_x10_div_h_x_l639_639513


namespace hyperbola_eccentricity_l639_639825

-- Define the properties of the parabola and hyperbola
structure Parabola (ρ : ℝ) :=
  (focus : ℝ × ℝ := (ρ / 2, 0))
  (directrix : ℝ := -ρ / 2)

structure Hyperbola (a b : ℝ) :=
  (asymptote : ℝ → ℝ := fun x => (b / a * x))

noncomputable def find_eccentricity (a b ρ : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_ρ_pos : ρ > 0) : ℝ :=
  -- Since the asymptote is y = (b/a)x, we can express coordinates of point A
  let A := (ρ / 2, ρ * b / (2 * a))
  -- The condition |AF| = ρ is equivalent to b/a = 1
  let h_ba_eq_one : b / a = 1 := 
    by 
      sorry 
  -- Finally, the eccentricity e of the hyperbola C2 is sqrt(1 + (b/a)^2)
  sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity (a b ρ : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_ρ_pos : ρ > 0) :
  find_eccentricity a b ρ h_a_pos h_b_pos h_ρ_pos = sqrt 2 :=
  sorry

end hyperbola_eccentricity_l639_639825


namespace find_number_l639_639162

noncomputable def N : ℕ :=
  76

theorem find_number :
  (N % 13 = 11) ∧ (N % 17 = 9) :=
by
  -- These are the conditions translated to Lean 4, as stated:
  have h1 : N % 13 = 11 := by sorry
  have h2 : N % 17 = 9 := by sorry
  exact ⟨h1, h2⟩

end find_number_l639_639162


namespace problem_sequences_l639_639331

open Real

-- Definition of sequences and properties
def seq (a : ℕ+ → ℝ) := ∀ n : ℕ+, a n > 0

-- Definitions from the problem
def T (a : ℕ+ → ℝ) (n : ℕ+) : ℝ := (finset.range n).sum (λ k, a k * a (k + 1))

def b (T : ℕ+ → ℝ) (n : ℕ) : ℝ := 
if n = 1 then T 2 - 2 * T 1
else T (n + 1) + T (n - 1) - 2 * T n

-- Main Theorem as described in step c)
theorem problem_sequences (a : ℕ+ → ℝ) :
(seq a) →
(∃ k : ℕ+, T a k = 2017) = false ∧
((a 1 = 3) ∧ (∀ n : ℕ+, T n = 6^n - 1) → 
  ∀ n : ℕ *,
    a n = if (2 ∣ n) 
    then (5/3) * (6: ℝ)^((n - 2) / 2)
    else (3) * (6: ℝ)^((n - 1) / 2)) ∧
(∀ a : ℕ+ → ℝ, (∀ n : ℕ+,
(a (n + 1) - a n = a (n + 2) - a (n + 1)) ∧
(b T n = b T (n + 1) - b T n = 3))) := 
begin
  sorry
end

end problem_sequences_l639_639331


namespace area_of_AMK_l639_639076

open_locale big_operators

-- Definitions of the points and ratios
variables (A B C M K : Type*)
variables [triangle ABC]
variables (area_ABC : ℝ) (ratio_AM_MB : ℝ) (ratio_AK_KC : ℝ)
variables (area_AMK : ℝ)

-- Conditions given in the problem
axiom triangle_area : area_ABC = 36
axiom ratio_condition_1 : ratio_AM_MB = 1 / 3
axiom ratio_condition_2 : ratio_AK_KC = 2 / 1

-- Goal to prove
theorem area_of_AMK :
  area_AMK = 6 :=
by
  sorry

end area_of_AMK_l639_639076


namespace proof_q_is_true_l639_639453

variable (p q : Prop)

-- Assuming the conditions
axiom h1 : p ∨ q   -- p or q is true
axiom h2 : ¬ p     -- not p is true

-- Theorem statement to prove q is true
theorem proof_q_is_true : q :=
by
  sorry

end proof_q_is_true_l639_639453


namespace required_integers_l639_639284

theorem required_integers (n : ℕ) (h : n ≥ 2) :
  (∀ a b : ℕ, Nat.coprime a n ∧ Nat.coprime b n → (a ≡ b [MOD n] ↔ a * b ≡ 1 [MOD n])) ↔ 
  (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 12 ∨ n = 24) :=
by
  intro h1
  sorry

end required_integers_l639_639284


namespace students_count_geometry_history_science_l639_639312

noncomputable def number_of_students (geometry_only history_only science_only 
                                      geometry_and_history geometry_and_science : ℕ) : ℕ :=
  geometry_only + history_only + science_only

theorem students_count_geometry_history_science (geometry_total history_only science_only 
                                                 geometry_and_history geometry_and_science : ℕ) :
  geometry_total = 30 →
  geometry_and_history = 15 →
  history_only = 15 →
  geometry_and_science = 8 →
  science_only = 10 →
  number_of_students (geometry_total - geometry_and_history - geometry_and_science)
                     history_only
                     science_only = 32 :=
by
  sorry

end students_count_geometry_history_science_l639_639312


namespace brad_speed_l639_639548

theorem brad_speed (d_total : ℕ) (v_Maxwell : ℕ) (d_Maxwell : ℕ) (v_Brad : ℕ) : 
  d_total = 40 ∧ v_Maxwell = 3 ∧ d_Maxwell = 15 → v_Brad = 5 := 
by 
  -- Assume the given conditions
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h3,

  -- Derive the time t
  let t := d_Maxwell / v_Maxwell,

  -- Show that they meet after 5 hours
  have ht : t = 5,
  {
    rw h3, rw h2, 
    norm_num,
  },

  -- Express the distance Brad travels
  let d_Brad := d_total - d_Maxwell,

  -- Show that v_Brad * t = d_Brad
  have hb : v_Brad * t = d_Brad,
  {
    rw ht,  -- substitute t = 5
    rw h1, rw h3,
    norm_num,
  },

  -- Solving for v_Brad
  have v_Brad_correct : v_Brad = 5,
  {
    rw hb,
    norm_num,
  },

  exact v_Brad_correct,
  
  sorry

end brad_speed_l639_639548


namespace apbc_quadrilateral_pa_plus_pb_eq_2pd_l639_639521

variables {A P B C D: Type} 
variables [cyclic_quadrilateral A P B C]
variables [segment_eq AC BC]
variables [perpendicular_foot C P B D]

theorem apbc_quadrilateral_pa_plus_pb_eq_2pd :
  PA + PB = 2 * PD :=
sorry

end apbc_quadrilateral_pa_plus_pb_eq_2pd_l639_639521


namespace points_X_on_line_l639_639046

variables {α : Type*} [EuclideanSpace α]

theorem points_X_on_line
  (O A B C D X : α)
  (omega1 omega2 : Circle α)
  (h1 : omega1 ∩ omega2 = {O})
  (omega : Circle α)
  (center_O : omega.center = O)
  (h2 : A ∈ omega ∧ B ∈ omega ∧ A ∈ omega1 ∧ B ∈ omega1)
  (h3 : C ∈ omega ∧ D ∈ omega ∧ C ∈ omega2 ∧ D ∈ omega2)
  (line_AC : is_line AC)
  (line_BD : is_line BD)
  (X_intersect : X ∈ line_AC ∧ X ∈ line_BD) :
  ∃ l : Line α, ∀ X', X' ∈ line_AC ∧ X' ∈ line_BD → X' ∈ l :=
sorry

end points_X_on_line_l639_639046


namespace distance_covered_downstream_l639_639185

noncomputable def speed_in_still_water := 16 -- km/hr
noncomputable def speed_of_stream := 5 -- km/hr
noncomputable def time_taken := 5 -- hours
noncomputable def effective_speed_downstream := speed_in_still_water + speed_of_stream -- km/hr

theorem distance_covered_downstream :
  (effective_speed_downstream * time_taken = 105) :=
by
  sorry

end distance_covered_downstream_l639_639185


namespace capital_growth_rate_l639_639711

theorem capital_growth_rate
  (loan_amount : ℝ) (interest_rate : ℝ) (repayment_period : ℝ) (surplus : ℝ) (growth_rate : ℝ) :
  loan_amount = 2000000 ∧ interest_rate = 0.08 ∧ repayment_period = 2 ∧ surplus = 720000 ∧
  (loan_amount * (1 + growth_rate)^repayment_period = loan_amount * (1 + interest_rate) + surplus) →
  growth_rate = 0.2 :=
by
  sorry

end capital_growth_rate_l639_639711


namespace BD_plus_DC_eq_AC_l639_639496

-- Definitions
variable {A B C D : Type*} [EuclideanGeometry A B C D]

-- Given Conditions
axiom acute_triangle (ABC : triangle) : is_acute ABC
axiom point_D_on_parallel_line (AC : line) (B D : point) : on_line D (parallel_line_from B AC)
axiom angle_BDC (BD DC : line) (angle_BAC : angle) (α : real) : angle_measure BDC = 2 * angle_measure angle_BAC
axiom convex_quadrilateral (ABDC : quadrilateral) : is_convex ABDC

-- To Prove
theorem BD_plus_DC_eq_AC (A B C D : point) (AC : line) (BD DC : line) (ABC : triangle) (ABDC : quadrilateral)
  (h1 : is_acute ABC) (h2 : on_line D (parallel_line_from B AC)) 
  (h3 : angle_measure BDC = 2 * angle_measure (angle A B C))
  (h4 : is_convex ABDC) : 
  length (line_segment B D) + length (line_segment D C) = length (line_segment A C) := 
sorry

end BD_plus_DC_eq_AC_l639_639496


namespace apples_to_cucumbers_l639_639894

-- Definitions based on conditions
def apples := ℕ
def bananas := ℕ
def cucumbers := ℕ

axiom apples_cost_bananas (a b : apples) (c d : bananas) : 12 * a = 6 * b
axiom bananas_cost_cucumbers (e f : bananas) (g h : cucumbers) : 3 * e = 4 * g

-- Theorem statement
theorem apples_to_cucumbers {a b : apples} {e f : bananas} {g h : cucumbers} 
  (hb: apples_cost_bananas a b) 
  (hc: bananas_cost_cucumbers e g) 
  : 24 * a = 16 * g :=
sorry

end apples_to_cucumbers_l639_639894


namespace tan_identity_given_condition_l639_639364

variable (α : Real)

theorem tan_identity_given_condition :
  (Real.tan α + 1 / Real.tan α = 9 / 4) →
  (Real.tan α ^ 2 + 1 / (Real.sin α * Real.cos α) + 1 / Real.tan α ^ 2 = 85 / 16) := 
by
  sorry

end tan_identity_given_condition_l639_639364


namespace aquarium_surface_area_l639_639167

theorem aquarium_surface_area : 
  ∀ (side_length : ℕ), side_length = 20 → (6 * side_length * side_length = 2400) :=
by
  intros side_length h
  rw h
  norm_num

end aquarium_surface_area_l639_639167


namespace acute_isosceles_triangle_inscribed_circle_l639_639236

variables {A B C D : Type} [EuclideanGeometry A B C D]

theorem acute_isosceles_triangle_inscribed_circle (h1 : is_acute A B C)
    (h2 : is_isosceles A B C) (h3 : inscribed_in_circle A B C)
    (h4 : are_tangents B C D) (h5 : ∠ ABC = ∠ ACB = 3 * ∠ D)
    (h6 : tan_from_circle_point B C D) :
    (∠ BAC = 5 * π / 11) :=
sorry

end acute_isosceles_triangle_inscribed_circle_l639_639236


namespace solve_poly_eq_l639_639776

-- Definitions and conditions
def poly_eq_zero (x : ℂ) : Prop := x^4 - 81 = 0

-- Theorem statement
theorem solve_poly_eq (x : ℂ) : poly_eq_zero x ↔ (x = 3 ∨ x = -3 ∨ x = 3 * complex.I ∨ x = -3 * complex.I) :=
by
  sorry

end solve_poly_eq_l639_639776


namespace M_subset_P_l639_639045

def M := {x : ℕ | ∃ a : ℕ, 0 < a ∧ x = a^2 + 1}
def P := {y : ℕ | ∃ b : ℕ, 0 < b ∧ y = b^2 - 4*b + 5}

theorem M_subset_P : M ⊂ P :=
by
  sorry

end M_subset_P_l639_639045


namespace tangent_lines_through_pointM_sum_of_slopes_is_constant_l639_639907

-- Definitions based on conditions given in the problem
def circle (x y : ℝ) := (x - 1)^2 + y^2 = 4
def pointM := (3, -3 : ℝ)
def pointP := (3, 0 : ℝ)

-- Definition of a line given a slope m and a point (x₀, y₀)
def line_through (m x₀ y₀ x y : ℝ) := y = m * (x - x₀) + y₀

-- Definition of tangency condition using the distance formula
def tangent_line (m : ℝ) (x₀ y₀ r : ℝ) :=
  abs (m * x₀ - y₀) / sqrt (1 + m^2) = r

-- The first part to prove the equations of the tangent lines
theorem tangent_lines_through_pointM :
  ∃ m, 
  (line_through m 3 (-3) 3 0 = true ∨ (m = -5/12 ∧ line_through (-5/12) 3 (-3) 5 0 = true))
  ∧ (tangent_line m (1) 0 2 = true) :=
sorry

-- The second part to prove the sum of slopes of lines PA and PB
theorem sum_of_slopes_is_constant (A B : ℝ × ℝ) (hA : circle A.1 A.2) (hB : circle B.1 B.2)
  (hPA : A ≠ P) (hPB : B ≠ P) (hAM : ∃ k : ℝ, line_through k 3 (-3) A.1 A.2 = true)
  (hBM : ∃ k : ℝ, line_through k 3 (-3) B.1 B.2 = true) :
  ∃ k, 
  (A.x + B.x = 2 * (3 * k^2 + 3 * k + 1) / (1 + k^2)) ∧
  (A.x * B.x = (9 * (k + 1)^2 - 3) / (1 + k^2)) ∧
  (2 * k - (3 * ((1 / (A.x - 3)) + (1 / (B.x - 3)))) = 4/3) :=
sorry

end tangent_lines_through_pointM_sum_of_slopes_is_constant_l639_639907


namespace isosceles_trapezoid_sum_of_bases_l639_639026

noncomputable def sum_of_bases (a b h : ℝ) : ℝ :=
  real.sqrt (a ^ 2 - h ^ 2) + real.sqrt (b ^ 2 - h ^ 2)

theorem isosceles_trapezoid_sum_of_bases (a b h : ℝ) (h_pos : 0 < h) (a_bigger_h : h < a) (b_bigger_h : h < b) :
  sum_of_bases a b h = real.sqrt (a ^ 2 - h ^ 2) + real.sqrt (b ^ 2 - h ^ 2) :=
by
  -- proof to be implemented
  sorry

end isosceles_trapezoid_sum_of_bases_l639_639026


namespace mean_power_inequality_l639_639565

variable {α β : ℝ}
variable {n : ℕ}
variable {x : ℝ}
variable {xs : Fin n → ℝ}

noncomputable def S_alpha (α : ℝ) (xs : Fin n → ℝ) : ℝ :=
  ( ( finset.univ.sum (λ i, (xs i) ^ α) / n ) ) ^ (1 / α)

theorem mean_power_inequality
  (h1 : α < β)
  (h2 : α * β ≠ 0) :
  S_alpha α xs ≤ S_alpha β xs :=
sorry

end mean_power_inequality_l639_639565


namespace G_not_isomorphic_to_H_l639_639041

-- Define the group G
def G : set (matrix (fin 2) (fin 2) ℂ) := { A | complex.abs (matrix.det A) = 1 }

-- Define the group H
def H : set (matrix (fin 2) (fin 2) ℂ) := { A | matrix.det A = 1 }

theorem G_not_isomorphic_to_H :
  ¬(∃ f : matrix (fin 2) (fin 2) ℂ → matrix (fin 2) (fin 2) ℂ, 
     is_group_hom f ∧ bijective f ∧ (∀ x ∈ G, f x ∈ H) ∧ (∀ x ∈ H, f.symm x ∈ G)) :=
sorry

end G_not_isomorphic_to_H_l639_639041


namespace evaluate_expression_l639_639054

theorem evaluate_expression (x : Int) (h : x = -2023) : abs (abs (abs x - x) + abs x) + x = 4046 :=
by
  rw [h]
  sorry

end evaluate_expression_l639_639054


namespace minimum_common_correct_questions_l639_639460

open Finset

-- Define the students
inductive Student
| Xiaoxi | Xiaofei | Xianguan | Xialan
deriving DecidableEq

-- Define the problem set as a finset of questions
def questions : Finset ℕ := (range 10).toFinset

-- Define the set of questions each student answered correctly
variable (Xiaoxi_correct Xiaofei_correct Xianguan_correct Xialan_correct : Finset ℕ)

-- Define the conditions
axiom condition1 : Xiaoxi_correct.card = 8
axiom condition2 : Xiaofei_correct.card = 8
axiom condition3 : Xianguan_correct.card = 8
axiom condition4 : Xialan_correct.card = 8

-- Define the Lean statement to be proven
theorem minimum_common_correct_questions :
  ∃ common_questions : Finset ℕ, common_questions.card ≥ 2 ∧
    common_questions ⊆ Xiaoxi_correct ∧ 
    common_questions ⊆ Xiaofei_correct ∧ 
    common_questions ⊆ Xianguan_correct ∧ 
    common_questions ⊆ Xialan_correct :=
sorry

end minimum_common_correct_questions_l639_639460


namespace count_integers_with_increasing_digits_l639_639423

theorem count_integers_with_increasing_digits :
  let count_integers := 
    ∑ second_digit in ({3, 4, 5} : Finset ℕ), 
      ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ second_digit}, 
        1 in

  count_integers = 15 :=
by
  have step1 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 3}, 1 = 6,
  { -- Explanation: If second digit is 3, third can be 4, 5, 6, 7, 8, 9 -> 6 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step2 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 4}, 1 = 5,
  { -- Explanation: If second digit is 4, third can be 5, 6, 7, 8, 9 -> 5 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step3 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 5}, 1 = 4,
  { -- Explanation: If second digit is 5, third can be 6, 7, 8, 9 -> 4 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have count_integers := step1 + step2 + step3,
  simp only [count_integers, add_comm],
  exact eq.refl 15

end count_integers_with_increasing_digits_l639_639423


namespace rectangle_area_l639_639626

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l639_639626


namespace find_GL_l639_639180

def Triangle (α : Type) := α → α → α → Prop

variables {P₁ P₂ P₃ : Type} {point : Type} [linear_ordered_field P₁] [linear_ordered_field P₂] [linear_ordered_field P₃]

structure Circle (P : Type) :=
(center : P)
(radius : P₁)

noncomputable def GHI_inscribed_in_JKL (G H I J K L : point) : Prop := 
G ∈ [K, L] ∧ H ∈ [J, L] ∧ I ∈ [J, K]

noncomputable def circumcircle (A B C D : point) (circ : Circle point) : Prop :=
circ.center = circumcenter A B C

variables (G H I J K L : point)
variables (p q : ℕ)
variables [decidable_eq point]

-- Given conditions
axiom cond1 : GHI_inscribed_in_JKL G H I J K L 
axiom cond2 : circumcircle G I J P₁
axiom cond3 : circumcircle H L G P₂
axiom cond4 : circumcircle I J H P₃
axiom cond5 : (J to K : line) = 20
axiom cond6 : (K to L : line) = 27
axiom cond7 : (J to L : line) = 25
axiom cond8 : length HG = length IC
axiom cond9 : length JH = length GC
axiom cond10 : length IJ = length GL

-- Question: length of GL == 11.5 (23 / 2) => p = 23, q = 2 => p + q = 25
theorem find_GL : p + q = 25 :=
sorry

end find_GL_l639_639180


namespace tangent_line_to_circle_l639_639779

theorem tangent_line_to_circle
  (x y : ℝ)
  (h : x^2 + y^2 - 4 * x = 0)
  (hx : x = 1)
  (hy : y = real.sqrt 3) :
  x - real.sqrt 3 * y + 2 = 0 :=
sorry

end tangent_line_to_circle_l639_639779


namespace quarters_to_dimes_difference_l639_639102

variable (p : ℝ)

theorem quarters_to_dimes_difference :
  let Susan_quarters := 7 * p + 3
  let George_quarters := 2 * p + 9
  let difference_quarters := Susan_quarters - George_quarters
  let difference_dimes := 2.5 * difference_quarters
  difference_dimes = 12.5 * p - 15 :=
by
  let Susan_quarters := 7 * p + 3
  let George_quarters := 2 * p + 9
  let difference_quarters := Susan_quarters - George_quarters
  let difference_dimes := 2.5 * difference_quarters
  sorry

end quarters_to_dimes_difference_l639_639102


namespace find_k_l639_639222

-- Definitions based on conditions in step a)
def acute_isosceles_triangle_inscribed (A B C : Type) : Prop := sorry -- Formal definition of the triangle being acute isosceles and inscribed in a circle
def tangents_meeting_at_point (A B C D : Type) : Prop := sorry -- Formal definition of tangents through B and C meeting at D
def angle_relation (ABC D : Type) (theta : ℝ) : Prop := 3 * theta = sorry -- Formal definition of \(\angle ABC = \angle ACB = 3 \angle D\)
def angle_BAC (k : ℝ) (theta : ℝ) : Prop := theta = k * real.pi -- Formal definition of \(\angle BAC = k \pi\)

-- Theorem statement for our proof problem
theorem find_k
  (A B C D : Type)
  (h1 : acute_isosceles_triangle_inscribed A B C)
  (h2 : tangents_meeting_at_point A B C D)
  (theta : ℝ)
  (h3 : angle_relation ABC D theta)
  (k : ℝ)
  (h4 : angle_BAC k theta) :
  k = 1 / 13 := by
  sorry

end find_k_l639_639222


namespace gcd_n_squared_plus_4_n_plus_3_l639_639792

theorem gcd_n_squared_plus_4_n_plus_3 (n : ℕ) (hn_gt_four : n > 4) : 
  (gcd (n^2 + 4) (n + 3)) = if n % 13 = 10 then 13 else 1 := 
sorry

end gcd_n_squared_plus_4_n_plus_3_l639_639792


namespace find_b_100_l639_639283

def sequence (b : ℕ → ℤ) : Prop :=
  b 1 = 3 ∧ ∀ n, b (n + 1) = b n + 2 * n + 1

theorem find_b_100 (b : ℕ → ℤ) (h : sequence b) : b 100 = 10002 :=
  sorry

end find_b_100_l639_639283


namespace largest_k_not_exceeding_48_l639_639179

noncomputable def floor (a : ℝ) : ℤ := ⌊a⌋

theorem largest_k_not_exceeding_48 :
  ∃ k : ℕ,
    k ≤ 48 ∧ 
    (∏ i in finset.range (k + 1), (floor ((i + 1 : ℕ) / 7) + 1) % 13) = 7 ∧
    ∀ m : ℕ, m > k → m ≤ 48 →
    (∏ i in finset.range (m + 1), (floor ((i + 1 : ℕ) / 7) + 1) % 13) ≠ 7 :=
begin
  use 45,
  split,
  -- Proof of k ≤ 48
  {
    norm_num,
  },
  split,
  -- Proof of product up to k = 45 being equivalent to 7 modulo 13
  {
    sorry,  -- Proof here
  },
  -- Proof that no larger m up to 48 satisfies the condition
  {
    sorry,  -- Proof here
  },
end

end largest_k_not_exceeding_48_l639_639179


namespace k_value_correct_l639_639245

-- Let k be the value such that ∠BAC = k * π
def k : ℝ := 5 / 11

-- Define the isosceles triangle ABC inscribed in a circle
variables (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
variables [tangents_through B C D] (angle_BAC ABC ACB D : ℝ)

# Define the angles:
-- ∠ABC = ∠ACB = 3 * ∠D
axiom angle_equivalence : (ABC = 3 * D) ∧ (ACB = 3 * D)
-- ∠BAC = k * π
axiom angle_BAC_def : angle_BAC = k * real.pi

-- The proof assertion:
theorem k_value_correct (h : k = 5/11) : 
  (∀ (A B C D : Type) [acute_isosceles_triangle A B C] [inscribed_in_circle A B C]
     [tangents_through B C D] (angle_BAC ABC ACB D : ℝ),
  angle_equivalence → angle_BAC_def) → 
  angle_BAC = (5 / 11 : ℝ) * real.pi :=
by
  sorry

end k_value_correct_l639_639245


namespace correct_propositions_l639_639256

-- Define the function f
def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

-- Definitions of the propositions
def prop1 (b c: ℝ) : Prop := (∀ x : ℝ, f (-x) b c = - f x b c) → c = 0
def prop2 (c: ℝ) : Prop := f = λ x, x * |x| + c → ∃! x : ℝ, f x 0 c = 0
def prop3 (b c : ℝ) : Prop := ∀ x : ℝ, f (-x) b c = -x * |x| - b * x + c → f x b c + f (-x) b c = 2 * c
def prop4 (b c : ℝ) : Prop := b ≠ 0 → ∃ x1 x2 x3 : ℝ, f x1 b c = 0 ∧ f x2 b c = 0 ∧ f x3 b c = 0

-- Prove the correct propositions
theorem correct_propositions (b c : ℝ) :
prop1 b c ∧ prop2 c ∧ prop3 b c ∧ ¬ prop4 b c := sorry

end correct_propositions_l639_639256


namespace sum_of_primes_div_by_60_l639_639358

open Nat

theorem sum_of_primes_div_by_60 (p q r s : ℕ) (hp : p > 5) (hprimes : Prime p ∧ Prime q ∧ Prime r ∧ Prime s)
  (hq : p < q) (hr : q < r) (hs : r < s) (hbound : s < p + 10) : (p + q + r + s) % 60 = 0 := 
sorry

end sum_of_primes_div_by_60_l639_639358


namespace find_15th_term_l639_639292

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 2 = 4 ∧ ∀ n, a (n + 2) = a n

theorem find_15th_term :
  ∃ a : ℕ → ℕ, seq a ∧ a 15 = 3 :=
by
  sorry

end find_15th_term_l639_639292


namespace interest_rate_calculation_l639_639593

-- Define the problem conditions and proof statement in Lean
theorem interest_rate_calculation 
  (P : ℝ) (r : ℝ) (T : ℝ) (CI SI diff : ℝ) 
  (principal_condition : P = 6000.000000000128)
  (time_condition : T = 2)
  (diff_condition : diff = 15)
  (CI_formula : CI = P * (1 + r)^T - P)
  (SI_formula : SI = P * r * T)
  (difference_condition : CI - SI = diff) : 
  r = 0.05 := 
by 
  sorry

end interest_rate_calculation_l639_639593


namespace find_n_cosine_l639_639781

theorem find_n_cosine : ∃ (n : ℤ), -180 ≤ n ∧ n ≤ 180 ∧ (∃ m : ℤ, n = 25 + 360 * m ∨ n = -25 + 360 * m) :=
by
  sorry

end find_n_cosine_l639_639781


namespace sum_possible_n_l639_639207

theorem sum_possible_n (a b : ℕ) (h₁ : a = 7) (h₂ : b = 10) : 
  (∑ k in Finset.range (17 - 4 + 1), k + 4) = 260 := by
  sorry

end sum_possible_n_l639_639207


namespace average_after_17th_inning_l639_639681

-- Define the conditions.
variable (A : ℚ) -- The initial average after 16 innings

-- Define the score in the 17th inning and the increment in the average.
def runs_in_17th_inning : ℚ := 87
def increment_in_average : ℚ := 3

-- Define the equation derived from the conditions.
theorem average_after_17th_inning :
  (16 * A + runs_in_17th_inning) / 17 = A + increment_in_average →
  A + increment_in_average = 39 :=
sorry

end average_after_17th_inning_l639_639681


namespace ellipse_equation_proof_point_D_proof_l639_639355

noncomputable
def ellipse_equation (a b : ℝ) : Prop :=
a > b ∧ b > 0 ∧
  (\frac{x²}{a²} + \frac{y²}{b²} = 1) ∧ a = √2 ∧ b = 1

def point_D_exists (D : ℝ × ℝ) : Prop :=
D = (2, 0)
 
theorem ellipse_equation_proof :
  ∀ (e a b : ℝ)
    (h_cond1 : a > b)
    (h_cond2 : b > 0)
    (h_ecc : e = \frac{sqrt 2}{2})
    (h_c : c = 1),
    a = √2 ∧ b = 1 ∧ ellipse_equation a b
by {
    sorry
}

theorem point_D_proof :
  ∀ (e a b : ℝ)
    (h_cond1 : a > b)
    (h_cond2 : b > 0)
    (h_ecc : e = \frac{sqrt 2}{2}),
    a = √2 ∧ b = 1 ∧ point_D_exists (2, 0)
by {
    sorry
}

end ellipse_equation_proof_point_D_proof_l639_639355


namespace cos_product_zero_l639_639379

theorem cos_product_zero (x : ℝ) :
  sin x ^ 2 + sin (3 * x) ^ 2 + sin (5 * x) ^ 2 + sin (7 * x) ^ 2 = 2 →
  ∃ a b c : ℕ, cos (a * x) * cos (b * x) * cos (c * x) = 0 ∧ a + b + c = 14 :=
by
  intro h
  use 2, 4, 8
  split
  { sorry }
  { norm_num }

end cos_product_zero_l639_639379


namespace rectangle_area_l639_639606

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_area_l639_639606


namespace max_profit_l639_639632

-- Define the conditions
def profit (m : ℝ) := (m - 8) * (900 - 15 * m)

-- State the theorem
theorem max_profit (m : ℝ) : 
  ∃ M, M = profit 34 ∧ ∀ x, profit x ≤ M :=
by
  -- the proof goes here
  sorry

end max_profit_l639_639632


namespace monic_polynomials_equal_l639_639047

noncomputable def P : Polynomial ℤ := sorry -- assuming polynomial P
noncomputable def Q : Polynomial ℤ := sorry -- assuming polynomial Q

theorem monic_polynomials_equal
  (hP : P.monic)
  (hQ : Q.monic)
  (h : P.eval (P.X) = Q.eval (Q.X)) :
  P = Q := sorry

end monic_polynomials_equal_l639_639047


namespace exact_two_students_choose_A_l639_639797

-- Define the basic problem setup
def students := Finset.univ (Fin 4) -- This represents the four students as a finite set.

def courses := {'A', 'B', 'C'} -- This represents the three courses.

-- Define the condition that exactly two students choose course A.
def chooses_course_A (s : Finset (Fin 4)) : Prop := s.card = 2

-- Define the remaining two students can choose between courses B or C.
def remaining_students_choose_B_or_C : Prop := true -- This is a simplification; the actual logic would quantify over the possible choices.

-- Main theorem to be proved
theorem exact_two_students_choose_A : 
  (choose 4 2) * (choose 2 1) * (choose 2 1) = 24 :=
by sorry

end exact_two_students_choose_A_l639_639797


namespace triangle_square_side_length_ratio_l639_639736

theorem triangle_square_side_length_ratio (t s : ℝ) (ht : 3 * t = 12) (hs : 4 * s = 12) : 
  t / s = 4 / 3 :=
by
  sorry

end triangle_square_side_length_ratio_l639_639736


namespace count_increasing_order_digits_l639_639414

def count_valid_numbers : ℕ :=
  let numbers := [234, 235, 236, 237, 238, 239, 245, 246, 247, 248, 249] in
  numbers.length

theorem count_increasing_order_digits : count_valid_numbers = 11 :=
by
  sorry

end count_increasing_order_digits_l639_639414


namespace number_of_correct_propositions_l639_639108

theorem number_of_correct_propositions :
  let proposition1 := false -- A quadrilateral with diagonals perpendicular and equal is a square.
  let proposition2 := true  -- The sum of interior angles of a hexagon is 720°.
  let proposition3 := false -- Equal central angles correspond to equal arcs.
  let proposition4 := true  -- The quadrilateral formed by connecting the midpoints of a rhombus in sequence is a rectangle.
  let proposition5 := false -- If the midpoints of the sides of quadrilateral ABCD are connected in sequence, the resulting figure is a rectangle.
  (if proposition1 then 1 else 0 + 
   if proposition2 then 1 else 0 + 
   if proposition3 then 1 else 0 + 
   if proposition4 then 1 else 0 + 
   if proposition5 then 1 else 0) = 2 :=
by
  -- Define the propositions
  let proposition1 := false
  let proposition2 := true
  let proposition3 := false
  let proposition4 := true
  let proposition5 := false

  -- Counting the correct propositions
  have h : (if proposition1 then 1 else 0 + 
            if proposition2 then 1 else 0 + 
            if proposition3 then 1 else 0 + 
            if proposition4 then 1 else 0 + 
            if proposition5 then 1 else 0) = 2 := 
  by norm_num
  
  exact h

end number_of_correct_propositions_l639_639108


namespace angles_does_not_exist_l639_639834

theorem angles_does_not_exist (a1 a2 a3 : ℝ) 
  (h1 : a1 + a2 = 90) 
  (h2 : a2 + a3 = 180) 
  (h3 : a3 = 18) : False :=
by
  sorry

end angles_does_not_exist_l639_639834


namespace combined_weight_l639_639136

theorem combined_weight (y z : ℝ) 
  (h_avg : (25 + 31 + 35 + 33) / 4 = (25 + 31 + 35 + 33 + y + z) / 6) :
  y + z = 62 :=
by
  sorry

end combined_weight_l639_639136


namespace sale_in_second_month_l639_639194

theorem sale_in_second_month :
  ∀ (sale1 sale3 sale4 sale5 average : ℕ),
    sale1 = 5420 →
    sale3 = 6350 →
    sale4 = 6500 →
    sale5 = 6200 →
    average = 6300 →
    let total_sales := average * 5 in
    let known_sales := sale1 + sale3 + sale4 + sale5 in
    let sale2 := total_sales - known_sales in
    sale2 = 7030 :=
by 
  intros sale1 sale3 sale4 sale5 average H1 H3 H4 H5 Havg
  let total_sales := average * 5
  let known_sales := sale1 + sale3 + sale4 + sale5
  let sale2 := total_sales - known_sales
  have Hsale1 : sale1 = 5420 := H1
  have Hsale3 : sale3 = 6350 := H3
  have Hsale4 : sale4 = 6500 := H4
  have Hsale5 : sale5 = 6200 := H5
  have Havg := avg
  sorry

end sale_in_second_month_l639_639194


namespace cone_volume_is_correct_l639_639193

noncomputable def cone_volume (lateral_surf_area : ℝ) (base_surf_area : ℝ) : ℝ :=
  let r : ℝ := Real.sqrt (base_surf_area / π) in
  let l : ℝ := (2 * lateral_surf_area) / (2 * π) in
  let h : ℝ := Real.sqrt (l^2 - r^2) in
  (1 / 3) * π * r^2 * h

theorem cone_volume_is_correct : cone_volume 2 π = (Real.sqrt 3 * π) / 3 :=
by
  unfold cone_volume
  have r_def : Real.sqrt (π / π) = 1 := by sorry
  have l_def : (2 * 2) / (2 * π) = 2 := by sorry
  have h_def : Real.sqrt (2^2 - 1^2) = Real.sqrt 3 := by sorry
  rw [r_def, l_def, h_def]
  norm_num
  rw [← Real.sqrt_div (by norm_num : 3 ≥ 0) (by norm_num : 1 ≥ 0), Real.sqrt_one]
  rw [Real.sqrt_mul_self (by norm_num : 3 ≥ 0), Real.sqrt_div_self (by norm_num : 3 ≥ 0)]
  norm_num
  sorry

end cone_volume_is_correct_l639_639193


namespace allie_skate_distance_l639_639652

theorem allie_skate_distance
  (A B : Type)
  (AB : ℝ)
  (AB_100 : AB = 100)
  (t : ℝ)
  (allie_speed : ℝ := 8)
  (billie_speed : ℝ := 7)
  (angle_60 : ℝ := 60)
  (AC : ℝ := allie_speed * t)
  (BC : ℝ := billie_speed * t)
  (cos_60 : ℝ := 0.5) : 
  AC = 8 * 20 :=
sorry

# Let A and B be points (types).
# Let AB = 100 m.
# Let allie_speed be 8 m/s and billie_speed be 7 m/s.
# Let the angle between AB and Allie's path be 60 degrees.
# Define AC as the distance Allie skates: 8 * t.
# Define BC as the distance Billie skates: 7 * t.
# Cosine of 60 degrees is 0.5.
# Prove that AC equals the distance skated by Allie in 20 seconds.

end allie_skate_distance_l639_639652


namespace liquid_fill_time_l639_639693

-- Define the inflow and outflow rates
variables {x y t1 t2 : ℝ}

-- Define the initial conditions as separate hypotheses
theorem liquid_fill_time
  (h1 : y - x = 1 / 4)
  (h2 : t2 = 66 / y)
  (h3 : t2 - t1 = 3)
  (h4 : t1 = 40 / x) :
  t1 = 5 ∨ t1 = 96 :=
sorry

end liquid_fill_time_l639_639693


namespace find_k_l639_639229

noncomputable theory
open_locale classical

variables (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)

-- Condition 1: ABC is an acute isosceles triangle inscribed in a circle
-- Condition 2: Tangents from B and C meet at D
-- Condition 3: angle ABC = angle ACB = 3 * angle D
    
def tangents_intersect_at_D (ABC : triangle) (B C D : point ABC) : Prop :=
  tangent_to_circle_at B D ∧ tangent_to_circle_at C D ∧ B ≠ C

def isosceles_triangle_condition (angle : point ABC → point ABC → point ABC → real) (B C : point ABC) : Prop :=
  angle ABC B C = angle ABC C B

def angle_triple_D (angle : point ABC → point ABC → point ABC → real) (B C D : point ABC) : Prop :=
  ∃ k, angle ABC B C = 3 * angle ABC D C ∧ angle ABC A B = k * π

theorem find_k (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)
  (h1 : tangents_intersect_at_D ABC B C D)
  (h2 : isosceles_triangle_condition angle B C)
  (h3 : angle_triple_D angle B C D) :
  ∃ k : real, angle ABC A B = (5 / 11) * π ∧ k = 5 / 11 :=
by sorry

end find_k_l639_639229


namespace AM_GM_Inequality_l639_639062

theorem AM_GM_Inequality 
  (a b c : ℝ) 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (habc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end AM_GM_Inequality_l639_639062


namespace parallelogram_area_l639_639463

-- Define the vertices of the parallelogram
variables {x p a b : ℝ}

-- Define conditions where 'a' and 'b' are displacements, 'x' is the x-coordinate, and 'p' is the height
def vertices (x p a b : ℝ) : Prop := 
  true -- no specific constraints needed for the vertices as per the problem given

-- Define the base of the parallelogram from the conditions
def base (a : ℝ) : ℝ :=
  2 * a

-- Define the height of the parallelogram from the conditions
def height (p : ℝ) : ℝ :=
  2 * p

-- Define the area of the parallelogram using base and height
def area (a p : ℝ) : ℝ :=
  base a * height p

-- Prove that the area of the parallelogram is 4 * a * p
theorem parallelogram_area {x p a b : ℝ} (h : vertices x p a b) : area a p = 4 * a * p :=
by
  unfold base height area
  simp
  sorry

end parallelogram_area_l639_639463


namespace polygon_same_perimeter_l639_639651

theorem polygon_same_perimeter (s : ℝ) (H : s ≠ 0) :
  let sides_first := 50
  let side_length_first := 3 * s
  let perimeter_first := sides_first * side_length_first
  let side_length_second := s
  let perimeter_second := perimeter_first
  let sides_second := perimeter_second / side_length_second
  sides_second = 150 :=
by
  let sides_first := 50
  let side_length_first := 3 * s
  let perimeter_first := sides_first * side_length_first
  let side_length_second := s
  let perimeter_second := perimeter_first
  let sides_second := perimeter_second / side_length_second
  calc
    sides_second = perimeter_second / side_length_second : rfl
             ...  = (sides_first * side_length_first) / side_length_second : by rw [perimeter_second, perimeter_first]
             ...  = (50 * (3 * s)) / s : rfl
             ...  = 150 : by field_simp [H]

end polygon_same_perimeter_l639_639651


namespace find_number_l639_639325

noncomputable def N := 953.87

theorem find_number (h : (0.47 * N - 0.36 * 1412) + 65 = 5) : N = 953.87 := sorry

end find_number_l639_639325


namespace arithmetic_sequence_y_l639_639285

theorem arithmetic_sequence_y :
  ∃ y : ℤ, (∃ a b : ℤ, a = 3^2 ∧ b = 3^4 ∧ (a + b) / 2 = y) ∧ y = 45 :=
by
  use 45
  use 9, 81
  sorry

end arithmetic_sequence_y_l639_639285


namespace sum_of_coefficients_condition_l639_639964

theorem sum_of_coefficients_condition 
  (t : ℕ → ℤ) 
  (d e f : ℤ) 
  (h0 : t 0 = 3) 
  (h1 : t 1 = 7) 
  (h2 : t 2 = 17) 
  (h3 : t 3 = 86)
  (rec_relation : ∀ k ≥ 2, t (k + 1) = d * t k + e * t (k - 1) + f * t (k - 2)) : 
  d + e + f = 14 :=
by
  sorry

end sum_of_coefficients_condition_l639_639964


namespace chessboard_max_pieces_l639_639987

theorem chessboard_max_pieces {m n : ℕ} (rows cols : fin m → fin n → bool)
  (condition1 : m = 200) (condition2 : n = 200)
  (condition3 : ∀ r c, rows r c = true → (∃ dr, rows (r + dr) c = false) ∧ (∃ dc, rows r (c + dc) = false))
  (condition4 : ∀ r c, cols r c = true → (∃ dr, cols (r + dr) c = false) ∧ (∃ dc, cols r (c + dc) = false))
  (condition5 : ∀ r c, (rows r c = true ∨ cols r c = true) →
    (∃ a b, a ≠ b ∧ (rows r a = true ∨ cols r a = true) ∧ (rows r b = true ∨ cols r b = true) ∧
      (rows r a = false ∧ cols r a = false ∧ rows r b = false ∧ cols r b = false)))
  : ∑ r c, rows r c + cols r c ≤ 3800 := 
sorry

end chessboard_max_pieces_l639_639987


namespace new_average_weight_l639_639137

def num_people := 6
def avg_weight1 := 154
def weight_seventh := 133

theorem new_average_weight :
  (num_people * avg_weight1 + weight_seventh) / (num_people + 1) = 151 := by
  sorry

end new_average_weight_l639_639137


namespace reflection_squared_is_identity_l639_639956

noncomputable def reflection_matrix (v : ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let (a, b) := v in
  let norm := real.sqrt (a * a + b * b) in
  ![(a * a - b * b) / norm^2, (2 * a * b) / norm^2; (2 * a * b) / norm^2, (b * b - a * a) / norm^2]

theorem reflection_squared_is_identity :
  let S := reflection_matrix (4, -2) in S * S = 1 :=
sorry

end reflection_squared_is_identity_l639_639956


namespace percentage_female_guests_from_jay_family_l639_639978

def total_guests : ℕ := 240
def female_guests_percentage : ℕ := 60
def female_guests_from_jay_family : ℕ := 72

theorem percentage_female_guests_from_jay_family :
  (female_guests_from_jay_family : ℚ) / (total_guests * (female_guests_percentage / 100) : ℚ) * 100 = 50 := by
  sorry

end percentage_female_guests_from_jay_family_l639_639978


namespace marble_total_weight_l639_639559

theorem marble_total_weight :
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 + 0.21666666666666667 + 0.4583333333333333 + 0.12777777777777778 = 1.5527777777777777 :=
by
  sorry

end marble_total_weight_l639_639559


namespace janet_income_difference_l639_639037

def janet_current_job_income (hours_per_week : ℕ) (weeks_per_month : ℕ) (hourly_rate : ℝ) : ℝ :=
  hours_per_week * weeks_per_month * hourly_rate

def janet_freelance_income (hours_per_week : ℕ) (weeks_per_month : ℕ) (hourly_rate : ℝ) : ℝ :=
  hours_per_week * weeks_per_month * hourly_rate

def extra_fica_taxes (weekly_tax : ℝ) (weeks_per_month : ℕ) : ℝ :=
  weekly_tax * weeks_per_month

def healthcare_premiums (monthly_premium : ℝ) : ℝ :=
  monthly_premium

def janet_net_freelance_income (freelance_income : ℝ) (additional_costs : ℝ) : ℝ :=
  freelance_income - additional_costs

theorem janet_income_difference
  (hours_per_week : ℕ)
  (weeks_per_month : ℕ)
  (current_hourly_rate : ℝ)
  (freelance_hourly_rate : ℝ)
  (weekly_tax : ℝ)
  (monthly_premium : ℝ)
  (H_hours : hours_per_week = 40)
  (H_weeks : weeks_per_month = 4)
  (H_current_rate : current_hourly_rate = 30)
  (H_freelance_rate : freelance_hourly_rate = 40)
  (H_weekly_tax : weekly_tax = 25)
  (H_monthly_premium : monthly_premium = 400) :
  janet_net_freelance_income (janet_freelance_income 40 4 40) (extra_fica_taxes 25 4 + healthcare_premiums 400) 
  - janet_current_job_income 40 4 30 = 1100 := 
  by 
    sorry

end janet_income_difference_l639_639037


namespace count_integers_with_increasing_digits_200_to_250_l639_639427

theorem count_integers_with_increasing_digits_200_to_250 :
  {n : ℕ | 200 ≤ n ∧ n ≤ 250 ∧ 
          (let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 in 
           d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
           d1 < d2 ∧ d2 < d3)}.card = 11 :=
by
  sorry

end count_integers_with_increasing_digits_200_to_250_l639_639427


namespace identical_roots_l639_639855

def f (x a b : ℝ) : ℝ := x^2 + a*x + b * Real.cos x

theorem identical_roots (a b : ℝ) :
  (∀ x, f x a b = 0 ↔ f (f x a b) a b = 0) ↔ (b = 0 ∧ 0 ≤ a ∧ a < 4) :=
by
  sorry

end identical_roots_l639_639855


namespace value_of_dowry_l639_639595

theorem value_of_dowry :
  (∀ (c : ℝ), 
    let calf := (10 / 3) * c in
    let cow := calf + 4000 in
    let chicken := (5 / 3000) * calf in
    5 * cow + 7 * calf + 9 * c + chicken = 108210 →
    cow + calf + c + chicken = 17810) :=
begin
  intros c H,
  sorry
end

end value_of_dowry_l639_639595


namespace count_integers_between_200_and_250_with_increasing_digits_l639_639430

theorem count_integers_between_200_and_250_with_increasing_digits :
  ∃ n, n = 11 ∧ ∀ x, 200 ≤ x ∧ x ≤ 250 ∧ 
  (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k) → 
  n = ∑ x in {x | 200 ≤ x ∧ x ≤ 250 ∧ (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k)}, 1
:= 
sorry

end count_integers_between_200_and_250_with_increasing_digits_l639_639430


namespace sequence_term_x_l639_639789

theorem sequence_term_x:
  ∃ (x : ℤ),
    let a := [8, 62, x, -4, -12],
        diff := list.zip_with (λ i j, j - i) a.init.tail a.tail in
    diff.nth 0 = some 54 ∧
    diff.nth 3 = some (-8) ∧
    diff.nth 1 = some (x - 62) ∧
    diff.nth 2 = some (-4 - x) ∧
    x = 29 :=
by
  existsi 29
  sorry

end sequence_term_x_l639_639789


namespace perimeter_is_11_or_13_l639_639017

-- Define the basic length properties of the isosceles triangle
structure IsoscelesTriangle where
  a b c : ℝ
  a_eq_b_or_a_eq_c : a = b ∨ a = c

-- Conditions of the problem
def sides : IsoscelesTriangle where
  a := 3
  b := 5
  c := 5
  a_eq_b_or_a_eq_c := Or.inr rfl

def sides' : IsoscelesTriangle where
  a := 3
  b := 3
  c := 5
  a_eq_b_or_a_eq_c := Or.inl rfl

-- Prove the perimeters
theorem perimeter_is_11_or_13 : (sides.a + sides.b + sides.c = 13) ∨ (sides'.a + sides'.b + sides'.c = 11) :=
by
  sorry

end perimeter_is_11_or_13_l639_639017


namespace problem1_problem2_l639_639581

-- Problem 1
theorem problem1 (x : ℝ) (h : 2^(x+3) * 3^(x+3) = 36^(x-2)) : x = 7 :=
sorry

-- Problem 2
theorem problem2 (α β : ℝ) (h1 : 10^(-2*α) = 3) (h2 : 10^(-β) = 1/5) : 10^(2*α - 2*β) = 1/75 :=
sorry

end problem1_problem2_l639_639581


namespace count_integers_between_200_and_250_with_increasing_digits_l639_639434

theorem count_integers_between_200_and_250_with_increasing_digits :
  ∃ n, n = 11 ∧ ∀ x, 200 ≤ x ∧ x ≤ 250 ∧ 
  (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k) → 
  n = ∑ x in {x | 200 ≤ x ∧ x ≤ 250 ∧ (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k)}, 1
:= 
sorry

end count_integers_between_200_and_250_with_increasing_digits_l639_639434


namespace labor_hired_l639_639462

noncomputable def Q_d (P : ℝ) : ℝ := 60 - 14 * P
noncomputable def Q_s (P : ℝ) : ℝ := 20 + 6 * P
noncomputable def MPL (L : ℝ) : ℝ := 160 / (L^2)
def wage : ℝ := 5

theorem labor_hired (L P : ℝ) (h_eq_price: 60 - 14 * P = 20 + 6 * P) (h_eq_wage: 160 / (L^2) * 2 = wage) :
  L = 8 :=
by
  have h1 : 60 - 14 * P = 20 + 6 * P := h_eq_price
  have h2 : 160 / (L^2) * 2 = wage := h_eq_wage
  sorry

end labor_hired_l639_639462


namespace annual_interest_rate_approx_l639_639936

noncomputable def principal := 150
noncomputable def total_paid := 165
noncomputable def periods := 3  -- 18 months with semi-annual compounding

def r := (total_paid / principal) ^ (1 / periods : ℝ) - 1
def annual_rate := 2 * r

theorem annual_interest_rate_approx :
  annual_rate ≈ 0.0646 := 
sorry

end annual_interest_rate_approx_l639_639936


namespace remainder_equivalence_l639_639663

theorem remainder_equivalence (x y q r : ℕ) (hxy : x = q * y + r) (hy_pos : 0 < y) (h_r : 0 ≤ r ∧ r < y) : 
  ((x - 3 * q * y) % y) = r := 
by 
  sorry

end remainder_equivalence_l639_639663


namespace range_of_a_l639_639384

noncomputable def f (a x : ℝ) := (x^2 - a * x) * exp x

noncomputable def g (a x : ℝ) := x^2 + (2 - a) * x - a

theorem range_of_a {a : ℝ} :
  (∀ x ∈ set.Icc (-1 : ℝ) 1, ∃ x₀ ∈ set.Icc (-1 : ℝ) 1, ∇ (f a x) x₀ > 0)
  → a < 3 / 2 :=
sorry

end range_of_a_l639_639384


namespace isosceles_triangle_perimeter_l639_639015

theorem isosceles_triangle_perimeter 
    (a b : ℕ) (h_iso : a = 3 ∨ a = 5) (h_other : b = 3 ∨ b = 5) 
    (h_distinct : a ≠ b) : 
    ∃ p : ℕ, p = (3 + 3 + 5) ∨ p = (5 + 5 + 3) :=
by
  sorry

end isosceles_triangle_perimeter_l639_639015


namespace count_integers_between_200_and_250_with_increasing_digits_l639_639435

theorem count_integers_between_200_and_250_with_increasing_digits :
  ∃ n, n = 11 ∧ ∀ x, 200 ≤ x ∧ x ≤ 250 ∧ 
  (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k) → 
  n = ∑ x in {x | 200 ≤ x ∧ x ≤ 250 ∧ (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k)}, 1
:= 
sorry

end count_integers_between_200_and_250_with_increasing_digits_l639_639435


namespace distance_ratio_l639_639106

variable {A B C X : Type} [BC: LineSegment B C]
variables [BX CX AB AC : Real] [d_b d_c : Real]
variables (X_on_BC : X ∈ BC)
variables (d_b_perp_AB : PerpendicularDist X AB d_b)
variables (d_c_perp_AC : PerpendicularDist X AC d_c)

theorem distance_ratio :
  d_b / d_c = (BX * AC) / (CX * AB) :=
sorry

end distance_ratio_l639_639106


namespace solution_l639_639662

open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def prime_factors (n : ℕ) : list ℕ := factors n
def smallest_integer (predicate : ℕ → Prop) : ℕ := if h: ∃ n, predicate n then Classical.choose h else 0

noncomputable def Problem : ℕ :=
smallest_integer (λ n, 
  (¬ is_prime n ∧ ¬ is_square n) ∧
  (∀ p ∈ prime_factors n, 60 ≤ p))

theorem solution : Problem = 290977 :=
sorry

end solution_l639_639662


namespace product_of_roots_of_polynomial_l639_639504

/-- 
Construct a polynomial Q(x) with rational coefficients such that 
  1. The polynomial is of least possible degree.
  2. It has the root \sqrt[4]{11} + (\sqrt[4]{11})^2.
Prove that the product of all roots of Q(x) equals -11.
-/
theorem product_of_roots_of_polynomial :
  ∃ Q : Polynomial ℚ, Q.degree = 8 ∧ (∀ r, IsRoot Q r → r = (rootOf4 11) + (rootOf4 11)^2) ∧ 
  (Q.roots.map (λ r, r.coeff 0)).prod = -11 :=
sorry

end product_of_roots_of_polynomial_l639_639504


namespace number_of_sets_B_l639_639976

open Set

theorem number_of_sets_B : 
  {B : Set ℕ // {1, 3} ∪ B = {1, 3, 5}}.card = 4 := 
by 
  sorry

end number_of_sets_B_l639_639976


namespace inches_of_rain_received_so_far_l639_639021

def total_days_in_year : ℕ := 365
def days_left_in_year : ℕ := 100
def rain_per_day_initial_avg : ℝ := 2
def rain_per_day_required_avg : ℝ := 3

def total_annually_expected_rain : ℝ := rain_per_day_initial_avg * total_days_in_year
def days_passed_in_year : ℕ := total_days_in_year - days_left_in_year
def total_rain_needed_remaining : ℝ := rain_per_day_required_avg * days_left_in_year

variable (S : ℝ) -- inches of rain received so far

theorem inches_of_rain_received_so_far (S : ℝ) :
  S + total_rain_needed_remaining = total_annually_expected_rain → S = 430 :=
  by
  sorry

end inches_of_rain_received_so_far_l639_639021


namespace walking_speed_l639_639198

theorem walking_speed (total_time : ℕ) (distance : ℕ) (rest_interval : ℕ) (rest_time : ℕ) (rest_periods: ℕ) 
  (total_rest_time: ℕ) (total_walking_time: ℕ) (hours: ℕ) 
  (H1 : total_time = 332) 
  (H2 : distance = 50) 
  (H3 : rest_interval = 10) 
  (H4 : rest_time = 8)
  (H5 : rest_periods = distance / rest_interval - 1) 
  (H6 : total_rest_time = rest_periods * rest_time)
  (H7 : total_walking_time = total_time - total_rest_time) 
  (H8 : hours = total_walking_time / 60) : 
  (distance / hours) = 10 :=
by {
  -- proof omitted
  sorry
}

end walking_speed_l639_639198


namespace power_mean_inequality_l639_639562

noncomputable def power_mean (α : ℝ) (x : list ℝ) : ℝ :=
  (list.sum (x.map (λ xi : ℝ, xi^α)) / x.length.to_real)^(1/α)

theorem power_mean_inequality (α β : ℝ) (x : list ℝ) :
  α < β → α * β ≠ 0 → power_mean α x ≤ power_mean β x :=
by
-- Proof goes here
sorry

end power_mean_inequality_l639_639562


namespace kanul_spent_on_raw_materials_eq_500_l639_639493

variable (total_amount : ℕ)
variable (machinery_cost : ℕ)
variable (cash_percentage : ℕ)

def amount_spent_on_raw_materials (total_amount machinery_cost cash_percentage : ℕ) : ℕ :=
  total_amount - machinery_cost - (total_amount * cash_percentage / 100)

theorem kanul_spent_on_raw_materials_eq_500 :
  total_amount = 1000 →
  machinery_cost = 400 →
  cash_percentage = 10 →
  amount_spent_on_raw_materials total_amount machinery_cost cash_percentage = 500 :=
by
  intros
  sorry

end kanul_spent_on_raw_materials_eq_500_l639_639493


namespace reflection_twice_is_identity_l639_639954

theorem reflection_twice_is_identity :
  let a := ![4, -2]
  let S := reflection_matrix a
  S * S = (1 : Matrix (Fin 2) (Fin 2) ℝ) := 
by
  let a := ![4, -2]
  let S := reflection_matrix a
  sorry

end reflection_twice_is_identity_l639_639954


namespace imaginary_part_of_division_l639_639601

open Complex

-- Given conditions
def complex1 : ℂ := 1 - I
def complex2 : ℂ := 1 - 2 * I

-- Question: Imaginary part of the division of these complex numbers
def division : ℂ := complex1 / complex2

-- Prove that the imaginary part of the division is 1/5
theorem imaginary_part_of_division :
  (division).im = 1 / 5 := by
  sorry

end imaginary_part_of_division_l639_639601


namespace area_trapezoid_l639_639032

theorem area_trapezoid (a b : ℝ) (hBC : BC = a) (hMC : MC = b) (hAngle : angle M C B = 150) :
  area ABCD = (a * b) / 2 :=
sorry

end area_trapezoid_l639_639032


namespace quadratic_inequality_l639_639116

theorem quadratic_inequality (x : ℝ) : x^2 - x + 1 ≥ 0 :=
sorry

end quadratic_inequality_l639_639116


namespace olympiad_problem_solution_l639_639719

theorem olympiad_problem_solution
  (n : ℕ)
  (b g : Fin n → ℕ)
  (h1 : ∀ i, b i = 2 * g i ∨ g i = 2 * b i) :
  (∑ i, b i + g i) ≠ 2000 :=
by
  sorry

end olympiad_problem_solution_l639_639719


namespace number_of_true_propositions_is_one_l639_639845

/-- Given a set of four propositions, the number of true propositions is 1. -/
theorem number_of_true_propositions_is_one :
  let P1 := ∀ (b a x̄ ȳ : ℝ), ∀ (y x : ℝ), y = b * x + a → (x̄, ȳ) := ((x + x̄) / 2, (y + ȳ) / 2) → true
  let P2 := ∀ (x : ℝ), (x = 6 → x^2 - 5*x - 6 = 0) ∧ (¬(x^2 - 5*x - 6 = 0 → x = 6))
  let P3 := ¬ (∃ (x₀ : ℝ), x₀^2 + 2*x₀ + 3 < 0) ↔ ∀ (x : ℝ), x^2 + 2*x + 3 ≥ 0
  let P4 := ∀ (p q : Prop), (p ∨ q) → (¬p ∧ ¬q)
  in (if P1 = true then 1 else 0) + (if P2 = true then 1 else 0) + (if P3 = true then 1 else 0) + (if P4 = true then 1 else 0) = 1 :=
begin
  sorry,
end

end number_of_true_propositions_is_one_l639_639845


namespace prod_sum_leq_four_l639_639535

theorem prod_sum_leq_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 4) :
  ab + bc + cd + da ≤ 4 :=
sorry

end prod_sum_leq_four_l639_639535


namespace a3_l639_639182

   noncomputable def a_n (n : ℕ) : ℝ := sorry
   theorem a3 (h : ∀ n, (list.range n).prod (λ i, a_n (i + 1)) = n + 1) : a_n 3 = 4 / 3 := sorry
   
end a3_l639_639182


namespace one_student_will_deduce_l639_639145

-- Let's define the problem in Lean

inductive Student : Type
| A
| B

structure GameState where
  a b x y : ℕ
  h1 : x < y
  h2 : (a + b = x) ∨ (a + b = y)

noncomputable def eventually_one_student_deduces (s : Student) (gs : GameState) : Prop := 
  ∃ N, ∀ n ≥ N, if h : s = Student.A 
    then (gs.a + n = gs.y) ∨ (gs.a + n = gs.x)
    else (gs.b + n = gs.y) ∨ (gs.b + n = gs.x)

theorem one_student_will_deduce (gs : GameState) : 
  ∃ (s : Student), eventually_one_student_deduces s gs := 
sorry

end one_student_will_deduce_l639_639145


namespace prob_at_least_seven_is_one_ninth_l639_639771

def prob_at_least_seven_stayed (fixed: ℕ) (unsure: ℕ) (prob_unsure_stay: ℚ) : ℚ :=
  let exactly_three_unsure : ℚ := (4.choose 3) * (prob_unsure_stay ^ 3) * ((1 - prob_unsure_stay) ^ 1)
  let all_eight : ℚ := prob_unsure_stay ^ 4
  exactly_three_unsure + all_eight

theorem prob_at_least_seven_is_one_ninth :
  prob_at_least_seven_stayed 4 4 (1/3) = 1 / 9 :=
by
  sorry

end prob_at_least_seven_is_one_ninth_l639_639771


namespace tan_a_over_tan_b_plus_tan_b_over_tan_a_l639_639960

theorem tan_a_over_tan_b_plus_tan_b_over_tan_a {a b : ℝ} 
  (h1 : (Real.sin a / Real.cos b) + (Real.sin b / Real.cos a) = 2)
  (h2 : (Real.cos a / Real.sin b) + (Real.cos b / Real.sin a) = 4) :
  (Real.tan a / Real.tan b) + (Real.tan b / Real.tan a) = 44 / 5 :=
sorry

end tan_a_over_tan_b_plus_tan_b_over_tan_a_l639_639960


namespace fill_tank_time_l639_639078

theorem fill_tank_time (t_A t_B : ℕ) (hA : t_A = 20) (hB : t_B = t_A / 4) :
  t_B = 4 := by
  sorry

end fill_tank_time_l639_639078


namespace log_9_256_eq_4_log_2_3_l639_639299

noncomputable def logBase9Base2Proof : Prop :=
  (Real.log 256 / Real.log 9 = 4 * (Real.log 3 / Real.log 2))

theorem log_9_256_eq_4_log_2_3 : logBase9Base2Proof :=
by
  sorry

end log_9_256_eq_4_log_2_3_l639_639299


namespace pages_per_hour_l639_639127

-- Definitions corresponding to conditions
def lunch_time : ℕ := 4 -- time taken to grab lunch and come back (in hours)
def total_pages : ℕ := 4000 -- total pages in the book
def book_time := 2 * lunch_time  -- time taken to read the book is twice the lunch_time

-- Statement of the problem to be proved
theorem pages_per_hour : (total_pages / book_time = 500) := 
  by
    -- We assume the definitions and want to show the desired property
    sorry

end pages_per_hour_l639_639127


namespace measure_4_liters_with_3_and_5_l639_639927

theorem measure_4_liters_with_3_and_5 (V1 V2 : ℕ) (C1 C2 : ℕ) (V1_Cap V2_Cap : ℕ) (V1_Init V2_Init : ℕ) :
  V1_Cap = 3 → V2_Cap = 5 → V1_Init = 0 → V2_Init = 0 →
  ∃ (steps : list (ℕ × ℕ)), ((∀ (v : ℕ × ℕ), v ∈ steps →
    (∃ (c : ℕ), c = V1 ∧ c ≤ V1_Cap) ∧ (∃ (c : ℕ), c = V2 ∧ c ≤ V2_Cap))) ∧
    (∃ s, s = steps.last ∧ s.2 = 4) :=
by
  { -- conditions of vessels and initial states
    intros hVC1 hVC2 hV_Init1 hV_Init2,
    sorry
  }

end measure_4_liters_with_3_and_5_l639_639927


namespace sales_quota_50_l639_639334

theorem sales_quota_50 :
  let cars_sold_first_three_days := 5 * 3
  let cars_sold_next_four_days := 3 * 4
  let additional_cars_needed := 23
  let total_quota := cars_sold_first_three_days + cars_sold_next_four_days + additional_cars_needed
  total_quota = 50 :=
by
  -- proof goes here
  sorry

end sales_quota_50_l639_639334


namespace B_complete_work_in_8_days_l639_639188

theorem B_complete_work_in_8_days : ∃ B : ℝ, (1 / 20 + 2 / B) * 3 = 1 ∧ B = 8 := by
  sorry

end B_complete_work_in_8_days_l639_639188


namespace equilateral_triangle_square_ratio_l639_639734

theorem equilateral_triangle_square_ratio (t s : ℕ) (h_t : 3 * t = 12) (h_s : 4 * s = 12) :
  t / s = 4 / 3 := by
  sorry

end equilateral_triangle_square_ratio_l639_639734


namespace fifteenth_term_is_three_l639_639294

noncomputable def sequence (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 else 4

theorem fifteenth_term_is_three : sequence 15 = 3 :=
  by sorry

end fifteenth_term_is_three_l639_639294


namespace f_99_99_lt_1_99_l639_639947

def f (m n : ℕ) : ℝ :=
  if m = 0 then 2^n else
  if n = 0 then 1 else
  2 * (f (m - 1) n) * (f m (n - 1)) / ((f (m - 1) n) + (f m (n - 1)))

theorem f_99_99_lt_1_99 : f 99 99 < 1.99 := by
  sorry

end f_99_99_lt_1_99_l639_639947


namespace select_proper_simulation_function_predict_falling_months_l639_639191

variable {p q : ℝ}
variable {f : ℝ → ℝ}

-- Hypotheses
def is_price_rise_beginning_end_fall_middle := ∃ (f: ℝ → ℝ), 
  (∀ x, x ∈ {0, 1, 2, 3, 4, 5} → f = x(x-q)^2 + p) ∧ 
  (f(0) = 4) ∧ (f(2) = 6) 

def selected_function_property := ∀ q > 1, (f = x(x-q)^2 + p)

-- Statements
theorem select_proper_simulation_function: selected_function_property :=
  sorry

theorem predict_falling_months : is_price_rise_beginning_end_fall_middle :=
  sorry

end select_proper_simulation_function_predict_falling_months_l639_639191


namespace currency_exchange_l639_639731

def exchange_rate : ℝ := 5000 / 2.50
def dollars_to_lire (d : ℝ) : ℝ := exchange_rate * d
def lire_to_dollars (l : ℝ) : ℝ := l / exchange_rate

theorem currency_exchange (d : ℝ) (l : ℝ) :
  dollars_to_lire 50 = 100000 ∧ lire_to_dollars 7500 = 3.75 :=
by 
  sorry

end currency_exchange_l639_639731


namespace jerry_shelf_l639_639491

theorem jerry_shelf :
  let initial_action_figures := 7 in
  let initial_books := 2 in
  let additional_books := 4 in
  let final_books := initial_books + additional_books in
  initial_action_figures - final_books = 1 :=
by
  sorry

end jerry_shelf_l639_639491


namespace part1_part2_l639_639382

-- Defining the conditions for part 1
def f (x a : ℝ) : ℝ := real.exp (x - a) - x

-- Part 1 statement: If f(x) has two zeros, then a > 1
theorem part1 (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) : a > 1 := sorry

-- Defining the conditions for part 2
def g (x a : ℝ) : ℝ := real.exp (x - a) - x * real.log x + (1 - a) * x

-- Define the local minimum h(a)
noncomputable def h (a : ℝ) : ℝ := 
  let x2 := -- value of x2 that satisfies certain condition (defined here abstractly)
  g x2 a

-- Part 2 statement: For the given range of a, the range of h(a) is [-3, 1)
theorem part2 (a : ℝ) (ha : 1 < a ∧ a ≤ 3 - real.log 3) : -3 ≤ h(a) ∧ h(a) < 1 := sorry

end part1_part2_l639_639382


namespace problem1_problem2_l639_639091

theorem problem1 :
  (7 + 4 * Real.sqrt 3) ^ (1 / 2) - 81 ^ (1 / 8) + 32 ^ (3 / 5) - 2 * (1 / 8) ^ (-2 / 3) + 
  32 * (4 ^ (-1 / 3)) ^ (-1) = 4 :=
  sorry

theorem problem2 :
  (Real.log 6 2) ^ 2 + (Real.log 6 3) ^ 2 + 3 * (Real.log 6 2) * (Real.log 6 318 - (1 / 3) * Real.log 6 2) = 1 :=
  sorry

end problem1_problem2_l639_639091


namespace geo_seq_sum_l639_639586

theorem geo_seq_sum (a : ℤ) (S : ℕ → ℤ) :
  (∀ n : ℕ, S n = 2^n + a) →
  a = -1 :=
sorry

end geo_seq_sum_l639_639586


namespace problem_proof_l639_639445

theorem problem_proof (a b c : ℝ) (h1 : 15 ^ a = 25) (h2 : 5 ^ b = 25) (h3 : 3 ^ c = 25) :
  1 / a + 1 / b - 1 / c = 1 :=
by
  sorry

end problem_proof_l639_639445


namespace train_pass_time_l639_639171

-- Define the given conditions
def train_length : ℝ := 200 -- meters
def train_speed_kmph : ℝ := 80 -- km per hour
def train_speed_mps : ℝ := train_speed_kmph * (1000 / 1) * (1 / 3600) -- Convert kmph to mps

-- State the theorem we want to prove
theorem train_pass_time : (train_length / train_speed_mps) = 9 :=
  by
  sorry

end train_pass_time_l639_639171


namespace area_of_shape_formed_by_P_l639_639810

-- Definition for the set of points P in the plane
def P_points (α : ℝ) : set (ℝ × ℝ) := 
  {P : ℝ × ℝ | let (x, y) := P in (x - 2 * real.cos α)^2 + (y - 2 * real.sin α)^2 = 16}

-- Statement of the problem as a Lean theorem
theorem area_of_shape_formed_by_P :
  let P_set := ⋃ α : ℝ, P_points α in
  ∃ (A : ℝ), A = 32 * real.pi ∧ area_of_shape P_set A :=
sorry

end area_of_shape_formed_by_P_l639_639810


namespace trains_crossing_time_l639_639147

/-- Conditions -/
def length_of_train : ℝ := 120
def time_to_cross_post_1 : ℝ := 9
def time_to_cross_post_2 : ℝ := 15
def speed_of_train_1 : ℝ := length_of_train / time_to_cross_post_1
def speed_of_train_2 : ℝ := length_of_train / time_to_cross_post_2
def relative_speed : ℝ := speed_of_train_1 + speed_of_train_2
def total_distance : ℝ := length_of_train + length_of_train

/-- Proof that the two trains will cross each other in approximately 11.25 seconds when travelling in opposite directions -/
theorem trains_crossing_time :
  total_distance / relative_speed ≈ 11.25 := 
sorry

end trains_crossing_time_l639_639147


namespace equilateral_triangle_square_ratio_l639_639733

theorem equilateral_triangle_square_ratio (t s : ℕ) (h_t : 3 * t = 12) (h_s : 4 * s = 12) :
  t / s = 4 / 3 := by
  sorry

end equilateral_triangle_square_ratio_l639_639733


namespace difference_of_areas_l639_639031

-- Definitions of the problem elements.
variables {A B C D F : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space F] 

-- Coordinates and essential lengths given:
variables (A B C F D : ℝ)
variables (ab : ℝ) (bc : ℝ) (af : ℝ)
variables [h_ab : ab = 6] [h_bc : bc = 8] [h_af : af = 10]

-- Declaring that the angles involved are right angles:
variables (angle_FAB angle_ABC : ℝ) [right_FAB : angle_FAB = pi / 2] [right_ABC : angle_ABC = pi / 2]
variables (angle_AFC angle_BFD : ℝ) [right_AFC : angle_AFC = pi / 2] [right_BFD : angle_BFD = pi / 2]

-- The target statement:
theorem difference_of_areas : (area_of_triangle A D F) - (area_of_triangle B D C) = 6 :=
sorry

end difference_of_areas_l639_639031


namespace fg_eq_gf_condition_l639_639052

/-- Definitions of the functions f and g --/
def f (m n c x : ℝ) : ℝ := m * x + n + c
def g (p q c x : ℝ) : ℝ := p * x + q + c

/-- The main theorem stating the equivalence of the condition for f(g(x)) = g(f(x)) --/
theorem fg_eq_gf_condition (m n p q c x : ℝ) :
  f m n c (g p q c x) = g p q c (f m n c x) ↔ n * (1 - p) - q * (1 - m) + c * (m - p) = 0 := by
  sorry

end fg_eq_gf_condition_l639_639052


namespace spy_kidnap_gentlemen_l639_639477

-- Define the structure of the clubs and gentlemen
structure Club where
  id : ℕ

structure Gentleman where
  id : ℕ

-- Define the conditions provided in the problem
variable (Clubs : Fin (10^10))
variable (Members : Fin 10)

-- Assume the property that for any two clubs there is a gentleman in both
axiom gentleman_in_both_clubs : ∀ (C1 C2 : Club), C1 ≠ C2 → ∃ (g : Gentleman), g ∈ Members C1 ∧ g ∈ Members C2

-- Prove the main statement
theorem spy_kidnap_gentlemen :
  ∃ (gents : Finset Gentleman), gents.card = 9 ∧ ∀ (C : Club), ∃ (g : Gentleman), g ∈ gents ∧ g ∈ Members C := by
  sorry

end spy_kidnap_gentlemen_l639_639477


namespace Rachel_plant_arrangement_l639_639082

-- We define Rachel's plants and lamps
inductive Plant : Type
| basil1
| basil2
| aloe
| cactus

inductive Lamp : Type
| white1
| white2
| red1
| red2

def arrangements (plants : List Plant) (lamps : List Lamp) : Nat :=
  -- This would be the function counting all valid arrangements
  -- I'm skipping the implementation
  sorry

def Rachel_arrangement_count : Nat :=
  arrangements [Plant.basil1, Plant.basil2, Plant.aloe, Plant.cactus]
                [Lamp.white1, Lamp.white2, Lamp.red1, Lamp.red2]

theorem Rachel_plant_arrangement : Rachel_arrangement_count = 22 := by
  sorry

end Rachel_plant_arrangement_l639_639082


namespace smaller_circle_radius_l639_639030

theorem smaller_circle_radius
  (radius_largest : ℝ)
  (h1 : radius_largest = 10)
  (aligned_circles : ℝ)
  (h2 : 4 * aligned_circles = 2 * radius_largest) :
  aligned_circles / 2 = 2.5 :=
by
  sorry

end smaller_circle_radius_l639_639030


namespace correct_option_B_l639_639665

theorem correct_option_B :
  (∀ (A : \frac{4}{5} * -\frac{5}{4} = 1) → False) ∧
  (∀ (C : -8 - 8 = 0) → False) ∧
  (∀ (D : -2 / -4 = 2) → False) →
  (-4^2 = -16) :=
by
  intro h
  exact h sorry

end correct_option_B_l639_639665


namespace painters_complete_three_rooms_in_three_hours_l639_639007

theorem painters_complete_three_rooms_in_three_hours :
  ∃ P, (∀ (P : ℕ), (P * 3) = 3) ∧ (9 * 9 = 27) → P = 3 := by
  sorry

end painters_complete_three_rooms_in_three_hours_l639_639007


namespace count_two_digit_primes_with_odd_tens_and_units_three_l639_639877

def is_two_digit_prime_with_odd_tens_and_units_three (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (n % 10 = 3) ∧ (odd (n / 10)) ∧ (Nat.Prime n)

theorem count_two_digit_primes_with_odd_tens_and_units_three :
  {n : ℕ // is_two_digit_prime_with_odd_tens_and_units_three n}.card = 3 :=
by
  sorry

end count_two_digit_primes_with_odd_tens_and_units_three_l639_639877


namespace median_line_equation_is_correct_l639_639377

structure Point :=
(x : ℝ)
(y : ℝ)

def midpoint (A B : Point) : Point :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2 }

def slope (P Q : Point) : ℝ :=
(P.y - Q.y) / (P.x - Q.x)

def line_equation (m : ℝ) (P : Point) : ℝ → ℝ := 
λ x, P.y + m * (x - P.x)

theorem median_line_equation_is_correct :
  let A := Point.mk 1 0,
      B := Point.mk 2 (-3),
      C := Point.mk 3 3,
      D := midpoint A B,
      k_CD := slope C D,
      line := line_equation k_CD C in
  ∀ x y : ℝ, y = line x ↔ 3 * x - y - 6 = 0 := 
by
  sorry

end median_line_equation_is_correct_l639_639377


namespace correct_triangle_statements_l639_639664

theorem correct_triangle_statements :
  (∀ (Δ : Type) [triangle Δ],
    ((∀ (A B C : angle Δ), A + B > C → A < 90 ∨ B < 90) ∧
     (∀ (median altitude bisector : segment Δ), is_line_segment median ∧ is_line_segment altitude ∧ is_line_segment bisector))) :=
by
  sorry

end correct_triangle_statements_l639_639664


namespace line_equation_l639_639315

variable (t : ℝ)
variable (x y : ℝ)

def param_x (t : ℝ) : ℝ := 3 * t + 2
def param_y (t : ℝ) : ℝ := 5 * t - 7

theorem line_equation :
  ∃ m b : ℝ, ∀ t : ℝ, y = param_y t ∧ x = param_x t → y = m * x + b := by
  use (5 / 3)
  use (-31 / 3)
  sorry

end line_equation_l639_639315


namespace number_of_odd_digits_in_base4_of_517_l639_639785

theorem number_of_odd_digits_in_base4_of_517 : 
  let base4_representation_of_517 := [2, 0, 0, 1, 1]
  let is_odd_digit (d : ℕ) := d = 1 ∨ d = 3
  (list.countp is_odd_digit base4_representation_of_517) = 3 := 
by 
  sorry

end number_of_odd_digits_in_base4_of_517_l639_639785


namespace paths_to_spell_amc9_l639_639905

theorem paths_to_spell_amc9 :
  let A_to_M := 4,
      M_to_C := 3,
      C_to_9 := 3 in
  1 * A_to_M * M_to_C * C_to_9 = 36 :=
by {
  let A_to_M := 4,
  let M_to_C := 3,
  let C_to_9 := 3,
  have h : 1 * A_to_M * M_to_C * C_to_9 = 36, by norm_num,
  exact h
}. затем sorry

end paths_to_spell_amc9_l639_639905


namespace smallest_positive_period_of_f_f_monotonically_decreasing_intervals_l639_639865

noncomputable def f (x : ℝ) : ℝ :=
  (sin (2 * x)) * (cos (3 * Real.pi / 4)) - (cos (2 * x)) * (sin (3 * Real.pi / 4))

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

theorem f_monotonically_decreasing_intervals (x : ℝ) (H : 0 ≤ x ∧ x ≤ Real.pi) :
  (∀ y ∈ set.Icc 0 (Real.pi / 8), -f.deriv y > 0)
  ∧ (∀ y ∈ set.Icc (5 * Real.pi / 8) Real.pi, -f.deriv y > 0) :=
sorry

end smallest_positive_period_of_f_f_monotonically_decreasing_intervals_l639_639865


namespace find_four_numbers_l639_639314

theorem find_four_numbers (a b c d : ℕ) 
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7)
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) :
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := 
  by
    sorry

end find_four_numbers_l639_639314


namespace used_decorations_l639_639073

-- Definitions based on the conditions
def num_boxes : ℕ := 4
def decorations_per_box : ℕ := 15
def decorations_given_away : ℕ := 25

-- Calculate the total number of decorations
def total_decorations : ℕ := num_boxes * decorations_per_box

-- Prove that Mrs. Jackson used 35 decorations
theorem used_decorations : total_decorations - decorations_given_away = 35 := by
  calc
    total_decorations - decorations_given_away = (num_boxes * decorations_per_box) - decorations_given_away : by rfl
    ... = (4 * 15) - 25 : by simp [num_boxes, decorations_per_box]
    ... = 60 - 25 : by rfl
    ... = 35 : by norm_num

#check used_decorations

end used_decorations_l639_639073


namespace find_a_l639_639011

noncomputable def f (x a : ℝ) : ℝ := x^3 + (x - a)^2

theorem find_a (a : ℝ) (h : ∀ x : ℝ, deriv (f x a) = 3 * x^2 + 2 * (x - a)) (local_min : deriv (f 2 a) = 0) : a = 8 :=
by
  sorry

end find_a_l639_639011


namespace sequence_properties_l639_639066

open Nat

theorem sequence_properties
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (hS : ∀ n : ℕ, n > 0 → 
    (S n)^2 - (↑n^2 + ↑n - 3) * S n - 3 * (↑n^2 + ↑n) = 0)
  (ha_pos : ∀ n : ℕ, n > 0 → a n > 0) :
  (a 1 = 2) ∧
  (∀ n : ℕ, n > 0 → a n = 2 * n) ∧
  (∀ n : ℕ, n > 0 →
    (∑ i in range n, 1 / (a (i + 1) * (a (i + 1) + 1))) < 1/3):
sorry

end sequence_properties_l639_639066


namespace more_than_half_millet_on_friday_l639_639553

def initial_seeds : ℝ := 1.0
def millet_ratio : ℝ := 0.3
def sunflower_ratio : ℝ := 0.7
def daily_addition (n : ℕ) : ℝ := 1 + 0.5 * (n - 1)

def seeds_after_eating (n : ℕ) (millet sunflower : ℝ) : ℝ × ℝ :=
  (0.7 * millet + millet_ratio * daily_addition n, 0.5 * sunflower + sunflower_ratio * daily_addition n)

noncomputable def seeds_on_day (n : ℕ) : ℝ × ℝ :=
nat.rec (0.3 * initial_seeds, 0.7 * initial_seeds) seeds_after_eating (n-1)

theorem more_than_half_millet_on_friday : (seeds_on_day 5).1 > 0.5 * (seeds_on_day 5).1 + (seeds_on_day 5).2 :=
sorry

end more_than_half_millet_on_friday_l639_639553


namespace sum_of_valid_b_values_of_quadratic_with_rational_roots_l639_639764

theorem sum_of_valid_b_values_of_quadratic_with_rational_roots :
  let b_set := {b : ℕ | ∃ k : ℕ, 49 - 12 * b = k^2 ∧ 49 > k^2}
  ∑ b in b_set, b = 6 :=
begin
  sorry
end

end sum_of_valid_b_values_of_quadratic_with_rational_roots_l639_639764


namespace find_number_l639_639163

theorem find_number (N : ℕ) : 
  (N % 13 = 11) ∧ (N % 17 = 9) ↔ N = 141 :=
by 
  sorry

end find_number_l639_639163


namespace hyperbola_eccentricity_l639_639826

theorem hyperbola_eccentricity (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) :
  let c := Real.sqrt (a^2 + b^2) in -- The distance from the center to a focus
  let F1 := (-c, 0) in
  let F2 := (c, 0) in
  let A := (x1, y1) in -- Intersection points with the hyperbola
  let B := (x2, y2) in
  (x1 - c)^2 + y1^2 = a^2 + b^2 ∧ -- Point A lies on the hyperbola
  (x2 - c)^2 + y2^2 = a^2 + b^2 ∧ -- Point B lies on the hyperbola
  ((A.1 - F2.1)^2 + (A.2 - F2.2)^2 = m^2) ∧ -- A and F2
  ((B.1 - F2.1)^2 + (B.2 - F2.2)^2 = (2*m)^2) ∧ -- B and F2
  ((A.1 - F1.1)^2 + (A.2 - F1.2)^2 = (2*a + m)^2) ∧ -- A and F1
  ((B.1 - F1.1)^2 + (B.2 - F1.2)^2 = (2*a + 2*m)^2) ∧ -- B and F1
  let e := c / a in
  e = Real.sqrt 17 / 3 := 
sorry

end hyperbola_eccentricity_l639_639826


namespace find_b_l639_639113

theorem find_b (a b : ℝ) (h1 : a + b = 60) (h2 : a = 3 * b) (h3 : ∀ (k : ℝ), a * b = k) (k : ℝ) :
  a = 12 → b = 56.25 :=
by
  have h4 : b = 15, 
  { sorry }
  have h5 : k = 675,
  { sorry }
  have h6 : 12 * b = 675,
  { sorry }
  exact sorry

end find_b_l639_639113


namespace num_triangles_with_perimeter_13_l639_639874

theorem num_triangles_with_perimeter_13 : 
  (∃ (a b c : ℕ), (a + b + c = 13) ∧ (a + b > c) ∧ (b + c > a) ∧ (c + a > b)) → 40 :=
by sorry

end num_triangles_with_perimeter_13_l639_639874


namespace average_speed_joey_round_trip_l639_639928

noncomputable def average_speed_round_trip
  (d : ℝ) (t₁ : ℝ) (r : ℝ) (s₂ : ℝ) : ℝ :=
  2 * d / (t₁ + d / s₂)

-- Lean statement for the proof problem
theorem average_speed_joey_round_trip :
  average_speed_round_trip 6 1 6 12 = 8 := sorry

end average_speed_joey_round_trip_l639_639928


namespace similar_triangles_heights_l639_639148

theorem similar_triangles_heights
  (h_similar: SimilarTriangles T1 T2)
  (h_area_ratio: AreaRatio T1 T2 = 1 / 9)
  (h_height_small: Height T1 = 5) :
  Height T2 = 15 :=
  sorry

end similar_triangles_heights_l639_639148


namespace find_a_and_m_l639_639390

theorem find_a_and_m : ∃ (a m : ℝ), 
  let k1 := -2 in
  let k2 := -a in
  let k3 := (m - 4) / (-2 - m) in
  (k1 * k2 = -1) ∧ (k1 = k3) ∧ (a = -0.5) ∧ (m = -8) :=
by {
  use [-0.5, -8],
  simp,
  sorry
}

end find_a_and_m_l639_639390


namespace average_age_increase_l639_639589

theorem average_age_increase 
  (average_age_students : ℕ)
  (number_of_students : ℕ)
  (teacher_age : ℕ)
  (average_age_with_teacher_include : ℕ)
  (increase_in_average_age : ℕ)
  (h1 : average_age_students = 10)
  (h2 : number_of_students = 15)
  (h3 : teacher_age = 26) :
  h4 : average_age_with_teacher_include = (average_age_students * number_of_students + teacher_age) / (number_of_students + 1) :=
begin
  sorry
end

end average_age_increase_l639_639589


namespace find_interest_rate_of_initial_investment_l639_639744

def initial_investment : ℝ := 1400
def additional_investment : ℝ := 700
def total_investment : ℝ := 2100
def additional_interest_rate : ℝ := 0.08
def target_total_income_rate : ℝ := 0.06
def target_total_income : ℝ := target_total_income_rate * total_investment

theorem find_interest_rate_of_initial_investment (r : ℝ) :
  (initial_investment * r + additional_investment * additional_interest_rate = target_total_income) → 
  (r = 0.05) :=
by
  sorry

end find_interest_rate_of_initial_investment_l639_639744


namespace sum_of_roots_l639_639543

theorem sum_of_roots (f : ℝ → ℝ) (h_symmetric : ∀ x, f (3 + x) = f (3 - x)) (h_roots : ∃ (roots : Finset ℝ), roots.card = 6 ∧ ∀ r ∈ roots, f r = 0) : 
  ∃ S, S = 18 :=
by
  sorry

end sum_of_roots_l639_639543


namespace cucumbers_for_24_apples_l639_639895

theorem cucumbers_for_24_apples :
  (∀ (a b c d apples bananas cucumbers : ℕ), (12 * a = 6 * b) → (3 * b = 4 * d) → (24 * a = n * cucumbers)) :=
by
  intros a b c d apples bananas cucumbers h1 h2
  have h3 : 2 * (12 * a) = 2 * (6 * b) := sorry
  have h4 : 24 * a = 12 * b := by
    rw [←h3]
    exact sorry
  have h5 : 4 * (3 * b) = 4 * (4 * d) := sorry
  have h6 : 12 * b = 16 * d := by
    rw [←h5]
    exact sorry
  have h7 : 24 * a = 16 * d := by
    rw [h4, h6]
    exact sorry
  exact eq.symm h7

end cucumbers_for_24_apples_l639_639895


namespace sum_of_functions_eq_x_l639_639891

-- Definitions based on conditions in a)
def f (x : ℝ) : ℝ := real.sqrt x
def g (x : ℝ) : ℝ := x - real.sqrt x

-- Lean statement for the proof problem in c)
theorem sum_of_functions_eq_x (x : ℝ) (hx : 0 ≤ x) : f x + g x = x :=
by
  unfold f g
  -- proof step skipped
  sorry

end sum_of_functions_eq_x_l639_639891


namespace smallest_tournament_with_ordered_group_l639_639467

-- Define the concept of a tennis tournament with n players
def tennis_tournament (n : ℕ) := 
  ∀ (i j : ℕ), (i < n) → (j < n) → (i ≠ j) → (i < j) ∨ (j < i)

-- Define what it means for a group of four players to be "ordered"
def ordered_group (p1 p2 p3 p4 : ℕ) : Prop := 
  ∃ (winner : ℕ), ∃ (loser : ℕ), 
    (winner ≠ loser) ∧ (winner = p1 ∨ winner = p2 ∨ winner = p3 ∨ winner = p4) ∧ 
    (loser = p1 ∨ loser = p2 ∨ loser = p3 ∨ loser = p4)

-- Prove that any tennis tournament with 8 players has an ordered group
theorem smallest_tournament_with_ordered_group : 
  ∀ (n : ℕ), ∀ (tournament : tennis_tournament n), 
    (n ≥ 8) → 
    (∃ (p1 p2 p3 p4 : ℕ), ordered_group p1 p2 p3 p4) :=
  by
  -- proof omitted
  sorry

end smallest_tournament_with_ordered_group_l639_639467


namespace unique_prime_value_l639_639943

def T : ℤ := 2161

theorem unique_prime_value :
  ∃ p : ℕ, (∃ n : ℤ, n^4 - 898 * n^2 + T - 2160 = p) ∧ Prime p ∧ (∀ q, (∃ n : ℤ, n^4 - 898 * n^2 + T - 2160 = q) → q = p) :=
  sorry

end unique_prime_value_l639_639943


namespace mountain_height_is_1700m_l639_639124

noncomputable def height_of_mountain (temp_base : ℝ) (temp_summit : ℝ) (rate_decrease : ℝ) : ℝ :=
  ((temp_base - temp_summit) / rate_decrease) * 100

theorem mountain_height_is_1700m :
  height_of_mountain 26 14.1 0.7 = 1700 :=
by
  sorry

end mountain_height_is_1700m_l639_639124


namespace nurse_total_hours_correct_l639_639118

def students_SchoolA := [26, 19, 20, 25, 30, 28]
def students_SchoolB := [22, 25, 18, 24, 27, 30]
def students_SchoolC := [24, 20, 22, 26, 28, 26]

def time_lice_check := 2
def time_vision_test := 2
def time_hearing_test := 3

def break_time_per_grade := 10
def lunch_break_time := 60

def travel_A_to_B := 10
def travel_B_to_C := 15

def calculate_total_hours 
  (students_A : List ℕ) 
  (students_B : List ℕ) 
  (students_C : List ℕ) 
  (time_lice : ℕ) 
  (time_vision : ℕ) 
  (time_hearing : ℕ) 
  (break_time_per_grade : ℕ) 
  (lunch_break_time : ℕ) 
  (travel_A_to_B : ℕ) 
  (travel_B_to_C : ℕ) 
: ℕ := 
  let total_students := (students_A.sum + students_B.sum + students_C.sum)
  let total_time_per_student := time_lice + time_vision + time_hearing
  let total_time_students := total_students * total_time_per_student
  let total_breaks_time := 3 * (5 * break_time_per_grade + lunch_break_time)
  let total_travel_time := travel_A_to_B + travel_B_to_C
  let total_time := total_time_students + total_breaks_time + total_travel_time
  total_time / 60

theorem nurse_total_hours_correct 
  : calculate_total_hours students_SchoolA students_SchoolB students_SchoolC time_lice_check 
    time_vision_test time_hearing_test break_time_per_grade lunch_break_time travel_A_to_B travel_B_to_C 
    = 57.25 := 
  sorry

end nurse_total_hours_correct_l639_639118


namespace blue_pill_cost_l639_639746

theorem blue_pill_cost :
  ∃ (y : ℝ), (∀ (d : ℝ), d = 45) ∧
  (∀ (b : ℝ) (r : ℝ), b = y ∧ r = y - 2) ∧
  ((21 : ℝ) * 45 = 945) ∧
  (b + r = 45) ∧
  y = 23.5 := 
by
  sorry

end blue_pill_cost_l639_639746


namespace sum_of_angles_of_solutions_l639_639120

theorem sum_of_angles_of_solutions : 
  ∀ (z : ℂ), z^5 = 32 * Complex.I → ∃ θs : Fin 5 → ℝ, 
  (∀ k, 0 ≤ θs k ∧ θs k < 360) ∧ (θs 0 + θs 1 + θs 2 + θs 3 + θs 4 = 810) :=
by
  sorry

end sum_of_angles_of_solutions_l639_639120


namespace count_integers_with_increasing_digits_l639_639421

theorem count_integers_with_increasing_digits :
  let count_integers := 
    ∑ second_digit in ({3, 4, 5} : Finset ℕ), 
      ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ second_digit}, 
        1 in

  count_integers = 15 :=
by
  have step1 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 3}, 1 = 6,
  { -- Explanation: If second digit is 3, third can be 4, 5, 6, 7, 8, 9 -> 6 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step2 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 4}, 1 = 5,
  { -- Explanation: If second digit is 4, third can be 5, 6, 7, 8, 9 -> 5 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have step3 : ∑ third_digit in ({4, 5, 6, 7, 8, 9} : Finset ℕ) \ {d | d ≤ 5}, 1 = 4,
  { -- Explanation: If second digit is 5, third can be 6, 7, 8, 9 -> 4 options
    rw [Finset.card_sdiff, Finset.card_singleton, Finset.card],
    simp only [Finset.filter_subset, Finset.card_singleton],
  },
  have count_integers := step1 + step2 + step3,
  simp only [count_integers, add_comm],
  exact eq.refl 15

end count_integers_with_increasing_digits_l639_639421


namespace new_bookstore_acquisition_l639_639686

theorem new_bookstore_acquisition (x : ℝ) 
  (h1 : (1 / 2) * x + (1 / 4) * x + 50 = x - 200) : x = 1000 :=
by {
  sorry
}

end new_bookstore_acquisition_l639_639686


namespace find_f_n_equals_n_l639_639527

def f : ℕ → ℕ := sorry
axiom functional_eq {m n : ℕ} : f (f m + f n) = m + n

theorem find_f_n_equals_n :
  ∀ (n : ℕ), f n = n :=
begin
  -- proof goes here
  sorry
end

end find_f_n_equals_n_l639_639527


namespace paint_quantity_l639_639801

variable (totalPaint : ℕ) (blueRatio greenRatio whiteRatio : ℕ)

theorem paint_quantity 
  (h_total_paint : totalPaint = 45)
  (h_ratio_blue : blueRatio = 5)
  (h_ratio_green : greenRatio = 3)
  (h_ratio_white : whiteRatio = 7) :
  let totalRatio := blueRatio + greenRatio + whiteRatio
  let partQuantity := totalPaint / totalRatio
  let bluePaint := blueRatio * partQuantity
  let greenPaint := greenRatio * partQuantity
  let whitePaint := whiteRatio * partQuantity
  bluePaint = 15 ∧ greenPaint = 9 ∧ whitePaint = 21 :=
by
  sorry

end paint_quantity_l639_639801


namespace power_mean_inequality_l639_639563

noncomputable def power_mean (α : ℝ) (x : list ℝ) : ℝ :=
  (list.sum (x.map (λ xi : ℝ, xi^α)) / x.length.to_real)^(1/α)

theorem power_mean_inequality (α β : ℝ) (x : list ℝ) :
  α < β → α * β ≠ 0 → power_mean α x ≤ power_mean β x :=
by
-- Proof goes here
sorry

end power_mean_inequality_l639_639563


namespace t_a_equals_neg_one_l639_639447

theorem t_a_equals_neg_one (a t : ℝ) (i : ℂ) (hi : i = complex.I) :
  a + i = (1 + 2 * i) * t * i → t + a = -1 :=
by intros h sorry

end t_a_equals_neg_one_l639_639447


namespace acute_isosceles_triangle_k_l639_639248

theorem acute_isosceles_triangle_k (ABC : Triangle) (circ : Circle)
  (D : Point)
  (h1 : ABC.angles.B = ABC.angles.C) -- Isosceles property
  (h2 : ∀ P ∈ circ, is_tangent B P circ) -- Tangent property through B
  (h3 : ∀ Q ∈ circ, is_tangent C Q circ) -- Tangent property through C
  (h4 : angle ABC.angles.B = 3 * angle D )
  (h5 : ∃ k, angle ABC.angles.A = k * π ) :
  ∃ k, k = 5 / 11 :=
by
  sorry

end acute_isosceles_triangle_k_l639_639248


namespace area_of_trapezium_l639_639777

-- Defining the lengths of the sides and the distance
def a : ℝ := 12  -- 12 cm
def b : ℝ := 16  -- 16 cm
def h : ℝ := 14  -- 14 cm

-- Statement that the area of the trapezium is 196 cm²
theorem area_of_trapezium : (1 / 2) * (a + b) * h = 196 :=
by
  sorry

end area_of_trapezium_l639_639777


namespace arithmetic_sequence_and_sum_of_terms_l639_639959

theorem arithmetic_sequence_and_sum_of_terms :
  (∃ (aₙ : ℕ → ℕ), -- There exists an arithmetic sequence aₙ
    (∃ d ≠ 0,      -- with a non-zero common difference d
      (9 * aₙ 5 = 45) ∧                        -- S₉ = 45, revealing a₅
      (aₙ 2) ^ 2 = (aₙ 1) * (aₙ 4)) ∧        -- a₁, a₂, a₄ form a geometric sequence
      ((∀ n, aₙ n = n) ∧                       -- General formula for aₙ
        ∀ bₙ Tₙ,                              -- For bₙ = 1 / (aₙ * aₙ(n+1))
          (∀ n, bₙ n = 1 / (aₙ n * aₙ(n+1))) → -- Definition of bₙ
          (∀ n, Tₙ n = (∑ i in finset.range n, bₙ i) → -- Sum definition
                 Tₙ n = n / (n+1))) sorry)      -- Tₙ = n/(n+1)

end arithmetic_sequence_and_sum_of_terms_l639_639959


namespace find_triples_count_l639_639796

theorem find_triples_count :
  ∃ (x y z n : ℕ), (x + y + z = 90) ∧ (x / n = y / (n + 1)) ∧ (y / (n + 1) = z / (n + 2)) ∧ (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (n > 0) :=
begin
  let count := 7,
  sorry -- Proof to be completed
end

end find_triples_count_l639_639796


namespace rectangle_area_l639_639607

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_area_l639_639607


namespace rectangle_area_l639_639622

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l639_639622


namespace Liza_reads_more_pages_than_Suzie_l639_639070

def Liza_reading_speed : ℕ := 20
def Suzie_reading_speed : ℕ := 15
def hours : ℕ := 3

theorem Liza_reads_more_pages_than_Suzie :
  Liza_reading_speed * hours - Suzie_reading_speed * hours = 15 := by
  sorry

end Liza_reads_more_pages_than_Suzie_l639_639070


namespace triangle_ABC_A_is_30_degree_l639_639454

noncomputable def find_angle_A
  (a b : ℝ) (cos_B : ℝ) (h_a : a = 3) (h_b : b = 24 / 5) (h_cos_B : cos_B = 3 / 5) : ℝ :=
if h : angle B < π / 2
then
  let sin_B := real.sqrt (1 - cos_B ^ 2) in
  let sin_A := a * sin_B / b in
  real.arcsin sin_A
else
  0 -- angle B is not acute in this case, which contradicts our initial condition

theorem triangle_ABC_A_is_30_degree
  (a b : ℝ) (cos_B : ℝ) (h_a : a = 3) (h_b : b = 24 / 5) (h_cos_B : cos_B = 3 / 5) :
  find_angle_A a b cos_B h_a h_b h_cos_B = π / 6 :=
sorry

end triangle_ABC_A_is_30_degree_l639_639454


namespace prime_square_minus_one_divisible_by_24_l639_639992

theorem prime_square_minus_one_divisible_by_24 (n : ℕ) (h_prime : Prime n) (h_n_neq_2 : n ≠ 2) (h_n_neq_3 : n ≠ 3) : 24 ∣ (n^2 - 1) :=
sorry

end prime_square_minus_one_divisible_by_24_l639_639992


namespace find_number_l639_639164

theorem find_number (N : ℕ) : 
  (N % 13 = 11) ∧ (N % 17 = 9) ↔ N = 141 :=
by 
  sorry

end find_number_l639_639164


namespace goods_train_speed_l639_639692

variable (length_platform : ℝ) (length_train : ℝ) (time_cross_platform : ℝ)

def total_distance_covered := length_train + length_platform
def speed_train := total_distance_covered / time_cross_platform

theorem goods_train_speed (h1 : length_platform = 250)
                           (h2 : length_train = 270.0416)
                           (h3 : time_cross_platform = 26) :
  abs (speed_train length_platform length_train time_cross_platform - 20) < 1 :=
by
  unfold total_distance_covered
  unfold speed_train
  rw [h1, h2, h3]
  norm_num
  sorry

end goods_train_speed_l639_639692


namespace meadow_total_money_l639_639983

/-
  Meadow orders 30 boxes of diapers weekly.
  Each box contains 40 packs.
  Each pack contains 160 diapers.
  Meadow sells each diaper for $5.
  Prove that the total money Meadow makes from selling all her diapers is $960000.
-/

theorem meadow_total_money :
  let number_of_boxes := 30 in
  let packs_per_box := 40 in
  let diapers_per_pack := 160 in
  let price_per_diaper := 5 in
  let total_packs := number_of_boxes * packs_per_box in
  let total_diapers := total_packs * diapers_per_pack in
  let total_money := total_diapers * price_per_diaper in
  total_money = 960000 :=
by
  sorry

end meadow_total_money_l639_639983


namespace sum_of_edges_l639_639638

theorem sum_of_edges (a r : ℝ) 
  (h_vol : (a / r) * a * (a * r) = 432) 
  (h_surf_area : 2 * ((a * a) / r + (a * a) * r + a * a) = 384) 
  (h_geom_prog : r ≠ 1) :
  4 * ((6 * Real.sqrt 2) / r + 6 * Real.sqrt 2 + (6 * Real.sqrt 2) * r) = 72 * (Real.sqrt 2) := 
sorry

end sum_of_edges_l639_639638


namespace proj_v_w_l639_639326

def v : ℝ × ℝ := (3, 4)
def w : ℝ × ℝ := (1, 2)

def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2
  let scalar := (dot_product v w) / (dot_product w w)
  (scalar * w.1, scalar * w.2)

theorem proj_v_w : proj v w = (11 / 5, 22 / 5) := 
by sorry

end proj_v_w_l639_639326


namespace geometric_seq_sum_sequence_l639_639065

def seq_a_n (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop := 
  a 1 = 1 ∧ (∀ n : ℕ, n > 0 → n * (a (n + 1)) = (n + 2) * (S n))

def seq_S_n (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = (∑ k in Finset.range n, a (k + 1))

theorem geometric_seq {a S : ℕ → ℕ} (h1 : seq_a_n a S) (h2 : seq_S_n S a) :
  ∃ q : ℕ, (∀ n : ℕ, n > 0 → (S n) / n = q^(n-1)) :=
by 
  sorry

theorem sum_sequence {a S : ℕ → ℕ} (h1 : seq_a_n a S) (h2 : seq_S_n S a) :
  ∀ n : ℕ, n > 0 → (∑ k in Finset.range n, S (k + 1)) = (n-1) * 2^n + 1 :=
by 
  sorry

end geometric_seq_sum_sequence_l639_639065


namespace find_k_in_isosceles_triangle_l639_639226

theorem find_k_in_isosceles_triangle 
  (A B C D : Type)
  (h_triangle : triangle ABC)
  (h_acute : acute ABC)
  (h_isosceles : isosceles ABC)
  (h_circumscribed : circumscribed ABC)
  (h_tangents : tangents B C D)
  (h_angle_relation : ∠ABC = ∠ACB ∧ ∠ABC = 3 * ∠D)
  : ∠BAC = (5 * π / 11) := 
by 
  sorry

end find_k_in_isosceles_triangle_l639_639226


namespace problem_gets_solved_prob_l639_639630

-- Define conditions for probabilities
def P_A_solves := 2 / 3
def P_B_solves := 3 / 4

-- Calculate the probability that the problem is solved
theorem problem_gets_solved_prob :
  let P_A_not_solves := 1 - P_A_solves
  let P_B_not_solves := 1 - P_B_solves
  let P_both_not_solve := P_A_not_solves * P_B_not_solves
  let P_solved := 1 - P_both_not_solve
  P_solved = 11 / 12 :=
by
  -- Skip proof
  sorry

end problem_gets_solved_prob_l639_639630


namespace tom_total_spent_l639_639647

theorem tom_total_spent 
  (football_game_original_price : ℝ := 16)
  (football_game_discount_percent : ℝ := 0.10)
  (strategy_game_original_price : ℝ := 9.46)
  (strategy_game_sales_tax_percent : ℝ := 0.05)
  (batman_game_price_eur : ℝ := 11)
  (exchange_rate_usd_per_eur : ℝ := 1.15) 
  : 
  let football_game_price_after_discount := football_game_original_price * (1 - football_game_discount_percent),
      strategy_game_price_after_tax := (strategy_game_original_price * (1 + strategy_game_sales_tax_percent)).roundDigits 2,
      batman_game_price_usd := batman_game_price_eur * exchange_rate_usd_per_eur,
      total_amount_spent := football_game_price_after_discount + strategy_game_price_after_tax + batman_game_price_usd
  in total_amount_spent = 36.98 :=
by
  sorry

end tom_total_spent_l639_639647


namespace smallest_integer_m_l639_639545

def l₁ := (Real.cos (Real.pi / 60), Real.sin (Real.pi / 60))
def l₂ := (Real.cos (Real.pi / 45), Real.sin (Real.pi / 45))

def reflect_line (angle l₁: Real) : Real :=
  2 * l₁ - angle

def R (angle: Real): Real :=
  let refl_l1 := reflect_line angle (Real.pi / 60)
  reflect_line refl_l1 (Real.pi / 45)

def angle_l := Real.atan 1/3

noncomputable def Rₙ : ℕ → Real
| 0     := angle_l
| (n+1) := R (Rₙ n)

theorem smallest_integer_m :
  ∃ m : ℕ, m > 0 ∧ Rₙ m = angle_l ∧ m = 180 :=
by {
  sorry
}

end smallest_integer_m_l639_639545


namespace problem_equivalent_l639_639178

theorem problem_equivalent (
  AB BC CD DA E F B' C' : ℝ
  (h1 : ∀ A B C D : ℝ, A B B' C' forms_rectangle)
  (h2 : BE < CF)
  (h3 : is_folding_point E F B C)
  (h4 : C_folds_to C' AD)
  (h5 : B_folds_to B' AD)
  (h6 : angle AB'C' = angle B'EA)
  (h7 : AB' = 7)
  (h8 : BE = 17)
) : p = 28 ∧ q = 333 ∧ r = 153 ∧ p + q + r = 514 :=
by
  unfold AB BC CD DA E F B' C' BE CF C_folds_to B_folds_to
  unfold is_folding_point
  sorry

end problem_equivalent_l639_639178


namespace ellipse_major_axis_length_l639_639732

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def major_axis_length (f1 f2 : ℝ × ℝ) : ℝ :=
  2 * distance (f1.1 * -1, f1.2) f2

theorem ellipse_major_axis_length :
  major_axis_length (10, 25) (50, 65) = 10 * real.sqrt 117 :=
by
  sorry

end ellipse_major_axis_length_l639_639732


namespace find_real_pairs_l639_639974

theorem find_real_pairs :
    ∃ X : Finset (ℝ × ℝ), X.card = 2 ∧ 
    ∀ (x, y) ∈ X, 
        (y = x^2 + 2 * x + 1) ∧ 
        (x^2 * y + x = 2) :=
by
    sorry

end find_real_pairs_l639_639974


namespace christopher_age_l639_639265

variables (C G : ℕ)

theorem christopher_age :
  (C = 2 * G) ∧ (C - 9 = 5 * (G - 9)) → C = 24 :=
by
  intro h
  sorry

end christopher_age_l639_639265


namespace deposit_on_Jan_1_2008_l639_639075

-- Let a be the initial deposit amount in yuan.
-- Let x be the annual interest rate.

def compound_interest (a : ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  a * (1 + x) ^ n

theorem deposit_on_Jan_1_2008 (a : ℝ) (x : ℝ) : 
  compound_interest a x 5 = a * (1 + x) ^ 5 := 
by
  sorry

end deposit_on_Jan_1_2008_l639_639075


namespace nearest_integer_to_expansion_l639_639155

theorem nearest_integer_to_expansion : 
  let n := (3 + 2)^6 in 
  n = 9794 :=
by
  sorry

end nearest_integer_to_expansion_l639_639155


namespace find_number_l639_639161

noncomputable def N : ℕ :=
  76

theorem find_number :
  (N % 13 = 11) ∧ (N % 17 = 9) :=
by
  -- These are the conditions translated to Lean 4, as stated:
  have h1 : N % 13 = 11 := by sorry
  have h2 : N % 17 = 9 := by sorry
  exact ⟨h1, h2⟩

end find_number_l639_639161


namespace compare_f_values_l639_639847

def f (x : ℝ) : ℝ := x^2 - Real.cos x

theorem compare_f_values :
  f 0 < f (-0.5) ∧ f (-0.5) < f 0.6 :=
by {
  -- The proof would go here
  sorry
}

end compare_f_values_l639_639847


namespace guests_did_not_respond_l639_639085

theorem guests_did_not_respond
  (total_guests : ℕ)
  (yes_percentage : ℝ)
  (no_percentage : ℝ)
  (yes_responses : ℕ)
  (no_responses : ℕ)
  (did_not_respond : ℕ)
  (h_yes_responses : yes_responses = (yes_percentage * total_guests).to_nat)
  (h_no_responses : no_responses = (no_percentage * total_guests).to_nat)
  (h_total : total_guests = 200)
  (h_yes_percentage : yes_percentage = 0.83)
  (h_no_percentage : no_percentage = 0.09) :
  did_not_respond = total_guests - (yes_responses + no_responses) := sorry

end guests_did_not_respond_l639_639085


namespace max_knights_neighbors_l639_639258

def round_table_statement : Prop :=
  ∀ (people : Fin 10 → Bool), (0 < people.count(λ p, p = true)) ∧ (0 < people.count(λ p, p = false)) →
  (∃ (count : Nat), count ≤ 10 ∧ (
    count = 9 ∧ (∃ liar, ∀ i : Fin 10, if people i then
      (people ((i + 1) % 10) = true ∧ people ((i + 9) % 10) = true)
      else (people ((i + 1) % 10) = false ∨ people ((i + 9) % 10) = false)) ∧
    (∀ (total : ℕ), (∃ n : ℕ, n = total) →
      (total > count → (total = 10 → false) ∨
      (∃ liar_conf : Fin 10, (people liar_conf = false ∧
      (people ((liar_conf + 1) % 10) = false ∨
        people ((liar_conf + 9) % 10) = false)))))
  ))

theorem max_knights_neighbors : round_table_statement :=
sorry

end max_knights_neighbors_l639_639258


namespace rectangle_area_l639_639604

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_area_l639_639604


namespace length_of_RU_l639_639924

theorem length_of_RU
  (P Q R S T U : Point)
  (PQ QR RP : ℝ)
  (PQ_dist : dist P Q = 13)
  (QR_dist : dist Q R = 20)
  (RP_dist : dist R P = 15)
  (angle_bisector : is_angle_bisector P Q R S)
  (T_on_circumcircle : T ≠ P ∧ T ∈ circumcircle (triangle P Q R))
  (U_on_circumcircle : P ≠ U ∧ U ∈ circumcircle (triangle P S T))
  (U_on_PQ : collinear P U Q) :
  dist R U = 20 :=
sorry

end length_of_RU_l639_639924


namespace find_k_l639_639220

-- Definitions based on conditions in step a)
def acute_isosceles_triangle_inscribed (A B C : Type) : Prop := sorry -- Formal definition of the triangle being acute isosceles and inscribed in a circle
def tangents_meeting_at_point (A B C D : Type) : Prop := sorry -- Formal definition of tangents through B and C meeting at D
def angle_relation (ABC D : Type) (theta : ℝ) : Prop := 3 * theta = sorry -- Formal definition of \(\angle ABC = \angle ACB = 3 \angle D\)
def angle_BAC (k : ℝ) (theta : ℝ) : Prop := theta = k * real.pi -- Formal definition of \(\angle BAC = k \pi\)

-- Theorem statement for our proof problem
theorem find_k
  (A B C D : Type)
  (h1 : acute_isosceles_triangle_inscribed A B C)
  (h2 : tangents_meeting_at_point A B C D)
  (theta : ℝ)
  (h3 : angle_relation ABC D theta)
  (k : ℝ)
  (h4 : angle_BAC k theta) :
  k = 1 / 13 := by
  sorry

end find_k_l639_639220


namespace complement_union_correct_l639_639863

-- Define the universal set U, set A, and set B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- The complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The union of the complement of A and set B
def union_complement_U_A_B : Set ℕ := complement_U_A ∪ B

-- State the theorem to prove
theorem complement_union_correct : union_complement_U_A_B = {2, 3, 4, 5} := 
by 
  sorry

end complement_union_correct_l639_639863


namespace probability_sum_odd_set_12345_l639_639799

theorem probability_sum_odd_set_12345 :
  let S := {1, 2, 3, 4, 5} in
  let total_pairs := Finset.card (Finset.pairs S) in
  let odd_even_pairs := S.card.choose 3 * S.card.choose 2 in
  (total_pairs ≠ 0) →
  (odd_even_pairs / total_pairs : ℚ) = 3 / 5 :=
by
  let S := {1, 2, 3, 4, 5}
  let total_pairs := Finset.card (Finset.pairs S)
  let odd_even_pairs := S.card.choose 3 * S.card.choose 2
  have h : total_pairs ≠ 0, sorry -- Proof here
  show (odd_even_pairs / total_pairs : ℚ) = 3 / 5, sorry -- Proof here

end probability_sum_odd_set_12345_l639_639799


namespace jerry_initial_candy_l639_639038

theorem jerry_initial_candy
  (total_bags : ℕ)
  (chocolate_hearts_bags : ℕ)
  (chocolate_kisses_bags : ℕ)
  (nonchocolate_pieces : ℕ)
  (remaining_bags : ℕ)
  (pieces_per_bag : ℕ)
  (initial_candy : ℕ)
  (h_total_bags : total_bags = 9)
  (h_chocolate_hearts_bags : chocolate_hearts_bags = 2)
  (h_chocolate_kisses_bags : chocolate_kisses_bags = 3)
  (h_nonchocolate_pieces : nonchocolate_pieces = 28)
  (h_remaining_bags : remaining_bags = total_bags - chocolate_hearts_bags - chocolate_kisses_bags)
  (h_pieces_per_bag : pieces_per_bag = nonchocolate_pieces / remaining_bags)
  (h_initial_candy : initial_candy = total_bags * pieces_per_bag) :
  initial_candy = 63 := by
  sorry

end jerry_initial_candy_l639_639038


namespace problem_l639_639805

theorem problem (a : ℕ → ℕ)(n : ℕ):
  (∀ i j : ℕ, i ≠ j → i < 1999 ∧ j < 1999 → (a i - a j) ∣ (a i + a j)) →
  (n = ∏ k in finset.range 1999, a k) →
  (∀ i j : ℕ, i ≠ j → i ≤ 1999 ∧ j ≤ 1999 → ((if i = 0 then n else n + a (i-1)) - (if j = 0 then n else n + a (j-1))) ∣ ((if i = 0 then n else n + a (i-1)) + (if j = 0 then n else n + a (j-1)))) :=
by
  sorry

end problem_l639_639805


namespace intersection_A_B_union_A_complement_B_l639_639861

open Set

noncomputable def U : Set ℝ := Univ

def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}

def B : Set ℝ := {x | (x^2 - 2 * x - 3) ≥ 0}

def complement_B : Set ℝ := U \ B

-- Proving A ∩ B = {x | -5 < x ≤ -1}
theorem intersection_A_B :
  { x | -5 < x ∧ x ≤ -1 } = A ∩ B := by
  sorry

-- Proving A ∪ (complement_U B) = {x | -5 < x < 3}
theorem union_A_complement_B :
  { x | -5 < x ∧ x < 3 } = A ∪ complement_B := by
  sorry

end intersection_A_B_union_A_complement_B_l639_639861


namespace min_value_of_expression_l639_639965

theorem min_value_of_expression (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x + y + z = 5) : 
  (9 / x + 16 / y + 25 / z) ≥ 28.8 :=
by sorry

end min_value_of_expression_l639_639965


namespace prod_sum_leq_four_l639_639536

theorem prod_sum_leq_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 4) :
  ab + bc + cd + da ≤ 4 :=
sorry

end prod_sum_leq_four_l639_639536


namespace product_of_2019_smallest_primes_in_sequence_l639_639756

/-- Definition of the sequence based on given conditions --/
def sequence (n : ℕ) : ℕ
| 0 => 2
| (n + 1) => sequence n + (nat.prod (nat.factors (sequence n)))

/-- One property we are interested in --/
def product_of_first_n_primes (n : ℕ) : ℕ :=
  nat.prod (finset.range n).1.map (λ k, nat.prime.seq k).val

/-- The theorem to be proven --/
theorem product_of_2019_smallest_primes_in_sequence : 
  ∃ n : ℕ, sequence n = product_of_first_n_primes 2019 := by
  sorry

end product_of_2019_smallest_primes_in_sequence_l639_639756


namespace length_MN_l639_639741

theorem length_MN {A B C D M N : ℝ} 
  (hm : M = (A + C) / 2) (hn : N = (D + B) / 2) (hAB : B - A = 10) (hCD : D - C = 2) : 
  (M - N).abs = 6 := 
by 
  sorry

end length_MN_l639_639741


namespace part_a_part_b_l639_639359

variables (A B C D: Type) 

def is_midpoint (B C D : Type) : Prop := sorry

def angle (X Y Z : Type) : angle := sorry

theorem part_a (h_midpoint: is_midpoint B C D)
  (h_angleB: angle A B C = 105)
  (h_angleBDA: angle B D A = 45) :
  angle A C B = 30 :=
sorry

theorem part_b (h_angleC: angle A C B = 30)
  (h_angleB: angle A B C = 105)
  (h_angleBDA: angle B D A = 45) :
  is_midpoint B C D :=
sorry

end part_a_part_b_l639_639359


namespace beri_always_wins_l639_639255

theorem beri_always_wins :
  ∀ (a : ℕ) (ha : 1 ≤ a ∧ a ≤ 2020),
  ∃ (b : ℕ), (b = a + 1 ∨ b = a - 1) ∧
  ((∃ (x : ℤ), x^2 - x * a + b = 0) ∨ (∃ (y : ℤ), y^2 - y * b + a = 0)) :=
by
  intros a ha
  have h1 : 1 < a → a < 2020 → ∃ b, (b = a + 1 ∨ b = a - 1)
    sorry
  have h2 : (a = 1 ∨ a = 2020) → (∃ b, b = a + 1 ∨ b = a - 1)
    sorry
  cases ha with hA hB
  cases
    case hA :  a = 1
    apply h2 hA
    sorry
  case hB : a = 2020
    apply h2 hB
    sorry
  case  _ : 1 < a < 2020
    apply h1
    sorry  

end beri_always_wins_l639_639255


namespace probability_A_B_different_groups_l639_639093

theorem probability_A_B_different_groups :
  let n := 20
  let m := 12
  let p := (m : ℚ) / n
  p = (3 / 5 : ℚ) := 
by
  sorry

end probability_A_B_different_groups_l639_639093


namespace intersection_complement_l639_639864

open Set

def U : Set ℝ := univ
def A : Set ℤ := {x : ℤ | -3 < x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement (U A B : Set ℝ) : A ∩ (U \ B) = {0, 1} := sorry

end intersection_complement_l639_639864


namespace bisectors_perpendicular_intersect_midline_l639_639973

variables {A B C D K L M N : Point}

-- Let ABCD be a cyclic quadrilateral
axiom cyclic_quadrilateral (A B C D : Point) : IsCyclicQuadrilateral A B C D

-- Points K and L are the intersections of extensions of opposite sides
axiom intersect_opposite_sides (A B C D K L : Point) :
  AreIntersecting (line A B) (line C D) K ∧ AreIntersecting (line B C) (line A D) L

-- M is the midpoint of AC and N is the midpoint of BD
axiom midpoints (A C B D M N : Point) : Midpoint A C M ∧ Midpoint B D N

-- Question to prove: Bisectors of angles ∠BKC and ∠BLA are perpendicular and intersect on the line connecting M and N
theorem bisectors_perpendicular_intersect_midline (A B C D K L M N : Point) 
  [cyclic_quadrilateral A B C D] [intersect_opposite_sides A B C D K L] [midpoints A C B D M N] :
  ArePerpendicular (angle_bisector (angle B K C)) (angle_bisector (angle B L A)) ∧
  IsLineConnecting M N (intersection (angle_bisector (angle B K C)) (angle_bisector (angle B L A))) :=
sorry

end bisectors_perpendicular_intersect_midline_l639_639973


namespace problem1_problem2_l639_639393

-- Definitions for the sets A and B
def setA (a : ℝ) : set ℝ := {x | a - 1 < x ∧ x < a + 1}
def setB : set ℝ := {x | 0 < x ∧ x < 3}

-- Problem 1: If a = 0, prove A ∩ B = { x | 0 < x < 1 }
theorem problem1 (a : ℝ) (h : a = 0) : setA a ∩ setB = { x | 0 < x ∧ x < 1 } :=
by sorry

-- Problem 2: If A ⊆ B, prove 1 ≤ a ≤ 2
theorem problem2 (a : ℝ) (h : setA a ⊆ setB) : 1 ≤ a ∧ a ≤ 2 :=
by sorry

end problem1_problem2_l639_639393


namespace max_points_on_ellipse_l639_639356

theorem max_points_on_ellipse (a b : ℝ) (h1 : a = 2 * Real.sqrt 2) 
  (h2 : b = Real.sqrt 6) (d : ℝ) (h : d > 1 / 5) :
  ∃ n : ℕ, n ≤ 15 :=
by
  let c := Real.sqrt (a^2 - b^2)
  have min_dist := a - c
  have max_dist := a + c
  have sequence_range := max_dist - min_dist
  have bound := Real.sqrt 2
  have inequality : sequence_range / (d * bound) ≤ (15 - 1) :=
    sorry -- Detailed proof omitted
  use 15
  exact inequality

end max_points_on_ellipse_l639_639356


namespace p_eval_at_neg_one_l639_639517

noncomputable def p (x : ℝ) : ℝ :=
  x^2 - 2*x + 9

theorem p_eval_at_neg_one : p (-1) = 12 := by
  sorry

end p_eval_at_neg_one_l639_639517


namespace uncle_zhang_age_l639_639653

theorem uncle_zhang_age (z l : ℕ) (h1 : z + l = 56) (h2 : z = l - (l / 2)) : z = 24 :=
by sorry

end uncle_zhang_age_l639_639653


namespace total_money_l639_639977

theorem total_money (m c : ℝ) (hm : m = 5 / 8) (hc : c = 7 / 20) : m + c = 0.975 := sorry

end total_money_l639_639977


namespace probability_average_is_five_l639_639800

-- Definitions and conditions
def numbers : List ℕ := [1, 3, 4, 6, 7, 9]

def average_is_five (a b : ℕ) : Prop := (a + b) / 2 = 5

-- Desired statement
theorem probability_average_is_five : 
  ∃ p : ℚ, p = 1 / 5 ∧ (∃ a b : ℕ, a ∈ numbers ∧ b ∈ numbers ∧ average_is_five a b) := 
sorry

end probability_average_is_five_l639_639800


namespace molecular_weight_neutralization_l639_639658

def molecular_weight_acetic_acid : ℝ := 
  (12.01 * 2) + (1.008 * 4) + (16.00 * 2)

def molecular_weight_sodium_hydroxide : ℝ := 
  22.99 + 16.00 + 1.008

def total_weight_acetic_acid (moles : ℝ) : ℝ := 
  molecular_weight_acetic_acid * moles

def total_weight_sodium_hydroxide (moles : ℝ) : ℝ := 
  molecular_weight_sodium_hydroxide * moles

def total_molecular_weight (moles_ac: ℝ) (moles_naoh : ℝ) : ℝ :=
  total_weight_acetic_acid moles_ac + 
  total_weight_sodium_hydroxide moles_naoh

theorem molecular_weight_neutralization :
  total_molecular_weight 7 10 = 820.344 :=
by
  sorry

end molecular_weight_neutralization_l639_639658


namespace P_is_integer_for_all_n_l639_639500

noncomputable def P (n : ℕ) (x y : ℝ) : ℝ := (x ^ n - y ^ n) / (x - y)

theorem P_is_integer_for_all_n (x y : ℝ) (k : ℕ) :
  x ≠ y →
  (∀ i : ℕ, k ≤ i ∧ i < k + 4 → (P i x y).denom = 1) →
  ∀ n : ℕ, (P n x y).denom = 1 :=
by
  intros hxy consec_int_conds n
  sorry

end P_is_integer_for_all_n_l639_639500


namespace customer_cost_bound_l639_639592

theorem customer_cost_bound (
  h_cake_min : 1.80 <= cake <= 2.40,
  h_milk_tea_min : 1.20 <= milk_tea <= 3.00,
  h_chocolate_brownie_min : 2.00 <= chocolate_brownie <= 3.60,
  h_cake_discount : discount_cake = 12/100,
  h_milk_tea_discount : discount_milk_tea = 8/100,
  h_chocolate_brownie_tax : tax_chocolate_brownie = 15/100
) :
  let cost_least := 5 * 1.80 * (1 - 12/100) + 3 * 1.20 * (1 - 8/100) + 2 * 2.00 * (1 + 15/100) in
  let cost_greatest := 5 * 2.40 * (1 - 12/100) + 3 * 3.00 * (1 - 8/100) + 2 * 3.60 * (1 + 15/100) in
  15.832 <= cost_least ∧ cost_greatest <= 27.12 :=
by
  sorry

end customer_cost_bound_l639_639592


namespace find_k_in_isosceles_triangle_l639_639227

theorem find_k_in_isosceles_triangle 
  (A B C D : Type)
  (h_triangle : triangle ABC)
  (h_acute : acute ABC)
  (h_isosceles : isosceles ABC)
  (h_circumscribed : circumscribed ABC)
  (h_tangents : tangents B C D)
  (h_angle_relation : ∠ABC = ∠ACB ∧ ∠ABC = 3 * ∠D)
  : ∠BAC = (5 * π / 11) := 
by 
  sorry

end find_k_in_isosceles_triangle_l639_639227


namespace problem_proof_l639_639330

theorem problem_proof (c d : ℕ) (h1 : c > 0) (h2 : d > 0)
    (h3 : (∀ i, c ≤ i ∧ i < d → log (i+1) / log i = 1) ∧ log ((d+1) / c) = 3)
    (h4 : d + 1 - c + 1 = 1000) : c + d = 1009 :=
sorry

end problem_proof_l639_639330


namespace limo_gas_price_l639_639404

theorem limo_gas_price
  (hourly_wage : ℕ := 15)
  (ride_payment : ℕ := 5)
  (review_bonus : ℕ := 20)
  (hours_worked : ℕ := 8)
  (rides_given : ℕ := 3)
  (gallons_gas : ℕ := 17)
  (good_reviews : ℕ := 2)
  (total_owed : ℕ := 226) :
  total_owed = (hours_worked * hourly_wage) + (rides_given * ride_payment) + (good_reviews * review_bonus) + (gallons_gas * 3) :=
by
  sorry

end limo_gas_price_l639_639404


namespace value_of_s_l639_639291

-- Define the variables as integers (they represent non-zero digits)
variables {a p v e s r : ℕ}

-- Define the conditions as hypotheses
theorem value_of_s (h1 : a + p = v) (h2 : v + e = s) (h3 : s + a = r) (h4 : p + e + r = 14) :
  s = 7 :=
by
  sorry

end value_of_s_l639_639291


namespace number_of_ellipses_with_eccentricity_log_l639_639628

noncomputable def different_shapes_of_ellipses (p q : ℕ) : ℕ :=
  if h : 2 ≤ q ∧ q < p ∧ p ≤ 9 then 1 else 0

theorem number_of_ellipses_with_eccentricity_log {p q : ℕ} :
  (Σ' (p q : ℕ), different_shapes_of_ellipses p q) = 26 :=
  sorry

end number_of_ellipses_with_eccentricity_log_l639_639628


namespace intersects_at_7_98_l639_639374

noncomputable def h : ℝ → ℝ := sorry
noncomputable def j : ℝ → ℝ := sorry

theorem intersects_at_7_98 :
  h 1 = 1 ∧ h 3 = 9 ∧ h 5 = 25 ∧ h 7 = 49 ∧
  j 1 = 1 ∧ j 3 = 9 ∧ j 5 = 25 ∧ j 7 = 49 →
  ∃ a b, h (2 * a) = b ∧ 2 * j a = b ∧ a = 7 ∧ b = 98 :=
by
  intros h1 h3 h5 h7 j1 j3 j5 j7
  use 7
  use 98
  split
  · exact h7
  split
  · exact eq.symm (congr_arg (2 * ·) j7)
  · split
  · exact rfl
  · exact rfl
  sorry

end intersects_at_7_98_l639_639374


namespace lines_intersect_at_one_point_l639_639816

noncomputable section

-- Let triangle ABC
variables {A B C : Type} [euclidean_geometry A B C]

-- A_1, B_1, C_1 are midpoints of sides BC, AC, and AB respectively
variables {A1 B1 C1 : A}
variable [is_midpoint A1 B C]
variable [is_midpoint B1 A C]
variable [is_midpoint C1 A B]

-- Lines a, b, c passing through A1, B1, C1 respectively and parallel to angle bisectors of opposite angles
variables {a b c : line A}
variable [a_passing_through_A1 : passes_through A1 a]
variable [b_passing_through_B1 : passes_through B1 b]
variable [c_passing_through_C1 : passes_through C1 c]
variable [a_parallel_to_angle_bisector_A : parallel_to_angle_bisector a]
variable [b_parallel_to_angle_bisector_B : parallel_to_angle_bisector b]
variable [c_parallel_to_angle_bisector_C : parallel_to_angle_bisector c]

-- Prove these lines intersect at a single point
theorem lines_intersect_at_one_point :
    intersects_at_single_point a b c := 
sorry

end lines_intersect_at_one_point_l639_639816


namespace remainder_when_divided_by_x_minus_2_l639_639788

-- Define the polynomial
def f (x : ℕ) : ℕ := x^3 - x^2 + 4 * x - 1

-- Statement of the problem: Prove f(2) = 11 using the Remainder Theorem
theorem remainder_when_divided_by_x_minus_2 : f 2 = 11 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l639_639788


namespace y_intercept_line_l639_639708

theorem y_intercept_line : 
  ∃ m b : ℝ, 
  (2 * m + b = -3) ∧ 
  (6 * m + b = 5) ∧ 
  b = -7 :=
by 
  sorry

end y_intercept_line_l639_639708


namespace bus_speed_l639_639117

noncomputable def speed_of_bus (radius_cm : ℝ) (rpm : ℝ) : ℝ :=
  let circumference_cm := 2 * Real.pi * radius_cm
  let distance_per_minute_cm := circumference_cm * rpm
  let distance_per_hour_cm := distance_per_minute_cm * 60
  distance_per_hour_cm / 100000

theorem bus_speed :
  speed_of_bus 140 125.11373976342128 ≈ 66.06 := sorry

end bus_speed_l639_639117


namespace categorize_correctly_l639_639166

def problem_numbers : List ℝ := 
  [-(1/10), real.cbrt 8, 0.3, -(real.pi / 3), -(real.sqrt 64), 0, real.sqrt 0.9, 22 / 7]

def is_positive_fraction (x : ℝ) : Prop := x > 0 ∧ ∃ p q : ℤ, q ≠ 0 ∧ x = p / q
def is_integer (x : ℝ) : Prop := ∃ z : ℤ, x = z
def is_irrational (x : ℝ) : Prop := ¬ ∃ p q : ℤ, q ≠ 0 ∧ x = p / q

def positive_fractions : List ℝ := [0.3, 22 / 7]
def integers : List ℝ := [real.cbrt 8, -(real.sqrt 64), 0]
def irrationals : List ℝ := [-(real.pi / 3), real.sqrt 0.9]

theorem categorize_correctly :
  (∀ x ∈ positive_fractions, is_positive_fraction x) ∧
  (∀ x ∈ integers, is_integer x) ∧
  (∀ x ∈ irrationals, is_irrational x) := by
  sorry

end categorize_correctly_l639_639166


namespace sphere_velocity_at_point_C_l639_639144

theorem sphere_velocity_at_point_C :
  ∀ (Q q : ℚ) (AB AC : ℚ) (m g k : ℚ) (v : ℚ),
  Q = -40 * 10^(-6) →
  q = 50 * 10^(-6) →
  AB = 4 →
  AC = 5 →
  m = 0.1 →
  g = 10 →
  k = 9 * 10^9 →
  v = real.sqrt 17920 →
  v ≈ 7.8 :=
by
  intros Q q AB AC m g k v hQ hq hAB hAC hm hg hk hv
  sorry

end sphere_velocity_at_point_C_l639_639144


namespace general_formula_sequence_a_positive_integer_k_minimum_value_of_lambda_l639_639813

-- Condition: S_n = 2a_n - 2
def S (n : ℕ) (a : ℕ → ℕ) : ℕ := sum (λ i => a i) (range (n + 1))
def a (n : ℕ) : ℕ := 2 ^ n

theorem general_formula_sequence_a :
  (∀ n : ℕ, S n a = 2 * a n - 2) → (∀ n : ℕ, a n = 2 ^ n) :=
sorry

-- Condition: b_1 = 8 and b_{n+1} = 2b_n - 2^{n+1}
def b : ℕ → ℕ 
| 0       => 0
| (n + 1) => if n = 0 then 8 else 2 * b n - 2^(n + 1)

def T (n : ℕ) : ℕ := sum (λ i => b i) (range (n + 1))

theorem positive_integer_k :
  (b 1 = 8 ∧ ∀ n : ℕ, b (n + 1) = 2 * b n - 2^(n + 1)) → ∃ k : ℕ, k = 4 ∧ ∀ n : ℕ, T k ≥ T n :=
sorry

-- Condition: Sequence {a_n} and c_n = a_{n+1} / ((1 + a_n)(1 + a_{n+1}))
def c (n : ℕ) : ℕ := a (n + 1) / ((1 + a n) * (1 + a (n + 1)))

def R (n : ℕ) : ℕ := sum (λ i => c i) (range (n + 1))

theorem minimum_value_of_lambda :
  (∀ n : ℕ, R n < λ) → ∃ min_lambda : ℚ, min_lambda = 2/3 :=
sorry

end general_formula_sequence_a_positive_integer_k_minimum_value_of_lambda_l639_639813


namespace same_set_condition_l639_639214

-- Definitions based on the conditions
def cond_1_M : Set Int := {3, -1}
def cond_1_P : Set (Int × Int) := {(3, -1)}

def cond_2_M : Set (Int × Int) := {(3, 1)}
def cond_2_P : Set (Int × Int) := {(1, 3)}

def cond_3_M : Set Real := {y | ∃ x : Real, y = x^2 - 1}
def cond_3_P : Set Real := {a | ∃ x : Real, a = x^2 - 1}

def cond_4_M : Set Real := {y | ∃ x : Real, y = x^2 - 1}
def cond_4_P : Set (Real × Real) := {(x, y) | ∃ x : Real, y = x^2 - 1}

-- Lean statement to prove M and P represent the same set given condition ③
theorem same_set_condition : (cond_1_M = cond_1_P ∨ cond_2_M = cond_2_P ∨ cond_3_M = cond_3_P ∨ cond_4_M = cond_4_P) → (cond_3_M = cond_3_P) :=
by 
  intros h_same
  sorry

end same_set_condition_l639_639214


namespace count_integers_between_200_250_l639_639440

theorem count_integers_between_200_250 :
  {n : ℕ // 200 ≤ n ∧ n < 250 ∧ 
            (let d2 := (n / 10) % 10, d3 := n % 10 in
             (n / 100 = 2) ∧ (d2 ≠ d3) ∧ (2 < d2) ∧ (d2 < d3)
            )}.to_finset.card = 11 :=
by
  -- Start the proof process here
  sorry

end count_integers_between_200_250_l639_639440


namespace mike_seeds_initial_l639_639549

def number_of_seeds_initial (a b c d: ℕ) : ℕ := a + 2 * b + c + d

theorem mike_seeds_initial :
  number_of_seeds_initial 20 20 30 30 = 120 :=
by
  -- Conditions
  have h1 : 20 = 20 := rfl
  have h2 : 2 * 20 = 40 := rfl
  have h3 : 30 = 30 := rfl
  have h4 : 30 = 30 := rfl

  -- Calculation using conditions
  show number_of_seeds_initial 20 20 30 30 = 120, from calc
    20 + 2 * 20 + 30 + 30 = 20 + 40 + 30 + 30 : by rw h2
                   ... = 20 + 40 + 60 : by rw add_assoc 30 30
                   ... = 120 : by rfl

end mike_seeds_initial_l639_639549


namespace central_angle_of_sector_l639_639466

-- Define the given conditions
def radius : ℝ := 10
def area : ℝ := 100

-- The statement to be proved
theorem central_angle_of_sector (α : ℝ) (h : area = (1 / 2) * α * radius ^ 2) : α = 2 :=
by
  sorry

end central_angle_of_sector_l639_639466


namespace dihedral_angle_planes_ADB_ADC_l639_639743

-- Definitions and conditions
variables (A B C D : Type)
variables [plane ABC] [plane ADB] [plane ADC]
variables [point A] [point B] [point C] [point D]
variables (AB BC CD AC1 AD1 BD1 S_ABC S_ACD S_ADB : ℝ)

-- Given conditions and lengths
def AB := 1
def BC := 2
def CD := 3
def right_triangle_base := true -- Right-angled triangle at base ABC
def DC_height := true -- DC is the height of the pyramid

-- Proving the dihedral angle between planes ADB and ADC
theorem dihedral_angle_planes_ADB_ADC :
  let AC := real.sqrt (AB^2 + BC^2),
      AD := real.sqrt (AC^2 + CD^2),
      BD := real.sqrt (BC^2 + CD^2),
      S_ACD := 1/2 * AC * CD,
      S_ADB := 1/2 * AB * BD,
      S_ABC := 1/2 * AB * BC in
  real.arcsin (2 * S_ADB / real.sqrt (S_ACD^2 + S_ADB^2)) = 
  real.arcsin (2 * real.sqrt 14 / real.sqrt 65) :=
begin
  sorry
end

end dihedral_angle_planes_ADB_ADC_l639_639743


namespace homothety_solution_l639_639095

variables {S1 S2 : Type*} [metric_space S1] [metric_space S2]
variables {f : S2 → S1} {r : ℝ}

-- Assuming S1 and S2 are circles in metric spaces with r as the radius ratio (1/3)

def homothety (X : S2) (f : S2 → S1) (r : ℝ) : S2 → S2 :=
  λ x, sorry -- Define the homothety function

noncomputable def find_Y (S1 S2 : Type*) [metric_space S1] [metric_space S2]
  (X : S2) (f : S2 → S1) (r : ℝ) : S1 :=
  sorry -- Y is obtained from intersection

def segment_XY (X : S2) (Y : S1) : Type* :=
  sorry -- A line segment between X and Y

theorem homothety_solution (S1 S2 : Type*) [metric_space S1] [metric_space S2]
  (X : S2) (f : S2 → S1) (r : ℝ) (h : 0 < r) (hX : X ∈ S2)
  (Y : S1) :
  Y = find_Y S1 S2 X f r →
  ∃ XY : Type*, segment_XY X Y :=
sorry

end homothety_solution_l639_639095


namespace paint_price_and_max_boxes_l639_639574

theorem paint_price_and_max_boxes (x y m : ℕ) 
  (h1 : x + 2 * y = 56) 
  (h2 : 2 * x + y = 64) 
  (h3 : 24 * m + 16 * (200 - m) ≤ 3920) : 
  x = 24 ∧ y = 16 ∧ m ≤ 90 := 
  by 
    sorry

end paint_price_and_max_boxes_l639_639574


namespace subtraction_equals_eleven_l639_639484

theorem subtraction_equals_eleven (K A N G R O : ℕ) (h1: K ≠ A) (h2: K ≠ N) (h3: K ≠ G) (h4: K ≠ R) (h5: K ≠ O) (h6: A ≠ N) (h7: A ≠ G) (h8: A ≠ R) (h9: A ≠ O) (h10: N ≠ G) (h11: N ≠ R) (h12: N ≠ O) (h13: G ≠ R) (h14: G ≠ O) (h15: R ≠ O) (sum_eq : 100 * K + 10 * A + N + 10 * G + A = 100 * R + 10 * O + O) : 
  (10 * R + N) - (10 * K + G) = 11 := 
by 
  sorry

end subtraction_equals_eleven_l639_639484


namespace wire_length_l639_639104

theorem wire_length (d h1 h2 : ℝ) (h_dist : d = 18) (h_height1 : h1 = 9) (h_height2 : h2 = 24) :
  (real.sqrt (d^2 + (h2 - h1)^2)) = real.sqrt 549 := by 
  sorry

end wire_length_l639_639104


namespace length_XY_l639_639044

-- Definitions and conditions from the problem
variables (A B C D P Q X Y : Type) 
variables [point : circle A B C D]

-- Given conditions
def AB := 15
def CD := 25
def AP := 8
def CQ := 14
def PQ := 35

-- The theorem to prove
theorem length_XY : XY = 41.12 :=
sorry

end length_XY_l639_639044


namespace infinitely_many_pairs_dividing_sq_plus_one_l639_639088

theorem infinitely_many_pairs_dividing_sq_plus_one :
  ∃ᶠ (a b : ℤ), a * (a + 1) ∣ b^2 + 1 :=
sorry

end infinitely_many_pairs_dividing_sq_plus_one_l639_639088


namespace correct_probability_l639_639946

noncomputable def T : ℕ := 44
noncomputable def num_books : ℕ := T - 35
noncomputable def n : ℕ := 9
noncomputable def favorable_outcomes : ℕ := (Nat.choose n 6) * 2
noncomputable def total_arrangements : ℕ := (Nat.factorial n)
noncomputable def probability : Rat := (favorable_outcomes : ℚ) / (total_arrangements : ℚ)
noncomputable def m : ℕ := 1
noncomputable def p : Nat := Nat.gcd 168 362880
noncomputable def final_prob_form : Rat := 1 / 2160
noncomputable def answer : ℕ := m + 2160

theorem correct_probability : 
  probability = final_prob_form ∧ answer = 2161 := 
by
  sorry

end correct_probability_l639_639946


namespace acute_isosceles_triangle_k_l639_639251

theorem acute_isosceles_triangle_k (ABC : Triangle) (circ : Circle)
  (D : Point)
  (h1 : ABC.angles.B = ABC.angles.C) -- Isosceles property
  (h2 : ∀ P ∈ circ, is_tangent B P circ) -- Tangent property through B
  (h3 : ∀ Q ∈ circ, is_tangent C Q circ) -- Tangent property through C
  (h4 : angle ABC.angles.B = 3 * angle D )
  (h5 : ∃ k, angle ABC.angles.A = k * π ) :
  ∃ k, k = 5 / 11 :=
by
  sorry

end acute_isosceles_triangle_k_l639_639251


namespace cristina_catches_nicky_l639_639172

def cristina_pace := 4 -- meters per second
def nicky_pace := 3 -- meters per second
def head_start := 36 -- meters

theorem cristina_catches_nicky :
  ∃ t : ℕ, (cristina_pace * t = head_start + nicky_pace * t) ∧ t = 36 :=
by
  use 36
  simp [cristina_pace, nicky_pace, head_start]
  sorry

end cristina_catches_nicky_l639_639172


namespace find_k_l639_639867

theorem find_k (k : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-1, k)
  (a.1 * (2 * a.1 - b.1) + a.2 * (2 * a.2 - b.2)) = 0 → k = 12 := sorry

end find_k_l639_639867


namespace range_of_a_l639_639584

noncomputable def p (x a : ℝ) := x^2 - 4 * a * x + 3 * a^2 < 0
noncomputable def q (x : ℝ) := (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2 * x - 8 > 0)

theorem range_of_a (a : ℝ) :
  a < 0 → 
  ((∀ x, p x a → q x) ∧ ¬ (∀ x, q x → p x a)) ↔ a ∈ set.Icc (-2/3 : ℝ) 0 ∪ set.Iic (-4 : ℝ) := 
by { sorry }

end range_of_a_l639_639584


namespace area_of_quadrilateral_Q1Q2Q3Q4_l639_639950

/-- 
  Let P1 P2 P3 P4 P5 P6 be a regular hexagon with edge-centered distance 
  (distance from the center to the midpoint of a side) of 2. If Qi is the midpoint 
  of the side Pi Pi+1 for i=1,2,3,4, then the area of quadrilateral Q1 Q2 Q3 Q4 is 3√3.
-/
theorem area_of_quadrilateral_Q1Q2Q3Q4 :
  let hexagon : Type := regular_hexagon in
  let edge_center_distance := 2 in
  let O : hexagon.center in
  let Q1 Q2 Q3 Q4 : hexagon.midpoints in
  -- Define the area of the quadrilateral Q1Q2Q3Q4
  area Q1 Q2 Q3 Q4 = 3 * Real.sqrt 3 :=
sorry

end area_of_quadrilateral_Q1Q2Q3Q4_l639_639950


namespace modulus_of_conjugate_complex_l639_639882

noncomputable def Z : ℂ := (1 - 2 * complex.i) / (1 + complex.i)
noncomputable def z_bar : ℂ := complex.conj Z

theorem modulus_of_conjugate_complex : |z_bar| = real.sqrt 10 / 2 :=
by
  sorry

end modulus_of_conjugate_complex_l639_639882


namespace cannot_be_a_lt_b_lt_c_l639_639888

theorem cannot_be_a_lt_b_lt_c (a b c : ℝ) (h : Real.log a 2 < Real.log b 2 ∧ Real.log b 2 < Real.log c 2) : ¬ (a < b ∧ b < c) :=
sorry

end cannot_be_a_lt_b_lt_c_l639_639888


namespace compute_sum_l639_639267

open BigOperators

theorem compute_sum : 
  (1 / 2 ^ 2010 : ℝ) * ∑ n in Finset.range 1006, (-3 : ℝ) ^ n * (Nat.choose 2010 (2 * n)) = -1 / 2 :=
by
  sorry

end compute_sum_l639_639267


namespace candy_count_l639_639757

theorem candy_count (S : ℕ) (H1 : 32 + S - 35 = 39) : S = 42 :=
by
  sorry

end candy_count_l639_639757


namespace find_k_l639_639219

-- Definitions based on conditions in step a)
def acute_isosceles_triangle_inscribed (A B C : Type) : Prop := sorry -- Formal definition of the triangle being acute isosceles and inscribed in a circle
def tangents_meeting_at_point (A B C D : Type) : Prop := sorry -- Formal definition of tangents through B and C meeting at D
def angle_relation (ABC D : Type) (theta : ℝ) : Prop := 3 * theta = sorry -- Formal definition of \(\angle ABC = \angle ACB = 3 \angle D\)
def angle_BAC (k : ℝ) (theta : ℝ) : Prop := theta = k * real.pi -- Formal definition of \(\angle BAC = k \pi\)

-- Theorem statement for our proof problem
theorem find_k
  (A B C D : Type)
  (h1 : acute_isosceles_triangle_inscribed A B C)
  (h2 : tangents_meeting_at_point A B C D)
  (theta : ℝ)
  (h3 : angle_relation ABC D theta)
  (k : ℝ)
  (h4 : angle_BAC k theta) :
  k = 1 / 13 := by
  sorry

end find_k_l639_639219


namespace sin_cos_sum_l639_639830

variable (θ : ℝ)

-- Conditions
def in_second_quadrant (θ : ℝ) : Prop := π / 2 < θ ∧ θ < π
def tan_sum_eq_half (θ : ℝ) : Prop := tan (θ + π / 4) = 1 / 2

-- Theorem stating the problem
theorem sin_cos_sum (h1 : in_second_quadrant θ) (h2 : tan_sum_eq_half θ) : 
  sin θ + cos θ = - sqrt 10 / 5 :=
sorry

end sin_cos_sum_l639_639830


namespace rectangle_area_l639_639619

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end rectangle_area_l639_639619


namespace count_integers_between_200_and_250_with_increasing_digits_l639_639432

theorem count_integers_between_200_and_250_with_increasing_digits :
  ∃ n, n = 11 ∧ ∀ x, 200 ≤ x ∧ x ≤ 250 ∧ 
  (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k) → 
  n = ∑ x in {x | 200 ≤ x ∧ x ≤ 250 ∧ (∀ i j k, x = 100 * i + 10 * j + k → i < j ∧ j < k ∧ i ≠ j ∧ j ≠ k ∧ i ≠ k)}, 1
:= 
sorry

end count_integers_between_200_and_250_with_increasing_digits_l639_639432


namespace matchstick_area_l639_639290

theorem matchstick_area (n : ℕ) (m : ℕ) (A1 : ℕ) (A2 : ℕ) :
  n = 7 ∧ m = 13 ∧ A1 = 5 ∧ A2 = 15 → A2 = 3 * A1 :=
by
  intros h
  cases h
  rw [h.left, h.right.left, h.right.right.left, h.right.right.right]
  sorry

end matchstick_area_l639_639290


namespace least_positive_integer_f_f_n_eq_n_l639_639971

-- Definitions
def largest_prime_factor (n : ℕ) : ℕ := 
  if n = 1 then 1 else sorry -- Implementation of largest prime factor of n^2 + 1

-- Conditions
def f (n : ℕ) : ℕ := largest_prime_factor (n^2 + 1)

-- Theorem statement
theorem least_positive_integer_f_f_n_eq_n : ∃ n : ℕ, 0 < n ∧ f(f(n)) = n ∧ n = 89 :=
begin
  sorry
end

end least_positive_integer_f_f_n_eq_n_l639_639971


namespace committee_formation_l639_639690

theorem committee_formation :
  let total_members := 12
  let specific_members := 2
  let remaining_members := total_members - specific_members
  let committee_size := 5
  let members_to_choose := committee_size - specific_members
  choose (remaining_members, members_to_choose) = 120 :=
by
  sorry

end committee_formation_l639_639690


namespace combination_sum_l639_639552

variable {C : ℕ → ℕ → ℕ}

theorem combination_sum (n : ℕ) : 
  (C (2 * n) n) = (Finset.sum (Finset.range (n + 1)) (λ k, (C n k) * (C n (n - k)))) := 
sorry

end combination_sum_l639_639552


namespace calculate_fg3_l639_639509

def f (x: ℝ) := 2 * x + 4
def g (x: ℝ) := x^2 - 8

theorem calculate_fg3 : f(g(3)) = 6 :=
by
  sorry

end calculate_fg3_l639_639509


namespace investment2_rate_l639_639197

-- Define the initial conditions
def total_investment : ℝ := 10000
def investment1 : ℝ := 4000
def rate1 : ℝ := 0.05
def investment2 : ℝ := 3500
def income1 : ℝ := investment1 * rate1
def yearly_income_goal : ℝ := 500
def remaining_investment : ℝ := total_investment - investment1 - investment2
def rate3 : ℝ := 0.064
def income3 : ℝ := remaining_investment * rate3

-- The main theorem
theorem investment2_rate (rate2 : ℝ) : 
  income1 + income3 + investment2 * (rate2 / 100) = yearly_income_goal → rate2 = 4 := 
by 
  sorry

end investment2_rate_l639_639197


namespace equivalent_expression_l639_639506

noncomputable def problem_statement (α β γ δ p q : ℝ) :=
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -2 * (p^2 - q^2) + 4

theorem equivalent_expression
  (α β γ δ p q : ℝ)
  (h1 : ∀ x, x^2 + p * x + 2 = 0 → (x = α ∨ x = β))
  (h2 : ∀ x, x^2 + q * x + 2 = 0 → (x = γ ∨ x = δ)) :
  problem_statement α β γ δ p q :=
by sorry

end equivalent_expression_l639_639506


namespace sqrt_domain_l639_639478

def inequality_holds (x : ℝ) : Prop := x + 5 ≥ 0

theorem sqrt_domain (x : ℝ) : inequality_holds x ↔ x ≥ -5 := by
  sorry

end sqrt_domain_l639_639478


namespace range_of_m_l639_639831

theorem range_of_m (k : ℝ) (m : ℝ) (y x : ℝ)
  (h1 : ∀ x, y = k * (x - 1) + m)
  (h2 : y = 3 ∧ x = -2)
  (h3 : (∃ x, x < 0 ∧ y > 0) ∧ (∃ x, x < 0 ∧ y < 0) ∧ (∃ x, x > 0 ∧ y < 0)) :
  m < - (3 / 2) :=
sorry

end range_of_m_l639_639831


namespace no_odd_prime_pn_plus_1_eq_2m_no_odd_prime_pn_minus_1_eq_2m_l639_639994

open Nat

theorem no_odd_prime_pn_plus_1_eq_2m (n p m : ℕ)
  (hn : n > 1) (hp : p.Prime) (hp_odd : Odd p) (hm : m > 0) :
  p^n + 1 ≠ 2^m := by
  sorry

theorem no_odd_prime_pn_minus_1_eq_2m (n p m : ℕ)
  (hn : n > 2) (hp : p.Prime) (hp_odd : Odd p) (hm : m > 0) :
  p^n - 1 ≠ 2^m := by
  sorry

end no_odd_prime_pn_plus_1_eq_2m_no_odd_prime_pn_minus_1_eq_2m_l639_639994


namespace x_sq_pos_necessary_but_not_sufficient_l639_639289

theorem x_sq_pos_necessary_but_not_sufficient (x : ℝ) : 
  (x > 0 → x^2 > 0) ∧ (¬ (x^2 > 0 → x > 0)) :=
by 
  split
  sorry
  sorry

end x_sq_pos_necessary_but_not_sufficient_l639_639289


namespace average_playtime_10_hours_per_person_l639_639678

theorem average_playtime_10_hours_per_person :
  ∀ (persons : ℕ) (start_time end_time hours_per_day : ℕ), 
  persons = 8 →
  start_time = 8 →
  end_time = 18 →
  hours_per_day = (end_time - start_time) →
  average_hours_per_person (persons * hours_per_day) (persons) = 10 :=
by
  intros persons start_time end_time hours_per_day h_persons h_start h_end h_hours_per_day
  sorry

noncomputable def average_hours_per_person (total_hours persons : ℕ) : ℕ :=
  total_hours / persons

end average_playtime_10_hours_per_person_l639_639678


namespace no_equal_differences_between_products_l639_639072

theorem no_equal_differences_between_products (a b c d : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h_order : a < b ∧ b < c ∧ c < d) :
    ¬ (∃ k : ℕ, ac - ab = k ∧ ad - ac = k ∧ bc - ad = k ∧ bd - bc = k ∧ cd - bd = k) :=
by
  sorry

end no_equal_differences_between_products_l639_639072


namespace ellipse_equation_fixed_points_l639_639353

namespace ProofProblem

-- Part I: Prove the equation of the ellipse
theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 2 * a = 4) (h4 : c = a / 2) (h5 : a^2 = b^2 + c^2) :
  (a = 2) ∧ (b = Real.sqrt 3) ∧ (c = 1) ∧ ∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) := 
by
  sorry

-- Part II: Prove the circle with diameter MN passes through fixed points
theorem fixed_points (a b : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 3) :
  ∀ P: ℝ × ℝ, P ∈ set_of (λ p : ℝ × ℝ, p.fst^2 / 4 + p.snd^2 / 3 = 1) → 
    ∃ Q: ℝ × ℝ, (Q = (1, 0)) ∨ (Q = (7, 0)) :=
by
  sorry
end ProofProblem

end ellipse_equation_fixed_points_l639_639353


namespace paint_price_max_boxes_paint_A_l639_639570

theorem paint_price :
  ∃ x y : ℕ, (x + 2 * y = 56 ∧ 2 * x + y = 64) ∧ x = 24 ∧ y = 16 := by
  sorry

theorem max_boxes_paint_A (m : ℕ) :
  24 * m + 16 * (200 - m) ≤ 3920 → m ≤ 90 := by
  sorry

end paint_price_max_boxes_paint_A_l639_639570


namespace identify_odd_function_l639_639388

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem identify_odd_function :
  ¬ is_odd_function (λ x, x^2) ∧
  ¬ is_odd_function (λ x, 2^x) ∧
  ¬ is_odd_function (λ x, Math.cos x) ∧
  is_odd_function (λ x, -x^3) := by
  sorry

end identify_odd_function_l639_639388


namespace frank_money_l639_639335

theorem frank_money (X : ℝ) (h1 : (3/4) * (4/5) * X = 360) : X = 600 :=
sorry

end frank_money_l639_639335


namespace num_second_visits_correct_l639_639257

-- Problem definition
variable (total_customers first_visit_cost second_visit_cost third_visit_count total_revenue : ℕ)

-- Given conditions
def conditions : Prop :=
  total_customers = 100 ∧
  first_visit_cost = 10 ∧
  second_visit_cost = 8 ∧
  third_visit_count = 10 ∧
  total_revenue = 1240

-- Question to be proved
def num_second_visits (x : ℕ) : Prop :=
  x = 20

-- Statement to prove
theorem num_second_visits_correct (x : ℕ) (h1 : conditions total_customers first_visit_cost second_visit_cost third_visit_count total_revenue) :
  num_second_visits total_customers first_visit_cost second_visit_cost third_visit_count total_revenue x :=
begin
  sorry
end

end num_second_visits_correct_l639_639257


namespace maximum_daily_sales_revenue_l639_639190

noncomputable def P (t : ℕ) : ℤ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

noncomputable def Q (t : ℕ) : ℤ :=
  if 0 < t ∧ t ≤ 30 then -t + 40 else 0

noncomputable def y (t : ℕ) : ℤ := P t * Q t

theorem maximum_daily_sales_revenue : 
  ∃ (t : ℕ), 0 < t ∧ t ≤ 30 ∧ y t = 1125 :=
by
  sorry

end maximum_daily_sales_revenue_l639_639190


namespace minimum_value_98_l639_639962

noncomputable theory
open_locale big_operators

-- Definition of the variables and conditions
variables (a b c d e f g : ℝ)
variables (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) (hg : g > 0)
variable (h_sum : a + b + c + d + e + f + g = 8)

-- The statement of the problem
theorem minimum_value_98 : 
  (1/a + 4/b + 9/c + 16/d + 25/e + 36/f + 49/g) = 98 :=
sorry

end minimum_value_98_l639_639962


namespace determine_radii_l639_639068

theorem determine_radii (a c : ℝ) (h : c ≠ 0) :
  ∃ (r R : ℝ), r = (a / 2) * (4 / c - 1) ∧ R = (a / 2) * (4 / c + 1) :=
by
  have hc : c ≠ 0 := h
  let r := (a / 2) * (4 / c - 1)
  let R := (a / 2) * (4 / c + 1)
  use [r, R]
  split
  . exact rfl
  . exact rfl

end determine_radii_l639_639068


namespace solve_for_x_l639_639449

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end solve_for_x_l639_639449


namespace parabola_vertex_l639_639105

-- Define the parabola equation as a function
def parabola (x : ℝ) : ℝ := x^2 - 2

-- Prove that the vertex of the parabola y = x^2 - 2 is (0, -2)
theorem parabola_vertex : ∃ (h k : ℝ), (∀ x : ℝ, parabola x = (x - h)^2 + k) ∧ h = 0 ∧ k = -2 :=
by
  use 0, -2
  split
  . intros x
    dsimp [parabola]
    ring
  . split
    . rfl
    . rfl

end parabola_vertex_l639_639105


namespace nikka_stamp_collection_l639_639986

def total_stamps : ℕ := 500
def chinese_percentage : ℝ := 0.40
def us_percentage : ℝ := 0.25
def japanese_percentage : ℝ := 0.15
def british_percentage : ℝ := 0.10

def chinese_stamps : ℕ := chinese_percentage * total_stamps
def us_stamps : ℕ := us_percentage * total_stamps
def japanese_stamps : ℕ := japanese_percentage * total_stamps
def british_stamps : ℕ := british_percentage * total_stamps
def other_percentage : ℝ := 1.0 - (chinese_percentage + us_percentage + japanese_percentage + british_percentage)
def other_stamps : ℕ := other_percentage * total_stamps

theorem nikka_stamp_collection :
  chinese_stamps = 200 ∧
  us_stamps = 125 ∧
  japanese_stamps = 75 ∧
  british_stamps = 50 ∧
  other_stamps = 50 := by
  sorry

end nikka_stamp_collection_l639_639986


namespace value_of_z_l639_639111

theorem value_of_z : 
  let m1 := (6 + 10 + 22) / 3 in 
  let z := (31 / 3) in 
  (m1 = (15 + z) / 2) →
  z = 31 / 3 :=
by 
  let m1 := (6 + 10 + 22) / 3 
  let z := (31 / 3)
  sorry

end value_of_z_l639_639111


namespace min_value_quadratic_expr_l639_639783

theorem min_value_quadratic_expr (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : a > 0) 
  (h2 : x₁ ≠ x₂) 
  (h3 : x₁^2 - 4*a*x₁ + 3*a^2 < 0) 
  (h4 : x₂^2 - 4*a*x₂ + 3*a^2 < 0)
  (h5 : x₁ + x₂ = 4*a)
  (h6 : x₁ * x₂ = 3*a^2) : 
  x₁ + x₂ + a / (x₁ * x₂) = 4 * a + 1 / (3 * a) := 
sorry

end min_value_quadratic_expr_l639_639783


namespace snowmobile_overtakes_atv_on_11th_lap_l639_639205

noncomputable def snowmobile_speed_loose : ℝ := 32
noncomputable def snowmobile_speed_dense : ℝ := 36
noncomputable def atv_speed_loose : ℝ := 16
noncomputable def atv_speed_dense : ℝ := 48
def track_length : ℝ := 1 -- assume length of track as 1 unit

def time_snowmobile : ℝ := (track_length / 4) / snowmobile_speed_loose + (3 * track_length / 4) / snowmobile_speed_dense
def time_atv : ℝ := (track_length / 4) / atv_speed_loose + (3 * track_length / 4) / atv_speed_dense

def lap_snowmobile (n : ℕ) : ℝ := time_snowmobile * n
def lap_atv (n : ℕ) : ℝ := time_atv * n

theorem snowmobile_overtakes_atv_on_11th_lap : ∃ n m : ℕ, n = 11 ∧ m = 10 ∧ lap_snowmobile n = lap_atv m + track_length := 
by
  sorry

end snowmobile_overtakes_atv_on_11th_lap_l639_639205


namespace range_of_x_l639_639811

theorem range_of_x (x t: ℝ) (S_n: ℕ → ℝ) (a: ℕ → ℝ) (h1: a 1 = 1)
  (h2: ∀ n, a (n + 1) = a n / 2)
  (h3: ∀ n, S_n n = (2 - (1/2)^(n - 1)))
  (h4: ∀ t ∈ Icc (-1 : ℝ) 1, ∀ n ∈ Ioi 0, x^2 + t * x + 1 > S_n n) :
  x ∈ Icc (-(1 + Real.sqrt 5) / 2) ((1 + Real.sqrt 5) / 2) :=
sorry

end range_of_x_l639_639811


namespace multiplication_problem_l639_639920

def unique_char_digit (char : Char) (digit : ℕ) : Prop :=
  digit >= 0 ∧ digit < 10 ∧ ∀ char', char ≠ char' → digit ≠ unique_char_digit char' digit

def multiplication_product : list Char → ℕ → Prop
| [char1, char2, char3, char4, char5] 39672 :=
    unique_char_digit char1 3 ∧ 
    unique_char_digit char2 9 ∧ 
    unique_char_digit char3 6 ∧ 
    unique_char_digit char4 7 ∧ 
    unique_char_digit char5 2
| _ _ := false

theorem multiplication_problem : ∃ chars, multiplication_product chars 39672 :=
by sorry

end multiplication_problem_l639_639920


namespace polar_equations_and_segment_length_l639_639475

noncomputable theory

def circle_param_eq (θ : ℝ) : ℝ × ℝ :=
  (2 + 2 * real.cos θ, 2 * real.sin θ)

def line_l1_eq (x : ℝ) : Prop :=
  x + 1 = 0

def line_l2_polar_eq (θ : ℝ) : Prop :=
  θ = real.pi / 3

theorem polar_equations_and_segment_length (O P Q : ℝ × ℝ)
  (hO : O = (0, 0))
  (hP : P = (1, real.sqrt 3))
  (hQ : Q = (-1, -real.sqrt 3)) :
  (∀ θ, (fst (circle_param_eq θ))^2 + (snd (circle_param_eq θ))^2 - 4 * (fst (circle_param_eq θ)) = 0) ∧
  (∀ x, line_l1_eq x → (real.sqrt ((fst P - fst Q)^2 + (snd P - snd Q)^2) = 4)) := 
by
  sorry

end polar_equations_and_segment_length_l639_639475


namespace acute_isosceles_triangle_inscribed_circle_l639_639237

variables {A B C D : Type} [EuclideanGeometry A B C D]

theorem acute_isosceles_triangle_inscribed_circle (h1 : is_acute A B C)
    (h2 : is_isosceles A B C) (h3 : inscribed_in_circle A B C)
    (h4 : are_tangents B C D) (h5 : ∠ ABC = ∠ ACB = 3 * ∠ D)
    (h6 : tan_from_circle_point B C D) :
    (∠ BAC = 5 * π / 11) :=
sorry

end acute_isosceles_triangle_inscribed_circle_l639_639237


namespace complex_inequality_l639_639944

open Complex

noncomputable def condition (a b c : ℂ) := a * Complex.abs (b * c) + b * Complex.abs (c * a) + c * Complex.abs (a * b) = 0

theorem complex_inequality (a b c : ℂ) (h : condition a b c) :
  Complex.abs ((a - b) * (b - c) * (c - a)) ≥ 3 * Real.sqrt 3 * Complex.abs (a * b * c) := 
sorry

end complex_inequality_l639_639944


namespace perpendicular_HH_l639_639742

-- Definitions and conditions directly from the problem
variable (A B C D E F G L M N U V H H' : Type)
variable [is_complete_quadrilateral A B E C F D]
variable (AC_inter_BD : A ∩C = BD)
variable (Miquel_points_1 : is_miquel_point ABECFD L)
variable (Miquel_points_2 : is_miquel_point ADFCBG M)
variable (Miquel_points_3 : is_miquel_point ABECDG N)
variable (L_inter_EG : L ∩ EG = U)
variable (LN_inter_FG : LN ∩ FG = V)
variable (H'_perp_FG : H' ∩ M = FG)
variable (H'_perp_EG : H' ∩ N = EG)
variable (perpendiculars_concurrent : are_concurrent [from E to LN, from F to LM, from G to MN])

-- Proof problem statement
theorem perpendicular_HH'_UV :
  perpendicular H H' U V := sorry

end perpendicular_HH_l639_639742


namespace always_composite_l639_639277

theorem always_composite (p : ℕ) (hp : Nat.Prime p) : ¬Nat.Prime (p^2 + 35) ∧ ¬Nat.Prime (p^2 + 55) :=
by
  sorry

end always_composite_l639_639277


namespace find_k_l639_639230

noncomputable theory
open_locale classical

variables (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)

-- Condition 1: ABC is an acute isosceles triangle inscribed in a circle
-- Condition 2: Tangents from B and C meet at D
-- Condition 3: angle ABC = angle ACB = 3 * angle D
    
def tangents_intersect_at_D (ABC : triangle) (B C D : point ABC) : Prop :=
  tangent_to_circle_at B D ∧ tangent_to_circle_at C D ∧ B ≠ C

def isosceles_triangle_condition (angle : point ABC → point ABC → point ABC → real) (B C : point ABC) : Prop :=
  angle ABC B C = angle ABC C B

def angle_triple_D (angle : point ABC → point ABC → point ABC → real) (B C D : point ABC) : Prop :=
  ∃ k, angle ABC B C = 3 * angle ABC D C ∧ angle ABC A B = k * π

theorem find_k (ABC : Type*) [triangle ABC] [inscribed_in_circle ABC] [isosceles_triangle ABC]
  (B C D : point ABC) (angle : point ABC → point ABC → point ABC → real)
  (h1 : tangents_intersect_at_D ABC B C D)
  (h2 : isosceles_triangle_condition angle B C)
  (h3 : angle_triple_D angle B C D) :
  ∃ k : real, angle ABC A B = (5 / 11) * π ∧ k = 5 / 11 :=
by sorry

end find_k_l639_639230


namespace smallest_n_geometric_series_ineq_l639_639001

theorem smallest_n_geometric_series_ineq (n : ℕ) (h_pos : 0 < n) : 
  (∑ i in finset.range n, (1 / 2) ^ (i + 1)) > (315 / 412) ↔ n = 3 := by
  sorry

end smallest_n_geometric_series_ineq_l639_639001


namespace area_of_square_on_semicircle_l639_639707

theorem area_of_square_on_semicircle (r : ℝ) (A B C D : RealPoint) (h_square : is_square A B C D) 
(h1 : B ∈ Circle (0, 0) r) (h2 : C ∈ Circle (0, 0) r)
(h3 : A.1 = -r) (h4 : C.1 = r) : 
  square_area h_square = (4 / 5) := 
sorry

end area_of_square_on_semicircle_l639_639707


namespace g_is_zero_for_all_x_l639_639511

def g (x : ℝ) : ℝ := 
  Real.sqrt ((Real.sin x)^6 + 9 * (Real.cos x)^2) - Real.sqrt ((Real.cos x)^6 + 9 * (Real.sin x)^2)

theorem g_is_zero_for_all_x (x : ℝ) : 
  g(x) = 0 := 
sorry

end g_is_zero_for_all_x_l639_639511


namespace frisbee_cost_l639_639769

theorem frisbee_cost 
  {initial_amount kite_cost final_amount : ℝ}
  (initial_condition : initial_amount = 78)
  (kite_condition : kite_cost = 8)
  (final_condition : final_amount = 61) : 
  ∃ frisbee_cost : ℝ, initial_amount - kite_cost - frisbee_cost = final_amount ∧ frisbee_cost = 9 :=
by 
  have h1 : initial_amount - kite_cost = 70 := by 
    rw [initial_condition, kite_condition]
    norm_num
  use (70 - final_amount)
  split
  { 
    rw [h1, final_condition]
    norm_num
  }
  {
    rw final_condition
    norm_num
  }

end frisbee_cost_l639_639769


namespace curve_equation_l639_639778

noncomputable def satisfies_conditions (f : ℝ → ℝ) (M₀ : ℝ × ℝ) : Prop :=
  (f M₀.1 = M₀.2) ∧ 
  (∀ (x y : ℝ) (h_tangent : ∀ x y, y = (f x) → x * y - 2 * (f x) * x = 0),
    y = f x → x * y / (y / x) = 2 * x)

theorem curve_equation (f : ℝ → ℝ) :
  satisfies_conditions f (1, 4) →
  (∀ x : ℝ, f x * x = 4) :=
by
  intro h
  sorry

end curve_equation_l639_639778


namespace part_one_part_two_l639_639342

def p (m : ℝ) := ∀ x ∈ set.Icc (0 : ℝ) 1, 2 * x - 2 ≥ m^2 - 3 * m
def q (m a : ℝ) := ∃ x ∈ set.Icc (-1 : ℝ) 1, m ≤ a * x

theorem part_one (m : ℝ) : p m → 1 ≤ m ∧ m ≤ 2 :=
by sorry

theorem part_two (m a : ℝ) (ha : a = 1) : ¬ (p m ∧ q m a) ∧ (p m ∨ q m a) → m < 1 ∨ (1 < m ∧ m ≤ 2) :=
by sorry

end part_one_part_two_l639_639342


namespace binomial_coefficient_multiple_of_4_l639_639328

theorem binomial_coefficient_multiple_of_4 :
  ∃ (S : Finset ℕ), (∀ k ∈ S, 0 ≤ k ∧ k ≤ 2014 ∧ (Nat.choose 2014 k) % 4 = 0) ∧ S.card = 991 :=
sorry

end binomial_coefficient_multiple_of_4_l639_639328


namespace find_a6_l639_639351

-- Definition of arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Given conditions
variables (a1 a3 a4 : ℤ)
hypothesis (diff2 : ∀ n, arithmetic_sequence a1 2 2 = a3
                        ∧ arithmetic_sequence a1 2 3 = a4 
                        ∧ a3^2 = a1 * a4)

-- Statement to prove
theorem find_a6 : a1 = -8 → ∀ d = 2, arithmetic_sequence a1 d 5 = 2 := 
by {
    sorry,
}

end find_a6_l639_639351


namespace triangle_area_l639_639714

theorem triangle_area (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 15) (h₃ : c = 17) (h₄ : a^2 + b^2 = c^2) : 
  1 / 2 * a * b = 60 :=
by 
  -- Assign the values to constants
  let a := 8
  let b := 15
  let c := 17
  let hypotenuse_check := 8 ^ 2 + 15 ^ 2 = 17 ^ 2

  -- Check the hypotenuse
  have h5 : 8 ^ 2 + 15 ^ 2 = 17 ^ 2 := by simp [show 8 ^ 2 + 15 ^ 2 = 17 ^ 2, from hypotenuse_check]

  -- Prove the area as 60
  show 1 / 2 * (8 * 15) = 60, by linarith

end triangle_area_l639_639714


namespace josie_money_left_over_l639_639938

def money_left_over (money_given milk_price bread_price detergent_price detergent_coupon banana_price_per_pound banana_pounds milk_discount) :=
  let total_cost := (milk_price * milk_discount) + bread_price + (detergent_price - detergent_coupon) + (banana_price_per_pound * banana_pounds)
  money_given - total_cost

theorem josie_money_left_over :
  money_left_over 20.00 4.00 3.50 10.25 1.25 0.75 2 0.5 = 4.00 :=
by
  -- This proof is a placeholder and should be filled in.
  sorry

end josie_money_left_over_l639_639938


namespace neg_exists_n_sq_gt_two_pow_n_l639_639541

open Classical

theorem neg_exists_n_sq_gt_two_pow_n :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by
  sorry

end neg_exists_n_sq_gt_two_pow_n_l639_639541


namespace apples_to_cucumbers_l639_639893

-- Definitions based on conditions
def apples := ℕ
def bananas := ℕ
def cucumbers := ℕ

axiom apples_cost_bananas (a b : apples) (c d : bananas) : 12 * a = 6 * b
axiom bananas_cost_cucumbers (e f : bananas) (g h : cucumbers) : 3 * e = 4 * g

-- Theorem statement
theorem apples_to_cucumbers {a b : apples} {e f : bananas} {g h : cucumbers} 
  (hb: apples_cost_bananas a b) 
  (hc: bananas_cost_cucumbers e g) 
  : 24 * a = 16 * g :=
sorry

end apples_to_cucumbers_l639_639893


namespace simplify_and_evaluate_expression_l639_639092

theorem simplify_and_evaluate_expression
    (a b : ℤ)
    (h1 : a = -1/3)
    (h2 : b = -2) :
  ((3 * a + b)^2 - (3 * a + b) * (3 * a - b)) / (2 * b) = -3 :=
by
  sorry

end simplify_and_evaluate_expression_l639_639092


namespace dog_older_than_max_by_18_l639_639327

-- Definition of the conditions
def human_to_dog_years_ratio : ℕ := 7
def max_age : ℕ := 3
def dog_age_in_human_years : ℕ := 3

-- Translate the question: How much older, in dog years, will Max's dog be?
def age_difference_in_dog_years : ℕ :=
  dog_age_in_human_years * human_to_dog_years_ratio - max_age

-- The proof statement
theorem dog_older_than_max_by_18 : age_difference_in_dog_years = 18 := by
  sorry

end dog_older_than_max_by_18_l639_639327


namespace solve_for_s_l639_639446

theorem solve_for_s (s t : ℚ) (h1 : 7 * s + 8 * t = 150) (h2 : s = 2 * t + 3) : s = 162 / 11 := 
by
  sorry

end solve_for_s_l639_639446


namespace local_minimum_at_1_l639_639542

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1)^2 + 1

theorem local_minimum_at_1 (x : ℝ) :
  is_local_min f 1 ∧ ¬(∃ x, is_local_max f x) :=
by
  sorry

end local_minimum_at_1_l639_639542


namespace particle_position_after_2023_minutes_l639_639199

def particle_position : ℕ × ℕ
 | 0 := (0, 0)
 | 1 := (1, 0)
 | (n + 1) :=
   let (x, y) := particle_position n in
   if (n + 1) % 4 == 1 then (x - n // 2 - 1, y - n // 2 - 1)
   else if (n + 1) % 4 == 3 then (x + n // 2 + 1, y - n // 2 - 1)
   else if (n + 1) % 4 == 0 then (x + n // 2 + 1, y + n // 2 + 1)
   else (x - n // 2 - 1, y + n // 2 + 1)

theorem particle_position_after_2023_minutes :
  particle_position 2023 = (-43, -43) :=
sorry

end particle_position_after_2023_minutes_l639_639199


namespace charley_fraction_pulled_black_l639_639264

variable (white_beads total_beads pulled_white pulled_total pulled_black : ℕ)
variable (x : ℚ)

-- Initial conditions
def charley_initial_condition (white_beads black_beads : ℕ) : Prop := 
    white_beads = 51 ∧ black_beads = 90

-- Conditions of beads pulled out
def charley_pulled_conditions (pulled_white pulled_total : ℕ) : Prop := 
    pulled_white = 51 / 3 ∧ pulled_total = 32

-- Definition of x
def fraction_pulled_black_beads (x : ℚ) (total_black : ℕ) : ℚ :=
    x * total_black

-- The theorem to be proved
theorem charley_fraction_pulled_black 
    (initial_cond : charley_initial_condition white_beads total_beads)
    (pulled_cond : charley_pulled_conditions pulled_white pulled_total)
    (x_eq : fraction_pulled_black_beads x total_beads) :
    x = 1 / 6 :=
begin
    sorry
end

end charley_fraction_pulled_black_l639_639264


namespace ab_bc_cd_da_le_four_l639_639530

theorem ab_bc_cd_da_le_four (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 :=
by
  sorry

end ab_bc_cd_da_le_four_l639_639530


namespace complement_A_union_B_in_U_l639_639397

-- Define the universe set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | (x - 2) * (x + 1) ≤ 0}

-- Define set B
def B : Set ℝ := {x | 0 ≤ x ∧ x < 3}

-- Define the union of A and B
def A_union_B : Set ℝ := {x | (-1 ≤ x ∧ x < 3)}

-- Define the complement of A ∪ B in U
def C_U_A_union_B : Set ℝ := {x | x < -1 ∨ x ≥ 3}

-- Proof Statement
theorem complement_A_union_B_in_U :
  {x | x < -1 ∨ x ≥ 3} = {x | x ∈ U ∧ (x ∉ A_union_B)} :=
sorry

end complement_A_union_B_in_U_l639_639397


namespace ray_reflection_and_distance_l639_639697

-- Define the initial conditions
def pointA : ℝ × ℝ := (-3, 3)
def circleC_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Definitions of the lines for incident and reflected rays
def incident_ray_line (x y : ℝ) : Prop := 4*x + 3*y + 3 = 0
def reflected_ray_line (x y : ℝ) : Prop := 3*x + 4*y - 3 = 0

-- Distance traveled by the ray
def distance_traveled (A T : ℝ × ℝ) := 7

theorem ray_reflection_and_distance :
  ∃ (x₁ y₁ : ℝ), incident_ray_line x₁ y₁ ∧ reflected_ray_line x₁ y₁ ∧ circleC_eq x₁ y₁ ∧ 
  (∀ (P : ℝ × ℝ), P = pointA → distance_traveled P (x₁, y₁) = 7) :=
sorry

end ray_reflection_and_distance_l639_639697


namespace rhombus_area_l639_639656

-- Define the rhombus with given conditions
def rhombus (a d1 d2 : ℝ) : Prop :=
  a = 9 ∧ abs (d1 - d2) = 10 

-- The theorem stating the area of the rhombus
theorem rhombus_area (a d1 d2 : ℝ) (h : rhombus a d1 d2) : 
  (d1 * d2) / 2 = 72 :=
by
  sorry

#check rhombus_area

end rhombus_area_l639_639656
