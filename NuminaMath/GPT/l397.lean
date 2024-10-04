import Mathlib

namespace least_number_of_digits_in_repeating_block_of_7_over_13_l397_397962

-- Statement Definition
theorem least_number_of_digits_in_repeating_block_of_7_over_13 : 
  ∃ n : ℕ, n = 6 ∧ repeating_block_length (7 / 13) = n :=
begin
  sorry
end

end least_number_of_digits_in_repeating_block_of_7_over_13_l397_397962


namespace num_distinguishable_octahedrons_l397_397215

-- Define the given conditions
def num_faces : ℕ := 8
def num_colors : ℕ := 8
def total_permutations : ℕ := Nat.factorial num_colors
def distinct_orientations : ℕ := 24

-- Prove the main statement
theorem num_distinguishable_octahedrons : total_permutations / distinct_orientations = 1680 :=
by
  sorry

end num_distinguishable_octahedrons_l397_397215


namespace avg_weight_B_correct_l397_397082

-- Definitions of the conditions
def students_A : ℕ := 24
def students_B : ℕ := 16
def avg_weight_A : ℝ := 40
def avg_weight_class : ℝ := 38

-- Definition of the total weight calculation for sections A and B
def total_weight_A : ℝ := students_A * avg_weight_A
def total_weight_class : ℝ := (students_A + students_B) * avg_weight_class

-- Defining the average weight of section B as the unknown to be proven
noncomputable def avg_weight_B : ℝ := 35

-- The theorem to prove that the average weight of section B is 35 kg
theorem avg_weight_B_correct : 
  total_weight_A + students_B * avg_weight_B = total_weight_class :=
by
  sorry

end avg_weight_B_correct_l397_397082


namespace determine_V_3030_l397_397234

def arithmetic_sequence (b e : ℝ) (n : ℕ) : ℝ :=
  b + (n - 1) * e

def U_n (b e : ℝ) (n : ℕ) : ℝ :=
  (2 * b + (n - 1) * e) / 2 * n

def V_n (b e : ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range n, U_n b e (k + 1)

theorem determine_V_3030 (b e : ℝ) :
  (λ n, U_n b e n) 2020 = U_n b e 2020 → 
  ∃ (U: ℝ), U = (U_n b e 2020) → 
  ∃ (V: ℝ), V = (V_n b e 3030) :=
by
  sorry

end determine_V_3030_l397_397234


namespace equation_has_8_roots_l397_397717

-- Defining the quadratic polynomials
def p1 (x : ℝ) (p1 q1 : ℝ) : ℝ := x^2 + p1 * x + q1
def p2 (x : ℝ) (p2 q2 : ℝ) : ℝ := x^2 + p2 * x + q2
def p3 (x : ℝ) (p3 q3 : ℝ) : ℝ := x^2 + p3 * x + q3

-- Statement of the problem
theorem equation_has_8_roots (p1 p2 p3 q1 q2 q3 : ℝ) :
  (∃ (x : ℝ), |p1 x p1 q1| + |p2 x p2 q2| = |p3 x p3 q3|)
  → 8 := sorry

end equation_has_8_roots_l397_397717


namespace ball_curve_distance_l397_397142

theorem ball_curve_distance :
  ∀ (x : ℝ), (x^2 + x - 12 = 2) → (y : ℝ), (y^2 + y - 12 = -10) → 
  (abs ((-1 + real.sqrt 57) / 2 - (-2)) = 5.275 ∨ abs ((-1 - real.sqrt 57) / 2 - 1) = 5.275) :=
begin
  sorry
end

end ball_curve_distance_l397_397142


namespace investment_time_p_l397_397073

theorem investment_time_p (p_investment q_investment p_profit q_profit : ℝ) (p_invest_time : ℝ) (investment_ratio_pq : p_investment / q_investment = 7 / 5.00001) (profit_ratio_pq : p_profit / q_profit = 7.00001 / 10) (q_invest_time : q_invest_time = 9.999965714374696) : p_invest_time = 50 :=
sorry

end investment_time_p_l397_397073


namespace area_of_sector_l397_397825

noncomputable def radius : ℝ := 4
noncomputable def theta1 : ℝ := π / 3
noncomputable def theta2 : ℝ := 2 * π / 3

theorem area_of_sector : 
  let θ := theta2 - theta1,
      r := radius 
  in
  (1 / 2) * r^2 * θ = (8 * π) / 3 :=
by
  sorry

end area_of_sector_l397_397825


namespace length_of_BC_l397_397927

-- Definitions based on the conditions provided in the original problem
def triangle (A B C : Type) [metric_space A] := true

def is_right_triangle {A B C : Type} [metric_space A] [metric_space B] [metric_space C] 
  (AB AC BC : ℝ) (angle_BAC : ℝ) : Prop := (angle_BAC = 90)

def has_median_AD {A B C D : Type} [metric_space A]
  (AB AC BC AD : ℝ) : Prop := AD = (1 / 2) * real.sqrt (2 * AB^2 + 2 * AC^2 - BC^2)

-- The proof problem statement
theorem length_of_BC 
  {A B C D : Type} [metric_space A]
  (AB AC BC AD : ℝ)
  (h1 : AB = 6) 
  (h2 : AC = 8) 
  (h3 : AD = 4.5)
  (h4 : is_right_triangle AB AC BC 90) 
  (h5 : has_median_AD AB AC BC AD) :
  BC = real.sqrt 119 :=
by sorry

end length_of_BC_l397_397927


namespace product_odd_probability_l397_397859

open Finset
open BigOperators

-- Definition of the set and probability calculation
def set : Finset ℕ := {1, 2, 3, 4, 5, 6}
def odd_numbers : Finset ℕ := {1, 3, 5}

-- Calculation of combinations
def total_combinations := (set.choose 3).card
def odd_combinations := (odd_numbers.choose 3).card

-- Probability of selecting three numbers such that their product is odd
def probability_of_odd_product := (odd_combinations : ℚ) / (total_combinations : ℚ)

theorem product_odd_probability :
  probability_of_odd_product = 1 / 20 :=
by
  -- This is where the proof would go
  sorry

end product_odd_probability_l397_397859


namespace third_number_in_sequence_l397_397848

def is_prime (n : ℕ) : Prop := nat.prime n

def is_1n1_format (n : ℕ) : Prop := 
  match n with
  | 101 => true
  | 111 => true
  | 117 => true
  | 119 => true
  | 123 => true
  | 129 => true
  | _ => false

theorem third_number_in_sequence : ∃ n : ℕ, 
  n = 101 ∧ 
  ((¬ is_prime 12) ∧ is_prime 13 ∧ (¬ is_prime 12 ∨ is_1n1_format 12)) ∧
  ∃ (a b c d e f : ℕ),
    (a = 12 ∧ b = 13 ∧ d = 17 ∧
     e = 111 ∧ f = 113 ∧
     (n = 101 ∧ is_1n1_format n) ∧
     a < b < c < d < e < f < 131 ∧
     is_1n1_format e ∧ is_prime f) := 
sorry

end third_number_in_sequence_l397_397848


namespace PQ_eq_OO_l397_397505

theorem PQ_eq_OO'
  (O O' : Circle)
  (A B P Q R : Point)
  (h_rad_eq : O.radius = O'.radius)
  (h_A_in_O : O.contains A)
  (h_A_in_O' : O'.contains A)
  (h_B_in_O : O.contains B)
  (h_B_in_O' : O'.contains B)
  (h_AB_neq : A ≠ B)
  (h_P_in_O : O.contains P)
  (h_Q_in_O' : O'.contains Q)
  (h_P_neq_AB : P ≠ A ∧ P ≠ B)
  (h_Q_neq_AB : Q ≠ A ∧ Q ≠ B)
  (h_PAQR_parallelogram : Parallelogram P A Q R)
  (h_BRQP_cyclic : CyclicQuadrilateral B R Q P) :
  distance P Q = distance (center O) (center O') := sorry

end PQ_eq_OO_l397_397505


namespace q_has_1970_roots_l397_397779

-- Conditions
def n : ℕ := sorry
def a : Fin n → ℝ := sorry
def p (x : ℝ) : ℝ := (x ^ n) + List.sum (List.ofFn (fun i => (a ⟨i, sorry⟩) * x ^ i))
def q (x : ℝ) : ℝ := ∏ j in (Finset.range 2015).map Fin.val.succ, p (x + j)
def condition_p_2015 : p 2015 = 2015 := sorry

-- Theorem to Prove
theorem q_has_1970_roots (h_n : n > 1) (h_roots_p : ∃ xs : Fin n → ℝ, ∀ i, p (xs i) = 0) :
  ∃ rs : Fin 1970 → ℝ, (∀ j, |rs j| < 2015) ∧ ∀ i ≠ j, rs i ≠ rs j ∧ q (rs i) = 0 := 
sorry

end q_has_1970_roots_l397_397779


namespace decagon_triangle_probability_l397_397544

theorem decagon_triangle_probability :
  let n := 10 in
  let total_ways := Nat.choose n 3 in
  let favorable_ways := n in
  (favorable_ways / total_ways : ℚ) = 1 / 12 :=
by
  sorry

end decagon_triangle_probability_l397_397544


namespace max_value_fg_gh_hj_fj_l397_397491

theorem max_value_fg_gh_hj_fj : 
  ∃ (f g h j : ℕ), 
  f ∈ {6, 7, 8, 9} ∧ 
  g ∈ {6, 7, 8, 9} ∧ 
  h ∈ {6, 7, 8, 9} ∧ 
  j ∈ {6, 7, 8, 9} ∧ 
  f ≠ g ∧ g ≠ h ∧ h ≠ j ∧ j ≠ f ∧ f ≠ h ∧ g ≠ j ∧
  (fg + gh + hj + fj = 225) :=
sorry

end max_value_fg_gh_hj_fj_l397_397491


namespace sin_pi_div_five_lt_cos_pi_div_five_l397_397955

theorem sin_pi_div_five_lt_cos_pi_div_five :
  let f := λ x => Real.sin x
  let g := λ x => Real.cos x
  f (Real.pi / 5) < g (Real.pi / 5) :=
by
  have h1 : g (Real.pi / 5) = f (3 * Real.pi / 10) := sorry
  have h2 : ∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y := sorry
  sorry

end sin_pi_div_five_lt_cos_pi_div_five_l397_397955


namespace part1_solution_part2_solution_l397_397708

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

def f (x : ℝ) : ℝ := abs x

theorem part1_solution (x : ℝ) : f(x) + f(x - 1) ≤ 2 ↔ x ∈ set.Icc (-1/2 : ℝ) (3/2 : ℝ) := by sorry

theorem part2_solution (a b : ℝ) (x m : ℝ) (h : a + b = 1) (ha : 0 < a) (hb : 0 < b) :
  f(x - m) - abs(x + 2) ≤ 1/a + 1/b ↔ -6 ≤ m ∧ m ≤ 2 := by 
  have h1 : 1/a + 1/b = 4 := sorry
  sorry

end part1_solution_part2_solution_l397_397708


namespace sum_of_fractions_l397_397221

theorem sum_of_fractions : (1 / 3 : ℚ) + (2 / 7) = 13 / 21 :=
by
  sorry

end sum_of_fractions_l397_397221


namespace Alex_Hours_Upside_Down_Per_Month_l397_397172

-- Define constants and variables based on the conditions
def AlexCurrentHeight : ℝ := 48
def AlexRequiredHeight : ℝ := 54
def NormalGrowthPerMonth : ℝ := 1 / 3
def UpsideDownGrowthPerHour : ℝ := 1 / 12
def MonthsInYear : ℕ := 12

-- Compute total growth needed and additional required growth terms
def TotalGrowthNeeded : ℝ := AlexRequiredHeight - AlexCurrentHeight
def NormalGrowthInYear : ℝ := NormalGrowthPerMonth * MonthsInYear
def AdditionalGrowthNeeded : ℝ := TotalGrowthNeeded - NormalGrowthInYear
def TotalUpsideDownHours : ℝ := AdditionalGrowthNeeded * 12
def UpsideDownHoursPerMonth : ℝ := TotalUpsideDownHours / MonthsInYear

-- The statement to prove
theorem Alex_Hours_Upside_Down_Per_Month : UpsideDownHoursPerMonth = 2 := by
  sorry

end Alex_Hours_Upside_Down_Per_Month_l397_397172


namespace second_derivative_ln_at_one_l397_397736

-- Define the function f(x) = ln(x)
def f (x : ℝ) : ℝ := Real.log x

-- State the problem to prove that the second derivative at x = 1 is -1
theorem second_derivative_ln_at_one : deriv (deriv f) 1 = -1 := 
by
  sorry

end second_derivative_ln_at_one_l397_397736


namespace max_x_possible_value_l397_397103

theorem max_x_possible_value : ∃ x : ℚ, 
  (∃ y : ℚ, y = (5 * x - 20) / (4 * x - 5) ∧ (y^2 + y = 20)) ∧
  x = 9 / 5 :=
begin
  sorry
end

end max_x_possible_value_l397_397103


namespace white_space_area_is_31_l397_397494

-- Definitions and conditions from the problem
def board_width : ℕ := 4
def board_length : ℕ := 18
def board_area : ℕ := board_width * board_length

def area_C : ℕ := 4 + 2 + 2
def area_O : ℕ := (4 * 3) - (2 * 1)
def area_D : ℕ := (4 * 3) - (2 * 1)
def area_E : ℕ := 4 + 3 + 3 + 3

def total_black_area : ℕ := area_C + area_O + area_D + area_E

def white_space_area : ℕ := board_area - total_black_area

-- Proof problem statement
theorem white_space_area_is_31 : white_space_area = 31 := by
  sorry

end white_space_area_is_31_l397_397494


namespace maximum_cos_product_l397_397686

theorem maximum_cos_product {α β γ : ℝ} (h : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) :
  cos α * cos β * cos γ ≤ 2 * Real.sqrt 6 / 9 :=
sorry

end maximum_cos_product_l397_397686


namespace increasing_interval_of_f_l397_397734

noncomputable def f (a x : ℝ) : ℝ := log a (2 * x^2 + x)

theorem increasing_interval_of_f (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 1/2 → 0 < log a (2 * x^2 + x)) :
  ∀ x : ℝ, x < -1/2 → (∀ y : ℝ, y < x → f a y < f a x) :=
sorry

end increasing_interval_of_f_l397_397734


namespace evaluate_g_3_times_l397_397416

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2 * n - 1
  else 2 * n + 3

theorem evaluate_g_3_times : g (g (g 3)) = 65 := by
  sorry

end evaluate_g_3_times_l397_397416


namespace axis_of_symmetry_l397_397314

-- Given the condition that ∀ x, g(x) = g(3 - x)
-- Prove that x = 1.5 is an axis of symmetry for the graph y = g(x)

variable (g : ℝ → ℝ)
theorem axis_of_symmetry (h : ∀ x, g(x) = g(3 - x)) : 
  ∀ x, g (x) = g (1.5 * 2 - x) :=
by 
  sorry

end axis_of_symmetry_l397_397314


namespace acute_angle_at_7_20_is_100_degrees_l397_397513

theorem acute_angle_at_7_20_is_100_degrees :
  let minute_hand_angle := 4 * 30 -- angle of the minute hand (in degrees)
  let hour_hand_progress := 20 / 60 -- progress of hour hand between 7 and 8
  let hour_hand_angle := 7 * 30 + hour_hand_progress * 30 -- angle of the hour hand (in degrees)

  ∃ angle_acute : ℝ, 
  angle_acute = abs (minute_hand_angle - hour_hand_angle) ∧
  angle_acute = 100 :=
by
  sorry

end acute_angle_at_7_20_is_100_degrees_l397_397513


namespace distance_between_perpendiculars_is_constant_l397_397097

-- Define the problem in Lean 4
structure Rectangle :=
  (A B C D O : Point)
  (AC BD : Line)
  (circumcircle : Circle)
  (O_center : circumcircle.center = O)

axiom Rectangle_diagonals_intersect {rect : Rectangle} :
  intersection rect.AC rect.BD = {rect.O}

axiom Point_on_circumcircle (rect : Rectangle) (P : Point) :
  P ∈ rect.circumcircle

axiom Perpendiculars_from_P (rect : Rectangle) (P : Point) :
  ∃ Q R : Point, perpendicular_from P to rect.AC = Q ∧ perpendicular_from P to rect.BD = R

-- The theorem we want to prove
theorem distance_between_perpendiculars_is_constant (rect : Rectangle) :
  ∀ (P : Point), P ∈ rect.circumcircle →
  ∃ Q R : Point, perpendicular_from P to rect.AC = Q ∧ perpendicular_from P to rect.BD = R ∧
  (distance Q R = some_constant_length) :=
by
  sorry

end distance_between_perpendiculars_is_constant_l397_397097


namespace number_div_by_4_is_even_iff_not_even_not_div_by_4_l397_397844

open Classical

theorem number_div_by_4_is_even_iff_not_even_not_div_by_4 :
  (∀ n : ℤ, n % 4 = 0 → n % 2 = 0) ↔ (∀ n : ℤ, ¬(n % 2 = 0) → ¬(n % 4 = 0)) :=
begin
  sorry
end

end number_div_by_4_is_even_iff_not_even_not_div_by_4_l397_397844


namespace minimally_intersecting_quadruples_mod_1000_l397_397203

open Set

/-- Define function to count minimally intersecting quadruple sets modulo 1000 -/
noncomputable def count_minimally_intersecting_quadruples : ℕ :=
let universe := {1, 2, 3, 4, 5, 6, 7, 8} in
let count_ways :=
  ((Finset.card universe).choose 4) * (5 ^ 4) in
count_ways % 1000

theorem minimally_intersecting_quadruples_mod_1000 :
  count_minimally_intersecting_quadruples = 0 :=
sorry

end minimally_intersecting_quadruples_mod_1000_l397_397203


namespace zero_of_f_in_interval_l397_397060

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 / x

theorem zero_of_f_in_interval : ∃ x ∈ Ioo (Real.exp 1) 3, f x = 0 := sorry

end zero_of_f_in_interval_l397_397060


namespace ilya_pasha_exchange_eq_l397_397855

theorem ilya_pasha_exchange_eq (a b : ℕ) :
  ∀ n, 
  (∃ (u : ℕ → ℕ), 
    (∀ i, 1 ≤ i ∧ i ≤ b → u i = 0) ∧
    (a = (∑ i, u i)) ∧
    (n = (∑ i, (i * u i)))) ↔
  (∃ (v : ℕ → ℕ),
    (∀ j, 1 ≤ j ∧ j ≤ b → v j = 0) ∧
    ((a + b) = (∑ j, v j)) ∧
    (n = (∑ j, (j * v j)))) :=
sorry

end ilya_pasha_exchange_eq_l397_397855


namespace decagon_triangle_probability_l397_397545

theorem decagon_triangle_probability :
  let n := 10 in
  let total_ways := Nat.choose n 3 in
  let favorable_ways := n in
  (favorable_ways / total_ways : ℚ) = 1 / 12 :=
by
  sorry

end decagon_triangle_probability_l397_397545


namespace find_number_and_remainder_l397_397158

theorem find_number_and_remainder :
  ∃ (N r : ℕ), (3927 + 2873) * (3 * (3927 - 2873)) + r = N ∧ r < (3927 + 2873) :=
sorry

end find_number_and_remainder_l397_397158


namespace alice_needs_7_fills_to_get_3_cups_l397_397928

theorem alice_needs_7_fills_to_get_3_cups (needs : ℚ) (cup_size : ℚ) (has : ℚ) :
  needs = 3 ∧ cup_size = 1 / 3 ∧ has = 2 / 3 →
  (needs - has) / cup_size = 7 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end alice_needs_7_fills_to_get_3_cups_l397_397928


namespace right_triangle_AB_length_l397_397758

theorem right_triangle_AB_length
  (A B C : Type)
  [has_angle A B = real.pi / 2]
  [has_angle C B A = real.pi / 180 * 40]
  [BC = 7] : 
  AB ≈ 8.3 := 
by sorry

end right_triangle_AB_length_l397_397758


namespace minimum_rooms_to_accommodate_fans_l397_397983

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l397_397983


namespace range_of_a_l397_397705

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → x1 < -1/2 → x2 < -1/2 →
    (log (1 / 3) (x2^2 - a * x2 - a) - log (1 / 3) (x1^2 - a * x1 - a)) / (x2 - x1) > 0) →
  (-1 ≤ a ∧ a ≤ 1 / 2) :=
by
  sorry

end range_of_a_l397_397705


namespace largest_number_of_square_test_plots_l397_397922

theorem largest_number_of_square_test_plots :
  ∀ (length width fence_length : ℕ), 
  length = 60 → width = 30 → fence_length = 2500 →
  ∃ (n : ℕ), n = 8 ∧ 
  (∃ (s : ℕ), 
      (width % s = 0) ∧ (length % s = 0) ∧ 
      (3600 - 90 * s ≤ fence_length) ∧ 
      n = (width * length) / s^2) :=
begin
  intros length width fence_length h1 h2 h3,
  use 8,
  split,
  { refl },
  sorry
end

end largest_number_of_square_test_plots_l397_397922


namespace number_of_games_in_season_l397_397131

theorem number_of_games_in_season 
  (num_teams : ℕ)
  (games_per_pair : ℕ)
  (h_num_teams : num_teams = 14)
  (h_games_per_pair : games_per_pair = 5)
  : (num_teams * (num_teams - 1) / 2) * games_per_pair = 455 :=
by
  -- subgoals: show each intermediate calculation matches
  rw [h_num_teams, h_games_per_pair]
  -- Intermediate steps can be added if necessary for detailed breakdown
  calc 
    (14 * (14 - 1) / 2) * 5 = (14 * 13 / 2) * 5 : by sorry
                     ... = (182 / 2) * 5 : by sorry
                     ... = 91 * 5 : by sorry
                     ... = 455 : by sorry

end number_of_games_in_season_l397_397131


namespace sqrt_factorial_product_l397_397951

open Nat

theorem sqrt_factorial_product (h : fact 4 = 24) : sqrt (fact 4 * fact 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l397_397951


namespace greatest_num_trucks_rented_l397_397026

theorem greatest_num_trucks_rented
  (total_trucks : ℕ)
  (rented_pct : ℕ)
  (returned_pct : ℕ)
  (at_least_trucks_on_lot : ℕ)
  (total_trucks_eq : total_trucks = 20)
  (rented_pct_eq : rented_pct = 100)
  (returned_pct_eq : returned_pct = 50)
  (at_least_trucks_on_lot_eq : at_least_trucks_on_lot = 10) :
  let R := 2 * at_least_trucks_on_lot in
  R = 20 :=
by
  let R := (2 * at_least_trucks_on_lot) / (returned_pct / 100)
  have R_eq : R = total_trucks_eq by sorry
  exact R_eq

end greatest_num_trucks_rented_l397_397026


namespace last_sampled_student_id_l397_397503

theorem last_sampled_student_id (total_students : ℕ) (sample_size : ℕ) (first_sample_id : ℕ) :
  total_students = 2000 ∧ sample_size = 50 ∧ first_sample_id = 3 →
  ∃ last_sample_id : ℕ, last_sample_id = 1963 :=
by
  intro h,
  obtain ⟨htotal, hsize, hfirst⟩ := h,
  have sampling_interval := total_students / sample_size,
  have last_position := sample_size,
  have last_sample_id := first_sample_id + sampling_interval * (last_position - 1),
  use last_sample_id,
  sorry

end last_sampled_student_id_l397_397503


namespace production_days_l397_397892

noncomputable def daily_production (n : ℕ) : Prop :=
50 * n + 90 = 58 * (n + 1)

theorem production_days (n : ℕ) (h : daily_production n) : n = 4 :=
by sorry

end production_days_l397_397892


namespace g_is_odd_l397_397391

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  sorry

end g_is_odd_l397_397391


namespace sum_of_numbers_mod_11_l397_397653

def sum_mod_eleven : ℕ := 123456 + 123457 + 123458 + 123459 + 123460 + 123461

theorem sum_of_numbers_mod_11 :
  sum_mod_eleven % 11 = 10 :=
sorry

end sum_of_numbers_mod_11_l397_397653


namespace rooms_needed_l397_397991

-- Defining all necessary conditions
constant fans : ℕ := 100
constant teams : ℕ := 3
constant max_people_per_room : ℕ := 3
constant groups : ℕ := 6

-- Proposition that the number of rooms required is exactly 37.
theorem rooms_needed (h1 : ∀ i, 1 ≤ i ∧ i ≤ groups) 
  (h2 : groups = 2 * teams) 
  (h3 : teams = 3)
  (h4 : max_people_per_room = 3)
  (h5 : ∃ i, 0 ≤ i ∧ ∑ i in finset.range groups, i = fans):
  37 = (∃ (n : ℕ), ∃ (i : ℕ), n = ∑ i in finset.range groups, nat.ceil (i / max_people_per_room)):
sorry

end rooms_needed_l397_397991


namespace seating_arrangements_l397_397235

noncomputable theory

def jupiterian_seats := {1}
def neptunian_seats := {12}
def total_seats := 12

-- Define committee structure and rules
inductive Species
| Jupiterian
| Saturnian
| Neptunian

def committee_seats := list Species

def valid_seating_arrangement (seats : committee_seats) : Prop :=
  (seats.length = total_seats) ∧
  (seats.head = Species.Jupiterian) ∧
  (seats.last = Species.Neptunian) ∧
  (∀ i, (i > 1) → (i ≤ total_seats) →
    ((seats.nth i = some Species.Jupiterian → seats.nth (i - 1) ≠ some Species.Saturnian) ∧
     (seats.nth i = some Species.Saturnian → seats.nth (i - 1) ≠ some Species.Neptunian) ∧
     (seats.nth i = some Species.Neptunian → seats.nth (i - 1) ≠ some Species.Jupiterian)))

-- Definitions for calculating the number of valid arrangements
def num_valid_arrangements : ℕ

theorem seating_arrangements : num_valid_arrangements = 56 * (nat.factorial 4) ^ 3 :=
sorry

end seating_arrangements_l397_397235


namespace find_c_l397_397710

-- Definition of the function f
def f (x a b c : ℤ) : ℤ := x^3 + a * x^2 + b * x + c

-- Theorem statement
theorem find_c (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : f a a b c = a^3)
  (h4 : f b a b c = b^3) : c = 16 :=
by
  sorry

end find_c_l397_397710


namespace intersection_M_P_l397_397012

variable {x a : ℝ}

def M (a : ℝ) : Set ℝ := { x | x > a ∧ a^2 - 12*a + 20 < 0 }
def P : Set ℝ := { x | x ≤ 10 }

theorem intersection_M_P (a : ℝ) (h : 2 < a ∧ a < 10) : 
  M a ∩ P = { x | a < x ∧ x ≤ 10 } :=
sorry

end intersection_M_P_l397_397012


namespace correct_answer_l397_397593

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

def problem_statement : Prop :=
(is_odd (λ x, |x|) ∨ is_increasing_on (λ x, |x|) (-1) 1) ∧
(is_odd (λ x, real.sin x) ∧ is_increasing_on (λ x, real.sin x) (-1) 1) ∧
(is_odd (λ x, exp x + exp (-x)) ∨ is_increasing_on (λ x, exp x + exp (-x)) (-1) 1) ∧
(is_odd (λ x, -x^3) ∧ is_increasing_on (λ x, -x^3) (-1) 1)

theorem correct_answer : is_odd (λ x, real.sin x) ∧ is_increasing_on (λ x, real.sin x) (-1) 1 :=
sorry

end correct_answer_l397_397593


namespace number_of_digits_l397_397212

theorem number_of_digits (x : ℕ) (y : ℕ) (z : ℕ) (h : x = 2 ∧ y = 15 ∧ 5 = 5 ∧ z = 3) :
    (natDigits 10 (x ^ 15 * 5 ^ 10 * z)).length = 12 := by sorry

end number_of_digits_l397_397212


namespace concurrency_of_tangent_lines_l397_397418

open EuclideanGeometry

-- Definitions from conditions
variable {A B C : Point}
variable {Γ : Circle}
variable {GammaA GammaB GammaC : Circle}
variable {A' B' C' : Point}

-- Conditions
def circle_tangent_to_sides (ΓA : Circle) (AB AC : Segment) (Γ : Circle) (A' : Point) :=
  (ΓA.tangentTo AB) ∧ (ΓA.tangentTo AC) ∧ (ΓA.tangentTo Γ) ∧ (ΓA.center ∈ ΓInside)

axiom circumscribed_circle (ABC : Triangle) : Γ
axiom tangent_circle_A (ABC : Triangle) (Γ : Circle) : ∃ ΓA A', circle_tangent_to_sides ΓA ABC.AB ABC.AC Γ A'
axiom tangent_circle_B (ABC : Triangle) (Γ : Circle) : ∃ ΓB B', circle_tangent_to_sides ΓB ABC.BC ABC.BA Γ B'
axiom tangent_circle_C (ABC : Triangle) (Γ : Circle) : ∃ ΓC C', circle_tangent_to_sides ΓC ABC.CA ABC.CB Γ C'

-- Proof problem statement
theorem concurrency_of_tangent_lines (ABC : Triangle) :
  let Γ := circumscribed_circle ABC in
  (∃ (ΓA : Circle) (A' : Point), circle_tangent_to_sides ΓA ABC.AB ABC.AC Γ A') →
  (∃ (ΓB : Circle) (B' : Point), circle_tangent_to_sides ΓB ABC.BC ABC.BA Γ B') →
  (∃ (ΓC : Circle) (C' : Point), circle_tangent_to_sides ΓC ABC.CA ABC.CB Γ C') →
  ∃ X : Point, collinear [ABC.A, A', X] ∧ collinear [ABC.B, B', X] ∧ collinear [ABC.C, C', X]
:= by sorry

end concurrency_of_tangent_lines_l397_397418


namespace four_prime_prime_l397_397893

-- Define the function based on the given condition
def q' (q : ℕ) : ℕ := 3 * q - 3

-- The statement to prove
theorem four_prime_prime : (q' (q' 4)) = 24 := by
  sorry

end four_prime_prime_l397_397893


namespace maximum_at_a_implies_a_in_interval_l397_397282

theorem maximum_at_a_implies_a_in_interval (a : ℝ) (f : ℝ → ℝ) 
  (h_deriv : ∀ x, f' x = a * (x + 1) * (x - a)) 
  (h_max : ∀ x, diff_f f a = 0 ∧ (∀ y, y ≠ x → f y < f x)) :
  a ∈ set.Ioo (-1 : ℝ) 0 :=
sorry

end maximum_at_a_implies_a_in_interval_l397_397282


namespace equation_solution_l397_397815

theorem equation_solution (x y z : ℕ) :
  x^2 + y^2 = 2^z ↔ ∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2 * n + 1 := 
sorry

end equation_solution_l397_397815


namespace quadratic_root_s_val_l397_397689

theorem quadratic_root_s_val (r s : ℝ) (h : IsRoot ((fun x => 2 * x ^ 2 + r * x + s)) (4 + 3 * Complex.i)) : 
  s = 50 := 
by
  sorry

end quadratic_root_s_val_l397_397689


namespace right_triangle_roots_l397_397003

noncomputable theory

open Complex

def right_triangle (z0 z1 z2 : ℂ) : Prop :=
(z1 - z0) * (z2 - z0) ∈ {z : ℂ | arg z = π / 2}

theorem right_triangle_roots (p q z1 z2 : ℂ)
    (hroots : ∀z, z^2 + p * z + q = 0 → z = z1 ∨ z = z2)
    (htriangle : right_triangle 0 z1 z2) :
    (p * p / q) = 2 :=
sorry

end right_triangle_roots_l397_397003


namespace product_invariance_of_tangent_line_l397_397489

noncomputable def rhombus_inscribed_circle_product_invariance
  (Rhombus : Type)
  [Plane_geometric_figure Rhombus]
  (inscribed_circle : Circle Rhombus)
  (tangent_line : Line Rhombus → Circle Rhombus → Point Rhombus × Point Rhombus)
  (A B C : Point Rhombus)
  (K L : Point Rhombus)
  (α : ℝ) : Prop :=
∀ (l : Line Rhombus),
  let E := tangent_line l inscribed_circle in
  let F := tangent_line l inscribed_circle in
  let AE := distance A E.1 in
  let FC := distance F.2 C in
  AE * FC = invariant_distance (A B C K L α)

theorem product_invariance_of_tangent_line
  {Rhombus : Type}
  [Plane_geometric_figure Rhombus]
  {inscribed_circle : Circle Rhombus}
  {tangent_line : Line Rhombus → Circle Rhombus → Point Rhombus × Point Rhombus}
  {A B C : Point Rhombus}
  {K L : Point Rhombus}
  {α : ℝ} :
  rhombus_inscribed_circle_product_invariance Rhombus inscribed_circle tangent_line A B C K L α :=
sorry

end product_invariance_of_tangent_line_l397_397489


namespace kombucha_bottles_l397_397726

theorem kombucha_bottles (b_m : ℕ) (c : ℝ) (r : ℝ) (m : ℕ)
  (hb : b_m = 15) (hc : c = 3.00) (hr : r = 0.10) (hm : m = 12) :
  (b_m * m * r) / c = 6 := by
  sorry

end kombucha_bottles_l397_397726


namespace potential_values_of_k_l397_397617

theorem potential_values_of_k :
  ∃ k : ℚ, ∀ (a b : ℕ), 
  (10 * a + b = k * (a + b)) ∧ (10 * b + a = (13 - k) * (a + b)) → k = 11/2 :=
by
  sorry

end potential_values_of_k_l397_397617


namespace g_is_odd_l397_397394

def g (x : ℝ) : ℝ := 1 / (3 ^ x - 1) + 1 / 3

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_is_odd_l397_397394


namespace trapezoid_is_plane_figure_l397_397883

-- Definitions related to the basic properties of planes

-- Three non-collinear points determine a plane
def three_points_determine_plane (P Q R : Set Point) (non_collinear : ¬ collinear P Q R) : Prop :=
  ∃ α : Plane, P ∈ α ∧ Q ∈ α ∧ R ∈ α

-- A quadrilateral could also be a spatial quadrilateral
def is_spatial_quadrilateral (quad : Quadrilateral) : Prop :=
  ↑quad ∉ ⋂ (α : Plane), α

-- Two parallel lines determine a plane
def parallel_lines_determine_plane (l m : Line) (parallel : Parallel l m) : Prop :=
  ∃ α : Plane, l ∈ α ∧ m ∈ α

-- Points at the intersection of two planes are collinear
def planes_intersection_collinear (α β : Plane) : Prop :=
  ∀ (P Q R : Point), P ∈ α ∧ P ∈ β ∧ Q ∈ α ∧ Q ∈ β ∧ R ∈ α ∧ R ∈ β → collinear P Q R

-- Statement of the theorem
theorem trapezoid_is_plane_figure (T : Trapezoid) : 
  (three_points_determine_plane P Q R non_collinear) →
  (¬ is_spatial_quadrilateral quad) →
  (parallel_lines_determine_plane l m parallel) →
  (planes_intersection_collinear α β) →
  IsPlaneFigure T := sorry

end trapezoid_is_plane_figure_l397_397883


namespace number_of_perfect_square_divisors_of_P_l397_397948

def product_factorial (n : ℕ) : ℕ := (list.range' 1 n).prod!

def P : ℕ := product_factorial 11

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_perfect_square_divisors (n : ℕ) : ℕ :=
  (list.range' 1 (n+1)).filter (λ d, n % d = 0 ∧ is_perfect_square d).length

theorem number_of_perfect_square_divisors_of_P : count_perfect_square_divisors P = 1120 := 
by
  sorry

end number_of_perfect_square_divisors_of_P_l397_397948


namespace hotel_room_allocation_l397_397973

theorem hotel_room_allocation :
  ∀ (fans : ℕ) (num_groups : ℕ) (room_capacity : ℕ), 
  fans = 100 → num_groups = 6 → room_capacity = 3 →
  (∀ (group_fans : ℕ) (groups : fin num_groups), group_fans ≤ fans →
   (∃ (total_rooms: ℕ), total_rooms = 37 ∧
    ∀ (group_fan_count : fin num_groups → ℕ), 
      (∀ g, group_fan_count g ≤ fans / num_groups + if (fans % num_groups > 0) then 1 else 0) →
      ∃ (rooms : fin num_groups → ℕ), 
        sum (λ g, rooms g) = total_rooms ∧ 
        ∀ g, group_fan_count g ≤ rooms g * room_capacity)) :=
by {
  intros,
  sorry
}

end hotel_room_allocation_l397_397973


namespace oranges_count_l397_397500

noncomputable def n : ℝ := 3.0
noncomputable def r : ℝ := 1.333333333
def total_oranges : ℝ := n * r

theorem oranges_count :
  total_oranges = 4 :=
by
  sorry

end oranges_count_l397_397500


namespace albums_created_l397_397396

def phone_pics : ℕ := 2
def camera_pics : ℕ := 4
def pics_per_album : ℕ := 2
def total_pics : ℕ := phone_pics + camera_pics

theorem albums_created : total_pics / pics_per_album = 3 := by
  sorry

end albums_created_l397_397396


namespace train_speed_l397_397926

def train_length : ℝ := 250
def bridge_length : ℝ := 150
def time_to_cross : ℝ := 32

theorem train_speed :
  (train_length + bridge_length) / time_to_cross = 12.5 :=
by {
  sorry
}

end train_speed_l397_397926


namespace alice_prime_sum_l397_397588

-- Conditions and problem definitions
def smallest_prime_divisor (n : ℕ) : ℕ :=
  if (n = 2) then 2 else (2 :: (List.range' 3 (n - 2)).filter (λ d, n % d = 0 ∧ Nat.Prime d)).head!

def next_number (a_k : ℕ) : ℕ := a_k - (smallest_prime_divisor a_k)

-- Proof problem
theorem alice_prime_sum : 
  ∀ (a_2022 : ℕ), 
    Nat.Prime a_2022 → 
    a_2022 = 2 → 
    (∃ a_0 : ℕ, 
     (∀ n < 2022, next_number (a_0 - (2 * n)) = a_0 - (2 * (n + 1))) ∧ 
     (a_0 - 4044 = a_2022)) → 
    (a_0 = 4046 ∨ a_0 = 4047) → 
    (4046 + 4047 = 8093) := 
by 
  sorry

end alice_prime_sum_l397_397588


namespace prime_count_between_50_and_100_with_prime_remainder_div_12_l397_397305

open Nat

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_between (a b : ℕ) : list ℕ :=
  (list.range' a (b - a + 1)).filter is_prime

def prime_remainders (p : ℕ) : list ℕ := [2, 3, 5, 7, 11]

theorem prime_count_between_50_and_100_with_prime_remainder_div_12 : 
  (primes_between 50 100).filter (λ p, (p % 12) ∈ prime_remainders p).length = 7 :=
by
  sorry

end prime_count_between_50_and_100_with_prime_remainder_div_12_l397_397305


namespace find_multiplier_l397_397117

theorem find_multiplier (N x : ℕ) (h₁ : N = 12) (h₂ : N * x - 3 = (N - 7) * 9) : x = 4 :=
by
  sorry

end find_multiplier_l397_397117


namespace minimum_ladder_rungs_l397_397176

theorem minimum_ladder_rungs (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b): ∃ n, n = a + b - 1 :=
by
    sorry

end minimum_ladder_rungs_l397_397176


namespace shaded_area_l397_397649

theorem shaded_area (side_len : ℕ) (triangle_base : ℕ) (triangle_height : ℕ)
  (h1 : side_len = 40) (h2 : triangle_base = side_len / 2)
  (h3 : triangle_height = side_len / 2) : 
  side_len^2 - 2 * (1/2 * triangle_base * triangle_height) = 1200 := 
  sorry

end shaded_area_l397_397649


namespace equally_likely_events_A_and_B_l397_397370

-- Given conditions definitions
def number_of_balls (N : ℕ) := N
def main_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 10
def additional_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 8
def lucky_sum_main (s : finset ℕ) := s.sum = 63
def lucky_sum_additional (s : finset ℕ) := s.sum = 44

-- Prove the events A and B are equally likely if and only if N = 18
theorem equally_likely_events_A_and_B (N : ℕ) :
  (∃ s ∈ main_draw_balls N, lucky_sum_main s) ∧ (∃ t ∈ additional_draw_balls N, lucky_sum_additional t)
  → (N = 18) :=
sorry

end equally_likely_events_A_and_B_l397_397370


namespace sequence_inequality_l397_397382

open Real

noncomputable def a_seq (t : ℝ) : ℕ → ℝ
| 1     := t
| 2     := t^2
| (n+1) := t * (a_seq t n + a_seq t (n-1))

noncomputable def b_seq (a_seq : ℕ → ℝ) (n : ℕ) : ℝ :=
2 * a_seq n / (1 + (a_seq n)^2)

theorem sequence_inequality (t : ℝ) (n : ℕ) (h_t_range : 1/2 < t ∧ t < 2) :
  (∑ i in Finset.range n, (1 / b_seq (a_seq t) (i + 1))) < 2^n - 2^(-n / 2) := sorry

end sequence_inequality_l397_397382


namespace monotonicity_intervals_when_a_equals_1_tangent_point_abscissa_is_one_l397_397417

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x - log x

theorem monotonicity_intervals_when_a_equals_1 :
  ∀ x : ℝ, (0 < x ∧ x < 1/2 → deriv (f 1) x < 0) ∧ (1/2 < x → deriv (f 1) x > 0) :=
by
  sorry

theorem tangent_point_abscissa_is_one :
  ∀ a : ℝ, let f := λ x : ℝ, x^2 + a * x - log x in
  ∃ t : ℝ, t = 1 ∧ (∀ x : ℝ, (x > 0 → deriv (f x) = f x / x) ∧ 
                            (f t / t = 2 * t + a - 1 / t) ∧ 
                            (t^2 - 1 + log t = 0)) :=
by
  sorry

end monotonicity_intervals_when_a_equals_1_tangent_point_abscissa_is_one_l397_397417


namespace equally_likely_events_A_and_B_l397_397372

-- Given conditions definitions
def number_of_balls (N : ℕ) := N
def main_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 10
def additional_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 8
def lucky_sum_main (s : finset ℕ) := s.sum = 63
def lucky_sum_additional (s : finset ℕ) := s.sum = 44

-- Prove the events A and B are equally likely if and only if N = 18
theorem equally_likely_events_A_and_B (N : ℕ) :
  (∃ s ∈ main_draw_balls N, lucky_sum_main s) ∧ (∃ t ∈ additional_draw_balls N, lucky_sum_additional t)
  → (N = 18) :=
sorry

end equally_likely_events_A_and_B_l397_397372


namespace bianca_final_deletion_l397_397885

namespace FileDeletion

variables (d_p d_s d_t d_v r_p r_v : ℕ)

def initial_deletion : ℕ := d_p + d_s + d_t + d_v

def files_restored : ℕ := r_p + r_v

def final_deletion : ℕ := initial_deletion d_p d_s d_t d_v - files_restored r_p r_v

theorem bianca_final_deletion :
  d_p = 5 ∧ d_s = 12 ∧ d_t = 10 ∧ d_v = 6 ∧ r_p = 3 ∧ r_v = 4 → final_deletion d_p d_s d_t d_v r_p r_v = 26 :=
begin
  intros h,
  sorry
end

end FileDeletion

end bianca_final_deletion_l397_397885


namespace expression_to_fraction_form_l397_397607

theorem expression_to_fraction_form :
  ∃ a b c : ℕ, c > 0 ∧
    let expr := (√6 + 1/√6 + √8 + 1/√8)
    let form := (a * √6 + b * √2) / c
    expr = form ∧ a + b + c = 280 :=
sorry

end expression_to_fraction_form_l397_397607


namespace triangle_great_iff_right_isosceles_l397_397873

theorem triangle_great_iff_right_isosceles (A B C D P Q : Type)
  [Triangle ABC]
  (D_on_BC : D ∈ line_segment BC)
  (P_perp_AB : ∀ D, line PQ ⊥ AB)
  (Q_perp_AC: ∀ D, line PQ ⊥ AC)
  (reflection_D_circum : ∀ D, reflection D (line PQ) ∈ circumcircle ABC) :
  (is_great_triangle ABC ↔ angle A = 90 ∧ side AB = side AC) :=
sorry

end triangle_great_iff_right_isosceles_l397_397873


namespace rooms_needed_l397_397992

-- Defining all necessary conditions
constant fans : ℕ := 100
constant teams : ℕ := 3
constant max_people_per_room : ℕ := 3
constant groups : ℕ := 6

-- Proposition that the number of rooms required is exactly 37.
theorem rooms_needed (h1 : ∀ i, 1 ≤ i ∧ i ≤ groups) 
  (h2 : groups = 2 * teams) 
  (h3 : teams = 3)
  (h4 : max_people_per_room = 3)
  (h5 : ∃ i, 0 ≤ i ∧ ∑ i in finset.range groups, i = fans):
  37 = (∃ (n : ℕ), ∃ (i : ℕ), n = ∑ i in finset.range groups, nat.ceil (i / max_people_per_room)):
sorry

end rooms_needed_l397_397992


namespace count_divisible_by_perfect_cube_l397_397230

/--
The number of positive integers less than 1,000,000 that are divisible by some perfect cube greater than 1 is 168089.
-/
theorem count_divisible_by_perfect_cube : 
  (finset.filter (λ n : ℕ, ∃ p : ℕ, nat.prime p ∧ p ^ 3 ∣ n) (finset.Ico 1 1000000)).card = 168089 :=
sorry

end count_divisible_by_perfect_cube_l397_397230


namespace arithmetic_seq_a7_value_l397_397266

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ): Prop := 
  ∀ n : ℕ, a (n+1) = a n + d

theorem arithmetic_seq_a7_value
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : a 4 = 4)
  (h3 : a 3 + a 8 = 5) :
  a 7 = 1 := 
sorry

end arithmetic_seq_a7_value_l397_397266


namespace min_braking_distance_l397_397828

theorem min_braking_distance :
  ∀ (v g μ a : ℝ),
  v = 30 →
  g = 10 →
  μ = (sin (real.pi / 6) / cos (real.pi / 6)) →
  a = μ * g →
  μ ≈ 0.577 →
  a ≈ 5.77 →
  (μ * g = a) →
  ∀ (S : ℝ), 
  S = (v^2) / (2 * a) →
  S ≈ 78 := by
  sorry

end min_braking_distance_l397_397828


namespace value_of_y_when_x_equals_8_l397_397492

variables (x y k : ℝ)

theorem value_of_y_when_x_equals_8 
  (hp : x * y = k)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (hx8 : x = 8) :
  y = 25 :=
sorry

end value_of_y_when_x_equals_8_l397_397492


namespace max_writers_editors_is_18_l397_397541

noncomputable def max_writers_editors (T W E : ℕ) (hE : E > 36) : ℕ :=
  let x := 100 - (W + E - x + 2 * x)
  x

theorem max_writers_editors_is_18 :
  max_writers_editors 100 45 37 (by norm_num) = 18 :=
sorry

end max_writers_editors_is_18_l397_397541


namespace second_part_of_ratio_l397_397556

theorem second_part_of_ratio (h_ratio : ∀ (x : ℝ), 25 = 0.5 * (25 + x)) : ∃ x : ℝ, x = 25 :=
by
  sorry

end second_part_of_ratio_l397_397556


namespace double_meat_sandwich_bread_count_l397_397508

theorem double_meat_sandwich_bread_count (x : ℕ) :
  14 * 2 + 12 * x = 64 → x = 3 := by
  intro h
  sorry

end double_meat_sandwich_bread_count_l397_397508


namespace items_in_storeroom_l397_397945

-- Conditions definitions
def restocked_items : ℕ := 4458
def sold_items : ℕ := 1561
def total_items_left : ℕ := 3472

-- Statement of the proof
theorem items_in_storeroom : (total_items_left - (restocked_items - sold_items)) = 575 := 
by
  sorry

end items_in_storeroom_l397_397945


namespace decagon_triangle_probability_l397_397543

theorem decagon_triangle_probability : 
  let total_vertices := 10
  let total_triangles := Nat.choose total_vertices 3
  let favorable_triangles := 10
  (total_triangles > 0) → 
  (favorable_triangles / total_triangles : ℚ) = 1 / 12 :=
by
  sorry

end decagon_triangle_probability_l397_397543


namespace total_activity_time_l397_397192

def jog_distance := 3 -- miles
def jog_speed := 6 -- miles per hour

def stroll_distance := 5 -- miles
def stroll_speed := 4 -- miles per hour

def walk_distance := 8 -- miles
def walk_speed := 5 -- miles per hour

def cooldown_distance := 1 -- mile
def cooldown_speed := 3 -- miles per hour

def jog_time := jog_distance / jog_speed  -- calculating time in hours for jog
def stroll_time := stroll_distance / stroll_speed  -- calculating time in hours for stroll
def walk_time := walk_distance / walk_speed  -- calculating time in hours for brisk walk
def cooldown_time := cooldown_distance / cooldown_speed  -- calculating time in hours for cooldown

noncomputable def total_time := jog_time + stroll_time + walk_time + cooldown_time

theorem total_activity_time : total_time = 3.6833 := by
  -- proof would go here
  sorry

end total_activity_time_l397_397192


namespace lucky_sum_equal_prob_l397_397362

theorem lucky_sum_equal_prob (N : ℕ) (S63_main_draw : Finset ℕ) (S44_additional_draw : Finset ℕ)
  (hN_pos : 0 < N)
  (hS63 : S63_main_draw.card = 10)
  (hSum63 : S63_main_draw.sum id = 63)
  (hS44 : S44_additional_draw.card = 8)
  (hSum44 : S44_additional_draw.sum id = 44)
  (hS_distinct : ∀ (n ∈ S63_main_draw ∪ S44_additional_draw), n ∈ Finset.range (N + 1)) :
  N = 18 :=
by
  sorry

end lucky_sum_equal_prob_l397_397362


namespace repeating_decimal_as_fraction_l397_397201

-- Define the repeating decimal 4.25252525... as x
def repeating_decimal : ℚ := 4 + 25 / 99

-- Theorem statement to prove the equivalence
theorem repeating_decimal_as_fraction :
  repeating_decimal = 421 / 99 :=
by
  sorry

end repeating_decimal_as_fraction_l397_397201


namespace unique_solution_single_element_l397_397279

theorem unique_solution_single_element (a : ℝ) 
  (h : ∀ x y : ℝ, (a * x^2 + a * x + 1 = 0) → (a * y^2 + a * y + 1 = 0) → x = y) : a = 4 := 
by
  sorry

end unique_solution_single_element_l397_397279


namespace smallest_digit_to_correct_sum_l397_397823

theorem smallest_digit_to_correct_sum (x y z w : ℕ) (hx : x = 753) (hy : y = 946) (hz : z = 821) (hw : w = 2420) :
  ∃ d, d = 7 ∧ (753 + 946 + 821 - 100 * d = 2420) :=
by sorry

end smallest_digit_to_correct_sum_l397_397823


namespace car_will_hit_house_l397_397756

def is_possible_to_avoid_house (grid : ℕ × ℕ → ℕ) (house : ℕ × ℕ) : Prop :=
  ∀ path : list (ℕ × ℕ), (∀ (x y : ℕ), (x < grid.1) ∧ (y < grid.2) → ∃ (next_x next_y : ℕ), (next_x < grid.1) ∧ (next_y < grid.2) ∧ ((next_x, next_y) ∈ grid)) →
  (∃ (start_x start_y : ℕ), ((start_x = 0 ∨ start_x = grid.1 - 1 ∨ start_y = 0 ∨ start_y = grid.2 - 1) → 
  ∀ (contains_path_to_house : ∃ (path_to_house : list (ℕ × ℕ)), path_to_house.head = (start_x, start_y) ∧ path_to_house.ilast = house, false))

theorem car_will_hit_house : ¬ is_possible_to_avoid_house (101, 101) (50, 50) :=
by
  sorry

end car_will_hit_house_l397_397756


namespace period_of_cos_cos_l397_397878

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.cos x)

theorem period_of_cos_cos :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
begin
  use π,
  refine ⟨Real.pi_pos, _⟩,
  sorry -- The rest of the proof goes here
end

end period_of_cos_cos_l397_397878


namespace shaded_area_of_rectangle_and_semicircles_l397_397380

theorem shaded_area_of_rectangle_and_semicircles :
  let s := 3 in   -- each segment length
  let diam_large := 7 * s in -- diameter of the large semicircle
  let radius_large := diam_large / 2 in -- radius of the large semicircle
  let area_large := (1/2:ℝ) * Real.pi * radius_large^2 in -- area of the large semicircle
  let area_small := (1/2:ℝ) * Real.pi * (s/2)^2 in -- area of a small semicircle
  let rect_length := diam_large in
  let rect_height := radius_large / 2 in
  let area_rect := rect_length * rect_height in
  let total_area_semicircles := area_large + 6 * area_small in -- total area of all semicircles
  let shaded_area := area_rect - total_area_semicircles
  shaded_area = (495/8:ℝ) * Real.pi :=
begin
 sorry,
end

end shaded_area_of_rectangle_and_semicircles_l397_397380


namespace triangle_side_c_l397_397262

-- Defining the main theorem.
theorem triangle_side_c (A B C : ℝ) (a b c : ℝ)
  (cos_A : cos A = 3 / 5)
  (cos_B : cos B = 5 / 13)
  (b_val : b = 3) :
  c = 14 / 5 := 
by
  sorry

end triangle_side_c_l397_397262


namespace curve_equation_l397_397565

section ParametricCurve

def parametrize_curve (t : ℝ) : ℝ × ℝ :=
  (3 * Real.sin t, Real.cos t + 2 * Real.sin t)

theorem curve_equation (a b c : ℝ) :
  (∀ t, let (x, y) := parametrize_curve t in a * x^2 + b * x * y + c * y^2 = 4) →
  (a, b, c) = (20 / 9 : ℝ, -16 / 3 : ℝ, 4 : ℝ) :=
by
  sorry

end ParametricCurve

end curve_equation_l397_397565


namespace min_rooms_needed_l397_397996

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l397_397996


namespace min_steps_to_determine_polynomial_l397_397046

theorem min_steps_to_determine_polynomial
  (P : ℕ → ℕ) 
  (hP : ∃ a b c : ℕ, P = λ x, a * x^2 + b * x + c ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :
  ∃ n, ∀ p, p = 2 :=
begin
  sorry
end

end min_steps_to_determine_polynomial_l397_397046


namespace animal_lifespan_probability_l397_397561

theorem animal_lifespan_probability
    (P_B : ℝ) (hP_B : P_B = 0.8)
    (P_A : ℝ) (hP_A : P_A = 0.4)
    : (P_A / P_B = 0.5) :=
by
    sorry

end animal_lifespan_probability_l397_397561


namespace sum_of_numbers_Carolyn_removes_l397_397818

/-
  Given n = 10 and Carolyn's first move removes 3,
  prove that the sum of the numbers Carolyn removes is 9.
-/

theorem sum_of_numbers_Carolyn_removes (n : ℕ) (h1 : n = 10) (c_first_removal : 3 ∈ [3]) : 
  ∃ (removed_numbers : list ℕ), (removed_numbers = [3, 6]) ∧ removed_numbers.sum = 9 :=
by
  sorry

end sum_of_numbers_Carolyn_removes_l397_397818


namespace petya_can_win_l397_397028

-- Define a game state
structure GameState where
  remainingDigits : List ℕ

-- Initial game state
def initialGameState : GameState := {
  remainingDigits := [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7]
}

-- Predicate to represent a winning strategy for Petya
def petyaWins (s : GameState) : Prop :=
  ∃ strategy: (GameState → Fin 10) → GameState, -- Exists a strategy function
  (∀ (move: GameState → Fin 10), 
    -- Applying the strategy results in a state where Petya wins
    strategy move = ⟨0, sorry⟩ -- "0" stands for Petya wins; replace "sorry" with valid proof term

theorem petya_can_win : petyaWins initialGameState := by
  sorry

end petya_can_win_l397_397028


namespace james_total_cost_l397_397774

def courseCost (units: Nat) (cost_per_unit: Nat) : Nat :=
  units * cost_per_unit

def totalCostForFall : Nat :=
  courseCost 12 60 + courseCost 8 45

def totalCostForSpring : Nat :=
  let science_cost := courseCost 10 60
  let science_scholarship := science_cost / 2
  let humanities_cost := courseCost 10 45
  (science_cost - science_scholarship) + humanities_cost

def totalCostForSummer : Nat :=
  courseCost 6 80 + courseCost 4 55

def totalCostForWinter : Nat :=
  let science_cost := courseCost 6 80
  let science_scholarship := 3 * science_cost / 4
  let humanities_cost := courseCost 4 55
  (science_cost - science_scholarship) + humanities_cost

def totalAmountSpent : Nat :=
  totalCostForFall + totalCostForSpring + totalCostForSummer + totalCostForWinter

theorem james_total_cost: totalAmountSpent = 2870 :=
  by sorry

end james_total_cost_l397_397774


namespace odd_and_monotonically_increasing_unique_l397_397840

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x

def is_monotonically_increasing (f : ℝ → ℝ) : Prop := ∀ a b, 0 < a ∧ a < b → f a < f b

theorem odd_and_monotonically_increasing_unique :
  is_odd (λ x : ℝ, x^3) ∧ is_monotonically_increasing (λ x : ℝ, x^3) ∧ 
  (∀ f : ℝ → ℝ, (f = (λ x, x⁻¹) ∨ f = sqrt ∨ f = exp) →
    ¬ (is_odd f ∧ is_monotonically_increasing f)) :=
begin
  sorry
end

end odd_and_monotonically_increasing_unique_l397_397840


namespace semicircle_area_l397_397164

theorem semicircle_area
  (length_rect : ℝ) (width_rect : ℝ) (hypotenuse_triangle : ℝ)
  (length_triangle_leg1 : ℝ) (length_triangle_leg2 : ℝ)
  (rect_inscribed: length_rect = 3 ∧ width_rect = 1) 
  (tri_inscribed: length_triangle_leg1 = 1 ∧ length_triangle_leg2 = sqrt 8)
  (valid_hypotenuse : hypotenuse_triangle = sqrt (length_triangle_leg1^2 + length_triangle_leg2^2) ∧ hypotenuse_triangle = length_rect) 
  : (area : ℝ) ∧ area = 9 * π / 8 :=
by
  sorry

end semicircle_area_l397_397164


namespace right_triangle_AB_length_l397_397759

theorem right_triangle_AB_length
  (A B C : Type)
  [has_angle A B = real.pi / 2]
  [has_angle C B A = real.pi / 180 * 40]
  [BC = 7] : 
  AB ≈ 8.3 := 
by sorry

end right_triangle_AB_length_l397_397759


namespace equally_likely_events_A_and_B_l397_397373

-- Given conditions definitions
def number_of_balls (N : ℕ) := N
def main_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 10
def additional_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 8
def lucky_sum_main (s : finset ℕ) := s.sum = 63
def lucky_sum_additional (s : finset ℕ) := s.sum = 44

-- Prove the events A and B are equally likely if and only if N = 18
theorem equally_likely_events_A_and_B (N : ℕ) :
  (∃ s ∈ main_draw_balls N, lucky_sum_main s) ∧ (∃ t ∈ additional_draw_balls N, lucky_sum_additional t)
  → (N = 18) :=
sorry

end equally_likely_events_A_and_B_l397_397373


namespace annual_interest_rate_equivalence_l397_397214

theorem annual_interest_rate_equivalence 
  (annual_rate : ℝ) (quarterly_rate : ℝ) (r : ℝ)
  (hrate : annual_rate = 0.08) 
  (qrate : quarterly_rate = annual_rate / 4) 
  (compounded_annually : (1 + quarterly_rate) ^ 4 - 1 = r) 
  : r ≈ 0.0824 :=
by 
  have q_rate := annual_rate / 4
  have e_arr := (1 + q_rate) ^ 4 - 1
  sorry

end annual_interest_rate_equivalence_l397_397214


namespace variance_of_shooter_score_l397_397925

theorem variance_of_shooter_score : 
  (10 + x + 10 + 8) / 4 = 9 → (x = 8) → 
  let μ := (10 + x + 10 + 8) / 4 in
  let σ² := ((10 - μ) ^ 2 + (x - μ) ^ 2 + (10 - μ) ^ 2 + (8 - μ) ^ 2) / 4 in
  σ² = 1 :=
sorry

end variance_of_shooter_score_l397_397925


namespace greatest_consecutive_integer_l397_397896

-- Definitions from conditions
def consecutive_integers (x : ℤ) := (x, x + 1, x + 2)
def sum_is_36 (x : ℤ) := x + (x + 1) + (x + 2) = 36

-- Statement of the problem
theorem greatest_consecutive_integer {x : ℤ} (h : sum_is_36 x) : 
  let (a, b, c) := consecutive_integers x in c = 13 :=
sorry

end greatest_consecutive_integer_l397_397896


namespace polynomial_roots_le_degree_l397_397430
-- Import necessary libraries

-- Define the polynomial structure and the conditions
variables {R : Type*} [CommRing R]
variables (P : Polynomial R)
variables {n : ℕ}
variables {a : Fin n → R}
variables {k : Fin n → ℕ}

-- Lean statement for the proof problem
theorem polynomial_roots_le_degree (P : Polynomial R)
  (hP : P.degree = n)
  (ha : ∀ i, P.is_root (a i).to_multiplicity (k i)) :
  (Finset.univ.sum (λ i : Fin n, k i)) ≤ n :=
sorry

end polynomial_roots_le_degree_l397_397430


namespace sqrt_frac_meaningful_l397_397330

theorem sqrt_frac_meaningful (x : ℝ) (h : 1 / (x - 1) > 0) : x > 1 :=
sorry

end sqrt_frac_meaningful_l397_397330


namespace andy_more_candies_than_caleb_l397_397191

theorem andy_more_candies_than_caleb :
  let billy_initial := 6
  let caleb_initial := 11
  let andy_initial := 9
  let father_packet := 36
  let billy_additional := 8
  let caleb_additional := 11
  let billy_total := billy_initial + billy_additional
  let caleb_total := caleb_initial + caleb_additional
  let total_given := billy_additional + caleb_additional
  let andy_additional := father_packet - total_given
  let andy_total := andy_initial + andy_additional
  andy_total - caleb_total = 4 :=
by {
  sorry
}

end andy_more_candies_than_caleb_l397_397191


namespace problem_statement_l397_397898

theorem problem_statement
  (m n : ℕ)
  (h_m_gt_1 : 1 < m)
  (h_n_gt_1 : 1 < n)
  (h_m_ge_n : m ≥ n)
  (a : Fin n.succ → ℕ)
  (h_a_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_a_le_m : ∀ i, a i ≤ m)
  (h_a_coprime : ∀ i j, i ≠ j → Nat.coprime (a i) (a j)) :
  ∀ x : ℝ, ∃ i : Fin n.succ, ‖a i * x‖ ≥ 2 / (m * (m + 1)) * ‖x‖ :=
begin
  sorry
end

end problem_statement_l397_397898


namespace find_AB_l397_397761

noncomputable def AB_find (A : ℝ) (B : ℝ) (BC : ℝ) : ℝ :=
  BC / Real.tan A

theorem find_AB :
  ∀ (A B : ℝ) (BC : ℝ), A = Real.pi * 40 / 180 ∧ B = Real.pi / 2 ∧ BC = 7 → AB_find A B BC ≈ 9.2
:= by
  intros A B BC h
  sorry

end find_AB_l397_397761


namespace cube_paint_cost_l397_397052

theorem cube_paint_cost (C L : ℝ) (H1 : C = 36.50) (H2 : L^2 = (36.50 * 6 / 16)^(-1) * 876) : L = 8 :=
by
  sorry

end cube_paint_cost_l397_397052


namespace yoongi_hoseok_age_sum_l397_397451

-- Definitions of given conditions
def age_aunt : ℕ := 38
def diff_aunt_yoongi : ℕ := 23
def diff_yoongi_hoseok : ℕ := 4

-- Definitions related to ages of Yoongi and Hoseok derived from given conditions
def age_yoongi : ℕ := age_aunt - diff_aunt_yoongi
def age_hoseok : ℕ := age_yoongi - diff_yoongi_hoseok

-- The theorem we need to prove
theorem yoongi_hoseok_age_sum : age_yoongi + age_hoseok = 26 := by
  sorry

end yoongi_hoseok_age_sum_l397_397451


namespace negation_of_squared_inequality_l397_397326

theorem negation_of_squared_inequality (p : ∀ n : ℕ, n^2 ≤ 2*n + 5) : 
  ∃ n : ℕ, n^2 > 2*n + 5 :=
sorry

end negation_of_squared_inequality_l397_397326


namespace A_card_2023_l397_397037

noncomputable def A : ℕ → Finset ℕ
| 0     := {3}
| (n+1) := { x + 2 | x ∈ A n } ∪ { x * (x + 1) / 2 | x ∈ A n }

theorem A_card_2023 : (A 2023).card = 2 ^ 2023 :=
by
  sorry

end A_card_2023_l397_397037


namespace range_of_a_l397_397287

theorem range_of_a (a : ℝ) (h : 2 * a - 1 ≤ 11) : a < 6 :=
by
  sorry

end range_of_a_l397_397287


namespace time_to_fill_cistern_proof_l397_397914

-- Define the filling rate F and emptying rate E
def filling_rate : ℚ := 1 / 3 -- cisterns per hour
def emptying_rate : ℚ := 1 / 6 -- cisterns per hour

-- Define the net rate as the difference between filling and emptying rates
def net_rate : ℚ := filling_rate - emptying_rate

-- Define the time to fill the cistern given the net rate
def time_to_fill_cistern (net_rate : ℚ) : ℚ := 1 / net_rate

-- The proof statement
theorem time_to_fill_cistern_proof : time_to_fill_cistern net_rate = 6 := 
by sorry

end time_to_fill_cistern_proof_l397_397914


namespace is_factor_l397_397966

-- Define the polynomial
def poly (x : ℝ) := x^4 + 4 * x^2 + 4

-- Define a candidate for being a factor
def factor_candidate (x : ℝ) := x^2 + 2

-- Proof problem: prove that factor_candidate is a factor of poly
theorem is_factor : ∀ x : ℝ, poly x = factor_candidate x * factor_candidate x := 
by
  intro x
  unfold poly factor_candidate
  sorry

end is_factor_l397_397966


namespace inverse_function_sum_l397_397056

noncomputable def f : ℝ → ℝ :=
λ x, if x < 3 then x - 3 else real.sqrt (x + 1)

noncomputable def f_inv : ℝ → ℝ :=
λ x, if x < 0 then x + 3 else (x^2 - 1)

theorem inverse_function_sum :
  f_inv (-6) + f_inv (-5) + f_inv (-4) + f_inv (-3) +
  f_inv (2) + f_inv (3) + f_inv (4) + f_inv (5) + f_inv (6) = 19 :=
by 
  sorry

end inverse_function_sum_l397_397056


namespace centroid_on_MN_l397_397295

theorem centroid_on_MN
  (A B C M N : Point)
  (hM : lies_on M (Segment A B))
  (hN : lies_on N (Segment A C))
  (h_ratio : (dist B M / dist M A) + (dist C N / dist N A) = 1) :
  lies_on (centroid A B C) (Segment M N) :=
by sorry

end centroid_on_MN_l397_397295


namespace S_fg_ge_S_h_squared_l397_397672

noncomputable def S (p : Polynomial ℝ) : ℝ :=
  p.coeffs.sum (λ a, a ^ 2)

theorem S_fg_ge_S_h_squared 
  (f g h : Polynomial ℝ) 
  (H : f * g = h^2) : 
  S f * S g ≥ S h ^ 2 := 
by
  sorry

end S_fg_ge_S_h_squared_l397_397672


namespace sequence_general_term_l397_397256

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

theorem sequence_general_term (h : ∀ n : ℕ, S n = 2 * n - a n) :
  ∀ n : ℕ, a n = (2^n - 1) / (2^(n-1)) :=
by
  sorry

end sequence_general_term_l397_397256


namespace prime_divides_next_term_l397_397134

theorem prime_divides_next_term 
  (a : ℕ → ℕ) (p m : ℕ) 
  (h1 : a 0 = 2) 
  (h2 : a 1 = 1) 
  (h3 : ∀ n ≥ 1, a (n + 1) = a n + a (n - 1))
  (h4 : even m) 
  (h5 : prime p) 
  (h6 : p ∣ (a m - 2)) : 
  p ∣ (a (m + 1) - 1) := 
sorry

end prime_divides_next_term_l397_397134


namespace petya_can_win_l397_397027

-- Define a game state
structure GameState where
  remainingDigits : List ℕ

-- Initial game state
def initialGameState : GameState := {
  remainingDigits := [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7]
}

-- Predicate to represent a winning strategy for Petya
def petyaWins (s : GameState) : Prop :=
  ∃ strategy: (GameState → Fin 10) → GameState, -- Exists a strategy function
  (∀ (move: GameState → Fin 10), 
    -- Applying the strategy results in a state where Petya wins
    strategy move = ⟨0, sorry⟩ -- "0" stands for Petya wins; replace "sorry" with valid proof term

theorem petya_can_win : petyaWins initialGameState := by
  sorry

end petya_can_win_l397_397027


namespace conclusion_2_conclusion_4_l397_397757

variable (JarA : Finset Ball) (JarB : Finset Ball)
variable (red white black : Ball → Prop)

-- Defining probabilities and events
variable [Fintype {x // x ∈ JarA}] [Fintype {x // x ∈ JarB}]
def P (e : Finset {x // x ∈ JarA}) : ℚ :=
  (e.card : ℚ) / (JarA.card : ℚ)

variable (A1 : Finset {x // x ∈ JarA} := (JarA.filter red))
variable (A2 : Finset {x // x ∈ JarA} := (JarA.filter white))
variable (A3 : Finset {x // x ∈ JarA} := (JarA.filter black))
variable (B : Finset {x // x ∈ JarB} := (JarB.filter red))

-- Probability formulas
noncomputable def P_B_given_A1 := (B ∪ A1).card.toRat / (JarB.card + 1).toRat

-- Statements to be proven
theorem conclusion_2: P_B_given_A1 = 5 / 11 := sorry

theorem conclusion_4: Disjoint A1 A2 ∧ Disjoint A2 A3 ∧ Disjoint A1 A3 := sorry

end conclusion_2_conclusion_4_l397_397757


namespace find_x_l397_397743

theorem find_x (A B D : ℝ) (BC CD x : ℝ) 
  (hA : A = 60) (hB : B = 90) (hD : D = 90) 
  (hBC : BC = 2) (hCD : CD = 3) 
  (hResult : x = 8 / Real.sqrt 3) : 
  AB = x :=
by
  sorry

end find_x_l397_397743


namespace find_k_l397_397243

def vector (α : Type*) := α × α

def a : vector ℝ := (1, 3)
def b (k : ℝ) : vector ℝ := (-2, k)

def vector_parallel (u v : vector ℝ) : Prop :=
  ∃ (λ:ℝ), λ • u = v

theorem find_k (k : ℝ) : 
  vector_parallel ((a.1 + 2 * b(k).1, a.2 + 2 * b(k).2))
                  ((3 * a.1 - b(k).1, 3 * a.2 - b(k).2))
  → k = -6 := 
by 
  sorry

end find_k_l397_397243


namespace third_candidate_more_votes_than_john_l397_397546

-- Define the given conditions
def total_votes : ℕ := 1150
def john_votes : ℕ := 150
def remaining_votes : ℕ := total_votes - john_votes
def james_votes : ℕ := (7 * remaining_votes) / 10
def john_and_james_votes : ℕ := john_votes + james_votes
def third_candidate_votes : ℕ := total_votes - john_and_james_votes

-- Stating the problem to prove
theorem third_candidate_more_votes_than_john : third_candidate_votes - john_votes = 150 := 
by
  sorry

end third_candidate_more_votes_than_john_l397_397546


namespace time_to_pass_l397_397095

def length_of_train : ℝ := 37.5
def speed_faster_train : ℝ := 46 * (1000 / 3600) -- converting km/hr to m/s
def speed_slower_train : ℝ := 36 * (1000 / 3600) -- converting km/hr to m/s
def relative_speed := speed_faster_train - speed_slower_train
def total_distance : ℝ := 2 * length_of_train

theorem time_to_pass :
  total_distance / relative_speed ≈ 27.027 := by
  sorry

end time_to_pass_l397_397095


namespace line_intersects_x_axis_at_point_l397_397920

theorem line_intersects_x_axis_at_point : 
  let x1 := 3
  let y1 := 7
  let x2 := -1
  let y2 := 3
  let m := (y2 - y1) / (x2 - x1) -- slope formula
  let b := y1 - m * x1        -- y-intercept formula
  let x_intersect := -b / m  -- x-coordinate where the line intersects x-axis
  (x_intersect, 0) = (-4, 0) :=
by
  sorry

end line_intersects_x_axis_at_point_l397_397920


namespace simplify_expression_l397_397438

theorem simplify_expression : (-real.sqrt 3) ^ 2 ^ (-1 / 2 : ℚ) = real.sqrt 3 / 3 :=
by sorry

end simplify_expression_l397_397438


namespace find_complex_z_l397_397270

noncomputable def z := Complex

theorem find_complex_z (z : ℂ) (h : 3 * z + complex.conj(z) = 4 / (1 - I)) : z = 1/2 + I :=
sorry

end find_complex_z_l397_397270


namespace entrepreneur_investment_interest_rate_l397_397598

theorem entrepreneur_investment_interest_rate :
  let initial_investment := 20000
  let first_term_duration := 9 / 12 -- 9 months in years
  let first_annual_rate := 8 / 100 -- 8%
  let first_term_interest_rate := initial_investment * (1 + first_term_duration * first_annual_rate)

  let second_term_duration := 9 / 12 -- 9 months in years
  let final_amount := 22446.40

  ∃ s,
  let second_term_interest_rate := s / 100
  let second_term_proceeds := first_term_interest_rate * (1 + second_term_duration * second_term_interest_rate)
  second_term_proceeds ≈ final_amount :=
by
  sorry

end entrepreneur_investment_interest_rate_l397_397598


namespace number_of_multiples_of_77_l397_397300

theorem number_of_multiples_of_77 : 
  {n : ℕ | ∃ (k : ℕ) (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 99 ∧ 77 * k = 10^j - 10^i }.to_finset.card = 784 :=
by 
  sorry

end number_of_multiples_of_77_l397_397300


namespace decimal_to_fraction_l397_397099

theorem decimal_to_fraction (x : ℚ) (h : x = 3.68) : x = 92 / 25 := by
  sorry

end decimal_to_fraction_l397_397099


namespace lowest_possible_students_l397_397528

-- Definitions based on conditions
def isDivisibleBy (n m : ℕ) : Prop := n % m = 0

def canBeDividedIntoTeams (num_students num_teams : ℕ) : Prop := isDivisibleBy num_students num_teams

-- Theorem statement for the lowest possible number of students
theorem lowest_possible_students (n : ℕ) : 
  (canBeDividedIntoTeams n 8) ∧ (canBeDividedIntoTeams n 12) → n = 24 := by
  sorry

end lowest_possible_students_l397_397528


namespace probability_three_green_is_14_over_99_l397_397908

noncomputable def probability_three_green :=
  let total_combinations := Nat.choose 12 4
  let successful_outcomes := (Nat.choose 5 3) * (Nat.choose 7 1)
  (successful_outcomes : ℚ) / total_combinations

theorem probability_three_green_is_14_over_99 :
  probability_three_green = 14 / 99 :=
by
  sorry

end probability_three_green_is_14_over_99_l397_397908


namespace sum_of_series_l397_397634

theorem sum_of_series : 
  (1 / (1 * 2) + 1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7)) = 6 / 7 := 
 by sorry

end sum_of_series_l397_397634


namespace min_value_sin6_cos6_l397_397963

theorem min_value_sin6_cos6 :
  ∀ x : ℝ, 1 - 3/4 * (sin (2 * x))^2 ≥ 1/4 :=
by
  intros x
  have H : sin^2 x + cos^2 x = 1 := by sorry
  sorry

end min_value_sin6_cos6_l397_397963


namespace find_AB_l397_397760

noncomputable def AB_find (A : ℝ) (B : ℝ) (BC : ℝ) : ℝ :=
  BC / Real.tan A

theorem find_AB :
  ∀ (A B : ℝ) (BC : ℝ), A = Real.pi * 40 / 180 ∧ B = Real.pi / 2 ∧ BC = 7 → AB_find A B BC ≈ 9.2
:= by
  intros A B BC h
  sorry

end find_AB_l397_397760


namespace decagon_triangle_probability_l397_397542

theorem decagon_triangle_probability : 
  let total_vertices := 10
  let total_triangles := Nat.choose total_vertices 3
  let favorable_triangles := 10
  (total_triangles > 0) → 
  (favorable_triangles / total_triangles : ℚ) = 1 / 12 :=
by
  sorry

end decagon_triangle_probability_l397_397542


namespace find_f_value_l397_397796

theorem find_f_value (f: ℕ+ → ℕ+)
  (h1: ∀ (a b: ℕ+), a ≠ b → a * f a + b * f b > a * f b + b * f a)
  (h2: ∀ (n: ℕ+), f (f n) = 3 * n):
  f 1 + f 6 + f 28 = 66 :=
by
  sorry

end find_f_value_l397_397796


namespace sum_f_equals_1003_l397_397007

noncomputable def f (x : ℝ) : ℝ := 2008^(2 * x) / (2008 + 2008^(2 * x))

theorem sum_f_equals_1003 : (∑ k in Finset.range 2006 + 1, f (k / 2007)) = 1003 := 
by 
  sorry

end sum_f_equals_1003_l397_397007


namespace Andy_has_4_more_candies_than_Caleb_l397_397189

-- Define the initial candies each person has
def Billy_initial_candies : ℕ := 6
def Caleb_initial_candies : ℕ := 11
def Andy_initial_candies : ℕ := 9

-- Define the candies bought by the father and their distribution
def father_bought_candies : ℕ := 36
def Billy_received_from_father : ℕ := 8
def Caleb_received_from_father : ℕ := 11

-- Calculate the remaining candies for Andy after distribution
def Andy_received_from_father : ℕ := father_bought_candies - (Billy_received_from_father + Caleb_received_from_father)

-- Calculate the total candies each person has
def Billy_total_candies : ℕ := Billy_initial_candies + Billy_received_from_father
def Caleb_total_candies : ℕ := Caleb_initial_candies + Caleb_received_from_father
def Andy_total_candies : ℕ := Andy_initial_candies + Andy_received_from_father

-- Prove that Andy has 4 more candies than Caleb
theorem Andy_has_4_more_candies_than_Caleb :
  Andy_total_candies = Caleb_total_candies + 4 :=
by {
  -- Skipping the proof
  sorry
}

end Andy_has_4_more_candies_than_Caleb_l397_397189


namespace lola_pop_tarts_baked_l397_397015

theorem lola_pop_tarts_baked :
  ∃ P : ℕ, (13 + P + 8) + (16 + 12 + 14) = 73 ∧ P = 10 := by
  sorry

end lola_pop_tarts_baked_l397_397015


namespace avg_difference_in_circumferences_l397_397865

-- Define the conditions
def inner_circle_diameter : ℝ := 30
def min_track_width : ℝ := 10
def max_track_width : ℝ := 15

-- Define the average difference in the circumferences of the two circles
theorem avg_difference_in_circumferences :
  let avg_width := (min_track_width + max_track_width) / 2
  let outer_circle_diameter := inner_circle_diameter + 2 * avg_width
  let inner_circle_circumference := Real.pi * inner_circle_diameter
  let outer_circle_circumference := Real.pi * outer_circle_diameter
  outer_circle_circumference - inner_circle_circumference = 25 * Real.pi :=
by
  sorry

end avg_difference_in_circumferences_l397_397865


namespace bills_difference_is_zero_l397_397929

-- Define the variables and conditions
variables {a b c : ℝ}

-- Conditions based on the problem statement
def condition1 : Prop := 0.20 * a = 4
def condition2 : Prop := 0.15 * b = 3
def condition3 : Prop := 0.25 * c = 5

-- Define the assertion to prove
theorem bills_difference_is_zero (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  max a (max b c) - min a (min b c) = 0 := 
sorry

end bills_difference_is_zero_l397_397929


namespace average_weight_of_class_l397_397084

section AverageWeight

variable (A_students : ℕ) (B_students : ℕ) (C_students : ℕ) (D_students : ℕ)
variable (A_avg_weight : ℝ) (B_avg_weight : ℝ) (C_avg_weight : ℝ) (D_avg_weight : ℝ)

def total_students := A_students + B_students + C_students + D_students

def total_weight := 
  (A_students * A_avg_weight) + 
  (B_students * B_avg_weight) + 
  (C_students * C_avg_weight) + 
  (D_students * D_avg_weight)

def class_avg_weight := total_weight / total_students

theorem average_weight_of_class :
  A_students = 26 → B_students = 34 → C_students = 40 → D_students = 50 → 
  A_avg_weight = 50 → B_avg_weight = 45 → C_avg_weight = 35 → D_avg_weight = 30 → 
  class_avg_weight A_students B_students C_students D_students A_avg_weight B_avg_weight C_avg_weight D_avg_weight = 38.2 :=
by
  intros hA hB hC hD hA_w hB_w hC_w hD_w
  sorry

end AverageWeight

end average_weight_of_class_l397_397084


namespace lucky_sum_probability_eq_l397_397378

/--
Given that there are N balls numbered from 1 to N,
where 10 balls are selected in the main draw with their sum being 63,
and 8 balls are selected in the additional draw with their sum being 44,
we need to prove that N = 18 such that the events are equally likely.
-/
theorem lucky_sum_probability_eq (N : ℕ) (h1 : ∃ (S : Finset ℕ), S.card = 10 ∧ S.sum id = 63) 
    (h2 : ∃ (T : Finset ℕ), T.card = 8 ∧ T.sum id = 44) : N = 18 :=
sorry

end lucky_sum_probability_eq_l397_397378


namespace triangle_square_side_ratio_l397_397599

theorem triangle_square_side_ratio 
  (perimeter_triangle perimeter_square : ℕ)
  (h_triangle : perimeter_triangle = 48)
  (h_square : perimeter_square = 48) : 
  (16 / 12 : ℚ) = (4 / 3 : ℚ) :=
by 
  have triangle_side_length : ℚ := perimeter_triangle / 3 := by sorry
  have square_side_length : ℚ := perimeter_square / 4 := by sorry
  have ratio := triangle_side_length / square_side_length := by sorry
  exact ratio = (4 / 3 : ℚ)

end triangle_square_side_ratio_l397_397599


namespace intersecting_points_l397_397810

def number_of_intersection_points (n₁ n₂ n₃ n₄ : ℕ) (h₀ : n₁ = 4) (h₁ : n₂ = 5) (h₂ : n₃ = 7) (h₃ : n₄ = 9) : ℕ :=
  2 * n₁ + 2 * n₁ + 2 * n₁ + 2 * n₂ + 2 * n₂ + 2 * n₃

theorem intersecting_points :
  number_of_intersection_points 4 5 7 9 4.refl 5.refl 7.refl 9.refl = 58 :=
by
  sorry

end intersecting_points_l397_397810


namespace quadrilateral_parallelogram_l397_397118

variables (A B C D : Type*) [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]

def is_parallel (x y : Type*) [AddCommGroup x] [AddCommGroup y] := sorry

def is_equal_length (x y : Type*) [AddCommGroup x] [AddCommGroup y] := sorry

def is_parallelogram (A B C D : Type*) [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] := sorry

theorem quadrilateral_parallelogram (AB CD : Type*) [AddCommGroup AB] [AddCommGroup CD]
  (h1 : is_parallel AB CD) (h2 : is_equal_length AB CD) : is_parallelogram A B C D :=
sorry

end quadrilateral_parallelogram_l397_397118


namespace perimeter_of_smaller_rectangle_l397_397846

theorem perimeter_of_smaller_rectangle :
  ∀ (L W n : ℕ), 
  L = 16 → W = 20 → n = 10 →
  (∃ (x y : ℕ), L % 2 = 0 ∧ W % 5 = 0 ∧ 2 * y = L ∧ 5 * x = W ∧ (L * W) / n = x * y ∧ 2 * (x + y) = 24) :=
by
  intros L W n H1 H2 H3
  use 4, 8
  sorry

end perimeter_of_smaller_rectangle_l397_397846


namespace locus_of_points_K_l397_397137

theorem locus_of_points_K {A B C K : (ℝ × ℝ)} {r R : ℝ}
  (hA : A = (0, 0))
  (hB : ∃ r : ℝ, B = (r, 0))
  (hC : ∃ a b : ℝ, a^2 + b^2 = R^2 ∧ C = (a, b))
  (hK : ∃ n : ℝ, ∃ M1 M2 : (ℝ × ℝ),
    M1 = (r, n) ∧ 
    M2 = (λ a b, (a + (b * n / R), b - (a * n / R))) (C.fst, C.snd) ∧
    dist M1 M2 = 2 * n ∧
    K = ((M1.fst + M2.fst) / 2, (M1.snd + M2.snd) / 2)) :
  2 * K.fst^2 + 2 * K.snd^2 = r^2 + R^2 :=
sorry

end locus_of_points_K_l397_397137


namespace complement_union_eq_l397_397293

universe u
variable (U A B : Set ℕ)

def U := {1, 2, 3, 4, 5, 6, 7}
def A := {2, 4, 5, 7}
def B := {3, 4, 5}

theorem complement_union_eq :
  ((U \ A) ∪ (U \ B)) = ({1, 2, 3, 6, 7} : Set ℕ) := by sorry

end complement_union_eq_l397_397293


namespace hotel_room_allocation_l397_397976

theorem hotel_room_allocation :
  ∀ (fans : ℕ) (num_groups : ℕ) (room_capacity : ℕ), 
  fans = 100 → num_groups = 6 → room_capacity = 3 →
  (∀ (group_fans : ℕ) (groups : fin num_groups), group_fans ≤ fans →
   (∃ (total_rooms: ℕ), total_rooms = 37 ∧
    ∀ (group_fan_count : fin num_groups → ℕ), 
      (∀ g, group_fan_count g ≤ fans / num_groups + if (fans % num_groups > 0) then 1 else 0) →
      ∃ (rooms : fin num_groups → ℕ), 
        sum (λ g, rooms g) = total_rooms ∧ 
        ∀ g, group_fan_count g ≤ rooms g * room_capacity)) :=
by {
  intros,
  sorry
}

end hotel_room_allocation_l397_397976


namespace drying_time_ratio_l397_397773

variables (x : ℝ) (t_short : ℝ) (t_full : ℝ)

-- Given statements from the conditions
def short_dog_dry_time := 10 -- minutes to dry a short-haired dog
def num_short_haired_dogs := 6
def num_full_haired_dogs := 9
def total_dry_time := 240 -- minutes for drying all dogs

-- Define total drying times
def total_short_dry_time := num_short_haired_dogs * short_dog_dry_time
def total_full_dry_time := num_full_haired_dogs * t_full

-- Equation representing the total time spent drying all dogs
def total_time_eq : ℝ := total_short_dry_time + total_full_dry_time

-- The statement to prove
theorem drying_time_ratio (h1 : t_short = short_dog_dry_time)
                          (h2 : t_full = 10 * x)
                          (h3 : total_time_eq = total_dry_time) :
  x = 2 := sorry

end drying_time_ratio_l397_397773


namespace primes_with_prime_remainders_count_l397_397307

-- Define a predicate to check for prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime remainders when divided by 12
def prime_remainders := {1, 2, 3, 5, 7, 11}

-- Function to list primes between 50 and 100
def primes_between_50_and_100 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Function to count such primes with prime remainder when divided by 12
noncomputable def count_primes_with_prime_remainder : ℕ :=
  list.count (λ n, n % 12 ∈ prime_remainders) primes_between_50_and_100

-- The theorem to state the problem in Lean
theorem primes_with_prime_remainders_count : count_primes_with_prime_remainder = 10 :=
by {
  /- proof steps to be provided here, if required. -/
 sorry
}

end primes_with_prime_remainders_count_l397_397307


namespace determine_g_2023_l397_397410

noncomputable def g (x : ℕ) : ℝ := sorry

axiom g_pos (x : ℕ) (hx : x > 0) : g x > 0

axiom g_property (x y : ℕ) (h1 : x > 2 * y) (h2 : 0 < y) : 
  g (x - y) = Real.sqrt (g (x / y) + 3)

theorem determine_g_2023 : g 2023 = (1 + Real.sqrt 13) / 2 :=
by
  sorry

end determine_g_2023_l397_397410


namespace average_price_approx_l397_397532

variable (large_bottles : ℕ := 1365)
variable (small_bottles : ℕ := 720)
variable (price_large : ℝ := 1.89)
variable (price_small : ℝ := 1.42)

theorem average_price_approx :
  (large_bottles * price_large + small_bottles * price_small) / (large_bottles + small_bottles) ≈ 1.73 :=
by sorry

end average_price_approx_l397_397532


namespace events_equally_likely_iff_N_eq_18_l397_397366

variable (N : ℕ)

-- Define the number of combinations in the draws
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the sums of selecting balls
noncomputable def S_63_10 (N: ℕ) : ℕ := sorry -- placeholder definition
noncomputable def S_44_8 (N: ℕ) : ℕ := sorry 

-- Condition for the events being equally likely
theorem events_equally_likely_iff_N_eq_18 : 
  (S_63_10 N * C N 8 = S_44_8 N * C N 10) ↔ N = 18 :=
sorry

end events_equally_likely_iff_N_eq_18_l397_397366


namespace point_inside_circle_l397_397290

-- Define the conditions
def P : ℝ × ℝ := (3, 2)
def circle_center : ℝ × ℝ := (2, 3)
def radius : ℝ := 2
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- The statement to prove that the point is inside the circle
theorem point_inside_circle (P : ℝ × ℝ) (circle_center : ℝ × ℝ) (radius : ℝ)
  (hP : P = (3, 2)) (hC : circle_center = (2, 3)) (hR : radius = 2) :
  distance P circle_center < radius :=
by
  rw [hP, hC, hR]
  sorry

end point_inside_circle_l397_397290


namespace hotel_room_allocation_l397_397977

theorem hotel_room_allocation :
  ∀ (fans : ℕ) (num_groups : ℕ) (room_capacity : ℕ), 
  fans = 100 → num_groups = 6 → room_capacity = 3 →
  (∀ (group_fans : ℕ) (groups : fin num_groups), group_fans ≤ fans →
   (∃ (total_rooms: ℕ), total_rooms = 37 ∧
    ∀ (group_fan_count : fin num_groups → ℕ), 
      (∀ g, group_fan_count g ≤ fans / num_groups + if (fans % num_groups > 0) then 1 else 0) →
      ∃ (rooms : fin num_groups → ℕ), 
        sum (λ g, rooms g) = total_rooms ∧ 
        ∀ g, group_fan_count g ≤ rooms g * room_capacity)) :=
by {
  intros,
  sorry
}

end hotel_room_allocation_l397_397977


namespace concyclic_quad_right_angle_l397_397427

-- Define points and geometric properties
variables (A B C D E P O : Type)
-- Assume given conditions
variables (h1 : A ∈ circumference O)
variables (h2 : B ∈ circumference O)
variables (h3 : C ∈ circumference O)
variables (h4 : D ∈ circumference O)
variables (h5 : ∃ (E : Type), intersects (line A B) (line C D) E)
variables (h6 : ∃ (P : Type), intersection (circumcircle (triangle A E C)) (circumcircle (triangle B E D)) = {E, P})

-- Statement for part (a)
theorem concyclic_quad (h1 : A ∈ circumference O) (h2 : B ∈ circumference O) (h3 : C ∈ circumference O) (h4 : D ∈ circumference O) (h5 : ∃ (E : Type), intersects (line A B) (line C D) E) (h6 : ∃ (P : Type), intersection (circumcircle (triangle A E C)) (circumcircle (triangle B E D)) = {E, P}) :
  concyclic A D P O :=
sorry

-- Statement for part (b)
theorem right_angle (h1 : A ∈ circumference O) (h2 : B ∈ circumference O) (h3 : C ∈ circumference O) (h4 : D ∈ circumference O) (h5 : ∃ (E : Type), intersects (line A B) (line C D) E) (h6 : ∃ (P : Type), intersection (circumcircle (triangle A E C)) (circumcircle (triangle B E D)) = {E, P}) :
  angle E P O = 90 :=
sorry

end concyclic_quad_right_angle_l397_397427


namespace savings_correct_l397_397568

noncomputable def window_cost (n : Nat) : Int :=
  if n <= 5 then n * 120
  else 5 * 120 + (n - 5) * 100

def dave_windows : Nat := 10
def doug_windows : Nat := 13

def total_separate_cost : Int :=
  window_cost dave_windows + window_cost doug_windows

def total_joint_cost : Int :=
  window_cost (dave_windows + doug_windows)

def savings : Int :=
  total_separate_cost - total_joint_cost

theorem savings_correct : savings = 100 := by
  -- proof goes here
  sorry

end savings_correct_l397_397568


namespace constant_term_in_expansion_l397_397467

noncomputable def binomial_term (n k : ℕ) (a b : ℤ) (m : ℚ) : ℚ :=
  (Nat.choose n k) * (b ^ k) * (a ^ (n - k)) * (m : ℚ)

theorem constant_term_in_expansion : 
  let x : ℚ := (2, 3, 2); 
  let a : ℤ := 2;
  let b : ℤ := -1;
  let n : ℕ := 6;
  let k : ℕ := 4;
  let m : ℚ := (1 : ℚ) / 81 * 4;
  binomial_term n k a b 1 = m :=
by
  -- proof would go here
  sorry

end constant_term_in_expansion_l397_397467


namespace sum_seq1_correct_alternating_sum_squares_correct_l397_397900

-- Define the sequence for Part 1
def seq1 (n : ℕ) : ℝ := (2^n + 1) / (4^n)

-- Define the sum of the first n terms for Part 1
def sum_seq1 (n : ℕ) : ℝ := (List.range n).sum (λ k, seq1 (k + 1))

-- Define the expected sum for Part 1
def expected_sum_seq1 (n : ℕ) : ℝ := (4 / 3) - (1 / (2^n)) - (1 / (3 * (4^n)))

-- Theorem for Part 1
theorem sum_seq1_correct (n : ℕ) : sum_seq1 n = expected_sum_seq1 n := by sorry

-- Define the sequence for Part 2
def seq2 (n : ℕ) : ℝ := 3 * n - 1

-- Define the alternating sum of squares for Part 2
def alternating_sum_squares (n : ℕ) : ℝ :=
  let indices := List.range n
  indices.sum (λ k, seq2 (2*k + 1)^2 - seq2 (2*k + 2)^2)

-- Theorem for Part 2
theorem alternating_sum_squares_correct : alternating_sum_squares 10 = -1830 := by sorry

end sum_seq1_correct_alternating_sum_squares_correct_l397_397900


namespace athul_downstream_distance_l397_397944

-- Define the conditions
def upstream_distance : ℝ := 16
def upstream_time : ℝ := 4
def speed_of_stream : ℝ := 1
def downstream_time : ℝ := 4

-- Translate the conditions into properties and prove the downstream distance
theorem athul_downstream_distance (V : ℝ) 
  (h1 : upstream_distance = (V - speed_of_stream) * upstream_time) :
  (V + speed_of_stream) * downstream_time = 24 := 
by
  -- Given the conditions, the proof would be filled here
  sorry

end athul_downstream_distance_l397_397944


namespace coloring_circle_l397_397628

open Nat

theorem coloring_circle (n : ℕ) (h : n ≥ 2) : 
  ∃ (a : ℕ → ℕ), a n = 2 * (-1) ^ n + 2 ^ n := 
sorry

end coloring_circle_l397_397628


namespace unique_function_satisfying_condition_l397_397782

theorem unique_function_satisfying_condition :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) ↔ f = id :=
sorry

end unique_function_satisfying_condition_l397_397782


namespace solution_of_fraction_inequality_l397_397786

open Function Set

variable {R : Type} [Real]

-- Defining an odd function
def odd_function (f : R → R) : Prop :=
  ∀ x, f (-x) = -f x

-- Defining an even function
def even_function (g : R → R) : Prop :=
  ∀ x, g (-x) = g x

-- Main theorem statement
theorem solution_of_fraction_inequality (f g : R → R) (h₀ : odd_function f) (h₁ : even_function g)
  (hg : ∀ x, g x ≠ 0) (h₂ : f (-3) = 0) (h₃ : ∀ x, x < 0 → f' x * g x < f x * g' x) :
  { x | (f x / g x) < 0 } = Ioo (-∞) (-3) ∪ Ioi 3 :=
by
  sorry

end solution_of_fraction_inequality_l397_397786


namespace part1_hexagon_properties_part2_polygon_properties_l397_397135

theorem part1_hexagon_properties :
  (∀ (n : ℕ), n = 6 → 
     (∃ (d : ℕ) (t : ℕ) (s : ℕ),
      d = 3 ∧ t = 4 ∧ s = 720)) :=
by
  intros n hn
  exists 3, 4, 720
  exact ⟨rfl, rfl, by linarith [hn, n]⟩

theorem part2_polygon_properties :
  (∀ (e_sum : ℝ), e_sum = 360 →
     ∃ (n : ℕ) (i_sum : ℝ),
     180 * (n - 2) = 720 - 180 ∧ 2 * e_sum = 720 - 180 ∧ n = 5 ∧ i_sum = 540) :=
by
  intros e_sum he_sum
  exists 5, 540
  split
  { linarith }
  split
  { linarith }
  exact ⟨rfl, rfl⟩

end part1_hexagon_properties_part2_polygon_properties_l397_397135


namespace complete_residue_system_l397_397223

theorem complete_residue_system (m : ℕ) : 
    (∀ n : ℕ, n > 0 → ∃ t : ℕ, (∀ a b : ℕ, a ≠ b → (a^n ≡ b^n [MOD m]) = a ≡ b [MOD m])) ↔ 
        (∀ p : ℕ, prime p → p^2 ∣ m → false) :=
begin
    sorry
end

end complete_residue_system_l397_397223


namespace problem1_problem2_l397_397406

-- Definitions based on conditions in the problem
def f (x a : ℝ) : ℝ := |x + a|
def g (x : ℝ) : ℝ := |x + 3| - x
def M (a : ℝ) : set ℝ := {x | f x a < g x}

-- Problem (1): Prove that if a - 3 ∈ M(a), then 0 < a < 3
theorem problem1 (a : ℝ) (h: a - 3 ∈ M a) : 0 < a ∧ a < 3 :=
sorry

-- Problem (2): Prove that if [-1, 1] ⊆ M(a), then -2 < a < 2
theorem problem2 (a : ℝ) (h: ∀ x, x ∈ Icc (-1 : ℝ) 1 → x ∈ M a) : -2 < a ∧ a < 2 :=
sorry

end problem1_problem2_l397_397406


namespace smallest_positive_multiple_of_6_factorial_l397_397879

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem smallest_positive_multiple_of_6_factorial : 
  ∃ (x : Nat), x > 0 ∧ x * factorial 6 = 720 := by
  have fact6 : factorial 6 = 720 := by
    unfold factorial
    rfl
  use 1
  split
  . exact Nat.one_pos
  . rw [Nat.one_mul]
    exact fact6

end smallest_positive_multiple_of_6_factorial_l397_397879


namespace value_at_neg2_l397_397277

noncomputable def f : ℝ → ℝ
| x := if h : 1 ≤ x ∧ x ≤ 5 then x^3 + 1 else sorry -- definition of f on [1, 5]

theorem value_at_neg2 : f(-2) = -9 :=
by
  -- Given that f is an odd function
  have odd_f : ∀ x : ℝ, f(-x) = - (f(x)) := sorry
  -- Given that f(x) = x^3 + 1 for x ∈ [1, 5]
  have def_f_on_1_5 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → f(x) = x^3 + 1 := sorry
  -- Calculate f(2)
  have f_2 : f(2) = 9, from calc
    f(2) = (2)^3 + 1 : by exact def_f_on_1_5 2 (by norm_num)
    ... = 9       : by norm_num
  -- Using the odd function property
  show f(-2) = -9, from calc
    f(-2) = - (f(2)) : by exact odd_f 2
    ... = -9        : by exact congr_arg Neg.neg f_2

end value_at_neg2_l397_397277


namespace sum_of_factorials_is_square_l397_397797

open Nat

theorem sum_of_factorials_is_square (m : ℕ) (S : ℕ) :
  S = ∑ i in range (m + 1), i.factorial →
  is_square S →
  m = 1 ∨ m = 3 :=
  sorry

end sum_of_factorials_is_square_l397_397797


namespace binomial_12_11_eq_12_l397_397613

-- Define binomial coefficient function
def binomial (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement to prove
theorem binomial_12_11_eq_12 :
  binomial 12 11 = 12 := by
  sorry

end binomial_12_11_eq_12_l397_397613


namespace fraction_power_multiplication_power_multiplication_example_l397_397512

theorem fraction_power_multiplication 
  (a b c d k : ℕ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b : ℚ) ^ k * (c / d : ℚ) ^ k = (a * c / (b * d) : ℚ) ^ k := 
by sorry

theorem power_multiplication_example :
  (8 / 9 : ℚ) ^ 3 * (3 / 5 : ℚ) ^ 3 = 512 / 3375 :=
by
  have h := fraction_power_multiplication 8 9 3 5 3 (by norm_num) (by norm_num)
  rw [h]
  norm_num
  sorry

end fraction_power_multiplication_power_multiplication_example_l397_397512


namespace greatest_possible_value_of_x_l397_397107

theorem greatest_possible_value_of_x : 
  (∀ x : ℚ, ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20) → 
  x ≤ 9/5 := sorry

end greatest_possible_value_of_x_l397_397107


namespace find_q_minus_p_l397_397291

noncomputable def a_n (n : ℕ) (h : 0 < n) : ℚ := 
  5 * (2/5)^(2 * n - 2) - 4 * (2/5)^(n - 1)

theorem find_q_minus_p (p q : ℕ) (hp : 0 < p) (hq : 0 < q) 
  (hp_max : ∀ n > 0, a_n n (nat.pos_of_gt (zero_lt_of_lt hp)) ≤ a_n p (nat.pos_of_gt hp))
  (hq_min : ∀ n > 0, a_n q (nat.pos_of_gt (zero_lt_of_lt hq)) ≤ a_n n (nat.pos_of_gt hq)) :
  q - p = 1 :=
sorry

end find_q_minus_p_l397_397291


namespace teresa_speed_correct_l397_397449

variable (distance time : ℝ)

def teresa_speed (distance time : ℝ) : ℝ :=
  distance / time

theorem teresa_speed_correct : teresa_speed 25 5 = 5 := 
  by
  sorry

end teresa_speed_correct_l397_397449


namespace adam_initial_money_l397_397171

theorem adam_initial_money :
  let cost_of_airplane := 4.28
  let change_received := 0.72
  cost_of_airplane + change_received = 5.00 :=
by
  sorry

end adam_initial_money_l397_397171


namespace inequality_proof_l397_397534

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a + b + c + d + 8 / (a*b + b*c + c*d + d*a) ≥ 6 := 
by
  sorry

end inequality_proof_l397_397534


namespace ratio_AD_DC_l397_397333
-- Lean 4 statement

variable (A B C D : Type)
variable [HilbertGeometry A]
variable (A B C : Point A)
variable (D : Point A)
variable (AB BC AC BD AD DC : ℝ)

-- Conditions
variable (h1 : dist A B = 6)
variable (h2 : dist B C = 8)
variable (h3 : dist A C = 10)
variable (h4 : dist B D = 6)
variable (h5 : onLine D (lineThrough A C))

-- The proof goal
theorem ratio_AD_DC (h1 : dist A B = 6) (h2 : dist B C = 8)
  (h3 : dist A C = 10) (h4 : dist B D = 6) (h5 : onLine D (lineThrough A C)):
  dist A D / dist D C = 18 / 7 := sorry

end ratio_AD_DC_l397_397333


namespace right_angled_triangle_not_congruent_by_equal_acute_angles_l397_397119

variable {ΔABC ΔDEF : Type} [Triangle ΔABC] [Triangle ΔDEF]
variable (A B C D E F : Point)
variable (a1 a2 a3 a4 a5 a6 : ℝ)
variable (h1 : RightAngled ΔABC)
variable (h2 : RightAngled ΔDEF)
variable (c1 : ∠ A B C = 90)
variable (c2 : ∠ D E F = 90)
variable (c3 : ∠ A B C = ∠ D E F)
variable (c4 : ∠ B A C = ∠ E D F)
variable (c5 : length (AB) = a1)
variable (c6 : length (BC) = a2)
variable (c7 : length (CA) = a3)
variable (c8 : length (DE) = a4)
variable (c9 : length (EF) = a5)
variable (c10 : length (FD) = a6)

theorem right_angled_triangle_not_congruent_by_equal_acute_angles :
  ¬(ΔABC ≅ ΔDEF) :=
by
  sorry

end right_angled_triangle_not_congruent_by_equal_acute_angles_l397_397119


namespace probability_red_ball_distribution_of_X_expected_value_of_X_l397_397755

theorem probability_red_ball :
  let pB₁ := 2 / 3
  let pB₂ := 1 / 3
  let pA_B₁ := 1 / 2
  let pA_B₂ := 3 / 4
  (pB₁ * pA_B₁ + pB₂ * pA_B₂) = 7 / 12 := by
  sorry

theorem distribution_of_X :
  let p_minus2 := 1 / 12
  let p_0 := 1 / 12
  let p_1 := 11 / 24
  let p_3 := 7 / 48
  let p_4 := 5 / 24
  let p_6 := 1 / 48
  (p_minus2 = 1 / 12) ∧ (p_0 = 1 / 12) ∧ (p_1 = 11 / 24) ∧ (p_3 = 7 / 48) ∧ (p_4 = 5 / 24) ∧ (p_6 = 1 / 48) := by
  sorry

theorem expected_value_of_X :
  let E_X := (-2 * (1 / 12) + 0 * (1 / 12) + 1 * (11 / 24) + 3 * (7 / 48) + 4 * (5 / 24) + 6 * (1 / 48))
  E_X = 27 / 16 := by
  sorry

end probability_red_ball_distribution_of_X_expected_value_of_X_l397_397755


namespace modular_inverse_17_mod_500_l397_397112

theorem modular_inverse_17_mod_500 :
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 499 ∧ (17 * x ≡ 1 [MOD 500]) :=
begin
  use 295,
  split,
  { norm_num }, -- 0 ≤ 295
  split,
  { norm_num }, -- 295 ≤ 499
  { norm_num, exact modeq.symm (modeq_of_dvd_by_mod (by norm_num1)) }
end

end modular_inverse_17_mod_500_l397_397112


namespace sales_volume_function_max_profit_min_boxes_for_2000_profit_l397_397558

-- Definitions and conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def init_boxes_sold : ℝ := 250
def price_increase_effect : ℝ := 10
def max_selling_price : ℝ := 38

-- Question 1: Find functional relationship between daily sales volume y and selling price per box x
theorem sales_volume_function (x : ℝ) (hx : x ≥ min_selling_price) :
  ∃ y, y = -10 * x + 500 := by
  sorry

-- Question 2: Find the price per box to maximize daily sales profit and the maximum profit
theorem max_profit (x : ℝ) (hx : x = 35) :
  ∃ P, P = -10 * (x-20) * (x) + 500 * (x-20) := by
  sorry

-- Question 3: Determine min boxes sold to make at least 2000 yuan given price does not exceed 38 yuan
theorem min_boxes_for_2000_profit (x : ℝ) (hx : min_selling_price ≤ x ∧ x ≤ max_selling_price ∧ 
                             -10 * (x-20) * (-10 * x + 500) ≥ 2000) :
  ∃ y, y = -10 * x + 500 ∧ y ≥ 120 := by
  sorry

end sales_volume_function_max_profit_min_boxes_for_2000_profit_l397_397558


namespace factor_expression_l397_397469

theorem factor_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b)^2 * (b - c)^2 * (c - a)^2 * (a + b + c) :=
sorry

end factor_expression_l397_397469


namespace cart_max_speed_l397_397168

/-- A small cart with a jet engine is positioned on rails. The rails are laid out
    in the shape of a circle with radius R = 5 meters. The cart starts from rest,
    and the jet force has a constant value. Given that its acceleration during
    this time should not exceed a = 1 m/s², what is the maximum speed the cart 
    can reach after completing one full circle -/
noncomputable def maxSpeedAfterOneCircle (R a : ℝ) : ℝ :=
  Real.root 4 ((16 * Real.pi ^ 2 * R ^ 2 * a ^ 2) / (16 * Real.pi ^ 2 + 1))

theorem cart_max_speed 
  (R : ℝ) (hR : R = 5) 
  (a : ℝ) (ha : a = 1) :
  maxSpeedAfterOneCircle R a = 2.23 :=
by
  sorry

end cart_max_speed_l397_397168


namespace exist_ten_digit_divisible_by_11_with_all_digits_once_l397_397629

def is_all_digits_present (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup 
  digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_divisible_by_11 (n : ℕ) : Prop :=
  let digits := n.digits 10
  let sum_odd := digits.enum.filter_map (λ ⟨i, d⟩, if i % 2 = 0 then some d else none).sum
  let sum_even := digits.enum.filter_map (λ ⟨i, d⟩, if i % 2 = 1 then some d else none).sum
  (sum_odd - sum_even) % 11 = 0

theorem exist_ten_digit_divisible_by_11_with_all_digits_once :
  ∃ n : ℕ, n.digits 10 = [9, 5, 7, 6, 8, 4, 3, 2, 1, 0] ∧ is_divisible_by_11 n :=
by
  use 9576843210
  split
  sorry -- Proof showing digits 9576843210 == [9, 5, 7, 6, 8, 4, 3, 2, 1, 0]
  sorry -- Proof showing 9576843210 is divisible by 11 (as per above definitions)

end exist_ten_digit_divisible_by_11_with_all_digits_once_l397_397629


namespace derivative_at_one_max_value_l397_397699

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Prove that f'(1) = 0
theorem derivative_at_one : deriv f 1 = 0 :=
by sorry

-- Prove that the maximum value of f(x) is 2
theorem max_value : ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = 2 :=
by sorry

end derivative_at_one_max_value_l397_397699


namespace greatest_possible_value_of_x_l397_397109

theorem greatest_possible_value_of_x : 
  (∀ x : ℚ, ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20) → 
  x ≤ 9/5 := sorry

end greatest_possible_value_of_x_l397_397109


namespace total_length_l397_397573

def length_pencil : ℕ := 12
def length_pen : ℕ := length_pencil - 2
def length_rubber : ℕ := length_pen - 3

theorem total_length : length_pencil + length_pen + length_rubber = 29 := by
  simp [length_pencil, length_pen, length_rubber]
  sorry

end total_length_l397_397573


namespace number_of_subsets_l397_397681

open Set

-- Definitions of sets A and B
def A : Set ℕ := {1}
def B : Set ℕ := {x | x < 3}

-- Proposition to prove
theorem number_of_subsets (A_sub : A ⊆ P) (P_sub : P ⊆ B) : 
  A ⊆ P ∧ P ⊆ B → (@Finset.univ {P : Set ℕ | A ⊆ P ∧ P ⊆ B}.toFinset.card = 4) :=
by
  sorry

end number_of_subsets_l397_397681


namespace an_general_term_sum_bn_l397_397695

open Nat

variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (T : ℕ → ℕ)

-- Conditions
axiom a3 : a 3 = 3
axiom S6 : S 6 = 21
axiom Sn : ∀ n, S n = n * (a 1 + a n) / 2

-- Define bn based on the given condition for bn = an + 2^n
def bn (n : ℕ) : ℕ := a n + 2^n

-- Define Tn based on the given condition for Tn.
def Tn (n : ℕ) : ℕ := (n * (n + 1)) / 2 + (2^(n + 1) - 2)

-- Prove the general term formula of the arithmetic sequence an
theorem an_general_term (n : ℕ) : a n = n :=
by
  sorry

-- Prove the sum of the first n terms of the sequence bn
theorem sum_bn (n : ℕ) : T n = Tn n :=
by
  sorry

end an_general_term_sum_bn_l397_397695


namespace choir_combinations_modulo_l397_397150

theorem choir_combinations_modulo (N : ℕ) : (N = 4095) → (N % 100 = 95) :=
by {
  assume h : N = 4095,
  calc
  N % 100 = 4095 % 100 : by rw h
  ... = 95 : by norm_num,
}

end choir_combinations_modulo_l397_397150


namespace range_of_x_for_sqrt_l397_397329

theorem range_of_x_for_sqrt (x : ℝ) (hx : sqrt (1 / (x - 1)) = sqrt (1 / (x - 1))) : x > 1 :=
by
  sorry

end range_of_x_for_sqrt_l397_397329


namespace smallest_t_for_entire_circle_l397_397834

theorem smallest_t_for_entire_circle (t : ℝ) :
  (∀ θ, 0 ≤ θ ∧ θ ≤ t → (∃ r, r = cos θ))
  → (t = π) :=
sorry

end smallest_t_for_entire_circle_l397_397834


namespace recreation_percentage_l397_397776

theorem recreation_percentage (W : ℝ) (hW : W > 0) :
  (0.40 * W) / (0.15 * W) * 100 = 267 := by
  sorry

end recreation_percentage_l397_397776


namespace g_is_odd_l397_397390

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  sorry

end g_is_odd_l397_397390


namespace find_plane_equation_l397_397643

noncomputable def point1 : EuclideanSpace ℝ (Fin 3) := ![(-1 : ℝ), (2 : ℝ), (3 : ℝ)]
noncomputable def point2 : EuclideanSpace ℝ (Fin 3) := ![2, -1, 4]
noncomputable def point3 : EuclideanSpace ℝ (Fin 3) := ![0, 0, 5]
noncomputable def plane1 : EuclideanSpace ℝ (Fin 4) := ![1, 2, 3, -6]

theorem find_plane_equation :
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) ∧
    (∀ (x y z : ℝ), ([x, y, z] = point1 ∨ [x, y, z] = point2 ∨ [x, y, z] = point3) →
                   (↑A * x + ↑B * y + ↑C * z + ↑D = 0)) ∧
    (∃ (normal_vector : EuclideanSpace ℝ (Fin 3)),
      (normal_vector ⬝ ![1, 2, 3] = 0) ∧ (normal_vector = ![0, 5, 3])) ∧
    (A = 0 ∧ B = 5 ∧ C = 3 ∧ D = -19) :=
sorry

end find_plane_equation_l397_397643


namespace min_value_x2_2xy_y2_l397_397647

theorem min_value_x2_2xy_y2 (x y : ℝ) : ∃ (a b : ℝ), (x = a ∧ y = b) → x^2 + 2*x*y + y^2 = 0 :=
by {
  sorry
}

end min_value_x2_2xy_y2_l397_397647


namespace train_distance_after_braking_l397_397584

theorem train_distance_after_braking : 
  (∃ t : ℝ, (27 * t - 0.45 * t^2 = 0) ∧ (∀ s : ℝ, s = 27 * t - 0.45 * t^2) ∧ s = 405) :=
sorry

end train_distance_after_braking_l397_397584


namespace ratio_of_speeds_l397_397122

-- Define the speeds V1 and V2
variable {V1 V2 : ℝ}

-- Given the initial conditions
def bike_ride_time_min := 10 -- in minutes
def subway_ride_time_min := 40 -- in minutes
def total_bike_only_time_min := 210 -- 3.5 hours in minutes

-- Prove the ratio of subway speed to bike speed is 5:1
theorem ratio_of_speeds (h : bike_ride_time_min * V1 + subway_ride_time_min * V2 = total_bike_only_time_min * V1) :
  V2 = 5 * V1 :=
by
  sorry

end ratio_of_speeds_l397_397122


namespace original_paint_intensity_l397_397554

theorem original_paint_intensity 
  (P : ℝ)
  (H1 : 0 ≤ P ∧ P ≤ 100)
  (H2 : ∀ (unit : ℝ), unit = 100)
  (H3 : ∀ (replaced_fraction : ℝ), replaced_fraction = 1.5)
  (H4 : ∀ (new_intensity : ℝ), new_intensity = 30)
  (H5 : ∀ (solution_intensity : ℝ), solution_intensity = 0.25) :
  P = 15 := 
by
  sorry

end original_paint_intensity_l397_397554


namespace hyperbola_asymptote_focal_length_l397_397711

theorem hyperbola_asymptote_focal_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : c = 2 * Real.sqrt 5) (h4 : b / a = 2) : a = 2 :=
by
  sorry

end hyperbola_asymptote_focal_length_l397_397711


namespace polynomial_solution_l397_397639

theorem polynomial_solution (P : ℝ → ℝ)
  (h : ∀ x y z : ℝ, x + y + z = 0 →
         P(x + y)^3 + P(y + z)^3 + P(z + x)^3 = 3 * P((x + y) * (y + z) * (z + x))) :
  P = (λ x, 0) ∨ P = (λ x, x) ∨ P = (λ x, -x) :=
sorry

end polynomial_solution_l397_397639


namespace problem1_problem2_l397_397562

-- Define the initial conditions
def initial_profit (n : ℕ) (profit_per_person : ℕ) := n * profit_per_person

-- Define the profit after adjusting x employees
def adjusted_average_profit_per_person (a : ℕ) (x : ℕ) := 10 * (a - 3 * x / 500)
def remaining_employee_profit_increase_percent (x : ℕ) := 1 + (0.002 * x : ℝ)
def remaining_employee_profit (initial_profit_per_person : ℕ) (n : ℕ) (x : ℕ) :=
  (initial_profit_per_person * n) * remaining_employee_profit_increase_percent x

-- Problem 1: Prove that x ≤ 500 under the given condition
theorem problem1 (n : ℕ) (initial_profit_per_person : ℕ) (x : ℕ) :
  (remaining_employee_profit initial_profit_per_person (n - x) x) ≥ initial_profit n initial_profit_per_person → x ≤ 500 :=
begin
  sorry
end

-- Problem 2: Prove the range for a under the given conditions
theorem problem2 (a : ℕ) (n : ℕ) (initial_profit_per_person : ℕ) (x : ℕ) :
  (remaining_employee_profit initial_profit_per_person (n - x) x) ≥ initial_profit n initial_profit_per_person →
  (adjusted_average_profit_per_person a x) * x ≤ remaining_employee_profit initial_profit_per_person (n - x) x →
  0 < a ∧ a ≤ 5 :=
begin
  sorry
end

end problem1_problem2_l397_397562


namespace value_of_b_l397_397740

theorem value_of_b (y : ℝ) (b : ℝ) (h_pos : y > 0) (h_eqn : (7 * y) / b + (3 * y) / 10 = 0.6499999999999999 * y) : 
  b = 70 / 61.99999999999999 :=
sorry

end value_of_b_l397_397740


namespace sequence_properties_l397_397204

def arithmetic_sequence (a d n : ℤ) : ℤ :=
  a + (n - 1) * d

def is_multiple_of_10 (n : ℤ) : Prop :=
  n % 10 = 0

noncomputable def sum_of_multiples_of_10 (a d n : ℤ) : ℤ :=
  ∑ i in (Finset.range (n + 1)).filter (λ k, is_multiple_of_10 (arithmetic_sequence a d k)), 
  arithmetic_sequence a d i

theorem sequence_properties :
  let a := -45
  let d := 7 in
  let last := 98 in
  let n := 21 in
  -- Number of terms in the sequence
  (arithmetic_sequence a d n ≤ last) ∧
  -- Sum of numbers in the sequence that are multiples of 10
  (sum_of_multiples_of_10 a d n = 60) :=
by
  sorry

end sequence_properties_l397_397204


namespace lucky_sum_equal_prob_l397_397361

theorem lucky_sum_equal_prob (N : ℕ) (S63_main_draw : Finset ℕ) (S44_additional_draw : Finset ℕ)
  (hN_pos : 0 < N)
  (hS63 : S63_main_draw.card = 10)
  (hSum63 : S63_main_draw.sum id = 63)
  (hS44 : S44_additional_draw.card = 8)
  (hSum44 : S44_additional_draw.sum id = 44)
  (hS_distinct : ∀ (n ∈ S63_main_draw ∪ S44_additional_draw), n ∈ Finset.range (N + 1)) :
  N = 18 :=
by
  sorry

end lucky_sum_equal_prob_l397_397361


namespace minimize_y_l397_397790

theorem minimize_y (a b : ℝ) : 
  ∃ x, x = (a + b) / 2 ∧ ∀ x', ((x' - a)^3 + (x' - b)^3) ≥ ((x - a)^3 + (x - b)^3) :=
sorry

end minimize_y_l397_397790


namespace determine_p_q_l397_397627

noncomputable def func_y (p q t : ℝ) : ℝ :=
  -(t - p)^2 + p^2 + q + 1

-- The main theorem asserting the values of p, q.
theorem determine_p_q :
  (∃ (p q : ℝ), (∀ (t : ℝ), t ∈ set.Icc (-1 : ℝ) 1 →
    (1 - t^2 + 2 * p * t + q) ≤ 9 ∧
    (1 - t^2 + 2 * p * t + q) ≥ 6) ∧
    (func_y p q 1 = 9) ∧
    (func_y p q (-1) = 6)) →
  (p = (real.sqrt 3 - 1) ∨ p = -(real.sqrt 3 - 1)) ∧ q = 4 + 2 * real.sqrt 3 := 
sorry

end determine_p_q_l397_397627


namespace find_n_l397_397712

variable (a : ℝ) (0 < a ∧ a ≤ 2)

noncomputable def a_seq : ℕ → ℝ
| 0     := a
| (n+1) := if a_seq n > 2 then a_seq n - 2 else - a_seq n + 3

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a_seq a i

theorem find_n (h : S a 1344 = 2016) : n = 1344 :=
sorry

end find_n_l397_397712


namespace max_value_of_f_l397_397833

def f : ℕ → ℕ
| n := if n < 15 then n + 15 else f (n - 7)

theorem max_value_of_f : ∃ n : ℕ, f n = 29 := 
by {
  existsi 14,
  simp [f],
  exact if_pos (by norm_num),
}

end max_value_of_f_l397_397833


namespace minimum_value_f_l397_397285

open Real

noncomputable def f (m : ℝ) (x : ℝ) := x^4 * cos x + m * x^2 + 2 * x

theorem minimum_value_f'' (m : ℝ) :
  (∀ x ∈ Icc (-4 : ℝ) 4, has_deriv_at (deriv (deriv (f m))) x (16)) →
  ∃ c ∈ Icc (-4 : ℝ) 4, deriv (deriv (f m)) c = -12 := 
sorry

end minimum_value_f_l397_397285


namespace pizza_order_l397_397156

theorem pizza_order (couple_want: ℕ) (child_want: ℕ) (num_couples: ℕ) (num_children: ℕ) (slices_per_pizza: ℕ)
  (hcouple: couple_want = 3) (hchild: child_want = 1) (hnumc: num_couples = 1) (hnumch: num_children = 6) (hsp: slices_per_pizza = 4) :
  (couple_want * 2 * num_couples + child_want * num_children) / slices_per_pizza = 3 := 
by
  -- Proof here
  sorry

end pizza_order_l397_397156


namespace sum_first_three_cards_example_l397_397728

-- Definitions based on the conditions
inductive Color
  | Blue
  | Green

def Card := {number : ℕ // number > 0}

def is_valid_sequence (seq : List (Color × Card)) : Prop :=
  seq.length > 2 ∧ 
  (∀ (n i : ℕ), (i < seq.length - 1) → 
                 (seq.nth i).isSome ∧ 
                 ((seq.nth (i + 1)).isSome ∧ 
                  match (seq.nth i).get_or_else (Color.Blue, ⟨0, by decide⟩), (seq.nth (i + 1)).get_or_else (Color.Blue, ⟨0, by decide⟩) with
                  | (Color.Blue, ⟨b, _⟩), (Color.Green, ⟨g, _⟩) => b = g ∨ b = g + 1
                  | (Color.Green, ⟨g, _⟩), (Color.Blue, ⟨b, _⟩) => g = b ∨ g = b + 1
                  | (_, _), (_, _) => False
                  end
                )
              )

def total (cards : List (Color × Card)) : ℕ := 
  cards.foldl (λ acc (_, card) => acc + card.1) 0

-- Given an example valid sequence, the sum of the first three cards is 5
theorem sum_first_three_cards_example : 
  ∃ seq : List (Color × Card),
  (seq.length ≥ 3 ∧ is_valid_sequence seq) ∧ total (seq.take 3) = 5 :=
by
  sorry


end sum_first_three_cards_example_l397_397728


namespace solve_equation_l397_397041

-- Define the polynomial equation given the condition
noncomputable def equation : ℂ → ℂ :=
  λ x, (x^3 + 4 * x^2 * complex.sqrt 3 + 12 * x + 8 * complex.sqrt 3) + (x + complex.sqrt 3)

-- Define the proof statement that the three specific values are the roots of the equation
theorem solve_equation : ∀ x : ℂ,
  equation x = 0 ↔
  (x = -complex.sqrt 3) ∨
  (x = complex.I - complex.sqrt 3) ∨
  (x = -complex.I - complex.sqrt 3) :=
by
  sorry

end solve_equation_l397_397041


namespace find_length_FC_l397_397899

-- Definitions based on given conditions
def Rectangle (A B C D : Type) : Prop :=
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) ∧
    (l > 0 ∧ w > 0) ∧ (dist A B = l) ∧ (dist B C = w) ∧ (dist C D = l) ∧ (dist D A = w)

def Midpoint (E C D : Type) : Prop :=
  dist E C = dist E D ∧ collinear C E D

def OnSide (F B C : Type) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ≤ 1 ∧ F = t • B + (1 - t) • C

def RightAngle (A E F : Type) : Prop :=
  interior_angle A E F = π / 2

def Length (AF : ℝ) (F A : Type) : Prop :=
  dist F A = AF

def LengthBF (BF : ℝ) (F B : Type) : Prop :=
  dist F B = BF

-- The statement to prove
theorem find_length_FC {A B C D E F : Type} (AF : ℝ) (BF : ℝ) :
   Rectangle A B C D  → Midpoint E C D → OnSide F B C → RightAngle A E F → 
   Length 7 F A → LengthBF 4 F B → 
   dist F C = 1.5 :=
by sorry

end find_length_FC_l397_397899


namespace part_a_part_b_part_c_part_d_l397_397905

-- Part (a)
theorem part_a (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  ∃ (x_n : ℕ → ℕ), (∀ n, x_n n ∈ {0, 1, 2}) ∧ (x = ∑' n, x_n n / 3^n) :=
sorry

-- Part (b)
theorem part_b (x : ℝ) (x_n y_n : ℕ → ℕ)
  (hx : x = ∑' n, x_n n / 3^n) 
  (hy : x = ∑' n, y_n n / 3^n)
  (hx_non_trunc : ∑' n, abs (x_n n) = ⊤)
  (hy_non_trunc : ∑' n, abs (y_n n) = ⊤) :
  ∀ n, x_n n = y_n n :=
sorry

-- Part (c)
def cantor_set : set ℝ := {x | ∃ (x_n : ℕ → ℕ), (∀ n, x_n n ∈ {0, 2}) ∧ (x = ∑' n, x_n n / 3^n)}

theorem part_c :
  cantor_set = {x | ∃ (x_n : ℕ → ℕ), (∀ n, x_n n ∈ {0, 2}) ∧ (x = ∑' n, x_n n / 3^n)} :=
sorry

-- Part (d)
def metric_space_sequences := {s : ℕ → ℕ // ∀ n, s n ∈ {0, 1}}
def sequence_metric (x y : metric_space_sequences) : ℝ := ∑' n, abs (x.val n - y.val n) / 2^n

theorem part_d :
  ∃ (ϕ : cantor_set → metric_space_sequences)
    (hϕ : continuous ϕ)
    (hϕ_inv : continuous (inverse ϕ)),
    bijective ϕ :=
sorry

end part_a_part_b_part_c_part_d_l397_397905


namespace probability_vertex_B_bottom_vertex_l397_397197

-- Define the vertices
-- A, B are arbitrary vertices following the movement pattern described

variable (top_vertex : Type)
variable (middle_vertex : Type)
variable (bottom_vertex : Type)

-- Total count of bottom vertices
constant num_bottom_vertices : ℕ := 3

-- The movement patterns in the dodecahedron
constant adjacent_middle_vertices : (top_vertex → set middle_vertex)
constant adjacent_vertices_from_middle : (middle_vertex → set middle_vertex)
constant bottom_vertex_adjacent : (middle_vertex → set bottom_vertex)

-- Initial conditions: all these relationships are sets of adjacent vertices
axiom (top_vertex: top_vertex) :
  set.size (adjacent_middle_vertices top_vertex) = 3

axiom (middle_vertex: middle_vertex) :
  set.size (adjacent_vertices_from_middle middle_vertex) = 2 ∧
  set.size (bottom_vertex_adjacent middle_vertex) = 1

theorem probability_vertex_B_bottom_vertex
  (top_vertex : top_vertex)
  (A B : top_vertex) :
  (P (B = bottom_vertex)) = 1 / 3 :=
sorry

end probability_vertex_B_bottom_vertex_l397_397197


namespace triangle_inequality_l397_397419

variables {R : Type*} [ordered_field R]

structure Point2D (R : Type*) := 
(x y : R)

def f (a b c : R) (p : Point2D R) : R := a * p.x + b * p.y + c

theorem triangle_inequality 
  (a b c : R) 
  (P Q R : Point2D R) 
  (fP : R) 
  (fQ : R) 
  (fR : R) 
  (A : Point2D R) 
  (hP : f a b c P = fP) 
  (hQ : f a b c Q = fQ) 
  (hR : f a b c R = fR) 
  (hA_in_ΔPQR : ∃λ r₁ r₂ r₃ : R, (r₁ + r₂ + r₃ = 1) ∧ (0 ≤ r₁) ∧ (0 ≤ r₂) ∧ (0 ≤ r₃) ∧ (A.x = r₁ * P.x + r₂ * Q.x + r₃ * R.x) ∧ (A.y = r₁ * P.y + r₂ * Q.y + r₃ * R.y)) :
  f a b c A ≤ max (max fP fQ) fR := sorry


end triangle_inequality_l397_397419


namespace top_coat_drying_time_l397_397399

theorem top_coat_drying_time : 
  ∀ (base_coat_time color_coat_time color_coats total_dry_time : ℕ), 
  base_coat_time = 2 → 
  color_coat_time = 3 → 
  color_coats = 2 → 
  total_dry_time = 13 → 
  base_coat_time + (color_coat_time * color_coats) + top_coat_time = total_dry_time → 
  top_coat_time = 5 := 
by 
  intros base_coat_time color_coat_time color_coats total_dry_time 
         h_base h_color h_coats h_total h_eq.
  sorry

end top_coat_drying_time_l397_397399


namespace greatest_possible_value_of_x_l397_397108

theorem greatest_possible_value_of_x : 
  (∀ x : ℚ, ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20) → 
  x ≤ 9/5 := sorry

end greatest_possible_value_of_x_l397_397108


namespace x_intercept_of_line_l397_397642

theorem x_intercept_of_line : ∃ x : ℚ, (6 * x, 0) = (35 / 6, 0) :=
by
  use 35 / 6
  sorry

end x_intercept_of_line_l397_397642


namespace eight_machines_produce_ninety_six_bottles_in_three_minutes_l397_397894

-- Define the initial conditions
def rate_per_machine: ℕ := 16 / 4 -- bottles per minute per machine

def total_bottles_8_machines_3_minutes: ℕ := 8 * rate_per_machine * 3

-- Prove the question
theorem eight_machines_produce_ninety_six_bottles_in_three_minutes:
  total_bottles_8_machines_3_minutes = 96 :=
by
  sorry

end eight_machines_produce_ninety_six_bottles_in_three_minutes_l397_397894


namespace egyptian_pi_approx_l397_397450

/- Definitions for the geometric conditions -/
def side_of_square (d : ℝ) : ℝ := (8 / 9) * d
def area_circle (d : ℝ) (pi : ℝ) : ℝ := pi * (d / 2) ^ 2
def area_square (s : ℝ) : ℝ := s ^ 2

/- Theorem stating the equality and the approximation of pi -/
theorem egyptian_pi_approx (d : ℝ) :
  (area_square (side_of_square d) = area_circle d 3.16) → 
  ∃ pi, pi = 3.16 :=
by
  sorry -- proof goes here

end egyptian_pi_approx_l397_397450


namespace length_of_AE_l397_397332

noncomputable def AE_calculation (AB AC AD : ℝ) (h : ℝ) (AE : ℝ) : Prop :=
  AB = 3.6 ∧ AC = 3.6 ∧ AD = 1.2 ∧ 
  (0.5 * AC * h = 0.5 * AE * (1/3) * h) →
  AE = 10.8

theorem length_of_AE {h : ℝ} : AE_calculation 3.6 3.6 1.2 h 10.8 :=
sorry

end length_of_AE_l397_397332


namespace reject_null_hypothesis_l397_397238

noncomputable def test_statistic (xbar : ℝ) (mu_0 : ℝ) (n : ℕ) (sigma : ℝ) : ℝ :=
  ((xbar - mu_0) * real.sqrt n) / sigma

theorem reject_null_hypothesis :
  let xbar := 27.56
  let mu_0 := 26
  let n := 100
  let sigma := 5.2
  let alpha := 0.05
  let u_critical := 1.96
  test_statistic xbar mu_0 n sigma > u_critical :=
by
  sorry

end reject_null_hypothesis_l397_397238


namespace g_is_odd_l397_397392

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  sorry

end g_is_odd_l397_397392


namespace profit_percent_is_50_l397_397074

-- Define the ratio of the cost price to the selling price
def ratio_of_cp_sp := ∃ x > 0, ∀ CP SP, CP = 2 * x ∧ SP = 3 * x

-- Define the profit percentage calculation
def profit_percent (CP SP : ℝ) : ℝ := ((SP - CP) / CP) * 100

-- State the theorem
theorem profit_percent_is_50 :
  ratio_of_cp_sp → ∀ CP SP, profit_percent CP SP = 50 :=
by
  intro h x CP SP hx₀ (hx₁, hx₂)
  -- proof would go here
  sorry

end profit_percent_is_50_l397_397074


namespace formula_a_formula_b_geometric_c_sum_inequality_l397_397278

-- Definitions and conditions
def a (n : ℕ) := 2 * n - 1
def b (n : ℕ) := 4 ^ n
def c (n : ℕ) := b (2 * n) + 1 / b n

-- Questions transformed into Lean statements
theorem formula_a (n : ℕ) : a n = 2 * n - 1 := sorry
theorem formula_b (n : ℕ) : b n = 4 ^ n := sorry
theorem geometric_c : ∃ r : ℝ, ∀ n : ℕ, c (n + 1) ^ 2 - c (2 * (n + 1)) = r * (c n ^ 2 - c (2 * n)) := sorry
theorem sum_inequality (n : ℕ) : ∑ k in Finset.range n, Real.sqrt (a k * a (k + 1) / (c k ^ 2 - c (2 * k))) < 2 * Real.sqrt 2 := sorry

end formula_a_formula_b_geometric_c_sum_inequality_l397_397278


namespace shortest_distance_pyramid_l397_397691

noncomputable theory

def pyramid (height : ℝ) (side_length : ℝ) : Prop :=
  (height = 2) ∧ (side_length = sqrt 2)

def points_on_segments (P Q : ℝ) (BD SC : ℝ) : Prop :=
  (P ∈ segment BD) ∧ (Q ∈ segment SC)

def shortest_distance (P Q : ℝ) (d : ℝ) : Prop :=
  d = dist P Q

theorem shortest_distance_pyramid :
  ∀ (P Q : ℝ), pyramid 2 (sqrt 2) ∧ points_on_segments P Q (BD: ℝ) (SC: ℝ) →
  shortest_distance P Q (2 * sqrt 5 / 5) := 
by
  sorry

end shortest_distance_pyramid_l397_397691


namespace largest_digit_for_divisibility_l397_397110

theorem largest_digit_for_divisibility (N : ℕ) (h1 : N % 2 = 0) (h2 : (3 + 6 + 7 + 2 + N) % 3 = 0) : N = 6 :=
sorry

end largest_digit_for_divisibility_l397_397110


namespace negation_equivalence_l397_397063

-- Definition of the original proposition
def proposition (x : ℝ) : Prop := x > 1 → Real.log x > 0

-- Definition of the negated proposition
def negation (x : ℝ) : Prop := ¬ (x > 1 → Real.log x > 0)

-- The mathematically equivalent proof problem as Lean statement
theorem negation_equivalence (x : ℝ) : 
  (¬ (x > 1 → Real.log x > 0)) ↔ (x ≤ 1 → Real.log x ≤ 0) := 
by 
  sorry

end negation_equivalence_l397_397063


namespace angle_in_equilateral_triangle_surrounded_by_squares_l397_397938

/-- In an equilateral triangle surrounded by three squares with specific side and angle relationships,
    the angle formed at one vertex of the triangle (as described by \(∠QPR\)) is 30 degrees. -/
theorem angle_in_equilateral_triangle_surrounded_by_squares :
  ∀ (P Q R : Type) 
    (RS RT : P = R) 
    (PR RQ : Q = R) 
    (RS_eq_RT PR_eq_RQ : PR = RQ) 
    (angle_PRS angle_QRT angle_SRT : ℝ),
  angle_PRS = 90 → angle_QRT = 90 → angle_SRT = 60 →
  angle_PRS + angle_QRT + angle_SRT = 360 →
  angle_PRQ = 120 →
  angle_QPR = 30 :=
begin
  intros P Q R RS RT PR RQ RS_eq_RT PR_eq_RQ angle_PRS angle_QRT angle_SRT
    h_PRS h_QRT h_SRT h_sum angle_PRQ,
  sorry,
end

end angle_in_equilateral_triangle_surrounded_by_squares_l397_397938


namespace number_of_small_numbers_l397_397014

def is_small_number (n : ℕ) : Prop :=
  (n >= 10^9) ∧ (n < 10^10) ∧ (∀ m, (m >= 10^9) ∧ (m < 10^10) ∧ (digit_sum m = digit_sum n) → n ≤ m)

theorem number_of_small_numbers : ∃ (count : ℕ), count = 90 ∧ 
  ∀ n, is_small_number n ↔ ∃ k, k ∈ (finset.range 90).map (λ x, x+1) ∧ digit_sum n = k :=
sorry

end number_of_small_numbers_l397_397014


namespace min_boys_needed_l397_397050

theorem min_boys_needed
  (T : ℕ) -- total apples
  (n : ℕ) -- total number of boys
  (x : ℕ) -- number of boys collecting 20 apples each
  (y : ℕ) -- number of boys collecting 20% of total apples each
  (h1 : n = x + y)
  (h2 : T = 20 * x + Nat.div (T * 20 * y) 100)
  (hx_pos : x > 0) 
  (hy_pos : y > 0) : n ≥ 2 :=
sorry

end min_boys_needed_l397_397050


namespace angle_F1PF2_is_120_l397_397537

-- Define the hyperbola with foci and conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 4 - y^2 / 45 = 1

-- Define the properties of the focus points F1 and F2
def is_focus (F1 F2 : ℝ × ℝ) : Prop :=
  F1.1 = -7 ∧ F1.2 = 0 ∧ F2.1 = 7 ∧ F2.2 = 0

-- Define the arithmetic sequence condition
def arith_seq_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  let PF1 := real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2),
      PF2 := real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2),
      F1F2 := real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) in
  2 * PF2 = PF1 + F1F2

-- The condition that the common difference is greater than 0
def positive_common_difference (P F1 F2 : ℝ × ℝ) : Prop :=
  let PF1 := real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2),
      PF2 := real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) in
  PF2 - PF1 > 0

-- Prove that the given angle is 120 degrees
theorem angle_F1PF2_is_120 (P F1 F2 : ℝ × ℝ)
  (hp: hyperbola_eq P.1 P.2)
  (hf: is_focus F1 F2)
  (hac: arith_seq_condition P F1 F2)
  (hpd: positive_common_difference P F1 F2) :
  ∠ F1 P F2 = 120 :=
sorry -- the proof is omitted

end angle_F1PF2_is_120_l397_397537


namespace samia_walk_distance_l397_397435

theorem samia_walk_distance
  (bike_speed : ℕ := 15)
  (walk_speed : ℕ := 4)
  (total_time_minutes : ℕ := 56) :
  Samia_walk_distance = 2.4 :=
by
  -- Define the total distance components and equation setup
  let total_time_hours := (total_time_minutes : ℝ) / 60
  let bike_fraction := 2 / 3
  let walk_fraction := 1 / 3

  -- Calculate walk distance
  let walk_distance := (total_time_hours * bike_speed * walk_speed) / (2 * walk_speed + 15 * walk_fraction)
  
  have h : walk_distance = 2.4 := by
    -- Proof should go here
    sorry

  exact h

end samia_walk_distance_l397_397435


namespace relationship_correlation_l397_397933

def student's_learning_attitude
def academic_performance
def teacher's_teaching_level
def student's_height
def family's_economic_condition

theorem relationship_correlation :
  (∃ f : student's_learning_attitude → academic_performance, ∀ s, correlation f (s))
  ∧ (∃ g : teacher's_teaching_level → academic_performance, ∀ t, correlation g (t)) :=
by
  sorry

end relationship_correlation_l397_397933


namespace quadrilateral_perimeter_l397_397515

theorem quadrilateral_perimeter
  (AB BC DC BD : ℕ)
  (H1 : AB = 7)
  (H2 : BC = 15)
  (H3 : DC = 7)
  (H4 : BD = 24)
  (H5 : ∀ (a b d : ℕ), (a^2 + b^2 = d^2) → (a = 7 ∧ b = 24) → d = 25) :
  AB + BC + DC + 25 = 54 := 
by
  rw [H1, H2, H3]
  norm_num
  sorry

end quadrilateral_perimeter_l397_397515


namespace hyperbola_equation_proof_l397_397252

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - y^2 / 3 = 1

noncomputable def parabola_focus : ℝ × ℝ := (2, 0)

noncomputable def passes_through_point : Prop :=
  hyperbola_eq (real.sqrt 2) (real.sqrt 3)

theorem hyperbola_equation_proof :
  (hyperbola_eq (real.sqrt 2) (real.sqrt 3))
  ∧ (∃ (λ : ℝ), λ > 0 
  ∧ ∀ (P : ℝ × ℝ) (hP : hyperbola_eq P.1 P.2 ∧ P.1 > 0 ∧ P.2 > 0), 
    ∃ (A : ℝ × ℝ) (F : ℝ × ℝ),
    A.1 = -real.sqrt 4 ∧ A.2 = 0 
    ∧ F.1 = 2 ∧ F.2 = 0
    ∧ angle P F A = 2 * angle P A F) :=
begin
  -- Insert proof here
  sorry
end

end hyperbola_equation_proof_l397_397252


namespace average_of_middle_three_l397_397464

theorem average_of_middle_three
  (a b c d e : ℕ)
  (h_distinct : a < b ∧ b < c ∧ c < d ∧ d < e)
  (h_sum : a + b + c + d + e = 25)
  (h_max_diff : ∀ x y : ℕ, x + y = 24 → (e - a) ≥ (y - x)) :
  (b + c + d) / 3 = 3 :=
by
  sorry

end average_of_middle_three_l397_397464


namespace find_a_l397_397094

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^4 - 18 * x^3 + ((86 : ℝ)) * x^2 + 200 * x - 1984

-- Define the condition and statement
theorem find_a (α β γ δ : ℝ) (hαβγδ : α * β * γ * δ = -1984)
  (hαβ : α * β = -32) (hγδ : γ * δ = 62) :
  (∀ a : ℝ, a = 86) :=
  sorry

end find_a_l397_397094


namespace minimum_left_translation_symmetry_l397_397474

noncomputable def transformed_function (x m : ℝ) : ℝ :=
  2 * Real.sin (x + m + Real.pi / 3)

noncomputable def original_function (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.cos x + Real.sin x

noncomputable def is_symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x) = -f (-x)

theorem minimum_left_translation_symmetry :
  ∃ m : ℝ, m > 0 ∧ is_symmetric_about_origin (transformed_function · m) ∧
    (∀ m' : ℝ, m' > 0 ∧ is_symmetric_about_origin (transformed_function · m') → m ≤ m') :=
exists.intro (2 * Real.pi / 3) (and.intro 
  (by norm_num [Real.pi, Real.pi_pos])  -- to show pi > 0
  (and.intro
    (by {
        intros x,
        simp [transformed_function, Real.sin],
        sorry
    })
    (by {
        intros m' h1 h2,
        sorry
    })
  ))

end minimum_left_translation_symmetry_l397_397474


namespace acute_angles_triangle_rulers_l397_397729

theorem acute_angles_triangle_rulers : 
  let angles := {30, 45, 60}
  ∃ S : Set ℤ, ∀ θ ∈ S, 0 < θ ∧ θ < 90 ∧ S = {15, 30, 45, 60, 75} ∧ S.card = 5 :=
by
  sorry

end acute_angles_triangle_rulers_l397_397729


namespace equal_likelihood_of_lucky_sums_solution_l397_397354

variables (N : ℕ)

def main_draw := 10
def additional_draw := 8

def lucky_sum_main (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = main_draw ∧ S.sum = sum
def lucky_sum_additional (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = additional_draw ∧ S.sum = sum

theorem equal_likelihood_of_lucky_sums :
  lucky_sum_main N 63 ↔ lucky_sum_additional N 44 :=
sorry

theorem solution :
  equal_likelihood_of_lucky_sums 18 :=
sorry

end equal_likelihood_of_lucky_sums_solution_l397_397354


namespace volume_weight_proportionality_l397_397493

noncomputable def proportional_volume (k : ℝ) (W : ℝ) : ℝ :=
  k * W

theorem volume_weight_proportionality (W' : ℝ) (hW' : W' < 112) :
  let k := (48 / 112) in
  proportional_volume k W' = 0.4286 * W' :=
by
  sorry

end volume_weight_proportionality_l397_397493


namespace sales_volume_function_max_profit_min_boxes_for_2000_profit_l397_397557

-- Definitions and conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def init_boxes_sold : ℝ := 250
def price_increase_effect : ℝ := 10
def max_selling_price : ℝ := 38

-- Question 1: Find functional relationship between daily sales volume y and selling price per box x
theorem sales_volume_function (x : ℝ) (hx : x ≥ min_selling_price) :
  ∃ y, y = -10 * x + 500 := by
  sorry

-- Question 2: Find the price per box to maximize daily sales profit and the maximum profit
theorem max_profit (x : ℝ) (hx : x = 35) :
  ∃ P, P = -10 * (x-20) * (x) + 500 * (x-20) := by
  sorry

-- Question 3: Determine min boxes sold to make at least 2000 yuan given price does not exceed 38 yuan
theorem min_boxes_for_2000_profit (x : ℝ) (hx : min_selling_price ≤ x ∧ x ≤ max_selling_price ∧ 
                             -10 * (x-20) * (-10 * x + 500) ≥ 2000) :
  ∃ y, y = -10 * x + 500 ∧ y ≥ 120 := by
  sorry

end sales_volume_function_max_profit_min_boxes_for_2000_profit_l397_397557


namespace events_equally_likely_iff_N_eq_18_l397_397367

variable (N : ℕ)

-- Define the number of combinations in the draws
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the sums of selecting balls
noncomputable def S_63_10 (N: ℕ) : ℕ := sorry -- placeholder definition
noncomputable def S_44_8 (N: ℕ) : ℕ := sorry 

-- Condition for the events being equally likely
theorem events_equally_likely_iff_N_eq_18 : 
  (S_63_10 N * C N 8 = S_44_8 N * C N 10) ↔ N = 18 :=
sorry

end events_equally_likely_iff_N_eq_18_l397_397367


namespace max_area_of_cut_triangle_l397_397414

noncomputable def max_possible_area_of_triangle : ℝ :=
  let AB := 13
  let BC := 14
  let CA := 15
  let Perimeter := AB + BC + CA
  let max_area := (1323 / 26 : ℝ)
  max_area

theorem max_area_of_cut_triangle (AB BC CA : ℝ) (h1 : AB = 13) (h2 : BC = 14) (h3 : CA = 15) :
  ∃ (ℓ : ℝ), line_cuts_triangle_in_half_perimeter AB BC CA ℓ → 
  max_possible_area_of_triangle = 1323 / 26 :=
by
  sorry

end max_area_of_cut_triangle_l397_397414


namespace length_CD_eq_14_l397_397384

theorem length_CD_eq_14 (A B C D E : Type)
  (hAB_AC : AB = AC)
  (hD_mid_AB : midpoint D AB)
  (hD_mid_CE : midpoint D CE)
  (hBC_14 : length BC = 14) :
  length CD = 14 :=
sorry

end length_CD_eq_14_l397_397384


namespace totalPieces_l397_397434

   -- Definitions given by the conditions
   def packagesGum := 21
   def packagesCandy := 45
   def packagesMints := 30
   def piecesPerGumPackage := 9
   def piecesPerCandyPackage := 12
   def piecesPerMintPackage := 8

   -- Define the total pieces of gum, candy, and mints
   def totalPiecesGum := packagesGum * piecesPerGumPackage
   def totalPiecesCandy := packagesCandy * piecesPerCandyPackage
   def totalPiecesMints := packagesMints * piecesPerMintPackage

   -- The mathematical statement to prove
   theorem totalPieces :
     totalPiecesGum + totalPiecesCandy + totalPiecesMints = 969 :=
   by
     -- Proof is skipped
     sorry
   
end totalPieces_l397_397434


namespace find_x_l397_397722

variable (x : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (x, 1)
def u : ℝ × ℝ := (1 + 2 * x, 4)
def v : ℝ × ℝ := (2 - 2 * x, 2)

theorem find_x (h : 2 * (1 + 2 * x) = 4 * (2 - 2 * x)) : x = 1 / 2 := by
  sorry

end find_x_l397_397722


namespace min_max_value_in_interval_l397_397236

theorem min_max_value_in_interval : ∀ (x : ℝ),
  -2 < x ∧ x < 5 →
  ∃ (y : ℝ), (y = -1.5 ∨ y = 1.5) ∧ y = (x^2 - 4 * x + 6) / (2 * x - 4) := 
by sorry

end min_max_value_in_interval_l397_397236


namespace restaurant_total_spent_l397_397443

theorem restaurant_total_spent (appetizer_cost : ℕ) (entree_cost : ℕ) (num_entrees : ℕ) (tip_rate : ℚ) 
  (H1 : appetizer_cost = 10) (H2 : entree_cost = 20) (H3 : num_entrees = 4) (H4 : tip_rate = 0.20) :
  appetizer_cost + num_entrees * entree_cost + (appetizer_cost + num_entrees * entree_cost) * tip_rate = 108 :=
by
  sorry

end restaurant_total_spent_l397_397443


namespace average_of_middle_three_l397_397462

theorem average_of_middle_three
  (a b c d e : ℕ)
  (h_distinct : a < b ∧ b < c ∧ c < d ∧ d < e)
  (h_sum : a + b + c + d + e = 25)
  (h_max_diff : ∀ x y : ℕ, x + y = 24 → (e - a) ≥ (y - x)) :
  (b + c + d) / 3 = 3 :=
by
  sorry

end average_of_middle_three_l397_397462


namespace min_max_values_in_interval_l397_397646

def func (x y : ℝ) : ℝ := 3 * x^2 * y - 2 * x * y^2

theorem min_max_values_in_interval :
  (∀ x y, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → func x y ≥ -1/3) ∧
  (∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ func x y = -1/3) ∧
  (∀ x y, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → func x y ≤ 9/8) ∧
  (∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ func x y = 9/8) :=
by
  sorry

end min_max_values_in_interval_l397_397646


namespace functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l397_397560

-- Definitions for the problem conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def max_selling_price : ℝ := 38
def base_sales_volume : ℝ := 250
def price_decrease_effect : ℝ := 10
def profit_requirement : ℝ := 2000

-- Given the initial conditions
noncomputable def sales_volume (x : ℝ) : ℝ := base_sales_volume - price_decrease_effect * (x - min_selling_price)

-- Target problem statement
-- Part 1: Functional relationship between y and x
theorem functional_relationship (x : ℝ) : sales_volume x = -10 * x + 500 := by
sorry

-- Part 2: Maximizing profit
noncomputable def profit (x : ℝ) : ℝ := (x - cost_per_box) * sales_volume x

theorem maximizing_profit : ∃ (x : ℝ), x = 35 ∧ profit x = 2250 := by
sorry

-- Part 3: Minimum number of boxes to sell for at least 2000 yuan profit
theorem minimum_boxes_for_2000_profit (x : ℝ) : x ≤ max_selling_price → profit x ≥ profit_requirement → sales_volume x ≥ 120 := by
sorry

end functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l397_397560


namespace parabola_transformation_correct_l397_397067

theorem parabola_transformation_correct:
  let initial_eq : ℝ → ℝ := λ x, 2 * (x + 1)^2 - 3
  let transformed_eq : ℝ → ℝ := λ x, 2 * x^2
  ∃ seq : list (ℝ → ℝ → ℝ → ℝ),
    (seq = [λ x y c, y = 2 * (x - 1)^2 - 3, λ x y c, y = y + 3]) ∧
    ∀ x, (seq.head x (initial_eq x) 0 = transformed_eq x) :=
sorry

end parabola_transformation_correct_l397_397067


namespace measure_angle_ZXY_l397_397136

-- Define the structure of the problem.
structure IsoscelesTriangle (X Y Z : Type) : Prop :=
(eqSides : X = Y)

-- Assume the conditions of the problem.
variables {X Y Z : Type} [Inhabited X] [Inhabited Y] [Inhabited Z]

axiom angle_sum_of_triangle (α β γ : ℕ) : α + β + γ = 180
axiom is_isosceles (XZeqYZ : X = Y) (a : ℕ) : IsoscelesTriangle X Y Z
axiom measure_angle_Z (m∠Z : ℕ) : m∠Z = 50

-- Prove the result for m∠ZXY.
theorem measure_angle_ZXY : ∀ {X Y Z : Type} [Inhabited X] [Inhabited Y] [Inhabited Z],
  (is_isosceles (XZeqYZ : X = Y) (a : ℕ)) → 
  (measure_angle_Z (m∠Z : ℕ)) →
  m∠ZXY = 65 :=
by {
  sorry
}

end measure_angle_ZXY_l397_397136


namespace quadrilateral_is_trapezoid_l397_397353

variables {V : Type*} [AddCommGroup V] [Module ℝ V] -- Define the type of vectors and vector space over the reals
variables (a b : V) -- Vectors a and b
variables (AB BC CD AD : V) -- Vectors representing sides of quadrilateral

-- Condition: vectors a and b are not collinear
def not_collinear (a b : V) : Prop := ∀ k : ℝ, k ≠ 0 → a ≠ k • b

-- Given Conditions
def conditions (a b AB BC CD : V) : Prop :=
  AB = a + 2 • b ∧
  BC = -4 • a - b ∧
  CD = -5 • a - 3 • b ∧
  not_collinear a b

-- The to-be-proven property
def is_trapezoid (AB BC CD AD : V) : Prop :=
  AD = 2 • BC

theorem quadrilateral_is_trapezoid 
  (a b AB BC CD : V) 
  (h : conditions a b AB BC CD)
  : is_trapezoid AB BC CD (AB + BC + CD) :=
sorry

end quadrilateral_is_trapezoid_l397_397353


namespace equalizer_planes_count_l397_397658

noncomputable def numberOfEqualizerPlanes (A B C D : Point3D) : Nat :=
if non_coplanar A B C D then 7 else 0

theorem equalizer_planes_count (A B C D : Point3D) (h_non_coplanar : non_coplanar A B C D) : 
  numberOfEqualizerPlanes A B C D = 7 := by
sorry

end equalizer_planes_count_l397_397658


namespace complex_numbers_sum_l397_397853

theorem complex_numbers_sum
  (a b c d e f : ℂ)
  (h1 : d = 2)
  (h2 : e = -a - 2 * c)
  (h3 : a + c + e + (b + d + f) * complex.I = -7 * complex.I) :
  b + f = -9 :=
by
  -- Insert the proof here
  sorry

end complex_numbers_sum_l397_397853


namespace sin_double_angle_l397_397733

theorem sin_double_angle (α : ℝ) (h1 : cos ((π / 2) - α) = 1 / 3) 
  (h2 : π / 2 < α ∧ α < π) : sin (2 * α) = -4 * sqrt 2 / 9 :=
sorry

end sin_double_angle_l397_397733


namespace tangency_of_BO_l397_397031

open_locale classical

variables {Point : Type}
variables (A B C X Y O M N : Point)
variables [incidence_geometry Point]

-- Conditions
axiom cond1 : lies_on Y (line_through A B)
axiom cond2 : lies_on X (line_through B C)
axiom cond3 : concyclic A X Y C
axiom cond4 : intersection (line_through A X) (line_through C Y) = O
axiom midpoint_M : midpoint M A C
axiom midpoint_N : midpoint N X Y

-- Statement to prove
theorem tangency_of_BO :
  tangent (circle_through M O N) (line_through B O) :=
sorry

end tangency_of_BO_l397_397031


namespace smallest_result_obtained_l397_397518

/-- Problem Statement:
Given a set of numbers {2, 4, 6, 8, 10, 12}, we choose three different numbers 
from the set, subtract the smallest from the largest, and then multiply 
the result by the third number. Prove that the smallest result that can 
be obtained from this process is 4.
-/

def set_of_numbers := {2, 4, 6, 8, 10, 12}

def process (a b c : ℕ) : ℕ :=
  (b - a) * c

theorem smallest_result_obtained : 
  ∃ a b c : ℕ, a ∈ set_of_numbers ∧ b ∈ set_of_numbers ∧ c ∈ set_of_numbers ∧
               a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
               (process a b c = 4) :=
sorry

end smallest_result_obtained_l397_397518


namespace number_of_correct_propositions_l397_397932

-- Definitions of the propositions
def p1 := ∀ (A B : Triangle), A.equilateral → B.equilateral → A.similar_to B
def p2 := ∀ (A B : Triangle), A.right → B.right → A.similar_to B
def p3 := ∀ (A B : Triangle), A.isosceles → B.isosceles → A.similar_to B
def p4 := ∀ (A B : Triangle), A.acute → B.acute → A.similar_to B
def p5 := ∀ (A B : Triangle), A.isosceles → B.isosceles → A.congruent_to B
def p6 := ∀ (A B : Triangle), A.isosceles → B.isosceles → (∃ θ, A.has_angle θ ∧ B.has_angle θ) → A.similar_to B
def p7 := ∀ (A B : Triangle), A.isosceles → B.isosceles → (∃ θ, θ > 90 ∧ A.has_angle θ ∧ B.has_angle θ) → A.similar_to B
def p8 := ∀ (A B : Triangle), A.congruent_to B → A.similar_to B

-- Proof problem statement
theorem number_of_correct_propositions : (count_correct_propositions [p1, p2, p3, p4, p5, p6, p7, p8] = 3) := sorry

end number_of_correct_propositions_l397_397932


namespace sequence_property_l397_397257

theorem sequence_property (a : ℕ+ → ℤ) (h_add : ∀ p q : ℕ+, a (p + q) = a p + a q) (h_a2 : a 2 = -6) :
  a 10 = -30 := 
sorry

end sequence_property_l397_397257


namespace trig_eq_solutions_l397_397887

theorem trig_eq_solutions (n k : ℤ) :
  (∃ x : ℝ, (sin (2 * x) + sqrt 3 * cos (2 * x))^2 = 2 - 2 * cos ((2 / 3) * π - x) ↔
  x = (2 * π / 5) * n ∨ x = (2 * π / 9) * (3 * k + 1)) :=
by
  sorry

end trig_eq_solutions_l397_397887


namespace tangent_line_intersection_x_l397_397147

noncomputable def center1 : ℝ × ℝ := (0, 0)
noncomputable def radius1 : ℝ := 3
noncomputable def center2 : ℝ × ℝ := (17, 0)
noncomputable def radius2 : ℝ := 8

theorem tangent_line_intersection_x:
  ∃ x : ℝ, x > 0 ∧ (tangent_line_to_circle1_and_circle2 ((0, 0), 3) ((17, 0), 8)).intersects_x_axis_at (x, 0)
  → x = 51 / 11 :=
by
  sorry

end tangent_line_intersection_x_l397_397147


namespace exists_divisible_by_digit_sum_l397_397431

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_divisible_by_digit_sum (a b : ℕ) (h1 : a ≤ b) (h2 : b ≤ 2005) (h3 : b - a = 17) :
  ∃ n ∈ (Icc a b), n % sum_of_digits n = 0 :=
  sorry

end exists_divisible_by_digit_sum_l397_397431


namespace smallest_angle_product_l397_397600

-- Define an isosceles triangle with angle at B being the smallest angle
def isosceles_triangle (α : ℝ) : Prop :=
  α < 90 ∧ α = 180 / 7

-- Proof that the smallest angle multiplied by 6006 is 154440
theorem smallest_angle_product : 
  isosceles_triangle α → (180 / 7) * 6006 = 154440 :=
by
  intros
  sorry

end smallest_angle_product_l397_397600


namespace teacher_is_52_l397_397454

-- Define the given conditions
def total_age_students (n : ℕ) (average_age : ℕ) : ℕ := 
  n * average_age

def total_age_with_teacher (n : ℕ) (new_average_age : ℕ) : ℕ := 
  (n + 1) * new_average_age

-- Given the number of students and their average age
def num_students := 25
def average_age_students := 26
def new_average_age := 27

-- Total age of the 25 students
def S := total_age_students num_students average_age_students

-- Total age of 25 students and the teacher
def S_T := total_age_with_teacher num_students new_average_age

-- The teacher's age
def teacher_age := S_T - S

theorem teacher_is_52 : teacher_age = 52 := by 
  unfold teacher_age S_T S total_age_with_teacher total_age_students 
  rw [nat.add_comm, nat.mul_add, nat.add_comm]
  sorry

end teacher_is_52_l397_397454


namespace Cassini_l397_397953

-- Define the Fibonacci sequence
def Fibonacci : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 2) => Fibonacci (n + 1) + Fibonacci n

-- State Cassini's Identity theorem
theorem Cassini (n : ℕ) : Fibonacci (n + 1) * Fibonacci (n - 1) - (Fibonacci n) ^ 2 = (-1) ^ n := 
by sorry

end Cassini_l397_397953


namespace set_intersection_complement_l397_397716

universe u
variable (U : Set.{u}) (A : Set.{u}) (B : Set.{u})

theorem set_intersection_complement :
  U = {1, 2, 3, 4, 5} → A = {1, 2, 3} → B = {2, 5} → 
  (A ∩ (U \ B)) = {1, 3} :=
by
  intros hU hA hB
  sorry

end set_intersection_complement_l397_397716


namespace find_k_l397_397242

def vector (α : Type) := (α × α)
def a : vector ℝ := (1, 3)
def b (k : ℝ) : vector ℝ := (-2, k)
def add (v1 v2 : vector ℝ) : vector ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def smul (c : ℝ) (v : vector ℝ) : vector ℝ := (c * v.1, c * v.2)
def cross_product (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.2 - v1.2 * v2.1

theorem find_k (k : ℝ) (h : cross_product (add a (smul 2 (b k)))
                                          (add (smul 3 a) (smul (-1) (b k))) = 0) : k = -6 :=
sorry

end find_k_l397_397242


namespace glee_club_female_members_l397_397344

theorem glee_club_female_members (m f : ℕ) 
  (h1 : f = 2 * m) 
  (h2 : m + f = 18) : 
  f = 12 :=
by
  sorry

end glee_club_female_members_l397_397344


namespace totalExerciseTime_l397_397937

-- Define the conditions
def caloriesBurnedRunningPerMinute := 10
def caloriesBurnedWalkingPerMinute := 4
def totalCaloriesBurned := 450
def runningTime := 35

-- Define the problem as a theorem to be proven
theorem totalExerciseTime :
  ((runningTime * caloriesBurnedRunningPerMinute) + 
  ((totalCaloriesBurned - runningTime * caloriesBurnedRunningPerMinute) / caloriesBurnedWalkingPerMinute)) = 60 := 
sorry

end totalExerciseTime_l397_397937


namespace evaluate_expression_l397_397217

variable (x y : ℝ)

theorem evaluate_expression :
  (1 + x^2 + y^3) * (1 - x^3 - y^3) = 1 + x^2 - x^3 - y^3 - x^5 - x^2 * y^3 - x^3 * y^3 - y^6 :=
by
  sorry

end evaluate_expression_l397_397217


namespace winning_percentage_l397_397351

theorem winning_percentage (total_votes majority : ℕ) (h1 : total_votes = 455) (h2 : majority = 182) :
  ∃ P : ℕ, P = 70 ∧ (P / 100) * total_votes - ((100 - P) / 100) * total_votes = majority := 
sorry

end winning_percentage_l397_397351


namespace sample_older_employees_count_l397_397149

-- Definitions of known quantities
def N := 400
def N_older := 160
def N_no_older := 240
def n := 50

-- The proof statement showing that the number of employees older than 45 in the sample equals 20
theorem sample_older_employees_count : 
  let proportion_older := (N_older:ℝ) / (N:ℝ)
  let n_older := proportion_older * (n:ℝ)
  n_older = 20 := by
  sorry

end sample_older_employees_count_l397_397149


namespace x_axis_intercept_l397_397476

theorem x_axis_intercept (x y : ℝ) : x + 2 * y + 1 = 0 → y = 0 → x = -1 :=
by
  intros h_line h_y
  -- Substitute y = 0 into the equation x + 2 * y + 1 = 0
  have h_eq : x + 1 = 0 := by
    rw [h_y] at h_line
    linarith
  -- Solving for x, we get x = -1
  linear_combination h_eq, 1
  -- Sorry this part as it follows the solution directly
  sorry

end x_axis_intercept_l397_397476


namespace no_solution_for_s_l397_397232

theorem no_solution_for_s : ∀ s : ℝ,
  (s^2 - 6 * s + 8) / (s^2 - 9 * s + 20) ≠ (s^2 - 3 * s - 18) / (s^2 - 2 * s - 15) :=
by
  intros s
  sorry

end no_solution_for_s_l397_397232


namespace funnel_height_l397_397915

def volume_of_cone (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

theorem funnel_height (h : ℝ) (r : ℝ := 3) (V : ℝ := 120) :
  volume_of_cone r h = V → h = 40 / Real.pi :=
by
  sorry

end funnel_height_l397_397915


namespace problem_statement_l397_397002

theorem problem_statement (x : ℤ) (h : x = -2023) : 
  |(|x| - x) - |x + 7| | - x - 7 = 4046 :=
by
  sorry

end problem_statement_l397_397002


namespace quadrilateral_area_l397_397340

theorem quadrilateral_area (a b e s : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < e)
  (h4 : e < s) (h5 : (s/√2) ≤ e)
  (hconvex : is_convex_quadrilateral a b e s): 
  T = (1 / 4) * (s^2 - e^2) :=
by
  sorry

end quadrilateral_area_l397_397340


namespace briefcase_pen_ratio_l397_397801

def ratio_of_briefcase_to_pen (pen_price briefcase_multiple total_cost : ℕ) : ℕ :=
  let briefcase_price := pen_price * briefcase_multiple in
  if pen_price + briefcase_price = total_cost then briefcase_multiple else 0

theorem briefcase_pen_ratio : ratio_of_briefcase_to_pen 4 5 24 = 5 :=
by
  sorry

end briefcase_pen_ratio_l397_397801


namespace mean_of_three_l397_397049

theorem mean_of_three (a b c : ℝ) (h : (a + b + c + 105) / 4 = 92) : (a + b + c) / 3 = 87.7 :=
by
  sorry

end mean_of_three_l397_397049


namespace rooms_needed_l397_397993

-- Defining all necessary conditions
constant fans : ℕ := 100
constant teams : ℕ := 3
constant max_people_per_room : ℕ := 3
constant groups : ℕ := 6

-- Proposition that the number of rooms required is exactly 37.
theorem rooms_needed (h1 : ∀ i, 1 ≤ i ∧ i ≤ groups) 
  (h2 : groups = 2 * teams) 
  (h3 : teams = 3)
  (h4 : max_people_per_room = 3)
  (h5 : ∃ i, 0 ≤ i ∧ ∑ i in finset.range groups, i = fans):
  37 = (∃ (n : ℕ), ∃ (i : ℕ), n = ∑ i in finset.range groups, nat.ceil (i / max_people_per_room)):
sorry

end rooms_needed_l397_397993


namespace three_number_product_l397_397127

theorem three_number_product
  (x y z : ℝ)
  (h1 : x + y = 18)
  (h2 : x ^ 2 + y ^ 2 = 220)
  (h3 : z = x - y) :
  x * y * z = 104 * Real.sqrt 29 :=
sorry

end three_number_product_l397_397127


namespace equally_likely_events_A_and_B_l397_397369

-- Given conditions definitions
def number_of_balls (N : ℕ) := N
def main_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 10
def additional_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 8
def lucky_sum_main (s : finset ℕ) := s.sum = 63
def lucky_sum_additional (s : finset ℕ) := s.sum = 44

-- Prove the events A and B are equally likely if and only if N = 18
theorem equally_likely_events_A_and_B (N : ℕ) :
  (∃ s ∈ main_draw_balls N, lucky_sum_main s) ∧ (∃ t ∈ additional_draw_balls N, lucky_sum_additional t)
  → (N = 18) :=
sorry

end equally_likely_events_A_and_B_l397_397369


namespace repeating_decimal_sum_l397_397484

theorem repeating_decimal_sum (c d : ℕ) (h : (4 : ℚ) / 13 = 0.cdcdcd...) : c + d = 3 := 
begin
  sorry
end

end repeating_decimal_sum_l397_397484


namespace angle_of_pyramid_l397_397169

variables {m n : ℝ} (h_nonneg : m ≥ 0 ∧ n ≥ 0)

noncomputable def angle_between_height_and_lateral_face : ℝ :=
  let sine_angle := (real.cbrt (m + n) - real.cbrt m) / (real.cbrt (m + n) + real.cbrt m) in
  real.arcsin sine_angle

theorem angle_of_pyramid
  (m n : ℝ) (h_nonneg : m ≥ 0 ∧ n ≥ 0) :
  ∃ θ : ℝ, θ = angle_between_height_and_lateral_face h_nonneg ∧ θ = real.arcsin ((real.cbrt (m + n) - real.cbrt m) / (real.cbrt (m + n) + real.cbrt m)) :=
begin
  use angle_between_height_and_lateral_face h_nonneg,
  split,
  { refl },
  { refl },
end

end angle_of_pyramid_l397_397169


namespace average_speed_is_65_l397_397077

-- Definitions based on the problem's conditions
def speed_first_hour : ℝ := 100 -- 100 km in the first hour
def speed_second_hour : ℝ := 30 -- 30 km in the second hour
def total_distance : ℝ := speed_first_hour + speed_second_hour -- total distance
def total_time : ℝ := 2 -- total time in hours (1 hour + 1 hour)

-- Problem: prove that the average speed is 65 km/h
theorem average_speed_is_65 : (total_distance / total_time) = 65 := by
  sorry

end average_speed_is_65_l397_397077


namespace sum_d_e_f_l397_397832

-- Define the variables
variables (d e f : ℤ)

-- Given conditions
def condition1 : Prop := ∀ x : ℤ, x^2 + 18 * x + 77 = (x + d) * (x + e)
def condition2 : Prop := ∀ x : ℤ, x^2 - 19 * x + 88 = (x - e) * (x - f)

-- Prove the statement
theorem sum_d_e_f : condition1 d e → condition2 e f → d + e + f = 26 :=
by
  intros h1 h2
  -- Proof omitted
  sorry

end sum_d_e_f_l397_397832


namespace sixth_number_is_52_l397_397578

def systematic_sampling : Type := List Nat

def sample_from_population (population_size : Nat) (segments : Nat) (start : Nat) : systematic_sampling :=
  List.map (fun k => 
    (start + k * (population_size / segments)) % population_size) (List.range segments)

theorem sixth_number_is_52 (i : Nat) (H : i = 7) :  
  (sample_from_population 100 10 i).nth 5 = some 52 :=
by {
  -- We have to reason with the indices and segmentation rules
  -- List.range will give us the indices [0, 1, ..., 9]
  sorry
}

end sixth_number_is_52_l397_397578


namespace BC_length_l397_397829

theorem BC_length (E A B C D : Point) 
    (h1 : convex_quadrilateral A B C D) 
    (h2 : diagonal_intersection A B C D E)
    (h3 : area_triangle A B E = 1)
    (h4 : area_triangle D C E = 1)
    (h5 : area_quadrilateral A B C D ≤ 4)
    (h6 : distance A D = 3) : 
    distance B C = 3 :=
by
  sorry

end BC_length_l397_397829


namespace sum_multiples_up_to_3000_l397_397591

theorem sum_multiples_up_to_3000 : 
  let LCM := 140
  let multiples := (finset.range 3001).filter (λ x => x % LCM = 0)
  let sum_of_multiples := multiples.sum
  sum_of_multiples = 32340 := by {
  sorry
}

end sum_multiples_up_to_3000_l397_397591


namespace semicircle_area_in_quarter_circle_l397_397411

theorem semicircle_area_in_quarter_circle (r : ℝ) (A : ℝ) (π : ℝ) (one : ℝ) :
    r = 1 / (Real.sqrt (2) + 1) →
    A = π * r^2 →
    120 * A / π = 20 :=
sorry

end semicircle_area_in_quarter_circle_l397_397411


namespace basketball_tournament_l397_397092

theorem basketball_tournament (x : ℕ) 
  (h1 : ∀ n, ((n * (n - 1)) / 2) = 28 -> n = x) 
  (h2 : (x * (x - 1)) / 2 = 28) : 
  (1 / 2 : ℚ) * x * (x - 1) = 28 :=
by 
  sorry

end basketball_tournament_l397_397092


namespace max_value_of_expression_l397_397767

theorem max_value_of_expression (a b c : ℝ) 
  (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_choices : a ∈ {2, 3, 6} ∧ b ∈ {2, 3, 6} ∧ c ∈ {2, 3, 6})
  (h_value : a / (b / c) = 9) : 
  a = 3 := 
sorry

end max_value_of_expression_l397_397767


namespace num_bags_of_cookies_l397_397526

theorem num_bags_of_cookies (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 703) (h2 : cookies_per_bag = 19) : total_cookies / cookies_per_bag = 37 :=
by
  sorry

end num_bags_of_cookies_l397_397526


namespace smallest_of_four_l397_397934

theorem smallest_of_four (a b c d : ℝ) (h1 : a = -3) (h2 : b = -sqrt 2) (h3 : c = -1) (h4 : d = 0) :
  min a (min b (min c d)) = -3 := 
sorry

end smallest_of_four_l397_397934


namespace unique_solution_triple_l397_397641

def satisfies_system (x y z : ℝ) :=
  x^3 = 3 * x - 12 * y + 50 ∧
  y^3 = 12 * y + 3 * z - 2 ∧
  z^3 = 27 * z + 27 * x

theorem unique_solution_triple (x y z : ℝ) :
  satisfies_system x y z ↔ (x = 2 ∧ y = 4 ∧ z = 6) :=
by sorry

end unique_solution_triple_l397_397641


namespace perpendicular_line_plane_implies_perpendicular_lines_l397_397250

variables {Line Plane : Type}
variables (m n : Line) (α : Plane)

-- Assume inclusion of lines in planes, parallelism, and perpendicularity properties.
variables (parallel : Line → Plane → Prop) (perpendicular : Line → Plane → Prop) (subset : Line → Plane → Prop) (perpendicular_lines : Line → Line → Prop)

-- Given definitions based on the conditions
variable (is_perpendicular : perpendicular m α)
variable (is_subset : subset n α)

-- Prove that m is perpendicular to n
theorem perpendicular_line_plane_implies_perpendicular_lines
  (h1 : perpendicular m α)
  (h2 : subset n α) :
  perpendicular_lines m n :=
sorry

end perpendicular_line_plane_implies_perpendicular_lines_l397_397250


namespace poker_cards_count_l397_397919

theorem poker_cards_count (total_cards kept_away : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : kept_away = 7) : 
  total_cards - kept_away = 45 :=
by 
  sorry

end poker_cards_count_l397_397919


namespace max_rectangles_in_triangle_l397_397021

theorem max_rectangles_in_triangle : 
  ∀ (triangle : Type) (grid : Type), 
  (right_angled triangle) ∧ (leg_lengths triangle = (6, 6)) ∧ (outlined_grid grid triangle) 
  → (max_rectangles grid triangle = 126) :=
sorry

end max_rectangles_in_triangle_l397_397021


namespace hotel_room_allocation_l397_397974

theorem hotel_room_allocation :
  ∀ (fans : ℕ) (num_groups : ℕ) (room_capacity : ℕ), 
  fans = 100 → num_groups = 6 → room_capacity = 3 →
  (∀ (group_fans : ℕ) (groups : fin num_groups), group_fans ≤ fans →
   (∃ (total_rooms: ℕ), total_rooms = 37 ∧
    ∀ (group_fan_count : fin num_groups → ℕ), 
      (∀ g, group_fan_count g ≤ fans / num_groups + if (fans % num_groups > 0) then 1 else 0) →
      ∃ (rooms : fin num_groups → ℕ), 
        sum (λ g, rooms g) = total_rooms ∧ 
        ∀ g, group_fan_count g ≤ rooms g * room_capacity)) :=
by {
  intros,
  sorry
}

end hotel_room_allocation_l397_397974


namespace largest_percentage_increase_l397_397602

theorem largest_percentage_increase :
  let n2003 := 50
  let n2004 := 58
  let n2005 := 65
  let n2006 := 75
  let n2007 := 80
  let n2008 := 100
  let perc_increase (n1 n2 : ℕ) := (n2 - n1 : ℚ) / n1 * 100
  perc_increase n2003 n2004 < perc_increase n2007 n2008 ∧
  perc_increase n2004 n2005 < perc_increase n2007 n2008 ∧
  perc_increase n2005 n2006 < perc_increase n2007 n2008 ∧
  perc_increase n2006 n2007 < perc_increase n2007 n2008 :=
by {
  let n2003 := 50
  let n2004 := 58
  let n2005 := 65
  let n2006 := 75
  let n2007 := 80
  let n2008 := 100
  let perc_increase (n1 n2 : ℕ) := (n2 - n1 : ℚ) / n1 * 100
  have h1 : perc_increase n2003 n2004 = 16 := 
    -- Calculation proof skipped
    sorry,
  have h2 : perc_increase n2004 n2005 = 100 * (7 / 58) := 
    -- Calculation proof skipped
    sorry,
  have h3 : perc_increase n2005 n2006 = 100 * (10 / 65) := 
    -- Calculation proof skipped
    sorry,
  have h4 : perc_increase n2006 n2007 = 100 * (5 / 75) := 
    -- Calculation proof skipped
    sorry,
  have h5 : perc_increase n2007 n2008 = 25 := 
    -- Calculation proof skipped
    sorry,
  exact And.intro 
    (lt_of_le_of_lt sorry sorry)  -- place specific inequalities here
    (And.intro 
      (lt_of_le_of_lt sorry sorry) 
      (And.intro 
        (lt_of_le_of_lt sorry sorry)
        (lt_of_le_of_lt sorry sorry)))
}

end largest_percentage_increase_l397_397602


namespace value_of_f_l397_397665

noncomputable def f (α : ℝ) : ℝ :=
  (Real.cos (Real.pi / 2 + α) * Real.cos (Real.pi - α)) / Real.sin (Real.pi + α)

theorem value_of_f (α : ℝ) (h1 : 3*Real.pi/2 < α ∧ α < 2*Real.pi) (h2 : Real.cos (α - 3*Real.pi/2) = 1/5) :
  f α = 2*Real.sqrt(6)/5 :=
by
  sorry

end value_of_f_l397_397665


namespace german_russian_students_l397_397630

open Nat

theorem german_russian_students (G R : ℕ) (G_cap_R : ℕ) 
  (h_total : 1500 = G + R - G_cap_R)
  (hG_lb : 1125 ≤ G) (hG_ub : G ≤ 1275)
  (hR_lb : 375 ≤ R) (hR_ub : R ≤ 525) :
  300 = (max (G_cap_R) - min (G_cap_R)) :=
by
  -- Proof would go here
  sorry

end german_russian_students_l397_397630


namespace tan_alpha_eq_7_over_5_l397_397313

theorem tan_alpha_eq_7_over_5
  (α : ℝ)
  (h : Real.tan (α - π / 4) = 1 / 6) :
  Real.tan α = 7 / 5 :=
by
  sorry

end tan_alpha_eq_7_over_5_l397_397313


namespace second_largest_values_correct_l397_397571

def num_possible_values_second_largest 
(list_of_ints : List ℕ) 
(h_length : list_of_ints.length = 5) 
(h_positive : ∀ n ∈ list_of_ints, n > 0) 
(h_mean : (list_of_ints.sum / 5) = 15)
(h_range : list_of_ints.maximum - list_of_ints.minimum = 24)
(h_mode_median : list_of_ints.sorted.nth 2 = 10 ∧ (list_of_ints.count 10 > 1)) : 
ℕ :=
  10 -- This represents the final answer that should be proven: the number of different values possible for the second largest element is 10

theorem second_largest_values_correct :
  num_possible_values_second_largest
    [a,b,c,d,e]
    (by simp)
    (by simp)
    (by linarith)
    (by linarith)
    (by simp) = 10 := 
sorry

end second_largest_values_correct_l397_397571


namespace rooms_needed_l397_397987

-- Defining all necessary conditions
constant fans : ℕ := 100
constant teams : ℕ := 3
constant max_people_per_room : ℕ := 3
constant groups : ℕ := 6

-- Proposition that the number of rooms required is exactly 37.
theorem rooms_needed (h1 : ∀ i, 1 ≤ i ∧ i ≤ groups) 
  (h2 : groups = 2 * teams) 
  (h3 : teams = 3)
  (h4 : max_people_per_room = 3)
  (h5 : ∃ i, 0 ≤ i ∧ ∑ i in finset.range groups, i = fans):
  37 = (∃ (n : ℕ), ∃ (i : ℕ), n = ∑ i in finset.range groups, nat.ceil (i / max_people_per_room)):
sorry

end rooms_needed_l397_397987


namespace simplify_expression_l397_397813

-- Define the necessary components for the question
noncomputable def expression (x : ℝ) : ℝ := 
  (sqrt (x - 2 * sqrt 2) / sqrt (x^2 - 4 * x * sqrt 2 + 8)) -
  (sqrt (x + 2 * sqrt 2) / sqrt (x^2 + 4 * x * sqrt 2 + 8))

-- Define the proof statement
theorem simplify_expression : expression 3 = 2 :=
  sorry

end simplify_expression_l397_397813


namespace lucky_sum_equal_prob_l397_397359

theorem lucky_sum_equal_prob (N : ℕ) (S63_main_draw : Finset ℕ) (S44_additional_draw : Finset ℕ)
  (hN_pos : 0 < N)
  (hS63 : S63_main_draw.card = 10)
  (hSum63 : S63_main_draw.sum id = 63)
  (hS44 : S44_additional_draw.card = 8)
  (hSum44 : S44_additional_draw.sum id = 44)
  (hS_distinct : ∀ (n ∈ S63_main_draw ∪ S44_additional_draw), n ∈ Finset.range (N + 1)) :
  N = 18 :=
by
  sorry

end lucky_sum_equal_prob_l397_397359


namespace three_tenths_of_number_l397_397531

theorem three_tenths_of_number (x : ℝ) (h : (1/3) * (1/4) * x = 18) : (3/10) * x = 64.8 :=
sorry

end three_tenths_of_number_l397_397531


namespace solve_for_x_l397_397964

theorem solve_for_x (x : ℝ) (h : (x * (x ^ (5 / 2))) ^ (1 / 4) = 4) : 
  x = 4 ^ (8 / 7) :=
sorry

end solve_for_x_l397_397964


namespace coloring_methods_count_l397_397835

theorem coloring_methods_count :
  ∃ (f : ℤ × ℤ → bool),
  (∀ i j, f i j = tt ∨ f i j = ff) ∧  -- f maps each cell to either color or not
  (∃ r1 r2, (r1 ≠ r2) ∧ 
         (f (r1, 0) = tt ∨ f (r1, 1) = tt ∨ f (r1, 2) = tt) ∧
         (f (r2, 0) = tt ∨ f (r2, 1) = tt ∨ f (r2, 2) = tt)) ∧ -- Each row has at least one colored
  (∃ c1 c2, (c1 ≠ c2) ∧ 
         (f (0, c1) = tt ∨ f (1, c1) = tt) ∧
         (f (0, c2) = tt ∨ f (1, c2) = tt)) ∧ -- Each column has at least one colored
  (let num_grays := (finset.univ.sum (λ (ij : ℤ × ℤ), if f ij then 1 else 0)) in
     num_grays = 4) ∧  -- Exactly 4 cells colored
  (count_distinct_colorings f = 5) -- Proving the counted distinct colorings equals to 5
:= {
  -- The definition of count_distinct_colorings and the proof will come here
  sorry
}

end coloring_methods_count_l397_397835


namespace hotel_room_allocation_l397_397971

theorem hotel_room_allocation :
  ∀ (fans : ℕ) (num_groups : ℕ) (room_capacity : ℕ), 
  fans = 100 → num_groups = 6 → room_capacity = 3 →
  (∀ (group_fans : ℕ) (groups : fin num_groups), group_fans ≤ fans →
   (∃ (total_rooms: ℕ), total_rooms = 37 ∧
    ∀ (group_fan_count : fin num_groups → ℕ), 
      (∀ g, group_fan_count g ≤ fans / num_groups + if (fans % num_groups > 0) then 1 else 0) →
      ∃ (rooms : fin num_groups → ℕ), 
        sum (λ g, rooms g) = total_rooms ∧ 
        ∀ g, group_fan_count g ≤ rooms g * room_capacity)) :=
by {
  intros,
  sorry
}

end hotel_room_allocation_l397_397971


namespace ratio_octahedron_to_cube_volume_l397_397446

theorem ratio_octahedron_to_cube_volume (x : ℝ) (h : x > 0) :
  let volume_cube := (2 * x) ^ 3,
      s := x * Real.sqrt 2,
      volume_octahedron := (s ^ 3 * Real.sqrt 2) / 3
  in (volume_octahedron / volume_cube) = 1 / 6 :=
by
  sorry

end ratio_octahedron_to_cube_volume_l397_397446


namespace reaction_rate_reduction_l397_397133

theorem reaction_rate_reduction (k : ℝ) (NH3 Br2 NH3_new : ℝ) (v1 v2 : ℝ):
  (v1 = k * NH3^8 * Br2) →
  (v2 = k * NH3_new^8 * Br2) →
  (v2 / v1 = 60) →
  NH3_new = 60 ^ (1 / 8) :=
by
  intro hv1 hv2 hratio
  sorry

end reaction_rate_reduction_l397_397133


namespace average_of_middle_three_l397_397457

-- Let A be the set of five different positive whole numbers
def avg (lst : List ℕ) : ℚ := (lst.foldl (· + ·) 0 : ℚ) / lst.length
def maximize_diff (lst : List ℕ) : ℕ := lst.maximum' - lst.minimum'

theorem average_of_middle_three 
  (A : List ℕ) 
  (h_diff : maximize_diff A) 
  (h_avg : avg A = 5) 
  (h_len : A.length = 5) 
  (h_distinct : A.nodup) : 
  avg (A.erase_max.erase_min) = 3 := 
sorry

end average_of_middle_three_l397_397457


namespace smallest_n_value_l397_397582

theorem smallest_n_value : ∃ (n : ℕ), (∀ m : ℕ, (m < n → ¬(∃ x : ℕ, 1.04 * x = m))) ∧ ∃ x : ℕ, 1.04 * x = n :=
by
  sorry

end smallest_n_value_l397_397582


namespace find_k_l397_397233

noncomputable def f (x : ℝ) : ℝ := Real.cot (x / 3) - Real.cot (x / 2)

theorem find_k :
  (∀ x, f x = (Real.sin (1/6 * x)) / ((Real.sin (x / 3)) * (Real.sin (x / 2)))) →
  k = 1/6 :=
  sorry

end find_k_l397_397233


namespace find_k_l397_397241

def vector (α : Type) := (α × α)
def a : vector ℝ := (1, 3)
def b (k : ℝ) : vector ℝ := (-2, k)
def add (v1 v2 : vector ℝ) : vector ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def smul (c : ℝ) (v : vector ℝ) : vector ℝ := (c * v.1, c * v.2)
def cross_product (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.2 - v1.2 * v2.1

theorem find_k (k : ℝ) (h : cross_product (add a (smul 2 (b k)))
                                          (add (smul 3 a) (smul (-1) (b k))) = 0) : k = -6 :=
sorry

end find_k_l397_397241


namespace average_of_middle_three_is_three_l397_397460

theorem average_of_middle_three_is_three :
  ∃ (a b c d e : ℕ), 
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (d ≠ e) ∧
    (a + b + c + d + e = 25) ∧
    (∃ (min max : ℕ), min = min a b c d e ∧ max = max a b c d e ∧ (max - min) = 14) ∧
    (a ≠ min a b c d e ∧ a ≠ max a b c d e) ∧
    (b ≠ min a b c d e ∧ b ≠ max a b c d e) ∧
    (c ≠ min a b c d e ∧ c ≠ max a b c d e) ∧
    (d ≠ min a b c d e ∧ d ≠ max a b c d e) ∧
    (e ≠ min a b c d e ∧ e ≠ max a b c d e) ∧
    (a + b + c + d + e = 25) → 
    (a ≠ min a b c d e ∧ a ≠ max a b c d e) ∧
    (b ≠ min a b c d e ∧ b ≠ max a b c d e) ∧
    (c ≠ min a b c d e ∧ c ≠ max a b c d e) →  
  ((a + b + c) / 3 = 3) :=
by
  sorry

end average_of_middle_three_is_three_l397_397460


namespace solve_system_of_equations_l397_397044

theorem solve_system_of_equations (x y z t : ℤ) :
  (3 * x - 2 * y + 4 * z + 2 * t = 19) ∧ (5 * x + 6 * y - 2 * z + 3 * t = 23) →
  (x = 16 * z - 18 * y - 11) ∧ (t = 28 * y - 26 * z + 26) :=
by {
  sorry
}

end solve_system_of_equations_l397_397044


namespace range_of_a_l397_397284

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, -1 ≤ x → f a x ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by sorry

end range_of_a_l397_397284


namespace geometric_problem_l397_397501

/-- Three circles ω1, ω2, and ω3 with radius 28 touch each other externally. 
    On each circle, points P1, P2, and P3 are chosen such that P1P2 = P2P3 = P3P1 
    and each P1P2, P2P3, and P3P1 touches ω2, ω3, and ω1 respectively. 
    The area of triangle P1P2P3 is expressed as sqrt(a) + sqrt(b). 
    We prove that a + b equals 627. -/
theorem geometric_problem (r : ℝ := 28) (a b : ℕ) :
  let A := (sqrt a + sqrt b) in (a + b = 627) :=
by
  sorry

end geometric_problem_l397_397501


namespace arithmetic_identity_l397_397609

theorem arithmetic_identity : Real.sqrt 16 + ((1/2) ^ (-2:ℤ)) = 8 := 
by 
  sorry

end arithmetic_identity_l397_397609


namespace age_difference_constant_l397_397153

theorem age_difference_constant (seokjin_age_mother_age_diff : ∀ (t : ℕ), 33 - 7 = 26) : 
  ∀ (n : ℕ), 33 + n - (7 + n) = 26 := 
by
  sorry

end age_difference_constant_l397_397153


namespace intersection_equiv_l397_397715

-- Define the sets M and N based on the given conditions
def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x | x < 0 ∨ x > 2}

-- The main proof statement
theorem intersection_equiv : M ∩ N = {-1, 3} :=
by
  sorry -- proof goes here

end intersection_equiv_l397_397715


namespace f_11_f_2021_eq_neg_one_l397_397520

def f (n : ℕ) : ℚ := sorry

axiom recurrence_relation (n : ℕ) : f (n + 3) = (f n - 1) / (f n + 1)
axiom f1_ne_zero : f 1 ≠ 0
axiom f1_ne_one : f 1 ≠ 1
axiom f1_ne_neg_one : f 1 ≠ -1

theorem f_11_f_2021_eq_neg_one : f 11 * f 2021 = -1 := 
by
  sorry

end f_11_f_2021_eq_neg_one_l397_397520


namespace quadrant_of_half_angle_in_second_quadrant_l397_397275

theorem quadrant_of_half_angle_in_second_quadrant (θ : ℝ) (h : π / 2 < θ ∧ θ < π) :
  (0 < θ / 2 ∧ θ / 2 < π / 2) ∨ (π < θ / 2 ∧ θ / 2 < 3 * π / 2) :=
by
  sorry

end quadrant_of_half_angle_in_second_quadrant_l397_397275


namespace area_of_triangle_ABC_l397_397251

-- Define the circle and its properties
def center : ℝ × ℝ := (1, 0)
def radius : ℝ := 1
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the parametric line and its conversion
def param_line (t : ℝ) : ℝ × ℝ := (-1 + (real.sqrt 2 / 2) * t, 3 - (real.sqrt 2 / 2) * t)

-- Define the standard line equation
def standard_line (x y : ℝ) : Prop := x + y = 2

-- The distance from the center to the line
def distance_from_center_to_line := real.sqrt 2 / 2

-- The length of the chord AB
def chord_length := real.sqrt 2

-- The area of triangle ABC
def triangle_area := 1 / 2

theorem area_of_triangle_ABC : 
  ∃ A B : ℝ × ℝ, 
  circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧ 
  standard_line A.1 A.2 ∧ standard_line B.1 B.2 ∧
  dist center (A+B)/2 = distance_from_center_to_line ∧ -- midpoint distance from center
  dist A B = chord_length ∧ 
  (1 / 2) * chord_length * distance_from_center_to_line = triangle_area :=
sorry

end area_of_triangle_ABC_l397_397251


namespace lengths_of_two_medians_do_not_determine_unique_shape_l397_397423

-- Define the data types and properties for the given problem
structure Triangle :=
(sides : ℝ × ℝ × ℝ)
(medians : ℝ × ℝ × ℝ)
(altitude_base : ℝ × ℝ)
(angles : ℝ × ℝ)

-- Establish the equivalence condition that the lengths of two medians do not uniquely determine the shape of a triangle
def does_two_medians_determines_triangle_shape : Triangle → Prop :=
λ t, ∀ m₁ m₂, (m₁, m₂) ∈ {v : ℝ × ℝ | ∃ s₁ s₂ s₃ : ℝ, v = (median_length s₁ s₂ s₃ 1, median_length s₁ s₂ s₃ 2)}
→ ∃! t', t'.medians = (m₁, m₂)

-- Theorem Statement: The lengths of two medians do not determine the unique shape of a triangle
theorem lengths_of_two_medians_do_not_determine_unique_shape (t : Triangle) :
  ¬ does_two_medians_determines_triangle_shape t :=
sorry

end lengths_of_two_medians_do_not_determine_unique_shape_l397_397423


namespace Alex_Hours_Upside_Down_Per_Month_l397_397173

-- Define constants and variables based on the conditions
def AlexCurrentHeight : ℝ := 48
def AlexRequiredHeight : ℝ := 54
def NormalGrowthPerMonth : ℝ := 1 / 3
def UpsideDownGrowthPerHour : ℝ := 1 / 12
def MonthsInYear : ℕ := 12

-- Compute total growth needed and additional required growth terms
def TotalGrowthNeeded : ℝ := AlexRequiredHeight - AlexCurrentHeight
def NormalGrowthInYear : ℝ := NormalGrowthPerMonth * MonthsInYear
def AdditionalGrowthNeeded : ℝ := TotalGrowthNeeded - NormalGrowthInYear
def TotalUpsideDownHours : ℝ := AdditionalGrowthNeeded * 12
def UpsideDownHoursPerMonth : ℝ := TotalUpsideDownHours / MonthsInYear

-- The statement to prove
theorem Alex_Hours_Upside_Down_Per_Month : UpsideDownHoursPerMonth = 2 := by
  sorry

end Alex_Hours_Upside_Down_Per_Month_l397_397173


namespace initial_percentage_acid_l397_397916

theorem initial_percentage_acid (P : ℝ) (h1 : 27 * P / 100 = 18 * 60 / 100) : P = 40 :=
sorry

end initial_percentage_acid_l397_397916


namespace domain_of_c_x_l397_397960

theorem domain_of_c_x (k : ℝ) :
  (∀ x : ℝ, -5 * x ^ 2 + 3 * x + k ≠ 0) ↔ k < -9 / 20 := 
sorry

end domain_of_c_x_l397_397960


namespace sequence_sum_l397_397260

theorem sequence_sum :
  (∀ (a : ℕ → ℝ), (a 1 = 2) ∧ (∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n + 2) →
  (b : ℕ → ℝ) → (∀ n : ℕ, b n = Real.log (a n + 1) / Real.log 3) →
  ∑ k in Finset.range (100), b (k + 1) = 5050) :=
by
  sorry

end sequence_sum_l397_397260


namespace kombucha_bottles_after_refund_l397_397725

noncomputable def bottles_per_month : ℕ := 15
noncomputable def cost_per_bottle : ℝ := 3.0
noncomputable def refund_per_bottle : ℝ := 0.10
noncomputable def months_in_year : ℕ := 12

theorem kombucha_bottles_after_refund :
  let bottles_per_year := bottles_per_month * months_in_year
  let total_refund := bottles_per_year * refund_per_bottle
  let bottles_bought_with_refund := total_refund / cost_per_bottle
  bottles_bought_with_refund = 6 := sorry

end kombucha_bottles_after_refund_l397_397725


namespace equilateral_BHD_l397_397216

-- Define the vertices of the equilateral triangles
variables (A B C D E H K : Point)

-- Assume the triangles are equilateral
def is_equilateral_triangle (T : Triangle) : Prop :=
  (T.side1 = T.side2) ∧ (T.side2 = T.side3)

variables (ABC : Triangle) (CDE : Triangle) (EHK : Triangle)
variables [is_equilateral_triangle ABC] [is_equilateral_triangle CDE] [is_equilateral_triangle EHK]

-- Assume vertices are traversed counterclockwise
def is_counterclockwise (T : Triangle) : Prop :=
  -- Your definition for counterclockwise, e.g., using determinants
  sorry 

variables [is_counterclockwise ABC] [is_counterclockwise CDE] [is_counterclockwise EHK]

-- Given vector equality condition
variables [eq (vector A D) (vector D K)]

-- The theorem we need to prove
theorem equilateral_BHD : is_equilateral (triangle B H D) :=
  sorry

end equilateral_BHD_l397_397216


namespace correct_propositions_l397_397174

def line (P: Type) := P → P → Prop  -- A line is a relation between points in a plane

variables (plane1 plane2: Type) -- Define two types representing two planes
variables (P1 P2: plane1) -- Points in plane1
variables (Q1 Q2: plane2) -- Points in plane2

axiom perpendicular_planes : ¬∃ l1 : line plane1, ∀ l2 : line plane2, ¬ (∀ p1 p2, l1 p1 p2 ∧ ∀ q1 q2, l2 q1 q2)

theorem correct_propositions : 3 = 3 := by
  sorry

end correct_propositions_l397_397174


namespace largest_k_divides_factorial_l397_397194

theorem largest_k_divides_factorial : 
  ∃ k : ℕ, 2012 = 2^2 * 503 ∧ (nat.find_greatest (λ k, 2012^k ∣ nat.factorial 2012) 2012! = k) ∧ k = 3 := 
by
  sorry

end largest_k_divides_factorial_l397_397194


namespace max_rectangles_in_triangle_l397_397022

theorem max_rectangles_in_triangle : 
  ∀ (triangle : Type) (grid : Type), 
  (right_angled triangle) ∧ (leg_lengths triangle = (6, 6)) ∧ (outlined_grid grid triangle) 
  → (max_rectangles grid triangle = 126) :=
sorry

end max_rectangles_in_triangle_l397_397022


namespace problem1_problem2_l397_397714

-- First proof problem
theorem problem1 (a : ℝ) :
  a = 1 →
  let A := {x : ℝ | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
  let B := {x : ℝ | - 1 / 2 < x ∧ x < 2}
  let CR_B := {x : ℝ | x ≤ - 1 / 2 ∨ x ≥ 2}
  (CR_B ∪ A) = {x : ℝ | x ≤ 1 ∨ x ≥ 2} :=
by
  intros h
  rw h
  sorry

-- Second proof problem
theorem problem2 (a : ℝ) :
  let A := {x : ℝ | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
  let B := {x : ℝ | - 1 / 2 < x ∧ x < 2}
  (A ⊆ B) ↔ -1 < a ∧ a ≤ 1 :=
by
  sorry

end problem1_problem2_l397_397714


namespace fifth_tangent_exists_l397_397087

noncomputable def construct_fifth_tangent 
  (F : Point) (d : Line) (e : Line) 
  (e1 e2 e3 e4 : Line) (direction : Vector) : Line :=
sorry

theorem fifth_tangent_exists (F : Point) (d : Line) (e : Line) 
  (e1 e2 e3 e4 : Line) (direction : Vector) : 
  ∃ t : Line, construct_fifth_tangent F d e e1 e2 e3 e4 direction = t :=
sorry

end fifth_tangent_exists_l397_397087


namespace intersection_value_l397_397768

open Real

theorem intersection_value
    (slope_l : ℝ)
    (x_m y_m : ℝ)
    (polar_eq : ℝ → ℝ)
    (intersect_eqn : ℝ → ℝ → Prop)
    (t1 t2 : ℝ) :
    slope_l = π/4 ∧ x_m = 2 ∧ y_m = 1 ∧
    (∀ θ, polar_eq θ = 4 * sqrt 2 * sin (θ + π / 4)) ∧
    (∀ x y, intersect_eqn x y ↔ x^2 + y^2 - 4 * y - 4 * y = 0) ∧
    t1 + t2 = sqrt 2 ∧ t1 * t2 = -7 →
    1 / (((t1^2 - sqrt 2 * t1 - 7)^0.5)) + 1 / (((t2^2 - sqrt 2 * t2 - 7)^0.5)) = sqrt 30 / 7 :=
sorry

end intersection_value_l397_397768


namespace termite_path_impossibility_l397_397741

-- Define the setup of the problem
structure Cube :=
(center_outer_face : ℕ)
(visited : ℕ → Bool)
(move_parallel : ℕ → ℕ)

-- Define the main question as a theorem
theorem termite_path_impossibility (start : Cube) : ¬ (∃ path : list ℕ, 
  ∀ i ∈ path, i ≠ start.center_outer_face ∧ 
  (∀ j ∈ path, j ≠ start.center_outer_face → start.move_parallel j) ∧ 
  list.nodup path ∧ path.head = start.center_outer_face ∧ path.reverse.head = 14) :=
sorry

end termite_path_impossibility_l397_397741


namespace grid_color_remainder_l397_397744

theorem grid_color_remainder :
  let n := (Nat.choose 25 4) * 3 * 8 + 1
  in n % 1000 = 601 :=
by
  let n := (Nat.choose 25 4) * 3 * 8 + 1
  have : n = 303601 := by
    rw [Nat.choose_eq_factorial_div]
    -- (Nat.factorial 25) / (Nat.factorial 4 * Nat.factorial 21) = 12650
    sorry
  show n % 1000 = 601
  rw [this]
  norm_num
  -- 303601 % 1000 = 601
  sorry

end grid_color_remainder_l397_397744


namespace minimum_rooms_to_accommodate_fans_l397_397980

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l397_397980


namespace yi_speed_is_27_l397_397857

-- Definitions and Assumptions
variables (V : ℝ) (S_Jia S_Yi : ℝ) (Jia Yi : ℝ)
variables (t_jia t_yi : ℝ) (buses_interval : ℝ)

-- Conditions
def buses_every_5_minutes : Prop := buses_interval = 5 
def jia_caught_by_9th_bus : Prop := t_jia = (8 * buses_interval) / 60
def yi_caught_by_6th_bus : Prop := t_yi = (5 * buses_interval) / 60
def yi_reached_A_with_8th_bus : Prop := yf_last_reach = (7 * buses_interval) / 60

-- Distances conditions
def jia_21_km_away_when_yi_reaches_A : Prop := (S_Jia - V * ((t_yf_last_reach / 60) - (t_jia / 60))) = 21

-- Problem
theorem yi_speed_is_27 :
  buses_every_5_minutes ∧ jia_caught_by_9th_bus ∧ yi_caught_by_6th_bus ∧ yi_reached_A_with_8th_bus ∧ jia_21_km_away_when_yi_reaches_A → Yi = 27 :=
begin
  sorry
end

end yi_speed_is_27_l397_397857


namespace sqrt_polynomial_eq_l397_397220

variable (a b c : ℝ)

def polynomial := 16 * a * c + 4 * a^2 - 12 * a * b + 9 * b^2 - 24 * b * c + 16 * c^2

theorem sqrt_polynomial_eq (a b c : ℝ) : 
  (polynomial a b c) ^ (1 / 2) = (2 * a - 3 * b + 4 * c) :=
by
  sorry

end sqrt_polynomial_eq_l397_397220


namespace liter_kerosene_cost_friday_l397_397753

-- Define initial conditions.
def cost_pound_rice_monday : ℚ := 0.36
def cost_dozen_eggs_monday : ℚ := cost_pound_rice_monday
def cost_half_liter_kerosene_monday : ℚ := (8 / 12) * cost_dozen_eggs_monday

-- Define the Wednesday price increase.
def percent_increase_rice : ℚ := 0.20
def cost_pound_rice_wednesday : ℚ := cost_pound_rice_monday * (1 + percent_increase_rice)
def cost_half_liter_kerosene_wednesday : ℚ := cost_half_liter_kerosene_monday * (1 + percent_increase_rice)

-- Define the Friday discount on eggs.
def percent_discount_eggs : ℚ := 0.10
def cost_dozen_eggs_friday : ℚ := cost_dozen_eggs_monday * (1 - percent_discount_eggs)
def cost_per_egg_friday : ℚ := cost_dozen_eggs_friday / 12

-- Define the price calculation for a liter of kerosene on Wednesday.
def cost_liter_kerosene_wednesday : ℚ := 2 * cost_half_liter_kerosene_wednesday

-- Define the final goal.
def cost_liter_kerosene_friday := cost_liter_kerosene_wednesday

theorem liter_kerosene_cost_friday : cost_liter_kerosene_friday = 0.576 := by
  sorry

end liter_kerosene_cost_friday_l397_397753


namespace glee_club_female_members_l397_397345

theorem glee_club_female_members (m f : ℕ) 
  (h1 : f = 2 * m) 
  (h2 : m + f = 18) : 
  f = 12 :=
by
  sorry

end glee_club_female_members_l397_397345


namespace chick_hit_count_l397_397121

theorem chick_hit_count :
  ∃ x y z : ℕ,
    9 * x + 5 * y + 2 * z = 61 ∧
    x + y + z = 10 ∧
    x ≥ 1 ∧
    y ≥ 1 ∧
    z ≥ 1 ∧
    x = 5 :=
by
  sorry

end chick_hit_count_l397_397121


namespace same_cost_duration_l397_397527

noncomputable def cost_plan_a (x : ℝ) : ℝ := 
  if x ≤ 6 then 0.60 else 0.60 + 0.06 * (x - 6)

def cost_plan_b (x : ℝ) : ℝ := 0.08 * x

theorem same_cost_duration : ∃ x : ℝ, cost_plan_a x = cost_plan_b x ∧ x = 12 :=
by
  use 12
  have h_plan_a : cost_plan_a 12 = 0.60 + 0.06 * (12 - 6) := by sorry
  have h_plan_b : cost_plan_b 12 = 0.08 * 12 := by sorry
  rw [h_plan_a, h_plan_b]
  norm_num
  rw [add_comm, add_sub_assoc, sub_self, add_zero]

end same_cost_duration_l397_397527


namespace leftover_space_l397_397177

noncomputable def desk_length : ℝ := 2
noncomputable def bookcase_length : ℝ := 1.5
noncomputable def chair_length : ℝ := 0.5
noncomputable def wall_length : ℝ := 30

theorem leftover_space : ∃ w : ℝ, w = wall_length - 7 * (desk_length + bookcase_length + chair_length) ∧ w = 2 :=
by {
  let set_length := desk_length + bookcase_length + chair_length,
  let num_of_sets := floor (wall_length / set_length),
  have h_num_of_sets : num_of_sets = 7 := by norm_num,
  let total_occupied_length := num_of_sets * set_length,
  have h_total_occupied_length : total_occupied_length = 28 := by norm_num,
  let w := wall_length - total_occupied_length,
  use w,
  have h_w : w = wall_length - total_occupied_length := by norm_num,
  split; finish,
}

end leftover_space_l397_397177


namespace pentagon_boundary_set_l397_397255

def regular_pentagon (A₁ A₂ A₃ A₄ A₅ : ℝ × ℝ) : Prop :=
  -- Definitions for regular pentagon properties
  sorry

def is_on_boundary (p : ℝ × ℝ) (A₁ A₂ A₃ A₄ A₅ : ℝ × ℝ) : Prop :=
  -- Definition for checking if a point is on the boundary of the pentagon
  sorry

def is_outside (p : ℝ × ℝ) (A₁ A₂ A₃ A₄ A₅ : ℝ × ℝ) : Prop :=
  -- Definition for checking if a point is outside the pentagon
  sorry

def is_inside (p : ℝ × ℝ) (A₁ A₂ A₃ A₄ A₅ : ℝ × ℝ) : Prop :=
  -- Definition for checking if a point is inside the pentagon
  sorry

theorem pentagon_boundary_set (A₁ A₂ A₃ A₄ A₅ : ℝ × ℝ)
  (h1 : regular_pentagon A₁ A₂ A₃ A₄ A₅) :
  ∃ S : set (ℝ × ℝ), (∀ p, is_outside p A₁ A₂ A₃ A₄ A₅ → ∃ a b ∈ S, a ≠ b ∧ p ∈ line_segment a b) ∧
                    (∀ q, is_inside q A₁ A₂ A₃ A₄ A₅ → ∀ a b ∈ S, a ≠ b → q ∉ line_segment a b) :=
  sorry

end pentagon_boundary_set_l397_397255


namespace average_output_l397_397936

theorem average_output :=
  let totalCogs := 60 + 60 + 80 + 50
  let totalTime := 60 / 20 + 60 / 60 + 80 / 40 + 50 / 70
  (totalCogs : ℝ) / totalTime = 250

-- Starting definition to allow Lean to move forward on proving
calc
250 = (250 : ℝ) / 1 : by ring
... = 250 / (47/7) * (7/47) : by ring
... = 250 / ((42/7) + (5/7)) * (7/47) : by ring
... = 250 / (3 + 1 + 2 + 5/7) * (7/47) : by ring
... = 250 / ((60 / 20) + (60 / 60) + (80 / 40) + (50 / 70)) * (7/47) : by ring
... = 250 / totalTime * (7/47) : by ring
... = 250 cogs / totalTime : by sorry  -- Include this inside goal


end average_output_l397_397936


namespace amplitude_period_initial_phase_max_value_set_transformation_from_sine_l397_397709

open Real

noncomputable def f (x φ A : ℝ) : ℝ := sin (2 * x + φ) + A

theorem amplitude_period_initial_phase (φ A : ℝ) :
  ∃ amp prd in_phs : ℝ, 
    amp = A ∧ 
    prd = π ∧ 
    in_phs = φ :=
begin
  use [A, π, φ],
  repeat { split, refl }
end

theorem max_value_set (φ A : ℝ) : 
  ∃ x_max_set : set ℝ, 
    x_max_set = { x | ∃ k : ℤ, x = π / 4 + k * π } :=
begin
  use { x | ∃ k : ℤ, x = π / 4 + k * π },
  simp
end

theorem transformation_from_sine (φ A : ℝ) :
  ∃ (transformations : list (ℝ → ℝ → ℝ → ℝ)),
    transformations = [
      λ y φ A, sin(y + φ),
      λ y φ A, sin(2 * y + φ),
      λ y φ A, sin(2 * y + φ) + A ] :=
begin
  use [
    λ y φ A, sin(y + φ),
    λ y φ A, sin(2 * y + φ),
    λ y φ A, sin(2 * y + φ) + A ],
  repeat { split, refl }
end

end amplitude_period_initial_phase_max_value_set_transformation_from_sine_l397_397709


namespace female_members_count_l397_397342

theorem female_members_count (M F : ℕ) (h1 : F = 2 * M) (h2 : F + M = 18) : F = 12 :=
by
  -- the proof will go here
  sorry

end female_members_count_l397_397342


namespace binomial_12_11_eq_12_l397_397614

-- Define binomial coefficient function
def binomial (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement to prove
theorem binomial_12_11_eq_12 :
  binomial 12 11 = 12 := by
  sorry

end binomial_12_11_eq_12_l397_397614


namespace final_milk_quantity_l397_397125

def initial_milk_volume := 90
def removed_milk_volume := 9

theorem final_milk_quantity :
  let final_milk := (initial_milk_volume - removed_milk_volume) * ((initial_milk_volume - removed_milk_volume : ℝ) / initial_milk_volume) in
  final_milk + removed_milk_volume * (initial_milk_volume - removed_milk_volume : ℝ) / initial_milk_volume = 72.9 :=
by sorry

end final_milk_quantity_l397_397125


namespace courtyard_width_l397_397564

theorem courtyard_width (length : ℝ) (area_per_brick : ℝ) (num_bricks : ℕ) (total_area : ℝ) (width : ℝ) :
  length = 25 ∧ area_per_brick = 0.02 ∧ num_bricks = 22500 ∧ total_area = length * width ∧ total_area = num_bricks * area_per_brick 
  → width = 18 :=
begin
  sorry
end

end courtyard_width_l397_397564


namespace find_a_plus_b_l397_397054

-- Define the constants and conditions
variables (a b c : ℤ)
variables (a_cond : 0 ≤ a ∧ a < 5) (b_cond : 0 ≤ b ∧ b < 13)
variables (frac_decomp : (1 : ℚ) / 2015 = (a : ℚ) / 5 + (b : ℚ) / 13 + (c : ℚ) / 31)

-- State the theorem
theorem find_a_plus_b (a b c : ℤ) (a_cond : 0 ≤ a ∧ a < 5) (b_cond : 0 ≤ b ∧ b < 13) (frac_decomp : (1 : ℚ) / 2015 = (a : ℚ) / 5 + (b : ℚ) / 13 + (c : ℚ) / 31) :
  a + b = 14 := 
sorry

end find_a_plus_b_l397_397054


namespace fraction_of_4d_nails_l397_397943

variables (fraction2d fraction2d_or_4d fraction4d : ℚ)

theorem fraction_of_4d_nails
  (h1 : fraction2d = 0.25)
  (h2 : fraction2d_or_4d = 0.75) :
  fraction4d = 0.50 :=
by
  sorry

end fraction_of_4d_nails_l397_397943


namespace greatest_possible_value_of_x_l397_397105

theorem greatest_possible_value_of_x :
  ∃ (x : ℚ), x = 9 / 5 ∧ 
  (\left(5 * x - 20) / (4 * x - 5)) ^ 2 + \left((5 * x - 20) / (4 * x - 5)) = 20 ∧ x ≥ 0 :=
begin
  existsi (9 / 5 : ℚ),
  split,
  { refl },
  split,
  { sorry },
  { sorry }
end

end greatest_possible_value_of_x_l397_397105


namespace cos_sin_pow_l397_397787

noncomputable def imaginary_unit : ℂ := complex.I

theorem cos_sin_pow (n : ℕ) (θ : ℝ) (h1 : 0 ≤ θ) (h2 : θ < 2 * real.pi) :
  (complex.cos θ + complex.I * complex.sin θ) ^ n = 
  complex.cos (n * θ) + complex.I * complex.sin (n * θ) :=
sorry

end cos_sin_pow_l397_397787


namespace m_is_7_digit_l397_397477

theorem m_is_7_digit (m : ℕ) (h : log 10 m = 6.32) : 10^6 < m ∧ m < 10^7 :=
sorry

end m_is_7_digit_l397_397477


namespace quadratic_roots_are_real_l397_397325

theorem quadratic_roots_are_real (k : ℝ) (h : (16 - 8 * k) ≥ 0) : 
  ∃ a b c : ℝ, ∃ p q : ℝ, a = 1 ∧ b = p ∧ c = q ∧
  (p^2 - 4 * p + 2 * k = 0) ∧ (q^2 - 4 * q + 2 * k = 0) ∧ 
  p = 2 ∧ q = 2 ∧ a + b + c = 5 := 
by
  sorry

end quadratic_roots_are_real_l397_397325


namespace translate_parabola_l397_397069

theorem translate_parabola :
  ∀ (x : ℝ), (λ x, 2 * (x + 1) ^ 2 - 3) (x - 1) + 3 = 2 * x ^ 2 :=
by
  intro x
  sorry

end translate_parabola_l397_397069


namespace sum_one_over_pq_l397_397413

open Nat

/-- Given any integer n ≥ 2, the sum of 1/(pq) where the summation is over all integers p, q satisfying 
  0 < p < q ≤ n, p + q > n, and gcd(p, q) = 1, equals 1/2. -/
theorem sum_one_over_pq (n : ℕ) (hn : 2 ≤ n) :
  (∑ p in (finset.range n).filter (λ p, 0 < p),
   ∑ q in (finset.range (n+1)).filter (λ q, p < q ∧ p + q > n ∧ Nat.gcd p q = 1),
   1 / (p * q : ℚ)) = 1 / 2 := 
by
  -- Code here for detailed conditions and the final proof
  sorry

end sum_one_over_pq_l397_397413


namespace equation_of_parallel_line_through_point_l397_397227

theorem equation_of_parallel_line_through_point :
  ∃ m b, (∀ x y, y = m * x + b → (∃ k, k = 3 ^ 2 - 9 * 2 + 1)) ∧ 
         (∀ x y, y = 3 * x + b → y - 0 = 3 * (x - (-2))) :=
sorry

end equation_of_parallel_line_through_point_l397_397227


namespace find_a_l397_397769

theorem find_a (a : ℝ) : 
  (∃ S, S = {x : ℝ | (x - 1) * (x^2 + a * x + 4) = 0} ∧ (S.sum = 3)) ↔ (a = -4) :=
sorry

end find_a_l397_397769


namespace new_student_weight_l397_397129

theorem new_student_weight (old_weight : ℤ) (new_weight : ℤ) (avg_decrease : ℤ) (total_students : ℕ) 
  (h1 : avg_decrease = 8) (h2 : old_weight = 96) (h3 : total_students = 4)
  (total_decrease : ℤ) (h4 : total_decrease = total_students * avg_decrease) :
  new_weight = old_weight - total_decrease := 
by
  have h5 : total_decrease = 32 := by sorry
  have h6 : new_weight = old_weight - h5 := by sorry
  show new_weight = old_weight - total_decrease from 
  begin
    rw h5 at h6,
    exact h6,
  end

end new_student_weight_l397_397129


namespace total_cakes_served_today_l397_397165

def cakes_served_lunch : ℕ := 6
def cakes_served_dinner : ℕ := 9
def total_cakes_served (lunch cakes_served_dinner : ℕ) : ℕ :=
  lunch + cakes_served_dinner

theorem total_cakes_served_today : total_cakes_served cakes_served_lunch cakes_served_dinner = 15 := 
by
  sorry

end total_cakes_served_today_l397_397165


namespace slower_train_speed_l397_397870

def relative_speed_of_trains (V1 V2 : ℝ) : ℝ := V1 - V2

def convert_speed_km_per_hr_to_m_per_sec (speed : ℝ) : ℝ := speed * (5/18)

def total_length_of_trains (length : ℝ) : ℝ := length + length

def time_to_pass (distance speed : ℝ) : ℝ := distance / speed

theorem slower_train_speed (V : ℝ) (time : ℝ) (length : ℝ)
  (faster_train_speed_kmh : ℝ) 
  (pass_time_seconds : ℝ)
  (train_length_meters : ℝ) :
  (V = 36) →
  (faster_train_speed_kmh = 46) →
  (pass_time_seconds = 54) →
  (train_length_meters = 75) →
  time_to_pass (total_length_of_trains train_length_meters)
       (convert_speed_km_per_hr_to_m_per_sec (relative_speed_of_trains faster_train_speed_kmh V)) = pass_time_seconds :=
by
  sorry

end slower_train_speed_l397_397870


namespace max_x_possible_value_l397_397101

theorem max_x_possible_value : ∃ x : ℚ, 
  (∃ y : ℚ, y = (5 * x - 20) / (4 * x - 5) ∧ (y^2 + y = 20)) ∧
  x = 9 / 5 :=
begin
  sorry
end

end max_x_possible_value_l397_397101


namespace bitcoin_donation_l397_397398

theorem bitcoin_donation (x : ℝ) (h : 3 * (80 - x) / 2 - 10 = 80) : x = 20 :=
sorry

end bitcoin_donation_l397_397398


namespace cos_angle_unit_circle_l397_397281

theorem cos_angle_unit_circle (α : ℝ) (h : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x = 1/2 ∧ y = sqrt(3)/2) : 
  real.cos α = 1 / 2 :=
by
  sorry

end cos_angle_unit_circle_l397_397281


namespace parabola_intersection_l397_397868

def parabola1 (x : ℝ) : ℝ := 3 * x ^ 2 - 4 * x + 2
def parabola2 (x : ℝ) : ℝ := 9 * x ^ 2 + 6 * x + 2

theorem parabola_intersection :
  ∃ (x y : ℝ), (parabola1 x = y ∧ parabola2 x = y) ∧ 
                ((x = 0 ∧ y = 2) ∨ (x = -5 / 3 ∧ y = 17)) :=
by
  sorry

end parabola_intersection_l397_397868


namespace minimum_balls_to_guarantee_18_l397_397144

theorem minimum_balls_to_guarantee_18 (
    red_balls : ℕ,
    green_balls : ℕ,
    yellow_balls : ℕ,
    blue_balls : ℕ,
    purple_balls : ℕ
) : (red_balls = 34) →
    (green_balls = 25) →
    (yellow_balls = 18) →
    (blue_balls = 21) →
    (purple_balls = 13) →
    ∃ (n : ℕ), n = 82 ∧
        ∀ (drawn : ℕ), (drawn >= 82) → (
            red_balls + green_balls + yellow_balls + blue_balls + purple_balls >= drawn →
            ∃ (color : string), (
                (color = "red" ∧ red_balls ≥ 18) ∨ 
                (color = "green" ∧ green_balls ≥ 18) ∨ 
                (color = "yellow" ∧ yellow_balls ≥ 18) ∨ 
                (color = "blue" ∧ blue_balls ≥ 18) ∨ 
                (color = "purple" ∧ purple_balls ≥ 18)
            )
        )
:= by {
    sorry
}

end minimum_balls_to_guarantee_18_l397_397144


namespace trig_identity_l397_397626

theorem trig_identity : 
  sin (real.pi * 17 / 180) * cos (real.pi * 43 / 180) + sin (real.pi * 73 / 180) * sin (real.pi * 43 / 180) = real.sqrt 3 / 2 := 
by
  sorry

end trig_identity_l397_397626


namespace no_digit_3_in_35th_l397_397620

theorem no_digit_3_in_35th {l : List ℕ} (h : l = List.range' 1 7) : 
  ∃ n : ℕ, n ∉ [4, 5, 6, 7] ∧ (l.combination 4).getLast [] = [4, 5, 6, 7] := 
sorry

end no_digit_3_in_35th_l397_397620


namespace statement_equivalence_l397_397294

variables (α β γ : Plane)
variables (a b c : Line)
variable (θ : Real)

-- Assumptions
axiom angle_between_planes : ∀ {P Q : Plane}, (P ≠ Q) → (∃ θ : Real, 0 < θ ∧ θ ≤ π ∧ Angle (P, Q) = θ)
axiom intersection_lines : α ∩ β = a ∧ β ∩ γ = b ∧ γ ∩ α = c
axiom statement1 : θ > π / 3
axiom statement2 : ∃ P : Point, P ∈ a ∧ P ∈ b ∧ P ∈ c

-- Theorem to prove
theorem statement_equivalence : (θ > π / 3) ↔ (∃ P : Point, P ∈ a ∧ P ∈ b ∧ P ∈ c) :=
sorry

end statement_equivalence_l397_397294


namespace find_k_l397_397719

open Function

-- Definitions of the given vectors and condition of collinearity
def vec_OA (k : ℝ) : ℝ × ℝ := (k, 12)
def vec_OB : ℝ × ℝ := (4, 5)
def vec_OC (k : ℝ) : ℝ × ℝ := (-k, 10)

def vec_AB (k : ℝ) : ℝ × ℝ := (4 - k, 5 - 12)
def vec_AC (k : ℝ) : ℝ × ℝ := (-k - k, 10 - 12)

def are_collinear (k : ℝ) : Prop :=
  ∃ λ : ℝ, vec_AB k = (RATSMUL λ (vec_AC k))

-- Statement to be proven
theorem find_k (k : ℝ) (h : are_collinear k) : k = ((-2) / 3) := sorry

end find_k_l397_397719


namespace mystic_aquarium_feeding_l397_397180

-- Define the conditions given in the problem.
variables (R : ℝ)
def shark_buckets := 4
def dolphin_buckets := 4 * R
def other_sea_animal_buckets := 5 * shark_buckets
def total_daily_buckets := shark_buckets + dolphin_buckets + other_sea_animal_buckets
def total_weeks_buckets := 546
def num_days := 21

-- Define the goal: The ratio of the number of buckets fed to the dolphins to the number of buckets fed to the sharks.
def goal := R = 1/2

theorem mystic_aquarium_feeding : total_daily_buckets * num_days = total_weeks_buckets → goal :=
by
  sorry

end mystic_aquarium_feeding_l397_397180


namespace max_percentage_difference_l397_397179

-- Define sales data for each group and each month
def sales (month : String) (group : String) : Nat :=
  match month, group with
  | "January", "Drummers" => 6
  | "January", "Bugle players" => 4
  | "January", "Percussionists" => 3
  | "February", "Drummers" => 5
  | "February", "Bugle players" => 6
  | "February", "Percussionists" => 2
  | "March", "Drummers" => 7
  | "March", "Bugle players" => 5
  | "March", "Percussionists" => 5
  | "April", "Drummers" => 4
  | "April", "Bugle players" => 6
  | "April", "Percussionists" => 3
  | "May", "Drummers" => 8
  | "May", "Bugle players" => 3
  | "May", "Percussionists" => 4
  | _, _ => 0

-- Define the percentage difference formula
def percentage_difference (maxGroup : Nat) (secondGroup : Nat) (thirdGroup : Nat) : Float :=
  ((maxGroup.toFloat - (secondGroup.toFloat + thirdGroup.toFloat)) / (secondGroup.toFloat + thirdGroup.toFloat)) * 100.0

-- Calculate the percentage difference for each month
def percentage_difference_january : Float :=
  percentage_difference (max (sales "January" "Drummers") (max (sales "January" "Bugle players") (sales "January" "Percussionists")))
                         (sales "January" "Bugle players" + sales "January" "Percussionists")

def percentage_difference_february : Float :=
  percentage_difference (max (sales "February" "Drummers") (max (sales "February" "Bugle players") (sales "February" "Percussionists")))
                         (sales "February" "Bugle players" + sales "February" "Percussionists")

def percentage_difference_march : Float :=
  percentage_difference (max (sales "March" "Drummers") (max (sales "March" "Bugle players") (sales "March" "Percussionists")))
                         (sales "March" "Bugle players" + sales "March" "Percussionists")

def percentage_difference_april : Float :=
  percentage_difference (max (sales "April" "Drummers") (max (sales "April" "Bugle players") (sales "April" "Percussionists")))
                         (sales "April" "Bugle players" + sales "April" "Percussionists")

def percentage_difference_may : Float :=
  percentage_difference (max (sales "May" "Drummers") (max (sales "May" "Bugle players") (sales "May" "Percussionists")))
                         (sales "May" "Bugle players" + sales "May" "Percussionists")

-- The theorem to state the problem
theorem max_percentage_difference : 
  (percentage_difference_may > percentage_difference_january) ∧
  (percentage_difference_may > percentage_difference_february) ∧
  (percentage_difference_may > percentage_difference_march) ∧
  (percentage_difference_may > percentage_difference_april) := 
sorry

end max_percentage_difference_l397_397179


namespace lucky_sum_equal_prob_l397_397360

theorem lucky_sum_equal_prob (N : ℕ) (S63_main_draw : Finset ℕ) (S44_additional_draw : Finset ℕ)
  (hN_pos : 0 < N)
  (hS63 : S63_main_draw.card = 10)
  (hSum63 : S63_main_draw.sum id = 63)
  (hS44 : S44_additional_draw.card = 8)
  (hSum44 : S44_additional_draw.sum id = 44)
  (hS_distinct : ∀ (n ∈ S63_main_draw ∪ S44_additional_draw), n ∈ Finset.range (N + 1)) :
  N = 18 :=
by
  sorry

end lucky_sum_equal_prob_l397_397360


namespace star_1_4_3_2_eq_3_7_l397_397958

def fractional_part (x : ℝ) : ℝ := x - x.floor

def star_operation (a b : ℝ) : ℝ :=
  2 * fractional_part (a / 2) + 3 * fractional_part ((a + b) / 6)

theorem star_1_4_3_2_eq_3_7 : star_operation 1.4 3.2 = 3.7 :=
by {
  sorry
}

end star_1_4_3_2_eq_3_7_l397_397958


namespace range_of_x_l397_397662

theorem range_of_x (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) : 1 ≤ x ∧ x < 5 / 4 := 
  sorry

end range_of_x_l397_397662


namespace inequality_in_triangle_l397_397033

variables {A B C M : Type*}
variables {d_a d_b d_c R_a R_b R_c : ℝ}

-- Assume point M is inside the triangle ABC
-- Assume distances d_a, d_b, d_c from M to sides BC, CA, AB respectively
-- Assume R_a, R_b, R_c are the radii of the excircles opposite vertices A, B, C respectively

theorem inequality_in_triangle
  (h_A : A) (h_B : B) (h_C : C) (h_M : M)
  (h_d_a : d_a > 0) (h_d_b : d_b > 0) (h_d_c : d_c > 0)
  (h_R_a : R_a > 0) (h_R_b : R_b > 0) (h_R_c : R_c > 0) :
  d_a * R_a + d_b * R_b + d_c * R_c ≥ 2 * (d_a * d_b + d_b * d_c + d_c * d_a) :=
begin
  -- Proof goes here
  sorry
end

end inequality_in_triangle_l397_397033


namespace molecular_weight_determination_impossible_l397_397113

theorem molecular_weight_determination_impossible :
  ∀ (other_compound : Type) 
    (molecular_weight_CaH2_10_moles : ℝ), 
    (molecular_weight_CaH2_10_moles = 420) → 
    ∃ (chemical_formula_other_compound : Type), 
      ∀ (molecular_weight_other_compound_10_moles : ℝ), 
         molecular_weight_other_compound_10_moles = 10 * molecular_weight_CaH2_10_moles →
           (∃ (molecular_weight_other_compound : ℝ), molecular_weight_other_compound = molecular_weight_other_compound_10_moles / 10) → 
             False :=
begin
  sorry,
end

end molecular_weight_determination_impossible_l397_397113


namespace sum_of_polynomial_l397_397078

variable (P m : ℝ)

theorem sum_of_polynomial : P + (m^2 + m) = m^2 - 2m ↔ P = -3m := 
by 
  sorry

end sum_of_polynomial_l397_397078


namespace shortest_distance_between_circles_l397_397517

-- Conditions
def first_circle (x y : ℝ) : Prop := x^2 - 10 * x + y^2 - 4 * y - 7 = 0
def second_circle (x y : ℝ) : Prop := x^2 + 14 * x + y^2 + 6 * y + 49 = 0

-- Goal: Prove the shortest distance between the two circles is 4
theorem shortest_distance_between_circles : 
  -- Given conditions about the equations of the circles
  (∀ x y : ℝ, first_circle x y ↔ (x - 5)^2 + (y - 2)^2 = 36) ∧ 
  (∀ x y : ℝ, second_circle x y ↔ (x + 7)^2 + (y + 3)^2 = 9) →
  -- Assert the shortest distance between the two circles is 4
  13 - (6 + 3) = 4 :=
by
  sorry

end shortest_distance_between_circles_l397_397517


namespace sum_inequality_for_natural_number_l397_397032

theorem sum_inequality_for_natural_number (n : ℕ) :
  ∑ i in finset.range n, ((2 * i - 1) * (2 * i)^2 - 2 * i * (2 * i + 1)^2) = -(n * (n + 1) * (4 * n + 3)) :=
by
  sorry

end sum_inequality_for_natural_number_l397_397032


namespace parity_of_f_l397_397700

noncomputable def f (x : ℝ) : ℝ := (sqrt (9 - x^2)) / (|6 - x| - 6)

theorem parity_of_f :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ x ≠ 0 → f (-x) = -f x) ∧ ¬ (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ x ≠ 0 → f x = f (-x)) :=
by
  sorry

end parity_of_f_l397_397700


namespace congruent_triangles_from_colored_circle_l397_397913

theorem congruent_triangles_from_colored_circle :
  ∀ (circle_divided_into_432_points : Fin 432) (coloring : Fin 432 → Fin 4),
  (∃ (red_indices green_indices blue_indices yellow_indices : Finset (Fin 432)),
    red_indices.card = 108 ∧ green_indices.card = 108 ∧ blue_indices.card = 108 ∧ yellow_indices.card = 108 ∧
    (∀ i, i ∈ red_indices → coloring i = 0) ∧
    (∀ i, i ∈ green_indices → coloring i = 1) ∧
    (∀ i, i ∈ blue_indices → coloring i = 2) ∧
    (∀ i, i ∈ yellow_indices → coloring i = 3)) →
  ∃ red_triangle green_triangle blue_triangle yellow_triangle : Finset (Fin 432),
    red_triangle.card = 3 ∧ green_triangle.card = 3 ∧ blue_triangle.card = 3 ∧ yellow_triangle.card = 3 ∧
    (∀ triangle, triangle.card = 3 → 
      ∃ rotation : Fin 432 → Fin 432,
        rotation '' red_triangle = green_triangle ∧
        rotation '' green_triangle = blue_triangle ∧
        rotation '' blue_triangle = yellow_triangle ∧
        rotation '' yellow_triangle = red_triangle) :=
λ circle_divided_into_432_points coloring color_distribution_exists,
begin
  sorry
end

end congruent_triangles_from_colored_circle_l397_397913


namespace prob_A_not_less_than_B_after_green_first_prob_dist_expectation_B_after_red_l397_397746

-- Conditions setup
def balls : List (String × ℕ) := [("white", 1), ("white", 1), ("white", 1), ("yellow", 2), ("yellow", 2), ("yellow", 2), ("red", 3), ("green", 4)]

-- Question 1: Probability that A's score is not less than B's
theorem prob_A_not_less_than_B_after_green_first:
  -- condition: first ball drawn by A is green
  -- result: probability that A's score is not less than B's score
  (prob_A_ge_B : ℚ) :=
  prob_A_ge_B = (3 / 7) := sorry

-- Question 2: Probability distribution and expectation of B's score ξ given first ball red
theorem prob_dist_expectation_B_after_red:
  -- condition: first ball drawn by B is red
  -- result: probability distribution of B's score and its mathematical expectation
  (ξ : fin 6 → ℚ) (Eξ : ℚ) :=
  ξ 0 = (1 / 35) ∧
  ξ 1 = (9 / 35) ∧
  ξ 2 = (9 / 35) ∧
  ξ 3 = (4 / 35) ∧
  ξ 4 = (9 / 35) ∧
  ξ 5 = (3 / 35) ∧
  Eξ = (60 / 7) := sorry

end prob_A_not_less_than_B_after_green_first_prob_dist_expectation_B_after_red_l397_397746


namespace telepathic_connection_probability_l397_397352

theorem telepathic_connection_probability :
  let balls := [6, 7, 8, 9],
      all_pairs := List.product balls balls,
      favorable_pairs := List.filter (λ (p : ℕ × ℕ), (|p.1 - p.2| <= 1)) all_pairs,
      probability := favorable_pairs.length / all_pairs.length in
  probability = 5 / 8 :=
by
  sorry

end telepathic_connection_probability_l397_397352


namespace andy_more_candies_than_caleb_l397_397190

theorem andy_more_candies_than_caleb :
  let billy_initial := 6
  let caleb_initial := 11
  let andy_initial := 9
  let father_packet := 36
  let billy_additional := 8
  let caleb_additional := 11
  let billy_total := billy_initial + billy_additional
  let caleb_total := caleb_initial + caleb_additional
  let total_given := billy_additional + caleb_additional
  let andy_additional := father_packet - total_given
  let andy_total := andy_initial + andy_additional
  andy_total - caleb_total = 4 :=
by {
  sorry
}

end andy_more_candies_than_caleb_l397_397190


namespace eval_expression_l397_397631

theorem eval_expression : 9^9 * 3^3 / 3^30 = 1 / 19683 := by
  sorry

end eval_expression_l397_397631


namespace problem_statement_l397_397001

noncomputable def p : ℝ := sorry
noncomputable def q : ℝ := sorry
noncomputable def r : ℝ := sorry

def midpoint (x y : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((x.1 + y.1) / 2, (x.2 + y.2) / 2, (x.3 + y.3) / 2)

noncomputable def A : ℝ × ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ × ℝ := sorry
noncomputable def C : ℝ × ℝ × ℝ := sorry

axiom midpoint_BC : midpoint B C = (p, p, 0)
axiom midpoint_AC : midpoint A C = (0, q, q)
axiom midpoint_AB : midpoint A B = (r, 0, r)

theorem problem_statement :
  (dist A B)^2 + (dist A C)^2 + (dist B C)^2 = 8 * (p^2 + q^2 + r^2) :=
begin
  sorry
end

end problem_statement_l397_397001


namespace car_actual_speed_l397_397124

theorem car_actual_speed (S : ℝ) 
    (h1 : ∀ (S : ℝ), (\frac{5}{7} * S) = 35)
    (h2 : 42 / \frac{42}{25} = 25)  :
    S = 35 := by
  sorry

end car_actual_speed_l397_397124


namespace minimum_rooms_to_accommodate_fans_l397_397985

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l397_397985


namespace distance_AB_eq_sqrt_13_l397_397680

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_AB_eq_sqrt_13 :
  let A : point := (2, 1)
  let B : point := (5, -1)
  distance A B = real.sqrt 13 :=
by
  sorry

end distance_AB_eq_sqrt_13_l397_397680


namespace d_is_rth_power_of_integer_l397_397006

theorem d_is_rth_power_of_integer 
  (d r : ℤ) 
  (a b : ℤ) 
  (hr : r ≠ 0) 
  (hab : (a, b) ≠ (0, 0)) 
  (h_eq : a ^ r = d * b ^ r) : 
  ∃ (δ : ℤ), d = δ ^ r :=
sorry

end d_is_rth_power_of_integer_l397_397006


namespace arrangement_ways_l397_397547

theorem arrangement_ways :
  let P := {x // 1 ≤ x ∧ x ≤ 6} in
  let non_adjacent (a b : ℕ) : Prop := abs (a - b) ≠ 1 in
  let valid_positions := {2, 3, 4, 5} in
  let count_ways := λ valid_positions, 288 in
  ∃ (A B : ℕ) (people : List ℕ), 
     A ∈ valid_positions ∧ 
     B ∈ P ∧ 
     A ≠ B ∧ 
     non_adjacent A B ∧ 
     length people = 6 ∧ 
     count_ways valid_positions = 288 :=
by
  sorry

end arrangement_ways_l397_397547


namespace simplify_expression_l397_397437

theorem simplify_expression :
  20 * (14 / 15) * (2 / 18) * (5 / 4) = 70 / 27 :=
by sorry

end simplify_expression_l397_397437


namespace value_of_expression_l397_397320

theorem value_of_expression (n : ℝ) (h : n + 1/n = 10) : n^2 + (1/n^2) + 6 = 104 :=
by
  sorry

end value_of_expression_l397_397320


namespace first_number_divisible_by_3_and_7_in_100_to_600_l397_397086

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b
noncomputable def first_divisible (low high lcm_val : ℕ) : ℕ :=
  let m := (low + lcm_val - 1) / lcm_val
  m * lcm_val

theorem first_number_divisible_by_3_and_7_in_100_to_600 (low high : ℕ) (h1 : low = 100) (h2 : high = 600) :
  first_divisible low high (lcm 3 7) = 105 :=
  by
  rw [h1, h2]
  sorry

end first_number_divisible_by_3_and_7_in_100_to_600_l397_397086


namespace response_rate_percentage_correct_l397_397555

theorem response_rate_percentage_correct :
  let responses_needed := 300
  let questionnaires_mailed := 428.5714285714286
  (responses_needed / questionnaires_mailed) * 100 ≈ 70 := 
by
  sorry

end response_rate_percentage_correct_l397_397555


namespace infinitely_many_n_l397_397034

-- Definition capturing the condition: equation \( (x + y + z)^3 = n^2 xyz \)
def equation (x y z n : ℕ) : Prop := (x + y + z)^3 = n^2 * x * y * z

-- The main statement: proving the existence of infinitely many positive integers n such that the equation has a solution
theorem infinitely_many_n :
  ∃ᶠ n : ℕ in at_top, ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z n :=
sorry

end infinitely_many_n_l397_397034


namespace math_proof_problem_l397_397010

/-- Given three real numbers a, b, and c such that a ≥ b ≥ 1 ≥ c ≥ 0 and a + b + c = 3.

Part (a): Prove that 2 ≤ ab + bc + ca ≤ 3.
Part (b): Prove that (24 / (a^3 + b^3 + c^3)) + (25 / (ab + bc + ca)) ≥ 14.
--/
theorem math_proof_problem (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ 1) (h3 : 1 ≥ c)
  (h4 : c ≥ 0) (h5 : a + b + c = 3) :
  (2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 3) ∧ 
  (24 / (a^3 + b^3 + c^3) + 25 / (a * b + b * c + c * a) ≥ 14) 
  :=
by
  sorry

end math_proof_problem_l397_397010


namespace find_equation_of_line_l397_397228

variable (x y : ℝ)

def line_parallel (x y : ℝ) (m : ℝ) :=
  x - 2*y + m = 0

def line_through_point (x y : ℝ) (px py : ℝ) (m : ℝ) :=
  (px - 2 * py + m = 0)
  
theorem find_equation_of_line :
  let px := -1
  let py := 3
  ∃ m, line_parallel x y m ∧ line_through_point x y px py m ∧ m = 7 :=
by
  sorry

end find_equation_of_line_l397_397228


namespace region_area_l397_397751

noncomputable def area_of_region_outside_hexagon_inside_semicircles (s : ℝ) : ℝ :=
  let area_hexagon := (3 * Real.sqrt 3 / 2) * s^2
  let area_semicircle := (1/2) * Real.pi * (s/2)^2
  let total_area_semicircles := 6 * area_semicircle
  let total_area_circles := 6 * Real.pi * (s/2)^2
  total_area_circles - area_hexagon

theorem region_area (s := 2) : area_of_region_outside_hexagon_inside_semicircles s = (6 * Real.pi - 6 * Real.sqrt 3) :=
by
  sorry  -- Proof is skipped.

end region_area_l397_397751


namespace check_prime_large_number_l397_397764

def large_number := 23021^377 - 1

theorem check_prime_large_number : ¬ Prime large_number := by
  sorry

end check_prime_large_number_l397_397764


namespace liquid_left_after_evaporation_l397_397439

-- Definitions
def solution_y (total_mass : ℝ) : ℝ × ℝ :=
  (0.30 * total_mass, 0.70 * total_mass) -- liquid_x, water

def evaporate_water (initial_water : ℝ) (evaporated_mass : ℝ) : ℝ :=
  initial_water - evaporated_mass

-- Condition that new solution is 45% liquid x
theorem liquid_left_after_evaporation 
  (initial_mass : ℝ) 
  (evaporated_mass : ℝ)
  (added_mass : ℝ)
  (new_percentage_liquid_x : ℝ) :
  initial_mass = 8 → 
  evaporated_mass = 4 → 
  added_mass = 4 →
  new_percentage_liquid_x = 0.45 →
  solution_y initial_mass = (2.4, 5.6) →
  evaporate_water 5.6 evaporated_mass = 1.6 →
  solution_y added_mass = (1.2, 2.8) →
  2.4 + 1.2 = 3.6 →
  1.6 + 2.8 = 4.4 →
  0.45 * (3.6 + 4.4) = 3.6 →
  4 = 2.4 + 1.6 := sorry

end liquid_left_after_evaporation_l397_397439


namespace inverse_exponent_l397_397193

theorem inverse_exponent : (2/3 : ℝ) ⁻² = 9/4 := 
by
  sorry

end inverse_exponent_l397_397193


namespace min_value_of_seq_l397_397258

variable {a : ℕ → ℕ}
variable {n : ℕ}

/-- Definition of the sequence satisfying given conditions -/
def a (n : ℕ) : ℕ := match n with
  | 0     => 25
  | n + 1 => a n + 2 * n

/-- The main theorem to prove the minimum value of a_n / n -/
theorem min_value_of_seq : ∃ n : ℕ, (∀ k : ℕ, k ≠ n → a k / k > 9) ∧ (a n / n = 9) :=
sorry

end min_value_of_seq_l397_397258


namespace triangle_construction_l397_397957

noncomputable def construct_triangle (R MS d : ℝ) : Prop :=
  ∃ (M S K : ℝ × ℝ) (triangle : Set (ℝ × ℝ)),
    -- Euler line relationship
    dist M S = 2 * dist S K ∧ 
    -- Circumcircle radius
    dist K (0, 0) = R ∧ 
    -- Reflection property
    ∃ M' : ℝ × ℝ, dist M M' = 2 * d ∧ M' ∈ circle K R ∧ 
    -- Define the side using perpendicular bisector property
    ∃ A B C : ℝ × ℝ, A ∈ circle K R ∧ B ∈ circle K R ∧ C ∈ circle K R ∧ 
    is_perpendicular_bisector (A, B) (M, M') ∧ is_perpendicular (C, (A, B)) 
    -- Ensuring triangle is formed
    ⟨A, B, C⟩ = triangle

theorem triangle_construction (R MS d : ℝ) : construct_triangle R MS d :=
  sorry

end triangle_construction_l397_397957


namespace sum_possible_initial_numbers_correct_l397_397589

def smallest_prime_divisor (n : ℕ) : ℕ :=
  if n < 2 then n else List.min' (List.filter (fun d => n % d = 0) (List.range n).tail {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997}.erase n)

-- Given conditions
def initial_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n + k * 2022 = some prime number

-- All possible initial numbers, denoted by s
noncomputable def possible_initial_numbers (s : list ℕ) : Prop :=
  s = [4046, 4047]

-- Goal statement
theorem sum_possible_initial_numbers_correct : 
  (∑ n in possible_initial_numbers, n) = 8093 := by
  sorry

end sum_possible_initial_numbers_correct_l397_397589


namespace trader_gain_percentage_l397_397606

theorem trader_gain_percentage {C : ℝ} (hC : C > 0) :
  let gain_percentage := (379 / 1225) * 100 in
  gain_percentage ≈ 30.94 :=
by
  have h1 : (379 * 100 / 1225 : ℝ) = Approximate (30.94 : ℝ) := by sorry
  exact h1

end trader_gain_percentage_l397_397606


namespace calc_256_neg_exp_l397_397950

theorem calc_256_neg_exp : (256 : ℝ) ^ (-2 ^ (-3 : ℝ)) = 1 / 2 := 
  sorry

end calc_256_neg_exp_l397_397950


namespace area_of_all_plots_equal_26_hectares_l397_397452

theorem area_of_all_plots_equal_26_hectares :
  let ratio1 := 11 / 4,
      ratio2 := 11 / 6,
      ratio3 := 11 / 8,
      yield := 18,
      difference := 72 in
  ∃ (k : ℝ), 
    (6 * k * yield - 4 * k * yield = difference) ∧
    (6 * k + 4 * k + 3 * k = 26) := by
  sorry

end area_of_all_plots_equal_26_hectares_l397_397452


namespace perpendicular_lines_b_value_l397_397473

theorem perpendicular_lines_b_value :
  ( ∀ x y : ℝ, 2 * x + 3 * y + 4 = 0)  →
  ( ∀ x y : ℝ, b * x + 3 * y - 1 = 0) →
  ( - (2 : ℝ) / (3 : ℝ) * - b / (3 : ℝ) = -1 ) →
  b = - (9 : ℝ) / (2 : ℝ) :=
by
  intros h1 h2 h3
  sorry

end perpendicular_lines_b_value_l397_397473


namespace travel_distance_l397_397161

theorem travel_distance (speed time : ℕ) (h_speed : speed = 20) (h_time : time = 8) : 
  ∃ distance : ℕ, distance = speed * time ∧ distance = 160 :=
by {
  use (speed * time),
  split,
  { 
    sorry,
  },
  {
    sorry,
  }
}

end travel_distance_l397_397161


namespace max_z_plus_reciprocal_l397_397496

noncomputable def sum_of_numbers : ℕ := 2024
noncomputable def sum_of_reciprocals : ℕ := 2024
noncomputable def quantity_of_numbers : ℕ := 2023

theorem max_z_plus_reciprocal (numbers : Fin quantity_of_numbers → ℝ)
  (h1 : ∀ i, numbers i > 0)
  (h2 : (∑ i, numbers i) = sum_of_numbers)
  (h3 : (∑ i, (numbers i)⁻¹) = sum_of_reciprocals)
  (z : ℝ)
  (hz : z ∈ set.range numbers) :
  z + (1/z) ≤ (8129049/2024) := by
  sorry

end max_z_plus_reciprocal_l397_397496


namespace sum_of_prime_factors_1722_l397_397949

theorem sum_of_prime_factors_1722 : 
  (∑ p in (Multiset.eraseDup (Multiset.filter Nat.Prime (Nat.factors 1722))), p) = 53 := by
  sorry

end sum_of_prime_factors_1722_l397_397949


namespace cone_slant_height_l397_397694

theorem cone_slant_height {r : ℝ} (r_cone : r = 1) (r_sphere : r = 1) (vol_eq : (1 / 3) * π * r^2 * 4 = (4 / 3) * π * r^3) :
  real.sqrt (r^2 + 4^2) = real.sqrt 17 := 
by 
  let r = 1 
  have h : 4 = 4 := by rfl  -- intermediate steps omitted
  sorry

end cone_slant_height_l397_397694


namespace contrapositive_of_original_l397_397843

theorem contrapositive_of_original (a b : ℝ) :
  (a > b → a - 1 > b - 1) ↔ (a - 1 ≤ b - 1 → a ≤ b) :=
by
  sorry

end contrapositive_of_original_l397_397843


namespace remainder_of_sum_mod_11_l397_397651

theorem remainder_of_sum_mod_11 :
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 :=
by
  sorry

end remainder_of_sum_mod_11_l397_397651


namespace smallest_positive_period_of_f_monotonically_increasing_interval_of_f_range_of_f_l397_397298

section

variable (x : ℝ)

def m : ℝ × ℝ := (Real.cos x, -1)
def n : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos x ^ 2)
noncomputable def f : ℝ := (m.1 * n.1 + m.2 * n.2) + (1/2 : ℝ)

theorem smallest_positive_period_of_f :
    ∃ T > 0, ∀ x, f(2 * x - π / 6) = f(x + T) := sorry

theorem monotonically_increasing_interval_of_f :
    ∀ k : ℤ, ∀ x : ℝ, f x = Real.sin (2*x - π / 6) ∧ 
    k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3 := sorry

theorem range_of_f :
    ∀ (x : ℝ), 0 < x ∧ x < π / 2 → - (1 / 2) ≤ f x ∧ f x ≤ 1 := sorry

end

end smallest_positive_period_of_f_monotonically_increasing_interval_of_f_range_of_f_l397_397298


namespace greatest_possible_value_of_x_l397_397104

theorem greatest_possible_value_of_x :
  ∃ (x : ℚ), x = 9 / 5 ∧ 
  (\left(5 * x - 20) / (4 * x - 5)) ^ 2 + \left((5 * x - 20) / (4 * x - 5)) = 20 ∧ x ≥ 0 :=
begin
  existsi (9 / 5 : ℚ),
  split,
  { refl },
  split,
  { sorry },
  { sorry }
end

end greatest_possible_value_of_x_l397_397104


namespace tan_double_alpha_alpha_plus_beta_l397_397664

variables (α β : ℝ)

-- Conditions
axiom tan_alpha_eq : tan α = 1 / 3
axiom cos_beta_eq : cos β = sqrt 5 / 5
axiom alpha_range : 0 < α ∧ α < π / 2
axiom beta_range : 3 * π / 2 < β ∧ β < 2 * π

-- Statements to prove
-- 1. Prove that tan(2*α) = 3/4
theorem tan_double_alpha : tan (2 * α) = 3 / 4
  := sorry

-- 2. Prove that α + β = 7 * π / 4
theorem alpha_plus_beta : α + β = 7 * π / 4
  := sorry

end tan_double_alpha_alpha_plus_beta_l397_397664


namespace numbers_on_board_l397_397025

theorem numbers_on_board (x : ℕ → ℕ) (k : ℕ) (n : ℕ) (h_pairwise : ∀ (i j : ℕ), i ≠ j → x i ≠ x j)
  (h_sorted : ∀ (i j : ℕ), i < j → x i < x j) (h_min : x 0 = 13 * k) (h_max : x (n-1) = 31 * k) :
  (477 = 32 * x 0 + (∑ i in finset.range (n-1), x i) ∧ 
   477 = x 0 + 14 * x (n-1) + (∑ i in finset.range (n-1), x i)) →
  (set.range x = {13, 14, 16, 31} ∨ set.range x = {13, 30, 31}) :=
by
  sorry

end numbers_on_board_l397_397025


namespace restaurant_total_spent_l397_397444

theorem restaurant_total_spent (appetizer_cost : ℕ) (entree_cost : ℕ) (num_entrees : ℕ) (tip_rate : ℚ) 
  (H1 : appetizer_cost = 10) (H2 : entree_cost = 20) (H3 : num_entrees = 4) (H4 : tip_rate = 0.20) :
  appetizer_cost + num_entrees * entree_cost + (appetizer_cost + num_entrees * entree_cost) * tip_rate = 108 :=
by
  sorry

end restaurant_total_spent_l397_397444


namespace quadrilateral_area_is_7_5_l397_397178

open Real

def area_of_irregular_quadrilateral (v1 v2 v3 v4: ℝ × ℝ) : ℝ :=
  abs((v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v4.2 + v4.1 * v1.2) -
      (v1.2 * v2.1 + v2.2 * v3.1 + v3.2 * v4.1 + v4.2 * v1.1)) / 2

theorem quadrilateral_area_is_7_5 : 
  let v1 := (2, 1)
  let v2 := (4, 3)
  let v3 := (7, 1)
  let v4 := (4, 6)
  area_of_irregular_quadrilateral v1 v2 v3 v4 = 7.5 :=
by
  let v1 := (2, 1)
  let v2 := (4, 3)
  let v3 := (7, 1)
  let v4 := (4, 6)
  show area_of_irregular_quadrilateral v1 v2 v3 v4 = 7.5 from 
    sorry

end quadrilateral_area_is_7_5_l397_397178


namespace time_for_one_large_division_l397_397490

/-- The clock face is divided into 12 equal parts by the 12 numbers (12 large divisions). -/
def num_large_divisions : ℕ := 12

/-- Each large division is further divided into 5 small divisions. -/
def num_small_divisions_per_large : ℕ := 5

/-- The second hand moves 1 small division every second. -/
def seconds_per_small_division : ℕ := 1

/-- The time it takes for the second hand to move one large division is 5 seconds. -/
def time_per_large_division : ℕ := num_small_divisions_per_large * seconds_per_small_division

theorem time_for_one_large_division : time_per_large_division = 5 := by
  sorry

end time_for_one_large_division_l397_397490


namespace common_tangent_line_l397_397323

def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := x^2 + a * x

theorem common_tangent_line (a : ℝ) :
  (∃ (x1 x2 : ℝ), (f' x1 = g' x2) ∧ (f x1 = g x2) ∧ (f' x1 = 1)) →
  (a = 3 ∨ a = -1) :=
begin
  sorry
end

end common_tangent_line_l397_397323


namespace equal_likelihood_of_lucky_sums_solution_l397_397358

variables (N : ℕ)

def main_draw := 10
def additional_draw := 8

def lucky_sum_main (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = main_draw ∧ S.sum = sum
def lucky_sum_additional (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = additional_draw ∧ S.sum = sum

theorem equal_likelihood_of_lucky_sums :
  lucky_sum_main N 63 ↔ lucky_sum_additional N 44 :=
sorry

theorem solution :
  equal_likelihood_of_lucky_sums 18 :=
sorry

end equal_likelihood_of_lucky_sums_solution_l397_397358


namespace find_curve_C_equation_exist_pos_m_l397_397667

noncomputable def curve_C_equation (x y : ℝ) : Prop :=
  y^2 = 4 * x ∧ x > 0

def condition_dist_diff (x y : ℝ) : Prop :=
  abs (√ ((x-1)^2 + y^2) - x) = 1

def point_on_curve (x y : ℝ) : Prop :=
  ∃ (x y : ℝ), condition_dist_diff x y → curve_C_equation x y

theorem find_curve_C_equation :
  ∀ (x y : ℝ), condition_dist_diff x y → curve_C_equation x y :=
sorry

noncomputable def intersection_points (m t : ℝ) (x1 x2 y1 y2 : ℝ) : Prop :=
  y1^2 - 4 * t * y1 - 4 * m = 0 ∧
  y2^2 - 4 * t * y2 - 4 * m = 0 ∧
  x1 = y1^2 / 4 ∧
  x2 = y2^2 / 4

def fa_fb_dot_product_neg (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 - 1) * (x2 - 1) + y1 * y2 < 0

theorem exist_pos_m :
  ∃ (m : ℝ), 3 - 2 * real.sqrt 2 < m ∧ m < 3 + 2 * real.sqrt 2 ∧
  ∀ (t x1 x2 y1 y2 : ℝ), intersection_points m t x1 x2 y1 y2 → fa_fb_dot_product_neg x1 y1 x2 y2 :=
sorry

end find_curve_C_equation_exist_pos_m_l397_397667


namespace automotive_test_l397_397126

noncomputable def total_distance (D : ℝ) (t : ℝ) : ℝ := 3 * D

theorem automotive_test (D : ℝ) (h_time : (D / 4 + D / 5 + D / 6 = 37)) : total_distance D 37 = 180 :=
  by
    -- This skips the proof, only the statement is given
    sorry

end automotive_test_l397_397126


namespace number_of_digits_l397_397211

theorem number_of_digits (x : ℕ) (y : ℕ) (z : ℕ) (h : x = 2 ∧ y = 15 ∧ 5 = 5 ∧ z = 3) :
    (natDigits 10 (x ^ 15 * 5 ^ 10 * z)).length = 12 := by sorry

end number_of_digits_l397_397211


namespace base_8_to_base_10_l397_397621

theorem base_8_to_base_10 (n : ℕ) (h : n = 2453) : 
  (2 * 8^3 + 4 * 8^2 + 5 * 8^1 + 3 * 8^0) = 1323 :=
by
  rw h
  exact rfl

end base_8_to_base_10_l397_397621


namespace probability_both_multiples_of_2_l397_397240

open Finset

def set_1234 : Finset ℕ := {1, 2, 3, 4}

def multiples_of_2 (s : Finset ℕ) : Finset ℕ :=
  s.filter (fun x => x % 2 = 0)

theorem probability_both_multiples_of_2 :
  let chosen_pairs := set_1234.pairs
  let chosen_pairs_of_2 := (multiples_of_2 set_1234).pairs
  (chosen_pairs.card) = 6 →
  (chosen_pairs_of_2.card) = 1 →
  (1 : ℚ) / (6 : ℚ) = (chosen_pairs_of_2.card) / (chosen_pairs.card) :=
by sorry

end probability_both_multiples_of_2_l397_397240


namespace sum_of_possible_b_l397_397422

theorem sum_of_possible_b (b : ℤ) :
  (∀ r s : ℤ, r * s = 48 ∧ r + s = -b → r < 0 ∧ s < 0 ∧ r ≠ s) →
  (∑ b in {49, 26, 19, 16, 14}, b) = 124 :=
by
  sorry

end sum_of_possible_b_l397_397422


namespace rooms_needed_l397_397988

-- Defining all necessary conditions
constant fans : ℕ := 100
constant teams : ℕ := 3
constant max_people_per_room : ℕ := 3
constant groups : ℕ := 6

-- Proposition that the number of rooms required is exactly 37.
theorem rooms_needed (h1 : ∀ i, 1 ≤ i ∧ i ≤ groups) 
  (h2 : groups = 2 * teams) 
  (h3 : teams = 3)
  (h4 : max_people_per_room = 3)
  (h5 : ∃ i, 0 ≤ i ∧ ∑ i in finset.range groups, i = fans):
  37 = (∃ (n : ℕ), ∃ (i : ℕ), n = ∑ i in finset.range groups, nat.ceil (i / max_people_per_room)):
sorry

end rooms_needed_l397_397988


namespace problem1_problem2_l397_397259

variables (a : ℕ+ → ℝ)
variables (b : ℕ+ → ℝ)
variables (c : ℕ+ → ℝ)
variables (S : ℕ+ → ℝ)
variable (λ : ℝ)

-- For Problem 1
theorem problem1 (h1 : ∀ n : ℕ+, a n = a 1 + 2 * (n - 1))
  (hb : ∀ n : ℕ+, (n + 1) * b n = a (n + 1) - S n / n)
  (hc : ∀ n : ℕ+, (n + 2) * c n = (a (n + 1) + a (n + 2)) / 2 - S n / n) :
  ∀ n : ℕ+, c n = 1 :=
sorry

-- For Problem 2
theorem problem2
  (hb : ∀ n : ℕ+, (n + 1) * b n = a (n + 1) - S n / n)
  (hc : ∀ n : ℕ+, (n + 2) * c n = (a (n + 1) + a (n + 2)) / 2 - S n / n)
  (hλ : ∀ n : ℕ+, b n ≤ λ ∧ λ ≤ c n) :
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d :=
sorry

end problem1_problem2_l397_397259


namespace min_rooms_needed_l397_397998

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l397_397998


namespace area_of_triangle_ABC_l397_397334

theorem area_of_triangle_ABC :
  ∀ (A B C: Type) (a b c: ℝ),
  AC = √7 ∧ BC = 2 ∧ ∠ABC = 60 → area_of_triangle_ABC = (3 * sqrt 3) / 2 :=
begin
  sorry,
end

end area_of_triangle_ABC_l397_397334


namespace functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l397_397559

-- Definitions for the problem conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def max_selling_price : ℝ := 38
def base_sales_volume : ℝ := 250
def price_decrease_effect : ℝ := 10
def profit_requirement : ℝ := 2000

-- Given the initial conditions
noncomputable def sales_volume (x : ℝ) : ℝ := base_sales_volume - price_decrease_effect * (x - min_selling_price)

-- Target problem statement
-- Part 1: Functional relationship between y and x
theorem functional_relationship (x : ℝ) : sales_volume x = -10 * x + 500 := by
sorry

-- Part 2: Maximizing profit
noncomputable def profit (x : ℝ) : ℝ := (x - cost_per_box) * sales_volume x

theorem maximizing_profit : ∃ (x : ℝ), x = 35 ∧ profit x = 2250 := by
sorry

-- Part 3: Minimum number of boxes to sell for at least 2000 yuan profit
theorem minimum_boxes_for_2000_profit (x : ℝ) : x ≤ max_selling_price → profit x ≥ profit_requirement → sales_volume x ≥ 120 := by
sorry

end functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l397_397559


namespace sqrt_two_precision_l397_397861

theorem sqrt_two_precision 
  (x y : ℝ) 
  (sqrt2 : ℝ) 
  (h1 : sqrt2 * x + 8.59 * y = 9.98) 
  (h2 : 1.41 * x + 8.59 * y = 10)
  (sqrt2_error_bound : |sqrt2 - Real.sqrt 2| < 10^(-5)) :
  sqrt2 ≈ 1.41421 :=
by
  -- Proof skipped
  sorry

end sqrt_two_precision_l397_397861


namespace bottom_right_not_divisible_by_2011_l397_397424

section Problem

-- Define the size of the board
def n : ℕ := 2012

-- Define the function A for the numbers on the board
def A : ℕ × ℕ → ℤ
| (0, c) := 1  -- All numbers in the upper row are 1's
| (r, 0) := 1  -- All numbers in the leftmost column are 1's
| (r, c) := if r + c = n - 1 && r ≠ 0 && c ≠ n - 1 && r ≠ n - 1 && c ≠ 0 then 0 else A (r - 1, c) + A (r, c - 1)

theorem bottom_right_not_divisible_by_2011 :
  A (2011, 2011) % 2011 ≠ 0 := 
sorry

end Problem

end bottom_right_not_divisible_by_2011_l397_397424


namespace part1_part2_l397_397245

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.cos x

theorem part1 (a : ℝ) :
  let f' (x : ℝ) := deriv (λ x, Real.exp x + a * Real.cos x)
  (f 0 = 1 + a) ∧ (f' 0 = 1) ∧ (∃ y, y = f' 0 ∧ y = 6 - (f 1 - f 0) ) → a = 4 :=
by {
  -- Skip the proof
  sorry
}

theorem part2 (a : ℝ) :
  (∀ x ∈ set.Icc 0 (Real.pi / 2), f a x ≥ a * x) →
  -1 ≤ a ∧ a ≤ (2 * Real.exp (Real.pi / 2)) / Real.pi :=
by {
  -- Skip the proof
  sorry
}

end part1_part2_l397_397245


namespace find_number_l397_397540

theorem find_number (number : ℝ) (h : 0.75 / 100 * number = 0.06) : number = 8 := 
by
  sorry

end find_number_l397_397540


namespace compare_negative_values_l397_397611

-- Conditions
def abs_neg_3_14 := |(-3.14)| = 3.14
def abs_neg_pi := |(-π)| = π
def pi_approximation := π ≈ 3.14159

-- Theorem to prove
theorem compare_negative_values 
  (h1: abs_neg_3_14)
  (h2: abs_neg_pi)
  (h3: pi_approximation) : -3.14 > -π :=
by
  sorry

end compare_negative_values_l397_397611


namespace delta_polynomial_degree_l397_397432

-- Define a polynomial of degree m+1
def is_polynomial_of_degree (Q : ℝ → ℝ) (n : ℕ) : Prop :=
  ∃ (a : Fin n → ℝ), ∀ x, Q x = ∑ i in Finset.range n, (a i) * (x^i)

def polynomial_degree (Q : ℝ → ℝ) : ℕ :=
  Inf {n | is_polynomial_of_degree Q (n + 1)}

-- Define the difference operator
def delta (Q : ℝ → ℝ) (x : ℝ) : ℝ :=
  Q (x + 1) - Q x

theorem delta_polynomial_degree (Q : ℝ → ℝ) (m : ℕ) 
  (h : polynomial_degree Q = m + 1) : 
  polynomial_degree (delta Q) = m :=
  sorry

end delta_polynomial_degree_l397_397432


namespace total_biking_time_is_correct_l397_397502

def distance_time (distance speed: ℝ) : ℝ := distance / speed

def monday_time : ℝ :=
  (distance_time 20 25) + (distance_time 22 23)

def tuesday_time : ℝ :=
  (distance_time 18 20) + (distance_time 20 30)

def wednesday_time : ℝ :=
  (distance_time 22 26) + (distance_time 20 24)

def thursday_time : ℝ :=
  (distance_time 18 22) + (distance_time 22 28)

def friday_time : ℝ :=
  (distance_time 20 25) + (distance_time 18 27)

def saturday_time : ℝ :=
  distance_time 100 15

def sunday_time : ℝ :=
  distance_time 100 35

def total_week_time : ℝ :=
  monday_time + tuesday_time + wednesday_time + thursday_time + friday_time + saturday_time + sunday_time

theorem total_biking_time_is_correct :
  total_week_time ≈ 16.7 :=
by
  unfold total_week_time
  unfold monday_time tuesday_time wednesday_time thursday_time friday_time saturday_time sunday_time
  unfold distance_time
  -- use calc or other tactics to simplify and approximate the value
  sorry

end total_biking_time_is_correct_l397_397502


namespace brady_total_earnings_l397_397605

def basic_rate := 0.70
def gourmet_rate := 0.90
def advanced_rate := 1.10

def tier_bonus (n : ℕ) : ℕ :=
  if n >= 200 then 40
  else if n >= 150 then 30
  else if n >= 100 then 20
  else if n >= 50 then 10
  else 0

def total_earnings 
  (basic_cards : ℕ) (gourmet_cards : ℕ) (advanced_cards : ℕ) (total_cards : ℕ) : ℝ :=
    basic_cards * basic_rate + gourmet_cards * gourmet_rate + advanced_cards * advanced_rate + tier_bonus total_cards

theorem brady_total_earnings :
  total_earnings 110 60 40 210 = 215 := 
by
  -- Proof steps here
  sorry

end brady_total_earnings_l397_397605


namespace parallel_condition_sufficient_not_necessary_l397_397420

variables (x : ℝ)
def a : ℝ × ℝ := (2, x - 1)
def b : ℝ × ℝ := (x + 1, 4)

theorem parallel_condition_sufficient_not_necessary :
  x = 3 → (a.1 / b.1 = a.2 / b.2) :=
sorry

end parallel_condition_sufficient_not_necessary_l397_397420


namespace price_of_second_variety_l397_397047

-- Defining the problem conditions
def first_variety := 126
def third_variety := 175.5
def mixture_price := 153
def total_weight := 4
def total_cost := 612

-- The theorem to prove the price of the second variety
theorem price_of_second_variety (second_variety : ℝ) 
  (h1 : first_variety + second_variety + 2 * third_variety = total_cost) :
  second_variety = 135 := by
    -- Given h1 simplifies as:
    have h2 : 126 + second_variety + 2 * 175.5 = 612, from h1
    -- Further simplifications and steps to be done in the proof
    sorry

end price_of_second_variety_l397_397047


namespace average_of_middle_three_l397_397458

-- Let A be the set of five different positive whole numbers
def avg (lst : List ℕ) : ℚ := (lst.foldl (· + ·) 0 : ℚ) / lst.length
def maximize_diff (lst : List ℕ) : ℕ := lst.maximum' - lst.minimum'

theorem average_of_middle_three 
  (A : List ℕ) 
  (h_diff : maximize_diff A) 
  (h_avg : avg A = 5) 
  (h_len : A.length = 5) 
  (h_distinct : A.nodup) : 
  avg (A.erase_max.erase_min) = 3 := 
sorry

end average_of_middle_three_l397_397458


namespace find_abc_sum_l397_397907

-- Definitions and conditions
def radius : ℝ := 6
def height : ℝ := 8
def arc_angle_deg : ℝ := 120

-- Derive expected area
noncomputable def area_unpainted_face : ℝ := 30 * Real.sqrt 3 + 20 * Real.pi

-- Define integers a, b, and c such that area = a * pi + b * sqrt c
def a : ℕ := 20
def b : ℕ := 30
def c : ℕ := 3

-- Proof statement
theorem find_abc_sum : a + b + c = 53 :=
by
  -- Placeholder for actual proof
  sorry

end find_abc_sum_l397_397907


namespace part1_part2_l397_397157

variables (a k1 k2 : ℝ)
def A : ℝ × ℝ := (a, -1)
def parabola := {p : ℝ × ℝ | p.snd = p.fst ^ 2}
def tangent_slope (k : ℝ) (a : ℝ) := k ≠ 0 ∧ ∃ y, y = k * a
def is_tangent (P : ℝ × ℝ) :=
  ∃ k, tangent_slope k P.fst ∧ P.snd = P.fst ^ 2

theorem part1 (AP AQ : ℝ × ℝ) :
  is_tangent AP → is_tangent AQ → (k1 = (AP.snd - A.snd) / (AP.fst - A.fst)) → 
  (k2 = (AQ.snd - A.snd) / (AQ.fst - A.fst)) → k1 * k2 = -4 :=
by
  sorry

theorem part2 (AP AQ : ℝ × ℝ) :
  is_tangent AP → is_tangent AQ →
  (AP.fst, AP.snd) ≠ (AQ.fst, AQ.snd) →
  ∃ P Q : ℝ × ℝ, is_tangent P ∧ is_tangent Q ∧ 
  (A.fst = P.fst ∨ A.fst = Q.fst) → 
  ∃ (fixed_point : ℝ × ℝ), fixed_point = (0, 1) ∧
  (fixed_point.snd - AP.snd) / (fixed_point.fst - AP.fst) =
  (fixed_point.snd - AQ.snd) / (fixed_point.fst - AQ.fst) :=
by
  sorry

end part1_part2_l397_397157


namespace factorization_problem_l397_397596

-- Defining the conditions
def condA := (6 * a * b = 2 * a * 3 * b)
def condB := ((x + 5) * (x - 2) = x^2 + 3 * x - 10)
def condC := (x^2 - 8*x + 16 = (x - 4)^2)
def condD := (x^2 - 9 + 6*x = (x - 3) * (x + 3) + 6*x)

-- The problem statement
theorem factorization_problem : condC := 
  sorry

end factorization_problem_l397_397596


namespace lcm_of_4_5_6_9_is_180_l397_397579

theorem lcm_of_4_5_6_9_is_180 : Nat.lcm (Nat.lcm 4 5) (Nat.lcm 6 9) = 180 :=
by
  sorry

end lcm_of_4_5_6_9_is_180_l397_397579


namespace angle_BDC_l397_397538

-- Definitions based on given conditions in the problem
def right_triangle (A B C : Point) : Prop :=
  ∠ A C B = 90 ∧ ∠ A = 30

def angle_bisector (A B C D : Point) : Prop :=
  D ∈ line_segment A C ∧ ∠ DBC = ∠ ABD / 2

-- Theorem statement matching the mathematically equivalent proof problem
theorem angle_BDC (A B C D : Point) (h1 : right_triangle A B C) (h2 : angle_bisector A B C D) : 
  ∠ BDC = 60 :=
sorry

end angle_BDC_l397_397538


namespace inverse_of_f_l397_397008

-- Definition of the function f
def f (x : ℝ) : ℝ := arccos x + 2 * arcsin x

-- Statement of the problem: what is the inverse function of f
theorem inverse_of_f : ∀ x, 0 ≤ x ∧ x ≤ π → f⁻¹ x = -cos x := 
sorry

end inverse_of_f_l397_397008


namespace solve_for_x_l397_397814

open Real

-- Define the condition and the target result
def target (x : ℝ) : Prop :=
  sqrt (9 + sqrt (16 + 3 * x)) + sqrt (3 + sqrt (4 + x)) = 3 + 3 * sqrt 2

theorem solve_for_x (x : ℝ) (h : target x) : x = 8 * sqrt 2 / 3 :=
  sorry

end solve_for_x_l397_397814


namespace mean_of_combined_set_l397_397838

-- Define the conditions as Lean definitions
def mean_set_one (s₁ : Set ℝ) (h₁ : ∀ x ∈ s₁, x ∈ [16]) : Prop :=
  s₁.card = 5 ∧ s₁.mean = 16

def mean_set_two (s₂ : Set ℝ) (h₂ : ∀ x ∈ s₂, x ∈ [21]) : Prop :=
  s₂.card = 8 ∧ s₂.mean = 21

-- The theorem stating the problem and its conditions
theorem mean_of_combined_set :
  ∃ s₁ s₂ s₃ : Set ℝ,
    mean_set_one s₁ ∧ mean_set_two s₂ ∧
    s₃ = s₁ ∪ s₂ ∧
    s₃.card = 13 ∧ s₃.mean = 19.08 :=
begin
  sorry,
end

end mean_of_combined_set_l397_397838


namespace circle_relationship_l397_397697

noncomputable def f : ℝ × ℝ → ℝ := sorry

variables {x y x₁ y₁ x₂ y₂ : ℝ}
variables (h₁ : f (x₁, y₁) = 0) (h₂ : f (x₂, y₂) ≠ 0)

theorem circle_relationship :
  f (x, y) - f (x₁, y₁) - f (x₂, y₂) = 0 ↔ f (x, y) = f (x₂, y₂) :=
sorry

end circle_relationship_l397_397697


namespace max_traffic_volume_and_speed_l397_397862
# Noncomputable theory typically needed for analytic contexts

def traffic_speed (x : ℝ) : ℝ :=
if (0 ≤ x ∧ x ≤ 30) then 60 else 
if (30 < x ∧ x ≤ 210) then (-1/3) * x + 70 else 0

def traffic_volume (x : ℝ) : ℝ :=
x * traffic_speed x

theorem max_traffic_volume_and_speed :
  ∃ x_max : ℝ, 
  (x_max = 105) ∧ (traffic_volume x_max = 3675) ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 210 → traffic_volume x ≤ 3675) :=
by sorry

end max_traffic_volume_and_speed_l397_397862


namespace minimum_rooms_to_accommodate_fans_l397_397981

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l397_397981


namespace car_speed_ratio_l397_397601

theorem car_speed_ratio 
  (t D : ℝ) 
  (v_alpha v_beta : ℝ)
  (H1 : (v_alpha + v_beta) * t = D)
  (H2 : v_alpha * 4 = D - v_alpha * t)
  (H3 : v_beta * 1 = D - v_beta * t) : 
  v_alpha / v_beta = 2 :=
by
  sorry

end car_speed_ratio_l397_397601


namespace sum_infinite_series_l397_397780

noncomputable def p : ℕ → (ℝ → ℝ)
| 0 := λ x, x
| (n+1) := λ x, ∫ t in 0..x, p n t 

theorem sum_infinite_series :
  let p_n : ℕ → (ℝ → ℝ) := λ n x, p n x in
  ∑' n, p_n n 2009 = 2009 * (Real.exp 2009 - 1) :=
by
  sorry

end sum_infinite_series_l397_397780


namespace count_valid_x_l397_397841

open BigOperators

def diamond (a b : ℕ) : ℕ :=
a^2 - b

theorem count_valid_x :
  {x : ℕ // (∃ (k : ℕ), 20^2 - x = k^2) ∧ x > 0}.card = 19 :=
by sorry

end count_valid_x_l397_397841


namespace vector_addition_l397_397661

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 5)

-- State the theorem that we want to prove
theorem vector_addition : a + 3 • b = (-1, 18) :=
  sorry

end vector_addition_l397_397661


namespace max_x_possible_value_l397_397102

theorem max_x_possible_value : ∃ x : ℚ, 
  (∃ y : ℚ, y = (5 * x - 20) / (4 * x - 5) ∧ (y^2 + y = 20)) ∧
  x = 9 / 5 :=
begin
  sorry
end

end max_x_possible_value_l397_397102


namespace right_triangle_legs_l397_397348

theorem right_triangle_legs (a b c : ℝ) 
  (h : ℝ) 
  (h_h : h = 12) 
  (h_perimeter : a + b + c = 60) 
  (h1 : a^2 + b^2 = c^2) 
  (h_altitude : h = a * b / c) :
  (a = 15 ∧ b = 20) ∨ (a = 20 ∧ b = 15) :=
by
  sorry

end right_triangle_legs_l397_397348


namespace quilt_shaded_fraction_l397_397615

/-- Prove that the fraction of the 4x4 quilt that is shaded is 1/2 given the conditions. -/
theorem quilt_shaded_fraction :
  ∃ (fraction_shaded : ℚ),
    (let quilt := 4 * 4 in
     let shaded_whole_squares := 4 in
     let shaded_triangles := 8 in
     let area_of_whole_squares := shaded_whole_squares in
     let area_of_triangles := (shaded_triangles * (1 / 2 : ℚ)) in
     let total_shaded_area := area_of_whole_squares + area_of_triangles in
     fraction_shaded = total_shaded_area / quilt) ∧ fraction_shaded = 1 / 2 :=
begin
  sorry
end

end quilt_shaded_fraction_l397_397615


namespace value_of_s_l397_397316

-- Conditions: (m - 8) is a factor of m^2 - sm - 24

theorem value_of_s (s : ℤ) (m : ℤ) (h : (m - 8) ∣ (m^2 - s*m - 24)) : s = 5 :=
by
  sorry

end value_of_s_l397_397316


namespace a_runs_4_times_faster_than_b_l397_397911

theorem a_runs_4_times_faster_than_b (v_A v_B : ℝ) (k : ℝ) 
    (h1 : v_A = k * v_B) 
    (h2 : 92 / v_A = 23 / v_B) : 
    k = 4 := 
sorry

end a_runs_4_times_faster_than_b_l397_397911


namespace antonio_and_maria_same_height_l397_397421

-- Definitions of people and heights
variables {L M A J : ℝ}

-- Conditions
def condition1 : Prop := A < L
def condition2 : Prop := J < A
def condition3 : Prop := M < L
def condition4 : Prop := J < M

-- Statement to prove Antônio and Maria have the same height
theorem antonio_and_maria_same_height (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : A = M :=
sorry

end antonio_and_maria_same_height_l397_397421


namespace maximum_rectangles_in_stepwise_triangle_l397_397023

theorem maximum_rectangles_in_stepwise_triangle (n : ℕ) (h : n = 6) : 
  let num_rectangles := ∑ i in range (n+1), ∑ j in range (n+1-i), (i * (i + 1) * j * (j + 1)) / 4 in
  num_rectangles = 126 :=
by
  sorry

end maximum_rectangles_in_stepwise_triangle_l397_397023


namespace angle_bisector_TAD_l397_397781

noncomputable section
open Classical

variables {A B C D X Y P M T : Type}
variables [HilbertSpace A]

-- Conditions
variables (ABC : Triangle) (right_angle_A : right_angle A B C)
variables (D_midpoint_BC : midpoint D B C)
variables (line_D_XY : line_through D ≠ line X Y)
variables (M_midpoint_PD_XY : midpoint M P D ∧ midpoint M X Y)
variables (T_perpendicular_BC : ∀ T, perp BC P T)

-- Theorem to prove
theorem angle_bisector_TAD (h : right_triangle ABC right_angle_A) 
  (h1 : midpoint D B C D_midpoint_BC) 
  (h2 : line_through D X Y line_D_XY) 
  (h3 : midpoint M P D ∧ midpoint M X Y M_midpoint_PD_XY)
  (h4 : ∀ T, perp BC P T T_perpendicular_BC) :
  is_angle_bisector AM (angle_at T A D) := 
sorry

end angle_bisector_TAD_l397_397781


namespace unique_intersection_point_l397_397659

theorem unique_intersection_point {c : ℝ} :
  (∀ x y : ℝ, y = |x - 20| + |x + 18| → y = x + c → (x = 20 ∧ y = 38)) ↔ c = 18 :=
by
  sorry

end unique_intersection_point_l397_397659


namespace goods_train_length_l397_397566

noncomputable def speed_kmh : ℕ := 72  -- Speed of the goods train in km/hr
noncomputable def platform_length : ℕ := 280  -- Length of the platform in meters
noncomputable def time_seconds : ℕ := 26  -- Time taken to cross the platform in seconds
noncomputable def speed_mps : ℤ := speed_kmh * 1000 / 3600 -- Speed of the goods train in meters/second

theorem goods_train_length : 20 * time_seconds = 280 + 240 :=
by
  sorry

end goods_train_length_l397_397566


namespace b_n_arithmetic_and_a_n_formula_T_n_sum_formula_l397_397713

-- Part 1: b_n is an arithmetic sequence and find a_n
theorem b_n_arithmetic_and_a_n_formula (a : ℕ+ → ℚ) (b : ℕ+ → ℚ)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) = 1 - 1 / (4 * a n))
  (h₃ : ∀ n, b n = 2 / (2 * a n - 1)) :
  (∀ n, b (n + 1) - b n = 2) ∧ (∀ n, a n = (n + 1) / (2 * n)) :=
by
  sorry

-- Part 2: T_n sum formula for c_n * c_(n+1)
theorem T_n_sum_formula (a : ℕ+ → ℚ) (c : ℕ+ → ℚ) (T_n : ℕ+ → ℚ)
  (h₁ : ∀ n, a n = (n + 1) / (2 * n))
  (h₂ : ∀ n, c n = 4 * a n / (n + 1))
  (h₃ : ∀ n, T_n n = (finset.range n).sum (λ k, c k * c (k + 1))) :
  ∀ n, T_n n = 4 * n / (n + 1) :=
by
  sorry

end b_n_arithmetic_and_a_n_formula_T_n_sum_formula_l397_397713


namespace woman_waits_for_man_l397_397572

noncomputable def man_speed := 5 / 60 -- miles per minute
noncomputable def woman_speed := 15 / 60 -- miles per minute
noncomputable def passed_time := 2 -- minutes

noncomputable def catch_up_time (man_speed woman_speed : ℝ) (passed_time : ℝ) : ℝ :=
  (woman_speed * passed_time) / man_speed

theorem woman_waits_for_man
  (man_speed woman_speed : ℝ)
  (passed_time : ℝ)
  (h_man_speed : man_speed = 5 / 60)
  (h_woman_speed : woman_speed = 15 / 60)
  (h_passed_time : passed_time = 2) :
  catch_up_time man_speed woman_speed passed_time = 6 := 
by
  -- actual proof skipped
  sorry

end woman_waits_for_man_l397_397572


namespace min_sum_complementary_events_l397_397317

theorem min_sum_complementary_events (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hP : (1 / y) + (4 / x) = 1) : x + y ≥ 9 :=
sorry

end min_sum_complementary_events_l397_397317


namespace unique_sequence_l397_397637

theorem unique_sequence (a : ℕ → ℤ) (b : ℕ → ℤ) :
  (∀ i, 3 ≤ i ∧ i ≤ 2008 → (a i = 1 ∨ a i = -1)) →
  (∀ i, 3 ≤ i ∧ i ≤ 2008 → (b i = 1 ∨ b i = -1)) →
  (∑ i in finset.range 2009 \ finset.range 3, a i * 2^i = 2008) →
  (∑ i in finset.range 2009 \ finset.range 3, b i * 2^i = 2008) →
  (∀ i, 3 ≤ i ∧ i ≤ 2008 → a i = b i) :=
by
  intros ha hb hsa hsb
  sorry

end unique_sequence_l397_397637


namespace rooms_needed_l397_397986

-- Defining all necessary conditions
constant fans : ℕ := 100
constant teams : ℕ := 3
constant max_people_per_room : ℕ := 3
constant groups : ℕ := 6

-- Proposition that the number of rooms required is exactly 37.
theorem rooms_needed (h1 : ∀ i, 1 ≤ i ∧ i ≤ groups) 
  (h2 : groups = 2 * teams) 
  (h3 : teams = 3)
  (h4 : max_people_per_room = 3)
  (h5 : ∃ i, 0 ≤ i ∧ ∑ i in finset.range groups, i = fans):
  37 = (∃ (n : ℕ), ∃ (i : ℕ), n = ∑ i in finset.range groups, nat.ceil (i / max_people_per_room)):
sorry

end rooms_needed_l397_397986


namespace inequalities_of_function_nonneg_l397_397004

theorem inequalities_of_function_nonneg (a b A B : ℝ)
  (h : ∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.sin (2 * θ) - B * Real.cos (2 * θ) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := sorry

end inequalities_of_function_nonneg_l397_397004


namespace circumscribed_quadrilateral_sides_l397_397089

theorem circumscribed_quadrilateral_sides (a b c d : ℕ) (h_ratio : a * 2 = b ∧ b * 3/2 = c) (h_perimeter : a + b + c + d = 24) (h_tangential : a + c = b + d) :
  {a, b, c, d} = {3, 6, 9, 6} :=
by
  sorry

end circumscribed_quadrilateral_sides_l397_397089


namespace least_number_of_people_l397_397804

-- Conditions
def first_caterer_cost (x : ℕ) : ℕ := 120 + 18 * x
def second_caterer_cost (x : ℕ) : ℕ := 250 + 15 * x

-- Proof Statement
theorem least_number_of_people (x : ℕ) (h : x ≥ 44) : first_caterer_cost x > second_caterer_cost x :=
by sorry

end least_number_of_people_l397_397804


namespace sum_max_min_cardinality_B_l397_397788

-- Definition of sets A and B with the given conditions
def A (n : ℕ) : Set ℝ := sorry -- We define A to be a set of n real numbers

def B (A : Set ℝ) : Set ℝ := {z | ∃ x y ∈ A, x ≠ y ∧ z = x * y}

-- Statement to prove the required equality
theorem sum_max_min_cardinality_B (A : Set ℝ) (hA : ∃ (n : ℕ), n = 7 ∧ A.card = n) : 
  let set_B := B A in
  set.card {s | s ∈ set_B} ≤ 21 ∧ 9 ≤ set.card {s | s ∈ set_B} → 
  (set.card {s | s ∈ set_B} = 21 ∧ set.card {s | s ∈ set_B} = 9) ∨ false :=
sorry

end sum_max_min_cardinality_B_l397_397788


namespace whichNumbersAreLessThanTheirReciprocals_l397_397884

def isLessThanReciprocal (x : ℚ) : Prop :=
  x < 1 ∧ x ≠ 0

def listOfGivenNumbers := [(-3/2 : ℚ), (-1 : ℚ), (2/3 : ℚ), (1 : ℚ), (3 : ℚ)]

theorem whichNumbersAreLessThanTheirReciprocals :
  (listOfGivenNumbers.filter isLessThanReciprocal) = [(-3/2 : ℚ), (2/3 : ℚ)] :=
by
  sorry

end whichNumbersAreLessThanTheirReciprocals_l397_397884


namespace max_perfect_squares_in_sequence_l397_397924

theorem max_perfect_squares_in_sequence (m : ℕ) (a : ℕ → ℕ)
  (h₁ : a 0 = m)
  (h₂ : ∀ n, a (n + 1) = a n ^ 5 + 487) :
  (∀ n, ∃ k, a n = k^2 → m = 9 := sorry

end max_perfect_squares_in_sequence_l397_397924


namespace cyclic_quadrilateral_prove_equal_sides_l397_397830

open EuclideanGeometry

variables {A B C D O P : Point}
variable {γ : Circle}
variable [non_empty (Quadrilateral A B C D)]

-- Definitions from the conditions
def cyclic_quadrilateral (Q : Quadrilateral A B C D) (γ : Circle): Prop :=
  ∀ {P : Point}, Circle.Inside γ P = (OnCircle γ P ∨ Q)

def diagonals_intersect (A B C D P : Point) : Prop :=
  Line (A, C).intersect (B, D) = some P

def perpendicular (O P : Point) (L : Line) : Prop :=
  Line.perp_to O P L

-- Theorem statement
theorem cyclic_quadrilateral_prove_equal_sides
  (h_cyclic : cyclic_quadrilateral (Quadrilateral.mk A B C D) γ)
  (h_diag_int : diagonals_intersect A B C D P)
  (h_perp : perpendicular O P (Line.mk B C)) :
  distance A B = distance C D :=
sorry

end cyclic_quadrilateral_prove_equal_sides_l397_397830


namespace max_handshakes_l397_397139

theorem max_handshakes (n : ℕ) (h : n = 30) : (n * (n - 1)) / 2 = 435 :=
by
  rw h
  -- The goal now reduces to proving (30 * 29) / 2 = 435
  sorry

end max_handshakes_l397_397139


namespace lucky_sum_probability_eq_l397_397375

/--
Given that there are N balls numbered from 1 to N,
where 10 balls are selected in the main draw with their sum being 63,
and 8 balls are selected in the additional draw with their sum being 44,
we need to prove that N = 18 such that the events are equally likely.
-/
theorem lucky_sum_probability_eq (N : ℕ) (h1 : ∃ (S : Finset ℕ), S.card = 10 ∧ S.sum id = 63) 
    (h2 : ∃ (T : Finset ℕ), T.card = 8 ∧ T.sum id = 44) : N = 18 :=
sorry

end lucky_sum_probability_eq_l397_397375


namespace correct_factorization_l397_397175

theorem correct_factorization :
  ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end correct_factorization_l397_397175


namespace no_real_c_l397_397849

noncomputable def seq_a : ℕ → ℝ
| 0       := 1 / 6
| (n + 1) := (seq_a n + seq_b n) * (seq_a n + seq_c n) / ((seq_a n - seq_b n) * (seq_a n - seq_c n))

noncomputable def seq_b : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := (seq_b n + seq_a n) * (seq_b n + seq_c n) / ((seq_b n - seq_a n) * (seq_b n - seq_c n))

noncomputable def seq_c : ℕ → ℝ
| 0       := 1 / 3
| (n + 1) := (seq_c n + seq_a n) * (seq_c n + seq_b n) / ((seq_c n - seq_a n) * (seq_c n - seq_b n))

noncomputable def polynomial_A (N : ℕ) (x : ℝ) : ℝ :=
(finset.range (2 * N + 1)).sum (λ k, seq_a k * x^k)

noncomputable def polynomial_B (N : ℕ) (x : ℝ) : ℝ :=
(finset.range (2 * N + 1)).sum (λ k, seq_b k * x^k)

noncomputable def polynomial_C (N : ℕ) (x : ℝ) : ℝ :=
(finset.range (2 * N + 1)).sum (λ k, seq_c k * x^k)

theorem no_real_c (N : ℕ) : ¬ ∃ c : ℝ, polynomial_A N c = 0 ∧ polynomial_B N c = 0 ∧ polynomial_C N c = 0 :=
sorry

end no_real_c_l397_397849


namespace probability_of_x_y_less_than_3_is_1_over_4_l397_397921

noncomputable def probability_of_x_y_less_than_3 : ℝ :=
  let area_triangle := (1 / 2) * 3 * 2 in
  let area_rectangle := 6 * 2 in
  area_triangle / area_rectangle

theorem probability_of_x_y_less_than_3_is_1_over_4 :
  probability_of_x_y_less_than_3 = 1 / 4 :=
by 
  unfold probability_of_x_y_less_than_3
  simp
  sorry

end probability_of_x_y_less_than_3_is_1_over_4_l397_397921


namespace positional_relationship_l397_397688

-- Declare the types for Line and Plane
variable (Line : Type) (Plane : Type)

-- Define the given conditions as hypotheses
variable (a b : Line) (β : Plane)
variable (intersects : ∀ a b : Line, Prop)
variable (parallel : ∀ l : Line, p : Plane, Prop)

-- Introduce the specific conditions for this problem
variable (intersects_a_b : intersects a b)
variable (parallel_a_beta : parallel a β)

-- State the theorem with the required proof goal
theorem positional_relationship (a b : Line) (β : Plane)
    (intersects_a_b : intersects a b) 
    (parallel_a_beta : parallel a β) : 
    (parallel b β) ∨ (∃ p : Line, intersects b p ∧ p ∈ β) :=
by sorry

end positional_relationship_l397_397688


namespace geometric_sum_eight_terms_l397_397186

theorem geometric_sum_eight_terms :
  let a0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 4
  let n := 8
  let S_n := a0 * (1 - r^n) / (1 - r)
  S_n = 65535 / 147456 := by
  sorry

end geometric_sum_eight_terms_l397_397186


namespace count_of_sets_without_perfect_squares_is_146_l397_397783

noncomputable def S_i' (i : ℕ) : set ℤ := {n | 200 * i ≤ n ∧ n < 200 * (i + 1)}

def perfect_square (n : ℤ) : Prop := ∃ k : ℤ, k * k = n

def contains_perfect_square (s : set ℤ) : Prop := ∃ n ∈ s, perfect_square n

def count_sets_without_perfect_squares : ℕ := (finset.range 500).filter (λ i, ¬ contains_perfect_square (S_i' i)).card

theorem count_of_sets_without_perfect_squares_is_146 :
  count_sets_without_perfect_squares = 146 :=
sorry

end count_of_sets_without_perfect_squares_is_146_l397_397783


namespace blood_flow_scientific_notation_l397_397048

theorem blood_flow_scientific_notation (blood_flow : ℝ) (h : blood_flow = 4900) : 
  4900 = 4.9 * (10 ^ 3) :=
by
  sorry

end blood_flow_scientific_notation_l397_397048


namespace circular_pond_area_l397_397155

theorem circular_pond_area (XY ZO : ℝ) (hXY : XY = 12) (hZO : ZO = 8) :
  let Z := XY / 2
  let XZ := Z
  let XO := real.sqrt (XZ^2 + ZO^2)
  XO = 10 ∧ π * XO^2 = 100 * π :=
by
  have hXZ : XZ = 6 := by rw [Z, hXY]; norm_num
  have hXO : XO = real.sqrt (XZ^2 + ZO^2) := by rw [real.sqrt_eq_iff', hXZ, hZO]; norm_num
  have hXO_val : XO = 10 := by linarith
  split
  · exact hXO_val
  · rw [hXO_val]; norm_num

end circular_pond_area_l397_397155


namespace initial_water_in_bucket_l397_397182

theorem initial_water_in_bucket (poured_out left_in_bucket : ℝ) (h_poured_out : poured_out = 0.2) (h_left_in_bucket : left_in_bucket = 0.6) : 
  let initial := left_in_bucket + poured_out in
  initial = 0.8 := 
by
  have h : initial = 0.6 + 0.2 := by rw [h_poured_out, h_left_in_bucket]
  have h_initial : initial = 0.8 := by norm_num
  exact h_initial

end initial_water_in_bucket_l397_397182


namespace find_savings_l397_397475

theorem find_savings (I E : ℕ) (h1 : I = 21000) (h2 : I / E = 7 / 6) : I - E = 3000 := by
  sorry

end find_savings_l397_397475


namespace coefficient_x_neg_16_in_binomial_expansion_l397_397160

theorem coefficient_x_neg_16_in_binomial_expansion
  (a : ℝ) (h_a_pos : a > 0) (h_area : ∫ x in 0..1, 2 * sqrt (a * x) = 4 / 3)
  : (Nat.choose 20 18) = 190 :=
by
  sorry

end coefficient_x_neg_16_in_binomial_expansion_l397_397160


namespace simplification_l397_397039

-- Let m be a real number with the conditions that m ≠ -1 and m ≠ -2.
variables {m : ℝ} (h₁ : m ≠ -1) (h₂ : m ≠ -2)

-- Define the expression (4m+5)/(m+1) + m - 1).
def expr1 := ((4 * m + 5) / (m + 1)) + m - 1

-- Define the entire expression (expr1) / ((m+2)/(m+1))
def expr2 := expr1 / ((m + 2) / (m + 1))

-- Prove that this expression simplifies to m+2
theorem simplification :
  expr2 = m + 2 :=
sorry

end simplification_l397_397039


namespace solve_inequality_l397_397441

noncomputable def inequality := 
  ∀ x : ℝ, 
    0 < x ∧ (x ≤ 2^(-Real.sqrt (Real.log 6) / Real.log 2) ∨ (x = 1) ∨ x ≥ 2^(Real.sqrt (Real.log 6) / Real.log 2)) → 
    (Real.Pow 4 (Real.log x^2 / (Real.log (Real.sqrt 2) / Real.log 2)))^(1 / 7) + 6 ≥ 
    x^(Real.log x / Real.log 2) + 6 * (x )^(Real.log x / Real.log 2)^(1 / 7)

theorem solve_inequality : inequality :=
  sorry

end solve_inequality_l397_397441


namespace max_min_values_monotonicity_condition_l397_397055

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * a * x + 2

theorem max_min_values (a : ℝ) (h : a = -1) :
  ∃ (x_min x_max : ℝ), x_min ∈ Icc (-5:ℝ) 5 ∧ x_max ∈ Icc (-5:ℝ) 5 ∧
  (∀ x ∈ Icc (-5:ℝ) 5, f x a ≥ f x_min a) ∧ 
  (∀ x ∈ Icc (-5:ℝ) 5, f x a ≤ f x_max a) ∧ 
  f x_min a = 1 ∧ f x_max a = 37 :=
by
  sorry

theorem monotonicity_condition (a : ℝ) :
  (∀ ⦃x₁ x₂ : ℝ⦄, x₁ ∈ Icc (-5:ℝ) 5 → x₂ ∈ Icc (-5:ℝ) 5 → x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∨
  (∀ ⦃x₁ x₂ : ℝ⦄, x₁ ∈ Icc (-5:ℝ) 5 → x₂ ∈ Icc (-5:ℝ) 5 → x₁ ≤ x₂ → f x₁ a ≥ f x₂ a) ↔
  a ≤ -5 ∨ a ≥ 5 :=
by
  sorry

end max_min_values_monotonicity_condition_l397_397055


namespace position_1011th_square_l397_397059

-- Define the initial position and transformations
inductive SquarePosition
| ABCD : SquarePosition
| DABC : SquarePosition
| BADC : SquarePosition
| DCBA : SquarePosition

open SquarePosition

def R1 (p : SquarePosition) : SquarePosition :=
  match p with
  | ABCD => DABC
  | DABC => BADC
  | BADC => DCBA
  | DCBA => ABCD

def R2 (p : SquarePosition) : SquarePosition :=
  match p with
  | ABCD => DCBA
  | DCBA => ABCD
  | DABC => BADC
  | BADC => DABC

def transform : ℕ → SquarePosition
| 0 => ABCD
| n + 1 => if n % 2 = 0 then R1 (transform n) else R2 (transform n)

theorem position_1011th_square : transform 1011 = DCBA :=
by {
  sorry
}

end position_1011th_square_l397_397059


namespace minimum_rooms_to_accommodate_fans_l397_397982

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l397_397982


namespace bob_work_days_per_week_l397_397946

theorem bob_work_days_per_week (daily_hours : ℕ) (monthly_hours : ℕ) (average_days_per_month : ℕ) (days_per_week : ℕ)
  (h1 : daily_hours = 10)
  (h2 : monthly_hours = 200)
  (h3 : average_days_per_month = 30)
  (h4 : days_per_week = 7) :
  (monthly_hours / daily_hours) / (average_days_per_month / days_per_week) = 5 := by
  -- Now we will skip the proof itself. The focus here is on the structure.
  sorry

end bob_work_days_per_week_l397_397946


namespace concentric_circles_ratio_l397_397088

theorem concentric_circles_ratio (d1 d2 d3 : ℝ) (h1 : d1 = 2) (h2 : d2 = 4) (h3 : d3 = 6) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let r3 := d3 / 2
  let A_red := π * r1 ^ 2
  let A_middle := π * r2 ^ 2
  let A_large := π * r3 ^ 2
  let A_blue := A_middle - A_red
  let A_green := A_large - A_middle
  (A_green / A_blue) = 5 / 3 := 
by
  sorry

end concentric_circles_ratio_l397_397088


namespace part_I_part_II_l397_397703

noncomputable def function_expression : ℝ → ℝ :=
  λ x, 2 * sin (2 * x - π / 6) - 1

def highest_point : Prop := (function_expression (π / 3) = 1)

def lowest_point : Prop := (function_expression (-π / 6) = -3)

def monotone_interval : Set ℝ :=
  { x | ∃ k : ℤ, (π / 3 + k * π ≤ x ∧ x ≤ -π / 6 + k * π) }

theorem part_I :
  ∀ x : ℝ, function_expression x = 2 * sin (2 * x - π / 6) - 1 ∧
  (∃ k : ℤ, (π / 3 + k * π ≤ x ∧ x ≤ -π / 6 + k * π)) := 
  sorry

def vector_dot_product (A B C : ℝ) : Prop :=
  let a := A * B
  let c := A * C
  a * c * (cos (π - B)) = - (1 / 2) * a * c

theorem part_II (A : ℝ) (B C : ℝ) (h : vector_dot_product A B C) :
  -2 < function_expression A ∧ function_expression A ≤ 1 :=
  sorry

end part_I_part_II_l397_397703


namespace average_of_middle_three_is_three_l397_397461

theorem average_of_middle_three_is_three :
  ∃ (a b c d e : ℕ), 
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (d ≠ e) ∧
    (a + b + c + d + e = 25) ∧
    (∃ (min max : ℕ), min = min a b c d e ∧ max = max a b c d e ∧ (max - min) = 14) ∧
    (a ≠ min a b c d e ∧ a ≠ max a b c d e) ∧
    (b ≠ min a b c d e ∧ b ≠ max a b c d e) ∧
    (c ≠ min a b c d e ∧ c ≠ max a b c d e) ∧
    (d ≠ min a b c d e ∧ d ≠ max a b c d e) ∧
    (e ≠ min a b c d e ∧ e ≠ max a b c d e) ∧
    (a + b + c + d + e = 25) → 
    (a ≠ min a b c d e ∧ a ≠ max a b c d e) ∧
    (b ≠ min a b c d e ∧ b ≠ max a b c d e) ∧
    (c ≠ min a b c d e ∧ c ≠ max a b c d e) →  
  ((a + b + c) / 3 = 3) :=
by
  sorry

end average_of_middle_three_is_three_l397_397461


namespace cheryl_more_eggs_than_others_l397_397968

def kevin_eggs : ℕ := 5
def bonnie_eggs : ℕ := 13
def george_eggs : ℕ := 9
def cheryl_eggs : ℕ := 56

theorem cheryl_more_eggs_than_others : cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 :=
by
  sorry

end cheryl_more_eggs_than_others_l397_397968


namespace primes_with_prime_remainders_count_l397_397306

-- Define a predicate to check for prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime remainders when divided by 12
def prime_remainders := {1, 2, 3, 5, 7, 11}

-- Function to list primes between 50 and 100
def primes_between_50_and_100 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Function to count such primes with prime remainder when divided by 12
noncomputable def count_primes_with_prime_remainder : ℕ :=
  list.count (λ n, n % 12 ∈ prime_remainders) primes_between_50_and_100

-- The theorem to state the problem in Lean
theorem primes_with_prime_remainders_count : count_primes_with_prime_remainder = 10 :=
by {
  /- proof steps to be provided here, if required. -/
 sorry
}

end primes_with_prime_remainders_count_l397_397306


namespace single_elimination_games_l397_397752

theorem single_elimination_games (n : ℕ) (h : n = 512) : ∃ g : ℕ, g = 511 := by
  sorry

end single_elimination_games_l397_397752


namespace total_cost_full_apartments_l397_397910

theorem total_cost_full_apartments :
  ∀ (total_units two_bedroom_units one_bedroom_cost two_bedroom_cost : ℕ),
    total_units = 12 →
    two_bedroom_units = 7 →
    one_bedroom_cost = 360 →
    two_bedroom_cost = 450 →
    (total_units - two_bedroom_units) * one_bedroom_cost +
    two_bedroom_units * two_bedroom_cost = 4950 :=
by
  intros total_units two_bedroom_units one_bedroom_cost two_bedroom_cost
  intro h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end total_cost_full_apartments_l397_397910


namespace minimum_rooms_to_accommodate_fans_l397_397984

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l397_397984


namespace ratio_of_radii_l397_397581

noncomputable def volume_of_truncated_cone (R r h : ℝ) : ℝ :=
  (π * h * (R^2 + R * r + r^2)) / 3

noncomputable def volume_of_sphere (s : ℝ) : ℝ :=
  (4 * π * s^3) / 3

theorem ratio_of_radii (R r : ℝ) (h : ℝ) (s : ℝ)
  (h₁ : s = Real.sqrt (R * r))
  (h₂ : volume_of_truncated_cone R r h = 3 * volume_of_sphere s) :
  R / r = (5 + Real.sqrt 21) / 2 :=
by
  sorry

end ratio_of_radii_l397_397581


namespace souvenirs_total_cost_l397_397181

theorem souvenirs_total_cost (T : ℝ) (H1 : 347 = T + 146) : T + 347 = 548 :=
by
  -- To ensure the validity of the Lean statement but without the proof.
  sorry

end souvenirs_total_cost_l397_397181


namespace total_length_of_rubber_pen_pencil_l397_397575

variable (rubber pen pencil : ℕ)

theorem total_length_of_rubber_pen_pencil 
  (h1 : pen = rubber + 3)
  (h2 : pen = pencil - 2)
  (h3 : pencil = 12) : rubber + pen + pencil = 29 := by
  sorry

end total_length_of_rubber_pen_pencil_l397_397575


namespace find_integer_l397_397638

theorem find_integer (n : ℕ) (hn1 : n % 20 = 0) (hn2 : 8.2 < (n : ℝ)^(1/3)) (hn3 : (n : ℝ)^(1/3) < 8.3) : n = 560 := sorry

end find_integer_l397_397638


namespace amy_7_mile_run_time_l397_397935

-- Define the conditions
variable (rachel_time_per_9_miles : ℕ) (amy_time_per_4_miles : ℕ) (amy_time_per_mile : ℕ) (amy_time_per_7_miles: ℕ)

-- State the conditions
def conditions : Prop :=
  rachel_time_per_9_miles = 36 ∧
  amy_time_per_4_miles = 1 / 3 * rachel_time_per_9_miles ∧
  amy_time_per_mile = amy_time_per_4_miles / 4 ∧
  amy_time_per_7_miles = amy_time_per_mile * 7

-- The main statement to prove
theorem amy_7_mile_run_time (rachel_time_per_9_miles : ℕ) (amy_time_per_4_miles : ℕ) (amy_time_per_mile : ℕ) (amy_time_per_7_miles: ℕ) :
  conditions rachel_time_per_9_miles amy_time_per_4_miles amy_time_per_mile amy_time_per_7_miles → 
  amy_time_per_7_miles = 21 := 
by
  intros h
  sorry

end amy_7_mile_run_time_l397_397935


namespace problem_1_problem_2_problem_3_l397_397902

-- Definition for question 1:
def gcd_21n_4_14n_3 (n : ℕ) : Prop := (Nat.gcd (21 * n + 4) (14 * n + 3)) = 1

-- Definition for question 2:
def gcd_n_factorial_plus_1 (n : ℕ) : Prop := (Nat.gcd (Nat.factorial n + 1) (Nat.factorial (n + 1) + 1)) = 1

-- Definition for question 3:
def fermat_number (k : ℕ) : ℕ := 2^(2^k) + 1
def gcd_fermat_numbers (m n : ℕ) (h : m ≠ n) : Prop := (Nat.gcd (fermat_number m) (fermat_number n)) = 1

-- Theorem statements
theorem problem_1 (n : ℕ) (h_pos : 0 < n) : gcd_21n_4_14n_3 n := sorry

theorem problem_2 (n : ℕ) (h_pos : 0 < n) : gcd_n_factorial_plus_1 n := sorry

theorem problem_3 (m n : ℕ) (h_pos1 : 0 ≠ m) (h_pos2 : 0 ≠ n) (h_neq : m ≠ n) : gcd_fermat_numbers m n h_neq := sorry

end problem_1_problem_2_problem_3_l397_397902


namespace quadrilateral_centers_perpendicular_l397_397791

-- Defining the setup
noncomputable def centers_of_equilateral_triangles ({A B C D : ℝ} [C AB AC BD] (h: AC = BD)) :=
let O1 := centroid (equilateral_Δ AB)
let O2 := centroid (equilateral_Δ BC)
let O3 := centroid (equilateral_Δ CD)
let O4 := centroid (equilateral_Δ DA)
in O1, O2, O3, O4

theorem quadrilateral_centers_perpendicular
    {A B C D : ℝ} 
    (h1: convex_quadrilateral A B C D)
    (h2: AC = BD) :
    let ⟨O1, O2, O3, O4⟩ := centers_of_equilateral_triangles A B C D h2
    in is_perpendicular (O3 - O1) (O4 - O2) :=
sorry

end quadrilateral_centers_perpendicular_l397_397791


namespace find_all_polynomials_l397_397098

-- Definitions for the given problem
variables {R : Type*} [CommRing R] {P : R[X]} (k : ℕ)
def satisfies_equation (P : R[X]) (k : ℕ) : Prop :=
  ∀ (x : R), P (x^k) - P (k * x) = x^k * P (x)

theorem find_all_polynomials (k : ℕ) (hk : k ≥ 1) :
  satisfies_equation R P k → P = 0 ∨ (k = 2 ∧ ∃ a : R, P = a * (X^2 - 4)) :=
sorry

end find_all_polynomials_l397_397098


namespace sin_two_alpha_eq_three_fourths_l397_397248

theorem sin_two_alpha_eq_three_fourths (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : sqrt 2 * cos (2 * α) = sin (α + π / 4)) :
  sin (2 * α) = 3 / 4 :=
by 
  sorry

end sin_two_alpha_eq_three_fourths_l397_397248


namespace proof_correct_props_l397_397592

variable (p1 : Prop) (p2 : Prop) (p3 : Prop) (p4 : Prop)

def prop1 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ (1 / 2) * x₀ < (1 / 3) * x₀
def prop2 : Prop := ∃ (x₀ : ℝ), 0 < x₀ ∧ x₀ < 1 ∧ Real.log x₀ / Real.log (1 / 2) > Real.log x₀ / Real.log (1 / 3)
def prop3 : Prop := ∀ (x : ℝ), 0 < x ∧ (1 / 2) ^ x > Real.log x / Real.log (1 / 2)
def prop4 : Prop := ∀ (x : ℝ), 0 < x ∧ x < 1 / 3 ∧ (1 / 2) ^ x < Real.log x / Real.log (1 / 3)

theorem proof_correct_props : prop2 ∧ prop4 :=
by
  sorry -- Proof goes here

end proof_correct_props_l397_397592


namespace AL_plus_BM_eq_LM_l397_397668

variable {A B C D E L M : Point}
variable (ABCD : Quadrilateral)
variable (cyclic_ABCD : cyclic ABCD)
variable (bisector_AE : bisectsAngle E (A, D))
variable (bisector_BE : bisectsAngle E (B, C))
variable (parallel_CDE : parallel (lineThrough E) (lineThrough CD))
variable (Intersect_AD_L : intersect (lineThrough E (parallelLineThrough CD) A D) L)
variable (Intersect_BC_M : intersect (lineThrough E (parallelLineThrough CD) B C) M)

theorem AL_plus_BM_eq_LM : (distance A L) + (distance B M) = distance L M :=
sorry

end AL_plus_BM_eq_LM_l397_397668


namespace problem1_problem2_l397_397901

section Problem1

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (h1 : a = b → false) -- to simulate non-collinearity
variables (h2 : ∥a∥ = 2)
variables (h3 : ∥b∥ = 2)
variables (h4 : ⟪a, b⟫ = 2) -- cos(60°) = 1/2 implies inner product is 2

theorem problem1 : ∥a - b∥ = 2 :=
sorry

end Problem1

section Problem2

variables {V : Type*} [add_comm_group V] [module ℝ V]
variables (e1 e2 : V) (h5 : (3 : ℝ) • e1 + (-4 : ℝ) • e2 ≠ (0 : V)) -- non-collinearity

variables (x y : ℝ)
variables (h6 : (3 * x - 4 * y) • e1 + (2 * x - 3 * y) • e2 = 6 • e1 + 3 • e2)

theorem problem2 : x - y = 3 :=
sorry

end Problem2

end problem1_problem2_l397_397901


namespace stats_correct_l397_397465

def data_set : list ℤ := [4, 0, 1, -2, 2, 1]

def average (s : list ℤ) : ℚ := (s.sum : ℚ) / s.length

def mode (s : list ℤ) : list ℤ :=
  let counts := s.foldl (λ m x => m.insert x (m.findD x 0 + 1)) (native.rb_map ℤ ℤ)
  let max_count := counts.fold (λ _ v max_v => max v max_v) 0
  counts.fold (λ k v acc => if v = max_count then k :: acc else acc) []

def median (s : list ℤ) : ℚ :=
  let sorted := s.qsort (≤)
  if sorted.length % 2 = 1 then
    sorted.nth_le (sorted.length / 2) sorry
  else
    let mid := sorted.length / 2
    (sorted.nth_le (mid - 1) sorry + sorted.nth_le mid sorry : ℚ) / 2

def range (s : list ℤ) : ℤ := s.maximum' - s.minimum'

def variance (s : list ℤ) : ℚ :=
  let avg := average s
  (s.foldl (λ acc x => acc + ((x - avg) * (x - avg) : ℚ)) 0) / (s.length - 1)

def standard_deviation (s : list ℤ) : ℚ := real.sqrt (variance s)

theorem stats_correct :
  average data_set = 1 ∧
  mode data_set = [1] ∧
  median data_set = 1 ∧
  range data_set = 6 ∧
  variance data_set = 4 ∧
  standard_deviation data_set = 2 :=
by
  sorry

end stats_correct_l397_397465


namespace volume_correct_l397_397784

noncomputable def volume_of_solid (a b c : ℕ) : Prop :=
  a = 343 ∧ b = 24 ∧ c = 2 ∧ Nat.coprime a b ∧ c > 0 ∧
  (∀ p : ℕ, Prime p → ¬ p^2 ∣ c) ∧ 
  (∀ (x y : ℝ), abs (x^2 - 4 * x - 5) + y ≤ 15 ∧ y - 2 * x ≥ 10 →
    volume (revolve_region (x, y) (y - 2 * x)) = (343 * Real.pi / (24 * Real.sqrt 2)))

theorem volume_correct : ∃ (a b c : ℕ), volume_of_solid a b c ∧ a + b + c = 369 :=
by
  sorry

end volume_correct_l397_397784


namespace probability_is_one_third_l397_397918

noncomputable def probability_purple_greater_than_green_but_less_than_three_times : ℝ :=
  ∫ (y : ℝ) in 0..1, (∫ (x : ℝ) in 0..1, if x < y ∧ y < 3 * x then 1 else 0) / 1

theorem probability_is_one_third :
  probability_purple_greater_than_green_but_less_than_three_times = 1 / 3 :=
sorry

end probability_is_one_third_l397_397918


namespace solution_set_f_lt_half_l397_397011

open Set Real

def f (x : ℝ) : ℝ := min (3 - x) (log x / log 2)

theorem solution_set_f_lt_half :
  { x : ℝ | f x < 1 / 2 } = { x : ℝ | (0 < x ∧ x < sqrt 2) ∨ (5 / 2 < x) } :=
by
  sorry

end solution_set_f_lt_half_l397_397011


namespace correct_option_D_option_A_false_option_B_false_option_C_false_l397_397523

-- Define variables
variables {a : ℝ}

-- Define the statements as boolean expressions
def option_A := (a^2)^3 = a^5
def option_B := (2 * a^2) / a = 2
def option_C := (2 * a)^2 = 2 * a^2
def option_D := a * a^3 = a^4

theorem correct_option_D : option_D :=
by {
  -- Proof is required here
  sorry
}

-- Additional statement to show options A, B, C are false
theorem option_A_false : ¬ option_A :=
by {
  -- Proof is required here
  sorry
}

theorem option_B_false : ¬ option_B :=
by {
  -- Proof is required here
  sorry
}

theorem option_C_false : ¬ option_C :=
by {
  -- Proof is required here
  sorry
}

end correct_option_D_option_A_false_option_B_false_option_C_false_l397_397523


namespace number_of_solutions_l397_397480

theorem number_of_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, sqrt (7 - x) = x * sqrt (7 - x)) ∧ S.card = 2 :=
by
  sorry

end number_of_solutions_l397_397480


namespace eq_determines_ratio_l397_397436

theorem eq_determines_ratio (a b x y : ℝ) (h : a * x^3 + b * x^2 * y + b * x * y^2 + a * y^3 = 0) :
  ∃ t : ℝ, t = x / y ∧ (a * t^3 + b * t^2 + b * t + a = 0) :=
sorry

end eq_determines_ratio_l397_397436


namespace inequality_solution_l397_397043

theorem inequality_solution (a x : ℝ) (h₁ : 0 < a) : 
  (0 < a ∧ a < 1 → 2 < x ∧ x < (a-2)/(a-1) → (a * (x - 1)) / (x-2) > 1) ∧ 
  (a = 1 → 2 < x → (a * (x - 1)) / (x-2) > 1 ∧ true) ∧ 
  (a > 1 → (2 < x ∨ x < (a-2)/(a-1)) → (a * (x - 1)) / (x-2) > 1) := 
sorry

end inequality_solution_l397_397043


namespace no_finite_set_S_exists_l397_397205

theorem no_finite_set_S_exists :
  ∀ (S : Set ℕ), (∀ p ∈ S, Nat.Prime p) →
    ¬ ∀ n ≥ 2, ∃ p ∈ S, p ∣ (∑ i in Finset.range (n+1), i^2) - 1 :=
by
  sorry

end no_finite_set_S_exists_l397_397205


namespace N_cross_u_l397_397645

noncomputable def vec : Type := matrix (fin 3) (fin 1) ℝ
noncomputable def mat : Type := matrix (fin 3) (fin 3) ℝ

def cross_product (a b : vec) : vec := 
  ![
    [a 1 0 * b 2 0 - a 2 0 * b 1 0],
    [a 2 0 * b 0 0 - a 0 0 * b 2 0],
    [a 0 0 * b 1 0 - a 1 0 * b 0 0]
  ]

def u : vec := ![![1], ![2], ![3]]

def N : mat := ![![0, 6, -4], [-6, 0, 3], [4, -3, 0]]

theorem N_cross_u (u : vec) : 
  N ⬝ u = cross_product ![![3], ![-4], ![6]] u := 
by sorry

end N_cross_u_l397_397645


namespace minimize_Sn_l397_397765

theorem minimize_Sn (a : ℕ → ℤ) (a1 : a 1 = -28) (d : ∃ c, ∀ n, a n = a 1 + (n - 1) * c ∧ c = 4) :
  (∀ n, n = 7 ∨ n = 8 -> S n = min (S 7) (S 8)) :=
begin
  sorry
end

end minimize_Sn_l397_397765


namespace solve_fractional_eq_l397_397042

noncomputable def fractional_eq (x : ℝ) : Prop := 
  (3 / (x^2 - 3 * x) + (x - 1) / (x - 3) = 1)

noncomputable def not_zero_denom (x : ℝ) : Prop := 
  (x^2 - 3 * x ≠ 0) ∧ (x - 3 ≠ 0)

theorem solve_fractional_eq : fractional_eq (-3/2) ∧ not_zero_denom (-3/2) :=
by
  sorry

end solve_fractional_eq_l397_397042


namespace initial_amount_proof_l397_397311

noncomputable def initial_amount (A B : ℝ) : ℝ :=
  A + B

theorem initial_amount_proof :
  ∃ (A B : ℝ), B = 4000.0000000000005 ∧ 
               (A * 0.15 * 2 = B * 0.18 * 2 + 360) ∧ 
               initial_amount A B = 10000.000000000002 :=
by
  sorry

end initial_amount_proof_l397_397311


namespace slope_range_of_inclination_l397_397690

theorem slope_range_of_inclination (α : ℝ) (hα : α ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4)) :
    ∃ k : ℝ, k ∈ Set.Iic (-1) ∨ k ∈ Set.Ici 1 := 
by
  have k1 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have k2 : Real.tan (3 * Real.pi / 4) = -1 := by norm_num [Real.tan_eq_sin_div_cos, Real.sin_pi_div_two, Real.cos_pi_div_two, Real.sin_add, Real.cos_add]
  sorry

end slope_range_of_inclination_l397_397690


namespace collect_all_balls_into_urn_l397_397872

theorem collect_all_balls_into_urn (d n : ℕ) : 
  ∃ (boxes : ℕ → ℕ), 
  (∀ i, 1 ≤ i ∧ i ≤ d → boxes i = i → 
    ∃ (boxes' : ℕ → ℕ),
       (boxes' 0 = boxes 0 + 1) ∧ 
       (∀ j, 1 ≤ j ∧ j ≤ i-1 → boxes' j = boxes j + 1) ∧ 
       (∀ j, i ≤ j ∧ j ≤ d → boxes' j = boxes j)) →
  (∃ (final_boxes : ℕ → ℕ), 
     (∀ i, 1 ≤ i ∧ i ≤ d → final_boxes i = 0) ∧ 
     (final_boxes 0 = n)) :=
begin
  sorry
end

end collect_all_balls_into_urn_l397_397872


namespace max_val_xa_minus_2x_max_val_ab_plus_bc_plus_ac_l397_397539

-- First proof problem
theorem max_val_xa_minus_2x (x a : ℝ) (h1 : 0 < x) (h2 : 2 * x < a) :
  ∃ y, (y = x * (a - 2 * x)) ∧ y ≤ a^2 / 8 :=
sorry

-- Second proof problem
theorem max_val_ab_plus_bc_plus_ac (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 4) :
  ab + bc + ac ≤ 4 :=
sorry

end max_val_xa_minus_2x_max_val_ab_plus_bc_plus_ac_l397_397539


namespace base_conversion_proof_l397_397827

theorem base_conversion_proof (A B : ℕ) (hA : A < 8) (hB : B < 5) 
  (h : 8 * A + B = 5 * B + A) : 8 * 4 + 1 = 33 :=
by
  have h_eq : 7 * A = 4 * B := by linarith [h]
  have hA' : A = 4 := by sorry -- replace with proof that shows A = 4
  have hB' : B = 1 := by sorry -- replace with proof that shows B = 1
  rw [hA', hB']
  norm_num

end base_conversion_proof_l397_397827


namespace find_ellipse_eq_l397_397468

-- Define the first ellipse with its equation and foci conditions
def ellipse1_eq (x y : ℝ) : Prop := 3 * x^2 + 8 * y^2 = 24
def ellipse1_foci : Set (ℝ × ℝ) := { (sqrt 5, 0), (-sqrt 5, 0) }

-- Define the second ellipse, which should pass through the point (3, 2)
def ellipse2_eq (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- The foci condition based on the major and minor axes relationship
def ellipse2_foci (a b : ℝ) : Set (ℝ × ℝ) :=
  let c := sqrt (abs (a^2 - b^2)) in { (c, 0), (-c, 0) }

-- The proof statement to be developed
theorem find_ellipse_eq :
  ∃ (a b : ℝ), ellipse2_eq 3 2 a b ∧ ellipse2_foci a b = ellipse1_foci ∧ (a, b) = (sqrt 15, sqrt 10) := sorry

end find_ellipse_eq_l397_397468


namespace towel_bleach_volume_decrease_l397_397583

theorem towel_bleach_volume_decrease :
  ∀ (L B T : ℝ) (L' B' T' : ℝ),
  (L' = L * 0.75) →
  (B' = B * 0.70) →
  (T' = T * 0.90) →
  (L * B * T = 1000000) →
  ((L * B * T - L' * B' * T') / (L * B * T) * 100) = 52.75 :=
by
  intros L B T L' B' T' hL' hB' hT' hV
  sorry

end towel_bleach_volume_decrease_l397_397583


namespace number_of_boys_l397_397488

theorem number_of_boys (x g : ℕ) (h1 : x + g = 100) (h2 : g = x) : x = 50 := by
  sorry

end number_of_boys_l397_397488


namespace isosceles_triangle_formed_l397_397619

noncomputable theory

open Real

def line1 (x : ℝ) : ℝ := 2 * x + 3
def line2 (x : ℝ) : ℝ := -2 * x + 3
def line3 (x : ℝ) : ℝ := 1
def line4 (x : ℝ) : ℝ := -2 * x - 2

def point1 := (0, 3)
def point2 := (-1, 1)
def point3 := (1, 1)

def distance (p1 p2 : ℝ × ℝ) : ℝ := sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem isosceles_triangle_formed :
  let d1 := distance point1 point2
  let d2 := distance point1 point3
  let d3 := distance point2 point3
  isosceles_triangle (tuple.point1, tuple.point2, tuple.point3) :=
sorry

end isosceles_triangle_formed_l397_397619


namespace max_cos_product_l397_397685

open Real

theorem max_cos_product (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
                                       (hβ : 0 < β ∧ β < π / 2)
                                       (hγ : 0 < γ ∧ γ < π / 2)
                                       (h : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) : 
  cos α * cos β * cos γ ≤ 2 * Real.sqrt 6 / 9 := 
by sorry

end max_cos_product_l397_397685


namespace translate_parabola_l397_397068

theorem translate_parabola :
  ∀ (x : ℝ), (λ x, 2 * (x + 1) ^ 2 - 3) (x - 1) + 3 = 2 * x ^ 2 :=
by
  intro x
  sorry

end translate_parabola_l397_397068


namespace find_B_l397_397123

theorem find_B (A B C : ℝ) (h1 : A = B + C) (h2 : A + B = 1/25) (h3 : C = 1/35) : B = 1/175 :=
by
  sorry

end find_B_l397_397123


namespace shortest_path_length_l397_397552

-- Defining a regular tetrahedron
structure RegularTetrahedron where
  edgeLength : ℝ

-- Problem-specific assumptions
def TetrahedronWithEdgeLength2 := {T : RegularTetrahedron // T.edgeLength = 2}

-- Defining vertices and midpoints on the tetrahedron
structure Vertex where
  x : ℝ
  y : ℝ
  z : ℝ

structure Edge where
  v1 : Vertex
  v2 : Vertex

def midpoint (e : Edge) : Vertex :=
  {x := (e.v1.x + e.v2.x) / 2, y := (e.v1.y + e.v2.y) / 2, z := (e.v1.z + e.v2.z) / 2}

-- Statement problem:
theorem shortest_path_length (T : TetrahedronWithEdgeLength2) (A : Vertex) (M : Vertex) (e : Edge) 
  (h1 : e.v1 ≠ A ∧ e.v2 ≠ A) (h2 : M = midpoint e) : 
  ∃ (path_length : ℝ), path_length = 3 := 
sorry

end shortest_path_length_l397_397552


namespace g_is_odd_l397_397395

def g (x : ℝ) : ℝ := 1 / (3 ^ x - 1) + 1 / 3

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_is_odd_l397_397395


namespace multiply_polynomials_l397_397802

theorem multiply_polynomials (x : ℂ) : 
  (x^6 + 27 * x^3 + 729) * (x^3 - 27) = x^12 + 27 * x^9 - 19683 * x^3 - 531441 :=
by
  sorry

end multiply_polynomials_l397_397802


namespace total_books_per_year_l397_397888

variable (c s : ℕ)

theorem total_books_per_year (hc : 0 < c) (hs : 0 < s) :
  6 * 12 * (c * s) = 72 * c * s := by
  sorry

end total_books_per_year_l397_397888


namespace exists_six_distinct_naturals_l397_397967

theorem exists_six_distinct_naturals :
  ∃ (a b c d e f : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
    d ≠ e ∧ d ≠ f ∧ 
    e ≠ f ∧ 
    a + b + c + d + e + f = 3528 ∧
    (1/a + 1/b + 1/c + 1/d + 1/e + 1/f : ℝ) = 3528 / 2012 :=
sorry

end exists_six_distinct_naturals_l397_397967


namespace average_of_middle_three_is_three_l397_397459

theorem average_of_middle_three_is_three :
  ∃ (a b c d e : ℕ), 
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (d ≠ e) ∧
    (a + b + c + d + e = 25) ∧
    (∃ (min max : ℕ), min = min a b c d e ∧ max = max a b c d e ∧ (max - min) = 14) ∧
    (a ≠ min a b c d e ∧ a ≠ max a b c d e) ∧
    (b ≠ min a b c d e ∧ b ≠ max a b c d e) ∧
    (c ≠ min a b c d e ∧ c ≠ max a b c d e) ∧
    (d ≠ min a b c d e ∧ d ≠ max a b c d e) ∧
    (e ≠ min a b c d e ∧ e ≠ max a b c d e) ∧
    (a + b + c + d + e = 25) → 
    (a ≠ min a b c d e ∧ a ≠ max a b c d e) ∧
    (b ≠ min a b c d e ∧ b ≠ max a b c d e) ∧
    (c ≠ min a b c d e ∧ c ≠ max a b c d e) →  
  ((a + b + c) / 3 = 3) :=
by
  sorry

end average_of_middle_three_is_three_l397_397459


namespace isosceles_triangle_of_vector_condition_l397_397663

open EuclideanGeometry

variables {A B C P : Point}

theorem isosceles_triangle_of_vector_condition 
  (h : (vector P B - vector P A) * (vector P B + vector P A - 2 * vector P C) = 0) :
  is_isosceles_triangle A B C :=
sorry

end isosceles_triangle_of_vector_condition_l397_397663


namespace sum_of_areas_of_triangles_on_legs_eq_area_on_hypotenuse_l397_397549

theorem sum_of_areas_of_triangles_on_legs_eq_area_on_hypotenuse
    (a b c : ℝ)
    (ha : a = 5)
    (hb : b = 12)
    (hc : c = 13)
    (area_eq : ∀ x : ℝ, (√3 / 4) * x^2)
    (A B C : ℝ)
    (hA : A = area_eq a)
    (hB : B = area_eq b)
    (hC : C = area_eq c):
    A + B = C :=
by
    sorry

end sum_of_areas_of_triangles_on_legs_eq_area_on_hypotenuse_l397_397549


namespace k_range_hyperbola_l397_397276

theorem k_range_hyperbola (k : ℝ) :
  (∀ b : ℝ, ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1) →
  -real.sqrt 2 / 2 < k ∧ k < real.sqrt 2 / 2 :=
by
  sorry

end k_range_hyperbola_l397_397276


namespace angle_between_is_pi_over_2_l397_397721

-- Define the vectors a and b
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (4, -6)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition that the dot product of a and b is zero
def dot_product_zero : Prop := dot_product a b = 0

-- Define the angle between two vectors when their dot product is zero
def angle_between_perpendicular_vectors_is_pi_over_2 (h : dot_product_zero) : Real.Angle :=
  ⟨π / 2, λ h, by sorry⟩

-- Prove that the angle between a and b is π/2 given dot_product_zero
theorem angle_between_is_pi_over_2 : angle_between_perpendicular_vectors_is_pi_over_2 (by sorry) = ⟨π / 2, by sorry⟩ :=
by sorry

end angle_between_is_pi_over_2_l397_397721


namespace total_trips_correct_l397_397511

def trays_per_trip : ℕ := 6
def trays_per_table : List ℕ := [23, 5, 12, 18, 27]

noncomputable def trips_for_table (trays : ℕ) (capacity : ℕ) : ℕ := 
  (trays + capacity - 1) / capacity

def total_trips (tables : List ℕ) (capacity : ℕ) : ℕ := 
  tables.foldl (λ acc trays => acc + trips_for_table trays capacity) 0

theorem total_trips_correct : total_trips trays_per_table trays_per_trip = 15 := 
by 
  sorry

end total_trips_correct_l397_397511


namespace min_value_of_sum_inv_a_l397_397405

noncomputable def min_sum_inv_a (a : Fin 12 → ℝ) : ℝ :=
  ∑ i, 1 / a i

theorem min_value_of_sum_inv_a :
  ∀ (a : Fin 12 → ℝ), (∀ i, 0 < a i) → (∑ i, a i = 1) → min_sum_inv_a a ≥ 144 :=
by
  intros a h_pos h_sum
  sorry

end min_value_of_sum_inv_a_l397_397405


namespace sqrt_quotient_power_l397_397470

noncomputable def expression_quotient : ℝ := (5:ℝ)^(1/6) / (5:ℝ)^(1/5)

theorem sqrt_quotient_power :
  expression_quotient = (5:ℝ)^(-1/30) :=
sorry

end sqrt_quotient_power_l397_397470


namespace equal_likelihood_of_lucky_sums_solution_l397_397356

variables (N : ℕ)

def main_draw := 10
def additional_draw := 8

def lucky_sum_main (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = main_draw ∧ S.sum = sum
def lucky_sum_additional (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = additional_draw ∧ S.sum = sum

theorem equal_likelihood_of_lucky_sums :
  lucky_sum_main N 63 ↔ lucky_sum_additional N 44 :=
sorry

theorem solution :
  equal_likelihood_of_lucky_sums 18 :=
sorry

end equal_likelihood_of_lucky_sums_solution_l397_397356


namespace Susie_possible_values_l397_397822

theorem Susie_possible_values (n : ℕ) (h1 : n > 43) (h2 : 2023 % n = 43) : 
  (∃ count : ℕ, count = 19 ∧ ∀ n, n > 43 ∧ 2023 % n = 43 → 1980 ∣ (2023 - 43)) :=
sorry

end Susie_possible_values_l397_397822


namespace count_not_sum_of_two_l397_397249

def numbers_between (start end : ℕ) : List ℕ :=
  List.range (end - start + 1) |>.map (λ x => x + start)

def sum_of_two_elements_exists (s : Set ℕ) (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ x + y = n

theorem count_not_sum_of_two (A : Set ℕ) : 
  let range := numbers_between 3 89
  let not_sum_count := range.filter (λ n => ¬ sum_of_two_elements_exists A n)
  A = {1, 2, 3, 5, 8, 13, 21, 34, 55} → 
  not_sum_count.length = 51 :=
by
  sorry

end count_not_sum_of_two_l397_397249


namespace events_equally_likely_iff_N_eq_18_l397_397368

variable (N : ℕ)

-- Define the number of combinations in the draws
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the sums of selecting balls
noncomputable def S_63_10 (N: ℕ) : ℕ := sorry -- placeholder definition
noncomputable def S_44_8 (N: ℕ) : ℕ := sorry 

-- Condition for the events being equally likely
theorem events_equally_likely_iff_N_eq_18 : 
  (S_63_10 N * C N 8 = S_44_8 N * C N 10) ↔ N = 18 :=
sorry

end events_equally_likely_iff_N_eq_18_l397_397368


namespace range_of_x_for_sqrt_l397_397328

theorem range_of_x_for_sqrt (x : ℝ) (hx : sqrt (1 / (x - 1)) = sqrt (1 / (x - 1))) : x > 1 :=
by
  sorry

end range_of_x_for_sqrt_l397_397328


namespace part_I_part_II_l397_397677

-- Definitions based on the given conditions
def ellipse_eqn (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2)/(a^2) + (y^2)/(b^2) = 1

def focus_condition (a c : ℝ) : Prop :=
  a = sqrt 2 * c

def line_segment_condition (a b : ℝ) : Prop :=
  2 * b^2 / a = sqrt 2

def relationship_abc (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2

-- The theorem statements
theorem part_I (a b c : ℝ) 
    (h1 : focus_condition a c) 
    (h2 : line_segment_condition a b) 
    (h3 : relationship_abc a b c) :
  a = sqrt 2 ∧ b = 1 ∧ c = 1 := sorry

theorem part_II (a b c k t : ℝ)
    (h1 : focus_condition a c) 
    (h2 : line_segment_condition a b) 
    (h3 : relationship_abc a b c)
    (h4 : k^2 = 7 / 2) 
    (h5 : t = 4) :
  maximum_area (sqrt ((8 * t) / ((t + 4)^2)) = (sqrt 2) / 2) := sorry

end part_I_part_II_l397_397677


namespace benefits_of_public_consultation_l397_397603

-- Given condition definitions
def deepening_taxi_reform {T : Type} (docs : T) : Prop := 
  docs = "Opinions on Deepening the Reform of the Taxi Industry in Jinan"

def management_online_booking {T : Type} (docs : T) : Prop := 
  docs = "Implementation Rules for the Management of Online Booking Taxi Services (Provisional)"

def public_response_methods (methods : list string) : Prop := 
  methods = ["letters", "faxes", "emails"]

-- Proving that the benefits ① and ② are met
theorem benefits_of_public_consultation
  {T : Type} (docs : T) (methods : list string) :
  deepening_taxi_reform docs →
  management_online_booking docs →
  public_response_methods methods →
  (∃ (benefits : list (string × bool)),
    benefits = [("Fully reflecting public opinion", true),
                ("Enhancing enthusiasm of citizens", true),
                ("Ensuring right to supervise", false),
                ("Improving grassroots self-governance", false)]) :=
by {
  intros deepening_taxi_reform management_online_booking public_response_methods,
  sorry
}

end benefits_of_public_consultation_l397_397603


namespace tan_monotone_increasing_tan_130_le_tan_140_l397_397594

theorem tan_monotone_increasing {x y : ℝ} (hx : 90 < x) (hx180 : x ≤ 180) (hy : 90 < y) (hy180 : y ≤ 180) (hxy : x < y) :
  Real.tan x ≤ Real.tan y :=
by sorry

-- Specific case: proving tan 130° <= tan 140°
theorem tan_130_le_tan_140 :
  Real.tan (130 * Real.pi / 180) ≤ Real.tan (140 * Real.pi / 180) :=
by {
  apply tan_monotone_increasing,
  { norm_num },
  { norm_num },
  { norm_num },
  { norm_num },
  { norm_num },
}

end tan_monotone_increasing_tan_130_le_tan_140_l397_397594


namespace range_of_a_l397_397794

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then 2^x else Real.log x / Real.log 2

def g : ℝ → ℝ :=
  λ x, 2 / x

theorem range_of_a (a : ℝ) : f (g a) ≤ 1 ↔ a ∈ Iio 0 ∪ Ici 2 := by
  sorry

end range_of_a_l397_397794


namespace max_value_f_max_value_of_f_l397_397379

open Real

--- Definitions from the conditions in the problem
def pointA : ℝ × ℝ := (-2, 0)
def pointB : ℝ × ℝ := (0, 2 * sqrt 3)
def pointsC (θ : ℝ) : ℝ × ℝ := (2 * cos θ, 2 * sin θ)
def pointD : ℝ × ℝ := (1, 0)
def pointsE (a : ℝ) : ℝ × ℝ := (a, 0)

--- The problem statement to prove
theorem max_value_f (θ : ℝ) (a : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π / 2) :
  let t := cos θ in
  f (θ t) (a t) = -3 * t^2 + 2 * t - 1 := sorry

noncomputable def f (θ : ℝ) (t : ℝ) := -3 * t^2 + 2 * t - 1

theorem max_value_of_f {a : ℝ} :
  a ≥ 3 / 2 → ∀ θ ∈ Icc 0 (π / 2), f = -1 := sorry

end max_value_f_max_value_of_f_l397_397379


namespace sum_of_segments_l397_397200

-- Given data
variables (A B C P A' B' C' : Point) 
variables (a b c : ℝ)
variables (h1 : a = 13) (h2 : b = 14) (h3 : c = 15)
variables (centroidP : centroid A B C P)
variables (median_A' : on_median P A B C A')
variables (median_B' : on_median P B A C B')
variables (median_C' : on_median P C A B C')

-- Prove the sum of segments
theorem sum_of_segments : 
  AA' + BB' + CC' = 21 :=
sorry

end sum_of_segments_l397_397200


namespace average_consecutive_pairs_l397_397100

open Finset

variable {α : Type} [DecidableEq α]

-- Definition of the set from which we choose the integers
def set_30 := range 30

-- Definition of a subset of size 5 from the set_30
def is_valid_subset (S : Finset ℕ) : Prop := S.card = 5 ∧ S ⊆ set_30 ∧ S.pairwise (≠)

-- Definition of consecutive pairs in the subset
def consecutive_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
{ (a, b) ∈ S ×ˢ S | a + 1 = b }

-- The number of consecutive pairs in a given subset
def num_consecutive_pairs (S : Finset ℕ) : ℕ :=
(consecutive_pairs S).card

-- The statement to prove
theorem average_consecutive_pairs : 
  (∑ S in (univ.filter is_valid_subset), num_consecutive_pairs S).toR / (univ.filter is_valid_subset).card.toR = 2 / 3 :=
sorry

end average_consecutive_pairs_l397_397100


namespace subset_sum_divisible_by_m_l397_397789

theorem subset_sum_divisible_by_m
  (m n : ℤ)
  (h1 : n ≥ m)
  (h2 : m ≥ 2)
  (S : finset ℤ)
  (hS : S.card = n)
  : ∃ (T : finset (finset ℤ)), T.card ≥ 2^(n - m + 1) ∧ ∀ t ∈ T, m ∣ t.sum id :=
sorry

end subset_sum_divisible_by_m_l397_397789


namespace average_of_middle_three_l397_397463

theorem average_of_middle_three
  (a b c d e : ℕ)
  (h_distinct : a < b ∧ b < c ∧ c < d ∧ d < e)
  (h_sum : a + b + c + d + e = 25)
  (h_max_diff : ∀ x y : ℕ, x + y = 24 → (e - a) ≥ (y - x)) :
  (b + c + d) / 3 = 3 :=
by
  sorry

end average_of_middle_three_l397_397463


namespace least_n_l397_397229

-- Define the form of the polynomial P(x)
def P (x : ℝ) (n : ℕ) (a : Fin (2 * n + 1) → ℝ) : ℝ :=
  (Finset.range (2 * n + 1)).sum (λ i, a i * x ^ (2 * n - i))

-- Define the conditions
def coefficients_condition (a : Fin (2 * (2014 : ℕ) + 1) → ℝ) : Prop :=
  ∀ i, 2014 ≤ a i ∧ a i ≤ 2015

def root_condition_exists (a : Fin (2 * (2014 : ℕ) + 1) → ℝ) : Prop :=
  ∃ ξ : ℝ, P ξ 2014 a = 0

-- Statement to prove the least positive integer n
theorem least_n (n : ℕ) (a : Fin (2 * n + 1) → ℝ) :
  (∀ (m : ℕ) (b : Fin (2 * m + 1) → ℝ), coefficients_condition b → root_condition_exists b → 2014 = m) → n = 2014 := 
by sorry

end least_n_l397_397229


namespace cylinder_height_increase_l397_397866

theorem cylinder_height_increase 
  (r h : ℝ) (r_increased : ℝ) (y : ℝ) 
  (h_cyl : r = 5 ∧ h = 4)
  (h_radius : r_increased = r + 2)
  (h_volumes : π * r_increased^2 * h = π * r^2 * (h + y)) :
  y = 96 / 25 :=
by
  cases h_cyl with hr hh
  simp [hr, hh, add_mul, pow_add] at *
  sorry

end cylinder_height_increase_l397_397866


namespace point_in_same_region_as_origin_l397_397061

theorem point_in_same_region_as_origin : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y + 5
  ∧ ∀ (P: ℝ × ℝ), line_eq 0 0 > 0 → line_eq P.1 P.2 > 0 → P = (-3, 4) := 
  sorry

end point_in_same_region_as_origin_l397_397061


namespace find_b_collinear_points_l397_397852

theorem find_b_collinear_points :
  ∃ b : ℚ, 4 * 11 - 6 * (-3 * b + 4) = 5 * (b + 3) - 1 * 4 ∧ b = 11 / 26 :=
by
  sorry

end find_b_collinear_points_l397_397852


namespace algebraic_expression_evaluation_l397_397315

theorem algebraic_expression_evaluation
  (x y p q : ℝ)
  (h1 : x + y = 0)
  (h2 : p * q = 1) : (x + y) - 2 * (p * q) = -2 :=
by
  sorry

end algebraic_expression_evaluation_l397_397315


namespace box_max_volume_l397_397941

/-- Given a square metal sheet of side length 60 cm, 
    equal-sized squares are cut out from its four corners, and the edges are folded 
    to form an open-top square-bottom box.
    Prove that the side length of the bottom of the box when the volume is maximized is 40 cm 
    and the maximum volume of the box is 16000 cm³. -/
theorem box_max_volume :
  ∃ (x : ℝ), 
  (0 < x ∧ x < 60) ∧
  let V := (λ x : ℝ, (60 * x^2 - x^3) / 2) in
  (∀ y, (0 < y ∧ y < 60) → V y ≤ V x) ∧
  x = 40 ∧ V x = 16000 :=
begin
  sorry
end

end box_max_volume_l397_397941


namespace solve_math_problem_l397_397187

noncomputable def math_problem : ℝ :=
  2 * real.cos (real.pi / 4) + abs (1 - real.sqrt 2) - real.cbrt 8 + (-1) ^ 2023

theorem solve_math_problem : 
  2 * real.cos (real.pi / 4) + abs (1 - real.sqrt 2) - real.cbrt 8 + (-1) ^ 2023 = 2 * real.sqrt 2 - 4 :=
by {
  have h1: real.cos (real.pi / 4) = real.sqrt 2 / 2 := by sorry,
  have h2: abs (1 - real.sqrt 2) = real.sqrt 2 - 1 := by sorry,
  have h3: real.cbrt 8 = 2 := by sorry,
  have h4: (-1) ^ 2023 = -1 := by sorry,
  sorry
}

end solve_math_problem_l397_397187


namespace g_18_equals_324_l397_397409

def is_strictly_increasing (g : ℕ → ℕ) :=
  ∀ n : ℕ, n > 0 → g (n + 1) > g n

def multiplicative (g : ℕ → ℕ) :=
  ∀ m n : ℕ, m > 0 → n > 0 → g (m * n) = g m * g n

def m_n_condition (g : ℕ → ℕ) :=
  ∀ m n : ℕ, m ≠ n → m > 0 → n > 0 → m ^ n = n ^ m → (g m = n ∨ g n = m)

noncomputable def g : ℕ → ℕ := sorry

theorem g_18_equals_324 :
  is_strictly_increasing g →
  multiplicative g →
  m_n_condition g →
  g 18 = 324 :=
sorry

end g_18_equals_324_l397_397409


namespace arithmetic_sequence_fifth_term_l397_397851

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 15)
  (h2 : a + 10 * d = 18) : 
  a + 4 * d = 0 := 
sorry

end arithmetic_sequence_fifth_term_l397_397851


namespace valid_punching_settings_l397_397482

theorem valid_punching_settings :
  let total_patterns := 2^9
  let symmetric_patterns := 2^6
  total_patterns - symmetric_patterns = 448 :=
by
  sorry

end valid_punching_settings_l397_397482


namespace find_c_for_two_zeros_l397_397057

noncomputable def f (x c : ℝ) : ℝ := x^3 - 3*x + c

theorem find_c_for_two_zeros (c : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 c = 0 ∧ f x2 c = 0) ↔ c = -2 ∨ c = 2 :=
sorry

end find_c_for_two_zeros_l397_397057


namespace find_y_l397_397445

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3 - t) (h2 : y = 3 * t + 6) (h3 : x = -6) : y = 33 := by
  sorry

end find_y_l397_397445


namespace tourists_went_free_l397_397585

theorem tourists_went_free (x : ℕ) : 
  (13 + 4 * x = x + 100) → x = 29 :=
by
  intros h
  sorry

end tourists_went_free_l397_397585


namespace cafeteria_seats_taken_l397_397495

def table1_count : ℕ := 10
def table1_seats : ℕ := 8
def table2_count : ℕ := 5
def table2_seats : ℕ := 12
def table3_count : ℕ := 5
def table3_seats : ℕ := 10
noncomputable def unseated_ratio1 : ℝ := 1/4
noncomputable def unseated_ratio2 : ℝ := 1/3
noncomputable def unseated_ratio3 : ℝ := 1/5

theorem cafeteria_seats_taken : 
  ((table1_count * table1_seats) - (unseated_ratio1 * (table1_count * table1_seats))) + 
  ((table2_count * table2_seats) - (unseated_ratio2 * (table2_count * table2_seats))) + 
  ((table3_count * table3_seats) - (unseated_ratio3 * (table3_count * table3_seats))) = 140 :=
by sorry

end cafeteria_seats_taken_l397_397495


namespace lowest_price_is_ten_percent_l397_397580

/-- A definition for the list price of the jersey. -/
def list_price : ℝ := 120

/-- A definition for the maximum initial discount percentage (as a fraction). -/
def max_initial_discount_fraction : ℝ := 0.80

/-- A definition for the additional discount percentage for the highest discount range (as a fraction). -/
def additional_discount_fraction : ℝ := 0.10

/-- A definition for the lowest possible total sale price calculation. -/
def lowest_possible_total_sale_price (list_price : ℝ) (max_initial_discount_fraction : ℝ) (additional_discount_fraction : ℝ) : ℝ :=
let initial_discount := max_initial_discount_fraction * list_price in
let price_after_initial_discount := list_price - initial_discount in
let additional_discount := additional_discount_fraction * list_price in
price_after_initial_discount - additional_discount

/-- A theorem that states the lowest possible total sale price for the jersey is 10% of the list price. -/
theorem lowest_price_is_ten_percent : (lowest_possible_total_sale_price list_price max_initial_discount_fraction additional_discount_fraction / list_price) * 100 = 10 :=
by
  sorry

end lowest_price_is_ten_percent_l397_397580


namespace two_digit_numbers_count_l397_397586

def is_digit_sum_divisor (n : ℕ) : Prop :=
  let a := n / 10
  let b := n % 10
  let S := a + b
  S > 0 ∧ n % S = 0

def is_digit_diff_divisor (n : ℕ) : Prop :=
  let a := n / 10
  let b := n % 10
  let D := abs (a - b)
  D > 0 ∧ n % D = 0

theorem two_digit_numbers_count : 
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ is_digit_sum_divisor n ∧ is_digit_diff_divisor n}.to_finset.card = 19 :=
by
  sorry

end two_digit_numbers_count_l397_397586


namespace distance_between_boats_l397_397509

-- Definitions based on the given conditions
def boat_speed : ℝ := 20
def time_elapsed : ℝ := 2
def distance_traveled : ℝ := boat_speed * time_elapsed
def angle_between_paths : ℝ := 30
def cos_angle_between_paths : ℝ := real.cos (angle_between_paths * (real.pi / 180))

-- Main theorem to prove the distance between the two boats
theorem distance_between_boats : 
  sqrt ((distance_traveled)^2 + (distance_traveled)^2 - 2 * (distance_traveled)^2 * cos_angle_between_paths) = 
  20 * (sqrt 6 - sqrt 2) :=
by
  sorry

end distance_between_boats_l397_397509


namespace find_value_of_reciprocal_squares_l397_397817

variable {α : ℝ}

-- Given quadratic equations
def equation1 := λ x : ℝ, x^2 + x * Real.sin α + 1 = 0
def equation2 := λ x : ℝ, x^2 + x * Real.cos α - 1 = 0

-- Defining the roots
def roots1 : set ℝ := {a | equation1 a}
def roots2 : set ℝ := {c | equation2 c}

-- The statement
theorem find_value_of_reciprocal_squares :
  (∃ a b : ℝ, a ∈ roots1 ∧ b ∈ roots1 ∧ a ≠ b) →
  (∃ c d : ℝ, c ∈ roots2 ∧ d ∈ roots2 ∧ c ≠ d) →
  ∃ a b c d : ℝ, a ∈ roots1 ∧ b ∈ roots1 ∧ c ∈ roots2 ∧ d ∈ roots2 →
  (1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 = 1) := sorry

end find_value_of_reciprocal_squares_l397_397817


namespace equal_likelihood_of_lucky_sums_solution_l397_397357

variables (N : ℕ)

def main_draw := 10
def additional_draw := 8

def lucky_sum_main (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = main_draw ∧ S.sum = sum
def lucky_sum_additional (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = additional_draw ∧ S.sum = sum

theorem equal_likelihood_of_lucky_sums :
  lucky_sum_main N 63 ↔ lucky_sum_additional N 44 :=
sorry

theorem solution :
  equal_likelihood_of_lucky_sums 18 :=
sorry

end equal_likelihood_of_lucky_sums_solution_l397_397357


namespace plane_divided_by_n_lines_l397_397096

-- Definition of the number of regions created by n lines in a plane
def regions (n : ℕ) : ℕ :=
  if n = 0 then 1 else (n * (n + 1)) / 2 + 1 -- Using the given formula directly

-- Theorem statement to prove the formula holds
theorem plane_divided_by_n_lines (n : ℕ) : 
  regions n = (n * (n + 1)) / 2 + 1 :=
sorry

end plane_divided_by_n_lines_l397_397096


namespace horizontal_asymptote_of_rational_function_l397_397286

theorem horizontal_asymptote_of_rational_function :
  ∀ (x : ℝ), (y = (7 * x^2 - 5) / (4 * x^2 + 6 * x + 3)) → (∃ b : ℝ, b = 7 / 4) :=
by
  intro x y
  sorry

end horizontal_asymptote_of_rational_function_l397_397286


namespace find_d_l397_397483

open Real

theorem find_d (a b c d : ℝ) 
  (h : a^3 + b^3 + c^3 + a^2 + b^2 + 1 = d^2 + d + sqrt (a + b + c - 2 * d)) : 
  d = 1 ∨ d = -(4 / 3) :=
sorry

end find_d_l397_397483


namespace sqrt_frac_meaningful_l397_397331

theorem sqrt_frac_meaningful (x : ℝ) (h : 1 / (x - 1) > 0) : x > 1 :=
sorry

end sqrt_frac_meaningful_l397_397331


namespace son_l397_397889

theorem son's_age (S M : ℕ) (h1 : M = S + 20) (h2 : M + 2 = 2 * (S + 2)) : S = 18 := by
  sorry

end son_l397_397889


namespace max_value_expression_l397_397208

theorem max_value_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1 / Real.sqrt 3) :
  27 * a * b * c + a * Real.sqrt (a^2 + 2 * b * c) + b * Real.sqrt (b^2 + 2 * c * a) + c * Real.sqrt (c^2 + 2 * a * b) ≤ 2 / (3 * Real.sqrt 3) :=
sorry

end max_value_expression_l397_397208


namespace parabola_transformation_correct_l397_397066

theorem parabola_transformation_correct:
  let initial_eq : ℝ → ℝ := λ x, 2 * (x + 1)^2 - 3
  let transformed_eq : ℝ → ℝ := λ x, 2 * x^2
  ∃ seq : list (ℝ → ℝ → ℝ → ℝ),
    (seq = [λ x y c, y = 2 * (x - 1)^2 - 3, λ x y c, y = y + 3]) ∧
    ∀ x, (seq.head x (initial_eq x) 0 = transformed_eq x) :=
sorry

end parabola_transformation_correct_l397_397066


namespace p_or_q_p_and_q_not_p_l397_397268

def p : Prop := true  -- the diagonals of a square are perpendicular to each other
def q : Prop := true  -- the diagonals of a square are equal

theorem p_or_q : p ∨ q :=
by exact Or.inl p

theorem p_and_q : p ∧ q :=
by exact And.intro p q

theorem not_p : ¬p = false :=
by { simp, exact p }

end p_or_q_p_and_q_not_p_l397_397268


namespace ratio_of_a_b_c_l397_397530

theorem ratio_of_a_b_c (a b c : ℕ) (h : a : b : c = 3 : 4 : 7) : (a + b + c) : c = 2 : 1 := by 
  sorry

end ratio_of_a_b_c_l397_397530


namespace conditional_convergence_l397_397386

noncomputable def integrand (x : ℝ) : ℝ := 
  2 * x * Real.sin (Real.pi / (x ^ 2)) - 2 * Real.pi / x * Real.cos (Real.pi / (x ^ 2))

theorem conditional_convergence :
  ∃ l : ℝ, tendsto (λ a : ℝ, ∫ x in a..2, integrand x) (𝓝 0) (𝓝 l) ∧
       ¬ summable (λ n, ∫ x in n..(n+1), |integrand x|) :=
sorry

end conditional_convergence_l397_397386


namespace value_of_expression_l397_397319

theorem value_of_expression (n : ℝ) (h : n + 1/n = 10) : n^2 + (1/n^2) + 6 = 104 :=
by
  sorry

end value_of_expression_l397_397319


namespace _l397_397013

noncomputable theorem symmetric_difference_convergence
  {Ω : Type*} [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)
  (A : Set Ω) (A_n : ℕ → Set Ω) :
  (P (A ∆ (MeasureTheory.limsup A_n)) = 0) →
  (P (A ∆ (MeasureTheory.liminf A_n)) = 0) →
  ∀ ε, ∃ N, ∀ n ≥ N, P (A ∆ A_n) < ε := 
sorry

end _l397_397013


namespace centers_lie_on_line_l397_397673

open Real

theorem centers_lie_on_line (k : ℕ) (hk : 1 ≤ k) :
  let center := (k - 1, 3 * k) in
  (center.2 = 3 * (center.1 + 1)) :=
by
  sorry

end centers_lie_on_line_l397_397673


namespace events_equally_likely_iff_N_eq_18_l397_397365

variable (N : ℕ)

-- Define the number of combinations in the draws
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the sums of selecting balls
noncomputable def S_63_10 (N: ℕ) : ℕ := sorry -- placeholder definition
noncomputable def S_44_8 (N: ℕ) : ℕ := sorry 

-- Condition for the events being equally likely
theorem events_equally_likely_iff_N_eq_18 : 
  (S_63_10 N * C N 8 = S_44_8 N * C N 10) ↔ N = 18 :=
sorry

end events_equally_likely_iff_N_eq_18_l397_397365


namespace graph_existence_conditions_l397_397660

noncomputable def exists_graph_with_degree_constraints (n : ℕ) : Prop :=
∃ (G : SimpleGraph (Fin (n * (n + 1) / 2))), 
  ∀ i : Fin (n + 1), (∃ v : Fin (n * (n + 1) / 2), G.degree v = i)

theorem graph_existence_conditions (n : ℕ) : exists_graph_with_degree_constraints n →
  n % 4 = 0 ∨ n % 4 = 3 :=
sorry

end graph_existence_conditions_l397_397660


namespace option_A_option_B_option_C_option_D_l397_397723

noncomputable def a : ℝ  := -3
noncomputable def b : ℝ := 2
noncomputable def c (λ : ℝ) := λ
noncomputable def d : ℝ := -1

-- Definition of projection
noncomputable def projection (v w : ℝ × ℝ) :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let norm_square := w.1 * w.1 + w.2 * w.2
  (dot_product / norm_square) * w

theorem option_A (λ : ℝ) (h : λ = 1) :
  projection (a + 2 * b, d + 2 * d) (λ, -1) = -3/2 * (λ, -1) :=
by
  have a := (-3, 2)
  have b := (2, 1)
  have v := (1, 4)
  have w := (1, -1)
  sorry

theorem option_B :
  (2 / real.sqrt 5, 1 / real.sqrt 5) ≠ (2 * real.sqrt 5 / 5, real.sqrt 5 / 5) :=
by sorry

theorem option_C (t λ : ℝ) (h : a + b * t + c λ = (-3, 2)) :
  λ + t ≠ -4 :=
by sorry

theorem option_D :
  ∃ (μ : ℝ), (a + μ * b, d + μ * d) = 7 * real.sqrt 5 / 5 :=
by sorry

end option_A_option_B_option_C_option_D_l397_397723


namespace oil_height_in_tank_l397_397152

-- Define the cylindrical parameters
variables (r h : ℝ)
constants (V : ℝ)
-- Conditions stated in the given problem
def cylindrical_oil_tank (r h V : ℝ) :=
  r = 2 ∧ V = 48 ∧ V = π * r^2 * h

-- Theorem to prove the height of the oil in the tank 
theorem oil_height_in_tank : cylindrical_oil_tank r h V → h = 12 / π := 
begin
  sorry
end

end oil_height_in_tank_l397_397152


namespace shaded_area_correct_l397_397772

-- Let s be the side length of the regular hexagon.
def s := 4

-- Define the radius of each semicircle based on the side length.
def r := s / 2

-- Define the total number of semicircles.
def num_semicircles := 6

-- Define the area of the regular hexagon based on the side length.
def area_hexagon := (3 * Real.sqrt 3 / 2) * s^2

-- Define the area of one semicircle based on the radius.
def area_one_semicircle := (Real.pi * r^2 / 2)

-- Define the total area of all semicircles.
def total_area_semicircles := num_semicircles * area_one_semicircle

-- Define the area of the region inside the hexagon but outside all semicircles.
def area_shaded_region := area_hexagon - total_area_semicircles

-- State the theorem to prove that the area of the shaded region is equal to 24 * sqrt(3) - 12 * pi.
theorem shaded_area_correct : area_shaded_region = 24 * Real.sqrt 3 - 12 * Real.pi := 
by
  sorry

end shaded_area_correct_l397_397772


namespace initial_dog_cat_ratio_l397_397748

theorem initial_dog_cat_ratio (C : ℕ) :
  75 / (C + 20) = 15 / 11 →
  (75 / C) = 15 / 7 :=
by
  sorry

end initial_dog_cat_ratio_l397_397748


namespace min_B1P_PQ_l397_397381

theorem min_B1P_PQ (AB BC AA1 : ℝ) (hp : AB = 2 * real.sqrt 2) (hq1 : BC = 2) (hq2 : AA1 = 2) :
  ∃ P Q, P ∈ AC1 ∧ Q ∈ ABCD_base ∧ B1P_PQ = 4 :=
by sorry

end min_B1P_PQ_l397_397381


namespace value_of_x_l397_397247

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2 * x

theorem value_of_x (x : ℝ) : f x = 3 → x = sqrt 3 :=
by
  sorry

end value_of_x_l397_397247


namespace arun_remaining_work_days_l397_397891

noncomputable def arun_and_tarun_work_in_days (W : ℝ) := 10
noncomputable def arun_alone_work_in_days (W : ℝ) := 60
noncomputable def arun_tarun_together_days := 4

theorem arun_remaining_work_days (W : ℝ) :
  (arun_and_tarun_work_in_days W = 10) ∧
  (arun_alone_work_in_days W = 60) ∧
  (let complete_work_days := arun_tarun_together_days;
  let remaining_work := W - (complete_work_days / arun_and_tarun_work_in_days W * W);
  let arun_remaining_days := (remaining_work / W) * arun_alone_work_in_days W;
  arun_remaining_days = 36) :=
sorry

end arun_remaining_work_days_l397_397891


namespace smallest_k_divisor_l397_397253

theorem smallest_k_divisor 
  (n : ℕ) (hn : n ≥ 3)
  (d : Fin n → ℕ)
  (gcd_d : Nat.gcd (Finset.univ.product (fun i => d i)) = 1)
  (div_d_sum : ∀i : Fin n, d i ∣ Finset.univ.sum (fun i => d i))
  : ∃ k : ℕ, k = n - 2 ∧ (Finset.univ.product (fun i => d i)) ∣ (Finset.univ.sum (fun i => d i)) ^ k :=
sorry

end smallest_k_divisor_l397_397253


namespace find_x_l397_397738

/-- Let x be a real number such that the square roots of a positive number are given by x - 4 and 3. 
    Prove that x equals 1. -/
theorem find_x (x : ℝ) 
  (h₁ : ∃ n : ℝ, n > 0 ∧ n.sqrt = x - 4 ∧ n.sqrt = 3) : 
  x = 1 :=
by
  sorry

end find_x_l397_397738


namespace unique_solution_system_l397_397195

theorem unique_solution_system :
  ∃! (x y z w : ℝ),
    x = Real.sin (z + w + z * w * x) ∧
    y = Real.sin (w + x + w * x * y) ∧
    z = Real.sin (x + y + x * y * z) ∧
    w = Real.sin (y + z + y * z * w) ∧
    Real.cos (x + y + z + w) = 1 :=
begin
  sorry
end

end unique_solution_system_l397_397195


namespace curve_C_prime_eq_trajectory_eq_l397_397289

-- Definition of the parametric equations and coordinate transformation.
def parametric_equation (θ : Real) : Real × Real := (2 * cos θ, 3 * sin θ)
def coordinate_transformation (x y : Real) : Real × Real := (x / 2, y / 3)

-- Statement for the curve C'
theorem curve_C_prime_eq :
  ∀ (θ : Real),
  let (x', y') := coordinate_transformation (2 * cos θ) (3 * sin θ) in
  x'^2 + y'^2 = 1 := sorry

-- Statement for the trajectory equation of midpoint P
theorem trajectory_eq (x y : Real) :
  (let (x0, y0) := (2 * x - 3, 2 * y) in
  x0^2 + y0^2 = 1) →
  (x - 3 / 2)^2 + y^2 = 1 / 4 := sorry

end curve_C_prime_eq_trajectory_eq_l397_397289


namespace ratio_of_area_l397_397869

-- Define the equilateral triangle ABC with side length 1
def A := (0 : ℝ, 0 : ℝ)
def B := (1 : ℝ, 0 : ℝ)
def C := (1/2 : ℝ, (real.sqrt 3) / 2 : ℝ)

-- Definition for the paths of particles
def particle1 (t : ℝ) : ℝ × ℝ := 
  if t < 1 then (t, 0)
  else if t < 2 then (1 - (t - 1) / 2, (t - 1) * (real.sqrt 3) / 2)
  else ((3 - t) / 2, (2 - t) * (real.sqrt 3) / 2)
  
def particle2 (t : ℝ) : ℝ × ℝ := 
  if t < 1 then (1 - t, 0)
  else if t < 2 then ((t - 1) / 2, (2 -t) * (real.sqrt 3) / 2)
  else ((t - 2), ((t - 2) * (real.sqrt 3) / 2))

-- Definition for the midpoint of the two particles
def midpoint (t : ℝ) : ℝ × ℝ :=
  ((particle1 t).1 + (particle2 t).1) / 2, ((particle1 t).2 + (particle2 t).2) / 2

-- Proof of the problem statement
theorem ratio_of_area (side_length : ℝ) (h : side_length = 1) : 
  (area_of_midpoint_path / area_of_triangle A B C = 1 / 4) :=
sorry

end ratio_of_area_l397_397869


namespace median_siblings_l397_397058

theorem median_siblings :
  let distribution := [ (0, 2), (1, 3), (2, 2), (3, 1), (4, 2), (5, 1) ]
  ∑ x in distribution, x.2 = 11 →
  (let flatList := (List.replicate 2 0 ++ List.replicate 3 1 ++ List.replicate 2 2 ++ List.replicate 1 3 ++ List.replicate 2 4 ++ List.replicate 1 5).qsort (· ≤ ·)
   in flatList.length = 11 ∧ flatList.nth (5) = some 2) :=
sorry

end median_siblings_l397_397058


namespace intersection_points_diff_l397_397842

theorem intersection_points_diff : 
  let y1 := λ x : ℝ, 3 * x^2 - 6 * x + 3
  let y2 := λ x : ℝ, - x^2 - 3 * x + 3
  ∃ p r : ℝ, (r ≥ p) ∧ 
            (∀ y : ℝ, y = y1 p → y = y2 p) ∧ 
            (∀ y : ℝ, y = y1 r → y = y2 r) ∧
            (r - p = 3/4) :=
begin
  sorry,
end

end intersection_points_diff_l397_397842


namespace even_binomial_sum_l397_397608

theorem even_binomial_sum (n : ℕ) : 
  ∑ k in finset.range (n + 1), nat.choose (2 * n) (2 * k) = (2 ^ (2 * n - 1)) - 1 :=
sorry

end even_binomial_sum_l397_397608


namespace determine_n_l397_397959

theorem determine_n (n : ℕ) (h1 : n ≥ 1) 
  (h2 : ∃ (a : Fin n → ℕ), 
    (∀ k : ℕ, 1 ≤ k → k ≤ n → (∑ i in finset.range k, a ⟨i % n, nat.mod_lt i (lt_of_lt_of_le (nat.pos_of_ne_zero (nat.pred_ne_zero (k - 1))) h1)⟩) % k = 0)) :
    n = 1 ∨ n = 3 :=
begin
  sorry
end

end determine_n_l397_397959


namespace marble_prob_difference_l397_397338

def red_marbles := 2003
def black_marbles := 2003
def total_marbles := red_marbles + black_marbles

def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

def P_s := (binom red_marbles 2 + binom black_marbles 2) / binom total_marbles 2
def P_d := (red_marbles * black_marbles) / binom total_marbles 2

theorem marble_prob_difference : 
  |P_s - P_d| = 997 / 8020010 :=
sorry

end marble_prob_difference_l397_397338


namespace find_interval_l397_397657

theorem find_interval (n : ℕ) 
  (h1 : n < 500) 
  (h2 : n ∣ 9999) 
  (h3 : n + 4 ∣ 99) : (1 ≤ n) ∧ (n ≤ 125) := 
sorry

end find_interval_l397_397657


namespace production_equipment_B_l397_397854

theorem production_equipment_B :
  ∃ (X Y : ℕ), X + Y = 4800 ∧ (50 / 80 = 5 / 8) ∧ (X / 4800 = 5 / 8) ∧ Y = 1800 :=
by
  sorry

end production_equipment_B_l397_397854


namespace minimum_distinct_lines_l397_397346

theorem minimum_distinct_lines (n : ℕ) (h : n = 31) : 
  ∃ (k : ℕ), k = 9 :=
by
  sorry

end minimum_distinct_lines_l397_397346


namespace complex_equation_l397_397271

theorem complex_equation (m n : ℝ) (i : ℂ)
  (hi : i^2 = -1)
  (h1 : m * (1 + i) = 1 + n * i) :
  ( (m + n * i) / (m - n * i) )^2 = -1 :=
sorry

end complex_equation_l397_397271


namespace equally_likely_events_A_and_B_l397_397371

-- Given conditions definitions
def number_of_balls (N : ℕ) := N
def main_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 10
def additional_draw_balls (N : ℕ) := (finset.range (N + 1)).powerset_len 8
def lucky_sum_main (s : finset ℕ) := s.sum = 63
def lucky_sum_additional (s : finset ℕ) := s.sum = 44

-- Prove the events A and B are equally likely if and only if N = 18
theorem equally_likely_events_A_and_B (N : ℕ) :
  (∃ s ∈ main_draw_balls N, lucky_sum_main s) ∧ (∃ t ∈ additional_draw_balls N, lucky_sum_additional t)
  → (N = 18) :=
sorry

end equally_likely_events_A_and_B_l397_397371


namespace sum_of_numbers_is_216_l397_397130

-- Define the conditions and what needs to be proved.
theorem sum_of_numbers_is_216 
  (x : ℕ) 
  (h_lcm : Nat.lcm (2 * x) (Nat.lcm (3 * x) (7 * x)) = 126) : 
  2 * x + 3 * x + 7 * x = 216 :=
by
  sorry

end sum_of_numbers_is_216_l397_397130


namespace at_least_7n_segments_l397_397803

theorem at_least_7n_segments {n : ℕ} (h1 : 4 * n = card (points : set ℝ²))
  (h2 : ∀ (p1 p2 : ℝ²), dist p1 p2 = 1 ↔ (p1, p2) ∈ connected_by_segment)
  (h3 : ∀ (P : set ℝ²), P ⊆ points → card P ≥ n + 1 → ∃ (p1 p2 : ℝ²), p1 ∈ P ∧ p2 ∈ P ∧ (p1, p2) ∈ connected_by_segment) :
  ∃ (segments : set (ℝ² × ℝ²)), card segments ≥ 7 * n :=
sorry

end at_least_7n_segments_l397_397803


namespace hotel_room_allocation_l397_397970

theorem hotel_room_allocation :
  ∀ (fans : ℕ) (num_groups : ℕ) (room_capacity : ℕ), 
  fans = 100 → num_groups = 6 → room_capacity = 3 →
  (∀ (group_fans : ℕ) (groups : fin num_groups), group_fans ≤ fans →
   (∃ (total_rooms: ℕ), total_rooms = 37 ∧
    ∀ (group_fan_count : fin num_groups → ℕ), 
      (∀ g, group_fan_count g ≤ fans / num_groups + if (fans % num_groups > 0) then 1 else 0) →
      ∃ (rooms : fin num_groups → ℕ), 
        sum (λ g, rooms g) = total_rooms ∧ 
        ∀ g, group_fan_count g ≤ rooms g * room_capacity)) :=
by {
  intros,
  sorry
}

end hotel_room_allocation_l397_397970


namespace problem_statement_l397_397644

def largest_N_20x20 : ℕ :=
  209

theorem problem_statement :
  ∀ (f : Fin 400 → Fin 20 × Fin 20), ∃ i j : Fin 20, i ≠ j ∧
  (abs ((↑(f ⟨i * 20 + j, sorry⟩), ↑(f ⟨i * 20 + j, sorry ⟩)) -
     (↑(f ⟨i * 20 + j, sorry⟩+ 1, sorry⟩), ↑(f ⟨i * 20 + j, sorry ⟩+ 1, sorry ⟩))) ≥ largest_N_20x20 :=
by
  sorry

end problem_statement_l397_397644


namespace total_length_of_rubber_pen_pencil_l397_397576

variable (rubber pen pencil : ℕ)

theorem total_length_of_rubber_pen_pencil 
  (h1 : pen = rubber + 3)
  (h2 : pen = pencil - 2)
  (h3 : pencil = 12) : rubber + pen + pencil = 29 := by
  sorry

end total_length_of_rubber_pen_pencil_l397_397576


namespace eval_prod_floor_ceil_l397_397218

-- Define the given conditions and question
def prod_floor_ceil : Int :=
  let expr := (List.range 6).map (λ n => Int.floor (- (n + 0.5)) * Int.ceil (n + 0.5))
  expr.product

theorem eval_prod_floor_ceil :
  prod_floor_ceil = -518400 :=
sorry

end eval_prod_floor_ceil_l397_397218


namespace solve_inequality_l397_397442

theorem solve_inequality (x : ℝ) :
  (x < 3) → (x ≠ 2) →
  (2 ^ ((log 2) ^ 2 * log 2 x) - 12 * x ^ (log 0.5 x) < 3 - real.log (x^2 - 6*x + 9) / (log (3-x))) →
  x ∈ set.Ioo (2 ^ (-real.sqrt 2)) 2 ∪ set.Ioo 2 (2 ^ real.sqrt 2) :=
by
  intro h1 h2 h3
  sorry

end solve_inequality_l397_397442


namespace angle_AOB_l397_397093

-- Define the angles and their relationships
def is_tangent_triangle (P A B O : Point) : Prop :=
  Tangent P A O ∧ Tangent P B O ∧ Tangent A B O

variables {P A B O : Point}
variable (h_tangent_triangle : is_tangent_triangle P A B O)
variable (angle_APB : ℝ)
variable (angle_APB_eq : angle_APB = 50)

theorem angle_AOB (h_tangent_triangle : is_tangent_triangle P A B O) (h_angle_APB_eq : angle_APB = 50) : ∠AOB = 65 :=
by
  sorry

end angle_AOB_l397_397093


namespace solution_set_for_inequality_l397_397408

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem solution_set_for_inequality
  (h1 : is_odd f)
  (h2 : f 2 = 0)
  (h3 : ∀ x > 0, x * deriv f x - f x < 0) :
  {x : ℝ | f x / x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_for_inequality_l397_397408


namespace smallest_value_f9_l397_397407

noncomputable def f : ℝ → ℝ := sorry -- the actual polynomial definition is not given

theorem smallest_value_f9 :
  ∃ (f : ℝ → ℝ) (H : ∀ x, 0 ≤ f x), f 3 = 9 ∧ f 18 = 972 → f 9 = 0 :=
begin
  sorry
end

end smallest_value_f9_l397_397407


namespace range_of_function_l397_397072

theorem range_of_function : ∀ x : ℝ, 1 ≤ abs (Real.sin x) + 2 * abs (Real.cos x) ∧ abs (Real.sin x) + 2 * abs (Real.cos x) ≤ Real.sqrt 5 :=
by
  intro x
  sorry

end range_of_function_l397_397072


namespace binomial_12_11_eq_12_l397_397612

-- Define binomial coefficient function
def binomial (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement to prove
theorem binomial_12_11_eq_12 :
  binomial 12 11 = 12 := by
  sorry

end binomial_12_11_eq_12_l397_397612


namespace coin_arrangement_l397_397912

-- Define the initial setup of the circle, the coins, and their positions.
def sectors : List ℕ := List.range 14

def coins := {1, 2, 3, 5}

-- Define a function to determine the new position after jumping three sectors into the fourth.
def move (position : ℕ) : ℕ := (position + 3) % 14

-- Define initial positions of the coins in terms of their parity.
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the problem statement in Lean, requiring a proof.
theorem coin_arrangement (initial_positions : List ℕ) (final_positions : List ℕ) :
  (∀ pos ∈ initial_positions, pos < 14 ∧ ( is_odd pos ∨ is_odd pos) ∧ (is_even pos ∨ is_even pos)) →
  final_positions = initial_positions ∨ 
  (final_positions = List.map move initial_positions ∧ 
   ∀ pos ∈ final_positions, pos < 14  ∧ (is_odd pos ∨ is_odd pos) ∧ (is_even pos ∨ is_even pos)) :=
sorry

end coin_arrangement_l397_397912


namespace solve_inequality_l397_397080

theorem solve_inequality (x : ℝ) : x ≠ 1 → (1 / (x - 1) > 1 → (1 < x ∧ x < 2)) := by
  intros h hx
  have h1 : x - 1 ≠ 0 := h
  sorry

end solve_inequality_l397_397080


namespace rebecca_end_of_day_money_eq_l397_397808

-- Define the costs for different services
def haircut_cost   := 30
def perm_cost      := 40
def dye_job_cost   := 60
def extension_cost := 80

-- Define the supply costs for the services
def haircut_supply_cost   := 5
def dye_job_supply_cost   := 10
def extension_supply_cost := 25

-- Today's appointments
def num_haircuts   := 5
def num_perms      := 3
def num_dye_jobs   := 2
def num_extensions := 1

-- Additional incomes and expenses
def tips           := 75
def daily_expenses := 45

-- Calculate the total earnings and costs
def total_service_revenue : ℕ := 
  num_haircuts * haircut_cost +
  num_perms * perm_cost +
  num_dye_jobs * dye_job_cost +
  num_extensions * extension_cost

def total_revenue : ℕ := total_service_revenue + tips

def total_supply_cost : ℕ := 
  num_haircuts * haircut_supply_cost +
  num_dye_jobs * dye_job_supply_cost +
  num_extensions * extension_supply_cost

def end_of_day_money : ℕ := total_revenue - total_supply_cost - daily_expenses

-- Lean statement to prove Rebecca will have $430 at the end of the day
theorem rebecca_end_of_day_money_eq : end_of_day_money = 430 := by
  sorry

end rebecca_end_of_day_money_eq_l397_397808


namespace hotel_room_allocation_l397_397975

theorem hotel_room_allocation :
  ∀ (fans : ℕ) (num_groups : ℕ) (room_capacity : ℕ), 
  fans = 100 → num_groups = 6 → room_capacity = 3 →
  (∀ (group_fans : ℕ) (groups : fin num_groups), group_fans ≤ fans →
   (∃ (total_rooms: ℕ), total_rooms = 37 ∧
    ∀ (group_fan_count : fin num_groups → ℕ), 
      (∀ g, group_fan_count g ≤ fans / num_groups + if (fans % num_groups > 0) then 1 else 0) →
      ∃ (rooms : fin num_groups → ℕ), 
        sum (λ g, rooms g) = total_rooms ∧ 
        ∀ g, group_fan_count g ≤ rooms g * room_capacity)) :=
by {
  intros,
  sorry
}

end hotel_room_allocation_l397_397975


namespace impossible_delegates_arrangement_l397_397939

theorem impossible_delegates_arrangement:
  (∀ (n : ℕ) (c1 c2 : set ℕ),
    2 * n = 54 ∧
    (∀ i : ℕ, c1 i ∈ finset.range 54) ∧
    (∀ i : ℕ, c2 i ∈ finset.range 54) ∧
    (∀ i : ℕ, c1 i ≠ c2 i) ∧
    ∀ i : ℕ, (c1 i + 10) % 54 = c2 i ∨ (c2 i + 10) % 54 = c1 i) →
  false :=
begin
  sorry
end

end impossible_delegates_arrangement_l397_397939


namespace backpacks_weight_l397_397871

variables (w_y w_g : ℝ)

theorem backpacks_weight :
  (2 * w_y + 3 * w_g = 44) ∧
  (w_y + w_g + w_g / 2 = w_g + w_y / 2) →
  (w_g = 4) ∧ (w_y = 12) :=
by
  intros h
  sorry

end backpacks_weight_l397_397871


namespace miles_walked_l397_397428

theorem miles_walked (flips : ℕ) (last_day : ℕ) (steps_per_mile : ℕ) (cycles : ℕ) 
    (cycle_steps : ℕ) (total_steps : ℕ) (miles : ℕ)
    (h1 : flips = 50) (h2 : last_day = 25000) (h3 : steps_per_mile = 1500)
    (h4 : cycle_steps = 100000) (h5 : cycles = flips) (h6 : total_steps = cycles * cycle_steps + last_day)
    (h7 : miles = total_steps / steps_per_mile) :
    abs (miles - 3500) ≤ abs (miles - 3000) ∧ abs (miles - 3500) ≤ abs (miles - 4000) ∧ 
    abs (miles - 3500) ≤ abs (miles - 4500) ∧ abs (miles - 3500) ≤ abs (miles - 5000) := sorry

end miles_walked_l397_397428


namespace problem_solution_l397_397224

noncomputable def solution_set : Set ℝ := {x : ℝ | 12 ≤ x ∧ x < 12.08333}

theorem problem_solution (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 144) ↔ (x ∈ solution_set) :=
by
  sorry

end problem_solution_l397_397224


namespace car_a_distance_behind_car_b_l397_397610

theorem car_a_distance_behind_car_b :
  ∃ D : ℝ, D = 40 ∧ 
    (∀ (t : ℝ), t = 4 →
    ((58 - 50) * t + 8) = D + 8)
  := by
  sorry

end car_a_distance_behind_car_b_l397_397610


namespace probability_odd_dots_on_top_l397_397020

-- Definitions based on conditions
def standard_die_dots : ℕ := 21
def number_of_dots_removed : ℕ := 2

-- Proposition stating the problem and expected solution
theorem probability_odd_dots_on_top :
  -- Given that we remove two dots
  let total_faces := 6 in
  let probability_of_removal (dots_removed : ℕ) := (∑ k in (finset.range standard_die_dots), k) / (comb standard_die_dots dots_removed) in
  -- Our result should be the calculated probability
  probability_of_removal number_of_dots_removed = 2 / 7 :=
sorry

end probability_odd_dots_on_top_l397_397020


namespace furniture_definition_based_on_vocabulary_study_l397_397529

theorem furniture_definition_based_on_vocabulary_study (term : String) (h : term = "furniture") :
  term = "furniture" :=
by
  sorry

end furniture_definition_based_on_vocabulary_study_l397_397529


namespace tiling_L_shaped_l397_397807

theorem tiling_L_shaped (n : ℕ) (board : ℕ -> ℕ -> Prop) (removed_square : ℕ -> ℕ) :
  (board (2^n) (2^n)) → 
  (∀ i j, i < 2^n → j < 2^n → board i j ∧ ¬ board (removed_square i j)) →
  ∃ tiling : ℕ -> ℕ -> ℕ -> Prop, 
    (∀ i j, i < (2^n) → j < (2^n) → ∃ t, tiling t i j ∧
    ∀ x y, tiling t x y → (x = i ∧ y = j ∨ 
                            board x y ∧ ¬board (removed_square x y) ∧
                            (x = i + 1 ∧ y = j ∨ x = i ∧ y = j + 1 ∨ x = i + 1 ∧ y = j + 1))) :=
sorry

end tiling_L_shaped_l397_397807


namespace prime_count_between_50_and_100_with_prime_remainder_div_12_l397_397302

open Nat

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_between (a b : ℕ) : list ℕ :=
  (list.range' a (b - a + 1)).filter is_prime

def prime_remainders (p : ℕ) : list ℕ := [2, 3, 5, 7, 11]

theorem prime_count_between_50_and_100_with_prime_remainder_div_12 : 
  (primes_between 50 100).filter (λ p, (p % 12) ∈ prime_remainders p).length = 7 :=
by
  sorry

end prime_count_between_50_and_100_with_prime_remainder_div_12_l397_397302


namespace negative_cube_root_l397_397881

theorem negative_cube_root (a : ℝ) : ∃ x : ℝ, x ^ 3 = -a^2 - 1 ∧ x < 0 :=
by
  sorry

end negative_cube_root_l397_397881


namespace clara_reversed_score_difference_l397_397954

theorem clara_reversed_score_difference:
  ∃ (a b : ℕ), (10 * b + a + 45 + 54) - (10 * a + b + 45 + 54) = 132 → 
  (9 * |b - a| = 126) := by
  sorry

end clara_reversed_score_difference_l397_397954


namespace part_one_part_two_l397_397261

-- Defining the sequence {a_n} with the sum of the first n terms.
def S (n : ℕ) : ℕ := 3 * n ^ 2 + 10 * n

-- Defining a_n in terms of the sum S_n
def a (n : ℕ) : ℕ :=
  if n = 1 then S 1 else S n - S (n - 1)

-- Defining the arithmetic sequence {b_n}
def b (n : ℕ) : ℕ := 3 * n + 2

-- Defining the sequence {c_n}
def c (n : ℕ) : ℕ := (a n + 1)^(n + 1) / (b n + 2)^n

-- Defining the sum of the first n terms of {c_n}
def T (n : ℕ) : ℕ :=
  (3 * n + 1) * 2^(n + 2) - 4

-- Theorem to prove general term formula for {b_n}
theorem part_one : ∀ n : ℕ, b n = 3 * n + 2 := 
by sorry

-- Theorem to prove the sum of the first n terms of {c_n}
theorem part_two (n : ℕ) : ∀ n : ℕ, T n = (3 * n + 1) * 2^(n + 2) - 4 :=
by sorry

end part_one_part_two_l397_397261


namespace problem_1_problem_2_l397_397904

def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := -|x + 1| + 4

theorem problem_1:
  { x : ℝ // 0 ≤ x ∧ x ≤ 6 } = { x : ℝ // f x ≤ 1 } :=
sorry

theorem problem_2:
  { m : ℝ // m ≤ -3 } = { m : ℝ // ∀ x : ℝ, f x - g x ≥ m + 1 } :=
sorry

end problem_1_problem_2_l397_397904


namespace sum_of_numbers_mod_11_l397_397655

def sum_mod_eleven : ℕ := 123456 + 123457 + 123458 + 123459 + 123460 + 123461

theorem sum_of_numbers_mod_11 :
  sum_mod_eleven % 11 = 10 :=
sorry

end sum_of_numbers_mod_11_l397_397655


namespace min_rooms_needed_l397_397999

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l397_397999


namespace integer_part_div_sum_l397_397837

theorem integer_part_div_sum : 
  let s : Real := (List.sum [0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59])
  in Int.floor (16 / s) = 1 :=
by
  sorry

end integer_part_div_sum_l397_397837


namespace expected_number_of_ties_l397_397507

-- Assuming necessary definitions and functions exist in Mathlib

theorem expected_number_of_ties :
  ∀ (n : ℕ), n = 5 →
  let I (k : ℕ) : ℝ := if k % 2 = 0 then (nat.choose (2 * k) k : ℝ) / (2 : ℝ)^(2 * k) else 0 in
  let E (n : ℕ) : ℝ := ∑ k in finset.range n, I (k + 1)
  E 5 = 1.707 :=
  assume n hn,
  let I (k : ℕ) : ℝ := if k % 2 = 0 then (nat.choose (2 * k) k : ℝ) / (2 : ℝ)^(2 * k) else 0 in
  let E (n : ℕ) : ℝ := ∑ k in finset.range n, I (k + 1) in
  sorry

end expected_number_of_ties_l397_397507


namespace average_speed_round_trip_l397_397145

theorem average_speed_round_trip (v1 v2 : ℝ) (h1 : v1 = 60) (h2 : v2 = 100) :
  (2 * v1 * v2) / (v1 + v2) = 75 :=
by
  sorry

end average_speed_round_trip_l397_397145


namespace isosceles_triangle_perimeter_l397_397679

theorem isosceles_triangle_perimeter
  {a b : ℝ}
  (h_iso : (a = 3 ∧ b = 7) ∨ (a = 7 ∧ b = 3))
  (h_triangle : (a + a > b ∧ 2 * a > b) ∨ (b + b > a ∧ 2 * b > a)) :
  (2 * a + b = 17) ∨ (2 * b + a = 17) :=
begin
  sorry
end

end isosceles_triangle_perimeter_l397_397679


namespace min_rooms_needed_l397_397995

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l397_397995


namespace cos_sum_simplify_l397_397811

noncomputable def omega : ℂ := complex.exp (2 * complex.pi * complex.I / 15)

theorem cos_sum_simplify : 
  omega^15 = 1 →
  let x := real.cos (2 * real.pi / 15) + real.cos (4 * real.pi / 15) + real.cos (8 * real.pi / 15) in
  x = 1 :=
by
  sorry

end cos_sum_simplify_l397_397811


namespace lucky_sum_probability_eq_l397_397376

/--
Given that there are N balls numbered from 1 to N,
where 10 balls are selected in the main draw with their sum being 63,
and 8 balls are selected in the additional draw with their sum being 44,
we need to prove that N = 18 such that the events are equally likely.
-/
theorem lucky_sum_probability_eq (N : ℕ) (h1 : ∃ (S : Finset ℕ), S.card = 10 ∧ S.sum id = 63) 
    (h2 : ∃ (T : Finset ℕ), T.card = 8 ∧ T.sum id = 44) : N = 18 :=
sorry

end lucky_sum_probability_eq_l397_397376


namespace find_n_l397_397132

theorem find_n (n : ℕ) : (Nat.lcm n 10 = 36) ∧ (Nat.gcd n 10 = 5) → n = 18 :=
by
  -- The proof will be provided here
  sorry

end find_n_l397_397132


namespace find_b1_l397_397847

theorem find_b1 (k b_1 : ℕ) (b : ℕ → ℕ)
    (h1 : ∀ n ≥ 2, k * b_1 + k * (∑ i in finset.range n, b i) = n^2 * k * b n)
    (h2 : b 70 = 2)
    (h3 : k = 3) :
    b_1 = 4970 :=
sorry

end find_b1_l397_397847


namespace remainder_of_sum_mod_11_l397_397652

theorem remainder_of_sum_mod_11 :
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 :=
by
  sorry

end remainder_of_sum_mod_11_l397_397652


namespace proof_AK_eq_CK_l397_397806

theorem proof_AK_eq_CK 
  (ABC : Type) [triangle ABC]
  (O : point) 
  (circumcenter ABC O) 
  (acute_angled ABC) 
  (N : point) (M : point)
  (on_side N ABC AB)
  (on_side M ABC BC)
  (angle_eq : ∠ ABC A = ∠ N O A)
  (angle_eq' : ∠ ABC C = ∠ M O C)
  (K : point)
  (circumcenter (triangle M B N) K) :
  AK = CK :=
sorry

end proof_AK_eq_CK_l397_397806


namespace machines_collective_production_l397_397800

-- Define constant rates and times for each machine
def rateA (rateB : ℝ) : ℝ := 2 * rateB
def rateB : ℝ := 100 / 20
def rateC : ℝ := 100 / 15
def rateD : ℝ := 100 / 10

-- Define the number of parts each machine produces in 45 minutes
def partsInTime (rate : ℝ) (time : ℝ) : ℝ := rate * time
def partsA (t : ℝ) : ℝ := partsInTime (rateA rateB) t
def partsB (t : ℝ) : ℝ := partsInTime rateB t
def partsC (t : ℝ) : ℝ := partsInTime rateC t
def partsD (t : ℝ) : ℝ := partsInTime rateD t

-- Define the total number of parts produced collectively by all machines in 45 minutes
def totalParts (t : ℝ) : ℝ := partsA t + partsB t + partsC t + partsD t
def t := 45

theorem machines_collective_production : totalParts 45 = 1425 :=
by
  sorry

end machines_collective_production_l397_397800


namespace billboards_in_third_hour_l397_397426

-- Define the given conditions
def first_hour_billboards : Nat := 17
def second_hour_billboards : Nat := 20
def average_billboards_per_hour : Nat := 20
def total_hours : Nat := 3

-- Compute the total number of billboards expected
def total_expected_billboards : Nat := average_billboards_per_hour * total_hours

-- Theorem statement: compute number of billboards in the third hour based on the given conditions
theorem billboards_in_third_hour :
  first_hour_billboards + second_hour_billboards + x = total_expected_billboards → x = 23 := 
by
  intro h
  simp only [first_hour_billboards, second_hour_billboards, total_expected_billboards] at h
  linarith
  sorry

end billboards_in_third_hour_l397_397426


namespace water_percentage_in_honey_l397_397522

variables (nectar_weight honey_weight : ℝ)
variables (nectar_water_percentage solids_percentage : ℝ)

-- Conditions
def processed_nectar := 1.7 -- kg of nectar
def honey_yield := 1.0 -- kg of honey
def nectar_contains_water := 0.5 -- 50% water
def nectar_contains_solids := 0.5 -- 50% solids

-- Theorem to be proved
theorem water_percentage_in_honey : 
  (nectar_weight = 1.7 ∧ honey_weight = 1 ∧ nectar_water_percentage = 0.5 ∧ solids_percentage = 0.5) →
  ( (honey_yield - (nectar_weight * solids_percentage)) / honey_yield ) * 100 = 15 :=
by 
  intros h
  sorry

end water_percentage_in_honey_l397_397522


namespace stream_speed_l397_397850

theorem stream_speed (x : ℝ) (d : ℝ) (v_b : ℝ) (t : ℝ) (h : v_b = 8) (h1 : d = 210) (h2 : t = 56) : x = 2 :=
by
  sorry

end stream_speed_l397_397850


namespace alice_prime_sum_l397_397587

-- Conditions and problem definitions
def smallest_prime_divisor (n : ℕ) : ℕ :=
  if (n = 2) then 2 else (2 :: (List.range' 3 (n - 2)).filter (λ d, n % d = 0 ∧ Nat.Prime d)).head!

def next_number (a_k : ℕ) : ℕ := a_k - (smallest_prime_divisor a_k)

-- Proof problem
theorem alice_prime_sum : 
  ∀ (a_2022 : ℕ), 
    Nat.Prime a_2022 → 
    a_2022 = 2 → 
    (∃ a_0 : ℕ, 
     (∀ n < 2022, next_number (a_0 - (2 * n)) = a_0 - (2 * (n + 1))) ∧ 
     (a_0 - 4044 = a_2022)) → 
    (a_0 = 4046 ∨ a_0 = 4047) → 
    (4046 + 4047 = 8093) := 
by 
  sorry

end alice_prime_sum_l397_397587


namespace Tamara_is_95_inches_l397_397447

/- Defining the basic entities: Kim's height (K), Tamara's height, Gavin's height -/
def Kim_height (K : ℝ) := K
def Tamara_height (K : ℝ) := 3 * K - 4
def Gavin_height (K : ℝ) := 2 * K + 6

/- Combined height equation -/
def combined_height (K : ℝ) := (Tamara_height K) + (Kim_height K) + (Gavin_height K) = 200

/- Given that Kim's height satisfies the combined height condition,
   proving that Tamara's height is 95 inches -/
theorem Tamara_is_95_inches (K : ℝ) (h : combined_height K) : Tamara_height K = 95 :=
by
  sorry

end Tamara_is_95_inches_l397_397447


namespace second_container_sand_capacity_l397_397151

def volume (h: ℕ) (w: ℕ) (l: ℕ) : ℕ := h * w * l

def sand_capacity (v1: ℕ) (s1: ℕ) (v2: ℕ) : ℕ := (s1 * v2) / v1

theorem second_container_sand_capacity:
  let h1 := 3
  let w1 := 4
  let l1 := 6
  let s1 := 72
  let h2 := 3 * h1
  let w2 := 2 * w1
  let l2 := l1
  let v1 := volume h1 w1 l1
  let v2 := volume h2 w2 l2
  sand_capacity v1 s1 v2 = 432 :=
by {
  sorry
}

end second_container_sand_capacity_l397_397151


namespace sales_profit_equation_l397_397146

variable (x : ℝ) (cost selling_price initial_sales : ℝ)
variable (sales_decrease_per_yuan profit_target : ℝ)

-- Assign the given values to the variables
def cost : ℝ := 40
def selling_price : ℝ := 50
def initial_sales : ℝ := 500
def sales_decrease_per_yuan : ℝ := 10
def profit_target : ℝ := 8000

axiom x_gt_50 : x > 50

theorem sales_profit_equation
    (h1 : cost = 40)
    (h2 : selling_price = 50)
    (h3 : initial_sales = 500)
    (h4 : sales_decrease_per_yuan = 10)
    (h5 : profit_target = 8000)
    (h6 : x > 50) :
  (x - cost) * (initial_sales - sales_decrease_per_yuan * (x - selling_price)) = profit_target :=
begin
  sorry
end

end sales_profit_equation_l397_397146


namespace necessary_but_not_sufficient_l397_397429

variable {α : Type} [OrderedField α] (f : α → α) (a b : α)

theorem necessary_but_not_sufficient (h₁ : ∀ x ∈ set.Ioo a b, 0 < (deriv^[2]) f x) (h₂ : ∀ x ∈ set.Ioo a b, deriv f x > 0) :
  (∀ x ∈ set.Ioo a b, 0 < deriv f x) ↔ (∀ x ∈ set.Ioo a b, deriv f x > 0 ∧ ContinuousOn (deriv f) (set.Ioo a b)) :=
by
  split
  sorry

end necessary_but_not_sufficient_l397_397429


namespace remainder_when_divided_by_8_l397_397521

theorem remainder_when_divided_by_8 (x : ℤ) (k : ℤ) (h : x = 72 * k + 19) : x % 8 = 3 :=
by sorry

end remainder_when_divided_by_8_l397_397521


namespace remainder_is_310_l397_397412

noncomputable def remainder_of_increasing_order_numbers : ℕ := 310

theorem remainder_is_310 :
  let M := nat.choose 17 8 in
  M % 1000 = remainder_of_increasing_order_numbers := by
  sorry

end remainder_is_310_l397_397412


namespace students_called_in_sick_l397_397623

-- Conditions
def total_cupcakes : ℕ := 2 * 12 + 6
def total_people : ℕ := 27 + 1 + 1
def cupcakes_left : ℕ := 4
def cupcakes_given_out : ℕ := total_cupcakes - cupcakes_left

-- Statement to prove
theorem students_called_in_sick : total_people - cupcakes_given_out = 3 := by
  -- The proof steps would be implemented here
  sorry

end students_called_in_sick_l397_397623


namespace rs_length_l397_397336

/-- In triangle PQR with angle PQR = 120 degrees, PQ = 4, and QR = 5,
    if perpendiculars are constructed from P to PQ and from R to QR,
    and they meet at S, then RS is equal to 5√3/2. -/
theorem rs_length
  (P Q R S : Type)
  (angle_PQR : ℝ)
  (PQ QR : ℝ)
  (perpendicular_at_P : is_perpendicular_to PQ at P)
  (perpendicular_at_R : is_perpendicular_to QR at R)
  (meet_at_S : meet_at_perpendiculars PQ QR = S) :
  angle_PQR = 120 ∧ PQ = 4 ∧ QR = 5 → distance R S = 5 * sqrt 3 / 2 :=
by sorry

end rs_length_l397_397336


namespace descent_property_l397_397809

def quadratic_function (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

theorem descent_property (x : ℝ) (h : x < 3) : (quadratic_function (x + 1) < quadratic_function x) :=
sorry

end descent_property_l397_397809


namespace product_102_108_l397_397196

theorem product_102_108 : (102 = 105 - 3) → (108 = 105 + 3) → (102 * 108 = 11016) := by
  sorry

end product_102_108_l397_397196


namespace exists_integer_solution_l397_397310

theorem exists_integer_solution (a b c : ℤ) :
  let d := Int.gcd a b in
  (∃ x y : ℤ, a * x + b * y = c) ↔ d ∣ c :=
by
  sorry

end exists_integer_solution_l397_397310


namespace Chris_age_l397_397455

variable (a b c : ℕ)

theorem Chris_age : a + b + c = 36 ∧ b = 2*c + 9 ∧ b = a → c = 4 :=
by
  sorry

end Chris_age_l397_397455


namespace calc_expression_l397_397184

theorem calc_expression : 3 * 3^4 - 9^32 / 9^30 = 162 := by
  -- We would provide the proof here, but skipping with sorry
  sorry

end calc_expression_l397_397184


namespace count_diff_of_squares_excluding_5_l397_397731

theorem count_diff_of_squares_excluding_5 :
  let is_diff_of_squares (n : ℕ) : Prop := ∃ a b : ℕ, n = a^2 - b^2
  let integers_between (x y : ℕ) : List ℕ := List.filter (λ n, x ≤ n ∧ n ≤ y) (List.range (y + 1))
  let count_valid_numbers : ℕ :=
      (integers_between 1 2000).count (λ n, is_diff_of_squares n ∧ ¬ (n % 5 = 0))
  count_valid_numbers = 1200 :=
by
  -- Proof omitted
  sorry

end count_diff_of_squares_excluding_5_l397_397731


namespace coloring_satisfies_conditions_l397_397425

-- Definitions of point colors
inductive Color
| Red
| White
| Black

def color_point (x y : ℤ) : Color :=
  if (x + y) % 2 = 1 then Color.Red
  else if (x % 2 = 1 ∧ y % 2 = 0) then Color.White
  else Color.Black

-- Problem statement
theorem coloring_satisfies_conditions :
  (∀ y : ℤ, ∃ x1 x2 x3 : ℤ, 
    color_point x1 y = Color.Red ∧ 
    color_point x2 y = Color.White ∧
    color_point x3 y = Color.Black)
  ∧ 
  (∀ (x1 y1 x2 y2 x3 y3 : ℤ),
    color_point x1 y1 = Color.White →
    color_point x2 y2 = Color.Red →
    color_point x3 y3 = Color.Black →
    ∃ x4 y4, 
      color_point x4 y4 = Color.Red ∧ 
      x4 = x3 + (x1 - x2) ∧ 
      y4 = y3 + (y1 - y2)) :=
by
  sorry

end coloring_satisfies_conditions_l397_397425


namespace a_4_is_4_l397_397292

-- Define the general term formula of the sequence
def a (n : ℕ) : ℤ := (-1)^n * n

-- State the desired proof goal
theorem a_4_is_4 : a 4 = 4 :=
by
  -- Proof to be provided here,
  -- adding 'sorry' as we are only defining the statement, not solving it
  sorry

end a_4_is_4_l397_397292


namespace negation_universal_proposition_l397_397478

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, 0 < x → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, 0 < x ∧ x^2 + x + 1 ≤ 0) :=
by
  sorry

end negation_universal_proposition_l397_397478


namespace period_cosine_function_l397_397516

def B : ℝ := 3

def f : ℝ → ℝ := λ x => cos (B * x - π / 4)

theorem period_cosine_function : (∃ T : ℝ, T = 2 * π / |B| ∧ ∀ x : ℝ, f (x + T) = f x) :=
by
  have h : B ≠ 0 := by linarith
  exists (2 * π / |B|)
  split
  · sorry -- Prove T = 2 * π / |B|
  · intro x
    sorry -- Prove f (x + T) = f x

end period_cosine_function_l397_397516


namespace primes_with_prime_remainders_count_l397_397308

-- Define a predicate to check for prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime remainders when divided by 12
def prime_remainders := {1, 2, 3, 5, 7, 11}

-- Function to list primes between 50 and 100
def primes_between_50_and_100 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Function to count such primes with prime remainder when divided by 12
noncomputable def count_primes_with_prime_remainder : ℕ :=
  list.count (λ n, n % 12 ∈ prime_remainders) primes_between_50_and_100

-- The theorem to state the problem in Lean
theorem primes_with_prime_remainders_count : count_primes_with_prime_remainder = 10 :=
by {
  /- proof steps to be provided here, if required. -/
 sorry
}

end primes_with_prime_remainders_count_l397_397308


namespace pairs_divisible_by_4_l397_397016

-- Define the set of valid pairs of digits from 00 to 99
def valid_pairs : List (Fin 100) := List.filter (λ n => n % 4 = 0) (List.range 100)

-- State the theorem
theorem pairs_divisible_by_4 : valid_pairs.length = 25 := by
  sorry

end pairs_divisible_by_4_l397_397016


namespace polynomial_roots_l397_397231

noncomputable def polynomial := 6 * X ^ 5 + 29 * X ^ 4 - 71 * X ^ 3 - 10 * X ^ 2 + 24 * X + 8

theorem polynomial_roots :
  (∀ x : ℚ, polynomial.eval x polynomial = 0 ↔ x ∈ {-2, 1/2, 1, 4/3, -2/3}) :=
sorry

end polynomial_roots_l397_397231


namespace cosine_midline_l397_397604

theorem cosine_midline (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_range : ∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) : 
  d = 3 := 
by 
  sorry

end cosine_midline_l397_397604


namespace sasha_fractions_l397_397036

theorem sasha_fractions (x y z t : ℕ) 
  (hx : x ≠ y) (hxy : x ≠ z) (hxz : x ≠ t)
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t) :
  ∃ (q1 q2 : ℚ), (q1 ≠ q2) ∧ 
    (q1 = x / y ∨ q1 = x / z ∨ q1 = x / t ∨ q1 = y / x ∨ q1 = y / z ∨ q1 = y / t ∨ q1 = z / x ∨ q1 = z / y ∨ q1 = z / t ∨ q1 = t / x ∨ q1 = t / y ∨ q1 = t / z) ∧ 
    (q2 = x / y ∨ q2 = x / z ∨ q2 = x / t ∨ q2 = y / x ∨ q2 = y / z ∨ q2 = y / t ∨ q2 = z / x ∨ q2 = z / y ∨ q2 = z / t ∨ q2 = t / x ∨ q2 = t / y ∨ q2 = t / z) ∧ 
    |q1 - q2| ≤ 11 / 60 := by 
  sorry

end sasha_fractions_l397_397036


namespace range_of_a1_l397_397675

-- Define parameters and conditions based on the problem
variables {α : Type*} [field α] [sin α] [cos α]

-- Define the arithmetic sequence
def arithmetic_sequence (a_1 d : α) (n : ℕ) : α := a_1 + n * d

theorem range_of_a1 (a_1 d : α) (h1 : d ∈ Ioo (-1 : α) (0 : α))
                      (h2 : ∀ n, n ≠ 9 → arithmetic_sequence a_1 d n + arithmetic_sequence a_1 d (n + 1) / 2 ≠ 
                      ((arithmetic_sequence a_1 d 6)^2 * (arithmetic_sequence a_1 d 9)^2 - 
                       (arithmetic_sequence a_1 d 9)^2 * (arithmetic_sequence a_1 d 6)^2) / 
                      sin (arithmetic_sequence a_1 d 7 + arithmetic_sequence a_1 d 8)) :
  a_1 ∈ set.Ioo (4 * π / 3) (3 * π / 2) :=
sorry

end range_of_a1_l397_397675


namespace excircle_power_of_point_l397_397400

/-- Let I_B be the center of the excircle opposite to vertex B of triangle ABC. 
    Let M be the midpoint of the arc BC of the circumcircle ω of ΔABC, that does not include A.
    Let T be the intersection point of the line MI_B with the circumcircle ω.
    We aim to prove that TI_B^2 = TB * TC. -/
theorem excircle_power_of_point 
  (ABC : Type) 
  [triangle ABC] 
  [excircle_center I_B opposite B of ABC] 
  [circumcircle ω of ABC]
  [midpoint M of arc BC of ω, excluding A]
  (T ∈ line(I_B, M) ∩ ω) 
  : (TI_B) ^ 2 = (TB) * (TC) :=
sorry

end excircle_power_of_point_l397_397400


namespace f_log2_7_l397_397246

noncomputable def f : ℝ → ℝ
| x => if x > 0 then f (x - 2) else 2^x - 1

theorem f_log2_7 : f (Real.log 7 / Real.log 2) = -9/16 := sorry

end f_log2_7_l397_397246


namespace regular_polygon_diag_intersections_l397_397254

theorem regular_polygon_diag_intersections 
  {n : ℕ} (hn : n ≥ 5)
  (A : Fin (2 * n) → Point)
  (O : Point)
  (h_regular : is_regular_2n_gon A O n)
  (F : Point)
  (hF : ∃ k m: Fin (2 * n), k.val = 2 ∧ m.val = n - 1 ∧ diag_intersect A k m = F)
  (P : Point)
  (hP : ∃ i j : Fin (2 * n), i.val = 1 ∧ j.val = 3 ∧ diag_intersect A i j = P) :
  dist P F = dist P O := sorry

end regular_polygon_diag_intersections_l397_397254


namespace clock_strikes_total_l397_397148

theorem clock_strikes_total (h: ℕ) (h_half: ℕ) (hours_in_a_day: ℕ) (total_strikes: ℕ) :
  (∀ x, 1 ≤ x ∧ x ≤ 12 → h = x) ∧ (h_half = 1) ∧ (hours_in_a_day = 24) ∧
  (total_strikes = 2 * (1 + 2 + ... + 12) + hours_in_a_day) →
  total_strikes = 180 :=
by
  sorry

end clock_strikes_total_l397_397148


namespace relation_among_abc_l397_397269

def a : ℝ := 2 ^ 0.6
def b : ℝ := Real.logb 2 2
def c : ℝ := Real.log 0.6

theorem relation_among_abc : a > b ∧ b > c := by
  have h1 : a > 1 := sorry -- Proof that 2^0.6 > 1
  have h2 : b = 1 := sorry -- Proof that log_2(2) = 1
  have h3 : c < 0 := sorry -- Proof that ln(0.6) < 0
  exact ⟨h1.trans (by rw h2), h2.symm.trans_lt h3⟩

end relation_among_abc_l397_397269


namespace quadrilateral_area_l397_397226

theorem quadrilateral_area (d h1 h2 : ℝ) (hd : d = 40) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  1 / 2 * d * h1 + 1 / 2 * d * h2 = 300 := 
by
  sorry

end quadrilateral_area_l397_397226


namespace smaller_number_is_neg_five_l397_397079

theorem smaller_number_is_neg_five (x y : ℤ) (h1 : x + y = 30) (h2 : x - y = 40) : y = -5 :=
by
  sorry

end smaller_number_is_neg_five_l397_397079


namespace area_ratio_triangle_l397_397199

noncomputable def area_ratio (x y : ℝ) (n m : ℕ) : ℝ :=
(x * y) / (2 * n) / ((x * y) / (2 * m))

theorem area_ratio_triangle (x y : ℝ) (n m : ℕ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  area_ratio x y n m = (m : ℝ) / (n : ℝ) := by
  sorry

end area_ratio_triangle_l397_397199


namespace exponentiation_condition_l397_397666

theorem exponentiation_condition (a b : ℝ) (h0 : a > 0) (h1 : a ≠ 1) : 
  (a ^ b > 1 ↔ (a - 1) * b > 0) :=
sorry

end exponentiation_condition_l397_397666


namespace summation_Gn_over_5n_value_l397_397402
noncomputable def modified_fibonacci_seq : ℕ → ℕ
| 0 := 2
| 1 := 1
| (n + 2) := modified_fibonacci_seq (n + 1) + modified_fibonacci_seq n

noncomputable def summation_Gn_over_5n : ℝ := ∑' (n : ℕ), (modified_fibonacci_seq n : ℝ) / (5 ^ n)

theorem summation_Gn_over_5n_value : summation_Gn_over_5n = 280 / 99 :=
by
  sorry

end summation_Gn_over_5n_value_l397_397402


namespace derivative_at_pi_div_2_l397_397669

noncomputable def f (x : ℝ) : ℝ := x * Real.sin (2 * x)

theorem derivative_at_pi_div_2 : deriv f (Real.pi / 2) = -Real.pi := by
  sorry

end derivative_at_pi_div_2_l397_397669


namespace length_of_second_train_l397_397140

/-- Here we define the given conditions -/
def first_train_length : ℝ := 350
def first_train_speed_km_per_hr : ℝ := 120
def second_train_speed_km_per_hr : ℝ := 100
def crossing_time_seconds : ℝ := 8

/-- Convert speeds from km/hr to m/s -/
def km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  (speed * 1000) / 3600

def first_train_speed_m_per_s : ℝ :=
  km_per_hr_to_m_per_s first_train_speed_km_per_hr

def second_train_speed_m_per_s : ℝ :=
  km_per_hr_to_m_per_s second_train_speed_km_per_hr

/-- Calculate the relative speed of the two trains -/
def relative_speed : ℝ :=
  first_train_speed_m_per_s + second_train_speed_m_per_s

/-- Calculate the total distance covered when the trains cross each other -/
def total_distance_covered : ℝ :=
  relative_speed * crossing_time_seconds

/-- Prove the length of the second train L in meters, given the conditions -/
theorem length_of_second_train : 
  let L := total_distance_covered - first_train_length in
  L = 138.88 := by
  sorry

end length_of_second_train_l397_397140


namespace distance_z111_from_origin_l397_397202

noncomputable def seq : ℕ → ℂ
| 1     := 0
| (n+1) := (seq n)^2 + 1 + complex.I

theorem distance_z111_from_origin : complex.abs (seq 111) = 7 * real.sqrt 2 :=
by sorry

end distance_z111_from_origin_l397_397202


namespace solve_inequality_l397_397327

-- Given conditions
variables (a b c : ℝ) (a_neg : a < 0)

-- Define the roots of the quadratic equation ax^2 + bx + c = 0
def roots (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the solution sets in the problem statement
def sol_set_geq (x : ℝ) : Prop := -1/3 ≤ x ∧ x ≤ 2
def sol_set_lt (x : ℝ) : Prop := -3 < x ∧ x < 1/2

-- Main statement to be proved
theorem solve_inequality :
  (∀ x, sol_set_geq a b c x ↔ x ∈ Icc (-1/3) (2)) ∧ 
  (a_neg → (∀ x, a * x^2 + b * x + c ≥ 0 → sol_set_geq x)) →
  (∀ x, c * x^2 + b * x + a < 0 ↔ sol_set_lt x) :=
by sorry

end solve_inequality_l397_397327


namespace vector_magnitude_l397_397296

open Real

theorem vector_magnitude (x : ℝ) (h : (1:ℝ) * x + (-1) * (2:ℝ) = 0) :
  sqrt ((1 + x) ^ 2 + (2 - 1) ^ 2) = sqrt 10 := 
by
  have hx : x = 2 := by linarith
  rw [hx]
  norm_num
  sorry

end vector_magnitude_l397_397296


namespace constant_term_in_binomial_expansion_l397_397766

theorem constant_term_in_binomial_expansion (n : ℕ) (h_pos : 0 < n)
    (h_eq_coeff : binomial n 2 = binomial n 4) : 
    (constant_term_in_expansion (x + x⁻¹) n) = 20 :=
sorry

def binomial (n k : ℕ) : ℕ :=
if h : k ≤ n then nat.choose n k else 0

def constant_term_in_expansion (expr : ℕ → ℕ → ℝ) (n : ℕ) : ℕ :=
-- Define the function that calculates the constant term in the binomial expansion
sorry

end constant_term_in_binomial_expansion_l397_397766


namespace original_number_is_28_l397_397116

theorem original_number_is_28 (N : ℤ) :
  (∃ k : ℤ, N - 11 = 17 * k) → N = 28 :=
by
  intro h
  obtain ⟨k, h₁⟩ := h
  have h₂: N = 17 * k + 11 := by linarith
  have h₃: k = 1 := sorry
  linarith [h₃]
 
end original_number_is_28_l397_397116


namespace tan_sum_eq_two_l397_397403

theorem tan_sum_eq_two (a b c : ℝ) (A B C : ℝ) (h1 : a / b + b / a = 4 * Real.cos C)
  (h2 : a = b * Real.sin A / Real.sin B) (h3 : c = (a^2 + b^2 - 2 * a * b * Real.cos C) / a)
  (h4 : C = Real.acos (a^2 + b^2 - c^2) / (2 * a * b)) :
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 2 :=
by
  sorry

end tan_sum_eq_two_l397_397403


namespace hexagon_projection_area_theorem_l397_397824

noncomputable def hexagon_projected_area (α : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * Real.cos α

theorem hexagon_projection_area_theorem (α : ℝ) :
  let S := hexagon_projected_area α in
  S = (3 * Real.sqrt 3 / 2) * Real.cos α :=
by
  let S := hexagon_projected_area α
  -- Using the definition to conclude the proof
  show S = (3 * Real.sqrt 3 / 2) * Real.cos α,
  from rfl

end hexagon_projection_area_theorem_l397_397824


namespace total_weight_of_8_moles_BaBr2_l397_397514

noncomputable def atomic_weight_Ba : ℝ := 137.33
noncomputable def atomic_weight_Br : ℝ := 79.90
noncomputable def moles_BaBr2 : ℝ := 8

def molecular_weight_BaBr2 : ℝ :=
  atomic_weight_Ba + 2 * atomic_weight_Br

def total_weight (molecular_weight : ℝ) (moles : ℝ) : ℝ :=
  molecular_weight * moles

theorem total_weight_of_8_moles_BaBr2 :
  total_weight molecular_weight_BaBr2 moles_BaBr2 = 2377.04 :=
by
  sorry

end total_weight_of_8_moles_BaBr2_l397_397514


namespace rational_cos_finite_set_l397_397778

theorem rational_cos_finite_set (x y : ℝ) (A : set ℝ) 
  (h : A = {z | ∃ n : ℕ, z = cos (n * π * x) + cos (n * π * y)} ∧ A.finite) : 
  x ∈ ℚ ∧ y ∈ ℚ := 
sorry

end rational_cos_finite_set_l397_397778


namespace surface_area_of_sphere_l397_397577

theorem surface_area_of_sphere (d : ℝ) (A : ℝ) (h₁ : d = 1) (h₂ : A = π) :
  4 * π * (real.sqrt (1 + 1))^2 = 8 * π := 
by
  sorry

end surface_area_of_sphere_l397_397577


namespace min_rooms_needed_l397_397994

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l397_397994


namespace length_segment_AB_max_area_triangle_ABP_l397_397288

noncomputable def line_l : ℝ × ℝ → Prop := λ t, (t.1 = 2 + t.2) ∧ (t.1 = -1 - t.2)

def curve_C : ℝ × ℝ → Prop := λ p, p.1^2 + p.2^2 = 9

theorem length_segment_AB : 
  ∃ A B : ℝ × ℝ, 
    line_l A ∧ line_l B ∧ curve_C A ∧ curve_C B ∧ 
    (dist A B = sqrt 34) :=
by
  sorry

theorem max_area_triangle_ABP : 
  ∃ A B P : ℝ × ℝ, 
    line_l A ∧ line_l B ∧ curve_C A ∧ curve_C B ∧ curve_C P ∧ 
    (let d := dist P (2, -1) - dist (2, -1) (line_l t : {t // t in [A, B]})
    in (area_triangle A B P = (sqrt 34) * d / 2) :=
by
  sorry

end length_segment_AB_max_area_triangle_ABP_l397_397288


namespace hyperbola_standard_eq_hyperbola_properties_l397_397718

section HyperbolaProposition

variables (A B : ℝ × ℝ)
-- Conditions
def point_A : ℝ × ℝ := (9 / 4, 5)
def point_B : ℝ × ℝ := (3, -4 * Real.sqrt 2)

-- The standard equation of the hyperbola
theorem hyperbola_standard_eq (A := point_A) (B := point_B) : 
  ∃ m n : ℝ, m = 9 ∧ n = 16 ∧ 
  (A.1^2 / m - A.2^2 / n = 1) ∧ 
  (B.1^2 / m - B.2^2 / n = 1) ∧ 
  (m * n < 0) :=
sorry

-- The other properties of the hyperbola
theorem hyperbola_properties (A := point_A) (B := point_B) : 
  let a := 4 in let b := 3 in let c := 5 in
  let f1 := (0, -c) in let f2 := (0, c) in
  let real_axis := 2 * a in let imag_axis := 2 * b in let e := c / a in 
  (a^2 = 16) ∧ (b^2 = 9) ∧ (c = (Real.sqrt (16 + 9))) ∧ 
  (f1 = (0, -5)) ∧ (f2 = (0, 5)) ∧
  (real_axis = 8) ∧ (imag_axis = 6) ∧ (e = 5 / 4) :=
sorry

end HyperbolaProposition

end hyperbola_standard_eq_hyperbola_properties_l397_397718


namespace paul_bought_150_books_l397_397805

theorem paul_bought_150_books (initial_books sold_books books_now : ℤ)
  (h1 : initial_books = 2)
  (h2 : sold_books = 94)
  (h3 : books_now = 58) :
  initial_books - sold_books + books_now = 150 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end paul_bought_150_books_l397_397805


namespace probability_all_vertical_faces_green_l397_397969

theorem probability_all_vertical_faces_green :
  let color_prob := (1 / 2 : ℚ)
  let total_arrangements := 2^6
  let valid_arrangements := 2 + 12 + 6
  ((valid_arrangements : ℚ) / total_arrangements) = 5 / 16 := by
  sorry

end probability_all_vertical_faces_green_l397_397969


namespace max_planes_l397_397349

/-!
# Number of Planes Determined by Five Points

Given five points in three-dimensional space where no three points are collinear, and only four triangular pyramids can be constructed, we aim to prove that these points can determine at most nine planes.
-/

-- Definitions of points and their properties
noncomputable def points : Type := ℝ × ℝ × ℝ

def non_collinear (p1 p2 p3 : points) : Prop :=
  ∃ v1 v2 : points, ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1 ∧
    v1 ≠ v2 ∧
    p1 = (a • v1 + p2) ∧ p2 = (b • v2 + p3)

def tetrahedron (p1 p2 p3 p4 : points) : Prop :=
  non_collinear p1 p2 p3 ∧
  non_collinear p1 p2 p4 ∧
  non_collinear p1 p3 p4 ∧
  non_collinear p2 p3 p4

-- Main theorem to prove
theorem max_planes (p1 p2 p3 p4 p5 : points)
  (h1 : non_collinear p1 p2 p3)
  (h2 : non_collinear p1 p2 p4)
  (h3 : non_collinear p1 p2 p5)
  (h4 : non_collinear p1 p3 p4)
  (h5 : non_collinear p1 p3 p5)
  (h6 : non_collinear p1 p4 p5)
  (h7 : non_collinear p2 p3 p4)
  (h8 : non_collinear p2 p3 p5)
  (h9 : non_collinear p2 p4 p5)
  (h10 : non_collinear p3 p4 p5)
  (h11 : tetrahedron p1 p2 p3 p4)
  (h12 : tetrahedron p2 p3 p4 p5)
  (h13 : tetrahedron p3 p4 p5 p1)
  (h14 : tetrahedron p4 p5 p1 p2) :
  ∃ n : ℕ, n = 9 :=
sorry

end max_planes_l397_397349


namespace g_is_odd_l397_397389

def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end g_is_odd_l397_397389


namespace team_selection_ways_l397_397498

theorem team_selection_ways :
  let ways (n k : ℕ) := Nat.choose n k
  (ways 6 3) * (ways 6 3) = 400 := 
by
  let ways := Nat.choose
  -- Proof is omitted
  sorry

end team_selection_ways_l397_397498


namespace events_equally_likely_iff_N_eq_18_l397_397364

variable (N : ℕ)

-- Define the number of combinations in the draws
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the sums of selecting balls
noncomputable def S_63_10 (N: ℕ) : ℕ := sorry -- placeholder definition
noncomputable def S_44_8 (N: ℕ) : ℕ := sorry 

-- Condition for the events being equally likely
theorem events_equally_likely_iff_N_eq_18 : 
  (S_63_10 N * C N 8 = S_44_8 N * C N 10) ↔ N = 18 :=
sorry

end events_equally_likely_iff_N_eq_18_l397_397364


namespace complete_square_k_value_l397_397739

theorem complete_square_k_value : 
  ∃ k : ℝ, ∀ x : ℝ, (x^2 - 8*x = (x - 4)^2 + k) ∧ k = -16 :=
by
  use -16
  intro x
  sorry

end complete_square_k_value_l397_397739


namespace most_suitable_method_l397_397839

theorem most_suitable_method {x : ℝ} (h : (x - 1) ^ 2 = 4) :
  "Direct method of taking square root" = "Direct method of taking square root" :=
by
  -- We observe that the equation is already in a form 
  -- that is conducive to applying the direct method of taking the square root,
  -- because the equation is already a perfect square on one side and a constant on the other side.
  sorry

end most_suitable_method_l397_397839


namespace total_weight_of_carrots_and_cucumbers_is_875_l397_397083

theorem total_weight_of_carrots_and_cucumbers_is_875 :
  ∀ (carrots : ℕ) (cucumbers : ℕ),
    carrots = 250 →
    cucumbers = (5 * carrots) / 2 →
    carrots + cucumbers = 875 := 
by
  intros carrots cucumbers h_carrots h_cucumbers
  rw [h_carrots, h_cucumbers]
  sorry

end total_weight_of_carrots_and_cucumbers_is_875_l397_397083


namespace K3_3_non_planar_l397_397090

theorem K3_3_non_planar : ¬ (∃ (φ : (K 3 3) → ℝ × ℝ), 
                      ∀ (e₁ e₂ : (K 3 3).edges),
                        e₁ ≠ e₂ → 
                          disjoint (φ ∘ e₁) (φ ∘ e₂)) := 
sorry

end K3_3_non_planar_l397_397090


namespace part1_solution_set_part2_range_of_a_l397_397707

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := abs (x + a) + abs (2 * x + 1)

-- (1) Theorem statement for a = 1
theorem part1_solution_set (x : ℝ) : f x 1 ≤ 1 ↔ -1 ≤ x ∧ x ≤ -1 / 3 := sorry

-- (2) Theorem statement for the range of a
theorem part2_range_of_a (a : ℝ) : (∀ x ∈ set.uIcc (-1 : ℝ) (-1 / 4), f x a ≤ -2 * x + 1) ↔ -3 / 4 ≤ a ∧ a ≤ 5 / 4 := sorry

end part1_solution_set_part2_range_of_a_l397_397707


namespace number_of_groups_l397_397692

theorem number_of_groups (max_value min_value interval : ℕ) (h_max : max_value = 36) (h_min : min_value = 15) (h_interval : interval = 4) : 
  ∃ groups : ℕ, groups = 6 :=
by 
  sorry

end number_of_groups_l397_397692


namespace find_percentage_in_solution_a_l397_397798

def solution_a_weight := 600
def solution_b_weight := 700
def total_weight := solution_a_weight + solution_b_weight
def percentage_in_solution_b := 0.018
def percentage_mixed_solution := 0.0174

def amount_of_liquid_x_in_solution_b := percentage_in_solution_b * solution_b_weight
def total_liquid_x_in_mixed_solution := percentage_mixed_solution * total_weight

def amount_of_liquid_x_in_solution_a (P : ℝ) := P * solution_a_weight

theorem find_percentage_in_solution_a :
  ∃ P : ℝ, amount_of_liquid_x_in_solution_a P + amount_of_liquid_x_in_solution_b = total_liquid_x_in_mixed_solution ∧ P * 100 = 1.67 :=
begin
  sorry
end

end find_percentage_in_solution_a_l397_397798


namespace lucky_sum_equal_prob_l397_397363

theorem lucky_sum_equal_prob (N : ℕ) (S63_main_draw : Finset ℕ) (S44_additional_draw : Finset ℕ)
  (hN_pos : 0 < N)
  (hS63 : S63_main_draw.card = 10)
  (hSum63 : S63_main_draw.sum id = 63)
  (hS44 : S44_additional_draw.card = 8)
  (hSum44 : S44_additional_draw.sum id = 44)
  (hS_distinct : ∀ (n ∈ S63_main_draw ∪ S44_additional_draw), n ∈ Finset.range (N + 1)) :
  N = 18 :=
by
  sorry

end lucky_sum_equal_prob_l397_397363


namespace strictly_increasing_interval_l397_397207

noncomputable def f (x : ℝ) : ℝ :=
  Real.logb (1/3) (x^2 - 4 * x + 3)

theorem strictly_increasing_interval : ∀ x y : ℝ, x < 1 → y < 1 → x < y → f x < f y :=
by
  sorry

end strictly_increasing_interval_l397_397207


namespace smallest_repeating_block_7_over_13_l397_397624

theorem smallest_repeating_block_7_over_13 : 
  ∃ n : ℕ, (∀ d : ℕ, d < n → 
  (∃ (q r : ℕ), r < 13 ∧ 10 ^ (d + 1) * 7 % 13 = q * 10 ^ n + r)) ∧ n = 6 := sorry

end smallest_repeating_block_7_over_13_l397_397624


namespace interval_monotonically_increasing_inequality_range_l397_397701

-- Definitions for the vectors and function
def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
def f (x : ℝ) := (m x).fst * (n x).fst + (m x).snd * (n x).snd

-- Proof problems
theorem interval_monotonically_increasing : 
  ∀ k : ℤ, ∀ x : ℝ, -Real.pi / 3 + ↑k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + ↑k * Real.pi 
    → ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ < f x₂ := sorry

theorem inequality_range (m : ℝ) : 
  m < 0 → ∀ x ∈ Icc 0 (Real.pi / 2), f x - m > 0 := sorry

end interval_monotonically_increasing_inequality_range_l397_397701


namespace ratio_PE_ED_l397_397897

variables (x λ y μ u v : ℝ)

-- Given conditions
def AP := x
def PB := λ * x
def BQ := y
def QC := μ * y
def PE := u
def ED := v

-- Problem statement
theorem ratio_PE_ED : u / v = 3 / 20 := 
by {
  sorry
}

end ratio_PE_ED_l397_397897


namespace ratio_male_whales_l397_397397

def num_whales_first_trip_males : ℕ := 28
def num_whales_first_trip_females : ℕ := 56
def num_whales_second_trip_babies : ℕ := 8
def num_whales_second_trip_parents_males : ℕ := 8
def num_whales_second_trip_parents_females : ℕ := 8
def num_whales_third_trip_females : ℕ := 56
def total_whales : ℕ := 178

theorem ratio_male_whales (M : ℕ) (ratio : ℕ × ℕ) 
  (h_total_whales : num_whales_first_trip_males + num_whales_first_trip_females 
    + num_whales_second_trip_babies + num_whales_second_trip_parents_males 
    + num_whales_second_trip_parents_females + M + num_whales_third_trip_females = total_whales) 
  (h_ratio : ratio = ((M : ℕ) / Nat.gcd M num_whales_first_trip_males, 
                       num_whales_first_trip_males / Nat.gcd M num_whales_first_trip_males)) 
  : ratio = (1, 2) :=
by
  sorry

end ratio_male_whales_l397_397397


namespace area_curvilinear_trapezoid_l397_397185

theorem area_curvilinear_trapezoid :
  (∫ x in -1..2, (9 - x^2)) = 24 :=
by
  -- sorry is used here to ignore the proof steps for this example
  sorry

end area_curvilinear_trapezoid_l397_397185


namespace order_of_magnitude_l397_397625

noncomputable def a : ℝ := 5^0.6
noncomputable def b : ℝ := (0.6)^5
noncomputable def c : ℝ := log (0.6) 5

theorem order_of_magnitude : c < b ∧ b < a :=
sorry

end order_of_magnitude_l397_397625


namespace function_zeros_condition_l397_397702

theorem function_zeros_condition (a : ℝ) (H : ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ 
  2 * Real.exp (2 * x1) - 2 * a * x1 + a - 2 * Real.exp 1 - 1 = 0 ∧ 
  2 * Real.exp (2 * x2) - 2 * a * x2 + a - 2 * Real.exp 1 - 1 = 0) :
  2 * Real.exp 1 - 1 < a ∧ a < 2 * Real.exp (2:ℝ) - 2 * Real.exp 1 - 1 := 
sorry

end function_zeros_condition_l397_397702


namespace cube_root_fraction_l397_397633

theorem cube_root_fraction :
  ∀ x : ℝ, (x = 22.5) → (∛(9 / x) = ∛(2 / 5)) :=
by
  intros x hx
  rw [hx]
  have h1 : (22.5 : ℝ) = 45 / 2 := by norm_num
  rw [h1]
  have h2 : ∛(9 / (45 / 2)) = ∛(9 * (2 / 45)) := by field_simp
  rw [h2]
  have h3 : 9 * (2 / 45) = 2 / 5 := by norm_num
  rw [h3]
  apply congr_arg
  sorry

end cube_root_fraction_l397_397633


namespace quadratic_function_value_l397_397162

theorem quadratic_function_value :
  ∃ (d e f : ℝ), 
    g(x) = d*x^2 + e*x + f ∧
    g(-2) = 3 ∧
    g(0) = 7 ∧
    d + e + 2 * f = 19 :=
by sorry

end quadratic_function_value_l397_397162


namespace physics_teacher_min_count_l397_397923

theorem physics_teacher_min_count 
  (maths_teachers : ℕ) 
  (chemistry_teachers : ℕ) 
  (max_subjects_per_teacher : ℕ) 
  (min_total_teachers : ℕ) 
  (physics_teachers : ℕ)
  (h1 : maths_teachers = 7)
  (h2 : chemistry_teachers = 5)
  (h3 : max_subjects_per_teacher = 3)
  (h4 : min_total_teachers = 6) 
  (h5 : 7 + physics_teachers + 5 ≤ 6 * 3) :
  0 < physics_teachers :=
  by 
  sorry

end physics_teacher_min_count_l397_397923


namespace expand_expression_l397_397632

theorem expand_expression (x y : ℕ) : 
  (x + 15) * (3 * y + 20) = 3 * x * y + 20 * x + 45 * y + 300 :=
by 
  sorry

end expand_expression_l397_397632


namespace num_integer_coordinates_between_A_and_B_l397_397550

open Real

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (100, 1000)

def line_eq (x : ℝ) : ℝ := 10 * x - 9

def is_between (p : ℝ × ℝ) :=
  A.1 < p.1 ∧ p.1 < B.1 ∧ 
  A.2 < p.2 ∧ p.2 < B.2

def is_integer_coords (p : ℝ × ℝ) :=
  p.1 ∈ ℤ ∧ p.2 ∈ ℤ

def valid_points (p : ℝ × ℝ) :=
  is_between p ∧ is_integer_coords p ∧ p.2 = line_eq p.1

def count_valid_points : ℕ :=
  Finset.card (Finset.filter valid_points (Finset.image (λ n : ℤ, (n, line_eq n)) (Finset.range 99 \ Finset.range 1)))

theorem num_integer_coordinates_between_A_and_B : count_valid_points = 8 := sorry

end num_integer_coordinates_between_A_and_B_l397_397550


namespace cookies_eaten_l397_397159

theorem cookies_eaten (original remaining : ℕ) (h_original : original = 18) (h_remaining : remaining = 9) :
    original - remaining = 9 := by
  sorry

end cookies_eaten_l397_397159


namespace remainder_of_sum_mod_11_l397_397650

theorem remainder_of_sum_mod_11 :
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 :=
by
  sorry

end remainder_of_sum_mod_11_l397_397650


namespace square_side_length_test_square_side_length_l397_397775

theorem square_side_length (a b : ℕ) (A : a = 9) (B : b = 16) : 
  let area_rect := a * b 
  let area_square := 144 
  in area_rect = area_square → sqrt 144 = 12 :=
by
  intros
  have h : area_rect = 144 := by rw [A, B]
  have hs : sqrt 144 = 12 := by norm_num
  exact hs
  
-- Test that the provided length of the square side is indeed 12 cm
theorem test_square_side_length : sqrt 144 = 12 := by norm_num

end square_side_length_test_square_side_length_l397_397775


namespace cosine_problem_l397_397312

noncomputable def given_conditions (α : ℝ) :=
  sin(3 * Real.pi + α) = -1 / 2

theorem cosine_problem (α : ℝ) (h : given_conditions α) :
  cos((7 * Real.pi) / 2 - α) = -1 / 2 :=
sorry

end cosine_problem_l397_397312


namespace slope_of_line_through_points_l397_397114

def slope (x1 y1 x2 y2 : ℚ) : ℚ := (y2 - y1) / (x2 - x1)

theorem slope_of_line_through_points : slope (-2) 4 3 (-4) = -8 / 5 := by
  sorry

end slope_of_line_through_points_l397_397114


namespace equal_likelihood_of_lucky_sums_solution_l397_397355

variables (N : ℕ)

def main_draw := 10
def additional_draw := 8

def lucky_sum_main (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = main_draw ∧ S.sum = sum
def lucky_sum_additional (N : ℕ) (sum : ℕ) := ∃ (S : finset ℕ), S.card = additional_draw ∧ S.sum = sum

theorem equal_likelihood_of_lucky_sums :
  lucky_sum_main N 63 ↔ lucky_sum_additional N 44 :=
sorry

theorem solution :
  equal_likelihood_of_lucky_sums 18 :=
sorry

end equal_likelihood_of_lucky_sums_solution_l397_397355


namespace probability_of_part_selected_l397_397337

open Rational

theorem probability_of_part_selected (total_parts : ℕ) (selected_parts : ℕ) 
  (H_total_parts : total_parts = 120) (H_selected_parts : selected_parts = 20):
  selected_parts / total_parts = 1 / 6 := by
  sorry

end probability_of_part_selected_l397_397337


namespace smallest_a_satisfying_equation_l397_397273

noncomputable def satisfies_equation (a x : ℝ) : Prop :=
  (cos (π * (a - x))) ^ 2 - 2 * cos (π * (a - x)) + cos (3 * π * x / (2 * a)) * cos ((π * x / (2 * a)) + (π / 3)) + 2 = 0

theorem smallest_a_satisfying_equation : ∃ a : ℤ, a > 0 ∧ (∃ x : ℝ, satisfies_equation a x) ∧ a = 6 :=
by sorry

end smallest_a_satisfying_equation_l397_397273


namespace ratio_arms_martians_to_aliens_l397_397930

def arms_of_aliens : ℕ := 3
def legs_of_aliens : ℕ := 8
def legs_of_martians := legs_of_aliens / 2

def limbs_of_5_aliens := 5 * (arms_of_aliens + legs_of_aliens)
def limbs_of_5_martians (arms_of_martians : ℕ) := 5 * (arms_of_martians + legs_of_martians)

theorem ratio_arms_martians_to_aliens (A_m : ℕ) (h1 : limbs_of_5_aliens = limbs_of_5_martians A_m + 5) :
  (A_m : ℚ) / arms_of_aliens = 2 :=
sorry

end ratio_arms_martians_to_aliens_l397_397930


namespace derivative_at_neg_one_l397_397735

-- Define the function f
def f (x : ℝ) : ℝ := x ^ 6

-- Define the derivative f'
def f' (x : ℝ) : ℝ := 6 * x ^ 5

-- The statement we want to prove
theorem derivative_at_neg_one : f' (-1) = -6 := sorry

end derivative_at_neg_one_l397_397735


namespace arith_seq_sum_proof_l397_397487

open Function

variable (a : ℕ → ℕ) -- Define the arithmetic sequence
variables (S : ℕ → ℕ) -- Define the sum function of the sequence

-- Conditions: S_8 = 9 and S_5 = 6
axiom S8 : S 8 = 9
axiom S5 : S 5 = 6

-- Mathematical equivalence
theorem arith_seq_sum_proof : S 13 = 13 :=
sorry

end arith_seq_sum_proof_l397_397487


namespace midpoint_of_line_cut_by_ellipse_l397_397051

theorem midpoint_of_line_cut_by_ellipse :
  let line_eq := λ x : ℝ, x + 1
  let ellipse_eq := λ x y : ℝ, x^2 + 2*y^2 - 4
  ∃ (x1 x2 : ℝ),
    ellipse_eq x1 (line_eq x1) = 0 ∧ ellipse_eq x2 (line_eq x2) = 0 ∧
    (x1 + x2) / 2 = -2 / 3 ∧
    (line_eq ((x1 + x2) / 2)) = 1 / 3 := 
sorry

end midpoint_of_line_cut_by_ellipse_l397_397051


namespace find_varphi_l397_397322

theorem find_varphi
  (ϕ : ℝ)
  (h : ∃ k : ℤ, ϕ = (π / 8) + (k * π / 2)) :
  ϕ = π / 8 :=
by
  sorry

end find_varphi_l397_397322


namespace is_optionC_quadratic_l397_397880

def is_quadratic (eq : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ eq a b c

def optionA (x y : ℕ) : Prop := x^2 - 3 * x + y = 0

def optionB (x : ℕ) : Prop := x^2 + 2 * x = 1/x

def optionC (x : ℕ) : Prop := x^2 + 5 * x = 0

def optionD (x : ℕ) : Prop := x * (x^2 - 4 * x) = 3

theorem is_optionC_quadratic : is_quadratic (λ a b c, a*x^2 + b*x + c = 0 ∧ x = x) :=
  sorry

end is_optionC_quadratic_l397_397880


namespace integer_roots_of_quadratic_l397_397704

theorem integer_roots_of_quadratic (m : ℝ) :
  (∀ (x : ℝ), x^2 + m * x - 2 * m - 1 = 0 → ∃ (z : ℤ), x = z) ↔ m ∈ {0, -8} :=
sorry

end integer_roots_of_quadratic_l397_397704


namespace no_solutions_exists_l397_397237

theorem no_solutions_exists (r : ℝ) :
  (∀ x y : ℝ, x^2 = y^2 → (x - r)^2 + y^2 ≠ 1) ↔ r ∈ Set.Ioo (negSqrt 2) (-∞) ∪ Set.Ioo (sqrt 2) (∞) :=
sorry

end no_solutions_exists_l397_397237


namespace find_coordinates_of_B_l397_397506

theorem find_coordinates_of_B (A B : ℝ × ℝ)
  (h1 : ∃ (C1 C2 : ℝ × ℝ), C1.2 = 0 ∧ C2.2 = 0 ∧ (dist C1 A = dist C1 B) ∧ (dist C2 A = dist C2 B) ∧ (A ≠ B))
  (h2 : A = (-3, 2)) :
  B = (-3, -2) :=
sorry

end find_coordinates_of_B_l397_397506


namespace inscribed_circle_diameter_l397_397875

theorem inscribed_circle_diameter (DE DF EF : ℕ) (hDE : DE = 13) (hDF : DF = 14) (hEF : EF = 15) :
  let s := (DE + DF + EF) / 2 in
  let A := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  let r := A / s in
  2 * r = 8 :=
by
  sorry

end inscribed_circle_diameter_l397_397875


namespace slant_asymptote_f_l397_397206

/-- Define the rational function f(x) -/
def f (x : ℝ) : ℝ := (3 * x ^ 2 - 2 * x - 4)/(2 * x - 5)

/-- Statement of the proof problem:
    Given the rational function f(x), prove that the equation of the slant asymptote is y = 1.5x + 4.25.
 -/
theorem slant_asymptote_f : 
  ∀ (x : ℝ), f x = (1.5 * x + 4.25) + (39 / 4) / (2 * x - 5) :=
by sorry

end slant_asymptote_f_l397_397206


namespace problem_solution_l397_397525

def expr := 1 + 1 / (1 + 1 / (1 + 1))
def answer : ℚ := 5 / 3

theorem problem_solution : expr = answer :=
by
  sorry

end problem_solution_l397_397525


namespace total_balloons_l397_397504

-- Define the number of yellow balloons each person has
def tom_balloons : Nat := 18
def sara_balloons : Nat := 12
def alex_balloons : Nat := 7

-- Prove that the total number of balloons is 37
theorem total_balloons : tom_balloons + sara_balloons + alex_balloons = 37 := 
by 
  sorry

end total_balloons_l397_397504


namespace enclosed_area_between_circles_l397_397754

theorem enclosed_area_between_circles (a : ℝ) :
  let r := a * Real.sqrt 3 / 6,
      S_Δ := Real.sqrt 3 / 4 * a^2,
      S_o := π * (a * Real.sqrt 3 / 6)^2,
      S_c := π * (0.5 * a)^2 / 6
  in S_c - (1/3) * (S_Δ - S_o) = (a^2 / 72) * (4 * π - 3 * Real.sqrt 3) := by
  -- skipping the proof as per the instructions
  sorry

end enclosed_area_between_circles_l397_397754


namespace problem1_solution_problem2_solution_l397_397536

-- Problem 1 definition
def problem1_expr : ℝ :=
  (1 / 2)^(-1) - 2 * Real.tan (Real.pi / 4) + |1 - Real.sqrt 2|

theorem problem1_solution :
  problem1_expr = Real.sqrt 2 - 1 :=
by sorry

-- Problem 2 definition
def problem2_expr (a : ℝ) : ℝ :=
  ((a / (a^2 - 4)) + 1 / (2 - a)) / ((2 * a + 4) / (a^2 + 4 * a + 4))

theorem problem2_solution :
  problem2_expr (Real.sqrt 3 + 2) = - (Real.sqrt 3) / 3 :=
by sorry

end problem1_solution_problem2_solution_l397_397536


namespace find_common_difference_l397_397404

noncomputable theory
open Real

-- Definitions
def a (c b : ℝ) := sqrt (c * b)
def log_ac_geometric (c a b : ℝ) : Prop := b^2 = a * c
def log_ac_arithmetic (a b c : ℝ) (d : ℝ) : Prop := (log c a - log b c = d) ∧ (log b c - log a b = d)

-- Theorem statement
theorem find_common_difference 
  (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : log_ac_geometric a b c)
  (d : ℝ) :
  log_ac_arithmetic a b c d → d = 3 / 2 :=
by
  sorry

end find_common_difference_l397_397404


namespace percentage_rotten_bananas_l397_397167

theorem percentage_rotten_bananas :
  let total_oranges := 600
  let total_bananas := 400
  let rotten_oranges_percentage := 0.15
  let good_condition_percentage := 0.878
  let total_fruits := total_oranges + total_bananas 
  let rotten_oranges := rotten_oranges_percentage * total_oranges 
  let good_fruits := good_condition_percentage * total_fruits
  let rotten_fruits := total_fruits - good_fruits
  let rotten_bananas := rotten_fruits - rotten_oranges
  (rotten_bananas / total_bananas) * 100 = 8 := by
  {
    -- Calculations and simplifications go here
    sorry
  }

end percentage_rotten_bananas_l397_397167


namespace remainder_when_divided_l397_397519

-- Conditions
def divisor : ℤ[X] := X^2 - 5*X + 6
def numerator : ℤ[X] := X^50

-- Correct Answer
def remainder : ℤ[X] := 3^50 * (X - 2) - 2^50 * (X - 3)

-- Theorem Statement
theorem remainder_when_divided :
  ∃ Q : ℤ[X], numerator = divisor * Q + remainder :=
by
  sorry

end remainder_when_divided_l397_397519


namespace sum_quotient_remainder_div12_l397_397886

noncomputable def original_number := 622 / 2

theorem sum_quotient_remainder_div12 :
  let n := original_number in
  let q := n / 12 in
  let r := n % 12 in
  q + r = 36 :=
by
  sorry

end sum_quotient_remainder_div12_l397_397886


namespace new_average_score_after_drop_l397_397750

theorem new_average_score_after_drop
  (avg_score : ℝ) (num_students : ℕ) (drop_score : ℝ) (remaining_students : ℕ) :
  avg_score = 62.5 →
  num_students = 16 →
  drop_score = 70 →
  remaining_students = 15 →
  (num_students * avg_score - drop_score) / remaining_students = 62 :=
by
  intros h_avg h_num h_drop h_remain
  rw [h_avg, h_num, h_drop, h_remain]
  norm_num

end new_average_score_after_drop_l397_397750


namespace range_of_a_l397_397321

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ := x^2 - a * x + 1

-- Hypothesis: the quadratic equation has exactly one root in the interval (0, 1)
def has_exactly_one_root_in_interval {a : ℝ} : Prop :=
  ∃! x, x > 0 ∧ x < 1 ∧ quadratic_eq a x = 0

-- The theorem to prove the range of a
theorem range_of_a (a : ℝ) (h : has_exactly_one_root_in_interval) : a > 2 :=
by
  sorry

end range_of_a_l397_397321


namespace solution_set_of_inequality_l397_397283

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 3^(x + 1) else real.log x / real.log (1/2)

theorem solution_set_of_inequality : {x : ℝ | f x > 1} = set.Ioo (-1 : ℝ) (1 / 2) := 
by 
  sorry

end solution_set_of_inequality_l397_397283


namespace probability_of_square_divisor_of_factorial_23_l397_397479

/-- 
Given the prime factorization of 23!, prove that the probability of 
a randomly chosen divisor of 23! being a perfect square is 1/640. 
-/
theorem probability_of_square_divisor_of_factorial_23 :
  (∃ (p : ℕ), ∏ (p_n : ℕ) in
    finset.filter (λ n, nat.prime n) (finset.range 24), 
    p_n ^ 
      (nat.factors (nat.factorial 23)).count p_n = 
    (2^19) * (3^9) * (5^4) * (7^3) * (11^2) * (13^1) * (17^1) * (19^1) * (23^1)) 
  → (∃ (total_divisors perfect_square_divisors : ℕ), 
    total_divisors = 20 * 10 * 5 * 4 * 3 * 2 * 2 * 2 * 2 ∧ 
    perfect_square_divisors = 10 * 5 * 3 * 2 * 2 * 1 * 1 * 1 * 1 ∧ 
    (perfect_square_divisors : ℚ) / total_divisors = 1 / 640) :=
begin
  sorry
end

end probability_of_square_divisor_of_factorial_23_l397_397479


namespace find_minimum_distance_l397_397297

noncomputable def minimum_distance_condition {R : Type*} [LinearOrderedField R] 
  (a b c : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ( ‖a‖ = 2 ) ∧
  ( ‖b‖ = 1 ) ∧
  ( InnerProductSpace.inner a b = 1 ) ∧
  ( InnerProductSpace.inner (a - 2 • c) (b - c) = 0 )

noncomputable def minimum_distance {R : Type*} [LinearOrderedField R] : ℝ :=
  (Real.sqrt 7 - Real.sqrt 2) / 2

theorem find_minimum_distance {R : Type*} [LinearOrderedField R] :
  ∀ (a b c : EuclideanSpace ℝ (Fin 2)),
    minimum_distance_condition a b c → 
    ∃ d : ℝ, d = minimum_distance ∧ d = dist a c := 
by
  intros a b c h
  unfold minimum_distance_condition at h
  sorry

end find_minimum_distance_l397_397297


namespace lucky_sum_probability_eq_l397_397377

/--
Given that there are N balls numbered from 1 to N,
where 10 balls are selected in the main draw with their sum being 63,
and 8 balls are selected in the additional draw with their sum being 44,
we need to prove that N = 18 such that the events are equally likely.
-/
theorem lucky_sum_probability_eq (N : ℕ) (h1 : ∃ (S : Finset ℕ), S.card = 10 ∧ S.sum id = 63) 
    (h2 : ∃ (T : Finset ℕ), T.card = 8 ∧ T.sum id = 44) : N = 18 :=
sorry

end lucky_sum_probability_eq_l397_397377


namespace find_angle_A_range_of_perimeter_l397_397264

variables {A B C a b c : ℝ}

-- Let angles A, B, and C be the angles of triangle ABC
-- Let a, b, and c be the sides opposite to angles A, B, and C respectively
-- Given condition in the problem
axiom given_condition : 2 * a * Real.cos C + c = 2 * b

-- Ultimate proofs we are going to state
theorem find_angle_A (h : given_condition) : A = π/3 :=
sorry

theorem range_of_perimeter (hA : A = π/3) (ha : a = 1) : 
  let b := 2 * (Real.sin B) / (Real.sqrt 3)
  let c := 2 * (Real.sin C) / (Real.sqrt 3)
  let perimeter := a + b + c
  2 < perimeter ∧ perimeter ≤ 3 :=
sorry

end find_angle_A_range_of_perimeter_l397_397264


namespace total_handshakes_unique_l397_397942

theorem total_handshakes_unique (twins_sets : ℕ) (triplets_sets : ℕ)
  (twins_shake : ∀ x : ℕ, x ∈ finset.range twins_sets → x ≠ x.succ)
  (triplets_shake : ∀ y : ℕ, y ∈ finset.range triplets_sets → y ≠ y.succ)
  (cross_shake_twins : ℕ) (cross_shake_triplets : ℕ)
  (h_t : twins_sets = 7) (h_tn : 2 * 7 = 14) 
  (h_tr : triplets_sets = 4) (h_trn : 3 * 4 = 12)
  (h_w1 : ∀ x, x ∈ finset.range 14 → ∃ ys : finset ℕ, ys.card = 12)
  (h_w2 : ∀ x, x ∈ finset.range 14 → ∃ zs : finset ℕ, zs.card = 4)
  (h_w3 : ∀ y, y ∈ finset.range 12 → ∃ xs : finset ℕ, xs.card = 9)
  (h_w4 : ∀ y, y ∈ finset.range 12 → ∃ ws : finset ℕ, ws.card = 3) :
  ∃ total_handshakes : ℕ, total_handshakes = 184 := 
by
  sorry

end total_handshakes_unique_l397_397942


namespace smallest_integer_larger_than_root_diff_power_l397_397115

theorem smallest_integer_larger_than_root_diff_power :
  let z := (Real.sqrt 5 - Real.sqrt 3) ^ 8 in
  Int.ceil z = 9737 :=
by
  sorry

end smallest_integer_larger_than_root_diff_power_l397_397115


namespace largest_five_digit_number_tens_place_l397_397510

theorem largest_five_digit_number_tens_place :
  ∀ (n : ℕ), n = 87315 → (n % 100) / 10 = 1 := 
by
  intros n h
  sorry

end largest_five_digit_number_tens_place_l397_397510


namespace fraction_difference_l397_397876

theorem fraction_difference : 7 / 12 - 3 / 8 = 5 / 24 := 
by 
  sorry

end fraction_difference_l397_397876


namespace maximum_cos_product_l397_397687

theorem maximum_cos_product {α β γ : ℝ} (h : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) :
  cos α * cos β * cos γ ≤ 2 * Real.sqrt 6 / 9 :=
sorry

end maximum_cos_product_l397_397687


namespace euler_totient_inequality_l397_397819

def Euler_totient (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else ((Finset.range (n+1)).filter (λ m, Nat.coprime m n)).card

theorem euler_totient_inequality (n k : ℕ) (h1 : n > 0) (h2 : k > 0)
  (h3 : (∃ m, m = n ∧ (n: ℕ : ℕ) = (Nat.recOn k ((m: ℕ : ℕ)) (λ (k: ℕ) IH, IH >>= Euler_totient) ) ) : 
  n ≤ 3^k :=
sorry

end euler_totient_inequality_l397_397819


namespace train_speed_l397_397141

theorem train_speed
  (length_train1 : ℝ) (speed_train2 : ℝ) (time_cross : ℝ) (length_train2 : ℝ)
  (condition1 : length_train1 = 90) 
  (condition2 : speed_train2 = 80) 
  (condition3 : time_cross = 9) 
  (condition4 : length_train2 = 410.04) :
  let V_rel := (V_1 + speed_train2) * 1000 / 3600
      D := length_train1 + length_train2
  in V_1 = 120.016 :=
by
  sorry

end train_speed_l397_397141


namespace sum_possible_initial_numbers_correct_l397_397590

def smallest_prime_divisor (n : ℕ) : ℕ :=
  if n < 2 then n else List.min' (List.filter (fun d => n % d = 0) (List.range n).tail {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997}.erase n)

-- Given conditions
def initial_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n + k * 2022 = some prime number

-- All possible initial numbers, denoted by s
noncomputable def possible_initial_numbers (s : list ℕ) : Prop :=
  s = [4046, 4047]

-- Goal statement
theorem sum_possible_initial_numbers_correct : 
  (∑ n in possible_initial_numbers, n) = 8093 := by
  sorry

end sum_possible_initial_numbers_correct_l397_397590


namespace sum_of_smallest_and_largest_primes_in_range_1_to_30_l397_397622

theorem sum_of_smallest_and_largest_primes_in_range_1_to_30 : 
  ∃ smallest largest, (smallest ∈ {p ∈ Set.range (2:ℕ) ∣ Nat.Prime p ∧ p ≤ 30})
  ∧ (largest ∈ {p ∈ Set.range (2:ℕ) ∣ Nat.Prime p ∧ p ≤ 30})
  ∧ smallest = Nat.min ({p ∈ Set.range (2:ℕ) ∣ Nat.Prime p ∧ p ≤ 30} : Type ℕ)
  ∧ largest = Nat.max ({p ∈ Set.range (2:ℕ) ∣ Nat.Prime p ∧ p ≤ 30} : Type ℕ)
  ∧ smallest + largest = 31 := sorry

end sum_of_smallest_and_largest_primes_in_range_1_to_30_l397_397622


namespace log_change_of_base_log_change_of_base_with_b_l397_397890

variable {a b x : ℝ}
variable (h₁ : 0 < a ∧ a ≠ 1)
variable (h₂ : 0 < b ∧ b ≠ 1)
variable (h₃ : 0 < x)

theorem log_change_of_base (h₁ : 0 < a ∧ a ≠ 1) (h₂ : 0 < b ∧ b ≠ 1) (h₃ : 0 < x) : 
  Real.log x / Real.log a = Real.log x / Real.log b := by
  sorry

theorem log_change_of_base_with_b (h₁ : 0 < a ∧ a ≠ 1) (h₂ : 0 < b ∧ b ≠ 1) : 
  Real.log b / Real.log a = 1 / Real.log a := by
  sorry

end log_change_of_base_log_change_of_base_with_b_l397_397890


namespace red_chairs_count_l397_397821

-- Given conditions
variables {R Y B : ℕ} -- Assuming the number of chairs are natural numbers

-- Main theorem statement
theorem red_chairs_count : 
  Y = 4 * R ∧ B = Y - 2 ∧ R + Y + B = 43 -> R = 5 :=
by
  sorry

end red_chairs_count_l397_397821


namespace pyramid_volume_correct_l397_397198

-- Define the vertices of the original triangle
def A := (0 : ℝ, 0 : ℝ)
def B := (40 : ℝ, 0 : ℝ)
def C := (20 : ℝ, 30 : ℝ)

-- Define the midpoints of the sides of the triangle
def D := ((40 + 20) / 2 : ℝ, (0 + 30) / 2 : ℝ)
def E := ((20 + 0) / 2 : ℝ, (30 + 0) / 2 : ℝ)
def F := ((40 + 0) / 2 : ℝ, (0 + 0) / 2 : ℝ)

-- Orthocenter of the midpoint triangle
def O := (20 : ℝ, 15 : ℝ)

-- Function to calculate area of a triangle given three points
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.1 * C.2) - (B.1 * A.2))

-- Calculating the area of triangle BFC
def area_BFC := triangle_area B F C

-- Height from orthocenter to BC
def height_O_BC := (O.2 - 0 : ℝ) -- since BC is at y = 0

-- Volume of the pyramid: (1/3) * base area * height
def volume_pyramid := (1 / 3 : ℝ) * area_BFC * height_O_BC

-- The final proof statement
theorem pyramid_volume_correct : volume_pyramid = 3000 := by
  sorry

end pyramid_volume_correct_l397_397198


namespace minimum_value_condition_l397_397535

def f (a x : ℝ) : ℝ := -x^3 + 0.5 * (a + 3) * x^2 - a * x - 1

theorem minimum_value_condition (a : ℝ) (h : a ≥ 3) : 
  (∃ x₀ : ℝ, f a x₀ < f a 1) ∨ (f a 1 > f a ((a/3))) := 
sorry

end minimum_value_condition_l397_397535


namespace primes_with_prime_remainders_count_l397_397309

-- Define a predicate to check for prime numbers
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

-- Define the set of prime remainders when divided by 12
def prime_remainders := {1, 2, 3, 5, 7, 11}

-- Function to list primes between 50 and 100
def primes_between_50_and_100 := [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Function to count such primes with prime remainder when divided by 12
noncomputable def count_primes_with_prime_remainder : ℕ :=
  list.count (λ n, n % 12 ∈ prime_remainders) primes_between_50_and_100

-- The theorem to state the problem in Lean
theorem primes_with_prime_remainders_count : count_primes_with_prime_remainder = 10 :=
by {
  /- proof steps to be provided here, if required. -/
 sorry
}

end primes_with_prime_remainders_count_l397_397309


namespace terminating_decimal_l397_397219

theorem terminating_decimal : (47 : ℚ) / (2 * 5^4) = 376 / 10^4 :=
by sorry

end terminating_decimal_l397_397219


namespace Andy_has_4_more_candies_than_Caleb_l397_397188

-- Define the initial candies each person has
def Billy_initial_candies : ℕ := 6
def Caleb_initial_candies : ℕ := 11
def Andy_initial_candies : ℕ := 9

-- Define the candies bought by the father and their distribution
def father_bought_candies : ℕ := 36
def Billy_received_from_father : ℕ := 8
def Caleb_received_from_father : ℕ := 11

-- Calculate the remaining candies for Andy after distribution
def Andy_received_from_father : ℕ := father_bought_candies - (Billy_received_from_father + Caleb_received_from_father)

-- Calculate the total candies each person has
def Billy_total_candies : ℕ := Billy_initial_candies + Billy_received_from_father
def Caleb_total_candies : ℕ := Caleb_initial_candies + Caleb_received_from_father
def Andy_total_candies : ℕ := Andy_initial_candies + Andy_received_from_father

-- Prove that Andy has 4 more candies than Caleb
theorem Andy_has_4_more_candies_than_Caleb :
  Andy_total_candies = Caleb_total_candies + 4 :=
by {
  -- Skipping the proof
  sorry
}

end Andy_has_4_more_candies_than_Caleb_l397_397188


namespace constant_term_sqrt_x_plus_3_div_x_pow_10_l397_397874

theorem constant_term_sqrt_x_plus_3_div_x_pow_10 :
  (constant_term (sqrt x + 3 / x)^10) = 59049 := by
  sorry

end constant_term_sqrt_x_plus_3_div_x_pow_10_l397_397874


namespace prime_count_between_50_and_100_with_prime_remainder_div_12_l397_397303

open Nat

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_between (a b : ℕ) : list ℕ :=
  (list.range' a (b - a + 1)).filter is_prime

def prime_remainders (p : ℕ) : list ℕ := [2, 3, 5, 7, 11]

theorem prime_count_between_50_and_100_with_prime_remainder_div_12 : 
  (primes_between 50 100).filter (λ p, (p % 12) ∈ prime_remainders p).length = 7 :=
by
  sorry

end prime_count_between_50_and_100_with_prime_remainder_div_12_l397_397303


namespace range_of_a_l397_397698

variable (a : ℝ) (f : ℝ → ℝ)
axiom func_def : ∀ x, f x = a^x
axiom a_pos : a > 0
axiom a_ne_one : a ≠ 1
axiom decreasing : ∀ m n : ℝ, m > n → f m < f n

theorem range_of_a : 0 < a ∧ a < 1 :=
sorry

end range_of_a_l397_397698


namespace rooms_needed_l397_397990

-- Defining all necessary conditions
constant fans : ℕ := 100
constant teams : ℕ := 3
constant max_people_per_room : ℕ := 3
constant groups : ℕ := 6

-- Proposition that the number of rooms required is exactly 37.
theorem rooms_needed (h1 : ∀ i, 1 ≤ i ∧ i ≤ groups) 
  (h2 : groups = 2 * teams) 
  (h3 : teams = 3)
  (h4 : max_people_per_room = 3)
  (h5 : ∃ i, 0 ≤ i ∧ ∑ i in finset.range groups, i = fans):
  37 = (∃ (n : ℕ), ∃ (i : ℕ), n = ∑ i in finset.range groups, nat.ceil (i / max_people_per_room)):
sorry

end rooms_needed_l397_397990


namespace small_circles_sixth_figure_l397_397045

-- Defining the function to calculate the number of circles in the nth figure
def small_circles (n : ℕ) : ℕ :=
  n * (n + 1) + 4

-- Statement of the theorem
theorem small_circles_sixth_figure :
  small_circles 6 = 46 :=
by sorry

end small_circles_sixth_figure_l397_397045


namespace lucky_sum_probability_eq_l397_397374

/--
Given that there are N balls numbered from 1 to N,
where 10 balls are selected in the main draw with their sum being 63,
and 8 balls are selected in the additional draw with their sum being 44,
we need to prove that N = 18 such that the events are equally likely.
-/
theorem lucky_sum_probability_eq (N : ℕ) (h1 : ∃ (S : Finset ℕ), S.card = 10 ∧ S.sum id = 63) 
    (h2 : ∃ (T : Finset ℕ), T.card = 8 ∧ T.sum id = 44) : N = 18 :=
sorry

end lucky_sum_probability_eq_l397_397374


namespace count_number_of_digits_l397_397209

theorem count_number_of_digits : ∀ (n : ℕ), (let v := 2^15 * 5^10 * 3 in
                                             let number_of_digits := Nat.log10 v + 1 in
                                             number_of_digits = 12) :=
by
  sorry

end count_number_of_digits_l397_397209


namespace find_ST_length_l397_397335

noncomputable theory

open_locale classical

variables {P Q R S T U : Type*}
variable [geometry P Q R]

def is_midpoint (S : P) (Q R : P) [metric_space P] : Prop :=
  dist S Q = dist S R ∧ dist Q R = 2 * dist S Q

def bisects_angle (PT : line P) (Q R : P) [metric_space P] : Prop :=
  let angle_PQPT := ∡ P Q PT in
  let angle_PTPR := ∡ PT P R in
  angle_PQPT = angle_PTPR

def is_perpendicular (QT : line P) (PT : line P) [metric_space P] : Prop :=
  QT ⊥ PT

variables (PQ_length PR_length : ℝ)
variable [fact (PQ_length = 10)]
variable [fact (PR_length = 23)]

theorem find_ST_length
  (S_midpoint : is_midpoint S Q R)
  (PT_bisector : bisects_angle PT Q R)
  (QT_perpendicular : is_perpendicular QT PT)
  (PQ_eq : dist P Q = PQ_length)
  (PR_eq : dist P R = PR_length) : 
  dist S T = 5.75 :=
begin
  sorry
end

end find_ST_length_l397_397335


namespace locus_of_points_l397_397267

theorem locus_of_points (A : ℝ × ℝ) (B : ℝ × ℝ) (k : ℝ):
  (A = (0, 0)) → (∃ a : ℝ, B = (a, 0)) →
  ∀ M : ℝ × ℝ, let x := M.1, y := M.2 in ((x^2 + y^2) - ((x - a)^2 + y^2) = k) → x = (k + a^2) / (2 * a) := 
begin
  intros hA hB M,
  let a := classical.some hB,
  let x := M.1,
  let y := M.2,
  assume h,
  sorry,
end

end locus_of_points_l397_397267


namespace perfect_square_subset_exists_l397_397401

theorem perfect_square_subset_exists (n : ℕ) (h : n ≥ 1) (nums : Fin (n + 1) → List ℕ) (primes : Fin n → ℕ)
  (h1 : ∀ i, ∀ j ∈ nums i, j ∈ Set.Range (Fin.val ∘ primes)) :
  ∃ S : Finset (Fin (n + 1)), (S.Nonempty ∧ (∀ s ∈ S, ∀ j ∈ nums s, j ∈ Set.Range (Fin.val ∘ primes)) ∧
  S.Product (λ i, nums i).Prod = (S.Product (λ i, nums i).Prod) ^ 2) := 
by
  sorry

end perfect_square_subset_exists_l397_397401


namespace pow_2023_eq_one_or_neg_one_l397_397017

theorem pow_2023_eq_one_or_neg_one (x : ℂ) (h : (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0) : 
  x^2023 = 1 ∨ x^2023 = -1 := 
by 
{
  sorry
}

end pow_2023_eq_one_or_neg_one_l397_397017


namespace general_formula_sum_of_sequence_l397_397265

variable (a : ℕ → ℤ) (d : ℤ) (n : ℕ)

-- Define the arithmetic sequence conditions
axiom arithmetic_sequence (h : ∀ n, a n = a 1 + (n-1) * d)

-- Given conditions
axiom a1_eq_25 : a 1 = 25
axiom non_zero_difference (h₁ : d ≠ 0)
axiom geometric_condition (h₂ : (a 11) ^ 2 = a 1 * a 13)

-- Find the general formula for {a_n}
theorem general_formula (n : ℕ) : a n = -2 * n + 27 := 
sorry

-- Define the finite sum of the sequence
def sum_sequence (n : ℕ) : ℤ := ∑ k in (finset.range n), a (3 * k + 1)

-- Find the sum of a₁ + a₄ + a₇ + ⋯ + a₃ₙ₋₂
theorem sum_of_sequence (n : ℕ) : sum_sequence a n = -3 * n^2 + 28 * n := 
sorry

end general_formula_sum_of_sequence_l397_397265


namespace g_is_odd_l397_397387

def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end g_is_odd_l397_397387


namespace distinct_products_count_l397_397299

theorem distinct_products_count :
  let s := {1, 2, 3, 5, 7, 11}
  ∃ (products : set ℕ), (∀ a b : ℕ, (a ∈ s) → (b ∈ s) → (a ≠ b) → (a * b ∈ products)) ∧ sizeof products = 26 :=
begin
  sorry,
end

end distinct_products_count_l397_397299


namespace solve_for_x_l397_397040

theorem solve_for_x : ∀ x : ℝ, (x - 5) ^ 3 = (1 / 27)⁻¹ → x = 8 := by
  intro x
  intro h
  sorry

end solve_for_x_l397_397040


namespace sum_of_solutions_f_eq_6_l397_397415

def f (x : ℝ) : ℝ :=
  if x < -3 then 3 * x + 9
  else -x^2 + 2 * x + 8
  
theorem sum_of_solutions_f_eq_6 : 
  (∑ x in {x : ℝ | f x = 6}, x) = 2 := by
  sorry

end sum_of_solutions_f_eq_6_l397_397415


namespace find_median_of_data_set_l397_397674

theorem find_median_of_data_set (x : ℕ) (h_mode : multiset.mode {3, 8, 4, 5, x} = 8) : multiset.median {3, 8, 4, 5, x} = 5 :=
sorry

end find_median_of_data_set_l397_397674


namespace isosceles_triangle_perimeter_l397_397678

theorem isosceles_triangle_perimeter
  {a b : ℝ}
  (h_iso : (a = 3 ∧ b = 7) ∨ (a = 7 ∧ b = 3))
  (h_triangle : (a + a > b ∧ 2 * a > b) ∨ (b + b > a ∧ 2 * b > a)) :
  (2 * a + b = 17) ∨ (2 * b + a = 17) :=
begin
  sorry
end

end isosceles_triangle_perimeter_l397_397678


namespace price_of_smaller_portrait_is_5_l397_397035

-- Definition of the conditions
def price_of_smaller_portrait := 5
def price_of_larger_portrait := 2 * price_of_smaller_portrait
def smaller_portraits_sold_per_day := 3
def larger_portraits_sold_per_day := 5
def total_earnings_every_3_days := 195

-- Proving the price of the smaller portrait
theorem price_of_smaller_portrait_is_5 :
  let daily_earnings := total_earnings_every_3_days / 3 in
  let earnings_from_smaller := smaller_portraits_sold_per_day * price_of_smaller_portrait in
  let earnings_from_larger := larger_portraits_sold_per_day * price_of_larger_portrait in
  let total_daily_earnings := earnings_from_smaller + earnings_from_larger in
  total_daily_earnings = daily_earnings ->
  price_of_smaller_portrait = 5 :=
by
  sorry

end price_of_smaller_portrait_is_5_l397_397035


namespace minimum_rooms_to_accommodate_fans_l397_397978

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l397_397978


namespace existence_of_k_good_function_l397_397793

def is_k_good_function (f : ℕ+ → ℕ+) (k : ℕ) : Prop :=
  ∀ m n : ℕ+, m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

theorem existence_of_k_good_function (k : ℕ) :
  (∃ f : ℕ+ → ℕ+, is_k_good_function f k) ↔ k ≥ 2 := sorry

end existence_of_k_good_function_l397_397793


namespace number_of_solutions_l397_397009

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem number_of_solutions : 
  ∃ (c : set ℝ), (∀ x ∈ c, g(g(g(g(x)))) = 2) ∧ c.card = 16 :=
by
  sorry

end number_of_solutions_l397_397009


namespace third_shiny_penny_on_sixth_draw_l397_397143

def total_pennies : ℕ := 9
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := total_pennies - shiny_pennies

def event_E (draws : list bool) : Prop :=
  length draws = total_pennies ∧
  (count (λ x, x) (take 6 draws) = 3) ∧
  nth draws 5 = some true ∧
  (count (λ x, x) (take 5 draws) = 2)

def prob_event_E : ℚ :=
  (choose 5 2 * choose 3 1 : ℕ) / (choose 9 4 : ℕ) 

theorem third_shiny_penny_on_sixth_draw :
  ∃ draws, event_E draws ∧ prob_event_E = 5 / 21 :=
sorry

end third_shiny_penny_on_sixth_draw_l397_397143


namespace semicircle_radius_approx_l397_397166

/-- The perimeter of a semicircle is given by the formula P = πr + 2r. 
    Given the perimeter of the semicircle as 102.83185307179586, 
    the radius of the semicircle is approximately 20 units. -/
theorem semicircle_radius_approx (P : ℝ) (hP : P = 102.83185307179586) :
  ∃ r : ℝ, (real.pi * r + 2 * r = P) ∧ r ≈ 20 :=
by
  use 102.83185307179586 / (real.pi + 2)
  split
  { sorry }
  { linarith }

end semicircle_radius_approx_l397_397166


namespace number_doubled_is_12_l397_397895

theorem number_doubled_is_12 (A B C D E : ℝ) (h1 : (A + B + C + D + E) / 5 = 6.8)
  (X : ℝ) (h2 : ((A + B + C + D + E - X) + 2 * X) / 5 = 9.2) : X = 12 :=
by
  sorry

end number_doubled_is_12_l397_397895


namespace total_spent_on_toys_l397_397931

def football_price : ℝ := 12.99
def teddy_bear_price : ℝ := 15.35
def crayon_pack_price : ℝ := 4.65
def puzzle_price : ℝ := 7.85
def doll_price : ℝ := 14.50
def teddy_bear_discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08

theorem total_spent_on_toys :
  let discounted_teddy_bear_price := (teddy_bear_price * (1 - teddy_bear_discount_rate)).round(2)
  let total_cost_before_tax := football_price + discounted_teddy_bear_price + crayon_pack_price + puzzle_price + doll_price
  let sales_tax := (total_cost_before_tax * sales_tax_rate).round(2)
  let total_cost_including_tax := total_cost_before_tax + sales_tax
  total_cost_including_tax = 57.23 := by
{
  sorry
}

end total_spent_on_toys_l397_397931


namespace range_of_k_constant_value_AM_AN_equation_of_line_l397_397671

-- Range of k
theorem range_of_k (A : Point) (C : Circle)
  (hA : A = ⟨0, 1⟩) (hC : C = { center := ⟨2, 3⟩, radius := 1 })
  (k : ℝ) : 
  (∃ M N : Point, M ≠ N ∧ Line_through A k M ∧ Line_through A k N ∧ On_circle M C ∧ On_circle N C) →
  \(\frac{4 - \sqrt{7}}{3}\) < k ∧ k < \(\frac{4 + \sqrt{7}}{3}\) := sorry

-- Constant value of AM ⋅ AN
theorem constant_value_AM_AN (A M N : Point) (C : Circle)
  (hA : A = ⟨0, 1⟩) (hC : C = { center := ⟨2, 3⟩, radius := 1 })
  (k : ℝ) (hM : Line_through A k M) (hN : Line_through A k N) (hM_on_C : On_circle M C) (hN_on_C : On_circle N C) :
  (\(\overrightarrow{AM} \cdot \overrightarrow{AN} = 7\)) := sorry

-- Equation of the line l
theorem equation_of_line (A M N : Point) (O : Point) (C : Circle) 
  (hA : A = ⟨0, 1⟩) (hC : C = { center := ⟨2, 3⟩, radius := 1 }) 
  (hO : O = ⟨0, 0⟩) 
  (k : ℝ) (hOMN : \(\overrightarrow{OM} \cdot \overrightarrow{ON} = 12\)) :
  \(\text{equation of line } l : x - y + 1 = 0\) := sorry

end range_of_k_constant_value_AM_AN_equation_of_line_l397_397671


namespace decagon_diagonal_intersection_probability_l397_397616

theorem decagon_diagonal_intersection_probability :
  let n := 10
  let diagonals := Finset.range n.choose 2 - n
  let pairs_diagonals := diagonals.choose 2
  let intersecting_diagonals := n.choose 4
  (intersecting_diagonals : ℚ) / pairs_diagonals = 42 / 119 :=
by
  let n := 10
  let diagonals := Nat.choose n 2 - n
  let pairs_diagonals := Nat.choose diagonals 2
  let intersecting_diagonals := Nat.choose n 4
  have h1 : intersecting_diagonals = 210 := sorry
  have h2 : pairs_diagonals = 595 := sorry
  have h3 : nat_gcd 210 595 = 5 := sorry
  have h4 : 210 / 595 = 42 / 119 := by 
    rw [div_eq_div_iff]
    exact_mod_cast (mul_div_cancel_left 210 5)
  nth_rewrite 0 ←h4
  rfl

end decagon_diagonal_intersection_probability_l397_397616


namespace tangent_lines_intersect_at_diameter_l397_397858

-- Define the centers of the three circles
variables {O1 O2 O3 : Point}

-- Define the points of tangency between the circles
variables {A : Point} -- Tangency point of O1 and O2
variables {B : Point} -- Tangency point of O2 and O3
variables {C : Point} -- Tangency point of O1 and O3

-- Define the three circles
variables {circle1 : Circle} (Hcircle1 : circle1.center = O1) (Htouch1 : circle1.tangentAt A)
variables {circle2 : Circle} (Hcircle2 : circle2.center = O2) (Htouch2 : circle2.tangentAt A ∧ circle2.tangentAt B)
variables {circle3 : Circle} (Hcircle3 : circle3.center = O3) (Htouch3 : circle3.tangentAt B ∧ circle3.tangentAt C)

-- Define the theorem to prove
theorem tangent_lines_intersect_at_diameter :
  (∃ M K : Point, line_through A B ∩ circle3 = {M, K} ∧ M ≠ K
  ∧ dist M O3 = dist K O3 ∧ line_through M K = line_through O3 A ∨ line_through O3 B ∨ line_through O3 C) := 
sorry

end tangent_lines_intersect_at_diameter_l397_397858


namespace base_problem_l397_397481

theorem base_problem (c d : Nat) (pos_c : c > 0) (pos_d : d > 0) (h : 5 * c + 8 = 8 * d + 5) : c + d = 15 :=
sorry

end base_problem_l397_397481


namespace arrangement_count_seq_no_three_consec_inc_dec_l397_397485

theorem arrangement_count_seq_no_three_consec_inc_dec :
  ∃ (l : list ℕ), l ~ [1, 2, 3, 4, 5] ∧ 
    (∀ (a b c : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ a < b ∧ b < c → false) ∧
    (∀ (a b c : ℕ), a ∈ l ∧ b ∈ l ∧ c ∈ l ∧ a > b ∧ b > c → false)
    ∧ l.length = 5 :=
begin
  sorry
end

end arrangement_count_seq_no_three_consec_inc_dec_l397_397485


namespace cosine_of_angle_magnitude_of_sum_l397_397720

noncomputable def vector_a : Type := EuclideanSpace ℝ (Fin 3)
noncomputable def vector_b : Type := EuclideanSpace ℝ (Fin 3)

variables (a b : ℝ)

-- Conditions
axiom norm_a : ∥ a ∥ = 5
axiom norm_b : ∥ b ∥ = 3
axiom dot_product_cond : (a - b) • (2 • a + 3 • b) = 13

-- Statements to prove
theorem cosine_of_angle :
  let θ := real.angle_between_vectors a b in
  real.cos θ = -2/3 :=
sorry

theorem magnitude_of_sum :
  ∥ a + 2 • b ∥ = real.sqrt 21 :=
sorry

end cosine_of_angle_magnitude_of_sum_l397_397720


namespace kombucha_bottles_after_refund_l397_397724

noncomputable def bottles_per_month : ℕ := 15
noncomputable def cost_per_bottle : ℝ := 3.0
noncomputable def refund_per_bottle : ℝ := 0.10
noncomputable def months_in_year : ℕ := 12

theorem kombucha_bottles_after_refund :
  let bottles_per_year := bottles_per_month * months_in_year
  let total_refund := bottles_per_year * refund_per_bottle
  let bottles_bought_with_refund := total_refund / cost_per_bottle
  bottles_bought_with_refund = 6 := sorry

end kombucha_bottles_after_refund_l397_397724


namespace positive_integers_satisfy_inequality_l397_397301

theorem positive_integers_satisfy_inequality :
  ∃ (count : ℕ), count = 15049 ∧
  ∀ n : ℕ, (n > 0) →
  (\left\lceil (n : ℝ) / 101 \right\rceil + 1 > (n : ℝ) / 100) ↔
  (n ≤ nth_count count) :=
begin
  sorry
end

end positive_integers_satisfy_inequality_l397_397301


namespace count_number_of_digits_l397_397210

theorem count_number_of_digits : ∀ (n : ℕ), (let v := 2^15 * 5^10 * 3 in
                                             let number_of_digits := Nat.log10 v + 1 in
                                             number_of_digits = 12) :=
by
  sorry

end count_number_of_digits_l397_397210


namespace find_v6_l397_397956

def v_n : ℕ → ℝ
| 4 := 6
| 7 := 87
| n := if h1 : n ≥ 4 then (v_n (n-1) * 3 - v_n (n-2) * 2) else 0 -- This handles the pre-defined values for n=4 and n=7.

theorem find_v6 : v_n 6 = 40.713 := by
  sorry

end find_v6_l397_397956


namespace ellipse_properties_l397_397676

theorem ellipse_properties :
  ∃ (a b : ℝ), 
    0 < b ∧ b < a ∧
    (∀ (e : ℝ), e = (Real.sqrt 3) / 2 → c^2 = a^2 - b^2) ∧
    (∀ (x y : ℝ), x = 2 ∧ y = 1 → (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    (let slope_OM := (1 / 2) in 
     ∀ (m : ℝ),
       (-2 < m ∧ m < 2) ∧
       ellipse_eq : (x^2 / 8 + y^2 / 2 = 1)) :=
sorry

end ellipse_properties_l397_397676


namespace classification_of_solutions_l397_397274

noncomputable def existence_real_solutions
  (a b c : ℝ) (h : a ≠ 0) (n : ℕ) (x : fin (n + 1) → ℝ)
  (h1 : ∀ i : fin n, a * x i ^ 2 + b * x i + c = x ⟨i + 1, nat.succ_lt_succ i.2⟩)
  (h2 : a * x n ^ 2 + b * x n + c = x 0) 
  : Prop :=
  if discr : (b - 1) ^ 2 - 4 * a * c < 0 then false
  else if discr = 0 then true
  else true

theorem classification_of_solutions
  (a b c : ℝ) (h : a ≠ 0) (n : ℕ)
  (h1 : ∀ x : fin (n + 1) → ℝ, (∀ i : fin n, a * x i ^ 2 + b * x i + c = x ⟨i + 1, nat.succ_lt_succ i.2⟩) 
                               → (a * x n ^ 2 + b * x n + c = x 0) 
                               → existence_real_solutions a b c h n x)
  : 
  (if (b - 1) ^ 2 - 4 * a * c < 0 then ∀ x, ∃! y, ¬(a*x^2 + (b-1)*x + c = y)
   else if (b - 1) ^ 2 - 4 * a * c = 0 then ∃! y, a*y^2 + (b-1)*y + c = y
   else ∃ y y', y ≠ y' ∧ a*y^2 + (b-1)*y + c = y ∧ a*y'^2 + (b-1)*y' + c = y') := sorry

end classification_of_solutions_l397_397274


namespace find_x_plus_y_l397_397683

theorem find_x_plus_y (x y : ℝ) 
  (h1 : x + Real.cos y = 1005) 
  (h2 : x + 1005 * Real.sin y = 1003) 
  (h3 : π ≤ y ∧ y ≤ 3 * π / 2) : 
  x + y = 1005 + 3 * π / 2 :=
sorry

end find_x_plus_y_l397_397683


namespace lattice_points_integer_area_l397_397569

/-- 
A lattice point is defined as a point (x, y) where both x and y are integers.
This theorem states that any set of 5 lattice points contains three points that
form the vertices of a triangle with an integer area.
-/
theorem lattice_points_integer_area (n : ℕ) (h : n = 5) : 
  ∀ (points : Fin n → ℤ × ℤ), 
    ∃ (a b c : Fin n), 
      let (x1, y1) := points a
      let (x2, y2) := points b
      let (x3, y3) := points c
      let area := (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2
      area ∈ ℤ :=
begin
  sorry
end

end lattice_points_integer_area_l397_397569


namespace selling_price_l397_397597

/-
An article is bought for Rs. 560 and sold for some amount. 
The loss percent is 39.285714285714285%. 
What was the selling price of the article?

Conditions:
- Cost Price (CP) = Rs. 560
- Loss Percent = 39.285714285714285%
- Selling Price (SP) = Cost Price - (Cost Price * Loss Percent/100)
-/

theorem selling_price (CP : ℝ) (loss_percent : ℝ) (SP : ℝ) :
  CP = 560 → loss_percent = 39.285714285714285 → SP = CP - (CP * (loss_percent / 100)) → SP = 340 :=
by
  intros hCP hLoss hSP
  rw [hCP, hLoss]
  norm_num at hSP
  exact hSP

end selling_price_l397_397597


namespace sum_first_2004_terms_l397_397076

noncomputable def x : ℕ → ℤ
| 1       := 1001
| 2       := 1003
| (n + 2) := (x (n + 1) - 2004) / x n

theorem sum_first_2004_terms :
  (Finset.range 2004).sum (λ n => x (n + 1)) = 1338004 :=
  sorry

end sum_first_2004_terms_l397_397076


namespace pyramid_volume_l397_397433

-- Define the lengths AB, BC, and PB
def AB : ℝ := 10
def BC : ℝ := 5
def PB : ℝ := 20

-- Define the area of the base rectangle ABCD
def base_area : ℝ := AB * BC

-- Using Pythagorean theorem to get the height PA
def PA : ℝ := Real.sqrt (PB ^ 2 - AB ^ 2) -- should be 10 * Real.sqrt 3

-- Define the volume of the pyramid PABCD
def volume : ℝ := (1 / 3) * base_area * PA

-- The theorem to prove the volume is what we calculated
theorem pyramid_volume : volume = (500 * Real.sqrt 3) / 3 := 
by 
  sorry

end pyramid_volume_l397_397433


namespace seventh_number_pattern_l397_397696

theorem seventh_number_pattern :
  let seq := λ n : ℕ, if n % 2 = 1 then - (n / (n^2 + 1) : ℝ) else (n / (n^2 + 1) : ℝ) in
  seq 7 = - (7 / 50 : ℝ) :=
by
  sorry

end seventh_number_pattern_l397_397696


namespace similar_triangles_l397_397239

noncomputable theory

open_locale classical

variables {A X Y Z : Type*}
variables [metric_space A] [metric_space X] [metric_space Y] [metric_space Z]
variables {circle : set A}

-- Definitions and conditions
def is_tangent (A X : A) (circle : set A) : Prop := 
sorry -- Assume we have the definition of a tangent here

def is_secant (A Y Z : A) (circle : set A) : Prop := 
sorry -- Likewise, assume the definition of a secant here

def is_on_circle (X Y Z : A) (circle : set A) : Prop := 
sorry -- Assume we have a predicate stating that X, Y, Z are on the circle

-- Condition: A is outside the circle
axiom A_outside_circle : A ∉ circle
-- Condition: AX is tangent to the circle at X
axiom AX_tangent : is_tangent A X circle
-- Condition: Secant through A intersects the circle at Y and Z
axiom AYZ_secant : is_secant A Y Z circle
-- Condition: X, Y, Z are on the circle
axiom XYZ_on_circle : is_on_circle X Y Z circle

-- To be proved: Triangles AXY and AXZ are similar
theorem similar_triangles: 
  ∀ {A X Y Z : A}, is_tangent A X circle → 
                   is_secant A Y Z circle → 
                   is_on_circle X Y Z circle → 
                   (∃ α β γ, triangle A X Y α β γ A X Y β γ α) →
                   (∃ δ ε ζ, triangle A X Z δ ε ζ A X Z ε ζ δ) →
                   ∃ θ ι κ, triangle_similar A X Y θ ι κ A X Z θ ι κ :=
sorry

end similar_triangles_l397_397239


namespace max_cos_product_l397_397684

open Real

theorem max_cos_product (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
                                       (hβ : 0 < β ∧ β < π / 2)
                                       (hγ : 0 < γ ∧ γ < π / 2)
                                       (h : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) : 
  cos α * cos β * cos γ ≤ 2 * Real.sqrt 6 / 9 := 
by sorry

end max_cos_product_l397_397684


namespace lowest_score_of_14_scores_l397_397453

theorem lowest_score_of_14_scores (mean_14 : ℝ) (new_mean_12 : ℝ) (highest_score : ℝ) (lowest_score : ℝ) :
  mean_14 = 85 ∧ new_mean_12 = 88 ∧ highest_score = 105 → lowest_score = 29 :=
by
  sorry

end lowest_score_of_14_scores_l397_397453


namespace sum_of_numbers_mod_11_l397_397654

def sum_mod_eleven : ℕ := 123456 + 123457 + 123458 + 123459 + 123460 + 123461

theorem sum_of_numbers_mod_11 :
  sum_mod_eleven % 11 = 10 :=
sorry

end sum_of_numbers_mod_11_l397_397654


namespace distance_between_midpoints_correct_l397_397745

variables (a b c d : ℝ)

/-- Initial midpoint M between points A(a, b) and B(c, d) --/
def midpoint (a b c d : ℝ) : ℝ × ℝ :=
  ( (a + c) / 2, (b + d) / 2 )

def moved_point_A (a b : ℝ) : ℝ × ℝ :=
  (a - 5, b - 3)

def moved_point_B (c d : ℝ) : ℝ × ℝ :=
  (c + 4, d + 6)

/-- New midpoint M' after moving points A and B --/
def new_midpoint (a b c d : ℝ) : ℝ × ℝ :=
  ( (a - 5 + c + 4) / 2, (b - 3 + d + 6) / 2 )

def distance_between_midpoints (a b c d : ℝ) : ℝ :=
  sqrt ( ((a + c) / 2 - (a + c - 1) / 2)^2 + ((b + d) / 2 - (b + d + 3) / 2)^2 )

theorem distance_between_midpoints_correct (a b c d : ℝ) :
  distance_between_midpoints a b c d = sqrt(10) / 2 :=
by
  sorry

end distance_between_midpoints_correct_l397_397745


namespace arrangement_male_A_first_arrangement_male_A_B_middle_arrangement_male_A_not_first_B_not_last_arrangement_male_A_B_together_arrangement_females_together_arrangement_no_two_females_adjacent_arrangement_fixed_order_ABC_l397_397085

def arrangements_count (condition : Prop) : Nat := sorry

theorem arrangement_male_A_first :
  arrangements_count (Male_A_first : Male A is in the first position) = Nat.perm 9 :=
sorry

theorem arrangement_male_A_B_middle :
  arrangements_count (Male_A_B_middle : Male A and Male B are in the middle) = Nat.perm 2 * Nat.perm 7 :=
sorry

theorem arrangement_male_A_not_first_B_not_last :
  arrangements_count (Male_A_not_first_B_not_last : Male A is not in the first position ∧ Male B is not in the last position) =
  Nat.perm 10 - 2 * Nat.perm 9 + Nat.perm 8 :=
sorry

theorem arrangement_male_A_B_together :
  arrangements_count (Male_A_B_together : Male A and Male B are together) = Nat.perm 2 * Nat.perm 8 :=
sorry

theorem arrangement_females_together :
  arrangements_count (Females_together : The 4 females are standing together) = Nat.perm 4 * Nat.perm 7 :=
sorry

theorem arrangement_no_two_females_adjacent :
  arrangements_count (No_two_females_adjacent : No two females can be adjacent) = Nat.perm 6 * Nat.choose 7 4 :=
sorry

theorem arrangement_fixed_order_ABC :
  arrangements_count (Fixed_order_ABC : The order of Males A, B, and C is fixed) = Nat.perm 10 / Nat.perm 3 :=
sorry

end arrangement_male_A_first_arrangement_male_A_B_middle_arrangement_male_A_not_first_B_not_last_arrangement_male_A_B_together_arrangement_females_together_arrangement_no_two_females_adjacent_arrangement_fixed_order_ABC_l397_397085


namespace correct_division_answer_l397_397341

theorem correct_division_answer (q_wrong: ℕ) (div_wrong div_correct result_correct: ℕ) (h1: div_wrong = 63) (h2: q_wrong = 24) (h3: div_correct = 36) : 
  (div_wrong * q_wrong) / div_correct = result_correct -> result_correct = 42 :=
by
  intros h
  have h_calc : (63 * 24) / 36 = result_correct,
    rw [← h1, ← h2, ← h3]
  calc 
    (63 * 24) / 36 = 1512 / 36 : by norm_num
    ... = 42 : by norm_num
  exact h

end correct_division_answer_l397_397341


namespace sum_b_k_equals_6385_l397_397486

noncomputable def a : ℕ → ℝ
| 1     := 1
| (n+2) := some (λ (x : ℝ), x ^ 2 + 3 * (n + 1) * x + b (n + 1) = 0)

noncomputable def b (n : ℕ) : ℝ := 
(a n) * (a (n + 1))

theorem sum_b_k_equals_6385 : (∑ k in range 20, b (k + 1)) = 6385 := sorry

end sum_b_k_equals_6385_l397_397486


namespace polar_eq_line_l_cartesian_eq_curve_C_angle_POQ_l397_397763

-- Definitions and conditions
def parametric_eq_line_l (t : ℝ) : ℝ × ℝ :=
  (1 - real.sqrt 3 * t, 1 + t)

def polar_eq_curve_C (θ : ℝ) : ℝ :=
  2 * real.cos θ

-- Theorem statements
theorem polar_eq_line_l (ρ θ : ℝ) (h1 : ρ (real.cos θ + real.sqrt 3 * real.sin θ) = 1 + real.sqrt 3) :
  parametric_eq_line_l t = polar_eq_curve_C θ := sorry

theorem cartesian_eq_curve_C (x y : ℝ) (h2 : x ^ 2 + y ^ 2 = 2 * x) :
  x ^ 2 + y ^ 2 - 2 * x = 0 := sorry

-- Given the correct answer, we prove the intersection angle in polar coordinates
theorem angle_POQ (θ1 θ2 : ℝ) (h3: abs (θ1 - θ2) = real.pi / 6) :
  ∃ P Q, 
  polar_eq_curve_C θ1 = polar_eq_curve_C θ2 ∧
  polar_eq_curve_C θ1 = polar_eq_curve_C θ2 ∧
  abs (real.arctan (ρ * real.sin θ1 / ρ * real.cos θ1) - real.arctan (ρ * real.sin θ2 / ρ * real.cos θ2)) = real.pi / 6 :=
sorry

end polar_eq_line_l_cartesian_eq_curve_C_angle_POQ_l397_397763


namespace find_f1_and_f_prime1_l397_397472

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Conditions
axiom differentiable_f : Differentiable ℝ f
axiom f_def : ∀ x : ℝ, f x = 2 * x^2 - f' 1 * x - 3

-- Proof using conditions
theorem find_f1_and_f_prime1 : f 1 + (f' 1) = -1 :=
sorry

end find_f1_and_f_prime1_l397_397472


namespace solve_problem_l397_397795

def f (x : ℝ) : ℝ := if x ≥ 0 then Real.logb 2 (x + 1) else g x
def g (x : ℝ) : ℝ := - Real.logb 2 (-x + 1)

theorem solve_problem : g (f (-7)) = -2 :=
by 
  sorry

end solve_problem_l397_397795


namespace diagonal_length_of_rectangle_l397_397111

-- Define the conditions
def a : ℝ := 30 * Real.sqrt 3
def b : ℝ := 30

-- State the theorem
theorem diagonal_length_of_rectangle : a^2 + b^2 = (60 : ℝ)^2 :=
by 
  sorry

end diagonal_length_of_rectangle_l397_397111


namespace modulusOne_l397_397466

-- Defining the condition
def comCond (z : ℂ) : Prop := (1 + complex.i) * z = 1 - complex.i

-- Defining the theorem to prove
theorem modulusOne (z : ℂ) (h : comCond z) : |z| = 1 :=
by
  sorry

end modulusOne_l397_397466


namespace gcd_of_gx_x_is_one_l397_397272

noncomputable def gcd_gx_x (x : ℤ) : ℤ :=
  let g := (3 * x + 4) * (8 * x + 5) * (15 * x + 9) * (x + 15)
  gcd g x

theorem gcd_of_gx_x_is_one (x : ℤ) (h : ∃ k : ℤ, x = 37521 * k) : gcd_gx_x x = 1 :=
by
  sorry

end gcd_of_gx_x_is_one_l397_397272


namespace correct_statement_about_propositions_l397_397882

theorem correct_statement_about_propositions :
  let A := ¬ (∀ x : ℝ, x^2 = 1 → x = 1)
  let B := ¬ (x = -1 → x^2 - 5*x - 6 = 0 ∧ ¬(x^2 - 5*x - 6 = 0 → x = -1))
  let C := ¬ (¬ (∀ x : ℝ, x^2 + x + 1 ≥ 0))
  let D := (∀ x y : ℝ, (x = y → sin x = sin y) ↔ (sin x ≠ sin y → x ≠ y))
  A = false ∧ B = false ∧ C = false → D = true :=
by
  sorry

end correct_statement_about_propositions_l397_397882


namespace bombay_express_speed_l397_397183

-- Let's define the conditions
def train_conditions (V_BE : ℝ) : Prop :=
  ∃ (V_RE speed met_distance : ℝ),
    V_RE = 80 ∧
    speed = 80 ∧
    met_distance = 480 ∧
    met_distance / V_BE = met_distance / V_RE + 2

-- Statement to prove: the speed of Bombay Express is 60 km/h
theorem bombay_express_speed : train_conditions 60 :=
by
  unfold train_conditions
  use [80, 80, 480]
  simp
  sorry

end bombay_express_speed_l397_397183


namespace divide_diagonal_into_three_equal_parts_l397_397062

variable {Point : Type} [AffineSpace ℝ Point]

def midpoint (A B : Point) : Point := sorry  -- Definition of midpoint

variable (A B C D E F N M : Point)

variables [h_parallelogram : parallelogram ABCD]
  (h_E_midpoint : E = midpoint B C)
  (h_F_midpoint : F = midpoint A D)

theorem divide_diagonal_into_three_equal_parts
  (hN_intersects : ∃ N, lineThrough B F ∩ lineThrough A C = {N})
  (hM_intersects : ∃ M, lineThrough E D ∩ lineThrough A C = {M}) :
  dist A N = dist N M ∧ dist N M = dist M C :=
sorry

end divide_diagonal_into_three_equal_parts_l397_397062


namespace susan_average_speed_l397_397820

noncomputable def total_distance (segment1 segment2 : ℕ) : ℕ := segment1 + segment2

noncomputable def time_taken (distance : ℕ) (speed : ℕ) : ℚ := distance / speed

noncomputable def total_time (time1 time2 : ℚ) : ℚ := time1 + time2

noncomputable def average_speed (total_distance : ℕ) (total_time : ℚ) : ℚ := total_distance / total_time

theorem susan_average_speed :
  let distance1 := 40
      speed1 := 30
      distance2 := 40
      speed2 := 15
      total_distance := total_distance distance1 distance2
      time1 := time_taken distance1 speed1
      time2 := time_taken distance2 speed2
      total_time := total_time time1 time2
      average_speed := average_speed total_distance total_time
  in
  average_speed = 20 := by
  -- Variables definitions
  let distance1 := 40
  let speed1 := 30
  let distance2 := 40
  let speed2 := 15

  -- Calculations
  let total_distance := total_distance distance1 distance2
  let time1 := time_taken distance1 speed1
  let time2 := time_taken distance2 speed2
  let total_time := total_time time1 time2
  let average_speed := average_speed total_distance total_time

  -- Result expectation
  show average_speed = 20 from sorry

end susan_average_speed_l397_397820


namespace jake_weight_l397_397318

theorem jake_weight (J S B : ℝ) (h1 : J - 8 = 2 * S)
                            (h2 : B = 2 * J + 6)
                            (h3 : J + S + B = 480)
                            (h4 : B = 1.25 * S) :
  J = 230 :=
by
  sorry

end jake_weight_l397_397318


namespace inscribed_sphere_radius_l397_397263

theorem inscribed_sphere_radius 
  (V : ℝ) (S1 S2 S3 S4 : ℝ) (R : ℝ) :
  (1 / 3) * R * (S1 + S2 + S3 + S4) = V ↔ R = (3 * V) / (S1 + S2 + S3 + S4) := 
by
  sorry

end inscribed_sphere_radius_l397_397263


namespace box_depth_is_18_l397_397553

open Nat

-- Using the given conditions
axiom box_length : ℕ := 36
axiom box_width : ℕ := 45
axiom num_cubes : ℕ := 40
axiom volume_cube : ℕ := (gcd box_length box_width) ^ 3

-- Volume of the box
def volume_box : ℕ := num_cubes * volume_cube

-- Assuming the depth of the box to be some natural number
variable depth : ℕ

-- Asserting the volume of the box is correctly filled using its length, width, and depth
axiom box_volume_eq : volume_box = box_length * box_width * depth

-- The proposition stating the depth of the box
theorem box_depth_is_18 : depth = 18 :=
by
  sorry

end box_depth_is_18_l397_397553


namespace triangle_angle_sum_l397_397385

open scoped Real

theorem triangle_angle_sum (A B C : ℝ) 
  (hA : A = 25) (hB : B = 55) : C = 100 :=
by
  have h1 : A + B + C = 180 := sorry
  rw [hA, hB] at h1
  linarith

end triangle_angle_sum_l397_397385


namespace triangle_inequality_proof_l397_397350

noncomputable def triangle_inequality (a b c h_a r : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (h_a > 0) ∧ (r > 0) ∧
  ((∑ cyc (λ x y z : ℝ, sqrt (x * (y - 2 * r) / ((3 * x + y + z) * (y + 2 * r))))
       a b c h_a) ≤ 3 / 4)

theorem triangle_inequality_proof (a b c h_a r : ℝ) :
  triangle_inequality a b c h_a r :=
sorry

end triangle_inequality_proof_l397_397350


namespace complex_fraction_identity_l397_397785

theorem complex_fraction_identity
  (a b : ℂ) (ζ : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : ζ ^ 3 = 1) (h4 : ζ ≠ 1) 
  (h5 : a ^ 2 + a * b + b ^ 2 = 0) :
  (a ^ 9 + b ^ 9) / ((a - b) ^ 9) = (2 : ℂ) / (81 * (ζ - 1)) :=
sorry

end complex_fraction_identity_l397_397785


namespace pay_per_task_l397_397091

def tasks_per_day : ℕ := 100
def days_per_week : ℕ := 6
def weekly_pay : ℕ := 720

theorem pay_per_task :
  (weekly_pay : ℚ) / (tasks_per_day * days_per_week) = 1.20 := 
sorry

end pay_per_task_l397_397091


namespace quadrilateral_area_l397_397019

-- Define the dimensions of the rectangles
variables (AB BC EF FG : ℝ)
variables (AFCH_area : ℝ)

-- State the conditions explicitly
def conditions : Prop :=
  (AB = 9) ∧ 
  (BC = 5) ∧ 
  (EF = 3) ∧ 
  (FG = 10)

-- State the theorem to prove
theorem quadrilateral_area (h: conditions AB BC EF FG) : 
  AFCH_area = 52.5 := 
sorry

end quadrilateral_area_l397_397019


namespace meeting_point_midpoint_l397_397799

theorem meeting_point_midpoint :
  let lucas := (3, 12)
  let emma := (7, 4)
  let midpoint := ((fst lucas + fst emma) / 2, (snd lucas + snd emma) / 2)
  midpoint = (5, 8) :=
by
  let lucas := (3, 12)
  let emma := (7, 4)
  let midpoint := ((fst lucas + fst emma) / 2, (snd lucas + snd emma) / 2)
  have h1 : (fst lucas + fst emma) = 3 + 7 := rfl
  have h2 : (snd lucas + snd emma) = 12 + 4 := rfl
  have hx : (fst lucas + fst emma) / 2 = 10 / 2 := by rw h1
  have hy : (snd lucas + snd emma) / 2 = 16 / 2 := by rw h2
  have hx5 : 10 / 2 = 5 := rfl
  have hy8 : 16 / 2 = 8 := rfl
  exact (hx.1.trans hx5).congr (hy.1.trans hy8)
  sorry

end meeting_point_midpoint_l397_397799


namespace cell_value_l397_397448

variable (P Q R S : ℕ)

-- Condition definitions
def topLeftCell (P : ℕ) : ℕ := P
def topMiddleCell (P Q : ℕ) : ℕ := P + Q
def centerCell (P Q R S : ℕ) : ℕ := P + Q + R + S
def bottomLeftCell (S : ℕ) : ℕ := S

-- Given Conditions
axiom bottomLeftCell_value : bottomLeftCell S = 13
axiom topMiddleCell_value : topMiddleCell P Q = 18
axiom centerCell_value : centerCell P Q R S = 47

-- To prove: R = 16
theorem cell_value : R = 16 :=
by
  sorry

end cell_value_l397_397448


namespace cars_parked_l397_397070

def front_parking_spaces : ℕ := 52
def back_parking_spaces : ℕ := 38
def filled_back_spaces : ℕ := back_parking_spaces / 2
def available_spaces : ℕ := 32
def total_parking_spaces : ℕ := front_parking_spaces + back_parking_spaces
def filled_spaces : ℕ := total_parking_spaces - available_spaces

theorem cars_parked : 
  filled_spaces = 58 := by
  sorry

end cars_parked_l397_397070


namespace find_common_ratio_l397_397670

variable {a : ℕ → ℝ}
variable (q : ℝ)

-- Definition of geometric sequence condition
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given conditions
def conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 2 + a 4 = 20) ∧ (a 3 + a 5 = 40)

-- Proposition to be proved
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geo : is_geometric_sequence a q) (h_cond : conditions a q) : q = 2 :=
by 
  sorry

end find_common_ratio_l397_397670


namespace find_xy_l397_397640

theorem find_xy (x y : ℝ) :
  x^2 + y^2 = 2 ∧ (x^2 / (2 - y) + y^2 / (2 - x) = 2) → (x = 1 ∧ y = 1) :=
by
  sorry

end find_xy_l397_397640


namespace greatest_of_three_consecutive_integers_with_sum_21_l397_397877

theorem greatest_of_three_consecutive_integers_with_sum_21 :
  ∃ (x : ℤ), (x + (x + 1) + (x + 2) = 21) ∧ ((x + 2) = 8) :=
by
  sorry

end greatest_of_three_consecutive_integers_with_sum_21_l397_397877


namespace max_yes_answers_l397_397138

-- Define the problem:
def is_knight (p : Person) : Prop := sorry
def is_liar (p : Person) : Prop := sorry
def rows : list (list Person) := sorry
def more_than_half_liars (row : list Person) : Prop := sorry

-- State the main theorem:
theorem max_yes_answers (people : list Person) (h_arranged : length people = 30) (h_rows : length rows = 6) (h_each_row : ∀ r ∈ rows, length r = 5) (h_knights_or_liars : ∀ p ∈ people, is_knight p ∨ is_liar p) : 
  (∑ r in rows, ite (more_than_half_liars r) (count_lies r) (count_truths r)) ≤ 21 :=
sorry

end max_yes_answers_l397_397138


namespace min_rooms_needed_l397_397997

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l397_397997


namespace cuboid_volume_l397_397826

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 3) (h2 : b * c = 5) (h3 : a * c = 15) : a * b * c = 15 :=
sorry

end cuboid_volume_l397_397826


namespace solution_l397_397648

def factorial : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 2
| 3 := 6
| 4 := 24
| 5 := 120
| 6 := 720
| 7 := 5040
| 8 := 40320
| 9 := 362880
| n := n * factorial (n - 1)

def digit_factorial_sum (n : ℕ) : ℕ :=
(n.to_digits 10).map factorial |>.sum

theorem solution : {n : ℕ | n < 1000 ∧ n = digit_factorial_sum n} = {1, 2, 145} :=
by
  sorry

end solution_l397_397648


namespace correct_choice_is_D_l397_397120

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

def test_function := λ x : ℝ, x + sin x

theorem correct_choice_is_D : is_odd_function test_function ∧ is_increasing_on test_function (Ioi 0) :=
sorry

end correct_choice_is_D_l397_397120


namespace choose_4_boxes_out_of_10_l397_397081

def num_ways_to_select_boxes : ℕ := 210

theorem choose_4_boxes_out_of_10 :
  ∃ ways : ℕ, ways = nat.choose 10 4 ∧ ways = num_ways_to_select_boxes :=
by {
  use nat.choose 10 4,
  split,
  refl,
  sorry
}

end choose_4_boxes_out_of_10_l397_397081


namespace right_triangle_leg_length_l397_397347

theorem right_triangle_leg_length (a c b : ℝ) (h : a = 4) (h₁ : c = 5) (h₂ : a^2 + b^2 = c^2) : b = 3 := 
by
  -- by is used for the proof, which we are skipping using sorry.
  sorry

end right_triangle_leg_length_l397_397347


namespace probability_odd_number_from_digits_l397_397831

theorem probability_odd_number_from_digits (digits : Finset ℕ) (h_digits : digits = {1, 2, 4, 5}) :
  (∑ n in (digits.to_list.permutations.filter (λ x, (x.head!).odd)), 1 : ℕ) / 
  ((digits.to_list.permutations.length : ℕ)) = 1/2 := 
by
  sorry

end probability_odd_number_from_digits_l397_397831


namespace set_A_main_inequality_l397_397706

def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 2|
def A : Set ℝ := {x | f x < 3}

theorem set_A :
  A = {x | -2 / 3 < x ∧ x < 0} :=
sorry

theorem main_inequality (s t : ℝ) (hs : -2 / 3 < s ∧ s < 0) (ht : -2 / 3 < t ∧ t < 0) :
  |1 - t / s| < |t - 1 / s| :=
sorry

end set_A_main_inequality_l397_397706


namespace find_length_AD_l397_397742

noncomputable def length_AD (AB AC BC : ℝ) (is_equal_AB_AC : AB = AC) (BD DC : ℝ) (D_midpoint : BD = DC) : ℝ :=
  let BE := BC / 2
  let AE := Real.sqrt (AB ^ 2 - BE ^ 2)
  AE

theorem find_length_AD (AB AC BC BD DC : ℝ) (is_equal_AB_AC : AB = AC) (D_midpoint : BD = DC) (H1 : AB = 26) (H2 : AC = 26) (H3 : BC = 24) (H4 : BD = 12) (H5 : DC = 12) :
  length_AD AB AC BC is_equal_AB_AC BD DC D_midpoint = 2 * Real.sqrt 133 :=
by
  -- the steps of the proof would go here
  sorry

end find_length_AD_l397_397742


namespace kombucha_bottles_l397_397727

theorem kombucha_bottles (b_m : ℕ) (c : ℝ) (r : ℝ) (m : ℕ)
  (hb : b_m = 15) (hc : c = 3.00) (hr : r = 0.10) (hm : m = 12) :
  (b_m * m * r) / c = 6 := by
  sorry

end kombucha_bottles_l397_397727


namespace pipe_P_fill_time_l397_397029

theorem pipe_P_fill_time :
  (∃ x : ℝ, 
    (1/x) + (1/9) + (1/18) = 1/2 ∧ x = 3) :=
begin
  use 3,
  field_simp,
  norm_num,
  sorry
end

end pipe_P_fill_time_l397_397029


namespace smallest_prime_factor_of_n_l397_397693

theorem smallest_prime_factor_of_n
  (a b : ℚ)
  (s : ℚ)
  (m n : ℕ)
  (hn : 0 < n)
  (hm : 0 < m)
  (hgcd : Nat.gcd m n = 1)
  (has : a + b = s)
  (hbs : a^2 + b^2 = s)
  (hns : s = m / n)
  (hnot_int : ¬ ∃ k : ℤ, s = k) :
  ∃ p : ℕ, Nat.prime p ∧ p ∣ n ∧ p = 5 :=
by
  sorry

end smallest_prime_factor_of_n_l397_397693


namespace ratio_of_angles_not_equal_to_ratio_of_sides_sum_of_sides_and_sins_two_possible_triangles_not_all_angles_acute_l397_397771

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
def triangle_sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ A B C, 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π ∧
           (a / sin A = b / sin B = c / sin C)

-- Statement 1: Incorrectness of the ratio of angles being equal to the ratio of sides
theorem ratio_of_angles_not_equal_to_ratio_of_sides (h : triangle_sides_opposite_angles a b c A B C) :
  ¬ ( A : B : C = a : b : c ) := sorry

-- Statement 2: Correctness of the proportion involving the sums
theorem sum_of_sides_and_sins (h : triangle_sides_opposite_angles a b c A B C):
  (a + b + c) / (sin A + sin B + sin C) = a / sin A := sorry

-- Statement 3: Existence of two triangles given specific conditions
theorem two_possible_triangles (h : triangle_sides_opposite_angles 4 3 4 A B C) (hB : B = π / 4):
  ∃ A1 A2, A1 ≠ A2 ∧ triangle_sides_opposite_angles 4 3 4 A1 B C ∧ triangle_sides_opposite_angles 4 3 4 A2 B C := sorry

-- Statement 4: Incorrectness of all angles being acute given specific inequality
theorem not_all_angles_acute (h : triangle_sides_opposite_angles a b c A B C) (hineq : a^2 + b^2 > c^2) :
  ¬ (∀ A B C, 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π ∧ A < π/2 ∧ B < π/2 ∧ C < π/2) := sorry

end ratio_of_angles_not_equal_to_ratio_of_sides_sum_of_sides_and_sins_two_possible_triangles_not_all_angles_acute_l397_397771


namespace find_angle_MOC_l397_397383

noncomputable def angle_MOC (s : ℝ) : Real :=
  let M := ⟨s / 2, 0⟩
  let O := some_intersection_point -- hypothetical function to get O
  let C := ⟨s, 0⟩
  let angle := ∠MOC
  Real.atan angle

theorem find_angle_MOC (s : ℝ) :
  let M := ⟨s / 2, 0⟩
  let O := some_intersection_point
  let C := ⟨s, 0⟩
  let angle := ∠MOC
  angle = Real.atan 3 := sorry

end find_angle_MOC_l397_397383


namespace intersection_M_N_l397_397903

open Set

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := 
by 
sorry

end intersection_M_N_l397_397903


namespace g_is_odd_l397_397388

def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end g_is_odd_l397_397388


namespace total_length_l397_397574

def length_pencil : ℕ := 12
def length_pen : ℕ := length_pencil - 2
def length_rubber : ℕ := length_pen - 3

theorem total_length : length_pencil + length_pen + length_rubber = 29 := by
  simp [length_pencil, length_pen, length_rubber]
  sorry

end total_length_l397_397574


namespace max_chord_line_eqn_l397_397570

def circle (x y : ℝ) : Prop := x^2 + y^2 + 4 * x + 3 = 0

def line_through_point (l : ℝ → ℝ → Prop) : Prop :=
  ∃ x y : ℝ, l x y ∧ (x, y) = (2, 3)

theorem max_chord_line_eqn :
  (∀ l : ℝ → ℝ → Prop, line_through_point l ∧ (∀ x y, l x y → circle x y) →
    (∃ a b c : ℝ, a * 2 + b * 3 + c = 0 ∧ (∀ x y, l x y ↔ a * x + b * y + c = 0)) ∧
    (∃ u v w : ℝ, u * x + v * y + w = 0 ∧ (u * -2 + v * 0 + w = 0))) →
  ∃ a b c : ℝ, a = 3 ∧ b = -4 ∧ c = 6 :=
sorry

end max_chord_line_eqn_l397_397570


namespace polynomial_factorization_c_is_8_l397_397071

theorem polynomial_factorization_c_is_8 {c q : ℝ} 
  (h : (3 * q + 4 = 0) ∧ (3x^3 + cx + 12 = (x^2 + qx + 2) * (3x + 4))) :
  c = 8 :=
by
  obtain ⟨hq, hf⟩ := h
  have : q = -4 / 3 := by linarith
  rw this at hf
  sorry


end polynomial_factorization_c_is_8_l397_397071


namespace total_volume_of_all_cubes_l397_397952

def volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume_of_cubes (num_cubes : ℕ) (side_length : ℕ) : ℕ :=
  num_cubes * volume side_length

theorem total_volume_of_all_cubes :
  total_volume_of_cubes 3 3 + total_volume_of_cubes 4 4 = 337 :=
by
  sorry

end total_volume_of_all_cubes_l397_397952


namespace matrix_not_invertible_iff_det_zero_l397_397656

theorem matrix_not_invertible_iff_det_zero :
  let M : Matrix (Fin 2) (Fin 2) ℝ := ![![2 + x, 9], ![4 - x, 10]] in
  ¬Matrix.invertible M ↔ x = 16 / 19 := by
  sorry

end matrix_not_invertible_iff_det_zero_l397_397656


namespace number_of_boxes_l397_397497

theorem number_of_boxes (eggs_per_box : ℕ) (total_eggs : ℕ) (h1 : eggs_per_box = 3) (h2 : total_eggs = 6) : total_eggs / eggs_per_box = 2 := by
  sorry

end number_of_boxes_l397_397497


namespace stratified_sampling_young_employees_l397_397339

variable (total_employees elderly_employees middle_aged_employees young_employees sample_size : ℕ)

-- Conditions
axiom total_employees_eq : total_employees = 750
axiom elderly_employees_eq : elderly_employees = 150
axiom middle_aged_employees_eq : middle_aged_employees = 250
axiom young_employees_eq : young_employees = 350
axiom sample_size_eq : sample_size = 15

-- The proof problem
theorem stratified_sampling_young_employees :
  young_employees / total_employees * sample_size = 7 :=
by
  sorry

end stratified_sampling_young_employees_l397_397339


namespace smallest_in_list_l397_397595

theorem smallest_in_list :
  let n1 := 111111 in
  let n2 := 2 * 6^2 + 1 * 6^1 + 0 * 6^0 in
  let n3 := 1 * 4^3 + 0 * 4^2 + 0 * 4^1 + 0 * 4^0 in
  let n4 := 1 * 8^2 + 0 * 8^1 + 1 * 8^0 in
  n1 < n2 ∧ n1 < n3 ∧ n1 < n4 :=
by
  let n1 := 63
  let n2 := 78
  let n3 := 64
  let n4 := 65
  exact ⟨by norm_num, by norm_num, by norm_num⟩

end smallest_in_list_l397_397595


namespace prime_count_between_50_and_100_with_prime_remainder_div_12_l397_397304

open Nat

def is_prime (n : ℕ) : Prop := nat.prime n

def primes_between (a b : ℕ) : list ℕ :=
  (list.range' a (b - a + 1)).filter is_prime

def prime_remainders (p : ℕ) : list ℕ := [2, 3, 5, 7, 11]

theorem prime_count_between_50_and_100_with_prime_remainder_div_12 : 
  (primes_between 50 100).filter (λ p, (p % 12) ∈ prime_remainders p).length = 7 :=
by
  sorry

end prime_count_between_50_and_100_with_prime_remainder_div_12_l397_397304


namespace sequence_convergence_l397_397777

-- Define a sequence of real numbers in the interval [1, ∞)
variable (x : ℕ → ℝ)
variable (hx : ∀ n, 1 ≤ x n)

-- Define the greatest integer function
def floor (r : ℝ) : ℤ := ⌊r⌋

-- Define the sequence [x_n^k] for a given k
def floor_sequence (k : ℕ) (n : ℕ) := floor ((x n) ^ k)

-- Assume floor_sequence is convergent for all natural numbers k
variable (hconv : ∀ k, ∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (floor_sequence k n - L) < ε)

theorem sequence_convergence : ∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (x n - L) < ε := sorry

end sequence_convergence_l397_397777


namespace average_of_middle_three_l397_397456

-- Let A be the set of five different positive whole numbers
def avg (lst : List ℕ) : ℚ := (lst.foldl (· + ·) 0 : ℚ) / lst.length
def maximize_diff (lst : List ℕ) : ℕ := lst.maximum' - lst.minimum'

theorem average_of_middle_three 
  (A : List ℕ) 
  (h_diff : maximize_diff A) 
  (h_avg : avg A = 5) 
  (h_len : A.length = 5) 
  (h_distinct : A.nodup) : 
  avg (A.erase_max.erase_min) = 3 := 
sorry

end average_of_middle_three_l397_397456


namespace hotel_room_allocation_l397_397972

theorem hotel_room_allocation :
  ∀ (fans : ℕ) (num_groups : ℕ) (room_capacity : ℕ), 
  fans = 100 → num_groups = 6 → room_capacity = 3 →
  (∀ (group_fans : ℕ) (groups : fin num_groups), group_fans ≤ fans →
   (∃ (total_rooms: ℕ), total_rooms = 37 ∧
    ∀ (group_fan_count : fin num_groups → ℕ), 
      (∀ g, group_fan_count g ≤ fans / num_groups + if (fans % num_groups > 0) then 1 else 0) →
      ∃ (rooms : fin num_groups → ℕ), 
        sum (λ g, rooms g) = total_rooms ∧ 
        ∀ g, group_fan_count g ≤ rooms g * room_capacity)) :=
by {
  intros,
  sorry
}

end hotel_room_allocation_l397_397972


namespace three_u_plus_two_v_l397_397499

noncomputable def number_prime_factors (n : ℕ) : ℕ := sorry

theorem three_u_plus_two_v (a b : ℕ) (u v : ℕ) 
  (h1 : log 2 a + 2 * log 2 (Nat.gcd a b) = 30)
  (h2 : log 2 b + 2 * log 2 (Nat.lcm a b) = 300)
  (hu : u = number_prime_factors a) (hv : v = number_prime_factors b) :
  3 * u + 2 * v = 230 := sorry

end three_u_plus_two_v_l397_397499


namespace total_cost_after_discount_l397_397154

def num_children : Nat := 6
def num_adults : Nat := 10
def num_seniors : Nat := 4

def child_ticket_price : Real := 12
def adult_ticket_price : Real := 20
def senior_ticket_price : Real := 15

def group_discount_rate : Real := 0.15

theorem total_cost_after_discount :
  let total_cost_before_discount :=
    num_children * child_ticket_price +
    num_adults * adult_ticket_price +
    num_seniors * senior_ticket_price
  let discount := group_discount_rate * total_cost_before_discount
  let total_cost := total_cost_before_discount - discount
  total_cost = 282.20 := by
  sorry

end total_cost_after_discount_l397_397154


namespace first_term_of_geometric_sequence_l397_397471

theorem first_term_of_geometric_sequence (a r : ℕ) :
  (a * r ^ 3 = 54) ∧ (a * r ^ 4 = 162) → a = 2 :=
by
  -- Provided conditions and the goal
  sorry

end first_term_of_geometric_sequence_l397_397471


namespace quotient_base6_division_l397_397635

theorem quotient_base6_division :
  let a := 2045
  let b := 14
  let base := 6
  a / b = 51 :=
by
  sorry

end quotient_base6_division_l397_397635


namespace cos_equiv_n_92_or_268_l397_397961

theorem cos_equiv_n_92_or_268 (n : ℤ) (h1 : 0 ≤ n ∧ n ≤ 360) :
  cos (n:ℝ) = cos 812 → n = 92 ∨ n = 268 := by
  sorry

end cos_equiv_n_92_or_268_l397_397961


namespace pure_imaginary_iff_a_eq_2_l397_397000

theorem pure_imaginary_iff_a_eq_2 (a : ℝ) : (∃ k : ℝ, (∃ x : ℝ, (2-a) / 2 = x ∧ x = 0) ∧ (2+a)/2 = k ∧ k ≠ 0) ↔ a = 2 :=
by
  sorry

end pure_imaginary_iff_a_eq_2_l397_397000


namespace square_side_difference_l397_397563

theorem square_side_difference
  (r h a b : ℝ)
  (h1: b + h = real.sqrt (r^2 - (b^2 / 4)))
  (h2: a - h = real.sqrt (r^2 - (a^2 / 4))) :
  b - a = (8 / 5) * h :=
by
  sorry

end square_side_difference_l397_397563


namespace points_in_circle_l397_397030

theorem points_in_circle 
  (points : Fin 51 → (Fin (1:ℝ) × Fin (1:ℝ))) :
  ∃ c r, r = (1 : ℝ) / 7 ∧ ∃ S ⊆ (Set.univ : Set (Fin 51)), S.card ≥ 3 ∧ ∀ p ∈ S, dist p.1 c ≤ r :=
sorry

end points_in_circle_l397_397030


namespace f_2011_l397_397682

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 4) = f x + f 2
axiom f_1 : f 1 = 2

theorem f_2011 : f 2011 = -2 :=
by sorry

end f_2011_l397_397682


namespace adult_ticket_cost_l397_397860

-- Conditions
def child_ticket_cost : ℕ := 6
def total_tickets_sold : ℕ := 225
def total_revenue : ℕ := 1875
def number_of_children : ℕ := 50

-- Conclusion to prove
theorem adult_ticket_cost :
  ∃ (A : ℕ), 
    let adult_tickets_sold := total_tickets_sold - number_of_children in
    let children_revenue := number_of_children * child_ticket_cost in
    let adult_revenue := total_revenue - children_revenue in
    adult_revenue = A * adult_tickets_sold ∧
    A = 9 :=
by
  sorry

end adult_ticket_cost_l397_397860


namespace temperature_on_friday_l397_397128

variables {M T W Th F : ℝ}

theorem temperature_on_friday
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 41) :
  F = 33 :=
  sorry

end temperature_on_friday_l397_397128


namespace apples_in_refrigerator_l397_397909

theorem apples_in_refrigerator (initial_apples : ℕ) (apples_for_muffins : ℕ): 
(initial_apples = 62) → (apples_for_muffins = 6) → 
(apples_in_refrigerator : ℕ) :=
by
  assume h1 : initial_apples = 62
  assume h2 : apples_for_muffins = 6
  
  -- Calculations using the conditions provided
  let half_apples := initial_apples / 2  -- Apples set aside for pie
  let remaining_apples := half_apples - apples_for_muffins  -- Apples left for refrigerator and muffins
  
  have half_apples_eq : half_apples = 31 := by sorry
  have remaining_apples_eq : remaining_apples = 25 := by sorry
  
  exact remaining_apples_eq

end apples_in_refrigerator_l397_397909


namespace arithmetic_sequence_S9_l397_397280

-- Define the sum of an arithmetic sequence: S_n
def arithmetic_sequence_sum (a d : ℕ → ℕ) (n : ℕ) : ℕ := (n * (2 * a (0) + (n - 1) * d (0))) / 2

-- Conditions
variable (a d : ℕ → ℕ)
variable (S_n : ℕ → ℕ)
variable (h1 : S_n 3 = 9)
variable (h2 : S_n 6 = 27)

-- Question: Prove that S_9 = 54
theorem arithmetic_sequence_S9 : S_n 9 = 54 := by
    sorry

end arithmetic_sequence_S9_l397_397280


namespace find_original_number_of_men_l397_397567

noncomputable def original_number_of_men : ℕ := 22

theorem find_original_number_of_men 
  (x : ℕ)
  (work_done_by_x : x * 20 = x * 20)
  (work_done_by_x_minus_2 : (x - 2) * 22 = (x - 2) * 22) :
  x = original_number_of_men :=
by
  -- We need to add the equivalence relation derived from the conditions.
  have h1: x * 20 = (x - 2) * 22 := by sorry
  -- Getting to the simplified form from h1.
  have h2: 20 * x = 22 * x - 44 := by sorry
  -- Rearranging and solving for x.
  have h3: 20 * x - 22 * x = -44 := by sorry
  -- Combine like terms.
  have h4: -2 * x = -44 := by sorry
  -- Divide both sides by -2.
  have h5: x = 22 := by sorry
  exact h5

end find_original_number_of_men_l397_397567


namespace max_distance_M_to_line_l397_397762

open Real

section problem
variables (α θ ρ : ℝ) (x y : ℝ)

def parametric_curve : Prop :=
  (x = 2 * sqrt 3 * cos α) ∧ (y = 2 * sin α) ∧ (0 < α ∧ α < π)

def polar_point_P : Prop :=
  (ρ = 4 * sqrt 2) ∧ (θ = π / 4)

def polar_line_eqn : Prop :=
  ρ * sin (θ - π / 4) + 5 * sqrt 2 = 0

def cartesian_eqn_line : Prop :=
  x - y - 10 = 0

def standard_eqn_curve : Prop := 
  (x^2 / 12 + y^2 / 4 = 1) ∧ (y > 0)

def midpoint_M (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def distance_to_line (M : ℝ × ℝ) : ℝ :=
  abs (M.1 - M.2 - 10) / sqrt 2

theorem max_distance_M_to_line :
  ∃ (α : ℝ), parametric_curve α ∧ polar_point_P ∧ 
    polar_line_eqn ∧ 
    (distance_to_line (midpoint_M (4, 4) (2 * sqrt 3 * cos α, 2 * sin α)) ≤ 6 * sqrt 2) :=
sorry

end problem

end max_distance_M_to_line_l397_397762


namespace jane_preferred_number_l397_397222
  
def digitSum (n : ℕ) : ℕ :=
  n.digits.sum

theorem jane_preferred_number :
  ∃ n, 100 ≤ n ∧ n ≤ 180 ∧ 13 ∣ n ∧ ¬ 2 ∣ n ∧ digitSum n % 4 = 0 ∧ (n = 143 ∨ n = 169) :=
by
  sorry

end jane_preferred_number_l397_397222


namespace accept_batch_l397_397551

-- Define the parameters given in the problem
def prob_defective := 0.02
def sample_size := 480
def num_defective := 12
def sample_proportion := num_defective / sample_size.toFloat
def significance_level := 0.05
def critical_value := 1.645
def p0 := 0.02
def q0 := 1 - p0
def sqrt_n := Real.sqrt sample_size.toFloat
def sqrt_p0_q0 := Real.sqrt (p0 * q0)
def test_statistic := (sample_proportion - p0) * sqrt_n / sqrt_p0_q0

-- The batch can be accepted if the test statistic is less than the critical value
theorem accept_batch : test_statistic < critical_value := by
  have sample_proportion := (12 : Float) / 480
  have sqrt_n := Real.sqrt 480
  have sqrt_p0_q0 := Real.sqrt (0.02 * (1 - 0.02))
  have test_statistic := (sample_proportion - 0.02) * sqrt_n / sqrt_p0_q0
  show test_statistic < 1.645
  sorry

end accept_batch_l397_397551


namespace rectangle_perimeter_l397_397170

theorem rectangle_perimeter (s : ℝ) (h1 : 4 * s = 160) : 2 * (s + s / 2) = 120 :=
by
  have s_val : s = 40 := by
    linarith
  have length : s = 40 := s_val
  have width : s / 2 = 20 := by
    rw [s_val]
    norm_num
  rw [length, width]
  norm_num
  sorry  -- Placeholder for the actual proof

end rectangle_perimeter_l397_397170


namespace shortest_path_cylinder_proof_l397_397845

noncomputable def shortest_path_cylinder (r h : ℝ) : ℝ :=
  Real.sqrt (h^2 + (π * r)^2)

theorem shortest_path_cylinder_proof (r h : ℝ) :
  shortest_path_cylinder r h = Real.sqrt (h^2 + π^2 * r^2) :=
by
  sorry

end shortest_path_cylinder_proof_l397_397845


namespace cotangent_tangent_relation_l397_397770

theorem cotangent_tangent_relation
  (A B C : ℝ)
  (hABC : A + B + C = 180)
  (h1 : (Real.cot A) * (Real.cot C) = 1 / 4)
  (h2 : (Real.cot B) * (Real.cot C) = 1 / 9) :
  Real.tan C = 35.635 :=
by
  sorry

end cotangent_tangent_relation_l397_397770


namespace simplify_and_evaluate_expression_l397_397812

theorem simplify_and_evaluate_expression (a b : ℤ) (h₁ : a = -2) (h₂ : b = 1) :
  ((a - 2 * b) ^ 2 - (a + 3 * b) * (a - 2 * b)) / b = 20 :=
by
  sorry

end simplify_and_evaluate_expression_l397_397812


namespace female_members_count_l397_397343

theorem female_members_count (M F : ℕ) (h1 : F = 2 * M) (h2 : F + M = 18) : F = 12 :=
by
  -- the proof will go here
  sorry

end female_members_count_l397_397343


namespace range_of_g_l397_397940

noncomputable def g (x : ℝ) : ℝ :=
  (Real.sin x) ^ 6 + (Real.cos x) ^ 2

theorem range_of_g : set.Icc ((3 * Real.sqrt 3 - 2) / (3 * Real.sqrt 3)) 1 ⊆ set.range g :=
by
  sorry

end range_of_g_l397_397940


namespace find_k_l397_397244

def vector (α : Type*) := α × α

def a : vector ℝ := (1, 3)
def b (k : ℝ) : vector ℝ := (-2, k)

def vector_parallel (u v : vector ℝ) : Prop :=
  ∃ (λ:ℝ), λ • u = v

theorem find_k (k : ℝ) : 
  vector_parallel ((a.1 + 2 * b(k).1, a.2 + 2 * b(k).2))
                  ((3 * a.1 - b(k).1, 3 * a.2 - b(k).2))
  → k = -6 := 
by 
  sorry

end find_k_l397_397244


namespace percentage_of_Y_pay_X_is_paid_correct_l397_397867

noncomputable def percentage_of_Y_pay_X_is_paid
  (total_pay : ℝ) (Y_pay : ℝ) : ℝ :=
  let X_pay := total_pay - Y_pay
  (X_pay / Y_pay) * 100

theorem percentage_of_Y_pay_X_is_paid_correct :
  percentage_of_Y_pay_X_is_paid 700 318.1818181818182 = 120 := 
by
  unfold percentage_of_Y_pay_X_is_paid
  sorry

end percentage_of_Y_pay_X_is_paid_correct_l397_397867


namespace shortest_distance_is_zero_l397_397792

noncomputable def line1 (t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + 3 * t, 2 - 3 * t, 3 + 2 * t)

noncomputable def line2 (s : ℝ) : ℝ × ℝ × ℝ :=
  (2 * s, 1 + 4 * s, 5 - 3 * s)

noncomputable def distance_squared (t s : ℝ) : ℝ :=
  let P := line1 t
  let Q := line2 s
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2

theorem shortest_distance_is_zero :
  ∃ t s : ℝ, distance_squared t s = 0 :=
sorry

end shortest_distance_is_zero_l397_397792


namespace imaginary_part_of_z_l397_397836

def z : ℂ := (2 - (Real.sqrt 3) / 4) * Complex.I

theorem imaginary_part_of_z :
  Complex.im z = 2 - (Real.sqrt 3) / 4 :=
sorry

end imaginary_part_of_z_l397_397836


namespace customer_bought_two_pens_l397_397747

noncomputable def combination (n k : ℕ) : ℝ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem customer_bought_two_pens :
  ∃ n : ℕ, combination 5 n / combination 8 n = 0.3571428571428571 ↔ n = 2 := by
  sorry

end customer_bought_two_pens_l397_397747


namespace minimum_rooms_to_accommodate_fans_l397_397979

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l397_397979


namespace closest_point_distance_l397_397018

def point := ℝ × ℝ

def region_D (p : point) : Prop :=
  (p.1 ≥ 2) ∧ (p.1 + p.2 ≤ 0) ∧ (p.1 - p.2 - 10 ≤ 0)

def reflection (p : point) : point :=
  let (x, y) := p in (1 / 5 * (4 * x - y), 1 / 5 * (2 * y - 3 * x))

def region_E (p : point) : Prop := region_D (reflection p)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def minimum_distance (D E : point → Prop) : ℝ :=
  Inf {d | ∃ p1 p2, D p1 ∧ E p2 ∧ distance p1 p2 = d}

theorem closest_point_distance :
  minimum_distance region_D region_E = 12 * real.sqrt 5 / 5 := 
sorry

end closest_point_distance_l397_397018


namespace increase_in_students_l397_397749

theorem increase_in_students {total_students : ℕ} (initial_percent final_percent : ℚ) 
  (h1 : total_students = 650) (h2 : initial_percent = 0.70) (h3 : final_percent = 0.80) : 
  (final_percent - initial_percent) * total_students = 65 := 
  by
  have h4 : final_percent - initial_percent = 0.10 := sorry
  have h5 : 0.10 * total_students = 65 := sorry
  sorry

end increase_in_students_l397_397749


namespace semicircle_area_l397_397548

theorem semicircle_area (x : ℝ) (y : ℝ) (r : ℝ) (h1 : x = 1) (h2 : y = 3) (h3 : x^2 + y^2 = (2*r)^2) :
  (1/2) * π * r^2 = (13 * π) / 8 :=
by
  sorry

end semicircle_area_l397_397548


namespace exists_coprime_integers_divisible_l397_397005

theorem exists_coprime_integers_divisible {a b p : ℤ} : ∃ k l : ℤ, gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
by
  sorry

end exists_coprime_integers_divisible_l397_397005


namespace greatest_possible_value_of_x_l397_397106

theorem greatest_possible_value_of_x :
  ∃ (x : ℚ), x = 9 / 5 ∧ 
  (\left(5 * x - 20) / (4 * x - 5)) ^ 2 + \left((5 * x - 20) / (4 * x - 5)) = 20 ∧ x ≥ 0 :=
begin
  existsi (9 / 5 : ℚ),
  split,
  { refl },
  split,
  { sorry },
  { sorry }
end

end greatest_possible_value_of_x_l397_397106


namespace m_and_n_must_have_same_parity_l397_397856

-- Define the problem conditions
def square_has_four_colored_edges (square : Type) : Prop :=
  ∃ (colors : Fin 4 → square), true

def m_and_n_same_parity (m n : ℕ) : Prop :=
  (m % 2 = n % 2)

-- Formalize the proof statement based on the conditions
theorem m_and_n_must_have_same_parity (m n : ℕ) (square : Type)
  (H : square_has_four_colored_edges square) : 
  m_and_n_same_parity m n :=
by 
  sorry

end m_and_n_must_have_same_parity_l397_397856


namespace louie_monthly_payment_l397_397533

def monthly_payment (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  let A := P * (1 + r/n) ^ (n * t)
  in A / t

theorem louie_monthly_payment : 
  monthly_payment 1000 0.1 1 3 ≈ 444 :=
by
  -- Here we assert the correctness of the calculation and result
  sorry

end louie_monthly_payment_l397_397533


namespace students_taking_either_but_not_both_l397_397636

theorem students_taking_either_but_not_both (C P_both P_only : ℕ) (hC : C = 30) (hP_both : P_both = 15) (hP_only : P_only = 12) : C - P_both + P_only = 27 :=
by 
  rw [hC, hP_both, hP_only]
  exact rfl

end students_taking_either_but_not_both_l397_397636


namespace negation_of_all_men_are_tall_l397_397064

variable {α : Type}
variable (man : α → Prop) (tall : α → Prop)

theorem negation_of_all_men_are_tall :
  (¬ ∀ x, man x → tall x) ↔ ∃ x, man x ∧ ¬ tall x :=
sorry

end negation_of_all_men_are_tall_l397_397064


namespace line_parameterization_l397_397213

theorem line_parameterization (r k : ℝ) (t : ℝ) :
  (∀ x y : ℝ, (x, y) = (r + 3 * t, 2 + k * t) → (y = 2 * x - 5) ) ∧
  (t = 0 → r = 7 / 2) ∧
  (t = 1 → k = 6) :=
by
  sorry

end line_parameterization_l397_397213


namespace job_pay_per_pound_l397_397816

def p := 2
def M := 8 -- Monday
def T := 3 * M -- Tuesday
def W := 0 -- Wednesday
def R := 18 -- Thursday
def total_picked := M + T + W + R -- total berries picked
def money := 100 -- total money wanted

theorem job_pay_per_pound :
  total_picked = 50 → p = money / total_picked :=
by
  intro h
  rw [h]
  norm_num
  exact rfl

end job_pay_per_pound_l397_397816


namespace reciprocal_of_sum_of_fractions_l397_397075

theorem reciprocal_of_sum_of_fractions :
  (1 / (1 / 4 + 1 / 6)) = 12 / 5 :=
by
  sorry

end reciprocal_of_sum_of_fractions_l397_397075


namespace volume_of_cube_is_1000_l397_397917

noncomputable def volume_of_cube_with_diagonal (d : ℝ) : ℝ :=
  let s := d / Real.sqrt 3
  in s^3

theorem volume_of_cube_is_1000 (d : ℝ) (h : d = 10 * Real.sqrt 3) : volume_of_cube_with_diagonal d = 1000 :=
by {
  rw h, -- Replace d with 10 * sqrt 3
  unfold volume_of_cube_with_diagonal,
  rw [Real.mul_div_cancel' _ (Real.sqrt_ne_zero'.mpr (Real.sqrt_pos.mpr three_pos)), Real.pow_succ, Real.mul_assoc, Real.div_self (Real.sqrt_ne_zero'.mpr three_pos), Real.sqrt_mul_self three_pos],
  norm_num,
}

end volume_of_cube_is_1000_l397_397917


namespace tile_2x24_grid_with_dominoes_l397_397732

theorem tile_2x24_grid_with_dominoes : 
    let grid_width := 24
    let grid_height := 2
    let dominoes_tiling_ways := 27 in 
    ∃ n : ℕ, (n = dominoes_tiling_ways ∧ n = 27) :=
begin
  sorry,
end

end tile_2x24_grid_with_dominoes_l397_397732


namespace parallel_lines_slope_l397_397324

theorem parallel_lines_slope (m : ℝ) (h : (x + (1 + m) * y + m - 2 = 0) ∧ (m * x + 2 * y + 6 = 0)) :
  m = 1 ∨ m = -2 :=
  sorry

end parallel_lines_slope_l397_397324


namespace maximum_rectangles_in_stepwise_triangle_l397_397024

theorem maximum_rectangles_in_stepwise_triangle (n : ℕ) (h : n = 6) : 
  let num_rectangles := ∑ i in range (n+1), ∑ j in range (n+1-i), (i * (i + 1) * j * (j + 1)) / 4 in
  num_rectangles = 126 :=
by
  sorry

end maximum_rectangles_in_stepwise_triangle_l397_397024


namespace three_digit_condition_l397_397225

-- Define the three-digit number and its rotated variants
def num (a b c : ℕ) := 100 * a + 10 * b + c
def num_bca (a b c : ℕ) := 100 * b + 10 * c + a
def num_cab (a b c : ℕ) := 100 * c + 10 * a + b

-- The main statement to prove
theorem three_digit_condition (a b c: ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) (h_c : 0 ≤ c ∧ c ≤ 9) :
  2 * num a b c = num_bca a b c + num_cab a b c ↔ 
  (num a b c = 111 ∨ num a b c = 222 ∨ 
  num a b c = 333 ∨ num a b c = 370 ∨ 
  num a b c = 407 ∨ num a b c = 444 ∨ 
  num a b c = 481 ∨ num a b c = 518 ∨ 
  num a b c = 555 ∨ num a b c = 592 ∨ 
  num a b c = 629 ∨ num a b c = 666 ∨ 
  num a b c = 777 ∨ num a b c = 888 ∨ 
  num a b c = 999) := by
  sorry

end three_digit_condition_l397_397225


namespace calculate_power_product_l397_397947

theorem calculate_power_product :
  ( (2 / 3) ^ 4 * ( (2 / 3) ^ (-2) ) ) = 4 / 9 := 
  by sorry

end calculate_power_product_l397_397947


namespace tommy_writing_time_l397_397864

def numUniqueLettersTommy : Nat := 5
def numRearrangementsPerMinute : Nat := 20
def totalRearrangements : Nat := numUniqueLettersTommy.factorial
def minutesToComplete : Nat := totalRearrangements / numRearrangementsPerMinute
def hoursToComplete : Rat := minutesToComplete / 60

theorem tommy_writing_time :
  hoursToComplete = 0.1 := by
  sorry

end tommy_writing_time_l397_397864


namespace friendship_count_same_l397_397038

theorem friendship_count_same (n : ℕ) (Friend : Fin n → Fin n → Prop)
  (mutual_friendship : ∀ i j, Friend i j ↔ Friend j i) :
  ∃ i j : Fin n, i ≠ j ∧ (λ x, (Finset.univ.filter (Friend x)).card) i = (λ x, (Finset.univ.filter (Friend x)).card) j :=
by
  -- to be proven
  sorry

end friendship_count_same_l397_397038


namespace no_such_set_exists_l397_397965

open Nat Set

theorem no_such_set_exists (M : Set ℕ) : 
  (∀ m : ℕ, m > 1 → ∃ a b : ℕ, a ∈ M ∧ b ∈ M ∧ a + b = m) →
  (∀ a b c d : ℕ, a ∈ M → b ∈ M → c ∈ M → d ∈ M → 
    a > 10 → b > 10 → c > 10 → d > 10 → a + b = c + d → a = c ∨ a = d) → 
  False := by
  sorry

end no_such_set_exists_l397_397965


namespace rate_per_kg_for_apples_l397_397863

theorem rate_per_kg_for_apples (A : ℝ) :
  (8 * A + 9 * 45 = 965) → (A = 70) :=
by
  sorry

end rate_per_kg_for_apples_l397_397863


namespace not_p_and_not_q_true_l397_397737

variable (p q: Prop)

theorem not_p_and_not_q_true (h1: ¬ (p ∧ q)) (h2: ¬ (p ∨ q)) : ¬ p ∧ ¬ q :=
by
  sorry

end not_p_and_not_q_true_l397_397737


namespace rooms_needed_l397_397989

-- Defining all necessary conditions
constant fans : ℕ := 100
constant teams : ℕ := 3
constant max_people_per_room : ℕ := 3
constant groups : ℕ := 6

-- Proposition that the number of rooms required is exactly 37.
theorem rooms_needed (h1 : ∀ i, 1 ≤ i ∧ i ≤ groups) 
  (h2 : groups = 2 * teams) 
  (h3 : teams = 3)
  (h4 : max_people_per_room = 3)
  (h5 : ∃ i, 0 ≤ i ∧ ∑ i in finset.range groups, i = fans):
  37 = (∃ (n : ℕ), ∃ (i : ℕ), n = ∑ i in finset.range groups, nat.ceil (i / max_people_per_room)):
sorry

end rooms_needed_l397_397989


namespace no_positive_integer_solutions_l397_397440

theorem no_positive_integer_solutions (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^2017 - 1 ≠ (x - 1) * (y^2015 - 1) :=
by sorry

end no_positive_integer_solutions_l397_397440


namespace biological_experiment_correct_l397_397524

-- Define the descriptions of biological experiments as propositions.
def A : Prop := ∀ (amylase_specificity : bool) (iodine_solution : bool), 
                 amylase_specificity → ¬ iodine_solution

def B : Prop := ∀ (yeast_cells_counted : bool) (pipette_usage : bool)
                          (cover_slip : bool), 
                 yeast_cells_counted → (pipette_usage ∧ cover_slip)

def C : Prop := ∀ (bacteriophage_infecting : bool) 
                          (bacterial_transformation : bool) 
                          (genetic_material_continuity : bool),
                 bacteriophage_infecting = bacterial_transformation 
                 ∧ bacterial_transformation = genetic_material_continuity

def D : Prop := ∀ (rose_cuttings : bool) (low_2_4d : bool) 
                          (high_2_4d : bool) (root_count : ℕ),
                 low_2_4d → high_2_4d → root_count

-- The problem statement
theorem biological_experiment_correct : C :=
by sorry

end biological_experiment_correct_l397_397524


namespace g_is_odd_l397_397393

def g (x : ℝ) : ℝ := 1 / (3 ^ x - 1) + 1 / 3

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_is_odd_l397_397393


namespace license_plate_count_l397_397730

/-- The number of license plates consisting of 3 letters followed by 3 digits,
    with the first digit being odd, the second digit being even, and the third digit being any digit,
    is 17,576,000. --/
theorem license_plate_count : 
  let letters := 26 in
  let odd_digits := 5 in
  let even_digits := 5 in
  let any_digits := 10 in
  letters * letters * letters * odd_digits * even_digits * any_digits = 17_576_000 := 
by
  sorry

end license_plate_count_l397_397730


namespace hyperbola_asymptotes_l397_397053

theorem hyperbola_asymptotes:
  (∀ x y : ℝ, (x^2) / 3 - (y^2) / 4 = 1 → ((y = (2 * sqrt 3 / 3) * x) ∨ (y = -(2 * sqrt 3 / 3) * x))) :=
by
  intro x y
  intro h
  sorry

end hyperbola_asymptotes_l397_397053


namespace product_equation_l397_397163

theorem product_equation (a b : ℝ) (h1 : ∀ (a b : ℝ), 0.2 * b = 0.9 * a - b) : 
  0.9 * a - b = 0.2 * b :=
by
  sorry

end product_equation_l397_397163


namespace systematic_sampling_methods_l397_397906

-- Define the conditions for each sampling method
def method_1 : Prop :=
  ∃ i_0 ∈ (finset.range 16), ∀ i ∈ ({i_0, i_0 + 5, i_0 + 10} : finset ℕ),
  if i > 15 then (i - 15) ∈ (finset.range 16) else i ∈ (finset.range 16)

def method_2 : Prop :=
  ∃ k ≥ 1, ∀ t ∈ (finset.range 60).filter (λ n, n % 5 = 0), (20 * k * t / 60) < 1

def method_3 : Prop :=
  ∃ (P : ℕ → bool), ¬∃ N, ∀ n, n < N → P n

def method_4 : Prop :=
  ∀ (n : ℕ), n ∈ (finset.range 16) → (n % 14 + 1 = 14)

-- Define systematic sampling
def is_systematic (method : Prop) : Prop :=
  method

-- Translate the problem to prove the question equals the answer given the conditions
theorem systematic_sampling_methods :
  (is_systematic method_1 ∧ is_systematic method_2 ∧ ¬ is_systematic method_3 ∧ is_systematic method_4) ↔ ([1, 2, 4]) :=
sorry

end systematic_sampling_methods_l397_397906


namespace eccentricity_range_no_lambda_l397_397618

noncomputable theory

variables {a b : ℝ} (h : a > b > 0) (e : ℝ) (lambda : ℝ)
def c := sqrt (a^2 - b^2)
def e := sqrt (1 - (b^2) / (a^2))
def ellipsoid_c1 (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1
def P := {p : ℝ × ℝ // ellipsoid_c1 p.1 p.2}
def F1 : ℝ × ℝ := (-c, 0)
def F2 : ℝ × ℝ := (c, 0)
def right_vertex_A : ℝ × ℝ := (a, 0)
def inner_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def max_value_in_range (P : ℝ × ℝ) : Prop :=
  c^2 ≤ inner_product (P.1 - F1.1, P.2 - F1.2) (P.1 - F2.1, P.2 - F2.2) ∧
  inner_product (P.1 - F1.1, P.2 - F1.2) (P.1 - F2.1, P.2 - F2.2) ≤ 3 * c^2
def hyperbola_c2 (x y : ℝ) : Prop := (x^2) / (c^2) - (y^2) / (a^2 - c^2) = 1
def B := {p : ℝ × ℝ // hyperbola_c2 p.1 p.2 ∧ 0 < p.1 ∧ 0 < p.2}
def angle_BAF1 λ := ∀ (B : ℝ × ℝ), λ * (atan2 (F1.2 - B.2) (F1.1 - B.1)) = (atan2 (F1.2 - B.2) (F1.1 - B.1))

theorem eccentricity_range : e ∈ set.Icc (sqrt 2 / 2) 1 :=
sorry

theorem no_lambda : ¬ ∃ (λ : ℝ), λ > 0 ∧ ∀ (B : ℝ × ℝ) (hB : B ∈ B), 
  angle_BAF1 λ B :=
sorry

end eccentricity_range_no_lambda_l397_397618


namespace number_of_nonempty_proper_subsets_l397_397065

open Finset

theorem number_of_nonempty_proper_subsets :
  let S := {y ∈ range 7 | ∃ x ∈ range 3, y = 6 - x^2} in
  S = {2, 5, 6} ∧ (card (powerset S) - 2) = 6 :=
by 
  let S := {y ∈ range 7 | ∃ x ∈ range 3, y = 6 - x^2}
  have hs : S = {2, 5, 6} := sorry
  have h_subsets : card (powerset S) = 8 := sorry
  exact ⟨hs, by rw [h_subsets, Nat.sub_eq_of_eq_add]; exact rfl⟩

end number_of_nonempty_proper_subsets_l397_397065
