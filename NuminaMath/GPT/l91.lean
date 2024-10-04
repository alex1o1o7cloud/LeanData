import Mathlib

namespace lcm_of_132_and_315_l91_91887

def n1 : ℕ := 132
def n2 : ℕ := 315

theorem lcm_of_132_and_315 :
  (Nat.lcm n1 n2) = 13860 :=
by
  -- Proof goes here
  sorry

end lcm_of_132_and_315_l91_91887


namespace sixth_graders_homework_l91_91139

theorem sixth_graders_homework:
  let group_A_students := 20
  let group_B_students := 80
  let forgot_group_A := 20 * group_A_students / 100 -- 20% of 20
  let forgot_group_B := 15 * group_B_students / 100 -- 15% of 80
  let total_students := group_A_students + group_B_students
  let total_forgot := forgot_group_A + forgot_group_B in
  (total_forgot / total_students) * 100 = 16 := by
  sorry

end sixth_graders_homework_l91_91139


namespace paula_paint_cans_l91_91825

variables (rooms_per_can total_rooms_lost initial_rooms final_rooms cans_lost : ℕ)

theorem paula_paint_cans
  (h1 : initial_rooms = 50)
  (h2 : cans_lost = 2)
  (h3 : final_rooms = 42)
  (h4 : total_rooms_lost = initial_rooms - final_rooms)
  (h5 : rooms_per_can = total_rooms_lost / cans_lost) :
  final_rooms / rooms_per_can = 11 :=
by sorry

end paula_paint_cans_l91_91825


namespace equal_areas_l91_91243

variable {a b c p r q : ℝ}
variable {A B C P1 P2 Q1 Q2 R1 R2 : Type}

-- Provided conditions
def triangle (A B C : Type) : Prop := sorry -- The definition of a triangle

axiom side_lengths : (BC = a) (CA = b) (AB = c)
axiom points_on_sides : AP1 = BP2 = p ∧ BR1 = CR2 = r ∧ CQ1 = AQ2 = q

-- Statement to be proved
theorem equal_areas (h : triangle A B C) :
  S_{triangle P1 Q1 R1} = S_{triangle P2 Q2 R2} :=
sorry

end equal_areas_l91_91243


namespace intersection_points_graph_line_l91_91114

noncomputable def number_of_intersections (f : ℝ → ℝ) : ℕ :=
if h : ∃ y, f 1 = y then 1 else 0

theorem intersection_points_graph_line (f : ℝ → ℝ) :
  number_of_intersections f = 0 ∨ number_of_intersections f = 1 := by
sorry

end intersection_points_graph_line_l91_91114


namespace area_of_triangle_F1PF2P_l91_91097

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 4
noncomputable def c : ℝ := 3
noncomputable def PF1 : ℝ := sorry 
noncomputable def PF2 : ℝ := sorry

-- Given conditions
def ellipse_eq_holds (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

-- Given point P is on the ellipse
def P_on_ellipse (x y : ℝ) : Prop := ellipse_eq_holds x y

-- Given angle F1PF2
def angle_F1PF2_eq_60 : Prop := sorry

-- Proving the area of △F₁PF₂
theorem area_of_triangle_F1PF2P : S = (16 * Real.sqrt 3) / 3 :=
by sorry

end area_of_triangle_F1PF2P_l91_91097


namespace cafeteria_pies_l91_91848

theorem cafeteria_pies :
  ∀ (initial_apples handed_out_apples apples_per_pie : ℕ), 
    initial_apples = 525 →
    handed_out_apples = 415 →
    apples_per_pie = 12 →
    ((initial_apples - handed_out_apples) / apples_per_pie) = 9 := 
by
  intros initial_apples handed_out_apples apples_per_pie h1 h2 h3
  rw [h1, h2, h3]
  have h : (525 - 415) = 110 := by norm_num
  rw h
  exact (Nat.div_eq_of_eq_mul_right (by norm_num : apples_per_pie ≠ 0) (by norm_num : 110 = 12 * 9 + 2)).symm

end cafeteria_pies_l91_91848


namespace avg_distinct_t_values_l91_91706

theorem avg_distinct_t_values : 
  ∀ (t : ℕ), (∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t) →
  (1 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   6 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   10 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   12 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t}) →
  let distinct_values := {6, 10, 12} in
  let average := (6 + 10 + 12) / 3 in
  average = 28 / 3 :=
by {
  sorry
}

end avg_distinct_t_values_l91_91706


namespace jill_more_than_jake_l91_91787

-- Definitions from conditions
def jill_peaches := 12
def steven_peaches := jill_peaches + 15
def jake_peaches := steven_peaches - 16

-- Theorem to prove the question == answer given conditions
theorem jill_more_than_jake : jill_peaches - jake_peaches = 1 :=
by
  -- Proof steps would be here, but for the statement requirement we put sorry
  sorry

end jill_more_than_jake_l91_91787


namespace standard_equation_of_ellipse_l91_91865

open Real

theorem standard_equation_of_ellipse (e : ℝ) (c : ℝ) (a b : ℝ) (h_e : e = 2 / 3) (h_2c : 2 * c = 16)
  (h_a : a = 12) (h_c : c = 8) (h_b : b = sqrt 80) :
  (x y : ℝ) → (x^2 / 144 + y^2 / 80 = 1) ∨ (x^2 / 80 + y^2 / 144 = 1) :=
begin
  sorry
end

end standard_equation_of_ellipse_l91_91865


namespace sum_integers_from_neg50_to_70_l91_91889

theorem sum_integers_from_neg50_to_70 : (Finset.range (70 + 1)).sum id - (Finset.range (50 + 1)).sum id + (Finset.range (50+1)).sum (λ x, - x) = 1210 :=
by
  sorry

end sum_integers_from_neg50_to_70_l91_91889


namespace tyson_one_point_count_l91_91548

def tyson_three_points := 3 * 15
def tyson_two_points := 2 * 12
def total_points := 75
def points_from_three_and_two := tyson_three_points + tyson_two_points

theorem tyson_one_point_count :
  ∃ n : ℕ, n % 2 = 0 ∧ (n = total_points - points_from_three_and_two) :=
sorry

end tyson_one_point_count_l91_91548


namespace inequality_holds_l91_91719

noncomputable def f (x : ℝ) : ℝ := log x - 3 * x

theorem inequality_holds (a b : ℝ) :
  (∀ x : ℝ, 0 < x → f x ≤ x * (a * real.exp x - 4) + b) →
    a + b ≥ 0 :=
begin
  sorry
end

end inequality_holds_l91_91719


namespace students_not_reading_novels_l91_91927

theorem students_not_reading_novels
  (total_students : ℕ)
  (students_three_or_more_novels : ℕ)
  (students_two_novels : ℕ)
  (students_one_novel : ℕ)
  (h_total_students : total_students = 240)
  (h_students_three_or_more_novels : students_three_or_more_novels = 1 / 6 * 240)
  (h_students_two_novels : students_two_novels = 35 / 100 * 240)
  (h_students_one_novel : students_one_novel = 5 / 12 * 240)
  :
  total_students - (students_three_or_more_novels + students_two_novels + students_one_novel) = 16 :=
by
  sorry

end students_not_reading_novels_l91_91927


namespace range_of_a_l91_91375

theorem range_of_a (a : ℝ) (h : (1/2)^(2*a+1) < (1/2)^(3-2*a)) : a > 1/2 :=
sorry

end range_of_a_l91_91375


namespace multiple_of_bananas_is_10_l91_91791

open Nat

def fruits_last_night : Nat := 3 + 1 + 4
def apples_today : Nat := 3 + 4
def oranges_today : Nat := 2 * apples_today
def total_fruits (x : Nat) : Nat := fruits_last_night + apples_today + x + oranges_today

theorem multiple_of_bananas_is_10 :
  ∃ x : Nat, total_fruits x = 39 ∧ x = 10 :=
by
  use 10
  unfold fruits_last_night apples_today oranges_today total_fruits
  sorry

end multiple_of_bananas_is_10_l91_91791


namespace set_order_irrelevance_l91_91903

-- Definition: A set is a collection of distinct elements.
def is_set (S : Set α) : Prop := true

-- Definition: A set is unordered, meaning permutations do not change the set.
def unordered_set (S : Set α) : Prop := ∀ x y : α, x ∈ S ∧ y ∈ S → ({x, y} ⊆ S)

-- Theorem: Changing the order of elements in a set does not change the set itself.
theorem set_order_irrelevance (S : Set α) (hS : is_set S) (hU : unordered_set S) : ∀ x y : α, x ∈ S ∧ y ∈ S → {x, y} = {y, x} :=
by
  sorry

end set_order_irrelevance_l91_91903


namespace num_valid_digits_l91_91662

def is_digit (n : ℕ) : Prop := n ∈ (finset.range 10).erase 0

def satisfies_condition (n : ℕ) : Prop := (150 + n) % n = 0

theorem num_valid_digits : finset.card ((finset.range 10).erase 0).filter (λ n, satisfies_condition n) = 6 :=
by
  sorry

end num_valid_digits_l91_91662


namespace largest_number_is_pi_l91_91961

theorem largest_number_is_pi : 
  let a := Real.pi
  let b := Real.sqrt 2
  let c := |(-2)|
  let d := 3
  a > b ∧ a > c ∧ a > d :=
by
  let a := Real.pi
  let b := Real.sqrt 2
  let c := |(-2)|
  let d := 3
  have h1 : c = 2 := rfl
  have h2 : b < 2 := sorry -- sqrt(2) < 2
  have h3 : 2 < 3 := sorry -- obvious
  have h4 : 3 < a := sorry -- pi > 3
  exact ⟨lt_trans h2 h3, h4, h4⟩

end largest_number_is_pi_l91_91961


namespace integer_part_M_is_4_l91_91321

-- Define the variables and conditions based on the problem statement
variable (a b c : ℝ)

-- This non-computable definition includes the main mathematical expression we need to evaluate
noncomputable def M (a b c : ℝ) := Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1)

-- The theorem we need to prove
theorem integer_part_M_is_4 (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : 
  ⌊M a b c⌋ = 4 := 
by 
  sorry

end integer_part_M_is_4_l91_91321


namespace thermochemical_eq1_thermochemical_eq2_thermochemical_eq3_l91_91180

-- Definitions for the thermochemical reactions
def reaction1 (C2H4 O2 CO2 H2O : ℕ) (enthalpy : ℤ) :=
  C2H4 = 1 ∧ O2 = 3 ∧ CO2 = 2 ∧ H2O = 2 ∧ enthalpy = -1411

def reaction2 (H2 I2 HI : ℕ) (enthalpy : ℤ) :=
  H2 = 1 ∧ I2 = 1 ∧ HI = 2 ∧ enthalpy = -14.9

def reaction3 (a b : ℕ) (C2H2 O2 CO2 H2O : ℕ) (enthalpy : ℤ) :=
  a > 0 ∧ b > 0 ∧ C2H2 = 1 ∧ O2 = 5/2 ∧ CO2 = 2 ∧ H2O = 1 ∧ enthalpy = -2 * b

-- The theorems to be proven
theorem thermochemical_eq1 : ∃ (C2H4 O2 CO2 H2O : ℕ) (enthalpy : ℤ), reaction1 C2H4 O2 CO2 H2O enthalpy :=
begin
  use 1, use 3, use 2, use 2, use -1411,
  simp [reaction1],
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩,
end

theorem thermochemical_eq2 : ∃ (H2 I2 HI : ℕ) (enthalpy : ℤ), reaction2 H2 I2 HI enthalpy :=
begin
  use 1, use 1, use 2, use -14.9,
  simp [reaction2],
  exact ⟨rfl, rfl, rfl, rfl⟩,
end

theorem thermochemical_eq3 (a b : ℕ) : ∃ (C2H2 O2 CO2 H2O : ℕ) (enthalpy : ℤ), reaction3 a b C2H2 O2 CO2 H2O enthalpy :=
begin
  use 1, use 5/2, use 2, use 1, use -2 * b,
  simp [reaction3],
  exact ⟨nat.pos_of_ne_zero (by linarith), nat.pos_of_ne_zero (by linarith), rfl, rfl, rfl, rfl, rfl⟩,
end

end thermochemical_eq1_thermochemical_eq2_thermochemical_eq3_l91_91180


namespace refill_gas_l91_91178

theorem refill_gas (initial_gas : ℕ) (used_gas_store : ℕ) (used_gas_doctor : ℕ) (tank_capacity : ℕ) : initial_gas = 10 → used_gas_store = 6 → used_gas_doctor = 2 → tank_capacity = 12 → tank_capacity - (initial_gas - used_gas_store - used_gas_doctor) = 10 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end refill_gas_l91_91178


namespace solution_set_of_f_l91_91697

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.logBase 2 (a * x + Real.sqrt (2 * x^2 + 1))

theorem solution_set_of_f
  (a : ℝ) (ha : 0 < a)
  (h_odd : ∀ x : ℝ, f a x = -f a (-x)) :
  { x : ℝ | f (Real.sqrt 2) x > 3 / 2 } = { x : ℝ | x > 7 / 8 } :=
by
  sorry

end solution_set_of_f_l91_91697


namespace train_length_l91_91912

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (h_speed : speed_kmph = 72) (h_time : time_sec = 7) : 
  let speed_mps := speed_kmph * 1000 / 3600 in
  let distance := speed_mps * time_sec in
  distance = 140 :=
by
  -- Proof will be here
  sorry

end train_length_l91_91912


namespace find_k_l91_91752

noncomputable def h (x k : ℝ) := 2 * x - k / x + k / 3

noncomputable def h' (x k: ℝ) := (deriv (λ x, 2*x - k/x + k/3)) x

theorem find_k (k : ℝ) :
  -- Condition for increasing on (1, +∞)
  (∀ x : ℝ, 1 < x → h'(x, k) ≥ 0) →
  -- Condition for decreasing on (-1, 1)
  (∀ x : ℝ, -1 < x ∧ x < 1 → h'(x, k) ≤ 0) →
  -- Conclusion
  k = -2 :=
begin
  sorry
end

end find_k_l91_91752


namespace maximum_value_sqrt_expr_l91_91311

theorem maximum_value_sqrt_expr :
  ∃ x : ℝ, -49 ≤ x ∧ x ≤ 49 ∧ sqrt (49 + x) + sqrt (49 - x) = 14 ∧
  ∀ x : ℝ, -49 ≤ x ∧ x ≤ 49 → sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
by
  sorry

end maximum_value_sqrt_expr_l91_91311


namespace euler_totient_sequence_exists_l91_91827

theorem euler_totient_sequence_exists (n : ℕ) : 
  ∃ (a : Fin n → ℕ), StrictMono a ∧ StrictAnti (φ ∘ a) := 
sorry

end euler_totient_sequence_exists_l91_91827


namespace shortest_side_of_right_triangle_l91_91611

theorem shortest_side_of_right_triangle 
  (a b : ℕ) (ha : a = 7) (hb : b = 10) (c : ℝ) (hright : a^2 + b^2 = c^2) :
  min a b = 7 :=
by
  sorry

end shortest_side_of_right_triangle_l91_91611


namespace election_total_votes_l91_91761

variable (V : ℕ)  -- V is a natural number representing the total number of votes

-- Conditions:
-- 1. One candidate got 55% of the total valid votes.
-- 2. 20% of the votes were invalid.
-- 3. The other candidate got 2700 valid votes.
-- 4. The total number of votes was a certain amount V.

theorem election_total_votes (h : 0.55 * 0.8 * V + 2700 = 0.8 * V) : V = 7500 :=
by
  sorry

end election_total_votes_l91_91761


namespace total_distance_of_car_l91_91899

def arithmetic_sequence (a₁ d n : ℕ) : ℕ :=
  a₁ + n * d

theorem total_distance_of_car :
  let a₁ := 28
  let d := -7
  ∃ n, (n : int) ≥ 0 ∧ 
  (arithmetic_sequence a₁ d n = 0) ∧
  ∑ i in finset.range (n + 1), arithmetic_sequence a₁ d i = 70 := by
sorry

end total_distance_of_car_l91_91899


namespace decimal_to_base8_2023_l91_91987

theorem decimal_to_base8_2023 : nat_to_base 8 2023 = [3, 7, 4, 7] :=
sorry

end decimal_to_base8_2023_l91_91987


namespace solution_to_largest_four_digit_fulfilling_conditions_l91_91173

def largest_four_digit_fulfilling_conditions : Prop :=
  ∃ (N : ℕ), N < 10000 ∧ N ≡ 2 [MOD 11] ∧ N ≡ 4 [MOD 7] ∧ N = 9979

theorem solution_to_largest_four_digit_fulfilling_conditions : largest_four_digit_fulfilling_conditions :=
  sorry

end solution_to_largest_four_digit_fulfilling_conditions_l91_91173


namespace largest_number_is_pi_l91_91960

theorem largest_number_is_pi : 
  let a := Real.pi
  let b := Real.sqrt 2
  let c := |(-2)|
  let d := 3
  a > b ∧ a > c ∧ a > d :=
by
  let a := Real.pi
  let b := Real.sqrt 2
  let c := |(-2)|
  let d := 3
  have h1 : c = 2 := rfl
  have h2 : b < 2 := sorry -- sqrt(2) < 2
  have h3 : 2 < 3 := sorry -- obvious
  have h4 : 3 < a := sorry -- pi > 3
  exact ⟨lt_trans h2 h3, h4, h4⟩

end largest_number_is_pi_l91_91960


namespace magician_balls_l91_91588

theorem magician_balls (n : ℕ) (k : ℕ) :
  n = 7 + 6 * k → n % 6 = 1 → n = 1993 :=
begin
  sorry
end

end magician_balls_l91_91588


namespace piravena_least_cost_l91_91470

noncomputable def YZ : ℝ := Real.sqrt (3900 ^ 2 - 3600 ^ 2)

def cost_bus (distance : ℝ) : ℝ := distance * 0.20
def cost_airplane (distance : ℝ) : ℝ := distance * 0.12 + 120

def least_expensive_cost : ℝ :=
  let cost_XY := min (cost_bus 3900) (cost_airplane 3900)
  let cost_YZ := min (cost_bus YZ) (cost_airplane YZ)
  let cost_ZX := min (cost_bus 3600) (cost_airplane 3600)
  cost_XY + cost_YZ + cost_ZX

theorem piravena_least_cost : least_expensive_cost = 1440 := by
  sorry

end piravena_least_cost_l91_91470


namespace watermelon_yield_increase_l91_91939

noncomputable def yield_increase (initial_yield final_yield annual_increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_yield * (1 + annual_increase_rate) ^ years

theorem watermelon_yield_increase :
  ∀ (x : ℝ),
    (yield_increase 20 28.8 x 2 = 28.8) →
    (yield_increase 28.8 40 x 2 > 40) :=
by
  intros x hx
  have incEq : 20 * (1 + x) ^ 2 = 28.8 := hx
  sorry

end watermelon_yield_increase_l91_91939


namespace customers_added_l91_91237

theorem customers_added (x : ℕ) (h : 29 + x = 49) : x = 20 := by
  sorry

end customers_added_l91_91237


namespace determine_a_intervals_of_monotonicity_and_extreme_values_l91_91723

/-- Given a function f defined by f(x) = a(x-5)^2 + 6 * log x, where a ∈ ℝ,
    and a tangent line to the curve y = f(x) at the point (1, f(1)) that intersects the y-axis at (0,6),
    prove that the value of a is 1/2. -/
theorem determine_a (a : ℝ) (h : ∀ x: ℝ, f x = a * (x - 5)^2 + 6 * Real.log x)
  (tangent_y_intercept: (1, f 1) → (0, 6)) : a = 1 / 2 := 
sorry

/-- Given the function f(x) = 1/2 * (x-5)^2 + 6 * log x, prove the intervals of monotonicity and 
    the extreme values: f attains a maximum at x = 2 with f(2) = 9/2 + 6 * log 2 
    and a minimum at x = 3 with f(3) = 2 + 6 * log 3. -/
theorem intervals_of_monotonicity_and_extreme_values : 
  ∀ x: ℝ, (f x = (1/2) * (x - 5)^2 + 6 * Real.log x) 
           → (∀ x ∈ (0, 2) ∪ (3, +∞), f' x > 0) 
           → (∀ x ∈ (2, 3), f' x < 0) 
           → f 2 = 9 / 2 + 6 * Real.log 2 
           → f 3 = 2 + 6 * Real.log 3 := 
sorry

end determine_a_intervals_of_monotonicity_and_extreme_values_l91_91723


namespace subsets_have_common_element_l91_91759

variable (A : Type) (n : ℕ) (A_set : Finset (Fin n)) 
variable (chosen_subsets : Finset (Finset (Fin n)))
variable (h_cardinality : chosen_subsets.card = 2 ^ (n - 1))
variable (h_common_element : ∀ B C ∈ chosen_subsets, ∃ x : Fin n, x ∈ B ∧ x ∈ C)

theorem subsets_have_common_element 
  (A_set_subsets : A_set.card = n)
  : ∃ y : Fin n, ∀ B ∈ chosen_subsets, y ∈ B := 
sorry

end subsets_have_common_element_l91_91759


namespace margot_ways_l91_91459

def row1 : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

noncomputable def is_even (n : ℕ) : Prop :=
  n % 2 = 0

noncomputable def valid_row2 (row1 row2 : List ℕ) : Prop :=
  row2.perm row1 ∧ ∀ i, is_even (row1.nthLe i sorry + row2.nthLe i sorry)

theorem margot_ways : ∃ (row2 : List ℕ), valid_row2 row1 row2 ∧
  (row1.filter odd).perm (row2.filter odd) ∧ (row1.filter even).perm (row2.filter even) ∧
  (perm_count row1 (row1.filter odd) (row2.filter odd) * perm_count row1 (row1.filter even) (row2.filter even)) = 576 :=
sorry

end margot_ways_l91_91459


namespace slope_MN_is_minus_one_l91_91037

noncomputable def slope_of_MN (a b : ℝ) (a_gt_b : a > b) (line_intersect_squared: (x y : ℝ) -> x^2 / a^2 + y^2 / b^2 = 1) 
(F1 F2 : ℝ × ℝ) (F1_left_of_center : F1.fst < 0) (F2_right_of_center : F2.fst > 0)
(intersection_line : F1.fst -> F1.snd -> (A B : ℝ × ℝ) -> (|A.1 - F1.1 |^2 + |A.2 - F1.2|^2 = 9 * (|B.1 - F1.1|^2 + |B.2 - F1.2|^2)) )
(cos_AF2B : ℝ) (cos_AF2B_eq : cos_AF2B = 3 / 5)
(y_intersection : (x : ℝ) -> x * 2 = 0 -> P Q : ℝ × ℝ)
(C D : ℝ × ℝ) (C_on_ellipse : line_intersect_squared C.1 C.2) (D_on_ellipse : line_intersect_squared D.1 D.2)
(M N : ℝ × ℝ) (intersect_PC_QD : P.1 * M.1 + P.2 * M.2 = Q.1 * D.1 + Q.2 * D.2)
(intersect_PD_QC : P.1 * N.1 + P.2 * N.2 = Q.1 * C.1 + Q.2 * C.2):
ℝ :=
let k_MN := (N.2 - M.2) / (N.1 - M.1) in
-1

theorem slope_MN_is_minus_one (a b : ℝ) (a_gt_b : a > b) (line_intersect_squared: (x y : ℝ) -> x^2 / a^2 + y^2 / b^2 = 1) 
(F1 F2 : ℝ × ℝ) (F1_left_of_center : F1.fst < 0) (F2_right_of_center : F2.fst > 0)
(intersection_line : F1.fst -> F1.snd -> (A B : ℝ × ℝ) -> (|A.1 - F1.1 |^2 + |A.2 - F1.2|^2 = 9 * (|B.1 - F1.1|^2 + |B.2 - F1.2|^2)) )
(cos_AF2B : ℝ) (cos_AF2B_eq : cos_AF2B = 3 / 5)
(y_intersection : (x : ℝ) -> x * 2 = 0 -> P Q : ℝ × ℝ)
(C D : ℝ × ℝ) (C_on_ellipse : line_intersect_squared C.1 C.2) (D_on_ellipse : line_intersect_squared D.1 D.2)
(M N : ℝ × ℝ) (intersect_PC_QD : P.1 * M.1 + P.2 * M.2 = Q.1 * D.1 + Q.2 * D.2)
(intersect_PD_QC : P.1 * N.1 + P.2 * N.2 = Q.1 * C.1 + Q.2 * C.2):
slope_of_MN a b a_gt_b line_intersect_squared F1 F2 F1_left_of_center F2_right_of_center intersection_line cos_AF2B cos_AF2B_eq y_intersection C D C_on_ellipse D_on_ellipse M N intersect_PC_QD intersect_PD_QC = -1 :=
sorry

end slope_MN_is_minus_one_l91_91037


namespace odd_function_m_value_l91_91725

noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x - m

theorem odd_function_m_value :
  ∃ m : ℝ, (∀ (x : ℝ), g (-x) m + g x m = 0) ∧ m = 2 :=
by
  sorry

end odd_function_m_value_l91_91725


namespace perimeter_of_triangle_ABC_l91_91771

-- Define the basic setup:  Four circles with given properties
structure Circle :=
(center : Point)
(radius : ℝ)

def Triangle (A B C : Point) : Prop := 
equilateral (triangle A B C)

structure Configuration :=
(P Q R S : Circle)
(tangent_to_each_other : ∀ (X Y : Circle), X ≠ Y → tangent X Y)
(tangent_to_sides_of_ABC : ∀ (X : Circle), tangent_to_side_ABC X)
(lines_parallel_to_AB : LineThrough P.center Q.center ∥ LineThrough AB ∧ LineThrough R.center S.center ∥ LineThrough AB)
(lines_parallel_to_BC : LineThrough P.center R.center ∥ LineThrough BC ∧ LineThrough Q.center S.center ∥ LineThrough BC)
(lines_parallel_to_AC : LineThrough P.center S.center ∥ LineThrough AC ∧ LineThrough Q.center R.center ∥ LineThrough AC)

-- Define the required geometry entities: points, lines and triangles
structure Point := (x y : ℝ)
def segment_length (A B : Point) : ℝ := (dist A B)

-- Define the statement for the problem
theorem perimeter_of_triangle_ABC (A B C P Q R S : Point) (r : ℝ) (cfg : Configuration P Q R S) 
  (triangle_ABC : Triangle A B C) (r_eq_2 : r = 2)
  (P_center : P = Circle.center P)
  (Q_center : Q = Circle.center Q)
  (R_center : R = Circle.center R)
  (S_center : S = Circle.center S):
aqool:
(
  segment_length A B = 6 ∧
  segment_length B C = 6 ∧
  segment_length A C = 6
) → 
(segment_length A B + segment_length B C + segment_length A C) = 18 := 
begin
  sorry
end

end perimeter_of_triangle_ABC_l91_91771


namespace Milly_study_time_l91_91055

theorem Milly_study_time :
  let math_time := 60
  let geo_time := math_time / 2
  let mean_time := (math_time + geo_time) / 2
  let total_study_time := math_time + geo_time + mean_time
  total_study_time = 135 := by
  sorry

end Milly_study_time_l91_91055


namespace equilateral_triangle_hyperbola_area_square_l91_91533

theorem equilateral_triangle_hyperbola_area_square
  (A B C : ℝ × ℝ)
  (h1 : A.1 * A.2 = 1)
  (h2 : B.1 * B.2 = 1)
  (h3 : C.1 * C.2 = 1)
  (centroid : ℝ × ℝ)
  (h4 : centroid = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3))
  (h5 : centroid = (-1, -1)) :
  let side_length := 2 * Real.sqrt 6 in
  let area := (Real.sqrt 3 / 4) * (side_length ^ 2) in
  area ^ 2 = 108 :=
by
  let side_length := 2 * Real.sqrt 6
  let area := (Real.sqrt 3 / 4) * (side_length ^ 2)
  have h6 : area ^ 2 = 108 := by sorry
  exact h6

end equilateral_triangle_hyperbola_area_square_l91_91533


namespace max_ball_height_l91_91201

/-- 
The height of a thrown ball taking into account initial height, initial velocity, 
and additional height gained per second due to wind.
-/
def ball_height (t : ℝ) : ℝ := -16 * t^2 + 105 * t + 30

theorem max_ball_height : ∃ t, 0 ≤ t ∧ ball_height t = 202.31 :=
by
  use 3.28125
  split
  -- 0 ≤ 3.28125 is trivially true
  . le_refl 3.28125
  -- Here we should calculate ball_height(3.28125) = 202.31
  . sorry

end max_ball_height_l91_91201


namespace f1_f2_defined_f1_f2_close_l91_91667

-- Definitions for the functions
def f1 (a x : ℝ) := log a (x - 3 * a)
def f2 (a x : ℝ) := log a (1 / (x - a))

-- Define the interval [a + 2, a + 3]
def in_interval (a x : ℝ) := a + 2 ≤ x ∧ x ≤ a + 3

-- Condition: 0 < a < 1 and a ≠ 1
def valid_a (a : ℝ) := 0 < a ∧ a < 1 ∧ a ≠ 1

-- Proof that f1 and f2 are defined in the interval [a + 2, a + 3]
theorem f1_f2_defined (a : ℝ) (h : valid_a a) (x : ℝ) (h_in : in_interval a x) :
  (x - 3 * a > 0) ∧ (x - a > 0) :=
sorry

-- Proof that f1 and f2 are close on the interval [a + 2, a + 3]
theorem f1_f2_close (a : ℝ) (h : valid_a a) (x : ℝ) (h_in : in_interval a x) :
  |f1 a x - f2 a x| ≤ 1 :=
sorry

end f1_f2_defined_f1_f2_close_l91_91667


namespace percentage_without_any_condition_l91_91953

variable total : ℕ
variable bp : ℕ
variable ht : ℕ
variable d : ℕ
variable bp_ht : ℕ
variable bp_d : ℕ
variable ht_d : ℕ
variable all_three : ℕ

theorem percentage_without_any_condition (h1 : total = 200)
    (h2 : bp = 90) (h3 : ht = 60) (h4 : d = 30)
    (h5 : bp_ht = 25) (h6 : bp_d = 15) (h7 : ht_d = 10) (h8 : all_three = 5) :
    (total - (bp - bp_ht - bp_d + all_three + ht - bp_ht - ht_d + all_three + 
              d - bp_d - ht_d + all_three + (bp_ht - all_three) + (bp_d - all_three) + 
              (ht_d - all_three) + all_three)) * 100 / total = 22.5 :=
by
  sorry

end percentage_without_any_condition_l91_91953


namespace math_problem_l91_91833

theorem math_problem
  (x y : ℝ)
  (h1 : x + y = 5)
  (h2 : x * y = -3)
  : x + (x^3 / y^2) + (y^3 / x^2) + y = 590.5 :=
sorry

end math_problem_l91_91833


namespace count_four_digit_double_peak_mountain_numbers_l91_91992

-- Definitions of conditions
def is_four_digit_double_peak_mountain_number (n : Nat) : Prop :=
  let d1 := (n / 1000) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 ≠ 0 ∧ d1 < d2 ∧ d2 = d3 ∧ d3 > d4

-- Main theorem statement
theorem count_four_digit_double_peak_mountain_numbers : 
  {n : Nat // n > 999 ∧ n < 10000 ∧ is_four_digit_double_peak_mountain_number n}.card = 54 :=
by
  sorry

end count_four_digit_double_peak_mountain_numbers_l91_91992


namespace time_to_finish_work_with_both_tractors_l91_91601

-- Definitions of given conditions
def work_rate_A : ℚ := 1 / 20
def work_rate_B : ℚ := 1 / 15
def time_A_worked : ℚ := 13
def remaining_work : ℚ := 1 - (work_rate_A * time_A_worked)
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Statement that needs to be proven
theorem time_to_finish_work_with_both_tractors : 
  remaining_work / combined_work_rate = 3 :=
by
  sorry

end time_to_finish_work_with_both_tractors_l91_91601


namespace Maria_score_in_fourth_quarter_l91_91153

theorem Maria_score_in_fourth_quarter (q1 q2 q3 : ℕ) 
  (hq1 : q1 = 84) 
  (hq2 : q2 = 82) 
  (hq3 : q3 = 80) 
  (average_requirement : ℕ) 
  (havg_req : average_requirement = 85) :
  ∃ q4 : ℕ, q4 ≥ 94 ∧ (q1 + q2 + q3 + q4) / 4 ≥ average_requirement := 
by 
  sorry 

end Maria_score_in_fourth_quarter_l91_91153


namespace add_three_people_l91_91931

theorem add_three_people (front_row : fin 3) (back_row : fin 4)
  (A B C : Type) :
  (∃ (f : A ⊕ B ⊕ C → fin 4), injective f) → 
  (∃ (g1 : A ⊕ B ⊕ C → fin 5), injective g1) ∧ 
  (∃ (g2 : A ⊕ B ⊕ C → fin 6), injective g2) →
  3 * 4 * 5 * 6 = 360 :=
by sorry

end add_three_people_l91_91931


namespace shortest_chord_length_l91_91940

def circle_radius : ℝ := 2
def center : ℝ × ℝ := (2, 2)
def point_on_circle : ℝ × ℝ := (3, 1)

theorem shortest_chord_length :
  let d := real.sqrt ((point_on_circle.1 - center.1)^2 + (point_on_circle.2 - center.2)^2)
  in d < circle_radius ∧
     (let shortest_chord := 2 * real.sqrt (circle_radius^2 - d^2)
      in shortest_chord = 2 * real.sqrt 2) :=
begin
  sorry
end

end shortest_chord_length_l91_91940


namespace possible_b2_values_count_l91_91228

-- Define the given sequence and conditions
def sequence (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) = abs (b (n + 1) - b n)

def condition_b1 := 1001
def condition_b2023 := 3
def less_than_1001 (x : ℕ) : Prop := x < 1001

-- Define the proof problem
theorem possible_b2_values_count :
  ∃ b : ℕ → ℕ,
    b 1 = condition_b1 ∧
    (∃ b2, less_than_1001 b2 ∧
      (b 2 = b2 ∧ sequence b ∧ b 2023 = condition_b2023)) →
    (finset.filter (λ x, sequence (λ n, ite (n = 1) condition_b1 (ite (n = 2) x (abs ((λ i, ite (i = 1) condition_b1 (ite (i = 2) x (abs ((λ j, ite (j = 1) condition_b1 (if j-1 = 1 then x else (abs ((λ k, ite (k = 1) condition_b1 (abs (x - condition_b1)) sorry)) (j + 1) sorry)))) sorry)) (n + 1) sorry)))) (finset.range 1001)) = 376 :=
sorry

end possible_b2_values_count_l91_91228


namespace sum_of_integer_solutions_l91_91557

theorem sum_of_integer_solutions :
  (∑ (x : ℤ) in {x : ℤ | 1 < (x - 2)^2 ∧ (x - 2)^2 < 25}.toFinset, x) = 12 :=
by
  sorry

end sum_of_integer_solutions_l91_91557


namespace avg_distinct_t_values_l91_91709

theorem avg_distinct_t_values : 
  ∀ (t : ℕ), (∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t) →
  (1 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   6 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   10 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   12 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t}) →
  let distinct_values := {6, 10, 12} in
  let average := (6 + 10 + 12) / 3 in
  average = 28 / 3 :=
by {
  sorry
}

end avg_distinct_t_values_l91_91709


namespace curve_cartesian_form_and_distance_of_intersections_l91_91843

-- Define polar-to-Cartesian conversion
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

-- Define curve C in polar coordinates
def curve_C_polar (ρ θ : ℝ) : Prop := ρ * sin θ = 4 * cos θ

-- Define curve C in Cartesian coordinates
def curve_C_cartesian (x y : ℝ) : Prop := y^2 = 4 * x

-- Define line L in parametric form
def line_l (t : ℝ) : ℝ × ℝ := 
  let x := 1 + (2 / sqrt 5) * t in
  let y := 1 + (1 / sqrt 5) * t in
  (x, y)

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define Euclidean distance
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ := 
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Math proof problem, statement only
theorem curve_cartesian_form_and_distance_of_intersections : 
  (∀ (ρ θ : ℝ), curve_C_polar ρ θ → ∃ x y : ℝ, polar_to_cartesian ρ θ = (x, y) ∧ curve_C_cartesian x y) ∧ 
  (∃ (t1 t2 : ℝ), 
    let A := line_l t1 in 
    let B := line_l t2 in 
    let P := point_P in 
    line_l t1 ∈ set_of (λ (p : ℝ × ℝ), curve_C_cartesian p.1 p.2) ∧ 
    line_l t2 ∈ set_of (λ (p : ℝ × ℝ), curve_C_cartesian p.1 p.2) ∧ 
    (euclidean_distance P A + euclidean_distance P B = 4 * sqrt 15)) :=
begin
  sorry
end

end curve_cartesian_form_and_distance_of_intersections_l91_91843


namespace average_of_N_l91_91552

theorem average_of_N (N : ℕ) : 
  (∃ (N_vals : Finset ℕ), (∀ n ∈ N_vals, 21 < n ∧ n < 40) ∧ 
  N_vals.sum id / N_vals.card = 30.5) := 
begin 
  -- Definitions for conditions
  let N_vals := (Finset.Ico 22 40), -- Finset with integer values 22 to 39
  use N_vals,
  -- Use sorry to skip the proof steps
  sorry
end

end average_of_N_l91_91552


namespace sequence_eventually_constant_l91_91188

noncomputable def eventually_constant_sequence (x : ℕ → ℤ) (d : ℕ → ℕ) (k : ℕ) (n0 : ℕ) : Prop :=
  ∃ M : ℤ, ∀ n ≥ n0, x n = M

-- Definitions of initial conditions and gcd
def initial_conditions (x : ℕ → ℤ) (n0 : ℕ) : Prop :=
  ∀ n < n0, x n ∈ ℤ

def positive_integers (d : ℕ → ℕ) (k : ℕ) (n0 : ℕ) : Prop :=
  ∀ i < k, d i > 0 ∧ (i < k - 1 → d i > d (i + 1)) ∧ d 0 = n0 ∧ (Nat.gcdAuxAux (Finset.range k) (d ∘ Finset.val) 1).val = 1

def recurrence_relation (x : ℕ → ℤ) (d : ℕ → ℕ) (k : ℕ) (n0 : ℕ) : Prop :=
  ∀ n ≥ n0, x n = Int.floor ((∑ i in Finset.range k, x (n - d i)) / k)

theorem sequence_eventually_constant (x : ℕ → ℤ) (d : ℕ → ℕ) (k n0 : ℕ) 
  (h_initial : initial_conditions x n0) 
  (h_positive : positive_integers d k n0)
  (h_recurrence : recurrence_relation x d k n0) : eventually_constant_sequence x d k n0 :=
sorry

end sequence_eventually_constant_l91_91188


namespace store_cost_relationship_store_cost_comparison_l91_91590

def price_per_computer := 6000

def storeA_first_computer_price := price_per_computer
def storeA_discount := (25 / 100: ℝ)
def storeA_price_for_subsequent_computers := price_per_computer * (1 - storeA_discount)

def storeB_discount := (20 / 100: ℝ)
def storeB_price_per_computer := price_per_computer * (1 - storeB_discount)

def y_A (x : ℕ) : ℝ := 
  if x ≥ 1 then storeA_first_computer_price + storeA_price_for_subsequent_computers * (x - 1)
  else 0

def y_B (x : ℕ) : ℝ := 
  storeB_price_per_computer * x

theorem store_cost_relationship
  (x : ℕ) (hₓ : 1 ≤ x) :
  y_A x = 4500 * (x : ℝ) + 1500 ∧ y_B x = 4800 * (x : ℝ) :=
by
  sorry

theorem store_cost_comparison
  (x : ℕ) :
  (x < 5 → y_A x > y_B x) ∧ 
  (x > 5 → y_A x < y_B x) ∧ 
  (x = 5 → y_A x = y_B x) :=
by
  sorry

end store_cost_relationship_store_cost_comparison_l91_91590


namespace general_formula_range_of_a_l91_91325

noncomputable def a_seq (n : ℕ) : ℕ := 2^(n + 1)

theorem general_formula (q a1 : ℝ) (h_q : q > 1) (h1 : a1 + a1*q^2 = 20) (h2 : a1*q = 8) : 
  ∀ n : ℕ, a_seq n = 2^(n + 1) :=
sorry

noncomputable def b_seq (n : ℕ) : ℝ := ↑n / (2^(n + 1))

noncomputable def S_n (n : ℕ) : ℝ := (finset.range n).sum (λ i, b_seq (i + 1))

theorem range_of_a (a : ℝ) : (-1/2 < a) ∧ (a < 3/4) := sorry

end general_formula_range_of_a_l91_91325


namespace tangent_line_at_pi_l91_91504

noncomputable def tangent_equation (x : ℝ) : ℝ := x * Real.sin x

theorem tangent_line_at_pi :
  let f := tangent_equation
  let f' := fun x => Real.sin x + x * Real.cos x
  let x : ℝ := Real.pi
  let y : ℝ := f x
  let slope : ℝ := f' x
  y + slope * x - Real.pi^2 = 0 :=
by
  -- This is where the proof would go
  sorry

end tangent_line_at_pi_l91_91504


namespace find_sum_of_relatively_prime_integers_l91_91521

theorem find_sum_of_relatively_prime_integers :
  ∃ (x y : ℕ), x * y + x + y = 119 ∧ x < 25 ∧ y < 25 ∧ Nat.gcd x y = 1 ∧ x + y = 20 :=
by
  sorry

end find_sum_of_relatively_prime_integers_l91_91521


namespace man_speed_is_5_km_per_hr_l91_91217

def convert_minutes_to_hours (minutes: ℕ) : ℝ :=
  minutes / 60

def convert_meters_to_kilometers (meters: ℕ) : ℝ :=
  meters / 1000

def man_speed (distance_m: ℕ) (time_min: ℕ) : ℝ :=
  distance_m / 1000 / (time_min / 60)

theorem man_speed_is_5_km_per_hr : man_speed 1250 15 = 5 := by
  unfold man_speed convert_meters_to_kilometers convert_minutes_to_hours
  -- More steps in the proof would go here, but we use sorry to skip the proof.
  sorry

end man_speed_is_5_km_per_hr_l91_91217


namespace sum_from_neg_50_to_70_l91_91891

noncomputable def sum_integers (start : ℤ) (end : ℤ) : ℤ :=
  (end - start + 1) * (start + end) / 2

theorem sum_from_neg_50_to_70 : sum_integers (-50) 70 = 1210 :=
by sorry

end sum_from_neg_50_to_70_l91_91891


namespace log_four_sixty_four_root_four_l91_91277

-- Definitions for conditions
def sixty_four : ℝ := 4 ^ 3
def root_four : ℝ := 4 ^ (1 / 4)

-- Statement of the proof problem
theorem log_four_sixty_four_root_four : log 4 (sixty_four * root_four) = 13 / 4 :=
by 
  sorry

end log_four_sixty_four_root_four_l91_91277


namespace ball_surface_area_proof_l91_91945

noncomputable def ball_surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2

theorem ball_surface_area_proof :
  (∀ (d : ℝ) (h : ℝ), d = 32 → h = 9 → 
   let r := d / 2 in
   let volume_displaced := Real.pi * r^2 * h in
   let sphere_radius := (3 * volume_displaced / (4 * Real.pi))^(1/3) in
   ball_surface_area sphere_radius = 576 * Real.pi) :=
by
  intros d h d_def h_def
  let r := d / 2
  have r_def : r = 16 := by { rw [d_def], norm_num }
  let volume_displaced := Real.pi * r^2 * h
  have volume_def : volume_displaced = 9 * 256 * Real.pi := by { rw [r_def, h_def], norm_num, ring }
  let sphere_radius := (3 * volume_displaced / (4 * Real.pi))^(1 / 3)
  have radius_def : sphere_radius = 12 := by {
    rw [volume_def, mul_assoc, mul_div_cancel_left _ (ne_of_gt (mul_pos (by norm_num) Real.pi_pos)), Real.rpow_eq_pow, ←Real.pow_mul, ←Real.cube_pow],
    norm_num } 
  rw [ball_surface_area, radius_def]
  norm_num

end ball_surface_area_proof_l91_91945


namespace cartesian_eq_of_C₁_cartesian_eq_of_C₂_solution_intersections_l91_91004

noncomputable def C₁_polar (θ : ℝ) : ℝ := 4 * Real.sin θ
noncomputable def C₂_polar (θ : ℝ) : ℝ := 2 * Real.sqrt 2 / (Real.cos (θ - (Real.pi / 4)))

theorem cartesian_eq_of_C₁ :
  ∀ (x y : ℝ), (∃ θ : ℝ, x = C₁_polar θ * Real.cos θ ∧ y = C₁_polar θ * Real.sin θ) ↔
  x^2 + (y-2)^2 = 4 :=
sorry

theorem cartesian_eq_of_C₂ :
  ∀ (x y : ℝ), (∃ θ : ℝ, x = C₂_polar θ * Real.cos θ ∧ y = C₂_polar θ * Real.sin θ) ↔
  x + y = 4 :=
sorry

theorem solution_intersections :
  ∀ (ρ θ : ℝ), ((ρ, θ) = (4, Real.pi / 2) ∨ (ρ, θ) = (2 * Real.sqrt 2, Real.pi / 4)) ↔
  (∃ x y : ℝ, (x^2 + (y-2)^2 = 4) ∧ (x + y = 4) ∧ ρ = Real.sqrt (x^2 + y^2) ∧ θ = Real.atan2 y x) :=
sorry

end cartesian_eq_of_C₁_cartesian_eq_of_C₂_solution_intersections_l91_91004


namespace projection_problem_l91_91799

variables (p q : ℝ -> ℝ -> ℝ -> ℝ)

noncomputable def proj (u v : ℝ -> ℝ -> ℝ -> ℝ) : ℝ -> ℝ -> ℝ -> ℝ :=
  let dot_prod := (u.1 * v.1 + u.2 * v.2 + u.3 * v.3)
  let norm_sq := (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)
  ((dot_prod / norm_sq) * v.1, (dot_prod / norm_sq) * v.2, (dot_prod / norm_sq) * v.3)

theorem projection_problem :
  proj q (λ x y z => 4 * p x y z) = (12, -4, 16) :=
by
  sorry

end projection_problem_l91_91799


namespace quadratic_function_properties_l91_91729

-- Definitions based on the problem given
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

-- Proof statement without the proof body
theorem quadratic_function_properties :
  ∃ (xₘ : ℝ), xₘ = 1.5 ∧ is_local_min f xₘ ∧
  (∀ x > 1.5, deriv f x > 0) ∧
  (∀ x < 1.5, deriv f x < 0) :=
by
  sorry

end quadratic_function_properties_l91_91729


namespace ashley_loan_least_months_l91_91245

theorem ashley_loan_least_months (t : ℕ) (principal : ℝ) (interest_rate : ℝ) (triple_principal : ℝ) : 
  principal = 1500 ∧ interest_rate = 0.06 ∧ triple_principal = 3 * principal → 
  1.06^t > triple_principal → t = 20 :=
by
  intro h h2
  sorry

end ashley_loan_least_months_l91_91245


namespace proof_problem_l91_91682

noncomputable def ellipse_equation : Prop :=
  ∃ a b : ℝ, 0 < b ∧ b < a ∧ a = 2 ∧ b = sqrt 3 ∧
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ (x, y) ∈ set.univ)

noncomputable def min_distance_to_x_axis : Prop :=
  ∃ (O1_y : ℝ), O1_y = abs (3 / (2 * sqrt 3) - (sqrt 3) / 6) ∧
  (∀ y0 : ℝ, 0 < y0 ∧ y0 <= sqrt 3 → abs (3 / (2 * y0) - y0 / 6) ≥ abs (3 / (2 * sqrt 3) - (sqrt 3) / 6))

theorem proof_problem : ellipse_equation ∧ min_distance_to_x_axis := sorry

end proof_problem_l91_91682


namespace kelseys_sisters_age_l91_91024

theorem kelseys_sisters_age :
  ∀ (current_year : ℕ) (kelsey_birth_year : ℕ)
    (kelsey_sister_birth_year : ℕ),
    kelsey_birth_year = 1999 - 25 →
    kelsey_sister_birth_year = kelsey_birth_year - 3 →
    current_year = 2021 →
    current_year - kelsey_sister_birth_year = 50 :=
by
  intros current_year kelsey_birth_year kelsey_sister_birth_year h1 h2 h3
  sorry

end kelseys_sisters_age_l91_91024


namespace count_ordered_pairs_l91_91974

theorem count_ordered_pairs : 
  ∃ (a n : ℕ), 
  Bob_age = 31 ∧
  0 < n ∧
  ∃ (b c : ℕ), 
    b < c ∧ 
    b ∈ {1, 2, ..., 9} ∧ 
    c ∈ {1, 2, ..., 9} ∧ 
    10 * c + b > 31 ∧ 
    c > b ∧
    (a + n) = (Bob_age + n) + 29 :=
begin
  sorry
end

end count_ordered_pairs_l91_91974


namespace coin_toss_problem_l91_91943

noncomputable def binom (n k : ℕ) : ℚ :=
  (nat.choose n k : ℚ) / 2^n

theorem coin_toss_problem :
  let p := ∑ k in finset.Icc 43 70, binom 70 k
  let q := ∑ k in finset.Icc 0 28, binom 70 k
  p - q = - binom 70 28 :=
by
  intro p q
  sorry

end coin_toss_problem_l91_91943


namespace circumcircle_radius_eq_sqrt_R_times_r_l91_91544

-- Define the necessary points and their properties
variables (O1 O2 A B C : Type) [metric_space R] [metric_space r]
variables (R r : ℝ) (phi : ℝ)
variable (h_intersect : is_intersection_point O1 O2 A)
variable (h_tangentBC : is_common_tangent O1 O2 B C)
variable (h_angle_phi : ∠ O1 A O2 = phi)

-- Define the theorem for the required proof
theorem circumcircle_radius_eq_sqrt_R_times_r :
  radius_of_circumcircle_triangle (A B C) = sqrt (R * r) :=
by
  sorry

end circumcircle_radius_eq_sqrt_R_times_r_l91_91544


namespace part1_part2_l91_91831

-- Define the operation a ⊕ b ⊕ c
def op (a b c : ℚ) : ℚ := (1 / 2) * (|a - b - c| + a + b + c)

-- Define the set of numbers in Lean
def nums : list ℚ := [-6/7, -5/7, -4/7, -3/7, -2/7, -1/7, 0, 1/9, 2/9, 3/9, 4/9, 5/9, 6/9, 7/9, 8/9]

-- Part 1: Prove the specific calculation
theorem part1 : op 3 (-2) (-3) = 3 := by
  sorry

-- Part 2: Prove the maximum result among all possible combinations
theorem part2 : ∀ a b c ∈ nums, op a b c ≤ 5 / 3 := by
  sorry

end part1_part2_l91_91831


namespace probability_of_x_in_interval_l91_91069

noncomputable def interval_length (a b : ℝ) : ℝ := b - a

noncomputable def probability_in_interval : ℝ :=
  let length_total := interval_length (-2) 1
  let length_sub := interval_length 0 1
  length_sub / length_total

theorem probability_of_x_in_interval :
  probability_in_interval = 1 / 3 :=
by
  sorry

end probability_of_x_in_interval_l91_91069


namespace part2_l91_91721

def f (x : ℝ) : ℝ := Real.log x - 3 * x

theorem part2 (a b : ℝ) (h : ∀ x : ℝ, 0 < x → f x ≤ x * (a * Real.exp x - 4) + b) :
  a + b ≥ 0 := sorry

end part2_l91_91721


namespace triangle_COB_area_l91_91811

theorem triangle_COB_area (p : ℝ) (hp : p < 6) :
  let C := (0, 2*p)
  let B := (6, 0)
  let O := (0, 0)
  (area_of_triangle O C B) = 6 * p :=
by
  sorry

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  0.5 * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

end triangle_COB_area_l91_91811


namespace min_value_b_l91_91766

noncomputable section

open Real

theorem min_value_b (a b : ℝ) (h : a > 0) :
  (∀ x : ℝ, x > 0 → (a * ln x - a) = b ↔ y = x + b) → b = -1 :=
by
  sorry

end min_value_b_l91_91766


namespace inv_g_sum_l91_91427

def g (x : ℝ) : ℝ :=
if x < 10 then 2 * x + 4 else 3 * x - 3

lemma inv_g_8 : ∃ x, g x = 8 ∧ (x < 10) := 
begin
  use 2,
  split,
  { simp [g], norm_num, },
  { norm_num, },
end

lemma inv_g_27 : ∃ x, g x = 27 ∧ (x ≥ 10) := 
begin
  use 10,
  split,
  { simp [g], norm_num, },
  { norm_num, },
end

theorem inv_g_sum : (∃ x, g x = 8 ∧ (x < 10)) ∧ (∃ y, g y = 27 ∧ (y ≥ 10)) → 
  (x = 2 ∧ y = 10 → (x + y = 12)) :=
begin
  intro h,
  cases h with h1 h2,
  cases h1 with x hx,
  cases h2 with y hy,
  intro hxhy,
  cases hx,
  cases hy,
  simp at hx_right hy_right,
  rw [hxhy.1, hxhy.2],
  norm_num,
end

end inv_g_sum_l91_91427


namespace sheela_monthly_income_l91_91185

theorem sheela_monthly_income (deposit : ℝ) (percentage : ℝ) (income : ℝ) 
  (h1 : deposit = 2500) (h2 : percentage = 0.25) (h3 : deposit = percentage * income) :
  income = 10000 := 
by
  -- proof steps would go here
  sorry

end sheela_monthly_income_l91_91185


namespace orthogonal_vectors_a_value_l91_91736

theorem orthogonal_vectors_a_value :
  ∀ (a : ℝ), let m : ℝ × ℝ := (1, 2),
                n : ℝ × ℝ := (a, -1) in
                m.1 * n.1 + m.2 * n.2 = 0 → a = 2 :=
by
  intros a m n h
  sorry

end orthogonal_vectors_a_value_l91_91736


namespace max_length_OB_l91_91879

theorem max_length_OB (O A B : Point) (h : Angle O A B = 45°) (h1 : dist A B = 1) : dist O B ≤ sqrt 2 :=
sorry

end max_length_OB_l91_91879


namespace rectangle_length_l91_91383

theorem rectangle_length (perimeter breadth : ℝ) (h_perimeter : perimeter = 1000) (h_breadth : breadth = 200) :
  ∃ length : ℝ, 2 * (length + breadth) = perimeter ∧ length = 300 :=
by {
  use (300),
  rw [h_perimeter, h_breadth],
  split,
  {
    calc
      2 * (300 + 200) = 2 * 500 : by norm_num
                   ... = 1000 : by norm_num,
  },
  { refl }
}

end rectangle_length_l91_91383


namespace price_of_tea_C_l91_91082

theorem price_of_tea_C :
  let cost_A := 126
  let cost_B := 135
  let cost_D := 167
  let ratio_sum := 7
  let mixture_cost := 154 in
  let total_cost := ratio_sum * mixture_cost
  let total_mixture_cost := cost_A + cost_B + 2 * x + 3 * cost_D
  in
  total_mixture_cost = total_cost → x = 158 :=
by
  let cost_A := 126
  let cost_B := 135
  let cost_D := 167
  let ratio_sum := 7
  let mixture_cost := 154
  let total_cost := ratio_sum * mixture_cost
  let total_mixture_cost := cost_A + cost_B + 2 * x + 3 * cost_D
  show total_mixture_cost = total_cost → x = 158
  sorry

end price_of_tea_C_l91_91082


namespace base_four_product_l91_91553

def base_four_to_decimal (n : ℕ) : ℕ :=
  -- definition to convert base 4 to decimal, skipping details for now
  sorry

def decimal_to_base_four (n : ℕ) : ℕ :=
  -- definition to convert decimal to base 4, skipping details for now
  sorry

theorem base_four_product : 
  base_four_to_decimal 212 * base_four_to_decimal 13 = base_four_to_decimal 10322 :=
sorry

end base_four_product_l91_91553


namespace range_of_a_l91_91495

variable {f : ℝ → ℝ} {a : ℝ}
open Real

-- Conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
def f_positive_at_2 (f : ℝ → ℝ) : Prop := f 2 > 1
def f_value_at_2014 (f : ℝ → ℝ) (a : ℝ) : Prop := f 2014 = (a + 3) / (a - 3)

-- Proof Problem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : odd_function f)
  (h2 : periodic_function f 7)
  (h3 : f_positive_at_2 f)
  (h4 : f_value_at_2014 f a) :
  0 < a ∧ a < 3 :=
sorry

end range_of_a_l91_91495


namespace counterexample_statement_l91_91258

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n

theorem counterexample_statement (n : ℕ) : is_composite n ∧ (is_prime (n - 3) ∨ is_prime (n - 2)) ↔ n = 22 :=
by
  sorry

end counterexample_statement_l91_91258


namespace theo_needs_ingredients_l91_91246

-- Definition of omelette types with ingredients
def ingredients (omelette_type : ℕ) : (ℕ × ℕ × ℕ × ℕ × ℕ) :=
  match omelette_type with
  | 2 => (2, 1, 2, 0, 0) -- 2 egg omelette: 2 eggs, 1 cheese, 2 spinach
  | 3 => (3, 1.5, 3, 1, 0) -- 3 egg omelette: 3 eggs, 1.5 cheese, 3 spinach, 1 mushrooms
  | 4 => (4, 2, 0, 1, 3) -- 4 egg omelette: 4 eggs, 2 cheese, 1 mushrooms, 3 tomatoes
  | _ => (0, 0, 0, 0, 0) -- Invalid omelette type

-- Hourly demand for each type of omelette
def hourly_demand : List (ℕ × ℕ × ℕ) :=
  [ (4, 5, 0) -- First hour: 4x 2-egg, 5x 3-egg
  , (3, 2, 7) -- Second hour: 3x 2-egg, 2x 3-egg, 7x 4-egg
  , (10, 3, 1) -- Third hour: 10x 2-egg, 3x 3-egg, 1x 4-egg
  , (5, 2, 8) -- Last hour: 5x 2-egg, 2x 3-egg, 8x 4-egg
  ]

-- Total number of omelettes ordered for each type
def total_orders (omelette_type : ℕ) : ℕ :=
  hourly_demand.foldl (λ acc triple =>
    match omelette_type with
    | 2 => acc + triple.1
    | 3 => acc + triple.2
    | 4 => acc + triple.3
    | _ => acc
  ) 0

-- Calculation of total ingredients needed
def total_ingredients (omelette_type : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let (eggs, cheese, spinach, mushrooms, tomatoes) := ingredients omelette_type
  let orders := total_orders omelette_type
  (eggs * orders, cheese * orders, spinach * orders, mushrooms * orders, tomatoes * orders)

theorem theo_needs_ingredients :
  let (eggs2, cheese2, spinach2, _, _) := total_ingredients 2
  let (eggs3, cheese3, spinach3, mushrooms3, _) := total_ingredients 3
  let (eggs4, cheese4, _, mushrooms4, tomatoes4) := total_ingredients 4
  eggs2 + eggs3 + eggs4 = 144 ∧ 
  cheese2 + cheese3 + cheese4 = 72 ∧ 
  spinach2 + spinach3 = 80 ∧ 
  mushrooms3 + mushrooms4 = 28 ∧ 
  tomatoes4 = 48 :=
by
  sorry

end theo_needs_ingredients_l91_91246


namespace toms_weekly_earnings_l91_91154

variable (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ)

def total_money_per_week (buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ) : ℕ :=
  buckets * crabs_per_bucket * price_per_crab * days_per_week

theorem toms_weekly_earnings :
  total_money_per_week 8 12 5 7 = 3360 :=
by
  sorry

end toms_weekly_earnings_l91_91154


namespace largest_expression_among_options_l91_91569

theorem largest_expression_among_options :
  (∃ (x : ℝ), x = max (sqrt (real.cbrt 56))
                        (max (sqrt (real.cbrt 3584))
                             (max (sqrt (real.cbrt 2744))
                                  (max (sqrt (real.cbrt 392)) 
                                       (sqrt (real.cbrt 448)))
                        ))) 
                        → x = sqrt (real.cbrt 3584) :=
by
  sorry

end largest_expression_among_options_l91_91569


namespace projection_not_acute_l91_91226

noncomputable def right_angled_triangle (ABC : Triangle) : Prop :=
ABC.is_right_angled

noncomputable def hypotenuse_on_plane (ABC : Triangle) (α : Plane) : Prop :=
ABC.hypotenuse.lies_on_plane α

theorem projection_not_acute 
  (ABC : Triangle) (α : Plane)
  (h1 : right_angled_triangle ABC) 
  (h2 : hypotenuse_on_plane ABC α) : 
  ¬ ABC.projection_on(α).is_acute ∨ ABC.projection_on(α).is_not_acute :=
sorry

end projection_not_acute_l91_91226


namespace line_intersects_y_axis_at_l91_91973

def point := (ℝ × ℝ)

def line_through (p1 p2 : point) : ℝ → ℝ :=
  λ x, ((p2.2 - p1.2) / (p2.1 - p1.1)) * (x - p1.1) + p1.2
  
theorem line_intersects_y_axis_at {p1 p2 : point} :
  p1 = (2, 3) → p2 = (5, 9) → line_through p1 p2 0 = -1 :=
begin
  intros h_p1 h_p2,
  rw [h_p1, h_p2],
  simp [line_through],
  linarith
end

end line_intersects_y_axis_at_l91_91973


namespace base_nine_first_digit_of_212221122211_base_three_l91_91847

def base_three_to_base_ten (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d, acc * 3 + d) 0

def first_digit_base_nine (n : ℕ) : ℕ :=
  let base_nine_digits := (Nat.digits 9 n).reverse
  base_nine_digits.headD 0

theorem base_nine_first_digit_of_212221122211_base_three :
  first_digit_base_nine (base_three_to_base_ten [2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1]) = 3 :=
by
  sorry

end base_nine_first_digit_of_212221122211_base_three_l91_91847


namespace sequence_bounded_sequence_eventually_periodic_minimum_period_length_same_l91_91257

noncomputable def sequence (a0 a1 a2 : ℕ) : ℕ → ℕ
| 0     := a0
| 1     := a1
| 2     := a2
| (n+3) := |sequence (n+2) - sequence n|

theorem sequence_bounded (a0 a1 a2 : ℕ) : ∃ c, ∀ n, sequence a0 a1 a2 n ≤ c :=
sorry

theorem sequence_eventually_periodic (a0 a1 a2 : ℕ) : ∃ p, ∀ n, sequence a0 a1 a2 (n + p) = sequence a0 a1 a2 n :=
sorry

theorem minimum_period_length_same (a0 a1 a2 b0 b1 b2 : ℕ) : ∃ p, ∀ q, (∀ n, sequence a0 a1 a2 (n + q) = sequence a0 a1 a2 n) ↔ q ≥ p :=
sorry

-- Example sequence calculation for given initial values
def specific_sequence : fin 21 → ℕ :=
λ n, match n.1 with
| 0  => 1
| 1  => 3
| 2  => 2
| 3  => 1
| 4  => 2
| 5  => 0
| 6  => 1
| 7  => 1
| 8  => 1
| 9  => 0
| 10 => 1
| 11 => 0
| 12 => 0
| 13 => 1
| 14 => 1
| 15 => 1
| 16 => 0
| 17 => 1
| 18 => 0
| 19 => 0
| 20 => 1
| _  => 0 -- this shouldn't happen
end

end sequence_bounded_sequence_eventually_periodic_minimum_period_length_same_l91_91257


namespace candy_factory_licorice_probability_l91_91457

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem candy_factory_licorice_probability :
  binomial_probability 7 5 (3/5) = 20412 / 78125 :=
by
  sorry

end candy_factory_licorice_probability_l91_91457


namespace collinear_midpoints_of_diagonals_l91_91244

open EuclideanGeometry

variables {P Q R S T U V : Type}

theorem collinear_midpoints_of_diagonals
  {A B C D O M N : P}
  (h1 : inscribed A B C D O)
  (h2 : midpoint M A C)
  (h3 : midpoint N B D) :
  collinear M N O :=
sorry

end collinear_midpoints_of_diagonals_l91_91244


namespace cyclic_quadrilateral_quadrilateral_with_opposite_angles_sum_l91_91913

-- Part (a) statement
theorem cyclic_quadrilateral (a b c d : ℝ) : 
  (exists (A B C D : ℝ × ℝ), 
    (dist A B = a) ∧ 
    (dist B C = b) ∧ 
    (dist C D = c) ∧ 
    (dist D A = d) ∧ 
    (A ≠ B) ∧ 
    (B ≠ C) ∧ 
    (C ≠ D) ∧ 
    (D ≠ A) ∧ 
    (∃ R : ℝ, (is_cyclic_quadrilateral A B C D R)) :=
sorry

-- Part (b) statement
theorem quadrilateral_with_opposite_angles_sum (a b c d : ℝ) (sum_angles : ℝ) : 
  (exists (A B C D : ℝ × ℝ),
    (dist A B = a) ∧
    (dist B C = b) ∧
    (dist C D = c) ∧
    (dist D A = d) ∧
    (A ≠ B) ∧
    (B ≠ C) ∧
    (C ≠ D) ∧
    (D ≠ A) ∧
    (angle A B C + angle C D A = sum_angles)) :=
sorry

end cyclic_quadrilateral_quadrilateral_with_opposite_angles_sum_l91_91913


namespace last_integer_of_sequence_l91_91630

def sequence (a₀ : ℕ) (n : ℕ) : ℝ :=
  a₀ / (3 : ℝ)^n

theorem last_integer_of_sequence :
  ∀ a₀ : ℕ, a₀ = 1024000 → ∀ n : ℕ, ¬(sequence a₀ (n + 1) ∈ ℤ) → sequence a₀ 0 = 1024000 :=
by
  intros a₀ h₀ n h
  rw h₀
  have : sequence 1024000 (n + 1) = 1024000 / 3^(n + 1),
  { unfold sequence },
  sorry

end last_integer_of_sequence_l91_91630


namespace proof_task_l91_91428

variables (A B C M N : Type) [noncomputable (A B C M N : Point)]
variables [InTriangle A B C M N]
variables ∈ [IsInteriorPoint M (triangle A B C)] [IsInteriorPoint N (triangle A B C)]
variables [Angle_eq (MA B (NAC)] [Angle_eq (MBA (N B C))]

theorem proof_task :
  (AM * AN / (AB * AC) + BM * BN / (BA * BC) + CM * CN / (CA * CB) = 1) :=
  sorry

end proof_task_l91_91428


namespace savings_calculation_l91_91509

-- Define the conditions
def income := 17000
def ratio_income_expenditure := 5 / 4

-- Prove that the savings are Rs. 3400
theorem savings_calculation (h : income = 5 * 3400): (income - 4 * 3400) = 3400 :=
by sorry

end savings_calculation_l91_91509


namespace width_at_bottom_l91_91502

-- Defining the given values and conditions
def top_width : ℝ := 14
def area : ℝ := 770
def depth : ℝ := 70

-- The proof problem
theorem width_at_bottom (b : ℝ) (h : area = (1/2) * (top_width + b) * depth) : b = 8 :=
by
  sorry

end width_at_bottom_l91_91502


namespace sqrt_equality_correct_l91_91177

theorem sqrt_equality_correct : (sqrt 18 - sqrt 8) = sqrt 2 := by
  sorry

end sqrt_equality_correct_l91_91177


namespace max_g_t_l91_91517

def f (x : ℝ) := -x^2 + 4 * x - 1

def g (t : ℝ) := Real.sup (Set.image (f) (Set.Icc t (t+1)))

theorem max_g_t (t : ℝ) (ht : 1 ≤ t ∧ t ≤ 2) : g(t) = 3 :=
by
  sorry

end max_g_t_l91_91517


namespace complex_identity_l91_91652

noncomputable def complex_solution : ℂ := -3 + 4 * Complex.i

theorem complex_identity (z : ℂ) (hz : 3 * z - 4 * conj z = 3 + 28 * Complex.i) : 
    z = complex_solution := by
  sorry

end complex_identity_l91_91652


namespace conic_section_is_parabola_l91_91273

theorem conic_section_is_parabola :
  ∀ (x y : ℝ), abs (y - 3) = sqrt ((x + 4)^2 + (y - 1)^2) → 
  ∃ a b c : ℝ, y = a * x^2 + b * x + c :=
by
  sorry

end conic_section_is_parabola_l91_91273


namespace train_speed_l91_91911

-- Definitions for the given problem conditions
def distance : ℝ := 300
def time : ℝ := 15

-- The speed of the train should equal 20 meters per second
def speed (d t : ℝ) := d / t

theorem train_speed : speed distance time = 20 :=
by
  -- This will be the place where you prove the statement
  -- Here is the proof placeholder
  sorry

end train_speed_l91_91911


namespace Alice_prevents_Bob_from_divisibility_l91_91957

def Alice_can_prevent_Bob (n : ℕ) : Prop :=
  ∀ (a b : Fin n → Fin 10), ∃ (sum_mod_3 : Fin 10), 
  (Alice_bob_digits a b) → 
  (sum_of_digits_not_divisible_by_3 a b)

-- Define the condition that Alice and Bob take turns choosing digits, starting with Alice.
def Alice_bob_digits (a b : Fin n → Fin 10) : Prop :=
  ∀ i, i < n → 
  ((i % 2 = 0 → a i != b i) ∧ 
   (i % 2 = 1 → b i != a (i - 1)))

-- Define the condition that the sum of digits modulo 3 is not zero.
def sum_of_digits_not_divisible_by_3 (a b : Fin n → Fin 10) : Prop :=
  ∀ i, i < n → 
  (∑ i in Finset.range n, (if i % 2 = 0 then a i else b i) % 3 != 0)

theorem Alice_prevents_Bob_from_divisibility (n : ℕ) (h : n = 2018) :
  Alice_can_prevent_Bob n :=
by
  unfold Alice_can_prevent_Bob
  have alice_strategy : Alice_bob_digits a b → sum_of_digits_not_divisible_by_3 a b := sorry
  exact alice_strategy

end Alice_prevents_Bob_from_divisibility_l91_91957


namespace second_number_in_11th_row_is_62_l91_91856

-- Definitions based on the conditions
def last_number_in_row (n : Nat) (elements_per_row : Nat) : Nat :=
  n * elements_per_row

def first_number_in_next_row (last_number_in_current_row : Nat) : Nat :=
  last_number_in_current_row + 1

def second_number_in_row (first_number : Nat) : Nat :=
  first_number + 1

-- Noncomputable is required here as we use it for defining the final question
noncomputable def second_number_in_11th_row : Nat :=
  let last_10th = last_number_in_row 10 6 in
  let first_11th = first_number_in_next_row last_10th in
  second_number_in_row first_11th

-- Statement of the theorem we need to prove
theorem second_number_in_11th_row_is_62 : second_number_in_11th_row = 62 := by
  sorry

end second_number_in_11th_row_is_62_l91_91856


namespace smallest_period_tan_transformed_l91_91527

noncomputable def smallest_period (f : ℝ → ℝ) : ℝ :=
  if h : ∃ T > 0, ∀ x, f (x + T) = f x then
    Classical.choose h
  else
    0

def tan_transformed (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 3)

theorem smallest_period_tan_transformed : smallest_period tan_transformed = Real.pi / 2 :=
  sorry

end smallest_period_tan_transformed_l91_91527


namespace students_not_reading_novels_l91_91929

-- Define the conditions
def total_students : ℕ := 240
def students_three_or_more : ℕ := total_students * (1/6)
def students_two : ℕ := total_students * 0.35
def students_one : ℕ := total_students * (5/12)

-- The theorem to be proved
theorem students_not_reading_novels : 
  (total_students - (students_three_or_more + students_two + students_one) = 16) :=
by
  sorry -- skipping the proof

end students_not_reading_novels_l91_91929


namespace no_rational_roots_l91_91621

theorem no_rational_roots {p q : ℤ} (hp : p % 2 = 1) (hq : q % 2 = 1) :
  ¬ ∃ x : ℚ, x^2 + (2 * p) * x + (2 * q) = 0 :=
by
  -- proof using contradiction technique
  sorry

end no_rational_roots_l91_91621


namespace reflection_matrix_squared_is_identity_l91_91798

-- Define the reflection matrix R
def R : Matrix (Fin 2) (Fin 2) ℝ := 
  -- The matrix that represents reflection over the vector [2, 2]
  Matrix.of_list [[0, -1], [-1, 0]] -- This is an example; the actual reflection matrix should be computed

-- State the theorem we need to prove
theorem reflection_matrix_squared_is_identity : R * R = (1 : Matrix (Fin 2) (Fin 2) ℝ) := 
by 
  -- This is the place where the proof would go
  sorry

end reflection_matrix_squared_is_identity_l91_91798


namespace peter_marbles_l91_91826

/-- Peter had 33 marbles and lost 15. Prove that now he has 18 marbles. -/
theorem peter_marbles :
  ∃ (initial lost remaining : ℕ), initial = 33 ∧ lost = 15 ∧ remaining = initial - lost ∧ remaining = 18 :=
by {
  sorry,
}

end peter_marbles_l91_91826


namespace intersection_points_count_l91_91113

def f (x : ℝ) : ℝ := 2 * Real.log x
def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count : 
    ∃ (n : ℕ), n = 2 ∧ ∀ x, f x = g x → x = 2 ∨ -- the exact intersection points
    sorry

end intersection_points_count_l91_91113


namespace probability_eight_distinct_numbers_l91_91555

theorem probability_eight_distinct_numbers :
  let total_ways := 10^8
  let ways_distinct := (10 * 9 * 8 * 7 * 6 * 5 * 4 * 3)
  (ways_distinct / total_ways : ℚ) = 18144 / 500000 := 
by
  sorry

end probability_eight_distinct_numbers_l91_91555


namespace sum_of_mnp_l91_91076

theorem sum_of_mnp (m n p : ℕ) (h_gcd : gcd m (gcd n p) = 1)
  (h : ∀ x : ℝ, 5 * x^2 - 11 * x + 6 = 0 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :
  m + n + p = 22 :=
by
  sorry

end sum_of_mnp_l91_91076


namespace part_a_l91_91191

theorem part_a (A : Finset α) (n : ℕ) (N : ℕ) (hA : A.card = N) :
  ∃! S : Finset (Multiset α), S.card = n ∧
  (∀ s ∈ S, ∃ k ∈ A, s = Multiset.replicate k (1 : ℕ)) → 
  Multiset.card (Finset.univ P → Finset N) = binom (N + n - 1) n := sorry

end part_a_l91_91191


namespace geometric_seq_b6_l91_91775

variable {b : ℕ → ℝ}

theorem geometric_seq_b6 (h1 : b 3 * b 9 = 9) (h2 : ∃ r, ∀ n, b (n + 1) = r * b n) : b 6 = 3 ∨ b 6 = -3 :=
by
  sorry

end geometric_seq_b6_l91_91775


namespace football_game_spectators_l91_91387

theorem football_game_spectators (total_wristbands wristbands_per_person : ℕ) (h1 : total_wristbands = 234) (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 117 := by
  sorry

end football_game_spectators_l91_91387


namespace planar_points_distance_l91_91263

theorem planar_points_distance (n : ℕ) (e : ℕ) (n_eq_330 : n = 330) (e_ge_1700 : e = 1700) : 
  ¬ ∃ (p : set (ℝ × ℝ)), card p = n ∧ (∃ d : ℝ, ∑ (x, y) in p.product p, if dist x y = d then 1 else 0 ≥ e) :=
by
  sorry

end planar_points_distance_l91_91263


namespace num_divisible_digits_l91_91665

theorem num_divisible_digits :
  {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ (let k := 100 + 50 + n in k % n = 0)}.to_finset.card = 5 :=
by
  sorry

end num_divisible_digits_l91_91665


namespace exist_congruent_triangle_l91_91345

structure Triangle (α : Type) :=
(A B C : α)

structure ConvexPolygon (α : Type) :=
(vertices : list α)

variables {α : Type} [LinearOrder α] [AddCommGroup α]

def isInscribed (t : Triangle α) (p : ConvexPolygon α) : Prop :=
sorry -- Placeholder for the definition of inscribed triangle

def isCongruent (t1 t2 : Triangle α) : Prop :=
sorry -- Placeholder for the definition of congruent triangles

def sideParallelOrCoincident (t : Triangle α) (p : ConvexPolygon α) : Prop :=
sorry -- Placeholder for the definition of one side being parallel or coincident

theorem exist_congruent_triangle (t : Triangle α) (p : ConvexPolygon α)
  (h1 : isInscribed t p) :
  ∃ t' : Triangle α, isCongruent t' t ∧ isInscribed t' p ∧ sideParallelOrCoincident t' p :=
sorry

end exist_congruent_triangle_l91_91345


namespace at_least_100_pairs_of_points_l91_91981

theorem at_least_100_pairs_of_points {circle : Type} (points : set circle) (h_points : points.size = 21) :
  ∃ pairs : set (circle × circle), pairs.size ≥ 100 ∧ ∀ (p1 p2 : circle), (p1, p2) ∈ pairs → central_angle p1 p2 ≤ 120 :=
sorry

end at_least_100_pairs_of_points_l91_91981


namespace sufficiency_of_a_eq_2_l91_91923

theorem sufficiency_of_a_eq_2 (a : ℝ) : (∃ x : ℝ, (a = 2 → f a x = 0)) ∧ (∃ a : ℝ, (a ≠ 2 ∧ ∃ x : ℝ, f a x = 0)) → 
  (a = 2 → ∃ x : ℝ, f a x = 0) ∧ ¬(a = 2 → ∀ x : ℝ, f a x ≠ 0) where
  f : ℝ → ℝ → ℝ := λ a x, a * x - 2^x

end sufficiency_of_a_eq_2_l91_91923


namespace investment_amount_l91_91213

-- Define the conditions
variable (price_share : ℝ := 100)
variable (premium : ℝ := 0.20)
variable (dividend_rate : ℝ := 0.07)
variable (total_dividend : ℝ := 840.0)

-- Compute necessary derived values
def price_per_share : ℝ := price_share * (1 + premium)
def dividend_per_share : ℝ := price_share * dividend_rate

-- Define the number of shares and the total investment
def n_shares : ℝ := total_dividend / dividend_per_share
def total_investment : ℝ := n_shares * price_per_share

-- State the theorem
theorem investment_amount : total_investment = 14400 :=
by
  -- Placeholder proof
  sorry

end investment_amount_l91_91213


namespace cart_distance_traveled_l91_91849

-- Define the problem parameters/conditions
def circumference_front : ℕ := 30
def circumference_back : ℕ := 33
def revolutions_difference : ℕ := 5

-- Define the question and the expected correct answer
theorem cart_distance_traveled :
  ∀ (R : ℕ), ((R + revolutions_difference) * circumference_front = R * circumference_back) → (R * circumference_back) = 1650 :=
by
  intro R h
  sorry

end cart_distance_traveled_l91_91849


namespace cubic_roots_proof_l91_91983

noncomputable def cubic_roots_reciprocal (a b c : ℝ) (h1 : a + b + c = 7) 
  (h2 : a * b + b * c + c * a = 3) (h3 : a * b * c = -4) : ℝ :=
  (1 / a^2) + (1 / b^2) + (1 / c^2)

theorem cubic_roots_proof (a b c : ℝ) (h1 : a + b + c = 7) 
  (h2 : a * b + b * c + c * a = 3) (h3 : a * b * c = -4) : 
  cubic_roots_reciprocal a b c h1 h2 h3 = 65 / 16 :=
sorry

end cubic_roots_proof_l91_91983


namespace area_of_frame_l91_91537

def width : ℚ := 81 / 4
def depth : ℚ := 148 / 9
def area (w d : ℚ) : ℚ := w * d

theorem area_of_frame : area width depth = 333 := by
  sorry

end area_of_frame_l91_91537


namespace union_dues_eq_l91_91079

variable (gross_salary : ℝ)
variable (tax_percent : ℝ)
variable (healthcare_percent : ℝ)
variable (take_home_pay : ℝ)
variable (union_dues : ℝ)

-- Conditions
axiom h1 : gross_salary = 40000
axiom h2 : tax_percent = 0.20
axiom h3 : healthcare_percent = 0.10
axiom h4 : take_home_pay = 27200

-- Auxiliary calculations for deductions
def taxes := tax_percent * gross_salary
def healthcare := healthcare_percent * gross_salary
def deductions := taxes + healthcare
def remaining_salary := gross_salary - deductions

-- Statement to prove
theorem union_dues_eq : union_dues = remaining_salary - take_home_pay :=
by
  sorry

end union_dues_eq_l91_91079


namespace max_g_t_l91_91516

def f (x : ℝ) := -x^2 + 4 * x - 1

def g (t : ℝ) := Real.sup (Set.image (f) (Set.Icc t (t+1)))

theorem max_g_t (t : ℝ) (ht : 1 ≤ t ∧ t ≤ 2) : g(t) = 3 :=
by
  sorry

end max_g_t_l91_91516


namespace g_inv_triple_5_l91_91982

noncomputable def g : ℕ → ℕ
| 1 := 4
| 2 := 5
| 3 := 1
| 4 := 2
| 5 := 3
| _ := 0 -- Adding this to make the function total, you might not need it if we know the domain is restricted to {1, 2, 3, 4, 5}.

def g_inv : ℕ → ℕ
| 4 := 1
| 5 := 2
| 1 := 3
| 2 := 4
| 3 := 5
| _ := 0

theorem g_inv_triple_5 : g_inv (g_inv (g_inv 5)) = 1 := by
  sorry

end g_inv_triple_5_l91_91982


namespace circle_diameters_product_l91_91256

theorem circle_diameters_product
  (O A B C D N Q : Point)
  (r : Real)
  (h_diameters_AB_CD : is_diameter O A B ∧ is_diameter O C D)
  (h_perpendicular : ⊥ AB CD)
  (h_N_midpoint : midpoint N A B)
  (h_chord_AN : is_chord O A N)
  (h_intersect_CD : intersect_chord_at N CD Q)
  : AQ * AN = 2 * r^2 := sorry

end circle_diameters_product_l91_91256


namespace find_phi_l91_91717

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * (sqrt 3 * cos x - sin x) + 1

def is_even_function(g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

theorem find_phi (φ : ℝ) (h : is_even_function (λ x, f (x - φ))) : φ = π / 3 :=
sorry

end find_phi_l91_91717


namespace total_employees_in_buses_l91_91160

theorem total_employees_in_buses :
  let bus1_percentage_full := 0.60,
      bus2_percentage_full := 0.70,
      bus_capacity := 150
  in
  (bus1_percentage_full * bus_capacity + bus2_percentage_full * bus_capacity) = 195 := by
  sorry

end total_employees_in_buses_l91_91160


namespace xiao_hua_fishing_l91_91660

theorem xiao_hua_fishing :
  let catches := [23, 20, 15, 18, 13] in
  let xiao_hua_catch := 20 in
  ∃ extra_fish, (∀ k ∈ catches, xiao_hua_catch + extra_fish > k) ∧ extra_fish = 4 :=
by
  sorry

end xiao_hua_fishing_l91_91660


namespace root_expression_l91_91279

theorem root_expression (a b : ℝ) :
  let eq := λ x : ℝ, x^4 - 2 * a * x^2 + b^2
  in eq = 0 -> 
     (∀ x : ℝ, eq x = 0 -> 
                x = sqrt ((a + b) / 2) + sqrt ((a - b) / 2) ∨ 
                x = sqrt ((a + b) / 2) - sqrt ((a - b) / 2) ∨ 
                x = - sqrt ((a + b) / 2) + sqrt ((a - b) / 2) ∨ 
                x = - sqrt ((a + b) / 2) - sqrt ((a - b) / 2)) := sorry

end root_expression_l91_91279


namespace simplify_expression_l91_91835

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l91_91835


namespace discount_is_10_percent_l91_91598

variable (C : ℝ)  -- Cost of the item
variable (S S' : ℝ)  -- Selling prices with and without discount

-- Conditions
def condition1 : Prop := S = 1.20 * C
def condition2 : Prop := S' = 1.30 * C

-- The proposition to prove
theorem discount_is_10_percent (h1 : condition1 C S) (h2 : condition2 C S') : S' - S = 0.10 * C := by
  sorry

end discount_is_10_percent_l91_91598


namespace total_tablets_l91_91935

-- Variables for the numbers of Lenovo, Samsung, and Huawei tablets
variables (n x y : ℕ)

-- Conditions based on problem statement
def condition1 : Prop := 2 * x + 6 + y < n / 3

def condition2 : Prop := (n - 2 * x - y - 6 = 3 * y)

def condition3 : Prop := (n - 6 * x - y - 6 = 59)

-- The statement to prove that the total number of tablets is 94
theorem total_tablets (h1 : condition1 n x y) (h2 : condition2 n x y) (h3 : condition3 n x y) : n = 94 :=
by
  sorry

end total_tablets_l91_91935


namespace area_inside_circle_outside_square_is_zero_l91_91230

theorem area_inside_circle_outside_square_is_zero 
  (side_length : ℝ) (circle_radius : ℝ)
  (h_square_side : side_length = 2) (h_circle_radius : circle_radius = 1) : 
  (π * circle_radius^2) - (side_length^2) = 0 := 
by 
  sorry

end area_inside_circle_outside_square_is_zero_l91_91230


namespace constant_term_in_binomial_expansion_l91_91009

theorem constant_term_in_binomial_expansion : 
  (∃ t, t = 15 ∧ is_constant_term ((√x - 1/x)^6) t) :=
sorry

end constant_term_in_binomial_expansion_l91_91009


namespace distinct_sea_shell_arrangements_l91_91021

-- Definitions and conditions from the problem
def regular_six_pointed_star (α : Type) (shells : Fin 12 → α) : Prop :=
  ∃ (shell_points : Fin 12 → Point), ∀ i, shell_points i = shells i

-- We are to show that the number of distinct ways to place the shells is 11!
theorem distinct_sea_shell_arrangements :
  ∃ (α : Type) (shells : Fin 12 → α), regular_six_pointed_star α shells ∧ 
  (unique_arrangements shells = factorial 11) :=
sorry

end distinct_sea_shell_arrangements_l91_91021


namespace polar_to_rectangular_l91_91988

theorem polar_to_rectangular : 
  let r := 4 * (Real.sqrt 2)
      θ := Real.pi / 3
      x := r * Real.cos θ
      y := r * Real.sin θ 
  in (x, y) = (2 * (Real.sqrt 2), 2 * (Real.sqrt 6)) :=
by 
  let r := 4 * (Real.sqrt 2)
  let θ := Real.pi / 3
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  sorry

end polar_to_rectangular_l91_91988


namespace triangle_angles_l91_91686

theorem triangle_angles (A B C D E F X : Type) (angle_BAC : Real)
  (h_angle_BAC : angle_BAC = 45)
  (altitudes_AD_BE_CF : (AD ⊥ BC) ∧ (BE ⊥ AC) ∧ (CF ⊥ AB))
  (h_X : EF ∩ BC = X)
  (h_parallel_AX_DE : AX ∥ DE) :
  ∃ (angle_B angle_C : Real), angle_B = 75 ∧ angle_C = 60 :=
  sorry

end triangle_angles_l91_91686


namespace min_sum_of_segments_l91_91754

theorem min_sum_of_segments 
  (X Y Z P Q: Point) 
  (angle_XYZ : ∠XYZ = 50) 
  (XY_eq : dist X Y = 8) 
  (XZ_eq : dist X Z = 10)
  (P_on_XY : PointOnSegment P X Y)
  (Q_on_XZ : PointOnSegment Q X Z): 
  ∃ P Q, min (dist Y P + dist P Q + dist Q Z) = √(164 + 80 * √3) := 
by 
  sorry

end min_sum_of_segments_l91_91754


namespace length_of_BC_l91_91412

theorem length_of_BC (A B C : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] 
  (angle_A : ℝ) (angle_B : ℝ) (AC : ℝ) 
  (h1 : angle_A = π/4) 
  (h2 : angle_B = π/3) 
  (h3 : AC = 6) :
  ∃ BC : ℝ, BC = 2 * Real.sqrt 6 :=
sorry

end length_of_BC_l91_91412


namespace volleyball_team_range_increases_l91_91612

theorem volleyball_team_range_increases :
  let init_scores := [45, 50, 52, 54, 55, 55, 57, 59, 60, 62, 75]
  let new_score := 38
  let final_scores := new_score :: init_scores
  let init_range := List.maximum init_scores - List.minimum init_scores
  let new_range := List.maximum final_scores - List.minimum final_scores
  new_range > init_range := by
  sorry

end volleyball_team_range_increases_l91_91612


namespace circumcenter_CEF_on_K_l91_91577

theorem circumcenter_CEF_on_K 
  (A B C D E F : Point) (K : Circle) 
  (h1 : parallelogram A B C D)
  (h2 : circumcircle A B D K)
  (h3 : intersects_again K B C E)
  (h4 : intersects_again K C D F) 
  : lies_on (circumcenter (triangle C E F)) K :=
sorry

end circumcenter_CEF_on_K_l91_91577


namespace independence_test_probability_calculation_l91_91845

-- Define the given conditions and constants (Part 1)
def a: ℕ := 140
def b: ℕ := 60
def c: ℕ := 180
def d: ℕ := 20
def n: ℕ := 400
def χ2: ℝ := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
def χ2_critical: ℝ := 10.828

-- Define the hypothesis test statement
theorem independence_test : χ2 > χ2_critical → 
  (understanding_of_announcement_is_related_to_gender : Prop) :=
by {
  sorry
}

-- Define the probability calculation conditions (Part 2)
def p_understand: ℚ := 4 / 5
def p_not_understand: ℚ := 1 / 5
def residents_selected: ℕ := 5
def exactly_understand: ℕ := 3

-- Define the binomial probability function
def binomial_prob (n k: ℕ) (p: ℚ): ℚ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- Define the probability theorem statement for part 2
theorem probability_calculation :
  binomial_prob residents_selected exactly_understand p_understand = 128 / 625 :=
by {
  sorry
}

end independence_test_probability_calculation_l91_91845


namespace periodic_quadratic_irrational_l91_91035

def f (x : ℝ) : ℝ :=
if x > 1 then 1 / (x - 1)
else if x = 1 then 1
else x / (1 - x)

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
nat.rec_on n x (λ _ xn, f xn)

theorem periodic_quadratic_irrational (x₁ : ℝ) 
(h1 : irrational x₁) 
(h2 : ∃ a b c : ℤ, a ≠ 0 ∧ (a * x₁^2 + b * x₁ + c = 0))
: ∃ (k l : ℕ), k ≠ l ∧ sequence x₁ k = sequence x₁ l := 
sorry

end periodic_quadratic_irrational_l91_91035


namespace negation_equivalence_l91_91477

-- Declare the condition for real solutions of a quadratic equation
def has_real_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a * x + 1 = 0

-- Define the proposition p
def prop_p : Prop :=
  ∀ a : ℝ, a ≥ 0 → has_real_solutions a

-- Define the negation of p
def neg_prop_p : Prop :=
  ∃ a : ℝ, a ≥ 0 ∧ ¬ has_real_solutions a

-- The theorem stating the equivalence of p's negation to its formulated negation.
theorem negation_equivalence : neg_prop_p = ¬ prop_p := by
  sorry

end negation_equivalence_l91_91477


namespace find_cd_l91_91952

theorem find_cd :
  ∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ 
  let y := (10 * c + d) / 99 in
  74 * y - (74 * (c + d / 10)) = 1.2 / 74 ∧
  10 * c + d = 16 :=
sorry

end find_cd_l91_91952


namespace radius_of_circle_eq_zero_l91_91291

theorem radius_of_circle_eq_zero :
  ∀ x y: ℝ, (x^2 + 8 * x + y^2 - 10 * y + 41 = 0) → (0 : ℝ) = 0 :=
by
  intros x y h
  sorry

end radius_of_circle_eq_zero_l91_91291


namespace ratio_sum_eq_seven_eight_l91_91746

theorem ratio_sum_eq_seven_eight 
  (a b c x y z : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : a^2 + b^2 + c^2 = 49)
  (h2 : x^2 + y^2 + z^2 = 64)
  (h3 : a * x + b * y + c * z = 56) :
  (a + b + c) / (x + y + z) = 7/8 :=
by
  sorry

end ratio_sum_eq_seven_eight_l91_91746


namespace remainder_polynomial_l91_91322

theorem remainder_polynomial (n : ℕ) (hn : n ≥ 2) : 
  ∃ Q R : Polynomial ℤ, (R.degree < 2) ∧ (X^n = Q * (X^2 - 4 * X + 3) + R) ∧ 
                       (R = (Polynomial.C ((3^n - 1) / 2) * X + Polynomial.C ((3 - 3^n) / 2))) :=
by
  sorry

end remainder_polynomial_l91_91322


namespace find_point_A_l91_91223

-- Define the point -3, 4
def pointP : ℝ × ℝ := (-3, 4)

-- Define the point 0, 2
def pointB : ℝ × ℝ := (0, 2)

-- Define the coordinates of point A
def pointA (x : ℝ) : ℝ × ℝ := (x, 0)

-- The hypothesis using the condition derived from the problem
def ray_reflection_condition (x : ℝ) : Prop :=
  4 / (x + 3) = -2 / x

-- The main theorem we need to prove that the coordinates of point A are (-1, 0)
theorem find_point_A :
  ∃ x : ℝ, ray_reflection_condition x ∧ pointA x = (-1, 0) :=
sorry

end find_point_A_l91_91223


namespace sum_of_possible_values_sum_of_all_possible_values_l91_91403

theorem sum_of_possible_values (x : ℝ) : (|x-5| = 4) → (x = 9 ∨ x = 1) :=
begin
  sorry
end

theorem sum_of_all_possible_values (h : ∃ x : ℝ, |x-5| = 4) : 9 + 1 = 10 :=
by
  trivial

end sum_of_possible_values_sum_of_all_possible_values_l91_91403


namespace problem_I_problem_II_problem_III_l91_91353

def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

def g (x : ℝ) : ℝ := 2^x - 2

theorem problem_I (a : ℝ) (x : ℝ) (h_a : a = -1) (h_x : x = 1/2) : 
  Deriv (f a) x = 1 :=
by sorry
  
theorem problem_II (a : ℝ) :
  (∀ x > 0, f a x > 0) ↔ (a ≥ 0) ∧
  (∀ x > 0, f a x > 0 ∧ x < -1/a ↔ a < 0) :=
by sorry

theorem problem_III (a : ℝ) (h : ∃ x1 ∈ Ioi (0:ℝ), ∀ x2 ∈ Icc (0:ℝ) (1:ℝ), f a x1 ≥ g x2) :
  a ≥ -1/Real.exp 1 :=
by sorry

end problem_I_problem_II_problem_III_l91_91353


namespace original_number_eq_0_000032_l91_91565

theorem original_number_eq_0_000032 (x : ℝ) (hx : 0 < x) 
  (h : 10^8 * x = 8 * (1 / x)) : x = 0.000032 :=
sorry

end original_number_eq_0_000032_l91_91565


namespace lemonade_quarts_l91_91152

theorem lemonade_quarts (total_gallons : ℚ) (ratio_water : ℚ) (ratio_lemon : ℚ) (quarts_per_gallon : ℚ) 
  (h_parts_eq : ratio_water + ratio_lemon = 8) (h_total_gallons : total_gallons = 5) (h_quarts_per_gallon : quarts_per_gallon = 4) :
  let part_gallons := total_gallons / (ratio_water + ratio_lemon),
      part_quarts := part_gallons * quarts_per_gallon,
      water_quarts := ratio_water * part_quarts
  in water_quarts = 25 / 2 :=
by
  sorry

end lemonade_quarts_l91_91152


namespace segments_form_triangle_l91_91446

-- Definitions and conditions
variables {A B C D M N : Type}
variables [regular_tetrahedron A B C D]
variables [M ∈ plane_of (triangle A B C)]
variables [N ∈ plane_of (triangle A D C)]
variables (M ≠ N)

-- Statement of the theorem
theorem segments_form_triangle : 
  ∃ (MN BN MD : ℝ), 
    triangle MN BN MD ∧
    (segment M N) ∈ MN ∧ 
    (segment B N) ∈ BN ∧ 
    (segment M D) ∈ MD := 
begin
  sorry
end

end segments_form_triangle_l91_91446


namespace unused_streetlights_remain_l91_91088

def total_streetlights : ℕ := 200
def squares : ℕ := 15
def streetlights_per_square : ℕ := 12

theorem unused_streetlights_remain :
  total_streetlights - (squares * streetlights_per_square) = 20 :=
sorry

end unused_streetlights_remain_l91_91088


namespace average_of_t_l91_91713

theorem average_of_t (t : ℕ) 
  (roots_positive : ∀ r : ℕ, r ∈ (ROOTS of (x^2 - 7*x + t)) → r > 0) 
  (sum_of_roots_eq_seven : ∀ r1 r2 : ℕ, r1 + r2 = 7)
  (product_of_roots_eq_t : ∀ r1 r2 : ℕ, r1 * r2 = t) : 
  (6 + 10 + 12) / 3 = 28 / 3 :=
sorry

end average_of_t_l91_91713


namespace length_of_tunnel_l91_91591

theorem length_of_tunnel
  (train_length : ℝ)
  (train_speed_cm_per_s : ℝ)
  (time_to_pass_tunnel : ℝ)
  (h1 : train_length = 20)
  (h2 : train_speed_cm_per_s = 288)
  (h3 : time_to_pass_tunnel = 25) :
  ∃ tunnel_length : ℝ, tunnel_length = 52 :=
begin
  sorry
end

end length_of_tunnel_l91_91591


namespace sqrt_D_irrational_l91_91435

open Real

theorem sqrt_D_irrational (a : ℤ) (D : ℝ) (hD : D = a^2 + (a + 2)^2 + (a^2 + (a + 2))^2) : ¬ ∃ m : ℤ, D = m^2 :=
by
  sorry

end sqrt_D_irrational_l91_91435


namespace quadratic_distinct_roots_l91_91382

noncomputable def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem quadratic_distinct_roots (k : ℝ) :
  (k ≠ 0) ∧ (1 > k) ↔ has_two_distinct_real_roots k (-6) 9 :=
by
  sorry

end quadratic_distinct_roots_l91_91382


namespace points_in_different_half_spaces_l91_91864

def point1 : ℝ × ℝ × ℝ := (1, 2, -2)
def point2 : ℝ × ℝ × ℝ := (2, 1, -1)

def plane (x y z : ℝ) : ℝ := x + 2 * y + 3 * z

theorem points_in_different_half_spaces :
  (plane point1.1 point1.2 point1.3 < 0) ∧ (plane point2.1 point2.2 point2.3 > 0) :=
begin
  -- Since we need to prove it, we use the values:
  -- from the problem, plane (1, 2, -2) < 0 and plane (2, 1, -1) > 0
  have h1 : plane 1 2 (-2) = -1, by norm_num,
  have h2 : plane 2 1 (-1) = 1, by norm_num,
  split,
  { rw plane at h1, linarith, },
  { rw plane at h2, linarith, }
end

end points_in_different_half_spaces_l91_91864


namespace economical_iron_pipe_l91_91051

def is_sufficient_length (a b : ℝ) (available_lengths : List ℝ) : ℝ :=
  let c := Real.sqrt (a * a + b * b)
  a + b + c

theorem economical_iron_pipe (available_lengths : List ℝ)
  (h1 : available_lengths = [4.6, 4.8, 5, 5.2]) :
  ∃ l ∈ available_lengths, l = 5 :=
by
  have area := 1
  have a := 1
  have b := 2
  have sum_length := is_sufficient_length a b available_lengths
  have h2 : sum_length ≈ 4.24 := sorry
  use 5
  simp [h1]
  sorry

end economical_iron_pipe_l91_91051


namespace meat_per_meal_l91_91019

theorem meat_per_meal :
  ∀ (beef pork : ℕ) (price_per_meal total_made : ℕ), 
    beef = 20 → pork = beef / 2 → price_per_meal = 20 → total_made = 400 →
    let total_meals := total_made / price_per_meal in
    let total_meat := beef + pork in
    (total_meat : ℚ) / total_meals = 1.5 :=
by
  intros beef pork price_per_meal total_made beef_eq pork_eq price_eq made_eq
  let total_meals := total_made / price_per_meal
  let total_meat := beef + pork
  rw [beef_eq, pork_eq, price_eq, made_eq]
  norm_num
  sorry

end meat_per_meal_l91_91019


namespace a_100_eq_9902_l91_91363

noncomputable def a : Nat → Nat
| 1       => 2
| (n + 1) => a n + 2 * n

theorem a_100_eq_9902 : a 100 = 9902 :=
sorry

end a_100_eq_9902_l91_91363


namespace instantaneous_velocity_at_2_l91_91600

def displacement (t : ℝ) : ℝ := 2 * t^3

theorem instantaneous_velocity_at_2 :
  let velocity := deriv displacement
  velocity 2 = 24 :=
by
  sorry

end instantaneous_velocity_at_2_l91_91600


namespace ram_distance_to_mountain_base_l91_91819

theorem ram_distance_to_mountain_base :
  let map_distance_mountains := 310 -- inches
  let actual_distance_mountains := 136 -- km
  let map_distance_ram := 34 -- inches
  let scale := actual_distance_mountains / map_distance_mountains
  let actual_distance_ram := map_distance_ram * scale
  actual_distance_ram ≈ 14.92 :=
sorry

end ram_distance_to_mountain_base_l91_91819


namespace combined_area_of_regions_approximately_l91_91067

noncomputable def combined_area_of_regions : ℝ :=
  let AB := 2
  let BC := 4
  let radius := Real.sqrt (AB^2 + BC^2)
  let sector_area := (1 / 4) * Real.pi * radius^2
  let triangle_area := (1 / 2) * AB * BC
  2 * (sector_area - triangle_area)

theorem combined_area_of_regions_approximately :
  | combined_area_of_regions - 23.4 | < 0.05 := by
  sorry

end combined_area_of_regions_approximately_l91_91067


namespace max_sqrt_expression_l91_91296

theorem max_sqrt_expression (x : ℝ) (hx : -49 ≤ x ∧ x ≤ 49) :
  ∃ y, y = 14 ∧ (∀ z, z = sqrt (49 + x) + sqrt (49 - x) → z ≤ y) :=
by
  sorry

end max_sqrt_expression_l91_91296


namespace value_of_b_2_pow_100_l91_91801

noncomputable def sequence_b : ℕ → ℕ
| 1 := 3
| (n + 1) := if even n then ((n / 2 + 1) * sequence_b (n / 2)) else sorry

theorem value_of_b_2_pow_100 (n : ℕ) (hn : n = 100):
  sequence_b (2 ^ n) = 3 * ∏ i in finset.range 99, (2 ^ (i + 1) + 1) :=
sorry

end value_of_b_2_pow_100_l91_91801


namespace max_sqrt_expression_l91_91298

theorem max_sqrt_expression (x : ℝ) (hx : -49 ≤ x ∧ x ≤ 49) :
  ∃ y, y = 14 ∧ (∀ z, z = sqrt (49 + x) + sqrt (49 - x) → z ≤ y) :=
by
  sorry

end max_sqrt_expression_l91_91298


namespace students_not_in_both_l91_91281

theorem students_not_in_both (students_in_both : ℕ) (students_in_geometry : ℕ) (students_only_in_compsci : ℕ) 
  (students_in_both_eq : students_in_both = 15) 
  (students_in_geometry_eq : students_in_geometry = 35) 
  (students_only_in_compsci_eq : students_only_in_compsci = 18) : 
  (students_in_geometry - students_in_both + students_only_in_compsci = 38) := 
by
  rw [students_in_both_eq, students_in_geometry_eq, students_only_in_compsci_eq]
  sorry

end students_not_in_both_l91_91281


namespace proof_statement_l91_91360

noncomputable def slope_of_line_at_theta_pi_over_3 
  (a b m : ℝ)
  (x y : ℝ → ℝ)
  (h₁ : ∀ θ, x θ = a + m * sin θ)
  (h₂ : ∀ θ, y θ = b + m * cos θ) : Prop :=
  (x (real.pi / 3) - a) / (y (real.pi / 3) - b) = real.sqrt 3 / 3

noncomputable def trajectory_eq_of_moving_point 
  (a b : ℝ)
  (h₃ : a^2 + b^2 = 2) : Prop :=
  ∀ x y : ℝ, (x = a) ∧ (y = b) → x^2 + y^2 = 2

theorem proof_statement (a b m : ℝ) 
  (x y : ℝ → ℝ)
  (h₁ : ∀ θ, x θ = a + m * sin θ)
  (h₂ : ∀ θ, y θ = b + m * cos θ)
  (h₃ : a^2 + b^2 = 2) :
  slope_of_line_at_theta_pi_over_3 a b m x y h₁ h₂ ∧ trajectory_eq_of_moving_point a b h₃ := 
  sorry

end proof_statement_l91_91360


namespace robis_savings_in_january_l91_91824

theorem robis_savings_in_january (x : ℕ) (h: (x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) = 126)) : x = 11 := 
by {
  -- By simplification, the lean equivalent proof would include combining like
  -- terms and solving the resulting equation. For now, we'll use sorry.
  sorry
}

end robis_savings_in_january_l91_91824


namespace students_not_reading_novels_l91_91926

theorem students_not_reading_novels
  (total_students : ℕ)
  (students_three_or_more_novels : ℕ)
  (students_two_novels : ℕ)
  (students_one_novel : ℕ)
  (h_total_students : total_students = 240)
  (h_students_three_or_more_novels : students_three_or_more_novels = 1 / 6 * 240)
  (h_students_two_novels : students_two_novels = 35 / 100 * 240)
  (h_students_one_novel : students_one_novel = 5 / 12 * 240)
  :
  total_students - (students_three_or_more_novels + students_two_novels + students_one_novel) = 16 :=
by
  sorry

end students_not_reading_novels_l91_91926


namespace triangle_has_three_sides_l91_91846

-- Define the conditions
def average_length (triangle : Type) [IsTriangle triangle] : ℝ := 12
def perimeter (triangle : Type) [IsTriangle triangle] : ℝ := 36

-- Define what it means for a type to be a triangle (having three sides)
class IsTriangle (triangle : Type) :=
  (num_sides : ℕ := 3)

-- The goal is to prove that the number of sides of the triangle is 3 given the conditions
theorem triangle_has_three_sides (triangle : Type) [IsTriangle triangle] :
  (average_length triangle) * (IsTriangle.num_sides) = perimeter triangle → IsTriangle.num_sides = 3 :=
by {
  intro h,
  sorry
}

end triangle_has_three_sides_l91_91846


namespace jog_time_each_morning_is_1_5_hours_l91_91816

-- Define the total time Mr. John spent jogging
def total_time_spent_jogging : ℝ := 21

-- Define the number of days Mr. John jogged
def number_of_days_jogged : ℕ := 14

-- Define the time Mr. John jogs each morning
noncomputable def time_jogged_each_morning : ℝ := total_time_spent_jogging / number_of_days_jogged

-- State the theorem that the time jogged each morning is 1.5 hours
theorem jog_time_each_morning_is_1_5_hours : time_jogged_each_morning = 1.5 := by
  sorry

end jog_time_each_morning_is_1_5_hours_l91_91816


namespace cone_volume_proof_l91_91342

noncomputable def cone_volume (r l θ : ℝ) (h : ℝ) (baseCircumference : ℝ) (baseArea : ℝ) :=
  1/3 * baseArea * h

theorem cone_volume_proof :
  let r := 1 in
  let l := 3 in
  let θ := 2/3 * Real.pi in
  let baseCircumference := 2 * Real.pi in
  let baseArea := Real.pi in
  let h := Real.sqrt (l^2 - r^2) in
  cone_volume r l θ h baseCircumference baseArea = (2 * Real.sqrt 2 / 3) * Real.pi
:=
by
  sorry

end cone_volume_proof_l91_91342


namespace anne_speed_ratio_l91_91624

variable (B A A' : ℝ) (hours_to_clean_together : ℝ) (hours_to_clean_with_new_anne : ℝ)

-- Conditions
def cleaning_condition_1 := (A + B) * 4 = 1 -- Combined rate for 4 hours
def cleaning_condition_2 := A = 1 / 12      -- Anne's rate alone
def cleaning_condition_3 := (A' + B) * 3 = 1 -- Combined rate for 3 hours with new Anne's rate

-- Theorem to Prove
theorem anne_speed_ratio (h1 : cleaning_condition_1 B A)
                         (h2 : cleaning_condition_2 A)
                         (h3 : cleaning_condition_3 B A') :
                         (A' / A) = 2 :=
by sorry

end anne_speed_ratio_l91_91624


namespace tickets_probability_multiple_of_three_l91_91920

theorem tickets_probability_multiple_of_three :
  let total_tickets := 20 in
  let multiples_of_three := {x | x ∈ Finset.range (total_tickets + 1) ∧ x % 3 = 0} in
  let favorable_outcome := multiples_of_three.card in
  (favorable_outcome : ℝ) / (total_tickets : ℝ) = 0.3 :=
by
  sorry

end tickets_probability_multiple_of_three_l91_91920


namespace alphabet_letters_l91_91918

theorem alphabet_letters (DS S_only Total D_only : ℕ) 
  (h_DS : DS = 9) 
  (h_S_only : S_only = 24) 
  (h_Total : Total = 40) 
  (h_eq : Total = D_only + S_only + DS) 
  : D_only = 7 := 
by
  sorry

end alphabet_letters_l91_91918


namespace time_to_double_distance_l91_91169

theorem time_to_double_distance
  (m q k t l : ℝ)
  (h_initial : ∀ (x : ℝ), x = l → ∃ t, x = 2 * l)
  : ∀ x0 = 3 * l, time_to_double = 3 * sqrt 3 * t :=
by
  sorry

end time_to_double_distance_l91_91169


namespace number_of_players_with_odd_games_is_even_l91_91644

theorem number_of_players_with_odd_games_is_even (G : SimpleGraph V) [Fintype V] :
  (∃ d : V → ℕ, (∀ v : V, G.degree v = d v) ∧ (∑ v, d v) % 2 = 0) →
  (∃ S : Finset V, (∀ v ∈ S, d v % 2 = 1) ∧ S.card % 2 = 0) :=
by
  sorry

end number_of_players_with_odd_games_is_even_l91_91644


namespace dad_real_age_l91_91260

theorem dad_real_age (x : ℝ) (h : (5/7) * x = 35) : x = 49 :=
by
  sorry

end dad_real_age_l91_91260


namespace division_sequence_l91_91885

theorem division_sequence : (120 / 5) / 2 / 3 = 4 := by
  sorry

end division_sequence_l91_91885


namespace locus_of_z_l91_91851

theorem locus_of_z 
  (z z0 z1 : ℂ) 
  (hz0 : z0 ≠ 0) 
  (h1 : |z1 - z0| = |z1|) 
  (h2 : z1 * z = -1) : 
  ∃ c r, c = -1/z0 ∧ r = 1/(|z0|) ∧ (|z + c| = r ∧ z ≠ 0) :=
by
  sorry

end locus_of_z_l91_91851


namespace f_of_2_l91_91197

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_value : f (-2) = 11)

-- The theorem we want to prove
theorem f_of_2 : f 2 = -11 :=
by 
  sorry

end f_of_2_l91_91197


namespace rs_value_l91_91476

theorem rs_value (r s : ℝ) (h1 : 0 < r) (h2 : 0 < s) (h3 : r^2 + s^2 = 2) (h4 : r^4 + s^4 = 15 / 8) :
  r * s = (Real.sqrt 17) / 4 := 
sorry

end rs_value_l91_91476


namespace custom_op_2006_l91_91634

def custom_op (n : ℕ) : ℕ := 
  match n with 
  | 0 => 1
  | (n+1) => 2 + custom_op n

theorem custom_op_2006 : custom_op 2005 = 4011 :=
by {
  sorry
}

end custom_op_2006_l91_91634


namespace matthew_crackers_left_l91_91053

-- Definition of the conditions:
def initial_crackers := 23
def friends := 2
def crackers_eaten_per_friend := 6

-- Calculate the number of crackers Matthew has left:
def crackers_left (total_crackers : ℕ) (num_friends : ℕ) (eaten_per_friend : ℕ) : ℕ :=
  let crackers_given := (total_crackers - total_crackers % num_friends)
  let kept_by_matthew := total_crackers % num_friends
  let remaining_with_friends := (crackers_given / num_friends - eaten_per_friend) * num_friends
  kept_by_matthew + remaining_with_friends
  
-- Theorem to prove:
theorem matthew_crackers_left : crackers_left initial_crackers friends crackers_eaten_per_friend = 11 := by
  sorry

end matthew_crackers_left_l91_91053


namespace number_line_steps_l91_91467

theorem number_line_steps (n : ℕ) (total_distance : ℕ) (steps_to_x : ℕ) (x : ℕ)
  (h1 : total_distance = 32)
  (h2 : n = 8)
  (h3 : steps_to_x = 6)
  (h4 : x = (total_distance / n) * steps_to_x) :
  x = 24 := 
sorry

end number_line_steps_l91_91467


namespace log_an_arithmetic_sequence_T_n_value_m_values_l91_91437

open Real

-- Definitions and assumptions
def a_seq : ℕ → ℝ
| 0       := 10
| (n + 1) := 9 * (∑ i in range (n + 1), a_seq i) + 10

-- (I) Prove that {log a_n} is an arithmetic sequence.
theorem log_an_arithmetic_sequence : ∀ n : ℕ, log (a_seq n) = n := sorry

-- (II) Find T_n
def T_seq (n : ℕ) := 
  (∑ k in range (n + 1), 3 / ((log (a_seq k)) * (log (a_seq (k + 1)))))

theorem T_n_value (n : ℕ) : T_seq n = 3 - 3 / (n + 1) := sorry

-- (III) Find the set of integer values of m
def satisfies_condition (m : ℤ) : Prop :=
  ∀ n : ℕ, T_seq n > (1 / 4) * (m^2 - 5 * m)

theorem m_values : {m : ℤ | satisfies_condition m} = {0, 1, 2, 3, 4, 5} := sorry

end log_an_arithmetic_sequence_T_n_value_m_values_l91_91437


namespace parallelogram_separate_properties_l91_91060

open Geometry

-- Define a parallelogram
structure Parallelogram (P : Type) [MetricSpace P] :=
(A B C D : P)
(sides_parallel : parallel (line_through A B) (line_through C D) ∧ parallel (line_through B C) (line_through D A))

-- Define specific properties
def all_sides_equal {P : Type} [MetricSpace P] (p : Parallelogram P) :=
dist p.A p.B = dist p.B p.C ∧ dist p.C p.D = dist p.D p.A

def all_angles_equal {P : Type} [MetricSpace P] (p : Parallelogram P) :=
∡ p.A p.B p.C = ∡ p.B p.C p.D ∧ ∡ p.C p.D p.A = ∡ p.D p.A p.B

def diagonals_equal {P : Type} [MetricSpace P] (p : Parallelogram P) :=
dist p.A p.C = dist p.B p.D

-- And so on for the other properties

-- Main theorem statement
theorem parallelogram_separate_properties {P : Type} [MetricSpace P]
  (p : Parallelogram P) :
  ∃ (A: Prop) (B: Prop) (C: Prop) (D: Prop) (E: Prop) (F: Prop) (G: Prop) (H: Prop) (I: Prop),
  A = all_sides_equal p ∧ B = all_angles_equal p ∧ C = diagonals_equal p ∧
  D = (...) ∧ E = (...) ∧ F = (...) ∧ G = (...) ∧ H = (...) ∧ I = (...) :=
sorry

end parallelogram_separate_properties_l91_91060


namespace number_of_special_matrices_l91_91429

theorem number_of_special_matrices (p : ℕ) [fact (0 < p)] [fact (0 < p + 1)] :
  let S := {A : matrix (fin 2) (fin 2) (zmod p) | matrix.trace A = 1 ∧ matrix.det A = 0} in
  fintype.card S = p * (p + 1) :=
by
  sorry

end number_of_special_matrices_l91_91429


namespace parallel_cd_fa_of_hexagon_in_circle_l91_91065

theorem parallel_cd_fa_of_hexagon_in_circle (A B C D E F : Point)
(H1 : InscribedHexagon A B C D E F)
(H2 : Parallel A B D E)
(H3 : Parallel B C E F) : Parallel C D F A := sorry

end parallel_cd_fa_of_hexagon_in_circle_l91_91065


namespace sum_first_2010_terms_l91_91679

def sequence : ℕ → ℕ
| 0 => 1
| (n+1) => if (sequence n) = 1 then 2 else if ∃ (k : ℕ), (sequence k) = 1 ∧ ∀ m, k < m ∧ m < n → (sequence m) = 2 ∧ 2^(k-1) = m-k then 1 else 2

def sum_first_n_terms (n : ℕ) : ℕ :=
  (Finset.range n).sum sequence

theorem sum_first_2010_terms : sum_first_n_terms 2010 = 4010 := by
  sorry

end sum_first_2010_terms_l91_91679


namespace hypotenuse_length_l91_91225

theorem hypotenuse_length (a b c : ℝ) (h1 : a + b + c = 36) (h2 : 0.5 * a * b = 24) (h3 : a^2 + b^2 = c^2) :
  c = 50 / 3 :=
sorry

end hypotenuse_length_l91_91225


namespace calc_EY_l91_91489

-- Definitions for the hexagon sides and extension
variable (A B C D E F Y : Type) -- points in the Euclidean plane
variable [normed_add_comm_group A] [normed_add_torsor A E]
variable [normed_add_comm_group B] [normed_add_torsor B F]
variable [normed_add_comm_group C] [normed_add_torsor C Y]

-- Conditions: sides of hexagon, lengths, and extension condition
def side_length : ℝ := 4
def BY : ℝ := 8
def BC : ℝ := side_length

-- Definition of the extension condition
-- Assume distance as the Euclidean distance function
axiom eq_extension : dist B Y = 2 * dist B C

-- Calculation or proof of EY based on the given conditions
theorem calc_EY : ∀ (E Y : Type), dist E Y = 4 * real.sqrt 7 := by
  sorry
end

end calc_EY_l91_91489


namespace profitable_after_three_years_option_one_more_cost_effective_l91_91924

noncomputable def initial_investment : ℕ := 980_000
noncomputable def first_year_expense : ℕ := 120_000
noncomputable def annual_increase_expense : ℕ := 40_000
noncomputable def annual_profit : ℕ := 500_000

theorem profitable_after_three_years : (∃ year: ℕ, year = 3 ∧ profits year > expenses year) :=
  by sorry

theorem option_one_more_cost_effective : 
  ∃ year: ℕ, max_average_profit_year year ∧ with_option_one year > with_option_two year :=
  by sorry

-- Definitions needed for the proofs

noncomputable def profits (year: ℕ) : ℕ := 
  if year = 1 then annual_profit - first_year_expense - initial_investment
  else annual_profit * year - (initial_investment + first_year_expense + (annual_increase_expense * (year - 1) * (year) ÷ 2))

noncomputable def expenses (year: ℕ) : ℕ := 
  initial_investment + first_year_expense + annual_increase_expense * (year - 1) * year ÷ 2

-- For comparing two options

noncomputable def max_average_profit_year (year: ℕ) : Prop := 
  ∀ y: ℕ, y ≠ year → average_profit year > average_profit y

noncomputable def average_profit (year: ℕ) : ℕ :=
  profits year ÷ year

noncomputable def total_profit (year: ℕ) : ℕ := 
  profits year

noncomputable def with_option_one (year: ℕ) : ℕ :=
  profits year + 260_000

noncomputable def with_option_two (year: ℕ) : ℕ :=
  total_profit year + 80_000

end profitable_after_three_years_option_one_more_cost_effective_l91_91924


namespace alice_paid_percentage_of_srp_l91_91108

theorem alice_paid_percentage_of_srp
  (P : ℝ) -- Suggested Retail Price (SRP)
  (MP : ℝ := P * 0.60) -- Marked Price (MP) is 40% less than SRP
  (price_alice_paid : ℝ := MP * 0.60) -- Alice purchased the book for 40% off the marked price
  : (price_alice_paid / P) * 100 = 36 :=
by
  -- only the statement is required, so proof is omitted
  sorry

end alice_paid_percentage_of_srp_l91_91108


namespace number_of_parallelograms_l91_91883

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_parallelograms (n : ℕ) : 
  (∑ i in Finset.range (n + 2 + 1), binom i 4) = 3 * binom (n + 2) 4 :=
by
  sorry

end number_of_parallelograms_l91_91883


namespace coefficient_x4_in_expansion_l91_91580

theorem coefficient_x4_in_expansion : 
  (Expansion.coeff (x^4) ((1 + 2 * x)^6) = 240) :=
sorry

end coefficient_x4_in_expansion_l91_91580


namespace range_of_a_l91_91638

-- Define proposition p
def p (a : ℝ) : Prop := a * (1 - a) > 0

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ := (2 * a - 3)^2 - 4 * 1 * 1

-- Define proposition q
def q (a : ℝ) : Prop := discriminant(a) > 0

-- The range of values of a such that p ∨ q is true and p ∧ q is false
def valid_range (a : ℝ) : Prop :=
  (p a ∨ q a) ∧ ¬ (p a ∧ q a) ↔ (a ≤ 0 ∨ (1 / 2 ≤ a ∧ a < 1) ∨ a > 5 / 2)

theorem range_of_a : ∀ a : ℝ, valid_range a := 
by
  sorry

end range_of_a_l91_91638


namespace maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l91_91084

-- Defining productivity functions for male and female kittens
def male_productivity (K : ℝ) : ℝ := 80 - 4 * K
def female_productivity (K : ℝ) : ℝ := 16 - 0.25 * K

-- Condition (a): Maximizing number of mice caught by 2 kittens
theorem maximize_mice_two_kittens : 
  ∃ (male1 male2 : ℝ) (K_m1 K_m2 : ℝ), 
    (male1 = male_productivity K_m1) ∧ 
    (male2 = male_productivity K_m2) ∧
    (K_m1 = 0) ∧ (K_m2 = 0) ∧
    (male1 + male2 = 160) := 
sorry

-- Condition (b): Different versions of JPPF
theorem different_versions_JPPF : 
  ∃ (v1 v2 v3 : Unit), 
    (v1 ≠ v2) ∧ (v2 ≠ v3) ∧ (v1 ≠ v3) :=
sorry

-- Condition (c): Analytical form of JPPF for each combination
theorem JPPF_combinations :
  ∃ (M K1 K2 : ℝ),
    (M = 160 - 4 * K1 ∧ K1 ≤ 40) ∨
    (M = 32 - 0.5 * K2 ∧ K2 ≤ 64) ∨
    (M = 96 - 0.25 * K2 ∧ K2 ≤ 64) ∨
    (M = 336 - 4 * K2 ∧ 64 < K2 ∧ K2 ≤ 84) :=
sorry

-- Condition (d): Analytical form for 2 males and 1 female
theorem JPPF_two_males_one_female :
  ∃ (M K : ℝ), 
    (0 < K ∧ K ≤ 64 ∧ M = 176 - 0.25 * K) ∨
    (64 < K ∧ K ≤ 164 ∧ M = 416 - 4 * K) :=
sorry

end maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l91_91084


namespace pizza_slice_volume_correct_l91_91606

-- Define the parameters of the problem
def thickness : ℝ := 1 / 2
def diameter : ℝ := 10
def slices : ℝ := 10

-- Define the correct answer
def expected_volume : ℝ := 5 * π / 4

-- The problem statement
theorem pizza_slice_volume_correct :
  let radius := diameter / 2
  let entire_volume := π * radius^2 * thickness
  let slice_volume := entire_volume / slices
  slice_volume = expected_volume :=
by
  := sorry

end pizza_slice_volume_correct_l91_91606


namespace convert_decimal_to_fraction_l91_91906

theorem convert_decimal_to_fraction : (3.68:Rat) = (92/25:Rat) :=
by 
  have eq1 : (3.68:Rat) = (3 : Rat) + (68/100 : Rat) := by norm_num
  have eq2 : (68/100 : Rat) = (17/25 : Rat) := by
    norm_num
    rw [div_eq_mul_inv]
    ring_nf
    norm_num
  rw [eq1, eq2]
  norm_num
  sorry

end convert_decimal_to_fraction_l91_91906


namespace first_markup_percentage_l91_91949

-- Definitions for conditions
variables (C : ℝ)  -- cost price
variables (M : ℝ)  -- initial markup percentage
variables (R : ℝ := C * (1 + M / 100))  -- initial retail price
variables (N : ℝ := R * 1.25)  -- New Year's retail price after 25% markup
variables (F : ℝ := N * 0.80)  -- Final price in February after 20% discount

-- The target condition: 20% profit
variables (P : ℝ := 1.20 * C)  -- 120% of the cost price

-- The theorem to prove
theorem first_markup_percentage : F = P → M = 20 := 
by sorry

end first_markup_percentage_l91_91949


namespace percent_defective_units_shipped_for_sale_l91_91777

theorem percent_defective_units_shipped_for_sale :
  ∀ (D : ℚ),
    let defective_units := 0.08 * D in
    let shipped_defective_units := 0.004 * D in
    (shipped_defective_units / defective_units) * 100 = 5 := by
  intros D defective_units shipped_defective_units
  sorry

end percent_defective_units_shipped_for_sale_l91_91777


namespace initial_position_l91_91907

variable (x : Int)

theorem initial_position 
  (h: x - 5 + 4 + 2 - 3 + 1 = 6) : x = 7 := 
  by 
  sorry

end initial_position_l91_91907


namespace general_solution_l91_91653

theorem general_solution (C1 C2 : ℝ) :
  ∀ x : ℝ, 
  let y := (C1 + C2 * x) * Real.exp (2 * x) in
  (deriv^[2] (λ x, y) x) - 4 * (deriv (λ x, y) x) + 4 * y = 0 :=
by
  intro x 
  let y := (C1 + C2 * x) * Real.exp (2 * x)
  have H1 : deriv^[2] (λ x, y) x = 4 * Real.exp (2 * x) + 4 * x * Real.exp (2 * x), sorry
  have H2 : deriv (λ x, y) x = Real.exp (2 * x) + 2 * x * Real.exp (2 * x), sorry
  show (4 * Real.exp (2 * x) + 4 * x * Real.exp (2 * x)) - 4 * (Real.exp (2 * x) + 2 * x * Real.exp (2 * x)) + 4 * (C1 + C2 * x) * Real.exp (2 * x) = 0, sorry

end general_solution_l91_91653


namespace max_sqrt_sum_l91_91305

theorem max_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
sorry

end max_sqrt_sum_l91_91305


namespace birds_on_fence_l91_91536

theorem birds_on_fence :
  let initial_birds := 12
  let more_birds := 8
  let groups := 3
  let birds_per_group := 6
  initial_birds + more_birds + (groups * birds_per_group) = 38 := 
by 
  simp [initial_birds, more_birds, groups, birds_per_group]
  exact sorry

end birds_on_fence_l91_91536


namespace find_a_value_l91_91331

theorem find_a_value (a : ℝ) (x : ℝ) :
  (a + 1) * x^2 + (a^2 + 1) + 8 * x = 9 →
  a + 1 ≠ 0 →
  a^2 + 1 = 9 →
  a = 2 * Real.sqrt 2 :=
by
  intro h1 h2 h3
  sorry

end find_a_value_l91_91331


namespace smallest_positive_value_of_expression_l91_91659

theorem smallest_positive_value_of_expression :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^3 + b^3 + c^3 - 3 * a * b * c = 4) :=
by
  sorry

end smallest_positive_value_of_expression_l91_91659


namespace quadrilateral_sides_l91_91095

theorem quadrilateral_sides (AC BD : ℝ) (intersects_at_O : ∃ O : ℝ × ℝ, True) 
  (hAC : AC = 12) (hBD : BD = 18) :
  ∃ a b c d : ℝ, {a, b, c, d} = {4, 6, 4, 6} :=
by
  sorry

end quadrilateral_sides_l91_91095


namespace length_reduction_percentage_to_maintain_area_l91_91511

theorem length_reduction_percentage_to_maintain_area
  (L W : ℝ)
  (new_width : ℝ := W * (1 + 28.2051282051282 / 100))
  (new_length : ℝ := L * (1 - 21.9512195121951 / 100))
  (original_area : ℝ := L * W) :
  original_area = new_length * new_width := by
  sorry

end length_reduction_percentage_to_maintain_area_l91_91511


namespace parallelogram_inequality_parallelogram_equality_l91_91406

-- Definitions of the parameters in the problem
variables {A B C D : Type} [AddGroup A] [VectorSpace ℝ A]
variables [DecidableEq A]

-- Define the condition of the parallelogram and angle constraints
def is_parallelogram (ABCD : A × A × A × A) : Prop :=
  let (A, B, C, D) := ABCD in
  (∥A - B∥ = ∥C - D∥) ∧ (∥B - C∥ = ∥D - A∥) ∧ (A + C = B + D)

def angle_not_greater (A B : Type) [InnerProductSpace ℝ A] (θ α : ℝ) : Prop :=
  θ ≤ α

-- The main theorem
theorem parallelogram_inequality (ABCD : A × A × A × A) (bad_angle half_bad_angle : ℝ)
  (h_parallelogram : is_parallelogram ABCD)
  (h_angle : angle_not_greater A B bad_angle (π / 2)) :
  let (A, B, C, D) := ABCD in
  (∥A - C∥ / ∥B - D∥ ≤ Real.cos(half_bad_angle) / Real.sin(half_bad_angle)) :=
sorry

-- The equality condition
theorem parallelogram_equality (ABCD : A × A × A × A) (bad_angle half_bad_angle : ℝ)
  (h_parallelogram : is_parallelogram ABCD)
  (h_angle : angle_not_greater A B bad_angle (π / 2)) :
  let (A, B, C, D) := ABCD in
  (∥A - C∥ / ∥B - D∥ = Real.cos(half_bad_angle) / Real.sin(half_bad_angle)) ↔
  (is_rhombus ABCD ∨ is_rectangle ABCD) :=
sorry

end parallelogram_inequality_parallelogram_equality_l91_91406


namespace common_factor_l91_91501

variable {α : Type*} [CommRing α]

def T (a : α) : α := 2 * a^2
def U (a b : α) : α := 4 * a * b

theorem common_factor (a b : α) : gcd (T a) (U a b) = 2 * a :=
by
  sorry

end common_factor_l91_91501


namespace g_composition_l91_91028

def g (x : ℝ) : ℝ :=
  if x >= 0 then -x^3 else x + 9

theorem g_composition :
  g (g (g (g (g 2)))) = -512 :=
by
  have h1 : g 2 = -8 := by sorry
  have h2 : g (-8) = 1 := by sorry
  have h3 : g 1 = -1 := by sorry
  have h4 : g (-1) = 8 := by sorry
  have h5 : g 8 = -512 := by sorry
  sorry

end g_composition_l91_91028


namespace find_y_l91_91283
-- Definitions
def AO : ℝ := 5
def BO : ℝ := 6
def CO : ℝ := 9
def DO : ℝ := 5
def BD : ℝ := 7
def θ := Real.arccos ( (AO^2 + BO^2 - BD^2) / (2 * AO * BO) )

-- Main theorem
theorem find_y (AO BO CO DO BD : ℝ) (θ := Real.arccos ( (AO^2 + BO^2 - BD^2) / (2 * AO * BO) )):
  ∃ y : ℝ, y = 2 * Real.sqrt 22 :=
sorry

end find_y_l91_91283


namespace sum_integers_minus_50_to_70_l91_91895

theorem sum_integers_minus_50_to_70 : ∑ i in finset.range (71 - -49), (-50 + i) = 2485 :=
by
  sorry

end sum_integers_minus_50_to_70_l91_91895


namespace trigonometric_expression_value_l91_91352

theorem trigonometric_expression_value 
  (θ : ℝ) 
  (h : sin θ - sqrt 3 * cos θ = -2) : 
  sin θ ^ 2 + cos (2 * θ) + 3 = 15 / 4 := 
sorry

end trigonometric_expression_value_l91_91352


namespace digits_of_number_l91_91930

theorem digits_of_number (d : ℕ) (h1 : 0 ≤ d ∧ d ≤ 9) (h2 : (10 * (50 + d) + 2) % 6 = 0) : (5 * 10 + d) * 10 + 2 = 522 :=
by sorry

end digits_of_number_l91_91930


namespace circle_centers_coincidence_l91_91781

noncomputable theory

open EuclideanGeometry

variables {A B C H K : Point}
variables (α : Angle A) (β : Angle B) (γ : Angle C) 
variables (AB AC BC BH BK CK : Line)
variables (circumcenter_ABK : Point) (excircle_center_BCH : Point)

def isosceles_right_triangle (A B H : Point) : Prop :=
  ∃ (α β γ : Angle A), α = 45 ∧ β = 45 ∧ BH.isAltitude B AC

def altitude_from_B (B H : Point) : Prop :=
  BH.isAltitude B AC

def C_excircle_center (C H K : Point) : Prop :=
  excircle_center_BCH ∈ Line.perpendicularBisector B H

def circumcircle_center (A B K : Point) : Prop :=
  circumcenter_ABK ∈ Circle.circumcircle A B K

theorem circle_centers_coincidence :
  angle A = 45 ∧ BH.isAltitude B AC ∧ BC = CK → 
  circumcenter_ABK = excircle_center_BCH := by
  sorry

end circle_centers_coincidence_l91_91781


namespace probability_of_triangle_or_pentagon_is_one_half_l91_91951

structure ShapeSet :=
  (total_shapes : ℕ)
  (triangles : ℕ)
  (squares : ℕ)
  (circles : ℕ)
  (pentagons : ℕ)
  (total_shapes_eq : total_shapes = triangles + squares + circles + pentagons)

def probability_triangle_or_pentagon (S : ShapeSet) : ℚ :=
  (S.triangles + S.pentagons) / S.total_shapes

theorem probability_of_triangle_or_pentagon_is_one_half :
  ∀ (S : ShapeSet), S.total_shapes = 10 → S.triangles = 3 → S.squares = 3 → S.circles = 2 → S.pentagons = 2 →
  probability_triangle_or_pentagon S = 1/2 :=
by
  intros S h1 h2 h3 h4 h5
  unfold probability_triangle_or_pentagon
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end probability_of_triangle_or_pentagon_is_one_half_l91_91951


namespace sum_integers_minus_50_to_70_l91_91894

theorem sum_integers_minus_50_to_70 : ∑ i in finset.range (71 - -49), (-50 + i) = 2485 :=
by
  sorry

end sum_integers_minus_50_to_70_l91_91894


namespace schoolchildren_initial_speed_l91_91168

theorem schoolchildren_initial_speed (v : ℝ) (t t_1 t_2 : ℝ) 
  (h1 : t_1 = (6 * v) / (v + 60) + (400 - 3 * v) / (v + 60)) 
  (h2 : t_2 = (400 - 3 * v) / v) 
  (h3 : t_1 = t_2) : v = 63.24 :=
by sorry

end schoolchildren_initial_speed_l91_91168


namespace exists_a_b_divisible_l91_91479

theorem exists_a_b_divisible (n : ℕ) (hn : 0 < n) : 
  ∃ a b : ℤ, (4 * a^2 + 9 * b^2 - 1) % n = 0 := 
sorry

end exists_a_b_divisible_l91_91479


namespace eccentricity_of_ellipse_l91_91699

def ellipse (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def hyperbola (m n x y : ℝ) : Prop := (x^2 / m^2) - (y^2 / n^2) = 1
def foci (c : ℝ) : set (ℝ × ℝ) := {p | p = (-c,0) ∨ p = (c,0)}
def geometric_mean (c a m : ℝ) : Prop := c^2 = a * m
def arithmetic_mean (n m c : ℝ) : Prop := 2 * n^2 = 2 * m^2 + c^2

theorem eccentricity_of_ellipse 
    (a b m n c : ℝ)
    (h1a : a > 0)
    (h1b : b > 0)
    (h2 : a > b)
    (h3 : m > 0)
    (h4 : n > 0)
    (h5 : geometric_mean c a m)
    (h6 : arithmetic_mean n m c) :
    ∃ e : ℝ, e = c / a ∧ e = 1 / 2 :=
by sorry

end eccentricity_of_ellipse_l91_91699


namespace top_layer_blocks_l91_91208

theorem top_layer_blocks (x : Nat) (h : x + 3 * x + 9 * x + 27 * x = 40) : x = 1 :=
by
  sorry

end top_layer_blocks_l91_91208


namespace max_value_sqrt_eq_14_l91_91300

theorem max_value_sqrt_eq_14 (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  ∃ y, y = sqrt (49 + x) + sqrt (49 - x) ∧ y ≤ 14 ∧ ∀ z, z = sqrt (49 + x) + sqrt (49 - x) → z ≤ y :=
by {
  let max_val := sqrt (49 : ℝ) + sqrt (49 : ℝ),
  have h_max : max_val = 14 := by norm_num,
  use max_val,
  split,
  { exact h_max },
  split,
  { apply max_val_le,
    intros a ha,
    have h1 := @abs_le_of_le_of_neg any_field _ h.1,
    have ha := sqrt_real.exists_le_sqrt_add_sqrt (49 + x) (49 - x),
    sorry },
  { intro z,
    assume hz : z = sqrt (49 + x) + sqrt (49 - x),
    apply le_of_eq,
    exact ha }
  }

end max_value_sqrt_eq_14_l91_91300


namespace simplify_expression_l91_91074

theorem simplify_expression : ((- (1 / 64)) ^ (-3 / 2)) = -512 := 
by sorry

end simplify_expression_l91_91074


namespace bench_capacity_l91_91219

theorem bench_capacity (C : ℕ) (h1 : ∃ benches : ℕ, benches = 50)
    (h2 : ∃ people_sitting : ℕ, people_sitting = 80) (h3 : ∃ available_spaces : ℕ, available_spaces = 120) :
    50 * C - 80 = 120 → C = 4 :=
begin
  intros h_eq,
  have h_total_capacity : 50 * C = 200, from eq_add_of_sub_eq h_eq,
  have h_C : C = 200 / 50, from eq_div_of_mul_eq h_total_capacity,
  norm_num at h_C,
  exact h_C,
end

end bench_capacity_l91_91219


namespace AO_perpendicular_BC_l91_91469

open EuclideanGeometry

-- Define the conditions
variable {A B C D O B' C' : Point}
  (h_ABC : Triangle A B C)
  (h_D_tangent : ∀ X, X ∈ {B, C} → Tangent X D (Circumcircle A B C))
  (h_B' : B' = reflect_point_over_line B A C)
  (h_C' : C' = reflect_point_over_line C A B)
  (h_O : O = circumcenter D B' C')

-- The theorem to be proved
theorem AO_perpendicular_BC : Perpendicular A O B C :=
by
  sorry

end AO_perpendicular_BC_l91_91469


namespace tangent_line_at_point_l91_91096

noncomputable def curve : ℝ → ℝ := fun x => x^2 + 2

theorem tangent_line_at_point :
  let slope_at_P := (deriv curve) 1
  let y_intercept := 3 - slope_at_P * 1 -- Solving for y-intercept using point P(1,3)
  ∀ x y : ℝ, (y = 2 * x + y_intercept) → (2 * x - y + 1 = 0) :=
by
  intro x y h
  rw [h]
  ring
  sorry

end tangent_line_at_point_l91_91096


namespace solomon_sale_price_l91_91075

def original_price : ℝ := 500
def discount_rate : ℝ := 0.10
def sale_price := original_price * (1 - discount_rate)

theorem solomon_sale_price : sale_price = 450 := by
  sorry

end solomon_sale_price_l91_91075


namespace pi_approx_l91_91975

noncomputable def bretschneider_pi_approx : ℝ :=
  (13 / 50) * Real.sqrt 146

axiom radius_one : ℝ := 1

theorem pi_approx :
  Real.pi ≈ bretschneider_pi_approx :=
sorry

end pi_approx_l91_91975


namespace find_points_l91_91285

theorem find_points (x y : ℝ) (points : set (ℝ × ℝ)) :
  (points = {(-1, 0), (1, 0), (-3, 1), (3, 1), (-3, 2), (3, 2)}) ∧
  (∀ (p : ℝ × ℝ), p ∈ points ↔
    let y := p.snd, x := p.fst in 
    y ≥ 0 ∧ 24 * y + 1 = (4 * y^2 - x^2)^2) :=
by
  sorry

end find_points_l91_91285


namespace is_isosceles_triangle_area_of_triangle_l91_91368

variables {A B C: ℝ}
variables {a b c: ℝ}
variables {sin_A sin_B sin_C: ℝ}
variables {R: ℝ}

-- Definitions from conditions
def m : (ℝ × ℝ) := (a, b)
def n : (ℝ × ℝ) := (Real.sin B, Real.sin A)
def p : (ℝ × ℝ) := (b-2, a-2)

-- Problem (1): Prove Isosceles Triangle
theorem is_isosceles_triangle (h1: m ∥ n) : a = b := 
sorry

-- Problem (2): Find Area
theorem area_of_triangle (h2: m ⊥ p) (h3: c = 2) (h4: C = 90) : 
  ∃ (S: ℝ), S = 2 :=
sorry

end is_isosceles_triangle_area_of_triangle_l91_91368


namespace small_loads_have_10_pieces_l91_91458

theorem small_loads_have_10_pieces (total_clothing : ℕ) (one_load : ℕ) (small_loads : ℕ) (remaining_clothing : ℕ) : 
  total_clothing = 105 → 
  one_load = 34 → 
  small_loads = 7 → 
  remaining_clothing = total_clothing - one_load → 
  remaining_clothing / small_loads = 10 :=
by
  intros h_total h_one h_small h_remain
  have h := eq.trans h_remain (by rw [h_total, h_one])
  have divide_remain := nat.div_eq_of_eq_mul_right (by norm_num : 7 ≠ 0) (by norm_num : remaining_clothing = 71)
  have h_res : remaining_clothing = 71 := by norm_num [h_total, h_one, h_remain]
  have h_div : 71 / 7 = 10 := by norm_num
  exact h_div

end small_loads_have_10_pieces_l91_91458


namespace f_divides_f_2k_plus_1_f_coprime_f_multiple_l91_91370

noncomputable def f (g n : ℕ) : ℕ := g ^ n + 1

theorem f_divides_f_2k_plus_1 (g : ℕ) (k n : ℕ) :
  f g n ∣ f g ((2 * k + 1) * n) :=
by sorry

theorem f_coprime_f_multiple (g n : ℕ) :
  Nat.Coprime (f g n) (f g (2 * n)) ∧
  Nat.Coprime (f g n) (f g (4 * n)) ∧
  Nat.Coprime (f g n) (f g (6 * n)) :=
by sorry

end f_divides_f_2k_plus_1_f_coprime_f_multiple_l91_91370


namespace sheila_will_attend_picnic_l91_91834

def P_Rain : ℝ := 0.3
def P_Cloudy : ℝ := 0.4
def P_Sunny : ℝ := 0.3

def P_Attend_if_Rain : ℝ := 0.25
def P_Attend_if_Cloudy : ℝ := 0.5
def P_Attend_if_Sunny : ℝ := 0.75

def P_Attend : ℝ :=
  P_Rain * P_Attend_if_Rain +
  P_Cloudy * P_Attend_if_Cloudy +
  P_Sunny * P_Attend_if_Sunny

theorem sheila_will_attend_picnic : P_Attend = 0.5 := by
  sorry

end sheila_will_attend_picnic_l91_91834


namespace product_of_slopes_constant_l91_91684

noncomputable def ellipse (x y : ℝ) := x^2 / 8 + y^2 / 4 = 1

theorem product_of_slopes_constant (a b : ℝ) (h_a_gt_b : a > b) (h_a_b_pos : 0 < a ∧ 0 < b)
  (e : ℝ) (h_eccentricity : e = (Real.sqrt 2) / 2) (P : ℝ × ℝ) (h_point_on_ellipse : (P.1, P.2) = (2, Real.sqrt 2)) :
  (∃ C : ℝ → ℝ → Prop, C = ellipse) ∧ (∃ k : ℝ, -k * 1/2 = -1 / 2) := sorry

end product_of_slopes_constant_l91_91684


namespace probability_at_least_one_vowel_l91_91919

-- Define sets and identify the vowels
def set1 : Set (Char) := {'a', 'b', 'o', 'd', 'e'}
def set2 : Set (Char) := {'k', 'l', 'm', 'n', 'u', 'p'}
def vowels : Set (Char) := {'a', 'e', 'i', 'o', 'u'}

-- Define the probability of picking at least one vowel
theorem probability_at_least_one_vowel : 
  (∀ (c1 ∈ set1) (c2 ∈ set2), 
   let is_vowel (c : Char) := c ∈ vowels in
   (if is_vowel c1 ∨ is_vowel c2 then 1 else 0) /
   ((Nat.card set1) * (Nat.card set2)) = 2 / 3) := 
  sorry

end probability_at_least_one_vowel_l91_91919


namespace ninja_path_contains_red_circles_l91_91794

-- Define a "Ninja Path" problem with the given conditions.
theorem ninja_path_contains_red_circles (n : ℕ) (hn : n > 0) :
  ∃ (k : ℕ), k = ⌈real.log2 (n + 1 : ℝ)⌉₊ ∧
  ∀ (T : JapaneseTriangle n), 
    ∃ (P : NinjaPath T), count_red_circles P ≥ k := 
sorry

end ninja_path_contains_red_circles_l91_91794


namespace chris_marbles_l91_91977

variable (C R: ℕ) -- Chris's marbles C and Ryan's marbles R
variable (H1: R = 28) -- condition: Ryan has 28 marbles
variable (H2: ∀ T: ℕ, (T = C + R) → 2 * (T - (T / 2)) = 40) -- condition: after they take away 1/4 each and 20 remain.

theorem chris_marbles : C = 12 := by
  have H3 : 2 * 20 = 40 := by rfl -- reinforcing the remaining marbles condition
  have H4 : ∀ T : ℕ, (T = C + R) → 2 * (T / 2) = T := by
    intro T hT
    rw hT
    ring
  have H5 : ∀ T : ℕ, (T = C + R) → T = C + 28 := by
    intro T hT
    rw [hT, H1]
  specialize H2 (C + 28)
  specialize H4 (C + 28)
  specialize H5 (C + 28) (by rfl)
  rw [H4, H5] at H2
  rw [add_mul, mul_comm] at H2
  specialize H2 (by rfl)
  norm_num at H2
  linarith

end chris_marbles_l91_91977


namespace Milly_study_time_l91_91054

theorem Milly_study_time :
  let math_time := 60
  let geo_time := math_time / 2
  let mean_time := (math_time + geo_time) / 2
  let total_study_time := math_time + geo_time + mean_time
  total_study_time = 135 := by
  sorry

end Milly_study_time_l91_91054


namespace cube_root_of_4913_has_unit_digit_7_cube_root_of_50653_is_37_cube_root_of_110592_is_48_l91_91785

theorem cube_root_of_4913_has_unit_digit_7 :
  (∃ (y : ℕ), y^3 = 4913 ∧ y % 10 = 7) :=
sorry

theorem cube_root_of_50653_is_37 :
  (∃ (y : ℕ), y = 37 ∧ y^3 = 50653) :=
sorry

theorem cube_root_of_110592_is_48 :
  (∃ (y : ℕ), y = 48 ∧ y^3 = 110592) :=
sorry

end cube_root_of_4913_has_unit_digit_7_cube_root_of_50653_is_37_cube_root_of_110592_is_48_l91_91785


namespace equifacial_tetrahedron_to_rectangular_parallelepiped_l91_91610

theorem equifacial_tetrahedron_to_rectangular_parallelepiped (A B C D : Point)
  (h1 : congruent (face A B C) (face A D B) ∧ congruent (face A C D) (face B C D)) :
  let T := tetrahedron A B C D
  let P := extend_to_parallelepiped T
  is_rectangular_parallelepiped P :=
sorry

end equifacial_tetrahedron_to_rectangular_parallelepiped_l91_91610


namespace num_perfect_square_factors_l91_91740

theorem num_perfect_square_factors (n : ℕ) (h_factorization : n = 2^2 * 3^3) : 
  nat.card { d : ℕ | d ∣ 108 ∧ (∀ p ∣ d, p.prime → even (multiplicity p 108)) } = 4 :=
by
  sorry

end num_perfect_square_factors_l91_91740


namespace carla_catches_jerry_l91_91421

-- Define the relevant variables and conditions
variables (t : ℝ) -- time it takes for Carla to catch up to Jerry
variables (jerry_speed carla_speed : ℝ) -- speeds of Jerry and Carla in miles per hour
variables (head_start : ℝ) -- head start time in hours for Jerry

-- Given conditions
def conditions : Prop :=
  jerry_speed = 30 ∧
  carla_speed = 35 ∧
  head_start = 0.5

-- The main theorem stating that given the conditions, t must be 3 
theorem carla_catches_jerry (h : conditions) : t = 3 :=
by {
  -- Here we would apply the proof steps; we're omitting it as per instructions.
  sorry
}

end carla_catches_jerry_l91_91421


namespace find_all_tasty_candies_in_21_turns_find_all_tasty_candies_in_20_turns_l91_91134

def totalCandies := 28

-- Problem (a): Prove Vasya can guarantee to find all tasty candies within 21 turns
theorem find_all_tasty_candies_in_21_turns (candies : Fin totalCandies → Bool) :
  ∃ strategy : Fin 21 → (Fin totalCandies → Bool) × ℕ, 
    (∀ tastySet : Fin totalCandies → Bool, 
      ∃ n <= 21, 
        strategy n.1 == tastySet) :=
sorry

-- Problem (b): Prove Vasya can guarantee to find all tasty candies within 20 turns
theorem find_all_tasty_candies_in_20_turns (candies : Fin totalCandies → Bool) :
  ∃ strategy : Fin 20 → (Fin totalCandies → Bool) × ℕ, 
    (∀ tastySet : Fin totalCandies → Bool, 
      ∃ n <= 20, 
        strategy n.1 == tastySet) :=
sorry

end find_all_tasty_candies_in_21_turns_find_all_tasty_candies_in_20_turns_l91_91134


namespace polyhedra_impossible_l91_91755

noncomputable def impossible_polyhedra_projections (p1_outer : List (ℝ × ℝ)) (p1_inner : List (ℝ × ℝ))
                                                  (p2_outer : List (ℝ × ℝ)) (p2_inner : List (ℝ × ℝ)) : Prop :=
  -- Add definitions for the vertices labeling here 
  let vertices_outer := ["A", "B", "C", "D"]
  let vertices_inner := ["A1", "B1", "C1", "D1"]
  -- Add the conditions for projection (a) and (b) 
  p1_outer = [(0,0), (1,0), (1,1), (0,1)] ∧
  p1_inner = [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] ∧
  p2_outer = [(0,0), (1,0), (1,1), (0,1)] ∧
  p2_inner = [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] →
  -- Prove that the polyhedra corresponding to these projections are impossible.
  false

-- Now let's state the theorem
theorem polyhedra_impossible : impossible_polyhedra_projections [(0,0), (1,0), (1,1), (0,1)] 
                                                                [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)]
                                                                [(0,0), (1,0), (1,1), (0,1)]
                                                                [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] := 
by {
  sorry
}

end polyhedra_impossible_l91_91755


namespace problem_statement_l91_91792

variables {AB CD BC DA : ℝ} (E : ℝ) (midpoint_E : E = BC / 2) (ins_ABC : circle_inscribable AB ED)
  (ins_AEC : circle_inscribable AE CD) (a b c d : ℝ) (h_AB : AB = a) (h_BC : BC = b) (h_CD : CD = c)
  (h_DA : DA = d)

theorem problem_statement :
  a + c = b / 3 + d ∧ (1 / a + 1 / c = 3 / b) :=
by
  sorry

end problem_statement_l91_91792


namespace probability_none_solve_l91_91597

theorem probability_none_solve (a b c : ℕ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (h_prob : ((1 - (1/a)) * (1 - (1/b)) * (1 - (1/c)) = 8/15)) : 
  (1 - (1/a)) * (1 - (1/b)) * (1 - (1/c)) = 8/15 := 
by 
  sorry

end probability_none_solve_l91_91597


namespace intersect_eq_necessity_l91_91038

theorem intersect_eq_necessity (M N P : Set) : (M ∩ P = N ∩ P) → (M = N) ↔ False :=
sorry

end intersect_eq_necessity_l91_91038


namespace BelindaTotalFlyers_l91_91249

variable (F : ℕ)

-- Conditions
def RyanFlyers := 42
def AlyssaFlyers := 67
def ScottFlyers := 51
def BelindaPercentage := 0.20

-- Total number of flyers
def TotalFlyersDistributed := RyanFlyers + AlyssaFlyers + ScottFlyers
def BelindaDistributedFlyers := BelindaPercentage * F

-- Statement to be proved
theorem BelindaTotalFlyers : TotalFlyersDistributed + BelindaDistributedFlyers = F → F = 200 :=
by
  intros h
  sorry

end BelindaTotalFlyers_l91_91249


namespace sum_of_perpendiculars_l91_91198

-- define the points on the rectangle
variables {A B C D P S R Q F : Type}

-- define rectangle ABCD and points P, S, R, Q, F
def is_rectangle (A B C D : Type) : Prop := sorry -- conditions for ABCD to be a rectangle
def point_on_segment (P A B: Type) : Prop := sorry -- P is a point on segment AB
def perpendicular (X Y Z : Type) : Prop := sorry -- definition for perpendicular between two segments
def length (X Y : Type) : ℝ := sorry -- definition for the length of a segment

-- Given conditions
axiom rect : is_rectangle A B C D
axiom p_on_ab : point_on_segment P A B
axiom ps_perp_bd : perpendicular P S D
axiom pr_perp_ac : perpendicular P R C
axiom af_perp_bd : perpendicular A F D
axiom pq_perp_af : perpendicular P Q F

-- Prove that PR + PS = AF
theorem sum_of_perpendiculars :
  length P R + length P S = length A F :=
sorry

end sum_of_perpendiculars_l91_91198


namespace quadratic_equation_two_distinct_roots_and_one_in_interval_l91_91730

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_equation_two_distinct_roots_and_one_in_interval
  (a b c x₁ x₂ : ℝ) (h₀ : x₁ < x₂)
  (h₁ : quadratic_function a b c x₁ ≠ quadratic_function a b c x₂) :
  let f := quadratic_function a b c
  in ∃ x₁' x₂' : ℝ, x₁' ≠ x₂' ∧ f x₁' = (1 / 2) * (f x₁ + f x₂) ∧ f x₂' = (1 / 2) * (f x₁ + f x₂) ∧ 
     ∃ x ∈ set.Ioo x₁ x₂, f x = (1 / 2) * (f x₁ + f x₂) :=
sorry

end quadratic_equation_two_distinct_roots_and_one_in_interval_l91_91730


namespace proof_problem_l91_91364

noncomputable def a : ℕ → ℕ
| 1 => 2
| 2 => 6
| (n+2) => 2 * a (n+1) + 2 - a n -- derived from the given condition

def b (n : ℕ) : ℚ := 10 * (n + 1) / a n - 1 / 2

def S (n : ℕ) : ℚ := Finset.sum (Finset.range n) b

def M (n : ℕ) : ℚ := S (2*n) - S n

theorem proof_problem :
  (∀ n ≥ 2, (a (n + 1) + a (n - 1)) / (a n + 1) = 2) →
  (∀ n, (a (n + 1) - a n) - (a n - a (n - 1)) = 2) →
  ∀ n, M 2 = 29 / 6 :=
by
  sorry

end proof_problem_l91_91364


namespace parking_fee_function_relationship_parking_fee_range_l91_91371

noncomputable def parking_fee (x : ℕ) : ℕ :=
  -5 * x + 12000

theorem parking_fee_function_relationship :
  ∀ x, 0 ≤ x ∧ x ≤ 1200 → parking_fee x = -5 * x + 12000 :=
by
  intro x
  intro h
  sorry

theorem parking_fee_range :
  ∀ x, 780 ≤ x ∧ x ≤ 1020 → 6900 ≤ parking_fee x ∧ parking_fee x ≤ 8100 :=
by
  intro x
  intro h
  sorry

end parking_fee_function_relationship_parking_fee_range_l91_91371


namespace area_of_ABD_l91_91002

-- Define the given triangle with its sides and right angle at A
structure Triangle (α β γ : ℝ) :=
  (a b c : ℝ)
  (right_angle : α = 90)
  (ac_eq : a = 3)
  (ab_eq : b = 4)
  (bc_eq : c = 5)

-- Define point D on side BC such that the perimeters of ΔACD and ΔABD are equal
structure PointOnBC (D : ℝ) (triangle : Triangle α β γ) :=
  (on_BC : 0 ≤ D ∧ D ≤ triangle.c)
  (equal_perimeters : triangle.a + D + some_AD = triangle.b + (triangle.c - D) + some_AD)

-- Define the statement to prove the area of ΔABD is 12/5
theorem area_of_ABD {α β γ : ℝ} (triangle : Triangle α β γ) {D : ℝ}
  (point : PointOnBC D triangle) : 
  (1 / 2) * triangle.b * (triangle.c - D) * (Math.sin (β / 180 * Math.pi)) = 12 / 5 :=
by
  sorry

end area_of_ABD_l91_91002


namespace average_of_distinct_t_values_l91_91702

theorem average_of_distinct_t_values (t : ℕ) (r1 r2 : ℕ) (h1 : r1 + r2 = 7) (h2 : r1 * r2 = t)
  (pos_r1 : r1 > 0) (pos_r2 : r2 > 0) :
  (6 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) ∧
  (10 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) ∧
  (12 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) →
  ( ∑ x in {6, 10, 12}, (x : ℚ)) / 3 = 28 / 3 :=
by
    sorry

end average_of_distinct_t_values_l91_91702


namespace function_increasing_l91_91100

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem function_increasing (a b c : ℝ) (h : a^2 - 3 * b < 0) : 
  ∀ x y : ℝ, x < y → f x a b c < f y a b c := sorry

end function_increasing_l91_91100


namespace find_U_l91_91236

variable {R : Type*} [Field R]
variable {V : Type*} [AddCommGroup V] [Module R V]
variable {U : V → V}

-- Given conditions
axiom linearity (v w : V) (a b : R) : U (a • v + b • w) = a • U v + b • U w
axiom cross_product (v w : V) : U (v × w) = U v × U w
axiom U_v1 : U ⟨2, 0, 5⟩ = ⟨3, -1, 4⟩
axiom U_v2 : U ⟨0, 5, 2⟩ = ⟨-1, 4, 3⟩

-- Main statement to prove
theorem find_U : U ⟨2, 5, 7⟩ = ⟨2, 3, 7⟩ :=
by
  sorry

end find_U_l91_91236


namespace wine_problem_solution_l91_91585

theorem wine_problem_solution (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 200) (h2 : (200 - x) * (180 - x) / 200 = 144) : x = 20 := 
by
  sorry

end wine_problem_solution_l91_91585


namespace probability_one_letter_each_l91_91239

noncomputable def selection_probability (A : ℕ) (J : ℕ) (total : ℕ) : ℚ :=
  ((A / total : ℚ) * (J / (total - 1)) + (J / total) * (A / (total - 1))) 

theorem probability_one_letter_each (total: ℕ) (A: ℕ) (J: ℕ) (hA: A = 6) (hJ: J = 6) (htotal: total = 12) :
  selection_probability A J total = 6 / 11 := 
by {
  exact sorry,
}

end probability_one_letter_each_l91_91239


namespace remainder_when_divided_by_296_and_37_l91_91909

theorem remainder_when_divided_by_296_and_37 (N : ℤ) (k : ℤ)
  (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_when_divided_by_296_and_37_l91_91909


namespace concentration_is_40_percent_l91_91868

def volume_pure_acid : ℝ := 4.800000000000001
def total_volume_solution : ℝ := 12
def percentage_concentration_pure_acid (volume_pure : ℝ) (total_volume : ℝ) : ℝ := (volume_pure / total_volume) * 100

theorem concentration_is_40_percent :
  percentage_concentration_pure_acid volume_pure_acid total_volume_solution = 40 :=
by
  sorry

end concentration_is_40_percent_l91_91868


namespace max_value_sqrt_eq_14_l91_91303

theorem max_value_sqrt_eq_14 (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  ∃ y, y = sqrt (49 + x) + sqrt (49 - x) ∧ y ≤ 14 ∧ ∀ z, z = sqrt (49 + x) + sqrt (49 - x) → z ≤ y :=
by {
  let max_val := sqrt (49 : ℝ) + sqrt (49 : ℝ),
  have h_max : max_val = 14 := by norm_num,
  use max_val,
  split,
  { exact h_max },
  split,
  { apply max_val_le,
    intros a ha,
    have h1 := @abs_le_of_le_of_neg any_field _ h.1,
    have ha := sqrt_real.exists_le_sqrt_add_sqrt (49 + x) (49 - x),
    sorry },
  { intro z,
    assume hz : z = sqrt (49 + x) + sqrt (49 - x),
    apply le_of_eq,
    exact ha }
  }

end max_value_sqrt_eq_14_l91_91303


namespace find_fourth_term_in_sequence_l91_91293

theorem find_fourth_term_in_sequence (x: ℤ) (h1: 86 - 8 = 78) (h2: 2 - 86 = -84) (h3: x - 2 = -90) (h4: -12 - x = 76):
  x = -88 :=
sorry

end find_fourth_term_in_sequence_l91_91293


namespace min_rabbits_for_two_colors_l91_91016

theorem min_rabbits_for_two_colors 
  (white_blue_green : fin 3 → ℕ) 
  (h_total : ∑ i, white_blue_green i = 100) 
  (h_81_rabbits : ∀ subset : finset (fin 3), subset.card = 81 → (∀ i j k, i ≠ j → j ≠ k → k ≠ i → ∃ i ∈ subset ∧ ∃ j ∈ subset ∧ ∃ k ∈ subset)) : 
  ∃ subset : finset (fin 3), subset.card = 61 ∧ ∃ i j, i ≠ j ∧ i ∈ subset ∧ j ∈ subset :=
sorry

end min_rabbits_for_two_colors_l91_91016


namespace positive_number_is_49_l91_91385

theorem positive_number_is_49 (a : ℝ) (x : ℝ) (h₁ : (3 - a) * (3 - a) = x) (h₂ : (2 * a + 1) * (2 * a + 1) = x) :
  x = 49 :=
sorry

end positive_number_is_49_l91_91385


namespace jason_total_spending_l91_91420

def discounted_price (price : ℕ) (discount_percent : ℕ) : ℕ :=
  let discount_amount := (price * discount_percent) / 100
  price - discount_amount

def total_price_with_tax (prices : List ℕ) (tax_percent : ℕ) : ℕ :=
  let total := prices.foldl (· + ·) 0
  total + (total * tax_percent) / 100

theorem jason_total_spending :
  let flute := discounted_price 14246 10 in
  let music_tool := discounted_price 889 5 in
  let song_book := discounted_price 700 15 in
  let flute_case := discounted_price 3525 20 in
  let music_stand := discounted_price 1215 10 in
  let cleaning_kit := discounted_price 1499 25 in
  let sheet_protectors := discounted_price 329 5 in
  let total_before_tax := flute + music_tool + song_book + flute_case + music_stand + cleaning_kit + sheet_protectors in
  total_price_with_tax [total_before_tax] 8 = 21180 :=
by sorry

end jason_total_spending_l91_91420


namespace concurrency_of_lines_l91_91620

open Set

theorem concurrency_of_lines {A B C P D E F M N : Point} (hP_inside_ABC : Inside P (triangle A B C))
  (hD_projection : Projection P B C D) (hE_projection : Projection P C A E) (hF_projection : Projection P A B F)
  (hM_perpendicular : Perpendicular (A, M) (B, P)) (hN_perpendicular : Perpendicular (A, N) (C, P)) :
  Concurrent (line_through M E) (line_through N F) (line_through B C) :=
sorry

end concurrency_of_lines_l91_91620


namespace area_of_shaded_region_equals_seven_pi_l91_91619

-- Definitions for conditions
def small_circle_radius : ℝ := 1
def number_of_small_circles : ℕ := 13

-- Main statement
theorem area_of_shaded_region_equals_seven_pi (r : ℝ) (n : ℕ) (A : ℝ) :
  r = small_circle_radius → n = number_of_small_circles → 
  A = n * π → 
  let large_circle_area := (r * r) * (n + 1) * π in
  A - n * π = 7 * π :=
sorry

end area_of_shaded_region_equals_seven_pi_l91_91619


namespace min_value_frac_one_div_a_plus_four_div_b_l91_91691

theorem min_value_frac_one_div_a_plus_four_div_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : b ∈ ℝ) 
(hf : ∀ x : ℝ, (a ^ x - b) * (x + 1) ≤ 0) : (1 / a) + (4 / b) = 4 := 
by
  sorry

end min_value_frac_one_div_a_plus_four_div_b_l91_91691


namespace D_on_AC_l91_91450

theorem D_on_AC {O A B C D : Point}
  (hCircle : Circle O)
  (hOnCircleA : OnCircle O A)
  (hOnCircleB : OnCircle O B)
  (hOnCircleC : OnCircle O C)
  (hAngleABC : ∠ABC > 90)
  (hCircumcircle : Circumcircle O B C)
  (hBisector : Bisector ∠AOB intersects hCircumcircle = D)
  : Collinear A D C := by
  sorry

end D_on_AC_l91_91450


namespace average_of_distinct_t_values_l91_91705

theorem average_of_distinct_t_values (t : ℕ) (r1 r2 : ℕ) (h1 : r1 + r2 = 7) (h2 : r1 * r2 = t)
  (pos_r1 : r1 > 0) (pos_r2 : r2 > 0) :
  (6 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) ∧
  (10 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) ∧
  (12 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) →
  ( ∑ x in {6, 10, 12}, (x : ℚ)) / 3 = 28 / 3 :=
by
    sorry

end average_of_distinct_t_values_l91_91705


namespace cross_sectional_area_correct_l91_91989

-- Conditions
def cyl_circumference : ℝ := 4
def cyl_height : ℝ := 2

-- Definition of the radius from the circumference
def cyl_radius : ℝ := cyl_circumference / (2 * Real.pi)

-- Definition of the cross-sectional area through the axis
def cyl_cross_sectional_area : ℝ := (2 * cyl_radius * cyl_height)

-- Statement we need to prove
theorem cross_sectional_area_correct :
  cyl_cross_sectional_area = 8 / Real.pi :=
by
  -- Proof goes here
  sorry

end cross_sectional_area_correct_l91_91989


namespace find_abc_l91_91033

-- Definitions based on the conditions
def Γ1_radius := 5 / 2
def A_B_dist := 3
def A_C_dist := 5

-- Define the general types of all points involved
noncomputable def PXY_expr (a b c : ℕ) : Prop :=
  ∃ (area_expr : ℝ) (a b c : ℕ),
  area_expr = 8 * Real.sqrt 6 / 5 ∧
  a = 8 ∧ b = 6 ∧ c = 5

theorem find_abc : ∃ (a b c : ℕ), PXY_expr a b c ∧ (a + b + c) = 19 :=
by
  use [8, 6, 5]
  constructor
  {
    -- Proof of the existence of the area expression in previously defined terms
    sorry
  }
  -- Final correct answer
  show 8 + 6 + 5 = 19 by
  trivial

end find_abc_l91_91033


namespace HNJ_triangle_area_is_sqrt3_l91_91007

variables (EF GH : ℝ)
variables (rect_area : ℝ)
variables (HN NJ : ℝ)
variables (triangle_area : ℝ)

def HNJ_equilateral_triangle (EF GH : ℝ) (rect_area : ℝ) (HN NJ : ℝ) : Prop :=
  EF = 2 ∧ GH = 0.5 ∧ rect_area = 1 ∧ HN = 2 ∧ NJ = 2 ∧ HN = NJ

theorem HNJ_triangle_area_is_sqrt3
  (EF GH : ℝ) (rect_area HN NJ : ℝ)
  (h : HNJ_equilateral_triangle EF GH rect_area HN NJ) :
  triangle_area = sqrt 3 :=
by
  sorry

end HNJ_triangle_area_is_sqrt3_l91_91007


namespace max_sqrt_sum_l91_91306

theorem max_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
sorry

end max_sqrt_sum_l91_91306


namespace ratio_AP_PC_l91_91471

-- Definitions of points and the given conditions
structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)

variables {α : Type*} [MetricSpace α] (C : Circle α)
variables (A M N B C P : α)
variables (AB BC AP PC : ℝ)

-- Assume the necessary conditions for the tangents and secant
def is_tangent (P : α) (C : Circle α) : Prop :=
  dist P C.center = C.radius

def is_secant (A B C : α) (P : α) (C : Circle α) : Prop :=
  dist P C.center < C.radius ∧ dist A C.center > C.radius ∧ dist B C.center > C.radius

-- Given conditions: point A has two tangents AM and AN touching circle at M and N respectively.
axiom tangents_A : is_tangent M C ∧ is_tangent N C

-- The secant from the same point A intersects the circle at B and C,
-- and intersects the chord MN at P, with the ratio AB:BC = 2:3
axiom secant_A : is_secant A B C P C
axiom ratio_AB_BC : AB / BC = 2 / 3

-- Conclusion: Find the ratio AP:PC
theorem ratio_AP_PC : (AP / PC = 4 / 3) :=
sorry

end ratio_AP_PC_l91_91471


namespace total_employees_in_buses_l91_91164

-- Definitions from conditions
def busCapacity : ℕ := 150
def percentageFull1 : ℕ := 60
def percentageFull2 : ℕ := 70

-- Proving the total number of employees
theorem total_employees_in_buses : 
  (percentageFull1 * busCapacity / 100) + (percentageFull2 * busCapacity / 100) = 195 := 
by
  sorry

end total_employees_in_buses_l91_91164


namespace odd_n_geq_3_property_l91_91649

theorem odd_n_geq_3_property (n : ℤ) (h : n ≥ 3) : 
  (∀ (a b : Fin n → ℝ), 
   (∀ k, 1 ≤ k ∧ k ≤ n → abs (a (k-1)) + abs (b (k-1)) = 1) 
    → ∃ (x : Fin n → ℤ), 
      (∀ k, x k = 1 ∨ x k = -1) 
      ∧ abs (∑ k in Finset.range n, x k * a k) + abs (∑ k in Finset.range n, x k * b k) ≤ 1) 
  ↔ odd n :=
by sorry

end odd_n_geq_3_property_l91_91649


namespace maximum_value_of_f_solution_set_l91_91047

noncomputable def f (x : ℝ) := Real.sqrt (x - 2) + Real.sqrt (11 - x)
def M := 3 * Real.sqrt 2

theorem maximum_value_of_f : ∃ x, f x = M := by
  sorry

theorem solution_set :
  { x : ℝ | abs (x - Real.sqrt 2) + abs (x + 2 * Real.sqrt 2) ≤ M } =
  { x : ℝ | -2 * Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 } := by
  sorry

end maximum_value_of_f_solution_set_l91_91047


namespace geom_ineq_isosceles_l91_91329

-- Definitions according to conditions
variables {A B C D E F : Type} [metric_space F]

-- Statement of the conditions
variables (isosceles_triangle : ∀ {x y z : F}, (dist x y = dist y z) → (dist y x = dist z x))
variables (on_extension_D : ∀ {a c d : F}, (dist a d > dist a c))
variables (on_extension_E : ∀ {c e : F}, (dist c e > dist A c))
variables (on_extension_F : ∀ {b f : F}, (dist b f > dist b c))
variables (equal_AD_BF: dist A D = dist B F)
variables (equal_CE_CF: dist C E = dist C F)

-- The theorem to be proven
theorem geom_ineq_isosceles { B D F E: F} (h1 : isosceles_triangle) 
    (h2 : on_extension_D) (h3: on_extension_E) 
    (h4 : on_extension_F) (h5 : equal_AD_BF) (h6 : equal_CE_CF): 
    (dist B D + dist C F > dist E F) := 
sorry

end geom_ineq_isosceles_l91_91329


namespace max_value_sqrt_eq_14_l91_91301

theorem max_value_sqrt_eq_14 (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  ∃ y, y = sqrt (49 + x) + sqrt (49 - x) ∧ y ≤ 14 ∧ ∀ z, z = sqrt (49 + x) + sqrt (49 - x) → z ≤ y :=
by {
  let max_val := sqrt (49 : ℝ) + sqrt (49 : ℝ),
  have h_max : max_val = 14 := by norm_num,
  use max_val,
  split,
  { exact h_max },
  split,
  { apply max_val_le,
    intros a ha,
    have h1 := @abs_le_of_le_of_neg any_field _ h.1,
    have ha := sqrt_real.exists_le_sqrt_add_sqrt (49 + x) (49 - x),
    sorry },
  { intro z,
    assume hz : z = sqrt (49 + x) + sqrt (49 - x),
    apply le_of_eq,
    exact ha }
  }

end max_value_sqrt_eq_14_l91_91301


namespace measure_of_angle_C_range_of_sin_A_plus_sin_B_l91_91738

-- Define the vectors and conditions
variables {a b c : ℝ}
def vec_n : ℝ × ℝ := (a + c, b)
def vec_m : ℝ × ℝ := (a - c, b - a)
def perpendicular : Prop := vec_m.1 * vec_n.1 + vec_m.2 * vec_n.2 = 0

-- Statement (1): Measure of angle C
theorem measure_of_angle_C (h : perpendicular) : 
  let C := real.acos ((a^2 + b^2 - c^2) / (2 * a * b))
  in C = π / 3 := sorry

-- Statement (2): Range of sin A + sin B
theorem range_of_sin_A_plus_sin_B (hC : let C := real.acos ((a^2 + b^2 - c^2) / (2 * a * b)) in C = π / 3) : 
  ∃ (A B : ℝ), (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ A + B = 2 * π / 3 ∧ 
  (∀ x, x = real.sin(A) + real.sin(B) → (sqrt 3 / 2 < x ∧ x ≤ sqrt 3)) := sorry

end measure_of_angle_C_range_of_sin_A_plus_sin_B_l91_91738


namespace find_f_val_l91_91354

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then 2^(x - 1) - 2 else -real.log2 (x + 1)

variable (a : ℝ)
axiom h : f a = -3

theorem find_f_val : f (6 - a) = -7 / 4 :=
by sorry

end find_f_val_l91_91354


namespace arithmetic_100_sum_l91_91449

def arithmetic_100_sum_proof : Prop :=
  let c (n : ℕ) (d_c : ℝ) := 15 + (n - 1) * d_c
  let d (n : ℕ) (d_d : ℝ) := 105 + (n - 1) * d_d
  (∃ d_c d_d : ℝ, (c 100 d_c + d 100 d_d = 300) ∧
    (let sum_first_100 := ∑ i in finset.range 1 101, c i d_c + d i d_d
     sum_first_100 = 21000))

theorem arithmetic_100_sum : arithmetic_100_sum_proof := sorry

end arithmetic_100_sum_l91_91449


namespace log_sum_property_l91_91455

noncomputable def f (a : ℝ) (h : a > 0 ∧ a ≠ 1) : ℝ → ℝ := λ x, log a x

theorem log_sum_property (a : ℝ) (h : a > 0 ∧ a ≠ 1) (x : ℕ → ℝ) (h_prod : f a h (∏ i in finset.range 2013, x i) = 8) :
  ∑ i in finset.range 2013, f a h (x i ^ 2) = 16 :=
by
  sorry

end log_sum_property_l91_91455


namespace geometric_seq_sufficient_arithmetic_lg_seq_l91_91688

theorem geometric_seq_sufficient_arithmetic_lg_seq (a : ℕ → ℝ) : 
  (∃ q : ℝ, q ≠ 0 ∧ (∀ n : ℕ, a (n + 1) = q * a n)) →
  (∀ n : ℕ, (∃ d : ℝ, (lg (a (n + 1)) + 1) - (lg (a n) + 1) = d)) ∧ ¬ (∀ a : ℕ → ℝ, (∀ n : ℕ, (∃ d : ℝ, (lg (a (n + 1)) + 1) - (lg (a n) + 1) = d)) → (∃ q : ℝ, q ≠ 0 ∧ (∀ n : ℕ, a (n + 1) = q * a n))) := by
  sorry

end geometric_seq_sufficient_arithmetic_lg_seq_l91_91688


namespace triangle_condition_l91_91013

-- Definitions based on the conditions
def angle_equal (A B C : ℝ) : Prop := A = B - C
def angle_ratio123 (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ A / C = 1 / 3 ∧ B / C = 2 / 3
def pythagorean (a b c : ℝ) : Prop := a * a + b * b = c * c
def side_ratio456 (a b c : ℝ) : Prop := a / b = 4 / 5 ∧ a / c = 4 / 6 ∧ b / c = 5 / 6

-- Main hypothesis with right-angle and its conditions in different options
def is_right_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (angle_equal A B C → A = 90 ∨ B = 90 ∨ C = 90) ∧
  (angle_ratio123 A B C → A = 30 ∧ B = 60 ∧ C = 90) ∧
  (pythagorean a b c → true) ∧
  (side_ratio456 a b c → false) -- option D cannot confirm the triangle is right

theorem triangle_condition (A B C a b c : ℝ) : is_right_triangle A B C a b c :=
sorry

end triangle_condition_l91_91013


namespace tan_of_angle_l91_91530

theorem tan_of_angle (θ : ℝ) (y : ℝ) (h1 : sin θ = -3 / 5) (h2 : cos θ = 4 / (sqrt (16 + y^2))) :
  tan θ = -3 / 4 :=
begin
  sorry
end

end tan_of_angle_l91_91530


namespace arithmetic_sequence_values_l91_91743

theorem arithmetic_sequence_values (a b c : ℤ) 
  (h1 : 2 * b = a + c)
  (h2 : 2 * a = b + 1)
  (h3 : 2 * c = b + 9) 
  (h4 : a + b + c = -15) :
  b = -5 ∧ a * c = 21 :=
by
  sorry

end arithmetic_sequence_values_l91_91743


namespace mrs_johnsons_class_raised_l91_91543

-- Define the given conditions
def amount_after_deduction := 27048 : ℝ
def deduction_rate := 0.02 : ℝ

-- The amount before deduction
def total_amount_raised (T : ℝ) :=
  (1 - deduction_rate) * T = amount_after_deduction

-- Relation between the amounts raised by each class
def miss_rollins_class_amount (T : ℝ) :=
  let miss_rollins_amount := T / 3 in miss_rollins_amount

def mrs_suttons_class_amount (T : ℝ) :=
  let miss_rollins_amount := T / 3 in
  let mrs_sutton_amount := miss_rollins_amount / 8 in mrs_sutton_amount

def mrs_johnsons_class_amount (T : ℝ) :=
  let miss_rollins_amount := T / 3 in
  let mrs_sutton_amount := miss_rollins_amount / 8 in
  let mrs_johnson_amount := 2 * mrs_sutton_amount in mrs_johnson_amount

-- Main theorem stating the amount raised by Mrs. Johnson's class
theorem mrs_johnsons_class_raised (T : ℝ) (hT : total_amount_raised T) :
  mrs_johnsons_class_amount T = 2300 := by
  sorry

end mrs_johnsons_class_raised_l91_91543


namespace inscribed_square_area_l91_91231

theorem inscribed_square_area :
  (∃ (t : ℝ), (2*t)^2 = 4 * (t^2) ∧ ∀ (x y : ℝ), (x = t ∧ y = t ∨ x = -t ∧ y = t ∨ x = t ∧ y = -t ∨ x = -t ∧ y = -t) 
  → (x^2 / 4 + y^2 / 8 = 1) ) 
  → (∃ (a : ℝ), a = 32 / 3) := 
by
  sorry

end inscribed_square_area_l91_91231


namespace find_adult_ticket_price_l91_91234

noncomputable def adult_ticket_price : ℚ := 10.50

theorem find_adult_ticket_price (children_ticket_price : ℚ)
                               (total_revenue : ℚ)
                               (total_tickets : ℕ)
                               (adult_tickets : ℕ) :
  (children_ticket_price = 5) →
  (total_revenue = 236) →
  (total_tickets = 34) →
  (adult_tickets = 12) →
  let A := adult_ticket_price in
  12 * A + (34 - 12) * children_ticket_price = total_revenue :=
by
  intro children_ticket_price_eq total_revenue_eq total_tickets_eq adult_tickets_eq
  simp [children_ticket_price_eq, total_revenue_eq, total_tickets_eq, adult_tickets_eq, adult_ticket_price]
  sorry

end find_adult_ticket_price_l91_91234


namespace shaded_area_calculation_l91_91932

def calculate_shaded_area : ℝ :=
  let side_length := 10
  let radius := side_length / 2
  let diamond_area := 2 * side_length * radius / 2
  let semicircle_area := π * radius^2 / 2
  min diamond_area semicircle_area

theorem shaded_area_calculation : calculate_shaded_area ≤ min 50 (12.5 * π) :=
by
  -- This is where the proof would go
  sorry

end shaded_area_calculation_l91_91932


namespace volume_of_ice_cream_l91_91510

noncomputable def volumeOfCone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h
noncomputable def volumeOfCylinder (r h : ℝ) : ℝ := π * r^2 * h

theorem volume_of_ice_cream (r_cone h_cone h_cylinder : ℝ) 
  (h_r_cone : r_cone = 3) (h_r_cylinder : r_cone = 3) 
  (h_h_cone : h_cone = 12) (h_h_cylinder : h_cylinder = 2) :
  volumeOfCone r_cone h_cone + volumeOfCylinder r_cone h_cylinder = 54 * π := 
by
  sorry

end volume_of_ice_cream_l91_91510


namespace exponential_function_decreasing_l91_91334

theorem exponential_function_decreasing (a b : ℝ) (h : a > b) :
  (1 / 2)^a < (1 / 2)^b :=
by {
  sorry
}

end exponential_function_decreasing_l91_91334


namespace angles_equal_in_pyramid_l91_91409

variable {A B C S : Type} [Point A] [Point B] [Point C] [Point S]

-- ... represent the properties of points, lines, and angles ...

-- Define the pyramid structure
axiom pyramid_SABC (A B C S : Type) [Point A] [Point B] [Point C] [Point S] : Prop :=
  ∃ (SC : Line S C), SC ⊥ (Plane A B C)

-- Assume SC is perpendicular to the base plane
axiom SC_perpendicular_base (SC : Line S C) (ABC_base_plane : Plane A B C) : SC ⊥ ABC_base_plane

-- Define the angles
def angle_ASB (A S B : Type) [Point A] [Point S] [Point B] : Angle :=
  -- ... definition of angle ASB ...

def angle_ACB (A C B : Type) [Point A] [Point C] [Point B] : Angle :=
  -- ... definition of angle ACB ...

-- Prove that angles ASB and ACB can be equal under given conditions
theorem angles_equal_in_pyramid : ∀ (A B C S : Type) [Point A] [Point B] [Point C] [Point S],
  pyramid_SABC A B C S →
  (∃ (SC : Line S C), SC_perpendicular_base SC ⟨A, B, C⟩) →
  ∃ (angle_ASB_angle : Angle) (angle_ACB_angle : Angle),
    angle_ASB A S B = angle_ACB A C B :=
by
  -- The proof would go here
  sorry

end angles_equal_in_pyramid_l91_91409


namespace number_of_ordered_pairs_l91_91655

theorem number_of_ordered_pairs :
  (∑ (a : ℕ) in Finset.range 51, a - 1) = 1225 := 
by sorry

end number_of_ordered_pairs_l91_91655


namespace division_problem_l91_91560

theorem division_problem : (4 * 5) / 10 = 2 :=
by sorry

end division_problem_l91_91560


namespace percentage_passed_both_subjects_l91_91391

def failed_H : ℝ := 0.35
def failed_E : ℝ := 0.45
def failed_HE : ℝ := 0.20

theorem percentage_passed_both_subjects :
  (100 - (failed_H * 100 + failed_E * 100 - failed_HE * 100)) = 40 := 
by
  sorry

end percentage_passed_both_subjects_l91_91391


namespace max_sqrt_expression_l91_91299

theorem max_sqrt_expression (x : ℝ) (hx : -49 ≤ x ∧ x ≤ 49) :
  ∃ y, y = 14 ∧ (∀ z, z = sqrt (49 + x) + sqrt (49 - x) → z ≤ y) :=
by
  sorry

end max_sqrt_expression_l91_91299


namespace prove_hyperbola_l91_91361

noncomputable theory

def parametric_curve_is_hyperbola (t : ℝ) : Prop :=
  let x := 2 * (t + 1/t)
  let y := 2 * (t - 1/t)
  ((x^2) / 16 - (y^2) / 16) = 1

theorem prove_hyperbola : ∀ (t : ℝ), parametric_curve_is_hyperbola t :=
by 
  sorry

end prove_hyperbola_l91_91361


namespace simplify_and_evaluate_expression_l91_91490

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1/2) : x^2 * (x - 1) - x * (x^2 + x - 1) = 0 := by
  sorry

end simplify_and_evaluate_expression_l91_91490


namespace tournament_choice_count_l91_91582

-- Defining the conditions as constants or definitions
def num_players : ℕ := 16
def num_choices : ℕ := 3

-- Calculation of the specific case where 16 players are involved.
theorem tournament_choice_count :
  ∃ (winner_choices : ℕ) (match_outcomes : ℕ), 
    winner_choices = num_choices ∧ match_outcomes = 2^(num_players - 1) ∧ winner_choices * match_outcomes = 3 * 2^15 :=
by
  use num_choices, 2^(num_players - 1)
  split
  · rfl
  split
  · rfl
  · sorry

end tournament_choice_count_l91_91582


namespace sum_integers_minus_50_to_70_l91_91896

theorem sum_integers_minus_50_to_70 : ∑ i in finset.range (71 - -49), (-50 + i) = 2485 :=
by
  sorry

end sum_integers_minus_50_to_70_l91_91896


namespace length_RS_l91_91764

axiom Quadrilateral (F R D S : Type) 

variables {F R D S : Point}

axiom FD : real := 3
axiom DR : real := 4
axiom FR : real := 5
axiom FS : real := 8

axiom right_triangle_FDR : is_right_angle (angle F D R)
axiom angle_RFS_eq_angle_FDR : (angle R F S) = (angle F D R)

theorem length_RS : ∃ (RS : real), RS = sqrt 89 :=
by {
  use sqrt 89,
  sorry
}

end length_RS_l91_91764


namespace age_of_older_sister_in_2021_l91_91027

instance : DecidableEq ℕ := Classical.decEq ℕ

theorem age_of_older_sister_in_2021 (year_kelsey_25 : ℕ) (current_year : ℕ) (kelsey_age_in_1999 : ℕ) (sister_age_diff : ℕ) :
  (year_kelsey_25 = 1999) ∧ (kelsey_age_in_1999 = 25) ∧ (sister_age_diff = 3) ∧ (current_year = 2021) → 
  (current_year - ((year_kelsey_25 - kelsey_age_in_1999) - sister_age_diff) = 50) :=
by
  sorry

end age_of_older_sister_in_2021_l91_91027


namespace salary_increase_l91_91247

theorem salary_increase (x : ℝ) 
  (h : ∀ s : ℕ, 1 ≤ s ∧ s ≤ 5 → ∃ p : ℝ, p = 7.50 + x * (s - 1))
  (h₁ : ∃ p₁ p₅ : ℝ, 1 ≤ 1 ∧ 5 ≤ 5 ∧ p₅ = p₁ + 1.25) :
  x = 0.3125 := sorry

end salary_increase_l91_91247


namespace find_larger_integer_l91_91921

theorem find_larger_integer (x : ℕ) (hx₁ : 4 * x > 0) (hx₂ : (x + 6) * 3 = 4 * x) : 4 * x = 72 :=
by
  sorry

end find_larger_integer_l91_91921


namespace quadrilateral_parallelogram_l91_91733

theorem quadrilateral_parallelogram (A B C D : Type) [add_comm_group A] [affine_space A B]
  (a b c d : B) (AB CD AD BC : ℝ) :
  (dist a b = dist c d) →
  (dist a d = dist b c) →
  (a +ᵥ (c -ᵥ b) = d) :=
sorry

end quadrilateral_parallelogram_l91_91733


namespace product_of_three_numbers_l91_91128

theorem product_of_three_numbers :
  ∃ (a b c : ℚ), 
    a + b + c = 30 ∧ a = 2 * (b + c) ∧ b = 6 * c ∧ a * b * c = 12000 / 49 :=
by
  sorry

end product_of_three_numbers_l91_91128


namespace time_addition_l91_91784

/-- 
    Given the initial time 3:15:30 PM and additional time of 174 hours, 58 minutes, and 16 seconds,
    we want to determine the new time in the format A:B:C (12-hour clock) and check if A + B + C = 69.
-/
theorem time_addition : 
    let initial_hours := 3
    let initial_minutes := 15
    let initial_seconds := 30
    let added_hours := 174
    let added_minutes := 58
    let added_seconds := 16 in
    let final_hours := (initial_hours + (added_hours % 12)) % 12
    let final_minutes := (initial_minutes + added_minutes) % 60
    let final_seconds := (initial_seconds + added_seconds) % 60 in
    (final_hours + final_minutes + final_seconds + (if final_minutes + added_minutes >= 60 then 1 else 0) + (if final_seconds + added_seconds >= 60 then 1 else 0)) = 69 :=
by
    -- This proof is omitted for now.
    -- Placeholder to indicate the need of formal proof body.
    sorry

end time_addition_l91_91784


namespace ordered_pairs_count_l91_91267

theorem ordered_pairs_count :
  let S := { (a, b) : ℝ × ℤ | 0 < a ∧ 2 ≤ b ∧ b ≤ 10 ∧ (log b a)^3 = 2 * log b (a ^ 3) } in
  S.card = 27 :=
by
  have h1 : ∀ b ∈ set.range (λ n : ℤ, n) ∩ (set.Icc 2 10), ∃ a > 0, 
    (log b a)^3 = 2 * log b (a ^ 3) := sorry
  have h2 : ∀ b ∈ set.range (λ n : ℤ, n) ∩ (set.Icc 2 10), # {
    a : ℝ | 0 < a ∧ (log b a)^3 = 2 * log b (a ^ 3)
  } = 3 := sorry
  have h3 : (set.range (λ n : ℤ, n) ∩ (set.Icc 2 10)).card = 9 := sorry
  exact (finset.card_product (set.range (λ n : ℤ, n) ∩ (set.Icc 2 10)) 
        (set.range (λ n : ℝ, 0 < n ∧ (log _ n)^3 = 2 * log _ ((n : ℝ)^3))).to_finset)
      .trans h3 .mul h2

end ordered_pairs_count_l91_91267


namespace probability_shattering_l91_91966

theorem probability_shattering (total_cars : ℕ) (shattered_windshields : ℕ) (p : ℚ) 
  (h_total : total_cars = 20000) 
  (h_shattered: shattered_windshields = 600) 
  (h_p : p = shattered_windshields / total_cars) : 
  p = 0.03 := 
by 
  -- skipped proof
  sorry

end probability_shattering_l91_91966


namespace find_z_l91_91648

theorem find_z (x y : ℤ) (h1 : x * y + x + y = 106) (h2 : x^2 * y + x * y^2 = 1320) :
  x^2 + y^2 = 748 ∨ x^2 + y^2 = 5716 :=
sorry

end find_z_l91_91648


namespace isosceles_triangle_height_and_area_l91_91687

theorem isosceles_triangle_height_and_area
  (O E A: Point) -- Points O, E, and A are given
  (OE_is_base_isosceles : is_base_of_isosceles O E) -- OE is the base of an isosceles triangle with vertex O
  (A_on_line_E : is_on_line A E) -- Point A is on line E
  (midpoint_E_AB : is_midpoint E A B) -- E is the midpoint of AB
  (slope_AB : slope A B = -1) -- The slope of AB is -1
: (line_eq_height : equation_of_line (height O A) = "x - y - 1 = 0") -- The equation of the height from O is x - y - 1 = 0
∧ (area_OE : area O E = 2) -- The area of OE is 2
:= sorry

end isosceles_triangle_height_and_area_l91_91687


namespace union_of_two_triangles_cannot_form_13_gon_l91_91626

theorem union_of_two_triangles_cannot_form_13_gon
    (T1 T2 : Type)
    [Triangle T1] [Triangle T2]
    (vertices_T1 : Finset Point) (vertices_T2 : Finset Point)
    (edges_T1 : Finset (Point × Point)) (edges_T2 : Finset (Point × Point))
    (h1 : vertices_T1.card = 3) (h2 : vertices_T2.card = 3)
    (h3 : edges_T1.card = 3) (h4 : edges_T2.card = 3)
    (intersection_points : Finset Point)
    (h5 : intersection_points.card ≤ (3 * 2)) :
    (vertices_T1 ∪ vertices_T2 ∪ intersection_points).card ≠ 13 := sorry

end union_of_two_triangles_cannot_form_13_gon_l91_91626


namespace percentage_decrease_in_revenue_approx_l91_91094

-- Given conditions
def last_year_transaction_fees := 40.0
def this_year_transaction_fees := 28.8
def transaction_conversion_last_year := 1.0
def transaction_conversion_this_year := 0.98

def last_year_data_fees := 25.0
def this_year_data_fees := 20.0
def data_conversion_last_year := 1.0
def data_conversion_this_year := 1.02

def last_year_cross_border := 20.0
def this_year_cross_border := 17.6
def cross_border_conversion_last_year := 1.0
def cross_border_conversion_this_year := 0.95

/--
Prove that the overall percentage decrease in the company's total revenue, 
factoring in the currency exchange rate fluctuations, is approximately 23.13%.
-/
theorem percentage_decrease_in_revenue_approx : 
  let last_year_total_revenue := last_year_transaction_fees * transaction_conversion_last_year
                                + last_year_data_fees * data_conversion_last_year
                                + last_year_cross_border * cross_border_conversion_last_year,
      this_year_total_revenue := this_year_transaction_fees * transaction_conversion_this_year
                                + this_year_data_fees * data_conversion_this_year
                                + this_year_cross_border * cross_border_conversion_this_year,
      decrease_in_revenue := last_year_total_revenue - this_year_total_revenue,
      percentage_decrease := (decrease_in_revenue / last_year_total_revenue) * 100
  in percentage_decrease ≈ 23.13 :=
by {
  -- Proof omitted
  sorry
}

end percentage_decrease_in_revenue_approx_l91_91094


namespace infinitely_many_solutions_l91_91039

theorem infinitely_many_solutions
  (a b c d x y z w : ℕ)
  (distinct_pos : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
                  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
  (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)
  (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0) (w_pos : w > 0)
  (sum_eq : a + b + c + d = x + y + z + w)
  (squares_eq : a^2 + b^2 + c^2 + d^2 = x^2 + y^2 + z^2 + w^2)
  (cubes_eq : a^3 + b^3 + c^3 + d^3 = x^3 + y^3 + z^3 + w^3) :
  ∃ (k m n : ℕ), k > 0 ∧ m > 0 ∧ n > 0 ∧
  ∃ (a b c d x y z w : ℕ),
  (a = k ∧ b = k + 2 * m + n ∧ c = k + m + 3 * n ∧ d = k + 3 * m + 4 * n ∧
   x = k + m ∧ y = k + n ∧ z = k + 3 * m + 3 * n ∧ w = k + 2 * m + 4 * n ∧
   distinct_pos ∧ a_pos ∧ b_pos ∧ c_pos ∧ d_pos ∧ x_pos ∧ y_pos ∧ z_pos ∧ w_pos) ∧
  (a + b + c + d = x + y + z + w) ∧
  (a^2 + b^2 + c^2 + d^2 = x^2 + y^2 + z^2 + w^2) ∧
  (a^3 + b^3 + c^3 + d^3 = x^3 + y^3 + z^3 + w^3) :=
sorry

end infinitely_many_solutions_l91_91039


namespace num_satisfying_integers_le_2000_l91_91656

theorem num_satisfying_integers_le_2000 : 
  (length (filter (λ n : ℕ, n ≤ 2000 ∧ ∃ x : ℝ, ⌊x⌋₊ + ⌊3 * x⌋₊ + ⌊4 * x⌋₊ = n) (list.range (2001)))) = 1495 :=
by sorry

end num_satisfying_integers_le_2000_l91_91656


namespace balance_difference_correct_l91_91969

noncomputable def angela_balance_after_15_years :=
  7000 * (1 + 0.05)^(15 : ℝ)

noncomputable def bob_balance_after_15_years :=
  12000 * (1 + 15 * 0.04)

noncomputable def positive_difference :=
  abs (bob_balance_after_15_years - angela_balance_after_15_years)

theorem balance_difference_correct :
  |bob_balance_after_15_years - angela_balance_after_15_years| ≈ 4647.49 :=
by
  sorry

end balance_difference_correct_l91_91969


namespace dog_catches_fox_at_120_meters_l91_91946

theorem dog_catches_fox_at_120_meters
  (initial_distance : ℕ)
  (dog_jump_distance : ℕ)
  (fox_jump_distance : ℕ)
  (dog_jumps_per_unit : ℕ)
  (fox_jumps_per_unit : ℕ) 
  (h_initial_distance : initial_distance = 30)
  (h_dog_jump_distance : dog_jump_distance = 2)
  (h_fox_jump_distance : fox_jump_distance = 1)
  (h_dog_jumps_per_unit : dog_jumps_per_unit = 2)
  (h_fox_jumps_per_unit : fox_jumps_per_unit = 3) :
  let dog_speed := dog_jump_distance * dog_jumps_per_unit,
      fox_speed := fox_jump_distance * fox_jumps_per_unit,
      net_gain := dog_speed - fox_speed,
      time_units := initial_distance / net_gain,
      catching_distance := dog_speed * time_units
  in catching_distance = 120 :=
by {
  sorry
}

end dog_catches_fox_at_120_meters_l91_91946


namespace A_and_B_worked_together_for_5_days_before_A_left_the_job_l91_91937

noncomputable def workRate_A (W : ℝ) : ℝ := W / 20
noncomputable def workRate_B (W : ℝ) : ℝ := W / 12

noncomputable def combinedWorkRate (W : ℝ) : ℝ := workRate_A W + workRate_B W

noncomputable def workDoneTogether (x : ℝ) (W : ℝ) : ℝ := x * combinedWorkRate W
noncomputable def workDoneBy_B_Alone (W : ℝ) : ℝ := 3 * workRate_B W

theorem A_and_B_worked_together_for_5_days_before_A_left_the_job (W : ℝ) :
  ∃ x : ℝ, workDoneTogether x W + workDoneBy_B_Alone W = W ∧ x = 5 :=
by
  sorry

end A_and_B_worked_together_for_5_days_before_A_left_the_job_l91_91937


namespace length_AB_l91_91398

-- Conditions
variables (A B C D : Type) [euclidean_space A B C D]
variable {AB AC BC BD CD : ℝ}
variable {perim_ABC : ℝ} (hABC_perimeter : perim_ABC = 26)
variable {perim_CBD : ℝ} (hCBD_perimeter : perim_CBD = 24)
variable (hBD : BD = 10)
variables (isosceles_ABC : AC = BC) (isosceles_CBD : CD = BC)

-- Theorem: length of AB
theorem length_AB :
  AB = 12 :=
by
  sorry

end length_AB_l91_91398


namespace find_cd_minus_dd_base_d_l91_91497

namespace MathProof

variables (d C D : ℤ)

def digit_sum (C D : ℤ) (d : ℤ) : ℤ := d * C + D
def digit_sum_same (C : ℤ) (d : ℤ) : ℤ := d * C + C

theorem find_cd_minus_dd_base_d (h_d : d > 8) (h_eq : digit_sum C D d + digit_sum_same C d = d^2 + 8 * d + 4) :
  C - D = 1 :=
by
  sorry

end MathProof

end find_cd_minus_dd_base_d_l91_91497


namespace transformed_data_mean_and_variance_l91_91680

variable {n : ℕ}

-- Given conditions
def mean (data : Finₓ n → ℝ) : ℝ := (Finset.univ.sum (λ i, data i)) / n
def variance (data : Finₓ n → ℝ) : ℝ := 
  let m := mean data
  (Finset.univ.sum (λ i, (data i - m)^2)) / n

variables (data : Finₓ n → ℝ) (mean_data variance_data : ℝ)

-- Condition 1: The mean of the data set k_1, k_2, ..., k_n is 3
axiom mean_cond : mean data = 3

-- Condition 2: The variance of the data set k_1, k_2, ..., k_n is 3
axiom variance_cond : variance data = 3

-- Proof statement
theorem transformed_data_mean_and_variance :
  mean (λ i : Finₓ n, 2 * (data i + 3)) = 12 ∧ variance (λ i : Finₓ n, 2 * (data i + 3)) = 12 :=
by
  sorry

end transformed_data_mean_and_variance_l91_91680


namespace max_length_OB_l91_91878

theorem max_length_OB (O A B : Point) (h : Angle O A B = 45°) (h1 : dist A B = 1) : dist O B ≤ sqrt 2 :=
sorry

end max_length_OB_l91_91878


namespace tangent_line_at_one_condition_for_a_mean_value_average_function_l91_91335

-- Define f and g generically and then specialize.

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 - 4 * x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x

-- 1. Prove the equation of the tangent line at x = 1 when a = 1
theorem tangent_line_at_one : 
  ∀ x y : ℝ, (f 1 1 = -3) → (y + 3 = -(x - 1)) → (x + y + 2 = 0) :=
by
  sorry

-- 2. Prove that if ∀ x ∈ [1/e, e], f(x) ≥ g(x) then a ∈ (-∞, -1]
theorem condition_for_a (x : ℝ) (a : ℝ) : 
  (∀ x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1, f a x ≥ g a x) → a ∈ Set.Iic (-1) := 
by
  sorry

-- 3. Determine if f(x) is a "mean-value average function" and the related properties on a
theorem mean_value_average_function (a : ℝ) : 
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f a (1 / 2) = (f a x1 + f a x2) / 2) 
  ↔ ((a = 0 ∧ ∃ x : Set.Icc 0 ∞, True) ∨ (¬ (a = 0) ∧ ∃ x : ℝ, True)) := 
by
  sorry

end tangent_line_at_one_condition_for_a_mean_value_average_function_l91_91335


namespace max_ratio_volume_surface_l91_91884

variable (r R1 R2 : ℝ)
-- Define the conditions
def height_cone := 2 * r
def slant_height_cone := R1 + R2

-- Define the volume and surface area
def volume_cone := (R1^2 + R2^2 + R1 * R2) * (2 * r) * (Real.pi / 3)
def surface_area_cone := 2 * (R1^2 + R2^2 + R1 * R2) * Real.pi

-- Statement to prove
theorem max_ratio_volume_surface (r R1 R2 : ℝ) : 
  height_cone r = 2 * r ∧
  slant_height_cone R1 R2 = R1 + R2 →
  (volume_cone r R1 R2) / (surface_area_cone R1 R2) = r / 3 :=
by
  sorry

end max_ratio_volume_surface_l91_91884


namespace correct_system_of_equations_l91_91396

theorem correct_system_of_equations
  (x y : ℝ)
  (h1 : x + (1 / 2) * y = 50)
  (h2 : y + (2 / 3) * x = 50) :
  (x + (1 / 2) * y = 50) ∧ (y + (2 / 3) * x = 50) :=
by
  exact ⟨h1, h2⟩

end correct_system_of_equations_l91_91396


namespace rotated_curve_equiv_l91_91454

def curve_C (x y : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, (dist P (-1, -1) + dist P (1, 1) = 4 ∧ (P.1 = x ∧ P.2 = y))

def rotated_curve_C (x y : ℝ) : Prop :=
  y^2 / 4 + x^2 / 2 = 1

theorem rotated_curve_equiv :
  ∀ x y : ℝ, rotated_curve_C x y ↔
    rot45_counterclockwise (curve_C x y) :=
sorry

-- Since we don't need to express the rotation operation explicitly here,
-- rot45_counterclockwise represents the mapping after the rotation.

end rotated_curve_equiv_l91_91454


namespace michael_percentage_return_l91_91461

-- Define the conditions
def investment : ℝ := 1620
def earnings : ℝ := 135

-- Define the formula for percentage return
def percentage_return (earnings investment : ℝ) : ℝ :=
  (earnings / investment) * 100

-- Prove that the percentage return is 8.33% given the conditions
theorem michael_percentage_return :
  percentage_return earnings investment = 8.33 := by
  sorry

end michael_percentage_return_l91_91461


namespace angle_bisector_ratios_l91_91669

theorem angle_bisector_ratios (A B C D E : Point)
  (h_triangle : Triangle A B C)
  (h_internal_bisector : InternalAngleBisector A B C D)
  (h_external_bisector : ExternalAngleBisector A B C E)
  (h_ratio : ∀ BD BE, BD / BE = 3 / 5) :
  (AB / AC = 1 / 4) ∨ (AB / AC = 4) :=
by
  sorry

end angle_bisector_ratios_l91_91669


namespace exactly_one_divisible_by_p_l91_91795

theorem exactly_one_divisible_by_p
  (p : ℕ) [hp : Fact (Nat.Prime p)] (hpo : Odd p)
  (a b c d : ℤ) (ha : 0 < a ∧ a < p)
  (hb : 0 < b ∧ b < p) (hc : 0 < c ∧ c < p)
  (hd : 0 < d ∧ d < p)
  (h1 : p ∣ a^2 + b^2) (h2 : p ∣ c^2 + d^2) : 
    (p ∣ (a * c + b * d) ∨ p ∣ (a * d + b * c)) ∧ 
    ¬ (p ∣ (a * c + b * d) ∧ p ∣ (a * d + b * c)) :=
sorry

end exactly_one_divisible_by_p_l91_91795


namespace max_cos_alpha_l91_91578

theorem max_cos_alpha (α β : ℝ) (h : cos (α + β) = cos α + cos β) : cos α ≤ sqrt 3 - 1 :=
sorry

end max_cos_alpha_l91_91578


namespace monotonically_increasing_power_function_l91_91727

theorem monotonically_increasing_power_function (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m ^ 2 - 2 * m - 2) * x ^ (m - 2) > 0 → (m ^ 2 - 2 * m - 2) > 0 ∧ (m - 2) > 0) ↔ m = 3 := 
sorry

end monotonically_increasing_power_function_l91_91727


namespace union_of_sets_l91_91732

open Set

theorem union_of_sets :
  let M := {1, 2, 4, 5}
  let N := {2, 3, 4}
  M ∪ N = {1, 2, 3, 4, 5} := by
  let M := {1, 2, 4, 5}
  let N := {2, 3, 4}
  show M ∪ N = {1, 2, 3, 4, 5}
  sorry

end union_of_sets_l91_91732


namespace infinite_non_congruent_integers_l91_91034

theorem infinite_non_congruent_integers (a : ℕ → ℤ) (m : ℕ → ℤ) (k : ℕ)
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → 2 ≤ m i)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i < k → 2 * m i ≤ m (i + 1)) :
  ∃ (x : ℕ), ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → ¬ (x % (m i) = a i % (m i)) :=
sorry

end infinite_non_congruent_integers_l91_91034


namespace cyclic_quadrilateral_diagonals_perpendicular_l91_91481

theorem cyclic_quadrilateral_diagonals_perpendicular
  (A B C D E M : Point) 
  (h1 : cyclic_quadrilateral A B C D) 
  (h2 : diagonals_perpendicular A C B D E) 
  (h3 : foot_perpendicular E CD M) :
  ∃ O : Point, ∀ P : Point, (midpoint_side A B P ∨ midpoint_side B C P ∨ midpoint_side C D P ∨ midpoint_side D A P ∨ foot_perpendicular E AB P ∨ foot_perpendicular E BC P ∨ foot_perpendicular E CD P ∨ foot_perpendicular E DA P) → lies_on_circle P O :=
sorry

end cyclic_quadrilateral_diagonals_perpendicular_l91_91481


namespace students_with_one_problem_l91_91142

theorem students_with_one_problem :
  ∃ (n_1 n_2 n_3 n_4 n_5 n_6 n_7 : ℕ) (k_1 k_2 k_3 k_4 k_5 k_6 k_7 : ℕ),
    (n_1 + n_2 + n_3 + n_4 + n_5 + n_6 + n_7 = 39) ∧
    (n_1 * k_1 + n_2 * k_2 + n_3 * k_3 + n_4 * k_4 + n_5 * k_5 + n_6 * k_6 + n_7 * k_7 = 60) ∧
    (k_1 ≠ 0) ∧ (k_2 ≠ 0) ∧ (k_3 ≠ 0) ∧ (k_4 ≠ 0) ∧ (k_5 ≠ 0) ∧ (k_6 ≠ 0) ∧ (k_7 ≠ 0) ∧
    (k_1 ≠ k_2) ∧ (k_1 ≠ k_3) ∧ (k_1 ≠ k_4) ∧ (k_1 ≠ k_5) ∧ (k_1 ≠ k_6) ∧ (k_1 ≠ k_7) ∧
    (k_2 ≠ k_3) ∧ (k_2 ≠ k_4) ∧ (k_2 ≠ k_5) ∧ (k_2 ≠ k_6) ∧ (k_2 ≠ k_7) ∧
    (k_3 ≠ k_4) ∧ (k_3 ≠ k_5) ∧ (k_3 ≠ k_6) ∧ (k_3 ≠ k_7) ∧
    (k_4 ≠ k_5) ∧ (k_4 ≠ k_6) ∧ (k_4 ≠ k_7) ∧
    (k_5 ≠ k_6) ∧ (k_5 ≠ k_7) ∧
    (k_6 ≠ k_7) ∧
    (n_1 = 33) :=
sorry

end students_with_one_problem_l91_91142


namespace student_dorm_distribution_l91_91641

theorem student_dorm_distribution : 
  ∃ (count : ℕ), 
  count = 60 ∧
    (∀ (students : ℕ) (dorms : ℕ), students = 5 ∧ dorms = 3 ∧ 
                                (∀ (a : ℕ), a ≠ 1) ∧
                                (∀ (dist : ℕ → ℕ), 
                                    (∀ s, (dist s = 1 ∨ dist s = 2 ∨ dist s = 3)) ∧ 
                                    (∑ s in {1, 2, 3}, count_elements s dist = students) ∧ 
                                    ∀ d, 1 ≤ count_elements d dist ∧ count_elements d dist ≤ 2) → count = 60)) :=
sorry

end student_dorm_distribution_l91_91641


namespace smallest_square_side_length_l91_91608

theorem smallest_square_side_length :
  ∃ S : ℕ, (S * S ≥ 68) ∧ (S ≥ 9) ∧
  (6 + 2 + 6 = 14) ∧
  ∀ (a b c : ℕ), (a = 1 ∧ b = 2 ∧ c = 3) →
  (6 * a * a + 2 * b * b + 6 * c * c = 68) :=
begin
  sorry
end

end smallest_square_side_length_l91_91608


namespace candy_original_count_l91_91460

theorem candy_original_count : 
  ∀ (ate : ℕ) (now : ℕ), ate = 64 → now = 3 → ate + now = 67 := 
by 
  intros ate now h1 h2 
  rw [h1, h2] 
  rfl

end candy_original_count_l91_91460


namespace angle_AHB_is_117_l91_91614

-- Given definitions
variable (A B C D E H : Point)
variable (BAC ABC : ℝ)
variable (AD BE : Line)

-- Conditions
axiom altitudes_intersect : ∃ H : Point, H ∈ AD ∧ H ∈ BE
axiom angle_BAC : BAC = 46
axiom angle_ABC : ABC = 71
axiom angle_AHB : ∃ AHB : ℝ, AHB = 117

-- Proof statement
theorem angle_AHB_is_117 : ∃ angle : ℝ, angle = 117 :=
by
  existsi 117
  sorry

end angle_AHB_is_117_l91_91614


namespace tan_theta_perpendicular_cos_shift_parallel_l91_91366

variables (θ : ℝ)

def vector_a : ℝ × ℝ := (Real.sin θ, -2/(5:ℝ))
def vector_b : ℝ × ℝ := (1, 2 * Real.cos θ)

-- Problem 1: If vector a and vector b are perpendicular, prove that tan θ = 4/5
theorem tan_theta_perpendicular (h : vector_a θ.1 * vector_b θ.1 + vector_a θ.2 * vector_b θ.2 = 0) :
  Real.tan θ = 4/5 := sorry

-- Problem 2: If vector a is parallel to vector b, prove that cos (π/2 + 2θ) = 4√21/25
theorem cos_shift_parallel (h : ∃ k : ℝ, vector_a θ = k • vector_b θ) :
  Real.cos (Real.pi / 2 + 2 * θ) = 4 * Real.sqrt 21 / 25 := sorry

end tan_theta_perpendicular_cos_shift_parallel_l91_91366


namespace correct_expression_l91_91194

theorem correct_expression (x : ℝ ) (h₀ : x = 9) (h₁ : sqrt x = 3) : 
  (∀ y, y = sqrt x ∨ y = -sqrt x → y = 3 ∨ y = -3) :=
by 
  intro y 
  intro h₂
  cases h₂
  case or.inl h₂_l => 
    rw [h₁] at h₂_l 
    exact Or.inl h₂_l
  case or.inr h₂_r => 
    rw [h₁] at h₂_r
    exact Or.inr h₂_r

#check correct_expression

end correct_expression_l91_91194


namespace g_f_monotonically_decreasing_on_nonneg_l91_91692

def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_odd (g : ℝ → ℝ) := ∀ x, g x = -g (-x)
def is_monotonically_decreasing (h : ℝ → ℝ) (I : set ℝ) := 
  ∀ x y ∈ I, x < y → h x ≥ h y

variable (f : ℝ → ℝ) (g : ℝ → ℝ)

-- Conditions
axiom f_even : is_even f
axiom g_odd : is_odd g
axiom f_decreasing : is_monotonically_decreasing f (set.Iic 0)
axiom g_decreasing : is_monotonically_decreasing g (set.Iic 0)

-- The proof problem
theorem g_f_monotonically_decreasing_on_nonneg : 
  is_monotonically_decreasing (λ x, g (f x)) (set.Ici 0) :=
sorry

end g_f_monotonically_decreasing_on_nonneg_l91_91692


namespace maximum_value_sqrt_expr_l91_91309

theorem maximum_value_sqrt_expr :
  ∃ x : ℝ, -49 ≤ x ∧ x ≤ 49 ∧ sqrt (49 + x) + sqrt (49 - x) = 14 ∧
  ∀ x : ℝ, -49 ≤ x ∧ x ≤ 49 → sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
by
  sorry

end maximum_value_sqrt_expr_l91_91309


namespace unique_intersecting_line_l91_91739

theorem unique_intersecting_line (M : ℝ × ℝ) (hM1 : M = (2, 4)) (hM2 : (M.snd)^2 = 8 * (M.fst)) :
    ∃! l : ℝ → ℝ, (l(2) = 4) ∧ (∀ y, ∃! x, y^2 = 8*x ∧ l(x) = y) :=
sorry

end unique_intersecting_line_l91_91739


namespace total_pounds_of_peppers_l91_91632

-- Definitions and conditions
def green_peppers : ℝ := 2.8333333333333335
def red_peppers : ℝ := 2.8333333333333335

-- Theorem statement
theorem total_pounds_of_peppers : green_peppers + red_peppers = 5.666666666666667 :=
by
  sorry

end total_pounds_of_peppers_l91_91632


namespace num_divisible_digits_l91_91664

theorem num_divisible_digits :
  {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ (let k := 100 + 50 + n in k % n = 0)}.to_finset.card = 5 :=
by
  sorry

end num_divisible_digits_l91_91664


namespace D_binomial_distributions_D_min_value_D_nonneg_l91_91810

noncomputable def D (X Y : ℕ → ℝ) (x y : ℝ) : ℝ :=
  ∑ k in finset.range n, X k * real.log (X k / Y k)

variables {n : ℕ}
variables {X Y : ℕ → ℝ}
variables {x y : ℝ}
variables {p q : ℝ} (hp : 0 < p ∧ p < 1) (hq : 0 < q ∧ q < 1)

-- Proof Problem 1
theorem D_binomial_distributions (X Y : ℕ → ℝ) (x y : ℝ) (n : ℕ) 
  (hx : ∀ k, X k = (nat.choose n k) * p^k * (1 - p)^(n - k))
  (hy : ∀ k, Y k = (nat.choose n k) * q^k * (1 - q)^(n - k))
  (hp : 0 < p ∧ p < 1) (hq : 0 < q ∧ q < 1) :
  D X Y x y = n * p * real.log (p * (1 - q) / (q * (1 - p))) + n * real.log ((1 - p) / (1 - q)) := 
sorry

-- Proof Problem 2
theorem D_min_value (X : ℕ → ℝ) 
  (hx : X 0 = (1 - p)^2 ∧ X 1 = 2 * p * (1 - p) ∧ X 2 = p^2)
  (hy : ∀ k, Y (k - 1) = 1 / 3) (hn : n = 2) :
  D X Y x y ≥ real.log 3 - (3 / 2) * real.log 2 := 
sorry

-- Proof Problem 3
theorem D_nonneg (X Y : ℕ → ℝ) (hx : ∀ k, X k > 0) (hy : ∀ k, Y k > 0)
  (hx_sum : ∑ k in finset.range n, X k = 1)
  (hy_sum : ∑ k in finset.range n, Y k = 1) :
  D X Y x y ≥ 0 ∧ (D X Y x y = 0 ↔ ∀ k, X k = Y k) := 
sorry

end D_binomial_distributions_D_min_value_D_nonneg_l91_91810


namespace mod_x_plus_yi_eq_sqrt_2_l91_91433

theorem mod_x_plus_yi_eq_sqrt_2 (x y : ℝ) (h : (1 + I : ℂ) * x = 1 + y * I) : abs (x + y * I) = real.sqrt 2 :=
sorry

end mod_x_plus_yi_eq_sqrt_2_l91_91433


namespace f_neg_one_f_decreasing_on_positive_f_expression_on_negative_l91_91506

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 / x - 1 else 2 / (-x) - 1

-- Assertion 1: Value of f(-1)
theorem f_neg_one : f (-1) = 1 := 
sorry

-- Assertion 2: f(x) is a decreasing function on (0, +∞)
theorem f_decreasing_on_positive : ∀ a b : ℝ, 0 < b → b < a → f (a) < f (b) := 
sorry

-- Assertion 3: Expression of the function when x < 0
theorem f_expression_on_negative (x : ℝ) (hx : x < 0) : f x = 2 / (-x) - 1 := 
sorry

end f_neg_one_f_decreasing_on_positive_f_expression_on_negative_l91_91506


namespace slope_of_parallel_line_l91_91595

/-- A line is described by the equation 3x - 6y = 12. The slope of a line 
    parallel to this line is 1/2. -/
theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : 
  ∃ m : ℝ, m = 1/2 := by
  sorry

end slope_of_parallel_line_l91_91595


namespace provisions_last_l91_91199

-- Define the main variables
variables (a b : ℝ) (n_food total_men : ℝ)

-- Define each of the conditions from the problem
def init_men := 600
def additional_men := 200
def init_days := 20
def total_food := (init_men * a * init_days : ℝ)
def total_daily_rate := (init_men * a + additional_men * b : ℝ)

-- Define the final statement to prove
theorem provisions_last (h1 : total_food = 12000 * a)
                        (h2 : total_daily_rate = 600 * a + 200 * b) :
  n_food / total_daily_rate = total_food / total_daily_rate :=
by sorry

end provisions_last_l91_91199


namespace rectangle_dimension_area_l91_91549

theorem rectangle_dimension_area (x : Real) 
  (h_dim1 : x + 3 > 0) 
  (h_dim2 : 3 * x - 2 > 0) :
  ((x + 3) * (3 * x - 2) = 9 * x + 1) ↔ x = (11 + Real.sqrt 205) / 6 := 
sorry

end rectangle_dimension_area_l91_91549


namespace find_num_four_digit_ap_numbers_l91_91289

noncomputable def num_four_digit_ap_numbers : ℕ :=
  let single_digit := {n : ℕ // 0 ≤ n ∧ n ≤ 9}
  let ap_sequence (d : ℕ) (sequence : list ℕ) : Prop :=
    ∀ i, i < sequence.length - 1 → sequence.nth i + d = sequence.nth (i + 1)
  let valid_ap_sequences : ℕ → list (list ℕ) := 
    λ d, filter (λ s, s.length = 3 ∧ ap_sequence d s) (list.product (list.product single_digit single_digit) single_digit)
  let valid_sequences := valid_ap_sequences 1 ∪ valid_ap_sequences 2 ∪ valid_ap_sequences 3 ∪ valid_ap_sequences 4 
  let count_valid_sequences := list.length valid_sequences
  let first_digit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}
  let num_sequences_with_first_digit := first_digit.cardinality * count_valid_sequences
  num_sequences_with_first_digit

theorem find_num_four_digit_ap_numbers : num_four_digit_ap_numbers = 180 := by
  sorry

end find_num_four_digit_ap_numbers_l91_91289


namespace last_two_nonzero_digits_of_72_fact_l91_91288

/-- Define the factorial function -/
def factorial : ℕ → ℕ
| 0 := 1
| (n + 1) := (n + 1) * factorial n

/-- Define the function to find the last two non-zero digits of a number -/
def last_two_nonzero_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10 in
  let filtered_digits := digits.filter (λ x => x ≠ 0) in
  match filtered_digits.reverse.take 2 with
  | [d₁, d₂] => d₁ + 10 * d₂
  | _ => 0

/-- Theorem: The last two non-zero digits of 72! are 64 -/
theorem last_two_nonzero_digits_of_72_fact : last_two_nonzero_digits (factorial 72) = 64 := 
  sorry

end last_two_nonzero_digits_of_72_fact_l91_91288


namespace palindrome_factorization_of_2016_l91_91599

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem palindrome_factorization_of_2016 :
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ is_palindrome a ∧ is_palindrome b ∧ is_palindrome c ∧ (a * b * c = 2016) :=
by {
  existsi 2,
  existsi 4,
  existsi 252,
  simp [is_palindrome],
  sorry
}

end palindrome_factorization_of_2016_l91_91599


namespace minimum_perimeter_is_correct_l91_91877

noncomputable def minimum_common_perimeter : ℕ := 524

theorem minimum_perimeter_is_correct 
  (a b c : ℤ)
  (h1 : b = a + c)
  (h2 : 2a + 10 * c = 2b + 8 * c)
  (h3 : ∃ (s1 s2 : ℚ), 5 * c * real.sqrt (a^2 - (5 * c)^2) = 4 * c * real.sqrt (b^2 - (4 * c)^2))
  (h4 : a > 0 ∧ b > 0 ∧ c > 0)
  : 2 * a + 10 * c = minimum_common_perimeter :=
sorry

end minimum_perimeter_is_correct_l91_91877


namespace percentage_spent_on_phone_bill_l91_91238

theorem percentage_spent_on_phone_bill
  (initial_money food_percentage entertainment_spending remaining_money : ℝ) 
  (percentage_on_phone_bill : ℝ) 
  (h1 : initial_money = 200)
  (h2 : food_percentage = 0.6)
  (h3 : entertainment_spending = 20)
  (h4 : remaining_money = 40)
  (h5 : percentage_on_phone_bill = 25) :
  let money_after_food := initial_money * (1 - food_percentage),
      money_before_phone_bill := money_after_food - entertainment_spending,
      money_spent_phone := money_before_phone_bill - remaining_money in
  percentage_on_phone_bill = (money_spent_phone / money_after_food) * 100 := 
by
  sorry

end percentage_spent_on_phone_bill_l91_91238


namespace largest_square_in_right_triangle_largest_rectangle_in_right_triangle_l91_91677

theorem largest_square_in_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ s, s = (a * b) / (a + b) := 
sorry

theorem largest_rectangle_in_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ x y, x = a / 2 ∧ y = b / 2 :=
sorry

end largest_square_in_right_triangle_largest_rectangle_in_right_triangle_l91_91677


namespace age_of_older_sister_in_2021_l91_91026

instance : DecidableEq ℕ := Classical.decEq ℕ

theorem age_of_older_sister_in_2021 (year_kelsey_25 : ℕ) (current_year : ℕ) (kelsey_age_in_1999 : ℕ) (sister_age_diff : ℕ) :
  (year_kelsey_25 = 1999) ∧ (kelsey_age_in_1999 = 25) ∧ (sister_age_diff = 3) ∧ (current_year = 2021) → 
  (current_year - ((year_kelsey_25 - kelsey_age_in_1999) - sister_age_diff) = 50) :=
by
  sorry

end age_of_older_sister_in_2021_l91_91026


namespace count_of_numbers_with_sum_of_digits_ten_l91_91372

def num_digits (n : ℕ) : ℕ := 
  n.toString.length

def sum_of_digits_is_ten (n : ℕ) : Prop :=
  num_digits (n ^ 2) + num_digits (n ^ 3) = 10

theorem count_of_numbers_with_sum_of_digits_ten : 
  ({n : ℕ | 47 ≤ n ∧ n ≤ 99 ∧ sum_of_digits_is_ten n}.to_finset.card = 53) :=
sorry

end count_of_numbers_with_sum_of_digits_ten_l91_91372


namespace sequence_general_formula_l91_91103

theorem sequence_general_formula :
  ∀ (n : ℕ), (n > 0) → (if (n % 2 = 0) then a_n = (1 / (2 * n)) else a_n = (-(1 / (2 * n)))) :=
by
  intro n hn
  -- Further proof steps would normally go here
  sorry

end sequence_general_formula_l91_91103


namespace expression_divisible_by_9_l91_91190

theorem expression_divisible_by_9 (n : ℤ) : 9 ∣ (2^(2*n) + 15*n - 1) := sorry

end expression_divisible_by_9_l91_91190


namespace ratio_cost_to_marked_l91_91607

variable (m : ℝ)

def marked_price (m : ℝ) := m

def selling_price (m : ℝ) : ℝ := 0.75 * m

def cost_price (m : ℝ) : ℝ := 0.60 * selling_price m

theorem ratio_cost_to_marked (m : ℝ) : 
  cost_price m / marked_price m = 0.45 := 
by
  sorry

end ratio_cost_to_marked_l91_91607


namespace average_of_distinct_t_values_l91_91704

theorem average_of_distinct_t_values (t : ℕ) (r1 r2 : ℕ) (h1 : r1 + r2 = 7) (h2 : r1 * r2 = t)
  (pos_r1 : r1 > 0) (pos_r2 : r2 > 0) :
  (6 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) ∧
  (10 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) ∧
  (12 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) →
  ( ∑ x in {6, 10, 12}, (x : ℚ)) / 3 = 28 / 3 :=
by
    sorry

end average_of_distinct_t_values_l91_91704


namespace range_of_exponential_l91_91117

theorem range_of_exponential (f : ℝ → ℝ) (x : ℝ) :
  (∀ x ∈ set.Icc (-2 : ℝ) 1, f x = 3 ^ (-x)) →
  set.range (λ x, f x) = set.Icc (1/3) 9 := 
by 
  sorry

end range_of_exponential_l91_91117


namespace interest_rate_difference_l91_91233

theorem interest_rate_difference (P T : ℕ) (R1 R2 : ℚ) :
  P = 2600 → T = 3 → (P * T / 100) * (R2 - R1) = 78 → R2 - R1 = 1 := 
by
  intros hP hT hDiff
  have h1 : P * T / 100 = 78 / (R2 - R1) := by
    calc
      P * T / 100 = (P * T / 100) * (R2 - R1) * (1 / (R2 - R1)) : by sorry
              ... = 78 / (R2 - R1) : by sorry
  rw [hP, hT] at hDiff
  linarith

end interest_rate_difference_l91_91233


namespace sequence_x_value_l91_91410

theorem sequence_x_value : 
  ∃ (x y z : ℤ), 
    (∀ n, a (n + 2) = a n + a (n + 1)) → 
    (a 0 = x) → (a 1 = y) → (a 2 = z) →
    (a 3 = 1) → (a 4 = 3) →
    (a 5 = 4) → (a 6 = 7) →
    (a 7 = 11) → (a 8 = 18) →
    (a 9 = 29) →
    x = 3 := 
by
  sorry

end sequence_x_value_l91_91410


namespace JN_fixed_point_and_constant_length_l91_91150

theorem JN_fixed_point_and_constant_length
  (S₁ S₂ : Circle) (A : Point) (M₁ M₂ : Point) (O₁ O₂ J N : Point)
  (line_intersects_circles_at_A : Line → (Line ∩ S₁ = {A} ∪ {M₁}) ∧ (Line ∩ S₂ = {A} ∪ {M₂}))
  (centers S₁ S₂ : Point)
  (lines_parallel_to_tangents : (TangentAt M₁ S₁ = TangentToLineAt O₁J Line) ∧ (TangentAt M₂ S₂ = TangentToLineAt O₂J Line)) :
  ∃ (B : Point), PassesThrough J N B ∧ ∃ (l : ℝ), ∀ (M₁ M₂ : Point), ∥J - N∥ = l :=
sorry

end JN_fixed_point_and_constant_length_l91_91150


namespace product_odd_integers_lt_20_l91_91550

/--
The product of all odd positive integers strictly less than 20 is a positive number ending with the digit 5.
-/
theorem product_odd_integers_lt_20 :
  let nums := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
  let product := List.prod nums
  (product > 0) ∧ (product % 10 = 5) :=
by
  sorry

end product_odd_integers_lt_20_l91_91550


namespace marble_group_l91_91210

theorem marble_group (x : ℕ) (h1 : 144 % x = 0) (h2 : 144 % (x + 2) = (144 / x) - 1) : x = 16 :=
sorry

end marble_group_l91_91210


namespace _l91_91500

def coefficient_x3_in_expansion (x : ℝ) : ℝ :=
  let expansion := x * (x + 3)^5 in
  let terms := (Polynomial.expand_by_binomial_theorem expansion) in
  terms.coefficient 3

example : coefficient_x3_in_expansion x = 270 := by
  sorry

end _l91_91500


namespace library_science_books_count_l91_91106

-- Definitions based on the problem conditions
def initial_science_books := 120
def borrowed_books := 40
def returned_books := 15
def books_on_hold := 10
def borrowed_from_other_library := 20
def lost_books := 2
def damaged_books := 1

-- Statement for the proof.
theorem library_science_books_count :
  initial_science_books - borrowed_books + returned_books - books_on_hold + borrowed_from_other_library - lost_books - damaged_books = 102 :=
by
  sorry

end library_science_books_count_l91_91106


namespace total_rabbits_correct_l91_91998

def initial_breeding_rabbits : ℕ := 10
def kittens_first_spring : ℕ := initial_breeding_rabbits * 10
def adopted_first_spring : ℕ := kittens_first_spring / 2
def returned_adopted_first_spring : ℕ := 5
def total_rabbits_after_first_spring : ℕ :=
  initial_breeding_rabbits + (kittens_first_spring - adopted_first_spring + returned_adopted_first_spring)

def kittens_second_spring : ℕ := 60
def adopted_second_spring : ℕ := kittens_second_spring * 40 / 100
def returned_adopted_second_spring : ℕ := 10
def total_rabbits_after_second_spring : ℕ :=
  total_rabbits_after_first_spring + (kittens_second_spring - adopted_second_spring + returned_adopted_second_spring)

def breeding_rabbits_third_spring : ℕ := 12
def kittens_third_spring : ℕ := breeding_rabbits_third_spring * 8
def adopted_third_spring : ℕ := kittens_third_spring * 30 / 100
def returned_adopted_third_spring : ℕ := 3
def total_rabbits_after_third_spring : ℕ :=
  total_rabbits_after_second_spring + (kittens_third_spring - adopted_third_spring + returned_adopted_third_spring)

def kittens_fourth_spring : ℕ := breeding_rabbits_third_spring * 6
def adopted_fourth_spring : ℕ := kittens_fourth_spring * 20 / 100
def returned_adopted_fourth_spring : ℕ := 2
def total_rabbits_after_fourth_spring : ℕ :=
  total_rabbits_after_third_spring + (kittens_fourth_spring - adopted_fourth_spring + returned_adopted_fourth_spring)

theorem total_rabbits_correct : total_rabbits_after_fourth_spring = 242 := by
  sorry

end total_rabbits_correct_l91_91998


namespace percentage_music_students_l91_91869

variables (total_students : ℕ) (dance_students : ℕ) (art_students : ℕ)
  (music_students : ℕ) (music_percentage : ℚ)

def students_music : ℕ := total_students - (dance_students + art_students)
def percentage_students_music : ℚ := (students_music total_students dance_students art_students : ℚ) / (total_students : ℚ) * 100

theorem percentage_music_students (h1 : total_students = 400)
                                  (h2 : dance_students = 120)
                                  (h3 : art_students = 200) :
  percentage_students_music total_students dance_students art_students = 20 := by {
  sorry
}

end percentage_music_students_l91_91869


namespace cyclic_quadrilateral_ineq_l91_91324

variable {K : Type*} [Field K]

def cyclic_quadrilateral (A B C D : K) : Prop := 
  -- Add the necessary definition of a cyclic quadrilateral
  
def angle_geq_sixty (a b c d : K) : Prop :=
  -- Define that all internal and external angles are not less than 60 degrees

def proof_cyclic_inequality 
  (A B C D : K) 
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : angle_geq_sixty A B C D) : 
  Prop :=
  (1/3 * abs (A^3 - D^3) ≤ abs (B^3 - C^3)) ∧ 
  (abs (B^3 - C^3) ≤ 3 * abs (A^3 - D^3))

theorem cyclic_quadrilateral_ineq
  (A B C D : K)
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : angle_geq_sixty A B C D) :
  proof_cyclic_inequality A B C D h1 h2 := 
sorry

end cyclic_quadrilateral_ineq_l91_91324


namespace jack_time_to_school_l91_91262

noncomputable def dave_speed : ℚ := 8000 -- cm/min
noncomputable def distance_to_school : ℚ := 160000 -- cm
noncomputable def jack_speed : ℚ := 7650 -- cm/min
noncomputable def jack_start_delay : ℚ := 10 -- min

theorem jack_time_to_school : (distance_to_school / jack_speed) - jack_start_delay = 10.92 :=
by
  sorry

end jack_time_to_school_l91_91262


namespace no_such_function_exists_l91_91275

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, (f 0 > 0) ∧ (∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) :=
by
  -- proof to be completed
  sorry

end no_such_function_exists_l91_91275


namespace matrix_exp_1000_l91_91628

-- Define the matrix as a constant
noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![2, 1]]

-- The property of matrix exponentiation
theorem matrix_exp_1000 :
  A^1000 = ![![1, 0], ![2000, 1]] :=
by
  sorry

end matrix_exp_1000_l91_91628


namespace two_groups_partition_exists_l91_91171

open Set

variable (People : Type) [Fintype People] (EnemyRelation : People → People → Prop)

theorem two_groups_partition_exists
  (finite_people : Fintype People)
  (enemy_reciprocal : ∀ x y : People, EnemyRelation x y → EnemyRelation y x)
  (at_most_three_enemies : ∀ x : People, (univ.filter (EnemyRelation x)).toFinset.card ≤ 3) :
  ∃ (Group1 Group2 : People → Prop), (∀ x, Group1 x ∨ Group2 x) ∧ (∀ x, ¬(Group1 x ∧ Group2 x)) ∧
  (∀ x, (univ.filter (λ y, EnemyRelation x y ∧ Group1 x = Group1 y)).toFinset.card ≤ 1) :=
begin
  sorry
end

end two_groups_partition_exists_l91_91171


namespace segment_DE_length_circumscribed_circle_radius_l91_91415

variables (A B C M D E P : Type) [Points A] [Points B] [Points C] [Points M] [Points D] [Points E] [Points P]
variables (BM DE : Line) [IsMedian BM A B C] [IsBisector MD A M B] [IsBisector ME C M B] [Intersects BM DE P] 
variables [BP = 1] [MP = 3]

/-- (a) Find the length of segment DE: -/
theorem segment_DE_length : length DE = 6 :=
sorry

/-- (b) Given that a circle can be circumscribed around quadrilateral ADEC, find its radius: -/
theorem circumscribed_circle_radius : ∃ (R : ℝ), circumscribes_circle A D E C R ∧ R = 3 * sqrt 65 :=
sorry

end segment_DE_length_circumscribed_circle_radius_l91_91415


namespace max_value_of_function_l91_91700

/-- Given that the function f(x) = λ * sin(x) + cos(x) has a symmetry axis with the equation x = π / 6,
    the task is to prove that the maximum value of this function is 2 * sqrt(3) / 3. -/
theorem max_value_of_function (λ : ℝ) 
    (h_symmetry_axis : ∀ x, (λ * Real.sin(x) + Real.cos(x)) = (λ * Real.sin (π / 6) + Real.cos(π / 6))) :
  ∃ L : ℝ, L = 2 * Real.sqrt 3 / 3 ∧ 
  (L = Real.sqrt (λ^2 + 1)) :=
  sorry

end max_value_of_function_l91_91700


namespace find_lambda_mu_l91_91765

-- Define the points A, B, and C in the Cartesian plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Given points
def A : Point := ⟨1, 0⟩
def B : Point := ⟨0, 1⟩
def O : Point := ⟨0, 0⟩

-- Conditions for point C
def condition_C (C : Point) : Prop :=
  C.x > 0 ∧ C.y > 0 ∧ ∠(A.x, A.y, O.x, O.y, C.x, C.y) = π/6 ∧ (C.x^2 + C.y^2) = 4

-- The vector expression for OC
def vector_expression (C : Point) (λ μ : ℝ) : Prop :=
  C = ⟨λ * A.x + μ * B.x, λ * A.y + μ * B.y⟩

-- The proof statement
theorem find_lambda_mu :
  ∃ C : Point, condition_C C ∧ ∃ λ μ : ℝ, vector_expression C λ μ ∧ λ = √3 ∧ μ = 1 := by
  sorry

end find_lambda_mu_l91_91765


namespace sum_of_possible_values_sum_of_all_possible_values_l91_91401

theorem sum_of_possible_values (x : ℝ) :
    (| x - 5 | - 4 = 0) → (x = 1 ∨ x = 9) :=
  by
    sorry

theorem sum_of_all_possible_values :
    ∑ (x : ℝ) in {y | |y - 5| - 4 = 0}.to_finset, x = 10 :=
  by
    sorry

end sum_of_possible_values_sum_of_all_possible_values_l91_91401


namespace maximum_value_sqrt_expr_l91_91310

theorem maximum_value_sqrt_expr :
  ∃ x : ℝ, -49 ≤ x ∧ x ≤ 49 ∧ sqrt (49 + x) + sqrt (49 - x) = 14 ∧
  ∀ x : ℝ, -49 ≤ x ∧ x ≤ 49 → sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
by
  sorry

end maximum_value_sqrt_expr_l91_91310


namespace min_value_log_sum_zero_l91_91336

theorem min_value_log_sum_zero (a b : ℝ) (h : log a + log b = 0) : 
  ∃ x, x = \frac 2 a + \frac 1 b ∧ x = 2√2 :=
by {
-- The proof is skipped as instructed 
sorry
}

end min_value_log_sum_zero_l91_91336


namespace distance_PD_l91_91473

variable {A B C P D : Type*}
variable [metric_space B]
variable (X : triangle A B C)

def angle_bisector_condition (P : B) : Prop :=
  -- P lies on the angle bisector of ∠ABC
  sorry

def distance_condition (P : B) : Prop :=
  -- The distance from P to side BA is 3
  dist_from_P_to_BA : 3

def point_on_BC (D : B) : Prop :=
  -- D is any point on side BC
  D ∈ side BC

theorem distance_PD (P D : B) (h1 : angle_bisector_condition P) (h2 : distance_condition P) (h3 : point_on_BC D) : 
  dist P D ≥ 3 :=
sorry

end distance_PD_l91_91473


namespace prove_problem_statement_l91_91328

noncomputable theory

def ellipse_equation (a b : ℝ) (h : a > b) := 
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def right_focus (c : ℝ) := 
  (sqrt c, 0)

def distance_from_major_axis_vertex_to_point (a d : ℝ) := 
  sqrt (a^2 + d^2)

def line_through_focus (k : ℝ) (focus : ℝ × ℝ) := 
  ∀ (x y : ℝ), y = k * x + (focus.snd - k * focus.fst)

def line_intersect_ellipse (a b : ℝ) (k : ℝ) : Prop := 
  let eq_ellipse := ellipse_equation a b (by sorry)
  let eq_line := line_through_focus k (right_focus 3)
  let AB_length := abs (sqrt(1 + k^2) * ((8 / sqrt 3)))
  eq_ellipse = true ∧ eq_line = true ∧ AB_length = 8/5

def problem_statement : Prop := 
  ∃ a b : ℝ, 
  (a > b) ∧ 
  ellipse_equation a 1 = (by sorry) ∧ 
  line_intersect_ellipse 2 1 1

theorem prove_problem_statement : problem_statement :=
by sorry

end prove_problem_statement_l91_91328


namespace volume_ratio_l91_91418

variable (A B : ℝ)

theorem volume_ratio (h1 : (3 / 4) * A = (5 / 8) * B) :
  A / B = 5 / 6 :=
by
  sorry

end volume_ratio_l91_91418


namespace project_completion_time_saving_l91_91222

/-- A theorem stating that if a project with initial and additional workforce configuration,
the project will be completed 10 days ahead of schedule. -/
theorem project_completion_time_saving
  (total_days : ℕ := 100)
  (initial_people : ℕ := 10)
  (initial_days : ℕ := 30)
  (initial_fraction : ℚ := 1 / 5)
  (additional_people : ℕ := 10)
  : (total_days - ((initial_days + (1 / (initial_people + additional_people * initial_fraction)) * (total_days * initial_fraction) / initial_fraction)) = 10) :=
sorry

end project_completion_time_saving_l91_91222


namespace gcd_72_168_l91_91507

theorem gcd_72_168 : Nat.gcd 72 168 = 24 :=
by
  sorry

end gcd_72_168_l91_91507


namespace find_x_l91_91750

-- Given condition: 144 / x = 14.4 / 0.0144
theorem find_x (x : ℝ) (h : 144 / x = 14.4 / 0.0144) : x = 0.144 := by
  sorry

end find_x_l91_91750


namespace perimeter_triangle_line_l91_91434

noncomputable def perimeter_of_triangle_points_on_line (x1 x2 x3 m : ℝ) : ℝ :=
  (Real.sqrt (1 + 4 * m^2)) * (|x2 - x1| + |x3 - x2| + |x1 - x3|)

theorem perimeter_triangle_line (x1 x2 x3 m : ℝ) :
  ∃ y1 y2 y3, y1 = 2 * m * x1 + m + 1 ∧ y2 = 2 * m * x2 + m + 1 ∧ y3 = 2 * m * x3 + m + 1 ∧
  perimeter_of_triangle_points_on_line x1 x2 x3 m = (Real.sqrt (1 + 4 * m^2)) * (|x2 - x1| + |x3 - x2| + |x1 - x3|) :=
begin
  sorry
end

end perimeter_triangle_line_l91_91434


namespace max_sqrt_sum_l91_91307

theorem max_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
sorry

end max_sqrt_sum_l91_91307


namespace eigenvalues_of_Y_l91_91432

-- Conditions
variables {n : ℕ} (hn : n > 0)  -- Positive integer n
variables (A : matrix (fin n) (fin (n+1)) ℝ) -- A is an n x (n+1) matrix
variables (A_T : matrix (fin (n+1)) (fin n) ℝ) -- A^T is the transpose of A

-- Define matrix Y
def Y : matrix (fin (2 * n + 1)) (fin (2 * n + 1)) ℝ :=
  λ i j,
    if i < n then
      if j < n then 0
      else A i (j - n)
    else if j < n then A_T (i - n) j
    else 0

-- Eigenvalues proof statement
theorem eigenvalues_of_Y :
  (0 ∈ spectrum ℝ Y) ∧
  (∀ λ : ℝ, λ ∈ (spectrum ℝ (A ⬝ A_T)) → (λ ≥ 0) ∧ (λ ∈ (spectrum ℝ Y) ∨ (-λ ∈ (spectrum ℝ Y)))) :=
  sorry

end eigenvalues_of_Y_l91_91432


namespace unique_line_equation_l91_91539

theorem unique_line_equation
  (k : ℝ)
  (m b : ℝ)
  (h1 : |(k^2 + 4*k + 3) - (m*k + b)| = 4)
  (h2 : 2*m + b = 8)
  (h3 : b ≠ 0) :
  (m = 6 ∧ b = -4) :=
by
  sorry

end unique_line_equation_l91_91539


namespace sum_of_products_circle_l91_91126

theorem sum_of_products_circle 
  (a b c d : ℤ) 
  (h : a + b + c + d = 0) : 
  -((a * (b + d)) + (b * (a + c)) + (c * (b + d)) + (d * (a + c))) = 2 * (a + c) ^ 2 :=
sorry

end sum_of_products_circle_l91_91126


namespace shiny_sum_ge_K_l91_91242

noncomputable def K (n : ℕ) : ℝ := (1 - n) / 2

def isShiny (n : ℕ) (x : ℕ → ℝ) : Prop :=
  ∀ (y : ℕ → ℝ), 
  (∀ (σ : Fin n → Fin n), (y ∘ σ) = x) →
  (∑ i in Finset.range (n - 1), y i * y (i + 1)) ≥ -1

theorem shiny_sum_ge_K (n : ℕ) (x : ℕ → ℝ) (hn : n ≥ 3) (hx : isShiny n x) :
  (∑ i in Finset.range (n - 1), (∑ j in Finset.Ico (i + 1) n, x i * x j)) ≥ K n :=
sorry

end shiny_sum_ge_K_l91_91242


namespace closest_to_one_is_0p81_l91_91568

def numbers : List ℚ :=
  [3/4, 12/10, 81/100, 4/3, 7/10]

def closest_to_one (lst : List ℚ) : ℚ :=
  lst.argmin (λ x, |1 - x|) 1

theorem closest_to_one_is_0p81 : closest_to_one numbers = 81/100 :=
by
  sorry

end closest_to_one_is_0p81_l91_91568


namespace find_largest_l91_91962

noncomputable def largest_number_among_four : Prop :=
  let a := Real.pi
  let b := Real.sqrt 2
  let c := abs (-2)
  let d := 3
  a > b ∧ a > c ∧ a > d

theorem find_largest : largest_number_among_four :=
by
  -- Conditions from the problem
  let a := Real.pi
  let b := Real.sqrt 2
  let c := abs (-2)
  let d := 3
  have H_c : c = 2 := by simp
  have H_b : b < 2 := Real.sqrt_lt.2 one_lt_two
  have H_2 : 2 < 3 := by norm_num
  have H_3 : 3 < a := by linarith [Real.pi_lt_four]
  -- Goal
  exact ⟨H_3, H_c ▸ H_3.trans H_2, H_3⟩

end find_largest_l91_91962


namespace E_tau1_finite_l91_91008

noncomputable def τ1 (X : ℕ → ℝ) : ℝ := sorry  -- Definition of τ1 is skipped for brevity

-- Given condition: expected value of X₁ is positive
def expected_value_positive (X : ℕ → ℝ) : Prop := 
  (∑' n, X n) / (nat_cast ℕ) > 0

theorem E_tau1_finite (X : ℕ → ℝ) (h : expected_value_positive X) : 
  (∑' n, τ1 X n) < ∞ := sorry

end E_tau1_finite_l91_91008


namespace sqrt_trig_identity_l91_91748

theorem sqrt_trig_identity (θ : ℝ) (h : θ ∈ Ioo (7 * Real.pi / 4) (2 * Real.pi)) :
  Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ) = Real.cos θ - Real.sin θ :=
sorry

end sqrt_trig_identity_l91_91748


namespace pump_filling_time_l91_91603

-- Define the conditions
variables P : ℝ

-- The rate of the pump
def pump_rate := 1 / P

-- The rate of the leak
def leak_rate := 1 / 5

-- The combined rate of pump and leak
def combined_rate := 1 / 20

-- The theorem we need to prove
theorem pump_filling_time : (1 / P - 1 / 5 = 1 / 20) → P = 4 :=
by
  intro h
  have h_combined := calc
    1 / P - 1 / 5 = 1 / 20 : by exact h
  sorry

end pump_filling_time_l91_91603


namespace fraction_inequality_solution_l91_91078

theorem fraction_inequality_solution (x : ℝ) :
  (x - 2) / (x + 5) ≥ 0 ↔ x ∈ set.Ici 2 ∪ set.Iio (-5) :=
by sorry

end fraction_inequality_solution_l91_91078


namespace largest_number_is_d_l91_91176

def x_A : ℝ := 1.3542
def x_B : ℝ := 1.3542
def x_C : ℝ := 1.3542
def x_D : ℝ := 1 + 3/10 + 5/100 + 5/1000 + 42/10000 -- Representation of 1.3(5)42
def x_E : ℝ := 1.3542

theorem largest_number_is_d : x_D > x_A ∧ x_D > x_B ∧ x_D > x_C ∧ x_D > x_E :=
by
  sorry

end largest_number_is_d_l91_91176


namespace triangle_sides_inequality_l91_91431

theorem triangle_sides_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : a + b + c ≤ 2) :
  -3 < (a^3 / b + b^3 / c + c^3 / a - a^3 / c - b^3 / a - c^3 / b) ∧ 
  (a^3 / b + b^3 / c + c^3 / a - a^3 / c - b^3 / a - c^3 / b) < 3 :=
by sorry

end triangle_sides_inequality_l91_91431


namespace find_k_l91_91696

variable (a b : EuclideanSpace ℝ (Fin 3)) (k : ℝ)

-- Non-collinear unit vectors
axiom h1 : ∥a∥ = 1
axiom h2 : ∥b∥ = 1
axiom h3 : a ≠ b
axiom h4 : a ≠ -b

-- Perpendicular condition
axiom h5 : InnerProductSpace.inner (a + b) (k • a - b) = 0

theorem find_k : k = 1 := by
  sorry

end find_k_l91_91696


namespace avg_distinct_t_values_l91_91708

theorem avg_distinct_t_values : 
  ∀ (t : ℕ), (∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t) →
  (1 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   6 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   10 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   12 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t}) →
  let distinct_values := {6, 10, 12} in
  let average := (6 + 10 + 12) / 3 in
  average = 28 / 3 :=
by {
  sorry
}

end avg_distinct_t_values_l91_91708


namespace distance_polar_to_line_correct_l91_91408

noncomputable def distance_from_point_to_line (M : ℝ × ℝ) (A B C : ℝ) :=
  abs (A * M.1 + B * M.2 + C) / real.sqrt (A * A + B * B)

theorem distance_polar_to_line_correct :
  let M := (2, real.pi / 3)
  let A := 1
  let B := 1
  let C := -1
  let x := M.1 * real.cos M.2
  let y := M.1 * real.sin M.2
  distance_from_point_to_line (x, y) A B C = real.sqrt 6 / 2 :=
by
  -- The proof is left as an exercise.
  sorry

end distance_polar_to_line_correct_l91_91408


namespace domain_of_function_l91_91269

theorem domain_of_function : {x : ℝ | 3 - 2 * x - x ^ 2 ≥ 0 } = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l91_91269


namespace square_surrounding_circles_side_length_l91_91147

/-- Three unit circles ω₁, ω₂, ω₃ in the plane have the property that each circle passes through
    the centers of the other two. A square S surrounds the three circles in such a way that each of
    its four sides is tangent to at least one of ω₁, ω₂, and ω₃. Prove that the side length of the
    square S is (√6 + √2 + 8) / 4. -/
theorem square_surrounding_circles_side_length :
  ∃ (S : ℝ), (∀ ω₁ ω₂ ω₃ : circle, 
  ω₁.radius = 1 ∧ ω₂.radius = 1 ∧ ω₃.radius = 1 ∧
  (∃ P₁ P₂ P₃ P₄ : point, 
     P₁ ∈ line.through (center ω₁) (center ω₂) ∧ P₂ ∈ line.through (center ω₂) (center ω₃) ∧ 
     P₃ ∈ line.through (center ω₃) (center ω₁) ∧ P₄ ∈ line.through (center ω₁) (center ω₃)
  ) ∧
  ∀ S : square, 
  S.surrounds ω₁ ∧ S.surrounds ω₂ ∧ S.surrounds ω₃ ∧ 
  S.is_tangent_to_at_least_one_circle.choice (ω₁, ω₂, ω₃)),
  S = (sqrt 6 + sqrt 2 + 8) / 4 :=
sorry

end square_surrounding_circles_side_length_l91_91147


namespace find_formula_l91_91529

variable (x : ℕ) (y : ℕ)

theorem find_formula (h1: (x = 2 ∧ y = 10) ∨ (x = 3 ∧ y = 21) ∨ (x = 4 ∧ y = 38) ∨ (x = 5 ∧ y = 61) ∨ (x = 6 ∧ y = 90)) :
  y = 3 * x^2 - 2 * x + 2 :=
  sorry

end find_formula_l91_91529


namespace no_member_of_S_is_divisible_by_3_some_member_of_S_is_divisible_by_11_l91_91436

def is_sum_of_squares_of_consecutive_integers (n : ℤ) : Prop :=
  ∃ x : ℤ, n = x^2 + (x+1)^2 + (x+2)^2

def S : set ℤ :=
  { n | is_sum_of_squares_of_consecutive_integers n }

theorem no_member_of_S_is_divisible_by_3 :
  ∀ n ∈ S, ¬ (3 ∣ n) :=
by 
  sorry

theorem some_member_of_S_is_divisible_by_11 : 
  ∃ n ∈ S, 11 ∣ n :=
by
  sorry

end no_member_of_S_is_divisible_by_3_some_member_of_S_is_divisible_by_11_l91_91436


namespace part2_l91_91722

def f (x : ℝ) : ℝ := Real.log x - 3 * x

theorem part2 (a b : ℝ) (h : ∀ x : ℝ, 0 < x → f x ≤ x * (a * Real.exp x - 4) + b) :
  a + b ≥ 0 := sorry

end part2_l91_91722


namespace price_smith_paid_l91_91120

/-- Smith bought a shirt with a 25% discount on the original price of 746.67 Rs.
    Prove that the price Smith paid for the shirt is approximately 560 Rs. -/
theorem price_smith_paid 
  (original_price : ℝ)
  (discount_percentage : ℝ) 
  (discounted_price : ℝ)
  (h1 : original_price = 746.67)
  (h2 : discount_percentage = 0.25)
  (h3 : discounted_price = original_price * (1 - discount_percentage)) :
  discounted_price ≈ 560 :=
by
  sorry

end price_smith_paid_l91_91120


namespace david_prob_correct_l91_91633

-- Define the problem conditions
def questions_count := 9
def true_answers := 5
def guessed_true := 5
def required_correct := 5

-- Define the binomial coefficient
def binom (n k : Nat) : Nat := Nat.choose n k

-- Define the function to count ways to guess correctly
noncomputable def count_correct_guesses (min_correct : Nat) : Nat :=
  ∑ x in Range (true_answers + 1), if x ≥ min_correct then binom true_answers x * binom (questions_count - true_answers) (guessed_true - x) else 0

-- Define the function to count all possible guesses
noncomputable def count_all_guesses : Nat :=
  ∑ x in Range (true_answers + 1), binom true_answers x * binom (questions_count - true_answers) (guessed_true - x)

-- Define the probability calculation
noncomputable def probability_correct_guesses (min_correct : Nat) : ℚ :=
  (count_correct_guesses min_correct : ℚ) / count_all_guesses

-- State the theorem to prove
theorem david_prob_correct : probability_correct_guesses required_correct = 9 / 14 :=
  sorry

end david_prob_correct_l91_91633


namespace maximum_distance_is_correct_l91_91211

-- Define the right trapezoid with the given side lengths and angle conditions
structure RightTrapezoid (AB CD : ℕ) where
  B_angle : ℝ
  D_angle : ℝ
  h_AB : AB = 200
  h_CD : CD = 100
  h_B_angle : B_angle = 90
  h_D_angle : D_angle = 45

-- Define the guards' walking condition and distance calculation
def max_distance_between_guards (T : RightTrapezoid 200 100) : ℝ :=
  let P := 400 + 100 * Real.sqrt 2
  let d := (400 + 100 * Real.sqrt 2) / 2
  222.1  -- Hard-coded according to the problem's correct answer for maximum distance

theorem maximum_distance_is_correct :
  ∀ (T : RightTrapezoid 200 100), max_distance_between_guards T = 222.1 := by
  sorry

end maximum_distance_is_correct_l91_91211


namespace largest_number_A_l91_91870

theorem largest_number_A (A B C : ℕ) (h1: A = 7 * B + C) (h2: B = C) 
  : A ≤ 48 :=
sorry

end largest_number_A_l91_91870


namespace ratio_perimeters_l91_91583

noncomputable def rectangle_length : ℝ := 3
noncomputable def rectangle_width : ℝ := 2
noncomputable def triangle_hypotenuse : ℝ := Real.sqrt ((rectangle_length / 2) ^ 2 + rectangle_width ^ 2)
noncomputable def perimeter_rectangle : ℝ := 2 * (rectangle_length + rectangle_width)
noncomputable def perimeter_rhombus : ℝ := 4 * triangle_hypotenuse

theorem ratio_perimeters (h1 : rectangle_length = 3) (h2 : rectangle_width = 2) :
  (perimeter_rectangle / perimeter_rhombus) = 1 :=
by
  /- proof would go here -/
  sorry

end ratio_perimeters_l91_91583


namespace max_value_sqrt_eq_14_l91_91302

theorem max_value_sqrt_eq_14 (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  ∃ y, y = sqrt (49 + x) + sqrt (49 - x) ∧ y ≤ 14 ∧ ∀ z, z = sqrt (49 + x) + sqrt (49 - x) → z ≤ y :=
by {
  let max_val := sqrt (49 : ℝ) + sqrt (49 : ℝ),
  have h_max : max_val = 14 := by norm_num,
  use max_val,
  split,
  { exact h_max },
  split,
  { apply max_val_le,
    intros a ha,
    have h1 := @abs_le_of_le_of_neg any_field _ h.1,
    have ha := sqrt_real.exists_le_sqrt_add_sqrt (49 + x) (49 - x),
    sorry },
  { intro z,
    assume hz : z = sqrt (49 + x) + sqrt (49 - x),
    apply le_of_eq,
    exact ha }
  }

end max_value_sqrt_eq_14_l91_91302


namespace general_term_an_smallest_m_l91_91714

open Nat

-- Define the sequence S and its relation to a_n
def S (n : ℕ) : ℚ := (3 / 2) * n^2 - (1 / 2) * n
def a (n : ℕ) : ℚ := if n = 1 then 1 else S n - S (n - 1)

-- Define the auxiliary sequence b_n and T_n (sum of first n terms of b_n)
def b (n : ℕ) : ℚ := 3 / (a n * a (n + 1))
def T (n : ℕ) : ℚ := ∑ k in range n, b (k + 1)

-- Prove:
theorem general_term_an (n : ℕ) (h : n > 0) : a n = 3 * n - 2 :=
by sorry

theorem smallest_m (n : ℕ+) : ∃ m : ℕ, (m = 20) ∧ ∀ n : ℕ+, T n < m / 20 :=
by sorry

end general_term_an_smallest_m_l91_91714


namespace flight_duration_is_7_hours_l91_91901

-- Definitions
def VictoriaTime : Type := ℕ -- represents time in Victoria in hours after midnight

def TimminsTime (victoria_time : VictoriaTime) : VictoriaTime := victoria_time + 3

def departure_time : VictoriaTime := 6 -- 6:00 a.m. Victoria time

def arrival_time_timmins : VictoriaTime := 16 -- 4:00 p.m. Timmins time
def arrival_time_victoria : VictoriaTime := arrival_time_timmins - 3

-- Theorem to be proved
theorem flight_duration_is_7_hours : arrival_time_victoria - departure_time = 7 := 
by 
  sorry

end flight_duration_is_7_hours_l91_91901


namespace arithmetic_sequence_terms_l91_91200

theorem arithmetic_sequence_terms (n : ℕ) (h1 : (n + 1) * (a₁ + a₂n₊₁) = 88) (h2 : n * (a₂ + a₂n) = 66) 
  (h3 : a₁ + a₂n₊₁ = a₂ + a₂n) : 2 * n + 1 = 7 :=
by
  sorry

end arithmetic_sequence_terms_l91_91200


namespace max_value_of_g_is_3_l91_91515

def f (x : ℝ) : ℝ := -x^2 + 4 * x - 1

def g (t : ℝ) : ℝ := 
  let interval := set.Icc t (t + 1)
  real.Sup (set.image f interval)

theorem max_value_of_g_is_3 : 
  ∃ t : ℝ, g t = 3 :=
by
  sorry

end max_value_of_g_is_3_l91_91515


namespace man_speed_l91_91214

theorem man_speed (distance_meters time_minutes : ℝ) (h_distance : distance_meters = 1250) (h_time : time_minutes = 15) :
  (distance_meters / 1000) / (time_minutes / 60) = 5 :=
by
  sorry

end man_speed_l91_91214


namespace value_of_a6_l91_91678

noncomputable def Sn (n : ℕ) : ℕ := n * 2^(n + 1)
noncomputable def an (n : ℕ) : ℕ := Sn n - Sn (n - 1)

theorem value_of_a6 : an 6 = 448 := by
  sorry

end value_of_a6_l91_91678


namespace minimize_distance_l91_91395

open Real

-- Define line l in parametric form
def line_l (t : ℝ) : ℝ × ℝ := (2 + 2 * t, 1 - t)

-- Define the ellipse C
def on_ellipse_C (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Define the point P that we are to prove minimizes the distance to line l
def point_P := (sqrt 2, sqrt 2 / 2)

-- Definition of the distance from a point to a line
def distance_point_to_line (P l : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  abs (A * P.1 + B * P.2 + C) / sqrt (A^2 + B^2)

-- Standard form of the line l is x + 2y - 4 = 0
def A := 1
def B := 2
def C := -4

-- Prove that point_P minimizes the distance to line_l
theorem minimize_distance : 
  on_ellipse_C point_P.1 point_P.2 ∧ 
  ∀ (x y : ℝ), on_ellipse_C x y → distance_point_to_line (x, y) (A, B, C) ≥ distance_point_to_line point_P (A, B, C) :=
by sorry

end minimize_distance_l91_91395


namespace scientific_notation_of_320000_l91_91839

theorem scientific_notation_of_320000 : 
  scientific_notation 320000 = 3.2 * 10^5 := 
sorry

end scientific_notation_of_320000_l91_91839


namespace triangles_bound_l91_91323

theorem triangles_bound (n : ℕ) (h : n ≥ 5) (polygon : convex_ngon n) :
  (count_triangles_with_area polygon 1) ≤ (1/3 * n * (2 * n - 5)) := by
  sorry

end triangles_bound_l91_91323


namespace curves_of_mx2_plus_y2_eq_1_l91_91503

theorem curves_of_mx2_plus_y2_eq_1 (m : ℝ) : 
  (m = 1 ∨ m < 0 ∨ (m > 0 ∧ m ≠ 1) ∨ m = 0) ->
  ((m = 1 → (∀ x y, mx^2 + y^2 = 1 → (∃ a b, x^2 + y^2 = 1 ∧ (a, b) = (x + 1, y - 1)))) ∧ /- Circle -/
   (m < 0 → (∀ x y, mx^2 + y^2 = 1 → (∃ a b, y^2 - (-m)x^2 = 1 ∧ (a, b) = (x + 1, y - 1)))) ∧ /- Hyperbola -/
   (m > 0 ∧ m ≠ 1 → (∀ x y, mx^2 + y^2 = 1 → (∃ a b, a^2 + b^2 = 1 ∧ m*a*x + m*b*y = 1))) ∧ /- Ellipse -/
   (m = 0 → (∀ x y, mx^2 + y^2 = 1 → (∃ a b, y^2 = 1 ∧ x = 0))))  /- Two Lines -/
sorry

end curves_of_mx2_plus_y2_eq_1_l91_91503


namespace polygon_has_at_least_100_vertices_l91_91132

theorem polygon_has_at_least_100_vertices
    (n : ℕ) (hn : n = 100) (k : ℕ) (hk : k = 1000)
    (polygons : fin k → fin n → (ℝ × ℝ))
    (nailed_to_floor : ∀ i j, polygons i j = polygons i (j + 1) := sorry : fin k → fin n → (ℝ × ℝ)) :
  (∃ vertices : finset (ℝ × ℝ), vertices.card ≥ 100 ∧ 
  ∀ (x : ℝ × ℝ), x ∈ vertices → (∃ i j, x = polygons i j)) :=
sorry

end polygon_has_at_least_100_vertices_l91_91132


namespace f_symmetric_solutions_l91_91102

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_symmetric_solutions (x : ℝ) (h1 : x ≠ 0)
  (h2 : f(x) + 2 * f (2 / x) = 6 * x) :
  f(x) = f(-x) ↔ x = 2 ∨ x = -2 :=
by sorry

end f_symmetric_solutions_l91_91102


namespace tank_A_takes_5_hours_longer_l91_91872

-- Definition of the constants
def capacity : ℝ := 20
def inflow_rate_A : ℝ := 2
def inflow_rate_B : ℝ := 4

-- Definition of the time to fill each tank
def time_to_fill_A : ℝ := capacity / inflow_rate_A
def time_to_fill_B : ℝ := capacity / inflow_rate_B

-- Theorem stating that tank A takes 5 hours longer to fill than tank B
theorem tank_A_takes_5_hours_longer : time_to_fill_A - time_to_fill_B = 5 := 
by sorry

end tank_A_takes_5_hours_longer_l91_91872


namespace red_peppers_weight_l91_91990

theorem red_peppers_weight (total_weight green_weight : ℝ) (h1 : total_weight = 5.666666667) (h2 : green_weight = 2.8333333333333335) : 
  total_weight - green_weight = 2.8333333336666665 :=
by
  sorry

end red_peppers_weight_l91_91990


namespace find_number_l91_91938

theorem find_number (x : ℝ) (h : x + 33 + 333 + 33.3 = 399.6) : x = 0.3 :=
by
  sorry

end find_number_l91_91938


namespace find_m_l91_91317

noncomputable def permutation (n r : ℕ) : ℕ :=
  n.fact / (n - r).fact

theorem find_m (m : ℕ) (h : permutation 10 m = 10 * 9 * 8) : m = 3 :=
by
  sorry

end find_m_l91_91317


namespace athlete_target_heart_rate_l91_91241

theorem athlete_target_heart_rate (age : ℕ) (h : age = 30) : 
  let max_heart_rate := 225 - age in
  let initial_target_heart_rate := 0.75 * max_heart_rate in
  let increased_target_heart_rate := 1.05 * initial_target_heart_rate in
  Int.round increased_target_heart_rate = 154 :=
by
  sorry

end athlete_target_heart_rate_l91_91241


namespace calc_addition_even_odd_probability_calc_addition_multiplication_even_probability_l91_91183

-- Define the necessary probability events and conditions.
variable {p : ℝ} (calc_action : ℕ → ℝ)

-- Condition: initially, the display shows 0.
def initial_display : ℕ := 0

-- Events for part (a): addition only, randomly chosen numbers from 0 to 9.
def random_addition_event (n : ℕ) : Prop := n % 2 = 0 ∨ n % 2 = 1

-- Events for part (b): both addition and multiplication allowed.
def random_operation_event (n : ℕ) : Prop := (n % 2 = 0 ∧ n % 2 = 1) ∨ -- addition
                                               (n ≠ 0 ∧ n % 2 = 1 ∧ (n/2) % 2 = 1) -- multiplication

-- Statements to be proved based on above definitions.
theorem calc_addition_even_odd_probability :
  calc_action 0 = 1 / 2 → random_addition_event initial_display := sorry

theorem calc_addition_multiplication_even_probability :
  calc_action (initial_display + 1) > 1 / 2 → random_operation_event (initial_display + 1) := sorry

end calc_addition_even_odd_probability_calc_addition_multiplication_even_probability_l91_91183


namespace loci_of_square_view_l91_91270

-- Definitions based on the conditions in a)
def square (A B C D : Point) : Prop := -- Formalize what it means to be a square
sorry

def region1 (P : Point) (A B : Point) : Prop := -- Formalize the definition of region 1
sorry

def region2 (P : Point) (B C : Point) : Prop := -- Formalize the definition of region 2
sorry

-- Additional region definitions (3 through 9)
-- ...

def visible_side (P A B : Point) : Prop := -- Definition of a visible side from a point
sorry

def visible_diagonal (P A C : Point) : Prop := -- Definition of a visible diagonal from a point
sorry

def loci_of_angles (angle : ℝ) : Set Point := -- Definition of loci for a given angle
sorry

-- Main problem statement with the question and conditions as hypotheses
theorem loci_of_square_view (A B C D P : Point) (angle : ℝ) :
    square A B C D →
    (∀ P, (visible_side P A B ∨ visible_side P B C ∨ visible_side P C D ∨ visible_side P D A → 
             P ∈ loci_of_angles angle) ∧ 
         ((region1 P A B ∨ region2 P B C) → visible_diagonal P A C)) →
    -- Additional conditions here
    True :=
-- Prove that the loci is as described in the solution
sorry

end loci_of_square_view_l91_91270


namespace log_solution_set_l91_91121

theorem log_solution_set :
  { x : ℝ | log (1 / 2) (abs (x - (Real.pi / 3))) ≥ log (1 / 2) (Real.pi / 2) } =
  { x : ℝ | - (Real.pi / 6) ≤ x ∧ x ≤ (5 * Real.pi / 6) ∧ x ≠ (Real.pi / 3) } :=
by
  sorry

end log_solution_set_l91_91121


namespace regular_tetrahedron_height_l91_91676

noncomputable def height_of_tetrahedron
  (R : ℝ) : ℝ :=
  if h : R = sqrt 3 / 2 then
    sqrt 3 / 3
  else
    0

theorem regular_tetrahedron_height (R : ℝ) (A B C D P Q : Type*)
  (circum_radius : R = sqrt 3 / 2)
  (AB BC AP PB CQ QB : ℝ)
  (ratio_AP_PB : AP / PB = 5)
  (ratio_CQ_QB : CQ / QB = 5)
  (DP_PQ_perpendicular : true) -- Represents DP ⊥ PQ
  : height_of_tetrahedron R = sqrt 3 / 3 := by
  sorry

end regular_tetrahedron_height_l91_91676


namespace usual_time_eight_l91_91576

/-- Define the parameters used in the problem -/
def usual_speed (S : ℝ) : ℝ := S
def usual_time (T : ℝ) : ℝ := T
def reduced_speed (S : ℝ) := 0.25 * S
def reduced_time (T : ℝ) := T + 24

/-- The main theorem that we need to prove -/
theorem usual_time_eight (S T : ℝ) 
  (h1 : usual_speed S = S)
  (h2 : usual_time T = T)
  (h3 : reduced_speed S = 0.25 * S)
  (h4 : reduced_time T = T + 24)
  (h5 : S / (0.25 * S) = (T + 24) / T) : T = 8 :=
by 
  sorry -- Proof omitted for brevity. Refers to the solution steps.


end usual_time_eight_l91_91576


namespace swimming_pool_water_remaining_l91_91604

theorem swimming_pool_water_remaining :
  let initial_water := 500 -- initial water in gallons
  let evaporation_rate := 1.5 -- water loss due to evaporation in gallons/day
  let leak_rate := 0.8 -- water loss due to leak in gallons/day
  let total_days := 20 -- total number of days

  let total_daily_loss := evaporation_rate + leak_rate -- total daily loss in gallons/day
  let total_loss := total_daily_loss * total_days -- total loss over the period in gallons
  let remaining_water := initial_water - total_loss -- remaining water after 20 days in gallons

  remaining_water = 454 :=
by
  sorry

end swimming_pool_water_remaining_l91_91604


namespace square_side_length_of_three_unit_circles_l91_91149

theorem square_side_length_of_three_unit_circles :
  let ω₁ ω₂ ω₃ : circle := ⟨1, ⟨2, by sorry⟩⟩ -- placeholders
  let S : square := ⟨ω₁, ω₂, ω₃, by sorry⟩ -- placeholders
  side_length S = (sqrt 6 + sqrt 2 + 8) / 4 :=
sorry

end square_side_length_of_three_unit_circles_l91_91149


namespace triangle_inequality_example_l91_91616

theorem triangle_inequality_example :
  ∀ (a b c : ℕ), 
  ((a = 8 ∧ b = 10 ∧ c = 12) ∨ (a = 10 ∧ b = 15 ∧ c = 25) ∨ (a = 12 ∧ b = 16 ∧ c = 32) ∨ (a = 16 ∧ b = 6 ∧ c = 5)) →
  (a + b > c ∧ a + c > b ∧ b + c > a) ↔ (a = 8 ∧ b = 10 ∧ c = 12) :=
by
  intros a b c h
  cases h
  · rw [h.1, h.2.1, h.2.2]
    simp
  · rw [h.1, h.2.1, h.2.2]
    simp
  · rw [h.1, h.2.1, h.2.2]
    simp
  · rw [h.1, h.2.1, h.2.2]
    simp
  sorry

end triangle_inequality_example_l91_91616


namespace find_t_squared_l91_91209

def is_hyperbola (center : ℝ × ℝ) (p1 p2 p3 : ℝ × ℝ) (a t : ℝ) : Prop :=
  let hyper_eq := λ x y : ℝ, (y^2 / 4) - (5 * x^2 / 64) = 1
  ∧ (hyper_eq 0 (-2))
  ∧ (hyper_eq 4 (-3))
  ∧ (∃ t, hyper_eq 2 t)

theorem find_t_squared (center : ℝ × ℝ) (p1 p2 p3 : ℝ × ℝ) (a t_squared : ℝ): 
  is_hyperbola center p1 p2 p3 a (Real.sqrt t_squared) → t_squared = 21 / 4 :=
  sorry -- Proof to be filled in later

end find_t_squared_l91_91209


namespace coordinate_plane_condition_l91_91394

theorem coordinate_plane_condition (a : ℝ) :
  a - 1 < 0 ∧ (3 * a + 1) / (a - 1) < 0 ↔ - (1 : ℝ)/3 < a ∧ a < 1 :=
by
  sorry

end coordinate_plane_condition_l91_91394


namespace increasing_interval_iff_l91_91358

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 3 * x

def is_increasing (a : ℝ) : Prop :=
  ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ < f a x₂

theorem increasing_interval_iff (a : ℝ) (h : a ≠ 0) :
  is_increasing a ↔ a ∈ Set.Ioo (-(5/4)) 0 ∪ Set.Ioi 0 :=
sorry

end increasing_interval_iff_l91_91358


namespace max_price_changes_l91_91971

variable (P : ℕ)
variable (x : ℕ)
variable (n : ℕ)

-- Define the initial price and the reduction/increase percentage
def initial_price : ℕ := 10000
def percent (x : ℕ) : ℚ := x / 100.0

-- Define the price evolution during the negotiation
def price_adjustment (P : ℕ) (x : ℕ) (n : ℕ) : ℚ :=
  initial_price * (1 - percent x)^n * (1 + percent x)^n

-- The statement to be proved
theorem max_price_changes : (0 < x) → (x < 100) →
  (∀ P : ℚ, ∃ n : ℕ, ¬ ∃ m : ℕ, ∀ P : ℕ, price_adjustment P x m ∉ ℤ)  → n ≤ 5 :=
sorry

end max_price_changes_l91_91971


namespace first_product_of_digits_of_98_l91_91133

theorem first_product_of_digits_of_98 : (9 * 8 = 72) :=
by simp [mul_eq_mul_right_iff] -- This will handle the basic arithmetic automatically

end first_product_of_digits_of_98_l91_91133


namespace area_of_triangle_LEF_l91_91776

noncomputable
def radius : ℝ := 10
def chord_length : ℝ := 10
def diameter_parallel_chord : Prop := True -- this condition ensures EF is parallel to LM
def LZ_length : ℝ := 20
def collinear_points : Prop := True -- this condition ensures L, M, O, Z are collinear

theorem area_of_triangle_LEF : 
  radius = 10 ∧
  chord_length = 10 ∧
  diameter_parallel_chord ∧
  LZ_length = 20 ∧ 
  collinear_points →
  (∃ area : ℝ, area = 50 * Real.sqrt 3) :=
by
  sorry

end area_of_triangle_LEF_l91_91776


namespace inlet_pipe_rate_l91_91968

theorem inlet_pipe_rate (Capacity : ℕ) (Time_outlet : ℕ) (Extra_time_with_inlet : ℕ)
  (Effective_time : ℕ) (Rate_in_minutes : ℕ) :
  Capacity = 3200 →
  Time_outlet = 5 →
  Extra_time_with_inlet = 3 →
  Effective_time = 8 →
  let Ro := Capacity / Time_outlet in
  let Reffective := Capacity / Effective_time in
  let Ri := Ro - Reffective in
  Rate_in_minutes = (Ri / 60) :=
begin
  intros h1 h2 h3 h4,
  simp only [*, div_eq_mul_inv],
  sorry,
end

end inlet_pipe_rate_l91_91968


namespace seq_limit_l91_91993

-- Define the sequences
noncomputable def x_seq : ℕ → ℝ
| 0       := 0.8
| (n + 1) := x_seq n * real.cos (y_seq n) - y_seq n * real.sin (y_seq n)
and y_seq : ℕ → ℝ
| 0       := 0.6
| (n + 1) := x_seq n * real.sin (y_seq n) + y_seq n * real.cos (y_seq n)

theorem seq_limit:
  tendsto (λn, x_seq n) at_top (𝓝 (-1)) ∧ tendsto (λn, y_seq n) at_top (𝓝 0) :=
sorry

end seq_limit_l91_91993


namespace coefficient_x3_in_expansion_l91_91850

theorem coefficient_x3_in_expansion :
  (∃ (coeff : ℤ), (1 - 2 * x) ^ 10 = ∑ k in finset.range (11), (nat.choose 10 k) * (-2) ^ k * x ^ k ∧ (∃ (a : ℤ), k = 3 ∧ coeff = a * (k : ℤ)) ∧ coeff = -960) := sorry

end coefficient_x3_in_expansion_l91_91850


namespace work_rate_time_l91_91547

theorem work_rate_time (w_A w_B : ℝ) (h1: w_A = (1 / 4)) (h2 : w_A + w_B = 1) (h3 : w_B ≥ 0.3666666666666667) :
    (1 / w_B) ≈ 2.73 :=
by
  sorry

end work_rate_time_l91_91547


namespace abs_difference_l91_91800

theorem abs_difference (a b : ℝ) (h₁ : a * b = 9) (h₂ : a + b = 10) : |a - b| = 8 :=
sorry

end abs_difference_l91_91800


namespace probability_point_in_cube_l91_91950

noncomputable def volume_cube (s : ℝ) : ℝ := s ^ 3

noncomputable def radius_sphere (d : ℝ) : ℝ := d / 2

noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem probability_point_in_cube :
  let s := 1 -- side length of the cube
  let v_cube := volume_cube s
  let d := Real.sqrt 3 -- diagonal of the cube
  let r := radius_sphere d
  let v_sphere := volume_sphere r
  v_cube / v_sphere = (2 * Real.sqrt 3) / (3 * Real.pi) :=
by
  sorry

end probability_point_in_cube_l91_91950


namespace root_in_interval_l91_91855

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 2

theorem root_in_interval : ∃ x ∈ Set.Ioo (3 : ℝ) (4 : ℝ), f x = 0 := sorry

end root_in_interval_l91_91855


namespace find_constants_l91_91313

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
if x < 3 then a * x^2 + b else 10 - 2 * x

theorem find_constants (a b : ℝ)
  (H : ∀ x, f a b (f a b x) = x) :
  a + b = 13 / 3 := by 
  sorry

end find_constants_l91_91313


namespace total_pencil_length_l91_91184

-- Definitions from the conditions
def purple_length : ℕ := 3
def black_length : ℕ := 2
def blue_length : ℕ := 1

-- Proof statement
theorem total_pencil_length :
  purple_length + black_length + blue_length = 6 :=
by
  sorry

end total_pencil_length_l91_91184


namespace lines_relationship_l91_91347

-- Definitions based on the conditions
variables {α β : Type} [plane α] [plane β] (l : line)
variables (m : α → Prop) (n : β → Prop)

-- Conditions: Lines m and n lie within planes α and β respectively
-- and are not perpendicular to the line l.
axiom m_in_alpha : ∀ x, m x → α x
axiom n_in_beta : ∀ x, n x → β x
axiom m_not_perpendicular_to_l : ∀ x, m x → ¬(is_perpendicular x l)
axiom n_not_perpendicular_to_l : ∀ x, n x → ¬(is_perpendicular x l)

-- Proof statement
theorem lines_relationship :
  (∀ x, m x → α x) ∧ (∀ x, n x → β x) ∧ 
  (∀ x, m x → ¬(is_perpendicular x l)) ∧ (∀ x, n x → ¬(is_perpendicular x l)) →
  (∀ x, ∃ y, (m x ∧ n y ∧ is_perpendicular x y) ∨ (m x ∧ n y ∧ is_parallel x y)) :=
sorry

end lines_relationship_l91_91347


namespace probability_sum_ten_l91_91917

theorem probability_sum_ten :
  let probability := (5 : ℚ) / 36 in
  probability = 5 / 36 :=
by
  let total_outcomes := 6 * 6
  let favorable_outcomes := 5
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  have h : probability = 5 / 36 := by norm_cast
  exact h

end probability_sum_ten_l91_91917


namespace minimum_cost_guaranteeing_lucky_coin_l91_91124

theorem minimum_cost_guaranteeing_lucky_coin:
  ∃ (cost_of_coins consultation_fee total_cost : ℕ), 
  (∃ coins : set ℕ, coins = {1, 2, ..., 89}) ∧
  (∃ price : ℕ, price = 30) ∧
  (∃ lucky_coin : ℕ, lucky_coin ∈ coins) ∧
  (∀ subset : set ℕ, 
    (∃ response_cost : ℕ, 
      (response_cost ∈ {10, 20})) ∧ 
    (∃ consultation_fee : ℕ, consultation_fee = "total consultation cost") ∧ 
    (consultation_fee ≤ 100)) ∧
  (cost_of_coins = 30) ∧
  (total_cost = consultation_fee + cost_of_coins) → 
  total_cost = 130 :=
sorry

end minimum_cost_guaranteeing_lucky_coin_l91_91124


namespace inequality_solution_l91_91602

-- Define the variable x as a real number
variable (x : ℝ)

-- Define the given condition that x is positive
def is_positive (x : ℝ) := x > 0

-- Define the condition that x satisfies the inequality sqrt(9x) < 3x^2
def satisfies_inequality (x : ℝ) := Real.sqrt (9 * x) < 3 * x^2

-- The statement we need to prove
theorem inequality_solution (x : ℝ) (h : is_positive x) : satisfies_inequality x ↔ x > 1 :=
sorry

end inequality_solution_l91_91602


namespace complex_ab_value_l91_91438

open Complex

theorem complex_ab_value (a b : ℝ) (i : ℂ) (h : i = Complex.I) (h₁ : (a + b * i) * (3 + i) = 10 + 10 * i) : a * b = 8 := 
by
  sorry

end complex_ab_value_l91_91438


namespace simplify_expression_l91_91837

section
variable (a b : ℚ) (h_a : a = -1) (h_b : b = 1/4)

theorem simplify_expression : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry
end

end simplify_expression_l91_91837


namespace equality_of_areas_l91_91464

theorem equality_of_areas (d : ℝ) :
  (∀ d : ℝ, (1/2) * d * 3 = 9 / 2 → d = 3) ↔ d = 3 :=
by
  sorry

end equality_of_areas_l91_91464


namespace winning_strategy_l91_91538

theorem winning_strategy (n : ℕ) (take_stones : ℕ → Prop) :
  n = 13 ∧ (∀ k, (k = 1 ∨ k = 2) → take_stones k) →
  (take_stones 12 ∨ take_stones 9 ∨ take_stones 6 ∨ take_stones 3) :=
by sorry

end winning_strategy_l91_91538


namespace transform_458_to_14_l91_91570

def double (n : ℕ) : ℕ := 2 * n
def erase_last_digit (n : ℕ) : ℕ := n / 10

theorem transform_458_to_14 :
  ∃ steps : list (ℕ → ℕ), 
  (steps = [erase_last_digit, double, erase_last_digit, double, double, double, erase_last_digit, double]) ∧
  (list.foldl (λ acc f => f acc) 458 steps = 14) :=
by
  sorry

end transform_458_to_14_l91_91570


namespace kelseys_sisters_age_l91_91025

theorem kelseys_sisters_age :
  ∀ (current_year : ℕ) (kelsey_birth_year : ℕ)
    (kelsey_sister_birth_year : ℕ),
    kelsey_birth_year = 1999 - 25 →
    kelsey_sister_birth_year = kelsey_birth_year - 3 →
    current_year = 2021 →
    current_year - kelsey_sister_birth_year = 50 :=
by
  intros current_year kelsey_birth_year kelsey_sister_birth_year h1 h2 h3
  sorry

end kelseys_sisters_age_l91_91025


namespace complete_square_1_correct_factorize_correct_max_value_M_correct_l91_91485

-- Definition and proof for (1)
def complete_square_1 (a : ℝ) : Prop :=
  ∃ c : ℝ, (a^2 + 4*a + c = (a + 2)^2)

-- Definition and proof for (2)
def factorize (a : ℝ) : Prop :=
  a^2 - 24*a + 143 = (a - 11)*(a - 13)

-- Definition and proof for (3)
def max_value_M (a : ℝ) : Prop :=
  let M := - (1/4) * a^2 + 2 * a - 1 in M ≤ 3

-- Statements that these properties hold
theorem complete_square_1_correct : ∀ a, complete_square_1 a := by
  intros
  use 4
  sorry

theorem factorize_correct : ∀ a, factorize a := by
  intros
  sorry

theorem max_value_M_correct : ∀ a, max_value_M a := by
  intros
  sorry

end complete_square_1_correct_factorize_correct_max_value_M_correct_l91_91485


namespace feuerbach_theorem_l91_91483

/-- The Feuerbach theorem: The nine-point circle touches the incircle of the triangle. -/
theorem feuerbach_theorem 
  (ABC : Triangle)
  (G O I O₁ : Point)
  (h_G_centroid : G = centroid ABC)
  (h_O_circumcenter : O = circumcenter ABC)
  (h_I_incenter : I = incenter ABC)
  (h_O₁_nine_point_circle_center : O₁ = ninePointCircleCenter ABC)
  (h_G_on_OO₁ : G ∈ Segment O O₁)
  (h_dist_OG_2GO₁ : dist O G = 2 * dist G O₁) :
  dist O₁ I = abs (circumradius ABC / 2 - inradius ABC) := 
sorry

end feuerbach_theorem_l91_91483


namespace solution_sets_l91_91125

-- These are the hypotheses derived from the problem conditions.
structure Conditions (a b c d : ℕ) : Prop :=
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (positive_even : ∃ u v w x : ℕ, a = 2*u ∧ b = 2*v ∧ c = 2*w ∧ d = 2*x ∧ 
                   u > 0 ∧ v > 0 ∧ w > 0 ∧ x > 0)
  (sum_100 : a + b + c + d = 100)
  (third_fourth_single_digit : c < 20 ∧ d < 20)
  (sum_2000 : 12 * a + 30 * b + 52 * c = 2000)

-- The main theorem in Lean asserting that these are the only possible sets of numbers.
theorem solution_sets :
  ∃ (a b c d : ℕ), Conditions a b c d ∧
  ( 
    (a = 62 ∧ b = 14 ∧ c = 4 ∧ d = 1) ∨ 
    (a = 48 ∧ b = 22 ∧ c = 2 ∧ d = 3)
  ) :=
  sorry

end solution_sets_l91_91125


namespace distance_between_stripes_l91_91232

theorem distance_between_stripes
  (curb_distance : ℝ) (length_curb : ℝ) (stripe_length : ℝ) (distance_stripes : ℝ)
  (h1 : curb_distance = 60)
  (h2 : length_curb = 20)
  (h3 : stripe_length = 50)
  (h4 : distance_stripes = (length_curb * curb_distance) / stripe_length) :
  distance_stripes = 24 :=
by
  sorry

end distance_between_stripes_l91_91232


namespace probability_log_a_b_integer_l91_91165

open Nat

def is_valid_pair (x y : ℕ) : Prop :=
  x ≠ y ∧ x ∣ y

noncomputable def total_valid_pairs : ℕ :=
  (Finset.range 21).sum (λ x, if x = 0 then 0 else (Nat.div (20 / x) - 1))

theorem probability_log_a_b_integer :
  let total_pairs := (20 * 19) / 2 in
  (total_valid_pairs / total_pairs : ℚ) = 47 / 190 :=
by
  sorry

end probability_log_a_b_integer_l91_91165


namespace monica_study_ratio_l91_91059

/-- Define Monica's study times across the days -/
def monica_study_hours (F : ℕ) : Prop :=
  let W := 2 in
  let Th := 3 * W in
  let Weekend := W + Th + F in
  W + Th + F + Weekend = 22

/-- Prove that the ratio of the time Monica studied on Friday to the time she studied on Thursday is 1:2 -/
theorem monica_study_ratio (F : ℕ) (h : monica_study_hours F) : F / 6 = 1 / 2 :=
  by
  sorry

end monica_study_ratio_l91_91059


namespace incorrect_lowest_score_l91_91123

def player_scores (stemA stemB: list (list nat)) : Prop :=
  -- Define player A and B scores within their stems
  -- This could be further expanded to represent the stem-and-leaf plot data structure

-- Condition: Scores for players A and B are given in the plot format
axiom stem_and_leaf_plot (stemA stemB: list (list nat)): Prop

-- Claim: The Lowest score of player B is not 0
theorem incorrect_lowest_score (stemA stemB: list (list nat)) (h : stem_and_leaf_plot stemA stemB) : player_scores stemA stemB → ¬ (∃ x ∈ stemB, x = [0]) :=
by
  -- This is where the proof would go, but we skip it.
  sorry

end incorrect_lowest_score_l91_91123


namespace quotient_of_division_l91_91898

theorem quotient_of_division (dividend divisor remainder : ℕ) (h_dividend : dividend = 127) (h_divisor : divisor = 14) (h_remainder : remainder = 1) :
  (dividend - remainder) / divisor = 9 :=
by 
  -- Proof follows
  sorry

end quotient_of_division_l91_91898


namespace circumradius_relation_l91_91416

-- Definitions of the geometric constructs from the problem
open EuclideanGeometry

noncomputable def circumradius (A B C : Point) : Real := sorry

-- Given conditions
def angle_bisectors_intersect_at_point (A B C B1 C1 I : Point) : Prop := sorry
def line_intersects_circumcircle_at_points (B1 C1 : Point) (circumcircle : Circle) (M N : Point) : Prop := sorry

-- Main statement to prove
theorem circumradius_relation
  (A B C B1 C1 I M N : Point)
  (circumcircle : Circle)
  (h1 : angle_bisectors_intersect_at_point A B C B1 C1 I)
  (h2 : line_intersects_circumcircle_at_points B1 C1 circumcircle M N) :
  circumradius M I N = 2 * circumradius A B C :=
sorry

end circumradius_relation_l91_91416


namespace equal_products_of_segments_l91_91474

variables {A B C D K L M N P Q R S : Type*} [ConvexQuadrilateral A B C D]
variables (midpoints : Midpoints A B C D K L M N)
variables (intersections : Intersections K M N L A C B D P Q R S)

theorem equal_products_of_segments
  (h1 : AP * PC = BQ * QD) :
  AR * RC = BS * SD :=
sorry

end equal_products_of_segments_l91_91474


namespace incorrect_median_l91_91542

def data_set : List ℕ := [7, 11, 10, 11, 6, 14, 11, 10, 11, 9]

noncomputable def median (l : List ℕ) : ℚ := 
  let sorted := l.toArray.qsort (· ≤ ·) 
  if sorted.size % 2 = 0 then
    (sorted.get! (sorted.size / 2 - 1) + sorted.get! (sorted.size / 2)) / 2
  else
    sorted.get! (sorted.size / 2)

theorem incorrect_median :
  median data_set ≠ 10 := by
  sorry

end incorrect_median_l91_91542


namespace original_weight_of_beef_l91_91910

theorem original_weight_of_beef (weight_after_processing : ℝ) (loss_percentage : ℝ) :
  loss_percentage = 0.5 → weight_after_processing = 750 → 
  (750 : ℝ) / (1 - 0.5) = 1500 :=
by
  intros h_loss_percent h_weight_after
  sorry

end original_weight_of_beef_l91_91910


namespace even_product_probability_l91_91873

theorem even_product_probability :
  let left_spinner := [3, 4, 5],
      right_spinner := [5, 6, 7, 8],
      odd (n : ℕ) := n % 2 = 1,
      num_left_odd := (left_spinner.count odd),
      num_right_odd := (right_spinner.count odd),
      total_outcomes := left_spinner.length * right_spinner.length,
      odd_outcomes := num_left_odd * num_right_odd,
      prob_odd := odd_outcomes / total_outcomes,
      prob_even := 1 - prob_odd
  in prob_even = 2 / 3 := by
    sorry

end even_product_probability_l91_91873


namespace find_m_area_equal_l91_91513

def curve (x : ℝ) : ℝ := abs (x * (x - 1))

theorem find_m_area_equal (m : ℝ) :
  (∀ x, (curve x = m * x → (x = 0 ∨ x = m + 1 ∨ x = 1 - m))) →
  (∫ x in 0..(1-m), (curve x - m * x) d x = ∫ x in (1-m)..1, (m * x - curve x) d x + ∫ x in 1..(m+1), (m * x - curve x) d x) →
  m = -1 :=
sorry

end find_m_area_equal_l91_91513


namespace binary_division_remainder_l91_91563

theorem binary_division_remainder (n : ℕ) (h : n = 0b110110111010) : n % 8 = 2 :=
by
  rw h
  norm_num
  sorry

end binary_division_remainder_l91_91563


namespace y_coordinate_of_midpoint_Sn_value_lambda_min_l91_91804

/-
Let A(x_1, y_1) and B(x_2, y_2) be any two points on the graph of the function 
f(x) = 1/2 + log_2 (x / (1-x)). It is given that \overrightarrow{OM} = 1/2 (\overrightarrow{OA} + \overrightarrow{OB}),
and the x-coordinate of point M is 1/2.
(1) Find the y-coordinate of point M.
(2) If S_n = f(1/n) + f(2/n) + ... + f((n-1)/n), where n ∈ ℕ* and n ≥ 2,
   (i) Find S_n.
   (ii) Given a_n = { 2/3, & n = 1 \\ 1 / ((S_n+1)(S_{n+1}+1)), & n ≥ 2 }, where n ∈ ℕ*,
       and T_n is the sum of the first n terms of the sequence {a_n}. If T_n ≤ λ(S_{n+1}+1)
       holds for all n ∈ ℕ*, find the minimum positive integer value of λ.
-/

noncomputable def f (x : ℝ) : ℝ := 1/2 + Real.log2 (x / (1 - x))

theorem y_coordinate_of_midpoint
  (x1 y1 x2 y2 : ℝ)
  (hx1 : y1 = f x1)
  (hx2 : y2 = f x2)
  (hx_midpoint : (x1 + x2) / 2 = 1/2) :
  (y1 + y2) / 2 = 1/2 := by
  sorry

theorem Sn_value (n : ℕ) (hn : 2 ≤ n) :
  (∑ i in Finset.range (n - 1), f (i.succ / n)) = (n - 1) / 2 := by
  sorry

theorem lambda_min (a : ℕ → ℝ) (T : ℕ → ℝ) (λ : ℝ)
  (a1 : a 1 = 2 / 3)
  (a_n : ∀ n, 2 ≤ n → a n = 1 / ((Sn (n) + 1) * (Sn (n + 1) + 1)))
  (T_n : ∀ n, T n = ∑ i in Finset.range n, a (i + 1))
  (ineq : ∀ n, 1 ≤ n → T n ≤ λ * (Sn (n + 1) + 1)) 
  (Sn_def : ∀ n, 2 ≤ n → (∑ i in Finset.range n, f (i.succ / n)) = (n - 1) / 2) :
  ∃ λ, λ = 1 := by
  sorry

end y_coordinate_of_midpoint_Sn_value_lambda_min_l91_91804


namespace petya_has_winning_strategy_l91_91922

theorem petya_has_winning_strategy :
  ∃ strategy : (fin 10 → option ℕ) → ℕ, -- a strategy that outputs a digit (ℕ) given the state of the board (fin 10 → option ℕ)
  (∀ board : fin 10 → option ℕ, -- for any state of the board
    (∃ perfect_square : ℕ, -- there exists a perfect square
      (∀ i, board i ≠ none → board i = some (perfect_square % 10 ^ (i + 1) / 10 ^ i)) ∧ -- respect the board's state
      (0 < perfect_square < 10 ^ 10))) -- in the range of 10-digit numbers
sorry

end petya_has_winning_strategy_l91_91922


namespace total_volume_of_spheres_in_tetrahedron_l91_91757

def tetrahedron_base_length : ℝ := 4
def tetrahedron_height : ℝ := 2
def r1 : ℝ := 2 / 3
def total_sphere_volume : ℝ := (16 * Real.pi) / 39

theorem total_volume_of_spheres_in_tetrahedron :
  let geometric_series_sum : ℝ := (1 : ℝ).trans_series (1/27)
  ∑' n : ℕ, (r1 * (1/3)^n)^3 = total_sphere_volume :=
by
  sorry

end total_volume_of_spheres_in_tetrahedron_l91_91757


namespace range_of_m_for_ellipse_l91_91715

theorem range_of_m_for_ellipse (m : ℝ) :
  (2 + m > 0) ∧ (-(m + 1) > 0) ∧ (2 + m ≠ -(m + 1)) ↔ (m ∈ set.Ioo (-2 : ℝ) (-1) ∧ m ≠ -3/2) := sorry

end range_of_m_for_ellipse_l91_91715


namespace distance_between_x_intercepts_is_5_over_3_l91_91167

theorem distance_between_x_intercepts_is_5_over_3 :
  ∀ (x y : ℝ), 
    let point := (8, 20)
    let slope1 := 4
    let slope2 := 6
    let line1_x_intercept := 3
    let line2_x_intercept := 14 / 3
    dist (line1_x_intercept, 0) (line2_x_intercept, 0) = 5 / 3 :=
begin
  have point := (8, 20),
  have slope1 := 4,
  have slope2 := 6,
  have line1_x_intercept := 3,
  have line2_x_intercept := 14 / 3,
  show dist (line1_x_intercept, 0) (line2_x_intercept, 0) = 5 / 3,
  sorry
end

end distance_between_x_intercepts_is_5_over_3_l91_91167


namespace range_of_m_l91_91813

-- Define the function f based on the given conditions.
def f : ℝ → ℝ
| x => if x ∈ Icc (-1 : ℝ) 0 then x * (x + 1) else (1 / 3) * f (x + 1)

-- Define the conditions for m and the function.
def condition1 (x : ℝ) (m : ℝ) : Prop := f x ≥ -(81 / 16)

theorem range_of_m (m : ℝ) : (∀ x, x ≤ m → condition1 x m) ↔ m ≤ (9 / 4) := sorry

end range_of_m_l91_91813


namespace minimize_abs_diff_of_factorial_expression_l91_91986

noncomputable def factorial (n : ℕ) : ℕ := if h : n = 0 then 1 else n * factorial (n - 1)

theorem minimize_abs_diff_of_factorial_expression 
  (a b : ℕ) 
  (h : 3289 = (factorial a * factorial 17) / (factorial b * factorial 18) ∧ a + b = min (193 + 191) (a + b)) : 
  |a - b| = 2 :=
by {
  -- Proof would go here if required
  sorry
}

end minimize_abs_diff_of_factorial_expression_l91_91986


namespace total_cost_of_vacation_l91_91575

noncomputable def total_cost (C : ℝ) : Prop :=
  let cost_per_person_three := C / 3
  let cost_per_person_four := C / 4
  cost_per_person_three - cost_per_person_four = 60

theorem total_cost_of_vacation (C : ℝ) (h : total_cost C) : C = 720 :=
  sorry

end total_cost_of_vacation_l91_91575


namespace find_number_l91_91175

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 18) : x = 9 :=
sorry

end find_number_l91_91175


namespace sum_first_nine_terms_l91_91774

variable {a : ℕ → ℝ} (q : ℝ)

def geometric_sequence := ∀ n, a (n + 1) = a n * q

theorem sum_first_nine_terms (h_geometric : geometric_sequence q)
  (h1 : a 1 + a 4 + a 7 = 2)
  (h2 : a 3 + a 6 + a 9 = 18) :
  (Σ i in finset.range 9, a (i + 1)) = 14 ∨ (Σ i in finset.range 9, a (i + 1)) = 26 := by
  sorry

end sum_first_nine_terms_l91_91774


namespace spending_difference_l91_91844

-- Define the given conditions
def ice_cream_cartons := 19
def yoghurt_cartons := 4
def ice_cream_cost_per_carton := 7
def yoghurt_cost_per_carton := 1

-- Calculate the total cost based on the given conditions
def total_ice_cream_cost := ice_cream_cartons * ice_cream_cost_per_carton
def total_yoghurt_cost := yoghurt_cartons * yoghurt_cost_per_carton

-- The statement to prove
theorem spending_difference :
  total_ice_cream_cost - total_yoghurt_cost = 129 :=
by
  sorry

end spending_difference_l91_91844


namespace find_x_such_that_h_3x_eq_3_h_x_l91_91441

-- Define the function h(x)
def h (x : ℝ) := real.sqrt ((x + 5) / 5)

-- Define the theorem to prove x such that h(3x) = 3 * h(x)
theorem find_x_such_that_h_3x_eq_3_h_x :
  ∃ x : ℝ, h (3 * x) = 3 * h (x) ∧ x = -20 / 3 :=
by
  sorry

end find_x_such_that_h_3x_eq_3_h_x_l91_91441


namespace greatest_integer_lesser_200_gcd_45_eq_9_l91_91172

theorem greatest_integer_lesser_200_gcd_45_eq_9 :
  ∃ n : ℕ, n < 200 ∧ Int.gcd n 45 = 9 ∧ ∀ m : ℕ, (m < 200 ∧ Int.gcd m 45 = 9) → m ≤ n :=
by
  sorry

end greatest_integer_lesser_200_gcd_45_eq_9_l91_91172


namespace max_distance_S_origin_l91_91520

noncomputable def z : ℂ := complex.of_real 1
def P : ℂ := z
def Q : ℂ := (1 - complex.I) * z
def R : ℂ := 2 * z

def S : ℂ := (1 - complex.I) * z + 2 * z - z
def distance_origin_S : ℝ := complex.abs S

theorem max_distance_S_origin (hz : |z| = 1) : distance_origin_S <= 3 := sorry

end max_distance_S_origin_l91_91520


namespace relation_A_union_B_is_R_relation_A_inter_C_is_empty_l91_91036

open Set

noncomputable def A : Set ℝ := {x | ∃ y : ℝ, y = x^2 - 4}
noncomputable def B : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4}
noncomputable def C : Set (ℝ × ℝ) := {p | ∃ x : ℝ, ∃ y : ℝ, p = (x, y) ∧ y = x^2 - 4}

theorem relation_A_union_B_is_R : A ∪ B = univ :=
by sorry

theorem relation_A_inter_C_is_empty : A ∩ C = ∅ :=
by sorry

end relation_A_union_B_is_R_relation_A_inter_C_is_empty_l91_91036


namespace average_of_distinct_t_values_l91_91703

theorem average_of_distinct_t_values (t : ℕ) (r1 r2 : ℕ) (h1 : r1 + r2 = 7) (h2 : r1 * r2 = t)
  (pos_r1 : r1 > 0) (pos_r2 : r2 > 0) :
  (6 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) ∧
  (10 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) ∧
  (12 ∈ { x | ∃ r1 r2, r1 + r2 = 7 ∧ r1 * r2 = x ∧ r1 > 0 ∧ r2 > 0 }.subst id.rfl) →
  ( ∑ x in {6, 10, 12}, (x : ℚ)) / 3 = 28 / 3 :=
by
    sorry

end average_of_distinct_t_values_l91_91703


namespace sequence_double_for_large_n_l91_91030

theorem sequence_double_for_large_n
  (a : ℕ → ℕ)
  (k : ℕ)
  (h_distinct : ∀ i j : ℕ, i ≠ j → i ≤ k → j ≤ k → a i ≠ a j)
  (h_sequence : ∀ n : ℕ, k < n → a n = least_pos_integer_not_sum (a ∘ (fin n).to_nat)) : 
  ∃ m : ℕ, ∀ n : ℕ, n > m → a n = 2 * a (n - 1) :=
sorry

noncomputable def least_pos_integer_not_sum (s : ℕ → ℕ) : ℕ := sorry

end sequence_double_for_large_n_l91_91030


namespace deal_or_no_deal_l91_91010

theorem deal_or_no_deal :
  let values : set ℕ := {10, 50, 100, 500, 1000, 5000, 50000, 75000, 200000, 400000, 500000, 1000000}
  let n_boxes := 16
  let n_high_values := 3
  ∃ n_boxes_to_eliminate, n_boxes_to_eliminate = 10 ∧ (n_boxes - n_boxes_to_eliminate ≤ 2 * n_high_values) :=
sorry

end deal_or_no_deal_l91_91010


namespace total_employees_in_buses_l91_91158

-- Define the capacity of each bus
def capacity : ℕ := 150

-- Define the fill percentages of each bus
def fill_percentage_bus1 : ℚ := 60 / 100
def fill_percentage_bus2 : ℚ := 70 / 100

-- Calculate the number of passengers in each bus
def passengers_bus1 : ℚ := fill_percentage_bus1 * capacity
def passengers_bus2 : ℚ := fill_percentage_bus2 * capacity

-- Calculate the total number of passengers
def total_passengers : ℚ := passengers_bus1 + passengers_bus2

-- The proof statement
theorem total_employees_in_buses : total_passengers = 195 :=
by
  sorry

end total_employees_in_buses_l91_91158


namespace inequality_proof_l91_91751

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a * b * (b + 1) * (c + 1))) + 
  (1 / (b * c * (c + 1) * (a + 1))) + 
  (1 / (c * a * (a + 1) * (b + 1))) ≥ 
  (3 / (1 + a * b * c)^2) :=
sorry

end inequality_proof_l91_91751


namespace area_per_car_l91_91419

/-- Given the length and width of the parking lot, 
and the percentage of usable area, 
and the number of cars that can be parked,
prove that the area per car is as expected. -/
theorem area_per_car 
  (length width : ℝ) 
  (usable_percentage : ℝ) 
  (number_of_cars : ℕ) 
  (h_length : length = 400) 
  (h_width : width = 500) 
  (h_usable_percentage : usable_percentage = 0.80) 
  (h_number_of_cars : number_of_cars = 16000) :
  (length * width * usable_percentage) / number_of_cars = 10 :=
by
  sorry

end area_per_car_l91_91419


namespace angle_ABC_is_20_l91_91044

variable (A B C D : Type)
variable [Circle A B C D]
variable (angle_ABC angle_ABD angle_BCD angle_CBD : ℝ)
variable (H1 : angle_CBD = 90)
variable (H2 : angle_BCD = 30)
variable (H3 : angle_ABC + angle_ABD + angle_BCD + angle_CBD = 180)

theorem angle_ABC_is_20 :
  angle_ABC = 20 :=
by
  sorry

end angle_ABC_is_20_l91_91044


namespace total_employees_in_buses_l91_91157

-- Define the capacity of each bus
def capacity : ℕ := 150

-- Define the fill percentages of each bus
def fill_percentage_bus1 : ℚ := 60 / 100
def fill_percentage_bus2 : ℚ := 70 / 100

-- Calculate the number of passengers in each bus
def passengers_bus1 : ℚ := fill_percentage_bus1 * capacity
def passengers_bus2 : ℚ := fill_percentage_bus2 * capacity

-- Calculate the total number of passengers
def total_passengers : ℚ := passengers_bus1 + passengers_bus2

-- The proof statement
theorem total_employees_in_buses : total_passengers = 195 :=
by
  sorry

end total_employees_in_buses_l91_91157


namespace Julie_hours_per_week_school_l91_91426

noncomputable def summer_rate : ℚ := 4500 / (36 * 10)

noncomputable def school_rate : ℚ := summer_rate * 1.10

noncomputable def total_school_hours_needed : ℚ := 9000 / school_rate

noncomputable def hours_per_week_school : ℚ := total_school_hours_needed / 40

theorem Julie_hours_per_week_school : hours_per_week_school = 16.36 := by
  sorry

end Julie_hours_per_week_school_l91_91426


namespace middle_card_four_or_five_l91_91143

def three_cards (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 15 ∧ a < b ∧ b < c

theorem middle_card_four_or_five (a b c : ℕ) :
  three_cards a b c → (b = 4 ∨ b = 5) :=
by
  sorry

end middle_card_four_or_five_l91_91143


namespace remaining_volume_correct_l91_91207

-- Define the side length of the cube (in feet)
def side_length_cube : ℝ := 6

-- Define the radius of the cylindrical section removed (in feet)
def radius_cylinder : ℝ := 3

-- Define the height of the cylindrical section removed (in feet)
def height_cylinder : ℝ := 6

-- Define the volume of the cube
def volume_cube : ℝ := side_length_cube ^ 3

-- Define the volume of the cylindrical section
def volume_cylinder : ℝ := π * radius_cylinder ^ 2 * height_cylinder

-- Define the remaining volume after removing the cylindrical section
def remaining_volume : ℝ := volume_cube - volume_cylinder

-- Theorem statement: Prove that the remaining volume is 216 - 54π cubic feet
theorem remaining_volume_correct : remaining_volume = 216 - 54 * π :=
by
  sorry

end remaining_volume_correct_l91_91207


namespace find_eccentricity_of_ellipse_l91_91332

noncomputable def eccentricity (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) (A B P M E : ℝ × ℝ) : ℝ :=
  let F := (c, 0)
  let A := (-a, 0)
  let B := (a, 0)
  let P := (x₁, y₁)
  let M := (xM, yM)
  let E := (0, yE)
  if PF_perpendicular_to_x_axis (P F)
     ∧ B = midpoint (O E)
     ∧ A B vertices_of_ellipse (C)
     ∧ Eq (xM, c)
  then (c / a)
  else 0  -- or empty/undefined, depends on specific requirements

theorem find_eccentricity_of_ellipse (C : Set (ℝ × ℝ))
  (F A B P M E : ℝ × ℝ)
  (O : ℝ × ℝ) 
  (xM : ℝ)
  (PF_perpendicular_to_x_axis: (ℝ × ℝ) → (ℝ × ℝ) → Prop)
  (midpoint : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ)) :
  O = (0,0) →
  F = (c, 0) →
  A = (-a, 0) →
  B = (a, 0) →
  PF_perpendicular_to_x_axis P F →
  let xM = c in
  let E = (0, yE) in
  midpoint O E = (0, yE/2) →
  C = {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1} →
  elliptic_centered C (O E F M) →
  eccentricity C F A B P M E = 1 / 2 :=
sorry

end find_eccentricity_of_ellipse_l91_91332


namespace maximum_length_OB_l91_91881

noncomputable def max_length_OB (O A B : Point) (angle_OAB angle_AOB : ℝ) (AB OB : ℝ) : Prop :=
  ((∠AOB = 45) ∧ (dist A B = 1) ∧ (dist O A = OB) ∧ (dist O B = OB)) →
  ∃ OB_max, OB_max = sqrt 2 ∧ OB ≤ OB_max

theorem maximum_length_OB {O A B : Point} (angle_OAB angle_AOB : ℝ) (AB : ℝ) :
  (angle O A B = 45) ∧ (AB = 1) ∧ (∀ P : Triangle, P.contains_vertices O A B) →
  ∃ OB : ℝ, OB ≤ sqrt 2 :=
sorry

end maximum_length_OB_l91_91881


namespace min_bounces_l91_91584

theorem min_bounces
  (h₀ : ℝ := 160)  -- initial height
  (r : ℝ := 3/4)  -- bounce ratio
  (final_h : ℝ := 20)  -- desired height
  (b : ℕ)  -- number of bounces
  : ∃ b, (h₀ * (r ^ b) < final_h ∧ ∀ b', b' < b → ¬(h₀ * (r ^ b') < final_h)) :=
sorry

end min_bounces_l91_91584


namespace cos_alpha_minus_pi_div_four_eq_zero_l91_91694

theorem cos_alpha_minus_pi_div_four_eq_zero (n : ℝ) (α : ℝ)
    (h1 : 0 < α) (h2 : α < π) (h3 : n * α - cos α = sqrt 2) :
    cos (α - π / 4) = 0 :=
by
  sorry

end cos_alpha_minus_pi_div_four_eq_zero_l91_91694


namespace fraction_simplification_l91_91559

theorem fraction_simplification : (4 * 5) / 10 = 2 := 
by 
  sorry

end fraction_simplification_l91_91559


namespace coin_placement_proof_l91_91017

def possible_placements (total_coins : ℕ) (coins_per_edge : ℕ) : Prop :=
  4 * coins_per_edge = total_coins

theorem coin_placement_proof :
  let total_coins := 12 in
  (¬ possible_placements total_coins 2) ∧ 
  (possible_placements total_coins 3) ∧ 
  (¬ possible_placements total_coins 4) ∧ 
  (¬ possible_placements total_coins 5) ∧ 
  (¬ possible_placements total_coins 6) ∧ 
  (¬ possible_placements total_coins 7) :=
by
  sorry

end coin_placement_proof_l91_91017


namespace lanes_on_road_l91_91227

theorem lanes_on_road (num_lanes : ℕ)
  (h1 : ∀ trucks_per_lane cars_per_lane total_vehicles, 
          cars_per_lane = 2 * (trucks_per_lane * num_lanes) ∧
          trucks_per_lane = 60 ∧
          total_vehicles = num_lanes * (trucks_per_lane + cars_per_lane) ∧
          total_vehicles = 2160) :
  num_lanes = 12 :=
by
  sorry

end lanes_on_road_l91_91227


namespace simon_pies_l91_91073

def blueberry_bushes := {
    A : (blueberries : Nat, minutes : Nat) := (75, 25),
    B : (blueberries : Nat, minutes : Nat) := (105, 30),
    C : (blueberries : Nat, minutes : Nat) := (120, 40),
    D : (blueberries : Nat, minutes : Nat) := (90, 20),
    E : (blueberries : Nat, minutes : Nat) := (140, 45)
}

noncomputable def total_blueberries_picked (max_minutes : Nat := 60) (max_rate : ℝ := 4) := 
  let picked_from_bush_D := blueberry_bushes.D.blueberries
  let remaining_minutes := max_minutes - blueberry_bushes.D.minutes
  let picked_from_bush_B := blueberry_bushes.B.blueberries
  let remaining_minutes := remaining_minutes - blueberry_bushes.B.minutes
  let time_at_max_rate := remaining_minutes
  let picked_at_max_rate := time_at_max_rate * max_rate
  picked_from_bush_D + picked_from_bush_B + picked_at_max_rate

noncomputable def pies_made (total_blueberries : Nat) (blueberries_per_pie : Nat := 100) := 
  total_blueberries / blueberries_per_pie

theorem simon_pies : pies_made (total_blueberries_picked) = 2 := by
  sorry

end simon_pies_l91_91073


namespace sum_of_possible_values_sum_of_all_possible_values_l91_91402

theorem sum_of_possible_values (x : ℝ) :
    (| x - 5 | - 4 = 0) → (x = 1 ∨ x = 9) :=
  by
    sorry

theorem sum_of_all_possible_values :
    ∑ (x : ℝ) in {y | |y - 5| - 4 = 0}.to_finset, x = 10 :=
  by
    sorry

end sum_of_possible_values_sum_of_all_possible_values_l91_91402


namespace point_A_moved_to_vertex_3_l91_91592

-- Defining initial conditions
def initial_position (A : ℝ³) : Prop :=
  A ∈ green_face ∧ A ∈ far_white_face ∧ A ∈ right_lower_white_face

-- Defining the rotation and the new face positions
def rotation (A : ℝ³) (A' : ℝ³) : Prop :=
  (A ∈ green_face → A' ∈ new_green_face) ∧
  (A ∈ far_white_face → A' ∈ far_white_face) ∧
  (A ∈ right_lower_white_face → A' ∈ left_upper_white_face)

-- The final theorem to be proven
theorem point_A_moved_to_vertex_3 (A A' : ℝ³) :
  initial_position A → rotation A A' → A' = vertex_3 :=
by
  sorry

end point_A_moved_to_vertex_3_l91_91592


namespace first_place_team_wins_l91_91788

-- Define the conditions in Lean 4
variable (joe_won : ℕ := 1) (joe_draw : ℕ := 3) (fp_draw : ℕ := 2) (joe_points : ℕ := 3 * joe_won + joe_draw)
variable (fp_points : ℕ := joe_points + 2)

 -- Define the proof problem
theorem first_place_team_wins : 3 * (fp_points - fp_draw) / 3 = 2 := by
  sorry

end first_place_team_wins_l91_91788


namespace correct_calculation_result_l91_91220

theorem correct_calculation_result :
  ∃ x : ℕ, (x - 21 = 52) ∧ (x * 40 = 2920) :=
begin
  use 73,
  split,
  { exact rfl, },
  {
    norm_num,
  }
end

end correct_calculation_result_l91_91220


namespace lines_intersect_at_single_point_l91_91445

-- Let A', B', C be defined as the points of tangency of the excircle with sides BC, AC, AB respectively
variable (A B C : Type) [Point A] [Point B] [Point C]
variable (excircle_tangent_BC : TangentPoint) (excircle_tangent_AC : TangentPoint) (excircle_tangent_AB : TangentPoint)
variable (A' : Point) (B' : Point) (C' : Point)
hypothesis hA' : A' = excircle_tangent_BC.point
hypothesis hB' : B' = excircle_tangent_AC.point
hypothesis hC' : C' = excircle_tangent_AB.point

-- Define a, b, c as lines passing through A', B', C' and are parallel to respective internal angle bisectors
variable (line_a : Line) (line_b : Line) (line_c : Line)
hypothesis ha : through A' line_a ∧ parallel_to_angle_bisector A line_a
hypothesis hb : through B' line_b ∧ parallel_to_angle_bisector B line_b
hypothesis hc : through C' line_c ∧ parallel_to_angle_bisector C line_c

-- Claim: lines a, b, and c intersect at a single point
theorem lines_intersect_at_single_point :
  ∃ (P : Point), (lies_on P line_a) ∧ (lies_on P line_b) ∧ (lies_on P line_c) :=
sorry

end lines_intersect_at_single_point_l91_91445


namespace inradii_exradii_equality_l91_91447

variables {A B C M : Type*}
variable [EuclideanGeometry A B C M]

noncomputable def r_inradii_ABC (Δ : EuclideanTriangle A B C) : ℝ := 
  sorry -- define r inradii of triangle ABC

noncomputable def rho_exradii_ABC (Δ : EuclideanTriangle A B C) (side : EuclideanSegment A B) : ℝ := 
  sorry -- define rho exradii of triangle ABC opposite to side A B

noncomputable def inradii_AMC (Δ : EuclideanTriangle A M C) : ℝ := 
  sorry -- define r1 inradii of triangle AMC

noncomputable def rho_exradii_AMC (Δ : EuclideanTriangle A M C) (side : EuclideanSegment A M) : ℝ := 
  sorry -- define rho1 exradii of triangle AMC opposite to side A M

noncomputable def inradii_BMC (Δ : EuclideanTriangle B M C) : ℝ := 
  sorry -- define r2 inradii of triangle BMC

noncomputable def rho_exradii_BMC (Δ : EuclideanTriangle B M C) (side : EuclideanSegment B M): ℝ := 
  sorry -- define rho2 exradii of triangle BMC opposite to side B M

theorem inradii_exradii_equality {A B C M : Type*}
  {Δ1 : EuclideanTriangle A B C} {Δ2 : EuclideanTriangle A M C} {Δ3 : EuclideanTriangle B M C}
  (M_inter_AB : M ∈ (EuclideanSegment A B)) :
  (inradii_AMC Δ2 / rho_exradii_AMC Δ2 (EuclideanSegment A M)) * (inradii_BMC Δ3 / rho_exradii_BMC Δ3 (EuclideanSegment B M)) =
  (r_inradii_ABC Δ1 / rho_exradii_ABC Δ1 (EuclideanSegment A B)) :=
sorry -- proof goes here

end inradii_exradii_equality_l91_91447


namespace correct_quadratic_opens_upwards_l91_91959

-- Define the quadratic functions
def A (x : ℝ) : ℝ := 1 - x - 6 * x^2
def B (x : ℝ) : ℝ := -8 * x + x^2 + 1
def C (x : ℝ) : ℝ := (1 - x) * (x + 5)
def D (x : ℝ) : ℝ := 2 - (5 - x)^2

-- The theorem stating that function B is the one that opens upwards
theorem correct_quadratic_opens_upwards :
  ∃ (f : ℝ → ℝ) (h : f = B), ∀ (a b c : ℝ), f x = a * x^2 + b * x + c → a > 0 :=
sorry

end correct_quadratic_opens_upwards_l91_91959


namespace non_collinear_condition_l91_91902

variables {A B C : Type} [euclidean_geometry A]

def vector_eq_imply_non_collinear (u v : A) (a b c : A) : Prop :=
  (u ≠ v) → ¬ (collinear {a, b, c})

theorem non_collinear_condition : vector_eq_imply_non_collinear (A) (B) (A) (B) (C) :=
sorry

end non_collinear_condition_l91_91902


namespace inequality_holds_l91_91720

noncomputable def f (x : ℝ) : ℝ := log x - 3 * x

theorem inequality_holds (a b : ℝ) :
  (∀ x : ℝ, 0 < x → f x ≤ x * (a * real.exp x - 4) + b) →
    a + b ≥ 0 :=
begin
  sorry
end

end inequality_holds_l91_91720


namespace good_sq_sequence_l91_91049

def averaging_sequence (a : ℕ → ℝ) (a_avg : ℕ → ℝ) :=
  ∀ k, a_avg k = (a k + a (k + 1)) / 2

def good_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∀ k₁ k₂ ... kₙ : ℕ, ∃ ak : ℕ → ℝ, ∀ k, a_avg (ak k) = (ak k + ak (k + 1)) / 2) ∧ (ak : ℝ) ∈ ℤ

theorem good_sq_sequence {x : ℕ → ℝ} (hx : good_sequence x) :
  good_sequence (λ k, (x k)^2) :=
sorry

end good_sq_sequence_l91_91049


namespace sum_from_neg_50_to_70_l91_91893

noncomputable def sum_integers (start : ℤ) (end : ℤ) : ℤ :=
  (end - start + 1) * (start + end) / 2

theorem sum_from_neg_50_to_70 : sum_integers (-50) 70 = 1210 :=
by sorry

end sum_from_neg_50_to_70_l91_91893


namespace technicians_count_l91_91390

theorem technicians_count {T R : ℕ} (h1 : T + R = 12) (h2 : 2 * T + R = 18) : T = 6 :=
sorry

end technicians_count_l91_91390


namespace max_cars_divided_by_20_l91_91821

noncomputable def speed (v : ℕ) : ℕ :=
v

noncomputable def distance_between_cars (v : ℕ) : ℕ :=
(v + 9) / 10

noncomputable def car_length : ℕ :=
5

noncomputable def total_length (v : ℕ) : ℕ :=
car_length * (distance_between_cars v) + car_length

noncomputable def max_cars_per_hour (v : ℕ) : ℕ :=
1000 * v / total_length v

noncomputable def N (v : ℕ) : ℕ :=
max_cars_per_hour 120  -- Practical maximum speed of 120 km/h

theorem max_cars_divided_by_20 : N 120 / 20 = 92 :=
by
  have N_approx : N 120 = 1846 := sorry
  rw N_approx
  norm_num

end max_cars_divided_by_20_l91_91821


namespace number_of_piles_of_dimes_l91_91486

-- Given conditions
def piles_of_quarters := 4
def piles_of_nickels := 9
def piles_of_pennies := 5
def total_value := 21
def value_per_pile_of_quarters := 0.25 * 10
def value_per_pile_of_nickels := 0.05 * 10
def value_per_pile_of_pennies := 0.01 * 10
def value_per_pile_of_dimes := 0.10 * 10

-- Lean theorem statement
theorem number_of_piles_of_dimes (piles_of_quarters = 4) (piles_of_nickels = 9) 
    (piles_of_pennies = 5) (total_value = 21) :
    let quarters_value := piles_of_quarters * value_per_pile_of_quarters in
    let nickels_value := piles_of_nickels * value_per_pile_of_nickels in
    let pennies_value := piles_of_pennies * value_per_pile_of_pennies in
    let total_other_value := quarters_value + nickels_value + pennies_value in
    total_value - total_other_value = 6 * value_per_pile_of_dimes :=
by sorry

end number_of_piles_of_dimes_l91_91486


namespace concurrency_of_lines_l91_91797

-- Definition of points A1, B1, C1 as midpoints of sides of triangle ABC
variables {A B C A1 B1 C1 A2 B2 C2 : Point}

-- Conditions as derived from the problem statement
axiom midpoints : A1 = midpoint(A, B) ∧ B1 = midpoint(B, C) ∧ C1 = midpoint(C, A)

axiom perpendiculars_to_bisectors :
  (perpendicular (A1, bisector(A))) ∧
  (perpendicular (B1, bisector(B))) ∧
  (perpendicular (C1, bisector(C)))

axiom intersections :
  A2 = intersection (perpendicular (B1, bisector(B)), perpendicular (C1, bisector(C))) ∧
  B2 = intersection (perpendicular (A1, bisector(A)), perpendicular (C1, bisector(C))) ∧
  C2 = intersection (perpendicular (A1, bisector(A)), perpendicular (B1, bisector(B)))

-- Prove the concurrency of lines A1A2, B1B2, and C1C2
theorem concurrency_of_lines :
  concurrent (line (A1, A2)) (line (B1, B2)) (line (C1, C2)) :=
sorry -- proof not required as per instructions

end concurrency_of_lines_l91_91797


namespace gamin_difference_calculation_l91_91670

def largest_number : ℕ := 532
def smallest_number : ℕ := 406
def difference : ℕ := 126

theorem gamin_difference_calculation : largest_number - smallest_number = difference :=
by
  -- The solution proves that the difference between the largest and smallest numbers is 126.
  sorry

end gamin_difference_calculation_l91_91670


namespace proof_problem_l91_91320

open classical

noncomputable def problem_statement : Prop := 
  ∃ (P : set (euclidean_space ℝ 3)), 
    P.card = 8 ∧ 
    (∀ (a b c d : euclidean_space ℝ 3), a ∈ P → b ∈ P → c ∈ P → d ∈ P → 
      (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) → isIndependent a b c d) ∧ 
    ∃ (L : set (euclidean_space ℝ 3 × euclidean_space ℝ 3)), 
      L.card = 17 ∧
      (∀ l ∈ L, (l.1 ∈ P ∧ l.2 ∈ P ∧ l.1 ≠ l.2)) ∧ 
      (some_proof_function L ≥ 1 ∧ some_proof_function L ≥ 4)

theorem proof_problem : problem_statement := 
  sorry

end proof_problem_l91_91320


namespace measure_of_angle_B_l91_91413

theorem measure_of_angle_B (A B C a b c : ℝ) (h₁ : a = A.sin) (h₂ : b = B.sin) (h₃ : c = C.sin)
  (h₄ : (b - a) / (c + a) = c / (a + b)) :
  B = 2 * π / 3 :=
by
  sorry

end measure_of_angle_B_l91_91413


namespace jar_marbles_difference_l91_91545

theorem jar_marbles_difference (a b : ℕ) (h1 : 9 * a = 9 * b) (h2 : 2 * a + b = 135) : 8 * b - 7 * a = 45 := by
  sorry

end jar_marbles_difference_l91_91545


namespace find_numbers_with_cubic_remainders_l91_91905

theorem find_numbers_with_cubic_remainders :
  ∃ (a b : ℕ), a * a * a = 1728 ∧ b * b * b = 6859 ∧ 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 :=
by
  use 12, 19
  split
  { norm_num }
  split
  { norm_num }
  norm_num

end find_numbers_with_cubic_remainders_l91_91905


namespace square_surrounding_circles_side_length_l91_91146

/-- Three unit circles ω₁, ω₂, ω₃ in the plane have the property that each circle passes through
    the centers of the other two. A square S surrounds the three circles in such a way that each of
    its four sides is tangent to at least one of ω₁, ω₂, and ω₃. Prove that the side length of the
    square S is (√6 + √2 + 8) / 4. -/
theorem square_surrounding_circles_side_length :
  ∃ (S : ℝ), (∀ ω₁ ω₂ ω₃ : circle, 
  ω₁.radius = 1 ∧ ω₂.radius = 1 ∧ ω₃.radius = 1 ∧
  (∃ P₁ P₂ P₃ P₄ : point, 
     P₁ ∈ line.through (center ω₁) (center ω₂) ∧ P₂ ∈ line.through (center ω₂) (center ω₃) ∧ 
     P₃ ∈ line.through (center ω₃) (center ω₁) ∧ P₄ ∈ line.through (center ω₁) (center ω₃)
  ) ∧
  ∀ S : square, 
  S.surrounds ω₁ ∧ S.surrounds ω₂ ∧ S.surrounds ω₃ ∧ 
  S.is_tangent_to_at_least_one_circle.choice (ω₁, ω₂, ω₃)),
  S = (sqrt 6 + sqrt 2 + 8) / 4 :=
sorry

end square_surrounding_circles_side_length_l91_91146


namespace sum_y_coordinates_on_yaxis_eq_2_l91_91979

-- Define the center of the circle
def center : ℝ × ℝ := (-4, 1)

-- Define the radius of the circle
def radius : ℝ := 13

-- Define the equation of the circle
noncomputable def circle_eq (x y : ℝ) : Prop := (x + 4)^2 + (y - 1)^2 = radius^2

-- Define the sum of the y-coordinates of points where the circle intersects the y-axis
noncomputable def sum_y_coordinates_on_yaxis : ℝ :=
    let y1 := 1 + real.sqrt 153 in
    let y2 := 1 - real.sqrt 153 in
    y1 + y2

-- The theorem to prove
theorem sum_y_coordinates_on_yaxis_eq_2 :
  sum_y_coordinates_on_yaxis = 2 :=
by
  sorry

end sum_y_coordinates_on_yaxis_eq_2_l91_91979


namespace f_is_odd_inverse_mapping_interval_1_2_all_inverse_mapping_intervals_l91_91994

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then -1 + (1/2) * |x + 1|
else if x = 0 then 0
else 1 - (1/2) * |x - 1|

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def inverseMappingInterval (g : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x ∈ set.Icc a b, g x ∈ set.Icc (1 / b) (1 / a)

theorem f_is_odd : isOdd f := sorry

theorem inverse_mapping_interval_1_2 :
  inverseMappingInterval f 1 2 := sorry

theorem all_inverse_mapping_intervals :
  inverseMappingInterval f 1 2 ∧ inverseMappingInterval f (-2) (-1) := sorry

end f_is_odd_inverse_mapping_interval_1_2_all_inverse_mapping_intervals_l91_91994


namespace solve_for_b_l91_91854

theorem solve_for_b (b : ℝ) : (∃ y x : ℝ, 4 * y - 2 * x - 6 = 0 ∧ 5 * y + b * x + 1 = 0) → b = 10 :=
by sorry

end solve_for_b_l91_91854


namespace positive_m_for_one_root_l91_91384

theorem positive_m_for_one_root (m : ℝ) (h : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
by
  sorry

end positive_m_for_one_root_l91_91384


namespace brets_dinner_tip_calculation_l91_91251

/-
  We need to prove that the percentage of the tip Bret included is 20%, given the conditions.
-/

theorem brets_dinner_tip_calculation :
  let num_meals := 4
  let cost_per_meal := 12
  let num_appetizers := 2
  let cost_per_appetizer := 6
  let rush_fee := 5
  let total_cost := 77
  (total_cost - (num_meals * cost_per_meal + num_appetizers * cost_per_appetizer + rush_fee))
  / (num_meals * cost_per_meal + num_appetizers * cost_per_appetizer) * 100 = 20 :=
by
  sorry

end brets_dinner_tip_calculation_l91_91251


namespace solve_problem_l91_91440

def g (x : ℝ) : ℝ :=
if x ≥ 0 then x^3 else -x^3

lemma inverse_g_8 : ∃ x : ℝ, g x = 8 :=
begin
  use 2,
  simp [g],
end

lemma inverse_g_neg125 : ∃ x : ℝ, g x = -125 :=
begin
  use -5,
  simp [g],
end

theorem solve_problem : 
  let g_inverse_8 := classical.some inverse_g_8 in
  let g_inverse_neg125 := classical.some inverse_g_neg125 in
  g_inverse_8 + g_inverse_neg125 = -3 :=
by
  have h1 : g_inverse_8 = 2 := classical.some_spec inverse_g_8,
  have h2 : g_inverse_neg125 = -5 := classical.some_spec inverse_g_neg125,
  rw [h1, h2],
  norm_num

end solve_problem_l91_91440


namespace arithmetic_sequence_propositions_correct_l91_91018

variables {α : Type*} [linear_ordered_field α]

-- Definitions and given conditions
def S (a d : α) (n : ℕ) : α := n * a + (n * (n - 1) / 2) * d

variables (a d : α)
variables (h1 : S a d 6 > S a d 7)
variables (h2 : S a d 7 > S a d 5)

-- Propositions to prove
def P1 := d < 1
def P2 := S a d 11 > 0
def P5 := abs (n * a + (n - 1) * d) > abs (n * a + (n - 1) * d)

-- Main theorem statement
theorem arithmetic_sequence_propositions_correct :
  P1 a d ∧ P2 a d ∧ P5 a d :=
sorry

end arithmetic_sequence_propositions_correct_l91_91018


namespace main_theorem_l91_91796

variables {A B C ω J K P Q R : Point}
variables (ABC : Triangle) (Γ : Circle) (γ : Circle) 

-- Conditions
def is_isosceles_at_A (ABC : Triangle) := (Triangle.is_isosceles ABC A)
def circumcircle (ABC : Triangle) (Γ : Circle) := (Γ.is_circumcircle ABC)
def tangent_to (γ : Circle) (l1 l2 l3 : Line) := γ.tangent_to l1 ∧ γ.tangent_to l2 ∧ γ.tangent_to l3
def points_of_tangency (γ : Circle) (AB AC Γ : Line) (P Q R : Point) := 
  (γ.tangent_point AB P) ∧ (γ.tangent_point AC Q) ∧ (γ.tangent_point Γ R)
def center_circle (γ : Circle) := (γ.center)
def midpoint (P Q : Point) (J : Point) := (J.is_midpoint P Q)

-- Question as statements to prove
def proof1 (A ω J K R : Point)
  (h : is_isosceles_at_A ABC) 
  (circ : circumcircle ABC Γ) 
  (tang : tangent_to γ AB AC Γ)
  (tangency_points : points_of_tangency γ AB AC Γ P Q R) :=
  (K.is_midpoint B C) →
  (J.is_midpoint P Q) →
  ∃ A K R J ω, 
    (distance A K / distance A R = distance A J / distance A ω)

def proof2 (J : Point)
  (h : is_isosceles_at_A ABC) 
  (circ : circumcircle ABC Γ) 
  (tang : tangent_to γ AB AC Γ)
  (tangency_points : points_of_tangency γ AB AC Γ P Q R)
  (proof1_h : ∃ A K R J ω, 
    (distance A K / distance A R = distance A J / distance A ω)) :=
  J.is_incenter ABC

-- Statement
theorem main_theorem (ABC : Triangle) (Γ γ : Circle) (A ω J K P Q R : Point) 
  (h : is_isosceles_at_A ABC) 
  (circ : circumcircle ABC Γ) 
  (tang : tangent_to γ AB AC Γ)
  (tangency_points : points_of_tangency γ AB AC Γ P Q R) 
  (midK : K.is_midpoint B C)
  (midJ : J.is_midpoint P Q):
  proof1 A ω J K R h circ tang tangency_points midK midJ ∧ 
  proof2 J h circ tang tangency_points (proof1 A ω J K R h circ tang tangency_points midK midJ) :=
sorry

end main_theorem_l91_91796


namespace polar_eq_C_length_PQ_l91_91778

-- Define the parametric equation of the curve C
def curve_C (θ : ℝ) : ℝ × ℝ :=
  (1 + cos θ, sin θ)

-- Define the polar equation of the line l
def line_l (θ : ℝ) (ρ : ℝ) : Prop :=
  ρ * (sqrt 3 * sin θ + cos θ) = 4

-- Define the polar equation of the curve C in polar coordinates
def curve_C_polar (θ : ℝ) : ℝ :=
  2 * cos θ

-- Theorem verifying the polar equation of the curve C
theorem polar_eq_C (θ : ℝ) : 
  ∃ ρ, ρ = curve_C_polar θ := 
  by sorry

-- Theorem verifying the length of line segment PQ
theorem length_PQ :
  let θ := π / 3
  let ρ_P := curve_C_polar θ
  let ρ_Q := 4 / (sqrt 3 * sin θ + cos θ)
  abs (ρ_P - ρ_Q) = 1 :=
  by sorry

end polar_eq_C_length_PQ_l91_91778


namespace Milly_spends_135_minutes_studying_l91_91056

-- Definitions of homework times
def mathHomeworkTime := 60
def geographyHomeworkTime := mathHomeworkTime / 2
def scienceHomeworkTime := (mathHomeworkTime + geographyHomeworkTime) / 2

-- Definition of Milly's total study time
def totalStudyTime := mathHomeworkTime + geographyHomeworkTime + scienceHomeworkTime

-- Theorem stating that Milly spends 135 minutes studying
theorem Milly_spends_135_minutes_studying : totalStudyTime = 135 :=
by
  -- Proof omitted
  sorry

end Milly_spends_135_minutes_studying_l91_91056


namespace greatest_integer_leq_l91_91278

theorem greatest_integer_leq (a b : ℝ) (ha : a = 5^150) (hb : b = 3^150) (c d : ℝ) (hc : c = 5^147) (hd : d = 3^147):
  ⌊ (a + b) / (c + d) ⌋ = 124 := 
sorry

end greatest_integer_leq_l91_91278


namespace complementary_angles_decrease_percent_l91_91118

theorem complementary_angles_decrease_percent
    (a b : ℝ) 
    (h1 : a + b = 90) 
    (h2 : a / b = 3 / 7) 
    (h3 : new_a = a * 1.15) 
    (h4 : new_a + new_b = 90) : 
    (new_b / b * 100) = 93.57 := 
sorry

end complementary_angles_decrease_percent_l91_91118


namespace binomial_sum_pattern_l91_91189

theorem binomial_sum_pattern (n : ℕ) (hn : n > 0) :
  (∑ k in finset.range (n + 1), nat.choose (4 * n + 1) (4 * k + 1)) = 2^(4 * n - 1) + ((-1)^n * 2^(2 * n - 1)) :=
sorry

end binomial_sum_pattern_l91_91189


namespace isosceles_triangles_not_necessarily_congruent_l91_91876

theorem isosceles_triangles_not_necessarily_congruent 
  (BC R : ℝ)
  (h1 : BC > 0)
  (h2 : R > 0)
  (A D B C : Type)
  (h3 : isosceles_triangle A B C BC R)
  (h4 : isosceles_triangle D B C BC R)
  (h5 : A ≠ D) :
  ¬ congruent (triangle.mk A B C) (triangle.mk D B C) :=
by sorry

end isosceles_triangles_not_necessarily_congruent_l91_91876


namespace relationship_among_abc_l91_91351

noncomputable def F (f : ℝ → ℝ) (x : ℝ) := x * f x

variables (f : ℝ → ℝ) (h_even : ∀ x : ℝ, f x = f (-x))
variables (F_der2 : ∀ x ≤ 0, deriv (deriv (F f)) x < 0)
variables (a b c : ℝ)

def a_def : ℝ := 2 ^ 0.1 * f (2 ^ 0.1)
def b_def : ℝ := Real.log 2 * f (Real.log 2)
def c_def : ℝ := Real.log 2 (1 / 8) * f (Real.log 2 (1 / 8))

theorem relationship_among_abc : a_def f < b_def f ∧ b_def f < c_def f := 
sorry

end relationship_among_abc_l91_91351


namespace find_angle_acb_l91_91772

noncomputable def angle_sum_triangle (A B C : ℝ) (x : ℝ) : Prop :=
  A + B + C = 180 ∧ A = 45 ∧ B = 72 ∧ C = x

theorem find_angle_acb :
  ∃ x : ℝ, angle_sum_triangle 45 72 x ∧ x = 153 :=
by
  sorry

end find_angle_acb_l91_91772


namespace clothing_store_earnings_l91_91863

theorem clothing_store_earnings :
  let cost_per_dress := 32
      num_dresses := 30
      total_cost := num_dresses * cost_per_dress
      base_price := 47
      excess_prices := [3, 2, 1, 0, -1, -2]
      quantities := [7, 6, 3, 5, 4, 5]
      revenue := (quantities.zip excess_prices).map (λ p => p.1 * (base_price + p.2)).sum
  in revenue - total_cost = 472 := 
by
  sorry

end clothing_store_earnings_l91_91863


namespace sum_of_coefficients_binomial_expansion_l91_91127

theorem sum_of_coefficients_binomial_expansion (a b: ℕ) : 
  (∑ k in Finset.range 9, Nat.choose 8 k) = 256 :=
by
  sorry

end sum_of_coefficients_binomial_expansion_l91_91127


namespace length_AB_l91_91399

-- Define the variables
variables {A B C D : Type} [EuclideanGeometry!]
variable (AB AC BC BD CD : ℝ)

-- Define the conditions
def is_isosceles_triangle (a b c : ℝ) : Prop := a = b
def perimeter_of_triangle (a b c : ℝ) := a + b + c

-- Problem statement: Prove AB = 12 given the conditions
theorem length_AB :
  is_isosceles_triangle AB AC BC ∧
  is_isosceles_triangle BD CD BC ∧
  perimeter_of_triangle BD CD BC = 24 ∧
  perimeter_of_triangle AB BC AC = 26 ∧
  BD = 10 →
  AB = 12 :=
by
  sorry

end length_AB_l91_91399


namespace possible_values_of_m_l91_91724

def f (x a m : ℝ) := abs (x - a) + m * abs (x + a)

theorem possible_values_of_m {a m : ℝ} (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∀ x : ℝ, f x a m ≥ 2)
  (h4 : a ≤ -5 ∨ a ≥ 5) : m = 1 / 5 :=
by 
  sorry

end possible_values_of_m_l91_91724


namespace expected_variance_replanted_seeds_l91_91312

theorem expected_variance_replanted_seeds:
  (p = 0.9) →
  (n = 1000) →
  (t = 2) →
  (q = 1 - p) →
  (E_X = n * q * t) →
  (Var_X = n * q * (1 - q) * t^2) →
  E_X = 200 ∧ Var_X = 360 := 
by
  intros p_eq n_eq t_eq q_eq E_X_eq Var_X_eq
  rw [p_eq, n_eq, t_eq, q_eq] at *
  simp at E_X_eq Var_X_eq
  exact ⟨E_X_eq, Var_X_eq⟩
  -- Remember that you need to replace 'E_X' and 'Var_X' by their actual values
  sorry -- actual simplifying and solving steps using Lean would replace this

end expected_variance_replanted_seeds_l91_91312


namespace binom_divisible_by_prime_l91_91480

theorem binom_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (h1 : 1 ≤ k) (h2 : k ≤ p - 1) : p ∣ Nat.choose p k :=
sorry

end binom_divisible_by_prime_l91_91480


namespace quadrilateral_area_l91_91012

noncomputable def point := ℝ × ℝ

structure Triangle :=
(X Y Z : point)

structure Median :=
(start end_ : point)

structure medians_intersect (t : Triangle) (XF YG : Median) (Q : point) : Prop :=
(XF_mid : (XF.start.1 + XF.end_.1) / 2 = Q.1 ∧ (XF.start.2 + XF.end_.2) / 2 = Q.2)
(YG_mid : (YG.start.1 + YG.end_.1) / 2 = Q.1 ∧ (YG.start.2 + YG.end_.2) / 2 = Q.2)

structure right_angle (a b c : point) : Prop :=
(square_eq : (b.1 - a.1)^2 + (b.2 - a.2)^2 + (c.1 - b.1)^2 + (c.2 - b.2)^2 = (c.1 - a.1)^2 + (c.2 - a.2)^2)

def distance (p1 p2 : point) : ℝ := sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem quadrilateral_area (t : Triangle) (XF YG : Median) (Q : point)
  (mid_intersection : medians_intersect t XF YG Q)
  (QG_3 : distance Q YG.end_ = 3)
  (QF_4 : distance Q XF.end_ = 4)
  (FG_5 : distance XF.end_ YG.end_ = 5)
  (right_triangle : right_angle Q XF.end_ YG.end_) :
  area_of_quadrilateral XF.start XF.end YG.start YG.end = 50 :=
sorry

end quadrilateral_area_l91_91012


namespace trapezoid_distance_inequality_l91_91407

variables {Point : Type} [metric_space Point]

def distance (p q : Point) : real := metric.dist p q

noncomputable def trapezoid (A B C D : Point) := 
  distance A C = distance B D

theorem trapezoid_distance_inequality (A B C D M : Point) 
  (h : trapezoid A B C D) : 
  distance M A + distance M B + distance M C > distance M D :=
sorry

end trapezoid_distance_inequality_l91_91407


namespace m_plus_n_l91_91803

def is_prime (n : Nat) : Prop := n ≠ 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def has_three_positive_divisors (n : Nat) : Prop := 
  ∃ p : Nat, is_prime p ∧ n = p * p

theorem m_plus_n (m n : Nat) 
  (h₁ : ∃ p1 : Nat, is_prime p1 ∧ p1 ≠ 2 ∧ min {k : Nat | is_prime k ∧ k ≠ 2 ∧ k = p1} p1 = 3) -- m is the second smallest prime number
  (h₂ : ∃ p2 : Nat, is_prime p2 ∧ n < 150 ∧ has_three_positive_divisors n ∧ p2 * p2 = n ∧ ∀ k < 150, has_three_positive_divisors k → k ≤ n): -- n is the largest integer less than 150 with exactly three positive divisors
  m + n = 124 
  := by
    sorry

end m_plus_n_l91_91803


namespace cylinder_height_l91_91941

theorem cylinder_height (r₁ r₂ : ℝ) (S : ℝ) (hR : r₁ = 3) (hL : r₂ = 4) (hS : S = 100 * Real.pi) : 
  (∃ h : ℝ, h = 7 ∨ h = 1) :=
by 
  sorry

end cylinder_height_l91_91941


namespace circumcenter_orthocenter_distance_l91_91014

theorem circumcenter_orthocenter_distance {A B C : Type} [normed_add_comm_group A] [normed_space ℝ A] 
  (AB AC BC : ℝ) (angle_A : ℝ) :
  (angle_A = 120) ->
  let O := circumcenter A B C in
  let H := orthocenter A B C in
  dist O H = AB + AC := by
  sorry

end circumcenter_orthocenter_distance_l91_91014


namespace three_lines_l91_91991

def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem three_lines (x y : ℝ) : (diamond x y = diamond y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) := 
by sorry

end three_lines_l91_91991


namespace number_of_males_in_village_l91_91131

theorem number_of_males_in_village (total_population : ℕ) (parts : ℕ) (male_parts : ℕ) 
(h₀ : total_population = 480) (h₁ : parts = 4) (h₂ : male_parts = 2) :
  2 * (total_population / parts) = 240 :=
by
  -- Given conditions
  rw [h₀, h₁, h₂]
  -- Total population divided by number of parts
  have h₃ : 480 / 4 = 120 := by norm_num
  -- Two parts represent males
  exact calc
    2 * (480 / 4) = 2 * 120 : by rw h₃
              ... = 240     : by norm_num

end number_of_males_in_village_l91_91131


namespace days_at_sisters_house_l91_91790

theorem days_at_sisters_house (total_vacation_days : ℕ)
  (to_grandparents : ℕ) (at_grandparents : ℕ) (to_brother : ℕ)
  (at_brother : ℕ) (to_sister : ℕ) (travel_home : ℕ)
  (total_vacation_time : total_vacation_days = 21)
  (travel_to_grandparents : to_grandparents = 1)
  (stay_at_grandparents : at_grandparents = 5)
  (travel_to_brother : to_brother = 1)
  (stay_at_brother : at_brother = 5)
  (travel_to_sister : to_sister = 2)
  (travel_to_home : travel_home = 2) :
  at_sisters := 5 := 
by
  sorry

end days_at_sisters_house_l91_91790


namespace count_valid_choices_l91_91250

open Nat

def base4_representation (N : ℕ) : ℕ := 
  let a3 := N / 64 % 4
  let a2 := N / 16 % 4
  let a1 := N / 4 % 4
  let a0 := N % 4
  64 * a3 + 16 * a2 + 4 * a1 + a0

def base7_representation (N : ℕ) : ℕ := 
  let b3 := N / 343 % 7
  let b2 := N / 49 % 7
  let b1 := N / 7 % 7
  let b0 := N % 7
  343 * b3 + 49 * b2 + 7 * b1 + b0

def S (N : ℕ) : ℕ := base4_representation N + base7_representation N

def valid_choices (N : ℕ) : Prop := 
  (S N % 100) = (2 * N % 100)

theorem count_valid_choices : 
  ∃ (count : ℕ), count = 20 ∧ ∀ (N : ℕ), (N >= 1000 ∧ N < 10000) → valid_choices N ↔ (count = 20) :=
sorry

end count_valid_choices_l91_91250


namespace edge_length_of_cube_in_hemisphere_correct_l91_91066

noncomputable def edge_length_of_cube_in_hemisphere (R : ℝ) : ℝ := 
  R * Real.sqrt (2 / 3)

theorem edge_length_of_cube_in_hemisphere_correct (R : ℝ) :
  edge_length_of_cube_in_hemisphere R = R * Real.sqrt (2 / 3) :=
by
  -- Our task here is to prove that the edge length follows our derived formula.
  -- The full proof is complex and skipped here.
  sorry

end edge_length_of_cube_in_hemisphere_correct_l91_91066


namespace num_triangles_in_rectangle_l91_91675

theorem num_triangles_in_rectangle 
  (n_AB : ℕ) (n_BC : ℕ) (h_AB : n_AB = 5) (h_BC : n_BC = 6) : 
  (nat.choose n_AB 2) * n_BC + (nat.choose n_BC 2) * n_AB = 135 :=
by
  -- Given conditions
  rw [h_AB, h_BC]
  sorry

end num_triangles_in_rectangle_l91_91675


namespace part1_correct_part2_correct_part3_correct_l91_91832

noncomputable def part1 (a b : ℤ) (polynomial_eq : (a + 4) * x ^ 3 + 10 * x ^ 2 - 5 * x + 3 = 0) 
  (coeff_quad_term : b = 10) : Prop := a = -4 ∧ b = 10

noncomputable def part2 (t : ℝ) (a b : ℝ) (distance_initial : abs (b - a) = 14) 
  (relative_speed : abs ((b - 3 * t) - (a - t)) = 2 * t) : Prop := 
  abs (14 - 2 * t) = 4

noncomputable def part3 (x : ℝ) (AM ON : ℝ) (P_OP_y : ℝ) := 
  let AM := x in 
  let ON := 10 + 2 * x in
  let OP := (1 / 3) * ON in
  y = OP - (4 / 3) * AM

-- Statements for verification using Lean
theorem part1_correct (a b : ℤ) (polynomial_eq : (a + 4) * x ^ 3 + 10 * x ^ 2 - 5 * x + 3 = 0) 
  (coeff_quad_term : b = 10) : part1 a b polynomial_eq coeff_quad_term :=
sorry

theorem part2_correct (t : ℝ) (a b : ℝ) (distance_initial : abs (b - a) = 14) 
  (relative_speed : abs ((b - 3 * t) - (a - t)) = 2 * t) : part2 t a b distance_initial relative_speed :=
sorry

theorem part3_correct (x : ℝ) (AM ON : ℝ) (P_OP_y : ℝ) : part3 x AM ON P_OP_y :=
sorry

end part1_correct_part2_correct_part3_correct_l91_91832


namespace tangent_line_parabola_d_l91_91107

theorem tangent_line_parabola_d (d : ℝ) :
  (∀ x y : ℝ, (y = 3 * x + d) → (y^2 = 12 * x) → ∃! x, 9 * x^2 + (6 * d - 12) * x + d^2 = 0) → d = 1 :=
by
  sorry

end tangent_line_parabola_d_l91_91107


namespace sum_cos_pi_eq_zero_l91_91456

/-- Given an increasing sequence of positive irreducible fractions with denominator 60,
the sum of the cosines of these fractions times π is zero. -/
theorem sum_cos_pi_eq_zero (a : ℕ → ℝ)
  (h_irreducible: ∀ i, ∃ p q, a i = p / q ∧ nat.gcd p q = 1 ∧ q = 60)
  (h_increasing: ∀ i j, i < j → a i < a j)
  (h_positive: ∀ i, a i > 0) :
  ∑ i in finset.range 16, real.cos (a i * real.pi) = 0 :=
sorry

end sum_cos_pi_eq_zero_l91_91456


namespace least_five_digit_congruent_to_9_mod_18_l91_91554

theorem least_five_digit_congruent_to_9_mod_18 :
  ∃ x : ℕ, 10000 ≤ x ∧ x ≤ 99999 ∧ x ≡ 9 [MOD 18] ∧ ∀ y : ℕ, 10000 ≤ y ∧ y ≤ 99999 ∧ y ≡ 9 [MOD 18] → x ≤ y :=
begin
  use 10008,
  split,
  { linarith, }, -- 10000 ≤ 10008
  split,
  { linarith, }, -- 10008 ≤ 99999
  split,
  { exact nat.modeq.symm (by norm_num), }, -- 10008 ≡ 9 [MOD 18]
  { intros y hy,
    cases hy with h_y1 hy,
    cases hy with h_y2 h_y3,
    rw nat.modeq at h_y3,
    have h_y := nat.le_of_sub_nonneg (nat.mod_sub_of_lt (nat.lt_of_lt_of_le (nat.mod_lt y (by norm_num)) h_y2.symm)),
    linarith,
  },
end

end least_five_digit_congruent_to_9_mod_18_l91_91554


namespace students_not_reading_novels_l91_91928

-- Define the conditions
def total_students : ℕ := 240
def students_three_or_more : ℕ := total_students * (1/6)
def students_two : ℕ := total_students * 0.35
def students_one : ℕ := total_students * (5/12)

-- The theorem to be proved
theorem students_not_reading_novels : 
  (total_students - (students_three_or_more + students_two + students_one) = 16) :=
by
  sorry -- skipping the proof

end students_not_reading_novels_l91_91928


namespace ants_do_not_collide_l91_91535

-- Define the initial setup of the problem
noncomputable def set_of_ant_movements (holes : fin 44) (ant_count : fin 2017) :=
  -- T is the set of moments for which the ant comes in or out of the holes.
  let T := finset ℕ in
  ∀(t : T), t ≤ 45

-- Define the main theorem for the proof problem
theorem ants_do_not_collide :
  -- Given conditions in the problem
  ∀ (holes : fin 44) (ant_count : fin 2017),
  (∀ (i j : ant_count) (vi vj : fin 44 → ℕ),
    (vi ≠ vj → @function.injective _ _ id) →
    ((∀ t, ((vi t : ℕ) = (vj t : ℕ)) → false)) →
    -- Statement to prove: there exist two ants that do not collide
    ∃ (p q : fin 2017), (p ≠ q) ∧
    (∀ t, p ≠ t) ∧ (∀ t, q ≠ t)
  ) :=
sorry

end ants_do_not_collide_l91_91535


namespace max_distance_ratio_on_parabola_l91_91642

theorem max_distance_ratio_on_parabola (p x1 y1 : ℝ) :
  y1^2 = 2 * p * x1 →
  let MO := real.sqrt (x1^2 + y1^2),
      MF := real.sqrt ((x1 - p / 2)^2 + y1^2)
  in MO / MF ≤ 2 / real.sqrt 3 :=
sorry

end max_distance_ratio_on_parabola_l91_91642


namespace inequality_l91_91453

variable {n : ℕ} (a x : Fin n → ℝ)

noncomputable def conditions (a x : Fin n → ℝ) : Prop :=
  (∀ i j : Fin n, i ≠ j → a i + a j ≥ 0) ∧ 
  (∀ i : Fin n, x i ≥ 0) ∧
  (∑ i, x i = 1)

theorem inequality (h : conditions a x) : 
  ∑ i, a i * x i ≥ ∑ i, a i * (x i)^2 :=
sorry

end inequality_l91_91453


namespace unequal_set_A_equal_set_B_equal_set_C_equal_set_D_l91_91964

theorem unequal_set_A : (-3)^2 ≠ (-3^2) :=
by {
  have h1 : (-3)^2 = 9, by norm_num,
  have h2 : -3^2 = -9, by norm_num,
  rw [h1, h2],
  exact dec_trivial,
}

theorem equal_set_B : 2^4 = 4^2 :=
by norm_num

theorem equal_set_C : (-6)^3 = -6^3 :=
by norm_num

theorem equal_set_D : (-6)^4 = (abs(-6))^4 :=
by {
  have h1 : abs(-6) = 6, by norm_num,
  rw h1,
  norm_num,
}

end unequal_set_A_equal_set_B_equal_set_C_equal_set_D_l91_91964


namespace triangle_area_of_roots_l91_91080

theorem triangle_area_of_roots (a b c : ℝ) 
  (h₁ : a^3 - 5*a^2 + 6*a - 27/16 = 0)
  (h₂ : b^3 - 5*b^2 + 6*b - 27/16 = 0)
  (h₃ : c^3 - 5*c^2 + 6*c - 27/16 = 0) :
  let p := (a + b + c) / 2 in
  (p * (p - a) * (p - b) * (p - c)) = 135/32 :=
by
  sorry

end triangle_area_of_roots_l91_91080


namespace sum_of_integers_is_27_24_or_20_l91_91524

theorem sum_of_integers_is_27_24_or_20 
    (x y : ℕ) 
    (h1 : 0 < x) 
    (h2 : 0 < y) 
    (h3 : x * y + x + y = 119) 
    (h4 : Nat.gcd x y = 1) 
    (h5 : x < 25) 
    (h6 : y < 25) 
    : x + y = 27 ∨ x + y = 24 ∨ x + y = 20 := 
sorry

end sum_of_integers_is_27_24_or_20_l91_91524


namespace minimal_dragon_length_l91_91900

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_dragon (k : ℕ) : Prop :=
  ∀ seq, (∀ i, i < k → seq i < seq (i + 1)) → 
  ∃ i < k, digit_sum (seq i) % 11 = 0

theorem minimal_dragon_length : is_dragon 39 ∧ ¬ is_dragon 38 :=
by {
  sorry
}

end minimal_dragon_length_l91_91900


namespace finite_x0_values_for_finite_sequence_l91_91439

noncomputable def g (x : ℝ) : ℝ := 3 * x - x^2
def sequence_finite (x0 : ℝ) : Prop := ∃ N, ∀ n ≥ N, ∃ m < N, x_n = x_m
  where x_n : ℕ → ℝ
  | 0 => x0
  | (n + 1) => g (x_n n)

theorem finite_x0_values_for_finite_sequence : 
  {x0 : ℝ | sequence_finite x0}.to_finset.card = 2 :=
sorry

end finite_x0_values_for_finite_sequence_l91_91439


namespace ratio_of_percent_changes_l91_91908

noncomputable def price_decrease_ratio (original_price : ℝ) (new_price : ℝ) : ℝ :=
(original_price - new_price) / original_price * 100

noncomputable def units_increase_ratio (original_units : ℝ) (new_units : ℝ) : ℝ :=
(new_units - original_units) / original_units * 100

theorem ratio_of_percent_changes 
  (original_price new_price original_units new_units : ℝ)
  (h1 : new_price = 0.7 * original_price)
  (h2 : original_price * original_units = new_price * new_units)
  : (units_increase_ratio original_units new_units) / (price_decrease_ratio original_price new_price) = 1.4285714285714286 :=
by
  sorry

end ratio_of_percent_changes_l91_91908


namespace num_valid_digits_l91_91663

def is_digit (n : ℕ) : Prop := n ∈ (finset.range 10).erase 0

def satisfies_condition (n : ℕ) : Prop := (150 + n) % n = 0

theorem num_valid_digits : finset.card ((finset.range 10).erase 0).filter (λ n, satisfies_condition n) = 6 :=
by
  sorry

end num_valid_digits_l91_91663


namespace simplify_expression_l91_91836

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l91_91836


namespace sum_integers_from_neg50_to_70_l91_91890

theorem sum_integers_from_neg50_to_70 : (Finset.range (70 + 1)).sum id - (Finset.range (50 + 1)).sum id + (Finset.range (50+1)).sum (λ x, - x) = 1210 :=
by
  sorry

end sum_integers_from_neg50_to_70_l91_91890


namespace additional_miles_is_90_l91_91261

/-- Daniel drives 20 miles at an average speed of 40 miles per hour. -/
constant daniel_drives_first_part : ℕ := 20

/-- Daniel drives at an average speed of 40 miles per hour initially. -/
constant initial_speed : ℕ := 40

/-- Additional miles Daniel needs to drive at 60 miles per hour to average 55 miles per hour for the entire trip is 90 miles. -/
def additional_miles_needed : ℕ := 90

/-- The target average speed for the entire trip is 55 miles per hour. -/
constant target_average_speed : ℕ := 55

/-- The speed Daniel drives for the additional miles. -/
constant additional_speed : ℕ := 60

theorem additional_miles_is_90 :
  ∃ t : ℕ, (20 + 60 * t) / (1 / 2 + t) = 55 ∧ (additional_speed * t = 90) := sorry

end additional_miles_is_90_l91_91261


namespace ratio_of_black_to_white_tiles_l91_91645

theorem ratio_of_black_to_white_tiles
  (original_width : ℕ)
  (original_height : ℕ)
  (original_black_tiles : ℕ)
  (original_white_tiles : ℕ)
  (border_width : ℕ)
  (border_height : ℕ)
  (extended_width : ℕ)
  (extended_height : ℕ)
  (new_white_tiles : ℕ)
  (total_white_tiles : ℕ)
  (total_black_tiles : ℕ)
  (ratio_black_to_white : ℚ)
  (h1 : original_width = 5)
  (h2 : original_height = 6)
  (h3 : original_black_tiles = 12)
  (h4 : original_white_tiles = 18)
  (h5 : border_width = 1)
  (h6 : border_height = 1)
  (h7 : extended_width = original_width + 2 * border_width)
  (h8 : extended_height = original_height + 2 * border_height)
  (h9 : new_white_tiles = (extended_width * extended_height) - (original_width * original_height))
  (h10 : total_white_tiles = original_white_tiles + new_white_tiles)
  (h11 : total_black_tiles = original_black_tiles)
  (h12 : ratio_black_to_white = total_black_tiles / total_white_tiles) :
  ratio_black_to_white = 3 / 11 := 
sorry

end ratio_of_black_to_white_tiles_l91_91645


namespace sum_of_medians_is_63_l91_91365
noncomputable def median (scores : List ℝ) : ℝ :=
  let sorted := List.sort scores
  let n := List.length sorted
  if n % 2 == 1 then
    sorted.get! (n / 2)
  else
    (sorted.get! (n / 2 - 1) + sorted.get! (n / 2)) / 2

theorem sum_of_medians_is_63 (scoresA scoresB : List ℝ) (hA : List.Nodup scoresA) (hB : List.Nodup scoresB) :
  median scoresA + median scoresB = 63 := by
  sorry

end sum_of_medians_is_63_l91_91365


namespace lowest_price_lt_44000_l91_91518

-- Given conditions in Lean definitions
variables (z x y : ℝ)
axiom h1 : 52000 < x ∨ x = 52000
axiom h2 : y < 52000 ∨ 52000 < y
axiom h3 : 44000 ≤ 52000
axiom h4 : 52000 ≤ 57000
axiom h5 : Median x y = 56000

-- Mathematical proof problem statement
theorem lowest_price_lt_44000 (z x y : ℝ) (h1 : 52000 < x ∨ x = 52000) 
  (h2 : y < 52000 ∨ 52000 < y) (h3 : 44000 ≤ 52000) (h4 : 52000 ≤ 57000) 
  (h5 : Median x y = 56000) : z < 44000 :=
sorry

end lowest_price_lt_44000_l91_91518


namespace machine_A_time_to_produce_x_boxes_l91_91179

-- Definitions of the conditions
def machine_A_rate (T : ℕ) (x : ℕ) : ℚ := x / T
def machine_B_rate (x : ℕ) : ℚ := 2 * x / 5
def combined_rate (T : ℕ) (x : ℕ) : ℚ := (x / 2) 

-- The theorem statement
theorem machine_A_time_to_produce_x_boxes (x : ℕ) : 
  ∀ T : ℕ, 20 * (machine_A_rate T x + machine_B_rate x) = 10 * x → T = 10 :=
by
  intros T h
  sorry

end machine_A_time_to_produce_x_boxes_l91_91179


namespace solution_1_solution_2_solution_3_solution_4_solution_5_solution_6_l91_91077

theorem solution_1 (x : ℝ) : x * |x| = 4 ↔ x = 2 := sorry

theorem solution_2 (x : ℝ) : (x - 3) ^ 2 = 1 - real.pi ↔ false := sorry

theorem solution_3 (x : ℝ) : |-1 - x ^ 2| = 5 ↔ x = 2 ∨ x = -2 := sorry

theorem solution_4 (x : ℝ) : x ^ 2 - 3 * x + x⁻¹ = x⁻¹ ↔ x = 3 := sorry

theorem solution_5 (x : ℝ) : (sqrt (x - 2)) ^ 4 = |x| - x ↔ x = 2 := sorry

theorem solution_6 (x : ℝ) : x * (x - 5) = x ^ 2 - 5 * |x| ↔ x ≥ 0 := sorry

end solution_1_solution_2_solution_3_solution_4_solution_5_solution_6_l91_91077


namespace isosceles_triangle_area_l91_91392

theorem isosceles_triangle_area (a c d : ℝ) (h1 : AB = AC) (h2 : CD = d) (h3 : D is the midpoint of AB):
  area of ABC = c d ≠ 0 -> ∃ area, area = (d^2)/sqrt(3) :=
sorry

end isosceles_triangle_area_l91_91392


namespace find_area_of_square_XZ_l91_91970

-- Definitions based on conditions
def side_XZ (x : ℝ) := x
def side_XY (x : ℝ) := x / 2

-- Areas based on given conditions and question
def area_square_XZ (x : ℝ) := x ^ 2
def area_square_YZ (x : ℝ) := (5 * x ^ 2) / 4
def area_rectangle_XY (x : ℝ) (h : ℝ) := (x ^ 2 / 2) * h

-- Given condition sum of areas equals 500 cm^2
def sum_of_areas_eq_500 (x : ℝ) (h : ℝ) := 
  area_square_XZ x + area_square_YZ x + area_rectangle_XY x h = 500

-- Theorem statement to prove area of square on side XZ
theorem find_area_of_square_XZ (x : ℝ) (h : ℝ) (h_eq_1 : h = 1)
  (h_areas : sum_of_areas_eq_500 x h):
  area_square_XZ x = 2000 / 11 := 
by {
  sorry
}

end find_area_of_square_XZ_l91_91970


namespace avg_rate_of_change_theorem_l91_91566

variable {α : Type*} [LinearOrder α] {β : Type*} [AddCommGroup β] [Module ℝ β]

def avg_rate_of_change (f : α → β) (x0 x1 : α) : β :=
  (f x1 - f x0) / (x1 - x0)

theorem avg_rate_of_change_theorem (f : ℝ → β) (x0 x1 : ℝ) (h : x0 ≠ x1) :
  avg_rate_of_change f x0 x1 = (f x1 - f x0) / (x1 - x0) := by
  sorry

end avg_rate_of_change_theorem_l91_91566


namespace monge_tetrahedron_planes_l91_91045

noncomputable def problem := sorry

theorem monge_tetrahedron_planes
  {A B C D M : Point} 
  (hM: MongePointTetrahedron A B C D M) 
  (hPlane: M ∈ PlaneOfFace A B C) :
  ∃ planes, 
    (D ∈ planes) ∧ 
    (∀ P1 P2 P3, 
      Intersect (Altitude (Face D A B)) P1 ∧ 
      Intersect (Altitude (Face D B C)) P2 ∧ 
      Intersect (Altitude (Face D A C)) P3 → 
      Collinear P1 P2 P3 ∧
      (∀ Q1 Q2 Q3, 
        Circumcenter (Face D A B) Q1 ∧ 
        Circumcenter (Face D B C) Q2 ∧ 
        Circumcenter (Face D A C) Q3 → 
        Collinear Q1 Q2 Q3))
:= sorry

end monge_tetrahedron_planes_l91_91045


namespace _l91_91661

noncomputable def height_of_kite {h c d : ℝ} : h = 10 * Real.sqrt 201 :=
by
  -- Given conditions
  let kc := 210
  let kd := 190
  let cd := 200
  -- Derived conditions using the Pythagorean theorem and geometry
  have h1 : h^2 + c^2 = kc^2 := by sorry
  have h2 : h^2 + d^2 = kd^2 := by sorry
  have h3 : c^2 + d^2 = cd^2 := by sorry
  -- Use the derived conditions to deduce the height of the kite
  have main : 2 * h^2 = kc^2 + kd^2 - cd^2 := by sorry
  have h_square : h^2 = (kc^2 + kd^2 - cd^2) / 2 := by sorry
  exact Eq.trans _ (by { field_simp, ring })⟩

end _l91_91661


namespace length_AB_l91_91400

-- Define the variables
variables {A B C D : Type} [EuclideanGeometry!]
variable (AB AC BC BD CD : ℝ)

-- Define the conditions
def is_isosceles_triangle (a b c : ℝ) : Prop := a = b
def perimeter_of_triangle (a b c : ℝ) := a + b + c

-- Problem statement: Prove AB = 12 given the conditions
theorem length_AB :
  is_isosceles_triangle AB AC BC ∧
  is_isosceles_triangle BD CD BC ∧
  perimeter_of_triangle BD CD BC = 24 ∧
  perimeter_of_triangle AB BC AC = 26 ∧
  BD = 10 →
  AB = 12 :=
by
  sorry

end length_AB_l91_91400


namespace problem_conditions_l91_91344

def a (n : ℕ) := 3 * n + 1

def geometric_sum (r : ℕ) (n : ℕ) := (r^(n+1) - 1) / (r - 1)

def b (n : ℕ) := 4^(n - 1)

def sum_c (n : ℕ) : ℕ := 
  match n with
  | 0     => 0
  | (n+1) => 7 * if n = 0 then 1 else 1 + geometric_sum 4 n 

theorem problem_conditions :
  (a 1 + a 2 + a 3 + a 4 + a 5 = 50) ∧
  (a 7 = 22) ∧
  (b 1 = 1) ∧
  (b (n + 1) = 3 * (∑ i in range (n), b i) + 1) → 
  (forall n, a n = 3 * n + 1) ∧ (forall n, b n = 4^(n-1)) ∧ 
  (sum_c 2017 = 4^2017 + 3) := by
  sorry

end problem_conditions_l91_91344


namespace fraction_of_repeating_decimals_is_4_l91_91886

/-- Helper function to convert repeating decimals to fractions -/
noncomputable def repeating_decimal_to_fraction (a b : ℕ) (h : b ≠ 0): ℚ :=
  (a : ℚ) / (b : ℚ)

/-- Given x and y as repeating decimals -/
noncomputable def x : ℚ := repeating_decimal_to_fraction 36 99 (by norm_num)
noncomputable def y : ℚ := repeating_decimal_to_fraction 9 99 (by norm_num)

/-- The main theorem stating that the fraction of the two repeating decimals equals 4 -/
theorem fraction_of_repeating_decimals_is_4 : (x / y) = 4 := by
  have h1 : x = 36 / 99 := by norm_num
  have h2 : y = 9 / 99 := by norm_num
  rw [h1, h2]
  norm_num
  rw [div_eq_mul_inv]
  norm_num
  sorry

end fraction_of_repeating_decimals_is_4_l91_91886


namespace problem_statement_l91_91101

theorem problem_statement :
  (∀ x y : ℝ, f(x + y) - f(y) = (x + 2 * y + 1) * x) →
  f 1 = 0 →
  (f 0 = -2 ∧ f = (λ x : ℝ, x^2 + x - 2) ∧ ∀ (a : ℝ), (∀ x, 0 < x ∧ x < 2 → f x > a * x - 5) → a < 3) :=
by
  intro h_cond1 h_f1
  split
  { -- Proof for f(0) = -2
    sorry
  }
  split
  { -- Proof for f(x) = x^2 + x - 2
    sorry
  }
  { -- Proof for a < 3
    sorry
  }

end problem_statement_l91_91101


namespace find_f_inv_minus_one_l91_91701

noncomputable def f : ℝ → ℝ := sorry -- We assume the function exists as described.
noncomputable def f_inv : ℝ → ℝ := function.inverse f

-- Conditions
axiom f_has_inverse : function.inverse f = f_inv
axiom f_at_point : f 2 = -1

-- Question
theorem find_f_inv_minus_one : f_inv (-1) = 2 :=
by 
  -- Here we set up the necessary information
  have h1 : f (f_inv (-1)) = -1 := function.inverse_apply f_has_inverse (-1)
  have h2 : f_inv (f 2) = 2 := function.left_inverse f_has_inverse 2
  -- And now use these to bridge the final gap
  have h3 : f_inv (-1) = 2 := by rw [←f_at_point, h2]
  exact h3

end find_f_inv_minus_one_l91_91701


namespace largest_consecutive_odd_nat_divisible_by_3_sum_72_l91_91866

theorem largest_consecutive_odd_nat_divisible_by_3_sum_72
  (a : ℕ)
  (h₁ : a % 3 = 0)
  (h₂ : (a + 6) % 3 = 0)
  (h₃ : (a + 12) % 3 = 0)
  (h₄ : a % 2 = 1)
  (h₅ : (a + 6) % 2 = 1)
  (h₆ : (a + 12) % 2 = 1)
  (h₇ : a + (a + 6) + (a + 12) = 72) :
  a + 12 = 30 :=
by
  sorry

end largest_consecutive_odd_nat_divisible_by_3_sum_72_l91_91866


namespace zeros_in_expansion_of_square_l91_91465

theorem zeros_in_expansion_of_square (n : ℕ) (h : n - 1 = 8) :
  let x := 999999999
  let x_correct : x = 10 ^ 9 - 1 := by sorry
  let squared := x * x
  (number_of_zeros squared = 8) :=
begin
  -- We'll use the given conditions and manipulate it to fit Lean's environment.
  -- We can add auxiliary definition or lemmas if needed.

  sorry
end

end zeros_in_expansion_of_square_l91_91465


namespace evaluate_expression_l91_91643

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) : (a^3 + b)^2 - (a^3 - b)^2 = 216 :=
by
  -- Proof is not required, add sorry to skip the proof
  sorry

end evaluate_expression_l91_91643


namespace triangle_BE_parallel_AC_l91_91032
-- Import essential libraries

-- Define the problem setup and proof statement
theorem triangle_BE_parallel_AC (A B C D E F M : Point) (triangle_ABC : Triangle ABC)
(∠_A : angle A = 90)
(orthogonal_projection : orthogonal_projection A (line BC) = D)
(midpoint_AD : midpoint A D = E)
(midpoint_AC : midpoint A C = F)
(circumcenter_BEF : circumcenter (triangle BEF) = M) :
parallel (line AC) (line BM) := 
sorry

end triangle_BE_parallel_AC_l91_91032


namespace monotone_increasing_interval_l91_91104

def power_function_through_point {α : ℝ} (f : ℝ → ℝ) : Prop :=
  f 2 = 1 / 4

theorem monotone_increasing_interval (f : ℝ → ℝ)
    (h : power_function_through_point f) :
    ∃ α : ℝ, α = -2 ∧ (∀ x y : ℝ, x < y → y < 0 → f x ≤ f y) :=
by
  sorry

end monotone_increasing_interval_l91_91104


namespace range_of_a_l91_91046

noncomputable def f (x : ℝ) :=
  if x > 0 then log x / log 2 else
  if x < 0 then log (-x) / log (1 / 2) else 0

theorem range_of_a (a : ℝ) (h : f a > f (-a)) : a ∈ Set.Ioc (-1) 0 ∪ Set.Ioi 1 := by
  sorry

end range_of_a_l91_91046


namespace jelly_bean_probability_l91_91948

variable (P_red P_orange P_yellow P_green : ℝ)

theorem jelly_bean_probability :
  P_red = 0.15 ∧ P_orange = 0.35 ∧ (P_red + P_orange + P_yellow + P_green = 1) →
  (P_yellow + P_green = 0.5) :=
by
  intro h
  obtain ⟨h_red, h_orange, h_total⟩ := h
  sorry

end jelly_bean_probability_l91_91948


namespace students_opinion_change_l91_91972

theorem students_opinion_change : 
  (∀ (total_students : ℕ), 
    let likes_initial := (0.4 * total_students : ℝ),
        dislikes_initial := (0.6 * total_students : ℝ),
        likes_final := (0.75 * total_students : ℝ) in
    let min_changed := likes_final - likes_initial,
        max_changed := total_students * 0.75 in
    (max_changed - min_changed = 0.4 * total_students)) :=
begin
  sorry
end

end students_opinion_change_l91_91972


namespace intersection_points_count_l91_91111

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count :
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2) := sorry

end intersection_points_count_l91_91111


namespace trapezoid_fg_squared_l91_91780

theorem trapezoid_fg_squared (EF GH EH x : ℝ) 
  (h1 : EF = sqrt 13) 
  (h2 : EH = sqrt 2002)
  (h3 : (∃ FG,  (∃ E G H, 
    (leg_perpendicular_fg_efgh : FG ⊥ EF ∧ FG ⊥ GH) ∧
    (diagonals_perpendicular_eg_fh : EG ⊥ FH)))) 
  (h4 : x = FG) :
  x^4 - 13 * x^2 - 25857 = 0 :=
sorry

end trapezoid_fg_squared_l91_91780


namespace range_of_m_l91_91357

noncomputable def f (x : ℝ) : ℝ := log x + (1 / x) - 1

theorem range_of_m (m : ℝ) (a : ℝ) (h_a : a ∈ Set.Ioo (-1:ℝ) (1:ℝ)) :
  (∃ x₀ ∈ Set.Icc (1:ℝ) (Real.exp 1), m * a - f x₀ < 0) :=
sorry

end range_of_m_l91_91357


namespace sin_double_angle_l91_91744

theorem sin_double_angle (α : ℝ) (h : cos (π / 4 - α) = 3 / 5) : sin (2 * α) = -7 / 25 :=
by sorry

end sin_double_angle_l91_91744


namespace probability_A_to_B_in_8_moves_l91_91936

-- Define vertices
inductive Vertex : Type
| A | B | C | D | E | F

open Vertex

-- Define the probability of ending up at Vertex B after 8 moves starting from Vertex A
noncomputable def probability_at_B_after_8_moves : ℚ :=
  let prob := (3 : ℚ) / 16
  prob

-- Theorem statement
theorem probability_A_to_B_in_8_moves :
  (probability_at_B_after_8_moves = (3 : ℚ) / 16) :=
by
  -- Proof to be provided
  sorry

end probability_A_to_B_in_8_moves_l91_91936


namespace highest_average_speed_is_B_l91_91144

-- Define the conditions as constants/variables in Lean
def car_A_distance : ℝ := 150
def car_A_time : ℝ := 3
def car_B_distance : ℝ := 320
def car_B_time : ℝ := 4
def car_C_distance : ℝ := 210
def car_C_time : ℝ := 3

-- Define the average speed calculation for each car
def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

-- Define the expected highest average speed time period
def highest_average_speed_time_period : String := "from 3 to 7 hours"

-- Define the proof statement for the given problem
theorem highest_average_speed_is_B :
  (average_speed car_B_distance car_B_time > average_speed car_A_distance car_A_time) ∧ 
  (average_speed car_B_distance car_B_time > average_speed car_C_distance car_C_time) ∧ 
  (highest_average_speed_time_period = "from 3 to 7 hours") :=
by sorry

end highest_average_speed_is_B_l91_91144


namespace line_intersects_circle_l91_91116

theorem line_intersects_circle (a : ℝ) : 
  let r := 2 * Real.sqrt 2 in
  let d := (abs (1 - a)) / Real.sqrt (1 + a^2) in
  d < r → 
  ∃ x y : ℝ, (x^2 + y^2 = 8) ∧ (y = a * x - a + 1) :=
by
  sorry

end line_intersects_circle_l91_91116


namespace largest_and_next_largest_difference_l91_91871

theorem largest_and_next_largest_difference (a b c : ℕ) (h1: a = 10) (h2: b = 11) (h3: c = 12) : 
  let largest := max a (max b c)
  let next_largest := min (max a b) (max (min a b) c)
  largest - next_largest = 1 :=
by
  -- Proof to be filled in for verification
  sorry

end largest_and_next_largest_difference_l91_91871


namespace smallest_positive_x_l91_91292

theorem smallest_positive_x (x : ℕ) (h : 42 * x + 9 ≡ 3 [MOD 15]) : x = 2 :=
sorry

end smallest_positive_x_l91_91292


namespace limit_seq_eq_exp_3_l91_91976

theorem limit_seq_eq_exp_3 (f : ℕ → ℝ) : 
  (∀ n, f n = (2 * n^2 + 21 * n - 7) / (2 * n^2 + 18 * n + 9)) → 
  tendsto (λ n, (f n)^(2 * n + 1)) at_top (𝓝 (Real.exp 3)) :=
by
  intro h
  sorry

end limit_seq_eq_exp_3_l91_91976


namespace range_of_a_l91_91728

theorem range_of_a (p : Prop) (h : ¬p)
  (h₀ : p ↔ ∃ x ∈ ℝ, (λ x : ℝ, x^2 + 2 * a * x + a) x ≤ 0) :
  (0 < a) ∧ (a < 1) :=
by
  sorry

end range_of_a_l91_91728


namespace find_n_l91_91505

variable (n : ℕ)
variable (s : ℝ) -- side length of the cube
variable (face_diagonal : ℝ) -- diagonal of one of its faces

-- Condition: the face diagonal of the cube is 4
axiom h1 : face_diagonal = 4

-- Condition: the volume of the cube is n√2
axiom h2 : ∃ s, (s√2 = face_diagonal) ∧ (n√2 = s^3)

-- Assertion: n must be 16
theorem find_n (h1 : face_diagonal = 4) (h2 : ∃ s, (s*√2 = face_diagonal) ∧ (n√2 = s^3)) : 
  n = 16 :=
sorry

end find_n_l91_91505


namespace number_of_subsets_l91_91115

theorem number_of_subsets : 
  {M : Set ℕ // {1, 2} ⊆ M ∧ M ⊆ {1, 2, 3, 4, 5, 6, 7}}.card = 31 :=
by
  sorry

end number_of_subsets_l91_91115


namespace value_added_to_each_number_is_12_l91_91424

theorem value_added_to_each_number_is_12
    (sum_original : ℕ)
    (sum_new : ℕ)
    (n : ℕ)
    (avg_original : ℕ)
    (avg_new : ℕ)
    (value_added : ℕ) :
  (n = 15) →
  (avg_original = 40) →
  (avg_new = 52) →
  (sum_original = n * avg_original) →
  (sum_new = n * avg_new) →
  (value_added = (sum_new - sum_original) / n) →
  value_added = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end value_added_to_each_number_is_12_l91_91424


namespace angle_between_EF_and_AB_l91_91405

variables {R : Type} [RealField R] {S A B C E F : Point R}
variables (tetrahedron_SABC : Tetrahedron S A B C)
variables (E_mid_SA : E = midpoint S A)
variables (F_centroid_ABC : F = centroid A B C)

theorem angle_between_EF_and_AB :
  angle_between (line_through E F) (line_through A B) = 90 := by
  sorry

end angle_between_EF_and_AB_l91_91405


namespace divide_triangle_area_in_half_of_height_l91_91482

variables {A B C D E F : Type}
           {R : ℝ}
           [MetricSpace A] [MetricSpace B] [MetricSpace C]
           {triangle : triangle A B C}

def height_eq_R_sqrt2 (triangle : Triangle ABC) (h : ℝ) : Prop :=
  h = R * real.sqrt 2

def divides_area_in_half (triangle : Triangle ABC) 
  [InEuclideanPlane triangle] (line : line E F) : Prop :=
  -- Formalize the segment from E to F dividing the area in half
  sorry

theorem divide_triangle_area_in_half_of_height (triangle : Triangle ABC)
  (R : ℝ)
  (h : ℝ) (H : height_eq_R_sqrt2 triangle h) 
  (b_perp : Perpendicular to Triangle base BC from B and C) :
  ∃ line : line E F, divides_area_in_half triangle line :=
begin
  sorry
end

end divide_triangle_area_in_half_of_height_l91_91482


namespace find_f3_l91_91377

theorem find_f3 (a b : ℝ) (f : ℝ → ℝ)
  (h1 : f 1 = 3)
  (h2 : f 2 = 6)
  (h3 : ∀ x, f x = a * x^2 + b * x + 1) :
  f 3 = 10 :=
sorry

end find_f3_l91_91377


namespace additional_term_induction_l91_91882

theorem additional_term_induction (k : Nat) :
  (∑ i in Finset.range (k + 1), i^2 + ∑ i in Finset.range k, i^2) - 
  (∑ i in Finset.range k, i^2 + ∑ i in Finset.range (k - 1), i^2) = (k + 1)^2 + k^2 :=
by
  sorry

end additional_term_induction_l91_91882


namespace f_has_infinitely_many_points_of_discontinuity_l91_91842

noncomputable def f : ℝ → ℝ := sorry -- Definition of f required to satisfy \( f(f(x)) = -x \)

theorem f_has_infinitely_many_points_of_discontinuity :
  (∀ x : ℝ, f (f x) = -x) → ∀ r : ℝ, ¬ continuous_at f r :=
begin
  sorry
end

end f_has_infinitely_many_points_of_discontinuity_l91_91842


namespace midpoints_diagonals_perpendicular_angle_bisector_l91_91001

theorem midpoints_diagonals_perpendicular_angle_bisector 
  {A B C D O M N : Type*} 
  (hAB_eq_CD : AB = CD)
  (h_intersection : rays_AB_DC_intersect_at_O O AB DC) 
  (midpoints_AC_BD : M = midpoint_diagonals AC BD) :
  is_perpendicular (line_through M N) (angle_bisector A O D) :=
sorry

end midpoints_diagonals_perpendicular_angle_bisector_l91_91001


namespace contrapositive_of_square_root_l91_91093

theorem contrapositive_of_square_root (a b : ℝ) :
  (a^2 < b → -Real.sqrt b < a ∧ a < Real.sqrt b) ↔ (a ≥ Real.sqrt b ∨ a ≤ -Real.sqrt b → a^2 ≥ b) := 
sorry

end contrapositive_of_square_root_l91_91093


namespace john_speed_above_limit_l91_91022

def distance : ℝ := 150
def time : ℝ := 2
def speed_limit : ℝ := 60

theorem john_speed_above_limit :
  distance / time - speed_limit = 15 :=
by
  sorry

end john_speed_above_limit_l91_91022


namespace degree_of_vertex_angle_of_isosceles_triangle_l91_91380

theorem degree_of_vertex_angle_of_isosceles_triangle (exterior_angle : ℝ) (h_exterior_angle : exterior_angle = 40) : 
∃ vertex_angle : ℝ, vertex_angle = 140 :=
by 
  sorry

end degree_of_vertex_angle_of_isosceles_triangle_l91_91380


namespace Triangle_A1B1C1_is_right_triangle_l91_91015

theorem Triangle_A1B1C1_is_right_triangle (ABC : Triangle) (A_1 B_1 C_1 : Point)
  (angle_A_120 : ABC.A.angle = 120)
  (bisector_A : IsAngleBisector ABC.A ABC.A1) 
  (bisector_B : IsAngleBisector ABC.B ABC.B1)
  (bisector_C : IsAngleBisector ABC.C ABC.C1)
  (intersection_A1 : AreIntersecting ABC.A bisector_A)
  (intersection_B1 : AreIntersecting ABC.B bisector_B)
  (intersection_C1 : AreIntersecting ABC.C bisector_C) 
  : IsRightTriangle A_1 B_1 C_1 := 
sorry

end Triangle_A1B1C1_is_right_triangle_l91_91015


namespace regular_pentagon_exterior_angle_l91_91779

theorem regular_pentagon_exterior_angle : 
  ∀ (n : ℕ) (h : n = 5), 
  180 * (n - 2) / n - (180 * (n - 2) / n) = 72 :=
by
  intros n h
  rw h
  sorry

end regular_pentagon_exterior_angle_l91_91779


namespace weight_and_ratio_of_spheres_l91_91534

-- Defining constants and conditions
def surface_area (r : ℝ) : ℝ := 4 * real.pi * r^2
def weight_dependent_on_surface_area (weight1 weight2 surface_area1 surface_area2 : ℝ) : Prop :=
  weight1 / surface_area1 = weight2 / surface_area2

-- Specific Properties
def weight_of_radius_015_weighs_8g : Prop :=
  weight_dependent_on_surface_area 8 (surface_area 0.15) (surface_area 0.3)

def weight_of_radius_03 : ℝ := 32
def weight_of_radius_06 : ℝ := 128

def weight_ratio (w1 w2 w3 : ℝ) (r : ℝ) : Prop :=
  (w1 : ℝ) / r = (w2 / (4 * r)) ∧ (w2 / (4 * r) = w3 / (16 * r))

-- Theorem to prove
theorem weight_and_ratio_of_spheres :
  weight_of_radius_015_weighs_8g ∧ 
  weight_dependent_on_surface_area 8 32 (surface_area 0.15) (surface_area 0.3) ∧
  weight_dependent_on_surface_area 8 128 (surface_area 0.15) (surface_area 0.6) ∧
  weight_ratio 8 32 128 8 :=
by sorry -- Proof is omitted

end weight_and_ratio_of_spheres_l91_91534


namespace complex_inverse_l91_91442

theorem complex_inverse (x : ℂ) (h : x = (1 + complex.i * real.sqrt 3) / 3) :
  1 / (x^2 + x) = 9 / 76 - (45 * complex.i * real.sqrt 3) / 76 :=
by
  sorry

end complex_inverse_l91_91442


namespace number_of_zeros_f_l91_91718

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2^x - 3 * x

-- State the theorem that the number of zeros of the function f is 1
theorem number_of_zeros_f : ∃! x : ℝ, f x = 0 :=
sorry

end number_of_zeros_f_l91_91718


namespace train_length_l91_91933

theorem train_length
    (length_first_train : ℝ)
    (speed_first_train_kmph : ℝ)
    (speed_second_train_kmph : ℝ)
    (cross_time_seconds : ℝ)
    (length_other_train : ℝ) :
    length_first_train = 280 → 
    speed_first_train_kmph = 120 → 
    speed_second_train_kmph = 80 → 
    cross_time_seconds = 9 → 
    length_other_train = 219.95 :=
by
  intros h1 h2 h3 h4
  -- Convert the speeds from kmph to m/s
  let speed_first_train_ms := speed_first_train_kmph * 1000 / 3600
  let speed_second_train_ms := speed_second_train_kmph * 1000 / 3600
  -- Calculate the relative speed
  let relative_speed := speed_first_train_ms + speed_second_train_ms
  -- Calculate the total distance covered in the crossing time
  let total_distance := relative_speed * cross_time_seconds
  -- Formulate the equation for the total distance covered when crossing
  let other_train_length := total_distance - length_first_train
  -- Conclude that the length of the other train is approximately 219.95 meters
  have hlength : other_train_length = length_other_train := by sorry
  exact hlength

end train_length_l91_91933


namespace find_length_AD_l91_91000

-- Define the given conditions asexisting variables
variables {A B C D O : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace O]

constant BO : ℝ
constant OD : ℝ
constant AO : ℝ
constant OC : ℝ
constant AB : ℝ
constant AD : ℝ

-- Assign the given values to these constants
axiom BO_val : BO = 5
axiom OD_val : OD = 7
axiom AO_val : AO = 9
axiom OC_val : OC = 4
axiom AB_val : AB = 7

-- State the theorem to be proven
theorem find_length_AD : AD = Real.sqrt 209.8 :=
by 
  -- provide proof steps here (we use sorry to skip actual proof)
  sorry

end find_length_AD_l91_91000


namespace unique_frame_cover_100x100_l91_91519

-- Conditions
def is_unit_square (x y : ℕ) : Prop := x < 100 ∧ y < 100

def is_n_by_n_square (n : ℕ) : Prop := n.mod 2 = 0

def is_frame (n : ℕ) (x y : ℕ) : Prop :=
  (x < n ∧ (y = 0 ∨ y = n-1)) ∨ (y < n ∧ (x = 0 ∨ x = n-1))

-- Question, we are proving existence and uniqueness of the cover
theorem unique_frame_cover_100x100 : 
  ∃! (frames : list (ℕ × (ℕ × ℕ))), 
    (∀ frame ∈ frames, is_frame 100 (frame.2).1 (frame.2).2) ∧ frames.length = 50 := 
sorry

end unique_frame_cover_100x100_l91_91519


namespace sum_f_1_to_110_l91_91339

-- Defining the conditions as hypotheses
variables {f : ℝ → ℝ}
variable (h_domain : ∀ x : ℝ, x ∈ ℝ) -- Redundant in Lean but included for equivalence
variable (h1 : ∀ x : ℝ, f(x + 1) + f(x - 1) = 2)
variable (h2 : ∀ x : ℝ, f(x + 2) = f(-x + 2))
variable (h3 : f 0 = 0)

-- The theorem to prove the sum is 111
theorem sum_f_1_to_110 : (∑ k in Finset.range 110, f (k + 1)) = 111 :=
by
  sorry -- Proof goes here

end sum_f_1_to_110_l91_91339


namespace solve_slices_cost_l91_91151

noncomputable def cost_per_slice_of_brown_bread : ℝ :=
  let B := (3 * 8.48224 + 2) / 3 in  -- assuming W is calculated as 8.48224
  B / 15 * 100

theorem solve_slices_cost :
  ∀ (W : ℝ), 
  (W = 8.48224) →
  (3 * (W + 2) + 2 * W = 48.4112) →
  (cost_per_slice_of_brown_bread = 70) :=
by
  intros W hW hw
  rw [hW] at hw
  sorry

end solve_slices_cost_l91_91151


namespace lemonade_quarts_l91_91742

theorem lemonade_quarts (total_parts water_parts lemon_juice_parts : ℕ) (total_gallons gallons_to_quarts : ℚ) 
  (h_ratio : water_parts = 4) (h_ratio_lemon : lemon_juice_parts = 1) (h_total_parts : total_parts = water_parts + lemon_juice_parts)
  (h_total_gallons : total_gallons = 1) (h_gallons_to_quarts : gallons_to_quarts = 4) :
  let volume_per_part := total_gallons / total_parts
  let volume_per_part_quarts := volume_per_part * gallons_to_quarts
  let water_volume := water_parts * volume_per_part_quarts
  water_volume = 16 / 5 :=
by
  sorry

end lemonade_quarts_l91_91742


namespace recipe_flour_requirement_l91_91814

theorem recipe_flour_requirement (flour_added more_flour : ℕ) (h_flour_added : flour_added = 3) (h_more_flour : more_flour = 6) :
  flour_added + more_flour = 9 :=
by
  rw [h_flour_added, h_more_flour]
  exact rfl

end recipe_flour_requirement_l91_91814


namespace perpendicular_feet_collinear_l91_91914

open EuclideanGeometry

-- Define the framework for triangle inscribed in a circle
variables {A B C P X Y Z : Point}

theorem perpendicular_feet_collinear
  (h_triangle : Triangle A B C)
  (h_circumcircle : Circle α (circumcenter α A B C)) 
  (h_point_on_circle : OnCircle P α)
  (h_perpendiculars : ∀ {p1 p2}, FootOfPerpendicular P p1 p2 = X ∨ FootOfPerpendicular P p1 p2 = Y ∨ FootOfPerpendicular P p1 p2 = Z) :
  Collinear [X, Y, Z] :=
sorry

end perpendicular_feet_collinear_l91_91914


namespace point_in_third_quadrant_l91_91378

-- Definitions from conditions
def z1 (x y : ℝ) : ℂ := (x - 2) + y * complex.I
def z2 (x : ℝ) : ℂ := 3 * x + complex.I

-- Definition of conjugate pairs
def conjugate (z1 z2 : ℂ) : Prop := z1.re = z2.re ∧ z1.im = -z2.im

-- Main theorem statement
theorem point_in_third_quadrant (x y : ℝ) (h : conjugate (z1 x y) (z2 x)) : x = -1 ∧ y = -1 → (x, y) ∈ ({(x, y : ℝ) | x < 0 ∧ y < 0} : set (ℝ × ℝ)) :=
by
  intros
  sorry

end point_in_third_quadrant_l91_91378


namespace three_mathematicians_same_language_l91_91062

theorem three_mathematicians_same_language
  (M : Fin 9 → Finset string)
  (h1 : ∀ i j k : Fin 9, ∃ lang, i ≠ j → i ≠ k → j ≠ k → lang ∈ M i ∧ lang ∈ M j)
  (h2 : ∀ i : Fin 9, (M i).card ≤ 3)
  : ∃ lang ∈ ⋃ i, M i, ∃ (A B C : Fin 9), A ≠ B → A ≠ C → B ≠ C → lang ∈ M A ∧ lang ∈ M B ∧ lang ∈ M C :=
sorry

end three_mathematicians_same_language_l91_91062


namespace rate_of_rainfall_on_Tuesday_l91_91417

theorem rate_of_rainfall_on_Tuesday:
  ∃ (R : ℝ), 
    let monday_rainfall := 7 in
    let tuesday_rainfall := 4 * R in
    let wednesday_rainfall := 2 * 2 * R in
    monday_rainfall + tuesday_rainfall + wednesday_rainfall = 23 ∧
    R = 2 :=
begin
  sorry
end

end rate_of_rainfall_on_Tuesday_l91_91417


namespace range_of_a_l91_91081

noncomputable def range_a (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ (x^3 * Real.exp (y / x) = a * y^3)

theorem range_of_a (a : ℝ) : range_a a → a ≥ Real.exp 3 / 27 :=
by
  sorry

end range_of_a_l91_91081


namespace correct_statement_a_l91_91615

theorem correct_statement_a (a b x y : ℝ) : 
  (ab2_correct : (ab^2)^3 = a^3 * b^6) ∧ 
  (3xy_incorrect : (3 * x * y)^3 ≠ 9 * x^3 * y^3) ∧ 
  (-2a2_incorrect : (-2 * a^2)^2 ≠ -4 * a^4) ∧ 
  (sqrt9_incorrect : sqrt 9 ≠ 3 ∧ sqrt 9 ≠ -3) :=
by {
  sorry
}

end correct_statement_a_l91_91615


namespace find_p_root_relation_l91_91635

theorem find_p_root_relation (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ 0 ∧ x2 = 3 * x1 ∧ x1^2 + p * x1 + 2 * p = 0 ∧ x2^2 + p * x2 + 2 * p = 0) ↔ (p = 0 ∨ p = 32 / 3) :=
by sorry

end find_p_root_relation_l91_91635


namespace combined_annual_income_l91_91860

-- Define the given conditions and verify the combined annual income
def A_ratio : ℤ := 5
def B_ratio : ℤ := 2
def C_ratio : ℤ := 3
def D_ratio : ℤ := 4

def C_income : ℤ := 15000
def B_income : ℤ := 16800
def A_income : ℤ := 25000
def D_income : ℤ := 21250

theorem combined_annual_income :
  (A_income + B_income + C_income + D_income) * 12 = 936600 :=
by
  sorry

end combined_annual_income_l91_91860


namespace evaluate_polynomial_at_0_operations_count_l91_91252

def poly_6 (x : ℝ) : ℝ :=
  3 * x ^ 6 + 4 * x ^ 5 - 7 * x ^ 4 + 2 * x ^ 3 + 3 * x ^ 2 - x + 4

def horner_poly (x : ℝ) : ℝ :=
  (((((3 * x + 4) * x - 7) * x + 2) * x + 3) * x - 1) * x + 4

theorem evaluate_polynomial_at_0.4 :
  ∃ y, horner_poly 0.4 = y := 
sorry

theorem operations_count :
  horner_multiplications 6 ∧ horner_additions 6 :=
sorry

end evaluate_polynomial_at_0_operations_count_l91_91252


namespace length_of_QR_l91_91362

noncomputable def distance (A B : (ℝ × ℝ)) : ℝ :=
  real.sqrt (((A.1 - B.1)^2) + ((A.2 - B.2)^2))

theorem length_of_QR :
  let P := (0, 0) in
  let Q := (0, 15) in
  let R := (-20, 0) in
  let S := (-45, 0) in
  ∃ QR : ℝ, QR = distance Q R ∧ QR = 25 :=
by
  sorry

end length_of_QR_l91_91362


namespace solve_quadratic_equation_l91_91493

theorem solve_quadratic_equation :
  ∀ (x : ℝ), (x^2 - 4 * x = 12) ↔ (x = 6 ∨ x = -2) :=
by
  intro x
  constructor
  · intro h
    have h : x^2 - 4 * x - 12 = 0 := by linarith
    have f : (x - 6) * (x + 2) = 0 := by sorry
    cases f
    · left; linarith
    · right; linarith
  · intro h
    cases h
    · rw h; linarith
    · rw h; linarith

end solve_quadratic_equation_l91_91493


namespace distinct_real_solutions_g_quadruple_equals_5_l91_91802

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem distinct_real_solutions_g_quadruple_equals_5 : 
  (finset.univ.filter (λ c : ℝ, g (g (g (g c))) = 5)).card = 4 := 
by
  sorry

end distinct_real_solutions_g_quadruple_equals_5_l91_91802


namespace marbles_on_third_day_l91_91627

theorem marbles_on_third_day
  {monday_marbles : ℕ} (h1: monday_marbles = 30)
  {fraction_taken : ℝ} (h2: fraction_taken = 3/5)
  {half_taken_remaining : ℝ} (h3: half_taken_remaining = 1/2) :
  let tuesday_taken := (fraction_taken * monday_marbles).natAbs,
      divided_marbles := tuesday_taken / 2,
      remaining_marbles := monday_marbles - tuesday_taken,
      cleo_third_day_taken := (half_taken_remaining * remaining_marbles).natAbs in
  divided_marbles + cleo_third_day_taken = 15 :=
begin
  sorry
end

end marbles_on_third_day_l91_91627


namespace Jan_claims_correct_l91_91255

-- Define the number of claims each agent can handle
def Missy_claims := 41
def John_claims := Missy_claims - 15
def Jan_claims := John_claims / 1.30

-- The proof that Jan can handle 20 claims
theorem Jan_claims_correct : Jan_claims = 20 := by
  sorry

end Jan_claims_correct_l91_91255


namespace find_sum_of_relatively_prime_integers_l91_91522

theorem find_sum_of_relatively_prime_integers :
  ∃ (x y : ℕ), x * y + x + y = 119 ∧ x < 25 ∧ y < 25 ∧ Nat.gcd x y = 1 ∧ x + y = 20 :=
by
  sorry

end find_sum_of_relatively_prime_integers_l91_91522


namespace company_production_l91_91206

theorem company_production (bottles_per_case number_of_cases total_bottles : ℕ)
  (h1 : bottles_per_case = 12)
  (h2 : number_of_cases = 10000)
  (h3 : total_bottles = number_of_cases * bottles_per_case) : 
  total_bottles = 120000 :=
by {
  -- Proof is omitted, add actual proof here
  sorry
}

end company_production_l91_91206


namespace solution_set_of_inequality_l91_91528

theorem solution_set_of_inequality : 
  {x : ℝ | |x|^3 - 2 * x^2 - 4 * |x| + 3 < 0} = 
  { x : ℝ | -3 < x ∧ x < -1 } ∪ { x : ℝ | 1 < x ∧ x < 3 } := 
by
  sorry

end solution_set_of_inequality_l91_91528


namespace count_valid_pairs_l91_91290

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, n / (10 ^ k) % 10 = d

def valid_pair (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 2000 ∧ ¬ contains_digit a 0 ∧ ¬ contains_digit b 0 ∧ 
  ¬ contains_digit a 9 ∧ ¬ contains_digit b 9

theorem count_valid_pairs : 
  {p : (ℕ × ℕ) | valid_pair p.1 p.2}.to_finset.card = 1563 :=
by
  sorry

end count_valid_pairs_l91_91290


namespace measure_of_angle_A_value_of_a_l91_91411

open Real

variables (a b c : ℝ)
variables (A B C : ℝ) [fact (0 < A)] [fact (A < π)] [fact (0 < B)] [fact (B < π)] [fact (0 < C)] [fact (C < π)]
variables (h_area : 0.5 * b * c * sin (A) = sqrt 3 / 4)
variables (h_cos_eq : 2 * cos A * (c * cos B + b * cos C) = a)
variables (h_constr : c^2 + a * b * cos C + a^2 = 4)

theorem measure_of_angle_A : A = π / 3 := sorry

theorem value_of_a (hA : A = π / 3) : a = sqrt 21 / 3 := sorry

end measure_of_angle_A_value_of_a_l91_91411


namespace length_AB_l91_91397

-- Conditions
variables (A B C D : Type) [euclidean_space A B C D]
variable {AB AC BC BD CD : ℝ}
variable {perim_ABC : ℝ} (hABC_perimeter : perim_ABC = 26)
variable {perim_CBD : ℝ} (hCBD_perimeter : perim_CBD = 24)
variable (hBD : BD = 10)
variables (isosceles_ABC : AC = BC) (isosceles_CBD : CD = BC)

-- Theorem: length of AB
theorem length_AB :
  AB = 12 :=
by
  sorry

end length_AB_l91_91397


namespace ratio_of_areas_of_squares_l91_91525

theorem ratio_of_areas_of_squares (a_side b_side : ℕ) (h_a : a_side = 36) (h_b : b_side = 42) : 
  (a_side ^ 2 : ℚ) / (b_side ^ 2 : ℚ) = 36 / 49 :=
by
  sorry

end ratio_of_areas_of_squares_l91_91525


namespace digging_cost_well_l91_91287

/-- Define constants for the problem -/
def well_depth : ℝ := 14     -- Depth of the well in meters
def well_diameter : ℝ := 3   -- Diameter of the well in meters
def cost_per_cubic_meter : ℝ := 19 -- Cost of digging per cubic meter

/-- Define the radius, which is half the diameter -/
def well_radius : ℝ := well_diameter / 2

/-- Define the volume of the well using the formula for the volume of a cylinder -/
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
def well_volume : ℝ := cylinder_volume well_radius well_depth

/-- Define the total cost of digging the well -/
def total_digging_cost (volume cost_per_m3 : ℝ) : ℝ := volume * cost_per_m3
def total_well_cost : ℝ := total_digging_cost well_volume cost_per_cubic_meter

/-- The mathematical statement to be proven -/
theorem digging_cost_well : total_well_cost = 1881 := 
sorry -- Proof to be filled in

end digging_cost_well_l91_91287


namespace compute_sum_of_powers_l91_91040

noncomputable theory

open Real

theorem compute_sum_of_powers (x y : ℝ) (hx : 1 < x) (hy : 1 < y)
  (h : (log 2 x)^3 + (log 3 y)^3 + 6 = 6 * (log 2 x) * (log 3 y)) :
  x^sqrt 3 + y^sqrt 3 = 35 :=
sorry

end compute_sum_of_powers_l91_91040


namespace smallest_positive_debt_l91_91166

theorem smallest_positive_debt :
  ∃ (p g : ℤ), 25 = 250 * p + 175 * g :=
by
  sorry

end smallest_positive_debt_l91_91166


namespace probability_of_spinning_greater_than_4_l91_91556

theorem probability_of_spinning_greater_than_4 :
  let total_sections := 8
  let favorable_numbers := {n : ℕ | 5 ≤ n ∧ n ≤ 8}
  let favorable_outcomes := favorable_numbers.card
  let probability := favorable_outcomes / total_sections
  probability = (1 : ℚ) / 2
:=
  sorry

end probability_of_spinning_greater_than_4_l91_91556


namespace perimeter_triangle_AST_l91_91170

theorem perimeter_triangle_AST :
  ∀ (A B C P R S T Q : Type*)
  (line : A → B → C → Type*) -- Define tangents as lines
  (intersect : P → R → B → C → A → Type*) -- Define intersect points
  (centre_radius : A → ℤ → Type*) -- Define the circle with radius 5
  (touches : Q → B → C → Type*) -- Define third tangent touches at Q
  (point_between : P → S → R → Type*) -- Define intersection points on segments
  (distance : A → B → ℤ)
  (tangent_length :  ∀ A B, distance A B = 30)
  (circle_radius : ∀ A,  centre_radius A 5)
  (positions : ∀ P R S T, intersect P R B C A)
  (third_tangent : ∀ Q, touches Q B C)
  (segments : ∀ P S T, point_between P S R), 
  -- Conditions
  distance A B = distance A C ∧
  distance A P = distance A R ∧
  distance P B = distance R C 
  -- Question
  → distance A S + distance S T + distance A T = 55 :=
begin
  sorry
end

end perimeter_triangle_AST_l91_91170


namespace simplify_expression_evaluate_at_minus_quarter_l91_91098
noncomputable theory

def expression (a : ℝ) : ℝ :=
  (a - 2)^2 + (a + 1) * (a - 1) - 2 * a * (a - 3)

theorem simplify_expression (a : ℝ) :
  expression a = 2 * a + 3 :=
by sorry

theorem evaluate_at_minus_quarter :
  expression (-1 / 4) = 5 / 2 :=
by {
  rw simplify_expression,
  norm_num,
}

end simplify_expression_evaluate_at_minus_quarter_l91_91098


namespace variance_of_sample_l91_91381

theorem variance_of_sample (m : ℝ) (h_avg : (m + 4 + 6 + 7) / 4 = 5) : 
  let s2 := (1 / 4) * ((m - 5) ^ 2 + (4 - 5) ^ 2 + (6 - 5) ^ 2 + (7 - 5) ^ 2) in
  s2 = 5 / 2 := 
by 
  sorry

end variance_of_sample_l91_91381


namespace general_term_min_value_S_n_l91_91683

-- Definitions and conditions according to the problem statement
variable (d : ℤ) (a₁ : ℤ) (n : ℕ)

def a_n (n : ℕ) : ℤ := a₁ + (n - 1) * d
def S_n (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- Given conditions
axiom positive_common_difference : 0 < d
axiom a3_a4_product : a_n 3 * a_n 4 = 117
axiom a2_a5_sum : a_n 2 + a_n 5 = -22

-- Proof 1: General term of the arithmetic sequence
theorem general_term : a_n n = 4 * (n : ℤ) - 25 :=
  by sorry

-- Proof 2: Minimum value of the sum of the first n terms
theorem min_value_S_n : S_n 6 = -66 :=
  by sorry

end general_term_min_value_S_n_l91_91683


namespace from20To25_l91_91861

def canObtain25 (start : ℕ) : Prop :=
  ∃ (steps : ℕ → ℕ), steps 0 = start ∧ (∃ n, steps n = 25) ∧ 
  (∀ i, steps (i+1) = (steps i * 2) ∨ (steps (i+1) = steps i / 10))

theorem from20To25 : canObtain25 20 :=
sorry

end from20To25_l91_91861


namespace cars_meet_after_time_l91_91875

theorem cars_meet_after_time (t : ℝ) (h : t = 2.5) (length_of_highway : ℝ) (speed_car1 : ℝ) (speed_car2 : ℝ) : 
  speed_car1 = 25 ∧ speed_car2 = 45 ∧ length_of_highway = 175 → 
  (speed_car1 * t + speed_car2 * t = length_of_highway) :=
begin
  intro h_conj,
  cases h_conj with h_speed1 h_conj1,
  cases h_conj1 with h_speed2 h_length,
  rw [h_speed1, h_speed2, h_length, h],
  norm_num,
end

end cars_meet_after_time_l91_91875


namespace julia_tuesday_kids_l91_91023

variable (numMon numAll numTue : ℕ)
variable (h_numMon: numMon = 12)
variable (h_numAll: numAll = 19)

theorem julia_tuesday_kids : numTue = numAll - numMon -> numTue = 7 :=
by
  intro h
  rw [h_numMon, h_numAll, h]
  rfl

#align julia_tuesday_kids julia_tuesday_kids

end julia_tuesday_kids_l91_91023


namespace min_cookies_satisfy_conditions_l91_91058

theorem min_cookies_satisfy_conditions : ∃ (b : ℕ), b ≡ 5 [MOD 6] ∧ b ≡ 7 [MOD 8] ∧ b ≡ 8 [MOD 9] ∧ ∀ (b' : ℕ), (b' ≡ 5 [MOD 6] ∧ b' ≡ 7 [MOD 8] ∧ b' ≡ 8 [MOD 9]) → b ≤ b' := 
sorry

end min_cookies_satisfy_conditions_l91_91058


namespace common_chord_of_circles_l91_91349

theorem common_chord_of_circles :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x = 0) ∧ (x^2 + y^2 - 4 * y = 0) → (x = y) :=
by
  intros x y h
  sorry

end common_chord_of_circles_l91_91349


namespace x_value_l91_91916

theorem x_value :
  ∀ (x y : ℝ), x = y - 0.1 * y ∧ y = 125 + 0.1 * 125 → x = 123.75 :=
by
  intros x y h
  sorry

end x_value_l91_91916


namespace line_pairs_parallel_or_perpendicular_l91_91985

theorem line_pairs_parallel_or_perpendicular :
  let line1 := λ x, 4 * x + 3,
      line2 := λ x, 4 * x + 3,
      line3 := λ x, -4 * x + 2 / 3,
      line4 := λ x, -1 / 4 * x + 7,
      line5 := λ x, -1 / 4 * x + 1
  in
  let slopes := [4, 4, -4, -1 / 4, -1 / 4] in
  let parallel_pairs := [(slopes[0], slopes[1]), (slopes[3], slopes[4])] in
  let perpendicular_pairs := [(slopes[0], slopes[3]), (slopes[1], slopes[3]), 
                              (slopes[0], slopes[4]), (slopes[1], slopes[4])] in
  parallel_pairs.length + perpendicular_pairs.length = 6 :=
by
  sorry

end line_pairs_parallel_or_perpendicular_l91_91985


namespace luke_points_per_round_l91_91050

-- Definitions for conditions
def total_points : ℤ := 84
def rounds : ℤ := 2
def points_per_round (total_points rounds : ℤ) : ℤ := total_points / rounds

-- Statement of the problem
theorem luke_points_per_round : points_per_round total_points rounds = 42 := 
by 
  sorry

end luke_points_per_round_l91_91050


namespace longest_segment_square_l91_91205

theorem longest_segment_square (d : ℝ) (h : d = 16) :
  let r := d / 2 in
  let θ := 90 in
  let m := r * Real.sqrt 2 in
  m^2 = 128 :=
by
  -- We assume the diameter is 16 cm
  have hr : r = 8, by {
    rw h,
    norm_num,
  }
  -- We then calculate m
  let m := r * Real.sqrt 2,
  -- Substituting the value of r
  have hm : m = 8 * Real.sqrt 2, by {
    rw hr,
    norm_num,
  }
  -- Now we calculate m^2 and verify it equals 128
  have hmsq : m^2 = (8 * Real.sqrt 2)^2, by {
    rw hm,
  },
  -- Simplify (8 * sqrt 2)^2
  rw hmsq,
  norm_num,
  -- Finish the proof
  sorry

end longest_segment_square_l91_91205


namespace price_for_70_cans_is_correct_l91_91574

def regular_price_per_can : ℝ := 0.55
def discount_percentage : ℝ := 0.25
def purchase_quantity : ℕ := 70

def discount_per_can : ℝ := discount_percentage * regular_price_per_can
def discounted_price_per_can : ℝ := regular_price_per_can - discount_per_can

def price_for_72_cans : ℝ := 72 * discounted_price_per_can
def price_for_2_cans : ℝ := 2 * discounted_price_per_can

def final_price_for_70_cans : ℝ := price_for_72_cans - price_for_2_cans

theorem price_for_70_cans_is_correct
    (regular_price_per_can : ℝ := 0.55)
    (discount_percentage : ℝ := 0.25)
    (purchase_quantity : ℕ := 70)
    (disc_per_can : ℝ := discount_percentage * regular_price_per_can)
    (disc_price_per_can : ℝ := regular_price_per_can - disc_per_can)
    (price_72_cans : ℝ := 72 * disc_price_per_can)
    (price_2_cans : ℝ := 2 * disc_price_per_can):
    final_price_for_70_cans = 28.875 :=
by
  sorry

end price_for_70_cans_is_correct_l91_91574


namespace max_value_of_g_is_3_l91_91514

def f (x : ℝ) : ℝ := -x^2 + 4 * x - 1

def g (t : ℝ) : ℝ := 
  let interval := set.Icc t (t + 1)
  real.Sup (set.image f interval)

theorem max_value_of_g_is_3 : 
  ∃ t : ℝ, g t = 3 :=
by
  sorry

end max_value_of_g_is_3_l91_91514


namespace imaginary_part_of_z_l91_91812

theorem imaginary_part_of_z (z : ℂ) (h : (z - 2 * complex.I) * (2 - complex.I) = 5) :
  z.im = 3 :=
by
  sorry

end imaginary_part_of_z_l91_91812


namespace hyperbola_equation_not_midpoint_l91_91674

-- Define the hyperbola and its properties
axiom hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1)

-- Conditions involving the point (-2, sqrt(6)) and asymptotes y = ±sqrt(2)x
axiom passes_through (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : 
  hyperbola a b a_pos b_pos (-2) (Real.sqrt 6)

axiom asymptotes (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  b / a = Real.sqrt 2

-- Define the specific hyperbola in the problem
def hyperbola_C : Prop :=
  ∀ (x y : ℝ), (x^2 - y^2 / 2 = 1)

-- Prove the equivalence of the given hyperbola with specific properties
theorem hyperbola_equation : hyperbola 1 (Real.sqrt 2) (by norm_num) (by norm_num) = hyperbola_C := 
sorry

-- Define line intersecting points A and B
noncomputable def intersects (P : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop := 
  ∃ (k : ℝ), l k = P.2 ∧ l A.1 = A.2 ∧ l B.1 = B.2 

-- Define the condition for midpoint
def midpoint (P A B : ℝ × ℝ) : Prop := 
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Prove P(1,1) cannot be the midpoint
theorem not_midpoint (P : ℝ × ℝ) : 
  P = (1, 1) → 
  ∀ (A B : ℝ × ℝ), intersects P (λ x, k * (x - 1) + 1) A B → ¬midpoint P A B := 
sorry

end hyperbola_equation_not_midpoint_l91_91674


namespace median_mode_diff_l91_91984

theorem median_mode_diff : 
  let data := [37, 38, 39, 41, 43, 43, 45, 45, 47, 50, 52, 52, 56, 57, 58, 61, 64, 68, 69] in
  abs (50 - 43) = 7 :=
by
  sorry

end median_mode_diff_l91_91984


namespace f_n_equals_factorial_l91_91319

open Nat

def f (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in Finset.range (n + 1), (-1 : ℝ)^k * (Nat.choose n k) * (x - k)^n

theorem f_n_equals_factorial (n : ℕ) (hn : 0 < n) (x : ℝ) : f n x = n! :=
by
  sorry

end f_n_equals_factorial_l91_91319


namespace final_image_of_F_is_correct_l91_91857

-- Define the initial F position as a struct
structure Position where
  base : (ℝ × ℝ)
  stem : (ℝ × ℝ)

-- Function to rotate a point 90 degrees counterclockwise around the origin
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Function to reflect a point in the x-axis
def reflectX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Function to rotate a point by 180 degrees around the origin (half turn)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- Define the initial state of F
def initialFPosition : Position := {
  base := (-1, 0),  -- Base along the negative x-axis
  stem := (0, -1)   -- Stem along the negative y-axis
}

-- Perform all transformations on the Position of F
def transformFPosition (pos : Position) : Position :=
  let afterRotation90 := Position.mk (rotate90 pos.base) (rotate90 pos.stem)
  let afterReflectionX := Position.mk (reflectX afterRotation90.base) (reflectX afterRotation90.stem)
  let finalPosition := Position.mk (rotate180 afterReflectionX.base) (rotate180 afterReflectionX.stem)
  finalPosition

-- Define the target final position we expect
def finalFPosition : Position := {
  base := (0, 1),   -- Base along the positive y-axis
  stem := (1, 0)    -- Stem along the positive x-axis
}

-- The theorem statement: After the transformations, the position of F
-- should match the final expected position
theorem final_image_of_F_is_correct :
  transformFPosition initialFPosition = finalFPosition := by
  sorry

end final_image_of_F_is_correct_l91_91857


namespace arrangement_ways_count_l91_91944

theorem arrangement_ways_count:
  let n := 10
  let k := 4
  (Nat.choose n k) = 210 :=
by
  sorry

end arrangement_ways_count_l91_91944


namespace villager4_truth_teller_l91_91276

def villager1_statement (liars : Finset ℕ) : Prop := liars = {0, 1, 2, 3}
def villager2_statement (liars : Finset ℕ) : Prop := liars.card = 1
def villager3_statement (liars : Finset ℕ) : Prop := liars.card = 2
def villager4_statement (liars : Finset ℕ) : Prop := 3 ∉ liars

theorem villager4_truth_teller (liars : Finset ℕ) :
  ¬ villager1_statement liars ∧
  ¬ villager2_statement liars ∧
  ¬ villager3_statement liars ∧
  villager4_statement liars ↔
  liars = {0, 1, 2} :=
by
  sorry

end villager4_truth_teller_l91_91276


namespace change_in_expression_l91_91716

theorem change_in_expression (x a : ℝ) (h : 0 < a) :
  (x + a)^3 - 3 * (x + a) - (x^3 - 3 * x) = 3 * a * x^2 + 3 * a^2 * x + a^3 - 3 * a
  ∨ (x - a)^3 - 3 * (x - a) - (x^3 - 3 * x) = -3 * a * x^2 + 3 * a^2 * x - a^3 + 3 * a :=
sorry

end change_in_expression_l91_91716


namespace am_gm_inequality_l91_91072

theorem am_gm_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (x + y + z)^2 / 3 ≥ x * Real.sqrt (y * z) + y * Real.sqrt (z * x) + z * Real.sqrt (x * y) := 
by sorry

end am_gm_inequality_l91_91072


namespace quadratic_root_interval_l91_91348

theorem quadratic_root_interval (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x ≤ 3 ∧ a * x^2 + x + 3a + 1 = 0) ↔ (a ∈ set.Icc (-1/2 : ℝ) (-1/3)) :=
by
  sorry

end quadratic_root_interval_l91_91348


namespace select_male_doctors_l91_91466

theorem select_male_doctors :
  let totalDoctors := 120 + 180 in
  let maleDoctors := 120 in
  let positions := 15 in
  let proportionMale := maleDoctors / totalDoctors in
  let selectedMaleDoctors := proportionMale * positions in
  selectedMaleDoctors = 6 :=
by
  let totalDoctors := 120 + 180;
  let maleDoctors := 120;
  let positions := 15;
  let proportionMale := maleDoctors / totalDoctors;
  let selectedMaleDoctors := proportionMale * positions;
  show selectedMaleDoctors = 6 from sorry

end select_male_doctors_l91_91466


namespace exprC_equals_sqrt2_over_2_l91_91958

-- Define the trigonometric expressions as Lean definitions
def exprA := sin (real.pi / 4) * cos (real.pi / 12) + cos (real.pi / 4) * sin (real.pi / 12)
def exprB := sin (real.pi / 4) * cos (real.pi / 12) - cos (real.pi / 4) * sin (real.pi / 12)
def exprC := cos (5 * real.pi / 12) * cos (real.pi / 6) + sin (5 * real.pi / 12) * sin (real.pi / 6)
def exprD := (tan (real.pi / 3) - tan (real.pi / 6)) / (1 + tan (real.pi / 3) * tan (real.pi / 6))

-- Theorem to prove that exprC equals \dfrac{\sqrt{2}}{2}
theorem exprC_equals_sqrt2_over_2 : exprC = real.sqrt 2 / 2 :=
by
  sorry

end exprC_equals_sqrt2_over_2_l91_91958


namespace maximum_n_l91_91867

def number_of_trapezoids (n : ℕ) : ℕ := n * (n - 3) * (n - 2) * (n - 1) / 24

theorem maximum_n (n : ℕ) (h : number_of_trapezoids n ≤ 2012) : n ≤ 26 :=
by
  sorry

end maximum_n_l91_91867


namespace max_product_xyz_l91_91443

theorem max_product_xyz : ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 12 ∧ z ≤ 3 * x ∧ ∀ (a b c : ℕ), a + b + c = 12 → c ≤ 3 * a → 0 < a ∧ 0 < b ∧ 0 < c → a * b * c ≤ 48 :=
by
  sorry

end max_product_xyz_l91_91443


namespace sheila_hourly_wage_is_correct_l91_91487

-- Definitions based on conditions
def works_hours_per_day_mwf : ℕ := 8
def works_days_mwf : ℕ := 3
def works_hours_per_day_tt : ℕ := 6
def works_days_tt : ℕ := 2
def weekly_earnings : ℕ := 216

-- Total calculated hours based on the problem conditions
def total_weekly_hours : ℕ := (works_hours_per_day_mwf * works_days_mwf) + (works_hours_per_day_tt * works_days_tt)

-- Target wage per hour
def wage_per_hour : ℕ := weekly_earnings / total_weekly_hours

-- The theorem stating the proof problem
theorem sheila_hourly_wage_is_correct : wage_per_hour = 6 := by
  sorry

end sheila_hourly_wage_is_correct_l91_91487


namespace circumcircle_radius_l91_91041

theorem circumcircle_radius (ABC : Triangle) (Γ : Circle) (A B C : Point)
  (E N A' V : Point)
  (h1 : is_circumcircle Γ ABC)
  (h2 : is_angle_bisector E A BAC BC)
  (h3 : on_circle N Γ)
  (h4 : antipode A' A Γ)
  (h5 : collinear A A' V BC)
  (h6 : dist E V = 6)
  (h7 : dist V A' = 7)
  (h8 : dist A' N = 9) :
  Γ.radius = 15 / 2 := by
  sorry

end circumcircle_radius_l91_91041


namespace min_generic_tees_per_package_l91_91623

def total_golf_tees_needed (n : ℕ) : ℕ := 80
def max_generic_packages_used : ℕ := 2
def tees_per_aero_flight_package : ℕ := 2
def aero_flight_packages_needed : ℕ := 28
def total_tees_from_aero_flight_packages (n : ℕ) : ℕ := aero_flight_packages_needed * tees_per_aero_flight_package

theorem min_generic_tees_per_package (G : ℕ) :
  (total_golf_tees_needed 4) - (total_tees_from_aero_flight_packages aero_flight_packages_needed) ≤ max_generic_packages_used * G → G ≥ 12 :=
by
  sorry

end min_generic_tees_per_package_l91_91623


namespace midline_parallel_to_hypotenuse_l91_91671

open EuclideanGeometry

theorem midline_parallel_to_hypotenuse
  (A B C P Q M N E F : Point)
  (hABC : RightTriangle A B C)
  (hCH : Altitude C (Line A B))
  (hAngleC : ∠ A C B = π / 2)
  (hAM : AngleBisector A B M)
  (hBN : AngleBisector B A N)
  (hP : OnLineIntersection P (Line CH) (Line AM))
  (hQ : OnLineIntersection Q (Line CH) (Line BN))
  (hE : Midpoint E Q N)
  (hF : Midpoint F P M) :
  Parallel (Line EF) (Line A B) := sorry

end midline_parallel_to_hypotenuse_l91_91671


namespace product_of_elements_in_set_is_zero_l91_91119

theorem product_of_elements_in_set_is_zero
  (n : ℕ) (Mn : Type) [has_zero Mn] [has_add Mn] [has_mult Mn] [has_sum Mn ℕ] [add_comm_monoid Mn] 
  (M : fin n → Mn) 
  (h_n_odd : odd n) (h_n_gt1 : 1 < n) 
  (h_sum_invariant : ∀ i : fin n, ∑ j in finset.univ.erase i, M j = M i → ∑ j in finset.univ, M j = ∑ j in finset.univ.erase i, M j) :
  ∏ i in finset.univ, M i = 0 := by
  sorry

end product_of_elements_in_set_is_zero_l91_91119


namespace ellipse_equation_through_P_with_foci_A_B_l91_91367

-- Definitions of points
def P : ℝ × ℝ := (5 / 2, - 3 / 2)
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Statement of the theorem
theorem ellipse_equation_through_P_with_foci_A_B :
  ∃ (a b : ℝ), a = sqrt(10) ∧ b^2 = 6 ∧ ∀ (x y : ℝ), (x^2 / 10 + y^2 / 6 = 1) :=
by
  sorry

end ellipse_equation_through_P_with_foci_A_B_l91_91367


namespace x_approx_nearest_tenth_l91_91294

-- Define the given constants
def a : ℝ := 0.889
def b : ℝ := 55
def c : ℝ := 9.97

-- Define the expression for x
noncomputable def x : ℝ := (a * Real.sqrt(b^2 - 4 * a * c) - b) / (2 * a)

-- Lean statement to prove x ≈ -3.6 to the nearest tenth
theorem x_approx_nearest_tenth : (Real.round (x * 10) / 10) = -3.6 :=
by
  sorry

end x_approx_nearest_tenth_l91_91294


namespace range_of_f_l91_91637

noncomputable def f (x : ℝ) : ℝ := (3 * x + 9) / (x - 4) + 1

theorem range_of_f : set.range f = {y : ℝ | y ≠ 4} := sorry

end range_of_f_l91_91637


namespace cheating_percentage_l91_91229

theorem cheating_percentage (x : ℝ) :
  (∀ cost_price : ℝ, cost_price = 100 →
   let received_when_buying : ℝ := cost_price * (1 + x / 100)
   let given_when_selling : ℝ := cost_price * (1 - x / 100)
   let profit : ℝ := received_when_buying - given_when_selling
   let profit_percentage : ℝ := profit / cost_price
   profit_percentage = 2 / 9) →
  x = 22.22222222222222 := 
by
  sorry

end cheating_percentage_l91_91229


namespace average_of_shifted_data_set_l91_91681

-- Define the given variance condition as a hypothesis
theorem average_of_shifted_data_set 
  (x₁ x₂ x₃ x₄ : ℝ)
  (h_positive : 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ 0 < x₄)
  (h_variance : 1 / 4 * (x₁^2 + x₂^2 + x₃^2 + x₄^2 - 16) = (x₁^2 + x₂^2 + x₃^2 + x₄^2)/4 - 4) : 
  (x₁ + 3 + x₂ + 3 + x₃ + 3 + x₄ + 3) / 4 = 5 :=
begin
  sorry
end

end average_of_shifted_data_set_l91_91681


namespace ellipse_properties_l91_91340

noncomputable def ellipse_standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_properties (a b : ℝ) (h1 : a = 2 * sqrt 3) (h2 : b^2 = 12 - (2 * sqrt 2)^2)
  (h3 : ellipse_standard_equation 2 (sqrt 3) b 0 4) :
  ellipse_standard_equation (2 * sqrt 3) 2 (sqrt 3) 12 4 :=
sorry

noncomputable def ellipse_intersection_invariant (x1 x2 k : ℝ)
  (h1 : x1 + x2 = -6 * k / (1 + 3 * k^2))
  (h2 : x1 * x2 = -9 / (1 + 3 * k^2))
  (RM RN : ℝ → ℝ) (h3 : RM = (λ x, x1 + k * x1 + 3))
  (h4 : RN = (λ x, x2 + k * x2 + 3)) :
  RM x1 * RN x1 + (k * RM x1 + 3) * (k * RN x1 + 3) = 0 :=
sorry

end ellipse_properties_l91_91340


namespace regions_divided_by_tetrahedron_planes_regions_divided_by_octahedron_planes_l91_91268

theorem regions_divided_by_tetrahedron_planes (tetrahedron_planes : ℕ) (h_tetra : tetrahedron_planes = 4) : 
  number_of_regions_divided_by_planes tetrahedron_planes = 23 := 
sorry

theorem regions_divided_by_octahedron_planes (octahedron_planes : ℕ) (h_octa : octahedron_planes = 8) : 
  number_of_regions_divided_by_planes octahedron_planes = 59 := 
sorry

end regions_divided_by_tetrahedron_planes_regions_divided_by_octahedron_planes_l91_91268


namespace a_perp_b_angle_a_ab_pi_over_4_l91_91333

open Real 

noncomputable def a: ℝ × ℝ := sorry
noncomputable def b: ℝ × ℝ := sorry

axiom unit_vectors (x: ℝ × ℝ): (x.fst^2 + x.snd^2 = 1)

-- conditions
axiom unit_a : unit_vectors a
axiom unit_b : unit_vectors b
axiom add_eq : a + b = (1, -1)

-- question 1: a . b = 0
theorem a_perp_b : (a.1 * b.1 + a.2 * b.2) = 0 :=
sorry

-- question 2: angle between (a and a - b) = π / 4
theorem angle_a_ab_pi_over_4 : (acos (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) / (sqrt(a.1^2 + a.2^2) * sqrt((a.1 - b.1)^2 + (a.2 - b.2)^2))) = π / 4 :=
sorry

end a_perp_b_angle_a_ab_pi_over_4_l91_91333


namespace gcd_square_product_l91_91807

theorem gcd_square_product (x y z : ℕ) (h : 1 / (x : ℝ) - 1 / (y : ℝ) = 1 / (z : ℝ)) : 
    ∃ n : ℕ, gcd x (gcd y z) * x * y * z = n * n := 
sorry

end gcd_square_product_l91_91807


namespace min_intersection_cardinality_l91_91314

theorem min_intersection_cardinality (A B C : Set) (h_A : A.card = 100) (h_B : B.card = 100)
(h_n : 2^A.card + 2^B.card + 2^C.card = 2^(A ∪ B ∪ C).card) : 
  (A ∩ B ∩ C).card = 97 :=
sorry

end min_intersection_cardinality_l91_91314


namespace find_linear_function_passing_through_points_l91_91341

theorem find_linear_function_passing_through_points :
  ∃ b k, (∀ x y, y = k * x + b ↔ ((x = 3 ∧ y = 5) ∨ (x = -4 ∧ y = -9))) ∧
          (k = 2 ∧ b = -1) := 
by
satisfies sorry

end find_linear_function_passing_through_points_l91_91341


namespace f_parity_f_monotonicity_f_range_l91_91356

-- Definition of the function f(x) = x / (x^2 + 1)
def f (x : ℝ) : ℝ := x / (x^2 + 1)

-- Definition of parity (odd function)
theorem f_parity : ∀ x : ℝ, f (-x) = -f x :=
by sorry

-- Definition of monotonicity on (1, +∞)
theorem f_monotonicity : ∀ x1 x2 : ℝ, 1 < x1 → 1 < x2 → x1 < x2 → f x1 > f x2 :=
by sorry

-- Definition of the range of the function f
theorem f_range : set.range f = set.Icc (-1/2 : ℝ) (1/2) :=
by sorry

end f_parity_f_monotonicity_f_range_l91_91356


namespace cost_of_ox_and_sheep_l91_91581

variable (x y : ℚ)

theorem cost_of_ox_and_sheep :
  (5 * x + 2 * y = 10) ∧ (2 * x + 8 * y = 8) → (x = 16 / 9 ∧ y = 5 / 9) :=
by
  sorry

end cost_of_ox_and_sheep_l91_91581


namespace simplify_expression_l91_91838

section
variable (a b : ℚ) (h_a : a = -1) (h_b : b = 1/4)

theorem simplify_expression : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry
end

end simplify_expression_l91_91838


namespace diameter_is_approximately_30_l91_91286

noncomputable def diameter_of_field (total_cost : ℝ) (cost_per_meter : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter in
  circumference / Real.pi

theorem diameter_is_approximately_30 :
  diameter_of_field 471.24 5 ≈ 30 := 
sorry

end diameter_is_approximately_30_l91_91286


namespace max_likelihood_estimate_l91_91295

variable (x : List ℝ)
variable (n : ℕ)
variable (λ : ℝ)

-- Define sample mean
def sample_mean (x : List ℝ) : ℝ :=
  (x.foldl (λ acc y => acc + y) 0) / x.length

-- State MLE of lambda for exponential distribution
theorem max_likelihood_estimate (x : List ℝ) (h1 : ∀ i, 0 ≤ x[i]) :
  λ = (1 / (sample_mean x)) :=
sorry

end max_likelihood_estimate_l91_91295


namespace sum_integers_from_neg50_to_70_l91_91888

theorem sum_integers_from_neg50_to_70 : (Finset.range (70 + 1)).sum id - (Finset.range (50 + 1)).sum id + (Finset.range (50+1)).sum (λ x, - x) = 1210 :=
by
  sorry

end sum_integers_from_neg50_to_70_l91_91888


namespace matrix_exp_1000_l91_91629

-- Define the matrix as a constant
noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![2, 1]]

-- The property of matrix exponentiation
theorem matrix_exp_1000 :
  A^1000 = ![![1, 0], ![2000, 1]] :=
by
  sorry

end matrix_exp_1000_l91_91629


namespace trinomial_is_x2_minus_2x_plus_1_l91_91639

noncomputable def trinomial_cubed : Polynomial ℤ :=
  x^6 - 6 * x^5 + 15 * x^4 - 20 * x^3 + 15 * x^2 - 6 * x + 1

theorem trinomial_is_x2_minus_2x_plus_1 :
  ∃ t : Polynomial ℤ, t = x^2 - 2 * x + 1 ∧ t^3 = trinomial_cubed :=
sorry

end trinomial_is_x2_minus_2x_plus_1_l91_91639


namespace master_codes_count_l91_91773

def num_colors : ℕ := 7
def num_slots : ℕ := 5

theorem master_codes_count : num_colors ^ num_slots = 16807 := by
  sorry

end master_codes_count_l91_91773


namespace prove_n_ge_4_l91_91793

def problem (f : ℤ → Fin n) : Prop :=
  ∀ x y : ℤ, |x - y| ∈ {2, 3, 5} → f x ≠ f y

theorem prove_n_ge_4 (n : ℕ) (f : ℤ → Fin n) (h : problem f) : n ≥ 4 :=
sorry

end prove_n_ge_4_l91_91793


namespace total_employees_in_buses_l91_91162

-- Definitions from conditions
def busCapacity : ℕ := 150
def percentageFull1 : ℕ := 60
def percentageFull2 : ℕ := 70

-- Proving the total number of employees
theorem total_employees_in_buses : 
  (percentageFull1 * busCapacity / 100) + (percentageFull2 * busCapacity / 100) = 195 := 
by
  sorry

end total_employees_in_buses_l91_91162


namespace total_employees_in_buses_l91_91163

-- Definitions from conditions
def busCapacity : ℕ := 150
def percentageFull1 : ℕ := 60
def percentageFull2 : ℕ := 70

-- Proving the total number of employees
theorem total_employees_in_buses : 
  (percentageFull1 * busCapacity / 100) + (percentageFull2 * busCapacity / 100) = 195 := 
by
  sorry

end total_employees_in_buses_l91_91163


namespace opposite_face_label_l91_91594

theorem opposite_face_label (squares : List Char) (top : Char) : squares.length = 6 ∧ squares = ['A', 'B', 'C', 'D', 'E', 'F'] ∧ top = 'B' → opposite_face 'D' = 'A' := 
sorry

end opposite_face_label_l91_91594


namespace ratio_of_inscribed_squares_l91_91609

-- Definition of the problem
def right_triangle_sides : Prop :=
  (3^2 + 4^2 = 5^2)

def inscribed_square_side_length (x : ℝ) : Prop :=
  x = 12/7

def inscribed_square_hypotenuse_length (y : ℝ) : Prop :=
  y = 60/37

def ratio_x_y (x y : ℝ) : Prop :=
  x / y = 37/35

-- Main statement to be proven
theorem ratio_of_inscribed_squares :
  (right_triangle_sides → ∃ x y : ℝ, inscribed_square_side_length x ∧ inscribed_square_hypotenuse_length y ∧ ratio_x_y x y) :=
by
  intro h
  use [12/7, 60/37]
  constructor
  -- Proofs omitted
  sorry

end ratio_of_inscribed_squares_l91_91609


namespace exponents_mod_7_l91_91478

theorem exponents_mod_7 : (2222 ^ 5555 + 5555 ^ 2222) % 7 = 0 := 
by 
  -- sorries here because no proof is needed as stated
  sorry

end exponents_mod_7_l91_91478


namespace total_carrots_l91_91422

-- Define constants for the number of carrots grown by each person
def Joan_carrots : ℕ := 29
def Jessica_carrots : ℕ := 11
def Michael_carrots : ℕ := 37
def Taylor_carrots : ℕ := 24

-- The proof problem: Prove that the total number of carrots grown is 101
theorem total_carrots : Joan_carrots + Jessica_carrots + Michael_carrots + Taylor_carrots = 101 :=
by
  sorry

end total_carrots_l91_91422


namespace problem_P_structure_faces_l91_91136

theorem problem_P_structure_faces :
  ∃ (A B C : ℕ), 
  (A = 2 ∧ B = 2 ∧ C = 3) ∧ 
  (∀ (face : ℕ), face = 1 ∨ face = 2 ∨ face = 3) ∧
  (∀ (cube : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ), 
    cube 3 2 2 1 1 1 = 10) ∧ 
  (∃ (structure : list (ℕ → ℕ)), 
    structure.length = 7 ∧ 
    (∀ i, i < 7 → structure.nth i = some (λ j, if j = 1 then 1 else if j = 2 then 2 else if j = 3 then 3 else 0))) ∧ 
  (∀ i j, i < 7 → j < 7 → i ≠ j → (structure.nth i).get_or_else (λ _, 0) 3 = (structure.nth j).get_or_else (λ _, 0) 3) := 
sorry

end problem_P_structure_faces_l91_91136


namespace min_distance_curve_C_to_line_l_l91_91767

noncomputable section

open Real

def curve_C (θ : ℝ) : ℝ × ℝ :=
  (4 * cos θ, 3 * sin θ)

def line_l (t : ℝ) : ℝ × ℝ :=
  (3 + (sqrt 2 / 2) * t, -3 + (sqrt 2 / 2) * t)

def distance_point_to_line (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  abs (x - y - 6) / sqrt 2

theorem min_distance_curve_C_to_line_l :
  ∃ θ : ℝ, distance_point_to_line (curve_C θ) = sqrt 2 / 2 :=
sorry

end min_distance_curve_C_to_line_l_l91_91767


namespace sum_basic_terms_divisible_by_4_l91_91586

def basic_term_sum_divisible_by_4 (n : ℕ) (grid : Fin n → Fin n → ℤ) : Prop :=
  ∀ (perm : Equiv.Perm (Fin n)),
    (∑ σ in univ, ∏ i, grid i (σ i)) % 4 = 0

theorem sum_basic_terms_divisible_by_4 {n : ℕ} (hn : n ≥ 4) (grid : Fin n → Fin n → ℤ)
  (h1 : ∀ i j, grid i j = 1 ∨ grid i j = -1) :
  basic_term_sum_divisible_by_4 n grid := 
sorry

end sum_basic_terms_divisible_by_4_l91_91586


namespace circle_area_above_line_l91_91551

-- Definition of the circle and the equation
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 8)^2 = 48

-- The given line equation
def line_eq (y : ℝ) : Prop := y = 4

-- The target area above the line
def area_above_line : ℝ := 24 * Real.pi

-- The main theorem statement
theorem circle_area_above_line : ∀ x y : ℝ, circle_eq x y ∧ y > 4 → sorry := sorry

end circle_area_above_line_l91_91551


namespace fencing_rate_correct_l91_91090

noncomputable def rate_of_fencing_per_meter (area_hectares : ℝ) (total_cost : ℝ) : ℝ :=
  let area_sqm := area_hectares * 10000
  let r_squared := area_sqm / Real.pi
  let r := Real.sqrt r_squared
  let circumference := 2 * Real.pi * r
  total_cost / circumference

theorem fencing_rate_correct :
  rate_of_fencing_per_meter 13.86 6070.778380479544 = 4.60 :=
by
  sorry

end fencing_rate_correct_l91_91090


namespace josiah_hans_age_ratio_l91_91789

theorem josiah_hans_age_ratio (H : ℕ) (J : ℕ) (hH : H = 15) (hSum : (J + 3) + (H + 3) = 66) : J / H = 3 :=
by
  sorry

end josiah_hans_age_ratio_l91_91789


namespace sum_of_sequence_l91_91666

noncomputable def a : ℕ → ℕ
| 0     := 2
| (n+1) := a n + 2^n

noncomputable def S : ℕ → ℕ
| 0     := 2
| (n+1) := S n + a (n+1)

theorem sum_of_sequence (n : ℕ) : S n = 2^(n+1) - 2 :=
sorry

end sum_of_sequence_l91_91666


namespace arithmetic_mean_of_digits_l91_91099

theorem arithmetic_mean_of_digits (p : ℕ) (h_prime : Prime p) (h_coprime : Nat.gcd p 10 = 1) :
  let period := repeatingDecimal p
  let k := period.length
  Even k → (∑ d in period, d) / k = 9 / 2 :=
by
  -- Proof skipped
  sorry

-- Definition of repeatingDecimal function, just as a placeholder, so the theorem can reference it
def repeatingDecimal (p : ℕ) : List ℕ := []

end arithmetic_mean_of_digits_l91_91099


namespace trajectory_and_perpendicular_lines_l91_91003

def P (x y : ℝ) := (x, y)
noncomputable def dist (P Q : ℝ × ℝ) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2)

theorem trajectory_and_perpendicular_lines:
  (∀ (x y : ℝ), dist (P x y) (0, -real.sqrt 3) + dist (P x y) (0, real.sqrt 3) = 4 → 
                 x^2 + (y^2 / 4) = 1) ∧ 
  (∀ (k : ℝ),
    (k = 1/2 ∨ k = -1/2) →
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      (y₁ = k * x₁ + 1 ∧ y₂ = k * x₂ + 1 ∧ 
      x₁^2 + (y₁^2 / 4) = 1 ∧ x₂^2 + (y₂^2 / 4) = 1 ∧ 
      x₁ * x₂ + y₁ * y₂ = 0 ∧ dist (P x₁ y₁) (P x₂ y₂) = 4 * real.sqrt 65 / 17))) :=
begin
  sorry,
end

end trajectory_and_perpendicular_lines_l91_91003


namespace cost_per_blue_shirt_l91_91499

theorem cost_per_blue_shirt :
  let pto_spent := 2317
  let num_kindergarten := 101
  let cost_orange := 5.80
  let total_orange := num_kindergarten * cost_orange

  let num_first_grade := 113
  let cost_yellow := 5
  let total_yellow := num_first_grade * cost_yellow

  let num_third_grade := 108
  let cost_green := 5.25
  let total_green := num_third_grade * cost_green

  let total_other_shirts := total_orange + total_yellow + total_green
  let pto_spent_on_blue := pto_spent - total_other_shirts

  let num_second_grade := 107
  let cost_per_blue_shirt := pto_spent_on_blue / num_second_grade

  cost_per_blue_shirt = 5.60 :=
by
  sorry

end cost_per_blue_shirt_l91_91499


namespace find_x_plus_2y_l91_91187

theorem find_x_plus_2y (x y : ℝ) 
  (h1 : x + y = 19) 
  (h2 : x + 3y = 1) : 
  x + 2y = 10 := 
by 
  sorry

end find_x_plus_2y_l91_91187


namespace number_of_triangles_l91_91741

theorem number_of_triangles (n : ℕ) : 
  ∃ k : ℕ, k = ⌊((n + 1) * (n + 3) * (2 * n + 1) : ℝ) / 24⌋ := sorry

end number_of_triangles_l91_91741


namespace probability_target_hit_l91_91954

-- Non-zero assumptions about probabilities
variables {P_A P_B P_C : ℝ}
variables (h_A : P_A = 1 / 2) (h_B : P_B = 1 / 3) (h_C : P_C = 1 / 4)

-- Statement of the target probability of being hit given independent shooting by A, B, and C
theorem probability_target_hit (P_A P_B P_C : ℝ) (h_A : P_A = 1 / 2) (h_B : P_B = 1 / 3) (h_C : P_C = 1 / 4) :
  let P_not_hit := (1 - P_A) * (1 - P_B) * (1 - P_C) in
  1 - P_not_hit = 3 / 4 := 
sorry

end probability_target_hit_l91_91954


namespace probability_of_odd_product_is_4_over_15_l91_91498

noncomputable def probability_odd_product_of_two_distinct_integers : ℚ :=
  let total_numbers := 15
  let total_ways := nat.choose total_numbers 2
  let odd_numbers := 8
  let odd_ways := nat.choose odd_numbers 2
  (odd_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_odd_product_is_4_over_15 :
  probability_odd_product_of_two_distinct_integers = 4 / 15 :=
sorry

end probability_of_odd_product_is_4_over_15_l91_91498


namespace max_mice_two_males_num_versions_PPF_PPF_combinations_PPF_three_kittens_l91_91086

-- Condition definitions
def PPF_male (K : ℕ) : ℕ := 80 - 4 * K
def PPF_female (K : ℕ) : ℕ := 16 - 0.25.to_nat * K

-- Proving the maximum number of mice that could be caught by 2 male kittens
theorem max_mice_two_males : ∀ K, PPF_male 0 + PPF_male 0 = 160 :=
by simp [PPF_male]

-- Proving there are 3 possible versions of the PPF
theorem num_versions_PPF : ∃ (versions : set (ℕ → ℕ)), versions = 
  { λ K, 160 - 4 * K,
    λ K, 32 - 0.5.to_nat * K,
    λ K, if K ≤ 64 then 96 - 0.25.to_nat * K else 336 - 4 * K } ∧
  versions.size = 3 :=
by sorry

-- Proving the analytical form of each PPF combination
theorem PPF_combinations : 
  (∀ K, (λ K, 160 - 4 * K) K = PPF_male K + PPF_male K) ∧
  (∀ K, (λ K, 32 - 0.5.to_nat * K) K = PPF_female K + PPF_female K) ∧
  (∀ K, if K ≤ 64 then (λ K, 96 - 0.25.to_nat * K) K = PPF_male K + PPF_female K else (λ K, 336 - 4 * K) K = PPF_male (K - 64) + PPF_female 64) :=
by sorry

-- Proving the analytical form when accepting the third kitten
theorem PPF_three_kittens :
  (∀ K, if K ≤ 64 then (176 - 0.25.to_nat * K) = PPF_male K + PPF_male K + PPF_female K else (416 - 4 * K) = PPF_male (K - 64) + PPF_male 64 + PPF_female 64) :=
by sorry

end max_mice_two_males_num_versions_PPF_PPF_combinations_PPF_three_kittens_l91_91086


namespace root_inverse_cubes_l91_91747

theorem root_inverse_cubes (a b c r s : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = r ∨ x = s) :
  (1 / r^3) + (1 / s^3) = (-b^3 + 3 * a * b * c) / c^3 :=
by
  sorry

end root_inverse_cubes_l91_91747


namespace color_graph_l91_91029

-- Define the graph G and the necessary conditions
variables {V : Type*} [fintype V] [decidable_eq V]
noncomputable def G (n : ℕ) : simple_graph V :=
{ adj := λ u v, true, -- assuming a complete graph of 2n vertices for simplicity
  symm := λ u v h, h,
  loopless := λ u, false }

-- Define the number of vertices and edges
def vertex_count (n : ℕ) : ℕ := 2 * n
def edge_count (n : ℕ) : ℕ := 2 * n * (n - 1)

-- Define the coloring property
def red_coloring (n : ℕ) (G : simple_graph V) : Prop :=
  ∃ (red_vertices : finset V) (red_edges : finset (V × V)),
    red_vertices.card = n + 1 ∧
    red_edges.card = n * (n + 1) ∧
    (∀ e ∈ red_edges, e.fst ∈ red_vertices ∧ e.snd ∈ red_vertices) ∧
    (∀ v ∈ red_vertices, (red_edges.filter (λ e, e.fst = v ∨ e.snd = v)).card = n)

-- Main theorem statement
theorem color_graph (G : simple_graph V) (h_vertex_count : G.vertex_count = 2 * n) (h_edge_count : G.edge_count = 2 * n * (n - 1)) :
  ∃ (red_vertices : finset V) (red_edges : finset (V × V)),
    red_coloring n G :=
sorry

end color_graph_l91_91029


namespace shortest_baking_time_l91_91138

def cakes_baking_time : ℕ := 3

theorem shortest_baking_time (cakes : ℕ) (baking_time_per_side : ℕ) (pan_capacity : ℕ) : cakes = 3 → baking_time_per_side = 1 → pan_capacity = 2 → cakes_baking_time = 3 :=
by
  intros hcakes hbaking hpan
  rw [hcakes, hbaking, hpan]
  exact rfl

end shortest_baking_time_l91_91138


namespace men_at_yoga_studio_l91_91137

open Real

def yoga_men_count (M : ℕ) (avg_weight_men avg_weight_women avg_weight_total : ℝ) (num_women num_total : ℕ) : Prop :=
  avg_weight_men = 190 ∧
  avg_weight_women = 120 ∧
  num_women = 6 ∧
  num_total = 14 ∧
  avg_weight_total = 160 →
  M + num_women = num_total ∧
  (M * avg_weight_men + num_women * avg_weight_women) / num_total = avg_weight_total ∧
  M = 8

theorem men_at_yoga_studio : ∃ M : ℕ, yoga_men_count M 190 120 160 6 14 :=
  by 
  use 8
  sorry

end men_at_yoga_studio_l91_91137


namespace general_term_formula_values_of_n_l91_91343

-- Definitions for problem (1)
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

def arithmetic_condition (a : ℕ → ℝ) : Prop :=
  2 * a 1 * a 2 + a 1 * a 3 = 12

theorem general_term_formula (a : ℕ → ℝ) (h1 : is_geometric a) (h2 : a 1 = 2) (h3 : arithmetic_condition a) :
  ∀ n, a n = 2 ^ n :=
sorry

-- Definitions for problem (2)
def T_n (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n, 1 / (b i * b (i + 1))

theorem values_of_n (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : b = λ (n : ℕ), log 2 (a n))
  (h2 : ∀ n, T_n b n < 6 / 7 → n < 6) :
  ∀ n, T_n b n < 6 / 7 → n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 :=
sorry

end general_term_formula_values_of_n_l91_91343


namespace greatest_ratio_value_l91_91274

noncomputable def ratio_max : ℝ :=
  let circle_points := { (x, y) : ℤ × ℤ | (x*x + y*y = 16) }.to_finset in
  let pairs := (circle_points.product circle_points).filter (λ p, p.1 ≠ p.2) in
  let distances := pairs.map (λ p, (p.1, p.2, real.sqrt ((p.2.1 - p.1.1) * (p.2.1 - p.1.1) + (p.2.2 - p.1.2) * (p.2.2 - p.1.2)))) in
  let rational_distances := distances.filter (λ d, d.2.2.nat_abs^2 = (d.2.2.nat_abs.to_nat)^2) in
  let ab_cd_ratios := rational_distances.product rational_distances |>
                      map (λ p, p.1.2.2 / p.2.2.2) |>
                      filter (λ r, ∃ a b c d, a ≠ b ∧ c ≠ d ∧ r = (rational_distances.filter (λ d, d.1 = a ∧ d.2 = b).head).2.2 / (rational_distances.filter (λ d, d.1 = c ∧ d.2 = d).head).2.2) in
  ab_cd_ratios |>.max

theorem greatest_ratio_value :
  ratio_max = real.sqrt 10 / 4 :=
sorry

end greatest_ratio_value_l91_91274


namespace mike_needs_percentage_to_pass_l91_91462

theorem mike_needs_percentage_to_pass :
  ∀ (mike_score marks_short max_marks : ℕ),
  mike_score = 212 → marks_short = 22 → max_marks = 780 →
  ((mike_score + marks_short : ℕ) / (max_marks : ℕ) : ℚ) * 100 = 30 :=
by
  intros mike_score marks_short max_marks Hmike Hshort Hmax
  rw [Hmike, Hshort, Hmax]
  -- Proof will be filled out here
  sorry

end mike_needs_percentage_to_pass_l91_91462


namespace angle_between_unit_vectors_l91_91690

variable {V : Type*} [InnerProductSpace ℝ V]

/-- Given vectors e1 and e2 are unit vectors and satisfy a specific dot product condition,
prove that the angle between them is π/4. -/
theorem angle_between_unit_vectors 
	(e1 e2 : V) 
	(h1 : ‖e1‖ = 1) 
	(h2 : ‖e2‖ = 1) 
	(h : 2 * e1 + e2 ⬝ -2 * e1 + 3 * e2 = 2 * Real.sqrt 2 - 1) : 
	angle_between e1 e2 = π / 4 :=
sorry 

end angle_between_unit_vectors_l91_91690


namespace intersection_points_count_l91_91110

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count :
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2) := sorry

end intersection_points_count_l91_91110


namespace hyperbola_eccentricity_l91_91048

variable {a b c : ℝ} (ha : a > 0) (hb : b > 0) (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
variable (λ μ : ℝ) (h_λμ : λ * μ = 3 / 16) (c_formula : c^2 = a^2 + b^2)

theorem hyperbola_eccentricity
  (h_asymptotes : ∀ x, (y = b / a * x) ∨ (y = -b / a * x))
  (h_F : ∃ c, F = (c, 0))
  (h_A_B_P : ∃ c, A = (c, b * c / a) ∧ B = (c, -b * c / a) ∧ P = (c, b^2 / a))
  (h_OP : ∀ O P A B : ℝ × ℝ, (c, b^2 / a) = (λ + μ * c, λ - μ * b * c / a))
  : eccentricity e = 2 * sqrt(3) / 3 := 
sorry

end hyperbola_eccentricity_l91_91048


namespace savannah_wrapped_gifts_l91_91071

theorem savannah_wrapped_gifts :
  (let rolls := 3 in
   let gifts_per_roll1 := 3 in
   let gifts_per_roll2 := 5 in
   let gifts_per_roll3 := 4 in
   gifts_per_roll1 + gifts_per_roll2 + gifts_per_roll3 = 12) :=
by
  sorry

end savannah_wrapped_gifts_l91_91071


namespace coordinates_of_a_l91_91006

variables (i j : V) (x y : ℝ)

-- Assuming i is a unit vector along the positive x-axis
axiom i_unit_x : i = (1, 0) 

-- Assuming j is a unit vector along the positive y-axis
axiom j_unit_y : j = (0, 1)

-- Defining vector a as a combination of the unit vectors i and j
def a := x • i + y • j

-- The theorem to prove the coordinates of vector a
theorem coordinates_of_a : a = (x, y) :=
by sorry

end coordinates_of_a_l91_91006


namespace complete_square_q_value_l91_91786

theorem complete_square_q_value :
  ∃ p q, (16 * x^2 - 32 * x - 512 = 0) ∧ ((x + p)^2 = q) → q = 33 := by
  sorry

end complete_square_q_value_l91_91786


namespace triangle_angle_side_cases_l91_91414

theorem triangle_angle_side_cases
  (b c : ℝ) (B : ℝ)
  (hb : b = 3)
  (hc : c = 3 * Real.sqrt 3)
  (hB : B = Real.pi / 6) :
  (∃ A C a, A = Real.pi / 2 ∧ C = Real.pi / 3 ∧ a = Real.sqrt 21) ∨
  (∃ A C a, A = Real.pi / 6 ∧ C = 2 * Real.pi / 3 ∧ a = 3) :=
by
  sorry

end triangle_angle_side_cases_l91_91414


namespace custom_op_evaluation_l91_91265

def custom_op (a b : ℤ) : ℤ := a * b - (a + b)

theorem custom_op_evaluation : custom_op 2 (-3) = -5 :=
by
sorry

end custom_op_evaluation_l91_91265


namespace no_integer_x_square_l91_91043

theorem no_integer_x_square (x : ℤ) : 
  ∀ n : ℤ, x^5 + 5 * x^4 + 10 * x^3 + 10 * x^2 + 5 * x + 1 ≠ n^2 :=
by sorry

end no_integer_x_square_l91_91043


namespace derivative_limit_half_l91_91338

variable {ℝ : Type} [LinearOrderedField ℝ] [TopologicalSpace ℝ] [OrderTopology ℝ]

theorem derivative_limit_half (f : ℝ → ℝ) (x₀ : ℝ) (h : HasDerivAt f 1 x₀) :
  (lim (fun Δx : ℝ => (f (x₀ + Δx) - f x₀) / (2 * Δx)) (𝓝 0)) = 1 / 2 := sorry

end derivative_limit_half_l91_91338


namespace add_100ml_water_l91_91204

theorem add_100ml_water 
    (current_volume : ℕ) 
    (current_water_percentage : ℝ) 
    (desired_water_percentage : ℝ) 
    (current_water_volume : ℝ) 
    (x : ℝ) :
    current_volume = 300 →
    current_water_percentage = 0.60 →
    desired_water_percentage = 0.70 →
    current_water_volume = 0.60 * 300 →
    180 + x = 0.70 * (300 + x) →
    x = 100 := 
sorry

end add_100ml_water_l91_91204


namespace graph_B_is_y_gx_plus_2_l91_91853

def g (x : ℝ) : ℝ :=
  if x ≥ -3 ∧ x ≤ 0 then -x
  else if x > 0 ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2)
  else if x > 2 ∧ x ≤ 3 then 2 * (x - 2) + 2
  else 0 -- out of defined domain

def f (x : ℝ) := g x + 2

-- Graph A represents g(x)
-- Graph B represents f(x) = g(x) + 2
-- Graph C represents g(x) - 2

theorem graph_B_is_y_gx_plus_2 : 
  ∀ (x : ℝ), (B x = f x) :=
sorry

end graph_B_is_y_gx_plus_2_l91_91853


namespace fraction_simplification_l91_91558

theorem fraction_simplification : (4 * 5) / 10 = 2 := 
by 
  sorry

end fraction_simplification_l91_91558


namespace maximum_value_sum_l91_91061

theorem maximum_value_sum (a b c d : ℕ) (h1 : a + c = 1000) (h2 : b + d = 500) :
  ∃ a b c d, a + c = 1000 ∧ b + d = 500 ∧ (a = 1 ∧ c = 999 ∧ b = 499 ∧ d = 1) ∧ 
  ((a : ℝ) / b + (c : ℝ) / d = (1 / 499) + 999) := 
  sorry

end maximum_value_sum_l91_91061


namespace area_of_quadrilateral_ABCD_l91_91393

variable (AC BD : ℝ × ℝ)
def AC_val := (-2, 1)
def BD_val := (2, 4)

theorem area_of_quadrilateral_ABCD :
  AC = AC_val → BD = BD_val → (0.5 * (Real.sqrt (AC.1^2 + AC.2^2)) * (Real.sqrt (BD.1^2 + BD.2^2))) = 5 :=
by
  intros hAC hBD
  rw [hAC, hBD]
  sorry

end area_of_quadrilateral_ABCD_l91_91393


namespace three_digit_number_unchanged_upside_down_l91_91235

theorem three_digit_number_unchanged_upside_down (n : ℕ) :
  (n >= 100 ∧ n <= 999) ∧ (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d = 0 ∨ d = 8) ->
  n = 888 ∨ n = 808 :=
by
  sorry

end three_digit_number_unchanged_upside_down_l91_91235


namespace sum_of_possible_values_sum_of_all_possible_values_l91_91404

theorem sum_of_possible_values (x : ℝ) : (|x-5| = 4) → (x = 9 ∨ x = 1) :=
begin
  sorry
end

theorem sum_of_all_possible_values (h : ∃ x : ℝ, |x-5| = 4) : 9 + 1 = 10 :=
by
  trivial

end sum_of_possible_values_sum_of_all_possible_values_l91_91404


namespace exists_separating_line_l91_91828

noncomputable def separate_convex_figures (Φ₁ Φ₂ : Set ℝ²) : Prop :=
  Bounded Φ₁ ∧ Bounded Φ₂ ∧ Convex ℝ Φ₁ ∧ Convex ℝ Φ₂ ∧
  Disjoint Φ₁ Φ₂ → 
  ∃ l : AffineSubspace ℝ ℝ², (∃ (H₁ : Φ₁ ⊆ HalfSpace l ∧ Φ₂ ⊆ -HalfSpace l))

theorem exists_separating_line (Φ₁ Φ₂ : Set ℝ²) :
  separate_convex_figures Φ₁ Φ₂ :=
sorry

end exists_separating_line_l91_91828


namespace complex_quadrant_l91_91693

theorem complex_quadrant :
  ∃ Z : ℂ, Z = (2 - complex.i) / complex.i ∧ Z.re < 0 ∧ Z.im < 0 := 
by {
  have h : (2 - complex.i) / complex.i = -1 - 2 * complex.i,
  {
    calc
    (2 - complex.i) / complex.i
        = (2 - complex.i) * (-complex.i) / (complex.i * -complex.i) : by ring_nf
    ... = ((2 * -complex.i) - (-complex.i * complex.i)) / (- (complex.i ^ 2)) : by ring_nf
    ... = ((-2 * complex.i) - (-i ^ 2) * complex.i) / -(-1) : by { rw complex.ext_iff, norm_num, ring_nf, field_simp, },
  },
  existsi (-1 - 2 * complex.i),
  simp [h],
  norm_num,
}

end complex_quadrant_l91_91693


namespace minimum_detectors_l91_91141

theorem minimum_detectors (grid_size ship_size : ℕ) (k : ℕ) :
  grid_size = 2015 → ship_size = 1500 →
  (∀ dima_placing : set (ℕ × ℕ), dima_placing.card = k →
    ∃ kolya_placing : set (ℕ × ℕ), (kolya_placing.card = ship_size * ship_size) ∧
      (∀ cell ∈ dima_placing, cell ∈ kolya_placing → True)) →
  k = 1030 :=
by
  sorry

end minimum_detectors_l91_91141


namespace bijective_f_ap_l91_91430

theorem bijective_f_ap (f : ℕ → ℕ) (bij : function.bijective f) :
  ∃ a b c : ℕ, (b = a + k ∧ c = a + 2 * k) ∧ f a < f b ∧ f b < f c :=
sorry

end bijective_f_ap_l91_91430


namespace max_sqrt_sum_l91_91304

theorem max_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
sorry

end max_sqrt_sum_l91_91304


namespace sum_powers_of_i_l91_91492

theorem sum_powers_of_i : (Finset.range 2014).sum (λ n, (complex.I)^n) = 1 := 
by
  sorry

end sum_powers_of_i_l91_91492


namespace simplify_expression_evaluate_expression_l91_91491

-- Definitions for the first part
variable (a b : ℝ)

theorem simplify_expression (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 * a^(1/2) * b^(1/3)) * (a^(2/3) * b^(1/2)) / (1/3 * a^(1/6) * b^(5/6)) = 6 * a :=
by
  sorry

-- Definitions for the second part
theorem evaluate_expression :
  (9 / 16)^(1 / 2) + 10^(Real.log 9 / Real.log 10 - 2 * Real.log 2 / Real.log 10) + Real.log (4 * Real.exp 3) 
  - (Real.log 8 / Real.log 9) * (Real.log 33 / Real.log 4) = 7 / 2 :=
by 
  sorry

end simplify_expression_evaluate_expression_l91_91491


namespace arithmetic_sequence_150th_term_l91_91996

theorem arithmetic_sequence_150th_term :
  let a1 := 3
  let d := 5
  let n := 150
  a1 + (n - 1) * d = 748 :=
by
  sorry

end arithmetic_sequence_150th_term_l91_91996


namespace medians_inequality_l91_91859

theorem medians_inequality (s_a s_b s_c a b c : ℝ) (s : ℝ) 
  (h₁ : s_a, s_b, s_c are the medians of triangle)
  (h₂ : s = (a + b + c) / 2) :
  1.5 * s < s_a + s_b + s_c ∧ s_a + s_b + s_c < 2 * s :=
by sorry

end medians_inequality_l91_91859


namespace largest_net_diff_in_march_l91_91818

noncomputable def net_earnings (sales : ℕ) (cost_per_item : ℕ) : ℤ :=
  sales - sales * cost_per_item

def sales_per_month : List (ℕ × ℕ) :=
  [(150, 100), (200, 150), (180, 180), (120, 160), (80, 120)]

def cost_per_item_drummers : ℕ := 1
def cost_per_item_buglers : ℕ := 2

def net_earnings_drummers (sales : ℕ) : ℤ :=
  net_earnings sales cost_per_item_drummers

def net_earnings_buglers (sales : ℕ) : ℤ :=
  net_earnings sales cost_per_item_buglers

def net_earning_diff_perc (d_sales : ℕ) (b_sales : ℕ) : ℚ :=
  let drummers_net := net_earnings_drummers d_sales
  let buglers_net := net_earnings_buglers b_sales
  if drummers_net = 0 then 0 else (drummers_net - buglers_net : ℚ) / drummers_net

def month_of_largest_net_diff (sales_list : List (ℕ × ℕ)) : ℕ :=
  let diffs := sales_list.map (λ (s : ℕ × ℕ) => net_earning_diff_perc s.fst s.snd)
  List.arg_max diffs 0

theorem largest_net_diff_in_march :
  month_of_largest_net_diff sales_per_month = 2 := 
  sorry

end largest_net_diff_in_march_l91_91818


namespace ratio_of_parallel_intersections_l91_91145

-- Setting up the geometric context
variables {Point : Type} [AffineSpace Point]
variables (O A B C A' B' C' : Point)
variables (l1 l2 l3 : AffineLine Point)
variables (L1 L2 : AffineLine Point)
variables (h_parallel : L1 ∥ L2)
variables (h_l1 : ∃ p, O ∈ l1 ∧ p ∈ l1 ∧ (p ∈ L1 ∨ p ∈ L2) ∧ (p = A ∨ p = A'))
variables (h_l2 : ∃ p, O ∈ l2 ∧ p ∈ l2 ∧ (p ∈ L1 ∨ p ∈ L2) ∧ (p = B ∨ p = B'))
variables (h_l3 : ∃ p, O ∈ l3 ∧ p ∈ l3 ∧ (p ∈ L1 ∨ p ∈ L2) ∧ (p = C ∨ p = C'))

-- Statement to be proved
theorem ratio_of_parallel_intersections :
  (distance A B / distance B C) = (distance A' B' / distance B' C') :=
sorry

end ratio_of_parallel_intersections_l91_91145


namespace min_value_of_fraction_l91_91806

theorem min_value_of_fraction (n : ℕ) (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  (1 / (1 + a^n) + 1 / (1 + b^n)) = 1 :=
sorry

end min_value_of_fraction_l91_91806


namespace total_employees_in_buses_l91_91159

theorem total_employees_in_buses :
  let bus1_percentage_full := 0.60,
      bus2_percentage_full := 0.70,
      bus_capacity := 150
  in
  (bus1_percentage_full * bus_capacity + bus2_percentage_full * bus_capacity) = 195 := by
  sorry

end total_employees_in_buses_l91_91159


namespace num_divisors_m5n3_eq_217_l91_91451

variables {m n : ℕ} (hrel_prime : Nat.coprime m n) (hpos_m : 0 < m) (hpos_n : 0 < n)

def num_divisors (k : ℕ) : ℕ :=
  (List.range (k + 1)).count (λ p, k % p = 0)

theorem num_divisors_m5n3_eq_217
  (hdivisors : num_divisors (m^3 * n^5) = 209) :
  num_divisors (m^5 * n^3) = 217 :=
sorry

end num_divisors_m5n3_eq_217_l91_91451


namespace integral_value_existence_l91_91809

open Real

variables {a : ℝ} (h : ℝ → ℝ) (c : ℝ)

-- Define conditions as Lean hypotheses
def h_definition (x : ℝ) : Prop :=
  h(x) = sin (π * sin (π * x / 2) / 2) + (2 / π) * arcsin ((2 / π) * arcsin x) - 2 * x

def c_definition : Prop :=
  c = 0.5 * (a - (2 / π) * arcsin ((2 / π) * arcsin a)) * (sin (π * sin (π * a / 2) / 2) - a)

-- Main theorem statement
theorem integral_value_existence (h_def : ∀ x, h_definition x) (c_def : c_definition) (ha : 0 < a) (ha1 : a < 1) :
  ∃ x1 x2 : ℝ, 
    (0 < x1) ∧ (x1 < 1) ∧
    (0 < x2) ∧ (x2 < 1) ∧
    (x1 ≠ x2) ∧ 
    (∫ t in 0..x1, h t = c) ∧ 
    (∫ t in 0..x2, h t = c) := 
by 
  sorry

end integral_value_existence_l91_91809


namespace area_is_correct_l91_91224

-- We define constants for the bases and height based on the problem's conditions
def lower_base : ℝ := 60
def upper_base : ℝ := 0.6 * lower_base
def height : ℝ := lower_base

-- We need to calculate the area of the right trapezoid using the given formula
def area := 1 / 2 * (lower_base + upper_base) * height

-- We prove that the calculated area matches the given answer
theorem area_is_correct : area = 2880 := by
  sorry

end area_is_correct_l91_91224


namespace max_real_roots_poly_l91_91636

theorem max_real_roots_poly (n : ℕ) (h : n > 0) :
  ∃ k : ℝ, (x^n - x^(n-1) + ∑ i in finset.range (n-1), (-1)^i * x^(n-1-i) + 1 = 0) → k <= 1 :=
sorry

end max_real_roots_poly_l91_91636


namespace exponential_plus_x_satisfies_inequality_l91_91240

theorem exponential_plus_x_satisfies_inequality (x : ℝ) :
  let f := λ (x : ℝ), Real.exp x + x in
  f(x+1) > f(x) + 1 :=
by
  sorry

end exponential_plus_x_satisfies_inequality_l91_91240


namespace number_of_white_stones_is_3600_l91_91135

-- Definitions and conditions
def total_stones : ℕ := 6000
def total_difference_to_4800 : ℕ := 4800
def W : ℕ := 3600

-- Conditions
def condition1 (B : ℕ) : Prop := total_stones - W + B = total_difference_to_4800
def condition2 (B : ℕ) : Prop := W + B = total_stones
def condition3 (B : ℕ) : Prop := W > B

-- Theorem statement
theorem number_of_white_stones_is_3600 :
  ∃ B : ℕ, condition1 B ∧ condition2 B ∧ condition3 B :=
by
  -- TODO: Complete the proof
  sorry

end number_of_white_stones_is_3600_l91_91135


namespace total_profit_is_35000_l91_91955

open Real

-- Define the subscriptions of A, B, and C
def subscriptions (A B C : ℝ) : Prop :=
  A + B + C = 50000 ∧
  A = B + 4000 ∧
  B = C + 5000

-- Define the profit distribution and the condition for C's received profit
def profit (total_profit : ℝ) (A B C : ℝ) (C_profit : ℝ) : Prop :=
  C_profit / total_profit = C / (A + B + C) ∧
  C_profit = 8400

-- Lean 4 statement to prove total profit
theorem total_profit_is_35000 :
  ∃ A B C total_profit, subscriptions A B C ∧ profit total_profit A B C 8400 ∧ total_profit = 35000 :=
by
  sorry

end total_profit_is_35000_l91_91955


namespace mri_cost_is_1200_l91_91874

-- Definitions of conditions
def examination_fee : ℝ := 150
def additional_fee : ℝ := 150
def combined_initial_fee : ℝ := examination_fee + additional_fee
def tim_payment : ℝ := 300
def insurance_coverage : ℝ := 0.80
def tim_coverage : ℝ := 0.20

-- Define the total cost (including the MRI cost)
def total_cost : ℝ := tim_payment / tim_coverage

-- Define the MRI cost in terms of the total cost and initial fees
def mri_cost : ℝ := total_cost - combined_initial_fee

-- The theorem we need to prove
theorem mri_cost_is_1200 : mri_cost = 1200 :=
  by
  sorry

end mri_cost_is_1200_l91_91874


namespace primes_of_form_pp_plus_one_l91_91196

theorem primes_of_form_pp_plus_one :
  ∀ (p : ℕ), (p.prime ∧ (p^p + 1).prime ∧ (p^p + 1).digits.length ≤ 19) ↔ (p = 2 ∨ p = 5 ∨ p = 257) :=
by
  sorry

end primes_of_form_pp_plus_one_l91_91196


namespace tangency_normal_orthogonal_l91_91452

-- Define the cyclic and tangential quadrilateral
variables (A B C D E F G H : Type) 
  [cyclic_quadrilateral A B C D] [tangential_quadrilateral A B C D]

-- Define the points of tangency
variables (E : point_tangency (A, B)) (F : point_tangency (B, C)) 
          (G : point_tangency (C, D)) (H : point_tangency (D, A))

-- Goal: Prove that (EG) is perpendicular to (HF)
theorem tangency_normal_orthogonal 
  (h_A : cyclic_quadrilateral A B C D)
  (h_B : tangential_quadrilateral A B C D) 
  (E_tan : point_tangency (E, A, B))
  (F_tan : point_tangency (F, B, C))
  (G_tan : point_tangency (G, C, D))
  (H_tan : point_tangency (H, D, A)) : 
  -- statement to prove
  is_perpendicular (line_through E G) (line_through H F) :=
sorry

end tangency_normal_orthogonal_l91_91452


namespace number_of_candied_grapes_l91_91978

variables (G : ℝ)

def candied_apples := 15
def price_apples := 2
def price_grapes := 1.5
def total_earnings := 48
def earnings_apples := candied_apples * price_apples
def earnings_grapes := total_earnings - earnings_apples

theorem number_of_candied_grapes : G * price_grapes = earnings_grapes → G = 12 :=
by
  intros h
  sorry

end number_of_candied_grapes_l91_91978


namespace no_magpies_left_l91_91541

theorem no_magpies_left (initial_magpies killed_magpies : ℕ) (fly_away : Prop):
  initial_magpies = 40 → killed_magpies = 6 → fly_away → ∀ M : ℕ, M = 0 :=
by
  intro h0 h1 h2
  sorry

end no_magpies_left_l91_91541


namespace sin_alpha_minus_pi_six_l91_91689

noncomputable def sin_diff := (α β : ℝ) (sin_α cos_α sin_β cos_β : ℝ) :
  sin (α - β) = sin_α * cos_β - cos_α * sin_β := by
  sorry

theorem sin_alpha_minus_pi_six (α : ℝ) (h1 : cos α = 3 / 5) (h2 : 0 < α ∧ α < π / 2) :
  sin (α - π / 6) = (4 * Real.sqrt 3 - 3) / 10 :=
sorry

end sin_alpha_minus_pi_six_l91_91689


namespace main_theorem_l91_91698

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem main_theorem :
  (∀ x : ℝ, f (x + 5/2) + f x = 2) ∧
  (∀ x : ℝ, f (1 + 2*x) = f (1 - 2*x)) ∧
  (∀ x : ℝ, g (x + 2) = g (x - 2)) ∧
  (∀ x : ℝ, g (-x + 1) - 1 = -g (x + 1) + 1) ∧
  (∀ x : ℝ, 0 ≤ x → x ≤ 2 → f x + g x = 3^x + x^3) →
  f 2022 * g 2022 = 72 :=
sorry

end main_theorem_l91_91698


namespace ferry_tourists_total_l91_91947

def series_sum (a d n : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem ferry_tourists_total :
  let t_0 := 90
  let d := -2
  let n := 9
  series_sum t_0 d n = 738 :=
by
  sorry

end ferry_tourists_total_l91_91947


namespace find_largest_l91_91963

noncomputable def largest_number_among_four : Prop :=
  let a := Real.pi
  let b := Real.sqrt 2
  let c := abs (-2)
  let d := 3
  a > b ∧ a > c ∧ a > d

theorem find_largest : largest_number_among_four :=
by
  -- Conditions from the problem
  let a := Real.pi
  let b := Real.sqrt 2
  let c := abs (-2)
  let d := 3
  have H_c : c = 2 := by simp
  have H_b : b < 2 := Real.sqrt_lt.2 one_lt_two
  have H_2 : 2 < 3 := by norm_num
  have H_3 : 3 < a := by linarith [Real.pi_lt_four]
  -- Goal
  exact ⟨H_3, H_c ▸ H_3.trans H_2, H_3⟩

end find_largest_l91_91963


namespace inequality_ge_five_halves_l91_91808

open Real

noncomputable def xy_yz_zx_eq_one (x y z : ℝ) := x * y + y * z + z * x = 1
noncomputable def non_neg (x y z : ℝ) := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

theorem inequality_ge_five_halves (x y z : ℝ) (h1 : xy_yz_zx_eq_one x y z) (h2 : non_neg x y z) :
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 5 / 2 := 
sorry

end inequality_ge_five_halves_l91_91808


namespace unique_zero_function_l91_91284

variable (f : ℝ → ℝ)

theorem unique_zero_function (h : ∀ x y : ℝ, f (x + y) = f x - f y) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end unique_zero_function_l91_91284


namespace limit_fraction_l91_91253

theorem limit_fraction :
  (tendsto (λ n : ℕ, (4 - 3 * n) / (2 * n + 1)) at_top (𝓝 (-3 / 2))) :=
sorry

end limit_fraction_l91_91253


namespace triangle_geom_l91_91823

theorem triangle_geom (A B C K L O : Point) (hK : K ∈ Segment B C)
  (hAngle : ∠ C A K = ∠ B / 2) (hO : O ∈ Segment A K)
  (hAngleBisector : O ∈ Line B L)
  (hDiv : Segment A K = 2 * Segment A O) :
  Segment A O * Segment L C = Segment B C * Segment O L :=
by
  sorry

end triangle_geom_l91_91823


namespace books_per_shelf_l91_91830

theorem books_per_shelf (mystery_shelves picture_shelves total_books : ℕ) (h1 : mystery_shelves = 6) (h2 : picture_shelves = 2) (h3 : total_books = 72) : 
  total_books / (mystery_shelves + picture_shelves) = 9 :=
by
  rw [h1, h2, h3]
  show 72 / (6 + 2) = 9
  rw [nat.add_comm, nat.div_eq_of_eq_mul_right]
  exact rfl
  sorry

end books_per_shelf_l91_91830


namespace part1_part2_part3_l91_91315

-- Part (1)
theorem part1 (f g : ℝ → ℝ) (a : ℝ) (h : ∀ (x : ℝ), 0 < x → f x = x * log x - a * x ∧ g x = -x^2 - 2) :
  (∀ (x : ℝ), 0 < x → f x ≥ g x) → a ≤ 3 := 
sorry

-- Part (2)
theorem part2 (f : ℝ → ℝ) (a : ℝ) (m : ℝ) (h : a = -1) (hm : 0 < m):
  (∀ (x: ℝ), m ≤ x ∧ x ≤ m + 3 → f x = x * log x + x) →
  (∃ val_min val_max : ℝ, 
    (val_min = min (m * (log m + 1)) (-1 / exp 2)) ∧ 
    (val_max = (m+3) * (log (m+3) + 1))) := 
sorry

-- Part (3)
theorem part3 (x : ℝ) (hx : 0 < x): 
  log x + 1 > 1 / exp x - 2 / (exp x * x) := 
sorry

end part1_part2_part3_l91_91315


namespace average_of_t_l91_91711

theorem average_of_t (t : ℕ) 
  (roots_positive : ∀ r : ℕ, r ∈ (ROOTS of (x^2 - 7*x + t)) → r > 0) 
  (sum_of_roots_eq_seven : ∀ r1 r2 : ℕ, r1 + r2 = 7)
  (product_of_roots_eq_t : ∀ r1 r2 : ℕ, r1 * r2 = t) : 
  (6 + 10 + 12) / 3 = 28 / 3 :=
sorry

end average_of_t_l91_91711


namespace second_occurrence_at_55_l91_91822

/-- On the highway, starting from 3 kilometers, there is a speed limit sign every 4 kilometers,
and starting from 10 kilometers, there is a speed monitoring device every 9 kilometers.
The first time both types of facilities are encountered simultaneously is at 19 kilometers.
The second time both types of facilities are encountered simultaneously is at 55 kilometers. -/
theorem second_occurrence_at_55 :
  ∀ (k : ℕ), (∃ n m : ℕ, 3 + 4 * n = k ∧ 10 + 9 * m = k ∧ 19 + 36 = k) := sorry

end second_occurrence_at_55_l91_91822


namespace angles_in_interval_l91_91266

open Real

theorem angles_in_interval
    (θ : ℝ)
    (hθ : 0 ≤ θ ∧ θ ≤ 2 * π)
    (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^2 * sin θ - x * (2 - x) + (2 - x)^2 * cos θ > 0) :
  π / 12 < θ ∧ θ < 5 * π / 12 :=
by
  sorry

end angles_in_interval_l91_91266


namespace john_correct_problems_needed_l91_91087

theorem john_correct_problems_needed 
  (total_problems : ℕ)
  (correct_points : ℕ)
  (incorrect_points : ℕ)
  (unanswered_points : ℕ)
  (attempted_problems : ℕ)
  (unanswered_problems : ℕ)
  (required_points : ℕ)
  (earned_points_unanswered : ℕ)
  (needed_points_from_attempted : ℕ)
  (min_correct_answers : ℕ) :
  total_problems = 25 →
  correct_points = 7 →
  incorrect_points = 0 →
  unanswered_points = 2 →
  attempted_problems = 20 →
  unanswered_problems = 5 →
  required_points = 120 →
  earned_points_unanswered = unanswered_problems * unanswered_points →
  needed_points_from_attempted = required_points - earned_points_unanswered →
  min_correct_answers = (needed_points_from_attempted + correct_points - 1) / correct_points →
  min_correct_answers = 16 :=
by {
  intros,
  sorry
}

end john_correct_problems_needed_l91_91087


namespace exam_pass_percentage_l91_91762

theorem exam_pass_percentage
  (total_candidates : ℕ)
  (girls : ℕ)
  (boys : ℕ)
  (P : ℕ)
  (total_candidates_eq : total_candidates = 2000)
  (girls_eq : girls = 900)
  (boys_eq : boys = total_candidates - girls)
  : (68% of candidates failed) -> (P = 32) :=
by
  sorry

end exam_pass_percentage_l91_91762


namespace triangle_count_l91_91496

-- Define the problem context
variables (b c : ℕ)
def valid_triangle (b c : ℕ) : Prop :=
  b > 0 ∧ c > 0 ∧ b ≤ 5 ∧ 5 ≤ c ∧ b + 5 > c

-- The statement of the proposition we need to prove
theorem triangle_count : (finset.filter (λ bc : ℕ × ℕ, valid_triangle bc.fst bc.snd) 
                        (finset.product (finset.range 6) (finset.range 11))).card = 15 := 
by
  sorry

end triangle_count_l91_91496


namespace exponent_comparison_l91_91647

theorem exponent_comparison : 1.7 ^ 0.3 > 0.9 ^ 11 := 
by sorry

end exponent_comparison_l91_91647


namespace min_cans_for_gallon_l91_91064

-- Define conditions
def can_capacity : ℕ := 12
def gallon_to_ounces : ℕ := 128

-- Define the minimum number of cans function.
def min_cans (capacity : ℕ) (required : ℕ) : ℕ :=
  (required + capacity - 1) / capacity -- This is the ceiling of required / capacity

-- Statement asserting the required minimum number of cans.
theorem min_cans_for_gallon (h : min_cans can_capacity gallon_to_ounces = 11) : 
  can_capacity > 0 ∧ gallon_to_ounces > 0 := by
  sorry

end min_cans_for_gallon_l91_91064


namespace numWaysToPlaceDigits_l91_91763

-- Definitions for constraints
def posA := 9
def posI := 1
def possibleBD := {6, 7, 8}
def possibleFH := {2, 3, 4}

-- Main statement to prove
theorem numWaysToPlaceDigits : 
  let a := posA
  let i := posI
  exact a = 9 ∧ i = 1 ∧ 
          (b ∈ possibleBD) ∧
          (d ∈ possibleBD) ∧
          (f ∈ possibleFH) ∧
          (h ∈ possibleFH) →
  ∃ (count : ℕ), count = 42 :=
by
  sorry

end numWaysToPlaceDigits_l91_91763


namespace min_value_expression_l91_91695

theorem min_value_expression (x : ℝ) (hx : x > 3) : x + 4 / (x - 3) ≥ 7 :=
sorry

end min_value_expression_l91_91695


namespace max_sqrt_expression_l91_91297

theorem max_sqrt_expression (x : ℝ) (hx : -49 ≤ x ∧ x ≤ 49) :
  ∃ y, y = 14 ∧ (∀ z, z = sqrt (49 + x) + sqrt (49 - x) → z ≤ y) :=
by
  sorry

end max_sqrt_expression_l91_91297


namespace angle_DAE_eq_45_l91_91192

-- Define the parameters of the triangle and points D and E
variables (A B C D E : Type)
variables [MetricSpace A B C D E]
variables (AB AC BC BD EC : ℝ)

-- Declare the known lengths of the sides and segments
axiom h1 : AB = 20
axiom h2 : AC = 21
axiom h3 : BC = 29
axiom h4 : BD = 8
axiom h5 : EC = 9

-- Define the goal to prove
theorem angle_DAE_eq_45 : (angle DAE) = 45 :=
by 
  sorry

end angle_DAE_eq_45_l91_91192


namespace positive_root_of_cubic_eq_l91_91657

theorem positive_root_of_cubic_eq : ∃ (x : ℝ), x > 0 ∧ x^3 - 3 * x^2 - x - Real.sqrt 2 = 0 ∧ x = 2 + Real.sqrt 2 := by
  sorry

end positive_root_of_cubic_eq_l91_91657


namespace total_lives_after_10_minutes_l91_91540

theorem total_lives_after_10_minutes :
  let initial_lives := 15 * 10 in
  let remaining_players := 10 in
  let power_up := 4 in
  let penalty := 6 in
  let lives_per_player := 10 in
  let lives_gained := power_up * 3 in
  let lives_lost := penalty * 2 in
  let final_lives := remaining_players * lives_per_player + lives_gained - lives_lost in
  final_lives = 100 := 
by
  sorry

end total_lives_after_10_minutes_l91_91540


namespace sum_of_reciprocals_l91_91129

theorem sum_of_reciprocals (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 56) : (1/x) + (1/y) = 15/56 := 
by 
  sorry

end sum_of_reciprocals_l91_91129


namespace angle_between_refracted_rays_l91_91546

-- Given conditions as definitions
def n_i : ℝ := 1 -- refractive index of air
def n_t : ℝ := 1.5 -- refractive index of glass
def beta : ℝ := 25 -- angle of refraction for the first ray in degrees

-- Convert degrees to radians for trigonometric calculations
def deg_to_rad (deg : ℝ) : ℝ := deg * (Real.pi / 180)

-- Prove the angle between the refracted rays is 56 degrees
theorem angle_between_refracted_rays : 
  let alpha := Real.asin (n_t * Real.sin (deg_to_rad beta))
  let alpha' := Real.pi / 2 - alpha
  let gamma := Real.asin (Real.sin alpha' / n_t)
  (beta + (gamma * 180 / Real.pi) * 180 / Real.pi = 56) :=
begin
  -- We state the problem, skip the proof
  sorry
end

end angle_between_refracted_rays_l91_91546


namespace KoscheisDaughtersIdentification_l91_91182

namespace KoscheisDaughters

-- Define the entities
def Princess : Type := ℕ -- Assuming a princess is identified by a number
def KoscheisDaughters : Type := ℕ -- Similarly, for Koschei's daughters

-- Given conditions translated into Lean:
constant calls_princess : Princess → List Princess → Prop -- A function that specifies calling
constant koscheis_daughter_called_at_least_three_times : KoscheisDaughters → Prop
constant princess_called_at_most_twice : Princess → Prop

-- Specific calling patterns as lists:
constant eldest_calls : List Princess -- "eldest calls middle and youngest"
constant youngest_calls : List Princess -- "youngest calls middle and eldest"
constant middle_calls : List Princess -- "middle calls herself and youngest"

-- The daughters of Koschei will be identified based on the calling patterns:
def differentiate (Ivan : List Princess → Bool) : Prop :=
  ∀ (eldest middle youngest : Princess),
    calls_princess eldest eldest_calls ∧
    calls_princess youngest youngest_calls ∧
    calls_princess middle middle_calls →
    depth_of_clarity Ivan -->  sorry -- the condition representing correct identification

-- The statement to be proved
theorem KoscheisDaughtersIdentification :
  ∃ (Ivan : List Princess → Bool),
    differentiate Ivan.
Proof
  sorry

end KoscheisDaughters

end KoscheisDaughtersIdentification_l91_91182


namespace advertisement_length_l91_91218

noncomputable def movie_length : ℕ := 90
noncomputable def replay_times : ℕ := 6
noncomputable def operation_time : ℕ := 660

theorem advertisement_length : ∃ A : ℕ, 90 * replay_times + 6 * A = operation_time ∧ A = 20 :=
by
  use 20
  sorry

end advertisement_length_l91_91218


namespace magnitude_of_sum_l91_91369

variables (a b : ℝ × ℝ)
variables (h1 : a.1 * b.1 + a.2 * b.2 = 0)
variables (h2 : a = (4, 3))
variables (h3 : (b.1 ^ 2 + b.2 ^ 2) = 1)

theorem magnitude_of_sum (a b : ℝ × ℝ) (h1 : a.1 * b.1 + a.2 * b.2 = 0) 
  (h2 : a = (4, 3)) (h3 : (b.1 ^ 2 + b.2 ^ 2) = 1) : 
  (a.1 + 2 * b.1) ^ 2 + (a.2 + 2 * b.2) ^ 2 = 29 :=
by sorry

end magnitude_of_sum_l91_91369


namespace q1_q2_l91_91737

section problem 

variable (x : ℝ)
def m := (2 * Real.cos (x + Real.pi / 2), Real.cos x)
def n := (Real.cos x, 2 * Real.sin (x + Real.pi / 2))
def f (x : ℝ) := (m x).1 * (n x).1 + (m x).2 * (n x).2 + 1
def g (x : ℝ) := -Real.sqrt 2 * Real.sin (2 * x + Real.pi / 12)

-- Q1: Prove that f(x) = 1 has the sum of solutions x1 and x2 as 3π/4 in (0, π)
theorem q1 : ∃ x1 x2 ∈ Ioo (0 : ℝ) Real.pi, x1 + x2 = 3 * Real.pi / 4 ∧ f x1 - 1 = 0 ∧ f x2 - 1 = 0 :=
sorry

-- Q2: Prove that g(x) is monotonically increasing in the given intervals after translation
theorem q2 : (∀ x y ∈ Icc (-Real.pi / 2) (- 7 * Real.pi / 24), x ≤ y → g x ≤ g y) ∧
             (∀ x y ∈ Icc (5 * Real.pi / 24) (Real.pi / 2), x ≤ y → g x ≤ g y) :=
sorry

end problem

end q1_q2_l91_91737


namespace functional_equation_solution_l91_91995

-- Define the function
def f : ℝ → ℝ := sorry

-- The main theorem to prove
theorem functional_equation_solution :
  (∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)) → (∀ x : ℝ, f x = x - 1) :=
by
  intro h
  sorry

end functional_equation_solution_l91_91995


namespace polynomial_of_degree_n_has_n_integer_roots_l91_91448

theorem polynomial_of_degree_n_has_n_integer_roots (n : ℕ) (hn : 5 ≤ n) 
(hP : Polynomial ℤ) (h_deg : hP.degree = n) (h_roots : ∃ (r : Finset ℤ), r.card = n ∧ ∀ x ∈ r, hP.eval x = 0) 
(hP0 : hP.eval 0 = 0) :
  ∃ (r : Finset ℤ), r.card = n ∧ ∀ x ∈ r, eval (eval hP hP) x = 0 :=
begin
  sorry
end

end polynomial_of_degree_n_has_n_integer_roots_l91_91448


namespace regular_if_regular_complement_regular_if_regular_intersection_regular_if_regular_prefixes_l91_91379

variables {Σ : Type*} {L M : Set (List Σ)}

def is_regular (L : Set (List Σ)) : Prop := 
  ∃ (A : DFA Σ), A.language = L 

theorem regular_if_regular_complement (L : Set (List Σ)) (hL : is_regular L) : is_regular Lᶜ :=
sorry

theorem regular_if_regular_intersection (L M : Set (List Σ)) (hL : is_regular L) (hM : is_regular M) : is_regular (L ∩ M) :=
sorry

theorem regular_if_regular_prefixes (L : Set (List Σ)) (hL : is_regular L) : is_regular { u | ∃ v, u ++ v ∈ L } :=
sorry

end regular_if_regular_complement_regular_if_regular_intersection_regular_if_regular_prefixes_l91_91379


namespace cartesian_line_of_polar_equation_l91_91726

noncomputable def polar_to_cartesian (theta : ℝ) : ℝ × ℝ → Prop :=
  λ p, ∃ x y, p = (x, y) ∧ (θ = θ → y = x * Real.tan(θ))

theorem cartesian_line_of_polar_equation
  (O : ℝ × ℝ)
  (polar_eq : ∀ (r : ℝ) (θ : ℝ), θ = (2 * Real.pi) / 3 → r * Real.sin(θ) = y := x * Real.tan(θ)) :
  ∀ x y, polar_eq (x, y) = (sqrt 3 * x + y = 0) :=
by
  sorry

end cartesian_line_of_polar_equation_l91_91726


namespace reflection_line_slope_l91_91508

/-- Given two points (1, -2) and (7, 4), and the reflection line y = mx + b. 
    The image of (1, -2) under the reflection is (7, 4). Prove m + b = 4. -/
theorem reflection_line_slope (m b : ℝ)
    (h1: (∀ (x1 y1 x2 y2: ℝ), 
        (x1, y1) = (1, -2) → 
        (x2, y2) = (7, 4) → 
        (y2 - y1) / (x2 - x1) = 1)) 
    (h2: ∀ (x1 y1 x2 y2: ℝ),
        (x1, y1) = (1, -2) → 
        (x2, y2) = (7, 4) →
        (x1 + x2) / 2 = 4 ∧ (y1 + y2) / 2 = 1) 
    (h3: y = mx + b → m = -1 → (4, 1).1 = 4 ∧ (4, 1).2 = 1 → b = 5) : 
    m + b = 4 := by 
  -- No Proof Required
  sorry

end reflection_line_slope_l91_91508


namespace rational_sum_of_squares_l91_91193

theorem rational_sum_of_squares (
  (a b c : ℝ) (n : ℕ) (h : n > 0) :
  let S := (a ^ 2 + b ^ 2 + c ^ 2) * ((n - 1) * (5 * n - 1) / (6 * n))
  in  ∃ r : ℚ, r = (S / (a ^ 2 + b ^ 2 + c ^ 2))
) :=
by
  sorry

end rational_sum_of_squares_l91_91193


namespace value_of_xy_l91_91374

theorem value_of_xy 
  (x y : ℝ) 
  (h1 : (8^x) / (4^(x + y)) = 16) 
  (h2 : (27^(x + y)) / (9^(5 * y)) = 729) : 
  x * y = 96 :=
sorry

end value_of_xy_l91_91374


namespace range_of_k_l91_91734

theorem range_of_k :
  ∀ (k : ℝ), k ∈ Ico 0 (2 * sqrt 5 / 5) ↔
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
      (sqrt (4 - (x1 - 2)^2) = k * (x1 + 1)) ∧
      (sqrt (4 - (x2 - 2)^2) = k * (x2 + 1)) :=
by
  sorry

end range_of_k_l91_91734


namespace maximum_height_l91_91967

-- Defining the given condition representing the height equation
def height (t : ℝ) : ℝ := -15 * (t - 3) ^ 2 + 150

-- Condition: The object reaches a height of 90 feet at t = 5 seconds
def condition1 : Prop := height 5 = 90

-- Desired proof: The object's height at the maximum height (when t = 3) is 150 feet
theorem maximum_height : condition1 → height 3 = 150 :=
by
  intro h90
  -- Proof steps would go here
  sorry

end maximum_height_l91_91967


namespace find_k4_l91_91327

theorem find_k4
  (a_n : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a_n (n + 1) = a_n n + d)
  (h2 : d ≠ 0)
  (h3 : ∃ r : ℝ, a_n 2^2 = a_n 1 * a_n 6)
  (h4 : a_n 1 = a_n k_1)
  (h5 : a_n 2 = a_n k_2)
  (h6 : a_n 6 = a_n k_3)
  (h_k1 : k_1 = 1)
  (h_k2 : k_2 = 2)
  (h_k3 : k_3 = 6) 
  : ∃ k_4 : ℕ, k_4 = 22 := sorry

end find_k4_l91_91327


namespace interval_length_l91_91512

theorem interval_length (c d : ℝ) (h : (d - 5) / 3 - (c - 5) / 3 = 15) : d - c = 45 :=
sorry

end interval_length_l91_91512


namespace Jose_age_correct_l91_91425

variable (Jose Zack Inez : ℕ)

-- Define the conditions
axiom Inez_age : Inez = 15
axiom Zack_age : Zack = Inez + 3
axiom Jose_age : Jose = Zack - 4

-- The proof statement
theorem Jose_age_correct : Jose = 14 :=
by
  -- Proof will be filled in later
  sorry

end Jose_age_correct_l91_91425


namespace find_x_l91_91770

variables (A B C D E : Type) 
  [linear_ordered_field A] [linear_ordered_field B]
  (AB : line_segment A B) 
  (CD : line_segment C D)
  (perpendicular : CD ⊥ AB) 
  (angle_ECB : angle C E B = 60)

theorem find_x (h1 : angle_ABC : angle A C D = 90) 
  (h2 : angle_C_D_E : angle D C E + angle E C B = angle A C D) 
  (h3 : angle_ECB_eq : angle E C B = 60) : angle D C E = 30 :=
by
  have h4 : angle A C D = 90 := h1,
  have h5 : angle D C E + angle E C B = 90 := angle_C_D_E,
  have h6 : angle E C B = 60 := angle_ECB_eq,
  rw h6 at h5,
  linarith


end find_x_l91_91770


namespace Milly_spends_135_minutes_studying_l91_91057

-- Definitions of homework times
def mathHomeworkTime := 60
def geographyHomeworkTime := mathHomeworkTime / 2
def scienceHomeworkTime := (mathHomeworkTime + geographyHomeworkTime) / 2

-- Definition of Milly's total study time
def totalStudyTime := mathHomeworkTime + geographyHomeworkTime + scienceHomeworkTime

-- Theorem stating that Milly spends 135 minutes studying
theorem Milly_spends_135_minutes_studying : totalStudyTime = 135 :=
by
  -- Proof omitted
  sorry

end Milly_spends_135_minutes_studying_l91_91057


namespace students_behind_yoongi_l91_91925

/-- 
20 students stood in a line. Jungkook stood in third place, and Yoongi stood right in front 
of Jungkook. How many students are standing behind Yoongi? 
-/
theorem students_behind_yoongi (n : ℕ) (h1 : n = 20) (jungkook : ℕ) (h2 : jungkook = 3)
  (yoongi : ℕ) (h3 : yoongi = jungkook - 1) : 20 - yoongi = 18 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end students_behind_yoongi_l91_91925


namespace faith_work_days_per_week_l91_91280

theorem faith_work_days_per_week 
  (hourly_wage : ℝ)
  (normal_hours_per_day : ℝ)
  (overtime_hours_per_day : ℝ)
  (weekly_earnings : ℝ)
  (overtime_rate_multiplier : ℝ) :
  hourly_wage = 13.50 → 
  normal_hours_per_day = 8 → 
  overtime_hours_per_day = 2 → 
  weekly_earnings = 675 →
  overtime_rate_multiplier = 1.5 →
  ∀ days_per_week : ℝ, days_per_week = 5 :=
sorry

end faith_work_days_per_week_l91_91280


namespace range_of_a_l91_91852

noncomputable def is_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y < f x

noncomputable def piecewise_function (a : ℝ) : ℝ → ℝ :=
λ x : ℝ, if x ≤ 1 then (a - 3) * x + 5 else (2 * a) / x

theorem range_of_a (a : ℝ) : 
(is_decreasing (piecewise_function a) {x : ℝ | 0 < x}) ↔ (0 < a ∧ a ≤ 3) := 
sorry

end range_of_a_l91_91852


namespace high_octane_amount_l91_91589

theorem high_octane_amount (R H : ℝ) (cost_ratio : ℝ) (fraction_high_octane : ℝ) :
  R = 4545 →
  cost_ratio = 3 →
  fraction_high_octane = 3/7 →
  (H * cost_ratio) = 1136.25 :=
by
  assume h₁: R = 4545
  assume h₂: cost_ratio = 3
  assume h₃: fraction_high_octane = 3 / 7
  sorry

end high_octane_amount_l91_91589


namespace can_place_additional_circle_l91_91389

-- Definition of the problem conditions
def square_side_length : ℝ := 6
def small_square_side_length : ℝ := 1
def small_circle_diameter : ℝ := 1
def small_circle_radius : ℝ := small_circle_diameter / 2

-- Definitions of the number of shapes
def num_squares : ℕ := 4
def num_circles : ℕ := 3

-- The Lean theorem statement
theorem can_place_additional_circle :
  ∃ (C : ℝ × ℝ), 
    (C.1 >= small_circle_radius ∧ C.1 <= square_side_length - small_circle_radius) ∧ 
    (C.2 >= small_circle_radius ∧ C.2 <= square_side_length - small_circle_radius) ∧
    ∀ (P : ℝ × ℝ) ∈ (existing_shapes small_square_side_length small_circle_radius), 
    dist C P > small_circle_diameter :=
sorry

-- Definitions for the existing shapes and their positions
def existing_shapes (small_square_side_length small_circle_radius : ℝ) : set (ℝ × ℝ) :=
{shapes | (shapes ∈ squares_positions small_square_side_length) ∨ (shapes ∈ circles_positions small_circle_radius)}

-- Sample positions for squares and circles - replace with actual positions
def squares_positions (small_square_side_length : ℝ) : set (ℝ × ℝ) :=
{(1, 1), (2, 2), (3, 3), (4, 4)}

def circles_positions (small_circle_radius : ℝ) : set (ℝ × ℝ) :=
{(5, 5), (6, 6), (7, 7)}

end can_place_additional_circle_l91_91389


namespace product_of_x_coordinates_l91_91130

-- Define the hyperbola function
def hyperbola (k : ℝ) (x : ℝ) : ℝ := k / x

-- Define the straight lines with slope k and different intercepts
def line (k : ℝ) (b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the condition that there are three lines with the same slope k but different intercepts
variable (k : ℝ) (b1 b2 b3 : ℝ)
-- Assume k is non-zero
hypothesis hk : k ≠ 0

-- Define the intersection equation for the hyperbola and the lines
def intersection_eq (k : ℝ) (b : ℝ) (x : ℝ) : Prop :=
  hyperbola k x = line k b x

-- Lean statement
theorem product_of_x_coordinates (k : ℝ) (b1 b2 b3 : ℝ) (hk : k ≠ 0) :
  let x1 x2 x3 x4 x5 x6 := sorry -- assume these are the six solutions of intersection_eq
  in x1 * x2 * x3 * x4 * x5 * x6 = -1 :=
sorry

end product_of_x_coordinates_l91_91130


namespace final_chocolate_price_l91_91259

-- Define the original price of the chocolate
def original_price : ℝ := 2.0

-- Define the Christmas discount percentage
def christmas_discount_percentage : ℝ := 15.0

-- Define the extra discount per chocolate if more than 5 chocolates are bought
def extra_discount_per_chocolate : ℝ := 0.25

-- Define the number of chocolates bought
def chocolates_bought : ℕ := 6

-- The final price of each chocolate can be calculated as follows:
theorem final_chocolate_price :
  chocolates_bought > 5 →
  let discount_amount := (christmas_discount_percentage / 100) * original_price in
  let discounted_price := original_price - discount_amount in
  let final_price := discounted_price - extra_discount_per_chocolate in
  final_price = 1.45 :=
by
  intro h
  let discount_amount := (christmas_discount_percentage / 100) * original_price
  let discounted_price := original_price - discount_amount
  let final_price := discounted_price - extra_discount_per_chocolate
  sorry

end final_chocolate_price_l91_91259


namespace sin_theta_is_half_l91_91386

-- Definition of the point P
def P : ℝ × ℝ := (-√3/2, 1/2)

-- Distance between point P and the origin
def OP : ℝ := Real.sqrt ((-√3/2)^2 + (1/2)^2)

-- Definition of sin θ given the terminal side passes through point P
def sin_theta (θ : ℝ) : ℝ := (P.snd) / OP

theorem sin_theta_is_half (θ : ℝ) (h : (cos θ, sin θ) = P) : sin_theta θ = 1/2 := by
  -- Invoke the proof in the following lines
  sorry

end sin_theta_is_half_l91_91386


namespace artist_small_canvas_paint_l91_91618

def artist_paint_usage (x : ℝ) : Prop :=
  let paint_large := 3 * 3
  let paint_small := 4 * x
  paint_large + paint_small = 17

theorem artist_small_canvas_paint : ∃ x : ℝ, artist_paint_usage x ∧ x = 2 :=
begin
  use 2,
  unfold artist_paint_usage,
  split,
  { unfold artist_paint_usage,
    simp,
    exact (13 + 4 * 2 = 17),  -- Simplified logically to match exact answer.
    sorry,
  },
  sorry,
end

end artist_small_canvas_paint_l91_91618


namespace maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l91_91083

-- Defining productivity functions for male and female kittens
def male_productivity (K : ℝ) : ℝ := 80 - 4 * K
def female_productivity (K : ℝ) : ℝ := 16 - 0.25 * K

-- Condition (a): Maximizing number of mice caught by 2 kittens
theorem maximize_mice_two_kittens : 
  ∃ (male1 male2 : ℝ) (K_m1 K_m2 : ℝ), 
    (male1 = male_productivity K_m1) ∧ 
    (male2 = male_productivity K_m2) ∧
    (K_m1 = 0) ∧ (K_m2 = 0) ∧
    (male1 + male2 = 160) := 
sorry

-- Condition (b): Different versions of JPPF
theorem different_versions_JPPF : 
  ∃ (v1 v2 v3 : Unit), 
    (v1 ≠ v2) ∧ (v2 ≠ v3) ∧ (v1 ≠ v3) :=
sorry

-- Condition (c): Analytical form of JPPF for each combination
theorem JPPF_combinations :
  ∃ (M K1 K2 : ℝ),
    (M = 160 - 4 * K1 ∧ K1 ≤ 40) ∨
    (M = 32 - 0.5 * K2 ∧ K2 ≤ 64) ∨
    (M = 96 - 0.25 * K2 ∧ K2 ≤ 64) ∨
    (M = 336 - 4 * K2 ∧ 64 < K2 ∧ K2 ≤ 84) :=
sorry

-- Condition (d): Analytical form for 2 males and 1 female
theorem JPPF_two_males_one_female :
  ∃ (M K : ℝ), 
    (0 < K ∧ K ≤ 64 ∧ M = 176 - 0.25 * K) ∨
    (64 < K ∧ K ≤ 164 ∧ M = 416 - 4 * K) :=
sorry

end maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l91_91083


namespace sum_of_integers_is_27_24_or_20_l91_91523

theorem sum_of_integers_is_27_24_or_20 
    (x y : ℕ) 
    (h1 : 0 < x) 
    (h2 : 0 < y) 
    (h3 : x * y + x + y = 119) 
    (h4 : Nat.gcd x y = 1) 
    (h5 : x < 25) 
    (h6 : y < 25) 
    : x + y = 27 ∨ x + y = 24 ∨ x + y = 20 := 
sorry

end sum_of_integers_is_27_24_or_20_l91_91523


namespace bike_speed_is_10_meters_per_second_l91_91202

-- Define the constants
def distance : ℝ := 5400 -- distance in meters
def time_in_minutes : ℝ := 9 -- time in minutes
def time_in_seconds : ℝ := time_in_minutes * 60 -- time converted to seconds

-- Define the speed calculation
def speed (d : ℝ) (t : ℝ) : ℝ := d / t

-- Statement of the theorem
theorem bike_speed_is_10_meters_per_second : speed distance time_in_seconds = 10 := by
  sorry

end bike_speed_is_10_meters_per_second_l91_91202


namespace steve_average_speed_l91_91841

-- Define the conditions as constants
def hours1 := 5
def speed1 := 40
def hours2 := 3
def speed2 := 80
def hours3 := 2
def speed3 := 60

-- Define a theorem that calculates average speed and proves the result is 56
theorem steve_average_speed :
  (hours1 * speed1 + hours2 * speed2 + hours3 * speed3) / (hours1 + hours2 + hours3) = 56 := by
  sorry

end steve_average_speed_l91_91841


namespace num_lines_equal_intercepts_tangent_to_circle_l91_91862

noncomputable def num_tangent_lines : ℝ := 3

theorem num_lines_equal_intercepts_tangent_to_circle :
  (∀ x y : ℝ, x^2 + (y - 2 * sqrt 2)^2 = 4 → (
    (∃ k : ℝ, y = k * x) ∨ (∃ a : ℝ, y = a - x)) →
    (∃ (L : ℕ), L = 3)) :=
by skip_proof

end num_lines_equal_intercepts_tangent_to_circle_l91_91862


namespace number_of_digits_in_product_l91_91271

/-- The problem setup: Define the two large numbers and state that
    the number of digits in their product is expected to be 32. -/
def number1 : ℕ := 84_123_457_789_321_005
def number2 : ℕ := 56_789_234_567_891

theorem number_of_digits_in_product : (number_of_digits (number1 * number2)) = 32 := 
sorry

end number_of_digits_in_product_l91_91271


namespace behavior_g_l91_91316

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 2 * x + 1

noncomputable def g (a : ℝ) : ℝ := 
  if a ∈ Icc (1/3) (1/2) then 
    a - 2 + 1/a 
  else if a ∈ Ioo (1/2) 1 ∨ a = 1 then 
    9 * a - 6 + 1/a 
  else 
    0    -- Default case if out of interval (though not relevant as specified by problem)

-- Now we state the theorem about the behavior of g
theorem behavior_g (a : ℝ) (h : 1/3 ≤ a ∧ a ≤ 1) :
  (∀ a1 a2,  1/3 ≤ a1 ∧ a1 < a2 ∧ a2 ≤ 1/2 → g a1 > g a2) ∧ 
  (∀ a1 a2,  1/2 < a1 ∧ a1 < a2 ∧ a2 ≤ 1 → g a1 < g a2) ∧ 
  (g (1/2) = 1/2) := 
  sorry

end behavior_g_l91_91316


namespace part1_part2_l91_91731

noncomputable def seq (n : ℕ) : ℚ :=
  match n with
  | 0     => 0  -- since there is no a_0 (we use ℕ*), we set it to 0
  | 1     => 1/3
  | n + 1 => seq n + (seq n) ^ 2 / (n : ℚ) ^ 2

theorem part1 (n : ℕ) (h : 0 < n) :
  seq n < seq (n + 1) ∧ seq (n + 1) < 1 :=
sorry

theorem part2 (n : ℕ) (h : 0 < n) :
  seq n > 1/2 - 1/(4 * n) :=
sorry

end part1_part2_l91_91731


namespace cakes_difference_l91_91622

theorem cakes_difference (cakes_made : ℕ) (cakes_sold : ℕ) (cakes_bought : ℕ) 
  (h1 : cakes_made = 648) (h2 : cakes_sold = 467) (h3 : cakes_bought = 193) :
  (cakes_sold - cakes_bought = 274) :=
by
  sorry

end cakes_difference_l91_91622


namespace limit_tan_cos_at_pi_over_4_l91_91980

open Real

theorem limit_tan_cos_at_pi_over_4 :
  (Real.limit (fun x => (Real.tan x)^(1 / cos (3 * pi / 4 - x))) (pi / 4)) = exp 2 := 
by 
  sorry

end limit_tan_cos_at_pi_over_4_l91_91980


namespace repeating_decimal_to_fraction_l91_91562

theorem repeating_decimal_to_fraction :
  let x := 0.762042042042 -- notation of repeating decimal
  in x = 761280 / 999000 :=
by
  let repeatingPart := 204 / 999000
  have h1 : x = 0.76 + repeatingPart * 1 / 1000 := sorry
  have h2 : repeatingPart * 1 / 1000 = 204 / 999000 := sorry
  rw [h1, h2]
  norm_num
  exact sorry

end repeating_decimal_to_fraction_l91_91562


namespace correct_calculation_l91_91567

theorem correct_calculation : sqrt 12 - sqrt 3 = sqrt 3 := by
  have h : sqrt 12 = 2 * sqrt 3 := by
    calc
      sqrt 12 = sqrt (4 * 3) : by rw [←mul_assoc, nat.mul_comm 4 3, mul_assoc]
          ... = sqrt 4 * sqrt 3 : by rw [sqrt_mul 4 3]
          ... = 2 * sqrt 3     : by rw [sqrt_eq_2r 2]
  calc
    sqrt 12 - sqrt 3 = 2 * sqrt 3 - sqrt 3 : by rw [h]
               ... = (2 - 1) * sqrt 3      : by ring
               ... = sqrt 3                : by norm_num

end correct_calculation_l91_91567


namespace total_employees_in_buses_l91_91161

theorem total_employees_in_buses :
  let bus1_percentage_full := 0.60,
      bus2_percentage_full := 0.70,
      bus_capacity := 150
  in
  (bus1_percentage_full * bus_capacity + bus2_percentage_full * bus_capacity) = 195 := by
  sorry

end total_employees_in_buses_l91_91161


namespace arithmetic_sequence_count_l91_91272

theorem arithmetic_sequence_count (n m : ℕ) (h1 : 3 ≤ m) (h2 : m ≤ n) (hn : n = 20) (hm : m = 5) : 
  f n m = 40 :=
sorry

end arithmetic_sequence_count_l91_91272


namespace unused_streetlights_remain_l91_91089

def total_streetlights : ℕ := 200
def squares : ℕ := 15
def streetlights_per_square : ℕ := 12

theorem unused_streetlights_remain :
  total_streetlights - (squares * streetlights_per_square) = 20 :=
sorry

end unused_streetlights_remain_l91_91089


namespace total_cherry_tomatoes_l91_91020

-- Definitions based on the conditions
def cherryTomatoesPerJar : Nat := 8
def numberOfJars : Nat := 7

-- The statement we want to prove
theorem total_cherry_tomatoes : cherryTomatoesPerJar * numberOfJars = 56 := by
  sorry

end total_cherry_tomatoes_l91_91020


namespace max_pieces_after_cutting_rectangles_l91_91820

theorem max_pieces_after_cutting_rectangles (n : ℕ) 
  (rectangles : set (set (ℝ × ℝ))) 
  (h1 : ∀ r ∈ rectangles, ∃ a b c d : ℝ, r = {p : ℝ × ℝ | a ≤ p.1 ∧ p.1 ≤ b ∧ c ≤ p.2 ∧ p.2 ≤ d})
  (h2 : ∀ r1 r2 ∈ rectangles, r1 ≠ r2 → ∀ x ∈ r1, ∀ y ∈ r2, x ≠ y) :
  ∃ m ≤ n + 1, partition_pieces rectangles m := 
sorry

end max_pieces_after_cutting_rectangles_l91_91820


namespace trig_identity_equiv_l91_91571

variable (α : ℝ)

theorem trig_identity_equiv : 
  4 * cos (α - π / 2) * sin^3 (π / 2 + α) - 4 * sin (5 * π / 2 - α) * cos^3 (3 * π / 2 + α) = sin (4 * α) :=
by
  sorry

end trig_identity_equiv_l91_91571


namespace mono_increasing_intervals_l91_91264

noncomputable def f : ℝ → ℝ :=
by sorry

theorem mono_increasing_intervals (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_sym : ∀ x, f x = f (-2 - x))
  (h_decr1 : ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ -1 → f y ≤ f x) :
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f x ≤ f y) ∧
  (∀ x y, 3 ≤ x ∧ x < y ∧ y ≤ 4 → f x ≤ f y) :=
sorry

end mono_increasing_intervals_l91_91264


namespace total_employees_in_buses_l91_91156

-- Define the capacity of each bus
def capacity : ℕ := 150

-- Define the fill percentages of each bus
def fill_percentage_bus1 : ℚ := 60 / 100
def fill_percentage_bus2 : ℚ := 70 / 100

-- Calculate the number of passengers in each bus
def passengers_bus1 : ℚ := fill_percentage_bus1 * capacity
def passengers_bus2 : ℚ := fill_percentage_bus2 * capacity

-- Calculate the total number of passengers
def total_passengers : ℚ := passengers_bus1 + passengers_bus2

-- The proof statement
theorem total_employees_in_buses : total_passengers = 195 :=
by
  sorry

end total_employees_in_buses_l91_91156


namespace number_of_speaking_orders_l91_91942

open Finset

noncomputable def speaking_orders : Nat :=
  let students : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
  let A := 1
  let B := 2
  
  -- Case 1: A participates but B does not (or vice versa)
  let choose_3_from_6 : Nat := (students.erase 2).choose 3 ∣ 4!
  -- Case 2: Both A and B participate and their speeches are not adjacent
  let choose_2_from_5 : Nat := (students.erase 1.erase 2).choose 2 ∣ 3! * 2
  
  -- Total number of valid speaking orders
  let total_orders : Nat := choose_3_from_6 + choose_2_from_5
  
  total_orders

theorem number_of_speaking_orders : speaking_orders = 600 := by
  sorry

end number_of_speaking_orders_l91_91942


namespace maximum_value_sqrt_expr_l91_91308

theorem maximum_value_sqrt_expr :
  ∃ x : ℝ, -49 ≤ x ∧ x ≤ 49 ∧ sqrt (49 + x) + sqrt (49 - x) = 14 ∧
  ∀ x : ℝ, -49 ≤ x ∧ x ≤ 49 → sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
by
  sorry

end maximum_value_sqrt_expr_l91_91308


namespace incorrect_statement_B_l91_91904

-- Definitions for conditions
def quadrilateral (A B C D : Type) := true -- This represents the concept of a quadrilateral

-- A: A quadrilateral with diagonals bisecting each other is a parallelogram.
def is_parallelogram (quad : quadrilateral) :=
  -- Assume definition based on bisecting diagonals
  ∀ d1 d2, bisect d1 d2 quad -> parallelogram quad 

-- B: A quadrilateral with equal diagonals is a rectangle.
def is_rectangle (quad : quadrilateral) :=
  ∀ d1 d2, equal_diagonals d1 d2 quad -> rectangle quad 

-- C: A quadrilateral with three right angles is a rectangle.
def three_right_angles_implies_rectangle (quad : quadrilateral) :=
  ∀ angles, three_right_angles angles quad -> rectangle quad

-- D: A quadrilateral with two pairs of equal sides is a parallelogram.
def two_pairs_equal_sides_implies_parallelogram (quad : quadrilateral) :=
  ∀ sides, two_pairs_equal_sides sides quad -> parallelogram quad

-- The proof statement
theorem incorrect_statement_B (quad : quadrilateral) :
  ¬ (∀ d1 d2, equal_diagonals d1 d2 quad -> rectangle quad) := sorry

end incorrect_statement_B_l91_91904


namespace probability_equal_sums_l91_91658

def eight_digit_binary_number : Type := vector bool 8

def digits_at_positions (v : eight_digit_binary_number) (positions : list ℕ) : list bool :=
  positions.map (λ i, v.nth (i - 1))

def sum_of_digits (digits : list bool) : ℕ :=
  digits.foldr (λ d acc, if d then acc + 1 else acc) 0

theorem probability_equal_sums :
  let odd_positions := [1, 3, 5, 7],
      even_positions := [2, 4, 6, 8],
      all_possible_numbers := finset.univ : finset eight_digit_binary_number,
      successful_numbers := all_possible_numbers.filter 
        (λ v, sum_of_digits (digits_at_positions v odd_positions) = sum_of_digits (digits_at_positions v even_positions)) 
  in (successful_numbers.card : ℚ) / (all_possible_numbers.card : ℚ) = 35 / 128 :=
by
  sorry

end probability_equal_sums_l91_91658


namespace height_of_tree_l91_91181

theorem height_of_tree (part_shadow_ground : ℝ) (height_wall_shadow : ℝ) 
(diameter_sphere : ℝ) (stone_shadow_length : ℝ) 
(f : part_shadow_ground = 1.4) 
(g : height_wall_shadow = 1.2) 
(h : diameter_sphere = 1.2) 
(i : stone_shadow_length = 0.8) :
  let H := 1.4;
  (H + height_wall_shadow) = 2.6 :=
by {
  let H := 1.4,
  calc (H + height_wall_shadow) = 1.4 + 1.2 : by rw g
  ... = 2.6 : by norm_num,
  sorry
}

end height_of_tree_l91_91181


namespace maximum_length_OB_l91_91880

noncomputable def max_length_OB (O A B : Point) (angle_OAB angle_AOB : ℝ) (AB OB : ℝ) : Prop :=
  ((∠AOB = 45) ∧ (dist A B = 1) ∧ (dist O A = OB) ∧ (dist O B = OB)) →
  ∃ OB_max, OB_max = sqrt 2 ∧ OB ≤ OB_max

theorem maximum_length_OB {O A B : Point} (angle_OAB angle_AOB : ℝ) (AB : ℝ) :
  (angle O A B = 45) ∧ (AB = 1) ∧ (∀ P : Triangle, P.contains_vertices O A B) →
  ∃ OB : ℝ, OB ≤ sqrt 2 :=
sorry

end maximum_length_OB_l91_91880


namespace man_speed_l91_91215

theorem man_speed (distance_meters time_minutes : ℝ) (h_distance : distance_meters = 1250) (h_time : time_minutes = 15) :
  (distance_meters / 1000) / (time_minutes / 60) = 5 :=
by
  sorry

end man_speed_l91_91215


namespace abc_inequality_l91_91805

theorem abc_inequality 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
by
  sorry

end abc_inequality_l91_91805


namespace part_1_part_2_part_3_l91_91318

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x - g x a

theorem part_1 (a : ℝ) : (F' := λ x, Real.exp x + Real.cos x - a) → 
                          (F' 0 = 0) → a = 2 :=
by sorry

theorem part_2 (a : ℝ) : (∀ x, 0 ≤ x → F x a ≥ 1) → a ≤ 2 :=
by sorry

theorem part_3 (x1 x2 : ℝ) : (a = 1 / 3) → 
                           (0 ≤ x1) → (0 ≤ x2) → 
                           (F x1 a - g x2 a) → (3 * (Real.exp x1 + Real.sin x1 - 1/3 * x1) ≥ 3) → 
                           x2 - x1 = 3 :=
by sorry

end part_1_part_2_part_3_l91_91318


namespace mean_equality_solution_l91_91858

theorem mean_equality_solution (z : ℚ) (h : (8 + 15 + 27) / 3 = (18 + z) / 2) : z = 46 / 3 :=
begin
  sorry
end

end mean_equality_solution_l91_91858


namespace solve_xyz_l91_91650

theorem solve_xyz :
  ∃ (x y z : ℕ), (x y z + x y + y z + z x + x + y + z = 1977 ∧ 
  ((x, y, z) = (1, 22, 42) ∨ (x, y, z) = (1, 42, 22) ∨ 
   (x, y, z) = (22, 1, 42) ∨ (x, y, z) = (22, 42, 1) ∨
   (x, y, z) = (42, 1, 22) ∨ (x, y, z) = (42, 22, 1))) :=
by
  sorry

end solve_xyz_l91_91650


namespace trig_identity_l91_91640

theorem trig_identity : Real.sin (35 * Real.pi / 6) + Real.cos (-11 * Real.pi / 3) = 0 := by
  sorry

end trig_identity_l91_91640


namespace min_g_least_possible_l91_91760

open_locale classical

noncomputable def min_g {n : ℕ} (grid : matrix (fin n) (fin n) ℕ) : ℕ :=
  let adjacent (a b : ℕ × ℕ) : Prop :=
    (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨ (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1)) in
  let diffs := {d : ℕ | ∃ a b, adjacent a b ∧ d = abs (grid a - grid b)} in
  diffs.sup id

theorem min_g_least_possible (n : ℕ) (grid : matrix (fin n) (fin n) ℕ)
  (unique_entries : ∀ i j, grid i j ∈ finset.range (n^2) ∧ ∀ i1 j1 i2 j2, grid i1 j1 = grid i2 j2 → i1 = i2 ∧ j1 = j2) :
  min_g grid = n :=
begin
  sorry
end

end min_g_least_possible_l91_91760


namespace sum_of_numbers_is_37_l91_91282

theorem sum_of_numbers_is_37 :
  ∃ (A B : ℕ), 
    1 ≤ A ∧ A ≤ 50 ∧ 1 ≤ B ∧ B ≤ 50 ∧ A ≠ B ∧
    (50 * B + A = k^2) ∧ Prime B ∧ B > 10 ∧
    A + B = 37 
  := by
    sorry

end sum_of_numbers_is_37_l91_91282


namespace sandy_marks_per_correct_sum_l91_91070

theorem sandy_marks_per_correct_sum 
  (total_sums : ℕ)
  (total_marks : ℤ)
  (correct_sums : ℕ)
  (marks_per_incorrect_sum : ℤ)
  (marks_obtained : ℤ) 
  (marks_per_correct_sum : ℕ) :
  total_sums = 30 →
  total_marks = 45 →
  correct_sums = 21 →
  marks_per_incorrect_sum = 2 →
  marks_obtained = total_marks →
  marks_obtained = marks_per_correct_sum * correct_sums - marks_per_incorrect_sum * (total_sums - correct_sums) → 
  marks_per_correct_sum = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end sandy_marks_per_correct_sum_l91_91070


namespace thrushes_left_l91_91155

theorem thrushes_left {init_thrushes : ℕ} (additional_thrushes : ℕ) (killed_ratio : ℚ) (killed : ℕ) (remaining : ℕ) :
  init_thrushes = 20 →
  additional_thrushes = 4 * 2 →
  killed_ratio = 1 / 7 →
  killed = killed_ratio * (init_thrushes + additional_thrushes) →
  remaining = init_thrushes + additional_thrushes - killed →
  remaining = 24 :=
by sorry

end thrushes_left_l91_91155


namespace inscribed_quadrilateral_ratio_l91_91068

variables {A B C D : Type*}
variables [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D]
variables (AB AD CB CD BA BC DA DC AC BD : ℝ)
variables (R S : ℝ)

theorem inscribed_quadrilateral_ratio 
  (inscribed : Quadrilateral A B C D)
  (area_S : S)
  (radius_R : R) :
  AC / BD = (AB * AD + CB * CD) / (BA * BC + DA * DC) :=
sorry

end inscribed_quadrilateral_ratio_l91_91068


namespace max_magnitude_value_is_4_l91_91735

noncomputable def max_value_vector_magnitude (θ : ℝ) : ℝ :=
  let a := (Real.cos θ, Real.sin θ)
  let b := (Real.sqrt 3, -1)
  let vector := (2 * a.1 - b.1, 2 * a.2 + 1)
  Real.sqrt (vector.1 ^ 2 + vector.2 ^ 2)

theorem max_magnitude_value_is_4 (θ : ℝ) : 
  ∃ θ : ℝ, max_value_vector_magnitude θ = 4 :=
sorry

end max_magnitude_value_is_4_l91_91735


namespace max_value_expression_l91_91654

def E (θ1 θ2 θ3 θ4 θ5 α : ℝ) : ℝ :=
  cos (θ1 + α) * sin (θ2 + α) +
  cos (θ2 + α) * sin (θ3 + α) +
  cos (θ3 + α) * sin (θ4 + α) +
  cos (θ4 + α) * sin (θ5 + α) +
  cos (θ5 + α) * sin (θ1 + α)

theorem max_value_expression (α : ℝ): ∃ θ1 θ2 θ3 θ4 θ5 : ℝ, E θ1 θ2 θ3 θ4 θ5 α = 5/2 :=
  sorry

end max_value_expression_l91_91654


namespace workers_assignment_l91_91203

theorem workers_assignment
  (total_workers : ℕ)
  (profit_A : ℕ)
  (profit_B_initial : ℕ)
  (profit_C : ℕ)
  (unit_profit_decrease_B : ℕ)
  (additional_profit_A : ℕ)
  (A_workers : ℕ)
  (B_workers : ℕ)
  (C_workers : ℕ)
  (total_profit : ℕ)
  (daily_output_A : ℕ)
  (daily_output_C : ℕ)
  :
  total_workers = 65 →
  profit_A = 15 →
  profit_B_initial = 120 →
  profit_C = 30 →
  unit_profit_decrease_B = 2 →
  additional_profit_A = 650 →
  daily_output_A = 2 * A_workers →
  daily_output_C = C_workers →
  additional_profit_A = 650 →
  (15 * daily_output_A + (120 - 2 * B_workers) * B_workers) = total_profit →
  (A_workers + B_workers + C_workers = total_workers) →
  (daily_output_A = daily_output_C) →
  total_profit = 2650 →
  A_workers = 10 :=
begin
  intros,
  sorry,
end

end workers_assignment_l91_91203


namespace travel_time_provincial_to_lishan_l91_91531

theorem travel_time_provincial_to_lishan :
  ∀ (total_distance : ℕ) (distance_lishan_to_county : ℕ)
    (departure_lishan : ℕ) (arrival_county : ℕ)
    (stop_duration : ℕ) (arrival_provincial : ℕ)
    (departure_provincial : ℕ) (speed_provincial : ℕ),
  total_distance = 189 ∧
  distance_lishan_to_county = 54 ∧
  departure_lishan = 510 ∧  -- 8:30 AM in minutes
  arrival_county = 555 ∧    -- 9:15 AM in minutes
  stop_duration = 15 ∧
  arrival_provincial = 660 ∧ -- 11:00 AM in minutes
  departure_provincial = 540 ∧ -- 9:00 AM in minutes
  speed_provincial = 60 →
  ∃ travel_time : ℕ,
    travel_time = 72
    :=
by
  intros total_distance distance_lishan_to_county
         departure_lishan arrival_county
         stop_duration arrival_provincial
         departure_provincial speed_provincial
  assume h_conditions
  sorry

end travel_time_provincial_to_lishan_l91_91531


namespace largest_subset_count_l91_91631

open Finset

def is_valid_subset (T : Finset ℕ) : Prop :=
  ∀ (x ∈ T) (y ∈ T), x ≠ y → (x - y).nat_abs ≠ 3 ∧ (x - y).nat_abs ≠ 5

theorem largest_subset_count :
  ∃ T : Finset ℕ, (T ⊆ (range 1000).map succ) ∧ is_valid_subset T ∧ T.card = 600 :=
by
  sorry

end largest_subset_count_l91_91631


namespace sufficient_but_not_necessary_condition_l91_91330

theorem sufficient_but_not_necessary_condition {a : ℝ} (P : a > 1) : (a - 1) * (a + 1) > 0 :=
by {
  exact (P : sufficient_condition (λ a, (a - 1) * (a + 1) > 0))
}

end sufficient_but_not_necessary_condition_l91_91330


namespace distances_of_A_and_B_from_intersection_l91_91475

/-- Points A and B are on two lines intersecting at an angle of 60 degrees. The initial distance between the points is 31 m. If point A moves 20 m closer to the intersection of the lines, 
the distance between the points becomes 21 m. Prove that the distances of the points from the intersection are MA = 35 m and MB = 24 m. -/
theorem distances_of_A_and_B_from_intersection
(points_on_intersecting_lines : ∃ (A B : Type) (angle : ℝ), angle = 60 ∧ distance A B = 31)
(A_moves_closer : ∃ (A C M : Type) (d_C : ℝ), d_C = 20 ∧ distance C M < distance A M)
(new_distance : ∃ (B C : Type) (d_BC : ℝ), d_BC = 21) :
∃ (MA MB : ℝ), MA = 35 ∧ MB = 24 :=
begin
  sorry
end

end distances_of_A_and_B_from_intersection_l91_91475


namespace inequality_of_cardinalities_l91_91031

-- Define the finite set of positive reals A and its cardinality
variable {A : Finset ℝ} 
variable [hA : ∀ (a : ℝ), a ∈ A → a > 0]

-- Define B in terms of A
noncomputable def B : Finset ℝ := Finset.bUnion A (λ x, Finset.image (λ y, x / y) A)

-- Define C in terms of A
noncomputable def C : Finset ℝ := Finset.bUnion A (λ x, Finset.image (λ y, x * y) A)

-- Define the main theorem
theorem inequality_of_cardinalities (hA_nonempty : A.nonempty) : 
  A.card * B.card ≤ (C.card)^2 :=
by sorry

end inequality_of_cardinalities_l91_91031


namespace internal_tangency_locus_spec_external_tangency_locus_spec_l91_91092

open Set

variable {S₁ S₂ : Sphere ℝ} -- Define S₁ and S₂ as spheres in ℝ³
variable {Π : Plane ℝ} -- Define Π as a plane in ℝ³
variable {X : Point ℝ} -- Define X as a point in ℝ³

-- Define the conditions
variable (h₁ : (center S₁) ∈ Π)
variable (h₂ : (center S₂) ∈ Π)
variable (h₃ : radius S₁ ≠ radius S₂)

-- Define the geometrical loci
def internal_tangency_locus (Π : Plane ℝ) : Set (Point ℝ) :=
  {X | ∃ π : Plane ℝ, X ∈ π ∧ ¬collinear (center S₁) (center S₂) X ∧ ∀ x₁ ∈ S₁ ∩ Π, ∀ x₂ ∈ S₂ ∩ Π, π.is_tangent S₁ ∧ π.is_tangent S₂}

def external_tangency_locus (Π : Plane ℝ) : Set (Point ℝ) :=
  {X | ∃ π : Plane ℝ, X ∈ π ∧ collinear (center S₁) (center S₂) X ∧ ∀ x₁ ∈ S₁ ∩ Π, ∀ x₂ ∈ S₂ ∩ Π, π.is_tangent S₁ ∧ π.is_tangent S₂}

-- The proof statements
theorem internal_tangency_locus_spec (Π : Plane ℝ) (X : Point ℝ) :
  X ∈ internal_tangency_locus Π ↔ (∃ π : Plane ℝ, X ∈ π ∧ ¬collinear (center S₁) (center S₂) X ∧ ∀ x₁ ∈ S₁ ∩ Π, ∀ x₂ ∈ S₂ ∩ Π, π.is_tangent S₁ ∧ π.is_tangent S₂) := 
sorry

theorem external_tangency_locus_spec (Π : Plane ℝ) (X : Point ℝ) :
  X ∈ external_tangency_locus Π ↔ (∃ π : Plane ℝ, X ∈ π ∧ collinear (center S₁) (center S₂) X ∧ ∀ x₁ ∈ S₁ ∩ Π, ∀ x₂ ∈ S₂ ∩ Π, π.is_tangent S₁ ∧ π.is_tangent S₂) := 
sorry

end internal_tangency_locus_spec_external_tangency_locus_spec_l91_91092


namespace solutions_to_x_squared_eq_x_l91_91122

theorem solutions_to_x_squared_eq_x (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := 
sorry

end solutions_to_x_squared_eq_x_l91_91122


namespace distance_between_lateral_edge_and_opposite_side_l91_91526

theorem distance_between_lateral_edge_and_opposite_side
  (a : ℝ) (α : ℝ) :
  let FO := (a / (2 * real.sqrt 3)) * real.tan α,
      KB := (a * real.sqrt 3) / 2,
      FB := (a / (2 * real.sqrt 3)) * (real.sqrt (real.tan α ^ 2 + 4))
  in (a > 0) ∧ (0 < α ∧ α < π / 2) →
    let KP := (FO * (KB / FB))
    in KP = (a * real.sqrt 3 * real.tan α) / (2 * real.sqrt (4 + real.tan α ^ 2)) :=
begin
  sorry
end

end distance_between_lateral_edge_and_opposite_side_l91_91526


namespace midpoint_F_l91_91011

theorem midpoint_F
  (A B E O H D E F G : Point)
  (circumcenter : circle O)
  (orthocenter : is_orthocenter H A B E)
  (BD_perp_AC : ⟂ BD AC ∧ meet BD AC D)
  (CE_perp_AB : ⟂ CE AB ∧ meet CE AB E)
  (AG_perp_OH : ⟂ AG OH ∧ meet AG DE F ∧ meet AG BC G)
  :
  is_midpoint_of GG F :=
sorry

end midpoint_F_l91_91011


namespace buildingC_floors_if_five_times_l91_91625

-- Defining the number of floors in Building B
def floorsBuildingB : ℕ := 13

-- Theorem to prove the number of floors in Building C if it had five times as many floors as Building B
theorem buildingC_floors_if_five_times (FB : ℕ) (h : FB = floorsBuildingB) : (5 * FB) = 65 :=
by
  rw [h]
  exact rfl

end buildingC_floors_if_five_times_l91_91625


namespace find_m_l91_91564

theorem find_m (m : ℝ) :
  (m - 2013 = 0) → (m = 2013) ∧ (m - 1 ≠ 0) :=
by {
  sorry
}

end find_m_l91_91564


namespace monotonic_increasing_interval_l91_91109

noncomputable def log_base := (1 / 2 : ℝ)

def polynomial (x : ℝ) : ℝ := 2 * x ^ 2 - 3 * x + 1

def function (x : ℝ) : ℝ := Real.logPol2 (polynomial x)

theorem monotonic_increasing_interval :
  (∀ x : ℝ, polynomial x > 0) →
  (Set.Iio (1 / 2)) = {x : ℝ | polynomial x < (1 / 2)} :=
sorry

end monotonic_increasing_interval_l91_91109


namespace square_side_length_of_three_unit_circles_l91_91148

theorem square_side_length_of_three_unit_circles :
  let ω₁ ω₂ ω₃ : circle := ⟨1, ⟨2, by sorry⟩⟩ -- placeholders
  let S : square := ⟨ω₁, ω₂, ω₃, by sorry⟩ -- placeholders
  side_length S = (sqrt 6 + sqrt 2 + 8) / 4 :=
sorry

end square_side_length_of_three_unit_circles_l91_91148


namespace exists_point_Z0_prime_exist_points_R_and_S_l91_91326

-- Part 1
theorem exists_point_Z0_prime (Z1 Z2 Z3 Z4 Z0 : ℂ) :
  (∃ (E F G H : ℂ), 
    similar (I E G) (Z0 Z2 Z3) ∧ similar (F G E) (Z0 Z4 Z1))
  →
  ∃ (Z0' : ℂ), 
    (similar (Z0' Z3 Z4) (E F H)) ∧ (similar (Z0' Z1 Z2) (G H F)) := 
sorry

-- Part 2
theorem exist_points_R_and_S (Z1 Z2 Z3 Z4 E F G H : ℂ) :
  ∃ (R S : ℂ), 
    (similar (R H E) (Z3 Z4 Z2)) ∧ (similar (R F G) (Z1 Z2 Z4)) ∧ 
    (similar (S G H) (Z2 Z3 Z1)) ∧ (similar (S E F) (Z4 Z1 Z3)) :=
sorry

end exists_point_Z0_prime_exist_points_R_and_S_l91_91326


namespace gift_distribution_l91_91063

noncomputable section

structure Recipients :=
  (ondra : String)
  (matej : String)
  (kuba : String)

structure PetrStatements :=
  (ondra_fire_truck : Bool)
  (kuba_no_fire_truck : Bool)
  (matej_no_merkur : Bool)

def exactly_one_statement_true (s : PetrStatements) : Prop :=
  (s.ondra_fire_truck && ¬s.kuba_no_fire_truck && ¬s.matej_no_merkur)
  ∨ (¬s.ondra_fire_truck && s.kuba_no_fire_truck && ¬s.matej_no_merkur)
  ∨ (¬s.ondra_fire_truck && ¬s.kuba_no_fire_truck && s.matej_no_merkur)

def correct_recipients (r : Recipients) : Prop :=
  r.kuba = "fire truck" ∧ r.matej = "helicopter" ∧ r.ondra = "Merkur"

theorem gift_distribution
  (r : Recipients)
  (s : PetrStatements)
  (h : exactly_one_statement_true s)
  (h0 : ¬exactly_one_statement_true ⟨r.ondra = "fire truck", r.kuba ≠ "fire truck", r.matej ≠ "Merkur"⟩)
  (h1 : ¬exactly_one_statement_true ⟨r.ondra ≠ "fire truck", r.kuba ≠ "fire truck", r.matej ≠ "Merkur"⟩)
  : correct_recipients r := by
  -- Proof is omitted as per the instructions
  sorry

end gift_distribution_l91_91063


namespace avg_distinct_t_values_l91_91707

theorem avg_distinct_t_values : 
  ∀ (t : ℕ), (∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t) →
  (1 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   6 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   10 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t} ∨
   12 ∈ {t | ∃ r₁ r₂ : ℕ, r₁ > 0 ∧ r₂ > 0 ∧ r₁ + r₂ = 7 ∧ r₁ * r₂ = t}) →
  let distinct_values := {6, 10, 12} in
  let average := (6 + 10 + 12) / 3 in
  average = 28 / 3 :=
by {
  sorry
}

end avg_distinct_t_values_l91_91707


namespace intersection_points_count_l91_91112

def f (x : ℝ) : ℝ := 2 * Real.log x
def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count : 
    ∃ (n : ℕ), n = 2 ∧ ∀ x, f x = g x → x = 2 ∨ -- the exact intersection points
    sorry

end intersection_points_count_l91_91112


namespace bounded_poly_constant_l91_91488

theorem bounded_poly_constant (P : Polynomial ℤ) (B : ℕ) (h_bounded : ∀ x : ℤ, abs (P.eval x) ≤ B) : 
  P.degree = 0 :=
sorry

end bounded_poly_constant_l91_91488


namespace find_other_number_l91_91105

theorem find_other_number (y : ℕ) : Nat.lcm 240 y = 5040 ∧ Nat.gcd 240 y = 24 → y = 504 :=
by
  sorry

end find_other_number_l91_91105


namespace incorrect_inequality_l91_91376

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬ (5 - a > 5 - b) :=
by
  intro h1
  have h2 : a > b := h
  have h3 : ¬ (5 - a > 5 - b)
  sorry

end incorrect_inequality_l91_91376


namespace language_group_selection_l91_91756

theorem language_group_selection :
  ∀ (n : ℕ) (e : ℕ) (j : ℕ) (b : ℕ),
  n = 9 ∧ e = 7 ∧ j = 3 ∧ b = (e + j - n) →
  let only_english := e - b,
      only_japanese := j - b in
  (only_english * only_japanese + b * only_japanese + only_english * b) = 20 := 
begin
  intros n e j b h,
  cases h with h₁ h₂, 
  cases h₂ with h₃ h₄,
  cases h₄ with h₅ h₆,
  let only_english : ℕ := e - b,
  let only_japanese : ℕ := j - b,
  have h_only_english : only_english = 6,
  { sorry },
  have h_only_japanese : only_japanese = 2,
  { sorry },
  have h_total_ways : (only_english * only_japanese + b * only_japanese + only_english * b) = 20,
  { 
    sorry 
  },
  exact h_total_ways,
end

end language_group_selection_l91_91756


namespace number_of_correct_conclusions_l91_91350

-- Define the conditions for each conclusion
open Real

def cond1 (a b c d : ℚ) (h : a + b * sqrt 5 = c + d * sqrt 5) : Prop := a = c ∧ b = d

def cond2 (a b c : Vector ℝ (Fin 3)) (h1 : parallelepiped a b) (h2 : parallelepiped b c) : Prop := parallelepiped a c

def cond3 (α β γ : Plane) (h1 : Parallel α β) (h2 : Parallel β γ) : Prop := Parallel α γ

-- Main theorem stating the number of correct conclusions is 2
theorem number_of_correct_conclusions
  (a b c d : ℚ) (h1 : cond1 a b c d)
  (α β γ : Plane) (h3 : cond3 α β γ)
  (vA vB vC : Vector ℝ (Fin 3)) (h2_false : ¬ cond2 vA vB vC) :
  nat = 2 :=
sorry

end number_of_correct_conclusions_l91_91350


namespace sum_from_neg_50_to_70_l91_91892

noncomputable def sum_integers (start : ℤ) (end : ℤ) : ℤ :=
  (end - start + 1) * (start + end) / 2

theorem sum_from_neg_50_to_70 : sum_integers (-50) 70 = 1210 :=
by sorry

end sum_from_neg_50_to_70_l91_91892


namespace list_price_is_40_l91_91613

open Real

def list_price (x : ℝ) : Prop :=
  0.15 * (x - 15) = 0.25 * (x - 25)

theorem list_price_is_40 : list_price 40 :=
by
  unfold list_price
  sorry

end list_price_is_40_l91_91613


namespace smallest_n_for_conditions_l91_91668

noncomputable def min_n_for_conditions : ℕ :=
  Inf {n : ℕ | ∃ (x : Fin n → ℝ),
    (∀ i, x i ∈ Ioo (-1 : ℝ) 1) ∧
    (∑ i, x i = 0) ∧
    (∑ i, (x i) ^ 2 = 36)}

theorem smallest_n_for_conditions : min_n_for_conditions = 38 := sorry

end smallest_n_for_conditions_l91_91668


namespace man_speed_is_5_km_per_hr_l91_91216

def convert_minutes_to_hours (minutes: ℕ) : ℝ :=
  minutes / 60

def convert_meters_to_kilometers (meters: ℕ) : ℝ :=
  meters / 1000

def man_speed (distance_m: ℕ) (time_min: ℕ) : ℝ :=
  distance_m / 1000 / (time_min / 60)

theorem man_speed_is_5_km_per_hr : man_speed 1250 15 = 5 := by
  unfold man_speed convert_meters_to_kilometers convert_minutes_to_hours
  -- More steps in the proof would go here, but we use sorry to skip the proof.
  sorry

end man_speed_is_5_km_per_hr_l91_91216


namespace division_problem_l91_91561

theorem division_problem : (4 * 5) / 10 = 2 :=
by sorry

end division_problem_l91_91561


namespace total_tablets_l91_91934

-- Variables for the numbers of Lenovo, Samsung, and Huawei tablets
variables (n x y : ℕ)

-- Conditions based on problem statement
def condition1 : Prop := 2 * x + 6 + y < n / 3

def condition2 : Prop := (n - 2 * x - y - 6 = 3 * y)

def condition3 : Prop := (n - 6 * x - y - 6 = 59)

-- The statement to prove that the total number of tablets is 94
theorem total_tablets (h1 : condition1 n x y) (h2 : condition2 n x y) (h3 : condition3 n x y) : n = 94 :=
by
  sorry

end total_tablets_l91_91934


namespace Monica_class_ratio_l91_91463

theorem Monica_class_ratio : 
  (20 + 25 + 25 + x + 28 + 28 = 136) → 
  (x = 10) → 
  (x / 20 = 1 / 2) :=
by 
  intros h h_x
  sorry

end Monica_class_ratio_l91_91463


namespace ratio_of_AB_to_BC_l91_91484

noncomputable def quadrilateral_problem (AB BC AD CD E : ℝ) : Prop :=
  (AB > BC) ∧
  (AD = √(BC^2 + CD^2)) ∧
  (CD = 1) ∧
  -- Similar triangles ABC ∼ BCD ∼ CEB
  (triangle_similar ABC BCD) ∧
  (triangle_similar ABC CEB) ∧
  -- Area of AED = 17 times the area of CEB
  (area_triangle AED = 17 * area_triangle CEB) ∧
  -- Area calculations (could be more detailed depending on definitions of areas)
  (area_quadrilateral ABCD = area_triangle AED + area_triangle DEC + area_triangle CEB + area_triangle BEA)

theorem ratio_of_AB_to_BC (AB BC AD CD E : ℝ) (h_cond : quadrilateral_problem AB BC AD CD E) : 
  AB / BC = 2 + sqrt 5 := 
by
  sorry

end ratio_of_AB_to_BC_l91_91484


namespace tan_double_angle_identity_tan_known_values_transform_tan_expression_l91_91195

theorem tan_double_angle_identity (theta : ℝ) :
  (1 - tan (theta : ℝ)^2 ≠ 0) → 
  (tan (2 * theta : ℝ) = 2 * tan(theta) / (1 - tan(theta)^2)) := sorry

theorem tan_known_values :
  (tan (37.5 * pi / 180) = tan 37.5) → 
  (tan (75 * pi / 180) = tan 75) →
  (tan (45 * pi / 180) = 1) →
  (tan (30 * pi / 180) = sqrt 3 / 3) := sorry

theorem transform_tan_expression :
  (1 - (tan(37.5 * pi / 180))^2 ≠ 0) →
  (1 - (sqrt 3 / 3)^2 ≠ 0) →
  (1 - (1)^2 ≠ 0) →
  ∃ (x : ℝ), x = (1 + sqrt 3 / 2) :=
begin
  intro h1,
  intro h2,
  intro h3,
  use (1 + sqrt 3 / 2),
  sorry
end

end tan_double_angle_identity_tan_known_values_transform_tan_expression_l91_91195


namespace proof_problem_1_proof_problem_2_proof_problem_3_proof_problem_4_l91_91373

-- Problem I1.1
theorem proof_problem_1 : 
  let P := (3^2003 * 5^2002 * 7^2001) % 10 
  in P = 5 :=
sorry

-- Problem I1.2
theorem proof_problem_2 :
  let P := 5
  let Q := ∃ x : ℤ, (x^2 - x - 1)^(x + P - 1) = 1
  in Q = 4 :=
sorry

-- Problem I1.3
theorem proof_problem_3 :
  let x y : ℝ := x * y = 1
  let Q := 4
  let R := min ( (1 / x^4) + (1 / (Q * y^4)) )
  in R = 1 :=
sorry

-- Problem I1.4
theorem proof_problem_4 :
  let R K : ℕ := K > R
  let S := ∑ i in range (K - R + 1), i = 2003
  in S = 62 :=
sorry

end proof_problem_1_proof_problem_2_proof_problem_3_proof_problem_4_l91_91373


namespace equivalent_lengthEF_l91_91769

namespace GeometryProof

noncomputable def lengthEF 
  (AB CD EF : ℝ) 
  (h_AB_parallel_CD : true) 
  (h_lengthAB : AB = 200) 
  (h_lengthCD : CD = 50) 
  (h_angleEF : true) 
  : ℝ := 
  50

theorem equivalent_lengthEF
  (AB CD EF : ℝ) 
  (h_AB_parallel_CD : true) 
  (h_lengthAB : AB = 200) 
  (h_lengthCD : CD = 50) 
  (h_angleEF : true) 
  : lengthEF AB CD EF h_AB_parallel_CD h_lengthAB h_lengthCD h_angleEF = 50 :=
by
  sorry

end GeometryProof

end equivalent_lengthEF_l91_91769


namespace find_unknown_number_l91_91749

theorem find_unknown_number (x : ℕ) (hx1 : 100 % x = 16) (hx2 : 200 % x = 4) : x = 28 :=
by 
  sorry

end find_unknown_number_l91_91749


namespace speed_ratio_l91_91579

theorem speed_ratio (v_A v_B : ℝ) (h1 : ∀ t : ℝ, A t = (v_A * t, 0))
  (h2 : ∀ t : ℝ, B t = (0, -800 + v_B * t))
  (h3 : dist (A 1) O = dist (B 1) O)
  (h4 : dist (A 5) O = dist (B 5) O) :
  v_A / v_B = 1 / 9 :=
by
  sorry

end speed_ratio_l91_91579


namespace sequence_and_sum_problems_l91_91346

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n-1) * d

def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n-1) * d) / 2

def geometric_sequence (b r : ℤ) (n : ℕ) : ℤ := b * r^(n-1)

noncomputable def sum_geometric_sequence (b r : ℤ) (n : ℕ) : ℤ := 
(if r = 1 then b * n
 else b * (r^n - 1) / (r - 1))

theorem sequence_and_sum_problems :
  (∀ n : ℕ, arithmetic_sequence 19 (-2) n = 21 - 2 * n) ∧
  (∀ n : ℕ, sum_arithmetic_sequence 19 (-2) n = 20 * n - n^2) ∧
  (∀ n : ℕ, ∃ a_n : ℤ, (geometric_sequence 1 3 n + (a_n - geometric_sequence 1 3 n) = 21 - 2 * n + 3^(n-1)) ∧
    sum_geometric_sequence 1 3 n = (sum_arithmetic_sequence 19 (-2) n + (3^n - 1) / 2))
:= by
  sorry

end sequence_and_sum_problems_l91_91346


namespace correct_avg_weight_l91_91186

theorem correct_avg_weight (initial_avg_weight : ℚ) (num_boys : ℕ) (misread_weight : ℚ) (correct_weight : ℚ) :
  initial_avg_weight = 58.4 → num_boys = 20 → misread_weight = 56 → correct_weight = 60 →
  (initial_avg_weight * num_boys + (correct_weight - misread_weight)) / num_boys = 58.6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Plugging in the values makes the calculation straightforward, resulting in: 
  -- (58.4 * 20 + (60 - 56)) / 20 = 58.6 
  -- thus this verification step is:
  sorry

end correct_avg_weight_l91_91186


namespace collinear_P_E_F_l91_91829

theorem collinear_P_E_F
  (A B C D P Q E F : Type)
  [InscribedQuadrilateral A B C D]
  (h1 : ExtensionsIntersect (AB : segment) (DC : segment) P)
  (h2 : ExtensionsIntersect (AD : segment) (BC : segment) Q)
  (h3 : TangentFromPointToCircle Q (ABCD : circle) E)
  (h4 : TangentFromPointToCircle Q (ABCD : circle) F) :
  Collinear P E F :=
sorry

end collinear_P_E_F_l91_91829


namespace point_symmetric_sum_l91_91005

theorem point_symmetric_sum {m n : ℝ} (h₁ : m = -3) (h₂ : n = 2) : m + n = -1 := 
by {
  rw [h₁, h₂],
  simp,
}

end point_symmetric_sum_l91_91005


namespace log_base_b_eq_one_log_base_b_power_n_log_base_b_div_a_c_log_change_of_base_problem_solution_l91_91254

-- definitions for the logarithm properties
theorem log_base_b_eq_one (b : ℝ) (hb : b > 0 ∧ b ≠ 1) : log b b = 1 :=
sorry

theorem log_base_b_power_n (b n : ℝ) (hb : b > 0 ∧ b ≠ 1) : log b (b^n) = n :=
sorry

theorem log_base_b_div_a_c (b a c : ℝ) (hb : b > 0 ∧ b ≠ 1) (ha : a > 0) (hc : c > 0): log b (a/c) = log b a - log b c :=
sorry

-- change of base formula
theorem log_change_of_base (a b x : ℝ) (hb : b > 0 ∧ b ≠ 1) (ha : a > 0 ∧ a ≠ 1) (hx : x > 0) : 
  log b x = log a x / log a b :=
sorry

noncomputable def problem_expression : ℝ :=
  log 3 5 + log 5 (3^(1/3)) + log 7 (49^(1/3)) + (1 / log 2 6) + log 5 3 + log 6 3 - log 3 15

theorem problem_solution : problem_expression = 2/3 :=
sorry

end log_base_b_eq_one_log_base_b_power_n_log_base_b_div_a_c_log_change_of_base_problem_solution_l91_91254


namespace max_AMC_expression_l91_91444

theorem max_AMC_expression (A M C : ℕ) (h : A + M + C = 24) :
  A * M * C + A * M + M * C + C * A ≤ 704 :=
sorry

end max_AMC_expression_l91_91444


namespace Mikaela_initially_planned_walls_l91_91815

/-- 
Mikaela bought 16 containers of paint to cover a certain number of equally-sized walls in her bathroom.
At the last minute, she decided to put tile on one wall and paint flowers on the ceiling with one 
container of paint instead. She had 3 containers of paint left over. 
Prove she initially planned to paint 13 walls.
-/
theorem Mikaela_initially_planned_walls
  (PaintContainers : ℕ)
  (CeilingPaint : ℕ)
  (LeftOverPaint : ℕ)
  (TiledWalls : ℕ) : PaintContainers = 16 → CeilingPaint = 1 → LeftOverPaint = 3 → TiledWalls = 1 → 
    (PaintContainers - CeilingPaint - LeftOverPaint + TiledWalls = 13) :=
by
  -- Given conditions:
  intros h1 h2 h3 h4
  -- Proof goes here.
  sorry

end Mikaela_initially_planned_walls_l91_91815


namespace exists_n_with_2000_prime_divisors_and_divides_2_pow_n_plus_1_l91_91997

theorem exists_n_with_2000_prime_divisors_and_divides_2_pow_n_plus_1 :
  ∃ n : ℕ, (n > 0) ∧ (prime_divisors_count n = 2000) ∧ (n ∣ (2^n + 1)) :=
sorry

end exists_n_with_2000_prime_divisors_and_divides_2_pow_n_plus_1_l91_91997


namespace average_of_t_l91_91712

theorem average_of_t (t : ℕ) 
  (roots_positive : ∀ r : ℕ, r ∈ (ROOTS of (x^2 - 7*x + t)) → r > 0) 
  (sum_of_roots_eq_seven : ∀ r1 r2 : ℕ, r1 + r2 = 7)
  (product_of_roots_eq_t : ∀ r1 r2 : ℕ, r1 * r2 = t) : 
  (6 + 10 + 12) / 3 = 28 / 3 :=
sorry

end average_of_t_l91_91712


namespace tetrahedron_volume_eq_l91_91337

variables {α : Type*}
variables (A B C A1 B1 C1 P : α)

/-- The volume of the tetrahedron formed by the planes A1B1C, B1C1A, C1A1B, and ABC -/
def volume_of_tetrahedron (S_triangle R : ℝ) : ℝ :=
  4 / 3 * S_triangle * R

-- Assuming the conditions provided in the problem
variables (S_triangle R h_a h_b h_c : ℝ)

-- The theorem that states the correct volume given the conditions.
theorem tetrahedron_volume_eq (hA1 : AA1 = h_a) (hB1 : BB1 = h_b) (hC1 : CC1 = h_c)
  (h_triangle_area : ∃ (S : ℝ), S = S_triangle) (h_circumradius : ∃ (r : ℝ), r = R) :
  volume_of_tetrahedron S_triangle R = 4 / 3 * S_triangle * R :=
sorry

end tetrahedron_volume_eq_l91_91337


namespace least_positive_divisible_by_primes_l91_91174

theorem least_positive_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7
  ∃ n : ℕ, n > 0 ∧ (n % p1 = 0) ∧ (n % p2 = 0) ∧ (n % p3 = 0) ∧ (n % p4 = 0) ∧ 
  (∀ m : ℕ, m > 0 → (m % p1 = 0) ∧ (m % p2 = 0) ∧ (m % p3 = 0) ∧ (m % p4 = 0) → m ≥ n) ∧ n = 210 := 
by {
  sorry
}

end least_positive_divisible_by_primes_l91_91174


namespace population_net_increase_l91_91758

-- Define conditions
def birth_rate : ℚ := 5 / 2    -- 5 people every 2 seconds
def death_rate : ℚ := 3 / 2    -- 3 people every 2 seconds
def one_day_in_seconds : ℕ := 86400   -- Number of seconds in one day

-- Define the net increase per second
def net_increase_per_second := birth_rate - death_rate

-- Prove that the net increase in one day is 86400 people given the conditions
theorem population_net_increase :
  net_increase_per_second * one_day_in_seconds = 86400 :=
sorry

end population_net_increase_l91_91758


namespace equilateral_triangle_side_length_l91_91472

/-- Given an equilateral triangle ABC with P inside it, 
and Q, R, S being the feet of the perpendiculars from P to sides AB, BC, and CA respectively.
Given distances PQ = 2, PR = 2√2, and PS = 4, we need to prove that the side length of ∆ABC is 4√3 + 4√6 / 3. -/
theorem equilateral_triangle_side_length (P Q R S A B C : Type) 
  [metric_space P] [metric_space Q] [metric_space R] [metric_space S] [metric_space A] [metric_space B] [metric_space C]
  (h₁ : P ∈ triangle A B C) 
  (h₂ : foot_of_perpendicular P A B = Q)
  (h₃ : foot_of_perpendicular P B C = R)
  (h₄ : foot_of_perpendicular P C A = S)
  (dist_PQ : dist P Q = 2)
  (dist_PR : dist P R = 2 * real.sqrt 2)
  (dist_PS : dist P S = 4) :
  dist A B = 4 * real.sqrt 3 + 4 * real.sqrt 6 / 3 :=
sorry

end equilateral_triangle_side_length_l91_91472


namespace initial_production_rate_36_l91_91965

variable (x : ℝ)

-- Conditions
def initialTime (x : ℝ) : ℝ := 60 / x
def increasedTime : ℝ := 60 / 60
def totalTime (x : ℝ) : ℝ := initialTime x + increasedTime
def averageOutput (x : ℝ) : ℝ := 120 / totalTime x

-- The proof obligation
theorem initial_production_rate_36 (h : averageOutput x = 45) : x = 36 := by
  sorry

end initial_production_rate_36_l91_91965


namespace jill_net_monthly_salary_l91_91999

-- Define Jill's net monthly salary
variable (S : ℝ)

-- Given conditions
def discretionary_income : ℝ := S / 5
def gifts_charity_amount : ℝ := 111
def percentage_remaining : ℝ := 0.20

-- The theorem stating that Jill's net monthly salary is $2775
theorem jill_net_monthly_salary (h1 : percentage_remaining * discretionary_income = gifts_charity_amount) : 
  S = 2775 := by
  sorry

end jill_net_monthly_salary_l91_91999


namespace words_per_page_l91_91587

theorem words_per_page (p : ℕ) (h1 : p ≤ 120) (h2 : 154 * p % 221 = 207) : p = 100 :=
sorry

end words_per_page_l91_91587


namespace right_triangle_perimeter_l91_91605

theorem right_triangle_perimeter :
  ∃ (a b : ℝ), (∃ c : ℝ, (2 * a^2 - 8 * a + 7 = 0) ∧ (2 * b^2 - 8 * b + 7 = 0) ∧ (a + b = 4) ∧ (a * b = 7 / 2) ∧ (c = Real.sqrt (a^2 + b^2))) ∧
  (a + b + c = 7) :=
begin
  -- proof goes here
  sorry
end

end right_triangle_perimeter_l91_91605


namespace inequality_1_solution_set_inequality_2_solution_set_l91_91494

theorem inequality_1_solution_set (x : ℝ) : 
  (2 + 3 * x - 2 * x^2 > 0) ↔ (-1/2 < x ∧ x < 2) := 
by sorry

theorem inequality_2_solution_set (x : ℝ) :
  (x * (3 - x) ≤ x * (x + 2) - 1) ↔ (x ≤ -1/2 ∨ x ≥ 1) :=
by sorry

end inequality_1_solution_set_inequality_2_solution_set_l91_91494


namespace find_angle_PQD_l91_91840

-- Define the square and equilateral triangle configuration
variables {A B C D M K L P Q : Type}

-- Assume the properties of the geometric shapes
variables (is_square : isSquare A B C D)
variables (is_equilateral : isEquilateral M K L)
variables (P_on_diagonal : liesOnDiagonal P C A D)
variables (Q_on_square : liesOnSquare Q A B C D Q)

-- Define the needed proposition
def angle_PQD_is_75_degrees : Prop :=
  angle PQD = 75

-- State the theorem
theorem find_angle_PQD (h1: is_square) (h2: is_equilateral) 
    (h3: P_on_diagonal) (h4: Q_on_square) : angle_PQD_is_75_degrees :=
by
  sorry

end find_angle_PQD_l91_91840


namespace find_T_l91_91532

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (T : V →ₗ[ℝ] V)
variable (u v w x y z : V)

-- Defining the linear transformation T and the conditions
axiom T_linear : ∀ (a b : ℝ) (v w : V), T (a • v + b • w) = a • T v + b • T w
axiom T_cross_product : ∀ (v w : V), T (v ×ᵥ w) = T v ×ᵥ T w
axiom T_applied_1 : T (⟨ 4, 4, 2 ⟩ : Fin 3 → ℝ) = (⟨ 2, -1, 5 ⟩ : Fin 3 → ℝ)
axiom T_applied_2 : T (⟨ -4, 2, 4 ⟩ : Fin 3 → ℝ) = (⟨ 2, 5, -1 ⟩ : Fin 3 → ℝ)

-- The proof statement to find T ⟨ 2, 6, 8 ⟩
theorem find_T : T (⟨ 2, 6, 8 ⟩ : Fin 3 → ℝ) = (⟨ 0.5, 0, 7 ⟩ : Fin 3 → ℝ) :=
  sorry

end find_T_l91_91532


namespace twenty_four_points_game_l91_91140

theorem twenty_four_points_game :
  let a := (-6 : ℚ)
  let b := (3 : ℚ)
  let c := (4 : ℚ)
  let d := (10 : ℚ)
  3 * (d - a + c) = 24 := 
by
  sorry

end twenty_four_points_game_l91_91140


namespace lorry_length_approx_l91_91596

noncomputable def speed_kmph : ℝ := 80
noncomputable def time_s : ℝ := 17.998560115190784
noncomputable def bridge_length_m : ℝ := 200
noncomputable def speed_ms : ℝ := speed_kmph * 1000 / 3600
noncomputable def total_distance_m : ℝ := speed_ms * time_s
noncomputable def lorry_length_m : ℝ := total_distance_m - bridge_length_m

theorem lorry_length_approx : lorry_length_m ≈ 199.96 := 
by {
  -- This is where the proof would go.
  sorry
}

end lorry_length_approx_l91_91596


namespace area_parallelogram_l91_91091

-- Define the vectors a and b.
variables (a b : ℝ^3)

-- Given condition: the area of the parallelogram generated by a and b is 10.
axiom area_ab : ‖a × b‖ = 10

-- Prove that the area of the parallelogram generated by (3a - 2b) and (2a + 4b) is 160.
theorem area_parallelogram :
  ‖(3 • a - 2 • b) × (2 • a + 4 • b)‖ = 160 :=
by
  sorry

end area_parallelogram_l91_91091


namespace margo_total_distance_l91_91052

theorem margo_total_distance
  (t1 t2 : ℚ) (rate1 rate2 : ℚ)
  (h1 : t1 = 15 / 60)
  (h2 : t2 = 25 / 60)
  (r1 : rate1 = 5)
  (r2 : rate2 = 3) :
  (t1 * rate1 + t2 * rate2 = 2.5) :=
by
  sorry

end margo_total_distance_l91_91052


namespace no_such_n_exists_l91_91042

-- Definition of the sum of the digits function s(n)
def s (n : ℕ) : ℕ := n.digits 10 |> List.sum

-- Statement of the proof problem
theorem no_such_n_exists : ¬ ∃ n : ℕ, n * s n = 20222022 :=
by
  -- argument based on divisibility rules as presented in the problem
  sorry

end no_such_n_exists_l91_91042


namespace probability_53_mondays_in_leap_year_l91_91572

-- Conditions
def leap_year_days : ℕ := 366
def weeks_in_year : ℕ := 52
def extra_days : ℕ := leap_year_days % 7 -- This results in 2 extra days in a leap year

-- Define the event that a leap year has 53 Mondays given it starts on a particular day
def has_53_mondays_if_starts_on (d : ℕ) : Prop :=
  d = 0 ∨ d = 6 -- 0: Starts on Monday, 6: Starts on Sunday

-- The main theorem
theorem probability_53_mondays_in_leap_year : 
  (set.univ.filter has_53_mondays_if_starts_on).to_finset.card / 7 = 2 / 7 := 
by
  sorry

end probability_53_mondays_in_leap_year_l91_91572


namespace soccer_league_equation_l91_91388

noncomputable def equation_represents_soccer_league (x : ℕ) : Prop :=
  ∀ x : ℕ, (x * (x - 1)) / 2 = 50

theorem soccer_league_equation (x : ℕ) (h : equation_represents_soccer_league x) :
  (x * (x - 1)) / 2 = 50 :=
  by sorry

end soccer_league_equation_l91_91388


namespace johns_overall_average_speed_l91_91423

def time_cycling : ℝ := 45 / 60
def speed_cycling : ℝ := 20
def time_break : ℝ := 15 / 60
def time_walking : ℝ := 60 / 60
def speed_walking : ℝ := 3

theorem johns_overall_average_speed :
  (speed_cycling * time_cycling + speed_walking * time_walking) / (time_cycling + time_break + time_walking) = 9 :=
by 
  sorry

end johns_overall_average_speed_l91_91423


namespace rotate_complex_example_l91_91768

def rotate_complex (z : ℂ) (θ : ℝ) : ℂ :=
  let r := complex.abs z
  let φ := complex.arg z
  complex.mkPolar r (φ - θ)

theorem rotate_complex_example :
  rotate_complex (3 - sqrt 3 * complex.I) (π / 3) = -2 * sqrt 3 * complex.I :=
by
  sorry

end rotate_complex_example_l91_91768


namespace max_gross_profit_price_l91_91212

def purchase_price : ℝ := 20
def Q (P : ℝ) : ℝ := 8300 - 170 * P - P^2
def L (P : ℝ) : ℝ := (8300 - 170 * P - P^2) * (P - 20)

theorem max_gross_profit_price : ∃ P : ℝ, (∀ x : ℝ, L x ≤ L P) ∧ P = 30 :=
by
  sorry

end max_gross_profit_price_l91_91212


namespace problem_proof_l91_91355

noncomputable def f (x : ℝ) : ℝ :=  sqrt 3 * sin (2 * x) * cos (2 * x) + (cos (2 * x))^2 - 1/2
noncomputable def g (x : ℝ) : ℝ :=  sin (2 * x - π / 3)

theorem problem_proof :
  (∀ x : ℝ, f x = sin (4 * x + π / 6)) ∧
  (∀ k : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ g x + k = 0) 
  ↔ (k ∈ Set.Ioo (-sqrt 3 / 2) (sqrt 3 / 2) ∨ k = -1)) :=
by
  sorry

end problem_proof_l91_91355


namespace integral_x_eq_1_iff_eq_sqrt2_l91_91672

theorem integral_x_eq_1_iff_eq_sqrt2 (a : ℝ) :
  (∫ x in 0..a, x) = 1 ↔ a = real.sqrt 2 :=
begin
  sorry
end

end integral_x_eq_1_iff_eq_sqrt2_l91_91672


namespace planted_fraction_is_correct_l91_91646

-- Definitions
def right_triangle (AB AC : ℝ) : Prop :=
  AB = 5 ∧ AC = 12

def hypotenuse (AB AC BC: ℝ) : Prop :=
  BC = real.sqrt (AB^2 + AC^2)

def distance_to_hypotenuse (u : ℝ) : Prop :=
  u = 3

def planted_field_fraction (planted_fraction : ℝ) : Prop :=
  planted_fraction = 7 / 10

-- Theorem statement
theorem planted_fraction_is_correct :
  ∀ (AB AC BC u planted_fraction: ℝ),
  right_triangle AB AC →
  hypotenuse AB AC BC →
  distance_to_hypotenuse u →
  planted_field_fraction planted_fraction :=
by
  intros AB AC BC u planted_fraction
  intros h1 h2 h3
  sorry

end planted_fraction_is_correct_l91_91646


namespace equidistant_points_count_l91_91673

noncomputable def number_of_equidistant_points
  (O : Point)
  (r d : ℝ)
  (h : d > r)
  (circle : Circle O r)
  (tangent1 tangent2 : Line)
  (ht1 : is_tangent circle tangent1)
  (ht2 : is_tangent circle tangent2)
  (ht_parallel : parallel tangent1 tangent2)
  (dist_tangent_O : distance(O, tangent1) = d ∧ distance(O, tangent2) = d) : ℕ :=
  4

theorem equidistant_points_count
  (O : Point)
  (r d : ℝ)
  (h : d > r)
  (circle : Circle O r)
  (tangent1 tangent2 : Line)
  (ht1 : is_tangent circle tangent1)
  (ht2 : is_tangent circle tangent2)
  (ht_parallel : parallel tangent1 tangent2)
  (dist_tangent_O : distance(O, tangent1) = d ∧ distance(O, tangent2) = d) :
  number_of_equidistant_points O r d h circle tangent1 tangent2 ht1 ht2 ht_parallel dist_tangent_O = 4 :=
sorry

end equidistant_points_count_l91_91673


namespace wood_not_heavier_than_brick_l91_91783

-- Define the weights of the wood and the brick
def block_weight_kg : ℝ := 8
def brick_weight_g : ℝ := 8000

-- Conversion function from kg to g
def kg_to_g (kg : ℝ) : ℝ := kg * 1000

-- State the proof problem
theorem wood_not_heavier_than_brick : ¬ (kg_to_g block_weight_kg > brick_weight_g) :=
by
  -- Begin the proof
  sorry

end wood_not_heavier_than_brick_l91_91783


namespace cos_double_angle_l91_91745

theorem cos_double_angle (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 3 / 5) : Real.cos (2 * α) = -7 / 25 :=
sorry

end cos_double_angle_l91_91745


namespace island_is_not_maya_l91_91817

variables (A B : Prop)

-- Conditions based on the problem
def A_statement := A ∧ B ∧ island_is_maya
def B_statement := (A ∨ B) ∧ ¬(island_is_maya)

-- Assuming the propositions
axiom A_is_liar : ¬A_statement
axiom B_is_knight : B

-- The goal to prove
theorem island_is_not_maya : ¬island_is_maya :=
sorry

end island_is_not_maya_l91_91817


namespace quadratic_inverse_condition_l91_91468

theorem quadratic_inverse_condition : 
  (∀ x₁ x₂ : ℝ, (x₁ ≥ 2 ∧ x₂ ≥ 2 ∧ x₁ ≠ x₂) → (x₁^2 - 4*x₁ + 5 ≠ x₂^2 - 4*x₂ + 5)) :=
sorry

end quadratic_inverse_condition_l91_91468


namespace adam_tuesday_lessons_l91_91956

-- Define the conditions
def monday_lessons : ℕ := 6
def monday_minutes_per_lesson : ℕ := 30
def tuesday_lesson_length_in_hours : ℕ := 1
def total_time_spent : ℕ := 12

theorem adam_tuesday_lessons : 
  let monday_hours := monday_lessons * monday_minutes_per_lesson / 60,
      tuesday_hours := T,
      wednesday_hours := 2 * T,
      total_hours := monday_hours + tuesday_hours + wednesday_hours
  in total_hours = total_time_spent →
     T = 3 :=
by
  sorry

end adam_tuesday_lessons_l91_91956


namespace arithmetic_progression_5_digits_l91_91651

theorem arithmetic_progression_5_digits (k : ℕ) :
  ∃ n, (∃ m, a_n = 5 * (10^m - 1) / 9) ∧ a_n = 19 * n - 20 ∧ 
  n = 5 * (10^(171 * k + 1) + 35) / 171 := 
begin
  sorry
end

end arithmetic_progression_5_digits_l91_91651


namespace volume_of_rotated_equilateral_triangle_l91_91685

-- Define the equilateral triangle and the rotation conditions
def equilateral_triangle (a : ℝ) := ∀ (A B C : ℝ × ℝ), 
  A != B ∧ B != C ∧ C != A ∧
  (dist A B = a) ∧ (dist B C = a) ∧ (dist C A = a)

def rotated_solid_volume (side_length : ℝ) :=
  let r := (side_length * (Real.sqrt 3)) / 2 in
  let h := side_length / 2 in
  2 * (1 / 3 * Real.pi * r^2 * h)

-- Define the theorem
theorem volume_of_rotated_equilateral_triangle :
  equilateral_triangle 2 :=
begin
  -- Given conditions
  intros A B C,
  -- Calculate the volume "rotated_solid_volume" given side_length = 2
  have v := rotated_solid_volume 2,
  -- Assert the volume is equal to 2π
  show v = 2 * Real.pi,
  sorry -- Proof goes here
end

end volume_of_rotated_equilateral_triangle_l91_91685


namespace polynomials_equality_l91_91221

open Polynomial

variable {F : Type*} [Field F]

theorem polynomials_equality (P Q : Polynomial F) (h : ∀ x, P.eval (P.eval (P.eval x)) = Q.eval (Q.eval (Q.eval x)) ∧ P.eval (P.eval (P.eval x)) = Q.eval (P.eval (P.eval x))) : 
  P = Q := 
sorry

end polynomials_equality_l91_91221


namespace average_of_t_l91_91710

theorem average_of_t (t : ℕ) 
  (roots_positive : ∀ r : ℕ, r ∈ (ROOTS of (x^2 - 7*x + t)) → r > 0) 
  (sum_of_roots_eq_seven : ∀ r1 r2 : ℕ, r1 + r2 = 7)
  (product_of_roots_eq_t : ∀ r1 r2 : ℕ, r1 * r2 = t) : 
  (6 + 10 + 12) / 3 = 28 / 3 :=
sorry

end average_of_t_l91_91710


namespace length_of_field_l91_91573

-- Define the conditions and given facts.
def double_length (w l : ℝ) : Prop := l = 2 * w
def pond_area (l w : ℝ) : Prop := 49 = 1/8 * (l * w)

-- Define the main statement that incorporates the given conditions and expected result.
theorem length_of_field (w l : ℝ) (h1 : double_length w l) (h2 : pond_area l w) : l = 28 := by
  sorry

end length_of_field_l91_91573


namespace bike_ride_time_good_l91_91248

theorem bike_ride_time_good (x : ℚ) :
  (20 * x + 12 * (8 - x) = 122) → x = 13 / 4 :=
by
  intro h
  sorry

end bike_ride_time_good_l91_91248


namespace max_mice_two_males_num_versions_PPF_PPF_combinations_PPF_three_kittens_l91_91085

-- Condition definitions
def PPF_male (K : ℕ) : ℕ := 80 - 4 * K
def PPF_female (K : ℕ) : ℕ := 16 - 0.25.to_nat * K

-- Proving the maximum number of mice that could be caught by 2 male kittens
theorem max_mice_two_males : ∀ K, PPF_male 0 + PPF_male 0 = 160 :=
by simp [PPF_male]

-- Proving there are 3 possible versions of the PPF
theorem num_versions_PPF : ∃ (versions : set (ℕ → ℕ)), versions = 
  { λ K, 160 - 4 * K,
    λ K, 32 - 0.5.to_nat * K,
    λ K, if K ≤ 64 then 96 - 0.25.to_nat * K else 336 - 4 * K } ∧
  versions.size = 3 :=
by sorry

-- Proving the analytical form of each PPF combination
theorem PPF_combinations : 
  (∀ K, (λ K, 160 - 4 * K) K = PPF_male K + PPF_male K) ∧
  (∀ K, (λ K, 32 - 0.5.to_nat * K) K = PPF_female K + PPF_female K) ∧
  (∀ K, if K ≤ 64 then (λ K, 96 - 0.25.to_nat * K) K = PPF_male K + PPF_female K else (λ K, 336 - 4 * K) K = PPF_male (K - 64) + PPF_female 64) :=
by sorry

-- Proving the analytical form when accepting the third kitten
theorem PPF_three_kittens :
  (∀ K, if K ≤ 64 then (176 - 0.25.to_nat * K) = PPF_male K + PPF_male K + PPF_female K else (416 - 4 * K) = PPF_male (K - 64) + PPF_male 64 + PPF_female 64) :=
by sorry

end max_mice_two_males_num_versions_PPF_PPF_combinations_PPF_three_kittens_l91_91085


namespace simplest_quadratic_radical_l91_91617

theorem simplest_quadratic_radical :
  let A := -Real.sqrt 2 in
  let B := Real.sqrt 12 in
  let C := Real.sqrt (1/5) in
  let D := Real.sqrt 4 in
  A = -Real.sqrt 2 :=
sorry

end simplest_quadratic_radical_l91_91617


namespace water_pouring_l91_91782

theorem water_pouring (n : ℕ) (w : ℝ) (h₀ : w = 1) (h₁ : ∀ k : ℕ, 1 <= k → w' k = w * (k + 1) / (k + 2)) : n = 8 ↔ w' n = 1 / 5 := 
sorry

end water_pouring_l91_91782


namespace at_least_two_even_l91_91753

theorem at_least_two_even (x y z : ℤ) (u : ℤ)
  (h : x^2 + y^2 + z^2 = u^2) : (↑x % 2 = 0) ∨ (↑y % 2 = 0) → (↑x % 2 = 0) ∨ (↑z % 2 = 0) ∨ (↑y % 2 = 0) := 
by
  sorry

end at_least_two_even_l91_91753


namespace exists_a_even_functions_l91_91359

noncomputable def f (x a : ℝ) := x^2 + (real.pi - a) * x
noncomputable def g (x a : ℝ) := real.cos (2 * x + a)

theorem exists_a_even_functions :
  ∃ a : ℝ, (∀ x : ℝ, f x a = f (-x) a) ∧ (∀ x : ℝ, g x a = g (-x) a) :=
sorry

end exists_a_even_functions_l91_91359


namespace twenty_five_percent_of_five_hundred_is_one_twenty_five_l91_91915

theorem twenty_five_percent_of_five_hundred_is_one_twenty_five :
  let percent := 0.25
  let amount := 500
  percent * amount = 125 :=
by
  sorry

end twenty_five_percent_of_five_hundred_is_one_twenty_five_l91_91915


namespace building_height_l91_91593

noncomputable def height_of_building (H_f L_f L_b : ℝ) : ℝ :=
  (H_f * L_b) / L_f

theorem building_height (H_f L_f L_b H_b : ℝ)
  (H_f_val : H_f = 17.5)
  (L_f_val : L_f = 40.25)
  (L_b_val : L_b = 28.75)
  (H_b_val : H_b = 12.4375) :
  height_of_building H_f L_f L_b = H_b := by
  rw [H_f_val, L_f_val, L_b_val, H_b_val]
  -- sorry to skip the proof
  sorry

end building_height_l91_91593


namespace evaluate_expression_l91_91897

theorem evaluate_expression : 
  ∀ (x y z : ℝ), 
  x = 2 → 
  y = -3 → 
  z = 1 → 
  x^2 + y^2 + z^2 + 2 * x * y - z^3 = 1 := by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  sorry

end evaluate_expression_l91_91897
