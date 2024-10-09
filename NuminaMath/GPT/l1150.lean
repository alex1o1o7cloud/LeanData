import Mathlib

namespace layla_goals_l1150_115001

variable (L K : ℕ)
variable (average_score : ℕ := 92)
variable (goals_difference : ℕ := 24)
variable (total_games : ℕ := 4)

theorem layla_goals :
  K = L - goals_difference →
  (L + K) = (average_score * total_games) →
  L = 196 :=
by
  sorry

end layla_goals_l1150_115001


namespace ratio_of_divisor_to_quotient_l1150_115040

noncomputable def r : ℕ := 5
noncomputable def n : ℕ := 113

-- Assuming existence of k and quotient Q
axiom h1 : ∃ (k Q : ℕ), (3 * r + 3 = k * Q) ∧ (n = (3 * r + 3) * Q + r)

theorem ratio_of_divisor_to_quotient : ∃ (D Q : ℕ), (D = 3 * r + 3) ∧ (n = D * Q + r) ∧ (D / Q = 3) :=
  by sorry

end ratio_of_divisor_to_quotient_l1150_115040


namespace weight_of_four_cakes_l1150_115003

variable (C B : ℕ)  -- We declare C and B as natural numbers representing the weights in grams.

def cake_bread_weight_conditions (C B : ℕ) : Prop :=
  (3 * C + 5 * B = 1100) ∧ (C = B + 100)

theorem weight_of_four_cakes (C B : ℕ) 
  (h : cake_bread_weight_conditions C B) : 
  4 * C = 800 := 
by 
  {sorry}

end weight_of_four_cakes_l1150_115003


namespace triangle_angle_A_l1150_115041

theorem triangle_angle_A (A B C : ℝ) (h1 : C = 3 * B) (h2 : B = 30) (h3 : A + B + C = 180) : A = 60 := by
  sorry

end triangle_angle_A_l1150_115041


namespace square_side_increase_l1150_115050

theorem square_side_increase (s : ℝ) :
  let new_side := 1.5 * s
  let new_area := new_side^2
  let original_area := s^2
  let new_perimeter := 4 * new_side
  let original_perimeter := 4 * s
  let new_diagonal := new_side * Real.sqrt 2
  let original_diagonal := s * Real.sqrt 2
  (new_area - original_area) / original_area * 100 = 125 ∧
  (new_perimeter - original_perimeter) / original_perimeter * 100 = 50 ∧
  (new_diagonal - original_diagonal) / original_diagonal * 100 = 50 :=
by
  sorry

end square_side_increase_l1150_115050


namespace complex_seventh_root_of_unity_l1150_115076

theorem complex_seventh_root_of_unity (r : ℂ) (h1 : r^7 = 1) (h2: r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by
  sorry

end complex_seventh_root_of_unity_l1150_115076


namespace proof_equivalence_l1150_115077

variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables {α β γ δ : ℝ} -- angles are real numbers

-- Definition of cyclic quadrilateral
def cyclic_quadrilateral (α β γ δ : ℝ) : Prop :=
α + γ = 180 ∧ β + δ = 180

-- Definition of the problem statements
def statement1 (α γ : ℝ) : Prop :=
α = γ → α = 90

def statement3 (α γ : ℝ) : Prop :=
180 - α + 180 - γ = 180

def statement2 (α β : ℝ) (ψ χ : ℝ) : Prop := 
α = β → cyclic_quadrilateral α β ψ χ → ψ = χ ∨ (α = β ∧ α = ψ ∧ α = χ)

def statement4 (α β γ δ : ℝ) : Prop :=
1*α + 2*β + 3*γ + 4*δ = 360

-- Theorem statement
theorem proof_equivalence (α β γ δ : ℝ) :
  cyclic_quadrilateral α β γ δ →
  (statement1 α γ) ∧ (statement3 α γ) ∧ ¬(statement2 α β γ δ) ∧ ¬(statement4 α β γ δ) :=
by
  sorry

end proof_equivalence_l1150_115077


namespace x_is_perfect_square_l1150_115058

theorem x_is_perfect_square {x y : ℕ} (hx : x > 0) (hy : y > 0) (h : (x^2 + y^2 - x) % (2 * x * y) = 0) : ∃ z : ℕ, x = z^2 :=
by
  -- The proof will proceed here
  sorry

end x_is_perfect_square_l1150_115058


namespace natural_solution_unique_l1150_115024

theorem natural_solution_unique (n : ℕ) (h : (2 * n - 1) / n^5 = 3 - 2 / n) : n = 1 := by
  sorry

end natural_solution_unique_l1150_115024


namespace usual_time_is_180_l1150_115036

variable (D S1 T : ℝ)

-- Conditions
def usual_time : Prop := T = D / S1
def reduced_speed : Prop := ∃ S2 : ℝ, S2 = 5 / 6 * S1
def total_delay : Prop := 6 + 12 + 18 = 36
def total_time_reduced_speed_stops : Prop := ∃ T' : ℝ, T' + 36 = 6 / 5 * T
def time_equation : Prop := T + 36 = 6 / 5 * T

-- Proof problem statement
theorem usual_time_is_180 (h1 : usual_time D S1 T)
                          (h2 : reduced_speed S1)
                          (h3 : total_delay)
                          (h4 : total_time_reduced_speed_stops T)
                          (h5 : time_equation T) :
                          T = 180 := by
  sorry

end usual_time_is_180_l1150_115036


namespace Sam_scored_points_l1150_115091

theorem Sam_scored_points (total_points friend_points S: ℕ) (h1: friend_points = 12) (h2: total_points = 87) (h3: total_points = S + friend_points) : S = 75 :=
by
  sorry

end Sam_scored_points_l1150_115091


namespace points_lie_on_parabola_l1150_115069

noncomputable def lies_on_parabola (t : ℝ) : Prop :=
  let x := Real.cos t ^ 2
  let y := Real.sin t * Real.cos t
  y ^ 2 = x * (1 - x)

-- Statement to prove
theorem points_lie_on_parabola : ∀ t : ℝ, lies_on_parabola t :=
by
  intro t
  sorry

end points_lie_on_parabola_l1150_115069


namespace factorize_difference_of_squares_l1150_115018

theorem factorize_difference_of_squares (y : ℝ) : y^2 - 4 = (y + 2) * (y - 2) := 
by
  sorry

end factorize_difference_of_squares_l1150_115018


namespace Dabbie_spends_99_dollars_l1150_115023

noncomputable def total_cost_turkeys (w1 w2 w3 w4 : ℝ) (cost_per_kg : ℝ) : ℝ :=
  (w1 + w2 + w3 + w4) * cost_per_kg

theorem Dabbie_spends_99_dollars :
  let w1 := 6
  let w2 := 9
  let w3 := 2 * w2
  let w4 := (w1 + w2 + w3) / 2
  let cost_per_kg := 2
  total_cost_turkeys w1 w2 w3 w4 cost_per_kg = 99 := 
by
  sorry

end Dabbie_spends_99_dollars_l1150_115023


namespace find_ab_l1150_115053

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 + b

theorem find_ab (a b : ℝ) (h₀ : a ≠ 0) (h₁ : f a b 2 = 2) (h₂ : f a b 3 = 5) :
    (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 3) :=
by 
  sorry

end find_ab_l1150_115053


namespace find_p_l1150_115084

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def hyperbola_focus : ℝ × ℝ :=
  (2, 0)

theorem find_p (p : ℝ) (h : p > 0) (hp : parabola_focus p = hyperbola_focus) : p = 4 :=
by
  sorry

end find_p_l1150_115084


namespace remainder_when_divided_by_18_l1150_115032

theorem remainder_when_divided_by_18 (n : ℕ) (r3 r6 r9 : ℕ)
  (hr3 : r3 = n % 3)
  (hr6 : r6 = n % 6)
  (hr9 : r9 = n % 9)
  (h_sum : r3 + r6 + r9 = 15) :
  n % 18 = 17 := sorry

end remainder_when_divided_by_18_l1150_115032


namespace find_m_of_transformed_point_eq_l1150_115063

theorem find_m_of_transformed_point_eq (m : ℝ) (h : m + 1 = 5) : m = 4 :=
by
  sorry

end find_m_of_transformed_point_eq_l1150_115063


namespace range_of_x_l1150_115075

theorem range_of_x (x : ℝ) : (x ≠ -3) ∧ (x ≤ 4) ↔ (x ≤ 4) ∧ (x ≠ -3) :=
by { sorry }

end range_of_x_l1150_115075


namespace line_passes_through_2nd_and_4th_quadrants_l1150_115071

theorem line_passes_through_2nd_and_4th_quadrants (b : ℝ) :
  (∀ x : ℝ, x > 0 → -2 * x + b < 0) ∧ (∀ x : ℝ, x < 0 → -2 * x + b > 0) :=
by
  sorry

end line_passes_through_2nd_and_4th_quadrants_l1150_115071


namespace Walter_age_in_2010_l1150_115059

-- Define Walter's age in 2005 as y
def Walter_age_2005 (y : ℕ) : Prop :=
  (2005 - y) + (2005 - 3 * y) = 3858

-- Define Walter's age in 2010
theorem Walter_age_in_2010 (y : ℕ) (hy : Walter_age_2005 y) : y + 5 = 43 :=
by
  sorry

end Walter_age_in_2010_l1150_115059


namespace part1_eq_part2_if_empty_intersection_then_a_geq_3_l1150_115087

open Set

variable {U : Type} {a : ℝ}

def universal_set : Set ℝ := univ
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def B1 (a : ℝ) : Set ℝ := {x : ℝ | x > a}
def complement_B1 (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def intersection_with_complement (a : ℝ) : Set ℝ := A ∩ complement_B1 a

-- Statement for part (1)
theorem part1_eq {a : ℝ} (h : a = 2) : intersection_with_complement a = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by sorry

-- Statement for part (2)
theorem part2_if_empty_intersection_then_a_geq_3 
(h : A ∩ B1 a = ∅) : a ≥ 3 :=
by sorry

end part1_eq_part2_if_empty_intersection_then_a_geq_3_l1150_115087


namespace total_nails_to_cut_l1150_115008

theorem total_nails_to_cut :
  let dogs := 4 
  let legs_per_dog := 4
  let nails_per_dog_leg := 4
  let parrots := 8
  let legs_per_parrot := 2
  let nails_per_parrot_leg := 3
  let extra_nail := 1
  let total_dog_nails := dogs * legs_per_dog * nails_per_dog_leg
  let total_parrot_nails := (parrots * legs_per_parrot * nails_per_parrot_leg) + extra_nail
  total_dog_nails + total_parrot_nails = 113 :=
sorry

end total_nails_to_cut_l1150_115008


namespace smallest_x_solution_l1150_115000

theorem smallest_x_solution :
  (∃ x : ℚ, abs (4 * x + 3) = 30 ∧ ∀ y : ℚ, abs (4 * y + 3) = 30 → x ≤ y) ↔ x = -33 / 4 := by
  sorry

end smallest_x_solution_l1150_115000


namespace value_of_x_squared_plus_9y_squared_l1150_115039

theorem value_of_x_squared_plus_9y_squared (x y : ℝ)
  (h1 : x + 3 * y = 5)
  (h2 : x * y = -8) : x^2 + 9 * y^2 = 73 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l1150_115039


namespace maple_trees_planted_plant_maple_trees_today_l1150_115022

-- Define the initial number of maple trees
def initial_maple_trees : ℕ := 2

-- Define the number of maple trees the park will have after planting
def final_maple_trees : ℕ := 11

-- Define the number of popular trees, though it is irrelevant for the proof
def initial_popular_trees : ℕ := 5

-- The main statement to prove: number of maple trees planted today
theorem maple_trees_planted : ℕ :=
  final_maple_trees - initial_maple_trees

-- Prove that the number of maple trees planted today is 9
theorem plant_maple_trees_today :
  maple_trees_planted = 9 :=
by
  sorry

end maple_trees_planted_plant_maple_trees_today_l1150_115022


namespace extremum_condition_l1150_115037

noncomputable def quadratic_polynomial (a x : ℝ) : ℝ := a * x^2 + x + 1

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, ∃ f' : ℝ → ℝ, 
     (f' = (fun x => 2 * a * x + 1)) ∧ 
     (f' x = 0) ∧ 
     (∃ (f'' : ℝ → ℝ), (f'' = (fun x => 2 * a)) ∧ (f'' x ≠ 0))) ↔ a < 0 := 
sorry

end extremum_condition_l1150_115037


namespace function_machine_output_l1150_115019

-- Define the initial input
def input : ℕ := 12

-- Define the function machine steps
def functionMachine (x : ℕ) : ℕ :=
  if x * 3 <= 20 then (x * 3) / 2
  else (x * 3) - 2

-- State the property we want to prove
theorem function_machine_output : functionMachine 12 = 34 :=
by
  -- Skip the proof
  sorry

end function_machine_output_l1150_115019


namespace common_difference_is_minus_3_l1150_115048

variable (a_n : ℕ → ℤ) (a1 d : ℤ)

-- Definitions expressing the conditions of the problem
def arithmetic_prog : Prop := ∀ (n : ℕ), a_n n = a1 + (n - 1) * d

def condition1 : Prop := a1 + (a1 + 6 * d) = -8

def condition2 : Prop := a1 + d = 2

-- The statement we need to prove
theorem common_difference_is_minus_3 :
  arithmetic_prog a_n a1 d ∧ condition1 a1 d ∧ condition2 a1 d → d = -3 :=
by {
  -- The proof would go here
  sorry
}

end common_difference_is_minus_3_l1150_115048


namespace complement_union_l1150_115080

def R := Set ℝ

def A : Set ℝ := {x | x ≥ 1}

def B : Set ℝ := {y | ∃ x, x ≥ 1 ∧ y = Real.exp x}

theorem complement_union (R : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  (A ∪ B)ᶜ = {x | x < 1} := by
  sorry

end complement_union_l1150_115080


namespace length_less_than_twice_width_l1150_115081

def length : ℝ := 24
def width : ℝ := 13.5

theorem length_less_than_twice_width : 2 * width - length = 3 := by
  sorry

end length_less_than_twice_width_l1150_115081


namespace violet_has_27_nails_l1150_115083

def nails_tickletoe : ℕ := 12  -- T
def nails_violet : ℕ := 2 * nails_tickletoe + 3

theorem violet_has_27_nails (h : nails_tickletoe + nails_violet = 39) : nails_violet = 27 :=
by
  sorry

end violet_has_27_nails_l1150_115083


namespace max_green_socks_l1150_115010

theorem max_green_socks (g y : ℕ) (h1 : g + y ≤ 2025)
  (h2 : (g * (g - 1))/(g + y) * (g + y - 1) = 1/3) : 
  g ≤ 990 := 
sorry

end max_green_socks_l1150_115010


namespace incorrect_inequality_l1150_115095

theorem incorrect_inequality (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) : ¬ (a^2 < a * b) :=
by
  sorry

end incorrect_inequality_l1150_115095


namespace equilateral_is_cute_specific_triangle_is_cute_find_AB_length_l1150_115068

-- Definition of a cute triangle
def is_cute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2 ∨ a^2 + c^2 = 2 * b^2 ∨ b^2 + c^2 = 2 * a^2

-- 1. Prove an equilateral triangle is a cute triangle
theorem equilateral_is_cute (a : ℝ) : is_cute_triangle a a a :=
by
  sorry

-- 2. Prove the triangle with sides 4, 2√6, and 2√5 is a cute triangle
theorem specific_triangle_is_cute : is_cute_triangle 4 (2*Real.sqrt 6) (2*Real.sqrt 5) :=
by
  sorry

-- 3. Prove the length of AB for the given right triangle is 2√6 or 2√3
theorem find_AB_length (AB BC : ℝ) (AC : ℝ := 2*Real.sqrt 2) (h_cute : is_cute_triangle AB BC AC) : AB = 2*Real.sqrt 6 ∨ AB = 2*Real.sqrt 3 :=
by
  sorry

end equilateral_is_cute_specific_triangle_is_cute_find_AB_length_l1150_115068


namespace range_of_m_l1150_115049

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x ^ 2 - 2 * (4 - m) * x + 1
def g (m : ℝ) (x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ (0 < m ∧ m < 8) :=
sorry

end range_of_m_l1150_115049


namespace chad_sandwiches_l1150_115007

-- Definitions representing the conditions
def crackers_per_sleeve : ℕ := 28
def sleeves_per_box : ℕ := 4
def boxes : ℕ := 5
def nights : ℕ := 56
def crackers_per_sandwich : ℕ := 2

-- Definition representing the final question about the number of sandwiches
def sandwiches_per_night (crackers_per_sleeve sleeves_per_box boxes nights crackers_per_sandwich : ℕ) : ℕ :=
  (crackers_per_sleeve * sleeves_per_box * boxes) / nights / crackers_per_sandwich

-- The theorem that states Chad makes 5 sandwiches each night
theorem chad_sandwiches :
  sandwiches_per_night crackers_per_sleeve sleeves_per_box boxes nights crackers_per_sandwich = 5 :=
by
  -- Proof outline:
  -- crackers_per_sleeve * sleeves_per_box * boxes = 28 * 4 * 5 = 560
  -- 560 / nights = 560 / 56 = 10 crackers per night
  -- 10 / crackers_per_sandwich = 10 / 2 = 5 sandwiches per night
  sorry

end chad_sandwiches_l1150_115007


namespace one_fourth_of_six_point_eight_eq_seventeen_tenths_l1150_115046

theorem one_fourth_of_six_point_eight_eq_seventeen_tenths :  (6.8 / 4) = (17 / 10) :=
by
  -- Skipping proof
  sorry

end one_fourth_of_six_point_eight_eq_seventeen_tenths_l1150_115046


namespace sum_first_five_terms_arithmetic_sequence_l1150_115002

theorem sum_first_five_terms_arithmetic_sequence (a d : ℤ)
  (h1 : a + 5 * d = 10)
  (h2 : a + 6 * d = 15)
  (h3 : a + 7 * d = 20) :
  5 * (2 * a + (5 - 1) * d) / 2 = -25 := by
  sorry

end sum_first_five_terms_arithmetic_sequence_l1150_115002


namespace convert_yahs_to_bahs_l1150_115014

theorem convert_yahs_to_bahs :
  (∀ (bahs rahs yahs : ℝ), (10 * bahs = 18 * rahs) 
    ∧ (6 * rahs = 10 * yahs) 
    → (1500 * yahs / (10 / 6) / (18 / 10) = 500 * bahs)) :=
by
  intros bahs rahs yahs h
  sorry

end convert_yahs_to_bahs_l1150_115014


namespace line_parallel_l1150_115089

theorem line_parallel (x y : ℝ) :
  ∃ m b : ℝ, 
    y = m * (x - 2) + (-4) ∧ 
    m = 2 ∧ 
    (∀ (x y : ℝ), y = 2 * x - 8 → 2 * x - y - 8 = 0) :=
sorry

end line_parallel_l1150_115089


namespace largest_among_four_l1150_115054

theorem largest_among_four (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  max (max a (max (a + b) (a - b))) (ab) = a - b :=
by {
  sorry
}

end largest_among_four_l1150_115054


namespace sum_of_bases_l1150_115064

theorem sum_of_bases (F1 F2 : ℚ) (R1 R2 : ℕ) (hF1_R1 : F1 = (3 * R1 + 7) / (R1^2 - 1) ∧ F2 = (7 * R1 + 3) / (R1^2 - 1))
    (hF1_R2 : F1 = (2 * R2 + 5) / (R2^2 - 1) ∧ F2 = (5 * R2 + 2) / (R2^2 - 1)) : 
    R1 + R2 = 19 := 
sorry

end sum_of_bases_l1150_115064


namespace range_of_a_l1150_115061

theorem range_of_a (p q : Set ℝ) (a : ℝ) (h1 : ∀ x, 2 * x^2 - 3 * x + 1 ≤ 0 → x ∈ p) 
                             (h2 : ∀ x, x^2 - (2 * a + 1) * x + a^2 + a ≤ 0 → x ∈ q)
                             (h3 : ∀ x, p x → q x ∧ ∃ x, ¬p x ∧ q x) : 
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end range_of_a_l1150_115061


namespace a_neg_half_not_bounded_a_bounded_range_l1150_115079

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  1 + a * (1/3)^x + (1/9)^x

theorem a_neg_half_not_bounded (a : ℝ) :
  a = -1/2 → ¬(∃ M > 0, ∀ x < 0, |f x a| ≤ M) :=
by
  sorry

theorem a_bounded_range (a : ℝ) : 
  (∀ x ≥ 0, |f x a| ≤ 4) → -6 ≤ a ∧ a ≤ 2 :=
by
  sorry

end a_neg_half_not_bounded_a_bounded_range_l1150_115079


namespace smallest_four_digit_divisible_by_9_l1150_115044

theorem smallest_four_digit_divisible_by_9 
    (n : ℕ) 
    (h1 : 1000 ≤ n ∧ n < 10000) 
    (h2 : n % 9 = 0)
    (h3 : n % 10 % 2 = 1)
    (h4 : (n / 1000) % 2 = 1)
    (h5 : (n / 10) % 10 % 2 = 0)
    (h6 : (n / 100) % 10 % 2 = 0) :
  n = 3609 :=
sorry

end smallest_four_digit_divisible_by_9_l1150_115044


namespace negation_of_forall_inequality_l1150_115093

theorem negation_of_forall_inequality:
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 - x > 0 :=
by
  sorry

end negation_of_forall_inequality_l1150_115093


namespace rotation_locus_l1150_115090

-- Definitions for points and structure of the cube
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
(A : Point3D) (B : Point3D) (C : Point3D) (D : Point3D)
(E : Point3D) (F : Point3D) (G : Point3D) (H : Point3D)

-- Function to perform the required rotations and return the locus geometrical representation
noncomputable def locus_points_on_surface (c : Cube) : Set Point3D :=
sorry

-- Mathematical problem rephrased in Lean 4 statement
theorem rotation_locus (c : Cube) :
  locus_points_on_surface c = {c.D, c.A} ∪ {c.A, c.C} ∪ {c.C, c.D} :=
sorry

end rotation_locus_l1150_115090


namespace arithmetic_sequence_max_sum_proof_l1150_115042

noncomputable def arithmetic_sequence_max_sum (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n * a_1 + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_max_sum_proof (a_1 d : ℝ) 
  (h1 : 3 * a_1 + 6 * d = 9)
  (h2 : a_1 + 5 * d = -9) :
  ∃ n : ℕ, n = 3 ∧ arithmetic_sequence_max_sum a_1 d n = 21 :=
by
  sorry

end arithmetic_sequence_max_sum_proof_l1150_115042


namespace total_volume_is_10_l1150_115020

noncomputable def total_volume_of_final_mixture (V : ℝ) : ℝ :=
  2.5 + V

theorem total_volume_is_10 :
  ∃ (V : ℝ), 
  (0.30 * 2.5 + 0.50 * V = 0.45 * (2.5 + V)) ∧ 
  total_volume_of_final_mixture V = 10 :=
by
  sorry

end total_volume_is_10_l1150_115020


namespace distance_to_line_l1150_115011

theorem distance_to_line (a : ℝ) (d : ℝ)
  (h1 : d = 6)
  (h2 : |3 * a + 6| / 5 = d) :
  a = 8 ∨ a = -12 :=
by
  sorry

end distance_to_line_l1150_115011


namespace baskets_picked_l1150_115021

theorem baskets_picked
  (B : ℕ) -- How many baskets did her brother pick?
  (S : ℕ := 15) -- Each basket contains 15 strawberries
  (H1 : (8 * B * S) + (B * S) + ((8 * B * S) - 93) = 4 * 168) -- Total number of strawberries when divided equally
  (H2 : S = 15) -- Number of strawberries in each basket
: B = 3 :=
sorry

end baskets_picked_l1150_115021


namespace f_10_l1150_115051

namespace MathProof

variable (f : ℤ → ℤ)

-- Condition 1: f(1) + 1 > 0
axiom cond1 : f 1 + 1 > 0

-- Condition 2: f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y for any x, y ∈ ℤ
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y

-- Condition 3: 2 * f(x) = f(x + 1) - x + 1 for any x ∈ ℤ
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

-- We need to prove f(10) = 1014
theorem f_10 : f 10 = 1014 :=
by
  sorry

end MathProof

end f_10_l1150_115051


namespace laura_park_time_percentage_l1150_115098

theorem laura_park_time_percentage (num_trips: ℕ) (time_in_park: ℝ) (walking_time: ℝ) 
    (total_percentage_in_park: ℝ) 
    (h1: num_trips = 6) 
    (h2: time_in_park = 2) 
    (h3: walking_time = 0.5) 
    (h4: total_percentage_in_park = 80) : 
    (time_in_park * num_trips) / ((time_in_park + walking_time) * num_trips) * 100 = total_percentage_in_park :=
by
  sorry

end laura_park_time_percentage_l1150_115098


namespace initial_money_l1150_115099

theorem initial_money (B S G M : ℕ) 
  (hB : B = 8) 
  (hS : S = 2 * B) 
  (hG : G = 3 * S) 
  (change : ℕ) 
  (h_change : change = 28)
  (h_total : B + S + G + change = M) : 
  M = 100 := 
by 
  sorry

end initial_money_l1150_115099


namespace total_flour_amount_l1150_115013

-- Define the initial amount of flour in the bowl
def initial_flour : ℝ := 2.75

-- Define the amount of flour added by the baker
def added_flour : ℝ := 0.45

-- Prove that the total amount of flour is 3.20 kilograms
theorem total_flour_amount : initial_flour + added_flour = 3.20 :=
by
  sorry

end total_flour_amount_l1150_115013


namespace no_rational_roots_of_polynomial_l1150_115092

theorem no_rational_roots_of_polynomial :
  ¬ ∃ (x : ℚ), (3 * x^4 - 7 * x^3 - 4 * x^2 + 8 * x + 3 = 0) :=
by
  sorry

end no_rational_roots_of_polynomial_l1150_115092


namespace john_thrice_tom_years_ago_l1150_115096

-- Define the ages of Tom and John
def T : ℕ := 16
def J : ℕ := 36

-- Condition that John will be 2 times Tom's age in 4 years
def john_twice_tom_in_4_years (J T : ℕ) : Prop := J + 4 = 2 * (T + 4)

-- The number of years ago John was thrice as old as Tom
def years_ago (J T x : ℕ) : Prop := J - x = 3 * (T - x)

-- Prove that the number of years ago John was thrice as old as Tom is 6
theorem john_thrice_tom_years_ago (h1 : john_twice_tom_in_4_years 36 16) : years_ago 36 16 6 :=
by
  -- Import initial values into the context
  unfold john_twice_tom_in_4_years at h1
  unfold years_ago
  -- Solve the steps, more details in the actual solution
  sorry

end john_thrice_tom_years_ago_l1150_115096


namespace melanie_plums_l1150_115073

variable (initialPlums : ℕ) (givenPlums : ℕ)

theorem melanie_plums :
  initialPlums = 7 → givenPlums = 3 → initialPlums - givenPlums = 4 :=
by
  intro h1 h2
  -- proof omitted
  exact sorry

end melanie_plums_l1150_115073


namespace value_of_phi_l1150_115028

theorem value_of_phi { φ : ℝ } (hφ1 : 0 < φ) (hφ2 : φ < π)
  (symm_condition : ∃ k : ℤ, -π / 8 + φ = k * π + π / 2) : φ = 3 * π / 4 := 
by 
  sorry

end value_of_phi_l1150_115028


namespace transformed_curve_l1150_115094

def curve_C (x y : ℝ) := (x - y)^2 + y^2 = 1

theorem transformed_curve (x y : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
    A = ![![2, -2], ![0, 1]] →
    (∃ (x0 y0 : ℝ), curve_C x0 y0 ∧ x = 2 * x0 - 2 * y0 ∧ y = y0) →
    (∃ (x y : ℝ), (x^2 / 4 + y^2 = 1)) :=
by
  -- Proof to be completed
  sorry

end transformed_curve_l1150_115094


namespace range_of_a_l1150_115070

variable (a : ℝ)

def p (a : ℝ) : Prop := 3/2 < a ∧ a < 5/2
def q (a : ℝ) : Prop := 2 ≤ a ∧ a ≤ 4

theorem range_of_a (h₁ : ¬(p a ∧ q a)) (h₂ : p a ∨ q a) : (3/2 < a ∧ a < 2) ∨ (5/2 ≤ a ∧ a ≤ 4) :=
sorry

end range_of_a_l1150_115070


namespace original_wage_l1150_115027

theorem original_wage (W : ℝ) 
  (h1: 1.40 * W = 28) : 
  W = 20 :=
sorry

end original_wage_l1150_115027


namespace fixed_point_of_inverse_l1150_115057

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1) + 4

theorem fixed_point_of_inverse (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  f a (5) = 1 :=
by
  unfold f
  sorry

end fixed_point_of_inverse_l1150_115057


namespace shortest_tree_height_proof_l1150_115029

def tallest_tree_height : ℕ := 150
def middle_tree_height : ℕ := (2 * tallest_tree_height) / 3
def shortest_tree_height : ℕ := middle_tree_height / 2

theorem shortest_tree_height_proof : shortest_tree_height = 50 := by
  sorry

end shortest_tree_height_proof_l1150_115029


namespace luxury_class_adults_l1150_115006

def total_passengers : ℕ := 300
def adult_percentage : ℝ := 0.70
def luxury_percentage : ℝ := 0.15

def total_adults (p : ℕ) : ℕ := (p * 70) / 100
def adults_in_luxury (a : ℕ) : ℕ := (a * 15) / 100

theorem luxury_class_adults :
  adults_in_luxury (total_adults total_passengers) = 31 :=
by
  sorry

end luxury_class_adults_l1150_115006


namespace quadratic_real_roots_k_eq_one_l1150_115072

theorem quadratic_real_roots_k_eq_one 
  (k : ℕ) 
  (h_nonneg : k ≥ 0) 
  (h_real_roots : ∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) : 
  k = 1 := 
sorry

end quadratic_real_roots_k_eq_one_l1150_115072


namespace min_z_value_l1150_115038

theorem min_z_value (x y z : ℝ) (h1 : 2 * x + y = 1) (h2 : z = 4 ^ x + 2 ^ y) : z ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_z_value_l1150_115038


namespace range_of_a_l1150_115060

variable (f : ℝ → ℝ)
variable (a : ℝ)

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def holds_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, (1/2) ≤ x ∧ x ≤ 1 → f (a*x + 1) ≤ f (x - 2)

theorem range_of_a (h1 : is_even f)
                   (h2 : is_increasing_on_nonneg f)
                   (h3 : holds_on_interval f a) :
  -2 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l1150_115060


namespace tangent_line_circle_l1150_115043

theorem tangent_line_circle (r : ℝ) (h : 0 < r) :
  (∀ x y : ℝ, x + y = r → x * x + y * y ≠ 4 * r) →
  r = 8 :=
by
  sorry

end tangent_line_circle_l1150_115043


namespace maximize_probability_l1150_115030

def numbers_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def pairs_summing_to_12 (l : List Int) : List (Int × Int) :=
  List.filter (fun (p : Int × Int) => p.1 + p.2 = 12) (List.product l l)

def distinct_pairs (pairs : List (Int × Int)) : List (Int × Int) :=
  List.filter (fun (p : Int × Int) => p.1 ≠ p.2) pairs

def valid_pairs (l : List Int) : List (Int × Int) :=
  distinct_pairs (pairs_summing_to_12 l)

def count_valid_pairs (l : List Int) : Nat :=
  List.length (valid_pairs l)

def remove_and_check (x : Int) : List Int :=
  List.erase numbers_list x

theorem maximize_probability :
  ∀ x : Int, count_valid_pairs (remove_and_check 6) ≥ count_valid_pairs (remove_and_check x) :=
sorry

end maximize_probability_l1150_115030


namespace find_number_of_folders_l1150_115055

theorem find_number_of_folders :
  let price_pen := 1
  let price_notebook := 3
  let price_folder := 5
  let pens_bought := 3
  let notebooks_bought := 4
  let bill := 50
  let change := 25
  let total_cost_pens_notebooks := pens_bought * price_pen + notebooks_bought * price_notebook
  let amount_spent := bill - change
  let amount_spent_on_folders := amount_spent - total_cost_pens_notebooks
  let number_of_folders := amount_spent_on_folders / price_folder
  number_of_folders = 2 :=
by
  sorry

end find_number_of_folders_l1150_115055


namespace same_color_probability_l1150_115086

/-- There are 7 red plates and 5 blue plates. We want to prove that the probability of
    selecting 3 plates, where all are of the same color, is 9/44. -/
theorem same_color_probability :
  let total_plates := 12
  let total_ways_to_choose := Nat.choose total_plates 3
  let red_plates := 7
  let blue_plates := 5
  let ways_to_choose_red := Nat.choose red_plates 3
  let ways_to_choose_blue := Nat.choose blue_plates 3
  let favorable_ways_to_choose := ways_to_choose_red + ways_to_choose_blue
  ∃ (prob : ℚ), prob = (favorable_ways_to_choose : ℚ) / (total_ways_to_choose : ℚ) ∧
                 prob = 9 / 44 :=
by
  sorry

end same_color_probability_l1150_115086


namespace proof_f_derivative_neg1_l1150_115078

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * x ^ 4 + b * x ^ 2 + c

noncomputable def f_derivative (x : ℝ) (a b : ℝ) : ℝ :=
  4 * a * x ^ 3 + 2 * b * x

theorem proof_f_derivative_neg1
  (a b c : ℝ) (h : f_derivative 1 a b = 2) :
  f_derivative (-1) a b = -2 :=
by
  sorry

end proof_f_derivative_neg1_l1150_115078


namespace find_f2_l1150_115097

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem find_f2 (a b : ℝ) (h : (∃ x : ℝ, f x a b = 10 ∧ x = 1)):
  f 2 a b = 18 ∨ f 2 a b = 11 :=
sorry

end find_f2_l1150_115097


namespace quadratic_has_distinct_real_roots_l1150_115004

theorem quadratic_has_distinct_real_roots :
  let a := 2
  let b := 3
  let c := -4
  (b^2 - 4 * a * c) > 0 := by
  sorry

end quadratic_has_distinct_real_roots_l1150_115004


namespace t_shirt_cost_calculation_l1150_115025

variables (initial_amount ticket_cost food_cost money_left t_shirt_cost : ℕ)

axiom h1 : initial_amount = 75
axiom h2 : ticket_cost = 30
axiom h3 : food_cost = 13
axiom h4 : money_left = 9

theorem t_shirt_cost_calculation : 
  t_shirt_cost = initial_amount - (ticket_cost + food_cost) - money_left :=
sorry

end t_shirt_cost_calculation_l1150_115025


namespace asymptotes_of_hyperbola_l1150_115033

theorem asymptotes_of_hyperbola (b : ℝ) (h_focus : 2 * Real.sqrt 2 ≠ 0) :
  2 * Real.sqrt 2 = Real.sqrt ((2 * 2) + b^2) → 
  (∀ (x y : ℝ), ((x^2 / 4) - (y^2 / b^2) = 1 → x^2 - y^2 = 4)) → 
  (∀ (x y : ℝ), ((x^2 - y^2 = 4) → y = x ∨ y = -x)) := 
  sorry

end asymptotes_of_hyperbola_l1150_115033


namespace arctan_sum_pi_div_two_l1150_115045

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l1150_115045


namespace at_least_one_has_two_distinct_roots_l1150_115031

theorem at_least_one_has_two_distinct_roots
  (p q1 q2 : ℝ)
  (h : p = q1 + q2 + 1) :
  (1 - 4 * q1 > 0) ∨ ((q1 + q2 + 1) ^ 2 - 4 * q2 > 0) :=
by sorry

end at_least_one_has_two_distinct_roots_l1150_115031


namespace arithmetic_sequence_inequality_l1150_115082

variable {α : Type*} [OrderedRing α]

theorem arithmetic_sequence_inequality
  (a : ℕ → α) (d : α)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_d_pos : d > 0)
  (n : ℕ)
  (h_n_gt_1 : n > 1) :
  a 1 * a (n + 1) < a 2 * a n := 
sorry

end arithmetic_sequence_inequality_l1150_115082


namespace dragons_total_games_l1150_115074

theorem dragons_total_games (y x : ℕ) (h1 : x = 60 * y / 100) (h2 : (x + 8) = 55 * (y + 11) / 100) : y + 11 = 50 :=
by
  sorry

end dragons_total_games_l1150_115074


namespace response_rate_is_60_percent_l1150_115015

-- Definitions based on conditions
def responses_needed : ℕ := 900
def questionnaires_mailed : ℕ := 1500

-- Derived definition
def response_rate_percentage : ℚ := (responses_needed : ℚ) / (questionnaires_mailed : ℚ) * 100

-- The theorem stating the problem
theorem response_rate_is_60_percent :
  response_rate_percentage = 60 := 
sorry

end response_rate_is_60_percent_l1150_115015


namespace interest_groups_ranges_l1150_115067

variable (A B C : Finset ℕ)

-- Given conditions
axiom card_A : A.card = 5
axiom card_B : B.card = 4
axiom card_C : C.card = 7
axiom card_A_inter_B : (A ∩ B).card = 3
axiom card_A_inter_B_inter_C : (A ∩ B ∩ C).card = 2

-- Mathematical statement to be proved
theorem interest_groups_ranges :
  2 ≤ ((A ∪ B) ∩ C).card ∧ ((A ∪ B) ∩ C).card ≤ 5 ∧
  8 ≤ (A ∪ B ∪ C).card ∧ (A ∪ B ∪ C).card ≤ 11 := by
  sorry

end interest_groups_ranges_l1150_115067


namespace john_bought_more_than_ray_l1150_115066

variable (R_c R_d M_c M_d J_c J_d : ℕ)

-- Define the conditions
def conditions : Prop :=
  (R_c = 10) ∧
  (R_d = 3) ∧
  (M_c = R_c + 6) ∧
  (M_d = R_d + 1) ∧
  (J_c = M_c + 5) ∧
  (J_d = M_d + 2)

-- Define the question
def john_more_chickens_and_ducks (J_c R_c J_d R_d : ℕ) : ℕ :=
  (J_c - R_c) + (J_d - R_d)

-- The proof problem statement
theorem john_bought_more_than_ray :
  conditions R_c R_d M_c M_d J_c J_d → john_more_chickens_and_ducks J_c R_c J_d R_d = 14 :=
by
  intro h
  sorry

end john_bought_more_than_ray_l1150_115066


namespace problem_part1_problem_part2_l1150_115088

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 2 * Real.sqrt 3 * (Real.sin x) * (Real.cos x) + a

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 6) + 3

theorem problem_part1 (h : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x 2 ≥ 2) :
    ∃ a : ℝ, a = 2 ∧ 
    ∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6), 
    ∃ m : ℤ, x = (m * Real.pi / 2 + Real.pi / 12) ∨ x = (m * Real.pi / 2 + Real.pi / 4) := sorry

theorem problem_part2 :
    ∀ x ∈ Set.Icc 0 (Real.pi / 2), g x = 4 → 
    ∃ s : ℝ, s = Real.pi / 3 := sorry

end problem_part1_problem_part2_l1150_115088


namespace interest_rate_calculation_l1150_115016

theorem interest_rate_calculation (P : ℝ) (r : ℝ) (h1 : P * (1 + r / 100)^3 = 800) (h2 : P * (1 + r / 100)^4 = 820) :
  r = 2.5 := 
  sorry

end interest_rate_calculation_l1150_115016


namespace acute_angle_sine_l1150_115005
--import Lean library

-- Define the problem conditions and statement
theorem acute_angle_sine (a : ℝ) (h1 : 0 < a) (h2 : a < π / 2) (h3 : Real.sin a = 0.6) :
  π / 6 < a ∧ a < π / 4 :=
by 
  sorry

end acute_angle_sine_l1150_115005


namespace find_number_l1150_115012

theorem find_number 
  (x : ℚ) 
  (h : (3 / 4) * x - (8 / 5) * x + 63 = 12) : 
  x = 60 := 
by
  sorry

end find_number_l1150_115012


namespace Kamal_biology_marks_l1150_115056

theorem Kamal_biology_marks 
  (E : ℕ) (M : ℕ) (P : ℕ) (C : ℕ) (A : ℕ) (N : ℕ) (B : ℕ) 
  (hE : E = 66)
  (hM : M = 65)
  (hP : P = 77)
  (hC : C = 62)
  (hA : A = 69)
  (hN : N = 5)
  (h_total : N * A = E + M + P + C + B) 
  : B = 75 :=
by
  sorry

end Kamal_biology_marks_l1150_115056


namespace compare_polynomials_l1150_115085

theorem compare_polynomials (x : ℝ) (h : x ≥ 0) : 
  (x > 2 → 5*x^2 - 1 > 3*x^2 + 3*x + 1) ∧ 
  (x = 2 → 5*x^2 - 1 = 3*x^2 + 3*x + 1) ∧ 
  (0 ≤ x → x < 2 → 5*x^2 - 1 < 3*x^2 + 3*x + 1) :=
sorry

end compare_polynomials_l1150_115085


namespace cube_square_third_smallest_prime_l1150_115009

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end cube_square_third_smallest_prime_l1150_115009


namespace fraction_of_students_with_mentor_l1150_115052

theorem fraction_of_students_with_mentor (s n : ℕ) (h : n / 2 = s / 3) :
  (n / 2 + s / 3 : ℚ) / (n + s : ℚ) = 2 / 5 := by
  sorry

end fraction_of_students_with_mentor_l1150_115052


namespace number_of_draw_matches_eq_points_difference_l1150_115034

-- Definitions based on the conditions provided
def teams : ℕ := 16
def matches_per_round : ℕ := 8
def rounds : ℕ := 16
def total_points : ℕ := 222
def total_matches : ℕ := matches_per_round * rounds
def hypothetical_points : ℕ := total_matches * 2
def points_difference : ℕ := hypothetical_points - total_points

-- Theorem stating the equivalence to be proved
theorem number_of_draw_matches_eq_points_difference : 
  points_difference = 34 := 
by
  sorry

end number_of_draw_matches_eq_points_difference_l1150_115034


namespace symmetric_point_with_respect_to_y_eq_x_l1150_115065

theorem symmetric_point_with_respect_to_y_eq_x :
  ∃ x₀ y₀ : ℝ, (∃ (M : ℝ × ℝ), M = (3, 1) ∧
  ((x₀ + 3) / 2 = (y₀ + 1) / 2) ∧
  ((y₀ - 1) / (x₀ - 3) = -1)) ∧
  (x₀ = 1 ∧ y₀ = 3) :=
by
  sorry

end symmetric_point_with_respect_to_y_eq_x_l1150_115065


namespace students_with_dogs_l1150_115062

theorem students_with_dogs (total_students : ℕ) (half_students : total_students / 2 = 50)
  (percent_girls_with_dogs : ℕ → ℚ) (percent_boys_with_dogs : ℕ → ℚ)
  (girls_with_dogs : ∀ (total_girls: ℕ), percent_girls_with_dogs total_girls = 0.2)
  (boys_with_dogs : ∀ (total_boys: ℕ), percent_boys_with_dogs total_boys = 0.1) :
  ∀ (total_girls total_boys students_with_dogs: ℕ),
  total_students = 100 →
  total_girls = total_students / 2 →
  total_boys = total_students / 2 →
  total_girls = 50 →
  total_boys = 50 →
  students_with_dogs = (percent_girls_with_dogs (total_students / 2) * (total_students / 2) + 
                        percent_boys_with_dogs (total_students / 2) * (total_students / 2)) →
  students_with_dogs = 15 :=
by
  intros total_girls total_boys students_with_dogs h1 h2 h3 h4 h5 h6
  sorry

end students_with_dogs_l1150_115062


namespace value_of_f_neg2_l1150_115035

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_f_neg2 : f (-2) = 11 := by
  sorry

end value_of_f_neg2_l1150_115035


namespace volume_of_A_is_2800_l1150_115017

-- Define the dimensions of the fishbowl and water heights
def fishbowl_side_length : ℝ := 20
def height_with_A : ℝ := 16
def height_without_A : ℝ := 9

-- Compute the volume of water with and without object (A)
def volume_with_A : ℝ := fishbowl_side_length ^ 2 * height_with_A
def volume_without_A : ℝ := fishbowl_side_length ^ 2 * height_without_A

-- The volume of object (A)
def volume_A : ℝ := volume_with_A - volume_without_A

-- Prove that this volume is 2800 cubic centimeters
theorem volume_of_A_is_2800 :
  volume_A = 2800 := by
  sorry

end volume_of_A_is_2800_l1150_115017


namespace sandwich_price_l1150_115047

-- Definitions based on conditions
def price_of_soda : ℝ := 0.87
def total_cost : ℝ := 6.46
def num_soda : ℝ := 4
def num_sandwich : ℝ := 2

-- The key equation based on conditions
def total_cost_equation (S : ℝ) : Prop := 
  num_sandwich * S + num_soda * price_of_soda = total_cost

theorem sandwich_price :
  ∃ S : ℝ, total_cost_equation S ∧ S = 1.49 :=
by
  sorry

end sandwich_price_l1150_115047


namespace opposite_of_neg2016_l1150_115026

theorem opposite_of_neg2016 : -(-2016) = 2016 := 
by 
  sorry

end opposite_of_neg2016_l1150_115026
