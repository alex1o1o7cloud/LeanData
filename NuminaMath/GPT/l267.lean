import Mathlib

namespace greatest_integer_b_l267_26761

theorem greatest_integer_b (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ 0) ↔ b ≤ 6 := 
by
  sorry

end greatest_integer_b_l267_26761


namespace fencing_required_l267_26750

theorem fencing_required
  (L : ℝ) (A : ℝ) (h_L : L = 20) (h_A : A = 400) : 
  (2 * (A / L) + L) = 60 :=
by
  sorry

end fencing_required_l267_26750


namespace problem_l267_26769

variable (a : ℕ → ℝ) -- {a_n} is a sequence
variable (S : ℕ → ℝ) -- S_n represents the sum of the first n terms
variable (d : ℝ) -- non-zero common difference
variable (a1 : ℝ) -- first term of the sequence

-- Define an arithmetic sequence with common difference d and first term a1
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (a1 : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- S_n is the sum of the first n terms of an arithmetic sequence
def sum_of_arithmetic_sequence (S a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem problem 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (d : ℝ) 
  (a1 : ℝ) 
  (h_non_zero : d ≠ 0)
  (h_sequence : is_arithmetic_sequence a d a1)
  (h_sum : sum_of_arithmetic_sequence S a)
  (h_S5_eq_S6 : S 5 = S 6) :
  S 11 = 0 := 
sorry

end problem_l267_26769


namespace usual_time_proof_l267_26782

noncomputable 
def usual_time (P T : ℝ) := (P * T) / (100 - P)

theorem usual_time_proof (P T U : ℝ) (h1 : P > 0) (h2 : P < 100) (h3 : T > 0) (h4 : U = usual_time P T) : U = (P * T) / (100 - P) :=
by
    sorry

end usual_time_proof_l267_26782


namespace unique_integer_solution_l267_26710

theorem unique_integer_solution (n : ℤ) :
  (⌊n^2 / 4 + n⌋ - ⌊n / 2⌋^2 = 5) ↔ (n = 10) :=
by sorry

end unique_integer_solution_l267_26710


namespace julia_average_speed_l267_26765

theorem julia_average_speed :
  let distance1 := 45
  let speed1 := 15
  let distance2 := 15
  let speed2 := 45
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 18 := by
sorry

end julia_average_speed_l267_26765


namespace exponent_property_l267_26796

theorem exponent_property (a x y : ℝ) (hx : a ^ x = 2) (hy : a ^ y = 3) : a ^ (x + y) = 6 := by
  sorry

end exponent_property_l267_26796


namespace kevin_expected_away_time_l267_26781

theorem kevin_expected_away_time
  (leak_rate : ℝ)
  (bucket_capacity : ℝ)
  (bucket_factor : ℝ)
  (leak_rate_eq : leak_rate = 1.5)
  (bucket_capacity_eq : bucket_capacity = 36)
  (bucket_factor_eq : bucket_factor = 2)
  : ((bucket_capacity / bucket_factor) / leak_rate) = 12 :=
by
  rw [bucket_capacity_eq, leak_rate_eq, bucket_factor_eq]
  sorry

end kevin_expected_away_time_l267_26781


namespace find_a_l267_26700

theorem find_a 
  (a : ℝ) 
  (h : 1 - 2 * a = a - 2) 
  (h1 : 1 - 2 * a = a - 2) 
  : a = 1 := 
by 
  -- proof goes here
  sorry

end find_a_l267_26700


namespace probability_of_drawing_letter_in_name_l267_26758

theorem probability_of_drawing_letter_in_name :
  let total_letters := 26
  let alonso_letters := ['a', 'l', 'o', 'n', 's']
  let number_of_alonso_letters := alonso_letters.length
  number_of_alonso_letters / total_letters = 5 / 26 :=
by
  sorry

end probability_of_drawing_letter_in_name_l267_26758


namespace tshirt_costs_more_than_jersey_l267_26701

open Nat

def cost_tshirt : ℕ := 192
def cost_jersey : ℕ := 34

theorem tshirt_costs_more_than_jersey :
  cost_tshirt - cost_jersey = 158 :=
by sorry

end tshirt_costs_more_than_jersey_l267_26701


namespace money_received_from_mom_l267_26753

-- Define the given conditions
def initial_amount : ℕ := 48
def amount_spent : ℕ := 11
def amount_after_getting_money : ℕ := 58
def amount_left_after_spending : ℕ := initial_amount - amount_spent

-- Define the proof statement
theorem money_received_from_mom : (amount_after_getting_money - amount_left_after_spending) = 21 :=
by
  -- placeholder for the proof
  sorry

end money_received_from_mom_l267_26753


namespace derivative_of_f_l267_26706

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos x

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = Real.exp (2 * x) * (2 * Real.cos x - Real.sin x) :=
by
  intro x
  -- We skip the proof here
  sorry

end derivative_of_f_l267_26706


namespace cos_double_angle_l267_26756

theorem cos_double_angle (α : ℝ) (h : Real.sin (α / 2) = Real.sqrt 3 / 3) : Real.cos α = 1 / 3 :=
sorry

end cos_double_angle_l267_26756


namespace smallest_pythagorean_sum_square_l267_26730

theorem smallest_pythagorean_sum_square (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p^2 + q^2 = r^2) :
  ∃ (k : ℤ), k = 4 ∧ (p + q + r)^2 ≥ k :=
by
  sorry

end smallest_pythagorean_sum_square_l267_26730


namespace factorize_ab_factorize_x_l267_26740

-- Problem 1: Factorization of a^3 b - 2 a^2 b^2 + a b^3
theorem factorize_ab (a b : ℤ) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = a * b * (a - b)^2 := 
by sorry

-- Problem 2: Factorization of (x^2 + 4)^2 - 16 x^2
theorem factorize_x (x : ℤ) : (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 :=
by sorry

end factorize_ab_factorize_x_l267_26740


namespace simplify_expression_l267_26776

theorem simplify_expression (a b : ℤ) : 
  (18 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 40 * b) = 21 * a + 41 * b := 
by
  sorry

end simplify_expression_l267_26776


namespace probability_route_X_is_8_over_11_l267_26791

-- Definitions for the graph paths and probabilities
def routes_from_A_to_B (X Y : Nat) : Nat := 2 + 6 + 3

def routes_passing_through_X (X Y : Nat) : Nat := 2 + 6

def probability_passing_through_X (total_routes passing_routes : Nat) : Rat :=
  (passing_routes : Rat) / (total_routes : Rat)

theorem probability_route_X_is_8_over_11 :
  let total_routes := routes_from_A_to_B 2 3
  let passing_routes := routes_passing_through_X 2 3
  probability_passing_through_X total_routes passing_routes = 8 / 11 :=
by
  -- Assumes correct route calculations from the conditions and aims to prove the probability value
  sorry

end probability_route_X_is_8_over_11_l267_26791


namespace inequality_proof_l267_26774

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + (3/(a * b + b * c + c * a)) ≥ 6/(a + b + c) := 
sorry

end inequality_proof_l267_26774


namespace total_amount_shared_l267_26734

theorem total_amount_shared (J Jo B : ℝ) (r1 r2 r3 : ℝ)
  (H1 : r1 = 2) (H2 : r2 = 4) (H3 : r3 = 6) (H4 : J = 1600) (part_value : ℝ)
  (H5 : part_value = J / r1) (H6 : Jo = r2 * part_value) (H7 : B = r3 * part_value) :
  J + Jo + B = 9600 :=
sorry

end total_amount_shared_l267_26734


namespace proof_problem_l267_26767

-- Defining a right triangle ΔABC with ∠BCA=90°
structure RightTriangle :=
(a b c : ℝ)  -- sides a, b, c with c as the hypotenuse
(hypotenuse_eq : c^2 = a^2 + b^2)  -- Pythagorean relation

-- Define the circles K1 and K2 with radii r1 and r2 respectively
structure CirclesOnTriangle (Δ : RightTriangle) :=
(r1 r2 : ℝ)  -- radii of the circles K1 and K2

-- Prove the relationship r1 + r2 = a + b - c
theorem proof_problem (Δ : RightTriangle) (C : CirclesOnTriangle Δ) :
  C.r1 + C.r2 = Δ.a + Δ.b - Δ.c := by
  sorry

end proof_problem_l267_26767


namespace bed_length_l267_26748

noncomputable def volume (length width height : ℝ) : ℝ :=
  length * width * height

theorem bed_length
  (width height : ℝ)
  (bags_of_soil soil_volume_per_bag total_volume : ℝ)
  (needed_bags : ℝ)
  (L : ℝ) :
  width = 4 →
  height = 1 →
  needed_bags = 16 →
  soil_volume_per_bag = 4 →
  total_volume = needed_bags * soil_volume_per_bag →
  total_volume = 2 * volume L width height →
  L = 8 :=
by
  intros
  sorry

end bed_length_l267_26748


namespace rate_per_sq_meter_l267_26721

theorem rate_per_sq_meter
  (L : ℝ) (W : ℝ) (total_cost : ℝ)
  (hL : L = 6) (hW : W = 4.75) (h_total_cost : total_cost = 25650) :
  total_cost / (L * W) = 900 :=
by
  sorry

end rate_per_sq_meter_l267_26721


namespace danny_bottle_caps_l267_26718

variable (caps_found : Nat) (caps_existing : Nat)
variable (wrappers_found : Nat) (wrappers_existing : Nat)

theorem danny_bottle_caps:
  caps_found = 58 → caps_existing = 12 →
  wrappers_found = 25 → wrappers_existing = 11 →
  (caps_found + caps_existing) - (wrappers_found + wrappers_existing) = 34 := 
by
  intros h1 h2 h3 h4
  sorry

end danny_bottle_caps_l267_26718


namespace set_listing_l267_26799

open Set

def A : Set (ℤ × ℤ) := {p | ∃ x y : ℤ, p = (x, y) ∧ x^2 = y + 1 ∧ |x| < 2}

theorem set_listing :
  A = {(-1, 0), (0, -1), (1, 0)} :=
by {
  sorry
}

end set_listing_l267_26799


namespace p_and_q_work_together_l267_26777

-- Given conditions
variable (Wp Wq : ℝ)

-- Condition that p is 50% more efficient than q
def efficiency_relation : Prop := Wp = 1.5 * Wq

-- Condition that p can complete the work in 25 days
def work_completion_by_p : Prop := Wp = 1 / 25

-- To be proved that p and q working together can complete the work in 15 days
theorem p_and_q_work_together (h1 : efficiency_relation Wp Wq)
                              (h2 : work_completion_by_p Wp) :
                              1 / (Wp + (Wp / 1.5)) = 15 :=
by
  sorry

end p_and_q_work_together_l267_26777


namespace operation_example_result_l267_26712

def myOperation (A B : ℕ) : ℕ := (A^2 + B^2) / 3

theorem operation_example_result : myOperation (myOperation 6 3) 9 = 102 := by
  sorry

end operation_example_result_l267_26712


namespace a_cubed_value_l267_26792

theorem a_cubed_value (a b : ℝ) (k : ℝ) (h1 : a^3 * b^2 = k) (h2 : a = 5) (h3 : b = 2) : 
  ∃ (a : ℝ), (64 * a^3 = 500) → (a^3 = 125 / 16) :=
by
  sorry

end a_cubed_value_l267_26792


namespace percentage_of_oysters_with_pearls_l267_26704

def jamie_collects_oysters (oysters_per_dive dives total_pearls : ℕ) : ℕ :=
  oysters_per_dive * dives

def percentage_with_pearls (total_pearls total_oysters : ℕ) : ℕ :=
  (total_pearls * 100) / total_oysters

theorem percentage_of_oysters_with_pearls :
  ∀ (oysters_per_dive dives total_pearls : ℕ),
  oysters_per_dive = 16 →
  dives = 14 →
  total_pearls = 56 →
  percentage_with_pearls total_pearls (jamie_collects_oysters oysters_per_dive dives total_pearls) = 25 :=
by
  intros
  sorry

end percentage_of_oysters_with_pearls_l267_26704


namespace purple_marble_probability_l267_26741

theorem purple_marble_probability (P_blue P_green P_purple : ℝ) (h1 : P_blue = 0.35) (h2 : P_green = 0.45) (h3 : P_blue + P_green + P_purple = 1) :
  P_purple = 0.2 := 
by sorry

end purple_marble_probability_l267_26741


namespace proof_of_p_and_not_q_l267_26789

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x
def q : Prop := ∀ x : ℝ, exp x > 1

theorem proof_of_p_and_not_q : p ∧ ¬q :=
by {
  sorry
}

end proof_of_p_and_not_q_l267_26789


namespace complement_setP_in_U_l267_26716

def setU : Set ℝ := {x | -1 < x ∧ x < 3}
def setP : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem complement_setP_in_U : (setU \ setP) = {x | 2 < x ∧ x < 3} :=
by
  sorry

end complement_setP_in_U_l267_26716


namespace rulers_in_drawer_l267_26742

-- conditions
def initial_rulers : ℕ := 46
def additional_rulers : ℕ := 25

-- question: total rulers in the drawer
def total_rulers : ℕ := initial_rulers + additional_rulers

-- proof statement: prove that total_rulers is 71
theorem rulers_in_drawer : total_rulers = 71 := by
  sorry

end rulers_in_drawer_l267_26742


namespace tangent_line_equation_at_x_zero_l267_26714

noncomputable def curve (x : ℝ) : ℝ := x + Real.exp (2 * x)

theorem tangent_line_equation_at_x_zero :
  ∃ (k b : ℝ), (∀ x : ℝ, curve x = k * x + b) :=
by
  let df := fun (x : ℝ) => (deriv curve x)
  have k : ℝ := df 0
  have b : ℝ := curve 0 - k * 0
  use k, b
  sorry

end tangent_line_equation_at_x_zero_l267_26714


namespace range_of_distance_l267_26755

noncomputable def A (α : ℝ) : ℝ × ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α, 1)
noncomputable def B (β : ℝ) : ℝ × ℝ × ℝ := (2 * Real.cos β, 2 * Real.sin β, 1)

theorem range_of_distance (α β : ℝ) :
  1 ≤ Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 + (3 * Real.sin α - 2 * Real.sin β)^2) ∧
  Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 + (3 * Real.sin α - 2 * Real.sin β)^2) ≤ 5 :=
by
  sorry

end range_of_distance_l267_26755


namespace participating_girls_l267_26778

theorem participating_girls (total_students boys_participation girls_participation participating_students : ℕ)
  (h1 : total_students = 800)
  (h2 : boys_participation = 2)
  (h3 : girls_participation = 3)
  (h4 : participating_students = 550) :
  (4 / total_students) * (boys_participation / 3) * total_students + (4 * girls_participation / 4) * total_students = 4 * 150 :=
by
  sorry

end participating_girls_l267_26778


namespace sin_of_cos_of_angle_l267_26760

-- We need to assume that A is an angle of a triangle, hence A is in the range (0, π).
theorem sin_of_cos_of_angle (A : ℝ) (hA : 0 < A ∧ A < π) (h_cos : Real.cos A = -3/5) : Real.sin A = 4/5 := by
  sorry

end sin_of_cos_of_angle_l267_26760


namespace symmetry_with_respect_to_line_x_eq_1_l267_26744

theorem symmetry_with_respect_to_line_x_eq_1 (f : ℝ → ℝ) :
  ∀ x, f (x - 1) = f (1 - x) ↔ x - 1 = 1 - x :=
by
  sorry

end symmetry_with_respect_to_line_x_eq_1_l267_26744


namespace anne_cleaning_time_l267_26785

noncomputable def cleaning_rates (B A : ℝ) : Prop :=
  B + A = 1 / 4 ∧ B + 2 * A = 1 / 3

theorem anne_cleaning_time (B A : ℝ) (h : cleaning_rates B A) : 
  (1 / A) = 12 :=
by
  sorry

end anne_cleaning_time_l267_26785


namespace friend_spending_l267_26794

-- Definitions based on conditions
def total_spent (you friend : ℝ) : Prop := you + friend = 15
def friend_spent (you friend : ℝ) : Prop := friend = you + 1

-- Prove that the friend's spending equals $8 given the conditions
theorem friend_spending (you friend : ℝ) (htotal : total_spent you friend) (hfriend : friend_spent you friend) : friend = 8 :=
by
  sorry

end friend_spending_l267_26794


namespace subsets_union_l267_26759

theorem subsets_union (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 2^(n-1) + 1) 
  (A : Fin m → Finset (Fin n)) (hA : ∀ i j, i ≠ j → A i ≠ A j) 
  (hB : ∀ i, A i ≠ ∅) : 
  ∃ i j k, i ≠ j ∧ A i ∪ A j = A k := 
sorry

end subsets_union_l267_26759


namespace vector_dot_product_l267_26795

-- Define the vectors
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)

-- Vector addition and scalar multiplication
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The mathematical statement to prove
theorem vector_dot_product : dot_product (vec_add (scalar_mul 2 vec_a) vec_b) vec_a = 6 :=
by
  -- Sorry is used to skip the proof; it's just a placeholder.
  sorry

end vector_dot_product_l267_26795


namespace smallest_arithmetic_mean_divisible_by_1111_l267_26747

theorem smallest_arithmetic_mean_divisible_by_1111 :
  ∃ (n : ℕ), (∀ i : ℕ, i < 9 → 1111 ∣ (n + i)) ∧ (n + 4 = 97) :=
by
  sorry

end smallest_arithmetic_mean_divisible_by_1111_l267_26747


namespace exists_integer_multiple_of_3_2008_l267_26780

theorem exists_integer_multiple_of_3_2008 :
  ∃ k : ℤ, 3 ^ 2008 ∣ (k ^ 3 - 36 * k ^ 2 + 51 * k - 97) :=
sorry

end exists_integer_multiple_of_3_2008_l267_26780


namespace intersection_M_N_l267_26784

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 10 }
def N : Set ℝ := { x | x > 7 ∨ x < 1 }
def MN_intersection : Set ℝ := { x | (-1 ≤ x ∧ x < 1) ∨ (7 < x ∧ x ≤ 10) }

theorem intersection_M_N :
  M ∩ N = MN_intersection :=
by
  sorry

end intersection_M_N_l267_26784


namespace perpendicular_slope_l267_26764

theorem perpendicular_slope (x y : ℝ) : (∃ b : ℝ, 4 * x - 5 * y = 10) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro h
  sorry

end perpendicular_slope_l267_26764


namespace length_of_field_l267_26763

variable (w l : ℝ)
variable (H1 : l = 2 * w)
variable (pond_area : ℝ := 64)
variable (field_area : ℝ := l * w)
variable (H2 : pond_area = (1 / 98) * field_area)

theorem length_of_field : l = 112 :=
by
  sorry

end length_of_field_l267_26763


namespace loss_percentage_l267_26717

/-
Books Problem:
Determine the loss percentage on the first book given:
1. The cost of the first book (C1) is Rs. 280.
2. The total cost of two books is Rs. 480.
3. The second book is sold at a gain of 19%.
4. Both books are sold at the same price.
-/

theorem loss_percentage (C1 C2 SP1 SP2 : ℝ) (h1 : C1 = 280)
  (h2 : C1 + C2 = 480) (h3 : SP2 = C2 * 1.19) (h4 : SP1 = SP2) : 
  (C1 - SP1) / C1 * 100 = 15 := 
by
  sorry

end loss_percentage_l267_26717


namespace tank_filling_l267_26724

theorem tank_filling (A_rate B_rate : ℚ) (hA : A_rate = 1 / 9) (hB : B_rate = 1 / 18) :
  (1 / (A_rate - B_rate)) = 18 :=
by
  sorry

end tank_filling_l267_26724


namespace remainder_of_division_l267_26719

theorem remainder_of_division :
  ∀ (x : ℝ), (3 * x ^ 5 - 2 * x ^ 3 + 5 * x - 8) % (x ^ 2 - 3 * x + 2) = 74 * x - 76 :=
by
  sorry

end remainder_of_division_l267_26719


namespace total_cases_after_three_weeks_l267_26752

theorem total_cases_after_three_weeks (week1_cases week2_cases week3_cases : ℕ) 
  (h1 : week1_cases = 5000)
  (h2 : week2_cases = week1_cases + week1_cases / 10 * 3)
  (h3 : week3_cases = week2_cases - week2_cases / 10 * 2) :
  week1_cases + week2_cases + week3_cases = 16700 := 
by
  sorry

end total_cases_after_three_weeks_l267_26752


namespace point_in_second_quadrant_l267_26770

def isInSecondQuadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant : isInSecondQuadrant (-1) 1 :=
by
  sorry

end point_in_second_quadrant_l267_26770


namespace trajectory_parabola_l267_26783

noncomputable def otimes (x1 x2 : ℝ) : ℝ := (x1 + x2)^2 - (x1 - x2)^2

theorem trajectory_parabola (x : ℝ) (h : 0 ≤ x) : 
  ∃ (y : ℝ), y^2 = 8 * x ∧ (∀ P : ℝ × ℝ, P = (x, y) → (P.snd^2 = 8 * P.fst)) :=
by
  sorry

end trajectory_parabola_l267_26783


namespace profit_percentage_l267_26757

-- Given conditions
def CP : ℚ := 25 / 15
def SP : ℚ := 32 / 12

-- To prove profit percentage is 60%
theorem profit_percentage (CP SP : ℚ) (hCP : CP = 25 / 15) (hSP : SP = 32 / 12) :
  (SP - CP) / CP * 100 = 60 := 
by 
  sorry

end profit_percentage_l267_26757


namespace one_girl_made_a_mistake_l267_26708

variables (c_M c_K c_L c_O : ℤ)

theorem one_girl_made_a_mistake (h₁ : c_M + c_K = c_L + c_O + 12) (h₂ : c_K + c_L = c_M + c_O - 7) :
  false := by
  -- Proof intentionally missing
  sorry

end one_girl_made_a_mistake_l267_26708


namespace problem1_problem2_problem3_problem4_l267_26751

-- Problem (1)
theorem problem1 : (-8 - 6 + 24) = 10 :=
by sorry

-- Problem (2)
theorem problem2 : (-48 / 6 + -21 * (-1 / 3)) = -1 :=
by sorry

-- Problem (3)
theorem problem3 : ((1 / 8 - 1 / 3 + 1 / 4) * -24) = -1 :=
by sorry

-- Problem (4)
theorem problem4 : (-1^4 - (1 + 0.5) * (1 / 3) * (1 - (-2)^2)) = 0.5 :=
by sorry

end problem1_problem2_problem3_problem4_l267_26751


namespace RU_eq_825_l267_26773

variables (P Q R S T U : Type)
variables (PQ QR RP QS SR : ℝ)
variables (RU : ℝ)
variables (hPQ : PQ = 13)
variables (hQR : QR = 30)
variables (hRP : RP = 26)
variables (hQS : QS = 10)
variables (hSR : SR = 20)

theorem RU_eq_825 :
  RU = 8.25 :=
sorry

end RU_eq_825_l267_26773


namespace speed_ratio_l267_26720

theorem speed_ratio (L v_a v_b : ℝ) (h1 : v_a = c * v_b) (h2 : (L / v_a) = (0.8 * L / v_b)) :
  v_a / v_b = 5 / 4 :=
by
  sorry

end speed_ratio_l267_26720


namespace cubic_yard_to_cubic_meter_l267_26745

/-- Define the conversion from yards to meters. -/
def yard_to_meter : ℝ := 0.9144

/-- Theorem stating how many cubic meters are in one cubic yard. -/
theorem cubic_yard_to_cubic_meter :
  (yard_to_meter ^ 3 : ℝ) = 0.7636 :=
by
  sorry

end cubic_yard_to_cubic_meter_l267_26745


namespace Mitya_age_l267_26790

/--
Assume Mitya's current age is M and Shura's current age is S. If Mitya is 11 years older than Shura,
and when Mitya was as old as Shura is now, he was twice as old as Shura,
then prove that M = 27.5.
-/
theorem Mitya_age (S M : ℝ) (h1 : M = S + 11) (h2 : M - S = 2 * (S - (M - S))) : M = 27.5 :=
by
  sorry

end Mitya_age_l267_26790


namespace M_subset_N_l267_26754

def M (x : ℝ) : Prop := ∃ k : ℤ, x = (↑k * Real.pi / 2) + (Real.pi / 4)
def N (x : ℝ) : Prop := ∃ k : ℤ, x = (↑k * Real.pi / 4) + (Real.pi / 2)

theorem M_subset_N : ∀ x, M x → N x := 
by
  sorry

end M_subset_N_l267_26754


namespace trajectory_of_midpoint_l267_26722

theorem trajectory_of_midpoint (M : ℝ × ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ) :
  (P.1^2 + P.2^2 = 1) ∧
  (P.1 = M.1 ∧ P.2 = 2 * M.2) ∧ 
  (N.1 = P.1 ∧ N.2 = 0) ∧ 
  (M.1 = (P.1 + N.1) / 2 ∧ M.2 = (P.2 + N.2) / 2)
  → M.1^2 + 4 * M.2^2 = 1 := 
by
  sorry

end trajectory_of_midpoint_l267_26722


namespace simplify_and_rationalize_denominator_l267_26762

noncomputable def problem (x : ℚ) := 
  x = 1 / (1 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_denominator (x : ℚ) (h: problem x) :
  x = (Real.sqrt 5 - 1) / 4 :=
by
  sorry

end simplify_and_rationalize_denominator_l267_26762


namespace union_P_complement_Q_l267_26735

open Set

def P : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
def Q : Set ℝ := { x | x^2 ≥ 4 }
def R : Set ℝ := { x | -2 < x ∧ x < 2 }
def PQ_union : Set ℝ := P ∪ R

theorem union_P_complement_Q : PQ_union = { x | -2 < x ∧ x ≤ 3 } :=
by sorry

end union_P_complement_Q_l267_26735


namespace sum_of_two_primes_is_multiple_of_six_l267_26771

theorem sum_of_two_primes_is_multiple_of_six
  (p q r : ℕ)
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) (hr_gt_3 : r > 3)
  (h_sum_prime : Nat.Prime (p + q + r)) : 
  (p + q) % 6 = 0 ∨ (p + r) % 6 = 0 ∨ (q + r) % 6 = 0 :=
sorry

end sum_of_two_primes_is_multiple_of_six_l267_26771


namespace cubic_roots_inequalities_l267_26779

theorem cubic_roots_inequalities 
  (a b c d : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ z : ℂ, (a * z^3 + b * z^2 + c * z + d = 0) → z.re < 0) :
  a * b > 0 ∧ b * c - a * d > 0 ∧ a * d > 0 :=
by
  sorry

end cubic_roots_inequalities_l267_26779


namespace sum_of_roots_eq_l267_26797

theorem sum_of_roots_eq (k : ℝ) : ∃ x1 x2 : ℝ, (2 * x1 ^ 2 - 3 * x1 + k = 7) ∧ (2 * x2 ^ 2 - 3 * x2 + k = 7) ∧ (x1 + x2 = 3 / 2) :=
by sorry

end sum_of_roots_eq_l267_26797


namespace Albert_more_than_Joshua_l267_26715

def Joshua_rocks : ℕ := 80

def Jose_rocks : ℕ := Joshua_rocks - 14

def Albert_rocks : ℕ := Jose_rocks + 20

theorem Albert_more_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_than_Joshua_l267_26715


namespace forgotten_angles_correct_l267_26798

theorem forgotten_angles_correct (n : ℕ) (h1 : (n - 2) * 180 = 2520) (h2 : 2345 + 175 = 2520) : 
  ∃ a b : ℕ, a + b = 175 :=
by
  sorry

end forgotten_angles_correct_l267_26798


namespace average_pages_per_day_l267_26711

variable (total_pages : ℕ := 160)
variable (pages_read : ℕ := 60)
variable (days_left : ℕ := 5)

theorem average_pages_per_day : (total_pages - pages_read) / days_left = 20 := by
  sorry

end average_pages_per_day_l267_26711


namespace percentage_increase_l267_26733

theorem percentage_increase (a : ℕ) (x : ℝ) (b : ℝ) (r : ℝ) 
    (h1 : a = 1500) 
    (h2 : r = 0.6) 
    (h3 : b = 1080) 
    (h4 : a * (1 + x / 100) * r = b) : 
    x = 20 := 
by 
  sorry

end percentage_increase_l267_26733


namespace probability_of_three_even_numbers_l267_26775

theorem probability_of_three_even_numbers (n : ℕ) (k : ℕ) (p_even : ℚ) (p_odd : ℚ) (comb : ℕ → ℕ → ℕ) 
    (h_n : n = 5) (h_k : k = 3) (h_p_even : p_even = 1/2) (h_p_odd : p_odd = 1/2) 
    (h_comb : comb 5 3 = 10) :
    comb n k * (p_even ^ k) * (p_odd ^ (n - k)) = 5 / 16 :=
by sorry

end probability_of_three_even_numbers_l267_26775


namespace fault_line_movement_l267_26709

theorem fault_line_movement
  (moved_past_year : ℝ)
  (moved_year_before : ℝ)
  (h1 : moved_past_year = 1.25)
  (h2 : moved_year_before = 5.25) :
  moved_past_year + moved_year_before = 6.50 :=
by
  sorry

end fault_line_movement_l267_26709


namespace triangle_sides_ratio_l267_26705

theorem triangle_sides_ratio (A B C : ℝ) (a b c : ℝ)
  (h1 : a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 2 * a)
  (ha_pos : a > 0) : b / a = Real.sqrt 2 :=
sorry

end triangle_sides_ratio_l267_26705


namespace seq_contains_exactly_16_twos_l267_26728

-- Define a helper function to count occurrences of a digit in a number
def count_digit (d : Nat) (n : Nat) : Nat :=
  (n.digits 10).count d

-- Define a function to sum occurrences of the digit '2' in a list of numbers
def total_twos_in_sequence (seq : List Nat) : Nat :=
  seq.foldl (λ acc n => acc + count_digit 2 n) 0

-- Define the sequence we are interested in
def seq : List Nat := [2215, 2216, 2217, 2218, 2219, 2220, 2221]

-- State the theorem we need to prove
theorem seq_contains_exactly_16_twos : total_twos_in_sequence seq = 16 := 
by
  -- We do not provide the proof here according to the given instructions
  sorry

end seq_contains_exactly_16_twos_l267_26728


namespace polynomial_solution_l267_26723

noncomputable def P : ℝ → ℝ := sorry

theorem polynomial_solution (x : ℝ) :
  (∃ P : ℝ → ℝ, (∀ x, P x = (P 0) + (P 1) * x + (P 2) * x^2) ∧ 
  (P (-2) = 4)) →
  (P x = (4 * x^2 - 6 * x) / 7) :=
by
  sorry

end polynomial_solution_l267_26723


namespace greatest_x_lcm_105_l267_26736

theorem greatest_x_lcm_105 (x : ℕ) (h_lcm : lcm (lcm x 15) 21 = 105) : x ≤ 105 := 
sorry

end greatest_x_lcm_105_l267_26736


namespace c_negative_l267_26737

theorem c_negative (a b c : ℝ) (h₁ : a + b + c < 0) (h₂ : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) : 
  c < 0 :=
sorry

end c_negative_l267_26737


namespace minimum_value_l267_26743

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 2) : 
  (1 / m + 2 / n) ≥ 4 :=
sorry

end minimum_value_l267_26743


namespace perimeter_correct_l267_26739

open EuclideanGeometry

noncomputable def perimeter_of_figure : ℝ := 
  let AB : ℝ := 6
  let BC : ℝ := AB
  let AD : ℝ := AB / 2
  let DC : ℝ := AD
  let DE : ℝ := AD
  let EA : ℝ := DE
  let EF : ℝ := EA / 2
  let FG : ℝ := EF
  let GH : ℝ := FG / 2
  let HJ : ℝ := GH
  let JA : ℝ := HJ
  AB + BC + DC + DE + EF + FG + GH + HJ + JA

theorem perimeter_correct : perimeter_of_figure = 23.25 :=
by
  -- proof steps would go here, but are not required for this problem transformation
  sorry

end perimeter_correct_l267_26739


namespace total_gray_trees_l267_26772

theorem total_gray_trees :
  (∃ trees_first trees_second trees_third gray1 gray2,
    trees_first = 100 ∧
    trees_second = 90 ∧
    trees_third = 82 ∧
    gray1 = trees_first - trees_third ∧
    gray2 = trees_second - trees_third ∧
    trees_first + trees_second - 2 * trees_third = gray1 + gray2) →
  (gray1 + gray2 = 26) :=
by
  intros
  sorry

end total_gray_trees_l267_26772


namespace roller_coaster_cost_l267_26732

variable (ferris_wheel_rides : Nat) (log_ride_rides : Nat) (rc_rides : Nat)
variable (ferris_wheel_cost : Nat) (log_ride_cost : Nat)
variable (initial_tickets : Nat) (additional_tickets : Nat)
variable (total_needed_tickets : Nat)

theorem roller_coaster_cost :
  ferris_wheel_rides = 2 →
  log_ride_rides = 7 →
  rc_rides = 3 →
  ferris_wheel_cost = 2 →
  log_ride_cost = 1 →
  initial_tickets = 20 →
  additional_tickets = 6 →
  total_needed_tickets = initial_tickets + additional_tickets →
  let total_ride_costs := ferris_wheel_rides * ferris_wheel_cost + log_ride_rides * log_ride_cost
  let rc_cost := (total_needed_tickets - total_ride_costs) / rc_rides
  rc_cost = 5 := by
  sorry

end roller_coaster_cost_l267_26732


namespace jill_second_bus_time_l267_26788

-- Define constants representing the times
def wait_time_first_bus : ℕ := 12
def ride_time_first_bus : ℕ := 30

-- Define a function to calculate the total time for the first bus
def total_time_first_bus (wait : ℕ) (ride : ℕ) : ℕ :=
  wait + ride

-- Define a function to calculate the time for the second bus
def time_second_bus (total_first_bus_time : ℕ) : ℕ :=
  total_first_bus_time / 2

-- The theorem to prove
theorem jill_second_bus_time : 
  time_second_bus (total_time_first_bus wait_time_first_bus ride_time_first_bus) = 21 := by
  sorry

end jill_second_bus_time_l267_26788


namespace sin_order_l267_26726

theorem sin_order :
  ∀ (sin₁ sin₂ sin₃ sin₄ sin₆ : ℝ),
  sin₁ = Real.sin 1 ∧ 
  sin₂ = Real.sin 2 ∧ 
  sin₃ = Real.sin 3 ∧ 
  sin₄ = Real.sin 4 ∧ 
  sin₆ = Real.sin 6 →
  sin₂ > sin₁ ∧ sin₁ > sin₃ ∧ sin₃ > sin₆ ∧ sin₆ > sin₄ :=
by
  sorry

end sin_order_l267_26726


namespace taqeesha_grade_l267_26729

theorem taqeesha_grade (s : ℕ → ℕ) (h1 : (s 16) = 77) (h2 : (s 17) = 78) : s 17 - s 16 = 94 :=
by
  -- Add definitions and sorry to skip the proof
  sorry

end taqeesha_grade_l267_26729


namespace proof_4_minus_a_l267_26713

theorem proof_4_minus_a :
  ∀ (a b : ℚ),
    (5 + a = 7 - b) →
    (3 + b = 8 + a) →
    4 - a = 11 / 2 :=
by
  intros a b h1 h2
  sorry

end proof_4_minus_a_l267_26713


namespace total_distance_covered_l267_26787

noncomputable def speed_train_a : ℚ := 80          -- Speed of Train A in kmph
noncomputable def speed_train_b : ℚ := 110         -- Speed of Train B in kmph
noncomputable def duration : ℚ := 15               -- Duration in minutes
noncomputable def conversion_factor : ℚ := 60      -- Conversion factor from hours to minutes

theorem total_distance_covered : 
    (speed_train_a / conversion_factor) * duration + 
    (speed_train_b / conversion_factor) * duration = 47.5 :=
by
  sorry

end total_distance_covered_l267_26787


namespace owners_riding_to_total_ratio_l267_26725

theorem owners_riding_to_total_ratio (R W : ℕ) (h1 : 4 * R + 6 * W = 90) (h2 : R + W = 18) : R / (R + W) = 1 / 2 :=
by
  sorry

end owners_riding_to_total_ratio_l267_26725


namespace trig_expression_l267_26793

theorem trig_expression (θ : ℝ) (h : Real.tan (Real.pi / 4 + θ) = 3) : 
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
by sorry

end trig_expression_l267_26793


namespace steven_falls_correct_l267_26746

/-
  We will model the problem where we are given the conditions about the falls of Steven, Stephanie,
  and Sonya, and then prove that the number of times Steven fell is 3.
-/

variables (S : ℕ) -- Steven's falls

-- Conditions
def stephanie_falls := S + 13
def sonya_falls := 6 
def sonya_condition := 6 = (stephanie_falls / 2) - 2

-- Theorem statement
theorem steven_falls_correct : S = 3 :=
by {
  -- Note: the actual proof steps would go here, but are omitted per instructions
  sorry
}

end steven_falls_correct_l267_26746


namespace no_d1_d2_multiple_of_7_l267_26766
open Function

theorem no_d1_d2_multiple_of_7 (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 100) :
  let d1 := a^2 + 3^a + a * 3^((a+1)/2)
  let d2 := a^2 + 3^a - a * 3^((a+1)/2)
  ¬(d1 * d2 % 7 = 0) :=
by
  let d1 := a^2 + 3^a + a * 3^((a+1)/2)
  let d2 := a^2 + 3^a - a * 3^((a+1)/2)
  sorry

end no_d1_d2_multiple_of_7_l267_26766


namespace length_XW_l267_26768

theorem length_XW {XY XZ YZ XW : ℝ}
  (hXY : XY = 15)
  (hXZ : XZ = 17)
  (hAngle : XY^2 + YZ^2 = XZ^2)
  (hYZ : YZ = 8) :
  XW = 15 :=
by
  sorry

end length_XW_l267_26768


namespace peanuts_added_l267_26707

theorem peanuts_added (initial_peanuts final_peanuts added_peanuts : ℕ) 
(h1 : initial_peanuts = 10) 
(h2 : final_peanuts = 18) 
(h3 : final_peanuts = initial_peanuts + added_peanuts) : 
added_peanuts = 8 := 
by {
  sorry
}

end peanuts_added_l267_26707


namespace suitable_survey_set_l267_26749

def Survey1 := "Investigate the lifespan of a batch of light bulbs"
def Survey2 := "Investigate the household income situation in a city"
def Survey3 := "Investigate the vision of students in a class"
def Survey4 := "Investigate the efficacy of a certain drug"

-- Define what it means for a survey to be suitable for sample surveys
def suitable_for_sample_survey (survey : String) : Prop :=
  survey = Survey1 ∨ survey = Survey2 ∨ survey = Survey4

-- The question is to prove that the surveys suitable for sample surveys include exactly (1), (2), and (4).
theorem suitable_survey_set :
  {Survey1, Survey2, Survey4} = {s : String | suitable_for_sample_survey s} :=
by
  sorry

end suitable_survey_set_l267_26749


namespace sum_due_is_42_l267_26786

-- Define the conditions
def BD : ℝ := 42
def TD : ℝ := 36

-- Statement to prove
theorem sum_due_is_42 (H1 : BD = 42) (H2 : TD = 36) : ∃ (FV : ℝ), FV = 42 := by
  -- Proof Placeholder
  sorry

end sum_due_is_42_l267_26786


namespace problem_one_problem_two_l267_26738

variable {α : ℝ}

theorem problem_one (h : Real.tan (π + α) = -1 / 2) :
  (2 * Real.cos (π - α) - 3 * Real.sin (π + α)) / (4 * Real.cos (α - 2 * π) + Real.sin (4 * π - α)) = -7 / 9 :=
sorry

theorem problem_two (h : Real.tan (π + α) = -1 / 2) :
  Real.sin (α - 7 * π) * Real.cos (α + 5 * π) = -2 / 5 :=
sorry

end problem_one_problem_two_l267_26738


namespace cakes_remaining_l267_26731

theorem cakes_remaining (initial_cakes : ℕ) (bought_cakes : ℕ) (h1 : initial_cakes = 169) (h2 : bought_cakes = 137) : initial_cakes - bought_cakes = 32 :=
by
  sorry

end cakes_remaining_l267_26731


namespace real_part_of_z_given_condition_l267_26703

open Complex

noncomputable def real_part_of_z (z : ℂ) : ℝ :=
  z.re

theorem real_part_of_z_given_condition :
  ∀ (z : ℂ), (i * (z + 1) = -3 + 2 * i) → real_part_of_z z = 1 :=
by
  intro z h
  sorry

end real_part_of_z_given_condition_l267_26703


namespace solve_inequality_l267_26702

-- Declare the necessary conditions as variables in Lean
variables (a c : ℝ)

-- State the Lean theorem
theorem solve_inequality :
  (∀ x : ℝ, (ax^2 + 5 * x + c > 0) ↔ (1/3 < x ∧ x < 1/2)) →
  a < 0 →
  a = -6 ∧ c = -1 :=
  sorry

end solve_inequality_l267_26702


namespace total_pieces_l267_26727

def gum_packages : ℕ := 28
def candy_packages : ℕ := 14
def pieces_per_package : ℕ := 6

theorem total_pieces : (gum_packages * pieces_per_package) + (candy_packages * pieces_per_package) = 252 :=
by
  sorry

end total_pieces_l267_26727
