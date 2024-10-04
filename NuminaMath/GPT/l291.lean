import Mathlib
import Mathlib.Algebra.ArithmeticMean
import Mathlib.Algebra.CubicDiscriminant
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Ring
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialStirling
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Init.Data.Int.Basic
import Mathlib.Prob.Basic
import Mathlib.Probability.Independence
import Mathlib.Set.Basic
import Mathlib.Tactic
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Pigeonhole
import Mathlib.Utils.UnsafeCast

namespace area_of_triangle_is_168_l291_291491

-- Define the curve equation
def curve_eq (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define the x-intercepts
def x_intercepts (y : ℝ) : Prop := y = 0

-- Define the y-intercept
def y_intercept (x : ℝ) : ℝ := curve_eq 0

-- Define the base of the triangle (distance between x-intercepts)
def base : ℝ := 4 - (-3)

-- Define the height of the triangle (y-intercept value)
def height : ℝ := y_intercept 0

-- Define the area calculation for the triangle
def triangle_area : ℝ := (1 / 2) * base * height

-- The theorem to prove the area of the triangle is 168
theorem area_of_triangle_is_168 : triangle_area = 168 :=
by sorry

end area_of_triangle_is_168_l291_291491


namespace perp_implies_parallel_l291_291579

variables {Point Line Plane : Type}
variable [Non-coincident_planes : ∀ (α β : Plane), α ≠ β]
variable [Non-coincident_lines : ∀ (m n : Line), m ≠ n]
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- Define the conditions
axiom m_perp_alpha (m : Line) (α : Plane) : perpendicular m α
axiom m_perp_beta (m : Line) (β : Plane) : perpendicular m β
axiom alpha_ne_beta (α β : Plane) : α ≠ β

-- Define the statement to be proved
theorem perp_implies_parallel (m : Line) (α β : Plane) (h1 : perpendicular m α) (h2 : perpendicular m β) : parallel α β := 
sorry

end perp_implies_parallel_l291_291579


namespace cost_of_tissues_l291_291468
-- Import the entire Mathlib library

-- Define the context and the assertion without computing the proof details
theorem cost_of_tissues
  (n_tp : ℕ) -- Number of toilet paper rolls
  (c_tp : ℝ) -- Cost per toilet paper roll
  (n_pt : ℕ) -- Number of paper towels rolls
  (c_pt : ℝ) -- Cost per paper towel roll
  (n_t : ℕ) -- Number of tissue boxes
  (T : ℝ) -- Total cost of all items
  (H_tp : n_tp = 10) -- Given: 10 rolls of toilet paper
  (H_c_tp : c_tp = 1.5) -- Given: $1.50 per roll of toilet paper
  (H_pt : n_pt = 7) -- Given: 7 rolls of paper towels
  (H_c_pt : c_pt = 2) -- Given: $2 per roll of paper towel
  (H_t : n_t = 3) -- Given: 3 boxes of tissues
  (H_T : T = 35) -- Given: total cost is $35
  : (T - (n_tp * c_tp + n_pt * c_pt)) / n_t = 2 := -- Conclusion: the cost of one box of tissues is $2
by {
  sorry -- Proof details to be supplied here
}

end cost_of_tissues_l291_291468


namespace sum_log_divisors_eq_522_l291_291358

theorem sum_log_divisors_eq_522 (n : ℕ) : 
  (∑ a in Finset.range (n + 1), ∑ b in Finset.range (n + 1), (a * Real.log10 2 + b * Real.log10 3)) = 522 →
  n = 8 :=
by
  sorry

end sum_log_divisors_eq_522_l291_291358


namespace second_largest_number_is_9723_l291_291758

/-- The set of digits allowed in our number -/
def allowed_digits : finset ℕ := {2, 3, 7, 9}

/-- Condition that a number consists only of the allowed digits without repetition -/
def valid_number (n : ℕ) : Prop :=
  let digits := Int.to_nat <$> Int.digits 10 n
  digits.to_finset = allowed_digits

/-- The number 9723 is the second largest number formed by the digits 2, 3, 7, and 9 -/
theorem second_largest_number_is_9723 : ∃ (n : ℕ), valid_number n ∧ ∀ m, valid_number m → m < 9732 → m ≤ 9723 :=
sorry

end second_largest_number_is_9723_l291_291758


namespace jane_stick_length_l291_291715

variable (P U S J F : ℕ)
variable (h1 : P = 30)
variable (h2 : U = P - 7)
variable (h3 : U = S / 2)
variable (h4 : F = 2 * 12)
variable (h5 : J = S - F)

theorem jane_stick_length : J = 22 := by
  sorry

end jane_stick_length_l291_291715


namespace watermelon_and_banana_weight_l291_291381

variables (w b : ℕ)
variables (h1 : 2 * w + b = 8100)
variables (h2 : 2 * w + 3 * b = 8300)

theorem watermelon_and_banana_weight (Hw : w = 4000) (Hb : b = 100) :
  2 * w + b = 8100 ∧ 2 * w + 3 * b = 8300 :=
by
  sorry

end watermelon_and_banana_weight_l291_291381


namespace complex_arith_example_l291_291488

theorem complex_arith_example : (7 - 3*complex.i) - 3*(2 + 5*complex.i) = 1 - 18*complex.i :=
by
  sorry

end complex_arith_example_l291_291488


namespace min_composite_pair_diff_l291_291821

-- Definitions required based on the conditions in a)
def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, 2 ≤ p ∧ 2 ≤ q ∧ p * q = n

def is_valid_composite_pair (a b : ℕ) : Prop :=
  is_composite a ∧ is_composite b ∧ a + b = 91

-- The theorem to state the required proof
theorem min_composite_pair_diff :
  ∃ a b : ℕ, is_valid_composite_pair a b ∧ (abs (a - b) = 7) :=
sorry

end min_composite_pair_diff_l291_291821


namespace shortest_multicolored_cycle_l291_291704

-- Let's define the conditions and the main goal
theorem shortest_multicolored_cycle (s : ℕ) (a : fin s → vertex) (b : fin s → vertex) (H : multicolored_cycle (a, b)) :
  s = 2 := 
by {
  -- Proof omitted, add actual proof here
  sorry
}

end shortest_multicolored_cycle_l291_291704


namespace sum_floor_log_base_3_l291_291518

theorem sum_floor_log_base_3 : 
  (∑ N in Finset.range (2048 + 1), (⌊Real.log N / Real.log 3⌋)) = 12120 :=
by
  sorry

end sum_floor_log_base_3_l291_291518


namespace find_n_l291_291939

theorem find_n (n : ℝ) (x : ℝ) (hx1 : log 10 (sin x) + log 10 (cos x) = -2)
(hx2 : log 10 (sin x + cos x) = 1 / 2 * (log 10 n - 2)) : n ≈ 102.33 := sorry

end find_n_l291_291939


namespace probability_of_selection_l291_291636

theorem probability_of_selection : 
  ∀ (n k : ℕ), n = 121 ∧ k = 20 → (P : ℚ) = 20 / 121 :=
by
  intros n k h
  sorry

end probability_of_selection_l291_291636


namespace thyme_leaves_per_pot_l291_291088

-- Definitions for the conditions
def basil_pots := 3
def basil_leaves_per_pot := 4
def rosemary_pots := 9
def rosemary_leaves_per_pot := 18
def thyme_pots := 6
def total_leaves := 354

-- Equation for total basil and rosemary leaves
def basil_leaves := basil_pots * basil_leaves_per_pot
def rosemary_leaves := rosemary_pots * rosemary_leaves_per_pot

-- Equation for total thyme leaves
def thyme_leaves (x : ℕ) := thyme_pots * x

-- Total leaves equation
def total (x : ℕ) := basil_leaves + rosemary_leaves + thyme_leaves x

-- Proof statement
theorem thyme_leaves_per_pot : ∃ x, x = 30 ∧ total x = total_leaves :=
by 
  use 30
  split
  . rfl
  . sorry

end thyme_leaves_per_pot_l291_291088


namespace units_digit_of_n_l291_291906

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : m % 10 = 4) : n % 10 = 4 :=
by
  sorry

end units_digit_of_n_l291_291906


namespace original_price_l291_291837

theorem original_price 
  (SP : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h1 : SP = 15)
  (h2 : gain_percent = 0.50)
  (h3 : SP = P * (1 + gain_percent)) :
  P = 10 :=
by
  sorry

end original_price_l291_291837


namespace pentagon_area_sum_l291_291346

theorem pentagon_area_sum (a b c d e m n : ℕ)
  (h1 : a = 2) (h2 : b = 2) (h3 : c = 2) (h4 : d = 2) (h5 : e = 2)
  (h6 : m = 11) (h7 : n = 12)
  (area_eq : sqrt m + sqrt n = sqrt(27))
  : m + n = 23 :=
begin
  sorry
end

end pentagon_area_sum_l291_291346


namespace scientific_notation_example_l291_291828

theorem scientific_notation_example : 3790000 = 3.79 * 10^6 := 
sorry

end scientific_notation_example_l291_291828


namespace count_solutions_l291_291601

noncomputable def num_solutions : ℕ :=
  let eq1 (x y : ℝ) := 2 * x + 5 * y = 10
  let eq2 (x y : ℝ) := abs (abs (x + 1) - abs (y - 1)) = 1
  sorry

theorem count_solutions : num_solutions = 2 := by
  sorry

end count_solutions_l291_291601


namespace number_of_persons_l291_291036

theorem number_of_persons (n : ℕ) (h : n * (n - 1) / 2 = 78) : n = 13 :=
sorry

end number_of_persons_l291_291036


namespace days_to_clear_messages_l291_291689

theorem days_to_clear_messages 
  (initial_messages : ℕ)
  (messages_read_per_day : ℕ)
  (new_messages_per_day : ℕ) 
  (net_messages_cleared_per_day : ℕ)
  (d : ℕ) :
  initial_messages = 98 →
  messages_read_per_day = 20 →
  new_messages_per_day = 6 →
  net_messages_cleared_per_day = messages_read_per_day - new_messages_per_day →
  d = initial_messages / net_messages_cleared_per_day →
  d = 7 :=
by
  intros h_initial h_read h_new h_net h_days
  sorry

end days_to_clear_messages_l291_291689


namespace sum_of_distances_l291_291348

structure Quadrilateral :=
(A B C D : Point)
(convex : Convex ABCD)
(no_parallel_sides : ¬ (AreParallel (A, B) (C, D)) ∧ ¬ (AreParallel (B, C) (A, D)))

noncomputable def distance_to_sides (P : Point) (ABCD : Quadrilateral) : ℝ :=
  d(P, line(ABCD.A, ABCD.B)) + d(P, line(ABCD.B, ABCD.C)) +
  d(P, line(ABCD.C, ABCD.D)) + d(P, line(ABCD.D, ABCD.A))

theorem sum_of_distances 
  (ABCD : Quadrilateral) 
  (P1 P2 : Point)
  (HP1_in : inside_quadrilateral P1 ABCD)
  (HP2_in : inside_quadrilateral P2 ABCD)
  (m : ℝ) 
  (Hdist1 : distance_to_sides P1 ABCD = m)
  (Hdist2 : distance_to_sides P2 ABCD = m) :
  ∀ P (H_on_segment : on_segment P P1 P2), distance_to_sides P ABCD = m :=
by
  sorry

end sum_of_distances_l291_291348


namespace calculate_vans_l291_291437

theorem calculate_vans (buses : ℕ) (vans_capacity : ℕ) (bus_capacity : ℕ) (total_people : ℕ) 
  (h1 : buses = 10) 
  (h2 : vans_capacity = 8)
  (h3 : bus_capacity = 27)
  (h4 : total_people = 342) :
  (total_people - buses * bus_capacity) / vans_capacity = 9 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end calculate_vans_l291_291437


namespace number_of_true_propositions_two_l291_291264

/-- Definitions based on the conditions -/
variable (a b c : Vector) -- non-zero vectors
variable (P A B C O : Point)
variable (λ μ : Real) -- real numbers
variables (h1: λ ≠ 0 ∧ μ ≠ 0)

def propositions := !([3.noncollinear_points A B C, O A B C,
  linarith λ μ ] ) -- Example definitions based on provided analysis

theorem number_of_true_propositions_two :
  let prop1 := ¬is_basis ℝ [a, b, c] ∧ coplanar a b c in
  let prop2 := ¬is_basis ℝ [a, b] ∧ collinear a b in
  let prop3 := (∀ O A B C, ‖O A B C ‖ P = (2 *  ‖OA‖ -  2 * ‖OB‖  - ‖2 OC‖))  →  coplanar P A B C in
  let prop4 := ¬is_collinear (a b) ∧ c = λ a + μ b → is_basis ℝ [a, b, c] in
  let prop5 := is_basis ℝ [a, b, c] ∧ is_basis ℝ [a+b, b+c+2a, c+a] in
  (1.prop1 ∧ 2.prop1)∧ ¬(prop3 ∨ prop4 ∨ prop5)  := true sorry -- Concludes that two are true.
  
end number_of_true_propositions_two_l291_291264


namespace jia_profits_1_yuan_l291_291272

-- Definition of the problem conditions
def initial_cost : ℝ := 1000
def profit_rate : ℝ := 0.1
def loss_rate : ℝ := 0.1
def resale_rate : ℝ := 0.9

-- Defined transactions with conditions
def jia_selling_price1 : ℝ := initial_cost * (1 + profit_rate)
def yi_selling_price_to_jia : ℝ := jia_selling_price1 * (1 - loss_rate)
def jia_selling_price2 : ℝ := yi_selling_price_to_jia * resale_rate

-- Final net income calculation
def jia_net_income : ℝ := -initial_cost + jia_selling_price1 - yi_selling_price_to_jia + jia_selling_price2

-- Lean statement to be proved
theorem jia_profits_1_yuan : jia_net_income = 1 := sorry

end jia_profits_1_yuan_l291_291272


namespace find_3a_plus_3b_l291_291608

theorem find_3a_plus_3b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 8 * a + 2 * b = 50) :
  3 * a + 3 * b = 73 / 2 := 
sorry

end find_3a_plus_3b_l291_291608


namespace apples_to_pears_value_l291_291739

/-- Suppose 1/2 of 12 apples are worth as much as 10 pears. -/
def apples_per_pears_ratio : ℚ := 10 / (1 / 2 * 12)

/-- Prove that 3/4 of 6 apples are worth as much as 7.5 pears. -/
theorem apples_to_pears_value : (3 / 4 * 6) * apples_per_pears_ratio = 7.5 := 
by
  sorry

end apples_to_pears_value_l291_291739


namespace length_of_segment_A_l291_291183

def A : (ℝ × ℝ × ℝ) := (3, 5, -7)
def B : (ℝ × ℝ × ℝ) := (-2, 4, 3)

def A'_projection : (ℝ × ℝ × ℝ) := (3, 0, 0)
def B'_projection : (ℝ × ℝ × ℝ) := (0, 0, 3)

def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Math.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2)

theorem length_of_segment_A'_B' : distance A'_projection B'_projection = 3 * Math.sqrt 2 :=
  sorry

end length_of_segment_A_l291_291183


namespace problem_statement_l291_291941

theorem problem_statement (x : ℝ) (h : 0 < x) : x + 2016^2016 / x^2016 ≥ 2017 := 
by
  sorry

end problem_statement_l291_291941


namespace sin_cos_from_tan_in_second_quadrant_l291_291577

theorem sin_cos_from_tan_in_second_quadrant (α : ℝ) 
  (h1 : Real.tan α = -2) 
  (h2 : α ∈ Set.Ioo (π / 2) π) : 
  Real.sin α = 2 * Real.sqrt 5 / 5 ∧ Real.cos α = -Real.sqrt 5 / 5 :=
by
  sorry

end sin_cos_from_tan_in_second_quadrant_l291_291577


namespace condition_on_a_and_b_l291_291285

variable (x a b : ℝ)

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem condition_on_a_and_b
  (h1 : a > 0)
  (h2 : b > 0) :
  (∀ x : ℝ, |f x + 3| < a ↔ |x - 1| < b) ↔ (b^2 + 2*b + 3 ≤ a) :=
sorry

end condition_on_a_and_b_l291_291285


namespace anna_likes_numbers_divisible_by_5_number_of_possible_unit_digits_l291_291302

theorem anna_likes_numbers_divisible_by_5 :
  ∀ n : ℕ, (∃ k : ℕ, n = 5 * k) → ((n % 10 = 0) ∨ (n % 10 = 5)) :=
by sorry

theorem number_of_possible_unit_digits :
  ∃ unit_digits : Finset ℕ, (∀ n : ℕ, (∃ k : ℕ, n = 5 * k) → (n % 10 ∈ unit_digits)) ∧ unit_digits.card = 2 :=
by
  use {0, 5}
  split
  · intros n h
    obtain ⟨k, rfl⟩ := h
    have : n % 10 = 0 ∨ n % 10 = 5,
    exact anna_likes_numbers_divisible_by_5 n ⟨k, rfl⟩
    simp [this]
  · simp

end anna_likes_numbers_divisible_by_5_number_of_possible_unit_digits_l291_291302


namespace left_handed_ratio_l291_291308

-- Given the conditions:
-- total number of players
def total_players : ℕ := 70
-- number of throwers who are all right-handed 
def throwers : ℕ := 37 
-- total number of right-handed players
def right_handed : ℕ := 59

-- Define the necessary variables based on the given conditions.
def non_throwers : ℕ := total_players - throwers
def non_throwing_right_handed : ℕ := right_handed - throwers
def left_handed_non_throwers : ℕ := non_throwers - non_throwing_right_handed

-- State the theorem to prove that the ratio of 
-- left-handed non-throwers to the rest of the team (excluding throwers) is 1:3
theorem left_handed_ratio : 
  (left_handed_non_throwers : ℚ) / (non_throwers : ℚ) = 1 / 3 := by
    sorry

end left_handed_ratio_l291_291308


namespace find_k_distance_from_N_to_PM_l291_291979

-- Define the points P, M, and N
def P : ℝ × ℝ × ℝ := (-2, 0, 2)
def M : ℝ × ℝ × ℝ := (-1, 1, 2)
def N : ℝ × ℝ × ℝ := (-3, 0, 4)

-- Define the vectors a and b
def vector_a : ℝ × ℝ × ℝ := (M.1 - P.1, M.2 - P.2, M.3 - P.3)  -- (1, 1, 0)
def vector_b : ℝ × ℝ × ℝ := (N.1 - P.1, N.2 - P.2, N.3 - P.3)  -- (-1, 0, 2)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- The first part of the problem: finding k
theorem find_k (k : ℝ) :
  dot_product (k • vector_a + vector_b) (k • vector_a - 2 • vector_b) = 0 ↔ (k = 2 ∨ k = -5/2) :=
by
  let a := vector_a
  let b := vector_b
  sorry

-- The second part of the problem: finding the distance from N to the line PM
theorem distance_from_N_to_PM :
  let a := vector_a
  let b := vector_b
  let u := (1 / real.sqrt 2) • (1, 1, 0) in  -- direction vector of PM
  real.sqrt (dot_product b b - (dot_product b u) ^ 2) = 3 * real.sqrt 2 / 2 :=
by
  let a := vector_a
  let b := vector_b
  sorry

end find_k_distance_from_N_to_PM_l291_291979


namespace multiplicative_order_1_plus_pq_mod_p2q3_l291_291293

-- Given conditions
variables {p q : ℕ} (hp : p.prime) (hq : q.prime) (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) (h_distinct : p ≠ q)

-- Theorem statement
theorem multiplicative_order_1_plus_pq_mod_p2q3 : 
  (multiplicativeOrder (1 + p * q) (p * p * q * q * q)) = p * q * q :=
sorry

end multiplicative_order_1_plus_pq_mod_p2q3_l291_291293


namespace total_votes_l291_291039

theorem total_votes (T F A : ℝ)
  (h1 : F = A + 68)
  (h2 : A = 0.40 * T)
  (h3 : T = F + A) :
  T = 340 :=
by sorry

end total_votes_l291_291039


namespace find_fg_length_l291_291460

theorem find_fg_length
  (A B C D E F G : Point)
  (hABC : EquilateralTriangle A B C)
  (hADE : EquilateralTriangle A D E)
  (hBDG : EquilateralTriangle B D G)
  (hCEF : EquilateralTriangle C E F)
  (side_length_ABC : AB = 10)
  (AD_length : AD = 3) :
  FG = 4 :=
sorry

end find_fg_length_l291_291460


namespace marie_messages_days_l291_291691

theorem marie_messages_days (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) (days : ℕ) :
  initial_messages = 98 ∧ read_per_day = 20 ∧ new_per_day = 6 → days = 7 :=
by
  sorry

end marie_messages_days_l291_291691


namespace total_cost_pencils_l291_291073

theorem total_cost_pencils
  (boxes : ℕ)
  (cost_per_box : ℕ → ℕ → ℕ)
  (price_regular : ℕ)
  (price_bulk : ℕ)
  (box_size : ℕ)
  (bulk_threshold : ℕ)
  (total_pencils : ℕ) :
  total_pencils = 3150 →
  box_size = 150 →
  price_regular = 40 →
  price_bulk = 35 →
  bulk_threshold = 2000 →
  boxes = (total_pencils + box_size - 1) / box_size →
  (total_pencils > bulk_threshold → cost_per_box boxes price_bulk = boxes * price_bulk) →
  (total_pencils ≤ bulk_threshold → cost_per_box boxes price_regular = boxes * price_regular) →
  total_pencils > bulk_threshold →
  cost_per_box boxes price_bulk = 735 :=
by
  intro h_total_pencils
  intro h_box_size
  intro h_price_regular
  intro h_price_bulk
  intro h_bulk_threshold
  intro h_boxes
  intro h_cost_bulk
  intro h_cost_regular
  intro h_bulk_discount_passt
  -- sorry statement as we don't provide the actual proof here
  sorry

end total_cost_pencils_l291_291073


namespace sell_price_equal_percentage_l291_291759

theorem sell_price_equal_percentage (SP : ℝ) (CP : ℝ) :
  (SP - CP) / CP * 100 = (CP - 1280) / CP * 100 → 
  (1937.5 = CP + 0.25 * CP) → 
  SP = 1820 :=
by 
  -- Note: skip proof with sorry
  apply sorry

end sell_price_equal_percentage_l291_291759


namespace quadratic_properties_l291_291969

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4*x + 3

theorem quadratic_properties :
  (∃ h k, h = 2 ∧ k = 7 ∧ (∀ x, f x = -x^2 + 4*x + 3) ∧ (f 1 = 6) ∧ (f 4 = 7)) ∧
  (∀ x, f x = -(x - 2)^2 + 7) ∧
  (∀ x ∈ set.Icc 1 4, f x ≤ 7 ∧ f x = 6 → x = 1) ∧
  (∀ x ∈ set.Icc 1 4, f x ≥ 6 ∧ f x = 7 → x = 4) :=
by
  sorry

end quadratic_properties_l291_291969


namespace find_initial_red_marbles_l291_291606

theorem find_initial_red_marbles (x y : ℚ) 
  (h1 : 2 * x = 3 * y) 
  (h2 : 5 * (x - 15) = 2 * (y + 25)) 
  : x = 375 / 11 := 
by
  sorry

end find_initial_red_marbles_l291_291606


namespace constant_term_in_expansion_l291_291196

theorem constant_term_in_expansion (n : ℕ) 
  (h_sum : (2^n = 64)) : 
  (∑ k in Finset.range (n+1), (Nat.choose n k) * ((-2)^k) * (x^(n - 3/2*k)) | x := 1) = 240 := by
sorry

end constant_term_in_expansion_l291_291196


namespace part1_part2_l291_291977

def point := ℝ × ℝ × ℝ
def vector := ℝ × ℝ × ℝ

def dot_product (v₁ v₂ : vector) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

def magnitude (v : vector) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def perpendicular (v₁ v₂ : vector) : Prop :=
  dot_product v₁ v₂ = 0

def distance_from_point_to_line (p₀ p₁ p₂ : point) : ℝ :=
  let u := (p₁.1 - p₀.1, p₁.2 - p₀.2, p₁.3 - p₀.3)
  let v := (p₂.1 - p₀.1, p₂.2 - p₀.2, p₂.3 - p₀.3)
  let u_magnitude := magnitude u
  let u_unit := (u.1 / u_magnitude, u.2 / u_magnitude, u.3 / u_magnitude)
  let proj := dot_product v u_unit
  let distance_squared := dot_product v v - proj * proj
  real.sqrt distance_squared

def P : point := (-2, 0, 2)
def M : point := (-1, 1, 2)
def N : point := (-3, 0, 4)
def a : vector := (1, 1, 0)
def b : vector := (-1, 0, 2)

theorem part1 :
  (k : ℝ) (perpendicular ((k * a.1 + b.1, k * a.2 + b.2, k * a.3 + b.3))
                        (k * a.1 - 2 * b.1, k * a.2 - 2 * b.2, k * a.3 - 2 * b.3)) ↔ k = 2 ∨ k = -5 / 2 :=
sorry

theorem part2 :
  distance_from_point_to_line P M N = 3 * real.sqrt 2 / 2 :=
sorry

end part1_part2_l291_291977


namespace percentage_increase_14point4_from_12_l291_291242

theorem percentage_increase_14point4_from_12 (x : ℝ) (h : x = 14.4) : 
  ((x - 12) / 12) * 100 = 20 := 
by
  sorry

end percentage_increase_14point4_from_12_l291_291242


namespace value_of_first_equation_l291_291586

theorem value_of_first_equation (x y a : ℝ) 
  (h₁ : 2 * x + y = a) 
  (h₂ : x + 2 * y = 10) 
  (h₃ : (x + y) / 3 = 4) : 
  a = 12 :=
by 
  sorry

end value_of_first_equation_l291_291586


namespace Jane_stick_length_l291_291717

theorem Jane_stick_length
  (Pat_stick_length : ℕ)
  (dirt_covered_length : ℕ)
  (Sarah_stick_double : ℕ)
  (Jane_stick_diff : ℕ) :
  Pat_stick_length = 30 →
  dirt_covered_length = 7 →
  Sarah_stick_double = 2 →
  Jane_stick_diff = 24 →
  (Pat_stick_length - dirt_covered_length) * Sarah_stick_double - Jane_stick_diff = 22 := 
by
  intros Pat_length dirt_length Sarah_double Jane_diff
  intro h1
  intro h2
  intro h3
  intro h4
  rw [h1, h2, h3, h4]
  sorry

end Jane_stick_length_l291_291717


namespace hyperbola_eccentricity_l291_291215

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (asymptote_eq : ∀ x, x ≠ 0 → (2 * x = a / b * x)) : 
  (eccentricity : ℝ) :=
begin
  -- Definitions by conditions
  let c := sqrt (a^2 + b^2),
  let e := c / a,
  -- Assert the provided condition
  have hab : a / b = 2,
  { intros x hx,
    rw asymptote_eq x hx,
    symmetry,
    exact div_eq_2 (ne_of_gt ha) (ne_of_gt hb) }, -- Ensuring the validity of equality

  -- Verify the eccentricity
  have eccentricity_correct : e = sqrt 5 / 2,
  { rw [e, c, hab],
    field_simp [sqrt, pow_two, sqrt_five] },
  exact eccentricity_correct,
end

end hyperbola_eccentricity_l291_291215


namespace cos_alpha_implies_sin_alpha_tan_theta_implies_expr_l291_291424

-- Problem Part 1
theorem cos_alpha_implies_sin_alpha (alpha : ℝ) (h1 : Real.cos alpha = -4/5) (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.sin alpha = -3/5 := sorry

-- Problem Part 2
theorem tan_theta_implies_expr (theta : ℝ) (h1 : Real.tan theta = 3) : 
  (Real.sin theta + Real.cos theta) / (2 * Real.sin theta + Real.cos theta) = 4 / 7 := sorry

end cos_alpha_implies_sin_alpha_tan_theta_implies_expr_l291_291424


namespace west_movement_80_eq_neg_80_l291_291097

-- Define conditions
def east_movement (distance : ℤ) : ℤ := distance

-- Prove that moving westward is represented correctly
theorem west_movement_80_eq_neg_80 : east_movement (-80) = -80 :=
by
  -- Theorem proof goes here
  sorry

end west_movement_80_eq_neg_80_l291_291097


namespace smallest_mersenne_prime_gt_30_l291_291427

-- Definition of a Mersenne prime
def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, n > 1 ∧ nat.prime n ∧ p = 2^n - 1 ∧ nat.prime p

-- Main theorem statement
theorem smallest_mersenne_prime_gt_30 : ∃ p : ℕ, is_mersenne_prime p ∧ p > 30 ∧ (∀ q : ℕ, is_mersenne_prime q → q > 30 → p ≤ q) :=
sorry

end smallest_mersenne_prime_gt_30_l291_291427


namespace percent_increase_qrs_company_l291_291042

theorem percent_increase_qrs_company :
  let P : ℝ := 1 in
  let April_Profit : ℝ := P * 1.35 in
  let May_Profit : ℝ := April_Profit * 0.80 in
  let June_Profit : ℝ := May_Profit * 1.50 in
  let Overall_Increase : ℝ := ((June_Profit - P) / P) * 100 in
  Overall_Increase = 62 :=
by
  sorry

end percent_increase_qrs_company_l291_291042


namespace gcd_60_75_l291_291388

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l291_291388


namespace max_points_with_unique_3_order_color_sequences_l291_291890

theorem max_points_with_unique_3_order_color_sequences (n : ℕ) (colors : Fin n → Bool) 
  (h : ∀ i j : Fin n, i ≠ j →  
    (colors i ≠ colors (i + 1) ∨ colors (i + 2) ≠ colors (j + 1) ∨ colors (i + 2) ≠ colors (j + 2))) : 
  n ≤ 8 :=
sorry

end max_points_with_unique_3_order_color_sequences_l291_291890


namespace part1_part2_l291_291925

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2 * a * Real.log (1 - x)

theorem part1 (a : ℝ) (h : a = -2) :
  (∀ x, -1 < x ∧ x < 1 → f a x < f a x + 1) ∧
  (∀ x, x < -1 → f a x > f a x - 1) ∧
  (f (-2) (-1) = 1 - 4 * Real.log 2) := by
  sorry

theorem part2 (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x < 1 → f' a x ≥ 0) ↔ a ≤ -2 ∨ a ≥ 1/4 := by
  sorry

end part1_part2_l291_291925


namespace james_calories_ratio_l291_291657

theorem james_calories_ratio:
  ∀ (dancing_sessions_per_day : ℕ) (hours_per_session : ℕ) 
  (days_per_week : ℕ) (calories_per_hour_walking : ℕ) 
  (total_calories_dancing_per_week : ℕ),
  dancing_sessions_per_day = 2 →
  hours_per_session = 1/2 →
  days_per_week = 4 →
  calories_per_hour_walking = 300 →
  total_calories_dancing_per_week = 2400 →
  300 * 2 = 600 →
  (total_calories_dancing_per_week / (dancing_sessions_per_day * hours_per_session * days_per_week)) / calories_per_hour_walking = 2 :=
by
  sorry

end james_calories_ratio_l291_291657


namespace solution_set_of_inequality_l291_291765

theorem solution_set_of_inequality (x : ℝ) :
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end solution_set_of_inequality_l291_291765


namespace paths_D_to_A_paths_E_to_A_paths_K_to_A_l291_291421

def unusual_paths (start end : String) : Nat :=
  if start = "A" then 1 else if start = "B" then 1 else if start = "C" then 2 else
    if start = "D" then 3 else -- There are 3 unusual paths from D to A
    if start = "E" then unusual_paths "D" "A" + unusual_paths "C" "A" -- Paths from E are the sum of paths from D and C
    else if start = "K" then 
      unusual_paths "J" "A" + unusual_paths "I" "A"
    else 0 -- Base cases for other points
  
-- Given the custom recursive nature described in parts (a) and (b) for the number of unusual paths.

theorem paths_D_to_A : unusual_paths "D" "A" = 3 := by sorry
theorem paths_E_to_A : unusual_paths "E" "A" = unusual_paths "D" "A" + unusual_paths "C" "A" := by sorry
theorem paths_K_to_A : unusual_paths "K" "A" = 89 := by sorry

end paths_D_to_A_paths_E_to_A_paths_K_to_A_l291_291421


namespace division_of_8_identical_books_into_3_piles_l291_291819

-- Definitions for the conditions
def identical_books_division_ways (n : ℕ) (p : ℕ) : ℕ :=
  if n = 8 ∧ p = 3 then 5 else sorry

-- Theorem statement
theorem division_of_8_identical_books_into_3_piles :
  identical_books_division_ways 8 3 = 5 := by
  sorry

end division_of_8_identical_books_into_3_piles_l291_291819


namespace tan_sum_l291_291940

-- Define the conditions as local variables
variables {α β : ℝ} (h₁ : Real.tan α = -2) (h₂ : Real.tan β = 5)

-- The statement to prove
theorem tan_sum : Real.tan (α + β) = 3 / 11 :=
by 
  -- Proof goes here, using 'sorry' as placeholder
  sorry

end tan_sum_l291_291940


namespace water_percentage_proof_l291_291713

def percentageWaterInMixture (percentage1 percentage2 : ℝ) (parts1 parts2 : ℕ) : ℝ :=
  let totalWater := parts1 * percentage1 + parts2 * percentage2
  let totalMixture := parts1 + parts2
  (totalWater / totalMixture) * 100

theorem water_percentage_proof :
  percentageWaterInMixture 0.10 0.15 5 2 = 11.43 := by
  sorry

end water_percentage_proof_l291_291713


namespace customOp_sub_eq_l291_291615

-- Define the custom operation
def customOp (x y : ℕ) : ℕ := x * y - 3 * x

-- The main theorem to prove
theorem customOp_sub_eq : (customOp 7 4) - (customOp 4 7) = -9 := by
  sorry

end customOp_sub_eq_l291_291615


namespace a_b_powers_of_two_l291_291277

def largest_odd_divisor (x : ℕ) : ℕ := (1 to x).filter (λ d, d.odd ∧ x % d = 0).max

theorem a_b_powers_of_two (a b : ℕ) (h1 : Nat.gcd a b = 1) (h2 : ∃ c, a + largest_odd_divisor (b + 1) = 2^c) (h3 : ∃ d, b + largest_odd_divisor (a + 1) = 2^d) : 
  ∃ x, a + 1 = 2^x ∧ ∃ y, b + 1 = 2^y := 
sorry

end a_b_powers_of_two_l291_291277


namespace problem1_problem2_l291_291164

-- Define the conditions
def conditions (α : ℝ) : Prop :=
  tan α = -3 / 2 ∧ ( ∃ k : ℤ, α = (k + 1 / 2) * π + arccos(-3 / 2) )

-- First statement to verify the first question and its answer
theorem problem1 (α : ℝ) (h : conditions α) :
  (sin (-α - π / 2) * cos (3 * π / 2 + α) * tan (π - α)) / 
  (tan (-α - π) * sin (-π - α)) = 2 / sqrt 13 := 
sorry

-- Second statement to verify the second question and its answer
theorem problem2 (α : ℝ) (h : conditions α) :
  (1 / (cos α * sqrt (1 + tan α ^ 2)) +
  sqrt ((1 + sin α) / (1 - sin α)) - 
  sqrt ((1 - sin α) / (1 + sin α))) = 2 :=
sorry

end problem1_problem2_l291_291164


namespace total_marbles_l291_291605

theorem total_marbles (r b g : ℕ) (h_ratio : r = 1 ∧ b = 5 ∧ g = 3) (h_green : g = 27) :
  (r + b + g) * 3 = 81 :=
  sorry

end total_marbles_l291_291605


namespace max_min_sum_l291_291721

theorem max_min_sum 
  (n : ℕ)
  (a : Fin n → ℝ)
  (b : Fin n → ℝ)
  (ha : ∀ i j : Fin n, i ≤ j → a i ≥ a j)
  (hb : ∀ i j : Fin n, i ≤ j → b i ≥ b j) :
  (∃ k : Fin n → Fin n, BiFunction.injective k ∧
    (∀ k1 k2, Σ (a k1 * b k1) t < Σ (a k2 * b k2)) ∨ Σ (a k * b k) = a 1 * b 1 + a 2 * b 2 + ... + a n * b n) :=
sorry

end max_min_sum_l291_291721


namespace find_general_term_and_sum_l291_291177

variable (a : ℕ → ℕ)
variable (d : ℕ → ℕ)

def arithmetic_seq (a : ℕ → ℕ) (d : ℕ → ℕ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n + d n

def forms_geom_seq (a : ℕ → ℕ) (k1 k2 k3 : ℕ) : Prop :=
  (a k2)^2 = (a k1) * (a k3)

theorem find_general_term_and_sum (a : ℕ → ℕ) (d : ℕ → ℕ) (k1 k2 k3 : ℕ) 
  (h_arith : arithmetic_seq a d) 
  (k1_def : k1 = 1) (k2_def : k2 = 5) (k3_def : k3 = 17)
  (h_geom : forms_geom_seq a k1 k2 k3) :
  (∀ n : ℕ, k n = 2 * 3^(n-1) - 1) ∧
  (∀ n : ℕ, T n = 3^n - n - 1) :=
sorry

end find_general_term_and_sum_l291_291177


namespace ratio_65_13_l291_291009

theorem ratio_65_13 : 65 / 13 = 5 := 
by
  sorry

end ratio_65_13_l291_291009


namespace decreasing_intervals_of_f_l291_291342

noncomputable def f (x : ℝ) : ℝ := -x^2 + abs x

theorem decreasing_intervals_of_f :
  (∀ x y : ℝ, x ∈ Icc (-1/2) 0 → y ∈ Icc (-1/2) 0 → x < y → f x > f y) ∧
  (∀ x y : ℝ, x ∈ Ici (1/2) → y ∈ Ici (1/2) → x < y → f x > f y) :=
by
  sorry

end decreasing_intervals_of_f_l291_291342


namespace find_sum_of_variables_l291_291336

variables (a b c d : ℤ)

theorem find_sum_of_variables
    (h1 : a - b + c = 7)
    (h2 : b - c + d = 8)
    (h3 : c - d + a = 4)
    (h4 : d - a + b = 3)
    (h5 : a + b + c - d = 10) :
    a + b + c + d = 16 := 
sorry

end find_sum_of_variables_l291_291336


namespace total_wings_count_l291_291271

theorem total_wings_count (num_planes : ℕ) (wings_per_plane : ℕ) (h_planes : num_planes = 54) (h_wings : wings_per_plane = 2) : num_planes * wings_per_plane = 108 :=
by 
  sorry

end total_wings_count_l291_291271


namespace shortest_multicolored_cycle_l291_291703

-- Define vertices and edges
variables {V: Type*} [fintype V] -- Vertex set
variables {E: Type*} [fintype E] -- Edge set
variables (cycle : list E) (color : E → ℕ) -- Cycle and color function

-- Define the conditions
def is_vertex_horizontal (v : V) : Prop := sorry -- Predicate for horizontal vertices
def is_vertex_vertical (v : V) : Prop := sorry -- Predicate for vertical vertices
def edges_of_cycle (cycle : list E) : list (V × V) := sorry -- Extract edges from the cycle
def are_edges_multicolored (edges : list (V × V)) : Prop := sorry -- Check if edges are multicolored

-- Define the length of the cycle
def cycle_length : ℕ := cycle.length

-- Prove the shortest multicolored cycle has 4 edges
theorem shortest_multicolored_cycle (h_cycle: ∀ (a_i b_i : V) (h1: is_vertex_horizontal a_i) (h2: is_vertex_vertical b_i),
  ∃ s, edges_of_cycle cycle = list.zip (list.repeat a_i s) (list.repeat b_i s) ∧ cycle_length = 2 * s)
    (h_s: ∃ s, s > 2)
  : ∃ s, s = 2 :=
by
  sorry

end shortest_multicolored_cycle_l291_291703


namespace gcd_60_75_l291_291387

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l291_291387


namespace median_of_data_set_l291_291450

-- Definitions of the conditions.
def data_set : list ℕ := [6, 5, 7, 6, 6]

-- The statement to be proven: the median of the data set is 6.
theorem median_of_data_set : median data_set = 6 :=
by sorry

end median_of_data_set_l291_291450


namespace angle_between_East_and_Southwest_l291_291435

-- Define the context and the problem
theorem angle_between_East_and_Southwest :
  let rays := (range 8).map (λ i, i * 45) in
  rays[2] - rays[7] = 135 :=
by
  let rays := (range 8).map (λ i, i * 360 / 8)
  have rays_def : rays = [0, 45, 90, 135, 180, 225, 270, 315],
  sorry,
  -- Prove the specific angle calculation
  calc rays[7] - rays[2]
       = 135 : by sorry
  -- Continue the proof specific to calculations

end angle_between_East_and_Southwest_l291_291435


namespace largest_difference_l291_291483

noncomputable def Chicago_estimate : ℕ := 40000
noncomputable def Brooklyn_estimate : ℕ := 45000

noncomputable def Chicago_actual : ℕ := Chicago_estimate * 105 / 100
noncomputable def Brooklyn_actual : ℕ := Brooklyn_estimate * 95 / 100

theorem largest_difference : |Brooklyn_actual - Chicago_actual| = 750 → round (|Brooklyn_actual - Chicago_actual| / 1000) * 1000 = 1000 := 
by
  sorry

end largest_difference_l291_291483


namespace range_of_m_l291_291207

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 10 then (1 / 10) ^ x else -Real.log10 (x + 2)

theorem range_of_m :
  {m : ℝ | f (8 - m^2) < f (2 * m)} = {m : ℝ | -4 < m ∧ m < 2} :=
by
  sorry

end range_of_m_l291_291207


namespace family_members_l291_291773

theorem family_members (N : ℕ) (income : ℕ → ℕ) (average_income : ℕ) :
  average_income = 10000 ∧
  income 0 = 8000 ∧
  income 1 = 15000 ∧
  income 2 = 6000 ∧
  income 3 = 11000 ∧
  (income 0 + income 1 + income 2 + income 3) = 4 * average_income →
  N = 4 :=
by {
  sorry
}

end family_members_l291_291773


namespace planting_ways_l291_291002

-- Noncomputable theory as we are not evaluating the actual configurations
noncomputable theory

-- Definitions based on the conditions
def num_trees : ℕ := 6
def tree (n : ℕ) := if n % 2 = 0 then "Birch" else "Oak"

-- The main theorem statement
theorem planting_ways : ∃ (ways : ℕ), ways = 2 := by
  -- Planting trees such that each tree of one kind has a neighbor of the other kind
  -- Hence, there are exactly 2 valid configurations: "BOBOBO" and "OBOBOB"
  exact ⟨2, rfl⟩

end planting_ways_l291_291002


namespace minimum_additional_coins_l291_291861

theorem minimum_additional_coins
  (friends : ℕ) (initial_coins : ℕ)
  (h_friends : friends = 15) (h_coins : initial_coins = 100) :
  ∃ additional_coins : ℕ, additional_coins = 20 :=
by
  have total_needed_coins : ℕ := (friends * (friends + 1)) / 2
  have total_coins : ℕ := initial_coins
  have additional_coins_needed : ℕ := total_needed_coins - total_coins
  have h_additional_coins : additional_coins_needed = 20 := by calculate 
  -- Finishing the proof with the result we calculated
  use additional_coins_needed
  exact h_additional_coins

end minimum_additional_coins_l291_291861


namespace shortest_multicolored_cycle_l291_291705

-- Let's define the conditions and the main goal
theorem shortest_multicolored_cycle (s : ℕ) (a : fin s → vertex) (b : fin s → vertex) (H : multicolored_cycle (a, b)) :
  s = 2 := 
by {
  -- Proof omitted, add actual proof here
  sorry
}

end shortest_multicolored_cycle_l291_291705


namespace max_triangle_area_l291_291200

noncomputable theory

/--
Given the ellipse \(C\) with equation \(\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1\) 
with \(a > b > 0\), and right focus \(F(\sqrt{2}, 0)\), 
find that:
1. The equation of the ellipse simplifies to \(\frac{x^2}{4} + \frac{y^2}{2} = 1\).
2. For the line \(l : y = kx + m\) (where \(km \neq 0\)), intersecting 
the ellipse \(C\) at points \(A\) and \(B\), and the midpoint of \(AB\) lies on 
the line \(x + 2y = 0\), the maximum area of \(\triangle FAB\) is \(8/3\).
-/
theorem max_triangle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (exists a b : ℝ, a = 2 ∧ b = √2 ∧ (∀ x y : ℝ, x^2 / 4 + y^2 / 2 = 1)) ∧
  (∀ k m : ℝ, k ≠ 0 ∧ m ≠ 0 → ∃ max_area : ℝ, max_area = 8 / 3) :=
begin
  sorry
end

end max_triangle_area_l291_291200


namespace sum_first_2017_terms_l291_291155

def floor (x : ℝ) : ℤ := Int.floor x

def a_n (n : ℕ) : ℤ := floor (n / 3)

def S (n : ℕ) : ℤ := (Finset.range n).sum (λ i => a_n (i + 1))

theorem sum_first_2017_terms
  (floor_eq : ∀ (x : ℝ), floor x = Int.floor x)
  (h1: ∀ (n : ℕ), a_n (n + 1) = floor ((n + 1) / 3)) :
  S 2017 = 677712 := 
by
  sorry

end sum_first_2017_terms_l291_291155


namespace range_of_a_l291_291241

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 :=
by
  intro h
  sorry

end range_of_a_l291_291241


namespace sum_angles_satisfying_condition_l291_291903

theorem sum_angles_satisfying_condition :
  (∑ x in (Finset.filter (λ x, sin (Real.pi * x / 180) ^ 6 - cos (Real.pi * x / 180) ^ 6 = 1 / cos (Real.pi * x / 180) ^ 2 - 1 / sin (Real.pi * x / 180) ^ 2) (Finset.range 361)), x) = 720 :=
by
  sorry

end sum_angles_satisfying_condition_l291_291903


namespace not_possible_for_runners_in_front_l291_291446

noncomputable def runnerInFrontAtAnyMoment 
  (track_length : ℝ)
  (stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) : Prop := 
  ∀ t : ℝ, ∃ i : Fin 10, 
  ∃ n : ℤ, 
  (runners_speeds i * t - n * track_length) % track_length ≤ stands_length

theorem not_possible_for_runners_in_front 
  (track_length stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) 
  (h_track : track_length = 1)
  (h_stands : stands_length = 0.1)
  (h_speeds : ∀ i : Fin 10, 20 + i = runners_speeds i) : 
  ¬ runnerInFrontAtAnyMoment track_length stands_length runners_speeds :=
sorry

end not_possible_for_runners_in_front_l291_291446


namespace number_of_valid_b_l291_291542

theorem number_of_valid_b :
  let S := { b : ℕ | (∀ x : ℕ, 2 < x → x = 3 ∨ (3 * x - b > -9)) } in
  ∃ (b_collection : Finset ℕ), 
  b_collection = {15, 16, 17} ∧
  ∀ b ∈ b_collection, b ∈ S ∧
  ∀ b ∉ b_collection, b ∉ S :=
by sorry

end number_of_valid_b_l291_291542


namespace count_integers_divis_by_8_l291_291909

theorem count_integers_divis_by_8 : 
  ∃ k : ℕ, k = 49 ∧ ∀ n : ℕ, 2 ≤ n ∧ n ≤ 80 → (∃ m : ℤ, (n-1) * n * (n+1) = 8 * m) ↔ (∃ m : ℕ, m ≤ k) :=
by 
  sorry

end count_integers_divis_by_8_l291_291909


namespace quadratic_root_k_value_l291_291565

theorem quadratic_root_k_value 
  (k : ℝ) 
  (h_roots : ∀ x : ℝ, (5 * x^2 + 7 * x + k = 0) → (x = ( -7 + Real.sqrt (-191) ) / 10 ∨ x = ( -7 - Real.sqrt (-191) ) / 10)) : 
  k = 12 :=
sorry

end quadratic_root_k_value_l291_291565


namespace min_room_dimensions_l291_291048

theorem min_room_dimensions 
    (S1 S2 : ℕ) 
    (condition : ∀ S: ℕ, S ≥ nat.ceil (real.sqrt (10^2 + 12^2))) :
    S1 = 16 ∧ S2 = 16 :=
by 
  sorry

end min_room_dimensions_l291_291048


namespace mode_correct_median_correct_average_correct_estimated_total_donation_correct_l291_291259

def amounts : List ℕ := [5, 10, 15, 20]
def number_of_students : List ℕ := [1, 5, 3, 1]
def total_students : ℕ := 2200

-- Define mode function
def mode (amounts : List ℕ) (counts : List ℕ) : ℕ := 
amounts[counts.indexOf(counts.foldr max 0)]

-- Define median function
def median (amounts : List ℕ) (counts : List ℕ) : ℕ :=
  let sorted_data := List.range(counts.length).bind (λ i => List.replicate counts[i] amounts[i])
  let n := sorted_data.length
  if n % 2 = 0 then
    (sorted_data[(n/2) - 1] + sorted_data[n/2]) / 2
  else
    sorted_data[n / 2]

-- Define average function
def average (amounts : List ℕ) (counts : List ℕ) : ℕ :=
  (List.zipWith (λ a n => a * n) amounts counts).sum / counts.sum

-- Define estimated total donation
def estimated_total_donation (average_donation : ℕ) (total_students : ℕ) : ℕ :=
  average_donation * total_students

-- Theorems to prove
theorem mode_correct : mode amounts number_of_students = 10 := 
  sorry

theorem median_correct : median amounts number_of_students = 10 := 
  sorry

theorem average_correct : average amounts number_of_students = 12 := 
  sorry

theorem estimated_total_donation_correct : estimated_total_donation (average amounts number_of_students) total_students = 26400 :=
  sorry

end mode_correct_median_correct_average_correct_estimated_total_donation_correct_l291_291259


namespace sequence_b_3_pow_100_l291_291675

def sequence_b (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if ∃ k, n = 3 * k then ((n / 3) ^ 2) * (sequence_b (n / 3))
  else 0

theorem sequence_b_3_pow_100 : sequence_b (3^100) = 9^99 := by
  sorry

end sequence_b_3_pow_100_l291_291675


namespace aaron_pages_sixth_day_l291_291466

theorem aaron_pages_sixth_day 
  (h1 : 18 + 12 + 23 + 10 + 17 + y = 6 * 15) : 
  y = 10 :=
by
  sorry

end aaron_pages_sixth_day_l291_291466


namespace increasing_iff_positive_difference_l291_291674

variable (a : ℕ → ℝ) (d : ℝ)

def arithmetic_sequence (aₙ : ℕ → ℝ) (d : ℝ) := ∃ (a₁ : ℝ), ∀ n : ℕ, aₙ n = a₁ + n * d

theorem increasing_iff_positive_difference (a : ℕ → ℝ) (d : ℝ) (h : arithmetic_sequence a d) :
  (∀ n, a (n+1) > a n) ↔ d > 0 :=
by
  sorry

end increasing_iff_positive_difference_l291_291674


namespace total_hoodies_l291_291539

def Fiona_hoodies : ℕ := 3
def Casey_hoodies : ℕ := Fiona_hoodies + 2

theorem total_hoodies : (Fiona_hoodies + Casey_hoodies) = 8 := by
  sorry

end total_hoodies_l291_291539


namespace orthic_triangle_perimeter_leq_half_l291_291254

theorem orthic_triangle_perimeter_leq_half (ABC : Triangle) (h_acute : IsAcute ABC)
  (A1 B1 C1 : Point)
  (h_altitude_A : IsAltitude (ABC.a) (ABC.b) (ABC.c) A1)
  (h_altitude_B : IsAltitude (ABC.b) (ABC.a) (ABC.c) B1)
  (h_altitude_C : IsAltitude (ABC.c) (ABC.a) (ABC.b) C1) :
  Perimeter (Triangle.mk A1 B1 C1) ≤ (Perimeter ABC) / 2 := 
sorry

end orthic_triangle_perimeter_leq_half_l291_291254


namespace find_angle_B_prove_third_condition1_prove_third_condition2_prove_third_condition3_l291_291580

variables {A B C : ℝ} {a b c : ℝ}
variables {S : ℝ}

-- Conditions
def triangle_sides := (a = side opposite to angle A) ∧ (b = side opposite to angle B) ∧ (c = side opposite to angle C)
def given_condition := b * sin A = a * cos B * sqrt 3 
def area_condition := S = (1/2) * a * c * sin B
def side_condition := a + c = 6

-- Questions rephrased as Lean goals
theorem find_angle_B (h : given_condition) : B = π / 3 := 
by sorry

theorem prove_third_condition1 (h1 : b = 3) (h2 : S = 9 * sqrt 3 / 4) : a + c = 6 :=
by sorry 

theorem prove_third_condition2 (h1 : S = 9 * sqrt 3 / 4) (h2 : a + c = 6) : b = 3 :=
by sorry 

theorem prove_third_condition3 (h1 : b = 3) (h2 : a + c = 6) : S = 9 * sqrt 3 / 4 :=
by sorry 

end find_angle_B_prove_third_condition1_prove_third_condition2_prove_third_condition3_l291_291580


namespace investment_plans_count_l291_291443

theorem investment_plans_count : 
  let num_projects := 3 in
  let num_locations := 6 in
  let scenario1 := @Finset.card (Finset {x : Fin _ // x.val < num_locations}) (Finset.univ.filter (λ s, s.card = num_projects)) * 
                   (Finset.card (Finset.perm (Finset.card (Finset.range num_projects) = num_projects)).attach.set) in
  let scenario2 := @Finset.card (Finset {x : Fin _ // x.val < num_locations}) (Finset.univ.filter (λ s, s.card = 2)) * 
                   (Finset.card (Finset.range num_projects).filter (λ s, s.card = 2)) in
  scenario1 + scenario2 = 210 :=
by
  sorry

end investment_plans_count_l291_291443


namespace tan_sub_eq_one_eight_tan_add_eq_neg_four_seven_l291_291616

variable (α β : ℝ)

theorem tan_sub_eq_one_eight (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
sorry

theorem tan_add_eq_neg_four_seven (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -4 / 7 := 
sorry

end tan_sub_eq_one_eight_tan_add_eq_neg_four_seven_l291_291616


namespace volume_of_box_with_ratio_125_l291_291844

def volumes : Finset ℕ := {60, 80, 100, 120, 200}

theorem volume_of_box_with_ratio_125 : 80 ∈ volumes ∧ ∃ (x : ℕ), 10 * x^3 = 80 :=
by {
  -- Skipping the proof, as only the statement is required.
  sorry
}

end volume_of_box_with_ratio_125_l291_291844


namespace correct_option_given_condition_l291_291296

def f (x : ℝ) (b : ℝ) := x^3 - 12*x + b
def derivative_f (x : ℝ) := 3*x^2 - 12

theorem correct_option_given_condition (b : ℝ) :
  b = -6 → tangent_at f (-2) (-2, f -2 (-6)) = 10 :=
begin
  sorry
end

end correct_option_given_condition_l291_291296


namespace largest_natural_number_lt_5300_l291_291530

theorem largest_natural_number_lt_5300 : ∃ n : ℕ, n^200 < 5^300 ∧ ∀ m : ℕ, m^200 < 5^300 → m ≤ n :=
by
  use 11
  split
  sorry

end largest_natural_number_lt_5300_l291_291530


namespace option_d_not_equivalent_l291_291802

def is_equivalent (a b : ℝ) : Prop :=
  a = b

def not_equivalent_to_0_000000375 (x : ℝ) : Prop :=
  ¬ is_equivalent x 3.75e-7

theorem option_d_not_equivalent : not_equivalent_to_0_000000375 (3 / 8 * 10 ^ (-7)) :=
by
  simp [is_equivalent, not_equivalent_to_0_000000375]
  sorry

end option_d_not_equivalent_l291_291802


namespace exists_infinitely_many_pairs_exists_single_statement_only_l291_291180

-- Define the sequence
def sequence (α β a b c : ℤ) : ℤ → ℤ
| 1 => α
| 2 => β
| (n + 2) => a * (sequence (α) (β) (a) (b) (c) (n + 1)) + b * (sequence (α) (β) (a) (b) (c) n) + c

-- First part: there are infinitely many pairs (α, β) such that u_2023 = 2^2022
theorem exists_infinitely_many_pairs (α β : ℤ) :
  ∀ k : ℤ, ∃ (α β : ℤ), sequence α β 3 -2 -1 2023 = 2^(2022) := 
  sorry

-- Second part: proves that there exists an n_0 such that only one of the given conditions is true
theorem exists_single_statement_only (α β : ℤ) :
  ∃ (n_0 : ℕ), (∀ m : ℕ, (sequence α β 3 -2 -1 (n_0 + m + 1) = 7^(2023) ∨ sequence α β 3 -2 -1 (n_0 + m + 1) = 17^(2023)) 
  ∨ (∀ k : ℕ, sequence α β 3 -2 -1 (n_0 + k + 1) - 1 = 2023)) := 
  sorry

end exists_infinitely_many_pairs_exists_single_statement_only_l291_291180


namespace cost_of_horse_l291_291035

theorem cost_of_horse (H C : ℝ) 
  (h1 : 4 * H + 9 * C = 13400)
  (h2 : 0.4 * H + 1.8 * C = 1880) :
  H = 2000 :=
by
  sorry

end cost_of_horse_l291_291035


namespace painting_problem_l291_291867

theorem painting_problem (initial_painters : ℕ) (initial_days : ℚ) (initial_rate : ℚ) (new_days : ℚ) (new_rate : ℚ) : 
  initial_painters = 6 ∧ initial_days = 5/2 ∧ initial_rate = 2 ∧ new_days = 2 ∧ new_rate = 2.5 →
  ∃ additional_painters : ℕ, additional_painters = 0 :=
by
  intros h
  sorry

end painting_problem_l291_291867


namespace range_of_a_l291_291204

variable {a : ℝ}

theorem range_of_a (h : ∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (5 - a)) : -3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l291_291204


namespace range_of_a_l291_291627

theorem range_of_a (a : ℝ) :
  (∃! x : ℕ, x^2 - (a + 2) * x + 2 - a < 0) ↔ (1/2 < a ∧ a ≤ 2/3) := 
sorry

end range_of_a_l291_291627


namespace cost_per_tissue_box_l291_291476

-- Given conditions
def rolls_toilet_paper : ℝ := 10
def cost_per_toilet_paper : ℝ := 1.5
def rolls_paper_towels : ℝ := 7
def cost_per_paper_towel : ℝ := 2
def boxes_tissues : ℝ := 3
def total_cost : ℝ := 35

-- Deduction of individual costs
def cost_toilet_paper := rolls_toilet_paper * cost_per_toilet_paper
def cost_paper_towels := rolls_paper_towels * cost_per_paper_towel
def cost_tissues := total_cost - cost_toilet_paper - cost_paper_towels

-- Prove the cost for one box of tissues
theorem cost_per_tissue_box : (cost_tissues / boxes_tissues) = 2 :=
by
  sorry

end cost_per_tissue_box_l291_291476


namespace gcf_60_75_l291_291401

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l291_291401


namespace Alyssa_cookie_count_l291_291083

/--
  Alyssa had some cookies.
  Aiyanna has 140 cookies.
  Aiyanna has 11 more cookies than Alyssa.
  How many cookies does Alyssa have? 
-/
theorem Alyssa_cookie_count 
  (aiyanna_cookies : ℕ) 
  (more_cookies : ℕ)
  (h1 : aiyanna_cookies = 140)
  (h2 : more_cookies = 11)
  (h3 : aiyanna_cookies = alyssa_cookies + more_cookies) :
  alyssa_cookies = 129 := 
sorry

end Alyssa_cookie_count_l291_291083


namespace num_parallel_edge_pairs_correct_l291_291602

-- Define a rectangular prism with given dimensions
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

-- Function to count the number of pairs of parallel edges
def num_parallel_edge_pairs (p : RectangularPrism) : ℕ :=
  4 * ((p.length + p.width + p.height) - 3)

-- Given conditions
def given_prism : RectangularPrism := { length := 4, width := 3, height := 2 }

-- Main theorem statement
theorem num_parallel_edge_pairs_correct :
  num_parallel_edge_pairs given_prism = 12 :=
by
  -- Skipping proof steps
  sorry

end num_parallel_edge_pairs_correct_l291_291602


namespace problem1_problem2_l291_291915

variable (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)

theorem problem1 : 
  (a * b + a + b + 1) * (a * b + a * c + b * c + c ^ 2) ≥ 16 * a * b * c := 
by sorry

theorem problem2 : 
  (b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3 := 
by sorry

end problem1_problem2_l291_291915


namespace tom_batteries_used_total_l291_291375

def batteries_used_in_flashlights : Nat := 2 * 3
def batteries_used_in_toys : Nat := 4 * 5
def batteries_used_in_controllers : Nat := 2 * 6
def total_batteries_used : Nat := batteries_used_in_flashlights + batteries_used_in_toys + batteries_used_in_controllers

theorem tom_batteries_used_total : total_batteries_used = 38 :=
by
  sorry

end tom_batteries_used_total_l291_291375


namespace range_and_mode_l291_291927

def dataSet : Finset ℕ := {15, 13, 15, 16, 17, 16, 14, 15}

theorem range_and_mode :
  let dataList := [15, 13, 15, 16, 17, 16, 14, 15]
  let range := list.maximum dataList - list.minimum dataList
  let mode := (list.count_occur dataList).max_by (λ x, x.value).key
  range = 4 ∧ mode = 15 :=
sorry

end range_and_mode_l291_291927


namespace greatest_solution_of_equation_l291_291133

theorem greatest_solution_of_equation : ∀ x : ℝ, x ≠ 9 ∧ (x^2 - x - 90) / (x - 9) = 4 / (x + 6) → x ≤ -7 :=
by
  intros x hx
  sorry

end greatest_solution_of_equation_l291_291133


namespace number_of_seniors_in_statistics_l291_291696

theorem number_of_seniors_in_statistics (total_students : ℕ) (half_enrolled_in_statistics : ℕ) (percentage_seniors : ℚ) (students_in_statistics seniors_in_statistics : ℕ) 
(h1 : total_students = 120)
(h2 : half_enrolled_in_statistics = total_students / 2)
(h3 : students_in_statistics = half_enrolled_in_statistics)
(h4 : percentage_seniors = 0.90)
(h5 : seniors_in_statistics = students_in_statistics * percentage_seniors) : 
seniors_in_statistics = 54 := 
by sorry

end number_of_seniors_in_statistics_l291_291696


namespace probability_between_C_and_D_l291_291314

theorem probability_between_C_and_D :
  ∀ (A B C D : ℝ) (AB AD BC : ℝ),
    AB = 3 * AD ∧ AB = 6 * BC ∧ D - A = AD ∧ C - A = AD + BC ∧ B - A = AB →
    (C < D) →
    ∃ p : ℝ, p = 1 / 2 := by
  sorry

end probability_between_C_and_D_l291_291314


namespace distance_between_foci_l291_291544

theorem distance_between_foci (a b : ℝ) (h1 : a = 5) (h2 : b = 3) : 2 * real.sqrt (a^2 - b^2) = 8 :=
by
  rw [h1, h2]
  norm_num
  sorry

end distance_between_foci_l291_291544


namespace solution_correct_l291_291416

theorem solution_correct (a b c d : ℝ) (h : a * b * c * d = (sqrt ((a + 2) * (b + 3))) / (c + 1) * real.sin d) :
  (6:ℝ) * (15:ℝ) * (11:ℝ) * (30:ℝ) = 0.5 :=
by
  -- Assigning the values a, b, c, and d
  let a := (6:ℝ)
  let b := (15:ℝ)
  let c := (11:ℝ)
  let d := (30:ℝ)
  have h := h
  sorry

end solution_correct_l291_291416


namespace intersection_point_l291_291134

theorem intersection_point : 
  ∃ x y : ℚ, (5 * x - 3 * y = 19) ∧ (6 * x + 2 * y = 14) ∧ (x = 20 / 7) ∧ (y = -11 / 7) :=
by 
  use 20 / 7, -11 / 7
  constructor
  { -- Show 5 * (20/7) - 3 * (-11/7) = 19
    rw [mul_div_cancel' _ (by norm_num : (7:ℚ) ≠ 0), mul_div_cancel' _ (by norm_num : (7:ℚ) ≠ 0), 
        sub_neg_eq_add]
    norm_num },
  constructor
  { -- Show 6 * (20/7) + 2 * (-11/7) = 14
    rw [mul_div_cancel' _ (by norm_num : (7:ℚ) ≠ 0), mul_div_cancel' _ (by norm_num : (7:ℚ) ≠ 0)]
    norm_num }
  constructor
  { refl }
  { refl }

end intersection_point_l291_291134


namespace discount_savings_difference_l291_291752

theorem discount_savings_difference :
  let sticker_price := 30
  let discount1 := 5
  let discount2_percent := 0.25
  let cost1 := 0.75 * (sticker_price - discount1)
  let cost2 := (sticker_price * 0.75) - discount1 in
  (cost1 - cost2) * 100 = 125
:= by
  let sticker_price := 30
  let discount1 := 5
  let discount2_percent := 0.25
  let cost1 := 0.75 * (sticker_price - discount1)
  let cost2 := (sticker_price * 0.75) - discount1
  show (cost1 - cost2) * 100 = 125
  sorry

end discount_savings_difference_l291_291752


namespace percentage_more_than_third_num_l291_291789

theorem percentage_more_than_third_num (x : ℝ) (p : ℝ) (h1 : p = 0.15) :
  let second_number := x + 0.40 * x,
      first_number := p * second_number
  in abs (first_number - x) / x * 100 = 79 := 
by
  let second_number := x + 0.40 * x
  let first_number := p * second_number
  have h2 : first_number = 0.21 * x := by
    rw [h1]
    unfold second_number
    rw [mul_add]
    norm_num
  sorry

end percentage_more_than_third_num_l291_291789


namespace no_four_of_a_kind_pair_set_aside_re_rollematch_l291_291907

noncomputable def probability_at_least_four_same (d : Dice) : ℕ :=
1 / 216

theorem no_four_of_a_kind_pair_set_aside_re_rollematch :
  ∀ (d1 d2 d3 d4 d5 : ℕ), -- Five standard six-sided dice are rolled
  d1 ≠ d2 → d1 ≠ d3 → d1 ≠ d4 → d1 ≠ d5 → -- There is no four-of-a-kind
  d2 ≠ d3 → d2 ≠ d4 → d2 ≠ d5 →
  d3 ≠ d4 → d3 ≠ d5 →
  d4 ≠ d5 →
  (∃ (pair : ℕ) (rest : list ℕ),
    rest.length = 3 ∧ -- Two dice showing the same number are set aside, the other three are re-rolled
    list.map (λ r, dice_roll d r) rest = repeat pair 3) →
  probability_at_least_four_same d = (1 / 216) := -- The probability that at least four of the five dice show the same value
sorry

end no_four_of_a_kind_pair_set_aside_re_rollematch_l291_291907


namespace count_two_digit_numbers_with_8_l291_291993

theorem count_two_digit_numbers_with_8 : 
  (card {n : ℕ | 10 <= n ∧ n < 100 ∧ (n / 10 = 8 ∨ n % 10 = 8)}) = 17 := 
by 
  sorry

end count_two_digit_numbers_with_8_l291_291993


namespace minimum_additional_coins_l291_291854

-- The conditions
def total_friends : ℕ := 15
def current_coins : ℕ := 100

-- The fact that the total coins required to give each friend a unique number of coins from 1 to 15 is 120
def total_required_coins : ℕ := (total_friends * (total_friends + 1)) / 2

-- The theorem stating the required number of additional coins
theorem minimum_additional_coins (total_friends : ℕ) (current_coins : ℕ) (total_required_coins : ℕ) : ℕ :=
  sorry

end minimum_additional_coins_l291_291854


namespace triangle_max_area_l291_291244

noncomputable def triangle_properties (A B C a b c : ℝ) : Prop :=
  ∀ (A B C a b c : ℝ),
  ∀ (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : A + B + C = π) 
  (h5 : c = 2) 
  (h6 : (cos C / sin C) = (cos A + cos B) / (sin A + sin B)),
  ∃ (C_area : ℝ), 
    C = π / 3 ∧ 
    C_area = sqrt 3

theorem triangle_max_area (A B C a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : A + B + C = π)
  (h5 : c = 2) 
  (h6 : (cos C / sin C) = (cos A + cos B) / (sin A + sin B)) :
  C = π / 3 ∧ (∃ ab_area : ℝ, ab_area = sqrt 3) := 
  sorry

end triangle_max_area_l291_291244


namespace trapezoid_scalar_product_l291_291743

noncomputable def scalar_product_AD_BC (AB CD : ℝ) (a b : ℝ) 
  (h1 : AB = 101) (h2 : CD = 20) (h3 : a * b = 0) : ℝ :=
let AD := a + (20 / 101) * b
let BC := b + (20 / 101) * a in
AD * BC

theorem trapezoid_scalar_product : scalar_product_AD_BC 101 20 = 2020 := by
  sorry

end trapezoid_scalar_product_l291_291743


namespace length_RS_is_24_l291_291763

-- Definitions based on the given conditions
def edges : set ℕ := {12, 19, 24, 33, 42, 51}

def edge_PQ : ℕ := 51

-- This is the theorem we want to prove
theorem length_RS_is_24 (h : edge_PQ = 51) (h₁ : 12 ∈ edges) (h₂ : 19 ∈ edges) 
  (h₃ : 24 ∈ edges) (h₄ : 33 ∈ edges) (h₅ : 42 ∈ edges) (h₆ : 51 ∈ edges) : 
    (24 ∈ edges) :=
by sorry

end length_RS_is_24_l291_291763


namespace simplify_expression_l291_291095

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  ((2 * x^2)^3 - 6 * x^3 * (x^3 - 2 * x^2)) / (2 * x^4) = x^2 + 6 * x :=
by 
  -- We provide 'sorry' hack to skip the proof
  -- Replace this with the actual proof to ensure correctness.
  sorry

end simplify_expression_l291_291095


namespace find_element_in_compound_l291_291827

noncomputable def compound_formula : String := "X2O3"

def molecular_weight_compound : ℝ := 160

def oxygen_atomic_weight : ℝ := 16

def total_weight_oxygen (num_atoms : ℝ) : ℝ := num_atoms * oxygen_atomic_weight

def weight_two_atoms_X (molecular_weight : ℝ) (oxygen_weight : ℝ) : ℝ :=
  molecular_weight - oxygen_weight

def atomic_weight_X (weight_two_atoms : ℝ) (num_atoms : ℝ) : ℝ :=
  weight_two_atoms / num_atoms

theorem find_element_in_compound
  (compound_formula = "X2O3") 
  (molecular_weight = 160)
  (oxygen_atomic_weight = 16) :
  let num_oxygen_atoms := 3 in
  let num_X_atoms := 2 in
  let weight_oxygen := total_weight_oxygen num_oxygen_atoms in
  let weight_X := weight_two_atoms_X molecular_weight weight_oxygen in
  let atomic_weight_X := atomic_weight_X weight_X num_X_atoms in
  atomic_weight_X = 56 :=
sorry

end find_element_in_compound_l291_291827


namespace Ada_initial_seat_l291_291731

theorem Ada_initial_seat (seat : Fin 6) (Ada_initially : seat = 1 ∨ seat = 2) 
  (empty_seat_initially : Fin 6) (initially_empty : empty_seat_initially ≠ Ada_initially)
  (Bea_initially : Fin 6) (Ceci_initially : Fin 6)
  (Bea_moves_3_right : Bea_initially + 3 = if Bea_initially + 3 < 6 then Bea_initially + 3 else Bea_initially + 3 - 6)
  (Ceci_moves_2_right : Ceci_initially + 2 = if Ceci_initially + 2 < 6 then Ceci_initially + 2 else Ceci_initially + 2 - 6)
  (Dee_initially Edie_initially : Fin 6) (Dee_and_Edie_switch : Dee_initially = Edie_initially ∧ Edie_initially = Dee_initially)
  (Ada_initial_position_is_end_empty : Ada_initially = 1 ∨ Ada_initially = 6) :
  Ada_initially = 2 :=
sorry

end Ada_initial_seat_l291_291731


namespace remaining_black_cards_l291_291425

-- Definition of the problem conditions
def total_cards_in_deck := 52
def colors_in_deck := 2
def cards_per_color := total_cards_in_deck / colors_in_deck
def black_cards := cards_per_color
def black_cards_taken_out := 4

-- Desired proof statement
theorem remaining_black_cards :
  black_cards - black_cards_taken_out = 22 :=
by
  have h1 : cards_per_color = 26 := by
    unfold cards_per_color
    simp [total_cards_in_deck, colors_in_deck]
  have h2 : black_cards = 26 := by
    unfold black_cards
    exact h1
  show 26 - 4 = 22
  simp [black_cards_taken_out]
  exact rfl

end remaining_black_cards_l291_291425


namespace line_BC_eq_l291_291459

-- Definitions for conditions in the problem
def A : ℝ × ℝ := (3, -1) -- The coordinates of point A
def bisector_B := ∀ x, x = 0  -- The bisector of ∠B
def bisector_C := ∀ x y, y = x -- The bisector of ∠C

-- The proof statement
theorem line_BC_eq : ∀ (A : ℝ × ℝ) (bisector_B : ∀ x, x = 0) (bisector_C : ∀ x y, y = x),
  A = (3, -1) → 
  (bisector_B = (λ x, x = 0)) →
  (bisector_C = (λ x y, y = x)) →
  ∃ (a b c : ℝ), a * 2 + b * (-1) + c = 0 ∧ (a, b, c) = (2, -1, 5) :=
by 
  intro A bisector_B bisector_C hA hB hC
  use [2, -1, 5]
  split
  . exact sorry -- Placeholder for the proof that 2 * 2 + (-1) * (-1) + 5 = 0
  . exact sorry -- Placeholder for the proof that (a, b, c) = (2, -1, 5)

end line_BC_eq_l291_291459


namespace constant_term_in_binomial_expansion_l291_291525

theorem constant_term_in_binomial_expansion :
  let x := (λ (x : ℝ), (sqrt x - (2 / x^(1 / 3)))^5) in
  ∃ x0 : ℝ, x0 = -80 :=
by
  sorry

end constant_term_in_binomial_expansion_l291_291525


namespace range_of_a_l291_291352

-- Define the line
def line (a : ℝ) : set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ ax - y - 2a - 1 = 0}

-- Define the endpoints of the line segment
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (5, 2)

-- Define the condition for a point to be on the line segment AB
def on_segment (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in
    (y - 3) * (5 + 2) = (2 - 3) * (x + 2) ∧ -2 ≤ x ∧ x ≤ 5

-- Define the intersecting condition
def intersects (a : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, line a p ∧ on_segment p

-- Define the main theorem
theorem range_of_a (a : ℝ) : intersects a ↔ a ∈ set.Iic (-1) ∪ set.Ici 1 := by
  sorry

end range_of_a_l291_291352


namespace C_days_to_finish_l291_291030

theorem C_days_to_finish (A B C : ℝ) 
  (h1 : A + B = 1 / 15)
  (h2 : A + B + C = 1 / 11) :
  1 / C = 41.25 :=
by
  -- Given equations
  have h1 : A + B = 1 / 15 := sorry
  have h2 : A + B + C = 1 / 11 := sorry
  -- Calculate C
  let C := 1 / 11 - 1 / 15
  -- Calculate days taken by C
  let days := 1 / C
  -- Prove the days equal to 41.25
  have days_eq : 41.25 = 165 / 4 := sorry
  exact sorry

end C_days_to_finish_l291_291030


namespace trapezoid_angles_sum_l291_291379

theorem trapezoid_angles_sum {α β γ δ : ℝ} (h : α + β + γ + δ = 360) (h1 : α = 60) (h2 : β = 120) :
  γ + δ = 180 :=
by
  sorry

end trapezoid_angles_sum_l291_291379


namespace triangle_divided_by_midlines_l291_291767

theorem triangle_divided_by_midlines (S : ℝ) : 
  let original_triangle_area := 4 * S in 
  true := by sorry

end triangle_divided_by_midlines_l291_291767


namespace part1_part2_part3_l291_291726

-- Part 1
theorem part1 :
  3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 →
  Real.intPart (Real.sqrt 10) = 3 ∧ Real.decPart (Real.sqrt 10) = Real.sqrt 10 - 3 :=
by
  sorry

-- Part 2
theorem part2 :
  let a := Real.sqrt 6 - 2
  let b := 3
  a + b - Real.sqrt 6 = 1 :=
by
  sorry

-- Part 3
theorem part3 :
  let x := 13
  let y := Real.sqrt 3 - 1
  (12 + Real.sqrt 3 = x + y ∧ 0 < y ∧ y < 1) →
  -(x - y) = Real.sqrt 3 - 14 :=
by
  sorry

end part1_part2_part3_l291_291726


namespace shoe_pairing_probability_l291_291377

theorem shoe_pairing_probability :
  let total_adults := 12
  let total_pairings := Nat.factorial total_adults
  let good_pairings := Nat.factorial 11 + (Nat.factorial 12 / (2 * (Nat.factorial 6)^2)) * (Nat.factorial 5)^2
  let prob := good_pairings / total_pairings
  let frac := Rational.reduce 1 10
  let (m, n) := (frac.num, frac.denom)

  m + n = 11 := by
  sorry

end shoe_pairing_probability_l291_291377


namespace snail_maximum_movement_l291_291455

-- Define the conditions given in the problem
def snail_movement := ∃ (observe: ℕ → ℝ), (∀ t: ℕ, (0 ≤ t ∧ t < 6) → (observe t = 1)) ∧ 
  (∀ t1 t2 : ℕ, t1 ≠ t2 → (0 ≤ t1 ∧ t1 < 6) → (0 ≤ t2 ∧ t2 < 6) →

  ∃ segment_distance: ℝ, (segment_distance ≥ 0) ∧ 
  (segment_distance ≤ 10)

theorem snail_maximum_movement : snail_movement := sorry

end snail_maximum_movement_l291_291455


namespace range_of_f_l291_291750

def floor (x : ℝ) : ℤ := ⌊x⌋

def f (x : ℝ) : ℝ := ↑(floor x) - 2 * x

theorem range_of_f : Set.Icc (-∞ : ℝ) 0 = {y : ℝ | ∃ x : ℝ, f x = y} :=
sorry

end range_of_f_l291_291750


namespace shortest_multicolored_cycle_l291_291702

-- Define vertices and edges
variables {V: Type*} [fintype V] -- Vertex set
variables {E: Type*} [fintype E] -- Edge set
variables (cycle : list E) (color : E → ℕ) -- Cycle and color function

-- Define the conditions
def is_vertex_horizontal (v : V) : Prop := sorry -- Predicate for horizontal vertices
def is_vertex_vertical (v : V) : Prop := sorry -- Predicate for vertical vertices
def edges_of_cycle (cycle : list E) : list (V × V) := sorry -- Extract edges from the cycle
def are_edges_multicolored (edges : list (V × V)) : Prop := sorry -- Check if edges are multicolored

-- Define the length of the cycle
def cycle_length : ℕ := cycle.length

-- Prove the shortest multicolored cycle has 4 edges
theorem shortest_multicolored_cycle (h_cycle: ∀ (a_i b_i : V) (h1: is_vertex_horizontal a_i) (h2: is_vertex_vertical b_i),
  ∃ s, edges_of_cycle cycle = list.zip (list.repeat a_i s) (list.repeat b_i s) ∧ cycle_length = 2 * s)
    (h_s: ∃ s, s > 2)
  : ∃ s, s = 2 :=
by
  sorry

end shortest_multicolored_cycle_l291_291702


namespace octal_rep_square_l291_291626

theorem octal_rep_square (a b c : ℕ) (n : ℕ) (h : n^2 = 8^3 * a + 8^2 * b + 8 * 3 + c) (h₀ : a ≠ 0) : c = 1 :=
sorry

end octal_rep_square_l291_291626


namespace highest_power_of_2_dividing_15_pow_4_minus_9_pow_4_l291_291114

theorem highest_power_of_2_dividing_15_pow_4_minus_9_pow_4 :
  (∃ k, 15^4 - 9^4 = 2^k * m ∧ ¬ ∃ m', m = 2 * m') ∧ (k = 5) :=
by
  sorry

end highest_power_of_2_dividing_15_pow_4_minus_9_pow_4_l291_291114


namespace geometric_sum_n_l291_291563

theorem geometric_sum_n (
  m n : ℕ,
  a : ℕ → ℕ,
  h_seq : ∀ k, a k = 3 * 2^(k-1),
  h_m_lt_n : m < n,
  h_sum : (∑ k in finset.range (n - m + 1), a (m + k)) = 360
) : n = 7 := by
  sorry

end geometric_sum_n_l291_291563


namespace area_of_triangle_APB_l291_291570

theorem area_of_triangle_APB (A B C D P : Point) (s : ℝ)
  (h_square : Square A B C D s) 
  (h_sides : s = 8) 
  (h_dist : dist P A = dist P B ∧ dist P B = dist P C)
  (h_perp : Perpendicular (line_through P D) (line_through A B)) :
  area (Triangle A P B) = 12 := 
sorry

end area_of_triangle_APB_l291_291570


namespace minimum_value_4_l291_291919

open Set Real

/-- Define the functions -/
def f1 (x : ℝ) : ℝ := x^2 - 4x + 8
def f2 (x : ℝ) : ℝ := (x^2 - 2x + 5) / (x - 1)

/-- State the problem as a theorem -/
theorem minimum_value_4 (x : ℝ) (hx : 1 < x) :
  (Inf (set.range f1) = 4 ∨ Inf (set.range f2) = 4) := sorry

end minimum_value_4_l291_291919


namespace cos_B_value_l291_291631

theorem cos_B_value 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : (Real.sin A - Real.sin B) * (a + b) = (1/2 * a - c) * Real.sin C)
  (h2 : a = abs(C) * Real.sin A)
  (h3 : b = abs(C) * Real.sin B)
  (h4 : c = abs(C) * Real.sin C)
  : Real.cos B = 1 / 4 := 
  by sorry

end cos_B_value_l291_291631


namespace aubrey_average_speed_l291_291486

theorem aubrey_average_speed : 
  ∀ (distance time : ℕ), 
  distance = 88 →
  time = 4 →
  distance / time = 22 := by
  intros distance time
  intros h_distance h_time
  rw [h_distance, h_time]
  norm_num
  exact rfl

end aubrey_average_speed_l291_291486


namespace length_of_first_platform_l291_291079

-- Definitions corresponding to conditions
def length_train := 310
def time_first_platform := 15
def length_second_platform := 250
def time_second_platform := 20

-- Time-speed relationship
def speed_first_platform (L : ℕ) : ℚ := (length_train + L) / time_first_platform
def speed_second_platform : ℚ := (length_train + length_second_platform) / time_second_platform

-- Theorem to prove length of first platform
theorem length_of_first_platform (L : ℕ) (h : speed_first_platform L = speed_second_platform) : L = 110 :=
by
  sorry

end length_of_first_platform_l291_291079


namespace tom_dance_lessons_cost_l291_291372

theorem tom_dance_lessons_cost (total_lessons free_lessons : ℕ) (cost_per_lesson : ℕ) (h1 : total_lessons = 10) (h2 : free_lessons = 2) (h3 : cost_per_lesson = 10) : total_lessons * cost_per_lesson - free_lessons * cost_per_lesson = 80 :=
by
  rw [h1, h2, h3]
  sorry

end tom_dance_lessons_cost_l291_291372


namespace algae_increase_l291_291301

/-- Milford Lake was originally blue because it only had 809 algae plants.
    Now there are 3263 algae plants, and the lake has turned green.
    Prove that there are 2454 more algae plants in Milford Lake now. -/
theorem algae_increase (original_count current_count : ℕ) 
    (h_original : original_count = 809) 
    (h_current : current_count = 3263) : 
    current_count - original_count = 2454 :=
by
  rw [h_current, h_original]
  norm_num

end algae_increase_l291_291301


namespace sequence_a_general_term_sum_first_n_terms_b_l291_291218

namespace sequence_math_problems

open_locale big_operators

def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 4 else sequence_a (n - 1) + 2^(n - 1) + 3

def sequence_b (n : ℕ) : ℝ :=
  (sequence_a n) / 2^n

def sum_sequence_b (n : ℕ) : ℝ :=
  ∑ i in finset.range n, sequence_b (i + 1)

theorem sequence_a_general_term :
  ∀ n ≥ 1, n = 1 → sequence_a n = 4 ∧ ∀ n ≥ 2, sequence_a n = sequence_a (n - 1) + 2^(n - 1) + 3 → sequence_a n = 2^n + 3*n - 1 :=
begin
  sorry
end

theorem sum_first_n_terms_b (n : ℕ) : sum_sequence_b n = n + 5 - (3 * n + 5) / 2^n :=
begin
  sorry
end

end sequence_math_problems

end sequence_a_general_term_sum_first_n_terms_b_l291_291218


namespace functional_equation_solution_l291_291127

variable {f : ℝ → ℝ} (a b c : ℝ)

-- Conditions
def satisfies_condition (f : ℝ → ℝ) : Prop :=
∀ (a b : ℝ), 0 < a → 0 < b →
  f(b) - f(a) = (b - a) * (f' (√(a * b)))

-- Desired form of the function
def desired_form (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
∀ (x : ℝ), 0 < x →
  f(x) = a + b * x + c / x

-- Theorem
theorem functional_equation_solution {f : ℝ → ℝ} (hf : satisfies_condition f) :
  ∃ (a b c : ℝ), desired_form f a b c :=
sorry

end functional_equation_solution_l291_291127


namespace log_monotonic_decreasing_interval_l291_291755

noncomputable def f (x : ℝ) := Real.log(1/2) (-x^2 + 6 * x - 5)

theorem log_monotonic_decreasing_interval :
  (∀ x, 1 < x ∧ x ≤ 3 → -x^2 + 6 * x - 5 > 0) →
  (∀ x, 3 ≤ x ∧ x < 5 → -x^2 + 6 * x - 5 > 0) →
  (∀ x1 x2, 1 < x1 ∧ x1 ≤ 3 ∧ x2 ≤ 3 → x1 < x2 → f x1 > f x2) →
  ∃ a b : ℝ, 1 < a ∧ b ≤ 3 ∧ ∀ x, a < x ∧ x ≤ b → f x > f (x + ε) :=
  sorry

end log_monotonic_decreasing_interval_l291_291755


namespace no_continuous_coverage_l291_291449

noncomputable def running_track_problem : Prop :=
  let track_length := 1 -- 1 km
  let stands_arc_length := 0.1 -- 100 meters = 0.1 km
  let runners_speeds := [20, 21, 22, 23, 24, 25, 26, 27, 28, 29] -- km/h
  ∃ (runner_positions : Fin 10 → ℝ), -- Starting positions (unit: km) of 10 runners
    ∀ (t : ℝ), -- For any time (unit: hours)
      ∃ (i : Fin 10), -- There exists a runner (among 10)
        let position := (runner_positions i + runners_speeds.nth i * t) % track_length
        in  position ≤ stands_arc_length

theorem no_continuous_coverage :
  ¬running_track_problem :=
sorry

end no_continuous_coverage_l291_291449


namespace bee_always_wins_game_l291_291084

noncomputable def initial_pile : ℕ := 2020

structure GameState where
  piles : List ℕ
  deriving Inhabited

inductive Player
| Amy
| Bee

def is_move_possible (piles : List ℕ) : Prop :=
  ∃ p ∈ piles, p > 1

def divide_pile (pile : ℕ) (divs : List ℕ) : List ℕ :=
  if list.sum divs = pile ∧ list.all divs (fun d => d > 0) ∧ (list.length divs = 2 ∨ list.length divs = 3) then divs else [pile]

noncomputable def make_move (piles : List ℕ) (pile : ℕ) (divs : List ℕ) : List ℕ :=
  match list.partition (λ p => p = pile) piles with
  | ([], _) => piles  -- no matching pile found, invalid move, return original state (assuming no move possible here)
  | ([p], remaining_piles) => divide_pile p divs ++ remaining_piles
  | (_, _) => piles  -- should not reach here, as only one pile can be chosen

def next_player (current : Player) : Player :=
  match current with
  | Player.Amy => Player.Bee
  | Player.Bee => Player.Amy

theorem bee_always_wins_game :
  ∀ (state : GameState) (current_player : Player),
  state.piles = [initial_pile, initial_pile, initial_pile] →
  ∀ (move_possible : is_move_possible state.piles),
  ∃ (next_state : GameState),
  (∀ move_possible_in_next_state : is_move_possible next_state.piles, next_player current_player = Player.Bee) →  -- Bee's strategy can always respond
  ∀ state'.piles.len = 1 ∧ state'.piles.head = 1,  -- Amy cannot make a move
  next_player current_player = Player.Amy →  -- It will be Amy's turn
  ¬ is_move_possible state'.piles :=
sorry

end bee_always_wins_game_l291_291084


namespace max_min_PA_l291_291165

open Classical

variables (A B P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace P]
          (dist_AB : ℝ) (dist_PA_PB : ℝ)

noncomputable def max_PA (A B : Type) [MetricSpace A] [MetricSpace B] (dist_AB : ℝ) : ℝ := sorry
noncomputable def min_PA (A B : Type) [MetricSpace A] [MetricSpace B] (dist_AB : ℝ) : ℝ := sorry

theorem max_min_PA (A B : Type) [MetricSpace A] [MetricSpace B] [Inhabited P]
                   (dist_AB : ℝ) (dist_PA_PB : ℝ) :
  dist_AB = 4 → dist_PA_PB = 6 → max_PA A B 4 = 5 ∧ min_PA A B 4 = 1 :=
by
  intros h_AB h_PA_PB
  sorry

end max_min_PA_l291_291165


namespace number_of_even_digits_in_base7_of_528_l291_291139

/-
  Define the base-7 representation of a number and a predicate to count even digits.
-/

-- Definition of base-7 digit representation
def base7_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else (List.unfoldr (λ n, if n = 0 then Option.none else some (n % 7, n / 7)) n).reverse

-- Predicate to check if a digit is even
def is_even (d : ℕ) : Bool := d % 2 = 0

-- Counting the even digits in base-7 representation
def count_even_digits_in_base7 (n : ℕ) : ℕ :=
  (base7_repr n).countp is_even

-- The target theorem to prove
theorem number_of_even_digits_in_base7_of_528 : count_even_digits_in_base7 528 = 0 :=
by sorry

end number_of_even_digits_in_base7_of_528_l291_291139


namespace find_second_number_l291_291741

def average (nums : List ℕ) : ℕ :=
  nums.sum / nums.length

theorem find_second_number (nums : List ℕ) (a b : ℕ) (avg : ℕ) :
  average [10, 70, 28] = 36 ∧ average (10 :: 70 :: 28 :: []) + 4 = avg ∧ average (a :: b :: nums) = avg ∧ a = 20 ∧ b = 60 → b = 60 :=
by
  sorry

end find_second_number_l291_291741


namespace meeting_success_probability_l291_291053

noncomputable def probability_meeting_success : ℝ :=
  let engineers_arrival_times := (0, 0, 0) in
  let boss_arrival_time := 0 in
  let meeting_successful := 
    engineers_arrival_times.1 ∈ set.Icc 0 3 ∧ 
    engineers_arrival_times.2 ∈ set.Icc 0 3 ∧ 
    engineers_arrival_times.3 ∈ set.Icc 0 3 ∧ 
    boss_arrival_time ∈ set.Icc 0 3 ∧
    engineers_arrival_times.1 - engineers_arrival_times.2 ≤ 1.5 ∧
    engineers_arrival_times.2 - engineers_arrival_times.3 ≤ 1.5 ∧
    engineers_arrival_times.3 - engineers_arrival_times.1 ≤ 1.5 ∧
    boss_arrival_time > engineers_arrival_times.1 ∧
    boss_arrival_time > engineers_arrival_times.2 ∧
    boss_arrival_time > engineers_arrival_times.3
  in
  if meeting_successful
  then 1/20
  else 0

theorem meeting_success_probability : probability_meeting_success = 1/20 := sorry

end meeting_success_probability_l291_291053


namespace jane_stick_length_l291_291716

variable (P U S J F : ℕ)
variable (h1 : P = 30)
variable (h2 : U = P - 7)
variable (h3 : U = S / 2)
variable (h4 : F = 2 * 12)
variable (h5 : J = S - F)

theorem jane_stick_length : J = 22 := by
  sorry

end jane_stick_length_l291_291716


namespace min_abs_phi_l291_291625

noncomputable def minimum_abs_phi_symmetrical (ϕ : ℝ) : ℝ :=
if h : (∃ k : ℤ, ϕ = k * real.pi - (5 * real.pi / 6)) then
  |∃ k, ((λ k, k * real.pi - (5 * real.pi / 6)) k = ϕ)|
else
  0

theorem min_abs_phi (ϕ : ℝ) :
  (∃ k : ℤ, ϕ = k * real.pi - (5 * real.pi / 6)) →
  minimum_abs_phi_symmetrical ϕ = real.pi / 6 := 
by
sorry

end min_abs_phi_l291_291625


namespace first_tree_height_l291_291109

theorem first_tree_height
  (branches_first : ℕ)
  (branches_second : ℕ)
  (height_second : ℕ)
  (branches_third : ℕ)
  (height_third : ℕ)
  (branches_fourth : ℕ)
  (height_fourth : ℕ)
  (average_branches_per_foot : ℕ) :
  branches_first = 200 →
  height_second = 40 →
  branches_second = 180 →
  height_third = 60 →
  branches_third = 180 →
  height_fourth = 34 →
  branches_fourth = 153 →
  average_branches_per_foot = 4 →
  branches_first / average_branches_per_foot = 50 :=
by
  sorry

end first_tree_height_l291_291109


namespace initial_boxes_l291_291430

theorem initial_boxes (x : ℕ) (h1 : 80 + 165 = 245) (h2 : 2000 * 245 = 490000) 
                      (h3 : 4 * 245 * x + 245 * x = 1225 * x) : x = 400 :=
by
  sorry

end initial_boxes_l291_291430


namespace coefficient_a2b3c3_in_a_plus_b_plus_c_pow_8_general_term_formula_a_plus_b_plus_c_pow_n_l291_291913

-- Part (1): Prove the coefficient of a^2b^3c^3 in the expansion of (a + b + c)^8 is 560
theorem coefficient_a2b3c3_in_a_plus_b_plus_c_pow_8 : 
  ∃ coef : ℕ, coef = 560 ∧ 
            (∃ T : (ℕ × ℕ × ℕ) → ℕ, 
              T (2, 3, 3) = coef ∧ 
              T = λ (i j k : ℕ) => nat.choose 8 (i+j) * nat.choose (i+j) i) :=
sorry

-- Part (2): Prove the general term formula for the expansion of (a + b + c)^n
theorem general_term_formula_a_plus_b_plus_c_pow_n (n r k : ℕ) (h1 : k ≤ r) (h2 : r ≤ n) :
  ∃ T : (ℕ × ℕ × ℕ) → ℕ, 
    T (r - k, k, n - r) = nat.choose n r * nat.choose r k :=
sorry

end coefficient_a2b3c3_in_a_plus_b_plus_c_pow_8_general_term_formula_a_plus_b_plus_c_pow_n_l291_291913


namespace cost_price_per_meter_l291_291454

-- Given conditions
def total_selling_price : ℕ := 18000
def total_meters_sold : ℕ := 400
def loss_per_meter : ℕ := 5

-- Statement to be proven
theorem cost_price_per_meter : 
    ((total_selling_price + (loss_per_meter * total_meters_sold)) / total_meters_sold) = 50 := 
by
    sorry

end cost_price_per_meter_l291_291454


namespace solution_set_of_inequality_l291_291922

theorem solution_set_of_inequality
  (a b : ℝ)
  (x y : ℝ)
  (h1 : a * (-2) + b = 3)
  (h2 : a * (-1) + b = 2)
  :  -x + 1 < 0 ↔ x > 1 :=
by 
  -- Proof goes here
  sorry

end solution_set_of_inequality_l291_291922


namespace election_winner_percentage_l291_291639

theorem election_winner_percentage (total_votes winner_votes loser_votes : ℕ) 
  (h_winner_votes : winner_votes = 837)
  (h_won_by : winner_votes - loser_votes = 324)
  (h_total : total_votes = winner_votes + loser_votes) :
  (winner_votes : ℝ) / (total_votes : ℝ) * 100 ≈ 62 :=
by
  sorry

end election_winner_percentage_l291_291639


namespace arithmetic_series_sum_l291_291904

theorem arithmetic_series_sum
  (a₁ an d : ℝ) (h₀ : a₁ = 10) (h₁ : an = 30) (h₂ : d = 0.5) :
  let n := ((an - a₁) / d) + 1 in
  let S := (n * (a₁ + an)) / 2 in
  S = 820 :=
by 
  sorry

end arithmetic_series_sum_l291_291904


namespace triangle_medians_area_square_side_AB_l291_291754

theorem triangle_medians_area_square_side_AB (A B C : Point)
  (BC_len : dist B C = 28) 
  (AC_len : dist A C = 44)
  (medians_perpendicular : ∃ E F M, is_midpoint E B C ∧ is_midpoint F A C ∧ is_centroid M A B C ∧ ∠ BMF = π/2) :
  let AB := dist A B in 
  AB * AB = 544 := 
sorry

end triangle_medians_area_square_side_AB_l291_291754


namespace area_of_rhombus_l291_291785

-- Define the conditions in Lean.
def side_length : ℝ := 2
def angle_DAB : ℝ := 45 / 180 * Real.pi -- angle in radians

-- Define the problem as a theorem.
theorem area_of_rhombus (a b : ℝ) (angle : ℝ)
  (h1 : a = side_length)
  (h2 : b = side_length)
  (h3 : angle = angle_DAB ∨ angle = -angle_DAB):
  Real := 
  let base := a * Real.sin(angle)
  let height := b * Real.cos(angle)
  base * height
  
#eval area_of_rhombus 2 2 (Real.pi / 4) ((by rfl) sorry sorry)

end area_of_rhombus_l291_291785


namespace subset_contains_three_numbers_l291_291274

-- Define the set X
def X : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- The main theorem statement
theorem subset_contains_three_numbers (A B : Set ℕ) (hAB : A ∪ B = X) (h_disjoint : A ∩ B = ∅) :
  ∃ (S : Set ℕ), S ⊆ A ∨ S ⊆ B ∧ ∃ (a b c ∈ S), a + b = 2 * c :=
begin
  sorry
end

end subset_contains_three_numbers_l291_291274


namespace largestCompositeMersenneLessThan300_l291_291426

theorem largestCompositeMersenneLessThan300 :
  ∃ n : ℕ, 2 ^ n - 1 < 300 ∧ ¬nat.prime (2 ^ n - 1) ∧ 
  ∀ m : ℕ, 2^m - 1 < 300 ∧ ¬nat.prime (2^m - 1) → 2^m - 1 ≤ 2^n - 1 := 
by
  sorry

end largestCompositeMersenneLessThan300_l291_291426


namespace expected_audience_l291_291779

theorem expected_audience (Sat Mon Wed Fri : ℕ) (extra_people expected_total : ℕ)
  (h1 : Sat = 80)
  (h2 : Mon = 80 - 20)
  (h3 : Wed = Mon + 50)
  (h4 : Fri = Sat + Mon)
  (h5 : extra_people = 40)
  (h6 : expected_total = Sat + Mon + Wed + Fri - extra_people) :
  expected_total = 350 := 
sorry

end expected_audience_l291_291779


namespace chad_cat_food_packages_l291_291497

theorem chad_cat_food_packages (c : ℕ)
  (h1 : ∃ c, True)
  (h2 : ∀ c, 9 * c = 2 * 3 + 48) :
  c = 6 :=
sorry

end chad_cat_food_packages_l291_291497


namespace positive_integer_geometric_divisors_l291_291522

-- Setting up the Lean environment to work with the requisite set of imports.

theorem positive_integer_geometric_divisors (n : ℕ) (h : 4 ≤ (nat.divisors n).length) :
  (∃ p : ℕ, nat.prime p ∧ ∃ α : ℕ, α ≥ 3 ∧ n = p ^ α) :=
by
  sorry

end positive_integer_geometric_divisors_l291_291522


namespace no_real_roots_contradiction_l291_291369

open Real

variables (a b : ℝ)

theorem no_real_roots_contradiction (h : ∀ x : ℝ, a * x^3 + a * x + b ≠ 0) : false :=
by
  sorry

end no_real_roots_contradiction_l291_291369


namespace complement_of_A_in_S_l291_291672

universe u

def S : Set ℕ := {x | 0 ≤ x ∧ x ≤ 5}
def A : Set ℕ := {x | 1 < x ∧ x < 5}

theorem complement_of_A_in_S : S \ A = {0, 1, 5} := 
by sorry

end complement_of_A_in_S_l291_291672


namespace num_even_digits_in_base7_of_528_is_zero_l291_291142

def is_digit_even_base7 (d : ℕ) : Prop :=
  (d = 0) ∨ (d = 2) ∨ (d = 4) ∨ (d = 6)

def base7_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0] else (List.unfoldr (λ x, if x = 0 then none else some (x % 7, x / 7)) n).reverse

def number_of_even_digits_base7 (n : ℕ) : ℕ :=
  List.countp is_digit_even_base7 (base7_representation n)

theorem num_even_digits_in_base7_of_528_is_zero : number_of_even_digits_base7 528 = 0 :=
by
  sorry

end num_even_digits_in_base7_of_528_is_zero_l291_291142


namespace distance_downstream_24_l291_291059

-- Variables:
def V_m : ℝ := 5  -- speed of the man in still water
def T : ℝ := 4    -- time taken for both upstream and downstream swims

-- Given that man swims upstream 16 km in 4 hours.
def Distance_upstream : ℝ := 16

-- The theorem to prove: The distance the man swims downstream.
theorem distance_downstream_24 :
  ∃ V_r : ℝ, let V_downstream := V_m + V_r
  ∧ let V_upstream := V_m - V_r
  ∧ (Distance_upstream = V_upstream * T)
  ∧ (Distance_downstream = V_downstream * T)
  ∧ Distance_downstream = 24 :=
begin
  use 1,
  simp [V_downstream, V_upstream, Distance_upstream, V_m, T],
  split,
  { linarith },
  split,
  { simp [Distance_downstream], linarith },
  { sorry }
end

end distance_downstream_24_l291_291059


namespace find_fg_length_l291_291461

theorem find_fg_length
  (A B C D E F G : Point)
  (hABC : EquilateralTriangle A B C)
  (hADE : EquilateralTriangle A D E)
  (hBDG : EquilateralTriangle B D G)
  (hCEF : EquilateralTriangle C E F)
  (side_length_ABC : AB = 10)
  (AD_length : AD = 3) :
  FG = 4 :=
sorry

end find_fg_length_l291_291461


namespace main_inequality_l291_291569

noncomputable def sequence (n : ℕ) : ℝ := sorry

axiom positive_sequence (n : ℕ) : 0 < sequence n
axiom monotonically_decreasing_sequence (n m : ℕ) (h : n ≤ m) : sequence m ≤ sequence n
axiom given_inequality (n : ℕ) : finset.sum (finset.range n).filter (λ k, ∃ m, k = m^2) (λ k, sequence k / (k + 1)) ≤ 1

theorem main_inequality (n : ℕ) : finset.sum (finset.range n) (λ k, sequence k / (k + 1)) < 3 :=
sorry

end main_inequality_l291_291569


namespace value_of_M_l291_291357

theorem value_of_M {a b c d e M : ℤ} 
  (h1 : ArithmeticSequence [a, _, _, _, d, _, _])
  (h2 : ArithmeticSequence [32, 25, c])
  (h3 : ArithmeticSequence [M, d, a, e, 25, 32])
  (h4 : d = 11)
  (h5 : e = -10) :
  M = -6 := by
  sorry

end value_of_M_l291_291357


namespace minimum_value_expr_l291_291287

noncomputable def expr (x y z : ℝ) : ℝ :=
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) +
  (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2)))

theorem minimum_value_expr : ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) →
  expr x y z ≥ 2 :=
by sorry

end minimum_value_expr_l291_291287


namespace prove_subset_zero_in_A_l291_291594

variable (A : set ℤ)
def SetA : set ℤ := {x | x > -1}

theorem prove_subset_zero_in_A : {0} ⊆ SetA := by 
  sorry

end prove_subset_zero_in_A_l291_291594


namespace skew_lines_sufficient_condition_l291_291251

noncomputable def intersecting_planes (α β : Plane) : Prop := 
  α ∩ β ≠ ∅

noncomputable def projection_to_plane (l : Line) (α : Plane) : Line :=
  sorry

noncomputable def are_skew_lines (l1 l2 : Line) : Prop :=
  (∀ α β : Plane, intersecting_planes α β → 
    let S1 := projection_to_plane l1 α;
    let S2 := projection_to_plane l2 α;
    let t1 := projection_to_plane l1 β;
    let t2 := projection_to_plane l2 β;
    (S1.parallel S2 ∧ t1.intersects t2) ∨ (t1.parallel t2 ∧ S1.intersects S2))

theorem skew_lines_sufficient_condition (α β : Plane) (l1 l2 : Line) :
  intersecting_planes α β →
  let S1 := projection_to_plane l1 α;
  let S2 := projection_to_plane l2 α;
  let t1 := projection_to_plane l1 β;
  let t2 := projection_to_plane l2 β;
  (S1.parallel S2 ∧ t1.intersects t2) ∨ (t1.parallel t2 ∧ S1.intersects S2) → 
  are_skew_lines l1 l2 :=
sorry

end skew_lines_sufficient_condition_l291_291251


namespace compare_travel_times_l291_291256

variable (v : ℝ) (t1 t2 : ℝ)

def travel_time_first := t1 = 100 / v
def travel_time_second := t2 = 200 / v

theorem compare_travel_times (h1 : travel_time_first v t1) (h2 : travel_time_second v t2) : 
  t2 = 2 * t1 :=
by
  sorry

end compare_travel_times_l291_291256


namespace compare_powers_l291_291916

axiom a : ℝ := 2 ^ (4 / 3)
axiom b : ℝ := 3 ^ (2 / 3)
axiom c : ℝ := 2.5 ^ (1 / 3)

theorem compare_powers : c < b ∧ b < a :=
by
  sorry

end compare_powers_l291_291916


namespace p_sufficient_but_not_necessary_for_q_l291_291968

-- Definitions corresponding to conditions
def p (x : ℝ) : Prop := x > 1
def q (x : ℝ) : Prop := 1 / x < 1

-- Theorem stating the relationship between p and q
theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l291_291968


namespace orthogonal_trajectory_eqn_l291_291132

theorem orthogonal_trajectory_eqn (a C : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 2 * a * x) → 
  (∃ C : ℝ, ∀ x y : ℝ, x^2 + y^2 = C * y) :=
sorry

end orthogonal_trajectory_eqn_l291_291132


namespace single_elimination_games_l291_291637

theorem single_elimination_games (n : ℕ) (h : n = 128) : (n - 1) = 127 :=
by
  sorry

end single_elimination_games_l291_291637


namespace water_left_after_experiment_l291_291685

theorem water_left_after_experiment (initial_water : ℚ) (water_used : ℚ) :
  initial_water = 3 → water_used = 5 / 4 → (initial_water - water_used) = 7 / 4 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end water_left_after_experiment_l291_291685


namespace slope_ratios_equal_l291_291203

theorem slope_ratios_equal (x y : ℝ) (P A B C M N : ℝ × ℝ)
  (h_E : ∀ (x y : ℝ), x^2 + 4*y^2 = 4)
  (h_M : M = (-2, 2))
  (h_N : N = (2, 0))
  (h_P : P = (-2, 2))
  (h_A1 : A.1 > 0)
  (h_A2 : A.2 > 0)
  (h_B1 : B.1 > 0)
  (h_B2 : B.2 > 0)
  (h_A_on_BP : (B.A P))
  (h_OP_intersects_NA_at_C : (OP ∩ NA = C))
  (k_AM k_AC k_MB k_MC : ℝ)
  (h_k_AM : k_AM = (A.2 - M.2) / (A.1 - M.1))
  (h_k_AC : k_AC = (A.2 - C.2) / (A.1 - C.1))
  (h_k_MB : k_MB = (B.2 - M.2) / (B.1 - M.1))
  (h_k_MC : k_MC = (C.2 - M.2) / (C.1 - M.1))
  : k_MB / k_AM = k_AC / k_MC :=
begin
  sorry
end

end slope_ratios_equal_l291_291203


namespace unique_solution_x_l291_291881

theorem unique_solution_x (x : ℕ) : 
  (∃ n : ℕ, 2 * x + 1 = n^2) ∧ 
  (∀ k : ℕ, 2 ≤ k → k ≤ x → ¬ ∃ m : ℕ, 2 * x + k = m^2) → 
  x = 4 :=
begin
  sorry
end

end unique_solution_x_l291_291881


namespace cost_per_tissue_box_l291_291474

-- Given conditions
def rolls_toilet_paper : ℝ := 10
def cost_per_toilet_paper : ℝ := 1.5
def rolls_paper_towels : ℝ := 7
def cost_per_paper_towel : ℝ := 2
def boxes_tissues : ℝ := 3
def total_cost : ℝ := 35

-- Deduction of individual costs
def cost_toilet_paper := rolls_toilet_paper * cost_per_toilet_paper
def cost_paper_towels := rolls_paper_towels * cost_per_paper_towel
def cost_tissues := total_cost - cost_toilet_paper - cost_paper_towels

-- Prove the cost for one box of tissues
theorem cost_per_tissue_box : (cost_tissues / boxes_tissues) = 2 :=
by
  sorry

end cost_per_tissue_box_l291_291474


namespace restore_to_original_without_limited_discount_restore_to_original_with_limited_discount_l291_291062

-- Let P be the original price of the jacket
variable (P : ℝ)

-- The price of the jacket after successive reductions
def price_after_discount (P : ℝ) : ℝ := 0.60 * P

-- The price of the jacket after all discounts including the limited-time offer
def price_after_full_discount (P : ℝ) : ℝ := 0.54 * P

-- Prove that to restore 0.60P back to P a 66.67% increase is needed
theorem restore_to_original_without_limited_discount :
  ∀ (P : ℝ), (P > 0) → (0.60 * P) * (1 + 66.67 / 100) = P :=
by sorry

-- Prove that to restore 0.54P back to P an 85.19% increase is needed
theorem restore_to_original_with_limited_discount :
  ∀ (P : ℝ), (P > 0) → (0.54 * P) * (1 + 85.19 / 100) = P :=
by sorry

end restore_to_original_without_limited_discount_restore_to_original_with_limited_discount_l291_291062


namespace true_weight_of_pudding_l291_291085

noncomputable def pudding_true_weight (x W1 W2 : ℝ) : Prop :=
  (W1 = (9 / 11) * x + 4) ∧
  (W2 = W1 + 48) ∧
  ((W1 + W2) / 2 = x)

theorem true_weight_of_pudding : ∃ x, pudding_true_weight x ((9 / 11) * x + 4) ( (9 / 11) * x + 52 ) ∧ x = 154 :=
by
  use 154
  split
  -- Condition 1: W1 = (9 / 11) * x + 4
  show (9 / 11) * 154 + 4 = (9 / 11) * 154 + 4
  -- Condition 2: W2 = W1 + 48
  show (9 / 11) * 154 + 4 + 48 = ( (9 / 11) * 154 + 4 ) + 48
  -- Condition 3: (W1 + W2) / 2 = x
  show ((9 / 11) * 154 + 4 + (9 / 11) * 154 + 52) / 2 = 154
  -- State x = 154
  show 154 = 154
  sorry

end true_weight_of_pudding_l291_291085


namespace circle_properties_l291_291981

open Real

structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

def Circle1 : Circle := ⟨(1, 0), 1⟩
def Circle2 : Circle := ⟨(-1, 2), 2⟩

def circle_eq (C : Circle) : Set (ℝ × ℝ) :=
  {p | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2}

def point_on_line (A B P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

def line_perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  let m1 := (l1.1.2 - l1.2.2) / (l1.1.1 - l1.2.1)
  let m2 := (l2.1.2 - l2.2.2) / (l2.1.1 - l2.2.1)
  m1 * m2 = -1

theorem circle_properties (A B P : ℝ × ℝ) :
  (A ∈ circle_eq Circle1 ∧ A ∈ circle_eq Circle2) →
  (B ∈ circle_eq Circle1 ∧ B ∈ circle_eq Circle2) →
  point_on_line A B P →
  let AB := {(A, B)} in
  let C1C2 := {(Circle1.center, Circle2.center)} in
  (line_perpendicular AB C1C2) ∧
  (∀ x y : ℝ, (4 * x - 4 * y + 1 = 0) → (x, y) ∈ AB) ∧
  (∀ x y : ℝ, (x + y - 1 = 0) → (x, y) ∈ {(C1C2.1.1 / 2 + C1C2.2.1 / 2, C1C2.1.2 / 2 + C1C2.2.2 / 2)}) :=
by
  sorry

end circle_properties_l291_291981


namespace johns_profit_l291_291663

def profit : ℚ := 727.25

theorem johns_profit
  (woodburnings_sales : ℕ := 20)
  (woodburning_price : ℚ := 15)
  (metal_sculptures_sales : ℕ := 15)
  (metal_sculpture_price : ℚ := 25)
  (paintings_sales : ℕ := 10)
  (painting_price : ℚ := 40)
  (wood_cost : ℚ := 100)
  (metal_cost : ℚ := 150)
  (paint_cost : ℚ := 120)
  (discount_rate : ℚ := 0.1)
  (tax_rate : ℚ := 0.05) :
  let total_woodburning_sales := woodburnings_sales * woodburning_price
      discount := total_woodburning_sales * discount_rate
      discounted_wood_sales := total_woodburning_sales - discount
      total_metal_sales := metal_sculptures_sales * metal_sculpture_price
      total_painting_sales := paintings_sales * painting_price
      total_sales_after_discount := discounted_wood_sales + total_metal_sales + total_painting_sales
      sales_tax := total_sales_after_discount * tax_rate
      total_collected := total_sales_after_discount + sales_tax
      total_cost := wood_cost + metal_cost + paint_cost
      profit := total_collected - total_cost
  in profit = 727.25 :=
by
  sorry

end johns_profit_l291_291663


namespace count_two_digit_numbers_with_digit_8_l291_291992

theorem count_two_digit_numbers_with_digit_8 : 
  let two_digit_integers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let has_eight (n : ℕ) := n / 10 = 8 ∨ n % 10 = 8
  (two_digit_integers.filter has_eight).card = 18 :=
by
  let two_digit_integers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let has_eight (n : ℕ) := n / 10 = 8 ∨ n % 10 = 8
  show (two_digit_integers.filter has_eight).card = 18
  sorry

end count_two_digit_numbers_with_digit_8_l291_291992


namespace find_vector_b_l291_291163

def vec_a := (Real.sqrt 3, Real.sqrt 5)
def vec_b (x y : ℝ) := (x, y)
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_vector_b (x y : ℝ) :
  let b := vec_b x y in
  perpendicular vec_a b ∧ magnitude b = 2 ↔
  (b = (-Real.sqrt 10 / 2, Real.sqrt 6 / 2) ∨ b = (Real.sqrt 10 / 2, -Real.sqrt 6 / 2)) :=
by
  sorry

end find_vector_b_l291_291163


namespace max_t_value_l291_291619

theorem max_t_value :
  (∀ (u v : ℝ), (u + 5 - 2 * v)^2 + (u - v^2)^2 ≥ t^2) → t > 0 → t ≤ 2 * real.sqrt 2 :=
by
  intros h1 h2
  sorry

end max_t_value_l291_291619


namespace derivative_y_is_l291_291746

-- Define the function y = cos(2x^2 + x)
def y (x : ℝ) : ℝ := cos (2 * x^2 + x)

-- The statement to prove: the derivative of y w.r.t x
theorem derivative_y_is : ∀ (x : ℝ), deriv y x = -(4 * x + 1) * sin (2 * x^2 + x) :=
by
  intro x
  sorry

end derivative_y_is_l291_291746


namespace necessary_and_sufficient_condition_l291_291745

-- Define the first circle
def circle1 (m : ℝ) : Set (ℝ × ℝ) :=
  { p | (p.1 + m)^2 + p.2^2 = 1 }

-- Define the second circle
def circle2 : Set (ℝ × ℝ) :=
  { p | (p.1 - 2)^2 + p.2^2 = 4 }

-- Define the condition -1 ≤ m ≤ 1
def condition (m : ℝ) : Prop :=
  -1 ≤ m ∧ m ≤ 1

-- Define the property for circles having common points
def circlesHaveCommonPoints (m : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ circle1 m ∧ p ∈ circle2

-- The final statement
theorem necessary_and_sufficient_condition (m : ℝ) :
  condition m → circlesHaveCommonPoints m ↔ (-5 ≤ m ∧ m ≤ 1) :=
by
  sorry

end necessary_and_sufficient_condition_l291_291745


namespace unoccupied_volume_is_correct_l291_291788

-- Definitions based on conditions
def cone_radius : ℝ := 15
def cone_height : ℝ := 15
def sphere_radius : ℝ := 8
def cylinder_radius : ℝ := 15
def cylinder_height : ℝ := 30
def pi := Real.pi

-- Volumes based on geometric formulas
def volume_cylinder : ℝ := pi * (cylinder_radius ^ 2) * cylinder_height
def volume_cone : ℝ := (1/3) * pi * (cone_radius ^ 2) * cone_height
def volume_sphere : ℝ := (4/3) * pi * (sphere_radius ^ 3)

-- The unoccupied volume in the cylinder
noncomputable def unoccupied_volume_cylinder : ℝ :=
  volume_cylinder - (2 * volume_cone) - volume_sphere

-- Theorem statement
theorem unoccupied_volume_is_correct :
  unoccupied_volume_cylinder = 3817.33 * pi := by sorry

end unoccupied_volume_is_correct_l291_291788


namespace edge_of_new_cube_l291_291041

theorem edge_of_new_cube (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) :
  ∃ d : ℝ, d^3 = a^3 + b^3 + c^3 ∧ d = 12 :=
by
  sorry

end edge_of_new_cube_l291_291041


namespace dot_product_result_l291_291223

noncomputable def vector_a := sorry -- Definition of vector a
noncomputable def vector_b := sorry -- Definition of vector b

theorem dot_product_result :
  ∥vector_a∥ = 1 ∧ ∥vector_b∥ = 1 ∧ real.angle vector_a vector_b = real.pi / 3 →
  (vector_a • vector_a + vector_a • vector_b) = 3 / 2 :=
by
  sorry

end dot_product_result_l291_291223


namespace no_continuous_coverage_l291_291448

noncomputable def running_track_problem : Prop :=
  let track_length := 1 -- 1 km
  let stands_arc_length := 0.1 -- 100 meters = 0.1 km
  let runners_speeds := [20, 21, 22, 23, 24, 25, 26, 27, 28, 29] -- km/h
  ∃ (runner_positions : Fin 10 → ℝ), -- Starting positions (unit: km) of 10 runners
    ∀ (t : ℝ), -- For any time (unit: hours)
      ∃ (i : Fin 10), -- There exists a runner (among 10)
        let position := (runner_positions i + runners_speeds.nth i * t) % track_length
        in  position ≤ stands_arc_length

theorem no_continuous_coverage :
  ¬running_track_problem :=
sorry

end no_continuous_coverage_l291_291448


namespace tom_dance_lessons_cost_l291_291373

theorem tom_dance_lessons_cost (total_lessons free_lessons : ℕ) (cost_per_lesson : ℕ) (h1 : total_lessons = 10) (h2 : free_lessons = 2) (h3 : cost_per_lesson = 10) : total_lessons * cost_per_lesson - free_lessons * cost_per_lesson = 80 :=
by
  rw [h1, h2, h3]
  sorry

end tom_dance_lessons_cost_l291_291373


namespace train_speed_l291_291806

def train_length : ℝ := 640
def time_taken : ℝ := 16

theorem train_speed : train_length / time_taken = 40 := by
  have h : train_length / time_taken = 640 / 16 := by sorry
  have h1 : 640 / 16 = 40 := by sorry
  exact eq.trans h h1

end train_speed_l291_291806


namespace quadratic_expression_rewrite_l291_291727

theorem quadratic_expression_rewrite :
  ∃ a b c : ℚ, (∀ k : ℚ, 12 * k^2 + 8 * k - 16 = a * (k + b)^2 + c) ∧ c + 3 * b = -49/3 :=
sorry

end quadratic_expression_rewrite_l291_291727


namespace workerB_time_to_complete_job_l291_291804

theorem workerB_time_to_complete_job 
  (time_A : ℝ) (time_together: ℝ) (time_B : ℝ) 
  (h1 : time_A = 5) 
  (h2 : time_together = 3.333333333333333) 
  (h3 : 1 / time_A + 1 / time_B = 1 / time_together) 
  : time_B = 10 := 
  sorry

end workerB_time_to_complete_job_l291_291804


namespace find_p_l291_291935

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

noncomputable def point_A_on_parabola (x y p : ℝ) : Prop := y^2 = 2 * p * x

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_p (x y p : ℝ) (h_p : p > 0)
  (h_A : point_A_on_parabola x y p)
  (h_dist_to_focus : distance x y (p / 2) 0 = 12)
  (h_dist_to_yaxis : real.abs x = 9) 
  : p = 6 :=
sorry

end find_p_l291_291935


namespace exists_zero_in_interval_l291_291528

def f (x : ℝ) : ℝ := log (x + 1) - 3 / x

theorem exists_zero_in_interval (h1 : f 2 < 0) (h2 : f 3 > 0) : ∃ x ∈ Ioo 2 3, f x = 0 := sorry

end exists_zero_in_interval_l291_291528


namespace divisibility_of_powers_l291_291322

theorem divisibility_of_powers (n : ℤ) : 65 ∣ (7^4 * n - 4^4 * n) :=
by
  sorry

end divisibility_of_powers_l291_291322


namespace transformation_f_g_l291_291351

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2) ^ 2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0 -- Define the function outside the given ranges for completeness

def g (x : ℝ) : ℝ := -f (x - 3)

theorem transformation_f_g : ∀ (x : ℝ), g x = -f (x - 3) := 
by
  intro x
  simp [g, f]
  sorry -- Omit the proof

end transformation_f_g_l291_291351


namespace sum_first_10_terms_l291_291971

noncomputable def a (n : ℕ) := 1 / (4 * (n + 1) ^ 2 - 1)

theorem sum_first_10_terms : (Finset.range 10).sum a = 10 / 21 :=
by
  sorry

end sum_first_10_terms_l291_291971


namespace average_length_of_strings_l291_291782

-- Define lengths of the three strings
def length1 := 4  -- length of the first string in inches
def length2 := 5  -- length of the second string in inches
def length3 := 7  -- length of the third string in inches

-- Define the total length and number of strings
def total_length := length1 + length2 + length3
def num_strings := 3

-- Define the average length calculation
def average_length := total_length / num_strings

-- The proof statement
theorem average_length_of_strings : average_length = 16 / 3 := 
by 
  sorry

end average_length_of_strings_l291_291782


namespace complex_exponential_quadrant_l291_291081

theorem complex_exponential_quadrant :
  ∀ (x : ℝ), x = 2 * real.pi / 3 → 
  let z := complex.exp (complex.I * x) in
  (z.re < 0) ∧ (z.im > 0) :=
begin
  -- proof goes here
  sorry
end

end complex_exponential_quadrant_l291_291081


namespace bisect_B1E_l291_291253

open EuclideanGeometry

noncomputable def midpoint (A : Point) (B : Point) : Point := sorry
noncomputable def circumcircle (A : Point) (B : Point) (C : Point) : Circle := sorry
noncomputable def intersect_again (ω_1 : Circle) (ω_2 : Circle) : Point := sorry
noncomputable def midpoint_of_arc (ω : Circle) (A : Point) (B : Point) : Point := sorry
noncomputable def intersection (ℓ_1 : Line) (ℓ_2 : Line) : Point := sorry

theorem bisect_B1E (A B C M B1 K Q E : Point)
  (hM : M = midpoint A B)
  (hCB : dist C B = dist C B1)
  (hK : K = intersect_again (circumcircle A B C) (circumcircle B M B1))
  (hQ : Q = midpoint_of_arc (circumcircle A B C) A B)
  (hE : E = intersection (line B1 Q) (line B C)) :
  collinear [K, C, E] ∧ bisects (line K C) (segment B1 E) := sorry

end bisect_B1E_l291_291253


namespace parabola_focus_distance_l291_291937

theorem parabola_focus_distance (p : ℝ) (hp : p > 0) (A : ℝ × ℝ)
  (hA_on_parabola : A.2 ^ 2 = 2 * p * A.1)
  (hA_focus_dist : dist A (p / 2, 0) = 12)
  (hA_yaxis_dist : abs A.1 = 9) : p = 6 :=
sorry

end parabola_focus_distance_l291_291937


namespace roots_of_poly_l291_291524

noncomputable def poly (x : ℝ) : ℝ := x^3 - 4 * x^2 - x + 4

theorem roots_of_poly :
  (poly 1 = 0) ∧ (poly (-1) = 0) ∧ (poly 4 = 0) ∧
  (∀ x, poly x = 0 → x = 1 ∨ x = -1 ∨ x = 4) :=
by
  sorry

end roots_of_poly_l291_291524


namespace no_integer_m_l291_291668

theorem no_integer_m (n r m : ℕ) (hn : 1 ≤ n) (hr : 2 ≤ r) : 
  ¬ (∃ m : ℕ, n * (n + 1) * (n + 2) = m ^ r) :=
sorry

end no_integer_m_l291_291668


namespace Jane_stick_length_l291_291718

theorem Jane_stick_length
  (Pat_stick_length : ℕ)
  (dirt_covered_length : ℕ)
  (Sarah_stick_double : ℕ)
  (Jane_stick_diff : ℕ) :
  Pat_stick_length = 30 →
  dirt_covered_length = 7 →
  Sarah_stick_double = 2 →
  Jane_stick_diff = 24 →
  (Pat_stick_length - dirt_covered_length) * Sarah_stick_double - Jane_stick_diff = 22 := 
by
  intros Pat_length dirt_length Sarah_double Jane_diff
  intro h1
  intro h2
  intro h3
  intro h4
  rw [h1, h2, h3, h4]
  sorry

end Jane_stick_length_l291_291718


namespace find_adelka_numbers_l291_291467

def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

def adelka_numbers (a b : ℕ) :=
  let d := gcd a b
  let l := lcm a b
  d < 100 ∧ a < 100 ∧ b < 100 ∧ l < 100 ∧ 
  a ≠ b ∧ a ≠ d ∧ a ≠ l ∧ b ≠ d ∧ b ≠ l ∧ d ≠ l ∧
  l = d * (a / d) * (b / d) ∧ 
  (l / d) = gcd a b

theorem find_adelka_numbers : ∃ (a b : ℕ), adelka_numbers a b ∧ a = 12 ∧ b = 18 :=
by
  sorry

end find_adelka_numbers_l291_291467


namespace square_pattern_1111111_l291_291366

theorem square_pattern_1111111 :
  11^2 = 121 ∧ 111^2 = 12321 ∧ 1111^2 = 1234321 → 1111111^2 = 1234567654321 :=
by
  sorry

end square_pattern_1111111_l291_291366


namespace distance_from_origin_l291_291444

theorem distance_from_origin (x y : ℝ) (n : ℝ)
  (h1 : y = 15)
  (h2 : real.sqrt ((x - 5)^2 + (y - 8)^2) = 13)
  (h3 : x > 5)
  (h4 : n = real.sqrt (x^2 + y^2)) :
  n = real.sqrt (370 + 20 * real.sqrt 30) :=
by sorry

end distance_from_origin_l291_291444


namespace dot_path_length_l291_291833

/- Define the conditions as functions and hypotheses -/
def edge_length := 2 -- edge length of the cube
def radius := edge_length / 2 -- radius is half the edge length
def n_quarter_circles := 4 -- dot traces four quarter-circles

/- Hypothesis that cube rolls and returns dot to top face -/
variable (rolls_to_top : Bool)

/- The goal is to prove the final distance for dot path is dπ where d = 5 -/
theorem dot_path_length (h1 : rolls_to_top = true) : 
  let d := 5 in
  (4 * radius * Real.pi + radius * Real.pi / 2) = d * Real.pi := 
by
  sorry

end dot_path_length_l291_291833


namespace partition_without_single_cells_l291_291548

open Set Int

def initial_grid := (Fin 9) × (Fin 9) -- Representation of a 9x9 grid

def removed_cells : Set (Fin 9 × Fin 9) :=
  { (i, j) | i.val % 2 = 0 ∧ i.val ≠ 0 ∧ j.val % 2 = 0 ∧ j.val ≠ 0 }

def remaining_cells := initial_grid \ removed_cells

theorem partition_without_single_cells :
  ∃ (partition : Set (Set (Fin 9 × Fin 9))),
    (∀ rect ∈ partition, ∃ a b c d : Fin 9, rect = { (i, j) | a ≤ i.val ∧ i.val ≤ b ∧ c ≤ j.val ∧ j.val ≤ d } ∧
    (b.val - a.val + 1) * (d.val - c.val + 1) > 1) ∧
    (∀ cell ∈ remaining_cells, ∃ rect ∈ partition, cell ∈ rect) ∧
    (∀ rect1 rect2 ∈ partition, rect1 ≠ rect2 → disjoint rect1 rect2)
:= sorry

end partition_without_single_cells_l291_291548


namespace problem_solution_l291_291887

noncomputable def problem : ℚ :=
  (Finset.sum (Finset.range 2022) (λ k, (2022 - k : ℚ) / (k + 1))) /
  (Finset.sum (Finset.range 2022) (λ k, 1 / (k + 2))) 

theorem problem_solution : problem = 2023 :=
by {
  sorry -- Proof omitted as per instructions
}

end problem_solution_l291_291887


namespace janes_score_l291_291659

theorem janes_score (jane_score tom_score : ℕ) (h1 : jane_score = tom_score + 50) (h2 : (jane_score + tom_score) / 2 = 90) :
  jane_score = 115 :=
sorry

end janes_score_l291_291659


namespace min_value_of_f_l291_291350

noncomputable def f (a x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

theorem min_value_of_f (a : ℝ) (h1 : ∀x ∈ set.Icc (0:ℝ) (2:ℝ), f a x ≥ 3) (h2 : ∃x ∈ set.Icc (0:ℝ) (2:ℝ), f a x = 3) : 
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 :=
by sorry

end min_value_of_f_l291_291350


namespace area_of_triangle_XYZ_l291_291642

-- Define the right triangle (is it a right triangle)
noncomputable def is_right_triangle (X Y Z : Type*) [metric_space X] (a b c : X) (h: ∠a b c = π / 2) : Prop := sorry

-- Define the isosceles right triangle condition (angles equality)
def is_isosceles_right_triangle (X Y Z : Type*) [metric_space X] (a b c : X) : Prop :=
  is_right_triangle X Y Z a b c ∧ ∠b a c = ∠c a b

-- Define the given right triangle and its properties
noncomputable def is_right_isosceles_triangle_example (X : Type*) [metric_space X] : Prop :=
  let A := point.mk 0 0
  let B := point.mk 8 0
  let C := point.mk 8 (8*sqrt 2)
  is_isosceles_right_triangle X A B C ∧ dist A C = 8*sqrt 2

-- Define and assert the area of the given triangle
theorem area_of_triangle_XYZ (X : Type*) [metric_space X] : 
  is_right_isosceles_triangle_example X → area (triangle.mk A B C) = 32 :=
begin
  sorry
end

end area_of_triangle_XYZ_l291_291642


namespace smallest_k_is_one_l291_291762

noncomputable def smallest_k (S : Finset (Finset ℕ)) (N : ℕ) : ℕ :=
  S.min' (by sorry)

theorem smallest_k_is_one 
  (numbers : Finset ℕ)
  (h1 : ∀ n ∈ numbers, 10000 ≤ n ∧ n < 100000)
  (h2 : ∀ n ∈ numbers, let ds := n.digits 10 in ds = ds.sorted <)
  : smallest_k numbers 13579 = 1 :=
sorry

end smallest_k_is_one_l291_291762


namespace equilateral_triangle_segment_length_l291_291464

theorem equilateral_triangle_segment_length :
  ∀ (A B C D E F G : Type) [IsEquilateralTriangle ABC] (h1 : side_length ABC = 10)
  (h2 : AD = 3) [IsEquilateralTriangle ADE] [IsEquilateralTriangle BDG] [IsEquilateralTriangle CEF],
  (FG = 4) := by
  sorry

end equilateral_triangle_segment_length_l291_291464


namespace expression_undefined_count_l291_291882

theorem expression_undefined_count :
  let denom := (x : ℝ) → (x^2 + 3 * x - 4) * (x - 4)
  (∃ x1 x2 x3 : ℝ, denom x1 = 0 ∧ denom x2 = 0 ∧ denom x3 = 0
    ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  ∃ n : ℕ, n = 3 :=
by
  sorry

end expression_undefined_count_l291_291882


namespace sum_of_roots_cubic_eq_l291_291905

theorem sum_of_roots_cubic_eq (a b c d e f g h : ℝ) :
  sum_roots_cubic (3*x^3 - 6*x^2 - 4*x + 24)
  + sum_roots_cubic (4*x^3 + 8*x^2 - 20*x - 60) = 0 := 
sorry

end sum_of_roots_cubic_eq_l291_291905


namespace cyclic_sum_inequality_l291_291166

variable (a b c : ℝ)
variable (pos_a : a > 0)
variable (pos_b : b > 0)
variable (pos_c : c > 0)

theorem cyclic_sum_inequality :
  ( (a^3 + b^3) / (a^2 + a * b + b^2) + 
    (b^3 + c^3) / (b^2 + b * c + c^2) + 
    (c^3 + a^3) / (c^2 + c * a + a^2) ) ≥ 
  (2 / 3) * (a + b + c) := 
  sorry

end cyclic_sum_inequality_l291_291166


namespace exceeding_speed_limit_percentages_overall_exceeding_speed_limit_percentage_l291_291307

theorem exceeding_speed_limit_percentages
  (percentage_A : ℕ) (percentage_B : ℕ) (percentage_C : ℕ)
  (H_A : percentage_A = 30)
  (H_B : percentage_B = 20)
  (H_C : percentage_C = 25) :
  percentage_A = 30 ∧ percentage_B = 20 ∧ percentage_C = 25 := by
  sorry

theorem overall_exceeding_speed_limit_percentage
  (percentage_A percentage_B percentage_C : ℕ)
  (H_A : percentage_A = 30)
  (H_B : percentage_B = 20)
  (H_C : percentage_C = 25) :
  (percentage_A + percentage_B + percentage_C) / 3 = 25 := by
  sorry

end exceeding_speed_limit_percentages_overall_exceeding_speed_limit_percentage_l291_291307


namespace count_two_digit_numbers_with_digit_8_l291_291991

theorem count_two_digit_numbers_with_digit_8 : 
  let two_digit_integers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let has_eight (n : ℕ) := n / 10 = 8 ∨ n % 10 = 8
  (two_digit_integers.filter has_eight).card = 18 :=
by
  let two_digit_integers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let has_eight (n : ℕ) := n / 10 = 8 ∨ n % 10 = 8
  show (two_digit_integers.filter has_eight).card = 18
  sorry

end count_two_digit_numbers_with_digit_8_l291_291991


namespace problem_statement_l291_291147

def base7_representation (n : ℕ) : ℕ :=
  let rec digits (n : ℕ) (acc : ℕ) (power : ℕ) : ℕ :=
    if n = 0 then acc
    else digits (n / 7) (acc + (n % 7) * power) (power * 10)
  digits n 0 1

def even_digits_count (n : ℕ) : ℕ :=
  let rec count (n : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else let d := n % 10 in
         count (n / 10) (if d % 2 = 0 then acc + 1 else acc)
  count n 0

theorem problem_statement : even_digits_count (base7_representation 528) = 0 := sorry

end problem_statement_l291_291147


namespace shaded_area_ratio_l291_291010

-- Definitions based on conditions
def large_square_area : ℕ := 16
def shaded_components : ℕ := 4
def component_fraction : ℚ := 1 / 2
def shaded_square_area : ℚ := shaded_components * component_fraction
def large_square_area_q : ℚ := large_square_area

-- Goal statement
theorem shaded_area_ratio : (shaded_square_area / large_square_area_q) = (1 / 8) :=
by sorry

end shaded_area_ratio_l291_291010


namespace thabo_books_l291_291740

theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 280) : H = 55 :=
by
  sorry

end thabo_books_l291_291740


namespace paco_initial_cookies_l291_291714

theorem paco_initial_cookies :
  ∀ (total_cookies initially_ate initially_gave : ℕ),
    initially_ate = 14 →
    initially_gave = 13 →
    initially_ate = initially_gave + 1 →
    total_cookies = initially_ate + initially_gave →
    total_cookies = 27 :=
by
  intros total_cookies initially_ate initially_gave h_ate h_gave h_diff h_sum
  sorry

end paco_initial_cookies_l291_291714


namespace tom_pays_l291_291371

-- Definitions based on the conditions
def number_of_lessons : Nat := 10
def cost_per_lesson : Nat := 10
def free_lessons : Nat := 2

-- Desired proof statement
theorem tom_pays {number_of_lessons cost_per_lesson free_lessons : Nat} :
  (number_of_lessons - free_lessons) * cost_per_lesson = 80 :=
by
  sorry

end tom_pays_l291_291371


namespace decreased_cost_l291_291810

theorem decreased_cost (original_cost : ℝ) (decrease_percentage : ℝ) (h1 : original_cost = 200) (h2 : decrease_percentage = 0.50) : 
  (original_cost - original_cost * decrease_percentage) = 100 :=
by
  -- This is the proof placeholder
  sorry

end decreased_cost_l291_291810


namespace find_r_l291_291769

open Real.Matrix

def a : Fin 3 → ℝ := ![2, 3, -1]
def b : Fin 3 → ℝ := ![-1, 1, 0]
def c : Fin 3 → ℝ := ![3, -2, 4]

theorem find_r : 
  ∃ p q r : ℝ, c = p • a + q • b + r • (a ⬝ b) ∧ r = 25 / 27 :=
by
  sorry

end find_r_l291_291769


namespace geometric_series_sum_l291_291504

theorem geometric_series_sum :
  (∑ k in Finset.range 12, (3/4)^(k+1)) = (49340325 / 16777216) :=
by
  sorry

end geometric_series_sum_l291_291504


namespace no_int_solutions_for_exponential_equation_l291_291883

theorem no_int_solutions_for_exponential_equation :
  ¬ ∃ (a b c d : ℤ), 4^a + 5^b = 2^c + 2^d + 1 :=
by
  sorry

end no_int_solutions_for_exponential_equation_l291_291883


namespace num_nonempty_proper_subsets_A_range_of_m_l291_291278

def A (x : ℝ) := (1 / 32 : ℝ) ≤ 2^(-x) ∧ 2^(-x) ≤ 4
def B (x : ℝ) (m : ℝ) := (x - m + 1) * (x - 2 * m - 1) < 0 

theorem num_nonempty_proper_subsets_A :
  ∀ (x : ℝ), A x → x ∈ ℕ → (∃ n : ℕ, n = 62) :=
by
  intros x Ax x_nat
  sorry

theorem range_of_m (A_superset_B : ∀ x, B x m → A x) :
  m = -2 ∨ (-1 ≤ m ∧ m ≤ 2) :=
by
  intros A_superset_B
  sorry

end num_nonempty_proper_subsets_A_range_of_m_l291_291278


namespace program_output_is_21_l291_291115

noncomputable def final_value_of_S : ℕ :=
let i := 1 in
let (final_i, S) := (Nat.iterate (fun (i, S) => (i + 2, 2 * (i + 2) + 3))
                     ((0, 0) : ℕ × ℕ) 4) in S

theorem program_output_is_21 : final_value_of_S = 21 := by
  -- Define the initial state
  let i := 1
  let (init_i, init_S) := (i, 0)
  -- Execute four iterations of the loop
  let (i1, S1) := (init_i + 2, 2 * (init_i + 2) + 3)
  let (i2, S2) := (i1 + 2, 2 * (i1 + 2) + 3)
  let (i3, S3) := (i2 + 2, 2 * (i2 + 2) + 3)
  let (i4, S4) := (i3 + 2, 2 * (i3 + 2) + 3)
  -- i4 will be 9 and S4 should be 21
  have : S4 = 21 := by sorry
  exact this

end program_output_is_21_l291_291115


namespace determine_value_of_y_l291_291368

variable (s y : ℕ)
variable (h_pos : s > 30)
variable (h_eq : s * s = (s - 15) * (s + y))

theorem determine_value_of_y (h_pos : s > 30) (h_eq : s * s = (s - 15) * (s + y)) : 
  y = 15 * s / (s + 15) :=
by
  sorry

end determine_value_of_y_l291_291368


namespace repeating_decimal_computation_l291_291496

noncomputable def x := 864 / 999
noncomputable def y := 579 / 999
noncomputable def z := 135 / 999

theorem repeating_decimal_computation :
  x - y - z = 50 / 333 :=
by
  sorry

end repeating_decimal_computation_l291_291496


namespace number_of_1000_digit_integers_with_odd_and_adjacent_diff_by_2_l291_291597

theorem number_of_1000_digit_integers_with_odd_and_adjacent_diff_by_2 :
  (∑ n in (finset.range 10000).filter (λ n, (n.digits 10).length = 1000 ∧ 
      (∀ i < 999, ((n.digits 10).nth i).get_or_else 0 % 2 = 1) ∧ 
      (∀ i < 999, nat.abs ((n.digits 10).nth i).get_or_else 0 - (n.digits 10).nth (i + 1)).get_or_else 0 = 2)), 1) = 8 * 3^499 :=
by
  sorry

end number_of_1000_digit_integers_with_odd_and_adjacent_diff_by_2_l291_291597


namespace student_passing_percentage_l291_291075

/--
 A student has to obtain a certain percentage of the total marks to pass. Given:
 - The student got 150 marks
 - The student failed by 30 marks
 - The maximum marks are 400
 
 Prove that the student needs to obtain 45% of the total marks to pass.
-/
theorem student_passing_percentage (marks_obtained : ℕ) (marks_failed_by : ℕ) (max_marks : ℕ) :
  marks_obtained = 150 → marks_failed_by = 30 → max_marks = 400 →
  (marks_obtained + marks_failed_by) * 100 / max_marks = 45 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end student_passing_percentage_l291_291075


namespace modulus_of_complex_num_l291_291512

-- Defining the complex number
def complex_num : ℂ := 7 / 4 + 3 * Complex.I

-- The hypothesis (Conditions)
axiom modulus_formula (z : ℂ) : |z| = Real.sqrt (z.re * z.re + z.im * z.im)

-- The theorem to prove
theorem modulus_of_complex_num : |complex_num| = Real.sqrt 193 / 4 := by
  sorry

end modulus_of_complex_num_l291_291512


namespace decimal_expansion_ninthy_ninth_digit_l291_291794

theorem decimal_expansion_ninthy_ninth_digit :
  ∀ (n : ℕ), n = 99 → 
  (fractional_part (2 / 9) + fractional_part (3 / 11)).digit n = 4 := by
  sorry

end decimal_expansion_ninthy_ninth_digit_l291_291794


namespace not_possible_for_runners_in_front_l291_291447

noncomputable def runnerInFrontAtAnyMoment 
  (track_length : ℝ)
  (stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) : Prop := 
  ∀ t : ℝ, ∃ i : Fin 10, 
  ∃ n : ℤ, 
  (runners_speeds i * t - n * track_length) % track_length ≤ stands_length

theorem not_possible_for_runners_in_front 
  (track_length stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) 
  (h_track : track_length = 1)
  (h_stands : stands_length = 0.1)
  (h_speeds : ∀ i : Fin 10, 20 + i = runners_speeds i) : 
  ¬ runnerInFrontAtAnyMoment track_length stands_length runners_speeds :=
sorry

end not_possible_for_runners_in_front_l291_291447


namespace Diane_net_loss_l291_291298

variable (x y a b: ℝ)

axiom h1 : x * a = 65
axiom h2 : y * b = 150

theorem Diane_net_loss : (y * b) - (x * a) = 50 := by
  sorry

end Diane_net_loss_l291_291298


namespace work_completion_time_l291_291688

theorem work_completion_time : 
  let work_total : ℝ := 1 in
  let mahesh_days : ℝ := 20 in
  let rajesh_days : ℝ := 30 in
  let mahesh_rate := 1 / 45 in
  let mahesh_work := mahesh_days * mahesh_rate in
  let remaining_work := work_total - mahesh_work in
  remaining_work / rajesh_days = 1 / 30 →
  mahesh_days + rajesh_days = 50 := 
by
  intros h
  sorry

end work_completion_time_l291_291688


namespace seq_arithmetic_implies_an_arithmetic_l291_291790

open Nat

theorem seq_arithmetic_implies_an_arithmetic 
  (a b : ℕ → ℕ) 
  (h1 : ∀ n : ℕ+, b n = (a 1 + ∑ i in finset.range n, (i + 1) * a (i + 1)) / (∑ i in finset.range n, i + 1)) 
  (h2 : ∀ n : ℕ+, b (n + 1) - b n = b 2 - b 1) : 
  ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1 := 
sorry

end seq_arithmetic_implies_an_arithmetic_l291_291790


namespace S_inequality_l291_291172

-- Define the sequence a_n
def sequence_a (n : ℕ) (hn : n > 0) : ℚ := (n : ℚ) / ((n + 1) : ℚ)

-- Define the transformed sequence sequence_b
def sequence_b (n : ℕ) (hn : n > 0) : ℚ := 1 / (1 - (sequence_a n hn))

-- Define T_n
def T (n : ℕ) (hn : n > 0) : ℚ :=
  if n = 1 then 1 else 
  (finite_range_product (fun k => sequence_a k (nat.succ_pos k)) (n-1))

-- Define S_n
def S (n : ℕ) : ℚ := (∑ k in finset.range n, T (k + 1) (nat.succ_pos k))

-- Main theorem to prove the inequality
theorem S_inequality (n : ℕ) (hn : n > 0) : 
  1 / 2 ≤ S (2 * n) - S n ∧ S (2 * n) - S n < 3 / 4 :=
sorry 

end S_inequality_l291_291172


namespace final_sale_price_is_correct_l291_291836

-- Define the required conditions
def original_price : ℝ := 1200.00
def first_discount_rate : ℝ := 0.10
def second_discount_rate : ℝ := 0.20
def final_discount_rate : ℝ := 0.05

-- Define the expression to calculate the sale price after the discounts
def first_discount_price := original_price * (1 - first_discount_rate)
def second_discount_price := first_discount_price * (1 - second_discount_rate)
def final_sale_price := second_discount_price * (1 - final_discount_rate)

-- Prove that the final sale price equals $820.80
theorem final_sale_price_is_correct : final_sale_price = 820.80 := by
  sorry

end final_sale_price_is_correct_l291_291836


namespace element_in_subset_A_l291_291102

theorem element_in_subset_A (n : ℕ) (E : Finset ℕ) (c : ℕ) (A : Finset ℕ) (k : ℕ) (hk : 2 ^ k ∣ c) :
  E = Finset.range (2 * n + 1) →
  c ∈ E →
  (∀ {x y : ℕ}, x ∈ A → y ∈ A → x ≠ y → ¬ (x ∣ y) → ¬ (y ∣ x)) →
  (∀ {x y : ℕ}, x ∈ A → y ∈ A → x ≠ y → ¬ (x ∣ y)) ↔
  c > n * (2 / 3)^(k + 1) := 
by
  intro hE hcinE hA_div
  sorry

end element_in_subset_A_l291_291102


namespace parallelogram_base_length_l291_291418

variable (base height : ℝ)
variable (Area : ℝ)

theorem parallelogram_base_length (h₁ : Area = 162) (h₂ : height = 2 * base) (h₃ : Area = base * height) : base = 9 := 
by
  sorry

end parallelogram_base_length_l291_291418


namespace fruit_boxes_needed_l291_291778

noncomputable def fruit_boxes : ℕ × ℕ × ℕ :=
  let baskets : ℕ := 7
  let peaches_per_basket : ℕ := 23
  let apples_per_basket : ℕ := 19
  let oranges_per_basket : ℕ := 31
  let peaches_eaten : ℕ := 7
  let apples_eaten : ℕ := 5
  let oranges_eaten : ℕ := 3
  let peaches_box_size : ℕ := 13
  let apples_box_size : ℕ := 11
  let oranges_box_size : ℕ := 17

  let total_peaches := baskets * peaches_per_basket
  let total_apples := baskets * apples_per_basket
  let total_oranges := baskets * oranges_per_basket

  let remaining_peaches := total_peaches - peaches_eaten
  let remaining_apples := total_apples - apples_eaten
  let remaining_oranges := total_oranges - oranges_eaten

  let peaches_boxes := (remaining_peaches + peaches_box_size - 1) / peaches_box_size
  let apples_boxes := (remaining_apples + apples_box_size - 1) / apples_box_size
  let oranges_boxes := (remaining_oranges + oranges_box_size - 1) / oranges_box_size

  (peaches_boxes, apples_boxes, oranges_boxes)

theorem fruit_boxes_needed :
  fruit_boxes = (12, 12, 13) := by 
  sorry

end fruit_boxes_needed_l291_291778


namespace number_of_even_digits_in_base7_of_528_l291_291141

/-
  Define the base-7 representation of a number and a predicate to count even digits.
-/

-- Definition of base-7 digit representation
def base7_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else (List.unfoldr (λ n, if n = 0 then Option.none else some (n % 7, n / 7)) n).reverse

-- Predicate to check if a digit is even
def is_even (d : ℕ) : Bool := d % 2 = 0

-- Counting the even digits in base-7 representation
def count_even_digits_in_base7 (n : ℕ) : ℕ :=
  (base7_repr n).countp is_even

-- The target theorem to prove
theorem number_of_even_digits_in_base7_of_528 : count_even_digits_in_base7 528 = 0 :=
by sorry

end number_of_even_digits_in_base7_of_528_l291_291141


namespace sine_increasing_interval_l291_291865

theorem sine_increasing_interval :
  ∃ (I : set ℝ), I = [- (Real.pi / 2), (Real.pi / 2)] ∧ 
                 (∀ x y ∈ I, x < y → Real.sin x < Real.sin y) := by
  sorry

end sine_increasing_interval_l291_291865


namespace exists_equilateral_triangle_l291_291276

theorem exists_equilateral_triangle (A B C M N : Point)
  (hABC_acute : ∀ X, X ∈ (triangle A B C).angles → X < π / 2)
  (hAngle_A : ∠ A = 60°)
  (hBisector_B : M ∈ line_through B (line_segment A C) ∧ is_angle_bisector B A C M)
  (hBisector_C : N ∈ line_through C (line_segment A B) ∧ is_angle_bisector C A B N) :
  ∃ P : Point, P ∈ line_segment B C ∧ equilateral_triangle M N P :=
sorry

end exists_equilateral_triangle_l291_291276


namespace rock_height_at_30_l291_291848

theorem rock_height_at_30 (t : ℝ) (h : ℝ) 
  (h_eq : h = 80 - 9 * t - 5 * t^2) 
  (h_30 : h = 30) : 
  t = 2.3874 :=
by
  -- Proof omitted
  sorry

end rock_height_at_30_l291_291848


namespace nonagon_arithmetic_mean_property_l291_291089

open Real

theorem nonagon_arithmetic_mean_property :
  ∃ (f : Fin 9 → ℝ), 
  (∀ i, f i = 2016 + i) ∧ (
    ∀ i j k : Fin 9, 
    (j = (i + 3) % 9.toFin) ∧ (k = (i + 6) % 9.toFin) →
    f j = (f i + f k) / 2
  ) := 
sorry

end nonagon_arithmetic_mean_property_l291_291089


namespace spy_undetectable_probability_l291_291851

-- Definitions based on conditions
def forest_size : ℝ := 10
def rdf_radius : ℝ := 10
def operational_rdfs : ℕ := 3

-- Probability calculation (we assert the result based on our given conditions)
theorem spy_undetectable_probability : 
  (operational_rdfs = 3) → 
  (forest_size = 10) → 
  (rdf_radius = 10) → 
  ∃ p : ℝ, p = 0.087 :=
by
  intros
  use 0.087
  sorry

end spy_undetectable_probability_l291_291851


namespace find_n_l291_291113

theorem find_n (n a b : ℕ) (h1 : n ≥ 2)
  (h2 : n = a^2 + b^2)
  (h3 : a = Nat.minFac n)
  (h4 : b ∣ n) : n = 8 ∨ n = 20 := 
sorry

end find_n_l291_291113


namespace first_chapter_pages_calculation_l291_291428

-- Define the constants and conditions
def second_chapter_pages : ℕ := 11
def first_chapter_pages_more : ℕ := 37

-- Main proof problem
theorem first_chapter_pages_calculation : first_chapter_pages_more + second_chapter_pages = 48 := by
  sorry

end first_chapter_pages_calculation_l291_291428


namespace train_speed_is_correct_l291_291412

-- Define the conditions as given in the problem
def train_length : ℝ := 200  -- The length of the train in meters
def crossing_time : ℝ := 9   -- The time to cross the man in seconds

-- Define the speed function
def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- The theorem stating the speed of the train
theorem train_speed_is_correct :
  speed train_length crossing_time = 22.22 :=
begin
  -- Simplify the speed expression
  dsimp [speed, train_length, crossing_time],
  -- Perform the exact float division
  have h : 200 / 9 = 22.222222222222223, by norm_num,
  have rounded_val : Float.round (200 / 9 * 100) / 100 = 22.22, by norm_num,
  rw rounded_val,
  simp [h],
  sorry  -- Skipping the rigorous proof steps
end

end train_speed_is_correct_l291_291412


namespace proof_problem_l291_291554

variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem proof_problem 
  (h0 : f a b c 0 = f a b c 4)
  (h1 : f a b c 0 > f a b c 1) : 
  a > 0 ∧ 4 * a + b = 0 :=
by
  sorry

end proof_problem_l291_291554


namespace area_of_triangles_l291_291258

-- Definitions from the conditions
variables (A B C D E : Type)
variables [rect : IsRectangle A B C D]
variables (hAD_perp_DC : ∀ (A B C D : Type), Perpendicular (AD : LineSegment A D) (DC : LineSegment D C))
variables (hAD_AB : AD.length = AB.length)
variables (hAD_4 : AD.length = 4)
variables (hDC_8 : DC.length = 8)
variables (hBE_parallel_AD : ∀ (E : Type), Parallel (BE : LineSegment B E) (AD : LineSegment A D))

-- The theorem statement
theorem area_of_triangles (hE_on_DC : lies_on E (DC : LineSegment D C)) :
  area (triangle A B E) = 8 ∧ area (triangle D E C) = 8 :=
by sorry

end area_of_triangles_l291_291258


namespace joan_total_seashells_l291_291273

def seashells_given_to_Sam : ℕ := 43
def seashells_left_with_Joan : ℕ := 27
def total_seashells_found := seashells_given_to_Sam + seashells_left_with_Joan

theorem joan_total_seashells : total_seashells_found = 70 := by
  -- proof goes here, but for now we will use sorry
  sorry

end joan_total_seashells_l291_291273


namespace thomas_task_completion_l291_291835

theorem thomas_task_completion :
  (∃ T E : ℝ, (1 / T + 1 / E = 1 / 8) ∧ (13 / T + 6 / E = 1)) →
  ∃ T : ℝ, T = 14 :=
by
  sorry

end thomas_task_completion_l291_291835


namespace find_a_and_b_l291_291208

-- Given conditions
def f (a b x : ℝ) : ℝ := a * Real.log x - (2 * b) / x

-- We need the values of a and b
theorem find_a_and_b (a b : ℝ) (h₁ : f a b 1 = 1) (h₂ : deriv (f a b) 1 = 0) :
  a = 1 ∧ b = -1 / 2 :=
sorry

end find_a_and_b_l291_291208


namespace question1_inequality_solution_question2_bound_l291_291961

noncomputable def f (a x : ℝ) := a * x^2 + x - a

-- Question 1
theorem question1_inequality_solution (a : ℝ) (x : ℝ) (hx1 : x ∈ set.Icc (-1:ℝ) 1)
  (h : f a 0 = f a 1) : abs (f a x - 1) < a * x + 3 / 4 ↔ x ∈ set.Ico (-1:ℝ) (1 / 2) := 
sorry

-- Question 2
theorem question2_bound (a x : ℝ) (hx1 : x ∈ set.Icc (-1:ℝ) 1) (ha : abs a ≤ 1) :
  abs (f a x) ≤ 5 / 4 :=
sorry

end question1_inequality_solution_question2_bound_l291_291961


namespace garden_width_l291_291872

variable (W : ℝ) (L : ℝ := 225) (small_gate : ℝ := 3) (large_gate: ℝ := 10) (total_fencing : ℝ := 687)

theorem garden_width :
  2 * L + 2 * W - (small_gate + large_gate) = total_fencing → W = 125 := 
by
  sorry

end garden_width_l291_291872


namespace solve_for_x_l291_291325

theorem solve_for_x (x : ℝ) (h : sqrt (3 / x + 3) = 5 / 3) : x = -27 / 2 :=
sorry

end solve_for_x_l291_291325


namespace find_t_l291_291975

variable (t : ℝ)
def a := (1, t)
def b := (-2, 1)
def orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_t (h : orthogonal ((2 * a.1 - b.1, 2 * a.2 - b.2)) b) : t = 9 / 2 :=
sorry

end find_t_l291_291975


namespace cost_per_tissue_box_l291_291475

-- Given conditions
def rolls_toilet_paper : ℝ := 10
def cost_per_toilet_paper : ℝ := 1.5
def rolls_paper_towels : ℝ := 7
def cost_per_paper_towel : ℝ := 2
def boxes_tissues : ℝ := 3
def total_cost : ℝ := 35

-- Deduction of individual costs
def cost_toilet_paper := rolls_toilet_paper * cost_per_toilet_paper
def cost_paper_towels := rolls_paper_towels * cost_per_paper_towel
def cost_tissues := total_cost - cost_toilet_paper - cost_paper_towels

-- Prove the cost for one box of tissues
theorem cost_per_tissue_box : (cost_tissues / boxes_tissues) = 2 :=
by
  sorry

end cost_per_tissue_box_l291_291475


namespace f_increasing_and_odd_l291_291354

noncomputable def f : ℝ → ℝ :=
λ x, if x < 0 then log (1 / (1 - x)) else log (1 + x)

theorem f_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end f_increasing_and_odd_l291_291354


namespace melted_ice_cream_depth_l291_291069

theorem melted_ice_cream_depth :
  let r_sphere := 3
  let r_cylinder := 12
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2
  ∃ h : Real, V_sphere = V_cylinder * h ∧ h = 1 / 4 :=
begin
  sorry
end

end melted_ice_cream_depth_l291_291069


namespace indefinite_integral_correct_l291_291812

noncomputable def integrand (x : ℝ) : ℝ := 
  (-3 * x^3 + 13 * x^2 - 13 * x + 1) / ((x - 2)^2 * (x^2 - x + 1))

noncomputable def answer (x : ℝ) (C : ℝ) : ℝ := 
  -1 / (x - 2) - (3 / 2) * real.log (abs (x^2 - x + 1)) - (real.sqrt 3) * real.arctan ((2 * x - 1) / (real.sqrt 3)) + C

theorem indefinite_integral_correct (C : ℝ) : 
  ∃ f : ℝ → ℝ, ∀ x : ℝ, has_deriv_at f (integrand x) x ∧ f x = answer x C := 
sorry

end indefinite_integral_correct_l291_291812


namespace jack_black_balloons_l291_291303

def nancy_balloons := 7
def mary_balloons := 4 * nancy_balloons
def total_mary_nancy_balloons := nancy_balloons + mary_balloons
def jack_balloons := total_mary_nancy_balloons + 3

theorem jack_black_balloons : jack_balloons = 38 := by
  -- proof goes here
  sorry

end jack_black_balloons_l291_291303


namespace line_segment_parameters_l291_291353

theorem line_segment_parameters :
  (∃ (a b c d : ℝ), 
    (∀ t : ℝ, -1 ≤ t ∧ t ≤ 1 → ((at + b, ct + d) = if t = -1 then (-4, 10) else if t = 1 then (2, -3) else (-4 + t * 6, 10 - t * 13))) ∧ 
    (a^2 + b^2 + c^2 + d^2 = 321)) := by {
  use [6, -4, -13, 10],
  split,
  use [λ t ht, if t=h1 then (-4, 10) else if t=h2 then (2, -3) else sorry],
  sorry 
}

end line_segment_parameters_l291_291353


namespace sum_of_solutions_eq_l291_291329

open Real

noncomputable def solve_eq (x : ℝ) : Prop :=
  15 / (x * (cbrt (35 - 8 * x^3))) = 2 * x + cbrt (35 - 8 * x^3)

theorem sum_of_solutions_eq : ∑ (x : ℝ) in {x : ℝ | solve_eq x}, x = 2.5 := sorry

end sum_of_solutions_eq_l291_291329


namespace number_of_real_solutions_is_15_l291_291876

theorem number_of_real_solutions_is_15 :
  let equations (u v s t : ℝ) :=
    (u = s + t + s * u * t) ∧
    (v = t + u + t * u * v) ∧
    (s = u + v + u * v * s) ∧
    (t = v + s + v * s * t)
  in
  ∃ u v s t, equations u v s t → 15 :=
begin
  sorry
end

end number_of_real_solutions_is_15_l291_291876


namespace garden_area_increase_l291_291822

-- Define the dimensions and perimeter of the rectangular garden
def length_rect : ℕ := 30
def width_rect : ℕ := 12
def area_rect : ℕ := length_rect * width_rect

def perimeter_rect : ℕ := 2 * (length_rect + width_rect)

-- Define the side length and area of the new square garden
def side_square : ℕ := perimeter_rect / 4
def area_square : ℕ := side_square * side_square

-- Define the increase in area
def increase_in_area : ℕ := area_square - area_rect

-- Prove the increase in area is 81 square feet
theorem garden_area_increase : increase_in_area = 81 := by
  sorry

end garden_area_increase_l291_291822


namespace cow_feed_problem_l291_291832

theorem cow_feed_problem 
    (feed_per_day : ℕ)
    (total_feed : ℕ)
    (days : ℕ)
    (leftover_feed : ℕ)
    (h1 : feed_per_day = 28)
    (h2 : total_feed = 890)
    (h3 : days = total_feed / feed_per_day)
    (h4 : leftover_feed = total_feed % feed_per_day) :
    days = 31 ∧ leftover_feed = 22 := 
by
  split
  · sorry
  · sorry

end cow_feed_problem_l291_291832


namespace least_k_for_divisibility_l291_291550

-- Given conditions and goal statement
theorem least_k_for_divisibility (k : ℕ) :
  (k <= 1004 -> ∃ a b : ℕ, a < b ∧ b ≤ 2005 ∧ ∃ m n : ℕ, 1 ≤ m ∧ 1 ≤ n ∧ m ≠ n ∧ (a, b ∈ s ∧ (m*2^a = n*2^b))) :=
by sorry

end least_k_for_divisibility_l291_291550


namespace sum_of_sequence_l291_291170

noncomputable def curve (x : ℝ) (n : ℕ) : ℝ := x^n * (1 - x)

noncomputable def a_n (n : ℕ) : ℝ := 
  let k := n * 2^(n-1) - (n + 1) * 2^n
  -(k * 2) + 2^n -- y at x = 2 with tangent line

noncomputable def b_n (n : ℕ) : ℝ := a_n(n) / (n + 1)

def sum_sequence (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b_n (k + 1))

theorem sum_of_sequence (n : ℕ) (hn : n > 0) : sum_sequence n = 2^(n+1) - 2 := by
  sorry

end sum_of_sequence_l291_291170


namespace charles_bananas_loss_indeterminate_l291_291409

theorem charles_bananas_loss_indeterminate (willie_initial_bananas : ℕ) (charles_initial_bananas : ℕ) (willie_final_bananas : ℕ) :
  willie_initial_bananas = 48 → charles_initial_bananas = 14 → willie_final_bananas = 13 →
  ∃ (charles_lost_bananas : ℕ), charles_lost_bananas ≠ charles_initial_bananas - charles_final_bananas :=
by
  assume h1 : willie_initial_bananas = 48,
  assume h2 : charles_initial_bananas = 14,
  assume h3 : willie_final_bananas = 13,
  -- Cannot determine exact number of bananas Charles lost from the provided information.
  sorry

end charles_bananas_loss_indeterminate_l291_291409


namespace length_AB_l291_291585

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 + Real.cos θ, Real.sin θ)

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + 3 * t, 2 - 4 * t)

theorem length_AB :
  let curve_eq := λ x y, (x - 2)^2 + y^2 = 1 in
  let line_x := λ t, 1 + 3 * t in
  let line_y := λ t, 2 - 4 * t in
  ∃ t₁ t₂ : ℝ, curve_eq (line_x t₁) (line_y t₁) ∧ curve_eq (line_x t₂) (line_y t₂) ∧
  (∃ A B : ℝ × ℝ, A = (line_x t₁, line_y t₁) ∧ B = (line_x t₂, line_y t₂) ∧ 
  Real.sqrt ((line_x t₂ - line_x t₁)^2 + (line_y t₂ - line_y t₁)^2) = 5 * Real.sqrt((t₁ - t₂)^2) / 5 →
  abs (t₁ - t₂) = (2 * Real.sqrt 21) / 5 →
  Real.sqrt(3^2 + 4^2) * abs (t₁ - t₂) = 2 * Real.sqrt 21 / 5) :=
begin
  sorry
end

end length_AB_l291_291585


namespace distance_from_origin_P_l291_291261

-- Cartesian point definition
def point : Type := ℝ × ℝ

-- The distance function in the Cartesian coordinate system
def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Specific points given in the problem
def origin : point := (0, 0)
def P : point := (3, -4)

-- Theorem stating the goal
theorem distance_from_origin_P : distance origin P = 5 :=
by 
  sorry

end distance_from_origin_P_l291_291261


namespace sum_f_1_to_2013_l291_291877

noncomputable def f (x : ℝ) : ℝ := 
  if -3 ≤ x ∧ x < -1 then -(x + 2)^2
  else if -1 ≤ x ∧ x < 3 then x
  else f (x - 6)

theorem sum_f_1_to_2013 : (∑ k in finset.range 2013, f (k+1)) = 337 := sorry

end sum_f_1_to_2013_l291_291877


namespace sequence_contains_all_integers_l291_291751

theorem sequence_contains_all_integers (a : ℕ → ℕ) 
  (h1 : ∀ i ≥ 0, 0 ≤ a i ∧ a i ≤ i)
  (h2 : ∀ k ≥ 0, (∑ i in Finset.range (k + 1), nat.choose k (a i)) = 2^k) :
  ∀ N ≥ 0, ∃ i ≥ 0, a i = N := 
sorry

end sequence_contains_all_integers_l291_291751


namespace chord_length_eq_l291_291098

theorem chord_length_eq {C_a C_b C_c : Type} [MetricSpace C_a] [MetricSpace C_b] [MetricSpace C_c]
  (O_a O_b O_c : Type)  -- centers of the circles
  (r_a r_b r_c : ℝ)  -- radii of the circles
  (h1 : r_a = 5) 
  (h2 : r_b = 12)
  (h3 : r_c = r_a + r_b)  -- radii conditions
  (collinear_centers : collinear O_a O_b O_c)  -- centers of the circles are collinear
  (externally_tangent : externally_tangent C_a C_b)  -- C_a and C_b are externally tangent
  (internally_tangent_to_C_c : internally_tangent C_a C_c ∧ internally_tangent C_b C_c)  -- C_a and C_b are internally tangent to C_c
  (chord_length : ∃ (AB : ℝ), is_chord_of_tangent C_a C_b C_c AB) -- chord AB is also a common external tangent
  : is_chord_length AB (20 * (real.sqrt 47) / 7) := 
sorry

end chord_length_eq_l291_291098


namespace cost_ratio_l291_291159

theorem cost_ratio (S J M : ℝ) (h1 : S = 4) (h2 : M = 0.75 * (S + J)) (h3 : S + J + M = 21) : J / S = 2 :=
by
  sorry

end cost_ratio_l291_291159


namespace problem1_problem2_l291_291816

-- Problem 1
theorem problem1 (f : ℝ → ℝ) (h : ∀ x ≥ 0, f (√x + 1) = x + 2 * √x) : ∀ x ≥ 1, f x = x^2 - 2 * x :=
sorry

-- Problem 2
theorem problem2 (f : ℝ → ℝ) (h1 : ∃ k b : ℝ, ∀ x, f x = k * x + b) (h2 : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) : ∀ x, f x = 2 * x + 7 :=
sorry

end problem1_problem2_l291_291816


namespace M_has_4_subsets_implies_m_eq_2_l291_291219

theorem M_has_4_subsets_implies_m_eq_2
  (M : Set ℕ)
  (h1 : M = {x : ℤ | 1 ≤ x ∧ x ≤ 2}) :
  (∃ m : ℤ, (M = {x : ℤ | 1 ≤ x ∧ x ≤ m}) ∧ 2^(m - 1 + 1) = 4) → m = 2 :=
by
  sorry

end M_has_4_subsets_implies_m_eq_2_l291_291219


namespace product_inequality_l291_291320

theorem product_inequality (n : ℕ) (hn : n > 0) :
  (∏ k in finset.range n, (2 * k + 1 : ℝ) / (2 * (k + 1) : ℝ)) < 1 / real.sqrt (2 * n + 1) :=
sorry

end product_inequality_l291_291320


namespace find_AX_length_l291_291655

noncomputable def AX_length (AC BC BX : ℕ) : ℚ :=
AC * (BX / BC)

theorem find_AX_length :
  let AC := 25
  let BC := 35
  let BX := 30
  AX_length AC BC BX = 150 / 7 :=
by
  -- proof is omitted using 'sorry'
  sorry

end find_AX_length_l291_291655


namespace smallest_positive_integer_to_subtract_l291_291012

theorem smallest_positive_integer_to_subtract : ∃ (k : ℕ), k > 0 ∧ (425 - k) % 5 = 0 ∧ ∀ (m : ℕ), (m > 0 ∧ (425 - m) % 5 = 0) → m ≥ k :=
begin
  -- Placeholder for the proof
  sorry
end

end smallest_positive_integer_to_subtract_l291_291012


namespace reflection_sum_coordinates_l291_291313

theorem reflection_sum_coordinates :
  ∀ (C D : ℝ × ℝ), 
  C = (5, -3) →
  D = (5, -C.2) →
  (C.1 + C.2 + D.1 + D.2 = 10) :=
by
  intros C D hC hD
  rw [hC, hD]
  simp
  sorry

end reflection_sum_coordinates_l291_291313


namespace length_of_unfenced_side_l291_291547

theorem length_of_unfenced_side :
  ∃ L W : ℝ, L * W = 320 ∧ 2 * W + L = 56 ∧ L = 40 :=
by
  sorry

end length_of_unfenced_side_l291_291547


namespace problem_statement_l291_291149

def base7_representation (n : ℕ) : ℕ :=
  let rec digits (n : ℕ) (acc : ℕ) (power : ℕ) : ℕ :=
    if n = 0 then acc
    else digits (n / 7) (acc + (n % 7) * power) (power * 10)
  digits n 0 1

def even_digits_count (n : ℕ) : ℕ :=
  let rec count (n : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else let d := n % 10 in
         count (n / 10) (if d % 2 = 0 then acc + 1 else acc)
  count n 0

theorem problem_statement : even_digits_count (base7_representation 528) = 0 := sorry

end problem_statement_l291_291149


namespace MrLiBoardsUpperClassBus_l291_291252

-- There are three classes of buses: upper, middle, and lower
inductive BusClass
| upper
| middle
| lower

-- Enumerate all possible departure sequences
def possibleSequences : List (BusClass × BusClass × BusClass) :=
  [ (BusClass.lower, BusClass.middle, BusClass.upper),
    (BusClass.lower, BusClass.upper, BusClass.middle),
    (BusClass.middle, BusClass.lower, BusClass.upper),
    (BusClass.middle, BusClass.upper, BusClass.lower),
    (BusClass.upper, BusClass.lower, BusClass.middle),
    (BusClass.upper, BusClass.middle, BusClass.lower) ]

-- Define a function to check if Mr. Li boards an upper-class bus according to his strategy
def boardsUpperClass (seq : BusClass × BusClass × BusClass) : Bool :=
  match seq with
  | (_, BusClass.upper, _) => true
  -- If the second bus is not upper, check if third is upper
  | (first, second, third) =>
    if second > first then second = BusClass.upper else third = BusClass.upper

-- Calculate the number of sequences where Mr. Li boards an upper-class bus
def favorableSequences : List (BusClass × BusClass × BusClass) :=
  possibleSequences.filter boardsUpperClass

-- Assertion about the probability
theorem MrLiBoardsUpperClassBus :
  let totalSequences := possibleSequences.length
  let favorableCount := favorableSequences.length
  (favorableCount : ℚ) / (totalSequences : ℚ) = 1 / 2 := by
  sorry

end MrLiBoardsUpperClassBus_l291_291252


namespace initial_oranges_l291_291072

theorem initial_oranges (X : ℕ) (h1 : X - 37 + 7 = 10) : X = 40 :=
by
  sorry

end initial_oranges_l291_291072


namespace tim_paid_correct_amount_l291_291367

-- Define the conditions given in the problem
def mri_cost : ℝ := 1200
def doctor_hourly_rate : ℝ := 300
def doctor_time_hours : ℝ := 0.5 -- 30 minutes is half an hour
def fee_for_being_seen : ℝ := 150
def insurance_coverage_rate : ℝ := 0.80

-- Total amount Tim paid calculation
def total_cost_before_insurance : ℝ :=
  mri_cost + (doctor_hourly_rate * doctor_time_hours) + fee_for_being_seen

def insurance_coverage : ℝ :=
  total_cost_before_insurance * insurance_coverage_rate

def amount_tim_paid : ℝ :=
  total_cost_before_insurance - insurance_coverage

-- Prove that Tim paid $300
theorem tim_paid_correct_amount : amount_tim_paid = 300 :=
by
  sorry

end tim_paid_correct_amount_l291_291367


namespace number_of_minivans_filled_up_l291_291635

-- Define the given problem and conditions
def service_cost_per_vehicle := 2.10
def cost_per_liter := 0.70
def num_trucks := 2
def minivan_tank_capacity := 65
def truck_tank_increase_percent := 1.2
def total_cost := 347.20
def truck_tank_capacity := minivan_tank_capacity + truck_tank_increase_percent * minivan_tank_capacity

def minivan_cost := service_cost_per_vehicle + minivan_tank_capacity * cost_per_liter
def truck_cost := service_cost_per_vehicle + truck_tank_capacity * cost_per_liter

-- Target: Prove that the number of mini-vans filled up (m) is 3
theorem number_of_minivans_filled_up : 
  ∃ m : ℕ, m * minivan_cost + num_trucks * truck_cost = total_cost ∧ m = 3 :=
by
  sorry

end number_of_minivans_filled_up_l291_291635


namespace circle_chord_BC_length_l291_291250

/-- Given: AD is a diameter of the circle with center O,
    BO = 5,
    ∠ ABO = 60 degrees,
    ⌒ CD = 60 degrees.
    Prove: BC = 5. -/
theorem circle_chord_BC_length
  (O A B C D : ℝ)
  (hO_center : circle_center O)
  (hAD_diameter : diameter AD O)
  (hBO_len : BO = 5)
  (h_∠ABO_60 : ∠ ABO = 60)
  (h_frown_CD_60 : ⌒ CD = 60) :
  BC = 5 := 
sorry

end circle_chord_BC_length_l291_291250


namespace product_of_100_divisors_l291_291842

theorem product_of_100_divisors (A : ℕ) (h : nat.factors A.length = 100) : 
  (∏ d in (nat.divisors A), d) = A^50 := 
sorry

end product_of_100_divisors_l291_291842


namespace union_of_sets_l291_291972

def A : set ℝ := { x : ℝ | log x ≤ 0 }
def B : set ℝ := { x : ℝ | (2:ℝ)^x ≤ 1 }

theorem union_of_sets :
  A ∪ B = { x : ℝ | x ≤ 1 } := by
  sorry

end union_of_sets_l291_291972


namespace num_different_numerators_S_l291_291673

noncomputable def euler_totient (n : ℕ) : ℕ :=
nat.totient n

theorem num_different_numerators_S' : 
  let S' := {r : ℚ // 0 < r ∧ r < 1 ∧ ∃ a b : ℕ, (0 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ r = (10 * a + b) / 99} in
  (∀ x ∈ S', x.denom) = euler_totient 99 :=
by sorry

end num_different_numerators_S_l291_291673


namespace equilateral_triangle_segment_length_l291_291465

theorem equilateral_triangle_segment_length :
  ∀ (A B C D E F G : Type) [IsEquilateralTriangle ABC] (h1 : side_length ABC = 10)
  (h2 : AD = 3) [IsEquilateralTriangle ADE] [IsEquilateralTriangle BDG] [IsEquilateralTriangle CEF],
  (FG = 4) := by
  sorry

end equilateral_triangle_segment_length_l291_291465


namespace largest_divisor_same_remainder_l291_291007

theorem largest_divisor_same_remainder 
  (d : ℕ) (r : ℕ)
  (a b c : ℕ) 
  (h13511 : 13511 = a * d + r) 
  (h13903 : 13903 = b * d + r)
  (h14589 : 14589 = c * d + r) :
  d = 98 :=
by 
  sorry

end largest_divisor_same_remainder_l291_291007


namespace symmetric_histogram_height_l291_291845

-- Define the conditions and the problem statement
def height_of_symmetric_histogram (sticks : ℕ) : ℕ := 
  let eq := 2 * (fun n : ℕ, n * (n + 1)) in
  if h : ∃ n, eq n = sticks then 
    Nat.find h 
  else 
    0

-- The statement we need to prove
theorem symmetric_histogram_height : height_of_symmetric_histogram 130 = 7 := by
  sorry

end symmetric_histogram_height_l291_291845


namespace intelligent_robot_packing_rate_l291_291843

theorem intelligent_robot_packing_rate :
  (∀ (x : ℕ), 4 * (1600 / x) - 1600 / (5 * x) = 4 → 5 * x = 100) :=
begin
  intro x,
  intros h,
  sorry
end

end intelligent_robot_packing_rate_l291_291843


namespace pair_with_gcf_20_l291_291023

theorem pair_with_gcf_20 (a b : ℕ) (h1 : a = 20) (h2 : b = 40) : Nat.gcd a b = 20 := by
  rw [h1, h2]
  sorry

end pair_with_gcf_20_l291_291023


namespace integer_solutions_inequality_system_l291_291736

noncomputable def check_inequality_system (x : ℤ) : Prop :=
  (3 * x + 1 < x - 3) ∧ ((1 + x) / 2 ≤ (1 + 2 * x) / 3 + 1)

theorem integer_solutions_inequality_system :
  {x : ℤ | check_inequality_system x} = {-5, -4, -3} :=
by
  sorry

end integer_solutions_inequality_system_l291_291736


namespace prob_green_ball_l291_291106

-- Definitions for the conditions
def red_balls_X := 3
def green_balls_X := 7
def total_balls_X := red_balls_X + green_balls_X

def red_balls_YZ := 7
def green_balls_YZ := 3
def total_balls_YZ := red_balls_YZ + green_balls_YZ

-- The probability of selecting any container
def prob_select_container := 1 / 3

-- The probabilities of drawing a green ball from each container
def prob_green_given_X := green_balls_X / total_balls_X
def prob_green_given_YZ := green_balls_YZ / total_balls_YZ

-- The combined probability of selecting a green ball
theorem prob_green_ball : 
  prob_select_container * prob_green_given_X + 
  prob_select_container * prob_green_given_YZ + 
  prob_select_container * prob_green_given_YZ = 13 / 30 := 
  by sorry

end prob_green_ball_l291_291106


namespace find_negative_number_l291_291018

noncomputable def is_negative (x : ℝ) : Prop := x < 0

theorem find_negative_number : is_negative (-5) := by
  -- Proof steps would go here, but we'll skip them for now.
  sorry

end find_negative_number_l291_291018


namespace area_equals_375_l291_291899

def base := 25
def height := 15
def area_of_parallelogram := base * height

theorem area_equals_375 :
  area_of_parallelogram = 375 := by
  sorry

end area_equals_375_l291_291899


namespace sum_of_series_is_correct_l291_291495

noncomputable def geometric_series_sum_5_terms : ℚ :=
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  a * (1 - r^n) / (1 - r)

theorem sum_of_series_is_correct :
  geometric_series_sum_5_terms = 1023 / 3072 := by
  sorry

end sum_of_series_is_correct_l291_291495


namespace minimum_additional_coins_l291_291860

theorem minimum_additional_coins
  (friends : ℕ) (initial_coins : ℕ)
  (h_friends : friends = 15) (h_coins : initial_coins = 100) :
  ∃ additional_coins : ℕ, additional_coins = 20 :=
by
  have total_needed_coins : ℕ := (friends * (friends + 1)) / 2
  have total_coins : ℕ := initial_coins
  have additional_coins_needed : ℕ := total_needed_coins - total_coins
  have h_additional_coins : additional_coins_needed = 20 := by calculate 
  -- Finishing the proof with the result we calculated
  use additional_coins_needed
  exact h_additional_coins

end minimum_additional_coins_l291_291860


namespace dispatch_plans_l291_291783

theorem dispatch_plans (students : Finset ℕ) (h_students : students.card = 6) :
  ∃ plans : ℕ, plans = 180 ∧
  ∃ s, s ⊆ students ∧ s.card = 2 ∧
  ∃ f, f ⊆ (students \ s) ∧ f.card = 1 ∧
  ∃ sa, sa ⊆ (students \ (s ∪ f)) ∧ sa.card = 1 := by
  sorry

end dispatch_plans_l291_291783


namespace circles_area_ratio_l291_291787

-- Definition of the points and circles based on the given conditions
def point (x : ℝ) : ℝ := x

-- Conditions
def O : ℝ := 0
def Q : ℝ := 3
def Y := (Q - O) / 3

-- Radii of the circles
def radius_OQ := Q - O
def radius_OY := Y - O

-- Areas of the circles
def area_OQ := π * (radius_OQ) ^ 2
def area_OY := π * (radius_OY) ^ 2

-- Theorem statement: ratio of areas
theorem circles_area_ratio :
  (area_OY / area_OQ) = (1 / 9) :=
sorry

end circles_area_ratio_l291_291787


namespace general_term_formula_sum_of_first_n_terms_l291_291217

noncomputable def a (n : ℕ) : ℕ :=
(n + 2^n)^2

theorem general_term_formula :
  ∀ n : ℕ, a n = n^2 + n * 2^(n+1) + 4^n :=
sorry

noncomputable def S (n : ℕ) : ℕ :=
(n-1) * 2^(n+2) + 4 + (4^(n+1) - 4) / 3

theorem sum_of_first_n_terms :
  ∀ n : ℕ, S n = (n-1) * 2^(n+2) + 4 + (4^(n+1) - 4) / 3 :=
sorry

end general_term_formula_sum_of_first_n_terms_l291_291217


namespace max_value_h_l291_291676

/-- Let f(x) = 1/cos(x). The graph of f(x) is shifted to the right by π/3 units
to obtain the graph of g(x). Let h(x) = f(x) + g(x), where x ∈ [π/12, π/4].
Prove that the maximum value of h(x) is √6. -/
theorem max_value_h (x : ℝ) (h1 : x ∈ set.Icc (real.pi/12) (real.pi/4)) :
  let f := λ x, 1 / real.cos x,
      g := λ x, 1 / real.cos (x - real.pi / 3),
      h := λ x, f x + g x in
      ∃ m : ℝ, m = real.sqrt 6 ∧ (∀ y, y ∈ set.Icc (real.pi/12) (real.pi/4) → h y ≤ m) :=
sorry

end max_value_h_l291_291676


namespace problem1_solution_problem2_solution_problem3_solution_l291_291096

noncomputable def problem1 : Real :=
  3 * Real.sqrt 3 + Real.sqrt 8 - Real.sqrt 2 + Real.sqrt 27

theorem problem1_solution : problem1 = 6 * Real.sqrt 3 + Real.sqrt 2 := by
  sorry

noncomputable def problem2 : Real :=
  (1/2) * (Real.sqrt 3 + Real.sqrt 5) - (3/4) * (Real.sqrt 5 - Real.sqrt 12)

theorem problem2_solution : problem2 = 2 * Real.sqrt 3 - (1/4) * Real.sqrt 5 := by
  sorry

noncomputable def problem3 : Real :=
  (2 * Real.sqrt 5 + Real.sqrt 6) * (2 * Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - Real.sqrt 6) ^ 2

theorem problem3_solution : problem3 = 3 + 2 * Real.sqrt 30 := by
  sorry

end problem1_solution_problem2_solution_problem3_solution_l291_291096


namespace QuestionI_QuestionII_l291_291195

theorem QuestionI (a c : ℝ) 
(h₀ : ∀ x, (1 < x ∧ x < 3) ↔ (ax^2 + x + c > 0)) :
a = -1 / 4 ∧ c = -3 / 4 :=
sorry

theorem QuestionII (a c m : ℝ) 
(h₀ : a = -1 / 4 ∧ c = -3 / 4)
(h₁ : ∀ x, (ax^2 + 2x + 4c > 0) → (x + m > 0)) :
m ≥ -2 :=
sorry

end QuestionI_QuestionII_l291_291195


namespace sqrt10_parts_sqrt6_value_sqrt3_opposite_l291_291723

-- Problem 1
theorem sqrt10_parts : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 → (⌊Real.sqrt 10⌋ = 3 ∧ Real.sqrt 10 - 3 = Real.sqrt 10 - ⌊Real.sqrt 10⌋) :=
by
  sorry

-- Problem 2
theorem sqrt6_value (a b : ℝ) : a = Real.sqrt 6 - 2 ∧ b = 3 → (a + b - Real.sqrt 6 = 1) :=
by
  sorry

-- Problem 3
theorem sqrt3_opposite (x y : ℝ) : x = 13 ∧ y = Real.sqrt 3 - 1 → (-(x - y) = Real.sqrt 3 - 14) :=
by
  sorry

end sqrt10_parts_sqrt6_value_sqrt3_opposite_l291_291723


namespace find_radius_of_cone_base_l291_291950

def slant_height : ℝ := 5
def lateral_surface_area : ℝ := 15 * Real.pi

theorem find_radius_of_cone_base (A l : ℝ) (hA : A = lateral_surface_area) (hl : l = slant_height) : 
  ∃ r : ℝ, A = Real.pi * r * l ∧ r = 3 := 
by 
  sorry

end find_radius_of_cone_base_l291_291950


namespace max_value_sqrts_l291_291294

variable (a b c : ℝ)

theorem max_value_sqrts (h1 : a + b + c = 3) (h2 : a ≥ -1) (h3 : b ≥ -2) (h4 : c ≥ -3) :
  sqrt (2 * a + 2) + sqrt (4 * b + 8) + sqrt (6 * c + 18) ≤ 3 * sqrt 34 :=
sorry

end max_value_sqrts_l291_291294


namespace φ_sufficient_not_necessary_l291_291423

-- Definitions
def curve (φ : ℝ) (x : ℝ) : ℝ := sin(2 * x + φ)

-- Conditions
def passes_through_origin (φ : ℝ) : Prop := curve φ 0 = 0

-- Problem statement: φ = π is a sufficient but not necessary condition for the curve to pass through the origin
theorem φ_sufficient_not_necessary (φ : ℝ) : passes_through_origin π ∧ (¬ ∀ k : ℤ, φ = (k:ℝ) * π → φ = π) :=
by
  sorry

end φ_sufficient_not_necessary_l291_291423


namespace pollution_index_median_mode_l291_291248

def dataSet : List ℕ := [31, 35, 31, 34, 30, 32, 31]

def mode (l : List ℕ) : ℕ :=
  l.head' -- to fill with a proper implementation

def median (l : List ℕ) : ℕ :=
  l.head' -- to fill with a proper implementation

theorem pollution_index_median_mode :
  median dataSet = 31 ∧ mode dataSet = 31 := by
  /- Proof steps to be provided -/
  sorry

end pollution_index_median_mode_l291_291248


namespace ball_total_distance_after_fourth_bounce_l291_291820

noncomputable def initial_height : ℝ := 20
noncomputable def bounce_ratio : ℝ := 2 / 3

-- The total distance travelled by the ball after the fourth bounce.
theorem ball_total_distance_after_fourth_bounce (h₀ : ℝ) (r : ℝ) :
  h₀ = initial_height → 
  r = bounce_ratio → 
  let h₁ := h₀ * r,
      h₂ := h₁ * r,
      h₃ := h₂ * r in
  h₀ + 2 * (h₁) + 2 * (h₂) + 2 * (h₃) + h₃ * r ≈ 80 :=
by
  sorry

end ball_total_distance_after_fourth_bounce_l291_291820


namespace compare_xyz_l291_291738

variable {a b : ℝ}
-- We define the necessary conditions first
def conditions (a b : ℝ) : Prop := 0 < a ∧ a < b ∧ b < 1

-- Define the variables x, y, and z based on the conditions
def x (b a : ℝ) : ℝ := (1/(Real.sqrt b)) - (1/(Real.sqrt (b + a)))
def y (b a : ℝ) : ℝ := (1/(b - a)) - (1/b)
def z (b a : ℝ) : ℝ := (1/(Real.sqrt (b - a))) - (1/(Real.sqrt b))

-- State the main theorem using the definitions and conditions
theorem compare_xyz (h : conditions a b) : x b a < z b a ∧ z b a < y b a :=
by
  sorry

end compare_xyz_l291_291738


namespace wholesale_cost_approx_l291_291912

-- Define the conditions
def wholesale_cost := ℝ -- Wholesale cost W is a real number
def selling_price : ℝ := 28 -- Selling price is $28
def profit_percentage : ℝ := 0.17 -- Profit percentage is 17%

-- Define the mathematical relationship given in the problem
def selling_price_eq (w : wholesale_cost) : Prop :=
  selling_price = (1 + profit_percentage) * w

-- Define the theorem to state what we want to prove
theorem wholesale_cost_approx :
  ∃ (W : wholesale_cost), W ≈ 23.93 ∧ selling_price_eq W :=
by
  sorry

end wholesale_cost_approx_l291_291912


namespace count_two_digit_numbers_with_8_l291_291994

theorem count_two_digit_numbers_with_8 : 
  (card {n : ℕ | 10 <= n ∧ n < 100 ∧ (n / 10 = 8 ∨ n % 10 = 8)}) = 17 := 
by 
  sorry

end count_two_digit_numbers_with_8_l291_291994


namespace count_two_digit_numbers_with_8_l291_291995

theorem count_two_digit_numbers_with_8 : 
  (card {n : ℕ | 10 <= n ∧ n < 100 ∧ (n / 10 = 8 ∨ n % 10 = 8)}) = 17 := 
by 
  sorry

end count_two_digit_numbers_with_8_l291_291995


namespace gcf_60_75_l291_291402

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l291_291402


namespace cos_alpha_minus_beta_cos_beta_l291_291914

open Real

-- Given conditions
variables (α β : ℝ)
variables (cos_α sin_α cos_β sin_β : ℝ)
variables (hα : -π / 2 < α) (hα' : α < 0) (hβ : 0 < β) (hβ' : β < π / 2)
variables (ha : cos_α = cos α) (hb : sin_α = sin α)
variables (hab : cos_β = cos β) (hb' : sin_β = sin β)
variables (hmag : sqrt ((cos_α - cos_β)^2 + (sin_α - sin_β)^2) = √10 / 5)

-- Problem 1: Prove that cos(α - β) = 4 / 5
theorem cos_alpha_minus_beta : cos(α - β) = 4 / 5 :=
by sorry

-- Problem 2: Given cos α = 12 / 13, prove that cos β = 63 / 65
variables (hcos_α : cos_α = 12 / 13)
theorem cos_beta : cos β = 63 / 65 :=
by sorry

end cos_alpha_minus_beta_cos_beta_l291_291914


namespace range_of_x_plus_y_l291_291559

theorem range_of_x_plus_y 
  (x y : ℝ) 
  (h1 : 0 ≤ y) 
  (h2 : y ≤ x) 
  (h3 : x ≤ π / 2) 
  (h4 : 4 * cos y ^ 2 + 4 * cos x * sin y - 4 * cos x ^ 2 ≤ 1) :
  x + y ∈ (Set.Icc (0 : ℝ) (π / 6)) ∪ Set.Icc (5 * π / 6) π := 
sorry

end range_of_x_plus_y_l291_291559


namespace average_anchors_is_13_div_8_l291_291908
  
def is_anchor (n : ℕ) (S : Finset ℕ) : Prop :=
  n ∈ S ∧ (n + S.card) ∈ S
  
def num_of_anchors (S : Finset ℕ) : ℕ :=
  (Finset.range 16).filter (λ n => is_anchor n S).card

def average_anchors : ℚ :=
  (Finset.powerset (Finset.range 15)).sum (λ S => num_of_anchors S : ℚ) / 2^15

theorem average_anchors_is_13_div_8 : average_anchors = 13 / 8 := by
  sorry

end average_anchors_is_13_div_8_l291_291908


namespace max_hours_at_regular_rate_l291_291826

-- Define the maximum hours at regular rate H
def max_regular_hours (H : ℕ) : Prop := 
  let regular_rate := 16
  let overtime_rate := 16 + (0.75 * 16)
  let total_hours := 60
  let total_compensation := 1200
  16 * H + 28 * (total_hours - H) = total_compensation

theorem max_hours_at_regular_rate : ∃ H, max_regular_hours H ∧ H = 40 :=
sorry

end max_hours_at_regular_rate_l291_291826


namespace solve_for_x_l291_291406

theorem solve_for_x (x : ℝ) (h : (8 - x)^2 = x^2) : x = 4 := 
by 
  sorry

end solve_for_x_l291_291406


namespace max_perfect_squares_among_partials_l291_291044

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem max_perfect_squares_among_partials (a : Fin 100 → ℕ)
  (h_permutation : ∀ i, a i ∈ Finset.range 1 101 ∧ 
  (∀ j1 j2, a j1 = a j2 → j1 = j2)) :
  ∃ M : ℕ, (∀ S : Fin 100 → ℕ, 
  (∀ i : Fin 100, S i = ∑ j in Finset.range (i.1+1), a ⟨j, sorry⟩) →
  (M = Finset.card {i ∈ Finset.range 100 | is_perfect_square (S ⟨i, sorry⟩)})) ∧ M = 60 :=
begin
  sorry
end

end max_perfect_squares_among_partials_l291_291044


namespace cesaro_sum_100_terms_l291_291813

noncomputable def cesaro_sum (A : List ℝ) : ℝ :=
  let n := A.length
  (List.sum A) / n

theorem cesaro_sum_100_terms :
  ∀ (A : List ℝ), A.length = 99 →
  cesaro_sum A = 1000 →
  cesaro_sum (1 :: A) = 991 :=
by
  intros A h1 h2
  sorry

end cesaro_sum_100_terms_l291_291813


namespace negative_number_among_options_l291_291014

theorem negative_number_among_options :
  let A := |(-2 : ℤ)|
      B := real.sqrt 3
      C := (0 : ℤ)
      D := (-5 : ℤ)
  in D = -5 := 
by 
  sorry

end negative_number_among_options_l291_291014


namespace common_sequence_sum_l291_291777

open Int

def arithmetic_sequence_sum (a d n : ℕ) := n * (2 * a + (n - 1) * d) / 2

theorem common_sequence_sum :
  let a1 := 2
  let d1 := 4
  let l1 := 190
  let a2 := 2
  let d2 := 6
  let l2 := 200
  let common_diff := lcm d1 d2
  let n := 16
  let last_term := a1 + (n - 1) * common_diff
  arithmetic_sequence_sum a1 common_diff n = 1472 :=
by
  let a1 := 2
  let d1 := 4
  let l1 := 190
  let a2 := 2
  let d2 := 6
  let l2 := 200
  let common_diff := lcm d1 d2
  let n := 16
  let last_term := a1 + (n - 1) * common_diff
  show arithmetic_sequence_sum a1 common_diff n = 1472
  sorry

end common_sequence_sum_l291_291777


namespace find_negative_number_l291_291017

noncomputable def is_negative (x : ℝ) : Prop := x < 0

theorem find_negative_number : is_negative (-5) := by
  -- Proof steps would go here, but we'll skip them for now.
  sorry

end find_negative_number_l291_291017


namespace inscribed_circle_radius_l291_291404

theorem inscribed_circle_radius
  (DE DF EF : ℝ)
  (h1 : DE = 8) 
  (h2 : DF = 8) 
  (h3 : EF = 10) :
  let s := (DE + DF + EF) / 2 in
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  let r := K / s in
  r = 5 * Real.sqrt 195 / 13 :=
by
  sorry

end inscribed_circle_radius_l291_291404


namespace middle_segment_proportion_l291_291182

theorem middle_segment_proportion (a b c : ℝ) (h_a : a = 1) (h_b : b = 3) :
  (a / c = c / b) → c = Real.sqrt 3 :=
by
  sorry

end middle_segment_proportion_l291_291182


namespace shortest_multicolored_cycle_l291_291708

theorem shortest_multicolored_cycle (G : Graph) (cycle : List (Vertex × Vertex)) :
  (∀ (a_i b_i : Vertex), (a_i, b_i) ∈ cycle) →
  (length cycle = 2 * s) →
  (∀ a_i, a_i ∈ cycle → ∃ h : Horizontal, a_i = to_vertex h) →
  (∀ b_j, b_j ∈ cycle → ∃ v : Vertical, b_j = to_vertex v) →
  (∃ (s > 2 → False), shortest_multicolored_cycle_in_G = 4) := 
by
  sorry

end shortest_multicolored_cycle_l291_291708


namespace monotonic_intervals_l291_291529

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem monotonic_intervals :
  {x : ℝ | 0 ≤ deriv f x} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | x ≥ 0} :=
by
  sorry

end monotonic_intervals_l291_291529


namespace solve_equation_l291_291331

theorem solve_equation (x : ℝ) (h : (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2) : 
  x = -2/3 :=
sorry

end solve_equation_l291_291331


namespace hexagon_area_correct_l291_291070

noncomputable def side_length_square (A_square : ℝ) : ℝ := real.sqrt 16

noncomputable def side_length_hexagon (s : ℝ) : ℝ := 2 * s / 9

noncomputable def area_hexagon (t : ℝ) : ℝ := (3 * t^2 * real.sqrt 3) / 2

theorem hexagon_area_correct {s t : ℝ} 
  (h_square_area : s^2 = 16) 
  (h_perimeter_ratio : 4 * s = 18 * t) : 
  area_hexagon t = 32 * real.sqrt 3 / 27 :=
by 
  sorry

end hexagon_area_correct_l291_291070


namespace cost_of_tissues_l291_291470
-- Import the entire Mathlib library

-- Define the context and the assertion without computing the proof details
theorem cost_of_tissues
  (n_tp : ℕ) -- Number of toilet paper rolls
  (c_tp : ℝ) -- Cost per toilet paper roll
  (n_pt : ℕ) -- Number of paper towels rolls
  (c_pt : ℝ) -- Cost per paper towel roll
  (n_t : ℕ) -- Number of tissue boxes
  (T : ℝ) -- Total cost of all items
  (H_tp : n_tp = 10) -- Given: 10 rolls of toilet paper
  (H_c_tp : c_tp = 1.5) -- Given: $1.50 per roll of toilet paper
  (H_pt : n_pt = 7) -- Given: 7 rolls of paper towels
  (H_c_pt : c_pt = 2) -- Given: $2 per roll of paper towel
  (H_t : n_t = 3) -- Given: 3 boxes of tissues
  (H_T : T = 35) -- Given: total cost is $35
  : (T - (n_tp * c_tp + n_pt * c_pt)) / n_t = 2 := -- Conclusion: the cost of one box of tissues is $2
by {
  sorry -- Proof details to be supplied here
}

end cost_of_tissues_l291_291470


namespace range_of_a_plus_3b_l291_291933

theorem range_of_a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b) (h2 : a + b ≤ 1) (h3 : 1 ≤ a - 2b) (h4 : a - 2b ≤ 3) :
  -11/3 ≤ a + 3b ∧ a + 3b ≤ 1 :=
sorry

end range_of_a_plus_3b_l291_291933


namespace minimum_additional_coins_l291_291855

-- The conditions
def total_friends : ℕ := 15
def current_coins : ℕ := 100

-- The fact that the total coins required to give each friend a unique number of coins from 1 to 15 is 120
def total_required_coins : ℕ := (total_friends * (total_friends + 1)) / 2

-- The theorem stating the required number of additional coins
theorem minimum_additional_coins (total_friends : ℕ) (current_coins : ℕ) (total_required_coins : ℕ) : ℕ :=
  sorry

end minimum_additional_coins_l291_291855


namespace rotation_volumes_l291_291171

theorem rotation_volumes (a b c V1 V2 V3 : ℝ) (h : a^2 + b^2 = c^2)
    (hV1 : V1 = (1 / 3) * Real.pi * a^2 * b^2 / c)
    (hV2 : V2 = (1 / 3) * Real.pi * b^2 * a)
    (hV3 : V3 = (1 / 3) * Real.pi * a^2 * b) : 
    (1 / V1^2) = (1 / V2^2) + (1 / V3^2) :=
sorry

end rotation_volumes_l291_291171


namespace line_ellipse_intersection_l291_291557

-- Define the problem conditions and the proof problem statement.
theorem line_ellipse_intersection (k m : ℝ) : 
  (∀ x y, y - k * x - 1 = 0 → ((x^2 / 5) + (y^2 / m) = 1)) →
  (m ≥ 1) ∧ (m ≠ 5) ∧ (m < 5 ∨ m > 5) :=
sorry

end line_ellipse_intersection_l291_291557


namespace fractions_inequality_l291_291578

variable {a b c d : ℝ}
variable (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0)

theorem fractions_inequality : 
  (a > b) → (b > 0) → (c < d) → (d < 0) → (a / d < b / c) :=
by
  intros h1 h2 h3 h4
  sorry

end fractions_inequality_l291_291578


namespace geo_prog_c_value_gen_term_c_2_l291_291173

-- Define the sequence sum condition
def sequence_sum (n : ℕ) (c : ℕ) : ℕ := 2^n + c

-- Problem (1): Prove that if {a_n} is a geometric progression and S_n = 2^n + c, then c = -1.
theorem geo_prog_c_value {a : ℕ → ℕ} (c : ℕ) (h1 : ∀ n, 2 * a (n + 1) = a n * a (n + 2))
  (h2 : ∀ n, sequence_sum (n + 1) c = sequence_sum n c + a (n + 1)) :
  c = -1 :=
by sorry

-- Problem (2): Prove that if c = 2 and S_n = 2^n + 2, then the general term for the sequence {a_n} is as given.
theorem gen_term_c_2 {a : ℕ → ℕ} (h1 : ∀ n, sequence_sum (n + 1) 2 = sequence_sum n 2 + a (n + 1)) :
  ∀ n, a n = if n = 0 then 4 else 2^(n-1) :=
by sorry

end geo_prog_c_value_gen_term_c_2_l291_291173


namespace quadratic_poly_no_fixed_point_has_large_bound_l291_291154

/-- Given a quadratic polynomial f(x) with real coefficients such that the coefficient
of the x^2 term is positive and there is no real alpha such that f(alpha) = alpha, 
then there exists a positive integer n such that for any sequence {a_i} where 
a_i = f(a_{i-1}) for 1 ≤ i ≤ n, we have a_n > 2021. -/
theorem quadratic_poly_no_fixed_point_has_large_bound
  (f : ℝ → ℝ)
  (h_quad : ∃ a b c : ℝ, a > 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h_no_fixed : ∀ α : ℝ, f α ≠ α) :
  ∃ n : ℕ, ∀ (a0 : ℝ) (a : ℕ → ℝ), 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a (i + 1) = f (a i)) →
    a n > 2021 :=
sorry

end quadratic_poly_no_fixed_point_has_large_bound_l291_291154


namespace Mina_cookies_count_is_41_l291_291910

def Carl_cookie_area : ℝ := 4 * 6
def Carl_total_area : ℝ := 12 * Carl_cookie_area
def Mina_cookie_side : ℝ := 4
def Mina_cookie_area : ℝ := (Real.sqrt 3 / 4) * Mina_cookie_side^2
def Mina_cookies_count : ℝ := Carl_total_area / Mina_cookie_area

theorem Mina_cookies_count_is_41 : Mina_cookies_count = 41 :=
  by 
    sorry -- proof omitted

end Mina_cookies_count_is_41_l291_291910


namespace binomial_expansion_coefficient_l291_291884

theorem binomial_expansion_coefficient :
  let general_term (r : ℕ) := (5.choose r) * (-3)^r * x ^ ((10 - 3 * r) / 2)
  (function.eval ![(-3 : ℤ), x] (binom_sum 0 5 general_term)).coeff 2 = 90 :=
by
  sorry

end binomial_expansion_coefficient_l291_291884


namespace triangle_divided_into_similar_triangles_l291_291136

theorem triangle_divided_into_similar_triangles (ΔABC : Triangle ℝ) (v : Vertex) :
  (∃ l : Line, l.passes_through v ∧ divides_into_similar_triangles ΔABC l) ↔
    (ΔABC.is_right_triangle ∨ ΔABC.is_isosceles_triangle) ∧
    ∃ l : Line, l.is_altitude ΔABC v :=
sorry

end triangle_divided_into_similar_triangles_l291_291136


namespace bishops_arrangements_square_l291_291309

theorem bishops_arrangements_square :
  ∃ k : ℕ, (number_of_non_threatening_bishops_arrangements 8 8) = k * k :=
sorry

end bishops_arrangements_square_l291_291309


namespace find_volume_of_sphere_l291_291221

noncomputable def volume_of_sphere_proof_problem 
  (A1 A2 : ℝ) (d1 d2 : ℝ) (R : ℝ) (V : ℝ) : Prop :=
  (A1 = 5 * Real.pi) ∧
  (A2 = 8 * Real.pi) ∧
  (|d1 - d2| = 1) ∧
  (A1 = Real.pi * (sqrt (R^2 - d1^2))^2) ∧
  (A2 = Real.pi * (sqrt (R^2 - d2^2))^2) →
  V = (4 / 3) * Real.pi * R^3

theorem find_volume_of_sphere
  (A1 A2 : ℝ) (d1 d2 : ℝ)
  (h : volume_of_sphere_proof_problem A1 A2 d1 d2 3 36 * Real.pi) 
  : 36 * Real.pi = (4 / 3) * Real.pi * 3^3 :=
sorry

end find_volume_of_sphere_l291_291221


namespace bun_eating_problem_l291_291632

theorem bun_eating_problem
  (n k : ℕ)
  (H1 : 5 * n / 10 + 3 * k / 10 = 180) -- This corresponds to the condition that Zhenya eats 5 buns in 10 minutes, and Sasha eats 3 buns in 10 minutes, for a total of 180 minutes.
  (H2 : n + k = 70) -- This corresponds to the total number of buns eaten.
  : n = 40 ∧ k = 30 :=
by
  sorry

end bun_eating_problem_l291_291632


namespace quantile_80_percent_eq_9_l291_291174

-- Given data set
def dataSet : List ℕ := [2, 6, 5, 4, 7, 9, 8, 10]

-- Define a function to compute the quantile position
def quantile_position (p : ℚ) (data : List ℕ) : ℚ :=
  p * (data.length : ℚ)

-- Define the quantile function that rounds up to the nearest integer position
def quantile (p : ℚ) (data : List ℕ) : ℕ :=
  let pos := quantile_position p data
  data.nthLe ⌊pos⌋.toNat (by sorry)

-- Define a theorem to state the problem
theorem quantile_80_percent_eq_9 : quantile (80 / 100) dataSet = 9 := 
sorry

end quantile_80_percent_eq_9_l291_291174


namespace remainder_division_l291_291011

theorem remainder_division (a b c : ℤ) (h1 : a ≡ 1 [MOD 36]) (h2 : b ≡ 3 [MOD 36]) (h3 : c ≡ 5 [MOD 36]) : 
  (a^3 * b^4 * c^5) % 36 = 9 := by
  sorry

end remainder_division_l291_291011


namespace no_three_term_arith_progression_l291_291720

theorem no_three_term_arith_progression (n : ℕ) (hn : n ≥ 1) :
  ∃ (A : finset ℕ), A.card = 2^n ∧ (∀ (x y z : ℕ), x ∈ A → y ∈ A → z ∈ A → x < y → y < z → ¬(2 * y = x + z)) :=
sorry

end no_three_term_arith_progression_l291_291720


namespace solution_set_l291_291239

noncomputable def f : ℝ → ℝ := sorry

-- The function f is defined to be odd.
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- The function f is increasing on (-∞, 0).
axiom increasing_f : ∀ x y : ℝ, x < y ∧ y < 0 → f x < f y

-- Given f(2) = 0
axiom f_at_2 : f 2 = 0

-- Prove the solution set for x f(x + 1) < 0
theorem solution_set : { x : ℝ | x * f (x + 1) < 0 } = {x : ℝ | (-3 < x ∧ x < -1) ∨ (0 < x ∧ x < 1)} :=
by
  sorry

end solution_set_l291_291239


namespace suitable_outfits_l291_291996

def num_shirts : ℕ := 8
def num_pants : ℕ := 6
def num_hats : ℕ := 6
def num_colors : ℕ := 6

theorem suitable_outfits :
  let total_combinations := num_shirts * num_pants * num_hats,
      same_color_shirt_pants := num_colors * num_hats,
      same_color_pants_hats := num_colors * num_shirts,
      same_color_shirt_hats := num_colors * num_pants,
      overcounted := 6 in
  total_combinations - (same_color_shirt_pants + same_color_pants_hats + same_color_shirt_hats - overcounted) = 174 :=
by
  let total_combinations := num_shirts * num_pants * num_hats
  let same_color_shirt_pants := num_colors * num_hats
  let same_color_pants_hats := num_colors * num_shirts
  let same_color_shirt_hats := num_colors * num_pants
  let overcounted := 6
  sorry

end suitable_outfits_l291_291996


namespace area_pentagon_line_segments_l291_291344

theorem area_pentagon_line_segments (ABCDE : Type) (length : ℕ) (area : ℝ) (m n : ℝ):
  (∀ (a b c d e: ABCDE), length = 2) → (∃ (m n : ℝ), area = sqrt m + sqrt n ∧ m + n = 23) :=
sorry

end area_pentagon_line_segments_l291_291344


namespace not_divisible_2310_l291_291719

theorem not_divisible_2310 (n : ℕ) (h : n < 2310) : ¬ (2310 ∣ n * (2310 - n)) :=
sorry

end not_divisible_2310_l291_291719


namespace ravioli_to_tortellini_ratio_l291_291638

-- Definitions from conditions
def total_students : ℕ := 800
def ravioli_students : ℕ := 300
def tortellini_students : ℕ := 150

-- Ratio calculation as a theorem
theorem ravioli_to_tortellini_ratio : 2 = ravioli_students / Nat.gcd ravioli_students tortellini_students :=
by
  -- Given the defined values
  have gcd_val : Nat.gcd ravioli_students tortellini_students = 150 := by
    sorry
  have ratio_simp : ravioli_students / 150 = 2 := by
    sorry
  exact ratio_simp

end ravioli_to_tortellini_ratio_l291_291638


namespace expected_coin_worth_is_two_l291_291052

-- Define the conditions
def p_heads : ℚ := 4 / 5
def p_tails : ℚ := 1 / 5
def gain_heads : ℚ := 5
def loss_tails : ℚ := -10

-- Expected worth calculation
def expected_worth : ℚ := (p_heads * gain_heads) + (p_tails * loss_tails)

-- Lean 4 statement to prove
theorem expected_coin_worth_is_two : expected_worth = 2 := by
  sorry

end expected_coin_worth_is_two_l291_291052


namespace days_to_clear_messages_l291_291690

theorem days_to_clear_messages 
  (initial_messages : ℕ)
  (messages_read_per_day : ℕ)
  (new_messages_per_day : ℕ) 
  (net_messages_cleared_per_day : ℕ)
  (d : ℕ) :
  initial_messages = 98 →
  messages_read_per_day = 20 →
  new_messages_per_day = 6 →
  net_messages_cleared_per_day = messages_read_per_day - new_messages_per_day →
  d = initial_messages / net_messages_cleared_per_day →
  d = 7 :=
by
  intros h_initial h_read h_new h_net h_days
  sorry

end days_to_clear_messages_l291_291690


namespace cost_of_one_box_of_tissues_l291_291472

variable (num_toilet_paper : ℕ) (num_paper_towels : ℕ) (num_tissues : ℕ)
variable (cost_toilet_paper : ℝ) (cost_paper_towels : ℝ) (total_cost : ℝ)

theorem cost_of_one_box_of_tissues (num_toilet_paper = 10) 
                                   (num_paper_towels = 7) 
                                   (num_tissues = 3)
                                   (cost_toilet_paper = 1.50) 
                                   (cost_paper_towels = 2.00) 
                                   (total_cost = 35.00) :
  let total_cost_toilet_paper := num_toilet_paper * cost_toilet_paper,
      total_cost_paper_towels := num_paper_towels * cost_paper_towels,
      cost_left_for_tissues := total_cost - (total_cost_toilet_paper + total_cost_paper_towels),
      one_box_tissues_cost := cost_left_for_tissues / num_tissues
  in one_box_tissues_cost = 2.00 := 
sorry

end cost_of_one_box_of_tissues_l291_291472


namespace Miquel_theorem_l291_291817

open EuclideanGeometry

theorem Miquel_theorem
  (A B C P Q R : Point)
  (hP : ∃ (t : ℝ), P = t • B + (1 - t) • C)
  (hQ : ∃ (t : ℝ), Q = t • C + (1 - t) • A)
  (hR : ∃ (t : ℝ), R = t • A + (1 - t) • B) :
  ∃ (T : Point), 
    (IsOnCircumcircle T (triangle.mk A R Q)) ∧ 
    (IsOnCircumcircle T (triangle.mk B P R)) ∧ 
    (IsOnCircumcircle T (triangle.mk C Q P)) :=
sorry

end Miquel_theorem_l291_291817


namespace factorize_expression_l291_291125

theorem factorize_expression (x : ℝ) : -2 * x^2 + 2 * x - (1 / 2) = -2 * (x - (1 / 2))^2 :=
by
  sorry

end factorize_expression_l291_291125


namespace unique_paths_from_A_to_C_via_B_l291_291054

-- Define a type to represent points and a specific lattice function
inductive Point
| A | B | C | P | Q | R | S | T | U | V | W | X deriving DecidableEq

-- Given path constraints between points
def path_constraints : List (Point × Point) :=
  [
    (Point.A, Point.P), (Point.A, Point.Q),
    (Point.P, Point.B), (Point.Q, Point.B),
    (Point.B, Point.R), (Point.B, Point.S),
    (Point.R, Point.T), (Point.R, Point.U),
    (Point.S, Point.V), (Point.S, Point.W),
    (Point.T, Point.C), (Point.U, Point.C),
    (Point.V, Point.C), (Point.W, Point.C)
  ]

-- Path traversal constrains
structure Path (p1 p2 : Point) : Type :=
(kind : List Point)
(valid : kind.head = p1 ∧ kind.last = p2 ∧ 
        (∀ i ∈ kind.zip kind.tail, (i ∈ path_constraints)))

-- Function to count unique paths
def count_unique_paths_from_A_to_C_via_B (paths_from_A_to_B : Nat) (paths_from_B_to_C : Nat) : Nat :=
  paths_from_A_to_B * paths_from_B_to_C

-- Define paths from A to B, and B to C based on provided conditions
noncomputable def paths_from_A_to_B : Nat := 4
noncomputable def paths_from_B_to_C : Nat := 20

theorem unique_paths_from_A_to_C_via_B : count_unique_paths_from_A_to_C_via_B paths_from_A_to_B paths_from_B_to_C = 80 := by
  sorry

end unique_paths_from_A_to_C_via_B_l291_291054


namespace angle_between_vectors_l291_291942

theorem angle_between_vectors (a b : ℝ^3) (θ : ℝ) :
  |a| = sqrt 10 →
  a • b = - (5 * sqrt 30) / 2 →
  ((a - b) • (a + b)) = -15 →
  θ = 5 * π / 6 :=
by
  sorry

end angle_between_vectors_l291_291942


namespace area_of_sector_l291_291192

-- Defining the conditions
def l : ℝ := 4 * Real.pi
def r : ℝ := 8

-- Theorem statement
theorem area_of_sector : (1 / 2 * l * r) = 16 * Real.pi := by
  sorry

end area_of_sector_l291_291192


namespace min_magnitude_l291_291982

open_locale real

noncomputable def c (λ : ℝ) : ℝ × ℝ :=
  (3 * λ, 4 - 4 * λ)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem min_magnitude : ∃ λ : ℝ, magnitude (c λ) = 12 / 5 :=
begin
  use 16 / 25,
  unfold magnitude c,
  norm_num,
  rw [←real.sqrt_sq (by norm_num : 0 ≤ 144 / 25)],
  congr,
  field_simp,
  ring,
end

#print axioms min_magnitude

end min_magnitude_l291_291982


namespace complex_division_correct_l291_291193

noncomputable def z1 : ℂ := 2 - complex.I
noncomputable def z2 : ℂ := -complex.I

theorem complex_division_correct : (z1 / z2) = (1 + 2 * complex.I) := by
  sorry

end complex_division_correct_l291_291193


namespace contrapositive_statement_l291_291206

theorem contrapositive_statement 
  (a : ℝ) (b : ℝ) 
  (h1 : a > 0) 
  (h3 : a + b < 0) : 
  b < 0 :=
sorry

end contrapositive_statement_l291_291206


namespace maximum_value_40_l291_291622

theorem maximum_value_40 (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 10) :
  (a-b)^2 + (a-c)^2 + (a-d)^2 + (b-c)^2 + (b-d)^2 + (c-d)^2 ≤ 40 :=
sorry

end maximum_value_40_l291_291622


namespace sequence_an_general_term_sequence_bn_sum_l291_291583

theorem sequence_an_general_term
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (hS : ∀ n, S n = n^2 + 1/2 * n) :
  ∀ n, a n = 2 * n - 1/2 :=
sorry

theorem sequence_bn_sum
  (b : ℕ → ℚ)
  (a : ℕ → ℚ)
  (T : ℕ → ℚ)
  (hS : ∀ n, S n = n^2 + 1/2 * n)
  (hA : ∀ n, a n = 2 * n - 1/2)
  (hB : ∀ n, b n = 3^(a n + 1/2))
  (hT : ∀ n, T n = (3^2 * (1 - (3^2)^n)) / (1 - 3^2)) :
  ∀ n, T n = (9^(n + 1) - 9) / 8 :=
sorry

end sequence_an_general_term_sequence_bn_sum_l291_291583


namespace additional_coins_needed_l291_291856

theorem additional_coins_needed (friends : Nat) (current_coins : Nat) : 
  friends = 15 → current_coins = 100 → 
  let total_coins_needed := (friends * (friends + 1)) / 2 
  in total_coins_needed - current_coins = 20 :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end additional_coins_needed_l291_291856


namespace pink_highlighters_count_l291_291247

theorem pink_highlighters_count (yellow blue total : ℕ) (h_yellow : yellow = 8) (h_blue : blue = 5) (h_total : total = 22) :
  total - (yellow + blue) = 9 :=
by
  rw [h_yellow, h_blue, h_total]
  sorry

end pink_highlighters_count_l291_291247


namespace find_z_l291_291811

-- Definitions based on the conditions from the problem
def x : ℤ := sorry
def y : ℤ := x - 1
def z : ℤ := x - 2
def condition1 : x > y ∧ y > z := by
  sorry

def condition2 : 2 * x + 3 * y + 3 * z = 5 * y + 11 := by
  sorry

-- Statement to prove
theorem find_z : z = 3 :=
by
  -- Use the conditions to prove the statement
  have h1 : x > y ∧ y > z := condition1
  have h2 : 2 * x + 3 * y + 3 * z = 5 * y + 11 := condition2
  sorry

end find_z_l291_291811


namespace range_of_data_set_l291_291850

theorem range_of_data_set :
  let data_set := {1, 3, -3, 0, -Real.pi}
  let max_value := 3
  let min_value := -Real.pi
  (max_value - min_value) = 3 + Real.pi := by
  let data_set := {1, 3, -3, 0, -Real.pi}
  let max_value := 3
  let min_value := -Real.pi
  show (max_value - min_value) = 3 + Real.pi
  sorry

end range_of_data_set_l291_291850


namespace geometric_progression_terms_l291_291240

theorem geometric_progression_terms (a : ℝ) (m : ℕ) (h_ratio: ∑ i in range 6, a * 3 ^ i / ∑ i in range m, a * 3 ^ i = 28) : m = 3 := sorry

end geometric_progression_terms_l291_291240


namespace particle_probability_l291_291841

noncomputable def P : ℕ → ℕ → ℚ :=
  λ x y, if x = 0 && y = 0 then 1
  else if x = 0 || y = 0 then 0
  else (P (x - 1) y + P x (y - 1) + P (x - 1) (y - 1)) / 3

theorem particle_probability :
  let p : ℚ := P 5 3 in
  ∃ (m n : ℕ), p = m / 3^n ∧ m % 3 ≠ 0 ∧ m + n = 127 :=
by
  sorry

end particle_probability_l291_291841


namespace seniors_in_statistics_correct_l291_291698

-- Conditions
def total_students : ℕ := 120
def percentage_statistics : ℚ := 1 / 2
def percentage_seniors_in_statistics : ℚ := 9 / 10

-- Definitions based on conditions
def students_in_statistics : ℕ := total_students * percentage_statistics
def seniors_in_statistics : ℕ := students_in_statistics * percentage_seniors_in_statistics

-- Statement to prove
theorem seniors_in_statistics_correct :
  seniors_in_statistics = 54 :=
by
  -- Proof goes here
  sorry

end seniors_in_statistics_correct_l291_291698


namespace octagon_area_l291_291768

theorem octagon_area (side_large_square side_small_square : ℝ)
  (center_shared : Prop)
  (rotation_45_degrees : Prop)
  (segment_ab_length : ℝ)
  (h1 : side_large_square = 2)
  (h2 : side_small_square = 1)
  (h3 : segment_ab_length = 3 / 4)
  (h4 : center_shared)
  (h5 : rotation_45_degrees) :
  ∃ (m n : ℕ), m + n = 4 ∧ (3 : ℝ) = (3 / 1 : ℝ) ∧ (gcd m n = 1) :=
begin
  sorry
end

end octagon_area_l291_291768


namespace solve_inequality_l291_291556

noncomputable def f (x : ℝ) : ℝ :=
  x^3 + x + 2^x - 2^(-x)

theorem solve_inequality (x : ℝ) : 
  f (Real.exp x - x) ≤ 7/2 ↔ x = 0 := 
sorry

end solve_inequality_l291_291556


namespace vector_subtraction_result_l291_291222

-- Defining the vectors a and b as given in the conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- The main theorem stating that a - 2b results in the expected coordinates
theorem vector_subtraction_result :
  a - 2 • b = (7, -2) := by
  sorry

end vector_subtraction_result_l291_291222


namespace distance_AB_leq_1_over_100_l291_291317

open Real

theorem distance_AB_leq_1_over_100 :
  ∃ t > 0, ∀ t ≥ 100, |(t^3 + t + 1)^(1/3) - t| < 1 / 100 :=
by
  sorry

end distance_AB_leq_1_over_100_l291_291317


namespace count_not_div_by_4_l291_291156

def floorsum (a b c d n : ℕ) : ℕ :=
  (a / n) + (b / n) + (c / n) + (d / n)

theorem count_not_div_by_4 : 
  (finset.filter (λ n : ℕ, n ≤ 1200 ∧ ¬ (4 ∣ floorsum 1197 1198 1199 1200 n)) (finset.range 1201)).card = 35 :=
by
  sorry

end count_not_div_by_4_l291_291156


namespace primes_diff_power_of_two_divisible_by_three_l291_291043

theorem primes_diff_power_of_two_divisible_by_three
  (p q : ℕ) (m n : ℕ)
  (hp : Prime p) (hq : Prime q) (hp_gt : p > 3) (hq_gt : q > 3)
  (diff : q - p = 2^n ∨ p - q = 2^n) :
  3 ∣ (p^(2*m+1) + q^(2*m+1)) := by
  sorry

end primes_diff_power_of_two_divisible_by_three_l291_291043


namespace shop_owner_profit_approx_1_01_l291_291068

noncomputable def percentage_profit : ℝ :=
  let cost_price_professed : ℝ := 100
  let false_weight_ratio : ℝ := 0.10
  let actual_value_when_buying := cost_price_professed * (1 + false_weight_ratio)
  let actual_value_when_selling := cost_price_professed * (1 - false_weight_ratio)
  let effective_cost_price := actual_value_when_buying * (actual_value_when_selling / cost_price_professed)
  let profit := cost_price_professed - effective_cost_price
  let percentage_profit := (profit / effective_cost_price) * 100
  percentage_profit

theorem shop_owner_profit_approx_1_01 :
  percentage_profit ≈ 1.01 := sorry

end shop_owner_profit_approx_1_01_l291_291068


namespace s_4_eq_14916_l291_291417

-- Define the function that forms the integer by attaching the first n perfect squares
def s (n : ℕ) : ℕ :=
  let squares := List.range n |>.map (λ i => (i + 1) * (i + 1))
  let digits := squares.map (λ sq => sq.toString)
  digits.foldl (λ acc d => acc ++ d) "" |>.toNat

-- Define the task to prove the specific case for n=4
theorem s_4_eq_14916 : s 4 = 14916 := 
by
  sorry

end s_4_eq_14916_l291_291417


namespace quadrilateral_has_inscribed_circle_l291_291566

theorem quadrilateral_has_inscribed_circle
  (A1 A2 A3 A4 : Type)
  (r1 r2 r3 r4 d : ℝ)
  (h1 : r1 + r3 = r2 + r4)
  (h2 : r1 + r3 < d) :
  ∃ inscribed_circle, True :=
by 
-- The statement about existence of inscribed circle
sorry

end quadrilateral_has_inscribed_circle_l291_291566


namespace countSuperBalancedIntegers_l291_291086

def isSuperBalanced (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  let d₁ := n / 1000,
      d₂ := (n / 100) % 10,
      d₃ := (n / 10) % 10,
      d₄ := n % 10
  in d₁ + d₂ = 2 * (d₃ + d₄)

theorem countSuperBalancedIntegers : 
  { n : ℕ // isSuperBalanced n }.card = 274 :=
  sorry

end countSuperBalancedIntegers_l291_291086


namespace sum_of_roots_tan_quadratic_l291_291537

theorem sum_of_roots_tan_quadratic :
  (∑ x in {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ (tan x)^2 - 12 * tan x + 4 = 0}, x) = 3 * Real.pi := 
by
  sorry

end sum_of_roots_tan_quadratic_l291_291537


namespace plane_equation_l291_291061

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - 3 * t, 1 + s + 2 * t, 4 - s + t)

theorem plane_equation :
  ∃ (A B C D : ℤ), (∀ (x y z : ℝ), parametric_plane _ _ = (x, y, z) → (A * x + B * y + C * z + D = 0)) ∧
  (A = 3 ∧ B = -1 ∧ C = 7 ∧ D = -33) ∧
  (A > 0) ∧
  (∃ gcd : ℤ, is_gcd ( |A| ) ( |B| ) ( |C| ) ( |D| ) gcd ∧ gcd = 1) :=
sorry

end plane_equation_l291_291061


namespace range_of_a_l291_291974

theorem range_of_a (a x y : ℝ) (h1 : x - y = a + 3) (h2 : 2 * x + y = 5 * a) (h3 : x < y) : a < -3 :=
by
  sorry

end range_of_a_l291_291974


namespace minimum_modulus_of_z_l291_291562

noncomputable def Z : ℂ := sorry
noncomputable def complex_condition : Prop := abs (Z - (1 + complex.I)) = 1
noncomputable def min_modulus (Z : ℂ) : ℝ := abs Z

theorem minimum_modulus_of_z : complex_condition → min_modulus Z = real.sqrt 2 - 1 := by
  sorry

end minimum_modulus_of_z_l291_291562


namespace emilia_cartons_total_l291_291123

theorem emilia_cartons_total (strawberries blueberries supermarket : ℕ) (total_needed : ℕ)
  (h1 : strawberries = 2)
  (h2 : blueberries = 7)
  (h3 : supermarket = 33)
  (h4 : total_needed = strawberries + blueberries + supermarket) :
  total_needed = 42 :=
sorry

end emilia_cartons_total_l291_291123


namespace sin_cos_identity_proof_l291_291160

noncomputable def sin_cos_identity (α : ℝ) : Prop :=
  let t := -3 / 4 in
  tan α = t → sin α * (sin α - cos α) = 21 / 25

-- A statement of the mathematical proof problem.
theorem sin_cos_identity_proof (α : ℝ) (h : tan α = -3 / 4) : sin α * (sin α - cos α) = 21 / 25 :=
by
  have h_tan : tan α = -3 / 4 := h
  sorry

end sin_cos_identity_proof_l291_291160


namespace intersection_of_A_and_B_l291_291931

def A := {1, 2, 3}
def B := {x : ℤ | x^2 - 4 ≤ 0}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l291_291931


namespace gcd_of_60_and_75_l291_291393

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l291_291393


namespace construct_convex_quadrilateral_l291_291105

variables {ℝ : Type*} [linear_ordered_field ℝ]

-- Define the lengths of the sides of the quadrilateral
variables {AB BC CD DA KP : ℝ}

-- Assume the conditions
variables (hAB : 0 < AB) (hBC : 0 < BC) (hCD : 0 < CD) (hDA : 0 < DA) (hKP : 0 < KP)

-- Define the main theorem statement
theorem construct_convex_quadrilateral :
  ∃ (A B C D K P : Point),
    length A B = AB ∧ length B C = BC ∧ length C D = CD ∧ length D A = DA ∧
    midpoint A B = K ∧ midpoint C D = P ∧ length K P = KP ∧ convex_quadrilateral A B C D :=
begin
  sorry
end

end construct_convex_quadrilateral_l291_291105


namespace correct_statements_l291_291349

theorem correct_statements : ¬(∃ x : ℝ, sin x + cos x = 2) ∧
                            (∃ x : ℝ, sin x + 1 / sin x < 2) ∧
                            (∀ x : ℝ, 0 < x → x < π / 2 → tan x + 1 / tan x ≥ 2) ∧
                            (∃ x : ℝ, sin x + cos x = sqrt 2) := by
  sorry

end correct_statements_l291_291349


namespace find_negative_number_l291_291019

noncomputable def is_negative (x : ℝ) : Prop := x < 0

theorem find_negative_number : is_negative (-5) := by
  -- Proof steps would go here, but we'll skip them for now.
  sorry

end find_negative_number_l291_291019


namespace unique_triangle_determination_l291_291027

/--
Given the following conditions:
1. One angle and two sides; scalene triangle
2. One angle and the altitude to one of the sides; isosceles triangle
3. The radius of the circumscribed circle; regular pentagon (irrelevant)
4. Two sides and the included angle; right triangle
5. One side and the height; equilateral triangle

Prove that the combination which does not uniquely determine the indicated triangle is the one angle and the altitude to one of the sides in an isosceles triangle.
-/
theorem unique_triangle_determination (A B C D E : Prop)
  (hA : "one angle and two sides; scalene triangle" → unique_determination)
  (hB : ¬ ("one angle and the altitude to one of the sides; isosceles triangle" → unique_determination))
  (hC : "the radius of the circumscribed circle; regular pentagon" irrelevant)
  (hD : "two sides and the included angle; right triangle" → unique_determination)
  (hE : "one side and the height; equilateral triangle" → unique_determination) :
  B := sorry

end unique_triangle_determination_l291_291027


namespace net_increase_proof_l291_291275

def initial_cars := 50
def initial_motorcycles := 75
def initial_vans := 25

def car_arrival_rate : ℝ := 70
def car_departure_rate : ℝ := 40
def motorcycle_arrival_rate : ℝ := 120
def motorcycle_departure_rate : ℝ := 60
def van_arrival_rate : ℝ := 30
def van_departure_rate : ℝ := 20

def play_duration : ℝ := 2.5

def net_increase_car : ℝ := play_duration * (car_arrival_rate - car_departure_rate)
def net_increase_motorcycle : ℝ := play_duration * (motorcycle_arrival_rate - motorcycle_departure_rate)
def net_increase_van : ℝ := play_duration * (van_arrival_rate - van_departure_rate)

theorem net_increase_proof :
  net_increase_car = 75 ∧
  net_increase_motorcycle = 150 ∧
  net_increase_van = 25 :=
by
  -- Proof would go here.
  sorry

end net_increase_proof_l291_291275


namespace calories_in_piece_of_cake_l291_291793

-- Define the given conditions
def breakfast_calories : ℕ := 560
def lunch_calories : ℕ := 780
def pack_of_chips_calories : ℕ := 310
def bottle_of_coke_calories : ℕ := 215
def daily_caloric_limit : ℕ := 2500
def remaining_caloric_capacity : ℕ := 525

-- Define the question as a theorem
theorem calories_in_piece_of_cake :
  ∀ (x : ℕ),
    breakfast_calories + lunch_calories + pack_of_chips_calories + bottle_of_coke_calories + x 
    = daily_caloric_limit - remaining_caloric_capacity → 
    x = 110 :=
by {
  intro x,
  intro h,
  sorry
}

end calories_in_piece_of_cake_l291_291793


namespace simplify_expression_evaluate_at_2_l291_291324

theorem simplify_expression (a : ℤ) (h : a ≠ 0 ∧ a ≠ -1) :
  ((a - (2 * a - 1) / a) + (1 - a^2) / (a^2 + a)) = (a^2 - 3 * a + 2) / a :=
by sorry

theorem evaluate_at_2 :
  ((2 - (2 * 2 - 1) / 2) + (1 - 2^2) / (2^2 + 2)) = 0 :=
by trivial

end simplify_expression_evaluate_at_2_l291_291324


namespace power_of_a_point_l291_291786

theorem power_of_a_point {EF GH : Type} [Circle EF] [Circle GH] {Q : Point} 
  (EQ : ℝ) (FQ HQ : ℝ) (GQ : ℝ) (hEQ : EQ = 5) (hGQ : GQ = 7) 
  (hPowerOfPoint : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 7 / 5 := 
by
  rw [hEQ, hGQ] at hPowerOfPoint
  sorry

end power_of_a_point_l291_291786


namespace sum_of_intervals_l291_291541

-- Define the floor function and the given function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  let k := floor x in k * (2020 ^ (x - k) - 1)

-- Define the main theorem
theorem sum_of_intervals : 
  ∑ k in finset.range 2019, real.log (1 + 1 / k) / real.log 2020 = 1 := by
  sorry

end sum_of_intervals_l291_291541


namespace hexagon_coloring_l291_291511

-- Definitions based on conditions
variable (A B C D E F : ℕ)
variable (color : ℕ → ℕ)
variable (v1 v2 : ℕ)

-- The question is about the number of different colorings
theorem hexagon_coloring (h_distinct : ∀ (x y : ℕ), x ≠ y → color x ≠ color y) 
    (h_colors : ∀ (x : ℕ), x ∈ [A, B, C, D, E, F] → 0 < color x ∧ color x < 5) :
    4 * 3 * 3 * 3 * 3 * 3 = 972 :=
by
  sorry

end hexagon_coloring_l291_291511


namespace functional_equivalence_l291_291318

theorem functional_equivalence (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f(x + y) = f(x) + f(y)) ↔ (∀ x y : ℝ, f(x + y + xy) = f(x) + f(y) + f(xy)) :=
sorry

end functional_equivalence_l291_291318


namespace friends_picked_strawberries_with_Lilibeth_l291_291300

-- Define the conditions
def Lilibeth_baskets : ℕ := 6
def strawberries_per_basket : ℕ := 50
def total_strawberries : ℕ := 1200

-- Define the calculation of strawberries picked by Lilibeth
def Lilibeth_strawberries : ℕ := Lilibeth_baskets * strawberries_per_basket

-- Define the calculation of strawberries picked by friends
def friends_strawberries : ℕ := total_strawberries - Lilibeth_strawberries

-- Define the number of friends who picked strawberries
def friends_picked_with_Lilibeth : ℕ := friends_strawberries / Lilibeth_strawberries

-- The theorem we need to prove
theorem friends_picked_strawberries_with_Lilibeth : friends_picked_with_Lilibeth = 3 :=
by
  -- Proof goes here
  sorry

end friends_picked_strawberries_with_Lilibeth_l291_291300


namespace number_of_way_preferences_arranged_l291_291866

-- Number of songs
def numberOfSongs : ℕ := 5

-- Amy, Beth and Jo's preferences represent as sets
def AmyPref : Finset ℕ := sorry
def BethPref : Finset ℕ := sorry
def JoPref : Finset ℕ := sorry

-- Conditions
def no_song_liked_by_all_three (AmyPref BethPref JoPref : Finset ℕ) : Prop := 
  (AmyPref ∩ BethPref ∩ JoPref) = ∅

def at_least_one_song_liked_by_two_disliked_by_third (AmyPref BethPref JoPref : Finset ℕ) : Prop := 
  (AmyPref ∩ BethPref \ JoPref ≠ ∅) ∧
  (BethPref ∩ JoPref \ AmyPref ≠ ∅) ∧
  (JoPref ∩ AmyPref \ BethPref ≠ ∅)

def no_individual_likes_more_than_three (AmyPref BethPref JoPref : Finset ℕ) : Prop := 
  (AmyPref.card ≤ 3) ∧
  (BethPref.card ≤ 3) ∧
  (JoPref.card ≤ 3)

-- Main theorem statement
theorem number_of_way_preferences_arranged : 
    (∃ AmyPref BethPref JoPref : Finset ℕ, 
        AmyPref ∪ BethPref ∪ JoPref = (Finset.range numberOfSongs).attach ∧
        no_song_liked_by_all_three AmyPref BethPref JoPref ∧
        at_least_one_song_liked_by_two_disliked_by_third AmyPref BethPref JoPref ∧
        no_individual_likes_more_than_three AmyPref BethPref JoPref) →
    ∃ n, n = 1560 := 
    sorry

end number_of_way_preferences_arranged_l291_291866


namespace find_k_values_l291_291130

-- Definition of the problem statement
theorem find_k_values (k : ℝ) :
  (∃ a b : ℝ, 3 * a^2 + 2 * a + k = 0 ∧
               3 * b^2 + 2 * b + k = 0 ∧
               |a - b| = real.sqrt (a^2 + b^2)) 
  ↔ k = 0 ∨ k = -4 / 15 :=
sorry

end find_k_values_l291_291130


namespace square_root_of_259_21_l291_291852

theorem square_root_of_259_21 :
  ∃ (x : ℝ), x^2 = 259.21 ∧ (x = 16.1 ∨ x = -16.1) :=
by
  let table := λ (x : ℝ), x^2
  have h1 : table 16 = 256 := rfl
  have h2 : table 16.1 = 259.21 := rfl
  have h3 : table 16.2 = 262.44 := rfl
  have h4 : table 16.3 = 265.69 := rfl
  use 16.1
  split
  . exact h2
  . left; rfl
  sorry

end square_root_of_259_21_l291_291852


namespace minimum_sum_of_dimensions_of_3003_l291_291747

noncomputable def min_sum_of_dimensions (V : ℕ) : ℕ :=
  Inf {x : ℕ | ∃ (a b c : ℕ), x = a + b + c ∧ a * b * c = V ∧ 0 < a ∧ 0 < b ∧ 0 < c}

theorem minimum_sum_of_dimensions_of_3003 : min_sum_of_dimensions 3003 = 45 :=
by
  sorry

end minimum_sum_of_dimensions_of_3003_l291_291747


namespace find_largest_and_smallest_165_divisible_digit_number_l291_291900

def isDivisibleBy (m n : Nat) : Prop := n % m = 0

def isDigitSet (n : Nat) : Prop := 
  let digits := List.ofDigits (Nat.digits 10 n);
  digits.perm [0, 1, 2, 3, 4, 5, 6]

theorem find_largest_and_smallest_165_divisible_digit_number :
  ∀ n : Nat, isDigitSet n → isDivisibleBy 165 n →
  ∃ largest smallest : Nat,
    (isDigitSet largest ∧ isDivisibleBy 165 largest ∧ n = largest) ∧ 
    (isDigitSet smallest ∧ isDivisibleBy 165 smallest ∧ n = smallest) ∧ 
    largest = 6431205 ∧ smallest = 1042635 := by
  sorry

end find_largest_and_smallest_165_divisible_digit_number_l291_291900


namespace seventh_root_of_unity_sum_l291_291167

theorem seventh_root_of_unity_sum (z : ℂ) (h1 : z^7 = 1) (h2 : z ≠ 1) :
  z + z^2 + z^4 = (-1 + Complex.I * Real.sqrt 11) / 2 ∨ z + z^2 + z^4 = (-1 - Complex.I * Real.sqrt 11) / 2 := 
by sorry

end seventh_root_of_unity_sum_l291_291167


namespace count_two_digit_numbers_with_digit_8_l291_291990

theorem count_two_digit_numbers_with_digit_8 : 
  let two_digit_integers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let has_eight (n : ℕ) := n / 10 = 8 ∨ n % 10 = 8
  (two_digit_integers.filter has_eight).card = 18 :=
by
  let two_digit_integers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let has_eight (n : ℕ) := n / 10 = 8 ∨ n % 10 = 8
  show (two_digit_integers.filter has_eight).card = 18
  sorry

end count_two_digit_numbers_with_digit_8_l291_291990


namespace octal_to_base5_conversion_l291_291107

-- Define the octal to decimal conversion
def octalToDecimal (n : ℕ) : ℕ :=
  2 * 8^3 + 0 * 8^2 + 1 * 8^1 + 1 * 8^0

-- Define the base-5 number
def base5Representation : ℕ := 13113

-- Theorem statement
theorem octal_to_base5_conversion :
  octalToDecimal 2011 = base5Representation := 
sorry

end octal_to_base5_conversion_l291_291107


namespace square_division_l291_291323

theorem square_division (n : Nat) : (n > 5 → ∃ smaller_squares : List (Nat × Nat), smaller_squares.length = n ∧ (∀ s ∈ smaller_squares, IsSquare s)) ∧ (n = 2 ∨ n = 3 → ¬ ∃ smaller_squares : List (Nat × Nat), smaller_squares.length = n ∧ (∀ s ∈ smaller_squares, IsSquare s)) := 
by
  sorry

end square_division_l291_291323


namespace students_wearing_other_colors_l291_291038

-- Definitions according to the problem conditions
def total_students : ℕ := 900
def percentage_blue : ℕ := 44
def percentage_red : ℕ := 28
def percentage_green : ℕ := 10

-- Goal: Prove the number of students who wear other colors
theorem students_wearing_other_colors :
  (total_students * (100 - (percentage_blue + percentage_red + percentage_green))) / 100 = 162 :=
by
  -- Skipping the proof steps with sorry
  sorry

end students_wearing_other_colors_l291_291038


namespace original_weight_proof_l291_291051

-- Define the conditions and variables
def weight_after_division (W : ℝ) : ℝ := W / (W / 2)
def original_weight (W : ℝ) : ℝ := 24

-- State the theorem to be proved
theorem original_weight_proof (W : ℝ) (h : weight_after_division W = 12) : original_weight W = 24 := sorry

end original_weight_proof_l291_291051


namespace solve_for_a_l291_291234

theorem solve_for_a (a : ℚ) (h : a + a / 4 = 10 / 4) : a = 2 :=
sorry

end solve_for_a_l291_291234


namespace even_integers_count_l291_291985

def is_valid_digit (d : ℕ) : Prop := d ∈ {1, 3, 4, 5, 6, 8}
def is_even (n : ℕ) : Prop := n % 2 = 0
def digits_different (n : ℕ) : Prop := 
  let ds := List.ofDigits (Int.digits 10 n) in ds.nodup

def valid_set : Set ℕ := {1, 3, 4, 5, 6, 8}

noncomputable def valid_numbers := 
  { x | 300 ≤ x ∧ x < 800 ∧ 
    is_even x ∧ 
    (all_digits_in_set x valid_set) ∧
    (digits_different x) }

theorem even_integers_count : 
  (valid_numbers.satisfies) = 48
:= by sorry

end even_integers_count_l291_291985


namespace g_4_values_product_l291_291292

theorem g_4_values_product (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g (g x + y) = g x + g (g y + g (-x)) - x) :
  let m := (λ s : set ℝ, s.card) { y : ℝ | ∃ x : ℝ, g x = y },
      t := { y : ℝ | ∃ x : ℝ, g x = y }.sum id in
  m * t = -4 :=
by sorry

end g_4_values_product_l291_291292


namespace min_square_max_value_l291_291574

open Real

theorem min_square_max_value (a b c : ℝ) (λ : ℝ) (h : a^2 + b^2 + c^2 = λ) (hλ : 0 < λ) :
  ∃ x, (min ((a - b) ^ 2) ((b - c) ^ 2)).min ((c - a) ^ 2)) = λ / 2 :=
sorry

end min_square_max_value_l291_291574


namespace addition_result_l291_291875

theorem addition_result : 148 + 32 + 18 + 2 = 200 :=
by
  sorry

end addition_result_l291_291875


namespace transformer_minimum_load_l291_291356

-- Define the conditions as hypotheses
def running_current_1 := 40
def running_current_2 := 60
def running_current_3 := 25

def start_multiplier_1 := 2
def start_multiplier_2 := 3
def start_multiplier_3 := 4

def units_1 := 3
def units_2 := 2
def units_3 := 1

def starting_current_1 := running_current_1 * start_multiplier_1
def starting_current_2 := running_current_2 * start_multiplier_2
def starting_current_3 := running_current_3 * start_multiplier_3

def total_starting_current_1 := starting_current_1 * units_1
def total_starting_current_2 := starting_current_2 * units_2
def total_starting_current_3 := starting_current_3 * units_3

def total_combined_minimum_current_load := 
  total_starting_current_1 + total_starting_current_2 + total_starting_current_3

-- The theorem to prove that the total combined minimum current load is 700A
theorem transformer_minimum_load : total_combined_minimum_current_load = 700 := by
  sorry

end transformer_minimum_load_l291_291356


namespace clothing_tax_rate_l291_291711

-- Definitions based on the identified conditions
def total_amount : ℝ := 100 -- Assuming the total amount spent is $100 for simplicity

def clothing_spent : ℝ := 0.5 * total_amount -- 50% on clothing
def food_spent : ℝ := 0.1 * total_amount -- 10% on food
def other_items_spent : ℝ := 0.4 * total_amount -- 40% on other items

def tax_on_clothing (C : ℝ) : ℝ := (C / 100) * clothing_spent -- tax rate on clothing as C%
def tax_on_other_items : ℝ := 0.08 * other_items_spent -- 8% tax on other items
def total_tax (C : ℝ) : ℝ := tax_on_clothing C + tax_on_other_items -- Total tax paid

-- Total tax is 5.2% of total amount before taxes
axiom total_tax_is_correct (C : ℝ) : total_tax C = 0.052 * total_amount

-- Prove that the tax rate on clothing is 4%
theorem clothing_tax_rate : ∃ C : ℝ, (tax_on_clothing C + tax_on_other_items = 0.052 * total_amount) ∧ C = 4 := by
  sorry

end clothing_tax_rate_l291_291711


namespace pencils_given_out_l291_291360

theorem pencils_given_out
  (num_children : ℕ)
  (pencils_per_student : ℕ)
  (dozen : ℕ)
  (children : num_children = 46)
  (dozen_def : dozen = 12)
  (pencils_def : pencils_per_student = 4 * dozen) :
  num_children * pencils_per_student = 2208 :=
by {
  sorry
}

end pencils_given_out_l291_291360


namespace books_probability_l291_291700

/-- Given a set of 12 books where two students, Harold and Betty, each randomly select 6 books,
prove that the probability that they select exactly 3 books in common is 5/23. -/
theorem books_probability :
  let total_outcomes := (Nat.choose 12 6) * (Nat.choose 12 6),
      successful_outcomes := (Nat.choose 12 3) * (Nat.choose 9 3) * (Nat.choose 6 3) in
  (successful_outcomes : ℚ) / total_outcomes = 5/23 :=
by
  sorry

end books_probability_l291_291700


namespace average_weight_l291_291742

variable (A B C : ℕ)

theorem average_weight (h1 : A + B = 140) (h2 : B + C = 100) (h3 : B = 60) :
  (A + B + C) / 3 = 60 := 
sorry

end average_weight_l291_291742


namespace area_of_region_l291_291384

/-- The given equation defines a region. We need to prove that the area of this region is equal to 25/4 * pi. --/
theorem area_of_region (x y : ℝ) 
  (h : x^2 + y^2 + 3*x - 4*y = 4) : 
  (let r := 5/2 in π * r^2 = 25/4 * π) :=
begin
  sorry
end

end area_of_region_l291_291384


namespace am_gm_difference_lt_half_l291_291926

theorem am_gm_difference_lt_half (a : ℝ) (ha : 0 < a) : 
  let am := a + 1 / 2,
      gm := Real.sqrt (a * (a + 1))
  in am - gm < 1 / 2 := 
by
  let am := a + 1 / 2
  let gm := Real.sqrt (a * (a + 1))
  sorry

end am_gm_difference_lt_half_l291_291926


namespace ratio_of_riding_to_total_l291_291055

-- Define the primary conditions from the problem
variables (H R W : ℕ)
variables (legs_on_ground : ℕ := 50)
variables (total_owners : ℕ := 10)
variables (legs_per_horse : ℕ := 4)
variables (legs_per_owner : ℕ := 2)

-- Express the conditions
def conditions : Prop :=
  (legs_on_ground = 6 * W) ∧
  (total_owners = H) ∧
  (H = R + W) ∧
  (H = 10)

-- Define the theorem with the given conditions and prove the required ratio
theorem ratio_of_riding_to_total (H R W : ℕ) (h : conditions H R W) : R / 10 = 1 / 5 := by
  sorry

end ratio_of_riding_to_total_l291_291055


namespace compare_dog_area_l291_291438

-- Define the conditions and areas for each setup
def rope_length : ℝ := 10
def shed_length : ℝ := 20
def shed_width : ℝ := 10

-- Define the area accessible to the dog in each setup
def area_setup1 : ℝ := (1/2) * π * rope_length^2
def area_setup2 : ℝ := (3/4) * π * rope_length^2

-- State the theorem to prove the difference in area
theorem compare_dog_area : area_setup2 - area_setup1 = 25 * π := 
by {
    -- Placeholder for the proof
    sorry
}

end compare_dog_area_l291_291438


namespace prove_negative_ln_abs_sin_l291_291478

noncomputable def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
noncomputable def is_monotonically_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop := ∀ ⦃a b⦄, a ∈ I → b ∈ I → a < b → f a > f b
noncomputable def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def sin_abs (x : ℝ) : ℝ := Real.sin (Real.abs x)
def cos_abs (x : ℝ) : ℝ := Real.cos (Real.abs x)
def abs_tan (x : ℝ) : ℝ := Real.abs (Real.tan x)
def neg_ln_abs_sin (x : ℝ) : ℝ := -Real.log (Real.abs (Real.sin x))

theorem prove_negative_ln_abs_sin :
  is_periodic neg_ln_abs_sin Real.pi ∧
  is_monotonically_decreasing neg_ln_abs_sin (set.Ioo 0 (Real.pi / 2)) ∧
  is_odd neg_ln_abs_sin :=
by
  sorry

end prove_negative_ln_abs_sin_l291_291478


namespace count_4_digit_numbers_using_2033_l291_291598

-- Define the concepts from the problem
def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def uses_digits_exactly (n : ℕ) (digits : List ℕ) : Prop := 
  digits ~ (n.digits 10) -- n.digits 10 gives digit list of n in base 10 and ~ represents bag equality (permutation)

theorem count_4_digit_numbers_using_2033 : 
  (finset.range 10000).filter (λ n => is_4_digit n ∧ uses_digits_exactly n [2, 0, 3, 3]).card = 9 := 
sorry

end count_4_digit_numbers_using_2033_l291_291598


namespace line_intersects_midpoint_l291_291230

-- Definitions of the points
def point1 : ℝ × ℝ := (1, 4)
def point2 : ℝ × ℝ := (7, 10)

-- Midpoint of the segment
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Proof problem statement
theorem line_intersects_midpoint : 
  ∃ c : ℝ, (let m := midpoint point1 point2 in 2 * m.1 + m.2 = c) ∧ c = 15 :=
by
  sorry

end line_intersects_midpoint_l291_291230


namespace profit_percentage_previous_year_l291_291634

variable (R P : ℝ)

-- Conditions
def previous_year_revenue := R
def previous_year_profit_percentage := P / 100 * R
def revenue_2009 := 0.8 * R
def profit_2009 := 0.18 * revenue_2009
def profit_comparison := profit_2009 = 1.44 * (P / 100) * R

theorem profit_percentage_previous_year :
  profit_comparison → P = 10 := by
  intro h
  have h1 : profit_2009 = 0.144 * R := by rw [revenue_2009, mul_assoc, mul_comm 0.18 0.8, mul_assoc, mul_comm]
  rw [h1, mul_assoc, mul_comm 1.44 (P / 100), ← mul_assoc, div_eq_inv_mul, mul_inv_cancel_right₀, mul_assoc] at h
  linarith
  sorry

end profit_percentage_previous_year_l291_291634


namespace maximum_xyz_l291_291677

theorem maximum_xyz {x y z : ℝ} (hx: 0 < x) (hy: 0 < y) (hz: 0 < z) 
  (h : (x * y) + z = (x + z) * (y + z)) : xyz ≤ (1 / 27) :=
by
  sorry

end maximum_xyz_l291_291677


namespace cash_realized_without_brokerage_l291_291340

theorem cash_realized_without_brokerage (cash_with_brokerage : ℝ) (brokerage_rate : ℝ) 
  (h_cash_with_brokerage : cash_with_brokerage = 120.50) 
  (h_brokerage_rate : brokerage_rate = 1 / 4) : 
  let brokerage_amount := (brokerage_rate / 100) * cash_with_brokerage in
  cash_with_brokerage + brokerage_amount = 120.80 :=
by
  sorry

end cash_realized_without_brokerage_l291_291340


namespace problem_problem_trapezoid_l291_291181

noncomputable def circle_M (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 2) ^ 2 = 2
noncomputable def circle_N (x y : ℝ) : Prop := x ^ 2 + (y - 8) ^ 2 = 40
def line_through_origin (k : ℝ) : (ℝ × ℝ) → Prop := λ p, p.2 = k * p.1
def perpendicular_lines (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1
def trapezoid (A B C D : ℝ × ℝ) : Prop := (A.2 - B.2) / (A.1 - B.1) = (C.2 - D.2) / (C.1 - D.1)

theorem problem (k : ℝ) :
  (∃ A B : ℝ × ℝ, circle_M A.1 A.2 ∧ circle_M B.1 B.2 ∧ line_through_origin k A ∧ line_through_origin k B ∧ A ≠ B) →
  (∃ C D : ℝ × ℝ, circle_N C.1 C.2 ∧ circle_N D.1 D.2 ∧ line_through_origin (-1/k) C ∧ line_through_origin (-1/k) D ∧ C ≠ D) →
  2 - sqrt 3 < k ∧ k < sqrt 15 / 3 :=
sorry

theorem problem_trapezoid (k : ℝ) (A B C D : ℝ × ℝ) :
  circle_M A.1 A.2 ->
  circle_M B.1 B.2 ->
  circle_N C.1 C.2 ->
  circle_N D.1 D.2 ->
  line_through_origin k A ->
  line_through_origin k B ->
  line_through_origin (-1 / k) C ->
  line_through_origin (-1 / k) D ->
  A ≠ B -> 
  C ≠ D ->
  trapezoid A B C D ->
  k = 1 :=
sorry

end problem_problem_trapezoid_l291_291181


namespace lattice_sum_1990_l291_291310

def gcd (a b : Nat) : Nat := Nat.gcd a b

def lattice_points (n : Nat) : Nat := 
  if gcd n (n + 3) = 1 then 0 else 2

def sum_lattice_points (k : Nat) : Nat :=
  (List.range k).sumBy lattice_points

theorem lattice_sum_1990 : sum_lattice_points 1990 = 1326 := by
  sorry

end lattice_sum_1990_l291_291310


namespace probability_defective_is_three_tenths_l291_291477

open Classical

noncomputable def probability_of_defective_product (total_products defective_products: ℕ) : ℝ :=
  (defective_products * 1.0) / (total_products * 1.0)

theorem probability_defective_is_three_tenths :
  probability_of_defective_product 10 3 = 3 / 10 := by
  sorry

end probability_defective_is_three_tenths_l291_291477


namespace inequality_must_hold_l291_291560

section
variables {a b c : ℝ}

theorem inequality_must_hold (h : a > b) : (a - b) * c^2 ≥ 0 :=
sorry
end

end inequality_must_hold_l291_291560


namespace number_subtracted_from_15n_l291_291621

theorem number_subtracted_from_15n (m n : ℕ) (h_pos_n : 0 < n) (h_pos_m : 0 < m) (h_eq : m = 15 * n - 1) (h_remainder : m % 5 = 4) : 1 = 1 :=
by
  sorry

end number_subtracted_from_15n_l291_291621


namespace ratio_DE_over_TP_ratio_TE_over_PD_l291_291829

-- Given conditions for the problem
variables {P T M D E : Type}
variables [IsTriangle P T M] (circle : Circle P T) 
variables (intersects_MP_at_D : IntersectsAtDistinctPoints circle M P D) 
variables (intersects_MT_at_E : IntersectsAtDistinctPoints circle M T E)
variables (area_MDE_over_MPT : area_ratio (TriangleArea M D E) (TriangleArea M P T) = 1 / 4)
variables (area_MDT_over_MEP : area_ratio (TriangleArea M D T) (TriangleArea M E P) = 4 / 9)

-- Part (a)
theorem ratio_DE_over_TP : ratio_of_length (SegmentLength D E) (SegmentLength T P) = 1 / 2 := by
  sorry

-- Part (b)
theorem ratio_TE_over_PD : ratio_of_length (SegmentLength T E) (SegmentLength P D) = 1 / 4 := by
  sorry

end ratio_DE_over_TP_ratio_TE_over_PD_l291_291829


namespace biquadratic_exactly_two_distinct_roots_l291_291545

theorem biquadratic_exactly_two_distinct_roots {a : ℝ} :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^4 + a*x1^2 + a - 1 = 0) ∧ (x2^4 + a*x2^2 + a - 1 = 0) ∧
   ∀ x, x^4 + a*x^2 + a - 1 = 0 → (x = x1 ∨ x = x2)) ↔ a < 1 :=
by
  sorry

end biquadratic_exactly_two_distinct_roots_l291_291545


namespace cone_radius_l291_291948

open Real

theorem cone_radius
  (l : ℝ) (L : ℝ) (h_l : l = 5) (h_L : L = 15 * π) :
  ∃ r : ℝ, L = π * r * l ∧ r = 3 :=
by
  sorry

end cone_radius_l291_291948


namespace solution_set_is_complete_l291_291128

-- Define the problem
def eq1 (x y z : ℝ) := 3 * (x^2 + y^2 + z^2) = 1
def eq2 (x y z : ℝ) := x^2 * y^2 + y^2 * z^2 + z^2 * x^2 = x * y * z * (x + y + z)^2

-- Define the set of solutions
def solutions : set (ℝ × ℝ × ℝ) :=
  { (0, 0, sqrt(3)/3), (0, 0, -sqrt(3)/3),
    (0, sqrt(3)/3, 0), (0, -sqrt(3)/3, 0),
    (sqrt(3)/3, 0, 0), (-sqrt(3)/3, 0, 0),
    (1/3, 1/3, 1/3), (-1/3, -1/3, -1/3) }

-- Statement to prove
theorem solution_set_is_complete :
  ∀ (x y z : ℝ), eq1 x y z ∧ eq2 x y z ↔ (x, y, z) ∈ solutions := by
  sorry

end solution_set_is_complete_l291_291128


namespace total_palindromic_times_l291_291049

def is_palindrome (s : string) : Prop :=
  s = s.reverse

def valid_hour_palindrome (h : ℕ) : Prop :=
  h < 24 ∧ is_palindrome (string.pad_zero 2 (h.repr))

def valid_minute_palindrome (m : ℕ) : Prop :=
  m < 60 ∧ is_palindrome (string.pad_zero 2 (m.repr))

def count_palindromic_times : ℕ :=
  let three_digit_palindromes := 10 * 6 in
  let four_digit_palindromes := 23 in
  three_digit_palindromes + four_digit_palindromes

theorem total_palindromic_times : count_palindromic_times = 83 :=
  by sorry

end total_palindromic_times_l291_291049


namespace smallest_y_g_max_val_l291_291498

noncomputable def g (y : ℝ) : ℝ :=
  Real.sin (y / 5) + Real.sin (y / 7)

theorem smallest_y_g_max_val :
  ∃ y : ℝ, (y > 0) ∧ (y = 13230) ∧ (∀ ε > 0, g (y - ε) < g y ∧ g y > g (y + ε)) :=
begin
  sorry
end

end smallest_y_g_max_val_l291_291498


namespace f_is_odd_l291_291656

def f (x : ℝ) : ℝ := 1 / (3 ^ x - 1) + 1 / 3

theorem f_is_odd : ∀ x : ℝ, f x = -f (-x) := by
  sorry

end f_is_odd_l291_291656


namespace valid_configurations_count_l291_291604

noncomputable def is_valid_configuration (a b c d e f g h i : ℕ) : Prop :=
  a = 9 ∧ i = 1 ∧ 
  b ∈ {6, 7, 8} ∧ d ∈ {6, 7, 8} ∧ 
  f ∈ {2, 3, 4} ∧ h ∈ {2, 3, 4} ∧ 
  {a, b, c, d, e, f, g, h, i}.toFinset = {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem valid_configurations_count : 
  (Finset.univ.filter (λ t : Fin 9 → ℕ, is_valid_configuration (t 0) (t 1) (t 2) (t 3) (t 4) (t 5) (t 6) (t 7) (t 8))).card = 42 := 
  sorry

end valid_configurations_count_l291_291604


namespace exists_pair_who_spray_each_other_exists_student_not_targeted_l291_291815

variable (n : ℕ) 
variable (students : Fin (2 * n + 1) → ℝ × ℝ) -- students represented by their positions
variable (dist : (ℝ × ℝ) → (ℝ × ℝ) → ℝ) -- distance function 

-- Condition: distances between any two students are all different
axiom distinct_distances : ∀ i j : Fin (2 * n + 1), i ≠ j → dist (students i) (students j) ≠ dist (students i) (students j)

-- Condition: each student sprays the closest student with a water gun
noncomputable def closest_student (i : Fin (2 * n + 1)) : Fin (2 * n + 1) :=
  Fin.find (λ j, (j ≠ i) ∧ ∀ k, k ≠ i → dist (students i) (students j) < dist (students i) (students k))

-- Problem statement
theorem exists_pair_who_spray_each_other :
  ∃ a b : Fin (2 * n + 1), a ≠ b ∧ closest_student n students dist a = b ∧ closest_student n students dist b = a := sorry

theorem exists_student_not_targeted :
  ∃ a : Fin (2 * n + 1), ∀ b : Fin (2 * n + 1), closest_student n students dist b ≠ a := sorry

end exists_pair_who_spray_each_other_exists_student_not_targeted_l291_291815


namespace corrected_scores_correct_l291_291432

variables (n : ℕ) (initial_avg initial_var : ℝ) (scores : Fin n → ℝ)
variables (A_rec A_actual B_rec B_actual : ℝ)

/-- 
Given: 
- the number of students n,
- initial average score initial_avg,
- initial variance initial_var,
- students' recorded scores in an array scores,
- student A's recorded and actual score A_rec and A_actual,
- student B's recorded and actual score B_rec and B_actual,

Prove that: 
- the corrected average score remains the same,
- the corrected variance is 50.
-/
theorem corrected_scores_correct (n_ : n = 48)
 (initial_avg_ : initial_avg = 70)
 (initial_var_ : initial_var = 75)
 (A_rec_ : A_rec = 50)
 (A_actual_ : A_actual = 80)
 (B_rec_ : B_rec = 100)
 (B_actual_ : B_actual = 70)
 (h1 : scores.sum = 48 * 70) :
  let T := (48 * 70) - 50 - 100 + 80 + 70 in
  let avg' := T / n in
  avg' = 70 ∧
  let Δss := (80 - 70)^2 + (70 - 70)^2 - (50 - 70)^2 - (100 - 70)^2 in 
  let ss_initial := 75 * 48 in
  let ss_corrected := ss_initial + Δss in
  let var' := ss_corrected / n in
  var' = 50 := 
by
  sorry

end corrected_scores_correct_l291_291432


namespace eccentricity_of_ellipse_l291_291938

-- Defining the parameters and conditions of the problem
variables {a b c : ℝ} (e : ℝ)
variable (h1 : a > b > 0)
variable (h2 : b^2 = a^2 - c^2)
variable (h3 : 2 * c = 2 * (b^2 / a))

-- The goal to prove
theorem eccentricity_of_ellipse : e = (Real.sqrt 5 - 1) / 2 :=
begin
  sorry
end

end eccentricity_of_ellipse_l291_291938


namespace median_of_data_set_l291_291451

-- Definitions of the conditions.
def data_set : list ℕ := [6, 5, 7, 6, 6]

-- The statement to be proven: the median of the data set is 6.
theorem median_of_data_set : median data_set = 6 :=
by sorry

end median_of_data_set_l291_291451


namespace common_root_exists_l291_291116

open Polynomial

noncomputable def p (a : ℝ) : ℝ[X] := X^2 + C a * X + 1
noncomputable def q (a : ℝ) : ℝ[X] := X^2 + X + C a

theorem common_root_exists (a : ℝ) :
  (∃ x : ℝ, p(a).eval x = 0 ∧ q(a).eval x = 0) ↔ (a = 1 ∨ a = -2) := by
  sorry

end common_root_exists_l291_291116


namespace sum_of_decimals_as_fraction_simplified_fraction_final_sum_as_fraction_l291_291516

theorem sum_of_decimals_as_fraction:
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008) = (45678 / 100000) := by
  sorry

theorem simplified_fraction:
  (45678 / 100000) = (22839 / 50000) := by
  sorry

theorem final_sum_as_fraction:
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008) = (22839 / 50000) := by
  have h1 := sum_of_decimals_as_fraction
  have h2 := simplified_fraction
  rw [h1, h2]
  sorry

end sum_of_decimals_as_fraction_simplified_fraction_final_sum_as_fraction_l291_291516


namespace smallest_moves_to_balance_grid_l291_291263

-- Definitions of the grid and the counting functions
def Grid := Array (Array (Option Char))

def count_a_in_row (g: Grid) (r : Nat): Nat :=
  (g[r]).countp (fun c => c == some 'a')

def count_a_in_col (g: Grid) (c : Nat): Nat :=
  (Array.range 5).countp (fun r => g[r][c] == some 'a')

-- Initial grid configuration
def initial_grid : Grid :=
  #[#[some 'a', some 'a', some 'a', some 'a', none ],
    #[some 'a', some 'a', some 'a', some 'a', none ],
    #[none, none, none, none, none ],
    #[none, none, none, none, none ],
    #[none, none, none, none, none ]]

-- Goal Verification
def goal_met (g: Grid): Bool :=
  (Array.range 5).all (fun r => count_a_in_row g r == 3) ∧ 
  (Array.range 5).all (fun c => count_a_in_col g c == 3)

-- The theorem statement
theorem smallest_moves_to_balance_grid (g : Grid) :
  g = initial_grid → 
  (∃ g', (∑ i in [0,1,2,3,4], count_a_in_row g' i = 3) ∧ 
           (∑ j in [0,1,2,3,4], count_a_in_col g' j = 3) ∧ 
           (operations_to_balance g initial_grid g' = 2)) := sorry

end smallest_moves_to_balance_grid_l291_291263


namespace problem1_problem2_l291_291590

noncomputable def f1 (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 2)

noncomputable def f2 (x : ℝ) (m : ℝ) : ℝ := abs (x - 1) + abs (2 * x + m)

theorem problem1 : ∀ x : ℝ, (-4 / 3 ≤ x ∧ x ≤ 0) → f1 x ≤ 3 :=
by {
  intros x h,
  sorry
}

theorem problem2 : ∀ m : ℝ, (-3 ≤ m ∧ m ≤ 2) →
  ∃ x ∈ Icc 0 1, f2 x m ≤ abs (2 * x - 3) :=
by {
  intros m h,
  sorry
}

end problem1_problem2_l291_291590


namespace max_det_matrix_l291_291283

/-- Define the vector v. -/
def v : ℝ × ℝ × ℝ := (2, 1, -1)

/-- Define the vector w. -/
def w : ℝ × ℝ × ℝ := (1, 0, 3)

/-- The cross product of vectors v and w. -/
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.2 * b.2.1 - a.1.1 * b.2.2, a.1.1 * b.2.2 - a.2.1 * b.1.2, a.1.1 * b.2.1 - a.2.1 * b.2.2)

/-- Define the cross product v × w. -/
def v_cross_w : ℝ × ℝ × ℝ := cross_product v w

/-- State that u is a unit vector. -/
axiom unit_vector (u : ℝ × ℝ × ℝ) : ∥u∥ = 1

/-- Prove the largest possible determinant of the matrix with columns u, v, and w is sqrt(59). -/
theorem max_det_matrix (u : ℝ × ℝ × ℝ) (hv : v = (2, 1, -1)) (hw : w = (1, 0, 3)) :
  ∥v_cross_w∥ = real.sqrt 59 := sorry

end max_det_matrix_l291_291283


namespace grey_pairs_coincide_l291_291507

theorem grey_pairs_coincide (h₁ : 4 = orange_count / 2) 
                                (h₂ : 6 = green_count / 2)
                                (h₃ : 9 = grey_count / 2)
                                (h₄ : 3 = orange_pairs)
                                (h₅ : 4 = green_pairs)
                                (h₆ : 1 = orange_grey_pairs) :
    grey_pairs = 6 := by
  sorry

noncomputable def half_triangle_counts : (ℕ × ℕ × ℕ) := (4, 6, 9)

noncomputable def triangle_pairs : (ℕ × ℕ × ℕ) := (3, 4, 1)

noncomputable def prove_grey_pairs (orange_count green_count grey_count : ℕ)
                                   (orange_pairs green_pairs orange_grey_pairs : ℕ) : ℕ :=
  sorry

end grey_pairs_coincide_l291_291507


namespace pentagon_area_sum_l291_291347

theorem pentagon_area_sum (a b c d e m n : ℕ)
  (h1 : a = 2) (h2 : b = 2) (h3 : c = 2) (h4 : d = 2) (h5 : e = 2)
  (h6 : m = 11) (h7 : n = 12)
  (area_eq : sqrt m + sqrt n = sqrt(27))
  : m + n = 23 :=
begin
  sorry
end

end pentagon_area_sum_l291_291347


namespace spherical_distance_l291_291198

def earth_radius (R : ℝ) : Prop := R > 0
def latitude (α : ℝ) : Prop := 0 ≤ α ∧ α ≤ π / 2
def arc_length (R α : ℝ) : Prop := π * R * cos α

theorem spherical_distance (R α : ℝ) (hR : earth_radius R) (hα : latitude α) (hArc : arc_length R α):
  (π - 2 * α) * R = π * R - 2 * α * R := by sorry

end spherical_distance_l291_291198


namespace problem_statement_l291_291148

def base7_representation (n : ℕ) : ℕ :=
  let rec digits (n : ℕ) (acc : ℕ) (power : ℕ) : ℕ :=
    if n = 0 then acc
    else digits (n / 7) (acc + (n % 7) * power) (power * 10)
  digits n 0 1

def even_digits_count (n : ℕ) : ℕ :=
  let rec count (n : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else let d := n % 10 in
         count (n / 10) (if d % 2 = 0 then acc + 1 else acc)
  count n 0

theorem problem_statement : even_digits_count (base7_representation 528) = 0 := sorry

end problem_statement_l291_291148


namespace cistern_filling_time_l291_291830

theorem cistern_filling_time
  (rate_A_fills : ℝ := 1 / 5) 
  (rate_B_fills : ℝ := 1 / 7) 
  (rate_C_fills : ℝ := 1 / 9) 
  (rate_D_empties : ℝ := 1 / 12) 
  (rate_E_empties : ℝ := 1 / 15) :
  let net_rate := rate_A_fills + rate_B_fills + rate_C_fills - rate_D_empties - rate_E_empties in
  1260 / net_rate = 1260 / 383 := 
by { sorry }

end cistern_filling_time_l291_291830


namespace find_single_digit_A_l291_291439

theorem find_single_digit_A (A : ℕ) (h1 : 0 ≤ A) (h2 : A < 10) (h3 : (10 * A + A) * (10 * A + A) = 5929) : A = 7 :=
sorry

end find_single_digit_A_l291_291439


namespace monotonically_increasing_intervals_l291_291587

noncomputable def A : ℝ := 2
noncomputable def ω : ℝ := 2
noncomputable def φ : ℝ := π / 6

def f (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem monotonically_increasing_intervals :
  (∀ x ∈ Icc (0 : ℝ) (π / 6), deriv f x > 0) ∧
  (∀ x ∈ Icc (2 * π / 3 : ℝ) π, deriv f x > 0) :=
sorry

end monotonically_increasing_intervals_l291_291587


namespace minimum_additional_coins_l291_291859

theorem minimum_additional_coins
  (friends : ℕ) (initial_coins : ℕ)
  (h_friends : friends = 15) (h_coins : initial_coins = 100) :
  ∃ additional_coins : ℕ, additional_coins = 20 :=
by
  have total_needed_coins : ℕ := (friends * (friends + 1)) / 2
  have total_coins : ℕ := initial_coins
  have additional_coins_needed : ℕ := total_needed_coins - total_coins
  have h_additional_coins : additional_coins_needed = 20 := by calculate 
  -- Finishing the proof with the result we calculated
  use additional_coins_needed
  exact h_additional_coins

end minimum_additional_coins_l291_291859


namespace number_of_even_digits_in_base7_of_528_l291_291137

/-
  Define the base-7 representation of a number and a predicate to count even digits.
-/

-- Definition of base-7 digit representation
def base7_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else (List.unfoldr (λ n, if n = 0 then Option.none else some (n % 7, n / 7)) n).reverse

-- Predicate to check if a digit is even
def is_even (d : ℕ) : Bool := d % 2 = 0

-- Counting the even digits in base-7 representation
def count_even_digits_in_base7 (n : ℕ) : ℕ :=
  (base7_repr n).countp is_even

-- The target theorem to prove
theorem number_of_even_digits_in_base7_of_528 : count_even_digits_in_base7 528 = 0 :=
by sorry

end number_of_even_digits_in_base7_of_528_l291_291137


namespace hyperbola_equilateral_triangle_find_coordinates_l291_291591

theorem hyperbola_equilateral_triangle (P Q R : ℝ × ℝ) (C1 C2 : ℝ → Prop) 
(hyperbola : ∀ (x y : ℝ), x * y = 1 → C1 (x, y) ∨ C2 (x, y))
(hyperbola_cond : ∀ (x y : ℝ), C1 (x, y) ∨ C2 (x, y) → x * y = 1)
(equilateral_triangle : ∀ (P1 P2 P3 : ℝ × ℝ), P1 ≠ P2 ∧ P2 ≠ P3 ∧ P1 ≠ P3 → 
(((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2) = ((P2.1 - P3.1)^2 + (P2.2 - P3.2)^2) ∧ 
((P2.1 - P3.1)^2 + (P2.2 - P3.2)^2) = ((P1.1 - P3.1)^2 + (P1.2 - P3.2)^2)))
(hP : C2 (-1, -1)) (hQ : C1 Q) (hR : C1 R) :
¬ (C1 P ∧ C1 Q ∧ C1 R) :=
sorry

theorem find_coordinates (C1 C2 : ℝ → Prop) 
(hyperbola : ∀ (x y : ℝ), x * y = 1 → C1 (x, y) ∨ C2 (x, y))
(hyperbola_cond : ∀ (x y : ℝ), C1 (x, y) ∨ C2 (x, y) → x * y = 1)
(equilateral_triangle_cond : ∀ (P Q R : ℝ × ℝ), 
((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) ∧ 
((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = ((P.1 - R.1)^2 + (P.2 - R.2)^2)) 
(P : ℝ × ℝ) (hP : C2 P) : 
∃ Q R : ℝ × ℝ, C1 Q ∧ C1 R ∧ P ≠ Q ∧ P ≠ R ∧ Q ≠ R ∧ Q = (2 - real.sqrt 3, 2 + real.sqrt 3) ∧ R = (2 + real.sqrt 3, 2 - real.sqrt 3) :=
sorry

end hyperbola_equilateral_triangle_find_coordinates_l291_291591


namespace num_bounces_l291_291710

theorem num_bounces (length width : ℕ) (angle : ℕ) (P S : ℕ × ℕ)
  (h_length : length = 5) (h_width : width = 2) (h_angle : angle = 45) 
  (h_P : P = (0, 0)) (h_S : S = (5, 0)) :
  num_bounces length width angle P S = 5 :=
sorry

end num_bounces_l291_291710


namespace correct_equation_l291_291266

theorem correct_equation (x : ℕ) : 8 * x - 3 = 7 * x + 4 :=
by sorry

end correct_equation_l291_291266


namespace triangle_area_l291_291490

theorem triangle_area (f : ℝ → ℝ) (x1 x2 yIntercept base height area : ℝ)
  (h1 : ∀ x, f x = (x - 4)^2 * (x + 3))
  (h2 : f 0 = yIntercept)
  (h3 : x1 = -3)
  (h4 : x2 = 4)
  (h5 : base = x2 - x1)
  (h6 : height = yIntercept)
  (h7 : area = 1/2 * base * height) :
  area = 168 := sorry

end triangle_area_l291_291490


namespace actual_time_when_car_clock_is_8PM_l291_291744

noncomputable def car_clock_time : ℝ → ℝ := λ t, t + 10/540 * t

theorem actual_time_when_car_clock_is_8PM :
  let initial_time_real := 8 * 60 -- 8:00 AM in minutes
  let final_time_car := 20 * 60 -- 8:00 PM in minutes
  let gain_rate := (1710 - 8 * 60) / (1700 - 8 * 60) -- rate of gain
  let elapsed_real_time := (final_time_car - initial_time_real) / gain_rate + initial_time_real in
  elapsed_real_time = 7 * 60 + 47 := 
by 
  let initial_time_real := 8 * 60
  let final_time_car := 20 * 60
  let gain_rate := (1710 - 8 * 60) / (1700 - 8 * 60)
  let elapsed_real_time := (final_time_car - initial_time_real) / gain_rate + initial_time_real
  sorry

end actual_time_when_car_clock_is_8PM_l291_291744


namespace solve_system_of_equations_l291_291334

theorem solve_system_of_equations :
  (∃ x y : ℚ, 2 * x + 4 * y = 9 ∧ 3 * x - 5 * y = 8) ↔ 
  (∃ x y : ℚ, x = 7 / 2 ∧ y = 1 / 2) := by
  sorry

end solve_system_of_equations_l291_291334


namespace solve_system_of_equations_l291_291333

variables {x y : ℝ}
def system_of_equations : Prop :=
  ( (x - y) / 2 - (x + y) / 4 = -1 ) ∧ ( x + y = -8 )

theorem solve_system_of_equations (hx : x = -7) (hy : y = -1) : system_of_equations :=
by {
  rw [hx, hy],
  split;
  norm_num,
}

end solve_system_of_equations_l291_291333


namespace option_C_correct_l291_291801

theorem option_C_correct (a b : ℝ) : 
  (1 / (b / a) * (a / b) = a^2 / b^2) :=
sorry

end option_C_correct_l291_291801


namespace find_constants_l291_291214

def f (x : ℝ) (a : ℝ) : ℝ := 2 * x ^ 3 + a * x
def g (x : ℝ) (b c : ℝ) : ℝ := b * x ^ 2 + c
def f' (x : ℝ) (a : ℝ) : ℝ := 6 * x ^ 2 + a
def g' (x : ℝ) (b : ℝ) : ℝ := 2 * b * x

theorem find_constants (a b c : ℝ) :
  f 2 a = 0 ∧ g 2 b c = 0 ∧ f' 2 a = g' 2 b →
  a = -8 ∧ b = 4 ∧ c = -16 :=
by
  intro h
  sorry

end find_constants_l291_291214


namespace initial_number_110_l291_291775

/-
There are some numbers with an average of 20. If two numbers, namely 45 and 55, are discarded,
the average of the remaining numbers is 18.75. Prove that the initial number of numbers is 110.
-/

theorem initial_number_110 (n S : ℕ) 
(h1 : S = 20 * n) 
(h2 : (S - 100) / (n - 2) = 18.75) : 
  n = 110 :=
sorry

end initial_number_110_l291_291775


namespace cost_of_one_box_of_tissues_l291_291471

variable (num_toilet_paper : ℕ) (num_paper_towels : ℕ) (num_tissues : ℕ)
variable (cost_toilet_paper : ℝ) (cost_paper_towels : ℝ) (total_cost : ℝ)

theorem cost_of_one_box_of_tissues (num_toilet_paper = 10) 
                                   (num_paper_towels = 7) 
                                   (num_tissues = 3)
                                   (cost_toilet_paper = 1.50) 
                                   (cost_paper_towels = 2.00) 
                                   (total_cost = 35.00) :
  let total_cost_toilet_paper := num_toilet_paper * cost_toilet_paper,
      total_cost_paper_towels := num_paper_towels * cost_paper_towels,
      cost_left_for_tissues := total_cost - (total_cost_toilet_paper + total_cost_paper_towels),
      one_box_tissues_cost := cost_left_for_tissues / num_tissues
  in one_box_tissues_cost = 2.00 := 
sorry

end cost_of_one_box_of_tissues_l291_291471


namespace order_of_a_b_c_l291_291921

def a : ℝ := Real.logBase 0.6 2
def b : ℝ := Real.logBase 2 0.6
def c : ℝ := 0.6 ^ 2

theorem order_of_a_b_c : c > b ∧ b > a :=
by
  -- We need to prove the correct ordering c > b > a given the defined conditions a, b, c.
  sorry

end order_of_a_b_c_l291_291921


namespace sum_of_repeating_decimals_l291_291897

noncomputable def repeating_decimal_sum : ℚ :=
  let x := 1 / 3
  let y := 7 / 99
  let z := 8 / 999
  x + y + z

theorem sum_of_repeating_decimals :
  repeating_decimal_sum = 418 / 999 :=
by
  sorry

end sum_of_repeating_decimals_l291_291897


namespace june_must_receive_at_least_50_5_percent_to_win_l291_291664

-- Define the constants and conditions
def total_students := 200
def percentage_boys := 0.60
def percentage_girls := 0.40
def percentage_boys_vote_for_june := 0.675
def percentage_girls_vote_for_june := 0.25

-- Define the number of boys and girls
def number_of_boys := total_students * percentage_boys
def number_of_girls := total_students * percentage_girls

-- Calculate the votes June would receive from boys and girls
def votes_from_boys := number_of_boys * percentage_boys_vote_for_june
def votes_from_girls := number_of_girls * percentage_girls_vote_for_june
def total_votes := votes_from_boys + votes_from_girls

-- Calculate the percentage of total votes June would receive
def percentage_total_vote_june_receives := (total_votes / total_students) * 100

-- The proof statement
theorem june_must_receive_at_least_50_5_percent_to_win :
    percentage_total_vote_june_receives = 50.5 := 
    sorry

end june_must_receive_at_least_50_5_percent_to_win_l291_291664


namespace range_of_a_l291_291555

noncomputable def f : ℝ → ℝ → ℝ
| a, x =>
  if x ≥ -1 then a * x ^ 2 + 2 * x 
  else (1 - 3 * a) * x - 3 / 2

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) → 0 < a ∧ a ≤ 1/4 :=
sorry

end range_of_a_l291_291555


namespace mass_percentage_of_O_in_KBrO3_l291_291135

def molar_mass_K : ℝ := 39.10
def molar_mass_Br : ℝ := 79.90
def molar_mass_O : ℝ := 16.00

def total_molar_mass_O : ℝ := 3 * molar_mass_O
def molar_mass_KBrO3 : ℝ := molar_mass_K + molar_mass_Br + total_molar_mass_O

def mass_percentage_O : ℝ := (total_molar_mass_O / molar_mass_KBrO3) * 100

theorem mass_percentage_of_O_in_KBrO3 : mass_percentage_O = 28.74 :=
by sorry

end mass_percentage_of_O_in_KBrO3_l291_291135


namespace rolls_probability_l291_291503

theorem rolls_probability (p : ℚ) (m n : ℕ) (h1 : nat.coprime m n)
  (h2 : p = 8 / 33)
  (X Y : ℕ)
  (hx : X = number_of_rolls_until_six_dave)
  (hy : Y = number_of_rolls_until_six_linda) :
  m + n = 41 :=
begin
  sorry
end

end rolls_probability_l291_291503


namespace proof_f_sum_l291_291210

def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.log10 (2 - x)
  else 10 ^ (x - 1)

theorem proof_f_sum :
  f (-8) + f (Real.log10 40) = 6 := by
  sorry

end proof_f_sum_l291_291210


namespace travel_distance_bus_l291_291032

theorem travel_distance_bus (D P T B : ℝ) 
    (hD : D = 1800)
    (hP : P = D / 3)
    (hT : T = (2 / 3) * B)
    (h_total : P + T + B = D) :
    B = 720 := 
by
    sorry

end travel_distance_bus_l291_291032


namespace find_f_2013_l291_291624

   -- Define the function satisfying the given conditions
   noncomputable def f : ℝ → ℝ := sorry

   -- The conditions provided in the problem
   axiom cond1 : ∀ x : ℝ, f(1 - x) = f(1 + x)
   axiom cond2 : ∀ x : ℝ, f(1 - x) + f(3 + x) = 0
   axiom cond3 : ∀ x ∈ set.Icc 0 1, f x = x

   -- Prove that f(2013) = 1
   theorem find_f_2013 : f 2013 = 1 := 
   sorry
   
end find_f_2013_l291_291624


namespace count_two_digit_numbers_with_digit_8_l291_291989

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_digit_8 (n : ℕ) : Prop :=
  n / 10 = 8 ∨ n % 10 = 8

theorem count_two_digit_numbers_with_digit_8 : 
  (∃ S : Finset ℕ, (∀ n ∈ S, is_two_digit n ∧ has_digit_8 n) ∧ S.card = 18) :=
sorry

end count_two_digit_numbers_with_digit_8_l291_291989


namespace gain_percent_l291_291031

theorem gain_percent (CP SP : ℝ) (hCP : CP = 110) (hSP : SP = 125) : 
  (SP - CP) / CP * 100 = 13.64 := by
  sorry

end gain_percent_l291_291031


namespace sum_F_eq_4078380_l291_291540

def F (n : ℕ) : ℕ := 2 * (n - 1)

theorem sum_F_eq_4078380 : 
  (∑ n in Finset.range 2019, F (n + 2)) = 4078380 := by
  sorry

end sum_F_eq_4078380_l291_291540


namespace minimum_discount_correct_l291_291436

noncomputable def minimum_discount (total_weight: ℝ) (cost_price: ℝ) (sell_price: ℝ) 
                                   (profit_required: ℝ) : ℝ :=
  let first_half_profit := (total_weight / 2) * (sell_price - cost_price)
  let second_half_profit_with_discount (x: ℝ) := (total_weight / 2) * (sell_price * x - cost_price)
  let required_profit_condition (x: ℝ) := first_half_profit + second_half_profit_with_discount x ≥ profit_required
  (1 - (7 / 11))

theorem minimum_discount_correct : minimum_discount 1000 7 10 2000 = 4 / 11 := 
by {
  -- We need to solve the inequality step by step to reach the final answer
  sorry
}

end minimum_discount_correct_l291_291436


namespace correct_operation_l291_291863

theorem correct_operation (x : ℝ) : (x^3 * x^2 = x^5) :=
by sorry

end correct_operation_l291_291863


namespace initial_volume_is_425_l291_291840

variable {V : ℚ} -- Use rational numbers for simplicity in dealing with percentages

def initial_volume_condition : Prop :=
  let volume_of_initial_water := 0.10 * V in
  let added_water := 25 in
  let new_total_volume := V + added_water in
  let new_volume_of_water := volume_of_initial_water + added_water in
  new_volume_of_water = 0.15 * new_total_volume

theorem initial_volume_is_425 (h : initial_volume_condition) : V = 425 :=
by
  sorry

end initial_volume_is_425_l291_291840


namespace equilateral_triangle_complex_sum_squares_l291_291099

noncomputable def equilateral_triangle_side_length := 24
noncomputable def sum_complex_absolute := 48

theorem equilateral_triangle_complex_sum_squares
  (a b c : ℂ)
  (h1 : ∥a - b∥ = equilateral_triangle_side_length)
  (h2 : ∥b - c∥ = equilateral_triangle_side_length)
  (h3 : ∥c - a∥ = equilateral_triangle_side_length)
  (h4 : ∥a + b + c∥ = sum_complex_absolute) :
  ∥a^2 + b^2 + c^2∥ = 768 :=
sorry

end equilateral_triangle_complex_sum_squares_l291_291099


namespace compute_expression_l291_291100

theorem compute_expression :
  20 * ((144 / 3) + (36 / 6) + (16 / 32) + 2) = 1130 := sorry

end compute_expression_l291_291100


namespace rotate_cd_to_cd_l291_291024

def rotate180 (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, -p.2)

theorem rotate_cd_to_cd' :
  let C := (-1, 2)
  let C' := (1, -2)
  let D := (3, 2)
  let D' := (-3, -2)
  rotate180 C = C' ∧ rotate180 D = D' :=
by
  sorry

end rotate_cd_to_cd_l291_291024


namespace cost_of_one_dozen_pens_is_828_36_l291_291419

noncomputable def cost_final : ℚ :=
  let x := 260 / 20 in
  let cost_pen := 5 * x in
  let cost_dozen_pens := 12 * cost_pen in
  let discounted_price := cost_dozen_pens - (cost_dozen_pens * 10 / 100) in
  discounted_price + (discounted_price * 18 / 100)

theorem cost_of_one_dozen_pens_is_828_36 : cost_final = 828.36 := by
  sorry

end cost_of_one_dozen_pens_is_828_36_l291_291419


namespace area_of_R2_l291_291063

theorem area_of_R2 (a b : ℝ) :
  let R1_side1 := 3
  let R1_area := 24
  let R2_diagonal := 20
  let similarity_ratio := 8 / 3
  let R1_side2 := R1_area / R1_side1
  let a_squared := (9/73) * 400
  let a := sqrt a_squared
  let b := (8/3) * a
  let R2_area := a * b in

  a^2 + b^2 = R2_diagonal^2 ∧
  R2_area = 28800 / 219 :=
begin
  sorry
end

end area_of_R2_l291_291063


namespace find_angle_MON_l291_291553

-- Definitions of conditions
variables {A B O C M N : Type} -- Points in a geometric space
variables (angle_AOB : ℝ) (ray_OC : Prop) (bisects_OM : Prop) (bisects_ON : Prop)
variables (angle_MOB : ℝ) (angle_MON : ℝ)

-- Conditions
-- Angle AOB is 90 degrees
def angle_AOB_90 (angle_AOB : ℝ) : Prop := angle_AOB = 90

-- OC is a ray (using a placeholder property for ray, as Lean may not have geometric entities)
def OC_is_ray (ray_OC : Prop) : Prop := ray_OC

-- OM bisects angle BOC
def OM_bisects_BOC (bisects_OM : Prop) : Prop := bisects_OM

-- ON bisects angle AOC
def ON_bisects_AOC (bisects_ON : Prop) : Prop := bisects_ON

-- The problem statement as a theorem in Lean
theorem find_angle_MON
  (h1 : angle_AOB_90 angle_AOB)
  (h2 : OC_is_ray ray_OC)
  (h3 : OM_bisects_BOC bisects_OM)
  (h4 : ON_bisects_AOC bisects_ON) :
  angle_MON = 45 ∨ angle_MON = 135 :=
sorry

end find_angle_MON_l291_291553


namespace balanced_integer_count_l291_291482

-- Define a balanced integer
def balanced (n : ℕ) : Prop :=
  let d₅ := n % 10
  let d₄ := (n / 10) % 10
  let d₃ := (n / 100) % 10
  let d₂ := (n / 1000) % 10
  let d₁ := (n / 10000)
  d₁ + d₂ + d₃ = d₄ + d₅

-- Count the number of balanced integers from 10000 to 99999
noncomputable def count_balanced : ℕ :=
  ∑ s in Finset.range (10 + 1), (Nat.choose (s + 1) 2) * s + ∑ s in Finset.range (18 - 10 + 1), (Nat.choose (s + 10 + 1) 2) * (19 - (s + 10))

-- The main theorem stating the count of balanced integers in the given range
theorem balanced_integer_count : 
  (Finset.filter balanced (Finset.range (99999 - 10000 + 1) + 10000)).card = count_balanced := 
by
  sorry

end balanced_integer_count_l291_291482


namespace correct_conclusions_l291_291561

-- Definitions for events
def A (products : List ℕ) : Prop := ∀ (x ∈ products), x = 0
def B (products : List ℕ) : Prop := ∀ (x ∈ products), x ≠ 0
def C (products : List ℕ) : Prop := ¬A products

-- The conditions given for the problem
variable {products : List ℕ}

theorem correct_conclusions :
  (∀ p, A p → B p → False) ∧
  (∀ p, B p → C p → False) ∧
  (∀ p, C p → ¬A p) :=
by
  sorry

end correct_conclusions_l291_291561


namespace max_unique_sums_l291_291058

-- Definitions of the coin values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25
def half_dollar : ℕ := 50

-- List of coins in the purse
def coins := [penny, penny, penny, nickel, dime, dime, quarter, half_dollar]

-- Function to compute sums of different pairs
def pair_sums (coins : List ℕ) : List ℕ :=
  List.bagOf (fun x => x.1 + x.2) (List.pairwise coins (· ≠ ·))

-- Set of unique sums from the list of pairwise sums
def unique_sums (coins : List ℕ) : Finset ℕ :=
  (pair_sums coins).toFinset

-- Theorem statement
theorem max_unique_sums : unique_sums coins.card = 12 := sorry

end max_unique_sums_l291_291058


namespace calc_S_5_minus_S_4_l291_291187

def sum_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 2 * a n - 2

theorem calc_S_5_minus_S_4 {a : ℕ → ℕ} {S : ℕ → ℕ}
  (h : sum_sequence a S) : S 5 - S 4 = 32 :=
by
  sorry

end calc_S_5_minus_S_4_l291_291187


namespace shortest_multicolored_cycle_l291_291701

-- Define vertices and edges
variables {V: Type*} [fintype V] -- Vertex set
variables {E: Type*} [fintype E] -- Edge set
variables (cycle : list E) (color : E → ℕ) -- Cycle and color function

-- Define the conditions
def is_vertex_horizontal (v : V) : Prop := sorry -- Predicate for horizontal vertices
def is_vertex_vertical (v : V) : Prop := sorry -- Predicate for vertical vertices
def edges_of_cycle (cycle : list E) : list (V × V) := sorry -- Extract edges from the cycle
def are_edges_multicolored (edges : list (V × V)) : Prop := sorry -- Check if edges are multicolored

-- Define the length of the cycle
def cycle_length : ℕ := cycle.length

-- Prove the shortest multicolored cycle has 4 edges
theorem shortest_multicolored_cycle (h_cycle: ∀ (a_i b_i : V) (h1: is_vertex_horizontal a_i) (h2: is_vertex_vertical b_i),
  ∃ s, edges_of_cycle cycle = list.zip (list.repeat a_i s) (list.repeat b_i s) ∧ cycle_length = 2 * s)
    (h_s: ∃ s, s > 2)
  : ∃ s, s = 2 :=
by
  sorry

end shortest_multicolored_cycle_l291_291701


namespace triangle_50_40_l291_291879

-- Define the new operation
def triangle (a b : ℕ) : ℕ := a * b + (a - b) + 6

-- Prove the specified value for the operation
theorem triangle_50_40 : triangle 50 40 = 2016 := by
  unfold triangle
  -- Substituting and simplification steps directly in Lean
  have h1 : 50 * 40 = 2000 := by norm_num
  have h2 : 50 - 40 = 10 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end triangle_50_40_l291_291879


namespace kernel_red_given_popped_l291_291823

def prob_red_given_popped (P_red : ℚ) (P_green : ℚ) 
                           (P_popped_given_red : ℚ) (P_popped_given_green : ℚ) : ℚ :=
  let P_red_popped := P_red * P_popped_given_red
  let P_green_popped := P_green * P_popped_given_green
  let P_popped := P_red_popped + P_green_popped
  P_red_popped / P_popped

theorem kernel_red_given_popped : prob_red_given_popped (3/4) (1/4) (3/5) (3/4) = 12/17 :=
by
  sorry

end kernel_red_given_popped_l291_291823


namespace problem_a_problem_b_problem_c_problem_d_l291_291807

-- Problem Statement a
theorem problem_a (P : projective_transformation) (line_at_infinity : affine_plane.line) 
  (H : P.map_line line_at_infinity = line_at_infinity) : affine_transformation P := 
sorry

-- Problem Statement b
theorem problem_b (P : projective_transformation) (line_ex_line_a : affine_plane.line)
  (A B C D : affine_plane.point) 
  (H : ∀ (x : affine_plane.point), x ∈ line_ex_line_a -> x ∈ lines_parallel_to line_ex_line_a) :
  segment_ratio (P.map_point A) (P.map_point B) (P.map_point C) (P.map_point D) = segment_ratio A B C D :=
sorry

-- Problem Statement c
theorem problem_c (P : projective_transformation) (l₁ l₂ : affine_plane.line)
  (H : ∀ (x : affine_plane.line), parallel x l₁ ∧ parallel x l₂ -> parallel (P.map_line x) (P.map_line l₁) ∧ parallel (P.map_line x) (P.map_line l₂)) :
  affine_transformation P ∨ (exceptional_line P = parallel line.l₁ line.l₂) :=
sorry

-- Problem Statement d
theorem problem_d (P : transformation) 
  (H : ∀ (l : affine_plane.line), line_maps_to_line P (finite_inf_points l)) :
  projective_transformation P :=
sorry

end problem_a_problem_b_problem_c_problem_d_l291_291807


namespace range_of_k_l291_291645

theorem range_of_k 
  (k : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 = 4 → ∃ p : ℤ, p^3 - x > 0)
  (line_eq : ∀ x : ℝ, y : ℝ, y = k * (x + 2))
  (distance_cond : ∃ (P Q R : ℝ × ℝ), ∀ p : ℤ, p^3 - k < distance_cond ∧ distance_cond = 1)
  :
  k ∈ set.Icc (-real.sqrt 3 / 3) (real.sqrt 3 / 3) :=
begin
  sorry
end

end range_of_k_l291_291645


namespace product_f_g_l291_291614

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (x + 1))
noncomputable def g (x : ℝ) : ℝ := 1 / Real.sqrt x

theorem product_f_g (x : ℝ) (hx : 0 < x) : f x * g x = Real.sqrt (x + 1) := 
by 
  sorry

end product_f_g_l291_291614


namespace triangle_is_obtuse_l291_291279

theorem triangle_is_obtuse
  (A B C : ℝ)
  (h1 : 3 * A > 5 * B)
  (h2 : 3 * C < 2 * B)
  (h3 : A + B + C = 180) :
  A > 90 :=
sorry

end triangle_is_obtuse_l291_291279


namespace y1_greater_than_y2_l291_291943

-- Definitions of the conditions.
def point1_lies_on_line (y₁ b : ℝ) : Prop := y₁ = -3 * (-2 : ℝ) + b
def point2_lies_on_line (y₂ b : ℝ) : Prop := y₂ = -3 * (-1 : ℝ) + b

-- The theorem to prove: y₁ > y₂ given the conditions.
theorem y1_greater_than_y2 (y₁ y₂ b : ℝ) (h1 : point1_lies_on_line y₁ b) (h2 : point2_lies_on_line y₂ b) : y₁ > y₂ :=
by {
  sorry
}

end y1_greater_than_y2_l291_291943


namespace remainder_polynomial_div_x_minus_1_l291_291902

noncomputable def polynomial : ℕ → ℤ :=
λ x, x^3 - 3 * x^2 + 5

theorem remainder_polynomial_div_x_minus_1 :
  (polynomial 1) = 3 :=
by sorry

end remainder_polynomial_div_x_minus_1_l291_291902


namespace hyperbola_parabola_shared_focus_l291_291963

theorem hyperbola_parabola_shared_focus (a : ℝ) (h : a > 0) :
  (∃ b c : ℝ, b^2 = 3 ∧ c = 2 ∧ a^2 = c^2 - b^2 ∧ b ≠ 0) →
  a = 1 :=
by
  intro h_shared_focus
  sorry

end hyperbola_parabola_shared_focus_l291_291963


namespace arithmetic_seq_fraction_l291_291947

theorem arithmetic_seq_fraction (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) 
  (h2 : a 1 + a 10 = a 9) 
  (d_ne_zero : d ≠ 0) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) / a 10 = 27 / 8 := 
sorry

end arithmetic_seq_fraction_l291_291947


namespace min_perimeter_l291_291964

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the coordinates of the right focus, point on the hyperbola, and point M
def right_focus (F : ℝ × ℝ) : Prop := F = (2, 0)
def point_on_left_branch (P : ℝ × ℝ) : Prop := P.1 < 0 ∧ hyperbola P.1 P.2
def point_M (M : ℝ × ℝ) : Prop := M = (0, 2)

-- Perimeter of ΔPFM
noncomputable def perimeter (P F M : ℝ × ℝ) : ℝ :=
  let PF := (P.1 - F.1)^2 + (P.2 - F.2)^2
  let PM := (P.1 - M.1)^2 + (P.2 - M.2)^2
  let MF := (M.1 - F.1)^2 + (M.2 - F.2)^2
  PF.sqrt + PM.sqrt + MF.sqrt

-- Theorem statement
theorem min_perimeter (P F M : ℝ × ℝ) 
  (hF : right_focus F)
  (hP : point_on_left_branch P)
  (hM : point_M M) :
  ∃ P, perimeter P F M = 2 + 4 * Real.sqrt 2 :=
sorry

end min_perimeter_l291_291964


namespace fourth_derivative_of_y_l291_291526

noncomputable def y (x : ℝ) : ℝ := (3 * x - 7) * (3 : ℝ)⁻¹ ^ x

theorem fourth_derivative_of_y (x : ℝ) :
  deriv^[4] (y) x = (7 * real.log 3 - 12 - 3 * real.log 3 * x) * (real.log 3) ^ 3 * (3 : ℝ)⁻¹ ^ x :=
by
  sorry

end fourth_derivative_of_y_l291_291526


namespace prop_holds_for_positive_odds_l291_291445

variable {P : ℕ → Prop}

theorem prop_holds_for_positive_odds (h1 : P 1)
  (h2 : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) :
  ∀ n : ℕ, n > 0 → odd n → P n := by
  sorry

end prop_holds_for_positive_odds_l291_291445


namespace total_amount_of_currency_notes_l291_291665

theorem total_amount_of_currency_notes (x y : ℕ) (h1 : x + y = 85) (h2 : 50 * y = 3500) : 100 * x + 50 * y = 5000 := by
  sorry

end total_amount_of_currency_notes_l291_291665


namespace convert_to_cartesian_min_PQ_l291_291646

-- Definition of curve C1 in polar coordinates
def polar_curve (ρ θ : ℝ) : Prop := 3 * ρ^2 = 12 * ρ * cos θ - 10

-- Definition of curve C2 in Cartesian coordinates
def cartesian_curve (x y : ℝ) : Prop := (x^2 / 16) + (y^2 / 4) = 1

-- Definition of Cartesian equation of C1
def cartesian_C1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2 / 3

-- Minimum distance function
def min_distance (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Proof that the Cartesian equation of C1 follows from the given polar equation
theorem convert_to_cartesian (ρ θ : ℝ) (h : polar_curve ρ θ) : (x y : ℝ) → 
  ((ρ * cos θ = x) ∧ (ρ * sin θ = y)) → cartesian_C1 x y := by
  sorry

-- Proof that the minimum value of |PQ| is sqrt(6)/3
theorem min_PQ {P Q : ℝ × ℝ} (hP : cartesian_C1 P.1 P.2) (hQ : cartesian_curve Q.1 Q.2) :
  (P Q) → min_distance P Q = (sqrt 6) / 3 := by
  sorry

end convert_to_cartesian_min_PQ_l291_291646


namespace parabola_line_intersection_sum_l291_291289

theorem parabola_line_intersection_sum (r s : ℝ) (h_r : r = 20 - 10 * Real.sqrt 38) (h_s : s = 20 + 10 * Real.sqrt 38) :
  r + s = 40 := by
  sorry

end parabola_line_intersection_sum_l291_291289


namespace tan_double_angle_tan_angle_add_pi_div_4_l291_291611

theorem tan_double_angle (α : ℝ) (h : Real.tan α = -2) : Real.tan (2 * α) = 4 / 3 :=
by
  sorry

theorem tan_angle_add_pi_div_4 (α : ℝ) (h : Real.tan α = -2) : Real.tan (2 * α + Real.pi / 4) = -7 :=
by
  sorry

end tan_double_angle_tan_angle_add_pi_div_4_l291_291611


namespace surface_area_after_removing_corner_cubes_l291_291506

theorem surface_area_after_removing_corner_cubes (initial_surface_area : ℝ)
  (original_cube_side length corner_cube_side length : ℝ)
  (h_initial : initial_surface_area = 54)
  (h_original : original_cube_side length = 3)
  (h_corner : corner_cube_side length = 1)
  (remove_corner_cube_effect : initial_surface_area
   = initial_surface_area - 8 * (corner_cube_side length * 3) + 8 * (corner_cube_side length * 3)): 
  ∃ remaining_surface_area, remaining_surface_area = initial_surface_area - 0 :=
by {
  use 54,
  sorry
}

end surface_area_after_removing_corner_cubes_l291_291506


namespace proof_problem_l291_291880

def D_f (f : ℕ → ℕ) (n : ℕ) : ℕ :=
  Nat.find (λ m, ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → f i % m ≠ f j % m)

def f (x : ℕ) := x * (3 * x - 1)

theorem proof_problem (n : ℕ) : D_f f n = 3 ^ Nat.ceil (Real.log n) := by
  sorry

end proof_problem_l291_291880


namespace solve_abs_inequality_l291_291535

theorem solve_abs_inequality (x : ℝ) :
  abs ((6 - 2 * x + 5) / 4) < 3 ↔ -1 / 2 < x ∧ x < 23 / 2 := 
sorry

end solve_abs_inequality_l291_291535


namespace negatively_added_marks_l291_291255

theorem negatively_added_marks 
  (correct_marks_per_question : ℝ) 
  (total_marks : ℝ) 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (x : ℝ) 
  (h1 : correct_marks_per_question = 4)
  (h2 : total_marks = 420)
  (h3 : total_questions = 150)
  (h4 : correct_answers = 120) 
  (h5 : total_marks = (correct_answers * correct_marks_per_question) - ((total_questions - correct_answers) * x)) :
  x = 2 :=
by 
  sorry

end negatively_added_marks_l291_291255


namespace gcd_of_60_and_75_l291_291391

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l291_291391


namespace decode_last_e_l291_291057

-- Definitions:
def cyclic_shift (letter : Char) (shift : ℕ) : Char :=
  let base := if ('a' ≤ letter ∧ letter ≤ 'z') then 'a'.toNat else 'A'.toNat
  let letter_position := letter.toNat - base
  let new_position := (letter_position + shift) % 26
  Char.ofNat (base + new_position)

def sum_of_squares (n : ℕ) : ℕ :=
  (List.range (n + 1)).sumBy (fun x => x * x)

-- Condition extraction:
def letter_positions (message : String) (ch : Char) : List ℕ :=
  message.toList.enum.filter (fun (_, c) => c = ch).map (fun (i, _) => i)

-- Main theorem:
theorem decode_last_e : 
  let message := "What is essential is invisible to the eye."
  let last_e_index := letter_positions message 'e' |>.getLast none
  let occurrences := letter_positions message 'e'
  last_e_index = some 28 ∧ occurrences.length = 5 →
  cyclic_shift 'e' (sum_of_squares 5 % 26) = 'h' := 
by
  intros 
  apply And.intro
  exact rfl
  simp only [letter_positions, sum_of_squares, cyclic_shift, List.range, List.map, List.enum]
  sorry

end decode_last_e_l291_291057


namespace longest_side_of_rectangle_l291_291658

theorem longest_side_of_rectangle (l w : ℕ) 
  (h1 : 2 * l + 2 * w = 240) 
  (h2 : l * w = 1920) : 
  l = 101 ∨ w = 101 :=
sorry

end longest_side_of_rectangle_l291_291658


namespace rectangle_area_l291_291809

theorem rectangle_area (w d : ℝ) (h_w : w = 15) (h_d : d = 17) :
  ∃ l A : ℝ, l = 8 ∧ A = l * w ∧ A = 120 :=
by
  -- Assuming the rectangle has sides l, w and diagonal d
  let l := real.sqrt (d^2 - w^2)
  have h_l : l = 8, by
    calc l = real.sqrt (d^2 - w^2) : rfl
       ... = real.sqrt (17^2 - 15^2) : by rw [h_w, h_d]
       ... = real.sqrt (289 - 225) : by norm_num
       ... = real.sqrt 64 : by norm_num
       ... = 8 : by norm_num
  let A := l * w
  have h_A : A = 120, by
    calc A = l * w : rfl
       ... = 8 * 15 : by rw [h_w, h_l]
       ... = 120 : by norm_num
  exact ⟨l, A, h_l, h_A, rfl⟩

end rectangle_area_l291_291809


namespace sabertooth_tadpole_tails_value_l291_291784

-- Definitions for the numbers
variable (n k x : ℕ)

-- Given conditions
def triassic_tadpole_legs : ℕ := 5*n
def sabertooth_tadpole_legs : ℕ := 4*k
def total_legs : ℕ := triassic_tadpole_legs + sabertooth_tadpole_legs

def triassic_tadpole_tails : ℕ := n
def sabertooth_tadpole_tails : ℕ := x*k
def total_tails : ℕ := triassic_tadpole_tails + sabertooth_tadpole_tails

-- The theorem to prove
theorem sabertooth_tadpole_tails_value:
  total_legs = 100 → 
  total_tails = 64 → 
  (∃ x n k, (5 * n + 4 * k = 100) ∧ (n + x * k = 64) ∧ x = 3) :=
by
  intros h_leg h_tail
  use 3
  sorry

end sabertooth_tadpole_tails_value_l291_291784


namespace find_p_l291_291934

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

noncomputable def point_A_on_parabola (x y p : ℝ) : Prop := y^2 = 2 * p * x

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_p (x y p : ℝ) (h_p : p > 0)
  (h_A : point_A_on_parabola x y p)
  (h_dist_to_focus : distance x y (p / 2) 0 = 12)
  (h_dist_to_yaxis : real.abs x = 9) 
  : p = 6 :=
sorry

end find_p_l291_291934


namespace line_plane_relationship_l291_291169

theorem line_plane_relationship (l : Set Point) (P : Set Point) (A B : Point) (h1 : A ≠ B)
  (h2 : l = Set.univ) (h3 : A ∈ l) (h4 : B ∈ l) (h5 : P = Plane)
  (h6 : distance A P = distance B P) : (IsParallel l P) ∨ (intersects l P) :=
sorry

end line_plane_relationship_l291_291169


namespace polynomial_value_l291_291233

theorem polynomial_value (x y : ℝ) (h : x + 2 * y = 6) : 2 * x + 4 * y - 5 = 7 :=
by
  sorry

end polynomial_value_l291_291233


namespace exist_fixed_point_l291_291620

variable (S : Set α)
variable (f : Set α → Set α)

theorem exist_fixed_point (h_nonempty : S.nonempty)
  (h : ∀ X Y, X ⊆ S → Y ⊆ S → X ⊆ Y → f(X) ⊆ f(Y)) :
  ∃ A ⊆ S, f(A) = A := sorry

end exist_fixed_point_l291_291620


namespace proof_problem_l291_291888

noncomputable def expr : ℝ :=
  2016^0 - Real.logb 3 (3 * (3 / 8)) ^ (-1 / 3)

theorem proof_problem :
  expr = Real.logb 3 2 :=
by
  -- Sorry is used here to skip the proof part
  sorry

end proof_problem_l291_291888


namespace jane_trail_mix_chocolate_chips_l291_291737

theorem jane_trail_mix_chocolate_chips (c₁ : ℝ) (c₂ : ℝ) (c₃ : ℝ) (c₄ : ℝ) (c₅ : ℝ) :
  (c₁ = 0.30) → (c₂ = 0.70) → (c₃ = 0.45) → (c₄ = 0.35) → (c₅ = 0.60) →
  c₄ = 0.35 ∧ (c₅ - c₁) * 2 = 0.40 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end jane_trail_mix_chocolate_chips_l291_291737


namespace jane_project_time_l291_291270

theorem jane_project_time
  (J : ℝ)
  (work_rate_jane_ashley : ℝ := 1 / J + 1 / 40)
  (time_together : ℝ := 15.2 - 8)
  (work_done_together : ℝ := time_together * work_rate_jane_ashley)
  (ashley_alone_time : ℝ := 8)
  (work_done_ashley : ℝ := ashley_alone_time / 40)
  (jane_alone_time : ℝ := 4)
  (work_done_jane_alone : ℝ := jane_alone_time / J) :
  7.2 * (1 / J + 1 / 40) + 8 / 40 + 4 / J = 1 ↔ J = 18.06 :=
by 
  sorry

end jane_project_time_l291_291270


namespace initial_bushes_count_l291_291508

theorem initial_bushes_count (n : ℕ) (h : 2 * (27 * n - 26) + 26 = 190 + 26) : n = 8 :=
by
  sorry

end initial_bushes_count_l291_291508


namespace count_two_digit_numbers_with_digit_8_l291_291987

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_digit_8 (n : ℕ) : Prop :=
  n / 10 = 8 ∨ n % 10 = 8

theorem count_two_digit_numbers_with_digit_8 : 
  (∃ S : Finset ℕ, (∀ n ∈ S, is_two_digit n ∧ has_digit_8 n) ∧ S.card = 18) :=
sorry

end count_two_digit_numbers_with_digit_8_l291_291987


namespace arithmetic_sequence_l291_291225

variable (n : ℕ)

-- Conditions
axiom a1 : Int = 1
axiom S3 : 9 = 1 + (1 + d) + (1 + 2*d)

-- Question (1) : Find the general formula for the sequence \{a_n\}.
def an_formula : Int -> Int := fun n => 1 + (n-1)*d

-- Question (2):
def Sn_formula : Nat -> Int := fun n => n*n

def bn_n : Nat -> Int := fun n => (2*n - 1)^2

-- Prove the sum of the first \(n\) terms of the sequence \(\{b_n\}\), denoted as \(T_n\), is  
def Tn_formula : Nat -> Int := fun n => ∑ i in Finset.range n, bn_n (i + 1)

-- Lean 4 statement for the proof problem.
theorem arithmetic_sequence :
  (∀ n, (an_formula n = 2*n - 1)) ∧
  (∀ n, Tn_formula n = ∑ i in Finset.range n, (2*(i+1) - 1)^2) := 
sorry

end arithmetic_sequence_l291_291225


namespace tangent_line_at_one_extreme_values_of_f_l291_291209

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.log x

theorem tangent_line_at_one {a : ℝ} (ha : a = 2) : 
  let f := (λ x : ℝ, x - 2 * Real.log x) in 
  tangent_eq : ∀ x, tangent_eq at := x + y - 2 = 0
  sorry

theorem extreme_values_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x ∈ (Set.Ioi 0), ∃ x, differentiable x (f)) ∧
  (a > 0 → (∃ x, differentiable x (f) -> f(x) = f(x))) ∧
  (a > 0 → ∃ x = a, is_minimum x (f) f(a-a*Real.log(a)))
  sorry


end tangent_line_at_one_extreme_values_of_f_l291_291209


namespace distinct_triangles_from_dodecahedron_l291_291228

theorem distinct_triangles_from_dodecahedron : 
  let vertices := 12 in
  nat.choose vertices 3 = 220 :=
by 
  let vertices := 12
  have h : nat.choose vertices 3 = 220 := by sorry
  exact h

end distinct_triangles_from_dodecahedron_l291_291228


namespace coefficient_of_determination_correct_l291_291953

noncomputable def coefficient_of_determination (SSR TSS : ℝ) : ℝ :=
  1 - SSR / TSS

theorem coefficient_of_determination_correct :
  ∀ (SSR TSS : ℝ),
  SSR = 60 →
  TSS = 80 →
  coefficient_of_determination SSR TSS = 0.25 :=
by
  intros SSR TSS SSR_eq TSS_eq
  rw [SSR_eq, TSS_eq]
  unfold coefficient_of_determination
  norm_num
  sorry

end coefficient_of_determination_correct_l291_291953


namespace find_range_of_omega_l291_291211

noncomputable def range_of_omega (ω : ℝ) (k : ℤ) : set ℝ :=
  {ω | 8 * k + 2 ≤ ω ∧ ω ≤ 4 * k + 3}

theorem find_range_of_omega (ω : ℝ) (hω : ω > 0) (k : ℤ) (h_decreasing: ∀ x y, x ∈ Icc (π/4) (π/2) → y ∈ Icc (π/4) (π/2) → x ≤ y → ∂/∂x sin(ω * x) ≤ 0) :
  ω ∈ range_of_omega ω 0 :=
sorry

end find_range_of_omega_l291_291211


namespace angle_equality_l291_291312

variables {A B C D P : Type}
variables [parallelogram A B C D] {P : Type}
variables [outside P A B C D] (h1 : ∠ P A B = ∠ P C B)
variables [diff_half_planes P B A C]

theorem angle_equality (h1 : ∠ P A B = ∠ P C B) : ∠ A P B = ∠ D P C :=
sorry

end angle_equality_l291_291312


namespace difference_in_max_min_distance_from_circle_to_line_l291_291093

noncomputable def circle_center (x y : ℝ) : ℝ × ℝ := (2, 2)
noncomputable def circle_radius : ℝ := 3 * Real.sqrt 2

def point_to_line_distance (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

theorem difference_in_max_min_distance_from_circle_to_line :
  let A := 1
  let B := 1
  let C := -14
  let x₀ := 2
  let y₀ := 2
  point_to_line_distance x₀ y₀ A B C = 5 * Real.sqrt 2 → 
  2 * circle_radius = 6 * Real.sqrt 2
:=
by
  sorry

end difference_in_max_min_distance_from_circle_to_line_l291_291093


namespace number_of_written_numbers_is_three_l291_291834

theorem number_of_written_numbers_is_three (n : ℕ) (a : Fin n → ℝ) 
  (h : ∀ i, 0 < a i ∧ a i = (1 / 2) * ∑ j in Finset.univ \ {i}, a j) : n = 3 :=
by
  sorry

end number_of_written_numbers_is_three_l291_291834


namespace sum_of_eight_digits_l291_291911

open Nat

theorem sum_of_eight_digits {a b c d e f g h : ℕ} 
  (h_distinct : ∀ i j, i ∈ [a, b, c, d, e, f, g, h] → j ∈ [a, b, c, d, e, f, g, h] → i ≠ j → i ≠ j)
  (h_vertical_sum : a + b + c + d + e = 25)
  (h_horizontal_sum : f + g + h + b = 15) 
  (h_digits_set : ∀ x ∈ [a, b, c, d, e, f, g, h], x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ)) : 
  a + b + c + d + e + f + g + h - b = 39 := 
sorry

end sum_of_eight_digits_l291_291911


namespace cos_6_arccos_third_l291_291517

theorem cos_6_arccos_third :
  let x := Real.arccos (1/3) in
  Real.cos (6 * x) = 329 / 729 :=
by
  have h_cos_x : Real.cos x = 1/3 := sorry
  have h_cos_3x : Real.cos (3 * x) = 4 * (Real.cos x) ^ 3 - 3 * (Real.cos x) := sorry
  have h_cos_6x : Real.cos (6 * x) = 2 * (Real.cos (3 * x)) ^ 2 - 1 := sorry
  sorry

end cos_6_arccos_third_l291_291517


namespace even_function_must_be_two_l291_291955

def f (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + (m-2)*x + (m^2 - 7*m + 12)

theorem even_function_must_be_two (m : ℝ) :
  (∀ x : ℝ, f m (-x) = f m x) ↔ m = 2 :=
by
  sorry

end even_function_must_be_two_l291_291955


namespace gcd_60_75_l291_291390

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l291_291390


namespace cubic_root_exponent_l291_291513

theorem cubic_root_exponent (a : ℝ) (m n : ℕ) (h : n ≠ 0) : 
(∛(a ^ m)) = a ^ (m / n : ℝ) :=
by sorry

end cubic_root_exponent_l291_291513


namespace evaluate_T_l291_291111

def T (a b : ℤ) : ℤ := 4 * a - 7 * b

theorem evaluate_T : T 6 3 = 3 := by
  sorry

end evaluate_T_l291_291111


namespace polar_coordinate_equation_of_line_l_rectangular_coordinate_equation_of_curve_C_minimum_value_of_OM_over_ON_l291_291652

-- Definitions based on conditions
def line_l_parametric (t : ℝ) : ℝ × ℝ := (6 + sqrt 3 * t, -t)
def curve_C_polar (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2 ∧ ρ = 2 * cos α

-- Statements of the questions with answers
theorem polar_coordinate_equation_of_line_l :
  ∀ (ρ θ : ℝ),
    ∃ t : ℝ, line_l_parametric t = (ρ * cos θ, ρ * sin θ) →
    ρ * (cos θ + sqrt 3 * sin θ) = 6 := sorry

theorem rectangular_coordinate_equation_of_curve_C :
  ∀ (x y : ℝ), 
    curve_C_polar some_angle → 
    (x = ρ * cos some_angle) ∧ (y = ρ * sin some_angle) →
    (x - 1)^2 + y^2 = 1 := sorry

theorem minimum_value_of_OM_over_ON :
  ∀ (θ0 : ℝ),
    let ρ_M := λ θ₀ : ℝ, 3 / (sin (θ₀ + Real.pi / 6))
    let ρ_N := λ θ₀ : ℝ, 2 * cos θ₀
    ρ_M θ0 / ρ_N θ0 = 2 := sorry

end polar_coordinate_equation_of_line_l_rectangular_coordinate_equation_of_curve_C_minimum_value_of_OM_over_ON_l291_291652


namespace prob_intersection_prob_conditional_ABC_given_D_prob_C_given_AB_prob_D_given_ABC_l291_291288

variables {Ω : Type} [ProbabilitySpace Ω]
variables (A B C D : Event Ω)

axiom independent_events : Independent [A, B, C, D]
axiom prob_A : P[A] = 4 / 5
axiom prob_B : P[B] = 2 / 5
axiom prob_C : P[C] = 1 / 3
axiom prob_D : P[D] = 1 / 2

theorem prob_intersection : P[A ∩ B ∩ C ∩ D] = 4 / 75 :=
by sorry

theorem prob_conditional_ABC_given_D : P[A ∩ B ∩ C | D] = 8 / 75 :=
by sorry

theorem prob_C_given_AB : P[C | A ∩ B] = 1 / 3 :=
by sorry

theorem prob_D_given_ABC : P[D | A ∩ B ∩ C] = 1 / 2 :=
by sorry

end prob_intersection_prob_conditional_ABC_given_D_prob_C_given_AB_prob_D_given_ABC_l291_291288


namespace sequence_product_l291_291265

-- Definitions and conditions
def geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

def a4_value (a : ℕ → ℕ) : Prop :=
a 4 = 2

-- The statement to be proven
theorem sequence_product (a : ℕ → ℕ) (q : ℕ) (h_geo_seq : geometric_sequence a q) (h_a4 : a4_value a) :
  a 2 * a 3 * a 5 * a 6 = 16 :=
sorry

end sequence_product_l291_291265


namespace linear_equation_with_one_variable_eqC_l291_291862

def eqA (x : ℝ) : Prop := |(1 / x)| = 2
def eqB (x : ℝ) : Prop := x^2 - 2 = 1
def eqC (x : ℝ) : Prop := 2 * x - 3 = 5
def eqD (x y : ℝ) : Prop := x - y = 3

theorem linear_equation_with_one_variable_eqC : 
  ∃ x : ℝ, eqC x ∧ ∀ (eq : ℝ → Prop), eq = eqC → eq = eq {
  sorry
}

end linear_equation_with_one_variable_eqC_l291_291862


namespace probability_of_rolling_three_next_l291_291662

-- Define the probability space and events
def fair_die : ProbabilityMassFunction (Fin 6) :=
  ProbabilityMassFunction.uniformOfFin

-- Define the event of rolling a specific number (e.g. three)
def event_three : Set (Fin 6) := {3}

theorem probability_of_rolling_three_next :
  (∀ i : Fin 6, fair_die i = 1 / 6) →
  ∀ (previous_rolls : Vector (Fin 6) 6),
  ∀ (h : ∀ i ∈ previous_rolls.toList, i = 5),
  fair_die.toMeasure.Prob event_three = 1 / 6 :=
by
  intros h previous_rolls roll_condition
  sorry

end probability_of_rolling_three_next_l291_291662


namespace tan_alpha_minus_pi_over_4_eq_l291_291189

noncomputable def alpha : ℝ := sorry -- α is an acute angle
-- Using the condition cos α = sqrt(5)/5
def cos_alpha : ℝ := Real.cos alpha
def cos_alpha_eq : cos_alpha = (Real.sqrt 5) / 5 := sorry

-- Define α to be a positive acute angle
axiom alpha_acute : 0 < alpha ∧ alpha < Real.pi / 2

-- Define tan (α - π/4) using the tangent difference identity
def tan_alpha_minus_pi_over_4 : ℝ := (Real.tan alpha - 1) / (1 + Real.tan alpha)

-- Problem statement: Given the conditions, prove the required identity
theorem tan_alpha_minus_pi_over_4_eq :
  ∀ α,
    (0 < α ∧ α < Real.pi / 2) →
    Real.cos α = (Real.sqrt 5) / 5 →
    tan_alpha_minus_pi_over_4 = 1 / 3 :=
by
  intro alpha
  intro h₁ h₂
  sorry

end tan_alpha_minus_pi_over_4_eq_l291_291189


namespace installation_rates_l291_291339

variables (units_total : ℕ) (teamA_units : ℕ) (teamB_units : ℕ) (team_units_gap : ℕ)
variables (rate_teamA : ℕ) (rate_teamB : ℕ)

-- Conditions
def conditions : Prop :=
  units_total = 140 ∧
  teamA_units = 80 ∧
  teamB_units = units_total - teamA_units ∧
  team_units_gap = 5 ∧
  rate_teamA = rate_teamB + team_units_gap

-- Question to prove
def solution : Prop :=
  rate_teamB = 15 ∧ rate_teamA = 20

-- Statement of the proof
theorem installation_rates (units_total : ℕ) (teamA_units : ℕ) (teamB_units : ℕ) (team_units_gap : ℕ) (rate_teamA : ℕ) (rate_teamB : ℕ) :
  conditions units_total teamA_units teamB_units team_units_gap rate_teamA rate_teamB →
  solution rate_teamA rate_teamB :=
sorry

end installation_rates_l291_291339


namespace census_survey_is_suitable_l291_291026

def suitable_for_census (s: String) : Prop :=
  s = "Understand the vision condition of students in a class"

theorem census_survey_is_suitable :
  suitable_for_census "Understand the vision condition of students in a class" :=
by
  sorry

end census_survey_is_suitable_l291_291026


namespace units_digit_sum_l291_291973

theorem units_digit_sum :
  ∑ k in Finset.range 100, (k + 1) = 5050 →
  (∑ n in Finset.range 2021, n * (n^2 + n + 1)) % 10 = 3 :=
by intros h; sorry

end units_digit_sum_l291_291973


namespace books_distribution_l291_291119

theorem books_distribution : 
    ∃ n : ℕ, (n = 36 ∧ (∀ (books : Finset ℕ) (people : Finset ℕ), 
    books.card = 4 → people.card = 3 → 
    (∃ f : books → people, 
    (∀ p ∈ people, ∃ b ∈ books, f b = p) ∧ f.bij_on people))) := 
begin
  use 36,
  split,
  { refl },
  { 
    intros books people h_books h_people,
    sorry
  }
end

end books_distribution_l291_291119


namespace part1_part2_l291_291184

variable (m : ℝ)
def p := ∀ x : ℝ, x^2 + 1 ≥ m
def q := (−2 < m) ∧ (m < 2)

theorem part1 : p m → m ≤ 1 :=
by
  intros hp
  sorry

theorem part2 : (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ≤ -2 ∨ 1 < m ∧ m < 2 :=
by
  intros h
  sorry

end part1_part2_l291_291184


namespace number_of_elements_with_first_digit_7_l291_291671

open Real

noncomputable def S : Set ℝ := {n | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 5000 ∧ n = 7 ^ k}

def log_10_of_7 : ℝ := 4301 / 5000

def first_digit_is_7 (n : ℝ) : Prop := 
  7 ≤ 10 ^ (log 10 n % 1) ∧ 10 ^ (log 10 n % 1) < 8

theorem number_of_elements_with_first_digit_7 : 
  ∃ count : ℕ, count = 275 ∧ 
  count = Set.card {n ∈ S | first_digit_is_7 n} :=
begin
  sorry
end

end number_of_elements_with_first_digit_7_l291_291671


namespace sum_y_coordinates_opposite_vertices_rectangle_l291_291231

theorem sum_y_coordinates_opposite_vertices_rectangle :
  ∀ {A C : ℝ × ℝ}, 
    A = (4, 20) → C = (10, -6) → 
    ∃ B D : ℝ × ℝ, 
      let M := (4+10)/2, let N := (20+(-6))/2 
      (B.1 = B.1) ∧ (D.1 = D.1) ∧ 
      (B.2 + D.2 = 2 * N) :=
begin
  intros A C hA hC,
  use ((7, _), (7, _)),  -- B and D with unknown y-coordinates
  let M := 7,
  let N := 7,
  split, { -- B.1 = 7
    sorry
  },
  split, { -- D.1 = 7
    sorry
  },
  { -- y-coordinates calculation
    sorry
  }
end

end sum_y_coordinates_opposite_vertices_rectangle_l291_291231


namespace num_even_digits_in_base7_of_528_is_zero_l291_291145

def is_digit_even_base7 (d : ℕ) : Prop :=
  (d = 0) ∨ (d = 2) ∨ (d = 4) ∨ (d = 6)

def base7_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0] else (List.unfoldr (λ x, if x = 0 then none else some (x % 7, x / 7)) n).reverse

def number_of_even_digits_base7 (n : ℕ) : ℕ :=
  List.countp is_digit_even_base7 (base7_representation n)

theorem num_even_digits_in_base7_of_528_is_zero : number_of_even_digits_base7 528 = 0 :=
by
  sorry

end num_even_digits_in_base7_of_528_is_zero_l291_291145


namespace unique_solution_l291_291523

theorem unique_solution (x : ℝ) : (x ≠ -2) → (frac (x^3 + 2 * x^2) (x^2 + 3 * x + 2) + x = -6) → (x = -3/2) := 
by
  sorry

end unique_solution_l291_291523


namespace order_of_a_b_c_l291_291613

noncomputable def a : ℝ := (7 / 9 : ℝ) ^ (-1 / 4)
noncomputable def b : ℝ := (9 / 7 : ℝ) ^ (1 / 5)
noncomputable def c : ℝ := Real.logBase 2 (7 / 9 : ℝ)

theorem order_of_a_b_c : c < b ∧ b < a := by
  sorry

end order_of_a_b_c_l291_291613


namespace lines_1_and_4_perpendicular_l291_291118

/- Definition of the lines and their slopes -/
def line1 (x y : ℝ) := 4 * y - 3 * x = 15
def line2 (x y : ℝ) := -3 * x - 4 * y = 12
def line3 (x y : ℝ) := 4 * y + 3 * x = 15
def line4 (x y : ℝ) := 3 * y + 4 * x = 12

/- Definition of the slopes of these lines -/
def slope (line : ℝ → ℝ → Prop) (x y : ℝ) (m : ℝ) : Prop :=
  ∃ b : ℝ, ∀ x : ℝ, line x (m * x + b) 

/- Slopes of specific lines -/
def slope1 := 3 / 4
def slope2 := -3 / 4
def slope3 := -3 / 4
def slope4 := -4 / 3

/- Proof statement: Check that lines 1 and 4 are perpendicular -/
theorem lines_1_and_4_perpendicular :
  slope line1 slope1 0
  ∧ slope line4 slope4 0
  ∧ slope1 * slope4 = -1 :=
sorry

end lines_1_and_4_perpendicular_l291_291118


namespace data_set_sorted_ascend_l291_291849

def positive_integer_set := Σ' (x₁ x₂ x₃ x₄ : ℕ), 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ 0 < x₄

noncomputable def mean (x : Σ' (x₁ x₂ x₃ x₄ : ℕ), 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ 0 < x₄) : ℚ :=
(x.1.1 + x.1.2 + x.1.3 + x.1.4) / 4

def median (x : Σ' (x₁ x₂ x₃ x₄ : ℕ), 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ 0 < x₄) :=
if x.1.2 ≤ x.1.3 then (x.1.2 + x.1.3) / 2 else (x.1.2 + x.1.3) / 2

noncomputable def stddev (x : Σ' (x₁ x₂ x₃ x₄ : ℕ), 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ 0 < x₄) : ℚ :=
real.sqrt ((x.1.1 - real.of_int (mean x))^2 / 4 + (x.1.2 - real.of_int (mean x))^2 / 4 + (x.1.3 - real.of_int (mean x))^2 / 4 + (x.1.4 - real.of_int (mean x))^2 / 4).to_rat

def data_arranged (x : Σ' (x₁ x₂ x₃ x₄ : ℕ), 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ 0 < x₄) : Prop :=
[1, 2, 2, 3] = list.sort (≤) [x.1.1, x.1.2, x.1.3, x.1.4]

theorem data_set_sorted_ascend :
∀ (x : positive_integer_set),
  mean x = 2 →
  median x = 2 →
  stddev x = 1 →
  data_arranged x :=
by sorry

end data_set_sorted_ascend_l291_291849


namespace sum_of_coefficients_of_y_l291_291873

theorem sum_of_coefficients_of_y :
  let expr := (2 : ℤ) * (λ x y z, x) + (3 : ℤ) * (λ x y z, y) + 2 * (λ x y z, z)
              -- first term (4*(λx y z, x^2) + 2*(λx y z, x) + 5*(λx y z, y) + 6*(λx y z, 1))
  let prod := expr * (4 * (λ x y z, x^2) + 2 * (λ x y z, x) + 5 * (λ x y z, y) + 6 * (λ x y z, 1))
  sum_coefficients_y prod = 65 :=
by
  sorry -- The proof is omitted.

end sum_of_coefficients_of_y_l291_291873


namespace arithmetic_sequence_sum_ratio_l291_291282

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 7 / 3) :
  S 5 / S 3 = 5 := 
by
  sorry

end arithmetic_sequence_sum_ratio_l291_291282


namespace seniors_in_statistics_correct_l291_291699

-- Conditions
def total_students : ℕ := 120
def percentage_statistics : ℚ := 1 / 2
def percentage_seniors_in_statistics : ℚ := 9 / 10

-- Definitions based on conditions
def students_in_statistics : ℕ := total_students * percentage_statistics
def seniors_in_statistics : ℕ := students_in_statistics * percentage_seniors_in_statistics

-- Statement to prove
theorem seniors_in_statistics_correct :
  seniors_in_statistics = 54 :=
by
  -- Proof goes here
  sorry

end seniors_in_statistics_correct_l291_291699


namespace units_digit_of_exponentiated_product_l291_291117

theorem units_digit_of_exponentiated_product :
  (2 ^ 2101 * 5 ^ 2102 * 11 ^ 2103) % 10 = 0 := 
sorry

end units_digit_of_exponentiated_product_l291_291117


namespace differential_savings_is_4830_l291_291037

-- Defining the conditions
def initial_tax_rate : ℝ := 0.42
def new_tax_rate : ℝ := 0.28
def annual_income : ℝ := 34500

-- Defining the calculation of tax before and after the tax rate change
def tax_before : ℝ := annual_income * initial_tax_rate
def tax_after : ℝ := annual_income * new_tax_rate

-- Defining the differential savings
def differential_savings : ℝ := tax_before - tax_after

-- Statement asserting that the differential savings is $4830
theorem differential_savings_is_4830 : differential_savings = 4830 := by sorry

end differential_savings_is_4830_l291_291037


namespace spinner_probability_divisible_by_5_l291_291894

theorem spinner_probability_divisible_by_5 :
  let outcomes := [1, 2, 4, 5] in
  (∀ (a b c : ℕ), a ∈ outcomes ∧ b ∈ outcomes ∧ c ∈ outcomes →
   let number := 100 * a + 10 * b + c in
   (number % 5 = 0) ∈ (set.powerset {x | x = false | x = true}) →
   (1 / (outcomes.length ^ 3) = 1 / 4)) :=
by 
  let outcomes := [1, 2, 4, 5] in 
  have h_outcomes_len : outcomes.length = 4,
  from rfl,
  sorry

end spinner_probability_divisible_by_5_l291_291894


namespace bridge_length_correct_l291_291078

def length_of_bridge (length_of_train : ℝ) (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * (1000 / 3600)
  let distance_covered := speed_mps * time
  distance_covered - length_of_train

theorem bridge_length_correct :
  length_of_bridge 100 29.997600191984642 36 = 199.97600191984642 :=
by 
  sorry

end bridge_length_correct_l291_291078


namespace negation_of_exists_l291_291756

variable (a : ℝ)

theorem negation_of_exists (h : ¬ ∃ x : ℝ, x^2 + a * x + 1 < 0) : ∀ x : ℝ, x^2 + a * x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l291_291756


namespace monotonicity_of_f_g_increasing_on_2_infinity_l291_291212

section 
variable {x a m : ℝ}
def f (x : ℝ) (a : ℝ) : ℝ := exp x - a * x
def g (x m : ℝ) : ℝ := (x - m) * (exp x - x) - exp x + x^2 + x
def h (x : ℝ) : ℝ := (x * exp x + 1) / (exp x - 1)

theorem monotonicity_of_f (a : ℝ) : 
  (∀ x, f x a > 0) ∨ 
  ((f x a) = 0) ∧ (∀ x ≤ log a, f x a < 0) ∧ (∀ x > log a, f x a > 0)
  sorry

theorem g_increasing_on_2_infinity (m : ℝ) : 
  a = 1 → (∀ x > 2, (g x m)' ≥ 0) → m ≤ h 2
  sorry
end

end monotonicity_of_f_g_increasing_on_2_infinity_l291_291212


namespace nate_current_age_l291_291122

open Real

variables (E N : ℝ)

/-- Ember is half as old as Nate, so E = 1/2 * N. -/
def ember_half_nate (h1 : E = 1/2 * N) : Prop := True

/-- The age difference of 7 years remains constant, so 21 - 14 = N - E. -/
def age_diff_constant (h2 : 7 = N - E) : Prop := True

/-- Prove that Nate is currently 14 years old given the conditions. -/
theorem nate_current_age (h1 : E = 1/2 * N) (h2 : 7 = N - E) : N = 14 :=
by sorry

end nate_current_age_l291_291122


namespace olivia_spent_more_l291_291363

theorem olivia_spent_more 
  (initial_wallet : ℕ) 
  (atm_collection : ℕ) 
  (after_shopping : ℕ) 
  (total_money : ℕ := initial_wallet + atm_collection)
  (spent_money : ℕ := total_money - after_shopping)
  (difference : ℕ := spent_money - atm_collection) :
  initial_wallet = 53 ∧ atm_collection = 91 ∧ after_shopping = 14 → difference = 39 := 
by 
  intros h 
  cases h with h1 h2 
  cases h2 with h3 h4 
  rw [h1, h3, h4] 
  dsimp [total_money, spent_money, difference] 
  norm_num
  sorry

end olivia_spent_more_l291_291363


namespace find_negative_number_l291_291021

theorem find_negative_number : ∃ x ∈ ({} : set ℝ), x < 0 ∧ (x = -5) :=
by
  use -5
  split
  { 
    trivial 
  }
  {
    simp
  }

end find_negative_number_l291_291021


namespace gcf_60_75_l291_291399

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l291_291399


namespace max_Sn_arithmetic_sequence_l291_291572

theorem max_Sn_arithmetic_sequence :
  ∃ n : ℕ, n = 14 ∧
  let a_1 := 40
      d := -3
      a_n := λ n : ℕ, a_1 + (n-1) * d
      S_n := λ n : ℕ, n * (a_1 + a_n n) / 2
  in
  (∀ m : ℕ, m > n → a_n m ≤ 0) ∧
  (∀ m : ℕ, m ≤ n → a_n m > 0) :=
sorry

end max_Sn_arithmetic_sequence_l291_291572


namespace number_of_even_digits_in_base7_of_528_l291_291138

/-
  Define the base-7 representation of a number and a predicate to count even digits.
-/

-- Definition of base-7 digit representation
def base7_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else (List.unfoldr (λ n, if n = 0 then Option.none else some (n % 7, n / 7)) n).reverse

-- Predicate to check if a digit is even
def is_even (d : ℕ) : Bool := d % 2 = 0

-- Counting the even digits in base-7 representation
def count_even_digits_in_base7 (n : ℕ) : ℕ :=
  (base7_repr n).countp is_even

-- The target theorem to prove
theorem number_of_even_digits_in_base7_of_528 : count_even_digits_in_base7 528 = 0 :=
by sorry

end number_of_even_digits_in_base7_of_528_l291_291138


namespace solution_l291_291612

noncomputable def a := 15
noncomputable def b := 161

theorem solution : a + b = 176 :=
by
  rw [a, b]
  norm_num

end solution_l291_291612


namespace sqrt_expression_power_calculation_l291_291092

theorem sqrt_expression_power_calculation : (real.sqrt ((real.sqrt 2) ^ 2)) ^ 3 = 2 * real.sqrt 2 :=
by
  sorry

end sqrt_expression_power_calculation_l291_291092


namespace partition_sequences_l291_291868

-- Definitions and conditions from the problem
def sequence : Type := vector (bool) 2022

def is_compatible (s1 s2 : sequence) : Prop :=
  (s1.to_list.zip s2.to_list).countp (λ ⟨x, y⟩, x = y) = 4

-- Main theorem statement
theorem partition_sequences (sequences : finset sequence) (h : sequences.card = nat.choose 2022 1011) :
  ∃ (groups : fin 20 → finset sequence),
    (∀ i, (groups i). ⊆ sequences) ∧
    (∀ i j, i ≠ j → disjoint (groups i) (groups j)) ∧
    (∀ i, ∀ s1 s2 ∈ groups i, ¬ is_compatible s1 s2) :=
sorry

end partition_sequences_l291_291868


namespace area_of_circle_l291_291236

-- Given conditions
variables (r : ℝ) (π : ℝ) (C D : ℝ)
noncomputable def circumference := 2 * π * r
noncomputable def diameter := 2 * r

-- The given equation
axiom given_equation : 8 * (1 / circumference) + diameter = 6 * r

-- The goal to prove that the area of the circle equals 1.
theorem area_of_circle (hC : C = circumference) (hD : D = diameter) :
  π * r ^ 2 = 1 :=
by
  -- Here you would proceed with the proof steps
  sorry

end area_of_circle_l291_291236


namespace triangle_inequality_equivalence_l291_291319

theorem triangle_inequality_equivalence
    (a b c : ℝ) :
  (a < b + c ∧ b < a + c ∧ c < a + b) ↔
  (|b - c| < a ∧ a < b + c ∧ |a - c| < b ∧ b < a + c ∧ |a - b| < c ∧ c < a + b) ∧
  (max a (max b c) < b + c ∧ max a (max b c) < a + c ∧ max a (max b c) < a + b) :=
by sorry

end triangle_inequality_equivalence_l291_291319


namespace incorrect_statement_B_l291_291543

variable (a : Nat → Int) (S : Nat → Int)
variable (d : Int)

-- Given conditions
axiom S_5_lt_S_6 : S 5 < S 6
axiom S_6_eq_S_7 : S 6 = S 7
axiom S_7_gt_S_8 : S 7 > S 8
axiom S_n : ∀ n, S n = n * a n

-- Question to prove statement B is incorrect 
theorem incorrect_statement_B : ∃ (d : Int), (S 9 < S 5) :=
by 
  -- Proof goes here
  sorry

end incorrect_statement_B_l291_291543


namespace num_even_digits_in_base7_of_528_is_zero_l291_291144

def is_digit_even_base7 (d : ℕ) : Prop :=
  (d = 0) ∨ (d = 2) ∨ (d = 4) ∨ (d = 6)

def base7_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0] else (List.unfoldr (λ x, if x = 0 then none else some (x % 7, x / 7)) n).reverse

def number_of_even_digits_base7 (n : ℕ) : ℕ :=
  List.countp is_digit_even_base7 (base7_representation n)

theorem num_even_digits_in_base7_of_528_is_zero : number_of_even_digits_base7 528 = 0 :=
by
  sorry

end num_even_digits_in_base7_of_528_is_zero_l291_291144


namespace solve_for_a_l291_291748

-- Given the equation is quadratic, meaning the highest power of x in the quadratic term equals 2
theorem solve_for_a (a : ℚ) : (2 * a - 1 = 2) -> a = 3 / 2 :=
by
  sorry

end solve_for_a_l291_291748


namespace fans_who_received_all_three_l291_291893

theorem fans_who_received_all_three (n : ℕ) :
  (∀ k, k ≤ 5000 → (k % 100 = 0 → k % 40 = 0 → k % 60 = 0 → k / 600 ≤ n)) ∧
  (∀ k, k ≤ 5000 → (k % 100 = 0 → k % 40 = 0 → k % 60 = 0 → k / 600 ≤ 8)) :=
by
  sorry

end fans_who_received_all_three_l291_291893


namespace inequality_proof_l291_291315

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  2 * (a + b + c) + 9 / (a * b + b * c + c * a)^2 ≥ 7 :=
by
  sorry

end inequality_proof_l291_291315


namespace UF_opponent_score_l291_291382

theorem UF_opponent_score 
  (total_points : ℕ)
  (games_played : ℕ)
  (previous_points_avg : ℕ)
  (championship_score : ℕ)
  (opponent_score : ℕ)
  (total_points_condition : total_points = 720)
  (games_played_condition : games_played = 24)
  (previous_points_avg_condition : previous_points_avg = total_points / games_played)
  (championship_score_condition : championship_score = previous_points_avg / 2 - 2)
  (loss_by_condition : opponent_score = championship_score - 2) :
  opponent_score = 11 :=
by
  sorry

end UF_opponent_score_l291_291382


namespace f1_g0_product_l291_291946

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom odd_f : ∀ x, f(-x) = -f(x)
axiom even_g : ∀ x, g(-x) = g(x)
axiom relation_fg : ∀ x, f(x) - g(x) = 2^x

theorem f1_g0_product : f(1) * g(0) = -3 / 4 := by
  sorry

end f1_g0_product_l291_291946


namespace framed_painting_ratio_correct_l291_291442

/-- Define the conditions -/
def painting_height : ℕ := 30
def painting_width : ℕ := 20
def width_ratio : ℕ := 3

/-- Calculate the framed dimensions and check the area conditions -/
def framed_smaller_dimension (x : ℕ) : ℕ := painting_width + 2 * x
def framed_larger_dimension (x : ℕ) : ℕ := painting_height + 6 * x

theorem framed_painting_ratio_correct (x : ℕ) (h : (painting_width + 2 * x) * (painting_height + 6 * x) = 2 * (painting_width * painting_height)) :
  framed_smaller_dimension x / framed_larger_dimension x = 4 / 7 :=
by
  sorry

end framed_painting_ratio_correct_l291_291442


namespace gcd_of_60_and_75_l291_291396

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l291_291396


namespace train_cross_bridge_time_l291_291596

noncomputable def time_to_cross_bridge (length_of_train : ℝ) (speed_kmh : ℝ) (length_of_bridge : ℝ) : ℝ :=
  let total_distance := length_of_train + length_of_bridge
  let speed_mps := speed_kmh * (1000 / 3600)
  total_distance / speed_mps

theorem train_cross_bridge_time :
  time_to_cross_bridge 110 72 112 = 11.1 :=
by
  sorry

end train_cross_bridge_time_l291_291596


namespace identity_proof_l291_291722

theorem identity_proof (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    (b - c) / ((a - b) * (a - c)) + (c - a) / ((b - c) * (b - a)) + (a - b) / ((c - a) * (c - b)) =
    2 / (a - b) + 2 / (b - c) + 2 / (c - a) :=
by
  sorry

end identity_proof_l291_291722


namespace sum_squares_divisible_by_7_implies_both_divisible_l291_291628

theorem sum_squares_divisible_by_7_implies_both_divisible (a b : ℤ) (h : 7 ∣ (a^2 + b^2)) : 7 ∣ a ∧ 7 ∣ b :=
sorry

end sum_squares_divisible_by_7_implies_both_divisible_l291_291628


namespace range_of_x_in_second_quadrant_l291_291260

theorem range_of_x_in_second_quadrant (x : ℝ) (h1 : x - 2 < 0) (h2 : x > 0) : 0 < x ∧ x < 2 :=
sorry

end range_of_x_in_second_quadrant_l291_291260


namespace number_of_bouncy_balls_l291_291687

theorem number_of_bouncy_balls (x : ℕ)
    (red_packs : ℕ := 4)
    (yellow_packs : ℕ := 8)
    (green_packs : ℕ := 4)
    (total_balls : ℕ := 160)
    (total_packs : ℕ := red_packs + yellow_packs + green_packs) :
    total_packs * x = total_balls → x = 10 :=
by
  intro h
  have h' : 16 * x = 160 := by 
    rw [total_packs] at h 
    exact h
  sorry

end number_of_bouncy_balls_l291_291687


namespace volume_tetrahedron_ABMN_l291_291567

-- Define point and vector structures in 3D space
structure Point3D := (x y z : ℝ)

-- Define vertices A, B, C, D of the regular tetrahedron
def A : Point3D := ⟨0, 0, 0⟩
def B (a : ℝ) : Point3D := ⟨a, 0, 0⟩
def C (a : ℝ) : Point3D := ⟨a/2, a * Real.sqrt(3) / 2, 0⟩
def D (a : ℝ) : Point3D := ⟨a/2, a * Real.sqrt(3) / 6, a * Real.sqrt(6) / 3⟩

-- Define M and N are on the edges AC and AD respectively
-- To satisfy BMN is a right-angled triangle at B, and MN = 2a
-- We are assuming the coordinates of M and N as per the given plane’s intersection properties
variable (M N : Point3D)

theorem volume_tetrahedron_ABMN (a : ℝ) (M N : Point3D)
    (hBM : B a ≠ M)
    (hBN : B a ≠ N)
    (hRightAngle : ∥M - B a∥^2 + ∥N - B a∥^2 = (2 * a) ^ 2) :
    volume_tetrahedron (B a) M N = a^3 * Real.sqrt 2 / 3 :=
by
  sorry

-- You may need helper functions or additional structure definitions for complete accuracy.

end volume_tetrahedron_ABMN_l291_291567


namespace ratio_of_first_week_spent_to_allowance_l291_291374

-- Define the given conditions
def monthly_allowance : ℝ := 12
def amount_left_after_two_weeks : ℝ := 6

-- Define the amount spent in the first week as S
def spent_first_week (S : ℝ) : Prop :=
  let remaining_after_first_week := monthly_allowance - S in
  let spent_second_week := (1 / 4) * remaining_after_first_week in
  remaining_after_first_week - spent_second_week = amount_left_after_two_weeks

-- Define the ratio we're proving
def ratio (S : ℝ) : ℝ := S / monthly_allowance

-- Main theorem statement
theorem ratio_of_first_week_spent_to_allowance : ∃ S, spent_first_week S ∧ ratio S = 1 / 3 :=
by
  sorry

end ratio_of_first_week_spent_to_allowance_l291_291374


namespace fill_in_blanks_problem1_fill_in_blanks_problem2_fill_in_blanks_problem3_l291_291533

def problem1_seq : List ℕ := [102, 101, 100, 99, 98, 97, 96]
def problem2_seq : List ℕ := [190, 180, 170, 160, 150, 140, 130, 120, 110, 100]
def problem3_seq : List ℕ := [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]

theorem fill_in_blanks_problem1 :
  ∃ (a b c d : ℕ), [102, a, 100, b, c, 97, d] = [102, 101, 100, 99, 98, 97, 96] :=
by
  exact ⟨101, 99, 98, 96, rfl⟩ -- Proof omitted with exact values

theorem fill_in_blanks_problem2 :
  ∃ (a b c d e f g : ℕ), [190, a, b, 160, c, d, e, 120, f, g] = [190, 180, 170, 160, 150, 140, 130, 120, 110, 100] :=
by
  exact ⟨180, 170, 150, 140, 130, 110, 100, rfl⟩ -- Proof omitted with exact values

theorem fill_in_blanks_problem3 :
  ∃ (a b c d e f : ℕ), [5000, a, 6000, b, 7000, c, d, e, f, 9500] = [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500] :=
by
  exact ⟨5500, 6500, 7500, 8000, 8500, 9000, rfl⟩ -- Proof omitted with exact values

end fill_in_blanks_problem1_fill_in_blanks_problem2_fill_in_blanks_problem3_l291_291533


namespace find_fg_length_l291_291462

theorem find_fg_length
  (A B C D E F G : Point)
  (hABC : EquilateralTriangle A B C)
  (hADE : EquilateralTriangle A D E)
  (hBDG : EquilateralTriangle B D G)
  (hCEF : EquilateralTriangle C E F)
  (side_length_ABC : AB = 10)
  (AD_length : AD = 3) :
  FG = 4 :=
sorry

end find_fg_length_l291_291462


namespace proof_problem_l291_291997

noncomputable def a : ℝ := (11 + Real.sqrt 337) ^ (1 / 3)
noncomputable def b : ℝ := (11 - Real.sqrt 337) ^ (1 / 3)
noncomputable def x : ℝ := a + b

theorem proof_problem : x^3 + 18 * x = 22 := by
  sorry

end proof_problem_l291_291997


namespace gcd_60_75_l291_291389

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l291_291389


namespace find_k_distance_from_N_to_PM_l291_291980

-- Define the points P, M, and N
def P : ℝ × ℝ × ℝ := (-2, 0, 2)
def M : ℝ × ℝ × ℝ := (-1, 1, 2)
def N : ℝ × ℝ × ℝ := (-3, 0, 4)

-- Define the vectors a and b
def vector_a : ℝ × ℝ × ℝ := (M.1 - P.1, M.2 - P.2, M.3 - P.3)  -- (1, 1, 0)
def vector_b : ℝ × ℝ × ℝ := (N.1 - P.1, N.2 - P.2, N.3 - P.3)  -- (-1, 0, 2)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- The first part of the problem: finding k
theorem find_k (k : ℝ) :
  dot_product (k • vector_a + vector_b) (k • vector_a - 2 • vector_b) = 0 ↔ (k = 2 ∨ k = -5/2) :=
by
  let a := vector_a
  let b := vector_b
  sorry

-- The second part of the problem: finding the distance from N to the line PM
theorem distance_from_N_to_PM :
  let a := vector_a
  let b := vector_b
  let u := (1 / real.sqrt 2) • (1, 1, 0) in  -- direction vector of PM
  real.sqrt (dot_product b b - (dot_product b u) ^ 2) = 3 * real.sqrt 2 / 2 :=
by
  let a := vector_a
  let b := vector_b
  sorry

end find_k_distance_from_N_to_PM_l291_291980


namespace solve_equation_l291_291332

theorem solve_equation : ∀ x : ℝ, (x + 2) / 4 - 1 = (2 * x + 1) / 3 → x = -2 :=
by
  intro x
  intro h
  sorry  

end solve_equation_l291_291332


namespace range_afb_l291_291960

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2 else 2 ^ x

theorem range_afb (a b : ℝ) (h₁ : a < 0) (h₂ : 0 ≤ b) (h₃ : f a = f b) :
  ∃ y : ℝ, y ∈ set.Ico (-1 : ℝ) 0 ∧ y = a * f b :=
begin
  sorry
end

end range_afb_l291_291960


namespace percentage_increase_in_Al_l291_291321

variables 
  (A E : ℝ) 
  (P : ℝ) 

theorem percentage_increase_in_Al's_account 
  (h1: Al > Eliot)
  (h2: A - E = (1/12) * (A + E))
  (h3: E = 200)
  (h4: A * (1 + P / 100) = 1.2 * E + 20) :
  P = 10 := by
  sorry

end percentage_increase_in_Al_l291_291321


namespace correct_statement_l291_291025

-- Definitions for the sets and points
variables {Point : Type} {α β : set Point} {A B C D P Q : Point} {AB PQ CD : set Point}

-- Conditions used in the problem
def is_on_plane (P : Point) (α : set Point) : Prop := P ∈ α
def is_on_line_segment (A B P : Point) (AB : set Point) : Prop := AB ⊂ set.univ ⟨A, set.univ⟩ ⟨B, set.univ⟩ P

-- Statement to be proved
theorem correct_statement (h1 : AB ⊂ α) (h2 : AB ⊂ β) : A ∈ (α ∩ β) ∧ B ∈ (α ∩ β) :=
by
  sorry

end correct_statement_l291_291025


namespace smallest_difference_l291_291799

theorem smallest_difference :
  ∃ a b : ℕ, 
  -- Conditions
  (∃ (x y z u v : ℕ),
    ({x, y, z, u, v} = {0, 1, 2, 6, 9}) ∧
    (x ≠ y ∧ x ≠ z ∧ x ≠ u ∧ x ≠ v ∧ y ≠ z ∧ y ≠ u ∧ y ≠ v ∧ z ≠ u ∧ z ≠ v ∧ u ≠ v) ∧
    a = x * 100 + y * 10 + z ∧
    b = u * 10 + v) ∧
  -- Question and Answer
  a > b ∧ 
  a - b = 6 :=
sorry

end smallest_difference_l291_291799


namespace largest_term_binomial_expansion_l291_291803

theorem largest_term_binomial_expansion : 
  let a := 1;
      b := Real.sqrt 3;
      n := 100;
      term k := Nat.choose n k * b ^ k;
  n := 100;
  term 64 > term k for all k ≠ 64 :=
sorry

end largest_term_binomial_expansion_l291_291803


namespace equation_B_no_real_solution_l291_291889

theorem equation_B_no_real_solution : ∀ x : ℝ, |3 * x + 1| + 6 ≠ 0 := 
by 
  sorry

end equation_B_no_real_solution_l291_291889


namespace ratio_expression_value_l291_291617

theorem ratio_expression_value (a b : ℝ) (h : a / b = 4 / 1) : 
  (a - 3 * b) / (2 * a - b) = 1 / 7 := 
by 
  sorry

end ratio_expression_value_l291_291617


namespace number_of_labelings_l291_291101

open Set Function Finite.Function

def vertices : Set (Fin 3 → Bool) := {v | ∀ i, v i ∈ {true, false}}

def labelings : (Fin 3 → Bool) → ℕ := sorry -- placeholder for function definition

def euclidean_distance (v_i v_j : Fin 3 → Bool) : ℝ :=
  Real.sqrt (Finset.sum Finset.univ (λ i, if v_i i = v_j i then 0 else 1))

def satisfies_condition (f : (Fin 3 → Bool) → ℕ) := ∀ (v_i v_j : Fin 3 → Bool),  
  |(f v_i) - (f v_j)| ≥ (euclidean_distance v_i v_j) ^ 2 

theorem number_of_labelings : 
  let f : (Fin 3 → Bool) → ℕ := sorry -- placeholder for label function definition
  satisfies_condition f → 
  card (labelings f) = 144 := 
sorry

end number_of_labelings_l291_291101


namespace positive_divisors_multiple_of_15_l291_291229

theorem positive_divisors_multiple_of_15 (a b c : ℕ) (n : ℕ) (divisor : ℕ) (h_factorization : n = 6480)
  (h_prime_factorization : n = 2^4 * 3^4 * 5^1)
  (h_divisor : divisor = 2^a * 3^b * 5^c)
  (h_a_range : 0 ≤ a ∧ a ≤ 4)
  (h_b_range : 1 ≤ b ∧ b ≤ 4)
  (h_c_range : 1 ≤ c ∧ c ≤ 1) : sorry :=
sorry

end positive_divisors_multiple_of_15_l291_291229


namespace range_of_a_l291_291558

variable {x a : ℝ}

theorem range_of_a (h1 : x > 1) (h2 : a ≤ x + 1 / (x - 1)) : a ≤ 3 :=
sorry

end range_of_a_l291_291558


namespace prob_A_eq_prob_B_l291_291549

-- Given conditions
def total_balls : ℕ := 6
def red_balls : ℕ := 2
def white_balls : ℕ := 4
def balls_drawn : ℕ := 2

-- Event definitions
def event_A : Set (Finset ℕ) :=
  {s | s.card = balls_drawn ∧ 1 ≤ (s ∩ {0, 1}).card}

def event_B : Set (Finset ℕ) :=
  {s | s.card = balls_drawn ∧ (s ∩ {2, 3, 4, 5}).card ≤ 1}

-- Probability calculation placeholders
noncomputable def P (e : Set (Finset ℕ)) : ℚ := sorry

-- Proof problem statement
theorem prob_A_eq_prob_B : P(event_A) = P(event_B) :=
sorry

end prob_A_eq_prob_B_l291_291549


namespace ellipse_properties_l291_291201

theorem ellipse_properties :
  ∃ a b c d: ℝ, a = 6 ∧ b = 2 ∧ c = (1:ℝ) ∧ d = (-5)/9 ∧
  ellipse_eq : ellipse_eq C (λ x y : ℝ, x^2 / a + y^2 / b = 1) ∧
  ∃ E : ℝ × ℝ, E = (7/3, 0) ∧
  ∀ k : ℝ, k ≠ 0 →
    ∃ A B : ℝ × ℝ,
    y_of_line_eq k (x_of_point A) =
    k * (x_of_point A - 2) ∧
    y_of_line_eq k (x_of_point B) =
    k * (x_of_point B - 2) ∧
    (A.x - E.1) * (B.x - E.1) + A.y * B.y = d
:=
by
  sorry

end ellipse_properties_l291_291201


namespace heartsuit_ratio_l291_291112

-- Define the operation ⧡
def heartsuit (n m : ℕ) := n^(3+m) * m^(2+n)

-- The problem statement to prove
theorem heartsuit_ratio : heartsuit 2 4 / heartsuit 4 2 = 1 / 2 := by
  sorry

end heartsuit_ratio_l291_291112


namespace ten_row_triangle_piece_count_l291_291077

-- Define the conditions as functions
def rods_in_row (n : ℕ) : ℕ := 3 * n
def connectors_in_row (n : ℕ) : ℕ := 3 + n

def total_pieces (rows : ℕ) : ℕ :=
  let total_rods := ∑ i in Finset.range rows, rods_in_row (i + 1)
  let total_connectors := ∑ i in Finset.range rows, connectors_in_row (i + 1)
  total_rods + total_connectors

-- Prove the total number of pieces for a ten-row triangle
theorem ten_row_triangle_piece_count : total_pieces 10 = 250 :=
by
  sorry

end ten_row_triangle_piece_count_l291_291077


namespace pure_fuji_to_all_trees_ratio_l291_291434

variables {T F C : ℕ}

-- Given conditions
axiom H1 : C = 0.10 * T
axiom H2 : F + C = 204
axiom H3 : T = F + 36 + C

-- The problem statement
theorem pure_fuji_to_all_trees_ratio :
  ∃ F T : ℕ, (C = 0.10 * T) ∧ (F + C = 204) ∧ (T = F + 36 + C) → F / gcd F T = 3 ∧ T / gcd F T = 4 :=
by 
  sorry

end pure_fuji_to_all_trees_ratio_l291_291434


namespace moles_of_HCl_combined_eq_one_l291_291532

-- Defining the chemical species involved in the reaction
def NaHCO3 : Type := Nat
def HCl : Type := Nat
def NaCl : Type := Nat
def H2O : Type := Nat
def CO2 : Type := Nat

-- Defining the balanced chemical equation as a condition
def reaction (n_NaHCO3 n_HCl n_NaCl n_H2O n_CO2 : Nat) : Prop :=
  n_NaHCO3 + n_HCl = n_NaCl + n_H2O + n_CO2

-- Given conditions
def one_mole_of_NaHCO3 : Nat := 1
def one_mole_of_NaCl_produced : Nat := 1

-- Proof problem
theorem moles_of_HCl_combined_eq_one :
  ∃ (n_HCl : Nat), reaction one_mole_of_NaHCO3 n_HCl one_mole_of_NaCl_produced 1 1 ∧ n_HCl = 1 := 
by
  sorry

end moles_of_HCl_combined_eq_one_l291_291532


namespace find_largest_N_l291_291531

noncomputable def largest_N : ℕ :=
  by
    -- This proof needs to demonstrate the solution based on constraints.
    -- Proof will be filled here.
    sorry

theorem find_largest_N :
  largest_N = 44 := 
  by
    -- Proof to establish the largest N will be completed here.
    sorry

end find_largest_N_l291_291531


namespace rosa_called_pages_sum_l291_291227

theorem rosa_called_pages_sum :
  let week1_pages := 10.2
  let week2_pages := 8.6
  let week3_pages := 12.4
  week1_pages + week2_pages + week3_pages = 31.2 :=
by
  let week1_pages := 10.2
  let week2_pages := 8.6
  let week3_pages := 12.4
  sorry  -- proof will be done here

end rosa_called_pages_sum_l291_291227


namespace part1_part2_l291_291924

noncomputable theory

def z (a : ℝ) : ℂ := complex.mk a (-1)

theorem part1 (a : ℝ) (ha : 0 < a) (h : complex.add (z a) (complex.div 2 (z a)) ∈ ℝ) : a = 1 :=
by sorry

theorem part2 (m : ℝ) (hm : 1 < m ∧ m < 2) : 
(1 < m) ∧ (m < 2) := 
by sorry

end part1_part2_l291_291924


namespace triangle_AMC_area_correct_l291_291257

variables {A B C D M : Type*}
variable {AB AD: ℝ}

-- Conditions
def rectangle_ABCD (A B C D : Type*) (AB AD : ℝ) : Prop := 
  AB = 10 ∧ AD = 12

def midpoint (M : Type*) (C D : Type*) : Prop := 
  -- Midpoint condition
  M = (C + D) / 2

-- Proving the area of triangle AMC
def triangle_area_AMC (A B C D M : Type*) [rectangle_ABCD A B C D AB AD] [midpoint M C D] : ℝ := 
  30

-- The final theorem statement
theorem triangle_AMC_area_correct : 
  ∀ {A B C D M : Type*} {AB AD: ℝ} [rectangle_ABCD A B C D AB AD] [midpoint M C D],
  triangle_area_AMC A B C D M = 30 :=
by
  sorry

end triangle_AMC_area_correct_l291_291257


namespace widgets_difference_l291_291305

variable (t w : ℕ)
variable (h_w_eq : w = 2 * t)

theorem widgets_difference (t : ℕ) (w : ℕ) (h_w_eq : w = 2 * t) : 
  let monday_widgets := w * t,
      tuesday_widgets := (w + 5) * (t - 3)
  in monday_widgets - tuesday_widgets = t + 15 :=
by
  sorry

end widgets_difference_l291_291305


namespace median_of_data_l291_291453

def data : List ℕ := [6, 5, 7, 6, 6]

theorem median_of_data : List.median data = 6 := by
  sorry

end median_of_data_l291_291453


namespace solve_geometric_progression_l291_291186

theorem solve_geometric_progression :
  let a := Real.sin (3 * Real.pi / 4)
  let b := Real.sin x - Real.cos x
  let c := 2 ^ Real.cos (2 * Real.pi / 3)
  (a, b, c) ∈ set_of_geometric_progression → 
  x ∈ set.Icc 0 (2 * Real.pi) →
  x ∈ {Real.pi / 12, 5 * Real.pi / 12, 13 * Real.pi / 12, 17 * Real.pi / 12} :=
by sorry

end solve_geometric_progression_l291_291186


namespace range_of_a_l291_291965

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (a * x - 1) / x > 2 * a) ↔ a ∈ (Set.Ici (-1/2) : Set ℝ) :=
by
  sorry

end range_of_a_l291_291965


namespace parabola_focus_distance_l291_291936

theorem parabola_focus_distance (p : ℝ) (hp : p > 0) (A : ℝ × ℝ)
  (hA_on_parabola : A.2 ^ 2 = 2 * p * A.1)
  (hA_focus_dist : dist A (p / 2, 0) = 12)
  (hA_yaxis_dist : abs A.1 = 9) : p = 6 :=
sorry

end parabola_focus_distance_l291_291936


namespace mean_temperature_calc_l291_291338

def mean_temperature (temps : List ℚ) : ℚ :=
  temps.sum / temps.length

theorem mean_temperature_calc :
  let temps_f := [28]   -- Temperatures in Fahrenheit
  let temps_c := [-12, -6, -8, -3, 7, 0]   -- Temperatures in Celsius
  let converted_temps_c := temps_f.map (λ f, (5 / 9 : ℚ) * (f - 32))
  let all_temps_c := temps_c ++ converted_temps_c
  mean_temperature all_temps_c = -3.46 :=
by
  sorry

end mean_temperature_calc_l291_291338


namespace winning_strategy_l291_291000

/-- 
For each n ≥ 2, determine which player has a winning strategy.
Conditions:
1. Two players, A and B, take turns removing stones from a pile of n stones.
2. Player A goes first and must take at least 1 stone but cannot take all the stones.
3. Each player, after the first move, can take up to the number of stones taken by the previous player, but must take at least one stone.
4. The player who takes the last stone wins.
-/
theorem winning_strategy (n : ℕ) (h : 2 ≤ n) :
  (∃ k : ℕ, n = 2^k) ↔ wins B := sorry

end winning_strategy_l291_291000


namespace number_of_ordered_pairs_l291_291152

theorem number_of_ordered_pairs :
  {p : ℤ × ℤ | let m := p.1, n := p.2 in mn ≥ 0 ∧ m^3 + n^3 + 125 * m * n = 50^3 }.to_finset.card = 52 :=
sorry

end number_of_ordered_pairs_l291_291152


namespace simplify_and_evaluate_expression_l291_291730

theorem simplify_and_evaluate_expression (x : ℤ) (h : x = 2 ∨ x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  ((1 / (x:ℚ) - 1 / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1))) = (1 / 2) :=
by
  -- Skipping the proof
  sorry

end simplify_and_evaluate_expression_l291_291730


namespace ratio_of_overtime_to_regular_rate_l291_291839

def regular_rate : ℝ := 3
def regular_hours : ℕ := 40
def total_pay : ℝ := 186
def overtime_hours : ℕ := 11

theorem ratio_of_overtime_to_regular_rate 
  (r : ℝ) (h : ℕ) (T : ℝ) (h_ot : ℕ) 
  (h_r : r = regular_rate) 
  (h_h : h = regular_hours) 
  (h_T : T = total_pay)
  (h_hot : h_ot = overtime_hours) :
  (T - (h * r)) / h_ot / r = 2 := 
by {
  sorry 
}

end ratio_of_overtime_to_regular_rate_l291_291839


namespace value_range_of_c_l291_291869

theorem value_range_of_c (c : ℝ) :
  (∃ x ∈ (Icc 1 2), max (abs (x + c / x)) (abs (x + c / x + 2)) ≥ 5) → c ≤ -18 ∨ c ≥ 2 :=
by
  sorry

end value_range_of_c_l291_291869


namespace digits_of_smallest_n_l291_291286

noncomputable def smallest_n : ℕ :=
  let n := 30^10
  in n

theorem digits_of_smallest_n :
  (∃ (n : ℕ),  
    n % 30 = 0 ∧ 
    (∃ k1 : ℕ, n^2 = k1^4) ∧ 
    (∃ k2 : ℕ, n^3 = k2^5) ∧ 
    nat_digits n = 21) → 
  nat_digits smallest_n = 21 :=
by
  intro h
  sorry

end digits_of_smallest_n_l291_291286


namespace numFlags_l291_291108

-- Definitions of colors
inductive Color where
  | purple : Color
  | gold : Color
  | silver : Color

-- A flag consists of three horizontal stripes
structure Flag where
  stripe1 : Color
  stripe2 : Color
  stripe3 : Color

-- Condition: No two adjacent stripes may have the same color
def validFlag (f : Flag) : Prop :=
  f.stripe1 ≠ f.stripe2 ∧ f.stripe2 ≠ f.stripe3

-- The total number of valid flags
def totalValidFlags : Nat :=
  3 * 2 * 2

theorem numFlags : totalValidFlags = 12 := 
  by 
    -- Skipping the proof
    sorry

end numFlags_l291_291108


namespace rectangle_diagonals_not_perpendicular_l291_291407

theorem rectangle_diagonals_not_perpendicular (R : Type) [rect : Rectangle R] : 
  ¬ (Rectangles.diagonals_perpendicular R) := sorry

end rectangle_diagonals_not_perpendicular_l291_291407


namespace john_time_to_counter_l291_291408

def john_rate (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem john_time_to_counter
  (distance_moved : ℝ)
  (time_taken : ℝ)
  (remaining_distance : ℝ) :
  (john_rate distance_moved time_taken) = 2.5 →
  (remaining_distance / 2.5) = 60 :=
by
  intros hrate
  rw [← hrate]
  exact rfl

end john_time_to_counter_l291_291408


namespace sqrt_defined_range_l291_291653

theorem sqrt_defined_range (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 2)) ↔ x ≥ 2 := by
  sorry

end sqrt_defined_range_l291_291653


namespace mary_income_percentage_l291_291694

open Real

variables {S M J T : ℝ}

-- Conditions
def condition_juan_income (S : ℝ) : ℝ := 0.7 * S
def condition_tim_income (J : ℝ) : ℝ := 0.6 * J
def condition_mary_income (T : ℝ) : ℝ := 1.5 * T

-- Proof Statement
theorem mary_income_percentage (S : ℝ) :
  let J := condition_juan_income S,
      T := condition_tim_income J,
      M := condition_mary_income T in
  M / J = 0.9 := by sorry

end mary_income_percentage_l291_291694


namespace shortest_multicolored_cycle_l291_291709

theorem shortest_multicolored_cycle (G : Graph) (cycle : List (Vertex × Vertex)) :
  (∀ (a_i b_i : Vertex), (a_i, b_i) ∈ cycle) →
  (length cycle = 2 * s) →
  (∀ a_i, a_i ∈ cycle → ∃ h : Horizontal, a_i = to_vertex h) →
  (∀ b_j, b_j ∈ cycle → ∃ v : Vertical, b_j = to_vertex v) →
  (∃ (s > 2 → False), shortest_multicolored_cycle_in_G = 4) := 
by
  sorry

end shortest_multicolored_cycle_l291_291709


namespace KN_eq_KP_l291_291712

open Classical

variables {A B C K N M P : Type*}
variables [Point A] [Point B] [Point C] [Point K] [Point N] [Point M] [Point P]
variables {triangle_ABC : Triangle A B C}
variables {midpoint_AK : Midpoint N A K}
variables {midpoint_BC : Midpoint M B C}
variables {segment_NM : Segment N M}
variables {segment_CK : Segment C K}
variables {intersection_NM_CK : Intersect segment_NM segment_CK P}
variables {condition1 : OnSide K A B}
variables {condition2 : (AB : LineSegment A B) = (CK : LineSegment C K)}

theorem KN_eq_KP : (KN : LineSegment K N) = (KP : LineSegment K P) :=
sorry

end KN_eq_KP_l291_291712


namespace polynomial_inequality_coefficients_inequality_l291_291290

variable {R : Type*} [LinearOrderedField R]

-- Define the polynomial P with coefficients a_0, a_1, ..., a_n
structure PolynomialData (R : Type*) [Semiring R] :=
(a : ℕ → R)
(n : ℕ)
(distinct_roots : Set R)

/-- Given a polynomial P with distinct real roots, prove the given inequality. -/
theorem polynomial_inequality (data : PolynomialData R) {x : R} :
  (let P := ∑ i in Finset.range (data.n + 1), data.a i * x^i;
  let P' := ∑ i in Finset.range data.n, (i + 1) * data.a (i + 1) * x^i;
  let P'' := ∑ i in Finset.range (data.n - 1), (i + 2) * (i + 1) * data.a (i + 2) * x^i in
  P * P'' ≤ P' * P') :=
sorry

/-- Deduce the inequality for the coefficients. -/
theorem coefficients_inequality (data : PolynomialData R) (k : ℕ)
  (hk : 1 ≤ k ∧ k ≤ data.n - 1) :
  data.a (k - 1) * data.a (k + 1) ≤ data.a k^2 :=
sorry

end polynomial_inequality_coefficients_inequality_l291_291290


namespace calc_remainder_l291_291493

theorem calc_remainder : 
  (1 - 90 * Nat.choose 10 1 + 90^2 * Nat.choose 10 2 - 90^3 * Nat.choose 10 3 +
   90^4 * Nat.choose 10 4 - 90^5 * Nat.choose 10 5 + 90^6 * Nat.choose 10 6 -
   90^7 * Nat.choose 10 7 + 90^8 * Nat.choose 10 8 - 90^9 * Nat.choose 10 9 +
   90^10 * Nat.choose 10 10) % 88 = 1 := 
by sorry

end calc_remainder_l291_291493


namespace combination_3_choose_2_l291_291641

theorem combination_3_choose_2 : finset.card (finset.choose 2 (finset.range 3) : finset (finset ℕ)) = 3 :=
begin
  sorry
end

end combination_3_choose_2_l291_291641


namespace weekly_goal_cans_l291_291028

theorem weekly_goal_cans : (20 +  (20 * 1.5) + (20 * 2) + (20 * 2.5) + (20 * 3)) = 200 := by
  sorry

end weekly_goal_cans_l291_291028


namespace arrangement_count_l291_291771

theorem arrangement_count 
  (boys : Finset ℕ) 
  (girls : Finset ℕ) 
  (total_students : Finset ℕ) 
  (boy_A : ℕ)
  (condition1 : boy_A ∈ boys ∧ boy_A ∉ {1, 5})
  (condition2 : ∃ (g1 g2 : ℕ), g1 ≠ g2 ∧ g1 ∈ girls ∧ g2 ∈ girls ∧ adjacent g1 g2) :
  card {arrangement : list ℕ // valid_arrangement boys girls arrangement (boy_A) condition1 condition2} = 48 := 
sorry

end arrangement_count_l291_291771


namespace affine_coordinate_system_theorem_l291_291870

def alpha_affine_coordinate_system (α : ℝ) : Prop :=
  α > 0 ∧ α < π ∧ α ≠ π / 2

def vector_magnitude (α : ℝ) (a : ℝ × ℝ) : ℝ :=
  let (m, n) := a in sqrt (m^2 + n^2 + 2 * m * n * cos α)

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  let (m, n) := a in
  let (s, t) := b in
  m * t = n * s

def vectors_perpendicular (α : ℝ) (a b : ℝ × ℝ) : Prop :=
  let (m, n) := a in
  let (s, t) := b in
  m * s + n * t + (m * t + n * s) * sin α = 0

def correct_statements (α : ℝ) : Prop :=
  alpha_affine_coordinate_system α ∧
  (α ∈ (0, π) ∧ α ≠ π / 2) ∧
  (| vector_magnitude α (1, 1) = sqrt (1^2 + 1^2 + 2 * 1 * 1 * cos α) | ∧
  ∀ (a b : ℝ × ℝ), vectors_parallel a b → let (m, n) := a in let (s, t) := b in m * t - n * s = 0 ∧
  ∃ (a b : ℝ × ℝ), let (m, n) := a in let (s, t) := b in m = -1 ∧ n = 2 ∧ s = -2 ∧ t = 1 → α = π / 3)

theorem affine_coordinate_system_theorem (α : ℝ) : correct_statements α :=
by
  sorry

end affine_coordinate_system_theorem_l291_291870


namespace total_time_spent_racing_l291_291791

-- Define the conditions
def speed_in_lake : ℝ := 3 -- miles per hour
def speed_in_ocean : ℝ := 2.5 -- miles per hour
def total_races : ℕ := 10
def distance_per_race : ℝ := 3 -- miles

-- Given the conditions, prove the total time spent racing is 11 hours
theorem total_time_spent_racing : 
  let races_in_lake := total_races / 2,
      races_in_ocean := total_races / 2,
      total_distance_in_lake := distance_per_race * races_in_lake,
      total_distance_in_ocean := distance_per_race * races_in_ocean,
      time_for_lake_races := total_distance_in_lake / speed_in_lake,
      time_for_ocean_races := total_distance_in_ocean / speed_in_ocean in
  time_for_lake_races + time_for_ocean_races = 11 :=
by
  sorry

end total_time_spent_racing_l291_291791


namespace shares_of_stock_y_bought_l291_291359

def stocks_initial : List ℕ := [68, 112, 56, 94, 45]
def stock_x_before : ℕ := 56
def stock_y_before : ℕ := 94
def stock_x_sold : ℕ := 20
def range_increase : ℕ := 14

theorem shares_of_stock_y_bought (S : ℕ) :
  let stock_x_after := stock_x_before - stock_x_sold,
      initial_range := stocks_initial.maximum.getOrElse 0 - stocks_initial.minimum.getOrElse 0,
      new_range := initial_range + range_increase
  in
  (stock_y_before + S) - stock_x_after = new_range → S = 23 := by
  sorry

end shares_of_stock_y_bought_l291_291359


namespace points_P_count_l291_291168

noncomputable def line (x y : ℝ) := x + y - 3 = 0
def pointA : (ℝ × ℝ) := (3, 0)
def pointB : (ℝ × ℝ) := (0, 3)
def power_function (x : ℝ) : ℝ := x^2
def on_graph (P : ℝ × ℝ) : Prop := (P.snd = power_function P.fst)
def area_condition (P : ℝ × ℝ) : Prop := 
  let d := (|P.fst + P.snd - 3|) / Real.sqrt 2 in 
  0.5 * 3 * Real.sqrt 2 * d = 3

theorem points_P_count : 
  ∃ (P : ℝ × ℝ) (points : Finset (ℝ × ℝ)), (on_graph P) ∧ (area_condition P) ∧ points.card = 4 := 
sorry

end points_P_count_l291_291168


namespace f_2_plus_f_0_l291_291297

def odd_function_on_R (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

variable (f : ℝ → ℝ)

axiom odd_fn : odd_function_on_R f
axiom f_neg_2 : f(-2) = -3

theorem f_2_plus_f_0 : f(2) + f(0) = -3 :=
by
  have h1 : f(2) = 3 := 
    calc
      f(2) = -f(-2) : by exact odd_fn 2
          ... = 3      : by rw [f_neg_2]
  sorry

end f_2_plus_f_0_l291_291297


namespace annual_growth_rate_is_six_percent_l291_291246

noncomputable def annual_growth_rate : ℝ :=
let initial_income : ℝ := 255 in
let final_income : ℝ := 817 in
let years : ℕ := 20 in
let growth_factor := (final_income / initial_income)^(1 / years) in
growth_factor - 1

theorem annual_growth_rate_is_six_percent :
  abs (annual_growth_rate * 100 - 6) < 0.001 :=
by sorry

end annual_growth_rate_is_six_percent_l291_291246


namespace inequality_not_always_true_l291_291226

/-- Given x, y > 0, x^2 > y^2, and z ≠ 0, 
prove that the inequality x * z^3 > y * z^3 is not always true. -/
theorem inequality_not_always_true (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x^2 > y^2) (hz : z ≠ 0) :
  ¬ ∀ z : ℝ, z ≠ 0 → x * z^3 > y * z^3 :=
begin
  sorry
end

end inequality_not_always_true_l291_291226


namespace base_number_is_2_l291_291237

variable (x n : ℝ)
variable (b : ℝ := 19.99999999999999)

-- Define the conditions
def condition1 : Prop := n = x ^ 0.15
def condition2 : Prop := n ^ b = 8

-- State the problem
theorem base_number_is_2 (h1 : condition1 x n) (h2 : condition2 x n b) : x = 2 := by
  sorry

end base_number_is_2_l291_291237


namespace find_AC_squared_l291_291667

-- Definitions of the geometric entities and their properties
variables (A B C D E : Point)
variables (ω : Circle)
variables (AB AC BC CD AE : ℝ)
variable (on_ray : OnRay B C D)

-- Conditions translated to Lean
axiom is_isosceles_triangle (H1 : AB = AC) : is_isosceles_triangle AB AC BC
axiom circle_is_inscribed (H2 : InCircle ω A B C) : Circle_Inscribed ω A B C
axiom points_on_ray (H3 : CD = 6) : PointsOnRay CD B C D
axiom distance_given (H4 : AE = 7) : Distance A E = 7
axiom BC_given (H5 : BC = 14) : Distance B C = 14

-- The proof statement that AC^2 = 105
theorem find_AC_squared : AC^2 = 105 := by
  sorry

end find_AC_squared_l291_291667


namespace buses_encountered_l291_291091

theorem buses_encountered 
  (buses_a_to_s_interval : ℕ := 2) -- Buses from Austin to San Antonio leave every 2 hours
  (buses_s_to_a_interval : ℕ := 3) -- Buses from San Antonio to Austin leave every 3 hours
  (trip_duration : ℕ := 8) -- The trip from one city to the other takes 8 hours
  (same_highway : Prop := true) -- Buses travel on the same highway
  : ∀ start_time_a time_window, 
    (start_time_a ∈ (range 0 (trip_duration + 1)).filter (λ t, t % buses_a_to_s_interval = 0)) → 
    ((finset.range trip_duration).filter (λ t, t % buses_s_to_a_interval = 0)).card = 4 := 
  sorry

end buses_encountered_l291_291091


namespace elevator_ways_l291_291364

-- Define the problem conditions
def num_ways_to_get_off_elevator (people : List String) (floors : List ℕ) : ℕ :=
  let num_people := people.length
  let num_floors := floors.length

  -- Define combinations for Case 1
  let case1 := (∑ i in choose num_people 2, num_floors * (num_floors - 1))

  -- Define permutations for Case 2
  let case2 := !choose num_floors 3

  case1 + case2

-- Proof statement
theorem elevator_ways : num_ways_to_get_off_elevator ["A", "B", "C"] [3, 4, 5, 6, 7] = 120 := by
  sorry

end elevator_ways_l291_291364


namespace P_div_by_Q_iff_l291_291034

def P (x : ℂ) (n : ℕ) : ℂ := x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1
def Q (x : ℂ) : ℂ := x^4 + x^3 + x^2 + x + 1

theorem P_div_by_Q_iff (n : ℕ) : (Q x ∣ P x n) ↔ ¬(5 ∣ n) := sorry

end P_div_by_Q_iff_l291_291034


namespace positive_real_solutions_unique_l291_291929

variable (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (x y z : ℝ)

theorem positive_real_solutions_unique :
    x + y + z = a + b + c ∧
    4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = abc →
    (x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2) :=
by
  intros
  sorry

end positive_real_solutions_unique_l291_291929


namespace sum_of_solutions_eq_l291_291330

open Real

noncomputable def solve_eq (x : ℝ) : Prop :=
  15 / (x * (cbrt (35 - 8 * x^3))) = 2 * x + cbrt (35 - 8 * x^3)

theorem sum_of_solutions_eq : ∑ (x : ℝ) in {x : ℝ | solve_eq x}, x = 2.5 := sorry

end sum_of_solutions_eq_l291_291330


namespace decimal_zeros_l291_291797

theorem decimal_zeros (h : 2520 = 2^3 * 3^2 * 5 * 7) : 
  ∃ (n : ℕ), n = 2 ∧ (∃ d : ℚ, d = 5 / 2520 ∧ ↑d = 0.004) :=
by
  -- We assume the factorization of 2520 is correct
  have h_fact := h
  -- We need to prove there are exactly 2 zeros between the decimal point and the first non-zero digit
  sorry

end decimal_zeros_l291_291797


namespace cost_of_one_box_of_tissues_l291_291473

variable (num_toilet_paper : ℕ) (num_paper_towels : ℕ) (num_tissues : ℕ)
variable (cost_toilet_paper : ℝ) (cost_paper_towels : ℝ) (total_cost : ℝ)

theorem cost_of_one_box_of_tissues (num_toilet_paper = 10) 
                                   (num_paper_towels = 7) 
                                   (num_tissues = 3)
                                   (cost_toilet_paper = 1.50) 
                                   (cost_paper_towels = 2.00) 
                                   (total_cost = 35.00) :
  let total_cost_toilet_paper := num_toilet_paper * cost_toilet_paper,
      total_cost_paper_towels := num_paper_towels * cost_paper_towels,
      cost_left_for_tissues := total_cost - (total_cost_toilet_paper + total_cost_paper_towels),
      one_box_tissues_cost := cost_left_for_tissues / num_tissues
  in one_box_tissues_cost = 2.00 := 
sorry

end cost_of_one_box_of_tissues_l291_291473


namespace gorilla_exhibit_total_visitors_l291_291770

def hourly_visitors : List Nat := [50, 70, 90, 100, 70, 60, 80, 50]

def gorilla_percentages : List Float := [0.80, 0.75, 0.60, 0.40, 0.55, 0.70, 0.60, 0.80]

def visitors_to_gorilla_exhibit (visitors : Nat) (percentage : Float) : Float :=
  visitors * percentage

def total_gorilla_visitors : Float :=
  (List.zipWith visitors_to_gorilla_exhibit hourly_visitors gorilla_percentages).sum

theorem gorilla_exhibit_total_visitors :
  total_gorilla_visitors = 355 := 
by
  sorry

end gorilla_exhibit_total_visitors_l291_291770


namespace fraction_sum_l291_291519

theorem fraction_sum :
  (1 / 4 : ℚ) + (2 / 9) + (3 / 6) = 35 / 36 := 
sorry

end fraction_sum_l291_291519


namespace geometric_sequence_relation_l291_291607

theorem geometric_sequence_relation (a b c : ℝ) (r : ℝ)
  (h1 : -2 * r = a)
  (h2 : a * r = b)
  (h3 : b * r = c)
  (h4 : c * r = -8) :
  b = -4 ∧ a * c = 16 := by
  sorry

end geometric_sequence_relation_l291_291607


namespace land_plot_side_length_l291_291420

theorem land_plot_side_length (A : ℝ) (h : A = Real.sqrt 1024) : Real.sqrt A = 32 := 
by sorry

end land_plot_side_length_l291_291420


namespace parallel_lines_suff_cond_not_necess_l291_291185

theorem parallel_lines_suff_cond_not_necess (a : ℝ) :
  a = -2 → 
  (∀ x y : ℝ, (2 * x + y - 3 = 0) ∧ (2 * x + y + 4 = 0) → 
    (∃ a : ℝ, a = -2 ∨ a = 1)) ∧
    (a = -2 → ∃ a : ℝ, a = -2 ∨ a = 1) :=
by {
  sorry
}

end parallel_lines_suff_cond_not_necess_l291_291185


namespace benny_apples_l291_291487

theorem benny_apples (benny dan : ℕ) (total : ℕ) (H1 : dan = 9) (H2 : total = 11) (H3 : benny + dan = total) : benny = 2 :=
by
  sorry

end benny_apples_l291_291487


namespace butterfly_least_distance_l291_291847

theorem butterfly_least_distance (
  (r : ℝ) (sl : ℝ) (d1 : ℝ) (d2 : ℝ)
  (r_pos : r = 700)
  (sl_pos : sl = 250 * Real.sqrt 3)
  (start_pos : d1 = 150)
  (end_pos : d2 = 400)
  ) : 
  let θ := (2 * Real.pi * r) / sl
  let θ_half := θ / 2
  let A := (d1, 0)
  let B := (d2 * Real.cos θ_half, d2 * Real.sin θ_half)
  let d_AB := Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)
  in d_AB = 337.067 :=
by sorry

end butterfly_least_distance_l291_291847


namespace probability_all_three_colors_l291_291050

theorem probability_all_three_colors (R W Y : ℕ) (H_R : R = 2) (H_W : W = 3) (H_Y : Y = 4) :
  let total_ways := Nat.choose (R + W + Y) 4 in
  let favorable_ways :=
    (Nat.choose R 2) * (Nat.choose W 1) * (Nat.choose Y 1) +
    (Nat.choose R 1) * (Nat.choose W 2) * (Nat.choose Y 1) +
    (Nat.choose R 1) * (Nat.choose W 1) * (Nat.choose Y 2) in
  (favorable_ways.toRat / total_ways).num = 4 ∧
  (favorable_ways.toRat / total_ways).den = 7 :=
by
  sorry

end probability_all_three_colors_l291_291050


namespace starting_number_is_550_l291_291361

-- Definitions using the conditions of the problem
def approx_even_mult_of_55 (S : ℕ) : ℝ :=
  (1101 - S) / 110

def valid_multiple_of_110 (n : ℕ) : Prop :=
  110 * n ≤ 1101

-- Problem statement: Prove the starting number S must be 550 given the conditions
theorem starting_number_is_550 : 
  ∃ (S : ℕ), (approx_even_mult_of_55 S = 6.0181818181818185) ∧ valid_multiple_of_110 5 →
  S = 550 :=
by
  sorry

end starting_number_is_550_l291_291361


namespace range_of_a_l291_291920

open Real

theorem range_of_a (x y z a : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hsum : x + y + z = 1)
  (heq : a / (x * y * z) = (1 / x) + (1 / y) + (1 / z) - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
by
  sorry

end range_of_a_l291_291920


namespace hexagon_circle_count_l291_291670

-- Define a vertex type to represent the vertices of a regular hexagon
inductive Vertex : Type
| A | B | C | D | E | F

-- Define what it means for two vertices to be non-consecutive in a regular hexagon
def non_consecutive (v1 v2 : Vertex) : Prop :=
  match v1, v2 with
  | Vertex.A, Vertex.C | Vertex.A, Vertex.D | Vertex.A, Vertex.E
  | Vertex.B, Vertex.D | Vertex.B, Vertex.E | Vertex.B, Vertex.F
  | Vertex.C, Vertex.E | Vertex.C, Vertex.F
  | Vertex.D, Vertex.F | v, w => v ≠ w ∧ v ≠ (w.pred 1) ∧ v ≠ (w.succ 1)

-- Define a function to count the number of unique circles formed by these pairs
def count_circles (v : list (Vertex × Vertex)) : ℕ :=
  v.filter (λ p, non_consecutive p.1 p.2).length

-- State the main problem in Lean
theorem hexagon_circle_count : count_circles [(Vertex.A, Vertex.C), (Vertex.A, Vertex.D), (Vertex.A, Vertex.E),
                                              (Vertex.B, Vertex.D), (Vertex.B, Vertex.E), (Vertex.B, Vertex.F),
                                              (Vertex.C, Vertex.E), (Vertex.C, Vertex.F), (Vertex.D, Vertex.F)] = 3 := sorry

end hexagon_circle_count_l291_291670


namespace max_length_shortest_arc_l291_291179

theorem max_length_shortest_arc (C : ℝ) (hC : C = 84) : 
  ∃ shortest_arc_length : ℝ, shortest_arc_length = 2 :=
by
  -- now prove it
  sorry

end max_length_shortest_arc_l291_291179


namespace ellipse_C_equation_and_maximal_triangle_area_l291_291648

theorem ellipse_C_equation_and_maximal_triangle_area
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
  (eccentricity : real.sqrt (a^2 - b^2) / a = real.sqrt 3 / 2)
  (Q : ℝ × ℝ := (0, 2))
  (max_dist : ∀ P : ℝ × ℝ, P ∈ set_of (λ (P : ℝ × ℝ), P.fst^2 / a^2 + P.snd^2 / b^2 = 1) 
              → dist P Q ≤ 3) :
  (∀ (x y : ℝ), (x^2 / 3) + y^2 = 1 → 
    (∃ (m n : ℝ), (m^2 / 3) + n^2 ≤ 1 ∧ (dist (0, 0) (m, n)) < 1 ∧ 
    triangle_area ((0,0), (m,0), (0,n)) = real.sqrt 2 / 2)) :=
sorry

end ellipse_C_equation_and_maximal_triangle_area_l291_291648


namespace smallest_number_of_disks_required_l291_291299

def TotalFiles := 40
def Files_1p2MB := 4
def Files_1MB := 10
def Files_0p6MB := TotalFiles - (Files_1p2MB + Files_1MB)
def DiskSize := 2 -- in MB

theorem smallest_number_of_disks_required : 
  ∃ d : ℕ, d = 16 ∧ 
    (∀ (x_1 : ℕ), x_1 ≤ TotalFiles → (
      (Files_1p2MB ≤ x_1 ∧ 
      ∃ (y : ℕ), y = Files_0p6MB ∧ 
      x_1 + y = TotalFiles ∧ 
      ∀ x_2 : ℕ, x_2 ≤ Files_1MB → 
        (∃ (disk_size : ℕ), disk_size = DiskSize ∧ 
        (∃ (num_disks : ℕ), num_disks = d ∧ 
        x_1 ∙ 1.2 + x_2 ∙ 1 + y ∙ 0.6 ≤ num_disks ∙ disk_size))))) :=
sorry

end smallest_number_of_disks_required_l291_291299


namespace apprentice_assembling_speed_master_work_days_for_equal_assembly_l291_291045

noncomputable def total_motorcycles : ℕ := 28
noncomputable def apprentice_work_days : ℕ := 7
noncomputable def master_work_days_lt : ℕ → Prop := λ d, d < 7
noncomputable def apprentice_starts : ℕ := 2

def apprentice_average_speed (a : ℕ) : Prop :=
  total_motorcycles > apprentice_work_days * a

def master_average_speed (a m : ℕ) : Prop :=
  m = a + 2 ∧ total_motorcycles <= 6 * m

theorem apprentice_assembling_speed :
  ∃ (a : ℕ), apprentice_average_speed a ∧ a = 3 :=
begin
  sorry
end

theorem master_work_days_for_equal_assembly :
  ∀ (a m days: ℕ) (h_apprentice: apprentice_average_speed a) (h_master: master_average_speed a m),
  apprentice_starts * a + days * a = days * m → days = 3 :=
begin
  sorry
end

end apprentice_assembling_speed_master_work_days_for_equal_assembly_l291_291045


namespace inequality_sin_cos_l291_291814

theorem inequality_sin_cos 
  (a b : ℝ) (n : ℝ) (x : ℝ) 
  (ha : 0 < a) (hb : 0 < b) : 
  (a / (Real.sin x)^n) + (b / (Real.cos x)^n) ≥ (a^(2/(n+2)) + b^(2/(n+2)))^((n+2)/2) :=
sorry

end inequality_sin_cos_l291_291814


namespace winning_candidate_votes_l291_291780

def total_votes : ℕ := 100000
def winning_percentage : ℚ := 42 / 100
def expected_votes : ℚ := 42000

theorem winning_candidate_votes : winning_percentage * total_votes = expected_votes := by
  sorry

end winning_candidate_votes_l291_291780


namespace no_triangle_satisfies_sine_eq_l291_291891

theorem no_triangle_satisfies_sine_eq (A B C : ℝ) (a b c : ℝ) 
  (hA: 0 < A) (hB: 0 < B) (hC: 0 < C) 
  (hA_ineq: A < π) (hB_ineq: B < π) (hC_ineq: C < π) 
  (h_sum: A + B + C = π) 
  (sin_eq: Real.sin A + Real.sin B = Real.sin C)
  (h_tri_ineq: a + b > c ∧ a + c > b ∧ b + c > a) 
  (h_sines: a = 2 * (1) * Real.sin A ∧ b = 2 * (1) * Real.sin B ∧ c = 2 * (1) * Real.sin C) :
  False :=
sorry

end no_triangle_satisfies_sine_eq_l291_291891


namespace locus_eq_hyperbola_l291_291343

open Real

def circle (x y r : ℝ) : set (ℝ × ℝ) := {p | (p.1 - x)^2 + (p.2 - y)^2 = r^2}

def tangent_locus_eq (P : ℝ × ℝ) : Prop :=
  ∃ r, ∀ (x y : ℝ), 
    (circle (-3) 0 1).pairs (x, y) ∧ (circle 3 0 3).pairs (x, y) → 
    (dist P (-3, 0) = r + 1) ∧ (dist P (3, 0) = r + 3) ∧
    (dist P (3, 0) - dist P (-3, 0) = 2)

theorem locus_eq_hyperbola (x y : ℝ) (h : tangent_locus_eq (x, y)) : 
    x^2 - (y^2 / 8) = 1 ∧ x < 0 :=
sorry

end locus_eq_hyperbola_l291_291343


namespace digit_157_of_3_div_11_l291_291005

theorem digit_157_of_3_div_11 :
  let seq := "27" in
  let repeat_length := String.length seq in
  let n := 157 in
  let position := (n - 1) % repeat_length in
  (String.get seq position) = '2' :=
by
  -- concatenating 27 forever
  let seq := "27"
  let repeat_length := String.length seq
  let n := 157
  let position := (n - 1) % repeat_length
  have h1: (String.get seq position) = '2', from sorry,
  exact h1

end digit_157_of_3_div_11_l291_291005


namespace football_team_right_handed_players_count_l291_291304

theorem football_team_right_handed_players_count
  (total_players throwers : ℕ)
  (h1 : total_players = 120)
  (h2 : throwers = 45)
  (h3 : ∀ x, x ∈ throwers → right_handed x)
  (h4 : ∀ x, x ∉ throwers → x % 5 = 0 ∨ x % 5 = 1)
  (h5 : ∀ x, x ∉ throwers → (2 * x) % 5 = 0) : 
  let non_throwers := total_players - throwers in
  let left_handed_non_throwers := (2 / 5) * non_throwers in
  let right_handed_non_throwers := non_throwers - left_handed_non_throwers in
  let total_right_handed := throwers + right_handed_non_throwers in
  total_right_handed = 90 :=
by
  sorry

end football_team_right_handed_players_count_l291_291304


namespace equidistant_point_quadrants_l291_291966

theorem equidistant_point_quadrants (x y : ℝ) (h : 4 * x + 3 * y = 12) :
  (x > 0 ∧ y = 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end equidistant_point_quadrants_l291_291966


namespace find_negative_number_l291_291020

theorem find_negative_number : ∃ x ∈ ({} : set ℝ), x < 0 ∧ (x = -5) :=
by
  use -5
  split
  { 
    trivial 
  }
  {
    simp
  }

end find_negative_number_l291_291020


namespace tangent_line_l291_291957

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line (x y : ℝ) (h : f 2 = 6) : 13 * x - y - 20 = 0 :=
by
  -- Insert proof here
  sorry

end tangent_line_l291_291957


namespace english_textbook_cost_l291_291082

variable (cost_english_book : ℝ)

theorem english_textbook_cost :
  let geography_book_cost := 10.50
  let num_books := 35
  let total_order_cost := 630
  (num_books * cost_english_book + num_books * geography_book_cost = total_order_cost) →
  cost_english_book = 7.50 :=
by {
sorry
}

end english_textbook_cost_l291_291082


namespace second_smallest_packs_hot_dogs_l291_291510

theorem second_smallest_packs_hot_dogs (n : ℕ) :
  (∃ k : ℕ, n = 5 * k + 3) →
  n > 0 →
  ∃ m : ℕ, m < n ∧ (∃ k2 : ℕ, m = 5 * k2 + 3) →
  n = 8 :=
by
  sorry

end second_smallest_packs_hot_dogs_l291_291510


namespace quadratic_roots_is_correct_l291_291581

theorem quadratic_roots_is_correct (a b : ℝ) 
    (h1 : a + b = 16) 
    (h2 : a * b = 225) :
    (∀ x, x^2 - 16 * x + 225 = 0 ↔ x = a ∨ x = b) := sorry

end quadratic_roots_is_correct_l291_291581


namespace highest_geometric_frequency_count_l291_291120

-- Define the problem conditions and the statement to be proved
theorem highest_geometric_frequency_count :
  ∀ (vol : ℕ) (num_groups : ℕ) (cum_freq_first_seven : ℝ)
  (remaining_freqs : List ℕ) (total_freq_remaining : ℕ)
  (r : ℕ) (a : ℕ),
  vol = 100 → 
  num_groups = 10 → 
  cum_freq_first_seven = 0.79 → 
  total_freq_remaining = 21 → 
  r > 1 →
  remaining_freqs = [a, a * r, a * r ^ 2] → 
  a * (1 + r + r ^ 2) = total_freq_remaining → 
  ∃ max_freq, max_freq ∈ remaining_freqs ∧ max_freq = 12 :=
by
  intro vol num_groups cum_freq_first_seven remaining_freqs total_freq_remaining r a
  intros h_vol h_num_groups h_cum_freq_first h_total_freq_remaining h_r_pos h_geom_seq h_freq_sum
  use 12
  sorry

end highest_geometric_frequency_count_l291_291120


namespace find_complex_number_l291_291131

open Complex

def satisfies_condition_1 (z : ℂ) : Prop := abs (conj z - 3) = abs (conj z - 3 * I)

def satisfies_condition_2 (z : ℂ) : Prop := isReal (z - 1 + 5 / (z - 1))

theorem find_complex_number (z : ℂ) (h1 : satisfies_condition_1 z) (h2 : satisfies_condition_2 z) : 
  (z = 2 - 2 * I) ∨ (z = -1 + I) :=
by 
  -- This is where the proof would go
  sorry

end find_complex_number_l291_291131


namespace find_parabola_expression_l291_291898

-- Define the general form of a parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a*x^2 + b*x + c

-- Given conditions
def point_A (a b c : ℝ) : Prop := parabola a b c 4 = 0
def point_C (a b c : ℝ) : Prop := parabola a b c 0 = -4
def point_B (a b c : ℝ) : Prop := parabola a b c (-1) = 0

-- Theorem stating that the parabola y = x^2 - 3x - 4 satisfies the given conditions.
theorem find_parabola_expression : 
  ∃ (a b c : ℝ), point_A a b c ∧ point_C a b c ∧ point_B a b c ∧ (parabola a b c = λ x, x^2 - 3*x - 4) := 
sorry

end find_parabola_expression_l291_291898


namespace roller_coaster_cost_l291_291805

-- Definitions for the conditions in the problem
def ferris_wheel_cost := 2.0
def discount_multiple_rides := 1.0
def coupon_free_ticket := 1.0
def total_tickets_needed := 7.0

-- Definition for the cost of the roller coaster
variable (R : ℝ)

-- Compute the total cost, considering the conditions
def total_cost_post_discounts := (ferris_wheel_cost + R) - (discount_multiple_rides + coupon_free_ticket)

-- Assertion: the cost calculated after applying all conditions, sums up to total_tickets_needed
theorem roller_coaster_cost : 
  total_cost_post_discounts = total_tickets_needed → R = 7.0 := 
by
  admit -- skip the proof

end roller_coaster_cost_l291_291805


namespace polynomial_count_condition_l291_291500

theorem polynomial_count_condition :
  let S := { k : ℕ | k ≤ 9 }
  let Q (a b c d : ℕ) := a + b + c + d = 9 ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S
  ∃ n : ℕ, n = 220 ∧ (∃ a b c d : ℕ, Q a b c d) :=
by sorry

end polynomial_count_condition_l291_291500


namespace cyclic_quadrilateral_APCD_l291_291678

-- Given data and definitions
variables {α : Type*} [EuclideanGeometry α]
variables {A B C H A1 C1 D P D' : α}

-- Conditions
variables (h₁ : acute_triangle A B C)
variables (h₂ : is_orthocenter H A B C)
variables (h₃ : lies_on_line H A A1)
variables (h₄ : lies_on_line H C C1)
variables (h₅ : lies_on_line A1 B C)
variables (h₆ : lies_on_line C1 A B)
variables (h₇ : intersection_point H B A1 C1 D)
variables (h₈ : midpoint P B H)
variables (h₉ : reflection D A C D')

-- Prove that quadrilateral \( APCD' \) is cyclic
theorem cyclic_quadrilateral_APCD' :
  cyclic_quadrilateral A P C D' :=
sorry

end cyclic_quadrilateral_APCD_l291_291678


namespace vincent_total_laundry_loads_l291_291004

theorem vincent_total_laundry_loads :
  let wednesday_loads := 6
  let thursday_loads := 2 * wednesday_loads
  let friday_loads := thursday_loads / 2
  let saturday_loads := wednesday_loads / 3
  let total_loads := wednesday_loads + thursday_loads + friday_loads + saturday_loads
  total_loads = 26 :=
by {
  let wednesday_loads := 6
  let thursday_loads := 2 * wednesday_loads
  let friday_loads := thursday_loads / 2
  let saturday_loads := wednesday_loads / 3
  let total_loads := wednesday_loads + thursday_loads + friday_loads + saturday_loads
  show total_loads = 26
  sorry
}

end vincent_total_laundry_loads_l291_291004


namespace max_cells_primitive_dinosaur_l291_291479

section Dinosaur

universe u

-- Define a dinosaur as a structure with at least 2007 cells
structure Dinosaur (α : Type u) :=
(cells : ℕ) (connected : α → α → Prop)
(h_cells : cells ≥ 2007)
(h_connected : ∀ (x y : α), connected x y → connected y x)

-- Define a primitive dinosaur where the cells cannot be partitioned into two or more dinosaurs
structure PrimitiveDinosaur (α : Type u) extends Dinosaur α :=
(h_partition : ∀ (x : α), ¬∃ (d1 d2 : Dinosaur α), (d1.cells + d2.cells = cells) ∧ 
  (d1 ≠ d2 ∧ d1.cells ≥ 2007 ∧ d2.cells ≥ 2007))

-- Prove that the maximum number of cells in a Primitive Dinosaur is 8025
theorem max_cells_primitive_dinosaur : ∀ (α : Type u), ∃ (d : PrimitiveDinosaur α), d.cells = 8025 :=
sorry

end Dinosaur

end max_cells_primitive_dinosaur_l291_291479


namespace problem_part1_problem_part2_l291_291593

noncomputable def P (a : ℝ) : Set ℝ := { x | -1 < x ∧ x < a }
def Q : Set ℝ := { x | abs (x - 1) ≤ 1 }

theorem problem_part1 (a : ℝ) (h : a = 3) : P a = { x | -1 < x ∧ x < 3 } := 
by {
  rw h,
  refl,
  sorry
}

theorem problem_part2 (a : ℝ) : P a ∩ Q = Q → a ≥ 2 := 
by {
  sorry
}

end problem_part1_problem_part2_l291_291593


namespace fraction_simplifies_to_two_l291_291796

theorem fraction_simplifies_to_two :
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) = 2 := by
  sorry

end fraction_simplifies_to_two_l291_291796


namespace number_of_moles_H2O_formed_l291_291901

-- Define the reaction and initial conditions
def balanced_reaction : Prop :=
  ∀ (HCH3CO2 NaOH H2O CH3COONa : Type),
    (1 : ℕ) * HCH3CO2 + (1 : ℕ) * NaOH = (1 : ℕ) * H2O + (1 : ℕ) * CH3COONa

def initial_moles_HCH3CO2 := 1
def initial_moles_NaOH := 1

-- The main theorem to prove
theorem number_of_moles_H2O_formed : initial_moles_HCH3CO2 = 1 → initial_moles_NaOH = 1 → balanced_reaction → (1 : ℕ) = 1 :=
begin
  sorry
end

end number_of_moles_H2O_formed_l291_291901


namespace number_of_even_digits_in_base7_of_528_l291_291140

/-
  Define the base-7 representation of a number and a predicate to count even digits.
-/

-- Definition of base-7 digit representation
def base7_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else (List.unfoldr (λ n, if n = 0 then Option.none else some (n % 7, n / 7)) n).reverse

-- Predicate to check if a digit is even
def is_even (d : ℕ) : Bool := d % 2 = 0

-- Counting the even digits in base-7 representation
def count_even_digits_in_base7 (n : ℕ) : ℕ :=
  (base7_repr n).countp is_even

-- The target theorem to prove
theorem number_of_even_digits_in_base7_of_528 : count_even_digits_in_base7 528 = 0 :=
by sorry

end number_of_even_digits_in_base7_of_528_l291_291140


namespace find_x_l291_291110

noncomputable def f₁ (x : ℝ) : ℝ := 3/2 - 4 / (4 * x + 2)

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
if h : n = 1 then f₁ x else
have n > 0, from Nat.succ_le_succ_iff.mp (Nat.one_le_of_lt h),
f₁ (f (n-1) x)

theorem find_x : ∃ x : ℝ, f 6 x = 2 * x - 2 → x = 7/4 :=
sorry

end find_x_l291_291110


namespace muscovy_more_than_cayuga_l291_291774

theorem muscovy_more_than_cayuga
  (M C K : ℕ)
  (h1 : M + C + K = 90)
  (h2 : M = 39)
  (h3 : M = 2 * C + 3 + C) :
  M - C = 27 := by
  sorry

end muscovy_more_than_cayuga_l291_291774


namespace wilson_total_cost_l291_291029

noncomputable def total_cost_wilson_pays : ℝ :=
let hamburger_price : ℝ := 5
let cola_price : ℝ := 2
let fries_price : ℝ := 3
let sundae_price : ℝ := 4
let nugget_price : ℝ := 1.5
let salad_price : ℝ := 6.25
let hamburger_count : ℕ := 2
let cola_count : ℕ := 3
let nugget_count : ℕ := 4

let total_before_discounts := (hamburger_count * hamburger_price) +
                              (cola_count * cola_price) +
                              fries_price +
                              sundae_price +
                              (nugget_count * nugget_price) +
                              salad_price

let free_nugget_discount := 1 * nugget_price
let total_after_promotion := total_before_discounts - free_nugget_discount
let coupon_discount := 4
let total_after_coupon := total_after_promotion - coupon_discount
let loyalty_discount := 0.10 * total_after_coupon
let total_after_loyalty := total_after_coupon - loyalty_discount

total_after_loyalty

theorem wilson_total_cost : total_cost_wilson_pays = 26.77 := 
by
  sorry

end wilson_total_cost_l291_291029


namespace angle_relationship_l291_291104

-- Define the points and conditions
variables (x1 x2 : ℝ)
variables (y1 y2 : ℝ) (C_x C_y : ℝ)

-- Conditions
def are_on_graph (x1 x2 : ℝ) : Prop := (y1 = 1 / x1) ∧ (y2 = 1 / x2)
def x1_lt_x2 (x1 x2 : ℝ) : Prop := (0 < x1) ∧ (x1 < x2)
def midpoint_C (x1 x2 C_x : ℝ) (y1 y2 C_y : ℝ) : Prop := 
  (C_x = (x1 + x2) / 2) ∧ (C_y = (y1 + y2) / 2)
def distance_condition (x1 x2 : ℝ) (y1 y2 : ℝ) : Prop :=
  (√((x2 - x1)^2 + (y2 - y1)^2) = 2 * √(x1^2 + (1 / x1^2)))

-- Main theorem to prove
theorem angle_relationship
  (h₁ : are_on_graph x1 x2)
  (h₂ : x1_lt_x2 x1 x2)
  (h₃ : distance_condition x1 x2 y1 y2)
  (h₄ : midpoint_C x1 x2 C_x y1 y2 C_y) 
  : ∃ θ φ : ℝ, θ = 3 * φ :=
by sorry

end angle_relationship_l291_291104


namespace triangle_concyclic_points_l291_291669

-- Define points and properties based on the given conditions.
theorem triangle_concyclic_points
  (A B C O D E X Y : Type) [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D] [decidable_eq E] [decidable_eq X] [decidable_eq Y]
  (circumcenter : triangle ABC → O)
  (angle_bisector : ∀ {A B C : Type}, A ≠ B → (A → B → C → Type))
  (reflection : ∀ (D : Type) (M : midpoint B C), E)
  (midpoint : (B → C) → Type)
  (perpendicular : (line B C) → (D → X) → Type)
  (on_line : (line B C) → Type)
  (triangle_property1 : A ≠ C)
  (triangle_property2 : circumcenter (triangle ABC) = O)
  (bisector_D : bisector angle ⦃A B C⦄ (line B C))
  (reflection_E : reflection D (midpoint B C) = E)
  (perpendicular_X : perpendicular (line B C) D = X)
  (perpendicular_Y : perpendicular (line B C) E intersects (line AD) = Y) :

  -- Statement to prove
  ∃ (circle : Type), (on_circle B circle) ∧ (on_circle X circle) ∧ (on_circle C circle) ∧ (on_circle Y circle) :=
sorry

end triangle_concyclic_points_l291_291669


namespace unlike_radical_expressions_l291_291800

variable (x : ℝ)

theorem unlike_radical_expressions :
  ¬ ((∃ k : ℝ, ∀ y : ℝ, (y = sqrt 18) → (∃ m : ℝ, ∀ z : ℝ, (z = sqrt 18) → (k * y = m * z)))) ∧
  ¬ ((∃ k : ℝ, ∀ y : ℝ, (y = sqrt 12) → (∃ m : ℝ, ∀ z : ℝ, (z = sqrt 75) → (k * y = m * z)))) ∧
  ¬ ((∃ k : ℝ, ∀ y : ℝ, (y = sqrt (1 / 3)) → (∃ m : ℝ, ∀ z : ℝ, (z = sqrt 27) → (k * y = m * z)))) ∧
  (∀ k : ℝ, k ≠ 1 → k * sqrt x ≠ sqrt x) := by
  sorry

end unlike_radical_expressions_l291_291800


namespace sqrt10_parts_sqrt6_value_sqrt3_opposite_l291_291724

-- Problem 1
theorem sqrt10_parts : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 → (⌊Real.sqrt 10⌋ = 3 ∧ Real.sqrt 10 - 3 = Real.sqrt 10 - ⌊Real.sqrt 10⌋) :=
by
  sorry

-- Problem 2
theorem sqrt6_value (a b : ℝ) : a = Real.sqrt 6 - 2 ∧ b = 3 → (a + b - Real.sqrt 6 = 1) :=
by
  sorry

-- Problem 3
theorem sqrt3_opposite (x y : ℝ) : x = 13 ∧ y = Real.sqrt 3 - 1 → (-(x - y) = Real.sqrt 3 - 14) :=
by
  sorry

end sqrt10_parts_sqrt6_value_sqrt3_opposite_l291_291724


namespace polar_eq_line_l_rectangular_eq_curve_C_min_OM_ON_l291_291650

-- Definition of the parametric equations of the line l
def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (6 + real.sqrt 3 * t, -t)

-- Definition of the polar coordinate equation of curve C
def polar_curve_C (α : ℝ) (h : 0 < α ∧ α < real.pi / 2) : ℝ :=
  2 * real.cos α

-- Stating the theorems corresponding to parts (Ⅰ) and (Ⅱ)

-- Part (Ⅰ)
theorem polar_eq_line_l (ρ θ : ℝ) :
  ρ * real.cos θ + real.sqrt 3 * ρ * real.sin θ = 6 ↔
  ∃ t, (6 + real.sqrt 3 * t, -t) = (ρ * real.cos θ, ρ * real.sin θ) :=
sorry

theorem rectangular_eq_curve_C (x y : ℝ) :
  (x - 1)^2 + y^2 = 1 ↔
  ∃ α (h : 0 < α ∧ α < real.pi / 2), (x, y) = (polar_curve_C α h * real.cos α, polar_curve_C α h * real.sin α) :=
sorry

-- Part (Ⅱ)
theorem min_OM_ON (OM ON : ℝ) (θ₀ : ℝ) :
  (OM = 6 / (real.cos θ₀ + real.sqrt 3 * real.sin θ₀)) →
  (ON = 2 * real.cos θ₀) →
  ∃ θ₀, (real.sin (2 * θ₀ + real.pi / 6) + 1 / 2) ≠ 0 ∧
  (OM / ON = 2) :=
sorry

end polar_eq_line_l_rectangular_eq_curve_C_min_OM_ON_l291_291650


namespace area_pentagon_line_segments_l291_291345

theorem area_pentagon_line_segments (ABCDE : Type) (length : ℕ) (area : ℝ) (m n : ℝ):
  (∀ (a b c d e: ABCDE), length = 2) → (∃ (m n : ℝ), area = sqrt m + sqrt n ∧ m + n = 23) :=
sorry

end area_pentagon_line_segments_l291_291345


namespace find_m_l291_291983

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)

theorem find_m (h : (1 + 3, m - 2) = (4, m - 2) ∧ (4 * 3 + (m - 2) * (-2) = 0)) : m = 8 := by
  sorry

end find_m_l291_291983


namespace angle_C_60_range_of_k_l291_291571

variable (A B C a b c : ℝ)
variable (λ m n : ℝ × ℝ)

-- Conditions
axiom (h1 : m = (Real.sin A, b + c))
axiom (h2 : n = (Real.sin C - Real.sin B, a - b))
axiom (h3 : ∃ λ, m = λ • n)

-- First problem: Prove ∠C = 60°
theorem angle_C_60 (h1 : m = (Real.sin A, b + c)) (h2 : n = (Real.sin C - Real.sin B, a - b)) (h3 : ∃ λ, m = λ • n) :
  C = 60 := 
sorry

-- Second problem: Prove range of k
variable (k : ℝ)
axiom (h4 : a + b = k * c)

theorem range_of_k (h4 : a + b = k * c) (C_eq_60 : C = 60) :
  1 < k ∧ k ≤ 2 := 
sorry

end angle_C_60_range_of_k_l291_291571


namespace integer_coefficient_polynomial_l291_291520

theorem integer_coefficient_polynomial (f : ℤ[X]) (h : ∀ n : ℕ, 0 < n → f.eval (n : ℤ) ∣ 2^n - 1) : f = 1 ∨ f = -1 :=
by sorry

end integer_coefficient_polynomial_l291_291520


namespace p_necessary_not_sufficient_for_q_l291_291573

def p (x : ℝ) : Prop := abs x = -x
def q (x : ℝ) : Prop := x^2 ≥ -x

theorem p_necessary_not_sufficient_for_q : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l291_291573


namespace solve_p_plus_q_l291_291064

-- Definitions based on conditions
def width : ℝ := 15
def length : ℝ := 20
def area_of_triangle : ℝ := 50

-- Parameters of height in the form h = p / q where p and q are relatively prime integers
variables {p q : ℕ}
axiom relatively_prime : Nat.gcd p q = 1

-- Height expressed as quotient
noncomputable def height : ℝ := p / q

-- Derived: Coordinates of the centers of the faces
def center1 : (ℝ × ℝ × ℝ) := (width / 2, 0, height / 2)
def center2 : (ℝ × ℝ × ℝ) := (0, length / 2, height / 2)
def center3 : (ℝ × ℝ × ℝ) := (width / 2, length / 2, 0)

-- Calculate the sides of the triangle
def side1_length : ℝ := Real.sqrt ((width / 2) ^ 2 + (length / 2) ^ 2)
def side2_length : ℝ := Real.sqrt ((length / 2) ^ 2 + (height / 2) ^ 2)
def side3_length : ℝ := Real.sqrt ((width / 2) ^ 2 + (height / 2) ^ 2)

-- Prove that p + q = 40
theorem solve_p_plus_q : p + q = 40 := 
by
-- The proof steps would demonstrate that given the conditions and calculations, 
-- the integers p and q must satisfy p = 39 and q = 1, hence p + q = 40.
sorry

end solve_p_plus_q_l291_291064


namespace certain_number_eq_neg_thirteen_over_two_l291_291618

noncomputable def CertainNumber (w : ℝ) : ℝ := 13 * w / (1 - w)

theorem certain_number_eq_neg_thirteen_over_two (w : ℝ) (h : w ^ 2 = 1) (hz : 1 - w ≠ 0) :
  CertainNumber w = -13 / 2 :=
sorry

end certain_number_eq_neg_thirteen_over_two_l291_291618


namespace max_empty_squares_l291_291640

theorem max_empty_squares (board_size : ℕ) (total_cells : ℕ) 
  (initial_cockroaches : ℕ) (adjacent : ℕ → ℕ → Prop) 
  (different : ℕ → ℕ → Prop) :
  board_size = 8 → total_cells = 64 → initial_cockroaches = 2 →
  (∀ s : ℕ, s < total_cells → ∃ s1 s2 : ℕ, adjacent s s1 ∧ 
              adjacent s s2 ∧ 
              different s1 s2) →
  ∃ max_empty_cells : ℕ, max_empty_cells = 24 :=
by
  intros h_board_size h_total_cells h_initial_cockroaches h_moves
  sorry

end max_empty_squares_l291_291640


namespace transform_square_to_triangle_impossible_l291_291269

theorem transform_square_to_triangle_impossible :
  ¬ ∃ (polygon : Type) [is_square polygon] [is_equilateral_triangle polygon],
  (preserves_area polygon ∧ preserves_perimeter polygon) :=
begin
  sorry
end

-- Auxiliary Definitions and Classes (placeholders for actual definitions):
class is_square (polygon : Type) :=
  (side_length : ℝ)
  (area : ℝ := side_length * side_length)
  (perimeter : ℝ := 4 * side_length)

class is_equilateral_triangle (polygon : Type) :=
  (side_length : ℝ)
  (area : ℝ := (sqrt 3 / 4) * side_length^2)
  (perimeter : ℝ := 3 * side_length)

def preserves_area (polygon : Type) [is_square polygon] [is_equilateral_triangle polygon] : Prop :=
  is_square.area = is_equilateral_triangle.area polygon

def preserves_perimeter (polygon : Type) [is_square polygon] [is_equilateral_triangle polygon] : Prop :=
  is_square.perimeter = is_equilateral_triangle.perimeter polygon

end transform_square_to_triangle_impossible_l291_291269


namespace gcf_60_75_l291_291398

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l291_291398


namespace perp_condition_l291_291431

noncomputable def circle (α : Type) [metric_space α] := 
  { center : α // ∃ r : ℝ, r > 0 }

variables {α : Type} [metric_space α]

-- Given Definitions
variables {ω Γ : circle α}
variables {O : α} -- Center of ω
variables {S T P A B : α} -- Points as described

-- Conditions
variable (h_tangent_internal : ∃ rω rΓ : ℝ, rω > 0 ∧ rΓ > 0 ∧ dist O S = rΓ - rω)
variable (h_tangent_AB_T : ∃ l : α → α, is_line l ∧ ∀ x, mem x (segment AB) → tangent ω x T)
variable (h_P_on_AO : ∃ l : α → α, is_line l ∧ ∀ x, mem x (line_segment A O) → x = P)

-- Mathematical Problem
theorem perp_condition :
  is_perpendicular (line_segment P B) (line_segment A B) ↔ 
  is_perpendicular (line_segment P S) (line_segment T S) :=
sorry

end perp_condition_l291_291431


namespace midlines_perpendicular_iff_diagonals_equal_l291_291984

variable (A B C D E F G H : Type) [AddCommGroup A] [Module ℝ A]

-- Definitions of points and diagonals
variable (AB BC CD DA : A)
variable (AC BD : A)
variable (zero_vector : A := 0)

-- Midpoints
variable (E F G H : A)

-- Definitions for the proof
def midpoint (X Y : A) : A := (X + Y) / 2

-- Midlines defined as half-diagonals
def length (X : A) : ℝ := ↑(norm X.toAddMonoidHom)
def midline_EF := length (midpoint E F)
def midline_GH := length (midpoint G H)

-- Statement of the problem
theorem midlines_perpendicular_iff_diagonals_equal 
  (AC_eq_BD : length AC = length BD) : 
  (midline_EF = length AC / 2) ∧ (midline_GH = length BD / 2) ∧ 
  (midline_EF = midline_GH) ↔ (AC = BD) :=
by
  sorry

end midlines_perpendicular_iff_diagonals_equal_l291_291984


namespace sixtieth_term_correct_sum_of_first_sixty_terms_correct_l291_291749

variable (a1 : ℤ) (a15 : ℤ) (a60 : ℤ) (S60 : ℤ) (d : ℤ)

-- Conditions
def first_term : Prop := a1 = 7
def fifteenth_term : Prop := a15 = 35
def sixtieth_term_given_conditions := a60 = a1 + 59 * d
def sum_of_first_sixty_terms := S60 = 60 * (a1 + a60) / 2

-- Correct answers derived from conditions
theorem sixtieth_term_correct : first_term ∧ fifteenth_term ∧ sixtieth_term_given_conditions → a60 = 125 := by
  sorry

theorem sum_of_first_sixty_terms_correct : first_term ∧ sixtieth_term_given_conditions ∧ sum_of_first_sixty_terms → S60 = 3960 := by
  sorry

end sixtieth_term_correct_sum_of_first_sixty_terms_correct_l291_291749


namespace printer_time_equation_l291_291831

theorem printer_time_equation (x : ℝ) (rate1 rate2 : ℝ) (flyers1 flyers2 : ℝ)
  (h1 : rate1 = 100) (h2 : flyers1 = 1000) (h3 : flyers2 = 1000) 
  (h4 : flyers1 / rate1 = 10) (h5 : flyers1 / (rate1 + rate2) = 4) : 
  1 / 10 + 1 / x = 1 / 4 :=
by 
  sorry

end printer_time_equation_l291_291831


namespace find_profit_range_l291_291306

noncomputable def profit_range (x : ℝ) : Prop :=
  0 < x → 0.15 * (1 + 0.25 * x) * (100000 - x) ≥ 0.15 * 100000

theorem find_profit_range (x : ℝ) : profit_range x → 0 < x ∧ x ≤ 6 :=
by
  sorry

end find_profit_range_l291_291306


namespace shaded_areas_different_l291_291509

/-
Question: How do the shaded areas of three different large squares (I, II, and III) compare?
Conditions:
1. Square I has diagonals drawn, and small squares are shaded at each corner where diagonals meet the sides.
2. Square II has vertical and horizontal lines drawn through the midpoints, creating four smaller squares, with one centrally shaded.
3. Square III has one diagonal from one corner to the center and a straight line from the midpoint of the opposite side to the center, creating various triangles and trapezoids, with a trapezoid area around the center being shaded.
Proof:
Prove that the shaded areas of squares I, II, and III are all different given the conditions on how squares I, II, and III are partitioned and shaded.
-/
theorem shaded_areas_different :
  ∀ (a : ℝ) (A1 A2 A3 : ℝ), (A1 = 1/4 * a^2) ∧ (A2 = 1/4 * a^2) ∧ (A3 = 3/8 * a^2) → 
  A1 ≠ A3 ∧ A2 ≠ A3 :=
by
  sorry

end shaded_areas_different_l291_291509


namespace tangent_line_at_zero_monotonically_increasing_iff_l291_291589

variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - a * x^2 + 1

theorem tangent_line_at_zero : 
  ∃ (m b : ℝ), (∀ x, f x = m * x + b) ∧ m = 1 ∧ b = 2 := sorry

theorem monotonically_increasing_iff :
  (∀ x > 0, ∃ y, y ≥ x ∧ ∀ z, 0 < z → z ≤ y → f' x ≥ 0) ↔ a ≤ Real.exp 1 / 2 := sorry

end tangent_line_at_zero_monotonically_increasing_iff_l291_291589


namespace pencils_in_each_box_l291_291335

open Nat

theorem pencils_in_each_box (boxes pencils_given_to_Lauren pencils_left pencils_each_box more_pencils : ℕ)
  (h1 : boxes = 2)
  (h2 : pencils_given_to_Lauren = 6)
  (h3 : pencils_left = 9)
  (h4 : more_pencils = 3)
  (h5 : pencils_given_to_Matt = pencils_given_to_Lauren + more_pencils)
  (h6 : pencils_each_box = (pencils_given_to_Lauren + pencils_given_to_Matt + pencils_left) / boxes) :
  pencils_each_box = 12 := by
  sorry

end pencils_in_each_box_l291_291335


namespace _l291_291376

open_locale real

variables {A B C P D : Type} [triangle A B C] 

-- Definitions for distances
variables {PA PB PD AD CD : ℝ}
variables {APB ACB : ℝ}

noncomputable def conditions (A B C P D : Type) [triangle A B C] :=
  PA = PB ∧  -- P is equidistant from A and B
  angle APB = 2 * angle ACB ∧  -- angle APB is twice angle ACB
  PB = 5 ∧  -- PB = 5
  PD = 3 ∧  -- PD = 3
  (exists D : Point, is_intersection (Line_through_points A C) (Line_through_points B P) D) -- D is intersection

noncomputable def product_of_segments (A B C P D : Type) [triangle A B C] :=
  AD * CD = 14  -- The main proof statement

noncomputable theorem triangle_point_product
  {A B C P D : Type} [triangle A B C] :
  conditions A B C P D →
  product_of_segments A B C P D :=
by
  sorry

end _l291_291376


namespace average_weight_all_boys_l291_291040

theorem average_weight_all_boys 
  (a1 : ℝ) (b1 : ℕ) (a2 : ℝ) (b2 : ℕ)
  (h1 : a1 = 50.25) (h2 : b1 = 24) 
  (h3 : a2 = 45.15) (h4 : b2 = 8) :
  (b1 * a1 + b2 * a2) / (b1 + b2) = 49 :=
by 
  rw [h1, h2, h3, h4]
  norm_num
  done
  -- the proof should be placed here
  sorry

end average_weight_all_boys_l291_291040


namespace log_expression_in_terms_of_a_l291_291232

theorem log_expression_in_terms_of_a (a : ℝ) (h : 3 * a = 2) : log 3 8 - 2 * log 3 6 = a - 2 :=
by
  sorry

end log_expression_in_terms_of_a_l291_291232


namespace fraction_sum_eq_five_fourths_l291_291162

theorem fraction_sum_eq_five_fourths (a b c : ℚ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4) :
  (a + b) / c = 5 / 4 :=
by
  sorry

end fraction_sum_eq_five_fourths_l291_291162


namespace polar_eq_line_l_rectangular_eq_curve_C_min_OM_ON_l291_291649

-- Definition of the parametric equations of the line l
def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (6 + real.sqrt 3 * t, -t)

-- Definition of the polar coordinate equation of curve C
def polar_curve_C (α : ℝ) (h : 0 < α ∧ α < real.pi / 2) : ℝ :=
  2 * real.cos α

-- Stating the theorems corresponding to parts (Ⅰ) and (Ⅱ)

-- Part (Ⅰ)
theorem polar_eq_line_l (ρ θ : ℝ) :
  ρ * real.cos θ + real.sqrt 3 * ρ * real.sin θ = 6 ↔
  ∃ t, (6 + real.sqrt 3 * t, -t) = (ρ * real.cos θ, ρ * real.sin θ) :=
sorry

theorem rectangular_eq_curve_C (x y : ℝ) :
  (x - 1)^2 + y^2 = 1 ↔
  ∃ α (h : 0 < α ∧ α < real.pi / 2), (x, y) = (polar_curve_C α h * real.cos α, polar_curve_C α h * real.sin α) :=
sorry

-- Part (Ⅱ)
theorem min_OM_ON (OM ON : ℝ) (θ₀ : ℝ) :
  (OM = 6 / (real.cos θ₀ + real.sqrt 3 * real.sin θ₀)) →
  (ON = 2 * real.cos θ₀) →
  ∃ θ₀, (real.sin (2 * θ₀ + real.pi / 6) + 1 / 2) ≠ 0 ∧
  (OM / ON = 2) :=
sorry

end polar_eq_line_l_rectangular_eq_curve_C_min_OM_ON_l291_291649


namespace equilateral_triangle_segment_length_l291_291463

theorem equilateral_triangle_segment_length :
  ∀ (A B C D E F G : Type) [IsEquilateralTriangle ABC] (h1 : side_length ABC = 10)
  (h2 : AD = 3) [IsEquilateralTriangle ADE] [IsEquilateralTriangle BDG] [IsEquilateralTriangle CEF],
  (FG = 4) := by
  sorry

end equilateral_triangle_segment_length_l291_291463


namespace ellipse_x_intercepts_l291_291481

noncomputable def ellipse_foci : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 3), (4, 0))
noncomputable def x_intercept1 : (ℝ × ℝ) := (0, 0)
noncomputable def x_intercept2 : (ℝ × ℝ) := (11/2, 0)
noncomputable def x_intercept3 : (ℝ × ℝ) := (-3/2, 0)

theorem ellipse_x_intercepts :
  ∃ (x_intercepts : set (ℝ × ℝ)),
    x_intercepts = {x_intercept1, x_intercept2, x_intercept3} ∧
    (∀ (x y : ℝ), (x, y) ∈ x_intercepts → y = 0 ∧
                                abs (x - 0) + abs (x - 4) = 7) := sorry

end ellipse_x_intercepts_l291_291481


namespace cone_radius_l291_291949

open Real

theorem cone_radius
  (l : ℝ) (L : ℝ) (h_l : l = 5) (h_L : L = 15 * π) :
  ∃ r : ℝ, L = π * r * l ∧ r = 3 :=
by
  sorry

end cone_radius_l291_291949


namespace equal_sets_A_equal_sets_B_equal_sets_C_equal_sets_D_l291_291864

theorem equal_sets_A (M N : Set (ℤ × ℤ)) (hM : M = {(-9, 3)}) (hN : N = {-9, 3}) : M ≠ N := by
  intro h
  sorry

theorem equal_sets_B (M N : Set ℕ) (hM : M = ∅) (hN : N = {0}) : M ≠ N := by
  intro h
  sorry
  
theorem equal_sets_C (M N : Set ℝ) (hM : M = { x | -5 < x ∧ x < 3 }) (hN : N = { x | -5 < x ∧ x < 3 }) : M = N := by
  rw [hM, hN]
  sorry

theorem equal_sets_D (M N : Set ℝ) (hM : M = { x | x^2 - 3*x + 2 = 0 }) (hN : N = { y | y^2 - 3*y + 2 = 0 }) : M = N := by
  simp only [Set.setOf_eq, Function.comp, eq_self_iff_true]
  sorry

end equal_sets_A_equal_sets_B_equal_sets_C_equal_sets_D_l291_291864


namespace range_of_a_l291_291575

theorem range_of_a (a : ℝ) : 
  (∀ x, (x^2 + x + a = 0 → x ∈ ({-3, 2} : set ℝ))) ↔ (a = -6 ∨ a > 1/4) :=
by sorry

end range_of_a_l291_291575


namespace number_of_a_values_l291_291157

theorem number_of_a_values : 
  (∀ a : ℝ, vertex_of_parabola a ∈ line_through_vertex a) → 
  (number_of_solutions a^2 = a) = 2 :=
by
  def vertex_of_parabola (a : ℝ) : ℝ × ℝ := (0, a^2)
  def line_through_vertex (a : ℝ) (p : ℝ × ℝ) : Prop := p.2 = 2 * p.1 + a
  def number_of_solutions (eq : ℝ → ℝ) : ℕ := (solutions of eq).card
  sorry

end number_of_a_values_l291_291157


namespace part1_part2_l291_291978

def point := ℝ × ℝ × ℝ
def vector := ℝ × ℝ × ℝ

def dot_product (v₁ v₂ : vector) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

def magnitude (v : vector) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def perpendicular (v₁ v₂ : vector) : Prop :=
  dot_product v₁ v₂ = 0

def distance_from_point_to_line (p₀ p₁ p₂ : point) : ℝ :=
  let u := (p₁.1 - p₀.1, p₁.2 - p₀.2, p₁.3 - p₀.3)
  let v := (p₂.1 - p₀.1, p₂.2 - p₀.2, p₂.3 - p₀.3)
  let u_magnitude := magnitude u
  let u_unit := (u.1 / u_magnitude, u.2 / u_magnitude, u.3 / u_magnitude)
  let proj := dot_product v u_unit
  let distance_squared := dot_product v v - proj * proj
  real.sqrt distance_squared

def P : point := (-2, 0, 2)
def M : point := (-1, 1, 2)
def N : point := (-3, 0, 4)
def a : vector := (1, 1, 0)
def b : vector := (-1, 0, 2)

theorem part1 :
  (k : ℝ) (perpendicular ((k * a.1 + b.1, k * a.2 + b.2, k * a.3 + b.3))
                        (k * a.1 - 2 * b.1, k * a.2 - 2 * b.2, k * a.3 - 2 * b.3)) ↔ k = 2 ∨ k = -5 / 2 :=
sorry

theorem part2 :
  distance_from_point_to_line P M N = 3 * real.sqrt 2 / 2 :=
sorry

end part1_part2_l291_291978


namespace polynomial_coprime_sequence_exists_l291_291808

theorem polynomial_coprime_sequence_exists : 
  ∃ f : ℤ[X], degree f = 2007 ∧ (∀ n : ℤ, ∀ m ≥ 1, Nat.coprime (int.of_nat n) (int.of_nat (eval n (f^[m])))) :=
sorry

end polynomial_coprime_sequence_exists_l291_291808


namespace ratio_of_areas_l291_291365

-- Definitions based on given conditions
variable (n : ℝ) -- n is a real number
variable (square_area : ℝ) -- Area of the square

-- Given conditions
def right_triangle : Prop := true
def point_on_hypotenuse : Prop := true
def lines_parallel_to_legs : Prop := true

-- Conditions assumed in the problem
def conditions := right_triangle ∧ point_on_hypotenuse ∧ lines_parallel_to_legs ∧ 
                  (square_area = 1) ∧ (n > 0)

-- The desired proof problem
theorem ratio_of_areas (h : conditions) : 
  let triangle1_area := n * square_area in
  let r := 2 * n in
  let s := 1 / r in
  let triangle2_area := 1 / (4 * n) in
  (triangle2_area / square_area) = (1 / (4 * n)) := by
  sorry

end ratio_of_areas_l291_291365


namespace min_value_MP_plus_3_div_2_MF1_range_MP_plus_MF1_l291_291202

noncomputable def ellipse := {p : ℝ × ℝ // (p.1^2 / 9) + (p.2^2 / 5) = 1}
def point_P : ℝ × ℝ := (1, 1)
def focus_F1 : ℝ × ℝ := (-2, 0)
def dist (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def MP (M : ellipse) : ℝ := dist (M.1) point_P
def MF1 (M : ellipse) : ℝ := dist (M.1) focus_F1

theorem min_value_MP_plus_3_div_2_MF1 :
  ∃ M : ellipse, MP M + (3 / 2) * MF1 M = 11 / 2 :=
sorry

theorem range_MP_plus_MF1 :
  ∀ M : ellipse, 6 - real.sqrt 2 ≤ MP M + MF1 M ∧ MP M + MF1 M ≤ 6 + real.sqrt 2 :=
sorry

end min_value_MP_plus_3_div_2_MF1_range_MP_plus_MF1_l291_291202


namespace angle_A_l291_291243

variable (a b c : ℝ) (A B C : ℝ)

-- Hypothesis: In triangle ABC, (a + c)(a - c) = b(b + c)
def condition (a b c : ℝ) : Prop := (a + c) * (a - c) = b * (b + c)

-- The goal is to show that under given conditions, ∠A = 2π/3
theorem angle_A (h : condition a b c) : A = 2 * π / 3 :=
sorry

end angle_A_l291_291243


namespace marie_messages_days_l291_291692

theorem marie_messages_days (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) (days : ℕ) :
  initial_messages = 98 ∧ read_per_day = 20 ∧ new_per_day = 6 → days = 7 :=
by
  sorry

end marie_messages_days_l291_291692


namespace IncorrectStatement_l291_291013

-- Definitions of the events
def EventA (planeShot : ℕ → Prop) : Prop := planeShot 1 ∧ planeShot 2
def EventB (planeShot : ℕ → Prop) : Prop := ¬planeShot 1 ∧ ¬planeShot 2
def EventC (planeShot : ℕ → Prop) : Prop := (planeShot 1 ∧ ¬planeShot 2) ∨ (¬planeShot 1 ∧ planeShot 2)
def EventD (planeShot : ℕ → Prop) : Prop := planeShot 1 ∨ planeShot 2

-- Theorem statement to be proved (negation of the incorrect statement)
theorem IncorrectStatement (planeShot : ℕ → Prop) :
  ¬((EventA planeShot ∨ EventC planeShot) = (EventB planeShot ∨ EventD planeShot)) :=
by
  sorry

end IncorrectStatement_l291_291013


namespace locus_of_C_band_l291_291923

-- Let 'center' be the center of circle k
def center : Point := sorry

-- Define the circle k with radius 1 cm
def circle (p : Point) : Prop := dist center p = 1

-- Define the line e which is 2 cm from the center of the circle
def line_e (p : Point) : Prop := dist center p = 2 ∧ p.x = sorry

-- Define an arbitrary point A in the half-plane defined by e that does not contain k
def point_A : Point := sorry
def half_plane (p : Point) : Prop := point_A.x < p.x

-- Define the parallelogram ABCD with vertex B on circle k, and D on line e
structure Parallelogram (A B C D : Point) : Prop :=
(B_on_circle : circle B)
(D_on_line : line_e D)
(mid_AC_BD_symmetric : sorry)  -- This is to be formalized reflecting symmetry condition

-- The locus of point C
def locus_C (C : Point) : Prop :=
  ∃ (A B D : Point), Parallelogram A B C D ∧ 1 ≤ dist line_e.point C ≤ 3

-- Main statement defining the proof problem
theorem locus_of_C_band : 
  ∀ (C : Point), locus_C C → 
  1 ≤ dist line_e.point C ∧ dist line_e.point C ≤ 3 :=
sorry

end locus_of_C_band_l291_291923


namespace tomatoes_reaped_l291_291001

variable (Y : ℕ)

theorem tomatoes_reaped (h1 : ∀ Y, 2 * Y + 50 = 290) : Y = 120 := by
  have h2 : 2 * Y = 240 := by
    calc
    2 * Y + 50 = 290 : h1 Y
    2 * Y = 290 - 50 : by rw [Eq.subst (h1 Y)]
    2 * Y = 240 : by linarith
  exact Eq.symm (Nat.div_eq_of_eq_mul_left (dec_trivial : 0 < 2) h2)

end tomatoes_reaped_l291_291001


namespace cube_side_length_l291_291629

-- Given conditions for the problem
def surface_area (a : ℝ) : ℝ := 6 * a^2

-- Theorem statement
theorem cube_side_length (h : surface_area a = 864) : a = 12 :=
by
  sorry

end cube_side_length_l291_291629


namespace greatest_value_of_k_l291_291682

theorem greatest_value_of_k (k : ℕ) : ∃ k₀ : ℕ, (∀ k₁ : ℕ, b = 100^k₁ ∧ b ∣ 50! → k₁ ≤ k₀) ∧ k₀ = 6 :=
by {
  let a := (∏ i in (Finset.range 51).erase 0, i),
  let b := 2^(2*k) * 5^(2*k),
  use 6,
  sorry
}

end greatest_value_of_k_l291_291682


namespace alternating_series_sum_l291_291499

theorem alternating_series_sum :
  (∑ i in finset.range 101, if even i then 2020 - 10 * i else -(2010 - 10 * i)) = -950 := by
  sorry

end alternating_series_sum_l291_291499


namespace sum_paintable_integers_l291_291484

def paintable (a b c : ℕ) : Prop :=
  ∀ n k : ℕ, 
    (n ≡ 1 [MOD a] ∨
    n ≡ 2 [MOD b] ∨
    n ≡ 4 [MOD c]) → (n ≡ k [MOD a] ∨ n ≡ k [MOD b] ∨ n ≡ k [MOD c]) → k = n

theorem sum_paintable_integers : 
  ∑ a in {1, 2, 3, 4, 5}, ∑ b in {1, 2, 3, 4, 5}, ∑ c in {1, 2, 3, 4, 5}, 
  if paintable a b c then (100*a + 10*b + c) else 0 = 1123 := 
sorry

end sum_paintable_integers_l291_291484


namespace sum_a1_a5_l291_291970

def sequence_sum (S : ℕ → ℕ) := ∀ n : ℕ, S n = n^2 + 1

theorem sum_a1_a5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h_sum : sequence_sum S)
  (h_a1 : a 1 = S 1)
  (h_a5 : a 5 = S 5 - S 4) :
  a 1 + a 5 = 11 := by
  sorry

end sum_a1_a5_l291_291970


namespace find_x_if_perpendicular_l291_291224

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x - 5)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x : ℝ) : Prop :=
  let a := vector_a x
  let b := vector_b x
  a.1 * b.1 + a.2 * b.2 = 0

-- Prove that x = 3 if a and b are perpendicular
theorem find_x_if_perpendicular :
  ∃ x : ℝ, perpendicular x ∧ x = 3 :=
by
  sorry

end find_x_if_perpendicular_l291_291224


namespace gcf_60_75_l291_291397

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l291_291397


namespace area_of_shaded_region_is_40pi_l291_291874

-- Define the radii of the smaller circles
def r1 := 4
def r2 := 5

-- Distance between the centers of the two smaller circles (since they are externally tangent)
def dist_centers := r1 + r2

-- Radius of the large circle
def R := (dist_centers + r1 + r2) / 2

-- Define the areas of the circles
def area_large_circle := Real.pi * R^2
def area_small_circle1 := Real.pi * r1^2
def area_small_circle2 := Real.pi * r2^2

-- Define the area of the shaded region
def area_shaded_region := area_large_circle - area_small_circle1 - area_small_circle2

-- Prove that the area of the shaded region is 40π
theorem area_of_shaded_region_is_40pi : area_shaded_region = 40 * Real.pi :=
by
  -- Skip the proof here with sorry
  sorry

end area_of_shaded_region_is_40pi_l291_291874


namespace smallest_positive_debt_resolved_minimal_positive_debt_l291_291378

theorem smallest_positive_debt_resolved (p c : ℤ) :
  ∃ p c : ℤ, 1 = 10 * p + 7 * c :=
begin
  sorry
end

noncomputable def smallest_positive_debt := 25

theorem minimal_positive_debt :
  ∃ (p c : ℤ), smallest_positive_debt = 250 * p + 175 * c :=
begin
  sorry
end

end smallest_positive_debt_resolved_minimal_positive_debt_l291_291378


namespace count_two_digit_numbers_with_digit_8_l291_291988

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_digit_8 (n : ℕ) : Prop :=
  n / 10 = 8 ∨ n % 10 = 8

theorem count_two_digit_numbers_with_digit_8 : 
  (∃ S : Finset ℕ, (∀ n ∈ S, is_two_digit n ∧ has_digit_8 n) ∧ S.card = 18) :=
sorry

end count_two_digit_numbers_with_digit_8_l291_291988


namespace derivative_of_f_l291_291818

noncomputable def f (x : ℝ) := (x + 4) * (x - 7)

theorem derivative_of_f (x : ℝ) : deriv (λ x, (x + 4) * (x - 7)) x = 2 * x - 3 := by
  have h1 : (x + 4) * (x - 7) = x^2 - 3 * x - 28 := sorry
  rw [← h1]
  apply deriv_comp
  apply deriv_pow
  apply deriv_neg_const

#check derivative_of_f

end derivative_of_f_l291_291818


namespace cost_of_items_l291_291341

variable (p q r : ℝ)

theorem cost_of_items :
  8 * p + 2 * q + r = 4.60 → 
  2 * p + 5 * q + r = 3.90 → 
  p + q + 3 * r = 2.75 → 
  4 * p + 3 * q + 2 * r = 7.4135 :=
by
  intros h1 h2 h3
  sorry

end cost_of_items_l291_291341


namespace relationship_between_number_and_square_l291_291060

theorem relationship_between_number_and_square (n : ℕ) (h : n = 9) :
  (n + n^2) / 2 = 5 * n := by
    sorry

end relationship_between_number_and_square_l291_291060


namespace volume_ratio_eq_three_l291_291175

-- Definition for the volumes and the given condition
variables {A B C D A₁ B₁ C₁ D₁: Type} [VolumeSpace A B C D] [VolumeSpace A₁ B₁ C₁ D₁]

-- Condition: Points are on plane of the faces
axiom A_on_plane : A₁ ∈ plane A B C
axiom B_on_plane : B₁ ∈ plane A B D
axiom C_on_plane : C₁ ∈ plane A C D
axiom D_on_plane : D₁ ∈ plane B C D

-- Condition: Lines are parallel
axiom lines_parallel : parallel (line A A₁) (line B B₁) ∧ 
                        parallel (line A A₁) (line C C₁) ∧ 
                        parallel (line A A₁) (line D D₁)

-- Definition for the ratio of the volumes
noncomputable def volume_ratio : ℝ := volume (tetrahedron A₁ B₁ C₁ D₁) / volume (tetrahedron A B C D)

-- The main proof goal
theorem volume_ratio_eq_three : volume_ratio = 3 :=
by
  sorry

end volume_ratio_eq_three_l291_291175


namespace carpet_shaded_area_l291_291071

theorem carpet_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 2) :
  let S := 12 / 4 in
  let T := S / 2 in
  let total_area := 1 * S^2 + 8 * T^2 in
  total_area = 27 :=
by
  sorry

end carpet_shaded_area_l291_291071


namespace gcd_60_75_l291_291385

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l291_291385


namespace circle_tangent_to_circle_and_line_l291_291410

theorem circle_tangent_to_circle_and_line (a b : ℝ) :
  (|-1 - a| = 1) ∧ (√(a^2 + b^2) = 2) →
  (∃ c d r, (r = 1) ∧ (((c = 0) ∧ (d = 2) ∧ (x^2 + (y - d)^2 = r))
  ∨ ((c = 0) ∧ (d = -2) ∧ (x^2 + (y + d)^2 = r)) 
  ∨ ((c = -2) ∧ (d = 0) ∧ ((x + c)^2 + y^2 = r)))) :=
sorry

end circle_tangent_to_circle_and_line_l291_291410


namespace gcd_60_75_l291_291386

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l291_291386


namespace managers_participated_l291_291090

theorem managers_participated (teams : ℕ) (people_per_team : ℕ) (employees : ℕ)
    (total_people : teams * people_per_team = 30)
    (employees_count : employees = 7) : 
    ∃ managers : ℕ, managers = 23 :=
by
  have total : ℕ := 6 * 5
  have empl : ℕ := 7
  have mgrs : ℕ := total - empl
  have h1 : total = 30 := by sorry
  have h2 : empl = 7 := by sorry
  show ∃ managers, managers = mgrs
  existsi mgrs
  sorry

end managers_participated_l291_291090


namespace additional_coins_needed_l291_291857

theorem additional_coins_needed (friends : Nat) (current_coins : Nat) : 
  friends = 15 → current_coins = 100 → 
  let total_coins_needed := (friends * (friends + 1)) / 2 
  in total_coins_needed - current_coins = 20 :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end additional_coins_needed_l291_291857


namespace sum_of_lowest_scores_l291_291732

theorem sum_of_lowest_scores 
    (six_scores : List ℕ)
    (length_six : six_scores.length = 6)
    (mean_85 : (six_scores.sum / 6) = 85)
    (median_88 : six_scores.sorted.nth 2 = some 88 ∧ six_scores.sorted.nth 3 = some 88)
    (mode_90 : ∃ n : ℕ, six_scores.count 90 = n ∧ n ≥ 2) : 
    (∃ a b : ℕ, a + b = 154 ∧ List.take 2 (six_scores.sorted) = [a, b]) :=
sorry

end sum_of_lowest_scores_l291_291732


namespace compare_M_N_l291_291552

theorem compare_M_N (a : ℝ) : 
  let M := 2 * a * (a - 2) + 7
  let N := (a - 2) * (a - 3)
  M > N :=
by
  sorry

end compare_M_N_l291_291552


namespace average_percentage_increase_is_correct_l291_291781

def initial_prices : List ℝ := [300, 450, 600]
def price_increases : List ℝ := [0.10, 0.15, 0.20]

noncomputable def total_original_price : ℝ :=
  initial_prices.sum

noncomputable def total_new_price : ℝ :=
  (List.zipWith (λ p i => p * (1 + i)) initial_prices price_increases).sum

noncomputable def total_price_increase : ℝ :=
  total_new_price - total_original_price

noncomputable def average_percentage_increase : ℝ :=
  (total_price_increase / total_original_price) * 100

theorem average_percentage_increase_is_correct :
  average_percentage_increase = 16.11 := by
  sorry

end average_percentage_increase_is_correct_l291_291781


namespace find_negative_number_l291_291022

theorem find_negative_number : ∃ x ∈ ({} : set ℝ), x < 0 ∧ (x = -5) :=
by
  use -5
  split
  { 
    trivial 
  }
  {
    simp
  }

end find_negative_number_l291_291022


namespace num_even_digits_in_base7_of_528_is_zero_l291_291143

def is_digit_even_base7 (d : ℕ) : Prop :=
  (d = 0) ∨ (d = 2) ∨ (d = 4) ∨ (d = 6)

def base7_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0] else (List.unfoldr (λ x, if x = 0 then none else some (x % 7, x / 7)) n).reverse

def number_of_even_digits_base7 (n : ℕ) : ℕ :=
  List.countp is_digit_even_base7 (base7_representation n)

theorem num_even_digits_in_base7_of_528_is_zero : number_of_even_digits_base7 528 = 0 :=
by
  sorry

end num_even_digits_in_base7_of_528_is_zero_l291_291143


namespace probability_ball_two_at_least_twice_given_sum_is_seven_l291_291546

noncomputable def draws : List ℕ := [1, 2, 3, 4]

def sum_eq_seven (l : List ℕ) : Prop := l.sum = 7

def ball_two_at_least_twice (l : List ℕ) : Prop := (l.count 2) ≥ 2

theorem probability_ball_two_at_least_twice_given_sum_is_seven :
  (ProbSum : ℚ) = ((count_filter (λ l : List ℕ, ball_two_at_least_twice l ∧ sum_eq_seven l) 
  (product_tripples draws).card) / (count_filter sum_eq_seven (product_tripples draws)).card) :=
begin
  sorry
end

end probability_ball_two_at_least_twice_given_sum_is_seven_l291_291546


namespace num_even_digits_in_base7_of_528_is_zero_l291_291146

def is_digit_even_base7 (d : ℕ) : Prop :=
  (d = 0) ∨ (d = 2) ∨ (d = 4) ∨ (d = 6)

def base7_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0] else (List.unfoldr (λ x, if x = 0 then none else some (x % 7, x / 7)) n).reverse

def number_of_even_digits_base7 (n : ℕ) : ℕ :=
  List.countp is_digit_even_base7 (base7_representation n)

theorem num_even_digits_in_base7_of_528_is_zero : number_of_even_digits_base7 528 = 0 :=
by
  sorry

end num_even_digits_in_base7_of_528_is_zero_l291_291146


namespace range_of_a_l291_291962

noncomputable theory

variable {a : ℝ} {n : ℕ}

def sequence (n : ℕ) (a : ℝ) : ℝ := n^2 - 2 * a * n

def a4 (a : ℝ) : ℝ := sequence 4 a

theorem range_of_a :
  (∀ n : ℕ, n ≠ 4 → sequence n a > a4 a) ↔ (7 / 2 < a ∧ a < 9 / 2) :=
by
  sorry

end range_of_a_l291_291962


namespace john_needs_packs_l291_291661

-- Definitions based on conditions
def utensils_per_pack : Nat := 30
def utensils_types : Nat := 3
def spoons_per_pack : Nat := utensils_per_pack / utensils_types
def spoons_needed : Nat := 50

-- Statement to prove
theorem john_needs_packs : (50 / spoons_per_pack) = 5 :=
by
  -- To complete the proof
  sorry

end john_needs_packs_l291_291661


namespace angle_between_lines_l291_291595

theorem angle_between_lines (l₁ l₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ √3 * x - y + 2 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 3 * x + √3 * y - 5 = 0) :
  angle_between_lines l₁ l₂ = π / 3 := by
sorry

end angle_between_lines_l291_291595


namespace additional_friends_l291_291838

theorem additional_friends (x: ℕ): 
  (∀ f: ℕ, (f = 25) → (100 / f = 4) → (100 / (f + x) = 3) → x = 8) := 
by 
  intros f hf h4 h3
  unfold f at hf 
  rw hf
  have h : 100 = 4 * 25 := by norm_num
  rw mul_comm at h
  have h' : x = 8 := by sorry
  exact h'

end additional_friends_l291_291838


namespace multiple_of_a_power_l291_291729

theorem multiple_of_a_power (a n m : ℕ) (h : a^n ∣ m) : a^(n+1) ∣ (a+1)^m - 1 := 
sorry

end multiple_of_a_power_l291_291729


namespace proof_problem_l291_291191

-- Define the environment and essential entities
structure Point (α : Type) :=
(x : α) (y : α)

-- Given M lies on x^2 + y^2 = 4
def M_on_circle (M : Point ℝ) : Prop :=
  M.x^2 + M.y^2 = 4

-- N is the foot of the perpendicular from M to the x-axis
def N_on_x_axis (M N : Point ℝ) : Prop :=
  N.x = M.x ∧ N.y = 0

-- H is the midpoint of MN
def midpoint (M N H : Point ℝ) : Prop :=
  H.x = (M.x + N.x) / 2 ∧ H.y = (M.y + N.y) / 2

-- Define the locus of H
def locus_H (H : Point ℝ) : Prop :=
  (H.x^2) / 4 + (H.y^2) = 1

-- λ and μ conditions involving vectors
def vector_condition (P A B : Point ℝ) (λ : ℝ) : Prop :=
  ∀ qv, (P.x - A.x) = λ * (B.x - P.x) ∧ (P.y - A.y) = λ * (B.y - P.y)

def constant_fraction (λ μ : ℝ) : Prop :=
  1 / λ + 1 / μ = 8 / 3

-- Prove the final goal given all the conditions.
theorem proof_problem 
  (M N H P Q A B : Point ℝ) (λ μ : ℝ)
  (h1 : M_on_circle M)
  (h2 : N_on_x_axis M N)
  (h3 : midpoint M N H)
  (h4 : locus_H P)
  (h5 : locus_H Q)
  (h6 : A = Point.mk 0 0.5)
  (h7 : vector_condition P A B λ)
  (h8 : vector_condition Q A B μ) :
  locus_H H ∧ constant_fraction λ μ :=
by {
  sorry,
}

end proof_problem_l291_291191


namespace correctWeightDesign_l291_291080

-- Define the weights and their corresponding conditions
def weightDesigns : List (Nat × Nat × Nat × Nat) :=
  [(3, 3, 2, 2), (5, 2, 1, 2), (1, 2, 2, 5), (2, 3, 3, 2)]

-- Define a predicate that checks if a weight design meets the conditions.
def meetsConditions (w : Nat × Nat × Nat × Nat) : Prop :=
  w.1 > w.2 ∧ w.1 > w.4 ∧ w.2 = w.4 ∧ w.3 < w.2

-- Prove that the correct weight design is the one that meets all the conditions.
theorem correctWeightDesign : (5, 2, 1, 2) ∈ weightDesigns ∧ meetsConditions (5, 2, 1, 2) :=
by
  sorry

end correctWeightDesign_l291_291080


namespace negative_number_among_options_l291_291016

theorem negative_number_among_options :
  let A := |(-2 : ℤ)|
      B := real.sqrt 3
      C := (0 : ℤ)
      D := (-5 : ℤ)
  in D = -5 := 
by 
  sorry

end negative_number_among_options_l291_291016


namespace collinear_DIJ_l291_291284

theorem collinear_DIJ (A B C D E F I K L J : Point)
  (h1 : ¬ is_isosceles_triangle ABC)
  (h2 : incircle I ABC D E F)
  (h3 : line_through E ⊥ BI ∩ ⊙I = K)
  (h4 : line_through F ⊥ CI ∩ ⊙I = L)
  (h5 : midpoint J K L) :
  collinear D I J :=
sorry

end collinear_DIJ_l291_291284


namespace sin_sum_to_product_l291_291514

theorem sin_sum_to_product (x : ℝ) : sin (3 * x) + sin (5 * x) = 2 * sin (4 * x) * cos x :=
by sorry

end sin_sum_to_product_l291_291514


namespace combinatorial_nonexistence_l291_291609

open Finset

theorem combinatorial_nonexistence (n : ℕ) : 
  ¬ (nat.choose n 3 = nat.choose (n - 1) 3 + nat.choose (n - 1) 4) :=
sorry

end combinatorial_nonexistence_l291_291609


namespace shortest_multicolored_cycle_l291_291707

theorem shortest_multicolored_cycle (G : Graph) (cycle : List (Vertex × Vertex)) :
  (∀ (a_i b_i : Vertex), (a_i, b_i) ∈ cycle) →
  (length cycle = 2 * s) →
  (∀ a_i, a_i ∈ cycle → ∃ h : Horizontal, a_i = to_vertex h) →
  (∀ b_j, b_j ∈ cycle → ∃ v : Vertical, b_j = to_vertex v) →
  (∃ (s > 2 → False), shortest_multicolored_cycle_in_G = 4) := 
by
  sorry

end shortest_multicolored_cycle_l291_291707


namespace part_one_part_two_l291_291213

-- Definitions of the functions
def f (a : ℝ) (x : ℝ) : ℝ := (a-1) * x^a
def g (x : ℝ) : ℝ := |Real.log x|

-- Conditions
variable (a : ℝ)
variable (x1 x2 : ℝ)
variable h_distinct_roots : x1 ∈ (1:ℝ, 2) ∧ x2 ∈ (2, 3) ∧ f a 1 + g (x1-1) = 0 ∧ f a 1 + g (x2-1) = 0

-- Goals
theorem part_one (h_power_function : ∀ x, f a x = (a-1) * x^a = x^1) : a = 2 := 
sorry

theorem part_two (h_interval : x1 ∈ (1:ℝ, 3) ∧ x2 ∈ (1:ℝ, 3)) :
  2 - Real.log 2 < a + (1/x1) + (1/x2) ∧ a + (1/x1) + (1/x2) < 2 := 
sorry

end part_one_part_two_l291_291213


namespace calculate_survival_months_l291_291249

noncomputable def survival_months (P : ℝ) (initial_population : ℝ) (expected_survivors : ℝ) : ℝ :=
  let survival_rate := 1 - P
  let n := (Real.log expected_survivors - Real.log initial_population) / Real.log survival_rate
  n

theorem calculate_survival_months : survival_months (1 / 10) 700 510.3 ≈ 3 :=
by
  unfold survival_months
  have h : allocation := (Real.log 510.3 - Real.log 700) / Real.log (9 / 10)
  exact Real.log (510.3 / 700) / Real.log (9 / 10)
  sorry

end calculate_survival_months_l291_291249


namespace find_m_and_parity_l291_291956

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^m - 4 / x

theorem find_m_and_parity :
  (∃ m : ℝ, f 4 m = 3) ∧ (∀ m : ℝ, f (-x) m = - f x m) :=
by
  let f := λ x m, x^m - 4 / x
  sorry

end find_m_and_parity_l291_291956


namespace triangle_perimeter_is_five_l291_291194

noncomputable def triangle_perimeter (a b c : ℝ) : ℝ :=
a + b + c

theorem triangle_perimeter_is_five :
  ∃ x, (1 + 2 > x ∧ 1 + x > 2 ∧ 2 + x > 1) ∧ x^2 - 3 * x + 2 = 0 ∧ triangle_perimeter 1 2 x = 5 :=
by {
  use 2,
  split,
  -- x = 2 satisfies the triangle inequality
  split; linarith,
  split; linarith,
  split; linarith,
  -- x = 2 is a root of the quadratic equation
  norm_num,
  simp,
  -- The perimeter of the triangle is 5
  norm_num,
  simp [triangle_perimeter]
}

end triangle_perimeter_is_five_l291_291194


namespace statement_A_statement_B_statement_C_statement_D_statement_E_l291_291878

def T := { x : Int // x ≠ 0 }

def diamond (a b : Int) : Int := 3 * a * b + a + b

theorem statement_A : ∀ (a b : T), diamond a b = diamond b a :=
by sorry

theorem statement_B : ∃ (a b c : T), diamond a (diamond b c) ≠ diamond (diamond a b) c :=
by sorry

theorem statement_C : ∀ (a : T), diamond a 1 ≠ a ∨ diamond 1 a ≠ a :=
by sorry

theorem statement_D : ∃ (a : T), ∀ (b : T), diamond a b ≠ 1 ∧ diamond b a ≠ 1 :=
by sorry

theorem statement_E : ∀ (a : T), diamond a (-1 / (3 * a + 1)) ≠ 1 ∧ diamond (-1 / (3 * a + 1)) a ≠ 1 :=
by sorry

end statement_A_statement_B_statement_C_statement_D_statement_E_l291_291878


namespace sum_of_solutions_of_equation_l291_291327

theorem sum_of_solutions_of_equation :
  (∑ (x : ℝ) in (finset.filter (λ x, (15/x*( (35 - 8 * x^3)^(1/3) : ℝ)) = 2 * x + ( (35 - 8 * x^3)^(1/3) : ℝ)), (finset.range 100)), x) = 2.5 := 
  sorry

end sum_of_solutions_of_equation_l291_291327


namespace find_height_of_larger_cuboid_l291_291599

-- Define the larger cuboid dimensions
def Length_large : ℝ := 18
def Width_large : ℝ := 15
def Volume_large (Height_large : ℝ) : ℝ := Length_large * Width_large * Height_large

-- Define the smaller cuboid dimensions
def Length_small : ℝ := 5
def Width_small : ℝ := 6
def Height_small : ℝ := 3
def Volume_small : ℝ := Length_small * Width_small * Height_small

-- Define the total volume of 6 smaller cuboids
def Total_volume_small : ℝ := 6 * Volume_small

-- State the problem and the proof goal
theorem find_height_of_larger_cuboid : 
  ∃ H : ℝ, Volume_large H = Total_volume_small :=
by
  use 2
  sorry

end find_height_of_larger_cuboid_l291_291599


namespace total_distance_in_12_hours_is_672_l291_291766

def initial_distance : ℕ := 45
def hourly_increase : ℕ := 2
def total_hours : ℕ := 12

def distance_at_hour (n : ℕ) : ℕ :=
  initial_distance + (n - 1) * hourly_increase

def total_distance_traveled (hours : ℕ) : ℕ :=
  (List.range hours).sum (λ n => distance_at_hour (n + 1))

theorem total_distance_in_12_hours_is_672 :
  total_distance_traveled total_hours = 672 := sorry

end total_distance_in_12_hours_is_672_l291_291766


namespace gcd_of_60_and_75_l291_291392

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l291_291392


namespace proof_monotonically_increasing_and_range_of_a_l291_291958

-- Definitions for the function and conditions
def f (a x : ℝ) : ℝ := (2 * a + 1) / a - 1 / (a^2 * x)

theorem proof_monotonically_increasing_and_range_of_a (a m n : ℝ)
  (h_a_gt_0 : a > 0)
  (h_mn_gt_0 : m * n > 0) :
  (∀ x1 x2 ∈ Icc m n, x1 < x2 → f a x1 < f a x2) ∧ (0 < m ∧ m < n ∧ ∀ x ∈ Icc m n, f a x ∈ Icc m n → a > 1 / 2) := 
by
  sorry

end proof_monotonically_increasing_and_range_of_a_l291_291958


namespace product_prs_l291_291998

open Real

theorem product_prs (p r s : ℕ) 
  (h1 : 4 ^ p + 64 = 272) 
  (h2 : 3 ^ r = 81)
  (h3 : 6 ^ s = 478) : 
  p * r * s = 64 :=
by
  sorry

end product_prs_l291_291998


namespace find_principal_l291_291076

-- Define the conditions
variables (P R : ℝ) -- Define P and R as real numbers
variable (h : (P * 50) / 100 = 300) -- Introduce the equation obtained from the conditions

-- State the theorem
theorem find_principal (P R : ℝ) (h : (P * 50) / 100 = 300) : P = 600 :=
sorry

end find_principal_l291_291076


namespace find_angle_C_find_area_of_triangle_l291_291245

theorem find_angle_C (A B C : ℝ) (a b c : ℝ)
  (h_side_a : a = c * Real.sin(A))
  (h_side_b : b = c * Real.sin(B))
  (h_side_c : c * Real.cos(B) + (b - 2 * a) * Real.cos(C) = 0) :
  C = Real.pi / 3 :=
by
  sorry

theorem find_area_of_triangle (A B C : ℝ) (a b c : ℝ)
  (h_angle_C : C = Real.pi / 3)
  (h_side_c : c = 2)
  (h_side_ab_relation : a + b = a * b) :
  1 / 2 * a * b * Real.sin(C) = Real.sqrt(3) :=
by
  sorry

end find_angle_C_find_area_of_triangle_l291_291245


namespace red_balls_count_l291_291355

-- Define the conditions
def white_red_ratio : ℕ × ℕ := (5, 3)
def num_white_balls : ℕ := 15

-- Define the theorem to prove
theorem red_balls_count (r : ℕ) : r = num_white_balls / (white_red_ratio.1) * (white_red_ratio.2) :=
by sorry

end red_balls_count_l291_291355


namespace grid_mark_symmetry_l291_291633

def cells : Finset (Fin 4 × Fin 4) :=
  (Finset.univ : Finset (Fin 4)).product (Finset.univ : Finset (Fin 4))

def distinct_way_count_to_mark (n k : ℕ) : ℕ :=
  if k = 2 && n = 4 then 32 else 0

theorem grid_mark_symmetry :
  distinct_way_count_to_mark 4 2 = 32 := 
by
  sorry

end grid_mark_symmetry_l291_291633


namespace angle_ABC_degree_measure_l291_291760

theorem angle_ABC_degree_measure (O A B C : Type)
  (hO : ∀ P Q, P ≠ Q → O = center_of_circumscribed_circle P Q)
  (hBOC : angle_degrees B O C = 130)
  (hAOB : angle_degrees A O B = 135) :
  angle_degrees A B C = 47.5 :=
by
  sorry

end angle_ABC_degree_measure_l291_291760


namespace range_of_m_l291_291954

theorem range_of_m 
  (P Q : ℝ × ℝ) (A : ℝ × ℝ)
  (AP AQ : ℝ × ℝ)
  (curve_condition : P.1 = -√(4 - P.2^2))
  (line_condition : Q.1 = 6)
  (midpoint_condition : AP + AQ = (0,0))
  (A_def : A = (m, 0))
  : 2 ≤ m ∧ m ≤ 3 := sorry

end range_of_m_l291_291954


namespace combined_motion_properties_l291_291976

noncomputable def y (x : ℝ) := Real.sin x + (Real.sin x) ^ 2

theorem combined_motion_properties :
  (∀ x: ℝ, - (1/4: ℝ) ≤ y x ∧ y x ≤ 2) ∧ 
  (∃ x: ℝ, y x = 2) ∧
  (∃ x: ℝ, y x = -(1/4: ℝ)) :=
by
  -- The complete proofs for these statements are omitted.
  -- This theorem specifies the required properties of the function y.
  sorry

end combined_motion_properties_l291_291976


namespace smallest_non_prime_non_square_no_prime_factor_lt_60_l291_291405

-- Definitions based on the problem conditions
def is_prime (n : ℕ) : Prop := ∀ m, 1 < m ∧ m < n → n % m ≠ 0

def is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def has_prime_factor_less_than (n : ℕ) (k : ℕ) : Prop :=
  ∃ p, is_prime p ∧ p < k ∧ p ∣ n

-- Lean 4 statement of the proof problem
theorem smallest_non_prime_non_square_no_prime_factor_lt_60 :
  (∀ n, 0 < n ∧ ¬is_prime n ∧ ¬is_square n ∧ ¬has_prime_factor_less_than n 60 → 4087 ≤ n) ∧
  0 < 4087 ∧ ¬is_prime 4087 ∧ ¬is_square 4087 ∧ ¬has_prime_factor_less_than 4087 60 :=
begin
  sorry
end

end smallest_non_prime_non_square_no_prime_factor_lt_60_l291_291405


namespace part1_part2_part3_l291_291725

-- Part 1
theorem part1 :
  3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 →
  Real.intPart (Real.sqrt 10) = 3 ∧ Real.decPart (Real.sqrt 10) = Real.sqrt 10 - 3 :=
by
  sorry

-- Part 2
theorem part2 :
  let a := Real.sqrt 6 - 2
  let b := 3
  a + b - Real.sqrt 6 = 1 :=
by
  sorry

-- Part 3
theorem part3 :
  let x := 13
  let y := Real.sqrt 3 - 1
  (12 + Real.sqrt 3 = x + y ∧ 0 < y ∧ y < 1) →
  -(x - y) = Real.sqrt 3 - 14 :=
by
  sorry

end part1_part2_part3_l291_291725


namespace angle_measure_l291_291008

-- Define the angle in degrees
def angle (x : ℝ) : Prop :=
  180 - x = 3 * (90 - x)

-- Desired proof statement
theorem angle_measure :
  ∀ (x : ℝ), angle x → x = 45 := by
  intros x h
  sorry

end angle_measure_l291_291008


namespace geometric_sum_result_l291_291188

noncomputable theory

def geometric_seq_sum (a n : ℕ) (q : ℝ) : ℝ :=
a * q * (1 - q ^ n) / (1 - q)

theorem geometric_sum_result (a_n : ℕ → ℝ) (a2 a5 : ℝ) :
    (∃ q : ℝ, a2 = 2 ∧ a5 = 1 / 4 ∧ 
        ∀ n : ℕ, a_n n = 4 * (q ^ (n - 1))) →
    (∑ k in range n, a_n k * a_n (k + 1) = 32 / 3 * (1 - (1 / 4) ^ n)) :=
begin
  sorry
end

end geometric_sum_result_l291_291188


namespace chord_length_from_line_l291_291753

noncomputable def length_of_chord : ℝ :=
  let circle : ℝ × ℝ → Prop := λ p, (p.1)^2 + (p.2)^2 + 2 * p.1 - 2 * p.2 - 4 = 0
  let line   : ℝ × ℝ → Prop := λ p, p.1 + p.2 + 2 = 0
  let center : ℝ × ℝ := (-1, 1)
  let r : ℝ := Real.sqrt 6
  let d : ℝ := Real.sqrt 2
  2 * Real.sqrt (r * r - d * d)

theorem chord_length_from_line 
  (circle := λ p : ℝ × ℝ, (p.1)^2 + (p.2)^2 + 2 * p.1 - 2 * p.2 - 4 = 0)
  (line := λ p : ℝ × ℝ, p.1 + p.2 + 2 = 0) :
  length_of_chord = 4 := by
  sorry

end chord_length_from_line_l291_291753


namespace hyperbola_specific_eq_l291_291564

-- Given definitions and conditions
def hyperbola_eq (a b : ℝ) : Prop := ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2) = 1
def positive_a_and_b (a b : ℝ) : Prop := a > 0 ∧ b > 0
def focus_condition (c : ℝ) : Prop := c^2 = 4
def asymptote_slope (a b : ℝ) : Prop := b / a = sqrt 3

-- Prove the specific hyperbola equation given the conditions
theorem hyperbola_specific_eq (a b : ℝ) (h1 : positive_a_and_b a b)
  (h2 : ∃ (c : ℝ), focus_condition c)
  (h3 : asymptote_slope a b) : hyperbola_eq 1 (sqrt 3) := 
sorry

end hyperbola_specific_eq_l291_291564


namespace area_of_triangle_is_168_l291_291492

-- Define the curve equation
def curve_eq (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define the x-intercepts
def x_intercepts (y : ℝ) : Prop := y = 0

-- Define the y-intercept
def y_intercept (x : ℝ) : ℝ := curve_eq 0

-- Define the base of the triangle (distance between x-intercepts)
def base : ℝ := 4 - (-3)

-- Define the height of the triangle (y-intercept value)
def height : ℝ := y_intercept 0

-- Define the area calculation for the triangle
def triangle_area : ℝ := (1 / 2) * base * height

-- The theorem to prove the area of the triangle is 168
theorem area_of_triangle_is_168 : triangle_area = 168 :=
by sorry

end area_of_triangle_is_168_l291_291492


namespace student_wins_all_competitions_l291_291121

/-- Assume a school year has 44 competitions. Each competition is won by exactly 7 students. 
For every pair of competitions, there is exactly 1 student who won both. 
Prove that there exists a student who won all 44 competitions. -/
theorem student_wins_all_competitions :
  ∃ student, ∀ competition, student_won student competition :=
begin
  /- We introduce the assumption that there are 44 competitions and exactly 7 students win each competition.
     Moreover, for any two distinct competitions, there is exactly 1 student who won both -/
  let competitions := 44,
  let students_per_competition := 7,
  let competitions_pairs := (competitions choose 2),
  let wins_per_student := 7,

  /- We need to show that there exists a student who won all the competitions.
     Let's assume there does not exist such a student and derive a contradiction. -/
  by_contradiction,
  sorry,
end

end student_wins_all_competitions_l291_291121


namespace cut_letter_E_into_square_l291_291502

noncomputable def letter_E_area (height width length : ℝ) : ℝ :=
  (height * width) + 3 * (length * width)

noncomputable def square_side_length (area : ℝ) : ℝ :=
  real.sqrt area

theorem cut_letter_E_into_square (height width length : ℝ) :
  ∃ (parts : ℕ), (parts = 5 ∨ parts = 4) ∧
  (parts = 5 → ¬flip_allowed) ∧ (parts = 4 → flip_allowed) ∧
  (height > 0 ∧ width > 0 ∧ length > 0) ∧
  let area_E := letter_E_area height width length in 
  let side_length := square_side_length area_E in
  (area_E = side_length ^ 2)
:=
begin
  sorry
end

end cut_letter_E_into_square_l291_291502


namespace proof_problem_l291_291295

noncomputable def problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y = 1) : Prop :=
  (∀ x y, 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → (4 * x ^ 2 + y ^2) = 1 / 2) ∧
  (∀ x y, 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → (2 / x + 1 / y) = 9) ∧
  (∀ x y, 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → (sqrt (2 * x) + sqrt y) ≤ sqrt 2)

theorem proof_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y = 1)
: problem x y h1 h2 h3 := sorry

end proof_problem_l291_291295


namespace volume_box_constraint_l291_291411

theorem volume_box_constraint : ∀ x : ℕ, ((2 * x + 6) * (x^3 - 8) * (x^2 + 4) < 1200) → x = 2 :=
by
  intros x h
  -- Proof is skipped
  sorry

end volume_box_constraint_l291_291411


namespace material_wasted_l291_291846

theorem material_wasted
  (rect_length rect_width : ℝ)
  (h1 : rect_length = 10)
  (h2 : rect_width = 8) :
  let circle_diameter := min rect_length rect_width,
      circle_radius := circle_diameter / 2,
      circle_area := Real.pi * (circle_radius ^ 2),
      square_side := circle_diameter / Real.sqrt 2,
      square_area := square_side ^ 2,
      rect_area := rect_length * rect_width,
      waste := (rect_area - circle_area) + (circle_area - square_area)
  in waste = 48 := 
by
  sorry

end material_wasted_l291_291846


namespace real_part_of_solution_l291_291326

theorem real_part_of_solution (a b : ℝ) (z : ℂ) (h : z = a + b * Complex.I): 
  z * (z + Complex.I) * (z + 2 * Complex.I) = 1800 * Complex.I → a = 20.75 := by
  sorry

end real_part_of_solution_l291_291326


namespace spherical_coordinates_equivalence_l291_291643

theorem spherical_coordinates_equivalence :
  ∀ (ρ θ φ : ℝ), 
        ρ = 3 → θ = (2 * Real.pi / 7) → φ = (8 * Real.pi / 5) →
        (0 < ρ) → 
        (0 ≤ (2 * Real.pi / 7) ∧ (2 * Real.pi / 7) < 2 * Real.pi) →
        (0 ≤ (8 * Real.pi / 5) ∧ (8 * Real.pi / 5) ≤ Real.pi) →
      ∃ (ρ' θ' φ' : ℝ), 
        ρ' = ρ ∧ θ' = (9 * Real.pi / 7) ∧ φ' = (2 * Real.pi / 5) :=
by
    sorry

end spherical_coordinates_equivalence_l291_291643


namespace base_number_min_sum_l291_291871

theorem base_number_min_sum (a b : ℕ) (h₁ : 5 * a + 2 = 2 * b + 5) : a + b = 9 :=
by {
  -- this proof is skipped with sorry
  sorry
}

end base_number_min_sum_l291_291871


namespace john_allowance_is_150_l291_291415

def john_weekly_allowance (A : ℝ) : Prop :=
  let spent_at_arcade := (3 / 5) * A in
  let remaining_after_arcade := A - spent_at_arcade in
  let spent_at_toy_store := (1 / 3) * remaining_after_arcade in
  let remaining_after_toy_store := remaining_after_arcade - spent_at_toy_store in
  remaining_after_toy_store = 0.40

theorem john_allowance_is_150 :
  ∃ A : ℝ, john_weekly_allowance A ∧ A = 1.50 :=
begin
  sorry
end

end john_allowance_is_150_l291_291415


namespace sum_sequence_neg11_to_11_eq_zero_l291_291494

-- Define the sequence and the sum function in terms of Lean
def sequence (n : ℤ) : ℝ :=
  (-2 : ℝ) ^ n

-- Establish the problem and the corresponding sum
theorem sum_sequence_neg11_to_11_eq_zero :
  (∑ n in Finset.range (23), sequence (n - 11)) = 0 :=
    sorry

end sum_sequence_neg11_to_11_eq_zero_l291_291494


namespace simplify_expression_l291_291795

theorem simplify_expression : 2 - (-2 : ℚ)⁻² = 7 / 4 := 
  by 
  -- Proof to be filled in
  sorry

end simplify_expression_l291_291795


namespace line_perpendicular_to_plane_l291_291190

structure Vector3 :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def line_directional_vector : Vector3 := ⟨1, 0, 2⟩
def plane_normal_vector : Vector3 := ⟨-2, 0, -4⟩

def is_parallel (v₁ v₂ : Vector3) : Prop :=
  ∃ k : ℝ, v₁.x = k * v₂.x ∧ v₁.y = k * v₂.y ∧ v₁.z = k * v₂.z

theorem line_perpendicular_to_plane :
  is_parallel plane_normal_vector line_directional_vector → 
  (plane_normal_vector.x * line_directional_vector.x +
   plane_normal_vector.y * line_directional_vector.y +
   plane_normal_vector.z * line_directional_vector.z = 0) :=
by
  sorry

end line_perpendicular_to_plane_l291_291190


namespace net_distance_from_start_total_distance_driven_fuel_consumption_l291_291337

def driving_distances : List Int := [14, -3, 7, -3, 11, -4, -3, 11, 6, -7, 9]

theorem net_distance_from_start : List.sum driving_distances = 38 := by
  sorry

theorem total_distance_driven : List.sum (List.map Int.natAbs driving_distances) = 78 := by
  sorry

theorem fuel_consumption (fuel_rate : Float) (total_distance : Nat) : total_distance = 78 → total_distance.toFloat * fuel_rate = 7.8 := by
  intros h_total_distance
  rw [h_total_distance]
  norm_num
  sorry

end net_distance_from_start_total_distance_driven_fuel_consumption_l291_291337


namespace no_arith_geo_progression_S1_S2_S3_l291_291485

noncomputable def S_1 (A B C : Point) : ℝ := sorry -- area of triangle ABC
noncomputable def S_2 (A B E : Point) : ℝ := sorry -- area of triangle ABE
noncomputable def S_3 (A B D : Point) : ℝ := sorry -- area of triangle ABD

def bisecting_plane (A B D C E : Point) : Prop := sorry -- plane bisects dihedral angle at AB

theorem no_arith_geo_progression_S1_S2_S3 (A B C D E : Point) 
(h_bisect : bisecting_plane A B D C E) :
¬ (∃ (S1 S2 S3 : ℝ), S1 = S_1 A B C ∧ S2 = S_2 A B E ∧ S3 = S_3 A B D ∧ 
  (S2 = (S1 + S3) / 2 ∨ S2^2 = S1 * S3 )) :=
sorry

end no_arith_geo_progression_S1_S2_S3_l291_291485


namespace range_of_slope_ordinate_range_l291_291161

/-- Given a point A, a line l, and a circle C with center on line l -/
structure ProblemPart1 :=
  (A : ℝ × ℝ := (0, 3))
  (l_eqn : ℝ → ℝ := λ x, 2 * x - 4)
  (radius : ℝ := 1)
  (center_eqn : ℝ → ℝ := λ x, x - 1)
  (intersects : ℝ × ℝ → ℝ → Prop := λ M k, abs ((3 * k - 2 + 3) / sqrt (1 + k^2)) ≤ radius)

-- Prove that the range of the slope k of line m is [-3/4, 0]
theorem range_of_slope (part1 : ProblemPart1) : 
  ∀ k : ℝ, part1.intersects (3, 2) k → -3/4 ≤ k ∧ k ≤ 0 := 
sorry

/-- Given a point, a line l, and a circle C with radius 1 -/
structure ProblemPart2 :=
  (A : ℝ × ℝ := (0, 3))
  (l_eqn : ℝ → ℝ := λ x, 2 * x - 4)
  (radius : ℝ := 1)
  (center : (ℝ × ℝ) → Prop := λ C, C.snd = 2 * C.fst - 4)

-- Prove that the ordinate y of the center of the circle C is in the range [-4, 4/5]
theorem ordinate_range (part2 : ProblemPart2) : 
  ∀ a : ℝ, part2.center (a, 2 * a - 4) → 0 ≤ a ∧ a ≤ 12 / 5 → 
  -4 ≤ (2 * a - 4) ∧ (2 * a - 4) ≤ 4 / 5 := 
sorry

end range_of_slope_ordinate_range_l291_291161


namespace cars_condition_l291_291362

variable (C H S T X Y : ℕ)

theorem cars_condition (h1 : C - H = 1.5 * (C - X))
                      (h2 : C - S = 1.5 * (C - Y))
                      (h3 : C - T = 0.5 * (X + Y)) :
                      T ≥ H + S :=
by
  sorry

end cars_condition_l291_291362


namespace problem_statement_l291_291280

theorem problem_statement (x y : ℝ) (M N P : ℝ) 
  (hM_def : M = 2 * x + y)
  (hN_def : N = 2 * x - y)
  (hP_def : P = x * y)
  (hM : M = 4)
  (hN : N = 2) : P = 1.5 :=
by
  sorry

end problem_statement_l291_291280


namespace find_k_l291_291267

variable (m n k : ℝ)

def line_equation (x y : ℝ) : Prop :=
  x - 5/2 * y + 1 = 0

def point_on_line (x y : ℝ) : Prop :=
  line_equation x y

def point1 := point_on_line m n
def point2 := point_on_line (m + 1/2) (n + 1/k)

theorem find_k 
  (h1 : point1)
  (h2 : point2)
  (h3 : n + 1/k = n + 1) : 
  k = 1 :=
sorry

end find_k_l291_291267


namespace minimum_value_of_f_div_f_l291_291216

noncomputable def quadratic_function_min_value (a b c : ℝ) (h : 0 < b) (h₀ : 0 < a) (h₁ : 0 < c) (h₂ : b^2 ≤ 4*a*c) : ℝ :=
  (a + b + c) / b

theorem minimum_value_of_f_div_f' (a b c : ℝ) (h : 0 < b)
  (h₀ : 0 < a) (h₁ : 0 < c) (h₂ : b^2 ≤ 4*a*c) :
  quadratic_function_min_value a b c h h₀ h₁ h₂ = 2 :=
sorry

end minimum_value_of_f_div_f_l291_291216


namespace students_in_class_l291_291776

theorem students_in_class (total_pencils : ℕ) (pencils_per_student : ℕ) (n: ℕ) 
    (h1 : total_pencils = 18) 
    (h2 : pencils_per_student = 9) 
    (h3 : total_pencils = n * pencils_per_student) : 
    n = 2 :=
by 
  sorry

end students_in_class_l291_291776


namespace bridge_length_problem_l291_291456

noncomputable def length_of_bridge (num_carriages : ℕ) (length_carriage : ℕ) (length_engine : ℕ) (speed_kmph : ℕ) (crossing_time_min : ℕ) : ℝ :=
  let total_train_length := (num_carriages + 1) * length_carriage
  let speed_mps := (speed_kmph * 1000) / 3600
  let crossing_time_secs := crossing_time_min * 60
  let total_distance := speed_mps * crossing_time_secs
  let bridge_length := total_distance - total_train_length
  bridge_length

theorem bridge_length_problem :
  length_of_bridge 24 60 60 60 5 = 3501 :=
by
  sorry

end bridge_length_problem_l291_291456


namespace Emily_beads_l291_291124

-- Define the conditions and question
theorem Emily_beads (n k : ℕ) (h1 : k = 4) (h2 : n = 5) : n * k = 20 := by
  -- Sorry: this is a placeholder for the actual proof
  sorry

end Emily_beads_l291_291124


namespace domain_of_function_l291_291527

theorem domain_of_function :
  {x : ℝ | x ≠ -3 ∧ x ≠ -2} = {x : ℝ | x ∈ (-∞, -3) ∪ (-3, -2) ∪ (-2, ∞)} :=
sorry

end domain_of_function_l291_291527


namespace adjacent_angles_l291_291311

theorem adjacent_angles (α β : ℝ) (h1 : α = β + 30) (h2 : α + β = 180) : α = 105 ∧ β = 75 := by
  sorry

end adjacent_angles_l291_291311


namespace cotangent_bam_l291_291176

-- Define the conditions for the problem
variables {A B C M : Type} [HasAngle A B C] [HasMidpoint M B C]
variable {angle : A → B → C → Type} [HasCotangent angle] 

-- State the theorem
theorem cotangent_bam (hMidpointBC: Midpoint M B C) (hTriangle: Triangle A B C) : 
  cotangent (angle B A M) = 2 * cotangent (angle A B C) + cotangent (angle B C A) :=
by
  sorry

end cotangent_bam_l291_291176


namespace tennis_balls_per_can_is_three_l291_291457

-- Definition of the number of games in each round
def games_in_round (round: Nat) : Nat :=
  match round with
  | 1 => 8
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | _ => 0

-- Definition of the average number of cans used per game
def cans_per_game : Nat := 5

-- Total number of games in the tournament
def total_games : Nat :=
  games_in_round 1 + games_in_round 2 + games_in_round 3 + games_in_round 4

-- Total number of cans used
def total_cans : Nat :=
  total_games * cans_per_game

-- Total number of tennis balls used
def total_tennis_balls : Nat := 225

-- Number of tennis balls per can
def tennis_balls_per_can : Nat :=
  total_tennis_balls / total_cans

-- Theorem to prove
theorem tennis_balls_per_can_is_three :
  tennis_balls_per_can = 3 :=
by
  -- No proof required, using sorry to skip the proof
  sorry

end tennis_balls_per_can_is_three_l291_291457


namespace sphere_radius_given_cone_l291_291433

theorem sphere_radius_given_cone 
  (r_cone : ℝ) (h_cone : ℝ) (density_factor : ℝ) (V_cone : ℝ) (V_sphere : ℝ) 
  (effective_V : ℝ) : 
  r_cone = 2 ∧ h_cone = 6 ∧ density_factor = 2 ∧ V_cone = (1/3) * π * r_cone^2 * h_cone ∧ effective_V = 2 * V_cone ∧ V_sphere = (4/3) * π * (real.cbrt (12))^3 → 
  real.cbrt (12) = real.cbrt (12) :=
by 
  assume h1 : r_cone = 2 ∧ h_cone = 6 ∧ density_factor = 2 ∧ V_cone = (1/3) * π * r_cone^2 * h_cone ∧ effective_V = 2 * V_cone ∧ V_sphere = (4/3) * π * (real.cbrt (12))^3
  exact eq.refl (real.cbrt (12))

end sphere_radius_given_cone_l291_291433


namespace sum_of_solutions_of_equation_l291_291328

theorem sum_of_solutions_of_equation :
  (∑ (x : ℝ) in (finset.filter (λ x, (15/x*( (35 - 8 * x^3)^(1/3) : ℝ)) = 2 * x + ( (35 - 8 * x^3)^(1/3) : ℝ)), (finset.range 100)), x) = 2.5 := 
  sorry

end sum_of_solutions_of_equation_l291_291328


namespace number_of_roots_802_l291_291681

noncomputable def f : ℝ → ℝ := sorry

theorem number_of_roots_802 :
  (∀ x, f (2 - x) = f (2 + x)) →
  (∀ x, f (7 - x) = f (7 + x)) →
  (∀ x ∈ set.Icc 0 7, (x = 1 ∨ x = 3) → f x = 0) →
  (set.count (set_of (λ x, f x = 0)) (set.Icc (-2005) 2005) = 802) :=
begin
  intros h1 h2 h3,
  sorry
end

end number_of_roots_802_l291_291681


namespace quarters_per_machine_l291_291896

-- conditions
def num_dimes_per_machine := 100
def value_per_dime := 0.10
def total_money := 90
def num_machines := 3
def coin_value := 0.25

-- proof problem
theorem quarters_per_machine :
  ∃ (Q : ℕ), Q = 80 ∧ (num_machines * ((Q * coin_value) + num_dimes_per_machine * value_per_dime) = total_money) :=
by
  sorry

end quarters_per_machine_l291_291896


namespace gcd_of_60_and_75_l291_291394

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l291_291394


namespace expected_value_coin_flip_l291_291087

/-- 
An unfair coin lands on heads with a probability of 3/5 and on tails with a probability of 2/5.
Flipping heads gains $5, while flipping tails results in a loss of $6. 
Prove that the expected value of a coin flip is 0.6.
-/
theorem expected_value_coin_flip : 
  let p_heads := 3 / 5
  let p_tails := 2 / 5
  let gain_heads := 5
  let loss_tails := -6
  (p_heads * gain_heads + p_tails * loss_tails) = 0.6 := 
by
  let p_heads := 3 / 5
  let p_tails := 2 / 5
  let gain_heads := 5
  let loss_tails := -6
  let expected_value := p_heads * gain_heads + p_tails * loss_tails
  have : expected_value = 0.6 := sorry
  this


end expected_value_coin_flip_l291_291087


namespace complex_modulus_l291_291199

noncomputable def z := (i - 2) / (1 + i)

theorem complex_modulus : |z| = (Real.sqrt 10) / 2 := by
  sorry

end complex_modulus_l291_291199


namespace find_all_f_l291_291679

noncomputable def functional_equation_solution (α : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f(x^2 + y + f(y)) = f(x)^2 + α y

theorem find_all_f (α : ℝ) :
  (α = 2 → ∃! f : ℝ → ℝ, functional_equation_solution α f ∧ f = id)
  ∧ (α ≠ 2 → ¬ ∃ f : ℝ → ℝ, functional_equation_solution α f) :=
by
  sorry

end find_all_f_l291_291679


namespace quadrilateral_side_comparison_l291_291316

variable {A B C D : Type} [LinearOrderedField A]

def angles_equiv (a b : A) := a = b

def angle_greater (d c : A) := d > c

theorem quadrilateral_side_comparison (A B C D : A) 
  (h1 : angles_equiv A B)
  (h2 : angle_greater D C)
  : B > A := 
sorry

end quadrilateral_side_comparison_l291_291316


namespace distance_PB_l291_291630

-- Define a point in 3D space.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the origin point B and the point P inside the classroom.
def B : Point3D := { x := 0, y := 0, z := 0 }
def P : Point3D := { x := 3, y := 4, z := 1 }

-- Define the function to calculate the Euclidean distance between two points in 3D space.
def distance (A B : Point3D) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2)

-- The task is to prove that the distance from point P to the origin B is sqrt(26).
theorem distance_PB : distance P B = Real.sqrt 26 :=
by
  sorry

end distance_PB_l291_291630


namespace constant_polynomial_only_l291_291521

namespace Polynomial

open Int

theorem constant_polynomial_only (P : ℤ[X]) (h : ∀ m n : ℕ, m > 0 → n > 0 → (m + n ∣ (polynomial.derivative^[m]) P.eval n - (polynomial.derivative^[n]) P.eval m)) : 
  ∃ c : ℤ, P = polynomial.C c :=
sorry

end Polynomial

end constant_polynomial_only_l291_291521


namespace find_a_l291_291967

theorem find_a (a : ℝ) (h1 : ∀ θ : ℝ, x = a + 4 * Real.cos θ ∧ y = 1 + 4 * Real.sin θ)
  (h2 : ∃ p : ℝ × ℝ, (3 * p.1 + 4 * p.2 - 5 = 0 ∧ (∃ θ : ℝ, p = (a + 4 * Real.cos θ, 1 + 4 * Real.sin θ))))
  (h3 : ∀ (p1 p2 : ℝ × ℝ), 
        (3 * p1.1 + 4 * p1.2 - 5 = 0 ∧ 3 * p2.1 + 4 * p2.2 - 5 = 0) ∧
        (∃ θ1 : ℝ, p1 = (a + 4 * Real.cos θ1, 1 + 4 * Real.sin θ1)) ∧
        (∃ θ2 : ℝ, p2 = (a + 4 * Real.cos θ2, 1 + 4 * Real.sin θ2)) → p1 = p2) :
  a = 7 := by
  sorry

end find_a_l291_291967


namespace additional_coins_needed_l291_291858

theorem additional_coins_needed (friends : Nat) (current_coins : Nat) : 
  friends = 15 → current_coins = 100 → 
  let total_coins_needed := (friends * (friends + 1)) / 2 
  in total_coins_needed - current_coins = 20 :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end additional_coins_needed_l291_291858


namespace find_f_2016_l291_291944

def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f(x) = f(-x)
axiom periodic_f : ∀ x, f(x + 5) = f(x - 5)
axiom f_interval : ∀ x, (0 ≤ x ∧ x ≤ 5) → f(x) = x^2 - 4 * x

theorem find_f_2016 : f(2016) = 0 := sorry

end find_f_2016_l291_291944


namespace solve_for_n_l291_291885

theorem solve_for_n : ∃ n : ℕ, 2 * n * nat.factorial n + nat.factorial n = 2520 ∧ n = 10 := by
  sorry

end solve_for_n_l291_291885


namespace area_of_quadrilateral_YPWQ_l291_291654

/-- In triangle XYZ where XY = 80 and XZ = 40, with the area of the triangle being 240.
    Let W be the midpoint of XY, and V be the midpoint of XZ.
    The angle bisector of ∠YXZ intersects WV and YZ at points P and Q, respectively.
    Prove that the area of quadrilateral YPWQ is 159. -/
theorem area_of_quadrilateral_YPWQ (XY XZ : ℝ) (area_XYZ : ℝ) (W V P Q : Point) :
  XY = 80 →
  XZ = 40 →
  area_XYZ = 240 →
  W = midpoint X Y →
  V = midpoint X Z →
  angle_bisector Y X Z P Q X Y Z →
  area_YPWQ XY XZ W V P Q = 159 :=
begin
  sorry
end

end area_of_quadrilateral_YPWQ_l291_291654


namespace sum_of_decimals_as_fraction_simplified_fraction_final_sum_as_fraction_l291_291515

theorem sum_of_decimals_as_fraction:
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008) = (45678 / 100000) := by
  sorry

theorem simplified_fraction:
  (45678 / 100000) = (22839 / 50000) := by
  sorry

theorem final_sum_as_fraction:
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008) = (22839 / 50000) := by
  have h1 := sum_of_decimals_as_fraction
  have h2 := simplified_fraction
  rw [h1, h2]
  sorry

end sum_of_decimals_as_fraction_simplified_fraction_final_sum_as_fraction_l291_291515


namespace sum_mod_11_l291_291534

theorem sum_mod_11 (h1 : 8735 % 11 = 1) (h2 : 8736 % 11 = 2) (h3 : 8737 % 11 = 3) (h4 : 8738 % 11 = 4) :
  (8735 + 8736 + 8737 + 8738) % 11 = 10 :=
by
  sorry

end sum_mod_11_l291_291534


namespace sum_of_values_for_g_4z_eq_8_l291_291680

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - x + 2

-- The problem statement
theorem sum_of_values_for_g_4z_eq_8 :
  (∑ z in {z : ℝ | g(4 * z) = 8}.to_finset, z) = 1 / 16 :=
sorry

end sum_of_values_for_g_4z_eq_8_l291_291680


namespace exists_set_F5_l291_291892

theorem exists_set_F5 : 
  ∃ (F : Finset ℤ), 
  (∀ x ∈ F, ∃ a b ∈ F, x = a + b) ∧
  (∀ (k : Finset ℤ), k ⊆ F → k.card ≤ 5 → k.sum id ≠ 0) :=
sorry

end exists_set_F5_l291_291892


namespace square_side_length_eq_area_and_perimeter_l291_291536

theorem square_side_length_eq_area_and_perimeter (a : ℝ) (h : a^2 = 4 * a) : a = 4 :=
by sorry

end square_side_length_eq_area_and_perimeter_l291_291536


namespace variance_transformation_l291_291584

variables {n : ℕ} (x : Fin n → ℝ)

theorem variance_transformation (h : variance x = 2) :
  variance (fun i => 3 * x i + 2) = 18 :=
sorry

end variance_transformation_l291_291584


namespace least_subtracted_number_l291_291403

theorem least_subtracted_number (r : ℕ) : r = 10^1000 % 97 := 
sorry

end least_subtracted_number_l291_291403


namespace sum_first_10_terms_correct_l291_291568

def a_seq : ℕ → ℝ
| 0     := -3
| 1     := 1
| (n+2) := -1/3 * a_seq (n+1)

noncomputable def sum_first_ten_terms (a_seq : ℕ → ℝ) : ℝ :=
(a_seq 0 + a_seq 1 + a_seq 2 + a_seq 3 + a_seq 4 + a_seq 5 + a_seq 6 + a_seq 7 + a_seq 8 + a_seq 9)

theorem sum_first_10_terms_correct : sum_first_ten_terms a_seq = (9/4) * (3^{-10} - 1) := by
  sorry

end sum_first_10_terms_correct_l291_291568


namespace tan_alpha_plus_pi_over_4_eq_neg_one_third_l291_291197

variable (α : Real)
variable (P : Real → Real → Prop)

def tangent_angle (x y : Real) : Real := y / x

theorem tan_alpha_plus_pi_over_4_eq_neg_one_third (hyp : P (-1) 2) (h : tangent_angle (-1) 2 = -2) : 
  tangent_angle (1 + -2) (1 - -2) = -1/3 :=
by
  sorry

end tan_alpha_plus_pi_over_4_eq_neg_one_third_l291_291197


namespace question_1_question_2_l291_291959

noncomputable def f (x α : ℝ) : ℝ := (2:ℝ)^(x + Math.cos α) - (2:ℝ)^-(x + Math.cos α)

theorem question_1 (α : ℝ) (h₀ : 0 ≤ α ∧ α ≤ Real.pi)
  (h₁ : f 1 α = (3 * Real.sqrt 2) / 4) :
  α = 2 * Real.pi / 3 :=
sorry

theorem question_2 (m θ : ℝ) (h₂ : m < 1) (h₃ : |Math.cos θ| ≠ 1)
  (h₄ : ∀ x α, f x α = (2:ℝ)^(x - 1 / 2) - (2:ℝ)^-(x - 1 / 2)) :
  f (m * |Math.cos θ|) (2 * Real.pi / 3) + f (1 - m) (2 * Real.pi / 3) > 0 :=
sorry

end question_1_question_2_l291_291959


namespace problem_solution_l291_291158

noncomputable def incorrect_propositions_count : Nat :=
  let prop1: Prop := ¬ ∀ (A B C : Point), non_collinear A B C → (∃ (O : Point), Circle O A B C) -- Prop 1
  let prop2: Prop := ∀ (O : Point) (D : Diameter O) (C : Chord O perpendicular D), Bisects O C D  -- Prop 2
  let prop3: Prop := ∀ (O : Point) (A B : Point on O), (central_angle O A B) = (central_angle O B A) → (arc_len O A B) = (arc_len O B A)  -- Prop 3
  let prop4: Prop := ∀ (O : Point) (A B : Point on O), (arc_len O A B) = (arc_len O B A) ↔ (congruent_arc O A B)  -- Prop 4

  ([prop1, prop2, prop3, prop4].count (λ p, ¬p))

theorem problem_solution : incorrect_propositions_count = 1 :=
  sorry

end problem_solution_l291_291158


namespace rate_of_return_trip_l291_291429

theorem rate_of_return_trip :
  let d := 150 in
  let t1 := d / 50 in
  let r := 75 in
  let t2 := d / r in
  let total_time := t1 + t2 in
  let avg_speed := (2 * d) / total_time in
  avg_speed = 60 -> r = 75 :=
by
  intros d t1 r t2 total_time avg_speed h
  exact sorry

end rate_of_return_trip_l291_291429


namespace circle_area_l291_291644

theorem circle_area (x y : ℝ) :
  (3 * x^2 + 3 * y^2 - 15 * x + 9 * y - 45 = 0) →
  (π * (sqrt (47) / 2)^2 = (47 / 2) * π) :=
begin
  sorry
end

end circle_area_l291_291644


namespace grasshoppers_total_calculation_l291_291660

-- Definitions from the problem conditions
def initial_grasshoppers_on_daisy := 7
def baby_grasshoppers_in_dozens := 2
def dozen := 12
def baby_grasshoppers := baby_grasshoppers_in_dozens * dozen
def groups_in_bushes := 3
def grasshoppers_per_group := 9
def grasshoppers_in_bushes := groups_in_bushes * grasshoppers_per_group
def initial_total_grasshoppers := initial_grasshoppers_on_daisy + baby_grasshoppers + grasshoppers_in_bushes
def increase_percentage := 0.20
def increased_grasshoppers := increase_percentage * initial_total_grasshoppers
def increased_grasshoppers_rounded := increased_grasshoppers.round
def final_total_grasshoppers := initial_total_grasshoppers + increased_grasshoppers_rounded

-- The theorem to prove the final total number of grasshoppers
theorem grasshoppers_total_calculation : final_total_grasshoppers = 70 :=
by
  sorry

end grasshoppers_total_calculation_l291_291660


namespace number_of_knights_feb_l291_291262

variable (inhabitants : ℕ) (feb_yes : ℕ) (day30_yes : ℕ)
variable (knights_feb : ℕ) (liars_feb : ℕ)
variable (knights_30 : ℕ) (liars_30 : ℕ)

-- Defining the conditions
def conditions : Prop :=
  inhabitants = 366 ∧
  feb_yes = 100 ∧
  day30_yes = 60 ∧
  knights_feb ≤ 29 ∧
  knights_30 ≤ 11 ∧
  feb_yes = knights_feb + (100 - knights_feb) ∧
  day30_yes = knights_30 + (60 - knights_30)

theorem number_of_knights_feb :
  conditions inhabitants feb_yes day30_yes knights_feb liars_feb knights_30 liars_30 →
  knights_feb = 29 :=
by
  sorry

end number_of_knights_feb_l291_291262


namespace lines_perpendicular_l291_291928

-- Given a square ABCD with specific points P, M, N
variables {A B C D P M N : Point}

-- Point P on segment BC and M, N defined as intersections
variables (square_ABCD : square A B C D)
          (P_on_BC : P ∈ line_segment B C)
          (M_on_CD : M ∈ line_intersection (line_through A P) (line_segment C D))
          (N_on_AB : N ∈ line_intersection (line_through D P) (line_segment A B))

-- Prove that lines NC and BM are perpendicular
theorem lines_perpendicular : is_perpendicular (line_through N C) (line_through B M) :=
sorry

end lines_perpendicular_l291_291928


namespace perpendicular_case_parallel_case_l291_291220

-- Define the vectors a and b
def a (x : ℝ) := (1 : ℝ, x)
def b (x : ℝ) := (2 * x + 3, -x)

-- 1. Prove that if a is perpendicular to b, then x = -1 ∨ x = 3.
theorem perpendicular_case (x : ℝ) : 
  (a x).fst * (b x).fst + (a x).snd * (b x).snd = 0 ↔ (x = -1 ∨ x = 3) := sorry

-- 2. Prove the magnitude of the difference between a and b if a is parallel to b is either 2 or 2√5.
theorem parallel_case (x : ℝ) : 
  (∃ k : ℝ, (1 = k * (2 * x + 3)) ∧ (x = k * (-x))) → (∥a x - b x∥ = 2 ∨ ∥a x - b x∥ = 2 * real.sqrt 5) := sorry

end perpendicular_case_parallel_case_l291_291220


namespace invertible_elements_mod_8_l291_291383

theorem invertible_elements_mod_8 :
  {x : ℤ | (x * x) % 8 = 1} = {1, 3, 5, 7} :=
by
  sorry

end invertible_elements_mod_8_l291_291383


namespace cost_of_tissues_l291_291469
-- Import the entire Mathlib library

-- Define the context and the assertion without computing the proof details
theorem cost_of_tissues
  (n_tp : ℕ) -- Number of toilet paper rolls
  (c_tp : ℝ) -- Cost per toilet paper roll
  (n_pt : ℕ) -- Number of paper towels rolls
  (c_pt : ℝ) -- Cost per paper towel roll
  (n_t : ℕ) -- Number of tissue boxes
  (T : ℝ) -- Total cost of all items
  (H_tp : n_tp = 10) -- Given: 10 rolls of toilet paper
  (H_c_tp : c_tp = 1.5) -- Given: $1.50 per roll of toilet paper
  (H_pt : n_pt = 7) -- Given: 7 rolls of paper towels
  (H_c_pt : c_pt = 2) -- Given: $2 per roll of paper towel
  (H_t : n_t = 3) -- Given: 3 boxes of tissues
  (H_T : T = 35) -- Given: total cost is $35
  : (T - (n_tp * c_tp + n_pt * c_pt)) / n_t = 2 := -- Conclusion: the cost of one box of tissues is $2
by {
  sorry -- Proof details to be supplied here
}

end cost_of_tissues_l291_291469


namespace beam_cost_l291_291047

/-- Given 30 beams with dimensions 12 fingers thick, 16 fingers wide, and 14 cubits long, costing 100 nishka (coins),
    determine the cost of 14 beams with dimensions 8 fingers thick, 12 fingers wide, and 10 cubits long. -/
theorem beam_cost :
  let 
    nishka := 100
    vol_B1 := 12 * 16 * 14
    vol_B2 := 8 * 12 * 10
    total_vol_B1 := 30 * vol_B1
    total_vol_B2 := 14 * vol_B2
    cost_per_cubic_unit := nishka / total_vol_B1
    cost_of_B2 := total_vol_B2 * cost_per_cubic_unit
  in 
    cost_of_B2 = 50 / 3 :=
by
  sorry

end beam_cost_l291_291047


namespace sec_neg_45_eq_sqrt2_l291_291126

theorem sec_neg_45_eq_sqrt2 :
  ∀ (sec cos : ℝ → ℝ),
  (∀ (θ : ℝ), sec θ = 1 / cos θ) ∧
  (∀ (θ : ℝ), cos (-θ) = cos θ) ∧
  cos (real.pi / 4) = real.sqrt 2 / 2
  → sec (-real.pi / 4) = real.sqrt 2 :=
by
  intro sec cos
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  sorry

end sec_neg_45_eq_sqrt2_l291_291126


namespace gary_has_left_amount_l291_291551

def initial_amount : ℝ := 100
def cost_pet_snake : ℝ := 55
def cost_toy_car : ℝ := 12
def cost_novel : ℝ := 7.5
def cost_pack_stickers : ℝ := 3.25
def number_packs_stickers : ℕ := 3

theorem gary_has_left_amount : initial_amount - (cost_pet_snake + cost_toy_car + cost_novel + number_packs_stickers * cost_pack_stickers) = 15.75 :=
by
  sorry

end gary_has_left_amount_l291_291551


namespace solve_for_t_l291_291734

theorem solve_for_t (t : ℝ) :
  2 * 4^t + real.sqrt (16 * 16^t) = 34 ↔ t = real.logb 4 (17 / 3) :=
by
  sorry

end solve_for_t_l291_291734


namespace shortest_multicolored_cycle_l291_291706

-- Let's define the conditions and the main goal
theorem shortest_multicolored_cycle (s : ℕ) (a : fin s → vertex) (b : fin s → vertex) (H : multicolored_cycle (a, b)) :
  s = 2 := 
by {
  -- Proof omitted, add actual proof here
  sorry
}

end shortest_multicolored_cycle_l291_291706


namespace gcd_of_60_and_75_l291_291395

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l291_291395


namespace Mina_age_is_10_l291_291695

-- Define the conditions as Lean definitions
variable (S : ℕ)

def Minho_age := 3 * S
def Mina_age := 2 * S - 2

-- State the main problem as a theorem
theorem Mina_age_is_10 (h_sum : S + Minho_age S + Mina_age S = 34) : Mina_age S = 10 :=
by
  sorry

end Mina_age_is_10_l291_291695


namespace area_ratio_10th_to_1st_l291_291480

-- Define the conditions
def r10 : ℝ := 1
def r (n : ℕ) : ℝ := n * r10
def A (radius : ℝ) : ℝ := π * radius^2

-- The target theorem to prove based on the conditions
theorem area_ratio_10th_to_1st :
  (A r10) / (A (r 10) - A (r 9)) = 1 / 19 :=
by
  rw [r10, A, A, r]
  sorry

end area_ratio_10th_to_1st_l291_291480


namespace max_good_students_proof_l291_291066

noncomputable def max_good_students (n : ℕ) (k : ℕ) : ℕ :=
  if h : n = 30 ∧ (∀ i j, i ≠ j → score i ≠ score j) ∧ (∀ i, friends i = k) then 25 else 0

theorem max_good_students_proof :
  ∃ n k x, n = 30 ∧
  (∀ i j, i ≠ j → score i ≠ score j) ∧
  (∀ i, friends i = k) ∧
  (∀ i, good_student i ↔ ∃ num_friends_lower, num_friends_lower i > k / 2) ∧
  x = max_good_students n k ∧ x = 25 :=
sorry

end max_good_students_proof_l291_291066


namespace number_of_seniors_in_statistics_l291_291697

theorem number_of_seniors_in_statistics (total_students : ℕ) (half_enrolled_in_statistics : ℕ) (percentage_seniors : ℚ) (students_in_statistics seniors_in_statistics : ℕ) 
(h1 : total_students = 120)
(h2 : half_enrolled_in_statistics = total_students / 2)
(h3 : students_in_statistics = half_enrolled_in_statistics)
(h4 : percentage_seniors = 0.90)
(h5 : seniors_in_statistics = students_in_statistics * percentage_seniors) : 
seniors_in_statistics = 54 := 
by sorry

end number_of_seniors_in_statistics_l291_291697


namespace ladder_tournament_prob_even_winner_mod_l291_291440

def num_participants : ℕ := 2016
def prob_win_game : ℚ := 1 / 2
def even_seeds := (finset.range num_participants).filter (λ x, x % 2 = 0)

theorem ladder_tournament_prob_even_winner_mod :
  let S := 1 + 2^2015 in
  let p := S % 1000 in
  p = 769 :=
by
  sorry

end ladder_tournament_prob_even_winner_mod_l291_291440


namespace complex_plane_point_l291_291592

theorem complex_plane_point (z : ℂ) (i_unit : ℂ) (h : i_unit = complex.I) 
  (h_condition : z * (1 + i_unit) = 1 - i_unit) : 
  z = - i_unit :=
by {
  sorry,
}

end complex_plane_point_l291_291592


namespace intersection_of_lines_l291_291886

theorem intersection_of_lines : 
  ∃ (x y : ℝ), (3 * y = -2 * x + 6) ∧ (4 * y = 7 * x - 8) ∧ (x = 0) ∧ (y = 2) := by
  use 0, 2
  constructor
  · exact eq.refl (3 * 2)
  constructor
  · exact eq.refl (4 * 2)
  constructor
  · exact eq.refl 0
  · exact eq.refl 2
  sorry

end intersection_of_lines_l291_291886


namespace total_time_spent_racing_l291_291792

-- Define the conditions
def speed_in_lake : ℝ := 3 -- miles per hour
def speed_in_ocean : ℝ := 2.5 -- miles per hour
def total_races : ℕ := 10
def distance_per_race : ℝ := 3 -- miles

-- Given the conditions, prove the total time spent racing is 11 hours
theorem total_time_spent_racing : 
  let races_in_lake := total_races / 2,
      races_in_ocean := total_races / 2,
      total_distance_in_lake := distance_per_race * races_in_lake,
      total_distance_in_ocean := distance_per_race * races_in_ocean,
      time_for_lake_races := total_distance_in_lake / speed_in_lake,
      time_for_ocean_races := total_distance_in_ocean / speed_in_ocean in
  time_for_lake_races + time_for_ocean_races = 11 :=
by
  sorry

end total_time_spent_racing_l291_291792


namespace natural_number_triplets_l291_291129

theorem natural_number_triplets (x y z : ℕ) : 
  3^x + 4^y = 5^z → 
  (x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = 0 ∧ y = 1 ∧ z = 1) :=
by 
  sorry

end natural_number_triplets_l291_291129


namespace point_on_coordinate_axes_l291_291238

theorem point_on_coordinate_axes {x y : ℝ} 
  (h : x * y = 0) : (x = 0 ∨ y = 0) :=
by {
  sorry
}

end point_on_coordinate_axes_l291_291238


namespace problem_statement_l291_291945

-- Condition: function f is even
def even_function {α β : Type*} [AddGroup α] [HasEquiv α] [HasAdd ℝ] (f : α → β) :=
  ∀ x, f (-x) = f x

-- Condition: function f is increasing on [0, +∞)
def increasing_on_nonneg {α β : Type*} [Preorder α] [LinearOrder β] (f : α → β) :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Values a, b, c
noncomputable def a (f : ℝ → ℝ) := f (Real.sin (5 * Real.pi / 7))
noncomputable def b (f : ℝ → ℝ) := f (Real.cos (2 * Real.pi / 7))
noncomputable def c (f : ℝ → ℝ) := f (Real.tan (2 * Real.pi / 7))

-- The main theorem to prove
theorem problem_statement (f : ℝ → ℝ) 
  (heven : even_function f) 
  (hincreasing : increasing_on_nonneg f) :
  let a := a f 
  let b := b f 
  let c := c f 
  in b < a ∧ a < c := 
sorry

end problem_statement_l291_291945


namespace find_radius_of_cone_base_l291_291951

def slant_height : ℝ := 5
def lateral_surface_area : ℝ := 15 * Real.pi

theorem find_radius_of_cone_base (A l : ℝ) (hA : A = lateral_surface_area) (hl : l = slant_height) : 
  ∃ r : ℝ, A = Real.pi * r * l ∧ r = 3 := 
by 
  sorry

end find_radius_of_cone_base_l291_291951


namespace radius_of_wire_cross_section_l291_291441

-- Define the conditions
def radius_sphere : ℝ := 12
def length_wire : ℝ := 36

-- Define the volumes according to the conditions
def volume_sphere (r : ℝ) := (4 / 3) * Real.pi * r^3
def volume_cylinder (r h : ℝ) := Real.pi * r^2 * h

-- State the theorem
theorem radius_of_wire_cross_section : 
  volume_sphere radius_sphere = volume_cylinder r length_wire → r = 8 :=
sorry

end radius_of_wire_cross_section_l291_291441


namespace range_of_f_l291_291761

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (1 - 2 * x)

theorem range_of_f : ∀ y, (∃ x, x ≤ (1 / 2) ∧ f x = y) ↔ y ∈ Set.Iic 1 := by
  sorry

end range_of_f_l291_291761


namespace smallest_integer_in_A_l291_291764

noncomputable def A := {x : ℝ | |x - 2| ≤ 5 }

theorem smallest_integer_in_A : ∃ (n : ℤ), (n : ℝ) ∈ A ∧ ∀ (m : ℤ), (m : ℝ) ∈ A → n ≤ m :=
begin
  let n := -3,
  use n,
  split,
  { -- Prove that -3 is in the set A
    show (n : ℝ) ∈ A,
    simp [A],
    have h : |(-3 : ℝ) - 2| = |(-5 : ℝ)|,
    norm_num,
    linarith
  },
  { -- Prove that -3 is the smallest integer in the set A
    intros m hm,
    simp [A] at hm,
    have := abs_le.mp hm,
    cases this with h1 h2,
    exact_mod_cast h1,
  }
end

end smallest_integer_in_A_l291_291764


namespace explicit_formula_for_a_l291_291067

noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := (1 + 4 * a n + real.sqrt (1 + 24 * a n)) / 16

theorem explicit_formula_for_a (n : ℕ) (h : n > 0) :
  a n = ( (3 + 2^(2 - n))^2 - 1 ) / 24 := 
sorry

end explicit_formula_for_a_l291_291067


namespace area_enclosed_by_curves_l291_291103

theorem area_enclosed_by_curves (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, (x + a * y)^2 = 16 * a^2) ∧ (∀ x y : ℝ, (a * x - y)^2 = 4 * a^2) →
  ∃ A : ℝ, A = 32 * a^2 / (1 + a^2) :=
by
  sorry

end area_enclosed_by_curves_l291_291103


namespace old_workers_in_sample_l291_291458

theorem old_workers_in_sample (
  total_workers : ℕ := 430 
  (young_workers : ℕ := 160)
  (sample_young_workers : ℕ := 32)
  (middle_aged_workers : ℕ)
  (old_workers : ℕ)
  (middle_aged_twice_old : middle_aged_workers = 2 * old_workers)
  (total_eq : old_workers + middle_aged_workers + young_workers = total_workers)
) : old_workers = 90 → (sample_young_workers / young_workers * old_workers = 18) :=
by
  sorry

end old_workers_in_sample_l291_291458


namespace proof_problem_l291_291235

-- Definitions of the conditions
def cond1 (r : ℕ) : Prop := 2^r = 16
def cond2 (s : ℕ) : Prop := 5^s = 25

-- Statement of the problem
theorem proof_problem (r s : ℕ) (h₁ : cond1 r) (h₂ : cond2 s) : r + s = 6 :=
by
  sorry

end proof_problem_l291_291235


namespace crow_eats_nuts_l291_291824

theorem crow_eats_nuts (time_fifth_nuts : ℕ) (time_quarter_nuts : ℕ) (h : time_fifth_nuts = 8) :
  time_quarter_nuts = 10 :=
sorry

end crow_eats_nuts_l291_291824


namespace part_I_part_II_part_III_l291_291918

noncomputable def f (x : ℝ) (a : ℝ): ℝ := Real.exp x - a * x ^ 2

theorem part_I (a b : ℝ) (h1 : ∀ x, f x a = Real.exp x - a * x ^ 2)
  (h2 : (∀ x, f 1 a = Real.exp 1 - a))
  (h3 : (∀ x, f'.eval 1 = Real.exp 1 - 2*a))
  (h4 : f'.eval 1 = b)
  (h5 : f 1 a = b + 1) :
  a = 1 ∧ b = Real.exp 1 - 2 :=
sorry

theorem part_II (a : ℝ) (h1 : a = 1) :
  ∃ M, (∀ x ∈ Icc 0 1, f x a ≤ M) ∧ (∀ y, y ∈ {f 0 a, f 1 a} → y = f 1 a) ∧ M = Real.exp 1 - 1 :=
sorry

theorem part_III (a b : ℝ) (h1 : a = 1) (h2 : b = Real.exp 1 - 2):
  ∀ x, ∃ x1 x2, (f x a = b * x + 1) ∧ (x1 ≠ x2) :=
sorry

end part_I_part_II_part_III_l291_291918


namespace sum_of_reciprocals_of_roots_l291_291684

theorem sum_of_reciprocals_of_roots (a b : ℝ) (h₁ : 7 * a^2 + 2 * a + 6 = 0)
  (h₂ : 7 * b^2 + 2 * b + 6 = 0) : (1/a + 1/b) = -1/3 :=
by
  -- By Vieta's formulas, we know:
  have h_sum : a + b = -2 / 7 := sorry,
  have h_prod : a * b = 6 / 7 := sorry,
  -- Use these to evaluate 1/a + 1/b
  calc
    1/a + 1/b = (a + b) / (a * b) : by sorry
          ... = (-2 / 7) / (6 / 7) : by sorry
          ... = -1 / 3             : by sorry

end sum_of_reciprocals_of_roots_l291_291684


namespace polar_coordinate_equation_of_line_l_rectangular_coordinate_equation_of_curve_C_minimum_value_of_OM_over_ON_l291_291651

-- Definitions based on conditions
def line_l_parametric (t : ℝ) : ℝ × ℝ := (6 + sqrt 3 * t, -t)
def curve_C_polar (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2 ∧ ρ = 2 * cos α

-- Statements of the questions with answers
theorem polar_coordinate_equation_of_line_l :
  ∀ (ρ θ : ℝ),
    ∃ t : ℝ, line_l_parametric t = (ρ * cos θ, ρ * sin θ) →
    ρ * (cos θ + sqrt 3 * sin θ) = 6 := sorry

theorem rectangular_coordinate_equation_of_curve_C :
  ∀ (x y : ℝ), 
    curve_C_polar some_angle → 
    (x = ρ * cos some_angle) ∧ (y = ρ * sin some_angle) →
    (x - 1)^2 + y^2 = 1 := sorry

theorem minimum_value_of_OM_over_ON :
  ∀ (θ0 : ℝ),
    let ρ_M := λ θ₀ : ℝ, 3 / (sin (θ₀ + Real.pi / 6))
    let ρ_N := λ θ₀ : ℝ, 2 * cos θ₀
    ρ_M θ0 / ρ_N θ0 = 2 := sorry

end polar_coordinate_equation_of_line_l_rectangular_coordinate_equation_of_curve_C_minimum_value_of_OM_over_ON_l291_291651


namespace pyramid_surface_area_l291_291065

-- Definitions based on conditions
def upper_base_edge_length : ℝ := 2
def lower_base_edge_length : ℝ := 4
def side_edge_length : ℝ := 2

-- Problem statement in Lean
theorem pyramid_surface_area :
  let slant_height := Real.sqrt ((side_edge_length ^ 2) - (1 ^ 2))
  let perimeter_base := (4 * upper_base_edge_length) + (4 * lower_base_edge_length)
  let lsa := (perimeter_base * slant_height) / 2
  let total_surface_area := lsa + (upper_base_edge_length ^ 2) + (lower_base_edge_length ^ 2)
  total_surface_area = 10 * Real.sqrt 3 + 20 := sorry

end pyramid_surface_area_l291_291065


namespace problem_statement_l291_291151

def base7_representation (n : ℕ) : ℕ :=
  let rec digits (n : ℕ) (acc : ℕ) (power : ℕ) : ℕ :=
    if n = 0 then acc
    else digits (n / 7) (acc + (n % 7) * power) (power * 10)
  digits n 0 1

def even_digits_count (n : ℕ) : ℕ :=
  let rec count (n : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else let d := n % 10 in
         count (n / 10) (if d % 2 = 0 then acc + 1 else acc)
  count n 0

theorem problem_statement : even_digits_count (base7_representation 528) = 0 := sorry

end problem_statement_l291_291151


namespace problem1_problem2_l291_291735

-- Definition for Problem 1
def problem1_condition (x : ℝ) : Prop := x ∈ set.Iio 1 ∪ set.Ioi 6

theorem problem1 (x : ℝ) (h : problem1_condition x) : (4 * x - 9) / (x - 1) > 3 := 
by
  sorry

-- Definition for Problem 2
def problem2_condition (x : ℝ) : Prop := x ∈ set.Ioi 2

theorem problem2 (x : ℝ) (h : problem2_condition x) : abs (x - 6) < 2 * x := 
by
  sorry

end problem1_problem2_l291_291735


namespace lineup_ways_l291_291686

def liu_li_liu_child_li_child := (liu, li, liu_child, li_child : Type)

theorem lineup_ways 
(h1 : True) -- there are 6 people: Liu, Li, Liu's child, Li's child, and not relevant here
(h2 : True) -- fathers in first and last positions
(h3 : True) -- the children must stand together) :

∃ n : ℕ, n = 24 :=
begin
  -- There are 2 ways for the fathers to stand (Liu first and Li last, Li first and Liu last)
  -- There are 2 ways for the children to stand together (Liu's child can be first or last of the two)
  -- Considering the children as a single unit, 3 units can be arranged in 3! ways
  -- Total is 2 * 2 * 6 = 24
  sorry
end

end lineup_ways_l291_291686


namespace student_must_be_wrong_l291_291074

theorem student_must_be_wrong (m n : ℕ) (h1 : n > 0) (h2 : n ≤ 100) (h3 : 0.167 ≤ m / n ∧ m / n < 0.168) :
  ¬ (0 < 6000 * m - 1000 * n ∧ 6000 * m - 1000 * n < 800) :=
by
  sorry

end student_must_be_wrong_l291_291074


namespace numberOfBicycles_l291_291772

-- Definitions based on conditions
def numberOfTricycles : ℕ := 7
def wheelsPerBicycle : ℕ := 2
def wheelsPerTricycle : ℕ := 3
def totalWheels : ℕ := 53

-- Proof statement
theorem numberOfBicycles : ∃ b : ℕ, 2 * b + 3 * numberOfTricycles = totalWheels ∧ b = 16 :=
by {
  use 16,
  split,
  {
    -- showing 2 * 16 + 3 * 7 = 53
    calc
      2 * 16 + 3 * 7 = 32 + 21 : by norm_num
      ... = 53 : by norm_num
  },
  {
    -- showing b = 16
    refl
  }
}

end numberOfBicycles_l291_291772


namespace tom_pays_l291_291370

-- Definitions based on the conditions
def number_of_lessons : Nat := 10
def cost_per_lesson : Nat := 10
def free_lessons : Nat := 2

-- Desired proof statement
theorem tom_pays {number_of_lessons cost_per_lesson free_lessons : Nat} :
  (number_of_lessons - free_lessons) * cost_per_lesson = 80 :=
by
  sorry

end tom_pays_l291_291370


namespace problem_statement_l291_291150

def base7_representation (n : ℕ) : ℕ :=
  let rec digits (n : ℕ) (acc : ℕ) (power : ℕ) : ℕ :=
    if n = 0 then acc
    else digits (n / 7) (acc + (n % 7) * power) (power * 10)
  digits n 0 1

def even_digits_count (n : ℕ) : ℕ :=
  let rec count (n : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else let d := n % 10 in
         count (n / 10) (if d % 2 = 0 then acc + 1 else acc)
  count n 0

theorem problem_statement : even_digits_count (base7_representation 528) = 0 := sorry

end problem_statement_l291_291150


namespace triangle_area_l291_291489

theorem triangle_area (f : ℝ → ℝ) (x1 x2 yIntercept base height area : ℝ)
  (h1 : ∀ x, f x = (x - 4)^2 * (x + 3))
  (h2 : f 0 = yIntercept)
  (h3 : x1 = -3)
  (h4 : x2 = 4)
  (h5 : base = x2 - x1)
  (h6 : height = yIntercept)
  (h7 : area = 1/2 * base * height) :
  area = 168 := sorry

end triangle_area_l291_291489


namespace length_of_XY_l291_291666

namespace MathProof

-- Definitions for the rectangle and kite properties
def Rectangle (A B C D : ℝ^2) : Prop :=
  (dist A B = 5) ∧ (dist B C = 10) ∧ (dist C D = 5) ∧ (dist D A = 10) 

def Kite (W X Y Z : ℝ^2) : Prop :=
  (dist W X = dist W Z) ∧ (dist X Y = dist Y Z)

def RightAngle (A B C : ℝ^2) : Prop :=
  inner (B - A) (C - B) = 0

def kite_property (W X Y Z : ℝ^2) :=
  Kite W X Y Z ∧ RightAngle Z W X ∧ dist W X = dist W Z ∧ dist W X = Real.sqrt 13

-- Main theorem statement
theorem length_of_XY (A B C D W X Y Z : ℝ^2)
  (H1 : Rectangle A B C D)
  (H2 : kite_property W X Y Z)
  (H3 : dist X Y = dist Y Z) : 
  dist X Y = Real.sqrt 65 :=
sorry

end MathProof

end length_of_XY_l291_291666


namespace general_term_of_a_determine_t_find_m_l291_291178

-- First, define the sequences and conditions.
def a (n : ℕ) : ℕ := 2 ^ n
def b (n : ℕ) : ℕ := 2 * n
def c (n : ℕ) :=
  if n % 3 = 0 then 2 else if n % 3 = 1 then 2 ^ (n / 3 + 1) else 2

-- Define the arithmetic sequence condition
def is_arithmetic_mean (a : ℕ → ℕ) (q : ℕ) : Prop :=
  6 * a 3 = 8 * a 1 + a 5

-- Define the condition for b_n to be an arithmetic sequence
def is_arithmetic_seq (b : ℕ → ℕ) (t : ℝ) : Prop :=
  ∀ n : ℕ, n ≠ 0 → 2 * n^2 - (t + b n) * n + (3 / 2) * b n = 0

-- Define the condition for T_m = 2c_{m+1}
def T (n : ℕ) : ℕ := 
  (Finset.range n).sum c

def satisfies_condition (T : ℕ → ℕ) (c : ℕ → ℕ) (m : ℕ) : Prop :=
  T m = 2 * c (m + 1)

-- Now state the problems as propositions.
theorem general_term_of_a : 
  is_arithmetic_mean a 2 → ∀ n, a n = 2 ^ n :=
sorry

theorem determine_t :
  ∃ t : ℝ, is_arithmetic_seq b t ∧ ∀ n t, b n = 2 * n :=
sorry

theorem find_m :
  satisfies_condition T c 2 ∧ ∀ m, m ≠ 2 → ¬ satisfies_condition T c m :=
sorry

end general_term_of_a_determine_t_find_m_l291_291178


namespace max_numbers_vasya_l291_291003

def condition (a b : ℕ) : Prop := ¬ (a + b) % (a - b) = 0

def max_numbers : ℕ := 675

theorem max_numbers_vasya :
  ∃ S : set ℕ, S.card = max_numbers ∧ 
  (∀ n ∈ S, n ≤ 2023) ∧ 
  (∀ a b ∈ S, a ≠ b → condition a b) := 
sorry

end max_numbers_vasya_l291_291003


namespace point_coordinates_l291_291623

theorem point_coordinates (M : ℝ × ℝ) 
    (dist_x_axis : abs M.2 = 3)
    (dist_y_axis : abs M.1 = 2)
    (in_second_quadrant : M.1 < 0 ∧ M.2 > 0) : 
    M = (-2, 3) :=
begin
  sorry -- Proof to be filled in
end

end point_coordinates_l291_291623


namespace f_is_periodic_l291_291291

theorem f_is_periodic (f : ℝ → ℝ)
  (H : ∀ x : ℝ, f(x + 1) + f(x - 1) = Real.sqrt 2 * f(x)) :
  ∀ x : ℝ, f(x + 4) = f(x) :=
by
  sorry

end f_is_periodic_l291_291291


namespace OP_cannot_be_determined_l291_291281

-- Let O be the intersection point of medians AP and CQ of triangle ABC.
-- O is the centroid of triangle ABC.
-- Given OQ = 3 inches.
-- Prove that OP cannot be determined.
theorem OP_cannot_be_determined (O A B C P Q : Type) [MetricSpace O]
  (h1 : centroid_triangle O A Q)
  (h2 : OQ = 3) : ∃! (OP : ℝ), OP is_undefined :=
sorry

end OP_cannot_be_determined_l291_291281


namespace FiveDigitNumbers_count_l291_291757

theorem FiveDigitNumbers_count :
  let digits := {1, 2, 3, 4, 5}
  let adj (x y : ℕ) : Prop := 
    x = y + 1 ∨ x = y - 1
  ∀ (count : ℕ), 
  (∀ (l : List ℕ), 
   l.permutations.count 
    (λ l, l.nodup ∧ l.length = 5 ∧ 
           (list.all_diff l digits) ∧ 
           (¬ adj 5 1) ∧ (¬ adj 5 2))) = count → 
  count = 36 := 
sorry

end FiveDigitNumbers_count_l291_291757


namespace min_tries_to_get_blue_and_yellow_l291_291413

theorem min_tries_to_get_blue_and_yellow 
  (purple blue yellow : ℕ) 
  (h_purple : purple = 7) 
  (h_blue : blue = 5)
  (h_yellow : yellow = 11) :
  ∃ n, n = 9 ∧ (∀ tries, tries ≥ n → (∃ i j, (i ≤ purple ∧ j ≤ tries - i ∧ j ≤ blue) → (∃ k, k = tries - i - j ∧ k ≤ yellow))) :=
by sorry

end min_tries_to_get_blue_and_yellow_l291_291413


namespace number_of_solutions_l291_291603

theorem number_of_solutions (x : ℝ) :
  (∀ x, -30 < x ∧ x < 150 ∧ (cos x)^2 + 3 * (sin x)^2 = 1) → 
  (∃ n : ℕ, n = 57) :=
by sorry

end number_of_solutions_l291_291603


namespace polar_circle_eqn_length_segment_PQ_l291_291647

-- Definition of the parametric equations for circle C
def parametric_circle (ϕ : ℝ) := (1 + cos ϕ, sin ϕ)

-- Definition of the polar equation of line l
def polar_line (θ : ℝ) (ρ : ℝ) : Prop :=
  ρ * (sin θ + sqrt 3 * cos θ) = 3 * sqrt 3

-- Polar equation of the given circle C
theorem polar_circle_eqn : ∀ ρ θ : ℝ, (∃ ϕ, (1 + cos ϕ, sin ϕ) = (ρ * cos θ, ρ * sin θ)) → ρ = 2 * cos θ :=
by
  sorry

-- Length of segment PQ for the given conditions
theorem length_segment_PQ : ∀ θ ρ1 ρ2 : ℝ, 
  (θ = π / 3) ∧ (ρ1 = 2 * cos θ) ∧ (polar_line θ ρ2) → 
  (ρ1, θ) = (1, π / 3) ∧ (ρ2, θ) = (3, π / 3) ∧ abs (ρ1 - ρ2) = 2 :=
by
  sorry

end polar_circle_eqn_length_segment_PQ_l291_291647


namespace minimum_additional_coins_l291_291853

-- The conditions
def total_friends : ℕ := 15
def current_coins : ℕ := 100

-- The fact that the total coins required to give each friend a unique number of coins from 1 to 15 is 120
def total_required_coins : ℕ := (total_friends * (total_friends + 1)) / 2

-- The theorem stating the required number of additional coins
theorem minimum_additional_coins (total_friends : ℕ) (current_coins : ℕ) (total_required_coins : ℕ) : ℕ :=
  sorry

end minimum_additional_coins_l291_291853


namespace negative_number_among_options_l291_291015

theorem negative_number_among_options :
  let A := |(-2 : ℤ)|
      B := real.sqrt 3
      C := (0 : ℤ)
      D := (-5 : ℤ)
  in D = -5 := 
by 
  sorry

end negative_number_among_options_l291_291015


namespace sweet_potato_leftover_l291_291798

theorem sweet_potato_leftover (total_kg : ℝ) (per_person_kg : ℝ) (kg_to_g : ℝ) :
  total_kg = 52.5 → per_person_kg = 5 → kg_to_g = 1000 →
  (total_kg - int.floor (total_kg / per_person_kg) * per_person_kg) * kg_to_g = 2500 :=
by
  intros h1 h2 h3
  sorry

end sweet_potato_leftover_l291_291798


namespace bisected_angle_of_60_degree_l291_291006

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α
def bisect (α : ℝ) : ℝ := α / 2

theorem bisected_angle_of_60_degree : bisect (supplement (complement 60)) = 75 := by
  sorry

end bisected_angle_of_60_degree_l291_291006


namespace range_of_a_l291_291588

theorem range_of_a (a : ℝ) :
  (∀ (p q : ℝ), p ∈ Ioo 0 1 → q ∈ Ioo 0 1 → p ≠ q → (a * log (p + 2) - (p + 1)^2 - (a * log (q + 2) - (q + 1)^2)) / (p - q) > 1) →
  15 ≤ a :=
sorry

end range_of_a_l291_291588


namespace quadrilateral_area_l291_291610

-- Define the regular pentagon with apothem
structure RegularPentagon where
  apothem : ℝ
  is_regular : True -- This flag indicates the regularity of the pentagon

-- Define the midpoints structure for quadrilateral in the pentagon
structure PentMidpoints (pentagon : RegularPentagon) where
  S : Fin 5 → ℝ × ℝ
  is_midpoint : ∀ i : Fin 4, (S i).fst + (S i.succ).fst = pentagon.apothem ∧ (S i).snd + (S i.succ).snd = pentagon.apothem

-- Define the quadrilateral using midpoints
def Quadrilateral (pentagon : RegularPentagon) (midpoints : PentMidpoints pentagon) : Prop :=
  ∃ area : ℝ, area = 12.5 ∧ midpoints.is_midpoint

theorem quadrilateral_area (pentagon : RegularPentagon) (midpoints : PentMidpoints pentagon) :
  pentagon.apothem = 5 → Quadrilateral pentagon midpoints := 
by
  sorry

end quadrilateral_area_l291_291610


namespace candy_distribution_impossible_l291_291728

theorem candy_distribution_impossible :
  ¬ ∃ n : ℕ, 7 * n = 60 :=
by
  intro h
  cases h with n hn
  have h7 : n = 60 / 7 := by sorry -- For simplicity's sake
  linarith -- Indicating contradiction by the incompatible nature of the equation
  sorry -- Placeholder for any additional contraction demonstration if necessary

end candy_distribution_impossible_l291_291728


namespace p_sufficient_not_necessary_for_q_l291_291683

def p (x1 x2 : ℝ) : Prop := x1 > 1 ∧ x2 > 1
def q (x1 x2 : ℝ) : Prop := x1 + x2 > 2 ∧ x1 * x2 > 1

theorem p_sufficient_not_necessary_for_q : 
  (∀ x1 x2 : ℝ, p x1 x2 → q x1 x2) ∧ ¬ (∀ x1 x2 : ℝ, q x1 x2 → p x1 x2) :=
by 
  sorry

end p_sufficient_not_necessary_for_q_l291_291683


namespace find_general_terms_find_T_2016_l291_291952

noncomputable def a_n (n : ℕ) : ℕ := 3 * n
noncomputable def b_n (n : ℕ) : ℕ := 64 ^ n

def c_n (n : ℕ) : ℝ := Real.log 64 * n / Real.log 2 -- log_2(64^n) = 6n
def T_n (n : ℕ) : ℝ := (1 - 1 / (n + 1).toReal) / 36

theorem find_general_terms (n : ℕ) :
  (∃ d, (6 * a_n 1 + 15 * d = 5 * (2 * a_n 1 + d) + 18) ∧
        (a_n 1 + (3 * n - 1) * d = 3 * (a_n 1 + (n - 1) * d))) ∧ 
  a_n 1 = 3 ∧ b_n 1 = 64 ∧
  (∀ n ≥ 2, b_n n = 64 ^ n) :=
sorry

theorem find_T_2016 (n : ℕ) : T_n 2016 = 56 / 2017 :=
sorry

end find_general_terms_find_T_2016_l291_291952


namespace solve_x_eq_i_div_4_l291_291733

noncomputable def solveForX (x : ℂ) : Prop :=
  2 + 3 * complex.I * x = 4 - 5 * complex.I * x

theorem solve_x_eq_i_div_4 :
  ∃ x : ℂ, solveForX x ∧ x = (complex.I / 4) := by
  sorry

end solve_x_eq_i_div_4_l291_291733


namespace edward_spent_on_pens_l291_291895

variable {initial_money books_cost pens_cost remaining_money money_left_after_books : ℕ}

variables (h1 : initial_money = 41) (h2 : books_cost = 6) (h3 : remaining_money = 19)

theorem edward_spent_on_pens (h4 : money_left_after_books = initial_money - books_cost)
    (h5 : pens_cost = money_left_after_books - remaining_money) : pens_cost = 16 :=
by
    rw [h1, h2, h3, h4, h5]
    sorry

end edward_spent_on_pens_l291_291895


namespace new_coordinates_A_l291_291582

theorem new_coordinates_A' (A : ℝ × ℝ) (x_move y_move : ℝ) (hx : A = (5, 4)) 
  (hy_move : y_move = 3) (hx_move : x_move = 4) :
    (A.1 - x_move, A.2 - y_move) = (1, 1) :=
by
  have hA := hx
  rw [←hA]
  simp [hx_move, hy_move]
  sorry

end new_coordinates_A_l291_291582


namespace fraction_product_is_simplified_form_l291_291094

noncomputable def fraction_product : ℚ := (2 / 3) * (5 / 11) * (3 / 8)

theorem fraction_product_is_simplified_form :
  fraction_product = 5 / 44 :=
by
  sorry

end fraction_product_is_simplified_form_l291_291094


namespace value_of_x_plus_y_for_circle_center_l291_291538

theorem value_of_x_plus_y_for_circle_center (x y : ℝ) :
  (∃ (h : y = -3), x = 2) → x + y = -1 :=
by
  -- condition of the circle equation
  assume h : ∃ (h : y = -3), x = 2,
  sorry

end value_of_x_plus_y_for_circle_center_l291_291538


namespace gcf_60_75_l291_291400

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l291_291400


namespace interval_for_monotonically_increasing_g_l291_291205

noncomputable def f (x φ : ℝ) : ℝ := sqrt 3 * Real.sin (2 * x + φ) - Real.cos (2 * x + φ)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x - (π / 3))

theorem interval_for_monotonically_increasing_g :
  ∀ ϕ : ℝ, (0 < ϕ ∧ ϕ < π) →  
  ∃ a b : ℝ, a = -π / 3 ∧ b = π / 6 ∧ (∀ x : ℝ, a < x ∧ x < b → Deriv g x > 0) :=
by 
  intros φ h_ph
  use [-π / 3, π / 6]
  constructor
  . simp
  constructor
  . simp
  sorry

end interval_for_monotonically_increasing_g_l291_291205


namespace sin_cos_expression_l291_291932

theorem sin_cos_expression (a b θ : ℝ) (h : (sin θ)^6 / a^2 + (cos θ)^6 / b^2 = 1 / (a + b)) :
  (sin θ)^12 / a^5 + (cos θ)^12 / b^5 = 1 / (a + b)^5 := 
sorry

end sin_cos_expression_l291_291932


namespace number_of_divisors_of_36_l291_291986

theorem number_of_divisors_of_36: 
  (finset.of_list [1, 2, 3, 4, 6, 9, 12, 18, 36]).card * 2 = 18 := by
  sorry

end number_of_divisors_of_36_l291_291986


namespace marla_bags_per_trip_l291_291693

-- Define the conditions
def canvas_bag_co2_emissions : ℝ := 600
def plastic_bag_co2_emissions_ounces : ℝ := 4
def shopping_trips : ℕ := 300
def ounces_to_pounds : ℝ := 1 / 16

-- Define the plastic bag CO2 emissions in pounds
def plastic_bag_co2_emissions_pounds : ℝ :=
  plastic_bag_co2_emissions_ounces * ounces_to_pounds

-- The theorem to prove
theorem marla_bags_per_trip (x : ℝ) :
  (plastic_bag_co2_emissions_pounds * x * shopping_trips = canvas_bag_co2_emissions) →
  x = 8 :=
by
  sorry

end marla_bags_per_trip_l291_291693


namespace slower_train_speed_l291_291380

-- Define the conditions and the proof problem
theorem slower_train_speed:
  ∀ (V : ℝ), let train_length := 75 in
             let combined_length := 2 * train_length in
             let fast_speed := 46 in
             let relative_speed := (fast_speed - V) * (5/18) in
             let time_to_overtake := 54 in
             combined_length = relative_speed * time_to_overtake →
             V = 36 :=
by
  intros V train_length combined_length fast_speed relative_speed time_to_overtake h
  sorry

end slower_train_speed_l291_291380


namespace train_length_approx_l291_291033

def train_speed_km_hr : ℝ := 52
def time_seconds : ℝ := 9
def speed_m_s : ℝ := (train_speed_km_hr * 1000) / 3600
def length_of_train : ℝ := speed_m_s * time_seconds

theorem train_length_approx : abs (length_of_train - 129.96) < 0.01 := 
by
  sorry

end train_length_approx_l291_291033


namespace tan_of_angle_in_second_quadrant_l291_291576

theorem tan_of_angle_in_second_quadrant (α : ℝ) (h1 : π/2 < α ∧ α < π) (h2 : cos α = -12 / 13) : tan α = -5 / 12 :=
sorry

end tan_of_angle_in_second_quadrant_l291_291576


namespace sin_double_angle_sub_pi_six_l291_291917

theorem sin_double_angle_sub_pi_six (x : ℝ) 
  (h : cos (x + π / 6) + sin ((2 * π / 3) + x) = 1 / 2) : 
  sin (2 * x - π / 6) = 7 / 8 := 
by
  sorry

end sin_double_angle_sub_pi_six_l291_291917


namespace units_digit_factorial_sum_l291_291153

theorem units_digit_factorial_sum :
  (∑ n in Finset.range 2011, n! % 10) % 10 = 3 :=
by
  sorry

end units_digit_factorial_sum_l291_291153


namespace find_integer_n_l291_291999

theorem find_integer_n (a b : ℤ) (n : ℤ) (h₀ : a ≡ 27 [ZMOD 53]) (h₁ : b ≡ 88 [ZMOD 53]) (h_set : n ∈ set.Ico 120 172) : a - b ≡ n [ZMOD 53] := 
by { sorry }

end find_integer_n_l291_291999


namespace first_part_is_13_l291_291046

-- Definitions for the conditions
variables (x y : ℕ)

-- Conditions given in the problem
def condition1 : Prop := x + y = 24
def condition2 : Prop := 7 * x + 5 * y = 146

-- The theorem we need to prove
theorem first_part_is_13 (h1 : condition1 x y) (h2 : condition2 x y) : x = 13 :=
sorry

end first_part_is_13_l291_291046


namespace triangle_angle_side_relation_l291_291268

-- Definitions of the angles and sides in the triangle
variable {A B : ℝ} -- Angles of the triangle
variable {a b : ℝ} -- Sides opposite to angles A and B respectively

-- The theorem statement
theorem triangle_angle_side_relation (h1 : A > B) (h2 : ∑ {X Y Z : ℝ} = 180 ∧ a > 0 ∧ b > 0 ) :
  (A > B ↔ a > b) :=
sorry

end triangle_angle_side_relation_l291_291268


namespace city_renumbering_not_possible_l291_291505

-- Defining the problem conditions
def city_renumbering_invalid (city_graph : Type) (connected : city_graph → city_graph → Prop) : Prop :=
  ∃ (M N : city_graph), ∀ (renumber : city_graph → city_graph),
  (renumber M = N ∧ renumber N = M) → ¬(
    ∀ x y : city_graph,
    connected x y ↔ connected (renumber x) (renumber y)
  )

-- Statement of the problem
theorem city_renumbering_not_possible (city_graph : Type) (connected : city_graph → city_graph → Prop) :
  city_renumbering_invalid city_graph connected :=
sorry

end city_renumbering_not_possible_l291_291505


namespace ratio_3_7_not_possible_l291_291422

theorem ratio_3_7_not_possible (n : ℕ) (h : 30 < n ∧ n < 40) :
  ¬ (∃ k : ℕ, n = 10 * k) :=
by {
  sorry
}

end ratio_3_7_not_possible_l291_291422


namespace smallest_dimension_of_crate_is_10_l291_291056

noncomputable def cylinder_radius : ℝ := 5
noncomputable def crate_dimension2 : ℝ := 8
noncomputable def crate_dimension3 : ℝ := 12
noncomputable def crate_dimension1 : ℝ := 10

def smallest_crate_dimension : ℝ := Inf {crate_dimension1, crate_dimension2, crate_dimension3}

theorem smallest_dimension_of_crate_is_10 :
  smallest_crate_dimension = 10 :=
by sorry

end smallest_dimension_of_crate_is_10_l291_291056


namespace median_of_data_l291_291452

def data : List ℕ := [6, 5, 7, 6, 6]

theorem median_of_data : List.median data = 6 := by
  sorry

end median_of_data_l291_291452


namespace polynomial_expansion_correct_l291_291501

open Polynomial

-- Define the two polynomials in question.
def poly1 : Polynomial ℤ := 2 + (X^2)
def poly2 : Polynomial ℤ := 3 - (X^3) + (X^5)

-- The target polynomial after expansion.
def expandedPoly : Polynomial ℤ := 6 + 3 * (X^2) - 2 * (X^3) + X^5 + X^7

-- State the theorem to be proved
theorem polynomial_expansion_correct : poly1 * poly2 = expandedPoly := 
by
  sorry

end polynomial_expansion_correct_l291_291501


namespace four_digit_numbers_divisible_by_eleven_l291_291600

theorem four_digit_numbers_divisible_by_eleven (a b c d : ℕ) :
  (∀ a b c d, a + b + c + d = 16 ∧ ((a + c) - (b + d)) % 11 = 0 → 567) := sorry

end four_digit_numbers_divisible_by_eleven_l291_291600


namespace canal_depth_height_l291_291825

noncomputable def canal_section_geometry (h_t : ℝ) (π : ℝ) : Prop :=
  let a := 12
  let b := 8
  let r := 6
  let A_t := (1/2) * (a + b) * h_t
  let A_s := (1/2) * π * r ^ 2
  A_t + A_s = 1800

theorem canal_depth_height (π : ℝ) (h_t : ℝ) (approx_h_t := 180 - 1.8 * π) :
  canal_section_geometry h_t π → 
  h_t ≈ approx_h_t ∧ 6 = 6 := 
sorry

end canal_depth_height_l291_291825


namespace marks_of_A_l291_291414

variable (a b c d e : ℕ)

theorem marks_of_A:
  (a + b + c = 144) →
  (a + b + c + d = 188) →
  (e = d + 3) →
  (b + c + d + e = 192) →
  a = 43 := 
by 
  intros h1 h2 h3 h4
  sorry

end marks_of_A_l291_291414


namespace relationship_between_a_and_b_l291_291930

open Real

theorem relationship_between_a_and_b
   (a b : ℝ)
   (ha : 0 < a ∧ a < 1)
   (hb : 0 < b ∧ b < 1)
   (hab : (1 - a) * b > 1 / 4) :
   a < b := 
sorry

end relationship_between_a_and_b_l291_291930
