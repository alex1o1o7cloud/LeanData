import Mathlib

namespace farmer_profit_l130_13072

def piglet_cost_per_month : Int := 10
def pig_revenue : Int := 300
def num_piglets_sold_early : Int := 3
def num_piglets_sold_late : Int := 3
def early_sale_months : Int := 12
def late_sale_months : Int := 16

def total_profit (num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months : Int) 
  (piglet_cost_per_month pig_revenue : Int) : Int := 
  let early_cost := num_piglets_sold_early * piglet_cost_per_month * early_sale_months
  let late_cost := num_piglets_sold_late * piglet_cost_per_month * late_sale_months
  let total_cost := early_cost + late_cost
  let total_revenue := (num_piglets_sold_early + num_piglets_sold_late) * pig_revenue
  total_revenue - total_cost

theorem farmer_profit : total_profit num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months piglet_cost_per_month pig_revenue = 960 := by
  sorry

end farmer_profit_l130_13072


namespace train_crossing_time_approx_l130_13090

noncomputable def train_length : ℝ := 90 -- in meters
noncomputable def speed_kmh : ℝ := 124 -- in km/hr
noncomputable def conversion_factor : ℝ := 1000 / 3600 -- km/hr to m/s conversion factor
noncomputable def speed_ms : ℝ := speed_kmh * conversion_factor -- speed in m/s
noncomputable def time_to_cross : ℝ := train_length / speed_ms -- time in seconds

theorem train_crossing_time_approx :
  abs (time_to_cross - 2.61) < 0.01 := 
by 
  sorry

end train_crossing_time_approx_l130_13090


namespace complement_intersection_l130_13009

def U : Set ℤ := {1, 2, 3, 4, 5}
def P : Set ℤ := {2, 4}
def Q : Set ℤ := {1, 3, 4, 6}
def C_U_P : Set ℤ := {x ∈ U | x ∉ P}

theorem complement_intersection :
  (C_U_P ∩ Q) = {1, 3} :=
by sorry

end complement_intersection_l130_13009


namespace if_2_3_4_then_1_if_1_3_4_then_2_l130_13097

variables {Plane Line : Type} 
variables (α β : Plane) (m n : Line)

-- assuming the perpendicular relationships as predicates
variable (perp : Plane → Plane → Prop) -- perpendicularity between planes
variable (perp' : Line → Line → Prop) -- perpendicularity between lines
variable (perp'' : Line → Plane → Prop) -- perpendicularity between line and plane

theorem if_2_3_4_then_1 :
  perp α β → perp'' m β → perp'' n α → perp' m n :=
by
  sorry

theorem if_1_3_4_then_2 :
  perp' m n → perp'' m β → perp'' n α → perp α β :=
by
  sorry

end if_2_3_4_then_1_if_1_3_4_then_2_l130_13097


namespace reciprocal_of_negative_2023_l130_13091

theorem reciprocal_of_negative_2023 : (1 / (-2023 : ℤ)) = -(1 / (2023 : ℤ)) := by
  sorry

end reciprocal_of_negative_2023_l130_13091


namespace triangle_side_lengths_l130_13083

theorem triangle_side_lengths (a b c : ℝ) 
  (h1 : a + b + c = 18) 
  (h2 : a + b = 2 * c) 
  (h3 : b = 2 * a):
  a = 4 ∧ b = 8 ∧ c = 6 := 
by
  sorry

end triangle_side_lengths_l130_13083


namespace area_of_region_B_l130_13063

noncomputable def region_B_area : ℝ :=
  let square_area := 900
  let excluded_area := 28.125 * Real.pi
  square_area - excluded_area

theorem area_of_region_B : region_B_area = 900 - 28.125 * Real.pi :=
by {
  sorry
}

end area_of_region_B_l130_13063


namespace sum_of_primes_between_20_and_30_l130_13039

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l130_13039


namespace arccos_one_eq_zero_l130_13003

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end arccos_one_eq_zero_l130_13003


namespace line_equation_l130_13081

-- Define the conditions as given in the problem
def passes_through (P : ℝ × ℝ) (line : ℝ × ℝ) : Prop :=
  line.fst * P.fst + line.snd * P.snd + 1 = 0

def equal_intercepts (line : ℝ × ℝ) : Prop :=
  line.fst = line.snd

theorem line_equation (P : ℝ × ℝ) (hP : P = (-2, -1)) :
  (∃ (k : ℝ), passes_through P (1, -2 * k)) ∨ (∃ (m : ℝ), passes_through P (1, m) ∧ m = - 1) :=
sorry

end line_equation_l130_13081


namespace eric_green_marbles_l130_13071

theorem eric_green_marbles (total_marbles white_marbles blue_marbles : ℕ) (h_total : total_marbles = 20)
  (h_white : white_marbles = 12) (h_blue : blue_marbles = 6) :
  total_marbles - (white_marbles + blue_marbles) = 2 := 
by
  sorry

end eric_green_marbles_l130_13071


namespace special_number_exists_l130_13042

theorem special_number_exists (a b c d e : ℕ) (h1 : a < b ∧ b < c ∧ c < d ∧ d < e)
    (h2 : a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e) 
    (h_num : a * 10 + b = 13 ∧ c = 4 ∧ d * 10 + e = 52) :
    (10 * a + b) * c = 10 * d + e :=
by
  sorry

end special_number_exists_l130_13042


namespace dice_minimum_rolls_l130_13008

theorem dice_minimum_rolls (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6)
                           (h2 : 1 ≤ d2 ∧ d2 ≤ 6) (h3 : 1 ≤ d3 ∧ d3 ≤ 6) 
                           (h4 : 1 ≤ d4 ∧ d4 ≤ 6) :
  ∃ n, n = 43 ∧ ∀ (S : ℕ) (x : ℕ → ℕ), 
  (∀ i, 4 ≤ S ∧ S ≤ 24 ∧ x i = 4 ∧ (x i ≤ 6)) →
  (n ≤ 43) ∧ (∃ (k : ℕ), k ≥ 3) :=
sorry

end dice_minimum_rolls_l130_13008


namespace present_value_of_machine_l130_13073

theorem present_value_of_machine (r : ℝ) (t : ℕ) (V : ℝ) (P : ℝ) (h1 : r = 0.10) (h2 : t = 2) (h3 : V = 891) :
  V = P * (1 - r)^t → P = 1100 :=
by
  intro h
  rw [h3, h1, h2] at h
  -- The steps to solve for P are omitted as instructed
  sorry

end present_value_of_machine_l130_13073


namespace ac_le_bc_if_a_gt_b_and_c_le_zero_a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero_log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1_inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero_l130_13004

section

variable {a b c : ℝ}

-- Statement 1
theorem ac_le_bc_if_a_gt_b_and_c_le_zero (h1 : a > b) (h2 : c ≤ 0) : a * c ≤ b * c := 
  sorry

-- Statement 2
theorem a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero (h1 : a * c ^ 2 > b * c ^ 2) (h2 : b ≥ 0) : a ^ 2 > b ^ 2 := 
  sorry

-- Statement 3
theorem log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1 (h1 : a > b) (h2 : b > -1) : Real.log (a + 1) > Real.log (b + 1) := 
  sorry

-- Statement 4
theorem inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero (h1 : a > b) (h2 : a * b > 0) : 1 / a < 1 / b := 
  sorry

end

end ac_le_bc_if_a_gt_b_and_c_le_zero_a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero_log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1_inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero_l130_13004


namespace avg_first_3_is_6_l130_13032

theorem avg_first_3_is_6 (A B C D : ℝ) (X : ℝ)
  (h1 : (A + B + C) / 3 = X)
  (h2 : (B + C + D) / 3 = 5)
  (h3 : A + D = 11)
  (h4 : D = 4) :
  X = 6 := 
by
  sorry

end avg_first_3_is_6_l130_13032


namespace tangerine_initial_count_l130_13049

theorem tangerine_initial_count 
  (X : ℕ) 
  (h1 : X - 9 + 5 = 20) : 
  X = 24 :=
sorry

end tangerine_initial_count_l130_13049


namespace banana_to_pear_equiv_l130_13070

/-
Given conditions:
1. 5 bananas cost as much as 3 apples.
2. 9 apples cost the same as 6 pears.
Prove the equivalence between 30 bananas and 12 pears.

We will define the equivalences as constants and prove the cost equivalence.
-/

variable (cost_banana cost_apple cost_pear : ℤ)

noncomputable def cost_equiv : Prop :=
  (5 * cost_banana = 3 * cost_apple) ∧ 
  (9 * cost_apple = 6 * cost_pear) →
  (30 * cost_banana = 12 * cost_pear)

theorem banana_to_pear_equiv :
  cost_equiv cost_banana cost_apple cost_pear :=
by
  sorry

end banana_to_pear_equiv_l130_13070


namespace hexagon_colorings_l130_13065

-- Definitions based on conditions
def isValidColoring (A B C D E F : ℕ) (colors : Fin 7 → ℕ) : Prop :=
  -- Adjacent vertices must have different colors
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A ∧
  -- Diagonal vertices must have different colors
  A ≠ D ∧ B ≠ E ∧ C ≠ F

-- Function to count all valid colorings
def countValidColorings : ℕ :=
  let colors := List.range 7
  -- Calculate total number of valid colorings
  7 * 6 * 5 * 4 * 3 * 2

theorem hexagon_colorings : countValidColorings = 5040 := by
  sorry

end hexagon_colorings_l130_13065


namespace perp_to_par_perp_l130_13035

variable (m : Line)
variable (α β : Plane)

-- Conditions
axiom parallel_planes (α β : Plane) : Prop
axiom perp (m : Line) (α : Plane) : Prop

-- Statements
axiom parallel_planes_ax : parallel_planes α β
axiom perp_ax : perp m α

-- Goal
theorem perp_to_par_perp {m : Line} {α β : Plane} (h1 : perp m α) (h2 : parallel_planes α β) : perp m β := sorry

end perp_to_par_perp_l130_13035


namespace monkey_climbing_time_l130_13021

-- Define the conditions
def tree_height : ℕ := 20
def hop_distance : ℕ := 3
def slip_distance : ℕ := 2
def net_distance_per_hour : ℕ := hop_distance - slip_distance

-- Define the theorem statement
theorem monkey_climbing_time : ∃ (t : ℕ), t = 18 ∧ (net_distance_per_hour * (t - 1) + hop_distance) >= tree_height :=
by
  sorry

end monkey_climbing_time_l130_13021


namespace translation_correctness_l130_13015

-- Define the original function
def original_function (x : ℝ) : ℝ := 3 * x + 5

-- Define the translated function
def translated_function (x : ℝ) : ℝ := 3 * x

-- Define the condition for passing through the origin
def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

-- The theorem to prove the correct translation
theorem translation_correctness : passes_through_origin translated_function := by
  sorry

end translation_correctness_l130_13015


namespace fixed_point_translation_l130_13056

variable {R : Type*} [LinearOrderedField R]

def passes_through (f : R → R) (p : R × R) : Prop := f p.1 = p.2

theorem fixed_point_translation (f : R → R) (h : f 1 = 1) :
  passes_through (fun x => f (x + 2)) (-1, 1) :=
by
  sorry

end fixed_point_translation_l130_13056


namespace colten_chickens_l130_13068

/-
Define variables to represent the number of chickens each person has.
-/

variables (C : ℕ)   -- Number of chickens Colten has.
variables (S : ℕ)   -- Number of chickens Skylar has.
variables (Q : ℕ)   -- Number of chickens Quentin has.

/-
Define the given conditions
-/
def condition1 := Q + S + C = 383
def condition2 := Q = 2 * S + 25
def condition3 := S = 3 * C - 4

theorem colten_chickens : C = 37 :=
by
  -- Proof elaboration to be done with sorry for the auto proof
  sorry

end colten_chickens_l130_13068


namespace garden_area_garden_perimeter_l130_13031

noncomputable def length : ℝ := 30
noncomputable def width : ℝ := length / 2
noncomputable def area : ℝ := length * width
noncomputable def perimeter : ℝ := 2 * (length + width)

theorem garden_area :
  area = 450 :=
sorry

theorem garden_perimeter :
  perimeter = 90 :=
sorry

end garden_area_garden_perimeter_l130_13031


namespace length_of_NC_l130_13052

noncomputable def semicircle_radius (AB : ℝ) : ℝ := AB / 2

theorem length_of_NC : 
  ∀ (AB CD AN NB N M C NC : ℝ),
    AB = 10 ∧ AB = CD ∧ AN = NB ∧ AN + NB = AB ∧ M = N ∧ AB / 2 = semicircle_radius AB ∧ (NC^2 + semicircle_radius AB^2 = (2 * semicircle_radius AB)^2) →
    NC = 5 * Real.sqrt 3 := 
by 
  intros AB CD AN NB N M C NC h 
  rcases h with ⟨hAB, hCD, hAN, hSumAN, hMN, hRadius, hPythag⟩
  sorry

end length_of_NC_l130_13052


namespace find_integer_pairs_l130_13078

theorem find_integer_pairs (x y : ℤ) :
  x^4 + (y+2)^3 = (x+2)^4 ↔ (x, y) = (0, 0) ∨ (x, y) = (-1, -2) := sorry

end find_integer_pairs_l130_13078


namespace square_of_neg_3b_l130_13099

theorem square_of_neg_3b (b : ℝ) : (-3 * b)^2 = 9 * b^2 :=
by sorry

end square_of_neg_3b_l130_13099


namespace range_of_a_l130_13054

noncomputable def f (a x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x → (f a x) > 0) ↔ a ≤ 2 := 
by
  sorry

end range_of_a_l130_13054


namespace min_product_of_prime_triplet_l130_13088

theorem min_product_of_prime_triplet
  (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (hx_odd : x % 2 = 1) (hy_odd : y % 2 = 1) (hz_odd : z % 2 = 1)
  (h1 : x ∣ (y^5 + 1)) (h2 : y ∣ (z^5 + 1)) (h3 : z ∣ (x^5 + 1)) :
  (x * y * z) = 2013 := by
  sorry

end min_product_of_prime_triplet_l130_13088


namespace sum_g_equals_half_l130_13029

noncomputable def g (n : ℕ) : ℝ :=
  ∑' k, if k ≥ 3 then 1 / k ^ n else 0

theorem sum_g_equals_half : ∑' n : ℕ, g n.succ = 1 / 2 := 
sorry

end sum_g_equals_half_l130_13029


namespace new_difference_greater_l130_13025

theorem new_difference_greater (x y a b : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x > y) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab : a ≠ b) :
  (x + a) - (y - b) > x - y :=
by {
  sorry
}

end new_difference_greater_l130_13025


namespace items_sold_increase_by_20_percent_l130_13020

-- Assume initial variables P (price per item without discount) and N (number of items sold without discount)
variables (P N : ℝ)

-- Define the conditions and the final proof goal
theorem items_sold_increase_by_20_percent 
  (h1 : ∀ (P N : ℝ), P > 0 → N > 0 → (P * N > 0))
  (h2 : ∀ (P : ℝ), P' = P * 0.90)
  (h3 : ∀ (P' N' : ℝ), P' * N' = P * N * 1.08)
  : (N' - N) / N * 100 = 20 := 
sorry

end items_sold_increase_by_20_percent_l130_13020


namespace wheel_circumferences_satisfy_conditions_l130_13075

def C_f : ℝ := 24
def C_r : ℝ := 18

theorem wheel_circumferences_satisfy_conditions:
  360 / C_f = 360 / C_r + 4 ∧ 360 / (C_f - 3) = 360 / (C_r - 3) + 6 :=
by 
  have h1: 360 / C_f = 360 / C_r + 4 := sorry
  have h2: 360 / (C_f - 3) = 360 / (C_r - 3) + 6 := sorry
  exact ⟨h1, h2⟩

end wheel_circumferences_satisfy_conditions_l130_13075


namespace no_solution_system_l130_13059

theorem no_solution_system :
  ¬ ∃ (x y z : ℝ), (3 * x - 4 * y + z = 10) ∧ (6 * x - 8 * y + 2 * z = 5) ∧ (2 * x - y - z = 4) :=
by {
  sorry
}

end no_solution_system_l130_13059


namespace seashells_given_joan_to_mike_l130_13093

-- Declaring the context for the problem: Joan's seashells
def initial_seashells := 79
def remaining_seashells := 16

-- Proving how many seashells Joan gave to Mike
theorem seashells_given_joan_to_mike : (initial_seashells - remaining_seashells) = 63 :=
by
  -- This proof needs to be completed
  sorry

end seashells_given_joan_to_mike_l130_13093


namespace tan_alpha_solution_l130_13089

theorem tan_alpha_solution (α : Real) (h : Real.tan α = 1 / 2) : 
  (2 * Real.sin α + Real.cos α) / (4 * Real.sin α - Real.cos α) = 2 := 
by 
  sorry

end tan_alpha_solution_l130_13089


namespace find_y_given_x_eq_0_l130_13094

theorem find_y_given_x_eq_0 (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : 
  y = 21 / 2 :=
by
  sorry

end find_y_given_x_eq_0_l130_13094


namespace correct_transformation_l130_13062

-- Given transformations
def transformation_A (a : ℝ) : Prop := - (1 / a) = -1 / a
def transformation_B (a b : ℝ) : Prop := (1 / a) + (1 / b) = 1 / (a + b)
def transformation_C (a b : ℝ) : Prop := (2 * b^2) / a^2 = (2 * b) / a
def transformation_D (a b : ℝ) : Prop := (a + a * b) / (b + a * b) = a / b

-- Correct transformation is A.
theorem correct_transformation (a b : ℝ) : transformation_A a ∧ ¬transformation_B a b ∧ ¬transformation_C a b ∧ ¬transformation_D a b :=
sorry

end correct_transformation_l130_13062


namespace quadratic_inequality_solution_l130_13024

theorem quadratic_inequality_solution {a : ℝ} :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ a < -1 ∨ a > 3 :=
by sorry

end quadratic_inequality_solution_l130_13024


namespace exists_travel_route_l130_13018

theorem exists_travel_route (n : ℕ) (cities : Finset ℕ) 
  (ticket_price : ℕ → ℕ → ℕ)
  (h1 : cities.card = n)
  (h2 : ∀ c1 c2, c1 ≠ c2 → ∃ p, (ticket_price c1 c2 = p ∧ ticket_price c1 c2 = ticket_price c2 c1))
  (h3 : ∀ p1 p2 c1 c2 c3 c4,
    p1 ≠ p2 ∧ (ticket_price c1 c2 = p1) ∧ (ticket_price c3 c4 = p2) →
    p1 ≠ p2) :
  ∃ city : ℕ, ∀ m : ℕ, m = n - 1 →
  ∃ route : Finset (ℕ × ℕ),
  route.card = m ∧
  ∀ (t₁ t₂ : ℕ × ℕ), t₁ ∈ route → t₂ ∈ route → (t₁ ≠ t₂ → ticket_price t₁.1 t₁.2 < ticket_price t₂.1 t₂.2) :=
by
  sorry

end exists_travel_route_l130_13018


namespace triangle_area_is_correct_l130_13047

structure Point where
  x : ℝ
  y : ℝ

noncomputable def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)))

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩

theorem triangle_area_is_correct : area_of_triangle A B C = 2 := by
  sorry

end triangle_area_is_correct_l130_13047


namespace fencing_required_l130_13006

theorem fencing_required (L W : ℕ) (hL : L = 10) (hA : L * W = 600) : L + 2 * W = 130 :=
by
  sorry

end fencing_required_l130_13006


namespace swap_numbers_l130_13000

-- Define the initial state
variables (a b c : ℕ)
axiom initial_state : a = 8 ∧ b = 17

-- Define the assignment sequence
axiom swap_statement1 : c = b 
axiom swap_statement2 : b = a
axiom swap_statement3 : a = c

-- Define the theorem to be proved
theorem swap_numbers (a b c : ℕ) (initial_state : a = 8 ∧ b = 17)
  (swap_statement1 : c = b) (swap_statement2 : b = a) (swap_statement3 : a = c) :
  (a = 17 ∧ b = 8) :=
sorry

end swap_numbers_l130_13000


namespace range_of_k_l130_13010

theorem range_of_k (x y k : ℝ) (h1 : 2 * x - 3 * y = 5) (h2 : 2 * x - y = k) (h3 : x > y) : k > -5 :=
sorry

end range_of_k_l130_13010


namespace probability_of_two_boys_given_one_boy_l130_13001

-- Define the events and probabilities
def P_BB : ℚ := 1/4
def P_BG : ℚ := 1/4
def P_GB : ℚ := 1/4
def P_GG : ℚ := 1/4

def P_at_least_one_boy : ℚ := 1 - P_GG

def P_two_boys_given_at_least_one_boy : ℚ := P_BB / P_at_least_one_boy

-- Statement to be proven
theorem probability_of_two_boys_given_one_boy : P_two_boys_given_at_least_one_boy = 1/3 :=
by sorry

end probability_of_two_boys_given_one_boy_l130_13001


namespace geometric_sequence_sixth_term_l130_13046

theorem geometric_sequence_sixth_term (a : ℝ) (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^(7) = 2) :
  a * r^(5) = 16 :=
by
  sorry

end geometric_sequence_sixth_term_l130_13046


namespace Owen_final_turtle_count_l130_13011

variable (Owen_turtles : ℕ) (Johanna_turtles : ℕ)

def final_turtles (Owen_turtles Johanna_turtles : ℕ) : ℕ :=
  let initial_Owen_turtles := Owen_turtles
  let initial_Johanna_turtles := Owen_turtles - 5
  let Owen_after_month := initial_Owen_turtles * 2
  let Johanna_after_losing_half := initial_Johanna_turtles / 2
  let Owen_after_donation := Owen_after_month + Johanna_after_losing_half
  Owen_after_donation

theorem Owen_final_turtle_count : final_turtles 21 (21 - 5) = 50 :=
by
  sorry

end Owen_final_turtle_count_l130_13011


namespace percent_round_trip_tickets_l130_13098

variable (P : ℕ) -- total number of passengers

def passengers_with_round_trip_tickets (P : ℕ) : ℕ :=
  2 * (P / 5 / 2)

theorem percent_round_trip_tickets (P : ℕ) : 
  passengers_with_round_trip_tickets P = 2 * (P / 5 / 2) :=
by
  sorry

end percent_round_trip_tickets_l130_13098


namespace total_votes_l130_13014

theorem total_votes (P R : ℝ) (hP : P = 0.35) (diff : ℝ) (h_diff : diff = 1650) : 
  ∃ V : ℝ, P * V + (P * V + diff) = V ∧ V = 5500 :=
by
  use 5500
  sorry

end total_votes_l130_13014


namespace louie_mistakes_l130_13030

theorem louie_mistakes (total_items : ℕ) (percentage_correct : ℕ) 
  (h1 : total_items = 25) 
  (h2 : percentage_correct = 80) : 
  total_items - ((percentage_correct / 100) * total_items) = 5 := 
by
  sorry

end louie_mistakes_l130_13030


namespace determine_number_l130_13048

noncomputable def is_valid_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧
  (∃ d1 d2 d3, 
    n = d1 * 100 + d2 * 10 + d3 ∧ 
    (
      (d1 = 5 ∨ d1 = 1 ∨ d1 = 5 ∨ d1 = 2) ∧
      (d2 = 4 ∨ d2 = 4 ∨ d2 = 4) ∧
      (d3 = 3 ∨ d3 = 2 ∨ d3 = 6)
    ) ∧
    (
      (d1 ≠ 1 ∧ d1 ≠ 2 ∧ d1 ≠ 6) ∧
      (d2 ≠ 5 ∧ d2 ≠ 4 ∧ d2 ≠ 6 ∧ d2 ≠ 2) ∧
      (d3 ≠ 5 ∧ d3 ≠ 4 ∧ d3 ≠ 1 ∧ d3 ≠ 2)
    )
  )

theorem determine_number : ∃ n : ℕ, is_valid_number n ∧ n = 163 :=
by 
  existsi 163
  unfold is_valid_number
  sorry

end determine_number_l130_13048


namespace coordinates_of_P_respect_to_symmetric_y_axis_l130_13077

-- Definition of points in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

def symmetric_x_axis (p : Point) : Point :=
  { p with y := -p.y }

def symmetric_y_axis (p : Point) : Point :=
  { p with x := -p.x }

-- The given condition
def P_with_respect_to_symmetric_x_axis := Point.mk (-1) 2

-- The problem statement
theorem coordinates_of_P_respect_to_symmetric_y_axis :
    symmetric_y_axis (symmetric_x_axis P_with_respect_to_symmetric_x_axis) = Point.mk 1 (-2) :=
by
  sorry

end coordinates_of_P_respect_to_symmetric_y_axis_l130_13077


namespace turnip_bag_weight_l130_13016

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l130_13016


namespace max_non_attacking_mammonths_is_20_l130_13023

def mamonth_attacking_diagonal_count (b: board) (m: mamonth): ℕ := 
    sorry -- define the function to count attacking diagonals of a given mammoth on the board

def max_non_attacking_mamonths_board (b: board) : ℕ :=
    sorry -- function to calculate max non-attacking mammonths given a board setup

theorem max_non_attacking_mammonths_is_20 : 
  ∀ (b : board), (max_non_attacking_mamonths_board b) ≤ 20 :=
by
  sorry

end max_non_attacking_mammonths_is_20_l130_13023


namespace quadratic_function_properties_l130_13038

noncomputable def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  (m + 2) * x^(m^2 + m - 4)

theorem quadratic_function_properties :
  (∀ m, (m^2 + m - 4 = 2) → (m = -3 ∨ m = 2))
  ∧ (m = -3 → quadratic_function m 0 = 0) 
  ∧ (m = -3 → ∀ x, x > 0 → quadratic_function m x ≤ quadratic_function m 0 ∧ quadratic_function m x < 0)
  ∧ (m = -3 → ∀ x, x < 0 → quadratic_function m x ≤ quadratic_function m 0 ∧ quadratic_function m x < 0) :=
by
  -- Proof will be supplied here.
  sorry

end quadratic_function_properties_l130_13038


namespace cube_edge_length_l130_13087

theorem cube_edge_length (n_edges : ℕ) (total_length : ℝ) (length_one_edge : ℝ) 
  (h1: n_edges = 12) (h2: total_length = 96) : length_one_edge = 8 :=
by
  sorry

end cube_edge_length_l130_13087


namespace min_value_sin_cos_expr_l130_13033

open Real

theorem min_value_sin_cos_expr (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  ∃ min_val : ℝ, min_val = 3 * sqrt 2 ∧ ∀ β, (0 < β ∧ β < π / 2) → 
    sin β + cos β + (2 * sqrt 2) / sin (β + π / 4) ≥ min_val :=
by
  sorry

end min_value_sin_cos_expr_l130_13033


namespace probability_heads_is_one_eighth_l130_13084

-- Define the probability problem
def probability_heads_penny_nickel_dime (total_coins: ℕ) (successful_events: ℕ) : ℚ :=
  successful_events / total_coins

-- Define the constants
def total_outcomes : ℕ := 2^5  -- Number of possible outcomes with 5 coins
def successful_outcomes : ℕ := 4  -- Number of successful outcomes where penny, nickel, and dime are heads

-- State the theorem to be proven
theorem probability_heads_is_one_eighth : 
  probability_heads_penny_nickel_dime total_outcomes successful_outcomes = 1 / 8 :=
by
  sorry

end probability_heads_is_one_eighth_l130_13084


namespace option_A_is_translation_l130_13092

-- Define what constitutes a translation transformation
def is_translation (description : String) : Prop :=
  description = "Pulling open a drawer"

-- Define each option
def option_A : String := "Pulling open a drawer"
def option_B : String := "Viewing text through a magnifying glass"
def option_C : String := "The movement of the minute hand on a clock"
def option_D : String := "You and the image in a plane mirror"

-- The main theorem stating that option A is the translation transformation
theorem option_A_is_translation : is_translation option_A :=
by
  -- skip the proof, adding sorry
  sorry

end option_A_is_translation_l130_13092


namespace find_smallest_N_l130_13034

def smallest_possible_N (N : ℕ) : Prop :=
  ∃ (W : Fin N → ℝ), 
  (∀ i j, W i ≤ 1.25 * W j ∧ W j ≤ 1.25 * W i) ∧ 
  (∃ (P : Fin 10 → Finset (Fin N)), ∀ i j, i ≤ j →
    P i ≠ ∅ ∧ 
    Finset.sum (P i) W = Finset.sum (P j) W) ∧
  (∃ (V : Fin 11 → Finset (Fin N)), ∀ i j, i ≤ j →
    V i ≠ ∅ ∧ 
    Finset.sum (V i) W = Finset.sum (V j) W)

theorem find_smallest_N : smallest_possible_N 50 :=
sorry

end find_smallest_N_l130_13034


namespace number_of_balls_to_remove_l130_13022

theorem number_of_balls_to_remove:
  ∀ (x : ℕ), 120 - x = (48 : ℕ) / (0.75 : ℝ) → x = 56 :=
by sorry

end number_of_balls_to_remove_l130_13022


namespace sum_of_digits_l130_13036

noncomputable def A : ℕ := 3
noncomputable def B : ℕ := 9
noncomputable def C : ℕ := 2
noncomputable def BC : ℕ := B * 10 + C
noncomputable def ABC : ℕ := A * 100 + B * 10 + C

theorem sum_of_digits (H1: A ≠ 0) (H2: B ≠ 0) (H3: C ≠ 0) (H4: BC + ABC + ABC = 876):
  A + B + C = 14 :=
sorry

end sum_of_digits_l130_13036


namespace smaller_angle_clock_1245_l130_13037

theorem smaller_angle_clock_1245 
  (minute_rate : ℕ → ℝ) 
  (hour_rate : ℕ → ℝ) 
  (time : ℕ) 
  (minute_angle : ℝ) 
  (hour_angle : ℝ) 
  (larger_angle : ℝ) 
  (smaller_angle : ℝ) :
  (minute_rate 1 = 6) →
  (hour_rate 1 = 0.5) →
  (time = 45) →
  (minute_angle = minute_rate 45 * 45) →
  (hour_angle = hour_rate 45 * 45) →
  (larger_angle = |minute_angle - hour_angle|) →
  (smaller_angle = 360 - larger_angle) →
  smaller_angle = 112.5 :=
by
  intros
  sorry

end smaller_angle_clock_1245_l130_13037


namespace not_possible_sum_2017_l130_13061

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem not_possible_sum_2017 (A B : ℕ) (h1 : A + B = 2017) (h2 : sum_of_digits A = 2 * sum_of_digits B) : false := 
sorry

end not_possible_sum_2017_l130_13061


namespace constant_function_of_functional_equation_l130_13043

theorem constant_function_of_functional_equation {f : ℝ → ℝ} (h : ∀ x y : ℝ, 0 < x → 0 < y → f (x + y) = f (x^2 + y^2)) : ∃ c : ℝ, ∀ x : ℝ, 0 < x → f x = c := 
sorry

end constant_function_of_functional_equation_l130_13043


namespace necessary_but_not_sufficient_l130_13005

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a - b > 0 → a^2 - b^2 > 0) ∧ ¬(a^2 - b^2 > 0 → a - b > 0) := by
sorry

end necessary_but_not_sufficient_l130_13005


namespace translate_line_downwards_l130_13080

theorem translate_line_downwards :
  ∀ (x : ℝ), (∀ (y : ℝ), (y = 2 * x + 1) → (y - 2 = 2 * x - 1)) :=
by
  intros x y h
  rw [h]
  sorry

end translate_line_downwards_l130_13080


namespace problem_l130_13044

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 12 / Real.log 6

theorem problem : a > b ∧ b > c := by
  sorry

end problem_l130_13044


namespace total_subjects_is_41_l130_13026

-- Define the number of subjects taken by Monica, Marius, and Millie
def subjects_monica := 10
def subjects_marius := subjects_monica + 4
def subjects_millie := subjects_marius + 3

-- Define the total number of subjects taken by all three
def total_subjects := subjects_monica + subjects_marius + subjects_millie

theorem total_subjects_is_41 : total_subjects = 41 := by
  -- This is where the proof would be, but we only need the statement
  sorry

end total_subjects_is_41_l130_13026


namespace find_x_if_delta_phi_x_eq_3_l130_13085

def delta (x : ℚ) : ℚ := 2 * x + 5
def phi (x : ℚ) : ℚ := 9 * x + 6

theorem find_x_if_delta_phi_x_eq_3 :
  ∃ (x : ℚ), delta (phi x) = 3 ∧ x = -7/9 := by
sorry

end find_x_if_delta_phi_x_eq_3_l130_13085


namespace madeline_part_time_hours_l130_13066

theorem madeline_part_time_hours :
  let hours_in_class := 18
  let days_in_week := 7
  let hours_homework_per_day := 4
  let hours_sleeping_per_day := 8
  let leftover_hours := 46
  let hours_per_day := 24
  let total_hours_per_week := hours_per_day * days_in_week
  let total_homework_hours := hours_homework_per_day * days_in_week
  let total_sleeping_hours := hours_sleeping_per_day * days_in_week
  let total_other_activities := hours_in_class + total_homework_hours + total_sleeping_hours
  let available_hours := total_hours_per_week - total_other_activities
  available_hours - leftover_hours = 20 := by
  sorry

end madeline_part_time_hours_l130_13066


namespace ratio_sum_div_c_l130_13086

theorem ratio_sum_div_c (a b c : ℚ) (h : a / 3 = b / 4 ∧ b / 4 = c / 5) : (a + b + c) / c = 12 / 5 :=
by
  sorry

end ratio_sum_div_c_l130_13086


namespace sufficient_no_x_axis_intersections_l130_13017

/-- Sufficient condition for no x-axis intersections -/
theorem sufficient_no_x_axis_intersections
    (a b c : ℝ)
    (h : a ≠ 0)
    (h_sufficient : b^2 - 4 * a * c < -1) :
    ∀ x : ℝ, ¬(a * x^2 + b * x + c = 0) :=
by
  sorry

end sufficient_no_x_axis_intersections_l130_13017


namespace vasya_numbers_l130_13060

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l130_13060


namespace elevator_time_l130_13095

theorem elevator_time :
  ∀ (floors steps_per_floor steps_per_second extra_time : ℕ) (elevator_time_sec elevator_time_min : ℚ),
    floors = 8 →
    steps_per_floor = 30 →
    steps_per_second = 3 →
    extra_time = 30 →
    elevator_time_sec = ((floors * steps_per_floor) / steps_per_second) - extra_time →
    elevator_time_min = elevator_time_sec / 60 →
    elevator_time_min = 0.833 :=
by
  intros floors steps_per_floor steps_per_second extra_time elevator_time_sec elevator_time_min
  intros h_floors h_steps_per_floor h_steps_per_second h_extra_time h_elevator_time_sec h_elevator_time_min
  rw [h_floors, h_steps_per_floor, h_steps_per_second, h_extra_time] at *
  sorry

end elevator_time_l130_13095


namespace two_point_two_five_as_fraction_l130_13076

theorem two_point_two_five_as_fraction : (2.25 : ℚ) = 9 / 4 := 
by 
  -- Proof steps would be added here
  sorry

end two_point_two_five_as_fraction_l130_13076


namespace train_crossing_time_l130_13058

def train_length : ℕ := 1000
def train_speed_km_per_h : ℕ := 18
def train_speed_m_per_s := train_speed_km_per_h * 1000 / 3600

theorem train_crossing_time :
  train_length / train_speed_m_per_s = 200 := by
sorry

end train_crossing_time_l130_13058


namespace no_perfect_squares_in_sequence_l130_13064

theorem no_perfect_squares_in_sequence (x : ℕ → ℤ) (h₀ : x 0 = 1) (h₁ : x 1 = 3)
  (h_rec : ∀ n : ℕ, x (n + 1) = 6 * x n - x (n - 1)) 
  : ∀ n : ℕ, ¬ ∃ k : ℤ, x n = k * k := 
sorry

end no_perfect_squares_in_sequence_l130_13064


namespace count_four_digit_multiples_of_5_l130_13028

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l130_13028


namespace interest_rate_per_annum_l130_13096

theorem interest_rate_per_annum (P T : ℝ) (r : ℝ) 
  (h1 : P = 15000) 
  (h2 : T = 2)
  (h3 : P * (1 + r)^T - P - (P * r * T) = 150) : 
  r = 0.1 :=
by
  sorry

end interest_rate_per_annum_l130_13096


namespace find_m_l130_13082

theorem find_m (m : ℕ) :
  (∀ x : ℝ, -2 * x ^ 2 + 5 * x - 2 <= 9 / m) →
  m = 8 :=
sorry

end find_m_l130_13082


namespace match_processes_count_l130_13069

-- Define the sets and the number of interleavings
def team_size : ℕ := 4 -- Each team has 4 players

-- Define the problem statement
theorem match_processes_count :
  (Nat.choose (2 * team_size) team_size) = 70 := by
  -- This is where the proof would go, but we'll use sorry as specified
  sorry

end match_processes_count_l130_13069


namespace measure_of_angle_ABC_l130_13027

-- Define the angles involved and their respective measures
def angle_CBD : ℝ := 90 -- Given that angle CBD is a right angle
def angle_sum : ℝ := 160 -- Sum of the angles around point B
def angle_ABD : ℝ := 50 -- Given angle ABD

-- Define angle ABC to be determined
def angle_ABC : ℝ := angle_sum - (angle_ABD + angle_CBD)

-- Define the statement
theorem measure_of_angle_ABC :
  angle_ABC = 20 :=
by 
  -- Calculations omitted
  sorry

end measure_of_angle_ABC_l130_13027


namespace arithmetic_sequence_terms_l130_13041

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S : ℝ) 
  (h1 : a 0 + a 1 + a 2 = 4) 
  (h2 : a (n-3) + a (n-2) + a (n-1) = 7) 
  (h3 : (n * (a 0 + a (n-1)) / 2) = 22) : 
  n = 12 :=
sorry

end arithmetic_sequence_terms_l130_13041


namespace line_through_parabola_intersects_vertex_l130_13007

theorem line_through_parabola_intersects_vertex (y x k : ℝ) :
  (y = 6 * x ∨ 6 * x - 5 * y - 24 = 0) ∧ 
  (∃ P Q : ℝ × ℝ, (P.1)^2 = 4 * P.2 ∧ (Q.1)^2 = 4 * Q.2 ∧ 
   (P = (0, 0) ∨ Q = (0, 0)) ∧ 
   (y = 6 * x ∨ 6 * x - 5 * y - 24 = 0)) := sorry

end line_through_parabola_intersects_vertex_l130_13007


namespace eccentricity_squared_l130_13053

-- Define the hyperbola and its properties
variables (a b c e : ℝ) (x₁ y₁ x₂ y₂ : ℝ)

-- Define the hyperbola equation and conditions
def hyperbola_eq (a b x y : ℝ) := (x^2)/(a^2) - (y^2)/(b^2) = 1

def midpoint_eq (x₁ y₁ x₂ y₂ : ℝ) := x₁ + x₂ = -4 ∧ y₁ + y₂ = 2

def slope_eq (a b c : ℝ) := -b / c = (b^2 * (-4)) / (a^2 * 2)

-- Define the proof
theorem eccentricity_squared :
  a > 0 → b > 0 → hyperbola_eq a b x₁ y₁ → hyperbola_eq a b x₂ y₂ → midpoint_eq x₁ y₁ x₂ y₂ →
  slope_eq a b c → c^2 = a^2 + b^2 → (e = c / a) → e^2 = (Real.sqrt 2 + 1) / 2 :=
by
  intro ha hb h1 h2 h3 h4 h5 he
  sorry

end eccentricity_squared_l130_13053


namespace same_color_probability_correct_l130_13013

noncomputable def prob_same_color (green red blue : ℕ) : ℚ :=
  let total := green + red + blue
  (green / total) * (green / total) +
  (red / total) * (red / total) +
  (blue / total) * (blue / total)

theorem same_color_probability_correct :
  prob_same_color 5 7 3 = 83 / 225 :=
by
  sorry

end same_color_probability_correct_l130_13013


namespace mrs_hilt_additional_rocks_l130_13067

-- Definitions from the conditions
def total_rocks : ℕ := 125
def rocks_she_has : ℕ := 64
def additional_rocks_needed : ℕ := total_rocks - rocks_she_has

-- The theorem to prove the question equals the answer given the conditions
theorem mrs_hilt_additional_rocks : additional_rocks_needed = 61 := 
by
  sorry

end mrs_hilt_additional_rocks_l130_13067


namespace perimeter_difference_l130_13079

-- Define the height of the screen
def height_of_screen : ℕ := 100

-- Define the side length of the square paper
def side_of_square_paper : ℕ := 20

-- Define the perimeter of the square paper
def perimeter_of_paper : ℕ := 4 * side_of_square_paper

-- Prove the difference between the height of the screen and the perimeter of the paper
theorem perimeter_difference : height_of_screen - perimeter_of_paper = 20 := by
  -- Sorry is used here to skip the actual proof
  sorry

end perimeter_difference_l130_13079


namespace disproving_proposition_l130_13055

theorem disproving_proposition : ∃ (angle1 angle2 : ℝ), angle1 = angle2 ∧ angle1 + angle2 = 90 :=
by
  sorry

end disproving_proposition_l130_13055


namespace arithmetic_to_geometric_seq_l130_13050

theorem arithmetic_to_geometric_seq
  (d a : ℕ) 
  (h1 : d ≠ 0) 
  (a_n : ℕ → ℕ)
  (h2 : ∀ n, a_n n = a + (n - 1) * d)
  (h3 : (a + 2 * d) * (a + 2 * d) = a * (a + 8 * d))
  : (a_n 2 + a_n 4 + a_n 10) / (a_n 1 + a_n 3 + a_n 9) = 16 / 13 :=
by
  sorry

end arithmetic_to_geometric_seq_l130_13050


namespace price_of_magic_card_deck_l130_13012

def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3
def total_earnings : ℕ := 4
def decks_sold := initial_decks - remaining_decks
def price_per_deck := total_earnings / decks_sold

theorem price_of_magic_card_deck : price_per_deck = 2 := by
  sorry

end price_of_magic_card_deck_l130_13012


namespace log_sum_zero_l130_13051

theorem log_sum_zero (a b c N : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_N : 0 < N) (h_neq_N : N ≠ 1) (h_geom_mean : b^2 = a * c) : 
  1 / Real.logb a N - 2 / Real.logb b N + 1 / Real.logb c N = 0 :=
  by
  sorry

end log_sum_zero_l130_13051


namespace shauna_lowest_score_l130_13045

theorem shauna_lowest_score :
  ∀ (scores : List ℕ) (score1 score2 score3 : ℕ), 
    scores = [score1, score2, score3] → 
    score1 = 82 →
    score2 = 88 →
    score3 = 93 →
    (∃ (s4 s5 : ℕ), s4 + s5 = 162 ∧ s4 ≤ 100 ∧ s5 ≤ 100) ∧
    score1 + score2 + score3 + s4 + s5 = 425 →
    min s4 s5 = 62 := 
by 
  sorry

end shauna_lowest_score_l130_13045


namespace ladder_slip_l130_13074

theorem ladder_slip (l : ℝ) (d1 d2 : ℝ) (h1 h2 : ℝ) :
  l = 30 → d1 = 8 → h1^2 + d1^2 = l^2 → h2 = h1 - 4 → 
  (h2^2 + (d1 + d2)^2 = l^2) → d2 = 2 :=
by
  intros h_l h_d1 h_h1_eq h_h2 h2_eq_l   
  sorry

end ladder_slip_l130_13074


namespace polynomial_coefficient_sum_l130_13019

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (1 - 2 * x) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 →
  a₀ + a₁ + a₃ = -39 :=
by
  sorry

end polynomial_coefficient_sum_l130_13019


namespace find_second_discount_l130_13040

theorem find_second_discount 
    (list_price : ℝ)
    (final_price : ℝ)
    (first_discount : ℝ)
    (second_discount : ℝ)
    (h₁ : list_price = 65)
    (h₂ : final_price = 57.33)
    (h₃ : first_discount = 0.10)
    (h₄ : (list_price - (first_discount * list_price)) = 58.5)
    (h₅ : final_price = 58.5 - (second_discount * 58.5)) :
    second_discount = 0.02 := 
by
  sorry

end find_second_discount_l130_13040


namespace simplify_expression_eq_square_l130_13002

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end simplify_expression_eq_square_l130_13002


namespace complex_expression_calculation_l130_13057

noncomputable def complex_i := Complex.I -- Define the imaginary unit i

theorem complex_expression_calculation : complex_i * (1 - complex_i)^2 = 2 := by
  sorry

end complex_expression_calculation_l130_13057
