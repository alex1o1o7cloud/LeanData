import Mathlib

namespace determine_h_l229_22904

open Polynomial

noncomputable def f (x : ℚ) : ℚ := x^2

theorem determine_h (h : ℚ → ℚ) : 
  (∀ x, f (h x) = 9 * x^2 + 6 * x + 1) ↔ 
  (∀ x, h x = 3 * x + 1 ∨ h x = - (3 * x + 1)) :=
by
  sorry

end determine_h_l229_22904


namespace diff_square_mental_math_l229_22939

theorem diff_square_mental_math :
  75 ^ 2 - 45 ^ 2 = 3600 :=
by
  -- The proof would go here
  sorry

end diff_square_mental_math_l229_22939


namespace find_x_values_l229_22998

theorem find_x_values (x : ℝ) (h : x + 60 / (x - 3) = -12) : x = -3 ∨ x = -6 :=
sorry

end find_x_values_l229_22998


namespace equal_real_roots_iff_c_is_nine_l229_22985

theorem equal_real_roots_iff_c_is_nine (c : ℝ) : (∃ x : ℝ, x^2 + 6 * x + c = 0 ∧ ∃ Δ, Δ = 6^2 - 4 * 1 * c ∧ Δ = 0) ↔ c = 9 :=
by
  sorry

end equal_real_roots_iff_c_is_nine_l229_22985


namespace committee_of_4_from_10_eq_210_l229_22935

theorem committee_of_4_from_10_eq_210 :
  (Nat.choose 10 4) = 210 :=
by
  sorry

end committee_of_4_from_10_eq_210_l229_22935


namespace gcd_of_54000_and_36000_l229_22966

theorem gcd_of_54000_and_36000 : Nat.gcd 54000 36000 = 18000 := 
by sorry

end gcd_of_54000_and_36000_l229_22966


namespace D_times_C_eq_l229_22961

-- Define the matrices C and D
variable (C D : Matrix (Fin 2) (Fin 2) ℚ)

-- Add the conditions
axiom h1 : C * D = C + D
axiom h2 : C * D = ![![15/2, 9/2], ![-6/2, 12/2]]

-- Define the goal
theorem D_times_C_eq : D * C = ![![15/2, 9/2], ![-6/2, 12/2]] :=
sorry

end D_times_C_eq_l229_22961


namespace geometric_sum_common_ratio_l229_22955

theorem geometric_sum_common_ratio (a₁ a₂ : ℕ) (q : ℕ) (S₃ : ℕ)
  (h1 : S₃ = a₁ + 3 * a₂)
  (h2: S₃ = a₁ * (1 + q + q^2)) :
  q = 2 :=
by
  sorry

end geometric_sum_common_ratio_l229_22955


namespace cost_of_advanced_purchase_ticket_l229_22945

theorem cost_of_advanced_purchase_ticket
  (x : ℝ)
  (door_cost : ℝ := 14)
  (total_tickets : ℕ := 140)
  (total_money : ℝ := 1720)
  (advanced_tickets_sold : ℕ := 100)
  (door_tickets_sold : ℕ := total_tickets - advanced_tickets_sold)
  (advanced_revenue : ℝ := advanced_tickets_sold * x)
  (door_revenue : ℝ := door_tickets_sold * door_cost)
  (total_revenue : ℝ := advanced_revenue + door_revenue) :
  total_revenue = total_money → x = 11.60 :=
by
  intro h
  sorry

end cost_of_advanced_purchase_ticket_l229_22945


namespace length_of_faster_train_l229_22983

/-- Define the speeds of the trains in kmph -/
def speed_faster_train := 180 -- in kmph
def speed_slower_train := 90  -- in kmph

/-- Convert speeds to m/s -/
def kmph_to_mps (speed : ℕ) : ℕ := speed * 5 / 18

/-- Define the relative speed in m/s -/
def relative_speed := kmph_to_mps speed_faster_train - kmph_to_mps speed_slower_train

/-- Define the time it takes for the faster train to cross the man in seconds -/
def crossing_time := 15 -- in seconds

/-- Define the length of the train calculation in meters -/
noncomputable def length_faster_train := relative_speed * crossing_time

theorem length_of_faster_train :
  length_faster_train = 375 :=
by
  sorry

end length_of_faster_train_l229_22983


namespace shorter_piece_length_l229_22908

theorem shorter_piece_length (x : ℝ) (h : 3 * x = 60) : x = 20 :=
by
  sorry

end shorter_piece_length_l229_22908


namespace action_figure_cost_l229_22981

def initial_figures : ℕ := 7
def total_figures_needed : ℕ := 16
def total_cost : ℕ := 72

theorem action_figure_cost :
  total_cost / (total_figures_needed - initial_figures) = 8 := by
  sorry

end action_figure_cost_l229_22981


namespace eq_sqrt_pattern_l229_22922

theorem eq_sqrt_pattern (a t : ℝ) (ha : a = 6) (ht : t = a^2 - 1) (h_pos : 0 < a ∧ 0 < t) :
  a + t = 41 := by
  sorry

end eq_sqrt_pattern_l229_22922


namespace time_to_reach_julia_via_lee_l229_22927

theorem time_to_reach_julia_via_lee (d1 d2 d3 : ℕ) (t1 t2 : ℕ) :
  d1 = 2 → 
  t1 = 6 → 
  d3 = 3 → 
  (∀ v, v = d1 / t1) → 
  t2 = d3 / v → 
  t2 = 9 :=
by
  intros h1 h2 h3 hv ht2
  sorry

end time_to_reach_julia_via_lee_l229_22927


namespace central_cell_value_l229_22916

theorem central_cell_value (a b c d e f g h i : ℝ)
  (h_row1 : a * b * c = 10)
  (h_row2 : d * e * f = 10)
  (h_row3 : g * h * i = 10)
  (h_col1 : a * d * g = 10)
  (h_col2 : b * e * h = 10)
  (h_col3 : c * f * i = 10)
  (h_block1 : a * b * d * e = 3)
  (h_block2 : b * c * e * f = 3)
  (h_block3 : d * e * g * h = 3)
  (h_block4 : e * f * h * i = 3) :
  e = 0.00081 :=
sorry

end central_cell_value_l229_22916


namespace cost_of_each_pair_of_socks_eq_2_l229_22971

-- Definitions and conditions
def cost_of_shoes : ℤ := 74
def cost_of_bag : ℤ := 42
def paid_amount : ℤ := 118
def discount_rate : ℚ := 0.10

-- Given the conditions
def total_cost (x : ℚ) : ℚ := cost_of_shoes + 2 * x + cost_of_bag
def discount (x : ℚ) : ℚ := if total_cost x > 100 then discount_rate * (total_cost x - 100) else 0
def total_cost_after_discount (x : ℚ) : ℚ := total_cost x - discount x

-- Theorem to prove
theorem cost_of_each_pair_of_socks_eq_2 : 
  ∃ x : ℚ, total_cost_after_discount x = paid_amount ∧ 2 * x = 4 :=
by
  sorry

end cost_of_each_pair_of_socks_eq_2_l229_22971


namespace car_rental_cost_per_mile_l229_22982

theorem car_rental_cost_per_mile
    (daily_rental_cost : ℕ)
    (daily_budget : ℕ)
    (miles_limit : ℕ)
    (cost_per_mile : ℕ) :
    daily_rental_cost = 30 →
    daily_budget = 76 →
    miles_limit = 200 →
    cost_per_mile = (daily_budget - daily_rental_cost) * 100 / miles_limit →
    cost_per_mile = 23 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end car_rental_cost_per_mile_l229_22982


namespace polynomial_roots_identity_l229_22925

variables {c d : ℂ}

theorem polynomial_roots_identity (hc : c + d = 5) (hd : c * d = 6) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 :=
by {
  sorry
}

end polynomial_roots_identity_l229_22925


namespace min_value_l229_22974

theorem min_value (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 1) :
  (a + 1)^2 + 4 * b^2 + 9 * c^2 ≥ 144 / 49 :=
sorry

end min_value_l229_22974


namespace find_g_two_fifths_l229_22968

noncomputable def g : ℝ → ℝ :=
sorry -- The function g(x) is not explicitly defined.

theorem find_g_two_fifths :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g x = 0 → g 0 = 0) ∧
  (∀ x y, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 5) = g x / 3)
  → g (2 / 5) = 1 / 3 :=
sorry

end find_g_two_fifths_l229_22968


namespace find_x_if_perpendicular_l229_22977

-- Given definitions and conditions
def a : ℝ × ℝ := (-5, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Statement to be proved
theorem find_x_if_perpendicular (x : ℝ) :
  (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = 10 :=
by
  sorry

end find_x_if_perpendicular_l229_22977


namespace paperclips_in_64_volume_box_l229_22918

def volume_16 : ℝ := 16
def volume_32 : ℝ := 32
def volume_64 : ℝ := 64
def paperclips_50 : ℝ := 50
def paperclips_100 : ℝ := 100

theorem paperclips_in_64_volume_box :
  ∃ (k p : ℝ), 
  (paperclips_50 = k * volume_16^p) ∧ 
  (paperclips_100 = k * volume_32^p) ∧ 
  (200 = k * volume_64^p) :=
by
  sorry

end paperclips_in_64_volume_box_l229_22918


namespace fraction_representation_of_2_375_l229_22940

theorem fraction_representation_of_2_375 : 2.375 = 19 / 8 := by
  sorry

end fraction_representation_of_2_375_l229_22940


namespace range_of_function_l229_22953

open Set

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_of_function (S : Set ℝ) : 
    S = {y : ℝ | ∃ x : ℝ, x ≥ 1 ∧ y = 2 + log_base_2 x} 
    ↔ S = {y : ℝ | y ≥ 2} :=
by 
  sorry

end range_of_function_l229_22953


namespace part1_part2_l229_22933

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1 (a : ℝ) : (set_A ∪ set_B a = set_A ∩ set_B a) → a = 1 :=
sorry

theorem part2 (a : ℝ) : (set_A ∪ set_B a = set_A) → (a ≤ -1 ∨ a = 1) :=
sorry

end part1_part2_l229_22933


namespace initially_marked_points_l229_22907

theorem initially_marked_points (k : ℕ) (h : 4 * k - 3 = 101) : k = 26 :=
by
  sorry

end initially_marked_points_l229_22907


namespace area_of_park_l229_22926

theorem area_of_park (x : ℕ) (rate_per_meter : ℝ) (total_cost : ℝ)
  (ratio_len_wid : ℕ × ℕ)
  (h_ratio : ratio_len_wid = (3, 2))
  (h_cost : total_cost = 140)
  (unit_rate : rate_per_meter = 0.50)
  (h_perimeter : 10 * x * rate_per_meter = total_cost) :
  6 * x^2 = 4704 :=
by
  sorry

end area_of_park_l229_22926


namespace divisors_count_30_l229_22906

theorem divisors_count_30 : 
  (∃ n : ℤ, n > 1 ∧ 30 % n = 0) 
  → 
  (∃ k : ℕ, k = 14) :=
by
  sorry

end divisors_count_30_l229_22906


namespace solve_system_eq_l229_22960

theorem solve_system_eq (x y z t : ℕ) : 
  ((x^2 + t^2) * (z^2 + y^2) = 50) ↔
    (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨
    (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨
    (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨
    (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
by 
  sorry

end solve_system_eq_l229_22960


namespace unique_solution_k_l229_22929

theorem unique_solution_k (k : ℚ) : (∀ x : ℚ, x ≠ -2 → (x + 3)/(k*x - 2) = x) ↔ k = -3/4 :=
sorry

end unique_solution_k_l229_22929


namespace boat_distance_downstream_l229_22909

theorem boat_distance_downstream
  (speed_boat : ℕ)
  (speed_stream : ℕ)
  (time_downstream : ℕ)
  (h1 : speed_boat = 22)
  (h2 : speed_stream = 5)
  (h3 : time_downstream = 8) :
  speed_boat + speed_stream * time_downstream = 216 :=
by
  sorry

end boat_distance_downstream_l229_22909


namespace egg_count_l229_22942

theorem egg_count :
  ∃ x : ℕ, 
    (∀ e1 e10 e100 : ℤ, 
      (e1 = 1 ∨ e1 = -1) →
      (e10 = 10 ∨ e10 = -10) →
      (e100 = 100 ∨ e100 = -100) →
      7 * x + e1 + e10 + e100 = 3162) → 
    x = 439 :=
by 
  sorry

end egg_count_l229_22942


namespace shop_dimension_is_100_l229_22964

-- Given conditions
def monthly_rent : ℕ := 1300
def annual_rent_per_sqft : ℕ := 156

-- Define annual rent
def annual_rent : ℕ := monthly_rent * 12

-- Define dimension to prove
def dimension_of_shop : ℕ := annual_rent / annual_rent_per_sqft

-- The theorem statement
theorem shop_dimension_is_100 :
  dimension_of_shop = 100 :=
by
  sorry

end shop_dimension_is_100_l229_22964


namespace find_number_l229_22993

theorem find_number (x : ℝ) : 2.75 + 0.003 + x = 2.911 -> x = 0.158 := 
by
  intros h
  sorry

end find_number_l229_22993


namespace mean_of_remaining_three_l229_22950

theorem mean_of_remaining_three (a b c : ℝ) (h₁ : (a + b + c + 105) / 4 = 93) : (a + b + c) / 3 = 89 :=
  sorry

end mean_of_remaining_three_l229_22950


namespace sum_of_digits_of_N_l229_22944

theorem sum_of_digits_of_N :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ N.digits 10 = [7, 7] :=
by
  sorry

end sum_of_digits_of_N_l229_22944


namespace eggs_left_for_sunny_side_up_l229_22921

-- Given conditions:
def ordered_dozen_eggs : ℕ := 3 * 12
def eggs_used_for_crepes (total_eggs : ℕ) : ℕ := total_eggs * 1 / 4
def eggs_after_crepes (total_eggs : ℕ) (used_for_crepes : ℕ) : ℕ := total_eggs - used_for_crepes
def eggs_used_for_cupcakes (remaining_eggs : ℕ) : ℕ := remaining_eggs * 2 / 3
def eggs_left (remaining_eggs : ℕ) (used_for_cupcakes : ℕ) : ℕ := remaining_eggs - used_for_cupcakes

-- Proposition:
theorem eggs_left_for_sunny_side_up : 
  eggs_left (eggs_after_crepes ordered_dozen_eggs (eggs_used_for_crepes ordered_dozen_eggs)) 
            (eggs_used_for_cupcakes (eggs_after_crepes ordered_dozen_eggs (eggs_used_for_crepes ordered_dozen_eggs))) = 9 :=
sorry

end eggs_left_for_sunny_side_up_l229_22921


namespace nelly_payment_is_correct_l229_22972

-- Given definitions and conditions
def joes_bid : ℕ := 160000
def additional_amount : ℕ := 2000

-- Nelly's total payment
def nellys_payment : ℕ := (3 * joes_bid) + additional_amount

-- The proof statement we need to prove that Nelly's payment equals 482000 dollars
theorem nelly_payment_is_correct : nellys_payment = 482000 :=
by
  -- This is a placeholder for the actual proof.
  -- You can fill in the formal proof here.
  sorry

end nelly_payment_is_correct_l229_22972


namespace number_of_puppies_with_4_spots_is_3_l229_22980

noncomputable def total_puppies : Nat := 10
noncomputable def puppies_with_5_spots : Nat := 6
noncomputable def puppies_with_2_spots : Nat := 1
noncomputable def puppies_with_4_spots : Nat := total_puppies - puppies_with_5_spots - puppies_with_2_spots

theorem number_of_puppies_with_4_spots_is_3 :
  puppies_with_4_spots = 3 := 
sorry

end number_of_puppies_with_4_spots_is_3_l229_22980


namespace solve_quadratic_equation_l229_22956

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2 := by
sorry

end solve_quadratic_equation_l229_22956


namespace dalton_needs_more_money_l229_22923

-- Definitions based on the conditions
def jumpRopeCost : ℕ := 7
def boardGameCost : ℕ := 12
def ballCost : ℕ := 4
def savedAllowance : ℕ := 6
def moneyFromUncle : ℕ := 13

-- Computation of how much more money is needed
theorem dalton_needs_more_money : 
  let totalCost := jumpRopeCost + boardGameCost + ballCost
  let totalMoney := savedAllowance + moneyFromUncle
  totalCost - totalMoney = 4 := 
by 
  let totalCost := jumpRopeCost + boardGameCost + ballCost
  let totalMoney := savedAllowance + moneyFromUncle
  have h1 : totalCost = 23 := by rfl
  have h2 : totalMoney = 19 := by rfl
  calc
    totalCost - totalMoney = 23 - 19 := by rw [h1, h2]
    _ = 4 := by rfl

end dalton_needs_more_money_l229_22923


namespace solution_set_empty_l229_22999

variable (m x : ℝ)
axiom no_solution (h : (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) : (1 + m = 0)

theorem solution_set_empty :
  (∀ x, (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) ≠ -1) ↔ m = -1 := by
  sorry

end solution_set_empty_l229_22999


namespace find_x_y_sum_of_squares_l229_22931

theorem find_x_y_sum_of_squares :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (xy + x + y = 47) ∧ (x^2 * y + x * y^2 = 506) ∧ (x^2 + y^2 = 101) :=
by {
  sorry
}

end find_x_y_sum_of_squares_l229_22931


namespace daily_earnings_r_l229_22932

theorem daily_earnings_r (p q r s : ℝ)
  (h1 : p + q + r + s = 300)
  (h2 : p + r = 120)
  (h3 : q + r = 130)
  (h4 : s + r = 200)
  (h5 : p + s = 116.67) : 
  r = 75 :=
by
  sorry

end daily_earnings_r_l229_22932


namespace product_closest_to_106_l229_22911

theorem product_closest_to_106 :
  let product := (2.1 : ℝ) * (50.8 - 0.45)
  abs (product - 106) < abs (product - 105) ∧
  abs (product - 106) < abs (product - 107) ∧
  abs (product - 106) < abs (product - 108) ∧
  abs (product - 106) < abs (product - 110) :=
by
  sorry

end product_closest_to_106_l229_22911


namespace sum_of_squares_of_ages_l229_22976

theorem sum_of_squares_of_ages {a b c : ℕ} (h1 : 5 * a + b = 3 * c) (h2 : 3 * c^2 = 2 * a^2 + b^2) 
  (relatively_prime : Nat.gcd (Nat.gcd a b) c = 1) : 
  a^2 + b^2 + c^2 = 374 :=
by
  sorry

end sum_of_squares_of_ages_l229_22976


namespace geometric_sequence_problem_l229_22978

noncomputable def q : ℝ := 1 + Real.sqrt 2

theorem geometric_sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = (q : ℝ) * a n)
  (h_cond : a 2 = a 0 + 2 * a 1) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := 
sorry

end geometric_sequence_problem_l229_22978


namespace remaining_speed_20_kmph_l229_22975

theorem remaining_speed_20_kmph
  (D T : ℝ)
  (H1 : (2/3 * D) / (1/3 * T) = 80)
  (H2 : T = D / 40) :
  (D / 3) / (2/3 * T) = 20 :=
by 
  sorry

end remaining_speed_20_kmph_l229_22975


namespace additional_time_to_walk_1_mile_l229_22963

open Real

noncomputable def additional_time_per_mile
  (distance_child : ℝ) (time_child : ℝ)
  (distance_elderly : ℝ) (time_elderly : ℝ)
  : ℝ :=
  let speed_child := distance_child / time_child
  let time_per_mile_child := (time_child * 60) / distance_child
  let speed_elderly := distance_elderly / time_elderly
  let time_per_mile_elderly := (time_elderly * 60) / distance_elderly
  time_per_mile_elderly - time_per_mile_child

theorem additional_time_to_walk_1_mile
  (h1 : 15 = 15) (h2 : 3.5 = 3.5)
  (h3 : 10 = 10) (h4 : 4 = 4)
  : additional_time_per_mile 15 3.5 10 4 = 10 :=
  by
    sorry

end additional_time_to_walk_1_mile_l229_22963


namespace factor_expression_l229_22986

theorem factor_expression (x : ℝ) : 2 * x * (x + 3) + (x + 3) = (2 * x + 1) * (x + 3) :=
by
  sorry

end factor_expression_l229_22986


namespace intersection_or_parallel_lines_l229_22943

structure Triangle (Point : Type) :=
  (A B C : Point)

structure Plane (Point : Type) :=
  (P1 P2 P3 P4 : Point)

variables {Point : Type}
variables (triABC triA1B1C1 : Triangle Point)
variables (plane1 plane2 plane3 : Plane Point)

-- Intersection conditions
variable (AB_intersects_A1B1 : (triABC.A, triABC.B) = (triA1B1C1.A, triA1B1C1.B))
variable (BC_intersects_B1C1 : (triABC.B, triABC.C) = (triA1B1C1.B, triA1B1C1.C))
variable (CA_intersects_C1A1 : (triABC.C, triABC.A) = (triA1B1C1.C, triA1B1C1.A))

theorem intersection_or_parallel_lines :
  ∃ P : Point, (
    (∃ A1 : Point, (triABC.A, A1) = (P, P)) ∧
    (∃ B1 : Point, (triABC.B, B1) = (P, P)) ∧
    (∃ C1 : Point, (triABC.C, C1) = (P, P))
  ) ∨ (
    (∃ d1 d2 d3 : Point, 
      (∀ A1 B1 C1 : Point,
        (triABC.A, A1) = (d1, d1) ∧ 
        (triABC.B, B1) = (d2, d2) ∧ 
        (triABC.C, C1) = (d3, d3)
      )
    )
  ) := by
  sorry

end intersection_or_parallel_lines_l229_22943


namespace period_of_f_is_4_and_f_2pow_n_zero_l229_22997

noncomputable def f : ℝ → ℝ := sorry

variables (hf_diff : differentiable ℝ f)
          (hf_nonzero : ∃ x, f x ≠ 0)
          (hf_odd_2 : ∀ x, f (x + 2) = -f (-x - 2))
          (hf_even_2x1 : ∀ x, f (2 * x + 1) = f (-(2 * x + 1)))

theorem period_of_f_is_4_and_f_2pow_n_zero (n : ℕ) (hn : 0 < n) :
  (∀ x, f (x + 4) = f x) ∧ f (2^n) = 0 :=
sorry

end period_of_f_is_4_and_f_2pow_n_zero_l229_22997


namespace necessary_but_not_sufficient_l229_22910

variables {R : Type*} [Field R] (a b c : R)

def condition1 : Prop := (a / b) = (b / c)
def condition2 : Prop := b^2 = a * c

theorem necessary_but_not_sufficient :
  (condition1 a b c → condition2 a b c) ∧ ¬ (condition2 a b c → condition1 a b c) :=
by
  sorry

end necessary_but_not_sufficient_l229_22910


namespace earthquake_relief_team_selection_l229_22990

theorem earthquake_relief_team_selection : 
    ∃ (ways : ℕ), ways = 590 ∧ 
      ∃ (orthopedic neurosurgeon internist : ℕ), 
      orthopedic + neurosurgeon + internist = 5 ∧ 
      1 ≤ orthopedic ∧ 1 ≤ neurosurgeon ∧ 1 ≤ internist ∧
      orthopedic ≤ 3 ∧ neurosurgeon ≤ 4 ∧ internist ≤ 5 := 
  sorry

end earthquake_relief_team_selection_l229_22990


namespace problem_equivalent_l229_22959

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l229_22959


namespace strawberries_taken_out_l229_22965

theorem strawberries_taken_out : 
  ∀ (initial_total_strawberries buckets strawberries_left_per_bucket : ℕ),
  initial_total_strawberries = 300 → 
  buckets = 5 → 
  strawberries_left_per_bucket = 40 → 
  (initial_total_strawberries / buckets - strawberries_left_per_bucket = 20) :=
by
  intros initial_total_strawberries buckets strawberries_left_per_bucket h1 h2 h3
  sorry

end strawberries_taken_out_l229_22965


namespace tangent_line_eq_l229_22919

-- Definitions for the conditions
def curve (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x

def derivative_curve (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

-- Define the problem as a theorem statement
theorem tangent_line_eq (L : ℝ → ℝ) (hL : ∀ x, L x = 2 * x ∨ L x = - x/4) :
  (∀ x, x = 0 → L x = 0) →
  (∀ x x0, L x = curve x → derivative_curve x0 = derivative_curve 0 → x0 = 0 ∨ x0 = 3/2) →
  (L x = 2 * x - curve x ∨ L x = 4 * x + curve x) :=
by
  sorry

end tangent_line_eq_l229_22919


namespace div_by_27_l229_22912

theorem div_by_27 (n : ℕ) : 27 ∣ (10^n + 18 * n - 1) :=
sorry

end div_by_27_l229_22912


namespace ali_less_nada_l229_22995

variable (Ali Nada John : ℕ)

theorem ali_less_nada
  (h_total : Ali + Nada + John = 67)
  (h_john_nada : John = 4 * Nada)
  (h_john : John = 48) :
  Nada - Ali = 5 :=
by
  sorry

end ali_less_nada_l229_22995


namespace cubic_inequality_l229_22941

theorem cubic_inequality (x p q : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 := 
  sorry

end cubic_inequality_l229_22941


namespace solve_log_eq_l229_22996

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_log_eq (x : ℝ) (hx1 : x + 1 > 0) (hx2 : x - 1 > 0) :
  log_base (x + 1) (x^3 - 9 * x + 8) * log_base (x - 1) (x + 1) = 3 ↔ x = 3 := by
  sorry

end solve_log_eq_l229_22996


namespace snacks_in_3h40m_l229_22973

def minutes_in_hours (hours : ℕ) : ℕ := hours * 60

def snacks_in_time (total_minutes : ℕ) (snack_interval : ℕ) : ℕ := total_minutes / snack_interval

theorem snacks_in_3h40m : snacks_in_time (minutes_in_hours 3 + 40) 20 = 11 :=
by
  sorry

end snacks_in_3h40m_l229_22973


namespace S_div_T_is_one_half_l229_22915

def T (x y z : ℝ) := x >= 0 ∧ y >= 0 ∧ z >= 0 ∧ x + y + z = 1

def supports (a b c x y z : ℝ) := 
  (x >= a ∧ y >= b ∧ z < c) ∨ 
  (x >= a ∧ z >= c ∧ y < b) ∨ 
  (y >= b ∧ z >= c ∧ x < a)

def S (x y z : ℝ) := T x y z ∧ supports (1/4) (1/4) (1/2) x y z

theorem S_div_T_is_one_half :
  let area_T := 1 -- Normalizing since area of T is in fact √3 / 2 but we care about ratios
  let area_S := 1/2 * area_T -- Given by the problem solution
  area_S / area_T = 1/2 := 
sorry

end S_div_T_is_one_half_l229_22915


namespace find_number_l229_22967

theorem find_number (x : ℤ) (h : 300 + 8 * x = 340) : x = 5 := by
  sorry

end find_number_l229_22967


namespace maximum_candies_karlson_l229_22930

theorem maximum_candies_karlson (n : ℕ) (h_n : n = 40) :
  ∃ k, k = 780 :=
by
  sorry

end maximum_candies_karlson_l229_22930


namespace find_radius_l229_22905

theorem find_radius 
  (r : ℝ)
  (h1 : ∀ (x y : ℝ), ((x - r) ^ 2 + y ^ 2 = r ^ 2) → (4 * x ^ 2 + 9 * y ^ 2 = 36)) 
  (h2 : (4 * r ^ 2 + 9 * 0 ^ 2 = 36)) 
  (h3 : ∃ r : ℝ, r > 0) : 
  r = (2 * Real.sqrt 5) / 3 :=
sorry

end find_radius_l229_22905


namespace find_sales_tax_percentage_l229_22914

noncomputable def salesTaxPercentage (price_with_tax : ℝ) (price_difference : ℝ) : ℝ :=
  (price_difference * 100) / (price_with_tax - price_difference)

theorem find_sales_tax_percentage :
  salesTaxPercentage 2468 161.46 = 7 := by
  sorry

end find_sales_tax_percentage_l229_22914


namespace value_g2_l229_22946

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (g (x - y)) = g x * g y - g x + g y - x^3 * y^3

theorem value_g2 : g 2 = 8 :=
by sorry

end value_g2_l229_22946


namespace math_problem_l229_22991

noncomputable def proof : Prop :=
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  ( (1 / a + 1 / b) / (1 / a - 1 / b) = 1001 ) →
  ((a + b) / (a - b) = 1001)

theorem math_problem : proof := 
  by
    intros a b h₁ h₂ h₃
    sorry

end math_problem_l229_22991


namespace number_of_girls_l229_22902

theorem number_of_girls
  (total_pupils : ℕ)
  (boys : ℕ)
  (teachers : ℕ)
  (girls : ℕ)
  (h1 : total_pupils = 626)
  (h2 : boys = 318)
  (h3 : teachers = 36)
  (h4 : girls = total_pupils - boys - teachers) :
  girls = 272 :=
by
  rw [h1, h2, h3] at h4
  exact h4

-- Proof is not required, hence 'sorry' can be used for practical purposes
-- exact sorry

end number_of_girls_l229_22902


namespace find_a_l229_22954

theorem find_a (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_b : b = 1)
    (h_ab_ccb : (10 * a + b)^2 = 100 * c + 10 * c + b) (h_ccb_gt_300 : 100 * c + 10 * c + b > 300) :
    a = 2 :=
sorry

end find_a_l229_22954


namespace find_m_l229_22992

variables (a : ℕ → ℝ) (r : ℝ) (m : ℕ)

-- Define the conditions of the problem
def exponential_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

def condition_1 (a : ℕ → ℝ) (r : ℝ) : Prop :=
  a 5 * a 6 + a 4 * a 7 = 18

def condition_2 (a : ℕ → ℝ) (m : ℕ) : Prop :=
  a 1 * a m = 9

-- The theorem to prove based on the given conditions
theorem find_m
  (h_exp : exponential_sequence a r)
  (h_r_ne_1 : r ≠ 1)
  (h_cond1 : condition_1 a r)
  (h_cond2 : condition_2 a m) :
  m = 10 :=
sorry

end find_m_l229_22992


namespace simplify_and_evaluate_l229_22938

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) (h3 : x ≠ 1) :
  (x = -1) → ( (x-1) / (x^2 - 2*x + 1) / ((x^2 + x - 1) / (x-1) - (x + 1)) - 1 / (x - 2) = -2 / 3 ) :=
by 
  intro hx
  rw [hx]
  sorry

end simplify_and_evaluate_l229_22938


namespace mean_of_six_numbers_l229_22948

theorem mean_of_six_numbers (sum_six_numbers : ℚ) (h : sum_six_numbers = 3/4) : 
  (sum_six_numbers / 6) = 1/8 := by
  -- proof can be filled in here
  sorry

end mean_of_six_numbers_l229_22948


namespace find_single_digit_l229_22957

def isSingleDigit (n : ℕ) : Prop := n < 10

def repeatedDigitNumber (A : ℕ) : ℕ := 10 * A + A 

theorem find_single_digit (A : ℕ) (h1 : isSingleDigit A) (h2 : repeatedDigitNumber A + repeatedDigitNumber A = 132) : A = 6 :=
by
  sorry

end find_single_digit_l229_22957


namespace prob_two_more_heads_than_tails_eq_210_1024_l229_22917

-- Let P be the probability of getting exactly two more heads than tails when flipping 10 coins.
def P (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2^n : ℚ)

theorem prob_two_more_heads_than_tails_eq_210_1024 :
  P 10 6 = 210 / 1024 :=
by
  -- The steps leading to the proof are omitted and hence skipped
  sorry

end prob_two_more_heads_than_tails_eq_210_1024_l229_22917


namespace gcd_459_357_l229_22984

-- Define the numbers involved
def num1 := 459
def num2 := 357

-- State the proof problem
theorem gcd_459_357 : Int.gcd num1 num2 = 51 := by
  sorry

end gcd_459_357_l229_22984


namespace min_value_of_expression_l229_22969

theorem min_value_of_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + a * b + a * c + b * c = 4) : 2 * a + b + c ≥ 4 :=
sorry

end min_value_of_expression_l229_22969


namespace strictly_decreasing_exponential_l229_22901

theorem strictly_decreasing_exponential (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2*a - 1)^x > (2*a - 1)^y) → (1/2 < a ∧ a < 1) :=
by
  sorry

end strictly_decreasing_exponential_l229_22901


namespace work_completion_together_l229_22962

theorem work_completion_together (man_days : ℕ) (son_days : ℕ) (together_days : ℕ) 
  (h_man : man_days = 10) (h_son : son_days = 10) : together_days = 5 :=
by sorry

end work_completion_together_l229_22962


namespace main_diagonal_squares_second_diagonal_composite_third_diagonal_composite_l229_22947

-- Problem Statement in Lean 4

theorem main_diagonal_squares (k : ℕ) : ∃ m : ℕ, (4 * k * (k + 1) + 1 = m * m) := 
sorry

theorem second_diagonal_composite (k : ℕ) (hk : k ≥ 1) : ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ (4 * (2 * k * (2 * k - 1) - 1) + 1 = a * b) :=
sorry

theorem third_diagonal_composite (k : ℕ) : ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ (4 * ((4 * k + 3) * (4 * k - 1)) + 1 = a * b) :=
sorry

end main_diagonal_squares_second_diagonal_composite_third_diagonal_composite_l229_22947


namespace negation_of_existential_statement_l229_22937

theorem negation_of_existential_statement :
  (¬ ∃ x : ℝ, x ≥ 1 ∨ x > 2) ↔ ∀ x : ℝ, x < 1 :=
by
  sorry

end negation_of_existential_statement_l229_22937


namespace jasmine_max_cards_l229_22903

-- Define constants and conditions
def initial_card_price : ℝ := 0.95
def discount_card_price : ℝ := 0.85
def budget : ℝ := 9.00
def threshold : ℕ := 6

-- Define the condition for the total cost if more than 6 cards are bought
def total_cost (n : ℕ) : ℝ :=
  if n ≤ threshold then initial_card_price * n
  else initial_card_price * threshold + discount_card_price * (n - threshold)

-- Define the condition for the maximum number of cards Jasmine can buy 
def max_cards (n : ℕ) : Prop :=
  total_cost n ≤ budget ∧ ∀ m : ℕ, total_cost m ≤ budget → m ≤ n

-- Theore statement stating Jasmine can buy a maximum of 9 cards
theorem jasmine_max_cards : max_cards 9 :=
sorry

end jasmine_max_cards_l229_22903


namespace minimum_value_expr_l229_22952

noncomputable def expr (x y z : ℝ) : ℝ :=
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) +
  (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2)))

theorem minimum_value_expr : ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) →
  expr x y z ≥ 2 :=
by sorry

end minimum_value_expr_l229_22952


namespace triangle_angles_ratios_l229_22920

def angles_of_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

theorem triangle_angles_ratios (α β γ : ℝ)
  (h1 : α + β + γ = 180) 
  (h2 : β = 2 * α)
  (h3 : γ = 3 * α) : 
  angles_of_triangle 60 45 75 ∨ angles_of_triangle 45 22.5 112.5 :=
by
  sorry

end triangle_angles_ratios_l229_22920


namespace figure_can_be_cut_and_reassembled_into_square_l229_22989

-- Define the conditions
def is_square_area (n: ℕ) : Prop := ∃ k: ℕ, k * k = n

def can_form_square (area: ℕ) : Prop :=
area = 18 ∧ ¬ is_square_area area

-- The proof statement
theorem figure_can_be_cut_and_reassembled_into_square (area: ℕ) (hf: area = 18): 
  can_form_square area → ∃ (part1 part2 part3: Set (ℕ × ℕ)), true :=
by
  sorry

end figure_can_be_cut_and_reassembled_into_square_l229_22989


namespace equation_solution_l229_22987

theorem equation_solution : ∃ x : ℝ, (3 / 20) + (3 / x) = (8 / x) + (1 / 15) ∧ x = 60 :=
by
  use 60
  -- skip the proof
  sorry

end equation_solution_l229_22987


namespace fifth_number_21st_row_is_809_l229_22900

-- Define the sequence of positive odd numbers
def nth_odd_number (n : ℕ) : ℕ :=
  2 * n - 1

-- Define the last odd number in the nth row
def last_odd_number_in_row (n : ℕ) : ℕ :=
  nth_odd_number (n * n)

-- Define the position of the 5th number in the 21st row
def pos_5th_in_21st_row : ℕ :=
  let sum_first_20_rows := 400
  sum_first_20_rows + 5

-- The 5th number from the left in the 21st row
def fifth_number_in_21st_row : ℕ :=
  nth_odd_number pos_5th_in_21st_row

-- The proof statement
theorem fifth_number_21st_row_is_809 : fifth_number_in_21st_row = 809 :=
by
  -- proof omitted
  sorry

end fifth_number_21st_row_is_809_l229_22900


namespace total_cookies_l229_22994

-- Definitions from conditions
def cookies_per_guest : ℕ := 2
def number_of_guests : ℕ := 5

-- Theorem statement that needs to be proved
theorem total_cookies : cookies_per_guest * number_of_guests = 10 := by
  -- We skip the proof since only the statement is required
  sorry

end total_cookies_l229_22994


namespace eiffel_tower_height_l229_22924

-- Define the constants for heights and difference
def BurjKhalifa : ℝ := 830
def height_difference : ℝ := 506

-- The goal: Prove that the height of the Eiffel Tower is 324 m.
theorem eiffel_tower_height : BurjKhalifa - height_difference = 324 := 
by 
sorry

end eiffel_tower_height_l229_22924


namespace priyas_age_l229_22934

/-- 
  Let P be Priya's current age, and F be her father's current age. 
  Given:
  1. F = P + 31
  2. (P + 8) + (F + 8) = 69
  Prove: Priya's current age P is 11.
-/
theorem priyas_age 
  (P F : ℕ) 
  (h1 : F = P + 31) 
  (h2 : (P + 8) + (F + 8) = 69) 
  : P = 11 :=
by
  sorry

end priyas_age_l229_22934


namespace crayon_difference_l229_22958

theorem crayon_difference:
  let karen := 639
  let cindy := 504
  let peter := 752
  let rachel := 315
  max karen (max cindy (max peter rachel)) - min karen (min cindy (min peter rachel)) = 437 :=
by
  sorry

end crayon_difference_l229_22958


namespace sales_tax_difference_l229_22970

theorem sales_tax_difference (rate1 rate2 : ℝ) (price : ℝ) (h1 : rate1 = 0.075) (h2 : rate2 = 0.0625) (hprice : price = 50) : 
  rate1 * price - rate2 * price = 0.625 :=
by
  sorry

end sales_tax_difference_l229_22970


namespace number_of_subsets_l229_22951

def num_subsets (n : ℕ) : ℕ := 2 ^ n

theorem number_of_subsets (A : Finset α) (n : ℕ) (h : A.card = n) : A.powerset.card = num_subsets n :=
by
  have : A.powerset.card = 2 ^ A.card := sorry -- Proof omitted
  rw [h] at this
  exact this

end number_of_subsets_l229_22951


namespace joan_books_correct_l229_22936

def sam_books : ℕ := 110
def total_books : ℕ := 212

def joan_books : ℕ := total_books - sam_books

theorem joan_books_correct : joan_books = 102 := by
  sorry

end joan_books_correct_l229_22936


namespace arithmetic_seq_50th_term_l229_22979

def arithmetic_seq_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_seq_50th_term : 
  arithmetic_seq_nth_term 3 7 50 = 346 :=
by
  -- Intentionally left as sorry
  sorry

end arithmetic_seq_50th_term_l229_22979


namespace scorpion_millipedes_needed_l229_22949

theorem scorpion_millipedes_needed 
  (total_segments_required : ℕ)
  (eaten_millipede_1_segments : ℕ)
  (eaten_millipede_2_segments : ℕ)
  (segments_per_millipede : ℕ)
  (n_millipedes_needed : ℕ)
  (total_segments : total_segments_required = 800) 
  (segments_1 : eaten_millipede_1_segments = 60)
  (segments_2 : eaten_millipede_2_segments = 2 * 60)
  (needed_segments_calculation : 800 - (60 + 2 * (2 * 60)) = n_millipedes_needed * 50) 
  : n_millipedes_needed = 10 :=
by
  sorry

end scorpion_millipedes_needed_l229_22949


namespace sin_double_angle_half_pi_l229_22913

theorem sin_double_angle_half_pi (θ : ℝ) (h : Real.cos (θ + Real.pi) = -1 / 3) : 
  Real.sin (2 * θ + Real.pi / 2) = -7 / 9 := 
by
  sorry

end sin_double_angle_half_pi_l229_22913


namespace problem1_problem2_l229_22988

theorem problem1 : ( (2 / 3 - 1 / 4 - 5 / 6) * 12 = -5 ) :=
by sorry

theorem problem2 : ( (-3)^2 * 2 + 4 * (-3) - 28 / (7 / 4) = -10 ) :=
by sorry

end problem1_problem2_l229_22988


namespace garden_length_l229_22928

-- Define the perimeter and breadth
def perimeter : ℕ := 900
def breadth : ℕ := 190

-- Define a function to calculate the length using given conditions
def length (P : ℕ) (B : ℕ) : ℕ := (P / 2) - B

-- Theorem stating that for the given perimeter and breadth, the length is 260.
theorem garden_length : length perimeter breadth = 260 :=
by
  -- placeholder for proof
  sorry

end garden_length_l229_22928
