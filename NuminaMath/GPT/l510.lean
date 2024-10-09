import Mathlib

namespace total_legs_of_camden_dogs_l510_51069

-- Defining the number of dogs Justin has
def justin_dogs : ℕ := 14

-- Defining the number of dogs Rico has
def rico_dogs : ℕ := justin_dogs + 10

-- Defining the number of dogs Camden has
def camden_dogs : ℕ := 3 * rico_dogs / 4

-- Defining the total number of legs Camden's dogs have
def camden_dogs_legs : ℕ := camden_dogs * 4

-- The proof statement
theorem total_legs_of_camden_dogs : camden_dogs_legs = 72 :=
by
  -- skip proof
  sorry

end total_legs_of_camden_dogs_l510_51069


namespace papaya_production_l510_51003

theorem papaya_production (P : ℕ)
  (h1 : 2 * P + 3 * 20 = 80) :
  P = 10 := 
by sorry

end papaya_production_l510_51003


namespace stock_price_return_to_initial_l510_51038

variable (P₀ : ℝ) -- Initial price
variable (y : ℝ) -- Percentage increase during the fourth week

/-- The main theorem stating the required percentage increase in the fourth week -/
theorem stock_price_return_to_initial
  (h1 : P₀ * 1.30 * 0.75 * 1.20 = 117) -- Condition after three weeks
  (h2 : P₃ = P₀) : -- Price returns to initial
  y = -15 := 
by
  sorry

end stock_price_return_to_initial_l510_51038


namespace hypotenuse_length_l510_51014

theorem hypotenuse_length (a b c : ℝ)
  (h_a : a = 12)
  (h_area : 54 = 1 / 2 * a * b)
  (h_py : c^2 = a^2 + b^2) :
    c = 15 := by
  sorry

end hypotenuse_length_l510_51014


namespace option_C_not_like_terms_l510_51024

theorem option_C_not_like_terms :
  ¬ (2 * (m : ℝ) == 2 * (n : ℝ)) :=
by
  sorry

end option_C_not_like_terms_l510_51024


namespace inequality1_in_triangle_inequality2_in_triangle_l510_51045

theorem inequality1_in_triangle (a b c s : ℝ)
  (h1 : a + b + c = s) :
  (13 / 27) * s^2 ≤ a^2 + b^2 + c^2 + (4 / s) * a * b * c ∧ 
  a^2 + b^2 + c^2 + (4 / s) * a * b * c < s^2 / 2 :=
sorry

theorem inequality2_in_triangle (a b c s : ℝ)
  (h1 : a + b + c = s) :
  s^2 / 4 < a * b + b * c + c * a - (2 / s) * a * b * c ∧ 
  a * b + b * c + c * a - (2 / s) * a * b * c ≤ (7 / 27) * s^2 :=
sorry

end inequality1_in_triangle_inequality2_in_triangle_l510_51045


namespace problem_solution_l510_51029

noncomputable def omega : ℂ := sorry -- Choose a suitable representative for ω

variables (a b c d : ℝ) (h₀ : a ≠ -1) (h₁ : b ≠ -1) (h₂ : c ≠ -1) (h₃ : d ≠ -1)
          (hω : ω^3 = 1 ∧ ω ≠ 1)
          (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω)

theorem problem_solution (a b c d : ℝ) (h₀ : a ≠ -1) (h₁ : b ≠ -1) (h₂ : c ≠ -1) (h₃ : d ≠ -1)
  (hω : ω^3 = 1 ∧ ω ≠ 1)
  (h : (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω)) = 3 / ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 3 :=
sorry

end problem_solution_l510_51029


namespace students_at_end_of_year_l510_51093

def students_start : ℝ := 42.0
def students_left : ℝ := 4.0
def students_transferred : ℝ := 10.0
def students_end : ℝ := 28.0

theorem students_at_end_of_year :
  students_start - students_left - students_transferred = students_end := by
  sorry

end students_at_end_of_year_l510_51093


namespace log_comparison_l510_51077

theorem log_comparison :
  (Real.log 80 / Real.log 20) < (Real.log 640 / Real.log 80) :=
by
  sorry

end log_comparison_l510_51077


namespace area_evaluation_l510_51016

noncomputable def radius : ℝ := 6
noncomputable def central_angle : ℝ := 90
noncomputable def p := 18
noncomputable def q := 3
noncomputable def r : ℝ := -27 / 2

theorem area_evaluation :
  p + q + r = 7.5 :=
by
  sorry

end area_evaluation_l510_51016


namespace age_difference_l510_51048

theorem age_difference (h b m : ℕ) (ratio : h = 4 * m ∧ b = 3 * m ∧ 4 * m + 3 * m + 7 * m = 126) : h - b = 9 :=
by
  -- proof will be filled here
  sorry

end age_difference_l510_51048


namespace range_of_a_l510_51002

-- Given function
def f (x a : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- Derivative of the function
def f' (x a : ℝ) : ℝ := 3*x^2 + 2*a*x + (a + 6)

-- Discriminant of the derivative
def discriminant (a : ℝ) : ℝ := 4*a^2 - 12*(a + 6)

-- Proof that the range of 'a' is 'a < -3 or a > 6' for f(x) to have both maximum and minimum values
theorem range_of_a (a : ℝ) : discriminant a > 0 ↔ (a < -3 ∨ a > 6) :=
by
  sorry

end range_of_a_l510_51002


namespace number_of_candies_picked_up_l510_51097

-- Definitions of the conditions
def num_sides_decagon := 10
def diagonals_from_one_vertex (n : Nat) : Nat := n - 3

-- The theorem stating the number of candies Hyeonsu picked up
theorem number_of_candies_picked_up : diagonals_from_one_vertex num_sides_decagon = 7 := by
  sorry

end number_of_candies_picked_up_l510_51097


namespace pentagon_square_ratio_l510_51054

theorem pentagon_square_ratio (p s : ℝ) (h₁ : 5 * p = 20) (h₂ : 4 * s = 20) : p / s = 4 / 5 := 
by 
  sorry

end pentagon_square_ratio_l510_51054


namespace distance_AD_btw_41_and_42_l510_51084

noncomputable def distance_between (x y : ℝ × ℝ) : ℝ :=
  Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

theorem distance_AD_btw_41_and_42 :
  let A := (0, 0)
  let B := (15, 0)
  let C := (15, 5 * Real.sqrt 3)
  let D := (15, 5 * Real.sqrt 3 + 30)

  41 < distance_between A D ∧ distance_between A D < 42 :=
by
  sorry

end distance_AD_btw_41_and_42_l510_51084


namespace suitable_k_first_third_quadrants_l510_51030

theorem suitable_k_first_third_quadrants (k : ℝ) : 
  (∀ x : ℝ, (x > 0 → k * x > 0) ∧ (x < 0 → k * x < 0)) → k > 0 :=
by
  sorry

end suitable_k_first_third_quadrants_l510_51030


namespace candy_distribution_proof_l510_51090

theorem candy_distribution_proof :
  ∀ (candy_total Kate Robert Bill Mary : ℕ),
  candy_total = 20 →
  Kate = 4 →
  Robert = Kate + 2 →
  Bill = Mary - 6 →
  Kate = Bill + 2 →
  Mary > Robert →
  (Mary - Robert = 2) :=
by
  intros candy_total Kate Robert Bill Mary h1 h2 h3 h4 h5 h6
  sorry

end candy_distribution_proof_l510_51090


namespace x_intercept_l510_51061

theorem x_intercept (x y : ℝ) (h : 4 * x - 3 * y = 24) (hy : y = 0) : (x, y) = (6, 0) :=
by {
  sorry
}

end x_intercept_l510_51061


namespace parameterization_of_line_l510_51080

theorem parameterization_of_line : 
  ∀ (r k : ℝ),
  (∀ t : ℝ, (∃ x y : ℝ, (x, y) = (r, 2) + t • (3, k)) → y = 2 * x - 6) → (r = 4 ∧ k = 6) :=
by
  sorry

end parameterization_of_line_l510_51080


namespace problem_quadrilateral_inscribed_in_circle_l510_51060

theorem problem_quadrilateral_inscribed_in_circle
  (r : ℝ)
  (AB BC CD DA : ℝ)
  (h_radius : r = 300 * Real.sqrt 2)
  (h_AB : AB = 300)
  (h_BC : BC = 150)
  (h_CD : CD = 150) :
  DA = 750 :=
sorry

end problem_quadrilateral_inscribed_in_circle_l510_51060


namespace y_give_z_start_l510_51075

variables (Vx Vy Vz T : ℝ)
variables (D : ℝ)

-- Conditions
def condition1 : Prop := Vx * T = Vy * T + 100
def condition2 : Prop := Vx * T = Vz * T + 200
def condition3 : Prop := T > 0

theorem y_give_z_start (h1 : condition1 Vx Vy T) (h2 : condition2 Vx Vz T) (h3 : condition3 T) : (Vy - Vz) * T = 200 := 
by
  sorry

end y_give_z_start_l510_51075


namespace karen_average_speed_l510_51018

noncomputable def total_distance : ℚ := 198
noncomputable def start_time : ℚ := (9 * 60 + 40) / 60
noncomputable def end_time : ℚ := (13 * 60 + 20) / 60
noncomputable def total_time : ℚ := end_time - start_time
noncomputable def average_speed (distance : ℚ) (time : ℚ) : ℚ := distance / time

theorem karen_average_speed :
  average_speed total_distance total_time = 54 := by
  sorry

end karen_average_speed_l510_51018


namespace probability_of_rolling_greater_than_five_l510_51050

def probability_of_greater_than_five (dice_faces : Finset ℕ) (greater_than : ℕ) : ℚ := 
  let favorable_outcomes := dice_faces.filter (λ x => x > greater_than)
  favorable_outcomes.card / dice_faces.card

theorem probability_of_rolling_greater_than_five:
  probability_of_greater_than_five ({1, 2, 3, 4, 5, 6} : Finset ℕ) 5 = 1 / 6 :=
by
  sorry

end probability_of_rolling_greater_than_five_l510_51050


namespace largest_lcm_among_given_pairs_l510_51049

theorem largest_lcm_among_given_pairs : 
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by
  sorry

end largest_lcm_among_given_pairs_l510_51049


namespace tens_digit_of_8_pow_2048_l510_51026

theorem tens_digit_of_8_pow_2048 : (8^2048 % 100) / 10 = 8 := 
by
  sorry

end tens_digit_of_8_pow_2048_l510_51026


namespace train_length_is_correct_l510_51039

noncomputable def train_length (speed_kmph : ℝ) (time_sec : ℝ) (bridge_length : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time_sec
  total_distance - bridge_length

theorem train_length_is_correct :
  train_length 60 20.99832013438925 240 = 110 :=
by
  sorry

end train_length_is_correct_l510_51039


namespace rita_months_needed_l510_51000

def total_hours_required : ℕ := 2500
def backstroke_hours : ℕ := 75
def breaststroke_hours : ℕ := 25
def butterfly_hours : ℕ := 200
def hours_per_month : ℕ := 300

def total_completed_hours : ℕ := backstroke_hours + breaststroke_hours + butterfly_hours
def remaining_hours : ℕ := total_hours_required - total_completed_hours
def months_needed (remaining_hours hours_per_month : ℕ) : ℕ := (remaining_hours + hours_per_month - 1) / hours_per_month

theorem rita_months_needed : months_needed remaining_hours hours_per_month = 8 := by
  -- Lean 4 proof goes here
  sorry

end rita_months_needed_l510_51000


namespace value_of_f_2014_l510_51062

def f : ℕ → ℕ := sorry

theorem value_of_f_2014 : (∀ n : ℕ, f (f n) + f n = 2 * n + 3) → (f 0 = 1) → (f 2014 = 2015) := by
  intro h₁ h₀
  have h₂ := h₀
  sorry

end value_of_f_2014_l510_51062


namespace breadth_of_garden_l510_51028

theorem breadth_of_garden (P L B : ℝ) (hP : P = 1800) (hL : L = 500) : B = 400 :=
by
  sorry

end breadth_of_garden_l510_51028


namespace polynomial_div_remainder_l510_51081

theorem polynomial_div_remainder (x : ℝ) : 
  (x^4 % (x^2 + 7*x + 2)) = -315*x - 94 := 
by
  sorry

end polynomial_div_remainder_l510_51081


namespace find_interest_rate_l510_51083

-- Conditions
def principal1 : ℝ := 100
def rate1 : ℝ := 0.05
def time1 : ℕ := 48

def principal2 : ℝ := 600
def time2 : ℕ := 4

-- The given interest produced by the first amount
def interest1 : ℝ := principal1 * rate1 * time1

-- The interest produced by the second amount should be the same
def interest2 (rate2 : ℝ) : ℝ := principal2 * rate2 * time2

-- The interest rate to prove
def rate2_correct : ℝ := 0.1

theorem find_interest_rate :
  ∃ rate2 : ℝ, interest2 rate2 = interest1 ∧ rate2 = rate2_correct :=
by
  sorry

end find_interest_rate_l510_51083


namespace simplify_expression_l510_51006

theorem simplify_expression : 
  2 + (1 / (2 + Real.sqrt 5)) - (1 / (2 - Real.sqrt 5)) = 2 - 2 * Real.sqrt 5 := 
by 
  sorry

end simplify_expression_l510_51006


namespace white_pieces_total_l510_51015

theorem white_pieces_total (B W : ℕ) 
  (h_total_pieces : B + W = 300) 
  (h_total_piles : 100 * 3 = B + W) 
  (h_piles_1_white : {n : ℕ | n = 27}) 
  (h_piles_2_3_black : {m : ℕ | m = 42}) 
  (h_piles_3_black_3_white : 15 = 15) :
  W = 158 :=
by
  sorry

end white_pieces_total_l510_51015


namespace exists_n_satisfying_conditions_l510_51094

open Nat

-- Define that n satisfies the given conditions
theorem exists_n_satisfying_conditions :
  ∃ (n : ℤ), (∃ (k : ℤ), 2 * n + 1 = (2 * k + 1) ^ 2) ∧ 
            (∃ (h : ℤ), 3 * n + 1 = (2 * h + 1) ^ 2) ∧ 
            (40 ∣ n) := by
  sorry

end exists_n_satisfying_conditions_l510_51094


namespace ratio_w_to_y_l510_51064

variables {w x y z : ℝ}

theorem ratio_w_to_y
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 9) :
  w / y = 8 :=
by
  sorry

end ratio_w_to_y_l510_51064


namespace ratio_amy_jeremy_l510_51037

variable (Amy Chris Jeremy : ℕ)

theorem ratio_amy_jeremy (h1 : Amy + Jeremy + Chris = 132) (h2 : Jeremy = 66) (h3 : Chris = 2 * Amy) : 
  Amy / Jeremy = 1 / 3 :=
by
  sorry

end ratio_amy_jeremy_l510_51037


namespace area_of_triangle_BXC_l510_51087

-- Define a trapezoid ABCD with given conditions
structure Trapezoid :=
  (A B C D X : Type)
  (AB CD : ℝ)
  (area_ABCD : ℝ)
  (intersect_at_X : Prop)

theorem area_of_triangle_BXC (t : Trapezoid) (h1 : t.AB = 24) (h2 : t.CD = 40)
  (h3 : t.area_ABCD = 480) (h4 : t.intersect_at_X) : 
  ∃ (area_BXC : ℝ), area_BXC = 120 :=
by {
  -- skip the proof here by using sorry
  sorry
}

end area_of_triangle_BXC_l510_51087


namespace factorize_expression_l510_51079

theorem factorize_expression (x : ℝ) : x^2 - 2023 * x = x * (x - 2023) := 
by 
  sorry

end factorize_expression_l510_51079


namespace correct_system_equations_l510_51047

theorem correct_system_equations (x y : ℤ) : 
  (8 * x - y = 3) ∧ (y - 7 * x = 4) ↔ 
    (8 * x - y = 3) ∧ (y - 7 * x = 4) := by
  sorry

end correct_system_equations_l510_51047


namespace perfect_square_impossible_l510_51008
noncomputable def is_perfect_square (n : ℕ) : Prop :=
∃ m : ℕ, m * m = n

theorem perfect_square_impossible (a b c : ℕ) (a_positive : a > 0) (b_positive : b > 0) (c_positive : c > 0) :
  ¬ (is_perfect_square (a^2 + b + c) ∧ is_perfect_square (b^2 + c + a) ∧ is_perfect_square (c^2 + a + b)) :=
sorry

end perfect_square_impossible_l510_51008


namespace problem_equiv_l510_51059

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem problem_equiv (x y : ℝ) : dollar ((2 * x + y) ^ 2) ((x - 2 * y) ^ 2) = (3 * x ^ 2 + 8 * x * y - 3 * y ^ 2) ^ 2 := by
  sorry

end problem_equiv_l510_51059


namespace transport_connectivity_l510_51082

-- Define the condition that any two cities are connected by either an air route or a canal.
-- We will formalize this with an inductive type to represent the transport means: AirRoute or Canal.
inductive TransportMeans
| AirRoute : TransportMeans
| Canal : TransportMeans

open TransportMeans

-- Represent cities as a type 'City'
universe u
variable (City : Type u)

-- Connect any two cities by a transport means
variable (connected : City → City → TransportMeans)

-- We want to prove that for any set of cities, 
-- there exists a means of transport such that starting from any city,
-- it is possible to reach any other city using only that means of transport.
theorem transport_connectivity (n : ℕ) (h2 : n ≥ 2) : 
  ∃ (T : TransportMeans), ∀ (c1 c2 : City), connected c1 c2 = T :=
by
  sorry

end transport_connectivity_l510_51082


namespace find_pos_ints_l510_51007

theorem find_pos_ints (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
    (((m = 1) ∨ (a = 1) ∨ (a = 2 ∧ m = 3 ∧ 2 ≤ n)) →
    (a^m + 1 ∣ (a + 1)^n)) :=
by
  sorry

end find_pos_ints_l510_51007


namespace xn_plus_inv_xn_l510_51065

theorem xn_plus_inv_xn (θ : ℝ) (x : ℝ) (n : ℕ) (h₀ : 0 < θ) (h₁ : θ < π / 2)
  (h₂ : x + 1 / x = -2 * Real.sin θ) (hn_pos : 0 < n) :
  x ^ n + x⁻¹ ^ n = -2 * Real.sin (n * θ) := by
  sorry

end xn_plus_inv_xn_l510_51065


namespace range_of_b_l510_51052

def M := {p : ℝ × ℝ | p.1 ^ 2 + 2 * p.2 ^ 2 = 3}
def N (m b : ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + b}

theorem range_of_b (b : ℝ) : (∀ (m : ℝ), (∃ (p : ℝ × ℝ), p ∈ M ∧ p ∈ N m b)) ↔ 
  -Real.sqrt (6) / 2 ≤ b ∧ b ≤ Real.sqrt (6) / 2 :=
by
  sorry

end range_of_b_l510_51052


namespace wrappers_after_collection_l510_51095

theorem wrappers_after_collection (caps_found : ℕ) (wrappers_found : ℕ) (current_caps : ℕ) (initial_caps : ℕ) : 
  caps_found = 22 → wrappers_found = 30 → current_caps = 17 → initial_caps = 0 → 
  wrappers_found ≥ 30 := 
by 
  intros h1 h2 h3 h4
  -- Solution steps are omitted on purpose
  --- This is where the proof is written
  sorry

end wrappers_after_collection_l510_51095


namespace third_smallest_four_digit_in_pascals_triangle_l510_51001

def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (i j : ℕ), j ≤ i ∧ n = Nat.choose i j

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n : ℕ, is_in_pascals_triangle n ∧ is_four_digit_number n ∧
  (∀ m : ℕ, is_in_pascals_triangle m ∧ is_four_digit_number m 
   → m = 1000 ∨ m = 1001 ∨ m = n) ∧ n = 1002 := sorry

end third_smallest_four_digit_in_pascals_triangle_l510_51001


namespace total_days_2001_to_2004_l510_51041

def regular_year_days : ℕ := 365
def leap_year_days : ℕ := 366
def num_regular_years : ℕ := 3
def num_leap_years : ℕ := 1

theorem total_days_2001_to_2004 : 
  (num_regular_years * regular_year_days) + (num_leap_years * leap_year_days) = 1461 :=
by
  sorry

end total_days_2001_to_2004_l510_51041


namespace amount_invested_l510_51027

variables (P y : ℝ)

-- Conditions
def condition1 : Prop := 800 = P * (2 * y) / 100
def condition2 : Prop := 820 = P * ((1 + y / 100) ^ 2 - 1)

-- The proof we seek
theorem amount_invested (h1 : condition1 P y) (h2 : condition2 P y) : P = 8000 :=
by
  -- Place the proof here
  sorry

end amount_invested_l510_51027


namespace net_profit_is_correct_l510_51021

-- Define the known quantities
def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.20
def markup : ℝ := 45

-- Define the derived quantities based on the conditions
def overhead : ℝ := overhead_percentage * purchase_price
def total_cost : ℝ := purchase_price + overhead
def selling_price : ℝ := total_cost + markup
def net_profit : ℝ := selling_price - total_cost

-- The statement to prove
theorem net_profit_is_correct : net_profit = 45 := by
  sorry

end net_profit_is_correct_l510_51021


namespace Problem1_l510_51013

theorem Problem1 (x y : ℝ) (h : x^2 + y^2 = 1) : x^6 + 3*x^2*y^2 + y^6 = 1 := 
by
  sorry

end Problem1_l510_51013


namespace intersection_M_S_l510_51019

namespace ProofProblem

def M : Set ℕ := { x | 0 < x ∧ x < 4 }

def S : Set ℕ := { 2, 3, 5 }

theorem intersection_M_S :
  M ∩ S = { 2, 3 } := by
  sorry

end ProofProblem

end intersection_M_S_l510_51019


namespace linda_age_l510_51020

theorem linda_age 
  (J : ℕ)  -- Jane's current age
  (H1 : ∃ J, 2 * J + 3 = 13) -- Linda is 3 more than 2 times the age of Jane
  (H2 : (J + 5) + ((2 * J + 3) + 5) = 28) -- In 5 years, the sum of their ages will be 28
  : 2 * J + 3 = 13 :=
by {
  sorry
}

end linda_age_l510_51020


namespace find_common_ratio_l510_51092

theorem find_common_ratio (a_1 q : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS1 : S 1 = a_1)
  (hS2 : S 2 = a_1 * (1 + q))
  (hS3 : S 3 = a_1 * (1 + q + q^2))
  (ha2 : a 2 = a_1 * q)
  (ha3 : a 3 = a_1 * q^2)
  (hcond : 2 * (S 1 + 2 * a 2) = S 3 + a 3 + S 2 + a 2) :
  q = -1/2 :=
by
  sorry

end find_common_ratio_l510_51092


namespace sum_series_eq_l510_51031

theorem sum_series_eq : 
  (∑' k : ℕ, (k + 1) * (1/4)^(k + 1)) = 4 / 9 :=
by sorry

end sum_series_eq_l510_51031


namespace number_of_members_l510_51056

theorem number_of_members (n : ℕ) (h : n * n = 8649) : n = 93 :=
by
  sorry

end number_of_members_l510_51056


namespace point_on_circle_l510_51096

theorem point_on_circle (t : ℝ) : 
  let x := (1 - t^2) / (1 + t^2)
  let y := (3 * t) / (1 + t^2)
  x^2 + y^2 = 1 :=
by
  let x := (1 - t^2) / (1 + t^2)
  let y := (3 * t) / (1 + t^2)
  sorry

end point_on_circle_l510_51096


namespace Pascal_remaining_distance_l510_51098

theorem Pascal_remaining_distance (D T : ℕ) :
  let current_speed := 8
  let reduced_speed := current_speed - 4
  let increased_speed := current_speed + current_speed / 2
  (D = current_speed * T) →
  (D = reduced_speed * (T + 16)) →
  (D = increased_speed * (T - 16)) →
  D = 256 :=
by
  intros
  sorry

end Pascal_remaining_distance_l510_51098


namespace typist_current_salary_l510_51032

-- Define the initial conditions as given in the problem
def initial_salary : ℝ := 6000
def raise_percentage : ℝ := 0.10
def reduction_percentage : ℝ := 0.05

-- Define the calculations for raised and reduced salaries
def raised_salary := initial_salary * (1 + raise_percentage)
def current_salary := raised_salary * (1 - reduction_percentage)

-- State the theorem to prove the current salary
theorem typist_current_salary : current_salary = 6270 := 
by
  -- Sorry is used to skip proof, overriding with the statement to ensure code builds successfully
  sorry

end typist_current_salary_l510_51032


namespace exists_maximum_value_of_f_l510_51067

-- Define the function f(x, y)
noncomputable def f (x y : ℝ) : ℝ := (3 * x * y + 1) * Real.exp (-(x^2 + y^2))

-- Maximum value proof statement
theorem exists_maximum_value_of_f :
  ∃ (x y : ℝ), f x y = (3 / 2) * Real.exp (-1 / 3) :=
sorry

end exists_maximum_value_of_f_l510_51067


namespace wanda_blocks_l510_51063

theorem wanda_blocks (initial_blocks: ℕ) (additional_blocks: ℕ) (total_blocks: ℕ) : 
  initial_blocks = 4 → additional_blocks = 79 → total_blocks = initial_blocks + additional_blocks → total_blocks = 83 :=
by
  intros hi ha ht
  rw [hi, ha] at ht
  exact ht

end wanda_blocks_l510_51063


namespace weight_of_5_moles_H₂CO₃_l510_51040

-- Definitions based on the given conditions
def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

def num_H₂CO₃_H : ℕ := 2
def num_H₂CO₃_C : ℕ := 1
def num_H₂CO₃_O : ℕ := 3

def molecular_weight (num_H num_C num_O : ℕ) 
                     (weight_H weight_C weight_O : ℝ) : ℝ :=
  num_H * weight_H + num_C * weight_C + num_O * weight_O

-- Main proof statement
theorem weight_of_5_moles_H₂CO₃ :
  5 * molecular_weight num_H₂CO₃_H num_H₂CO₃_C num_H₂CO₃_O 
                       atomic_weight_H atomic_weight_C atomic_weight_O 
  = 310.12 := by
  sorry

end weight_of_5_moles_H₂CO₃_l510_51040


namespace sum_of_roots_eq_zero_product_of_roots_eq_neg_twentyfive_l510_51085

theorem sum_of_roots_eq_zero (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  ∃ x1 x2 : ℝ, (|x1| = 5) ∧ (|x2| = 5) ∧ x1 + x2 = 0 :=
by
  sorry

theorem product_of_roots_eq_neg_twentyfive (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  ∃ x1 x2 : ℝ, (|x1| = 5) ∧ (|x2| = 5) ∧ x1 * x2 = -25 :=
by
  sorry

end sum_of_roots_eq_zero_product_of_roots_eq_neg_twentyfive_l510_51085


namespace tan_three_theta_l510_51058

theorem tan_three_theta (θ : Real) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_three_theta_l510_51058


namespace work_days_together_l510_51068

variable (d : ℝ) (j : ℝ)

theorem work_days_together (hd : d = 1 / 5) (hj : j = 1 / 9) :
  1 / (d + j) = 45 / 14 := by
  sorry

end work_days_together_l510_51068


namespace tan_identity_l510_51034

open Real

-- Definition of conditions
def isPureImaginary (z : Complex) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem tan_identity (theta : ℝ) :
  isPureImaginary ((cos theta - 4/5) + (sin theta - 3/5) * Complex.I) →
  tan (theta - π / 4) = -7 :=
by
  sorry

end tan_identity_l510_51034


namespace find_p_q_sum_l510_51005

-- Define the number of trees
def pine_trees := 2
def cedar_trees := 3
def fir_trees := 4

-- Total number of trees
def total_trees := pine_trees + cedar_trees + fir_trees

-- Number of ways to arrange the 9 trees
def total_arrangements := Nat.choose total_trees fir_trees

-- Number of ways to place fir trees so no two are adjacent
def valid_arrangements := Nat.choose (pine_trees + cedar_trees + 1) fir_trees

-- Desired probability in its simplest form
def probability := valid_arrangements / total_arrangements

-- Denominator and numerator of the simplified fraction
def num := 5
def den := 42

-- Statement to prove that the probability is 5/42
theorem find_p_q_sum : (num + den) = 47 := by
  sorry

end find_p_q_sum_l510_51005


namespace arithmetic_sequence_term_count_l510_51004

def first_term : ℕ := 5
def common_difference : ℕ := 3
def last_term : ℕ := 203

theorem arithmetic_sequence_term_count :
  ∃ n : ℕ, last_term = first_term + (n - 1) * common_difference ∧ n = 67 :=
by
  sorry

end arithmetic_sequence_term_count_l510_51004


namespace length_of_woods_l510_51043

theorem length_of_woods (area width : ℝ) (h_area : area = 24) (h_width : width = 8) : (area / width) = 3 :=
by
  sorry

end length_of_woods_l510_51043


namespace labor_union_trees_l510_51074

theorem labor_union_trees (x : ℕ) :
  (∃ t : ℕ, t = 2 * x + 21) ∧ (∃ t' : ℕ, t' = 3 * x - 24) →
  2 * x + 21 = 3 * x - 24 :=
by
  sorry

end labor_union_trees_l510_51074


namespace sum_of_two_integers_l510_51009

theorem sum_of_two_integers (a b : ℕ) (h₁ : a * b + a + b = 135) (h₂ : Nat.gcd a b = 1) (h₃ : a < 30) (h₄ : b < 30) : a + b = 23 :=
sorry

end sum_of_two_integers_l510_51009


namespace complementary_events_A_B_l510_51023

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

def A (n : ℕ) : Prop := is_odd n
def B (n : ℕ) : Prop := is_even n
def C (n : ℕ) : Prop := is_multiple_of_3 n

theorem complementary_events_A_B :
  (∀ n, A n → ¬ B n) ∧ (∀ n, B n → ¬ A n) ∧ (∀ n, A n ∨ B n) :=
  sorry

end complementary_events_A_B_l510_51023


namespace large_rectangle_perimeter_correct_l510_51070

def perimeter_of_square (p : ℕ) : ℕ :=
  p / 4

def perimeter_of_rectangle (p : ℕ) (l : ℕ) : ℕ :=
  (p - 2 * l) / 2

def perimeter_of_large_rectangle (side_length_of_square side_length_of_rectangle : ℕ) : ℕ :=
  let height := side_length_of_square + 2 * side_length_of_rectangle
  let width := 3 * side_length_of_square
  2 * (height + width)

theorem large_rectangle_perimeter_correct :
  let side_length_of_square := perimeter_of_square 24
  let side_length_of_rectangle := perimeter_of_rectangle 16 side_length_of_square
  perimeter_of_large_rectangle side_length_of_square side_length_of_rectangle = 52 :=
by
  sorry

end large_rectangle_perimeter_correct_l510_51070


namespace value_of_g_neg2_l510_51046

def g (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem value_of_g_neg2 : g (-2) = -3 := 
by sorry

end value_of_g_neg2_l510_51046


namespace fraction_simplification_l510_51036

theorem fraction_simplification (a b : ℝ) : 9 * b / (6 * a + 3) = 3 * b / (2 * a + 1) :=
by sorry

end fraction_simplification_l510_51036


namespace find_smallest_M_l510_51099

/-- 
Proof of the smallest real number M such that 
for all real numbers a, b, and c, the following inequality holds:
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)|
    ≤ (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2. 
-/
theorem find_smallest_M (a b c : ℝ) : 
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| 
    ≤ (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2 :=
by
  sorry

end find_smallest_M_l510_51099


namespace determine_phi_l510_51011

variable (ω : ℝ) (varphi : ℝ)

noncomputable def f (ω varphi x: ℝ) : ℝ := Real.sin (ω * x + varphi)

theorem determine_phi
  (hω : ω > 0)
  (hvarphi : 0 < varphi ∧ varphi < π)
  (hx1 : f ω varphi (π/4) = Real.sin (ω * (π / 4) + varphi))
  (hx2 : f ω varphi (5 * π / 4) = Real.sin (ω * (5 * π / 4) + varphi))
  (hsym : ∀ x, f ω varphi x = f ω varphi (π - x))
  : varphi = π / 4 :=
sorry

end determine_phi_l510_51011


namespace speaker_discounted_price_correct_l510_51089

-- Define the initial price and the discount
def initial_price : ℝ := 475.00
def discount : ℝ := 276.00

-- Define the discounted price
def discounted_price : ℝ := initial_price - discount

-- The theorem to prove that the discounted price is 199.00
theorem speaker_discounted_price_correct : discounted_price = 199.00 :=
by
  -- Proof is omitted here, adding sorry to indicate it.
  sorry

end speaker_discounted_price_correct_l510_51089


namespace det_A_is_neg9_l510_51051

noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ := ![![-7, 5], ![6, -3]]

theorem det_A_is_neg9 : Matrix.det A = -9 := 
by 
  sorry

end det_A_is_neg9_l510_51051


namespace max_cities_l510_51025

def city (X : Type) := X

variable (A B C D E : Prop)

-- Conditions as given in the problem
axiom condition1 : A → B
axiom condition2 : D ∨ E
axiom condition3 : B ↔ ¬C
axiom condition4 : C ↔ D
axiom condition5 : E → (A ∧ D)

-- Proof problem: Given the conditions, prove that the maximum set of cities that can be visited is {C, D}
theorem max_cities (h1 : A → B) (h2 : D ∨ E) (h3 : B ↔ ¬C) (h4 : C ↔ D) (h5 : E → (A ∧ D)) : (C ∧ D) ∧ ¬A ∧ ¬B ∧ ¬E :=
by
  -- The core proof would use the constraints to show C and D, and exclude A, B, E
  sorry

end max_cities_l510_51025


namespace exterior_angle_of_polygon_l510_51066

theorem exterior_angle_of_polygon (n : ℕ) (h₁ : (n - 2) * 180 = 1800) (h₂ : n > 2) :
  360 / n = 30 := by
    sorry

end exterior_angle_of_polygon_l510_51066


namespace cone_lateral_surface_area_l510_51091

theorem cone_lateral_surface_area (a : ℝ) (π : ℝ) (sqrt_3 : ℝ) 
  (h₁ : 0 < a)
  (h_area : (1 / 2) * a^2 * (sqrt_3 / 2) = sqrt_3) :
  π * 1 * 2 = 2 * π :=
by
  sorry

end cone_lateral_surface_area_l510_51091


namespace fraction_field_planted_l510_51076

-- Define the problem conditions
structure RightTriangle (leg1 leg2 hypotenuse : ℝ) : Prop :=
  (right_angle : ∃ (A B C : ℝ), A = 5 ∧ B = 12 ∧ hypotenuse = 13 ∧ A^2 + B^2 = hypotenuse^2)

structure SquarePatch (shortest_distance : ℝ) : Prop :=
  (distance_to_hypotenuse : shortest_distance = 3)

-- Define the statement
theorem fraction_field_planted (T : RightTriangle 5 12 13) (P : SquarePatch 3) : 
  ∃ (fraction : ℚ), fraction = 7 / 10 :=
by
  sorry

end fraction_field_planted_l510_51076


namespace troy_buys_beef_l510_51072

theorem troy_buys_beef (B : ℕ) 
  (veg_pounds : ℕ := 6)
  (veg_cost_per_pound : ℕ := 2)
  (beef_cost_per_pound : ℕ := 3 * veg_cost_per_pound)
  (total_cost : ℕ := 36) :
  6 * veg_cost_per_pound + B * beef_cost_per_pound = total_cost → B = 4 :=
by
  sorry

end troy_buys_beef_l510_51072


namespace highway_extension_l510_51086

theorem highway_extension 
  (current_length : ℕ) 
  (desired_length : ℕ) 
  (first_day_miles : ℕ) 
  (miles_needed : ℕ) 
  (second_day_miles : ℕ) 
  (h1 : current_length = 200) 
  (h2 : desired_length = 650) 
  (h3 : first_day_miles = 50) 
  (h4 : miles_needed = 250) 
  (h5 : second_day_miles = desired_length - current_length - miles_needed - first_day_miles) :
  second_day_miles / first_day_miles = 3 := 
sorry

end highway_extension_l510_51086


namespace missing_digit_is_0_l510_51010

/- Define the known digits of the number. -/
def digit1 : ℕ := 6
def digit2 : ℕ := 5
def digit3 : ℕ := 3
def digit4 : ℕ := 4

/- Define the condition that ensures the divisibility by 9. -/
def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

/- The main theorem to prove: the value of the missing digit d is 0. -/
theorem missing_digit_is_0 (d : ℕ) 
  (h : is_divisible_by_9 (digit1 + digit2 + digit3 + digit4 + d)) : 
  d = 0 :=
sorry

end missing_digit_is_0_l510_51010


namespace arithmetic_seq_sum_l510_51055

theorem arithmetic_seq_sum(S : ℕ → ℝ) (d : ℝ) (h1 : S 5 < S 6) 
    (h2 : S 6 = S 7) (h3 : S 7 > S 8) : S 9 < S 5 := 
sorry

end arithmetic_seq_sum_l510_51055


namespace jackson_money_l510_51044

theorem jackson_money (W : ℝ) (H1 : 5 * W + W = 150) : 5 * W = 125 :=
by
  sorry

end jackson_money_l510_51044


namespace same_terminal_side_l510_51057

theorem same_terminal_side (k : ℤ) : ∃ k : ℤ, (2 * k * Real.pi - Real.pi / 6) = 11 * Real.pi / 6 := by
  sorry

end same_terminal_side_l510_51057


namespace new_container_volume_l510_51022

-- Define the original volume of the container 
def original_volume : ℝ := 4

-- Define the scale factor of each dimension (quadrupled)
def scale_factor : ℝ := 4

-- Define the new volume, which is original volume * (scale factor ^ 3)
def new_volume (orig_vol : ℝ) (scale : ℝ) : ℝ := orig_vol * (scale ^ 3)

-- The theorem we want to prove
theorem new_container_volume : new_volume original_volume scale_factor = 256 :=
by
  sorry

end new_container_volume_l510_51022


namespace inequality_abc_l510_51042

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) >= 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_abc_l510_51042


namespace laser_beam_total_distance_l510_51088

theorem laser_beam_total_distance :
  let A := (3, 5)
  let D := (7, 5)
  let D'' := (-7, -5)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)
  distance A D'' = 10 * Real.sqrt 2 :=
by
  -- definitions and conditions are captured
  sorry -- the proof goes here, no proof is required as per instructions

end laser_beam_total_distance_l510_51088


namespace sufficient_not_necessary_l510_51033

theorem sufficient_not_necessary (x : ℝ) : (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 1) ∧ ¬((x ≠ 1) → (x^2 - 3 * x + 2 ≠ 0)) :=
by
  sorry

end sufficient_not_necessary_l510_51033


namespace walter_bus_time_l510_51035

noncomputable def walter_schedule : Prop :=
  let wake_up_time := 6  -- Walter gets up at 6:00 a.m.
  let leave_home_time := 7  -- Walter catches the school bus at 7:00 a.m.
  let arrival_home_time := 17  -- Walter arrives home at 5:00 p.m.
  let num_classes := 8  -- Walter has 8 classes
  let class_duration := 45  -- Each class lasts 45 minutes
  let lunch_duration := 40  -- Walter has 40 minutes for lunch
  let additional_activities_hours := 2.5  -- Walter has 2.5 hours of additional activities

  -- Total time calculation
  let total_away_hours := arrival_home_time - leave_home_time
  let total_away_minutes := total_away_hours * 60

  -- School-related activities calculation
  let total_class_minutes := num_classes * class_duration
  let total_additional_activities_minutes := additional_activities_hours * 60
  let total_school_activity_minutes := total_class_minutes + lunch_duration + total_additional_activities_minutes

  -- Time spent on the bus
  let bus_time := total_away_minutes - total_school_activity_minutes
  bus_time = 50

-- Statement to prove
theorem walter_bus_time : walter_schedule :=
  sorry

end walter_bus_time_l510_51035


namespace find_line_through_intersection_and_perpendicular_l510_51017

-- Definitions for the given conditions
def line1 (x y : ℝ) : Prop := 3 * x - 2 * y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y + 4 = 0
def perpendicular (x y m : ℝ) : Prop := x + 3 * y + 4 = 0 ∧ 3 * x - y + m = 0

theorem find_line_through_intersection_and_perpendicular :
  ∃ m : ℝ, ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ perpendicular x y m → 3 * x - y + 2 = 0 :=
by
  sorry

end find_line_through_intersection_and_perpendicular_l510_51017


namespace value_after_increase_l510_51071

-- Definition of original number and percentage increase
def original_number : ℝ := 600
def percentage_increase : ℝ := 0.10

-- Theorem stating that after a 10% increase, the value is 660
theorem value_after_increase : original_number * (1 + percentage_increase) = 660 := by
  sorry

end value_after_increase_l510_51071


namespace sandy_siding_cost_l510_51053

theorem sandy_siding_cost
  (wall_length wall_height roof_base roof_height : ℝ)
  (siding_length siding_height siding_cost : ℝ)
  (num_walls num_roof_faces num_siding_sections : ℝ)
  (total_cost : ℝ)
  (h_wall_length : wall_length = 10)
  (h_wall_height : wall_height = 7)
  (h_roof_base : roof_base = 10)
  (h_roof_height : roof_height = 6)
  (h_siding_length : siding_length = 10)
  (h_siding_height : siding_height = 15)
  (h_siding_cost : siding_cost = 35)
  (h_num_walls : num_walls = 2)
  (h_num_roof_faces : num_roof_faces = 1)
  (h_num_siding_sections : num_siding_sections = 2)
  (h_total_cost : total_cost = 70) :
  (siding_cost * num_siding_sections) = total_cost := 
by
  sorry

end sandy_siding_cost_l510_51053


namespace smallest_a_divisible_by_1984_l510_51078

theorem smallest_a_divisible_by_1984 :
  ∃ a : ℕ, (∀ n : ℕ, n % 2 = 1 → 1984 ∣ (47^n + a * 15^n)) ∧ a = 1055 := 
by 
  sorry

end smallest_a_divisible_by_1984_l510_51078


namespace base16_to_base2_bits_l510_51073

theorem base16_to_base2_bits :
  ∀ (n : ℕ), n = 16^4 * 7 + 16^3 * 7 + 16^2 * 7 + 16 * 7 + 7 → (2^18 ≤ n ∧ n < 2^19) → 
  ∃ b : ℕ, b = 19 := 
by
  intros n hn hpow
  sorry

end base16_to_base2_bits_l510_51073


namespace parabola_tangent_angle_l510_51012

noncomputable def tangent_slope_angle : Real :=
  let x := (1 / 2 : ℝ)
  let y := x^2
  let slope := (deriv (fun x => x^2)) x
  Real.arctan slope

theorem parabola_tangent_angle :
  tangent_slope_angle = Real.pi / 4 :=
by
sorry

end parabola_tangent_angle_l510_51012
