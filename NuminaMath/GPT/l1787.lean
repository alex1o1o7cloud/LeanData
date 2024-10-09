import Mathlib

namespace sector_area_l1787_178772

theorem sector_area (theta r : ℝ) (h1 : theta = 2 * Real.pi / 3) (h2 : r = 2) :
  (1 / 2 * r ^ 2 * theta) = 4 * Real.pi / 3 := by
  sorry

end sector_area_l1787_178772


namespace number_of_space_diagonals_l1787_178722

theorem number_of_space_diagonals
  (V E F T Q : ℕ)
  (hV : V = 30)
  (hE : E = 70)
  (hF : F = 42)
  (hT : T = 30)
  (hQ : Q = 12):
  (V * (V - 1) / 2 - E - 2 * Q) = 341 :=
by
  sorry

end number_of_space_diagonals_l1787_178722


namespace symmetry_axis_of_transformed_function_l1787_178763

theorem symmetry_axis_of_transformed_function :
  let initial_func (x : ℝ) := Real.sin (4 * x - π / 6)
  let stretched_func (x : ℝ) := Real.sin (8 * x - π / 3)
  let transformed_func (x : ℝ) := Real.sin (8 * (x + π / 4) - π / 3)
  let ω := 8
  let φ := 5 * π / 3
  x = π / 12 :=
  sorry

end symmetry_axis_of_transformed_function_l1787_178763


namespace find_number_l1787_178724

theorem find_number (x : ℝ) (h : (5/4) * x = 40) : x = 32 := 
sorry

end find_number_l1787_178724


namespace domain_of_f_l1787_178736

noncomputable def f (t : ℝ) : ℝ :=  1 / ((abs (t - 1))^2 + (abs (t + 1))^2)

theorem domain_of_f : ∀ t : ℝ, (abs (t - 1))^2 + (abs (t + 1))^2 ≠ 0 :=
by
  intro t
  sorry

end domain_of_f_l1787_178736


namespace find_number_l1787_178743

theorem find_number (n : ℕ) :
  (n % 12 = 11) ∧
  (n % 11 = 10) ∧
  (n % 10 = 9) ∧
  (n % 9 = 8) ∧
  (n % 8 = 7) ∧
  (n % 7 = 6) ∧
  (n % 6 = 5) ∧
  (n % 5 = 4) ∧
  (n % 4 = 3) ∧
  (n % 3 = 2) ∧
  (n % 2 = 1)
  → n = 27719 := 
sorry

end find_number_l1787_178743


namespace smallest_n_common_factor_l1787_178749

theorem smallest_n_common_factor :
  ∃ n : ℕ, n > 0 ∧ (∀ d : ℕ, d > 1 → d ∣ (11 * n - 4) → d ∣ (8 * n - 5)) ∧ n = 15 :=
by {
  -- Define the conditions as given in the problem
  sorry
}

end smallest_n_common_factor_l1787_178749


namespace cucumbers_for_20_apples_l1787_178733

-- Definitions for all conditions
def apples := ℕ
def bananas := ℕ
def cucumbers := ℕ

def cost_equivalence_apples_bananas (a b : ℕ) : Prop := 10 * a = 5 * b
def cost_equivalence_bananas_cucumbers (b c : ℕ) : Prop := 3 * b = 4 * c

-- Main theorem statement
theorem cucumbers_for_20_apples :
  ∀ (a b c : ℕ),
    cost_equivalence_apples_bananas a b →
    cost_equivalence_bananas_cucumbers b c →
    ∃ k : ℕ, k = 13 :=
by
  intros
  sorry

end cucumbers_for_20_apples_l1787_178733


namespace remainder_317_l1787_178758

theorem remainder_317 (y : ℤ)
  (h1 : 4 + y ≡ 9 [ZMOD 16])
  (h2 : 6 + y ≡ 8 [ZMOD 81])
  (h3 : 8 + y ≡ 49 [ZMOD 625]) :
  y ≡ 317 [ZMOD 360] := 
sorry

end remainder_317_l1787_178758


namespace sum_of_x_y_l1787_178753

theorem sum_of_x_y (x y : ℕ) (h1 : 10 * x + y = 75) (h2 : 10 * y + x = 57) : x + y = 12 :=
sorry

end sum_of_x_y_l1787_178753


namespace sunset_time_l1787_178762

theorem sunset_time (length_of_daylight : Nat := 11 * 60 + 18) -- length of daylight in minutes
    (sunrise : Nat := 6 * 60 + 32) -- sunrise time in minutes after midnight
    : (sunrise + length_of_daylight) % (24 * 60) = 17 * 60 + 50 := -- sunset time calculation
by
  sorry

end sunset_time_l1787_178762


namespace rectangular_solid_surface_area_l1787_178781

theorem rectangular_solid_surface_area
  (a b c : ℕ)
  (h_prime_a : Prime a)
  (h_prime_b : Prime b)
  (h_prime_c : Prime c)
  (h_volume : a * b * c = 143) :
  2 * (a * b + b * c + c * a) = 382 := by
  sorry

end rectangular_solid_surface_area_l1787_178781


namespace coordinates_of_P_l1787_178734

variable (a : ℝ)

def y_coord (a : ℝ) : ℝ :=
  3 * a + 9

def x_coord (a : ℝ) : ℝ :=
  4 - a

theorem coordinates_of_P :
  (∃ a : ℝ, y_coord a = 0) → ∃ a : ℝ, (x_coord a, y_coord a) = (7, 0) :=
by
  -- The proof goes here
  sorry

end coordinates_of_P_l1787_178734


namespace total_elephants_in_two_parks_is_280_l1787_178793

def number_of_elephants_we_preserve_for_future : ℕ := 70
def multiple_factor : ℕ := 3

def number_of_elephants_gestures_for_good : ℕ := multiple_factor * number_of_elephants_we_preserve_for_future

def total_number_of_elephants : ℕ := number_of_elephants_we_preserve_for_future + number_of_elephants_gestures_for_good

theorem total_elephants_in_two_parks_is_280 : total_number_of_elephants = 280 :=
by
  sorry

end total_elephants_in_two_parks_is_280_l1787_178793


namespace sufficient_not_necessary_l1787_178769

variable (x : ℝ)
def p := x^2 > 4
def q := x > 2

theorem sufficient_not_necessary : (∀ x, q x -> p x) ∧ ¬ (∀ x, p x -> q x) :=
by sorry

end sufficient_not_necessary_l1787_178769


namespace three_f_x_expression_l1787_178790

variable (f : ℝ → ℝ)
variable (h : ∀ x > 0, f (3 * x) = 3 / (3 + 2 * x))

theorem three_f_x_expression (x : ℝ) (hx : x > 0) : 3 * f x = 27 / (9 + 2 * x) :=
by sorry

end three_f_x_expression_l1787_178790


namespace marlon_gift_card_balance_l1787_178754

theorem marlon_gift_card_balance 
  (initial_amount : ℕ) 
  (spent_monday : initial_amount / 2 = 100)
  (spent_tuesday : (initial_amount / 2) / 4 = 25) 
  : (initial_amount / 2) - (initial_amount / 2 / 4) = 75 :=
by
  sorry

end marlon_gift_card_balance_l1787_178754


namespace purchase_price_eq_360_l1787_178712

theorem purchase_price_eq_360 (P : ℝ) (M : ℝ) (H1 : M = 30) (H2 : M = 0.05 * P + 12) : P = 360 :=
by
  sorry

end purchase_price_eq_360_l1787_178712


namespace ellipse_eccentricity_l1787_178710

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l1787_178710


namespace equidistant_divisors_multiple_of_6_l1787_178765

open Nat

theorem equidistant_divisors_multiple_of_6 (n : ℕ) :
  (∃ a b : ℕ, a ≠ b ∧ a ∣ n ∧ b ∣ n ∧ 
    (a + b = 2 * (n / 3))) → 
  (∃ k : ℕ, n = 6 * k) := 
by
  sorry

end equidistant_divisors_multiple_of_6_l1787_178765


namespace arithmetic_sequence_tenth_term_l1787_178746

noncomputable def prove_tenth_term (a d: ℤ) (h1: a + 2*d = 10) (h2: a + 7*d = 30) : Prop :=
  a + 9*d = 38

theorem arithmetic_sequence_tenth_term (a d: ℤ) (h1: a + 2*d = 10) (h2: a + 7*d = 30) : prove_tenth_term a d h1 h2 :=
by
  sorry

end arithmetic_sequence_tenth_term_l1787_178746


namespace exists_convex_polygon_diagonals_l1787_178787

theorem exists_convex_polygon_diagonals :
  ∃ n : ℕ, n * (n - 3) / 2 = 54 :=
by
  sorry

end exists_convex_polygon_diagonals_l1787_178787


namespace mean_points_scored_l1787_178776

def Mrs_Williams_points : ℝ := 50
def Mr_Adams_points : ℝ := 57
def Mrs_Browns_points : ℝ := 49
def Mrs_Daniels_points : ℝ := 57

def total_points : ℝ := Mrs_Williams_points + Mr_Adams_points + Mrs_Browns_points + Mrs_Daniels_points
def number_of_classes : ℝ := 4

theorem mean_points_scored :
  (total_points / number_of_classes) = 53.25 :=
by
  sorry

end mean_points_scored_l1787_178776


namespace J_3_3_4_l1787_178717

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_3_3_4 : J 3 (3 / 4) 4 = 259 / 48 := 
by {
    -- We would normally include proof steps here, but according to the instruction, we use 'sorry'.
    sorry
}

end J_3_3_4_l1787_178717


namespace grain_output_l1787_178792

-- Define the condition regarding grain output.
def premier_goal (x : ℝ) : Prop :=
  x > 1.3

-- The mathematical statement that needs to be proved, given the condition.
theorem grain_output (x : ℝ) (h : premier_goal x) : x > 1.3 :=
by
  sorry

end grain_output_l1787_178792


namespace totalPayment_l1787_178779

def totalNumberOfTrees : Nat := 850
def pricePerDouglasFir : Nat := 300
def pricePerPonderosaPine : Nat := 225
def numberOfDouglasFirPurchased : Nat := 350
def numberOfPonderosaPinePurchased := totalNumberOfTrees - numberOfDouglasFirPurchased

def costDouglasFir := numberOfDouglasFirPurchased * pricePerDouglasFir
def costPonderosaPine := numberOfPonderosaPinePurchased * pricePerPonderosaPine

def totalCost := costDouglasFir + costPonderosaPine

theorem totalPayment : totalCost = 217500 := by
  sorry

end totalPayment_l1787_178779


namespace ratio_of_sister_to_Aaron_l1787_178784

noncomputable def Aaron_age := 15
variable (H S : ℕ)
axiom Henry_age_relation : H = 4 * S
axiom combined_age : H + S + Aaron_age = 240

theorem ratio_of_sister_to_Aaron : (S : ℚ) / Aaron_age = 3 := 
by
  -- Proof omitted
  sorry

end ratio_of_sister_to_Aaron_l1787_178784


namespace trigonometric_identity_solution_l1787_178759

open Real

noncomputable def x_sol1 (k : ℤ) : ℝ := (π / 2) * (4 * k - 1)
noncomputable def x_sol2 (l : ℤ) : ℝ := (π / 3) * (6 * l + 1)
noncomputable def x_sol2_neg (l : ℤ) : ℝ := (π / 3) * (6 * l - 1)

theorem trigonometric_identity_solution (x : ℝ) :
    (3 * sin (x / 2) ^ 2 * cos (3 * π / 2 + x / 2) +
    3 * sin (x / 2) ^ 2 * cos (x / 2) -
    sin (x / 2) * cos (x / 2) ^ 2 =
    sin (π / 2 + x / 2) ^ 2 * cos (x / 2)) →
    (∃ k : ℤ, x = x_sol1 k) ∨
    (∃ l : ℤ, x = x_sol2 l ∨ x = x_sol2_neg l) :=
by
  sorry

end trigonometric_identity_solution_l1787_178759


namespace sum_is_composite_l1787_178771

theorem sum_is_composite (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
    (h : a^2 - a * b + b^2 = c^2 - c * d + d^2) : ∃ k l : ℕ, k > 1 ∧ l > 1 ∧ k * l = a + b + c + d :=
by sorry

end sum_is_composite_l1787_178771


namespace server_multiplications_in_half_hour_l1787_178703

theorem server_multiplications_in_half_hour : 
  let rate := 5000
  let seconds_in_half_hour := 1800
  rate * seconds_in_half_hour = 9000000 := by
  sorry

end server_multiplications_in_half_hour_l1787_178703


namespace area_of_triangle_COD_l1787_178731

theorem area_of_triangle_COD (x p : ℕ) (hx : 0 < x) (hx' : x < 12) (hp : 0 < p) :
  (∃ A : ℚ, A = (x * p : ℚ) / 2) :=
sorry

end area_of_triangle_COD_l1787_178731


namespace hands_straight_line_time_l1787_178795

noncomputable def time_when_hands_straight_line : List (ℕ × ℚ) :=
  let x₁ := 21 + 9 / 11
  let x₂ := 54 + 6 / 11
  [(4, x₁), (4, x₂)]

theorem hands_straight_line_time :
  time_when_hands_straight_line = [(4, 21 + 9 / 11), (4, 54 + 6 / 11)] :=
by
  sorry

end hands_straight_line_time_l1787_178795


namespace cubs_more_home_runs_than_cardinals_l1787_178766

theorem cubs_more_home_runs_than_cardinals 
(h1 : 2 + 1 + 2 = 5) 
(h2 : 1 + 1 = 2) : 
5 - 2 = 3 :=
by sorry

end cubs_more_home_runs_than_cardinals_l1787_178766


namespace total_people_veg_l1787_178761

-- Definitions based on the conditions
def people_only_veg : ℕ := 13
def people_both_veg_nonveg : ℕ := 6

-- The statement we need to prove
theorem total_people_veg : people_only_veg + people_both_veg_nonveg = 19 :=
by
  sorry

end total_people_veg_l1787_178761


namespace not_monotonic_on_interval_l1787_178713

noncomputable def f (x : ℝ) : ℝ := (x^2 / 2) - Real.log x

theorem not_monotonic_on_interval (m : ℝ) : 
  (∃ x y : ℝ, m < x ∧ x < m + 1/2 ∧ m < y ∧ y < m + 1/2 ∧ (x ≠ y) ∧ f x ≠ f y ) ↔ (1/2 < m ∧ m < 1) :=
sorry

end not_monotonic_on_interval_l1787_178713


namespace roots_of_x2_eq_x_l1787_178768

theorem roots_of_x2_eq_x : ∀ x : ℝ, x^2 = x ↔ (x = 0 ∨ x = 1) := 
by
  sorry

end roots_of_x2_eq_x_l1787_178768


namespace meet_at_starting_line_l1787_178732

theorem meet_at_starting_line (henry_time margo_time : ℕ) (h_henry : henry_time = 7) (h_margo : margo_time = 12) : Nat.lcm henry_time margo_time = 84 :=
by
  rw [h_henry, h_margo]
  sorry

end meet_at_starting_line_l1787_178732


namespace largest_and_smallest_value_of_expression_l1787_178767

theorem largest_and_smallest_value_of_expression
  (w x y z : ℝ)
  (h1 : w + x + y + z = 0)
  (h2 : w^7 + x^7 + y^7 + z^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 :=
sorry

end largest_and_smallest_value_of_expression_l1787_178767


namespace number_of_recipes_needed_l1787_178730

def numStudents : ℕ := 150
def avgCookiesPerStudent : ℕ := 3
def cookiesPerRecipe : ℕ := 18
def attendanceDrop : ℝ := 0.40

theorem number_of_recipes_needed (n : ℕ) (c : ℕ) (r : ℕ) (d : ℝ) : 
  n = numStudents →
  c = avgCookiesPerStudent →
  r = cookiesPerRecipe →
  d = attendanceDrop →
  ∃ (recipes : ℕ), recipes = 15 :=
by
  intros
  sorry

end number_of_recipes_needed_l1787_178730


namespace calculate_overhead_cost_l1787_178751

noncomputable def overhead_cost (prod_cost revenue_cost : ℕ) (num_performances : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost - num_performances * prod_cost

theorem calculate_overhead_cost :
  overhead_cost 7000 16000 9 (9 * 16000) = 81000 :=
by
  sorry

end calculate_overhead_cost_l1787_178751


namespace solve_for_y_l1787_178798

theorem solve_for_y : ∀ (y : ℝ), (3 / 4 - 5 / 8 = 1 / y) → y = 8 :=
by
  intros y h
  sorry

end solve_for_y_l1787_178798


namespace lines_intersect_l1787_178785

noncomputable def line1 (t : ℚ) : ℚ × ℚ :=
(1 + 2 * t, 2 - 3 * t)

noncomputable def line2 (u : ℚ) : ℚ × ℚ :=
(-1 + 3 * u, 4 + u)

theorem lines_intersect :
  ∃ t u : ℚ, line1 t = (-5 / 11, 46 / 11) ∧ line2 u = (-5 / 11, 46 / 11) :=
sorry

end lines_intersect_l1787_178785


namespace additional_spending_required_l1787_178782

def cost_of_chicken : ℝ := 1.5 * 6.00
def cost_of_lettuce : ℝ := 3.00
def cost_of_cherry_tomatoes : ℝ := 2.50
def cost_of_sweet_potatoes : ℝ := 4 * 0.75
def cost_of_broccoli : ℝ := 2 * 2.00
def cost_of_brussel_sprouts : ℝ := 2.50
def total_cost : ℝ := cost_of_chicken + cost_of_lettuce + cost_of_cherry_tomatoes + cost_of_sweet_potatoes + cost_of_broccoli + cost_of_brussel_sprouts
def minimum_spending_for_free_delivery : ℝ := 35.00
def additional_amount_needed : ℝ := minimum_spending_for_free_delivery - total_cost

theorem additional_spending_required : additional_amount_needed = 11.00 := by
  sorry

end additional_spending_required_l1787_178782


namespace find_x_minus_y_l1787_178760

-- Variables and conditions
variables (x y : ℝ)
def abs_x_eq_3 := abs x = 3
def y_sq_eq_one_fourth := y^2 = 1 / 4
def x_plus_y_neg := x + y < 0

-- Proof problem stating that x - y must equal one of the two possible values
theorem find_x_minus_y (h1 : abs x = 3) (h2 : y^2 = 1 / 4) (h3 : x + y < 0) : 
  x - y = -7 / 2 ∨ x - y = -5 / 2 :=
  sorry

end find_x_minus_y_l1787_178760


namespace value_of_x_l1787_178788

theorem value_of_x (x y : ℕ) (h₁ : x / y = 12 / 5) (h₂ : y = 25) : x = 60 :=
sorry

end value_of_x_l1787_178788


namespace drug_price_reduction_l1787_178786

theorem drug_price_reduction :
  ∃ x : ℝ, 56 * (1 - x)^2 = 31.5 :=
by
  sorry

end drug_price_reduction_l1787_178786


namespace ceil_neg_sqrt_frac_l1787_178799

theorem ceil_neg_sqrt_frac :
  (Int.ceil (-Real.sqrt (64 / 9))) = -2 := 
sorry

end ceil_neg_sqrt_frac_l1787_178799


namespace increase_average_l1787_178745

variable (total_runs : ℕ) (innings : ℕ) (average : ℕ) (new_runs : ℕ) (x : ℕ)

theorem increase_average (h1 : innings = 10) 
                         (h2 : average = 30) 
                         (h3 : total_runs = average * innings) 
                         (h4 : new_runs = 74) 
                         (h5 : total_runs + new_runs = (average + x) * (innings + 1)) :
    x = 4 := 
sorry

end increase_average_l1787_178745


namespace system_solution_is_unique_l1787_178748

theorem system_solution_is_unique
  (a b : ℝ)
  (h1 : 2 - a * 5 = -1)
  (h2 : b + 3 * 5 = 8) :
  (∃ m n : ℝ, 2 * (m + n) - a * (m - n) = -1 ∧ b * (m + n) + 3 * (m - n) = 8 ∧ m = 3 ∧ n = -2) :=
by
  sorry

end system_solution_is_unique_l1787_178748


namespace sample_size_is_15_l1787_178752

-- Define the given conditions as constants and assumptions within the Lean environment.
def total_employees := 750
def young_workers := 350
def middle_aged_workers := 250
def elderly_workers := 150
def sample_young_workers := 7

-- Define the proposition that given these conditions, the sample size is 15.
theorem sample_size_is_15 : ∃ n : ℕ, (7 / n = 350 / 750) ∧ n = 15 := by
  sorry

end sample_size_is_15_l1787_178752


namespace solution_verification_l1787_178773

-- Define the differential equation
def diff_eq (y y' y'': ℝ → ℝ) : Prop :=
  ∀ x, y'' x - 4 * y' x + 5 * y x = 2 * Real.cos x + 6 * Real.sin x

-- General solution form
def general_solution (C₁ C₂ : ℝ) (y: ℝ → ℝ) : Prop :=
  ∀ x, y x = Real.exp (2 * x) * (C₁ * Real.cos x + C₂ * Real.sin x) + Real.cos x + 1/2 * Real.sin x

-- Proof problem statement
theorem solution_verification (C₁ C₂ : ℝ) (y y' y'': ℝ → ℝ) :
  (∀ x, y' x = deriv y x) →
  (∀ x, y'' x = deriv (deriv y) x) →
  diff_eq y y' y'' →
  general_solution C₁ C₂ y :=
by
  intros h1 h2 h3
  sorry

end solution_verification_l1787_178773


namespace ratio_of_side_length_to_radius_l1787_178741

theorem ratio_of_side_length_to_radius (r s : ℝ) (c d : ℝ) 
  (h1 : s = 2 * r)
  (h2 : s^2 = (c / d) * (s^2 - π * r^2)) : 
  (s / r) = (Real.sqrt (c * π) / Real.sqrt (d - c)) := by
  sorry

end ratio_of_side_length_to_radius_l1787_178741


namespace compute_complex_power_l1787_178750

theorem compute_complex_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 :=
by
  sorry

end compute_complex_power_l1787_178750


namespace alex_has_more_pens_than_jane_l1787_178740

-- Definitions based on the conditions
def starting_pens_alex : ℕ := 4
def pens_jane_after_month : ℕ := 16

-- Alex's pen count after each week
def pens_alex_after_week (w : ℕ) : ℕ :=
  starting_pens_alex * 2 ^ w

-- Proof statement
theorem alex_has_more_pens_than_jane :
  pens_alex_after_week 4 - pens_jane_after_month = 16 := by
  sorry

end alex_has_more_pens_than_jane_l1787_178740


namespace children_multiple_of_four_l1787_178715

theorem children_multiple_of_four (C : ℕ) 
  (h_event : ∃ (A : ℕ) (T : ℕ), A = 12 ∧ T = 4 ∧ 12 % T = 0 ∧ C % T = 0) : ∃ k : ℕ, C = 4 * k :=
by
  obtain ⟨A, T, hA, hT, hA_div, hC_div⟩ := h_event
  rw [hA, hT] at *
  sorry

end children_multiple_of_four_l1787_178715


namespace smallest_n_condition_l1787_178742

def pow_mod (a b m : ℕ) : ℕ := a^(b % m)

def n (r s : ℕ) : ℕ := 2^r - 16^s

def r_condition (r : ℕ) : Prop := ∃ k : ℕ, r = 3 * k + 1

def s_condition (s : ℕ) : Prop := ∃ h : ℕ, s = 3 * h + 2

theorem smallest_n_condition (r s : ℕ) (hr : r_condition r) (hs : s_condition s) :
  (n r s) % 7 = 5 → (n r s) = 768 := sorry

end smallest_n_condition_l1787_178742


namespace second_cart_travel_distance_l1787_178720

-- Given definitions:
def first_cart_first_term : ℕ := 6
def first_cart_common_difference : ℕ := 8
def second_cart_first_term : ℕ := 7
def second_cart_common_difference : ℕ := 9

-- Given times:
def time_first_cart : ℕ := 35
def time_second_cart : ℕ := 33

-- Arithmetic series sum formula
def arithmetic_series_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

-- Total distance traveled by the second cart
noncomputable def distance_second_cart : ℕ :=
  arithmetic_series_sum second_cart_first_term second_cart_common_difference time_second_cart

-- Theorem to prove the distance traveled by the second cart
theorem second_cart_travel_distance : distance_second_cart = 4983 :=
  sorry

end second_cart_travel_distance_l1787_178720


namespace isosceles_triangle_perimeter_l1787_178721

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 7) :
  ∃ (c : ℝ), (a = b ∧ 7 = c ∨ a = c ∧ 7 = b) ∧ a + b + c = 17 :=
by
  use 17
  sorry

end isosceles_triangle_perimeter_l1787_178721


namespace total_cookies_l1787_178789

-- Define the number of bags and the number of cookies per bag
def bags : ℕ := 37
def cookies_per_bag : ℕ := 19

-- State the theorem
theorem total_cookies : bags * cookies_per_bag = 703 :=
by
  sorry

end total_cookies_l1787_178789


namespace proof_problem_l1787_178718

variables (Books : Type) (Available : Books -> Prop)

def all_books_available : Prop := ∀ b : Books, Available b
def some_books_not_available : Prop := ∃ b : Books, ¬ Available b
def not_all_books_available : Prop := ¬ all_books_available Books Available

theorem proof_problem (h : ¬ all_books_available Books Available) : 
  some_books_not_available Books Available ∧ not_all_books_available Books Available :=
by 
  sorry

end proof_problem_l1787_178718


namespace cos_a2_plus_a8_eq_neg_half_l1787_178774

noncomputable def a_n (n : ℕ) (a₁ d : ℝ) : ℝ :=
  a₁ + (n - 1) * d

theorem cos_a2_plus_a8_eq_neg_half 
  (a₁ d : ℝ) 
  (h : a₁ + a_n 5 a₁ d + a_n 9 a₁ d = 5 * Real.pi)
  : Real.cos (a_n 2 a₁ d + a_n 8 a₁ d) = -1 / 2 :=
by
  sorry

end cos_a2_plus_a8_eq_neg_half_l1787_178774


namespace find_x_l1787_178701

theorem find_x (x : ℝ) (h : (2012 + x)^2 = x^2) : x = -1006 :=
by
  sorry

end find_x_l1787_178701


namespace largest_integer_y_l1787_178714

theorem largest_integer_y (y : ℤ) : (y / 4 + 3 / 7 : ℝ) < 9 / 4 → y ≤ 7 := by
  intros h
  sorry -- Proof needed

end largest_integer_y_l1787_178714


namespace maximum_weight_truck_can_carry_l1787_178716

-- Definitions for the conditions.
def weight_boxes : Nat := 100 * 100
def weight_crates : Nat := 10 * 60
def weight_sacks : Nat := 50 * 50
def weight_additional_bags : Nat := 10 * 40

-- Summing up all the weights.
def total_weight : Nat :=
  weight_boxes + weight_crates + weight_sacks + weight_additional_bags

-- The theorem stating the maximum weight.
theorem maximum_weight_truck_can_carry : total_weight = 13500 := by
  sorry

end maximum_weight_truck_can_carry_l1787_178716


namespace haley_seeds_l1787_178707

theorem haley_seeds (total_seeds seeds_big_garden total_small_gardens seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 56)
  (h2 : seeds_big_garden = 35)
  (h3 : total_small_gardens = 7)
  (h4 : total_seeds - seeds_big_garden = 21)
  (h5 : 21 / total_small_gardens = seeds_per_small_garden) :
  seeds_per_small_garden = 3 :=
by sorry

end haley_seeds_l1787_178707


namespace general_formula_and_arithmetic_sequence_l1787_178780

noncomputable def S_n (n : ℕ) : ℕ := 3 * n ^ 2 - 2 * n
noncomputable def a_n (n : ℕ) : ℕ := S_n n - S_n (n - 1)

theorem general_formula_and_arithmetic_sequence :
  (∀ n : ℕ, a_n n = 6 * n - 5) ∧
  (∀ n : ℕ, (n ≥ 2 → a_n n - a_n (n - 1) = 6) ∧ (a_n 1 = 1)) :=
by
  sorry

end general_formula_and_arithmetic_sequence_l1787_178780


namespace find_natural_number_l1787_178775

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_natural_number (n : ℕ) : sum_of_digits (2 ^ n) = 5 ↔ n = 5 := by
  sorry

end find_natural_number_l1787_178775


namespace statement_a_statement_b_statement_c_l1787_178756

theorem statement_a (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  0 ≤ a ∧ a ≤ 4 := sorry

theorem statement_b (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  -1 ≤ b ∧ b ≤ 3 := sorry

theorem statement_c (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  -2 ≤ 3 * a - 2 * b ∧ 3 * a - 2 * b ≤ 10 := sorry

end statement_a_statement_b_statement_c_l1787_178756


namespace math_question_l1787_178738

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end math_question_l1787_178738


namespace total_widgets_sold_after_15_days_l1787_178796

def widgets_sold_day_n (n : ℕ) : ℕ :=
  2 + (n - 1) * 3

def sum_of_widgets (n : ℕ) : ℕ :=
  n * (2 + widgets_sold_day_n n) / 2

theorem total_widgets_sold_after_15_days : 
  sum_of_widgets 15 = 345 :=
by
  -- Prove the arithmetic sequence properties and sum.
  sorry

end total_widgets_sold_after_15_days_l1787_178796


namespace james_prom_cost_l1787_178719

def total_cost (ticket_cost dinner_cost tip_percent limo_cost_per_hour limo_hours tuxedo_cost persons : ℕ) : ℕ :=
  (ticket_cost * persons) +
  ((dinner_cost * persons) + (tip_percent * dinner_cost * persons) / 100) +
  (limo_cost_per_hour * limo_hours) + tuxedo_cost

theorem james_prom_cost :
  total_cost 100 120 30 80 8 150 4 = 1814 :=
by
  sorry

end james_prom_cost_l1787_178719


namespace Emmy_money_l1787_178725

theorem Emmy_money {Gerry_money cost_per_apple number_of_apples Emmy_money : ℕ} 
    (h1 : Gerry_money = 100)
    (h2 : cost_per_apple = 2) 
    (h3 : number_of_apples = 150) 
    (h4 : number_of_apples * cost_per_apple = Gerry_money + Emmy_money) :
    Emmy_money = 200 :=
by
   sorry

end Emmy_money_l1787_178725


namespace carpet_length_l1787_178755

-- Define the conditions as hypotheses
def width_of_carpet : ℝ := 4
def area_of_living_room : ℝ := 60

-- Formalize the corresponding proof problem
theorem carpet_length (h : 60 = width_of_carpet * length) : length = 15 :=
sorry

end carpet_length_l1787_178755


namespace inequality_proof_l1787_178705

theorem inequality_proof (x y z : ℝ) (hx : 2 < x) (hx4 : x < 4) (hy : 2 < y) (hy4 : y < 4) (hz : 2 < z) (hz4 : z < 4) :
  (x / (y^2 - z) + y / (z^2 - x) + z / (x^2 - y)) > 1 :=
by
  sorry

end inequality_proof_l1787_178705


namespace approx_log_base_5_10_l1787_178777

noncomputable def log_base (b a : ℝ) : ℝ := (Real.log a) / (Real.log b)

theorem approx_log_base_5_10 :
  let lg2 := 0.301
  let lg3 := 0.477
  let lg10 := 1
  let lg5 := lg10 - lg2
  log_base 5 10 = 10 / 7 :=
  sorry

end approx_log_base_5_10_l1787_178777


namespace intersection_A_B_l1787_178709

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | x < -1 ∨ x > 4}

theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | 4 < x ∧ x < 7} :=
by
  sorry

end intersection_A_B_l1787_178709


namespace pyramid_volume_l1787_178778

theorem pyramid_volume
  (a b c : ℝ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 5)
  (angle_lateral : ℝ) (h4 : angle_lateral = 45) :
  ∃ (V : ℝ), V = 6 :=
by
  -- the proof steps would be included here
  sorry

end pyramid_volume_l1787_178778


namespace line_circle_separate_l1787_178764

def point_inside_circle (x0 y0 a : ℝ) : Prop :=
  x0^2 + y0^2 < a^2

def not_center_of_circle (x0 y0 : ℝ) : Prop :=
  x0^2 + y0^2 ≠ 0

theorem line_circle_separate (x0 y0 a : ℝ) (h1 : point_inside_circle x0 y0 a) (h2 : a > 0) (h3 : not_center_of_circle x0 y0) :
  ∀ (x y : ℝ), ¬ (x0 * x + y0 * y = a^2 ∧ x^2 + y^2 = a^2) :=
by
  sorry

end line_circle_separate_l1787_178764


namespace third_side_length_l1787_178726

noncomputable def calc_third_side (a b : ℕ) (hypotenuse : Bool) : ℝ :=
if hypotenuse then
  Real.sqrt (a^2 + b^2)
else
  Real.sqrt (abs (a^2 - b^2))

theorem third_side_length (a b : ℕ) (h_right_triangle : (a = 8 ∧ b = 15)) :
  calc_third_side a b true = 17 ∨ calc_third_side 15 8 false = Real.sqrt 161 :=
by {
  sorry
}

end third_side_length_l1787_178726


namespace base9_subtraction_multiple_of_seven_l1787_178757

theorem base9_subtraction_multiple_of_seven (b : ℕ) (h1 : 0 ≤ b ∧ b ≤ 9) 
(h2 : (3 * 9^6 + 1 * 9^5 + 5 * 9^4 + 4 * 9^3 + 6 * 9^2 + 7 * 9^1 + 2 * 9^0) - b % 7 = 0) : b = 0 :=
sorry

end base9_subtraction_multiple_of_seven_l1787_178757


namespace arrangement_valid_l1787_178706

def unique_digits (a b c d e f : Nat) : Prop :=
  (a = 4) ∧ (b = 1) ∧ (c = 2) ∧ (d = 5) ∧ (e = 6) ∧ (f = 3)

def sum_15 (x y z : Nat) : Prop :=
  x + y + z = 15

theorem arrangement_valid :
  ∃ a b c d e f : Nat, unique_digits a b c d e f ∧
  sum_15 a d e ∧
  sum_15 d b f ∧
  sum_15 f e c ∧
  sum_15 a b c ∧
  sum_15 a e f ∧
  sum_15 b d c :=
sorry

end arrangement_valid_l1787_178706


namespace maxwell_walking_speed_l1787_178797

theorem maxwell_walking_speed :
  ∀ (distance_between_homes : ℕ)
    (brad_speed : ℕ)
    (middle_travel_maxwell : ℕ)
    (middle_distance : ℕ),
    distance_between_homes = 36 →
    brad_speed = 4 →
    middle_travel_maxwell = 12 →
    middle_distance = 18 →
    (middle_travel_maxwell : ℕ) / (8 : ℕ) = (middle_distance - middle_travel_maxwell) / brad_speed :=
  sorry

end maxwell_walking_speed_l1787_178797


namespace reading_homework_pages_eq_three_l1787_178735

-- Define the conditions
def pages_of_math_homework : ℕ := 7
def difference : ℕ := 4

-- Define what we need to prove
theorem reading_homework_pages_eq_three (x : ℕ) (h : x + difference = pages_of_math_homework) : x = 3 := by
  sorry

end reading_homework_pages_eq_three_l1787_178735


namespace shirt_cost_l1787_178700

def cost_of_jeans_and_shirts (J S : ℝ) : Prop := (3 * J + 2 * S = 69) ∧ (2 * J + 3 * S = 81)

theorem shirt_cost (J S : ℝ) (h : cost_of_jeans_and_shirts J S) : S = 21 :=
by {
  sorry
}

end shirt_cost_l1787_178700


namespace find_y_l1787_178728

theorem find_y (a b y : ℝ) (h1 : s = (3 * a) ^ (2 * b)) (h2 : s = 5 * (a ^ b) * (y ^ b))
  (h3 : 0 < a) (h4 : 0 < b) : 
  y = 9 * a / 5 := by
  sorry

end find_y_l1787_178728


namespace age_of_B_l1787_178770

theorem age_of_B (A B C : ℕ) 
  (h1 : (A + B + C) / 3 = 22)
  (h2 : (A + B) / 2 = 18)
  (h3 : (B + C) / 2 = 25) : 
  B = 20 := 
by
  sorry

end age_of_B_l1787_178770


namespace mean_three_numbers_l1787_178727

open BigOperators

theorem mean_three_numbers (a b c : ℝ) (s : Finset ℝ) (h₀ : s.card = 20)
  (h₁ : (∑ x in s, x) / 20 = 45) 
  (h₂ : (∑ x in s ∪ {a, b, c}, x) / 23 = 50) : 
  (a + b + c) / 3 = 250 / 3 :=
by
  sorry

end mean_three_numbers_l1787_178727


namespace prob_less_than_8_prob_at_least_7_l1787_178739

def prob_9_or_above : ℝ := 0.56
def prob_8 : ℝ := 0.22
def prob_7 : ℝ := 0.12

theorem prob_less_than_8 : prob_7 + (1 - prob_9_or_above - prob_8) = 0.22 := 
sorry

theorem prob_at_least_7 : prob_9_or_above + prob_8 + prob_7 = 0.9 := 
sorry

end prob_less_than_8_prob_at_least_7_l1787_178739


namespace solve_for_a_l1787_178702

theorem solve_for_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a < 13) (h3 : (51^2012 + a) % 13 = 0) : a = 12 :=
by 
  sorry

end solve_for_a_l1787_178702


namespace correctFractions_equivalence_l1787_178737

def correctFractions: List (ℕ × ℕ) := [(26, 65), (16, 64), (19, 95), (49, 98)]

def isValidCancellation (num den: ℕ): Prop :=
  ∃ n₁ n₂ n₃ d₁ d₂ d₃: ℕ, 
    num = 10 * n₁ + n₂ ∧
    den = 10 * d₁ + d₂ ∧
    ((n₁ = d₁ ∧ n₂ = d₂) ∨ (n₁ = d₃ ∧ n₃ = d₂)) ∧
    n₁ ≠ 0 ∧ n₂ ≠ 0 ∧ d₁ ≠ 0 ∧ d₂ ≠ 0

theorem correctFractions_equivalence : 
  ∀ (frac : ℕ × ℕ), frac ∈ correctFractions → 
    ∃ a b: ℕ, correctFractions = [(a, b)] ∧ 
      isValidCancellation a b := sorry

end correctFractions_equivalence_l1787_178737


namespace circle_area_l1787_178794

-- Let r be the radius of the circle
-- The circumference of the circle is given by 2 * π * r, which is 36 cm
-- We need to prove that given this condition, the area of the circle is 324/π square centimeters

theorem circle_area (r : Real) (h : 2 * Real.pi * r = 36) : Real.pi * r^2 = 324 / Real.pi :=
by
  sorry

end circle_area_l1787_178794


namespace valid_passwords_count_l1787_178747

-- Define the total number of unrestricted passwords (each digit can be 0-9)
def total_passwords := 10^5

-- Define the number of restricted passwords (those starting with the sequence 8,3,2)
def restricted_passwords := 10^2

-- State the main theorem to be proved
theorem valid_passwords_count : total_passwords - restricted_passwords = 99900 := by
  sorry

end valid_passwords_count_l1787_178747


namespace highest_mean_possible_l1787_178744

def max_arithmetic_mean (g : Matrix (Fin 3) (Fin 3) ℕ) : ℚ := 
  let mean (a b c d : ℕ) : ℚ := (a + b + c + d : ℚ) / 4
  let circles := [
    mean (g 0 0) (g 0 1) (g 1 0) (g 1 1),
    mean (g 0 1) (g 0 2) (g 1 1) (g 1 2),
    mean (g 1 0) (g 1 1) (g 2 0) (g 2 1),
    mean (g 1 1) (g 1 2) (g 2 1) (g 2 2)
  ]
  (circles.sum / 4)

theorem highest_mean_possible :
  ∃ g : Matrix (Fin 3) (Fin 3) ℕ, 
  (∀ i j, 1 ≤ g i j ∧ g i j ≤ 9) ∧ 
  max_arithmetic_mean g = 6.125 :=
by
  sorry

end highest_mean_possible_l1787_178744


namespace temperature_rise_result_l1787_178723

def initial_temperature : ℤ := -2
def rise : ℤ := 3

theorem temperature_rise_result : initial_temperature + rise = 1 := 
by 
  sorry

end temperature_rise_result_l1787_178723


namespace tangent_line_equation_l1787_178783

theorem tangent_line_equation :
  ∃ (P : ℝ × ℝ) (m : ℝ), 
  P = (-2, 15) ∧ m = 2 ∧ 
  (∀ (x y : ℝ), (y = x^3 - 10 * x + 3) → (y - 15 = 2 * (x + 2))) :=
sorry

end tangent_line_equation_l1787_178783


namespace total_bill_l1787_178791

theorem total_bill (total_friends : ℕ) (extra_payment : ℝ) (total_bill : ℝ) (paid_by_friends : ℝ) :
  total_friends = 8 → extra_payment = 2.50 →
  (7 * ((total_bill / total_friends) + extra_payment)) = total_bill →
  total_bill = 140 :=
by
  intros h1 h2 h3
  sorry

end total_bill_l1787_178791


namespace calculate_hidden_dots_l1787_178708

def sum_faces_of_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6

def number_of_dice : ℕ := 4
def total_sum_of_dots : ℕ := number_of_dice * sum_faces_of_die

def visible_faces : List (ℕ × String) :=
  [(1, "red"), (1, "none"), (2, "none"), (2, "blue"),
   (3, "none"), (4, "none"), (5, "none"), (6, "none")]

def adjust_face_value (value : ℕ) (color : String) : ℕ :=
  match color with
  | "red" => 2 * value
  | "blue" => 2 * value
  | _ => value

def visible_sum : ℕ :=
  visible_faces.foldl (fun acc (face) => acc + adjust_face_value face.1 face.2) 0

theorem calculate_hidden_dots :
  (total_sum_of_dots - visible_sum) = 57 :=
sorry

end calculate_hidden_dots_l1787_178708


namespace x0_range_l1787_178729

noncomputable def f (x : ℝ) := (1 / 2) ^ x - Real.log x

theorem x0_range (x0 : ℝ) (h : f x0 > 1 / 2) : 0 < x0 ∧ x0 < 1 :=
by
  sorry

end x0_range_l1787_178729


namespace arcsin_eq_solution_domain_l1787_178704

open Real

theorem arcsin_eq_solution_domain (x : ℝ) (hx1 : abs (x * sqrt 5 / 3) ≤ 1)
  (hx2 : abs (x * sqrt 5 / 6) ≤ 1)
  (hx3 : abs (7 * x * sqrt 5 / 18) ≤ 1) :
  arcsin (x * sqrt 5 / 3) + arcsin (x * sqrt 5 / 6) = arcsin (7 * x * sqrt 5 / 18) ↔ 
  x = 0 ∨ x = 8 / 7 ∨ x = -8 / 7 := sorry

end arcsin_eq_solution_domain_l1787_178704


namespace fraction_irreducible_l1787_178711

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 1) = 1 :=
sorry

end fraction_irreducible_l1787_178711
