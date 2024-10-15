import Mathlib

namespace NUMINAMATH_GPT_speed_of_stream_l2377_237769

theorem speed_of_stream (v : ℝ) (h1 : 22 > 0) (h2 : 8 > 0) (h3 : 216 = (22 + v) * 8) : v = 5 := 
by 
  sorry

end NUMINAMATH_GPT_speed_of_stream_l2377_237769


namespace NUMINAMATH_GPT_two_zeros_range_l2377_237796

noncomputable def f (x k : ℝ) : ℝ := x * Real.exp x - k

theorem two_zeros_range (k : ℝ) : -1 / Real.exp 1 < k ∧ k < 0 → ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 k = 0 ∧ f x2 k = 0 :=
by
  sorry

end NUMINAMATH_GPT_two_zeros_range_l2377_237796


namespace NUMINAMATH_GPT_sum_first_5_terms_l2377_237789

variable {a : ℕ → ℝ}
variable (h : 2 * a 2 = a 1 + 3)

theorem sum_first_5_terms (a : ℕ → ℝ) (h : 2 * a 2 = a 1 + 3) : 
  (a 1 + a 2 + a 3 + a 4 + a 5) = 15 :=
sorry

end NUMINAMATH_GPT_sum_first_5_terms_l2377_237789


namespace NUMINAMATH_GPT_yellow_balls_count_l2377_237713

theorem yellow_balls_count (total_balls white_balls green_balls red_balls purple_balls : ℕ)
  (h_total : total_balls = 100)
  (h_white : white_balls = 20)
  (h_green : green_balls = 30)
  (h_red : red_balls = 37)
  (h_purple : purple_balls = 3)
  (h_prob : ((white_balls + green_balls + (total_balls - white_balls - green_balls - red_balls - purple_balls)) / total_balls : ℝ) = 0.6) :
  (total_balls - white_balls - green_balls - red_balls - purple_balls = 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_yellow_balls_count_l2377_237713


namespace NUMINAMATH_GPT_original_bales_correct_l2377_237710

-- Definitions
def total_bales_now : Nat := 54
def bales_stacked_today : Nat := 26
def bales_originally_in_barn : Nat := total_bales_now - bales_stacked_today

-- Theorem statement
theorem original_bales_correct :
  bales_originally_in_barn = 28 :=
by {
  -- We will prove this later
  sorry
}

end NUMINAMATH_GPT_original_bales_correct_l2377_237710


namespace NUMINAMATH_GPT_ice_bag_cost_correct_l2377_237735

def total_cost_after_discount (cost_small cost_large : ℝ) (num_bags num_small : ℕ) (discount_rate : ℝ) : ℝ :=
  let num_large := num_bags - num_small
  let total_cost_before_discount := num_small * cost_small + num_large * cost_large
  let discount := discount_rate * total_cost_before_discount
  total_cost_before_discount - discount

theorem ice_bag_cost_correct :
  total_cost_after_discount 0.80 1.46 30 18 0.12 = 28.09 :=
by
  sorry

end NUMINAMATH_GPT_ice_bag_cost_correct_l2377_237735


namespace NUMINAMATH_GPT_hiker_speed_correct_l2377_237771

variable (hikerSpeed : ℝ)
variable (cyclistSpeed : ℝ := 15)
variable (cyclistTravelTime : ℝ := 5 / 60)  -- Converted 5 minutes to hours
variable (hikerCatchUpTime : ℝ := 13.75 / 60)  -- Converted 13.75 minutes to hours
variable (cyclistDistance : ℝ := cyclistSpeed * cyclistTravelTime)

theorem hiker_speed_correct :
  (hikerSpeed * hikerCatchUpTime = cyclistDistance) →
  hikerSpeed = 60 / 11 :=
by
  intro hiker_eq_cyclist_distance
  sorry

end NUMINAMATH_GPT_hiker_speed_correct_l2377_237771


namespace NUMINAMATH_GPT_product_of_repeating_decimal_l2377_237775

   -- Definitions
   def repeating_decimal : ℚ := 456 / 999  -- 0.\overline{456}

   -- Problem Statement
   theorem product_of_repeating_decimal (t : ℚ) (h : t = repeating_decimal) : (t * 7) = 1064 / 333 :=
   by
     sorry
   
end NUMINAMATH_GPT_product_of_repeating_decimal_l2377_237775


namespace NUMINAMATH_GPT_fraction_of_third_is_eighth_l2377_237768

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_fraction_of_third_is_eighth_l2377_237768


namespace NUMINAMATH_GPT_sum_terms_a1_a17_l2377_237786

theorem sum_terms_a1_a17 (S : ℕ → ℤ) (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 2 * n - 1)
  (ha : ∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) :
  a 1 + a 17 = 29 :=
sorry

end NUMINAMATH_GPT_sum_terms_a1_a17_l2377_237786


namespace NUMINAMATH_GPT_kamal_average_marks_l2377_237726

theorem kamal_average_marks :
  let total_marks_obtained := 66 + 65 + 77 + 62 + 75 + 58
  let total_max_marks := 150 + 120 + 180 + 140 + 160 + 90
  (total_marks_obtained / total_max_marks.toFloat) * 100 = 48.0 :=
by
  sorry

end NUMINAMATH_GPT_kamal_average_marks_l2377_237726


namespace NUMINAMATH_GPT_cube_decomposition_l2377_237730

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end NUMINAMATH_GPT_cube_decomposition_l2377_237730


namespace NUMINAMATH_GPT_units_digit_3m_squared_plus_2m_l2377_237725

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_3m_squared_plus_2m : (3 * (m^2 + 2^m)) % 10 = 9 := by
  sorry

end NUMINAMATH_GPT_units_digit_3m_squared_plus_2m_l2377_237725


namespace NUMINAMATH_GPT_c_work_rate_l2377_237749

noncomputable def work_rate (days : ℕ) : ℝ := 1 / days

theorem c_work_rate (A B C: ℝ) 
  (h1 : A + B = work_rate 28) 
  (h2 : A + B + C = work_rate 21) : C = work_rate 84 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_c_work_rate_l2377_237749


namespace NUMINAMATH_GPT_car_average_speed_l2377_237756

-- Define the given conditions
def total_time_hours : ℕ := 5
def total_distance_miles : ℕ := 200

-- Define the average speed calculation
def average_speed (distance time : ℕ) : ℕ :=
  distance / time

-- State the theorem to be proved
theorem car_average_speed :
  average_speed total_distance_miles total_time_hours = 40 :=
by
  sorry

end NUMINAMATH_GPT_car_average_speed_l2377_237756


namespace NUMINAMATH_GPT_sequence_solution_l2377_237793

theorem sequence_solution (a : ℕ → ℤ) :
  a 0 = -1 →
  a 1 = 1 →
  (∀ n ≥ 2, a n = 2 * a (n - 1) + 3 * a (n - 2) + 3^n) →
  ∀ n, a n = (1 / 16) * ((4 * n - 3) * 3^(n + 1) - 7 * (-1)^n) :=
by
  -- Detailed proof steps will go here.
  sorry

end NUMINAMATH_GPT_sequence_solution_l2377_237793


namespace NUMINAMATH_GPT_deductive_reasoning_not_always_correct_l2377_237788

theorem deductive_reasoning_not_always_correct (P: Prop) (Q: Prop) 
    (h1: (P → Q) → (P → Q)) :
    (¬ (∀ P Q : Prop, (P → Q) → Q → Q)) :=
sorry

end NUMINAMATH_GPT_deductive_reasoning_not_always_correct_l2377_237788


namespace NUMINAMATH_GPT_f_periodic_with_period_4a_l2377_237739

-- Definitions 'f' and 'g' (functions on real numbers), and the given conditions:
variables {a : ℝ} (f g : ℝ → ℝ)
-- Condition on a: a ≠ 0
variable (ha : a ≠ 0)

-- Given conditions
variable (hf0 : f 0 = 1) (hga : g a = 1) (h_odd_g : ∀ x : ℝ, g x = -g (-x))

-- Functional equation
variable (h_func_eq : ∀ x y : ℝ, f (x - y) = f x * f y + g x * g y)

-- The theorem stating that f is periodic with period 4a
theorem f_periodic_with_period_4a : ∀ x : ℝ, f (x + 4 * a) = f x :=
by
  sorry

end NUMINAMATH_GPT_f_periodic_with_period_4a_l2377_237739


namespace NUMINAMATH_GPT_possible_values_of_inverse_sum_l2377_237736

open Set

theorem possible_values_of_inverse_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) : 
  ∃ s : Set ℝ, s = { x | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1 / a + 1 / b) } ∧ 
  s = Ici 2 :=
sorry

end NUMINAMATH_GPT_possible_values_of_inverse_sum_l2377_237736


namespace NUMINAMATH_GPT_ticket_count_l2377_237712

theorem ticket_count (x y : ℕ) 
  (h1 : x + y = 35)
  (h2 : 24 * x + 18 * y = 750) : 
  x = 20 ∧ y = 15 :=
by
  sorry

end NUMINAMATH_GPT_ticket_count_l2377_237712


namespace NUMINAMATH_GPT_perpendicular_line_eq_l2377_237762

theorem perpendicular_line_eq :
  ∃ (A B C : ℝ), (A * 0 + B * 4 + C = 0) ∧ (A = 3) ∧ (B = 1) ∧ (C = -4) ∧ (3 * 1 + 1 * -3 = 0) :=
sorry

end NUMINAMATH_GPT_perpendicular_line_eq_l2377_237762


namespace NUMINAMATH_GPT_continuous_sum_m_l2377_237763

noncomputable def g : ℝ → ℝ → ℝ
| x, m => if x < m then x^2 + 4 else 3 * x + 6

theorem continuous_sum_m :
  ∀ m1 m2 : ℝ, (∀ m : ℝ, (g m m1 = g m m2) → g m (m1 + m2) = g m m1 + g m m2) →
  m1 + m2 = 3 :=
sorry

end NUMINAMATH_GPT_continuous_sum_m_l2377_237763


namespace NUMINAMATH_GPT_car_rental_cost_l2377_237758

variable (x : ℝ)

theorem car_rental_cost (h : 65 + 0.40 * 325 = x * 325) : x = 0.60 :=
by 
  sorry

end NUMINAMATH_GPT_car_rental_cost_l2377_237758


namespace NUMINAMATH_GPT_theater_total_cost_l2377_237723

theorem theater_total_cost 
  (cost_orchestra : ℕ) (cost_balcony : ℕ)
  (total_tickets : ℕ) (ticket_difference : ℕ)
  (O B : ℕ)
  (h1 : cost_orchestra = 12)
  (h2 : cost_balcony = 8)
  (h3 : total_tickets = 360)
  (h4 : ticket_difference = 140)
  (h5 : O + B = total_tickets)
  (h6 : B = O + ticket_difference) :
  12 * O + 8 * B = 3320 :=
by
  sorry

end NUMINAMATH_GPT_theater_total_cost_l2377_237723


namespace NUMINAMATH_GPT_simplify_fractions_l2377_237721

theorem simplify_fractions : 5 * (21 / 6) * (18 / -63) = -5 := by
  sorry

end NUMINAMATH_GPT_simplify_fractions_l2377_237721


namespace NUMINAMATH_GPT_probability_sum_7_is_1_over_3_l2377_237759

def odd_die : Set ℕ := {1, 3, 5}
def even_die : Set ℕ := {2, 4, 6}

noncomputable def total_outcomes : ℕ := 6 * 6

noncomputable def favorable_outcomes : ℕ := 4 + 4 + 4

noncomputable def probability_sum_7 : ℚ := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_sum_7_is_1_over_3 :
  probability_sum_7 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_sum_7_is_1_over_3_l2377_237759


namespace NUMINAMATH_GPT_soda_cost_l2377_237778

theorem soda_cost (total_cost sandwich_price : ℝ) (num_sandwiches num_sodas : ℕ) (total : total_cost = 8.38)
  (sandwich_cost : sandwich_price = 2.45) (total_sandwiches : num_sandwiches = 2) (total_sodas : num_sodas = 4) :
  ((total_cost - (num_sandwiches * sandwich_price)) / num_sodas) = 0.87 :=
by
  sorry

end NUMINAMATH_GPT_soda_cost_l2377_237778


namespace NUMINAMATH_GPT_quadrilateral_side_inequality_quadrilateral_side_inequality_if_intersect_l2377_237717

variable (a b c d : ℝ)
variable (angle_B angle_D : ℝ)
variable (d_intersect_circle : Prop)

-- Condition that angles B and D sum up to more than 180 degrees.
def angle_condition : Prop := angle_B + angle_D > 180

-- Condition for sides of the convex quadrilateral
def side_condition1 : Prop := a + c > b + d

-- Condition for the circle touching sides a, b, and c
def circle_tangent : Prop := True -- Placeholder as no function to verify this directly in Lean

theorem quadrilateral_side_inequality (h1 : angle_condition angle_B angle_D) 
                                      (h2 : circle_tangent) 
                                      (h3 : ¬ d_intersect_circle) 
                                      : a + c > b + d :=
  sorry

theorem quadrilateral_side_inequality_if_intersect (h1 : angle_condition angle_B angle_D) 
                                                   (h2 : circle_tangent) 
                                                   (h3 : d_intersect_circle) 
                                                   : a + c < b + d :=
  sorry

end NUMINAMATH_GPT_quadrilateral_side_inequality_quadrilateral_side_inequality_if_intersect_l2377_237717


namespace NUMINAMATH_GPT_add_one_gt_add_one_l2377_237781

theorem add_one_gt_add_one (a b c : ℝ) (h : a > b) : (a + c) > (b + c) :=
sorry

end NUMINAMATH_GPT_add_one_gt_add_one_l2377_237781


namespace NUMINAMATH_GPT_determine_set_B_l2377_237780
open Set

/-- Given problem conditions and goal in Lean 4 -/
theorem determine_set_B (U A B : Set ℕ) (hU : U = { x | x < 10 } )
  (hA_inter_compl_B : A ∩ (U \ B) = {1, 3, 5, 7, 9} ) :
  B = {2, 4, 6, 8} :=
by
  sorry

end NUMINAMATH_GPT_determine_set_B_l2377_237780


namespace NUMINAMATH_GPT_triangle_area_546_l2377_237787

theorem triangle_area_546 :
  ∀ (a b c : ℕ), a = 13 ∧ b = 84 ∧ c = 85 ∧ a^2 + b^2 = c^2 →
  (1 / 2 : ℝ) * (a * b) = 546 :=
by
  intro a b c
  intro h
  sorry

end NUMINAMATH_GPT_triangle_area_546_l2377_237787


namespace NUMINAMATH_GPT_percentage_of_360_is_120_l2377_237772

theorem percentage_of_360_is_120 (part whole : ℝ) (h1 : part = 120) (h2 : whole = 360) : 
  ((part / whole) * 100 = 33.33) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_360_is_120_l2377_237772


namespace NUMINAMATH_GPT_solve_for_z_l2377_237703

theorem solve_for_z (i : ℂ) (z : ℂ) (h : 3 - 5 * i * z = -2 + 5 * i * z) (h_i : i^2 = -1) :
  z = -i / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_z_l2377_237703


namespace NUMINAMATH_GPT_profit_bicycle_l2377_237743

theorem profit_bicycle (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 650) 
  (h2 : x + 2 * y = 350) : 
  x = 150 ∧ y = 100 :=
by 
  sorry

end NUMINAMATH_GPT_profit_bicycle_l2377_237743


namespace NUMINAMATH_GPT_jared_yearly_earnings_l2377_237716

theorem jared_yearly_earnings (monthly_pay_diploma : ℕ) (multiplier : ℕ) (months_in_year : ℕ)
  (h1 : monthly_pay_diploma = 4000) (h2 : multiplier = 3) (h3 : months_in_year = 12) :
  (monthly_pay_diploma * multiplier * months_in_year) = 144000 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_jared_yearly_earnings_l2377_237716


namespace NUMINAMATH_GPT_solve_problem_l2377_237750

-- Define the polynomial g(x) as given in the problem
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

-- Define the condition given in the problem
def condition (p q r s t : ℝ) : Prop := g p q r s t (-2) = -4

-- State the theorem to be proved
theorem solve_problem (p q r s t : ℝ) (h : condition p q r s t) :
  16 * p - 8 * q + 4 * r - 2 * s + t = 4 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_solve_problem_l2377_237750


namespace NUMINAMATH_GPT_find_other_number_l2377_237751

theorem find_other_number (LCM : ℕ) (HCF : ℕ) (n1 : ℕ) (n2 : ℕ) 
  (h_lcm : LCM = 2310) (h_hcf : HCF = 26) (h_n1 : n1 = 210) :
  n2 = 286 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l2377_237751


namespace NUMINAMATH_GPT_three_digit_multiple_l2377_237714

open Classical

theorem three_digit_multiple (n : ℕ) (h₁ : n % 2 = 0) (h₂ : n % 5 = 0) (h₃ : n % 3 = 0) (h₄ : 100 ≤ n) (h₅ : n < 1000) :
  120 ≤ n ∧ n ≤ 990 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_multiple_l2377_237714


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l2377_237791

theorem cyclic_sum_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_cond : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (ab + 1) / (a + b)^2 + (bc + 1) / (b + c)^2 + (ca + 1) / (c + a)^2 ≥ 3 :=
by
  -- TODO: Provide proof here
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l2377_237791


namespace NUMINAMATH_GPT_find_largest_square_area_l2377_237784

def area_of_largest_square (XY YZ XZ : ℝ) (sum_of_areas : ℝ) (right_angle : Prop) : Prop :=
  sum_of_areas = XY^2 + YZ^2 + XZ^2 + 4 * YZ^2 ∧  -- sum of areas condition
  right_angle ∧                                    -- right angle condition
  XZ^2 = XY^2 + YZ^2 ∧                             -- Pythagorean theorem
  sum_of_areas = 650 ∧                             -- total area condition
  XY = YZ                                          -- assumption for simplified solving.

theorem find_largest_square_area (XY YZ XZ : ℝ) (sum_of_areas : ℝ):
  area_of_largest_square XY YZ XZ sum_of_areas (90 = 90) → 2 * XY^2 + 5 * YZ^2 = 650 → XZ^2 = 216.67 :=
sorry

end NUMINAMATH_GPT_find_largest_square_area_l2377_237784


namespace NUMINAMATH_GPT_enumerate_A_l2377_237700

-- Define the set A according to the given conditions
def A : Set ℕ := {X : ℕ | 8 % (6 - X) = 0}

-- The equivalent proof problem
theorem enumerate_A : A = {2, 4, 5} :=
by sorry

end NUMINAMATH_GPT_enumerate_A_l2377_237700


namespace NUMINAMATH_GPT_aquarium_pufferfish_problem_l2377_237767

/-- Define the problem constants and equations -/
theorem aquarium_pufferfish_problem :
  ∃ (P S : ℕ), S = 5 * P ∧ S + P = 90 ∧ P = 15 :=
by
  sorry

end NUMINAMATH_GPT_aquarium_pufferfish_problem_l2377_237767


namespace NUMINAMATH_GPT_ellipsoid_center_and_axes_sum_l2377_237752

theorem ellipsoid_center_and_axes_sum :
  let x₀ := -2
  let y₀ := 3
  let z₀ := 1
  let A := 6
  let B := 4
  let C := 2
  x₀ + y₀ + z₀ + A + B + C = 14 := 
by
  sorry

end NUMINAMATH_GPT_ellipsoid_center_and_axes_sum_l2377_237752


namespace NUMINAMATH_GPT_polygon_exterior_angle_l2377_237774

theorem polygon_exterior_angle (n : ℕ) (h : 36 = 360 / n) : n = 10 :=
sorry

end NUMINAMATH_GPT_polygon_exterior_angle_l2377_237774


namespace NUMINAMATH_GPT_fixed_point_exists_l2377_237731

theorem fixed_point_exists : ∃ (x y : ℝ), (∀ k : ℝ, (2 * k - 1) * x - (k + 3) * y - (k - 11) = 0) ∧ x = 2 ∧ y = 3 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_fixed_point_exists_l2377_237731


namespace NUMINAMATH_GPT_valid_assignments_count_l2377_237754

noncomputable def validAssignments : Nat := sorry

theorem valid_assignments_count : validAssignments = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_valid_assignments_count_l2377_237754


namespace NUMINAMATH_GPT_jemma_total_grasshoppers_l2377_237794

def number_of_grasshoppers_on_plant : Nat := 7
def number_of_dozen_baby_grasshoppers : Nat := 2
def number_in_a_dozen : Nat := 12

theorem jemma_total_grasshoppers :
  number_of_grasshoppers_on_plant + number_of_dozen_baby_grasshoppers * number_in_a_dozen = 31 := by
  sorry

end NUMINAMATH_GPT_jemma_total_grasshoppers_l2377_237794


namespace NUMINAMATH_GPT_sufficient_condition_for_one_positive_and_one_negative_root_l2377_237711

theorem sufficient_condition_for_one_positive_and_one_negative_root (a : ℝ) (h₀ : a ≠ 0) :
  a < -1 ↔ (∃ x y : ℝ, (a * x^2 + 2 * x + 1 = 0) ∧ (a * y^2 + 2 * y + 1 = 0) ∧ x > 0 ∧ y < 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_condition_for_one_positive_and_one_negative_root_l2377_237711


namespace NUMINAMATH_GPT_student_walking_time_l2377_237783

-- Define the conditions
def total_time_walking_and_bus : ℕ := 90  -- Total time walking to school and taking the bus back home
def total_time_bus_both_ways : ℕ := 30 -- Total time taking the bus both ways

-- Calculate the time taken for walking both ways
def time_bus_one_way : ℕ := total_time_bus_both_ways / 2
def time_walking_one_way : ℕ := total_time_walking_and_bus - time_bus_one_way
def total_time_walking_both_ways : ℕ := 2 * time_walking_one_way

-- State the theorem to be proved
theorem student_walking_time :
  total_time_walking_both_ways = 150 := by
  sorry

end NUMINAMATH_GPT_student_walking_time_l2377_237783


namespace NUMINAMATH_GPT_max_positive_root_eq_l2377_237702

theorem max_positive_root_eq (b c : ℝ) (h_b : |b| ≤ 3) (h_c : |c| ≤ 3) : 
  ∃ x, x = (3 + Real.sqrt 21) / 2 ∧ x^2 + b * x + c = 0 ∧ x ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_max_positive_root_eq_l2377_237702


namespace NUMINAMATH_GPT_convert_to_scientific_notation_l2377_237753

theorem convert_to_scientific_notation (N : ℕ) (h : 2184300000 = 2184.3 * 10^6) : 
    (2184300000 : ℝ) = 2.1843 * 10^7 :=
by 
  sorry

end NUMINAMATH_GPT_convert_to_scientific_notation_l2377_237753


namespace NUMINAMATH_GPT_janet_earnings_eur_l2377_237797

noncomputable def usd_to_eur (usd : ℚ) : ℚ :=
  usd * 0.85

def janet_earnings_usd : ℚ :=
  (130 * 0.25) + (90 * 0.30) + (30 * 0.40)

theorem janet_earnings_eur : usd_to_eur janet_earnings_usd = 60.78 :=
  by
    sorry

end NUMINAMATH_GPT_janet_earnings_eur_l2377_237797


namespace NUMINAMATH_GPT_basketball_team_lineup_l2377_237795

-- Define the problem conditions
def total_players : ℕ := 12
def twins : ℕ := 2
def lineup_size : ℕ := 5
def remaining_players : ℕ := total_players - twins
def positions_to_fill : ℕ := lineup_size - twins

-- Define the combination function as provided in the standard libraries
def combination (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem translating to the proof problem
theorem basketball_team_lineup : combination remaining_players positions_to_fill = 120 := 
sorry

end NUMINAMATH_GPT_basketball_team_lineup_l2377_237795


namespace NUMINAMATH_GPT_ratio_of_means_l2377_237701

-- Variables for means
variables (xbar ybar zbar : ℝ)
-- Variables for sample sizes
variables (m n : ℕ)

-- Given conditions
def mean_x (x : ℕ) (xbar : ℝ) := ∀ i, 1 ≤ i ∧ i ≤ x → xbar = xbar
def mean_y (y : ℕ) (ybar : ℝ) := ∀ i, 1 ≤ i ∧ i ≤ y → ybar = ybar
def combined_mean (m n : ℕ) (xbar ybar zbar : ℝ) := zbar = (1/4) * xbar + (3/4) * ybar

-- Assertion to be proved
theorem ratio_of_means (h1 : mean_x m xbar) (h2 : mean_y n ybar)
  (h3 : xbar ≠ ybar) (h4 : combined_mean m n xbar ybar zbar) :
  m / n = 1 / 3 := sorry

end NUMINAMATH_GPT_ratio_of_means_l2377_237701


namespace NUMINAMATH_GPT_evaluate_expression_l2377_237792

theorem evaluate_expression (x : ℝ) (h1 : x^4 + 2 * x + 2 ≠ 0)
    (h2 : x^4 - 2 * x + 2 ≠ 0) :
    ( ( ( (x + 2) ^ 3 * (x^3 - 2 * x + 2) ^ 3 ) / ( ( x^4 + 2 * x + 2) ) ^ 3 ) ^ 3 * 
      ( ( (x - 2) ^ 3 * ( x^3 + 2 * x + 2 ) ^ 3 ) / ( ( x^4 - 2 * x + 2 ) ) ^ 3 ) ^ 3 ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2377_237792


namespace NUMINAMATH_GPT_reporters_cover_local_politics_l2377_237715

-- Definitions of percentages and total reporters
def total_reporters : ℕ := 100
def politics_coverage_percent : ℕ := 20 -- Derived from 100 - 80
def politics_reporters : ℕ := (politics_coverage_percent * total_reporters) / 100
def not_local_politics_percent : ℕ := 40
def local_politics_percent : ℕ := 60 -- Derived from 100 - 40
def local_politics_reporters : ℕ := (local_politics_percent * politics_reporters) / 100

theorem reporters_cover_local_politics :
  (local_politics_reporters * 100) / total_reporters = 12 :=
by
  exact sorry

end NUMINAMATH_GPT_reporters_cover_local_politics_l2377_237715


namespace NUMINAMATH_GPT_units_digit_product_l2377_237785

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_product (a b c : ℕ) :
  units_digit a = 7 → units_digit b = 3 → units_digit c = 9 →
  units_digit ((a * b) * c) = 9 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_units_digit_product_l2377_237785


namespace NUMINAMATH_GPT_total_students_high_school_l2377_237728

theorem total_students_high_school (students_first_grade : ℕ) (total_sample : ℕ) 
  (sample_second_grade : ℕ) (sample_third_grade : ℕ) (total_students : ℕ) 
  (h1 : students_first_grade = 600) (h2 : total_sample = 45) 
  (h3 : sample_second_grade = 20) (h4 : sample_third_grade = 10)
  (h5 : 15 = total_sample - sample_second_grade - sample_third_grade) 
  (h6 : 15 * total_students = students_first_grade * total_sample) :
  total_students = 1800 :=
sorry

end NUMINAMATH_GPT_total_students_high_school_l2377_237728


namespace NUMINAMATH_GPT_proposition_3_proposition_4_l2377_237755

variable {Line Plane : Type} -- Introduce the types for lines and planes
variable (m n : Line) (α β : Plane) -- Introduce specific lines and planes

-- Define parallel and perpendicular relations
variables {parallel : Line → Plane → Prop} {perpendicular : Line → Plane → Prop}
variables {parallel_line : Line → Line → Prop} {perpendicular_line : Line → Line → Prop}
variables {parallel_plane : Plane → Plane → Prop} {perpendicular_plane : Plane → Plane → Prop}

-- Define subset: a line n is in a plane α
variable {subset : Line → Plane → Prop}

-- Hypotheses for propositions 3 and 4
axiom prop3_hyp1 : perpendicular m α
axiom prop3_hyp2 : parallel_line m n
axiom prop3_hyp3 : parallel_plane α β

axiom prop4_hyp1 : perpendicular_line m n
axiom prop4_hyp2 : perpendicular m α
axiom prop4_hyp3 : perpendicular n β

theorem proposition_3 (h1 : perpendicular m α) (h2 : parallel_line m n) (h3 : parallel_plane α β) : perpendicular n β := sorry

theorem proposition_4 (h1 : perpendicular_line m n) (h2 : perpendicular m α) (h3 : perpendicular n β) : perpendicular_plane α β := sorry

end NUMINAMATH_GPT_proposition_3_proposition_4_l2377_237755


namespace NUMINAMATH_GPT_calculate_expression_l2377_237724

theorem calculate_expression : (235 - 2 * 3 * 5) * 7 / 5 = 287 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2377_237724


namespace NUMINAMATH_GPT_cos_arith_prog_impossible_l2377_237747

theorem cos_arith_prog_impossible
  (x y z : ℝ)
  (sin_arith_prog : 2 * Real.sin y = Real.sin x + Real.sin z) :
  ¬ (2 * Real.cos y = Real.cos x + Real.cos z) :=
by
  sorry

end NUMINAMATH_GPT_cos_arith_prog_impossible_l2377_237747


namespace NUMINAMATH_GPT_has_only_one_zero_point_l2377_237779

noncomputable def f (x a : ℝ) := (x - 1) * Real.exp x + (a / 2) * x^2

theorem has_only_one_zero_point (a : ℝ) (h : -Real.exp 1 ≤ a ∧ a ≤ 0) :
  ∃! x : ℝ, f x a = 0 :=
sorry

end NUMINAMATH_GPT_has_only_one_zero_point_l2377_237779


namespace NUMINAMATH_GPT_age_problem_solution_l2377_237720

theorem age_problem_solution :
  ∃ (a1 a2 a3 a4 a5 : ℝ),
  a1 + a2 + a3 = 54 ∧
  a5 - a4 = 5 ∧
  a3 + a4 + a5 = 78 ∧
  a2 - a1 = 7 ∧
  a1 + a5 = 44 ∧
  a1 = 13 ∧
  a2 = 20 ∧
  a3 = 21 ∧
  a4 = 26 ∧
  a5 = 31 :=
by
  -- We should skip the implementation because the solution is provided in the original problem.
  sorry

end NUMINAMATH_GPT_age_problem_solution_l2377_237720


namespace NUMINAMATH_GPT_smallest_integer_y_l2377_237744

theorem smallest_integer_y (y : ℤ) (h: y < 3 * y - 15) : y ≥ 8 :=
by sorry

end NUMINAMATH_GPT_smallest_integer_y_l2377_237744


namespace NUMINAMATH_GPT_total_photos_newspaper_l2377_237764

theorem total_photos_newspaper (pages1 pages2 photos_per_page1 photos_per_page2 : ℕ)
  (h1 : pages1 = 12) (h2 : photos_per_page1 = 2)
  (h3 : pages2 = 9) (h4 : photos_per_page2 = 3) :
  (pages1 * photos_per_page1) + (pages2 * photos_per_page2) = 51 :=
by
  sorry

end NUMINAMATH_GPT_total_photos_newspaper_l2377_237764


namespace NUMINAMATH_GPT_molar_mass_of_compound_l2377_237790

variable (total_weight : ℝ) (num_moles : ℝ)

theorem molar_mass_of_compound (h1 : total_weight = 2352) (h2 : num_moles = 8) :
    total_weight / num_moles = 294 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_molar_mass_of_compound_l2377_237790


namespace NUMINAMATH_GPT_total_pages_proof_l2377_237777

/-
Conditions:
1. Johnny's essay has 150 words.
2. Madeline's essay is double the length of Johnny's essay.
3. Timothy's essay has 30 more words than Madeline's essay.
4. One page contains 260 words.

Question:
Prove that the total number of pages do Johnny, Madeline, and Timothy's essays fill is 5.
-/

def johnny_words : ℕ := 150
def words_per_page : ℕ := 260

def madeline_words : ℕ := 2 * johnny_words
def timothy_words : ℕ := madeline_words + 30

def pages (words : ℕ) : ℕ := (words + words_per_page - 1) / words_per_page  -- division rounding up

def johnny_pages : ℕ := pages johnny_words
def madeline_pages : ℕ := pages madeline_words
def timothy_pages : ℕ := pages timothy_words

def total_pages : ℕ := johnny_pages + madeline_pages + timothy_pages

theorem total_pages_proof : total_pages = 5 :=
by sorry

end NUMINAMATH_GPT_total_pages_proof_l2377_237777


namespace NUMINAMATH_GPT_total_number_of_coins_l2377_237746

-- Definitions and conditions
def num_coins_25c := 17
def num_coins_10c := 17

-- Statement to prove
theorem total_number_of_coins : num_coins_25c + num_coins_10c = 34 := by
  sorry

end NUMINAMATH_GPT_total_number_of_coins_l2377_237746


namespace NUMINAMATH_GPT_exists_three_digit_number_cube_ends_in_777_l2377_237742

theorem exists_three_digit_number_cube_ends_in_777 :
  ∃ x : ℤ, 100 ≤ x ∧ x < 1000 ∧ x^3 % 1000 = 777 := 
sorry

end NUMINAMATH_GPT_exists_three_digit_number_cube_ends_in_777_l2377_237742


namespace NUMINAMATH_GPT_john_weekly_earnings_after_raise_l2377_237770

theorem john_weekly_earnings_after_raise (original_earnings : ℝ) (raise_percentage : ℝ) (raise_amount new_earnings : ℝ) 
  (h1 : original_earnings = 50) (h2 : raise_percentage = 60) (h3 : raise_amount = (raise_percentage / 100) * original_earnings) 
  (h4 : new_earnings = original_earnings + raise_amount) : 
  new_earnings = 80 := 
by sorry

end NUMINAMATH_GPT_john_weekly_earnings_after_raise_l2377_237770


namespace NUMINAMATH_GPT_find_a_l2377_237734

variable {a b c : ℝ}

theorem find_a 
  (h1 : (a + b + c) ^ 2 = 3 * (a ^ 2 + b ^ 2 + c ^ 2))
  (h2 : a + b + c = 12) : 
  a = 4 := 
sorry

end NUMINAMATH_GPT_find_a_l2377_237734


namespace NUMINAMATH_GPT_not_perfect_square_l2377_237773

theorem not_perfect_square (a : ℤ) : ¬ (∃ x : ℤ, a^2 + 4 = x^2) := 
sorry

end NUMINAMATH_GPT_not_perfect_square_l2377_237773


namespace NUMINAMATH_GPT_mary_avg_speed_round_trip_l2377_237738

theorem mary_avg_speed_round_trip :
  let distance_to_school := 1.5 -- in km
  let time_to_school := 45 / 60 -- in hours (converted from minutes)
  let time_back_home := 15 / 60 -- in hours (converted from minutes)
  let total_distance := 2 * distance_to_school
  let total_time := time_to_school + time_back_home
  let avg_speed := total_distance / total_time
  avg_speed = 3 := by
  -- Definitions used directly appear in the conditions.
  -- Each condition used:
  -- Mary lives 1.5 km -> distance_to_school = 1.5
  -- Time to school 45 minutes -> time_to_school = 45 / 60
  -- Time back home 15 minutes -> time_back_home = 15 / 60
  -- Route is same -> total_distance = 2 * distance_to_school, total_time = time_to_school + time_back_home
  -- Proof to show avg_speed = 3
  sorry

end NUMINAMATH_GPT_mary_avg_speed_round_trip_l2377_237738


namespace NUMINAMATH_GPT_relationship_a_b_c_l2377_237766

noncomputable def a : ℝ := Real.sin (Real.pi / 16)
noncomputable def b : ℝ := 0.25
noncomputable def c : ℝ := 2 * Real.log 2 - Real.log 3

theorem relationship_a_b_c : a < b ∧ b < c :=
by
  sorry

end NUMINAMATH_GPT_relationship_a_b_c_l2377_237766


namespace NUMINAMATH_GPT_total_jokes_after_eight_days_l2377_237729

def jokes_counted (start_jokes : ℕ) (n : ℕ) : ℕ :=
  -- Sum of initial jokes until the nth day by doubling each day
  start_jokes * (2 ^ n - 1)

theorem total_jokes_after_eight_days (jessy_jokes : ℕ) (alan_jokes : ℕ) (tom_jokes : ℕ) (emily_jokes : ℕ)
  (total_days : ℕ) (days_per_week : ℕ) :
  total_days = 5 → days_per_week = 8 →
  jessy_jokes = 11 → alan_jokes = 7 → tom_jokes = 5 → emily_jokes = 3 →
  (jokes_counted jessy_jokes (days_per_week - total_days) +
   jokes_counted alan_jokes (days_per_week - total_days) +
   jokes_counted tom_jokes (days_per_week - total_days) +
   jokes_counted emily_jokes (days_per_week - total_days)) = 806 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_jokes_after_eight_days_l2377_237729


namespace NUMINAMATH_GPT_scientific_notation_correct_l2377_237776

noncomputable def scientific_notation (x : ℕ) : Prop :=
  x = 3010000000 → 3.01 * (10 ^ 9) = 3.01 * (10 ^ 9)

theorem scientific_notation_correct : 
  scientific_notation 3010000000 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l2377_237776


namespace NUMINAMATH_GPT_mike_profit_l2377_237704

-- Definition of initial conditions
def acres_bought := 200
def cost_per_acre := 70
def fraction_sold := 1 / 2
def selling_price_per_acre := 200

-- Definitions derived from conditions
def total_cost := acres_bought * cost_per_acre
def acres_sold := acres_bought * fraction_sold
def total_revenue := acres_sold * selling_price_per_acre
def profit := total_revenue - total_cost

-- Theorem stating the question and answer tuple
theorem mike_profit : profit = 6000 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_mike_profit_l2377_237704


namespace NUMINAMATH_GPT_derivative_y_at_1_l2377_237718

-- Define the function y = x^2 + 2
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the proposition that the derivative at x=1 is 2
theorem derivative_y_at_1 : deriv f 1 = 2 :=
by sorry

end NUMINAMATH_GPT_derivative_y_at_1_l2377_237718


namespace NUMINAMATH_GPT_points_for_correct_answer_l2377_237748

theorem points_for_correct_answer
  (x y a b : ℕ)
  (hx : x - y = 7)
  (hsum : a + b = 43)
  (hw_score : a * x - b * (20 - x) = 328)
  (hz_score : a * y - b * (20 - y) = 27) :
  a = 25 := 
sorry

end NUMINAMATH_GPT_points_for_correct_answer_l2377_237748


namespace NUMINAMATH_GPT_problems_completed_l2377_237706

theorem problems_completed (p t : ℕ) (h1 : p > 15) (h2 : pt = (2 * p - 6) * (t - 3)) : p * t = 216 := 
by
  sorry

end NUMINAMATH_GPT_problems_completed_l2377_237706


namespace NUMINAMATH_GPT_cards_in_center_pile_l2377_237757

/-- Represents the number of cards in each pile initially. -/
def initial_cards (x : ℕ) : Prop := x ≥ 2

/-- Represents the state of the piles after step 2. -/
def step2 (x : ℕ) (left center right : ℕ) : Prop :=
  left = x - 2 ∧ center = x + 2 ∧ right = x

/-- Represents the state of the piles after step 3. -/
def step3 (x : ℕ) (left center right : ℕ) : Prop :=
  left = x - 2 ∧ center = x + 3 ∧ right = x - 1

/-- Represents the state of the piles after step 4. -/
def step4 (x : ℕ) (left center : ℕ) : Prop :=
  left = 2 * x - 4 ∧ center = 5

/-- Prove that after performing all steps, the number of cards in the center pile is 5. -/
theorem cards_in_center_pile (x : ℕ) :
  initial_cards x →
  (∃ l₁ c₁ r₁, step2 x l₁ c₁ r₁) →
  (∃ l₂ c₂ r₂, step3 x l₂ c₂ r₂) →
  (∃ l₃ c₃, step4 x l₃ c₃) →
  ∃ (center_final : ℕ), center_final = 5 :=
by
  sorry

end NUMINAMATH_GPT_cards_in_center_pile_l2377_237757


namespace NUMINAMATH_GPT_translation_up_by_one_l2377_237719

def initial_function (x : ℝ) : ℝ := x^2

def translated_function (x : ℝ) : ℝ := x^2 + 1

theorem translation_up_by_one (x : ℝ) : translated_function x = initial_function x + 1 :=
by sorry

end NUMINAMATH_GPT_translation_up_by_one_l2377_237719


namespace NUMINAMATH_GPT_cube_and_reciprocal_l2377_237741

theorem cube_and_reciprocal (m : ℝ) (hm : m + 1/m = 10) : m^3 + 1/m^3 = 970 := 
by
  sorry

end NUMINAMATH_GPT_cube_and_reciprocal_l2377_237741


namespace NUMINAMATH_GPT_prove_k_eq_5_l2377_237799

variable (a b k : ℕ)

theorem prove_k_eq_5 (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (a^2 - 1 - b^2) / (a * b - 1) = k) : k = 5 :=
sorry

end NUMINAMATH_GPT_prove_k_eq_5_l2377_237799


namespace NUMINAMATH_GPT_remainder_of_prime_division_l2377_237745

theorem remainder_of_prime_division
  (p : ℕ) (hp : Nat.Prime p)
  (r : ℕ) (hr : r = p % 210) 
  (hcomp : ¬ Nat.Prime r)
  (hsum : ∃ a b : ℕ, r = a^2 + b^2) : 
  r = 169 := 
sorry

end NUMINAMATH_GPT_remainder_of_prime_division_l2377_237745


namespace NUMINAMATH_GPT_polynomial_mult_of_6_l2377_237798

theorem polynomial_mult_of_6 (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) : 6 ∣ P.eval 5 := 
sorry

end NUMINAMATH_GPT_polynomial_mult_of_6_l2377_237798


namespace NUMINAMATH_GPT_second_place_team_wins_l2377_237737
open Nat

def points (wins ties : Nat) : Nat :=
  2 * wins + ties

def avg_points (p1 p2 p3 : Nat) : Nat :=
  (p1 + p2 + p3) / 3

def first_place_points := points 12 4
def elsa_team_points := points 8 10

def second_place_wins (w : Nat) : Nat :=
  w

def second_place_points (w : Nat) : Nat :=
  points w 1

theorem second_place_team_wins :
  ∃ (W : Nat), avg_points first_place_points (second_place_points W) elsa_team_points = 27 ∧ W = 13 :=
by sorry

end NUMINAMATH_GPT_second_place_team_wins_l2377_237737


namespace NUMINAMATH_GPT_triangle_angle_inequality_l2377_237707

open Real

theorem triangle_angle_inequality (α β γ α₁ β₁ γ₁ : ℝ) 
  (h1 : α + β + γ = π)
  (h2 : α₁ + β₁ + γ₁ = π) :
  (cos α₁ / sin α) + (cos β₁ / sin β) + (cos γ₁ / sin γ) 
  ≤ (cos α / sin α) + (cos β / sin β) + (cos γ / sin γ) :=
sorry

end NUMINAMATH_GPT_triangle_angle_inequality_l2377_237707


namespace NUMINAMATH_GPT_cistern_width_l2377_237760

theorem cistern_width (l d A : ℝ) (h_l: l = 5) (h_d: d = 1.25) (h_A: A = 42.5) :
  ∃ w : ℝ, 5 * w + 2 * (1.25 * 5) + 2 * (1.25 * w) = 42.5 ∧ w = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_cistern_width_l2377_237760


namespace NUMINAMATH_GPT_find_A_when_A_clubsuit_7_equals_61_l2377_237722

-- Define the operation
def clubsuit (A B : ℝ) : ℝ := 3 * A^2 + 2 * B + 7

-- Define the main problem statement
theorem find_A_when_A_clubsuit_7_equals_61 : 
  ∃ A : ℝ, clubsuit A 7 = 61 ∧ A = (2 * Real.sqrt 30) / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_A_when_A_clubsuit_7_equals_61_l2377_237722


namespace NUMINAMATH_GPT_sin_alpha_beta_l2377_237761

theorem sin_alpha_beta (a b c α β : Real) (h₁ : a * Real.cos α + b * Real.sin α + c = 0)
  (h₂ : a * Real.cos β + b * Real.sin β + c = 0) (h₃ : 0 < α) (h₄ : α < β) (h₅ : β < π) :
  Real.sin (α + β) = (2 * a * b) / (a^2 + b^2) :=
by 
  sorry

end NUMINAMATH_GPT_sin_alpha_beta_l2377_237761


namespace NUMINAMATH_GPT_perpendicular_lines_a_l2377_237708

theorem perpendicular_lines_a (a : ℝ) :
  (∀ x y : ℝ, (a * x + (1 + a) * y = 3) ∧ ((a + 1) * x + (3 - 2 * a) * y = 2) → 
     a = -1 ∨ a = 3) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_l2377_237708


namespace NUMINAMATH_GPT_ratio_of_apples_l2377_237733

/-- The store sold 32 red apples and the combined amount of red and green apples sold was 44. -/
theorem ratio_of_apples (R G : ℕ) (h1 : R = 32) (h2 : R + G = 44) : R / 4 = 8 ∧ G / 4 = 3 :=
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_ratio_of_apples_l2377_237733


namespace NUMINAMATH_GPT_cost_of_books_purchasing_plans_l2377_237727

theorem cost_of_books (x y : ℕ) (h1 : 4 * x + 2 * y = 480) (h2 : 2 * x + 3 * y = 520) : x = 50 ∧ y = 140 :=
by
  -- proof can be filled in later
  sorry

theorem purchasing_plans (a b : ℕ) (h_total_cost : 50 * a + 140 * (20 - a) ≤ 1720) (h_quantity : a ≤ 2 * (20 - b)) : (a = 12 ∧ b = 8) ∨ (a = 13 ∧ b = 7) :=
by
  -- proof can be filled in later
  sorry

end NUMINAMATH_GPT_cost_of_books_purchasing_plans_l2377_237727


namespace NUMINAMATH_GPT_geometric_sequence_a5_l2377_237782

-- Definitions based on the conditions:
variable {a : ℕ → ℝ} -- the sequence {a_n}
variable (q : ℝ) -- the common ratio of the geometric sequence

-- The sequence is geometric and terms are given:
axiom seq_geom (n m : ℕ) : a n = a 0 * q ^ n
axiom a_3_is_neg4 : a 3 = -4
axiom a_7_is_neg16 : a 7 = -16

-- The specific theorem we are proving:
theorem geometric_sequence_a5 :
  a 5 = -8 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_a5_l2377_237782


namespace NUMINAMATH_GPT_right_handed_players_total_l2377_237705

theorem right_handed_players_total (total_players throwers : ℕ) (non_throwers: ℕ := total_players - throwers)
  (left_handed_non_throwers : ℕ := non_throwers / 3)
  (right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers)
  (all_throwers_right_handed : throwers = 37)
  (total_players_55 : total_players = 55)
  (one_third_left_handed : left_handed_non_throwers = non_throwers / 3)
  (right_handed_total: ℕ := throwers + right_handed_non_throwers)
  : right_handed_total = 49 := by
  sorry

end NUMINAMATH_GPT_right_handed_players_total_l2377_237705


namespace NUMINAMATH_GPT_minimum_value_expression_l2377_237709

theorem minimum_value_expression {a b c : ℝ} :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 13 → 
  (∃ x, x = (a^2 + b^3 + c^4 + 2019) / (10 * b + 123 * c + 26) ∧ ∀ y, y ≤ x) →
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l2377_237709


namespace NUMINAMATH_GPT_simplify_fraction_l2377_237765

theorem simplify_fraction :
  (175 / 1225) * 25 = 25 / 7 :=
by
  -- Code to indicate proof steps would go here.
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2377_237765


namespace NUMINAMATH_GPT_flower_bed_profit_l2377_237740

theorem flower_bed_profit (x : ℤ) :
  (3 + x) * (10 - x) = 40 :=
sorry

end NUMINAMATH_GPT_flower_bed_profit_l2377_237740


namespace NUMINAMATH_GPT_mrs_McGillicuddy_student_count_l2377_237732

theorem mrs_McGillicuddy_student_count :
  let morning_registered := 25
  let morning_absent := 3
  let early_afternoon_registered := 24
  let early_afternoon_absent := 4
  let late_afternoon_registered := 30
  let late_afternoon_absent := 5
  let evening_registered := 35
  let evening_absent := 7
  let morning_present := morning_registered - morning_absent
  let early_afternoon_present := early_afternoon_registered - early_afternoon_absent
  let late_afternoon_present := late_afternoon_registered - late_afternoon_absent
  let evening_present := evening_registered - evening_absent
  let total_present := morning_present + early_afternoon_present + late_afternoon_present + evening_present
  total_present = 95 :=
by
  sorry

end NUMINAMATH_GPT_mrs_McGillicuddy_student_count_l2377_237732
