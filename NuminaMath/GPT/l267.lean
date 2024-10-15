import Mathlib

namespace NUMINAMATH_GPT_value_of_ab_plus_bc_plus_ca_l267_26751

theorem value_of_ab_plus_bc_plus_ca (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end NUMINAMATH_GPT_value_of_ab_plus_bc_plus_ca_l267_26751


namespace NUMINAMATH_GPT_find_line_equation_through_point_intersecting_hyperbola_l267_26791

theorem find_line_equation_through_point_intersecting_hyperbola 
  (x y : ℝ) 
  (hx : x = -2 / 3)
  (hy : (x : ℝ) = 0) : 
  ∃ k : ℝ, (∀ x y : ℝ, y = k * x - 1 → ((x^2 / 2) - (y^2 / 5) = 1)) ∧ k = 1 := 
sorry

end NUMINAMATH_GPT_find_line_equation_through_point_intersecting_hyperbola_l267_26791


namespace NUMINAMATH_GPT_tile_difference_is_11_l267_26739

-- Define the initial number of blue and green tiles
def initial_blue_tiles : ℕ := 13
def initial_green_tiles : ℕ := 6

-- Define the number of additional green tiles added as border
def additional_green_tiles : ℕ := 18

-- Define the total number of green tiles in the new figure
def total_green_tiles : ℕ := initial_green_tiles + additional_green_tiles

-- Define the total number of blue tiles in the new figure (remains the same)
def total_blue_tiles : ℕ := initial_blue_tiles

-- Define the difference between the total number of green tiles and blue tiles
def tile_difference : ℕ := total_green_tiles - total_blue_tiles

-- The theorem stating that the difference between the total number of green tiles 
-- and the total number of blue tiles in the new figure is 11
theorem tile_difference_is_11 : tile_difference = 11 := by
  sorry

end NUMINAMATH_GPT_tile_difference_is_11_l267_26739


namespace NUMINAMATH_GPT_regions_bounded_by_blue_lines_l267_26771

theorem regions_bounded_by_blue_lines (n : ℕ) : 
  (2 * n^2 + 3 * n + 2) -(n - 1) * (2 * n + 1) ≥ 4 * n + 2 :=
by
  sorry

end NUMINAMATH_GPT_regions_bounded_by_blue_lines_l267_26771


namespace NUMINAMATH_GPT_total_coffee_needed_l267_26776

-- Conditions as definitions
def weak_coffee_amount_per_cup : ℕ := 1
def strong_coffee_amount_per_cup : ℕ := 2 * weak_coffee_amount_per_cup
def cups_of_weak_coffee : ℕ := 12
def cups_of_strong_coffee : ℕ := 12

-- Prove that the total amount of coffee needed equals 36 tablespoons
theorem total_coffee_needed : (weak_coffee_amount_per_cup * cups_of_weak_coffee) + (strong_coffee_amount_per_cup * cups_of_strong_coffee) = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_coffee_needed_l267_26776


namespace NUMINAMATH_GPT_quadrant_of_P_l267_26737

theorem quadrant_of_P (m n : ℝ) (h1 : m * n > 0) (h2 : m + n < 0) : (m < 0 ∧ n < 0) :=
by
  sorry

end NUMINAMATH_GPT_quadrant_of_P_l267_26737


namespace NUMINAMATH_GPT_prove_central_angle_of_sector_l267_26735

noncomputable def central_angle_of_sector (R α : ℝ) : Prop :=
  (2 * R + R * α = 8) ∧ (1 / 2 * α * R^2 = 4)

theorem prove_central_angle_of_sector :
  ∃ α R : ℝ, central_angle_of_sector R α ∧ α = 2 :=
sorry

end NUMINAMATH_GPT_prove_central_angle_of_sector_l267_26735


namespace NUMINAMATH_GPT_net_effect_sale_value_l267_26732

variable (P Q : ℝ) -- New price and quantity sold

theorem net_effect_sale_value (P Q : ℝ) :
  let new_sale_value := (0.75 * P) * (1.75 * Q)
  let original_sale_value := P * Q
  new_sale_value - original_sale_value = 0.3125 * (P * Q) := 
by
  sorry

end NUMINAMATH_GPT_net_effect_sale_value_l267_26732


namespace NUMINAMATH_GPT_simplify_expression_l267_26793

variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 9) - (x + 6) * (3 * x - 2) = 7 * x - 24 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l267_26793


namespace NUMINAMATH_GPT_correct_inequality_l267_26792

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_increasing : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0

theorem correct_inequality : f (-2) < f 1 ∧ f 1 < f 3 :=
by 
  sorry

end NUMINAMATH_GPT_correct_inequality_l267_26792


namespace NUMINAMATH_GPT_range_of_m_l267_26740

theorem range_of_m (m : ℝ) (h : m ≠ 0) :
  (∀ x : ℝ, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) →
  m < -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l267_26740


namespace NUMINAMATH_GPT_gcd_condition_l267_26712

def seq (a : ℕ → ℕ) := a 0 = 3 ∧ ∀ n, a (n + 1) - a n = n * (a n - 1)

theorem gcd_condition (a : ℕ → ℕ) (m : ℕ) (h : seq a) :
  m ≥ 2 → (∀ n, Nat.gcd m (a n) = 1) ↔ ∃ k : ℕ, m = 2^k ∧ k ≥ 1 := 
sorry

end NUMINAMATH_GPT_gcd_condition_l267_26712


namespace NUMINAMATH_GPT_vasya_max_triangles_l267_26707

theorem vasya_max_triangles (n : ℕ) (h1 : n = 100)
  (h2 : ∀ (a b c : ℕ), a + b ≤ c ∨ b + c ≤ a ∨ c + a ≤ b) :
  ∃ (t : ℕ), t = n := 
sorry

end NUMINAMATH_GPT_vasya_max_triangles_l267_26707


namespace NUMINAMATH_GPT_factory_minimize_salary_l267_26749

theorem factory_minimize_salary :
  ∃ x : ℕ, ∃ W : ℕ,
    x + (120 - x) = 120 ∧
    800 * x + 1000 * (120 - x) = W ∧
    120 - x ≥ 3 * x ∧
    x = 30 ∧
    W = 114000 :=
  sorry

end NUMINAMATH_GPT_factory_minimize_salary_l267_26749


namespace NUMINAMATH_GPT_expression_value_l267_26769

theorem expression_value (x y z : ℤ) (hx : x = 25) (hy : y = 30) (hz : z = 10) :
  (x - (y - z)) - ((x - y) - z) = 20 :=
by
  rw [hx, hy, hz]
  -- After substituting the values, we will need to simplify the expression to reach 20.
  sorry

end NUMINAMATH_GPT_expression_value_l267_26769


namespace NUMINAMATH_GPT_sophie_clothes_expense_l267_26786

theorem sophie_clothes_expense :
  let initial_fund := 260
  let shirt_cost := 18.50
  let trousers_cost := 63
  let num_shirts := 2
  let num_remaining_clothes := 4
  let total_spent := num_shirts * shirt_cost + trousers_cost
  let remaining_amount := initial_fund - total_spent
  let individual_item_cost := remaining_amount / num_remaining_clothes
  individual_item_cost = 40 := 
by 
  sorry

end NUMINAMATH_GPT_sophie_clothes_expense_l267_26786


namespace NUMINAMATH_GPT_minimum_value_expression_l267_26781

theorem minimum_value_expression :
  ∀ (r s t : ℝ), (1 ≤ r ∧ r ≤ s ∧ s ≤ t ∧ t ≤ 4) →
  (r - 1) ^ 2 + (s / r - 1) ^ 2 + (t / s - 1) ^ 2 + (4 / t - 1) ^ 2 = 4 * (Real.sqrt 2 - 1) ^ 2 := 
sorry

end NUMINAMATH_GPT_minimum_value_expression_l267_26781


namespace NUMINAMATH_GPT_amusement_park_total_cost_l267_26706

def rides_cost_ferris_wheel : ℕ := 5 * 6
def rides_cost_roller_coaster : ℕ := 7 * 4
def rides_cost_merry_go_round : ℕ := 3 * 10
def rides_cost_bumper_cars : ℕ := 4 * 7
def rides_cost_haunted_house : ℕ := 6 * 5
def rides_cost_log_flume : ℕ := 8 * 3

def snacks_cost_ice_cream : ℕ := 8 * 4
def snacks_cost_hot_dog : ℕ := 6 * 5
def snacks_cost_pizza : ℕ := 4 * 3
def snacks_cost_pretzel : ℕ := 5 * 2
def snacks_cost_cotton_candy : ℕ := 3 * 6
def snacks_cost_soda : ℕ := 2 * 7

def total_rides_cost : ℕ := 
  rides_cost_ferris_wheel + 
  rides_cost_roller_coaster + 
  rides_cost_merry_go_round + 
  rides_cost_bumper_cars + 
  rides_cost_haunted_house + 
  rides_cost_log_flume

def total_snacks_cost : ℕ := 
  snacks_cost_ice_cream + 
  snacks_cost_hot_dog + 
  snacks_cost_pizza + 
  snacks_cost_pretzel + 
  snacks_cost_cotton_candy + 
  snacks_cost_soda

def total_cost : ℕ :=
  total_rides_cost + total_snacks_cost

theorem amusement_park_total_cost :
  total_cost = 286 :=
by
  unfold total_cost total_rides_cost total_snacks_cost
  unfold rides_cost_ferris_wheel 
         rides_cost_roller_coaster 
         rides_cost_merry_go_round 
         rides_cost_bumper_cars 
         rides_cost_haunted_house 
         rides_cost_log_flume
         snacks_cost_ice_cream 
         snacks_cost_hot_dog 
         snacks_cost_pizza 
         snacks_cost_pretzel 
         snacks_cost_cotton_candy 
         snacks_cost_soda
  sorry

end NUMINAMATH_GPT_amusement_park_total_cost_l267_26706


namespace NUMINAMATH_GPT_theo_drinks_8_cups_per_day_l267_26717

/--
Theo, Mason, and Roxy are siblings. 
Mason drinks 7 cups of water every day.
Roxy drinks 9 cups of water every day. 
In one week, the siblings drink 168 cups of water together. 

Prove that Theo drinks 8 cups of water every day.
-/
theorem theo_drinks_8_cups_per_day (T : ℕ) :
  (∀ (d m r : ℕ), 
    (m = 7 ∧ r = 9 ∧ d + m + r = 168) → 
    (T * 7 = d) → T = 8) :=
by
  intros d m r cond1 cond2
  have h1 : d + 49 + 63 = 168 := by sorry
  have h2 : T * 7 = d := cond2
  have goal : T = 8 := by sorry
  exact goal

end NUMINAMATH_GPT_theo_drinks_8_cups_per_day_l267_26717


namespace NUMINAMATH_GPT_find_k_all_reals_l267_26753

theorem find_k_all_reals (a b c : ℝ) : 
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c :=
sorry

end NUMINAMATH_GPT_find_k_all_reals_l267_26753


namespace NUMINAMATH_GPT_inscribed_sphere_surface_area_l267_26733

theorem inscribed_sphere_surface_area (V S : ℝ) (hV : V = 2) (hS : S = 3) : 4 * Real.pi * (3 * V / S)^2 = 16 * Real.pi := by
  sorry

end NUMINAMATH_GPT_inscribed_sphere_surface_area_l267_26733


namespace NUMINAMATH_GPT_nina_total_amount_l267_26754

theorem nina_total_amount:
  ∃ (x y z w : ℕ), 
  x + y + z + w = 27 ∧
  y = 2 * z ∧
  z = 2 * x ∧
  7 < w ∧ w < 20 ∧
  10 * x + 5 * y + 2 * z + 3 * w = 107 :=
by 
  sorry

end NUMINAMATH_GPT_nina_total_amount_l267_26754


namespace NUMINAMATH_GPT_extreme_values_max_min_on_interval_coordinates_midpoint_parallel_tangents_l267_26711

-- Given function
def f (x : ℝ) : ℝ := x^3 - 12 * x + 12

-- Definition of derivative
def f' (x : ℝ) : ℝ := (3 : ℝ) * x^2 - (12 : ℝ)

-- Part 1: Extreme values
theorem extreme_values : 
  (f (-2) = 28) ∧ (f 2 = -4) :=
by
  sorry

-- Part 2: Maximum and minimum values on the interval [-3, 4]
theorem max_min_on_interval :
  (∀ x, -3 ≤ x ∧ x ≤ 4 → f x ≤ 28) ∧ (∀ x, -3 ≤ x ∧ x ≤ 4 → f x ≥ -4) :=
by
  sorry

-- Part 3: Coordinates of midpoint A and B with parallel tangents
theorem coordinates_midpoint_parallel_tangents :
  (f' x1 = f' x2 ∧ x1 + x2 = 0) → ((x1 + x2) / 2 = 0 ∧ (f x1 + f x2) / 2 = 12) :=
by
  sorry

end NUMINAMATH_GPT_extreme_values_max_min_on_interval_coordinates_midpoint_parallel_tangents_l267_26711


namespace NUMINAMATH_GPT_savings_calculation_l267_26715

def price_per_window : ℕ := 120
def discount_offer (n : ℕ) : ℕ := if n ≥ 10 then 2 else 0

def george_needs : ℕ := 9
def anne_needs : ℕ := 11

def cost (n : ℕ) : ℕ :=
  let free_windows := discount_offer n
  (n - free_windows) * price_per_window

theorem savings_calculation :
  let total_separate_cost := cost george_needs + cost anne_needs
  let total_windows := george_needs + anne_needs
  let total_cost_together := cost total_windows
  total_separate_cost - total_cost_together = 240 :=
by
  sorry

end NUMINAMATH_GPT_savings_calculation_l267_26715


namespace NUMINAMATH_GPT_work_completion_days_l267_26744

theorem work_completion_days (A B : Type) (A_work_rate B_work_rate : ℝ) :
  (1 / 16 : ℝ) = (1 / 20) + A_work_rate → B_work_rate = (1 / 80) := by
  sorry

end NUMINAMATH_GPT_work_completion_days_l267_26744


namespace NUMINAMATH_GPT_man_speed_l267_26709

theorem man_speed (rest_time_per_km : ℕ := 5) (total_km_covered : ℕ := 5) (total_time_min : ℕ := 50) : 
  (total_time_min - rest_time_per_km * (total_km_covered - 1)) / 60 * total_km_covered = 10 := by
  sorry

end NUMINAMATH_GPT_man_speed_l267_26709


namespace NUMINAMATH_GPT_multiply_by_3_l267_26790

variable (x : ℕ)  -- Declare x as a natural number

-- Define the conditions
def condition : Prop := x + 14 = 56

-- The goal to prove
theorem multiply_by_3 (h : condition x) : 3 * x = 126 := sorry

end NUMINAMATH_GPT_multiply_by_3_l267_26790


namespace NUMINAMATH_GPT_negative_cube_root_l267_26760

theorem negative_cube_root (a : ℝ) : ∃ x : ℝ, x ^ 3 = -a^2 - 1 ∧ x < 0 :=
by
  sorry

end NUMINAMATH_GPT_negative_cube_root_l267_26760


namespace NUMINAMATH_GPT_determine_q_l267_26731

theorem determine_q (p q : ℝ) 
  (h : ∀ x : ℝ, (x + 3) * (x + p) = x^2 + q * x + 12) : 
  q = 7 :=
by
  sorry

end NUMINAMATH_GPT_determine_q_l267_26731


namespace NUMINAMATH_GPT_polar_to_cartesian_l267_26757

theorem polar_to_cartesian (ρ : ℝ) (θ : ℝ) (hx : ρ = 3) (hy : θ = π / 6) :
  (ρ * Real.cos θ, ρ * Real.sin θ) = (3 * Real.cos (π / 6), 3 * Real.sin (π / 6)) := by
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_l267_26757


namespace NUMINAMATH_GPT_total_pages_read_l267_26763

variable (Jairus_pages : ℕ)
variable (Arniel_pages : ℕ)
variable (J_total : Jairus_pages = 20)
variable (A_total : Arniel_pages = 2 + 2 * Jairus_pages)

theorem total_pages_read : Jairus_pages + Arniel_pages = 62 := by
  rw [J_total, A_total]
  sorry

end NUMINAMATH_GPT_total_pages_read_l267_26763


namespace NUMINAMATH_GPT_max_pieces_with_three_cuts_l267_26799

def cake := Type

noncomputable def max_identical_pieces (cuts : ℕ) (max_cuts : ℕ) : ℕ :=
  if cuts = 3 ∧ max_cuts = 3 then 8 else sorry

theorem max_pieces_with_three_cuts : ∀ (c : cake), max_identical_pieces 3 3 = 8 :=
by
  intro c
  sorry

end NUMINAMATH_GPT_max_pieces_with_three_cuts_l267_26799


namespace NUMINAMATH_GPT_lines_skew_l267_26727

def line1 (b : ℝ) (t : ℝ) : ℝ × ℝ × ℝ := 
  (2 + 3 * t, 3 + 2 * t, b + 5 * t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ := 
  (5 + 6 * u, 4 + 3 * u, 1 + 2 * u)

theorem lines_skew (b : ℝ) : 
  ¬ ∃ t u : ℝ, line1 b t = line2 u ↔ b ≠ 4 := 
sorry

end NUMINAMATH_GPT_lines_skew_l267_26727


namespace NUMINAMATH_GPT_compare_powers_l267_26728

-- Definitions for the three numbers
def a : ℝ := 3 ^ 555
def b : ℝ := 4 ^ 444
def c : ℝ := 5 ^ 333

-- Statement to prove
theorem compare_powers : c < a ∧ a < b := sorry

end NUMINAMATH_GPT_compare_powers_l267_26728


namespace NUMINAMATH_GPT_greatest_integer_a_exists_l267_26782

theorem greatest_integer_a_exists (a x : ℤ) (h : (x - a) * (x - 7) + 3 = 0) : a ≤ 11 := by
  sorry

end NUMINAMATH_GPT_greatest_integer_a_exists_l267_26782


namespace NUMINAMATH_GPT_ratio_of_girls_to_boys_in_biology_class_l267_26722

-- Defining the conditions
def physicsClassStudents : Nat := 200
def biologyClassStudents := physicsClassStudents / 2
def boysInBiologyClass : Nat := 25
def girlsInBiologyClass := biologyClassStudents - boysInBiologyClass

-- Statement of the problem
theorem ratio_of_girls_to_boys_in_biology_class : girlsInBiologyClass / boysInBiologyClass = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_girls_to_boys_in_biology_class_l267_26722


namespace NUMINAMATH_GPT_stacy_days_to_complete_paper_l267_26716

variable (total_pages pages_per_day : ℕ)
variable (d : ℕ)

theorem stacy_days_to_complete_paper 
  (h1 : total_pages = 63) 
  (h2 : pages_per_day = 21) 
  (h3 : total_pages = pages_per_day * d) : 
  d = 3 := 
sorry

end NUMINAMATH_GPT_stacy_days_to_complete_paper_l267_26716


namespace NUMINAMATH_GPT_service_cleaning_fee_percentage_is_correct_l267_26752

noncomputable def daily_rate : ℝ := 125
noncomputable def pet_fee : ℝ := 100
noncomputable def duration : ℕ := 14
noncomputable def security_deposit_percentage : ℝ := 0.5
noncomputable def security_deposit : ℝ := 1110

noncomputable def total_expected_cost : ℝ := (daily_rate * duration) + pet_fee
noncomputable def entire_bill : ℝ := security_deposit / security_deposit_percentage
noncomputable def service_cleaning_fee : ℝ := entire_bill - total_expected_cost

theorem service_cleaning_fee_percentage_is_correct : 
  (service_cleaning_fee / entire_bill) * 100 = 16.67 :=
by 
  sorry

end NUMINAMATH_GPT_service_cleaning_fee_percentage_is_correct_l267_26752


namespace NUMINAMATH_GPT_find_x_l267_26734

-- Define the vectors and the condition of them being parallel
def vector_a : (ℝ × ℝ) := (3, 1)
def vector_b (x : ℝ) : (ℝ × ℝ) := (x, -1)
def parallel (a b : (ℝ × ℝ)) := ∃ k : ℝ, b = (k * a.1, k * a.2)

-- The theorem to prove
theorem find_x (x : ℝ) (h : parallel (3, 1) (x, -1)) : x = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l267_26734


namespace NUMINAMATH_GPT_maximum_x_plus_y_l267_26708

theorem maximum_x_plus_y (N x y : ℕ) 
  (hN : N = 19 * x + 95 * y) 
  (hp : ∃ k : ℕ, N = k^2) 
  (hN_le : N ≤ 1995) :
  x + y ≤ 86 :=
sorry

end NUMINAMATH_GPT_maximum_x_plus_y_l267_26708


namespace NUMINAMATH_GPT_solve_for_question_mark_l267_26777

def cube_root (x : ℝ) := x^(1/3)
def square_root (x : ℝ) := x^(1/2)

theorem solve_for_question_mark : 
  cube_root (5568 / 87) + square_root (72 * 2) = square_root 256 := by
  sorry

end NUMINAMATH_GPT_solve_for_question_mark_l267_26777


namespace NUMINAMATH_GPT_find_x_l267_26730

-- Definitions for the angles
def angle1 (x : ℝ) := 3 * x
def angle2 (x : ℝ) := 7 * x
def angle3 (x : ℝ) := 4 * x
def angle4 (x : ℝ) := 2 * x
def angle5 (x : ℝ) := x

-- The condition that the sum of the angles equals 360 degrees
def sum_of_angles (x : ℝ) := angle1 x + angle2 x + angle3 x + angle4 x + angle5 x = 360

-- The statement to prove
theorem find_x (x : ℝ) (hx : sum_of_angles x) : x = 360 / 17 := by
  -- Proof to be written here
  sorry

end NUMINAMATH_GPT_find_x_l267_26730


namespace NUMINAMATH_GPT_evaluate_polynomial_at_6_eq_1337_l267_26787

theorem evaluate_polynomial_at_6_eq_1337 :
  (3 * 6^2 + 15 * 6 + 7) + (4 * 6^3 + 8 * 6^2 - 5 * 6 + 10) = 1337 := by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_6_eq_1337_l267_26787


namespace NUMINAMATH_GPT_find_sphere_volume_l267_26778

noncomputable def sphere_volume (d: ℝ) (V: ℝ) : Prop := d = 3 * (16 / 9) * V

theorem find_sphere_volume :
  sphere_volume (2 / 3) (1 / 6) :=
by
  sorry

end NUMINAMATH_GPT_find_sphere_volume_l267_26778


namespace NUMINAMATH_GPT_tina_mother_took_out_coins_l267_26746

theorem tina_mother_took_out_coins :
  let first_hour := 20
  let next_two_hours := 30 * 2
  let fourth_hour := 40
  let total_coins := first_hour + next_two_hours + fourth_hour
  let coins_left_after_fifth_hour := 100
  let coins_taken_out := total_coins - coins_left_after_fifth_hour
  coins_taken_out = 20 :=
by
  sorry

end NUMINAMATH_GPT_tina_mother_took_out_coins_l267_26746


namespace NUMINAMATH_GPT_calculate_expression_l267_26714

theorem calculate_expression : 2.4 * 8.2 * (5.3 - 4.7) = 11.52 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l267_26714


namespace NUMINAMATH_GPT_cos_identity_arithmetic_sequence_in_triangle_l267_26797

theorem cos_identity_arithmetic_sequence_in_triangle
  {A B C : ℝ} {a b c : ℝ}
  (h1 : 2 * b = a + c)
  (h2 : a / Real.sin A = b / Real.sin B)
  (h3 : b / Real.sin B = c / Real.sin C)
  (h4 : A + B + C = Real.pi)
  : 5 * Real.cos A - 4 * Real.cos A * Real.cos C + 5 * Real.cos C = 4 := 
  sorry

end NUMINAMATH_GPT_cos_identity_arithmetic_sequence_in_triangle_l267_26797


namespace NUMINAMATH_GPT_range_of_m_l267_26758

variable {R : Type} [LinearOrderedField R]

def discriminant (a b c : R) : R := b^2 - 4 * a * c

theorem range_of_m (m : R) :
  (discriminant (1:R) m (m + 3) > 0) ↔ (m < -2 ∨ m > 6) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l267_26758


namespace NUMINAMATH_GPT_all_iterated_quadratic_eq_have_integer_roots_l267_26773

noncomputable def initial_quadratic_eq_has_integer_roots (p q : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, x1 + x2 = -p ∧ x1 * x2 = q

noncomputable def iterated_quadratic_eq_has_integer_roots (p q : ℤ) : Prop :=
  ∀ i : ℕ, i ≤ 9 → ∃ x1 x2 : ℤ, x1 + x2 = -(p + i) ∧ x1 * x2 = (q + i)

theorem all_iterated_quadratic_eq_have_integer_roots :
  ∃ p q : ℤ, initial_quadratic_eq_has_integer_roots p q ∧ iterated_quadratic_eq_has_integer_roots p q :=
sorry

end NUMINAMATH_GPT_all_iterated_quadratic_eq_have_integer_roots_l267_26773


namespace NUMINAMATH_GPT_a_ge_zero_of_set_nonempty_l267_26755

theorem a_ge_zero_of_set_nonempty {a : ℝ} (h : ∃ x : ℝ, x^2 = a) : a ≥ 0 :=
sorry

end NUMINAMATH_GPT_a_ge_zero_of_set_nonempty_l267_26755


namespace NUMINAMATH_GPT_complex_multiplication_imaginary_unit_l267_26723

theorem complex_multiplication_imaginary_unit 
  (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_imaginary_unit_l267_26723


namespace NUMINAMATH_GPT_find_vector_at_t_zero_l267_26719

variable (a d : ℝ × ℝ × ℝ)
variable (t : ℝ)

-- Given conditions
def condition1 := a - 2 * d = (2, 4, 10)
def condition2 := a + d = (-1, -3, -5)

-- The proof problem
theorem find_vector_at_t_zero 
  (h1 : condition1 a d)
  (h2 : condition2 a d) :
  a = (0, -2/3, 0) :=
sorry

end NUMINAMATH_GPT_find_vector_at_t_zero_l267_26719


namespace NUMINAMATH_GPT_range_x0_of_perpendicular_bisector_intersects_x_axis_l267_26718

open Real

theorem range_x0_of_perpendicular_bisector_intersects_x_axis
  (A B : ℝ × ℝ) 
  (hA : (A.1^2 / 9) + (A.2^2 / 8) = 1)
  (hB : (B.1^2 / 9) + (B.2^2 / 8) = 1)
  (N : ℝ × ℝ) 
  (P : ℝ × ℝ) 
  (hN : N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hP : P.2 = 0) 
  (hl : P.1 = N.1 + (8 * N.1) / (9 * N.2) * N.2)
  : -1/3 < P.1 ∧ P.1 < 1/3 :=
sorry

end NUMINAMATH_GPT_range_x0_of_perpendicular_bisector_intersects_x_axis_l267_26718


namespace NUMINAMATH_GPT_fraction_multiplication_l267_26743

-- Given fractions a and b
def a := (1 : ℚ) / 4
def b := (1 : ℚ) / 8

-- The first product result
def result1 := a * b

-- The final product result when multiplied by 4
def result2 := result1 * 4

-- The theorem to prove
theorem fraction_multiplication : result2 = (1 : ℚ) / 8 := by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_l267_26743


namespace NUMINAMATH_GPT_maci_pays_total_cost_l267_26779

def cost_blue_pen : ℝ := 0.10
def num_blue_pens : ℕ := 10
def num_red_pens : ℕ := 15
def cost_red_pen : ℝ := 2 * cost_blue_pen

def total_cost : ℝ := num_blue_pens * cost_blue_pen + num_red_pens * cost_red_pen

theorem maci_pays_total_cost : total_cost = 4 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_maci_pays_total_cost_l267_26779


namespace NUMINAMATH_GPT_sum_of_coefficients_of_y_terms_l267_26766

theorem sum_of_coefficients_of_y_terms: 
  let p := (5 * x + 3 * y + 2) * (2 * x + 5 * y + 3)
  ∃ (a b c: ℝ), p = (10 * x^2 + a * x * y + 19 * x + b * y^2 + c * y + 6) ∧ a + b + c = 65 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_of_y_terms_l267_26766


namespace NUMINAMATH_GPT_probability_of_female_selection_probability_of_male_host_selection_l267_26784

/-!
In a competition, there are eight contestants consisting of five females and three males.
If three contestants are chosen randomly to progress to the next round, what is the 
probability that all selected contestants are female? Additionally, from those who 
do not proceed, one is selected as a host. What is the probability that this host is male?
-/

noncomputable def number_of_ways_select_3_from_8 : ℕ := Nat.choose 8 3

noncomputable def number_of_ways_select_3_females_from_5 : ℕ := Nat.choose 5 3

noncomputable def probability_all_3_females : ℚ := number_of_ways_select_3_females_from_5 / number_of_ways_select_3_from_8

noncomputable def number_of_remaining_contestants : ℕ := 8 - 3

noncomputable def number_of_males_remaining : ℕ := 3 - 1

noncomputable def number_of_ways_select_1_male_from_2 : ℕ := Nat.choose 2 1

noncomputable def number_of_ways_select_1_from_5 : ℕ := Nat.choose 5 1

noncomputable def probability_host_is_male : ℚ := number_of_ways_select_1_male_from_2 / number_of_ways_select_1_from_5

theorem probability_of_female_selection : probability_all_3_females = 5 / 28 := by
  sorry

theorem probability_of_male_host_selection : probability_host_is_male = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_of_female_selection_probability_of_male_host_selection_l267_26784


namespace NUMINAMATH_GPT_distinct_positive_integers_factors_PQ_RS_l267_26747

theorem distinct_positive_integers_factors_PQ_RS (P Q R S : ℕ) (hP : P > 0) (hQ : Q > 0) (hR : R > 0) (hS : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDistinctPQ : P ≠ Q) (hDistinctRS : R ≠ S) (hPQR_S : P + Q = R - S) :
  P = 4 :=
by
  sorry

end NUMINAMATH_GPT_distinct_positive_integers_factors_PQ_RS_l267_26747


namespace NUMINAMATH_GPT_newborn_members_approximation_l267_26738

-- Defining the conditions
def survival_prob_first_month : ℚ := 7/8
def survival_prob_second_month : ℚ := 7/8
def survival_prob_third_month : ℚ := 7/8
def survival_prob_three_months : ℚ := (7/8) ^ 3
def expected_survivors : ℚ := 133.984375

-- Statement to prove that the number of newborn members, N, approximates to 200
theorem newborn_members_approximation (N : ℚ) : 
  N * survival_prob_three_months = expected_survivors → 
  N = 200 :=
by
  sorry

end NUMINAMATH_GPT_newborn_members_approximation_l267_26738


namespace NUMINAMATH_GPT_ab_ac_bc_nonpositive_l267_26788

theorem ab_ac_bc_nonpositive (a b c : ℝ) (h : a + b + c = 0) : ∃ y : ℝ, y = ab + ac + bc ∧ y ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_ab_ac_bc_nonpositive_l267_26788


namespace NUMINAMATH_GPT_profit_percent_calculation_l267_26774

variable (SP : ℝ) (CP : ℝ) (Profit : ℝ) (ProfitPercent : ℝ)
variable (h1 : CP = 0.75 * SP)
variable (h2 : Profit = SP - CP)
variable (h3 : ProfitPercent = (Profit / CP) * 100)

theorem profit_percent_calculation : ProfitPercent = 33.33 := 
sorry

end NUMINAMATH_GPT_profit_percent_calculation_l267_26774


namespace NUMINAMATH_GPT_a_beats_b_by_7_seconds_l267_26789

/-
  Given:
  1. A's time to finish the race is 28 seconds (tA = 28).
  2. The race distance is 280 meters (d = 280).
  3. A beats B by 56 meters (dA - dB = 56).
  
  Prove:
  A beats B by 7 seconds (tB - tA = 7).
-/

theorem a_beats_b_by_7_seconds 
  (tA : ℕ) (d : ℕ) (speedA : ℕ) (dB : ℕ) (tB : ℕ) 
  (h1 : tA = 28) 
  (h2 : d = 280) 
  (h3 : d - dB = 56) 
  (h4 : speedA = d / tA) 
  (h5 : dB = speedA * tA) 
  (h6 : tB = d / speedA) :
  tB - tA = 7 := 
sorry

end NUMINAMATH_GPT_a_beats_b_by_7_seconds_l267_26789


namespace NUMINAMATH_GPT_problem_bounds_l267_26783

theorem problem_bounds :
  ∀ (A_0 B_0 C_0 A_1 B_1 C_1 A_2 B_2 C_2 A_3 B_3 C_3 : Point),
    (A_0B_0 + B_0C_0 + C_0A_0 = 1) →
    (A_1B_1 = A_0B_0) →
    (B_1C_1 = B_0C_0) →
    (A_2 = A_1 ∧ B_2 = B_1 ∧ C_2 = C_1 ∨
     A_2 = A_1 ∧ B_2 = C_1 ∧ C_2 = B_1 ∨
     A_2 = B_1 ∧ B_2 = A_1 ∧ C_2 = C_1 ∨
     A_2 = B_1 ∧ B_2 = C_1 ∧ C_2 = A_1 ∨
     A_2 = C_1 ∧ B_2 = A_1 ∧ C_2 = B_1 ∨
     A_2 = C_1 ∧ B_2 = B_1 ∧ C_2 = A_1) →
    (A_3B_3 = A_2B_2) →
    (B_3C_3 = B_2C_2) →
    (A_3B_3 + B_3C_3 + C_3A_3) ≥ 1 / 3 ∧ 
    (A_3B_3 + B_3C_3 + C_3A_3) ≤ 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_problem_bounds_l267_26783


namespace NUMINAMATH_GPT_work_completion_time_l267_26736

theorem work_completion_time 
  (M W : ℝ) 
  (h1 : (10 * M + 15 * W) * 6 = 1) 
  (h2 : M * 100 = 1) 
  : W * 225 = 1 := 
by
  sorry

end NUMINAMATH_GPT_work_completion_time_l267_26736


namespace NUMINAMATH_GPT_evaluate_expression_l267_26785

theorem evaluate_expression : 12^2 + 2 * 12 * 5 + 5^2 = 289 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l267_26785


namespace NUMINAMATH_GPT_smallest_integer_value_l267_26756

theorem smallest_integer_value (x : ℤ) (h : 3 * |x| + 8 < 29) : x = -6 :=
sorry

end NUMINAMATH_GPT_smallest_integer_value_l267_26756


namespace NUMINAMATH_GPT_eval_nested_fractions_l267_26745

theorem eval_nested_fractions : (1 / (1 + 1 / (4 + 1 / 5))) = (21 / 26) :=
by
  sorry

end NUMINAMATH_GPT_eval_nested_fractions_l267_26745


namespace NUMINAMATH_GPT_find_a_l267_26750

theorem find_a (a : ℝ) (i : ℂ) (hi : i = Complex.I) (z : ℂ) (hz : z = a + i) (h : z^2 + z = 1 - 3 * Complex.I) :
  a = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_l267_26750


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l267_26700

noncomputable def eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem eccentricity_of_ellipse
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (l : ℝ → ℝ) (hl : l 0 = 0)
  (h_intersects : ∃ M N : ℝ × ℝ, M ≠ N ∧ (M.1 / a)^2 + (M.2 / b)^2 = 1 ∧ (N.1 / a)^2 + (N.2 / b)^2 = 1 ∧ l M.1 = M.2 ∧ l N.1 = N.2)
  (P : ℝ × ℝ) (hP : (P.1 / a)^2 + (P.2 / b)^2 = 1 ∧ P ≠ (0, 0))
  (h_product_slopes : ∀ (Mx Nx Px : ℝ) (k : ℝ),
    l Mx = k * Mx →
    l Nx = k * Nx →
    l Px ≠ k * Px →
    ((k * Mx - P.2) / (Mx - P.1)) * ((k * Nx - P.2) / (Nx - P.1)) = -1/3) :
  eccentricity a b h1 h2 = Real.sqrt (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l267_26700


namespace NUMINAMATH_GPT_part1_part2_l267_26798

-- Part 1: Proving the inequality
theorem part1 (a b c d : ℝ) : 
  (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := by
  sorry

-- Part 2: Maximizing 2a + b
theorem part2 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_constraint : a^2 + b^2 = 5) : 
  2 * a + b ≤ 5 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l267_26798


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l267_26765

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_b : ℝ × ℝ := (3, -Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * vector_b.1 + (vector_a x).2 * vector_b.2

theorem problem_part_1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) : 
  (vector_a x).1 * vector_b.2 = (vector_a x).2 * vector_b.1 → 
  x = 5 * Real.pi / 6 :=
by
  sorry

theorem problem_part_2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) :
  (∀ t, 0 ≤ t ∧ t ≤ Real.pi → f x ≤ f t) → x = 0 ∧ f 0 = 3 ∧ 
  (∀ t, 0 ≤ t ∧ t ≤ Real.pi → f x ≥ f t) → x = 5 * Real.pi / 6 ∧ f (5 * Real.pi / 6) = -2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l267_26765


namespace NUMINAMATH_GPT_root_implies_value_l267_26762

theorem root_implies_value (b c : ℝ) (h : 2 * b - c = 4) : 4 * b - 2 * c + 1 = 9 :=
by
  sorry

end NUMINAMATH_GPT_root_implies_value_l267_26762


namespace NUMINAMATH_GPT_abigail_writing_time_l267_26701

def total_additional_time (words_needed : ℕ) (words_per_half_hour : ℕ) (words_already_written : ℕ) (proofreading_time : ℕ) : ℕ :=
  let remaining_words := words_needed - words_already_written
  let half_hour_blocks := (remaining_words + words_per_half_hour - 1) / words_per_half_hour -- ceil(remaining_words / words_per_half_hour)
  let writing_time := half_hour_blocks * 30
  writing_time + proofreading_time

theorem abigail_writing_time :
  total_additional_time 1500 250 200 45 = 225 :=
by {
  -- Adding the proof in Lean:
  -- fail to show you the detailed steps, hence added sorry
  sorry
}

end NUMINAMATH_GPT_abigail_writing_time_l267_26701


namespace NUMINAMATH_GPT_cube_split_odd_numbers_l267_26795

theorem cube_split_odd_numbers (m : ℕ) (h1 : 1 < m) (h2 : ∃ k, (31 = 2 * k + 1 ∧ (m - 1) * m / 2 = k)) : m = 6 := 
by
  sorry

end NUMINAMATH_GPT_cube_split_odd_numbers_l267_26795


namespace NUMINAMATH_GPT_stratified_sampling_example_l267_26721

theorem stratified_sampling_example
  (students_ratio : ℕ → ℕ) -- function to get the number of students in each grade, indexed by natural numbers
  (ratio_cond : students_ratio 0 = 4 ∧ students_ratio 1 = 3 ∧ students_ratio 2 = 2) -- the ratio 4:3:2
  (third_grade_sample : ℕ) -- number of students in the third grade in the sample
  (third_grade_sample_eq : third_grade_sample = 10) -- 10 students from the third grade
  (total_sample_size : ℕ) -- the sample size n
 :
  total_sample_size = 45 := 
sorry

end NUMINAMATH_GPT_stratified_sampling_example_l267_26721


namespace NUMINAMATH_GPT_xiaofang_time_l267_26710

-- Definitions
def overlap_time (t : ℕ) : Prop :=
  t - t / 12 = 40

def opposite_time (t : ℕ) : Prop :=
  t - t / 12 = 40

-- Theorem statement
theorem xiaofang_time :
  ∃ (x y : ℕ), 
    480 + x = 8 * 60 + 43 ∧
    840 + y = 2 * 60 + 43 ∧
    overlap_time x ∧
    opposite_time y ∧
    (y + 840 - (x + 480)) = 6 * 60 :=
by
  sorry

end NUMINAMATH_GPT_xiaofang_time_l267_26710


namespace NUMINAMATH_GPT_milk_production_l267_26726

theorem milk_production (a b c d e : ℕ) (f g : ℝ) (hf : f = 0.8) (hg : g = 1.1) :
  ((d : ℝ) * e * g * (b : ℝ) / (a * c)) = 1.1 * b * d * e / (a * c) := by
  sorry

end NUMINAMATH_GPT_milk_production_l267_26726


namespace NUMINAMATH_GPT_total_rocks_needed_l267_26768

def rocks_already_has : ℕ := 64
def rocks_needed : ℕ := 61

theorem total_rocks_needed : rocks_already_has + rocks_needed = 125 :=
by
  sorry

end NUMINAMATH_GPT_total_rocks_needed_l267_26768


namespace NUMINAMATH_GPT_circle_area_from_circumference_l267_26759

theorem circle_area_from_circumference (r : ℝ) (π : ℝ) (h1 : 2 * π * r = 36) : (π * (r^2) = 324 / π) := by
  sorry

end NUMINAMATH_GPT_circle_area_from_circumference_l267_26759


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l267_26794

theorem eccentricity_of_ellipse (a b c e : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = Real.sqrt (a^2 - b^2)) 
  (h4 : b = Real.sqrt 3 * c) : e = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l267_26794


namespace NUMINAMATH_GPT_parabola_inequality_l267_26742

theorem parabola_inequality (a c y1 y2 : ℝ) (h1 : a < 0)
  (h2 : y1 = a * (-1 - 1)^2 + c)
  (h3 : y2 = a * (4 - 1)^2 + c) :
  y1 > y2 :=
sorry

end NUMINAMATH_GPT_parabola_inequality_l267_26742


namespace NUMINAMATH_GPT_ken_kept_pencils_l267_26767

def ken_total_pencils := 50
def pencils_given_to_manny := 10
def pencils_given_to_nilo := pencils_given_to_manny + 10
def pencils_given_away := pencils_given_to_manny + pencils_given_to_nilo

theorem ken_kept_pencils : ken_total_pencils - pencils_given_away = 20 := by
  sorry

end NUMINAMATH_GPT_ken_kept_pencils_l267_26767


namespace NUMINAMATH_GPT_sandra_stickers_l267_26729

theorem sandra_stickers :
  ∃ N : ℕ, N > 1 ∧ (N % 3 = 1) ∧ (N % 5 = 1) ∧ (N % 11 = 1) ∧ N = 166 :=
by {
  sorry
}

end NUMINAMATH_GPT_sandra_stickers_l267_26729


namespace NUMINAMATH_GPT_exist_ints_a_b_for_any_n_l267_26770

theorem exist_ints_a_b_for_any_n (n : ℤ) : ∃ a b : ℤ, n = Int.floor (a * Real.sqrt 2) + Int.floor (b * Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_exist_ints_a_b_for_any_n_l267_26770


namespace NUMINAMATH_GPT_sum_of_products_of_roots_l267_26702

theorem sum_of_products_of_roots (p q r : ℂ) (h : 4 * (p^3) - 2 * (p^2) + 13 * p - 9 = 0 ∧ 4 * (q^3) - 2 * (q^2) + 13 * q - 9 = 0 ∧ 4 * (r^3) - 2 * (r^2) + 13 * r - 9 = 0) :
  p*q + p*r + q*r = 13 / 4 :=
  sorry

end NUMINAMATH_GPT_sum_of_products_of_roots_l267_26702


namespace NUMINAMATH_GPT_tim_kittens_l267_26775

theorem tim_kittens (K : ℕ) (h1 : (3 / 5 : ℚ) * (2 / 3 : ℚ) * K = 12) : K = 30 :=
sorry

end NUMINAMATH_GPT_tim_kittens_l267_26775


namespace NUMINAMATH_GPT_marks_lost_per_wrong_answer_l267_26713

theorem marks_lost_per_wrong_answer (x : ℝ) : 
  (score_per_correct = 4) ∧ 
  (num_questions = 60) ∧ 
  (total_marks = 120) ∧ 
  (correct_answers = 36) ∧ 
  (wrong_answers = num_questions - correct_answers) ∧
  (wrong_answers = 24) ∧
  (total_score_from_correct = score_per_correct * correct_answers) ∧ 
  (total_marks_lost = total_score_from_correct - total_marks) ∧ 
  (total_marks_lost = wrong_answers * x) → 
  x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_marks_lost_per_wrong_answer_l267_26713


namespace NUMINAMATH_GPT_megan_homework_problems_l267_26704

theorem megan_homework_problems
  (finished_problems : ℕ)
  (pages_remaining : ℕ)
  (problems_per_page : ℕ)
  (total_problems : ℕ) :
  finished_problems = 26 →
  pages_remaining = 2 →
  problems_per_page = 7 →
  total_problems = finished_problems + (pages_remaining * problems_per_page) →
  total_problems = 40 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_megan_homework_problems_l267_26704


namespace NUMINAMATH_GPT_price_of_72_cans_l267_26725

def regular_price_per_can : ℝ := 0.30
def discount_percentage : ℝ := 0.15
def discounted_price_per_can := regular_price_per_can * (1 - discount_percentage)
def cans_purchased : ℕ := 72

theorem price_of_72_cans :
  cans_purchased * discounted_price_per_can = 18.36 :=
by sorry

end NUMINAMATH_GPT_price_of_72_cans_l267_26725


namespace NUMINAMATH_GPT_minimum_a2_plus_4b2_l267_26703

theorem minimum_a2_plus_4b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 1) : 
  a^2 + 4 * b^2 ≥ 32 :=
sorry

end NUMINAMATH_GPT_minimum_a2_plus_4b2_l267_26703


namespace NUMINAMATH_GPT_arithmetic_sequence_a12_l267_26720

theorem arithmetic_sequence_a12 (a : ℕ → ℝ)
    (h1 : a 3 + a 4 + a 5 = 3)
    (h2 : a 8 = 8)
    (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) :
    a 12 = 15 :=
by
  -- Since we aim to ensure the statement alone compiles, we leave the proof with 'sorry'.
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a12_l267_26720


namespace NUMINAMATH_GPT_original_class_strength_l267_26764

theorem original_class_strength 
  (x : ℕ) 
  (h1 : ∀ a_avg n, a_avg = 40 → n = x)
  (h2 : ∀ b_avg m, b_avg = 32 → m = 12)
  (h3 : ∀ new_avg, new_avg = 36 → ((x * 40 + 12 * 32) = ((x + 12) * 36))) : 
  x = 12 :=
by 
  sorry

end NUMINAMATH_GPT_original_class_strength_l267_26764


namespace NUMINAMATH_GPT_each_person_ate_2_cakes_l267_26772

def initial_cakes : ℕ := 8
def number_of_friends : ℕ := 4

theorem each_person_ate_2_cakes (h_initial_cakes : initial_cakes = 8)
  (h_number_of_friends : number_of_friends = 4) :
  initial_cakes / number_of_friends = 2 :=
by sorry

end NUMINAMATH_GPT_each_person_ate_2_cakes_l267_26772


namespace NUMINAMATH_GPT_basketball_total_points_l267_26796

variable (Jon_points Jack_points Tom_points : ℕ)

def Jon_score := 3
def Jack_score := Jon_score + 5
def Tom_score := (Jon_score + Jack_score) - 4

theorem basketball_total_points :
  Jon_score + Jack_score + Tom_score = 18 := by
  sorry

end NUMINAMATH_GPT_basketball_total_points_l267_26796


namespace NUMINAMATH_GPT_total_savings_correct_l267_26741

theorem total_savings_correct :
  let price_chlorine := 10
  let discount1_chlorine := 0.20
  let discount2_chlorine := 0.10
  let price_soap := 16
  let discount1_soap := 0.25
  let discount2_soap := 0.05
  let price_wipes := 8
  let bogo_discount_wipes := 0.50
  let quantity_chlorine := 4
  let quantity_soap := 6
  let quantity_wipes := 8
  let final_chlorine_price := (price_chlorine * (1 - discount1_chlorine)) * (1 - discount2_chlorine)
  let final_soap_price := (price_soap * (1 - discount1_soap)) * (1 - discount2_soap)
  let final_wipes_price_per_two := price_wipes + price_wipes * bogo_discount_wipes
  let final_wipes_price := final_wipes_price_per_two / 2
  let total_original_price := quantity_chlorine * price_chlorine + quantity_soap * price_soap + quantity_wipes * price_wipes
  let total_final_price := quantity_chlorine * final_chlorine_price + quantity_soap * final_soap_price + quantity_wipes * final_wipes_price
  let total_savings := total_original_price - total_final_price
  total_savings = 55.80 :=
by sorry

end NUMINAMATH_GPT_total_savings_correct_l267_26741


namespace NUMINAMATH_GPT_find_smallest_n_l267_26705

theorem find_smallest_n : ∃ n : ℕ, (n - 4)^3 > (n^3 / 2) ∧ ∀ m : ℕ, m < n → (m - 4)^3 ≤ (m^3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_n_l267_26705


namespace NUMINAMATH_GPT_skye_race_l267_26724

noncomputable def first_part_length := 3

theorem skye_race 
  (total_track_length : ℕ := 6)
  (speed_first_part : ℕ := 150)
  (distance_second_part : ℕ := 2)
  (speed_second_part : ℕ := 200)
  (distance_third_part : ℕ := 1)
  (speed_third_part : ℕ := 300)
  (avg_speed : ℕ := 180) :
  first_part_length = 3 :=
  sorry

end NUMINAMATH_GPT_skye_race_l267_26724


namespace NUMINAMATH_GPT_square_binomial_l267_26761

theorem square_binomial (x : ℝ) : (-x - 1) ^ 2 = x^2 + 2 * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_square_binomial_l267_26761


namespace NUMINAMATH_GPT_total_pencils_is_220_l267_26748

theorem total_pencils_is_220
  (A : ℕ) (B : ℕ) (P : ℕ) (Q : ℕ)
  (hA : A = 50)
  (h_sum : A + B = 140)
  (h_diff : B - A = P/2)
  (h_pencils : Q = P + 60)
  : P + Q = 220 :=
by
  sorry

end NUMINAMATH_GPT_total_pencils_is_220_l267_26748


namespace NUMINAMATH_GPT_no_integer_roots_quadratic_l267_26780

theorem no_integer_roots_quadratic (a b : ℤ) : 
  ∀ u : ℤ, ¬(u^2 + 3*a*u + 3*(2 - b^2) = 0) := 
by
  sorry

end NUMINAMATH_GPT_no_integer_roots_quadratic_l267_26780
