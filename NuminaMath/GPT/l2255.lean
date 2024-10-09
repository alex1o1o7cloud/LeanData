import Mathlib

namespace calculate_expression_l2255_225522

theorem calculate_expression : 2.4 * 8.2 * (5.3 - 4.7) = 11.52 := by
  sorry

end calculate_expression_l2255_225522


namespace box_weight_in_kg_l2255_225560

def weight_of_one_bar : ℕ := 125 -- Weight of one chocolate bar in grams
def number_of_bars : ℕ := 16 -- Number of chocolate bars in the box
def grams_to_kg (g : ℕ) : ℕ := g / 1000 -- Function to convert grams to kilograms

theorem box_weight_in_kg : grams_to_kg (weight_of_one_bar * number_of_bars) = 2 :=
by
  sorry -- Proof is omitted

end box_weight_in_kg_l2255_225560


namespace solve_for_y_in_equation_l2255_225504

theorem solve_for_y_in_equation : ∃ y : ℝ, 7 * (2 * y - 3) + 5 = -3 * (4 - 5 * y) ∧ y = -4 :=
by
  use -4
  sorry

end solve_for_y_in_equation_l2255_225504


namespace jogging_track_circumference_l2255_225562

theorem jogging_track_circumference (speed_deepak speed_wife : ℝ) (time_meet_minutes : ℝ) 
  (h1 : speed_deepak = 20) (h2 : speed_wife = 16) (h3 : time_meet_minutes = 36) : 
  let relative_speed := speed_deepak + speed_wife
  let time_meet_hours := time_meet_minutes / 60
  let circumference := relative_speed * time_meet_hours
  circumference = 21.6 :=
by
  sorry

end jogging_track_circumference_l2255_225562


namespace find_m_even_fn_l2255_225596

theorem find_m_even_fn (m : ℝ) (f : ℝ → ℝ) 
  (Hf : ∀ x : ℝ, f x = x * (10^x + m * 10^(-x))) 
  (Heven : ∀ x : ℝ, f (-x) = f x) : m = -1 := by
  sorry

end find_m_even_fn_l2255_225596


namespace tangent_lines_parabola_through_point_l2255_225587

theorem tangent_lines_parabola_through_point :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y = x ^ 2 + 1 → (y - 0) = m * (x - 0)) 
     ∧ ((m = 2 ∧ y = 2 * x) ∨ (m = -2 ∧ y = -2 * x)) :=
sorry

end tangent_lines_parabola_through_point_l2255_225587


namespace complex_multiplication_imaginary_unit_l2255_225535

theorem complex_multiplication_imaginary_unit 
  (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end complex_multiplication_imaginary_unit_l2255_225535


namespace probability_of_event_l2255_225591

def is_uniform (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1

theorem probability_of_event : 
  ∀ (a : ℝ), is_uniform a → ∀ (p : ℚ), (3 * a - 1 > 0) → p = 2 / 3 → 
  (∃ b, 0 ≤ b ∧ b ≤ 1 ∧ 3 * b - 1 > 0) := 
by
  intro a h_uniform p h_event h_prob
  sorry

end probability_of_event_l2255_225591


namespace ratio_of_girls_to_boys_in_biology_class_l2255_225552

-- Defining the conditions
def physicsClassStudents : Nat := 200
def biologyClassStudents := physicsClassStudents / 2
def boysInBiologyClass : Nat := 25
def girlsInBiologyClass := biologyClassStudents - boysInBiologyClass

-- Statement of the problem
theorem ratio_of_girls_to_boys_in_biology_class : girlsInBiologyClass / boysInBiologyClass = 3 :=
by
  sorry

end ratio_of_girls_to_boys_in_biology_class_l2255_225552


namespace tangent_line_condition_l2255_225526

theorem tangent_line_condition (k : ℝ) : 
  (∀ x y : ℝ, (x-2)^2 + (y-1)^2 = 1 → x - k * y - 1 = 0 → False) ↔ k = 0 :=
sorry

end tangent_line_condition_l2255_225526


namespace relationship_of_x_vals_l2255_225569

variables {k x1 x2 x3 : ℝ}

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := k / x

theorem relationship_of_x_vals (h1 : inverse_proportion_function k x1 = 1)
                              (h2 : inverse_proportion_function k x2 = -5)
                              (h3 : inverse_proportion_function k x3 = 3)
                              (hk : k < 0) :
                              x1 < x3 ∧ x3 < x2 :=
by
  sorry

end relationship_of_x_vals_l2255_225569


namespace man_speed_l2255_225536

theorem man_speed (rest_time_per_km : ℕ := 5) (total_km_covered : ℕ := 5) (total_time_min : ℕ := 50) : 
  (total_time_min - rest_time_per_km * (total_km_covered - 1)) / 60 * total_km_covered = 10 := by
  sorry

end man_speed_l2255_225536


namespace distance_between_points_l2255_225518

theorem distance_between_points (x : ℝ) :
  let M := (-1, 4)
  let N := (x, 4)
  dist (M, N) = 5 →
  (x = -6 ∨ x = 4) := sorry

end distance_between_points_l2255_225518


namespace cone_volume_ratio_l2255_225590

theorem cone_volume_ratio (r_C h_C r_D h_D : ℝ) (h_rC : r_C = 20) (h_hC : h_C = 40) 
  (h_rD : r_D = 40) (h_hD : h_D = 20) : 
  (1 / 3 * pi * r_C^2 * h_C) / (1 / 3 * pi * r_D^2 * h_D) = 1 / 2 :=
by
  rw [h_rC, h_hC, h_rD, h_hD]
  sorry

end cone_volume_ratio_l2255_225590


namespace f_properties_l2255_225505

noncomputable def f (x : ℝ) : ℝ :=
if -2 < x ∧ x < 0 then 2^x else sorry

theorem f_properties (f_odd : ∀ x : ℝ, f (-x) = -f x)
                     (f_periodic : ∀ x : ℝ, f (x + 3 / 2) = -f x) :
  f 2014 + f 2015 + f 2016 = 0 :=
by 
  -- The proof will go here
  sorry

end f_properties_l2255_225505


namespace arithmetic_sequence_a12_l2255_225566

theorem arithmetic_sequence_a12 (a : ℕ → ℝ)
    (h1 : a 3 + a 4 + a 5 = 3)
    (h2 : a 8 = 8)
    (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) :
    a 12 = 15 :=
by
  -- Since we aim to ensure the statement alone compiles, we leave the proof with 'sorry'.
  sorry

end arithmetic_sequence_a12_l2255_225566


namespace arithmetic_square_root_of_nine_l2255_225543

theorem arithmetic_square_root_of_nine : ∃ (x : ℝ), (x * x = 9) ∧ (x ≥ 0) ∧ (x = 3) :=
by
  sorry

end arithmetic_square_root_of_nine_l2255_225543


namespace jill_sod_area_needed_l2255_225507

def plot_width : ℕ := 200
def plot_length : ℕ := 50
def sidewalk_width : ℕ := 3
def sidewalk_length : ℕ := 50
def flower_bed1_depth : ℕ := 4
def flower_bed1_length : ℕ := 25
def flower_bed1_count : ℕ := 2
def flower_bed2_width : ℕ := 10
def flower_bed2_length : ℕ := 12
def flower_bed3_width : ℕ := 7
def flower_bed3_length : ℕ := 8

theorem jill_sod_area_needed :
  (plot_width * plot_length) - 
  (sidewalk_width * sidewalk_length + 
   flower_bed1_depth * flower_bed1_length * flower_bed1_count + 
   flower_bed2_width * flower_bed2_length + 
   flower_bed3_width * flower_bed3_length) = 9474 :=
by
  sorry

end jill_sod_area_needed_l2255_225507


namespace total_trip_duration_proof_l2255_225525

-- Naming all components
def driving_time : ℝ := 5
def first_jam_duration (pre_first_jam_drive : ℝ) : ℝ := 1.5 * pre_first_jam_drive
def second_jam_duration (between_first_and_second_drive : ℝ) : ℝ := 2 * between_first_and_second_drive
def third_jam_duration (between_second_and_third_drive : ℝ) : ℝ := 3 * between_second_and_third_drive
def pit_stop_duration : ℝ := 0.5
def pit_stops : ℕ := 2
def initial_drive : ℝ := 1
def second_drive : ℝ := 1.5

-- Additional drive time calculation
def remaining_drive : ℝ := driving_time - initial_drive - second_drive

-- Total duration calculation
def total_duration (initial_drive : ℝ) (second_drive : ℝ) (remaining_drive : ℝ) (first_jam_duration : ℝ) 
(second_jam_duration : ℝ) (third_jam_duration : ℝ) (pit_stop_duration : ℝ) (pit_stops : ℕ) : ℝ :=
  driving_time + first_jam_duration + second_jam_duration + third_jam_duration + (pit_stop_duration * pit_stops)

theorem total_trip_duration_proof :
  total_duration initial_drive second_drive remaining_drive (first_jam_duration initial_drive)
                  (second_jam_duration second_drive) (third_jam_duration remaining_drive) pit_stop_duration pit_stops 
  = 18 :=
by
  -- Proof steps would go here
  sorry

end total_trip_duration_proof_l2255_225525


namespace identity_eq_l2255_225517

theorem identity_eq (a b : ℤ) (h₁ : a = -1) (h₂ : b = 1) : 
  (∀ x : ℝ, ((2 * x + a) ^ 3) = (5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x)) := by
  sorry

end identity_eq_l2255_225517


namespace price_of_72_cans_l2255_225558

def regular_price_per_can : ℝ := 0.30
def discount_percentage : ℝ := 0.15
def discounted_price_per_can := regular_price_per_can * (1 - discount_percentage)
def cans_purchased : ℕ := 72

theorem price_of_72_cans :
  cans_purchased * discounted_price_per_can = 18.36 :=
by sorry

end price_of_72_cans_l2255_225558


namespace range_x0_of_perpendicular_bisector_intersects_x_axis_l2255_225550

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

end range_x0_of_perpendicular_bisector_intersects_x_axis_l2255_225550


namespace milk_production_l2255_225556

theorem milk_production (a b c d e : ℕ) (f g : ℝ) (hf : f = 0.8) (hg : g = 1.1) :
  ((d : ℝ) * e * g * (b : ℝ) / (a * c)) = 1.1 * b * d * e / (a * c) := by
  sorry

end milk_production_l2255_225556


namespace extreme_values_max_min_on_interval_coordinates_midpoint_parallel_tangents_l2255_225553

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

end extreme_values_max_min_on_interval_coordinates_midpoint_parallel_tangents_l2255_225553


namespace xiaofang_time_l2255_225531

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

end xiaofang_time_l2255_225531


namespace prove_parabola_points_l2255_225503

open Real

noncomputable def parabola_equation (x y : ℝ) : Prop := x^2 = 4 * y

noncomputable def dist_to_focus (x y focus_x focus_y : ℝ) : ℝ :=
  (sqrt ((x - focus_x)^2 + (y - focus_y)^2))

theorem prove_parabola_points :
  ∀ (x1 y1 x2 y2 : ℝ),
  parabola_equation x1 y1 →
  parabola_equation x2 y2 →
  dist_to_focus x1 y1 0 1 - dist_to_focus x2 y2 0 1 = 2 →
  (y1 + x1^2 - y2 - x2^2 = 10) :=
by
  intros x1 y1 x2 y2 h₁ h₂ h₃
  sorry

end prove_parabola_points_l2255_225503


namespace eccentricity_of_ellipse_l2255_225564

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

end eccentricity_of_ellipse_l2255_225564


namespace function_divisibility_l2255_225582

theorem function_divisibility
    (f : ℤ → ℕ)
    (h_pos : ∀ x, 0 < f x)
    (h_div : ∀ m n : ℤ, (f m - f n) % f (m - n) = 0) :
    ∀ m n : ℤ, f m ≤ f n → f m ∣ f n :=
by sorry

end function_divisibility_l2255_225582


namespace common_ratio_geometric_sequence_l2255_225502

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_arith : 2 * (1/2 * a 5) = a 3 + a 4) : q = (1 + Real.sqrt 5) / 2 :=
sorry

end common_ratio_geometric_sequence_l2255_225502


namespace quotient_is_12_l2255_225534

theorem quotient_is_12 (a b q : ℕ) (h1: q = a / b) (h2: q = a / 2) (h3: q = 6 * b) : q = 12 :=
by 
  sorry

end quotient_is_12_l2255_225534


namespace necessary_but_not_sufficient_l2255_225576

theorem necessary_but_not_sufficient (a : ℝ) : (a ≠ 1) → (a^2 ≠ 1) → (a ≠ 1) ∧ ¬((a ≠ 1) → (a^2 ≠ 1)) :=
by
  sorry

end necessary_but_not_sufficient_l2255_225576


namespace savings_calculation_l2255_225523

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

end savings_calculation_l2255_225523


namespace market_price_article_l2255_225544

theorem market_price_article (P : ℝ)
  (initial_tax_rate : ℝ := 0.035)
  (reduced_tax_rate : ℝ := 0.033333333333333)
  (difference_in_tax : ℝ := 11) :
  (initial_tax_rate * P - reduced_tax_rate * P = difference_in_tax) → 
  P = 6600 :=
by
  intro h
  /-
  We assume h: initial_tax_rate * P - reduced_tax_rate * P = difference_in_tax
  And we need to show P = 6600.
  The proof steps show that P = 6600 follows logically given h and the provided conditions.
  -/
  sorry

end market_price_article_l2255_225544


namespace k_values_l2255_225559

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def find_k (k : ℝ) : Prop :=
  (vector_dot (2, 3) (1, k) = 0) ∨
  (vector_dot (2, 3) (-1, k - 3) = 0) ∨
  (vector_dot (1, k) (-1, k - 3) = 0)

theorem k_values :
  ∃ k : ℝ, find_k k ∧ 
  (k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13) / 2 ∨ k = (3 - Real.sqrt 13 ) / 2) :=
by
  sorry

end k_values_l2255_225559


namespace recreation_percentage_correct_l2255_225509

noncomputable def recreation_percentage (W : ℝ) : ℝ :=
  let recreation_two_weeks_ago := 0.25 * W
  let wages_last_week := 0.95 * W
  let recreation_last_week := 0.35 * (0.95 * W)
  let wages_this_week := 0.95 * W * 0.85
  let recreation_this_week := 0.45 * (0.95 * W * 0.85)
  (recreation_this_week / recreation_two_weeks_ago) * 100

theorem recreation_percentage_correct (W : ℝ) : recreation_percentage W = 145.35 :=
by
  sorry

end recreation_percentage_correct_l2255_225509


namespace theo_drinks_8_cups_per_day_l2255_225516

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

end theo_drinks_8_cups_per_day_l2255_225516


namespace div_by_7_l2255_225593

theorem div_by_7 (n : ℕ) : (3 ^ (12 * n + 1) + 2 ^ (6 * n + 2)) % 7 = 0 := by
  sorry

end div_by_7_l2255_225593


namespace wombats_count_l2255_225572

theorem wombats_count (W : ℕ) (H : 4 * W + 3 = 39) : W = 9 := 
sorry

end wombats_count_l2255_225572


namespace k_value_correct_l2255_225537

theorem k_value_correct (k : ℚ) : 
  let f (x : ℚ) := 4 * x^2 - 3 * x + 5
  let g (x : ℚ) := x^2 + k * x - 8
  (f 5 - g 5 = 20) -> k = 53 / 5 :=
by
  intro h
  sorry

end k_value_correct_l2255_225537


namespace jump_rope_total_l2255_225583

theorem jump_rope_total :
  (56 * 3) + (35 * 4) = 308 :=
by
  sorry

end jump_rope_total_l2255_225583


namespace buy_items_ways_l2255_225539

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l2255_225539


namespace digit_in_2017th_place_l2255_225598

def digit_at_position (n : ℕ) : ℕ := sorry

theorem digit_in_2017th_place :
  digit_at_position 2017 = 7 :=
by sorry

end digit_in_2017th_place_l2255_225598


namespace radius_of_cone_l2255_225529

theorem radius_of_cone (S : ℝ) (h_S: S = 9 * Real.pi) (h_net: net_is_semi_circle) :
  ∃ (r : ℝ), r = Real.sqrt 3 :=
by
  sorry

end radius_of_cone_l2255_225529


namespace inverse_matrix_eigenvalues_l2255_225578

theorem inverse_matrix_eigenvalues 
  (c d : ℝ) 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (eigenvalue1 eigenvalue2 : ℝ) 
  (eigenvector1 eigenvector2 : Fin 2 → ℝ) :
  A = ![![1, 2], ![c, d]] →
  eigenvalue1 = 2 →
  eigenvalue2 = 3 →
  eigenvector1 = ![2, 1] →
  eigenvector2 = ![1, 1] →
  (A.vecMul eigenvector1 = (eigenvalue1 • eigenvector1)) →
  (A.vecMul eigenvector2 = (eigenvalue2 • eigenvector2)) →
  A⁻¹ = ![![2 / 3, -1 / 3], ![1 / 6, 1 / 6]] :=
sorry

end inverse_matrix_eigenvalues_l2255_225578


namespace compare_powers_l2255_225548

-- Definitions for the three numbers
def a : ℝ := 3 ^ 555
def b : ℝ := 4 ^ 444
def c : ℝ := 5 ^ 333

-- Statement to prove
theorem compare_powers : c < a ∧ a < b := sorry

end compare_powers_l2255_225548


namespace q_sufficient_but_not_necessary_for_p_l2255_225579

variable (x : ℝ)

def p : Prop := (x - 2) ^ 2 ≤ 1
def q : Prop := 2 / (x - 1) ≥ 1

theorem q_sufficient_but_not_necessary_for_p : 
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬ q x) := 
by
  sorry

end q_sufficient_but_not_necessary_for_p_l2255_225579


namespace amusement_park_total_cost_l2255_225527

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

end amusement_park_total_cost_l2255_225527


namespace abigail_writing_time_l2255_225555

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

end abigail_writing_time_l2255_225555


namespace emily_has_28_beads_l2255_225561

def beads_per_necklace : ℕ := 7
def necklaces : ℕ := 4

def total_beads : ℕ := necklaces * beads_per_necklace

theorem emily_has_28_beads : total_beads = 28 := by
  sorry

end emily_has_28_beads_l2255_225561


namespace coin_flip_sequences_l2255_225520

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l2255_225520


namespace bud_age_is_eight_l2255_225515

def uncle_age : ℕ := 24

def bud_age (uncle_age : ℕ) : ℕ := uncle_age / 3

theorem bud_age_is_eight : bud_age uncle_age = 8 :=
by
  sorry

end bud_age_is_eight_l2255_225515


namespace find_f4_l2255_225580

theorem find_f4 (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = -f (-x + 1)) 
  (h2 : ∀ x, f (x - 1) = f (-x - 1)) 
  (h3 : f 0 = 2) : 
  f 4 = -2 :=
sorry

end find_f4_l2255_225580


namespace matches_in_each_matchbook_l2255_225510

-- Conditions given in the problem
def one_stamp_worth_matches (s : ℕ) : Prop := s = 12
def tonya_initial_stamps (t : ℕ) : Prop := t = 13
def tonya_final_stamps (t : ℕ) : Prop := t = 3
def jimmy_initial_matchbooks (j : ℕ) : Prop := j = 5

-- Goal: prove M = 24
theorem matches_in_each_matchbook (M : ℕ) (s t_initial t_final j : ℕ) 
  (h1 : one_stamp_worth_matches s) 
  (h2 : tonya_initial_stamps t_initial) 
  (h3 : tonya_final_stamps t_final) 
  (h4 : jimmy_initial_matchbooks j) : M = 24 := by
  sorry

end matches_in_each_matchbook_l2255_225510


namespace maximum_x_plus_y_l2255_225513

theorem maximum_x_plus_y (N x y : ℕ) 
  (hN : N = 19 * x + 95 * y) 
  (hp : ∃ k : ℕ, N = k^2) 
  (hN_le : N ≤ 1995) :
  x + y ≤ 86 :=
sorry

end maximum_x_plus_y_l2255_225513


namespace find_vector_at_t_zero_l2255_225540

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

end find_vector_at_t_zero_l2255_225540


namespace equation_solutions_count_l2255_225519

theorem equation_solutions_count (n : ℕ) :
  (∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 2 * x + 3 * y + z + x^2 = n) →
  (n = 32 ∨ n = 33) :=
sorry

end equation_solutions_count_l2255_225519


namespace seating_arrangement_six_people_l2255_225592

theorem seating_arrangement_six_people : 
  ∃ (n : ℕ), n = 216 ∧ 
  (∀ (a b c d e f : ℕ),
    -- Alice, Bob, and Carla indexing
    1 ≤ a ∧ a ≤ 6 ∧ 
    1 ≤ b ∧ b ≤ 6 ∧ 
    1 ≤ c ∧ c ≤ 6 ∧ 
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
    (a ≠ b + 1 ∧ a ≠ b - 1) ∧
    (a ≠ c + 1 ∧ a ≠ c - 1) ∧
    
    -- Derek, Eric, and Fiona indexing
    1 ≤ d ∧ d ≤ 6 ∧ 
    1 ≤ e ∧ e ≤ 6 ∧ 
    1 ≤ f ∧ f ≤ 6 ∧ 
    d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
    (d ≠ e + 1 ∧ d ≠ e - 1) ∧
    (d ≠ f + 1 ∧ d ≠ f - 1)) -> 
  n = 216 := 
sorry

end seating_arrangement_six_people_l2255_225592


namespace bodies_distance_apart_l2255_225577

def distance_fallen (t : ℝ) : ℝ := 4.9 * t^2

theorem bodies_distance_apart (t : ℝ) (h₁ : 220.5 = distance_fallen t - distance_fallen (t - 5)) : t = 7 :=
by {
  sorry
}

end bodies_distance_apart_l2255_225577


namespace gcd_condition_l2255_225554

def seq (a : ℕ → ℕ) := a 0 = 3 ∧ ∀ n, a (n + 1) - a n = n * (a n - 1)

theorem gcd_condition (a : ℕ → ℕ) (m : ℕ) (h : seq a) :
  m ≥ 2 → (∀ n, Nat.gcd m (a n) = 1) ↔ ∃ k : ℕ, m = 2^k ∧ k ≥ 1 := 
sorry

end gcd_condition_l2255_225554


namespace find_x_l2255_225547

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

end find_x_l2255_225547


namespace find_worst_competitor_l2255_225589

structure Competitor :=
  (name : String)
  (gender : String)
  (generation : String)

-- Define the competitors
def man : Competitor := ⟨"man", "male", "generation1"⟩
def wife : Competitor := ⟨"wife", "female", "generation1"⟩
def son : Competitor := ⟨"son", "male", "generation2"⟩
def sister : Competitor := ⟨"sister", "female", "generation1"⟩

-- Conditions
def opposite_genders (c1 c2 : Competitor) : Prop :=
  c1.gender ≠ c2.gender

def different_generations (c1 c2 : Competitor) : Prop :=
  c1.generation ≠ c2.generation

noncomputable def worst_competitor : Competitor :=
  sister

def is_sibling (c1 c2 : Competitor) : Prop :=
  (c1 = man ∧ c2 = sister) ∨ (c1 = sister ∧ c2 = man)

-- Theorem statement
theorem find_worst_competitor (best_competitor : Competitor) :
  (opposite_genders worst_competitor best_competitor) ∧
  (different_generations worst_competitor best_competitor) ∧
  ∃ (sibling : Competitor), (is_sibling worst_competitor sibling) :=
  sorry

end find_worst_competitor_l2255_225589


namespace regular_price_one_pound_is_20_l2255_225511

variable (y : ℝ)
variable (discounted_price_quarter_pound : ℝ)

-- Conditions
axiom h1 : 0.6 * (y / 4) + 2 = discounted_price_quarter_pound
axiom h2 : discounted_price_quarter_pound = 2
axiom h3 : 0.1 * y = 2

-- Question: What is the regular price for one pound of cake?
theorem regular_price_one_pound_is_20 : y = 20 := 
  sorry

end regular_price_one_pound_is_20_l2255_225511


namespace sum_xyz_l2255_225574

variables {x y z : ℝ}

theorem sum_xyz (hx : x * y = 30) (hy : x * z = 60) (hz : y * z = 90) : 
  x + y + z = 11 * Real.sqrt 5 :=
sorry

end sum_xyz_l2255_225574


namespace sandra_stickers_l2255_225565

theorem sandra_stickers :
  ∃ N : ℕ, N > 1 ∧ (N % 3 = 1) ∧ (N % 5 = 1) ∧ (N % 11 = 1) ∧ N = 166 :=
by {
  sorry
}

end sandra_stickers_l2255_225565


namespace derivative_value_at_pi_over_12_l2255_225599

open Real

theorem derivative_value_at_pi_over_12 :
  let f (x : ℝ) := cos (2 * x + π / 3)
  deriv f (π / 12) = -2 :=
by
  let f (x : ℝ) := cos (2 * x + π / 3)
  sorry

end derivative_value_at_pi_over_12_l2255_225599


namespace probability_at_least_2_defective_is_one_third_l2255_225597

noncomputable def probability_at_least_2_defective (good defective : ℕ) (total_selected : ℕ) : ℚ :=
  let total_ways := Nat.choose (good + defective) total_selected
  let ways_2_defective_1_good := Nat.choose defective 2 * Nat.choose good 1
  let ways_3_defective := Nat.choose defective 3
  (ways_2_defective_1_good + ways_3_defective) / total_ways

theorem probability_at_least_2_defective_is_one_third :
  probability_at_least_2_defective 6 4 3 = 1 / 3 :=
by
  sorry

end probability_at_least_2_defective_is_one_third_l2255_225597


namespace cube_root_product_l2255_225538

theorem cube_root_product : (343 : ℝ)^(1/3) * (125 : ℝ)^(1/3) = 35 := 
by
  sorry

end cube_root_product_l2255_225538


namespace quadratic_equation_roots_transformation_l2255_225514

theorem quadratic_equation_roots_transformation (α β : ℝ) 
  (h1 : 3 * α^2 + 7 * α + 4 = 0)
  (h2 : 3 * β^2 + 7 * β + 4 = 0) :
  ∃ y : ℝ, 21 * y^2 - 23 * y + 6 = 0 :=
sorry

end quadratic_equation_roots_transformation_l2255_225514


namespace range_of_expression_l2255_225581

noncomputable def expression (a b c d : ℝ) : ℝ :=
  Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2) + 
  Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)

theorem range_of_expression (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 2)
  (h3 : 0 ≤ b) (h4 : b ≤ 2) (h5 : 0 ≤ c) (h6 : c ≤ 2)
  (h7 : 0 ≤ d) (h8 : d ≤ 2) :
  4 * Real.sqrt 2 ≤ expression a b c d ∧ expression a b c d ≤ 16 :=
by
  sorry

end range_of_expression_l2255_225581


namespace sum_of_reciprocals_ineq_l2255_225549

theorem sum_of_reciprocals_ineq (a b c : ℝ) (h : a + b + c = 3) : 
  (1 / (5 * a ^ 2 - 4 * a + 11)) + 
  (1 / (5 * b ^ 2 - 4 * b + 11)) + 
  (1 / (5 * c ^ 2 - 4 * c + 11)) ≤ 
  (1 / 4) := 
by {
  sorry
}

end sum_of_reciprocals_ineq_l2255_225549


namespace skye_race_l2255_225541

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

end skye_race_l2255_225541


namespace obtuse_angle_of_parallel_vectors_l2255_225585

noncomputable def is_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem obtuse_angle_of_parallel_vectors (θ : ℝ) :
  let a := (2, 1 - Real.cos θ)
  let b := (1 + Real.cos θ, 1 / 4)
  is_parallel a b → 90 < θ ∧ θ < 180 → θ = 135 :=
by
  intro ha hb
  sorry

end obtuse_angle_of_parallel_vectors_l2255_225585


namespace largest_divisor_of_n_cube_minus_n_minus_six_l2255_225595

theorem largest_divisor_of_n_cube_minus_n_minus_six (n : ℤ) : 6 ∣ (n^3 - n - 6) :=
by sorry

end largest_divisor_of_n_cube_minus_n_minus_six_l2255_225595


namespace find_b_l2255_225506

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x

def derivative_at_one (a : ℝ) : ℝ := a + 1

def tangent_line (b : ℝ) (x : ℝ) : ℝ := 2 * x + b

theorem find_b (a b : ℝ) (h_deriv : derivative_at_one a = 2) (h_tangent : tangent_line b 1 = curve a 1) :
  b = -1 :=
by
  sorry

end find_b_l2255_225506


namespace vasya_max_triangles_l2255_225528

theorem vasya_max_triangles (n : ℕ) (h1 : n = 100)
  (h2 : ∀ (a b c : ℕ), a + b ≤ c ∨ b + c ≤ a ∨ c + a ≤ b) :
  ∃ (t : ℕ), t = n := 
sorry

end vasya_max_triangles_l2255_225528


namespace smallest_n_l2255_225500

def power_tower (a : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => a
  | (n+1) => a ^ (power_tower a n)

def pow3_cubed : ℕ := 3 ^ (3 ^ (3 ^ 3))

theorem smallest_n : ∃ n, (∃ k : ℕ, (power_tower 2 n) = k ∧ k > pow3_cubed) ∧ ∀ m, (∃ k : ℕ, (power_tower 2 m) = k ∧ k > pow3_cubed) → m ≥ n :=
  by
  sorry

end smallest_n_l2255_225500


namespace least_number_of_stamps_is_6_l2255_225563

noncomputable def exist_stamps : Prop :=
∃ (c f : ℕ), 5 * c + 7 * f = 40 ∧ c + f = 6

theorem least_number_of_stamps_is_6 : exist_stamps :=
sorry

end least_number_of_stamps_is_6_l2255_225563


namespace determine_ordered_pair_l2255_225557

theorem determine_ordered_pair (s n : ℤ)
    (h1 : ∀ t : ℤ, ∃ x y : ℤ,
        (x, y) = (s + 2 * t, -3 + n * t)) 
    (h2 : ∀ x y : ℤ, y = 2 * x - 7) :
    (s, n) = (2, 4) :=
by
  sorry

end determine_ordered_pair_l2255_225557


namespace largest_sampled_item_l2255_225530

theorem largest_sampled_item (n : ℕ) (m : ℕ) (a : ℕ) (k : ℕ)
  (hn : n = 360)
  (hm : m = 30)
  (hk : k = n / m)
  (ha : a = 105) :
  ∃ b, b = 433 ∧ (∃ i, i < m ∧ a = 1 + i * k) → (∃ j, j < m ∧ b = 1 + j * k) :=
by
  sorry

end largest_sampled_item_l2255_225530


namespace ratio_of_red_to_total_l2255_225501

def hanna_erasers : Nat := 4
def tanya_total_erasers : Nat := 20

def rachel_erasers (hanna_erasers : Nat) : Nat :=
  hanna_erasers / 2

def tanya_red_erasers (rachel_erasers : Nat) : Nat :=
  2 * (rachel_erasers + 3)

theorem ratio_of_red_to_total (hanna_erasers tanya_total_erasers : Nat)
  (hanna_has_4 : hanna_erasers = 4) 
  (tanya_total_is_20 : tanya_total_erasers = 20) 
  (twice_as_many : hanna_erasers = 2 * (rachel_erasers hanna_erasers)) 
  (three_less_than_half : rachel_erasers hanna_erasers = (1 / 2:Rat) * (tanya_red_erasers (rachel_erasers hanna_erasers)) - 3) :
  (tanya_red_erasers (rachel_erasers hanna_erasers)) / tanya_total_erasers = 1 / 2 := by
  sorry

end ratio_of_red_to_total_l2255_225501


namespace initial_pocket_money_l2255_225575

variable (P : ℝ)

-- Conditions
axiom chocolates_expenditure : P * (1/9) ≥ 0
axiom fruits_expenditure : P * (2/5) ≥ 0
axiom remaining_money : P * (22/45) = 220

-- Theorem statement
theorem initial_pocket_money : P = 450 :=
by
  have h₁ : P * (1/9) + P * (2/5) = P * (23/45) := by sorry
  have h₂ : P * (1 - 23/45) = P * (22/45) := by sorry
  have h₃ : P = 220 / (22/45) := by sorry
  have h₄ : P = 220 * (45/22) := by sorry
  have h₅ : P = 450 := by sorry
  exact h₅

end initial_pocket_money_l2255_225575


namespace megan_homework_problems_l2255_225571

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

end megan_homework_problems_l2255_225571


namespace lines_skew_l2255_225545

def line1 (b : ℝ) (t : ℝ) : ℝ × ℝ × ℝ := 
  (2 + 3 * t, 3 + 2 * t, b + 5 * t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ := 
  (5 + 6 * u, 4 + 3 * u, 1 + 2 * u)

theorem lines_skew (b : ℝ) : 
  ¬ ∃ t u : ℝ, line1 b t = line2 u ↔ b ≠ 4 := 
sorry

end lines_skew_l2255_225545


namespace base6_subtraction_proof_l2255_225586

-- Define the operations needed
def base6_add (a b : Nat) : Nat := sorry
def base6_subtract (a b : Nat) : Nat := sorry

axiom base6_add_correct : ∀ (a b : Nat), base6_add a b = (a + b)
axiom base6_subtract_correct : ∀ (a b : Nat), base6_subtract a b = (if a ≥ b then a - b else 0)

-- Define the problem conditions in base 6
def a := 5*6^2 + 5*6^1 + 5*6^0
def b := 5*6^1 + 5*6^0
def c := 2*6^2 + 0*6^1 + 2*6^0

-- Define the expected result
def result := 6*6^2 + 1*6^1 + 4*6^0

-- State the proof problem
theorem base6_subtraction_proof : base6_subtract (base6_add a b) c = result :=
by
  rw [base6_add_correct, base6_subtract_correct]
  sorry

end base6_subtraction_proof_l2255_225586


namespace stacy_days_to_complete_paper_l2255_225532

variable (total_pages pages_per_day : ℕ)
variable (d : ℕ)

theorem stacy_days_to_complete_paper 
  (h1 : total_pages = 63) 
  (h2 : pages_per_day = 21) 
  (h3 : total_pages = pages_per_day * d) : 
  d = 3 := 
sorry

end stacy_days_to_complete_paper_l2255_225532


namespace determine_q_l2255_225551

theorem determine_q (p q : ℝ) 
  (h : ∀ x : ℝ, (x + 3) * (x + p) = x^2 + q * x + 12) : 
  q = 7 :=
by
  sorry

end determine_q_l2255_225551


namespace correct_transformation_l2255_225542

theorem correct_transformation (x : ℝ) (h : 3 * x - 7 = 2 * x) : 3 * x - 2 * x = 7 :=
sorry

end correct_transformation_l2255_225542


namespace evaluate_polynomial_at_4_l2255_225594

-- Define the polynomial f
noncomputable def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

-- Given x = 4, prove that f(4) = 1559
theorem evaluate_polynomial_at_4 : f 4 = 1559 :=
  by
    sorry

end evaluate_polynomial_at_4_l2255_225594


namespace cost_of_five_juices_l2255_225588

-- Given conditions as assumptions
variables {J S : ℝ}

axiom h1 : 2 * S = 6
axiom h2 : S + J = 5

-- Prove the statement
theorem cost_of_five_juices : 5 * J = 10 :=
sorry

end cost_of_five_juices_l2255_225588


namespace find_constants_l2255_225573

theorem find_constants (t s : ℤ) :
  (∀ x : ℤ, (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + s) = 15 * x^4 - 22 * x^3 + (41 + s) * x^2 - 34 * x + 9 * s) →
  t = -2 ∧ s = s :=
by
  intros h
  sorry

end find_constants_l2255_225573


namespace minimum_a2_plus_4b2_l2255_225570

theorem minimum_a2_plus_4b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 1) : 
  a^2 + 4 * b^2 ≥ 32 :=
sorry

end minimum_a2_plus_4b2_l2255_225570


namespace dice_composite_probability_l2255_225584

theorem dice_composite_probability :
  let total_outcomes := (8:ℕ)^6
  let non_composite_outcomes := 1 + 4 * 6 
  let composite_probability := 1 - (non_composite_outcomes / total_outcomes) 
  composite_probability = 262119 / 262144 := by
  sorry

end dice_composite_probability_l2255_225584


namespace largest_root_l2255_225508

theorem largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -6) (h3 : p * q * r = -8) :
  max (max p q) r = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end largest_root_l2255_225508


namespace find_smallest_n_l2255_225524

theorem find_smallest_n : ∃ n : ℕ, (n - 4)^3 > (n^3 / 2) ∧ ∀ m : ℕ, m < n → (m - 4)^3 ≤ (m^3 / 2) :=
by
  sorry

end find_smallest_n_l2255_225524


namespace speed_of_stream_l2255_225533

theorem speed_of_stream (v c : ℝ) (h1 : c - v = 6) (h2 : c + v = 10) : v = 2 :=
by
  sorry

end speed_of_stream_l2255_225533


namespace neg_p_necessary_but_not_sufficient_for_neg_q_l2255_225512

variable (p q : Prop)

theorem neg_p_necessary_but_not_sufficient_for_neg_q
  (h1 : p → q)
  (h2 : ¬ (q → p)) : 
  (¬p → ¬q) ∧ (¬q → ¬p) := 
sorry

end neg_p_necessary_but_not_sufficient_for_neg_q_l2255_225512


namespace sum_of_products_of_roots_l2255_225546

theorem sum_of_products_of_roots (p q r : ℂ) (h : 4 * (p^3) - 2 * (p^2) + 13 * p - 9 = 0 ∧ 4 * (q^3) - 2 * (q^2) + 13 * q - 9 = 0 ∧ 4 * (r^3) - 2 * (r^2) + 13 * r - 9 = 0) :
  p*q + p*r + q*r = 13 / 4 :=
  sorry

end sum_of_products_of_roots_l2255_225546


namespace marks_lost_per_wrong_answer_l2255_225521

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

end marks_lost_per_wrong_answer_l2255_225521


namespace stratified_sampling_example_l2255_225567

theorem stratified_sampling_example
  (students_ratio : ℕ → ℕ) -- function to get the number of students in each grade, indexed by natural numbers
  (ratio_cond : students_ratio 0 = 4 ∧ students_ratio 1 = 3 ∧ students_ratio 2 = 2) -- the ratio 4:3:2
  (third_grade_sample : ℕ) -- number of students in the third grade in the sample
  (third_grade_sample_eq : third_grade_sample = 10) -- 10 students from the third grade
  (total_sample_size : ℕ) -- the sample size n
 :
  total_sample_size = 45 := 
sorry

end stratified_sampling_example_l2255_225567


namespace find_k_such_that_product_minus_one_is_perfect_power_l2255_225568

noncomputable def product_of_first_n_primes (n : ℕ) : ℕ :=
  (List.take n (List.filter (Nat.Prime) (List.range n.succ))).prod

theorem find_k_such_that_product_minus_one_is_perfect_power :
  ∀ k : ℕ, ∃ a n : ℕ, (product_of_first_n_primes k) - 1 = a^n ∧ n > 1 ∧ k = 1 :=
by
  sorry

end find_k_such_that_product_minus_one_is_perfect_power_l2255_225568
