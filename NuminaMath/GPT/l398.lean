import Mathlib

namespace NUMINAMATH_GPT_distance_around_track_l398_39840

-- Define the conditions
def total_mileage : ℝ := 10
def distance_to_high_school : ℝ := 3
def round_trip_distance : ℝ := 2 * distance_to_high_school

-- State the question and the desired proof problem
theorem distance_around_track : 
  total_mileage - round_trip_distance = 4 := 
by
  sorry

end NUMINAMATH_GPT_distance_around_track_l398_39840


namespace NUMINAMATH_GPT_mutually_exclusive_event_l398_39849

-- Define the events
def hits_first_shot : Prop := sorry  -- Placeholder for "hits the target on the first shot"
def hits_second_shot : Prop := sorry  -- Placeholder for "hits the target on the second shot"
def misses_first_shot : Prop := ¬ hits_first_shot
def misses_second_shot : Prop := ¬ hits_second_shot

-- Define the main events in the problem
def hitting_at_least_once : Prop := hits_first_shot ∨ hits_second_shot
def missing_both_times : Prop := misses_first_shot ∧ misses_second_shot

-- Statement of the theorem
theorem mutually_exclusive_event :
  missing_both_times ↔ ¬ hitting_at_least_once :=
by sorry

end NUMINAMATH_GPT_mutually_exclusive_event_l398_39849


namespace NUMINAMATH_GPT_total_chocolate_bars_l398_39809

theorem total_chocolate_bars (small_boxes : ℕ) (bars_per_box : ℕ) 
  (h1 : small_boxes = 17) (h2 : bars_per_box = 26) 
  : small_boxes * bars_per_box = 442 :=
by sorry

end NUMINAMATH_GPT_total_chocolate_bars_l398_39809


namespace NUMINAMATH_GPT_least_possible_value_z_minus_x_l398_39861

theorem least_possible_value_z_minus_x
  (x y z : ℤ)
  (h1 : x < y)
  (h2 : y < z)
  (h3 : y - x > 5)
  (hx_even : x % 2 = 0)
  (hy_odd : y % 2 = 1)
  (hz_odd : z % 2 = 1) :
  z - x = 9 :=
  sorry

end NUMINAMATH_GPT_least_possible_value_z_minus_x_l398_39861


namespace NUMINAMATH_GPT_original_price_of_lens_is_correct_l398_39885

-- Definitions based on conditions
def current_camera_price : ℝ := 4000
def new_camera_price : ℝ := current_camera_price + 0.30 * current_camera_price
def combined_price_paid : ℝ := 5400
def lens_discount : ℝ := 200
def combined_price_before_discount : ℝ := combined_price_paid + lens_discount

-- Calculated original price of the lens
def lens_original_price : ℝ := combined_price_before_discount - new_camera_price

-- The Lean theorem statement to prove the price is correct
theorem original_price_of_lens_is_correct : lens_original_price = 400 := by
  -- You do not need to provide the actual proof steps
  sorry

end NUMINAMATH_GPT_original_price_of_lens_is_correct_l398_39885


namespace NUMINAMATH_GPT_min_value_expr_l398_39873

theorem min_value_expr (x : ℝ) (h : x > 1) : ∃ m, m = 5 ∧ ∀ y, y = x + 4 / (x - 1) → y ≥ m :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l398_39873


namespace NUMINAMATH_GPT_sequence_of_arrows_from_425_to_427_l398_39816

theorem sequence_of_arrows_from_425_to_427 :
  ∀ (arrows : ℕ → ℕ), (∀ n, arrows (n + 4) = arrows n) →
  (arrows 425, arrows 426, arrows 427) = (arrows 1, arrows 2, arrows 3) :=
by
  intros arrows h_period
  have h1 : arrows 425 = arrows 1 := by 
    sorry
  have h2 : arrows 426 = arrows 2 := by 
    sorry
  have h3 : arrows 427 = arrows 3 := by 
    sorry
  sorry

end NUMINAMATH_GPT_sequence_of_arrows_from_425_to_427_l398_39816


namespace NUMINAMATH_GPT_area_of_rectangle_l398_39814

theorem area_of_rectangle (w l : ℕ) (hw : w = 10) (hl : l = 2) : (w * l) = 20 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l398_39814


namespace NUMINAMATH_GPT_find_a1000_l398_39828

noncomputable def seq (a : ℕ → ℤ) : Prop :=
a 1 = 1009 ∧
a 2 = 1010 ∧
(∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n)

theorem find_a1000 (a : ℕ → ℤ) (h : seq a) : a 1000 = 1675 :=
sorry

end NUMINAMATH_GPT_find_a1000_l398_39828


namespace NUMINAMATH_GPT_table_covered_area_l398_39866

-- Definitions based on conditions
def length := 12
def width := 1
def number_of_strips := 4
def overlapping_strips := 3

-- Calculating the area of one strip
def area_of_one_strip := length * width

-- Calculating total area assuming no overlaps
def total_area_no_overlap := number_of_strips * area_of_one_strip

-- Calculating the total overlap area
def overlap_area := overlapping_strips * (width * width)

-- Final area after subtracting overlaps
def final_covered_area := total_area_no_overlap - overlap_area

-- Theorem stating the proof problem
theorem table_covered_area : final_covered_area = 45 :=
by
  sorry

end NUMINAMATH_GPT_table_covered_area_l398_39866


namespace NUMINAMATH_GPT_total_seeds_in_watermelon_l398_39894

theorem total_seeds_in_watermelon :
  let slices := 40
  let black_seeds_per_slice := 20
  let white_seeds_per_slice := 20
  let total_black_seeds := black_seeds_per_slice * slices
  let total_white_seeds := white_seeds_per_slice * slices
  total_black_seeds + total_white_seeds = 1600 := by
  sorry

end NUMINAMATH_GPT_total_seeds_in_watermelon_l398_39894


namespace NUMINAMATH_GPT_odd_integer_95th_l398_39813

theorem odd_integer_95th : (2 * 95 - 1) = 189 := 
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_odd_integer_95th_l398_39813


namespace NUMINAMATH_GPT_find_n_tan_eq_l398_39822

theorem find_n_tan_eq (n : ℝ) (h1 : -180 < n) (h2 : n < 180) (h3 : Real.tan (n * Real.pi / 180) = Real.tan (678 * Real.pi / 180)) : 
  n = 138 := 
sorry

end NUMINAMATH_GPT_find_n_tan_eq_l398_39822


namespace NUMINAMATH_GPT_rationalize_denominator_correct_l398_39811

noncomputable def rationalize_denominator : Prop :=
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_correct_l398_39811


namespace NUMINAMATH_GPT_rectangle_diagonal_opposite_vertex_l398_39870

theorem rectangle_diagonal_opposite_vertex :
  ∀ (x y : ℝ),
    (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
      (x1, y1) = (5, 10) ∧ (x2, y2) = (15, -6) ∧ (x3, y3) = (11, 2) ∧
      (∃ (mx my : ℝ), mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2 ∧
        mx = (x + x3) / 2 ∧ my = (y + y3) / 2) ∧
      x = 9 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_opposite_vertex_l398_39870


namespace NUMINAMATH_GPT_original_price_color_TV_l398_39871

theorem original_price_color_TV (x : ℝ) 
  (h : 1.12 * x - x = 144) : 
  x = 1200 :=
sorry

end NUMINAMATH_GPT_original_price_color_TV_l398_39871


namespace NUMINAMATH_GPT_total_brown_mms_3rd_4th_bags_l398_39878

def brown_mms_in_bags := (9 : ℕ) + (12 : ℕ) + (3 : ℕ)

def total_bags := 5

def average_mms_per_bag := 8

theorem total_brown_mms_3rd_4th_bags (x y : ℕ) 
  (h1 : brown_mms_in_bags + x + y = average_mms_per_bag * total_bags) : 
  x + y = 16 :=
by
  have h2 : brown_mms_in_bags + x + y = 40 := by sorry
  sorry

end NUMINAMATH_GPT_total_brown_mms_3rd_4th_bags_l398_39878


namespace NUMINAMATH_GPT_probability_of_pink_l398_39802

variable (B P : ℕ) -- number of blue and pink gumballs
variable (h_total : B + P > 0) -- there is at least one gumball in the jar
variable (h_prob_two_blue : (B / (B + P)) * (B / (B + P)) = 16 / 49) -- the probability of drawing two blue gumballs in a row

theorem probability_of_pink : (P / (B + P)) = 3 / 7 :=
sorry

end NUMINAMATH_GPT_probability_of_pink_l398_39802


namespace NUMINAMATH_GPT_volume_of_mixture_l398_39852

theorem volume_of_mixture
    (weight_a : ℝ) (weight_b : ℝ) (ratio_a_b : ℝ) (total_weight : ℝ)
    (h1 : weight_a = 900) (h2 : weight_b = 700)
    (h3 : ratio_a_b = 3/2) (h4 : total_weight = 3280) :
    ∃ Va Vb : ℝ, (Va / Vb = ratio_a_b) ∧ (weight_a * Va + weight_b * Vb = total_weight) ∧ (Va + Vb = 4) := 
by
  sorry

end NUMINAMATH_GPT_volume_of_mixture_l398_39852


namespace NUMINAMATH_GPT_point_in_second_quadrant_l398_39808

theorem point_in_second_quadrant (a : ℝ) : 
  ∃ q : ℕ, q = 2 ∧ (-1, a^2 + 1).1 < 0 ∧ 0 < (-1, a^2 + 1).2 :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l398_39808


namespace NUMINAMATH_GPT_chocolate_cost_is_3_l398_39807

-- Definitions based on the conditions
def dan_has_5_dollars : Prop := true
def cost_candy_bar : ℕ := 2
def cost_chocolate : ℕ := cost_candy_bar + 1

-- Theorem to prove
theorem chocolate_cost_is_3 : cost_chocolate = 3 :=
by {
  -- This is where the proof steps would go
  sorry
}

end NUMINAMATH_GPT_chocolate_cost_is_3_l398_39807


namespace NUMINAMATH_GPT_area_triangle_ABC_l398_39855

theorem area_triangle_ABC (x y : ℝ) (h : x * y ≠ 0) (hAOB : 1 / 2 * |x * y| = 4) : 
  1 / 2 * |(x * (-2 * y) + x * (2 * y) + (-x) * (2 * y))| = 8 :=
by
  sorry

end NUMINAMATH_GPT_area_triangle_ABC_l398_39855


namespace NUMINAMATH_GPT_smallest_n_mult_y_perfect_cube_l398_39881

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9

theorem smallest_n_mult_y_perfect_cube : ∃ n : ℕ, (∀ m : ℕ, y * n = m^3 → n = 1500) :=
sorry

end NUMINAMATH_GPT_smallest_n_mult_y_perfect_cube_l398_39881


namespace NUMINAMATH_GPT_find_L_for_perfect_square_W_l398_39864

theorem find_L_for_perfect_square_W :
  ∃ L W : ℕ, 1000 < W ∧ W < 2000 ∧ L > 1 ∧ W = 2 * L^3 ∧ ∃ m : ℕ, W = m^2 ∧ L = 8 :=
by sorry

end NUMINAMATH_GPT_find_L_for_perfect_square_W_l398_39864


namespace NUMINAMATH_GPT_no_real_solutions_for_inequality_l398_39869

theorem no_real_solutions_for_inequality (a : ℝ) :
  ¬∃ x : ℝ, ∀ y : ℝ, |(x^2 + a*x + 2*a)| ≤ 5 → y = x :=
sorry

end NUMINAMATH_GPT_no_real_solutions_for_inequality_l398_39869


namespace NUMINAMATH_GPT_part1_part2_l398_39872
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (3 * x - 1) + a * x + 3

theorem part1 (x : ℝ) : (f x 1) ≤ 5 ↔ (-1/2 : ℝ) ≤ x ∧ x ≤ 3/4 := by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, ∀ y : ℝ, f x a ≥ f y a) ↔ (-3 : ℝ) ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l398_39872


namespace NUMINAMATH_GPT_train_speed_l398_39831

-- Define the conditions in terms of distance and time
def train_length : ℕ := 160
def crossing_time : ℕ := 8

-- Define the expected speed
def expected_speed : ℕ := 20

-- The theorem stating the speed of the train given the conditions
theorem train_speed : (train_length / crossing_time) = expected_speed :=
by
  -- Note: The proof is omitted
  sorry

end NUMINAMATH_GPT_train_speed_l398_39831


namespace NUMINAMATH_GPT_adam_remaining_loads_l398_39850

-- Define the initial conditions
def total_loads : ℕ := 25
def washed_loads : ℕ := 6

-- Define the remaining loads as the total loads minus the washed loads
def remaining_loads (total_loads washed_loads : ℕ) : ℕ := total_loads - washed_loads

-- State the theorem to be proved
theorem adam_remaining_loads : remaining_loads total_loads washed_loads = 19 := by
  sorry

end NUMINAMATH_GPT_adam_remaining_loads_l398_39850


namespace NUMINAMATH_GPT_total_weight_of_snacks_l398_39810

-- Definitions for conditions
def weight_peanuts := 0.1
def weight_raisins := 0.4
def weight_almonds := 0.3

-- Theorem statement
theorem total_weight_of_snacks : weight_peanuts + weight_raisins + weight_almonds = 0.8 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_snacks_l398_39810


namespace NUMINAMATH_GPT_intersection_PQ_l398_39837

def P := {x : ℝ | x < 1}
def Q := {x : ℝ | x^2 < 4}
def PQ_intersection := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_PQ : P ∩ Q = PQ_intersection := by
  sorry

end NUMINAMATH_GPT_intersection_PQ_l398_39837


namespace NUMINAMATH_GPT_find_standard_equation_of_ellipse_l398_39819

noncomputable def ellipse_equation (a c b : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∨ (y^2 / a^2 + x^2 / b^2 = 1)

theorem find_standard_equation_of_ellipse (h1 : 2 * a = 12) (h2 : c / a = 1 / 3) :
  ellipse_equation 6 2 4 :=
by
  -- We are proving that given the conditions, the standard equation of the ellipse is as stated
  sorry

end NUMINAMATH_GPT_find_standard_equation_of_ellipse_l398_39819


namespace NUMINAMATH_GPT_arithmetic_sequence_l398_39889

theorem arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h_n : n > 0) 
  (h_Sn : S (2 * n) - S (2 * n - 1) + a 2 = 424) : 
  a (n + 1) = 212 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_l398_39889


namespace NUMINAMATH_GPT_whale_plankton_feeding_frenzy_l398_39845

theorem whale_plankton_feeding_frenzy
  (x y : ℕ)
  (h1 : x + 5 * y = 54)
  (h2 : 9 * x + 36 * y = 450) :
  y = 4 :=
sorry

end NUMINAMATH_GPT_whale_plankton_feeding_frenzy_l398_39845


namespace NUMINAMATH_GPT_pit_A_no_replant_exactly_one_pit_no_replant_at_least_one_replant_l398_39851

noncomputable def pit_a_no_replant_prob : ℝ := 0.875
noncomputable def one_pit_no_replant_prob : ℝ := 0.713
noncomputable def at_least_one_pit_replant_prob : ℝ := 0.330

theorem pit_A_no_replant (p : ℝ) (h1 : p = 0.5) : pit_a_no_replant_prob = 1 - (1 - p)^3 := by
  sorry

theorem exactly_one_pit_no_replant (p : ℝ) (h1 : p = 0.5) : one_pit_no_replant_prob = 1 - 3 * (1 - p)^3 * (p^3)^(2) := by
  sorry

theorem at_least_one_replant (p : ℝ) (h1 : p = 0.5) : at_least_one_pit_replant_prob = 1 - (1 - (1 - p)^3)^3 := by
  sorry

end NUMINAMATH_GPT_pit_A_no_replant_exactly_one_pit_no_replant_at_least_one_replant_l398_39851


namespace NUMINAMATH_GPT_smallest_integer_remainder_conditions_l398_39895

theorem smallest_integer_remainder_conditions :
  ∃ b : ℕ, (b % 3 = 0) ∧ (b % 4 = 2) ∧ (b % 5 = 3) ∧ (∀ n : ℕ, (n % 3 = 0) ∧ (n % 4 = 2) ∧ (n % 5 = 3) → b ≤ n) :=
sorry

end NUMINAMATH_GPT_smallest_integer_remainder_conditions_l398_39895


namespace NUMINAMATH_GPT_find_x_l398_39883

variable (x : ℝ)

def length := 4 * x
def width := x + 3

def area := length x * width x
def perimeter := 2 * length x + 2 * width x

theorem find_x (h : area x = 3 * perimeter x) : x = 5.342 := by
  sorry

end NUMINAMATH_GPT_find_x_l398_39883


namespace NUMINAMATH_GPT_dodecahedron_interior_diagonals_l398_39846

-- Definitions based on conditions
def dodecahedron_vertices : ℕ := 20
def vertices_connected_by_edges (v : ℕ) : ℕ := 3
def potential_internal_diagonals (v : ℕ) : ℕ := dodecahedron_vertices - vertices_connected_by_edges v - 1

-- Main statement to prove
theorem dodecahedron_interior_diagonals : (dodecahedron_vertices * potential_internal_diagonals 0) / 2 = 160 := by sorry

end NUMINAMATH_GPT_dodecahedron_interior_diagonals_l398_39846


namespace NUMINAMATH_GPT_work_completion_time_l398_39874

theorem work_completion_time (A_rate B_rate : ℝ) (hA : A_rate = 1/60) (hB : B_rate = 1/20) :
  1 / (A_rate + B_rate) = 15 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_time_l398_39874


namespace NUMINAMATH_GPT_quadratic_equal_roots_iff_a_eq_4_l398_39856

theorem quadratic_equal_roots_iff_a_eq_4 (a : ℝ) (h : ∃ x : ℝ, (a * x^2 - 4 * x + 1 = 0) ∧ (a * x^2 - 4 * x + 1 = 0)) :
  a = 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equal_roots_iff_a_eq_4_l398_39856


namespace NUMINAMATH_GPT_problem1_solution_correct_problem2_solution_correct_l398_39863

def problem1 (x : ℤ) : Prop := (x - 1) ∣ (x + 3)
def problem2 (x : ℤ) : Prop := (x + 2) ∣ (x^2 + 2)
def solution1 (x : ℤ) : Prop := x = -3 ∨ x = -1 ∨ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5
def solution2 (x : ℤ) : Prop := x = -8 ∨ x = -5 ∨ x = -4 ∨ x = -3 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 4

theorem problem1_solution_correct : ∀ x: ℤ, problem1 x ↔ solution1 x := by
  sorry

theorem problem2_solution_correct : ∀ x: ℤ, problem2 x ↔ solution2 x := by
  sorry

end NUMINAMATH_GPT_problem1_solution_correct_problem2_solution_correct_l398_39863


namespace NUMINAMATH_GPT_possible_third_side_l398_39879

theorem possible_third_side {x : ℕ} (h_option_A : x = 2) (h_option_B : x = 3) (h_option_C : x = 6) (h_option_D : x = 13) : 3 < x ∧ x < 13 ↔ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_possible_third_side_l398_39879


namespace NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l398_39859

namespace Inequalities

theorem option_A (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  a + (1/a) > b + (1/b) :=
sorry

theorem option_B (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  (m + 1) / (n + 1) < m / n :=
sorry

theorem option_C (c a b : ℝ) (hc : c > 0) (ha : a > 0) (hb : b > 0) (hca : c > a) (hab : a > b) :
  a / (c - a) > b / (c - b) :=
sorry

theorem option_D (a b : ℝ) (ha : a > -1) (hb : b > -1) (hab : a ≥ b) :
  a / (a + 1) ≥ b / (b + 1) :=
sorry

end Inequalities

end NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l398_39859


namespace NUMINAMATH_GPT_shaded_area_l398_39804

open Real

theorem shaded_area (AH HF GF : ℝ) (AH_eq : AH = 12) (HF_eq : HF = 16) (GF_eq : GF = 4) 
  (DG : ℝ) (DG_eq : DG = 3) (area_triangle_DGF : ℝ) (area_triangle_DGF_eq : area_triangle_DGF = 6) :
  let area_square : ℝ := 4 * 4
  let shaded_area : ℝ := area_square - area_triangle_DGF
  shaded_area = 10 := by
    sorry

end NUMINAMATH_GPT_shaded_area_l398_39804


namespace NUMINAMATH_GPT_find_y_given_x_eq_neg6_l398_39891

theorem find_y_given_x_eq_neg6 :
  ∀ (y : ℤ), (∃ (x : ℤ), x = -6 ∧ x^2 - x + 6 = y - 6) → y = 54 :=
by
  intros y h
  obtain ⟨x, hx1, hx2⟩ := h
  rw [hx1] at hx2
  simp at hx2
  linarith

end NUMINAMATH_GPT_find_y_given_x_eq_neg6_l398_39891


namespace NUMINAMATH_GPT_units_digit_47_power_47_l398_39868

theorem units_digit_47_power_47 : (47^47) % 10 = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_47_power_47_l398_39868


namespace NUMINAMATH_GPT_avg_age_adults_l398_39825

-- Given conditions
def num_members : ℕ := 50
def avg_age_members : ℕ := 20
def num_girls : ℕ := 25
def num_boys : ℕ := 20
def num_adults : ℕ := 5
def avg_age_girls : ℕ := 18
def avg_age_boys : ℕ := 22

-- Prove that the average age of the adults is 22 years
theorem avg_age_adults :
  (num_members * avg_age_members - num_girls * avg_age_girls - num_boys * avg_age_boys) / num_adults = 22 :=
by 
  sorry

end NUMINAMATH_GPT_avg_age_adults_l398_39825


namespace NUMINAMATH_GPT_problem_solution_l398_39830

noncomputable def proof_problem (x1 x2 x3 x4 x5 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) : Prop :=
  ((x1^2 - x3 * x5) * (x2^2 - x3 * x5) ≤ 0) ∧
  ((x2^2 - x4 * x1) * (x3^2 - x4 * x1) ≤ 0) ∧
  ((x3^2 - x5 * x2) * (x4^2 - x5 * x2) ≤ 0) ∧
  ((x4^2 - x1 * x3) * (x5^2 - x1 * x3) ≤ 0) ∧
  ((x5^2 - x2 * x4) * (x1^2 - x2 * x4) ≤ 0) → 
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5

theorem problem_solution (x1 x2 x3 x4 x5 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) :
  proof_problem x1 x2 x3 x4 x5 h1 h2 h3 h4 h5 :=
  by
    sorry

end NUMINAMATH_GPT_problem_solution_l398_39830


namespace NUMINAMATH_GPT_show_length_50_l398_39826

def Gina_sSis_three_as_often (G S : ℕ) : Prop := G = 3 * S
def sister_total_shows (G S : ℕ) : Prop := G + S = 24
def Gina_total_minutes (G : ℕ) (minutes : ℕ) : Prop := minutes = 900
def length_of_each_show (minutes shows length : ℕ) : Prop := length = minutes / shows

theorem show_length_50 (G S : ℕ) (length : ℕ) :
  Gina_sSis_three_as_often G S →
  sister_total_shows G S →
  Gina_total_minutes G 900 →
  length_of_each_show 900 G length →
  length = 50 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_show_length_50_l398_39826


namespace NUMINAMATH_GPT_smallest_integer_in_set_l398_39827

theorem smallest_integer_in_set (n : ℤ) (h : n+4 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) / 5)) : n ≥ 0 :=
by sorry

end NUMINAMATH_GPT_smallest_integer_in_set_l398_39827


namespace NUMINAMATH_GPT_valid_p_values_l398_39842

theorem valid_p_values (p : ℕ) (h : p = 3 ∨ p = 4 ∨ p = 5 ∨ p = 12) :
  0 < (4 * p + 34) / (3 * p - 8) ∧ (4 * p + 34) % (3 * p - 8) = 0 :=
by
  sorry

end NUMINAMATH_GPT_valid_p_values_l398_39842


namespace NUMINAMATH_GPT_integer_solutions_positive_product_l398_39897

theorem integer_solutions_positive_product :
  {a : ℤ | (5 + a) * (3 - a) > 0} = {-4, -3, -2, -1, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_positive_product_l398_39897


namespace NUMINAMATH_GPT_first_term_of_geometric_series_l398_39835

theorem first_term_of_geometric_series (a r S : ℝ) (h₁ : r = 1/4) (h₂ : S = 80) (h₃ : S = a / (1 - r)) : a = 60 :=
by
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_series_l398_39835


namespace NUMINAMATH_GPT_lifespan_difference_l398_39890

variable (H : ℕ)

theorem lifespan_difference (H : ℕ) (bat_lifespan : ℕ) (frog_lifespan : ℕ) (total_lifespan : ℕ) 
    (hb : bat_lifespan = 10)
    (hf : frog_lifespan = 4 * H)
    (ht : H + bat_lifespan + frog_lifespan = total_lifespan)
    (t30 : total_lifespan = 30) :
    bat_lifespan - H = 6 :=
by
  -- here would be the proof
  sorry

end NUMINAMATH_GPT_lifespan_difference_l398_39890


namespace NUMINAMATH_GPT_div_by_9_digit_B_l398_39892

theorem div_by_9_digit_B (B : ℕ) (h : (4 + B + B + 2) % 9 = 0) : B = 6 :=
by sorry

end NUMINAMATH_GPT_div_by_9_digit_B_l398_39892


namespace NUMINAMATH_GPT_sprint_team_total_miles_l398_39832

theorem sprint_team_total_miles (number_of_people : ℝ) (miles_per_person : ℝ) 
  (h1 : number_of_people = 150.0) (h2 : miles_per_person = 5.0) : 
  number_of_people * miles_per_person = 750.0 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_sprint_team_total_miles_l398_39832


namespace NUMINAMATH_GPT_solve_double_inequality_l398_39833

theorem solve_double_inequality (x : ℝ) :
  (-1 < (x^2 - 20 * x + 21) / (x^2 - 4 * x + 5) ∧
   (x^2 - 20 * x + 21) / (x^2 - 4 * x + 5) < 1) ↔ (2 < x ∨ 26 < x) := 
sorry

end NUMINAMATH_GPT_solve_double_inequality_l398_39833


namespace NUMINAMATH_GPT_complement_of_A_in_U_l398_39887

def U : Set ℝ := {x | x ≤ 1}
def A : Set ℝ := {x | x < 0}

theorem complement_of_A_in_U : (U \ A) = {x | 0 ≤ x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l398_39887


namespace NUMINAMATH_GPT_value_of_x_plus_y_l398_39854

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem value_of_x_plus_y
  (x y : ℝ)
  (h1 : x ≥ 1)
  (h2 : y ≥ 1)
  (h3 : x * y = 10)
  (h4 : x^(lg x) * y^(lg y) ≥ 10) :
  x + y = 11 :=
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l398_39854


namespace NUMINAMATH_GPT_jenny_ran_further_l398_39858

-- Define the distances Jenny ran and walked
def ran_distance : ℝ := 0.6
def walked_distance : ℝ := 0.4

-- Define the difference between the distances Jenny ran and walked
def difference : ℝ := ran_distance - walked_distance

-- The proof statement
theorem jenny_ran_further : difference = 0.2 := by
  sorry

end NUMINAMATH_GPT_jenny_ran_further_l398_39858


namespace NUMINAMATH_GPT_increase_by_1_or_prime_l398_39888

theorem increase_by_1_or_prime (a : ℕ → ℕ) :
  a 0 = 6 →
  (∀ n, a (n + 1) = a n + Nat.gcd (a n) (n + 1)) →
  ∀ n, n < 1000000 → (∃ p, p = 1 ∨ Nat.Prime p ∧ a (n + 1) = a n + p) :=
by
  intro ha0 ha_step
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_increase_by_1_or_prime_l398_39888


namespace NUMINAMATH_GPT_domain_lg_function_l398_39848

theorem domain_lg_function (x : ℝ) : (1 + x > 0 ∧ x - 1 > 0) ↔ (1 < x) :=
by
  sorry

end NUMINAMATH_GPT_domain_lg_function_l398_39848


namespace NUMINAMATH_GPT_probability_of_yellow_ball_l398_39801

theorem probability_of_yellow_ball 
  (red_balls : ℕ) 
  (yellow_balls : ℕ) 
  (blue_balls : ℕ) 
  (total_balls : ℕ)
  (h1 : red_balls = 2)
  (h2 : yellow_balls = 5)
  (h3 : blue_balls = 4)
  (h4 : total_balls = red_balls + yellow_balls + blue_balls) :
  (yellow_balls / total_balls : ℚ) = 5 / 11 :=
by 
  rw [h1, h2, h3] at h4  -- Substitute the ball counts into the total_balls definition.
  norm_num at h4  -- Simplify to verify the total is indeed 11.
  rw [h2, h4] -- Use the number of yellow balls and total number of balls to state the ratio.
  norm_num -- Normalize the fraction to show it equals 5/11.

#check probability_of_yellow_ball

end NUMINAMATH_GPT_probability_of_yellow_ball_l398_39801


namespace NUMINAMATH_GPT_find_first_number_l398_39862

theorem find_first_number (y x : ℤ) (h1 : (y + 76 + x) / 3 = 5) (h2 : x = -63) : y = 2 :=
by
  -- To be filled in with the proof steps
  sorry

end NUMINAMATH_GPT_find_first_number_l398_39862


namespace NUMINAMATH_GPT_sum_remainder_l398_39841

theorem sum_remainder (a b c d : ℤ) (h1 : a % 53 = 33) (h2 : b % 53 = 11) 
                       (h3 : c % 53 = 49) (h4 : d % 53 = 2) :
  (a + b + c + d) % 53 = 42 :=
sorry

end NUMINAMATH_GPT_sum_remainder_l398_39841


namespace NUMINAMATH_GPT_tiffany_lives_after_bonus_stage_l398_39896

theorem tiffany_lives_after_bonus_stage :
  let initial_lives := 250
  let lives_lost := 58
  let remaining_lives := initial_lives - lives_lost
  let additional_lives := 3 * remaining_lives
  let final_lives := remaining_lives + additional_lives
  final_lives = 768 :=
by
  let initial_lives := 250
  let lives_lost := 58
  let remaining_lives := initial_lives - lives_lost
  let additional_lives := 3 * remaining_lives
  let final_lives := remaining_lives + additional_lives
  exact sorry

end NUMINAMATH_GPT_tiffany_lives_after_bonus_stage_l398_39896


namespace NUMINAMATH_GPT_sum_nat_numbers_l398_39805

/-- 
If S is the set of all natural numbers n such that 0 ≤ n ≤ 200, n ≡ 7 [MOD 11], 
and n ≡ 5 [MOD 7], then the sum of elements in S is 351.
-/
theorem sum_nat_numbers (S : Finset ℕ) 
  (hs : ∀ n, n ∈ S ↔ n ≤ 200 ∧ n % 11 = 7 ∧ n % 7 = 5) 
  : S.sum id = 351 := 
sorry 

end NUMINAMATH_GPT_sum_nat_numbers_l398_39805


namespace NUMINAMATH_GPT_evaluate_expression_l398_39898

theorem evaluate_expression : 
  60 + 120 / 15 + 25 * 16 - 220 - 420 / 7 + 3 ^ 2 = 197 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l398_39898


namespace NUMINAMATH_GPT_age_difference_between_two_children_l398_39800

theorem age_difference_between_two_children 
  (avg_age_10_years_ago : ℕ)
  (present_avg_age : ℕ)
  (youngest_child_present_age : ℕ)
  (initial_family_members : ℕ)
  (current_family_members : ℕ)
  (H1 : avg_age_10_years_ago = 24)
  (H2 : present_avg_age = 24)
  (H3 : youngest_child_present_age = 3)
  (H4 : initial_family_members = 4)
  (H5 : current_family_members = 6) :
  ∃ (D: ℕ), D = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_between_two_children_l398_39800


namespace NUMINAMATH_GPT_percentage_reduction_is_20_l398_39875

noncomputable def reduction_in_length (L W : ℝ) (x : ℝ) := 
  (L * (1 - x / 100)) * (W * 1.25) = L * W

theorem percentage_reduction_is_20 (L W : ℝ) : 
  reduction_in_length L W 20 := 
by 
  unfold reduction_in_length
  sorry

end NUMINAMATH_GPT_percentage_reduction_is_20_l398_39875


namespace NUMINAMATH_GPT_find_x_tan_identity_l398_39876

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end NUMINAMATH_GPT_find_x_tan_identity_l398_39876


namespace NUMINAMATH_GPT_sin_240_eq_neg_sqrt3_over_2_l398_39821

theorem sin_240_eq_neg_sqrt3_over_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_240_eq_neg_sqrt3_over_2_l398_39821


namespace NUMINAMATH_GPT_proof_M1M2_product_l398_39836

theorem proof_M1M2_product : 
  (∀ x, (45 * x - 34) / (x^2 - 4 * x + 3) = M_1 / (x - 1) + M_2 / (x - 3)) →
  M_1 * M_2 = -1111 / 4 := 
by
  sorry

end NUMINAMATH_GPT_proof_M1M2_product_l398_39836


namespace NUMINAMATH_GPT_find_m_plus_n_l398_39886

-- Define the sets and variables
def M : Set ℝ := {x | x^2 - 4 * x < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < 5}
def K (n : ℝ) : Set ℝ := {x | 3 < x ∧ x < n}

theorem find_m_plus_n (m n : ℝ) 
  (hM: M = {x | 0 < x ∧ x < 4})
  (hK_true: K n = M ∩ N m) :
  m + n = 7 := 
  sorry

end NUMINAMATH_GPT_find_m_plus_n_l398_39886


namespace NUMINAMATH_GPT_downstream_speed_l398_39834

theorem downstream_speed 
  (upstream_speed : ℕ) 
  (still_water_speed : ℕ) 
  (hm_upstream : upstream_speed = 27) 
  (hm_still_water : still_water_speed = 31) 
  : (still_water_speed + (still_water_speed - upstream_speed)) = 35 :=
by
  sorry

end NUMINAMATH_GPT_downstream_speed_l398_39834


namespace NUMINAMATH_GPT_find_range_of_m_l398_39860

noncomputable def quadratic_equation := 
  ∀ (m : ℝ), 
  ∃ x y : ℝ, 
  (m + 3) * x^2 - 4 * m * x + (2 * m - 1) = 0 ∧ 
  (m + 3) * y^2 - 4 * m * y + (2 * m - 1) = 0 ∧ 
  x * y < 0 ∧ 
  |x| > |y| ∧ 
  m ∈ Set.Ioo (-3:ℝ) (0:ℝ)

theorem find_range_of_m : quadratic_equation := 
by
  sorry

end NUMINAMATH_GPT_find_range_of_m_l398_39860


namespace NUMINAMATH_GPT_complex_expression_equals_neg3_l398_39899

noncomputable def nonreal_root_of_x4_eq_1 : Type :=
{ζ : ℂ // ζ^4 = 1 ∧ ζ.im ≠ 0}

theorem complex_expression_equals_neg3 (ζ : nonreal_root_of_x4_eq_1) :
  (1 - ζ.val + ζ.val^3)^4 + (1 + ζ.val^2 - ζ.val^3)^4 = -3 :=
sorry

end NUMINAMATH_GPT_complex_expression_equals_neg3_l398_39899


namespace NUMINAMATH_GPT_find_a_l398_39882

def F (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := a * b^3 + c

theorem find_a (a : ℚ) : F a 2 3 = F a 3 8 → a = -5 / 19 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l398_39882


namespace NUMINAMATH_GPT_part1_part2_l398_39865

theorem part1 (x y : ℕ) (h1 : 25 * x + 30 * y = 1500) (h2 : x = 2 * y - 4) : x = 36 ∧ y = 20 :=
by
  sorry

theorem part2 (x y : ℕ) (h1 : x + y = 60) (h2 : x ≥ 2 * y)
  (h_profit : ∃ p, p = 7 * x + 10 * y) : 
  ∃ x y profit, x = 40 ∧ y = 20 ∧ profit = 480 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l398_39865


namespace NUMINAMATH_GPT_trapezoid_problem_case1_trapezoid_problem_case2_trapezoid_problem_case3_l398_39843

variable {a b : ℝ}
variable {M N : ℝ}

/-- Trapezoid problem statements -/
theorem trapezoid_problem_case1 (h : a < 2 * b) : M - N = a - 2 * b := 
sorry

theorem trapezoid_problem_case2 (h : a = 2 * b) : M - N = 0 := 
sorry

theorem trapezoid_problem_case3 (h : a > 2 * b) : M - N = 2 * b - a := 
sorry

end NUMINAMATH_GPT_trapezoid_problem_case1_trapezoid_problem_case2_trapezoid_problem_case3_l398_39843


namespace NUMINAMATH_GPT_simplify_expression_at_zero_l398_39818

-- Define the expression f(x)
def f (x : ℚ) : ℚ := (2 * x + 4) / (x^2 - 6 * x + 9) / ((2 * x - 1) / (x - 3) - 1)

-- State that for the given value x = 0, the simplified expression equals -2/3
theorem simplify_expression_at_zero :
  f 0 = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_at_zero_l398_39818


namespace NUMINAMATH_GPT_angle_B_equiv_60_l398_39824

noncomputable def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A

theorem angle_B_equiv_60 
  (a b c A B C : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A) (h5 : A < π)
  (h6 : 0 < B) (h7 : B < π)
  (h8 : 0 < C) (h9 : C < π)
  (h_triangle : A + B + C = π)
  (h_arith : triangle_condition a b c A B C) : 
  B = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_B_equiv_60_l398_39824


namespace NUMINAMATH_GPT_range_of_a_iff_l398_39803

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, |x| + |x - 1| ≤ a → a ≥ 1

theorem range_of_a_iff (a : ℝ) :
  (∃ x : ℝ, |x| + |x - 1| ≤ a) ↔ (a ≥ 1) :=
by sorry

end NUMINAMATH_GPT_range_of_a_iff_l398_39803


namespace NUMINAMATH_GPT_no_solution_if_and_only_if_zero_l398_39877

theorem no_solution_if_and_only_if_zero (n : ℝ) :
  ¬(∃ (x y z : ℝ), 2 * n * x + y = 2 ∧ 3 * n * y + z = 3 ∧ x + 2 * n * z = 2) ↔ n = 0 := 
  by
  sorry

end NUMINAMATH_GPT_no_solution_if_and_only_if_zero_l398_39877


namespace NUMINAMATH_GPT_total_bill_l398_39829

-- Definitions from conditions
def num_people : ℕ := 3
def amount_per_person : ℕ := 45

-- Mathematical proof problem statement
theorem total_bill : num_people * amount_per_person = 135 := by
  sorry

end NUMINAMATH_GPT_total_bill_l398_39829


namespace NUMINAMATH_GPT_javier_average_hits_l398_39847

-- Define the total number of games Javier plays and the first set number of games
def total_games := 30
def first_set_games := 20

-- Define the hit averages for the first set of games and the desired season average
def average_hits_first_set := 2
def desired_season_average := 3

-- Define the total hits Javier needs to achieve the desired average by the end of the season
def total_hits_needed : ℕ := total_games * desired_season_average

-- Define the hits Javier made in the first set of games
def hits_made_first_set : ℕ := first_set_games * average_hits_first_set

-- Define the remaining games and the hits Javier needs to achieve in these games to meet his target
def remaining_games := total_games - first_set_games
def hits_needed_remaining_games : ℕ := total_hits_needed - hits_made_first_set

-- Define the average hits Javier needs in the remaining games to meet his target
def average_needed_remaining_games (remaining_games hits_needed_remaining_games : ℕ) : ℕ :=
  hits_needed_remaining_games / remaining_games

theorem javier_average_hits : 
  average_needed_remaining_games remaining_games hits_needed_remaining_games = 5 := 
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_javier_average_hits_l398_39847


namespace NUMINAMATH_GPT_sam_distance_walked_l398_39893

variable (d : ℝ := 40) -- initial distance between Fred and Sam
variable (v_f : ℝ := 4) -- Fred's constant speed in miles per hour
variable (v_s : ℝ := 4) -- Sam's constant speed in miles per hour

theorem sam_distance_walked :
  (d / (v_f + v_s)) * v_s = 20 :=
by
  sorry

end NUMINAMATH_GPT_sam_distance_walked_l398_39893


namespace NUMINAMATH_GPT_width_of_road_correct_l398_39884

-- Define the given conditions
def sum_of_circumferences (r R : ℝ) : Prop := 2 * Real.pi * r + 2 * Real.pi * R = 88
def radius_relation (r R : ℝ) : Prop := r = (1/3) * R
def width_of_road (R r : ℝ) := R - r

-- State the main theorem
theorem width_of_road_correct (R r : ℝ) (h1 : sum_of_circumferences r R) (h2 : radius_relation r R) :
    width_of_road R r = 22 / Real.pi := by
  sorry

end NUMINAMATH_GPT_width_of_road_correct_l398_39884


namespace NUMINAMATH_GPT_trajectory_of_center_l398_39880

-- Define a structure for Point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the given point A
def A : Point := { x := -2, y := 0 }

-- Define a property for the circle being tangent to a line
def tangent_to_line (center : Point) (line_x : ℝ) : Prop :=
  center.x + line_x = 0

-- The main theorem to be proved
theorem trajectory_of_center :
  ∀ (C : Point), tangent_to_line C 2 → (C.y)^2 = -8 * C.x :=
sorry

end NUMINAMATH_GPT_trajectory_of_center_l398_39880


namespace NUMINAMATH_GPT_expression_A_expression_B_expression_C_expression_D_l398_39853

theorem expression_A :
  (Real.sin (7 * Real.pi / 180) * Real.cos (23 * Real.pi / 180) + 
   Real.sin (83 * Real.pi / 180) * Real.cos (67 * Real.pi / 180)) = 1 / 2 :=
sorry

theorem expression_B :
  (2 * Real.cos (75 * Real.pi / 180) * Real.sin (75 * Real.pi / 180)) = 1 / 2 :=
sorry

theorem expression_C :
  (Real.sqrt 3 * Real.cos (10 * Real.pi / 180) - Real.sin (10 * Real.pi / 180)) / 
   Real.sin (50 * Real.pi / 180) ≠ 1 / 2 :=
sorry

theorem expression_D :
  (1 / ((1 + Real.tan (27 * Real.pi / 180)) * (1 + Real.tan (18 * Real.pi / 180)))) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_expression_A_expression_B_expression_C_expression_D_l398_39853


namespace NUMINAMATH_GPT_g_at_12_l398_39820

def g (n : ℤ) : ℤ := n^2 + 2*n + 23

theorem g_at_12 : g 12 = 191 := by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_g_at_12_l398_39820


namespace NUMINAMATH_GPT_cylinder_height_l398_39867

theorem cylinder_height (r h : ℝ) (SA : ℝ) 
  (hSA : SA = 2 * Real.pi * r ^ 2 + 2 * Real.pi * r * h) 
  (hr : r = 3) (hSA_val : SA = 36 * Real.pi) : 
  h = 3 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_height_l398_39867


namespace NUMINAMATH_GPT_highest_place_value_734_48_l398_39839

theorem highest_place_value_734_48 : 
  (∃ k, 10^4 = k ∧ k * 10^4 ≤ 734 * 48 ∧ 734 * 48 < (k + 1) * 10^4) := 
sorry

end NUMINAMATH_GPT_highest_place_value_734_48_l398_39839


namespace NUMINAMATH_GPT_g_neg_two_is_zero_l398_39823

theorem g_neg_two_is_zero {f g : ℤ → ℤ} 
  (h_odd: ∀ x: ℤ, f (-x) + (-x) = -(f x + x)) 
  (hf_two: f 2 = 1) 
  (hg_def: ∀ x: ℤ, g x = f x + 1):
  g (-2) = 0 := 
sorry

end NUMINAMATH_GPT_g_neg_two_is_zero_l398_39823


namespace NUMINAMATH_GPT_correct_cd_value_l398_39857

noncomputable def repeating_decimal (c d : ℕ) : ℝ :=
  1 + c / 10.0 + d / 100.0 + (c * 10 + d) / 990.0

theorem correct_cd_value (c d : ℕ) (h : (c = 9) ∧ (d = 9)) : 90 * (repeating_decimal 9 9 - (1 + 9 / 10.0 + 9 / 100.0)) = 0.9 :=
by
  sorry

end NUMINAMATH_GPT_correct_cd_value_l398_39857


namespace NUMINAMATH_GPT_flour_needed_l398_39844

-- Define the given conditions
def F_total : ℕ := 9
def F_added : ℕ := 3

-- State the main theorem to be proven
theorem flour_needed : (F_total - F_added) = 6 := by
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_flour_needed_l398_39844


namespace NUMINAMATH_GPT_num_ways_distribute_plants_correct_l398_39817

def num_ways_to_distribute_plants : Nat :=
  let basil := 2
  let aloe := 1
  let cactus := 1
  let white_lamps := 2
  let red_lamp := 1
  let blue_lamp := 1
  let plants := basil + aloe + cactus
  let lamps := white_lamps + red_lamp + blue_lamp
  4
  
theorem num_ways_distribute_plants_correct :
  num_ways_to_distribute_plants = 4 :=
by
  sorry -- Proof of the correctness of the distribution

end NUMINAMATH_GPT_num_ways_distribute_plants_correct_l398_39817


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_value_l398_39812

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a n = a 1 + (n - 1) * d)
  (h_a2 : a 2 = 1)
  (h_a8 : a 8 = 2 * a 6 + a 4) : 
  a 5 = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_value_l398_39812


namespace NUMINAMATH_GPT_truth_probability_l398_39806

theorem truth_probability (P_A : ℝ) (P_A_and_B : ℝ) (P_B : ℝ) 
  (hA : P_A = 0.70) (hA_and_B : P_A_and_B = 0.42) : 
  P_A * P_B = P_A_and_B → P_B = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_truth_probability_l398_39806


namespace NUMINAMATH_GPT_algebraic_expression_value_l398_39838

theorem algebraic_expression_value (m : ℝ) (h : (2018 + m) * (2020 + m) = 2) : (2018 + m)^2 + (2020 + m)^2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l398_39838


namespace NUMINAMATH_GPT_zach_cookies_left_l398_39815

/- Defining the initial conditions on cookies baked each day -/
def cookies_monday : ℕ := 32
def cookies_tuesday : ℕ := cookies_monday / 2
def cookies_wednesday : ℕ := 3 * cookies_tuesday - 4 - 3
def cookies_thursday : ℕ := 2 * cookies_monday - 10 + 5
def cookies_friday : ℕ := cookies_wednesday - 6 - 4
def cookies_saturday : ℕ := cookies_monday + cookies_friday - 10

/- Aggregating total cookies baked throughout the week -/
def total_baked : ℕ := cookies_monday + cookies_tuesday + cookies_wednesday +
                      cookies_thursday + cookies_friday + cookies_saturday

/- Defining cookies lost each day -/
def daily_parents_eat : ℕ := 2 * 6
def neighbor_friday_eat : ℕ := 8
def friends_thursday_eat : ℕ := 3 * 2

def total_lost : ℕ := 4 + 3 + 10 + 6 + 4 + 10 + daily_parents_eat + neighbor_friday_eat + friends_thursday_eat

/- Calculating cookies left at end of six days -/
def cookies_left : ℕ := total_baked - total_lost

/- Proof objective -/
theorem zach_cookies_left : cookies_left = 200 := by
  sorry

end NUMINAMATH_GPT_zach_cookies_left_l398_39815
