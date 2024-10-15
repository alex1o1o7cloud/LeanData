import Mathlib

namespace NUMINAMATH_GPT_decreasing_f_range_l1188_118856

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

theorem decreasing_f_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) → (0 < a ∧ a ≤ 1/4) :=
by
  sorry

end NUMINAMATH_GPT_decreasing_f_range_l1188_118856


namespace NUMINAMATH_GPT_proportion_equal_l1188_118834

theorem proportion_equal (x : ℝ) : (0.25 / x = 2 / 6) → x = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_proportion_equal_l1188_118834


namespace NUMINAMATH_GPT_pieces_of_meat_per_slice_eq_22_l1188_118893

def number_of_pepperoni : Nat := 30
def number_of_ham : Nat := 2 * number_of_pepperoni
def number_of_sausage : Nat := number_of_pepperoni + 12
def total_meat : Nat := number_of_pepperoni + number_of_ham + number_of_sausage
def number_of_slices : Nat := 6

theorem pieces_of_meat_per_slice_eq_22 : total_meat / number_of_slices = 22 :=
by
  sorry

end NUMINAMATH_GPT_pieces_of_meat_per_slice_eq_22_l1188_118893


namespace NUMINAMATH_GPT_relation_of_M_and_N_l1188_118810

-- Define the functions for M and N
def M (x : ℝ) : ℝ := (x - 3) * (x - 4)
def N (x : ℝ) : ℝ := (x - 1) * (x - 6)

-- Formulate the theorem to prove M < N for all x
theorem relation_of_M_and_N (x : ℝ) : M x < N x := sorry

end NUMINAMATH_GPT_relation_of_M_and_N_l1188_118810


namespace NUMINAMATH_GPT_burger_cost_cents_l1188_118803

theorem burger_cost_cents 
  (b s : ℕ)
  (h1 : 4 * b + 3 * s = 550) 
  (h2 : 3 * b + 2 * s = 400) 
  (h3 : 2 * b + s = 250) : 
  b = 100 :=
by
  sorry

end NUMINAMATH_GPT_burger_cost_cents_l1188_118803


namespace NUMINAMATH_GPT_pen_distribution_l1188_118864

theorem pen_distribution (x : ℕ) :
  8 * x + 3 = 12 * (x - 2) - 1 :=
sorry

end NUMINAMATH_GPT_pen_distribution_l1188_118864


namespace NUMINAMATH_GPT_find_percentage_l1188_118870

theorem find_percentage (P N : ℕ) (h1 : N = 100) (h2 : (P : ℝ) / 100 * N = 50 / 100 * 40 + 10) :
  P = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l1188_118870


namespace NUMINAMATH_GPT_x_intercept_l1188_118867

theorem x_intercept (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -8) (hy2 : y2 = -6) : 
  ∃ x : ℝ, (y = 0) ∧ (∃ m : ℝ, m = (y2 - y1) / (x2 - x1) ∧ y1 - y = m * (x1 - x)) ∧ x = 4 :=
sorry

end NUMINAMATH_GPT_x_intercept_l1188_118867


namespace NUMINAMATH_GPT_weight_of_empty_jar_l1188_118882

variable (W : ℝ) -- Weight of the empty jar
variable (w : ℝ) -- Weight of water for one-fifth of the jar

-- Conditions
variable (h1 : W + w = 560)
variable (h2 : W + 4 * w = 740)

-- Theorem statement
theorem weight_of_empty_jar (W w : ℝ) (h1 : W + w = 560) (h2 : W + 4 * w = 740) : W = 500 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_empty_jar_l1188_118882


namespace NUMINAMATH_GPT_Alice_spent_19_percent_l1188_118861

variable (A B A': ℝ)
def Bob_less_money_than_Alice (A B : ℝ) : Prop :=
  B = 0.9 * A

def Alice_less_money_than_Bob (B A' : ℝ) : Prop :=
  A' = 0.9 * B

theorem Alice_spent_19_percent (A B A' : ℝ) 
  (h1 : Bob_less_money_than_Alice A B)
  (h2 : Alice_less_money_than_Bob B A') :
  ((A - A') / A) * 100 = 19 :=
by
  sorry

end NUMINAMATH_GPT_Alice_spent_19_percent_l1188_118861


namespace NUMINAMATH_GPT_x_fourth_minus_inv_fourth_l1188_118837

theorem x_fourth_minus_inv_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/(x^4) = 727 :=
by
  sorry

end NUMINAMATH_GPT_x_fourth_minus_inv_fourth_l1188_118837


namespace NUMINAMATH_GPT_find_q_in_geometric_sequence_l1188_118806

theorem find_q_in_geometric_sequence
  {q : ℝ} (q_pos : q > 0) 
  (a1_def : ∀(a : ℕ → ℝ), a 1 = 1 / q^2) 
  (S5_eq_S2_plus_2 : ∀(S : ℕ → ℝ), S 5 = S 2 + 2) :
  q = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_q_in_geometric_sequence_l1188_118806


namespace NUMINAMATH_GPT_range_of_a_l1188_118865

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, (x^2 + (a^2 + 1) * x + a - 2 = 0 ∧ y^2 + (a^2 + 1) * y + a - 2 = 0)
    ∧ x > 1 ∧ y < -1) ↔ (-1 < a ∧ a < 0) := sorry

end NUMINAMATH_GPT_range_of_a_l1188_118865


namespace NUMINAMATH_GPT_factor_is_2_l1188_118848

variable (x : ℕ) (f : ℕ)

theorem factor_is_2 (h₁ : x = 36)
                    (h₂ : ((f * (x + 10)) / 2) - 2 = 44) : f = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_factor_is_2_l1188_118848


namespace NUMINAMATH_GPT_find_n_with_divisors_sum_l1188_118809

theorem find_n_with_divisors_sum (n : ℕ) (d1 d2 d3 d4 : ℕ)
  (h1 : d1 = 1) (h2 : d2 = 2) (h3 : d3 = 5) (h4 : d4 = 10) 
  (hd : n = 130) : d1^2 + d2^2 + d3^2 + d4^2 = n :=
sorry

end NUMINAMATH_GPT_find_n_with_divisors_sum_l1188_118809


namespace NUMINAMATH_GPT_bike_covered_distance_l1188_118873

theorem bike_covered_distance
  (time : ℕ) 
  (truck_distance : ℕ) 
  (speed_difference : ℕ) 
  (bike_speed truck_speed : ℕ)
  (h_time : time = 8)
  (h_truck_distance : truck_distance = 112)
  (h_speed_difference : speed_difference = 3)
  (h_truck_speed : truck_speed = truck_distance / time)
  (h_speed_relation : truck_speed = bike_speed + speed_difference) :
  bike_speed * time = 88 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_bike_covered_distance_l1188_118873


namespace NUMINAMATH_GPT_original_price_of_computer_l1188_118899

theorem original_price_of_computer (P : ℝ) (h1 : 1.30 * P = 364) (h2 : 2 * P = 560) : P = 280 :=
by 
  -- The proof is skipped as per instruction
  sorry

end NUMINAMATH_GPT_original_price_of_computer_l1188_118899


namespace NUMINAMATH_GPT_reciprocal_opposite_abs_val_l1188_118875

theorem reciprocal_opposite_abs_val (a : ℚ) (h : a = -1 - 2/7) :
    (1 / a = -7/9) ∧ (-a = 1 + 2/7) ∧ (|a| = 1 + 2/7) := 
sorry

end NUMINAMATH_GPT_reciprocal_opposite_abs_val_l1188_118875


namespace NUMINAMATH_GPT_polynomial_inequality_l1188_118829

theorem polynomial_inequality (a b c : ℝ)
  (h1 : ∃ r1 r2 r3 : ℝ, (r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3) ∧ 
    (∀ t : ℝ, (t - r1) * (t - r2) * (t - r3) = t^3 + a*t^2 + b*t + c))
  (h2 : ¬ ∃ x : ℝ, (x^2 + x + 2013)^3 + a*(x^2 + x + 2013)^2 + b*(x^2 + x + 2013) + c = 0) :
  t^3 + a*2013^2 + b*2013 + c > 1 / 64 :=
sorry

end NUMINAMATH_GPT_polynomial_inequality_l1188_118829


namespace NUMINAMATH_GPT_max_integer_value_fraction_l1188_118845

theorem max_integer_value_fraction (x : ℝ) : 
  (∃ t : ℤ, t = 2 ∧ (∀ y : ℝ, y = (4*x^2 + 8*x + 21) / (4*x^2 + 8*x + 9) → y <= t)) :=
sorry

end NUMINAMATH_GPT_max_integer_value_fraction_l1188_118845


namespace NUMINAMATH_GPT_sum_of_integers_is_eleven_l1188_118824

theorem sum_of_integers_is_eleven (p q r s : ℤ) 
  (h1 : p - q + r = 7) 
  (h2 : q - r + s = 8) 
  (h3 : r - s + p = 4) 
  (h4 : s - p + q = 3) : 
  p + q + r + s = 11 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_is_eleven_l1188_118824


namespace NUMINAMATH_GPT_horner_method_value_v2_at_minus_one_l1188_118832

noncomputable def f (x : ℝ) : ℝ :=
  x^6 - 5*x^5 + 6*x^4 - 3*x^3 + 1.8*x^2 + 0.35*x + 2

theorem horner_method_value_v2_at_minus_one :
  let a : ℝ := -1
  let v_0 := 1
  let v_1 := v_0 * a - 5
  let v_2 := v_1 * a + 6
  v_2 = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_horner_method_value_v2_at_minus_one_l1188_118832


namespace NUMINAMATH_GPT_jovana_total_shells_l1188_118874

def initial_amount : ℕ := 5
def added_amount : ℕ := 23
def total_amount : ℕ := 28

theorem jovana_total_shells : initial_amount + added_amount = total_amount := by
  sorry

end NUMINAMATH_GPT_jovana_total_shells_l1188_118874


namespace NUMINAMATH_GPT_sin_cos_product_l1188_118894

theorem sin_cos_product (x : ℝ) (h₁ : 0 < x) (h₂ : x < π / 2) (h₃ : Real.sin x = 3 * Real.cos x) : 
  Real.sin x * Real.cos x = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_product_l1188_118894


namespace NUMINAMATH_GPT_num_from_1_to_200_not_squares_or_cubes_l1188_118876

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end NUMINAMATH_GPT_num_from_1_to_200_not_squares_or_cubes_l1188_118876


namespace NUMINAMATH_GPT_sahil_selling_price_l1188_118892

def initial_cost : ℝ := 14000
def repair_cost : ℝ := 5000
def transportation_charges : ℝ := 1000
def profit_percent : ℝ := 50

noncomputable def total_cost : ℝ := initial_cost + repair_cost + transportation_charges
noncomputable def profit : ℝ := profit_percent / 100 * total_cost
noncomputable def selling_price : ℝ := total_cost + profit

theorem sahil_selling_price :
  selling_price = 30000 := by
  sorry

end NUMINAMATH_GPT_sahil_selling_price_l1188_118892


namespace NUMINAMATH_GPT_taoqi_has_higher_utilization_rate_l1188_118885

noncomputable def area_square (side_length : ℝ) : ℝ :=
  side_length * side_length

noncomputable def area_circle (radius : ℝ) : ℝ :=
  Real.pi * radius * radius

noncomputable def utilization_rate (cut_area : ℝ) (original_area : ℝ) : ℝ :=
  cut_area / original_area

noncomputable def tao_qi_utilization_rate : ℝ :=
  let side_length := 9
  let square_area := area_square side_length
  let radius := side_length / 2
  let circle_area := area_circle radius
  utilization_rate circle_area square_area

noncomputable def xiao_xiao_utilization_rate : ℝ :=
  let diameter := 9
  let radius := diameter / 2
  let large_circle_area := area_circle radius
  let small_circle_radius := diameter / 6
  let small_circle_area := area_circle small_circle_radius
  let total_small_circles_area := 7 * small_circle_area
  utilization_rate total_small_circles_area large_circle_area

-- Theorem statement reflecting the proof problem:
theorem taoqi_has_higher_utilization_rate :
  tao_qi_utilization_rate > xiao_xiao_utilization_rate := by sorry

end NUMINAMATH_GPT_taoqi_has_higher_utilization_rate_l1188_118885


namespace NUMINAMATH_GPT_find_c_l1188_118849

theorem find_c (c : ℝ) 
  (h : (⟨9, c⟩ : ℝ × ℝ) = (11/13 : ℝ) • ⟨-3, 2⟩) : 
  c = 19 :=
sorry

end NUMINAMATH_GPT_find_c_l1188_118849


namespace NUMINAMATH_GPT_find_z_l1188_118863

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

end NUMINAMATH_GPT_find_z_l1188_118863


namespace NUMINAMATH_GPT_find_chocolate_cakes_l1188_118847

variable (C : ℕ)
variable (h1 : 12 * C + 6 * 22 = 168)

theorem find_chocolate_cakes : C = 3 :=
by
  -- this is the proof placeholder
  sorry

end NUMINAMATH_GPT_find_chocolate_cakes_l1188_118847


namespace NUMINAMATH_GPT_sum_of_geometric_sequence_l1188_118828

-- Consider a geometric sequence {a_n} with the first term a_1 = 1 and a common ratio of 1/3.
-- Let S_n denote the sum of the first n terms.
-- We need to prove that S_n = (3 - a_n) / 2, given the above conditions.
noncomputable def geometric_sequence_sum (n : ℕ) : ℝ :=
  let a_1 := 1
  let r := (1 : ℝ) / 3
  let a_n := r ^ (n - 1)
  (3 - a_n) / 2

theorem sum_of_geometric_sequence (n : ℕ) : geometric_sequence_sum n = 
  let a_1 := 1
  let r := (1 : ℝ) / 3
  let a_n := r ^ (n - 1)
  (3 - a_n) / 2 := sorry

end NUMINAMATH_GPT_sum_of_geometric_sequence_l1188_118828


namespace NUMINAMATH_GPT_total_caffeine_is_correct_l1188_118850

def first_drink_caffeine := 250 -- milligrams
def first_drink_size := 12 -- ounces

def second_drink_caffeine_per_ounce := (first_drink_caffeine / first_drink_size) * 3
def second_drink_size := 8 -- ounces
def second_drink_caffeine := second_drink_caffeine_per_ounce * second_drink_size

def third_drink_concentration := 18 -- milligrams per milliliter
def third_drink_size := 150 -- milliliters
def third_drink_caffeine := third_drink_concentration * third_drink_size

def caffeine_pill_caffeine := first_drink_caffeine + second_drink_caffeine + third_drink_caffeine

def total_caffeine_consumed := first_drink_caffeine + second_drink_caffeine + third_drink_caffeine + caffeine_pill_caffeine

theorem total_caffeine_is_correct : total_caffeine_consumed = 6900 :=
by
  sorry

end NUMINAMATH_GPT_total_caffeine_is_correct_l1188_118850


namespace NUMINAMATH_GPT_calculate_value_of_expression_l1188_118831

theorem calculate_value_of_expression :
  3.5 * 7.2 * (6.3 - 1.4) = 122.5 :=
  by
  sorry

end NUMINAMATH_GPT_calculate_value_of_expression_l1188_118831


namespace NUMINAMATH_GPT_depth_of_second_hole_l1188_118853

theorem depth_of_second_hole :
  let workers1 := 45
  let hours1 := 8
  let depth1 := 30
  let man_hours1 := workers1 * hours1 -- 360 man-hours
  let workers2 := 45 + 35 -- 80 workers
  let hours2 := 6
  let man_hours2 := workers2 * hours2 -- 480 man-hours
  let depth2 := (man_hours2 * depth1) / man_hours1 -- value to solve for
  depth2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_depth_of_second_hole_l1188_118853


namespace NUMINAMATH_GPT_henry_correct_answers_l1188_118852

theorem henry_correct_answers (c w : ℕ) (h1 : c + w = 15) (h2 : 6 * c - 3 * w = 45) : c = 10 :=
by
  sorry

end NUMINAMATH_GPT_henry_correct_answers_l1188_118852


namespace NUMINAMATH_GPT_solve_a_l1188_118880

variable (a : ℝ)

theorem solve_a (h : ∃ b : ℝ, (9 * x^2 + 12 * x + a) = (3 * x + b) ^ 2) : a = 4 :=
by
   sorry

end NUMINAMATH_GPT_solve_a_l1188_118880


namespace NUMINAMATH_GPT_initial_percentage_of_water_is_20_l1188_118891

theorem initial_percentage_of_water_is_20 : 
  ∀ (P : ℝ) (total_initial_volume added_water total_final_volume final_percentage initial_water_percentage : ℝ), 
    total_initial_volume = 125 ∧ 
    added_water = 8.333333333333334 ∧ 
    total_final_volume = total_initial_volume + added_water ∧ 
    final_percentage = 25 ∧ 
    initial_water_percentage = (initial_water_percentage / total_initial_volume) * 100 ∧ 
    (final_percentage / 100) * total_final_volume = added_water + (initial_water_percentage / 100) * total_initial_volume → 
    initial_water_percentage = 20 := 
by 
  sorry

end NUMINAMATH_GPT_initial_percentage_of_water_is_20_l1188_118891


namespace NUMINAMATH_GPT_rationalize_denominator_l1188_118860

theorem rationalize_denominator (cbrt : ℝ → ℝ) (h₁ : cbrt 81 = 3 * cbrt 3) :
  1 / (cbrt 3 + cbrt 81) = cbrt 9 / 12 :=
sorry

end NUMINAMATH_GPT_rationalize_denominator_l1188_118860


namespace NUMINAMATH_GPT_copper_zinc_mixture_mass_bounds_l1188_118872

theorem copper_zinc_mixture_mass_bounds :
  ∀ (x y : ℝ) (D1 D2 : ℝ),
    (400 = x + y) →
    (50 = x / D1 + y / D2) →
    (8.8 ≤ D1 ∧ D1 ≤ 9) →
    (7.1 ≤ D2 ∧ D2 ≤ 7.2) →
    (200 ≤ x ∧ x ≤ 233) ∧ (167 ≤ y ∧ y ≤ 200) :=
sorry

end NUMINAMATH_GPT_copper_zinc_mixture_mass_bounds_l1188_118872


namespace NUMINAMATH_GPT_cone_from_sector_radius_l1188_118830

theorem cone_from_sector_radius (r : ℝ) (slant_height : ℝ) : 
  (r = 9) ∧ (slant_height = 12) ↔ 
  (∃ (sector_angle : ℝ) (sector_radius : ℝ), 
    sector_angle = 270 ∧ sector_radius = 12 ∧ 
    slant_height = sector_radius ∧ 
    (2 * π * r = sector_angle / 360 * 2 * π * sector_radius)) :=
by
  sorry

end NUMINAMATH_GPT_cone_from_sector_radius_l1188_118830


namespace NUMINAMATH_GPT_checkered_rectangles_unique_gray_cells_l1188_118854

noncomputable def num_checkered_rectangles (num_gray_cells : ℕ) (num_blue_cells : ℕ) (rects_per_blue_cell : ℕ)
    (num_red_cells : ℕ) (rects_per_red_cell : ℕ) : ℕ :=
    (num_blue_cells * rects_per_blue_cell) + (num_red_cells * rects_per_red_cell)

theorem checkered_rectangles_unique_gray_cells : num_checkered_rectangles 40 36 4 4 8 = 176 := 
sorry

end NUMINAMATH_GPT_checkered_rectangles_unique_gray_cells_l1188_118854


namespace NUMINAMATH_GPT_min_moves_to_equalize_boxes_l1188_118811

def initialCoins : List ℕ := [5, 8, 11, 17, 20, 15, 10]

def targetCoins (boxes : List ℕ) : ℕ := boxes.sum / boxes.length

def movesRequiredToBalance : List ℕ → ℕ
| [5, 8, 11, 17, 20, 15, 10] => 22
| _ => sorry

theorem min_moves_to_equalize_boxes :
  movesRequiredToBalance initialCoins = 22 :=
by
  sorry

end NUMINAMATH_GPT_min_moves_to_equalize_boxes_l1188_118811


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1188_118807

variable {S : ℕ → ℕ}

def isArithmeticSum (S : ℕ → ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ n, S n = n * (2 * a + (n - 1) * d ) / 2

theorem sum_of_arithmetic_sequence :
  isArithmeticSum S →
  S 8 - S 4 = 12 →
  S 12 = 36 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1188_118807


namespace NUMINAMATH_GPT_at_least_one_real_root_l1188_118896

theorem at_least_one_real_root (a : ℝ) :
  (4*a)^2 - 4*(-4*a + 3) ≥ 0 ∨
  ((a - 1)^2 - 4*a^2) ≥ 0 ∨
  (2*a)^2 - 4*(-2*a) ≥ 0 := sorry

end NUMINAMATH_GPT_at_least_one_real_root_l1188_118896


namespace NUMINAMATH_GPT_find_students_with_equal_homework_hours_l1188_118844

theorem find_students_with_equal_homework_hours :
  let Dan := 6
  let Joe := 3
  let Bob := 5
  let Susie := 4
  let Grace := 1
  (Joe + Grace = Dan ∨ Joe + Bob = Dan ∨ Bob + Grace = Dan ∨ Dan + Bob = Dan ∨ Susie + Grace = Dan) → 
  (Bob + Grace = Dan) := 
by 
  intros
  sorry

end NUMINAMATH_GPT_find_students_with_equal_homework_hours_l1188_118844


namespace NUMINAMATH_GPT_inverse_five_eq_two_l1188_118895

-- Define the function f(x) = x^2 + 1 for x >= 0
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the condition x >= 0
def nonneg (x : ℝ) : Prop := x ≥ 0

-- State the problem: proving that the inverse function f⁻¹(5) = 2
theorem inverse_five_eq_two : ∃ x : ℝ, nonneg x ∧ f x = 5 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_inverse_five_eq_two_l1188_118895


namespace NUMINAMATH_GPT_probability_no_defective_pencils_l1188_118841

theorem probability_no_defective_pencils :
  let total_pencils := 9
  let defective_pencils := 2
  let total_ways_choose_3 := Nat.choose total_pencils 3
  let non_defective_pencils := total_pencils - defective_pencils
  let ways_choose_3_non_defective := Nat.choose non_defective_pencils 3
  (ways_choose_3_non_defective : ℚ) / total_ways_choose_3 = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_no_defective_pencils_l1188_118841


namespace NUMINAMATH_GPT_problem1_problem2_l1188_118862

noncomputable def tan_inv_3_value : ℝ := -4 / 5

theorem problem1 (α : ℝ) (h : Real.tan α = 3) :
  Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α = tan_inv_3_value := 
sorry

noncomputable def f (θ : ℝ) : ℝ := 
  (2 * Real.cos θ ^ 3 + Real.sin (2 * Real.pi - θ) ^ 2 + 
   Real.sin (Real.pi / 2 + θ) - 3) / 
  (2 + 2 * Real.cos (Real.pi + θ) ^ 2 + Real.cos (-θ))

theorem problem2 :
  f (Real.pi / 3) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1188_118862


namespace NUMINAMATH_GPT_cookie_radius_l1188_118868

theorem cookie_radius (x y : ℝ) (h : x^2 + y^2 + 2 * x - 4 * y = 4) : 
  ∃ r : ℝ, (x + 1)^2 + (y - 2)^2 = r^2 ∧ r = 3 := by
  sorry

end NUMINAMATH_GPT_cookie_radius_l1188_118868


namespace NUMINAMATH_GPT_time_for_type_Q_machine_l1188_118814

theorem time_for_type_Q_machine (Q : ℝ) (h1 : Q > 0)
  (h2 : 2 * (1 / Q) + 3 * (1 / 7) = 5 / 6) :
  Q = 84 / 17 :=
sorry

end NUMINAMATH_GPT_time_for_type_Q_machine_l1188_118814


namespace NUMINAMATH_GPT_carlos_goals_product_l1188_118833

theorem carlos_goals_product :
  ∃ (g11 g12 : ℕ), g11 < 8 ∧ g12 < 8 ∧ 
  (33 + g11) % 11 = 0 ∧ 
  (33 + g11 + g12) % 12 = 0 ∧ 
  g11 * g12 = 49 := 
by
  sorry

end NUMINAMATH_GPT_carlos_goals_product_l1188_118833


namespace NUMINAMATH_GPT_point_not_on_graph_l1188_118802

def on_graph (x y : ℚ) : Prop := y = x / (x + 2)

/-- Let's state the main theorem -/
theorem point_not_on_graph : ¬ on_graph 2 (2 / 3) := by
  sorry

end NUMINAMATH_GPT_point_not_on_graph_l1188_118802


namespace NUMINAMATH_GPT_parabola_properties_l1188_118812

theorem parabola_properties :
  let a := -2
  let b := 4
  let c := 8
  ∃ h k : ℝ, 
    (∀ x : ℝ, y = a * x^2 + b * x + c) ∧ 
    (h = 1) ∧ 
    (k = 10) ∧ 
    (a < 0) ∧ 
    (axisOfSymmetry = h) ∧ 
    (vertex = (h, k)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_properties_l1188_118812


namespace NUMINAMATH_GPT_Nell_initial_cards_l1188_118816

theorem Nell_initial_cards (n : ℕ) (h1 : n - 136 = 106) : n = 242 := 
by
  sorry

end NUMINAMATH_GPT_Nell_initial_cards_l1188_118816


namespace NUMINAMATH_GPT_prob_none_three_win_prob_at_least_two_not_win_l1188_118869

-- Definitions for probabilities
def prob_win : ℚ := 1 / 6
def prob_not_win : ℚ := 1 - prob_win

-- Problem 1: Prove probability that none of the three students win
theorem prob_none_three_win : (prob_not_win ^ 3) = 125 / 216 := by
  sorry

-- Problem 2: Prove probability that at least two of the three students do not win
theorem prob_at_least_two_not_win : 1 - (3 * (prob_win ^ 2) * prob_not_win + prob_win ^ 3) = 25 / 27 := by
  sorry

end NUMINAMATH_GPT_prob_none_three_win_prob_at_least_two_not_win_l1188_118869


namespace NUMINAMATH_GPT_value_of_ac_over_bd_l1188_118881

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end NUMINAMATH_GPT_value_of_ac_over_bd_l1188_118881


namespace NUMINAMATH_GPT_largest_circle_area_l1188_118820

theorem largest_circle_area (PQ QR PR : ℝ)
  (h_right_triangle: PR^2 = PQ^2 + QR^2)
  (h_circle_areas_sum: π * (PQ/2)^2 + π * (QR/2)^2 + π * (PR/2)^2 = 338 * π) :
  π * (PR/2)^2 = 169 * π :=
by
  sorry

end NUMINAMATH_GPT_largest_circle_area_l1188_118820


namespace NUMINAMATH_GPT_distance_between_circle_centers_l1188_118835

theorem distance_between_circle_centers
  (R r d : ℝ)
  (h1 : R = 7)
  (h2 : r = 4)
  (h3 : d = 5 + 1)
  (h_total_diameter : 5 + 8 + 1 = 14)
  (h_radius_R : R = 14 / 2)
  (h_radius_r : r = 8 / 2) : d = 6 := 
by sorry

end NUMINAMATH_GPT_distance_between_circle_centers_l1188_118835


namespace NUMINAMATH_GPT_tangent_line_touching_circle_l1188_118801

theorem tangent_line_touching_circle (a : ℝ) : 
  (∃ (x y : ℝ), 5 * x + 12 * y + a = 0 ∧ (x - 1)^2 + y^2 = 1) → 
  (a = 8 ∨ a = -18) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_touching_circle_l1188_118801


namespace NUMINAMATH_GPT_heartsuit_4_6_l1188_118855

-- Define the operation \heartsuit
def heartsuit (x y : ℤ) : ℤ := 5 * x + 3 * y

-- Prove that 4 \heartsuit 6 = 38 under the given operation definition
theorem heartsuit_4_6 : heartsuit 4 6 = 38 := by
  -- Using the definition of \heartsuit
  -- Calculation is straightforward and skipped by sorry
  sorry

end NUMINAMATH_GPT_heartsuit_4_6_l1188_118855


namespace NUMINAMATH_GPT_smallest_integer_ends_in_3_and_divisible_by_5_l1188_118805

theorem smallest_integer_ends_in_3_and_divisible_by_5 : ∃ (n : ℕ), n > 0 ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ n = 53 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_ends_in_3_and_divisible_by_5_l1188_118805


namespace NUMINAMATH_GPT_max_arithmetic_sum_l1188_118888

def a1 : ℤ := 113
def d : ℤ := -4

def S (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem max_arithmetic_sum : S 29 = 1653 :=
by
  sorry

end NUMINAMATH_GPT_max_arithmetic_sum_l1188_118888


namespace NUMINAMATH_GPT_calculate_oplus_l1188_118808

def op (X Y : ℕ) : ℕ :=
  (X + Y) / 2

theorem calculate_oplus : op (op 6 10) 14 = 11 := by
  sorry

end NUMINAMATH_GPT_calculate_oplus_l1188_118808


namespace NUMINAMATH_GPT_injective_functions_count_l1188_118842

theorem injective_functions_count (m n : ℕ) (h_mn : m ≥ n) (h_n2 : n ≥ 2) :
  ∃ k, k = Nat.choose m n * (2^n - n - 1) :=
sorry

end NUMINAMATH_GPT_injective_functions_count_l1188_118842


namespace NUMINAMATH_GPT_angle_same_terminal_side_315_l1188_118877

theorem angle_same_terminal_side_315 (k : ℤ) : ∃ α, α = k * 360 + 315 ∧ α = -45 :=
by
  use -45
  sorry

end NUMINAMATH_GPT_angle_same_terminal_side_315_l1188_118877


namespace NUMINAMATH_GPT_average_cost_of_fruit_l1188_118890

variable (apples bananas oranges total_cost total_pieces avg_cost : ℕ)

theorem average_cost_of_fruit (h1 : apples = 12)
                              (h2 : bananas = 4)
                              (h3 : oranges = 4)
                              (h4 : total_cost = apples * 2 + bananas * 1 + oranges * 3)
                              (h5 : total_pieces = apples + bananas + oranges)
                              (h6 : avg_cost = total_cost / total_pieces) :
                              avg_cost = 2 :=
by sorry

end NUMINAMATH_GPT_average_cost_of_fruit_l1188_118890


namespace NUMINAMATH_GPT_largest_valid_n_l1188_118800

def is_valid_n (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 10 * a + b ∧ n = a * (a + b)

theorem largest_valid_n : ∀ n : ℕ, is_valid_n n → n ≤ 48 := by sorry

example : is_valid_n 48 := by sorry

end NUMINAMATH_GPT_largest_valid_n_l1188_118800


namespace NUMINAMATH_GPT_part1_part2_l1188_118839

def is_regressive_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, x n + x (n + 2) - x (n + 1) = x m

theorem part1 (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = 3 ^ n) :
  ¬ is_regressive_sequence a := by
  sorry

theorem part2 (b : ℕ → ℝ) (h_reg : is_regressive_sequence b) (h_inc : ∀ n : ℕ, b n < b (n + 1)) :
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1188_118839


namespace NUMINAMATH_GPT_number_of_grade11_students_l1188_118859

-- Define the total number of students in the high school.
def total_students : ℕ := 900

-- Define the total number of students selected in the sample.
def sample_students : ℕ := 45

-- Define the number of Grade 10 students in the sample.
def grade10_students_sample : ℕ := 20

-- Define the number of Grade 12 students in the sample.
def grade12_students_sample : ℕ := 10

-- Prove the number of Grade 11 students in the school is 300.
theorem number_of_grade11_students :
  (sample_students - grade10_students_sample - grade12_students_sample) * (total_students / sample_students) = 300 :=
by
  sorry

end NUMINAMATH_GPT_number_of_grade11_students_l1188_118859


namespace NUMINAMATH_GPT_find_number_l1188_118821

theorem find_number (x : ℕ) (h1 : x > 7) (h2 : x ≠ 8) : x = 9 := by
  sorry

end NUMINAMATH_GPT_find_number_l1188_118821


namespace NUMINAMATH_GPT_triangle_height_l1188_118897

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) 
  (h_area : area = 615) (h_base : base = 123) 
  (area_formula : area = (base * height) / 2) : height = 10 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_height_l1188_118897


namespace NUMINAMATH_GPT_mailman_distribution_l1188_118879

theorem mailman_distribution 
    (total_mail_per_block : ℕ)
    (blocks : ℕ)
    (houses_per_block : ℕ)
    (h1 : total_mail_per_block = 32)
    (h2 : blocks = 55)
    (h3 : houses_per_block = 4) :
  total_mail_per_block / houses_per_block = 8 :=
by
  sorry

end NUMINAMATH_GPT_mailman_distribution_l1188_118879


namespace NUMINAMATH_GPT_depth_multiple_of_rons_height_l1188_118871

theorem depth_multiple_of_rons_height (h d : ℕ) (Ron_height : h = 13) (water_depth : d = 208) : d = 16 * h := by
  sorry

end NUMINAMATH_GPT_depth_multiple_of_rons_height_l1188_118871


namespace NUMINAMATH_GPT_range_of_x_l1188_118823

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x + 1

-- State the theorem to prove the condition
theorem range_of_x (x : ℝ) : f (1 - x) + f (2 * x) > 2 ↔ x > -1 :=
by {
  sorry -- Proof placeholder
}

end NUMINAMATH_GPT_range_of_x_l1188_118823


namespace NUMINAMATH_GPT_all_points_lie_on_line_l1188_118846

theorem all_points_lie_on_line:
  ∀ (s : ℝ), s ≠ 0 → ∀ (x y : ℝ),
  x = (2 * s + 3) / s → y = (2 * s - 3) / s → x + y = 4 :=
by
  intros s hs x y hx hy
  sorry

end NUMINAMATH_GPT_all_points_lie_on_line_l1188_118846


namespace NUMINAMATH_GPT_inequality_abc_l1188_118815

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (cond : a + b + c = (1/a) + (1/b) + (1/c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l1188_118815


namespace NUMINAMATH_GPT_kevin_ends_with_604_cards_l1188_118889

theorem kevin_ends_with_604_cards : 
  ∀ (initial_cards found_cards : ℕ), initial_cards = 65 → found_cards = 539 → initial_cards + found_cards = 604 :=
by
  intros initial_cards found_cards h_initial h_found
  sorry

end NUMINAMATH_GPT_kevin_ends_with_604_cards_l1188_118889


namespace NUMINAMATH_GPT_elena_snow_removal_l1188_118827

theorem elena_snow_removal :
  ∀ (length width depth : ℝ) (compaction_factor : ℝ), 
  length = 30 ∧ width = 3 ∧ depth = 0.75 ∧ compaction_factor = 0.90 → 
  (length * width * depth * compaction_factor = 60.75) :=
by
  intros length width depth compaction_factor h
  obtain ⟨length_eq, width_eq, depth_eq, compaction_factor_eq⟩ := h
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_elena_snow_removal_l1188_118827


namespace NUMINAMATH_GPT_find_base_b_l1188_118857

theorem find_base_b (b : ℕ) : ( (2 * b + 5) ^ 2 = 6 * b ^ 2 + 5 * b + 5 ) → b = 9 := 
by 
  sorry  -- Proof is not required as per instruction

end NUMINAMATH_GPT_find_base_b_l1188_118857


namespace NUMINAMATH_GPT_Julie_hours_per_week_school_l1188_118898

noncomputable def summer_rate : ℚ := 4500 / (36 * 10)

noncomputable def school_rate : ℚ := summer_rate * 1.10

noncomputable def total_school_hours_needed : ℚ := 9000 / school_rate

noncomputable def hours_per_week_school : ℚ := total_school_hours_needed / 40

theorem Julie_hours_per_week_school : hours_per_week_school = 16.36 := by
  sorry

end NUMINAMATH_GPT_Julie_hours_per_week_school_l1188_118898


namespace NUMINAMATH_GPT_A_and_C_complete_remaining_work_in_2_point_4_days_l1188_118804

def work_rate_A : ℚ := 1 / 12
def work_rate_B : ℚ := 1 / 15
def work_rate_C : ℚ := 1 / 18
def work_completed_B_in_10_days : ℚ := (10 : ℚ) * work_rate_B
def remaining_work : ℚ := 1 - work_completed_B_in_10_days
def combined_work_rate_AC : ℚ := work_rate_A + work_rate_C
def time_to_complete_remaining_work : ℚ := remaining_work / combined_work_rate_AC

theorem A_and_C_complete_remaining_work_in_2_point_4_days :
  time_to_complete_remaining_work = 2.4 := 
sorry

end NUMINAMATH_GPT_A_and_C_complete_remaining_work_in_2_point_4_days_l1188_118804


namespace NUMINAMATH_GPT_red_pigment_weight_in_brown_paint_l1188_118884

theorem red_pigment_weight_in_brown_paint :
  ∀ (M G : ℝ), 
    (M + G = 10) → 
    (0.5 * M + 0.3 * G = 4) →
    0.5 * M = 2.5 :=
by sorry

end NUMINAMATH_GPT_red_pigment_weight_in_brown_paint_l1188_118884


namespace NUMINAMATH_GPT_problem_solution_l1188_118883

variables (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x_0 : ℝ, x_0^2 + (a-1)*x_0 + 1 < 0

theorem problem_solution (h₁ : p a ∨ q a) (h₂ : ¬(p a ∧ q a)) :
  -1 ≤ a ∧ a ≤ 1 ∨ a > 3 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1188_118883


namespace NUMINAMATH_GPT_functions_satisfying_equation_are_constants_l1188_118838

theorem functions_satisfying_equation_are_constants (f g : ℝ → ℝ) :
  (∀ x y : ℝ, f (f (x + y)) = x * f y + g x) → ∃ k : ℝ, (∀ x : ℝ, f x = k) ∧ (∀ x : ℝ, g x = k * (1 - x)) :=
by
  sorry

end NUMINAMATH_GPT_functions_satisfying_equation_are_constants_l1188_118838


namespace NUMINAMATH_GPT_initial_deck_card_count_l1188_118878

theorem initial_deck_card_count (r n : ℕ) (h1 : n = 2 * r) (h2 : n + 4 = 3 * r) : r + n = 12 := by
  sorry

end NUMINAMATH_GPT_initial_deck_card_count_l1188_118878


namespace NUMINAMATH_GPT_mrs_hilt_has_more_money_l1188_118858

/-- Mrs. Hilt has two pennies, two dimes, and two nickels. 
    Jacob has four pennies, one nickel, and one dime. 
    Prove that Mrs. Hilt has $0.13 more than Jacob. -/
theorem mrs_hilt_has_more_money 
  (hilt_pennies hilt_dimes hilt_nickels : ℕ)
  (jacob_pennies jacob_dimes jacob_nickels : ℕ)
  (value_penny value_nickel value_dime : ℝ)
  (H1 : hilt_pennies = 2) (H2 : hilt_dimes = 2) (H3 : hilt_nickels = 2)
  (H4 : jacob_pennies = 4) (H5 : jacob_dimes = 1) (H6 : jacob_nickels = 1)
  (H7 : value_penny = 0.01) (H8 : value_nickel = 0.05) (H9 : value_dime = 0.10) :
  ((hilt_pennies * value_penny + hilt_dimes * value_dime + hilt_nickels * value_nickel) 
   - (jacob_pennies * value_penny + jacob_dimes * value_dime + jacob_nickels * value_nickel) 
   = 0.13) :=
by sorry

end NUMINAMATH_GPT_mrs_hilt_has_more_money_l1188_118858


namespace NUMINAMATH_GPT_board_division_condition_l1188_118843

open Nat

theorem board_division_condition (n : ℕ) : 
  (∃ k : ℕ, n = 4 * k) ↔ 
  (∃ v h : ℕ, v = h ∧ (2 * v + 2 * h = n * n ∧ n % 2 = 0)) := 
sorry

end NUMINAMATH_GPT_board_division_condition_l1188_118843


namespace NUMINAMATH_GPT_no_solutions_for_inequalities_l1188_118819

theorem no_solutions_for_inequalities (x y z t : ℝ) :
  |x| < |y - z + t| →
  |y| < |x - z + t| →
  |z| < |x - y + t| →
  |t| < |x - y + z| →
  False :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_for_inequalities_l1188_118819


namespace NUMINAMATH_GPT_tom_apple_fraction_l1188_118887

theorem tom_apple_fraction (initial_oranges initial_apples oranges_sold_fraction oranges_remaining total_fruits_remaining apples_initial apples_sold_fraction : ℕ→ℚ) :
  initial_oranges = 40 →
  initial_apples = 70 →
  oranges_sold_fraction = 1 / 4 →
  oranges_remaining = initial_oranges - initial_oranges * oranges_sold_fraction →
  total_fruits_remaining = 65 →
  total_fruits_remaining = oranges_remaining + (initial_apples - initial_apples * apples_sold_fraction) →
  apples_sold_fraction = 1 / 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_tom_apple_fraction_l1188_118887


namespace NUMINAMATH_GPT_remaining_days_to_finish_l1188_118840

-- Define initial conditions and constants
def initial_play_hours_per_day : ℕ := 4
def initial_days : ℕ := 14
def completion_fraction : ℚ := 0.40
def increased_play_hours_per_day : ℕ := 7

-- Define the calculation for total initial hours played
def total_initial_hours_played : ℕ := initial_play_hours_per_day * initial_days

-- Define the total hours needed to complete the game
def total_hours_to_finish := total_initial_hours_played / completion_fraction

-- Define the remaining hours needed to finish the game
def remaining_hours := total_hours_to_finish - total_initial_hours_played

-- Prove that the remaining days to finish the game is 12
theorem remaining_days_to_finish : (remaining_hours / increased_play_hours_per_day) = 12 := by
  sorry -- Proof steps go here

end NUMINAMATH_GPT_remaining_days_to_finish_l1188_118840


namespace NUMINAMATH_GPT_kona_additional_miles_l1188_118851

theorem kona_additional_miles 
  (d_apartment_to_bakery : ℕ := 9) 
  (d_bakery_to_grandmother : ℕ := 24) 
  (d_grandmother_to_apartment : ℕ := 27) : 
  (d_apartment_to_bakery + d_bakery_to_grandmother + d_grandmother_to_apartment) - (2 * d_grandmother_to_apartment) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_kona_additional_miles_l1188_118851


namespace NUMINAMATH_GPT_cost_of_shorts_l1188_118813

-- Define the given conditions and quantities
def initial_money : ℕ := 50
def jerseys_cost : ℕ := 5 * 2
def basketball_cost : ℕ := 18
def remaining_money : ℕ := 14

-- The total amount spent
def total_spent : ℕ := initial_money - remaining_money

-- The total cost of the jerseys and basketball
def jerseys_basketball_cost : ℕ := jerseys_cost + basketball_cost

-- The cost of the shorts
def shorts_cost : ℕ := total_spent - jerseys_basketball_cost

theorem cost_of_shorts : shorts_cost = 8 := sorry

end NUMINAMATH_GPT_cost_of_shorts_l1188_118813


namespace NUMINAMATH_GPT_chocolate_bar_cost_l1188_118825

theorem chocolate_bar_cost (x : ℝ) (total_bars : ℕ) (bars_sold : ℕ) (total_amount_made : ℝ)
    (h1 : total_bars = 7)
    (h2 : bars_sold = total_bars - 4)
    (h3 : total_amount_made = 9)
    (h4 : total_amount_made = bars_sold * x) : x = 3 :=
sorry

end NUMINAMATH_GPT_chocolate_bar_cost_l1188_118825


namespace NUMINAMATH_GPT_pens_bought_l1188_118818

theorem pens_bought
  (P : ℝ)
  (cost := 36 * P)
  (discount := 0.99 * P)
  (profit_percent := 0.1)
  (profit := (40 * discount) - cost)
  (profit_eq : profit = profit_percent * cost) :
  40 = 40 := 
by
  sorry

end NUMINAMATH_GPT_pens_bought_l1188_118818


namespace NUMINAMATH_GPT_find_fx_for_l1188_118886

theorem find_fx_for {f : ℕ → ℤ} (h1 : f 0 = 1) (h2 : ∀ x, f (x + 1) = f x + 2 * x + 3) : f 2012 = 4052169 :=
by
  sorry

end NUMINAMATH_GPT_find_fx_for_l1188_118886


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1188_118866

theorem sum_of_two_numbers (a b : ℕ) (h1 : (a + b) * (a - b) = 1996) (h2 : (a + b) % 2 = (a - b) % 2) (h3 : a + b > a - b) : a + b = 998 := 
sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1188_118866


namespace NUMINAMATH_GPT_gcd_three_numbers_l1188_118817

def a : ℕ := 8650
def b : ℕ := 11570
def c : ℕ := 28980

theorem gcd_three_numbers : Nat.gcd (Nat.gcd a b) c = 10 :=
by 
  sorry

end NUMINAMATH_GPT_gcd_three_numbers_l1188_118817


namespace NUMINAMATH_GPT_correct_answer_l1188_118822

def mary_initial_cards : ℝ := 18.0
def mary_bought_cards : ℝ := 40.0
def mary_left_cards : ℝ := 32.0
def mary_promised_cards (initial_cards : ℝ) (bought_cards : ℝ) (left_cards : ℝ) : ℝ :=
  initial_cards + bought_cards - left_cards

theorem correct_answer :
  mary_promised_cards mary_initial_cards mary_bought_cards mary_left_cards = 26.0 := by
  sorry

end NUMINAMATH_GPT_correct_answer_l1188_118822


namespace NUMINAMATH_GPT_system_solution_a_l1188_118826

theorem system_solution_a (x y a : ℝ) (h1 : 3 * x + y = a) (h2 : 2 * x + 5 * y = 2 * a) (hx : x = 3) : a = 13 :=
by
  sorry

end NUMINAMATH_GPT_system_solution_a_l1188_118826


namespace NUMINAMATH_GPT_max_area_14_5_l1188_118836

noncomputable def rectangle_max_area (P D : ℕ) (x y : ℝ) : ℝ :=
  if (2 * x + 2 * y = P) ∧ (x^2 + y^2 = D^2) then x * y else 0

theorem max_area_14_5 :
  ∃ (x y : ℝ), (2 * x + 2 * y = 14) ∧ (x^2 + y^2 = 5^2) ∧ rectangle_max_area 14 5 x y = 12.25 :=
by
  sorry

end NUMINAMATH_GPT_max_area_14_5_l1188_118836
