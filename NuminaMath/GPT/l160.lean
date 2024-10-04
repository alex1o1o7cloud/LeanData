import Mathlib

namespace jason_advertising_cost_l160_160543

def magazine_length : ℕ := 9
def magazine_width : ℕ := 12
def cost_per_square_inch : ℕ := 8
def half (x : ℕ) := x / 2
def area (L W : ℕ) := L * W
def total_cost (a c : ℕ) := a * c

theorem jason_advertising_cost :
  total_cost (half (area magazine_length magazine_width)) cost_per_square_inch = 432 := by
  sorry

end jason_advertising_cost_l160_160543


namespace limit_r_l160_160840

noncomputable def L (m : ℝ) : ℝ := (m - Real.sqrt (m^2 + 24)) / 2

noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

theorem limit_r (h : ∀ m : ℝ, m ≠ 0) : Filter.Tendsto r (nhds 0) (nhds (-1)) :=
sorry

end limit_r_l160_160840


namespace find_area_parallelogram_l160_160538

open Real

noncomputable def area_of_parallelogram (u v : Vec3) (h1 : ‖u‖ = 1) (h2 : ‖v‖ = 1) (h3 : angle u v = π / 4) : ℝ :=
  ‖(-(u + 3 * v) + (3 * u + v)) × (2 * (u + 3 * v))‖

theorem find_area_parallelogram (u v : Vec3) (h1 : ‖u‖ = 1) (h2 : ‖v‖ = 1) (h3 : angle u v = π / 4) :
  area_of_parallelogram u v h1 h2 h3 = 2 * sqrt 2 := 
sorry

end find_area_parallelogram_l160_160538


namespace max_value_of_g_l160_160859

def g (n : ℕ) : ℕ :=
  if n < 15 then n + 15 else g (n - 7)

theorem max_value_of_g : ∃ m, ∀ n, g n ≤ m ∧ (∃ k, g k = m) :=
by
  use 29
  sorry

end max_value_of_g_l160_160859


namespace weighted_average_plants_per_hour_l160_160245

theorem weighted_average_plants_per_hour :
  let heath_carrot_plants_100 := 100 * 275
  let heath_carrot_plants_150 := 150 * 325
  let heath_total_plants := heath_carrot_plants_100 + heath_carrot_plants_150
  let heath_total_time := 10 + 20
  
  let jake_potato_plants_50 := 50 * 300
  let jake_potato_plants_100 := 100 * 400
  let jake_total_plants := jake_potato_plants_50 + jake_potato_plants_100
  let jake_total_time := 12 + 18

  let total_plants := heath_total_plants + jake_total_plants
  let total_time := heath_total_time + jake_total_time
  let weighted_average := total_plants / total_time
  weighted_average = 2187.5 :=
by
  sorry

end weighted_average_plants_per_hour_l160_160245


namespace spherical_to_rectangular_coordinates_l160_160480

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  spherical_to_rectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coordinates_l160_160480


namespace tangent_line_circle_l160_160254

theorem tangent_line_circle (a : ℝ) : (∀ x y : ℝ, a * x + y + 1 = 0) → (∀ x y : ℝ, x^2 + y^2 - 4 * x = 0) → a = 3 / 4 :=
by
  sorry

end tangent_line_circle_l160_160254


namespace sum_of_interior_edges_l160_160630

theorem sum_of_interior_edges (frame_width : ℕ) (frame_area : ℕ) (outer_edge : ℕ) 
  (H1 : frame_width = 2) (H2 : frame_area = 32) (H3 : outer_edge = 7) : 
  2 * (outer_edge - 2 * frame_width) + 2 * (x : ℕ) = 8 :=
by
  sorry

end sum_of_interior_edges_l160_160630


namespace Q_no_negative_roots_and_at_least_one_positive_root_l160_160649

def Q (x : ℝ) : ℝ := x^7 - 2 * x^6 - 6 * x^4 - 4 * x + 16

theorem Q_no_negative_roots_and_at_least_one_positive_root :
  (∀ x, x < 0 → Q x > 0) ∧ (∃ x, x > 0 ∧ Q x = 0) := 
sorry

end Q_no_negative_roots_and_at_least_one_positive_root_l160_160649


namespace rectangle_area_l160_160464

theorem rectangle_area (a : ℕ) (w l : ℕ) (h_square_area : a = 36) (h_square_side : w * w = a) (h_rectangle_length : l = 3 * w) : w * l = 108 :=
by
  -- Placeholder for proof
  sorry

end rectangle_area_l160_160464


namespace inequality_proof_l160_160802

theorem inequality_proof
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (ha1 : 0 < a1) (ha2 : 0 < a2) (ha3 : 0 < a3)
  (hb1 : 0 < b1) (hb2 : 0 < b2) (hb3 : 0 < b3) :
  (a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2 + a3 * b1 + a1 * b3)^2 ≥
    4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) :=
sorry

end inequality_proof_l160_160802


namespace Haman_initial_trays_l160_160114

theorem Haman_initial_trays 
  (eggs_in_tray : ℕ)
  (total_eggs_sold : ℕ)
  (trays_dropped : ℕ)
  (additional_trays : ℕ)
  (trays_finally_sold : ℕ)
  (std_trays_sold : total_eggs_sold / eggs_in_tray = trays_finally_sold) 
  (eggs_in_tray_def : eggs_in_tray = 30) 
  (total_eggs_sold_def : total_eggs_sold = 540)
  (trays_dropped_def : trays_dropped = 2)
  (additional_trays_def : additional_trays = 7) :
  trays_finally_sold - additional_trays + trays_dropped = 13 := 
by 
  sorry

end Haman_initial_trays_l160_160114


namespace sin_pi_six_minus_alpha_eq_one_third_cos_two_answer_l160_160672

theorem sin_pi_six_minus_alpha_eq_one_third_cos_two_answer
  (α : ℝ) (h1 : Real.sin (π / 6 - α) = 1 / 3) :
  2 * Real.cos (π / 6 + α / 2) ^ 2 - 1 = 1 / 3 := by
  sorry

end sin_pi_six_minus_alpha_eq_one_third_cos_two_answer_l160_160672


namespace existence_of_root_l160_160156

noncomputable def f (x : ℝ) : ℝ := 2^x - x - 2

theorem existence_of_root : ∃ x ∈ Ioo (-2 : ℝ) (-1), f x = 0 := by
  sorry

end existence_of_root_l160_160156


namespace a_7_value_l160_160359

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n

-- Given conditions
def geometric_sequence_positive_terms (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

def geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (a 0 * (1 - ((a (1 + n)) / a 0))) / (1 - (a 1 / a 0))

def S_4_eq_3S_2 (S : ℕ → ℝ) : Prop :=
S 4 = 3 * S 2

def a_3_eq_2 (a : ℕ → ℝ) : Prop :=
a 3 = 2

-- The statement to prove
theorem a_7_value (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  geometric_sequence_positive_terms a →
  geometric_sequence_sum a S →
  S_4_eq_3S_2 S →
  a_3_eq_2 a →
  a 7 = 8 :=
by
  sorry

end a_7_value_l160_160359


namespace fraction_eq_repeating_decimal_l160_160066

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l160_160066


namespace repeating_decimal_as_fraction_l160_160051

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l160_160051


namespace most_economical_is_small_l160_160334

noncomputable def most_economical_size (c_S q_S c_M q_M c_L q_L : ℝ) :=
  c_M = 1.3 * c_S ∧
  q_M = 0.85 * q_L ∧
  q_L = 1.5 * q_S ∧
  c_L = 1.4 * c_M →
  (c_S / q_S < c_M / q_M) ∧ (c_S / q_S < c_L / q_L)

theorem most_economical_is_small (c_S q_S c_M q_M c_L q_L : ℝ) :
  most_economical_size c_S q_S c_M q_M c_L q_L := by 
  sorry

end most_economical_is_small_l160_160334


namespace fraction_is_one_fifth_l160_160748

theorem fraction_is_one_fifth (f : ℚ) (h1 : f * 50 - 4 = 6) : f = 1 / 5 :=
by
  sorry

end fraction_is_one_fifth_l160_160748


namespace negation_equiv_exists_l160_160841

theorem negation_equiv_exists : 
  ¬ (∀ x : ℝ, x^2 + 1 > 0) ↔ ∃ x_0 : ℝ, x_0^2 + 1 ≤ 0 := 
by 
  sorry

end negation_equiv_exists_l160_160841


namespace evaluate_expression_l160_160962

theorem evaluate_expression (c d : ℕ) (hc : c = 4) (hd : d = 2) :
  (c^c - c * (c - d)^c)^c = 136048896 := by
  sorry

end evaluate_expression_l160_160962


namespace sum_of_coordinates_D_l160_160547

theorem sum_of_coordinates_D (x y : Int) :
  let N := (4, 10)
  let C := (14, 6)
  let D := (x, y)
  N = ((x + 14) / 2, (y + 6) / 2) →
  x + y = 8 :=
by
  intros
  sorry

end sum_of_coordinates_D_l160_160547


namespace difference_of_squares_l160_160192

theorem difference_of_squares (a b : ℕ) (h₁ : a = 69842) (h₂ : b = 30158) :
  (a^2 - b^2) / (a - b) = 100000 :=
by
  rw [h₁, h₂]
  sorry

end difference_of_squares_l160_160192


namespace consecutive_integer_sum_l160_160314

theorem consecutive_integer_sum (a b c : ℕ) 
  (h1 : b = a + 2) 
  (h2 : c = a + 4) 
  (h3 : a + c = 140) 
  (h4 : b - a = 2) : a + b + c = 210 := 
sorry

end consecutive_integer_sum_l160_160314


namespace ratio_of_sums_of_sides_and_sines_l160_160999

theorem ratio_of_sums_of_sides_and_sines (A B C : ℝ) (a b c : ℝ) (hA : A = 60) (ha : a = 3) 
  (h : a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C) : 
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 3 := 
by 
  sorry

end ratio_of_sums_of_sides_and_sines_l160_160999


namespace repeating_decimal_eq_fraction_l160_160057

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l160_160057


namespace find_second_number_l160_160855

theorem find_second_number (x : ℕ) : 
  ((20 + 40 + 60) / 3 = 4 + ((x + 10 + 28) / 3)) → x = 70 :=
by {
  -- let lhs = (20 + 40 + 60) / 3
  -- let rhs = 4 + ((x + 10 + 28) / 3)
  -- rw rhs at lhs,
  -- value the lhs and rhs,
  -- prove x = 70
  sorry
}

end find_second_number_l160_160855


namespace graph_of_equation_l160_160918

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 :=
by sorry

end graph_of_equation_l160_160918


namespace ellipse_conjugate_diameters_l160_160992

variable (A B C D E : ℝ)

theorem ellipse_conjugate_diameters :
  (A * E - B * D = 0) ∧ (2 * B ^ 2 + (A - C) * A = 0) :=
sorry

end ellipse_conjugate_diameters_l160_160992


namespace product_of_solutions_l160_160596

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l160_160596


namespace compute_expression_l160_160767

noncomputable def a : ℝ := 125^(1/3)
noncomputable def b : ℝ := (-2/3)^0
noncomputable def c : ℝ := Real.log 8 / Real.log 2

theorem compute_expression : a - b - c = 1 := by
  sorry

end compute_expression_l160_160767


namespace cost_of_fencing_per_meter_l160_160158

theorem cost_of_fencing_per_meter
  (length breadth : ℕ)
  (total_cost : ℝ)
  (h1 : length = breadth + 20)
  (h2 : length = 60)
  (h3 : total_cost = 5300) :
  (total_cost / (2 * length + 2 * breadth)) = 26.5 := 
by
  sorry

end cost_of_fencing_per_meter_l160_160158


namespace polygon_coloring_l160_160231

theorem polygon_coloring (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 3) :
    ∃ b_n : ℕ, b_n = (m - 1) * ((m - 1) ^ (n - 1) + (-1 : ℤ) ^ n) :=
sorry

end polygon_coloring_l160_160231


namespace rectangles_cannot_cover_large_rectangle_l160_160102

theorem rectangles_cannot_cover_large_rectangle (n m : ℕ) (a b c d: ℕ) : 
  n = 14 → m = 9 → a = 2 → b = 3 → c = 3 → d = 2 → 
  (∀ (v_rects : ℕ) (h_rects : ℕ), v_rects = 10 → h_rects = 11 →
    (∀ (rect_area : ℕ), rect_area = n * m →
      (∀ (small_rect_area : ℕ), 
        small_rect_area = (v_rects * (a * b)) + (h_rects * (c * d)) →
        small_rect_area = rect_area → 
        false))) :=
by
  intros n_eq m_eq a_eq b_eq c_eq d_eq
       v_rects h_rects v_rects_eq h_rects_eq
       rect_area rect_area_eq small_rect_area small_rect_area_eq area_sum_eq
  sorry

end rectangles_cannot_cover_large_rectangle_l160_160102


namespace sum_of_five_consecutive_odd_integers_l160_160881

theorem sum_of_five_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 8) = 156) :
  n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 390 :=
by
  sorry

end sum_of_five_consecutive_odd_integers_l160_160881


namespace sum_of_remainders_l160_160399

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 9) : (n % 4 + n % 5) = 5 :=
by
  sorry

end sum_of_remainders_l160_160399


namespace probability_scrapped_l160_160746

variable (P_A P_B_given_not_A : ℝ)
variable (prob_scrapped : ℝ)

def fail_first_inspection (P_A : ℝ) := 1 - P_A
def fail_second_inspection_given_fails_first (P_B_given_not_A : ℝ) := 1 - P_B_given_not_A

theorem probability_scrapped (h1 : P_A = 0.8) (h2 : P_B_given_not_A = 0.9) (h3 : prob_scrapped = fail_first_inspection P_A * fail_second_inspection_given_fails_first P_B_given_not_A) :
  prob_scrapped = 0.02 := by
  sorry

end probability_scrapped_l160_160746


namespace probability_same_number_of_flips_l160_160640

theorem probability_same_number_of_flips (p_head : ℝ) (p_tail : ℝ) :
  (p_head = (1 / 3)) → (p_tail = (2 / 3)) →
  (∑ n in (set.Icc 1 (99 : ℕ)), ((p_tail) ^ (n - 1) * p_head) ^ 4) = (81 / 65) :=
by
  -- Given the conditions that p_head = 1/3 and p_tail = 2/3,
  assume h1 : p_head = 1 / 3,
  assume h2 : p_tail = 2 / 3,
  sorry

end probability_same_number_of_flips_l160_160640


namespace factorize_x_cubed_minus_9x_l160_160780

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l160_160780


namespace roots_are_distinct_l160_160236

theorem roots_are_distinct (a x1 x2 : ℝ) (h : x1 ≠ x2) :
  (∀ x, x^2 - a*x - 2 = 0 → x = x1 ∨ x = x2) → x1 ≠ x2 := sorry

end roots_are_distinct_l160_160236


namespace supremum_neg_frac_bound_l160_160138

noncomputable def supremum_neg_frac (a b : ℝ) : ℝ :=
  - (1 / (2 * a)) - (2 / b)

theorem supremum_neg_frac_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  supremum_neg_frac a b ≤ - 9 / 2 :=
sorry

end supremum_neg_frac_bound_l160_160138


namespace value_of_Z_4_3_l160_160645

def Z (a b : ℤ) : ℤ := a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3

theorem value_of_Z_4_3 : Z 4 3 = 1 := by
  sorry

end value_of_Z_4_3_l160_160645


namespace jerry_showers_l160_160654

variable (water_allowance : ℕ) (drinking_cooking : ℕ) (water_per_shower : ℕ) (pool_length : ℕ) 
  (pool_width : ℕ) (pool_height : ℕ) (gallons_per_cubic_foot : ℕ)

/-- Jerry can take 15 showers in July given the conditions. -/
theorem jerry_showers :
  water_allowance = 1000 →
  drinking_cooking = 100 →
  water_per_shower = 20 →
  pool_length = 10 →
  pool_width = 10 →
  pool_height = 6 →
  gallons_per_cubic_foot = 1 →
  (water_allowance - (drinking_cooking + (pool_length * pool_width * pool_height) * gallons_per_cubic_foot)) / water_per_shower = 15 :=
by
  intros h_water_allowance h_drinking_cooking h_water_per_shower h_pool_length h_pool_width h_pool_height h_gallons_per_cubic_foot
  sorry

end jerry_showers_l160_160654


namespace polynomial_sum_l160_160397

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l160_160397


namespace range_of_a_l160_160516

noncomputable def A : Set ℝ := Set.Ico 1 5 -- A = [1, 5)
noncomputable def B (a : ℝ) : Set ℝ := Set.Iio a -- B = (-∞, a)

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 5 ≤ a :=
sorry

end range_of_a_l160_160516


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160903

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160903


namespace product_of_all_solutions_triple_reciprocal_l160_160584

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l160_160584


namespace cos_of_sin_given_l160_160524

theorem cos_of_sin_given (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by
  sorry

end cos_of_sin_given_l160_160524


namespace rectangle_properties_l160_160225

noncomputable def diagonal (x1 y1 x2 y2 : ℕ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def area (length width : ℕ) : ℕ :=
  length * width

theorem rectangle_properties :
  diagonal 1 1 9 7 = 10 ∧ area (9 - 1) (7 - 1) = 48 := by
  sorry

end rectangle_properties_l160_160225


namespace find_other_root_l160_160011

theorem find_other_root (x : ℚ) (h: 63 * x^2 - 100 * x + 45 = 0) (hx: x = 5 / 7) : x = 1 ∨ x = 5 / 7 :=
by 
  -- Insert the proof steps here if needed.
  sorry

end find_other_root_l160_160011


namespace repeating_decimal_as_fraction_l160_160052

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l160_160052


namespace parking_spots_l160_160629

def numberOfLevels := 5
def openSpotsOnLevel1 := 4
def openSpotsOnLevel2 := openSpotsOnLevel1 + 7
def openSpotsOnLevel3 := openSpotsOnLevel2 + 6
def openSpotsOnLevel4 := 14
def openSpotsOnLevel5 := openSpotsOnLevel4 + 5
def totalOpenSpots := openSpotsOnLevel1 + openSpotsOnLevel2 + openSpotsOnLevel3 + openSpotsOnLevel4 + openSpotsOnLevel5

theorem parking_spots :
  openSpotsOnLevel5 = 19 ∧ totalOpenSpots = 65 := by
  sorry

end parking_spots_l160_160629


namespace poly_sum_correct_l160_160396

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := -4 * x^2 + 12 * x - 12

theorem poly_sum_correct : ∀ x : ℝ, p x + q x + r x = s x :=
by
  sorry

end poly_sum_correct_l160_160396


namespace garden_to_land_area_ratio_l160_160722

variables (l_ter w_ter l_gard w_gard : ℝ)

-- Condition 1: Width of the land rectangle is 3/5 of its length
def land_conditions : Prop := w_ter = (3 / 5) * l_ter

-- Condition 2: Width of the garden rectangle is 3/5 of its length
def garden_conditions : Prop := w_gard = (3 / 5) * l_gard

-- Problem: Ratio of the area of the garden to the area of the land is 36%.
theorem garden_to_land_area_ratio
  (h_land : land_conditions l_ter w_ter)
  (h_garden : garden_conditions l_gard w_gard) :
  (l_gard * w_gard) / (l_ter * w_ter) = 0.36 := sorry

end garden_to_land_area_ratio_l160_160722


namespace danil_claim_false_l160_160644

theorem danil_claim_false (E O : ℕ) (hE : E % 2 = 0) (hO : O % 2 = 0) (h : O = E + 15) : false :=
by sorry

end danil_claim_false_l160_160644


namespace infinite_solutions_l160_160214

-- Define the system of linear equations
def eq1 (x y : ℝ) : Prop := 3 * x - 4 * y = 1
def eq2 (x y : ℝ) : Prop := 6 * x - 8 * y = 2

-- State that there are an unlimited number of solutions
theorem infinite_solutions : ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧
  ∀ y : ℝ, ∃ x : ℝ, eq1 x y :=
by
  sorry

end infinite_solutions_l160_160214


namespace inner_circle_radius_is_sqrt_2_l160_160637

noncomputable def radius_of_inner_circle (side_length : ℝ) : ℝ :=
  let semicircle_radius := side_length / 4
  let distance_from_center_to_semicircle_center :=
    Real.sqrt ((side_length / 2) ^ 2 + (side_length / 2) ^ 2)
  let inner_circle_radius := (distance_from_center_to_semicircle_center - semicircle_radius)
  inner_circle_radius

theorem inner_circle_radius_is_sqrt_2 (side_length : ℝ) (h: side_length = 4) : 
  radius_of_inner_circle side_length = Real.sqrt 2 :=
by
  sorry

end inner_circle_radius_is_sqrt_2_l160_160637


namespace prism_volume_l160_160570

theorem prism_volume (x : ℝ) (L W H : ℝ) (hL : L = 2 * x) (hW : W = x) (hH : H = 1.5 * x) 
  (hedges_sum : 4 * L + 4 * W + 4 * H = 72) : 
  L * W * H = 192 := 
by
  sorry

end prism_volume_l160_160570


namespace daysRequired_l160_160690

-- Defining the structure of the problem
structure WallConstruction where
  m1 : ℕ    -- Number of men in the first scenario
  d1 : ℕ    -- Number of days in the first scenario
  m2 : ℕ    -- Number of men in the second scenario

-- Given values
def wallConstructionProblem : WallConstruction :=
  WallConstruction.mk 20 5 30

-- The total work constant
def totalWork (wc : WallConstruction) : ℕ :=
  wc.m1 * wc.d1

-- Proving the number of days required for m2 men
theorem daysRequired (wc : WallConstruction) (k : ℕ) : 
  k = totalWork wc → (wc.m2 * (k / wc.m2 : ℚ) = k) → (k / wc.m2 : ℚ) = 3.3 :=
by
  intro h1 h2
  sorry

end daysRequired_l160_160690


namespace evaluate_polynomial_at_4_l160_160028

-- Define the polynomial function
def polynomial (x : ℕ) : ℕ := x^4 + x^3 + x^2 + x + 1

-- Statement that evaluates the polynomial at x = 4 to be 341
theorem evaluate_polynomial_at_4 : polynomial 4 = 341 := 
by
  simp [polynomial]
  norm_num
  triv
  sorry

end evaluate_polynomial_at_4_l160_160028


namespace ninth_term_arith_seq_l160_160168

-- Define the arithmetic sequence.
def arith_seq (a₁ d : ℚ) (n : ℕ) := a₁ + n * d

-- Define the third and fifteenth terms of the sequence.
def third_term := (5 : ℚ) / 11
def fifteenth_term := (7 : ℚ) / 8

-- Prove that the ninth term is 117/176 given the conditions.
theorem ninth_term_arith_seq :
    ∃ (a₁ d : ℚ), 
    arith_seq a₁ d 2 = third_term ∧ 
    arith_seq a₁ d 14 = fifteenth_term ∧
    arith_seq a₁ d 8 = 117 / 176 :=
by
  sorry

end ninth_term_arith_seq_l160_160168


namespace fraction_for_repeating_56_l160_160037

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l160_160037


namespace students_not_in_same_column_or_row_l160_160120

-- Define the positions of student A and student B as conditions
structure Position where
  row : Nat
  col : Nat

-- Student A's position is in the 3rd row and 6th column
def StudentA : Position := {row := 3, col := 6}

-- Student B's position is described in a relative manner in terms of columns and rows
def StudentB : Position := {row := 6, col := 3}

-- Formalize the proof statement
theorem students_not_in_same_column_or_row :
  StudentA.row ≠ StudentB.row ∧ StudentA.col ≠ StudentB.col :=
by {
  sorry
}

end students_not_in_same_column_or_row_l160_160120


namespace intersecting_lines_l160_160424

theorem intersecting_lines (c d : ℝ) :
  (∀ x y : ℝ, (x = 1/3 * y + c ∧ y = 1/3 * x + d) → (x = 3 ∧ y = 3)) →
  c + d = 4 :=
by
  intros h
  -- We need to validate the condition holds at the intersection point
  have h₁ : 3 = 1/3 * 3 + c := by sorry
  have h₂ : 3 = 1/3 * 3 + d := by sorry
  -- Conclude that c = 2 and d = 2
  have hc : c = 2 := by sorry
  have hd : d = 2 := by sorry
  -- Thus the sum c + d = 4
  show 2 + 2 = 4 from rfl

end intersecting_lines_l160_160424


namespace lcm_12_18_24_l160_160577

theorem lcm_12_18_24 : Nat.lcm (Nat.lcm 12 18) 24 = 72 := by
  -- Given conditions (prime factorizations)
  have h1 : 12 = 2^2 * 3 := by norm_num
  have h2 : 18 = 2 * 3^2 := by norm_num
  have h3 : 24 = 2^3 * 3 := by norm_num
  -- Prove the LCM
  sorry

end lcm_12_18_24_l160_160577


namespace investor_profits_l160_160760

/-- Problem: Given the total contributions and profit sharing conditions, calculate the amount 
    each investor receives. -/

theorem investor_profits :
  ∀ (A_contribution B_contribution C_contribution D_contribution : ℝ) 
    (A_profit B_profit C_profit D_profit : ℝ) 
    (total_capital total_profit : ℝ),
    total_capital = 100000 → 
    A_contribution = B_contribution + 5000 →
    B_contribution = C_contribution + 10000 →
    C_contribution = D_contribution + 5000 →
    total_profit = 60000 →
    A_profit = (35 / 100) * total_profit * (1 + 10 / 100) →
    B_profit = (30 / 100) * total_profit * (1 + 8 / 100) →
    C_profit = (20 / 100) * total_profit * (1 + 5 / 100) → 
    D_profit = (15 / 100) * total_profit →
    (A_profit = 23100 ∧ B_profit = 19440 ∧ C_profit = 12600 ∧ D_profit = 9000) :=
by
  intros
  sorry

end investor_profits_l160_160760


namespace evaluate_polynomial_at_4_l160_160029

-- Define the polynomial function
def polynomial (x : ℕ) : ℕ := x^4 + x^3 + x^2 + x + 1

-- Statement that evaluates the polynomial at x = 4 to be 341
theorem evaluate_polynomial_at_4 : polynomial 4 = 341 := 
by
  simp [polynomial]
  norm_num
  triv
  sorry

end evaluate_polynomial_at_4_l160_160029


namespace Suresh_meeting_time_l160_160861

theorem Suresh_meeting_time :
  let C := 726
  let v1 := 75
  let v2 := 62.5
  C / (v1 + v2) = 5.28 := by
  sorry

end Suresh_meeting_time_l160_160861


namespace jacqueline_guavas_l160_160691

theorem jacqueline_guavas 
  (G : ℕ) 
  (plums : ℕ := 16) 
  (apples : ℕ := 21) 
  (given : ℕ := 40) 
  (remaining : ℕ := 15) 
  (initial_fruits : ℕ := plums + G + apples)
  (total_fruits_after_given : ℕ := remaining + given) : 
  initial_fruits = total_fruits_after_given → G = 18 := 
by
  intro h
  sorry

end jacqueline_guavas_l160_160691


namespace initial_visual_range_is_90_l160_160195

-- Define the initial visual range without the telescope (V).
variable (V : ℝ)

-- Define the condition that the visual range with the telescope is 150 km.
variable (condition1 : V + (2 / 3) * V = 150)

-- Define the proof problem statement.
theorem initial_visual_range_is_90 (V : ℝ) (condition1 : V + (2 / 3) * V = 150) : V = 90 :=
sorry

end initial_visual_range_is_90_l160_160195


namespace intersecting_lines_l160_160441

theorem intersecting_lines
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)
  (h_p1 : p1 = (3, 2))
  (h_line1 : ∀ x : ℝ, p1.2 = 3 * p1.1 + 4)
  (h_line2 : ∀ x : ℝ, p2.2 = - (1 / 3) * p2.1 + 3) :
  (∃ p : ℝ × ℝ, p = (-3 / 10, 31 / 10)) :=
by
  sorry

end intersecting_lines_l160_160441


namespace sum_of_powers_of_i_l160_160874

open Complex

def i := Complex.I

theorem sum_of_powers_of_i : (i + i^2 + i^3 + i^4) = 0 := 
by
  sorry

end sum_of_powers_of_i_l160_160874


namespace intersection_eq_set_l160_160284

def M : Set ℤ := { x | -4 < (x : Int) ∧ x < 2 }
def N : Set Int := { x | (x : ℝ) ^ 2 < 4 }
def intersection := M ∩ N

theorem intersection_eq_set : intersection = {-1, 0, 1} := 
sorry

end intersection_eq_set_l160_160284


namespace chiming_time_is_5_l160_160151

-- Define the conditions for the clocks
def queen_strikes (h : ℕ) : Prop := (2 * h) % 3 = 0
def king_strikes (h : ℕ) : Prop := (3 * h) % 2 = 0

-- Define the chiming synchronization at the same time condition
def chiming_synchronization (h: ℕ) : Prop :=
  3 * h = 2 * ((2 * h) + 2)

-- The proof statement
theorem chiming_time_is_5 : ∃ h: ℕ, queen_strikes h ∧ king_strikes h ∧ chiming_synchronization h ∧ h = 5 :=
by
  sorry

end chiming_time_is_5_l160_160151


namespace product_of_possible_values_of_x_l160_160262

theorem product_of_possible_values_of_x : 
  (∀ x, |x - 7| - 5 = 4 → x = 16 ∨ x = -2) -> (16 * -2 = -32) :=
by
  intro h
  have := h 16
  have := h (-2)
  sorry

end product_of_possible_values_of_x_l160_160262


namespace find_the_number_l160_160975

theorem find_the_number :
  ∃ x : ℕ, 72519 * x = 724827405 ∧ x = 10005 :=
by
  sorry

end find_the_number_l160_160975


namespace eggs_today_l160_160556

-- Condition definitions
def eggs_yesterday : ℕ := 10
def difference : ℕ := 59

-- Statement of the problem
theorem eggs_today : eggs_yesterday + difference = 69 := by
  sorry

end eggs_today_l160_160556


namespace repeating_decimal_is_fraction_l160_160063

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l160_160063


namespace projection_matrix_ordered_pair_l160_160864

theorem projection_matrix_ordered_pair (a c : ℚ)
  (P : Matrix (Fin 2) (Fin 2) ℚ) 
  (P := ![![a, 15 / 34], ![c, 25 / 34]]) :
  P * P = P ->
  (a, c) = (9 / 34, 15 / 34) :=
by
  sorry

end projection_matrix_ordered_pair_l160_160864


namespace lock_code_difference_l160_160290

theorem lock_code_difference :
  ∃ A B C D, A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
             (A = 4 ∧ B = 2 * C ∧ C = D) ∨
             (A = 9 ∧ B = 3 * C ∧ C = D) ∧
             (A * 100 + B * 10 + C - (D * 100 + (2 * D) * 10 + D)) = 541 :=
sorry

end lock_code_difference_l160_160290


namespace find_central_angle_l160_160103

theorem find_central_angle
  (θ r : ℝ)
  (h1 : r * θ = 2 * π)
  (h2 : (1 / 2) * r^2 * θ = 3 * π) :
  θ = 2 * π / 3 := 
sorry

end find_central_angle_l160_160103


namespace poly_sum_correct_l160_160395

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := -4 * x^2 + 12 * x - 12

theorem poly_sum_correct : ∀ x : ℝ, p x + q x + r x = s x :=
by
  sorry

end poly_sum_correct_l160_160395


namespace interior_edges_sum_l160_160635

theorem interior_edges_sum 
  (frame_thickness : ℝ)
  (frame_area : ℝ)
  (outer_length : ℝ)
  (frame_thickness_eq : frame_thickness = 2)
  (frame_area_eq : frame_area = 32)
  (outer_length_eq : outer_length = 7) 
  : ∃ interior_edges_sum : ℝ, interior_edges_sum = 8 := 
by
  sorry

end interior_edges_sum_l160_160635


namespace volume_ratio_spheres_l160_160107

theorem volume_ratio_spheres (R : ℝ) (hR : R > 0) :
  let r := (sqrt 3 / 2) * R in
  let volume_sphere := λ r : ℝ, (4 / 3) * π * r^3 in
  volume_sphere r / volume_sphere R = (3 / 8) * sqrt 3 :=
by
  sorry

end volume_ratio_spheres_l160_160107


namespace fraction_eq_repeating_decimal_l160_160065

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l160_160065


namespace students_not_enrolled_in_any_l160_160683

open Finset

variables (U : Finset ℕ) -- Universe of students

variables (F G S : Finset ℕ) -- Sets of students taking French, German, and Spanish
variables (h_card_U : U.card = 150)
variables (h_card_F : F.card = 60)
variables (h_card_G : G.card = 50)
variables (h_card_S : S.card = 40)
variables (h_card_FG : (F ∩ G).card = 20)
variables (h_card_FS : (F ∩ S).card = 15)
variables (h_card_GS : (G ∩ S).card = 10)
variables (h_card_FGS : (F ∩ G ∩ S).card = 5)

theorem students_not_enrolled_in_any :
  (U.card - (F ∪ G ∪ S).card) = 40 :=
by
  sorry

end students_not_enrolled_in_any_l160_160683


namespace simplify_expression_l160_160552

variables (a b : ℝ)

theorem simplify_expression : 
  (2 * a^2 - 3 * a * b + 8) - (-a * b - a^2 + 8) = 3 * a^2 - 2 * a * b :=
by sorry

-- Note:
-- ℝ denotes real numbers. Adjust types accordingly if using different numerical domains (e.g., ℚ, ℂ).

end simplify_expression_l160_160552


namespace series_sum_eq_one_sixth_l160_160218

noncomputable def series_sum := 
  ∑' n : ℕ, (3^n) / ((7^ (2^n)) + 1)

theorem series_sum_eq_one_sixth : series_sum = 1 / 6 := 
  sorry

end series_sum_eq_one_sixth_l160_160218


namespace tangent_line_at_neg_one_l160_160420

noncomputable def f (x : ℝ) : ℝ := x^3 + (1 / x)

noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem tangent_line_at_neg_one : 
  let x := -1 in
  let y := f x in
  2 * x - y = 0 :=
by
  sorry

end tangent_line_at_neg_one_l160_160420


namespace arc_length_l160_160373

theorem arc_length (r : ℝ) (α : ℝ) (h_r : r = 10) (h_α : α = 2 * Real.pi / 3) : 
  r * α = 20 * Real.pi / 3 := 
by {
sorry
}

end arc_length_l160_160373


namespace remainder_is_five_l160_160966

theorem remainder_is_five (A : ℕ) (h : 17 = 6 * 2 + A) : A = 5 :=
sorry

end remainder_is_five_l160_160966


namespace focus_of_parabola_l160_160973

-- Problem statement
theorem focus_of_parabola (x y : ℝ) : (2 * x^2 = -y) → (focus_coordinates = (0, -1 / 8)) :=
by
  sorry

end focus_of_parabola_l160_160973


namespace bruce_bhishma_meet_again_l160_160182

theorem bruce_bhishma_meet_again (L S_B S_H : ℕ) (hL : L = 600) (hSB : S_B = 30) (hSH : S_H = 20) : 
  ∃ t : ℕ, t = 60 ∧ (t * S_B - t * S_H) % L = 0 :=
by
  sorry

end bruce_bhishma_meet_again_l160_160182


namespace starting_lineups_count_l160_160340

theorem starting_lineups_count (players : Finset ℕ) (guards : Finset ℕ) (all_stars : Finset ℕ) (H_total : players.card = 15) (H_guards : guards.card = 5)
  (H_all_stars : all_stars.card = 3) (H_all_stars_subset : all_stars ⊆ players) (H_guards_subset : guards ⊆ players) 
  (H_all_stars_lineup : ∀ star ∈ all_stars, star ∈ players) (H_lineup : ∀ player ∈ players, player∈ guards ∨ player ∉ guards) :
  ((players \ all_stars).card.choose 4 * 15.choose 7 = 285) :=
by
  sorry

end starting_lineups_count_l160_160340


namespace paving_stone_size_l160_160202

theorem paving_stone_size (length_courtyard width_courtyard : ℕ) (num_paving_stones : ℕ) (area_courtyard : ℕ) (s : ℕ)
  (h₁ : length_courtyard = 30) 
  (h₂ : width_courtyard = 18)
  (h₃ : num_paving_stones = 135)
  (h₄ : area_courtyard = length_courtyard * width_courtyard)
  (h₅ : area_courtyard = num_paving_stones * s * s) :
  s = 2 := 
by
  sorry

end paving_stone_size_l160_160202


namespace like_terms_exponents_l160_160982

theorem like_terms_exponents (m n : ℕ) (h₁ : m + 3 = 5) (h₂ : 6 = 2 * n) : m^n = 8 :=
by
  sorry

end like_terms_exponents_l160_160982


namespace repeating_decimal_as_fraction_l160_160047

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l160_160047


namespace balloons_lost_l160_160533

-- Definitions corresponding to the conditions
def initial_balloons : ℕ := 7
def current_balloons : ℕ := 4

-- The mathematically equivalent proof problem
theorem balloons_lost : initial_balloons - current_balloons = 3 := by
  -- proof steps would go here, but we use sorry to skip them 
  sorry

end balloons_lost_l160_160533


namespace product_of_real_solutions_triple_property_l160_160588

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l160_160588


namespace find_coordinates_of_Q_l160_160548

noncomputable def unit_circle_arc_length_to_angle (arc_length : ℝ) : ℝ :=
  -arc_length -- clockwise direction represented by negative angle

noncomputable def point_on_unit_circle_after_move (start : ℝ × ℝ) (arc_length : ℝ) : ℝ × ℝ :=
  let angle := unit_circle_arc_length_to_angle arc_length in
  (Real.cos angle, Real.sin angle)

theorem find_coordinates_of_Q :
  point_on_unit_circle_after_move (1, 0) (2 * Real.pi / 3) = (-1 / 2, -Real.sqrt 3 / 2) :=
by
  -- proof is omitted
  sorry

end find_coordinates_of_Q_l160_160548


namespace geometric_sum_ratio_l160_160686

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

theorem geometric_sum_ratio (a : ℕ → ℝ) (q : ℝ)
  (h₀ : is_geometric_sequence a q)
  (h₁ : 6 * a 7 = (a 8 + a 9) / 2) :
  sum_first_n_terms a 6 / sum_first_n_terms a 3 = 28 :=
by
  sorry

end geometric_sum_ratio_l160_160686


namespace value_of_8x_minus_5_squared_l160_160226

theorem value_of_8x_minus_5_squared (x : ℝ) (h : 8 * x ^ 2 + 7 = 12 * x + 17) : (8 * x - 5) ^ 2 = 465 := 
sorry

end value_of_8x_minus_5_squared_l160_160226


namespace repeating_decimal_equiv_fraction_l160_160089

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l160_160089


namespace count_possible_values_l160_160829

open Nat

def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

def is_valid_addition (A B C D : ℕ) : Prop :=
  ∀ x y z w v u : ℕ, 
  (x = A) ∧ (y = B) ∧ (z = C) ∧ (w = D) ∧ (v = B) ∧ (u = D) →
  (A + C = D) ∧ (A + D = B) ∧ (B + B = D) ∧ (D + D = C)

theorem count_possible_values : ∀ (A B C D : ℕ), 
  distinct_digits A B C D → is_valid_addition A B C D → num_of_possible_D = 4 :=
by
  intro A B C D hd hv
  sorry

end count_possible_values_l160_160829


namespace meet_starting_point_together_at_7_40_AM_l160_160478

-- Definitions of the input conditions
def Charlie_time : Nat := 5
def Alex_time : Nat := 8
def Taylor_time : Nat := 10

-- The combined time when they meet again at the starting point
def LCM_time (a b c : Nat) : Nat := Nat.lcm a (Nat.lcm b c)

-- Proving that the earliest time they all coincide again is 40 minutes after the start
theorem meet_starting_point_together_at_7_40_AM :
  LCM_time Charlie_time Alex_time Taylor_time = 40 := 
by
  unfold Charlie_time Alex_time Taylor_time LCM_time
  sorry

end meet_starting_point_together_at_7_40_AM_l160_160478


namespace find_number_l160_160179

theorem find_number (x : ℤ) (N : ℤ) (h1 : 3 * x = (N - x) + 18) (hx : x = 11) : N = 26 :=
by
  sorry

end find_number_l160_160179


namespace find_number_divided_by_3_equals_subtracted_5_l160_160891

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l160_160891


namespace Cheryl_more_eggs_than_others_l160_160771

-- Definitions based on conditions
def KevinEggs : Nat := 5
def BonnieEggs : Nat := 13
def GeorgeEggs : Nat := 9
def CherylEggs : Nat := 56

-- Main theorem statement
theorem Cheryl_more_eggs_than_others : (CherylEggs - (KevinEggs + BonnieEggs + GeorgeEggs) = 29) :=
by
  -- Proof would go here
  sorry

end Cheryl_more_eggs_than_others_l160_160771


namespace austin_more_apples_than_dallas_l160_160341

-- Conditions as definitions
def dallas_apples : ℕ := 14
def dallas_pears : ℕ := 9
def austin_pears : ℕ := dallas_pears - 5
def austin_total_fruit : ℕ := 24

-- The theorem statement
theorem austin_more_apples_than_dallas 
  (austin_apples : ℕ) (h1 : austin_apples + austin_pears = austin_total_fruit) :
  austin_apples - dallas_apples = 6 :=
sorry

end austin_more_apples_than_dallas_l160_160341


namespace repeating_decimal_as_fraction_l160_160048

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l160_160048


namespace initial_innings_count_l160_160554

theorem initial_innings_count (n T L : ℕ) 
  (h1 : T = 50 * n)
  (h2 : 174 = L + 172)
  (h3 : (T - 174 - L) = 48 * (n - 2)) :
  n = 40 :=
by 
  sorry

end initial_innings_count_l160_160554


namespace find_number_eq_seven_point_five_l160_160915

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l160_160915


namespace analysis_method_correct_answer_l160_160641

axiom analysis_def (conclusion: Prop): 
  ∃ sufficient_conditions: (Prop → Prop), 
    (∀ proof_conclusion: Prop, proof_conclusion = conclusion → sufficient_conditions proof_conclusion)

theorem analysis_method_correct_answer :
  ∀ (conclusion : Prop) , ∃ sufficient_conditions: (Prop → Prop), 
  (∀ proof_conclusion: Prop, proof_conclusion = conclusion → sufficient_conditions proof_conclusion)
:= by 
  intros 
  sorry

end analysis_method_correct_answer_l160_160641


namespace complex_number_quadrant_l160_160869

theorem complex_number_quadrant (a : ℝ) : 
  (a^2 - 2 = 3 * a - 4) ∧ (a^2 - 2 < 0 ∧ 3 * a - 4 < 0) → a = 1 :=
by
  sorry

end complex_number_quadrant_l160_160869


namespace find_m_l160_160993

theorem find_m (m : ℝ) (P : Set ℝ) (Q : Set ℝ) (hP : P = {m^2 - 4, m + 1, -3})
  (hQ : Q = {m - 3, 2 * m - 1, 3 * m + 1}) (h_intersect : P ∩ Q = {-3}) :
  m = -4 / 3 :=
by
  sorry

end find_m_l160_160993


namespace fraction_eq_repeating_decimal_l160_160042

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l160_160042


namespace equal_prob_of_first_ace_equal_prob_with_ace_of_spades_l160_160823

section
variable [decidable_eq ℕ]
variable (deck_size: ℕ := 32)
variable (num_aces: ℕ := 4)
variable (players: Π (i: fin 4), ℕ := λ i, 1)
variable [uniform_dist: Probability_Mass_Function (fin deck_size)] 

-- Part (a): Probabilities for each player to get the first Ace
noncomputable def player1_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player2_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player3_prob (d: ℕ → fin deck_size) : ℝ := sorry
noncomputable def player4_prob (d: ℕ → fin deck_size) : ℝ := sorry

-- Equivalent statement
theorem equal_prob_of_first_ace :
  player1_prob deck = 1/8 ∧
  player2_prob deck = 1/8 ∧
  player3_prob deck = 1/8 ∧
  player4_prob deck = 1/8 :=
sorry

-- Part (b): Modify rules to deal until Ace of Spades
noncomputable def player_prob_ace_of_spades (d: ℕ → fin deck_size) : ℝ := sorry

-- Equivalent statement
theorem equal_prob_with_ace_of_spades :
  ∀(p: fin 4), player_prob_ace_of_spades deck = 1/4 :=
sorry
end

end equal_prob_of_first_ace_equal_prob_with_ace_of_spades_l160_160823


namespace polynomial_evaluation_x_eq_4_l160_160026

theorem polynomial_evaluation_x_eq_4 : 
  (4 ^ 4 + 4 ^ 3 + 4 ^ 2 + 4 + 1 = 341) := 
by 
  sorry

end polynomial_evaluation_x_eq_4_l160_160026


namespace find_minuend_l160_160737

variable (x y : ℕ)

-- Conditions
axiom h1 : x - y = 8008
axiom h2 : x - 10 * y = 88

-- Theorem statement
theorem find_minuend : x = 8888 :=
by
  sorry

end find_minuend_l160_160737


namespace field_length_l160_160559

theorem field_length (w l: ℕ) (hw1: l = 2 * w) (hw2: 8 * 8 = 64) (hw3: 64 = l * w / 2) : l = 16 := 
by
  sorry

end field_length_l160_160559


namespace rational_function_solution_eq_l160_160661

theorem rational_function_solution_eq (f : ℚ → ℚ) (h₀ : f 0 = 0) 
  (h₁ : ∀ x y : ℚ, f (f x + f y) = x + y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := 
sorry

end rational_function_solution_eq_l160_160661


namespace repeating_decimal_equiv_fraction_l160_160093

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l160_160093


namespace sum_of_factors_of_30_is_72_l160_160611

-- Define the set of whole-number factors of 30
def factors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15, 30}

-- Sum of the factors of 30
def sum_factors_30 : ℕ := factors_30.sum id

-- Theorem stating the sum of the whole-number factors of 30 is 72 
theorem sum_of_factors_of_30_is_72 : sum_factors_30 = 72 := by
  sorry

end sum_of_factors_of_30_is_72_l160_160611


namespace janet_earns_more_as_freelancer_l160_160692

-- Definitions for the problem conditions
def current_job_weekly_hours : ℕ := 40
def current_job_hourly_rate : ℕ := 30

def freelance_client_a_hours_per_week : ℕ := 15
def freelance_client_a_hourly_rate : ℕ := 45

def freelance_client_b_hours_project1_per_week : ℕ := 5
def freelance_client_b_hours_project2_per_week : ℕ := 10
def freelance_client_b_hourly_rate : ℕ := 40

def freelance_client_c_hours_per_week : ℕ := 20
def freelance_client_c_rate_range : ℕ × ℕ := (35, 42)

def weekly_fica_taxes : ℕ := 25
def monthly_healthcare_premiums : ℕ := 400
def monthly_increased_rent : ℕ := 750
def monthly_business_phone_internet : ℕ := 150
def business_expense_percentage : ℕ := 10

def weeks_in_month : ℕ := 4

-- Define the calculations
def current_job_monthly_earnings := current_job_weekly_hours * current_job_hourly_rate * weeks_in_month

def freelance_client_a_weekly_earnings := freelance_client_a_hours_per_week * freelance_client_a_hourly_rate
def freelance_client_b_weekly_earnings := (freelance_client_b_hours_project1_per_week + freelance_client_b_hours_project2_per_week) * freelance_client_b_hourly_rate
def freelance_client_c_weekly_earnings := freelance_client_c_hours_per_week * ((freelance_client_c_rate_range.1 + freelance_client_c_rate_range.2) / 2)

def total_freelance_weekly_earnings := freelance_client_a_weekly_earnings + freelance_client_b_weekly_earnings + freelance_client_c_weekly_earnings
def total_freelance_monthly_earnings := total_freelance_weekly_earnings * weeks_in_month

def total_additional_expenses := (weekly_fica_taxes * weeks_in_month) + monthly_healthcare_premiums + monthly_increased_rent + monthly_business_phone_internet

def business_expense_deduction := (total_freelance_monthly_earnings * business_expense_percentage) / 100
def adjusted_freelance_earnings_after_deduction := total_freelance_monthly_earnings - business_expense_deduction
def adjusted_freelance_earnings_after_expenses := adjusted_freelance_earnings_after_deduction - total_additional_expenses

def earnings_difference := adjusted_freelance_earnings_after_expenses - current_job_monthly_earnings

-- The theorem to be proved
theorem janet_earns_more_as_freelancer :
  earnings_difference = 1162 :=
sorry

end janet_earns_more_as_freelancer_l160_160692


namespace prime_arithmetic_sequence_l160_160291

theorem prime_arithmetic_sequence {p1 p2 p3 d : ℕ} 
  (hp1 : Nat.Prime p1) 
  (hp2 : Nat.Prime p2) 
  (hp3 : Nat.Prime p3)
  (h3_p1 : 3 < p1)
  (h3_p2 : 3 < p2)
  (h3_p3 : 3 < p3)
  (h_seq1 : p2 = p1 + d)
  (h_seq2 : p3 = p1 + 2 * d) : 
  d % 6 = 0 :=
by sorry

end prime_arithmetic_sequence_l160_160291


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l160_160604

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l160_160604


namespace line_passes_vertex_parabola_l160_160497

theorem line_passes_vertex_parabola :
  ∃ (b₁ b₂ : ℚ), (b₁ ≠ b₂) ∧ (∀ b, (b = b₁ ∨ b = b₂) → 
    (∃ x y, y = x + b ∧ y = x^2 + 4 * b^2 ∧ x = 0 ∧ y = 4 * b^2)) :=
by 
  sorry

end line_passes_vertex_parabola_l160_160497


namespace kanul_spent_on_machinery_l160_160274

theorem kanul_spent_on_machinery (total raw_materials cash M : ℝ) 
  (h_total : total = 7428.57) 
  (h_raw_materials : raw_materials = 5000) 
  (h_cash : cash = 0.30 * total) 
  (h_expenditure : total = raw_materials + M + cash) :
  M = 200 := 
by
  sorry

end kanul_spent_on_machinery_l160_160274


namespace divide_triangle_into_equal_areas_l160_160215

-- Given conditions
variables (A B C : Point)
variables (AC BC : ℝ)
variable (h : AC ≥ BC)

-- Define that there exists a line parallel to the internal bisector of ∠ C which divides the area evenly
theorem divide_triangle_into_equal_areas 
  (triangle : Triangle A B C)
  (f_c : Line) 
  (hf : is_angle_bisector f_c (angle A C B))
  : ∃ e : Line, is_parallel_to e f_c ∧ divides_triangle_into_equal_areas e triangle :=
sorry

end divide_triangle_into_equal_areas_l160_160215


namespace car_speed_second_hour_l160_160432

theorem car_speed_second_hour
  (S : ℕ)
  (first_hour_speed : ℕ := 98)
  (avg_speed : ℕ := 79)
  (total_time : ℕ := 2)
  (h_avg_speed : avg_speed = (first_hour_speed + S) / total_time) :
  S = 60 :=
by
  -- Proof steps omitted
  sorry

end car_speed_second_hour_l160_160432


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l160_160605

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l160_160605


namespace sally_balloons_l160_160412

theorem sally_balloons :
  (initial_orange_balloons : ℕ) → (lost_orange_balloons : ℕ) → 
  (remaining_orange_balloons : ℕ) → (doubled_orange_balloons : ℕ) → 
  initial_orange_balloons = 20 → 
  lost_orange_balloons = 5 →
  remaining_orange_balloons = initial_orange_balloons - lost_orange_balloons →
  doubled_orange_balloons = 2 * remaining_orange_balloons → 
  doubled_orange_balloons = 30 :=
by
  intro initial_orange_balloons lost_orange_balloons 
       remaining_orange_balloons doubled_orange_balloons
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h3] at h4
  sorry

end sally_balloons_l160_160412


namespace initial_number_of_persons_l160_160152

theorem initial_number_of_persons (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) (weight_diff : ℝ)
  (h1 : avg_increase = 2.5) 
  (h2 : old_weight = 75) 
  (h3 : new_weight = 95)
  (h4 : weight_diff = new_weight - old_weight)
  (h5 : weight_diff = avg_increase * n) : n = 8 := 
sorry

end initial_number_of_persons_l160_160152


namespace find_number_l160_160883

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l160_160883


namespace intersection_points_count_l160_160720

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2 ∧ (∀ x, f x = g x → x = x1 ∨ x = x2) :=
by
  sorry

end intersection_points_count_l160_160720


namespace remainder_of_product_l160_160609

open Nat

theorem remainder_of_product (a b : ℕ) (ha : a % 5 = 4) (hb : b % 5 = 3) :
  (a * b) % 5 = 2 :=
by
  sorry

end remainder_of_product_l160_160609


namespace middle_integer_is_zero_l160_160349

-- Mathematical equivalent proof problem in Lean 4

theorem middle_integer_is_zero
  (n : ℤ)
  (h : (n - 2) + n + (n + 2) = (1 / 5) * ((n - 2) * n * (n + 2))) :
  n = 0 :=
by
  sorry

end middle_integer_is_zero_l160_160349


namespace candy_problem_l160_160227

theorem candy_problem
  (G : Nat := 7) -- Gwen got 7 pounds of candy
  (C : Nat := 17) -- Combined weight of candy
  (F : Nat) -- Pounds of candy Frank got
  (h : F + G = C) -- Condition: Combined weight
  : F = 10 := 
by
  sorry

end candy_problem_l160_160227


namespace product_of_reals_tripled_when_added_to_reciprocal_l160_160586

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l160_160586


namespace cone_volume_correct_l160_160569

noncomputable def cone_volume (slant_height height : ℝ) : ℝ :=
  let r := real.sqrt (slant_height^2 - height^2)
  in (1/3) * real.pi * r^2 * height

theorem cone_volume_correct :
  cone_volume 15 9 = 432 * real.pi := by
  sorry

end cone_volume_correct_l160_160569


namespace probability_five_chords_form_convex_pentagon_l160_160707

noncomputable def probability_convex_pentagon (n k : ℕ) : ℚ :=
(combinatorics.choose n k) / (combinatorics.choose (combinatorics.choose 7 2) 5)

theorem probability_five_chords_form_convex_pentagon :
  probability_convex_pentagon 7 5 = 1 / 969 := sorry

end probability_five_chords_form_convex_pentagon_l160_160707


namespace inequality_proof_l160_160986

variable {x1 x2 y1 y2 z1 z2 : ℝ}

theorem inequality_proof (hx1 : x1 > 0) (hx2 : x2 > 0)
   (hxy1 : x1 * y1 - z1^2 > 0) (hxy2 : x2 * y2 - z2^2 > 0) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2)^2) ≤ 1 / (x1 * y1 - z1^2) + 1 / (x2 * y2 - z2^2) :=
  sorry

end inequality_proof_l160_160986


namespace average_remaining_numbers_l160_160711

theorem average_remaining_numbers (numbers : List ℝ) 
  (h_len : numbers.length = 50) 
  (h_avg : (numbers.sum / 50 : ℝ) = 38) 
  (h_discard : 45 ∈ numbers ∧ 55 ∈ numbers) :
  let new_sum := numbers.sum - 45 - 55
  let new_len := 50 - 2
  (new_sum / new_len : ℝ) = 37.5 :=
by
  sorry

end average_remaining_numbers_l160_160711


namespace product_of_solutions_l160_160597

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l160_160597


namespace union_of_A_and_B_l160_160106

namespace SetProof

def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x > 0}
def expectedUnion : Set ℝ := {x | -2 ≤ x}

theorem union_of_A_and_B : (A ∪ B) = expectedUnion := by
  sorry

end SetProof

end union_of_A_and_B_l160_160106


namespace find_x_l160_160977

theorem find_x (x : ℝ) : 
  3.5 * ( (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * x) ) = 2800.0000000000005 → x = 1.25 :=
by
  sorry

end find_x_l160_160977


namespace sqrt_of_square_neg_three_l160_160191

theorem sqrt_of_square_neg_three : Real.sqrt ((-3 : ℝ)^2) = 3 := by
  sorry

end sqrt_of_square_neg_three_l160_160191


namespace geometric_sequence_150th_term_l160_160688

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem geometric_sequence_150th_term :
  geometric_sequence 8 (-1 / 2) 150 = -8 * (1 / 2) ^ 149 :=
by
  -- This is the proof placeholder
  sorry

end geometric_sequence_150th_term_l160_160688


namespace cos_of_sin_given_l160_160523

theorem cos_of_sin_given (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by
  sorry

end cos_of_sin_given_l160_160523


namespace reciprocal_relationship_l160_160367

theorem reciprocal_relationship (a b : ℚ)
  (h1 : a = (-7 / 8) / (7 / 4 - 7 / 8 - 7 / 12))
  (h2 : b = (7 / 4 - 7 / 8 - 7 / 12) / (-7 / 8)) :
  a = - 1 / b :=
by sorry

end reciprocal_relationship_l160_160367


namespace sqrt_expression_l160_160475

theorem sqrt_expression :
  Real.sqrt 18 - 3 * Real.sqrt (1 / 2) + Real.sqrt 2 = (5 * Real.sqrt 2) / 2 :=
by
  sorry

end sqrt_expression_l160_160475


namespace original_number_of_men_l160_160935

theorem original_number_of_men 
  (x : ℕ)
  (H : 15 * 18 * x = 15 * 18 * (x - 8) + 8 * 15 * 18)
  (h_pos : x > 8) :
  x = 40 :=
sorry

end original_number_of_men_l160_160935


namespace Cheryl_more_eggs_than_others_l160_160770

-- Definitions based on conditions
def KevinEggs : Nat := 5
def BonnieEggs : Nat := 13
def GeorgeEggs : Nat := 9
def CherylEggs : Nat := 56

-- Main theorem statement
theorem Cheryl_more_eggs_than_others : (CherylEggs - (KevinEggs + BonnieEggs + GeorgeEggs) = 29) :=
by
  -- Proof would go here
  sorry

end Cheryl_more_eggs_than_others_l160_160770


namespace judgement_only_b_correct_l160_160555

theorem judgement_only_b_correct
  (A_expr : Int := 11 + (-14) + 19 - (-6))
  (A_computed : Int := 11 + 19 + ((-14) + (-6)))
  (A_result_incorrect : A_computed ≠ 10)
  (B_expr : ℚ := -2/3 - 1/5 + (-1/3))
  (B_computed : ℚ := (-2/3 + -1/3) + -1/5)
  (B_result_correct : B_computed = -6/5) :
  (A_computed ≠ 10 ∧ B_computed = -6/5) :=
by
  sorry

end judgement_only_b_correct_l160_160555


namespace main_theorem_l160_160799

noncomputable def proof_problem (α : ℝ) (h1 : 0 < α) (h2 : α < π) : Prop :=
  2 * Real.sin (2 * α) ≤ Real.cos (α / 2)

noncomputable def equality_case (α : ℝ) (h1 : 0 < α) (h2 : α < π) : Prop :=
  α = π / 3 → 2 * Real.sin (2 * α) = Real.cos (α / 2)

theorem main_theorem (α : ℝ) (h1 : 0 < α) (h2 : α < π) :
  proof_problem α h1 h2 ∧ equality_case α h1 h2 :=
by
  sorry

end main_theorem_l160_160799


namespace bus_driver_total_hours_l160_160745

def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
def total_compensation : ℝ := 976
def max_regular_hours : ℝ := 40

theorem bus_driver_total_hours :
  ∃ (hours_worked : ℝ), 
  (hours_worked = max_regular_hours + (total_compensation - (regular_rate * max_regular_hours)) / overtime_rate) ∧
  hours_worked = 52 :=
by
  sorry

end bus_driver_total_hours_l160_160745


namespace Steve_has_more_money_than_Wayne_by_2004_l160_160471

theorem Steve_has_more_money_than_Wayne_by_2004:
  (∀ n: ℕ, steve_money n = 100 * 2 ^ n ∧ wayne_money n = 10000 / 2 ^ n) →
  (∃ n: ℕ, 2000 <= n ∧ 2000 + n = 2004 ∧ steve_money (n + 2000) > wayne_money (n + 2000)) :=
by
  intro h
  sorry

end Steve_has_more_money_than_Wayne_by_2004_l160_160471


namespace find_number_eq_seven_point_five_l160_160911

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l160_160911


namespace required_speed_l160_160951

theorem required_speed
  (D T : ℝ) (h1 : 30 = D / T) 
  (h2 : 2 * D / 3 = 30 * (T / 3)) :
  (D / 3) / (2 * T / 3) = 15 :=
by
  sorry

end required_speed_l160_160951


namespace k_not_possible_l160_160806

theorem k_not_possible (S : ℕ → ℚ) (a b : ℕ → ℚ) (n k : ℕ) (k_gt_2 : k > 2) :
  (S n = (n^2 + n) / 2) →
  (a n = S n - S (n - 1)) →
  (b n = 1 / a n) →
  (2 * b (n + 2) = b n + b (n + k)) →
  k ≠ 4 ∧ k ≠ 10 :=
by
  -- Proof goes here (skipped)
  sorry

end k_not_possible_l160_160806


namespace find_value_of_expression_l160_160542

noncomputable def p : ℝ := 3
noncomputable def q : ℝ := 7
noncomputable def r : ℝ := 5

def inequality_holds (f : ℝ → ℝ) : Prop :=
  ∀ x, (f x ≥ 0 ↔ (x ∈ Set.Icc 3 7 ∨ x > 5))

def given_condition : Prop := p < q

theorem find_value_of_expression (f : ℝ → ℝ)
  (h : inequality_holds f)
  (hc : given_condition) :
  p + 2*q + 3*r = 32 := 
sorry

end find_value_of_expression_l160_160542


namespace one_sixths_in_fraction_l160_160116

theorem one_sixths_in_fraction :
  (11 / 3) / (1 / 6) = 22 :=
sorry

end one_sixths_in_fraction_l160_160116


namespace find_even_integer_l160_160496

def is_even (n : ℤ) : Prop := n % 2 = 0

def h (n : ℤ) : ℤ :=
  if is_even n then
    ∑ k in Finset.range ((n / 2) + 1), 2 * k
  else
    0

theorem find_even_integer (n : ℤ) (h₁ : is_even n) (h₂ : h(18) = 90) (h₃ : h(18) / h(n) = 3) : n = 10 :=
by sorry

end find_even_integer_l160_160496


namespace triangle_perimeter_l160_160434

theorem triangle_perimeter (L R B : ℕ) (hL : L = 12) (hR : R = L + 2) (hB : B = 24) : L + R + B = 50 :=
by
  -- proof steps go here
  sorry

end triangle_perimeter_l160_160434


namespace unique_odd_number_between_500_and_1000_l160_160938

theorem unique_odd_number_between_500_and_1000 :
  ∃! x : ℤ, 500 ≤ x ∧ x ≤ 1000 ∧ x % 25 = 6 ∧ x % 9 = 7 ∧ x % 2 = 1 :=
sorry

end unique_odd_number_between_500_and_1000_l160_160938


namespace area_to_be_painted_l160_160851

variable (h_wall : ℕ) (l_wall : ℕ)
variable (h_window : ℕ) (l_window : ℕ)
variable (h_door : ℕ) (l_door : ℕ)

theorem area_to_be_painted :
  ∀ (h_wall : ℕ) (l_wall : ℕ) (h_window : ℕ) (l_window : ℕ) (h_door : ℕ) (l_door : ℕ),
  h_wall = 10 → l_wall = 15 →
  h_window = 3 → l_window = 5 →
  h_door = 2 → l_door = 3 →
  (h_wall * l_wall) - ((h_window * l_window) + (h_door * l_door)) = 129 :=
by
  intros
  sorry

end area_to_be_painted_l160_160851


namespace initial_animal_types_l160_160339

theorem initial_animal_types (x : ℕ) (h1 : 6 * (x + 4) = 54) : x = 5 := 
sorry

end initial_animal_types_l160_160339


namespace sum_A_k_div_k_l160_160665

noncomputable def A (k : ℕ) : ℕ :=
  (Finset.filter (fun d => d % 2 = 1 ∧ d ≤ Nat.sqrt (2 * k - 1)) (Finset.range k)).card

noncomputable def sumExpression : ℝ :=
  ∑' k, (-1)^(k-1) * (A k / k : ℝ)

theorem sum_A_k_div_k : sumExpression = Real.pi^2 / 8 :=
  sorry

end sum_A_k_div_k_l160_160665


namespace cannot_form_1x1x2_blocks_l160_160752

theorem cannot_form_1x1x2_blocks :
  let edge_length := 7
  let total_cubes := edge_length * edge_length * edge_length
  let central_cube := (3, 3, 3)
  let remaining_cubes := total_cubes - 1
  let checkerboard_color (x y z : Nat) : Bool := (x + y + z) % 2 = 0
  let num_white (k : Nat) := if k % 2 = 0 then 25 else 24
  let num_black (k : Nat) := if k % 2 = 0 then 24 else 25
  let total_white := 170
  let total_black := 171
  total_black > total_white →
  ¬(remaining_cubes % 2 = 0 ∧ total_white % 2 = 0 ∧ total_black % 2 = 0) → 
  ∀ (block: Nat × Nat × Nat → Bool) (x y z : Nat), block (x, y, z) = ((x*y*z) % 2 = 0) := sorry

end cannot_form_1x1x2_blocks_l160_160752


namespace polynomial_evaluation_x_eq_4_l160_160027

theorem polynomial_evaluation_x_eq_4 : 
  (4 ^ 4 + 4 ^ 3 + 4 ^ 2 + 4 + 1 = 341) := 
by 
  sorry

end polynomial_evaluation_x_eq_4_l160_160027


namespace simplify_and_evaluate_l160_160149

theorem simplify_and_evaluate (m : ℝ) (h : m = 1) : 
  (1 - 1 / (m - 2)) / ((m^2 - 6 * m + 9) / (m - 2)) = -1/2 :=
by
  sorry

end simplify_and_evaluate_l160_160149


namespace tangent_curve_line_a_eq_neg1_l160_160369

theorem tangent_curve_line_a_eq_neg1 (a : ℝ) (x : ℝ) : 
  (∀ (x : ℝ), (e^x + a = x) ∧ (e^x = 1) ) → a = -1 :=
by 
  intro h
  sorry

end tangent_curve_line_a_eq_neg1_l160_160369


namespace sonny_received_45_boxes_l160_160414

def cookies_received (cookies_given_brother : ℕ) (cookies_given_sister : ℕ) (cookies_given_cousin : ℕ) (cookies_left : ℕ) : ℕ :=
  cookies_given_brother + cookies_given_sister + cookies_given_cousin + cookies_left

theorem sonny_received_45_boxes :
  cookies_received 12 9 7 17 = 45 :=
by
  sorry

end sonny_received_45_boxes_l160_160414


namespace sum_of_interior_edges_l160_160632

theorem sum_of_interior_edges {f : ℝ} {w : ℝ} (h_frame_area : f = 32) (h_outer_edge : w = 7) (h_frame_width : 2) :
  let i_length := w - 2 * h_frame_width in
  let i_other_length := (f - (w * (w  - 2 * h_frame_width))) / (w  + 2 * h_frame_width) in
  i_length + i_other_length + i_length + i_other_length = 8 :=
by
  let i_length := w - 2 * 2
  let i_other_length := (32 - (i_length * w)) / (w  + 2 * 2)
  let sum := i_length + i_other_length + i_length + i_other_length
  have h_sum : sum = 8, by sorry
  exact h_sum

end sum_of_interior_edges_l160_160632


namespace fraction_sum_l160_160141

namespace GeometricSequence

-- Given conditions in the problem
def q : ℕ := 2

-- Definition of the sum of the first n terms (S_n) of a geometric sequence
def S_n (a₁ : ℤ) (n : ℕ) : ℤ := 
  a₁ * (1 - q ^ n) / (1 - q)

-- Specific sum for the first 4 terms (S₄)
def S₄ (a₁ : ℤ) : ℤ := S_n a₁ 4

-- Define the 2nd term of the geometric sequence
def a₂ (a₁ : ℤ) : ℤ := a₁ * q

-- The statement to prove: $\dfrac{S_4}{a_2} = \dfrac{15}{2}$
theorem fraction_sum (a₁ : ℤ) : (S₄ a₁) / (a₂ a₁) = Rat.ofInt 15 / Rat.ofInt 2 :=
  by
  -- Implementation of proof will go here
  sorry

end GeometricSequence

end fraction_sum_l160_160141


namespace cone_volume_l160_160565

theorem cone_volume (l : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) 
  (hl : l = 15)    -- slant height
  (hh : h = 9)     -- vertical height
  (hr : r^2 = 144) -- radius squared from Pythagorean theorem
  : V = 432 * Real.pi :=
by
  -- Proof is omitted. Hence, we write sorry to denote skipped proof.
  sorry

end cone_volume_l160_160565


namespace valid_operation_l160_160617

theorem valid_operation :
  ∀ x : ℝ, x^2 + x^3 ≠ x^5 ∧
  ∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2 ∧
  ∀ m : ℝ, (|m| = m ↔ m ≥ 0) :=
by
  sorry

end valid_operation_l160_160617


namespace recurring_decimal_to_fraction_l160_160079

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l160_160079


namespace symmetric_coords_l160_160557

-- Define the initial point and the line equation
def initial_point : ℝ × ℝ := (-1, 1)
def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

-- Define what it means for one point to be symmetric to another point with respect to a line
def symmetric_point (p q : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), line_eq m p.1 ∧ line_eq m q.1 ∧ 
             p.1 + q.1 = 2 * m ∧
             p.2 + q.2 = 2 * m

-- The theorem we want to prove
theorem symmetric_coords : ∃ (symmetric : ℝ × ℝ), symmetric_point initial_point symmetric ∧ symmetric = (2, -2) :=
sorry

end symmetric_coords_l160_160557


namespace probability_white_ball_second_draw_l160_160372

noncomputable def probability_white_given_red (red_white_yellow_balls : Nat × Nat × Nat) : ℚ :=
  let (r, w, y) := red_white_yellow_balls
  let total_balls := r + w + y
  let p_A := (r : ℚ) / total_balls
  let p_AB := (r : ℚ) / total_balls * (w : ℚ) / (total_balls - 1)
  p_AB / p_A

theorem probability_white_ball_second_draw (r w y : Nat) (h_r : r = 2) (h_w : w = 3) (h_y : y = 1) :
  probability_white_given_red (r, w, y) = 3 / 5 :=
by
  rw [h_r, h_w, h_y]
  unfold probability_white_given_red
  simp
  sorry

end probability_white_ball_second_draw_l160_160372


namespace stars_per_classmate_is_correct_l160_160294

-- Define the given conditions
def total_stars : ℕ := 45
def num_classmates : ℕ := 9

-- Define the expected number of stars per classmate
def stars_per_classmate : ℕ := 5

-- Prove that the number of stars per classmate is 5 given the conditions
theorem stars_per_classmate_is_correct :
  total_stars / num_classmates = stars_per_classmate :=
sorry

end stars_per_classmate_is_correct_l160_160294


namespace solve_quadratic_equation_l160_160786

noncomputable def f (x : ℝ) := 
  5 / (Real.sqrt (x - 9) - 8) - 
  2 / (Real.sqrt (x - 9) - 5) + 
  6 / (Real.sqrt (x - 9) + 5) - 
  9 / (Real.sqrt (x - 9) + 8)

theorem solve_quadratic_equation :
  ∀ (x : ℝ), x ≥ 9 → f x = 0 → 
  x = 19.2917 ∨ x = 8.9167 :=
by sorry

end solve_quadratic_equation_l160_160786


namespace minor_premise_of_syllogism_l160_160193

theorem minor_premise_of_syllogism (P Q : Prop)
  (h1 : ¬ (P ∧ ¬ Q))
  (h2 : Q) :
  Q :=
by
  sorry

end minor_premise_of_syllogism_l160_160193


namespace initial_amount_A_correct_l160_160465

noncomputable def initial_amount_A :=
  let a := 21
  let b := 5
  let c := 9

  -- After A gives B and C
  let b_after_A := b + 5
  let c_after_A := c + 9
  let a_after_A := a - (5 + 9)

  -- After B gives A and C
  let a_after_B := a_after_A + (a_after_A / 2)
  let c_after_B := c_after_A + (c_after_A / 2)
  let b_after_B := b_after_A - (a_after_A / 2 + c_after_A / 2)

  -- After C gives A and B
  let a_final := a_after_B + 3 * a_after_B
  let b_final := b_after_B + 3 * b_after_B
  let c_final := c_after_B - (3 * a_final + b_final)

  (a_final = 24) ∧ (b_final = 16) ∧ (c_final = 8)

theorem initial_amount_A_correct : initial_amount_A := 
by
  -- Skipping proof details
  sorry

end initial_amount_A_correct_l160_160465


namespace ineq_x4_y4_l160_160849

theorem ineq_x4_y4 (x y : ℝ) (h1 : x > Real.sqrt 2) (h2 : y > Real.sqrt 2) :
    x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by
  sorry

end ineq_x4_y4_l160_160849


namespace probability_scrapped_l160_160747

variable (P_A P_B_given_not_A : ℝ)
variable (prob_scrapped : ℝ)

def fail_first_inspection (P_A : ℝ) := 1 - P_A
def fail_second_inspection_given_fails_first (P_B_given_not_A : ℝ) := 1 - P_B_given_not_A

theorem probability_scrapped (h1 : P_A = 0.8) (h2 : P_B_given_not_A = 0.9) (h3 : prob_scrapped = fail_first_inspection P_A * fail_second_inspection_given_fails_first P_B_given_not_A) :
  prob_scrapped = 0.02 := by
  sorry

end probability_scrapped_l160_160747


namespace find_speed_of_stream_l160_160000

-- Define the given conditions
def boat_speed_still_water : ℝ := 14
def distance_downstream : ℝ := 72
def time_downstream : ℝ := 3.6

-- Define the speed of the stream (to be proven)
def speed_of_stream : ℝ := 6

-- The statement of the problem
theorem find_speed_of_stream 
  (h1 : boat_speed_still_water = 14)
  (h2 : distance_downstream = 72)
  (h3 : time_downstream = 3.6)
  (speed_of_stream_eq : boat_speed_still_water + speed_of_stream = distance_downstream / time_downstream) :
  speed_of_stream = 6 := 
by 
  sorry

end find_speed_of_stream_l160_160000


namespace statement_l160_160280

variable {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Condition 2: f(x-2) = -f(x) for all x
def satisfies_periodicity (f : ℝ → ℝ) : Prop := ∀ x, f (x - 2) = -f x

-- Condition 3: f is decreasing on [0, 2]
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- The proof statement
theorem statement (h1 : is_odd_function f) (h2 : satisfies_periodicity f) (h3 : is_decreasing_on f 0 2) :
  f 5 < f 4 ∧ f 4 < f 3 :=
sorry

end statement_l160_160280


namespace sqrt_condition_sqrt_not_meaningful_2_l160_160820

theorem sqrt_condition (x : ℝ) : 1 - x ≥ 0 ↔ x ≤ 1 := 
by
  sorry

theorem sqrt_not_meaningful_2 : ¬(1 - 2 ≥ 0) :=
by
  sorry

end sqrt_condition_sqrt_not_meaningful_2_l160_160820


namespace yellow_green_block_weight_difference_l160_160134

theorem yellow_green_block_weight_difference :
  let yellow_weight := 0.6
  let green_weight := 0.4
  yellow_weight - green_weight = 0.2 := by
  sorry

end yellow_green_block_weight_difference_l160_160134


namespace range_of_m_l160_160300

theorem range_of_m :
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → (x < -1 ∨ x > 3)) ↔ (0 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l160_160300


namespace polynomial_expansion_l160_160030

theorem polynomial_expansion :
  (7 * X^2 + 5 * X - 3) * (3 * X^3 + 2 * X^2 + 1) = 
  21 * X^5 + 29 * X^4 + X^3 + X^2 + 5 * X - 3 :=
sorry

end polynomial_expansion_l160_160030


namespace product_of_solutions_l160_160589

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l160_160589


namespace real_solution_exists_l160_160969

theorem real_solution_exists (x : ℝ) (h1: x ≠ 5) (h2: x ≠ 6) : 
  (x = 1 ∨ x = 2 ∨ x = 3) ↔ 
  ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1) /
  ((x - 5) * (x - 6) * (x - 5)) = 1) := 
by 
  sorry

end real_solution_exists_l160_160969


namespace square_area_l160_160333

theorem square_area :
  ∀ (x1 x2 : ℝ), (x1^2 + 2 * x1 + 1 = 8) ∧ (x2^2 + 2 * x2 + 1 = 8) ∧ (x1 ≠ x2) →
  (abs (x1 - x2))^2 = 36 :=
by
  sorry

end square_area_l160_160333


namespace product_of_tripled_reciprocals_eq_neg_half_l160_160607

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l160_160607


namespace fraction_for_repeating_56_l160_160039

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l160_160039


namespace tan_value_l160_160671

theorem tan_value (θ : ℝ) (h : Real.sin (12 * Real.pi / 5 + θ) + 2 * Real.sin (11 * Real.pi / 10 - θ) = 0) :
  Real.tan (2 * Real.pi / 5 + θ) = 2 :=
by
  sorry

end tan_value_l160_160671


namespace calculate_difference_l160_160466

variable (σ : ℝ) -- Let \square be represented by a real number σ
def correct_answer := 4 * (σ - 3)
def incorrect_answer := 4 * σ - 3
def difference := correct_answer σ - incorrect_answer σ

theorem calculate_difference : difference σ = -9 := by
  sorry

end calculate_difference_l160_160466


namespace part1_inequality_solution_l160_160981

def f (x : ℝ) : ℝ := |x + 1| + |2 * x - 3|

theorem part1_inequality_solution :
  ∀ x : ℝ, f x ≤ 6 ↔ -4 / 3 ≤ x ∧ x ≤ 8 / 3 :=
by sorry

end part1_inequality_solution_l160_160981


namespace combinations_of_three_toppings_l160_160416

def number_of_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem combinations_of_three_toppings : number_of_combinations 10 3 = 120 := by
  sorry

end combinations_of_three_toppings_l160_160416


namespace solution_set_of_inequality_l160_160166

theorem solution_set_of_inequality (x : ℝ) : (x^2 + 4*x - 5 < 0) ↔ (-5 < x ∧ x < 1) :=
by sorry

end solution_set_of_inequality_l160_160166


namespace shift_gives_f_l160_160310

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem shift_gives_f :
  (∀ x, f x = g (x + Real.pi / 3)) :=
  by
  sorry

end shift_gives_f_l160_160310


namespace pens_bought_l160_160701

-- Define the given conditions
def num_notebooks : ℕ := 10
def cost_per_pen : ℕ := 2
def total_paid : ℕ := 30
def cost_per_notebook : ℕ := 0  -- Assumption that notebooks are free

-- Converted condition that 10N + 2P = 30 and N = 0
def equation (N P : ℕ) : Prop := (10 * N + 2 * P = total_paid)

-- Statement to prove that if notebooks are free, 15 pens were bought
theorem pens_bought (N : ℕ) (P : ℕ) (hN : N = cost_per_notebook) (h : equation N P) : P = 15 :=
by sorry

end pens_bought_l160_160701


namespace exists_k_ge_2_l160_160277

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def weak (a b n : ℕ) : Prop :=
  ¬ ∃ x y : ℕ, a * x + b * y = n

theorem exists_k_ge_2 (a b n : ℕ) (h_coprime : coprime a b) (h_positive : 0 < n) (h_weak : weak a b n) (h_bound : n < a * b / 6) :
  ∃ k : ℕ, 2 ≤ k ∧ weak a b (k * n) :=
sorry

end exists_k_ge_2_l160_160277


namespace number_of_solutions_eq_one_l160_160974

theorem number_of_solutions_eq_one :
  ∃! (n : ℕ), 0 < n ∧ 
              (∃ k : ℕ, (n + 1500) = 90 * k ∧ k = Int.floor (Real.sqrt n)) :=
sorry

end number_of_solutions_eq_one_l160_160974


namespace find_number_divided_by_3_equals_subtracted_5_l160_160896

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l160_160896


namespace total_revenue_correct_l160_160401

def items : Type := ℕ × ℝ

def magazines : items := (425, 2.50)
def newspapers : items := (275, 1.50)
def books : items := (150, 5.00)
def pamphlets : items := (75, 0.50)

def revenue (item : items) : ℝ := item.1 * item.2

def total_revenue : ℝ :=
  revenue magazines +
  revenue newspapers +
  revenue books +
  revenue pamphlets

theorem total_revenue_correct : total_revenue = 2262.50 := by
  sorry

end total_revenue_correct_l160_160401


namespace team_savings_with_discount_l160_160486

def regular_shirt_cost : ℝ := 7.50
def regular_pants_cost : ℝ := 15.00
def regular_socks_cost : ℝ := 4.50
def discounted_shirt_cost : ℝ := 6.75
def discounted_pants_cost : ℝ := 13.50
def discounted_socks_cost : ℝ := 3.75
def team_size : ℕ := 12

theorem team_savings_with_discount :
  let regular_uniform_cost := regular_shirt_cost + regular_pants_cost + regular_socks_cost
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  total_savings = 36 := by
  sorry

end team_savings_with_discount_l160_160486


namespace passengers_on_plane_l160_160766

variables (P : ℕ) (fuel_per_mile : ℕ := 20) (fuel_per_person : ℕ := 3) (fuel_per_bag : ℕ := 2)
variables (num_crew : ℕ := 5) (bags_per_person : ℕ := 2) (trip_distance : ℕ := 400)
variables (total_fuel : ℕ := 106000)

def total_people := P + num_crew
def total_bags := bags_per_person * total_people
def total_fuel_per_mile := fuel_per_mile + fuel_per_person * P + fuel_per_bag * total_bags
def total_trip_fuel := trip_distance * total_fuel_per_mile

theorem passengers_on_plane : total_trip_fuel = total_fuel → P = 33 := 
by
  sorry

end passengers_on_plane_l160_160766


namespace time_to_produce_one_item_l160_160009

-- Definitions based on the conditions
def itemsProduced : Nat := 300
def totalTimeHours : ℝ := 2.0
def minutesPerHour : ℝ := 60.0

-- The statement we need to prove
theorem time_to_produce_one_item : (totalTimeHours / itemsProduced * minutesPerHour) = 0.4 := by
  sorry

end time_to_produce_one_item_l160_160009


namespace max_mondays_in_59_days_l160_160175

theorem max_mondays_in_59_days (start_day : ℕ) : ∃ d : ℕ, d ≤ 6 ∧ 
  start_day = d → (d = 0 → ∃ m : ℕ, m = 9) :=
by 
  sorry

end max_mondays_in_59_days_l160_160175


namespace triangle_area_l160_160207

/-
A triangle with side lengths in the ratio 4:5:6 is inscribed in a circle of radius 5.
We need to prove that the area of the triangle is 250/9.
-/

theorem triangle_area (x : ℝ) (r : ℝ) (h_r : r = 5) (h_ratio : 6 * x = 2 * r) :
  (1 / 2) * (4 * x) * (5 * x) = 250 / 9 := by 
  -- Proof goes here.
  sorry

end triangle_area_l160_160207


namespace sin_150_eq_half_l160_160964

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_150_eq_half_l160_160964


namespace find_g_inv_l160_160818

noncomputable def g (x : ℝ) : ℝ :=
  (x^7 - 1) / 4

noncomputable def g_inv_value : ℝ :=
  (51 / 32)^(1/7)

theorem find_g_inv (h : g (g_inv_value) = 19 / 128) : g_inv_value = (51 / 32)^(1/7) :=
by
  sorry

end find_g_inv_l160_160818


namespace exponential_expression_is_rational_l160_160136

noncomputable def B (n : ℕ) : ℕ :=
nat.popcount n

noncomputable def S : ℝ :=
∑' n, (B n : ℝ) / (n * (n + 1))

theorem exponential_expression_is_rational :
  Real.exp S = 4 := sorry

end exponential_expression_is_rational_l160_160136


namespace angle_conversion_l160_160024

theorem angle_conversion :
  (12 * (Real.pi / 180)) = (Real.pi / 15) := by
  sorry

end angle_conversion_l160_160024


namespace remainder_seven_power_twenty_seven_l160_160879

theorem remainder_seven_power_twenty_seven :
  (7^27) % 1000 = 543 := 
sorry

end remainder_seven_power_twenty_seven_l160_160879


namespace possible_values_of_f_l160_160485

-- Define the function f(A, B, C)
def f (A B C : ℕ) : ℕ := A^3 + B^3 + C^3 - 3 * A * B * C

-- The statement of the theorem
theorem possible_values_of_f (n : ℕ) :
  (∃ A B C : ℕ, f A B C = n) ↔ ¬ ((n % 9 = 3) ∨ (n % 9 = 6)) :=
by
  sorry

end possible_values_of_f_l160_160485


namespace televisions_sold_this_black_friday_l160_160287

theorem televisions_sold_this_black_friday 
  (T : ℕ) 
  (h1 : ∀ (n : ℕ), n = 3 → (T + (50 * n) = 477)) 
  : T = 327 := 
sorry

end televisions_sold_this_black_friday_l160_160287


namespace product_513_12_l160_160016

theorem product_513_12 : 513 * 12 = 6156 := 
  by
    sorry

end product_513_12_l160_160016


namespace complex_number_second_quadrant_l160_160807

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i * (1 + i)

-- Define a predicate to determine if a complex number is in the second quadrant
def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- The main statement
theorem complex_number_second_quadrant : is_second_quadrant z := by
  sorry

end complex_number_second_quadrant_l160_160807


namespace geometric_sequence_ratio_l160_160685

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : 6 * a 7 = (a 8 + a 9) / 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = a 1 * (1 - q^n) / (1 - q)) :
  S 6 / S 3 = 28 :=
by
  -- The proof goes here
  sorry

end geometric_sequence_ratio_l160_160685


namespace distance_between_towns_l160_160546

-- Define the custom scale for conversion
def scale_in_km := 1.05  -- 1 km + 50 meters as 1.05 km

-- Input distances on the map and their conversion
def map_distance_in_inches := 6 + 11/16

noncomputable def actual_distance_in_km : ℝ :=
  let distance_in_inches := (6 * 8 + 11) / 16
  distance_in_inches * (8 / 3)

theorem distance_between_towns :
  actual_distance_in_km = 17.85 := by
  -- Equivalent mathematical steps and tests here
  sorry

end distance_between_towns_l160_160546


namespace interest_amount_eq_750_l160_160322

-- Definitions
def P : ℕ := 3000
def R : ℕ := 5
def T : ℕ := 5

-- Condition
def interest_less_than_sum := 2250

-- Simple interest formula
def simple_interest (P R T : ℕ) := (P * R * T) / 100

-- Theorem
theorem interest_amount_eq_750 : simple_interest P R T = P - interest_less_than_sum :=
by
  -- We assert that we need to prove the equality holds.
  sorry

end interest_amount_eq_750_l160_160322


namespace son_l160_160628

theorem son's_age (F S : ℕ) (h1 : F + S = 75) (h2 : F = 8 * (S - (F - S))) : S = 27 :=
sorry

end son_l160_160628


namespace rectangle_area_l160_160463

theorem rectangle_area (square_area : ℝ) (width length : ℝ) 
  (h1 : square_area = 36) 
  (h2 : width = real.sqrt square_area) 
  (h3 : length = 3 * width) : 
  width * length = 108 :=
by
  sorry

end rectangle_area_l160_160463


namespace percent_spent_on_other_items_l160_160145

def total_amount_spent (T : ℝ) : ℝ := T
def clothing_percent (p : ℝ) : Prop := p = 0.45
def food_percent (p : ℝ) : Prop := p = 0.45
def clothing_tax (t : ℝ) (T : ℝ) : ℝ := 0.05 * (0.45 * T)
def food_tax (t : ℝ) (T : ℝ) : ℝ := 0.0 * (0.45 * T)
def other_items_tax (p : ℝ) (T : ℝ) : ℝ := 0.10 * (p * T)
def total_tax (T : ℝ) (tax : ℝ) : Prop := tax = 0.0325 * T

theorem percent_spent_on_other_items (T : ℝ) (p_clothing p_food x : ℝ) (tax : ℝ) 
  (h1 : clothing_percent p_clothing) (h2 : food_percent p_food)
  (h3 : clothing_tax tax T = 0.05 * (0.45 * T))
  (h4 : food_tax tax T = 0.0)
  (h5 : other_items_tax x T = 0.10 * (x * T))
  (h6 : total_tax T (clothing_tax tax T + food_tax tax T + other_items_tax x T)) : 
  x = 0.10 :=
by
  sorry

end percent_spent_on_other_items_l160_160145


namespace martin_class_number_l160_160651

theorem martin_class_number (b : ℕ) (h1 : 100 < b) (h2 : b < 200) 
  (h3 : b % 3 = 2) (h4 : b % 4 = 1) (h5 : b % 5 = 1) : 
  b = 101 ∨ b = 161 := 
by
  sorry

end martin_class_number_l160_160651


namespace determine_b_l160_160808

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x < 1 then 3 * x - b else 2 ^ x

theorem determine_b (b : ℝ) :
  f (f (5 / 6) b) b = 4 ↔ b = 1 / 2 :=
by sorry

end determine_b_l160_160808


namespace snow_on_Monday_l160_160532

def snow_on_Tuesday : ℝ := 0.21
def snow_on_Monday_and_Tuesday : ℝ := 0.53

theorem snow_on_Monday : snow_on_Monday_and_Tuesday - snow_on_Tuesday = 0.32 :=
by
  sorry

end snow_on_Monday_l160_160532


namespace black_squares_in_35th_row_l160_160204

-- Define the condition for the starting color based on the row
def starts_with_black (n : ℕ) : Prop := n % 2 = 1
def ends_with_white (n : ℕ) : Prop := true  -- This is trivially true by the problem condition
def total_squares (n : ℕ) : ℕ := 2 * n 
-- Black squares are half of the total squares for rows starting with a black square
def black_squares (n : ℕ) : ℕ := total_squares n / 2

theorem black_squares_in_35th_row : black_squares 35 = 35 :=
sorry

end black_squares_in_35th_row_l160_160204


namespace cos_beta_of_tan_alpha_and_sin_alpha_plus_beta_l160_160504

theorem cos_beta_of_tan_alpha_and_sin_alpha_plus_beta 
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_tanα : Real.tan α = 3) (h_sin_alpha_beta : Real.sin (α + β) = 3 / 5) :
  Real.cos β = Real.sqrt 10 / 10 := 
sorry

end cos_beta_of_tan_alpha_and_sin_alpha_plus_beta_l160_160504


namespace fraction_eq_repeating_decimal_l160_160069

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l160_160069


namespace probability_A_fires_proof_l160_160927

noncomputable def probability_A_fires (prob_A : ℝ) (prob_B : ℝ) : ℝ :=
  1 / 6 + (5 / 6) * (5 / 6) * prob_A

theorem probability_A_fires_proof :
  ∀ prob_A prob_B : ℝ,
  prob_A = (1 / 6 + (5 / 6) * (5 / 6) * prob_A) →
  prob_A = 6 / 11 :=
by
  sorry

end probability_A_fires_proof_l160_160927


namespace find_number_l160_160884

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l160_160884


namespace number_of_fences_painted_l160_160346

-- Definitions based on the problem conditions
def meter_fee : ℝ := 0.2
def fence_length : ℝ := 500
def total_earnings : ℝ := 5000

-- Target statement
theorem number_of_fences_painted : (total_earnings / (fence_length * meter_fee)) = 50 := by
sorry

end number_of_fences_painted_l160_160346


namespace complex_inverse_identity_l160_160248

theorem complex_inverse_identity : ∀ (i : ℂ), i^2 = -1 → (3 * i - 2 * i⁻¹)⁻¹ = -i / 5 :=
by
  -- Let's introduce the variables and the condition.
  intro i h

  -- Sorry is used to signify the proof is omitted.
  sorry

end complex_inverse_identity_l160_160248


namespace negation_of_proposition_p_l160_160105

def f : ℝ → ℝ := sorry

theorem negation_of_proposition_p :
  (¬ (∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0)) ↔ (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) := 
by
  sorry

end negation_of_proposition_p_l160_160105


namespace ways_to_choose_socks_l160_160148

open Finset

theorem ways_to_choose_socks :
  let socks := { "blue1", "blue2", "brown", "black", "red", "purple", "green" }
  let choices := socks.subsets.filter (λ s, s.card = 4 ∧ 2 ≤ s.count "blue1" + s.count "blue2")
  choices.card = 30 :=
by
  sorry

end ways_to_choose_socks_l160_160148


namespace average_speed_remaining_l160_160002

theorem average_speed_remaining (D : ℝ) : 
    (0.4 * D / 40 + 0.6 * D / S) = D / 50 → S = 60 :=
by 
  sorry

end average_speed_remaining_l160_160002


namespace find_constants_l160_160348

variable (x : ℝ)

/-- Restate the equation problem and the constants A, B, C, D to be found. -/
theorem find_constants 
  (A B C D : ℝ)
  (h : ∀ x, x^3 - 7 = A * (x - 3) * (x - 5) * (x - 7) + B * (x - 2) * (x - 5) * (x - 7) + C * (x - 2) * (x - 3) * (x - 7) + D * (x - 2) * (x - 3) * (x - 5)) :
  A = 1/15 ∧ B = 5/2 ∧ C = -59/6 ∧ D = 42/5 :=
  sorry

end find_constants_l160_160348


namespace height_on_hypotenuse_l160_160259

theorem height_on_hypotenuse (a b : ℕ) (hypotenuse : ℝ)
  (ha : a = 3) (hb : b = 4) (h_c : hypotenuse = sqrt (a^2 + b^2)) :
  let S := (1/2 : ℝ) * a * b in
  ∃ h : ℝ, h = (2 * S) / hypotenuse ∧ h = 12/5 := by
  sorry

end height_on_hypotenuse_l160_160259


namespace g_negative_example1_g_negative_example2_g_negative_example3_l160_160139

noncomputable def g (a : ℚ) : ℚ := sorry

axiom g_mul (a b : ℚ) (ha : 0 < a) (hb : 0 < b) : g (a * b) = g a + g b
axiom g_prime (p : ℕ) (hp : Nat.Prime p) : g (p * p) = p

theorem g_negative_example1 : g (8/81) < 0 := sorry
theorem g_negative_example2 : g (25/72) < 0 := sorry
theorem g_negative_example3 : g (49/18) < 0 := sorry

end g_negative_example1_g_negative_example2_g_negative_example3_l160_160139


namespace probability_A_fires_l160_160929

theorem probability_A_fires :
  let p_A := (1 : ℚ) / 6 + (5 : ℚ) / 6 * (5 : ℚ) / 6 * p_A
  in p_A = 6 / 11 :=
by
  sorry

end probability_A_fires_l160_160929


namespace exponentiation_properties_l160_160796

theorem exponentiation_properties
  (a : ℝ) (m n : ℕ) (hm : a^m = 9) (hn : a^n = 3) : a^(m - n) = 3 :=
by
  sorry

end exponentiation_properties_l160_160796


namespace pow_simplification_l160_160476

theorem pow_simplification :
  9^6 * 3^3 / 27^4 = 27 :=
by
  sorry

end pow_simplification_l160_160476


namespace initial_marbles_l160_160643

-- Define the conditions as constants
def marbles_given_to_Juan : ℕ := 73
def marbles_left_with_Connie : ℕ := 70

-- Prove that Connie initially had 143 marbles
theorem initial_marbles (initial_marbles : ℕ) :
  initial_marbles = marbles_given_to_Juan + marbles_left_with_Connie → 
  initial_marbles = 143 :=
by
  intro h
  rw [h]
  rfl

end initial_marbles_l160_160643


namespace ursula_purchases_total_cost_l160_160573

variable (T C B Br : ℝ)
variable (hT : T = 10) (hTC : T = 2 * C) (hB : B = 0.8 * C) (hBr : Br = B / 2)

theorem ursula_purchases_total_cost : T + C + B + Br = 21 := by
  sorry

end ursula_purchases_total_cost_l160_160573


namespace transformed_function_equivalence_l160_160303

-- Define the original function
def original_function (x : ℝ) : ℝ := 2 * x + 1

-- Define the transformation involving shifting 2 units to the right
def transformed_function (x : ℝ) : ℝ := original_function (x - 2)

-- The theorem we want to prove
theorem transformed_function_equivalence : 
  ∀ x : ℝ, transformed_function x = 2 * x - 3 :=
by
  sorry

end transformed_function_equivalence_l160_160303


namespace length_lemma_l160_160118

def smallest_non_divisor (n : ℕ) : ℕ :=
  if h : n > 0 then Nat.find (Nat.exists_not_dvd h) else 1

def length (n : ℕ) : ℕ :=
  if hn : n >= 3 then Nat.part_rec_on (fun m => smallest_non_divisor m) (fun k => if k = 2 then 0 else 1) hn else 0

theorem length_lemma (n : ℕ) (h : n >= 3) :
  length n = if n % 2 = 0 then if (smallest_non_divisor (smallest_non_divisor n)) = 3 then 3 else 2 else 1 :=
sorry

end length_lemma_l160_160118


namespace original_number_increase_l160_160459

theorem original_number_increase (x : ℝ) (h : 1.20 * x = 1800) : x = 1500 :=
by
  sorry

end original_number_increase_l160_160459


namespace quadratic_to_vertex_form_l160_160659

theorem quadratic_to_vertex_form : ∃ m n : ℝ, (∀ x : ℝ, x^2 - 8*x + 3 = 0 ↔ (x - m)^2 = n) ∧ m + n = 17 :=
by sorry

end quadratic_to_vertex_form_l160_160659


namespace base_7_units_digit_of_product_359_72_l160_160714

def base_7_units_digit (n : ℕ) : ℕ := n % 7

theorem base_7_units_digit_of_product_359_72 : base_7_units_digit (359 * 72) = 4 := 
by
  sorry

end base_7_units_digit_of_product_359_72_l160_160714


namespace determine_d_l160_160699

variables (u v : ℝ × ℝ × ℝ) -- defining u and v as 3D vectors

noncomputable def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.2 * b.2.1 - a.2.1 * b.2.2, a.1 * b.2.2 - a.2.2 * b.1 , a.2.1 * b.1 - a.1 * b.2.1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

noncomputable def i : ℝ × ℝ × ℝ := (1, 0, 0)
noncomputable def j : ℝ × ℝ × ℝ := (0, 1, 0)
noncomputable def k : ℝ × ℝ × ℝ := (0, 0, 1)

theorem determine_d (u : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) :
  cross_product i (cross_product (u + v) i) +
  cross_product j (cross_product (u + v) j) +
  cross_product k (cross_product (u + v) k) =
  2 * (u + v) :=
sorry

end determine_d_l160_160699


namespace find_number_l160_160890

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l160_160890


namespace find_x_l160_160680

def operation (a b : ℝ) : ℝ := a * b^(1/2)

theorem find_x (x : ℝ) : operation x 9 = 12 → x = 4 :=
by
  intro h
  sorry

end find_x_l160_160680


namespace enrolled_percentage_l160_160017

theorem enrolled_percentage (total_students : ℝ) (non_bio_students : ℝ)
    (h_total : total_students = 880)
    (h_non_bio : non_bio_students = 440.00000000000006) : 
    ((total_students - non_bio_students) / total_students) * 100 = 50 := 
by
  rw [h_total, h_non_bio]
  norm_num
  sorry

end enrolled_percentage_l160_160017


namespace cone_volume_l160_160568

theorem cone_volume (s h : ℝ) (h_s : s = 15) (h_h : h = 9) : 
  ∃ V : ℝ, V = (1 / 3) * π * (sqrt (s^2 - h^2))^2 * h := 
by
  use 432 * π
  rw [h_s, h_h]
  have rsq : 15^2 - 9^2 = 144 := by norm_num
  have radius : sqrt (15^2 - 9^2) = 12 := by rw [rsq, Real.sqrt_sq_eq_abs, abs_of_nonneg]; norm_num
  simp [radius]
  norm_num
  sorry

end cone_volume_l160_160568


namespace part1_part2_l160_160189

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x > -1, (x^2 + 3*x + 6) / (x + 1) ≥ a) ↔ (a ≤ 5) := 
  sorry

-- Part 2
theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) : 
  2*a + (1/a) + 4*b + (8/b) ≥ 27 :=
  sorry

end part1_part2_l160_160189


namespace angle_C_is_65_deg_l160_160267

-- Defining a triangle and its angles.
structure Triangle :=
  (A B C : ℝ) -- representing the angles in degrees

-- Defining the conditions of the problem.
def given_triangle : Triangle :=
  { A := 75, B := 40, C := 180 - 75 - 40 }

-- Statement of the problem, proving that the measure of ∠C is 65°.
theorem angle_C_is_65_deg (t : Triangle) (hA : t.A = 75) (hB : t.B = 40) (hSum : t.A + t.B + t.C = 180) : t.C = 65 :=
  by sorry

end angle_C_is_65_deg_l160_160267


namespace eval_expression_l160_160020

theorem eval_expression : (Real.pi + 2023)^0 + 2 * Real.sin (45 * Real.pi / 180) - (1 / 2)^(-1 : ℤ) + abs (Real.sqrt 2 - 2) = 1 :=
by
  sorry

end eval_expression_l160_160020


namespace car_length_l160_160774

variables (L E C : ℕ)

theorem car_length (h1 : 150 * E = L + 150 * C) (h2 : 30 * E = L - 30 * C) : L = 113 * E :=
by
  sorry

end car_length_l160_160774


namespace fraction_eq_repeating_decimal_l160_160044

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l160_160044


namespace product_of_triple_when_added_to_reciprocal_l160_160580

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l160_160580


namespace temperature_difference_l160_160155

-- Define variables for the highest and lowest temperatures.
def highest_temp : ℤ := 18
def lowest_temp : ℤ := -2

-- Define the statement for the maximum temperature difference.
theorem temperature_difference : 
  highest_temp - lowest_temp = 20 := 
by 
  sorry

end temperature_difference_l160_160155


namespace sums_remainders_equal_l160_160278

-- Definition and conditions
variables (A A' D S S' s s' : ℕ) 
variables (h1 : A > A') 
variables (h2 : A % D = S) 
variables (h3 : A' % D = S') 
variables (h4 : (A + A') % D = s) 
variables (h5 : (S + S') % D = s')

-- Proof statement
theorem sums_remainders_equal : s = s' := 
  sorry

end sums_remainders_equal_l160_160278


namespace circle_radius_l160_160321

theorem circle_radius (r : ℝ) (x y : ℝ) (h₁ : x = π * r ^ 2) (h₂ : y = 2 * π * r - 6) (h₃ : x + y = 94 * π) : 
  r = 10 :=
sorry

end circle_radius_l160_160321


namespace total_revenue_is_405_l160_160575

-- Define the cost of rentals
def canoeCost : ℕ := 15
def kayakCost : ℕ := 18

-- Define terms for number of rentals
variables (C K : ℕ)

-- Conditions
axiom ratio_condition : 2 * C = 3 * K
axiom difference_condition : C = K + 5

-- Total revenue
def totalRevenue (C K : ℕ) : ℕ := (canoeCost * C) + (kayakCost * K)

-- Theorem statement
theorem total_revenue_is_405 (C K : ℕ) (H1 : 2 * C = 3 * K) (H2 : C = K + 5) : 
  totalRevenue C K = 405 := by
  sorry

end total_revenue_is_405_l160_160575


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160904

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160904


namespace angleina_speed_from_grocery_to_gym_l160_160764

variable (v : ℝ) (h1 : 720 / v - 40 = 240 / v)

theorem angleina_speed_from_grocery_to_gym : 2 * v = 24 :=
by
  sorry

end angleina_speed_from_grocery_to_gym_l160_160764


namespace product_of_real_solutions_triple_property_l160_160587

theorem product_of_real_solutions_triple_property :
  ∀ x : ℝ, (x + 1/x = 3 * x) → (∏ x in {x | x + 1/x = 3 * x}, x) = -1/2 := 
by
  sorry

end product_of_real_solutions_triple_property_l160_160587


namespace number_of_bad_arrangements_eq_2_l160_160866

def is_bad_arrangement (arr : List ℕ) : Prop :=
  ∀ n : ℕ, n ∈ Finset.range 21 → ∃ (subarr : List ℕ), subarr ≠ [] 
  ∧ (subarr.sum = n ∧ (∀ i j, List.take (j - i) (List.drop i arr) = subarr → List.drop i arr ≠ subarr))

theorem number_of_bad_arrangements_eq_2 : 
  ∃ bad_arrangements : Finset (List ℕ), bad_arrangements.card = 2 ∧ 
  (∀ p, p ∈ bad_arrangements → is_bad_arrangement p) :=
sorry

end number_of_bad_arrangements_eq_2_l160_160866


namespace part1_sales_increase_part2_price_reduction_l160_160205

-- Part 1: If the price is reduced by 4 yuan, the new average daily sales will be 28 items.
theorem part1_sales_increase (initial_sales : ℕ) (increase_per_yuan : ℕ) (reduction : ℕ) :
  initial_sales = 20 → increase_per_yuan = 2 → reduction = 4 →
  initial_sales + increase_per_yuan * reduction = 28 :=
by sorry

-- Part 2: By how much should the price of each item be reduced for a daily profit of 1050 yuan.
theorem part2_price_reduction (initial_sales : ℕ) (increase_per_yuan : ℕ) (initial_profit : ℕ) 
  (target_profit : ℕ) (min_profit_per_item : ℕ) (x : ℕ) :
  initial_sales = 20 → increase_per_yuan = 2 → initial_profit = 40 → target_profit = 1050 
  → min_profit_per_item = 25 → (40 - x) * (20 + 2 * x) = 1050 → (40 - x) ≥ 25 → x = 5 :=
by sorry

end part1_sales_increase_part2_price_reduction_l160_160205


namespace one_sixths_in_fraction_l160_160115

theorem one_sixths_in_fraction :
  (11 / 3) / (1 / 6) = 22 :=
sorry

end one_sixths_in_fraction_l160_160115


namespace cone_volume_l160_160567

variable (l h : ℝ) (π : ℝ := Real.pi)

noncomputable def radius (l h : ℝ) : ℝ := sqrt (l^2 - h^2)

noncomputable def volume_of_cone (r h : ℝ) (π : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem cone_volume (hl : l = 15) (hh : h = 9) :
  volume_of_cone (radius l h) h π = 432 * π :=
by
  sorry

end cone_volume_l160_160567


namespace repeating_decimal_eq_fraction_l160_160055

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l160_160055


namespace pollen_particle_diameter_in_scientific_notation_l160_160639

theorem pollen_particle_diameter_in_scientific_notation :
  0.0000078 = 7.8 * 10^(-6) :=
by
  sorry

end pollen_particle_diameter_in_scientific_notation_l160_160639


namespace frances_card_value_l160_160270

theorem frances_card_value (x : ℝ) (hx : 90 < x ∧ x < 180) :
  (∃ f : ℝ → ℝ, f = sin ∨ f = cos ∨ f = tan ∧
    f x = -1 ∧
    (∃ y : ℝ, y ≠ x ∧ (sin y ≠ -1 ∧ cos y ≠ -1 ∧ tan y ≠ -1))) :=
sorry

end frances_card_value_l160_160270


namespace sail_pressure_l160_160562

theorem sail_pressure (k : ℝ) :
  (forall (V A : ℝ), P = k * A * (V : ℝ)^2) 
  → (P = 1.25) → (V = 20) → (A = 1)
  → (A = 4) → (V = 40)
  → (P = 20) :=
by
  sorry

end sail_pressure_l160_160562


namespace intersection_P_Q_eq_Q_l160_160279

-- Definitions of P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Statement to prove P ∩ Q = Q
theorem intersection_P_Q_eq_Q : P ∩ Q = Q := 
by 
  sorry

end intersection_P_Q_eq_Q_l160_160279


namespace geom_seq_ratio_l160_160357
noncomputable section

theorem geom_seq_ratio (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h₁ : 0 < a_1)
  (h₂ : 0 < a_2)
  (h₃ : 0 < a_3)
  (h₄ : 0 < a_4)
  (h₅ : 0 < a_5)
  (h_seq : a_2 = a_1 * 2)
  (h_seq2 : a_3 = a_1 * 2^2)
  (h_seq3 : a_4 = a_1 * 2^3)
  (h_seq4 : a_5 = a_1 * 2^4)
  (h_ratio : a_4 / a_1 = 8) :
  (a_1 + a_2) * a_4 / ((a_1 + a_3) * a_5) = 3 / 10 := 
by
  sorry

end geom_seq_ratio_l160_160357


namespace expand_expression_l160_160223

theorem expand_expression (x y : ℝ) : 12 * (3 * x - 4 * y + 2) = 36 * x - 48 * y + 24 :=
by
  sorry

end expand_expression_l160_160223


namespace repeating_fraction_equality_l160_160085

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l160_160085


namespace neces_not_suff_cond_l160_160180

theorem neces_not_suff_cond (a : ℝ) (h : a ≠ 0) : (1 / a < 1) → (a > 1) :=
sorry

end neces_not_suff_cond_l160_160180


namespace find_a_value_l160_160803

theorem find_a_value
  (a : ℕ)
  (x y : ℝ)
  (h1 : a * x + y = -4)
  (h2 : 2 * x + y = -2)
  (hx_neg : x < 0)
  (hy_pos : y > 0) :
  a = 3 :=
by
  sorry

end find_a_value_l160_160803


namespace inequality_incorrect_l160_160247

theorem inequality_incorrect (a b : ℝ) (h : a > b) : ¬(-a + 2 > -b + 2) :=
by
  sorry

end inequality_incorrect_l160_160247


namespace geom_seq_sum_eq_31_over_4_l160_160499

/-- Definition of a geometric sequence sum -/
def sum_geom_series (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geom_seq_sum_eq_31_over_4 (a1 q : ℝ) (S : ℕ → ℝ) :
  (a1 + a1 * q = 3/4) ∧ (a1 * q^3 * (1 + q) = 6) →
  sum_geom_series a1 q 5 = 31/4 :=
by
  intros h,
  sorry

end geom_seq_sum_eq_31_over_4_l160_160499


namespace find_number_divided_by_3_equals_subtracted_5_l160_160892

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l160_160892


namespace bottle_total_height_l160_160462

theorem bottle_total_height (r1 r2 water_height_up water_height_down : ℝ) (h_r1 : r1 = 1) (h_r2 : r2 = 3) (h_water_height_up : water_height_up = 20) (h_water_height_down : water_height_down = 28) : 
    ∃ x : ℝ, (π * r1^2 * (x - water_height_up) = 9 * π * (x - water_height_down) ∧ x = 29) := 
by 
    sorry

end bottle_total_height_l160_160462


namespace money_left_l160_160836

def initial_money : ℝ := 18
def spent_on_video_games : ℝ := 6
def spent_on_snack : ℝ := 3
def toy_original_cost : ℝ := 4
def toy_discount : ℝ := 0.25

theorem money_left (initial_money spent_on_video_games spent_on_snack toy_original_cost toy_discount : ℝ) :
  initial_money = 18 →
  spent_on_video_games = 6 →
  spent_on_snack = 3 →
  toy_original_cost = 4 →
  toy_discount = 0.25 →
  (initial_money - (spent_on_video_games + spent_on_snack + (toy_original_cost * (1 - toy_discount)))) = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end money_left_l160_160836


namespace sum_of_squares_first_15_l160_160433

def sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem sum_of_squares_first_15 : sum_of_squares 15 = 3720 :=
by
  sorry

end sum_of_squares_first_15_l160_160433


namespace rectangle_area_l160_160950

open Real

theorem rectangle_area (A : ℝ) (s l w : ℝ) (h1 : A = 9 * sqrt 3) (h2 : A = (sqrt 3 / 4) * s^2)
  (h3 : w = s) (h4 : l = 3 * w) : w * l = 108 :=
by
  sorry

end rectangle_area_l160_160950


namespace find_243xyz_divisible_by_396_l160_160664

/--
Find the values of x, y, and z such that the number 243xyz is divisible by 396.
-/
theorem find_243xyz_divisible_by_396 :
  ∃ (x y z : ℕ),
    (y * 10 + z) % 4 = 0 ∧
    (2 + 4 + 3 + x + y + z) % 9 = 0 ∧
    ((2 + 3 + y) - (4 + x + z)) % 11 = 0 ∧
    (243 * 1000 + x * 100 + y * 10 + z) % 396 = 0 :=
begin
  use [5, 4, 0], -- one possible solution (243540)
  split,
  { exact dec_trivial }, -- y*10+z=54, 54 mod 4 = 0
  split,
  { exact dec_trivial }, -- 2+4+3+x+y+z=18, 18 mod 9 = 0
  split,
  { exact dec_trivial }, -- (2+3+y) - (4+x+z)=5-5, 0 mod 11 = 0
  { exact dec_trivial }  -- 243540 mod 396 = 0
end

end find_243xyz_divisible_by_396_l160_160664


namespace evaluate_f_a_plus_1_l160_160513

variable (a : ℝ)  -- The variable a is a real number.

def f (x : ℝ) : ℝ := x^2 + 1  -- The function f is defined as x^2 + 1.

theorem evaluate_f_a_plus_1 : f (a + 1) = a^2 + 2 * a + 2 := by
  -- Provide the proof here
  sorry

end evaluate_f_a_plus_1_l160_160513


namespace spherical_to_rectangular_coords_l160_160481

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), 
    ρ = 3 → θ = 3 * Real.pi / 2 → φ = Real.pi / 3 →
    (let x := ρ * Real.sin φ * Real.cos θ;
         y := ρ * Real.sin φ * Real.sin θ;
         z := ρ * Real.cos φ 
     in (x, y, z) = (0, - (3 * Real.sqrt 3) / 2, 3 / 2)) :=
by
  intros ρ θ φ hρ hθ hφ
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coords_l160_160481


namespace carbonic_acid_formation_l160_160662

-- Definition of amounts of substances involved
def moles_CO2 : ℕ := 3
def moles_H2O : ℕ := 3

-- Stoichiometric condition derived from the equation CO2 + H2O → H2CO3
def stoichiometric_ratio (a b c : ℕ) : Prop := (a = b) ∧ (a = c)

-- The main statement to prove
theorem carbonic_acid_formation : 
  stoichiometric_ratio moles_CO2 moles_H2O 3 :=
by
  sorry

end carbonic_acid_formation_l160_160662


namespace system_solutions_l160_160647

noncomputable def f (t : ℝ) : ℝ := 4 * t^2 / (1 + 4 * t^2)

theorem system_solutions (x y z : ℝ) :
  (f x = y ∧ f y = z ∧ f z = x) ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solutions_l160_160647


namespace trip_duration_60_mph_l160_160873

noncomputable def time_at_new_speed (initial_time : ℚ) (initial_speed : ℚ) (new_speed : ℚ) : ℚ :=
  initial_time * (initial_speed / new_speed)

theorem trip_duration_60_mph :
  time_at_new_speed (9 / 2) 70 60 = 5.25 := 
by
  sorry

end trip_duration_60_mph_l160_160873


namespace theta_in_third_quadrant_l160_160246

-- Define the mathematical conditions
variable (θ : ℝ)
axiom cos_theta_neg : Real.cos θ < 0
axiom cos_minus_sin_eq_sqrt : Real.cos θ - Real.sin θ = Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ)

-- Prove that θ is in the third quadrant
theorem theta_in_third_quadrant : 
  (∀ θ : ℝ, Real.cos θ < 0 → Real.cos θ - Real.sin θ = Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ) → 
    Real.sin θ < 0 ∧ Real.cos θ < 0) :=
by sorry

end theta_in_third_quadrant_l160_160246


namespace science_votes_percentage_l160_160260

theorem science_votes_percentage 
  (math_votes : ℕ) (english_votes : ℕ) (science_votes : ℕ) (history_votes : ℕ) (art_votes : ℕ) 
  (total_votes : ℕ := math_votes + english_votes + science_votes + history_votes + art_votes) 
  (percentage : ℕ := ((science_votes * 100) / total_votes)) :
  math_votes = 80 →
  english_votes = 70 →
  science_votes = 90 →
  history_votes = 60 →
  art_votes = 50 →
  percentage = 26 :=
by
  intros
  sorry

end science_votes_percentage_l160_160260


namespace rectangular_floor_paint_l160_160460

theorem rectangular_floor_paint (a b : ℕ) (ha : a > 0) (hb : b > a) (h1 : a * b = 2 * (a - 4) * (b - 4) + 32) : 
  ∃ pairs : Finset (ℕ × ℕ), pairs.card = 3 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → b > a :=
by 
  sorry

end rectangular_floor_paint_l160_160460


namespace seed_mixture_Y_is_25_percent_ryegrass_l160_160293

variables (X Y : ℝ) (R : ℝ)

def proportion_X_is_40_percent_ryegrass : Prop :=
  X = 40 / 100

def proportion_Y_contains_percent_ryegrass (R : ℝ) : Prop :=
  100 - R = 75 / 100 * 100

def mixture_contains_30_percent_ryegrass (X Y R : ℝ) : Prop :=
  (1/3) * (40 / 100) * 100 + (2/3) * (R / 100) * 100 = 30

def weight_of_mixture_is_33_percent_X (X Y : ℝ) : Prop :=
  X / (X + Y) = 1 / 3

theorem seed_mixture_Y_is_25_percent_ryegrass
  (X Y : ℝ) (R : ℝ) 
  (h1 : proportion_X_is_40_percent_ryegrass X)
  (h2 : proportion_Y_contains_percent_ryegrass R)
  (h3 : weight_of_mixture_is_33_percent_X X Y)
  (h4 : mixture_contains_30_percent_ryegrass X Y R) :
  R = 25 :=
sorry

end seed_mixture_Y_is_25_percent_ryegrass_l160_160293


namespace negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic_l160_160560

-- Definitions based on the conditions in the problem:
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b
def MonotonicFunction (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The proposition that 'All linear functions are monotonic functions'
def AllLinearAreMonotonic : Prop := ∀ (f : ℝ → ℝ), LinearFunction f → MonotonicFunction f

-- The correct answer to the question:
def SomeLinearAreNotMonotonic : Prop := ∃ (f : ℝ → ℝ), LinearFunction f ∧ ¬ MonotonicFunction f

-- The proof problem:
theorem negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic : 
  ¬ AllLinearAreMonotonic ↔ SomeLinearAreNotMonotonic :=
by
  sorry

end negation_of_AllLinearAreMonotonic_is_SomeLinearAreNotMonotonic_l160_160560


namespace modulus_difference_l160_160674

def z1 : Complex := 1 + 2 * Complex.I
def z2 : Complex := 2 + Complex.I

theorem modulus_difference :
  Complex.abs (z2 - z1) = Real.sqrt 2 := by sorry

end modulus_difference_l160_160674


namespace fraction_of_repeating_decimal_l160_160073

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l160_160073


namespace expression_for_an_l160_160104

noncomputable def arithmetic_sequence (d : ℕ) (n : ℕ) : ℕ :=
  2 + (n - 1) * d

theorem expression_for_an (d : ℕ) (n : ℕ) 
  (h1 : d > 0)
  (h2 : (arithmetic_sequence d 1) = 2)
  (h3 : (arithmetic_sequence d 1) < (arithmetic_sequence d 2))
  (h4 : (arithmetic_sequence d 2)^2 = 2 * (arithmetic_sequence d 4)) :
  arithmetic_sequence d n = 2 * n := sorry

end expression_for_an_l160_160104


namespace range_of_x_for_obtuse_angle_l160_160682

def vectors_are_obtuse (a b : ℝ × ℝ) : Prop :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  dot_product < 0

theorem range_of_x_for_obtuse_angle :
  ∀ (x : ℝ), vectors_are_obtuse (1, 3) (x, -1) ↔ (x < -1/3 ∨ (-1/3 < x ∧ x < 3)) :=
by
  sorry

end range_of_x_for_obtuse_angle_l160_160682


namespace angle_sum_triangle_l160_160265

theorem angle_sum_triangle (A B C : ℝ) (hA : A = 75) (hB : B = 40) (h_sum : A + B + C = 180) : C = 65 :=
by
  sorry

end angle_sum_triangle_l160_160265


namespace repeating_decimal_equiv_fraction_l160_160092

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l160_160092


namespace reading_time_per_disc_l160_160937

theorem reading_time_per_disc (total_minutes : ℕ) (disc_capacity : ℕ) (d : ℕ) (reading_per_disc : ℕ) :
  total_minutes = 528 ∧ disc_capacity = 45 ∧ d = 12 ∧ total_minutes = d * reading_per_disc → reading_per_disc = 44 :=
by
  sorry

end reading_time_per_disc_l160_160937


namespace equation_equivalence_l160_160250

theorem equation_equivalence (p q : ℝ) (hp₀ : p ≠ 0) (hp₅ : p ≠ 5) (hq₀ : q ≠ 0) (hq₇ : q ≠ 7) :
  (3 / p + 5 / q = 1 / 3) → p = 9 * q / (q - 15) :=
by
  sorry

end equation_equivalence_l160_160250


namespace rationalize_simplify_l160_160850

theorem rationalize_simplify :
  3 / (Real.sqrt 75 + Real.sqrt 3) = Real.sqrt 3 / 6 :=
by
  sorry

end rationalize_simplify_l160_160850


namespace mean_of_S_eq_651_l160_160455

theorem mean_of_S_eq_651 
  (s n : ℝ) 
  (h1 : (s + 1) / (n + 1) = s / n - 13) 
  (h2 : (s + 2001) / (n + 1) = s / n + 27) 
  (hn : n ≠ 0) : s / n = 651 := 
by 
  sorry

end mean_of_S_eq_651_l160_160455


namespace repeating_fraction_equality_l160_160083

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l160_160083


namespace sector_max_area_l160_160800

noncomputable def max_sector_area (R c : ℝ) : ℝ := 
  if h : R = c / 4 then c^2 / 16 else 0 -- This is just a skeleton, actual proof requires conditions
-- State the theorem that relates conditions to the maximum area.
theorem sector_max_area (R c α : ℝ) 
  (hc : c = 2 * R + R * α) : 
  (∃ R, R = c / 4) → max_sector_area R c = c^2 / 16 :=
by 
  sorry

end sector_max_area_l160_160800


namespace find_multiple_l160_160563

theorem find_multiple (a b m : ℤ) (h1 : a * b = m * (a + b) + 12) 
(h2 : b = 10) (h3 : b - a = 6) : m = 2 :=
by {
  sorry
}

end find_multiple_l160_160563


namespace space_is_volume_stuff_is_capacity_film_is_surface_area_l160_160871

-- Let's define the properties based on the conditions
def size_of_space (box : Type) : Type := 
  sorry -- This will be volume later

def stuff_can_hold (box : Type) : Type :=
  sorry -- This will be capacity later

def film_needed_to_cover (box : Type) : Type :=
  sorry -- This will be surface area later

-- Now prove the correspondences
theorem space_is_volume (box : Type) :
  size_of_space box = volume := 
by 
  sorry

theorem stuff_is_capacity (box : Type) :
  stuff_can_hold box = capacity := 
by 
  sorry

theorem film_is_surface_area (box : Type) :
  film_needed_to_cover box = surface_area := 
by 
  sorry

end space_is_volume_stuff_is_capacity_film_is_surface_area_l160_160871


namespace number_of_schools_l160_160657

-- Define the conditions as parameters and assumptions
structure CityContest (n : ℕ) :=
  (students_per_school : ℕ := 4)
  (total_students : ℕ := students_per_school * n)
  (andrea_percentile : ℕ := 75)
  (andrea_highest_team : Prop)
  (beth_rank : ℕ := 20)
  (carla_rank : ℕ := 47)
  (david_rank : ℕ := 78)
  (andrea_position : ℕ)
  (h3 : andrea_position = (3 * total_students + 1) / 4)
  (h4 : 3 * n > 78)

-- Define the main theorem statement
theorem number_of_schools (n : ℕ) (contest : CityContest n) (h5 : contest.andrea_highest_team) : n = 20 :=
  by {
    -- You would insert the detailed proof of the theorem based on the conditions here.
    sorry
  }

end number_of_schools_l160_160657


namespace fred_speed_5_mph_l160_160979

theorem fred_speed_5_mph (F : ℝ) (h1 : 50 = 25 + 25) (h2 : 25 / 5 = 5) (h3 : 25 / F = 5) : 
  F = 5 :=
by
  -- Since Fred's speed makes meeting with Sam in the same time feasible
  sorry

end fred_speed_5_mph_l160_160979


namespace element_of_set_l160_160243

theorem element_of_set : -1 ∈ { x : ℝ | x^2 - 1 = 0 } :=
sorry

end element_of_set_l160_160243


namespace nth_inequality_l160_160798

theorem nth_inequality (n : ℕ) (x : ℝ) (h : x > 0) : x + n^n / x^n ≥ n + 1 := 
sorry

end nth_inequality_l160_160798


namespace total_height_bottle_l160_160461

theorem total_height_bottle (r1 r2 h1 h2 : ℝ) (h_total : h1 + h2 = 29)
  (h_water_upright : ℝ) (h_water_upside_down : ℝ) :
  r1 = 1 ∧ r2 = 3 → h_water_upright = 20 ∧ h_water_upside_down = 28 → h_total = 29 := by
  intros
  exact h_total
  sorry

end total_height_bottle_l160_160461


namespace equal_prob_first_ace_l160_160826

theorem equal_prob_first_ace (deck : List ℕ) (players : Fin 4) (h_deck_size : deck.length = 32)
  (h_distinct : deck.nodup) (h_aces : ∀ _i, deck.filter (λ card, card = 1 ).length = 4)
  (h_shuffled : ∀ (card : ℕ), card ∈ deck → card ∈ (range 32)) :
  ∀ (player : Fin 4), let positions := List.range' (player + 1) (32 / 4) * 4 + player;
  (∀ (pos : ℕ), pos ∈ positions → deck.nth pos = some 1) →
  P(player) = 1 / 8 :=
by
  sorry

end equal_prob_first_ace_l160_160826


namespace part_I_part_II_l160_160140

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log x - k * x

theorem part_I (k : ℝ) (hk : k = 1) :
  (∀ x, 0 < x ∧ x < 1 → 0 < f 1 x - f 1 1)
  ∧ (∀ x, 1 < x → f 1 1 > f 1 x)
  ∧ f 1 1 = 0 :=
by
  sorry

theorem part_II (k : ℝ) (h_no_zeros : ∀ x, f k x ≠ 0) :
  k > 1 / exp 1 :=
by
  sorry

end part_I_part_II_l160_160140


namespace problem_statement_l160_160356

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x + 4
def g (x : ℝ) : ℝ := 2*x - 1

-- State the theorem and provide the necessary conditions
theorem problem_statement : f (g 5) - g (f 5) = 381 :=
by
  sorry

end problem_statement_l160_160356


namespace product_of_solutions_l160_160581

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l160_160581


namespace trig_expression_value_l160_160505

theorem trig_expression_value (x : ℝ) (h : Real.tan (3 * Real.pi - x) = 2) : 
  (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 :=
by 
  sorry

end trig_expression_value_l160_160505


namespace max_dot_product_OB_OA_l160_160508

theorem max_dot_product_OB_OA (P A O B : ℝ × ℝ)
  (h₁ : ∃ x y : ℝ, (x, y) = P ∧ x^2 / 16 - y^2 / 9 = 1)
  (t : ℝ)
  (h₂ : A = (t - 1) • P)
  (h₃ : P • O = 64)
  (h₄ : B = (0, 1)) :
  ∃ t : ℝ, abs (B • A) ≤ (24/5) := 
sorry

end max_dot_product_OB_OA_l160_160508


namespace factorize_x_cube_minus_9x_l160_160783

-- Lean statement: Prove that x^3 - 9x = x(x+3)(x-3)
theorem factorize_x_cube_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
by
  sorry

end factorize_x_cube_minus_9x_l160_160783


namespace factorize_x_cubed_minus_9x_l160_160779

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l160_160779


namespace negation_of_proposition_l160_160719

theorem negation_of_proposition (a : ℝ) : 
  ¬(a = -1 → a^2 = 1) ↔ (a ≠ -1 → a^2 ≠ 1) :=
by sorry

end negation_of_proposition_l160_160719


namespace minimum_boxes_to_eliminate_l160_160126

theorem minimum_boxes_to_eliminate (total_boxes remaining_boxes : ℕ) 
  (high_value_boxes : ℕ) (h1 : total_boxes = 30) (h2 : high_value_boxes = 10)
  (h3 : remaining_boxes = total_boxes - 20) :
  remaining_boxes ≥ high_value_boxes → remaining_boxes = 10 :=
by 
  sorry

end minimum_boxes_to_eliminate_l160_160126


namespace solution_set_of_inequality_l160_160872

theorem solution_set_of_inequality (x : ℝ) : x * (9 - x) > 0 ↔ 0 < x ∧ x < 9 := by
  sorry

end solution_set_of_inequality_l160_160872


namespace sqrt_of_square_neg_three_l160_160190

theorem sqrt_of_square_neg_three : Real.sqrt ((-3 : ℝ)^2) = 3 := by
  sorry

end sqrt_of_square_neg_three_l160_160190


namespace evaluate_expression_l160_160961

variable (x y z : ℚ) -- assuming x, y, z are rational numbers

theorem evaluate_expression (h1 : x = 1 / 4) (h2 : y = 3 / 4) (h3 : z = -8) :
  x^2 * y^3 * z^2 = 108 := by
  sorry

end evaluate_expression_l160_160961


namespace circles_intersect_range_l160_160867

def circle1_radius := 3
def circle2_radius := 5

theorem circles_intersect_range : 2 < d ∧ d < 8 :=
by
  let r1 := circle1_radius
  let r2 := circle2_radius
  have h1 : d > r2 - r1 := sorry
  have h2 : d < r2 + r1 := sorry
  exact ⟨h1, h2⟩

end circles_intersect_range_l160_160867


namespace repeating_fraction_equality_l160_160088

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l160_160088


namespace david_average_marks_l160_160484

-- Define the individual marks
def english_marks : ℕ := 74
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 90
def total_marks : ℕ := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

-- Define the average marks
def average_marks : ℚ := total_marks / num_subjects

-- Assert the average marks calculation
theorem david_average_marks : average_marks = 75.6 := by
  sorry

end david_average_marks_l160_160484


namespace product_of_all_solutions_triple_reciprocal_l160_160583

noncomputable def product_of_solutions : ℝ :=
  let s := {x : ℝ | x + 1/x = 3*x} in
  s.prod (λ x, x)

theorem product_of_all_solutions_triple_reciprocal :
  (product_of_solutions) = - 1 / 2 :=
by
  sorry

end product_of_all_solutions_triple_reciprocal_l160_160583


namespace walking_speed_is_correct_l160_160324

-- Define the conditions
def time_in_minutes : ℝ := 10
def distance_in_meters : ℝ := 1666.6666666666665
def speed_in_km_per_hr : ℝ := 2.777777777777775

-- Define the theorem to prove
theorem walking_speed_is_correct :
  (distance_in_meters / time_in_minutes) * 60 / 1000 = speed_in_km_per_hr :=
sorry

end walking_speed_is_correct_l160_160324


namespace max_pairs_of_corner_and_squares_l160_160437

def rectangle : ℕ := 3 * 100
def unit_squares_per_pair : ℕ := 4 + 3

-- Given conditions
def conditions := rectangle = 300 ∧ unit_squares_per_pair = 7

-- Proof statement
theorem max_pairs_of_corner_and_squares (h: conditions) : ∃ n, n = 33 ∧ n * unit_squares_per_pair ≤ rectangle := 
sorry

end max_pairs_of_corner_and_squares_l160_160437


namespace matrix_power_B150_l160_160386

open Matrix

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

-- Prove that B^150 = I
theorem matrix_power_B150 : 
  (B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ)) :=
by
  sorry

end matrix_power_B150_l160_160386


namespace minimum_value_is_12_l160_160695

noncomputable def smallest_possible_value (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) : ℝ :=
(a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (a + d)) + (1 / (b + c)) + (1 / (b + d)) + (1 / (c + d)))

theorem minimum_value_is_12 (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) :
  smallest_possible_value a b c d h1 h2 h3 h4 h5 h6 h7 ≥ 12 :=
sorry

end minimum_value_is_12_l160_160695


namespace value_of_M_l160_160698

noncomputable def a : ℝ := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2)
noncomputable def b : ℝ := Real.sqrt (5 - 2 * Real.sqrt 6)
noncomputable def M : ℝ := a - b

theorem value_of_M : M = 4 :=
by
  sorry

end value_of_M_l160_160698


namespace system_solution_l160_160844

theorem system_solution (a x0 : ℝ) (h : a ≠ 0) 
  (h1 : 3 * x0 + 2 * x0 = 15 * a) 
  (h2 : 1 / a * x0 + x0 = 9) 
  : x0 = 6 ∧ a = 2 :=
by {
  sorry
}

end system_solution_l160_160844


namespace circle_area_in_sq_cm_l160_160734

theorem circle_area_in_sq_cm (diameter_meters : ℝ) (h : diameter_meters = 5) : 
  let radius_meters := diameter_meters / 2
  let area_square_meters := π * radius_meters^2
  let area_square_cm := area_square_meters * 10000
  area_square_cm = 62500 * π :=
by
  sorry

end circle_area_in_sq_cm_l160_160734


namespace age_ratio_l160_160165

variable (R D : ℕ)

theorem age_ratio (h1 : D = 24) (h2 : R + 6 = 38) : R / D = 4 / 3 := by
  sorry

end age_ratio_l160_160165


namespace factor_expression_l160_160963

theorem factor_expression (x : ℝ) : 3 * x * (x - 5) + 7 * (x - 5) - 2 * (x - 5) = (3 * x + 5) * (x - 5) :=
by
  sorry

end factor_expression_l160_160963


namespace arithmetic_sequence_sum_six_terms_l160_160169

noncomputable def sum_of_first_six_terms (a : ℤ) (d : ℤ) : ℤ :=
  let a1 := a
  let a2 := a1 + d
  let a3 := a2 + d
  let a4 := a3 + d
  let a5 := a4 + d
  let a6 := a5 + d
  a1 + a2 + a3 + a4 + a5 + a6

theorem arithmetic_sequence_sum_six_terms
  (a3 a4 a5 : ℤ)
  (h3 : a3 = 8)
  (h4 : a4 = 13)
  (h5 : a5 = 18)
  (d : ℤ) (a : ℤ)
  (h_d : d = a4 - a3)
  (h_a : a + 2 * d = 8) :
  sum_of_first_six_terms a d = 63 :=
by
  sorry

end arithmetic_sequence_sum_six_terms_l160_160169


namespace solve_system_l160_160188

theorem solve_system :
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ : ℚ),
  x₁ + 12 * x₂ = 15 ∧
  x₁ - 12 * x₂ + 11 * x₃ = 2 ∧
  x₁ - 11 * x₃ + 10 * x₄ = 2 ∧
  x₁ - 10 * x₄ + 9 * x₅ = 2 ∧
  x₁ - 9 * x₅ + 8 * x₆ = 2 ∧
  x₁ - 8 * x₆ + 7 * x₇ = 2 ∧
  x₁ - 7 * x₇ + 6 * x₈ = 2 ∧
  x₁ - 6 * x₈ + 5 * x₉ = 2 ∧
  x₁ - 5 * x₉ + 4 * x₁₀ = 2 ∧
  x₁ - 4 * x₁₀ + 3 * x₁₁ = 2 ∧
  x₁ - 3 * x₁₁ + 2 * x₁₂ = 2 ∧
  x₁ - 2 * x₁₂ = 2 ∧
  x₁ = 37 / 12 ∧
  x₂ = 143 / 144 ∧
  x₃ = 65 / 66 ∧
  x₄ = 39 / 40 ∧
  x₅ = 26 / 27 ∧
  x₆ = 91 / 96 ∧
  x₇ = 13 / 14 ∧
  x₈ = 65 / 72 ∧
  x₉ = 13 / 15 ∧
  x₁₀ = 13 / 16 ∧
  x₁₁ = 13 / 18 ∧
  x₁₂ = 13 / 24 :=
by
  sorry

end solve_system_l160_160188


namespace largest_even_whole_number_l160_160423

theorem largest_even_whole_number (x : ℕ) (h1 : 9 * x < 150) (h2 : x % 2 = 0) : x ≤ 16 :=
by
  sorry

end largest_even_whole_number_l160_160423


namespace magnitude_of_z_l160_160361

namespace ComplexNumberProof

open Complex

noncomputable def z (b : ℝ) : ℂ := (3 - b * Complex.I) / Complex.I

theorem magnitude_of_z (b : ℝ) (h : (z b).re = (z b).im) : Complex.abs (z b) = 3 * Real.sqrt 2 :=
by
  sorry

end ComplexNumberProof

end magnitude_of_z_l160_160361


namespace factor_of_60n_l160_160454

theorem factor_of_60n
  (n : ℕ)
  (x : ℕ)
  (h_condition1 : ∃ k : ℕ, 60 * n = x * k)
  (h_condition2 : ∃ m : ℕ, 60 * n = 8 * m)
  (h_condition3 : n >= 8) :
  x = 60 :=
sorry

end factor_of_60n_l160_160454


namespace factorize_expression_l160_160224

variable (x y : ℝ)

theorem factorize_expression :
  4 * (x - y + 1) + y * (y - 2 * x) = (y - 2) * (y - 2 - 2 * x) :=
by 
  sorry

end factorize_expression_l160_160224


namespace number_of_k_solutions_l160_160677

theorem number_of_k_solutions :
  ∃ (n : ℕ), n = 1006 ∧
  (∀ k, (∃ a b : ℕ+, (a ≠ b) ∧ (k * (a + b) = 2013 * Nat.lcm a b)) ↔ k ≤ n ∧ 0 < k) :=
by
  sorry

end number_of_k_solutions_l160_160677


namespace pyramid_lateral_surface_area_l160_160705

noncomputable def lateral_surface_area (S : ℝ) (n : ℕ) (α : ℝ) : ℝ :=
  n * S

theorem pyramid_lateral_surface_area (S : ℝ) (n : ℕ) (α : ℝ) (A : ℝ) :
  A = n * S * (Real.cos α) →
  lateral_surface_area S n α = A / (Real.cos α) :=
by
  sorry

end pyramid_lateral_surface_area_l160_160705


namespace range_of_given_function_l160_160955

noncomputable def given_function (x : ℝ) : ℝ :=
  abs (Real.sin x) / (Real.sin x) + Real.cos x / abs (Real.cos x) + abs (Real.tan x) / Real.tan x

theorem range_of_given_function : Set.range given_function = {-1, 3} :=
by
  sorry

end range_of_given_function_l160_160955


namespace suitable_land_for_vegetables_l160_160535

def previous_property_acres : ℝ := 2
def enlargement_factor : ℝ := 10
def pond_area : ℝ := 1

theorem suitable_land_for_vegetables :
  let new_property_acres := previous_property_acres * enlargement_factor in
  let suitable_acres := new_property_acres - pond_area in
  suitable_acres = 19 :=
by
  sorry

end suitable_land_for_vegetables_l160_160535


namespace find_angle_x_l160_160512

def angle_ABC := 124
def angle_BAD := 30
def angle_BDA := 28
def angle_ABD := 180 - angle_ABC
def angle_x := 180 - (angle_BAD + angle_ABD)

theorem find_angle_x : angle_x = 94 :=
by
  repeat { sorry }

end find_angle_x_l160_160512


namespace find_number_l160_160885

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l160_160885


namespace distance_traveled_is_6000_l160_160447

-- Define the conditions and the question in Lean 4
def footprints_per_meter_Pogo := 4
def footprints_per_meter_Grimzi := 3 / 6
def combined_total_footprints := 27000

theorem distance_traveled_is_6000 (D : ℕ) :
  footprints_per_meter_Pogo * D + footprints_per_meter_Grimzi * D = combined_total_footprints →
  D = 6000 :=
by
  sorry

end distance_traveled_is_6000_l160_160447


namespace necessary_condition_inequality_l160_160936

theorem necessary_condition_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, x^2 - 2 * a * x + a > 0 := 
sorry

end necessary_condition_inequality_l160_160936


namespace max_marks_l160_160638

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 92 + 40) : M = 400 :=
by
  sorry

end max_marks_l160_160638


namespace max_eccentricity_of_ellipse_l160_160113

theorem max_eccentricity_of_ellipse 
  (R_large : ℝ)
  (r_cylinder : ℝ)
  (R_small : ℝ)
  (D_centers : ℝ)
  (a : ℝ)
  (b : ℝ)
  (e : ℝ) :
  R_large = 1 → 
  r_cylinder = 1 → 
  R_small = 1/4 → 
  D_centers = 10/3 → 
  a = 5/3 → 
  b = 1 → 
  e = Real.sqrt (1 - (b / a) ^ 2) → 
  e = 4/5 := by 
  sorry

end max_eccentricity_of_ellipse_l160_160113


namespace find_number_eq_seven_point_five_l160_160912

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l160_160912


namespace quadratic_roots_l160_160507

theorem quadratic_roots (a b c : ℝ) :
  ∃ x y : ℝ, (x ≠ y ∧ (x^2 - (a + b) * x + (ab - c^2) = 0) ∧ (y^2 - (a + b) * y + (ab - c^2) = 0)) ∧
  (x = y ↔ a = b ∧ c = 0) := sorry

end quadratic_roots_l160_160507


namespace sum_of_interior_edges_l160_160633

theorem sum_of_interior_edges {f : ℝ} {w : ℝ} (h_frame_area : f = 32) (h_outer_edge : w = 7) (h_frame_width : 2) :
  let i_length := w - 2 * h_frame_width in
  let i_other_length := (f - (w * (w  - 2 * h_frame_width))) / (w  + 2 * h_frame_width) in
  i_length + i_other_length + i_length + i_other_length = 8 :=
by
  let i_length := w - 2 * 2
  let i_other_length := (32 - (i_length * w)) / (w  + 2 * 2)
  let sum := i_length + i_other_length + i_length + i_other_length
  have h_sum : sum = 8, by sorry
  exact h_sum

end sum_of_interior_edges_l160_160633


namespace average_roots_of_quadratic_l160_160329

open Real

theorem average_roots_of_quadratic (a b : ℝ) (h_eq : ∃ x1 x2 : ℝ, a * x1^2 - 2 * a * x1 + b = 0 ∧ a * x2^2 - 2 * a * x2 + b = 0):
  (b = b) → (a ≠ 0) → (h_discriminant : (2 * a)^2 - 4 * a * b ≥ 0) → (x1 + x2) / 2 = 1 :=
by
  sorry

end average_roots_of_quadratic_l160_160329


namespace find_number_eq_seven_point_five_l160_160916

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l160_160916


namespace tangent_line_at_P_l160_160558

def tangent_line_eq (x y : ℝ) : ℝ := x - 2 * y + 1

theorem tangent_line_at_P (x y : ℝ) (h : x ^ 2 + y ^ 2 - 4 * x + 2 * y = 0 ∧ (x, y) = (1, 1)) :
    tangent_line_eq x y = 0 := 
sorry

end tangent_line_at_P_l160_160558


namespace liters_pepsi_144_l160_160198

/-- A drink vendor has 50 liters of Maaza, some liters of Pepsi, and 368 liters of Sprite. -/
def liters_maaza : ℕ := 50
def liters_sprite : ℕ := 368
def num_cans : ℕ := 281

/-- The total number of liters of drinks the vendor has -/
def total_liters (lit_pepsi: ℕ) : ℕ := liters_maaza + lit_pepsi + liters_sprite

/-- Given that the least number of cans required is 281, prove that the liters of Pepsi is 144. -/
theorem liters_pepsi_144 (P : ℕ) (h: total_liters P % num_cans = 0) : P = 144 :=
by
  sorry

end liters_pepsi_144_l160_160198


namespace mean_of_remaining_four_numbers_l160_160854

theorem mean_of_remaining_four_numbers (a b c d : ℝ) (h1 : (a + b + c + d + 106) / 5 = 92) : 
  (a + b + c + d) / 4 = 88.5 := 
sorry

end mean_of_remaining_four_numbers_l160_160854


namespace tan_frac_eq_l160_160353

theorem tan_frac_eq (x : ℝ) (h : Real.tan (x + π / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := 
  sorry

end tan_frac_eq_l160_160353


namespace rowing_rate_in_still_water_l160_160550

theorem rowing_rate_in_still_water (R C : ℝ) 
  (h1 : (R + C) * 2 = 26)
  (h2 : (R - C) * 4 = 26) : 
  R = 26 / 3 :=
by
  sorry

end rowing_rate_in_still_water_l160_160550


namespace cone_volume_l160_160564

def slant_height := 15
def height := 9
def radius := Real.sqrt (slant_height^2 - height^2)
def volume := (1/3) * Real.pi * radius^2 * height

theorem cone_volume : 
  radius = 12 ∧ volume = 432 * Real.pi :=
by
  split
  { -- radius = 12
    unfold radius
    sorry
  }
  { -- volume = 432 * Real.pi
    unfold volume
    sorry
  }

end cone_volume_l160_160564


namespace hunter_movies_count_l160_160342

theorem hunter_movies_count (H : ℕ) 
  (dalton_movies : ℕ := 7)
  (alex_movies : ℕ := 15)
  (together_movies : ℕ := 2)
  (total_movies : ℕ := 30)
  (all_different_movies : dalton_movies + alex_movies - together_movies + H = total_movies) :
  H = 8 :=
by
  -- The mathematical proof will go here
  sorry

end hunter_movies_count_l160_160342


namespace jen_problem_correct_answer_l160_160380

-- Definitions based on the conditions
def sum_178_269 : ℤ := 178 + 269
def round_to_nearest_hundred (n : ℤ) : ℤ :=
  if n % 100 >= 50 then n - (n % 100) + 100 else n - (n % 100)

-- Prove the statement
theorem jen_problem_correct_answer :
  round_to_nearest_hundred sum_178_269 = 400 :=
by
  have h1 : sum_178_269 = 447 := rfl
  have h2 : round_to_nearest_hundred 447 = 400 := by sorry
  exact h2

end jen_problem_correct_answer_l160_160380


namespace fraction_eq_repeating_decimal_l160_160046

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l160_160046


namespace gain_in_meters_l160_160941

noncomputable def cost_price : ℝ := sorry
noncomputable def selling_price : ℝ := 1.5 * cost_price
noncomputable def total_cost_price : ℝ := 30 * cost_price
noncomputable def total_selling_price : ℝ := 30 * selling_price
noncomputable def gain : ℝ := total_selling_price - total_cost_price

theorem gain_in_meters (S C : ℝ) (h_S : S = 1.5 * C) (h_gain : gain = 15 * C) :
  15 * C / S = 10 := by
  sorry

end gain_in_meters_l160_160941


namespace solution_set_ineq_min_value_sum_l160_160241

-- Part (1)
theorem solution_set_ineq (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| + |x - 2|) :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 2} :=
sorry

-- Part (2)
theorem min_value_sum (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| + |x - 2|)
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (hx : ∀ x, f x ≥ (1 / m) + (1 / n)) :
  m + n = 8 / 3 :=
sorry

end solution_set_ineq_min_value_sum_l160_160241


namespace isosceles_triangle_sum_x_l160_160127

noncomputable def sum_possible_values_of_x : ℝ :=
  let x1 : ℝ := 20
  let x2 : ℝ := 50
  let x3 : ℝ := 80
  x1 + x2 + x3

theorem isosceles_triangle_sum_x (x : ℝ) (h1 : x = 20 ∨ x = 50 ∨ x = 80) : sum_possible_values_of_x = 150 :=
  by
    sorry

end isosceles_triangle_sum_x_l160_160127


namespace cost_of_fencing_per_meter_l160_160159

theorem cost_of_fencing_per_meter
  (length breadth : ℕ)
  (total_cost : ℝ)
  (h1 : length = breadth + 20)
  (h2 : length = 60)
  (h3 : total_cost = 5300) :
  (total_cost / (2 * length + 2 * breadth)) = 26.5 := 
by
  sorry

end cost_of_fencing_per_meter_l160_160159


namespace common_point_arithmetic_progression_l160_160949

theorem common_point_arithmetic_progression (a b c : ℝ) (h : 2 * b = a + c) :
  ∃ (x y : ℝ), (∀ x, y = a * x^2 + b * x + c) ∧ x = -2 ∧ y = 0 :=
by
  sorry

end common_point_arithmetic_progression_l160_160949


namespace sum_bn_sequence_l160_160515

theorem sum_bn_sequence {a : ℕ → ℝ} {b : ℕ → ℝ} (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n / (2 * a n + 1))
  (h3 : ∀ n, b n = a n / (2 * n + 1)) :
  ∀ n, ∑ i in range n, b i = n / (2 * n + 1) := by
  sorry

end sum_bn_sequence_l160_160515


namespace determine_C_for_identity_l160_160816

theorem determine_C_for_identity :
  (∀ (x : ℝ), (1/2 * (Real.sin x)^2 + C = -1/4 * Real.cos (2 * x))) → C = -1/4 :=
by
  sorry

end determine_C_for_identity_l160_160816


namespace cheryl_found_more_eggs_l160_160772

theorem cheryl_found_more_eggs :
  let kevin_eggs := 5
  let bonnie_eggs := 13
  let george_eggs := 9
  let cheryl_eggs := 56
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
  sorry

end cheryl_found_more_eggs_l160_160772


namespace bugs_meeting_time_l160_160877

/-- Two circles with radii 7 inches and 3 inches are tangent at a point P. 
Two bugs start crawling at the same time from point P, one along the larger circle 
at 4π inches per minute, and the other along the smaller circle at 3π inches per minute. 
Prove they will meet again after 14 minutes and determine how far each has traveled.

The bug on the larger circle will have traveled 28π inches.
The bug on the smaller circle will have traveled 42π inches.
-/
theorem bugs_meeting_time
  (r₁ r₂ : ℝ) (v₁ v₂ : ℝ)
  (h₁ : r₁ = 7) (h₂ : r₂ = 3) 
  (h₃ : v₁ = 4 * Real.pi) (h₄ : v₂ = 3 * Real.pi) :
  ∃ t d₁ d₂, t = 14 ∧ d₁ = 28 * Real.pi ∧ d₂ = 42 * Real.pi := by
  sorry

end bugs_meeting_time_l160_160877


namespace cricket_average_score_l160_160712

theorem cricket_average_score (A : ℝ)
    (h1 : 3 * 30 = 90)
    (h2 : 5 * 26 = 130) :
    2 * A + 90 = 130 → A = 20 :=
by
  intros h
  linarith

end cricket_average_score_l160_160712


namespace shorter_leg_of_right_triangle_l160_160530

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : a^2 + b^2 = 65^2) (ha : a ≤ b) : a = 25 :=
by sorry

end shorter_leg_of_right_triangle_l160_160530


namespace quadrilateral_angle_E_l160_160531

theorem quadrilateral_angle_E (E F G H : ℝ)
  (h1 : E = 3 * F)
  (h2 : E = 4 * G)
  (h3 : E = 6 * H)
  (h_sum : E + F + G + H = 360) :
  E = 206 :=
by
  sorry

end quadrilateral_angle_E_l160_160531


namespace minimum_path_proof_l160_160758

noncomputable def minimum_path (r : ℝ) (h : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let R := Real.sqrt (r^2 + h^2)
  let theta := 2 * Real.pi * (R / (2 * Real.pi * r))
  let A := (d1, 0)
  let B := (-d2 * Real.cos (theta / 2), -d2 * Real.sin (theta / 2))
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem minimum_path_proof :
  minimum_path 800 (300 * Real.sqrt 3) 150 (450 * Real.sqrt 2) = 562.158 := 
by 
  sorry

end minimum_path_proof_l160_160758


namespace sum_factors_of_30_l160_160613

theorem sum_factors_of_30 : 
  (∑ i in {1, 2, 3, 5, 6, 10, 15, 30}, i) = 72 := 
by
  -- Sorry, to skip the proof here.
  sorry

end sum_factors_of_30_l160_160613


namespace repeating_decimal_eq_fraction_l160_160058

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l160_160058


namespace find_number_l160_160886

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l160_160886


namespace scientific_notation_10200000_l160_160172

theorem scientific_notation_10200000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 10.2 * 10^7 = a * 10^n := 
sorry

end scientific_notation_10200000_l160_160172


namespace leo_current_weight_l160_160183

theorem leo_current_weight (L K : ℕ) 
  (h1 : L + 10 = 3 * K / 2) 
  (h2 : L + K = 160)
  : L = 92 :=
sorry

end leo_current_weight_l160_160183


namespace weight_gain_difference_l160_160019

theorem weight_gain_difference :
  let orlando_gain := 5
  let jose_gain := 2 * orlando_gain + 2
  let total_gain := 20
  let fernando_gain := total_gain - (orlando_gain + jose_gain)
  let half_jose_gain := jose_gain / 2
  half_jose_gain - fernando_gain = 3 :=
by
  sorry

end weight_gain_difference_l160_160019


namespace cherry_sodas_correct_l160_160197

/-
A cooler is filled with 24 cans of cherry soda and orange pop. 
There are twice as many cans of orange pop as there are of cherry soda. 
Prove that the number of cherry sodas is 8.
-/
def num_cherry_sodas (C O : ℕ) : Prop :=
  O = 2 * C ∧ C + O = 24 → C = 8

theorem cherry_sodas_correct (C O : ℕ) (h : O = 2 * C ∧ C + O = 24) : C = 8 :=
by
  sorry

end cherry_sodas_correct_l160_160197


namespace find_min_value_l160_160336

theorem find_min_value : 
  ∃ x : ℝ, (∀ y, (y = x + 4 / x) → y ≠ 4) ∧
           (∀ x : ℝ, ∀ y, (y = -x^2 + 2 * x + 3) → y ≠ 4) ∧
           (∀ x : ℝ, (0 < x ∧ x < real.pi) → ∀ y, (y = real.sin x + 4 / real.sin x) → y ≠ 4) ∧
           ∃ x : ℝ, ∀ y, (y = real.exp x + 4 / real.exp x) → y = 4 := 
by 
  sorry

end find_min_value_l160_160336


namespace negation_of_exists_solution_l160_160718

theorem negation_of_exists_solution :
  ¬ (∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔ ∀ c : ℝ, c > 0 → ¬ (∃ x : ℝ, x^2 - x + c = 0) :=
by
  sorry

end negation_of_exists_solution_l160_160718


namespace lines_intersection_l160_160305

theorem lines_intersection (a b : ℝ) : 
  (2 : ℝ) = (1/3 : ℝ) * (1 : ℝ) + a →
  (1 : ℝ) = (1/3 : ℝ) * (2 : ℝ) + b →
  a + b = 2 := 
by
  intros h₁ h₂
  sorry

end lines_intersection_l160_160305


namespace product_of_solutions_l160_160595

theorem product_of_solutions {-1/2}
  (x : ℝ) (h : x + 1 / x = 3 * x) :
  (∃ (x1 x2 : ℕ), 
    x1 = √2 / 2 ∧ x2 = -√2 / 2 ∧ 
    (x1 * x2 = -1/2)) :=
begin
  sorry
end

end product_of_solutions_l160_160595


namespace smallest_diff_of_YZ_XY_l160_160438

theorem smallest_diff_of_YZ_XY (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 2509) (h4 : a + b > c) (h5 : b + c > a) (h6 : a + c > b) : b - a = 1 :=
by {
  sorry
}

end smallest_diff_of_YZ_XY_l160_160438


namespace binary_representation_l160_160306

theorem binary_representation (n : ℕ) (h1 : n % 17 = 0) (h2 : (nat.popcount n = 3)) :
  let zcount := nat.bits n - 3 in
  (6 ≤ zcount) ∧ (zcount = 7 → even n) :=
by
  let zcount := nat.bits n - 3
  have zero_count_six_or_more : 6 ≤ zcount := sorry
  have zero_count_seven_even : zcount = 7 → even n := sorry
  exact ⟨zero_count_six_or_more, zero_count_seven_even⟩

end binary_representation_l160_160306


namespace convex_n_hedral_angle_l160_160726

theorem convex_n_hedral_angle (n : ℕ) 
  (sum_plane_angles : ℝ) (sum_dihedral_angles : ℝ) 
  (h1 : sum_plane_angles = sum_dihedral_angles)
  (h2 : sum_plane_angles < 2 * Real.pi)
  (h3 : sum_dihedral_angles > (n - 2) * Real.pi) :
  n = 3 := 
by 
  sorry

end convex_n_hedral_angle_l160_160726


namespace repeating_fraction_equality_l160_160084

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l160_160084


namespace find_m_l160_160244

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, 2)
def vec_b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the addition of vectors
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

-- State the main theorem without proof
theorem find_m (m : ℝ) : dot_product (vec_add vec_a (vec_b m)) vec_a = 0 ↔ m = -7/2 := by
  sorry

end find_m_l160_160244


namespace jake_peaches_is_seven_l160_160131

-- Definitions based on conditions
def steven_peaches : ℕ := 13
def jake_peaches (steven : ℕ) : ℕ := steven - 6

-- The theorem we want to prove
theorem jake_peaches_is_seven : jake_peaches steven_peaches = 7 := sorry

end jake_peaches_is_seven_l160_160131


namespace sum_of_7_terms_arithmetic_seq_l160_160128

variable {α : Type*} [LinearOrderedField α]

def arithmetic_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_7_terms_arithmetic_seq (a : ℕ → α) (h_arith : arithmetic_seq a)
  (h_a4 : a 4 = 2) :
  (7 * (a 1 + a 7)) / 2 = 14 :=
sorry

end sum_of_7_terms_arithmetic_seq_l160_160128


namespace ratio_w_y_l160_160429

open Real

theorem ratio_w_y (w x y z : ℝ) (h1 : w / x = 5 / 2) (h2 : y / z = 3 / 2) (h3 : z / x = 1 / 4) : w / y = 20 / 3 :=
by
  sorry

end ratio_w_y_l160_160429


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160902

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160902


namespace convert_to_base10_sum_l160_160491

def base8_to_dec (d2 d1 d0 : Nat) : Nat :=
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

def base13_to_dec (d2 d1 d0 : Nat) : Nat :=
  d2 * 13^2 + d1 * 13^1 + d0 * 13^0

def convert_537_8 : Nat :=
  base8_to_dec 5 3 7

def convert_4C5_13 : Nat :=
  base13_to_dec 4 12 5

theorem convert_to_base10_sum : 
  convert_537_8 + convert_4C5_13 = 1188 := 
by 
  sorry

end convert_to_base10_sum_l160_160491


namespace learn_at_least_537_words_l160_160814

theorem learn_at_least_537_words (total_words : ℕ) (guess_percentage : ℝ) (required_percentage : ℝ) :
  total_words = 600 → guess_percentage = 0.05 → required_percentage = 0.90 → 
  ∀ (words_learned : ℕ), words_learned ≥ 537 → 
  (words_learned + guess_percentage * (total_words - words_learned)) / total_words ≥ required_percentage :=
by
  intros h_total_words h_guess_percentage h_required_percentage words_learned h_words_learned
  sorry

end learn_at_least_537_words_l160_160814


namespace joe_time_to_school_l160_160271

theorem joe_time_to_school
    (r_w : ℝ) -- Joe's walking speed
    (t_w : ℝ) -- Time to walk halfway
    (t_stop : ℝ) -- Time stopped at the store
    (r_running_factor : ℝ) -- Factor by which running speed is faster than walking speed
    (initial_walk_time_halfway : t_w = 10)
    (store_stop_time : t_stop = 3)
    (running_speed_factor : r_running_factor = 4) :
    t_w + t_stop + t_w / r_running_factor = 15.5 :=
by
    -- Implementation skipped, just verifying statement is correctly captured
    sorry

end joe_time_to_school_l160_160271


namespace axis_of_symmetry_exists_l160_160716

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

theorem axis_of_symmetry_exists :
  ∃ k : ℤ, ∃ x : ℝ, (x = -5 * Real.pi / 12 ∧ f x = Real.sin (Real.pi / 2 + k * Real.pi))
  ∨ (x = Real.pi / 12 + k * Real.pi / 2 ∧ f x = Real.sin (Real.pi / 2 + k * Real.pi)) :=
sorry

end axis_of_symmetry_exists_l160_160716


namespace solution_set_inequality_l160_160926

theorem solution_set_inequality (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ |2 * x - a| + a ≤ 6) → a = 2 :=
sorry

end solution_set_inequality_l160_160926


namespace consecutive_integer_cubes_sum_l160_160162

theorem consecutive_integer_cubes_sum : 
  ∀ (a : ℕ), 
  (a > 2) → 
  (a - 1) * a * (a + 1) * (a + 2) = 12 * ((a - 1) + a + (a + 1) + (a + 2)) →
  ((a - 1)^3 + a^3 + (a + 1)^3 + (a + 2)^3) = 224 :=
by
  intro a ha h
  sorry

end consecutive_integer_cubes_sum_l160_160162


namespace solve_fractional_equation_l160_160971

theorem solve_fractional_equation (x : ℝ) :
  (x ≠ 5) ∧ (x ≠ 6) ∧ ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)) / ((x - 5) * (x - 6) * (x - 5)) = 1 ↔ 
  x = 2 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
sorry

end solve_fractional_equation_l160_160971


namespace integer_count_in_interval_l160_160813

theorem integer_count_in_interval : 
  let lower_bound := Int.floor (-7 * Real.pi)
  let upper_bound := Int.ceil (12 * Real.pi)
  upper_bound - lower_bound + 1 = 61 :=
by
  let lower_bound := Int.floor (-7 * Real.pi)
  let upper_bound := Int.ceil (12 * Real.pi)
  have : upper_bound - lower_bound + 1 = 61 := sorry
  exact this

end integer_count_in_interval_l160_160813


namespace length_of_train_l160_160449

-- declare constants
variables (L S : ℝ)

-- state conditions
def condition1 : Prop := L = S * 50
def condition2 : Prop := L + 500 = S * 100

-- state the theorem to prove
theorem length_of_train (h1 : condition1 L S) (h2 : condition2 L S) : L = 500 :=
by sorry

end length_of_train_l160_160449


namespace sqrt_529000_pow_2_5_l160_160953

theorem sqrt_529000_pow_2_5 : (529000 ^ (1 / 2) ^ (5 / 2)) = 14873193 := by
  sorry

end sqrt_529000_pow_2_5_l160_160953


namespace inequality_range_l160_160717

theorem inequality_range (a : ℝ) : (∀ x : ℝ, x^2 - 1 ≥ a * |x - 1|) → a ≤ -2 :=
by
  sorry

end inequality_range_l160_160717


namespace jerry_showers_l160_160653

theorem jerry_showers :
  ∀ gallons_total gallons_drinking_cooking gallons_per_shower pool_length pool_width pool_height gallons_per_cubic_foot,
    gallons_total = 1000 →
    gallons_drinking_cooking = 100 →
    gallons_per_shower = 20 →
    pool_length = 10 →
    pool_width = 10 →
    pool_height = 6 →
    gallons_per_cubic_foot = 1 →
    let pool_volume := pool_length * pool_width * pool_height in
    pool_volume = 600 →
    let gallons_for_pool := pool_volume * gallons_per_cubic_foot in
    let gallons_for_showers := gallons_total - gallons_drinking_cooking - gallons_for_pool in
    let number_of_showers := gallons_for_showers / gallons_per_shower in
    number_of_showers = 15 :=
by
  intros
  sorry

end jerry_showers_l160_160653


namespace find_three_digit_perfect_square_l160_160032

noncomputable def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n % 100) / 10) * (n % 10)

theorem find_three_digit_perfect_square :
  ∃ (n H : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (n = H * H) ∧ (digit_product n = H - 1) :=
by {
  sorry
}

end find_three_digit_perfect_square_l160_160032


namespace solve_for_q_l160_160413

theorem solve_for_q (k l q : ℝ) 
  (h1 : 3 / 4 = k / 48)
  (h2 : 3 / 4 = (k + l) / 56)
  (h3 : 3 / 4 = (q - l) / 160) :
  q = 126 :=
  sorry

end solve_for_q_l160_160413


namespace factorize_expression_l160_160031

variable (a x : ℝ)

theorem factorize_expression : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := 
by 
  sorry

end factorize_expression_l160_160031


namespace factorize_x_cube_minus_9x_l160_160782

-- Lean statement: Prove that x^3 - 9x = x(x+3)(x-3)
theorem factorize_x_cube_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
by
  sorry

end factorize_x_cube_minus_9x_l160_160782


namespace sum_A_B_C_zero_l160_160282

noncomputable def poly : Polynomial ℝ := Polynomial.X^3 - 16 * Polynomial.X^2 + 72 * Polynomial.X - 27

noncomputable def exists_real_A_B_C 
  (p q r: ℝ) (hpqr: p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hrootsp: Polynomial.eval p poly = 0) (hrootsq: Polynomial.eval q poly = 0)
  (hrootsr: Polynomial.eval r poly = 0) :
  ∃ (A B C: ℝ), (∀ s, s ≠ p → s ≠ q → s ≠ r → (1 / (s^3 - 16*s^2 + 72*s - 27) = (A / (s - p)) + (B / (s - q)) + (C / (s - r)))) := sorry

theorem sum_A_B_C_zero 
  {p q r: ℝ} (hpqr: p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hrootsp: Polynomial.eval p poly = 0) (hrootsq: Polynomial.eval q poly = 0)
  (hrootsr: Polynomial.eval r poly = 0) 
  (hABC: ∃ (A B C: ℝ), (∀ s, s ≠ p → s ≠ q → s ≠ r → (1 / (s^3 - 16*s^2 + 72*s - 27) = (A / (s - p)) + (B / (s - q)) + (C / (s - r))))) :
  ∀ A B C, A + B + C = 0 := sorry

end sum_A_B_C_zero_l160_160282


namespace full_price_ticket_revenue_l160_160015

theorem full_price_ticket_revenue (f t : ℕ) (p : ℝ) 
  (h1 : f + t = 160) 
  (h2 : f * p + t * (p / 3) = 2500) 
  (h3 : p = 30) :
  f * p = 1350 := 
by sorry

end full_price_ticket_revenue_l160_160015


namespace total_drink_ounces_l160_160833

def total_ounces_entire_drink (coke_parts sprite_parts md_parts coke_ounces : ℕ) : ℕ :=
  let total_parts := coke_parts + sprite_parts + md_parts
  let ounces_per_part := coke_ounces / coke_parts
  total_parts * ounces_per_part

theorem total_drink_ounces (coke_parts sprite_parts md_parts coke_ounces : ℕ) (coke_cond : coke_ounces = 8) (parts_cond : coke_parts = 4 ∧ sprite_parts = 2 ∧ md_parts = 5) : 
  total_ounces_entire_drink coke_parts sprite_parts md_parts coke_ounces = 22 :=
by
  sorry

end total_drink_ounces_l160_160833


namespace fraction_of_repeating_decimal_l160_160076

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l160_160076


namespace no_three_natural_numbers_l160_160344

theorem no_three_natural_numbers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
    (h4 : b ∣ a^2 - 1) (h5 : a ∣ c^2 - 1) (h6 : b ∣ c^2 - 1) : false :=
by
  sorry

end no_three_natural_numbers_l160_160344


namespace max_ab_l160_160988

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ab ≤ 1 / 16 :=
by
  sorry

end max_ab_l160_160988


namespace recurring_decimal_to_fraction_l160_160080

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l160_160080


namespace blue_balls_balance_l160_160288

variables {R B O P : ℝ}

-- Given conditions
def cond1 : 4 * R = 8 * B := sorry
def cond2 : 3 * O = 7 * B := sorry
def cond3 : 8 * B = 6 * P := sorry

-- Proof problem: proving equal balance of 5 red balls, 3 orange balls, and 4 purple balls
theorem blue_balls_balance : 5 * R + 3 * O + 4 * P = (67 / 3) * B :=
by
  sorry

end blue_balls_balance_l160_160288


namespace part_I_part_II_l160_160811

-- Condition definitions:
def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 2|

-- Part I: Prove m = 1
theorem part_I (m : ℝ) : (∀ x : ℝ, f (x + 2) m ≥ 0) ↔ m = 1 :=
by
  sorry

-- Part II: Prove a + 2b + 3c ≥ 9
theorem part_II (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : a + 2 * b + 3 * c ≥ 9 :=
by
  sorry

end part_I_part_II_l160_160811


namespace ann_fare_90_miles_l160_160761

-- Define the conditions as given in the problem
def fare (distance : ℕ) : ℕ := 30 + distance * 2

-- Theorem statement
theorem ann_fare_90_miles : fare 90 = 210 := by
  sorry

end ann_fare_90_miles_l160_160761


namespace repeating_decimal_eq_fraction_l160_160056

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l160_160056


namespace number_of_boys_l160_160713

theorem number_of_boys (n : ℕ)
    (incorrect_avg_weight : ℝ)
    (misread_weight new_weight : ℝ)
    (correct_avg_weight : ℝ)
    (h1 : incorrect_avg_weight = 58.4)
    (h2 : misread_weight = 56)
    (h3 : new_weight = 66)
    (h4 : correct_avg_weight = 58.9)
    (h5 : n * correct_avg_weight = n * incorrect_avg_weight + (new_weight - misread_weight)) :
  n = 20 := by
  sorry

end number_of_boys_l160_160713


namespace Katrina_sold_in_morning_l160_160133

theorem Katrina_sold_in_morning :
  ∃ M : ℕ, (120 - 57 - 16 - 11) = M := sorry

end Katrina_sold_in_morning_l160_160133


namespace monkey_climb_ladder_l160_160326

theorem monkey_climb_ladder (n : ℕ) 
  (h1 : ∀ k, (k % 18 = 0 → (k - 18 + 10) % 26 = 8))
  (h2 : ∀ m, (m % 10 = 0 → (m - 10 + 18) % 26 = 18))
  (h3 : ∀ l, (l % 18 = 0 ∧ l % 10 = 0 → l = 0 ∨ l = 26)):
  n = 26 :=
by
  sorry

end monkey_climb_ladder_l160_160326


namespace mean_of_remaining_quiz_scores_l160_160164

theorem mean_of_remaining_quiz_scores (k : ℕ) (hk : k > 12) 
  (mean_k : ℝ) (mean_12 : ℝ) 
  (mean_class : mean_k = 8) 
  (mean_12_group : mean_12 = 14) 
  (mean_correct : mean_12 * 12 + mean_k * (k - 12) = 8 * k) :
  mean_k * (k - 12) = (8 * k - 168) := 
by {
  sorry
}

end mean_of_remaining_quiz_scores_l160_160164


namespace teacher_buys_total_21_pens_l160_160853

def num_black_pens : Nat := 7
def num_blue_pens : Nat := 9
def num_red_pens : Nat := 5
def total_pens : Nat := num_black_pens + num_blue_pens + num_red_pens

theorem teacher_buys_total_21_pens : total_pens = 21 := 
by
  unfold total_pens num_black_pens num_blue_pens num_red_pens
  rfl -- reflexivity (21 = 21)

end teacher_buys_total_21_pens_l160_160853


namespace inequality_holds_for_gt_sqrt2_l160_160846

theorem inequality_holds_for_gt_sqrt2 (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by {
  sorry
}

end inequality_holds_for_gt_sqrt2_l160_160846


namespace units_digit_m_sq_plus_2_m_l160_160137

def m := 2017^2 + 2^2017

theorem units_digit_m_sq_plus_2_m (m := 2017^2 + 2^2017) : (m^2 + 2^m) % 10 = 3 := 
by
  sorry

end units_digit_m_sq_plus_2_m_l160_160137


namespace repeating_decimal_is_fraction_l160_160064

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l160_160064


namespace calculation_l160_160210

theorem calculation : 
  let a := 20 / 9 
  let b := -53 / 4 
  (⌈ a * ⌈ b ⌉ ⌉ - ⌊ a * ⌊ b ⌋ ⌋) = 4 :=
by
  sorry

end calculation_l160_160210


namespace ab_sum_l160_160370

theorem ab_sum (a b : ℕ) (h1: (a + b) % 9 = 8) (h2: (a - b) % 11 = 7) : a + b = 8 :=
sorry

end ab_sum_l160_160370


namespace numberOfChromiumAtoms_l160_160751

noncomputable def molecularWeightOfCompound : ℕ := 296
noncomputable def atomicWeightOfPotassium : ℝ := 39.1
noncomputable def atomicWeightOfOxygen : ℝ := 16.0
noncomputable def atomicWeightOfChromium : ℝ := 52.0

def numberOfPotassiumAtoms : ℕ := 2
def numberOfOxygenAtoms : ℕ := 7

theorem numberOfChromiumAtoms
    (mw : ℕ := molecularWeightOfCompound)
    (awK : ℝ := atomicWeightOfPotassium)
    (awO : ℝ := atomicWeightOfOxygen)
    (awCr : ℝ := atomicWeightOfChromium)
    (numK : ℕ := numberOfPotassiumAtoms)
    (numO : ℕ := numberOfOxygenAtoms) :
  numK * awK + numO * awO + (mw - (numK * awK + numO * awO)) / awCr = 2 := 
by
  sorry

end numberOfChromiumAtoms_l160_160751


namespace repeating_fraction_equality_l160_160087

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l160_160087


namespace least_number_of_cubes_is_10_l160_160331

noncomputable def volume_of_block (length width height : ℕ) : ℕ :=
  length * width * height

noncomputable def gcd_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

noncomputable def volume_of_cube (side : ℕ) : ℕ :=
  side ^ 3

noncomputable def least_number_of_cubes (length width height : ℕ) : ℕ := 
  volume_of_block length width height / volume_of_cube (gcd_three_numbers length width height)

theorem least_number_of_cubes_is_10 : least_number_of_cubes 15 30 75 = 10 := by
  sorry

end least_number_of_cubes_is_10_l160_160331


namespace chessboard_polygon_l160_160187

-- Conditions
variable (A B a b : ℕ)

-- Statement of the theorem
theorem chessboard_polygon (A B a b : ℕ) : A - B = 4 * (a - b) :=
sorry

end chessboard_polygon_l160_160187


namespace equivalent_statements_l160_160768
  
variables {A B : Prop}

theorem equivalent_statements :
  ((A ∧ B) → ¬ (A ∨ B)) ↔ ((A ∨ B) → ¬ (A ∧ B)) :=
sorry

end equivalent_statements_l160_160768


namespace find_point_B_l160_160237

theorem find_point_B (A B : ℝ) (h1 : A = 2) (h2 : abs (B - A) = 5) : B = -3 ∨ B = 7 :=
by
  -- This is where the proof steps would go, but we can skip it with sorry.
  sorry

end find_point_B_l160_160237


namespace second_term_of_series_l160_160469

noncomputable def geometric_series_second_term (a r S : ℝ) := r * a

theorem second_term_of_series (a r : ℝ) (S : ℝ) (hr : r = 1/4) (hs : S = 16) 
  (hS_formula : S = a / (1 - r)) : geometric_series_second_term a r S = 3 :=
by
  -- Definitions are in place, applying algebraic manipulation steps here would follow
  sorry

end second_term_of_series_l160_160469


namespace girls_together_count_l160_160228

-- Define the problem conditions
def boys : ℕ := 4
def girls : ℕ := 2
def total_entities : ℕ := boys + (girls - 1) -- One entity for the two girls together

-- Calculate the factorial
noncomputable def factorial (n: ℕ) : ℕ :=
  if n = 0 then 1 else (List.range (n+1)).foldl (λx y => x * y) 1

-- Define the total number of ways girls can be together
noncomputable def ways_girls_together : ℕ :=
  factorial total_entities * factorial girls

-- State the theorem that needs to be proved
theorem girls_together_count : ways_girls_together = 240 := by
  sorry

end girls_together_count_l160_160228


namespace final_amount_after_bets_l160_160328

theorem final_amount_after_bets :
  let initial_amount := 128
  let num_bets := 8
  let num_wins := 4
  let num_losses := 4
  let bonus_per_win_after_loss := 10
  let win_multiplier := 3 / 2
  let loss_multiplier := 1 / 2
  ∃ final_amount : ℝ,
    (final_amount =
      initial_amount * (win_multiplier ^ num_wins) * (loss_multiplier ^ num_losses) + 2 * bonus_per_win_after_loss) ∧
    final_amount = 60.5 :=
sorry

end final_amount_after_bets_l160_160328


namespace pieces_of_paper_picked_up_l160_160545

theorem pieces_of_paper_picked_up (Olivia : ℕ) (Edward : ℕ) (h₁ : Olivia = 16) (h₂ : Edward = 3) : Olivia + Edward = 19 :=
by
  sorry

end pieces_of_paper_picked_up_l160_160545


namespace probability_after_50_bell_rings_l160_160409

noncomputable def game_probability : ℝ :=
  let p_keep_money := (1 : ℝ) / 4
  let p_give_money := (3 : ℝ) / 4
  let p_same_distribution := p_keep_money^3 + 2 * p_give_money^3
  p_same_distribution^50

theorem probability_after_50_bell_rings : abs (game_probability - 0.002) < 0.01 :=
by
  sorry

end probability_after_50_bell_rings_l160_160409


namespace maximum_profit_l160_160932

noncomputable def profit (x : ℝ) : ℝ :=
  5.06 * x - 0.15 * x^2 + 2 * (15 - x)

theorem maximum_profit : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 15 ∧ profit x = 45.6 :=
by
  sorry

end maximum_profit_l160_160932


namespace sum_of_distinct_integers_l160_160391

theorem sum_of_distinct_integers (a b c d e : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) 
(h_prod : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120) : 
a + b + c + d + e = 33 := 
sorry

end sum_of_distinct_integers_l160_160391


namespace barbara_wins_l160_160335

theorem barbara_wins (n : ℕ) (h : n = 15) (num_winning_sequences : ℕ) :
  num_winning_sequences = 8320 :=
sorry

end barbara_wins_l160_160335


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l160_160602

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l160_160602


namespace repeating_decimal_as_fraction_l160_160049

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l160_160049


namespace motorist_travel_distance_l160_160458

def total_distance_traveled (time_first_half time_second_half speed_first_half speed_second_half : ℕ) : ℕ :=
  (speed_first_half * time_first_half) + (speed_second_half * time_second_half)

theorem motorist_travel_distance :
  total_distance_traveled 3 3 60 48 = 324 :=
by sorry

end motorist_travel_distance_l160_160458


namespace min_sum_ab_max_product_ab_l160_160795

theorem min_sum_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) : a + b ≥ 2 :=
by
  sorry

theorem max_product_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : a * b ≤ 1 / 4 :=
by
  sorry

end min_sum_ab_max_product_ab_l160_160795


namespace grains_in_batch_l160_160553

-- Define the given constants from the problem
def total_rice_shi : ℕ := 1680
def sample_total_grains : ℕ := 250
def sample_containing_grains : ℕ := 25

-- Define the statement to be proven
theorem grains_in_batch : (total_rice_shi * (sample_containing_grains / sample_total_grains)) = 168 := by
  -- Proof steps will go here
  sorry

end grains_in_batch_l160_160553


namespace max_acute_angles_l160_160177

theorem max_acute_angles (n : ℕ) : 
  ∃ k : ℕ, k ≤ (2 * n / 3) + 1 :=
sorry

end max_acute_angles_l160_160177


namespace fraction_of_repeating_decimal_l160_160072

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l160_160072


namespace fraction_addition_target_l160_160315

open Rat

theorem fraction_addition_target (n : ℤ) : 
  (4 + n) / (7 + n) = 3 / 4 → 
  n = 5 := 
by
  intro h
  sorry

end fraction_addition_target_l160_160315


namespace fraction_for_repeating_56_l160_160040

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l160_160040


namespace find_sum_l160_160520

theorem find_sum (x y : ℝ) (h₁ : 3 * |x| + 2 * x + y = 20) (h₂ : 2 * x + 3 * |y| - y = 30) : x + y = 15 :=
sorry

end find_sum_l160_160520


namespace lines_intersect_sum_c_d_l160_160425

theorem lines_intersect_sum_c_d (c d : ℝ) 
    (h1 : ∃ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d) 
    (h2 : ∀ x y : ℝ, x = 3 ∧ y = 3) : 
    c + d = 4 :=
by sorry

end lines_intersect_sum_c_d_l160_160425


namespace inequality_holds_for_gt_sqrt2_l160_160847

theorem inequality_holds_for_gt_sqrt2 (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by {
  sorry
}

end inequality_holds_for_gt_sqrt2_l160_160847


namespace product_trippled_when_added_to_reciprocal_l160_160593

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l160_160593


namespace intersection_A_B_solution_inequalities_l160_160805

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}
def C : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = C :=
by
  sorry

theorem solution_inequalities (x : ℝ) :
  (2 * x^2 + x - 1 > 0) ↔ (x < -1 ∨ x > 1/2) :=
by
  sorry

end intersection_A_B_solution_inequalities_l160_160805


namespace students_per_bus_correct_l160_160724

def total_students : ℝ := 28
def number_of_buses : ℝ := 2.0
def students_per_bus : ℝ := 14

theorem students_per_bus_correct :
  total_students / number_of_buses = students_per_bus := 
by
  -- Proof should go here
  sorry

end students_per_bus_correct_l160_160724


namespace min_value_9x_plus_3y_l160_160517

noncomputable def minimum_value_of_expression : ℝ := 6

theorem min_value_9x_plus_3y (x y : ℝ) 
  (h1 : (x - 1) * 4 + 2 * y = 0) 
  (ha : ∃ (a1 a2 : ℝ), (a1, a2) = (x - 1, 2)) 
  (hb : ∃ (b1 b2 : ℝ), (b1, b2) = (4, y)) : 
  9^x + 3^y = minimum_value_of_expression :=
by
  sorry

end min_value_9x_plus_3y_l160_160517


namespace sugar_needed_l160_160276

variable (a b c d : ℝ)
variable (H1 : a = 2)
variable (H2 : b = 1)
variable (H3 : d = 5)

theorem sugar_needed (c : ℝ) : c = 2.5 :=
by
  have H : 2 / 1 = 5 / c := by {
    sorry
  }
  sorry

end sugar_needed_l160_160276


namespace triangle_inequality_l160_160697

variable {a b c : ℝ}

theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (habc1 : a + b > c) (habc2 : a + c > b) (habc3 : b + c > a) :
  (a / (b + c) + b / (c + a) + c / (a + b) < 2) :=
sorry

end triangle_inequality_l160_160697


namespace fraction_difference_l160_160642

theorem fraction_difference :
  (18/42 : ℚ) - (3/11 : ℚ) = 12/77 := 
by
  -- The proof can be filled in here, but is omitted for the exercise.
  sorry

end fraction_difference_l160_160642


namespace gcd_apb_ab_eq1_gcd_aplusb_aminsb_l160_160519

theorem gcd_apb_ab_eq1 (a b : ℤ) (h : Int.gcd a b = 1) : 
  Int.gcd (a + b) (a * b) = 1 ∧ Int.gcd (a - b) (a * b) = 1 := by
  sorry

theorem gcd_aplusb_aminsb (a b : ℤ) (h : Int.gcd a b = 1) : 
  Int.gcd (a + b) (a - b) = 1 ∨ Int.gcd (a + b) (a - b) = 2 := by
  sorry

end gcd_apb_ab_eq1_gcd_aplusb_aminsb_l160_160519


namespace three_digit_number_count_correct_l160_160365

def number_of_three_digit_numbers_with_repetition (digit_count : ℕ) (positions : ℕ) : ℕ :=
  let choices_for_repeated_digit := 5  -- 5 choices for repeated digit
  let ways_to_place_repeated_digit := 3 -- 3 ways to choose positions
  let choices_for_remaining_digit := 4 -- 4 choices for the remaining digit
  choices_for_repeated_digit * ways_to_place_repeated_digit * choices_for_remaining_digit

theorem three_digit_number_count_correct :
  number_of_three_digit_numbers_with_repetition 5 3 = 60 := 
sorry

end three_digit_number_count_correct_l160_160365


namespace probability_three_different_suits_l160_160382

noncomputable def pinochle_deck := 48
noncomputable def total_cards := 48
noncomputable def different_suits_probability := (36 / 47) * (23 / 46)

theorem probability_three_different_suits :
  different_suits_probability = 414 / 1081 :=
sorry

end probability_three_different_suits_l160_160382


namespace find_number_divided_by_3_equals_subtracted_5_l160_160899

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l160_160899


namespace fuel_consumption_l160_160655

open Real

theorem fuel_consumption (initial_fuel : ℝ) (final_fuel : ℝ) (distance_covered : ℝ) (consumption_rate : ℝ) (fuel_left : ℝ) (x : ℝ) :
  initial_fuel = 60 ∧ final_fuel = 50 ∧ distance_covered = 100 ∧ 
  consumption_rate = (initial_fuel - final_fuel) / distance_covered ∧ consumption_rate = 0.1 ∧ 
  fuel_left = initial_fuel - consumption_rate * x ∧ x = 260 →
  fuel_left = 34 :=
by
  sorry

end fuel_consumption_l160_160655


namespace compare_negatives_l160_160952

theorem compare_negatives : -1 < - (2 / 3) := by
  sorry

end compare_negatives_l160_160952


namespace competition_inequality_l160_160528

variable (a b k : ℕ)

-- Conditions
variable (h1 : b % 2 = 1) 
variable (h2 : b ≥ 3)
variable (h3 : ∀ (J1 J2 : ℕ), J1 ≠ J2 → ∃ num_students : ℕ, num_students ≤ a ∧ num_students ≤ k)

theorem competition_inequality (h1: b % 2 = 1) (h2: b ≥ 3) (h3: ∀ (J1 J2 : ℕ), J1 ≠ J2 → ∃ num_students : ℕ, num_students ≤ a ∧ num_students ≤ k) :
  (k: ℝ) / (a: ℝ) ≥ (b-1: ℝ) / (2*b: ℝ) := sorry

end competition_inequality_l160_160528


namespace triangle_angle_not_greater_than_60_l160_160704

theorem triangle_angle_not_greater_than_60 (A B C : Real) (h1 : A + B + C = 180) 
  : A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
by {
  sorry
}

end triangle_angle_not_greater_than_60_l160_160704


namespace reciprocal_sum_of_roots_l160_160985

theorem reciprocal_sum_of_roots :
  (∃ m n : ℝ, (m^2 + 2 * m - 3 = 0) ∧ (n^2 + 2 * n - 3 = 0) ∧ m ≠ n) →
  (∃ m n : ℝ, (1/m + 1/n = 2/3)) :=
by
  sorry

end reciprocal_sum_of_roots_l160_160985


namespace meaningful_expression_l160_160419

theorem meaningful_expression (x : ℝ) : (1 / Real.sqrt (x + 2) > 0) → (x > -2) := 
sorry

end meaningful_expression_l160_160419


namespace sum_of_first_8_terms_l160_160725

noncomputable def sum_of_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ :=
a * (1 - r^n) / (1 - r)

theorem sum_of_first_8_terms 
  (a r : ℝ)
  (h₁ : sum_of_geometric_sequence a r 4 = 5)
  (h₂ : sum_of_geometric_sequence a r 12 = 35) :
  sum_of_geometric_sequence a r 8 = 15 := 
sorry

end sum_of_first_8_terms_l160_160725


namespace jane_nail_polish_drying_time_l160_160379

theorem jane_nail_polish_drying_time :
  let base_coat := 4
  let color_coat_1 := 5
  let color_coat_2 := 6
  let color_coat_3 := 7
  let index_finger_1 := 8
  let index_finger_2 := 10
  let middle_finger := 12
  let ring_finger := 11
  let pinky_finger := 14
  let top_coat := 9
  base_coat + color_coat_1 + color_coat_2 + color_coat_3 + index_finger_1 + index_finger_2 + middle_finger + ring_finger + pinky_finger + top_coat = 86 :=
by sorry

end jane_nail_polish_drying_time_l160_160379


namespace find_number_divided_by_3_equals_subtracted_5_l160_160897

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l160_160897


namespace product_of_tripled_reciprocals_eq_neg_half_l160_160608

theorem product_of_tripled_reciprocals_eq_neg_half :
  (∀ x : ℝ, x + 1/x = 3*x → x = sqrt(2)/2 ∨ x = -sqrt(2)/2 ∧ ∏ (a : ℝ) [a = sqrt(2)/2 ∨ a = -sqrt(2)/2], a = -1/2) :=
by
  sorry

end product_of_tripled_reciprocals_eq_neg_half_l160_160608


namespace golden_ratio_eqn_value_of_ab_value_of_pq_n_l160_160163

-- Part (1): Finding the golden ratio
theorem golden_ratio_eqn {x : ℝ} (h1 : x^2 + x - 1 = 0) : x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
sorry

-- Part (2): Finding the value of ab
theorem value_of_ab {a b m : ℝ} (h1 : a^2 + m * a = 1) (h2 : b^2 - 2 * m * b = 4) (h3 : b ≠ -2 * a) : a * b = 2 :=
sorry

-- Part (3): Finding the value of pq - n
theorem value_of_pq_n {p q n : ℝ} (h1 : p ≠ q) (eq1 : p^2 + n * p - 1 = q) (eq2 : q^2 + n * q - 1 = p) : p * q - n = 0 :=
sorry

end golden_ratio_eqn_value_of_ab_value_of_pq_n_l160_160163


namespace ben_examined_7_trays_l160_160765

open Int

def trays_of_eggs (total_eggs : ℕ) (eggs_per_tray : ℕ) : ℕ := total_eggs / eggs_per_tray

theorem ben_examined_7_trays : trays_of_eggs 70 10 = 7 :=
by
  sorry

end ben_examined_7_trays_l160_160765


namespace find_number_l160_160889

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l160_160889


namespace tank_capacity_l160_160933

-- Define the initial fullness of the tank and the total capacity
def initial_fullness (w c : ℝ) : Prop :=
  w = c / 5

-- Define the fullness of the tank after adding 5 liters
def fullness_after_adding (w c : ℝ) : Prop :=
  (w + 5) / c = 2 / 7

-- The main theorem: if both conditions hold, c must equal to 35/3
theorem tank_capacity (w c : ℝ) (h1 : initial_fullness w c) (h2 : fullness_after_adding w c) : 
  c = 35 / 3 :=
sorry

end tank_capacity_l160_160933


namespace max_value_y_l160_160229

theorem max_value_y (x y : ℕ) (h₁ : 9 * (x + y) > 17 * x) (h₂ : 15 * x < 8 * (x + y)) :
  y ≤ 112 :=
sorry

end max_value_y_l160_160229


namespace geom_seq_sum_eq_six_l160_160506

theorem geom_seq_sum_eq_six 
    (a : ℕ → ℝ) 
    (r : ℝ) 
    (h_geom : ∀ n, a (n + 1) = a n * r) 
    (h_pos : ∀ n, a n > 0)
    (h_eq : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) 
    : a 5 + a 7 = 6 :=
sorry

end geom_seq_sum_eq_six_l160_160506


namespace common_ratio_geom_series_l160_160769

theorem common_ratio_geom_series 
  (a₁ a₂ : ℚ) 
  (h₁ : a₁ = 4 / 7) 
  (h₂ : a₂ = 20 / 21) :
  ∃ r : ℚ, r = 5 / 3 ∧ a₂ / a₁ = r := 
sorry

end common_ratio_geom_series_l160_160769


namespace each_person_gets_9_wings_l160_160007

noncomputable def chicken_wings_per_person (initial_wings : ℕ) (additional_wings : ℕ) (friends : ℕ) : ℕ :=
  (initial_wings + additional_wings) / friends

theorem each_person_gets_9_wings :
  chicken_wings_per_person 20 25 5 = 9 :=
by
  sorry

end each_person_gets_9_wings_l160_160007


namespace cone_prism_ratio_is_pi_over_16_l160_160939

noncomputable def cone_prism_volume_ratio 
  (prism_length : ℝ) (prism_width : ℝ) (prism_height : ℝ) 
  (cone_base_radius : ℝ) (cone_height : ℝ)
  (h_length : prism_length = 3) (h_width : prism_width = 4) (h_height : prism_height = 5)
  (h_radius_cone : cone_base_radius = 1.5) (h_cone_height : cone_height = 5) : ℝ :=
  (1/3) * Real.pi * cone_base_radius^2 * cone_height / (prism_length * prism_width * prism_height)

theorem cone_prism_ratio_is_pi_over_16
  (prism_length : ℝ) (prism_width : ℝ) (prism_height : ℝ)
  (cone_base_radius : ℝ) (cone_height : ℝ)
  (h_length : prism_length = 3) (h_width : prism_width = 4) (h_height : prism_height = 5)
  (h_radius_cone : cone_base_radius = 1.5) (h_cone_height : cone_height = 5) :
  cone_prism_volume_ratio prism_length prism_width prism_height cone_base_radius cone_height
    h_length h_width h_height h_radius_cone h_cone_height = Real.pi / 16 := 
by
  sorry

end cone_prism_ratio_is_pi_over_16_l160_160939


namespace tedra_harvested_2000kg_l160_160415

noncomputable def totalTomatoesHarvested : ℕ :=
  let wednesday : ℕ := 400
  let thursday : ℕ := wednesday / 2
  let total_wednesday_thursday := wednesday + thursday
  let remaining_friday : ℕ := 700
  let given_away_friday : ℕ := 700
  let friday := remaining_friday + given_away_friday
  total_wednesday_thursday + friday

theorem tedra_harvested_2000kg :
  totalTomatoesHarvested = 2000 := by
  sorry

end tedra_harvested_2000kg_l160_160415


namespace abs_expression_value_l160_160283

theorem abs_expression_value (x : ℤ) (h : x = -2023) :
  abs (2 * abs (abs x - x) - abs x) - x = 8092 :=
by {
  -- Proof will be provided here
  sorry
}

end abs_expression_value_l160_160283


namespace geometric_sequence_a3_l160_160831

theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 4 = 8)
  (h3 : ∀ k : ℕ, a (k + 1) = a k * q) : a 3 = 4 :=
sorry

end geometric_sequence_a3_l160_160831


namespace shaded_fraction_l160_160147

noncomputable def fraction_shaded (l w : ℝ) : ℝ :=
  1 - (1 / 8)

theorem shaded_fraction (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  fraction_shaded l w = 7 / 8 :=
by
  sorry

end shaded_fraction_l160_160147


namespace ursula_purchases_total_cost_l160_160574

variable (T C B Br : ℝ)
variable (hT : T = 10) (hTC : T = 2 * C) (hB : B = 0.8 * C) (hBr : Br = B / 2)

theorem ursula_purchases_total_cost : T + C + B + Br = 21 := by
  sorry

end ursula_purchases_total_cost_l160_160574


namespace expand_polynomials_l160_160222

variable (x : ℝ)

theorem expand_polynomials : 
  (3 * x^2 - 4 * x + 3) * (-4 * x^2 + 2 * x - 6) = -12 * x^4 + 22 * x^3 - 38 * x^2 + 30 * x - 18 :=
  by
  sorry

end expand_polynomials_l160_160222


namespace twelve_million_plus_twelve_thousand_l160_160728

theorem twelve_million_plus_twelve_thousand :
  12000000 + 12000 = 12012000 :=
by
  sorry

end twelve_million_plus_twelve_thousand_l160_160728


namespace intersecting_lines_l160_160442

theorem intersecting_lines
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)
  (h_p1 : p1 = (3, 2))
  (h_line1 : ∀ x : ℝ, p1.2 = 3 * p1.1 + 4)
  (h_line2 : ∀ x : ℝ, p2.2 = - (1 / 3) * p2.1 + 3) :
  (∃ p : ℝ × ℝ, p = (-3 / 10, 31 / 10)) :=
by
  sorry

end intersecting_lines_l160_160442


namespace probability_digit3_in_fraction_l160_160146

def repeating_sequence_of_fraction (n d : ℕ) := (7 : ℕ) / 11 = 0.636363...

theorem probability_digit3_in_fraction : 
  ∀ (n d : ℕ),
  repeating_sequence_of_fraction n d →
  (∃ b, b = "63") →
  (repeating_sequence_of_fraction n d .index 1 = 3) →
  (1/2 : ℚ) := 
sorry

end probability_digit3_in_fraction_l160_160146


namespace number_of_members_l160_160403

-- Define the conditions
def knee_pad_cost : ℕ := 6
def jersey_cost : ℕ := knee_pad_cost + 7
def wristband_cost : ℕ := jersey_cost + 3
def cost_per_member : ℕ := 2 * (knee_pad_cost + jersey_cost + wristband_cost)
def total_expenditure : ℕ := 4080

-- Prove the number of members in the club
theorem number_of_members (h1 : knee_pad_cost = 6)
                          (h2 : jersey_cost = 13)
                          (h3 : wristband_cost = 16)
                          (h4 : cost_per_member = 70)
                          (h5 : total_expenditure = 4080) :
                          total_expenditure / cost_per_member = 58 := 
by 
  sorry

end number_of_members_l160_160403


namespace caterpillar_length_difference_l160_160843

-- Define the lengths of the caterpillars
def green_caterpillar_length : ℝ := 3
def orange_caterpillar_length : ℝ := 1.17

-- State the theorem we need to prove
theorem caterpillar_length_difference :
  green_caterpillar_length - orange_caterpillar_length = 1.83 :=
by
  sorry

end caterpillar_length_difference_l160_160843


namespace minimize_sum_of_squares_of_roots_l160_160108

theorem minimize_sum_of_squares_of_roots (m : ℝ) (h : 100 - 20 * m ≥ 0) :
  (∀ a b : ℝ, (∀ x : ℝ, 5 * x^2 - 10 * x + m = 0 → x = a ∨ x = b) → (4 - 2 * m / 5) ≥ (4 - 2 * 5 / 5)) :=
by
  sorry

end minimize_sum_of_squares_of_roots_l160_160108


namespace line_a_minus_b_l160_160121

theorem line_a_minus_b (a b : ℝ)
  (h1 : (2 : ℝ) = a * (3 : ℝ) + b)
  (h2 : (26 : ℝ) = a * (7 : ℝ) + b) :
  a - b = 22 :=
by
  sorry

end line_a_minus_b_l160_160121


namespace prove_correct_statement_l160_160862

def correlation_coefficients (r1 r2 r3 r4 : ℝ) := (r1, r2, r3, r4)

def correct_statement (r1 r2 r3 r4 : ℝ) : Prop :=
  let corrs := correlation_coefficients r1 r2 r3 r4 in
  r1 = 0 ∧ r2 = -0.95 ∧ |r3| = 0.89 ∧ r4 = 0.75 →
  (¬ all_points_on_same_line r1) ∧
  (has_strongest_correlation r2 corrs) ∧
  (¬ has_strongest_correlation r3 corrs) ∧
  (¬ has_weakest_correlation r4 corrs)

noncomputable def all_points_on_same_line (r : ℝ) : Prop := r = 1 ∨ r = -1

noncomputable def has_strongest_correlation (r : ℝ) (corrs : ℝ × ℝ × ℝ × ℝ) : Prop := 
  ∀ (r' ∈ [corrs.1, corrs.2, corrs.3, corrs.4]), |r| ≥ |r'|

noncomputable def has_weakest_correlation (r : ℝ) (corrs : ℝ × ℝ × ℝ × ℝ) : Prop := 
  ∀ (r' ∈ [corrs.1, corrs.2, corrs.3, corrs.4]), |r| ≤ |r'|

theorem prove_correct_statement :
  ∃ (r1 r2 r3 r4 : ℝ), correct_statement r1 r2 r3 r4 :=
by 
  use 0, -0.95, 0.89, 0.75
  sorry

end prove_correct_statement_l160_160862


namespace average_speed_for_trip_l160_160320

theorem average_speed_for_trip (t₁ t₂ : ℝ) (v₁ v₂ : ℝ) (total_time : ℝ) 
  (h₁ : t₁ = 6) 
  (h₂ : v₁ = 30) 
  (h₃ : t₂ = 2) 
  (h₄ : v₂ = 46) 
  (h₅ : total_time = t₁ + t₂) 
  (h₆ : total_time = 8) :
  ((v₁ * t₁ + v₂ * t₂) / total_time) = 34 := 
  by 
    sorry

end average_speed_for_trip_l160_160320


namespace coeff_x2_in_x_minus_1_pow_4_l160_160830

theorem coeff_x2_in_x_minus_1_pow_4 :
  ∀ (x : ℝ), (∃ (p : ℕ), (x - 1) ^ 4 = p * x^2 + (other_terms) ∧ p = 6) :=
by sorry

end coeff_x2_in_x_minus_1_pow_4_l160_160830


namespace reciprocal_of_neg_five_l160_160870

theorem reciprocal_of_neg_five : (1 / (-5 : ℝ)) = -1 / 5 := 
by
  sorry

end reciprocal_of_neg_five_l160_160870


namespace sequence_mod_100_repeats_l160_160479

theorem sequence_mod_100_repeats (a0 : ℕ) : ∃ k l, k ≠ l ∧ (∃ seq : ℕ → ℕ, seq 0 = a0 ∧ (∀ n, seq (n + 1) = seq n + 54 ∨ seq (n + 1) = seq n + 77) ∧ (seq k % 100 = seq l % 100)) :=
by 
  sorry

end sequence_mod_100_repeats_l160_160479


namespace second_largest_subtract_smallest_correct_l160_160170

-- Definition of the elements
def numbers : List ℕ := [10, 11, 12, 13, 14]

-- Conditions derived from the problem
def smallest_number : ℕ := 10
def second_largest_number : ℕ := 13

-- Lean theorem statement representing the problem
theorem second_largest_subtract_smallest_correct :
  (second_largest_number - smallest_number) = 3 := 
by
  sorry

end second_largest_subtract_smallest_correct_l160_160170


namespace rakesh_fixed_deposit_percentage_l160_160411

-- Definitions based on the problem statement
def salary : ℝ := 4000
def cash_in_hand : ℝ := 2380
def spent_on_groceries : ℝ := 0.30

-- The theorem to prove
theorem rakesh_fixed_deposit_percentage (x : ℝ) 
  (H1 : cash_in_hand = 0.70 * (salary - (x / 100) * salary)) : 
  x = 15 := 
sorry

end rakesh_fixed_deposit_percentage_l160_160411


namespace factorize_x_cubed_minus_9x_l160_160778

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l160_160778


namespace points_on_planes_of_cubic_eq_points_on_planes_of_quintic_eq_l160_160954

-- Problem 1: Prove that if \(x^3 + y^3 + z^3 = (x + y + z)^3\), the points lie on the planes \(x + y = 0\), \(y + z = 0\), \(z + x = 0\).
theorem points_on_planes_of_cubic_eq (x y z : ℝ) (h : x^3 + y^3 + z^3 = (x + y + z)^3) :
  x + y = 0 ∨ y + z = 0 ∨ z + x = 0 :=
sorry

-- Problem 2: Prove that if \(x^5 + y^5 + z^5 = (x + y + z)^5\), the points lie on the planes \(x + y = 0\), \(y + z = 0\), \(z + x = 0\).
theorem points_on_planes_of_quintic_eq (x y z : ℝ) (h : x^5 + y^5 + z^5 = (x + y + z)^5) :
  x + y = 0 ∨ y + z = 0 ∨ z + x = 0 :=
sorry

end points_on_planes_of_cubic_eq_points_on_planes_of_quintic_eq_l160_160954


namespace Mike_owes_Laura_l160_160275

theorem Mike_owes_Laura :
  let rate_per_room := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let total_amount := (104 : ℚ) / 15
  rate_per_room * rooms_cleaned = total_amount :=
by
  sorry

end Mike_owes_Laura_l160_160275


namespace pairwise_coprime_triples_l160_160493

open Nat

theorem pairwise_coprime_triples (a b c : ℕ) 
  (h1 : a.gcd b = 1) (h2 : a.gcd c = 1) (h3 : b.gcd c = 1)
  (h4 : (a + b) ∣ c) (h5 : (a + c) ∣ b) (h6 : (b + c) ∣ a) :
  { (a, b, c) | (a = 1 ∧ b = 1 ∧ (c = 1 ∨ c = 2)) ∨ (a = 1 ∧ b = 2 ∧ c = 3) } :=
by
  -- Proof omitted for conciseness
  sorry

end pairwise_coprime_triples_l160_160493


namespace john_baseball_cards_l160_160272

theorem john_baseball_cards (new_cards old_cards cards_per_page : ℕ) (h1 : new_cards = 8) (h2 : old_cards = 16) (h3 : cards_per_page = 3) :
  (new_cards + old_cards) / cards_per_page = 8 := by
  sorry

end john_baseball_cards_l160_160272


namespace find_m_l160_160676

theorem find_m (m : ℝ) :
  (∃ x : ℝ, x^2 - m * x + m^2 - 19 = 0 ∧ (x = 2 ∨ x = 3))
  ∧ (∀ x : ℝ, x^2 - m * x + m^2 - 19 = 0 → x ≠ 2 ∧ x ≠ -4) 
  → m = -2 :=
by
  sorry

end find_m_l160_160676


namespace relay_race_total_time_l160_160345

theorem relay_race_total_time :
  let t1 := 55
  let t2 := t1 + 0.25 * t1
  let t3 := t2 - 0.20 * t2
  let t4 := t1 + 0.30 * t1
  let t5 := 80
  let t6 := t5 - 0.20 * t5
  let t7 := t5 + 0.15 * t5
  let t8 := t7 - 0.05 * t7
  t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 = 573.65 :=
by
  sorry

end relay_race_total_time_l160_160345


namespace average_pregnancies_per_kettle_l160_160934

-- Define the given conditions
def num_kettles : ℕ := 6
def babies_per_pregnancy : ℕ := 4
def survival_rate : ℝ := 0.75
def total_expected_babies : ℕ := 270

-- Calculate surviving babies per pregnancy
def surviving_babies_per_pregnancy : ℝ := babies_per_pregnancy * survival_rate

-- Prove that the average number of pregnancies per kettle is 15
theorem average_pregnancies_per_kettle : ∃ P : ℝ, num_kettles * P * surviving_babies_per_pregnancy = total_expected_babies ∧ P = 15 :=
by
  sorry

end average_pregnancies_per_kettle_l160_160934


namespace apples_for_juice_is_correct_l160_160418

noncomputable def apples_per_year : ℝ := 8 -- 8 million tons
noncomputable def percentage_mixed : ℝ := 0.30 -- 30%
noncomputable def remaining_apples := apples_per_year * (1 - percentage_mixed) -- Apples after mixed
noncomputable def percentage_for_juice : ℝ := 0.60 -- 60%
noncomputable def apples_for_juice := remaining_apples * percentage_for_juice -- Apples for juice

theorem apples_for_juice_is_correct :
  apples_for_juice = 3.36 :=
by
  sorry

end apples_for_juice_is_correct_l160_160418


namespace find_number_eq_seven_point_five_l160_160910

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l160_160910


namespace solution_m_in_interval_l160_160109

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x < 1 then -x^2 + 2 * m * x - 2 else 1 + Real.log x

theorem solution_m_in_interval :
  ∃ m : ℝ, (1 ≤ m ∧ m ≤ 2) ∧
  (∀ x < 1, ∀ y < 1, x ≤ y → f x m ≤ f y m) ∧
  (∀ x ≥ 1, ∀ y ≥ 1, x ≤ y → f x m ≤ f y m) ∧
  (∀ x < 1, ∀ y ≥ 1, f x m ≤ f y m) :=
by
  sorry

end solution_m_in_interval_l160_160109


namespace range_of_m_l160_160860

noncomputable def f (x m : ℝ) : ℝ := -x^2 - 4 * m * x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x1 x2 : ℝ, 2 ≤ x1 → x1 ≤ x2 → f x1 m ≥ f x2 m) ↔ m ≥ -1 := 
sorry

end range_of_m_l160_160860


namespace inequality_2_inequality_4_l160_160235

variables (a b : ℝ)
variables (h₁ : 0 < a) (h₂ : 0 < b)

theorem inequality_2 (h₁ : 0 < a) (h₂ : 0 < b) : a > |a - b| - b :=
by
  sorry

theorem inequality_4 (h₁ : 0 < a) (h₂ : 0 < b) : ab + 2 / ab > 2 :=
by
  sorry

end inequality_2_inequality_4_l160_160235


namespace no_real_solutions_l160_160865

theorem no_real_solutions :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 4 → ¬(3 * x^2 - 15 * x) / (x^2 - 4 * x) = x - 2) :=
by
  sorry

end no_real_solutions_l160_160865


namespace line_inclination_angle_l160_160217

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + (Real.sqrt 3) * y - 1 = 0

-- Define the condition of inclination angle in radians
def inclination_angle (θ : ℝ) : Prop := θ = Real.arctan (-1 / Real.sqrt 3) + Real.pi

-- The theorem to prove the inclination angle of the line
theorem line_inclination_angle (x y θ : ℝ) (h : line_eq x y) : inclination_angle θ :=
by
  sorry

end line_inclination_angle_l160_160217


namespace find_a_l160_160362

noncomputable def f (x a : ℝ) : ℝ := x - Real.log (x + 2) + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem find_a (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ a = 3) → a = -Real.log 2 - 1 :=
by
  sorry

end find_a_l160_160362


namespace playground_length_l160_160578

theorem playground_length
  (L_g : ℝ) -- length of the garden
  (L_p : ℝ) -- length of the playground
  (width_garden : ℝ := 24) -- width of the garden
  (width_playground : ℝ := 12) -- width of the playground
  (perimeter_garden : ℝ := 64) -- perimeter of the garden
  (area_garden : ℝ := L_g * 24) -- area of the garden
  (area_playground : ℝ := L_p * 12) -- area of the playground
  (areas_equal : area_garden = area_playground) -- equal areas
  (perimeter_condition : 2 * (L_g + 24) = 64) -- perimeter condition
  : L_p = 16 := 
by
  sorry

end playground_length_l160_160578


namespace second_projection_at_given_distance_l160_160875

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Line :=
  (point : Point)
  (direction : Point) -- Assume direction is given as a vector

def is_parallel (line1 line2 : Line) : Prop :=
  -- Function to check if two lines are parallel
  sorry

def distance (point1 point2 : Point) : ℝ := 
  -- Function to compute the distance between two points
  sorry

def first_projection_exists (M : Point) (a : Line) : Prop :=
  -- Check the projection outside the line a
  sorry

noncomputable def second_projection
  (M : Point)
  (a : Line)
  (d : ℝ)
  (h_parallel : is_parallel a (Line.mk ⟨0, 0, 0⟩ ⟨1, 0, 0⟩))
  (h_projection : first_projection_exists M a) :
  Point :=
  sorry

theorem second_projection_at_given_distance
  (M : Point)
  (a : Line)
  (d : ℝ)
  (h_parallel : is_parallel a (Line.mk ⟨0, 0, 0⟩ ⟨1, 0, 0⟩))
  (h_projection : first_projection_exists M a) :
  distance (second_projection M a d h_parallel h_projection) a.point = d :=
  sorry

end second_projection_at_given_distance_l160_160875


namespace third_side_length_not_4_l160_160253

theorem third_side_length_not_4 (x : ℕ) : 
  (5 < x + 9) ∧ (9 < x + 5) ∧ (x + 5 < 14) → ¬ (x = 4) := 
by
  intros h
  sorry

end third_side_length_not_4_l160_160253


namespace find_number_eq_seven_point_five_l160_160913

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l160_160913


namespace two_marbles_different_colors_probability_l160_160627

-- Definitions
def red_marbles : Nat := 3
def green_marbles : Nat := 4
def white_marbles : Nat := 5
def blue_marbles : Nat := 3
def total_marbles : Nat := red_marbles + green_marbles + white_marbles + blue_marbles

-- Combinations of different colored marbles
def red_green : Nat := red_marbles * green_marbles
def red_white : Nat := red_marbles * white_marbles
def red_blue : Nat := red_marbles * blue_marbles
def green_white : Nat := green_marbles * white_marbles
def green_blue : Nat := green_marbles * blue_marbles
def white_blue : Nat := white_marbles * blue_marbles

-- Total favorable outcomes
def total_favorable : Nat := red_green + red_white + red_blue + green_white + green_blue + white_blue

-- Total outcomes when drawing 2 marbles from the jar
def total_outcomes : Nat := Nat.choose total_marbles 2

-- Probability calculation
noncomputable def probability_different_colors : Rat := total_favorable / total_outcomes

-- Proof that the probability is 83/105
theorem two_marbles_different_colors_probability :
  probability_different_colors = 83 / 105 := by
  sorry

end two_marbles_different_colors_probability_l160_160627


namespace calculation_power_l160_160477

theorem calculation_power :
  (0.125 : ℝ) ^ 2012 * (2 ^ 2012) ^ 3 = 1 :=
sorry

end calculation_power_l160_160477


namespace smallest_number_among_10_11_12_l160_160727

theorem smallest_number_among_10_11_12 : min (min 10 11) 12 = 10 :=
by sorry

end smallest_number_among_10_11_12_l160_160727


namespace books_read_in_8_hours_l160_160549

def reading_speed := 100 -- pages per hour
def book_pages := 400 -- pages per book
def hours_available := 8 -- hours

theorem books_read_in_8_hours :
  (hours_available * reading_speed) / book_pages = 2 :=
by
  sorry

end books_read_in_8_hours_l160_160549


namespace find_constant_d_l160_160249

noncomputable def polynomial_g (d : ℝ) (x : ℝ) := d * x^4 + 17 * x^3 - 5 * d * x^2 + 45

theorem find_constant_d (d : ℝ) : polynomial_g d 5 = 0 → d = -4.34 :=
by
  sorry

end find_constant_d_l160_160249


namespace snakes_hiding_l160_160473

/-- The statement that given the total number of snakes and the number of snakes not hiding,
we can determine the number of snakes hiding. -/
theorem snakes_hiding (total_snakes : ℕ) (snakes_not_hiding : ℕ) (h1 : total_snakes = 95) (h2 : snakes_not_hiding = 31) :
  total_snakes - snakes_not_hiding = 64 :=
by {
  sorry
}

end snakes_hiding_l160_160473


namespace max_constant_N_l160_160095

theorem max_constant_N (a b c d : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0):
  (c^2 + d^2) ≠ 0 → ∃ N, N = 1 ∧ (a^2 + b^2) / (c^2 + d^2) ≤ 1 :=
by
  sorry

end max_constant_N_l160_160095


namespace min_value_inequality_l160_160392

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ( (x^2 + 4*x + 2) * (y^2 + 5*y + 3) * (z^2 + 6*z + 4) ) / (x * y * z) ≥ 336 := 
by
  sorry

end min_value_inequality_l160_160392


namespace length_of_garden_l160_160122

theorem length_of_garden (P B : ℕ) (hP : P = 1800) (hB : B = 400) : 
  ∃ L : ℕ, L = 500 ∧ P = 2 * (L + B) :=
by
  sorry

end length_of_garden_l160_160122


namespace work_together_days_l160_160323

noncomputable def A_per_day := 1 / 78
noncomputable def B_per_day := 1 / 39

theorem work_together_days 
  (A : ℝ) (B : ℝ) 
  (hA : A = 1 / 78)
  (hB : B = 1 / 39) : 
  1 / (A + B) = 26 :=
by
  rw [hA, hB]
  sorry

end work_together_days_l160_160323


namespace focus_parabola_y_eq_4x2_l160_160856

theorem focus_parabola_y_eq_4x2 :
  ∀ (x y : ℝ), y = 4 * x^2 → ∃ p : ℝ, p = 1 / 16 ∧ (x, y) = (0, p) :=
by
  intros x y hy
  use (1 / 16)
  split
  { refl }
  { suffices hr : y = 4 * x ^ 2, rw [hy, hr], norm_num }

end focus_parabola_y_eq_4x2_l160_160856


namespace expand_product_equivalence_l160_160658

variable (x : ℝ)  -- Assuming x is a real number

theorem expand_product_equivalence : (x + 5) * (x + 7) = x^2 + 12 * x + 35 :=
by
  sorry

end expand_product_equivalence_l160_160658


namespace complement_U_M_correct_l160_160812

open Set

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {x | x^2 - 4 * x + 3 = 0}
def complement_U_M : Set ℕ := U \ M

theorem complement_U_M_correct : complement_U_M = {2, 4} :=
by
  -- Proof will be provided here
  sorry

end complement_U_M_correct_l160_160812


namespace only_zero_function_satisfies_inequality_l160_160673

noncomputable def f (x : ℝ) : ℝ := sorry

theorem only_zero_function_satisfies_inequality (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  (∀ x y : ℝ, 0 < x → 0 < y →
    f x * f y ≥ (y^α / (x^α + x^β)) * (f x)^2 + (x^β / (y^α + y^β)) * (f y)^2) →
  ∀ x : ℝ, 0 < x → f x = 0 :=
sorry

end only_zero_function_satisfies_inequality_l160_160673


namespace complement_intersection_eq_4_l160_160994

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection_eq_4 (hU : U = {0, 1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4}) :
  ((U \ A) ∩ B) = {4} :=
by {
  -- Proof goes here
  exact sorry
}

end complement_intersection_eq_4_l160_160994


namespace tan_u_tan_v_sum_l160_160539

theorem tan_u_tan_v_sum (u v : ℝ) 
  (h1 : (sin u / cos v) + (sin v / cos u) = 2)
  (h2 : (cos u / sin v) + (cos v / sin u) = 3) :
  (tan u / tan v) + (tan v / tan u) = 8 / 7 :=
by
  sorry

end tan_u_tan_v_sum_l160_160539


namespace tank_capacity_75_l160_160753

theorem tank_capacity_75 (c w : ℝ) 
  (h₁ : w = c / 3) 
  (h₂ : (w + 5) / c = 2 / 5) : 
  c = 75 := 
  sorry

end tank_capacity_75_l160_160753


namespace find_number_eq_seven_point_five_l160_160917

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l160_160917


namespace percentage_of_first_pay_cut_l160_160537

theorem percentage_of_first_pay_cut
  (x : ℝ)
  (h1 : ∃ y z w : ℝ, y = 1 - x/100 ∧ z = 0.86 ∧ w = 0.82 ∧ y * z * w = 0.648784):
  x = 8.04 := by
-- The proof will be added here, this is just the statement
sorry

end percentage_of_first_pay_cut_l160_160537


namespace eighth_term_of_arithmetic_sequence_l160_160174

theorem eighth_term_of_arithmetic_sequence
  (a l : ℕ) (n : ℕ) (h₁ : a = 4) (h₂ : l = 88) (h₃ : n = 30) :
  (a + 7 * (l - a) / (n - 1) = (676 : ℚ) / 29) :=
by
  sorry

end eighth_term_of_arithmetic_sequence_l160_160174


namespace spherical_to_rectangular_coords_l160_160482

theorem spherical_to_rectangular_coords (
  {ρ θ φ : ℝ} 
) (hρ : ρ = 3) (hθ : θ = 3 * Real.pi / 2) (hφ : φ = Real.pi / 3) :
  let x := ρ * Real.sin φ * Real.cos θ,
      y := ρ * Real.sin φ * Real.sin θ,
      z := ρ * Real.cos φ in
  (x, y, z) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by 
  sorry

end spherical_to_rectangular_coords_l160_160482


namespace equal_prob_first_ace_l160_160824

/-
  Define the problem:
  In a 4-player card game with a 32-card deck containing 4 aces,
  prove that the probability of each player drawing the first ace is 1/8.
-/

namespace CardGame

def deck : list ℕ := list.range 32

def is_ace (card : ℕ) : Prop := card % 8 = 0

def player_turn (turn : ℕ) : ℕ := turn % 4

def first_ace_turn (deck : list ℕ) : ℕ :=
deck.find_index is_ace

theorem equal_prob_first_ace :
  ∀ (deck : list ℕ) (h : deck.cardinality = 32) (h_ace : ∑ (card ∈ deck) (is_ace card) = 4),
  ∀ (player : ℕ), player < 4 → (∃ n < 32, first_ace_turn deck = some n ∧ player_turn n = player) →
  (deck.countp is_ace) / 32 = 1 / 8 :=
by sorry

end CardGame

end equal_prob_first_ace_l160_160824


namespace repeating_decimal_is_fraction_l160_160061

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l160_160061


namespace coin_grid_probability_l160_160332

/--
A square grid is given where the edge length of each smallest square is 6 cm.
A hard coin with a diameter of 2 cm is thrown onto this grid.
Prove that the probability that the coin, after landing, will have a common point with the grid lines is 5/9.
-/
theorem coin_grid_probability :
  let square_edge_cm := 6
  let coin_diameter_cm := 2
  let coin_radius_cm := coin_diameter_cm / 2
  let grid_center_edge_cm := square_edge_cm - coin_diameter_cm
  let non_intersect_area_ratio := (grid_center_edge_cm ^ 2) / (square_edge_cm ^ 2)
  1 - non_intersect_area_ratio = 5 / 9 :=
by
  sorry

end coin_grid_probability_l160_160332


namespace partI_inequality_solution_partII_minimum_value_l160_160112

-- Part (I)
theorem partI_inequality_solution (x : ℝ) : 
  (abs (x + 1) + abs (2 * x - 1) ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 1) :=
sorry

-- Part (II)
theorem partII_minimum_value (a b c : ℝ) (h1 : a + b + c = 2) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) :
  (∀ a b c : ℝ, a + b + c = 2 ->  a > 0 -> b > 0 -> c > 0 -> 
    (1 / a + 1 / b + 1 / c) = (9 / 2)) :=
sorry

end partI_inequality_solution_partII_minimum_value_l160_160112


namespace line_equation_l160_160034

noncomputable def center_of_circle : (ℝ × ℝ) := (-1, 2)

noncomputable def slope : ℝ := 1

theorem line_equation :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 → slope = 1 →
  x - y + 3 = 0 :=
by sorry

end line_equation_l160_160034


namespace positive_integer_pairs_divisibility_l160_160646

theorem positive_integer_pairs_divisibility (a b : ℕ) (h : a * b^2 + b + 7 ∣ a^2 * b + a + b) :
  (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ ∃ k : ℕ, k > 0 ∧ a = 7 * k^2 ∧ b = 7 * k :=
sorry

end positive_integer_pairs_divisibility_l160_160646


namespace polygon_sides_l160_160510

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1980) : n = 13 := 
by sorry

end polygon_sides_l160_160510


namespace total_cups_l160_160868

theorem total_cups (b f s : ℕ) (ratio_bt_f_s : b / s = 1 / 5) (ratio_fl_b_s : f / s = 8 / 5) (sugar_cups : s = 10) :
  b + f + s = 28 :=
sorry

end total_cups_l160_160868


namespace fraction_eq_repeating_decimal_l160_160045

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l160_160045


namespace geometric_sequence_property_l160_160494

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n m : ℕ), a (n + 1) * a (m + 1) = a n * a m

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
(h_condition : a 2 * a 4 = 1/2) :
  a 1 * a 3 ^ 2 * a 5 = 1/4 :=
by
  sorry

end geometric_sequence_property_l160_160494


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l160_160591

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l160_160591


namespace find_x_l160_160301

variable (n : ℝ) (x : ℝ)

theorem find_x (h1 : n = 15.0) (h2 : 3 * n - x = 40) : x = 5.0 :=
by
  sorry

end find_x_l160_160301


namespace david_average_speed_l160_160922

theorem david_average_speed (d t : ℚ) (h1 : d = 49 / 3) (h2 : t = 7 / 3) :
  (d / t) = 7 :=
by
  rw [h1, h2]
  norm_num

end david_average_speed_l160_160922


namespace larger_number_of_two_l160_160678

theorem larger_number_of_two (x y : ℝ) (h1 : x * y = 30) (h2 : x + y = 13) : max x y = 10 :=
sorry

end larger_number_of_two_l160_160678


namespace length_AP_eq_sqrt2_l160_160261

/-- In square ABCD with side length 2, a circle ω with center at (1, 0)
    and radius 1 is inscribed. The circle intersects CD at point M,
    and line AM intersects ω at a point P different from M.
    Prove that the length of AP is √2. -/
theorem length_AP_eq_sqrt2 :
  let A := (0, 2)
  let M := (2, 0)
  let P : ℝ × ℝ := (1, 1)
  dist A P = Real.sqrt 2 :=
by
  sorry

end length_AP_eq_sqrt2_l160_160261


namespace intersection_point_of_lines_l160_160443

theorem intersection_point_of_lines
    : ∃ (x y: ℝ), y = 3 * x + 4 ∧ y = - (1 / 3) * x + 5 ∧ x = 3 / 10 ∧ y = 49 / 10 :=
by
  sorry

end intersection_point_of_lines_l160_160443


namespace fraction_numerator_less_denominator_l160_160160

theorem fraction_numerator_less_denominator (x : ℝ) (h : -3 ≤ x ∧ x ≤ 3) :
  (8 * x - 3 < 9 + 5 * x) ↔ (-3 ≤ x ∧ x < 3) :=
by sorry

end fraction_numerator_less_denominator_l160_160160


namespace solution_set_of_equation_l160_160393

theorem solution_set_of_equation (x : ℝ) : 
  (abs (2 * x - 1) = abs x + abs (x - 1)) ↔ (x ≤ 0 ∨ x ≥ 1) := 
by 
  sorry

end solution_set_of_equation_l160_160393


namespace set_contains_all_nonnegative_integers_l160_160394

theorem set_contains_all_nonnegative_integers (S : Set ℕ) :
  (∃ a b, a ∈ S ∧ b ∈ S ∧ 1 < a ∧ 1 < b ∧ Nat.gcd a b = 1) →
  (∀ x y, x ∈ S → y ∈ S → y ≠ 0 → (x * y) ∈ S ∧ (x % y) ∈ S) →
  (∀ n, n ∈ S) :=
by
  intros h1 h2
  sorry

end set_contains_all_nonnegative_integers_l160_160394


namespace ratio_of_numbers_l160_160123

theorem ratio_of_numbers (x y : ℕ) (h1 : x + y = 33) (h2 : x = 22) : y / x = 1 / 2 :=
by
  sorry

end ratio_of_numbers_l160_160123


namespace population_in_2050_l160_160960

def population : ℕ → ℕ := sorry

theorem population_in_2050 : population 2050 = 2700 :=
by
  -- sorry statement to skip the proof
  sorry

end population_in_2050_l160_160960


namespace strongest_correlation_l160_160863

variables (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ)
variables (abs_r3 : ℝ)

-- Define conditions as hypotheses
def conditions :=
  r1 = 0 ∧ r2 = -0.95 ∧ abs_r3 = 0.89 ∧ r4 = 0.75 ∧ abs r3 = abs_r3

-- Theorem stating the correct answer
theorem strongest_correlation (hyp : conditions r1 r2 r3 r4 abs_r3) : 
  abs r2 > abs r1 ∧ abs r2 > abs r3 ∧ abs r2 > abs r4 :=
by sorry

end strongest_correlation_l160_160863


namespace sum_of_factors_30_l160_160615

theorem sum_of_factors_30 : 
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ) in
  ( ∑ n in factors, n ) = 72 :=
by
  let factors := ({1, 2, 3, 5, 6, 10, 15, 30} : set ℕ)
  rw [finset.sum_eq_sum_of_elements] -- transposing to handle the sum notation
  let factors_list := [1, 2, 3, 5, 6, 10, 15, 30]
  have h_factors : factors = {x | x ∈ factors_list}.to_finset, by sorry
  simp_rw h_factors
  rw [finset.sum_to_finset]
  dsimp -- simplifying before asserting the sum
  norm_num -- proving that the sum is indeed 72
  sorry

end sum_of_factors_30_l160_160615


namespace find_number_eq_seven_point_five_l160_160914

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l160_160914


namespace nurses_count_l160_160317

theorem nurses_count (total : ℕ) (ratio_doc : ℕ) (ratio_nurse : ℕ) (nurses : ℕ) : 
  total = 200 → 
  ratio_doc = 4 → 
  ratio_nurse = 6 → 
  nurses = (ratio_nurse * total / (ratio_doc + ratio_nurse)) → 
  nurses = 120 := 
by 
  intros h_total h_ratio_doc h_ratio_nurse h_calc
  rw [h_total, h_ratio_doc, h_ratio_nurse] at h_calc
  simp at h_calc
  exact h_calc

end nurses_count_l160_160317


namespace number_of_blue_eyed_students_in_k_class_l160_160842

-- Definitions based on the given conditions
def total_students := 40
def blond_hair_to_blue_eyes_ratio := 2.5
def students_with_both := 8
def students_with_neither := 5

-- We need to prove that the number of blue-eyed students is 10
theorem number_of_blue_eyed_students_in_k_class 
  (x : ℕ)  -- number of blue-eyed students
  (H1 : total_students = 40)
  (H2 : ∀ x, blond_hair_to_blue_eyes_ratio * x = number_of_blond_students)
  (H3 : students_with_both = 8)
  (H4 : students_with_neither = 5)
  : x = 10 :=
sorry

end number_of_blue_eyed_students_in_k_class_l160_160842


namespace find_number_l160_160887

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l160_160887


namespace even_x_satisfies_remainder_l160_160721

theorem even_x_satisfies_remainder 
  (z : ℕ) 
  (hz : z % 4 = 0) : 
  ∃ (x : ℕ), x % 2 = 0 ∧ (z * (2 + x + z) + 3) % 2 = 1 := 
by
  sorry

end even_x_satisfies_remainder_l160_160721


namespace max_values_of_f_smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l160_160110

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)

theorem max_values_of_f (k : ℤ) : 
  ∃ x, f x = 2 ∧ x = 4 * (k : ℝ) * Real.pi - (2 * Real.pi / 3) := 
sorry

theorem smallest_positive_period_of_f : 
  ∃ T, T = 4 * Real.pi := 
sorry

theorem intervals_where_f_is_monotonically_increasing (k : ℤ) : 
  ∀ x, (-(5 * Real.pi / 3) + 4 * (k : ℝ) * Real.pi ≤ x) ∧ (x ≤ Real.pi / 3 + 4 * (k : ℝ) * Real.pi) → 
  ∀ y, (-(5 * Real.pi / 3) + 4 * (k : ℝ) * Real.pi ≤ y) ∧ (y ≤ Real.pi / 3 + 4 * (k : ℝ) * Real.pi) → 
  (x ≤ y ↔ f x ≤ f y) :=
sorry

end max_values_of_f_smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l160_160110


namespace min_b_minus_a_l160_160242

noncomputable def f (x : ℝ) : ℝ := 1 + x - (x^2) / 2 + (x^3) / 3
noncomputable def g (x : ℝ) : ℝ := 1 - x + (x^2) / 2 - (x^3) / 3
noncomputable def F (x : ℝ) : ℝ := f x * g x

theorem min_b_minus_a (a b : ℤ) (h : ∀ x, F x = 0 → a ≤ x ∧ x ≤ b) (h_a_lt_b : a < b) : b - a = 3 :=
sorry

end min_b_minus_a_l160_160242


namespace smallest_b_factors_l160_160099

theorem smallest_b_factors 
: ∃ b : ℕ, b > 0 ∧ 
    (∃ p q : ℤ, x^2 + b * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) ∧ 
    ∀ b': ℕ, (∃ p q: ℤ, x^2 + b' * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) → (b ≤ b') := 
sorry

end smallest_b_factors_l160_160099


namespace probability_A_fires_l160_160928

theorem probability_A_fires 
  (p_first_shot: ℚ := 1/6)
  (p_not_fire: ℚ := 5/6)
  (p_recur: ℚ := p_not_fire * p_not_fire) : 
  ∃ (P_A : ℚ), P_A = 6/11 :=
by
  have eq1 : P_A = p_first_shot + (p_recur * P_A) := sorry
  have eq2 : P_A * (1 - p_recur) = p_first_shot := sorry
  have eq3 : P_A = (p_first_shot * 36) / 11 := sorry
  exact ⟨P_A, sorry⟩

end probability_A_fires_l160_160928


namespace solve_abs_equation_l160_160972

-- Define the condition for the equation
def condition (x : ℝ) : Prop := 3 * x + 5 ≥ 0

-- The main theorem to prove that x = 1/5 is the only solution
theorem solve_abs_equation (x : ℝ) (h : condition x) : |2 * x - 6| = 3 * x + 5 ↔ x = 1 / 5 := by
  sorry

end solve_abs_equation_l160_160972


namespace polygon_interior_angle_sum_l160_160308

theorem polygon_interior_angle_sum (n : ℕ) (h : (n - 2) * 180 = 1800) : n = 12 :=
by sorry

end polygon_interior_angle_sum_l160_160308


namespace intersection_point_of_lines_l160_160444

theorem intersection_point_of_lines
    : ∃ (x y: ℝ), y = 3 * x + 4 ∧ y = - (1 / 3) * x + 5 ∧ x = 3 / 10 ∧ y = 49 / 10 :=
by
  sorry

end intersection_point_of_lines_l160_160444


namespace LCM_of_apple_and_cherry_pies_l160_160196

theorem LCM_of_apple_and_cherry_pies :
  let apple_pies := (13 : ℚ) / 2
  let cherry_pies := (21 : ℚ) / 4
  let lcm_numerators := Nat.lcm 26 21
  let common_denominator := 4
  (lcm_numerators : ℚ) / (common_denominator : ℚ) = 273 / 2 :=
by
  let apple_pies := (13 : ℚ) / 2
  let cherry_pies := (21 : ℚ) / 4
  let lcm_numerators := Nat.lcm 26 21
  let common_denominator := 4
  have h : (lcm_numerators : ℚ) / (common_denominator : ℚ) = 273 / 2 := sorry
  exact h

end LCM_of_apple_and_cherry_pies_l160_160196


namespace reservoir_water_l160_160338

-- Conditions definitions
def total_capacity (C : ℝ) : Prop :=
  ∃ (x : ℝ), x = C

def normal_level (C : ℝ) : ℝ :=
  C - 20

def water_end_of_month (C : ℝ) : ℝ :=
  0.75 * C

def condition_equation (C : ℝ) : Prop :=
  water_end_of_month C = 2 * normal_level C

-- The theorem proving the amount of water at the end of the month is 24 million gallons given the conditions
theorem reservoir_water (C : ℝ) (hC : total_capacity C) (h_condition : condition_equation C) : water_end_of_month C = 24 :=
by
  sorry

end reservoir_water_l160_160338


namespace a_2005_l160_160797

noncomputable def a : ℕ → ℤ := sorry 

axiom a3 : a 3 = 5
axiom a5 : a 5 = 8
axiom exists_n : ∃ (n : ℕ), n > 0 ∧ a n + a (n + 1) + a (n + 2) = 7

theorem a_2005 : a 2005 = -6 := by {
  sorry
}

end a_2005_l160_160797


namespace tangent_slope_at_A_l160_160307

open Real

theorem tangent_slope_at_A (x y : ℝ) (h : y = exp (-x)) : 
  deriv (λ x, exp (-x)) 0 = -1 := by
sorry

end tangent_slope_at_A_l160_160307


namespace fraction_of_repeating_decimal_l160_160075

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l160_160075


namespace new_average_after_multiplication_l160_160186

theorem new_average_after_multiplication
  (n : ℕ) (a : ℕ) (m : ℕ)
  (h1 : n = 7)
  (h2 : a = 25)
  (h3 : m = 5):
  (n * a * m / n) = 125 :=
by
  sorry


end new_average_after_multiplication_l160_160186


namespace solve_equation_l160_160167

theorem solve_equation :
  ∃ y : ℚ, 2 * (y - 3) - 6 * (2 * y - 1) = -3 * (2 - 5 * y) ↔ y = 6 / 25 :=
by
  sorry

end solve_equation_l160_160167


namespace acai_berry_cost_correct_l160_160427

def cost_superfruit_per_litre : ℝ := 1399.45
def cost_mixed_fruit_per_litre : ℝ := 262.85
def litres_mixed_fruit : ℝ := 36
def litres_acai_berry : ℝ := 24
def total_litres : ℝ := litres_mixed_fruit + litres_acai_berry
def expected_cost_acai_per_litre : ℝ := 3104.77

theorem acai_berry_cost_correct :
  cost_superfruit_per_litre * total_litres -
  cost_mixed_fruit_per_litre * litres_mixed_fruit = 
  expected_cost_acai_per_litre * litres_acai_berry :=
by sorry

end acai_berry_cost_correct_l160_160427


namespace total_expenditure_is_3000_l160_160740

/-- Define the Hall dimensions -/
def length : ℝ := 20
def width : ℝ := 15
def cost_per_square_meter : ℝ := 10

/-- Statement to prove --/
theorem total_expenditure_is_3000 
  (h_length : length = 20)
  (h_width : width = 15)
  (h_cost : cost_per_square_meter = 10) : 
  length * width * cost_per_square_meter = 3000 :=
sorry

end total_expenditure_is_3000_l160_160740


namespace find_expression_l160_160119

variable (a b E : ℝ)

-- Conditions
def condition1 := a / b = 4 / 3
def condition2 := E / (3 * a - 2 * b) = 3

-- Conclusion we want to prove
theorem find_expression : condition1 a b → condition2 a b E → E = 6 * b :=
by
  intro h1 h2
  sorry

end find_expression_l160_160119


namespace find_angle_C_proof_max_triangle_area_proof_l160_160129

open Real

noncomputable def find_angle_C (A B : ℝ) (C : ℝ) :=
  let m := (cos A, sin A)
  let n := (cos B, -sin B)
  let vector_distance := sqrt ((cos A - cos B)^2 + (sin A + sin B)^2)
  vector_distance = 1

noncomputable def max_triangle_area (A B C : ℝ) (a b c : ℝ) :=
  let cosC := -1/2
  let C := 2 * π / 3
  let triangle_area (a b C : ℝ) := 1/2 * a * b * sin C
  c = 3 ∧ a * b * sin C ≤ 3/2

theorem find_angle_C_proof (A B : ℝ) :
  (∃ C : ℝ, find_angle_C A B C) →
  C = 2 * π / 3 :=
by
  sorry

theorem max_triangle_area_proof (A B a b : ℝ) (h : ∃ C : ℝ, C = 2 * π / 3) :
  c = 3 → max_triangle_area A B h.some a b c → 
  (1/2 * a * b * sin (2 * π / 3) = 3 * sqrt 3 / 4) :=
by
  sorry

end find_angle_C_proof_max_triangle_area_proof_l160_160129


namespace ababab_divisible_by_7_l160_160286

theorem ababab_divisible_by_7 (a b : ℕ) (ha : a < 10) (hb : b < 10) : (101010 * a + 10101 * b) % 7 = 0 :=
by sorry

end ababab_divisible_by_7_l160_160286


namespace find_number_divided_by_3_equals_subtracted_5_l160_160894

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l160_160894


namespace height_on_hypotenuse_correct_l160_160258

noncomputable def height_on_hypotenuse (a b : ℝ) (ha : a = 3) (hb : b = 4) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let area := (a * b) / 2
  (2 * area) / c

theorem height_on_hypotenuse_correct (h : ℝ) : 
  height_on_hypotenuse 3 4 rfl rfl = 12 / 5 :=
by
  sorry

end height_on_hypotenuse_correct_l160_160258


namespace exists_coprime_less_than_100_l160_160406

theorem exists_coprime_less_than_100 (a b c : ℕ) (ha : a < 1000000) (hb : b < 1000000) (hc : c < 1000000) :
  ∃ d, d < 100 ∧ gcd d a = 1 ∧ gcd d b = 1 ∧ gcd d c = 1 :=
by sorry

end exists_coprime_less_than_100_l160_160406


namespace angle_C_is_65_deg_l160_160268

-- Defining a triangle and its angles.
structure Triangle :=
  (A B C : ℝ) -- representing the angles in degrees

-- Defining the conditions of the problem.
def given_triangle : Triangle :=
  { A := 75, B := 40, C := 180 - 75 - 40 }

-- Statement of the problem, proving that the measure of ∠C is 65°.
theorem angle_C_is_65_deg (t : Triangle) (hA : t.A = 75) (hB : t.B = 40) (hSum : t.A + t.B + t.C = 180) : t.C = 65 :=
  by sorry

end angle_C_is_65_deg_l160_160268


namespace probability_min_diff_at_least_3_l160_160309

theorem probability_min_diff_at_least_3 :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let total_ways := (Finset.card (Finset.powersetLen 3 S) : ℚ) 
  let valid_sets := [({1, 4, 7} : Finset ℕ), ({2, 5, 8} : Finset ℕ), ({3, 6, 9} : Finset ℕ)].length
  (valid_sets / total_ways) = (1 / 28) := by
  sorry

end probability_min_diff_at_least_3_l160_160309


namespace jenny_improvements_value_l160_160693

-- Definitions based on the conditions provided
def property_tax_rate : ℝ := 0.02
def initial_house_value : ℝ := 400000
def rail_project_increase : ℝ := 0.25
def affordable_property_tax : ℝ := 15000

-- Statement of the theorem
theorem jenny_improvements_value :
  let new_house_value := initial_house_value * (1 + rail_project_increase)
  let max_affordable_house_value := affordable_property_tax / property_tax_rate
  let value_of_improvements := max_affordable_house_value - new_house_value
  value_of_improvements = 250000 := 
by
  sorry

end jenny_improvements_value_l160_160693


namespace geometric_sequence_condition_l160_160838

variable {a : ℕ → ℝ}

-- Definitions based on conditions in the problem
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- The statement translating the problem
theorem geometric_sequence_condition (q : ℝ) (a : ℕ → ℝ) (h : is_geometric_sequence a q) : ¬((q > 1) ↔ is_increasing_sequence a) :=
  sorry

end geometric_sequence_condition_l160_160838


namespace domain_of_func_l160_160153

-- Define the function
def func (x : ℝ) : ℝ := Real.tan ((π / 6) * x + π / 3)

-- Define the set of problematic points where the function is undefined
def problem_points : Set ℝ := { x | ∃ k : ℤ, x = 1 + 6 * k }

-- Define the domain of the function
def func_domain : Set ℝ := { x | ¬ (x ∈ problem_points) }

-- Formulate the statement
theorem domain_of_func :
  (SetOf (λ x, func x)).domain = func_domain := 
  sorry

end domain_of_func_l160_160153


namespace least_number_to_add_divisible_l160_160923

theorem least_number_to_add_divisible (n d : ℕ) (h1 : n = 929) (h2 : d = 30) : 
  ∃ x, (n + x) % d = 0 ∧ x = 1 := 
by 
  sorry

end least_number_to_add_divisible_l160_160923


namespace repeating_fraction_equality_l160_160086

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end repeating_fraction_equality_l160_160086


namespace value_of_expression_l160_160490

theorem value_of_expression (V E F t h : ℕ) (H T : ℕ) 
  (h1 : V - E + F = 2)
  (h2 : F = 42)
  (h3 : T = 3)
  (h4 : H = 2)
  (h5 : t + h = 42)
  (h6 : E = (3 * t + 6 * h) / 2) :
  100 * H + 10 * T + V = 328 :=
sorry

end value_of_expression_l160_160490


namespace exists_prime_seq_satisfying_condition_l160_160221

theorem exists_prime_seq_satisfying_condition :
  ∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ (∀ m n, m < n → a m < a n) ∧ 
  (∀ i j, i ≠ j → (i * a j, j * a i) = (i, j)) :=
sorry

end exists_prime_seq_satisfying_condition_l160_160221


namespace range_of_a_l160_160990

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Define the conditions: f has a unique zero point x₀ and x₀ < 0
def unique_zero_point (a : ℝ) : Prop :=
  ∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0

-- The theorem we need to prove
theorem range_of_a (a : ℝ) : unique_zero_point a → a > 2 :=
sorry

end range_of_a_l160_160990


namespace repeating_decimal_is_fraction_l160_160062

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l160_160062


namespace number_of_terms_in_arithmetic_sequence_l160_160648

theorem number_of_terms_in_arithmetic_sequence 
  (a : ℕ)
  (d : ℕ)
  (an : ℕ)
  (h1 : a = 3)
  (h2 : d = 4)
  (h3 : an = 47) :
  ∃ n : ℕ, an = a + (n - 1) * d ∧ n = 12 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l160_160648


namespace find_polynomials_g_l160_160296

-- Assume f(x) = x^2
def f (x : ℝ) : ℝ := x ^ 2

-- Define the condition that f(g(x)) = 9x^2 - 6x + 1
def condition (g : ℝ → ℝ) : Prop := ∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1

-- Prove that the possible polynomials for g(x) are 3x - 1 or -3x + 1
theorem find_polynomials_g (g : ℝ → ℝ) (h : condition g) :
  (∀ x, g x = 3 * x - 1) ∨ (∀ x, g x = -3 * x + 1) :=
sorry

end find_polynomials_g_l160_160296


namespace fraction_eq_repeating_decimal_l160_160070

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l160_160070


namespace union_sets_l160_160233

noncomputable def setA : Set ℝ := { x | x^2 - 3*x - 4 ≤ 0 }
noncomputable def setB : Set ℝ := { x | 1 < x ∧ x < 5 }

theorem union_sets :
  (setA ∪ setB) = { x | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_sets_l160_160233


namespace repeating_decimal_eq_fraction_l160_160054

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l160_160054


namespace factorize_a3_sub_a_l160_160492

theorem factorize_a3_sub_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a3_sub_a_l160_160492


namespace claire_initial_balloons_l160_160022

theorem claire_initial_balloons (B : ℕ) (h : B - 12 - 9 + 11 = 39) : B = 49 :=
by sorry

end claire_initial_balloons_l160_160022


namespace complete_work_together_in_days_l160_160919

noncomputable def a_days := 16
noncomputable def b_days := 6
noncomputable def c_days := 12

noncomputable def work_rate (days: ℕ) : ℚ := 1 / days

theorem complete_work_together_in_days :
  let combined_rate := (work_rate a_days) + (work_rate b_days) + (work_rate c_days)
  let days_to_complete := 1 / combined_rate
  days_to_complete = 3.2 :=
  sorry

end complete_work_together_in_days_l160_160919


namespace recurring_decimal_to_fraction_l160_160081

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l160_160081


namespace integer_pairs_satisfying_equation_l160_160967

theorem integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), x * (x + 1) * (x + 7) * (x + 8) = y^2 →
    (x = 1 ∧ y = 12) ∨ (x = 1 ∧ y = -12) ∨ 
    (x = -9 ∧ y = 12) ∨ (x = -9 ∧ y = -12) ∨ 
    (x = -4 ∧ y = 12) ∨ (x = -4 ∧ y = -12) ∨ 
    (x = 0 ∧ y = 0) ∨ (x = -8 ∧ y = 0) ∨ 
    (x = -1 ∧ y = 0) ∨ (x = -7 ∧ y = 0) :=
by sorry

end integer_pairs_satisfying_equation_l160_160967


namespace total_cost_correct_l160_160318

def sandwich_cost : ℝ := 2.45
def soda_cost : ℝ := 0.87
def sandwich_quantity : ℕ := 2
def soda_quantity : ℕ := 4

theorem total_cost_correct :
  sandwich_quantity * sandwich_cost + soda_quantity * soda_cost = 8.38 := 
  by
    sorry

end total_cost_correct_l160_160318


namespace find_line_eq_l160_160789

noncomputable def line_eq (x y : ℝ) : Prop :=
  (∃ a : ℝ, a ≠ 0 ∧ (a * x - y = 0 ∨ x + y - a = 0)) 

theorem find_line_eq : line_eq 2 3 :=
by
  sorry

end find_line_eq_l160_160789


namespace sum_eq_zero_l160_160561

variable {R : Type} [Field R]

-- Define the conditions
def cond1 (a b c : R) : Prop := (a + b) / c = (b + c) / a
def cond2 (a b c : R) : Prop := (b + c) / a = (a + c) / b
def neq (b c : R) : Prop := b ≠ c

-- State the theorem
theorem sum_eq_zero (a b c : R) (h1 : cond1 a b c) (h2 : cond2 a b c) (h3 : neq b c) : a + b + c = 0 := 
by sorry

end sum_eq_zero_l160_160561


namespace prove_math_problem_l160_160689

noncomputable def ellipse_foci : Prop := 
  ∃ (a b : ℝ), 
  a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ),
  (x^2 / a^2 + y^2 / b^2 = 1) → 
  a = 2 ∧ b^2 = 3)

noncomputable def intersect_and_rhombus : Prop :=
  ∃ (m : ℝ) (t : ℝ),
  (3 * m^2 + 4) > 0 ∧ 
  t = 1 / (3 * m^2 + 4) ∧ 
  0 < t ∧ t < 1 / 4

theorem prove_math_problem : ellipse_foci ∧ intersect_and_rhombus :=
by sorry

end prove_math_problem_l160_160689


namespace number_of_correct_propositions_l160_160762

theorem number_of_correct_propositions : 
    (∀ a b : ℝ, a < b → ¬ (a^2 < b^2)) ∧ 
    (∀ a : ℝ, (∀ x : ℝ, |x + 1| + |x - 1| ≥ a ↔ a ≤ 2)) ∧ 
    (¬ (∃ x : ℝ, x^2 - x > 0) ↔ ∀ x : ℝ, x^2 - x ≤ 0) → 
    1 = 1 := 
by
  sorry

end number_of_correct_propositions_l160_160762


namespace avg_score_all_matches_l160_160742

-- Definitions from the conditions
variable (score1 score2 : ℕ → ℕ) 
variable (avg1 avg2 : ℕ)
variable (count1 count2 : ℕ)

-- Assumptions from the conditions
axiom avg_score1 : avg1 = 30
axiom avg_score2 : avg2 = 40
axiom count1_matches : count1 = 2
axiom count2_matches : count2 = 3

-- The proof statement
theorem avg_score_all_matches : 
  ((score1 0 + score1 1) + (score2 0 + score2 1 + score2 2)) / (count1 + count2) = 36 := 
  sorry

end avg_score_all_matches_l160_160742


namespace unit_digit_of_15_pow_100_l160_160451

-- Define a function to extract the unit digit of a number
def unit_digit (n : ℕ) : ℕ := n % 10

-- Given conditions:
def base : ℕ := 15
def exponent : ℕ := 100

-- Define what 'unit_digit' of a number raised to an exponent means
def unit_digit_pow (base exponent : ℕ) : ℕ :=
  unit_digit (base ^ exponent)

-- Goal: Prove that the unit digit of 15^100 is 5.
theorem unit_digit_of_15_pow_100 : unit_digit_pow base exponent = 5 :=
by
  sorry

end unit_digit_of_15_pow_100_l160_160451


namespace find_number_divided_by_3_equals_subtracted_5_l160_160893

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l160_160893


namespace range_of_m_l160_160670

variable (m : ℝ)

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1 }
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 10}

theorem range_of_m (h : A m ∪ B = B) : m ≤ 11 / 3 := by
  sorry

end range_of_m_l160_160670


namespace fourth_watercraft_is_submarine_l160_160500

-- Define the conditions as Lean definitions
def same_direction_speed (w1 w2 w3 w4 : Type) : Prop :=
  -- All watercraft are moving in the same direction at the same speed
  true

def separation (w1 w2 w3 w4 : Type) (d : ℝ) : Prop :=
  -- Each pair of watercraft is separated by distance d
  true

def cargo_ship (w : Type) : Prop := true
def fishing_boat (w : Type) : Prop := true
def passenger_vessel (w : Type) : Prop := true

-- Define that the fourth watercraft is unique
def unique_watercraft (w : Type) : Prop := true

-- Proof statement that the fourth watercraft is a submarine
theorem fourth_watercraft_is_submarine 
  (w1 w2 w3 w4 : Type)
  (h1 : same_direction_speed w1 w2 w3 w4)
  (h2 : separation w1 w2 w3 w4 100)
  (h3 : cargo_ship w1)
  (h4 : fishing_boat w2)
  (h5 : passenger_vessel w3) :
  unique_watercraft w4 := 
sorry

end fourth_watercraft_is_submarine_l160_160500


namespace iron_balls_count_l160_160763

-- Conditions
def length_bar := 12  -- in cm
def width_bar := 8    -- in cm
def height_bar := 6   -- in cm
def num_bars := 10
def volume_iron_ball := 8  -- in cubic cm

-- Calculate the volume of one iron bar
def volume_one_bar := length_bar * width_bar * height_bar

-- Calculate the total volume of the ten iron bars
def total_volume := volume_one_bar * num_bars

-- Calculate the number of iron balls
def num_iron_balls := total_volume / volume_iron_ball

-- The proof statement
theorem iron_balls_count : num_iron_balls = 720 := by
  sorry

end iron_balls_count_l160_160763


namespace cheryl_found_more_eggs_l160_160773

theorem cheryl_found_more_eggs :
  let kevin_eggs := 5
  let bonnie_eggs := 13
  let george_eggs := 9
  let cheryl_eggs := 56
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
  sorry

end cheryl_found_more_eggs_l160_160773


namespace max_triangles_formed_l160_160171

-- Define the triangles and their properties
structure EquilateralTriangle (α : Type) :=
(midpoint_segment : α) -- Each triangle has a segment connecting the midpoints of two sides

variables {α : Type} [OrderedSemiring α]

-- Define the condition of being mirrored horizontally
def areMirroredHorizontally (A B : EquilateralTriangle α) : Prop := 
  -- Placeholder for any formalization needed to specify mirrored horizontally
  sorry

-- Movement conditions and number of smaller triangles
def numberOfSmallerTrianglesAtMaxOverlap (A B : EquilateralTriangle α) (move_horizontally : α) : ℕ :=
  -- Placeholder function/modeling for counting triangles during movement
  sorry

-- Statement of our main theorem
theorem max_triangles_formed (A B : EquilateralTriangle α) (move_horizontally : α) 
  (h_mirrored : areMirroredHorizontally A B) :
  numberOfSmallerTrianglesAtMaxOverlap A B move_horizontally = 11 :=
sorry

end max_triangles_formed_l160_160171


namespace work_completion_time_l160_160001

theorem work_completion_time (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 5) (hC : C = 1 / 20) :
  1 / (A + B + C) = 2 :=
by
  -- Proof goes here
  sorry

end work_completion_time_l160_160001


namespace problem1_problem2_l160_160621

-- Problem 1
theorem problem1 (x : ℝ) : x * (x - 1) - 3 * (x - 1) = 0 → (x = 1) ∨ (x = 3) :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) : x^2 + 2*x - 1 = 0 → (x = -1 + Real.sqrt 2) ∨ (x = -1 - Real.sqrt 2) :=
by sorry

end problem1_problem2_l160_160621


namespace base_of_right_angled_triangle_l160_160014

theorem base_of_right_angled_triangle 
  (height : ℕ) (area : ℕ) (hypotenuse : ℕ) (b : ℕ) 
  (h_height : height = 8)
  (h_area : area = 24)
  (h_hypotenuse : hypotenuse = 10) 
  (h_area_eq : area = (1 / 2 : ℕ) * b * height)
  (h_pythagorean : hypotenuse^2 = height^2 + b^2) : 
  b = 6 := 
sorry

end base_of_right_angled_triangle_l160_160014


namespace stratified_sampling_female_students_l160_160827

-- Definitions from conditions
def male_students : ℕ := 800
def female_students : ℕ := 600
def drawn_male_students : ℕ := 40
def total_students : ℕ := 1400

-- Proof statement
theorem stratified_sampling_female_students : 
  (female_students * drawn_male_students) / male_students = 30 :=
by
  -- substitute and simplify
  sorry

end stratified_sampling_female_students_l160_160827


namespace kim_paid_with_amount_l160_160135

-- Define the conditions
def meal_cost : ℝ := 10
def drink_cost : ℝ := 2.5
def tip_rate : ℝ := 0.20
def change_received : ℝ := 5

-- Define the total amount paid formula
def total_cost_before_tip := meal_cost + drink_cost
def tip_amount := tip_rate * total_cost_before_tip
def total_cost_after_tip := total_cost_before_tip + tip_amount
def amount_paid := total_cost_after_tip + change_received

-- Statement of the theorem
theorem kim_paid_with_amount : amount_paid = 20 := by
  sorry

end kim_paid_with_amount_l160_160135


namespace at_least_six_consecutive_heads_l160_160006

noncomputable def flip_probability : ℚ :=
  let total_outcomes := 2^8
  let successful_outcomes := 7
  successful_outcomes / total_outcomes

theorem at_least_six_consecutive_heads : 
  flip_probability = 7 / 256 :=
by
  sorry

end at_least_six_consecutive_heads_l160_160006


namespace fraction_eq_repeating_decimal_l160_160041

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l160_160041


namespace greatest_m_value_l160_160723

noncomputable def find_greatest_m : ℝ := sorry

theorem greatest_m_value :
  ∃ m : ℝ, 
    (∀ x, x^2 - m * x + 8 = 0 → x ∈ {x | ∃ y, y^2 = 116}) ∧ 
    m = 2 * Real.sqrt 29 :=
sorry

end greatest_m_value_l160_160723


namespace problem1_solution_set_problem2_inequality_l160_160363

theorem problem1_solution_set (x : ℝ) : (-1 < x) ∧ (x < 9) ↔ (|x| + |x - 3| < x + 6) :=
by sorry

theorem problem2_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hn : 9 * x + y = 1) : x + y ≥ 16 * x * y :=
by sorry

end problem1_solution_set_problem2_inequality_l160_160363


namespace product_trippled_when_added_to_reciprocal_l160_160594

theorem product_trippled_when_added_to_reciprocal :
  (∀ x ∈ ℝ, x + x⁻¹ = 3 * x → ∃ m n ∈ ℝ, m ≠ n ∧ x = m ∨ x = n ∧ m * n = -1 / 2) :=
by
  sorry

end product_trippled_when_added_to_reciprocal_l160_160594


namespace ineq_x4_y4_l160_160848

theorem ineq_x4_y4 (x y : ℝ) (h1 : x > Real.sqrt 2) (h2 : y > Real.sqrt 2) :
    x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by
  sorry

end ineq_x4_y4_l160_160848


namespace max_blocks_fit_l160_160176

theorem max_blocks_fit :
  ∃ (blocks : ℕ), blocks = 12 ∧ 
  (∀ (a b c : ℕ), a = 3 ∧ b = 2 ∧ c = 1 → 
  ∀ (x y z : ℕ), x = 5 ∧ y = 4 ∧ z = 4 → 
  blocks = (x * y * z) / (a * b * c) ∧
  blocks = (y * z / (b * c) * (5 / a))) :=
sorry

end max_blocks_fit_l160_160176


namespace factorized_sum_is_33_l160_160858

theorem factorized_sum_is_33 (p q r : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 21 * x + 110 = (x + p) * (x + q))
  (h2 : ∀ x : ℤ, x^2 - 23 * x + 132 = (x - q) * (x - r)) : 
  p + q + r = 33 := by
  sorry

end factorized_sum_is_33_l160_160858


namespace cos_pi_over_2_plus_2theta_l160_160793

theorem cos_pi_over_2_plus_2theta (θ : ℝ) (hcos : Real.cos θ = 1 / 3) (hθ : 0 < θ ∧ θ < Real.pi) :
    Real.cos (Real.pi / 2 + 2 * θ) = - (4 * Real.sqrt 2) / 9 := 
sorry

end cos_pi_over_2_plus_2theta_l160_160793


namespace product_of_triple_when_added_to_reciprocal_l160_160579

theorem product_of_triple_when_added_to_reciprocal :
  let S := {x : ℝ | x + (1 / x) = 3 * x} in
  ∏ x in S, x = -1 / 2 :=
by
  sorry

end product_of_triple_when_added_to_reciprocal_l160_160579


namespace isosceles_trapezoid_height_l160_160710

/-- Given an isosceles trapezoid with area 100 and diagonals that are mutually perpendicular,
    we want to prove that the height of the trapezoid is 10. -/
theorem isosceles_trapezoid_height (BC AD h : ℝ) 
    (area_eq_100 : 100 = (1 / 2) * (BC + AD) * h)
    (height_eq_half_sum : h = (1 / 2) * (BC + AD)) :
    h = 10 :=
by
  sorry

end isosceles_trapezoid_height_l160_160710


namespace steve_more_than_wayne_first_time_at_2004_l160_160470

def initial_steve_money (year: ℕ) := if year = 2000 then 100 else 0
def initial_wayne_money (year: ℕ) := if year = 2000 then 10000 else 0

def steve_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_steve_money year
  else 2 * steve_money (year - 1)

def wayne_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_wayne_money year
  else wayne_money (year - 1) / 2

theorem steve_more_than_wayne_first_time_at_2004 :
  ∃ (year: ℕ), year = 2004 ∧ steve_money year > wayne_money year := by
  sorry

end steve_more_than_wayne_first_time_at_2004_l160_160470


namespace prob_at_least_6_heads_in_8_flips_l160_160005

def fairCoinFlipProb : ℕ -> ℚ
| 8 := 17 / 256
| _ := 0

theorem prob_at_least_6_heads_in_8_flips : fairCoinFlipProb 8 = 17 / 256 :=
by
  sorry

end prob_at_least_6_heads_in_8_flips_l160_160005


namespace fraction_for_repeating_56_l160_160038

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l160_160038


namespace train_speeds_l160_160206

theorem train_speeds (v t : ℕ) (h1 : t = 1)
  (h2 : v + v * t = 90)
  (h3 : 90 * t = 90) :
  v = 45 := by
  sorry

end train_speeds_l160_160206


namespace tank_capacity_l160_160756

theorem tank_capacity (w c: ℚ) (h1: w / c = 1 / 3) (h2: (w + 5) / c = 2 / 5) :
  c = 37.5 := 
by
  sorry

end tank_capacity_l160_160756


namespace alpha_plus_2beta_l160_160801

noncomputable def sin_square (θ : ℝ) := (Real.sin θ)^2
noncomputable def sin_double (θ : ℝ) := Real.sin (2 * θ)

theorem alpha_plus_2beta (α β : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) 
(hβ : 0 < β ∧ β < Real.pi / 2) 
(h1 : 3 * sin_square α + 2 * sin_square β = 1)
(h2 : 3 * sin_double α - 2 * sin_double β = 0) : 
α + 2 * β = 5 * Real.pi / 6 :=
by
  sorry

end alpha_plus_2beta_l160_160801


namespace two_std_dev_less_than_mean_l160_160741

def mean : ℝ := 14.0
def std_dev : ℝ := 1.5

theorem two_std_dev_less_than_mean : (mean - 2 * std_dev) = 11.0 := 
by sorry

end two_std_dev_less_than_mean_l160_160741


namespace factorize_x_cubed_minus_9x_l160_160777

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l160_160777


namespace sum_of_interior_edges_l160_160631

theorem sum_of_interior_edges (frame_width : ℕ) (frame_area : ℕ) (outer_edge : ℕ) 
  (H1 : frame_width = 2) (H2 : frame_area = 32) (H3 : outer_edge = 7) : 
  2 * (outer_edge - 2 * frame_width) + 2 * (x : ℕ) = 8 :=
by
  sorry

end sum_of_interior_edges_l160_160631


namespace value_of_coefficients_l160_160234

theorem value_of_coefficients (a₀ a₁ a₂ a₃ : ℤ) (x : ℤ) :
  (5 * x + 4) ^ 3 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 →
  x = -1 →
  (a₀ + a₂) - (a₁ + a₃) = -1 :=
by
  sorry

end value_of_coefficients_l160_160234


namespace factorials_sum_of_two_squares_l160_160784

-- Define what it means for a number to be a sum of two squares.
def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

theorem factorials_sum_of_two_squares :
  {n : ℕ | n < 14 ∧ is_sum_of_two_squares (n!)} = {2, 6} :=
by
  sorry

end factorials_sum_of_two_squares_l160_160784


namespace trig_identity_l160_160521

theorem trig_identity (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : 
  Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by 
  sorry

end trig_identity_l160_160521


namespace motorcycle_speed_for_10_minute_prior_arrival_l160_160619

noncomputable def distance_from_home_to_station (x : ℝ) : Prop :=
  x / 30 + 15 / 60 = x / 18 - 15 / 60

noncomputable def speed_to_arrive_10_minutes_before_departure (x : ℝ) (v : ℝ) : Prop :=
  v = x / (1 - 10 / 60)

theorem motorcycle_speed_for_10_minute_prior_arrival :
  (∀ x : ℝ, distance_from_home_to_station x) →
  (∃ x : ℝ, 
    ∃ v : ℝ, speed_to_arrive_10_minutes_before_departure x v ∧ v = 27) :=
by 
  intro h
  exists 22.5
  exists 27
  unfold distance_from_home_to_station at h
  unfold speed_to_arrive_10_minutes_before_departure
  sorry

end motorcycle_speed_for_10_minute_prior_arrival_l160_160619


namespace product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l160_160606

theorem product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half
  (x : ℝ) (hx : x + 1/x = 3 * x) :
  ∏ (sol : {x : ℝ // x + 1/x = 3 * x}) in { (1 / (real.sqrt 2)), (-1 / (real.sqrt 2))}, sol = -1/2 :=
by
  sorry

end product_of_real_numbers_tripled_when_added_to_reciprocal_is_minus_half_l160_160606


namespace consecutive_integers_sqrt19_sum_l160_160984

theorem consecutive_integers_sqrt19_sum :
  ∃ a b : ℤ, (a < ⌊Real.sqrt 19⌋ ∧ ⌊Real.sqrt 19⌋ < b ∧ a + 1 = b) ∧ a + b = 9 := 
by
  sorry

end consecutive_integers_sqrt19_sum_l160_160984


namespace tire_price_l160_160010

theorem tire_price (x : ℝ) (h : 3 * x + 10 = 310) : x = 100 :=
sorry

end tire_price_l160_160010


namespace find_p_l160_160264

noncomputable def area_of_ABC (p : ℚ) : ℚ :=
  128 - 6 * p

theorem find_p (p : ℚ) : area_of_ABC p = 45 → p = 83 / 6 := by
  intro h
  sorry

end find_p_l160_160264


namespace team_savings_correct_l160_160487

-- Define the costs without the discount
def cost_shirt := 7.50
def cost_pants := 15.00
def cost_socks := 4.50

-- Define the costs with the discount
def discounted_shirt := 6.75
def discounted_pants := 13.50
def discounted_socks := 3.75

-- Define the number of team members
def team_members := 12

-- Total cost of one uniform without discount
def total_cost_without_discount := cost_shirt + cost_pants + cost_socks

-- Total cost of one uniform with discount
def total_cost_with_discount := discounted_shirt + discounted_pants + discounted_socks

-- Savings per uniform
def savings_per_uniform := total_cost_without_discount - total_cost_with_discount

-- Total savings for the team
def total_savings_for_team := savings_per_uniform * team_members

-- Prove that the total savings for the team is $36.00
theorem team_savings_correct : total_savings_for_team = 36.00 := 
  by 
    sorry

end team_savings_correct_l160_160487


namespace repeating_decimal_equiv_fraction_l160_160094

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l160_160094


namespace probability_at_least_6_heads_l160_160004

theorem probability_at_least_6_heads (n : ℕ) (p : ℚ) : n = 8 ∧ p = (3 / 128) :=
  let total_outcomes := 2 ^ n in
  let successful_outcomes := 3 + 2 + 1 in
  n = 8 ∧ total_outcomes = 256 ∧ p = (successful_outcomes.to_rat / total_outcomes.to_rat) := sorry

end probability_at_least_6_heads_l160_160004


namespace tank_capacity_75_l160_160754

theorem tank_capacity_75 (c w : ℝ) 
  (h₁ : w = c / 3) 
  (h₂ : (w + 5) / c = 2 / 5) : 
  c = 75 := 
  sorry

end tank_capacity_75_l160_160754


namespace probability_of_X_conditioned_l160_160572

variables (P_X P_Y P_XY : ℝ)

-- Conditions
def probability_of_Y : Prop := P_Y = 2/5
def probability_of_XY : Prop := P_XY = 0.05714285714285714
def independent_selection : Prop := P_XY = P_X * P_Y

-- Theorem statement
theorem probability_of_X_conditioned (P_X P_Y P_XY : ℝ) 
  (h1 : probability_of_Y P_Y) 
  (h2 : probability_of_XY P_XY) 
  (h3 : independent_selection P_X P_Y P_XY) :
  P_X = 0.14285714285714285 := 
sorry

end probability_of_X_conditioned_l160_160572


namespace fraction_for_repeating_56_l160_160036

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l160_160036


namespace height_cylinder_l160_160337

variables (r_c h_c r_cy h_cy : ℝ)
variables (V_cone V_cylinder : ℝ)
variables (r_c_val : r_c = 15)
variables (h_c_val : h_c = 20)
variables (r_cy_val : r_cy = 30)
variables (V_cone_eq : V_cone = (1/3) * π * r_c^2 * h_c)
variables (V_cylinder_eq : V_cylinder = π * r_cy^2 * h_cy)

theorem height_cylinder : h_cy = 1.67 :=
by
  rw [r_c_val, h_c_val, r_cy_val] at *
  have V_cone := V_cone_eq
  have V_cylinder := V_cylinder_eq
  sorry

end height_cylinder_l160_160337


namespace product_of_consecutive_integers_l160_160694

theorem product_of_consecutive_integers (l : List ℤ) (h1 : l.length = 2019) (h2 : l.sum = 2019) : l.prod = 0 := 
sorry

end product_of_consecutive_integers_l160_160694


namespace factorize_x_cubed_minus_9x_l160_160775

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l160_160775


namespace sum_of_cubes_eq_neg2_l160_160255

theorem sum_of_cubes_eq_neg2 (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 1) : a^3 + b^3 = -2 := 
sorry

end sum_of_cubes_eq_neg2_l160_160255


namespace rentExpenses_l160_160468

noncomputable def monthlySalary : ℝ := 23000
noncomputable def milkExpenses : ℝ := 1500
noncomputable def groceriesExpenses : ℝ := 4500
noncomputable def educationExpenses : ℝ := 2500
noncomputable def petrolExpenses : ℝ := 2000
noncomputable def miscellaneousExpenses : ℝ := 5200
noncomputable def savings : ℝ := 2300

-- Calculating total non-rent expenses
noncomputable def totalNonRentExpenses : ℝ :=
  milkExpenses + groceriesExpenses + educationExpenses + petrolExpenses + miscellaneousExpenses

-- The rent expenses theorem
theorem rentExpenses : totalNonRentExpenses + savings + 5000 = monthlySalary :=
by sorry

end rentExpenses_l160_160468


namespace number_of_ordered_triples_l160_160696

open Finset Nat

-- Define the set S
def S : Finset ℕ := (range 66).filter (λ n, n > 0)

-- Define the problem statement
theorem number_of_ordered_triples (hS : ∀ n ∈ S, n ≤ 65) :
  (S.card = 65) →
  (∑ z in S, ∑ x in (S.filter (λ t, t < z)), ∑ y in (S.filter (λ t, t < z)), if (x < y) then 2 else 1) = 89440 :=
by
  sorry

end number_of_ordered_triples_l160_160696


namespace factorization_m_minus_n_l160_160660

theorem factorization_m_minus_n :
  ∃ (m n : ℤ), (6 * (x:ℝ)^2 - 5 * x - 6 = (6 * x + m) * (x + n)) ∧ (m - n = 5) :=
by {
  sorry
}

end factorization_m_minus_n_l160_160660


namespace minimum_discount_l160_160194

open Real

theorem minimum_discount (CP MP SP_min : ℝ) (profit_margin : ℝ) (discount : ℝ) :
  CP = 800 ∧ MP = 1200 ∧ SP_min = 960 ∧ profit_margin = 0.20 ∧
  MP * (1 - discount / 100) ≥ SP_min → discount = 20 :=
by
  intros h
  rcases h with ⟨h_cp, h_mp, h_sp_min, h_profit_margin, h_selling_price⟩
  simp [h_cp, h_mp, h_sp_min, h_profit_margin, sub_eq_self, div_eq_self] at *
  sorry

end minimum_discount_l160_160194


namespace A_alone_days_l160_160925

variable (r_A r_B r_C : ℝ)

-- Given conditions:
axiom cond1 : r_A + r_B = 1 / 3
axiom cond2 : r_B + r_C = 1 / 6
axiom cond3 : r_A + r_C = 4 / 15

-- Proposition stating the required proof, that A alone can do the job in 60/13 days:
theorem A_alone_days : r_A ≠ 0 → 1 / r_A = 60 / 13 :=
by
  intro h
  sorry

end A_alone_days_l160_160925


namespace repeating_decimal_equiv_fraction_l160_160091

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l160_160091


namespace no_partition_of_positive_integers_l160_160408

theorem no_partition_of_positive_integers :
  ∀ (A B C : Set ℕ), (∀ (x : ℕ), x ∈ A ∨ x ∈ B ∨ x ∈ C) →
  (∀ (x y : ℕ), x ∈ A ∧ y ∈ B → x^2 - x * y + y^2 ∈ C) →
  (∀ (x y : ℕ), x ∈ B ∧ y ∈ C → x^2 - x * y + y^2 ∈ A) →
  (∀ (x y : ℕ), x ∈ C ∧ y ∈ A → x^2 - x * y + y^2 ∈ B) →
  False := 
sorry

end no_partition_of_positive_integers_l160_160408


namespace quotient_calculation_l160_160735

theorem quotient_calculation (dividend divisor remainder expected_quotient : ℕ)
  (h₁ : dividend = 166)
  (h₂ : divisor = 18)
  (h₃ : remainder = 4)
  (h₄ : dividend = divisor * expected_quotient + remainder) :
  expected_quotient = 9 :=
by
  sorry

end quotient_calculation_l160_160735


namespace seating_arrangement_count_l160_160828

/-- In a row of 9 seats, three distinct people, A, B, and C, need to be seated such that each has empty seats to their left and right, with A seated between B and C. Prove the total number of different seating arrangements is 20. -/
theorem seating_arrangement_count :
  let seats : Finset ℕ := Finset.range 9 in
  let possible_positions (x : ℕ) := Finset.filter (λ n, n > 0 ∧ n < 8) seats in
  let count_positions (x : ℕ) := (possible_positions x).card in
  let arrangement_count :=
    (possible_positions A).sum (λ a, (Finset.filter (λ n, n ≠ a ∧ n ≠ a + 1 ∧ n ≠ a - 1) seats).card * 2) in
  arrangement_count = 20 :=
by
  sorry

end seating_arrangement_count_l160_160828


namespace subtract_largest_unit_fraction_l160_160407

theorem subtract_largest_unit_fraction
  (a b n : ℕ) (ha : a > 0) (hb : b > a) (hn : 1 ≤ b * n ∧ b * n <= a * n + b): 
  (a * n - b < a) := by
  sorry

end subtract_largest_unit_fraction_l160_160407


namespace find_multiple_l160_160327

theorem find_multiple (x m : ℤ) (hx : x = 13) (h : x + x + 2 * x + m * x = 104) : m = 4 :=
by
  -- Proof to be provided
  sorry

end find_multiple_l160_160327


namespace expression_equivalence_l160_160297

def algebraicExpression : String := "5 - 4a"
def wordExpression : String := "the difference of 5 and 4 times a"

theorem expression_equivalence : algebraicExpression = wordExpression := 
sorry

end expression_equivalence_l160_160297


namespace fraction_for_repeating_56_l160_160035

theorem fraction_for_repeating_56 : 
  let x := 0.56565656 in
  x = 56 / 99 := sorry

end fraction_for_repeating_56_l160_160035


namespace addition_even_odd_is_odd_subtraction_even_odd_is_odd_squared_sum_even_odd_is_odd_l160_160240

section OperationsAlwaysYieldOdd

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem addition_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd (a + b) :=
sorry

theorem subtraction_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd (a - b) :=
sorry

theorem squared_sum_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd ((a + b) * (a + b)) :=
sorry

end OperationsAlwaysYieldOdd

end addition_even_odd_is_odd_subtraction_even_odd_is_odd_squared_sum_even_odd_is_odd_l160_160240


namespace repeating_decimal_is_fraction_l160_160059

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l160_160059


namespace second_machine_completion_time_l160_160008

variable (time_first_machine : ℝ) (rate_first_machine : ℝ) (rate_combined : ℝ)
variable (rate_second_machine: ℝ) (y : ℝ)

def processing_rate_first_machine := rate_first_machine = 100
def processing_rate_combined := rate_combined = 1000 / 3
def processing_rate_second_machine := rate_second_machine = rate_combined - rate_first_machine
def completion_time_second_machine := y = 1000 / rate_second_machine

theorem second_machine_completion_time
  (h1: processing_rate_first_machine rate_first_machine)
  (h2: processing_rate_combined rate_combined)
  (h3: processing_rate_second_machine rate_combined rate_first_machine rate_second_machine)
  (h4: completion_time_second_machine rate_second_machine y) :
  y = 30 / 7 :=
sorry

end second_machine_completion_time_l160_160008


namespace sample_size_9_l160_160256

variable (X : Nat)

theorem sample_size_9 (h : 36 % X = 0 ∧ 36 % (X + 1) ≠ 0) : X = 9 := 
sorry

end sample_size_9_l160_160256


namespace yellow_candy_percentage_l160_160456

variable (b : ℝ) (y : ℝ) (r : ℝ)

-- Conditions from the problem
-- 14% more yellow candies than blue candies
axiom yellow_candies : y = 1.14 * b
-- 14% fewer red candies than blue candies
axiom red_candies : r = 0.86 * b
-- Total number of candies equals 1 (or 100%)
axiom total_candies : r + b + y = 1

-- Question to prove: The percentage of yellow candies in the jar is 38%
theorem yellow_candy_percentage  : y = 0.38 := by
  sorry

end yellow_candy_percentage_l160_160456


namespace largest_integer_odd_divides_expression_l160_160368

theorem largest_integer_odd_divides_expression (x : ℕ) (h_odd : x % 2 = 1) : 
    ∃ k, k = 384 ∧ ∀ m, m ∣ (8*x + 6) * (8*x + 10) * (4*x + 4) → m ≤ k :=
by {
  sorry
}

end largest_integer_odd_divides_expression_l160_160368


namespace total_sum_l160_160759

theorem total_sum (p q r s t : ℝ) (P : ℝ) 
  (h1 : q = 0.75 * P) 
  (h2 : r = 0.50 * P) 
  (h3 : s = 0.25 * P) 
  (h4 : t = 0.10 * P) 
  (h5 : s = 25) 
  :
  p + q + r + s + t = 260 :=
by 
  sorry

end total_sum_l160_160759


namespace train_speed_in_km_hr_l160_160946

noncomputable def train_length : ℝ := 320
noncomputable def crossing_time : ℝ := 7.999360051195905
noncomputable def speed_in_meter_per_sec : ℝ := train_length / crossing_time
noncomputable def meter_per_sec_to_km_hr (speed_mps : ℝ) : ℝ := speed_mps * 3.6
noncomputable def expected_speed : ℝ := 144.018001125

theorem train_speed_in_km_hr :
  meter_per_sec_to_km_hr speed_in_meter_per_sec = expected_speed := by
  sorry

end train_speed_in_km_hr_l160_160946


namespace rabbit_carrot_count_l160_160956

theorem rabbit_carrot_count
  (r h : ℕ)
  (hr : r = h - 3)
  (eq_carrots : 4 * r = 5 * h) :
  4 * r = 36 :=
by
  sorry

end rabbit_carrot_count_l160_160956


namespace robot_paths_from_A_to_B_l160_160940

/-- Define a function that computes the number of distinct paths a robot can take -/
def distinctPaths (A B : ℕ × ℕ) : ℕ := sorry

/-- Proof statement: There are 556 distinct paths from A to B, given the movement conditions -/
theorem robot_paths_from_A_to_B (A B : ℕ × ℕ) (h_move : (A, B) = ((0, 0), (10, 10))) :
  distinctPaths A B = 556 :=
sorry

end robot_paths_from_A_to_B_l160_160940


namespace min_cuts_for_payment_7_days_l160_160181

theorem min_cuts_for_payment_7_days (n : ℕ) (h : n = 7) : ∃ k, k = 1 :=
by sorry

end min_cuts_for_payment_7_days_l160_160181


namespace geometric_seq_fourth_term_l160_160421

theorem geometric_seq_fourth_term (a r : ℝ) (h_a : a = 1024) (h_r_pow : a * r ^ 5 = 125) :
  a * r ^ 3 = 2000 := by
  sorry

end geometric_seq_fourth_term_l160_160421


namespace multiplication_correct_l160_160453

theorem multiplication_correct : 3795421 * 8634.25 = 32774670542.25 := by
  sorry

end multiplication_correct_l160_160453


namespace repeating_decimal_as_fraction_l160_160050

noncomputable def repeating_decimal := 0.56565656 -- indicating the repeating decimal

def first_term : ℚ := 56 / 100 -- first term of the geometric series
def common_ratio : ℚ := 1 / 100 -- common ratio of the geometric series

theorem repeating_decimal_as_fraction : repeating_decimal = 56 / 99 := sorry

end repeating_decimal_as_fraction_l160_160050


namespace problem_statement_l160_160101

theorem problem_statement :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
    (∀ x : ℝ, 1 + x^5 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + 
              a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) ∧
    (a_0 = 2) ∧
    (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 33)) →
  (∃ a_1 a_2 a_3 a_4 a_5 : ℝ, a_1 + a_2 + a_3 + a_4 + a_5 = 31) :=
by
  sorry

end problem_statement_l160_160101


namespace find_pq_l160_160834

-- Define the constants function for the given equation and form
noncomputable def quadratic_eq (p q r : ℤ) : (ℤ × ℤ × ℤ) :=
(2*p*q, p^2 + 2*p*q + q^2 + r, q*q + r)

-- Define the theorem we want to prove
theorem find_pq (p q r: ℤ) (h : quadratic_eq 2 q r = (8, -24, -56)) : pq = -12 :=
by sorry

end find_pq_l160_160834


namespace constant_term_expanded_eq_neg12_l160_160232

theorem constant_term_expanded_eq_neg12
  (a w c d : ℤ)
  (h_eq : (a * x + w) * (c * x + d) = 6 * x ^ 2 + x - 12)
  (h_abs_sum : abs a + abs w + abs c + abs d = 12) :
  w * d = -12 := by
  sorry

end constant_term_expanded_eq_neg12_l160_160232


namespace suitable_land_acres_l160_160536

theorem suitable_land_acres (new_multiplier : ℝ) (previous_acres : ℝ) (pond_acres : ℝ) :
  new_multiplier = 10 ∧ previous_acres = 2 ∧ pond_acres = 1 → 
  (new_multiplier * previous_acres - pond_acres) = 19 :=
by
  intro h
  sorry

end suitable_land_acres_l160_160536


namespace shopkeeper_loss_percentage_l160_160636

theorem shopkeeper_loss_percentage
  (total_stock_value : ℝ)
  (overall_loss : ℝ)
  (first_part_percentage : ℝ)
  (first_part_profit_percentage : ℝ)
  (remaining_part_loss : ℝ)
  (total_worth_first_part : ℝ)
  (first_part_profit : ℝ)
  (remaining_stock_value : ℝ)
  (remaining_stock_loss : ℝ)
  (loss_percentage : ℝ) :
  total_stock_value = 16000 →
  overall_loss = 400 →
  first_part_percentage = 0.10 →
  first_part_profit_percentage = 0.20 →
  total_worth_first_part = total_stock_value * first_part_percentage →
  first_part_profit = total_worth_first_part * first_part_profit_percentage →
  remaining_stock_value = total_stock_value * (1 - first_part_percentage) →
  remaining_stock_loss = overall_loss + first_part_profit →
  loss_percentage = (remaining_stock_loss / remaining_stock_value) * 100 →
  loss_percentage = 5 :=
by intros; sorry

end shopkeeper_loss_percentage_l160_160636


namespace gregory_current_age_l160_160220

-- Given conditions
variables (D G y : ℕ)
axiom dm_is_three_times_greg_was (x : ℕ) : D = 3 * y
axiom future_age_sum : D + (3 * y) = 49
axiom greg_age_difference x y : D - (3 * y) = (3 * y) - x

-- Prove statement: Gregory's current age is 14
theorem gregory_current_age : G = 14 := by
  sorry

end gregory_current_age_l160_160220


namespace good_arrangement_iff_coprime_l160_160732

-- Definitions for the concepts used
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_good_arrangement (n m : ℕ) : Prop :=
  ∃ k₀, ∀ i, (n * k₀ * i) % (m + n) = (i % (m + n))

theorem good_arrangement_iff_coprime (n m : ℕ) : is_good_arrangement n m ↔ is_coprime n m := 
sorry

end good_arrangement_iff_coprime_l160_160732


namespace arithmetic_identity_l160_160213

theorem arithmetic_identity :
  65 * 1515 - 25 * 1515 + 1515 = 62115 :=
by
  sorry

end arithmetic_identity_l160_160213


namespace solve_xyz_system_l160_160388

theorem solve_xyz_system :
  ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 
    (x * (6 - y) = 9) ∧ 
    (y * (6 - z) = 9) ∧ 
    (z * (6 - x) = 9) ∧ 
    x = 3 ∧ y = 3 ∧ z = 3 :=
by
  sorry

end solve_xyz_system_l160_160388


namespace find_number_l160_160882

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l160_160882


namespace candy_boxes_system_l160_160467

-- Given conditions and definitions
def sheets_total (x y : ℕ) : Prop := x + y = 35
def sheet_usage (x y : ℕ) : Prop := 20 * x = 30 * y / 2

-- Statement
theorem candy_boxes_system (x y : ℕ) (h1 : sheets_total x y) (h2 : sheet_usage x y) : 
  (x + y = 35) ∧ (20 * x = 30 * y / 2) := 
by
sorry

end candy_boxes_system_l160_160467


namespace mixed_number_calculation_l160_160209

/-
  We need to define a proof that shows:
  75 * (2 + 3/7 - 5 * (1/3)) / (3 + 1/5 + 2 + 1/6) = -208 + 7/9
-/
theorem mixed_number_calculation :
  75 * ((17 / 7) - (16 / 3)) / ((16 / 5) + (13 / 6)) = -208 + 7 / 9 := by
  sorry

end mixed_number_calculation_l160_160209


namespace plane_distance_l160_160200

theorem plane_distance (D : ℕ) (h₁ : D / 300 + D / 400 = 7) : D = 1200 :=
sorry

end plane_distance_l160_160200


namespace range_of_m_l160_160991

-- Define the quadratic function f
def f (a c x : ℝ) := a * x^2 - 2 * a * x + c

-- State the theorem
theorem range_of_m (a c : ℝ) (h : f a c 2017 < f a c (-2016)) (m : ℝ) 
  : f a c m ≤ f a c 0 → 0 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l160_160991


namespace michael_points_scored_l160_160527

theorem michael_points_scored (team_points : ℕ) (other_players : ℕ) (average_points : ℕ) (michael_points : ℕ) :
  team_points = 72 → other_players = 8 → average_points = 9 → 
  michael_points = team_points - other_players * average_points → michael_points = 36 :=
by
  intro h_team_points h_other_players h_average_points h_calculation
  -- skip the actual proof for now
  sorry

end michael_points_scored_l160_160527


namespace quadratics_common_root_square_sum_6_l160_160358

theorem quadratics_common_root_square_sum_6
  (a b c : ℝ)
  (h_distinct: a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_common_root_1: ∃ x1, x1^2 + a * x1 + b = 0 ∧ x1^2 + b * x1 + c = 0)
  (h_common_root_2: ∃ x2, x2^2 + b * x2 + c = 0 ∧ x2^2 + c * x2 + a = 0)
  (h_common_root_3: ∃ x3, x3^2 + c * x3 + a = 0 ∧ x3^2 + a * x3 + b = 0) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end quadratics_common_root_square_sum_6_l160_160358


namespace cookies_per_sheet_is_16_l160_160750

-- Define the number of members
def members : ℕ := 100

-- Define the number of sheets each member bakes
def sheets_per_member : ℕ := 10

-- Define the total number of cookies baked
def total_cookies : ℕ := 16000

-- Calculate the total number of sheets baked
def total_sheets : ℕ := members * sheets_per_member

-- Define the number of cookies per sheet as a result of given conditions
def cookies_per_sheet : ℕ := total_cookies / total_sheets

-- Prove that the number of cookies on each sheet is 16 given the conditions
theorem cookies_per_sheet_is_16 : cookies_per_sheet = 16 :=
by
  -- Assuming all the given definitions and conditions
  sorry

end cookies_per_sheet_is_16_l160_160750


namespace domain_of_tan_l160_160154

noncomputable def is_excluded_from_domain (x : ℝ) : Prop :=
  ∃ k : ℤ, x = 1 + 6 * k

theorem domain_of_tan {x : ℝ} :
  ∀ x, ¬ is_excluded_from_domain x ↔ ¬ ∃ k : ℤ, x = 1 + 6 * k := 
by 
  sorry

end domain_of_tan_l160_160154


namespace batsman_average_l160_160316

theorem batsman_average (A : ℕ) (H : (16 * A + 82) / 17 = A + 3) : (A + 3 = 34) :=
sorry

end batsman_average_l160_160316


namespace frac_sum_eq_l160_160681

theorem frac_sum_eq (a b : ℝ) (h1 : a^2 + a - 1 = 0) (h2 : b^2 + b - 1 = 0) : 
  (a / b + b / a = 2) ∨ (a / b + b / a = -3) := 
sorry

end frac_sum_eq_l160_160681


namespace angle_sum_triangle_l160_160266

theorem angle_sum_triangle (A B C : ℝ) (hA : A = 75) (hB : B = 40) (h_sum : A + B + C = 180) : C = 65 :=
by
  sorry

end angle_sum_triangle_l160_160266


namespace smallest_b_factors_l160_160100

theorem smallest_b_factors 
: ∃ b : ℕ, b > 0 ∧ 
    (∃ p q : ℤ, x^2 + b * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) ∧ 
    ∀ b': ℕ, (∃ p q: ℤ, x^2 + b' * x + 1760 = (x + p) * (x + q) ∧ p * q = 1760) → (b ≤ b') := 
sorry

end smallest_b_factors_l160_160100


namespace reading_club_coordinator_selection_l160_160018

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem reading_club_coordinator_selection :
  let total_ways := choose 18 4
  let no_former_ways := choose 10 4
  total_ways - no_former_ways = 2850 := by
  sorry

end reading_club_coordinator_selection_l160_160018


namespace find_p_l160_160668

noncomputable section
open ProbabilityTheory

theorem find_p
  (X : ℕ → ℝ)
  (h1 : ∀ n, X n = binomial 2 p)
  (h2 : P (set_of (λ x, X x ≥ 1)) = 3 / 4)
  : p = 1 / 2 := by
  sorry

end find_p_l160_160668


namespace road_signs_count_l160_160203

theorem road_signs_count (n1 n2 n3 n4 : ℕ) (h1 : n1 = 40) (h2 : n2 = n1 + n1 / 4) (h3 : n3 = 2 * n2) (h4 : n4 = n3 - 20) : 
  n1 + n2 + n3 + n4 = 270 := 
by
  sorry

end road_signs_count_l160_160203


namespace sum_of_transformed_numbers_l160_160435

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
    3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l160_160435


namespace number_of_long_sleeved_jerseys_l160_160273

def cost_per_long_sleeved := 15
def cost_per_striped := 10
def num_striped_jerseys := 2
def total_spent := 80

theorem number_of_long_sleeved_jerseys (x : ℕ) :
  total_spent = cost_per_long_sleeved * x + cost_per_striped * num_striped_jerseys →
  x = 4 := by
  sorry

end number_of_long_sleeved_jerseys_l160_160273


namespace find_third_circle_radius_l160_160876

-- Define the context of circles and their tangency properties
variable (A B : ℝ → ℝ → Prop) -- Centers of circles
variable (r1 r2 : ℝ) -- Radii of circles

-- Define conditions from the problem
def circles_are_tangent (A B : ℝ → ℝ → Prop) (r1 r2 : ℝ) : Prop :=
  ∀ x y : ℝ, A x y → B (x + 7) y ∧ r1 = 2 ∧ r2 = 5

def third_circle_tangent_to_others_and_tangent_line (A B : ℝ → ℝ → Prop) (r3 : ℝ) : Prop :=
  ∃ D : ℝ → ℝ → Prop, ∀ x y : ℝ, D x y →
  ((A (x + r3) y ∧ B (x - r3) y) ∧ (r3 > 0))

theorem find_third_circle_radius (A B : ℝ → ℝ → Prop) (r1 r2 : ℝ) :
  circles_are_tangent A B r1 r2 →
  (∃ r3 : ℝ, r3 = 1 ∧ third_circle_tangent_to_others_and_tangent_line A B r3) :=
by
  sorry

end find_third_circle_radius_l160_160876


namespace intersection_point_of_lines_l160_160576

theorem intersection_point_of_lines :
  let line1 (x : ℝ) := 3 * x - 4
  let line2 (x : ℝ) := - (1 / 3) * x + 5
  (∃ x y : ℝ, line1 x = y ∧ line2 x = y ∧ x = 2.7 ∧ y = 4.1) :=
by {
    sorry
}

end intersection_point_of_lines_l160_160576


namespace probability_first_ace_equal_l160_160825

theorem probability_first_ace_equal (num_cards : ℕ) (num_aces : ℕ) (num_players : ℕ)
  (h1 : num_cards = 32) (h2 : num_aces = 4) (h3 : num_players = 4) :
  ∀ player : ℕ, player ∈ {1, 2, 3, 4} → (∃ positions : list ℕ, (∀ n ∈ positions, n % num_players = player - 1)) → 
  (positions.length = 8) →
  let P := 1 / 8 in
  P = 1 / num_players :=
begin
  sorry
end

end probability_first_ace_equal_l160_160825


namespace least_money_Moe_l160_160474

theorem least_money_Moe (Bo Coe Flo Jo Moe Zoe : ℝ)
  (H1 : Flo > Jo) 
  (H2 : Flo > Bo) 
  (H3 : Bo > Zoe) 
  (H4 : Coe > Zoe) 
  (H5 : Jo > Zoe) 
  (H6 : Bo > Jo) 
  (H7 : Zoe > Moe) : 
  (Moe < Bo) ∧ (Moe < Coe) ∧ (Moe < Flo) ∧ (Moe < Jo) ∧ (Moe < Zoe) :=
by
  sorry

end least_money_Moe_l160_160474


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l160_160600

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l160_160600


namespace total_number_of_coins_l160_160947

theorem total_number_of_coins (x : ℕ) (h : 1 * x + 5 * x + 10 * x + 50 * x + 100 * x = 332) : 5 * x = 10 :=
by {
  sorry
}

end total_number_of_coins_l160_160947


namespace min_candies_to_remove_l160_160436

theorem min_candies_to_remove {n : ℕ} (h : n = 31) : (∃ k, (n - k) % 5 = 0) → k = 1 :=
by
  sorry

end min_candies_to_remove_l160_160436


namespace allocation_schemes_l160_160431

theorem allocation_schemes (students factories: ℕ) (has_factory_a: Prop) (A_must_have_students: has_factory_a): students = 3 → factories = 4 → has_factory_a → (∃ n: ℕ, n = 4^3 - 3^3 ∧ n = 37) :=
by try { sorry }

end allocation_schemes_l160_160431


namespace frustum_volume_and_lateral_surface_area_l160_160230

theorem frustum_volume_and_lateral_surface_area (h : ℝ) 
    (A1 A2 : ℝ) (r R : ℝ) (V S_lateral : ℝ) : 
    A1 = 4 * Real.pi → 
    A2 = 25 * Real.pi → 
    h = 4 → 
    r = 2 → 
    R = 5 → 
    V = (1 / 3) * (A1 + A2 + Real.sqrt (A1 * A2)) * h → 
    S_lateral = Real.pi * r * Real.sqrt (h ^ 2 + (R - r) ^ 2) + Real.pi * R * Real.sqrt (h ^ 2 + (R - r) ^ 2) → 
    V = 42 * Real.pi ∧ S_lateral = 35 * Real.pi := by
  sorry

end frustum_volume_and_lateral_surface_area_l160_160230


namespace expr_simplified_l160_160211

theorem expr_simplified : |2 - Real.sqrt 2| - Real.sqrt (1 / 12) * Real.sqrt 27 + Real.sqrt 12 / Real.sqrt 6 = 1 / 2 := 
by 
  sorry

end expr_simplified_l160_160211


namespace polynomial_sum_l160_160398

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l160_160398


namespace g_periodic_6_l160_160679

def g (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a + b, b + c, a + c)

def g_iter (n : Nat) (triple : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match n with
  | 0 => triple
  | n + 1 => g (g_iter n triple).1 (g_iter n triple).2.1 (g_iter n triple).2.2

theorem g_periodic_6 {a b c : ℝ} (h : ∃ n : Nat, n > 0 ∧ g_iter n (a, b, c) = (a, b, c))
  (h' : (a, b, c) ≠ (0, 0, 0)) : g_iter 6 (a, b, c) = (a, b, c) :=
by
  sorry

end g_periodic_6_l160_160679


namespace parabola_line_intersection_l160_160312

theorem parabola_line_intersection :
  let a := (3 + Real.sqrt 11) / 2
  let b := (3 - Real.sqrt 11) / 2
  let p1 := (a, (9 + Real.sqrt 11) / 2)
  let p2 := (b, (9 - Real.sqrt 11) / 2)
  (3 * a^2 - 9 * a + 4 = (9 + Real.sqrt 11) / 2) ∧
  (-a^2 + 3 * a + 6 = (9 + Real.sqrt 11) / 2) ∧
  ((9 + Real.sqrt 11) / 2 = a + 3) ∧
  (3 * b^2 - 9 * b + 4 = (9 - Real.sqrt 11) / 2) ∧
  (-b^2 + 3 * b + 6 = (9 - Real.sqrt 11) / 2) ∧
  ((9 - Real.sqrt 11) / 2 = b + 3) :=
by
  sorry

end parabola_line_intersection_l160_160312


namespace sqrt_ab_equals_sqrt_2_l160_160298

theorem sqrt_ab_equals_sqrt_2 
  (a b : ℝ)
  (h1 : a ^ 2 = 16 / 25)
  (h2 : b ^ 3 = 125 / 8) : 
  Real.sqrt (a * b) = Real.sqrt 2 := 
by 
  -- proof will go here
  sorry

end sqrt_ab_equals_sqrt_2_l160_160298


namespace greatest_difference_47x_l160_160944

def is_multiple_of_4 (n : Nat) : Prop :=
  n % 4 = 0

def valid_digit (d : Nat) : Prop :=
  d < 10

theorem greatest_difference_47x :
  ∃ x y : Nat, (is_multiple_of_4 (470 + x) ∧ valid_digit x) ∧ (is_multiple_of_4 (470 + y) ∧ valid_digit y) ∧ (x < y) ∧ (y - x = 4) :=
sorry

end greatest_difference_47x_l160_160944


namespace initial_contestants_proof_l160_160257

noncomputable def initial_contestants (final_round : ℕ) : ℕ :=
  let fraction_remaining := 2 / 5
  let fraction_advancing := 1 / 2
  let fraction_final := fraction_remaining * fraction_advancing
  (final_round : ℕ) / fraction_final

theorem initial_contestants_proof : initial_contestants 30 = 150 :=
sorry

end initial_contestants_proof_l160_160257


namespace range_of_m_l160_160810

noncomputable def f (a x : ℝ) := a * (x^2 + 1) + Real.log x

theorem range_of_m (a m : ℝ) (h₁ : a ∈ Set.Ioo (-4 : ℝ) (-2))
  (h₂ : ∀ x ∈ Set.Icc (1 : ℝ) (3), ma - f a x > a^2) : m ≤ -2 := 
sorry

end range_of_m_l160_160810


namespace inequality_holds_l160_160501

theorem inequality_holds (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) (h_mul : a * b * c * d = 1) :
  (1 / a) + (1 / b) + (1 / c) + (1 / d) + (12 / (a + b + c + d)) ≥ 7 :=
by
  sorry

end inequality_holds_l160_160501


namespace solve_for_a_minus_b_l160_160117

theorem solve_for_a_minus_b (a b : ℚ) 
  (h1 : 2020 * a + 2024 * b = 2030) 
  (h2 : 2022 * a + 2026 * b = 2032) : 
  a - b = -4 := 
sorry

end solve_for_a_minus_b_l160_160117


namespace sum_of_squares_l160_160987

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 20) (h2 : ab + bc + ca = 5) : a^2 + b^2 + c^2 = 390 :=
by sorry

end sum_of_squares_l160_160987


namespace maximum_sum_of_O_and_square_l160_160998

theorem maximum_sum_of_O_and_square 
(O square : ℕ) (h1 : (O > 0) ∧ (square > 0)) 
(h2 : (O : ℚ) / 11 < (7 : ℚ) / (square))
(h3 : (7 : ℚ) / (square) < (4 : ℚ) / 5) : 
O + square = 18 :=
sorry

end maximum_sum_of_O_and_square_l160_160998


namespace infinite_perfect_squares_of_form_l160_160281

theorem infinite_perfect_squares_of_form (k : ℕ) (hk : 0 < k) : 
  ∃ (n : ℕ), ∀ m : ℕ, ∃ a : ℕ, (n + m) * 2^k - 7 = a^2 :=
sorry

end infinite_perfect_squares_of_form_l160_160281


namespace part_a_part_b_part_c_l160_160404

-- Part (a)
theorem part_a : 
  ∃ n : ℕ, n = 2023066 ∧ (∃ x y z : ℕ, x + y + z = 2013 ∧ x > 0 ∧ y > 0 ∧ z > 0) :=
sorry

-- Part (b)
theorem part_b : 
  ∃ n : ℕ, n = 1006 ∧ (∃ x y z : ℕ, x + y + z = 2013 ∧ x = y ∧ x > 0 ∧ y > 0 ∧ z > 0) :=
sorry

-- Part (c)
theorem part_c : 
  ∃ (x y z : ℕ), (x + y + z = 2013 ∧ (x * y * z = 671 * 671 * 671)) :=
sorry

end part_a_part_b_part_c_l160_160404


namespace express_y_in_terms_of_x_l160_160792

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + 3 * y = 4) : y = (4 - 2 * x) / 3 := 
by
  sorry

end express_y_in_terms_of_x_l160_160792


namespace eq_solutions_a2_eq_b_times_b_plus_7_l160_160216

theorem eq_solutions_a2_eq_b_times_b_plus_7 (a b : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h : a^2 = b * (b + 7)) :
  (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
sorry

end eq_solutions_a2_eq_b_times_b_plus_7_l160_160216


namespace coords_of_A_l160_160124

theorem coords_of_A :
  ∃ (x y : ℝ), y = Real.exp x ∧ (Real.exp x = 1) ∧ y = 1 :=
by
  use 0, 1
  have hx : Real.exp 0 = 1 := Real.exp_zero
  have hy : 1 = Real.exp 0 := hx.symm
  exact ⟨hy, hx, rfl⟩

end coords_of_A_l160_160124


namespace find_center_of_tangent_circle_l160_160625

theorem find_center_of_tangent_circle :
  ∃ (a b : ℝ), (abs a = 5) ∧ (abs b = 5) ∧ (4 * a - 3 * b + 10 = 25) ∧ (a = -5) ∧ (b = 5) :=
by {
  -- Here we would provide the proof in Lean, but for now, we state the theorem
  -- and leave the proof as an exercise.
  sorry
}

end find_center_of_tangent_circle_l160_160625


namespace verify_optionD_is_correct_l160_160738

-- Define the equations as options
def optionA : Prop := -abs (-6) = 6
def optionB : Prop := -(-6) = -6
def optionC : Prop := abs (-6) = -6
def optionD : Prop := -(-6) = 6

-- The proof problem to verify option D is correct
theorem verify_optionD_is_correct : optionD :=
by
  sorry

end verify_optionD_is_correct_l160_160738


namespace MrsYoung_puzzle_complete_l160_160544

theorem MrsYoung_puzzle_complete :
  let total_pieces := 500
  let children := 4
  let pieces_per_child := total_pieces / children
  let minutes := 120
  let pieces_Reyn := (25 * (minutes / 30))
  let pieces_Rhys := 2 * pieces_Reyn
  let pieces_Rory := 3 * pieces_Reyn
  let pieces_Rina := 4 * pieces_Reyn
  let total_pieces_placed := pieces_Reyn + pieces_Rhys + pieces_Rory + pieces_Rina
  total_pieces_placed >= total_pieces :=
by
  sorry

end MrsYoung_puzzle_complete_l160_160544


namespace solve_equation_l160_160743

theorem solve_equation :
  ∀ x : ℝ, 81 * (1 - x) ^ 2 = 64 ↔ x = 1 / 9 ∨ x = 17 / 9 :=
by
  sorry

end solve_equation_l160_160743


namespace probability_of_vowel_initials_l160_160821

/-- In a class with 26 students, each student has unique initials that are double letters
    (i.e., AA, BB, ..., ZZ). If the vowels are A, E, I, O, U, and W, then the probability of
    randomly picking a student whose initials are vowels is 3/13. -/
theorem probability_of_vowel_initials :
  let total_students := 26
  let vowels := ['A', 'E', 'I', 'O', 'U', 'W']
  let num_vowels := 6
  let probability := num_vowels / total_students
  probability = 3 / 13 :=
by
  sorry

end probability_of_vowel_initials_l160_160821


namespace surface_area_of_reassembled_solid_l160_160012

noncomputable def total_surface_area : ℕ :=
let height_E := 1/4
let height_F := 1/6
let height_G := 1/9 
let height_H := 1 - (height_E + height_F + height_G)
let face_area := 2 * 1
(face_area * 2)     -- Top and bottom surfaces
+ 2                -- Side surfaces (1 foot each side * 2 sides)
+ (face_area * 2)   -- Front and back surfaces 

theorem surface_area_of_reassembled_solid :
  total_surface_area = 10 :=
by
  sorry

end surface_area_of_reassembled_solid_l160_160012


namespace fraction_given_to_son_l160_160815

theorem fraction_given_to_son : 
  ∀ (blue_apples yellow_apples total_apples remaining_apples given_apples : ℕ),
    blue_apples = 5 →
    yellow_apples = 2 * blue_apples →
    total_apples = blue_apples + yellow_apples →
    remaining_apples = 12 →
    given_apples = total_apples - remaining_apples →
    (given_apples : ℚ) / total_apples = 1 / 5 :=
by
  intros
  sorry

end fraction_given_to_son_l160_160815


namespace probability_same_color_is_correct_l160_160822

/- Given that there are 5 balls in total, where 3 are white and 2 are black, and two balls are drawn randomly from the bag, we need to prove that the probability of drawing two balls of the same color is 2/5. -/

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def black_balls : ℕ := 2

def total_ways (n r : ℕ) : ℕ := n.choose r
def white_ways : ℕ := total_ways white_balls 2
def black_ways : ℕ := total_ways black_balls 2
def same_color_ways : ℕ := white_ways + black_ways
def total_draws : ℕ := total_ways total_balls 2

def probability_same_color := ((same_color_ways : ℚ) / total_draws)
def expected_probability := (2 : ℚ) / 5

theorem probability_same_color_is_correct :
  probability_same_color = expected_probability :=
by
  sorry

end probability_same_color_is_correct_l160_160822


namespace mean_temperature_is_correct_l160_160417

def temperatures : List ℤ := [-8, -6, -3, -3, 0, 4, -1]
def mean_temperature (temps : List ℤ) : ℚ := (temps.sum : ℚ) / temps.length

theorem mean_temperature_is_correct :
  mean_temperature temperatures = -17 / 7 :=
by
  sorry

end mean_temperature_is_correct_l160_160417


namespace tank_capacity_l160_160755

theorem tank_capacity (w c: ℚ) (h1: w / c = 1 / 3) (h2: (w + 5) / c = 2 / 5) :
  c = 37.5 := 
by
  sorry

end tank_capacity_l160_160755


namespace james_weight_gain_l160_160132

def cheezits_calories (bags : ℕ) (oz_per_bag : ℕ) (cal_per_oz : ℕ) : ℕ :=
  bags * oz_per_bag * cal_per_oz

def chocolate_calories (bars : ℕ) (cal_per_bar : ℕ) : ℕ :=
  bars * cal_per_bar

def popcorn_calories (bags : ℕ) (cal_per_bag : ℕ) : ℕ :=
  bags * cal_per_bag

def run_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def swim_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def cycle_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def total_calories_consumed : ℕ :=
  cheezits_calories 3 2 150 + chocolate_calories 2 250 + popcorn_calories 1 500

def total_calories_burned : ℕ :=
  run_calories 40 12 + swim_calories 30 15 + cycle_calories 20 10

def excess_calories : ℕ :=
  total_calories_consumed - total_calories_burned

def weight_gain (excess_cal : ℕ) (cal_per_lb : ℕ) : ℚ :=
  excess_cal / cal_per_lb

theorem james_weight_gain :
  weight_gain excess_calories 3500 = 770 / 3500 :=
sorry

end james_weight_gain_l160_160132


namespace four_thirds_of_twelve_fifths_l160_160033

theorem four_thirds_of_twelve_fifths : (4 / 3) * (12 / 5) = 16 / 5 := 
by sorry

end four_thirds_of_twelve_fifths_l160_160033


namespace revenue_and_empty_seats_l160_160426

-- Define seating and ticket prices
def seats_A : ℕ := 90
def seats_B : ℕ := 70
def seats_C : ℕ := 50
def VIP_seats : ℕ := 10

def ticket_A : ℕ := 15
def ticket_B : ℕ := 10
def ticket_C : ℕ := 5
def VIP_ticket : ℕ := 25

-- Define discounts
def discount : ℤ := 20

-- Define actual occupancy
def adults_A : ℕ := 35
def children_A : ℕ := 15
def adults_B : ℕ := 20
def seniors_B : ℕ := 5
def adults_C : ℕ := 10
def veterans_C : ℕ := 5
def VIP_occupied : ℕ := 10

-- Concession sales
def hot_dogs_sold : ℕ := 50
def hot_dog_price : ℕ := 4
def soft_drinks_sold : ℕ := 75
def soft_drink_price : ℕ := 2

-- Define the total revenue and empty seats calculation
theorem revenue_and_empty_seats :
  let revenue_from_tickets := (adults_A * ticket_A + children_A * ticket_A * (100 - discount) / 100 +
                               adults_B * ticket_B + seniors_B * ticket_B * (100 - discount) / 100 +
                               adults_C * ticket_C + veterans_C * ticket_C * (100 - discount) / 100 +
                               VIP_occupied * VIP_ticket)
  let revenue_from_concessions := (hot_dogs_sold * hot_dog_price + soft_drinks_sold * soft_drink_price)
  let total_revenue := revenue_from_tickets + revenue_from_concessions
  let empty_seats_A := seats_A - (adults_A + children_A)
  let empty_seats_B := seats_B - (adults_B + seniors_B)
  let empty_seats_C := seats_C - (adults_C + veterans_C)
  let empty_VIP_seats := VIP_seats - VIP_occupied
  total_revenue = 1615 ∧ empty_seats_A = 40 ∧ empty_seats_B = 45 ∧ empty_seats_C = 35 ∧ empty_VIP_seats = 0 := by
  sorry

end revenue_and_empty_seats_l160_160426


namespace find_c_l160_160817

theorem find_c (x : ℝ) (c : ℝ) (h : x = 0.3)
  (equ : (10 * x + 2) / c - (3 * x - 6) / 18 = (2 * x + 4) / 3) :
  c = 4 :=
by
  sorry

end find_c_l160_160817


namespace totalPeoplePresent_is_630_l160_160405

def totalParents : ℕ := 105
def totalPupils : ℕ := 698

def groupA_fraction : ℚ := 30 / 100
def groupB_fraction : ℚ := 25 / 100
def groupC_fraction : ℚ := 20 / 100
def groupD_fraction : ℚ := 15 / 100
def groupE_fraction : ℚ := 10 / 100

def groupA_attendance : ℚ := 90 / 100
def groupB_attendance : ℚ := 80 / 100
def groupC_attendance : ℚ := 70 / 100
def groupD_attendance : ℚ := 60 / 100
def groupE_attendance : ℚ := 50 / 100

def junior_fraction : ℚ := 30 / 100
def intermediate_fraction : ℚ := 35 / 100
def senior_fraction : ℚ := 20 / 100
def advanced_fraction : ℚ := 15 / 100

def junior_attendance : ℚ := 85 / 100
def intermediate_attendance : ℚ := 80 / 100
def senior_attendance : ℚ := 75 / 100
def advanced_attendance : ℚ := 70 / 100

noncomputable def totalPeoplePresent : ℚ := 
  totalParents * groupA_fraction * groupA_attendance +
  totalParents * groupB_fraction * groupB_attendance +
  totalParents * groupC_fraction * groupC_attendance +
  totalParents * groupD_fraction * groupD_attendance +
  totalParents * groupE_fraction * groupE_attendance +
  totalPupils * junior_fraction * junior_attendance +
  totalPupils * intermediate_fraction * intermediate_attendance +
  totalPupils * senior_fraction * senior_attendance +
  totalPupils * advanced_fraction * advanced_attendance

theorem totalPeoplePresent_is_630 : totalPeoplePresent.floor = 630 := 
by 
  sorry -- no proof required as per the instructions

end totalPeoplePresent_is_630_l160_160405


namespace recurring_decimal_to_fraction_l160_160078

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l160_160078


namespace smallest_n_contains_digit9_and_terminating_decimal_l160_160096

-- Define the condition that a number contains the digit 9
def contains_digit_9 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 9

-- Define the condition that a number is of the form 2^a * 5^b
def is_form_of_2a_5b (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2 ^ a * 5 ^ b

-- Define the main theorem
theorem smallest_n_contains_digit9_and_terminating_decimal : 
  ∃ (n : ℕ), contains_digit_9 n ∧ is_form_of_2a_5b n ∧ (∀ m, (contains_digit_9 m ∧ is_form_of_2a_5b m) → n ≤ m) ∧ n = 12500 :=
  sorry

end smallest_n_contains_digit9_and_terminating_decimal_l160_160096


namespace sin_of_right_triangle_l160_160529

open Real

theorem sin_of_right_triangle (Q : ℝ) (h : 3 * sin Q = 4 * cos Q) : sin Q = 4 / 5 :=
by
  sorry

end sin_of_right_triangle_l160_160529


namespace probability_of_a_firing_l160_160931

/-- 
Prove that the probability that A will eventually fire the bullet is 6/11, given the following conditions:
1. A and B take turns shooting with a six-shot revolver that has only one bullet.
2. They randomly spin the cylinder before each shot.
3. A starts the game.
-/
theorem probability_of_a_firing (p_a : ℝ) :
  (1 / 6) + (5 / 6) * (5 / 6) * p_a = p_a → p_a = 6 / 11 :=
by
  intro hyp
  have h : p_a - (25 / 36) * p_a = 1 / 6 := by
    rwa [← sub_eq_of_eq_add hyp, sub_self, zero_mul] 
  field_simp at h
  linarith
  sorry

end probability_of_a_firing_l160_160931


namespace sum_of_numbers_in_row_l160_160845

theorem sum_of_numbers_in_row 
  (n : ℕ)
  (sum_eq : (n * (3 * n - 1)) / 2 = 20112) : 
  n = 1006 :=
sorry

end sum_of_numbers_in_row_l160_160845


namespace find_k_l160_160995

open_locale real_inner_product_space

variables {E : Type*} [inner_product_space ℝ E]

-- Given conditions
variables (e1 e2 : E) (k : ℝ)
hypothesis he1_unit : ∥e1∥ = 1
hypothesis he2_unit : ∥e2∥ = 1
hypothesis he1_e2_angle : real.angle e1 e2 = 2/3 * real.pi
def a := e1 - (2 : ℝ) • e2
def b := k • e1 + e2
hypothesis a_perpendicular_b : inner_product_space.inner a b = 0

-- What we want to prove
theorem find_k : k = 5/4 :=
sorry

end find_k_l160_160995


namespace inverse_negation_l160_160157

theorem inverse_negation :
  (∀ x : ℝ, x ≥ 3 → x < 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ ¬ (x < 3)) :=
by
  sorry

end inverse_negation_l160_160157


namespace two_digit_number_multiple_l160_160446

theorem two_digit_number_multiple (x : ℕ) (h1 : x ≥ 10) (h2 : x < 100) 
(h3 : ∃ k : ℕ, x + 1 = 3 * k) 
(h4 : ∃ k : ℕ, x + 1 = 4 * k) 
(h5 : ∃ k : ℕ, x + 1 = 5 * k) 
(h6 : ∃ k : ℕ, x + 1 = 7 * k) 
: x = 83 := 
sorry

end two_digit_number_multiple_l160_160446


namespace find_f_2017_l160_160360

theorem find_f_2017 {f : ℤ → ℤ}
  (symmetry : ∀ x : ℤ, f (-x) = -f x)
  (periodicity : ∀ x : ℤ, f (x + 4) = f x)
  (f_neg_1 : f (-1) = 2) :
  f 2017 = -2 :=
sorry

end find_f_2017_l160_160360


namespace no_real_roots_for_pair_2_2_3_l160_160667

noncomputable def discriminant (A B : ℝ) : ℝ :=
  let a := 1 - 2 * B
  let b := -B
  let c := -A + A * B
  b ^ 2 - 4 * a * c

theorem no_real_roots_for_pair_2_2_3 : discriminant 2 (2 / 3) < 0 := by
  sorry

end no_real_roots_for_pair_2_2_3_l160_160667


namespace total_savings_l160_160488

def individual_shirt_cost : ℝ := 7.50
def individual_pants_cost : ℝ := 15.00
def individual_socks_cost : ℝ := 4.50

def discounted_shirt_cost : ℝ := 6.75
def discounted_pants_cost : ℝ := 13.50
def discounted_socks_cost : ℝ := 3.75

def team_size : ℕ := 12

theorem total_savings :
  let regular_uniform_cost := individual_shirt_cost + individual_pants_cost + individual_socks_cost in
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost in
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost in
  let total_savings := savings_per_uniform * team_size in
  total_savings = 36 :=
by
  let regular_uniform_cost := individual_shirt_cost + individual_pants_cost + individual_socks_cost
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  have h : total_savings = 36 := by
    calc
      total_savings = (7.50 + 15.00 + 4.50 - (6.75 + 13.50 + 3.75)) * 12 := by sorry
                  ... = 3 * 12 := by sorry
                  ... = 36 := by sorry
  exact h

end total_savings_l160_160488


namespace trig_identity_l160_160522

theorem trig_identity (α : ℝ) (h : Real.sin (Real.pi / 8 + α) = 3 / 4) : 
  Real.cos (3 * Real.pi / 8 - α) = 3 / 4 := 
by 
  sorry

end trig_identity_l160_160522


namespace cone_volume_l160_160566

-- Given conditions as definitions
def slant_height : ℝ := 15
def height : ℝ := 9
def radius : ℝ := Real.sqrt (slant_height^2 - height^2)

-- The volume of the cone
def volume : ℝ := (1/3) * Real.pi * radius^2 * height

-- Theorem to prove that the calculated volume is 432π cubic centimeters
theorem cone_volume : volume = 432 * Real.pi := by
  sorry

end cone_volume_l160_160566


namespace smaller_circle_radius_l160_160729

theorem smaller_circle_radius (r R : ℝ) (A1 A2 : ℝ) (hR : R = 5.0) (hA : A1 + A2 = 25 * Real.pi)
  (hap : A2 = A1 + 25 * Real.pi / 2) : r = 5 * Real.sqrt 2 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end smaller_circle_radius_l160_160729


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160901

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160901


namespace fixed_point_exists_l160_160804

variable {A : ℝ × ℝ}
variable {B : ℝ × ℝ}
variable {F : ℝ × ℝ}
variable {M : ℝ × ℝ}
variable {N : ℝ × ℝ}
variable {Q : ℝ × ℝ}
variable {p : ℝ}
variable {C : ℝ → ℝ}

noncomputable def parabola (p : ℝ) : ℝ → ℝ := fun x => real.sqrt (2 * p * x)

noncomputable def area_triangle (A B F : ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (F.2 - A.2) - (F.1 - A.1) * (B.2 - A.2))

theorem fixed_point_exists
  (hA : A = (-1, 0))
  (hB : B = (1, -1))
  (hF : F = (p / 2, 0))
  (hArea : area_triangle A B F = 1) :
  ∃!(P : ℝ × ℝ), ∀ M N Q, (C := parabola 2) ∧ (C M.1 = M.2) ∧ (C N.1 = N.2) ∧ (C Q.1 = Q.2)
  ∧ ∃ slope : ℝ, ∀ line_slope : ℝ, line_slope ≠ slope ∧ line_slope M.1 ≠ M.2 ∧ Q.1 ≠ N.1 
  ∧ nq := ensure_fixed_point ((Q.2 - N.2) / (Q.1 - N.1))
  hypothesis :
  P = (1, -4) :=
sorry

end fixed_point_exists_l160_160804


namespace quadrilateral_side_length_l160_160330

theorem quadrilateral_side_length (r a b c x : ℝ) (h_radius : r = 100 * Real.sqrt 6) 
    (h_a : a = 100) (h_b : b = 200) (h_c : c = 200) :
    x = 100 * Real.sqrt 2 := 
sorry

end quadrilateral_side_length_l160_160330


namespace rate_of_interest_l160_160943

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem rate_of_interest :
  ∃ R : ℝ, simple_interest 8925 R 5 = 4016.25 ∧ R = 9 := 
by
  use 9
  simp [simple_interest]
  norm_num
  sorry

end rate_of_interest_l160_160943


namespace quadratic_inequality_iff_l160_160978

noncomputable def quadratic_inequality_solution (x : ℝ) : Prop := x^2 + 4*x - 96 > abs x

theorem quadratic_inequality_iff (x : ℝ) : quadratic_inequality_solution x ↔ x < -12 ∨ x > 8 := by
  sorry

end quadratic_inequality_iff_l160_160978


namespace product_of_real_numbers_tripled_when_added_to_reciprocals_l160_160601

theorem product_of_real_numbers_tripled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1 / x = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  (sqrt (1 / 2) * -sqrt (1 / 2)) = -1 / 2 :=
begin
  sorry
end

end product_of_real_numbers_tripled_when_added_to_reciprocals_l160_160601


namespace avg_percentage_students_l160_160739

-- Define the function that calculates the average percentage of all students
def average_percent (n1 n2 : ℕ) (p1 p2 : ℕ) : ℕ :=
  (n1 * p1 + n2 * p2) / (n1 + n2)

-- Define the properties of the numbers of students and their respective percentages
def students_avg : Prop :=
  average_percent 15 10 70 90 = 78

-- The main theorem: Prove that given the conditions, the average percentage is 78%
theorem avg_percentage_students : students_avg :=
  by
    -- The proof will be provided here.
    sorry

end avg_percentage_students_l160_160739


namespace no_such_natural_number_exists_l160_160652

theorem no_such_natural_number_exists :
  ¬ ∃ n : ℕ, ∃ m : ℕ, 3^n + 2 * 17^n = m^2 :=
by sorry

end no_such_natural_number_exists_l160_160652


namespace parallel_vectors_l160_160518

variable (x : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (x, -2)

theorem parallel_vectors (h : (1 * (-2) - 2 * x = 0)) : x = -1 :=
by
  sorry

end parallel_vectors_l160_160518


namespace min_b_for_quadratic_factorization_l160_160098

theorem min_b_for_quadratic_factorization : ∃ b : ℕ, b = 84 ∧ ∃ p q : ℤ, p + q = b ∧ p * q = 1760 :=
by
  sorry

end min_b_for_quadratic_factorization_l160_160098


namespace max_pens_given_budget_l160_160942

-- Define the conditions.
def max_pens (x y : ℕ) := 12 * x + 20 * y

-- Define the main theorem stating the proof problem.
theorem max_pens_given_budget : ∃ (x y : ℕ), (10 * x + 15 * y ≤ 173) ∧ (max_pens x y = 224) :=
  sorry

end max_pens_given_budget_l160_160942


namespace recurring_decimal_to_fraction_l160_160077

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l160_160077


namespace least_number_subtracted_l160_160736

-- Define the original number and the divisor
def original_number : ℕ := 427398
def divisor : ℕ := 14

-- Define the least number to be subtracted
def remainder := original_number % divisor
def least_number := remainder

-- The statement to be proven
theorem least_number_subtracted : least_number = 6 :=
by
  sorry

end least_number_subtracted_l160_160736


namespace total_fish_l160_160700

theorem total_fish (fish_lilly fish_rosy : ℕ) (hl : fish_lilly = 10) (hr : fish_rosy = 14) :
  fish_lilly + fish_rosy = 24 := 
by 
  sorry

end total_fish_l160_160700


namespace johnson_and_martinez_tied_at_may_l160_160626

def home_runs_johnson (m : String) : ℕ :=
  if m = "January" then 2 else
  if m = "February" then 12 else
  if m = "March" then 20 else
  if m = "April" then 15 else
  if m = "May" then 9 else 0

def home_runs_martinez (m : String) : ℕ :=
  if m = "January" then 5 else
  if m = "February" then 9 else
  if m = "March" then 15 else
  if m = "April" then 20 else
  if m = "May" then 9 else 0

def cumulative_home_runs (player_home_runs : String → ℕ) (months : List String) : ℕ :=
  months.foldl (λ acc m => acc + player_home_runs m) 0

def months_up_to_may : List String :=
  ["January", "February", "March", "April", "May"]

theorem johnson_and_martinez_tied_at_may :
  cumulative_home_runs home_runs_johnson months_up_to_may
  = cumulative_home_runs home_runs_martinez months_up_to_may :=
by
    sorry

end johnson_and_martinez_tied_at_may_l160_160626


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160907

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160907


namespace total_boys_went_down_slide_l160_160472

theorem total_boys_went_down_slide :
  let boys_first_10_minutes := 22
  let boys_next_5_minutes := 13
  let boys_last_20_minutes := 35
  (boys_first_10_minutes + boys_next_5_minutes + boys_last_20_minutes) = 70 :=
by
  sorry

end total_boys_went_down_slide_l160_160472


namespace no_real_roots_range_l160_160819

theorem no_real_roots_range (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - 2*k + 3 ≠ 0) → k < 1 :=
by
  sorry

end no_real_roots_range_l160_160819


namespace repeating_decimal_eq_fraction_l160_160053

theorem repeating_decimal_eq_fraction : (∑' n : ℕ, 56 / 100^(n+1)) = (56 / 99) := 
by
  sorry

end repeating_decimal_eq_fraction_l160_160053


namespace B_pow_150_eq_I_l160_160385

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_150_eq_I : B^(150 : ℕ) = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by {
  sorry
}

end B_pow_150_eq_I_l160_160385


namespace no_five_consecutive_integers_with_fourth_powers_sum_l160_160351

theorem no_five_consecutive_integers_with_fourth_powers_sum:
  ∀ n : ℤ, n^4 + (n + 1)^4 + (n + 2)^4 + (n + 3)^4 ≠ (n + 4)^4 :=
by
  intros
  sorry

end no_five_consecutive_integers_with_fourth_powers_sum_l160_160351


namespace exists_linear_eq_exactly_m_solutions_l160_160706

theorem exists_linear_eq_exactly_m_solutions (m : ℕ) (hm : 0 < m) :
  ∃ (a b c : ℤ), ∀ (x y : ℕ), a * x + b * y = c ↔
    (1 ≤ x ∧ 1 ≤ y ∧ x + y = m + 1) :=
by
  sorry

end exists_linear_eq_exactly_m_solutions_l160_160706


namespace volleyball_team_probability_l160_160185

open Finset

def total_players : ℕ := 9
def chosen_players : ℕ := 6
def john_peter_count : ℕ := 2
def remaining_players : ℕ := total_players - john_peter_count
def remaining_chosen : ℕ := chosen_players - john_peter_count

theorem volleyball_team_probability : 
  (nat.choose remaining_players remaining_chosen) / (nat.choose total_players chosen_players) = 5 / 12 := 
by
  sorry

end volleyball_team_probability_l160_160185


namespace units_digit_of_3_pow_4_l160_160616

theorem units_digit_of_3_pow_4 : (3^4 % 10) = 1 :=
by
  sorry

end units_digit_of_3_pow_4_l160_160616


namespace problem1_problem2_l160_160111

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then -2 * x - 1
  else if 0 < x ∧ x ≤ 1 then -2 * x + 1
  else 0 -- considering the function is not defined outside the given range

-- Statement to prove that f(f(-1)) = -1
theorem problem1 : f (f (-1)) = -1 :=
by
  sorry

-- Statements to prove the solution set for |f(x)| < 1/2
theorem problem2 : { x : ℝ | |f x| < 1 / 2 } = { x : ℝ | -3/4 < x ∧ x < -1/4 } ∪ { x : ℝ | 1/4 < x ∧ x < 3/4 } :=
by
  sorry

end problem1_problem2_l160_160111


namespace interest_difference_l160_160452

def principal : ℝ := 3600
def rate : ℝ := 0.25
def time : ℕ := 2

def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

theorem interest_difference :
  let SI := simple_interest principal rate time;
  let CI := compound_interest principal rate time;
  CI - SI = 225 :=
by
  sorry

end interest_difference_l160_160452


namespace matrix_power_B150_l160_160387

open Matrix

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

-- Prove that B^150 = I
theorem matrix_power_B150 : 
  (B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ)) :=
by
  sorry

end matrix_power_B150_l160_160387


namespace find_number_divided_by_3_equals_subtracted_5_l160_160895

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l160_160895


namespace product_of_real_numbers_triple_when_added_to_their_reciprocal_l160_160603

theorem product_of_real_numbers_triple_when_added_to_their_reciprocal :
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = (1 / Real.sqrt 2) ∨ x = - (1 / Real.sqrt 2)) →
  (∀ x1 x2 : ℝ, (x1 = (1 / Real.sqrt 2) ∧ x2 = - (1 / Real.sqrt 2)) → x1 * x2 = - (1 / 2)) :=
by
  intros h1 h2
  sorry

end product_of_real_numbers_triple_when_added_to_their_reciprocal_l160_160603


namespace greatest_value_x_l160_160343

theorem greatest_value_x (x: ℤ) : 
  (∃ k: ℤ, (x^2 - 5 * x + 14) = k * (x - 4)) → x ≤ 14 :=
sorry

end greatest_value_x_l160_160343


namespace slope_of_line_l160_160219

theorem slope_of_line (x y : ℝ) (h : 4 * y = 5 * x + 20) : y = (5/4) * x + 5 :=
by {
  sorry
}

end slope_of_line_l160_160219


namespace janet_total_miles_run_l160_160835

/-- Janet was practicing for a marathon. She practiced for 9 days, running 8 miles each day.
Prove that Janet ran 72 miles in total. -/
theorem janet_total_miles_run (days_practiced : ℕ) (miles_per_day : ℕ) (total_miles : ℕ) 
  (h1 : days_practiced = 9) (h2 : miles_per_day = 8) : total_miles = 72 := by
  sorry

end janet_total_miles_run_l160_160835


namespace sum_factors_of_30_l160_160612

def factors (n : ℕ) : List ℕ :=
  (List.range (n+1)).filter (λ m, m > 0 ∧ n % m == 0)

theorem sum_factors_of_30 : (factors 30).sum = 72 := by
  sorry

end sum_factors_of_30_l160_160612


namespace total_cupcakes_baked_l160_160551

theorem total_cupcakes_baked
    (boxes : ℕ)
    (cupcakes_per_box : ℕ)
    (left_at_home : ℕ)
    (total_given_away : ℕ)
    (total_baked : ℕ)
    (h1 : boxes = 17)
    (h2 : cupcakes_per_box = 3)
    (h3 : left_at_home = 2)
    (h4 : total_given_away = boxes * cupcakes_per_box)
    (h5 : total_baked = total_given_away + left_at_home) :
    total_baked = 53 := by
  sorry

end total_cupcakes_baked_l160_160551


namespace average_age_of_town_l160_160684

-- Definitions based on conditions
def ratio_of_women_to_men (nw nm : ℕ) : Prop := nw * 8 = nm * 9

def young_men (nm : ℕ) (n_young_men : ℕ) (average_age_young : ℕ) : Prop :=
  n_young_men = 40 ∧ average_age_young = 25

def remaining_men_average_age (nm n_young_men : ℕ) (average_age_remaining : ℕ) : Prop :=
  average_age_remaining = 35

def women_average_age (average_age_women : ℕ) : Prop :=
  average_age_women = 30

-- Complete problem statement we need to prove
theorem average_age_of_town (nw nm : ℕ) (total_avg_age : ℕ) :
  ratio_of_women_to_men nw nm →
  young_men nm 40 25 →
  remaining_men_average_age nm 40 35 →
  women_average_age 30 →
  total_avg_age = 32 * 17 + 6 :=
sorry

end average_age_of_town_l160_160684


namespace a_is_constant_l160_160319

variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, 0 < a n)
variable (h_ineq : ∀ n, a n ≥ (a (n+2) + a (n+1) + a (n-1) + a (n-2)) / 4)

theorem a_is_constant : ∀ n m, a n = a m :=
by
  sorry

end a_is_constant_l160_160319


namespace solve_fractional_equation_l160_160970

theorem solve_fractional_equation (x : ℝ) :
  (x ≠ 5) ∧ (x ≠ 6) ∧ ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)) / ((x - 5) * (x - 6) * (x - 5)) = 1 ↔ 
  x = 2 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
sorry

end solve_fractional_equation_l160_160970


namespace solve_equation_l160_160525

theorem solve_equation (x : ℝ) (h : 3 + 1 / (2 - x) = 2 * (1 / (2 - x))) : x = 5 / 3 := 
  sorry

end solve_equation_l160_160525


namespace normalized_sequence_embedding_l160_160495

/-- Definitions for sequences and embeddings --/
def non_negative_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i, 0 ≤ a i

def interval_embedding (a : ℕ → ℝ) (x : ℕ → ℝ) (b c : ℝ) (k : ℕ) : Prop :=
  ∀ i < k, |x i - x (i - 1)| = a i ∧ x i ≤ c ∧ x i ≥ b

/-- For any given non-negative integer n, prove:(1) Any normalized sequence of length 2n+1 
can be embeded in the interval [0, 2- 1/(2^n)]. 
(2) There exists a normalized sequence of length 4n+3 that cannot be embeded in the interval [0, 2- 1/(2^n)]. --/
theorem normalized_sequence_embedding (n : ℕ) :
  (∀ a : ℕ → ℝ, (∀ i < 2 * n + 1, 0 ≤ a i ∧ a i ≤ 1) →
    ∃ x : ℕ → ℝ, interval_embedding a x 0 (2 - (1 / (2 ^ n))) (2 * n + 1)) ∧
  (∃ a : ℕ → ℝ, (∀ i < 4 * n + 3, 0 ≤ a i ∧ a i ≤ 1) ∧
    ¬ ∃ x : ℕ → ℝ, interval_embedding a x 0 (2 - (1 / (2 ^ n))) (4 * n + 3)) := sorry

end normalized_sequence_embedding_l160_160495


namespace part1_part2_l160_160389

-- Definitions for the sets A and B
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a - 1) * x + (a^2 - 5) = 0}

-- Proof problem (1): A ∩ B = {2} implies a = -5 or a = 1
theorem part1 (a : ℝ) (h : A ∩ B a = {2}) : a = -5 ∨ a = 1 := 
sorry

-- Proof problem (2): A ∪ B = A implies a > 3
theorem part2 (a : ℝ) (h : A ∪ B a = A) : 3 < a :=
sorry

end part1_part2_l160_160389


namespace evaluate_expression_l160_160656

theorem evaluate_expression : (24^36 / 72^18) = 8^18 := by
  sorry

end evaluate_expression_l160_160656


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160900

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160900


namespace sum_of_possible_values_l160_160687

theorem sum_of_possible_values :
  ∀ x, (|x - 5| - 4 = 3) → x = 12 ∨ x = -2 → (12 + (-2) = 10) :=
by
  sorry

end sum_of_possible_values_l160_160687


namespace find_expression_value_l160_160983

theorem find_expression_value (x y : ℚ) (h₁ : 3 * x + y = 6) (h₂ : x + 3 * y = 8) :
  9 * x ^ 2 + 15 * x * y + 9 * y ^ 2 = 1629 / 16 := 
sorry

end find_expression_value_l160_160983


namespace parallel_lines_slope_l160_160304

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → a = 3 / 2 :=
by
  sorry

end parallel_lines_slope_l160_160304


namespace cost_of_cookies_equal_3_l160_160534

def selling_price : ℝ := 1.5
def cost_price : ℝ := 1
def number_of_bracelets : ℕ := 12
def amount_left : ℝ := 3

theorem cost_of_cookies_equal_3 : 
  (selling_price - cost_price) * number_of_bracelets - amount_left = 3 := by
  sorry

end cost_of_cookies_equal_3_l160_160534


namespace smallest_number_divide_perfect_cube_l160_160880

theorem smallest_number_divide_perfect_cube (n : ℕ):
  n = 450 → (∃ m : ℕ, n * m = k ∧ ∃ k : ℕ, k ^ 3 = n * m) ∧ (∀ m₂ : ℕ, (n * m₂ = l ∧ ∃ l : ℕ, l ^ 3 = n * m₂) → m ≤ m₂) → m = 60 :=
by
  sorry

end smallest_number_divide_perfect_cube_l160_160880


namespace solution_of_system_l160_160445

theorem solution_of_system :
  ∃ x y z : ℚ,
    x + 2 * y = 12 ∧
    y + 3 * z = 15 ∧
    3 * x - z = 6 ∧
    x = 54 / 17 ∧
    y = 75 / 17 ∧
    z = 60 / 17 :=
by
  exists 54 / 17, 75 / 17, 60 / 17
  repeat { sorry }

end solution_of_system_l160_160445


namespace probability_no_more_than_once_probability_two_months_l160_160948

noncomputable def P : ℕ → ℝ
| 0 => 0.3
| 1 => 0.5
| 2 => 0.2
| _ => 0.0

def no_more_than_once :=
  P 0 + P 1 = 0.8

def independent_events (P1 P2 : ℕ → ℝ) :=
  ∀ (i j : ℕ), P1 i * P2 j = (P1 i) * (P2 j)

def P_months (n1 n2 : ℕ) : ℝ :=
  match n1, n2 with
  | 0, 2 => P 0 * P 2
  | 2, 0 => P 2 * P 0
  | 1, 1 => P 1 * P 1
  | _, _ => 0.0

def two_months :=
  P_months 0 2 + P_months 2 0 + P_months 1 1 = 0.37

axiom January_February_independent: independent_events P P

theorem probability_no_more_than_once : no_more_than_once := by
  sorry

theorem probability_two_months : two_months := by
  have h_independent := January_February_independent
  sorry

end probability_no_more_than_once_probability_two_months_l160_160948


namespace jack_jill_meet_distance_l160_160832

theorem jack_jill_meet_distance : 
  ∀ (total_distance : ℝ) (uphill_distance : ℝ) (headstart : ℝ) 
  (jack_speed_up : ℝ) (jack_speed_down : ℝ)
  (jill_speed_up : ℝ) (jill_speed_down : ℝ), 
  total_distance = 12 → 
  uphill_distance = 6 → 
  headstart = 1 / 4 → 
  jack_speed_up = 12 → 
  jack_speed_down = 18 → 
  jill_speed_up = 14 → 
  jill_speed_down = 20 → 
  ∃ meet_position : ℝ, meet_position = 15.75 :=
by
  sorry

end jack_jill_meet_distance_l160_160832


namespace fraction_eq_repeating_decimal_l160_160043

theorem fraction_eq_repeating_decimal :
  let x := 0.56 -- considering the 0.56565656... as a repetitive infinite decimal
  in (x = 56 / 99) :=
by
  sorry

end fraction_eq_repeating_decimal_l160_160043


namespace amelia_wins_probability_l160_160212

theorem amelia_wins_probability :
  let pA := 1 / 4
  let pB := 1 / 3
  let pC := 1 / 2
  let cycle_probability := (1 - pA) * (1 - pB) * (1 - pC)
  let infinite_series_sum := 1 / (1 - cycle_probability)
  let total_probability := pA * infinite_series_sum
  total_probability = 1 / 3 :=
by
  sorry

end amelia_wins_probability_l160_160212


namespace product_of_solutions_l160_160590

theorem product_of_solutions:
  ∃ x1 x2 : ℝ, (x1 + 1/x1 = 3 * x1) ∧ (x2 + 1/x2 = 3 * x2) ∧
  (x1 * x2 = -1/2) :=
begin
  sorry
end

end product_of_solutions_l160_160590


namespace coefficient_of_x4_l160_160252

theorem coefficient_of_x4 (a : ℝ) (h : 15 * a^4 = 240) : a = 2 ∨ a = -2 := 
sorry

end coefficient_of_x4_l160_160252


namespace fraction_of_smaller_jar_l160_160959

theorem fraction_of_smaller_jar (S L : ℝ) (W : ℝ) (F : ℝ) 
  (h1 : W = F * S) 
  (h2 : W = 1/2 * L) 
  (h3 : 2 * W = 2/3 * L) 
  (h4 : S = 2/3 * L) :
  F = 3 / 4 :=
by
  sorry

end fraction_of_smaller_jar_l160_160959


namespace real_solution_exists_l160_160968

theorem real_solution_exists (x : ℝ) (h1: x ≠ 5) (h2: x ≠ 6) : 
  (x = 1 ∨ x = 2 ∨ x = 3) ↔ 
  ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1) /
  ((x - 5) * (x - 6) * (x - 5)) = 1) := 
by 
  sorry

end real_solution_exists_l160_160968


namespace min_major_axis_l160_160238

theorem min_major_axis (a b c : ℝ) (h1 : b * c = 1) (h2 : a = Real.sqrt (b^2 + c^2)) : 2 * a ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_major_axis_l160_160238


namespace smallest_b_for_factorization_l160_160350

theorem smallest_b_for_factorization :
  ∃ (b : ℕ), (∀ r s : ℕ, (r * s = 3258) → (b = r + s)) ∧ (∀ c : ℕ, (∀ r' s' : ℕ, (r' * s' = 3258) → (c = r' + s')) → b ≤ c) :=
sorry

end smallest_b_for_factorization_l160_160350


namespace interior_edges_sum_l160_160634

theorem interior_edges_sum 
  (frame_thickness : ℝ)
  (frame_area : ℝ)
  (outer_length : ℝ)
  (frame_thickness_eq : frame_thickness = 2)
  (frame_area_eq : frame_area = 32)
  (outer_length_eq : outer_length = 7) 
  : ∃ interior_edges_sum : ℝ, interior_edges_sum = 8 := 
by
  sorry

end interior_edges_sum_l160_160634


namespace find_interval_solution_l160_160785

def interval_solution : Set ℝ := {x | 2 < x / (3 * x - 7) ∧ x / (3 * x - 7) <= 7}

theorem find_interval_solution (x : ℝ) :
  x ∈ interval_solution ↔
  x ∈ Set.Ioc (49 / 20 : ℝ) (14 / 5 : ℝ) := 
sorry

end find_interval_solution_l160_160785


namespace trains_cross_time_l160_160623

noncomputable def time_to_cross_trains : ℝ :=
  let l1 := 220 -- length of the first train in meters
  let s1 := 120 * (5 / 18) -- speed of the first train in meters per second
  let l2 := 280.04 -- length of the second train in meters
  let s2 := 80 * (5 / 18) -- speed of the second train in meters per second
  let relative_speed := s1 + s2 -- relative speed in meters per second
  let total_length := l1 + l2 -- total length to be crossed in meters
  total_length / relative_speed -- time in seconds

theorem trains_cross_time :
  abs (time_to_cross_trains - 9) < 0.01 := -- Allowing a small error to account for approximation
by
  sorry

end trains_cross_time_l160_160623


namespace ellipse_and_line_equation_l160_160503

theorem ellipse_and_line_equation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (ab_ratio : a / b = 2)
  (focus : ℝ × ℝ := (real.sqrt 3, 0)) (focus_pos : 0 < real.sqrt 3) (focus_eq : focus = (real.sqrt 3, 0))
  (line_pass_a : ℝ × ℝ := (1, 0)) (point_b : ℝ × ℝ := (0, 1))
  ⦃c : ℝ⦄ (c_value : c = real.sqrt 3) :
  (∃ (a b : ℝ), (a / b = 2 ∧ a^2 = b^2 + (real.sqrt 3)^2 ∧
    (x y : ℝ) (h : (x - 0)^2 / a^2 + y^2 / b^2 = 1 → false)), 
   (∃ (m : ℝ), (line_pass_a = (1, 0) ∧ point_b ∈ set.univ ∧ 
    (line_eq : linear_map ℝ × ℝ := (λ (y x), x = m * y + 1),
   line_eq = (λ (y x), x + y - 1 = 0 ∨ 3 * x - 5 * y - 3 = 0)))) :=
sorry

end ellipse_and_line_equation_l160_160503


namespace rhombus_area_correct_l160_160857

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_correct :
  rhombus_area 80 120 = 4800 :=
by 
  -- the proof is skipped by including sorry
  sorry

end rhombus_area_correct_l160_160857


namespace coordinates_of_P_l160_160251

open Real

theorem coordinates_of_P (P : ℝ × ℝ) (h1 : P.1 = 2 * cos (2 * π / 3)) (h2 : P.2 = 2 * sin (2 * π / 3)) :
  P = (-1, sqrt 3) :=
by
  sorry

end coordinates_of_P_l160_160251


namespace problem_solution_l160_160161

noncomputable def proof_problem : Prop :=
∀ x y : ℝ, y = (x + 1)^2 ∧ (x * y^2 + y = 1) → false

theorem problem_solution : proof_problem :=
by
  sorry

end problem_solution_l160_160161


namespace min_b_for_quadratic_factorization_l160_160097

theorem min_b_for_quadratic_factorization : ∃ b : ℕ, b = 84 ∧ ∃ p q : ℤ, p + q = b ∧ p * q = 1760 :=
by
  sorry

end min_b_for_quadratic_factorization_l160_160097


namespace find_seq_formula_l160_160376

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, 0 < n → 
  (∑ i in finset.range n, (a (i + 1) / (i + 1)^2)) = a n

theorem find_seq_formula (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, 0 < n → a n = (2 * n) / (n + 1) :=
by
  sorry

end find_seq_formula_l160_160376


namespace spilled_wax_amount_l160_160377

-- Definitions based on conditions
def car_wax := 3
def suv_wax := 4
def total_wax := 11
def remaining_wax := 2

-- The theorem to be proved
theorem spilled_wax_amount : car_wax + suv_wax + (total_wax - remaining_wax - (car_wax + suv_wax)) = total_wax - remaining_wax :=
by
  sorry


end spilled_wax_amount_l160_160377


namespace total_brushing_time_in_hours_l160_160285

-- Define the conditions as Lean definitions
def brushing_duration : ℕ := 2   -- 2 minutes per brushing session
def brushing_times_per_day : ℕ := 3  -- brushes 3 times a day
def days : ℕ := 30  -- for 30 days

-- Define the calculation of total brushing time in hours
theorem total_brushing_time_in_hours : (brushing_duration * brushing_times_per_day * days) / 60 = 3 := 
by 
  -- Sorry to skip the proof
  sorry

end total_brushing_time_in_hours_l160_160285


namespace count_valid_three_digit_numbers_l160_160366

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, n = 720 ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 → 
    (∀ d ∈ [m / 100, (m / 10) % 10, m % 10], d ∉ [2, 5, 7, 9])) := 
sorry

end count_valid_three_digit_numbers_l160_160366


namespace probability_A_shoots_l160_160930

theorem probability_A_shoots (P : ℚ) :
  (∀ n : ℕ, (2 * n + 1) % 2 = 1) →  -- A's turn is always the odd turn
  (∀ m : ℕ, (2 * m) % 2 = 0) →  -- B's turn is always the even turn
  let p_A_first_shot := (1 : ℚ) / 6 in  -- probability A fires on the first shot
  let p_A_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun
  let p_B_turn := (5 : ℚ) / 6 in  -- probability of not firing the gun for B
  let P_A := p_A_first_shot + (p_A_turn * p_B_turn * P) in  -- recursive definition
  P_A = 6 / 11 := -- final probability
sorry

end probability_A_shoots_l160_160930


namespace fraction_pairs_l160_160791

theorem fraction_pairs (n : ℕ) (h : n > 2009) : 
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 1 ≤ a ∧ a ≤ n ∧
  1 ≤ b ∧ b ≤ n ∧ 1 ≤ c ∧ c ≤ n ∧ 1 ≤ d ∧ d ≤ n ∧
  1/a + 1/b = 1/c + 1/d := 
sorry

end fraction_pairs_l160_160791


namespace inequality_solution_l160_160150

noncomputable def solve_inequality : Set ℝ :=
  {x | (x - 5) / ((x - 3)^2) < 0}

theorem inequality_solution :
  solve_inequality = {x | x < 3} ∪ {x | 3 < x ∧ x < 5} :=
by
  sorry

end inequality_solution_l160_160150


namespace children_neither_happy_nor_sad_l160_160144

-- conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10

-- proof problem
theorem children_neither_happy_nor_sad :
  total_children - happy_children - sad_children = 20 := by
  sorry

end children_neither_happy_nor_sad_l160_160144


namespace fraction_eq_repeating_decimal_l160_160067

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l160_160067


namespace parallelogram_area_l160_160239

theorem parallelogram_area (s : ℝ) (ratio : ℝ) (A : ℝ) :
  s = 3 → ratio = 2 * Real.sqrt 2 → A = 9 → 
  (A * ratio = 18 * Real.sqrt 2) :=
by
  sorry

end parallelogram_area_l160_160239


namespace fraction_eq_repeating_decimal_l160_160068

noncomputable def repeating_decimal_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem fraction_eq_repeating_decimal :
  let x := 0.56 + 0.0056 + 0.000056 + (Real.toRat (DenotationConvert.denom (DenotationConvert. torational 0.56)))
  x = 0 + x * (1/100):
  repeating_decimal_sum (56 / 100) (1 / 100) = (56 / 99) :=
by 
  -- proof goes here
  sorry

end fraction_eq_repeating_decimal_l160_160068


namespace recurring_decimal_to_fraction_l160_160082

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l160_160082


namespace integer_solutions_exist_l160_160989

theorem integer_solutions_exist (k : ℤ) :
  (∃ x : ℤ, 9 * x - 3 = k * x + 14) ↔ (k = 8 ∨ k = 10 ∨ k = -8 ∨ k = 26) :=
by
  sorry

end integer_solutions_exist_l160_160989


namespace pow_three_not_sum_of_two_squares_l160_160292

theorem pow_three_not_sum_of_two_squares (k : ℕ) (hk : 0 < k) : 
  ¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^2 + y^2 = 3^k :=
by
  sorry

end pow_three_not_sum_of_two_squares_l160_160292


namespace average_weight_l160_160299

def weights (A B C : ℝ) : Prop :=
  (A + B + C = 135) ∧
  (B + C = 86) ∧
  (B = 31)

theorem average_weight (A B C : ℝ) (h : weights A B C) :
  (A + B) / 2 = 40 :=
by
  sorry

end average_weight_l160_160299


namespace crates_lost_l160_160199

theorem crates_lost (total_crates : ℕ) (total_cost : ℕ) (desired_profit_percent : ℕ) 
(lost_crates remaining_crates : ℕ) (price_per_crate : ℕ) 
(h1 : total_crates = 10) (h2 : total_cost = 160) (h3 : desired_profit_percent = 25) 
(h4 : price_per_crate = 25) (h5 : remaining_crates = total_crates - lost_crates)
(h6 : price_per_crate * remaining_crates = total_cost + total_cost * desired_profit_percent / 100) :
  lost_crates = 2 :=
by
  sorry

end crates_lost_l160_160199


namespace unpainted_cubes_count_l160_160715

theorem unpainted_cubes_count :
  let L := 6
  let W := 6
  let H := 3
  (L - 2) * (W - 2) * (H - 2) = 16 :=
by
  sorry

end unpainted_cubes_count_l160_160715


namespace additional_tanks_needed_l160_160381

theorem additional_tanks_needed 
    (initial_tanks : ℕ) 
    (initial_capacity_per_tank : ℕ) 
    (total_fish_needed : ℕ) 
    (new_capacity_per_tank : ℕ)
    (h_t1 : initial_tanks = 3)
    (h_t2 : initial_capacity_per_tank = 15)
    (h_t3 : total_fish_needed = 75)
    (h_t4 : new_capacity_per_tank = 10) : 
    (total_fish_needed - initial_tanks * initial_capacity_per_tank) / new_capacity_per_tank = 3 := 
by {
    sorry
}

end additional_tanks_needed_l160_160381


namespace fraction_of_repeating_decimal_l160_160071

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l160_160071


namespace grid_permutation_exists_l160_160208

theorem grid_permutation_exists (n : ℕ) (grid : Fin n → Fin n → ℤ) 
  (cond1 : ∀ i : Fin n, ∃ unique j : Fin n, grid i j = 1)
  (cond2 : ∀ i : Fin n, ∃ unique j : Fin n, grid i j = -1)
  (cond3 : ∀ j : Fin n, ∃ unique i : Fin n, grid i j = 1)
  (cond4 : ∀ j : Fin n, ∃ unique i : Fin n, grid i j = -1)
  (cond5 : ∀ i j, grid i j = 0 ∨ grid i j = 1 ∨ grid i j = -1) :
  ∃ (perm_rows perm_cols : Fin n → Fin n),
    (∀ i j, grid (perm_rows i) (perm_cols j) = -grid i j) :=
by
  -- Proof goes here
  sorry

end grid_permutation_exists_l160_160208


namespace walking_times_relationship_l160_160618

theorem walking_times_relationship (x : ℝ) (h : x > 0) :
  (15 / x) - (15 / (x + 1)) = 1 / 2 :=
sorry

end walking_times_relationship_l160_160618


namespace garden_width_min_5_l160_160383

theorem garden_width_min_5 (width length : ℝ) (h_length : length = width + 20) (h_area : width * length ≥ 150) :
  width ≥ 5 :=
sorry

end garden_width_min_5_l160_160383


namespace division_remainder_l160_160184

theorem division_remainder : 
  ∀ (Dividend Divisor Quotient Remainder : ℕ), 
  Dividend = 760 → 
  Divisor = 36 → 
  Quotient = 21 → 
  Dividend = (Divisor * Quotient) + Remainder → 
  Remainder = 4 := 
by 
  intros Dividend Divisor Quotient Remainder h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  have h5 : 760 = 36 * 21 + Remainder := h4
  linarith

end division_remainder_l160_160184


namespace factorize_x_cube_minus_9x_l160_160781

-- Lean statement: Prove that x^3 - 9x = x(x+3)(x-3)
theorem factorize_x_cube_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
by
  sorry

end factorize_x_cube_minus_9x_l160_160781


namespace y_not_directly_nor_inversely_proportional_l160_160269

theorem y_not_directly_nor_inversely_proportional (x y : ℝ) :
  (∃ k : ℝ, x + y = 0 ∧ y = k * x) ∨
  (∃ k : ℝ, 3 * x * y = 10 ∧ x * y = k) ∨
  (∃ k : ℝ, x = 5 * y ∧ x = k * y) ∨
  (∃ k : ℝ, (y = 10 - x^2 - 3 * x) ∧ y ≠ k * x ∧ y * x ≠ k) ∨
  (∃ k : ℝ, x / y = Real.sqrt 3 ∧ x = k * y)
  → (∃ k : ℝ, y = 10 - x^2 - 3 * x ∧ y ≠ k * x ∧ y * x ≠ k) :=
by
  sorry

end y_not_directly_nor_inversely_proportional_l160_160269


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l160_160599

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l160_160599


namespace zoo_animals_left_l160_160430

noncomputable def totalAnimalsLeft (x : ℕ) : ℕ := 
  let initialFoxes := 2 * x
  let initialRabbits := 3 * x
  let foxesAfterMove := initialFoxes - 10
  let rabbitsAfterMove := initialRabbits / 2
  foxesAfterMove + rabbitsAfterMove

theorem zoo_animals_left (x : ℕ) (h : 20 * x - 100 = 39 * x / 2) : totalAnimalsLeft x = 690 := by
  sorry

end zoo_animals_left_l160_160430


namespace characterize_functions_l160_160347

open Function

noncomputable def f : ℚ → ℚ := sorry
noncomputable def g : ℚ → ℚ := sorry

axiom f_g_condition_1 : ∀ x y : ℚ, f (g (x) - g (y)) = f (g (x)) - y
axiom f_g_condition_2 : ∀ x y : ℚ, g (f (x) - f (y)) = g (f (x)) - y

theorem characterize_functions : 
  (∃ c : ℚ, ∀ x, f x = c * x) ∧ (∃ c : ℚ, ∀ x, g x = x / c) := 
sorry

end characterize_functions_l160_160347


namespace necessary_but_not_sufficient_condition_l160_160996

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
noncomputable def vector_b : ℝ × ℝ := (2, 1)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Statement: Prove x > 0 is a necessary but not sufficient condition for the angle between vectors a and b to be acute.
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (dot_product (vector_a x) vector_b > 0) ↔ (x > 0) := 
sorry

end necessary_but_not_sufficient_condition_l160_160996


namespace repeating_decimal_is_fraction_l160_160060

noncomputable def infinite_geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem repeating_decimal_is_fraction :
  let a := (56 : ℚ) / 100;
  let r := (1 : ℚ) / 100;
  infinite_geometric_series_sum a r = (56 : ℚ) / 99 :=
by
  sorry

end repeating_decimal_is_fraction_l160_160060


namespace initial_ratio_of_milk_to_water_l160_160125

variable (M W : ℕ)
noncomputable def M_initial := 45 - W
noncomputable def W_new := W + 9

theorem initial_ratio_of_milk_to_water :
  M_initial = 36 ∧ W = 9 →
  M_initial / (W + 9) = 2 ↔ 4 = M_initial / W := 
sorry

end initial_ratio_of_milk_to_water_l160_160125


namespace number_of_thrown_out_carrots_l160_160021

-- Definitions from the conditions
def initial_carrots : ℕ := 48
def picked_next_day : ℕ := 42
def total_carrots : ℕ := 45

-- Proposition stating the problem
theorem number_of_thrown_out_carrots (x : ℕ) : initial_carrots - x + picked_next_day = total_carrots → x = 45 :=
by
  sorry

end number_of_thrown_out_carrots_l160_160021


namespace team_leader_and_deputy_choice_l160_160571

def TeamLeaderSelection : Type := {x : Fin 5 // true}
def DeputyLeaderSelection (TL : TeamLeaderSelection) : Type := {x : Fin 5 // x ≠ TL.val}

theorem team_leader_and_deputy_choice : 
  (Σ TL : TeamLeaderSelection, DeputyLeaderSelection TL) → Fin 20 :=
by sorry

end team_leader_and_deputy_choice_l160_160571


namespace max_temp_range_l160_160450

theorem max_temp_range (avg_temp : ℝ) (lowest_temp : ℝ) (days : ℕ) (total_temp : ℝ) (range : ℝ) : 
  avg_temp = 45 → 
  lowest_temp = 42 → 
  days = 5 → 
  total_temp = avg_temp * days → 
  range = 6 := 
by 
  sorry

end max_temp_range_l160_160450


namespace man_l160_160457

theorem man's_salary 
  (food_fraction : ℚ := 1/5) 
  (rent_fraction : ℚ := 1/10) 
  (clothes_fraction : ℚ := 3/5) 
  (remaining_money : ℚ := 15000) 
  (S : ℚ) :
  (S * (1 - (food_fraction + rent_fraction + clothes_fraction)) = remaining_money) →
  S = 150000 := 
by
  intros h1
  sorry

end man_l160_160457


namespace minimum_n_value_l160_160263

theorem minimum_n_value : ∃ n : ℕ, n > 0 ∧ ∀ r : ℕ, (2 * n = 5 * r) → n = 5 :=
by
  sorry

end minimum_n_value_l160_160263


namespace root_difference_geom_prog_l160_160650

theorem root_difference_geom_prog
  (x1 x2 x3 : ℝ)
  (h1 : 8 * x1^3 - 22 * x1^2 + 15 * x1 - 2 = 0)
  (h2 : 8 * x2^3 - 22 * x2^2 + 15 * x2 - 2 = 0)
  (h3 : 8 * x3^3 - 22 * x3^2 + 15 * x3 - 2 = 0)
  (geom_prog : ∃ (a r : ℝ), x1 = a / r ∧ x2 = a ∧ x3 = a * r) :
  |x3 - x1| = 33 / 14 :=
by
  sorry

end root_difference_geom_prog_l160_160650


namespace bills_needed_can_pay_groceries_l160_160143

theorem bills_needed_can_pay_groceries 
  (cans_of_soup : ℕ := 6) (price_per_can : ℕ := 2)
  (loaves_of_bread : ℕ := 3) (price_per_loaf : ℕ := 5)
  (boxes_of_cereal : ℕ := 4) (price_per_box : ℕ := 3)
  (gallons_of_milk : ℕ := 2) (price_per_gallon : ℕ := 4)
  (apples : ℕ := 7) (price_per_apple : ℕ := 1)
  (bags_of_cookies : ℕ := 5) (price_per_bag : ℕ := 3)
  (bottles_of_olive_oil : ℕ := 1) (price_per_bottle : ℕ := 8)
  : ∃ (bills_needed : ℕ), bills_needed = 4 :=
by
  let total_cost := (cans_of_soup * price_per_can) + 
                    (loaves_of_bread * price_per_loaf) +
                    (boxes_of_cereal * price_per_box) +
                    (gallons_of_milk * price_per_gallon) +
                    (apples * price_per_apple) +
                    (bags_of_cookies * price_per_bag) +
                    (bottles_of_olive_oil * price_per_bottle)
  let bills_needed := (total_cost + 19) / 20   -- Calculating ceiling of total_cost / 20
  sorry

end bills_needed_can_pay_groceries_l160_160143


namespace parametric_eq_C1_rectangular_eq_C2_max_dist_PQ_l160_160375

-- Curve C1 given by x^2 / 9 + y^2 = 1, prove its parametric form
theorem parametric_eq_C1 (α : ℝ) : 
  (∃ (x y : ℝ), x = 3 * Real.cos α ∧ y = Real.sin α ∧ (x ^ 2 / 9 + y ^ 2 = 1)) := 
sorry

-- Curve C2 given by ρ^2 - 8ρ sin θ + 15 = 0, prove its rectangular form
theorem rectangular_eq_C2 (ρ θ : ℝ) : 
  (∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ 
    (ρ ^ 2 - 8 * ρ * Real.sin θ + 15 = 0) ↔ (x ^ 2 + y ^ 2 - 8 * y + 15 = 0)) := 
sorry

-- Prove the maximum value of |PQ|
theorem max_dist_PQ : 
  (∃ (P Q : ℝ × ℝ), 
    (P = (3 * Real.cos α, Real.sin α)) ∧ 
    (Q = (0, 4)) ∧ 
    (∀ α : ℝ, Real.sqrt ((3 * Real.cos α) ^ 2 + (Real.sin α - 4) ^ 2) ≤ 8)) := 
sorry

end parametric_eq_C1_rectangular_eq_C2_max_dist_PQ_l160_160375


namespace increase_circumference_l160_160023

theorem increase_circumference (d1 d2 : ℝ) (increase : ℝ) (P : ℝ) : 
  increase = 2 * Real.pi → 
  P = Real.pi * increase → 
  P = 2 * Real.pi ^ 2 := 
by 
  intros h_increase h_P
  rw [h_P, h_increase]
  sorry

end increase_circumference_l160_160023


namespace maria_towels_l160_160448

theorem maria_towels (green_towels white_towels given_towels : ℕ) (bought_green : green_towels = 40) 
(bought_white : white_towels = 44) (gave_mother : given_towels = 65) : 
  green_towels + white_towels - given_towels = 19 := by
sorry

end maria_towels_l160_160448


namespace find_number_divided_by_3_equals_subtracted_5_l160_160898

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l160_160898


namespace ratio_kid_to_adult_ticket_l160_160708

theorem ratio_kid_to_adult_ticket (A : ℝ) : 
  (6 * 5 + 2 * A = 50) → (5 / A = 1 / 2) :=
by
  sorry

end ratio_kid_to_adult_ticket_l160_160708


namespace number_of_correct_conclusions_l160_160666

noncomputable def A (x : ℝ) : ℝ := 2 * x^2
noncomputable def B (x : ℝ) : ℝ := x + 1
noncomputable def C (x : ℝ) : ℝ := -2 * x
noncomputable def D (y : ℝ) : ℝ := y^2
noncomputable def E (x y : ℝ) : ℝ := 2 * x - y

def conclusion1 (y : ℤ) : Prop := 
  0 < ((B (0 : ℝ)) * (C (0 : ℝ)) + A (0 : ℝ) + D y + E (0) (y : ℝ))

def conclusion2 : Prop := 
  ∃ (x y : ℝ), A x + D y + 2 * E x y = -2

def M (A B C : ℝ → ℝ) (x m : ℝ) : ℝ :=
  3 * (A x - B x) + m * B x * C x

def linear_term_exists (m : ℝ) : Prop :=
  (0 : ℝ) ≠ -3 - 2 * m

def conclusion3 : Prop := 
 ∀ m : ℝ, (¬ linear_term_exists m ∧ M A B C (0 : ℝ) m > -3) 

def p (x y : ℝ) := 
  2 * (x + 1) ^ 2 + (y - 1) ^ 2 = 1

theorem number_of_correct_conclusions : Prop := 
  (¬ conclusion1 1) ∧ (conclusion2) ∧ (¬ conclusion3)

end number_of_correct_conclusions_l160_160666


namespace floor_smallest_positive_root_of_g_eq_two_l160_160839

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - Real.cos x + 4 * (1 / Real.cos x)

theorem floor_smallest_positive_root_of_g_eq_two :
  let s := Inf {x : ℝ | 0 < x ∧ g x = 0} in
  ⌊s⌋ = 2 := 
begin
  sorry
end

end floor_smallest_positive_root_of_g_eq_two_l160_160839


namespace product_of_real_numbers_tripled_by_addition_to_reciprocal_l160_160592

theorem product_of_real_numbers_tripled_by_addition_to_reciprocal :
  (∀ x : ℝ, x + x⁻¹ = 3 * x → x = (1 / (real.sqrt 2)) ∨ x = -(1 / (real.sqrt 2))) →
    (1 / (real.sqrt 2)) * -(1 / (real.sqrt 2)) = -1 / 2 :=
by
  sorry

end product_of_real_numbers_tripled_by_addition_to_reciprocal_l160_160592


namespace general_term_formula_of_sequence_l160_160511

theorem general_term_formula_of_sequence {a : ℕ → ℝ} (S : ℕ → ℝ)
  (hS : ∀ n, S n = (2 / 3) * a n + 1 / 3) :
  (∀ n, a n = (-2) ^ (n - 1)) :=
by
  sorry

end general_term_formula_of_sequence_l160_160511


namespace kathleen_allowance_l160_160837

theorem kathleen_allowance (x : ℝ) (h1 : Kathleen_middleschool_allowance = x + 2)
(h2 : Kathleen_senior_allowance = 5 + 2 * (x + 2))
(h3 : Kathleen_senior_allowance = 2.5 * Kathleen_middleschool_allowance) :
x = 8 :=
by sorry

end kathleen_allowance_l160_160837


namespace patrons_per_golf_cart_l160_160489

theorem patrons_per_golf_cart (patrons_from_cars patrons_from_bus golf_carts total_patrons patrons_per_cart : ℕ) 
  (h1 : patrons_from_cars = 12)
  (h2 : patrons_from_bus = 27)
  (h3 : golf_carts = 13)
  (h4 : total_patrons = patrons_from_cars + patrons_from_bus)
  (h5 : patrons_per_cart = total_patrons / golf_carts) : 
  patrons_per_cart = 3 := 
by
  sorry

end patrons_per_golf_cart_l160_160489


namespace work_done_together_l160_160744

theorem work_done_together
    (fraction_work_left : ℚ)
    (A_days : ℕ)
    (B_days : ℚ) :
    A_days = 20 →
    fraction_work_left = 2 / 3 →
    4 * (1 / 20 + 1 / B_days) = 1 / 3 →
    B_days = 30 := 
by
  intros hA hfrac heq
  sorry

end work_done_together_l160_160744


namespace min_expression_value_l160_160541

theorem min_expression_value (x y z : ℝ) (xyz_eq : x * y * z = 1) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n : ℝ, (∀ x y z : ℝ, x * y * z = 1 → 0 < x → 0 < y → 0 < z → 2 * x^2 + 8 * x * y + 32 * y^2 + 16 * y * z + 8 * z^2 ≥ n)
    ∧ n = 72 :=
sorry

end min_expression_value_l160_160541


namespace B_pow_150_eq_I_l160_160384

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_150_eq_I : B^(150 : ℕ) = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by {
  sorry
}

end B_pow_150_eq_I_l160_160384


namespace dvds_rented_l160_160400

def total_cost : ℝ := 4.80
def cost_per_dvd : ℝ := 1.20

theorem dvds_rented : total_cost / cost_per_dvd = 4 := 
by
  sorry

end dvds_rented_l160_160400


namespace other_factor_of_936_mul_w_l160_160749

theorem other_factor_of_936_mul_w (w : ℕ) (h_w_pos : 0 < w)
  (h_factors_936w : ∃ k, 936 * w = k * (3^3)) 
  (h_factors_936w_2 : ∃ m, 936 * w = m * (10^2))
  (h_w : w = 120):
  ∃ n, n = 45 :=
by
  sorry

end other_factor_of_936_mul_w_l160_160749


namespace obtuse_triangle_has_exactly_one_obtuse_angle_l160_160997

-- Definition of an obtuse triangle
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90)

-- Definition of an obtuse angle
def is_obtuse_angle (angle : ℝ) : Prop :=
  angle > 90

-- The theorem statement
theorem obtuse_triangle_has_exactly_one_obtuse_angle {A B C : ℝ} 
  (h1 : is_obtuse_triangle A B C) : 
  (is_obtuse_angle A ∨ is_obtuse_angle B ∨ is_obtuse_angle C) ∧ 
  ¬(is_obtuse_angle A ∧ is_obtuse_angle B) ∧ 
  ¬(is_obtuse_angle A ∧ is_obtuse_angle C) ∧ 
  ¬(is_obtuse_angle B ∧ is_obtuse_angle C) :=
sorry

end obtuse_triangle_has_exactly_one_obtuse_angle_l160_160997


namespace minimum_d_value_l160_160201

theorem minimum_d_value :
  let d := (1 + 2 * Real.sqrt 10) / 3
  let distance := Real.sqrt ((2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2)
  distance = 4 * d :=
by
  let d := (1 + 2 * Real.sqrt 10) / 3
  let distance := Real.sqrt ((2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2)
  sorry

end minimum_d_value_l160_160201


namespace maximum_sum_set_l160_160390

def no_two_disjoint_subsets_have_equal_sums (S : Finset ℕ) : Prop :=
  ∀ (A B : Finset ℕ), A ≠ B ∧ A ∩ B = ∅ → (A.sum id) ≠ (B.sum id)

theorem maximum_sum_set (S : Finset ℕ) (h : ∀ x ∈ S, x ≤ 15) (h_subset_sum : no_two_disjoint_subsets_have_equal_sums S) : S.sum id = 61 :=
sorry

end maximum_sum_set_l160_160390


namespace sum_ratio_l160_160669

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {d : ℕ}

axiom arithmetic_sequence : ∀ n, a_n n = a_n 1 + (n - 1) * d
axiom sum_of_first_n_terms : ∀ n, S_n n = n * (a_n 1 + a_n n) / 2
axiom condition_a4 : a_n 4 = 2 * (a_n 2 + a_n 3)
axiom non_zero_difference : d ≠ 0

theorem sum_ratio : S_n 7 / S_n 4 = 7 / 4 := 
by
  sorry

end sum_ratio_l160_160669


namespace repeating_decimal_equiv_fraction_l160_160090

theorem repeating_decimal_equiv_fraction :
  (∀ S, S = (0.56 + 0.000056 + 0.00000056 + ... ) → S = 56 / 99) :=
begin
  sorry
end

end repeating_decimal_equiv_fraction_l160_160090


namespace probability_both_in_picture_l160_160410

/-- Rachel completes a lap every 90 seconds, Robert completes a lap every 80 seconds,
and both start running from the same line at the same time. A picture is taken at a
random time between 600 and 660 seconds, showing one-fourth of the track centered
on the starting line. Prove that the probability of both Rachel and Robert being in
the picture is 3/16. -/
theorem probability_both_in_picture :
  let lap_rachel := 90
  let lap_robert := 80
  let t_start := 600
  let t_end := 660
  let picture_cover := 1/4
  let time_range := t_end - t_start
  let prob := (11.25 / 60 : ℚ)
  prob = 3 / 16 := sorry

end probability_both_in_picture_l160_160410


namespace sum_of_percentages_l160_160439

theorem sum_of_percentages :
  let percent1 := 7.35 / 100
  let percent2 := 13.6 / 100
  let percent3 := 21.29 / 100
  let num1 := 12658
  let num2 := 18472
  let num3 := 29345
  let result := percent1 * num1 + percent2 * num2 + percent3 * num3
  result = 9689.9355 :=
by
  sorry

end sum_of_percentages_l160_160439


namespace find_a3_l160_160502

theorem find_a3 (a : ℝ) (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) 
    (h1 : (1 + x) * (a - x)^6 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7)
    (h2 : a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 = 0) :
  a = 1 → a3 = -5 := 
by 
  sorry

end find_a3_l160_160502


namespace math_club_competition_scores_l160_160757

open List

theorem math_club_competition_scores:
  let scores := [92, 89, 96, 94, 98, 96, 95] in
  let sorted_scores := [89, 92, 94, 95, 96, 96, 98] in
  let mode := 96 in
  let median := 95 in
  mode_of_list scores = Some mode ∧ median_of_list scores = some median := by
  sorry

end math_club_competition_scores_l160_160757


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160906

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160906


namespace eq_root_count_l160_160663

theorem eq_root_count (p : ℝ) : 
  (∀ x : ℝ, (2 * x^2 - 3 * p * x + 2 * p = 0 → (9 * p^2 - 16 * p = 0))) →
  (∃! p1 p2 : ℝ, (9 * p1^2 - 16 * p1 = 0) ∧ (9 * p2^2 - 16 * p2 = 0) ∧ p1 ≠ p2) :=
sorry

end eq_root_count_l160_160663


namespace a_and_b_together_finish_in_40_days_l160_160920

theorem a_and_b_together_finish_in_40_days (D : ℕ) 
    (W : ℕ)
    (day_with_b : ℕ)
    (remaining_days_a : ℕ)
    (a_alone_days : ℕ)
    (a_b_together : D = 40)
    (ha : (remaining_days_a = 15) ∧ (a_alone_days = 20) ∧ (day_with_b = 10))
    (work_done_total : 10 * (W / D) + 15 * (W / a_alone_days) = W) :
    D = 40 := 
    sorry

end a_and_b_together_finish_in_40_days_l160_160920


namespace minimum_value_l160_160355

open Real

variables {A B C M : Type}
variables (AB AC : ℝ) 
variables (S_MBC x y : ℝ)

-- Assume the given conditions
axiom dot_product_AB_AC : AB * AC = 2 * sqrt 3
axiom angle_BAC_30 : (30 : Real) = π / 6
axiom area_MBC : S_MBC = 1/2
axiom area_sum : x + y = 1/2

-- Define the minimum value problem
theorem minimum_value : 
  ∃ m, m = 18 ∧ (∀ x y, (1/x + 4/y) ≥ m) :=
sorry

end minimum_value_l160_160355


namespace cookie_combinations_l160_160624

theorem cookie_combinations (total_cookies kinds : Nat) (at_least_one : kinds > 0 ∧ ∀ k : Nat, k < kinds → k > 0) : 
  (total_cookies = 8 ∧ kinds = 4) → 
  (∃ comb : Nat, comb = 34) := 
by 
  -- insert proof here 
  sorry

end cookie_combinations_l160_160624


namespace grouping_equal_products_l160_160025

def group1 : List Nat := [12, 42, 95, 143]
def group2 : List Nat := [30, 44, 57, 91]

def product (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem grouping_equal_products :
  product group1 = product group2 := by
  sorry

end grouping_equal_products_l160_160025


namespace cat_count_l160_160703

def initial_cats : ℕ := 2
def female_kittens : ℕ := 3
def male_kittens : ℕ := 2
def total_kittens : ℕ := female_kittens + male_kittens
def total_cats : ℕ := initial_cats + total_kittens

theorem cat_count : total_cats = 7 := by
  unfold total_cats
  unfold initial_cats total_kittens
  unfold female_kittens male_kittens
  rfl

end cat_count_l160_160703


namespace roses_in_february_l160_160422

-- Define initial counts of roses
def roses_oct : ℕ := 80
def roses_nov : ℕ := 98
def roses_dec : ℕ := 128
def roses_jan : ℕ := 170

-- Define the differences
def diff_on : ℕ := roses_nov - roses_oct -- 18
def diff_nd : ℕ := roses_dec - roses_nov -- 30
def diff_dj : ℕ := roses_jan - roses_dec -- 42

-- The increment in differences
def inc : ℕ := diff_nd - diff_on -- 12

-- Express the difference from January to February
def diff_jf : ℕ := diff_dj + inc -- 54

-- The number of roses in February
def roses_feb : ℕ := roses_jan + diff_jf -- 224

theorem roses_in_february : roses_feb = 224 := by
  -- Provide the expected value for Lean to verify
  sorry

end roses_in_february_l160_160422


namespace no_square_number_divisible_by_six_between_50_and_120_l160_160965

theorem no_square_number_divisible_by_six_between_50_and_120 :
  ¬ ∃ x : ℕ, (∃ n : ℕ, x = n * n) ∧ (x % 6 = 0) ∧ (50 < x ∧ x < 120) := 
sorry

end no_square_number_divisible_by_six_between_50_and_120_l160_160965


namespace angle_measure_l160_160733

theorem angle_measure (x : ℝ) (h1 : 180 - x = 6 * (90 - x)) : x = 72 := by
  sorry

end angle_measure_l160_160733


namespace dice_sum_probability_l160_160878

theorem dice_sum_probability :
  let total_outcomes := 36
  let sum_le_8_outcomes := 13
  (sum_le_8_outcomes : ℕ) / (total_outcomes : ℕ) = (13 / 18 : ℝ) :=
by
  sorry

end dice_sum_probability_l160_160878


namespace quadratic_inequality_solution_l160_160178

theorem quadratic_inequality_solution (m: ℝ) (h: m > 1) :
  { x : ℝ | x^2 + (m - 1) * x - m ≥ 0 } = { x | x ≤ -m ∨ x ≥ 1 } :=
sorry

end quadratic_inequality_solution_l160_160178


namespace least_number_to_subtract_l160_160924

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (k : ℕ) (hk : 42398 % 15 = k) : k = 8 :=
by
  sorry

end least_number_to_subtract_l160_160924


namespace bisecting_chord_line_eqn_l160_160788

theorem bisecting_chord_line_eqn :
  ∀ (x1 y1 x2 y2 : ℝ),
  y1 ^ 2 = 16 * x1 →
  y2 ^ 2 = 16 * x2 →
  (x1 + x2) / 2 = 2 →
  (y1 + y2) / 2 = 1 →
  ∃ (a b c : ℝ), a = 8 ∧ b = -1 ∧ c = -15 ∧
  ∀ (x y : ℝ), y = 8 * x - 15 → a * x + b * y + c = 0 :=
by 
  sorry

end bisecting_chord_line_eqn_l160_160788


namespace simplify_expression_l160_160295

variable (y : ℝ)

theorem simplify_expression : 
    4 * y + 8 * y^2 + 6 - (3 - 4 * y - 8 * y^2) = 16 * y^2 + 8 * y + 3 :=
begin
  sorry
end

end simplify_expression_l160_160295


namespace find_number_eq_seven_point_five_l160_160909

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l160_160909


namespace smallest_z_value_l160_160374

theorem smallest_z_value :
  ∃ (x z : ℕ), (w = x - 2) ∧ (y = x + 2) ∧ (z = x + 4) ∧ ((x - 2)^3 + x^3 + (x + 2)^3 = (x + 4)^3) ∧ z = 2 := by
  sorry

end smallest_z_value_l160_160374


namespace tan_u_tan_v_sum_l160_160540

theorem tan_u_tan_v_sum (u v : ℝ) 
  (h1 : (sin u / cos v) + (sin v / cos u) = 2)
  (h2 : (cos u / sin v) + (cos v / sin u) = 3) :
  (tan u / tan v) + (tan v / tan u) = 8 / 7 :=
by
  sorry

end tan_u_tan_v_sum_l160_160540


namespace range_of_a_l160_160371

theorem range_of_a (a : ℝ) : 
  (∀ P Q : ℝ × ℝ, P ≠ Q ∧ P.snd = a * P.fst ^ 2 - 1 ∧ Q.snd = a * Q.fst ^ 2 - 1 ∧ 
  P.fst + P.snd = -(Q.fst + Q.snd)) →
  a > 3 / 4 :=
by
  sorry

end range_of_a_l160_160371


namespace equivalent_solution_l160_160302

theorem equivalent_solution (c x : ℤ) 
    (h1 : 3 * x + 9 = 6)
    (h2 : c * x - 15 = -5)
    (hx : x = -1) :
    c = -10 :=
sorry

end equivalent_solution_l160_160302


namespace product_of_solutions_l160_160582

theorem product_of_solutions : 
  let solutions := {x : ℝ | x + 1/x = 3 * x}
  ( ∀ x ∈ solutions, x = 1/√2 ∨ x = -1/√2 ) →
  ∏ (hx : x ∈ solutions), x = -1/2 :=
sorry

end product_of_solutions_l160_160582


namespace perpendicular_line_theorem_l160_160440

-- Mathematical definitions used in the condition.
def Line := Type
def Plane := Type

variables {l m : Line} {π : Plane}

-- Given the predicate that a line is perpendicular to another line on the plane
def is_perpendicular (l m : Line) (π : Plane) : Prop :=
sorry -- Definition of perpendicularity in Lean (abstracted here)

-- Given condition: l is perpendicular to the projection of m on plane π
axiom projection_of_oblique (m : Line) (π : Plane) : Line

-- The Perpendicular Line Theorem
theorem perpendicular_line_theorem (h : is_perpendicular l (projection_of_oblique m π) π) : is_perpendicular l m π :=
sorry

end perpendicular_line_theorem_l160_160440


namespace factorize_x_cubed_minus_9x_l160_160776

theorem factorize_x_cubed_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

end factorize_x_cubed_minus_9x_l160_160776


namespace smallest_ratio_is_three_l160_160013

theorem smallest_ratio_is_three (m n : ℕ) (a : ℕ) (h1 : 2^m + 1 = a * (2^n + 1)) (h2 : a > 1) : a = 3 :=
sorry

end smallest_ratio_is_three_l160_160013


namespace jellybean_count_l160_160957

theorem jellybean_count (x : ℕ) (h : (0.7 : ℝ) ^ 3 * x = 34) : x = 99 :=
sorry

end jellybean_count_l160_160957


namespace common_ratio_of_geometric_sequence_l160_160509

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ)
  (h_geom : ∃ q, ∀ n, a (n+1) = a n * q)
  (h1 : a 1 = 1 / 8)
  (h4 : a 4 = -1) :
  ∃ q, q = -2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l160_160509


namespace sum_factors_of_30_l160_160614

theorem sum_factors_of_30 : 
  (∑ i in (finset.filter (λ x, 30 % x = 0) (finset.range 31)), i) = 72 := by
  sorry

end sum_factors_of_30_l160_160614


namespace mixture_kerosene_l160_160289

theorem mixture_kerosene (x : ℝ) (h₁ : 0.25 * x + 1.2 = 0.27 * (x + 4)) : x = 6 :=
sorry

end mixture_kerosene_l160_160289


namespace find_BP_l160_160620

theorem find_BP
  (A B C D P : Type) 
  (AP PC BP DP : ℝ)
  (hAP : AP = 8) 
  (hPC : PC = 1)
  (hBD : BD = 6)
  (hBP_less_DP : BP < DP) 
  (hPower_of_Point : AP * PC = BP * DP)
  : BP = 2 := 
by {
  sorry
}

end find_BP_l160_160620


namespace product_of_reals_tripled_when_added_to_reciprocal_l160_160585

theorem product_of_reals_tripled_when_added_to_reciprocal :
  (∏ x in {x : ℝ | x + 1/x = 3 * x}.finite_to_set, x) = -1/2 :=
by
  sorry

end product_of_reals_tripled_when_added_to_reciprocal_l160_160585


namespace find_a_plus_b_l160_160364

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)

def parallel_condition (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.2 - a.2 * b.1 = 0)

theorem find_a_plus_b (m : ℝ) (h_parallel: 
  parallel_condition (⟨vector_a.1 + 2 * (vector_b m).1, vector_a.2 + 2 * (vector_b m).2⟩)
                     (⟨2 * vector_a.1 - (vector_b m).1, 2 * vector_a.2 - (vector_b m).2⟩)) :
  vector_a + vector_b (-1/2) = (-3/2, 3) := 
by
  sorry

end find_a_plus_b_l160_160364


namespace shaded_area_square_semicircles_l160_160852

theorem shaded_area_square_semicircles :
  let side_length := 2
  let radius_circle := side_length * Real.sqrt 2 / 2
  let area_circle := Real.pi * radius_circle^2
  let area_square := side_length^2
  let area_semicircle := Real.pi * (side_length / 2)^2 / 2
  let total_area_semicircles := 4 * area_semicircle
  let shaded_area := total_area_semicircles - area_circle
  shaded_area = 4 :=
by
  sorry

end shaded_area_square_semicircles_l160_160852


namespace calculate_x_l160_160976

theorem calculate_x :
  let a := 3
  let b := 5
  let c := 2
  let d := 4
  let term1 := (a ^ 2) * b * 0.47 * 1442
  let term2 := c * d * 0.36 * 1412
  (term1 - term2) + 63 = 26544.74 := by
  sorry

end calculate_x_l160_160976


namespace coefficient_a2_in_expansion_l160_160980

theorem coefficient_a2_in_expansion:
  let a := (x - 1)^4
  let expansion := a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4
  a2 = 6 :=
by
  sorry

end coefficient_a2_in_expansion_l160_160980


namespace total_players_l160_160622

-- Definitions based on problem conditions.
def players_kabadi : Nat := 10
def players_kho_kho_only : Nat := 20
def players_both_games : Nat := 5

-- Proof statement for the total number of players.
theorem total_players : (players_kabadi + players_kho_kho_only - players_both_games) = 25 := by
  sorry

end total_players_l160_160622


namespace trail_length_l160_160311

theorem trail_length (v_Q : ℝ) (v_P : ℝ) (d_P d_Q : ℝ) 
  (h_vP: v_P = 1.25 * v_Q) 
  (h_dP: d_P = 20) 
  (h_meet: d_P / v_P = d_Q / v_Q) :
  d_P + d_Q = 36 :=
sorry

end trail_length_l160_160311


namespace digit_1C3_multiple_of_3_l160_160352

theorem digit_1C3_multiple_of_3 :
  (∃ C : Fin 10, (1 + C.val + 3) % 3 = 0) ∧
  (∀ C : Fin 10, (1 + C.val + 3) % 3 = 0 → (C.val = 2 ∨ C.val = 5 ∨ C.val = 8)) :=
by
  sorry

end digit_1C3_multiple_of_3_l160_160352


namespace Tara_loss_point_l160_160709

theorem Tara_loss_point :
  ∀ (clarinet_cost initial_savings book_price total_books_sold additional_books books_sold_to_goal) 
  (H1 : initial_savings = 10)
  (H2 : clarinet_cost = 90)
  (H3 : book_price = 5)
  (H4 : total_books_sold = 25)
  (H5 : books_sold_to_goal = (clarinet_cost - initial_savings) / book_price)
  (H6 : additional_books = total_books_sold - books_sold_to_goal),
  additional_books * book_price = 45 :=
by
  intros clarinet_cost initial_savings book_price total_books_sold additional_books books_sold_to_goal
  intros H1 H2 H3 H4 H5 H6
  sorry

end Tara_loss_point_l160_160709


namespace part_a_l160_160921

-- Lean 4 statement equivalent to Part (a)
theorem part_a (n : ℕ) (x : ℝ) (hn : 0 < n) (hx : n^2 ≤ x) : 
  n * Real.sqrt (x - n^2) ≤ x / 2 := 
sorry

-- Lean 4 statement equivalent to Part (b)
noncomputable def find_xyz : ℕ × ℕ × ℕ :=
  ((2, 8, 18) : ℕ × ℕ × ℕ)

end part_a_l160_160921


namespace tournament_key_player_l160_160945

theorem tournament_key_player (n : ℕ) (plays : Fin n → Fin n → Bool) (wins : ∀ i j, plays i j → ¬plays j i) :
  ∃ X, ∀ (Y : Fin n), Y ≠ X → (plays X Y ∨ ∃ Z, plays X Z ∧ plays Z Y) :=
by
  sorry

end tournament_key_player_l160_160945


namespace fraction_of_repeating_decimal_l160_160074

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end fraction_of_repeating_decimal_l160_160074


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160905

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160905


namespace trig_identity_solution_l160_160794

theorem trig_identity_solution
  (α : ℝ) (β : ℝ)
  (h1 : Real.tan α = 1 / 2)
  (h2 : Real.tan β = -1 / 3) :
  (3 * Real.sin α * Real.cos β - Real.sin β * Real.cos α) / (Real.cos α * Real.cos β + 2 * Real.sin α * Real.sin β) = 11 / 4 :=
by
  sorry

end trig_identity_solution_l160_160794


namespace merchant_boxes_fulfill_order_l160_160325

theorem merchant_boxes_fulfill_order :
  ∃ (a b c d e : ℕ), 16 * a + 17 * b + 23 * c + 39 * d + 40 * e = 100 := sorry

end merchant_boxes_fulfill_order_l160_160325


namespace geometric_sequence_S5_l160_160498

noncomputable def S5 (a₁ q : ℝ) : ℝ :=
  a₁ * (1 - q^5) / (1 - q)

theorem geometric_sequence_S5 
  (a₁ q : ℝ) 
  (h₁ : a₁ * (1 + q) = 3 / 4)
  (h₄ : a₁ * q^3 * (1 + q) = 6) :
  S5 a₁ q = 31 / 4 := 
sorry

end geometric_sequence_S5_l160_160498


namespace final_amount_after_5_years_l160_160402

-- Define conditions as hypotheses
def principal := 200
def final_amount_after_2_years := 260
def time_2_years := 2

-- Define our final question and answer as a Lean theorem
theorem final_amount_after_5_years : 
  (final_amount_after_2_years - principal) = principal * (rate * time_2_years) →
  (rate * 3) = 90 →
  final_amount_after_2_years + (principal * rate * 3) = 350 :=
by
  intros h1 h2
  -- Proof skipped using sorry
  sorry

end final_amount_after_5_years_l160_160402


namespace quadratic_roots_identity_l160_160790

noncomputable def a := - (2 / 5 : ℝ)
noncomputable def b := (1 / 5 : ℝ)
noncomputable def quadraticRoots := (a, b)

theorem quadratic_roots_identity :
  a + b ^ 2 = - (9 / 25 : ℝ) := 
by 
  rw [a, b]
  sorry

end quadratic_roots_identity_l160_160790


namespace product_of_solutions_l160_160598

theorem product_of_solutions : 
  (∀ x : ℝ, x + 1/x = 3 * x → x = sqrt 2 / 2 ∨ x = -sqrt 2 / 2) →
  (\big (a b : ℝ), a = sqrt 2 / 2 ∧ b = -sqrt 2 / 2 → a * b = -1/2)
  :=
sorry

end product_of_solutions_l160_160598


namespace max_initial_segment_length_l160_160428

theorem max_initial_segment_length (sequence1 : ℕ → ℕ) (sequence2 : ℕ → ℕ)
  (period1 : ℕ) (period2 : ℕ)
  (h1 : ∀ n, sequence1 (n + period1) = sequence1 n)
  (h2 : ∀ n, sequence2 (n + period2) = sequence2 n)
  (p1 : period1 = 7) (p2 : period2 = 13) :
  ∃ max_length : ℕ, max_length = 18 :=
sorry

end max_initial_segment_length_l160_160428


namespace find_number_l160_160888

theorem find_number (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_l160_160888


namespace mode_is_48_median_is_48_range_is_38_quantile_5_percent_is_26_l160_160003

def data_set : List ℕ := [63, 38, 25, 42, 56, 48, 53, 39, 28, 47, 45, 52, 59, 48, 41, 62, 48, 50, 52, 27]

-- Prove the mode is 48
theorem mode_is_48 : (data_set.filter (fun x => x = 48)).length > (data_set.filter (fun x => x = 42)).length :=
by sorry

-- Prove the median is 48
theorem median_is_48 : List.median data_set = 48 :=
by sorry

-- Prove the range is 38
theorem range_is_38 : (List.maximum data_set - List.minimum data_set) = 38 :=
by sorry 

-- Prove the 5% quantile is 26
theorem quantile_5_percent_is_26 : (List.nth_le data_set 0 (by linarith) + List.nth_le data_set 1 (by linarith)) / 2 = 26 :=
by sorry

end mode_is_48_median_is_48_range_is_38_quantile_5_percent_is_26_l160_160003


namespace circle_eq_problem1_circle_eq_problem2_l160_160787

-- Problem 1
theorem circle_eq_problem1 :
  (∃ a b r : ℝ, (x - a)^2 + (y - b)^2 = r^2 ∧
  a - 2 * b - 3 = 0 ∧
  (2 - a)^2 + (-3 - b)^2 = r^2 ∧
  (-2 - a)^2 + (-5 - b)^2 = r^2) ↔
  (x + 1)^2 + (y + 2)^2 = 10 :=
sorry

-- Problem 2
theorem circle_eq_problem2 :
  (∃ D E F : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ∧
  (1:ℝ)^2 + (0:ℝ)^2 + D * 1 + E * 0 + F = 0 ∧
  (-1:ℝ)^2 + (-2:ℝ)^2 - D * 1 - 2 * E + F = 0 ∧
  (3:ℝ)^2 + (-2:ℝ)^2 + 3 * D - 2 * E + F = 0) ↔
  x^2 + y^2 - 2 * x + 4 * y + 1 = 0 :=
sorry

end circle_eq_problem1_circle_eq_problem2_l160_160787


namespace minimum_value_f_l160_160142

noncomputable def f (x : ℝ) (f1 f2 : ℝ) : ℝ :=
  f1 * x + f2 / x - 2

theorem minimum_value_f (f1 f2 : ℝ) (h1 : f2 = 2) (h2 : f1 = 3 / 2) :
  ∃ x > 0, f x f1 f2 = 2 * Real.sqrt 3 - 2 :=
by
  sorry

end minimum_value_f_l160_160142


namespace probability_product_divisible_by_8_l160_160173

theorem probability_product_divisible_by_8 :
  (let prob := 4 / 8;
       event := prob ^ 4 + prob ^ 3 * (3 / 8) + prob ^ 2 * (3 / 8)^2 * 3 + prob * (3 / 8)^3
   in 1 - event = 51 / 64) := by sorry

end probability_product_divisible_by_8_l160_160173


namespace probability_of_even_product_l160_160731

theorem probability_of_even_product :
  let spinner1 := [6, 7, 8, 9]
  let spinner2 := [10, 11, 12, 13, 14]
  (1 - (4 / ((4:ℚ) * 5))) = 4 / 5 := by
  sorry

end probability_of_even_product_l160_160731


namespace determine_a_l160_160809

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then x ^ 2 + a else 2 ^ x

theorem determine_a (a : ℝ) (h1 : a > -1) (h2 : f a (f a (-1)) = 4) : a = 1 :=
sorry

end determine_a_l160_160809


namespace find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160908

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end find_number_when_divided_by_3_is_equal_to_subtracting_5_l160_160908


namespace hall_reunion_attendees_l160_160730

theorem hall_reunion_attendees
  (total_guests : ℕ)
  (oates_attendees : ℕ)
  (both_attendees : ℕ)
  (h : total_guests = 100 ∧ oates_attendees = 50 ∧ both_attendees = 12) :
  ∃ (hall_attendees : ℕ), hall_attendees = 62 :=
by
  sorry

end hall_reunion_attendees_l160_160730


namespace p_necessary_but_not_sufficient_for_q_l160_160354

noncomputable def p (x : ℝ) : Prop := abs x ≤ 2
noncomputable def q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem p_necessary_but_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) := 
by 
  sorry

end p_necessary_but_not_sufficient_for_q_l160_160354


namespace function_monotonically_increasing_l160_160514

-- The function y = x^2 - 2x + 8
def f (x : ℝ) : ℝ := x^2 - 2 * x + 8

-- The theorem stating the function is monotonically increasing on (1, +∞)
theorem function_monotonically_increasing : ∀ x y : ℝ, (1 < x) → (x < y) → (f x < f y) :=
by
  -- Proof is omitted
  sorry

end function_monotonically_increasing_l160_160514


namespace slope_of_line_l160_160610

theorem slope_of_line (a b c : ℝ) (h : 3 * a = 4 * b - 9) : a = 4 / 3 * b - 3 :=
by
  sorry

end slope_of_line_l160_160610


namespace max_s_value_l160_160130

variables (X Y Z P X' Y' Z' : Type)
variables (p q r XX' YY' ZZ' s : ℝ)

-- Defining the conditions
def triangle_XYZ (p q r : ℝ) : Prop :=
p ≤ r ∧ r ≤ q ∧ p + q > r ∧ p + r > q ∧ q + r > p

def point_P_inside (X Y Z P : Type) : Prop :=
true -- Simplified assumption since point P is given to be inside

def segments_XX'_YY'_ZZ' (XX' YY' ZZ' : ℝ) : ℝ :=
XX' + YY' + ZZ'

def given_ratio (p q r : ℝ) : Prop :=
(p / (q + r)) = (r / (p + q))

-- The maximum value of s being 3p
def max_value_s_eq_3p (s p : ℝ) : Prop :=
s = 3 * p

-- The final theorem statement
theorem max_s_value 
  (p q r XX' YY' ZZ' s : ℝ)
  (h_triangle : triangle_XYZ p q r)
  (h_ratio : given_ratio p q r)
  (h_segments : s = segments_XX'_YY'_ZZ' XX' YY' ZZ') : 
  max_value_s_eq_3p s p :=
by
  sorry

end max_s_value_l160_160130


namespace spherical_to_rectangular_l160_160483

theorem spherical_to_rectangular :
  ∀ (ρ θ φ : ℝ),
  ρ = 3 →
  θ = (3 * π) / 2 →
  φ = π / 3 →
  let x := ρ * real.sin φ * real.cos θ,
      y := ρ * real.sin φ * real.sin θ,
      z := ρ * real.cos φ in
  (x, y, z) = (0, -3 * real.sqrt 3 / 2, 3 / 2) := by
  intros,
  -- The actual proof would go here
  sorry

end spherical_to_rectangular_l160_160483


namespace cat_count_l160_160702

def initial_cats : ℕ := 2
def female_kittens : ℕ := 3
def male_kittens : ℕ := 2
def total_kittens : ℕ := female_kittens + male_kittens
def total_cats : ℕ := initial_cats + total_kittens

theorem cat_count : total_cats = 7 := by
  unfold total_cats
  unfold initial_cats total_kittens
  unfold female_kittens male_kittens
  rfl

end cat_count_l160_160702


namespace weeks_saved_l160_160378

theorem weeks_saved (w : ℕ) :
  (10 * w / 2) - ((10 * w / 2) / 4) = 15 → 
  w = 4 := 
by
  sorry

end weeks_saved_l160_160378


namespace negation_of_proposition_l160_160675

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, 1 < x → (Real.log x / Real.log 2) + 4 * (Real.log 2 / Real.log x) > 4)) ↔
  (∃ x : ℝ, 1 < x ∧ (Real.log x / Real.log 2) + 4 * (Real.log 2 / Real.log x) ≤ 4) :=
sorry

end negation_of_proposition_l160_160675


namespace elliot_storeroom_blocks_l160_160958

def storeroom_volume (length: ℕ) (width: ℕ) (height: ℕ) : ℕ :=
  length * width * height

def inner_volume (length: ℕ) (width: ℕ) (height: ℕ) (thickness: ℕ) : ℕ :=
  (length - 2 * thickness) * (width - 2 * thickness) * (height - thickness)

def blocks_needed (outer_volume: ℕ) (inner_volume: ℕ) : ℕ :=
  outer_volume - inner_volume

theorem elliot_storeroom_blocks :
  let length := 15
  let width := 12
  let height := 8
  let thickness := 2
  let outer_volume := storeroom_volume length width height
  let inner_volume := inner_volume length width height thickness
  let required_blocks := blocks_needed outer_volume inner_volume
  required_blocks = 912 :=
by {
  -- Definitions and calculations as per conditions
  sorry
}

end elliot_storeroom_blocks_l160_160958


namespace fraction_division_l160_160313

variable {x : ℝ}
variable (hx : x ≠ 0)

theorem fraction_division (hx : x ≠ 0) : (3 / 8) / (5 * x / 12) = 9 / (10 * x) := 
by
  sorry

end fraction_division_l160_160313


namespace A_beats_B_by_63_l160_160526

variable (A B C : ℕ)

-- Condition: A beats C by 163 meters
def A_beats_C : Prop := A = 1000 - 163
-- Condition: B beats C by 100 meters
def B_beats_C (X : ℕ) : Prop := 1000 - X = 837 + 100
-- Main theorem statement
theorem A_beats_B_by_63 (X : ℕ) (h1 : A_beats_C A) (h2 : B_beats_C X): X = 63 :=
by
  sorry

end A_beats_B_by_63_l160_160526
