import Mathlib

namespace abc_inequality_l1764_176429

theorem abc_inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  (a * (a^2 + b * c)) / (b + c) + (b * (b^2 + c * a)) / (c + a) + (c * (c^2 + a * b)) / (a + b) ≥ a * b + b * c + c * a := 
by 
  sorry

end abc_inequality_l1764_176429


namespace tan_arccos_eq_2y_l1764_176412

noncomputable def y_squared : ℝ :=
  (-1 + Real.sqrt 17) / 8

theorem tan_arccos_eq_2y (y : ℝ) (hy : 0 < y) (htan : Real.tan (Real.arccos y) = 2 * y) :
  y^2 = y_squared := sorry

end tan_arccos_eq_2y_l1764_176412


namespace scientific_notation_of_million_l1764_176441

theorem scientific_notation_of_million (x : ℝ) (h : x = 2600000) : x = 2.6 * 10^6 := by
  sorry

end scientific_notation_of_million_l1764_176441


namespace total_pictures_l1764_176416

-- Definitions based on problem conditions
def Randy_pictures : ℕ := 5
def Peter_pictures : ℕ := Randy_pictures + 3
def Quincy_pictures : ℕ := Peter_pictures + 20
def Susan_pictures : ℕ := 2 * Quincy_pictures - 7
def Thomas_pictures : ℕ := Randy_pictures ^ 3

-- The proof statement
theorem total_pictures : Randy_pictures + Peter_pictures + Quincy_pictures + Susan_pictures + Thomas_pictures = 215 := by
  sorry

end total_pictures_l1764_176416


namespace monotonic_decreasing_interval_l1764_176463

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_decreasing_interval :
  {x : ℝ | x > 0} ∩ {x : ℝ | deriv f x < 0} = {x : ℝ | x > Real.exp 1} :=
by sorry

end monotonic_decreasing_interval_l1764_176463


namespace percent_of_amount_l1764_176425

theorem percent_of_amount (Part Whole : ℝ) (hPart : Part = 120) (hWhole : Whole = 80) :
  (Part / Whole) * 100 = 150 :=
by
  rw [hPart, hWhole]
  sorry

end percent_of_amount_l1764_176425


namespace mark_total_cost_is_correct_l1764_176485

variable (hours : ℕ) (hourly_rate part_cost : ℕ)

def total_cost (hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) :=
  hours * hourly_rate + part_cost

theorem mark_total_cost_is_correct : 
  hours = 2 → hourly_rate = 75 → part_cost = 150 → total_cost hours hourly_rate part_cost = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end mark_total_cost_is_correct_l1764_176485


namespace spadesuit_value_l1764_176424

-- Define the operation ♠ as a function
def spadesuit (a b : ℤ) : ℤ := |a - b|

theorem spadesuit_value : spadesuit 3 (spadesuit 5 8) = 0 :=
by
  -- Proof steps go here (we're skipping proof steps and directly writing sorry)
  sorry

end spadesuit_value_l1764_176424


namespace count_square_of_integer_fraction_l1764_176426

theorem count_square_of_integer_fraction :
  ∃ n_values : Finset ℤ, n_values = ({0, 15, 24} : Finset ℤ) ∧
  (∀ n ∈ n_values, ∃ k : ℤ, n / (30 - n) = k ^ 2) ∧
  n_values.card = 3 :=
by
  sorry

end count_square_of_integer_fraction_l1764_176426


namespace starting_even_number_l1764_176453

def is_even (n : ℤ) : Prop := n % 2 = 0

def span_covered_by_evens (count : ℤ) : ℤ := count * 2 - 2

theorem starting_even_number
  (count : ℤ)
  (end_num : ℤ)
  (H1 : is_even end_num)
  (H2 : count = 20)
  (H3 : end_num = 55) :
  ∃ start_num, is_even start_num ∧ start_num = end_num - span_covered_by_evens count + 1 := 
sorry

end starting_even_number_l1764_176453


namespace target_runs_l1764_176459

theorem target_runs (r1 r2 : ℝ) (o1 o2 : ℕ) (target : ℝ) :
  r1 = 3.6 ∧ o1 = 10 ∧ r2 = 6.15 ∧ o2 = 40 → target = (r1 * o1) + (r2 * o2) := by
  sorry

end target_runs_l1764_176459


namespace length_of_chord_l1764_176442

theorem length_of_chord {x1 x2 : ℝ} (h1 : ∃ (y : ℝ), y^2 = 8 * x1)
                                   (h2 : ∃ (y : ℝ), y^2 = 8 * x2)
                                   (h_midpoint : (x1 + x2) / 2 = 3) :
  x1 + x2 + 4 = 10 :=
sorry

end length_of_chord_l1764_176442


namespace horizontal_distance_parabola_l1764_176489

theorem horizontal_distance_parabola :
  ∀ x_p x_q : ℝ, 
  (x_p^2 + 3*x_p - 4 = 8) → 
  (x_q^2 + 3*x_q - 4 = 0) → 
  x_p ≠ x_q → 
  abs (x_p - x_q) = 2 :=
sorry

end horizontal_distance_parabola_l1764_176489


namespace selling_price_same_loss_as_profit_l1764_176484

theorem selling_price_same_loss_as_profit (cost_price selling_price_with_profit selling_price_with_loss profit loss : ℝ)
  (h1 : selling_price_with_profit - cost_price = profit)
  (h2 : cost_price - selling_price_with_loss = loss)
  (h3 : profit = loss) :
  selling_price_with_loss = 52 :=
by
  have h4 : selling_price_with_profit = 66 := by sorry
  have h5 : cost_price = 59 := by sorry
  have h6 : profit = 66 - 59 := by sorry
  have h7 : profit = 7 := by sorry
  have h8 : loss = 59 - selling_price_with_loss := by sorry
  have h9 : loss = 7 := by sorry
  have h10 : selling_price_with_loss = 59 - loss := by sorry
  have h11 : selling_price_with_loss = 59 - 7 := by sorry
  have h12 : selling_price_with_loss = 52 := by sorry
  exact h12

end selling_price_same_loss_as_profit_l1764_176484


namespace complex_fraction_l1764_176499

theorem complex_fraction (h : (1 : ℂ) - I = 1 - (I : ℂ)) :
  ((1 - I) * (1 - (2 * I))) / (1 + I) = -2 - I := 
by
  sorry

end complex_fraction_l1764_176499


namespace union_complement_with_B_l1764_176497

namespace SetTheory

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Definition of the complement of A relative to U in Lean
def C_U (A U : Set ℕ) : Set ℕ := U \ A

-- Theorem statement
theorem union_complement_with_B (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {0, 1, 2, 3}) (hB : B = {2, 3, 4}) : 
  (C_U A U) ∪ B = {2, 3, 4} :=
by
  -- Proof goes here
  sorry

end SetTheory

end union_complement_with_B_l1764_176497


namespace intersection_in_second_quadrant_l1764_176419

theorem intersection_in_second_quadrant (k : ℝ) (x y : ℝ) 
  (hk : 0 < k) (hk2 : k < 1/2) 
  (h1 : k * x - y = k - 1) 
  (h2 : k * y - x = 2 * k) : 
  x < 0 ∧ y > 0 := 
sorry

end intersection_in_second_quadrant_l1764_176419


namespace sum_three_digit_even_integers_l1764_176473

theorem sum_three_digit_even_integers :
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S = 247050 :=
by
  let a := 100
  let d := 2
  let l := 998
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  sorry

end sum_three_digit_even_integers_l1764_176473


namespace train_length_l1764_176406

theorem train_length
  (speed_km_hr : ℕ)
  (time_sec : ℕ)
  (length_train : ℕ)
  (length_platform : ℕ)
  (h_eq_len : length_train = length_platform)
  (h_speed : speed_km_hr = 108)
  (h_time : time_sec = 60) :
  length_train = 900 :=
by
  sorry

end train_length_l1764_176406


namespace percentage_saved_is_25_l1764_176402

def monthly_salary : ℝ := 1000

def increase_percentage : ℝ := 0.10

def saved_amount_after_increase : ℝ := 175

def calculate_percentage_saved (x : ℝ) : Prop := 
  1000 - (1000 - (x / 100) * monthly_salary) * (1 + increase_percentage) = saved_amount_after_increase

theorem percentage_saved_is_25 :
  ∃ x : ℝ, x = 25 ∧ calculate_percentage_saved x :=
sorry

end percentage_saved_is_25_l1764_176402


namespace count_final_numbers_l1764_176452

def initial_numbers : List ℕ := List.range' 1 20

def erased_even_numbers : List ℕ :=
  initial_numbers.filter (λ n => n % 2 ≠ 0)

def final_numbers : List ℕ :=
  erased_even_numbers.filter (λ n => n % 5 ≠ 4)

theorem count_final_numbers : final_numbers.length = 8 :=
by
  -- Here, we would proceed to prove the statement.
  sorry

end count_final_numbers_l1764_176452


namespace sphere_segment_volume_l1764_176492

theorem sphere_segment_volume (r : ℝ) (ratio_surface_to_base : ℝ) : r = 10 → ratio_surface_to_base = 10 / 7 → ∃ V : ℝ, V = 288 * π :=
by
  intros
  sorry

end sphere_segment_volume_l1764_176492


namespace best_fit_model_l1764_176447

theorem best_fit_model 
  (R2_model1 R2_model2 R2_model3 R2_model4 : ℝ)
  (h1 : R2_model1 = 0.976)
  (h2 : R2_model2 = 0.776)
  (h3 : R2_model3 = 0.076)
  (h4 : R2_model4 = 0.351) : 
  (R2_model1 > R2_model2) ∧ (R2_model1 > R2_model3) ∧ (R2_model1 > R2_model4) :=
by
  sorry

end best_fit_model_l1764_176447


namespace binomial_expansion_product_l1764_176405

theorem binomial_expansion_product (a a1 a2 a3 a4 a5 : ℤ)
  (h1 : (1 - 1)^5 = a + a1 + a2 + a3 + a4 + a5)
  (h2 : (1 - (-1))^5 = a - a1 + a2 - a3 + a4 - a5) :
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := by
  sorry

end binomial_expansion_product_l1764_176405


namespace base3_composite_numbers_l1764_176469

theorem base3_composite_numbers:
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 12002110 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2210121012 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 121212 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 102102 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 1001 * AB = a * b) :=
by {
  sorry
}

end base3_composite_numbers_l1764_176469


namespace area_of_triangle_l1764_176421

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 8 = 1

def foci_distance (F1 F2 : ℝ × ℝ) : Prop := (F1.1, F1.2) = (-3, 0) ∧ (F2.1, F2.2) = (3, 0)

def point_on_hyperbola (x y : ℝ) : Prop := hyperbola x y

def distance_ratios (P F1 F2 : ℝ × ℝ) : Prop := 
  let PF1 := (P.1 - F1.1)^2 + (P.2 - F1.2)^2
  let PF2 := (P.1 - F2.1)^2 + (P.2 - F2.2)^2
  PF1 / PF2 = 3 / 4

theorem area_of_triangle {P F1 F2 : ℝ × ℝ} 
  (H1 : foci_distance F1 F2)
  (H2 : point_on_hyperbola P.1 P.2)
  (H3 : distance_ratios P F1 F2) :
  let area := 1 / 2 * (6:ℝ) * (8:ℝ) * Real.sqrt 5
  area = 8 * Real.sqrt 5 := 
sorry

end area_of_triangle_l1764_176421


namespace find_y_l1764_176471

theorem find_y (x y : ℕ) (hx_positive : 0 < x) (hy_positive : 0 < y) (hmod : x % y = 9) (hdiv : (x : ℝ) / (y : ℝ) = 96.25) : y = 36 :=
sorry

end find_y_l1764_176471


namespace turnip_count_example_l1764_176408

theorem turnip_count_example : 6 + 9 = 15 := 
by
  -- Sorry is used to skip the actual proof
  sorry

end turnip_count_example_l1764_176408


namespace correct_statements_l1764_176446

theorem correct_statements (a b c x : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, ax^2 + bx + c ≤ 0 ↔ x ≤ -2 ∨ x ≥ 6)
  (hb : b = -4 * a)
  (hc : c = -12 * a) : 
  (a < 0) ∧ 
  (∀ x, cx^2 - bx + a < 0 ↔ -1/6 < x ∧ x < 1/2) ∧ 
  (a + b + c > 0) :=
by
  sorry

end correct_statements_l1764_176446


namespace subtracted_value_l1764_176466

-- Given conditions
def chosen_number : ℕ := 110
def result_number : ℕ := 110

-- Statement to prove
theorem subtracted_value : ∃ y : ℕ, 3 * chosen_number - y = result_number ∧ y = 220 :=
by
  sorry

end subtracted_value_l1764_176466


namespace gcd_digits_bounded_by_lcm_l1764_176407

theorem gcd_digits_bounded_by_lcm (a b : ℕ) (h_a : 10^6 ≤ a ∧ a < 10^7) (h_b : 10^6 ≤ b ∧ b < 10^7) (h_lcm : 10^10 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^11) : Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digits_bounded_by_lcm_l1764_176407


namespace complex_number_location_in_plane_l1764_176400

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem complex_number_location_in_plane :
  is_in_second_quadrant (-2) 5 :=
by
  sorry

end complex_number_location_in_plane_l1764_176400


namespace trucks_more_than_buses_l1764_176478

theorem trucks_more_than_buses (b t : ℕ) (h₁ : b = 9) (h₂ : t = 17) : t - b = 8 :=
by
  sorry

end trucks_more_than_buses_l1764_176478


namespace y1_gt_y2_for_line_through_points_l1764_176498

theorem y1_gt_y2_for_line_through_points (x1 y1 x2 y2 k b : ℝ) 
  (h_line_A : y1 = k * x1 + b) 
  (h_line_B : y2 = k * x2 + b) 
  (h_k_neq_0 : k ≠ 0)
  (h_k_pos : k > 0)
  (h_b_nonneg : b ≥ 0)
  (h_x1_gt_x2 : x1 > x2) : 
  y1 > y2 := 
  sorry

end y1_gt_y2_for_line_through_points_l1764_176498


namespace f_2023_value_l1764_176455

noncomputable def f : ℕ → ℝ := sorry

axiom f_condition (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : 2^n = a + b) : f a + f b = n^2 + 1

theorem f_2023_value : f 2023 = 107 :=
by 
  sorry

end f_2023_value_l1764_176455


namespace percent_decrease_of_y_l1764_176468

theorem percent_decrease_of_y (k x y q : ℝ) (h_inv_prop : x * y = k) (h_pos : 0 < x ∧ 0 < y) (h_q : 0 < q) :
  let x' := x * (1 + q / 100)
  let y' := y * 100 / (100 + q)
  (y - y') / y * 100 = (100 * q) / (100 + q) :=
by
  sorry

end percent_decrease_of_y_l1764_176468


namespace triangle_sine_inequality_l1764_176440

theorem triangle_sine_inequality
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (habc : a + b > c)
  (hbac : b + c > a)
  (hact : c + a > b)
  : |(a / (a + b)) + (b / (b + c)) + (c / (c + a)) - (3 / 2)| < (8 * Real.sqrt 2 - 5 * Real.sqrt 5) / 6 := 
sorry

end triangle_sine_inequality_l1764_176440


namespace fixed_point_of_tangent_line_l1764_176422

theorem fixed_point_of_tangent_line (x y : ℝ) (h1 : x = 3) 
  (h2 : ∃ m : ℝ, (3 - m)^2 + (y - 2)^2 = 4) :
  ∃ (k l : ℝ), k = 4 / 3 ∧ l = 2 :=
by
  sorry

end fixed_point_of_tangent_line_l1764_176422


namespace amount_of_b_l1764_176409

variable (A B : ℝ)

theorem amount_of_b (h₁ : A + B = 2530) (h₂ : (3 / 5) * A = (2 / 7) * B) : B = 1714 :=
sorry

end amount_of_b_l1764_176409


namespace ratio_addition_l1764_176404

theorem ratio_addition (a b : ℕ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := 
by sorry

end ratio_addition_l1764_176404


namespace edward_mowed_lawns_l1764_176432

theorem edward_mowed_lawns (L : ℕ) (h1 : 8 * L + 7 = 47) : L = 5 :=
by
  sorry

end edward_mowed_lawns_l1764_176432


namespace mask_production_decrease_l1764_176423

theorem mask_production_decrease (x : ℝ) : 
  (1 : ℝ) * (1 - x)^2 = 0.64 → 100 * (1 - x)^2 = 64 :=
by
  intro h
  sorry

end mask_production_decrease_l1764_176423


namespace sin_of_right_angle_l1764_176490

theorem sin_of_right_angle (A B C : Type)
  (angle_A : Real) (AB BC : Real)
  (h_angleA : angle_A = 90)
  (h_AB : AB = 16)
  (h_BC : BC = 24) :
  Real.sin (angle_A) = 1 :=
by
  sorry

end sin_of_right_angle_l1764_176490


namespace painting_methods_correct_l1764_176420

noncomputable def num_painting_methods : ℕ :=
  sorry 

theorem painting_methods_correct :
  num_painting_methods = 24 :=
by
  -- proof would go here
  sorry

end painting_methods_correct_l1764_176420


namespace remainder_of_99_times_101_divided_by_9_is_0_l1764_176434

theorem remainder_of_99_times_101_divided_by_9_is_0 : (99 * 101) % 9 = 0 :=
by
  sorry

end remainder_of_99_times_101_divided_by_9_is_0_l1764_176434


namespace radius_of_cone_base_l1764_176480

theorem radius_of_cone_base {R : ℝ} {theta : ℝ} (hR : R = 6) (htheta : theta = 120) :
  ∃ r : ℝ, r = 2 :=
by
  sorry

end radius_of_cone_base_l1764_176480


namespace vector_magnitude_proof_l1764_176430

theorem vector_magnitude_proof (a b : ℝ × ℝ) 
  (h₁ : ‖a‖ = 1) 
  (h₂ : ‖b‖ = 2)
  (h₃ : a - b = (Real.sqrt 3, Real.sqrt 2)) : 
‖a + (2:ℝ) • b‖ = Real.sqrt 17 := 
sorry

end vector_magnitude_proof_l1764_176430


namespace number_of_sides_of_polygon_24_deg_exterior_angle_l1764_176454

theorem number_of_sides_of_polygon_24_deg_exterior_angle :
  (∀ (n : ℕ), (∀ (k : ℕ), k = 360 / 24 → n = k)) :=
by
  sorry

end number_of_sides_of_polygon_24_deg_exterior_angle_l1764_176454


namespace cricket_avg_score_l1764_176481

theorem cricket_avg_score
  (avg_first_two : ℕ)
  (num_first_two : ℕ)
  (avg_all_five : ℕ)
  (num_all_five : ℕ)
  (avg_first_two_eq : avg_first_two = 40)
  (num_first_two_eq : num_first_two = 2)
  (avg_all_five_eq : avg_all_five = 22)
  (num_all_five_eq : num_all_five = 5) :
  ((num_all_five * avg_all_five - num_first_two * avg_first_two) / (num_all_five - num_first_two) = 10) :=
by
  sorry

end cricket_avg_score_l1764_176481


namespace car_owners_without_motorcycle_or_bicycle_l1764_176417

noncomputable def total_adults := 500
noncomputable def car_owners := 400
noncomputable def motorcycle_owners := 200
noncomputable def bicycle_owners := 150
noncomputable def car_motorcycle_owners := 100
noncomputable def motorcycle_bicycle_owners := 50
noncomputable def car_bicycle_owners := 30

theorem car_owners_without_motorcycle_or_bicycle :
  car_owners - car_motorcycle_owners - car_bicycle_owners = 270 := by
  sorry

end car_owners_without_motorcycle_or_bicycle_l1764_176417


namespace determine_n_l1764_176456

-- Define the condition
def eq1 := (1 : ℚ) / (2 ^ 10) + (1 : ℚ) / (2 ^ 9) + (1 : ℚ) / (2 ^ 8)
def eq2 (n : ℚ) := n / (2 ^ 10)

-- The lean statement for the proof problem
theorem determine_n : ∃ (n : ℤ), eq1 = eq2 n ∧ n > 0 ∧ n = 7 := by
  sorry

end determine_n_l1764_176456


namespace fin_solutions_l1764_176477

theorem fin_solutions (u : ℕ) (hu : u > 0) :
  ∃ N : ℕ, ∀ n a b : ℕ, n > N → ¬ (n! = u^a - u^b) :=
sorry

end fin_solutions_l1764_176477


namespace distance_from_point_to_circle_center_l1764_176445

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def circle_center : ℝ × ℝ := (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_from_point_to_circle_center :
  distance (polar_to_rect 2 (Real.pi / 3)) circle_center = Real.sqrt 3 := sorry

end distance_from_point_to_circle_center_l1764_176445


namespace discount_is_25_l1764_176451

def original_price : ℕ := 76
def discounted_price : ℕ := 51
def discount_amount : ℕ := original_price - discounted_price

theorem discount_is_25 : discount_amount = 25 := by
  sorry

end discount_is_25_l1764_176451


namespace a9_value_l1764_176403

theorem a9_value (a : ℕ → ℝ) (x : ℝ) (h : (1 + x) ^ 10 = 
  (a 0) + (a 1) * (1 - x) + (a 2) * (1 - x)^2 + 
  (a 3) * (1 - x)^3 + (a 4) * (1 - x)^4 + 
  (a 5) * (1 - x)^5 + (a 6) * (1 - x)^6 + 
  (a 7) * (1 - x)^7 + (a 8) * (1 - x)^8 + 
  (a 9) * (1 - x)^9 + (a 10) * (1 - x)^10) : 
  a 9 = -20 :=
sorry

end a9_value_l1764_176403


namespace max_real_root_lt_100_l1764_176487

theorem max_real_root_lt_100 (k a b c : ℕ) (r : ℝ)
  (ha : ∃ m : ℕ, a = k^m)
  (hb : ∃ n : ℕ, b = k^n)
  (hc : ∃ l : ℕ, c = k^l)
  (one_real_solution : b^2 = 4 * a * c)
  (r_is_root : ∃ r : ℝ, a * r^2 - b * r + c = 0)
  (r_lt_100 : r < 100) :
  r ≤ 64 := sorry

end max_real_root_lt_100_l1764_176487


namespace more_flour_than_sugar_l1764_176437

def cups_of_flour : Nat := 9
def cups_of_sugar : Nat := 6
def flour_added : Nat := 2
def flour_needed : Nat := cups_of_flour - flour_added -- 9 - 2 = 7

theorem more_flour_than_sugar : flour_needed - cups_of_sugar = 1 :=
by
  sorry

end more_flour_than_sugar_l1764_176437


namespace quadratic_equations_with_common_root_l1764_176472

theorem quadratic_equations_with_common_root :
  ∃ (p1 q1 p2 q2 : ℝ),
    p1 ≠ p2 ∧ q1 ≠ q2 ∧
    ∀ x : ℝ,
      (x^2 + p1 * x + q1 = 0 ∧ x^2 + p2 * x + q2 = 0) →
      (x = 2 ∨ (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ ((x = r1 ∧ x == 2) ∨ (x = r2 ∧ x == 2)))) :=
sorry

end quadratic_equations_with_common_root_l1764_176472


namespace average_is_correct_l1764_176488

def numbers : List ℕ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

def sum_of_numbers : ℕ := numbers.foldr (· + ·) 0

def number_of_values : ℕ := numbers.length

def average : ℚ := sum_of_numbers / number_of_values

theorem average_is_correct : average = 114391.82 := by
  sorry

end average_is_correct_l1764_176488


namespace total_people_in_group_l1764_176462

theorem total_people_in_group (men women children : ℕ)
  (h1 : men = 2 * women)
  (h2 : women = 3 * children)
  (h3 : children = 30) :
  men + women + children = 300 :=
by
  sorry

end total_people_in_group_l1764_176462


namespace count_unbroken_matches_l1764_176465

theorem count_unbroken_matches :
  let n_1 := 5 * 12  -- number of boxes in the first set
  let matches_1 := n_1 * 20  -- total matches in first set of boxes
  let broken_1 := n_1 * 3  -- total broken matches in first set of boxes
  let unbroken_1 := matches_1 - broken_1  -- unbroken matches in first set of boxes

  let n_2 := 4  -- number of extra boxes
  let matches_2 := n_2 * 25  -- total matches in extra boxes
  let broken_2 := (matches_2 / 5)  -- total broken matches in extra boxes (20%)
  let unbroken_2 := matches_2 - broken_2  -- unbroken matches in extra boxes

  let total_unbroken := unbroken_1 + unbroken_2  -- total unbroken matches

  total_unbroken = 1100 := 
by
  sorry

end count_unbroken_matches_l1764_176465


namespace distance_upstream_l1764_176495

variable (v : ℝ) -- speed of the stream in km/h
variable (t : ℝ := 6) -- time of each trip in hours
variable (d_down : ℝ := 24) -- distance for downstream trip in km
variable (u : ℝ := 3) -- speed of man in still water in km/h

/- The distance the man swam upstream -/
theorem distance_upstream : 
  24 = (u + v) * t → 
  ∃ (d_up : ℝ), 
    d_up = (u - v) * t ∧
    d_up = 12 :=
by
  sorry

end distance_upstream_l1764_176495


namespace max_value_3absx_2absy_l1764_176461

theorem max_value_3absx_2absy (x y : ℝ) (h : x^2 + y^2 = 9) : 
  3 * abs x + 2 * abs y ≤ 9 :=
sorry

end max_value_3absx_2absy_l1764_176461


namespace third_highest_score_l1764_176433

theorem third_highest_score
  (mean15 : ℕ → ℚ) (mean12 : ℕ → ℚ) 
  (sum15 : ℕ) (sum12 : ℕ) (highest : ℕ) (third_highest : ℕ) (third_is_100: third_highest = 100) :
  (mean15 15 = 90) →
  (mean12 12 = 85) →
  (highest = 120) →
  (sum15 = 15 * 90) →
  (sum12 = 12 * 85) →
  (sum15 - sum12 = highest + 210) →
  third_highest = 100 := 
by
  intros hm15 hm12 hhigh hsum15 hsum12 hdiff
  sorry

end third_highest_score_l1764_176433


namespace hamburger_cost_l1764_176457

variable (H : ℝ)

theorem hamburger_cost :
  (H + 2 + 3 = 20 - 11) → (H = 4) :=
by
  sorry

end hamburger_cost_l1764_176457


namespace exists_three_digit_number_l1764_176411

theorem exists_three_digit_number : ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ (100 * a + 10 * b + c = a^3 + b^3 + c^3) ∧ (100 * a + 10 * b + c ≥ 100 ∧ 100 * a + 10 * b + c < 1000) := 
sorry

end exists_three_digit_number_l1764_176411


namespace find_2u_plus_3v_l1764_176479

theorem find_2u_plus_3v (u v : ℚ) (h1 : 5 * u - 6 * v = 28) (h2 : 3 * u + 5 * v = -13) :
  2 * u + 3 * v = -7767 / 645 := 
sorry

end find_2u_plus_3v_l1764_176479


namespace kim_monthly_expenses_l1764_176448

-- Define the conditions

def initial_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def payback_period : ℕ := 10

-- Define the proof statement
theorem kim_monthly_expenses :
  ∃ (E : ℝ), 
    (payback_period * (monthly_revenue - E) = initial_cost) → (E = 1500) :=
by
  sorry

end kim_monthly_expenses_l1764_176448


namespace not_age_of_child_l1764_176470

theorem not_age_of_child (ages : Set ℕ) (h_ages : ∀ x ∈ ages, 4 ≤ x ∧ x ≤ 10) : 
  5 ∉ ages := by
  let number := 1122
  have h_number : number % 5 ≠ 0 := by decide
  have h_divisible : ∀ x ∈ ages, number % x = 0 := sorry
  exact sorry

end not_age_of_child_l1764_176470


namespace notebook_and_pen_prices_l1764_176450

theorem notebook_and_pen_prices (x y : ℕ) (h1 : 2 * x + y = 30) (h2 : x = 2 * y) :
  x = 12 ∧ y = 6 :=
by
  sorry

end notebook_and_pen_prices_l1764_176450


namespace plane_split_into_regions_l1764_176436

theorem plane_split_into_regions : 
  let line1 (x : ℝ) := 3 * x
  let line2 (x : ℝ) := (1 / 3) * x
  let line3 (x : ℝ) := 4 * x
  ∃ regions : ℕ, regions = 7 :=
by
  let line1 (x : ℝ) := 3 * x
  let line2 (x : ℝ) := (1 / 3) * x
  let line3 (x : ℝ) := 4 * x
  existsi 7
  sorry

end plane_split_into_regions_l1764_176436


namespace tangent_parallel_x_axis_tangent_45_degrees_x_axis_l1764_176467

-- Condition: Define the curve
def curve (x : ℝ) : ℝ := x^2 - 1

-- Condition: Calculate derivative
def derivative_curve (x : ℝ) : ℝ := 2 * x

-- Part (a): Point where tangent is parallel to the x-axis
theorem tangent_parallel_x_axis :
  (∃ x y : ℝ, y = curve x ∧ derivative_curve x = 0 ∧ x = 0 ∧ y = -1) :=
  sorry

-- Part (b): Point where tangent forms a 45 degree angle with the x-axis
theorem tangent_45_degrees_x_axis :
  (∃ x y : ℝ, y = curve x ∧ derivative_curve x = 1 ∧ x = 1/2 ∧ y = -3/4) :=
  sorry

end tangent_parallel_x_axis_tangent_45_degrees_x_axis_l1764_176467


namespace person_speed_l1764_176438

theorem person_speed (distance_m : ℝ) (time_min : ℝ) (h₁ : distance_m = 800) (h₂ : time_min = 5) : 
  let distance_km := distance_m / 1000
  let time_hr := time_min / 60
  distance_km / time_hr = 9.6 := 
by
  sorry

end person_speed_l1764_176438


namespace geometric_sequence_formula_l1764_176401

def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) (h_geom : geom_seq a)
  (h1 : a 3 = 2) (h2 : a 6 = 16) :
  ∀ n : ℕ, a n = 2 ^ (n - 2) :=
by
  sorry

end geometric_sequence_formula_l1764_176401


namespace higher_room_amount_higher_60_l1764_176410

variable (higher_amount : ℕ)

theorem higher_room_amount_higher_60 
  (total_rent : ℕ) (amount_credited_50 : ℕ)
  (total_reduction : ℕ)
  (condition1 : total_rent = 400)
  (condition2 : amount_credited_50 = 50)
  (condition3 : total_reduction = total_rent / 4)
  (condition4 : 10 * higher_amount - 10 * amount_credited_50 = total_reduction) :
  higher_amount = 60 := 
sorry

end higher_room_amount_higher_60_l1764_176410


namespace solve_for_nabla_l1764_176428

theorem solve_for_nabla (nabla : ℤ) (h : 5 * (-4) = nabla + 4) : nabla = -24 :=
by {
  sorry
}

end solve_for_nabla_l1764_176428


namespace evaluate_expression_l1764_176483

theorem evaluate_expression : ⌈(7 : ℝ) / 3⌉ + ⌊- (7 : ℝ) / 3⌋ = 0 := 
by 
  sorry

end evaluate_expression_l1764_176483


namespace trapezoid_smallest_angle_l1764_176482

theorem trapezoid_smallest_angle (a d : ℝ) 
  (h1 : a + 3 * d = 140)
  (h2 : 2 * a + 3 * d = 180) : 
  a = 20 :=
by
  sorry

end trapezoid_smallest_angle_l1764_176482


namespace grid_diagonal_segments_l1764_176449

theorem grid_diagonal_segments (m n : ℕ) (hm : m = 100) (hn : n = 101) :
    let d := m + n - gcd m n
    d = 200 := by
  sorry

end grid_diagonal_segments_l1764_176449


namespace bryce_raisins_l1764_176491

theorem bryce_raisins:
  ∃ x : ℕ, (x - 8 = x / 3) ∧ x = 12 :=
by 
  sorry

end bryce_raisins_l1764_176491


namespace calculate_Delta_l1764_176415

-- Define the Delta operation
def Delta (a b : ℚ) : ℚ := (a^2 + b^2) / (1 + a^2 * b^2)

-- Constants for the specific problem
def two := (2 : ℚ)
def three := (3 : ℚ)
def four := (4 : ℚ)

theorem calculate_Delta : Delta (Delta two three) four = 5945 / 4073 := by
  sorry

end calculate_Delta_l1764_176415


namespace ara_height_l1764_176431

/-
Conditions:
1. Shea's height increased by 25%.
2. Shea is now 65 inches tall.
3. Ara grew by three-quarters as many inches as Shea did.

Prove Ara's height is 61.75 inches.
-/

def shea_original_height (x : ℝ) : Prop := 1.25 * x = 65

def ara_growth (growth : ℝ) (shea_growth : ℝ) : Prop := growth = (3 / 4) * shea_growth

def shea_growth (original_height : ℝ) : ℝ := 0.25 * original_height

theorem ara_height (shea_orig_height : ℝ) (shea_now_height : ℝ) (ara_growth_inches : ℝ) :
  shea_original_height shea_orig_height → 
  shea_now_height = 65 →
  ara_growth ara_growth_inches (shea_now_height - shea_orig_height) →
  shea_orig_height + ara_growth_inches = 61.75 :=
by
  sorry

end ara_height_l1764_176431


namespace find_n_l1764_176435

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 101) (h3 : 100 * n % 101 = 72) : n = 29 := 
by
  sorry

end find_n_l1764_176435


namespace inversely_directly_proportional_l1764_176439

theorem inversely_directly_proportional (m n z : ℝ) (x : ℝ) (h₁ : x = 4) (hz₁ : z = 16) (hz₂ : z = 64) (hy : ∃ y : ℝ, y = n * Real.sqrt z) (hx : ∃ m y : ℝ, x = m / y^2)
: x = 1 :=
by
  sorry

end inversely_directly_proportional_l1764_176439


namespace set_equality_l1764_176474

theorem set_equality (M P : Set (ℝ × ℝ))
  (hM : M = {p : ℝ × ℝ | p.1 + p.2 < 0 ∧ p.1 * p.2 > 0})
  (hP : P = {p : ℝ × ℝ | p.1 < 0 ∧ p.2 < 0}) : M = P :=
by
  sorry

end set_equality_l1764_176474


namespace purely_imaginary_sufficient_but_not_necessary_l1764_176476

theorem purely_imaginary_sufficient_but_not_necessary (a b : ℝ) (h : ¬(b = 0)) : 
  (a = 0 → p ∧ q) → (q ∧ ¬p) :=
by
  sorry

end purely_imaginary_sufficient_but_not_necessary_l1764_176476


namespace find_f_one_l1764_176458

noncomputable def f_inv (x : ℝ) : ℝ := 2^(x + 1)

theorem find_f_one : ∃ f : ℝ → ℝ, (∀ y, f (f_inv y) = y) ∧ f 1 = -1 :=
by
  sorry

end find_f_one_l1764_176458


namespace gcd_9155_4892_l1764_176475

theorem gcd_9155_4892 : Nat.gcd 9155 4892 = 1 := 
by 
  sorry

end gcd_9155_4892_l1764_176475


namespace hydrogen_atoms_in_compound_l1764_176464

theorem hydrogen_atoms_in_compound :
  ∀ (H_atoms Br_atoms O_atoms total_molecular_weight weight_H weight_Br weight_O : ℝ),
  Br_atoms = 1 ∧ O_atoms = 3 ∧ total_molecular_weight = 129 ∧ 
  weight_H = 1 ∧ weight_Br = 79.9 ∧ weight_O = 16 →
  H_atoms = 1 :=
by
  sorry

end hydrogen_atoms_in_compound_l1764_176464


namespace jaxon_toys_l1764_176494

-- Definitions as per the conditions
def toys_jaxon : ℕ := sorry
def toys_gabriel : ℕ := 2 * toys_jaxon
def toys_jerry : ℕ := 2 * toys_jaxon + 8
def total_toys : ℕ := toys_jaxon + toys_gabriel + toys_jerry

-- Theorem to prove
theorem jaxon_toys : total_toys = 83 → toys_jaxon = 15 := sorry

end jaxon_toys_l1764_176494


namespace work_rate_ab_l1764_176486

variables (A B C : ℝ)

-- Defining the work rates as per the conditions
def work_rate_bc := 1 / 6 -- (b and c together in 6 days)
def work_rate_ca := 1 / 3 -- (c and a together in 3 days)
def work_rate_c := 1 / 8 -- (c alone in 8 days)

-- The main theorem that proves a and b together can complete the work in 4 days,
-- based on the above conditions.
theorem work_rate_ab : 
  (B + C = work_rate_bc) ∧ (C + A = work_rate_ca) ∧ (C = work_rate_c) 
  → (A + B = 1 / 4) :=
by sorry

end work_rate_ab_l1764_176486


namespace cost_of_each_shirt_l1764_176496

theorem cost_of_each_shirt (initial_money : ℕ) (cost_pants : ℕ) (money_left : ℕ) (shirt_cost : ℕ)
  (h1 : initial_money = 109)
  (h2 : cost_pants = 13)
  (h3 : money_left = 74)
  (h4 : initial_money - (2 * shirt_cost + cost_pants) = money_left) :
  shirt_cost = 11 :=
by
  sorry

end cost_of_each_shirt_l1764_176496


namespace total_beakers_count_l1764_176493

variable (total_beakers_with_ions : ℕ) 
variable (drops_per_test : ℕ)
variable (total_drops_used : ℕ) 
variable (beakers_without_ions : ℕ)

theorem total_beakers_count
  (h1 : total_beakers_with_ions = 8)
  (h2 : drops_per_test = 3)
  (h3 : total_drops_used = 45)
  (h4 : beakers_without_ions = 7) : 
  (total_drops_used / drops_per_test) = (total_beakers_with_ions + beakers_without_ions) :=
by
  -- Proof to be filled in
  sorry

end total_beakers_count_l1764_176493


namespace number_of_divisors_of_2744_l1764_176413

-- Definition of the integer and its prime factorization
def two := 2
def seven := 7
def n := two^3 * seven^3

-- Define the property for the number of divisors
def num_divisors (n : ℕ) : ℕ := (3 + 1) * (3 + 1)

-- Main proof statement
theorem number_of_divisors_of_2744 : num_divisors n = 16 := by
  sorry

end number_of_divisors_of_2744_l1764_176413


namespace sum_of_remainders_eq_3_l1764_176427

theorem sum_of_remainders_eq_3 (a b c : ℕ) (h1 : a % 59 = 28) (h2 : b % 59 = 15) (h3 : c % 59 = 19) (h4 : a = b + d ∨ b = c + d ∨ c = a + d) : 
  (a + b + c) % 59 = 3 :=
by {
  sorry -- Proof to be constructed
}

end sum_of_remainders_eq_3_l1764_176427


namespace find_k_l1764_176444

theorem find_k (m n : ℝ) 
  (h₁ : m = k * n + 5) 
  (h₂ : m + 2 = k * (n + 0.5) + 5) : 
  k = 4 :=
by
  sorry

end find_k_l1764_176444


namespace inequality_solutions_l1764_176443

theorem inequality_solutions :
  (∀ x : ℝ, 2 * x / (x + 1) < 1 ↔ -1 < x ∧ x < 1) ∧
  (∀ a x : ℝ,
    (x^2 + (2 - a) * x - 2 * a ≥ 0 ↔
      (a = -2 → True) ∧
      (a > -2 → (x ≤ -2 ∨ x ≥ a)) ∧
      (a < -2 → (x ≤ a ∨ x ≥ -2)))) :=
by
  sorry

end inequality_solutions_l1764_176443


namespace no_primes_divisible_by_45_l1764_176460

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_primes_divisible_by_45 : 
  ∀ p, is_prime p → ¬ (45 ∣ p) := 
by
  sorry

end no_primes_divisible_by_45_l1764_176460


namespace woman_first_half_speed_l1764_176414

noncomputable def first_half_speed (total_time : ℕ) (second_half_speed : ℕ) (total_distance : ℕ) : ℕ :=
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let second_half_time := second_half_distance / second_half_speed
  let first_half_time := total_time - second_half_time
  first_half_distance / first_half_time

theorem woman_first_half_speed : first_half_speed 20 24 448 = 21 := by
  sorry

end woman_first_half_speed_l1764_176414


namespace triangle_area_l1764_176418

theorem triangle_area (BC AC : ℝ) (angle_BAC : ℝ) (h1 : BC = 12) (h2 : AC = 5) (h3 : angle_BAC = π / 6) :
  1/2 * BC * (AC * Real.sin angle_BAC) = 15 :=
by
  sorry

end triangle_area_l1764_176418
