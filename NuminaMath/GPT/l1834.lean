import Mathlib

namespace problem_1_problem_2_l1834_183446

def set_A := { y : ℝ | 2 < y ∧ y < 3 }
def set_B := { x : ℝ | x > 1 ∨ x < -1 }

theorem problem_1 : { x : ℝ | x ∈ set_A ∧ x ∈ set_B } = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

def set_C := { x : ℝ | x ∈ set_B ∧ ¬(x ∈ set_A) }

theorem problem_2 : set_C = { x : ℝ | x < -1 ∨ (1 < x ∧ x ≤ 2) ∨ x ≥ 3 } :=
by
  sorry

end problem_1_problem_2_l1834_183446


namespace pump_fill_time_without_leak_l1834_183443

variable (T : ℕ)

def rate_pump (T : ℕ) : ℚ := 1 / T
def rate_leak : ℚ := 1 / 20

theorem pump_fill_time_without_leak : rate_pump T - rate_leak = rate_leak → T = 10 := by 
  intro h
  sorry

end pump_fill_time_without_leak_l1834_183443


namespace eq_abc_gcd_l1834_183408

theorem eq_abc_gcd
  (a b c d : ℕ)
  (h1 : a^a * b^(a + b) = c^c * d^(c + d))
  (h2 : Nat.gcd a b = 1)
  (h3 : Nat.gcd c d = 1) : 
  a = c ∧ b = d := 
sorry

end eq_abc_gcd_l1834_183408


namespace measure_of_angle_A_values_of_b_and_c_l1834_183400

variable (a b c : ℝ) (A : ℝ)

-- Declare the conditions as hypotheses
def condition1 (a b c : ℝ) := a^2 - c^2 = b^2 - b * c
def condition2 (a : ℝ) := a = 2
def condition3 (b c : ℝ) := b + c = 4

-- Proof that A = 60 degrees when the conditions are satisfied
theorem measure_of_angle_A (h : condition1 a b c) : A = 60 := by
  sorry

-- Proof that b and c are 2 when given conditions are satisfied
theorem values_of_b_and_c (h1 : condition1 2 b c) (h2 : condition3 b c) : b = 2 ∧ c = 2 := by
  sorry

end measure_of_angle_A_values_of_b_and_c_l1834_183400


namespace circle_occupies_62_8_percent_l1834_183412

noncomputable def largestCirclePercentage (length : ℝ) (width : ℝ) : ℝ :=
  let radius := width / 2
  let circle_area := Real.pi * radius^2
  let rectangle_area := length * width
  (circle_area / rectangle_area) * 100

theorem circle_occupies_62_8_percent : largestCirclePercentage 5 4 = 62.8 := 
by 
  /- Sorry, skipping the proof -/
  sorry

end circle_occupies_62_8_percent_l1834_183412


namespace tom_bought_6_hardcover_l1834_183499

-- Given conditions and statements
def toms_books_condition_1 (h p : ℕ) : Prop :=
  h + p = 10

def toms_books_condition_2 (h p : ℕ) : Prop :=
  28 * h + 18 * p = 240

-- The theorem to prove
theorem tom_bought_6_hardcover (h p : ℕ) 
  (h_condition : toms_books_condition_1 h p)
  (c_condition : toms_books_condition_2 h p) : 
  h = 6 :=
sorry

end tom_bought_6_hardcover_l1834_183499


namespace quadratic_has_two_distinct_roots_l1834_183486

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_has_two_distinct_roots 
  (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  discriminant a b c > 0 :=
sorry

end quadratic_has_two_distinct_roots_l1834_183486


namespace parallel_lines_slope_l1834_183490

theorem parallel_lines_slope (k : ℝ) :
  (∀ x : ℝ, 5 * x - 3 = (3 * k) * x + 7 -> ((3 * k) = 5)) -> (k = 5 / 3) :=
by
  -- Posing the conditions on parallel lines
  intro h_eq_slopes
  -- We know 3k = 5, hence k = 5 / 3
  have slope_eq : 3 * k = 5 := by sorry
  -- Therefore k = 5 / 3 follows from the fact 3k = 5
  have k_val : k = 5 / 3 := by sorry
  exact k_val

end parallel_lines_slope_l1834_183490


namespace kevin_wings_record_l1834_183484

-- Conditions
def alanWingsPerMinute : ℕ := 5
def additionalWingsNeeded : ℕ := 4
def kevinRecordDuration : ℕ := 8

-- Question and answer
theorem kevin_wings_record : 
  (alanWingsPerMinute + additionalWingsNeeded) * kevinRecordDuration = 72 :=
by
  sorry

end kevin_wings_record_l1834_183484


namespace eccentricity_hyperbola_l1834_183461

theorem eccentricity_hyperbola : 
  let a2 := 4
  let b2 := 5
  let e := Real.sqrt (1 + (b2 / a2))
  e = 3 / 2 := by
    apply sorry

end eccentricity_hyperbola_l1834_183461


namespace jigi_scored_55_percent_l1834_183429

noncomputable def jigi_percentage (max_score : ℕ) (avg_score : ℕ) (gibi_pct mike_pct lizzy_pct : ℕ) : ℕ := sorry

theorem jigi_scored_55_percent :
  jigi_percentage 700 490 59 99 67 = 55 :=
sorry

end jigi_scored_55_percent_l1834_183429


namespace intersection_M_N_l1834_183444

open Set

def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N := {x : ℝ | x > 1}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l1834_183444


namespace af2_plus_bfg_plus_cg2_geq_0_l1834_183406

theorem af2_plus_bfg_plus_cg2_geq_0 (a b c : ℝ) (f g : ℝ) :
  (a * f^2 + b * f * g + c * g^2 ≥ 0) ↔ (a ≥ 0 ∧ c ≥ 0 ∧ 4 * a * c ≥ b^2) := 
sorry

end af2_plus_bfg_plus_cg2_geq_0_l1834_183406


namespace trigonometric_identity_l1834_183452

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α)) / 
  (Real.cos (3 * Real.pi / 2 - α) + 2 * Real.cos (-Real.pi + α)) = -2 / 5 := 
by
  sorry

end trigonometric_identity_l1834_183452


namespace probability_of_same_type_is_correct_l1834_183482

noncomputable def total_socks : ℕ := 12 + 10 + 6
noncomputable def ways_to_pick_any_3_socks : ℕ := Nat.choose total_socks 3
noncomputable def ways_to_pick_3_black_socks : ℕ := Nat.choose 12 3
noncomputable def ways_to_pick_3_white_socks : ℕ := Nat.choose 10 3
noncomputable def ways_to_pick_3_striped_socks : ℕ := Nat.choose 6 3
noncomputable def ways_to_pick_3_same_type : ℕ := ways_to_pick_3_black_socks + ways_to_pick_3_white_socks + ways_to_pick_3_striped_socks
noncomputable def probability_same_type : ℚ := ways_to_pick_3_same_type / ways_to_pick_any_3_socks

theorem probability_of_same_type_is_correct :
  probability_same_type = 60 / 546 :=
by
  sorry

end probability_of_same_type_is_correct_l1834_183482


namespace radius_of_circle_is_ten_l1834_183453

noncomputable def radius_of_circle (diameter : ℝ) : ℝ :=
  diameter / 2

theorem radius_of_circle_is_ten :
  radius_of_circle 20 = 10 :=
by
  unfold radius_of_circle
  sorry

end radius_of_circle_is_ten_l1834_183453


namespace sufficient_but_not_necessary_to_increasing_l1834_183440

theorem sufficient_but_not_necessary_to_increasing (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → (x^2 - 2*a*x) ≤ (y^2 - 2*a*y)) ↔ (a ≤ 1) := sorry

end sufficient_but_not_necessary_to_increasing_l1834_183440


namespace sqrt_nine_factorial_over_72_eq_l1834_183421

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_nine_factorial_over_72_eq : 
  Real.sqrt ((factorial 9) / 72) = 12 * Real.sqrt 35 :=
by
  sorry

end sqrt_nine_factorial_over_72_eq_l1834_183421


namespace arithmetic_expression_evaluation_l1834_183409

theorem arithmetic_expression_evaluation :
  (-18) + (-12) - (-33) + 17 = 20 :=
by
  sorry

end arithmetic_expression_evaluation_l1834_183409


namespace beef_original_weight_l1834_183455

theorem beef_original_weight (W : ℝ) (h : 0.65 * W = 546): W = 840 :=
sorry

end beef_original_weight_l1834_183455


namespace gcd_35_x_eq_7_in_range_80_90_l1834_183476

theorem gcd_35_x_eq_7_in_range_80_90 {n : ℕ} (h₁ : Nat.gcd 35 n = 7) (h₂ : 80 < n) (h₃ : n < 90) : n = 84 :=
by
  sorry

end gcd_35_x_eq_7_in_range_80_90_l1834_183476


namespace remainder_calculation_l1834_183445

theorem remainder_calculation 
  (dividend divisor quotient : ℕ)
  (h1 : dividend = 140)
  (h2 : divisor = 15)
  (h3 : quotient = 9) :
  dividend = (divisor * quotient) + (dividend - (divisor * quotient)) := by
sorry

end remainder_calculation_l1834_183445


namespace similar_triangle_perimeter_l1834_183449

theorem similar_triangle_perimeter 
  (a b c : ℝ) (a_sim : ℝ)
  (h1 : a = b) (h2 : b = c)
  (h3 : a = 15) (h4 : a_sim = 45)
  (h5 : a_sim / a = 3) :
  a_sim + a_sim + a_sim = 135 :=
by
  sorry

end similar_triangle_perimeter_l1834_183449


namespace max_f_l1834_183419

open Real

noncomputable def f (x : ℝ) : ℝ := 3 + log x + 4 / log x

theorem max_f (h : 0 < x ∧ x < 1) : f x ≤ -1 :=
sorry

end max_f_l1834_183419


namespace find_t_l1834_183437

variable (g V V0 c S t : ℝ)
variable (h1 : V = g * t + V0 + c)
variable (h2 : S = (1/2) * g * t^2 + V0 * t + c * t^2)

theorem find_t
  (h1 : V = g * t + V0 + c)
  (h2 : S = (1/2) * g * t^2 + V0 * t + c * t^2) :
  t = 2 * S / (V + V0 - c) :=
sorry

end find_t_l1834_183437


namespace earnings_from_roosters_l1834_183487

-- Definitions from the conditions
def price_per_kg : Float := 0.50
def weight_of_rooster1 : Float := 30.0
def weight_of_rooster2 : Float := 40.0

-- The theorem we need to prove (mathematically equivalent proof problem)
theorem earnings_from_roosters (p : Float := price_per_kg)
                               (w1 : Float := weight_of_rooster1)
                               (w2 : Float := weight_of_rooster2) :
  p * w1 + p * w2 = 35.0 := 
by {
  sorry
}

end earnings_from_roosters_l1834_183487


namespace mean_temperature_is_88_75_l1834_183420

def temperatures : List ℕ := [85, 84, 85, 88, 91, 93, 94, 90]

theorem mean_temperature_is_88_75 : (List.sum temperatures : ℚ) / temperatures.length = 88.75 := by
  sorry

end mean_temperature_is_88_75_l1834_183420


namespace claire_sleep_hours_l1834_183438

def hours_in_day := 24
def cleaning_hours := 4
def cooking_hours := 2
def crafting_hours := 5
def tailoring_hours := crafting_hours

theorem claire_sleep_hours :
  hours_in_day - (cleaning_hours + cooking_hours + crafting_hours + tailoring_hours) = 8 := by
  sorry

end claire_sleep_hours_l1834_183438


namespace range_of_a_l1834_183411

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a * x + 1 < 0) → -2 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l1834_183411


namespace ella_stamps_value_l1834_183458

theorem ella_stamps_value :
  let total_stamps := 18
  let value_of_6_stamps := 18
  let consistent_value_per_stamp := value_of_6_stamps / 6
  total_stamps * consistent_value_per_stamp = 54 := by
  sorry

end ella_stamps_value_l1834_183458


namespace distinct_real_roots_find_other_root_and_k_l1834_183464

-- Definition of the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Part (1): Proving the discriminant condition
theorem distinct_real_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq 2 k (-1) x1 = 0 ∧ quadratic_eq 2 k (-1) x2 = 0 := by
  sorry

-- Part (2): Finding the other root and the value of k
theorem find_other_root_and_k : 
  ∃ k : ℝ, ∃ x2 : ℝ,
    quadratic_eq 2 1 (-1) (-1) = 0 ∧ quadratic_eq 2 1 (-1) x2 = 0 ∧ k = 1 ∧ x2 = 1/2 := by
  sorry

end distinct_real_roots_find_other_root_and_k_l1834_183464


namespace trees_died_due_to_typhoon_l1834_183462

-- defining the initial number of trees
def initial_trees : ℕ := 9

-- defining the additional trees grown after the typhoon
def additional_trees : ℕ := 5

-- defining the final number of trees after all events
def final_trees : ℕ := 10

-- we introduce D as the number of trees that died due to the typhoon
def trees_died (D : ℕ) : Prop := initial_trees - D + additional_trees = final_trees

-- the theorem we need to prove is that 4 trees died
theorem trees_died_due_to_typhoon : trees_died 4 :=
by
  sorry

end trees_died_due_to_typhoon_l1834_183462


namespace find_angle_A_l1834_183488

theorem find_angle_A (A B C a b c : ℝ) 
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : a > 0)
  (h5 : b > 0)
  (h6 : c > 0)
  (sin_eq : Real.sin (C + π / 6) = b / (2 * a)) :
  A = π / 6 :=
sorry

end find_angle_A_l1834_183488


namespace negation_of_at_most_four_is_at_least_five_l1834_183422

theorem negation_of_at_most_four_is_at_least_five :
  (∀ n : ℕ, n ≤ 4) ↔ (∃ n : ℕ, n ≥ 5) := 
sorry

end negation_of_at_most_four_is_at_least_five_l1834_183422


namespace nonrational_ab_l1834_183473

theorem nonrational_ab {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) : 
    ¬(∃ (p q r s : ℤ), q ≠ 0 ∧ s ≠ 0 ∧ a = p / q ∧ b = r / s) := by
  sorry

end nonrational_ab_l1834_183473


namespace cheryl_needed_first_material_l1834_183456

noncomputable def cheryl_material (x : ℚ) : ℚ :=
  x + 1 / 3 - 3 / 8

theorem cheryl_needed_first_material
  (h_total_used : 0.33333333333333326 = 1 / 3) :
  cheryl_material x = 1 / 3 → x = 3 / 8 :=
by
  intros
  rw [h_total_used] at *
  sorry

end cheryl_needed_first_material_l1834_183456


namespace area_of_triangle_CDE_l1834_183457

theorem area_of_triangle_CDE
  (DE : ℝ) (h : ℝ)
  (hDE : DE = 12) (hh : h = 15) :
  1/2 * DE * h = 90 := by
  sorry

end area_of_triangle_CDE_l1834_183457


namespace find_abc_l1834_183441

open Real

theorem find_abc 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h1 : a * (b + c) = 154)
  (h2 : b * (c + a) = 164) 
  (h3 : c * (a + b) = 172) : 
  (a * b * c = Real.sqrt 538083) := 
by 
  sorry

end find_abc_l1834_183441


namespace concert_ticket_sales_l1834_183480

theorem concert_ticket_sales (A C : ℕ) (total : ℕ) :
  (C = 3 * A) →
  (7 * A + 3 * C = 6000) →
  (total = A + C) →
  total = 1500 :=
by
  intros
  -- The proof is not required
  sorry

end concert_ticket_sales_l1834_183480


namespace avg_bc_eq_70_l1834_183498

-- Definitions of the given conditions
variables (a b c : ℝ)

def avg_ab (a b : ℝ) : Prop := (a + b) / 2 = 45
def diff_ca (a c : ℝ) : Prop := c - a = 50

-- The main theorem statement
theorem avg_bc_eq_70 (h1 : avg_ab a b) (h2 : diff_ca a c) : (b + c) / 2 = 70 :=
by
  sorry

end avg_bc_eq_70_l1834_183498


namespace calculate_wheel_radii_l1834_183459

theorem calculate_wheel_radii (rpmA rpmB : ℕ) (length : ℝ) (r R : ℝ) :
  rpmA = 1200 →
  rpmB = 1500 →
  length = 9 →
  (4 : ℝ) / 5 * r = R →
  2 * (R + r) = 9 →
  r = 2 ∧ R = 2.5 :=
by
  intros
  sorry

end calculate_wheel_radii_l1834_183459


namespace smallest_prime_factor_of_difference_l1834_183468

theorem smallest_prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 1 ≤ C ∧ C ≤ 9) (h_diff : A ≠ C) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) ∧ p = 3 :=
by
  sorry

end smallest_prime_factor_of_difference_l1834_183468


namespace distance_from_point_to_plane_l1834_183460

-- Definitions representing the conditions
def side_length_base := 6
def base_area := side_length_base * side_length_base
def volume_pyramid := 96

-- Proof statement
theorem distance_from_point_to_plane (h : ℝ) : 
  (1/3) * base_area * h = volume_pyramid → h = 8 := 
by 
  sorry

end distance_from_point_to_plane_l1834_183460


namespace number_pairs_sum_diff_prod_quotient_l1834_183470

theorem number_pairs_sum_diff_prod_quotient (x y : ℤ) (h : x ≥ y) :
  (x + y) + (x - y) + x * y + x / y = 800 ∨ (x + y) + (x - y) + x * y + x / y = 400 :=
sorry

-- Correct answers for A = 800
example : (38 + 19) + (38 - 19) + 38 * 19 + 38 / 19 = 800 := by norm_num
example : (-42 + -21) + (-42 - -21) + (-42 * -21) + (-42 / -21) = 800 := by norm_num
example : (72 + 9) + (72 - 9) + 72 * 9 + 72 / 9 = 800 := by norm_num
example : (-88 + -11) + (-88 - -11) + -(88 * -11) + (-88 / -11) = 800 := by norm_num
example : (128 + 4) + (128 - 4) + 128 * 4 + 128 / 4 = 800 := by norm_num
example : (-192 + -6) + (-192 - -6) + -192 * -6 + ( -192 / -6 ) = 800 := by norm_num
example : (150 + 3) + (150 - 3) + 150 * 3 + 150 / 3 = 800 := by norm_num
example : (-250 + -5) + (-250 - -5) + (-250 * -5) + (-250 / -5) = 800 := by norm_num
example : (200 + 1) + (200 - 1) + 200 * 1 + 200 / 1 = 800 := by norm_num
example : (-600 + -3) + (-600 - -3) + -600 * -3 + -600 / -3 = 800 := by norm_num

-- Correct answers for A = 400
example : (19 + 19) + (19 - 19) + 19 * 19 + 19 / 19 = 400 := by norm_num
example : (-21 + -21) + (-21 - -21) + (-21 * -21) + (-21 / -21) = 400 := by norm_num
example : (36 + 9) + (36 - 9) + 36 * 9 + 36 / 9 = 400 := by norm_num
example : (-44 + -11) + (-44 - -11) + (-44 * -11) + (-44 / -11) = 400 := by norm_num
example : (64 + 4) + (64 - 4) + 64 * 4 + 64 / 4 = 400 := by norm_num
example : (-96 + -6) + (-96 - -6) + (-96 * -6) + (-96 / -6) = 400 := by norm_num
example : (75 + 3) + (75 - 3) + 75 * 3 + 75 / 3 = 400 := by norm_num
example : (-125 + -5) + (-125 - -5) + (-125 * -5) + (-125 / -5) = 400 := by norm_num
example : (100 + 1) + (100 - 1) + 100 * 1 + 100 / 1 = 400 := by norm_num
example : (-300 + -3) + (-300 - -3) + (-300 * -3) + (-300 / -3) = 400 := by norm_num

end number_pairs_sum_diff_prod_quotient_l1834_183470


namespace calculate_max_marks_l1834_183478

theorem calculate_max_marks (shortfall_math : ℕ) (shortfall_science : ℕ) 
                            (shortfall_literature : ℕ) (shortfall_social_studies : ℕ)
                            (required_math : ℕ) (required_science : ℕ)
                            (required_literature : ℕ) (required_social_studies : ℕ)
                            (max_math : ℕ) (max_science : ℕ)
                            (max_literature : ℕ) (max_social_studies : ℕ) :
                            shortfall_math = 40 ∧ required_math = 95 ∧ max_math = 800 ∧
                            shortfall_science = 35 ∧ required_science = 92 ∧ max_science = 438 ∧
                            shortfall_literature = 30 ∧ required_literature = 90 ∧ max_literature = 300 ∧
                            shortfall_social_studies = 25 ∧ required_social_studies = 88 ∧ max_social_studies = 209 :=
by
  sorry

end calculate_max_marks_l1834_183478


namespace regular_polygon_sides_l1834_183435

theorem regular_polygon_sides (n : ℕ) (h : 2 ≤ n) (h_angle : 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l1834_183435


namespace david_marks_in_english_l1834_183454

theorem david_marks_in_english 
  (math : ℤ) (phys : ℤ) (chem : ℤ) (bio : ℤ) (avg : ℤ) 
  (marks_per_math : math = 85) 
  (marks_per_phys : phys = 92) 
  (marks_per_chem : chem = 87) 
  (marks_per_bio : bio = 95) 
  (avg_marks : avg = 89) 
  (num_subjects : ℤ := 5) :
  ∃ (eng : ℤ), eng + 85 + 92 + 87 + 95 = 89 * 5 ∧ eng = 86 :=
by
  sorry

end david_marks_in_english_l1834_183454


namespace infinite_geometric_series_sum_l1834_183417

noncomputable def a : ℚ := 5 / 3
noncomputable def r : ℚ := -1 / 2

theorem infinite_geometric_series_sum : 
  ∑' (n : ℕ), a * r^n = 10 / 9 := 
by sorry

end infinite_geometric_series_sum_l1834_183417


namespace sugar_ratio_l1834_183479

theorem sugar_ratio (total_sugar : ℕ)  (bags : ℕ) (remaining_sugar : ℕ) (sugar_each_bag : ℕ) (sugar_fell : ℕ)
  (h1 : total_sugar = 24) (h2 : bags = 4) (h3 : total_sugar - remaining_sugar = sugar_fell) 
  (h4 : total_sugar / bags = sugar_each_bag) (h5 : remaining_sugar = 21) : 
  2 * sugar_fell = sugar_each_bag := by
  -- proof goes here
  sorry

end sugar_ratio_l1834_183479


namespace div_by_20_l1834_183471

theorem div_by_20 (n : ℕ) : 20 ∣ (9 ^ (8 * n + 4) - 7 ^ (8 * n + 4)) :=
  sorry

end div_by_20_l1834_183471


namespace solution_set_of_inequality_l1834_183416

variable {R : Type} [LinearOrderedField R]

theorem solution_set_of_inequality (f : R -> R) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x y, 0 < x ∧ x < y → f x < f y) (h3 : f 1 = 0) :
  { x : R | (f x - f (-x)) / x < 0 } = { x : R | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) } :=
sorry

end solution_set_of_inequality_l1834_183416


namespace max_percent_liquid_X_l1834_183433

theorem max_percent_liquid_X (wA wB wC : ℝ) (XA XB XC YA YB YC : ℝ)
  (hXA : XA = 0.8 / 100) (hXB : XB = 1.8 / 100) (hXC : XC = 3.0 / 100)
  (hYA : YA = 2.0 / 100) (hYB : YB = 1.0 / 100) (hYC : YC = 0.5 / 100)
  (hwA : wA = 500) (hwB : wB = 700) (hwC : wC = 300)
  (H_combined_limit : XA * wA + XB * wB + XC * wC + YA * wA + YB * wB + YC * wC ≤ 0.025 * (wA + wB + wC)) :
  XA * wA + XB * wB + XC * wC ≤ 0.0171 * (wA + wB + wC) :=
sorry

end max_percent_liquid_X_l1834_183433


namespace value_less_than_mean_by_std_dev_l1834_183427

theorem value_less_than_mean_by_std_dev :
  ∀ (mean value std_dev : ℝ), mean = 16.2 → std_dev = 2.3 → value = 11.6 → 
  (mean - value) / std_dev = 2 :=
by
  intros mean value std_dev h_mean h_std_dev h_value
  -- The proof goes here, but per instructions, it is skipped
  -- So we put 'sorry' to indicate that the proof is intentionally left incomplete
  sorry

end value_less_than_mean_by_std_dev_l1834_183427


namespace david_marks_in_english_l1834_183469

theorem david_marks_in_english : 
  ∀ (E : ℕ), 
  let math_marks := 85 
  let physics_marks := 82 
  let chemistry_marks := 87 
  let biology_marks := 85 
  let avg_marks := 85 
  let total_subjects := 5 
  let total_marks := avg_marks * total_subjects 
  let total_known_subject_marks := math_marks + physics_marks + chemistry_marks + biology_marks 
  total_marks = total_known_subject_marks + E → 
  E = 86 :=
by 
  intros
  sorry

end david_marks_in_english_l1834_183469


namespace number_of_integer_pairs_l1834_183489

theorem number_of_integer_pairs (n : ℕ) : 
  (∀ x y : ℤ, 5 * x^2 - 6 * x * y + y^2 = 6^100) → n = 19594 :=
sorry

end number_of_integer_pairs_l1834_183489


namespace jessies_current_weight_l1834_183410

theorem jessies_current_weight (initial_weight lost_weight : ℝ) (h1 : initial_weight = 69) (h2 : lost_weight = 35) :
  initial_weight - lost_weight = 34 :=
by sorry

end jessies_current_weight_l1834_183410


namespace wall_thickness_is_correct_l1834_183414

-- Define the dimensions of the brick.
def brick_length : ℝ := 80
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the number of required bricks.
def num_bricks : ℝ := 2000

-- Define the dimensions of the wall.
def wall_length : ℝ := 800
def wall_height : ℝ := 600

-- The volume of one brick.
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- The volume of the wall.
def wall_volume (T : ℝ) : ℝ := wall_length * wall_height * T

-- The thickness of the wall to be proved.
theorem wall_thickness_is_correct (T_wall : ℝ) (h : num_bricks * brick_volume = wall_volume T_wall) : 
  T_wall = 22.5 :=
sorry

end wall_thickness_is_correct_l1834_183414


namespace num_students_third_class_num_students_second_class_l1834_183496

-- Definition of conditions for both problems
def class_student_bounds (n : ℕ) : Prop := 40 < n ∧ n ≤ 50
def option_one_cost (n : ℕ) : ℕ := 40 * n * 7 / 10
def option_two_cost (n : ℕ) : ℕ := 40 * (n - 6) * 8 / 10

-- Problem Part 1
theorem num_students_third_class (x : ℕ) (h1 : class_student_bounds x) (h2 : option_one_cost x = option_two_cost x) : x = 48 := 
sorry

-- Problem Part 2
theorem num_students_second_class (y : ℕ) (h1 : class_student_bounds y) (h2 : option_one_cost y < option_two_cost y) : y = 49 ∨ y = 50 := 
sorry

end num_students_third_class_num_students_second_class_l1834_183496


namespace rent_3600_rents_88_max_revenue_is_4050_l1834_183494

def num_total_cars : ℕ := 100
def initial_rent : ℕ := 3000
def rent_increase_step : ℕ := 50
def maintenance_cost_rented : ℕ := 150
def maintenance_cost_unrented : ℕ := 50

def rented_cars (rent : ℕ) : ℕ :=
  if rent < initial_rent then num_total_cars
  else num_total_cars - ((rent - initial_rent) / rent_increase_step)

def monthly_revenue (rent : ℕ) : ℕ :=
  let rented := rented_cars rent
  rent * rented - (rented * maintenance_cost_rented + (num_total_cars - rented) * maintenance_cost_unrented)

theorem rent_3600_rents_88 :
  rented_cars 3600 = 88 := by 
  sorry

theorem max_revenue_is_4050 :
  ∃ (rent : ℕ), rent = 4050 ∧ monthly_revenue rent = 37050 := by
  sorry

end rent_3600_rents_88_max_revenue_is_4050_l1834_183494


namespace discount_price_l1834_183428

theorem discount_price (P P_d : ℝ) 
  (h1 : P_d = 0.85 * P) 
  (P_final : ℝ) 
  (h2 : P_final = 1.25 * P_d) 
  (h3 : P - P_final = 5.25) :
  P_d = 71.4 :=
by
  sorry

end discount_price_l1834_183428


namespace lcm_of_36_and_45_l1834_183434

theorem lcm_of_36_and_45 : Nat.lcm 36 45 = 180 := by
  sorry

end lcm_of_36_and_45_l1834_183434


namespace largest_multiple_of_15_less_than_500_l1834_183436

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l1834_183436


namespace sam_bikes_speed_l1834_183472

noncomputable def EugeneSpeed : ℝ := 5
noncomputable def ClaraSpeed : ℝ := (3/4) * EugeneSpeed
noncomputable def SamSpeed : ℝ := (4/3) * ClaraSpeed

theorem sam_bikes_speed :
  SamSpeed = 5 :=
by
  -- Proof will be filled here.
  sorry

end sam_bikes_speed_l1834_183472


namespace range_of_m_plus_n_l1834_183418

noncomputable def f (m n x : ℝ) : ℝ := m * 2^x + x^2 + n * x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x : ℝ, f m n x = 0 ∧ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by
  sorry

end range_of_m_plus_n_l1834_183418


namespace gcd_2024_1728_l1834_183413

theorem gcd_2024_1728 : Int.gcd 2024 1728 = 8 := 
by
  sorry

end gcd_2024_1728_l1834_183413


namespace part1_solution_set_part2_value_of_t_l1834_183423

open Real

def f (t x : ℝ) : ℝ := x^2 - (t + 1) * x + t

-- Statement for the equivalent proof problem
theorem part1_solution_set (x : ℝ) : 
  (t = 3 → f 3 x > 0 ↔ (x < 1) ∨ (x > 3)) :=
by
  sorry

theorem part2_value_of_t (t : ℝ) :
  (∀ x : ℝ, f t x ≥ 0) → t = 1 :=
by
  sorry

end part1_solution_set_part2_value_of_t_l1834_183423


namespace symmetric_scanning_codes_count_l1834_183493

-- Definition of a symmetric 8x8 scanning code grid under given conditions
def is_symmetric_code (grid : Fin 8 → Fin 8 → Bool) : Prop :=
  ∀ i j : Fin 8, grid i j = grid (7 - i) (7 - j) ∧ grid i j = grid j i

def at_least_one_each_color (grid : Fin 8 → Fin 8 → Bool) : Prop :=
  ∃ i j k l : Fin 8, grid i j = true ∧ grid k l = false

def total_symmetric_scanning_codes : Nat :=
  1022

theorem symmetric_scanning_codes_count :
  ∀ (grid : Fin 8 → Fin 8 → Bool), is_symmetric_code grid ∧ at_least_one_each_color grid → 
  1022 = total_symmetric_scanning_codes :=
by
  sorry

end symmetric_scanning_codes_count_l1834_183493


namespace budget_spent_on_salaries_l1834_183491

theorem budget_spent_on_salaries :
  ∀ (B R U E S T : ℕ),
  R = 9 ∧
  U = 5 ∧
  E = 4 ∧
  S = 2 ∧
  T = (72 * 100) / 360 → 
  B = 100 →
  (B - (R + U + E + S + T)) = 60 :=
by sorry

end budget_spent_on_salaries_l1834_183491


namespace possible_value_of_a_l1834_183415

theorem possible_value_of_a (a : ℕ) : (5 + 8 > a ∧ a > 3) → (a = 9 → True) :=
by
  intros h ha
  sorry

end possible_value_of_a_l1834_183415


namespace determine_initial_fund_l1834_183448

def initial_amount_fund (n : ℕ) := 60 * n + 30 - 10

theorem determine_initial_fund (n : ℕ) (h : 50 * n + 110 = 60 * n - 10) : initial_amount_fund n = 740 :=
by
  -- we skip the proof steps here
  sorry

end determine_initial_fund_l1834_183448


namespace count_multiples_of_7_not_14_l1834_183424

theorem count_multiples_of_7_not_14 (n : ℕ) : (n < 500 ∧ n % 7 = 0 ∧ n % 14 ≠ 0) → ∃ (k : ℕ), k = 36 :=
by
  sorry

end count_multiples_of_7_not_14_l1834_183424


namespace min_value_of_sum_l1834_183463

theorem min_value_of_sum (a b : ℤ) (h : a * b = 150) : a + b = -151 :=
  sorry

end min_value_of_sum_l1834_183463


namespace div_30_div_510_div_66_div_large_l1834_183407

theorem div_30 (a : ℤ) : 30 ∣ (a^5 - a) := 
  sorry  

theorem div_510 (a : ℤ) : 510 ∣ (a^17 - a) := 
  sorry

theorem div_66 (a : ℤ) : 66 ∣ (a^11 - a) := 
  sorry

theorem div_large (a : ℤ) : (2 * 3 * 5 * 7 * 13 * 19 * 37 * 73) ∣ (a^73 - a) := 
  sorry  

end div_30_div_510_div_66_div_large_l1834_183407


namespace find_k_l1834_183450

noncomputable def g (a b c : ℤ) (x : ℤ) := a * x^2 + b * x + c

theorem find_k (a b c k : ℤ) 
  (h1 : g a b c (-1) = 0) 
  (h2 : 30 < g a b c 5) (h3 : g a b c 5 < 40)
  (h4 : 120 < g a b c 7) (h5 : g a b c 7 < 130)
  (h6 : 2000 * k < g a b c 50) (h7 : g a b c 50 < 2000 * (k + 1)) : 
  k = 5 := 
sorry

end find_k_l1834_183450


namespace object_reaches_max_height_at_three_l1834_183426

theorem object_reaches_max_height_at_three :
  ∀ (h : ℝ) (t : ℝ), h = -15 * (t - 3)^2 + 150 → t = 3 :=
by
  sorry

end object_reaches_max_height_at_three_l1834_183426


namespace parabola_trajectory_l1834_183475

theorem parabola_trajectory (P : ℝ × ℝ) : 
  (dist P (3, 0) = dist P (3 - 1, P.2 - 0)) → P.2^2 = 12 * P.1 := 
sorry

end parabola_trajectory_l1834_183475


namespace find_a_l1834_183403

noncomputable def A (a : ℝ) : Set ℝ := {2^a, 3}
def B : Set ℝ := {2, 3}
def C : Set ℝ := {1, 2, 3}

theorem find_a (a : ℝ) (h : A a ∪ B = C) : a = 0 :=
sorry

end find_a_l1834_183403


namespace solve_for_k_l1834_183492

theorem solve_for_k :
  (∀ x : ℤ, (2 * x + 4 = 4 * (x - 2)) ↔ ( -x + 17 = 2 * x - 1 )) :=
by
  sorry

end solve_for_k_l1834_183492


namespace game_cost_l1834_183474

theorem game_cost (initial_money : ℕ) (toys_count : ℕ) (toy_price : ℕ) (left_money : ℕ) : 
  initial_money = 63 ∧ toys_count = 5 ∧ toy_price = 3 ∧ left_money = 15 → 
  (initial_money - left_money = 48) :=
by
  sorry

end game_cost_l1834_183474


namespace length_ab_l1834_183467

section geometry

variables {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Define the lengths and perimeters as needed
variables (AB AC BC CD DE CE : ℝ)

-- Isosceles Triangle properties
axiom isosceles_abc : AC = BC
axiom isosceles_cde : CD = DE

-- Conditons given in the problem
axiom perimeter_cde : CE + CD + DE = 22
axiom perimeter_abc : AB + BC + AC = 24
axiom length_ce : CE = 8

-- Goal: To prove the length of AB
theorem length_ab : AB = 10 :=
by 
  sorry

end geometry

end length_ab_l1834_183467


namespace gcd_228_1995_l1834_183404

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l1834_183404


namespace total_games_l1834_183431

theorem total_games (teams : ℕ) (games_per_pair : ℕ) (h_teams : teams = 12) (h_games_per_pair : games_per_pair = 4) : 
  (teams * (teams - 1) / 2) * games_per_pair = 264 :=
by
  sorry

end total_games_l1834_183431


namespace three_number_product_l1834_183402

theorem three_number_product
  (x y z : ℝ)
  (h1 : x + y = 18)
  (h2 : x ^ 2 + y ^ 2 = 220)
  (h3 : z = x - y) :
  x * y * z = 104 * Real.sqrt 29 :=
sorry

end three_number_product_l1834_183402


namespace problem_statement_l1834_183405

theorem problem_statement
  (a b c d : ℕ)
  (h1 : (b + c + d) / 3 + 2 * a = 54)
  (h2 : (a + c + d) / 3 + 2 * b = 50)
  (h3 : (a + b + d) / 3 + 2 * c = 42)
  (h4 : (a + b + c) / 3 + 2 * d = 30) :
  a = 17 ∨ b = 17 ∨ c = 17 ∨ d = 17 :=
by
  sorry

end problem_statement_l1834_183405


namespace inequality_solution_l1834_183439

open Set

theorem inequality_solution (x : ℝ) : (1 - 7 / (2 * x - 1) < 0) ↔ (1 / 2 < x ∧ x < 4) := 
by
  sorry

end inequality_solution_l1834_183439


namespace complex_number_identity_l1834_183451

theorem complex_number_identity : |-i| + i^2018 = 0 := by
  sorry

end complex_number_identity_l1834_183451


namespace coefficient_x2y3_in_expansion_l1834_183466

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem coefficient_x2y3_in_expansion (x y : ℝ) : 
  binomial 5 3 * (2 : ℝ) ^ 2 * (-1 : ℝ) ^ 3 = -40 := by
sorry

end coefficient_x2y3_in_expansion_l1834_183466


namespace line_does_not_pass_through_third_quadrant_l1834_183497

variable {a b c : ℝ}

theorem line_does_not_pass_through_third_quadrant
  (hac : a * c < 0) (hbc : b * c < 0) : ¬ ∃ x y, x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0 :=
sorry

end line_does_not_pass_through_third_quadrant_l1834_183497


namespace point_in_first_quadrant_l1834_183495

-- Define the imaginary unit i
def i := Complex.I

-- Define the complex number z
def z := i * (2 - i)

-- Define a predicate that checks if a complex number is in the first quadrant
def isFirstQuadrant (x : ℂ) : Prop := x.re > 0 ∧ x.im > 0

-- State the theorem
theorem point_in_first_quadrant : isFirstQuadrant z := sorry

end point_in_first_quadrant_l1834_183495


namespace number_of_customers_trimmed_l1834_183477

-- Definitions based on the conditions
def total_sounds : ℕ := 60
def sounds_per_person : ℕ := 20

-- Statement to prove
theorem number_of_customers_trimmed :
  ∃ n : ℕ, n * sounds_per_person = total_sounds ∧ n = 3 :=
sorry

end number_of_customers_trimmed_l1834_183477


namespace tic_tac_toe_tie_fraction_l1834_183425

theorem tic_tac_toe_tie_fraction :
  let amys_win : ℚ := 5 / 12
  let lilys_win : ℚ := 1 / 4
  1 - (amys_win + lilys_win) = 1 / 3 :=
by
  sorry

end tic_tac_toe_tie_fraction_l1834_183425


namespace basketball_game_total_points_l1834_183430

theorem basketball_game_total_points :
  ∃ (a d b: ℕ) (r: ℝ), 
      a = b + 2 ∧     -- Eagles lead by 2 points at the end of the first quarter
      (a + d < 100) ∧ -- Points scored by Eagles in each quarter form an increasing arithmetic sequence
      (b * r < 100) ∧ -- Points scored by Lions in each quarter form an increasing geometric sequence
      (a + (a + d) + (a + 2 * d)) = b * (1 + r + r^2) ∧ -- Aggregate score tied at the end of the third quarter
      (a + (a + d) + (a + 2 * d) + (a + 3 * d) + b * (1 + r + r^2 + r^3) = 144) -- Total points scored by both teams 
   :=
sorry

end basketball_game_total_points_l1834_183430


namespace sum_of_coefficients_is_neg40_l1834_183485

noncomputable def p (x : ℝ) : ℝ := 3 * (x^8 - x^5 + 2 * x^3 - 6) - 5 * (x^4 + 3 * x^2) + 2 * (x^6 - 5)

theorem sum_of_coefficients_is_neg40 : p 1 = -40 := by
  sorry

end sum_of_coefficients_is_neg40_l1834_183485


namespace building_height_l1834_183447

-- We start by defining the heights of the stories.
def first_story_height : ℕ := 12
def additional_height_per_story : ℕ := 3
def number_of_stories : ℕ := 20
def first_ten_stories : ℕ := 10
def remaining_stories : ℕ := number_of_stories - first_ten_stories

-- Now we define what it means for the total height of the building to be 270 feet.
theorem building_height :
  first_ten_stories * first_story_height + remaining_stories * (first_story_height + additional_height_per_story) = 270 := by
  sorry

end building_height_l1834_183447


namespace total_amount_l1834_183432

variable (Brad Josh Doug : ℝ)

axiom h1 : Josh = 2 * Brad
axiom h2 : Josh = (3 / 4) * Doug
axiom h3 : Doug = 32

theorem total_amount : Brad + Josh + Doug = 68 := by
  sorry

end total_amount_l1834_183432


namespace foldable_topless_cubical_box_count_l1834_183481

def isFoldable (placement : Char) : Bool :=
  placement = 'C' ∨ placement = 'E' ∨ placement = 'G'

theorem foldable_topless_cubical_box_count :
  (List.filter isFoldable ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']).length = 3 :=
by
  sorry

end foldable_topless_cubical_box_count_l1834_183481


namespace correct_assignment_statement_l1834_183442

def is_assignment_statement (stmt : String) : Prop :=
  stmt = "a = 2a"

theorem correct_assignment_statement : is_assignment_statement "a = 2a" :=
by
  sorry

end correct_assignment_statement_l1834_183442


namespace price_per_ticket_is_six_l1834_183483

-- Definition of the conditions
def total_tickets (friends_tickets extra_tickets : ℕ) : ℕ :=
  friends_tickets + extra_tickets

def total_cost (tickets price_per_ticket : ℕ) : ℕ :=
  tickets * price_per_ticket

-- Given conditions
def friends_tickets : ℕ := 8
def extra_tickets : ℕ := 2
def total_spent : ℕ := 60

-- Formulate the problem to prove the price per ticket
theorem price_per_ticket_is_six :
  ∃ (price_per_ticket : ℕ), price_per_ticket = 6 ∧ 
  total_cost (total_tickets friends_tickets extra_tickets) price_per_ticket = total_spent :=
by
  -- The proof is not required; we assume its correctness here.
  sorry

end price_per_ticket_is_six_l1834_183483


namespace contribution_of_eight_families_l1834_183401

/-- Definition of the given conditions --/
def classroom := 200
def two_families := 2 * 20
def ten_families := 10 * 5
def missing_amount := 30

def total_raised (x : ℝ) : ℝ := two_families + ten_families + 8 * x

/-- The main theorem to prove the contribution of each of the eight families --/
theorem contribution_of_eight_families (x : ℝ) (h : total_raised x = classroom - missing_amount) : x = 10 := by
  sorry

end contribution_of_eight_families_l1834_183401


namespace real_roots_of_quadratic_l1834_183465

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem real_roots_of_quadratic (m : ℝ) : (∃ x : ℝ, x^2 - x - m = 0) ↔ m ≥ -1/4 := by
  sorry

end real_roots_of_quadratic_l1834_183465
