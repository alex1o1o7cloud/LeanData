import Mathlib

namespace NUMINAMATH_GPT_no_real_solutions_for_m_l2409_240933

theorem no_real_solutions_for_m (m : ℝ) :
  ∃! m, (4 * m + 2) ^ 2 - 4 * m = 0 → false :=
by 
  sorry

end NUMINAMATH_GPT_no_real_solutions_for_m_l2409_240933


namespace NUMINAMATH_GPT_gcf_36_54_81_l2409_240923

theorem gcf_36_54_81 : Nat.gcd (Nat.gcd 36 54) 81 = 9 :=
by
  -- The theorem states that the greatest common factor of 36, 54, and 81 is 9.
  sorry

end NUMINAMATH_GPT_gcf_36_54_81_l2409_240923


namespace NUMINAMATH_GPT_sum_of_readings_ammeters_l2409_240956

variables (I1 I2 I3 I4 I5 : ℝ)

noncomputable def sum_of_ammeters (I1 I2 I3 I4 I5 : ℝ) : ℝ :=
  I1 + I2 + I3 + I4 + I5

theorem sum_of_readings_ammeters :
  I1 = 2 ∧ I2 = I1 ∧ I3 = 2 * I1 ∧ I5 = I3 + I1 ∧ I4 = (5 / 3) * I5 →
  sum_of_ammeters I1 I2 I3 I4 I5 = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_readings_ammeters_l2409_240956


namespace NUMINAMATH_GPT_ellipse_x_intersection_l2409_240961

theorem ellipse_x_intersection 
  (F₁ F₂ : ℝ × ℝ)
  (origin : ℝ × ℝ)
  (x_intersect : ℝ × ℝ)
  (h₁ : F₁ = (0, 3))
  (h₂ : F₂ = (4, 0))
  (h₃ : origin = (0, 0))
  (h₄ : ∀ P : ℝ × ℝ, (dist P F₁ + dist P F₂ = 7) ↔ (P = origin ∨ P = x_intersect))
  : x_intersect = (56 / 11, 0) := sorry

end NUMINAMATH_GPT_ellipse_x_intersection_l2409_240961


namespace NUMINAMATH_GPT_max_value_of_expression_l2409_240963

theorem max_value_of_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 3) :
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 12 := 
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2409_240963


namespace NUMINAMATH_GPT_determine_a_l2409_240901

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 / (3 ^ x + 1)) - a

theorem determine_a (a : ℝ) :
  (∀ x : ℝ, f a (-x) = -f a x) ↔ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l2409_240901


namespace NUMINAMATH_GPT_evaluate_expression_l2409_240964

variable (x y : ℝ)

theorem evaluate_expression :
  (1 + x^2 + y^3) * (1 - x^3 - y^3) = 1 + x^2 - x^3 - y^3 - x^5 - x^2 * y^3 - x^3 * y^3 - y^6 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2409_240964


namespace NUMINAMATH_GPT_num_sets_of_consecutive_integers_sum_to_30_l2409_240914

theorem num_sets_of_consecutive_integers_sum_to_30 : 
  let S_n (n a : ℕ) := (n * (2 * a + n - 1)) / 2 
  ∃! (s : ℕ), s = 3 ∧ ∀ n, n ≥ 2 → ∃ a, S_n n a = 30 :=
by
  sorry

end NUMINAMATH_GPT_num_sets_of_consecutive_integers_sum_to_30_l2409_240914


namespace NUMINAMATH_GPT_inequality_solution_sets_l2409_240921

theorem inequality_solution_sets (a : ℝ) (h : a > 1) :
  ∀ x : ℝ, ((a = 2 → (x ≠ 1 → (a-1)*x*x - a*x + 1 > 0)) ∧
            (1 < a ∧ a < 2 → (x < 1 ∨ x > 1/(a-1) → (a-1)*x*x - a*x + 1 > 0)) ∧
            (a > 2 → (x < 1/(a-1) ∨ x > 1 → (a-1)*x*x - a*x + 1 > 0))) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_sets_l2409_240921


namespace NUMINAMATH_GPT_inequality_proof_l2409_240934

theorem inequality_proof (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z ≥ 1) : 
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2409_240934


namespace NUMINAMATH_GPT_find_number_l2409_240903

theorem find_number (x k : ℕ) (h₁ : x / k = 4) (h₂ : k = 6) : x = 24 := by
  sorry

end NUMINAMATH_GPT_find_number_l2409_240903


namespace NUMINAMATH_GPT_inv_88_mod_89_l2409_240997

theorem inv_88_mod_89 : (88 * 88) % 89 = 1 := by
  sorry

end NUMINAMATH_GPT_inv_88_mod_89_l2409_240997


namespace NUMINAMATH_GPT_hoseok_add_8_l2409_240977

theorem hoseok_add_8 (x : ℕ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end NUMINAMATH_GPT_hoseok_add_8_l2409_240977


namespace NUMINAMATH_GPT_liam_drinks_17_glasses_l2409_240957

def minutes_in_hours (h : ℕ) : ℕ := h * 60

def total_time_in_minutes (hours : ℕ) (extra_minutes : ℕ) : ℕ := 
  minutes_in_hours hours + extra_minutes

def rate_of_drinking (drink_interval : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / drink_interval

theorem liam_drinks_17_glasses : 
  rate_of_drinking 20 (total_time_in_minutes 5 40) = 17 :=
by
  sorry

end NUMINAMATH_GPT_liam_drinks_17_glasses_l2409_240957


namespace NUMINAMATH_GPT_minimum_value_of_f_l2409_240919

noncomputable def f (a b x : ℝ) := (a * x + b) / (x^2 + 4)

theorem minimum_value_of_f (a b : ℝ) (h1 : f a b (-1) = 1)
  (h2 : (deriv (f a b)) (-1) = 0) : 
  ∃ (x : ℝ), f a b x = -1 / 4 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_f_l2409_240919


namespace NUMINAMATH_GPT_fraction_identity_l2409_240926

theorem fraction_identity (a b : ℝ) (h : a ≠ b) (h₁ : (a + b) / (a - b) = 3) : a / b = 2 := by
  sorry

end NUMINAMATH_GPT_fraction_identity_l2409_240926


namespace NUMINAMATH_GPT_hamburgers_left_over_l2409_240912

theorem hamburgers_left_over (total_hamburgers served_hamburgers : ℕ) (h1 : total_hamburgers = 9) (h2 : served_hamburgers = 3) :
    total_hamburgers - served_hamburgers = 6 := by
  sorry

end NUMINAMATH_GPT_hamburgers_left_over_l2409_240912


namespace NUMINAMATH_GPT_specialist_time_l2409_240980

def hospital_bed_charge (days : ℕ) (rate : ℕ) : ℕ := days * rate

def total_known_charges (bed_charge : ℕ) (ambulance_charge : ℕ) : ℕ := bed_charge + ambulance_charge

def specialist_minutes (total_bill : ℕ) (known_charges : ℕ) (spec_rate_per_hour : ℕ) : ℕ := 
  ((total_bill - known_charges) / spec_rate_per_hour) * 60 / 2

theorem specialist_time (days : ℕ) (bed_rate : ℕ) (ambulance_charge : ℕ) (spec_rate_per_hour : ℕ) 
(total_bill : ℕ) (known_charges := total_known_charges (hospital_bed_charge days bed_rate) ambulance_charge)
(hospital_days := 3) (bed_charge_per_day := 900) (specialist_rate := 250) 
(ambulance_cost := 1800) (total_cost := 4625) :
  specialist_minutes total_cost known_charges specialist_rate = 15 :=
sorry

end NUMINAMATH_GPT_specialist_time_l2409_240980


namespace NUMINAMATH_GPT_vector_subtraction_identity_l2409_240947

variables (a b : ℝ)

theorem vector_subtraction_identity (a b : ℝ) :
  ((1 / 2) * a - b) - ((3 / 2) * a - 2 * b) = b - a :=
by
  sorry

end NUMINAMATH_GPT_vector_subtraction_identity_l2409_240947


namespace NUMINAMATH_GPT_calculate_delta_nabla_l2409_240939

-- Define the operations Δ and ∇
def delta (a b : ℤ) : ℤ := 3 * a + 2 * b
def nabla (a b : ℤ) : ℤ := 2 * a + 3 * b

-- Formalize the theorem
theorem calculate_delta_nabla : delta 3 (nabla 2 1) = 23 := 
by 
  -- Placeholder for proof, not required by the question
  sorry

end NUMINAMATH_GPT_calculate_delta_nabla_l2409_240939


namespace NUMINAMATH_GPT_initial_average_age_of_students_l2409_240996

theorem initial_average_age_of_students 
(A : ℕ) 
(h1 : 23 * A + 46 = (A + 1) * 24) : 
  A = 22 :=
by
  sorry

end NUMINAMATH_GPT_initial_average_age_of_students_l2409_240996


namespace NUMINAMATH_GPT_number_of_true_propositions_l2409_240955

theorem number_of_true_propositions : 
  (∃ x y : ℝ, (x * y = 1) ↔ (x = y⁻¹ ∨ y = x⁻¹)) ∧
  (¬(∀ x : ℝ, (x > -3) → x^2 - x - 6 ≤ 0)) ∧
  (¬(∀ a b : ℝ, (a > b) → (a^2 < b^2))) ∧
  (¬(∀ x : ℝ, (x - 1/x > 0) → (x > -1))) →
  True := by
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_l2409_240955


namespace NUMINAMATH_GPT_flowers_in_each_basket_l2409_240918

theorem flowers_in_each_basket
  (plants_per_daughter : ℕ)
  (num_daughters : ℕ)
  (grown_flowers : ℕ)
  (died_flowers : ℕ)
  (num_baskets : ℕ)
  (h1 : plants_per_daughter = 5)
  (h2 : num_daughters = 2)
  (h3 : grown_flowers = 20)
  (h4 : died_flowers = 10)
  (h5 : num_baskets = 5) :
  (plants_per_daughter * num_daughters + grown_flowers - died_flowers) / num_baskets = 4 :=
by
  sorry

end NUMINAMATH_GPT_flowers_in_each_basket_l2409_240918


namespace NUMINAMATH_GPT_ice_cream_volume_l2409_240908

theorem ice_cream_volume (r_cone h_cone r_hemisphere : ℝ) (h1 : r_cone = 3) (h2 : h_cone = 10) (h3 : r_hemisphere = 5) :
  (1 / 3 * π * r_cone^2 * h_cone + 2 / 3 * π * r_hemisphere^3) = (520 / 3) * π :=
by 
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_ice_cream_volume_l2409_240908


namespace NUMINAMATH_GPT_find_z_l2409_240916

theorem find_z (x y z : ℚ) (hx : x = 11) (hy : y = -8) (h : 2 * x - 3 * z = 5 * y) :
  z = 62 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_z_l2409_240916


namespace NUMINAMATH_GPT_find_a_l2409_240944

noncomputable def csc (x : ℝ) : ℝ := 1 / (Real.sin x)

theorem find_a (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : a * csc (b * (Real.pi / 6) + c) = 3) : a = 3 := 
sorry

end NUMINAMATH_GPT_find_a_l2409_240944


namespace NUMINAMATH_GPT_customer_paid_amount_l2409_240952

theorem customer_paid_amount (O : ℕ) (D : ℕ) (P : ℕ) (hO : O = 90) (hD : D = 20) (hP : P = O - D) : P = 70 :=
sorry

end NUMINAMATH_GPT_customer_paid_amount_l2409_240952


namespace NUMINAMATH_GPT_tires_should_be_swapped_l2409_240935

-- Define the conditions
def front_wear_out_distance : ℝ := 25000
def rear_wear_out_distance : ℝ := 15000

-- Define the distance to swap tires
def swap_distance : ℝ := 9375

-- Theorem statement
theorem tires_should_be_swapped :
  -- The distance for both tires to wear out should be the same
  swap_distance + (front_wear_out_distance - swap_distance) * (rear_wear_out_distance / front_wear_out_distance) = rear_wear_out_distance :=
sorry

end NUMINAMATH_GPT_tires_should_be_swapped_l2409_240935


namespace NUMINAMATH_GPT_part_whole_ratio_l2409_240928

theorem part_whole_ratio (N x : ℕ) (hN : N = 160) (hx : x + 4 = N / 4 - 4) :
  x / N = 1 / 5 :=
  sorry

end NUMINAMATH_GPT_part_whole_ratio_l2409_240928


namespace NUMINAMATH_GPT_mike_passing_percentage_l2409_240999

theorem mike_passing_percentage (mike_score shortfall max_marks : ℝ)
  (h_mike_score : mike_score = 212)
  (h_shortfall : shortfall = 16)
  (h_max_marks : max_marks = 760) :
  (mike_score + shortfall) / max_marks * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_mike_passing_percentage_l2409_240999


namespace NUMINAMATH_GPT_area_enclosed_by_curve_and_line_l2409_240982

theorem area_enclosed_by_curve_and_line :
  let f := fun x : ℝ => x^2 + 2
  let g := fun x : ℝ => 3 * x
  let A := ∫ x in (0 : ℝ)..1, (f x - g x) + ∫ x in (1 : ℝ)..2, (g x - f x)
  A = 1 := by
    sorry

end NUMINAMATH_GPT_area_enclosed_by_curve_and_line_l2409_240982


namespace NUMINAMATH_GPT_equation_of_line_through_point_with_equal_intercepts_l2409_240937

-- Define a structure for a 2D point
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the problem-specific points and conditions
def A : Point := {x := 4, y := -1}

-- Define the conditions and the theorem to be proven
theorem equation_of_line_through_point_with_equal_intercepts
  (p : Point)
  (h : p = A) : 
  ∃ (a : ℝ), a ≠ 0 → (∀ (a : ℝ), ((∀ (b : ℝ), b = a → b ≠ 0 → x + y - a = 0)) ∨ (x + 4 * y = 0)) :=
sorry

end NUMINAMATH_GPT_equation_of_line_through_point_with_equal_intercepts_l2409_240937


namespace NUMINAMATH_GPT_natalies_diaries_l2409_240945

theorem natalies_diaries : 
  ∀ (initial_diaries : ℕ) (tripled_diaries : ℕ) (total_diaries : ℕ) (lost_diaries : ℕ) (remaining_diaries : ℕ),
  initial_diaries = 15 →
  tripled_diaries = 3 * initial_diaries →
  total_diaries = initial_diaries + tripled_diaries →
  lost_diaries = 3 * total_diaries / 5 →
  remaining_diaries = total_diaries - lost_diaries →
  remaining_diaries = 24 :=
by
  intros initial_diaries tripled_diaries total_diaries lost_diaries remaining_diaries
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_natalies_diaries_l2409_240945


namespace NUMINAMATH_GPT_sum_of_cubes_l2409_240920

open Real

theorem sum_of_cubes (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
(h_eq : (a^3 + 6) / a = (b^3 + 6) / b ∧ (b^3 + 6) / b = (c^3 + 6) / c) :
  a^3 + b^3 + c^3 = -18 := 
by sorry

end NUMINAMATH_GPT_sum_of_cubes_l2409_240920


namespace NUMINAMATH_GPT_find_n_l2409_240962

theorem find_n (n : ℕ) (h : 20 * n = Nat.factorial (n - 1)) : n = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_n_l2409_240962


namespace NUMINAMATH_GPT_factor_3m2n_12mn_12n_factor_abx2_4y2ba_calculate_result_l2409_240902

-- Proof 1: Factorize 3m^2 n - 12mn + 12n
theorem factor_3m2n_12mn_12n (m n : ℤ) : 3 * m^2 * n - 12 * m * n + 12 * n = 3 * n * (m - 2)^2 :=
by sorry

-- Proof 2: Factorize (a-b)x^2 + 4y^2(b-a)
theorem factor_abx2_4y2ba (a b x y : ℤ) : (a - b) * x^2 + 4 * y^2 * (b - a) = (a - b) * (x + 2 * y) * (x - 2 * y) :=
by sorry

-- Proof 3: Calculate 2023 * 51^2 - 2023 * 49^2
theorem calculate_result : 2023 * 51^2 - 2023 * 49^2 = 404600 :=
by sorry

end NUMINAMATH_GPT_factor_3m2n_12mn_12n_factor_abx2_4y2ba_calculate_result_l2409_240902


namespace NUMINAMATH_GPT_sum_of_numbers_l2409_240958

theorem sum_of_numbers : 4.75 + 0.303 + 0.432 = 5.485 :=
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l2409_240958


namespace NUMINAMATH_GPT_selling_price_percentage_l2409_240978

-- Definitions for conditions
def ratio_cara_janet_jerry (c j je : ℕ) : Prop := 4 * (c + j + je) = 4 * c + 5 * j + 6 * je
def total_money (c j je total : ℕ) : Prop := c + j + je = total
def combined_loss (c j loss : ℕ) : Prop := c + j - loss = 36

-- The theorem statement to be proven
theorem selling_price_percentage (c j je total loss : ℕ) (h1 : ratio_cara_janet_jerry c j je) (h2 : total_money c j je total) (h3 : combined_loss c j loss)
    (h4 : total = 75) (h5 : loss = 9) : (36 * 100 / (c + j) = 80) := by
  sorry

end NUMINAMATH_GPT_selling_price_percentage_l2409_240978


namespace NUMINAMATH_GPT_donna_babysitting_hours_l2409_240951

theorem donna_babysitting_hours 
  (total_earnings : ℝ)
  (dog_walking_hours : ℝ)
  (dog_walking_rate : ℝ)
  (dog_walking_days : ℝ)
  (card_shop_hours : ℝ)
  (card_shop_rate : ℝ)
  (card_shop_days : ℝ)
  (babysitting_rate : ℝ)
  (days : ℝ)
  (total_dog_walking_earnings : ℝ := dog_walking_hours * dog_walking_rate * dog_walking_days)
  (total_card_shop_earnings : ℝ := card_shop_hours * card_shop_rate * card_shop_days)
  (total_earnings_dog_card : ℝ := total_dog_walking_earnings + total_card_shop_earnings)
  (babysitting_hours : ℝ := (total_earnings - total_earnings_dog_card) / babysitting_rate) :
  total_earnings = 305 → dog_walking_hours = 2 → dog_walking_rate = 10 → dog_walking_days = 5 →
  card_shop_hours = 2 → card_shop_rate = 12.5 → card_shop_days = 5 →
  babysitting_rate = 10 → babysitting_hours = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_donna_babysitting_hours_l2409_240951


namespace NUMINAMATH_GPT_trigonometric_order_l2409_240948

theorem trigonometric_order :
  (Real.sin 2 > Real.sin 1) ∧
  (Real.sin 1 > Real.sin 3) ∧
  (Real.sin 3 > Real.sin 4) := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_order_l2409_240948


namespace NUMINAMATH_GPT_total_jellybeans_l2409_240987

-- Define the conditions
def a := 3 * 12       -- Caleb's jellybeans
def b := a / 2        -- Sophie's jellybeans

-- Define the goal
def total := a + b    -- Total jellybeans

-- The theorem statement
theorem total_jellybeans : total = 54 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_total_jellybeans_l2409_240987


namespace NUMINAMATH_GPT_problem_l2409_240931

noncomputable def f : ℝ → ℝ := sorry

theorem problem :
  (∀ x : ℝ, f (x) + f (x + 2) = 0) →
  (f (1) = -2) →
  (f (2019) + f (2018) = 2) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_problem_l2409_240931


namespace NUMINAMATH_GPT_sets_are_equal_l2409_240953

-- Defining sets A and B as per the given conditions
def setA : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 1}
def setB : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4 * b + 5}

-- Proving that set A is equal to set B
theorem sets_are_equal : setA = setB :=
by
  sorry

end NUMINAMATH_GPT_sets_are_equal_l2409_240953


namespace NUMINAMATH_GPT_trapezoid_other_side_length_l2409_240905

theorem trapezoid_other_side_length (a h : ℕ) (A : ℕ) (b : ℕ) : 
  a = 20 → h = 13 → A = 247 → (1/2:ℚ) * (a + b) * h = A → b = 18 :=
by 
  intros h1 h2 h3 h4 
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_trapezoid_other_side_length_l2409_240905


namespace NUMINAMATH_GPT_find_k_l2409_240924

theorem find_k (a b c k : ℤ) (g : ℤ → ℤ)
  (h₁ : g 1 = 0)
  (h₂ : 10 < g 5 ∧ g 5 < 20)
  (h₃ : 30 < g 6 ∧ g 6 < 40)
  (h₄ : 3000 * k < g 100 ∧ g 100 < 3000 * (k + 1))
  (h_g : ∀ x, g x = a * x^2 + b * x + c) :
  k = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2409_240924


namespace NUMINAMATH_GPT_probability_at_least_one_woman_selected_l2409_240943

theorem probability_at_least_one_woman_selected:
  let men := 10
  let women := 5
  let totalPeople := men + women
  let totalSelections := Nat.choose totalPeople 4
  let menSelections := Nat.choose men 4
  let noWomenProbability := (menSelections : ℚ) / (totalSelections : ℚ)
  let atLeastOneWomanProbability := 1 - noWomenProbability
  atLeastOneWomanProbability = 11 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_woman_selected_l2409_240943


namespace NUMINAMATH_GPT_rohit_distance_from_start_l2409_240938

-- Define Rohit's movements
def rohit_walked_south (d: ℕ) : ℕ := d
def rohit_turned_left_walked_east (d: ℕ) : ℕ := d
def rohit_turned_left_walked_north (d: ℕ) : ℕ := d
def rohit_turned_right_walked_east (d: ℕ) : ℕ := d

-- Rohit's total movement in east direction
def total_distance_moved_east (d1 d2 : ℕ) : ℕ :=
  rohit_turned_left_walked_east d1 + rohit_turned_right_walked_east d2

-- Prove the distance from the starting point is 35 meters
theorem rohit_distance_from_start : 
  total_distance_moved_east 20 15 = 35 :=
by
  sorry

end NUMINAMATH_GPT_rohit_distance_from_start_l2409_240938


namespace NUMINAMATH_GPT_find_x_l2409_240976

theorem find_x (A B D : ℝ) (BC CD x : ℝ) 
  (hA : A = 60) (hB : B = 90) (hD : D = 90) 
  (hBC : BC = 2) (hCD : CD = 3) 
  (hResult : x = 8 / Real.sqrt 3) : 
  AB = x :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2409_240976


namespace NUMINAMATH_GPT_total_ingredients_l2409_240967

theorem total_ingredients (b f s : ℕ) (h_ratio : 2 * f = 5 * f) (h_flour : f = 15) : b + f + s = 30 :=
by 
  sorry

end NUMINAMATH_GPT_total_ingredients_l2409_240967


namespace NUMINAMATH_GPT_median_perimeter_ratio_l2409_240966

variables {A B C : Type*}
variables (AB BC AC AD BE CF : ℝ)
variable (l m : ℝ)

noncomputable def triangle_perimeter (AB BC AC : ℝ) : ℝ := AB + BC + AC
noncomputable def triangle_median_sum (AD BE CF : ℝ) : ℝ := AD + BE + CF

theorem median_perimeter_ratio (h1 : l = triangle_perimeter AB BC AC)
                                (h2 : m = triangle_median_sum AD BE CF) :
  m / l > 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_median_perimeter_ratio_l2409_240966


namespace NUMINAMATH_GPT_average_price_of_pig_l2409_240983

theorem average_price_of_pig :
  ∀ (total_cost : ℕ) (num_pigs num_hens : ℕ) (avg_hen_price avg_pig_price : ℕ),
    total_cost = 2100 →
    num_pigs = 5 →
    num_hens = 15 →
    avg_hen_price = 30 →
    avg_pig_price * num_pigs + avg_hen_price * num_hens = total_cost →
    avg_pig_price = 330 :=
by
  intros total_cost num_pigs num_hens avg_hen_price avg_pig_price
  intros h_total_cost h_num_pigs h_num_hens h_avg_hen_price h_eq
  rw [h_total_cost, h_num_pigs, h_num_hens, h_avg_hen_price] at h_eq
  sorry

end NUMINAMATH_GPT_average_price_of_pig_l2409_240983


namespace NUMINAMATH_GPT_complex_exponentiation_l2409_240970

-- Define the imaginary unit i where i^2 = -1.
def i : ℂ := Complex.I

-- Lean statement for proving the problem.
theorem complex_exponentiation :
  (1 + i)^6 = -8 * i :=
sorry

end NUMINAMATH_GPT_complex_exponentiation_l2409_240970


namespace NUMINAMATH_GPT_fraction_problem_l2409_240930

theorem fraction_problem (x : ℝ) (h : (3 / 4) * (1 / 2) * x * 5000 = 750.0000000000001) : 
  x = 0.4 :=
sorry

end NUMINAMATH_GPT_fraction_problem_l2409_240930


namespace NUMINAMATH_GPT_total_raisins_l2409_240910

noncomputable def yellow_raisins : ℝ := 0.3
noncomputable def black_raisins : ℝ := 0.4
noncomputable def red_raisins : ℝ := 0.5

theorem total_raisins : yellow_raisins + black_raisins + red_raisins = 1.2 := by
  sorry

end NUMINAMATH_GPT_total_raisins_l2409_240910


namespace NUMINAMATH_GPT_expected_number_of_letters_in_mailbox_A_l2409_240990

def prob_xi_0 : ℚ := 4 / 9
def prob_xi_1 : ℚ := 4 / 9
def prob_xi_2 : ℚ := 1 / 9

def expected_xi := 0 * prob_xi_0 + 1 * prob_xi_1 + 2 * prob_xi_2

theorem expected_number_of_letters_in_mailbox_A :
  expected_xi = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_expected_number_of_letters_in_mailbox_A_l2409_240990


namespace NUMINAMATH_GPT_max_value_of_3cosx_minus_sinx_l2409_240911

noncomputable def max_cosine_expression : ℝ :=
  Real.sqrt 10

theorem max_value_of_3cosx_minus_sinx : 
  ∃ x : ℝ, ∀ x : ℝ, 3 * Real.cos x - Real.sin x ≤ Real.sqrt 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_max_value_of_3cosx_minus_sinx_l2409_240911


namespace NUMINAMATH_GPT_value_of_b_l2409_240949

noncomputable def function_bounds := 
  ∃ (k b : ℝ), (∀ (x : ℝ), (-3 ≤ x ∧ x ≤ 1) → (-1 ≤ k * x + b ∧ k * x + b ≤ 8)) ∧ (b = 5 / 4 ∨ b = 23 / 4)

theorem value_of_b : function_bounds :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_l2409_240949


namespace NUMINAMATH_GPT_balls_initial_count_90_l2409_240946

theorem balls_initial_count_90 (n : ℕ) (total_initial_balls : ℕ)
  (initial_green_balls : ℕ := 3 * n)
  (initial_yellow_balls : ℕ := 7 * n)
  (remaining_green_balls : ℕ := initial_green_balls - 9)
  (remaining_yellow_balls : ℕ := initial_yellow_balls - 9)
  (h_ratio_1 : initial_green_balls = 3 * n)
  (h_ratio_2 : initial_yellow_balls = 7 * n)
  (h_ratio_3 : remaining_green_balls * 3 = remaining_yellow_balls * 1)
  (h_total : total_initial_balls = initial_green_balls + initial_yellow_balls)
  : total_initial_balls = 90 := 
by
  sorry

end NUMINAMATH_GPT_balls_initial_count_90_l2409_240946


namespace NUMINAMATH_GPT_product_equals_9_l2409_240917

theorem product_equals_9 :
  (1 + (1 / 1)) * (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * 
  (1 + (1 / 5)) * (1 + (1 / 6)) * (1 + (1 / 7)) * (1 + (1 / 8)) = 9 := 
by
  sorry

end NUMINAMATH_GPT_product_equals_9_l2409_240917


namespace NUMINAMATH_GPT_functionMachine_output_l2409_240927

-- Define the function machine according to the specified conditions
def functionMachine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 30 then step1 - 4 else step1
  let step3 := if step2 <= 20 then step2 + 8 else step2 - 5
  step3

-- Statement: Prove that the functionMachine applied to 10 yields 25
theorem functionMachine_output : functionMachine 10 = 25 :=
  by
    sorry

end NUMINAMATH_GPT_functionMachine_output_l2409_240927


namespace NUMINAMATH_GPT_arccos_neg1_l2409_240991

theorem arccos_neg1 : Real.arccos (-1) = Real.pi := 
sorry

end NUMINAMATH_GPT_arccos_neg1_l2409_240991


namespace NUMINAMATH_GPT_fred_cantaloupes_l2409_240950

def num_cantaloupes_K : ℕ := 29
def num_cantaloupes_J : ℕ := 20
def total_cantaloupes : ℕ := 65

theorem fred_cantaloupes : ∃ F : ℕ, num_cantaloupes_K + num_cantaloupes_J + F = total_cantaloupes ∧ F = 16 :=
by
  sorry

end NUMINAMATH_GPT_fred_cantaloupes_l2409_240950


namespace NUMINAMATH_GPT_find_unknown_number_l2409_240985

theorem find_unknown_number
  (n : ℕ)
  (h_lcm : Nat.lcm n 1491 = 5964) :
  n = 4 :=
sorry

end NUMINAMATH_GPT_find_unknown_number_l2409_240985


namespace NUMINAMATH_GPT_sum_of_tens_and_units_of_product_is_zero_l2409_240971

-- Define the repeating patterns used to create the 999-digit numbers
def pattern1 : ℕ := 400
def pattern2 : ℕ := 606

-- Function to construct a 999-digit number by repeating a 3-digit pattern 333 times
def repeat_pattern (pat : ℕ) (times : ℕ) : ℕ := pat * (10 ^ (3 * times - 3))

-- Define the two 999-digit numbers
def num1 : ℕ := repeat_pattern pattern1 333
def num2 : ℕ := repeat_pattern pattern2 333

-- Function to compute the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Function to compute the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Define the product of the two numbers
def product : ℕ := num1 * num2

-- Function to compute the sum of the tens and units digits of a number
def sum_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

-- The statement to be proven
theorem sum_of_tens_and_units_of_product_is_zero :
  sum_digits product = 0 := 
sorry -- Proof steps are omitted

end NUMINAMATH_GPT_sum_of_tens_and_units_of_product_is_zero_l2409_240971


namespace NUMINAMATH_GPT_natural_numbers_solution_l2409_240981

theorem natural_numbers_solution (a : ℕ) :
  ∃ k n : ℕ, k = 3 * a - 2 ∧ n = 2 * a - 1 ∧ (7 * k + 15 * n - 1) % (3 * k + 4 * n) = 0 :=
sorry

end NUMINAMATH_GPT_natural_numbers_solution_l2409_240981


namespace NUMINAMATH_GPT_surface_area_ratio_volume_ratio_l2409_240959

-- Given conditions
def tetrahedron_surface_area (S : ℝ) : ℝ := 4 * S
def tetrahedron_volume (V : ℝ) : ℝ := 27 * V
def polyhedron_G_surface_area (S : ℝ) : ℝ := 28 * S
def polyhedron_G_volume (V : ℝ) : ℝ := 23 * V

-- Statements to prove
theorem surface_area_ratio (S : ℝ) (h1 : S > 0) :
  tetrahedron_surface_area S / polyhedron_G_surface_area S = 9 / 7 := by
  simp [tetrahedron_surface_area, polyhedron_G_surface_area]
  sorry

theorem volume_ratio (V : ℝ) (h1 : V > 0) :
  tetrahedron_volume V / polyhedron_G_volume V = 27 / 23 := by
  simp [tetrahedron_volume, polyhedron_G_volume]
  sorry

end NUMINAMATH_GPT_surface_area_ratio_volume_ratio_l2409_240959


namespace NUMINAMATH_GPT_interval_of_monotonic_increase_l2409_240936

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem interval_of_monotonic_increase :
  (∃ α : ℝ, power_function α 2 = 4) →
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → power_function 2 x ≤ power_function 2 y) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_interval_of_monotonic_increase_l2409_240936


namespace NUMINAMATH_GPT_count_eligible_three_digit_numbers_l2409_240932

def is_eligible_digit (d : Nat) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem count_eligible_three_digit_numbers : 
  (∃ n : Nat, 100 ≤ n ∧ n < 1000 ∧
  (∀ d : Nat, d ∈ [n / 100, (n / 10) % 10, n % 10] → is_eligible_digit d)) →
  ∃ count : Nat, count = 343 :=
by
  sorry

end NUMINAMATH_GPT_count_eligible_three_digit_numbers_l2409_240932


namespace NUMINAMATH_GPT_find_number_l2409_240994

theorem find_number (a : ℤ) (h : a - a + 99 * (a - 99) = 19802) : a = 299 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l2409_240994


namespace NUMINAMATH_GPT_max_rubles_l2409_240974

theorem max_rubles (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2099) :
  (∃ k, n = 99 * k) → 
  31 ≤
    (if n % 1 = 0 then 1 else 0) +
    (if n % 3 = 0 then 3 else 0) +
    (if n % 5 = 0 then 5 else 0) +
    (if n % 7 = 0 then 7 else 0) +
    (if n % 9 = 0 then 9 else 0) +
    (if n % 11 = 0 then 11 else 0) :=
sorry

end NUMINAMATH_GPT_max_rubles_l2409_240974


namespace NUMINAMATH_GPT_max_neg_square_in_interval_l2409_240960

variable (f : ℝ → ℝ)

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, 0 < x → x < y → f x < f y

noncomputable def neg_square_val (x : ℝ) : ℝ :=
  - (f x) ^ 2

theorem max_neg_square_in_interval : 
  (∀ x_1 x_2 : ℝ, f (x_1 + x_2) = f x_1 + f x_2) →
  f 1 = 2 →
  is_increasing f →
  (∀ x : ℝ, f (-x) = - f x) →
  ∃ b ∈ (Set.Icc (-3) (-2)), 
  ∀ x ∈ (Set.Icc (-3) (-2)), neg_square_val f x ≤ neg_square_val f b ∧ neg_square_val f b = -16 := 
sorry

end NUMINAMATH_GPT_max_neg_square_in_interval_l2409_240960


namespace NUMINAMATH_GPT_freezer_temperature_l2409_240969

theorem freezer_temperature 
  (refrigeration_temp : ℝ)
  (freezer_temp_diff : ℝ)
  (h1 : refrigeration_temp = 4)
  (h2 : freezer_temp_diff = 22)
  : (refrigeration_temp - freezer_temp_diff) = -18 :=
by 
  sorry

end NUMINAMATH_GPT_freezer_temperature_l2409_240969


namespace NUMINAMATH_GPT_koby_sparklers_correct_l2409_240984

-- Define the number of sparklers in each of Koby's boxes as a variable
variable (S : ℕ)

-- Specify the conditions
def koby_sparklers : ℕ := 2 * S
def koby_whistlers : ℕ := 2 * 5
def cherie_sparklers : ℕ := 8
def cherie_whistlers : ℕ := 9
def total_fireworks : ℕ := koby_sparklers S + koby_whistlers + cherie_sparklers + cherie_whistlers

-- The theorem to prove that the number of sparklers in each of Koby's boxes is 3
theorem koby_sparklers_correct : total_fireworks S = 33 → S = 3 := by
  sorry

end NUMINAMATH_GPT_koby_sparklers_correct_l2409_240984


namespace NUMINAMATH_GPT_number_of_cases_ordered_in_may_l2409_240965

noncomputable def cases_ordered_in_may (ordered_in_april_cases : ℕ) (bottles_per_case : ℕ) (total_bottles : ℕ) : ℕ :=
  let bottles_in_april := ordered_in_april_cases * bottles_per_case
  let bottles_in_may := total_bottles - bottles_in_april
  bottles_in_may / bottles_per_case

theorem number_of_cases_ordered_in_may :
  ∀ (ordered_in_april_cases bottles_per_case total_bottles : ℕ),
  ordered_in_april_cases = 20 →
  bottles_per_case = 20 →
  total_bottles = 1000 →
  cases_ordered_in_may ordered_in_april_cases bottles_per_case total_bottles = 30 := by
  intros ordered_in_april_cases bottles_per_case total_bottles ha hbp htt
  sorry

end NUMINAMATH_GPT_number_of_cases_ordered_in_may_l2409_240965


namespace NUMINAMATH_GPT_min_rounds_needed_l2409_240973

-- Defining the number of players
def num_players : ℕ := 10

-- Defining the number of matches each player plays per round
def matches_per_round (n : ℕ) : ℕ := n / 2

-- Defining the scoring system
def win_points : ℝ := 1
def draw_points : ℝ := 0.5
def loss_points : ℝ := 0

-- Defining the total number of rounds needed for a clear winner to emerge
def min_rounds_for_winner : ℕ := 7

-- Theorem stating the minimum number of rounds required
theorem min_rounds_needed :
  ∀ (n : ℕ), n = num_players → (∃ r : ℕ, r = min_rounds_for_winner) :=
by
  intros n hn
  existsi min_rounds_for_winner
  sorry

end NUMINAMATH_GPT_min_rounds_needed_l2409_240973


namespace NUMINAMATH_GPT_distance_between_front_contestants_l2409_240925

noncomputable def position_a (pd : ℝ) : ℝ := pd - 10
def position_b (pd : ℝ) : ℝ := pd - 40
def position_c (pd : ℝ) : ℝ := pd - 60
def position_d (pd : ℝ) : ℝ := pd

theorem distance_between_front_contestants (pd : ℝ):
  position_d pd - position_a pd = 10 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_front_contestants_l2409_240925


namespace NUMINAMATH_GPT_maximum_abc_827_l2409_240904

noncomputable def maximum_abc (a b c : ℝ) := (a * b * c)

theorem maximum_abc_827 (a b c : ℝ) 
  (h1: a > 0) 
  (h2: b > 0) 
  (h3: c > 0) 
  (h4: (a * b) + c = (a + c) * (b + c)) 
  (h5: a + b + c = 2) : 
  maximum_abc a b c = 8 / 27 := 
by 
  sorry

end NUMINAMATH_GPT_maximum_abc_827_l2409_240904


namespace NUMINAMATH_GPT_domain_expression_l2409_240954

-- Define the conditions for the domain of the expression
def valid_numerator (x : ℝ) : Prop := 3 * x - 6 ≥ 0
def valid_denominator (x : ℝ) : Prop := 7 - 2 * x > 0

-- Proof problem statement
theorem domain_expression (x : ℝ) : valid_numerator x ∧ valid_denominator x ↔ 2 ≤ x ∧ x < 3.5 :=
sorry

end NUMINAMATH_GPT_domain_expression_l2409_240954


namespace NUMINAMATH_GPT_expression_equality_l2409_240940

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : (1 / x + 1 / y) = 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_equality_l2409_240940


namespace NUMINAMATH_GPT_count_valid_third_sides_l2409_240942

-- Define the lengths of the two known sides of the triangle.
def side1 := 8
def side2 := 11

-- Define the condition for the third side using the triangle inequality theorem.
def valid_third_side (x : ℕ) : Prop :=
  (x + side1 > side2) ∧ (x + side2 > side1) ∧ (side1 + side2 > x)

-- Define a predicate to check if a side length is an integer in the range (3, 19).
def is_valid_integer_length (x : ℕ) : Prop :=
  3 < x ∧ x < 19

-- Define the specific integer count
def integer_third_side_count : ℕ :=
  Finset.card ((Finset.Ico 4 19) : Finset ℕ)

-- Declare our theorem, which we need to prove
theorem count_valid_third_sides : integer_third_side_count = 15 := by
  sorry

end NUMINAMATH_GPT_count_valid_third_sides_l2409_240942


namespace NUMINAMATH_GPT_difference_between_two_numbers_l2409_240979

theorem difference_between_two_numbers (a : ℕ) (b : ℕ)
  (h1 : a + b = 24300)
  (h2 : b = 100 * a) :
  b - a = 23760 :=
by {
  sorry
}

end NUMINAMATH_GPT_difference_between_two_numbers_l2409_240979


namespace NUMINAMATH_GPT_total_fish_count_l2409_240993

def number_of_tables : ℕ := 32
def fish_per_table : ℕ := 2
def additional_fish_table : ℕ := 1
def total_fish : ℕ := (number_of_tables * fish_per_table) + additional_fish_table

theorem total_fish_count : total_fish = 65 := by
  sorry

end NUMINAMATH_GPT_total_fish_count_l2409_240993


namespace NUMINAMATH_GPT_cafeteria_seats_taken_l2409_240992

def table1_count : ℕ := 10
def table1_seats : ℕ := 8
def table2_count : ℕ := 5
def table2_seats : ℕ := 12
def table3_count : ℕ := 5
def table3_seats : ℕ := 10
noncomputable def unseated_ratio1 : ℝ := 1/4
noncomputable def unseated_ratio2 : ℝ := 1/3
noncomputable def unseated_ratio3 : ℝ := 1/5

theorem cafeteria_seats_taken : 
  ((table1_count * table1_seats) - (unseated_ratio1 * (table1_count * table1_seats))) + 
  ((table2_count * table2_seats) - (unseated_ratio2 * (table2_count * table2_seats))) + 
  ((table3_count * table3_seats) - (unseated_ratio3 * (table3_count * table3_seats))) = 140 :=
by sorry

end NUMINAMATH_GPT_cafeteria_seats_taken_l2409_240992


namespace NUMINAMATH_GPT_distinct_roots_l2409_240989

noncomputable def roots (a b c : ℝ) := ((b^2 - 4 * a * c) ≥ 0) ∧ ((-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a) * Real.sqrt (b^2 - 4 * a * c)) ≠ (0 : ℝ)

theorem distinct_roots{ p q r s : ℝ } (h1 : p ≠ q) (h2 : p ≠ r) (h3 : p ≠ s) (h4 : q ≠ r) 
(h5 : q ≠ s) (h6 : r ≠ s)
(h_roots_1 : roots 1 (-12*p) (-13*q))
(h_roots_2 : roots 1 (-12*r) (-13*s)) : 
(p + q + r + s = 2028) := sorry

end NUMINAMATH_GPT_distinct_roots_l2409_240989


namespace NUMINAMATH_GPT_divides_5n_4n_iff_n_is_multiple_of_3_l2409_240995

theorem divides_5n_4n_iff_n_is_multiple_of_3 (n : ℕ) (h : n > 0) : 
  61 ∣ (5^n - 4^n) ↔ ∃ k : ℕ, n = 3 * k :=
by
  sorry

end NUMINAMATH_GPT_divides_5n_4n_iff_n_is_multiple_of_3_l2409_240995


namespace NUMINAMATH_GPT_julia_money_left_l2409_240986

def initial_money : ℕ := 40
def spent_on_game : ℕ := initial_money / 2
def money_left_after_game : ℕ := initial_money - spent_on_game
def spent_on_in_game_purchases : ℕ := money_left_after_game / 4
def final_money_left : ℕ := money_left_after_game - spent_on_in_game_purchases

theorem julia_money_left : final_money_left = 15 := by
  sorry

end NUMINAMATH_GPT_julia_money_left_l2409_240986


namespace NUMINAMATH_GPT_value_of_a_l2409_240929

theorem value_of_a (a b c d : ℕ) (h : (18^a) * (9^(4*a-1)) * (27^c) = (2^6) * (3^b) * (7^d)) : a = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2409_240929


namespace NUMINAMATH_GPT_bird_counts_remaining_l2409_240975

theorem bird_counts_remaining
  (peregrine_falcons pigeons crows sparrows : ℕ)
  (chicks_per_pigeon chicks_per_crow chicks_per_sparrow : ℕ)
  (peregrines_eat_pigeons_percent peregrines_eat_crows_percent peregrines_eat_sparrows_percent : ℝ)
  (initial_peregrine_falcons : peregrine_falcons = 12)
  (initial_pigeons : pigeons = 80)
  (initial_crows : crows = 25)
  (initial_sparrows : sparrows = 15)
  (chicks_per_pigeon_cond : chicks_per_pigeon = 8)
  (chicks_per_crow_cond : chicks_per_crow = 5)
  (chicks_per_sparrow_cond : chicks_per_sparrow = 3)
  (peregrines_eat_pigeons_percent_cond : peregrines_eat_pigeons_percent = 0.4)
  (peregrines_eat_crows_percent_cond : peregrines_eat_crows_percent = 0.25)
  (peregrines_eat_sparrows_percent_cond : peregrines_eat_sparrows_percent = 0.1)
  : 
  (peregrine_falcons = 12) ∧
  (pigeons = 48) ∧
  (crows = 19) ∧
  (sparrows = 14) :=
by
  sorry

end NUMINAMATH_GPT_bird_counts_remaining_l2409_240975


namespace NUMINAMATH_GPT_Gemma_ordered_pizzas_l2409_240900

-- Definitions of conditions
def pizza_cost : ℕ := 10
def tip : ℕ := 5
def paid_amount : ℕ := 50
def change : ℕ := 5
def total_spent : ℕ := paid_amount - change

-- Statement of the proof problem
theorem Gemma_ordered_pizzas : 
  ∃ (P : ℕ), pizza_cost * P + tip = total_spent ∧ P = 4 :=
sorry

end NUMINAMATH_GPT_Gemma_ordered_pizzas_l2409_240900


namespace NUMINAMATH_GPT_definite_integral_sin8_l2409_240909

-- Define the definite integral problem and the expected result in Lean.
theorem definite_integral_sin8:
  ∫ x in (Real.pi / 2)..Real.pi, (2^8 * (Real.sin x)^8) = 32 * Real.pi :=
  sorry

end NUMINAMATH_GPT_definite_integral_sin8_l2409_240909


namespace NUMINAMATH_GPT_ice_cream_scoops_l2409_240907

def scoops_of_ice_cream : ℕ := 1 -- single cone has 1 scoop

def scoops_double_cone : ℕ := 2 * scoops_of_ice_cream -- double cone has two times the scoops of a single cone

def scoops_banana_split : ℕ := 3 * scoops_of_ice_cream -- banana split has three times the scoops of a single cone

def scoops_waffle_bowl : ℕ := scoops_banana_split + 1 -- waffle bowl has one more scoop than banana split

def total_scoops : ℕ := scoops_of_ice_cream + scoops_double_cone + scoops_banana_split + scoops_waffle_bowl

theorem ice_cream_scoops : total_scoops = 10 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_scoops_l2409_240907


namespace NUMINAMATH_GPT_polygon_sides_l2409_240906

theorem polygon_sides (n : ℕ) (h1 : ∀ i < n, (n > 2) → (150 * n = (n - 2) * 180)) : n = 12 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_polygon_sides_l2409_240906


namespace NUMINAMATH_GPT_find_ab_l2409_240941

-- Define the polynomials involved
def poly1 (x : ℝ) (a b : ℝ) : ℝ := a * x^4 + b * x^2 + 1
def poly2 (x : ℝ) : ℝ := x^2 - x - 2

-- Define the roots of the second polynomial
def root1 : ℝ := 2
def root2 : ℝ := -1

-- State the theorem to prove
theorem find_ab (a b : ℝ) :
  poly1 root1 a b = 0 ∧ poly1 root2 a b = 0 → a = 1/4 ∧ b = -5/4 :=
by
  -- Skipping the proof here
  sorry

end NUMINAMATH_GPT_find_ab_l2409_240941


namespace NUMINAMATH_GPT_sum_of_homothety_coeffs_geq_4_l2409_240922

theorem sum_of_homothety_coeffs_geq_4 (a : ℕ → ℝ)
  (h_pos : ∀ i, 0 < a i)
  (h_less_one : ∀ i, a i < 1)
  (h_sum_cubes : ∑' i, (a i)^3 = 1) :
  (∑' i, a i) ≥ 4 := sorry

end NUMINAMATH_GPT_sum_of_homothety_coeffs_geq_4_l2409_240922


namespace NUMINAMATH_GPT_problem_solution_l2409_240968

variable {a b c d : ℝ}
variable (h_a : a = 4 * π / 3)
variable (h_b : b = 10 * π)
variable (h_c : c = 62)
variable (h_d : d = 30)

theorem problem_solution : (b * c) / (a * d) = 15.5 :=
by
  rw [h_a, h_b, h_c, h_d]
  -- Continued steps according to identified solution steps
  -- and arithmetic operations.
  sorry

end NUMINAMATH_GPT_problem_solution_l2409_240968


namespace NUMINAMATH_GPT_range_of_expression_l2409_240972

theorem range_of_expression (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a * b = 2) :
  (a^2 + b^2) / (a - b) ≤ -4 :=
sorry

end NUMINAMATH_GPT_range_of_expression_l2409_240972


namespace NUMINAMATH_GPT_students_behind_Yoongi_l2409_240913

theorem students_behind_Yoongi 
  (total_students : ℕ) 
  (position_Jungkook : ℕ) 
  (students_between : ℕ) 
  (position_Yoongi : ℕ) : 
  total_students = 20 → 
  position_Jungkook = 1 → 
  students_between = 5 → 
  position_Yoongi = position_Jungkook + students_between + 1 → 
  (total_students - position_Yoongi) = 13 :=
by
  sorry

end NUMINAMATH_GPT_students_behind_Yoongi_l2409_240913


namespace NUMINAMATH_GPT_taxi_fare_proportional_l2409_240915

theorem taxi_fare_proportional (fare_50 : ℝ) (distance_50 distance_70 : ℝ) (proportional : Prop) (h_fare_50 : fare_50 = 120) (h_distance_50 : distance_50 = 50) (h_distance_70 : distance_70 = 70) :
  fare_70 = 168 :=
by {
  sorry
}

end NUMINAMATH_GPT_taxi_fare_proportional_l2409_240915


namespace NUMINAMATH_GPT_probability_of_color_change_l2409_240998

theorem probability_of_color_change :
  let cycle_duration := 100
  let green_duration := 45
  let yellow_duration := 5
  let red_duration := 50
  let green_to_yellow_interval := 5
  let yellow_to_red_interval := 5
  let red_to_green_interval := 5
  let total_color_change_duration := green_to_yellow_interval + yellow_to_red_interval + red_to_green_interval
  let observation_probability := total_color_change_duration / cycle_duration
  observation_probability = 3 / 20 := by sorry

end NUMINAMATH_GPT_probability_of_color_change_l2409_240998


namespace NUMINAMATH_GPT_dads_strawberries_l2409_240988

variable (M D : ℕ)

theorem dads_strawberries (h1 : M + D = 22) (h2 : M = 36) (h3 : D ≤ 22) :
  D + 30 = 46 :=
by
  sorry

end NUMINAMATH_GPT_dads_strawberries_l2409_240988
