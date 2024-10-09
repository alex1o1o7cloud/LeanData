import Mathlib

namespace max_sum_of_squares_eq_7_l832_83204

theorem max_sum_of_squares_eq_7 :
  ∃ (x y : ℤ), (x^2 + y^2 = 25 ∧ x + y = 7) ∧
  (∀ x' y' : ℤ, (x'^2 + y'^2 = 25 → x' + y' ≤ 7)) := by
sorry

end max_sum_of_squares_eq_7_l832_83204


namespace square_tiles_count_l832_83291

theorem square_tiles_count (p s : ℕ) (h1 : p + s = 30) (h2 : 5 * p + 4 * s = 122) : s = 28 :=
sorry

end square_tiles_count_l832_83291


namespace geom_cos_sequence_l832_83255

open Real

theorem geom_cos_sequence (b : ℝ) (hb : 0 < b ∧ b < 360) (h : cos (2*b) / cos b = cos (3*b) / cos (2*b)) : b = 180 :=
by
  sorry

end geom_cos_sequence_l832_83255


namespace price_after_discounts_l832_83268

noncomputable def final_price (initial_price : ℝ) : ℝ :=
  let first_discount := initial_price * (1 - 0.10)
  let second_discount := first_discount * (1 - 0.20)
  second_discount

theorem price_after_discounts (initial_price : ℝ) (h : final_price initial_price = 174.99999999999997) : 
  final_price initial_price = 175 := 
by {
  sorry
}

end price_after_discounts_l832_83268


namespace part_a_part_b_part_c_l832_83289

-- Part a
theorem part_a (n: ℕ) (h: n = 1): (n^2 - 5 * n + 4) / (n - 4) = 0 := by sorry

-- Part b
theorem part_b (n: ℕ) (h: (n^2 - 5 * n + 4) / (n - 4) = 5): n = 6 := 
  by sorry

-- Part c
theorem part_c (n: ℕ) (h : n ≠ 4): (n^2 - 5 * n + 4) / (n - 4) ≠ 3 := 
  by sorry

end part_a_part_b_part_c_l832_83289


namespace systematic_sampling_eighth_group_l832_83230

theorem systematic_sampling_eighth_group (total_students : ℕ) (groups : ℕ) (group_size : ℕ)
(start_number : ℕ) (group_number : ℕ)
(h1 : total_students = 480)
(h2 : groups = 30)
(h3 : group_size = 16)
(h4 : start_number = 5)
(h5 : group_number = 8) :
  (group_number - 1) * group_size + start_number = 117 := by
  sorry

end systematic_sampling_eighth_group_l832_83230


namespace find_widgets_l832_83233

theorem find_widgets (a b c d e f : ℕ) : 
  (3 * a + 11 * b + 5 * c + 7 * d + 13 * e + 17 * f = 3255) →
  (3 ^ a * 11 ^ b * 5 ^ c * 7 ^ d * 13 ^ e * 17 ^ f = 351125648000) →
  c = 3 :=
by
  sorry

end find_widgets_l832_83233


namespace system_of_equations_solution_l832_83267

theorem system_of_equations_solution :
  ∃ x y : ℝ, (x + y = 3) ∧ (2 * x - 3 * y = 1) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end system_of_equations_solution_l832_83267


namespace inequality_solution_l832_83245

theorem inequality_solution (x : ℝ) : x * |x| ≤ 1 ↔ x ≤ 1 := 
sorry

end inequality_solution_l832_83245


namespace isabel_pop_albums_l832_83296

theorem isabel_pop_albums (total_songs : ℕ) (country_albums : ℕ) (songs_per_album : ℕ) (pop_albums : ℕ)
  (h1 : total_songs = 72)
  (h2 : country_albums = 4)
  (h3 : songs_per_album = 8)
  (h4 : total_songs - country_albums * songs_per_album = pop_albums * songs_per_album) :
  pop_albums = 5 :=
by
  sorry

end isabel_pop_albums_l832_83296


namespace one_third_of_nine_times_seven_l832_83278

theorem one_third_of_nine_times_seven : (1 / 3) * (9 * 7) = 21 := 
by
  sorry

end one_third_of_nine_times_seven_l832_83278


namespace thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_l832_83249

theorem thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five :
  (35 * 99 ≠ 35 * 100 + 35) :=
by
  sorry

end thirty_five_times_ninety_nine_is_not_thirty_five_times_hundred_plus_thirty_five_l832_83249


namespace find_x_l832_83288

theorem find_x (x : ℚ) (h1 : 3 * x + (4 * x - 10) = 90) : x = 100 / 7 :=
by {
  sorry
}

end find_x_l832_83288


namespace simplify_expression_l832_83243

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) : 
  (x^2 - x) / (x^2 - 2 * x + 1) = 2 + Real.sqrt 2 :=
by
  sorry

end simplify_expression_l832_83243


namespace find_M_l832_83264

def grid_conditions :=
  ∃ (M : ℤ), 
  ∀ d1 d2 d3 d4, 
    (d1 = 22) ∧ (d2 = 6) ∧ (d3 = -34 / 6) ∧ (d4 = (8 - M) / 6) ∧
    (10 = 32 - d2) ∧ 
    (16 = 10 + d2) ∧ 
    (-2 = 10 - d2) ∧
    (32 - M = 34 / 6 * 6) ∧ 
    (M = -34 / 6 - (-17 / 3))

theorem find_M : grid_conditions → ∃ M : ℤ, M = 17 :=
by
  intros
  existsi (17 : ℤ) 
  sorry

end find_M_l832_83264


namespace price_increase_decrease_eq_l832_83282

theorem price_increase_decrease_eq (x : ℝ) (p : ℝ) (hx : x ≠ 0) :
  x * (1 + p / 100) * (1 - p / 200) = x * (1 + p / 300) → p = 100 / 3 :=
by
  intro h
  -- The proof would go here
  sorry

end price_increase_decrease_eq_l832_83282


namespace measure_of_angle_B_scalene_triangle_l832_83213

theorem measure_of_angle_B_scalene_triangle (A B C : ℝ) (hA_gt_0 : A > 0) (hB_gt_0 : B > 0) (hC_gt_0 : C > 0) 
(h_angles_sum : A + B + C = 180) (hB_eq_2A : B = 2 * A) (hC_eq_3A : C = 3 * A) : B = 60 :=
by
  sorry

end measure_of_angle_B_scalene_triangle_l832_83213


namespace find_expression_l832_83215

theorem find_expression (x : ℝ) (h : (1 / Real.cos (2022 * x)) + Real.tan (2022 * x) = 1 / 2022) :
  (1 / Real.cos (2022 * x)) - Real.tan (2022 * x) = 2022 :=
by
  sorry

end find_expression_l832_83215


namespace sum_invested_eq_2000_l832_83262

theorem sum_invested_eq_2000 (P : ℝ) (R1 R2 T : ℝ) (H1 : R1 = 18) (H2 : R2 = 12) 
  (H3 : T = 2) (H4 : (P * R1 * T / 100) - (P * R2 * T / 100) = 240): 
  P = 2000 :=
by 
  sorry

end sum_invested_eq_2000_l832_83262


namespace hungarian_license_plates_l832_83290

/-- 
In Hungarian license plates, digits can be identical. Based on observations, 
someone claimed that on average, approximately 3 out of every 10 vehicles 
have such license plates. Is this statement true?
-/
theorem hungarian_license_plates : 
  let total_numbers := 999
  let non_repeating := 720
  let repeating := total_numbers - non_repeating
  let probability := (repeating : ℝ) / total_numbers
  abs (probability - 0.3) < 0.05 :=
by {
  let total_numbers := 999
  let non_repeating := 720
  let repeating := total_numbers - non_repeating
  let probability := (repeating : ℝ) / total_numbers
  sorry
}

end hungarian_license_plates_l832_83290


namespace birdhouse_volume_difference_l832_83242

-- Definitions to capture the given conditions
def sara_width_ft : ℝ := 1
def sara_height_ft : ℝ := 2
def sara_depth_ft : ℝ := 2

def jake_width_in : ℝ := 16
def jake_height_in : ℝ := 20
def jake_depth_in : ℝ := 18

-- Convert Sara's dimensions to inches
def ft_to_in (x : ℝ) : ℝ := x * 12
def sara_width_in := ft_to_in sara_width_ft
def sara_height_in := ft_to_in sara_height_ft
def sara_depth_in := ft_to_in sara_depth_ft

-- Volume calculations
def volume (width height depth : ℝ) := width * height * depth
def sara_volume := volume sara_width_in sara_height_in sara_depth_in
def jake_volume := volume jake_width_in jake_height_in jake_depth_in

-- The theorem to prove the difference in volume
theorem birdhouse_volume_difference : sara_volume - jake_volume = 1152 := by
  -- Proof goes here
  sorry

end birdhouse_volume_difference_l832_83242


namespace sum_of_polynomials_l832_83208

theorem sum_of_polynomials (d : ℕ) :
  let expr1 := 15 * d + 17 + 16 * d ^ 2
  let expr2 := 3 * d + 2
  let sum_expr := expr1 + expr2
  let a := 16
  let b := 18
  let c := 19
  sum_expr = a * d ^ 2 + b * d + c ∧ a + b + c = 53 := by
    sorry

end sum_of_polynomials_l832_83208


namespace div_problem_l832_83221

variables (A B C : ℝ)

theorem div_problem (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 527) : B = 93 :=
by {
  sorry
}

end div_problem_l832_83221


namespace find_min_value_omega_l832_83226

noncomputable def min_value_ω (ω : ℝ) : Prop :=
  ∀ (f : ℝ → ℝ), (∀ (x : ℝ), f x = 2 * Real.sin (ω * x)) → ω > 0 →
  (∀ (x : ℝ), -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 4 → f x ≥ -2) →
  ω = 3 / 2

-- The statement to be proved:
theorem find_min_value_omega : ∃ ω : ℝ, min_value_ω ω :=
by
  use 3 / 2
  sorry

end find_min_value_omega_l832_83226


namespace rabbit_count_l832_83266

theorem rabbit_count (r1 r2 : ℕ) (h1 : r1 = 8) (h2 : r2 = 5) : r1 + r2 = 13 := 
by 
  sorry

end rabbit_count_l832_83266


namespace contrapositive_property_l832_83252

def is_divisible_by_6 (n : ℤ) : Prop := n % 6 = 0
def is_divisible_by_2 (n : ℤ) : Prop := n % 2 = 0

theorem contrapositive_property :
  (∀ n : ℤ, is_divisible_by_6 n → is_divisible_by_2 n) ↔ (∀ n : ℤ, ¬ is_divisible_by_2 n → ¬ is_divisible_by_6 n) :=
by
  sorry

end contrapositive_property_l832_83252


namespace trig_identity_l832_83232

theorem trig_identity : 2 * Real.sin (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - 1 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_l832_83232


namespace number_of_cows_l832_83253

variable {D C : ℕ}

theorem number_of_cows (h : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 :=
by sorry

end number_of_cows_l832_83253


namespace base_angle_isosceles_triangle_l832_83258

theorem base_angle_isosceles_triangle
  (sum_angles : ∀ (α β γ : ℝ), α + β + γ = 180)
  (isosceles : ∀ (α β : ℝ), α = β)
  (one_angle_forty : ∃ α : ℝ, α = 40) :
  ∃ β : ℝ, β = 70 ∨ β = 40 :=
by
  sorry

end base_angle_isosceles_triangle_l832_83258


namespace hyperbola_eq_from_conditions_l832_83286

-- Conditions of the problem
def hyperbola_center : Prop := ∃ (h : ℝ → ℝ → Prop), h 0 0
def hyperbola_eccentricity : Prop := ∃ e : ℝ, e = 2
def parabola_focus : Prop := ∃ p : ℝ × ℝ, p = (4, 0)
def parabola_equation : Prop := ∀ x y : ℝ, y^2 = 8 * x

-- Hyperbola equation to be proved
def hyperbola_equation : Prop := ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1

-- Lean 4 theorem statement
theorem hyperbola_eq_from_conditions 
  (h_center : hyperbola_center) 
  (h_eccentricity : hyperbola_eccentricity) 
  (p_focus : parabola_focus) 
  (p_eq : parabola_equation) 
  : hyperbola_equation :=
by
  sorry

end hyperbola_eq_from_conditions_l832_83286


namespace no_adjacent_teachers_l832_83280

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def permutation (n k : ℕ) : ℕ :=
  factorial n / factorial (n - k)

theorem no_adjacent_teachers (students teachers : ℕ)
  (h_students : students = 4)
  (h_teachers : teachers = 3) :
  ∃ (arrangements : ℕ), arrangements = (factorial students) * (permutation (students + 1) teachers) :=
by
  sorry

end no_adjacent_teachers_l832_83280


namespace water_formed_l832_83251

theorem water_formed (CaOH2 CO2 CaCO3 H2O : Nat) 
  (h_balanced : ∀ n, n * CaOH2 + n * CO2 = n * CaCO3 + n * H2O)
  (h_initial : CaOH2 = 2 ∧ CO2 = 2) : 
  H2O = 2 :=
by
  sorry

end water_formed_l832_83251


namespace solve_for_x_l832_83241

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l832_83241


namespace maria_mushrooms_l832_83274

theorem maria_mushrooms (potatoes carrots onions green_beans bell_peppers mushrooms : ℕ) 
  (h1 : carrots = 6 * potatoes)
  (h2 : onions = 2 * carrots)
  (h3 : green_beans = onions / 3)
  (h4 : bell_peppers = 4 * green_beans)
  (h5 : mushrooms = 3 * bell_peppers)
  (h0 : potatoes = 3) : 
  mushrooms = 144 :=
by
  sorry

end maria_mushrooms_l832_83274


namespace simple_interest_proof_l832_83254

def simple_interest (P R T: ℝ) : ℝ :=
  P * R * T

theorem simple_interest_proof :
  simple_interest 810 (4.783950617283951 / 100) 4 = 154.80 :=
by
  sorry

end simple_interest_proof_l832_83254


namespace find_numbers_l832_83259

theorem find_numbers (a b : ℝ) (h1 : a - b = 7.02) (h2 : a = 10 * b) : a = 7.8 ∧ b = 0.78 :=
by
  sorry

end find_numbers_l832_83259


namespace tanvi_min_candies_l832_83298

theorem tanvi_min_candies : 
  ∃ c : ℕ, 
  (c % 6 = 5) ∧ 
  (c % 8 = 7) ∧ 
  (c % 9 = 6) ∧ 
  (c % 11 = 0) ∧ 
  (∀ d : ℕ, 
    (d % 6 = 5) ∧ 
    (d % 8 = 7) ∧ 
    (d % 9 = 6) ∧ 
    (d % 11 = 0) → 
    c ≤ d) → 
  c = 359 :=
by sorry

end tanvi_min_candies_l832_83298


namespace male_contestants_l832_83285

theorem male_contestants (total_contestants : ℕ) (female_proportion : ℕ) (total_females : ℕ) :
  female_proportion = 3 ∧ total_contestants = 18 ∧ total_females = total_contestants / female_proportion →
  (total_contestants - total_females) = 12 :=
by
  sorry

end male_contestants_l832_83285


namespace digit_sum_9_l832_83257

def digits := {n : ℕ // n < 10}

theorem digit_sum_9 (a b : digits) 
  (h1 : (4 * 100) + (a.1 * 10) + 3 + 984 = (1 * 1000) + (3 * 100) + (b.1 * 10) + 7) 
  (h2 : (1 + b.1) - (3 + 7) % 11 = 0) 
: a.1 + b.1 = 9 :=
sorry

end digit_sum_9_l832_83257


namespace cricket_jumps_to_100m_l832_83202

theorem cricket_jumps_to_100m (x y : ℕ) (h : 9 * x + 8 * y = 100) : x + y = 12 :=
sorry

end cricket_jumps_to_100m_l832_83202


namespace triangle_internal_angles_external_angle_theorem_l832_83228

theorem triangle_internal_angles {A B C : ℝ}
 (mA : A = 64) (mB : B = 33) (mC_ext : C = 120) :
  180 - A - B = 83 :=
by
  sorry

theorem external_angle_theorem {A C D : ℝ}
 (mA : A = 64) (mC_ext : C = 120) :
  C = A + D → D = 56 :=
by
  sorry

end triangle_internal_angles_external_angle_theorem_l832_83228


namespace proportional_value_l832_83279

theorem proportional_value :
  ∃ (x : ℝ), 18 / 60 / (12 / 60) = x / 6 ∧ x = 9 := sorry

end proportional_value_l832_83279


namespace university_minimum_spend_l832_83212

def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15
def box_cost : ℝ := 1.20
def total_volume : ℝ := 3.06 * (10^6)

def box_volume : ℕ := box_length * box_width * box_height

noncomputable def number_of_boxes : ℕ := Nat.ceil (total_volume / box_volume)
noncomputable def total_cost : ℝ := number_of_boxes * box_cost

theorem university_minimum_spend : total_cost = 612 := by
  sorry

end university_minimum_spend_l832_83212


namespace fibonacci_eighth_term_l832_83211

theorem fibonacci_eighth_term
  (F : ℕ → ℕ)
  (h1 : F 9 = 34)
  (h2 : F 10 = 55)
  (fib : ∀ n, F (n + 2) = F (n + 1) + F n) :
  F 8 = 21 :=
by
  sorry

end fibonacci_eighth_term_l832_83211


namespace bridge_toll_fees_for_annie_are_5_l832_83236

-- Conditions
def start_fee : ℝ := 2.50
def cost_per_mile : ℝ := 0.25
def mike_miles : ℕ := 36
def annie_miles : ℕ := 16
def total_cost_mike : ℝ := start_fee + cost_per_mile * mike_miles

-- Hypothesis from conditions
axiom both_charged_same : ∀ (bridge_fees : ℝ), total_cost_mike = start_fee + cost_per_mile * annie_miles + bridge_fees

-- Proof problem
theorem bridge_toll_fees_for_annie_are_5 : ∃ (bridge_fees : ℝ), bridge_fees = 5 :=
by
  existsi 5
  sorry

end bridge_toll_fees_for_annie_are_5_l832_83236


namespace incorrect_median_l832_83207

/-- 
Given:
- A stem-and-leaf plot representation.
- Player B's scores are mainly between 30 and 40 points.
- Player B has 13 scores.
Prove:
The judgment "The median score of player B is 28" is incorrect.
-/
theorem incorrect_median (scores : List ℕ) (H_len : scores.length = 13) (H_range : ∀ x ∈ scores, 30 ≤ x ∧ x ≤ 40) 
  (H_median : ∃ median, median = scores.nthLe 6 sorry ∧ median = 28) : False := 
sorry

end incorrect_median_l832_83207


namespace regular_polygon_interior_angle_160_l832_83240

theorem regular_polygon_interior_angle_160 (n : ℕ) (h : 160 * n = 180 * (n - 2)) : n = 18 :=
by {
  sorry
}

end regular_polygon_interior_angle_160_l832_83240


namespace circle_trajectory_l832_83263

theorem circle_trajectory (x y : ℝ) (h1 : (x-5)^2 + (y+7)^2 = 16) (h2 : ∃ c : ℝ, c = ((x + 1 - 5)^2 + (y + 1 + 7)^2)): 
    ((x-5)^2+(y+7)^2 = 25 ∨ (x-5)^2+(y+7)^2 = 9) :=
by
  -- Proof is omitted
  sorry

end circle_trajectory_l832_83263


namespace find_smaller_number_l832_83220

theorem find_smaller_number (a b : ℕ) (h_ratio : 11 * a = 7 * b) (h_diff : b = a + 16) : a = 28 :=
by
  sorry

end find_smaller_number_l832_83220


namespace max_value_f_at_a1_f_div_x_condition_l832_83227

noncomputable def f (a x : ℝ) : ℝ := (a - x) * Real.exp x - 1

theorem max_value_f_at_a1 :
  ∀ x : ℝ, (f 1 0) = 0 ∧ ( ∀ y : ℝ, y ≠ 0 → f 1 y < f 1 0) := 
sorry

theorem f_div_x_condition :
  ∀ x : ℝ, x ≠ 0 → (((f 1 x) / x) < 1) :=
sorry

end max_value_f_at_a1_f_div_x_condition_l832_83227


namespace cost_per_person_l832_83246

theorem cost_per_person 
  (total_cost : ℕ) 
  (total_people : ℕ) 
  (total_cost_in_billion : total_cost = 40000000000) 
  (total_people_in_million : total_people = 200000000) :
  total_cost / total_people = 200 := 
sorry

end cost_per_person_l832_83246


namespace find_t_of_quadratic_root_l832_83271

variable (a t : ℝ)

def quadratic_root_condition (a : ℝ) : Prop :=
  ∃ t : ℝ, Complex.ofReal a + Complex.I * 3 = Complex.ofReal a - Complex.I * 3 ∧
           (Complex.ofReal a + Complex.I * 3).re * (Complex.ofReal a - Complex.I * 3).re = t

theorem find_t_of_quadratic_root (h : quadratic_root_condition a) : t = 13 :=
sorry

end find_t_of_quadratic_root_l832_83271


namespace grove_town_fall_expenditure_l832_83205

-- Define the expenditures at the end of August and November
def expenditure_end_of_august : ℝ := 3.0
def expenditure_end_of_november : ℝ := 5.5

-- Define the spending during fall months (September, October, November)
def spending_during_fall_months : ℝ := 2.5

-- Statement to be proved
theorem grove_town_fall_expenditure :
  expenditure_end_of_november - expenditure_end_of_august = spending_during_fall_months :=
by
  sorry

end grove_town_fall_expenditure_l832_83205


namespace mean_of_three_added_numbers_l832_83225

theorem mean_of_three_added_numbers (x y z : ℝ) :
  (∀ (s : ℝ), (s / 7 = 75) → (s + x + y + z) / 10 = 90) → (x + y + z) / 3 = 125 :=
by
  intro h
  sorry

end mean_of_three_added_numbers_l832_83225


namespace circle_y_axis_intersection_range_l832_83297

theorem circle_y_axis_intersection_range (m : ℝ) : (4 - 4 * (m + 6) > 0) → (-2 < 0) → (m + 6 > 0) → (-6 < m ∧ m < -5) :=
by 
  intros h1 h2 h3 
  sorry

end circle_y_axis_intersection_range_l832_83297


namespace smallest_positive_integer_with_12_divisors_l832_83260

theorem smallest_positive_integer_with_12_divisors : ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≠ n) → n ≤ m) ∧ ∃ d : ℕ → ℕ, (d n = 12) :=
by
  sorry

end smallest_positive_integer_with_12_divisors_l832_83260


namespace orthogonal_vectors_l832_83270

open Real

variables (r s : ℝ)

def a : ℝ × ℝ × ℝ := (5, r, -3)
def b : ℝ × ℝ × ℝ := (-1, 2, s)

theorem orthogonal_vectors
  (orthogonality : 5 * (-1) + r * 2 + (-3) * s = 0)
  (magnitude_condition : 34 + r^2 = 4 * (5 + s^2)) :
  ∃ (r s : ℝ), (2 * r - 3 * s = 5) ∧ (r^2 - 4 * s^2 = -14) :=
  sorry

end orthogonal_vectors_l832_83270


namespace chris_is_14_l832_83214

-- Definitions from the given conditions
variables (a b c : ℕ)
variables (h1 : (a + b + c) / 3 = 10)
variables (h2 : c - 4 = a)
variables (h3 : b + 5 = (3 * (a + 5)) / 4)

theorem chris_is_14 (h1 : (a + b + c) / 3 = 10) (h2 : c - 4 = a) (h3 : b + 5 = (3 * (a + 5)) / 4) : c = 14 := 
sorry

end chris_is_14_l832_83214


namespace value_of_expression_l832_83295

-- Given conditions
variable (n : ℤ)
def m : ℤ := 4 * n + 3

-- Main theorem statement
theorem value_of_expression (n : ℤ) : 
  (m n)^2 - 8 * (m n) * n + 16 * n^2 = 9 := 
  sorry

end value_of_expression_l832_83295


namespace tricycles_count_l832_83200

-- Define the conditions
variable (b t s : ℕ)

def total_children := b + t + s = 10
def total_wheels := 2 * b + 3 * t + 2 * s = 29

-- Provide the theorem to prove
theorem tricycles_count (h1 : total_children b t s) (h2 : total_wheels b t s) : t = 9 := 
by
  sorry

end tricycles_count_l832_83200


namespace sequence_filling_l832_83216

theorem sequence_filling :
  ∃ (a : Fin 8 → ℕ), 
    a 0 = 20 ∧ 
    a 7 = 16 ∧ 
    (∀ i : Fin 6, a i + a (i+1) + a (i+2) = 100) ∧ 
    (a 1 = 16) ∧ 
    (a 2 = 64) ∧ 
    (a 3 = 20) ∧ 
    (a 4 = 16) ∧ 
    (a 5 = 64) ∧ 
    (a 6 = 20) := 
by
  sorry

end sequence_filling_l832_83216


namespace least_positive_x_multiple_l832_83244

theorem least_positive_x_multiple (x : ℕ) : 
  (∃ k : ℕ, (2 * x + 41) = 53 * k) → 
  x = 6 :=
sorry

end least_positive_x_multiple_l832_83244


namespace solve_inequality_l832_83229

noncomputable def solution_set (a b : ℝ) (x : ℝ) : Prop :=
x < -1 / b ∨ x > 1 / a

theorem solve_inequality (a b : ℝ) (x : ℝ)
  (h_a : a > 0) (h_b : b > 0) :
  (-b < 1 / x ∧ 1 / x < a) ↔ solution_set a b x :=
by
  sorry

end solve_inequality_l832_83229


namespace book_cost_is_2_l832_83234

-- Define initial amount of money
def initial_amount : ℕ := 48

-- Define the number of books purchased
def num_books : ℕ := 5

-- Define the amount of money left after purchasing the books
def amount_left : ℕ := 38

-- Define the cost per book
def cost_per_book (initial amount_left : ℕ) (num_books : ℕ) : ℕ := (initial - amount_left) / num_books

-- The theorem to prove
theorem book_cost_is_2
    (initial_amount : ℕ := 48) 
    (amount_left : ℕ := 38) 
    (num_books : ℕ := 5) :
    cost_per_book initial_amount amount_left num_books = 2 :=
by
  sorry

end book_cost_is_2_l832_83234


namespace equilateral_triangle_t_gt_a_squared_l832_83248

theorem equilateral_triangle_t_gt_a_squared {a x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ a) :
  2 * x^2 - 2 * a * x + 3 * a^2 > a^2 :=
by {
  sorry
}

end equilateral_triangle_t_gt_a_squared_l832_83248


namespace pythagorean_theorem_example_l832_83299

noncomputable def a : ℕ := 6
noncomputable def b : ℕ := 8
noncomputable def c : ℕ := 10

theorem pythagorean_theorem_example :
  c = Real.sqrt (a^2 + b^2) := 
by
  sorry

end pythagorean_theorem_example_l832_83299


namespace find_a_range_l832_83281

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def g (x a : ℝ) : ℝ := 3 * Real.exp x + a

theorem find_a_range (a : ℝ) :
  (∃ x, x ∈ Set.Icc (-2 : ℝ) 2 ∧ f x > g x a) → a < Real.exp 2 :=
by
  sorry

end find_a_range_l832_83281


namespace shaded_to_largest_ratio_l832_83222

theorem shaded_to_largest_ratio :
  let r1 := 1
  let r2 := 2
  let r3 := 3
  let r4 := 4
  let area (r : ℝ) := π * r^2
  let largest_circle_area := area r4
  let innermost_shaded_area := area r1
  let outermost_shaded_area := area r3 - area r2
  let shaded_area := innermost_shaded_area + outermost_shaded_area
  shaded_area / largest_circle_area = 3 / 8 :=
by
  sorry

end shaded_to_largest_ratio_l832_83222


namespace jade_transactions_l832_83219

theorem jade_transactions (mabel anthony cal jade : ℕ) 
    (h1 : mabel = 90) 
    (h2 : anthony = mabel + (10 * mabel / 100)) 
    (h3 : cal = 2 * anthony / 3) 
    (h4 : jade = cal + 18) : 
    jade = 84 := by 
  -- Start with given conditions
  rw [h1] at h2 
  have h2a : anthony = 99 := by norm_num; exact h2 
  rw [h2a] at h3 
  have h3a : cal = 66 := by norm_num; exact h3 
  rw [h3a] at h4 
  norm_num at h4 
  exact h4

end jade_transactions_l832_83219


namespace bus_trip_times_l832_83239

/-- Given two buses traveling towards each other from points A and B which are 120 km apart.
The first bus stops for 10 minutes and the second bus stops for 5 minutes. The first bus reaches 
its destination 25 minutes before the second bus. The first bus travels 20 km/h faster than the 
second bus. Prove that the travel times for the buses are 
1 hour 40 minutes and 2 hours 5 minutes respectively. -/
theorem bus_trip_times (d : ℕ) (v1 v2 : ℝ) (t1 t2 t : ℝ) (h1 : d = 120) (h2 : v1 = v2 + 20) 
(h3 : t1 = d / v1 + 10) (h4 : t2 = d / v2 + 5) (h5 : t2 - t1 = 25) :
t1 = 100 ∧ t2 = 125 := 
by 
  sorry

end bus_trip_times_l832_83239


namespace min_q_difference_l832_83269

theorem min_q_difference (p q : ℕ) (hpq : 0 < p ∧ 0 < q) (ineq1 : (7:ℚ)/12 < p/q) (ineq2 : p/q < (5:ℚ)/8) (hmin : ∀ r s : ℕ, 0 < r ∧ 0 < s ∧ (7:ℚ)/12 < r/s ∧ r/s < (5:ℚ)/8 → q ≤ s) : q - p = 2 :=
sorry

end min_q_difference_l832_83269


namespace Yi_visited_city_A_l832_83277

variable (visited : String -> String -> Prop) -- denote visited "Student" "City"
variables (Jia Yi Bing : String) (A B C : String)

theorem Yi_visited_city_A
  (h1 : visited Jia A ∧ visited Jia C ∧ ¬ visited Jia B)
  (h2 : ¬ visited Yi C)
  (h3 : visited Jia A ∧ visited Yi A ∧ visited Bing A) :
  visited Yi A :=
by
  sorry

end Yi_visited_city_A_l832_83277


namespace trigonometric_inequality_proof_l832_83247

theorem trigonometric_inequality_proof : 
  ∀ (sin cos : ℝ → ℝ), 
  (∀ θ, 0 ≤ θ ∧ θ ≤ π/2 → sin θ = cos (π/2 - θ)) → 
  sin (π * 11 / 180) < sin (π * 12 / 180) ∧ sin (π * 12 / 180) < sin (π * 80 / 180) :=
by 
  intros sin cos identity
  sorry

end trigonometric_inequality_proof_l832_83247


namespace quadratic_has_distinct_real_roots_find_k_l832_83276

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem quadratic_has_distinct_real_roots (k : ℝ) : 
  let a := 1
  let b := 2 * k - 1
  let c := -k - 2
  let Δ := b^2 - 4 * a * c
  (Δ > 0) :=
by
  sorry

-- Part 2: Given the roots condition, find k
theorem find_k (x1 x2 k : ℝ)
  (h1 : x1 + x2 = -(2 * k - 1))
  (h2 : x1 * x2 = -k - 2)
  (h3 : x1 + x2 - 4 * x1 * x2 = 1) : 
  k = -4 :=
by
  sorry

end quadratic_has_distinct_real_roots_find_k_l832_83276


namespace largest_root_vieta_l832_83293

theorem largest_root_vieta 
  (a b c : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a * b + a * c + b * c = 11)
  (h3 : a * b * c = -6) : 
  max a (max b c) = 3 :=
sorry

end largest_root_vieta_l832_83293


namespace geometric_sequence_a3_value_l832_83294

theorem geometric_sequence_a3_value
  {a : ℕ → ℝ}
  (h1 : a 1 + a 5 = 82)
  (h2 : a 2 * a 4 = 81)
  (h3 : ∀ n : ℕ, a (n + 1) = a n * a 3 / a 2) :
  a 3 = 9 :=
sorry

end geometric_sequence_a3_value_l832_83294


namespace rationalize_denominator_l832_83209

theorem rationalize_denominator : (14 / Real.sqrt 14) = Real.sqrt 14 := by
  sorry

end rationalize_denominator_l832_83209


namespace unit_digit_product_is_zero_l832_83238

-- Definitions based on conditions in (a)
def a_1 := 6245
def a_2 := 7083
def a_3 := 9137
def a_4 := 4631
def a_5 := 5278
def a_6 := 3974

-- Helper function to get the unit digit of a number
def unit_digit (n : Nat) : Nat := n % 10

-- Main theorem to prove
theorem unit_digit_product_is_zero :
  unit_digit (a_1 * a_2 * a_3 * a_4 * a_5 * a_6) = 0 := by
  sorry

end unit_digit_product_is_zero_l832_83238


namespace quartic_to_quadratic_l832_83201

-- Defining the statement of the problem
theorem quartic_to_quadratic (a b c x : ℝ) (y : ℝ) :
  a * x^4 + b * x^3 + c * x^2 + b * x + a = 0 →
  y = x + 1 / x →
  ∃ y1 y2, (a * y^2 + b * y + (c - 2 * a) = 0) ∧
           (x^2 - y1 * x + 1 = 0 ∨ x^2 - y2 * x + 1 = 0) :=
by
  sorry

end quartic_to_quadratic_l832_83201


namespace rate_percent_calculation_l832_83250

theorem rate_percent_calculation (SI P T : ℝ) (R : ℝ) : SI = 640 ∧ P = 4000 ∧ T = 2 → SI = P * R * T / 100 → R = 8 :=
by
  intros
  sorry

end rate_percent_calculation_l832_83250


namespace soccer_ball_cost_l832_83235

theorem soccer_ball_cost (x : ℝ) (soccer_balls basketballs : ℕ) 
  (soccer_ball_cost basketball_cost : ℝ) 
  (h1 : soccer_balls = 2 * basketballs)
  (h2 : 5000 = soccer_balls * soccer_ball_cost)
  (h3 : 4000 = basketballs * basketball_cost)
  (h4 : basketball_cost = soccer_ball_cost + 30)
  (eqn : 5000 / soccer_ball_cost = 2 * (4000 / basketball_cost)) :
  soccer_ball_cost = x :=
by
  sorry

end soccer_ball_cost_l832_83235


namespace arctan_sum_eq_pi_div_two_l832_83210

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 7) + Real.arctan (7 / 3) = Real.pi / 2 := 
sorry

end arctan_sum_eq_pi_div_two_l832_83210


namespace y_value_l832_83287

theorem y_value (y : ℕ) : 8^3 + 8^3 + 8^3 + 8^3 = 2^y → y = 11 := 
by 
  sorry

end y_value_l832_83287


namespace negation_proposition_l832_83224

theorem negation_proposition :
  (∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) :=
by sorry

end negation_proposition_l832_83224


namespace speed_of_stream_l832_83237

theorem speed_of_stream
  (D : ℝ) (v : ℝ)
  (h : D / (72 - v) = 2 * D / (72 + v)) :
  v = 24 := by
  sorry

end speed_of_stream_l832_83237


namespace price_after_9_years_decreases_continuously_l832_83203

theorem price_after_9_years_decreases_continuously (price_current : ℝ) (price_after_9_years : ℝ) :
  (∀ k : ℕ, k % 3 = 0 → price_current = 8100 → price_after_9_years = 2400) :=
sorry

end price_after_9_years_decreases_continuously_l832_83203


namespace unique_arrangement_l832_83218

def valid_grid (arrangement : Matrix (Fin 4) (Fin 4) Char) : Prop :=
  (∀ i : Fin 4, (∃ j1 j2 j3 : Fin 4,
    j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3 ∧
    arrangement i j1 = 'A' ∧
    arrangement i j2 = 'B' ∧
    arrangement i j3 = 'C')) ∧
  (∀ j : Fin 4, (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 j = 'A' ∧
    arrangement i2 j = 'B' ∧
    arrangement i3 j = 'C')) ∧
  (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 i1 = 'A' ∧
    arrangement i2 i2 = 'B' ∧
    arrangement i3 i3 = 'C') ∧
  (∃ i1 i2 i3 : Fin 4,
    i1 ≠ i2 ∧ i2 ≠ i3 ∧ i1 ≠ i3 ∧
    arrangement i1 (Fin.mk (3 - i1.val) sorry) = 'A' ∧
    arrangement i2 (Fin.mk (3 - i2.val) sorry) = 'B' ∧
    arrangement i3 (Fin.mk (3 - i3.val) sorry) = 'C')

def fixed_upper_left (arrangement : Matrix (Fin 4) (Fin 4) Char) : Prop :=
  arrangement 0 0 = 'A'

theorem unique_arrangement : ∃! arrangement : Matrix (Fin 4) (Fin 4) Char,
  valid_grid arrangement ∧ fixed_upper_left arrangement :=
sorry

end unique_arrangement_l832_83218


namespace incorrect_intersection_point_l832_83217

def linear_function (x : ℝ) : ℝ := -2 * x + 4

theorem incorrect_intersection_point : ¬(linear_function 0 = 4) :=
by {
  /- Proof can be filled here later -/
  sorry
}

end incorrect_intersection_point_l832_83217


namespace line_through_A_with_equal_intercepts_l832_83273

theorem line_through_A_with_equal_intercepts (x y : ℝ) (A : ℝ × ℝ) (hx : A = (2, 1)) :
  (∃ k : ℝ, x + y = k ∧ x + y - 3 = 0) ∨ (x - 2 * y = 0) :=
sorry

end line_through_A_with_equal_intercepts_l832_83273


namespace min_value_frac_gcd_l832_83272

theorem min_value_frac_gcd {N k : ℕ} (hN_substring : N % 10^5 = 11235) (hN_pos : 0 < N) (hk_pos : 0 < k) (hk_bound : 10^k > N) : 
  (10^k - 1) / Nat.gcd N (10^k - 1) = 89 :=
by
  -- proof goes here
  sorry

end min_value_frac_gcd_l832_83272


namespace soup_options_l832_83223

-- Define the given conditions
variables (lettuce_types tomato_types olive_types total_options : ℕ)
variable (S : ℕ)

-- State the conditions
theorem soup_options :
  lettuce_types = 2 →
  tomato_types = 3 →
  olive_types = 4 →
  total_options = 48 →
  (lettuce_types * tomato_types * olive_types * S = total_options) →
  S = 2 :=
by
  sorry

end soup_options_l832_83223


namespace geom_seq_min_value_proof_l832_83284

noncomputable def geom_seq_min_value : ℝ := 3 / 2

theorem geom_seq_min_value_proof (a : ℕ → ℝ) (a1 : ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →
  a 2017 = a 2016 + 2 * a 2015 →
  a m * a n = 16 * a1^2 →
  (4 / m + 1 / n) = geom_seq_min_value :=
by {
  sorry
}

end geom_seq_min_value_proof_l832_83284


namespace compare_logs_l832_83265

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log (1 / 2)

theorem compare_logs : c < a ∧ a < b := by
  have h0 : a = Real.log 2 / Real.log 3 := rfl
  have h1 : b = Real.log 3 / Real.log 2 := rfl
  have h2 : c = Real.log 5 / Real.log (1 / 2) := rfl
  sorry

end compare_logs_l832_83265


namespace driving_time_eqn_l832_83256

open Nat

-- Define the variables and constants
def avg_speed_before := 80 -- km/h
def stop_time := 1 / 3 -- hour
def avg_speed_after := 100 -- km/h
def total_distance := 250 -- km
def total_time := 3 -- hours

variable (t : ℝ) -- the time in hours before the stop

-- State the main theorem
theorem driving_time_eqn :
  avg_speed_before * t + avg_speed_after * (total_time - stop_time - t) = total_distance := by
  sorry

end driving_time_eqn_l832_83256


namespace inequality_solution_set_l832_83292

theorem inequality_solution_set (a b x : ℝ) (h1 : a > 0) (h2 : b = a) : 
  ((a * x + b) * (x - 3) > 0 ↔ x < -1 ∨ x > 3) :=
by
  sorry

end inequality_solution_set_l832_83292


namespace mike_cards_remaining_l832_83283

-- Define initial condition
def mike_initial_cards : ℕ := 87

-- Define the cards bought by Sam
def sam_bought_cards : ℕ := 13

-- Define the expected remaining cards
def mike_final_cards := mike_initial_cards - sam_bought_cards

-- Theorem to prove the final count of Mike's baseball cards
theorem mike_cards_remaining : mike_final_cards = 74 := by
  sorry

end mike_cards_remaining_l832_83283


namespace original_rent_eq_l832_83261

theorem original_rent_eq (R : ℝ)
  (h1 : 4 * 800 = 3200)
  (h2 : 4 * 850 = 3400)
  (h3 : 3400 - 3200 = 200)
  (h4 : 200 = 0.25 * R) : R = 800 := by
  sorry

end original_rent_eq_l832_83261


namespace n_cubed_minus_n_plus_one_is_square_l832_83231

theorem n_cubed_minus_n_plus_one_is_square (n : ℕ) (h : (n^5 + n^4 + 1).divisors.card = 6) : ∃ k : ℕ, n^3 - n + 1 = k^2 :=
sorry

end n_cubed_minus_n_plus_one_is_square_l832_83231


namespace hammers_in_comparison_group_l832_83206

theorem hammers_in_comparison_group (H W x : ℝ) (h1 : 2 * H + 2 * W = 1 / 3 * (x * H + 5 * W)) (h2 : W = 2 * H) :
  x = 8 :=
sorry

end hammers_in_comparison_group_l832_83206


namespace apples_for_juice_l832_83275

def totalApples : ℝ := 6
def exportPercentage : ℝ := 0.25
def juicePercentage : ℝ := 0.60

theorem apples_for_juice : 
  let remainingApples := totalApples * (1 - exportPercentage)
  let applesForJuice := remainingApples * juicePercentage
  applesForJuice = 2.7 :=
by
  sorry

end apples_for_juice_l832_83275
