import Mathlib

namespace strictly_increasing_function_exists_l2182_218260

noncomputable def exists_strictly_increasing_function (f : ℕ → ℕ) :=
  (∀ n : ℕ, n = 1 → f n = 2) ∧
  (∀ n : ℕ, f (f n) = f n + n) ∧
  (∀ m n : ℕ, m < n → f m < f n)

theorem strictly_increasing_function_exists : 
  ∃ f : ℕ → ℕ,
  exists_strictly_increasing_function f :=
sorry

end strictly_increasing_function_exists_l2182_218260


namespace striped_shirts_more_than_shorts_l2182_218284

theorem striped_shirts_more_than_shorts :
  ∀ (total_students striped_students checkered_students short_students : ℕ),
    total_students = 81 →
    striped_students = total_students * 2 / 3 →
    checkered_students = total_students - striped_students →
    short_students = checkered_students + 19 →
    striped_students - short_students = 8 :=
by
  intros total_students striped_students checkered_students short_students
  sorry

end striped_shirts_more_than_shorts_l2182_218284


namespace major_axis_length_l2182_218271

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

def foci_1 : ℝ × ℝ := (3, 5)
def foci_2 : ℝ × ℝ := (23, 40)
def reflected_foci_1 : ℝ × ℝ := (-3, 5)

theorem major_axis_length :
  distance (reflected_foci_1.1) (reflected_foci_1.2) (foci_2.1) (foci_2.2) = Real.sqrt 1921 :=
sorry

end major_axis_length_l2182_218271


namespace paul_walking_time_l2182_218221

variable (P : ℕ)

def is_walking_time (P : ℕ) : Prop :=
  P + 7 * (P + 2) = 46

theorem paul_walking_time (h : is_walking_time P) : P = 4 :=
by sorry

end paul_walking_time_l2182_218221


namespace polynomial_roots_bc_product_l2182_218246

theorem polynomial_roots_bc_product : ∃ (b c : ℤ), 
  (∀ x, (x^2 - 2*x - 1 = 0 → x^5 - b*x^3 - c*x^2 = 0)) ∧ (b * c = 348) := by 
  sorry

end polynomial_roots_bc_product_l2182_218246


namespace relay_race_total_time_correct_l2182_218278

-- Conditions as definitions
def athlete1_time : ℕ := 55
def athlete2_time : ℕ := athlete1_time + 10
def athlete3_time : ℕ := athlete2_time - 15
def athlete4_time : ℕ := athlete1_time - 25
def athlete5_time : ℕ := 80
def athlete6_time : ℕ := athlete5_time - 20
def athlete7_time : ℕ := 70
def athlete8_time : ℕ := athlete7_time - 5

-- Sum of all athletes' times
def total_time : ℕ :=
  athlete1_time + athlete2_time + athlete3_time + athlete4_time + athlete5_time +
  athlete6_time + athlete7_time + athlete8_time

-- Statement to prove
theorem relay_race_total_time_correct : total_time = 475 :=
  by
  sorry

end relay_race_total_time_correct_l2182_218278


namespace apples_equation_l2182_218266

variable {A J H : ℕ}

theorem apples_equation:
    A + J = 12 →
    H = A + J + 9 →
    A = J + 8 →
    H = 21 :=
by
  intros h1 h2 h3
  sorry

end apples_equation_l2182_218266


namespace correct_calculation_l2182_218208

theorem correct_calculation :
  3 * Real.sqrt 2 - (Real.sqrt 2 / 2) = (5 / 2) * Real.sqrt 2 :=
by
  -- To proceed with the proof, we need to show:
  -- 3 * sqrt(2) - (sqrt(2) / 2) = (5 / 2) * sqrt(2)
  sorry

end correct_calculation_l2182_218208


namespace gcd_g_50_52_l2182_218297

/-- Define the polynomial function g -/
def g (x : ℤ) : ℤ := x^2 - 3 * x + 2023

/-- The theorem stating the gcd of g(50) and g(52) -/
theorem gcd_g_50_52 : Int.gcd (g 50) (g 52) = 1 := by
  sorry

end gcd_g_50_52_l2182_218297


namespace max_area_of_rectangular_garden_l2182_218219

-- Definitions corresponding to the conditions in the problem
def length1 (x : ℕ) := x
def length2 (x : ℕ) := 75 - x

-- Definition of the area
def area (x : ℕ) := x * (75 - x)

-- Statement to prove: there exists natural numbers x and y such that x + y = 75 and x * y = 1406
theorem max_area_of_rectangular_garden :
  ∃ (x : ℕ), (x + (75 - x) = 75) ∧ (x * (75 - x) = 1406) := 
by
  -- Due to the nature of this exercise, the actual proof is omitted.
  sorry

end max_area_of_rectangular_garden_l2182_218219


namespace tan_theta_determined_l2182_218285

theorem tan_theta_determined (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 4) (h_zero : Real.tan θ + Real.tan (4 * θ) = 0) :
  Real.tan θ = Real.sqrt (5 - 2 * Real.sqrt 5) :=
sorry

end tan_theta_determined_l2182_218285


namespace balls_per_color_l2182_218211

theorem balls_per_color (total_balls : ℕ) (total_colors : ℕ)
  (h1 : total_balls = 350) (h2 : total_colors = 10) : 
  total_balls / total_colors = 35 :=
by
  sorry

end balls_per_color_l2182_218211


namespace sin_240_deg_l2182_218233

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_240_deg_l2182_218233


namespace williams_farm_tax_l2182_218245

variables (T : ℝ)
variables (tax_collected : ℝ := 3840)
variables (percentage_williams_land : ℝ := 0.5)
variables (percentage_taxable_land : ℝ := 0.25)

theorem williams_farm_tax : (percentage_williams_land * tax_collected) = 1920 := by
  sorry

end williams_farm_tax_l2182_218245


namespace n_gon_partition_l2182_218236

-- Define a function to determine if an n-gon can be partitioned as required
noncomputable def canBePartitioned (n : ℕ) (h : n ≥ 3) : Prop :=
  n ≠ 4 ∧ n ≥ 3

theorem n_gon_partition (n : ℕ) (h : n ≥ 3) : canBePartitioned n h ↔ (n = 3 ∨ n ≥ 5) :=
by sorry

end n_gon_partition_l2182_218236


namespace remaining_time_for_P_l2182_218244

theorem remaining_time_for_P 
  (P_rate : ℝ) (Q_rate : ℝ) (together_time : ℝ) (remaining_time_minutes : ℝ)
  (hP_rate : P_rate = 1 / 3) 
  (hQ_rate : Q_rate = 1 / 18) 
  (h_together_time : together_time = 2) 
  (h_remaining_time_minutes : remaining_time_minutes = 40) :
  (((P_rate + Q_rate) * together_time) + P_rate * (remaining_time_minutes / 60)) = 1 :=
by  rw [hP_rate, hQ_rate, h_together_time, h_remaining_time_minutes]
    admit

end remaining_time_for_P_l2182_218244


namespace triangle_has_side_property_l2182_218267

theorem triangle_has_side_property (a b c : ℝ) (A B C : ℝ) 
  (h₀ : 3 * b * Real.cos C + 3 * c * Real.cos B = a^2)
  (h₁ : A + B + C = Real.pi)
  (h₂ : a = 3) :
  a = 3 := 
sorry

end triangle_has_side_property_l2182_218267


namespace find_a_tangent_line_l2182_218296

theorem find_a_tangent_line (a : ℝ) : 
  (∃ (x0 y0 : ℝ), y0 = a * x0^2 + (15/4 : ℝ) * x0 - 9 ∧ 
                  (y0 = 0 ∨ (x0 = 3/2 ∧ y0 = 27/4)) ∧ 
                  ∃ (m : ℝ), (0 - y0) = m * (1 - x0) ∧ (m = 2 * a * x0 + 15/4)) → 
  (a = -1 ∨ a = -25/64) := 
sorry

end find_a_tangent_line_l2182_218296


namespace sum_of_ages_l2182_218282

variables (P M Mo : ℕ)

def age_ratio_PM := 3 * M = 5 * P
def age_ratio_MMo := 3 * Mo = 5 * M
def age_difference := Mo = P + 64

theorem sum_of_ages : age_ratio_PM P M → age_ratio_MMo M Mo → age_difference P Mo → P + M + Mo = 196 :=
by
  intros h1 h2 h3
  sorry

end sum_of_ages_l2182_218282


namespace chessboard_not_divisible_by_10_l2182_218222

theorem chessboard_not_divisible_by_10 :
  ∀ (B : ℕ × ℕ → ℕ), 
  (∀ x y, B (x, y) < 10) ∧ 
  (∀ x y, x ≥ 0 ∧ x < 8 ∧ y ≥ 0 ∧ y < 8) →
  ¬ ( ∃ k : ℕ, ∀ x y, (B (x, y) + k) % 10 = 0 ) :=
by
  intros
  sorry

end chessboard_not_divisible_by_10_l2182_218222


namespace teena_distance_behind_poe_l2182_218277

theorem teena_distance_behind_poe (D : ℝ)
    (teena_speed : ℝ) (poe_speed : ℝ)
    (time_hours : ℝ) (teena_ahead : ℝ) :
    teena_speed = 55 
    → poe_speed = 40 
    → time_hours = 1.5 
    → teena_ahead = 15 
    → D + teena_ahead = (teena_speed - poe_speed) * time_hours 
    → D = 7.5 := 
by 
    intros 
    sorry

end teena_distance_behind_poe_l2182_218277


namespace trigonometric_identity_l2182_218201

theorem trigonometric_identity 
  (α : ℝ) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := 
sorry

end trigonometric_identity_l2182_218201


namespace average_height_Heidi_Lola_l2182_218234

theorem average_height_Heidi_Lola :
  (2.1 + 1.4) / 2 = 1.75 := by
  sorry

end average_height_Heidi_Lola_l2182_218234


namespace negate_statement_l2182_218265

variable (Students Teachers : Type)
variable (Patient : Students → Prop)
variable (PatientT : Teachers → Prop)
variable (a : ∀ t : Teachers, PatientT t)
variable (b : ∃ t : Teachers, PatientT t)
variable (c : ∀ s : Students, ¬ Patient s)
variable (d : ∀ s : Students, ¬ Patient s)
variable (e : ∃ s : Students, ¬ Patient s)
variable (f : ∀ s : Students, Patient s)

theorem negate_statement : (∃ s : Students, ¬ Patient s) ↔ ¬ (∀ s : Students, Patient s) :=
by sorry

end negate_statement_l2182_218265


namespace position_after_steps_l2182_218274

def equally_spaced_steps (total_distance num_steps distance_per_step steps_taken : ℕ) : Prop :=
  total_distance = num_steps * distance_per_step ∧ 
  ∀ k : ℕ, k ≤ num_steps → k * distance_per_step = distance_per_step * k

theorem position_after_steps (total_distance num_steps distance_per_step steps_taken : ℕ) 
  (h_eq : equally_spaced_steps total_distance num_steps distance_per_step steps_taken) 
  (h_total : total_distance = 32) (h_num : num_steps = 8) (h_steps : steps_taken = 6) : 
  steps_taken * (total_distance / num_steps) = 24 := 
by 
  sorry

end position_after_steps_l2182_218274


namespace new_area_of_card_l2182_218237

-- Conditions from the problem
def original_length : ℕ := 5
def original_width : ℕ := 7
def shortened_length := original_length - 2
def shortened_width := original_width - 1

-- Statement of the proof problem
theorem new_area_of_card : shortened_length * shortened_width = 18 :=
by
  sorry

end new_area_of_card_l2182_218237


namespace only_D_is_quadratic_l2182_218216

-- Conditions
def eq_A (x : ℝ) : Prop := x^2 + 1/x - 1 = 0
def eq_B (x : ℝ) : Prop := (2*x + 1) + x = 0
def eq_C (m x : ℝ) : Prop := 2*m^2 + x = 3
def eq_D (x : ℝ) : Prop := x^2 - x = 0

-- Proof statement
theorem only_D_is_quadratic :
  ∃ (x : ℝ), eq_D x ∧ 
  (¬(∃ x : ℝ, eq_A x) ∧ ¬(∃ x : ℝ, eq_B x) ∧ ¬(∃ (m x : ℝ), eq_C m x)) :=
by
  sorry

end only_D_is_quadratic_l2182_218216


namespace ratio_of_distances_l2182_218295

theorem ratio_of_distances
  (w x y : ℝ)
  (hw : w > 0)
  (hx : x > 0)
  (hy : y > 0)
  (h_eq_time : y / w = x / w + (x + y) / (5 * w)) :
  x / y = 2 / 3 :=
by
  sorry

end ratio_of_distances_l2182_218295


namespace monotonic_range_of_t_l2182_218239

noncomputable def f (x : ℝ) := (x^2 - 3 * x + 3) * Real.exp x

def is_monotonic_on_interval (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨ (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y)

theorem monotonic_range_of_t (t : ℝ) (ht : t > -2) :
  is_monotonic_on_interval (-2) t f ↔ (-2 < t ∧ t ≤ 0) :=
sorry

end monotonic_range_of_t_l2182_218239


namespace parabola_hyperbola_focus_l2182_218206

/-- 
Proof problem: If the focus of the parabola y^2 = 2px coincides with the right focus of the hyperbola x^2/3 - y^2/1 = 1, then p = 2.
-/
theorem parabola_hyperbola_focus (p : ℝ) :
    ∀ (focus_parabola : ℝ × ℝ) (focus_hyperbola : ℝ × ℝ),
      (focus_parabola = (p, 0)) →
      (focus_hyperbola = (2, 0)) →
      (focus_parabola = focus_hyperbola) →
        p = 2 :=
by
  intros focus_parabola focus_hyperbola h1 h2 h3
  sorry

end parabola_hyperbola_focus_l2182_218206


namespace hexahedron_has_six_faces_l2182_218249

-- Definition based on the condition
def is_hexahedron (P : Type) := 
  ∃ (f : P → ℕ), ∀ (x : P), f x = 6

-- Theorem statement based on the question and correct answer
theorem hexahedron_has_six_faces (P : Type) (h : is_hexahedron P) : 
  ∀ (x : P), ∃ (f : P → ℕ), f x = 6 :=
by 
  sorry

end hexahedron_has_six_faces_l2182_218249


namespace weight_of_raisins_proof_l2182_218213

-- Define the conditions
def weight_of_peanuts : ℝ := 0.1
def total_weight_of_snacks : ℝ := 0.5

-- Theorem to prove that the weight of raisins equals 0.4 pounds
theorem weight_of_raisins_proof : total_weight_of_snacks - weight_of_peanuts = 0.4 := by
  sorry

end weight_of_raisins_proof_l2182_218213


namespace flagpole_break_height_l2182_218257

theorem flagpole_break_height (h h_break distance : ℝ) (h_pos : 0 < h) (h_break_pos : 0 < h_break)
  (h_flagpole : h = 8) (d_distance : distance = 3) (h_relationship : (h_break ^ 2 + distance^2) = (h - h_break)^2) :
  h_break = Real.sqrt 3 :=
  sorry

end flagpole_break_height_l2182_218257


namespace women_fraction_half_l2182_218253

theorem women_fraction_half
  (total_people : ℕ)
  (married_fraction : ℝ)
  (max_unmarried_women : ℕ)
  (total_people_eq : total_people = 80)
  (married_fraction_eq : married_fraction = 1 / 2)
  (max_unmarried_women_eq : max_unmarried_women = 32) :
  (∃ (women_fraction : ℝ), women_fraction = 1 / 2) :=
by
  sorry

end women_fraction_half_l2182_218253


namespace seven_thousand_twenty_two_is_7022_l2182_218241

-- Define the translations of words to numbers
def seven_thousand : ℕ := 7000
def twenty_two : ℕ := 22

-- Define the full number by summing its parts
def seven_thousand_twenty_two : ℕ := seven_thousand + twenty_two

theorem seven_thousand_twenty_two_is_7022 : seven_thousand_twenty_two = 7022 := by
  sorry

end seven_thousand_twenty_two_is_7022_l2182_218241


namespace base6_div_by_7_l2182_218205

theorem base6_div_by_7 (k d : ℕ) (hk : 0 ≤ k ∧ k ≤ 5) (hd : 0 ≤ d ∧ d ≤ 5) (hkd : k = d) : 
  7 ∣ (217 * k + 42 * d) := 
by 
  rw [hkd]
  sorry

end base6_div_by_7_l2182_218205


namespace fourth_student_number_systematic_sampling_l2182_218280

theorem fourth_student_number_systematic_sampling :
  ∀ (students : Finset ℕ), students = Finset.range 55 →
  ∀ (sample_size : ℕ), sample_size = 4 →
  ∀ (numbers_in_sample : Finset ℕ),
  numbers_in_sample = {3, 29, 42} →
  ∃ (fourth_student : ℕ), fourth_student = 44 :=
  by sorry

end fourth_student_number_systematic_sampling_l2182_218280


namespace mary_screws_l2182_218242

theorem mary_screws (S : ℕ) (h : S + 2 * S = 24) : S = 8 :=
by sorry

end mary_screws_l2182_218242


namespace area_of_shaded_region_l2182_218270

def radius_of_first_circle : ℝ := 4
def radius_of_second_circle : ℝ := 5
def radius_of_third_circle : ℝ := 2
def radius_of_fourth_circle : ℝ := 9

theorem area_of_shaded_region :
  π * (radius_of_fourth_circle ^ 2) - π * (radius_of_first_circle ^ 2) - π * (radius_of_second_circle ^ 2) - π * (radius_of_third_circle ^ 2) = 36 * π :=
by {
  sorry
}

end area_of_shaded_region_l2182_218270


namespace opposite_of_2021_l2182_218264

theorem opposite_of_2021 : ∃ y : ℝ, 2021 + y = 0 ∧ y = -2021 :=
by
  sorry

end opposite_of_2021_l2182_218264


namespace ratio_S15_S5_l2182_218255

-- Definition of a geometric sequence sum and the given ratio S10/S5 = 1/2
noncomputable def geom_sum : ℕ → ℕ := sorry
axiom ratio_S10_S5 : geom_sum 10 / geom_sum 5 = 1 / 2

-- The goal is to prove that the ratio S15/S5 = 3/4
theorem ratio_S15_S5 : geom_sum 15 / geom_sum 5 = 3 / 4 :=
by sorry

end ratio_S15_S5_l2182_218255


namespace units_digit_7_pow_6_pow_5_l2182_218283

theorem units_digit_7_pow_6_pow_5 : (7 ^ (6 ^ 5)) % 10 = 7 := by
  -- Proof will go here
  sorry

end units_digit_7_pow_6_pow_5_l2182_218283


namespace larinjaitis_age_l2182_218224

theorem larinjaitis_age : 
  ∀ (birth_year : ℤ) (death_year : ℤ), birth_year = -30 → death_year = 30 → (death_year - birth_year + 1) = 1 :=
by
  intros birth_year death_year h_birth h_death
  sorry

end larinjaitis_age_l2182_218224


namespace problem1_problem2_l2182_218202

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := 
by
  sorry

theorem problem2 (a b : ℝ) (h1 : abs a < 1) (h2 : abs b < 1) : abs (1 - a * b) > abs (a - b) := 
by
  sorry

end problem1_problem2_l2182_218202


namespace distance_from_stream_to_meadow_l2182_218294

noncomputable def distance_from_car_to_stream : ℝ := 0.2
noncomputable def distance_from_meadow_to_campsite : ℝ := 0.1
noncomputable def total_distance_hiked : ℝ := 0.7

theorem distance_from_stream_to_meadow : 
  (total_distance_hiked - distance_from_car_to_stream - distance_from_meadow_to_campsite = 0.4) :=
by
  sorry

end distance_from_stream_to_meadow_l2182_218294


namespace fraction_to_decimal_l2182_218227

theorem fraction_to_decimal : (58 : ℚ) / 160 = 0.3625 := 
by sorry

end fraction_to_decimal_l2182_218227


namespace minimum_value_of_x_y_l2182_218275

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  x + y

theorem minimum_value_of_x_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1 - x) * (-y) = x) : minimum_value x y = 4 :=
  sorry

end minimum_value_of_x_y_l2182_218275


namespace unique_solutions_l2182_218290

noncomputable def is_solution (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ∣ (b^4 + 1) ∧ b ∣ (a^4 + 1) ∧ (Nat.floor (Real.sqrt a) = Nat.floor (Real.sqrt b))

theorem unique_solutions :
  ∀ (a b : ℕ), is_solution a b → (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by 
  sorry

end unique_solutions_l2182_218290


namespace square_side_length_l2182_218251

theorem square_side_length {s : ℝ} (h1 : 4 * s = 60) : s = 15 := 
by
  linarith

end square_side_length_l2182_218251


namespace self_descriptive_7_digit_first_digit_is_one_l2182_218203

theorem self_descriptive_7_digit_first_digit_is_one
  (A B C D E F G : ℕ)
  (h_total : A + B + C + D + E + F + G = 7)
  (h_B : B = 2)
  (h_C : C = 1)
  (h_D : D = 1)
  (h_E : E = 0)
  (h_A_zeroes : A = (if E = 0 then 1 else 0)) :
  A = 1 :=
by
  sorry

end self_descriptive_7_digit_first_digit_is_one_l2182_218203


namespace first_player_wins_if_not_power_of_two_l2182_218228

/-- 
  Prove that the first player can guarantee a win if and only if $n$ is not a power of two, under the given conditions. 
-/
theorem first_player_wins_if_not_power_of_two
  (n : ℕ) (h : n > 1) :
  (∃ k : ℕ, n = 2^k) ↔ false :=
sorry

end first_player_wins_if_not_power_of_two_l2182_218228


namespace ratio_longer_to_shorter_side_l2182_218261

-- Definitions of the problem
variables (l s : ℝ)
def rect_sheet_fold : Prop :=
  l = Real.sqrt (s^2 + (s^2 / l)^2)

-- The to-be-proved theorem
theorem ratio_longer_to_shorter_side (h : rect_sheet_fold l s) :
  l / s = Real.sqrt ((2 : ℝ) / (Real.sqrt 5 - 1)) :=
sorry

end ratio_longer_to_shorter_side_l2182_218261


namespace num_ways_to_write_360_as_increasing_seq_l2182_218215

def is_consecutive_sum (n k : ℕ) : Prop :=
  let seq_sum := k * n + k * (k - 1) / 2
  seq_sum = 360

def valid_k (k : ℕ) : Prop :=
  k ≥ 2 ∧ k ∣ 360 ∧ (k = 2 ∨ (k - 1) % 2 = 0)

noncomputable def count_consecutive_sums : ℕ :=
  Nat.card {k // valid_k k ∧ ∃ n : ℕ, is_consecutive_sum n k}

theorem num_ways_to_write_360_as_increasing_seq : count_consecutive_sums = 4 :=
sorry

end num_ways_to_write_360_as_increasing_seq_l2182_218215


namespace solve_Q1_l2182_218217

noncomputable def Q1 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y + y * f x) = f x + f y + x * f y

theorem solve_Q1 :
  ∀ f : ℝ → ℝ, Q1 f → f = (id : ℝ → ℝ) :=
  by sorry

end solve_Q1_l2182_218217


namespace recurring_decimal_to_fraction_l2182_218214

theorem recurring_decimal_to_fraction : (∃ (x : ℚ), x = 3 + 56 / 99) :=
by
  have x : ℚ := 3 + 56 / 99
  exists x
  sorry

end recurring_decimal_to_fraction_l2182_218214


namespace avg_combined_is_2a_plus_3b_l2182_218247

variables {x1 x2 x3 y1 y2 y3 a b : ℝ}

-- Given conditions
def avg_x_is_a (x1 x2 x3 a : ℝ) : Prop := (x1 + x2 + x3) / 3 = a
def avg_y_is_b (y1 y2 y3 b : ℝ) : Prop := (y1 + y2 + y3) / 3 = b

-- The statement to be proved
theorem avg_combined_is_2a_plus_3b
    (hx : avg_x_is_a x1 x2 x3 a) 
    (hy : avg_y_is_b y1 y2 y3 b) :
    ((2 * x1 + 3 * y1) + (2 * x2 + 3 * y2) + (2 * x3 + 3 * y3)) / 3 = 2 * a + 3 * b := 
by
  sorry

end avg_combined_is_2a_plus_3b_l2182_218247


namespace perfect_squares_l2182_218289

theorem perfect_squares (a b c : ℤ)
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l2182_218289


namespace find_a_b_find_range_of_x_l2182_218262

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (Real.log x / Real.log 2)^2 - 2 * a * (Real.log x / Real.log 2) + b

theorem find_a_b (a b : ℝ) :
  (f (1/4) a b = -1) → (a = -2 ∧ b = 3) :=
by
  sorry

theorem find_range_of_x (a b : ℝ) :
  a = -2 → b = 3 →
  ∀ x : ℝ, (f x a b < 0) → (1/8 < x ∧ x < 1/2) :=
by
  sorry

end find_a_b_find_range_of_x_l2182_218262


namespace negation_proposition_l2182_218226

open Classical

theorem negation_proposition :
  ¬ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ ∀ x : ℝ, x^2 - 2*x + 1 > 0 :=
by
  sorry

end negation_proposition_l2182_218226


namespace MrsHiltRows_l2182_218210

theorem MrsHiltRows :
  let (a : ℕ) := 16
  let (b : ℕ) := 14
  let (r : ℕ) := 5
  (a + b) / r = 6 := by
  sorry

end MrsHiltRows_l2182_218210


namespace inscribed_circle_diameter_l2182_218259

theorem inscribed_circle_diameter (PQ PR QR : ℝ) (h₁ : PQ = 13) (h₂ : PR = 14) (h₃ : QR = 15) :
  ∃ d : ℝ, d = 8 :=
by
  sorry

end inscribed_circle_diameter_l2182_218259


namespace abs_neg_three_l2182_218250

theorem abs_neg_three : |(-3 : ℝ)| = 3 := 
by
  -- The proof would go here, but we skip it for this exercise.
  sorry

end abs_neg_three_l2182_218250


namespace money_conditions_l2182_218276

theorem money_conditions (c d : ℝ) (h1 : 7 * c - d > 80) (h2 : 4 * c + d = 44) (h3 : d < 2 * c) :
  c > 124 / 11 ∧ d < 2 * c ∧ d = 12 :=
by
  sorry

end money_conditions_l2182_218276


namespace trig_identity_solution_l2182_218223

theorem trig_identity_solution
  (x : ℝ)
  (h : Real.sin (x + Real.pi / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 :=
by
  sorry

end trig_identity_solution_l2182_218223


namespace carson_clawed_total_l2182_218279

theorem carson_clawed_total :
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  wombats * wombat_claws + rheas * rhea_claws = 39 := by
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  show wombats * wombat_claws + rheas * rhea_claws = 39
  sorry

end carson_clawed_total_l2182_218279


namespace sum_series_eq_l2182_218254

theorem sum_series_eq : 
  ∑' n : ℕ, (n + 1) * (1 / 3 : ℝ)^n = 9 / 4 :=
by sorry

end sum_series_eq_l2182_218254


namespace num_bicycles_l2182_218220

theorem num_bicycles (spokes_per_wheel wheels_per_bicycle total_spokes : ℕ) (h1 : spokes_per_wheel = 10) (h2 : total_spokes = 80) (h3 : wheels_per_bicycle = 2) : total_spokes / spokes_per_wheel / wheels_per_bicycle = 4 := by
  sorry

end num_bicycles_l2182_218220


namespace present_age_of_son_l2182_218248

/-- A man is 46 years older than his son and in two years, the man's age will be twice the age of his son. Prove that the present age of the son is 44. -/
theorem present_age_of_son (M S : ℕ) (h1 : M = S + 46) (h2 : M + 2 = 2 * (S + 2)) : S = 44 :=
by {
  sorry
}

end present_age_of_son_l2182_218248


namespace problem_solution_l2182_218299
open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.foldl (· + ·) 0

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_seq : ℕ → ℕ → ℕ
| 0, n => f n
| (k+1), n => f (f_seq k n)

theorem problem_solution :
  f_seq 2016 9 = 8 :=
sorry

end problem_solution_l2182_218299


namespace total_revenue_is_405_l2182_218281

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

end total_revenue_is_405_l2182_218281


namespace optimal_price_l2182_218209

def monthly_sales (p : ℝ) : ℝ := 150 - 6 * p
def break_even (p : ℝ) : Prop := 40 ≤ monthly_sales p
def revenue (p : ℝ) : ℝ := p * monthly_sales p

theorem optimal_price : ∃ p : ℝ, p = 13 ∧ p ≤ 30 ∧ break_even p ∧ ∀ q : ℝ, q ≤ 30 → break_even q → revenue p ≥ revenue q := 
by
  sorry

end optimal_price_l2182_218209


namespace probability_neither_event_l2182_218225

theorem probability_neither_event (P_A P_B P_A_and_B : ℝ)
  (h1 : P_A = 0.25)
  (h2 : P_B = 0.40)
  (h3 : P_A_and_B = 0.20) :
  1 - (P_A + P_B - P_A_and_B) = 0.55 :=
by
  sorry

end probability_neither_event_l2182_218225


namespace cost_two_enchiladas_two_tacos_three_burritos_l2182_218272

variables (e t b : ℝ)

theorem cost_two_enchiladas_two_tacos_three_burritos 
  (h1 : 2 * e + 3 * t + b = 5.00)
  (h2 : 3 * e + 2 * t + 2 * b = 7.50) : 
  2 * e + 2 * t + 3 * b = 10.625 :=
sorry

end cost_two_enchiladas_two_tacos_three_burritos_l2182_218272


namespace inverse_composition_has_correct_value_l2182_218273

noncomputable def f (x : ℝ) : ℝ := 5 * x + 7
noncomputable def f_inv (x : ℝ) : ℝ := (x - 7) / 5

theorem inverse_composition_has_correct_value : 
  f_inv (f_inv 9) = -33 / 25 := 
by 
  sorry

end inverse_composition_has_correct_value_l2182_218273


namespace number_of_foons_correct_l2182_218293

-- Define the conditions
def area : ℝ := 5  -- Area in cm^2
def thickness : ℝ := 0.5  -- Thickness in cm
def total_volume : ℝ := 50  -- Total volume in cm^3

-- Define the proof problem
theorem number_of_foons_correct :
  (total_volume / (area * thickness) = 20) :=
by
  -- The necessary computation would go here, but for now we'll use sorry to indicate the outcome
  sorry

end number_of_foons_correct_l2182_218293


namespace marla_colors_green_squares_l2182_218212

-- Condition 1: Grid dimensions
def num_rows : ℕ := 10
def num_cols : ℕ := 15

-- Condition 2: Red squares
def red_rows : ℕ := 4
def red_squares_per_row : ℕ := 6
def red_squares : ℕ := red_rows * red_squares_per_row

-- Condition 3: Blue rows (first 2 and last 2)
def blue_rows : ℕ := 2 + 2
def blue_squares_per_row : ℕ := num_cols
def blue_squares : ℕ := blue_rows * blue_squares_per_row

-- Derived information
def total_squares : ℕ := num_rows * num_cols
def non_green_squares : ℕ := red_squares + blue_squares

-- The Lemma to prove
theorem marla_colors_green_squares : total_squares - non_green_squares = 66 := by
  sorry

end marla_colors_green_squares_l2182_218212


namespace arithmetic_sequence_a3_l2182_218252

theorem arithmetic_sequence_a3 :
  ∃ (a : ℕ → ℝ) (d : ℝ), 
    (∀ n, a n = 2 + (n - 1) * d) ∧
    (a 1 = 2) ∧
    (a 5 = a 4 + 2) →
    a 3 = 6 :=
sorry

end arithmetic_sequence_a3_l2182_218252


namespace point_in_fourth_quadrant_l2182_218287

-- Define complex number and evaluate it
noncomputable def z : ℂ := (2 - (1 : ℂ) * Complex.I) / (1 + (1 : ℂ) * Complex.I)

-- Prove that the complex number z lies in the fourth quadrant
theorem point_in_fourth_quadrant (hz: z = (1/2 : ℂ) - (3/2 : ℂ) * Complex.I) : z.im < 0 ∧ z.re > 0 :=
by
  -- Skipping the proof here
  sorry

end point_in_fourth_quadrant_l2182_218287


namespace quotient_multiple_of_y_l2182_218230

theorem quotient_multiple_of_y (x y m : ℤ) (h1 : x = 11 * y + 4) (h2 : 2 * x = 8 * m * y + 3) (h3 : 13 * y - x = 1) : m = 3 :=
by
  sorry

end quotient_multiple_of_y_l2182_218230


namespace average_marks_correct_l2182_218288

-- Define the marks obtained in each subject
def english_marks := 86
def mathematics_marks := 85
def physics_marks := 92
def chemistry_marks := 87
def biology_marks := 95

-- Calculate total marks and average marks
def total_marks := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects := 5
def average_marks := total_marks / num_subjects

-- Prove that Dacid's average marks are 89
theorem average_marks_correct : average_marks = 89 := by
  sorry

end average_marks_correct_l2182_218288


namespace roots_of_polynomial_l2182_218200

theorem roots_of_polynomial (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end roots_of_polynomial_l2182_218200


namespace find_ABC_l2182_218243

theorem find_ABC {A B C : ℕ} (h₀ : ∀ n : ℕ, n ≤ 9 → n ≤ 9) (h₁ : 0 ≤ A) (h₂ : A ≤ 9) 
  (h₃ : 0 ≤ B) (h₄ : B ≤ 9) (h₅ : 0 ≤ C) (h₆ : C ≤ 9) (h₇ : 100 * A + 10 * B + C = B^C - A) :
  100 * A + 10 * B + C = 127 := by {
  sorry
}

end find_ABC_l2182_218243


namespace factorization_correct_l2182_218258

theorem factorization_correct (a : ℝ) : a^2 - 2 * a - 15 = (a + 3) * (a - 5) := 
by 
  sorry

end factorization_correct_l2182_218258


namespace quadratic_transformation_l2182_218229

theorem quadratic_transformation :
  ∀ x : ℝ, (x^2 - 6 * x - 5 = 0) → ((x - 3)^2 = 14) :=
by
  intros x h
  sorry

end quadratic_transformation_l2182_218229


namespace eval_expression_l2182_218263

theorem eval_expression : 
  (8^5) / (4 * 2^5 + 16) = 2^11 / 9 :=
by
  sorry

end eval_expression_l2182_218263


namespace average_age_of_team_l2182_218286

variable (A : ℕ)
variable (captain_age : ℕ)
variable (wicket_keeper_age : ℕ)
variable (vice_captain_age : ℕ)

-- Conditions
def team_size := 11
def captain := 25
def wicket_keeper := captain + 3
def vice_captain := wicket_keeper - 4
def remaining_players := team_size - 3
def remaining_average := A - 1

-- Prove the average age of the whole team
theorem average_age_of_team :
  captain_age = 25 ∧
  wicket_keeper_age = captain_age + 3 ∧
  vice_captain_age = wicket_keeper_age - 4 ∧
  11 * A = (captain + wicket_keeper + vice_captain) + 8 * (A - 1) → 
  A = 23 :=
by
  sorry

end average_age_of_team_l2182_218286


namespace bars_per_set_correct_l2182_218291

-- Define the total number of metal bars and the number of sets
def total_metal_bars : ℕ := 14
def number_of_sets : ℕ := 2

-- Define the function to compute bars per set
def bars_per_set (total_bars : ℕ) (sets : ℕ) : ℕ :=
  total_bars / sets

-- The proof statement
theorem bars_per_set_correct : bars_per_set total_metal_bars number_of_sets = 7 := by
  sorry

end bars_per_set_correct_l2182_218291


namespace space_shuttle_speed_kmh_l2182_218298

-- Define the given conditions
def speedInKmPerSecond : ℕ := 4
def secondsInAnHour : ℕ := 3600

-- State the proof problem
theorem space_shuttle_speed_kmh : speedInKmPerSecond * secondsInAnHour = 14400 := by
  sorry

end space_shuttle_speed_kmh_l2182_218298


namespace Heesu_has_greatest_sum_l2182_218218

-- Define the numbers collected by each individual
def Sora_collected : (Nat × Nat) := (4, 6)
def Heesu_collected : (Nat × Nat) := (7, 5)
def Jiyeon_collected : (Nat × Nat) := (3, 8)

-- Calculate the sums
def Sora_sum : Nat := Sora_collected.1 + Sora_collected.2
def Heesu_sum : Nat := Heesu_collected.1 + Heesu_collected.2
def Jiyeon_sum : Nat := Jiyeon_collected.1 + Jiyeon_collected.2

-- The theorem to prove that Heesu has the greatest sum
theorem Heesu_has_greatest_sum :
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  sorry

end Heesu_has_greatest_sum_l2182_218218


namespace remainder_145_mul_155_div_12_l2182_218235

theorem remainder_145_mul_155_div_12 : (145 * 155) % 12 = 11 := by
  sorry

end remainder_145_mul_155_div_12_l2182_218235


namespace walking_time_l2182_218231

noncomputable def time_to_reach_destination (mr_harris_speed : ℝ) (mr_harris_time_to_store : ℝ) (your_speed : ℝ) (distance_factor : ℝ) : ℝ :=
  let store_distance := mr_harris_speed * mr_harris_time_to_store
  let your_destination_distance := distance_factor * store_distance
  your_destination_distance / your_speed

theorem walking_time (mr_harris_speed your_speed : ℝ) (mr_harris_time_to_store : ℝ) (distance_factor : ℝ) (h_speed : your_speed = 2 * mr_harris_speed) (h_time : mr_harris_time_to_store = 2) (h_factor : distance_factor = 3) :
  time_to_reach_destination mr_harris_speed mr_harris_time_to_store your_speed distance_factor = 3 :=
by
  rw [h_time, h_speed, h_factor]
  -- calculations based on given conditions
  sorry

end walking_time_l2182_218231


namespace total_volume_of_cubes_l2182_218256

theorem total_volume_of_cubes (s : ℕ) (n : ℕ) (h_s : s = 5) (h_n : n = 4) : 
  n * s^3 = 500 :=
by
  sorry

end total_volume_of_cubes_l2182_218256


namespace chris_birthday_days_l2182_218204

theorem chris_birthday_days (mod : ℕ → ℕ → ℕ) (day_of_week : ℕ → ℕ) :
  (mod 75 7 = 5) ∧ (mod 30 7 = 2) →
  (day_of_week 0 = 1) →
  (day_of_week 75 = 6) ∧ (day_of_week 30 = 3) := 
sorry

end chris_birthday_days_l2182_218204


namespace product_units_tens_not_divisible_by_8_l2182_218240

theorem product_units_tens_not_divisible_by_8 :
  ¬ (1834 % 8 = 0) → (4 * 3 = 12) :=
by
  intro h
  exact (by norm_num : 4 * 3 = 12)

end product_units_tens_not_divisible_by_8_l2182_218240


namespace find_levels_satisfying_surface_area_conditions_l2182_218207

theorem find_levels_satisfying_surface_area_conditions (n : ℕ) :
  let A_total_lateral := n * (n + 1) * Real.pi
  let A_total_vertical := Real.pi * n^2
  let A_total := n * (3 * n + 1) * Real.pi
  A_total_lateral = 0.35 * A_total → n = 13 :=
by
  intros A_total_lateral A_total_vertical A_total h
  sorry

end find_levels_satisfying_surface_area_conditions_l2182_218207


namespace investment_amount_correct_l2182_218269

-- Lean statement definitions based on conditions
def cost_per_tshirt : ℕ := 3
def selling_price_per_tshirt : ℕ := 20
def tshirts_sold : ℕ := 83
def total_revenue : ℕ := tshirts_sold * selling_price_per_tshirt
def total_cost_of_tshirts : ℕ := tshirts_sold * cost_per_tshirt
def investment_in_equipment : ℕ := total_revenue - total_cost_of_tshirts

-- Theorem statement
theorem investment_amount_correct : investment_in_equipment = 1411 := by
  sorry

end investment_amount_correct_l2182_218269


namespace cuberoot_condition_l2182_218232

/-- If \(\sqrt[3]{x-1}=3\), then \((x-1)^2 = 729\). -/
theorem cuberoot_condition (x : ℝ) (h : (x - 1)^(1/3) = 3) : (x - 1)^2 = 729 := 
  sorry

end cuberoot_condition_l2182_218232


namespace suraj_avg_after_10th_inning_l2182_218292

theorem suraj_avg_after_10th_inning (A : ℝ) 
  (h1 : ∀ A : ℝ, (9 * A + 200) / 10 = A + 8) :
  ∀ A : ℝ, A = 120 → (A + 8 = 128) :=
by
  sorry

end suraj_avg_after_10th_inning_l2182_218292


namespace students_after_last_stop_on_mondays_and_wednesdays_students_after_last_stop_on_tuesdays_and_thursdays_students_after_last_stop_on_fridays_l2182_218238

structure BusRoute where
  first_stop : Nat
  second_stop_on : Nat
  second_stop_off : Nat
  third_stop_on : Nat
  third_stop_off : Nat
  fourth_stop_on : Nat
  fourth_stop_off : Nat

def mondays_and_wednesdays := BusRoute.mk 39 29 12 35 18 27 15
def tuesdays_and_thursdays := BusRoute.mk 39 33 10 5 0 8 4
def fridays := BusRoute.mk 39 25 10 40 20 10 5

def students_after_last_stop (route : BusRoute) : Nat :=
  let stop1 := route.first_stop
  let stop2 := stop1 + route.second_stop_on - route.second_stop_off
  let stop3 := stop2 + route.third_stop_on - route.third_stop_off
  stop3 + route.fourth_stop_on - route.fourth_stop_off

theorem students_after_last_stop_on_mondays_and_wednesdays :
  students_after_last_stop mondays_and_wednesdays = 85 := by
  sorry

theorem students_after_last_stop_on_tuesdays_and_thursdays :
  students_after_last_stop tuesdays_and_thursdays = 71 := by
  sorry

theorem students_after_last_stop_on_fridays :
  students_after_last_stop fridays = 79 := by
  sorry

end students_after_last_stop_on_mondays_and_wednesdays_students_after_last_stop_on_tuesdays_and_thursdays_students_after_last_stop_on_fridays_l2182_218238


namespace proof_problem_l2182_218268

variables (α : ℝ)

-- Condition: tan(α) = 2
def tan_condition : Prop := Real.tan α = 2

-- First expression: (sin α + 2 cos α) / (4 cos α - sin α) = 2
def expression1 : Prop := (Real.sin α + 2 * Real.cos α) / (4 * Real.cos α - Real.sin α) = 2

-- Second expression: sqrt(2) * sin(2α + π/4) + 1 = 6/5
def expression2 : Prop := Real.sqrt 2 * Real.sin (2 * α + Real.pi / 4) + 1 = 6 / 5

-- Theorem: Prove the expressions given the condition
theorem proof_problem :
  tan_condition α → expression1 α ∧ expression2 α :=
by
  intro tan_cond
  have h1 : expression1 α := sorry
  have h2 : expression2 α := sorry
  exact ⟨h1, h2⟩

end proof_problem_l2182_218268
