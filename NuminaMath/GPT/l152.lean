import Mathlib

namespace NUMINAMATH_GPT_no_solution_to_inequality_l152_15299

theorem no_solution_to_inequality (x : ℝ) (h : x ≥ -1/4) : ¬(-1 - 1 / (3 * x + 4) < 2) :=
by sorry

end NUMINAMATH_GPT_no_solution_to_inequality_l152_15299


namespace NUMINAMATH_GPT_abs_m_plus_one_l152_15272

theorem abs_m_plus_one (m : ℝ) (h : |m| = m + 1) : (4 * m - 1) ^ 4 = 81 := by
  sorry

end NUMINAMATH_GPT_abs_m_plus_one_l152_15272


namespace NUMINAMATH_GPT_mushrooms_used_by_Karla_correct_l152_15222

-- Given conditions
def mushrooms_cut_each_mushroom : ℕ := 4
def mushrooms_cut_total : ℕ := 22 * mushrooms_cut_each_mushroom
def mushrooms_used_by_Kenny : ℕ := 38
def mushrooms_remaining : ℕ := 8
def mushrooms_total_used_by_Kenny_and_remaining : ℕ := mushrooms_used_by_Kenny + mushrooms_remaining
def mushrooms_used_by_Karla : ℕ := mushrooms_cut_total - mushrooms_total_used_by_Kenny_and_remaining

-- Statement to prove
theorem mushrooms_used_by_Karla_correct :
  mushrooms_used_by_Karla = 42 :=
by
  sorry

end NUMINAMATH_GPT_mushrooms_used_by_Karla_correct_l152_15222


namespace NUMINAMATH_GPT_discriminant_is_four_l152_15237

-- Define the quadratic equation components
def quadratic_a (a : ℝ) := 1
def quadratic_b (a : ℝ) := 2 * a
def quadratic_c (a : ℝ) := a^2 - 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) := quadratic_b a ^ 2 - 4 * quadratic_a a * quadratic_c a

-- Statement to prove: The discriminant is 4
theorem discriminant_is_four (a : ℝ) : discriminant a = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_discriminant_is_four_l152_15237


namespace NUMINAMATH_GPT_remainder_of_2_pow_33_mod_9_l152_15278

theorem remainder_of_2_pow_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end NUMINAMATH_GPT_remainder_of_2_pow_33_mod_9_l152_15278


namespace NUMINAMATH_GPT_circle_equation_standard_l152_15205

open Real

noncomputable def equation_of_circle : Prop :=
  ∃ R : ℝ, R = sqrt 2 ∧ 
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 → x + y - 2 = 0 → 0 ≤ x ∧ x ≤ 2)

theorem circle_equation_standard :
    equation_of_circle := sorry

end NUMINAMATH_GPT_circle_equation_standard_l152_15205


namespace NUMINAMATH_GPT_neha_mother_age_l152_15252

variable (N M : ℕ)

theorem neha_mother_age (h1 : M - 12 = 4 * (N - 12)) (h2 : M + 12 = 2 * (N + 12)) : M = 60 := by
  sorry

end NUMINAMATH_GPT_neha_mother_age_l152_15252


namespace NUMINAMATH_GPT_mean_score_74_l152_15257

theorem mean_score_74 
  (M SD : ℝ)
  (h1 : 58 = M - 2 * SD)
  (h2 : 98 = M + 3 * SD) : 
  M = 74 :=
by
  sorry

end NUMINAMATH_GPT_mean_score_74_l152_15257


namespace NUMINAMATH_GPT_second_hose_correct_l152_15267

/-- Define the problem parameters -/
def first_hose_rate : ℕ := 50
def initial_hours : ℕ := 3
def additional_hours : ℕ := 2
def total_capacity : ℕ := 390

/-- Define the total hours the first hose was used -/
def total_hours (initial_hours additional_hours : ℕ) : ℕ := initial_hours + additional_hours

/-- Define the amount of water sprayed by the first hose -/
def first_hose_total (first_hose_rate initial_hours additional_hours : ℕ) : ℕ :=
  first_hose_rate * (initial_hours + additional_hours)

/-- Define the remaining water needed to fill the pool -/
def remaining_water (total_capacity first_hose_total : ℕ) : ℕ :=
  total_capacity - first_hose_total

/-- Define the additional water sprayed by the first hose during the last 2 hours -/
def additional_first_hose (first_hose_rate additional_hours : ℕ) : ℕ :=
  first_hose_rate * additional_hours

/-- Define the water sprayed by the second hose -/
def second_hose_total (remaining_water additional_first_hose : ℕ) : ℕ :=
  remaining_water - additional_first_hose

/-- Define the rate of the second hose (output) -/
def second_hose_rate (second_hose_total additional_hours : ℕ) : ℕ :=
  second_hose_total / additional_hours

/-- Define the theorem we want to prove -/
theorem second_hose_correct :
  second_hose_rate
    (second_hose_total
        (remaining_water total_capacity (first_hose_total first_hose_rate initial_hours additional_hours))
        (additional_first_hose first_hose_rate additional_hours))
    additional_hours = 20 := by
  sorry

end NUMINAMATH_GPT_second_hose_correct_l152_15267


namespace NUMINAMATH_GPT_tangent_line_exponential_passing_through_origin_l152_15243

theorem tangent_line_exponential_passing_through_origin :
  ∃ (p : ℝ × ℝ) (m : ℝ), 
  (p = (1, Real.exp 1)) ∧ (m = Real.exp 1) ∧ 
  (∀ x : ℝ, x ≠ 1 → ¬ (∃ k : ℝ, k = (Real.exp x - 0) / (x - 0) ∧ k = Real.exp x)) :=
by 
  sorry

end NUMINAMATH_GPT_tangent_line_exponential_passing_through_origin_l152_15243


namespace NUMINAMATH_GPT_matrix_power_application_l152_15200

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
variable (v : Fin 2 → ℝ := ![4, -3])

theorem matrix_power_application :
  (B.mulVec v = ![8, -6]) →
  (B ^ 4).mulVec v = ![64, -48] :=
by
  intro h
  sorry

end NUMINAMATH_GPT_matrix_power_application_l152_15200


namespace NUMINAMATH_GPT_parabola_complementary_slope_l152_15285

theorem parabola_complementary_slope
  (p x0 y0 x1 y1 x2 y2 : ℝ)
  (hp : p > 0)
  (hy0 : y0 > 0)
  (hP : y0^2 = 2 * p * x0)
  (hA : y1^2 = 2 * p * x1)
  (hB : y2^2 = 2 * p * x2)
  (h_slopes : (y1 - y0) / (x1 - x0) = - (2 * p / (y2 + y0))) :
  (y1 + y2) / y0 = -2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_complementary_slope_l152_15285


namespace NUMINAMATH_GPT_merchant_gross_profit_l152_15218

-- Define the purchase price and markup rate
def purchase_price : ℝ := 42
def markup_rate : ℝ := 0.30
def discount_rate : ℝ := 0.20

-- Define the selling price equation given the purchase price and markup rate
def selling_price (S : ℝ) : Prop := S = purchase_price + markup_rate * S

-- Define the discounted selling price given the selling price and discount rate
def discounted_selling_price (S : ℝ) : ℝ := S - discount_rate * S

-- Define the gross profit as the difference between the discounted selling price and purchase price
def gross_profit (S : ℝ) : ℝ := discounted_selling_price S - purchase_price

theorem merchant_gross_profit : ∃ S : ℝ, selling_price S ∧ gross_profit S = 6 :=
by
  sorry

end NUMINAMATH_GPT_merchant_gross_profit_l152_15218


namespace NUMINAMATH_GPT_find_d_over_a_l152_15286

variable (a b c d : ℚ)

-- Conditions
def condition1 : Prop := a / b = 8
def condition2 : Prop := c / b = 4
def condition3 : Prop := c / d = 2 / 3

-- Theorem statement
theorem find_d_over_a (h1 : condition1 a b) (h2 : condition2 c b) (h3 : condition3 c d) : d / a = 3 / 4 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_find_d_over_a_l152_15286


namespace NUMINAMATH_GPT_collinear_points_count_l152_15260

-- Definitions for the problem conditions
def vertices_count := 8
def midpoints_count := 12
def face_centers_count := 6
def cube_center_count := 1
def total_points_count := vertices_count + midpoints_count + face_centers_count + cube_center_count

-- Lean statement to express the proof problem
theorem collinear_points_count :
  (total_points_count = 27) →
  (vertices_count = 8) →
  (midpoints_count = 12) →
  (face_centers_count = 6) →
  (cube_center_count = 1) →
  ∃ n, n = 49 :=
by
  intros
  existsi 49
  sorry

end NUMINAMATH_GPT_collinear_points_count_l152_15260


namespace NUMINAMATH_GPT_percentage_increase_l152_15255

theorem percentage_increase (original_interval : ℕ) (new_interval : ℕ) 
  (h1 : original_interval = 30) (h2 : new_interval = 45) :
  ((new_interval - original_interval) / original_interval) * 100 = 50 := 
by 
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_percentage_increase_l152_15255


namespace NUMINAMATH_GPT_arcsin_sqrt_three_over_two_l152_15242

theorem arcsin_sqrt_three_over_two : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_arcsin_sqrt_three_over_two_l152_15242


namespace NUMINAMATH_GPT_count_distinct_digits_l152_15249

theorem count_distinct_digits (n : ℕ) (h1 : ∃ (n : ℕ), n^3 = 125) : 
  n = 5 :=
by
  sorry

end NUMINAMATH_GPT_count_distinct_digits_l152_15249


namespace NUMINAMATH_GPT_inequality_proof_l152_15250

theorem inequality_proof (x y : ℝ) (h1 : x > y) (h2 : y > 2 / (x - y)) : x^2 > y^2 + 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l152_15250


namespace NUMINAMATH_GPT_root_interval_l152_15296

noncomputable def f (a b x : ℝ) : ℝ := 2 * a^x - b^x

theorem root_interval (a b : ℝ) (h₀ : 0 < a) (h₁ : b ≥ 2 * a) :
  ∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a b x = 0 := 
sorry

end NUMINAMATH_GPT_root_interval_l152_15296


namespace NUMINAMATH_GPT_solve_stream_speed_l152_15246

noncomputable def boat_travel (v : ℝ) : Prop :=
  let downstream_speed := 12 + v
  let upstream_speed := 12 - v
  let downstream_time := 60 / downstream_speed
  let upstream_time := 60 / upstream_speed
  upstream_time - downstream_time = 2

theorem solve_stream_speed : ∃ v : ℝ, boat_travel v ∧ v = 2.31 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_stream_speed_l152_15246


namespace NUMINAMATH_GPT_find_values_l152_15247

noncomputable def value_of_a (a : ℚ) : Prop :=
  4 + a = 2

noncomputable def value_of_b (b : ℚ) : Prop :=
  b^2 - 2 * b = 24 ∧ 4 * b^2 - 2 * b = 72

theorem find_values (a b : ℚ) (h1 : value_of_a a) (h2 : value_of_b b) :
  a = -2 ∧ b = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_values_l152_15247


namespace NUMINAMATH_GPT_trig_identity_proof_l152_15262

theorem trig_identity_proof : 
  (Real.cos (70 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) + 
   Real.cos (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l152_15262


namespace NUMINAMATH_GPT_greatest_prime_divisor_digits_sum_l152_15269

theorem greatest_prime_divisor_digits_sum (h : 8191 = 2^13 - 1) : (1 + 2 + 7) = 10 :=
by
  sorry

end NUMINAMATH_GPT_greatest_prime_divisor_digits_sum_l152_15269


namespace NUMINAMATH_GPT_female_kittens_count_l152_15281

theorem female_kittens_count (initial_cats total_cats male_kittens female_kittens : ℕ)
  (h1 : initial_cats = 2)
  (h2 : total_cats = 7)
  (h3 : male_kittens = 2)
  (h4 : female_kittens = total_cats - initial_cats - male_kittens) :
  female_kittens = 3 :=
by
  sorry

end NUMINAMATH_GPT_female_kittens_count_l152_15281


namespace NUMINAMATH_GPT_percentage_of_teachers_with_neither_issue_l152_15288

theorem percentage_of_teachers_with_neither_issue 
  (total_teachers : ℕ)
  (teachers_with_bp : ℕ)
  (teachers_with_stress : ℕ)
  (teachers_with_both : ℕ)
  (h1 : total_teachers = 150)
  (h2 : teachers_with_bp = 90)
  (h3 : teachers_with_stress = 60)
  (h4 : teachers_with_both = 30) :
  let neither_issue_teachers := total_teachers - (teachers_with_bp + teachers_with_stress - teachers_with_both)
  let percentage := (neither_issue_teachers * 100) / total_teachers
  percentage = 20 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_percentage_of_teachers_with_neither_issue_l152_15288


namespace NUMINAMATH_GPT_C_should_pay_correct_amount_l152_15232

def A_oxen_months : ℕ := 10 * 7
def B_oxen_months : ℕ := 12 * 5
def C_oxen_months : ℕ := 15 * 3
def D_oxen_months : ℕ := 20 * 6

def total_rent : ℚ := 225

def C_share_of_rent : ℚ :=
  total_rent * (C_oxen_months : ℚ) / (A_oxen_months + B_oxen_months + C_oxen_months + D_oxen_months)

theorem C_should_pay_correct_amount : C_share_of_rent = 225 * (45 : ℚ) / 295 := by
  sorry

end NUMINAMATH_GPT_C_should_pay_correct_amount_l152_15232


namespace NUMINAMATH_GPT_total_hours_is_900_l152_15201

-- Definitions for the video length, speeds, and number of videos watched
def video_length : ℕ := 100
def lila_speed : ℕ := 2
def roger_speed : ℕ := 1
def num_videos : ℕ := 6

-- Definition of total hours watched
def total_hours_watched : ℕ :=
  let lila_time_per_video := video_length / lila_speed
  let roger_time_per_video := video_length / roger_speed
  (lila_time_per_video * num_videos) + (roger_time_per_video * num_videos)

-- Prove that the total hours watched is 900
theorem total_hours_is_900 : total_hours_watched = 900 :=
by
  -- Proving the equation step-by-step
  sorry

end NUMINAMATH_GPT_total_hours_is_900_l152_15201


namespace NUMINAMATH_GPT_hyperbola_constant_ellipse_constant_l152_15202

variables {a b : ℝ} (a_pos_b_gt_a : 0 < a ∧ a < b)
variables {A B : ℝ × ℝ} (on_hyperbola_A : A.1^2 / a^2 - A.2^2 / b^2 = 1)
variables (on_hyperbola_B : B.1^2 / a^2 - B.2^2 / b^2 = 1) (perp_OA_OB : A.1 * B.1 + A.2 * B.2 = 0)

-- Hyperbola statement
theorem hyperbola_constant :
  (1 / (A.1^2 + A.2^2)) + (1 / (B.1^2 + B.2^2)) = 1 / a^2 - 1 / b^2 :=
sorry

variables {C D : ℝ × ℝ} (on_ellipse_C : C.1^2 / a^2 + C.2^2 / b^2 = 1)
variables (on_ellipse_D : D.1^2 / a^2 + D.2^2 / b^2 = 1) (perp_OC_OD : C.1 * D.1 + C.2 * D.2 = 0)

-- Ellipse statement
theorem ellipse_constant :
  (1 / (C.1^2 + C.2^2)) + (1 / (D.1^2 + D.2^2)) = 1 / a^2 + 1 / b^2 :=
sorry

end NUMINAMATH_GPT_hyperbola_constant_ellipse_constant_l152_15202


namespace NUMINAMATH_GPT_exists_f_prime_eq_inverses_l152_15258

theorem exists_f_prime_eq_inverses (f : ℝ → ℝ) (a b : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : ContinuousOn f (Set.Icc a b))
  (h4 : DifferentiableOn ℝ f (Set.Ioo a b)) :
  ∃ c ∈ Set.Ioo a b, (deriv f c) = (1 / (a - c)) + (1 / (b - c)) + (1 / (a + b)) :=
by
  sorry

end NUMINAMATH_GPT_exists_f_prime_eq_inverses_l152_15258


namespace NUMINAMATH_GPT_average_students_per_bus_l152_15203

-- Definitions
def total_students : ℕ := 396
def students_in_cars : ℕ := 18
def number_of_buses : ℕ := 7

-- Proof problem statement
theorem average_students_per_bus : (total_students - students_in_cars) / number_of_buses = 54 := by
  sorry

end NUMINAMATH_GPT_average_students_per_bus_l152_15203


namespace NUMINAMATH_GPT_total_marks_l152_15297

-- Variables and conditions
variables (M C P : ℕ)
variable (h1 : C = P + 20)
variable (h2 : (M + C) / 2 = 40)

-- Theorem statement
theorem total_marks (M C P : ℕ) (h1 : C = P + 20) (h2 : (M + C) / 2 = 40) : M + P = 60 :=
sorry

end NUMINAMATH_GPT_total_marks_l152_15297


namespace NUMINAMATH_GPT_intersection_points_of_circle_and_vertical_line_l152_15244

theorem intersection_points_of_circle_and_vertical_line :
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (3, y1) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 } ∧ 
                    (3, y2) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 } ∧ 
                    (3, y1) ≠ (3, y2)) := 
by
  sorry

end NUMINAMATH_GPT_intersection_points_of_circle_and_vertical_line_l152_15244


namespace NUMINAMATH_GPT_black_female_pigeons_more_than_males_l152_15230

theorem black_female_pigeons_more_than_males:
  let total_pigeons := 70
  let black_pigeons := total_pigeons / 2
  let black_male_percentage := 20 / 100
  let black_male_pigeons := black_pigeons * black_male_percentage
  let black_female_pigeons := black_pigeons - black_male_pigeons
  black_female_pigeons - black_male_pigeons = 21 := by
{
  let total_pigeons := 70
  let black_pigeons := total_pigeons / 2
  let black_male_percentage := 20 / 100
  let black_male_pigeons := black_pigeons * black_male_percentage
  let black_female_pigeons := black_pigeons - black_male_pigeons
  show black_female_pigeons - black_male_pigeons = 21
  sorry
}

end NUMINAMATH_GPT_black_female_pigeons_more_than_males_l152_15230


namespace NUMINAMATH_GPT_product_mod_17_eq_zero_l152_15226

theorem product_mod_17_eq_zero :
    (2001 * 2002 * 2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 := by
  sorry

end NUMINAMATH_GPT_product_mod_17_eq_zero_l152_15226


namespace NUMINAMATH_GPT_relationship_among_a_ae_ea_minus_one_l152_15298

theorem relationship_among_a_ae_ea_minus_one (a : ℝ) (h : 0 < a ∧ a < 1) :
  (Real.exp a - 1 > a ∧ a > Real.exp a - 1 ∧ a > a^(Real.exp 1)) :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_a_ae_ea_minus_one_l152_15298


namespace NUMINAMATH_GPT_cos_values_l152_15219

theorem cos_values (n : ℤ) : (0 ≤ n ∧ n ≤ 360) ∧ (Real.cos (n * Real.pi / 180) = Real.cos (310 * Real.pi / 180)) ↔ (n = 50 ∨ n = 310) :=
by
  sorry

end NUMINAMATH_GPT_cos_values_l152_15219


namespace NUMINAMATH_GPT_log_three_nine_cubed_l152_15266

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end NUMINAMATH_GPT_log_three_nine_cubed_l152_15266


namespace NUMINAMATH_GPT_cos_double_angle_l152_15284

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2*n) = 8) :
  Real.cos (2 * θ) = 3 / 4 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l152_15284


namespace NUMINAMATH_GPT_integer_solution_zero_l152_15256

theorem integer_solution_zero (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end NUMINAMATH_GPT_integer_solution_zero_l152_15256


namespace NUMINAMATH_GPT_radio_selling_price_l152_15239

noncomputable def sellingPrice (costPrice : ℝ) (lossPercentage : ℝ) : ℝ :=
  costPrice - (lossPercentage / 100 * costPrice)

theorem radio_selling_price :
  sellingPrice 490 5 = 465.5 :=
by
  sorry

end NUMINAMATH_GPT_radio_selling_price_l152_15239


namespace NUMINAMATH_GPT_PaulineDressCost_l152_15291

-- Lets define the variables for each dress cost
variable (P Jean Ida Patty : ℝ)

-- Condition statements
def condition1 : Prop := Patty = Ida + 10
def condition2 : Prop := Ida = Jean + 30
def condition3 : Prop := Jean = P - 10
def condition4 : Prop := P + Jean + Ida + Patty = 160

-- The proof problem statement
theorem PaulineDressCost : 
  condition1 Patty Ida →
  condition2 Ida Jean →
  condition3 Jean P →
  condition4 P Jean Ida Patty →
  P = 30 := by
  sorry

end NUMINAMATH_GPT_PaulineDressCost_l152_15291


namespace NUMINAMATH_GPT_perimeter_square_III_l152_15228

theorem perimeter_square_III (perimeter_I perimeter_II : ℕ) (hI : perimeter_I = 12) (hII : perimeter_II = 24) : 
  let side_I := perimeter_I / 4 
  let side_II := perimeter_II / 4 
  let side_III := side_I + side_II 
  4 * side_III = 36 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_square_III_l152_15228


namespace NUMINAMATH_GPT_last_bead_color_is_blue_l152_15265

def bead_color_cycle := ["Red", "Orange", "Yellow", "Yellow", "Green", "Blue", "Purple"]

def bead_color (n : Nat) : String :=
  bead_color_cycle.get! (n % bead_color_cycle.length)

theorem last_bead_color_is_blue :
  bead_color 82 = "Blue" := 
by
  sorry

end NUMINAMATH_GPT_last_bead_color_is_blue_l152_15265


namespace NUMINAMATH_GPT_proof_l152_15276

noncomputable def question (a b c : ℂ) : ℂ := (a^3 + b^3 + c^3) / (a * b * c)

theorem proof (a b c : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 15)
  (h5 : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2 * a * b * c) :
  question a b c = 18 :=
by
  sorry

end NUMINAMATH_GPT_proof_l152_15276


namespace NUMINAMATH_GPT_solve_for_y_l152_15240

theorem solve_for_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y) = 8) : y = 1 :=
by {
  -- Apply the conditions and derive the proof
  sorry
}

end NUMINAMATH_GPT_solve_for_y_l152_15240


namespace NUMINAMATH_GPT_largest_number_after_removal_l152_15254

theorem largest_number_after_removal :
  ∀ (s : Nat), s = 1234567891011121314151617181920 -- representing the start of the sequence
  → true
  := by
    sorry

end NUMINAMATH_GPT_largest_number_after_removal_l152_15254


namespace NUMINAMATH_GPT_mayor_cup_num_teams_l152_15279

theorem mayor_cup_num_teams (x : ℕ) (h : x * (x - 1) / 2 = 21) : 
    ∃ x, x * (x - 1) / 2 = 21 := 
by
  sorry

end NUMINAMATH_GPT_mayor_cup_num_teams_l152_15279


namespace NUMINAMATH_GPT_per_capita_income_growth_l152_15225

noncomputable def income2020 : ℝ := 3.2
noncomputable def income2022 : ℝ := 3.7
variable (x : ℝ)

/--
Prove the per capita disposable income model.
-/
theorem per_capita_income_growth :
  income2020 * (1 + x)^2 = income2022 :=
sorry

end NUMINAMATH_GPT_per_capita_income_growth_l152_15225


namespace NUMINAMATH_GPT_simplify_expression_l152_15216

variable (x y z : ℝ)

noncomputable def expr1 := (3 * x + y / 3 + 2 * z)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹ + (2 * z)⁻¹)
noncomputable def expr2 := (2 * y + 18 * x * z + 3 * z * x) / (6 * x * y * z * (9 * x + y + 6 * z))

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxyz : 3 * x + y / 3 + 2 * z ≠ 0) :
  expr1 x y z = expr2 x y z := by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l152_15216


namespace NUMINAMATH_GPT_total_students_correct_l152_15217

theorem total_students_correct (H : ℕ)
  (B : ℕ := 2 * H)
  (P : ℕ := H + 5)
  (S : ℕ := 3 * (H + 5))
  (h1 : B = 30)
  : (B + H + P + S) = 125 := by
  sorry

end NUMINAMATH_GPT_total_students_correct_l152_15217


namespace NUMINAMATH_GPT_games_within_division_l152_15234

/-- 
Given a baseball league with two four-team divisions,
where each team plays N games against other teams in its division,
and M games against teams in the other division.
Given that N > 2M and M > 6, and each team plays a total of 92 games in a season,
prove that each team plays 60 games within its own division.
-/
theorem games_within_division (N M : ℕ) (hN : N > 2 * M) (hM : M > 6) (h_total : 3 * N + 4 * M = 92) :
  3 * N = 60 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_games_within_division_l152_15234


namespace NUMINAMATH_GPT_find_ratio_l152_15241

variables {a b c d : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
variables (h5 : (5 * a + b) / (5 * c + d) = (6 * a + b) / (6 * c + d))
variables (h6 : (7 * a + b) / (7 * c + d) = 9)

theorem find_ratio (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
    (h5 : (5 * a + b) / (5 * c + d) = (6 * a + b) / (6 * c + d))
    (h6 : (7 * a + b) / (7 * c + d) = 9) :
    (9 * a + b) / (9 * c + d) = 9 := 
by {
    sorry
}

end NUMINAMATH_GPT_find_ratio_l152_15241


namespace NUMINAMATH_GPT_all_d_zero_l152_15209

def d (n m : ℕ) : ℤ := sorry -- or some explicit initial definition

theorem all_d_zero (n m : ℕ) (h₁ : n ≥ 0) (h₂ : 0 ≤ m) (h₃ : m ≤ n) :
  (m = 0 ∨ m = n → d n m = 0) ∧
  (0 < m ∧ m < n → m * d n m = m * d (n - 1) m + (2 * n - m) * d (n - 1) (m - 1))
:=
  sorry

end NUMINAMATH_GPT_all_d_zero_l152_15209


namespace NUMINAMATH_GPT_desired_percentage_of_alcohol_l152_15294

def solution_x_alcohol_by_volume : ℝ := 0.10
def solution_y_alcohol_by_volume : ℝ := 0.30
def volume_solution_x : ℝ := 200
def volume_solution_y : ℝ := 600

theorem desired_percentage_of_alcohol :
  ((solution_x_alcohol_by_volume * volume_solution_x + solution_y_alcohol_by_volume * volume_solution_y) / 
  (volume_solution_x + volume_solution_y)) * 100 = 25 := 
sorry

end NUMINAMATH_GPT_desired_percentage_of_alcohol_l152_15294


namespace NUMINAMATH_GPT_twenty_five_percent_of_x_l152_15268

-- Define the number x and the conditions
variable (x : ℝ)
variable (h : x - (3/4) * x = 100)

-- The theorem statement
theorem twenty_five_percent_of_x : (1/4) * x = 100 :=
by 
  -- Assume x satisfies the given condition
  sorry

end NUMINAMATH_GPT_twenty_five_percent_of_x_l152_15268


namespace NUMINAMATH_GPT_range_of_a_l152_15212

variables {a x : ℝ}

def P (a : ℝ) : Prop := ∀ x, ¬ (x^2 - (a + 1) * x + 1 ≤ 0)

def Q (a : ℝ) : Prop := ∀ x, |x - 1| ≥ a + 2

theorem range_of_a (a : ℝ) : 
  (¬ P a ∧ ¬ Q a) → a ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l152_15212


namespace NUMINAMATH_GPT_origami_papers_per_cousin_l152_15238

theorem origami_papers_per_cousin (total_papers : ℕ) (num_cousins : ℕ) (same_papers_each : ℕ) 
  (h1 : total_papers = 48) 
  (h2 : num_cousins = 6) 
  (h3 : same_papers_each = total_papers / num_cousins) : 
  same_papers_each = 8 := 
by 
  sorry

end NUMINAMATH_GPT_origami_papers_per_cousin_l152_15238


namespace NUMINAMATH_GPT_find_c_value_l152_15283

-- Given condition: x^2 + 300x + c = (x + a)^2
-- Problem statement: Prove that c = 22500 for the given conditions
theorem find_c_value (x a c : ℝ) : (x^2 + 300 * x + c = (x + 150)^2) → (c = 22500) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_c_value_l152_15283


namespace NUMINAMATH_GPT_mingi_initial_tomatoes_l152_15282

theorem mingi_initial_tomatoes (n m r : ℕ) (h1 : n = 15) (h2 : m = 20) (h3 : r = 6) : n * m + r = 306 := by
  sorry

end NUMINAMATH_GPT_mingi_initial_tomatoes_l152_15282


namespace NUMINAMATH_GPT_max_S_n_l152_15235

noncomputable def S (n : ℕ) : ℝ := sorry  -- Definition of the sum of the first n terms

theorem max_S_n (S : ℕ → ℝ) (h16 : S 16 > 0) (h17 : S 17 < 0) : ∃ n, S n = S 8 :=
sorry

end NUMINAMATH_GPT_max_S_n_l152_15235


namespace NUMINAMATH_GPT_combined_molecular_weight_l152_15215

theorem combined_molecular_weight {m1 m2 : ℕ} 
  (MW_C : ℝ) (MW_H : ℝ) (MW_O : ℝ)
  (Butanoic_acid : ℕ × ℕ × ℕ)
  (Propanoic_acid : ℕ × ℕ × ℕ)
  (MW_Butanoic_acid : ℝ)
  (MW_Propanoic_acid : ℝ)
  (weight_Butanoic_acid : ℝ)
  (weight_Propanoic_acid : ℝ)
  (total_weight : ℝ) :
MW_C = 12.01 → MW_H = 1.008 → MW_O = 16.00 →
Butanoic_acid = (4, 8, 2) → MW_Butanoic_acid = (4 * MW_C) + (8 * MW_H) + (2 * MW_O) →
Propanoic_acid = (3, 6, 2) → MW_Propanoic_acid = (3 * MW_C) + (6 * MW_H) + (2 * MW_O) →
m1 = 9 → weight_Butanoic_acid = m1 * MW_Butanoic_acid →
m2 = 5 → weight_Propanoic_acid = m2 * MW_Propanoic_acid →
total_weight = weight_Butanoic_acid + weight_Propanoic_acid →
total_weight = 1163.326 :=
by {
  intros;
  sorry
}

end NUMINAMATH_GPT_combined_molecular_weight_l152_15215


namespace NUMINAMATH_GPT_tan_sum_eq_one_l152_15253

theorem tan_sum_eq_one (a b : ℝ) (h1 : Real.tan a = 1 / 2) (h2 : Real.tan b = 1 / 3) :
    Real.tan (a + b) = 1 := 
by
  sorry

end NUMINAMATH_GPT_tan_sum_eq_one_l152_15253


namespace NUMINAMATH_GPT_negation_of_p_l152_15295

def p := ∃ n : ℕ, n^2 > 2 * n - 1

theorem negation_of_p : ¬ p ↔ ∀ n : ℕ, n^2 ≤ 2 * n - 1 :=
by sorry

end NUMINAMATH_GPT_negation_of_p_l152_15295


namespace NUMINAMATH_GPT_geometric_sequence_proof_l152_15271

-- Define a geometric sequence with first term 1 and common ratio q with |q| ≠ 1
noncomputable def geometric_sequence (q : ℝ) (n : ℕ) : ℝ :=
  if h : |q| ≠ 1 then (1 : ℝ) * q ^ (n - 1) else 0

-- m should be 11 given the conditions
theorem geometric_sequence_proof (q : ℝ) (m : ℕ) (h : |q| ≠ 1) 
  (hm : geometric_sequence q m = geometric_sequence q 1 * geometric_sequence q 2 * geometric_sequence q 3 * geometric_sequence q 4 * geometric_sequence q 5 ) : 
  m = 11 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_proof_l152_15271


namespace NUMINAMATH_GPT_novel_pages_total_l152_15207

-- Definitions based on conditions
def pages_first_two_days : ℕ := 2 * 50
def pages_next_four_days : ℕ := 4 * 25
def pages_six_days : ℕ := pages_first_two_days + pages_next_four_days
def pages_seventh_day : ℕ := 30
def total_pages : ℕ := pages_six_days + pages_seventh_day

-- Statement of the problem as a theorem in Lean 4
theorem novel_pages_total : total_pages = 230 := by
  sorry

end NUMINAMATH_GPT_novel_pages_total_l152_15207


namespace NUMINAMATH_GPT_rectangle_to_square_y_l152_15231

theorem rectangle_to_square_y (y : ℝ) (a b : ℝ) (s : ℝ) (h1 : a = 7) (h2 : b = 21)
  (h3 : s^2 = a * b) (h4 : y = s / 2) : y = 7 * Real.sqrt 3 / 2 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_rectangle_to_square_y_l152_15231


namespace NUMINAMATH_GPT_angle_remains_unchanged_l152_15259

-- Definition of magnification condition (though it does not affect angle in mathematics, we state it as given)
def magnifying_glass (magnification : ℝ) (initial_angle : ℝ) : ℝ := 
  initial_angle  -- Magnification does not change the angle in this context.

-- Given condition
def initial_angle : ℝ := 30

-- Theorem we want to prove
theorem angle_remains_unchanged (magnification : ℝ) (h_magnify : magnification = 100) :
  magnifying_glass magnification initial_angle = initial_angle :=
by
  sorry

end NUMINAMATH_GPT_angle_remains_unchanged_l152_15259


namespace NUMINAMATH_GPT_students_in_class_l152_15213

/-- Conditions:
1. 20 hands in Peter’s class, not including his.
2. Every student in the class has 2 hands.

Prove that the number of students in Peter’s class including him is 11.
-/
theorem students_in_class (hands_without_peter : ℕ) (hands_per_student : ℕ) (students_including_peter : ℕ) :
  hands_without_peter = 20 →
  hands_per_student = 2 →
  students_including_peter = (hands_without_peter + hands_per_student) / hands_per_student →
  students_including_peter = 11 :=
by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_students_in_class_l152_15213


namespace NUMINAMATH_GPT_tan_arith_seq_l152_15236

theorem tan_arith_seq (x y z : ℝ)
  (h₁ : y = x + π / 3)
  (h₂ : z = x + 2 * π / 3) :
  (Real.tan x * Real.tan y) + (Real.tan y * Real.tan z) + (Real.tan z * Real.tan x) = -3 :=
sorry

end NUMINAMATH_GPT_tan_arith_seq_l152_15236


namespace NUMINAMATH_GPT_volume_of_inscribed_sphere_l152_15223

noncomputable def volume_of_tetrahedron (R : ℝ) (S1 S2 S3 S4 : ℝ) : ℝ :=
  R * (S1 + S2 + S3 + S4)

theorem volume_of_inscribed_sphere (R S1 S2 S3 S4 V : ℝ) :
  V = R * (S1 + S2 + S3 + S4) :=
sorry

end NUMINAMATH_GPT_volume_of_inscribed_sphere_l152_15223


namespace NUMINAMATH_GPT_option_b_correct_l152_15275

theorem option_b_correct (a : ℝ) : (-a)^3 / (-a)^2 = -a :=
by sorry

end NUMINAMATH_GPT_option_b_correct_l152_15275


namespace NUMINAMATH_GPT_tangent_sphere_surface_area_l152_15229

noncomputable def cube_side_length (V : ℝ) : ℝ := V^(1/3)
noncomputable def sphere_radius (a : ℝ) : ℝ := a / 2
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem tangent_sphere_surface_area (V : ℝ) (hV : V = 64) : 
  sphere_surface_area (sphere_radius (cube_side_length V)) = 16 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_tangent_sphere_surface_area_l152_15229


namespace NUMINAMATH_GPT_possible_values_of_expression_l152_15287

theorem possible_values_of_expression (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  ∃ (vals : Finset ℤ), vals = {6, 2, 0, -2, -6} ∧
  (∃ val ∈ vals, val = (if p > 0 then 1 else -1) + 
                         (if q > 0 then 1 else -1) + 
                         (if r > 0 then 1 else -1) + 
                         (if s > 0 then 1 else -1) + 
                         (if (p * q * r) > 0 then 1 else -1) + 
                         (if (p * r * s) > 0 then 1 else -1)) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_expression_l152_15287


namespace NUMINAMATH_GPT_find_sheets_used_l152_15210

variable (x y : ℕ) -- define variables for x and y
variable (h₁ : 82 - x = y) -- 82 - x = number of sheets left
variable (h₂ : y = x - 6) -- number of sheets left = number of sheets used - 6

theorem find_sheets_used (h₁ : 82 - x = x - 6) : x = 44 := 
by
  sorry

end NUMINAMATH_GPT_find_sheets_used_l152_15210


namespace NUMINAMATH_GPT_shaded_area_calc_l152_15245

theorem shaded_area_calc (r1_area r2_area overlap_area circle_area : ℝ)
  (h_r1_area : r1_area = 36)
  (h_r2_area : r2_area = 28)
  (h_overlap_area : overlap_area = 21)
  (h_circle_area : circle_area = Real.pi) : 
  (r1_area + r2_area - overlap_area - circle_area) = 64 - Real.pi :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_calc_l152_15245


namespace NUMINAMATH_GPT_marbles_solution_l152_15224

def marbles_problem : Prop :=
  let total_marbles := 20
  let blue_marbles := 6
  let red_marbles := 9
  let total_prob_red_white := 0.7
  let white_marbles := 5
  total_marbles = blue_marbles + red_marbles + white_marbles ∧
  (white_marbles / total_marbles + red_marbles / total_marbles = total_prob_red_white)

theorem marbles_solution : marbles_problem :=
by {
  sorry
}

end NUMINAMATH_GPT_marbles_solution_l152_15224


namespace NUMINAMATH_GPT_proposition_p_neither_sufficient_nor_necessary_l152_15290

-- Define propositions p and q
def p (m : ℝ) : Prop := m = -1
def q (m : ℝ) : Prop := ∀ x y : ℝ, (x - 1 = 0) ∧ (x + m^2 * y = 0) → ∀ x' y' : ℝ, x' = x ∧ y' = y → (x - 1) * (x + m^2 * y) = 0

-- Main theorem statement
theorem proposition_p_neither_sufficient_nor_necessary (m : ℝ) : ¬ (p m → q m) ∧ ¬ (q m → p m) :=
by
  sorry

end NUMINAMATH_GPT_proposition_p_neither_sufficient_nor_necessary_l152_15290


namespace NUMINAMATH_GPT_find_y_l152_15280

-- Define the problem conditions
variable (x y : ℕ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (rem_eq : x % y = 3)
variable (div_eq : (x : ℝ) / y = 96.12)

-- The theorem to prove
theorem find_y : y = 25 :=
sorry

end NUMINAMATH_GPT_find_y_l152_15280


namespace NUMINAMATH_GPT_A_8_coords_l152_15292

-- Define point as a structure
structure Point where
  x : Int
  y : Int

-- Initial point A
def A : Point := {x := 3, y := 2}

-- Symmetric point about the y-axis
def sym_y (p : Point) : Point := {x := -p.x, y := p.y}

-- Symmetric point about the origin
def sym_origin (p : Point) : Point := {x := -p.x, y := -p.y}

-- Symmetric point about the x-axis
def sym_x (p : Point) : Point := {x := p.x, y := -p.y}

-- Function to get the n-th symmetric point in the sequence
def sym_point (n : Nat) : Point :=
  match n % 3 with
  | 0 => A
  | 1 => sym_y A
  | 2 => sym_origin (sym_y A)
  | _ => A  -- Fallback case (should not be reachable for n >= 0)

theorem A_8_coords : sym_point 8 = {x := 3, y := -2} := sorry

end NUMINAMATH_GPT_A_8_coords_l152_15292


namespace NUMINAMATH_GPT_arithmetic_series_sum_l152_15214

theorem arithmetic_series_sum :
  let a1 := 5
  let an := 105
  let d := 1
  let n := (an - a1) / d + 1
  (n * (a1 + an) / 2) = 5555 := by
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_l152_15214


namespace NUMINAMATH_GPT_geometric_sequence_sum_l152_15277

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) ^ 2 = a n * a (n + 2))
  (h_pos : ∀ n, 0 < a n) (h_given : a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16) :
  a 3 + a 6 = 4 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l152_15277


namespace NUMINAMATH_GPT_time_upstream_equal_nine_hours_l152_15211

noncomputable def distance : ℝ := 126
noncomputable def time_downstream : ℝ := 7
noncomputable def current_speed : ℝ := 2
noncomputable def downstream_speed := distance / time_downstream
noncomputable def boat_speed := downstream_speed - current_speed
noncomputable def upstream_speed := boat_speed - current_speed

theorem time_upstream_equal_nine_hours : (distance / upstream_speed) = 9 := by
  sorry

end NUMINAMATH_GPT_time_upstream_equal_nine_hours_l152_15211


namespace NUMINAMATH_GPT_conscript_from_western_village_l152_15261

/--
Given:
- The population of the northern village is 8758
- The population of the western village is 7236
- The population of the southern village is 8356
- The total number of conscripts needed is 378

Prove that the number of people to be conscripted from the western village is 112.
-/
theorem conscript_from_western_village (hnorth : ℕ) (hwest : ℕ) (hsouth : ℕ) (hconscripts : ℕ)
    (htotal : hnorth + hwest + hsouth = 24350) :
    let prop := (hwest / (hnorth + hwest + hsouth)) * hconscripts
    hnorth = 8758 → hwest = 7236 → hsouth = 8356 → hconscripts = 378 → prop = 112 :=
by
  intros
  simp_all
  sorry

end NUMINAMATH_GPT_conscript_from_western_village_l152_15261


namespace NUMINAMATH_GPT_problem_power_function_l152_15289

-- Defining the conditions
variable {f : ℝ → ℝ}
variable (a : ℝ)
variable (h₁ : ∀ x, f x = x^a)
variable (h₂ : f 2 = Real.sqrt 2)

-- Stating what we need to prove
theorem problem_power_function : f 4 = 2 :=
by sorry

end NUMINAMATH_GPT_problem_power_function_l152_15289


namespace NUMINAMATH_GPT_parabola_intersects_xaxis_at_least_one_l152_15273

theorem parabola_intersects_xaxis_at_least_one {a b c : ℝ} (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x1 x2, x1 ≠ x2 ∧ (a * x1^2 + 2 * b * x1 + c = 0) ∧ (a * x2^2 + 2 * b * x2 + c = 0)) ∨
  (∃ x1 x2, x1 ≠ x2 ∧ (b * x1^2 + 2 * c * x1 + a = 0) ∧ (b * x2^2 + 2 * c * x2 + a = 0)) ∨
  (∃ x1 x2, x1 ≠ x2 ∧ (c * x1^2 + 2 * a * x1 + b = 0) ∧ (c * x2^2 + 2 * a * x2 + b = 0)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_intersects_xaxis_at_least_one_l152_15273


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_l152_15204

theorem arithmetic_geometric_mean (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) : x^2 + y^2 = 1380 := by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_l152_15204


namespace NUMINAMATH_GPT_Mrs_Amaro_roses_l152_15270

theorem Mrs_Amaro_roses :
  ∀ (total_roses red_roses yellow_roses pink_roses remaining_roses white_and_purple white_roses purple_roses : ℕ),
    total_roses = 500 →
    5 * total_roses % 8 = 0 →
    red_roses = total_roses * 5 / 8 →
    yellow_roses = (total_roses - red_roses) * 1 / 8 →
    pink_roses = (total_roses - red_roses) * 2 / 8 →
    remaining_roses = total_roses - red_roses - yellow_roses - pink_roses →
    remaining_roses % 2 = 0 →
    white_roses = remaining_roses / 2 →
    purple_roses = remaining_roses / 2 →
    red_roses + white_roses + purple_roses = 430 :=
by
  intros total_roses red_roses yellow_roses pink_roses remaining_roses white_and_purple white_roses purple_roses
  intro total_roses_eq
  intro red_roses_divisible
  intro red_roses_def
  intro yellow_roses_def
  intro pink_roses_def
  intro remaining_roses_def
  intro remaining_roses_even
  intro white_roses_def
  intro purple_roses_def
  sorry

end NUMINAMATH_GPT_Mrs_Amaro_roses_l152_15270


namespace NUMINAMATH_GPT_car_average_speed_is_correct_l152_15221

noncomputable def average_speed_of_car : ℝ :=
  let d1 := 30
  let s1 := 30
  let d2 := 35
  let s2 := 55
  let t3 := 0.5
  let s3 := 70
  let t4 := 40 / 60 -- 40 minutes converted to hours
  let s4 := 36
  let t1 := d1 / s1
  let t2 := d2 / s2
  let d3 := s3 * t3
  let d4 := s4 * t4
  let total_distance := d1 + d2 + d3 + d4
  let total_time := t1 + t2 + t3 + t4
  total_distance / total_time

theorem car_average_speed_is_correct :
  average_speed_of_car = 44.238 := 
sorry

end NUMINAMATH_GPT_car_average_speed_is_correct_l152_15221


namespace NUMINAMATH_GPT_initial_candies_is_720_l152_15263

-- Definitions according to the conditions
def candies_remaining_after_day_n (initial_candies : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 1 => initial_candies / 2
  | 2 => (initial_candies / 2) / 3
  | 3 => (initial_candies / 2) / 3 / 4
  | 4 => (initial_candies / 2) / 3 / 4 / 5
  | 5 => (initial_candies / 2) / 3 / 4 / 5 / 6
  | _ => 0 -- For days beyond the fifth, this is nonsensical

-- Proof statement
theorem initial_candies_is_720 : ∀ (initial_candies : ℕ), candies_remaining_after_day_n initial_candies 5 = 1 → initial_candies = 720 :=
by
  intros initial_candies h
  sorry

end NUMINAMATH_GPT_initial_candies_is_720_l152_15263


namespace NUMINAMATH_GPT_jerry_remaining_debt_l152_15264

variable (two_months_ago_payment last_month_payment total_debt : ℕ)

def remaining_debt : ℕ := total_debt - (two_months_ago_payment + last_month_payment)

theorem jerry_remaining_debt :
  two_months_ago_payment = 12 →
  last_month_payment = 12 + 3 →
  total_debt = 50 →
  remaining_debt two_months_ago_payment last_month_payment total_debt = 23 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_jerry_remaining_debt_l152_15264


namespace NUMINAMATH_GPT_correct_calculation_l152_15206

theorem correct_calculation (n : ℕ) (h : n - 59 = 43) : n - 46 = 56 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l152_15206


namespace NUMINAMATH_GPT_train_cross_time_approx_24_seconds_l152_15208

open Real

noncomputable def time_to_cross (train_length : ℝ) (train_speed_km_h : ℝ) (man_speed_km_h : ℝ) : ℝ :=
  let train_speed_m_s := train_speed_km_h * (1000 / 3600)
  let man_speed_m_s := man_speed_km_h * (1000 / 3600)
  let relative_speed := train_speed_m_s - man_speed_m_s
  train_length / relative_speed

theorem train_cross_time_approx_24_seconds : 
  abs (time_to_cross 400 63 3 - 24) < 1 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_time_approx_24_seconds_l152_15208


namespace NUMINAMATH_GPT_room_length_l152_15274

theorem room_length (w : ℝ) (cost_rate : ℝ) (total_cost : ℝ) (h : w = 4) (h1 : cost_rate = 800) (h2 : total_cost = 17600) : 
  let L := total_cost / (w * cost_rate)
  L = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_room_length_l152_15274


namespace NUMINAMATH_GPT_molecular_physics_statements_l152_15293

theorem molecular_physics_statements :
  (¬A) ∧ B ∧ C ∧ D :=
by sorry

end NUMINAMATH_GPT_molecular_physics_statements_l152_15293


namespace NUMINAMATH_GPT_find_A_l152_15220

theorem find_A (A B : ℕ) (h1 : A + B = 1149) (h2 : A = 8 * B + 24) : A = 1024 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l152_15220


namespace NUMINAMATH_GPT_solve_swim_problem_l152_15251

/-- A man swims downstream 36 km and upstream some distance taking 3 hours each time. 
The speed of the man in still water is 9 km/h. -/
def swim_problem : Prop :=
  ∃ (v : ℝ) (d : ℝ),
    (9 + v) * 3 = 36 ∧ -- effective downstream speed and distance condition
    (9 - v) * 3 = d ∧ -- effective upstream speed and distance relation
    d = 18            -- required distance upstream is 18 km

theorem solve_swim_problem : swim_problem :=
  sorry

end NUMINAMATH_GPT_solve_swim_problem_l152_15251


namespace NUMINAMATH_GPT_max_xy_l152_15227

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 18) : xy <= 81 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_xy_l152_15227


namespace NUMINAMATH_GPT_largest_angle_measure_l152_15233

theorem largest_angle_measure (v : ℝ) (h : v > 3/2) :
  ∃ θ, θ = Real.arccos ((4 * v - 4) / (2 * Real.sqrt ((2 * v - 3) * (4 * v - 4)))) ∧
       θ = π - θ ∧
       θ = Real.arccos ((2 * v - 3) / (2 * Real.sqrt ((2 * v + 3) * (4 * v - 4)))) := 
sorry

end NUMINAMATH_GPT_largest_angle_measure_l152_15233


namespace NUMINAMATH_GPT_number_of_valid_ns_l152_15248

theorem number_of_valid_ns :
  ∃ (S : Finset ℕ), S.card = 13 ∧ ∀ n ∈ S, n ≤ 1000 ∧ Nat.floor (995 / n) + Nat.floor (996 / n) + Nat.floor (997 / n) % 4 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_number_of_valid_ns_l152_15248
