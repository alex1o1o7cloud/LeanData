import Mathlib

namespace NUMINAMATH_GPT_age_of_hospital_l1981_198142

theorem age_of_hospital (grant_current_age : ℕ) (future_ratio : ℚ)
                        (grant_future_age : grant_current_age + 5 = 30)
                        (hospital_age_ratio : future_ratio = 2 / 3) :
                        (grant_current_age = 25) → 
                        (grant_current_age + 5 = future_ratio * (grant_current_age + 5 + 5)) →
                        (grant_current_age + 5 + 5 - 5 = 40) :=
by
  sorry

end NUMINAMATH_GPT_age_of_hospital_l1981_198142


namespace NUMINAMATH_GPT_am_gm_inequality_l1981_198135

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (b * c)) + (b^3 / (c * a)) + (c^3 / (a * b)) ≥ a + b + c :=
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l1981_198135


namespace NUMINAMATH_GPT_system_solutions_l1981_198172

theorem system_solutions : {p : ℝ × ℝ | p.snd ^ 2 = p.fst ∧ p.snd = p.fst} = {⟨1, 1⟩, ⟨0, 0⟩} :=
by
  sorry

end NUMINAMATH_GPT_system_solutions_l1981_198172


namespace NUMINAMATH_GPT_maria_cookies_left_l1981_198118

def maria_cookies (initial: ℕ) (to_friend: ℕ) (to_family_divisor: ℕ) (eats: ℕ) : ℕ :=
  (initial - to_friend) / to_family_divisor - eats

theorem maria_cookies_left (h : maria_cookies 19 5 2 2 = 5): true :=
by trivial

end NUMINAMATH_GPT_maria_cookies_left_l1981_198118


namespace NUMINAMATH_GPT_aunt_may_morning_milk_l1981_198126

-- Defining the known quantities as variables
def evening_milk : ℕ := 380
def sold_milk : ℕ := 612
def leftover_milk : ℕ := 15
def milk_left : ℕ := 148

-- Main statement to be proven
theorem aunt_may_morning_milk (M : ℕ) :
  M + evening_milk + leftover_milk - sold_milk = milk_left → M = 365 := 
by {
  -- Skipping the proof
  sorry
}

end NUMINAMATH_GPT_aunt_may_morning_milk_l1981_198126


namespace NUMINAMATH_GPT_eq_rectangular_eq_of_polar_eq_max_m_value_l1981_198108

def polar_to_rectangular (ρ θ : ℝ) : Prop := (ρ = 4 * Real.cos θ) → ∀ x y : ℝ, ρ^2 = x^2 + y^2

theorem eq_rectangular_eq_of_polar_eq (ρ θ : ℝ) :
  polar_to_rectangular ρ θ → ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 :=
sorry

def max_m_condition (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 → |4 + 2 * m| / Real.sqrt 5 ≤ 2

theorem max_m_value :
  (max_m_condition (Real.sqrt 5 - 2)) :=
sorry

end NUMINAMATH_GPT_eq_rectangular_eq_of_polar_eq_max_m_value_l1981_198108


namespace NUMINAMATH_GPT_average_temperature_l1981_198101

def temperatures :=
  ∃ T_tue T_wed T_thu : ℝ,
    (44 + T_tue + T_wed + T_thu) / 4 = 48 ∧
    (T_tue + T_wed + T_thu + 36) / 4 = 46

theorem average_temperature :
  temperatures :=
by
  sorry

end NUMINAMATH_GPT_average_temperature_l1981_198101


namespace NUMINAMATH_GPT_proof_AC_time_l1981_198106

noncomputable def A : ℝ := 1/10
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := 1/30

def rate_A_B (A B : ℝ) := A + B = 1/6
def rate_B_C (B C : ℝ) := B + C = 1/10
def rate_A_B_C (A B C : ℝ) := A + B + C = 1/5

theorem proof_AC_time {A B C : ℝ} (h1 : rate_A_B A B) (h2 : rate_B_C B C) (h3 : rate_A_B_C A B C) : 
  (1 : ℝ) / (A + C) = 7.5 :=
sorry

end NUMINAMATH_GPT_proof_AC_time_l1981_198106


namespace NUMINAMATH_GPT_watch_cost_price_l1981_198143

theorem watch_cost_price (SP_loss SP_gain CP : ℝ) 
  (h1 : SP_loss = 0.9 * CP) 
  (h2 : SP_gain = 1.04 * CP) 
  (h3 : SP_gain - SP_loss = 196) 
  : CP = 1400 := 
sorry

end NUMINAMATH_GPT_watch_cost_price_l1981_198143


namespace NUMINAMATH_GPT_fraction_identity_proof_l1981_198144

theorem fraction_identity_proof (a b : ℝ) (h : 2 / a - 1 / b = 1 / (a + 2 * b)) :
  4 / (a ^ 2) - 1 / (b ^ 2) = 1 / (a * b) :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_proof_l1981_198144


namespace NUMINAMATH_GPT_find_xyz_sum_l1981_198105

theorem find_xyz_sum (x y z : ℝ) (h1 : x^2 + x * y + y^2 = 108)
                               (h2 : y^2 + y * z + z^2 = 49)
                               (h3 : z^2 + z * x + x^2 = 157) :
  x * y + y * z + z * x = 84 :=
sorry

end NUMINAMATH_GPT_find_xyz_sum_l1981_198105


namespace NUMINAMATH_GPT_terminating_decimal_expansion_of_13_over_320_l1981_198130

theorem terminating_decimal_expansion_of_13_over_320 : ∃ (b : ℕ) (a : ℚ), (13 : ℚ) / 320 = a / 10 ^ b ∧ a / 10 ^ b = 0.650 :=
by
  sorry

end NUMINAMATH_GPT_terminating_decimal_expansion_of_13_over_320_l1981_198130


namespace NUMINAMATH_GPT_florist_initial_roses_l1981_198186

theorem florist_initial_roses : 
  ∀ (R : ℕ), (R - 16 + 19 = 40) → (R = 37) :=
by
  intro R
  intro h
  sorry

end NUMINAMATH_GPT_florist_initial_roses_l1981_198186


namespace NUMINAMATH_GPT_Josh_pencils_left_l1981_198111

theorem Josh_pencils_left (initial_pencils : ℕ) (given_pencils : ℕ) (remaining_pencils : ℕ) 
  (h_initial : initial_pencils = 142) 
  (h_given : given_pencils = 31) 
  (h_remaining : remaining_pencils = 111) : 
  initial_pencils - given_pencils = remaining_pencils :=
by
  sorry

end NUMINAMATH_GPT_Josh_pencils_left_l1981_198111


namespace NUMINAMATH_GPT_smallest_bdf_l1981_198160

theorem smallest_bdf (a b c d e f : ℕ) (A : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : e > 0) (h6 : f > 0)
  (h7 : A = a * c * e / (b * d * f))
  (h8 : A = (a + 1) * c * e / (b * d * f) - 3)
  (h9 : A = a * (c + 1) * e / (b * d * f) - 4)
  (h10 : A = a * c * (e + 1) / (b * d * f) - 5) :
  b * d * f = 60 :=
by
  sorry

end NUMINAMATH_GPT_smallest_bdf_l1981_198160


namespace NUMINAMATH_GPT_radius_of_cone_is_8_l1981_198109

noncomputable def r_cylinder := 8 -- cm
noncomputable def h_cylinder := 2 -- cm
noncomputable def h_cone := 6 -- cm

theorem radius_of_cone_is_8 :
  exists (r_cone : ℝ), r_cone = 8 ∧ π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone :=
by
  let r_cone := 8
  have eq_volumes : π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone := 
    sorry
  exact ⟨r_cone, by simp, eq_volumes⟩

end NUMINAMATH_GPT_radius_of_cone_is_8_l1981_198109


namespace NUMINAMATH_GPT_original_price_of_suit_l1981_198119

theorem original_price_of_suit (P : ℝ) (h : 0.96 * P = 144) : P = 150 :=
sorry

end NUMINAMATH_GPT_original_price_of_suit_l1981_198119


namespace NUMINAMATH_GPT_largest_of_three_consecutive_integers_l1981_198175

theorem largest_of_three_consecutive_integers (N : ℤ) (h : N + (N + 1) + (N + 2) = 18) : N + 2 = 7 :=
sorry

end NUMINAMATH_GPT_largest_of_three_consecutive_integers_l1981_198175


namespace NUMINAMATH_GPT_slope_of_tangent_at_1_0_l1981_198180

noncomputable def f (x : ℝ) : ℝ :=
2 * x^2 - 2 * x

def derivative_f (x : ℝ) : ℝ :=
4 * x - 2

theorem slope_of_tangent_at_1_0 : derivative_f 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_tangent_at_1_0_l1981_198180


namespace NUMINAMATH_GPT_polynomial_problem_l1981_198179

theorem polynomial_problem :
  ∀ P : Polynomial ℤ,
    (∃ R : Polynomial ℤ, (X^2 + 6*X + 10) * P^2 - 1 = R^2) → 
    P = 0 :=
by { sorry }

end NUMINAMATH_GPT_polynomial_problem_l1981_198179


namespace NUMINAMATH_GPT_geometric_sequence_term_l1981_198177

noncomputable def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_term {a : ℕ → ℤ} {q : ℤ}
  (h1 : geometric_sequence a q)
  (h2 : a 7 = 10)
  (h3 : q = -2) :
  a 10 = -80 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_term_l1981_198177


namespace NUMINAMATH_GPT_cars_in_first_section_l1981_198169

noncomputable def first_section_rows : ℕ := 15
noncomputable def first_section_cars_per_row : ℕ := 10
noncomputable def total_cars_first_section : ℕ := first_section_rows * first_section_cars_per_row

theorem cars_in_first_section : total_cars_first_section = 150 :=
by
  sorry

end NUMINAMATH_GPT_cars_in_first_section_l1981_198169


namespace NUMINAMATH_GPT_max_dot_product_on_circle_l1981_198158

theorem max_dot_product_on_circle :
  (∃(x y : ℝ),
    x^2 + (y - 3)^2 = 1 ∧
    2 ≤ y ∧ y ≤ 4 ∧
    (∀(y : ℝ), (2 ≤ y ∧ y ≤ 4 →
      (x^2 + y^2 - 4) ≤ 12))) := by
  sorry

end NUMINAMATH_GPT_max_dot_product_on_circle_l1981_198158


namespace NUMINAMATH_GPT_basketball_game_points_l1981_198185

variable (J T K : ℕ)

theorem basketball_game_points (h1 : T = J + 20) (h2 : J + T + K = 100) (h3 : T = 30) : 
  T / K = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_basketball_game_points_l1981_198185


namespace NUMINAMATH_GPT_largest_whole_number_l1981_198173

theorem largest_whole_number (x : ℕ) : 9 * x < 150 → x ≤ 16 :=
by sorry

end NUMINAMATH_GPT_largest_whole_number_l1981_198173


namespace NUMINAMATH_GPT_johnny_money_left_l1981_198152

def total_saved (september october november : ℕ) : ℕ := september + october + november

def money_left (total amount_spent : ℕ) : ℕ := total - amount_spent

theorem johnny_money_left 
    (saved_september : ℕ)
    (saved_october : ℕ)
    (saved_november : ℕ)
    (spent_video_game : ℕ)
    (h1 : saved_september = 30)
    (h2 : saved_october = 49)
    (h3 : saved_november = 46)
    (h4 : spent_video_game = 58) :
    money_left (total_saved saved_september saved_october saved_november) spent_video_game = 67 := 
by sorry

end NUMINAMATH_GPT_johnny_money_left_l1981_198152


namespace NUMINAMATH_GPT_integer_product_is_192_l1981_198138

theorem integer_product_is_192 (A B C : ℤ)
  (h1 : A + B + C = 33)
  (h2 : C = 3 * B)
  (h3 : A = C - 23) :
  A * B * C = 192 :=
sorry

end NUMINAMATH_GPT_integer_product_is_192_l1981_198138


namespace NUMINAMATH_GPT_no_consecutive_positive_integers_have_sum_75_l1981_198127

theorem no_consecutive_positive_integers_have_sum_75 :
  ∀ n a : ℕ, (n ≥ 2) → (a ≥ 1) → (n * (2 * a + n - 1) = 150) → False :=
by
  intros n a hn ha hsum
  sorry

end NUMINAMATH_GPT_no_consecutive_positive_integers_have_sum_75_l1981_198127


namespace NUMINAMATH_GPT_deductive_reasoning_is_option_A_l1981_198134

-- Define the types of reasoning.
inductive ReasoningType
| Deductive
| Analogical
| Inductive

-- Define the options provided in the problem.
def OptionA : ReasoningType := ReasoningType.Deductive
def OptionB : ReasoningType := ReasoningType.Analogical
def OptionC : ReasoningType := ReasoningType.Inductive
def OptionD : ReasoningType := ReasoningType.Inductive

-- Statement to prove that Option A is Deductive reasoning.
theorem deductive_reasoning_is_option_A : OptionA = ReasoningType.Deductive := by
  -- proof
  sorry

end NUMINAMATH_GPT_deductive_reasoning_is_option_A_l1981_198134


namespace NUMINAMATH_GPT_evaluate_expression_l1981_198155

noncomputable def f : ℝ → ℝ := sorry

lemma f_condition (a : ℝ) : f (a + 1) = f a * f 1 := sorry

lemma f_one : f 1 = 2 := sorry

theorem evaluate_expression :
  (f 2018 / f 2017) + (f 2019 / f 2018) + (f 2020 / f 2019) = 6 :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l1981_198155


namespace NUMINAMATH_GPT_simple_interest_rate_l1981_198116

theorem simple_interest_rate (P : ℝ) (T : ℝ) (hT : T = 15)
  (doubles_in_15_years : ∃ R : ℝ, (P * 2 = P + (P * R * T) / 100)) :
  ∃ R : ℝ, R = 6.67 := 
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1981_198116


namespace NUMINAMATH_GPT_tetrahedron_inequality_l1981_198154

theorem tetrahedron_inequality (t1 t2 t3 t4 τ1 τ2 τ3 τ4 : ℝ) 
  (ht1 : t1 > 0) (ht2 : t2 > 0) (ht3 : t3 > 0) (ht4 : t4 > 0)
  (hτ1 : τ1 > 0) (hτ2 : τ2 > 0) (hτ3 : τ3 > 0) (hτ4 : τ4 > 0)
  (sphere_inscribed : ∀ {x y : ℝ}, x > 0 → y > 0 → x^2 / y^2 ≤ (x - 2 * y) ^ 2 / x ^ 2) :
  (τ1 / t1 + τ2 / t2 + τ3 / t3 + τ4 / t4) ≥ 1 
  ∧ (τ1 / t1 + τ2 / t2 + τ3 / t3 + τ4 / t4 = 1 ↔ t1 = t2 ∧ t2 = t3 ∧ t3 = t4) := by
  sorry

end NUMINAMATH_GPT_tetrahedron_inequality_l1981_198154


namespace NUMINAMATH_GPT_race_distance_l1981_198182

variables (a b c d : ℝ)
variables (h1 : d / a = (d - 30) / b)
variables (h2 : d / b = (d - 15) / c)
variables (h3 : d / a = (d - 40) / c)

theorem race_distance : d = 90 :=
by 
  sorry

end NUMINAMATH_GPT_race_distance_l1981_198182


namespace NUMINAMATH_GPT_three_pipes_time_l1981_198188

variable (R : ℝ) (T : ℝ)

-- Condition: Two pipes fill the tank in 18 hours
def two_pipes_fill : Prop := 2 * R * 18 = 1

-- Question: How long does it take for three pipes to fill the tank?
def three_pipes_fill : Prop := 3 * R * T = 1

theorem three_pipes_time (h : two_pipes_fill R) : three_pipes_fill R 12 :=
by
  sorry

end NUMINAMATH_GPT_three_pipes_time_l1981_198188


namespace NUMINAMATH_GPT_sin_three_pi_over_two_l1981_198110

theorem sin_three_pi_over_two : Real.sin (3 * Real.pi / 2) = -1 :=
by
  sorry

end NUMINAMATH_GPT_sin_three_pi_over_two_l1981_198110


namespace NUMINAMATH_GPT_polynomial_factor_l1981_198114

theorem polynomial_factor (x : ℝ) : (x^2 - 4*x + 4) ∣ (x^4 + 16) :=
sorry

end NUMINAMATH_GPT_polynomial_factor_l1981_198114


namespace NUMINAMATH_GPT_number_mul_five_l1981_198100

theorem number_mul_five (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 :=
by
  sorry

end NUMINAMATH_GPT_number_mul_five_l1981_198100


namespace NUMINAMATH_GPT_polynomial_identity_l1981_198140

theorem polynomial_identity
  (x a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ)
  (h : (x - 1)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  (a + a_2 + a_4 + a_6)^2 - (a_1 + a_3 + a_5 + a_7)^2 = 0 :=
by sorry

end NUMINAMATH_GPT_polynomial_identity_l1981_198140


namespace NUMINAMATH_GPT_k_range_l1981_198176

theorem k_range (k : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 0 ≤ 2 * x - 2 * k) → k ≤ 1 :=
by
  intro h
  have h1 := h 1 (by simp)
  have h3 := h 3 (by simp)
  sorry

end NUMINAMATH_GPT_k_range_l1981_198176


namespace NUMINAMATH_GPT_probability_calculation_l1981_198113

noncomputable def probability_in_ellipsoid : ℝ :=
  let prism_volume := (2 - (-2)) * (1 - (-1)) * (1 - (-1))
  let ellipsoid_volume := (4 * Real.pi / 3) * 1 * 2 * 2
  ellipsoid_volume / prism_volume

theorem probability_calculation :
  probability_in_ellipsoid = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_probability_calculation_l1981_198113


namespace NUMINAMATH_GPT_wholesale_cost_proof_l1981_198166

-- Definitions based on conditions
def wholesale_cost (W : ℝ) := W
def retail_price (W : ℝ) := 1.20 * W
def employee_paid (R : ℝ) := 0.90 * R

-- Theorem statement: given the conditions, prove that the wholesale cost is $200.
theorem wholesale_cost_proof : 
  ∃ W : ℝ, (retail_price W = 1.20 * W) ∧ (employee_paid (retail_price W) = 216) ∧ W = 200 :=
by 
  let W := 200
  have hp : retail_price W = 1.20 * W := by sorry
  have ep : employee_paid (retail_price W) = 216 := by sorry
  exact ⟨W, hp, ep, rfl⟩

end NUMINAMATH_GPT_wholesale_cost_proof_l1981_198166


namespace NUMINAMATH_GPT_determinant_inequality_l1981_198165

open Real

def det (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_inequality (x : ℝ) :
  det 7 (x^2) 2 1 > det 3 (-2) 1 x ↔ -5/2 < x ∧ x < 1 :=
by
  sorry

end NUMINAMATH_GPT_determinant_inequality_l1981_198165


namespace NUMINAMATH_GPT_certain_event_l1981_198193

-- Definitions for a line and plane
inductive Line
| mk : Line

inductive Plane
| mk : Plane

-- Definitions for parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def plane_parallel (p₁ p₂ : Plane) : Prop := sorry

-- Given conditions and the proof statement
theorem certain_event (l : Line) (α β : Plane) (h1 : perpendicular l α) (h2 : perpendicular l β) : plane_parallel α β :=
sorry

end NUMINAMATH_GPT_certain_event_l1981_198193


namespace NUMINAMATH_GPT_tetrahedron_volume_l1981_198184

noncomputable def volume_of_tetrahedron (A B C O : Point) (r : ℝ) :=
  1 / 3 * (Real.sqrt (3) / 4 * 2^2 * Real.sqrt 11)

theorem tetrahedron_volume 
  (A B C O : Point)
  (side_length : ℝ)
  (surface_area : ℝ)
  (radius : ℝ)
  (h : ℝ)
  (radius_eq : radius = Real.sqrt (37 / 3))
  (side_length_eq : side_length = 2)
  (surface_area_eq : surface_area = (4 * Real.pi * radius^2))
  (sphere_surface_area_eq : surface_area = 148 * Real.pi / 3)
  (height_eq : h^2 = radius^2 - (2 / 3 * 2 * Real.sqrt 3 / 2)^2)
  (height_value_eq : h = Real.sqrt 11) :
  volume_of_tetrahedron A B C O radius = Real.sqrt 33 / 3 := sorry

end NUMINAMATH_GPT_tetrahedron_volume_l1981_198184


namespace NUMINAMATH_GPT_percentage_failed_in_Hindi_l1981_198137

-- Let Hindi_failed denote the percentage of students who failed in Hindi.
-- Let English_failed denote the percentage of students who failed in English.
-- Let Both_failed denote the percentage of students who failed in both Hindi and English.
-- Let Both_passed denote the percentage of students who passed in both subjects.

variables (Hindi_failed English_failed Both_failed Both_passed : ℝ)
  (H_condition1 : English_failed = 44)
  (H_condition2 : Both_failed = 22)
  (H_condition3 : Both_passed = 44)

theorem percentage_failed_in_Hindi:
  Hindi_failed = 34 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_failed_in_Hindi_l1981_198137


namespace NUMINAMATH_GPT_correct_calculation_l1981_198167

noncomputable def option_A : Prop := (Real.sqrt 3 + Real.sqrt 2) ≠ Real.sqrt 5
noncomputable def option_B : Prop := (Real.sqrt 3 * Real.sqrt 5) = Real.sqrt 15 ∧ Real.sqrt 15 ≠ 15
noncomputable def option_C : Prop := Real.sqrt (32 / 8) = 2 ∧ (Real.sqrt (32 / 8) ≠ -2)
noncomputable def option_D : Prop := (2 * Real.sqrt 3) - Real.sqrt 3 = Real.sqrt 3

theorem correct_calculation : option_D :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1981_198167


namespace NUMINAMATH_GPT_daily_salary_of_manager_l1981_198187

theorem daily_salary_of_manager
  (M : ℕ)
  (salary_clerk : ℕ)
  (num_managers : ℕ)
  (num_clerks : ℕ)
  (total_salary : ℕ)
  (h1 : salary_clerk = 2)
  (h2 : num_managers = 2)
  (h3 : num_clerks = 3)
  (h4 : total_salary = 16)
  (h5 : 2 * M + 3 * salary_clerk = total_salary) :
  M = 5 := 
  sorry

end NUMINAMATH_GPT_daily_salary_of_manager_l1981_198187


namespace NUMINAMATH_GPT_sum_of_coefficients_is_zero_l1981_198147

noncomputable def expansion : Polynomial ℚ := (Polynomial.X^2 + Polynomial.X + 1) * (2*Polynomial.X - 2)^5

theorem sum_of_coefficients_is_zero :
  (expansion.coeff 0) + (expansion.coeff 1) + (expansion.coeff 2) + (expansion.coeff 3) + 
  (expansion.coeff 4) + (expansion.coeff 5) + (expansion.coeff 6) + (expansion.coeff 7) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_is_zero_l1981_198147


namespace NUMINAMATH_GPT_cos_330_eq_sqrt3_div_2_l1981_198120

theorem cos_330_eq_sqrt3_div_2
    (h1 : ∀ θ : ℝ, Real.cos (2 * Real.pi - θ) = Real.cos θ)
    (h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
    Real.cos (11 * Real.pi / 6) = Real.sqrt 3 / 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cos_330_eq_sqrt3_div_2_l1981_198120


namespace NUMINAMATH_GPT_invitations_per_package_l1981_198146

-- Definitions based on conditions in the problem.
def numPackages : Nat := 5
def totalInvitations : Nat := 45

-- Definition of the problem and proof statement.
theorem invitations_per_package :
  totalInvitations / numPackages = 9 :=
by
  sorry

end NUMINAMATH_GPT_invitations_per_package_l1981_198146


namespace NUMINAMATH_GPT_probability_two_identical_l1981_198128

-- Define the number of ways to choose 3 out of 4 attractions
def choose_3_out_of_4 := Nat.choose 4 3

-- Define the total number of ways for both tourists to choose 3 attractions out of 4
def total_basic_events := choose_3_out_of_4 * choose_3_out_of_4

-- Define the number of ways to choose exactly 2 identical attractions
def ways_to_choose_2_identical := Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 1 1

-- The probability that they choose exactly 2 identical attractions
def probability : ℚ := ways_to_choose_2_identical / total_basic_events

-- Prove that this probability is 3/4
theorem probability_two_identical : probability = 3 / 4 := by
  have h1 : choose_3_out_of_4 = 4 := by sorry
  have h2 : total_basic_events = 16 := by sorry
  have h3 : ways_to_choose_2_identical = 12 := by sorry
  rw [probability, h2, h3]
  norm_num

end NUMINAMATH_GPT_probability_two_identical_l1981_198128


namespace NUMINAMATH_GPT_case_m_eq_neg_1_case_m_gt_neg_1_case_m_lt_neg_1_l1981_198104

noncomputable def solution_set (m x : ℝ) : Prop :=
  x^2 + (m-1) * x - m > 0

theorem case_m_eq_neg_1 (x : ℝ) :
  solution_set (-1) x ↔ x ≠ 1 :=
sorry

theorem case_m_gt_neg_1 (m x : ℝ) (hm : m > -1) :
  solution_set m x ↔ (x < -m ∨ x > 1) :=
sorry

theorem case_m_lt_neg_1 (m x : ℝ) (hm : m < -1) :
  solution_set m x ↔ (x < 1 ∨ x > -m) :=
sorry

end NUMINAMATH_GPT_case_m_eq_neg_1_case_m_gt_neg_1_case_m_lt_neg_1_l1981_198104


namespace NUMINAMATH_GPT_sum_digits_of_3n_l1981_198168

noncomputable def sum_digits (n : ℕ) : ℕ :=
sorry  -- Placeholder for a proper implementation of sum_digits

theorem sum_digits_of_3n (n : ℕ) 
  (h1 : sum_digits n = 100) 
  (h2 : sum_digits (44 * n) = 800) : 
  sum_digits (3 * n) = 300 := 
by
  sorry

end NUMINAMATH_GPT_sum_digits_of_3n_l1981_198168


namespace NUMINAMATH_GPT_penny_money_left_is_5_l1981_198123

def penny_initial_money : ℤ := 20
def socks_pairs : ℤ := 4
def price_per_pair_of_socks : ℤ := 2
def price_of_hat : ℤ := 7

def total_cost_of_socks : ℤ := socks_pairs * price_per_pair_of_socks
def total_cost_of_hat_and_socks : ℤ := total_cost_of_socks + price_of_hat
def penny_money_left : ℤ := penny_initial_money - total_cost_of_hat_and_socks

theorem penny_money_left_is_5 : penny_money_left = 5 := by
  sorry

end NUMINAMATH_GPT_penny_money_left_is_5_l1981_198123


namespace NUMINAMATH_GPT_inequality_abc_l1981_198189

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := 
by 
  sorry

end NUMINAMATH_GPT_inequality_abc_l1981_198189


namespace NUMINAMATH_GPT_cost_of_figurine_l1981_198161

noncomputable def cost_per_tv : ℝ := 50
noncomputable def num_tvs : ℕ := 5
noncomputable def num_figurines : ℕ := 10
noncomputable def total_spent : ℝ := 260

theorem cost_of_figurine : 
  ((total_spent - (num_tvs * cost_per_tv)) / num_figurines) = 1 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_figurine_l1981_198161


namespace NUMINAMATH_GPT_contrapositive_of_proposition_l1981_198163

-- Proposition: If xy=0, then x=0
def proposition (x y : ℝ) : Prop := x * y = 0 → x = 0

-- Contrapositive: If x ≠ 0, then xy ≠ 0
def contrapositive (x y : ℝ) : Prop := x ≠ 0 → x * y ≠ 0

-- Proof that contrapositive of the given proposition holds
theorem contrapositive_of_proposition (x y : ℝ) : proposition x y ↔ contrapositive x y :=
by {
  sorry
}

end NUMINAMATH_GPT_contrapositive_of_proposition_l1981_198163


namespace NUMINAMATH_GPT_football_goals_even_more_probable_l1981_198125

-- Define the problem statement and conditions
variable (p_1 : ℝ) (h₀ : 0 ≤ p_1 ∧ p_1 ≤ 1) (h₁ : q_1 = 1 - p_1)

-- Define even and odd goal probabilities for the total match
def p : ℝ := p_1^2 + (1 - p_1)^2
def q : ℝ := 2 * p_1 * (1 - p_1)

-- The main statement to prove
theorem football_goals_even_more_probable (h₂ : q_1 = 1 - p_1) : p_1^2 + (1 - p_1)^2 ≥ 2 * p_1 * (1 - p_1) :=
  sorry

end NUMINAMATH_GPT_football_goals_even_more_probable_l1981_198125


namespace NUMINAMATH_GPT_trader_profit_percentage_l1981_198132

-- Define the conditions.
variables (indicated_weight actual_weight_given claimed_weight : ℝ)
variable (profit_percentage : ℝ)

-- Given conditions
def conditions :=
  indicated_weight = 1000 ∧
  actual_weight_given = claimed_weight / 1.5 ∧
  claimed_weight = indicated_weight ∧
  profit_percentage = (claimed_weight - actual_weight_given) / actual_weight_given * 100

-- Prove that the profit percentage is 50%
theorem trader_profit_percentage : conditions indicated_weight actual_weight_given claimed_weight profit_percentage → profit_percentage = 50 :=
by
  sorry

end NUMINAMATH_GPT_trader_profit_percentage_l1981_198132


namespace NUMINAMATH_GPT_bells_toll_together_l1981_198139

theorem bells_toll_together {a b c d : ℕ} (h1 : a = 9) (h2 : b = 10) (h3 : c = 14) (h4 : d = 18) :
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 630 :=
by
  sorry

end NUMINAMATH_GPT_bells_toll_together_l1981_198139


namespace NUMINAMATH_GPT_age_of_15th_student_l1981_198196

theorem age_of_15th_student (T : ℕ) (T8 : ℕ) (T6 : ℕ)
  (avg_15_students : T / 15 = 15)
  (avg_8_students : T8 / 8 = 14)
  (avg_6_students : T6 / 6 = 16) :
  (T - (T8 + T6)) = 17 := by
  sorry

end NUMINAMATH_GPT_age_of_15th_student_l1981_198196


namespace NUMINAMATH_GPT_r_power_four_identity_l1981_198115

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end NUMINAMATH_GPT_r_power_four_identity_l1981_198115


namespace NUMINAMATH_GPT_matrix_sum_correct_l1981_198178

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 0],
  ![1, 2]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-5, -7],
  ![4, -9]
]

def C : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-2, -7],
  ![5, -7]
]

theorem matrix_sum_correct : A + B = C := by 
  sorry

end NUMINAMATH_GPT_matrix_sum_correct_l1981_198178


namespace NUMINAMATH_GPT_quadratic_roots_sum_product_l1981_198199

theorem quadratic_roots_sum_product {p q : ℝ} 
  (h1 : p / 3 = 10) 
  (h2 : q / 3 = 15) : 
  p + q = 75 := sorry

end NUMINAMATH_GPT_quadratic_roots_sum_product_l1981_198199


namespace NUMINAMATH_GPT_maximum_value_of_expression_l1981_198153

noncomputable def max_function_value (x y z : ℝ) : ℝ := 
  (x^3 - x * y^2 + y^3) * (x^3 - x * z^2 + z^3) * (y^3 - y * z^2 + z^3)

theorem maximum_value_of_expression : 
  ∃ x y z : ℝ, (x >= 0) ∧ (y >= 0) ∧ (z >= 0) ∧ (x + y + z = 3) 
  ∧ max_function_value x y z = 2916 / 2187 := 
sorry

end NUMINAMATH_GPT_maximum_value_of_expression_l1981_198153


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l1981_198192

theorem algebraic_expression_evaluation (a b : ℝ) (h : 1 / a + 1 / (2 * b) = 3) :
  (2 * a - 5 * a * b + 4 * b) / (4 * a * b - 3 * a - 6 * b) = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l1981_198192


namespace NUMINAMATH_GPT_inequality_problem_l1981_198124

variable {R : Type*} [LinearOrderedField R]

theorem inequality_problem
  (a b : R) 
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hab : a + b = 1) :
  (a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2 := 
sorry

end NUMINAMATH_GPT_inequality_problem_l1981_198124


namespace NUMINAMATH_GPT_region_Z_probability_l1981_198190

variable (P : Type) [Field P]
variable (P_X P_Y P_W P_Z : P)

theorem region_Z_probability :
  P_X = 1 / 3 → P_Y = 1 / 4 → P_W = 1 / 6 → P_X + P_Y + P_Z + P_W = 1 → P_Z = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_region_Z_probability_l1981_198190


namespace NUMINAMATH_GPT_train_length_l1981_198129

theorem train_length 
  (bridge_length train_length time_seconds v : ℝ)
  (h1 : bridge_length = 300)
  (h2 : time_seconds = 36)
  (h3 : v = 40) :
  (train_length = v * time_seconds - bridge_length) →
  (train_length = 1140) := by
  -- solve in a few lines
  -- This proof is omitted for the purpose of this task
  sorry

end NUMINAMATH_GPT_train_length_l1981_198129


namespace NUMINAMATH_GPT_sum_of_digits_l1981_198156

variable {w x y z : ℕ}

theorem sum_of_digits :
  (w + x + y + z = 20) ∧ w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  (y + w = 11) ∧ (x + y = 9) ∧ (w + z = 10) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_l1981_198156


namespace NUMINAMATH_GPT_max_x2_plus_4y_plus_3_l1981_198149

theorem max_x2_plus_4y_plus_3 
  (x y : ℝ) 
  (h : x^2 + y^2 = 1) : 
  x^2 + 4*y + 3 ≤ 7 := sorry

end NUMINAMATH_GPT_max_x2_plus_4y_plus_3_l1981_198149


namespace NUMINAMATH_GPT_number_of_sheep_l1981_198122

theorem number_of_sheep (S H : ℕ) 
  (h1 : S / H = 5 / 7)
  (h2 : H * 230 = 12880) : 
  S = 40 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sheep_l1981_198122


namespace NUMINAMATH_GPT_circle_equation_l1981_198112

theorem circle_equation 
    (a : ℝ)
    (x y : ℝ)
    (tangent_lines : x + y = 0 ∧ x + y = 4)
    (center_line : x - y = a)
    (center_point : ∃ (a : ℝ), x = a ∧ y = a) :
    ∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l1981_198112


namespace NUMINAMATH_GPT_team_savings_with_discount_l1981_198164

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

end NUMINAMATH_GPT_team_savings_with_discount_l1981_198164


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1981_198121

variable (a d : ℕ)

-- Conditions
def condition1 := (a + d) + (a + 3 * d) = 10
def condition2 := a + (a + 2 * d) = 8

-- Fifth term calculation
def fifth_term := a + 4 * d

theorem arithmetic_sequence_fifth_term (h1 : condition1 a d) (h2 : condition2 a d) : fifth_term a d = 7 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1981_198121


namespace NUMINAMATH_GPT_circles_intersect_l1981_198159

def circle1 := { x : ℝ × ℝ | (x.1 - 1)^2 + (x.2 + 2)^2 = 1 }
def circle2 := { x : ℝ × ℝ | (x.1 - 2)^2 + (x.2 + 1)^2 = 1 / 4 }

theorem circles_intersect :
  ∃ x : ℝ × ℝ, x ∈ circle1 ∧ x ∈ circle2 :=
sorry

end NUMINAMATH_GPT_circles_intersect_l1981_198159


namespace NUMINAMATH_GPT_average_speed_of_rocket_l1981_198181

def distance_soared (speed_soaring : ℕ) (time_soaring : ℕ) : ℕ :=
  speed_soaring * time_soaring

def distance_plummeted : ℕ := 600

def total_distance (distance_soared : ℕ) (distance_plummeted : ℕ) : ℕ :=
  distance_soared + distance_plummeted

def total_time (time_soaring : ℕ) (time_plummeting : ℕ) : ℕ :=
  time_soaring + time_plummeting

def average_speed (total_distance : ℕ) (total_time : ℕ) : ℕ :=
  total_distance / total_time

theorem average_speed_of_rocket :
  let speed_soaring := 150
  let time_soaring := 12
  let time_plummeting := 3
  distance_soared speed_soaring time_soaring +
  distance_plummeted = 2400
  →
  total_time time_soaring time_plummeting = 15
  →
  average_speed (distance_soared speed_soaring time_soaring + distance_plummeted)
                (total_time time_soaring time_plummeting) = 160 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_rocket_l1981_198181


namespace NUMINAMATH_GPT_john_annual_profit_is_1800_l1981_198145

def tenant_A_monthly_payment : ℕ := 350
def tenant_B_monthly_payment : ℕ := 400
def tenant_C_monthly_payment : ℕ := 450
def john_monthly_rent : ℕ := 900
def utility_cost : ℕ := 100
def maintenance_fee : ℕ := 50

noncomputable def annual_profit : ℕ :=
  let total_monthly_income := tenant_A_monthly_payment + tenant_B_monthly_payment + tenant_C_monthly_payment
  let total_monthly_expenses := john_monthly_rent + utility_cost + maintenance_fee
  let monthly_profit := total_monthly_income - total_monthly_expenses
  monthly_profit * 12

theorem john_annual_profit_is_1800 : annual_profit = 1800 := by
  sorry

end NUMINAMATH_GPT_john_annual_profit_is_1800_l1981_198145


namespace NUMINAMATH_GPT_fraction_sum_59_l1981_198133

theorem fraction_sum_59 :
  ∃ (a b : ℕ), (0.84375 = (a : ℚ) / b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 59) :=
sorry

end NUMINAMATH_GPT_fraction_sum_59_l1981_198133


namespace NUMINAMATH_GPT_cactus_species_minimum_l1981_198170

theorem cactus_species_minimum :
  ∀ (collections : Fin 80 → Fin k → Prop),
  (∀ s : Fin k, ∃ (i : Fin 80), ¬ collections i s)
  → (∀ (c : Finset (Fin 80)), c.card = 15 → ∃ s : Fin k, ∀ (i : Fin 80), i ∈ c → collections i s)
  → 16 ≤ k := 
by 
  sorry

end NUMINAMATH_GPT_cactus_species_minimum_l1981_198170


namespace NUMINAMATH_GPT_factorize_expression_l1981_198150

variable {R : Type} [CommRing R] (a x y : R)

theorem factorize_expression :
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1981_198150


namespace NUMINAMATH_GPT_percentage_increase_in_population_due_to_birth_is_55_l1981_198191

/-- The initial population at the start of the period is 100,000 people. -/
def initial_population : ℕ := 100000

/-- The period of observation is 10 years. -/
def period : ℕ := 10

/-- The number of people leaving the area each year due to emigration is 2000. -/
def emigration_per_year : ℕ := 2000

/-- The number of people coming into the area each year due to immigration is 2500. -/
def immigration_per_year : ℕ := 2500

/-- The population at the end of the period is 165,000 people. -/
def final_population : ℕ := 165000

/-- The net migration per year is calculated by subtracting emigration from immigration. -/
def net_migration_per_year : ℕ := immigration_per_year - emigration_per_year

/-- The total net migration over the period is obtained by multiplying net migration per year by the number of years. -/
def net_migration_over_period : ℕ := net_migration_per_year * period

/-- The total population increase is the difference between the final and initial population. -/
def total_population_increase : ℕ := final_population - initial_population

/-- The increase in population due to birth is calculated by subtracting net migration over the period from the total population increase. -/
def increase_due_to_birth : ℕ := total_population_increase - net_migration_over_period

/-- The percentage increase in population due to birth is calculated by dividing the increase due to birth by the initial population, and then multiplying by 100 to convert to percentage. -/
def percentage_increase_due_to_birth : ℕ := (increase_due_to_birth * 100) / initial_population

/-- The final Lean statement to prove. -/
theorem percentage_increase_in_population_due_to_birth_is_55 :
  percentage_increase_due_to_birth = 55 := by
sorry

end NUMINAMATH_GPT_percentage_increase_in_population_due_to_birth_is_55_l1981_198191


namespace NUMINAMATH_GPT_distance_before_rest_l1981_198148

theorem distance_before_rest (total_distance after_rest_distance : ℝ) (h1 : total_distance = 1) (h2 : after_rest_distance = 0.25) :
  total_distance - after_rest_distance = 0.75 :=
by sorry

end NUMINAMATH_GPT_distance_before_rest_l1981_198148


namespace NUMINAMATH_GPT_grasshopper_jump_distance_l1981_198183

theorem grasshopper_jump_distance (g f m : ℕ)
    (h1 : f = g + 32)
    (h2 : m = f - 26)
    (h3 : m = 31) : g = 25 :=
by
  sorry

end NUMINAMATH_GPT_grasshopper_jump_distance_l1981_198183


namespace NUMINAMATH_GPT_cost_price_of_watch_l1981_198151

variable (CP SP1 SP2 : ℝ)

theorem cost_price_of_watch (h1 : SP1 = 0.9 * CP)
  (h2 : SP2 = 1.04 * CP)
  (h3 : SP2 = SP1 + 200) : CP = 10000 / 7 := 
by
  sorry

end NUMINAMATH_GPT_cost_price_of_watch_l1981_198151


namespace NUMINAMATH_GPT_min_value_x_fraction_l1981_198141

theorem min_value_x_fraction (x : ℝ) (h : x > 1) : 
  ∃ m, m = 3 ∧ ∀ y > 1, y + 1 / (y - 1) ≥ m :=
by
  sorry

end NUMINAMATH_GPT_min_value_x_fraction_l1981_198141


namespace NUMINAMATH_GPT_man_l1981_198197

noncomputable def speed_in_still_water (current_speed_kmph : ℝ) (distance_m : ℝ) (time_seconds : ℝ) : ℝ :=
   let current_speed_mps := current_speed_kmph * 1000 / 3600
   let downstream_speed_mps := distance_m / time_seconds
   let still_water_speed_mps := downstream_speed_mps - current_speed_mps
   let still_water_speed_kmph := still_water_speed_mps * 3600 / 1000
   still_water_speed_kmph

theorem man's_speed_in_still_water :
  speed_in_still_water 6 100 14.998800095992323 = 18 := by
  sorry

end NUMINAMATH_GPT_man_l1981_198197


namespace NUMINAMATH_GPT_eccentricity_range_l1981_198117

variable {a b c : ℝ} (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (e : ℝ)

-- Assume a > 0, b > 0, and the eccentricity of the hyperbola is given by c = e * a.
variable (a_pos : 0 < a) (b_pos : 0 < b) (hyperbola : (P.1 / a)^2 - (P.2 / b)^2 = 1)
variable (on_right_branch : P.1 > 0)
variable (foci_condition : dist P F₁ = 4 * dist P F₂)
variable (eccentricity_def : c = e * a)

theorem eccentricity_range : 1 < e ∧ e ≤ 5 / 3 := by
  sorry

end NUMINAMATH_GPT_eccentricity_range_l1981_198117


namespace NUMINAMATH_GPT_simplify_eval_expression_l1981_198103

theorem simplify_eval_expression (a b : ℤ) (h₁ : a = 2) (h₂ : b = -1) : 
  ((2 * a + 3 * b) * (2 * a - 3 * b) - (2 * a - b) ^ 2 - 2 * a * b) / (-2 * b) = -7 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_eval_expression_l1981_198103


namespace NUMINAMATH_GPT_complex_pow_diff_zero_l1981_198136

theorem complex_pow_diff_zero {i : ℂ} (h : i^2 = -1) : (2 + i)^(12) - (2 - i)^(12) = 0 := by
  sorry

end NUMINAMATH_GPT_complex_pow_diff_zero_l1981_198136


namespace NUMINAMATH_GPT_boxes_in_pantry_l1981_198198

theorem boxes_in_pantry (b p c: ℕ) (h: p = 100) (hc: c = 50) (g: b = 225) (weeks: ℕ) (consumption: ℕ)
    (total_birdseed: ℕ) (new_boxes: ℕ) (initial_boxes: ℕ) : 
    weeks = 12 → consumption = (100 + 50) * weeks → total_birdseed = 1800 →
    new_boxes = 3 → total_birdseed = b * 8 → initial_boxes = 5 :=
by
  sorry

end NUMINAMATH_GPT_boxes_in_pantry_l1981_198198


namespace NUMINAMATH_GPT_jed_speed_l1981_198157

theorem jed_speed
  (posted_speed_limit : ℕ := 50)
  (fine_per_mph_over_limit : ℕ := 16)
  (red_light_fine : ℕ := 75)
  (cellphone_fine : ℕ := 120)
  (parking_fine : ℕ := 50)
  (total_red_light_fines : ℕ := 2 * red_light_fine)
  (total_parking_fines : ℕ := 3 * parking_fine)
  (total_fine : ℕ := 1046)
  (non_speeding_fines : ℕ := total_red_light_fines + cellphone_fine + total_parking_fines)
  (speeding_fine : ℕ := total_fine - non_speeding_fines)
  (mph_over_limit : ℕ := speeding_fine / fine_per_mph_over_limit):
  (posted_speed_limit + mph_over_limit) = 89 :=
by
  sorry

end NUMINAMATH_GPT_jed_speed_l1981_198157


namespace NUMINAMATH_GPT_totalPeaches_l1981_198107

-- Definition of conditions in the problem
def redPeaches : Nat := 4
def greenPeaches : Nat := 6
def numberOfBaskets : Nat := 1

-- Mathematical proof problem
theorem totalPeaches : numberOfBaskets * (redPeaches + greenPeaches) = 10 := by
  sorry

end NUMINAMATH_GPT_totalPeaches_l1981_198107


namespace NUMINAMATH_GPT_philip_farm_animal_count_l1981_198194

def number_of_cows : ℕ := 20

def number_of_ducks : ℕ := number_of_cows * 3 / 2

def total_cows_and_ducks : ℕ := number_of_cows + number_of_ducks

def number_of_pigs : ℕ := total_cows_and_ducks / 5

def total_animals : ℕ := total_cows_and_ducks + number_of_pigs

theorem philip_farm_animal_count : total_animals = 60 := by
  sorry

end NUMINAMATH_GPT_philip_farm_animal_count_l1981_198194


namespace NUMINAMATH_GPT_no_prize_for_A_l1981_198195

variable (A B C D : Prop)

theorem no_prize_for_A 
  (hA : A → B) 
  (hB : B → C) 
  (hC : ¬D → ¬C) 
  (exactly_one_did_not_win : (¬A ∧ B ∧ C ∧ D) ∨ (A ∧ ¬B ∧ C ∧ D) ∨ (A ∧ B ∧ ¬C ∧ D) ∨ (A ∧ B ∧ C ∧ ¬D)) 
: ¬A := 
sorry

end NUMINAMATH_GPT_no_prize_for_A_l1981_198195


namespace NUMINAMATH_GPT_side_length_c_4_l1981_198102

theorem side_length_c_4 (A : ℝ) (b S c : ℝ) 
  (hA : A = 120) (hb : b = 2) (hS : S = 2 * Real.sqrt 3) : 
  c = 4 :=
sorry

end NUMINAMATH_GPT_side_length_c_4_l1981_198102


namespace NUMINAMATH_GPT_f_50_value_l1981_198162

def f : ℝ → ℝ := sorry

axiom f_condition : ∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3*x + 2) = 9 * x^2 - 15 * x

theorem f_50_value : f 50 = 146 :=
by
  sorry

end NUMINAMATH_GPT_f_50_value_l1981_198162


namespace NUMINAMATH_GPT_expression_min_value_l1981_198131

theorem expression_min_value (a b c k : ℝ) (h1 : a < c) (h2 : c < b) (h3 : b = k * c) (h4 : k > 1) :
  (1 : ℝ) / c^2 * ((k * c - a)^2 + (a + c)^2 + (c - a)^2) ≥ k^2 / 3 + 2 :=
sorry

end NUMINAMATH_GPT_expression_min_value_l1981_198131


namespace NUMINAMATH_GPT_travel_time_l1981_198174

theorem travel_time (speed distance time : ℕ) (h_speed : speed = 60) (h_distance : distance = 180) : 
  time = distance / speed → time = 3 := by
  sorry

end NUMINAMATH_GPT_travel_time_l1981_198174


namespace NUMINAMATH_GPT_total_payment_leila_should_pay_l1981_198171

-- Definitions of the conditions
def chocolateCakes := 3
def chocolatePrice := 12
def strawberryCakes := 6
def strawberryPrice := 22

-- Mathematical equivalent proof problem
theorem total_payment_leila_should_pay : 
  chocolateCakes * chocolatePrice + strawberryCakes * strawberryPrice = 168 := 
by 
  sorry

end NUMINAMATH_GPT_total_payment_leila_should_pay_l1981_198171
