import Mathlib

namespace NUMINAMATH_GPT_solution_to_system_l1834_183491

theorem solution_to_system (x y z : ℝ) (h1 : x^2 + y^2 = 6 * z) (h2 : y^2 + z^2 = 6 * x) (h3 : z^2 + x^2 = 6 * y) :
  (x = 3) ∧ (y = 3) ∧ (z = 3) :=
sorry

end NUMINAMATH_GPT_solution_to_system_l1834_183491


namespace NUMINAMATH_GPT_c_minus_a_value_l1834_183459

theorem c_minus_a_value (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50)
  (h2 : (b + c) / 2 = 70) : 
  c - a = 40 :=
by 
  sorry

end NUMINAMATH_GPT_c_minus_a_value_l1834_183459


namespace NUMINAMATH_GPT_desired_average_sale_l1834_183441

theorem desired_average_sale
  (sale1 sale2 sale3 sale4 sale5 sale6 : ℕ)
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h4 : sale4 = 7230)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 7991) :
  (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = 7000 :=
by
  sorry

end NUMINAMATH_GPT_desired_average_sale_l1834_183441


namespace NUMINAMATH_GPT_smaller_integer_of_two_digits_l1834_183410

theorem smaller_integer_of_two_digits (a b : ℕ) (ha : 10 ≤ a ∧ a ≤ 99) (hb: 10 ≤ b ∧ b ≤ 99) (h_diff : a ≠ b)
  (h_eq : (a + b) / 2 = a + b / 100) : a = 49 ∨ b = 49 := 
by
  sorry

end NUMINAMATH_GPT_smaller_integer_of_two_digits_l1834_183410


namespace NUMINAMATH_GPT_ratio_sprite_to_coke_l1834_183469

theorem ratio_sprite_to_coke (total_drink : ℕ) (coke_ounces : ℕ) (mountain_dew_parts : ℕ)
  (parts_coke : ℕ) (parts_mountain_dew : ℕ) (total_parts : ℕ) :
  total_drink = 18 →
  coke_ounces = 6 →
  parts_coke = 2 →
  parts_mountain_dew = 3 →
  total_parts = parts_coke + parts_mountain_dew + ((total_drink - coke_ounces - (parts_mountain_dew * (coke_ounces / parts_coke))) / (coke_ounces / parts_coke)) →
  (total_drink - coke_ounces - (parts_mountain_dew * (coke_ounces / parts_coke))) / coke_ounces = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_sprite_to_coke_l1834_183469


namespace NUMINAMATH_GPT_angle_between_sum_is_pi_over_6_l1834_183472

open Real EuclideanSpace

noncomputable def angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_u := sqrt (u.1^2 + u.2^2)
  let norm_v := sqrt (v.1^2 + v.2^2)
  arccos (dot_product / (norm_u * norm_v))

noncomputable def a : ℝ × ℝ := (1, 0)
noncomputable def b : ℝ × ℝ := (1/2 * cos (π / 3), 1/2 * sin (π / 3))

theorem angle_between_sum_is_pi_over_6 :
  angle_between_vectors (a.1 + 2 * b.1, a.2 + 2 * b.2) b = π / 6 :=
by
  sorry

end NUMINAMATH_GPT_angle_between_sum_is_pi_over_6_l1834_183472


namespace NUMINAMATH_GPT_bridge_max_weight_l1834_183421

variables (M K Mi B : ℝ)

-- Given conditions
def kelly_weight : K = 34 := sorry
def kelly_megan_relation : K = 0.85 * M := sorry
def mike_megan_relation : Mi = M + 5 := sorry
def total_excess : K + M + Mi = B + 19 := sorry

-- Proof goal: The maximum weight the bridge can hold is 100 kg.
theorem bridge_max_weight : B = 100 :=
by
  sorry

end NUMINAMATH_GPT_bridge_max_weight_l1834_183421


namespace NUMINAMATH_GPT_probability_A_seven_rolls_l1834_183480

noncomputable def probability_A_after_n_rolls (n : ℕ) : ℚ :=
  if n = 0 then 1 else 1/3 * (1 - (-1/2)^(n-1))

theorem probability_A_seven_rolls : probability_A_after_n_rolls 7 = 21 / 64 :=
by sorry

end NUMINAMATH_GPT_probability_A_seven_rolls_l1834_183480


namespace NUMINAMATH_GPT_negation_of_existence_l1834_183437

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) :=
sorry

end NUMINAMATH_GPT_negation_of_existence_l1834_183437


namespace NUMINAMATH_GPT_f_neg1_plus_f_2_l1834_183497

def f (x : Int) : Int :=
  if x = -3 then -1
  else if x = -2 then -5
  else if x = -1 then -2
  else if x = 0 then 0
  else if x = 1 then 2
  else if x = 2 then 1
  else if x = 3 then 4
  else 0  -- This handles x values not explicitly in the table, although technically unnecessary.

theorem f_neg1_plus_f_2 : f (-1) + f (2) = -1 := by
  sorry

end NUMINAMATH_GPT_f_neg1_plus_f_2_l1834_183497


namespace NUMINAMATH_GPT_maximum_sine_sum_l1834_183409

open Real

theorem maximum_sine_sum (x y z : ℝ) (hx : 0 ≤ x) (hy : x ≤ π / 2) (hz : 0 ≤ y) (hw : y ≤ π / 2) (hv : 0 ≤ z) (hu : z ≤ π / 2) :
  ∃ M, M = sqrt 2 - 1 ∧ ∀ x y z : ℝ, 0 ≤ x → x ≤ π / 2 → 0 ≤ y → y ≤ π / 2 → 0 ≤ z → z ≤ π / 2 → 
  sin (x - y) + sin (y - z) + sin (z - x) ≤ M :=
by
  sorry

end NUMINAMATH_GPT_maximum_sine_sum_l1834_183409


namespace NUMINAMATH_GPT_unique_intersection_l1834_183482

def line1 (x y : ℝ) : Prop := 3 * x - 2 * y - 9 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 4 * y - 12 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = -1

theorem unique_intersection : ∃! p : ℝ × ℝ, 
                             (line1 p.1 p.2) ∧ 
                             (line2 p.1 p.2) ∧ 
                             (line3 p.1) ∧ 
                             (line4 p.2) ∧ 
                             p = (3, -1) :=
by
  sorry

end NUMINAMATH_GPT_unique_intersection_l1834_183482


namespace NUMINAMATH_GPT_final_elephants_count_l1834_183454

def E_0 : Int := 30000
def R_exodus : Int := 2880
def H_exodus : Int := 4
def R_entry : Int := 1500
def H_entry : Int := 7
def E_final : Int := E_0 - (R_exodus * H_exodus) + (R_entry * H_entry)

theorem final_elephants_count : E_final = 28980 := by
  sorry

end NUMINAMATH_GPT_final_elephants_count_l1834_183454


namespace NUMINAMATH_GPT_inequality_solution_set_l1834_183455

theorem inequality_solution_set :
  {x : ℝ | 2 * x^2 - x > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1 / 2} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1834_183455


namespace NUMINAMATH_GPT_percentage_of_students_liking_chess_l1834_183415

theorem percentage_of_students_liking_chess (total_students : ℕ) (basketball_percentage : ℝ) (soccer_percentage : ℝ) 
(identified_chess_or_basketball : ℕ) (students_liking_basketball : ℕ) : 
total_students = 250 ∧ basketball_percentage = 0.40 ∧ soccer_percentage = 0.28 ∧ identified_chess_or_basketball = 125 ∧ 
students_liking_basketball = 100 → ∃ C : ℝ, C = 0.10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_students_liking_chess_l1834_183415


namespace NUMINAMATH_GPT_hexagon_angle_Q_l1834_183442

theorem hexagon_angle_Q
  (a1 a2 a3 a4 a5 : ℝ)
  (h1 : a1 = 134) 
  (h2 : a2 = 98) 
  (h3 : a3 = 120) 
  (h4 : a4 = 110) 
  (h5 : a5 = 96) 
  (sum_hexagon_angles : a1 + a2 + a3 + a4 + a5 + Q = 720) : 
  Q = 162 := by {
  sorry
}

end NUMINAMATH_GPT_hexagon_angle_Q_l1834_183442


namespace NUMINAMATH_GPT_find_functions_l1834_183463

-- Define the function f and its properties.
variable {f : ℝ → ℝ}

-- Define the condition given in the problem as a hypothesis.
def condition (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x * f x + f y) = y + f x ^ 2

-- State the theorem we want to prove.
theorem find_functions (hf : condition f) : (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
  sorry

end NUMINAMATH_GPT_find_functions_l1834_183463


namespace NUMINAMATH_GPT_find_other_number_l1834_183426

theorem find_other_number
  (n m lcm gcf : ℕ)
  (h_n : n = 40)
  (h_lcm : lcm = 56)
  (h_gcf : gcf = 10)
  (h_lcm_gcf : lcm * gcf = n * m) : m = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l1834_183426


namespace NUMINAMATH_GPT_roland_thread_length_l1834_183449

noncomputable def length_initial : ℝ := 12
noncomputable def length_two_thirds : ℝ := (2 / 3) * length_initial
noncomputable def length_increased : ℝ := length_initial + length_two_thirds
noncomputable def length_half_increased : ℝ := (1 / 2) * length_increased
noncomputable def length_total : ℝ := length_increased + length_half_increased
noncomputable def length_inches : ℝ := length_total / 2.54

theorem roland_thread_length : length_inches = 11.811 :=
by sorry

end NUMINAMATH_GPT_roland_thread_length_l1834_183449


namespace NUMINAMATH_GPT_ratio_proof_l1834_183457

theorem ratio_proof (x y z s : ℝ) (h1 : x < y) (h2 : y < z)
    (h3 : (x : ℝ) / y = y / z) (h4 : x + y + z = s) (h5 : x + y = z) :
    (x / y = (-1 + Real.sqrt 5) / 2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_proof_l1834_183457


namespace NUMINAMATH_GPT_distinct_real_solutions_l1834_183495

theorem distinct_real_solutions
  (a b c d e : ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃ x₁ x₂ x₃ x₄ : ℝ,
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (x₁ - a) * (x₁ - b) * (x₁ - c) * (x₁ - d) +
    (x₁ - a) * (x₁ - b) * (x₁ - c) * (x₁ - e) +
    (x₁ - a) * (x₁ - b) * (x₁ - d) * (x₁ - e) +
    (x₁ - a) * (x₁ - c) * (x₁ - d) * (x₁ - e) +
    (x₁ - b) * (x₁ - c) * (x₁ - d) * (x₁ - e) = 0 ∧
    (x₂ - a) * (x₂ - b) * (x₂ - c) * (x₂ - d) +
    (x₂ - a) * (x₂ - b) * (x₂ - c) * (x₂ - e) +
    (x₂ - a) * (x₂ - b) * (x₂ - d) * (x₂ - e) +
    (x₂ - a) * (x₂ - c) * (x₂ - d) * (x₂ - e) +
    (x₂ - b) * (x₂ - c) * (x₂ - d) * (x₂ - e) = 0 ∧
    (x₃ - a) * (x₃ - b) * (x₃ - c) * (x₃ - d) +
    (x₃ - a) * (x₃ - b) * (x₃ - c) * (x₃ - e) +
    (x₃ - a) * (x₃ - b) * (x₃ - d) * (x₃ - e) +
    (x₃ - a) * (x₃ - c) * (x₃ - d) * (x₃ - e) +
    (x₃ - b) * (x₃ - c) * (x₃ - d) * (x₃ - e) = 0 ∧
    (x₄ - a) * (x₄ - b) * (x₄ - c) * (x₄ - d) +
    (x₄ - a) * (x₄ - b) * (x₄ - c) * (x₄ - e) +
    (x₄ - a) * (x₄ - b) * (x₄ - d) * (x₄ - e) +
    (x₄ - a) * (x₄ - c) * (x₄ - d) * (x₄ - e) +
    (x₄ - b) * (x₄ - c) * (x₄ - d) * (x₄ - e) = 0 :=
  sorry

end NUMINAMATH_GPT_distinct_real_solutions_l1834_183495


namespace NUMINAMATH_GPT_shaded_region_equality_l1834_183413

-- Define the necessary context and variables
variable {r : ℝ} -- radius of the circle
variable {θ : ℝ} -- angle measured in degrees

-- Define the relevant trigonometric functions
noncomputable def tan_degrees (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)
noncomputable def tan_half_degrees (x : ℝ) : ℝ := Real.tan ((x / 2) * Real.pi / 180)

-- State the theorem we need to prove given the conditions
theorem shaded_region_equality (hθ1 : θ / 2 = 90 - θ) :
  tan_degrees θ + (tan_degrees θ)^2 * tan_half_degrees θ = (θ * Real.pi) / 180 - (θ^2 * Real.pi) / 360 :=
  sorry

end NUMINAMATH_GPT_shaded_region_equality_l1834_183413


namespace NUMINAMATH_GPT_cookies_fit_in_box_l1834_183435

variable (box_capacity_pounds : ℕ)
variable (cookie_weight_ounces : ℕ)
variable (ounces_per_pound : ℕ)

theorem cookies_fit_in_box (h1 : box_capacity_pounds = 40)
                           (h2 : cookie_weight_ounces = 2)
                           (h3 : ounces_per_pound = 16) :
                           box_capacity_pounds * (ounces_per_pound / cookie_weight_ounces) = 320 := by
  sorry

end NUMINAMATH_GPT_cookies_fit_in_box_l1834_183435


namespace NUMINAMATH_GPT_upgrade_days_to_sun_l1834_183436

/-- 
  Determine the minimum number of additional active days required for 
  a user currently at level 2 moons and 1 star to upgrade to 1 sun.
-/
theorem upgrade_days_to_sun (level_new_star : ℕ) (level_new_moon : ℕ) (active_days_initial : ℕ) : 
  active_days_initial =  9 * (9 + 4) → 
  level_new_star = 1 → 
  level_new_moon = 2 → 
  ∃ (days_required : ℕ), 
    (days_required + active_days_initial = 16 * (16 + 4)) ∧ (days_required = 203) :=
by
  sorry

end NUMINAMATH_GPT_upgrade_days_to_sun_l1834_183436


namespace NUMINAMATH_GPT_greatest_length_of_pieces_l1834_183433

theorem greatest_length_of_pieces (a b c : ℕ) (ha : a = 48) (hb : b = 60) (hc : c = 72) :
  Nat.gcd (Nat.gcd a b) c = 12 := by
  sorry

end NUMINAMATH_GPT_greatest_length_of_pieces_l1834_183433


namespace NUMINAMATH_GPT_lies_on_new_ellipse_lies_on_new_hyperbola_l1834_183420

variable (x y c d a : ℝ)

def new_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

-- Definition for new ellipse.
def is_new_ellipse (E : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  new_distance E F1 + new_distance E F2 = 2 * a

-- Definition for new hyperbola.
def is_new_hyperbola (H : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  |new_distance H F1 - new_distance H F2| = 2 * a

-- The point E lies on the new ellipse.
theorem lies_on_new_ellipse
  (E F1 F2 : ℝ × ℝ) (a : ℝ) :
  is_new_ellipse E F1 F2 a :=
by sorry

-- The point H lies on the new hyperbola.
theorem lies_on_new_hyperbola
  (H F1 F2 : ℝ × ℝ) (a : ℝ) :
  is_new_hyperbola H F1 F2 a :=
by sorry

end NUMINAMATH_GPT_lies_on_new_ellipse_lies_on_new_hyperbola_l1834_183420


namespace NUMINAMATH_GPT_carrie_expected_strawberries_l1834_183432

noncomputable def calculate_strawberries (base height : ℝ) (plants_per_sq_ft strawberries_per_plant : ℝ) : ℝ :=
  let area := (1/2) * base * height
  let total_plants := plants_per_sq_ft * area
  total_plants * strawberries_per_plant

theorem carrie_expected_strawberries : calculate_strawberries 10 12 5 8 = 2400 :=
by
  /-
  Given: base = 10, height = 12, plants_per_sq_ft = 5, strawberries_per_plant = 8
  - calculate the area of the right triangle garden
  - calculate the total number of plants
  - calculate the total number of strawberries
  -/
  sorry

end NUMINAMATH_GPT_carrie_expected_strawberries_l1834_183432


namespace NUMINAMATH_GPT_Rachel_age_when_father_is_60_l1834_183486

-- Given conditions
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Proof problem statement
theorem Rachel_age_when_father_is_60 : Rachel_age + (60 - Father_age) = 25 :=
by sorry

end NUMINAMATH_GPT_Rachel_age_when_father_is_60_l1834_183486


namespace NUMINAMATH_GPT_box_weight_without_balls_l1834_183499

theorem box_weight_without_balls :
  let number_of_balls := 30
  let weight_per_ball := 0.36
  let total_weight_with_balls := 11.26
  let total_weight_of_balls := number_of_balls * weight_per_ball
  let weight_of_box := total_weight_with_balls - total_weight_of_balls
  weight_of_box = 0.46 :=
by 
  sorry

end NUMINAMATH_GPT_box_weight_without_balls_l1834_183499


namespace NUMINAMATH_GPT_length_increase_percentage_l1834_183407

theorem length_increase_percentage 
  (L B : ℝ)
  (x : ℝ)
  (h1 : B' = B * 0.8)
  (h2 : L' = L * (1 + x / 100))
  (h3 : A = L * B)
  (h4 : A' = L' * B')
  (h5 : A' = A * 1.04) 
  : x = 30 :=
sorry

end NUMINAMATH_GPT_length_increase_percentage_l1834_183407


namespace NUMINAMATH_GPT_find_a_l1834_183479

-- Define the real numbers x, y, and a
variables (x y a : ℝ)

-- Define the conditions as premises
axiom cond1 : x + 3 * y + 5 ≥ 0
axiom cond2 : x + y - 1 ≤ 0
axiom cond3 : x + a ≥ 0

-- Define z as x + 2y and state its minimum value is -4
def z : ℝ := x + 2 * y
axiom min_z : z = -4

-- The theorem to prove the value of a given the above conditions
theorem find_a : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1834_183479


namespace NUMINAMATH_GPT_trajectory_of_Q_l1834_183456

/-- Let P(m, n) be a point moving on the circle x^2 + y^2 = 2.
     The trajectory of the point Q(m+n, 2mn) is y = x^2 - 2. -/
theorem trajectory_of_Q (m n : ℝ) (hyp : m^2 + n^2 = 2) : 
  ∃ x y : ℝ, x = m + n ∧ y = 2 * m * n ∧ y = x^2 - 2 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_Q_l1834_183456


namespace NUMINAMATH_GPT_tangent_line_min_slope_equation_l1834_183417

def curve (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 1

theorem tangent_line_min_slope_equation :
  ∃ (k : ℝ) (b : ℝ), (∀ x y, y = curve x → y = k * x + b)
  ∧ (k = 3)
  ∧ (b = -2)
  ∧ (3 * x - y - 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_min_slope_equation_l1834_183417


namespace NUMINAMATH_GPT_tan_eq_243_deg_l1834_183475

theorem tan_eq_243_deg (n : ℤ) : -90 < n ∧ n < 90 ∧ Real.tan (n * Real.pi / 180) = Real.tan (243 * Real.pi / 180) ↔ n = 63 :=
by sorry

end NUMINAMATH_GPT_tan_eq_243_deg_l1834_183475


namespace NUMINAMATH_GPT_mod_exp_equivalence_l1834_183462

theorem mod_exp_equivalence :
  (81^1814 - 25^1814) % 7 = 0 := by
  sorry

end NUMINAMATH_GPT_mod_exp_equivalence_l1834_183462


namespace NUMINAMATH_GPT_brother_raking_time_l1834_183419

theorem brother_raking_time (x : ℝ) (hx : x > 0)
  (h_combined : (1 / 30) + (1 / x) = 1 / 18) : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_brother_raking_time_l1834_183419


namespace NUMINAMATH_GPT_range_of_m_l1834_183414

variable (x y m : ℝ)

theorem range_of_m (h1 : Real.sin x = m * (Real.sin y)^3)
                   (h2 : Real.cos x = m * (Real.cos y)^3) :
                   1 ≤ m ∧ m ≤ Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1834_183414


namespace NUMINAMATH_GPT_area_of_triangle_bounded_by_line_and_axes_l1834_183401

theorem area_of_triangle_bounded_by_line_and_axes (x y : ℝ) (hx : 3 * x + 2 * y = 12) :
  ∃ (area : ℝ), area = 12 := by
sorry

end NUMINAMATH_GPT_area_of_triangle_bounded_by_line_and_axes_l1834_183401


namespace NUMINAMATH_GPT_value_at_7_6_l1834_183492

noncomputable def f : ℝ → ℝ := sorry

lemma periodic_f (x : ℝ) : f (x + 4) = f x := sorry

lemma f_on_interval (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 2) : f x = x := sorry

theorem value_at_7_6 : f 7.6 = -0.4 :=
by
  have p := periodic_f 7.6
  have q := periodic_f 3.6
  have r := f_on_interval (-0.4)
  sorry

end NUMINAMATH_GPT_value_at_7_6_l1834_183492


namespace NUMINAMATH_GPT_complex_division_result_l1834_183446

theorem complex_division_result :
  let z := (⟨0, 1⟩ - ⟨2, 0⟩) / (⟨1, 0⟩ + ⟨0, 1⟩ : ℂ)
  let a := z.re
  let b := z.im
  a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_complex_division_result_l1834_183446


namespace NUMINAMATH_GPT_intersection_P_Q_l1834_183424

def P : Set ℝ := { x | x > 1 }
def Q : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_P_Q : P ∩ Q = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l1834_183424


namespace NUMINAMATH_GPT_number_of_hens_l1834_183438

theorem number_of_hens (H C : ℕ) 
  (h1 : H + C = 60) 
  (h2 : 2 * H + 4 * C = 200) : H = 20 :=
sorry

end NUMINAMATH_GPT_number_of_hens_l1834_183438


namespace NUMINAMATH_GPT_empty_to_occupied_ratio_of_spheres_in_cylinder_package_l1834_183493

theorem empty_to_occupied_ratio_of_spheres_in_cylinder_package
  (R : ℝ) 
  (volume_sphere : ℝ)
  (volume_cylinder : ℝ)
  (sphere_occupies_fraction : ∀ R : ℝ, volume_sphere = (2 / 3) * volume_cylinder) 
  (num_spheres : ℕ) 
  (h_num_spheres : num_spheres = 5) :
  (num_spheres : ℝ) * volume_sphere = (5 * (2 / 3) * π * R^3) → 
  volume_sphere = (4 / 3) * π * R^3 → 
  volume_cylinder = 2 * π * R^3 → 
  (volume_cylinder - volume_sphere) / volume_sphere = 1 / 2 := by 
  sorry

end NUMINAMATH_GPT_empty_to_occupied_ratio_of_spheres_in_cylinder_package_l1834_183493


namespace NUMINAMATH_GPT_circle_diameter_l1834_183458

theorem circle_diameter (A : ℝ) (h : A = 64 * Real.pi) : ∃ d : ℝ, d = 16 :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_l1834_183458


namespace NUMINAMATH_GPT_find_a_values_l1834_183496

noncomputable def function_a_max_value (a : ℝ) : ℝ :=
  a^2 + 2 * a - 9

theorem find_a_values (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : function_a_max_value a = 6) : 
    a = 3 ∨ a = 1/3 :=
  sorry

end NUMINAMATH_GPT_find_a_values_l1834_183496


namespace NUMINAMATH_GPT_min_value_at_x_eq_2_l1834_183439

theorem min_value_at_x_eq_2 (x : ℝ) (h : x > 1) : 
  x + 1/(x-1) = 3 ↔ x = 2 :=
by sorry

end NUMINAMATH_GPT_min_value_at_x_eq_2_l1834_183439


namespace NUMINAMATH_GPT_ratio_largest_middle_l1834_183403

-- Definitions based on given conditions
def A : ℕ := 24  -- smallest number
def B : ℕ := 40  -- middle number
def C : ℕ := 56  -- largest number

theorem ratio_largest_middle (h1 : C = 56) (h2 : A = C - 32) (h3 : A = 24) (h4 : B = 40) :
  C / B = 7 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_largest_middle_l1834_183403


namespace NUMINAMATH_GPT_initial_amount_is_800_l1834_183483

variables (P R : ℝ)

theorem initial_amount_is_800
  (h1 : 956 = P * (1 + 3 * R / 100))
  (h2 : 1052 = P * (1 + 3 * (R + 4) / 100)) :
  P = 800 :=
sorry

end NUMINAMATH_GPT_initial_amount_is_800_l1834_183483


namespace NUMINAMATH_GPT_miranda_pillows_l1834_183490

theorem miranda_pillows (feathers_per_pound : ℕ) (total_feathers : ℕ) (pillows : ℕ)
  (h1 : feathers_per_pound = 300) (h2 : total_feathers = 3600) (h3 : pillows = 6) :
  (total_feathers / feathers_per_pound) / pillows = 2 := by
  sorry

end NUMINAMATH_GPT_miranda_pillows_l1834_183490


namespace NUMINAMATH_GPT_num_values_divisible_by_120_l1834_183406

theorem num_values_divisible_by_120 (n : ℕ) (h_seq : ∀ n, ∃ k, n = k * (k + 1)) :
  ∃ k, k = 8 := sorry

end NUMINAMATH_GPT_num_values_divisible_by_120_l1834_183406


namespace NUMINAMATH_GPT_math_problem_l1834_183476
open Real

noncomputable def problem_statement : Prop :=
  let a := 99
  let b := 3
  let c := 20
  let area := (99 * sqrt 3) / 20
  a + b + c = 122 ∧ 
  ∃ (AB: ℝ) (QR: ℝ), AB = 14 ∧ QR = 3 * sqrt 3 ∧ area = (1 / 2) * QR * (QR / (2 * (sqrt 3 / 2))) * (sqrt 3 / 2)

theorem math_problem : problem_statement := by
  sorry

end NUMINAMATH_GPT_math_problem_l1834_183476


namespace NUMINAMATH_GPT_solve_equation_l1834_183474

def f (x : ℝ) := |3 * x - 2|

theorem solve_equation 
  (x : ℝ) 
  (a : ℝ)
  (hx1 : x ≠ 3)
  (hx2 : x ≠ 0) :
  (3 * x - 2) ^ 2 = (x + a) ^ 2 ↔
  (a = -4 * x + 2) ∨ (a = 2 * x - 2) := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1834_183474


namespace NUMINAMATH_GPT_cookies_in_each_bag_l1834_183418

-- Definitions based on the conditions
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41
def baggies : ℕ := 6

-- Assertion of the correct answer
theorem cookies_in_each_bag : 
  (chocolate_chip_cookies + oatmeal_cookies) / baggies = 9 := by
  sorry

end NUMINAMATH_GPT_cookies_in_each_bag_l1834_183418


namespace NUMINAMATH_GPT_find_initial_amount_l1834_183460

noncomputable def initial_amount (diff : ℝ) : ℝ :=
  diff / (1.4641 - 1.44)

theorem find_initial_amount
  (diff : ℝ)
  (h : diff = 964.0000000000146) :
  initial_amount diff = 40000 :=
by
  -- the steps to prove this can be added here later
  sorry

end NUMINAMATH_GPT_find_initial_amount_l1834_183460


namespace NUMINAMATH_GPT_part_one_part_two_l1834_183443

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 - (a + 1/a) * x + 1

theorem part_one (x : ℝ) : f x (1/2) ≤ 0 ↔ (1/2 ≤ x ∧ x ≤ 2) :=
by
  sorry

theorem part_two (x a : ℝ) (h : a > 0) : 
  ((a < 1) → (f x a ≤ 0 ↔ (a ≤ x ∧ x ≤ 1/a))) ∧
  ((a > 1) → (f x a ≤ 0 ↔ (1/a ≤ x ∧ x ≤ a))) ∧
  ((a = 1) → (f x a ≤ 0 ↔ (x = 1))) :=
by
  sorry

end NUMINAMATH_GPT_part_one_part_two_l1834_183443


namespace NUMINAMATH_GPT_tournament_cycle_exists_l1834_183427

theorem tournament_cycle_exists :
  ∃ (A B C : Fin 12), 
  (∃ M : Fin 12 → Fin 12 → Bool, 
    (∀ p : Fin 12, ∃ q : Fin 12, q ≠ p ∧ M p q) ∧
    M A B = true ∧ M B C = true ∧ M C A = true) :=
sorry

end NUMINAMATH_GPT_tournament_cycle_exists_l1834_183427


namespace NUMINAMATH_GPT_exponent_equality_l1834_183473

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
sorry

end NUMINAMATH_GPT_exponent_equality_l1834_183473


namespace NUMINAMATH_GPT_time_taken_y_alone_l1834_183440

-- Define the work done in terms of rates
def work_done (Rx Ry Rz : ℝ) (W : ℝ) :=
  Rx = W / 8 ∧ (Ry + Rz) = W / 6 ∧ (Rx + Rz) = W / 4

-- Prove that the time taken by y alone is 24 hours
theorem time_taken_y_alone (Rx Ry Rz W : ℝ) (h : work_done Rx Ry Rz W) :
  (1 / Ry) = 24 :=
by
  sorry

end NUMINAMATH_GPT_time_taken_y_alone_l1834_183440


namespace NUMINAMATH_GPT_num_natural_numbers_divisible_by_7_l1834_183431

theorem num_natural_numbers_divisible_by_7 (a b : ℕ) (h₁ : 200 ≤ a) (h₂ : b ≤ 400) (h₃ : a = 203) (h₄ : b = 399) :
  (b - a) / 7 + 1 = 29 := 
by
  sorry

end NUMINAMATH_GPT_num_natural_numbers_divisible_by_7_l1834_183431


namespace NUMINAMATH_GPT_device_identification_l1834_183451

def sum_of_device_numbers (numbers : List ℕ) : ℕ :=
  numbers.foldr (· + ·) 0

def is_standard_device (d : List ℕ) : Prop :=
  (d = [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧ (sum_of_device_numbers d = 45)

theorem device_identification (d : List ℕ) : 
  (sum_of_device_numbers d = 45) → is_standard_device d :=
by
  sorry

end NUMINAMATH_GPT_device_identification_l1834_183451


namespace NUMINAMATH_GPT_correct_total_annual_salary_expression_l1834_183429

def initial_workers : ℕ := 8
def initial_salary : ℝ := 1.0 -- in ten thousand yuan
def new_workers : ℕ := 3
def new_worker_initial_salary : ℝ := 0.8 -- in ten thousand yuan
def salary_increase_rate : ℝ := 1.2 -- 20% increase each year

def total_annual_salary (n : ℕ) : ℝ :=
  (3 * n + 5) * salary_increase_rate^n + (new_workers * new_worker_initial_salary)

theorem correct_total_annual_salary_expression (n : ℕ) :
  total_annual_salary n = (3 * n + 5) * 1.2^n + 2.4 := 
by
  sorry

end NUMINAMATH_GPT_correct_total_annual_salary_expression_l1834_183429


namespace NUMINAMATH_GPT_purely_imaginary_z_implies_m_zero_l1834_183416

theorem purely_imaginary_z_implies_m_zero (m : ℝ) :
  m * (m + 1) = 0 → m ≠ -1 := by sorry

end NUMINAMATH_GPT_purely_imaginary_z_implies_m_zero_l1834_183416


namespace NUMINAMATH_GPT_fraction_simplification_l1834_183402

theorem fraction_simplification :
  (3 / 7 + 5 / 8 + 2 / 9) / (5 / 12 + 1 / 4) = 643 / 336 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1834_183402


namespace NUMINAMATH_GPT_waiter_tables_l1834_183425

theorem waiter_tables (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (tables : ℕ) :
  initial_customers = 62 → 
  customers_left = 17 → 
  people_per_table = 9 → 
  remaining_customers = initial_customers - customers_left →
  tables = remaining_customers / people_per_table →
  tables = 5 :=
by
  intros hinitial hleft hpeople hremaining htables
  rw [hinitial, hleft, hpeople] at *
  simp at *
  sorry

end NUMINAMATH_GPT_waiter_tables_l1834_183425


namespace NUMINAMATH_GPT_Ben_cards_left_l1834_183430

def BenInitialBasketballCards : ℕ := 4 * 10
def BenInitialBaseballCards : ℕ := 5 * 8
def BenTotalInitialCards : ℕ := BenInitialBasketballCards + BenInitialBaseballCards
def BenGivenCards : ℕ := 58
def BenRemainingCards : ℕ := BenTotalInitialCards - BenGivenCards

theorem Ben_cards_left : BenRemainingCards = 22 :=
by 
  -- The proof will be placed here.
  sorry

end NUMINAMATH_GPT_Ben_cards_left_l1834_183430


namespace NUMINAMATH_GPT_distinct_ordered_pairs_l1834_183484

theorem distinct_ordered_pairs (a b : ℕ) (h : a + b = 40) (ha : a > 0) (hb : b > 0) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 39 ∧ ∀ p ∈ pairs, p.1 + p.2 = 40 := 
sorry

end NUMINAMATH_GPT_distinct_ordered_pairs_l1834_183484


namespace NUMINAMATH_GPT_range_of_a_l1834_183412

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x - a) * (1 - x - a) < 1) → -1/2 < a ∧ a < 3/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1834_183412


namespace NUMINAMATH_GPT_f_7_5_l1834_183450

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x
axiom f_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem f_7_5 : f 7.5 = -0.5 := by
  sorry

end NUMINAMATH_GPT_f_7_5_l1834_183450


namespace NUMINAMATH_GPT_new_person_weight_l1834_183465

theorem new_person_weight 
    (W : ℝ) -- total weight of original 8 people
    (x : ℝ) -- weight of the new person
    (increase_by : ℝ) -- average weight increases by 2.5 kg
    (replaced_weight : ℝ) -- weight of the replaced person (55 kg)
    (h1 : increase_by = 2.5)
    (h2 : replaced_weight = 55)
    (h3 : x = replaced_weight + (8 * increase_by)) : x = 75 := 
by
  sorry

end NUMINAMATH_GPT_new_person_weight_l1834_183465


namespace NUMINAMATH_GPT_right_triangle_median_square_l1834_183478

theorem right_triangle_median_square (a b c k_a k_b : ℝ) :
  c = Real.sqrt (a^2 + b^2) → -- c is the hypotenuse
  k_a = Real.sqrt ((2 * b^2 + 2 * (a^2 + b^2) - a^2) / 4) → -- k_a is the median to side a
  k_b = Real.sqrt ((2 * a^2 + 2 * (a^2 + b^2) - b^2) / 4) → -- k_b is the median to side b
  c^2 = (4 / 5) * (k_a^2 + k_b^2) :=
by
  intros h_c h_ka h_kb
  sorry

end NUMINAMATH_GPT_right_triangle_median_square_l1834_183478


namespace NUMINAMATH_GPT_parabola_equation_l1834_183405

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

-- Define the standard equation form of the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Define the right vertex of the hyperbola
def right_vertex (a : ℝ) : ℝ × ℝ :=
  (a, 0)

-- State the final proof problem
theorem parabola_equation :
  hyperbola 4 0 →
  parabola 8 x y →
  y^2 = 16 * x :=
by
  -- Skip the proof for now
  sorry

end NUMINAMATH_GPT_parabola_equation_l1834_183405


namespace NUMINAMATH_GPT_common_root_values_max_n_and_a_range_l1834_183471

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a+1) * x - 4 * (a+5)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x + 5

-- Part 1
theorem common_root_values (a : ℝ) :
  (∃ x : ℝ, f a x = 0 ∧ g a x = 0) → a = -9/16 ∨ a = -6 ∨ a = -4 ∨ a = 0 :=
sorry

-- Part 2
theorem max_n_and_a_range (a : ℝ) (m n : ℕ) (x0 : ℝ) :
  (m < n ∧ (m : ℝ) < x0 ∧ x0 < (n : ℝ) ∧ f a x0 < 0 ∧ g a x0 < 0) →
  n = 4 ∧ -1 ≤ a ∧ a ≤ -2/9 :=
sorry

end NUMINAMATH_GPT_common_root_values_max_n_and_a_range_l1834_183471


namespace NUMINAMATH_GPT_B2F_base16_to_base10_l1834_183445

theorem B2F_base16_to_base10 :
  let d2 := 11
  let d1 := 2
  let d0 := 15
  d2 * 16^2 + d1 * 16^1 + d0 * 16^0 = 2863 :=
by
  let d2 := 11
  let d1 := 2
  let d0 := 15
  sorry

end NUMINAMATH_GPT_B2F_base16_to_base10_l1834_183445


namespace NUMINAMATH_GPT_greatest_product_of_digits_l1834_183408

theorem greatest_product_of_digits :
  ∀ a b : ℕ, (10 * a + b) % 35 = 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  ∃ ab_max : ℕ, ab_max = a * b ∧ ab_max = 15 :=
by
  sorry

end NUMINAMATH_GPT_greatest_product_of_digits_l1834_183408


namespace NUMINAMATH_GPT_find_a_integer_condition_l1834_183489

theorem find_a_integer_condition (a : ℚ) :
  (∀ n : ℕ, (a * (n * (n+2) * (n+3) * (n+4)) : ℚ).den = 1) ↔ ∃ k : ℤ, a = k / 6 := 
sorry

end NUMINAMATH_GPT_find_a_integer_condition_l1834_183489


namespace NUMINAMATH_GPT_running_race_l1834_183488

-- Define participants
inductive Participant : Type
| Anna
| Bella
| Csilla
| Dora

open Participant

-- Define positions
@[ext] structure Position :=
(first : Participant)
(last : Participant)

-- Conditions:
def conditions (p : Participant) (q : Participant) (r : Participant) (s : Participant)
  (pa : Position) : Prop :=
  (pa.first = r) ∧ -- Csilla was first
  (pa.first ≠ q) ∧ -- Bella was not first
  (pa.first ≠ p) ∧ (pa.last ≠ p) ∧ -- Anna was not first or last
  (pa.last = s) -- Dóra's statement about being last

-- Definition of the liar
def liar (p : Participant) : Prop :=
  p = Dora

-- Proof problem
theorem running_race : ∃ (pa : Position), liar Dora ∧ (pa.first = Csilla) :=
  sorry

end NUMINAMATH_GPT_running_race_l1834_183488


namespace NUMINAMATH_GPT_center_of_circle_is_at_10_3_neg5_l1834_183466

noncomputable def center_of_tangent_circle (x y : ℝ) : Prop :=
  (6 * x - 5 * y = 50 ∨ 6 * x - 5 * y = -20) ∧ (3 * x + 2 * y = 0)

theorem center_of_circle_is_at_10_3_neg5 :
  ∃ x y : ℝ, center_of_tangent_circle x y ∧ x = 10 / 3 ∧ y = -5 :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_is_at_10_3_neg5_l1834_183466


namespace NUMINAMATH_GPT_inequality_proof_l1834_183468

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y ≤ 1) : 
  8 * x * y ≤ 5 * x * (1 - x) + 5 * y * (1 - y) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1834_183468


namespace NUMINAMATH_GPT_find_initial_terms_l1834_183434

theorem find_initial_terms (a : ℕ → ℕ) (h : ∀ n, a (n + 3) = a (n + 2) * (a (n + 1) + 2 * a n))
  (a6 : a 6 = 2288) : a 1 = 5 ∧ a 2 = 1 ∧ a 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_terms_l1834_183434


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1834_183461

theorem sum_of_two_numbers (x y : ℕ) (h1 : y = x + 4) (h2 : y = 30) : x + y = 56 :=
by
  -- Asserts the conditions and goal statement
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1834_183461


namespace NUMINAMATH_GPT_three_digit_identical_divisible_by_37_l1834_183470

theorem three_digit_identical_divisible_by_37 (A : ℕ) (h : A ≤ 9) : 37 ∣ (111 * A) :=
sorry

end NUMINAMATH_GPT_three_digit_identical_divisible_by_37_l1834_183470


namespace NUMINAMATH_GPT_min_PA_squared_plus_PB_squared_l1834_183494

-- Let points A, B, and the circle be defined as given in the problem.
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-2, 0⟩
def B : Point := ⟨2, 0⟩

def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

def PA_squared (P : Point) : ℝ :=
  (P.x - A.x)^2 + (P.y - A.y)^2

def PB_squared (P : Point) : ℝ :=
  (P.x - B.x)^2 + (P.y - B.y)^2

def F (P : Point) : ℝ := PA_squared P + PB_squared P

theorem min_PA_squared_plus_PB_squared : ∃ P : Point, on_circle P ∧ F P = 26 := sorry

end NUMINAMATH_GPT_min_PA_squared_plus_PB_squared_l1834_183494


namespace NUMINAMATH_GPT_stadium_length_in_feet_l1834_183422

-- Assume the length of the stadium is 80 yards
def stadium_length_yards := 80

-- Assume the conversion factor is 3 feet per yard
def conversion_factor := 3

-- The length in feet is the product of the length in yards and the conversion factor
def length_in_feet := stadium_length_yards * conversion_factor

-- We want to prove that this length in feet is 240 feet
theorem stadium_length_in_feet : length_in_feet = 240 := by
  -- Definitions and conditions are directly restated here; the proof is sketched as 'sorry'
  sorry

end NUMINAMATH_GPT_stadium_length_in_feet_l1834_183422


namespace NUMINAMATH_GPT_distance_between_lines_correct_l1834_183487

noncomputable def distance_between_parallel_lines 
  (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines_correct :
  distance_between_parallel_lines 4 2 (-2) 1 = 3 * Real.sqrt 5 / 10 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_distance_between_lines_correct_l1834_183487


namespace NUMINAMATH_GPT_circumcircle_eqn_l1834_183453

def point := ℝ × ℝ

def A : point := (-1, 5)
def B : point := (5, 5)
def C : point := (6, -2)

def circ_eq (D E F : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + D * x + E * y + F = 0

theorem circumcircle_eqn :
  ∃ D E F : ℝ, (∀ (p : point), p ∈ [A, B, C] → circ_eq D E F p.1 p.2) ∧
              circ_eq (-4) (-2) (-20) = circ_eq D E F := by
  sorry

end NUMINAMATH_GPT_circumcircle_eqn_l1834_183453


namespace NUMINAMATH_GPT_min_value_of_linear_combination_of_variables_l1834_183444

-- Define the conditions that x and y are positive numbers and satisfy the equation x + 3y = 5xy
def conditions (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y

-- State the theorem that the minimum value of 3x + 4y given the conditions is 5
theorem min_value_of_linear_combination_of_variables (x y : ℝ) (h: conditions x y) : 3 * x + 4 * y ≥ 5 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_of_linear_combination_of_variables_l1834_183444


namespace NUMINAMATH_GPT_profit_percentage_each_portion_l1834_183481

theorem profit_percentage_each_portion (P : ℝ) (total_apples : ℝ) 
  (portion1_percentage : ℝ) (portion2_percentage : ℝ) (total_profit_percentage : ℝ) :
  total_apples = 280 →
  portion1_percentage = 0.4 →
  portion2_percentage = 0.6 →
  total_profit_percentage = 0.3 →
  portion1_percentage * P + portion2_percentage * P = total_profit_percentage →
  P = 0.3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_profit_percentage_each_portion_l1834_183481


namespace NUMINAMATH_GPT_hundredths_digit_of_power_l1834_183498

theorem hundredths_digit_of_power (n : ℕ) (h : n % 20 = 14) : 
  (8 ^ n % 1000) / 100 = 1 :=
by sorry

lemma test_power_hundredths_digit : (8 ^ 1234 % 1000) / 100 = 1 :=
hundredths_digit_of_power 1234 (by norm_num)

end NUMINAMATH_GPT_hundredths_digit_of_power_l1834_183498


namespace NUMINAMATH_GPT_total_value_of_horse_and_saddle_l1834_183464

def saddle_value : ℝ := 12.5
def horse_value : ℝ := 7 * saddle_value

theorem total_value_of_horse_and_saddle : horse_value + saddle_value = 100 := by
  sorry

end NUMINAMATH_GPT_total_value_of_horse_and_saddle_l1834_183464


namespace NUMINAMATH_GPT_sum_of_purchases_l1834_183423

variable (J : ℕ) (K : ℕ)

theorem sum_of_purchases :
  J = 230 →
  2 * J = K + 90 →
  J + K = 600 :=
by
  intros hJ hEq
  rw [hJ] at hEq
  sorry

end NUMINAMATH_GPT_sum_of_purchases_l1834_183423


namespace NUMINAMATH_GPT_pythagorean_triple_example_l1834_183400

noncomputable def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_example :
  is_pythagorean_triple 5 12 13 :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_triple_example_l1834_183400


namespace NUMINAMATH_GPT_simple_interest_rate_l1834_183485

theorem simple_interest_rate (P A T : ℝ) (H1 : P = 1750) (H2 : A = 2000) (H3 : T = 4) :
  ∃ R : ℝ, R = 3.57 ∧ A = P * (1 + (R * T) / 100) :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1834_183485


namespace NUMINAMATH_GPT_inequality_abc_l1834_183447

theorem inequality_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_abc_l1834_183447


namespace NUMINAMATH_GPT_total_handshakes_l1834_183477

-- Definitions and conditions
def num_dwarves := 25
def num_elves := 18

def handshakes_among_dwarves : ℕ := num_dwarves * (num_dwarves - 1) / 2
def handshakes_between_dwarves_and_elves : ℕ := num_elves * num_dwarves

-- Total number of handshakes
theorem total_handshakes : handshakes_among_dwarves + handshakes_between_dwarves_and_elves = 750 := by 
  sorry

end NUMINAMATH_GPT_total_handshakes_l1834_183477


namespace NUMINAMATH_GPT_problem_solution_l1834_183448

theorem problem_solution
  (a b : ℝ)
  (h_eqn : ∃ (a b : ℝ), 3 * a * a + 9 * a - 21 = 0 ∧ 3 * b * b + 9 * b - 21 = 0 )
  (h_vieta_sum : a + b = -3)
  (h_vieta_prod : a * b = -7) :
  (2 * a - 5) * (3 * b - 4) = 47 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1834_183448


namespace NUMINAMATH_GPT_ratio_of_weight_l1834_183404

theorem ratio_of_weight (B : ℝ) : 
    (2 * (4 + B) = 16) → ((B = 4) ∧ (4 + B) / 2 = 4) := by
  intro h
  have h₁ : B = 4 := by
    linarith
  have h₂ : (4 + B) / 2 = 4 := by
    rw [h₁]
    norm_num
  exact ⟨h₁, h₂⟩

end NUMINAMATH_GPT_ratio_of_weight_l1834_183404


namespace NUMINAMATH_GPT_problem_statement_l1834_183411

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x
noncomputable def g (x : ℝ) : ℝ := (2 ^ x) / 2 - 2 / (2 ^ x) - x + 1

theorem problem_statement (a : ℝ) (x₁ x₂ : ℝ) (h₀ : x₁ < x₂)
  (h₁ : f a x₁ = 0) (h₂ : f a x₂ = 0) : g x₁ + g x₂ > 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1834_183411


namespace NUMINAMATH_GPT_abs_inequality_range_l1834_183452

theorem abs_inequality_range (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x + 6| > a) ↔ a < 5 :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_range_l1834_183452


namespace NUMINAMATH_GPT_men_in_first_group_l1834_183428

theorem men_in_first_group (M : ℕ) (h1 : M * 35 = 7 * 50) : M = 10 := by
  sorry

end NUMINAMATH_GPT_men_in_first_group_l1834_183428


namespace NUMINAMATH_GPT_calc_a_minus_3b_l1834_183467

noncomputable def a : ℂ := 5 - 3 * Complex.I
noncomputable def b : ℂ := 2 + 3 * Complex.I

theorem calc_a_minus_3b : a - 3 * b = -1 - 12 * Complex.I := by
  sorry

end NUMINAMATH_GPT_calc_a_minus_3b_l1834_183467
