import Mathlib

namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_is_sqrt_six_l11_1168

def sum_of_squares_eq_seven_times_difference (a b : ℝ) : Prop := 
  a^2 + b^2 = 7 * (a - b)

theorem ratio_of_larger_to_smaller_is_sqrt_six {a b : ℝ} (h : sum_of_squares_eq_seven_times_difference a b) (h1 : a > b) : 
  a / b = Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_is_sqrt_six_l11_1168


namespace NUMINAMATH_GPT_find_q_of_polynomial_l11_1115

noncomputable def Q (x : ℝ) (p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q_of_polynomial (p q d : ℝ) (mean_zeros twice_product sum_coeffs : ℝ)
  (h1 : mean_zeros = -p / 3)
  (h2 : twice_product = -2 * d)
  (h3 : sum_coeffs = 1 + p + q + d)
  (h4 : d = 4)
  (h5 : mean_zeros = twice_product)
  (h6 : sum_coeffs = twice_product) :
  q = -37 :=
sorry

end NUMINAMATH_GPT_find_q_of_polynomial_l11_1115


namespace NUMINAMATH_GPT_inequality_proof_l11_1185

variable {a b c d : ℝ}

theorem inequality_proof
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (h_sum : a + b + c + d = 3) :
  1 / a^3 + 1 / b^3 + 1 / c^3 + 1 / d^3 ≤ 1 / (a^3 * b^3 * c^3 * d^3) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l11_1185


namespace NUMINAMATH_GPT_marina_drive_l11_1186

theorem marina_drive (a b c : ℕ) (x : ℕ) 
  (h1 : 1 ≤ a) 
  (h2 : a + b + c ≤ 9)
  (h3 : 90 * (b - a) = 60 * x)
  (h4 : x = 3 * (b - a) / 2) :
  a = 1 ∧ b = 3 ∧ c = 5 ∧ a^2 + b^2 + c^2 = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_marina_drive_l11_1186


namespace NUMINAMATH_GPT_arithmetic_expression_l11_1157

theorem arithmetic_expression : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l11_1157


namespace NUMINAMATH_GPT_taxi_ride_distance_l11_1102

theorem taxi_ride_distance (initial_fare additional_fare total_fare : ℝ) 
  (initial_distance : ℝ) (additional_distance increment_distance : ℝ) :
  initial_fare = 1.0 →
  additional_fare = 0.45 →
  initial_distance = 1/5 →
  increment_distance = 1/5 →
  total_fare = 7.3 →
  additional_distance = (total_fare - initial_fare) / additional_fare →
  (initial_distance + additional_distance * increment_distance) = 3 := 
by sorry

end NUMINAMATH_GPT_taxi_ride_distance_l11_1102


namespace NUMINAMATH_GPT_intersecting_circles_l11_1170

theorem intersecting_circles (m c : ℝ)
  (h1 : ∃ (x1 y1 x2 y2 : ℝ), x1 = 1 ∧ y1 = 3 ∧ x2 = m ∧ y2 = 1 ∧ x1 ≠ x2 ∧ y1 ≠ y2)
  (h2 : ∀ (x y : ℝ), (x - y + (c / 2) = 0) → (x = 1 ∨ y = 3)) :
  m + c = 3 :=
sorry

end NUMINAMATH_GPT_intersecting_circles_l11_1170


namespace NUMINAMATH_GPT_probability_blue_face_facing_up_l11_1176

-- Define the context
def octahedron_faces : ℕ := 8
def blue_faces : ℕ := 5
def red_faces : ℕ := 3
def total_faces : ℕ := blue_faces + red_faces

-- The probability calculation theorem
theorem probability_blue_face_facing_up (h : total_faces = octahedron_faces) :
  (blue_faces : ℝ) / (octahedron_faces : ℝ) = 5 / 8 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_probability_blue_face_facing_up_l11_1176


namespace NUMINAMATH_GPT_jellybean_probability_l11_1109

/-- A bowl contains 15 jellybeans: five red, three blue, five white, and two green. If you pick four 
    jellybeans from the bowl at random and without replacement, the probability that exactly three will 
    be red is 20/273. -/
theorem jellybean_probability :
  let total_jellybeans := 15
  let red_jellybeans := 5
  let blue_jellybeans := 3
  let white_jellybeans := 5
  let green_jellybeans := 2
  let total_combinations := Nat.choose total_jellybeans 4
  let favorable_combinations := (Nat.choose red_jellybeans 3) * (Nat.choose (total_jellybeans - red_jellybeans) 1)
  let probability := favorable_combinations / total_combinations
  probability = 20 / 273 :=
by
  sorry

end NUMINAMATH_GPT_jellybean_probability_l11_1109


namespace NUMINAMATH_GPT_cube_ratio_sum_l11_1163

theorem cube_ratio_sum (a b : ℝ) (h1 : |a| ≠ |b|) (h2 : (a + b) / (a - b) + (a - b) / (a + b) = 6) :
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 18 / 7 :=
by
  sorry

end NUMINAMATH_GPT_cube_ratio_sum_l11_1163


namespace NUMINAMATH_GPT_evaluate_expression_l11_1113

variable (x y z : ℝ)

theorem evaluate_expression (h : x / (30 - x) + y / (75 - y) + z / (50 - z) = 9) :
  6 / (30 - x) + 15 / (75 - y) + 10 / (50 - z) = 2.4 := 
sorry

end NUMINAMATH_GPT_evaluate_expression_l11_1113


namespace NUMINAMATH_GPT_difference_of_students_l11_1108

variable (G1 G2 G5 : ℕ)

theorem difference_of_students (h1 : G1 + G2 > G2 + G5) (h2 : G5 = G1 - 30) : 
  (G1 + G2) - (G2 + G5) = 30 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_students_l11_1108


namespace NUMINAMATH_GPT_remainder_of_5_pow_2023_mod_17_l11_1101

theorem remainder_of_5_pow_2023_mod_17 :
  5^2023 % 17 = 11 :=
by
  have h1 : 5^2 % 17 = 8 := by sorry
  have h2 : 5^4 % 17 = 13 := by sorry
  have h3 : 5^8 % 17 = -1 := by sorry
  have h4 : 5^16 % 17 = 1 := by sorry
  have h5 : 2023 = 16 * 126 + 7 := by sorry
  sorry

end NUMINAMATH_GPT_remainder_of_5_pow_2023_mod_17_l11_1101


namespace NUMINAMATH_GPT_taxi_fare_total_distance_l11_1144

theorem taxi_fare_total_distance (initial_fare additional_fare : ℝ) (total_fare : ℝ) (initial_distance additional_distance : ℝ) :
  initial_fare = 10 ∧ additional_fare = 1 ∧ initial_distance = 1/5 ∧ (total_fare = 59) →
  (total_distance = initial_distance + additional_distance * ((total_fare - initial_fare) / additional_fare)) →
  total_distance = 10 := 
by 
  sorry

end NUMINAMATH_GPT_taxi_fare_total_distance_l11_1144


namespace NUMINAMATH_GPT_average_marks_correct_l11_1133

def marks := [76, 65, 82, 62, 85]
def num_subjects := 5
def total_marks := marks.sum
def avg_marks := total_marks / num_subjects

theorem average_marks_correct : avg_marks = 74 :=
by sorry

end NUMINAMATH_GPT_average_marks_correct_l11_1133


namespace NUMINAMATH_GPT_votes_cast_l11_1197

theorem votes_cast (V : ℝ) (h1 : V = 0.33 * V + (0.33 * V + 833)) : V = 2447 := 
by
  sorry

end NUMINAMATH_GPT_votes_cast_l11_1197


namespace NUMINAMATH_GPT_rainfall_on_tuesday_l11_1192

theorem rainfall_on_tuesday 
  (r_Mon r_Wed r_Total r_Tue : ℝ)
  (h_Mon : r_Mon = 0.16666666666666666)
  (h_Wed : r_Wed = 0.08333333333333333)
  (h_Total : r_Total = 0.6666666666666666)
  (h_Tue : r_Tue = r_Total - (r_Mon + r_Wed)) :
  r_Tue = 0.41666666666666663 := 
sorry

end NUMINAMATH_GPT_rainfall_on_tuesday_l11_1192


namespace NUMINAMATH_GPT_remainder_when_divided_by_2_l11_1148

theorem remainder_when_divided_by_2 (n : ℕ) (h₁ : n > 0) (h₂ : (n + 1) % 6 = 4) : n % 2 = 1 :=
by sorry

end NUMINAMATH_GPT_remainder_when_divided_by_2_l11_1148


namespace NUMINAMATH_GPT_inequality_solution_l11_1145

noncomputable def f (x : ℝ) : ℝ :=
  (2 / (x + 2)) + (4 / (x + 8))

theorem inequality_solution {x : ℝ} :
  f x ≥ 1/2 ↔ ((-8 < x ∧ x ≤ -4) ∨ (-2 ≤ x ∧ x ≤ 2)) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l11_1145


namespace NUMINAMATH_GPT_simplify_expression_l11_1180

theorem simplify_expression :
  (0.7264 * 0.4329 * 0.5478) + (0.1235 * 0.3412 * 0.6214) - ((0.1289 * 0.5634 * 0.3921) / (0.3785 * 0.4979 * 0.2884)) - (0.2956 * 0.3412 * 0.6573) = -0.3902 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l11_1180


namespace NUMINAMATH_GPT_shark_ratio_l11_1199

theorem shark_ratio (N D : ℕ) (h1 : N = 22) (h2 : D + N = 110) (h3 : ∃ x : ℕ, D = x * N) : 
  (D / N) = 4 :=
by
  -- conditions use only definitions given in the problem.
  sorry

end NUMINAMATH_GPT_shark_ratio_l11_1199


namespace NUMINAMATH_GPT_student_weight_l11_1146

-- Definitions based on conditions
variables (S R : ℝ)

-- Conditions as assertions
def condition1 : Prop := S - 5 = 2 * R
def condition2 : Prop := S + R = 104

-- The statement we want to prove
theorem student_weight (h1 : condition1 S R) (h2 : condition2 S R) : S = 71 :=
by
  sorry

end NUMINAMATH_GPT_student_weight_l11_1146


namespace NUMINAMATH_GPT_frog_eyes_in_pond_l11_1147

-- Definitions based on conditions
def num_frogs : ℕ := 6
def eyes_per_frog : ℕ := 2

-- The property to be proved
theorem frog_eyes_in_pond : num_frogs * eyes_per_frog = 12 :=
by
  sorry

end NUMINAMATH_GPT_frog_eyes_in_pond_l11_1147


namespace NUMINAMATH_GPT_convert_neg_900_deg_to_rad_l11_1121

theorem convert_neg_900_deg_to_rad : (-900 : ℝ) * (Real.pi / 180) = -5 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_convert_neg_900_deg_to_rad_l11_1121


namespace NUMINAMATH_GPT_lines_perpendicular_to_same_plane_are_parallel_l11_1166

variables {Point Line Plane : Type}
variables (a b c : Line) (α β γ : Plane)
variables (perp_line_to_plane : Line → Plane → Prop) (parallel_lines : Line → Line → Prop)
variables (subset_line_in_plane : Line → Plane → Prop)

-- The conditions
axiom a_perp_alpha : perp_line_to_plane a α
axiom b_perp_alpha : perp_line_to_plane b α

-- The statement to prove
theorem lines_perpendicular_to_same_plane_are_parallel :
  parallel_lines a b :=
by sorry

end NUMINAMATH_GPT_lines_perpendicular_to_same_plane_are_parallel_l11_1166


namespace NUMINAMATH_GPT_total_students_course_l11_1131

theorem total_students_course 
  (T : ℕ)
  (H1 : (1 / 5 : ℚ) * T = (1 / 5) * T)
  (H2 : (1 / 4 : ℚ) * T = (1 / 4) * T)
  (H3 : (1 / 2 : ℚ) * T = (1 / 2) * T)
  (H4 : T = (1 / 5 : ℚ) * T + (1 / 4 : ℚ) * T + (1 / 2 : ℚ) * T + 30) : 
  T = 600 :=
sorry

end NUMINAMATH_GPT_total_students_course_l11_1131


namespace NUMINAMATH_GPT_factor_polynomial_l11_1174

noncomputable def gcd_coeffs : ℕ := Nat.gcd 72 180

theorem factor_polynomial (x : ℝ) (GCD_72_180 : gcd_coeffs = 36)
    (GCD_x5_x9 : ∃ (y: ℝ), x^5 = y ∧ x^9 = y * x^4) :
    72 * x^5 - 180 * x^9 = -36 * x^5 * (5 * x^4 - 2) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l11_1174


namespace NUMINAMATH_GPT_positive_integers_satisfying_condition_l11_1191

theorem positive_integers_satisfying_condition :
  ∃! n : ℕ, 0 < n ∧ 24 - 6 * n > 12 :=
by
  sorry

end NUMINAMATH_GPT_positive_integers_satisfying_condition_l11_1191


namespace NUMINAMATH_GPT_sin_45_eq_sqrt2_div_2_l11_1189

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_45_eq_sqrt2_div_2_l11_1189


namespace NUMINAMATH_GPT_value_of_2_star_3_l11_1187

def star (a b : ℕ) : ℕ := a * b ^ 3 - b + 2

theorem value_of_2_star_3 : star 2 3 = 53 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_value_of_2_star_3_l11_1187


namespace NUMINAMATH_GPT_tetrahedron_colorings_l11_1138

-- Define the problem conditions
def tetrahedron_faces : ℕ := 4
def colors : List String := ["red", "white", "blue", "yellow"]

-- The theorem statement
theorem tetrahedron_colorings :
  ∃ n : ℕ, n = 35 ∧ ∀ (c : List String), c.length = tetrahedron_faces → c ⊆ colors →
  (true) := -- Placeholder (you can replace this condition with the appropriate condition)
by
  -- The proof is omitted with 'sorry' as instructed
  sorry

end NUMINAMATH_GPT_tetrahedron_colorings_l11_1138


namespace NUMINAMATH_GPT_sockPairsCount_l11_1105

noncomputable def countSockPairs : ℕ :=
  let whitePairs := Nat.choose 6 2 -- 15
  let brownPairs := Nat.choose 7 2 -- 21
  let bluePairs := Nat.choose 3 2 -- 3
  let oneRedOneWhite := 4 * 6 -- 24
  let oneRedOneBrown := 4 * 7 -- 28
  let oneRedOneBlue := 4 * 3 -- 12
  let bothRed := Nat.choose 4 2 -- 6
  whitePairs + brownPairs + bluePairs + oneRedOneWhite + oneRedOneBrown + oneRedOneBlue + bothRed

theorem sockPairsCount : countSockPairs = 109 := by
  sorry

end NUMINAMATH_GPT_sockPairsCount_l11_1105


namespace NUMINAMATH_GPT_problem1_problem2_l11_1179

theorem problem1 : (Real.tan (10 * Real.pi / 180) - Real.sqrt 3) * Real.sin (40 * Real.pi / 180) = -1 := 
by
  sorry

theorem problem2 (x : ℝ) : 
  (2 * Real.cos x ^ 4 - 2 * Real.cos x ^ 2 + 1 / 2) /
  (2 * Real.tan (Real.pi / 4 - x) * Real.sin (Real.pi / 4 + x) ^ 2) = 
  Real.sin (2 * x) / 4 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l11_1179


namespace NUMINAMATH_GPT_find_range_of_x_l11_1167

variable (f : ℝ → ℝ) (x : ℝ)

-- Assume f is an increasing function on [-1, 1]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x ≤ y → f x ≤ f y

-- Main theorem statement based on the problem
theorem find_range_of_x (h_increasing : is_increasing_on_interval f (-1) 1)
                        (h_condition : f (x - 1) < f (1 - 3 * x)) :
  0 ≤ x ∧ x < (1 / 2) :=
sorry

end NUMINAMATH_GPT_find_range_of_x_l11_1167


namespace NUMINAMATH_GPT_simplify_expression_l11_1111

theorem simplify_expression : (Real.sin (15 * Real.pi / 180) + Real.sin (45 * Real.pi / 180)) / (Real.cos (15 * Real.pi / 180) + Real.cos (45 * Real.pi / 180)) = Real.tan (30 * Real.pi / 180) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l11_1111


namespace NUMINAMATH_GPT_large_circle_radius_l11_1177

noncomputable def radius_of_large_circle (R : ℝ) : Prop :=
  ∃ r : ℝ, (r = 2) ∧
           (R = r + r) ∧
           (r = 2) ∧
           (R - r = 2) ∧
           (R = 4)

theorem large_circle_radius :
  radius_of_large_circle 4 :=
by
  sorry

end NUMINAMATH_GPT_large_circle_radius_l11_1177


namespace NUMINAMATH_GPT_find_f1_l11_1150

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition_on_function (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, x ≤ 0 → f x = 2^x - 3 * x + 2 * m

theorem find_f1 (f : ℝ → ℝ) (m : ℝ)
  (h_odd : is_odd_function f)
  (h_condition : condition_on_function f m) :
  f 1 = -(5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_f1_l11_1150


namespace NUMINAMATH_GPT_polynomial_expansion_sum_l11_1136

theorem polynomial_expansion_sum :
  let A := 4
  let B := 10
  let C := 1
  let D := 21
  (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D →
  A + B + C + D = 36 := 
by
  -- Proof needs to be filled
  sorry

end NUMINAMATH_GPT_polynomial_expansion_sum_l11_1136


namespace NUMINAMATH_GPT_third_side_not_one_l11_1135

theorem third_side_not_one (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c ≠ 1) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end NUMINAMATH_GPT_third_side_not_one_l11_1135


namespace NUMINAMATH_GPT_volume_of_truncated_triangular_pyramid_l11_1149

variable {a b H α : ℝ} (h1 : H = Real.sqrt (a * b))

theorem volume_of_truncated_triangular_pyramid
  (h2 : H = Real.sqrt (a * b))
  (h3 : 0 < a)
  (h4 : 0 < b)
  (h5 : 0 < H)
  (h6 : 0 < α) :
  (volume : ℝ) = H^3 * Real.sqrt 3 / (4 * (Real.sin α)^2) := sorry

end NUMINAMATH_GPT_volume_of_truncated_triangular_pyramid_l11_1149


namespace NUMINAMATH_GPT_pure_ghee_added_l11_1123

theorem pure_ghee_added
  (Q : ℕ) (hQ : Q = 30)
  (P : ℕ)
  (original_pure_ghee : ℕ := (Q / 2))
  (original_vanaspati : ℕ := (Q / 2))
  (new_total_ghee : ℕ := Q + P)
  (new_vanaspati_fraction : ℝ := 0.3) :
  original_vanaspati = (new_vanaspati_fraction * ↑new_total_ghee : ℝ) → P = 20 := by
  sorry

end NUMINAMATH_GPT_pure_ghee_added_l11_1123


namespace NUMINAMATH_GPT_number_of_hard_drives_sold_l11_1103

theorem number_of_hard_drives_sold 
    (H : ℕ)
    (price_per_graphics_card : ℕ := 600)
    (price_per_hard_drive : ℕ := 80)
    (price_per_cpu : ℕ := 200)
    (price_per_ram_pair : ℕ := 60)
    (graphics_cards_sold : ℕ := 10)
    (cpus_sold : ℕ := 8)
    (ram_pairs_sold : ℕ := 4)
    (total_earnings : ℕ := 8960)
    (earnings_from_graphics_cards : graphics_cards_sold * price_per_graphics_card = 6000)
    (earnings_from_cpus : cpus_sold * price_per_cpu = 1600)
    (earnings_from_ram : ram_pairs_sold * price_per_ram_pair = 240)
    (earnings_from_hard_drives : H * price_per_hard_drive = 80 * H) :
  graphics_cards_sold * price_per_graphics_card +
  cpus_sold * price_per_cpu +
  ram_pairs_sold * price_per_ram_pair +
  H * price_per_hard_drive = total_earnings → H = 14 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_number_of_hard_drives_sold_l11_1103


namespace NUMINAMATH_GPT_polynomial_system_solution_l11_1165

variable {x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ}

theorem polynomial_system_solution (
  h1 : x₁ + 3 * x₂ + 5 * x₃ + 7 * x₄ + 9 * x₅ + 11 * x₆ + 13 * x₇ = 3)
  (h2 : 3 * x₁ + 5 * x₂ + 7 * x₃ + 9 * x₄ + 11 * x₅ + 13 * x₆ + 15 * x₇ = 15)
  (h3 : 5 * x₁ + 7 * x₂ + 9 * x₃ + 11 * x₄ + 13 * x₅ + 15 * x₆ + 17 * x₇ = 85) :
  7 * x₁ + 9 * x₂ + 11 * x₃ + 13 * x₄ + 15 * x₅ + 17 * x₆ + 19 * x₇ = 213 :=
sorry

end NUMINAMATH_GPT_polynomial_system_solution_l11_1165


namespace NUMINAMATH_GPT_sum_of_f_is_negative_l11_1143

noncomputable def f (x : ℝ) : ℝ := x + x^3 + x^5

theorem sum_of_f_is_negative (x₁ x₂ x₃ : ℝ)
  (h1: x₁ + x₂ < 0)
  (h2: x₂ + x₃ < 0) 
  (h3: x₃ + x₁ < 0) :
  f x₁ + f x₂ + f x₃ < 0 := 
sorry

end NUMINAMATH_GPT_sum_of_f_is_negative_l11_1143


namespace NUMINAMATH_GPT_divisibility_by_seven_l11_1162

theorem divisibility_by_seven : (∃ k : ℤ, (-8)^2019 + (-8)^2018 = 7 * k) :=
sorry

end NUMINAMATH_GPT_divisibility_by_seven_l11_1162


namespace NUMINAMATH_GPT_hyperbola_ellipse_equations_l11_1119

theorem hyperbola_ellipse_equations 
  (F1 F2 P : ℝ × ℝ) 
  (hF1 : F1 = (0, -5))
  (hF2 : F2 = (0, 5))
  (hP : P = (3, 4)) :
  (∃ a b : ℝ, a^2 = 40 ∧ b^2 = 16 ∧ 
    ∀ x y : ℝ, (y^2 / 40 + x^2 / 15 = 1 ↔ y^2 / a^2 + x^2 / (a^2 - 25) = 1) ∧
    (y^2 / 16 - x^2 / 9 = 1 ↔ y^2 / b^2 - x^2 / (25 - b^2) = 1)) :=
sorry

end NUMINAMATH_GPT_hyperbola_ellipse_equations_l11_1119


namespace NUMINAMATH_GPT_correct_proportion_expression_l11_1137

def is_fraction_correctly_expressed (numerator denominator : ℕ) (expression : String) : Prop :=
  -- Define the property of a correctly expressed fraction in English
  expression = "three-fifths"

theorem correct_proportion_expression : 
  is_fraction_correctly_expressed 3 5 "three-fifths" :=
by
  sorry

end NUMINAMATH_GPT_correct_proportion_expression_l11_1137


namespace NUMINAMATH_GPT_find_x_plus_inv_x_l11_1175

theorem find_x_plus_inv_x (x : ℝ) (hx_pos : 0 < x) (h : x^10 + x^5 + 1/x^5 + 1/x^10 = 15250) :
  x + 1/x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_plus_inv_x_l11_1175


namespace NUMINAMATH_GPT_friends_recycled_pounds_l11_1178

-- Definitions of given conditions
def points_earned : ℕ := 6
def pounds_per_point : ℕ := 8
def zoe_pounds : ℕ := 25

-- Calculation based on given conditions
def total_pounds := points_earned * pounds_per_point
def friends_pounds := total_pounds - zoe_pounds

-- Statement of the proof problem
theorem friends_recycled_pounds : friends_pounds = 23 := by
  sorry

end NUMINAMATH_GPT_friends_recycled_pounds_l11_1178


namespace NUMINAMATH_GPT_cans_in_each_package_of_cat_food_l11_1172

-- Definitions and conditions
def cans_per_package_cat (c : ℕ) := 9 * c
def cans_per_package_dog := 7 * 5
def extra_cans_cat := 55

-- Theorem stating the problem and the answer
theorem cans_in_each_package_of_cat_food (c : ℕ) (h: cans_per_package_cat c = cans_per_package_dog + extra_cans_cat) :
  c = 10 :=
sorry

end NUMINAMATH_GPT_cans_in_each_package_of_cat_food_l11_1172


namespace NUMINAMATH_GPT_terrell_total_distance_l11_1112

theorem terrell_total_distance (saturday_distance sunday_distance : ℝ) (h_saturday : saturday_distance = 8.2) (h_sunday : sunday_distance = 1.6) :
  saturday_distance + sunday_distance = 9.8 :=
by
  rw [h_saturday, h_sunday]
  -- sorry
  norm_num

end NUMINAMATH_GPT_terrell_total_distance_l11_1112


namespace NUMINAMATH_GPT_final_score_l11_1110

theorem final_score (questions_first_half questions_second_half points_per_question : ℕ) (h1 : questions_first_half = 5) (h2 : questions_second_half = 5) (h3 : points_per_question = 5) : 
  (questions_first_half + questions_second_half) * points_per_question = 50 :=
by
  sorry

end NUMINAMATH_GPT_final_score_l11_1110


namespace NUMINAMATH_GPT_area_of_black_region_l11_1161

-- Definitions for the side lengths of the smaller and larger squares
def s₁ : ℕ := 4
def s₂ : ℕ := 8

-- The mathematical problem statement in Lean 4
theorem area_of_black_region : (s₂ * s₂) - (s₁ * s₁) = 48 := by
  sorry

end NUMINAMATH_GPT_area_of_black_region_l11_1161


namespace NUMINAMATH_GPT_relationship_ab_c_l11_1155
open Real

noncomputable def a : ℝ := (1 / 3) ^ (log 3 / log (1 / 3))
noncomputable def b : ℝ := (1 / 3) ^ (log 4 / log (1 / 3))
noncomputable def c : ℝ := 3 ^ log 3

theorem relationship_ab_c : c > b ∧ b > a := by
  sorry

end NUMINAMATH_GPT_relationship_ab_c_l11_1155


namespace NUMINAMATH_GPT_find_angle_A_l11_1126

theorem find_angle_A (A B C : ℝ)
  (h1 : C = 2 * B)
  (h2 : B = A / 3)
  (h3 : A + B + C = 180) : A = 90 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_A_l11_1126


namespace NUMINAMATH_GPT_pipe_filling_time_l11_1195

/-- 
A problem involving two pipes filling and emptying a tank. 
Time taken for the first pipe to fill the tank is proven to be 16.8 minutes.
-/
theorem pipe_filling_time :
  ∃ T : ℝ, (∀ T, let r1 := 1 / T
                let r2 := 1 / 24
                let time_both_pipes_open := 36
                let time_first_pipe_only := 6
                (r1 - r2) * time_both_pipes_open + r1 * time_first_pipe_only = 1) ∧
           T = 16.8 :=
by
  sorry

end NUMINAMATH_GPT_pipe_filling_time_l11_1195


namespace NUMINAMATH_GPT_find_k_l11_1122

noncomputable def y (k x : ℝ) : ℝ := k / x

theorem find_k (k : ℝ) (h₁ : k ≠ 0) (h₂ : 1 ≤ 3) 
  (h₃ : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → x = 1 ∨ x = 3) 
  (h₄ : |y k 1 - y k 3| = 4) : k = 6 ∨ k = -6 :=
  sorry

end NUMINAMATH_GPT_find_k_l11_1122


namespace NUMINAMATH_GPT_minibus_children_count_l11_1171

theorem minibus_children_count
  (total_seats : ℕ)
  (seats_with_3_children : ℕ)
  (seats_with_2_children : ℕ)
  (children_per_seat_3 : ℕ)
  (children_per_seat_2 : ℕ)
  (h_seats_count : total_seats = 7)
  (h_seats_distribution : seats_with_3_children = 5 ∧ seats_with_2_children = 2)
  (h_children_per_seat : children_per_seat_3 = 3 ∧ children_per_seat_2 = 2) :
  seats_with_3_children * children_per_seat_3 + seats_with_2_children * children_per_seat_2 = 19 :=
by
  sorry

end NUMINAMATH_GPT_minibus_children_count_l11_1171


namespace NUMINAMATH_GPT_numberOfBaseballBoxes_l11_1194

-- Given conditions as Lean definitions and assumptions
def numberOfBasketballBoxes : ℕ := 4
def basketballCardsPerBox : ℕ := 10
def baseballCardsPerBox : ℕ := 8
def cardsGivenToClassmates : ℕ := 58
def cardsLeftAfterGiving : ℕ := 22

def totalBasketballCards : ℕ := numberOfBasketballBoxes * basketballCardsPerBox
def totalCardsBeforeGiving : ℕ := cardsLeftAfterGiving + cardsGivenToClassmates

-- Target number of baseball cards
def totalBaseballCards : ℕ := totalCardsBeforeGiving - totalBasketballCards

-- Prove that the number of baseball boxes is 5
theorem numberOfBaseballBoxes :
  totalBaseballCards / baseballCardsPerBox = 5 :=
sorry

end NUMINAMATH_GPT_numberOfBaseballBoxes_l11_1194


namespace NUMINAMATH_GPT_fourth_angle_of_quadrilateral_l11_1190

theorem fourth_angle_of_quadrilateral (A : ℝ) : 
  (120 + 85 + 90 + A = 360) ↔ A = 65 := 
by
  sorry

end NUMINAMATH_GPT_fourth_angle_of_quadrilateral_l11_1190


namespace NUMINAMATH_GPT_regular_pay_calculation_l11_1169

theorem regular_pay_calculation
  (R : ℝ)  -- defining the regular pay per hour
  (H1 : 40 * R + 20 * R = 180):  -- condition given based on the total actual pay calculation.
  R = 3 := 
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_regular_pay_calculation_l11_1169


namespace NUMINAMATH_GPT_min_value_expression_l11_1129

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ m : ℝ, (m = 4 + 6 * Real.sqrt 2) ∧ 
  ∀ a b : ℝ, (0 < a) → (0 < b) → m ≤ (Real.sqrt ((a^2 + b^2) * (2*a^2 + 4*b^2))) / (a * b) :=
by sorry

end NUMINAMATH_GPT_min_value_expression_l11_1129


namespace NUMINAMATH_GPT_rachel_homework_total_l11_1188

-- Definitions based on conditions
def math_homework : Nat := 8
def biology_homework : Nat := 3

-- Theorem based on the problem statement
theorem rachel_homework_total : math_homework + biology_homework = 11 := by
  -- typically, here you would provide a proof, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_rachel_homework_total_l11_1188


namespace NUMINAMATH_GPT_y_is_multiple_of_16_y_is_multiple_of_8_y_is_multiple_of_4_y_is_multiple_of_2_l11_1156

def y : ℕ := 112 + 160 + 272 + 432 + 1040 + 1264 + 4256

theorem y_is_multiple_of_16 : y % 16 = 0 :=
sorry

theorem y_is_multiple_of_8 : y % 8 = 0 :=
sorry

theorem y_is_multiple_of_4 : y % 4 = 0 :=
sorry

theorem y_is_multiple_of_2 : y % 2 = 0 :=
sorry

end NUMINAMATH_GPT_y_is_multiple_of_16_y_is_multiple_of_8_y_is_multiple_of_4_y_is_multiple_of_2_l11_1156


namespace NUMINAMATH_GPT_find_width_of_room_l11_1160

section RoomWidth

variable (l C P A W : ℝ)
variable (h1 : l = 5.5)
variable (h2 : C = 16500)
variable (h3 : P = 750)
variable (h4 : A = C / P)
variable (h5 : A = l * W)

theorem find_width_of_room : W = 4 := by
  sorry

end RoomWidth

end NUMINAMATH_GPT_find_width_of_room_l11_1160


namespace NUMINAMATH_GPT_sufficient_condition_for_m_l11_1183

variable (x m : ℝ)

def p (x : ℝ) : Prop := abs (x - 4) ≤ 6
def q (x m : ℝ) : Prop := x ≤ 1 + m

theorem sufficient_condition_for_m (h : ∀ x, p x → q x m ∧ ∃ x, ¬p x ∧ q x m) : m ≥ 9 :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_m_l11_1183


namespace NUMINAMATH_GPT_rectangle_width_decrease_l11_1151

theorem rectangle_width_decrease (A L W : ℝ) (h1 : A = L * W) (h2 : 1.5 * L * W' = A) : 
  (W' = (2/3) * W) -> by exact (W - W') / W = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_decrease_l11_1151


namespace NUMINAMATH_GPT_factorize_expression_l11_1114

theorem factorize_expression (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l11_1114


namespace NUMINAMATH_GPT_part_one_max_value_range_of_a_l11_1125

def f (x a : ℝ) : ℝ := |x + 2| - |x - 3| - a

theorem part_one_max_value (a : ℝ) (h : a = 1) : ∃ x : ℝ, f x a = 4 := 
by sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≤ 4 / a) :  (0 < a ∧ a ≤ 1) ∨ 4 ≤ a :=
by sorry

end NUMINAMATH_GPT_part_one_max_value_range_of_a_l11_1125


namespace NUMINAMATH_GPT_complex_div_eq_l11_1140

def complex_z : ℂ := ⟨1, -2⟩
def imaginary_unit : ℂ := ⟨0, 1⟩

theorem complex_div_eq :
  (complex_z + 2) / (complex_z - 1) = 1 + (3 / 2 : ℂ) * imaginary_unit :=
by
  sorry

end NUMINAMATH_GPT_complex_div_eq_l11_1140


namespace NUMINAMATH_GPT_equivalent_mod_l11_1127

theorem equivalent_mod (h : 5^300 ≡ 1 [MOD 1250]) : 5^9000 ≡ 1 [MOD 1000] :=
by 
  sorry

end NUMINAMATH_GPT_equivalent_mod_l11_1127


namespace NUMINAMATH_GPT_hyperbola_condition_l11_1152

variables (a b : ℝ)
def e1 : (ℝ × ℝ) := (2, 1)
def e2 : (ℝ × ℝ) := (2, -1)

theorem hyperbola_condition (h1 : e1 = (2, 1)) (h2 : e2 = (2, -1)) (p : ℝ × ℝ)
  (h3 : p = (2 * a + 2 * b, a - b)) :
  4 * a * b = 1 :=
sorry

end NUMINAMATH_GPT_hyperbola_condition_l11_1152


namespace NUMINAMATH_GPT_walking_speed_l11_1139

noncomputable def bridge_length : ℝ := 2500  -- length of the bridge in meters
noncomputable def crossing_time_minutes : ℝ := 15  -- time to cross the bridge in minutes
noncomputable def conversion_factor_time : ℝ := 1 / 60  -- factor to convert minutes to hours
noncomputable def conversion_factor_distance : ℝ := 1 / 1000  -- factor to convert meters to kilometers

theorem walking_speed (bridge_length crossing_time_minutes conversion_factor_time conversion_factor_distance : ℝ) : 
  bridge_length = 2500 → 
  crossing_time_minutes = 15 → 
  conversion_factor_time = 1 / 60 → 
  conversion_factor_distance = 1 / 1000 → 
  (bridge_length * conversion_factor_distance) / (crossing_time_minutes * conversion_factor_time) = 10 := 
by
  sorry

end NUMINAMATH_GPT_walking_speed_l11_1139


namespace NUMINAMATH_GPT_max_value_of_f_l11_1141

noncomputable def f (x a : ℝ) : ℝ := - (1/3) * x ^ 3 + (1/2) * x ^ 2 + 2 * a * x

theorem max_value_of_f (a : ℝ) (h0 : 0 < a) (h1 : a < 2)
  (h2 : ∀ x, 1 ≤ x → x ≤ 4 → f x a ≥ f 4 a)
  (h3 : f 4 a = -16 / 3) :
  f 2 a = 10 / 3 :=
sorry

end NUMINAMATH_GPT_max_value_of_f_l11_1141


namespace NUMINAMATH_GPT_probability_angle_AMB_acute_l11_1116

theorem probability_angle_AMB_acute :
  let side_length := 4
  let square_area := side_length * side_length
  let semicircle_area := (1 / 2) * Real.pi * (side_length / 2) ^ 2
  let probability := 1 - semicircle_area / square_area
  probability = 1 - (Real.pi / 8) :=
sorry

end NUMINAMATH_GPT_probability_angle_AMB_acute_l11_1116


namespace NUMINAMATH_GPT_smallest_perfect_square_divisible_by_4_and_5_l11_1182

theorem smallest_perfect_square_divisible_by_4_and_5 : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ (n = 400) := 
by
  sorry

end NUMINAMATH_GPT_smallest_perfect_square_divisible_by_4_and_5_l11_1182


namespace NUMINAMATH_GPT_exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power_l11_1159

open Nat

theorem exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power :
  ∃ (a : ℕ → ℕ), (∀ k : ℕ, (∃ b : ℕ, a k = b ^ 2)) ∧ (StrictMono a) ∧ (∀ k : ℕ, 13^k ∣ (a k + 1)) :=
sorry

end NUMINAMATH_GPT_exists_increasing_infinite_sequence_of_perfect_squares_divisible_by_13_power_l11_1159


namespace NUMINAMATH_GPT_only_polyC_is_square_of_binomial_l11_1173

-- Defining the polynomials
def polyA (m n : ℤ) : ℤ := (-m + n) * (m - n)
def polyB (a b : ℤ) : ℤ := (1/2 * a + b) * (b - 1/2 * a)
def polyC (x : ℤ) : ℤ := (x + 5) * (x + 5)
def polyD (a b : ℤ) : ℤ := (3 * a - 4 * b) * (3 * b + 4 * a)

-- Proving that only polyC fits the square of a binomial formula
theorem only_polyC_is_square_of_binomial (x : ℤ) :
  (polyC x) = (x + 5) * (x + 5) ∧
  (∀ m n : ℤ, polyA m n ≠ (m - n)^2) ∧
  (∀ a b : ℤ, polyB a b ≠ (1/2 * a + b)^2) ∧
  (∀ a b : ℤ, polyD a b ≠ (3 * a - 4 * b)^2) :=
by
  sorry

end NUMINAMATH_GPT_only_polyC_is_square_of_binomial_l11_1173


namespace NUMINAMATH_GPT_compute_special_op_l11_1128

-- Define the operation ※
def special_op (m n : ℚ) := (3 * m + n) * (3 * m - n) + n

-- Hypothesis for specific m and n
def m := (1 : ℚ) / 6
def n := (-1 : ℚ)

-- Proof goal
theorem compute_special_op : special_op m n = -7 / 4 := by
  sorry

end NUMINAMATH_GPT_compute_special_op_l11_1128


namespace NUMINAMATH_GPT_mushrooms_weight_change_l11_1184

-- Conditions
variables (x W : ℝ)
variable (initial_weight : ℝ := 100 * x)
variable (dry_weight : ℝ := x)
variable (final_weight_dry : ℝ := 2 * W / 100)

-- Given fresh mushrooms have moisture content of 99%
-- and dried mushrooms have moisture content of 98%
theorem mushrooms_weight_change 
  (h1 : dry_weight = x) 
  (h2 : final_weight_dry = x / 0.02) 
  (h3 : W = x / 0.02) 
  (initial_weight : ℝ := 100 * x) : 
  2 * W = initial_weight / 2 :=
by
  -- This is a placeholder for the proof steps which we skip
  sorry

end NUMINAMATH_GPT_mushrooms_weight_change_l11_1184


namespace NUMINAMATH_GPT_inequality_subtraction_l11_1104

theorem inequality_subtraction (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end NUMINAMATH_GPT_inequality_subtraction_l11_1104


namespace NUMINAMATH_GPT_lengths_of_legs_l11_1153

def is_right_triangle (a b c : ℕ) := a^2 + b^2 = c^2

theorem lengths_of_legs (a b : ℕ) 
  (h1 : is_right_triangle a b 60)
  (h2 : a + b = 84) 
  : (a = 48 ∧ b = 36) ∨ (a = 36 ∧ b = 48) :=
  sorry

end NUMINAMATH_GPT_lengths_of_legs_l11_1153


namespace NUMINAMATH_GPT_yan_ratio_distance_l11_1124

theorem yan_ratio_distance (w x y : ℕ) (h : w > 0) (h_eq_time : (y / w) = (x / w) + ((x + y) / (6 * w))) :
  x / y = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_yan_ratio_distance_l11_1124


namespace NUMINAMATH_GPT_total_raised_is_420_l11_1117

def pancake_cost : ℝ := 4.00
def bacon_cost : ℝ := 2.00
def stacks_sold : ℕ := 60
def slices_sold : ℕ := 90

theorem total_raised_is_420 : (pancake_cost * stacks_sold + bacon_cost * slices_sold) = 420.00 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_raised_is_420_l11_1117


namespace NUMINAMATH_GPT_statement_2_statement_3_l11_1120

variable {α : Type*} [LinearOrderedField α]

-- Given a quadratic function
def quadratic (a b c x : α) : α :=
  a * x^2 + b * x + c

-- Statement 2
theorem statement_2 (a b c p q : α) (hpq : p ≠ q) :
  quadratic a b c p = quadratic a b c q → quadratic a b c (p + q) = c :=
sorry

-- Statement 3
theorem statement_3 (a b c p q : α) (hpq : p ≠ q) :
  quadratic a b c (p + q) = c → (p + q = 0 ∨ quadratic a b c p = quadratic a b c q) :=
sorry

end NUMINAMATH_GPT_statement_2_statement_3_l11_1120


namespace NUMINAMATH_GPT_sequence_terminates_final_value_l11_1100

-- Define the function Lisa uses to update the number
def f (x : ℕ) : ℕ :=
  let a := x / 10
  let b := x % 10
  a + 4 * b

-- Prove that for any initial value x0, the sequence eventually becomes periodic and ends.
theorem sequence_terminates (x0 : ℕ) : ∃ N : ℕ, ∃ j : ℕ, N ≠ j ∧ (Nat.iterate f N x0) = (Nat.iterate f j x0) :=
  by sorry

-- Given the starting value, show the sequence stabilizes at 39
theorem final_value (x0 : ℕ) (h : x0 = 53^2022 - 1) : ∃ N : ℕ, Nat.iterate f N x0 = 39 :=
  by sorry

end NUMINAMATH_GPT_sequence_terminates_final_value_l11_1100


namespace NUMINAMATH_GPT_working_together_time_l11_1130

/-- A is 30% more efficient than B,
and A alone can complete the job in 23 days.
Prove that A and B working together take approximately 13 days to complete the job. -/
theorem working_together_time (Ea Eb : ℝ) (T : ℝ) (h1 : Ea = 1.30 * Eb) 
  (h2 : 1 / 23 = Ea) : T = 13 :=
sorry

end NUMINAMATH_GPT_working_together_time_l11_1130


namespace NUMINAMATH_GPT_max_area_with_22_matches_l11_1193

-- Definitions based on the conditions
def perimeter := 22

def is_valid_length_width (l w : ℕ) : Prop := l + w = 11

def area (l w : ℕ) : ℕ := l * w

-- Statement of the proof problem
theorem max_area_with_22_matches : 
  ∃ (l w : ℕ), is_valid_length_width l w ∧ (∀ l' w', is_valid_length_width l' w' → area l w ≥ area l' w') ∧ area l w = 30 :=
  sorry

end NUMINAMATH_GPT_max_area_with_22_matches_l11_1193


namespace NUMINAMATH_GPT_polynomial_no_linear_term_l11_1118

theorem polynomial_no_linear_term (m n : ℝ) :
  (∀ x : ℝ, (x - m) * (x - n) = x^2 + mn → n + m = 0) :=
sorry

end NUMINAMATH_GPT_polynomial_no_linear_term_l11_1118


namespace NUMINAMATH_GPT_max_min_sum_l11_1181

noncomputable def f : ℝ → ℝ := sorry

-- Define the interval and properties of the function f
def within_interval (x : ℝ) : Prop := -2016 ≤ x ∧ x ≤ 2016
def functional_eq (x1 x2 : ℝ) : Prop := f (x1 + x2) = f x1 + f x2 - 2016
def less_than_2016_proof (x : ℝ) : Prop := x > 0 → f x < 2016

-- Define the minimum and maximum values of the function f
def M : ℝ := sorry
def N : ℝ := sorry

-- Prove that M + N = 4032 given the properties and conditions
theorem max_min_sum : 
  (∀ x1 x2, within_interval x1 → within_interval x2 → functional_eq x1 x2) →
  (∀ x, x > 0 → less_than_2016_proof x) →
  M + N = 4032 :=
by {
  -- Define the formal proof here, placeholder for actual proof
  sorry
}

end NUMINAMATH_GPT_max_min_sum_l11_1181


namespace NUMINAMATH_GPT_smallest_n_l11_1196

theorem smallest_n (n : ℕ) (hn : 0 < n) (h : 253 * n % 15 = 989 * n % 15) : n = 15 := by
  sorry

end NUMINAMATH_GPT_smallest_n_l11_1196


namespace NUMINAMATH_GPT_principal_amount_is_approx_1200_l11_1107

noncomputable def find_principal_amount : Real :=
  let R := 0.10
  let n := 2
  let T := 1
  let SI (P : Real) := P * R * T / 100
  let CI (P : Real) := P * ((1 + R / n) ^ (n * T)) - P
  let diff (P : Real) := CI P - SI P
  let target_diff := 2.999999999999936
  let P := target_diff / (0.1025 - 0.10)
  P

theorem principal_amount_is_approx_1200 : abs (find_principal_amount - 1200) < 0.0001 := 
by
  sorry

end NUMINAMATH_GPT_principal_amount_is_approx_1200_l11_1107


namespace NUMINAMATH_GPT_percent_increase_share_price_l11_1134

theorem percent_increase_share_price (P : ℝ) 
  (h1 : ∃ P₁ : ℝ, P₁ = P + 0.25 * P)
  (h2 : ∃ P₂ : ℝ, P₂ = P + 0.80 * P)
  : ∃ percent_increase : ℝ, percent_increase = 44 := by
  sorry

end NUMINAMATH_GPT_percent_increase_share_price_l11_1134


namespace NUMINAMATH_GPT_problem_l11_1142

noncomputable def f : ℝ → ℝ := sorry

theorem problem (x : ℝ) :
  (f (x + 2) + f x = 0) →
  (∀ x, f (-(x - 1)) = -f (x - 1)) →
  (
    (∀ e, ¬(e > 0 ∧ ∀ x, f (x + e) = f x)) ∧
    (∀ x, f (x + 1) = f (-x + 1)) ∧
    (¬(∀ x, f x = f (-x)))
  ) :=
by
  sorry

end NUMINAMATH_GPT_problem_l11_1142


namespace NUMINAMATH_GPT_box_height_at_least_2_sqrt_15_l11_1154

def box_height (x : ℝ) : ℝ := 2 * x
def surface_area (x : ℝ) : ℝ := 10 * x ^ 2

theorem box_height_at_least_2_sqrt_15 (x : ℝ) (h : ℝ) :
  h = box_height x →
  surface_area x ≥ 150 →
  h ≥ 2 * Real.sqrt 15 :=
by
  intros h_eq sa_ge_150
  sorry

end NUMINAMATH_GPT_box_height_at_least_2_sqrt_15_l11_1154


namespace NUMINAMATH_GPT_centroid_traces_ellipse_l11_1132

noncomputable def fixed_base_triangle (A B : ℝ × ℝ) (d : ℝ) : Prop :=
(A.1 = 0 ∧ A.2 = 0) ∧ (B.1 = d ∧ B.2 = 0)

noncomputable def vertex_moving_on_semicircle (A B C : ℝ × ℝ) : Prop :=
(C.1 - (A.1 + B.1) / 2)^2 + C.2^2 = ((B.1 - A.1) / 2)^2 ∧ C.2 ≥ 0

noncomputable def is_centroid (A B C G : ℝ × ℝ) : Prop :=
G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

theorem centroid_traces_ellipse
  (A B C G : ℝ × ℝ) (d : ℝ) 
  (h1 : fixed_base_triangle A B d) 
  (h2 : vertex_moving_on_semicircle A B C)
  (h3 : is_centroid A B C G) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (G.1^2 / a^2 + G.2^2 / b^2 = 1) := 
sorry

end NUMINAMATH_GPT_centroid_traces_ellipse_l11_1132


namespace NUMINAMATH_GPT_bridge_length_correct_l11_1106

noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def crossing_time_seconds : ℝ := 30

noncomputable def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time_seconds
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem bridge_length_correct : bridge_length = 255 := by
  sorry

end NUMINAMATH_GPT_bridge_length_correct_l11_1106


namespace NUMINAMATH_GPT_solve_inequality_l11_1164

theorem solve_inequality (x : ℝ) : x^3 - 9*x^2 - 16*x > 0 ↔ (x < -1 ∨ x > 16) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l11_1164


namespace NUMINAMATH_GPT_probability_correct_digit_in_two_attempts_l11_1198

theorem probability_correct_digit_in_two_attempts :
  let total_digits := 10
  let probability_first_correct := 1 / total_digits
  let probability_first_incorrect := 9 / total_digits
  let probability_second_correct_if_first_incorrect := 1 / (total_digits - 1)
  (probability_first_correct + probability_first_incorrect * probability_second_correct_if_first_incorrect) = 1 / 5 := 
sorry

end NUMINAMATH_GPT_probability_correct_digit_in_two_attempts_l11_1198


namespace NUMINAMATH_GPT_coefficient_of_x5_in_expansion_l11_1158

-- Define the polynomial expansion of (x-1)(x+1)^8
def polynomial_expansion (x : ℚ) : ℚ :=
  (x - 1) * (x + 1) ^ 8

-- Define the binomial coefficient function
def binom_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem: The coefficient of x^5 in the expansion of (x-1)(x+1)^8 is 14
theorem coefficient_of_x5_in_expansion :
  binom_coeff 8 4 - binom_coeff 8 5 = 14 :=
sorry

end NUMINAMATH_GPT_coefficient_of_x5_in_expansion_l11_1158
