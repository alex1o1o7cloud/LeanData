import Mathlib

namespace cubic_roots_sum_of_reciprocal_squares_l511_51111

theorem cubic_roots_sum_of_reciprocal_squares :
  ∀ a b c : ℝ,
  (∀ x : ℝ, x^3 - 6*x^2 + 11*x - 6 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by
  sorry

end cubic_roots_sum_of_reciprocal_squares_l511_51111


namespace javier_speech_time_l511_51135

theorem javier_speech_time (outline_time writing_time practice_time total_time : ℕ) : 
  outline_time = 30 →
  writing_time = outline_time + 28 →
  practice_time = writing_time / 2 →
  total_time = outline_time + writing_time + practice_time →
  total_time = 117 :=
by sorry

end javier_speech_time_l511_51135


namespace weaving_problem_l511_51134

def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem weaving_problem (a₁ d : ℕ) (h₁ : a₁ > 0) (h₂ : d > 0) :
  (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + 
   arithmetic_sequence a₁ d 3 + arithmetic_sequence a₁ d 4 = 24) →
  (arithmetic_sequence a₁ d 7 = arithmetic_sequence a₁ d 1 * arithmetic_sequence a₁ d 2) →
  arithmetic_sequence a₁ d 10 = 21 := by
  sorry

end weaving_problem_l511_51134


namespace right_pyramid_base_side_l511_51164

/-- Represents a right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- Theorem: If the area of one lateral face of a right pyramid with a square base is 200 square meters
    and the slant height is 40 meters, then the length of the side of its base is 10 meters. -/
theorem right_pyramid_base_side (p : RightPyramid) 
  (h1 : p.lateral_face_area = 200)
  (h2 : p.slant_height = 40) : 
  p.base_side = 10 := by
  sorry

#check right_pyramid_base_side

end right_pyramid_base_side_l511_51164


namespace rachel_lunch_spending_l511_51190

theorem rachel_lunch_spending (initial_amount : ℝ) 
  (h1 : initial_amount = 200)
  (h2 : ∃ dvd_amount : ℝ, dvd_amount = initial_amount / 2)
  (h3 : ∃ amount_left : ℝ, amount_left = 50) :
  ∃ lunch_fraction : ℝ, lunch_fraction = 1 / 4 := by
  sorry

end rachel_lunch_spending_l511_51190


namespace ratio_problem_l511_51106

theorem ratio_problem (A B C D : ℝ) 
  (hA : A = 0.40 * B) 
  (hB : B = 0.25 * C) 
  (hD : D = 0.60 * C) : 
  A / D = 1 / 6 := by
sorry

end ratio_problem_l511_51106


namespace decreasing_power_function_l511_51153

theorem decreasing_power_function (m : ℝ) : 
  (m^2 - 2*m - 2 = 1) ∧ (-4*m - 2 < 0) → m = 3 :=
by sorry

end decreasing_power_function_l511_51153


namespace fraction_condition_l511_51143

theorem fraction_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a / b > 1 → b / a < 1) ∧
  (∃ a b, b / a < 1 ∧ a / b ≤ 1) :=
sorry

end fraction_condition_l511_51143


namespace triangular_numbers_and_squares_l511_51185

theorem triangular_numbers_and_squares (n a b : ℤ) :
  (n = (a^2 + a)/2 + (b^2 + b)/2) →
  (∃ x y : ℤ, 4*n + 1 = x^2 + y^2 ∧ x = a + b + 1 ∧ y = a - b) ∧
  (∀ x y : ℤ, 4*n + 1 = x^2 + y^2 →
    ∃ a' b' : ℤ, n = (a'^2 + a')/2 + (b'^2 + b')/2) :=
by sorry

end triangular_numbers_and_squares_l511_51185


namespace hyperbola_theorem_l511_51160

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the condition for points on the intersection line
def on_intersection_line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x - 2

-- Define the relation between points O, M, N, and D
def point_relation (xm ym xn yn xd yd t : ℝ) : Prop :=
  xm + xn = t * xd ∧ ym + yn = t * yd

theorem hyperbola_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_real_axis : 2 * a = 4 * Real.sqrt 3)
  (h_focus_asymptote : b * Real.sqrt (b^2 + a^2) / Real.sqrt (b^2 + a^2) = Real.sqrt 3) :
  (∃ (xm ym xn yn xd yd t : ℝ),
    hyperbola a b xm ym ∧
    hyperbola a b xn yn ∧
    hyperbola a b xd yd ∧
    on_intersection_line xm ym ∧
    on_intersection_line xn yn ∧
    point_relation xm ym xn yn xd yd t ∧
    a^2 = 12 ∧
    b^2 = 3 ∧
    t = 4 ∧
    xd = 4 * Real.sqrt 3 ∧
    yd = 3) :=
sorry

end hyperbola_theorem_l511_51160


namespace sum_to_target_l511_51122

theorem sum_to_target : ∃ x : ℝ, 0.003 + 0.158 + x = 2.911 ∧ x = 2.750 := by
  sorry

end sum_to_target_l511_51122


namespace average_of_numbers_is_one_l511_51132

def numbers : List Int := [-5, -2, 0, 4, 8]

theorem average_of_numbers_is_one :
  (numbers.sum : ℚ) / numbers.length = 1 := by
  sorry

end average_of_numbers_is_one_l511_51132


namespace largest_angle_convex_hexagon_l511_51137

theorem largest_angle_convex_hexagon (x : ℝ) :
  (x + 2) + (2 * x + 3) + (3 * x + 4) + (4 * x + 5) + (5 * x + 6) + (6 * x + 7) = 720 →
  max (x + 2) (max (2 * x + 3) (max (3 * x + 4) (max (4 * x + 5) (max (5 * x + 6) (6 * x + 7))))) = 205 :=
by sorry

end largest_angle_convex_hexagon_l511_51137


namespace complex_fraction_simplification_l511_51178

/-- Given that i is the imaginary unit, prove that (3 + 2i) / (1 - i) = 1/2 + 5/2 * i -/
theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 + 2 * i) / (1 - i) = 1/2 + 5/2 * i :=
by sorry

end complex_fraction_simplification_l511_51178


namespace complex_number_location_l511_51127

/-- The complex number z = 3 / (1 + 2i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_location (z : ℂ) (h : z = 3 / (1 + 2*I)) : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 := by
  sorry

end complex_number_location_l511_51127


namespace initial_staples_count_l511_51157

/-- The number of staples used per report -/
def staples_per_report : ℕ := 1

/-- The number of reports in a dozen -/
def reports_per_dozen : ℕ := 12

/-- The number of dozens of reports Stacie staples -/
def dozens_of_reports : ℕ := 3

/-- The number of staples remaining after stapling -/
def remaining_staples : ℕ := 14

/-- Theorem: The initial number of staples in the stapler is 50 -/
theorem initial_staples_count : 
  dozens_of_reports * reports_per_dozen * staples_per_report + remaining_staples = 50 := by
  sorry

end initial_staples_count_l511_51157


namespace f_composed_three_roots_l511_51163

/-- A quadratic function f(x) = x^2 - 4x + c -/
def f (c : ℝ) : ℝ → ℝ := fun x ↦ x^2 - 4*x + c

/-- The composition of f with itself -/
def f_composed (c : ℝ) : ℝ → ℝ := fun x ↦ f c (f c x)

/-- Predicate for a function having exactly three distinct real roots -/
def has_three_distinct_real_roots (g : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    g x = 0 ∧ g y = 0 ∧ g z = 0 ∧
    ∀ w, g w = 0 → w = x ∨ w = y ∨ w = z

theorem f_composed_three_roots :
  ∀ c : ℝ, has_three_distinct_real_roots (f_composed c) ↔ c = 8 :=
sorry

end f_composed_three_roots_l511_51163


namespace max_sections_five_lines_l511_51125

/-- The number of sections created by n line segments in a rectangle, 
    where each new line intersects all previous lines -/
def maxSections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else maxSections (n - 1) + n

/-- Theorem stating that 5 line segments can create at most 16 sections in a rectangle -/
theorem max_sections_five_lines :
  maxSections 5 = 16 :=
by sorry

end max_sections_five_lines_l511_51125


namespace prime_product_sum_squared_sum_l511_51188

theorem prime_product_sum_squared_sum (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a * b * c = 5 * (a + b + c) →
  a^2 + b^2 + c^2 = 78 :=
by sorry

end prime_product_sum_squared_sum_l511_51188


namespace triangle_perimeter_l511_51166

/-- The perimeter of a triangle with vertices A(3,7), B(-5,2), and C(3,2) is √89 + 13. -/
theorem triangle_perimeter : 
  let A : ℝ × ℝ := (3, 7)
  let B : ℝ × ℝ := (-5, 2)
  let C : ℝ × ℝ := (3, 2)
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B + d B C + d C A = Real.sqrt 89 + 13 := by sorry

end triangle_perimeter_l511_51166


namespace intersection_of_A_and_B_l511_51159

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | x > -1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 4} :=
sorry

end intersection_of_A_and_B_l511_51159


namespace remainder_6n_mod_4_l511_51170

theorem remainder_6n_mod_4 (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := by
  sorry

end remainder_6n_mod_4_l511_51170


namespace equation_solution_l511_51113

theorem equation_solution : 
  ∀ x : ℂ, (5 * x^2 - 3 * x + 2) / (x + 2) = 2 * x - 4 ↔ 
  x = (3 + Complex.I * Real.sqrt 111) / 6 ∨ x = (3 - Complex.I * Real.sqrt 111) / 6 :=
by sorry

end equation_solution_l511_51113


namespace lens_focal_length_theorem_l511_51147

/-- Represents a thin lens with a parallel beam of light falling normally on it. -/
structure ThinLens where
  focal_length : ℝ

/-- Represents a screen that can be placed at different distances from the lens. -/
structure Screen where
  distance : ℝ
  spot_diameter : ℝ

/-- Checks if the spot diameter remains constant when the screen is moved. -/
def constant_spot_diameter (lens : ThinLens) (screen1 screen2 : Screen) : Prop :=
  screen1.spot_diameter = screen2.spot_diameter

/-- Theorem stating the possible focal lengths of the lens given the problem conditions. -/
theorem lens_focal_length_theorem (lens : ThinLens) (screen1 screen2 : Screen) :
  screen1.distance = 80 →
  screen2.distance = 40 →
  constant_spot_diameter lens screen1 screen2 →
  lens.focal_length = 100 ∨ lens.focal_length = 60 :=
sorry

end lens_focal_length_theorem_l511_51147


namespace perfect_square_sequence_l511_51187

theorem perfect_square_sequence (a b : ℤ) :
  (∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) → a = 0 := by
  sorry

end perfect_square_sequence_l511_51187


namespace triangular_pyramid_volume_l511_51124

/-- Given a triangular pyramid with mutually perpendicular lateral faces of areas 6, 4, and 3, 
    its volume is 4. -/
theorem triangular_pyramid_volume (a b c : ℝ) 
  (h1 : a * b / 2 = 6) 
  (h2 : a * c / 2 = 4) 
  (h3 : b * c / 2 = 3) : 
  a * b * c / 6 = 4 := by
  sorry

#check triangular_pyramid_volume

end triangular_pyramid_volume_l511_51124


namespace corner_circle_radius_l511_51161

/-- The radius of a circle placed tangentially to four corner circles in a specific rectangle configuration -/
theorem corner_circle_radius (rectangle_width : ℝ) (rectangle_length : ℝ) 
  (h_width : rectangle_width = 3)
  (h_length : rectangle_length = 4)
  (large_circle_radius : ℝ)
  (h_large_radius : large_circle_radius = 2/3)
  (small_circle_radius : ℝ) :
  small_circle_radius = 1 :=
sorry

end corner_circle_radius_l511_51161


namespace cube_root_equation_solution_l511_51144

theorem cube_root_equation_solution (y : ℝ) : 
  (15 * y + (15 * y + 15) ^ (1/3 : ℝ)) ^ (1/3 : ℝ) = 15 → y = 224 := by
sorry

end cube_root_equation_solution_l511_51144


namespace bathtub_capacity_l511_51198

/-- The capacity of a bathtub given tap flow rate, filling time, and drain leak rate -/
theorem bathtub_capacity 
  (tap_flow : ℝ)  -- Tap flow rate in liters per minute
  (fill_time : ℝ)  -- Filling time in minutes
  (leak_rate : ℝ)  -- Drain leak rate in liters per minute
  (h1 : tap_flow = 21 / 6)  -- Tap flow rate condition
  (h2 : fill_time = 22.5)  -- Filling time condition
  (h3 : leak_rate = 0.3)  -- Drain leak rate condition
  : tap_flow * fill_time - leak_rate * fill_time = 72 := by
  sorry


end bathtub_capacity_l511_51198


namespace area_cyclic_quadrilateral_l511_51149

/-- The area of a convex cyclic quadrilateral -/
theorem area_cyclic_quadrilateral 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_convex : a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c) 
  (h_cyclic : ∃ (r : ℝ), r > 0 ∧ 
    a * c = (r + (a^2 / (4*r))) * (r + (c^2 / (4*r))) ∧ 
    b * d = (r + (b^2 / (4*r))) * (r + (d^2 / (4*r)))) :
  let p := (a + b + c + d) / 2
  ∃ (area : ℝ), area = Real.sqrt ((p-a)*(p-b)*(p-c)*(p-d)) := by
  sorry


end area_cyclic_quadrilateral_l511_51149


namespace john_squat_increase_l511_51173

/-- The additional weight John added to his squat after training -/
def additional_weight : ℝ := 265

/-- John's initial squat weight in pounds -/
def initial_weight : ℝ := 135

/-- The factor by which the magical bracer increases strength -/
def strength_increase_factor : ℝ := 7

/-- John's final squat weight in pounds -/
def final_weight : ℝ := 2800

theorem john_squat_increase :
  (initial_weight + additional_weight) * strength_increase_factor = final_weight :=
sorry

end john_squat_increase_l511_51173


namespace hyperbola_equation_l511_51181

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (2 : ℝ)^2 = a^2 + b^2 →
  (∀ (x y : ℝ), (b*x = a*y ∨ b*x = -a*y) → (x - 2)^2 + y^2 = 3) →
  (∀ (x y : ℝ), x^2 - y^2 / 3 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end hyperbola_equation_l511_51181


namespace elberta_money_l511_51102

theorem elberta_money (granny_smith : ℕ) (anjou : ℕ) (elberta : ℕ) : 
  granny_smith = 64 →
  anjou = granny_smith / 4 →
  elberta = anjou + 3 →
  elberta = 19 := by
  sorry

end elberta_money_l511_51102


namespace triangle_third_vertex_l511_51151

/-- Given a triangle with vertices (4, 3), (0, 0), and (x, 0) where x < 0,
    if the area of the triangle is 24 square units, then x = -16. -/
theorem triangle_third_vertex (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * abs x * 3 = 24 → x = -16 := by sorry

end triangle_third_vertex_l511_51151


namespace frequency_in_range_l511_51100

/-- Represents an interval with its frequency -/
structure IntervalData where
  lower : ℝ
  upper : ℝ
  frequency : ℕ

/-- Calculates the frequency of a sample within a given range -/
def calculateFrequency (data : List IntervalData) (range_start range_end : ℝ) (sample_size : ℕ) : ℝ :=
  sorry

/-- The given data set -/
def sampleData : List IntervalData := [
  ⟨10, 20, 2⟩,
  ⟨20, 30, 3⟩,
  ⟨30, 40, 4⟩,
  ⟨40, 50, 5⟩,
  ⟨50, 60, 4⟩,
  ⟨60, 70, 2⟩
]

theorem frequency_in_range : calculateFrequency sampleData 15 50 20 = 0.65 := by
  sorry

end frequency_in_range_l511_51100


namespace intersection_of_A_and_B_l511_51107

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 > 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_B_l511_51107


namespace a_is_integer_l511_51141

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => ((2 * n + 3) * a (n + 1) + 3 * (n + 1) * a n) / (n + 2)

theorem a_is_integer (n : ℕ) : ∃ k : ℤ, a n = k := by
  sorry

end a_is_integer_l511_51141


namespace modulus_of_complex_l511_51169

theorem modulus_of_complex (z : ℂ) : (1 - Complex.I) * z = 3 - Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_complex_l511_51169


namespace soccer_game_time_proof_l511_51194

/-- Calculates the total time in minutes for a soccer game and post-game ceremony -/
def total_time (game_hours : ℕ) (game_minutes : ℕ) (ceremony_minutes : ℕ) : ℕ :=
  game_hours * 60 + game_minutes + ceremony_minutes

/-- Proves that the total time for a 2 hour 35 minute game and 25 minute ceremony is 180 minutes -/
theorem soccer_game_time_proof :
  total_time 2 35 25 = 180 := by
  sorry

end soccer_game_time_proof_l511_51194


namespace total_questions_is_100_l511_51155

/-- Represents the scoring system and test results for a student. -/
structure TestResult where
  correct_responses : ℕ
  incorrect_responses : ℕ
  score : ℤ
  total_questions : ℕ

/-- Defines the properties of a valid test result based on the given conditions. -/
def is_valid_test_result (tr : TestResult) : Prop :=
  tr.score = tr.correct_responses - 2 * tr.incorrect_responses ∧
  tr.total_questions = tr.correct_responses + tr.incorrect_responses

/-- Theorem stating that given the conditions, the total number of questions is 100. -/
theorem total_questions_is_100 (tr : TestResult) 
  (h1 : is_valid_test_result tr) 
  (h2 : tr.score = 64) 
  (h3 : tr.correct_responses = 88) : 
  tr.total_questions = 100 := by
  sorry

end total_questions_is_100_l511_51155


namespace vector_equation_l511_51184

-- Define the vector type
variable {V : Type*} [AddCommGroup V]

-- Define points in space
variable (A B C D : V)

-- Define vectors
def vec (X Y : V) : V := Y - X

-- Theorem statement
theorem vector_equation (A B C D : V) :
  vec D A + vec C D - vec C B = vec B A := by
  sorry

end vector_equation_l511_51184


namespace smallest_circle_equation_l511_51165

/-- The equation of the circle with the smallest area that is tangent to the line 3x + 4y + 3 = 0
    and has its center on the curve y = 3/x (x > 0) -/
theorem smallest_circle_equation (x y : ℝ) :
  (∀ a : ℝ, a > 0 → ∃ r : ℝ, r > 0 ∧
    (∀ x₀ y₀ : ℝ, (x₀ - a)^2 + (y₀ - 3/a)^2 = r^2 →
      (3*x₀ + 4*y₀ + 3 = 0 → False) ∧
      (3*x₀ + 4*y₀ + 3 ≠ 0 → (3*x₀ + 4*y₀ + 3)^2 > 25*r^2))) →
  (x - 2)^2 + (y - 3/2)^2 = 9 :=
sorry

end smallest_circle_equation_l511_51165


namespace equation_solution_l511_51119

theorem equation_solution (x : ℚ) (h1 : x ≠ 0) (h2 : x ≠ -5) :
  (2 * x / (x + 5) - 1 = (x + 5) / x) ↔ (x = -5/3) := by sorry

end equation_solution_l511_51119


namespace arithmetic_sequence_sum_l511_51148

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 3rd, 5th, and 7th terms of an arithmetic sequence
    where the sum of the 2nd and 8th terms is 10. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 2 + a 8 = 10) : 
  a 3 + a 5 + a 7 = 15 := by
sorry

end arithmetic_sequence_sum_l511_51148


namespace triple_base_quadruple_exponent_l511_51186

theorem triple_base_quadruple_exponent 
  (a b : ℝ) (y : ℝ) (h1 : b ≠ 0) :
  let r := (3 * a) ^ (4 * b)
  r = a ^ b * y ^ b →
  y = 81 * a ^ 3 := by
sorry

end triple_base_quadruple_exponent_l511_51186


namespace vertical_angles_are_equal_not_equal_not_vertical_l511_51168

-- Define the concept of an angle
def Angle : Type := ℝ

-- Define the property of being vertical angles
def are_vertical_angles (a b : Angle) : Prop := sorry

-- Define the property of angles being equal
def are_equal (a b : Angle) : Prop := a = b

-- Theorem 1: If two angles are vertical angles, then they are equal
theorem vertical_angles_are_equal (a b : Angle) :
  are_vertical_angles a b → are_equal a b := by sorry

-- Theorem 2: If two angles are not equal, then they are not vertical angles
theorem not_equal_not_vertical (a b : Angle) :
  ¬(are_equal a b) → ¬(are_vertical_angles a b) := by sorry

end vertical_angles_are_equal_not_equal_not_vertical_l511_51168


namespace max_value_of_sum_of_roots_l511_51175

theorem max_value_of_sum_of_roots (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (2 * x + 27) + Real.sqrt (17 - x) + Real.sqrt (3 * x) ≤ 14.951 ∧
  ∃ x₀, x₀ = 17 ∧ Real.sqrt (2 * x₀ + 27) + Real.sqrt (17 - x₀) + Real.sqrt (3 * x₀) = 14.951 :=
by sorry

end max_value_of_sum_of_roots_l511_51175


namespace snow_probability_both_days_l511_51142

def prob_snow_monday : ℝ := 0.4
def prob_snow_tuesday : ℝ := 0.3

theorem snow_probability_both_days :
  let prob_both_days := prob_snow_monday * prob_snow_tuesday
  prob_both_days = 0.12 := by sorry

end snow_probability_both_days_l511_51142


namespace rectangular_field_area_l511_51120

/-- Given a rectangular field with one side uncovered and three sides fenced, 
    calculate its area. -/
theorem rectangular_field_area 
  (L : ℝ) -- length of the uncovered side
  (fence_length : ℝ) -- total length of fencing for three sides
  (h1 : L = 25) -- the uncovered side is 25 feet
  (h2 : fence_length = 95.4) -- the total fencing required is 95.4 feet
  : L * ((fence_length - L) / 2) = 880 := by
  sorry

#check rectangular_field_area

end rectangular_field_area_l511_51120


namespace scarf_to_tie_belt_ratio_l511_51189

-- Define the quantities given in the problem
def ties : ℕ := 34
def belts : ℕ := 40
def black_shirts : ℕ := 63
def white_shirts : ℕ := 42

-- Define the number of jeans based on the given condition
def jeans : ℕ := (2 * (black_shirts + white_shirts)) / 3

-- Define the number of scarves based on the given condition
def scarves : ℕ := jeans - 33

-- Theorem to prove
theorem scarf_to_tie_belt_ratio :
  scarves * 2 = ties + belts := by
  sorry


end scarf_to_tie_belt_ratio_l511_51189


namespace quadratic_function_properties_l511_51171

-- Define the quadratic function f(x) = ax^2 + bx + c
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties 
  (a b c : ℝ) 
  (h1 : f a b c 1 = -a/2) 
  (h2 : 3*a > 2*c) 
  (h3 : 2*c > 2*b) :
  (a > 0 ∧ -3 < b/a ∧ b/a < -3/4) ∧ 
  (∃ x, 0 < x ∧ x < 2 ∧ f a b c x = 0) ∧
  (∀ x₁ x₂, f a b c x₁ = 0 → f a b c x₂ = 0 → 
    Real.sqrt 2 ≤ |x₁ - x₂| ∧ |x₁ - x₂| < Real.sqrt 57 / 4) :=
by sorry

end quadratic_function_properties_l511_51171


namespace bowtie_equation_solution_l511_51110

/-- Definition of the bow tie operation -/
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem stating that if 5 ⋈ x = 12, then x = 42 -/
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 5 x = 12 → x = 42 :=
by
  sorry

end bowtie_equation_solution_l511_51110


namespace quadratic_below_x_axis_iff_a_in_range_l511_51167

/-- A quadratic function f(x) = ax^2 + 2ax - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x - 2

/-- The property that the graph of f is always below the x-axis -/
def always_below_x_axis (a : ℝ) : Prop :=
  ∀ x, f a x < 0

theorem quadratic_below_x_axis_iff_a_in_range :
  ∀ a : ℝ, always_below_x_axis a ↔ -2 < a ∧ a < 0 :=
sorry

end quadratic_below_x_axis_iff_a_in_range_l511_51167


namespace string_measurement_l511_51109

theorem string_measurement (string_length : Real) (cut_fraction : Real) : 
  string_length = 2/3 → 
  cut_fraction = 1/4 → 
  (1 - cut_fraction) * string_length = 1/2 := by
  sorry

end string_measurement_l511_51109


namespace parallel_vectors_x_value_l511_51192

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def u (x : ℝ) : ℝ × ℝ := a + 2 • (b x)
def v (x : ℝ) : ℝ × ℝ := 2 • a - b x

theorem parallel_vectors_x_value :
  ∃ x : ℝ, (∃ k : ℝ, u x = k • (v x)) ∧ x = 1/2 := by sorry

end parallel_vectors_x_value_l511_51192


namespace apples_given_to_neighbor_l511_51191

theorem apples_given_to_neighbor (initial_apples remaining_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : remaining_apples = 39) :
  initial_apples - remaining_apples = 88 := by
sorry

end apples_given_to_neighbor_l511_51191


namespace regular_polygon_with_144_degree_angles_has_10_sides_l511_51176

/-- The number of sides of a regular polygon with interior angles measuring 144 degrees -/
def regular_polygon_sides : ℕ :=
  let interior_angle : ℚ := 144
  let n : ℕ := 10
  n

/-- Theorem stating that a regular polygon with interior angles of 144 degrees has 10 sides -/
theorem regular_polygon_with_144_degree_angles_has_10_sides :
  let interior_angle : ℚ := 144
  (interior_angle = (180 * (regular_polygon_sides - 2) : ℚ) / regular_polygon_sides) ∧
  (regular_polygon_sides > 2) :=
by sorry

end regular_polygon_with_144_degree_angles_has_10_sides_l511_51176


namespace total_eggs_per_week_l511_51130

/-- Represents the three chicken breeds -/
inductive Breed
  | BCM  -- Black Copper Marans
  | RIR  -- Rhode Island Reds
  | LH   -- Leghorns

/-- Calculates the number of chickens for a given breed -/
def chickenCount (b : Breed) : Nat :=
  match b with
  | Breed.BCM => 125
  | Breed.RIR => 200
  | Breed.LH  => 175

/-- Calculates the number of hens for a given breed -/
def henCount (b : Breed) : Nat :=
  match b with
  | Breed.BCM => 81
  | Breed.RIR => 110
  | Breed.LH  => 105

/-- Represents the egg-laying rates for each breed -/
def eggRates (b : Breed) : List Nat :=
  match b with
  | Breed.BCM => [3, 4, 5]
  | Breed.RIR => [5, 6, 7]
  | Breed.LH  => [6, 7, 8]

/-- Represents the distribution of hens for each egg-laying rate -/
def henDistribution (b : Breed) : List Nat :=
  match b with
  | Breed.BCM => [32, 24, 25]
  | Breed.RIR => [22, 55, 33]
  | Breed.LH  => [26, 47, 32]

/-- Calculates the total eggs produced by a breed per week -/
def eggsByBreed (b : Breed) : Nat :=
  List.sum (List.zipWith (· * ·) (eggRates b) (henDistribution b))

/-- The main theorem: total eggs produced by all hens per week is 1729 -/
theorem total_eggs_per_week :
  (eggsByBreed Breed.BCM) + (eggsByBreed Breed.RIR) + (eggsByBreed Breed.LH) = 1729 := by
  sorry

#eval (eggsByBreed Breed.BCM) + (eggsByBreed Breed.RIR) + (eggsByBreed Breed.LH)

end total_eggs_per_week_l511_51130


namespace marbles_distribution_l511_51197

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 80 →
  num_boys = 8 →
  marbles_per_boy = total_marbles / num_boys →
  marbles_per_boy = 10 := by
  sorry

end marbles_distribution_l511_51197


namespace coconut_grove_yield_l511_51117

/-- Calculates the yield of the third group of trees in a coconut grove --/
theorem coconut_grove_yield (x : ℕ) (Y : ℕ) : x = 6 →
  ((x + 3) * 60 + x * 120 + (x - 3) * Y) / (3 * x) = 100 →
  Y = 180 := by
  sorry

end coconut_grove_yield_l511_51117


namespace ab_range_l511_51150

theorem ab_range (a b q : ℝ) (h1 : (1/3 : ℝ) ≤ q ∧ q ≤ 2) 
  (h2 : ∃ m : ℝ, ∃ r1 r2 r3 r4 : ℝ, 
    (r1^2 - a*r1 + 1)*(r1^2 - b*r1 + 1) = 0 ∧
    (r2^2 - a*r2 + 1)*(r2^2 - b*r2 + 1) = 0 ∧
    (r3^2 - a*r3 + 1)*(r3^2 - b*r3 + 1) = 0 ∧
    (r4^2 - a*r4 + 1)*(r4^2 - b*r4 + 1) = 0 ∧
    r1 = m ∧ r2 = m*q ∧ r3 = m*q^2 ∧ r4 = m*q^3) :
  4 ≤ a*b ∧ a*b ≤ 112/9 := by sorry

end ab_range_l511_51150


namespace expression_equals_zero_l511_51139

theorem expression_equals_zero (a : ℚ) (h : a = 4/3) : 
  (6*a^2 - 15*a + 5) * (3*a - 4) = 0 := by
  sorry

end expression_equals_zero_l511_51139


namespace inequality_proof_equality_condition_l511_51136

theorem inequality_proof (a b n : ℕ) (h1 : a > b) (h2 : a * b - 1 = n^2) :
  a - b ≥ Real.sqrt (4 * n - 3) := by
  sorry

theorem equality_condition (a b n : ℕ) (h1 : a > b) (h2 : a * b - 1 = n^2) :
  (a - b = Real.sqrt (4 * n - 3)) ↔ 
  (∃ u : ℕ, a = u^2 + 2*u + 2 ∧ b = u^2 + 1 ∧ n = u^2 + u + 1) := by
  sorry

end inequality_proof_equality_condition_l511_51136


namespace intersection_angle_zero_curve_intersects_y_axis_at_zero_angle_l511_51140

noncomputable def f (x : ℝ) := Real.exp x - x

theorem intersection_angle_zero : 
  let slope := (deriv f) 0
  slope = 0 := by sorry

-- The angle of intersection is the arctangent of the slope
theorem curve_intersects_y_axis_at_zero_angle : 
  Real.arctan ((deriv f) 0) = 0 := by sorry

end intersection_angle_zero_curve_intersects_y_axis_at_zero_angle_l511_51140


namespace congruence_in_range_l511_51174

theorem congruence_in_range : 
  ∀ n : ℤ, 10 ≤ n ∧ n ≤ 20 ∧ n ≡ 12345 [ZMOD 7] → n = 11 ∨ n = 18 := by
  sorry

end congruence_in_range_l511_51174


namespace total_followers_after_one_month_l511_51182

/-- Represents the number of followers on various social media platforms -/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ
  pinterest : ℕ
  snapchat : ℕ

/-- Calculates the total number of followers across all platforms -/
def total_followers (f : Followers) : ℕ :=
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube + f.pinterest + f.snapchat

/-- Represents the initial number of followers -/
def initial_followers : Followers := {
  instagram := 240,
  facebook := 500,
  twitter := (240 + 500) / 2,
  tiktok := 3 * ((240 + 500) / 2),
  youtube := 3 * ((240 + 500) / 2) + 510,
  pinterest := 120,
  snapchat := 120 / 2
}

/-- Represents the number of followers after one month -/
def followers_after_one_month : Followers := {
  instagram := initial_followers.instagram + (initial_followers.instagram * 15 / 100),
  facebook := initial_followers.facebook + (initial_followers.facebook * 20 / 100),
  twitter := initial_followers.twitter + 30,
  tiktok := initial_followers.tiktok + 45,
  youtube := initial_followers.youtube,
  pinterest := initial_followers.pinterest,
  snapchat := initial_followers.snapchat - 10
}

/-- Theorem stating that the total number of followers after one month is 4221 -/
theorem total_followers_after_one_month : 
  total_followers followers_after_one_month = 4221 := by
  sorry


end total_followers_after_one_month_l511_51182


namespace multiplication_value_problem_l511_51103

theorem multiplication_value_problem : 
  ∃ x : ℝ, (4.5 / 6) * x = 9 ∧ x = 12 := by
sorry

end multiplication_value_problem_l511_51103


namespace expand_product_l511_51146

theorem expand_product (x : ℝ) : (x + 3) * (x + 4 + 6) = x^2 + 13*x + 30 := by
  sorry

end expand_product_l511_51146


namespace max_cube_sum_on_unit_circle_l511_51108

theorem max_cube_sum_on_unit_circle :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 → |x^3| + |y^3| ≤ M) ∧
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ |x^3| + |y^3| = M) := by
sorry

end max_cube_sum_on_unit_circle_l511_51108


namespace quadratic_solution_value_l511_51128

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The solution set of the inequality f(x) < c -/
structure SolutionSet (f : ℝ → ℝ) (c : ℝ) where
  m : ℝ
  property : Set.Ioo m (m + 6) = {x | f x < c}

/-- The theorem stating that c = 9 given the conditions -/
theorem quadratic_solution_value
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h_f : f = QuadraticFunction a b)
  (h_range : Set.range f = Set.Ici 0)
  (h_solution : SolutionSet f c)
  : c = 9 := by
  sorry

end quadratic_solution_value_l511_51128


namespace largest_expression_l511_51115

theorem largest_expression : ∀ (a b c d e : ℕ),
  a = 3 + 1 + 2 + 8 →
  b = 3 * 1 + 2 + 8 →
  c = 3 + 1 * 2 + 8 →
  d = 3 + 1 + 2 * 8 →
  e = 3 * 1 * 2 * 8 →
  e ≥ a ∧ e ≥ b ∧ e ≥ c ∧ e ≥ d :=
by sorry

end largest_expression_l511_51115


namespace encryption_3859_l511_51116

def encrypt_digit (d : Nat) : Nat :=
  (d^3 + 1) % 10

def encrypt_number (n : List Nat) : List Nat :=
  n.map encrypt_digit

theorem encryption_3859 :
  encrypt_number [3, 8, 5, 9] = [8, 3, 6, 0] := by
  sorry

end encryption_3859_l511_51116


namespace geometric_and_arithmetic_properties_l511_51193

theorem geometric_and_arithmetic_properties :
  ∀ (s r h : ℝ) (a b : ℝ) (x : ℝ),
  s > 0 → r > 0 → h > 0 → b ≠ 0 →
  (2 * s)^2 = 4 * s^2 ∧
  (π * r^2 * (2 * h)) = 2 * (π * r^2 * h) ∧
  (2 * s)^3 = 8 * s^3 ∧
  (2 * a) / (b / 2) = 4 * (a / b) ∧
  x + 0 = x :=
by sorry

end geometric_and_arithmetic_properties_l511_51193


namespace lily_sees_leo_l511_51129

/-- The time Lily can see Leo given their speeds and distances -/
theorem lily_sees_leo (lily_speed leo_speed initial_distance final_distance : ℝ) : 
  lily_speed = 15 → 
  leo_speed = 9 → 
  initial_distance = 0.75 → 
  final_distance = 0.75 → 
  (initial_distance + final_distance) / (lily_speed - leo_speed) * 60 = 15 := by
  sorry

end lily_sees_leo_l511_51129


namespace work_hours_per_day_l511_51195

/-- Proves that working 56 hours over 14 days results in 4 hours of work per day -/
theorem work_hours_per_day (total_hours : ℕ) (total_days : ℕ) (hours_per_day : ℕ) : 
  total_hours = 56 → total_days = 14 → total_hours = total_days * hours_per_day → hours_per_day = 4 := by
  sorry

#check work_hours_per_day

end work_hours_per_day_l511_51195


namespace square_area_expansion_l511_51114

theorem square_area_expansion (a : ℝ) (h : a > 0) :
  (3 * a)^2 = 9 * a^2 := by sorry

end square_area_expansion_l511_51114


namespace circle_radius_from_circumference_and_area_l511_51180

/-- Given a circle with specified circumference and area, prove its radius is approximately 4 cm. -/
theorem circle_radius_from_circumference_and_area 
  (circumference : ℝ) 
  (area : ℝ) 
  (h_circumference : circumference = 25.132741228718345)
  (h_area : area = 50.26548245743669) :
  ∃ (radius : ℝ), abs (radius - 4) < 0.0001 ∧ 
    circumference = 2 * Real.pi * radius ∧ 
    area = Real.pi * radius ^ 2 := by
  sorry

end circle_radius_from_circumference_and_area_l511_51180


namespace sams_puppies_l511_51118

theorem sams_puppies (initial_spotted : ℕ) (initial_nonspotted : ℕ) 
  (given_away_spotted : ℕ) (given_away_nonspotted : ℕ) 
  (remaining_spotted : ℕ) (remaining_nonspotted : ℕ) : ℕ :=
  by
  have h1 : initial_spotted = 8 := by sorry
  have h2 : initial_nonspotted = 5 := by sorry
  have h3 : given_away_spotted = 2 := by sorry
  have h4 : given_away_nonspotted = 3 := by sorry
  have h5 : remaining_spotted = 6 := by sorry
  have h6 : remaining_nonspotted = 2 := by sorry
  have h7 : initial_spotted - given_away_spotted = remaining_spotted := by sorry
  have h8 : initial_nonspotted - given_away_nonspotted = remaining_nonspotted := by sorry
  exact initial_spotted + initial_nonspotted

end sams_puppies_l511_51118


namespace complement_A_intersect_B_l511_51112

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 ≤ 0}
def B : Set ℝ := {x | x > 7}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = Set.Ioi 7 := by sorry

end complement_A_intersect_B_l511_51112


namespace square_perimeter_sum_l511_51172

theorem square_perimeter_sum (a b : ℝ) (h1 : a^2 + b^2 = 85) (h2 : a^2 - b^2 = 45) :
  4*a + 4*b = 4*(Real.sqrt 65 + 2*Real.sqrt 5) := by
sorry

end square_perimeter_sum_l511_51172


namespace jack_afternoon_letters_l511_51199

/-- The number of letters Jack received in the morning -/
def morning_letters : ℕ := 8

/-- The number of letters Jack received in the afternoon -/
def afternoon_letters : ℕ := morning_letters - 1

theorem jack_afternoon_letters : afternoon_letters = 7 := by
  sorry

end jack_afternoon_letters_l511_51199


namespace ratio_of_powers_compute_power_ratio_l511_51158

theorem ratio_of_powers (a b : ℕ) (n : ℕ) (h : b ≠ 0) :
  (a ^ n) / (b ^ n) = (a / b) ^ n :=
sorry

theorem compute_power_ratio :
  (90000 ^ 5) / (30000 ^ 5) = 243 :=
sorry

end ratio_of_powers_compute_power_ratio_l511_51158


namespace fraction_calculation_l511_51154

theorem fraction_calculation : 
  let f1 := 531 / 135
  let f2 := 579 / 357
  let f3 := 753 / 975
  let f4 := 135 / 531
  (f1 + f2 + f3) * (f2 + f3 + f4) - (f1 + f2 + f3 + f4) * (f2 + f3) = 1 := by
  sorry

end fraction_calculation_l511_51154


namespace pedestrians_collinear_at_most_twice_l511_51138

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a pedestrian's motion in 2D space -/
structure Pedestrian where
  initial_pos : Point2D
  velocity : Point2D

/-- Three pedestrians walking in straight lines -/
def three_pedestrians (p1 p2 p3 : Pedestrian) : Prop :=
  -- Pedestrians have constant velocities
  ∀ t : ℝ, ∃ (pos1 pos2 pos3 : Point2D),
    pos1 = Point2D.mk (p1.initial_pos.x + p1.velocity.x * t) (p1.initial_pos.y + p1.velocity.y * t) ∧
    pos2 = Point2D.mk (p2.initial_pos.x + p2.velocity.x * t) (p2.initial_pos.y + p2.velocity.y * t) ∧
    pos3 = Point2D.mk (p3.initial_pos.x + p3.velocity.x * t) (p3.initial_pos.y + p3.velocity.y * t)

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- The main theorem -/
theorem pedestrians_collinear_at_most_twice
  (p1 p2 p3 : Pedestrian)
  (h_not_initially_collinear : ¬are_collinear p1.initial_pos p2.initial_pos p3.initial_pos)
  (h_walking : three_pedestrians p1 p2 p3) :
  ∃ (t1 t2 : ℝ), ∀ t : ℝ,
    are_collinear
      (Point2D.mk (p1.initial_pos.x + p1.velocity.x * t) (p1.initial_pos.y + p1.velocity.y * t))
      (Point2D.mk (p2.initial_pos.x + p2.velocity.x * t) (p2.initial_pos.y + p2.velocity.y * t))
      (Point2D.mk (p3.initial_pos.x + p3.velocity.x * t) (p3.initial_pos.y + p3.velocity.y * t))
    → t = t1 ∨ t = t2 :=
  sorry

end pedestrians_collinear_at_most_twice_l511_51138


namespace inequality_proof_l511_51101

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end inequality_proof_l511_51101


namespace inequality_proof_l511_51177

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : a / b < c / d) : 
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d := by
  sorry

end inequality_proof_l511_51177


namespace chips_cost_calculation_l511_51183

/-- Given the original cost and discount of chips, calculate the actual amount spent -/
theorem chips_cost_calculation (original_cost discount : ℚ) 
  (h1 : original_cost = 35)
  (h2 : discount = 17) :
  original_cost - discount = 18 := by
  sorry

end chips_cost_calculation_l511_51183


namespace platform_and_train_length_l511_51162

/-- The combined length of a platform and a train, given the speeds and passing times of two trains. -/
theorem platform_and_train_length
  (t1_platform_time : ℝ)
  (t1_man_time : ℝ)
  (t1_speed : ℝ)
  (t2_speed : ℝ)
  (t2_man_time : ℝ)
  (h1 : t1_platform_time = 16)
  (h2 : t1_man_time = 10)
  (h3 : t1_speed = 54 * 1000 / 3600)
  (h4 : t2_speed = 72 * 1000 / 3600)
  (h5 : t2_man_time = 12) :
  t1_speed * (t1_platform_time - t1_man_time) + t2_speed * t2_man_time = 330 :=
by sorry

end platform_and_train_length_l511_51162


namespace smallest_overlap_percentage_l511_51156

theorem smallest_overlap_percentage (smartphone_users laptop_users : ℝ) 
  (h1 : smartphone_users = 90) 
  (h2 : laptop_users = 80) : 
  ∃ (overlap : ℝ), overlap ≥ 70 ∧ 
    ∀ (x : ℝ), x < 70 → smartphone_users + laptop_users - x > 100 :=
by sorry

end smallest_overlap_percentage_l511_51156


namespace min_value_of_f_l511_51133

-- Define the function
def f (x : ℝ) : ℝ := x^4 - 4*x + 3

-- Define the interval
def interval : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 0 ∧ ∀ y ∈ interval, f y ≥ f x :=
sorry

end min_value_of_f_l511_51133


namespace extended_segment_endpoint_l511_51104

/-- Given a segment AB with endpoints A(3, 3) and B(15, 9), extended through B to point C
    such that BC = 1/2 * AB, the coordinates of point C are (21, 12). -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (3, 3) → B = (15, 9) → 
  (C.1 - B.1, C.2 - B.2) = (1/2 * (B.1 - A.1), 1/2 * (B.2 - A.2)) →
  C = (21, 12) := by
  sorry

end extended_segment_endpoint_l511_51104


namespace divisible_by_21_l511_51126

theorem divisible_by_21 (N : Finset ℕ) 
  (h_card : N.card = 46)
  (h_div_3 : (N.filter (fun n => n % 3 = 0)).card = 35)
  (h_div_7 : (N.filter (fun n => n % 7 = 0)).card = 12) :
  ∃ n ∈ N, n % 21 = 0 := by
  sorry

end divisible_by_21_l511_51126


namespace polynomial_equality_l511_51121

-- Define the polynomial (x+y)^8
def polynomial (x y : ℝ) : ℝ := (x + y)^8

-- Define the third term of the expansion
def third_term (x y : ℝ) : ℝ := 28 * x^6 * y^2

-- Define the fourth term of the expansion
def fourth_term (x y : ℝ) : ℝ := 56 * x^5 * y^3

theorem polynomial_equality (p q : ℝ) :
  p > 0 ∧ q > 0 ∧ p + q = 1 ∧ third_term p q = fourth_term p q → p = 2/3 := by
  sorry


end polynomial_equality_l511_51121


namespace gcd_of_45_and_75_l511_51105

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_of_45_and_75_l511_51105


namespace bf_equals_ce_l511_51152

-- Define the triangle ABC
variable (A B C : Point)

-- Define D as the foot of the angle bisector from A
def D : Point := sorry

-- Define E as the intersection of circumcircle ABD with AC
def E : Point := sorry

-- Define F as the intersection of circumcircle ADC with AB
def F : Point := sorry

-- Theorem statement
theorem bf_equals_ce : BF = CE := by sorry

end bf_equals_ce_l511_51152


namespace cubic_transformation_l511_51123

theorem cubic_transformation (x z : ℝ) (hz : z = x + 1/x) :
  x^3 - 3*x^2 + x + 2 = 0 ↔ x^2*(z^2 - z - 1) + 3 = 0 := by
  sorry

end cubic_transformation_l511_51123


namespace complex_multiplication_l511_51131

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 - 2*i) = 2 + i := by
  sorry

end complex_multiplication_l511_51131


namespace sqrt_meaningful_range_l511_51179

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 5) ↔ x ≥ 5 / 2 :=
by sorry

end sqrt_meaningful_range_l511_51179


namespace chloe_score_l511_51196

/-- The score for each treasure found in the game -/
def points_per_treasure : ℕ := 9

/-- The number of treasures found on the first level -/
def treasures_level_1 : ℕ := 6

/-- The number of treasures found on the second level -/
def treasures_level_2 : ℕ := 3

/-- Chloe's total score in the game -/
def total_score : ℕ := points_per_treasure * (treasures_level_1 + treasures_level_2)

/-- Theorem stating that Chloe's total score is 81 points -/
theorem chloe_score : total_score = 81 := by
  sorry

end chloe_score_l511_51196


namespace pinball_spend_proof_l511_51145

def half_dollar : ℚ := 0.5

def wednesday_spend : ℕ := 4
def thursday_spend : ℕ := 14

def total_spend : ℚ := (wednesday_spend * half_dollar) + (thursday_spend * half_dollar)

theorem pinball_spend_proof : total_spend = 9 := by
  sorry

end pinball_spend_proof_l511_51145
