import Mathlib

namespace rectangle_area_l291_29198

theorem rectangle_area (length width diagonal : ℝ) : 
  length = 16 →
  length / diagonal = 4 / 5 →
  length ^ 2 + width ^ 2 = diagonal ^ 2 →
  length * width = 192 :=
by
  sorry

end rectangle_area_l291_29198


namespace parabola_translation_l291_29185

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - k }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 8 0 0
  let translated := translate original 3 (-5)
  y = 8 * x^2 → y = translated.a * (x - 3)^2 + translated.b * (x - 3) + translated.c :=
by sorry

end parabola_translation_l291_29185


namespace football_player_average_increase_l291_29129

theorem football_player_average_increase :
  ∀ (total_goals : ℕ) (goals_fifth_match : ℕ) (num_matches : ℕ),
    total_goals = 16 →
    goals_fifth_match = 4 →
    num_matches = 5 →
    (total_goals : ℚ) / num_matches - ((total_goals - goals_fifth_match) : ℚ) / (num_matches - 1) = 0.2 := by
  sorry

end football_player_average_increase_l291_29129


namespace not_p_sufficient_not_necessary_for_not_q_l291_29169

-- Define propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≤ 4
def q (x : ℝ) : Prop := x^2 < 5*x - 6

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬p is a sufficient but not necessary condition for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  (∃ x : ℝ, not_q x ∧ ¬(not_p x)) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l291_29169


namespace mean_temperature_l291_29134

def temperatures : List ℝ := [82, 80, 83, 88, 84, 90, 92, 85, 89, 90]

theorem mean_temperature (temps := temperatures) : 
  (temps.sum / temps.length : ℝ) = 86.3 := by
  sorry

end mean_temperature_l291_29134


namespace no_alpha_sequence_exists_l291_29114

theorem no_alpha_sequence_exists : ¬∃ (α : ℝ) (a : ℕ → ℝ), 
  (0 < α ∧ α < 1) ∧ 
  (∀ n, 0 < a n) ∧
  (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) := by
  sorry

end no_alpha_sequence_exists_l291_29114


namespace semicircle_area_ratio_l291_29131

theorem semicircle_area_ratio (r : ℝ) (h : r > 0) :
  let semicircle_area := π * (r / Real.sqrt 2)^2 / 2
  let circle_area := π * r^2
  2 * semicircle_area / circle_area = 1 / 2 := by sorry

end semicircle_area_ratio_l291_29131


namespace figure_to_square_possible_l291_29130

/-- A figure on a grid paper -/
structure GridFigure where
  area : ℕ

/-- Represents a dissection of a figure into parts -/
structure Dissection where
  parts : ℕ

/-- Represents a square shape -/
structure Square where
  side_length : ℕ

/-- A function that checks if a figure can be dissected into parts and formed into a square -/
def can_form_square (figure : GridFigure) (d : Dissection) (s : Square) : Prop :=
  figure.area = s.side_length ^ 2 ∧ d.parts = 3

theorem figure_to_square_possible (figure : GridFigure) (d : Dissection) (s : Square) 
  (h_area : figure.area = 16) (h_parts : d.parts = 3) (h_side : s.side_length = 4) : 
  can_form_square figure d s := by
  sorry

#check figure_to_square_possible

end figure_to_square_possible_l291_29130


namespace smallest_base_perfect_square_l291_29113

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 3 ∧ 
  (∃ (n : ℕ), 4 * b + 5 = n^2) ∧ 
  (∀ (x : ℕ), x > 3 ∧ x < b → ¬∃ (m : ℕ), 4 * x + 5 = m^2) ∧
  b = 5 := by
  sorry

end smallest_base_perfect_square_l291_29113


namespace parallelogram_z_range_l291_29141

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (4, -2)

-- Define the function z
def z (x y : ℝ) : ℝ := 2 * x - 5 * y

-- Theorem statement
theorem parallelogram_z_range :
  ∀ (x y : ℝ), 
  (∃ (t₁ t₂ t₃ : ℝ), 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ 0 ≤ t₃ ∧ t₁ + t₂ + t₃ ≤ 1 ∧
    (x, y) = t₁ • A + t₂ • B + t₃ • C + (1 - t₁ - t₂ - t₃) • (A + C - B)) →
  -14 ≤ z x y ∧ z x y ≤ 20 :=
sorry

end parallelogram_z_range_l291_29141


namespace constant_difference_expressions_l291_29195

theorem constant_difference_expressions (x : ℤ) : 
  (∃ k : ℤ, (x^2 - 4*x + 5) - (2*x - 6) = k ∧ 
             (4*x - 8) - (x^2 - 4*x + 5) = k ∧ 
             (3*x^2 - 12*x + 11) - (4*x - 8) = k) ↔ 
  x = 4 := by
sorry

end constant_difference_expressions_l291_29195


namespace largest_prime_factor_of_4290_l291_29147

theorem largest_prime_factor_of_4290 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4290 ∧ ∀ q, Nat.Prime q → q ∣ 4290 → q ≤ p :=
by sorry

end largest_prime_factor_of_4290_l291_29147


namespace perpendicular_vectors_t_value_l291_29152

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_t_value :
  ∀ t : ℝ, perpendicular (3, 1) (t, -3) → t = 1 := by
  sorry

end perpendicular_vectors_t_value_l291_29152


namespace cricket_bat_selling_price_l291_29104

-- Define the profit amount
def profit : ℝ := 230

-- Define the profit percentage
def profitPercentage : ℝ := 37.096774193548384

-- Define the selling price
def sellingPrice : ℝ := 850

-- Theorem to prove
theorem cricket_bat_selling_price :
  (profit / (profitPercentage / 100) + profit) = sellingPrice := by
  sorry

end cricket_bat_selling_price_l291_29104


namespace cubic_equation_solution_l291_29182

theorem cubic_equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  (∃ a b : ℝ, ∀ x : ℝ, x = a * m + b * n → (x + m)^3 - (x + n)^3 = (m - n)^3) ↔
  (∀ x : ℝ, (x + m)^3 - (x + n)^3 = (m - n)^3 ↔ x = -m + n) :=
by sorry

end cubic_equation_solution_l291_29182


namespace total_distance_calculation_l291_29100

/-- Calculates the total distance covered by a man rowing upstream and downstream -/
theorem total_distance_calculation (upstream_speed : ℝ) (upstream_time : ℝ) 
  (downstream_speed : ℝ) (downstream_time : ℝ) : 
  upstream_speed * upstream_time + downstream_speed * downstream_time = 62 :=
by
  -- Proof goes here
  sorry

#check total_distance_calculation 12 2 38 1

end total_distance_calculation_l291_29100


namespace valid_B_values_l291_29125

def is_valid_B (B : ℕ) : Prop :=
  B < 10 ∧ (∃ k : ℤ, 40000 + 1110 * B + 2 = 9 * k)

theorem valid_B_values :
  ∀ B : ℕ, is_valid_B B ↔ (B = 1 ∨ B = 4 ∨ B = 7) :=
by sorry

end valid_B_values_l291_29125


namespace irreducible_fraction_l291_29115

theorem irreducible_fraction (n : ℕ) : 
  (Nat.gcd (21 * n + 4) (14 * n + 1) = 1) ↔ (n % 5 ≠ 1) := by sorry

end irreducible_fraction_l291_29115


namespace jane_exercise_hours_per_day_l291_29163

/-- Given Jane's exercise routine, prove the number of hours she exercises per day --/
theorem jane_exercise_hours_per_day 
  (days_per_week : ℕ) 
  (total_weeks : ℕ) 
  (total_hours : ℕ) 
  (h1 : days_per_week = 5)
  (h2 : total_weeks = 8)
  (h3 : total_hours = 40) :
  total_hours / (total_weeks * days_per_week) = 1 :=
by
  sorry

end jane_exercise_hours_per_day_l291_29163


namespace smallest_positive_integer_with_remainders_l291_29181

theorem smallest_positive_integer_with_remainders : 
  ∃ n : ℕ, n > 1 ∧ 
    n % 5 = 1 ∧ 
    n % 7 = 1 ∧ 
    n % 8 = 1 ∧ 
    (∀ m : ℕ, m > 1 → m % 5 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
    80 < n ∧ 
    n < 299 := by
  sorry

end smallest_positive_integer_with_remainders_l291_29181


namespace value_of_expression_l291_29178

theorem value_of_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 4) : 2 * (x + y) = 6 := by
  sorry

end value_of_expression_l291_29178


namespace english_spanish_difference_l291_29175

/-- Ryan's learning schedule for three days -/
structure LearningSchedule :=
  (day1_english : ℕ) (day1_chinese : ℕ) (day1_spanish : ℕ)
  (day2_english : ℕ) (day2_chinese : ℕ) (day2_spanish : ℕ)
  (day3_english : ℕ) (day3_chinese : ℕ) (day3_spanish : ℕ)

/-- Ryan's actual learning schedule -/
def ryans_schedule : LearningSchedule :=
  { day1_english := 7, day1_chinese := 2, day1_spanish := 4,
    day2_english := 6, day2_chinese := 3, day2_spanish := 5,
    day3_english := 8, day3_chinese := 1, day3_spanish := 3 }

/-- Calculate the total hours spent on a language over three days -/
def total_hours (schedule : LearningSchedule) (language : String) : ℕ :=
  match language with
  | "English" => schedule.day1_english + schedule.day2_english + schedule.day3_english
  | "Spanish" => schedule.day1_spanish + schedule.day2_spanish + schedule.day3_spanish
  | _ => 0

/-- Theorem: Ryan spends 9 more hours on English than Spanish -/
theorem english_spanish_difference :
  total_hours ryans_schedule "English" - total_hours ryans_schedule "Spanish" = 9 := by
  sorry

end english_spanish_difference_l291_29175


namespace polynomial_equality_unique_solution_l291_29192

theorem polynomial_equality_unique_solution :
  ∃! (a b c : ℤ), ∀ (x : ℝ), (x - a) * (x - 11) + 2 = (x + b) * (x + c) ∧
  a = 13 ∧ b = -13 ∧ c = -12 :=
sorry

end polynomial_equality_unique_solution_l291_29192


namespace geometric_sequence_product_roots_product_geometric_sequence_roots_product_l291_29117

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ i j k l : ℕ, i + j = k + l → a i * a j = a k * a l :=
sorry

theorem roots_product (p q r : ℝ) (x y : ℝ) (hx : p * x^2 + q * x + r = 0) (hy : p * y^2 + q * y + r = 0) :
  x * y = r / p :=
sorry

theorem geometric_sequence_roots_product (a : ℕ → ℝ) :
  geometric_sequence a →
  3 * (a 1)^2 + 7 * (a 1) - 9 = 0 →
  3 * (a 10)^2 + 7 * (a 10) - 9 = 0 →
  a 4 * a 7 = -3 :=
sorry

end geometric_sequence_product_roots_product_geometric_sequence_roots_product_l291_29117


namespace quadratic_property_l291_29150

/-- Quadratic function -/
def f (c : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + c

theorem quadratic_property (c : ℝ) (x₁ : ℝ) (hc : c < 0) (hx₁ : f c x₁ > 0) :
  f c (x₁ - 2) < 0 ∧ f c (x₁ + 2) < 0 := by
  sorry

end quadratic_property_l291_29150


namespace largest_x_sqrt_3x_eq_5x_squared_l291_29142

theorem largest_x_sqrt_3x_eq_5x_squared :
  let f : ℝ → ℝ := λ x => Real.sqrt (3 * x) - 5 * x^2
  ∃ (max_x : ℝ), max_x = (3/25)^(1/3) ∧
    (∀ x : ℝ, f x = 0 → x ≤ max_x) ∧
    f max_x = 0 :=
by sorry

end largest_x_sqrt_3x_eq_5x_squared_l291_29142


namespace F_composition_result_l291_29173

def F (x : ℝ) : ℝ := 2 * x - 1

theorem F_composition_result : F (F (F (F (F 2)))) = 33 := by
  sorry

end F_composition_result_l291_29173


namespace complex_square_l291_29140

theorem complex_square (a b : ℝ) (h : (a : ℂ) + Complex.I = 2 - b * Complex.I) :
  (a + b * Complex.I)^2 = 3 - 4 * Complex.I := by sorry

end complex_square_l291_29140


namespace fraction_subtraction_l291_29149

theorem fraction_subtraction : 
  (4 : ℚ) / 5 - (1 : ℚ) / 5 = (6 : ℚ) / 10 := by sorry

end fraction_subtraction_l291_29149


namespace eccentricity_of_hyperbola_with_diagonal_asymptotes_l291_29165

/-- A hyperbola with given asymptotes -/
structure Hyperbola where
  -- Asymptotes of the hyperbola are y = ±x
  asymptotes : (ℝ → ℝ) × (ℝ → ℝ)
  asymptotes_prop : asymptotes = ((fun x => x), (fun x => -x))

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: The eccentricity of a hyperbola with asymptotes y = ±x is √2 -/
theorem eccentricity_of_hyperbola_with_diagonal_asymptotes (h : Hyperbola) :
  eccentricity h = Real.sqrt 2 := by sorry

end eccentricity_of_hyperbola_with_diagonal_asymptotes_l291_29165


namespace semi_circle_perimeter_after_increase_l291_29143

/-- The perimeter of a semi-circle with radius 7.68 cm is approximately 39.50 cm. -/
theorem semi_circle_perimeter_after_increase : 
  let r : ℝ := 7.68
  let π : ℝ := 3.14159
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, |perimeter - 39.50| < ε :=
by sorry

end semi_circle_perimeter_after_increase_l291_29143


namespace employee_pay_percentage_l291_29132

theorem employee_pay_percentage (total_pay B_pay : ℝ) (h1 : total_pay = 550) (h2 : B_pay = 249.99999999999997) :
  let A_pay := total_pay - B_pay
  (A_pay / B_pay) * 100 = 120 := by
sorry

end employee_pay_percentage_l291_29132


namespace walkway_time_against_l291_29102

/-- Represents the scenario of a person walking on a moving walkway. -/
structure WalkwayScenario where
  length : ℝ  -- Length of the walkway in meters
  time_with : ℝ  -- Time to walk with the walkway in seconds
  time_stationary : ℝ  -- Time to walk when the walkway is not moving in seconds

/-- Calculates the time to walk against the walkway given a WalkwayScenario. -/
def time_against (scenario : WalkwayScenario) : ℝ :=
  sorry

/-- Theorem stating that for the given scenario, the time to walk against the walkway is 120 seconds. -/
theorem walkway_time_against 
  (scenario : WalkwayScenario)
  (h1 : scenario.length = 60)
  (h2 : scenario.time_with = 30)
  (h3 : scenario.time_stationary = 48) :
  time_against scenario = 120 :=
sorry

end walkway_time_against_l291_29102


namespace range_of_2m_plus_n_l291_29155

noncomputable def f (x : ℝ) := |Real.log x / Real.log 3|

theorem range_of_2m_plus_n (m n : ℝ) (h1 : 0 < m) (h2 : m < n) (h3 : f m = f n) :
  ∃ (lower : ℝ), lower = 2 * Real.sqrt 2 ∧
  (∀ x, x ≥ lower ↔ ∃ (m' n' : ℝ), 0 < m' ∧ m' < n' ∧ f m' = f n' ∧ 2 * m' + n' = x) :=
sorry

end range_of_2m_plus_n_l291_29155


namespace dave_guitar_strings_l291_29190

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 2

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 12

/-- The total number of guitar strings Dave needs to replace -/
def total_strings : ℕ := strings_per_night * shows_per_week * total_weeks

theorem dave_guitar_strings :
  total_strings = 144 := by sorry

end dave_guitar_strings_l291_29190


namespace greatest_integer_difference_l291_29136

theorem greatest_integer_difference (x y : ℝ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) :
  ∃ (n : ℕ), n = ⌊y - x⌋ ∧ n ≤ 6 ∧ ∀ (m : ℕ), m = ⌊y - x⌋ → m ≤ n :=
sorry

end greatest_integer_difference_l291_29136


namespace parametric_to_regular_equation_l291_29135

theorem parametric_to_regular_equation 
  (t : ℝ) (ht : t ≠ 0) 
  (x : ℝ) (hx : x = t + 1/t) 
  (y : ℝ) (hy : y = t^2 + 1/t^2) : 
  x^2 - y - 2 = 0 ∧ y ≥ 2 := by
  sorry

end parametric_to_regular_equation_l291_29135


namespace jenny_essay_copies_l291_29121

/-- Represents the problem of determining how many copies Jenny wants to print -/
theorem jenny_essay_copies : 
  let cost_per_page : ℚ := 1 / 10
  let essay_pages : ℕ := 25
  let num_pens : ℕ := 7
  let cost_per_pen : ℚ := 3 / 2
  let payment : ℕ := 2 * 20
  let change : ℕ := 12
  
  let total_spent : ℚ := payment - change
  let pen_cost : ℚ := num_pens * cost_per_pen
  let printing_cost : ℚ := total_spent - pen_cost
  let cost_per_copy : ℚ := cost_per_page * essay_pages
  let num_copies : ℚ := printing_cost / cost_per_copy
  
  num_copies = 7 := by sorry

end jenny_essay_copies_l291_29121


namespace coopers_fence_depth_l291_29139

/-- Proves that the depth of each wall in Cooper's fence is 2 bricks -/
theorem coopers_fence_depth (num_walls : ℕ) (wall_length : ℕ) (wall_height : ℕ) (total_bricks : ℕ) :
  num_walls = 4 →
  wall_length = 20 →
  wall_height = 5 →
  total_bricks = 800 →
  (total_bricks / (num_walls * wall_length * wall_height) : ℚ) = 2 := by
  sorry

end coopers_fence_depth_l291_29139


namespace chromium_percent_alloy1_l291_29153

-- Define the weights and percentages
def weight_alloy1 : ℝ := 15
def weight_alloy2 : ℝ := 30
def chromium_percent_alloy2 : ℝ := 8
def chromium_percent_new : ℝ := 9.333333333333334

-- Theorem statement
theorem chromium_percent_alloy1 :
  ∃ (x : ℝ),
    x ≥ 0 ∧ x ≤ 100 ∧
    (x / 100 * weight_alloy1 + chromium_percent_alloy2 / 100 * weight_alloy2) / (weight_alloy1 + weight_alloy2) * 100 = chromium_percent_new ∧
    x = 12 :=
by sorry

end chromium_percent_alloy1_l291_29153


namespace size_relationship_l291_29122

theorem size_relationship (x : ℝ) : 
  let a := x^2 + x + Real.sqrt 2
  let b := Real.log 3 / Real.log 10
  let c := Real.exp (-1/2)
  b < c ∧ c < a := by sorry

end size_relationship_l291_29122


namespace fixed_point_parabola_l291_29160

theorem fixed_point_parabola (k : ℝ) : 
  225 = 9 * (5 : ℝ)^2 + k * 5 - 5 * k := by sorry

end fixed_point_parabola_l291_29160


namespace power_function_coefficient_l291_29188

/-- A function f is a power function if it has the form f(x) = ax^n, where a and n are constants and n ≠ 0 -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), n ≠ 0 ∧ ∀ x, f x = a * x ^ n

/-- If f(x) = (2m-1)x^3 is a power function, then m = 1 -/
theorem power_function_coefficient (m : ℝ) :
  IsPowerFunction (fun x => (2 * m - 1) * x ^ 3) → m = 1 :=
by
  sorry

end power_function_coefficient_l291_29188


namespace fraction_division_equals_seventeen_l291_29174

theorem fraction_division_equals_seventeen :
  (-4/9 + 1/6 - 2/3) / (-1/18) = 17 := by sorry

end fraction_division_equals_seventeen_l291_29174


namespace parabola_translation_l291_29168

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 3 0 0
  let translated := translate original 2 (-3)
  y = 3 * x^2 → y = 3 * (x - 2)^2 - 3 := by
  sorry

end parabola_translation_l291_29168


namespace committee_meeting_attendance_l291_29137

theorem committee_meeting_attendance :
  ∀ (assoc_prof asst_prof : ℕ),
  2 * assoc_prof + asst_prof = 11 →
  assoc_prof + 2 * asst_prof = 16 →
  assoc_prof + asst_prof = 9 :=
by
  sorry

end committee_meeting_attendance_l291_29137


namespace fraction_problem_l291_29167

theorem fraction_problem (p q : ℚ) : 
  q = 5 → 
  1/7 + (2*q - p)/(2*q + p) = 4/7 → 
  p = 4 := by
sorry

end fraction_problem_l291_29167


namespace percentage_calculation_l291_29146

theorem percentage_calculation (p : ℝ) : 
  (p / 100) * 2348 / 4.98 = 528.0642570281125 → 
  ∃ ε > 0, |p - 112| < ε :=
by
  sorry

end percentage_calculation_l291_29146


namespace evaluate_g_l291_29156

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 8

theorem evaluate_g : 3 * g 3 + 2 * g (-3) = 160 := by
  sorry

end evaluate_g_l291_29156


namespace fraction_inequality_solution_set_l291_29164

theorem fraction_inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  1 / x ≤ 1 ↔ x ∈ Set.Ioo 0 1 ∪ Set.Ici 1 := by sorry

end fraction_inequality_solution_set_l291_29164


namespace stratified_sampling_third_year_students_l291_29196

theorem stratified_sampling_third_year_students 
  (total_students : ℕ) 
  (third_year_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1600) 
  (h2 : third_year_students = 400) 
  (h3 : sample_size = 160) :
  (sample_size * third_year_students) / total_students = 40 :=
by sorry

end stratified_sampling_third_year_students_l291_29196


namespace walking_distance_l291_29118

theorem walking_distance (initial_speed : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) 
  (h1 : initial_speed = 12)
  (h2 : faster_speed = 16)
  (h3 : additional_distance = 20) :
  ∃ (actual_distance : ℝ) (time : ℝ),
    actual_distance = initial_speed * time ∧
    actual_distance + additional_distance = faster_speed * time ∧
    actual_distance = 60 := by
  sorry

end walking_distance_l291_29118


namespace complex_modulus_one_l291_29199

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_modulus_one_l291_29199


namespace simplify_and_evaluate_l291_29144

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) :
  (a + 2 + 4 / (a - 2)) / (a^3 / (a^2 - 4*a + 4)) = 1/3 := by
  sorry

end simplify_and_evaluate_l291_29144


namespace square_condition_l291_29179

theorem square_condition (n : ℕ) : 
  (∃ k : ℕ, (n^3 + 39*n - 2)*n.factorial + 17*21^n + 5 = k^2) ↔ n = 1 :=
sorry

end square_condition_l291_29179


namespace positive_numbers_l291_29151

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : b * c + c * a + a * b > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end positive_numbers_l291_29151


namespace vertex_locus_is_parabola_l291_29180

/-- The locus of vertices of a family of parabolas forms another parabola -/
theorem vertex_locus_is_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  ∃ (A B C : ℝ), A ≠ 0 ∧
    ∀ (x y : ℝ), (∃ t : ℝ, x = -t / (2 * a) ∧ y = c - t^2 / (4 * a)) ↔
      y = A * x^2 + B * x + C :=
by sorry

end vertex_locus_is_parabola_l291_29180


namespace three_digit_prime_integers_count_l291_29145

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := sorry

-- Define a function to get all single-digit prime numbers
def singleDigitPrimes : List ℕ := sorry

-- Define a function to count the number of three-digit positive integers
-- where the digits are three different prime numbers
def countThreeDigitPrimeIntegers : ℕ := sorry

-- Theorem statement
theorem three_digit_prime_integers_count :
  countThreeDigitPrimeIntegers = 24 := by sorry

end three_digit_prime_integers_count_l291_29145


namespace inscribed_square_area_ratio_l291_29106

theorem inscribed_square_area_ratio : 
  ∀ (large_square_side : ℝ) (inscribed_square_side : ℝ),
    large_square_side = 4 →
    inscribed_square_side = 2 →
    (inscribed_square_side^2) / (large_square_side^2) = 1/4 := by
  sorry

end inscribed_square_area_ratio_l291_29106


namespace reciprocal_sum_equality_l291_29123

theorem reciprocal_sum_equality (a b c : ℝ) (n : ℕ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : 1 / a + 1 / b + 1 / c = 1 / (a + b + c)) : 
  1 / a^(2*n+1) + 1 / b^(2*n+1) + 1 / c^(2*n+1) = 
  1 / (a^(2*n+1) + b^(2*n+1) + c^(2*n+1)) := by
sorry

end reciprocal_sum_equality_l291_29123


namespace laser_reflection_distance_l291_29176

def laser_path_distance : ℝ → ℝ → ℝ → ℝ → ℝ := sorry

theorem laser_reflection_distance :
  let start_x : ℝ := 2
  let start_y : ℝ := 4
  let end_x : ℝ := 10
  let end_y : ℝ := 4
  laser_path_distance start_x start_y end_x end_y = 6 + 2 * Real.sqrt 29 := by
  sorry

end laser_reflection_distance_l291_29176


namespace fencing_cost_theorem_l291_29133

/-- Represents a rectangular park with given dimensions and fencing cost -/
structure RectangularPark where
  ratio : Rat  -- Ratio of longer side to shorter side
  area : ℝ     -- Area in square meters
  fencingCost : ℝ  -- Cost of fencing per meter in paise

/-- Calculates the cost of fencing a rectangular park -/
def calculateFencingCost (park : RectangularPark) : ℝ :=
  sorry

/-- Theorem: The cost of fencing the given park is 175 rupees -/
theorem fencing_cost_theorem (park : RectangularPark) 
  (h1 : park.ratio = 3/2)
  (h2 : park.area = 7350)
  (h3 : park.fencingCost = 50) : 
  calculateFencingCost park = 175 := by
  sorry

end fencing_cost_theorem_l291_29133


namespace chime_2023_date_l291_29186

/-- Represents a date with year, month, and day -/
structure Date :=
  (year : Nat) (month : Nat) (day : Nat)

/-- Represents a time with hour and minute -/
structure Time :=
  (hour : Nat) (minute : Nat)

/-- Calculates the number of chimes for a given hour -/
def chimes_for_hour (hour : Nat) : Nat :=
  if hour ≤ 12 then hour else hour - 12

/-- Calculates the total number of chimes in a day with the malfunction -/
def daily_chimes : Nat := 101

/-- Calculates the number of chimes from a given start time to midnight -/
def chimes_until_midnight (start_time : Time) : Nat :=
  sorry -- Implementation details omitted

/-- Calculates the date of the nth chime given a start date and time -/
def date_of_nth_chime (start_date : Date) (start_time : Time) (n : Nat) : Date :=
  sorry -- Implementation details omitted

theorem chime_2023_date :
  let start_date := Date.mk 2003 2 26
  let start_time := Time.mk 14 15
  date_of_nth_chime start_date start_time 2023 = Date.mk 2003 3 18 := by
  sorry

end chime_2023_date_l291_29186


namespace adam_apples_solution_l291_29154

/-- Adam's apple purchases over three days --/
def adam_apples (monday_quantity : ℕ) (tuesday_multiple : ℕ) (wednesday_multiple : ℕ) : Prop :=
  let tuesday_quantity := monday_quantity * tuesday_multiple
  let wednesday_quantity := tuesday_quantity * wednesday_multiple
  monday_quantity + tuesday_quantity + wednesday_quantity = 240

theorem adam_apples_solution :
  ∃ (wednesday_multiple : ℕ),
    adam_apples 15 3 wednesday_multiple ∧ wednesday_multiple = 4 := by
  sorry

end adam_apples_solution_l291_29154


namespace expression_evaluation_l291_29103

theorem expression_evaluation :
  let x : ℚ := 1/3
  let y : ℚ := -1/2
  5 * x^2 - 2 * (3 * y^2 + 6 * x * y) - (2 * x^2 - 6 * y^2) = 7/3 := by
  sorry

end expression_evaluation_l291_29103


namespace misha_grade_size_l291_29108

/-- The number of students in Misha's grade -/
def num_students : ℕ := 149

/-- Misha's position from the top of the grade -/
def position_from_top : ℕ := 75

/-- Misha's position from the bottom of the grade -/
def position_from_bottom : ℕ := 75

/-- Theorem: Given Misha's positions from top and bottom, prove the number of students in her grade -/
theorem misha_grade_size :
  position_from_top + position_from_bottom - 1 = num_students :=
by sorry

end misha_grade_size_l291_29108


namespace word_arrangements_l291_29170

/-- The number of distinct letters in the word -/
def n : ℕ := 6

/-- The number of units to be arranged after combining the T's -/
def k : ℕ := 5

/-- The number of ways to arrange the T's within their unit -/
def t : ℕ := 2

/-- The total number of arrangements -/
def total_arrangements : ℕ := k.factorial * t.factorial

theorem word_arrangements : total_arrangements = 240 := by
  sorry

end word_arrangements_l291_29170


namespace equation_proof_l291_29127

theorem equation_proof : Real.sqrt (72 * 2) + (5568 / 87) ^ (1/3) = Real.sqrt 256 := by
  sorry

end equation_proof_l291_29127


namespace instrument_probability_l291_29148

theorem instrument_probability (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 →
  at_least_one = 2 / 5 →
  two_or_more = 96 →
  (((at_least_one * total) - two_or_more) / total : ℚ) = 28 / 100 := by
  sorry

end instrument_probability_l291_29148


namespace sam_found_35_seashells_l291_29109

/-- The number of seashells Joan found -/
def joans_seashells : ℕ := 18

/-- The total number of seashells Sam and Joan found together -/
def total_seashells : ℕ := 53

/-- The number of seashells Sam found -/
def sams_seashells : ℕ := total_seashells - joans_seashells

theorem sam_found_35_seashells : sams_seashells = 35 := by
  sorry

end sam_found_35_seashells_l291_29109


namespace angle_xpy_is_45_deg_l291_29105

/-- A rectangle WXYZ with a point P on side WZ -/
structure RectangleWithPoint where
  /-- Length of side WZ -/
  wz : ℝ
  /-- Length of side XY -/
  xy : ℝ
  /-- Distance from W to P -/
  wp : ℝ
  /-- Angle WPY in radians -/
  angle_wpy : ℝ
  /-- Angle XPY in radians -/
  angle_xpy : ℝ
  /-- WZ is positive -/
  wz_pos : 0 < wz
  /-- XY is positive -/
  xy_pos : 0 < xy
  /-- P is on WZ -/
  wp_le_wz : 0 ≤ wp ∧ wp ≤ wz
  /-- Sine ratio condition -/
  sine_ratio : Real.sin angle_wpy / Real.sin angle_xpy = 2

/-- Theorem: If WZ = 8, XY = 4, and the sine ratio condition holds, then ∠XPY = 45° -/
theorem angle_xpy_is_45_deg (r : RectangleWithPoint) 
  (h1 : r.wz = 8) (h2 : r.xy = 4) : r.angle_xpy = π/4 := by
  sorry

end angle_xpy_is_45_deg_l291_29105


namespace problem_solution_l291_29159

theorem problem_solution (x y : ℚ) : 
  x = 51 → x^3 * y - 3 * x^2 * y + 2 * x * y = 122650 → y = 1/2 := by
  sorry

end problem_solution_l291_29159


namespace bombardment_death_percentage_l291_29128

/-- Represents the percentage of people who died by bombardment -/
def bombardment_percentage : ℝ := 10

/-- The initial population of the village -/
def initial_population : ℕ := 4200

/-- The final population after bombardment and departure -/
def final_population : ℕ := 3213

/-- The percentage of people who left after the bombardment -/
def departure_percentage : ℝ := 15

theorem bombardment_death_percentage :
  let remaining_after_bombardment := initial_population - (bombardment_percentage / 100) * initial_population
  let departed := (departure_percentage / 100) * remaining_after_bombardment
  initial_population - (bombardment_percentage / 100) * initial_population - departed = final_population :=
by sorry

end bombardment_death_percentage_l291_29128


namespace unique_solution_to_system_l291_29110

theorem unique_solution_to_system :
  ∃! (x y z : ℝ), 
    x^2 - 23*y + 66*z + 612 = 0 ∧
    y^2 + 62*x - 20*z + 296 = 0 ∧
    z^2 - 22*x + 67*y + 505 = 0 ∧
    x = -20 ∧ y = -22 ∧ z = -23 := by
  sorry

end unique_solution_to_system_l291_29110


namespace pizza_toppings_l291_29119

theorem pizza_toppings (total_slices pepperoni_slices mushroom_slices : ℕ) 
  (h1 : total_slices = 16)
  (h2 : pepperoni_slices = 8)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ 
     slice ∈ Finset.range mushroom_slices)) :
  ∃ both : ℕ, both = pepperoni_slices + mushroom_slices - total_slices :=
by sorry

end pizza_toppings_l291_29119


namespace correct_proposition_l291_29101

/-- Proposition p: The solution set of ax^2 + ax + 1 > 0 is ℝ, then a ∈ (0,4) -/
def p : Prop := ∀ x : ℝ, (∃ a : ℝ, a * x^2 + a * x + 1 > 0) → (∃ a : ℝ, 0 < a ∧ a < 4)

/-- Proposition q: "x^2 - 2x - 8 > 0" is a necessary but not sufficient condition for "x > 5" -/
def q : Prop := (∀ x : ℝ, x > 5 → x^2 - 2*x - 8 > 0) ∧ (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≤ 5)

/-- The correct proposition is (¬p) ∧ q -/
theorem correct_proposition : (¬p) ∧ q := by sorry

end correct_proposition_l291_29101


namespace hyperbola_eccentricity_l291_29157

/-- Given a hyperbola C and a circle F with specific properties, prove that the eccentricity of C is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let F := {(x, y) : ℝ × ℝ | (x - c)^2 + y^2 = c^2}
  let l := {(x, y) : ℝ × ℝ | y = -(a / b) * (x - 2 * a / 3)}
  ∃ (chord_length : ℝ), 
    (∀ (p q : ℝ × ℝ), p ∈ F ∧ q ∈ F ∧ p ∈ l ∧ q ∈ l → ‖p - q‖ = chord_length) ∧
    chord_length = 4 * Real.sqrt 2 * c / 3 →
  c / a = 2 := by
sorry

end hyperbola_eccentricity_l291_29157


namespace sales_growth_rate_l291_29112

theorem sales_growth_rate (x : ℝ) : (1 + x)^2 = 1 + 0.44 → x < 0.22 := by
  sorry

end sales_growth_rate_l291_29112


namespace difference_of_41st_terms_l291_29126

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem difference_of_41st_terms : 
  let C := arithmetic_sequence 50 15
  let D := arithmetic_sequence 50 (-15)
  |C 41 - D 41| = 1200 := by sorry

end difference_of_41st_terms_l291_29126


namespace function_equality_l291_29187

theorem function_equality (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (2 * x + f y) = x + y + f x) → 
  (∀ x : ℝ, f x = x) := by
sorry

end function_equality_l291_29187


namespace factorization_equality_l291_29162

theorem factorization_equality (a b : ℝ) : 
  a^2 - 4*b^2 - 2*a + 4*b = (a + 2*b - 2) * (a - 2*b) := by
  sorry

end factorization_equality_l291_29162


namespace odot_commutative_odot_no_identity_odot_associativity_undetermined_l291_29166

-- Define the binary operation
def odot (x y : ℝ) : ℝ := 2 * (x + 2) * (y + 2) - 3

-- Theorem for commutativity
theorem odot_commutative : ∀ x y : ℝ, odot x y = odot y x := by sorry

-- Theorem for non-existence of identity element
theorem odot_no_identity : ¬ ∃ e : ℝ, ∀ x : ℝ, odot x e = x ∧ odot e x = x := by sorry

-- Theorem for undetermined associativity
theorem odot_associativity_undetermined : 
  ¬ (∀ x y z : ℝ, odot (odot x y) z = odot x (odot y z)) ∧ 
  ¬ (∃ x y z : ℝ, odot (odot x y) z ≠ odot x (odot y z)) := by sorry

end odot_commutative_odot_no_identity_odot_associativity_undetermined_l291_29166


namespace lisa_flight_distance_l291_29111

/-- Given a speed of 32 miles per hour and a time of 8 hours, 
    the distance traveled is equal to 256 miles. -/
theorem lisa_flight_distance : 
  let speed : ℝ := 32
  let time : ℝ := 8
  let distance := speed * time
  distance = 256 := by sorry

end lisa_flight_distance_l291_29111


namespace horner_method_v2_l291_29197

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 6*x + 7

def horner_v2 (a b c d e f x : ℝ) : ℝ :=
  ((a * x + b) * x + c) * x + d

theorem horner_method_v2 :
  horner_v2 2 (-5) (-4) 3 (-6) 7 5 = 21 :=
by
  sorry

end horner_method_v2_l291_29197


namespace tan_inequality_l291_29107

open Real

theorem tan_inequality (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁ ∧ x₁ < π/2) 
  (h₂ : 0 < x₂ ∧ x₂ < π/2) 
  (h₃ : x₁ ≠ x₂) : 
  (1/2) * (tan x₁ + tan x₂) > tan ((x₁ + x₂)/2) := by
  sorry

end tan_inequality_l291_29107


namespace double_inequality_l291_29116

theorem double_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (0 < 1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1))) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) ≤ 1 / 8) ∧
  (1 / (x + y + z + 1) - 1 / ((x + 1) * (y + 1) * (z + 1)) = 1 / 8 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end double_inequality_l291_29116


namespace businessmen_drinks_l291_29177

theorem businessmen_drinks (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ) :
  total = 30 →
  coffee = 15 →
  tea = 13 →
  both = 6 →
  total - (coffee + tea - both) = 8 := by
  sorry

end businessmen_drinks_l291_29177


namespace length_AC_l291_29171

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 49}

structure PointsOnCircle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A ∈ Circle
  h_B : B ∈ Circle
  h_AB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64
  h_C : C ∈ Circle
  h_C_midpoint : C.1 = (A.1 + B.1) / 2 ∧ C.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem length_AC (points : PointsOnCircle) :
  (points.A.1 - points.C.1)^2 + (points.A.2 - points.C.2)^2 = 98 - 14 * Real.sqrt 33 := by
  sorry

end length_AC_l291_29171


namespace yellow_light_probability_is_one_twelfth_l291_29191

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the probability of seeing the yellow light -/
def yellowLightProbability (d : TrafficLightDuration) : ℚ :=
  d.yellow / (d.red + d.green + d.yellow)

/-- Theorem stating the probability of seeing the yellow light is 1/12 -/
theorem yellow_light_probability_is_one_twelfth :
  let d : TrafficLightDuration := ⟨30, 25, 5⟩
  yellowLightProbability d = 1 / 12 := by
  sorry

#check yellow_light_probability_is_one_twelfth

end yellow_light_probability_is_one_twelfth_l291_29191


namespace min_value_sin_cos_cubic_min_value_achievable_l291_29189

theorem min_value_sin_cos_cubic (x : ℝ) : 
  Real.sin x ^ 3 + 2 * Real.cos x ^ 3 ≥ -4 * Real.sqrt 2 / 3 :=
sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sin x ^ 3 + 2 * Real.cos x ^ 3 = -4 * Real.sqrt 2 / 3 :=
sorry

end min_value_sin_cos_cubic_min_value_achievable_l291_29189


namespace ellipse_axis_endpoint_distance_l291_29161

/-- Given an ellipse with equation 4(x-2)^2 + 16y^2 = 64, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√5. -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (C D : ℝ × ℝ),
    (∀ (x y : ℝ), 4 * (x - 2)^2 + 16 * y^2 = 64 → 
      ((x = C.1 ∧ y = C.2) ∨ (x = D.1 ∧ y = D.2))) →
    (C.1 - 2)^2 / 16 + C.2^2 / 4 = 1 →
    (D.1 - 2)^2 / 16 + D.2^2 / 4 = 1 →
    C.1 ≠ D.1 →
    C.2 ≠ D.2 →
    Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end ellipse_axis_endpoint_distance_l291_29161


namespace square_of_complex_number_l291_29184

theorem square_of_complex_number : 
  let z : ℂ := 1 - 2*I
  z^2 = -3 - 4*I :=
by sorry

end square_of_complex_number_l291_29184


namespace managers_salary_l291_29193

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) : 
  num_employees = 20 → 
  avg_salary = 1500 → 
  avg_increase = 1000 → 
  (num_employees * avg_salary + (num_employees + 1) * avg_increase) / (num_employees + 1) - avg_salary = 22500 :=
by sorry

end managers_salary_l291_29193


namespace edward_initial_amount_l291_29124

def initial_amount (book_price shirt_price shirt_discount meal_price
                    ticket_price ticket_discount amount_left : ℝ) : ℝ :=
  book_price +
  (shirt_price * (1 - shirt_discount)) +
  meal_price +
  (ticket_price - ticket_discount) +
  amount_left

theorem edward_initial_amount :
  initial_amount 9 25 0.2 15 10 2 17 = 69 := by
  sorry

end edward_initial_amount_l291_29124


namespace ivy_room_spiders_l291_29158

/-- Given the total number of spider legs in a room, calculate the number of spiders. -/
def spiders_in_room (total_legs : ℕ) : ℕ :=
  total_legs / 8

/-- Theorem: There are 4 spiders in Ivy's room given 32 total spider legs. -/
theorem ivy_room_spiders : spiders_in_room 32 = 4 := by
  sorry

end ivy_room_spiders_l291_29158


namespace master_zhang_apple_sales_l291_29172

/-- The number of apples Master Zhang must sell to make a profit of 15 yuan -/
def apples_to_sell : ℕ := 100

/-- The buying price in yuan per apple -/
def buying_price : ℚ := 1 / 4

/-- The selling price in yuan per apple -/
def selling_price : ℚ := 2 / 5

/-- The desired profit in yuan -/
def desired_profit : ℕ := 15

theorem master_zhang_apple_sales :
  apples_to_sell = (desired_profit : ℚ) / (selling_price - buying_price) := by sorry

end master_zhang_apple_sales_l291_29172


namespace square_sum_and_reciprocal_l291_29120

theorem square_sum_and_reciprocal (x : ℝ) (h : x + (1/x) = 2) : x^2 + (1/x^2) = 2 := by
  sorry

end square_sum_and_reciprocal_l291_29120


namespace complex_expression_value_l291_29183

theorem complex_expression_value : 
  (10 - (10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)))) * 20 = 192.6 := by
  sorry

end complex_expression_value_l291_29183


namespace multiplication_mistake_correction_l291_29194

theorem multiplication_mistake_correction (α : ℝ) :
  1.2 * α = 1.23 * α - 0.3 → 1.23 * α = 111 := by
sorry

end multiplication_mistake_correction_l291_29194


namespace blue_spotted_fish_count_l291_29138

theorem blue_spotted_fish_count (total_fish : ℕ) (blue_percentage : ℚ) (spotted_fraction : ℚ) : 
  total_fish = 150 →
  blue_percentage = 2/5 →
  spotted_fraction = 3/5 →
  (total_fish : ℚ) * blue_percentage * spotted_fraction = 36 := by
sorry

end blue_spotted_fish_count_l291_29138
