import Mathlib

namespace distance_circle_center_to_line_l2982_298270

/-- The distance from the center of the circle ρ = 4cos θ to the line tan θ = 1 is √2 -/
theorem distance_circle_center_to_line : 
  ∀ (θ : ℝ) (ρ : ℝ → ℝ) (x y : ℝ),
  (ρ θ = 4 * Real.cos θ) →  -- Circle equation
  (Real.tan θ = 1) →        -- Line equation
  (x - 2)^2 + y^2 = 4 →     -- Standard form of circle equation
  x - y = 0 →               -- Line equation in rectangular coordinates
  Real.sqrt 2 = |x - 2| / Real.sqrt ((1:ℝ)^2 + (-1:ℝ)^2) :=
by sorry

end distance_circle_center_to_line_l2982_298270


namespace science_to_novel_ratio_l2982_298264

/-- Given the page counts of different books, prove the ratio of science to novel pages --/
theorem science_to_novel_ratio :
  let history_pages : ℕ := 300
  let science_pages : ℕ := 600
  let novel_pages : ℕ := history_pages / 2
  science_pages / novel_pages = 4 := by
  sorry


end science_to_novel_ratio_l2982_298264


namespace smallest_k_no_real_roots_l2982_298208

theorem smallest_k_no_real_roots : 
  let f (k : ℤ) (x : ℝ) := 3 * x * (k * x - 5) - x^2 + 4
  ∀ k : ℤ, (∀ x : ℝ, f k x ≠ 0) → k ≥ 6 :=
by sorry

end smallest_k_no_real_roots_l2982_298208


namespace blocks_remaining_l2982_298200

theorem blocks_remaining (initial_blocks : ℕ) (first_tower : ℕ) (second_tower : ℕ)
  (h1 : initial_blocks = 78)
  (h2 : first_tower = 19)
  (h3 : second_tower = 25) :
  initial_blocks - first_tower - second_tower = 34 := by
  sorry

end blocks_remaining_l2982_298200


namespace fraction_equality_l2982_298227

theorem fraction_equality (x y : ℚ) (hx : x = 4/7) (hy : y = 8/11) : 
  (7*x + 11*y) / (49*x*y) = 231/56 := by
  sorry

end fraction_equality_l2982_298227


namespace khali_snow_volume_l2982_298236

/-- Calculates the total volume of snow on a rectangular sidewalk with two layers -/
def total_snow_volume (length width depth1 depth2 : ℝ) : ℝ :=
  length * width * (depth1 + depth2)

/-- Theorem: The total volume of snow on Khali's sidewalk is 90 cubic feet -/
theorem khali_snow_volume :
  let length : ℝ := 30
  let width : ℝ := 3
  let depth1 : ℝ := 0.6
  let depth2 : ℝ := 0.4
  total_snow_volume length width depth1 depth2 = 90 := by
  sorry

#eval total_snow_volume 30 3 0.6 0.4

end khali_snow_volume_l2982_298236


namespace percentage_non_defective_m3_l2982_298292

theorem percentage_non_defective_m3 (m1_percentage : Real) (m2_percentage : Real)
  (m1_defective : Real) (m2_defective : Real) (total_defective : Real) :
  m1_percentage = 0.4 →
  m2_percentage = 0.3 →
  m1_defective = 0.03 →
  m2_defective = 0.01 →
  total_defective = 0.036 →
  ∃ (m3_non_defective : Real),
    m3_non_defective = 0.93 ∧
    m1_percentage * m1_defective + m2_percentage * m2_defective +
    (1 - m1_percentage - m2_percentage) * (1 - m3_non_defective) = total_defective :=
by sorry

end percentage_non_defective_m3_l2982_298292


namespace farmer_brown_additional_cost_l2982_298251

/-- The additional cost for Farmer Brown's new hay requirements -/
theorem farmer_brown_additional_cost :
  let original_bales : ℕ := 10
  let original_cost_per_bale : ℕ := 15
  let new_bales : ℕ := 2 * original_bales
  let new_cost_per_bale : ℕ := 18
  (new_bales * new_cost_per_bale) - (original_bales * original_cost_per_bale) = 210 :=
by sorry

end farmer_brown_additional_cost_l2982_298251


namespace total_time_two_trips_l2982_298275

/-- Represents the time in minutes for a round trip to the beauty parlor -/
structure RoundTrip where
  to_parlor : ℕ
  from_parlor : ℕ
  delay : ℕ
  additional_time : ℕ

/-- Calculates the total time for a round trip -/
def total_time (trip : RoundTrip) : ℕ :=
  trip.to_parlor + trip.from_parlor + trip.delay + trip.additional_time

/-- Represents Naomi's two round trips to the beauty parlor -/
def naomi_trips : (RoundTrip × RoundTrip) :=
  ({ to_parlor := 60
   , from_parlor := 120
   , delay := 15
   , additional_time := 10 }
  ,{ to_parlor := 60
   , from_parlor := 120
   , delay := 20
   , additional_time := 30 })

/-- Theorem stating that the total time for both round trips is 435 minutes -/
theorem total_time_two_trips : 
  total_time naomi_trips.1 + total_time naomi_trips.2 = 435 := by
  sorry

end total_time_two_trips_l2982_298275


namespace add_2405_minutes_to_midnight_l2982_298260

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := (totalMinutes / 60) % 24, minutes := totalMinutes % 60 }

-- Theorem statement
theorem add_2405_minutes_to_midnight :
  addMinutes { hours := 0, minutes := 0 } 2405 = { hours := 16, minutes := 5 } := by
  sorry

end add_2405_minutes_to_midnight_l2982_298260


namespace triangle_problem_l2982_298297

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0)
  (h2 : t.c = 2)
  (h3 : t.a + t.b = t.a * t.b) :
  t.C = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end triangle_problem_l2982_298297


namespace binomial_divisibility_l2982_298204

theorem binomial_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℤ, (Nat.choose (2*p - 1) (p - 1) : ℤ) - 1 = k * p^3 := by
  sorry

end binomial_divisibility_l2982_298204


namespace smallest_integer_with_remainders_l2982_298212

theorem smallest_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 6 = 5) ∧ 
  (a % 8 = 7) ∧ 
  (∀ b : ℕ, b > 0 → b % 6 = 5 → b % 8 = 7 → a ≤ b) ∧
  (a = 23) := by
sorry

end smallest_integer_with_remainders_l2982_298212


namespace product_modulo_seven_l2982_298209

theorem product_modulo_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end product_modulo_seven_l2982_298209


namespace quadratic_equation_transformation_l2982_298244

theorem quadratic_equation_transformation (p q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -3 ∧ ∀ x : ℝ, x^2 + p*x + q = 0 ↔ (x = x₁ ∨ x = x₂)) →
  ∀ x : ℝ, x^2 + p*x + q = 0 ↔ (x - 2)*(x + 3) = 0 :=
by sorry

end quadratic_equation_transformation_l2982_298244


namespace first_grade_enrollment_proof_l2982_298210

theorem first_grade_enrollment_proof :
  ∃! a : ℕ,
    200 ≤ a ∧ a ≤ 300 ∧
    (∃ R : ℕ, a = 25 * R + 10) ∧
    (∃ L : ℕ, a = 30 * L - 15) ∧
    a = 285 := by
  sorry

end first_grade_enrollment_proof_l2982_298210


namespace least_positive_y_l2982_298237

-- Define variables
variable (c d : ℝ)
variable (y : ℝ)

-- Define the conditions
def condition1 : Prop := Real.tan y = (2 * c) / (3 * d)
def condition2 : Prop := Real.tan (2 * y) = (3 * d) / (2 * c + 3 * d)

-- State the theorem
theorem least_positive_y (h1 : condition1 c d y) (h2 : condition2 c d y) :
  y = Real.arctan (1 / 3) ∧ ∀ z, 0 < z ∧ z < y → ¬(condition1 c d z ∧ condition2 c d z) :=
sorry

end least_positive_y_l2982_298237


namespace car_length_is_113_steps_l2982_298238

/-- Represents the scenario of a person jogging alongside a moving car --/
structure CarJoggingScenario where
  /-- The length of the car in terms of the jogger's steps --/
  car_length : ℝ
  /-- The distance the car moves during one step of the jogger --/
  car_step : ℝ
  /-- The number of steps counted when jogging from rear to front --/
  steps_rear_to_front : ℕ
  /-- The number of steps counted when jogging from front to rear --/
  steps_front_to_rear : ℕ
  /-- The car is moving faster than the jogger --/
  car_faster : car_step > 0
  /-- The car has a positive length --/
  car_positive : car_length > 0
  /-- Equation for jogging from rear to front --/
  eq_rear_to_front : (steps_rear_to_front : ℝ) = car_length / car_step + steps_rear_to_front
  /-- Equation for jogging from front to rear --/
  eq_front_to_rear : (steps_front_to_rear : ℝ) = car_length / car_step - steps_front_to_rear

/-- The length of the car is 113 steps when jogging 150 steps rear to front and 30 steps front to rear --/
theorem car_length_is_113_steps (scenario : CarJoggingScenario) 
  (h1 : scenario.steps_rear_to_front = 150) 
  (h2 : scenario.steps_front_to_rear = 30) : 
  scenario.car_length = 113 := by
  sorry

end car_length_is_113_steps_l2982_298238


namespace parabola_coefficients_l2982_298295

/-- A parabola with vertex (4, 3), vertical axis of symmetry, passing through (2, 1) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a * 4^2 + b * 4 + c = 3
  symmetry : b = -2 * a * 4
  point : a * 2^2 + b * 2 + c = 1

/-- The coefficients of the parabola are (-1/2, 4, -5) -/
theorem parabola_coefficients (p : Parabola) : p.a = -1/2 ∧ p.b = 4 ∧ p.c = -5 := by
  sorry

end parabola_coefficients_l2982_298295


namespace remaining_tickets_l2982_298231

/-- Given an initial number of tickets, the number of tickets lost, and the number of tickets spent,
    the remaining number of tickets is equal to the initial number minus the lost tickets minus the spent tickets. -/
theorem remaining_tickets (initial lost spent : ℝ) : 
  initial - lost - spent = initial - (lost + spent) := by
  sorry

end remaining_tickets_l2982_298231


namespace josh_shopping_spending_l2982_298265

/-- The problem of calculating Josh's total spending at the shopping center -/
theorem josh_shopping_spending :
  let num_films : ℕ := 9
  let num_books : ℕ := 4
  let num_cds : ℕ := 6
  let cost_per_film : ℕ := 5
  let cost_per_book : ℕ := 4
  let cost_per_cd : ℕ := 3
  let total_spent := 
    num_films * cost_per_film + 
    num_books * cost_per_book + 
    num_cds * cost_per_cd
  total_spent = 79 := by
  sorry

end josh_shopping_spending_l2982_298265


namespace division_remainder_proof_l2982_298223

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 997)
  (h2 : divisor = 23)
  (h3 : quotient = 43)
  (h4 : dividend = divisor * quotient + remainder) :
  remainder = 8 := by
sorry

end division_remainder_proof_l2982_298223


namespace sqrt_eight_div_sqrt_two_equals_two_l2982_298266

theorem sqrt_eight_div_sqrt_two_equals_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end sqrt_eight_div_sqrt_two_equals_two_l2982_298266


namespace inequality_not_true_range_l2982_298248

theorem inequality_not_true_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - a| + |x - 12| < 6)) ↔ (a ≤ 6 ∨ a ≥ 18) := by
  sorry

end inequality_not_true_range_l2982_298248


namespace derivative_at_two_l2982_298278

theorem derivative_at_two (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 3 * (deriv f 2) * x) : 
  deriv f 2 = -2 := by
  sorry

end derivative_at_two_l2982_298278


namespace marias_gum_count_l2982_298273

/-- 
Given:
- Maria initially had 25 pieces of gum
- Tommy gave her 16 more pieces
- Luis gave her 20 more pieces

Prove that Maria now has 61 pieces of gum
-/
theorem marias_gum_count (initial : ℕ) (tommy : ℕ) (luis : ℕ) 
  (h1 : initial = 25)
  (h2 : tommy = 16)
  (h3 : luis = 20) :
  initial + tommy + luis = 61 := by
  sorry

end marias_gum_count_l2982_298273


namespace parallel_vectors_l2982_298249

/-- Two vectors in R² are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- Given vectors a and b, if they are parallel, then the x-component of a is 1/2 -/
theorem parallel_vectors (a b : ℝ × ℝ) (h : a.2 = 1 ∧ b = (2, 4)) :
  parallel a b → a.1 = 1/2 := by
  sorry

end parallel_vectors_l2982_298249


namespace twenty_percent_greater_than_forty_l2982_298291

/-- If x is 20 percent greater than 40, then x equals 48. -/
theorem twenty_percent_greater_than_forty (x : ℝ) : x = 40 * (1 + 0.2) → x = 48 := by
  sorry

end twenty_percent_greater_than_forty_l2982_298291


namespace first_number_is_30_l2982_298202

def fibonacci_like_sequence (a₁ a₂ : ℤ) : ℕ → ℤ
  | 0 => a₁
  | 1 => a₂
  | (n+2) => fibonacci_like_sequence a₁ a₂ n + fibonacci_like_sequence a₁ a₂ (n+1)

theorem first_number_is_30 (a₁ a₂ : ℤ) :
  fibonacci_like_sequence a₁ a₂ 6 = 5 ∧
  fibonacci_like_sequence a₁ a₂ 7 = 14 ∧
  fibonacci_like_sequence a₁ a₂ 8 = 33 →
  a₁ = 30 := by
sorry

end first_number_is_30_l2982_298202


namespace profit_increase_l2982_298283

theorem profit_increase (cost_price selling_price : ℝ) (a : ℝ) 
  (h1 : selling_price - cost_price = cost_price * (a / 100))
  (h2 : selling_price - (cost_price * 0.95) = (cost_price * 0.95) * ((a + 15) / 100)) :
  a = 185 := by
  sorry

end profit_increase_l2982_298283


namespace stratified_sampling_third_grade_l2982_298239

def total_students : ℕ := 270000
def third_grade_students : ℕ := 81000
def sample_size : ℕ := 3000

theorem stratified_sampling_third_grade :
  (third_grade_students * sample_size) / total_students = 900 := by
  sorry

end stratified_sampling_third_grade_l2982_298239


namespace special_calculator_problem_l2982_298203

-- Define a function to reverse digits of a number
def reverse_digits (n : ℕ) : ℕ := sorry

-- Define the calculator operation
def calculator_operation (x : ℕ) : ℕ := reverse_digits (2 * x) + 2

-- Theorem statement
theorem special_calculator_problem (x : ℕ) :
  x ≥ 10 ∧ x < 100 →  -- two-digit number condition
  calculator_operation x = 45 →
  x = 17 := by sorry

end special_calculator_problem_l2982_298203


namespace complex_modulus_l2982_298240

theorem complex_modulus (a b : ℝ) (h : b^2 + (4 + Complex.I) * b + 4 + a * Complex.I = 0) :
  Complex.abs (a + b * Complex.I) = 2 * Real.sqrt 2 := by
  sorry

end complex_modulus_l2982_298240


namespace museum_artifacts_per_wing_l2982_298230

/-- Represents a museum with paintings and artifacts -/
structure Museum where
  total_wings : ℕ
  painting_wings : ℕ
  large_paintings : ℕ
  small_painting_wings : ℕ
  small_paintings_per_wing : ℕ
  artifact_multiplier : ℕ

/-- Calculates the number of artifacts in each artifact wing -/
def artifacts_per_wing (m : Museum) : ℕ :=
  let total_paintings := m.large_paintings + m.small_painting_wings * m.small_paintings_per_wing
  let total_artifacts := total_paintings * m.artifact_multiplier
  let artifact_wings := m.total_wings - m.painting_wings
  (total_artifacts + artifact_wings - 1) / artifact_wings

/-- Theorem stating the number of artifacts in each artifact wing for the given museum -/
theorem museum_artifacts_per_wing :
  let m : Museum := {
    total_wings := 16,
    painting_wings := 6,
    large_paintings := 2,
    small_painting_wings := 4,
    small_paintings_per_wing := 20,
    artifact_multiplier := 8
  }
  artifacts_per_wing m = 66 := by sorry

end museum_artifacts_per_wing_l2982_298230


namespace spot_difference_l2982_298290

theorem spot_difference (granger cisco rover : ℕ) : 
  granger = 5 * cisco →
  granger + cisco = 108 →
  rover = 46 →
  cisco < rover / 2 →
  rover / 2 - cisco = 5 :=
by
  sorry

end spot_difference_l2982_298290


namespace rhombus_area_scaling_l2982_298221

theorem rhombus_area_scaling (d1 d2 : ℝ) :
  d1 > 0 → d2 > 0 → (d1 * d2) / 2 = 3 → ((5 * d1) * (5 * d2)) / 2 = 75 := by
  sorry

end rhombus_area_scaling_l2982_298221


namespace angle_C_measure_l2982_298253

-- Define a scalene triangle ABC
structure ScaleneTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  scalene : A ≠ B ∧ B ≠ C ∧ A ≠ C
  sum_180 : A + B + C = 180

-- Theorem statement
theorem angle_C_measure (t : ScaleneTriangle) 
  (h1 : t.B = t.A + 20)  -- Angle B is 20 degrees larger than angle A
  (h2 : t.C = 2 * t.A)   -- Angle C is twice the size of angle A
  : t.C = 80 := by
  sorry

end angle_C_measure_l2982_298253


namespace ball_probability_l2982_298233

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ)
  (h_total : total = 60)
  (h_white : white = 22)
  (h_green : green = 10)
  (h_yellow : yellow = 7)
  (h_red : red = 15)
  (h_purple : purple = 6)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 13 / 20 := by
sorry

end ball_probability_l2982_298233


namespace price_comparison_l2982_298245

theorem price_comparison (x : ℝ) (h : x > 0) : x * 1.1 * 0.9 < x := by
  sorry

end price_comparison_l2982_298245


namespace count_problems_requiring_selection_l2982_298263

-- Define a structure to represent a problem
structure Problem where
  id : Nat
  requires_selection : Bool

-- Define our set of problems
def problems : List Problem := [
  { id := 1, requires_selection := true },  -- absolute value
  { id := 2, requires_selection := false }, -- square perimeter
  { id := 3, requires_selection := true },  -- maximum of three numbers
  { id := 4, requires_selection := true }   -- function value
]

-- Theorem statement
theorem count_problems_requiring_selection :
  (problems.filter Problem.requires_selection).length = 3 := by
  sorry

end count_problems_requiring_selection_l2982_298263


namespace application_methods_for_five_graduates_three_universities_l2982_298232

/-- The number of different application methods for high school graduates to universities -/
def application_methods (num_graduates : ℕ) (num_universities : ℕ) : ℕ :=
  num_universities ^ num_graduates

/-- Theorem: Given 5 high school graduates and 3 universities, where each graduate can only apply to one university, the total number of different application methods is 3^5 -/
theorem application_methods_for_five_graduates_three_universities :
  application_methods 5 3 = 3^5 := by
  sorry

end application_methods_for_five_graduates_three_universities_l2982_298232


namespace problem_solution_l2982_298289

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : x * x^2 * (1/x) = 100/81) : x = 10/9 := by
  sorry

end problem_solution_l2982_298289


namespace log_simplification_l2982_298284

theorem log_simplification (p q r s t z : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hz : z > 0) :
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * t / (s * z)) = Real.log (z / t) := by
  sorry

end log_simplification_l2982_298284


namespace line_intersects_plane_implies_skew_line_exists_l2982_298207

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Predicate to check if a line intersects a plane -/
def intersects (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Predicate to check if a line is within a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Predicate to check if two lines are skew -/
def skew (l1 l2 : Line3D) : Prop := sorry

/-- Main theorem -/
theorem line_intersects_plane_implies_skew_line_exists (l : Line3D) (α : Plane3D) :
  intersects l α → ∃ m : Line3D, line_in_plane m α ∧ skew l m := by
  sorry

end line_intersects_plane_implies_skew_line_exists_l2982_298207


namespace simple_compound_interest_relation_l2982_298287

/-- Calculates simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates compound interest (annually compounded) -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

theorem simple_compound_interest_relation :
  ∀ (y : ℝ),
    compound_interest 6000 (y / 100) 2 = 615 →
    simple_interest 6000 (y / 100) 2 = 1200 :=
by
  sorry

end simple_compound_interest_relation_l2982_298287


namespace max_integer_difference_l2982_298247

theorem max_integer_difference (x y : ℤ) (hx : -6 < x ∧ x < -2) (hy : 4 < y ∧ y < 10) :
  (∀ (a b : ℤ), -6 < a ∧ a < -2 ∧ 4 < b ∧ b < 10 → b - a ≤ y - x) →
  y - x = 14 :=
by sorry

end max_integer_difference_l2982_298247


namespace wrapping_paper_fraction_l2982_298276

theorem wrapping_paper_fraction (total_used : ℚ) (num_presents : ℕ) (h1 : total_used = 1/2) (h2 : num_presents = 5) :
  total_used / num_presents = 1/10 :=
by sorry

end wrapping_paper_fraction_l2982_298276


namespace chord_length_30_60_l2982_298298

theorem chord_length_30_60 : 
  let A : ℝ × ℝ := (Real.cos (π / 6), Real.sin (π / 6))
  let B : ℝ × ℝ := (Real.cos (π / 3), Real.sin (π / 3))
  let chord_length := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  chord_length = (Real.sqrt 6 - Real.sqrt 2) / 2 := by
  sorry

#check chord_length_30_60

end chord_length_30_60_l2982_298298


namespace parabola_transformation_l2982_298269

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b
  , c := p.c + v }

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { a := 1
  , b := 0
  , c := 0 }

theorem parabola_transformation :
  let p1 := shift_horizontal original_parabola 3
  let p2 := shift_vertical p1 4
  p2 = { a := 1, b := -6, c := 13 } := by sorry

end parabola_transformation_l2982_298269


namespace product_pricing_and_purchase_l2982_298255

-- Define variables
variable (x : ℝ) -- Price of product A
variable (y : ℝ) -- Price of product B
variable (m : ℝ) -- Number of units of product A to be purchased

-- Define the conditions
def condition1 : Prop := 2 * x + 3 * y = 690
def condition2 : Prop := x + 4 * y = 720
def condition3 : Prop := m * x + (40 - m) * y ≤ 5400
def condition4 : Prop := m ≤ 3 * (40 - m)

-- State the theorem
theorem product_pricing_and_purchase (h1 : condition1 x y) (h2 : condition2 x y) 
  (h3 : condition3 x y m) (h4 : condition4 m) : 
  x = 120 ∧ y = 150 ∧ 20 ≤ m ∧ m ≤ 30 := by
  sorry

end product_pricing_and_purchase_l2982_298255


namespace inverse_variation_with_increase_l2982_298285

/-- Given two inversely varying quantities a and b, prove that when their product increases by 50% and a becomes 1600, b equals 0.375 -/
theorem inverse_variation_with_increase (a b a' b' : ℝ) : 
  (a * b = 800 * 0.5) →  -- Initial condition
  (a' * b' = 1.5 * (a * b)) →  -- 50% increase in product
  (a' = 1600) →  -- New value of a
  (b' = 0.375) :=  -- Theorem to prove
by sorry

end inverse_variation_with_increase_l2982_298285


namespace warriors_truth_count_l2982_298228

theorem warriors_truth_count :
  ∀ (total_warriors : ℕ) 
    (sword_yes spear_yes axe_yes bow_yes : ℕ),
  total_warriors = 33 →
  sword_yes = 13 →
  spear_yes = 15 →
  axe_yes = 20 →
  bow_yes = 27 →
  ∃ (truth_tellers : ℕ),
    truth_tellers = 12 ∧
    truth_tellers + (total_warriors - truth_tellers) * 3 = 
      sword_yes + spear_yes + axe_yes + bow_yes :=
by sorry

end warriors_truth_count_l2982_298228


namespace perpendicular_vectors_y_value_l2982_298206

theorem perpendicular_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) : 
  a = (2, 4) → 
  b = (-4, y) → 
  a.1 * b.1 + a.2 * b.2 = 0 → 
  y = 2 := by
sorry

end perpendicular_vectors_y_value_l2982_298206


namespace correct_sum_after_change_l2982_298279

def number1 : ℕ := 935641
def number2 : ℕ := 471850
def incorrect_sum : ℕ := 1417491
def digit_to_change : ℕ := 7
def new_digit : ℕ := 8

theorem correct_sum_after_change :
  ∃ (changed_number2 : ℕ),
    (changed_number2 ≠ number2) ∧
    (∃ (pos : ℕ),
      (number2 / 10^pos) % 10 = digit_to_change ∧
      changed_number2 = number2 + (new_digit - digit_to_change) * 10^pos) ∧
    (number1 + changed_number2 = incorrect_sum) :=
  sorry

end correct_sum_after_change_l2982_298279


namespace abby_damon_weight_l2982_298261

/-- The weights of four people satisfying certain conditions -/
structure Weights where
  a : ℝ  -- Abby's weight
  b : ℝ  -- Bart's weight
  c : ℝ  -- Cindy's weight
  d : ℝ  -- Damon's weight
  ab_sum : a + b = 300
  bc_sum : b + c = 280
  cd_sum : c + d = 290
  ac_bd_diff : a + c = b + d + 10

/-- Theorem stating that given the conditions, Abby and Damon's combined weight is 310 pounds -/
theorem abby_damon_weight (w : Weights) : w.a + w.d = 310 := by
  sorry

end abby_damon_weight_l2982_298261


namespace functional_equation_solution_l2982_298219

theorem functional_equation_solution (f g : ℚ → ℚ) 
  (h1 : ∀ x y : ℚ, f (g x - g y) = f (g x) - y)
  (h2 : ∀ x y : ℚ, g (f x - f y) = g (f x) - y) :
  ∃ c : ℚ, c ≠ 0 ∧ (∀ x : ℚ, f x = c * x) ∧ (∀ x : ℚ, g x = x / c) := by
  sorry

end functional_equation_solution_l2982_298219


namespace propane_tank_burner_cost_is_14_l2982_298256

def propane_tank_burner_cost (total_money sheet_cost rope_cost helium_cost_per_oz flight_height_per_oz max_height : ℚ) : ℚ :=
  let remaining_money := total_money - sheet_cost - rope_cost
  let helium_oz_needed := max_height / flight_height_per_oz
  let helium_cost := helium_oz_needed * helium_cost_per_oz
  remaining_money - helium_cost

theorem propane_tank_burner_cost_is_14 :
  propane_tank_burner_cost 200 42 18 1.5 113 9492 = 14 :=
by
  sorry

end propane_tank_burner_cost_is_14_l2982_298256


namespace inscribed_decagon_area_proof_l2982_298217

/-- The area of a decagon inscribed in a square with perimeter 150 cm, 
    where the vertices of the decagon divide each side of the square into five equal segments. -/
def inscribed_decagon_area : ℝ := 1181.25

/-- The perimeter of the square. -/
def square_perimeter : ℝ := 150

/-- The number of equal segments each side of the square is divided into. -/
def num_segments : ℕ := 5

/-- The number of triangles removed from the square to form the decagon. -/
def num_triangles : ℕ := 8

theorem inscribed_decagon_area_proof :
  let side_length := square_perimeter / 4
  let segment_length := side_length / num_segments
  let triangle_area := (1 / 2) * segment_length * segment_length
  let total_triangle_area := num_triangles * triangle_area
  let square_area := side_length * side_length
  square_area - total_triangle_area = inscribed_decagon_area := by sorry

end inscribed_decagon_area_proof_l2982_298217


namespace midpoint_fraction_l2982_298274

theorem midpoint_fraction : 
  let a := 3/4
  let b := 5/6
  (a + b) / 2 = 19/24 := by
sorry

end midpoint_fraction_l2982_298274


namespace amelia_win_probability_l2982_298262

/-- Represents the outcome of a single coin toss -/
inductive CoinToss
| Heads
| Tails

/-- Represents a player in the game -/
inductive Player
| Amelia
| Blaine

/-- The state of the game after each round -/
structure GameState :=
  (round : Nat)
  (currentPlayer : Player)

/-- The result of the game -/
inductive GameResult
| AmeliaWins
| BlaineWins
| Tie

/-- The probability of getting heads for each player -/
def headsProbability (player : Player) : ℚ :=
  match player with
  | Player.Amelia => 1/4
  | Player.Blaine => 1/3

/-- The probability of the game ending in a specific result -/
noncomputable def gameResultProbability (result : GameResult) : ℚ :=
  sorry

/-- The main theorem stating the probability of Amelia winning -/
theorem amelia_win_probability :
  gameResultProbability GameResult.AmeliaWins = 15/32 :=
sorry

end amelia_win_probability_l2982_298262


namespace projection_a_on_b_l2982_298214

def a : ℝ × ℝ := (-8, 1)
def b : ℝ × ℝ := (3, 4)

theorem projection_a_on_b : 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -4 := by
  sorry

end projection_a_on_b_l2982_298214


namespace root_equation_problem_l2982_298257

theorem root_equation_problem (a b c m : ℝ) : 
  (a^2 - 4*a + m = 0 ∧ b^2 - 4*b + m = 0) →
  (b^2 - 8*b + 5*m = 0 ∧ c^2 - 8*c + 5*m = 0) →
  m = 0 ∨ m = 3 := by
sorry

end root_equation_problem_l2982_298257


namespace ember_nate_ages_l2982_298259

/-- Given that Ember is initially half as old as Nate, and Nate is initially 14 years old,
    prove that when Ember's age becomes 14, Nate's age will be 21. -/
theorem ember_nate_ages (ember_initial : ℕ) (nate_initial : ℕ) (ember_final : ℕ) (nate_final : ℕ) :
  ember_initial = nate_initial / 2 →
  nate_initial = 14 →
  ember_final = 14 →
  nate_final = nate_initial + (ember_final - ember_initial) →
  nate_final = 21 := by
sorry

end ember_nate_ages_l2982_298259


namespace binomial_square_constant_l2982_298225

theorem binomial_square_constant (c : ℚ) : 
  (∃ a b : ℚ, ∀ x, 9 * x^2 + 27 * x + c = (a * x + b)^2) → c = 81/4 := by
  sorry

end binomial_square_constant_l2982_298225


namespace problem_17_l2982_298250

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

def p (a : ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → log_a a (x + 3) > log_a a (y + 3)

def q (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 + (2*a - 3)*x₁ + 1 = 0 ∧ x₂^2 + (2*a - 3)*x₂ + 1 = 0

theorem problem_17 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (a ∈ Set.Icc (1/2) 1 ∪ Set.Ioi (5/2)) :=
by sorry

end problem_17_l2982_298250


namespace antonella_coins_l2982_298229

theorem antonella_coins (num_coins : ℕ) (loonie_value toonie_value : ℚ) 
  (frappuccino_cost remaining_money : ℚ) :
  num_coins = 10 →
  loonie_value = 1 →
  toonie_value = 2 →
  frappuccino_cost = 3 →
  remaining_money = 11 →
  ∃ (num_loonies num_toonies : ℕ),
    num_loonies + num_toonies = num_coins ∧
    num_loonies * loonie_value + num_toonies * toonie_value = 
      remaining_money + frappuccino_cost ∧
    num_toonies = 4 :=
by sorry

end antonella_coins_l2982_298229


namespace log_equation_solution_l2982_298211

theorem log_equation_solution (x : ℝ) :
  x > 0 → ((Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 4 ↔ x ≠ 1) :=
by sorry

end log_equation_solution_l2982_298211


namespace correct_calculation_l2982_298272

theorem correct_calculation (x : ℤ) : 
  (713 + x = 928) → (713 - x = 498) := by
  sorry

end correct_calculation_l2982_298272


namespace consecutive_integers_product_812_l2982_298277

theorem consecutive_integers_product_812 (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end consecutive_integers_product_812_l2982_298277


namespace complex_magnitude_problem_l2982_298258

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I)^2 = 2 - Complex.I) :
  Complex.abs z = Real.sqrt 5 / 2 := by
  sorry

end complex_magnitude_problem_l2982_298258


namespace f_g_deriv_neg_l2982_298241

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos : ∀ x : ℝ, x > 0 → deriv f x > 0
axiom g_deriv_pos : ∀ x : ℝ, x > 0 → deriv g x > 0

-- State the theorem
theorem f_g_deriv_neg (x : ℝ) (h : x < 0) : deriv f x > 0 ∧ deriv g x < 0 :=
sorry

end f_g_deriv_neg_l2982_298241


namespace min_value_and_inequality_l2982_298226

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem statement
theorem min_value_and_inequality :
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a ∧ ∃ (x₀ : ℝ), f x₀ = a) ∧
  (a = 3) ∧
  (∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → p + q + r = 3 → p^2 + q^2 + r^2 ≥ 3) :=
by sorry

end min_value_and_inequality_l2982_298226


namespace evening_rice_fraction_l2982_298215

/-- 
Given:
- Rose initially has 10 kg of rice
- She cooks 9/10 kg in the morning
- She has 750 g left at the end
Prove that the fraction of remaining rice cooked in the evening is 1/4
-/
theorem evening_rice_fraction (initial_rice : ℝ) (morning_cooked : ℝ) (final_rice : ℝ) :
  initial_rice = 10 →
  morning_cooked = 9/10 →
  final_rice = 750/1000 →
  (initial_rice - morning_cooked - final_rice) / (initial_rice - morning_cooked) = 1/4 := by
  sorry

end evening_rice_fraction_l2982_298215


namespace geometric_progression_first_term_l2982_298242

theorem geometric_progression_first_term (S : ℝ) (sum_first_two : ℝ) 
  (h1 : S = 8) (h2 : sum_first_two = 5) :
  ∃ (a : ℝ), (a = 8 * (1 - Real.sqrt 6 / 4) ∨ a = 8 * (1 + Real.sqrt 6 / 4)) ∧
    (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) := by
  sorry

end geometric_progression_first_term_l2982_298242


namespace max_erased_dots_l2982_298222

/-- Represents a domino tile with two halves -/
structure Domino :=
  (left : ℕ)
  (right : ℕ)

/-- The problem setup -/
def DominoArrangement :=
  { tiles : List Domino // tiles.length = 8 }

/-- The sum of dots on all visible tiles -/
def visibleDots (arr : DominoArrangement) : ℕ :=
  (arr.val.take 7).foldl (fun acc tile => acc + tile.left + tile.right) 0

/-- The total number of dots including the erased half -/
def totalDots (arr : DominoArrangement) (erased : ℕ) : ℕ :=
  visibleDots arr + erased

theorem max_erased_dots (arr : DominoArrangement) 
  (h1 : visibleDots arr = 37)
  (h2 : ∀ n : ℕ, totalDots arr n % 4 = 0 → n ≤ 3) :
  ∃ (n : ℕ), n ≤ 3 ∧ totalDots arr n % 4 = 0 ∧ 
    ∀ (m : ℕ), totalDots arr m % 4 = 0 → m ≤ n :=
by
  sorry

end max_erased_dots_l2982_298222


namespace rose_bed_fraction_l2982_298224

/-- Proof that the rose bed occupies 1/20 of the park's area given the conditions -/
theorem rose_bed_fraction (park_length park_width : ℝ) 
  (flower_bed_fraction : ℝ) (rose_bed_fraction : ℝ) :
  park_length = 15 →
  park_width = 20 →
  flower_bed_fraction = 1/5 →
  rose_bed_fraction = 1/4 →
  (flower_bed_fraction * rose_bed_fraction * park_length * park_width) / 
  (park_length * park_width) = 1/20 := by
  sorry

#check rose_bed_fraction

end rose_bed_fraction_l2982_298224


namespace unfair_coin_probability_l2982_298267

/-- The probability of exactly k successes in n trials of a Bernoulli experiment -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of exactly 7 tails in 10 flips of an unfair coin -/
theorem unfair_coin_probability : 
  binomialProbability 10 7 (2/3) = 512/6561 := by
  sorry

end unfair_coin_probability_l2982_298267


namespace inequality_solution_set_l2982_298216

-- Define the set of real numbers satisfying the inequality
def S : Set ℝ := {x : ℝ | |x - 2| - |2*x - 1| > 0}

-- State the theorem
theorem inequality_solution_set : S = Set.Ioo (-1 : ℝ) 1 := by
  sorry

end inequality_solution_set_l2982_298216


namespace angle_tangent_relation_l2982_298294

theorem angle_tangent_relation (θ : Real) :
  (-(π / 2) < θ ∧ θ < 0) →  -- θ is in the fourth quadrant
  (Real.sin (θ + π / 4) = 3 / 5) →
  (Real.tan (θ - π / 4) = -4 / 3) :=
by sorry

end angle_tangent_relation_l2982_298294


namespace books_remaining_l2982_298282

theorem books_remaining (initial_books : ℕ) (donating_people : ℕ) (books_per_donation : ℕ) (borrowed_books : ℕ) :
  initial_books = 500 →
  donating_people = 10 →
  books_per_donation = 8 →
  borrowed_books = 220 →
  initial_books + donating_people * books_per_donation - borrowed_books = 360 :=
by
  sorry

end books_remaining_l2982_298282


namespace vectors_not_coplanar_l2982_298235

/-- Three vectors in ℝ³ -/
def a : Fin 3 → ℝ := ![3, 7, 2]
def b : Fin 3 → ℝ := ![-2, 0, -1]
def c : Fin 3 → ℝ := ![2, 2, 1]

/-- Scalar triple product of three vectors in ℝ³ -/
def scalarTripleProduct (u v w : Fin 3 → ℝ) : ℝ :=
  Matrix.det !![u 0, u 1, u 2; v 0, v 1, v 2; w 0, w 1, w 2]

/-- Theorem: The vectors a, b, and c are not coplanar -/
theorem vectors_not_coplanar : scalarTripleProduct a b c ≠ 0 := by
  sorry

end vectors_not_coplanar_l2982_298235


namespace boxes_with_neither_l2982_298281

theorem boxes_with_neither (total_boxes : ℕ) (marker_boxes : ℕ) (crayon_boxes : ℕ) (both_boxes : ℕ) 
  (h1 : total_boxes = 15)
  (h2 : marker_boxes = 9)
  (h3 : crayon_boxes = 5)
  (h4 : both_boxes = 4) :
  total_boxes - (marker_boxes + crayon_boxes - both_boxes) = 5 := by
  sorry

end boxes_with_neither_l2982_298281


namespace smallest_positive_integer_properties_l2982_298280

theorem smallest_positive_integer_properties : ∃ a : ℕ, 
  (∀ n : ℕ, n > 0 → a ≤ n) ∧ 
  (a^3 + 1 = 2) ∧ 
  ((a + 1) * (a^2 - a + 1) = 2) :=
by sorry

end smallest_positive_integer_properties_l2982_298280


namespace sin_translation_l2982_298213

/-- Given a function f(x) = 3sin(2x), translating its graph π/6 units to the left
    results in the function g(x) = 3sin(2x + π/3) -/
theorem sin_translation (x : ℝ) :
  (fun x => 3 * Real.sin (2 * x + π / 3)) x =
  (fun x => 3 * Real.sin (2 * (x + π / 6))) x := by
sorry

end sin_translation_l2982_298213


namespace quadratic_roots_and_graph_point_l2982_298296

theorem quadratic_roots_and_graph_point (a b c : ℝ) (x : ℝ) 
  (h1 : a ≠ 0)
  (h2 : Real.tan x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a))
  (h3 : Real.tan (π/4 - x) = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a))
  : a * 1^2 + b * 1 - c = 0 :=
by sorry

end quadratic_roots_and_graph_point_l2982_298296


namespace isosceles_right_triangle_division_l2982_298218

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  side : ℝ
  side_positive : side > 0

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  leg : ℝ
  base1_positive : base1 > 0
  base2_positive : base2 > 0
  leg_positive : leg > 0

/-- A division of an isosceles right triangle into trapezoids -/
def TriangleDivision (t : IsoscelesRightTriangle) := 
  List IsoscelesTrapezoid

theorem isosceles_right_triangle_division (t : IsoscelesRightTriangle) :
  ∃ (d : TriangleDivision t), d.length = 7 :=
sorry

end isosceles_right_triangle_division_l2982_298218


namespace inequality_proof_l2982_298220

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / Real.sqrt (3 * a + 2 * b + c) +
  b / Real.sqrt (3 * b + 2 * c + a) +
  c / Real.sqrt (3 * c + 2 * a + b) ≤
  (1 / Real.sqrt 2) * Real.sqrt (a + b + c) := by
  sorry

end inequality_proof_l2982_298220


namespace diana_wins_probability_l2982_298299

-- Define the type for a die roll (1 to 6)
def DieRoll : Type := Fin 6

-- Define the type for a pair of dice rolls
def DicePair : Type := DieRoll × DieRoll

-- Function to calculate the sum of a pair of dice rolls
def diceSum (pair : DicePair) : Nat :=
  pair.1.val + pair.2.val + 2

-- Define the sample space of all possible outcomes
def sampleSpace : Finset (DicePair × DicePair) :=
  sorry

-- Define the event where Diana's sum exceeds Apollo's by at least 2
def favorableEvent : Finset (DicePair × DicePair) :=
  sorry

-- Theorem to prove
theorem diana_wins_probability :
  (favorableEvent.card : Rat) / sampleSpace.card = 47 / 432 := by
  sorry

end diana_wins_probability_l2982_298299


namespace juice_price_ratio_l2982_298243

theorem juice_price_ratio :
  ∀ (v_B p_B : ℝ), v_B > 0 → p_B > 0 →
  let v_A := 1.25 * v_B
  let p_A := 0.85 * p_B
  (p_A / v_A) / (p_B / v_B) = 17 / 25 := by
sorry

end juice_price_ratio_l2982_298243


namespace set_equality_l2982_298205

theorem set_equality (x y : ℝ) : 
  (x^2 - y^2 = x / (x^2 + y^2) ∧ 2*x*y + y / (x^2 + y^2) = 3) ↔ 
  (x^3 - 3*x*y^2 + 3*y = 1 ∧ 3*x^2*y - 3*x - y^3 = 0) := by
  sorry

end set_equality_l2982_298205


namespace original_profit_percentage_l2982_298254

theorem original_profit_percentage (cost selling_price : ℝ) 
  (h1 : cost > 0) 
  (h2 : selling_price > cost) 
  (h3 : selling_price - (1.12 * cost) = 0.552 * selling_price) : 
  (selling_price - cost) / cost = 1.5 := by
sorry

end original_profit_percentage_l2982_298254


namespace exists_n_with_factorial_property_and_digit_sum_l2982_298201

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The main theorem -/
theorem exists_n_with_factorial_property_and_digit_sum :
  ∃ n : ℕ, n > 0 ∧ 
    (Nat.factorial (n + 1) + Nat.factorial (n + 2) = Nat.factorial n * 1001) ∧
    (sum_of_digits n = 3) := by
  sorry

end exists_n_with_factorial_property_and_digit_sum_l2982_298201


namespace mean_of_three_numbers_l2982_298246

theorem mean_of_three_numbers (a b c : ℝ) : 
  (a + b + c + 105) / 4 = 90 →
  (a + b + c) / 3 = 85 := by
  sorry

end mean_of_three_numbers_l2982_298246


namespace log_2_irrational_l2982_298293

theorem log_2_irrational : Irrational (Real.log 2) := by
  sorry

end log_2_irrational_l2982_298293


namespace boys_to_total_ratio_l2982_298271

/-- Represents a classroom with boys and girls -/
structure Classroom where
  total_students : ℕ
  num_boys : ℕ
  num_girls : ℕ
  boys_plus_girls : num_boys + num_girls = total_students

/-- The probability of choosing a student from a group -/
def prob_choose (group : ℕ) (total : ℕ) : ℚ :=
  group / total

theorem boys_to_total_ratio (c : Classroom) 
  (h1 : c.total_students > 0)
  (h2 : prob_choose c.num_boys c.total_students = 
        (3 / 4) * prob_choose c.num_girls c.total_students) :
  (c.num_boys : ℚ) / c.total_students = 3 / 7 := by
  sorry

#check boys_to_total_ratio

end boys_to_total_ratio_l2982_298271


namespace jenny_stamps_last_page_l2982_298234

/-- Represents the stamp collection system -/
structure StampCollection where
  initialBooks : ℕ
  pagesPerBook : ℕ
  initialStampsPerPage : ℕ
  newStampsPerPage : ℕ
  filledBooks : ℕ
  filledPagesInLastBook : ℕ

/-- Calculates the number of stamps on the last page after reorganization -/
def stampsOnLastPage (sc : StampCollection) : ℕ :=
  let totalStamps := sc.initialBooks * sc.pagesPerBook * sc.initialStampsPerPage
  let filledPages := sc.filledBooks * sc.pagesPerBook + sc.filledPagesInLastBook
  totalStamps - (filledPages * sc.newStampsPerPage)

/-- Theorem: Given Jenny's stamp collection details, the last page contains 8 stamps -/
theorem jenny_stamps_last_page :
  let sc : StampCollection := {
    initialBooks := 10,
    pagesPerBook := 50,
    initialStampsPerPage := 6,
    newStampsPerPage := 8,
    filledBooks := 6,
    filledPagesInLastBook := 45
  }
  stampsOnLastPage sc = 8 := by
  sorry

end jenny_stamps_last_page_l2982_298234


namespace quadratic_roots_condition_l2982_298252

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ - 1 = 0 ∧ k * x₂^2 - 2 * x₂ - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
sorry

end quadratic_roots_condition_l2982_298252


namespace similar_triangles_leg_sum_l2982_298288

theorem similar_triangles_leg_sum (a₁ a₂ : ℝ) (s : ℝ) :
  a₁ > 0 → a₂ > 0 → s > 0 →
  a₁ = 8 → a₂ = 200 → -- areas of the triangles
  s = 2 → -- shorter leg of smaller triangle
  ∃ (l₁ l₂ : ℝ), 
    l₁ > 0 ∧ l₂ > 0 ∧
    a₁ = (1/2) * s * l₁ ∧ -- area of smaller triangle
    a₂ = (1/2) * (5*s) * (5*l₁) ∧ -- area of larger triangle
    l₁ + l₂ = 50 -- sum of legs of larger triangle
  := by sorry

end similar_triangles_leg_sum_l2982_298288


namespace quadratic_minimum_l2982_298286

theorem quadratic_minimum (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 12*x + 35
  ∀ y : ℝ, f x ≤ f y ↔ x = 6 :=
by sorry

end quadratic_minimum_l2982_298286


namespace rug_strip_width_l2982_298268

/-- Given a rectangular floor and a rug, proves that the width of the uncovered strip is 2 meters -/
theorem rug_strip_width (floor_length floor_width rug_area : ℝ) 
  (h1 : floor_length = 10) 
  (h2 : floor_width = 8) 
  (h3 : rug_area = 24) : 
  ∃ w : ℝ, w > 0 ∧ w < floor_width / 2 ∧ 
  (floor_length - 2 * w) * (floor_width - 2 * w) = rug_area ∧ 
  w = 2 :=
sorry

end rug_strip_width_l2982_298268
