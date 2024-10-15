import Mathlib

namespace NUMINAMATH_CALUDE_second_term_arithmetic_sequence_l3506_350692

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * a 1 + (n * (n - 1) / 2) * (a n - a 1)

theorem second_term_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_first_term : a 1 = -2010) 
  (h_sum_condition : sum_arithmetic_sequence a 2010 / 2010 - sum_arithmetic_sequence a 2008 / 2008 = 2) :
  a 2 = -2008 :=
sorry

end NUMINAMATH_CALUDE_second_term_arithmetic_sequence_l3506_350692


namespace NUMINAMATH_CALUDE_smallest_in_S_l3506_350688

def S : Set ℕ := {5, 8, 1, 2, 6}

theorem smallest_in_S : ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_in_S_l3506_350688


namespace NUMINAMATH_CALUDE_scientific_notation_of_chip_size_l3506_350611

theorem scientific_notation_of_chip_size :
  ∃ (a : ℝ) (n : ℤ), 0.000000014 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4 ∧ n = -8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_chip_size_l3506_350611


namespace NUMINAMATH_CALUDE_simplify_expression_l3506_350657

theorem simplify_expression : 
  let x := 2 / (Real.sqrt 2 + Real.sqrt 3)
  let y := Real.sqrt 2 / (4 * Real.sqrt (97 + 56 * Real.sqrt 3))
  x + y = (3 * Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3506_350657


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_199_l3506_350606

theorem modular_inverse_of_3_mod_199 : ∃ x : ℕ, 0 < x ∧ x < 199 ∧ (3 * x) % 199 = 1 :=
by
  use 133
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_199_l3506_350606


namespace NUMINAMATH_CALUDE_work_completion_time_l3506_350665

theorem work_completion_time 
  (original_men : ℕ) 
  (original_days : ℕ) 
  (absent_men : ℕ) 
  (final_days : ℕ) :
  original_men = 15 →
  original_days = 40 →
  absent_men = 5 →
  final_days = 60 →
  original_men * original_days = (original_men - absent_men) * final_days :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3506_350665


namespace NUMINAMATH_CALUDE_difference_of_squares_330_270_l3506_350636

theorem difference_of_squares_330_270 : 330^2 - 270^2 = 36000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_330_270_l3506_350636


namespace NUMINAMATH_CALUDE_hua_method_uses_golden_ratio_l3506_350625

/-- The optimal selection method popularized by Hua Luogeng -/
structure OptimalSelectionMethod where
  author : String
  concept : String

/-- Definition of Hua Luogeng's optimal selection method -/
def huaMethod : OptimalSelectionMethod :=
  { author := "Hua Luogeng"
  , concept := "golden ratio" }

/-- Theorem stating that Hua Luogeng's optimal selection method uses the golden ratio -/
theorem hua_method_uses_golden_ratio :
  huaMethod.concept = "golden ratio" := by
  sorry

end NUMINAMATH_CALUDE_hua_method_uses_golden_ratio_l3506_350625


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_l3506_350635

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel m n → 
  subset n β → 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_l3506_350635


namespace NUMINAMATH_CALUDE_binomial_expansion_97_cubed_l3506_350669

theorem binomial_expansion_97_cubed : 97^3 + 3*(97^2) + 3*97 + 1 = 941192 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_97_cubed_l3506_350669


namespace NUMINAMATH_CALUDE_ellipse_properties_l3506_350640

/-- An ellipse with specific properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  max_distance_to_foci : ℝ
  min_distance_to_foci : ℝ

/-- The standard form of an ellipse equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

/-- Theorem about a specific ellipse and its properties -/
theorem ellipse_properties (C : Ellipse) 
    (h1 : C.center = (0, 0))
    (h2 : C.foci_on_x_axis = true)
    (h3 : C.max_distance_to_foci = 3)
    (h4 : C.min_distance_to_foci = 1) :
  (∃ (x y : ℝ), standard_equation 4 3 x y) ∧ 
  (∃ (P F₁ F₂ : ℝ × ℝ), 
    (standard_equation 4 3 P.1 P.2) →
    (F₁ = (-1, 0) ∧ F₂ = (1, 0)) →
    (∀ (Q : ℝ × ℝ), standard_equation 4 3 Q.1 Q.2 → 
      dot_product (P.1 - F₁.1, P.2 - F₁.2) (P.1 - F₂.1, P.2 - F₂.2) ≤ 3 ∧
      dot_product (Q.1 - F₁.1, Q.2 - F₁.2) (Q.1 - F₂.1, Q.2 - F₂.2) ≥ 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_properties_l3506_350640


namespace NUMINAMATH_CALUDE_power_function_through_point_l3506_350673

/-- A power function that passes through the point (2,8) -/
def f (x : ℝ) : ℝ := x^3

theorem power_function_through_point (x : ℝ) :
  f 2 = 8 ∧ f 3 = 27 := by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3506_350673


namespace NUMINAMATH_CALUDE_negative_a_squared_times_a_fourth_l3506_350642

theorem negative_a_squared_times_a_fourth (a : ℝ) : (-a)^2 * a^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_squared_times_a_fourth_l3506_350642


namespace NUMINAMATH_CALUDE_yard_fencing_l3506_350624

theorem yard_fencing (length width : ℝ) : 
  length > 0 → 
  width > 0 → 
  length * width = 320 → 
  2 * width + length = 56 → 
  length = 40 := by
sorry

end NUMINAMATH_CALUDE_yard_fencing_l3506_350624


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3506_350697

/-- An isosceles triangle with side lengths 2, 2, and 5 has a perimeter of 9 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b c : ℝ),
      a = 2 ∧ b = 2 ∧ c = 5 ∧  -- Two sides are 2, one side is 5
      (a = b ∨ b = c ∨ a = c) ∧  -- Triangle is isosceles
      a + b > c ∧ b + c > a ∧ a + c > b ∧  -- Triangle inequality
      perimeter = a + b + c ∧  -- Definition of perimeter
      perimeter = 9  -- The perimeter is 9

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 9 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3506_350697


namespace NUMINAMATH_CALUDE_length_PF1_is_seven_halves_l3506_350609

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Definition of the foci -/
def focus1 (c : ℝ) : ℝ × ℝ := (-c, 0)
def focus2 (c : ℝ) : ℝ × ℝ := (c, 0)

/-- Definition of point P -/
def point_P (c y : ℝ) : ℝ × ℝ := (c, y)

/-- The line through F₂ and P is perpendicular to x-axis -/
def line_perpendicular (c y : ℝ) : Prop := 
  point_P c y = (c, y)

/-- Theorem: Length of PF₁ is 7/2 -/
theorem length_PF1_is_seven_halves (c y : ℝ) : 
  is_on_ellipse c y → 
  line_perpendicular c y → 
  let p := point_P c y
  let f1 := focus1 c
  Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) = 7/2 := by sorry

end NUMINAMATH_CALUDE_length_PF1_is_seven_halves_l3506_350609


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l3506_350651

theorem angle_sum_around_point (x : ℝ) : 
  (3 * x + 6 * x + x + 2 * x = 360) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l3506_350651


namespace NUMINAMATH_CALUDE_tom_seashells_l3506_350608

theorem tom_seashells (initial_seashells : ℕ) (given_away : ℕ) :
  initial_seashells = 5 →
  given_away = 2 →
  initial_seashells - given_away = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l3506_350608


namespace NUMINAMATH_CALUDE_min_value_expression_l3506_350643

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b)^2 ≥ 7 ∧
  (2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b)^2 = 7 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3506_350643


namespace NUMINAMATH_CALUDE_final_lights_on_l3506_350607

/-- The number of lights -/
def n : ℕ := 56

/-- Function to count lights turned on by pressing every k-th switch -/
def count_lights (k : ℕ) : ℕ :=
  n / k

/-- Function to count lights affected by both operations -/
def count_overlap : ℕ :=
  n / 15

/-- The final number of lights turned on -/
def lights_on : ℕ :=
  count_lights 3 + count_lights 5 - count_overlap

theorem final_lights_on :
  lights_on = 26 := by
  sorry

end NUMINAMATH_CALUDE_final_lights_on_l3506_350607


namespace NUMINAMATH_CALUDE_sum_of_multiples_l3506_350678

theorem sum_of_multiples (n : ℕ) (h : n = 13) : n + n + 2 * n + 4 * n = 104 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l3506_350678


namespace NUMINAMATH_CALUDE_complex_cube_theorem_l3506_350639

theorem complex_cube_theorem (z : ℂ) (h : z = 1 - I) :
  ((1 + I) / z) ^ 3 = -I := by sorry

end NUMINAMATH_CALUDE_complex_cube_theorem_l3506_350639


namespace NUMINAMATH_CALUDE_sin_equality_proof_l3506_350631

theorem sin_equality_proof (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) :
  Real.sin (n * π / 180) = Real.sin (721 * π / 180) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_proof_l3506_350631


namespace NUMINAMATH_CALUDE_max_ice_cream_servings_l3506_350612

/-- Represents a day in February --/
structure FebruaryDay where
  dayOfMonth : Nat
  dayOfWeek : Nat
  h1 : dayOfMonth ≥ 1 ∧ dayOfMonth ≤ 28
  h2 : dayOfWeek ≥ 1 ∧ dayOfWeek ≤ 7

/-- Defines the ice cream eating rules --/
def iceCreamServings (day : FebruaryDay) : Nat :=
  if day.dayOfMonth % 2 = 0 ∧ (day.dayOfWeek = 3 ∨ day.dayOfWeek = 4) then 7
  else if (day.dayOfWeek = 1 ∨ day.dayOfWeek = 2) ∧ day.dayOfMonth % 2 = 1 then 3
  else if day.dayOfWeek = 5 then day.dayOfMonth
  else 0

/-- Theorem stating the maximum number of ice cream servings in February --/
theorem max_ice_cream_servings :
  (∃ (days : List FebruaryDay), days.length = 28 ∧
    (∀ d ∈ days, d.dayOfMonth ≥ 1 ∧ d.dayOfMonth ≤ 28 ∧ d.dayOfWeek ≥ 1 ∧ d.dayOfWeek ≤ 7) ∧
    (∀ i j, i ≠ j → (days.get i).dayOfMonth ≠ (days.get j).dayOfMonth) ∧
    (List.sum (days.map iceCreamServings) ≤ 110)) ∧
  (∃ (optimalDays : List FebruaryDay), optimalDays.length = 28 ∧
    (∀ d ∈ optimalDays, d.dayOfMonth ≥ 1 ∧ d.dayOfMonth ≤ 28 ∧ d.dayOfWeek ≥ 1 ∧ d.dayOfWeek ≤ 7) ∧
    (∀ i j, i ≠ j → (optimalDays.get i).dayOfMonth ≠ (optimalDays.get j).dayOfMonth) ∧
    (List.sum (optimalDays.map iceCreamServings) = 110)) := by
  sorry


end NUMINAMATH_CALUDE_max_ice_cream_servings_l3506_350612


namespace NUMINAMATH_CALUDE_range_of_x_l3506_350622

-- Define the set A as [2, 5]
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 5 }

-- Define the set B as {x | x < 1 ∨ x > 4}
def B : Set ℝ := { x | x < 1 ∨ x > 4 }

-- Define the statement S
def S (x : ℝ) : Prop := x ∈ A ∨ x ∈ B

-- Define the range R as [1, 2)
def R : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem range_of_x (x : ℝ) : ¬(S x) → x ∈ R := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l3506_350622


namespace NUMINAMATH_CALUDE_fifth_term_product_l3506_350694

/-- Given an arithmetic sequence a and a geometric sequence b with specified initial terms,
    prove that the product of their 5th terms is 80. -/
theorem fifth_term_product (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, b (n + 1) / b n = b 2 / b 1) →  -- geometric sequence condition
  a 1 = 1 → b 1 = 1 → 
  a 2 = 2 → b 2 = 2 → 
  a 5 * b 5 = 80 := by
sorry


end NUMINAMATH_CALUDE_fifth_term_product_l3506_350694


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3506_350663

/-- Simplification of a polynomial expression -/
theorem polynomial_simplification (y : ℝ) :
  4 * y - 8 * y^2 + 6 - (3 - 6 * y - 9 * y^2 + 2 * y^3) =
  -2 * y^3 + y^2 + 10 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3506_350663


namespace NUMINAMATH_CALUDE_complementary_of_35_is_55_l3506_350632

/-- The complementary angle of a given angle in degrees -/
def complementaryAngle (angle : ℝ) : ℝ := 90 - angle

/-- Theorem: The complementary angle of 35° is 55° -/
theorem complementary_of_35_is_55 :
  complementaryAngle 35 = 55 := by
  sorry

end NUMINAMATH_CALUDE_complementary_of_35_is_55_l3506_350632


namespace NUMINAMATH_CALUDE_simplify_expression_l3506_350610

theorem simplify_expression (x : ℝ) :
  2 * x * (4 * x^2 - 3) - 4 * (x^2 - 3 * x + 6) = 8 * x^3 - 4 * x^2 + 6 * x - 24 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3506_350610


namespace NUMINAMATH_CALUDE_prob_different_colors_example_l3506_350604

/-- A box containing colored balls. -/
structure Box where
  white : ℕ
  black : ℕ

/-- The probability of drawing two balls of different colors with replacement. -/
def prob_different_colors (b : Box) : ℚ :=
  (b.white * b.black + b.black * b.white) / ((b.white + b.black) * (b.white + b.black))

/-- Theorem: The probability of drawing two balls of different colors from a box 
    containing 2 white balls and 3 black balls, with replacement, is 12/25. -/
theorem prob_different_colors_example : 
  prob_different_colors ⟨2, 3⟩ = 12 / 25 := by
  sorry

#eval prob_different_colors ⟨2, 3⟩

end NUMINAMATH_CALUDE_prob_different_colors_example_l3506_350604


namespace NUMINAMATH_CALUDE_burattino_awake_journey_fraction_l3506_350617

theorem burattino_awake_journey_fraction (x : ℝ) (h : x > 0) :
  let distance_before_sleep := x / 2
  let distance_slept := x / 3
  let distance_after_wake := x - (distance_before_sleep + distance_slept)
  distance_after_wake = distance_slept / 2 →
  (distance_before_sleep + distance_after_wake) / x = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_burattino_awake_journey_fraction_l3506_350617


namespace NUMINAMATH_CALUDE_negation_of_conditional_l3506_350644

theorem negation_of_conditional (a b : ℝ) :
  ¬(a > b → a - 1 > b - 1) ↔ (a ≤ b → a - 1 ≤ b - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l3506_350644


namespace NUMINAMATH_CALUDE_power_equation_solution_l3506_350695

theorem power_equation_solution (n : ℕ) : 5^29 * 4^15 = 2 * 10^n → n = 29 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3506_350695


namespace NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_l3506_350699

-- Problem 1
theorem problem_one (α : Real) (h : Real.tan α = 3) :
  (3 * Real.sin α + Real.cos α) / (Real.sin α - 2 * Real.cos α) = 10 := by sorry

-- Problem 2
theorem problem_two (α : Real) :
  (-Real.sin (π + α) + Real.sin (-α) - Real.tan (2*π + α)) /
  (Real.tan (α + π) + Real.cos (-α) + Real.cos (π - α)) = -1 := by sorry

-- Problem 3
theorem problem_three (α : Real) (h1 : Real.sin α + Real.cos α = 1/2) (h2 : 0 < α) (h3 : α < π) :
  Real.sin α * Real.cos α = -3/8 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_l3506_350699


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l3506_350627

theorem largest_integer_in_interval : ∃ (x : ℤ), 
  (∀ (y : ℤ), (1/5 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/12 → y ≤ x) ∧
  (1/5 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 7/12 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l3506_350627


namespace NUMINAMATH_CALUDE_log_sum_minus_exp_equality_l3506_350662

theorem log_sum_minus_exp_equality : Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10 - 42 * 8^(1/4) - 2017^0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_minus_exp_equality_l3506_350662


namespace NUMINAMATH_CALUDE_arithmetic_sequence_triangle_cosine_identity_l3506_350680

theorem arithmetic_sequence_triangle_cosine_identity (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  2 * b = a + c →
  5 * Real.cos A - 4 * Real.cos A * Real.cos C + 5 * Real.cos C = 8 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_triangle_cosine_identity_l3506_350680


namespace NUMINAMATH_CALUDE_club_selection_count_l3506_350679

/-- A club with members of different genders and service lengths -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat
  longest_serving_boys : Nat
  longest_serving_girls : Nat

/-- The conditions for selecting president and vice-president -/
def valid_selection (c : Club) : Prop :=
  c.total_members = 30 ∧
  c.boys = 15 ∧
  c.girls = 15 ∧
  c.longest_serving_boys = 6 ∧
  c.longest_serving_girls = 6

/-- The number of ways to select a president and vice-president -/
def selection_count (c : Club) : Nat :=
  c.longest_serving_boys * c.girls + c.longest_serving_girls * c.boys

/-- Theorem stating the number of ways to select president and vice-president -/
theorem club_selection_count (c : Club) :
  valid_selection c → selection_count c = 180 := by
  sorry

end NUMINAMATH_CALUDE_club_selection_count_l3506_350679


namespace NUMINAMATH_CALUDE_independent_recruitment_probabilities_l3506_350650

/-- Represents a student in the independent recruitment process -/
inductive Student
| A
| B
| C

/-- The probability of passing the review for each student -/
def review_prob (s : Student) : ℝ :=
  match s with
  | Student.A => 0.5
  | Student.B => 0.6
  | Student.C => 0.4

/-- The probability of passing the cultural test after passing the review for each student -/
def cultural_test_prob (s : Student) : ℝ :=
  match s with
  | Student.A => 0.6
  | Student.B => 0.5
  | Student.C => 0.75

/-- The probability of obtaining qualification for independent recruitment for each student -/
def qualification_prob (s : Student) : ℝ :=
  review_prob s * cultural_test_prob s

/-- The number of students who obtain qualification for independent recruitment -/
def num_qualified : Fin 4 → ℝ
| 0 => (1 - qualification_prob Student.A) * (1 - qualification_prob Student.B) * (1 - qualification_prob Student.C)
| 1 => 3 * qualification_prob Student.A * (1 - qualification_prob Student.B) * (1 - qualification_prob Student.C)
| 2 => 3 * qualification_prob Student.A * qualification_prob Student.B * (1 - qualification_prob Student.C)
| 3 => qualification_prob Student.A * qualification_prob Student.B * qualification_prob Student.C

/-- The expected value of the number of students who obtain qualification -/
def expected_num_qualified : ℝ :=
  1 * num_qualified 1 + 2 * num_qualified 2 + 3 * num_qualified 3

theorem independent_recruitment_probabilities :
  (∀ s : Student, qualification_prob s = 0.3) ∧ expected_num_qualified = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_independent_recruitment_probabilities_l3506_350650


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3506_350674

def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x > 1}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3506_350674


namespace NUMINAMATH_CALUDE_sara_movie_rental_cost_l3506_350664

def movie_spending (theater_ticket_price : ℚ) (num_tickets : ℕ) (bought_movie_price : ℚ) (total_spent : ℚ) : Prop :=
  let theater_total : ℚ := theater_ticket_price * num_tickets
  let known_expenses : ℚ := theater_total + bought_movie_price
  let rental_cost : ℚ := total_spent - known_expenses
  rental_cost = 159/100

theorem sara_movie_rental_cost :
  movie_spending (1062/100) 2 (1395/100) (3678/100) :=
sorry

end NUMINAMATH_CALUDE_sara_movie_rental_cost_l3506_350664


namespace NUMINAMATH_CALUDE_trivia_team_tryouts_l3506_350677

theorem trivia_team_tryouts (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) :
  not_picked = 17 →
  groups = 8 →
  students_per_group = 6 →
  not_picked + groups * students_per_group = 65 :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_tryouts_l3506_350677


namespace NUMINAMATH_CALUDE_don_remaining_rum_l3506_350629

/-- The amount of rum Sally gave Don on his pancakes, in ounces. -/
def initial_rum : ℝ := 10

/-- The maximum factor by which Don can consume rum for a healthy diet. -/
def max_factor : ℝ := 3

/-- The amount of rum Don had earlier that day, in ounces. -/
def earlier_rum : ℝ := 12

/-- Calculates the maximum amount of rum Don can consume for a healthy diet. -/
def max_rum : ℝ := initial_rum * max_factor

/-- Calculates the total amount of rum Don has consumed so far. -/
def consumed_rum : ℝ := initial_rum + earlier_rum

/-- Theorem stating how much rum Don can have after eating all of the rum and pancakes. -/
theorem don_remaining_rum : max_rum - consumed_rum = 8 := by sorry

end NUMINAMATH_CALUDE_don_remaining_rum_l3506_350629


namespace NUMINAMATH_CALUDE_enclosure_posts_count_l3506_350601

/-- Calculates the number of posts needed for a rectangular enclosure with a stone wall --/
def calculate_posts (length width wall_length post_spacing : ℕ) : ℕ :=
  let long_side := max length width
  let short_side := min length width
  let long_side_posts := long_side / post_spacing + 1
  let short_side_posts := (short_side / post_spacing + 1) - 1
  long_side_posts + 2 * short_side_posts

/-- The number of posts required for the given enclosure is 19 --/
theorem enclosure_posts_count :
  calculate_posts 50 80 120 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_enclosure_posts_count_l3506_350601


namespace NUMINAMATH_CALUDE_football_team_size_l3506_350660

/-- The number of players on a football team -/
def total_players : ℕ := 70

/-- The number of throwers on the team -/
def throwers : ℕ := 46

/-- The number of right-handed players on the team -/
def right_handed : ℕ := 62

/-- All throwers are right-handed -/
axiom throwers_are_right_handed : throwers ≤ right_handed

/-- One third of non-throwers are left-handed -/
axiom one_third_left_handed :
  3 * (total_players - throwers - (right_handed - throwers)) = total_players - throwers

theorem football_team_size :
  total_players = 70 :=
sorry

end NUMINAMATH_CALUDE_football_team_size_l3506_350660


namespace NUMINAMATH_CALUDE_triangle_largest_angle_and_type_l3506_350621

-- Define the triangle with angle ratio 4:3:2
def triangle_angles (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 180 ∧
  4 * b = 3 * a ∧ 3 * c = 2 * b

-- Theorem statement
theorem triangle_largest_angle_and_type 
  (a b c : ℝ) (h : triangle_angles a b c) : 
  a = 80 ∧ a < 90 ∧ b < 90 ∧ c < 90 := by
  sorry


end NUMINAMATH_CALUDE_triangle_largest_angle_and_type_l3506_350621


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l3506_350603

/-- Represents a rectangular floor pattern with black and white tiles -/
structure FloorPattern where
  width : ℕ
  height : ℕ
  blackTiles : ℕ
  whiteTiles : ℕ

/-- Adds a border of white tiles to a floor pattern -/
def addWhiteBorder (pattern : FloorPattern) : FloorPattern :=
  { width := pattern.width + 2
  , height := pattern.height + 2
  , blackTiles := pattern.blackTiles
  , whiteTiles := pattern.whiteTiles + (pattern.width + 2) * (pattern.height + 2) - (pattern.width * pattern.height)
  }

/-- Calculates the ratio of black tiles to white tiles -/
def tileRatio (pattern : FloorPattern) : ℚ :=
  pattern.blackTiles / pattern.whiteTiles

theorem extended_pattern_ratio :
  let initialPattern : FloorPattern :=
    { width := 5
    , height := 7
    , blackTiles := 14
    , whiteTiles := 21
    }
  let extendedPattern := addWhiteBorder initialPattern
  tileRatio extendedPattern = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l3506_350603


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_48_l3506_350655

theorem smallest_positive_multiple_of_48 :
  ∀ n : ℕ, n > 0 → 48 ∣ n → n ≥ 48 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_48_l3506_350655


namespace NUMINAMATH_CALUDE_student_rank_from_right_l3506_350653

theorem student_rank_from_right 
  (total_students : Nat) 
  (rank_from_left : Nat) 
  (h1 : total_students = 21)
  (h2 : rank_from_left = 6) :
  total_students - rank_from_left + 1 = 17 :=
by sorry

end NUMINAMATH_CALUDE_student_rank_from_right_l3506_350653


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3506_350689

theorem complex_fraction_simplification :
  (7 + 18 * Complex.I) / (4 - 5 * Complex.I) = -62/41 + (107/41) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3506_350689


namespace NUMINAMATH_CALUDE_binomial_18_4_l3506_350638

theorem binomial_18_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_4_l3506_350638


namespace NUMINAMATH_CALUDE_complex_vector_properties_l3506_350620

/-- Represents a complex number in the Cartesian plane -/
structure ComplexVector where
  x : ℝ
  y : ℝ

/-- Given an imaginary number z, returns its corresponding vector representation -/
noncomputable def z_to_vector (z : ℂ) : ComplexVector :=
  { x := z.re, y := z.im }

/-- Given an imaginary number z, returns the vector representation of its conjugate -/
noncomputable def conj_to_vector (z : ℂ) : ComplexVector :=
  { x := z.re, y := -z.im }

/-- Given an imaginary number z, returns the vector representation of its reciprocal -/
noncomputable def recip_to_vector (z : ℂ) : ComplexVector :=
  { x := z.re / (z.re^2 + z.im^2), y := -z.im / (z.re^2 + z.im^2) }

/-- Checks if three points are collinear given two vectors -/
def are_collinear (v1 v2 : ComplexVector) : Prop :=
  v1.x * v2.y = v1.y * v2.x

/-- Adds two ComplexVectors -/
def add_vectors (v1 v2 : ComplexVector) : ComplexVector :=
  { x := v1.x + v2.x, y := v1.y + v2.y }

theorem complex_vector_properties (z : ℂ) (h : z^3 = 1) :
  let OA := z_to_vector z
  let OB := conj_to_vector z
  let OC := recip_to_vector z
  (are_collinear OB OC) ∧ (add_vectors OA OC = ComplexVector.mk (-1) 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_vector_properties_l3506_350620


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l3506_350671

theorem matrix_equation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, 5; -1, 4]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![3, -8; 4, -11]
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![4/13, -31/13; 5/13, -42/13]
  P * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l3506_350671


namespace NUMINAMATH_CALUDE_increasing_function_properties_l3506_350647

/-- A function f is increasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem increasing_function_properties
  (f : ℝ → ℝ) (hf : IncreasingFunction f) :
  (∀ a b : ℝ, a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b)) ∧
  (∀ a b : ℝ, f a + f b ≥ f (-a) + f (-b) → a + b ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_properties_l3506_350647


namespace NUMINAMATH_CALUDE_hyperbola_focus_smaller_x_l3506_350682

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 5)^2 / 7^2 - (y - 20)^2 / 15^2 = 1

-- Define the focus with smaller x-coordinate
def focus_smaller_x : ℝ × ℝ := (-11.55, 20)

-- Theorem statement
theorem hyperbola_focus_smaller_x :
  ∃ (f : ℝ × ℝ), 
    (∀ x y, hyperbola x y → (x - 5)^2 + (y - 20)^2 ≥ (f.1 - 5)^2 + (f.2 - 20)^2) ∧
    (∀ x y, hyperbola x y → x ≤ 5 → x ≥ f.1) ∧
    f = focus_smaller_x :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_smaller_x_l3506_350682


namespace NUMINAMATH_CALUDE_max_min_values_part1_unique_b_part2_l3506_350614

noncomputable section

def f (a b x : ℝ) : ℝ := a * x^2 + b * x - Real.log x

theorem max_min_values_part1 :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x ≤ max) ∧
    (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x = max) ∧
    (∀ x ∈ Set.Icc (1/2 : ℝ) 2, min ≤ f (-1) 3 x) ∧
    (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f (-1) 3 x = min) ∧
    max = 2 ∧
    min = Real.log 2 + 5/4 :=
sorry

theorem unique_b_part2 :
  ∃! b : ℝ,
    b > 0 ∧
    (∀ x ∈ Set.Ioo 0 (Real.exp 1),
      f 0 b x ≥ 3) ∧
    (∃ x ∈ Set.Ioo 0 (Real.exp 1),
      f 0 b x = 3) ∧
    b = Real.exp 2 :=
sorry

end NUMINAMATH_CALUDE_max_min_values_part1_unique_b_part2_l3506_350614


namespace NUMINAMATH_CALUDE_tangent_product_equals_three_l3506_350648

theorem tangent_product_equals_three :
  Real.tan (20 * π / 180) * Real.tan (40 * π / 180) * Real.tan (60 * π / 180) * Real.tan (80 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equals_three_l3506_350648


namespace NUMINAMATH_CALUDE_tessa_initial_apples_l3506_350656

/-- The number of apples needed to make a pie -/
def apples_for_pie : ℕ := 10

/-- The number of apples Anita gave to Tessa -/
def apples_from_anita : ℕ := 5

/-- The number of additional apples Tessa needs after receiving apples from Anita -/
def additional_apples_needed : ℕ := 1

/-- Tessa's initial number of apples -/
def initial_apples : ℕ := apples_for_pie - apples_from_anita - additional_apples_needed

theorem tessa_initial_apples :
  initial_apples = 4 := by sorry

end NUMINAMATH_CALUDE_tessa_initial_apples_l3506_350656


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l3506_350645

theorem gold_coin_distribution (x y : ℕ) (h : x^2 - y^2 = 49*(x - y)) : x + y = 49 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l3506_350645


namespace NUMINAMATH_CALUDE_abc_inequality_l3506_350667

theorem abc_inequality (a b c : ℝ) (ha : a = Real.rpow 0.8 0.8) 
  (hb : b = Real.rpow 0.8 0.9) (hc : c = Real.rpow 1.2 0.8) : c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3506_350667


namespace NUMINAMATH_CALUDE_poultry_pricing_l3506_350613

theorem poultry_pricing :
  ∃ (c d g : ℕ+),
    3 * c + d = 2 * g ∧
    c + 2 * d + 3 * g = 25 ∧
    c = 2 ∧ d = 4 ∧ g = 5 := by
  sorry

end NUMINAMATH_CALUDE_poultry_pricing_l3506_350613


namespace NUMINAMATH_CALUDE_log_relationship_l3506_350693

theorem log_relationship (a b : ℝ) : 
  a = (Real.log 125) / (Real.log 4) → 
  b = (Real.log 32) / (Real.log 5) → 
  a = (3 * b) / 10 := by
sorry

end NUMINAMATH_CALUDE_log_relationship_l3506_350693


namespace NUMINAMATH_CALUDE_train_length_l3506_350658

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 5 → speed * time * (1000 / 3600) = 125 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3506_350658


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l3506_350637

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 85 ∧ n % 9 = 3 ∧ ∀ m : ℕ, m < 85 ∧ m % 9 = 3 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l3506_350637


namespace NUMINAMATH_CALUDE_cos_120_degrees_l3506_350670

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l3506_350670


namespace NUMINAMATH_CALUDE_count_multiples_6_or_8_not_both_l3506_350672

theorem count_multiples_6_or_8_not_both : Nat := by
  -- Define the set of positive integers less than 201
  let S := Finset.range 201

  -- Define the subsets of multiples of 6 and 8
  let mult6 := S.filter (λ n => n % 6 = 0)
  let mult8 := S.filter (λ n => n % 8 = 0)

  -- Define the set of numbers that are multiples of either 6 or 8, but not both
  let result := (mult6 ∪ mult8) \ (mult6 ∩ mult8)

  -- The theorem statement
  have : result.card = 42 := by sorry

  -- Return the result
  exact result.card

end NUMINAMATH_CALUDE_count_multiples_6_or_8_not_both_l3506_350672


namespace NUMINAMATH_CALUDE_simplify_cube_root_exponent_sum_l3506_350600

theorem simplify_cube_root_exponent_sum (a b c : ℝ) : 
  ∃ (k : ℝ) (m n p : ℕ), 
    (54 * a^6 * b^8 * c^14)^(1/3) = k * a^m * b^n * c^p ∧ m + n + p = 8 :=
by sorry

end NUMINAMATH_CALUDE_simplify_cube_root_exponent_sum_l3506_350600


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3506_350659

theorem trigonometric_equation_solution (x : Real) :
  8.419 * Real.sin x + Real.sqrt (2 - Real.sin x ^ 2) + Real.sin x * Real.sqrt (2 - Real.sin x ^ 2) = 3 ↔
  ∃ k : ℤ, x = π / 2 + 2 * π * k :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3506_350659


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3506_350649

theorem quadratic_equation_solution (c : ℝ) : 
  ((-5 : ℝ)^2 + c * (-5) - 45 = 0) → c = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3506_350649


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3506_350696

theorem sqrt_x_minus_one_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3506_350696


namespace NUMINAMATH_CALUDE_two_distinct_solutions_l3506_350646

theorem two_distinct_solutions (a : ℝ) : 
  (16 * (a - 3) > 0) →
  (a > 0) →
  (a^2 - 16*a + 48 > 0) →
  (a ≠ 19) →
  (∃ (x₁ x₂ : ℝ), x₁ = a + 4 * Real.sqrt (a - 3) ∧ 
                   x₂ = a - 4 * Real.sqrt (a - 3) ∧ 
                   x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂) →
  (a > 3 ∧ a < 4) ∨ (a > 12 ∧ a < 19) ∨ (a > 19) :=
by sorry


end NUMINAMATH_CALUDE_two_distinct_solutions_l3506_350646


namespace NUMINAMATH_CALUDE_five_digit_reverse_multiplication_l3506_350630

theorem five_digit_reverse_multiplication (a b c d e : Nat) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 →
  4 * (a * 10000 + b * 1000 + c * 100 + d * 10 + e) = e * 10000 + d * 1000 + c * 100 + b * 10 + a →
  a + b + c + d + e = 27 := by
sorry

end NUMINAMATH_CALUDE_five_digit_reverse_multiplication_l3506_350630


namespace NUMINAMATH_CALUDE_small_triangle_perimeter_l3506_350684

/-- Given a triangle and three trapezoids formed by cuts parallel to its sides,
    this theorem proves the perimeter of the resulting small triangle. -/
theorem small_triangle_perimeter
  (original_perimeter : ℝ)
  (trapezoid1_perimeter trapezoid2_perimeter trapezoid3_perimeter : ℝ)
  (h1 : original_perimeter = 11)
  (h2 : trapezoid1_perimeter = 5)
  (h3 : trapezoid2_perimeter = 7)
  (h4 : trapezoid3_perimeter = 9) :
  trapezoid1_perimeter + trapezoid2_perimeter + trapezoid3_perimeter - original_perimeter = 10 :=
by sorry

end NUMINAMATH_CALUDE_small_triangle_perimeter_l3506_350684


namespace NUMINAMATH_CALUDE_mangoes_per_jar_l3506_350686

/-- Given the following conditions about Jordan's mangoes:
    - Jordan picked 54 mangoes
    - One-third of the mangoes were ripe
    - Two-thirds of the mangoes were unripe
    - Jordan kept 16 unripe mangoes
    - The remainder was given to his sister
    - His sister made 5 jars of pickled mangoes
    Prove that it takes 4 mangoes to fill one jar. -/
theorem mangoes_per_jar :
  ∀ (total_mangoes : ℕ)
    (ripe_fraction : ℚ)
    (unripe_fraction : ℚ)
    (kept_unripe : ℕ)
    (num_jars : ℕ),
  total_mangoes = 54 →
  ripe_fraction = 1/3 →
  unripe_fraction = 2/3 →
  kept_unripe = 16 →
  num_jars = 5 →
  (total_mangoes : ℚ) * unripe_fraction - kept_unripe = num_jars * 4 :=
by sorry

end NUMINAMATH_CALUDE_mangoes_per_jar_l3506_350686


namespace NUMINAMATH_CALUDE_equation_solution_l3506_350652

theorem equation_solution : ∃ m : ℤ, 3^4 - m = 4^3 + 2 ∧ m = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3506_350652


namespace NUMINAMATH_CALUDE_function_symmetry_implies_a_value_l3506_350683

theorem function_symmetry_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, 2*(1-x)^2 - a*(1-x) + 3 = 2*(1+x)^2 - a*(1+x) + 3) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_implies_a_value_l3506_350683


namespace NUMINAMATH_CALUDE_camp_cedar_counselor_ratio_l3506_350681

theorem camp_cedar_counselor_ratio : 
  ∀ (num_boys : ℕ) (num_girls : ℕ) (num_counselors : ℕ),
    num_boys = 40 →
    num_girls = 3 * num_boys →
    num_counselors = 20 →
    (num_boys + num_girls) / num_counselors = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_camp_cedar_counselor_ratio_l3506_350681


namespace NUMINAMATH_CALUDE_april_flower_sale_earnings_l3506_350618

theorem april_flower_sale_earnings 
  (rose_price : ℕ)
  (initial_roses : ℕ)
  (remaining_roses : ℕ)
  (h1 : rose_price = 7)
  (h2 : initial_roses = 9)
  (h3 : remaining_roses = 4) :
  (initial_roses - remaining_roses) * rose_price = 35 :=
by sorry

end NUMINAMATH_CALUDE_april_flower_sale_earnings_l3506_350618


namespace NUMINAMATH_CALUDE_determinant_equals_x_squared_plus_y_squared_l3506_350623

theorem determinant_equals_x_squared_plus_y_squared (x y : ℝ) : 
  Matrix.det !![1, x, y; 1, x - y, y; 1, x, y - x] = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equals_x_squared_plus_y_squared_l3506_350623


namespace NUMINAMATH_CALUDE_valid_rod_pairs_l3506_350616

def is_valid_polygon (a b c d e : ℕ) : Prop :=
  a + b + c + d > e ∧ a + b + c + e > d ∧ a + b + d + e > c ∧ 
  a + c + d + e > b ∧ b + c + d + e > a

def count_valid_pairs : ℕ → ℕ → ℕ → ℕ := sorry

theorem valid_rod_pairs : 
  let rod_lengths : List ℕ := List.range 50
  let selected_rods : List ℕ := [8, 12, 20]
  let remaining_rods : List ℕ := rod_lengths.filter (λ x => x ∉ selected_rods)
  count_valid_pairs 8 12 20 = 135 := by sorry

end NUMINAMATH_CALUDE_valid_rod_pairs_l3506_350616


namespace NUMINAMATH_CALUDE_stolen_newspapers_weight_l3506_350690

/-- Calculates the total weight of stolen newspapers in tons over a given number of weeks. -/
def total_weight_in_tons (weeks : ℕ) : ℚ :=
  let daily_papers : ℕ := 250
  let weekday_paper_weight : ℚ := 8 / 16
  let sunday_paper_weight : ℚ := 2 * weekday_paper_weight
  let weekdays_per_week : ℕ := 6
  let sundays_per_week : ℕ := 1
  let weekday_weight : ℚ := (weeks * weekdays_per_week * daily_papers * weekday_paper_weight)
  let sunday_weight : ℚ := (weeks * sundays_per_week * daily_papers * sunday_paper_weight)
  (weekday_weight + sunday_weight) / 2000

/-- Theorem stating that the total weight of stolen newspapers over 10 weeks is 5 tons. -/
theorem stolen_newspapers_weight : total_weight_in_tons 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_stolen_newspapers_weight_l3506_350690


namespace NUMINAMATH_CALUDE_adas_original_seat_l3506_350691

/-- Represents the seats in the movie theater. -/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends. -/
inductive Friend
| Ada
| Bea
| Ceci
| Dee
| Edie
| Fran

/-- Represents a seating arrangement. -/
def Arrangement := Friend → Seat

/-- Defines what it means for a seat to be an end seat. -/
def is_end_seat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.six

/-- Defines the movement of friends after Ada left. -/
def moved (initial final : Arrangement) : Prop :=
  (∃ s, initial Friend.Bea = s ∧ final Friend.Bea = Seat.six) ∧
  (initial Friend.Ceci = final Friend.Ceci) ∧
  (initial Friend.Dee = final Friend.Edie ∧ initial Friend.Edie = final Friend.Dee) ∧
  (∃ s t, initial Friend.Fran = s ∧ final Friend.Fran = t ∧ 
    (s = Seat.two ∧ t = Seat.one ∨
     s = Seat.three ∧ t = Seat.two ∨
     s = Seat.four ∧ t = Seat.three ∨
     s = Seat.five ∧ t = Seat.four ∨
     s = Seat.six ∧ t = Seat.five))

/-- The main theorem stating Ada's original seat. -/
theorem adas_original_seat (initial final : Arrangement) :
  moved initial final →
  is_end_seat (final Friend.Ada) →
  initial Friend.Ada = Seat.three :=
sorry

end NUMINAMATH_CALUDE_adas_original_seat_l3506_350691


namespace NUMINAMATH_CALUDE_fraction_C_is_simplest_l3506_350605

-- Define the fractions
def fraction_A (m n : ℚ) : ℚ := 2 * m / (10 * m * n)
def fraction_B (m n : ℚ) : ℚ := (m^2 - n^2) / (m + n)
def fraction_C (m n : ℚ) : ℚ := (m^2 + n^2) / (m + n)
def fraction_D (a : ℚ) : ℚ := 2 * a / a^2

-- Define what it means for a fraction to be in simplest form
def is_simplest_form (f : ℚ) : Prop := 
  ∀ (g : ℚ), g ≠ 1 → g ≠ -1 → f ≠ g * (↑(f.num) / ↑(f.den))

-- Theorem statement
theorem fraction_C_is_simplest : 
  ∀ (m n : ℚ), m + n ≠ 0 → is_simplest_form (fraction_C m n) := by sorry

end NUMINAMATH_CALUDE_fraction_C_is_simplest_l3506_350605


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3506_350634

theorem simplify_fraction_product : 24 * (3 / 4) * (2 / 11) * (5 / 8) = 45 / 22 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3506_350634


namespace NUMINAMATH_CALUDE_parallelogram_sides_l3506_350626

-- Define the triangle ABC
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)

-- Define the parallelogram BKLM
structure Parallelogram :=
  (BM : ℝ)
  (BK : ℝ)

-- Define the problem conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.AB = 18 ∧ t.BC = 12

def parallelogram_conditions (t : Triangle) (p : Parallelogram) : Prop :=
  -- Area of BKLM is 4/9 of the area of ABC
  p.BM * p.BK = (4/9) * (1/2) * t.AB * t.BC

-- Theorem statement
theorem parallelogram_sides (t : Triangle) (p : Parallelogram) :
  triangle_conditions t →
  parallelogram_conditions t p →
  ((p.BM = 8 ∧ p.BK = 6) ∨ (p.BM = 4 ∧ p.BK = 12)) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_sides_l3506_350626


namespace NUMINAMATH_CALUDE_negation_equivalence_l3506_350661

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3506_350661


namespace NUMINAMATH_CALUDE_congruence_problem_l3506_350698

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3506_350698


namespace NUMINAMATH_CALUDE_rectangle_uniquely_symmetric_l3506_350641

-- Define the properties
def axisymmetric (shape : Type) : Prop := sorry

def centrally_symmetric (shape : Type) : Prop := sorry

-- Define the shapes
def equilateral_triangle : Type := sorry
def rectangle : Type := sorry
def parallelogram : Type := sorry
def regular_pentagon : Type := sorry

-- Theorem statement
theorem rectangle_uniquely_symmetric :
  (axisymmetric equilateral_triangle ∧ centrally_symmetric equilateral_triangle) = False ∧
  (axisymmetric rectangle ∧ centrally_symmetric rectangle) = True ∧
  (axisymmetric parallelogram ∧ centrally_symmetric parallelogram) = False ∧
  (axisymmetric regular_pentagon ∧ centrally_symmetric regular_pentagon) = False :=
sorry

end NUMINAMATH_CALUDE_rectangle_uniquely_symmetric_l3506_350641


namespace NUMINAMATH_CALUDE_proportion_equality_l3506_350685

theorem proportion_equality (h : (7 : ℚ) / 11 * 66 = 42) : (13 : ℚ) / 17 * 289 = 221 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l3506_350685


namespace NUMINAMATH_CALUDE_art_club_officer_selection_l3506_350668

/-- Represents the Art Club with its members and officer selection rules -/
structure ArtClub where
  totalMembers : ℕ
  officerCount : ℕ
  andyIndex : ℕ
  breeIndex : ℕ
  carlosIndex : ℕ
  danaIndex : ℕ

/-- Calculates the number of ways to choose officers in the Art Club -/
def chooseOfficers (club : ArtClub) : ℕ :=
  sorry

/-- Theorem stating the correct number of ways to choose officers -/
theorem art_club_officer_selection (club : ArtClub) :
  club.totalMembers = 30 ∧
  club.officerCount = 4 ∧
  club.andyIndex ≠ club.breeIndex ∧
  club.carlosIndex ≠ club.danaIndex ∧
  club.andyIndex ≤ club.totalMembers ∧
  club.breeIndex ≤ club.totalMembers ∧
  club.carlosIndex ≤ club.totalMembers ∧
  club.danaIndex ≤ club.totalMembers →
  chooseOfficers club = 369208 :=
sorry

end NUMINAMATH_CALUDE_art_club_officer_selection_l3506_350668


namespace NUMINAMATH_CALUDE_custom_mult_solution_l3506_350602

/-- Custom multiplication operation -/
def star_mult (a b : ℝ) : ℝ := a * b + a + b

/-- Theorem stating that if 3 * x = 27 under the custom multiplication, then x = 6 -/
theorem custom_mult_solution :
  (∀ a b : ℝ, star_mult a b = a * b + a + b) →
  star_mult 3 x = 27 →
  x = 6 := by sorry

end NUMINAMATH_CALUDE_custom_mult_solution_l3506_350602


namespace NUMINAMATH_CALUDE_angle_D_measure_l3506_350615

/-- 
Given a geometric figure with angles A, B, C, D, and E, where:
1. The sum of angles A and B is 140 degrees
2. Angle C is equal to angle D
3. The sum of angles C, D, and E is 180 degrees

This theorem proves that the measure of angle D is 20 degrees.
-/
theorem angle_D_measure (A B C D E : ℝ) 
  (sum_AB : A + B = 140)
  (C_eq_D : C = D)
  (sum_CDE : C + D + E = 180) :
  D = 20 := by sorry

end NUMINAMATH_CALUDE_angle_D_measure_l3506_350615


namespace NUMINAMATH_CALUDE_sandy_paint_area_l3506_350676

/-- The area Sandy needs to paint on her bedroom wall -/
def area_to_paint (wall_height wall_length window_width window_height : ℝ) : ℝ :=
  wall_height * wall_length - window_width * window_height

/-- Theorem stating the area Sandy needs to paint -/
theorem sandy_paint_area :
  area_to_paint 9 12 2 4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sandy_paint_area_l3506_350676


namespace NUMINAMATH_CALUDE_range_of_a_l3506_350619

/-- Given sets A and B, and the condition that A is not a subset of B, 
    prove that the range of values for a is (1, 5) -/
theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := {x | 2*a < x ∧ x < a + 5}
  let B : Set ℝ := {x | x < 6}
  ¬(A ⊆ B) → a ∈ Set.Ioo 1 5 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l3506_350619


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l3506_350687

/-- Represents the total population size -/
def total_population : ℕ := 260

/-- Represents the elderly population size -/
def elderly_population : ℕ := 60

/-- Represents the number of elderly people selected in the sample -/
def elderly_sample : ℕ := 3

/-- Represents the total sample size -/
def total_sample : ℕ := 13

/-- Theorem stating that the total sample size is correct given the stratified sampling conditions -/
theorem stratified_sampling_size :
  (elderly_sample : ℚ) / elderly_population = (total_sample : ℚ) / total_population :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l3506_350687


namespace NUMINAMATH_CALUDE_harry_milk_bottles_l3506_350675

theorem harry_milk_bottles (initial : ℕ) (jason_bought : ℕ) (final : ℕ) 
  (h1 : initial = 35) 
  (h2 : jason_bought = 5) 
  (h3 : final = 24) : 
  initial - jason_bought - final = 6 := by
  sorry

#check harry_milk_bottles

end NUMINAMATH_CALUDE_harry_milk_bottles_l3506_350675


namespace NUMINAMATH_CALUDE_a_range_l3506_350633

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- State the theorem
theorem a_range (a : ℝ) :
  (f a 3 < f a 4) ∧
  (∀ n : ℕ, n ≥ 8 → f a n > f a (n + 1)) →
  -1/7 < a ∧ a < -1/17 :=
sorry

end NUMINAMATH_CALUDE_a_range_l3506_350633


namespace NUMINAMATH_CALUDE_manuels_savings_l3506_350666

/-- Calculates the total savings amount given initial deposit, weekly savings, and number of weeks -/
def totalSavings (initialDeposit weeklyAmount numWeeks : ℕ) : ℕ :=
  initialDeposit + weeklyAmount * numWeeks

/-- Theorem: Manuel's total savings after 19 weeks is $500 -/
theorem manuels_savings :
  totalSavings 177 17 19 = 500 := by
  sorry

end NUMINAMATH_CALUDE_manuels_savings_l3506_350666


namespace NUMINAMATH_CALUDE_initial_peanuts_count_l3506_350628

/-- 
Given a box of peanuts, prove that the initial number of peanuts was 10,
when 8 peanuts were added and the final count is 18.
-/
theorem initial_peanuts_count (initial final added : ℕ) 
  (h1 : added = 8)
  (h2 : final = 18)
  (h3 : final = initial + added) : 
  initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_peanuts_count_l3506_350628


namespace NUMINAMATH_CALUDE_paint_more_expensive_than_wallpaper_l3506_350654

/-- Proves that a can of paint costs more than a roll of wallpaper given specific purchase scenarios -/
theorem paint_more_expensive_than_wallpaper 
  (wallpaper_cost movie_ticket_cost paint_cost : ℝ) 
  (wallpaper_cost_positive : 0 < wallpaper_cost)
  (paint_cost_positive : 0 < paint_cost)
  (movie_ticket_cost_positive : 0 < movie_ticket_cost)
  (equal_spending : 4 * wallpaper_cost + 4 * paint_cost = 7 * wallpaper_cost + 2 * paint_cost + movie_ticket_cost) :
  paint_cost > wallpaper_cost := by
  sorry


end NUMINAMATH_CALUDE_paint_more_expensive_than_wallpaper_l3506_350654
