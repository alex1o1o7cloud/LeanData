import Mathlib

namespace NUMINAMATH_CALUDE_susan_bought_sixty_peaches_l772_77261

/-- Represents the number of peaches in Susan's knapsack -/
def knapsack_peaches : ℕ := 12

/-- Represents the number of cloth bags Susan has -/
def num_cloth_bags : ℕ := 2

/-- Calculates the number of peaches in each cloth bag -/
def peaches_per_cloth_bag : ℕ := 2 * knapsack_peaches

/-- Calculates the total number of peaches Susan bought -/
def total_peaches : ℕ := num_cloth_bags * peaches_per_cloth_bag + knapsack_peaches

/-- Theorem stating that Susan bought 60 peaches in total -/
theorem susan_bought_sixty_peaches : total_peaches = 60 := by
  sorry

end NUMINAMATH_CALUDE_susan_bought_sixty_peaches_l772_77261


namespace NUMINAMATH_CALUDE_limit_is_nonzero_real_l772_77253

noncomputable def f (x : ℝ) : ℝ := x^(5/3) * ((x + 1)^(1/3) + (x - 1)^(1/3) - 2 * x^(1/3))

theorem limit_is_nonzero_real : ∃ (L : ℝ), L ≠ 0 ∧ Filter.Tendsto f Filter.atTop (nhds L) := by
  sorry

end NUMINAMATH_CALUDE_limit_is_nonzero_real_l772_77253


namespace NUMINAMATH_CALUDE_monotonic_quadratic_constraint_l772_77295

/-- A function f is monotonic on an interval [a, b] if and only if
    its derivative f' does not change sign on (a, b) -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ (Set.Icc a b), (∀ y ∈ (Set.Icc a b), x ≤ y → f x ≤ f y) ∨
                       (∀ y ∈ (Set.Icc a b), x ≤ y → f y ≤ f x)

/-- The quadratic function f(x) = 4x² - kx - 8 -/
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

theorem monotonic_quadratic_constraint (k : ℝ) :
  IsMonotonic (f k) 5 8 ↔ k ∈ Set.Iic 40 ∪ Set.Ici 64 := by
  sorry

#check monotonic_quadratic_constraint

end NUMINAMATH_CALUDE_monotonic_quadratic_constraint_l772_77295


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l772_77289

theorem square_garden_perimeter (side : ℝ) (area perimeter : ℝ) : 
  area = side^2 → 
  perimeter = 4 * side → 
  area = 100 → 
  area = 2 * perimeter + 20 → 
  perimeter = 40 := by sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l772_77289


namespace NUMINAMATH_CALUDE_certain_part_of_number_l772_77264

theorem certain_part_of_number (x y : ℝ) : 
  x = 1925 → 
  (1 / 7) * x = y + 100 → 
  y = 175 := by
  sorry

end NUMINAMATH_CALUDE_certain_part_of_number_l772_77264


namespace NUMINAMATH_CALUDE_fraction_simplification_l772_77228

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l772_77228


namespace NUMINAMATH_CALUDE_rectangle_sides_l772_77280

theorem rectangle_sides (x y : ℝ) : 
  (2 * (x + y) = 124) →  -- Perimeter of rectangle is 124 cm
  (4 * Real.sqrt ((x/2)^2 + ((124/2 - x)/2)^2) = 100) →  -- Perimeter of rhombus is 100 cm
  ((x = 48 ∧ y = 14) ∨ (x = 14 ∧ y = 48)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_sides_l772_77280


namespace NUMINAMATH_CALUDE_cost_750_candies_l772_77206

/-- The cost of buying a given number of chocolate candies with a possible discount -/
def total_cost (candies_per_box : ℕ) (box_cost : ℚ) (num_candies : ℕ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let num_boxes := (num_candies + candies_per_box - 1) / candies_per_box
  let cost_before_discount := num_boxes * box_cost
  let discount := if num_candies > discount_threshold then discount_rate * cost_before_discount else 0
  cost_before_discount - discount

/-- The total cost to buy 750 chocolate candies is $180 -/
theorem cost_750_candies :
  total_cost 30 8 750 (1/10) 500 = 180 := by
  sorry

end NUMINAMATH_CALUDE_cost_750_candies_l772_77206


namespace NUMINAMATH_CALUDE_perpendicular_line_proof_l772_77251

def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 8 = 0

theorem perpendicular_line_proof :
  (∀ x y : ℝ, perpendicular_line x y → given_line x y → (x + 2) * (y - 3) = 0) ∧
  (∀ x y : ℝ, given_line x y → perpendicular_line x y → (x + 2) * (2 * x + y) + (y - 3) * (x + 2 * y) = 0) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_proof_l772_77251


namespace NUMINAMATH_CALUDE_hemisphere_with_spire_surface_area_l772_77290

/-- The total surface area of a hemisphere with a conical spire -/
theorem hemisphere_with_spire_surface_area :
  let r : ℝ := 8  -- radius of hemisphere
  let h : ℝ := 10 -- height of conical spire
  let l : ℝ := Real.sqrt (r^2 + h^2)  -- slant height of cone
  let area_base : ℝ := π * r^2  -- area of circular base
  let area_hemisphere : ℝ := 2 * π * r^2  -- surface area of hemisphere
  let area_cone : ℝ := π * r * l  -- lateral surface area of cone
  area_base + area_hemisphere + area_cone = 192 * π + 8 * π * Real.sqrt 164 :=
by sorry


end NUMINAMATH_CALUDE_hemisphere_with_spire_surface_area_l772_77290


namespace NUMINAMATH_CALUDE_log_comparison_l772_77287

theorem log_comparison : Real.log 4 / Real.log 3 > Real.log 5 / Real.log 4 := by sorry

end NUMINAMATH_CALUDE_log_comparison_l772_77287


namespace NUMINAMATH_CALUDE_initial_number_problem_l772_77267

theorem initial_number_problem : 
  let x : ℚ := 10
  ((x + 14) * 14 - 24) / 24 = 13 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_problem_l772_77267


namespace NUMINAMATH_CALUDE_vasyas_premium_will_increase_l772_77245

/-- Represents a car insurance policy -/
structure CarInsurancePolicy where
  premium : ℝ
  hadAccident : Bool

/-- Represents an insurance company -/
class InsuranceCompany where
  renewPolicy : CarInsurancePolicy → CarInsurancePolicy

/-- Axiom: Insurance companies increase premiums for policies with accidents -/
axiom premium_increase_after_accident (company : InsuranceCompany) (policy : CarInsurancePolicy) :
  policy.hadAccident → (company.renewPolicy policy).premium > policy.premium

/-- Theorem: Vasya's insurance premium will increase after his car accident -/
theorem vasyas_premium_will_increase (company : InsuranceCompany) (vasyas_policy : CarInsurancePolicy) 
    (h_accident : vasyas_policy.hadAccident) : 
  (company.renewPolicy vasyas_policy).premium > vasyas_policy.premium :=
by
  sorry


end NUMINAMATH_CALUDE_vasyas_premium_will_increase_l772_77245


namespace NUMINAMATH_CALUDE_quadratic_coefficients_4x2_eq_3_l772_77214

/-- Given a quadratic equation ax^2 + bx + c = 0, returns the tuple (a, b, c) -/
def quadratic_coefficients (f : ℝ → ℝ) : ℝ × ℝ × ℝ := sorry

theorem quadratic_coefficients_4x2_eq_3 :
  quadratic_coefficients (fun x => 4 * x^2 - 3) = (4, 0, -3) := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_4x2_eq_3_l772_77214


namespace NUMINAMATH_CALUDE_yellow_pairs_count_l772_77260

theorem yellow_pairs_count (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) 
  (total_pairs : ℕ) (blue_pairs : ℕ) :
  blue_students = 60 →
  yellow_students = 84 →
  total_students = 144 →
  total_pairs = 72 →
  blue_pairs = 28 →
  blue_students + yellow_students = total_students →
  2 * total_pairs = total_students →
  ∃ (yellow_pairs : ℕ), yellow_pairs = 40 ∧ 
    yellow_pairs + blue_pairs + (total_students - 2 * (yellow_pairs + blue_pairs)) / 2 = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_yellow_pairs_count_l772_77260


namespace NUMINAMATH_CALUDE_negation_of_proposition_l772_77283

theorem negation_of_proposition (p : Prop) : 
  (¬(∃ x₀ : ℝ, x₀ ∈ Set.Icc (-3) 3 ∧ x₀^2 + 2*x₀ + 1 ≤ 0)) ↔ 
  (∀ x : ℝ, x ∈ Set.Icc (-3) 3 → x^2 + 2*x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l772_77283


namespace NUMINAMATH_CALUDE_mathematician_project_time_l772_77270

theorem mathematician_project_time (project1 : ℕ) (project2 : ℕ) (daily_questions : ℕ) : 
  project1 = 518 → project2 = 476 → daily_questions = 142 → 
  (project1 + project2) / daily_questions = 7 := by
  sorry

end NUMINAMATH_CALUDE_mathematician_project_time_l772_77270


namespace NUMINAMATH_CALUDE_circle_rooted_polynomial_ab_neq_nine_l772_77226

/-- A polynomial of degree 4 with four distinct roots on a circle in the complex plane -/
structure CircleRootedPolynomial where
  a : ℂ
  b : ℂ
  roots_distinct : True  -- Placeholder for the distinctness condition
  roots_on_circle : True -- Placeholder for the circle condition

/-- The theorem stating that for a polynomial with four distinct roots on a circle, ab ≠ 9 -/
theorem circle_rooted_polynomial_ab_neq_nine (P : CircleRootedPolynomial) : P.a * P.b ≠ 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_rooted_polynomial_ab_neq_nine_l772_77226


namespace NUMINAMATH_CALUDE_max_take_home_pay_l772_77269

/-- The income that yields the maximum take-home pay given a specific tax rate -/
theorem max_take_home_pay :
  let income : ℝ → ℝ := λ x => 1000 * x
  let tax_rate : ℝ → ℝ := λ x => 0.02 * x
  let tax : ℝ → ℝ := λ x => (tax_rate x) * (income x)
  let take_home_pay : ℝ → ℝ := λ x => (income x) - (tax x)
  ∃ x : ℝ, x = 25 ∧ ∀ y : ℝ, take_home_pay y ≤ take_home_pay x :=
by sorry

end NUMINAMATH_CALUDE_max_take_home_pay_l772_77269


namespace NUMINAMATH_CALUDE_pizza_sharing_l772_77200

theorem pizza_sharing (total_slices : ℕ) (difference : ℕ) (y : ℕ) : 
  total_slices = 10 →
  difference = 2 →
  y + (y + difference) = total_slices →
  y = 4 := by
sorry

end NUMINAMATH_CALUDE_pizza_sharing_l772_77200


namespace NUMINAMATH_CALUDE_parabola_directrix_intersection_l772_77277

/-- The parabola equation: x^2 = 4y -/
def parabola_equation (x y : ℝ) : Prop := x^2 = 4*y

/-- The directrix equation for a parabola with equation x^2 = 4ay -/
def directrix_equation (a y : ℝ) : Prop := y = -a

/-- The y-axis equation -/
def y_axis (x : ℝ) : Prop := x = 0

theorem parabola_directrix_intersection :
  ∃ (a : ℝ), a = 1 ∧
  (∀ x y : ℝ, parabola_equation x y ↔ x^2 = 4*a*y) ∧
  (∃ y : ℝ, directrix_equation a y ∧ y_axis 0 ∧ y = -1) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_intersection_l772_77277


namespace NUMINAMATH_CALUDE_tan_arccos_three_fifths_l772_77272

theorem tan_arccos_three_fifths :
  Real.tan (Real.arccos (3/5)) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_arccos_three_fifths_l772_77272


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l772_77247

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- arithmetic sequence definition
  q > 1 →  -- common ratio condition
  a 1 + a 4 = 9 →  -- first condition
  a 2 * a 3 = 8 →  -- second condition
  (a 2015 + a 2016) / (a 2013 + a 2014) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l772_77247


namespace NUMINAMATH_CALUDE_geometric_sequence_a8_l772_77249

/-- A sequence where a_n + 2 forms a geometric sequence -/
def IsGeometricPlus2 (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, (a (n + 1) + 2) = (a n + 2) * q

theorem geometric_sequence_a8 (a : ℕ → ℝ) 
  (h_geom : IsGeometricPlus2 a) 
  (h_a2 : a 2 = -1) 
  (h_a4 : a 4 = 2) : 
  a 8 = 62 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a8_l772_77249


namespace NUMINAMATH_CALUDE_correct_average_calculation_l772_77219

/-- Proves that the correct average is 22 given the conditions of the problem -/
theorem correct_average_calculation (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) 
  (hn : n = 10) 
  (hinitial : initial_avg = 18) 
  (hincorrect : incorrect_num = 26)
  (hcorrect : correct_num = 66) :
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 22 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l772_77219


namespace NUMINAMATH_CALUDE_prob_exactly_four_questions_value_l772_77204

/-- The probability of correctly answering a single question -/
def p : ℝ := 0.8

/-- The number of questions in the competition -/
def n : ℕ := 5

/-- The event that a contestant exactly answers 4 questions before advancing -/
def exactly_four_questions (outcomes : Fin 4 → Bool) : Prop :=
  outcomes 1 = false ∧ outcomes 2 = true ∧ outcomes 3 = true

/-- The probability of the event that a contestant exactly answers 4 questions before advancing -/
def prob_exactly_four_questions : ℝ :=
  (1 - p) * p * p

theorem prob_exactly_four_questions_value :
  prob_exactly_four_questions = 0.128 := by
  sorry


end NUMINAMATH_CALUDE_prob_exactly_four_questions_value_l772_77204


namespace NUMINAMATH_CALUDE_elizabeth_subtraction_l772_77299

theorem elizabeth_subtraction (n : ℕ) (h1 : n = 50) (h2 : n^2 + 101 = (n+1)^2) : n^2 - (n-1)^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_subtraction_l772_77299


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l772_77233

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let line := fun (x y : ℝ) => x / a + y / b = 1
  let foci_distance_sum := 4 * c / 5
  let eccentricity := c / a
  (∀ x y, hyperbola x y → line x y) → 
  (foci_distance_sum = 2 * b) →
  eccentricity = 5 * Real.sqrt 21 / 21 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l772_77233


namespace NUMINAMATH_CALUDE_parabola_equation_and_range_l772_77279

/-- A parabola with equation y = x^2 - 2mx + m^2 - 1 -/
def Parabola (m : ℝ) (x y : ℝ) : Prop :=
  y = x^2 - 2*m*x + m^2 - 1

/-- The parabola intersects the y-axis at (0, 3) -/
def IntersectsYAxisAt3 (m : ℝ) : Prop :=
  Parabola m 0 3

/-- The vertex of the parabola is in the fourth quadrant -/
def VertexInFourthQuadrant (m : ℝ) : Prop :=
  let x_vertex := m  -- x-coordinate of vertex is m for this parabola
  let y_vertex := -1  -- y-coordinate of vertex is -1 for this parabola
  x_vertex > 0 ∧ y_vertex < 0

theorem parabola_equation_and_range (m : ℝ) 
  (h1 : IntersectsYAxisAt3 m) 
  (h2 : VertexInFourthQuadrant m) :
  (∀ x y, Parabola m x y ↔ y = x^2 - 4*x + 3) ∧
  (∀ x y, 0 ≤ x ∧ x ≤ 3 ∧ Parabola m x y → -1 ≤ y ∧ y ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_and_range_l772_77279


namespace NUMINAMATH_CALUDE_point_on_parabola_l772_77265

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 1

-- Define the theorem
theorem point_on_parabola (y w : ℝ) :
  parabola 3 = y → w = 2 → y = 4 * w := by
  sorry

end NUMINAMATH_CALUDE_point_on_parabola_l772_77265


namespace NUMINAMATH_CALUDE_angle_measure_proof_l772_77298

theorem angle_measure_proof : Real.arccos (Real.sin (19 * π / 180)) = 71 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l772_77298


namespace NUMINAMATH_CALUDE_solution_is_correct_l772_77256

def solution_set : Set (Nat × Nat × Nat × Nat) :=
  {(8, 3, 3, 1), (5, 4, 3, 1), (3, 2, 2, 2), (7, 6, 2, 1), (9, 5, 2, 1), (15, 4, 2, 1),
   (1, 1, 1, 7), (2, 1, 1, 5), (3, 2, 1, 3), (8, 3, 1, 2), (5, 4, 1, 2)}

def satisfies_conditions (x y z t : Nat) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧
  x * y * z = Nat.factorial t ∧
  (x + 1) * (y + 1) * (z + 1) = Nat.factorial (t + 1)

theorem solution_is_correct :
  ∀ x y z t, (x, y, z, t) ∈ solution_set ↔ satisfies_conditions x y z t :=
sorry

end NUMINAMATH_CALUDE_solution_is_correct_l772_77256


namespace NUMINAMATH_CALUDE_range_of_a_l772_77254

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, Real.exp x - a * Real.log (a * x - a) + a > 0) →
  a > 0 →
  0 < a ∧ a < Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l772_77254


namespace NUMINAMATH_CALUDE_line_y_intercept_l772_77244

/-- Given a line with slope 4 passing through the point (50, 300), prove that its y-intercept is 100. -/
theorem line_y_intercept (m : ℝ) (x y b : ℝ) :
  m = 4 →
  x = 50 →
  y = 300 →
  y = m * x + b →
  b = 100 := by
sorry

end NUMINAMATH_CALUDE_line_y_intercept_l772_77244


namespace NUMINAMATH_CALUDE_midpoint_property_l772_77248

/-- Given two points P and Q in the plane, their midpoint R satisfies 3x - 2y = -15 --/
theorem midpoint_property (P Q R : ℝ × ℝ) : 
  P = (-8, 15) → 
  Q = (6, -3) → 
  R.1 = (P.1 + Q.1) / 2 → 
  R.2 = (P.2 + Q.2) / 2 → 
  3 * R.1 - 2 * R.2 = -15 := by
sorry

end NUMINAMATH_CALUDE_midpoint_property_l772_77248


namespace NUMINAMATH_CALUDE_football_progress_l772_77255

def round1 : Int := -5
def round2 : Int := 9
def round3 : Int := -12
def round4 : Int := 17
def round5 : Int := -15
def round6 : Int := 24
def round7 : Int := -7

def overall_progress : Int := round1 + round2 + round3 + round4 + round5 + round6 + round7

theorem football_progress : overall_progress = 11 := by
  sorry

end NUMINAMATH_CALUDE_football_progress_l772_77255


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l772_77257

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x < 0, f x ≥ 0) ↔ (∀ x < 0, f x < 0) :=
by
  sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x < 0, x^2 - 3*x + 1 ≥ 0) ↔ (∀ x < 0, x^2 - 3*x + 1 < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l772_77257


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l772_77224

theorem sum_of_x_and_y (x y : ℝ) 
  (hx : |x| = 1) 
  (hy : |y| = 2) 
  (hxy : x * y > 0) : 
  x + y = 3 ∨ x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l772_77224


namespace NUMINAMATH_CALUDE_lace_per_ruffle_length_l772_77216

/-- Proves that given the conditions of Carolyn's dress trimming,
    the length of lace used for each ruffle is 20 cm. -/
theorem lace_per_ruffle_length
  (cuff_length : ℝ)
  (hem_length : ℝ)
  (num_cuffs : ℕ)
  (num_ruffles : ℕ)
  (lace_cost_per_meter : ℝ)
  (total_spent : ℝ)
  (h1 : cuff_length = 50)
  (h2 : hem_length = 300)
  (h3 : num_cuffs = 2)
  (h4 : num_ruffles = 5)
  (h5 : lace_cost_per_meter = 6)
  (h6 : total_spent = 36)
  : (total_spent / lace_cost_per_meter * 100 -
     (num_cuffs * cuff_length + hem_length / 3 + hem_length)) / num_ruffles = 20 := by
  sorry

end NUMINAMATH_CALUDE_lace_per_ruffle_length_l772_77216


namespace NUMINAMATH_CALUDE_inequality_proof_l772_77215

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  a^4 * b + b^4 * c + c^4 * a > a * b^4 + b * c^4 + c * a^4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l772_77215


namespace NUMINAMATH_CALUDE_smallest_x_value_l772_77230

/-- Given the equation x|x| = 3x + k and the inequality x + 2 ≤ 3,
    the smallest value of x that satisfies these conditions is -2 when k = 2. -/
theorem smallest_x_value (x : ℝ) (k : ℝ) : 
  (x * abs x = 3 * x + k) → 
  (x + 2 ≤ 3) → 
  (k = 2) →
  (∀ y : ℝ, (y * abs y = 3 * y + k) → (y + 2 ≤ 3) → (x ≤ y)) →
  x = -2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_value_l772_77230


namespace NUMINAMATH_CALUDE_binomial_coefficient_39_5_l772_77221

theorem binomial_coefficient_39_5 : 
  let n : ℕ := 39
  let binomial := n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / (2 * 3 * 4 * 5)
  binomial = 575757 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_39_5_l772_77221


namespace NUMINAMATH_CALUDE_nonagon_area_theorem_l772_77201

/-- Represents a right triangle with regular nonagons on its sides -/
structure RightTriangleWithNonagons where
  /-- Length of the hypotenuse -/
  a : ℝ
  /-- Length of one cathetus -/
  b : ℝ
  /-- Length of the other cathetus -/
  c : ℝ
  /-- Area of the nonagon on the hypotenuse -/
  A₁ : ℝ
  /-- Area of the nonagon on one cathetus -/
  A₂ : ℝ
  /-- Area of the nonagon on the other cathetus -/
  A₃ : ℝ
  /-- The triangle is a right triangle -/
  right_triangle : a^2 = b^2 + c^2
  /-- The areas of nonagons are proportional to the squares of the sides -/
  proportional_areas : A₁ / a^2 = A₂ / b^2 ∧ A₁ / a^2 = A₃ / c^2

/-- The main theorem -/
theorem nonagon_area_theorem (t : RightTriangleWithNonagons) 
    (h₁ : t.A₁ = 2019) (h₂ : t.A₂ = 1602) : t.A₃ = 417 := by
  sorry


end NUMINAMATH_CALUDE_nonagon_area_theorem_l772_77201


namespace NUMINAMATH_CALUDE_flashlight_battery_test_l772_77291

/-- Represents the minimum number of pairs to test to guarantee finding working batteries -/
def min_pairs_to_test (total : ℕ) (working : ℕ) (required : ℕ) : ℕ :=
  sorry

theorem flashlight_battery_test :
  min_pairs_to_test 12 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_flashlight_battery_test_l772_77291


namespace NUMINAMATH_CALUDE_sunny_subsets_l772_77241

theorem sunny_subsets (m n : ℕ) (S : Finset ℕ) (h1 : n ≥ m) (h2 : m ≥ 2) (h3 : Finset.card S = n) :
  ∃ T : Finset (Finset ℕ), (∀ X ∈ T, X ⊆ S ∧ m ∣ (Finset.sum X id)) ∧ Finset.card T ≥ 2^(n - m + 1) :=
sorry

end NUMINAMATH_CALUDE_sunny_subsets_l772_77241


namespace NUMINAMATH_CALUDE_one_fourth_of_six_point_three_l772_77203

theorem one_fourth_of_six_point_three (x : ℚ) : x = 6.3 / 4 → x = 63 / 40 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_six_point_three_l772_77203


namespace NUMINAMATH_CALUDE_fifteenth_digit_of_sum_one_eighth_one_sixth_l772_77210

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sumDecimalRepresentations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- Theorem: The 15th digit after the decimal point in the sum of 1/8 and 1/6 is 6 -/
theorem fifteenth_digit_of_sum_one_eighth_one_sixth : 
  sumDecimalRepresentations (1/8) (1/6) 15 = 6 := by sorry

end NUMINAMATH_CALUDE_fifteenth_digit_of_sum_one_eighth_one_sixth_l772_77210


namespace NUMINAMATH_CALUDE_pump_operations_proof_l772_77218

/-- The fraction of air remaining after one pump operation -/
def pump_efficiency : ℝ := 0.5

/-- The target fraction of air remaining -/
def target_fraction : ℝ := 0.001

/-- The minimum number of pump operations needed to reach the target fraction -/
def min_operations : ℕ := 10

theorem pump_operations_proof :
  (∀ n : ℕ, n < min_operations → (pump_efficiency ^ n : ℝ) > target_fraction) ∧
  (pump_efficiency ^ min_operations : ℝ) ≤ target_fraction :=
sorry

end NUMINAMATH_CALUDE_pump_operations_proof_l772_77218


namespace NUMINAMATH_CALUDE_aquarium_purchase_cost_l772_77225

/-- Calculates the total cost of an aquarium purchase with given discounts and tax rates -/
theorem aquarium_purchase_cost 
  (original_price : ℝ)
  (aquarium_discount : ℝ)
  (coupon_discount : ℝ)
  (additional_items_cost : ℝ)
  (aquarium_tax_rate : ℝ)
  (other_items_tax_rate : ℝ)
  (h1 : original_price = 120)
  (h2 : aquarium_discount = 0.5)
  (h3 : coupon_discount = 0.1)
  (h4 : additional_items_cost = 75)
  (h5 : aquarium_tax_rate = 0.05)
  (h6 : other_items_tax_rate = 0.08) :
  let discounted_price := original_price * (1 - aquarium_discount)
  let final_aquarium_price := discounted_price * (1 - coupon_discount)
  let aquarium_tax := final_aquarium_price * aquarium_tax_rate
  let other_items_tax := additional_items_cost * other_items_tax_rate
  let total_cost := final_aquarium_price + aquarium_tax + additional_items_cost + other_items_tax
  total_cost = 137.70 := by
sorry


end NUMINAMATH_CALUDE_aquarium_purchase_cost_l772_77225


namespace NUMINAMATH_CALUDE_flip_ratio_l772_77240

/-- The number of flips in a triple-flip -/
def tripleFlip : ℕ := 3

/-- The number of flips in a double-flip -/
def doubleFlip : ℕ := 2

/-- The number of triple-flips Jen performed -/
def jenFlips : ℕ := 16

/-- The number of double-flips Tyler performed -/
def tylerFlips : ℕ := 12

/-- Theorem stating that the ratio of Tyler's flips to Jen's flips is 1:2 -/
theorem flip_ratio :
  (tylerFlips * doubleFlip : ℚ) / (jenFlips * tripleFlip) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_flip_ratio_l772_77240


namespace NUMINAMATH_CALUDE_find_b_value_l772_77246

theorem find_b_value (x y b : ℝ) 
  (eq1 : 7^(3*x - 1) * b^(4*y - 3) = 49^x * 27^y)
  (eq2 : x + y = 4) : 
  b = 3 := by sorry

end NUMINAMATH_CALUDE_find_b_value_l772_77246


namespace NUMINAMATH_CALUDE_multiples_of_four_between_70_and_300_l772_77222

theorem multiples_of_four_between_70_and_300 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n > 70 ∧ n < 300) (Finset.range 300)).card = 57 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_between_70_and_300_l772_77222


namespace NUMINAMATH_CALUDE_natural_number_equality_l772_77231

def divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem natural_number_equality (a b : ℕ) 
  (h : ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, divisible (a^(n+1) + b^(n+1)) (a^n + b^n)) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_natural_number_equality_l772_77231


namespace NUMINAMATH_CALUDE_a_plus_b_minus_c_power_2004_l772_77207

theorem a_plus_b_minus_c_power_2004 (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = a*b + b*c + a*c) 
  (h2 : a = 1) : 
  (a + b - c)^2004 = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_minus_c_power_2004_l772_77207


namespace NUMINAMATH_CALUDE_f_minimum_F_monotonicity_intersection_property_l772_77239

noncomputable section

def f (x : ℝ) : ℝ := x * (1 + Real.log x)

def f' (x : ℝ) : ℝ := Real.log x + 2

def F (a : ℝ) (x : ℝ) : ℝ := a * x^2 + f' x

theorem f_minimum (x : ℝ) (hx : x > 0) :
  f x ≥ -1 / Real.exp 2 ∧ 
  f (1 / Real.exp 2) = -1 / Real.exp 2 :=
sorry

theorem F_monotonicity (a : ℝ) (x : ℝ) (hx : x > 0) :
  (a ≥ 0 → ∀ y > 0, x < y → F a x < F a y) ∧
  (a < 0 → ∃ c > 0, (∀ y, 0 < y ∧ y < c → F a x < F a y) ∧
                    (∀ y > c, F a y < F a x)) :=
sorry

theorem intersection_property (k x₁ x₂ : ℝ) (hx : 0 < x₁ ∧ x₁ < x₂) :
  k = (f' x₂ - f' x₁) / (x₂ - x₁) →
  x₁ < 1 / k ∧ 1 / k < x₂ :=
sorry

end NUMINAMATH_CALUDE_f_minimum_F_monotonicity_intersection_property_l772_77239


namespace NUMINAMATH_CALUDE_wage_decrease_increase_l772_77276

theorem wage_decrease_increase (original : ℝ) (h : original > 0) :
  let decreased := original * 0.5
  let increased := decreased * 1.5
  increased = original * 0.75 :=
by sorry

end NUMINAMATH_CALUDE_wage_decrease_increase_l772_77276


namespace NUMINAMATH_CALUDE_complex_number_values_l772_77243

theorem complex_number_values (z : ℂ) (a : ℝ) :
  z = (4 + 2*I) / (a + I) → Complex.abs z = Real.sqrt 10 →
  z = 3 - I ∨ z = -1 - 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_values_l772_77243


namespace NUMINAMATH_CALUDE_negation_of_divisible_by_5_is_odd_l772_77275

theorem negation_of_divisible_by_5_is_odd :
  (¬ ∀ n : ℤ, n % 5 = 0 → Odd n) ↔ (∃ n : ℤ, n % 5 = 0 ∧ ¬ Odd n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_divisible_by_5_is_odd_l772_77275


namespace NUMINAMATH_CALUDE_min_value_at_three_l772_77288

/-- The function f(y) = 3y^2 - 18y + 7 -/
def f (y : ℝ) : ℝ := 3 * y^2 - 18 * y + 7

/-- Theorem stating that the minimum value of f occurs when y = 3 -/
theorem min_value_at_three :
  ∃ (y_min : ℝ), ∀ (y : ℝ), f y ≥ f y_min ∧ y_min = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_at_three_l772_77288


namespace NUMINAMATH_CALUDE_distribute_5_8_l772_77296

/-- The number of ways to distribute n different items into m boxes with at most one item per box -/
def distribute (n m : ℕ) : ℕ :=
  (m - n + 1).factorial * (m.choose n)

/-- Theorem: The number of ways to distribute 5 different items into 8 boxes
    with at most one item per box is 6720 -/
theorem distribute_5_8 : distribute 5 8 = 6720 := by
  sorry

#eval distribute 5 8

end NUMINAMATH_CALUDE_distribute_5_8_l772_77296


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l772_77271

-- Define the polynomial
def p (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 12 * x - 12

-- Define the factors
def factor1 (x : ℝ) : ℝ := x - 2
def factor2 (x : ℝ) : ℝ := 3 * x^2 - 4

-- Theorem statement
theorem polynomial_divisibility :
  (∃ q1 q2 : ℝ → ℝ, ∀ x, p x = factor1 x * q1 x ∧ p x = factor2 x * q2 x) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l772_77271


namespace NUMINAMATH_CALUDE_inverse_square_function_l772_77294

/-- A function that varies inversely as the square of its input -/
noncomputable def f (y : ℝ) : ℝ := 
  9 / (y * y)

/-- Theorem stating that if f(y) = 1 for some y and f(2) = 2.25, then f(3) = 1 -/
theorem inverse_square_function (h1 : ∃ y, f y = 1) (h2 : f 2 = 2.25) : f 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_function_l772_77294


namespace NUMINAMATH_CALUDE_abs_value_difference_l772_77212

theorem abs_value_difference (x y : ℝ) (hx : |x| = 2) (hy : |y| = 3) (hxy : x > y) :
  x - y = 5 ∨ x - y = 1 := by
sorry

end NUMINAMATH_CALUDE_abs_value_difference_l772_77212


namespace NUMINAMATH_CALUDE_projection_equality_l772_77282

def a : Fin 2 → ℚ := ![3, -2]
def b : Fin 2 → ℚ := ![6, -1]
def p : Fin 2 → ℚ := ![9/10, -27/10]

theorem projection_equality (v : Fin 2 → ℚ) (hv : v ≠ 0) :
  (v • a / (v • v)) • v = (v • b / (v • b)) • v → 
  (v • a / (v • v)) • v = p :=
by sorry

end NUMINAMATH_CALUDE_projection_equality_l772_77282


namespace NUMINAMATH_CALUDE_average_annual_decrease_rate_optimal_price_reduction_l772_77268

-- Part 1: Average annual percentage decrease
def initial_price : ℝ := 200
def final_price : ℝ := 162
def num_years : ℕ := 2

-- Part 2: Unit price reduction
def selling_price : ℝ := 200
def initial_daily_sales : ℕ := 20
def price_decrease_step : ℝ := 3
def sales_increase_step : ℕ := 6
def target_daily_profit : ℝ := 1150

-- Theorem for Part 1
theorem average_annual_decrease_rate (x : ℝ) :
  initial_price * (1 - x)^num_years = final_price →
  x = 0.1 := by sorry

-- Theorem for Part 2
theorem optimal_price_reduction (m : ℝ) :
  (selling_price - m - 162) * (initial_daily_sales + 2 * m) = target_daily_profit →
  m = 15 := by sorry

end NUMINAMATH_CALUDE_average_annual_decrease_rate_optimal_price_reduction_l772_77268


namespace NUMINAMATH_CALUDE_colored_balls_permutations_l772_77223

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multiset_permutations (n : ℕ) (counts : List ℕ) : ℕ :=
  factorial n / (counts.map factorial).prod

theorem colored_balls_permutations :
  let total_balls : ℕ := 5
  let color_counts : List ℕ := [1, 1, 2, 1]  -- red, blue, yellow, white
  multiset_permutations total_balls color_counts = 60 := by
  sorry

end NUMINAMATH_CALUDE_colored_balls_permutations_l772_77223


namespace NUMINAMATH_CALUDE_circle_equation_l772_77208

-- Define the center of the circle
def center : ℝ × ℝ := (3, 1)

-- Define a point on the circle (the origin)
def origin : ℝ × ℝ := (0, 0)

-- Define the equation of a circle
def is_on_circle (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = (origin.1 - center.1)^2 + (origin.2 - center.2)^2

-- Theorem statement
theorem circle_equation : 
  ∀ x y : ℝ, is_on_circle x y ↔ (x - 3)^2 + (y - 1)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l772_77208


namespace NUMINAMATH_CALUDE_quadratic_with_irrational_root_l772_77234

theorem quadratic_with_irrational_root :
  ∃ (a b c : ℚ), a ≠ 0 ∧
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3) ∧
  a = 1 ∧ b = 6 ∧ c = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_with_irrational_root_l772_77234


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l772_77259

-- Define the basic structures
structure Line :=
  (id : ℕ)

structure Plane :=
  (id : ℕ)

-- Define the perpendicular relationships
def perpendicularToCountlessLines (l : Line) (p : Plane) : Prop :=
  sorry

def perpendicularToPlane (l : Line) : Plane → Prop :=
  sorry

-- Define the conditions p and q
def p (a : Line) (α : Plane) : Prop :=
  perpendicularToCountlessLines a α

def q (a : Line) (α : Plane) : Prop :=
  perpendicularToPlane a α

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ (a : Line) (α : Plane), q a α → p a α) ∧
  (∃ (a : Line) (α : Plane), p a α ∧ ¬(q a α)) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l772_77259


namespace NUMINAMATH_CALUDE_correct_ratio_maintenance_l772_77213

-- Define the original recipe ratios
def flour_original : ℚ := 4
def sugar_original : ℚ := 7
def salt_original : ℚ := 2

-- Define Mary's mistake
def flour_mistake : ℚ := 2

-- Define the function to calculate additional flour needed
def additional_flour (f_orig f_mistake s_orig : ℚ) : ℚ :=
  f_orig - f_mistake

-- Define the function to calculate the difference between additional flour and salt
def flour_salt_difference (f_orig f_mistake s_orig : ℚ) : ℚ :=
  additional_flour f_orig f_mistake s_orig - 0

-- Theorem statement
theorem correct_ratio_maintenance :
  flour_salt_difference flour_original flour_mistake salt_original = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_ratio_maintenance_l772_77213


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l772_77293

theorem no_solution_quadratic_inequality (x : ℝ) : 
  (5 * x^2 + 6 * x + 8 < 0) ∧ (abs x > 2) → False :=
by
  sorry


end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l772_77293


namespace NUMINAMATH_CALUDE_monday_rainfall_calculation_l772_77237

/-- The rainfall on Monday in inches -/
def monday_rainfall : ℝ := sorry

/-- The rainfall on Tuesday in inches -/
def tuesday_rainfall : ℝ := 0.2

/-- The difference in rainfall between Monday and Tuesday in inches -/
def rainfall_difference : ℝ := 0.7

theorem monday_rainfall_calculation : monday_rainfall = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_monday_rainfall_calculation_l772_77237


namespace NUMINAMATH_CALUDE_max_sum_of_product_3003_l772_77263

theorem max_sum_of_product_3003 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 3003 →
  A + B + C ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_product_3003_l772_77263


namespace NUMINAMATH_CALUDE_turner_rollercoaster_rides_l772_77217

/-- The number of times Turner wants to ride the rollercoaster -/
def R : ℕ := sorry

/-- The cost in tickets for one ride on the rollercoaster -/
def rollercoasterCost : ℕ := 4

/-- The cost in tickets for one ride on the Catapult -/
def catapultCost : ℕ := 4

/-- The cost in tickets for one ride on the Ferris wheel -/
def ferrisWheelCost : ℕ := 1

/-- The number of times Turner wants to ride the Catapult -/
def catapultRides : ℕ := 2

/-- The number of times Turner wants to ride the Ferris wheel -/
def ferrisWheelRides : ℕ := 1

/-- The total number of tickets Turner needs -/
def totalTickets : ℕ := 21

theorem turner_rollercoaster_rides :
  R * rollercoasterCost + catapultRides * catapultCost + ferrisWheelRides * ferrisWheelCost = totalTickets ∧ R = 3 := by
  sorry

end NUMINAMATH_CALUDE_turner_rollercoaster_rides_l772_77217


namespace NUMINAMATH_CALUDE_children_at_play_l772_77274

/-- Represents the number of children attending a play given specific conditions --/
def children_attending (adult_price child_price total_people total_revenue : ℕ) 
  (senior_citizens group_size : ℕ) : ℕ :=
  total_people - (total_revenue - child_price * (total_people - senior_citizens - group_size)) / 
    (adult_price - child_price)

/-- Theorem stating that under the given conditions, 20 children attended the play --/
theorem children_at_play : children_attending 12 6 80 840 3 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_children_at_play_l772_77274


namespace NUMINAMATH_CALUDE_net_increase_theorem_l772_77297

/-- Represents the different types of vehicles -/
inductive VehicleType
  | Car
  | Motorcycle
  | Van

/-- Represents the different phases of the play -/
inductive PlayPhase
  | BeforeIntermission
  | Intermission
  | AfterIntermission

/-- Initial number of vehicles in the back parking lot -/
def initialVehicles : VehicleType → ℕ
  | VehicleType.Car => 50
  | VehicleType.Motorcycle => 75
  | VehicleType.Van => 25

/-- Arrival rate per hour for each vehicle type during regular play time -/
def arrivalRate : VehicleType → ℕ
  | VehicleType.Car => 70
  | VehicleType.Motorcycle => 120
  | VehicleType.Van => 30

/-- Departure rate per hour for each vehicle type during regular play time -/
def departureRate : VehicleType → ℕ
  | VehicleType.Car => 40
  | VehicleType.Motorcycle => 60
  | VehicleType.Van => 20

/-- Duration of each phase in hours -/
def phaseDuration : PlayPhase → ℚ
  | PlayPhase.BeforeIntermission => 1
  | PlayPhase.Intermission => 1/2
  | PlayPhase.AfterIntermission => 3/2

/-- Net increase rate per hour for each vehicle type during a given phase -/
def netIncreaseRate (v : VehicleType) (p : PlayPhase) : ℚ :=
  match p with
  | PlayPhase.BeforeIntermission => (arrivalRate v - departureRate v : ℚ)
  | PlayPhase.Intermission => (arrivalRate v * 3/2 : ℚ)
  | PlayPhase.AfterIntermission => (arrivalRate v - departureRate v : ℚ)

/-- Total net increase for a given vehicle type -/
def totalNetIncrease (v : VehicleType) : ℚ :=
  (netIncreaseRate v PlayPhase.BeforeIntermission * phaseDuration PlayPhase.BeforeIntermission) +
  (netIncreaseRate v PlayPhase.Intermission * phaseDuration PlayPhase.Intermission) +
  (netIncreaseRate v PlayPhase.AfterIntermission * phaseDuration PlayPhase.AfterIntermission)

/-- Theorem stating the net increase for each vehicle type -/
theorem net_increase_theorem :
  ⌊totalNetIncrease VehicleType.Car⌋ = 127 ∧
  ⌊totalNetIncrease VehicleType.Motorcycle⌋ = 240 ∧
  ⌊totalNetIncrease VehicleType.Van⌋ = 47 := by
  sorry


end NUMINAMATH_CALUDE_net_increase_theorem_l772_77297


namespace NUMINAMATH_CALUDE_rhombus_closeness_range_l772_77273

-- Define the closeness function
def closeness (α β : ℝ) : ℝ := 180 - |α - β|

-- Theorem statement
theorem rhombus_closeness_range :
  ∀ α β : ℝ, 0 < α ∧ α < 180 → 0 < β ∧ β < 180 →
  0 < closeness α β ∧ closeness α β ≤ 180 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_closeness_range_l772_77273


namespace NUMINAMATH_CALUDE_impossible_corner_cut_l772_77266

theorem impossible_corner_cut (a b c : ℝ) : 
  a^2 + b^2 = 25 ∧ b^2 + c^2 = 36 ∧ c^2 + a^2 = 64 → False :=
by
  sorry

#check impossible_corner_cut

end NUMINAMATH_CALUDE_impossible_corner_cut_l772_77266


namespace NUMINAMATH_CALUDE_smallest_equal_gum_pieces_l772_77285

theorem smallest_equal_gum_pieces (n : ℕ) : n > 0 ∧ n % 6 = 0 ∧ n % 5 = 0 ∧ n % 8 = 0 → n ≥ 120 := by
  sorry

end NUMINAMATH_CALUDE_smallest_equal_gum_pieces_l772_77285


namespace NUMINAMATH_CALUDE_proposition_2_proposition_4_l772_77211

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the lines and planes
variable (m n : Line) (α β : Plane)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (subset : Line → Plane → Prop)

-- Axiom: m and n are different lines
axiom different_lines : m ≠ n

-- Axiom: α and β are different planes
axiom different_planes : α ≠ β

-- Proposition 2
theorem proposition_2 : 
  (perpendicular_line_line m n ∧ perpendicular_line_plane n α ∧ perpendicular_line_plane m β) → 
  perpendicular_plane_plane α β :=
sorry

-- Proposition 4
theorem proposition_4 : 
  (perpendicular_line_plane n β ∧ perpendicular_plane_plane α β) → 
  (parallel n α ∨ subset n α) :=
sorry

end NUMINAMATH_CALUDE_proposition_2_proposition_4_l772_77211


namespace NUMINAMATH_CALUDE_unique_number_exists_l772_77286

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    a ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    (a * 100 + b * 10 + c) % 4 = 0 ∧
    (b * 100 + c * 10 + d) % 5 = 0 ∧
    (c * 100 + d * 10 + e) % 3 = 0

theorem unique_number_exists : ∃! n : ℕ, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_unique_number_exists_l772_77286


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_shapes_l772_77235

/-- Given a right triangle ABC with legs AC = a and CB = b, prove:
    1. The side length of the largest square with vertex C inside the triangle
    2. The dimensions of the largest rectangle with vertex C inside the triangle -/
theorem right_triangle_inscribed_shapes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let square_side := a * b / (a + b)
  let rect_width := a / 2
  let rect_height := b / 2
  (∀ s : ℝ, s > 0 ∧ s ≤ a ∧ s ≤ b → s ≤ square_side) ∧
  (∀ w h : ℝ, w > 0 ∧ h > 0 ∧ w ≤ a ∧ h ≤ b ∧ w / a + h / b ≤ 1 → w * h ≤ rect_width * rect_height) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_shapes_l772_77235


namespace NUMINAMATH_CALUDE_right_triangle_sin_A_l772_77292

theorem right_triangle_sin_A (A B C : Real) (h1 : 0 < A ∧ A < π / 2) (h2 : 0 < B ∧ B < π / 2) (h3 : 0 < C ∧ C < π / 2) 
  (right_angle : B = π / 2) (angle_sum : A + B + C = π) (sin_cos_relation : 4 * Real.sin A = 5 * Real.cos A) :
  Real.sin A = 5 * Real.sqrt 41 / 41 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_A_l772_77292


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l772_77238

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  ((b - c)^2 = a^2 - b*c) →
  (a = 3) →
  (Real.sin C = 2 * Real.sin B) →
  (a = 2 * b * Real.sin (C/2)) →
  (b = 2 * c * Real.sin (A/2)) →
  (c = 2 * a * Real.sin (B/2)) →
  (A = π/3 ∧ (1/2 * b * c * Real.sin A = 3*Real.sqrt 3/2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l772_77238


namespace NUMINAMATH_CALUDE_power_of_power_l772_77242

theorem power_of_power (a : ℝ) : (a^5)^3 = a^15 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l772_77242


namespace NUMINAMATH_CALUDE_polynomial_identity_l772_77227

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l772_77227


namespace NUMINAMATH_CALUDE_optimal_racket_purchase_l772_77209

/-- Represents the purchase and selling prices of rackets -/
structure RacketPrices where
  tableTennisBuy : ℝ
  tableTennisSell : ℝ
  badmintonBuy : ℝ
  badmintonSell : ℝ

/-- Represents the quantity of rackets to purchase -/
structure RacketQuantities where
  tableTennis : ℝ
  badminton : ℝ

/-- Calculates the profit given prices and quantities -/
def calculateProfit (prices : RacketPrices) (quantities : RacketQuantities) : ℝ :=
  (prices.tableTennisSell - prices.tableTennisBuy) * quantities.tableTennis +
  (prices.badmintonSell - prices.badmintonBuy) * quantities.badminton

/-- The main theorem stating the optimal solution -/
theorem optimal_racket_purchase
  (prices : RacketPrices)
  (h1 : 2 * prices.tableTennisBuy + prices.badmintonBuy = 120)
  (h2 : 4 * prices.tableTennisBuy + 3 * prices.badmintonBuy = 270)
  (h3 : prices.tableTennisSell = 55)
  (h4 : prices.badmintonSell = 50)
  : ∃ (quantities : RacketQuantities),
    quantities.tableTennis + quantities.badminton = 300 ∧
    quantities.tableTennis ≥ (1/3) * quantities.badminton ∧
    prices.tableTennisBuy = 45 ∧
    prices.badmintonBuy = 30 ∧
    quantities.tableTennis = 75 ∧
    quantities.badminton = 225 ∧
    calculateProfit prices quantities = 5250 ∧
    ∀ (other : RacketQuantities),
      other.tableTennis + other.badminton = 300 →
      other.tableTennis ≥ (1/3) * other.badminton →
      calculateProfit prices quantities ≥ calculateProfit prices other := by
  sorry

end NUMINAMATH_CALUDE_optimal_racket_purchase_l772_77209


namespace NUMINAMATH_CALUDE_complex_set_is_line_l772_77202

/-- The set of complex numbers z such that (3+4i)z is real forms a line in the complex plane. -/
theorem complex_set_is_line : 
  let S : Set ℂ := {z | ∃ r : ℝ, (3 + 4*I) * z = r}
  ∃ a b : ℝ, S = {z | z.re = a * z.im + b} :=
sorry

end NUMINAMATH_CALUDE_complex_set_is_line_l772_77202


namespace NUMINAMATH_CALUDE_square_root_of_product_plus_one_l772_77262

theorem square_root_of_product_plus_one : 
  Real.sqrt ((34 : ℝ) * 32 * 28 * 26 + 1) = 170 := by sorry

end NUMINAMATH_CALUDE_square_root_of_product_plus_one_l772_77262


namespace NUMINAMATH_CALUDE_december_burger_expenditure_l772_77250

/-- The daily expenditure on burgers given the total monthly expenditure and number of days -/
def daily_burger_expenditure (total_expenditure : ℚ) (days : ℕ) : ℚ :=
  total_expenditure / days

theorem december_burger_expenditure :
  let total_expenditure : ℚ := 465
  let days : ℕ := 31
  daily_burger_expenditure total_expenditure days = 15 := by
sorry

end NUMINAMATH_CALUDE_december_burger_expenditure_l772_77250


namespace NUMINAMATH_CALUDE_max_b_value_l772_77236

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 4*a^2 * log x + b

theorem max_b_value (a : ℝ) (h_a : a > 0) :
  (∃ x₀ : ℝ, f a x₀ = g a b x₀ ∧ deriv (f a) x₀ = deriv (g a b) x₀) →
  (∃ b : ℝ, ∀ b' : ℝ, (∃ x₀ : ℝ, f a x₀ = g a b' x₀ ∧ deriv (f a) x₀ = deriv (g a b') x₀) → b' ≤ b) →
  (∃ b : ℝ, (∃ x₀ : ℝ, f a x₀ = g a b x₀ ∧ deriv (f a) x₀ = deriv (g a b) x₀) ∧
             ∀ b' : ℝ, (∃ x₀ : ℝ, f a x₀ = g a b' x₀ ∧ deriv (f a) x₀ = deriv (g a b') x₀) → b' ≤ b) →
  b = 2 * sqrt e :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l772_77236


namespace NUMINAMATH_CALUDE_age_ratio_proof_l772_77252

/-- Proves that given a person's age is 40, and 7 years earlier they were 11 times their daughter's age,
    the ratio of their age to their daughter's age today is 4:1 -/
theorem age_ratio_proof (your_age : ℕ) (daughter_age : ℕ) : 
  your_age = 40 →
  your_age - 7 = 11 * (daughter_age - 7) →
  your_age / daughter_age = 4 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l772_77252


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l772_77258

/-- Calculates the number of fence poles needed for a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  (2 * (length + width)) / pole_distance

/-- Theorem: A 60m by 50m rectangular plot with poles 5m apart needs 44 poles -/
theorem rectangular_plot_poles :
  fence_poles 60 50 5 = 44 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l772_77258


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l772_77281

theorem arithmetic_calculations : 
  (2 / 5 - 1 / 5 * (-5) + 3 / 5 = 2) ∧ 
  (-2^2 - (-3)^3 / 3 * (1 / 3) = -1) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l772_77281


namespace NUMINAMATH_CALUDE_correct_precipitation_forecast_interpretation_l772_77229

/-- Represents the possible interpretations of a precipitation forecast --/
inductive PrecipitationForecastInterpretation
  | RainDuration
  | AreaCoverage
  | Probability
  | NoMeaningfulForecast

/-- Represents a precipitation forecast --/
structure PrecipitationForecast where
  probability : ℝ
  interpretation : PrecipitationForecastInterpretation

/-- Asserts that a given interpretation is correct for a precipitation forecast --/
def is_correct_interpretation (forecast : PrecipitationForecast) : Prop :=
  forecast.interpretation = PrecipitationForecastInterpretation.Probability

/-- Theorem: Given an 80% precipitation forecast, the correct interpretation is that there's an 80% chance of rain --/
theorem correct_precipitation_forecast_interpretation 
  (forecast : PrecipitationForecast) 
  (h : forecast.probability = 0.8) :
  is_correct_interpretation forecast :=
sorry

end NUMINAMATH_CALUDE_correct_precipitation_forecast_interpretation_l772_77229


namespace NUMINAMATH_CALUDE_calculate_expression_quadratic_equation_roots_l772_77220

-- Problem 1
theorem calculate_expression : 
  (Real.sqrt 2 - Real.sqrt 12 + Real.sqrt (1/2)) * Real.sqrt 3 = 3 * Real.sqrt 6 / 2 - 6 := by sorry

-- Problem 2
theorem quadratic_equation_roots (c : ℝ) (h : (2 + Real.sqrt 3)^2 - 4*(2 + Real.sqrt 3) + c = 0) :
  ∃ (x : ℝ), x^2 - 4*x + c = 0 ∧ x ≠ 2 + Real.sqrt 3 ∧ x = 2 - Real.sqrt 3 ∧ c = 1 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_quadratic_equation_roots_l772_77220


namespace NUMINAMATH_CALUDE_largest_inscribed_semicircle_area_l772_77232

theorem largest_inscribed_semicircle_area (r : ℝ) (h : r = 1) : 
  let A := π * (1 / Real.sqrt 3)^2 / 2
  120 * A / π = 20 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_semicircle_area_l772_77232


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l772_77284

/-- 
Given a line L1 with equation 2x + 3y - 6 = 0 and a point P(1, -1),
prove that the line L2 with equation 2x + 3y + 1 = 0 is parallel to L1 and passes through P.
-/
theorem parallel_line_through_point (x y : ℝ) : 
  (2 * x + 3 * y - 6 = 0) →  -- Equation of L1
  (2 * 1 + 3 * (-1) + 1 = 0) →  -- L2 passes through P(1, -1)
  (∀ (x y : ℝ), 2 * x + 3 * y + 1 = 0 ↔ 
    (∃ (k : ℝ), 2 * x + 3 * y = 2 * 1 + 3 * (-1) + k * (2 * 1 + 3 * (-1) - (2 * 1 + 3 * (-1))))) :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l772_77284


namespace NUMINAMATH_CALUDE_largest_power_of_five_l772_77278

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := factorial n + factorial (n + 1) + factorial (n + 2)

theorem largest_power_of_five (n : ℕ) : 
  (∃ k : ℕ, sum_of_factorials 105 = 5^n * k) ∧ 
  (∀ m : ℕ, m > n → ¬∃ k : ℕ, sum_of_factorials 105 = 5^m * k) ↔ 
  n = 25 :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_five_l772_77278


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l772_77205

noncomputable def f (x : ℝ) : ℝ := (x + 2) * Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l772_77205
