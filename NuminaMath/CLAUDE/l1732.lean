import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_ellipse_foci_coincide_l1732_173287

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := y^2 / 2 - x^2 / m = 1

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

-- Define the major axis endpoints of the ellipse
def ellipse_major_axis_endpoints : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

-- Define the foci of the hyperbola
def hyperbola_foci (m : ℝ) : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

-- Theorem statement
theorem hyperbola_ellipse_foci_coincide (m : ℝ) :
  (∀ x y, hyperbola x y m → ellipse x y) ∧
  (hyperbola_foci m = ellipse_major_axis_endpoints) →
  m = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_foci_coincide_l1732_173287


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1732_173270

theorem cube_root_equation_solution :
  ∃ (x y z : ℕ+),
    (4 * ((7 : ℝ)^(1/3) - (6 : ℝ)^(1/3))^(1/2) = x^(1/3) + y^(1/3) - z^(1/3)) ∧
    (x + y + z = 51) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1732_173270


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l1732_173254

theorem largest_solution_of_equation (x : ℝ) :
  (x / 5 + 1 / (5 * x) = 1 / 2) → x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l1732_173254


namespace NUMINAMATH_CALUDE_sale_price_calculation_l1732_173231

theorem sale_price_calculation (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := 0.8 * original_price
  let final_price := 0.9 * first_sale_price
  final_price / original_price = 0.72 :=
by sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l1732_173231


namespace NUMINAMATH_CALUDE_isosceles_in_26gon_l1732_173299

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Predicate to check if three vertices form an isosceles triangle -/
def IsIsoscelesTriangle (p : RegularPolygon n) (v1 v2 v3 : Fin n) : Prop :=
  let d12 := dist (p.vertices v1) (p.vertices v2)
  let d23 := dist (p.vertices v2) (p.vertices v3)
  let d31 := dist (p.vertices v3) (p.vertices v1)
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- Main theorem: In a regular 26-gon, any 9 vertices contain an isosceles triangle -/
theorem isosceles_in_26gon (p : RegularPolygon 26) 
  (vertices : Finset (Fin 26)) (h : vertices.card = 9) :
  ∃ (v1 v2 v3 : Fin 26), v1 ∈ vertices ∧ v2 ∈ vertices ∧ v3 ∈ vertices ∧
    IsIsoscelesTriangle p v1 v2 v3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_in_26gon_l1732_173299


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1732_173241

theorem sufficient_but_not_necessary (x : ℝ) :
  (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ∧
  ¬(x^2 - 3*x + 2 ≠ 0 → x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1732_173241


namespace NUMINAMATH_CALUDE_lena_kevin_ratio_l1732_173249

-- Define the initial number of candy bars for Lena
def lena_initial : ℕ := 16

-- Define the number of additional candy bars Lena needs
def additional_candies : ℕ := 5

-- Define the relationship between Lena's and Nicole's candy bars
def lena_nicole_diff : ℕ := 5

-- Define the relationship between Nicole's and Kevin's candy bars
def nicole_kevin_diff : ℕ := 4

-- Calculate Nicole's candy bars
def nicole_candies : ℕ := lena_initial - lena_nicole_diff

-- Calculate Kevin's candy bars
def kevin_candies : ℕ := nicole_candies - nicole_kevin_diff

-- Calculate Lena's final number of candy bars
def lena_final : ℕ := lena_initial + additional_candies

-- Theorem stating the ratio of Lena's final candy bars to Kevin's candy bars
theorem lena_kevin_ratio : 
  lena_final / kevin_candies = 3 ∧ lena_final % kevin_candies = 0 := by
  sorry

end NUMINAMATH_CALUDE_lena_kevin_ratio_l1732_173249


namespace NUMINAMATH_CALUDE_base_to_lateral_area_ratio_l1732_173291

/-- Represents a cone where the height is equal to the diameter of its circular base -/
structure SpecialCone where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cone
  h_eq_diam : h = 2 * r  -- condition that height equals diameter

/-- The ratio of base area to lateral area for a SpecialCone is 1:√5 -/
theorem base_to_lateral_area_ratio (cone : SpecialCone) :
  (π * cone.r^2) / (π * cone.r * Real.sqrt (cone.h^2 + cone.r^2)) = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_base_to_lateral_area_ratio_l1732_173291


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l1732_173229

/-- The equation of a line passing through (0, 2) with slope 2 is y = 2x + 2 -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  y - 2 = 2 * (x - 0) → y = 2 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l1732_173229


namespace NUMINAMATH_CALUDE_deployment_plans_count_l1732_173292

def number_of_volunteers : ℕ := 6
def number_of_positions : ℕ := 4
def number_of_restricted_volunteers : ℕ := 2

theorem deployment_plans_count :
  (number_of_volunteers.choose number_of_positions * number_of_positions.factorial) -
  (number_of_restricted_volunteers * ((number_of_volunteers - 1).choose (number_of_positions - 1) * (number_of_positions - 1).factorial)) = 240 :=
sorry

end NUMINAMATH_CALUDE_deployment_plans_count_l1732_173292


namespace NUMINAMATH_CALUDE_sue_shoe_probability_l1732_173204

/-- Represents the number of pairs of shoes of a specific color -/
structure ShoeCount where
  pairs : ℕ

/-- Represents the total shoe collection -/
structure ShoeCollection where
  black : ShoeCount
  brown : ShoeCount
  gray : ShoeCount

def sue_shoes : ShoeCollection :=
  { black := { pairs := 7 },
    brown := { pairs := 4 },
    gray  := { pairs := 3 } }

def total_shoes (sc : ShoeCollection) : ℕ :=
  2 * (sc.black.pairs + sc.brown.pairs + sc.gray.pairs)

/-- The probability of picking two shoes of the same color,
    one left and one right, from Sue's shoe collection -/
def same_color_diff_foot_prob (sc : ShoeCollection) : ℚ :=
  let total := total_shoes sc
  let prob_black := (2 * sc.black.pairs : ℚ) / total * (sc.black.pairs : ℚ) / (total - 1)
  let prob_brown := (2 * sc.brown.pairs : ℚ) / total * (sc.brown.pairs : ℚ) / (total - 1)
  let prob_gray := (2 * sc.gray.pairs : ℚ) / total * (sc.gray.pairs : ℚ) / (total - 1)
  prob_black + prob_brown + prob_gray

theorem sue_shoe_probability :
  same_color_diff_foot_prob sue_shoes = 37 / 189 := by sorry

end NUMINAMATH_CALUDE_sue_shoe_probability_l1732_173204


namespace NUMINAMATH_CALUDE_investment_rate_problem_l1732_173258

/-- Given a sum of money invested for a certain period, this function calculates the simple interest. -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_problem (sum : ℝ) (time : ℝ) (base_rate : ℝ) (interest_difference : ℝ) 
  (higher_rate : ℝ) :
  sum = 14000 →
  time = 2 →
  base_rate = 0.12 →
  interest_difference = 840 →
  simpleInterest sum higher_rate time = simpleInterest sum base_rate time + interest_difference →
  higher_rate = 0.15 := by
sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l1732_173258


namespace NUMINAMATH_CALUDE_percent_relation_l1732_173208

/-- Given that x is p percent more than 1/y, prove that y = (100 + p) / (100x) -/
theorem percent_relation (x y p : ℝ) (h : x = (1 + p / 100) * (1 / y)) :
  y = (100 + p) / (100 * x) := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l1732_173208


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l1732_173252

theorem shoe_price_calculation (thursday_price : ℝ) (friday_increase : ℝ) (monday_decrease : ℝ) : 
  thursday_price = 50 →
  friday_increase = 0.2 →
  monday_decrease = 0.15 →
  thursday_price * (1 + friday_increase) * (1 - monday_decrease) = 51 := by
sorry


end NUMINAMATH_CALUDE_shoe_price_calculation_l1732_173252


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1732_173235

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1732_173235


namespace NUMINAMATH_CALUDE_square_of_binomial_l1732_173264

theorem square_of_binomial (b : ℝ) : 
  (∃ (a c : ℝ), ∀ x, 16*x^2 + 40*x + b = (a*x + c)^2) → b = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l1732_173264


namespace NUMINAMATH_CALUDE_probability_of_C_l1732_173250

-- Define the wheel with four parts
inductive WheelPart : Type
| A
| B
| C
| D

-- Define the probability function
def probability : WheelPart → ℚ
| WheelPart.A => 1/4
| WheelPart.B => 1/3
| WheelPart.C => 1/4  -- This is what we want to prove
| WheelPart.D => 1/6

-- State the theorem
theorem probability_of_C : probability WheelPart.C = 1/4 := by
  -- The sum of all probabilities must equal 1
  have sum_of_probabilities : 
    probability WheelPart.A + probability WheelPart.B + 
    probability WheelPart.C + probability WheelPart.D = 1 := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_probability_of_C_l1732_173250


namespace NUMINAMATH_CALUDE_product_of_roots_abs_equation_l1732_173289

theorem product_of_roots_abs_equation (x : ℝ) :
  (∃ a b : ℝ, a ≠ b ∧ 
   (abs a)^2 - 3 * abs a - 10 = 0 ∧
   (abs b)^2 - 3 * abs b - 10 = 0 ∧
   a * b = -25) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_abs_equation_l1732_173289


namespace NUMINAMATH_CALUDE_negation_equivalence_l1732_173294

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1732_173294


namespace NUMINAMATH_CALUDE_parabola_equation_parabola_final_equation_l1732_173261

/-- A parabola with axis of symmetry parallel to the y-axis -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ
  eq_def : ∀ x, eq x = a * (x - 1) * (x - 4)

/-- The parabola passes through points (1,0) and (4,0) -/
def passes_through_points (p : Parabola) : Prop :=
  p.eq 1 = 0 ∧ p.eq 4 = 0

/-- The line y = 2x -/
def line (x : ℝ) : ℝ := 2 * x

/-- The parabola is tangent to the line y = 2x -/
def is_tangent (p : Parabola) : Prop :=
  ∃ x : ℝ, p.eq x = line x ∧ 
  ∀ y : ℝ, y ≠ x → p.eq y ≠ line y

/-- The main theorem -/
theorem parabola_equation (p : Parabola) 
  (h1 : passes_through_points p) 
  (h2 : is_tangent p) : 
  p.a = -2/9 ∨ p.a = -2 := by
  sorry

/-- The final result -/
theorem parabola_final_equation (p : Parabola) 
  (h1 : passes_through_points p) 
  (h2 : is_tangent p) : 
  (∀ x, p.eq x = -2/9 * (x - 1) * (x - 4)) ∨ 
  (∀ x, p.eq x = -2 * (x - 1) * (x - 4)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_parabola_final_equation_l1732_173261


namespace NUMINAMATH_CALUDE_soccer_ball_seams_soccer_ball_seams_eq_90_l1732_173240

/-- The number of seams needed to make a soccer ball with pentagons and hexagons -/
theorem soccer_ball_seams (num_pentagons num_hexagons : ℕ) 
  (h_pentagons : num_pentagons = 12)
  (h_hexagons : num_hexagons = 20) : ℕ :=
  let total_sides := num_pentagons * 5 + num_hexagons * 6
  total_sides / 2

/-- Proof that a soccer ball with 12 pentagons and 20 hexagons requires 90 seams -/
theorem soccer_ball_seams_eq_90 :
  soccer_ball_seams 12 20 rfl rfl = 90 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_seams_soccer_ball_seams_eq_90_l1732_173240


namespace NUMINAMATH_CALUDE_circle_translation_sum_l1732_173244

/-- The equation of circle D before translation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 10*y = -14

/-- The center of the circle after translation -/
def new_center : ℝ × ℝ := (1, -2)

/-- Theorem stating the sum of new center coordinates and radius after translation -/
theorem circle_translation_sum :
  ∃ (r : ℝ), 
    (∀ x y : ℝ, circle_equation x y → 
      ∃ a b : ℝ, new_center = (a, b) ∧ 
        a + b + r = -1 + Real.sqrt 27) :=
sorry

end NUMINAMATH_CALUDE_circle_translation_sum_l1732_173244


namespace NUMINAMATH_CALUDE_community_center_pairing_l1732_173209

theorem community_center_pairing (s t : ℕ) : 
  s > 0 ∧ t > 0 ∧ 
  4 * (t / 4) = 3 * (s / 3) ∧ 
  t / 4 = s / 3 →
  (t / 4 + s / 3) / (t + s) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_community_center_pairing_l1732_173209


namespace NUMINAMATH_CALUDE_pizza_order_l1732_173220

theorem pizza_order (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) 
  (h1 : people = 10)
  (h2 : slices_per_person = 2)
  (h3 : slices_per_pizza = 4) :
  (people * slices_per_person) / slices_per_pizza = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_l1732_173220


namespace NUMINAMATH_CALUDE_smallest_percent_increase_l1732_173230

def question_value : Fin 15 → ℕ
  | 0 => 100
  | 1 => 200
  | 2 => 300
  | 3 => 500
  | 4 => 1000
  | 5 => 2000
  | 6 => 4000
  | 7 => 8000
  | 8 => 16000
  | 9 => 32000
  | 10 => 64000
  | 11 => 125000
  | 12 => 250000
  | 13 => 500000
  | 14 => 1000000

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def options : List (Fin 15 × Fin 15) :=
  [(0, 1), (1, 2), (2, 3), (10, 11), (13, 14)]

theorem smallest_percent_increase :
  ∀ (pair : Fin 15 × Fin 15), pair ∈ options →
    percent_increase (question_value pair.1) (question_value pair.2) ≥
    percent_increase (question_value 1) (question_value 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_percent_increase_l1732_173230


namespace NUMINAMATH_CALUDE_smallest_n_equality_l1732_173212

def C (n : ℕ) : ℚ := 989 * (1 - (1/3)^n) / (1 - 1/3)

def D (n : ℕ) : ℚ := 2744 * (1 - (-1/3)^n) / (1 + 1/3)

theorem smallest_n_equality : ∃ (n : ℕ), n > 0 ∧ C n = D n ∧ ∀ (m : ℕ), m > 0 ∧ m < n → C m ≠ D m :=
  sorry

end NUMINAMATH_CALUDE_smallest_n_equality_l1732_173212


namespace NUMINAMATH_CALUDE_difference_of_squares_divisible_by_nine_l1732_173265

theorem difference_of_squares_divisible_by_nine (a b : ℤ) : 
  ∃ k : ℤ, (3*a + 2)^2 - (3*b + 2)^2 = 9*k := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_divisible_by_nine_l1732_173265


namespace NUMINAMATH_CALUDE_difference_max_min_all_three_l1732_173203

/-- The total number of students in the school --/
def total_students : ℕ := 1500

/-- The minimum number of students studying English --/
def min_english : ℕ := 1050

/-- The maximum number of students studying English --/
def max_english : ℕ := 1125

/-- The minimum number of students studying Spanish --/
def min_spanish : ℕ := 750

/-- The maximum number of students studying Spanish --/
def max_spanish : ℕ := 900

/-- The minimum number of students studying German --/
def min_german : ℕ := 300

/-- The maximum number of students studying German --/
def max_german : ℕ := 450

/-- The function that calculates the number of students studying all three languages --/
def students_all_three (e s g : ℕ) : ℤ :=
  e + s + g - total_students

/-- The theorem stating the difference between the maximum and minimum number of students studying all three languages --/
theorem difference_max_min_all_three :
  (max_german - (max 0 (students_all_three min_english min_spanish min_german))) = 450 :=
sorry

end NUMINAMATH_CALUDE_difference_max_min_all_three_l1732_173203


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l1732_173200

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h : 2 * x₁^2 - 3 * x₁ - 1 = 0 ∧ 2 * x₂^2 - 3 * x₂ - 1 = 0) : 
  (1 / x₁ + 1 / x₂ = -3) ∧ 
  ((x₁^2 - x₂^2)^2 = 153 / 16) ∧ 
  (2 * x₁^2 + 3 * x₂ = 11 / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l1732_173200


namespace NUMINAMATH_CALUDE_smallest_y_theorem_l1732_173285

def x : ℕ := 6 * 18 * 42

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def smallest_y_for_perfect_cube : ℕ := 441

theorem smallest_y_theorem :
  (∀ y : ℕ, y < smallest_y_for_perfect_cube → ¬(is_perfect_cube (x * y))) ∧
  (is_perfect_cube (x * smallest_y_for_perfect_cube)) := by sorry

end NUMINAMATH_CALUDE_smallest_y_theorem_l1732_173285


namespace NUMINAMATH_CALUDE_factorization_equality_l1732_173206

theorem factorization_equality (x : ℝ) :
  3 * x^2 * (x - 4) + 5 * x * (x - 4) = (3 * x^2 + 5 * x) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1732_173206


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1732_173268

theorem min_value_of_expression (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 1 → 
  (∀ x y : ℝ, a * x + b * y = 1 → x^2 + y^2 - 2*x - 2*y = 0 → x = 1 ∧ y = 1) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' = 1 → 1/a' + 2/b' ≥ 3 + 2 * Real.sqrt 2) ∧
  (1/a + 2/b = 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1732_173268


namespace NUMINAMATH_CALUDE_largest_T_for_inequality_l1732_173236

theorem largest_T_for_inequality (a b c d e : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e) 
  (h_sum : a + b = c + d + e) : 
  ∃ T : ℝ, T = (5 * Real.sqrt 30 - 2 * Real.sqrt 5) / 6 ∧
  (∀ S : ℝ, (Real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2) ≥ 
    S * (Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d + Real.sqrt e)^2) → 
    S ≤ T) :=
by sorry

end NUMINAMATH_CALUDE_largest_T_for_inequality_l1732_173236


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l1732_173279

/-- Two lines in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem parallel_lines_slope (l1 l2 : Line) : 
  l1 = Line.mk 2 (-1) → 
  l2 = Line.mk a 1 → 
  parallel l1 l2 → 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l1732_173279


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1732_173217

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

-- Define an increasing sequence
def increasing_sequence (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : increasing_sequence a)
  (h3 : a 2 = 2)
  (h4 : a 4 - a 3 = 4) :
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1732_173217


namespace NUMINAMATH_CALUDE_fraction_equality_l1732_173234

theorem fraction_equality (x y : ℝ) (h : x / y = 4 / 3) : (x - y) / y = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1732_173234


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1732_173297

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum1 : a 1 + a 4 + a 7 = 39) 
  (h_sum2 : a 2 + a 5 + a 8 = 33) : 
  a 5 + a 8 + a 11 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1732_173297


namespace NUMINAMATH_CALUDE_unique_solution_implies_relation_l1732_173214

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  y = x^2 + a*x + b ∧ x = y^2 + a*y + b

-- Theorem statement
theorem unique_solution_implies_relation (a b : ℝ) :
  (∃! p : ℝ × ℝ, system a b p.1 p.2) →
  a^2 = 2*(a + 2*b) - 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_relation_l1732_173214


namespace NUMINAMATH_CALUDE_count_figures_l1732_173276

/-- The number of large triangles in Figure 1 -/
def large_triangles : ℕ := 8

/-- The number of medium triangles in Figure 1 -/
def medium_triangles : ℕ := 4

/-- The number of small triangles in Figure 1 -/
def small_triangles : ℕ := 4

/-- The number of small squares (1x1) in Figure 2 -/
def small_squares : ℕ := 20

/-- The number of medium squares (2x2) in Figure 2 -/
def medium_squares : ℕ := 10

/-- The number of large squares (3x3) in Figure 2 -/
def large_squares : ℕ := 4

/-- The number of largest squares (4x4) in Figure 2 -/
def largest_square : ℕ := 1

/-- Theorem stating the total number of triangles in Figure 1 and squares in Figure 2 -/
theorem count_figures :
  (large_triangles + medium_triangles + small_triangles = 16) ∧
  (small_squares + medium_squares + large_squares + largest_square = 35) := by
  sorry

end NUMINAMATH_CALUDE_count_figures_l1732_173276


namespace NUMINAMATH_CALUDE_sequence_a_property_l1732_173237

def sequence_a (n : ℕ) : ℚ := 2 * n^2 - n

theorem sequence_a_property :
  (sequence_a 1 = 1) ∧
  (∀ n m : ℕ, n ≠ 0 → m ≠ 0 → sequence_a m / m - sequence_a n / n = 2 * (m - n)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_property_l1732_173237


namespace NUMINAMATH_CALUDE_solve_for_m_l1732_173277

theorem solve_for_m : ∃ m : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = -1 → 2*x + m + y = 0) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_solve_for_m_l1732_173277


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1732_173207

theorem smallest_n_congruence : ∃! n : ℕ+, (∀ m : ℕ+, 5 * m ≡ 220 [MOD 26] → n ≤ m) ∧ 5 * n ≡ 220 [MOD 26] := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1732_173207


namespace NUMINAMATH_CALUDE_percent_equality_problem_l1732_173273

theorem percent_equality_problem (x : ℝ) : (80 / 100 * 600 = 50 / 100 * x) → x = 960 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_problem_l1732_173273


namespace NUMINAMATH_CALUDE_exists_shape_with_five_faces_l1732_173290

/-- A geometric shape. -/
structure Shape where
  faces : ℕ

/-- A square pyramid is a shape with 5 faces. -/
def SquarePyramid : Shape :=
  { faces := 5 }

/-- There exists a shape with exactly 5 faces. -/
theorem exists_shape_with_five_faces : ∃ (s : Shape), s.faces = 5 := by
  sorry

end NUMINAMATH_CALUDE_exists_shape_with_five_faces_l1732_173290


namespace NUMINAMATH_CALUDE_mothers_age_l1732_173222

theorem mothers_age (person_age mother_age : ℕ) : 
  person_age = (2 * mother_age) / 5 →
  person_age + 10 = (mother_age + 10) / 2 →
  mother_age = 50 := by
sorry

end NUMINAMATH_CALUDE_mothers_age_l1732_173222


namespace NUMINAMATH_CALUDE_meaningful_range_l1732_173280

def is_meaningful (x : ℝ) : Prop :=
  x + 3 ≥ 0 ∧ x ≠ 1

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≥ -3 ∧ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_meaningful_range_l1732_173280


namespace NUMINAMATH_CALUDE_perfect_square_sum_permutation_l1732_173255

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_permutation (n : ℕ) (p : Fin n → Fin n) : Prop :=
  Function.Bijective p ∧ ∀ i : Fin n, is_perfect_square ((i.val + 1) + (p i).val + 1)

theorem perfect_square_sum_permutation :
  (∃ p : Fin 9 → Fin 9, valid_permutation 9 p) ∧
  (¬ ∃ p : Fin 11 → Fin 11, valid_permutation 11 p) ∧
  (∃ p : Fin 1996 → Fin 1996, valid_permutation 1996 p) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_permutation_l1732_173255


namespace NUMINAMATH_CALUDE_integral_equality_l1732_173201

open Real MeasureTheory

theorem integral_equality : ∫ (x : ℝ) in (0)..(1), 
  Real.exp (Real.sqrt ((1 - x) / (1 + x))) / ((1 + x) * Real.sqrt (1 - x^2)) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_equality_l1732_173201


namespace NUMINAMATH_CALUDE_abc_sum_range_l1732_173284

theorem abc_sum_range (a b c : ℝ) (h : a + b + 2*c = 0) :
  (∃ y : ℝ, y < 0 ∧ ab + ac + bc = y) ∧ ab + ac + bc ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_abc_sum_range_l1732_173284


namespace NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l1732_173269

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else a^(x - 1)

-- State the theorem
theorem range_of_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Ioo 2 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l1732_173269


namespace NUMINAMATH_CALUDE_car_journey_equation_l1732_173274

theorem car_journey_equation (x : ℝ) (h : x > 0) :
  let distance : ℝ := 120
  let slow_car_speed : ℝ := x
  let fast_car_speed : ℝ := 1.5 * x
  let slow_car_delay : ℝ := 1
  let slow_car_travel_time : ℝ := distance / slow_car_speed - slow_car_delay
  let fast_car_travel_time : ℝ := distance / fast_car_speed
  slow_car_travel_time = fast_car_travel_time :=
by sorry

end NUMINAMATH_CALUDE_car_journey_equation_l1732_173274


namespace NUMINAMATH_CALUDE_reflection_property_l1732_173283

/-- A reflection in R^2 -/
def Reflection (v : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ := sorry

theorem reflection_property (r : ℝ × ℝ → ℝ × ℝ) :
  r (2, 4) = (10, -2) →
  r (1, 6) = (107/37, -198/37) :=
by sorry

end NUMINAMATH_CALUDE_reflection_property_l1732_173283


namespace NUMINAMATH_CALUDE_fraction_simplification_l1732_173278

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 2/7) 
  (hy : y = 8/11) : 
  (7*x + 11*y) / (77*x*y) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1732_173278


namespace NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_plus_x_cubed_l1732_173263

theorem integral_sqrt_4_minus_x_squared_plus_x_cubed : 
  ∫ x in (-1)..1, (Real.sqrt (4 - x^2) + x^3) = Real.sqrt 3 + (2 * Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_plus_x_cubed_l1732_173263


namespace NUMINAMATH_CALUDE_remainder_of_product_product_remainder_l1732_173293

theorem remainder_of_product (a b m : ℕ) : (a * b) % m = ((a % m) * (b % m)) % m := by sorry

theorem product_remainder : (2002 * 1493) % 300 = 86 := by
  -- The proof would go here, but we're omitting it as per instructions
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_product_remainder_l1732_173293


namespace NUMINAMATH_CALUDE_total_batteries_used_l1732_173215

theorem total_batteries_used (flashlight_batteries : ℕ) (toy_batteries : ℕ) (controller_batteries : ℕ)
  (h1 : flashlight_batteries = 2)
  (h2 : toy_batteries = 15)
  (h3 : controller_batteries = 2) :
  flashlight_batteries + toy_batteries + controller_batteries = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_batteries_used_l1732_173215


namespace NUMINAMATH_CALUDE_min_sum_squares_cube_edges_l1732_173248

/-- Represents a cube with 8 vertices -/
structure Cube :=
  (v1 v2 v3 v4 v5 v6 v7 v8 : ℝ)

/-- Calculates the sum of squares of differences on the edges of a cube -/
def sumOfSquaresOfDifferences (c : Cube) : ℝ :=
  (c.v1 - c.v2)^2 + (c.v1 - c.v3)^2 + (c.v1 - c.v5)^2 +
  (c.v2 - c.v4)^2 + (c.v2 - c.v6)^2 +
  (c.v3 - c.v4)^2 + (c.v3 - c.v7)^2 +
  (c.v4 - c.v8)^2 +
  (c.v5 - c.v6)^2 + (c.v5 - c.v7)^2 +
  (c.v6 - c.v8)^2 +
  (c.v7 - c.v8)^2

/-- Theorem stating the minimum sum of squares of differences on cube edges -/
theorem min_sum_squares_cube_edges :
  ∃ (c : Cube),
    c.v1 = 0 ∧
    c.v8 = 2013 ∧
    c.v2 = 2013/2 ∧
    c.v3 = 2013/2 ∧
    c.v4 = 2013/2 ∧
    c.v5 = 2013/2 ∧
    c.v6 = 2013/2 ∧
    c.v7 = 2013/2 ∧
    sumOfSquaresOfDifferences c = (3 * 2013^2) / 2 ∧
    ∀ (c' : Cube), c'.v1 = 0 ∧ c'.v8 = 2013 →
      sumOfSquaresOfDifferences c' ≥ sumOfSquaresOfDifferences c :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_cube_edges_l1732_173248


namespace NUMINAMATH_CALUDE_x_equals_two_l1732_173288

theorem x_equals_two : ∀ x : ℝ, 3*x - 2*x + x = 3 - 2 + 1 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_two_l1732_173288


namespace NUMINAMATH_CALUDE_find_number_l1732_173218

theorem find_number : ∃ x : ℝ, x + 0.303 + 0.432 = 5.485 ∧ x = 4.750 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1732_173218


namespace NUMINAMATH_CALUDE_bob_question_creation_l1732_173228

theorem bob_question_creation (x : ℕ) : 
  x + 2*x + 4*x = 91 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_bob_question_creation_l1732_173228


namespace NUMINAMATH_CALUDE_max_subgrid_sum_l1732_173223

/-- Represents a 5x5 grid filled with integers -/
def Grid := Fin 5 → Fin 5 → ℕ

/-- Checks if all numbers in the grid are unique and between 1 and 25 -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 25 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

/-- Calculates the sum of a 2x2 subgrid starting at (i, j) -/
def subgrid_sum (g : Grid) (i j : Fin 4) : ℕ :=
  g i j + g i (j+1) + g (i+1) j + g (i+1) (j+1)

/-- The main theorem -/
theorem max_subgrid_sum (g : Grid) (h : valid_grid g) :
  (∀ i j : Fin 4, 45 ≤ subgrid_sum g i j) ∧
  ¬∃ N > 45, ∀ i j : Fin 4, N ≤ subgrid_sum g i j :=
sorry

end NUMINAMATH_CALUDE_max_subgrid_sum_l1732_173223


namespace NUMINAMATH_CALUDE_semi_truck_journey_l1732_173262

/-- A problem about a semi truck's journey on paved and dirt roads. -/
theorem semi_truck_journey (total_distance : ℝ) (paved_time : ℝ) (dirt_speed : ℝ) 
  (speed_difference : ℝ) (h1 : total_distance = 200) 
  (h2 : paved_time = 2) (h3 : dirt_speed = 32) (h4 : speed_difference = 20) : 
  (total_distance - paved_time * (dirt_speed + speed_difference)) / dirt_speed = 3 := by
  sorry

#check semi_truck_journey

end NUMINAMATH_CALUDE_semi_truck_journey_l1732_173262


namespace NUMINAMATH_CALUDE_cookie_ratio_l1732_173245

/-- Represents the cookie distribution problem --/
def cookie_problem (initial_cookies : ℕ) (given_to_brother : ℕ) (left_at_end : ℕ) : Prop :=
  let mother_gift := given_to_brother / 2
  let total_after_mother := initial_cookies - given_to_brother + mother_gift
  let given_to_sister := total_after_mother - left_at_end
  (given_to_sister : ℚ) / total_after_mother = 2 / 3

/-- The main theorem stating the cookie distribution ratio --/
theorem cookie_ratio : 
  cookie_problem 20 10 5 := by sorry

end NUMINAMATH_CALUDE_cookie_ratio_l1732_173245


namespace NUMINAMATH_CALUDE_sin_2x_derivative_l1732_173224

open Real

theorem sin_2x_derivative (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x)
  (deriv f) x = 2 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_l1732_173224


namespace NUMINAMATH_CALUDE_largest_expression_l1732_173239

theorem largest_expression : 
  let a := 3 + 2 + 1 + 9
  let b := 3 * 2 + 1 + 9
  let c := 3 + 2 * 1 + 9
  let d := 3 + 2 + 1 / 9
  let e := 3 * 2 / 1 + 9
  b ≥ a ∧ b > c ∧ b > d ∧ b ≥ e := by
sorry

end NUMINAMATH_CALUDE_largest_expression_l1732_173239


namespace NUMINAMATH_CALUDE_masks_duration_for_andrew_family_l1732_173227

/-- The number of days a pack of masks lasts for a family -/
def masksDuration (familySize : ℕ) (packSize : ℕ) (daysPerMask : ℕ) : ℕ :=
  let masksUsedPer2Days := familySize
  let fullSets := packSize / masksUsedPer2Days
  let remainingMasks := packSize % masksUsedPer2Days
  let fullDays := fullSets * daysPerMask
  if remainingMasks ≥ familySize then
    fullDays + daysPerMask
  else
    fullDays + 1

/-- Theorem: A pack of 75 masks lasts 21 days for a family of 7, changing masks every 2 days -/
theorem masks_duration_for_andrew_family :
  masksDuration 7 75 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_masks_duration_for_andrew_family_l1732_173227


namespace NUMINAMATH_CALUDE_percentage_problem_l1732_173232

theorem percentage_problem (p : ℝ) (x : ℝ) : 
  (p / 100) * x = 100 → 
  (120 / 100) * x = 600 → 
  p = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1732_173232


namespace NUMINAMATH_CALUDE_power_division_l1732_173242

theorem power_division (x : ℕ) : 8^15 / 64^3 = 8^9 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1732_173242


namespace NUMINAMATH_CALUDE_range_of_m_l1732_173225

/-- Given an increasing function f on ℝ and the condition f(m^2) > f(-m),
    the range of m is (-∞, -1) ∪ (0, +∞) -/
theorem range_of_m (f : ℝ → ℝ) (h_incr : Monotone f) (m : ℝ) (h_cond : f (m^2) > f (-m)) :
  m ∈ Set.Iio (-1) ∪ Set.Ioi 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1732_173225


namespace NUMINAMATH_CALUDE_min_sum_squares_l1732_173247

theorem min_sum_squares (x y z : ℝ) (h : x - 2*y - 3*z = 4) :
  ∃ (m : ℝ), m = 8/7 ∧ ∀ (a b c : ℝ), a - 2*b - 3*c = 4 → a^2 + b^2 + c^2 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1732_173247


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1732_173216

def P : Set ℤ := {-1, 1}
def Q : Set ℤ := {0, 1, 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1732_173216


namespace NUMINAMATH_CALUDE_hyperbola_orthogonal_asymptotes_l1732_173257

/-- A hyperbola is defined by its coefficients a, b, c, d, e, f in the equation ax^2 + 2bxy + cy^2 + dx + ey + f = 0 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Asymptotes of a hyperbola are orthogonal -/
def has_orthogonal_asymptotes (h : Hyperbola) : Prop :=
  h.a + h.c = 0

/-- The theorem stating that a hyperbola has orthogonal asymptotes if and only if a + c = 0 -/
theorem hyperbola_orthogonal_asymptotes (h : Hyperbola) :
  has_orthogonal_asymptotes h ↔ h.a + h.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_orthogonal_asymptotes_l1732_173257


namespace NUMINAMATH_CALUDE_rotten_oranges_count_l1732_173233

/-- The number of rotten oranges on a truck --/
def rotten_oranges : ℕ :=
  let total_oranges : ℕ := 10 * 30
  let oranges_for_juice : ℕ := 30
  let oranges_sold : ℕ := 220
  total_oranges - oranges_for_juice - oranges_sold

/-- Theorem stating that the number of rotten oranges is 50 --/
theorem rotten_oranges_count : rotten_oranges = 50 := by
  sorry

end NUMINAMATH_CALUDE_rotten_oranges_count_l1732_173233


namespace NUMINAMATH_CALUDE_smallest_n_value_smallest_n_is_99000_l1732_173238

/-- The number of ordered quadruplets satisfying the conditions -/
def num_quadruplets : ℕ := 91000

/-- The given GCD value for all quadruplets -/
def given_gcd : ℕ := 55

/-- 
Proposition: The smallest positive integer n satisfying the following conditions is 99000:
1. There exist exactly 91000 ordered quadruplets of positive integers (a, b, c, d)
2. For each quadruplet, gcd(a, b, c, d) = 55
3. For each quadruplet, lcm(a, b, c, d) = n
-/
theorem smallest_n_value (n : ℕ) : 
  (∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)), 
    S.card = num_quadruplets ∧ 
    ∀ (a b c d : ℕ), (a, b, c, d) ∈ S → 
      Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = given_gcd ∧
      Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = n) →
  n ≥ 99000 :=
by sorry

/-- The smallest value of n satisfying the conditions is indeed 99000 -/
theorem smallest_n_is_99000 : 
  ∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)), 
    S.card = num_quadruplets ∧ 
    ∀ (a b c d : ℕ), (a, b, c, d) ∈ S → 
      Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = given_gcd ∧
      Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = 99000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_smallest_n_is_99000_l1732_173238


namespace NUMINAMATH_CALUDE_prob_even_product_two_dice_l1732_173226

/-- A fair six-sided die -/
def SixSidedDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability space for rolling two dice -/
def TwoDiceRoll : Finset (ℕ × ℕ) := SixSidedDie.product SixSidedDie

/-- The event of rolling an even product -/
def EvenProduct : Set (ℕ × ℕ) := {p | p.1 * p.2 % 2 = 0}

theorem prob_even_product_two_dice :
  Finset.card (TwoDiceRoll.filter (λ p => p.1 * p.2 % 2 = 0)) / Finset.card TwoDiceRoll = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_product_two_dice_l1732_173226


namespace NUMINAMATH_CALUDE_x_greater_than_one_sufficient_not_necessary_l1732_173296

theorem x_greater_than_one_sufficient_not_necessary :
  (∀ x : ℝ, x > 1 → x^2 - 2*x + 1 > 0) ∧
  (∃ x : ℝ, x ≤ 1 ∧ x^2 - 2*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_one_sufficient_not_necessary_l1732_173296


namespace NUMINAMATH_CALUDE_trigonometric_signs_l1732_173259

theorem trigonometric_signs :
  let expr1 := Real.sin (1125 * π / 180)
  let expr2 := Real.tan (37 * π / 12) * Real.sin (37 * π / 12)
  let expr3 := Real.sin 4 / Real.tan 4
  let expr4 := Real.sin (|(-1)|)
  (expr1 > 0) ∧ (expr2 < 0) ∧ (expr3 < 0) ∧ (expr4 > 0) := by sorry

end NUMINAMATH_CALUDE_trigonometric_signs_l1732_173259


namespace NUMINAMATH_CALUDE_pet_store_puppies_l1732_173260

theorem pet_store_puppies 
  (bought : ℝ) 
  (puppies_per_cage : ℝ) 
  (cages_used : ℝ) 
  (h1 : bought = 3.0)
  (h2 : puppies_per_cage = 5.0)
  (h3 : cages_used = 4.2) :
  cages_used * puppies_per_cage - bought = 18.0 := by
sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l1732_173260


namespace NUMINAMATH_CALUDE_fifth_closest_is_park_l1732_173298

def buildings := ["bank", "school", "stationery store", "convenience store", "park"]

theorem fifth_closest_is_park :
  buildings.get? 4 = some "park" :=
sorry

end NUMINAMATH_CALUDE_fifth_closest_is_park_l1732_173298


namespace NUMINAMATH_CALUDE_matrix_equation_solutions_l1732_173210

/-- The determinant of a 2x2 matrix [[a, c], [d, b]] is defined as ab - cd -/
def det2x2 (a b c d : ℝ) : ℝ := a * b - c * d

/-- The solutions to the matrix equation involving x -/
def solutions : Set ℝ := {x | det2x2 (3*x) (2*x-1) (x+1) (2*x) = 2}

/-- The theorem stating the solutions to the matrix equation -/
theorem matrix_equation_solutions :
  solutions = {(5 + Real.sqrt 57) / 8, (5 - Real.sqrt 57) / 8} := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solutions_l1732_173210


namespace NUMINAMATH_CALUDE_geometric_series_problem_l1732_173243

theorem geometric_series_problem (n : ℝ) : 
  let a₁ : ℝ := 15
  let b₁ : ℝ := 3
  let a₂ : ℝ := 15
  let b₂ : ℝ := 3 + n
  let r₁ : ℝ := b₁ / a₁
  let r₂ : ℝ := b₂ / a₂
  let S₁ : ℝ := a₁ / (1 - r₁)
  let S₂ : ℝ := a₂ / (1 - r₂)
  S₂ = 5 * S₁ → n = 9.6 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l1732_173243


namespace NUMINAMATH_CALUDE_motorcycle_journey_avg_speed_l1732_173267

/-- A motorcyclist's journey with specific conditions -/
def motorcycle_journey (distance_AB : ℝ) (speed_BC : ℝ) : Prop :=
  ∃ (time_AB time_BC : ℝ),
    time_AB > 0 ∧ time_BC > 0 ∧
    time_AB = 3 * time_BC ∧
    distance_AB = 120 ∧
    speed_BC = 60 ∧
    (distance_AB / 2) / time_BC = speed_BC ∧
    (distance_AB + distance_AB / 2) / (time_AB + time_BC) = 45

/-- Theorem stating that under the given conditions, the average speed is 45 mph -/
theorem motorcycle_journey_avg_speed :
  motorcycle_journey 120 60 :=
sorry

end NUMINAMATH_CALUDE_motorcycle_journey_avg_speed_l1732_173267


namespace NUMINAMATH_CALUDE_combined_building_time_l1732_173221

/-- The time it takes Felipe to build his house, in months -/
def felipe_time : ℕ := 30

/-- The time it takes Emilio to build his house, in months -/
def emilio_time : ℕ := 2 * felipe_time

/-- The combined time for both Felipe and Emilio to build their homes, in years -/
def combined_time_years : ℚ := (felipe_time + emilio_time) / 12

theorem combined_building_time :
  combined_time_years = 7.5 := by sorry

end NUMINAMATH_CALUDE_combined_building_time_l1732_173221


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l1732_173275

theorem polynomial_expansion_equality (x : ℝ) : 
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 2)*(x + 3) = 
  6*x^3 - 16*x^2 + 43*x - 70 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l1732_173275


namespace NUMINAMATH_CALUDE_orthogonal_vectors_imply_x_equals_two_l1732_173286

/-- Given two vectors a and b in ℝ², prove that if they are orthogonal
    and have the form a = (x-5, 3) and b = (2, x), then x = 2. -/
theorem orthogonal_vectors_imply_x_equals_two :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x - 5, 3)
  let b : ℝ × ℝ := (2, x)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 2 := by
sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_imply_x_equals_two_l1732_173286


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1732_173251

/-- Represents a cone with given slant height and lateral surface property -/
structure Cone where
  slant_height : ℝ
  lateral_surface_is_semicircle : Prop

/-- Calculates the lateral surface area of a cone -/
def lateral_surface_area (c : Cone) : ℝ := sorry

theorem cone_lateral_surface_area 
  (c : Cone) 
  (h1 : c.slant_height = 10) 
  (h2 : c.lateral_surface_is_semicircle) : 
  lateral_surface_area c = 50 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1732_173251


namespace NUMINAMATH_CALUDE_distinct_integers_with_divisibility_property_l1732_173281

theorem distinct_integers_with_divisibility_property (n : ℕ) (h : n ≥ 2) :
  ∃ (a : Fin n → ℕ+), (∀ i j, i.val < j.val → (a i).val ≠ (a j).val) ∧
    (∀ i j, i.val < j.val → ((a i).val - (a j).val) ∣ ((a i).val + (a j).val)) := by
  sorry

end NUMINAMATH_CALUDE_distinct_integers_with_divisibility_property_l1732_173281


namespace NUMINAMATH_CALUDE_equation_solutions_l1732_173282

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 13*x - 16) + 1 / (x^2 + 4*x - 16) + 1 / (x^2 - 15*x - 16) = 0) ∧ 
           (x^2 + 13*x - 16 ≠ 0) ∧ (x^2 + 4*x - 16 ≠ 0) ∧ (x^2 - 15*x - 16 ≠ 0)} = 
  {1, -16, 4, -4} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1732_173282


namespace NUMINAMATH_CALUDE_game_result_l1732_173205

def g (m : ℕ) : ℕ :=
  if m % 3 = 0 then 8
  else if m = 2 ∨ m = 3 ∨ m = 5 then 3
  else if m % 2 = 0 then 1
  else 0

def jack_rolls : List ℕ := [2, 5, 6, 4, 3]
def jill_rolls : List ℕ := [1, 6, 3, 2, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result : total_points jack_rolls * total_points jill_rolls = 420 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1732_173205


namespace NUMINAMATH_CALUDE_even_quadratic_implies_b_zero_l1732_173271

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem even_quadratic_implies_b_zero (b c : ℝ) :
  IsEven (f b c) → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_implies_b_zero_l1732_173271


namespace NUMINAMATH_CALUDE_dexter_total_cards_l1732_173211

/-- Calculates the total number of cards Dexter has given the following conditions:
  * Dexter filled 3 fewer plastic boxes with football cards than basketball cards
  * He filled 9 boxes with basketball cards
  * Each basketball card box has 15 cards
  * Each football card box has 20 cards
-/
def totalCards (basketballBoxes : Nat) (basketballCardsPerBox : Nat) 
               (footballCardsPerBox : Nat) (boxDifference : Nat) : Nat :=
  let basketballCards := basketballBoxes * basketballCardsPerBox
  let footballBoxes := basketballBoxes - boxDifference
  let footballCards := footballBoxes * footballCardsPerBox
  basketballCards + footballCards

/-- Theorem stating that given the problem conditions, Dexter has 255 cards in total -/
theorem dexter_total_cards : 
  totalCards 9 15 20 3 = 255 := by
  sorry

end NUMINAMATH_CALUDE_dexter_total_cards_l1732_173211


namespace NUMINAMATH_CALUDE_equal_numbers_exist_l1732_173295

/-- Triangle inequality for three sides --/
def is_triangle (x y z : ℝ) : Prop :=
  x ≤ y + z ∧ y ≤ x + z ∧ z ≤ x + y

/-- Main theorem --/
theorem equal_numbers_exist (a b c : ℝ) :
  (∀ n : ℕ, is_triangle (a^n) (b^n) (c^n)) →
  (a = b ∨ b = c ∨ a = c) :=
by sorry

end NUMINAMATH_CALUDE_equal_numbers_exist_l1732_173295


namespace NUMINAMATH_CALUDE_diophantine_approximation_2005_l1732_173256

theorem diophantine_approximation_2005 (m n : ℕ+) : 
  |n * Real.sqrt 2005 - m| > (1 : ℝ) / (90 * n) := by sorry

end NUMINAMATH_CALUDE_diophantine_approximation_2005_l1732_173256


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1732_173219

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 - 2*a - 1 = 0) → (b^2 - 2*b - 1 = 0) → a^2 + a + 3*b = 7 := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1732_173219


namespace NUMINAMATH_CALUDE_player_one_wins_l1732_173253

/-- Represents a player in the stone game -/
inductive Player
| One
| Two

/-- Represents the state of the game -/
structure GameState where
  piles : List Nat
  currentPlayer : Player

/-- Represents a move in the game -/
structure Move where
  pileIndices : List Nat
  stonesRemoved : List Nat

/-- Defines a valid move for Player One -/
def isValidMovePlayerOne (m : Move) : Prop :=
  m.pileIndices.length = 1 ∧ 
  m.stonesRemoved.length = 1 ∧
  (m.stonesRemoved.head! = 1 ∨ m.stonesRemoved.head! = 2 ∨ m.stonesRemoved.head! = 3)

/-- Defines a valid move for Player Two -/
def isValidMovePlayerTwo (m : Move) : Prop :=
  m.pileIndices.length = m.stonesRemoved.length ∧
  m.pileIndices.length ≤ 3 ∧
  m.stonesRemoved.all (· = 1)

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  state.piles.all (· = 0)

/-- Determines if a player has a winning strategy from a given state -/
def hasWinningStrategy (state : GameState) : Prop :=
  sorry

/-- The main theorem: Player One has a winning strategy in the initial game state -/
theorem player_one_wins :
  hasWinningStrategy (GameState.mk (List.replicate 11 10) Player.One) :=
  sorry

end NUMINAMATH_CALUDE_player_one_wins_l1732_173253


namespace NUMINAMATH_CALUDE_root_existence_implies_m_range_l1732_173272

theorem root_existence_implies_m_range :
  ∀ m : ℝ, (∃ x : ℝ, 25^(-|x+1|) - 4 * 5^(-|x+1|) - m = 0) → -3 ≤ m ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_root_existence_implies_m_range_l1732_173272


namespace NUMINAMATH_CALUDE_reflection_distance_l1732_173202

/-- The distance between a point (3, 2) and its reflection over the y-axis is 6. -/
theorem reflection_distance : 
  let D : ℝ × ℝ := (3, 2)
  let D' : ℝ × ℝ := (-3, 2)  -- Reflection of D over y-axis
  Real.sqrt ((D'.1 - D.1)^2 + (D'.2 - D.2)^2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_reflection_distance_l1732_173202


namespace NUMINAMATH_CALUDE_brandy_trail_mix_chocolate_chips_l1732_173266

/-- The weight of chocolate chips in Brandy's trail mix -/
def weight_chocolate_chips (total_weight peanuts_weight raisins_weight : ℚ) : ℚ :=
  total_weight - (peanuts_weight + raisins_weight)

/-- Theorem stating that the weight of chocolate chips in Brandy's trail mix is 0.17 pounds -/
theorem brandy_trail_mix_chocolate_chips :
  weight_chocolate_chips 0.42 0.17 0.08 = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_brandy_trail_mix_chocolate_chips_l1732_173266


namespace NUMINAMATH_CALUDE_function_identity_l1732_173246

theorem function_identity (f : ℕ → ℕ) 
  (h1 : f 1 > 0)
  (h2 : ∀ m n : ℕ, f (m^2 + n^2) = (f m)^2 + (f n)^2) :
  ∀ n : ℕ, f n = n :=
sorry

end NUMINAMATH_CALUDE_function_identity_l1732_173246


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l1732_173213

theorem product_remainder_mod_five :
  (2024 * 1980 * 1848 * 1720) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l1732_173213
