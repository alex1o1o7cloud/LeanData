import Mathlib

namespace NUMINAMATH_CALUDE_statement_2_statement_3_l1318_131808

-- Define the types for lines and planes
variable {Line Plane : Type}

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)
variable (lineParallelPlane : Line → Plane → Prop)
variable (linePerpendicularPlane : Line → Plane → Prop)

-- Statement 2
theorem statement_2 (α β : Plane) (m : Line) :
  planePerpendicular α β → lineParallelPlane m α → linePerpendicularPlane m β := by
  sorry

-- Statement 3
theorem statement_3 (α β : Plane) (m : Line) :
  linePerpendicularPlane m β → planeParallel β α → planePerpendicular α β := by
  sorry

end NUMINAMATH_CALUDE_statement_2_statement_3_l1318_131808


namespace NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l1318_131815

/-- Given a cubic polynomial q(x) with the following properties:
    1) It has roots at 2, -2, and 1
    2) The function f(x) = (x^3 - 2x^2 - 5x + 6) / q(x) has no horizontal asymptote
    3) q(4) = 24
    Then q(x) = (2/3)x^3 - (2/3)x^2 - (8/3)x + 8/3 -/
theorem cubic_polynomial_uniqueness (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = 2 ∨ x = -2 ∨ x = 1) →
  (∃ k, ∀ x, q x = k * (x - 2) * (x + 2) * (x - 1)) →
  q 4 = 24 →
  ∀ x, q x = (2/3) * x^3 - (2/3) * x^2 - (8/3) * x + 8/3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l1318_131815


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1318_131897

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 14)
  (h2 : x + |y| - y = 16) : 
  x + y = 26/5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1318_131897


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_product_l1318_131868

theorem largest_common_divisor_of_consecutive_odd_product (n : ℕ) (h : Odd n) :
  (∃ (k : ℕ), k > 15 ∧ ∀ (m : ℕ), Odd m → k ∣ (m * (m + 2) * (m + 4) * (m + 6) * (m + 8))) → False :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_product_l1318_131868


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1318_131824

theorem inscribed_circle_radius (AB AC BC : ℝ) (h_AB : AB = 8) (h_AC : AC = 10) (h_BC : BC = 12) :
  let s := (AB + AC + BC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  area / s = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1318_131824


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1318_131861

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_point_x_coordinate (x : ℝ) (h : x > 0) : 
  (deriv f x = 2) → x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1318_131861


namespace NUMINAMATH_CALUDE_twenty_sixth_term_is_79_l1318_131812

/-- An arithmetic sequence with first term 4 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

/-- The 26th term of the arithmetic sequence is 79 -/
theorem twenty_sixth_term_is_79 : arithmetic_sequence 26 = 79 := by
  sorry

end NUMINAMATH_CALUDE_twenty_sixth_term_is_79_l1318_131812


namespace NUMINAMATH_CALUDE_heathers_oranges_l1318_131831

theorem heathers_oranges (initial remaining taken : ℕ) : 
  remaining = initial - taken → 
  taken = 35 → 
  remaining = 25 → 
  initial = 60 := by sorry

end NUMINAMATH_CALUDE_heathers_oranges_l1318_131831


namespace NUMINAMATH_CALUDE_joes_journey_time_l1318_131859

/-- Represents Joe's journey from home to school with a detour -/
def joes_journey (d : ℝ) : Prop :=
  let walking_speed : ℝ := d / 3 / 9  -- Speed to walk 1/3 of d in 9 minutes
  let running_speed : ℝ := 4 * walking_speed
  let total_walking_distance : ℝ := 2 * d / 3
  let total_running_distance : ℝ := 2 * d / 3
  let total_walking_time : ℝ := total_walking_distance / walking_speed
  let total_running_time : ℝ := total_running_distance / running_speed
  total_walking_time + total_running_time = 40.5

/-- Theorem stating that Joe's journey takes 40.5 minutes -/
theorem joes_journey_time :
  ∃ d : ℝ, d > 0 ∧ joes_journey d :=
sorry

end NUMINAMATH_CALUDE_joes_journey_time_l1318_131859


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l1318_131834

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 11

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3 * x^2

/-- The point of tangency -/
def P : ℝ × ℝ := (1, 12)

/-- The slope of the tangent line at P -/
def m : ℝ := f' P.1

/-- The y-intercept of the tangent line -/
def b : ℝ := P.2 - m * P.1

theorem tangent_line_y_intercept :
  b = 9 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l1318_131834


namespace NUMINAMATH_CALUDE_riley_time_outside_l1318_131814

theorem riley_time_outside (D : ℝ) (jonsey_awake : ℝ) (jonsey_outside : ℝ) (riley_awake : ℝ) (inside_time : ℝ) :
  D = 24 →
  jonsey_awake = (2/3) * D →
  jonsey_outside = (1/2) * jonsey_awake →
  riley_awake = (3/4) * D →
  jonsey_awake - jonsey_outside + riley_awake - (riley_awake * (8/9)) = inside_time →
  inside_time = 10 →
  riley_awake * (8/9) = riley_awake - (inside_time - (jonsey_awake - jonsey_outside)) :=
by sorry

end NUMINAMATH_CALUDE_riley_time_outside_l1318_131814


namespace NUMINAMATH_CALUDE_min_abs_phi_l1318_131886

/-- Given a function y = 2sin(x + φ), prove that if the abscissa is shortened to 1/3
    and the graph is shifted right by π/4, resulting in symmetry about (π/3, 0),
    then the minimum value of |φ| is π/4. -/
theorem min_abs_phi (φ : Real) : 
  (∀ x, 2 * Real.sin (3 * x + φ - 3 * Real.pi / 4) = 
        2 * Real.sin (3 * (2 * Real.pi / 3 - x) + φ - 3 * Real.pi / 4)) → 
  (∃ k : ℤ, φ = Real.pi / 4 + k * Real.pi) ∧ 
  (∀ ψ : Real, (∃ k : ℤ, ψ = Real.pi / 4 + k * Real.pi) → |ψ| ≥ |φ|) := by
  sorry

end NUMINAMATH_CALUDE_min_abs_phi_l1318_131886


namespace NUMINAMATH_CALUDE_rectangle_area_measurement_error_l1318_131848

theorem rectangle_area_measurement_error 
  (L W : ℝ) (L_measured W_measured : ℝ) 
  (h1 : L_measured = L * 1.2) 
  (h2 : W_measured = W * 0.9) : 
  (L_measured * W_measured - L * W) / (L * W) * 100 = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_measurement_error_l1318_131848


namespace NUMINAMATH_CALUDE_weight_replacement_l1318_131850

theorem weight_replacement (n : ℕ) (new_weight avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : new_weight = 81)
  (h3 : avg_increase = 2) :
  let total_increase := n * avg_increase
  let replaced_weight := new_weight - total_increase
  replaced_weight = 65 := by sorry

end NUMINAMATH_CALUDE_weight_replacement_l1318_131850


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1318_131810

/-- The cubic function f(x) = x³ - kx + k² -/
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x + k^2

theorem cubic_function_properties (k : ℝ) :
  (∀ x y, x < y → f k x < f k y) ∨ 
  ((∃ x y z, x < y ∧ y < z ∧ f k x = 0 ∧ f k y = 0 ∧ f k z = 0) ↔ 0 < k ∧ k < 4/27) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1318_131810


namespace NUMINAMATH_CALUDE_car_speed_calculation_l1318_131843

/-- Represents the speed of a car during a journey -/
structure CarJourney where
  first_speed : ℝ  -- Speed for the first 160 km
  second_speed : ℝ  -- Speed for the next 160 km
  average_speed : ℝ  -- Average speed for the entire 320 km

/-- Theorem stating the speed of the car during the next 160 km -/
theorem car_speed_calculation (journey : CarJourney) 
  (h1 : journey.first_speed = 70)
  (h2 : journey.average_speed = 74.67) : 
  journey.second_speed = 80 := by
  sorry

#check car_speed_calculation

end NUMINAMATH_CALUDE_car_speed_calculation_l1318_131843


namespace NUMINAMATH_CALUDE_daeyoung_pencils_l1318_131840

/-- Given the conditions of Daeyoung's purchase, prove that he bought 3 pencils. -/
theorem daeyoung_pencils :
  ∀ (E P : ℕ),
  E + P = 8 →
  300 * E + 500 * P = 3000 →
  E ≥ 1 →
  P ≥ 1 →
  P = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_daeyoung_pencils_l1318_131840


namespace NUMINAMATH_CALUDE_circle_equation_implies_expression_value_l1318_131891

theorem circle_equation_implies_expression_value (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x*y - 3*x + y - 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_implies_expression_value_l1318_131891


namespace NUMINAMATH_CALUDE_append_five_to_two_digit_number_l1318_131858

/-- Given a two-digit number with tens' digit t and units' digit u,
    when the digit 5 is placed after this number,
    the resulting number is equal to 100t + 10u + 5. -/
theorem append_five_to_two_digit_number (t u : ℕ) 
  (h1 : t ≥ 1 ∧ t ≤ 9) (h2 : u ≥ 0 ∧ u ≤ 9) :
  (10 * t + u) * 10 + 5 = 100 * t + 10 * u + 5 := by
  sorry

end NUMINAMATH_CALUDE_append_five_to_two_digit_number_l1318_131858


namespace NUMINAMATH_CALUDE_herb_count_at_spring_end_l1318_131819

def spring_duration : ℕ := 6

def initial_basil : ℕ := 3
def initial_parsley : ℕ := 1
def initial_mint : ℕ := 2
def initial_rosemary : ℕ := 1
def initial_thyme : ℕ := 1

def basil_growth_rate : ℕ → ℕ := λ weeks => 2^(weeks / 2)
def parsley_growth_rate : ℕ → ℕ := λ weeks => weeks
def mint_growth_rate : ℕ → ℕ := λ weeks => 3^(weeks / 4)

def extra_basil_week : ℕ := 3
def mint_stop_week : ℕ := 3
def parsley_loss_week : ℕ := 5
def parsley_loss_amount : ℕ := 2

def final_basil_count : ℕ := initial_basil * basil_growth_rate spring_duration + 1
def final_parsley_count : ℕ := initial_parsley + parsley_growth_rate spring_duration - parsley_loss_amount
def final_mint_count : ℕ := initial_mint * mint_growth_rate mint_stop_week
def final_rosemary_count : ℕ := initial_rosemary
def final_thyme_count : ℕ := initial_thyme

theorem herb_count_at_spring_end :
  final_basil_count + final_parsley_count + final_mint_count + 
  final_rosemary_count + final_thyme_count = 35 := by
  sorry

end NUMINAMATH_CALUDE_herb_count_at_spring_end_l1318_131819


namespace NUMINAMATH_CALUDE_swimmer_laps_theorem_l1318_131853

/-- Represents the number of laps swum by a person in a given number of weeks -/
def laps_swum (laps_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) : ℕ :=
  laps_per_day * days_per_week * weeks

theorem swimmer_laps_theorem (x : ℕ) :
  laps_swum 12 5 x = 60 * x :=
by
  sorry

#check swimmer_laps_theorem

end NUMINAMATH_CALUDE_swimmer_laps_theorem_l1318_131853


namespace NUMINAMATH_CALUDE_characterization_of_special_numbers_l1318_131807

/-- Sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The set of numbers that satisfy n = S(n)^2 - S(n) + 1 -/
def specialNumbers : Set ℕ :=
  {n : ℕ | n = (sumOfDigits n)^2 - sumOfDigits n + 1}

/-- Theorem stating that the set of special numbers is exactly {1, 13, 43, 91, 157} -/
theorem characterization_of_special_numbers :
  specialNumbers = {1, 13, 43, 91, 157} := by sorry

end NUMINAMATH_CALUDE_characterization_of_special_numbers_l1318_131807


namespace NUMINAMATH_CALUDE_cereal_sugar_percentage_l1318_131801

/-- The percentage of sugar in cereal A -/
def sugar_a : ℝ := 10

/-- The ratio of cereal A to cereal B -/
def ratio : ℝ := 1

/-- The percentage of sugar in the final mixture -/
def sugar_mixture : ℝ := 6

/-- The percentage of sugar in cereal B -/
def sugar_b : ℝ := 2

theorem cereal_sugar_percentage :
  (sugar_a * ratio + sugar_b * ratio) / (ratio + ratio) = sugar_mixture :=
by sorry

end NUMINAMATH_CALUDE_cereal_sugar_percentage_l1318_131801


namespace NUMINAMATH_CALUDE_num_orderings_eq_1554_l1318_131841

/-- The number of designs --/
def n : ℕ := 12

/-- The set of all design labels --/
def designs : Finset ℕ := Finset.range n

/-- The set of completed designs --/
def completed : Finset ℕ := {10, 11}

/-- The set of designs that could still be in the pile --/
def remaining : Finset ℕ := (designs \ completed).filter (· ≤ 9)

/-- The number of possible orderings for completing the remaining designs --/
def num_orderings : ℕ :=
  Finset.sum (Finset.powerset remaining) (fun S => S.card + 2)

theorem num_orderings_eq_1554 : num_orderings = 1554 := by
  sorry

end NUMINAMATH_CALUDE_num_orderings_eq_1554_l1318_131841


namespace NUMINAMATH_CALUDE_find_unknown_number_l1318_131882

theorem find_unknown_number : ∃ x : ℝ, 
  (14 + x + 53) / 3 = (21 + 47 + 22) / 3 + 3 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_number_l1318_131882


namespace NUMINAMATH_CALUDE_largest_non_representable_l1318_131826

def is_representable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 6 * a + 9 * b + 20 * c

theorem largest_non_representable : 
  (∀ m : ℕ, m > 43 → is_representable m) ∧ 
  ¬(is_representable 43) := by sorry

end NUMINAMATH_CALUDE_largest_non_representable_l1318_131826


namespace NUMINAMATH_CALUDE_loss_percentage_l1318_131856

/-- Calculate the percentage of loss given the cost price and selling price -/
theorem loss_percentage (cost_price selling_price : ℝ) : 
  cost_price = 750 → selling_price = 600 → 
  (cost_price - selling_price) / cost_price * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_l1318_131856


namespace NUMINAMATH_CALUDE_tan_2x_value_l1318_131820

theorem tan_2x_value (f : ℝ → ℝ) (x : ℝ) :
  f x = Real.sin x + Real.cos x →
  (deriv f) x = 3 * f x →
  Real.tan (2 * x) = -4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_2x_value_l1318_131820


namespace NUMINAMATH_CALUDE_jackson_monday_earnings_l1318_131898

/-- Represents Jackson's fundraising activities for a week -/
structure FundraisingWeek where
  goal : ℕ
  monday_earnings : ℕ
  tuesday_earnings : ℕ
  houses_per_day : ℕ
  earnings_per_four_houses : ℕ
  working_days : ℕ

/-- Theorem stating that Jackson's Monday earnings were $300 -/
theorem jackson_monday_earnings 
  (week : FundraisingWeek)
  (h1 : week.goal = 1000)
  (h2 : week.tuesday_earnings = 40)
  (h3 : week.houses_per_day = 88)
  (h4 : week.earnings_per_four_houses = 10)
  (h5 : week.working_days = 5) :
  week.monday_earnings = 300 := by
  sorry


end NUMINAMATH_CALUDE_jackson_monday_earnings_l1318_131898


namespace NUMINAMATH_CALUDE_triangle_problem_l1318_131849

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ A < π) (h3 : 0 < B ∧ B < π) (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π) (h6 : b * (Real.cos C + Real.sin C) = a)
  (h7 : a * Real.sin B = b * Real.sin A) (h8 : b * Real.sin C = c * Real.sin B)
  (h9 : a * Real.sin C = c * Real.sin A) (h10 : a * (1/4) = a * Real.sin A * Real.sin C) :
  B = π/4 ∧ Real.cos A = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1318_131849


namespace NUMINAMATH_CALUDE_system_solution_equivalence_l1318_131851

-- Define the system of linear inequalities
def system (x : ℝ) : Prop := (x - 2 > 1) ∧ (x < 4)

-- Define the solution set
def solution_set : Set ℝ := {x | 3 < x ∧ x < 4}

-- Theorem statement
theorem system_solution_equivalence :
  {x : ℝ | system x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_equivalence_l1318_131851


namespace NUMINAMATH_CALUDE_fraction_sum_minus_eight_l1318_131836

theorem fraction_sum_minus_eight : 
  (7 : ℚ) / 3 + 11 / 5 + 19 / 9 + 37 / 17 - 8 = 628 / 765 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_minus_eight_l1318_131836


namespace NUMINAMATH_CALUDE_product_of_roots_l1318_131875

theorem product_of_roots (a b c : ℂ) : 
  (3 * a^3 - 9 * a^2 + 5 * a - 15 = 0) →
  (3 * b^3 - 9 * b^2 + 5 * b - 15 = 0) →
  (3 * c^3 - 9 * c^2 + 5 * c - 15 = 0) →
  a * b * c = 5 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1318_131875


namespace NUMINAMATH_CALUDE_distance_between_trees_l1318_131842

/-- Given a yard with trees planted at equal distances, calculate the distance between consecutive trees. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : 
  yard_length = 360 ∧ num_trees = 31 → 
  (yard_length / (num_trees - 1 : ℝ)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l1318_131842


namespace NUMINAMATH_CALUDE_lines_exist_iff_angle_geq_60_l1318_131874

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  normal : Point3D
  d : ℝ

-- Define the given point and planes
variable (P : Point3D) -- Given point
variable (givenPlane : Plane) -- Given plane
variable (firstProjectionPlane : Plane) -- First projection plane

-- Define the angle between two planes
def angleBetweenPlanes (p1 p2 : Plane) : ℝ := sorry

-- Define a line passing through a point
structure Line where
  point : Point3D
  direction : Point3D

-- Define the angle between a line and a plane
def angleLinePlane (l : Line) (p : Plane) : ℝ := sorry

-- Define the distance between a point and a plane
def distancePointPlane (point : Point3D) (plane : Plane) : ℝ := sorry

-- Theorem statement
theorem lines_exist_iff_angle_geq_60 :
  (∃ (l1 l2 : Line),
    l1.point = P ∧
    l2.point = P ∧
    angleLinePlane l1 firstProjectionPlane = 60 ∧
    angleLinePlane l2 firstProjectionPlane = 60 ∧
    distancePointPlane l1.point givenPlane = distancePointPlane l2.point givenPlane) ↔
  angleBetweenPlanes givenPlane firstProjectionPlane ≥ 60 :=
sorry

end NUMINAMATH_CALUDE_lines_exist_iff_angle_geq_60_l1318_131874


namespace NUMINAMATH_CALUDE_cone_volume_given_sphere_l1318_131816

/-- Given a sphere and a cone with specific properties, prove that the volume of the cone is 12288π cm³ -/
theorem cone_volume_given_sphere (r_sphere : ℝ) (h_cone : ℝ) (r_cone : ℝ) :
  r_sphere = 24 →
  h_cone = 2 * r_sphere →
  π * r_cone * (r_cone + Real.sqrt (r_cone^2 + h_cone^2)) = 4 * π * r_sphere^2 →
  (1/3) * π * r_cone^2 * h_cone = 12288 * π := by
  sorry

#check cone_volume_given_sphere

end NUMINAMATH_CALUDE_cone_volume_given_sphere_l1318_131816


namespace NUMINAMATH_CALUDE_best_strategy_is_red_l1318_131877

/-- Represents the color of a disk side -/
inductive Color
| Red
| Blue

/-- Represents a disk with two sides -/
structure Disk where
  side1 : Color
  side2 : Color

/-- The set of all disks in the hat -/
def diskSet : Finset Disk := sorry

/-- The total number of disks -/
def totalDisks : ℕ := 10

/-- The number of disks with both sides red -/
def redDisks : ℕ := 3

/-- The number of disks with both sides blue -/
def blueDisks : ℕ := 2

/-- The number of disks with one side red and one side blue -/
def mixedDisks : ℕ := 5

/-- The probability of observing a red side -/
def probRedSide : ℚ := 11 / 20

/-- The probability of observing a blue side -/
def probBlueSide : ℚ := 9 / 20

/-- The probability that the other side is red, given that a red side is observed -/
def probRedGivenRed : ℚ := 6 / 11

/-- The probability that the other side is red, given that a blue side is observed -/
def probRedGivenBlue : ℚ := 5 / 9

theorem best_strategy_is_red :
  probRedGivenRed > 1 / 2 ∧ probRedGivenBlue > 1 / 2 := by sorry

end NUMINAMATH_CALUDE_best_strategy_is_red_l1318_131877


namespace NUMINAMATH_CALUDE_total_mascots_is_16x_l1318_131880

/-- Represents the number of mascots Jina has -/
structure Mascots where
  x : ℕ  -- number of teddies
  y : ℕ  -- number of bunnies
  z : ℕ  -- number of koalas

/-- Calculates the total number of mascots after Jina's mom gives her more teddies -/
def totalMascots (m : Mascots) : ℕ :=
  let x_new := m.x + 2 * m.y
  x_new + m.y + m.z

/-- Theorem stating the total number of mascots is 16 times the original number of teddies -/
theorem total_mascots_is_16x (m : Mascots)
    (h1 : m.y = 3 * m.x)  -- Jina has 3 times more bunnies than teddies
    (h2 : m.z = 2 * m.y)  -- Jina has twice the number of koalas as she has bunnies
    : totalMascots m = 16 * m.x := by
  sorry

#check total_mascots_is_16x

end NUMINAMATH_CALUDE_total_mascots_is_16x_l1318_131880


namespace NUMINAMATH_CALUDE_line_segment_param_sum_l1318_131845

/-- Given a line segment connecting (1, -3) and (-4, 5), parameterized by x = pt + q and y = rt + s
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 99. -/
theorem line_segment_param_sum (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) →
  (q = 1 ∧ s = -3) →
  (p + q = -4 ∧ r + s = 5) →
  p^2 + q^2 + r^2 + s^2 = 99 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_l1318_131845


namespace NUMINAMATH_CALUDE_family_ages_l1318_131890

structure Family where
  father_age : ℝ
  eldest_son_age : ℝ
  daughter_age : ℝ
  youngest_son_age : ℝ

def is_valid_family (f : Family) : Prop :=
  f.father_age = f.eldest_son_age + 20 ∧
  f.father_age + 2 = 2 * (f.eldest_son_age + 2) ∧
  f.daughter_age = f.eldest_son_age - 5 ∧
  f.youngest_son_age = f.daughter_age / 2

theorem family_ages : 
  ∃ (f : Family), is_valid_family f ∧ 
    f.father_age = 38 ∧ 
    f.eldest_son_age = 18 ∧ 
    f.daughter_age = 13 ∧ 
    f.youngest_son_age = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_l1318_131890


namespace NUMINAMATH_CALUDE_quadratic_roots_result_l1318_131864

theorem quadratic_roots_result (k p : ℕ) (hk : k > 0) 
  (h_roots : ∃ (x₁ x₂ : ℕ), x₁ > 0 ∧ x₂ > 0 ∧ 
    (k - 1) * x₁^2 - p * x₁ + k = 0 ∧
    (k - 1) * x₂^2 - p * x₂ + k = 0 ∧
    x₁ ≠ x₂) :
  k^(k*p) * (p^p + k^k) + (p + k) = 1989 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_result_l1318_131864


namespace NUMINAMATH_CALUDE_zoo_recovery_time_l1318_131832

/-- The time spent recovering escaped animals from a zoo -/
theorem zoo_recovery_time 
  (lions : ℕ) 
  (rhinos : ℕ) 
  (recovery_time_per_animal : ℕ) 
  (h1 : lions = 3) 
  (h2 : rhinos = 2) 
  (h3 : recovery_time_per_animal = 2) : 
  (lions + rhinos) * recovery_time_per_animal = 10 := by
sorry

end NUMINAMATH_CALUDE_zoo_recovery_time_l1318_131832


namespace NUMINAMATH_CALUDE_power_function_through_point_l1318_131887

/-- If the point (√3/3, √3/9) lies on the graph of a power function f(x), then f(x) = x³ -/
theorem power_function_through_point (f : ℝ → ℝ) :
  (∃ α : ℝ, ∀ x : ℝ, f x = x^α) →
  f (Real.sqrt 3 / 3) = Real.sqrt 3 / 9 →
  ∀ x : ℝ, f x = x^3 := by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1318_131887


namespace NUMINAMATH_CALUDE_single_stuffed_animal_cost_l1318_131806

/-- Represents the cost of items at a garage sale. -/
structure GarageSaleCost where
  magnet : ℝ
  sticker : ℝ
  stuffed_animals : ℝ
  toy_car : ℝ
  discount_rate : ℝ
  max_budget : ℝ

/-- Conditions for the garage sale problem. -/
def garage_sale_conditions (cost : GarageSaleCost) : Prop :=
  cost.magnet = 6 ∧
  cost.magnet = 3 * cost.sticker ∧
  cost.magnet = cost.stuffed_animals / 4 ∧
  cost.toy_car = cost.stuffed_animals / 4 ∧
  cost.toy_car = 2 * cost.sticker ∧
  cost.discount_rate = 0.1 ∧
  cost.max_budget = 30

/-- The theorem to be proved. -/
theorem single_stuffed_animal_cost 
  (cost : GarageSaleCost) 
  (h : garage_sale_conditions cost) : 
  cost.stuffed_animals / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_single_stuffed_animal_cost_l1318_131806


namespace NUMINAMATH_CALUDE_lcm_of_40_90_150_l1318_131881

theorem lcm_of_40_90_150 : Nat.lcm 40 (Nat.lcm 90 150) = 1800 := by sorry

end NUMINAMATH_CALUDE_lcm_of_40_90_150_l1318_131881


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1318_131893

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 4 * x + y = 20) 
  (h2 : x + 4 * y = 16) : 
  17 * x^2 + 20 * x * y + 17 * y^2 = 656 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1318_131893


namespace NUMINAMATH_CALUDE_quadratic_with_complex_root_l1318_131838

theorem quadratic_with_complex_root (a b c : ℝ) :
  (∀ x : ℂ, a * x^2 + b * x + c = 0 ↔ x = -1 + 2*I ∨ x = -1 - 2*I) →
  a = 1 ∧ b = 2 ∧ c = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_with_complex_root_l1318_131838


namespace NUMINAMATH_CALUDE_sum_ratio_simplification_main_result_l1318_131809

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => n * double_factorial n

def sum_ratio (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (double_factorial (2*i+1)) / (double_factorial (2*i+2)))

theorem sum_ratio_simplification (n : ℕ) :
  ∃ (c : ℕ), Odd c ∧ sum_ratio n = c / 2^(2*n - 7) := by sorry

theorem main_result :
  ∃ (c : ℕ), Odd c ∧ sum_ratio 2010 = c / 2^4013 ∧ 4013 / 10 = 401.3 := by sorry

end NUMINAMATH_CALUDE_sum_ratio_simplification_main_result_l1318_131809


namespace NUMINAMATH_CALUDE_boat_distance_problem_l1318_131818

/-- Proves that given a boat with speed 9 kmph in standing water, a stream with speed 1.5 kmph,
    and a round trip time of 24 hours, the distance to the destination is 105 km. -/
theorem boat_distance_problem (boat_speed : ℝ) (stream_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  boat_speed = 9 →
  stream_speed = 1.5 →
  total_time = 24 →
  distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time →
  distance = 105 := by
sorry


end NUMINAMATH_CALUDE_boat_distance_problem_l1318_131818


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1318_131830

/-- The number of coins flipped -/
def n : ℕ := 10

/-- The probability of getting heads on a single coin flip -/
def p : ℚ := 1/2

/-- The probability of getting an equal number of heads and tails -/
def prob_equal : ℚ := (n.choose (n/2)) / 2^n

/-- The probability of getting more heads than tails -/
def prob_more_heads : ℚ := (1 - prob_equal) / 2

theorem coin_flip_probability : prob_more_heads = 193/512 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1318_131830


namespace NUMINAMATH_CALUDE_wedding_cost_theorem_l1318_131870

/-- Calculate the total cost of John's wedding based on given parameters. -/
def wedding_cost (venue_cost : ℕ) (cost_per_guest : ℕ) (johns_guests : ℕ) (wife_increase_percent : ℕ) : ℕ :=
  let total_guests := johns_guests + (johns_guests * wife_increase_percent) / 100
  venue_cost + total_guests * cost_per_guest

/-- Theorem stating the total cost of the wedding given the specified conditions. -/
theorem wedding_cost_theorem :
  wedding_cost 10000 500 50 60 = 50000 := by
  sorry

end NUMINAMATH_CALUDE_wedding_cost_theorem_l1318_131870


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1318_131821

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_function_properties :
  ∀ (a b : ℝ),
  (∀ x : ℝ, f a b x = f a b (2 - x)) →  -- Symmetry about x=1
  f a b 0 = 0 →                        -- Passes through origin
  (∀ x : ℝ, f a b x = x^2 - 2*x) ∧     -- Explicit expression
  Set.Icc (-1) 3 = {y | ∃ x ∈ Set.Ioo 0 3, f a b x = y} -- Range on (0, 3]
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1318_131821


namespace NUMINAMATH_CALUDE_brown_leaves_percentage_l1318_131896

/-- Given a collection of leaves with known percentages of green and yellow leaves,
    calculate the percentage of brown leaves. -/
theorem brown_leaves_percentage
  (total_leaves : ℕ)
  (green_percentage : ℚ)
  (yellow_count : ℕ)
  (h1 : total_leaves = 25)
  (h2 : green_percentage = 1/5)
  (h3 : yellow_count = 15) :
  (total_leaves : ℚ) - green_percentage * total_leaves - yellow_count = 1/5 * total_leaves :=
sorry

end NUMINAMATH_CALUDE_brown_leaves_percentage_l1318_131896


namespace NUMINAMATH_CALUDE_f_intersects_negative_axes_l1318_131894

def f (x : ℝ) : ℝ := -x - 1

theorem f_intersects_negative_axes :
  (∃ x, x < 0 ∧ f x = 0) ∧ (∃ y, y < 0 ∧ f 0 = y) := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_negative_axes_l1318_131894


namespace NUMINAMATH_CALUDE_smallest_square_coverage_l1318_131811

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- The number of rectangles needed to cover a square -/
def rectangles_needed (r : Rectangle) (s : Square) : ℕ :=
  (s.side * s.side) / (r.length * r.width)

/-- Checks if a square can be exactly covered by rectangles -/
def is_exactly_coverable (r : Rectangle) (s : Square) : Prop :=
  (s.side * s.side) % (r.length * r.width) = 0

theorem smallest_square_coverage (r : Rectangle) (s : Square) : 
  r.length = 3 ∧ r.width = 2 ∧ 
  s.side = 12 ∧
  is_exactly_coverable r s ∧
  rectangles_needed r s = 24 ∧
  (∀ s' : Square, s'.side < s.side → ¬(is_exactly_coverable r s')) := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_coverage_l1318_131811


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1318_131872

theorem nested_fraction_evaluation :
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1318_131872


namespace NUMINAMATH_CALUDE_shadow_height_calculation_l1318_131885

/-- Given a lamppost and a person casting shadows under the same light source,
    calculate the person's height using the ratio method. -/
theorem shadow_height_calculation
  (lamppost_height : ℝ)
  (lamppost_shadow : ℝ)
  (michael_shadow : ℝ)
  (h_lamppost_height : lamppost_height = 50)
  (h_lamppost_shadow : lamppost_shadow = 25)
  (h_michael_shadow : michael_shadow = 20 / 12)  -- Convert 20 inches to feet
  : ∃ (michael_height : ℝ),
    michael_height = (lamppost_height / lamppost_shadow) * michael_shadow ∧
    michael_height * 12 = 40 := by
  sorry

end NUMINAMATH_CALUDE_shadow_height_calculation_l1318_131885


namespace NUMINAMATH_CALUDE_marbles_theorem_l1318_131892

def marbles_problem (total : ℕ) (colors : ℕ) (red_lost : ℕ) : ℕ :=
  let marbles_per_color := total / colors
  let red_remaining := marbles_per_color - red_lost
  let blue_remaining := marbles_per_color - (2 * red_lost)
  let yellow_remaining := marbles_per_color - (3 * red_lost)
  red_remaining + blue_remaining + yellow_remaining

theorem marbles_theorem :
  marbles_problem 72 3 5 = 42 := by sorry

end NUMINAMATH_CALUDE_marbles_theorem_l1318_131892


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l1318_131878

theorem consecutive_integers_product_sum (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 336) → (a + b + c = 21) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l1318_131878


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1318_131800

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem tenth_term_of_sequence (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 3 = 10 →
  arithmetic_sequence a₁ d 6 = 16 →
  arithmetic_sequence a₁ d 10 = 24 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1318_131800


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l1318_131846

theorem chocolate_bar_cost (total_bars : ℕ) (unsold_bars : ℕ) (total_amount : ℚ) :
  total_bars = 8 →
  unsold_bars = 3 →
  total_amount = 20 →
  (total_bars - unsold_bars : ℚ) * (total_amount / (total_bars - unsold_bars : ℚ)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l1318_131846


namespace NUMINAMATH_CALUDE_rogers_final_balance_theorem_l1318_131899

/-- Calculates Roger's final balance in US dollars after all transactions -/
def rogers_final_balance (initial_balance : ℝ) (video_game_percentage : ℝ) 
  (euros_spent : ℝ) (euro_to_dollar : ℝ) (canadian_dollars_received : ℝ) 
  (canadian_to_dollar : ℝ) : ℝ :=
  let remaining_after_game := initial_balance * (1 - video_game_percentage)
  let remaining_after_euros := remaining_after_game - euros_spent * euro_to_dollar
  remaining_after_euros + canadian_dollars_received * canadian_to_dollar

/-- Theorem stating Roger's final balance after all transactions -/
theorem rogers_final_balance_theorem : 
  rogers_final_balance 45 0.35 20 1.2 46 0.8 = 42.05 := by
  sorry

end NUMINAMATH_CALUDE_rogers_final_balance_theorem_l1318_131899


namespace NUMINAMATH_CALUDE_point_division_theorem_l1318_131825

/-- Given a line segment AB and a point P on it such that AP:PB = 3:5,
    prove that P = (5/8)*A + (3/8)*B --/
theorem point_division_theorem (A B P : ℝ × ℝ) : 
  (∃ t : ℝ, P = A + t • (B - A)) → -- P is on line segment AB
  (dist A P : ℝ) / (dist P B) = 3 / 5 → -- AP:PB = 3:5
  P = (5/8 : ℝ) • A + (3/8 : ℝ) • B := by sorry

end NUMINAMATH_CALUDE_point_division_theorem_l1318_131825


namespace NUMINAMATH_CALUDE_total_ridges_l1318_131844

/-- The number of ridges on a single vinyl record -/
def ridges_per_record : ℕ := 60

/-- The number of cases Jerry has -/
def num_cases : ℕ := 4

/-- The number of shelves in each case -/
def shelves_per_case : ℕ := 3

/-- The number of records each shelf can hold -/
def records_per_shelf : ℕ := 20

/-- The percentage of shelf capacity that is full, represented as a rational number -/
def shelf_fullness : ℚ := 60 / 100

/-- Theorem stating the total number of ridges on Jerry's records -/
theorem total_ridges : 
  ridges_per_record * num_cases * shelves_per_case * records_per_shelf * shelf_fullness = 8640 := by
  sorry

end NUMINAMATH_CALUDE_total_ridges_l1318_131844


namespace NUMINAMATH_CALUDE_scientific_notation_13000_l1318_131822

theorem scientific_notation_13000 : 13000 = 1.3 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_13000_l1318_131822


namespace NUMINAMATH_CALUDE_weight_loss_difference_l1318_131862

/-- Given Barbi's and Luca's weight loss rates and durations, prove the difference in their total weight losses -/
theorem weight_loss_difference (barbi_monthly_loss : ℝ) (barbi_months : ℕ) 
  (luca_yearly_loss : ℝ) (luca_years : ℕ) : 
  barbi_monthly_loss = 1.5 → 
  barbi_months = 12 → 
  luca_yearly_loss = 9 → 
  luca_years = 11 → 
  luca_yearly_loss * luca_years - barbi_monthly_loss * barbi_months = 81 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_difference_l1318_131862


namespace NUMINAMATH_CALUDE_characterization_of_f_l1318_131865

-- Define the property for the function
def satisfies_property (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, (f x * f y) ∣ ((1 + 2 * x) * f y + (1 + 2 * y) * f x)

-- Define strictly increasing function
def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, x < y → f x < f y

-- Main theorem
theorem characterization_of_f :
  ∀ f : ℕ → ℕ, strictly_increasing f → satisfies_property f →
  (∀ x : ℕ, f x = 2 * x + 1) ∨ (∀ x : ℕ, f x = 4 * x + 2) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_f_l1318_131865


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l1318_131873

/-- Converts a natural number from base 2 to base 4 -/
def base2ToBase4 (n : ℕ) : ℕ := sorry

/-- The binary number 10101110₂ -/
def binaryNumber : ℕ := 174  -- 10101110₂ in decimal is 174

theorem binary_to_quaternary_conversion :
  base2ToBase4 binaryNumber = 2232 := by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l1318_131873


namespace NUMINAMATH_CALUDE_trajectory_equation_l1318_131802

-- Define the point M
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition for point M
def satisfiesCondition (M : Point) : Prop :=
  Real.sqrt ((M.y + 5)^2 + M.x^2) - Real.sqrt ((M.y - 5)^2 + M.x^2) = 8

-- Define the trajectory equation
def isOnTrajectory (M : Point) : Prop :=
  M.y^2 / 16 - M.x^2 / 9 = 1 ∧ M.y > 0

-- Theorem statement
theorem trajectory_equation (M : Point) :
  satisfiesCondition M → isOnTrajectory M :=
by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1318_131802


namespace NUMINAMATH_CALUDE_max_value_theorem_l1318_131857

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := 2 * x * Real.log (2 * x)

theorem max_value_theorem (x₁ x₂ t : ℝ) (h₁ : f x₁ = t) (h₂ : g x₂ = t) (h₃ : t > 0) :
  ∃ (m : ℝ), m = (2 : ℝ) / Real.exp 1 ∧ 
  ∀ (y : ℝ), y = (Real.log t) / (x₁ * x₂) → y ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1318_131857


namespace NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l1318_131889

/-- The intersection point of two lines in 2D space -/
def intersection_point : ℚ × ℚ := (-1/8, 1/2)

/-- First line equation: y = -4x -/
def line1 (x y : ℚ) : Prop := y = -4 * x

/-- Second line equation: y - 2 = 12x -/
def line2 (x y : ℚ) : Prop := y - 2 = 12 * x

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l1318_131889


namespace NUMINAMATH_CALUDE_min_sum_with_real_roots_l1318_131852

theorem min_sum_with_real_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 2*b*x + a = 0) :
  a + b ≥ Real.rpow 1728 (1/3) ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    (∃ x : ℝ, x^2 + a₀*x + 3*b₀ = 0) ∧
    (∃ x : ℝ, x^2 + 2*b₀*x + a₀ = 0) ∧
    a₀ + b₀ = Real.rpow 1728 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_real_roots_l1318_131852


namespace NUMINAMATH_CALUDE_x_greater_than_half_l1318_131888

theorem x_greater_than_half (x : ℝ) (h : (1/2) * x = 1) : 
  (x - 1/2) / (1/2) * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_half_l1318_131888


namespace NUMINAMATH_CALUDE_permutation_calculation_l1318_131835

-- Define the permutation function
def A (n : ℕ) (r : ℕ) : ℚ :=
  if r ≤ n then (Nat.factorial n) / (Nat.factorial (n - r)) else 0

-- State the theorem
theorem permutation_calculation :
  (4 * A 8 4 + 2 * A 8 5) / (A 8 6 - A 9 5) * Nat.factorial 0 = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_permutation_calculation_l1318_131835


namespace NUMINAMATH_CALUDE_magician_trick_possible_magician_trick_smallest_l1318_131884

/-- Represents a sequence of digits -/
def DigitSequence (n : ℕ) := Fin n → Fin 10

/-- Represents a pair of adjacent positions in a sequence -/
structure AdjacentPair (n : ℕ) where
  first : Fin n
  second : Fin n
  adjacent : second = first.succ

/-- 
Given a sequence of digits and a pair of adjacent positions,
returns the sequence with those positions covered
-/
def coverDigits (seq : DigitSequence n) (pair : AdjacentPair n) : 
  Fin (n - 2) → Fin 10 := sorry

/-- 
States that for any sequence of 101 digits, covering any two adjacent digits
still allows for unique determination of the original sequence
-/
theorem magician_trick_possible : 
  ∀ (seq : DigitSequence 101) (pair : AdjacentPair 101),
  ∃! (original : DigitSequence 101), coverDigits original pair = coverDigits seq pair :=
sorry

/-- 
States that 101 is the smallest number for which the magician's trick is always possible
-/
theorem magician_trick_smallest : 
  (∀ n < 101, ¬(∀ (seq : DigitSequence n) (pair : AdjacentPair n),
    ∃! (original : DigitSequence n), coverDigits original pair = coverDigits seq pair)) ∧
  (∀ (seq : DigitSequence 101) (pair : AdjacentPair 101),
    ∃! (original : DigitSequence 101), coverDigits original pair = coverDigits seq pair) :=
sorry

end NUMINAMATH_CALUDE_magician_trick_possible_magician_trick_smallest_l1318_131884


namespace NUMINAMATH_CALUDE_unique_root_of_R_l1318_131827

/-- Represents a quadratic trinomial ax^2 + bx + c -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a quadratic trinomial P, construct R by adding P to the trinomial formed by swapping P's a and c -/
def constructR (P : QuadraticTrinomial) : QuadraticTrinomial :=
  { a := P.a + P.c
  , b := 2 * P.b
  , c := P.a + P.c }

theorem unique_root_of_R (P : QuadraticTrinomial) :
  let R := constructR P
  (∃! x : ℝ, R.a * x^2 + R.b * x + R.c = 0) →
  (∃ x : ℝ, x = -2 ∨ x = 2 ∧ R.a * x^2 + R.b * x + R.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_of_R_l1318_131827


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1318_131828

theorem sqrt_inequality : Real.sqrt 6 - Real.sqrt 5 > 2 * Real.sqrt 2 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1318_131828


namespace NUMINAMATH_CALUDE_kids_left_playing_l1318_131817

theorem kids_left_playing (initial_kids : ℝ) (kids_gone_home : ℝ) 
  (h1 : initial_kids = 22.0) 
  (h2 : kids_gone_home = 14.0) : 
  initial_kids - kids_gone_home = 8.0 := by
sorry

end NUMINAMATH_CALUDE_kids_left_playing_l1318_131817


namespace NUMINAMATH_CALUDE_problem_solution_l1318_131871

def A : Set ℝ := {x | 0 ≤ x - 1 ∧ x - 1 ≤ 2}
def B (a : ℝ) : Set ℝ := {x | 1 < x - a ∧ x - a < 2*a + 3}

theorem problem_solution :
  (∀ x, x ∈ (A ∪ B 1) ↔ 1 ≤ x ∧ x < 6) ∧
  (∀ x, x ∈ (A ∩ (Set.univ \ B 1)) ↔ 1 ≤ x ∧ x ≤ 2) ∧
  (∀ a, (A ∩ B a).Nonempty ↔ -2/3 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1318_131871


namespace NUMINAMATH_CALUDE_kendall_driving_distance_l1318_131869

theorem kendall_driving_distance (mother_distance father_distance : ℝ) 
  (h1 : mother_distance = 0.17)
  (h2 : father_distance = 0.5) :
  mother_distance + father_distance = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_kendall_driving_distance_l1318_131869


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_intersection_l1318_131895

-- Define the ellipse and hyperbola
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / 10 + y^2 / m = 1
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2 / b = 1

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := x = Real.sqrt 10 / 3

-- Define that the ellipse and hyperbola have the same foci
def same_foci (m b : ℝ) : Prop := 10 - m = 1 + b

-- Theorem statement
theorem ellipse_hyperbola_intersection (m b : ℝ) :
  (∃ y, ellipse m (Real.sqrt 10 / 3) y ∧ 
        hyperbola b (Real.sqrt 10 / 3) y ∧
        intersection_point (Real.sqrt 10 / 3) y) →
  same_foci m b →
  m = 1 ∧ b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_intersection_l1318_131895


namespace NUMINAMATH_CALUDE_complex_power_sum_l1318_131867

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i ^ 2 = -1

-- Define the cyclic nature of powers of i
axiom i_cyclic (n : ℕ) : i ^ (n + 4) = i ^ n

-- State the theorem
theorem complex_power_sum : i^20 + i^33 - i^56 = i := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1318_131867


namespace NUMINAMATH_CALUDE_remaining_donuts_l1318_131883

theorem remaining_donuts (initial_donuts : ℕ) (missing_percentage : ℚ) 
  (h1 : initial_donuts = 30)
  (h2 : missing_percentage = 70/100) :
  ↑initial_donuts * (1 - missing_percentage) = 9 :=
by sorry

end NUMINAMATH_CALUDE_remaining_donuts_l1318_131883


namespace NUMINAMATH_CALUDE_oil_depth_in_horizontal_cylindrical_tank_l1318_131813

/-- Represents a horizontal cylindrical tank --/
structure HorizontalCylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Represents the oil in the tank --/
structure Oil where
  surfaceArea : ℝ

/-- Calculates the possible depths of oil in the tank --/
def oilDepths (tank : HorizontalCylindricalTank) (oil : Oil) : Set ℝ :=
  { h : ℝ | h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3 }

/-- Theorem statement --/
theorem oil_depth_in_horizontal_cylindrical_tank
  (tank : HorizontalCylindricalTank)
  (oil : Oil)
  (h_length : tank.length = 8)
  (h_diameter : tank.diameter = 4)
  (h_surface_area : oil.surfaceArea = 16) :
  ∀ h ∈ oilDepths tank oil, h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3 :=
by
  sorry

#check oil_depth_in_horizontal_cylindrical_tank

end NUMINAMATH_CALUDE_oil_depth_in_horizontal_cylindrical_tank_l1318_131813


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1318_131876

theorem inequality_system_solution_set :
  ∀ x : ℝ, (3/2 * x + 5 ≤ -1 ∧ x + 3 < 0) ↔ x ≤ -4 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1318_131876


namespace NUMINAMATH_CALUDE_number_of_children_l1318_131879

theorem number_of_children (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 2) (h2 : total_pencils = 30) :
  total_pencils / pencils_per_child = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l1318_131879


namespace NUMINAMATH_CALUDE_trapezoid_division_theorem_l1318_131854

/-- A trapezoid with sides a, b, c, d where b is parallel to d -/
structure Trapezoid (α : Type*) [LinearOrderedField α] :=
  (a b c d : α)
  (parallel : b ≠ d)

/-- The ratio in which a line parallel to the bases divides a trapezoid -/
def divisionRatio {α : Type*} [LinearOrderedField α] (t : Trapezoid α) (z : α) : α :=
  (t.d + t.b) / 2 + (t.d - t.b)^2 / (2 * (t.a + t.c))

/-- The condition that two trapezoids formed by a parallel line have equal perimeters -/
def equalPerimeters {α : Type*} [LinearOrderedField α] (t : Trapezoid α) (z : α) : Prop :=
  t.a + z + t.c + (t.d - z) = t.b + z + t.a + (t.d - z)

theorem trapezoid_division_theorem {α : Type*} [LinearOrderedField α] (t : Trapezoid α) (z : α) :
  equalPerimeters t z → z = divisionRatio t z :=
sorry

end NUMINAMATH_CALUDE_trapezoid_division_theorem_l1318_131854


namespace NUMINAMATH_CALUDE_dartboard_central_angle_l1318_131855

/-- The central angle of a region on a circular dartboard, given its probability -/
theorem dartboard_central_angle (probability : ℝ) (h : probability = 1 / 8) :
  probability * 360 = 45 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_central_angle_l1318_131855


namespace NUMINAMATH_CALUDE_parabola_equation_l1318_131805

/-- Represents a parabola with the given properties -/
structure Parabola where
  -- The equation of the parabola in the form ax^2 + bxy + cy^2 + dx + ey + f = 0
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  c_pos : c > 0
  gcd_one : Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs) e.natAbs) f.natAbs = 1
  passes_through : a * 2^2 + b * 2 * 8 + c * 8^2 + d * 2 + e * 8 + f = 0
  focus_y : ℤ
  focus_y_is_5 : focus_y = 5
  symmetry_parallel_x : b = 0 ∧ a = 0
  vertex_on_y_axis : d = 0

/-- The theorem stating that the specific equation represents the parabola with given properties -/
theorem parabola_equation : ∃ (p : Parabola), p.a = 0 ∧ p.b = 0 ∧ p.c = 2 ∧ p.d = 9 ∧ p.e = -20 ∧ p.f = 50 :=
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1318_131805


namespace NUMINAMATH_CALUDE_jane_babysitting_problem_l1318_131863

/-- Represents the problem of determining when Jane stopped babysitting --/
theorem jane_babysitting_problem (jane_start_age : ℕ) (jane_current_age : ℕ) (oldest_babysat_current_age : ℕ) :
  jane_start_age = 20 →
  jane_current_age = 32 →
  oldest_babysat_current_age = 22 →
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_start_age ≤ jane_age →
    jane_age ≤ jane_current_age →
    child_age ≤ oldest_babysat_current_age →
    child_age ≤ jane_age / 2) →
  jane_current_age - jane_start_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_jane_babysitting_problem_l1318_131863


namespace NUMINAMATH_CALUDE_initial_dimes_equation_l1318_131847

/-- The number of dimes Sam initially had -/
def initial_dimes : ℕ := sorry

/-- The number of dimes Sam gave away -/
def dimes_given_away : ℕ := 7

/-- The number of dimes Sam has left -/
def dimes_left : ℕ := 2

/-- Theorem: The initial number of dimes is equal to the sum of dimes given away and dimes left -/
theorem initial_dimes_equation : initial_dimes = dimes_given_away + dimes_left := by
  sorry

end NUMINAMATH_CALUDE_initial_dimes_equation_l1318_131847


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l1318_131866

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l1318_131866


namespace NUMINAMATH_CALUDE_quadratic_discriminant_equality_l1318_131833

theorem quadratic_discriminant_equality (a b c x : ℝ) (h1 : a ≠ 0) (h2 : a * x^2 + b * x + c = 0) : 
  b^2 - 4*a*c = (2*a*x + b)^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_equality_l1318_131833


namespace NUMINAMATH_CALUDE_worst_player_is_niece_l1318_131803

-- Define the players
inductive Player
| Grandmother
| Niece
| Grandson
| SonInLaw

-- Define the sex of a player
inductive Sex
| Male
| Female

-- Define the generation of a player
inductive Generation
| Old
| Middle
| Young

-- Function to determine the sex of a player
def sex : Player → Sex
| Player.Grandmother => Sex.Female
| Player.Niece => Sex.Female
| Player.Grandson => Sex.Male
| Player.SonInLaw => Sex.Male

-- Function to determine the generation of a player
def generation : Player → Generation
| Player.Grandmother => Generation.Old
| Player.Niece => Generation.Young
| Player.Grandson => Generation.Young
| Player.SonInLaw => Generation.Middle

-- Function to determine if two players are cousins
def areCousins : Player → Player → Bool
| Player.Niece, Player.Grandson => true
| Player.Grandson, Player.Niece => true
| _, _ => false

-- Theorem statement
theorem worst_player_is_niece :
  ∀ (worst best : Player),
  (∃ cousin : Player, areCousins worst cousin ∧ sex cousin ≠ sex best) →
  generation worst ≠ generation best →
  worst = Player.Niece :=
by sorry

end NUMINAMATH_CALUDE_worst_player_is_niece_l1318_131803


namespace NUMINAMATH_CALUDE_flower_shop_carnation_percentage_l1318_131839

theorem flower_shop_carnation_percentage :
  let c : ℝ := 1  -- number of carnations (arbitrary non-zero value)
  let v : ℝ := c / 3  -- number of violets
  let t : ℝ := v / 4  -- number of tulips
  let r : ℝ := t  -- number of roses
  let total : ℝ := c + v + t + r  -- total number of flowers
  (c / total) * 100 = 200 / 3 :=
by sorry

end NUMINAMATH_CALUDE_flower_shop_carnation_percentage_l1318_131839


namespace NUMINAMATH_CALUDE_no_real_solutions_l1318_131829

theorem no_real_solutions : ¬∃ (x : ℝ), x + 64 / (x + 3) = -13 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1318_131829


namespace NUMINAMATH_CALUDE_inequality_proof_l1318_131823

def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

theorem inequality_proof (m : ℝ) (a b c : ℝ) 
  (h1 : Set.Icc 0 2 = {x | f m (x + 1) ≥ 0})
  (h2 : 1/a + 1/(2*b) + 1/(3*c) = m) : 
  a + 2*b + 3*c ≥ 9 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1318_131823


namespace NUMINAMATH_CALUDE_sugar_theorem_l1318_131837

def sugar_problem (initial : ℝ) (day1_use day1_borrow : ℝ)
  (day2_buy day2_use day2_receive : ℝ)
  (day3_buy day3_use day3_return day3_borrow : ℝ)
  (day4_use day4_receive : ℝ)
  (day5_use day5_borrow day5_return : ℝ) : Prop :=
  let day1 := initial - day1_use - day1_borrow
  let day2 := day1 + day2_buy - day2_use + day2_receive
  let day3 := day2 + day3_buy - day3_use + day3_return - day3_borrow
  let day4 := day3 - day4_use + day4_receive
  let day5 := day4 - day5_use - day5_borrow + day5_return
  day5 = 63.3

theorem sugar_theorem : sugar_problem 65 18.5 5.3 30.2 12.7 4.75 20.5 8.25 2.8 1.2 9.5 6.35 10.75 3.1 3 := by
  sorry

end NUMINAMATH_CALUDE_sugar_theorem_l1318_131837


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1318_131860

/-- A normally distributed random variable -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  mean_pos : 0 < mean
  std_dev_pos : 0 < std_dev

/-- The probability that a normal random variable falls within an interval -/
noncomputable def prob_in_interval (X : NormalDistribution) (lower upper : ℝ) : ℝ := sorry

/-- Theorem: If P(0 < X < a) = 0.3 for X ~ N(a, d²), then P(0 < X < 2a) = 0.6 -/
theorem normal_distribution_probability 
  (X : NormalDistribution) 
  (h : prob_in_interval X 0 X.mean = 0.3) : 
  prob_in_interval X 0 (2 * X.mean) = 0.6 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1318_131860


namespace NUMINAMATH_CALUDE_largest_multiple_seven_l1318_131804

theorem largest_multiple_seven (n : ℤ) : n = 147 ↔ 
  (∃ k : ℤ, n = 7 * k) ∧ 
  (-n > -150) ∧ 
  (∀ m : ℤ, (∃ j : ℤ, m = 7 * j) → (-m > -150) → m ≤ n) := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_seven_l1318_131804
