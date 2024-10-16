import Mathlib

namespace NUMINAMATH_CALUDE_sugar_solution_replacement_l2970_297080

/-- Represents a sugar solution with a total weight and sugar percentage -/
structure SugarSolution where
  totalWeight : ℝ
  sugarPercentage : ℝ

/-- Represents the mixing of two sugar solutions -/
def mixSolutions (original : SugarSolution) (replacement : SugarSolution) (replacementFraction : ℝ) : SugarSolution :=
  { totalWeight := original.totalWeight,
    sugarPercentage := 
      (1 - replacementFraction) * original.sugarPercentage + 
      replacementFraction * replacement.sugarPercentage }

theorem sugar_solution_replacement (original : SugarSolution) (replacement : SugarSolution) :
  original.sugarPercentage = 12 →
  (mixSolutions original replacement (1/4)).sugarPercentage = 16 →
  replacement.sugarPercentage = 28 := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_replacement_l2970_297080


namespace NUMINAMATH_CALUDE_max_geometric_mean_of_sequence_l2970_297072

/-- Given a sequence of six numbers where one number is 1, any three consecutive numbers have the same arithmetic mean, and the arithmetic mean of all six numbers is A, the maximum value of the geometric mean of any three consecutive numbers is ∛((3A - 1)² / 4). -/
theorem max_geometric_mean_of_sequence (A : ℝ) (seq : Fin 6 → ℝ) 
  (h1 : ∃ i, seq i = 1)
  (h2 : ∀ i : Fin 4, (seq i + seq (i + 1) + seq (i + 2)) / 3 = 
                     (seq (i + 1) + seq (i + 2) + seq (i + 3)) / 3)
  (h3 : (seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5) / 6 = A) :
  ∃ i : Fin 4, (seq i * seq (i + 1) * seq (i + 2))^(1/3 : ℝ) ≤ ((3*A - 1)^2 / 4)^(1/3 : ℝ) ∧ 
  ∀ j : Fin 4, (seq j * seq (j + 1) * seq (j + 2))^(1/3 : ℝ) ≤ 
               (seq i * seq (i + 1) * seq (i + 2))^(1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_geometric_mean_of_sequence_l2970_297072


namespace NUMINAMATH_CALUDE_arcsin_plus_arccos_eq_pi_half_l2970_297041

theorem arcsin_plus_arccos_eq_pi_half (x : ℝ) :
  Real.arcsin x + Real.arccos (1 - x) = π / 2 → x = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_plus_arccos_eq_pi_half_l2970_297041


namespace NUMINAMATH_CALUDE_dog_length_calculation_l2970_297051

/-- Represents the length of a dog's body parts in inches -/
structure DogMeasurements where
  tail_length : ℝ
  body_length : ℝ
  head_length : ℝ

/-- Calculates the overall length of a dog given its measurements -/
def overall_length (d : DogMeasurements) : ℝ :=
  d.body_length + d.head_length

/-- Theorem stating the overall length of a dog with specific proportions -/
theorem dog_length_calculation (d : DogMeasurements) 
  (h1 : d.tail_length = d.body_length / 2)
  (h2 : d.head_length = d.body_length / 6)
  (h3 : d.tail_length = 9) :
  overall_length d = 21 := by
  sorry

#check dog_length_calculation

end NUMINAMATH_CALUDE_dog_length_calculation_l2970_297051


namespace NUMINAMATH_CALUDE_car_cost_sharing_l2970_297082

theorem car_cost_sharing (total_cost : ℕ) (initial_friends : ℕ) (car_wash_earnings : ℕ) (final_friends : ℕ) : 
  total_cost = 1700 →
  initial_friends = 6 →
  car_wash_earnings = 500 →
  final_friends = 5 →
  (total_cost - car_wash_earnings) / final_friends - (total_cost - car_wash_earnings) / initial_friends = 40 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_sharing_l2970_297082


namespace NUMINAMATH_CALUDE_number_2018_in_group_27_l2970_297064

/-- The sum of even numbers up to the k-th group -/
def S (k : ℕ) : ℕ := (3 * k^2 - k) / 2

/-- The proposition that 2018 belongs to the 27th group -/
theorem number_2018_in_group_27 : 
  S 26 < 1009 ∧ 1009 ≤ S 27 := by sorry

end NUMINAMATH_CALUDE_number_2018_in_group_27_l2970_297064


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2970_297065

/-- Given a triangle with exradii 2, 3, and 6 cm, the radius of the inscribed circle is 1 cm. -/
theorem inscribed_circle_radius (r₁ r₂ r₃ : ℝ) (hr₁ : r₁ = 2) (hr₂ : r₂ = 3) (hr₃ : r₃ = 6) :
  (1 / r₁ + 1 / r₂ + 1 / r₃)⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2970_297065


namespace NUMINAMATH_CALUDE_sum_of_other_x_coordinates_l2970_297068

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- The property that two points are opposite vertices of a rectangle --/
def are_opposite_vertices (p1 p2 : ℝ × ℝ) (r : Rectangle) : Prop :=
  (r.v1 = p1 ∧ r.v3 = p2) ∨ (r.v1 = p2 ∧ r.v3 = p1) ∨
  (r.v2 = p1 ∧ r.v4 = p2) ∨ (r.v2 = p2 ∧ r.v4 = p1)

/-- The theorem to be proved --/
theorem sum_of_other_x_coordinates (r : Rectangle) :
  are_opposite_vertices (2, 12) (8, 3) r →
  (r.v1.1 + r.v2.1 + r.v3.1 + r.v4.1) - (2 + 8) = 10 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_other_x_coordinates_l2970_297068


namespace NUMINAMATH_CALUDE_expression_equality_l2970_297044

theorem expression_equality (y b : ℝ) (h1 : y > 0) 
  (h2 : (4 * y) / b + (3 * y) / 10 = 0.5 * y) : b = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2970_297044


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l2970_297058

theorem butterflies_in_garden (initial : ℕ) (fraction : ℚ) (remaining : ℕ) : 
  initial = 9 → fraction = 1/3 → remaining = initial - (initial * fraction).floor → remaining = 6 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l2970_297058


namespace NUMINAMATH_CALUDE_fran_required_speed_l2970_297061

/-- Represents a bike ride with total time, break time, and average speed -/
structure BikeRide where
  totalTime : ℝ
  breakTime : ℝ
  avgSpeed : ℝ

/-- Calculates the distance traveled given a BikeRide -/
def distanceTraveled (ride : BikeRide) : ℝ :=
  ride.avgSpeed * (ride.totalTime - ride.breakTime)

theorem fran_required_speed (joann fran : BikeRide)
    (h1 : joann.totalTime = 4)
    (h2 : joann.breakTime = 1)
    (h3 : joann.avgSpeed = 10)
    (h4 : fran.totalTime = 3)
    (h5 : fran.breakTime = 0.5)
    (h6 : distanceTraveled joann = distanceTraveled fran) :
    fran.avgSpeed = 12 := by
  sorry

#check fran_required_speed

end NUMINAMATH_CALUDE_fran_required_speed_l2970_297061


namespace NUMINAMATH_CALUDE_system_solution_l2970_297092

theorem system_solution :
  ∀ x a : ℚ, x ≠ -2 →
  (a = -3*x^2 + 5*x - 2 ∧ (x+2)*a = 4*(x^2 - 1)) ↔ 
  ((x = 1 ∧ a = 0) ∨ (x = 0 ∧ a = -2) ∨ (x = -8/3 ∧ a = -110/3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2970_297092


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l2970_297090

/-- Given two lines with the same non-zero y-intercept, prove that the ratio of their x-intercepts is 1/2 -/
theorem x_intercept_ratio (b s t : ℝ) (hb : b ≠ 0) : 
  (0 = 8 * s + b) → (0 = 4 * t + b) → s / t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l2970_297090


namespace NUMINAMATH_CALUDE_taehyungs_mother_age_l2970_297079

/-- Given the age differences and the age of Taehyung's younger brother, 
    prove that Taehyung's mother is 43 years old. -/
theorem taehyungs_mother_age :
  ∀ (taehyung_age brother_age mother_age : ℕ),
    taehyung_age = brother_age + 5 →
    brother_age = 7 →
    mother_age = taehyung_age + 31 →
    mother_age = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_taehyungs_mother_age_l2970_297079


namespace NUMINAMATH_CALUDE_regions_in_circle_l2970_297093

/-- The number of regions created by radii and concentric circles in a circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem stating that 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (r : ℕ) (c : ℕ) 
  (h1 : r = 16) (h2 : c = 10) : 
  num_regions r c = 176 := by
  sorry

end NUMINAMATH_CALUDE_regions_in_circle_l2970_297093


namespace NUMINAMATH_CALUDE_divisibility_condition_l2970_297033

/-- Converts a base-9 number of the form 2d6d4₉ to base 10 --/
def base9_to_base10 (d : ℕ) : ℕ :=
  2 * 9^4 + d * 9^3 + 6 * 9^2 + d * 9 + 4

/-- Checks if a natural number is divisible by 13 --/
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

/-- States that 2d6d4₉ is divisible by 13 if and only if d = 4 --/
theorem divisibility_condition (d : ℕ) (h : d ≤ 8) :
  is_divisible_by_13 (base9_to_base10 d) ↔ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2970_297033


namespace NUMINAMATH_CALUDE_expected_value_of_strategic_die_rolling_l2970_297040

/-- Represents a 6-sided die -/
def Die := Fin 6

/-- The strategy for re-rolling -/
def rerollStrategy (roll : Die) : Bool :=
  roll.val < 4

/-- The expected value of a single roll of a 6-sided die -/
def singleRollExpectedValue : ℚ := 7/2

/-- The expected value after applying the re-roll strategy once -/
def strategicRollExpectedValue : ℚ := 17/4

/-- The final expected value after up to two re-rolls -/
def finalExpectedValue : ℚ := 17/4

theorem expected_value_of_strategic_die_rolling :
  finalExpectedValue = 17/4 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_strategic_die_rolling_l2970_297040


namespace NUMINAMATH_CALUDE_salary_percentage_increase_l2970_297026

theorem salary_percentage_increase 
  (initial_salary final_salary : ℝ) 
  (h1 : initial_salary = 50)
  (h2 : final_salary = 90) : 
  (final_salary - initial_salary) / initial_salary * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_salary_percentage_increase_l2970_297026


namespace NUMINAMATH_CALUDE_inequality_proof_l2970_297043

theorem inequality_proof (a b : ℝ) (h1 : 0 < b) (h2 : b < 1) (h3 : 1 < a) :
  a * b^2 < a * b ∧ a * b < a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2970_297043


namespace NUMINAMATH_CALUDE_dog_reachable_area_l2970_297067

/-- The area outside a regular pentagon that a tethered dog can reach -/
theorem dog_reachable_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 → rope_length = 3 → 
  ∃ (area : ℝ), area = 7.6 * Real.pi ∧ 
  area = (rope_length^2 * Real.pi * (288 / 360)) + 
         (2 * (side_length^2 * Real.pi * (72 / 360))) :=
sorry

end NUMINAMATH_CALUDE_dog_reachable_area_l2970_297067


namespace NUMINAMATH_CALUDE_yearly_salary_calculation_l2970_297017

/-- Proves that the yearly salary excluding turban is 160 rupees given the problem conditions --/
theorem yearly_salary_calculation (partial_payment : ℕ) (turban_value : ℕ) (months_worked : ℕ) (total_months : ℕ) :
  partial_payment = 50 →
  turban_value = 70 →
  months_worked = 9 →
  total_months = 12 →
  (partial_payment + turban_value : ℚ) / (months_worked : ℚ) * (total_months : ℚ) = 160 := by
sorry

end NUMINAMATH_CALUDE_yearly_salary_calculation_l2970_297017


namespace NUMINAMATH_CALUDE_rational_function_pair_l2970_297027

theorem rational_function_pair (f g : ℚ → ℚ)
  (h1 : ∀ x y : ℚ, f (g x - g y) = f (g x) - y)
  (h2 : ∀ x y : ℚ, g (f x - f y) = g (f x) - y) :
  ∃ c : ℚ, (∀ x : ℚ, f x = c * x) ∧ (∀ x : ℚ, g x = x / c) :=
sorry

end NUMINAMATH_CALUDE_rational_function_pair_l2970_297027


namespace NUMINAMATH_CALUDE_min_value_of_f_l2970_297037

def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2970_297037


namespace NUMINAMATH_CALUDE_derivative_sin_over_x_l2970_297012

open Real

theorem derivative_sin_over_x (x : ℝ) (h : x ≠ 0) :
  deriv (fun x => sin x / x) x = (x * cos x - sin x) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_over_x_l2970_297012


namespace NUMINAMATH_CALUDE_modified_ohara_triple_27_8_l2970_297097

/-- Definition of a Modified O'Hara Triple -/
def is_modified_ohara_triple (a b x : ℕ+) : Prop :=
  (a.val : ℝ)^(1/3) - (b.val : ℝ)^(1/3) = x.val

/-- Theorem: If (27, 8, x) is a Modified O'Hara triple, then x = 1 -/
theorem modified_ohara_triple_27_8 (x : ℕ+) :
  is_modified_ohara_triple 27 8 x → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_modified_ohara_triple_27_8_l2970_297097


namespace NUMINAMATH_CALUDE_remainder_problem_l2970_297052

theorem remainder_problem : 123456789012 % 200 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2970_297052


namespace NUMINAMATH_CALUDE_min_value_fraction_l2970_297014

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2*x + y = 1) :
  (x^2 + y^2 + x) / (x*y) ≥ 2*Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2970_297014


namespace NUMINAMATH_CALUDE_min_dot_product_l2970_297025

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus
def focus : ℝ × ℝ := (0, 1)

-- Define the line passing through the focus
def line_through_focus (x y : ℝ) : Prop := y = x + 1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := y = x - 1

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Statement of the theorem
theorem min_dot_product :
  ∃ (M N : ℝ × ℝ),
    parabola M.1 M.2 ∧
    parabola N.1 N.2 ∧
    line_through_focus M.1 M.2 ∧
    line_through_focus N.1 N.2 ∧
    (∀ (P : ℝ × ℝ), tangent_line P.1 P.2 →
      dot_product (M.1 - P.1, M.2 - P.2) (N.1 - P.1, N.2 - P.2) ≥ -14) ∧
    (∃ (P : ℝ × ℝ), tangent_line P.1 P.2 ∧
      dot_product (M.1 - P.1, M.2 - P.2) (N.1 - P.1, N.2 - P.2) = -14) :=
by
  sorry

end NUMINAMATH_CALUDE_min_dot_product_l2970_297025


namespace NUMINAMATH_CALUDE_miquel_point_midpoint_l2970_297084

-- Define the points
variable (A B C D O M T S : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define O as the intersection of diagonals
def is_diagonal_intersection (O A B C D : EuclideanPlane) : Prop := sorry

-- Define the circumcircles
def on_circumcircle (P Q R S : EuclideanPlane) : Prop := sorry

-- Define M as the intersection of circumcircles OAD and OBC
def is_circumcircle_intersection (M O A B C D : EuclideanPlane) : Prop := sorry

-- Define T and S on the line OM and on their respective circumcircles
def on_line_and_circumcircle (P Q R S T : EuclideanPlane) : Prop := sorry

-- Define the midpoint
def is_midpoint (M S T : EuclideanPlane) : Prop := sorry

-- The theorem
theorem miquel_point_midpoint 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_diagonal_intersection O A B C D)
  (h3 : is_circumcircle_intersection M O A B C D)
  (h4 : on_line_and_circumcircle O M A B T)
  (h5 : on_line_and_circumcircle O M C D S) :
  is_midpoint M S T := by sorry

end NUMINAMATH_CALUDE_miquel_point_midpoint_l2970_297084


namespace NUMINAMATH_CALUDE_sum_of_cubes_roots_l2970_297021

theorem sum_of_cubes_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - a*x₁ + a + 2 = 0 ∧ 
                x₂^2 - a*x₂ + a + 2 = 0 ∧ 
                x₁^3 + x₂^3 = -8) ↔ 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_roots_l2970_297021


namespace NUMINAMATH_CALUDE_circle_area_diameter_increase_l2970_297001

theorem circle_area_diameter_increase (A D A' D' : ℝ) :
  A' = 6 * A →
  A = (π / 4) * D^2 →
  A' = (π / 4) * D'^2 →
  D' = Real.sqrt 6 * D :=
by sorry

end NUMINAMATH_CALUDE_circle_area_diameter_increase_l2970_297001


namespace NUMINAMATH_CALUDE_least_value_of_quadratic_l2970_297042

theorem least_value_of_quadratic (y : ℝ) : 
  (5 * y^2 + 7 * y + 3 = 5) → y ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_least_value_of_quadratic_l2970_297042


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_m_greater_than_one_l2970_297062

/-- Theorem: If for all real x, x^2 - 2x + m > 0 is true, then m > 1 -/
theorem quadratic_always_positive_implies_m_greater_than_one (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + m > 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_m_greater_than_one_l2970_297062


namespace NUMINAMATH_CALUDE_inscribed_square_area_specific_inscribed_square_area_l2970_297032

/-- The area of a square inscribed in a right triangle -/
theorem inscribed_square_area (LP SN : ℝ) (h1 : LP > 0) (h2 : SN > 0) :
  let x := Real.sqrt (LP * SN)
  (x : ℝ) ^ 2 = LP * SN := by sorry

/-- The specific case where LP = 30 and SN = 70 -/
theorem specific_inscribed_square_area :
  let LP : ℝ := 30
  let SN : ℝ := 70
  let x := Real.sqrt (LP * SN)
  (x : ℝ) ^ 2 = 2100 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_specific_inscribed_square_area_l2970_297032


namespace NUMINAMATH_CALUDE_vasilya_wins_l2970_297069

/-- Represents a stick with a given length -/
structure Stick where
  length : ℝ
  length_pos : length > 0

/-- Represents a game state with a list of sticks -/
structure GameState where
  sticks : List Stick

/-- Represents a player's strategy for breaking sticks -/
def Strategy := GameState → Nat → Stick

/-- Defines the initial game state with a single 10 cm stick -/
def initialState : GameState :=
  { sticks := [{ length := 10, length_pos := by norm_num }] }

/-- Defines the game play for 18 breaks with alternating players -/
def playGame (petyaStrategy vasilyaStrategy : Strategy) : GameState :=
  sorry -- Implementation of game play

/-- Theorem stating that Vasilya can always ensure at least one stick is not shorter than 1 cm -/
theorem vasilya_wins (petyaStrategy : Strategy) : 
  ∃ (vasilyaStrategy : Strategy), ∃ (s : Stick), s ∈ (playGame petyaStrategy vasilyaStrategy).sticks ∧ s.length ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_vasilya_wins_l2970_297069


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_perimeter_twice_area_l2970_297031

theorem isosceles_right_triangle_perimeter_twice_area :
  ∃! a : ℝ, a > 0 ∧ (2 * a + a * Real.sqrt 2 = 2 * (1 / 2 * a^2)) := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_perimeter_twice_area_l2970_297031


namespace NUMINAMATH_CALUDE_min_value_expression_l2970_297066

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 4 / y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2970_297066


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2970_297028

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 1 = 2 ∧
  a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) : a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2970_297028


namespace NUMINAMATH_CALUDE_units_digit_of_sum_cubes_l2970_297083

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sum_cubes : units_digit (24^3 + 42^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_cubes_l2970_297083


namespace NUMINAMATH_CALUDE_tangent_line_points_l2970_297087

/-- The function f(x) = x³ + ax² -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x

theorem tangent_line_points (a : ℝ) :
  ∃ (x₀ : ℝ), (f_deriv a x₀ = -1 ∧ x₀ + f a x₀ = 0) →
  ((x₀ = 1 ∧ f a x₀ = -1) ∨ (x₀ = -1 ∧ f a x₀ = 1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_points_l2970_297087


namespace NUMINAMATH_CALUDE_n_value_for_specific_x_y_l2970_297099

theorem n_value_for_specific_x_y : ∀ (x y n : ℝ), 
  x = 3 → y = -1 → n = x - y^(x-y) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_n_value_for_specific_x_y_l2970_297099


namespace NUMINAMATH_CALUDE_odot_commutative_odot_no_identity_odot_associativity_undetermined_l2970_297019

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

end NUMINAMATH_CALUDE_odot_commutative_odot_no_identity_odot_associativity_undetermined_l2970_297019


namespace NUMINAMATH_CALUDE_expression_evaluation_l2970_297013

theorem expression_evaluation : (47 + 21)^2 - (47^2 + 21^2) - 7 * 47 = 1645 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2970_297013


namespace NUMINAMATH_CALUDE_alvin_marbles_l2970_297039

theorem alvin_marbles (initial : ℕ) (lost : ℕ) (won : ℕ) (final : ℕ) : 
  initial = 57 → lost = 18 → won = 25 → final = 64 →
  final = initial - lost + won :=
by sorry

end NUMINAMATH_CALUDE_alvin_marbles_l2970_297039


namespace NUMINAMATH_CALUDE_defective_units_shipped_l2970_297096

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ)
  (h1 : defective_rate = 0.04)
  (h2 : shipped_rate = 0.04) :
  (defective_rate * shipped_rate * total_units) / total_units = 0.0016 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l2970_297096


namespace NUMINAMATH_CALUDE_parallelogram_angle_equality_l2970_297029

-- Define the points
variable (A B C D P : Point)

-- Define the parallelogram property
def is_parallelogram (A B C D : Point) : Prop := sorry

-- Define the angle equality
def angle_eq (P Q R S T U : Point) : Prop := sorry

-- State the theorem
theorem parallelogram_angle_equality 
  (h_parallelogram : is_parallelogram A B C D)
  (h_angle_eq : angle_eq P A D P C D) :
  angle_eq P B C P D C := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_equality_l2970_297029


namespace NUMINAMATH_CALUDE_hyperbola_C_equation_P_not_midpoint_l2970_297054

/-- Hyperbola C passing through (-2,√6) with asymptotic lines y=±√2x -/
def hyperbola_C (x y : ℝ) : Prop :=
  x^2 - y^2/2 = 1

/-- Point P -/
def point_P : ℝ × ℝ := (1, 1)

/-- Line l passing through P and intersecting C at A and B -/
structure Line_l where
  k : ℝ
  intersects_C : ∃ (A B : ℝ × ℝ), 
    A ≠ B ∧ 
    hyperbola_C A.1 A.2 ∧ 
    hyperbola_C B.1 B.2 ∧
    A.2 = k * (A.1 - point_P.1) + point_P.2 ∧
    B.2 = k * (B.1 - point_P.1) + point_P.2

theorem hyperbola_C_equation : 
  ∀ a b : ℝ, a > 0 → b > 0 →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ hyperbola_C x y) ∧
  hyperbola_C (-2) (Real.sqrt 6) ∧
  (∀ x : ℝ, (Real.sqrt 2 * x)^2 = (x^2/a^2 - 1) * b^2) :=
sorry

theorem P_not_midpoint (l : Line_l) :
  ∀ A B : ℝ × ℝ,
  A ≠ B →
  hyperbola_C A.1 A.2 →
  hyperbola_C B.1 B.2 →
  A.2 = l.k * (A.1 - point_P.1) + point_P.2 →
  B.2 = l.k * (B.1 - point_P.1) + point_P.2 →
  (A.1 + B.1) / 2 ≠ point_P.1 ∨ (A.2 + B.2) / 2 ≠ point_P.2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_C_equation_P_not_midpoint_l2970_297054


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2970_297088

theorem probability_of_red_ball (p_red_white p_red_black : ℝ) 
  (h1 : p_red_white = 0.58)
  (h2 : p_red_black = 0.62) :
  p_red_white + p_red_black - 1 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2970_297088


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2970_297060

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2970_297060


namespace NUMINAMATH_CALUDE_parabola_c_value_l2970_297035

/-- A parabola with equation y = ax^2 + bx + c, vertex at (3, -5), and passing through (0, -2) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := -5
  point_x : ℝ := 0
  point_y : ℝ := -2

/-- The c-value of the parabola is -2 -/
theorem parabola_c_value (p : Parabola) : p.c = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2970_297035


namespace NUMINAMATH_CALUDE_interior_angles_sum_not_270_l2970_297086

theorem interior_angles_sum_not_270 (n : ℕ) (h : 3 ≤ n ∧ n ≤ 5) :
  (n - 2) * 180 ≠ 270 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_not_270_l2970_297086


namespace NUMINAMATH_CALUDE_larger_smaller_division_l2970_297011

theorem larger_smaller_division (L S Q : ℕ) : 
  L - S = 1311 → 
  L = 1430 → 
  L = S * Q + 11 → 
  Q = 11 := by
sorry

end NUMINAMATH_CALUDE_larger_smaller_division_l2970_297011


namespace NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l2970_297089

/-- The manufacturing cost of a shoe given specific conditions -/
theorem manufacturing_cost_of_shoe (transportation_cost : ℚ) (selling_price : ℚ) (gain_percentage : ℚ) :
  transportation_cost = 500 / 100 →
  selling_price = 270 →
  gain_percentage = 20 / 100 →
  ∃ (manufacturing_cost : ℚ),
    manufacturing_cost = selling_price / (1 + gain_percentage) - transportation_cost ∧
    manufacturing_cost = 220 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l2970_297089


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l2970_297038

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l2970_297038


namespace NUMINAMATH_CALUDE_beth_winning_strategy_l2970_297010

/-- Represents a wall of bricks in the game --/
structure Wall :=
  (size : Nat)

/-- Represents a game state with multiple walls --/
structure GameState :=
  (walls : List Wall)

/-- Calculates the nim-value of a single wall --/
def nimValue (w : Wall) : Nat :=
  sorry

/-- Calculates the nim-value of a game state --/
def gameNimValue (state : GameState) : Nat :=
  sorry

/-- Checks if a game state is a losing position for the current player --/
def isLosingPosition (state : GameState) : Prop :=
  gameNimValue state = 0

/-- The main theorem to prove --/
theorem beth_winning_strategy (startState : GameState) :
  startState.walls = [Wall.mk 6, Wall.mk 2, Wall.mk 1] →
  isLosingPosition startState :=
sorry

end NUMINAMATH_CALUDE_beth_winning_strategy_l2970_297010


namespace NUMINAMATH_CALUDE_external_diagonals_condition_l2970_297000

/-- Represents the lengths of external diagonals of a right regular prism -/
structure ExternalDiagonals where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if the given lengths could be valid external diagonals of a right regular prism -/
def isValidExternalDiagonals (d : ExternalDiagonals) : Prop :=
  d.x^2 + d.y^2 > d.z^2 ∧ d.y^2 + d.z^2 > d.x^2 ∧ d.x^2 + d.z^2 > d.y^2

theorem external_diagonals_condition (d : ExternalDiagonals) :
  d.x > 0 ∧ d.y > 0 ∧ d.z > 0 → isValidExternalDiagonals d :=
by sorry

end NUMINAMATH_CALUDE_external_diagonals_condition_l2970_297000


namespace NUMINAMATH_CALUDE_triple_minus_double_equals_eight_point_five_l2970_297056

theorem triple_minus_double_equals_eight_point_five (x : ℝ) : 
  3 * x = 2 * x + 8.5 → 3 * x - 2 * x = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_triple_minus_double_equals_eight_point_five_l2970_297056


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2970_297063

theorem solution_set_equivalence :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2970_297063


namespace NUMINAMATH_CALUDE_part_one_part_two_l2970_297004

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | 2*x^2 - 3*x + 1 ≤ 0}
def Q (a : ℝ) : Set ℝ := {x : ℝ | (x - a)*(x - a - 1) ≤ 0}

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Part 1: Prove that when a = 1, (∁_U P) ∩ Q = {x | 1 < x ≤ 2}
theorem part_one : (Set.compl P) ∩ (Q 1) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

-- Part 2: Prove that P ∩ Q = P if and only if a ∈ [0, 1/2]
theorem part_two : ∀ a : ℝ, P ∩ (Q a) = P ↔ 0 ≤ a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2970_297004


namespace NUMINAMATH_CALUDE_min_Q_zero_at_two_thirds_l2970_297048

/-- The quadratic form representing the expression to be minimized -/
def Q (k : ℝ) (x y : ℝ) : ℝ :=
  5 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 5 * x - 6 * y + 7

/-- The theorem stating that 2/3 is the value of k that makes the minimum of Q zero -/
theorem min_Q_zero_at_two_thirds :
  (∃ (k : ℝ), ∀ (x y : ℝ), Q k x y ≥ 0 ∧ (∃ (x₀ y₀ : ℝ), Q k x₀ y₀ = 0)) ↔ k = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_Q_zero_at_two_thirds_l2970_297048


namespace NUMINAMATH_CALUDE_probability_two_teams_play_l2970_297023

/-- The probability that two specific teams play each other in a single-elimination tournament -/
theorem probability_two_teams_play (n : ℕ) (h : n = 16) : 
  (2 : ℚ) / ((n : ℚ) * (n - 1)) = 1 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_two_teams_play_l2970_297023


namespace NUMINAMATH_CALUDE_power_of_power_l2970_297046

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2970_297046


namespace NUMINAMATH_CALUDE_yellow_light_probability_is_one_twelfth_l2970_297007

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

end NUMINAMATH_CALUDE_yellow_light_probability_is_one_twelfth_l2970_297007


namespace NUMINAMATH_CALUDE_missing_number_equation_l2970_297022

theorem missing_number_equation : ∃ x : ℤ, 1234562 - 12 * 3 * x = 1234490 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l2970_297022


namespace NUMINAMATH_CALUDE_cookies_eaten_l2970_297024

theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (h1 : initial = 28) (h2 : remaining = 7) :
  initial - remaining = 21 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l2970_297024


namespace NUMINAMATH_CALUDE_multiply_by_eleven_l2970_297045

theorem multiply_by_eleven (x : ℝ) : 11 * x = 103.95 → x = 9.45 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_eleven_l2970_297045


namespace NUMINAMATH_CALUDE_m_is_always_odd_l2970_297005

theorem m_is_always_odd (a b : ℤ) (h1 : b = a + 1) (c : ℤ) (h2 : c = a * b) :
  ∃ (M : ℤ), M^2 = a^2 + b^2 + c^2 ∧ Odd M := by
  sorry

end NUMINAMATH_CALUDE_m_is_always_odd_l2970_297005


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2970_297016

theorem complex_magnitude_product : 
  Complex.abs ((Real.sqrt 8 - Complex.I * 2) * (Real.sqrt 3 * 2 + Complex.I * 6)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2970_297016


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2970_297085

theorem complex_division_simplification (z : ℂ) : 
  z = (4 + 3*I) / (1 + 2*I) → z = 2 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2970_297085


namespace NUMINAMATH_CALUDE_election_result_l2970_297094

/-- Represents the number of votes for each candidate in an election --/
structure ElectionResults where
  total_votes : Nat
  john_votes : Nat
  james_percentage : Rat
  john_votes_le_total : john_votes ≤ total_votes

/-- Calculates the difference in votes between the third candidate and John --/
def vote_difference (e : ElectionResults) : Int :=
  e.total_votes - e.john_votes - 
  Nat.floor (e.james_percentage * (e.total_votes - e.john_votes : Rat)) - e.john_votes

/-- Theorem stating the vote difference for the given election scenario --/
theorem election_result : 
  ∀ (e : ElectionResults), 
    e.total_votes = 1150 ∧ 
    e.john_votes = 150 ∧ 
    e.james_percentage = 7/10 → 
    vote_difference e = 150 := by
  sorry


end NUMINAMATH_CALUDE_election_result_l2970_297094


namespace NUMINAMATH_CALUDE_morgans_list_count_l2970_297081

theorem morgans_list_count : ∃ n : ℕ, n = 871 ∧ 
  n = (Finset.range (27000 / 30 + 1) \ Finset.range (900 / 30)).card := by
  sorry

end NUMINAMATH_CALUDE_morgans_list_count_l2970_297081


namespace NUMINAMATH_CALUDE_local_face_value_diff_problem_l2970_297095

/-- The difference between the local value and face value of a digit in a number -/
def local_face_value_diff (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (10 ^ position) - digit

theorem local_face_value_diff_problem : 
  local_face_value_diff 3 3 - local_face_value_diff 7 1 = 2934 := by
  sorry

end NUMINAMATH_CALUDE_local_face_value_diff_problem_l2970_297095


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2970_297077

def M : Set ℤ := {0, 1}
def N : Set ℤ := {x | ∃ y, y = Real.sqrt (1 - x)}

theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2970_297077


namespace NUMINAMATH_CALUDE_max_cross_section_area_l2970_297078

/-- A regular hexagonal prism with side length 8 and vertical edges parallel to the z-axis -/
structure HexagonalPrism where
  side_length : ℝ
  side_length_eq : side_length = 8

/-- The plane that cuts the prism -/
def cutting_plane (x y z : ℝ) : Prop :=
  5 * x - 8 * y + 3 * z = 40

/-- The cross-section formed by cutting the prism with the plane -/
def cross_section (p : HexagonalPrism) (x y z : ℝ) : Prop :=
  cutting_plane x y z

/-- The area of the cross-section -/
noncomputable def cross_section_area (p : HexagonalPrism) : ℝ :=
  sorry

/-- The theorem stating that the maximum area of the cross-section is 144√3 -/
theorem max_cross_section_area (p : HexagonalPrism) :
    ∃ (a : ℝ), cross_section_area p = a ∧ a ≤ 144 * Real.sqrt 3 ∧
    ∀ (b : ℝ), cross_section_area p ≤ b → b ≥ 144 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l2970_297078


namespace NUMINAMATH_CALUDE_work_completion_proof_l2970_297098

/-- The number of days x needs to finish the work alone -/
def x_days : ℕ := 20

/-- The number of days y worked before leaving -/
def y_worked : ℕ := 12

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℕ := 5

/-- The number of days y needs to finish the work alone -/
def y_days : ℕ := 16

theorem work_completion_proof :
  (1 : ℚ) / x_days * x_remaining + (1 : ℚ) / y_days * y_worked = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l2970_297098


namespace NUMINAMATH_CALUDE_polynomial_has_solution_mod_prime_l2970_297073

/-- The polynomial f(x) = x^6 - 11x^4 + 36x^2 - 36 -/
def f (x : ℤ) : ℤ := x^6 - 11*x^4 + 36*x^2 - 36

/-- For any prime p, there exists an x such that f(x) ≡ 0 (mod p) -/
theorem polynomial_has_solution_mod_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ x : ℤ, f x ≡ 0 [ZMOD p] := by sorry

end NUMINAMATH_CALUDE_polynomial_has_solution_mod_prime_l2970_297073


namespace NUMINAMATH_CALUDE_min_lcm_ac_is_90_l2970_297034

def min_lcm_ac (a b c : ℕ) : Prop :=
  (Nat.lcm a b = 20) ∧ (Nat.lcm b c = 18) → Nat.lcm a c ≥ 90

theorem min_lcm_ac_is_90 :
  ∃ (a b c : ℕ), min_lcm_ac a b c ∧ Nat.lcm a c = 90 :=
sorry

end NUMINAMATH_CALUDE_min_lcm_ac_is_90_l2970_297034


namespace NUMINAMATH_CALUDE_complex_roots_on_circle_l2970_297091

theorem complex_roots_on_circle : ∃ (r : ℝ), r = 2 / Real.sqrt 3 ∧
  ∀ (z : ℂ), (z + 2)^6 = 64 * z^6 → Complex.abs (z - Complex.ofReal (2/3)) = r :=
sorry

end NUMINAMATH_CALUDE_complex_roots_on_circle_l2970_297091


namespace NUMINAMATH_CALUDE_complex_norm_squared_l2970_297074

theorem complex_norm_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 3 - 5*I) : 
  Complex.abs z^2 = 17/3 := by
sorry

end NUMINAMATH_CALUDE_complex_norm_squared_l2970_297074


namespace NUMINAMATH_CALUDE_last_remaining_number_l2970_297003

/-- Represents the state of the number sequence --/
structure SequenceState where
  numbers : List Nat
  markStart : Nat

/-- Marks every third number in the sequence --/
def markEveryThird (state : SequenceState) : SequenceState := sorry

/-- Reverses the remaining numbers in the sequence --/
def reverseRemaining (state : SequenceState) : SequenceState := sorry

/-- Performs one round of marking and reversing --/
def performRound (state : SequenceState) : SequenceState := sorry

/-- Continues the process until only one number remains --/
def processUntilOne (state : SequenceState) : Nat := sorry

/-- The main theorem to be proved --/
theorem last_remaining_number :
  processUntilOne { numbers := List.range 120, markStart := 1 } = 57 := by sorry

end NUMINAMATH_CALUDE_last_remaining_number_l2970_297003


namespace NUMINAMATH_CALUDE_lottery_win_probability_l2970_297047

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def winnerBallDrawCount : ℕ := 6

def lotteryWinProbability : ℚ :=
  2 / (megaBallCount * (winnerBallCount.choose winnerBallDrawCount))

theorem lottery_win_probability :
  lotteryWinProbability = 2 / 477621000 := by sorry

end NUMINAMATH_CALUDE_lottery_win_probability_l2970_297047


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l2970_297071

theorem binomial_expansion_example : 16^3 + 3*(16^2)*2 + 3*16*(2^2) + 2^3 = (16 + 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l2970_297071


namespace NUMINAMATH_CALUDE_dinner_tasks_is_four_l2970_297018

/-- Represents Trey's chore list for Sunday -/
structure ChoreList where
  clean_house_tasks : Nat
  shower_tasks : Nat
  dinner_tasks : Nat
  time_per_task : Nat
  total_time : Nat

/-- Calculates the number of dinner tasks given the chore list -/
def calculate_dinner_tasks (chores : ChoreList) : Nat :=
  (chores.total_time - (chores.clean_house_tasks + chores.shower_tasks) * chores.time_per_task) / chores.time_per_task

/-- Theorem stating that the number of dinner tasks is 4 -/
theorem dinner_tasks_is_four (chores : ChoreList) 
  (h1 : chores.clean_house_tasks = 7)
  (h2 : chores.shower_tasks = 1)
  (h3 : chores.time_per_task = 10)
  (h4 : chores.total_time = 120) :
  calculate_dinner_tasks chores = 4 := by
  sorry

#eval calculate_dinner_tasks { clean_house_tasks := 7, shower_tasks := 1, dinner_tasks := 0, time_per_task := 10, total_time := 120 }

end NUMINAMATH_CALUDE_dinner_tasks_is_four_l2970_297018


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reverse_property_l2970_297036

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  let ones := n % 10
  let tens := n / 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_two_digit_prime_with_reverse_property : 
  ∀ n : ℕ, 
    n ≥ 20 ∧ n < 30 ∧ 
    is_prime n ∧
    is_composite (reverse_digits n) ∧
    (reverse_digits n % 3 = 0 ∨ reverse_digits n % 7 = 0) →
    n ≥ 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reverse_property_l2970_297036


namespace NUMINAMATH_CALUDE_printing_presses_theorem_l2970_297015

/-- The number of printing presses used in the first scenario -/
def P : ℕ := 35

/-- The time taken (in hours) by P presses to print 500,000 papers -/
def time_P : ℕ := 15

/-- The number of presses used in the second scenario -/
def presses_2 : ℕ := 25

/-- The time taken (in hours) by presses_2 to print 500,000 papers -/
def time_2 : ℕ := 21

theorem printing_presses_theorem :
  P * time_P = presses_2 * time_2 :=
sorry

end NUMINAMATH_CALUDE_printing_presses_theorem_l2970_297015


namespace NUMINAMATH_CALUDE_inequality_solution_l2970_297049

theorem inequality_solution (x : ℝ) : 
  2 - 1 / (2 * x + 3) < 4 ↔ x < -7/4 ∨ x > -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2970_297049


namespace NUMINAMATH_CALUDE_valid_numbers_l2970_297075

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : ℕ) (m : ℕ) (p : ℕ),
    n = m + 10^k * a + 10^(k+2) * p ∧
    0 ≤ a ∧ a < 100 ∧
    m < 10^k ∧
    n = 87 * (m + 10^k * p) ∧
    n ≥ 10^99 ∧ n < 10^100

theorem valid_numbers :
  {n : ℕ | is_valid_number n} =
    {435 * 10^97, 1305 * 10^96, 2175 * 10^96, 3045 * 10^96} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l2970_297075


namespace NUMINAMATH_CALUDE_expression_evaluation_l2970_297009

theorem expression_evaluation (c d : ℝ) (hc : c = 3) (hd : d = 2) :
  (c^2 + d + 1)^2 - (c^2 - d - 1)^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2970_297009


namespace NUMINAMATH_CALUDE_h_function_proof_l2970_297070

theorem h_function_proof (x : ℝ) (h : ℝ → ℝ) : 
  (12 * x^4 + 4 * x^3 - 2 * x + 3 + h x = 6 * x^3 + 8 * x^2 - 10 * x + 6) →
  (h x = -12 * x^4 + 2 * x^3 + 8 * x^2 - 8 * x + 3) :=
by
  sorry

end NUMINAMATH_CALUDE_h_function_proof_l2970_297070


namespace NUMINAMATH_CALUDE_loan_amount_proof_l2970_297008

/-- The annual interest rate A charges B -/
def interest_rate_A : ℝ := 0.10

/-- The annual interest rate B charges C -/
def interest_rate_B : ℝ := 0.115

/-- The number of years for which the loan is considered -/
def years : ℝ := 3

/-- B's gain over the loan period -/
def gain : ℝ := 1125

/-- The amount lent by A to B -/
def amount : ℝ := 25000

theorem loan_amount_proof :
  gain = (interest_rate_B - interest_rate_A) * years * amount := by sorry

end NUMINAMATH_CALUDE_loan_amount_proof_l2970_297008


namespace NUMINAMATH_CALUDE_prob_ace_of_spades_l2970_297053

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)
  (rank_count : (cards.image Prod.fst).card = 13)
  (suit_count : (cards.image Prod.snd).card = 4)

/-- The probability of drawing a specific card from a shuffled deck -/
def prob_draw_specific_card (d : Deck) : ℚ :=
  1 / 52

/-- Theorem: The probability of drawing the Ace of Spades from a shuffled standard deck is 1/52 -/
theorem prob_ace_of_spades (d : Deck) :
  prob_draw_specific_card d = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_of_spades_l2970_297053


namespace NUMINAMATH_CALUDE_stock_price_change_l2970_297057

theorem stock_price_change (initial_price : ℝ) (h_pos : initial_price > 0) : 
  let price_after_decrease := initial_price * (1 - 0.08)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 1.2 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l2970_297057


namespace NUMINAMATH_CALUDE_rectangle_circle_tangent_l2970_297076

theorem rectangle_circle_tangent (r : ℝ) (w l : ℝ) : 
  r = 6 →  -- Circle radius is 6 cm
  w = 2 * r →  -- Width of rectangle is diameter of circle
  l * w = 3 * (π * r^2) →  -- Area of rectangle is 3 times area of circle
  l = 9 * π :=  -- Length of longer side is 9π cm
by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_tangent_l2970_297076


namespace NUMINAMATH_CALUDE_prob_a_wins_l2970_297002

/-- Given a chess game between players A and B, this theorem proves
    the probability of player A winning, given the probabilities of
    a draw and A not losing. -/
theorem prob_a_wins (prob_draw prob_a_not_lose : ℚ)
  (h_draw : prob_draw = 1/2)
  (h_not_lose : prob_a_not_lose = 5/6) :
  prob_a_not_lose - prob_draw = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_prob_a_wins_l2970_297002


namespace NUMINAMATH_CALUDE_power_mod_congruence_l2970_297050

theorem power_mod_congruence (h : 5^500 ≡ 1 [ZMOD 1000]) :
  5^10000 ≡ 1 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_power_mod_congruence_l2970_297050


namespace NUMINAMATH_CALUDE_sqrt_three_inequality_l2970_297055

theorem sqrt_three_inequality (n : ℕ+) :
  (n : ℝ) + 3 < n * Real.sqrt 3 ∧ n * Real.sqrt 3 < (n : ℝ) + 4 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_inequality_l2970_297055


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_equation_l2970_297020

theorem sum_of_squares_of_roots (a b c : ℚ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * r₁^2 + b * r₁ + c = 0 ∧ a * r₂^2 + b * r₂ + c = 0 →
  r₁^2 + r₂^2 = (b/a)^2 - 2*(c/a) :=
by sorry

theorem sum_of_squares_of_roots_specific_equation :
  let a : ℚ := 5
  let b : ℚ := 6
  let c : ℚ := -15
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁^2 + r₂^2 = 186 / 25 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_equation_l2970_297020


namespace NUMINAMATH_CALUDE_rocky_ran_36_miles_l2970_297006

/-- Rocky's training schedule for the first three days -/
def rocky_training : ℕ → ℕ
| 1 => 4  -- Day one: 4 miles
| 2 => 2 * rocky_training 1  -- Day two: double of day one
| 3 => 3 * rocky_training 2  -- Day three: triple of day two
| _ => 0  -- Other days (not relevant for this problem)

/-- The total miles Rocky ran in the first three days of training -/
def total_miles : ℕ := rocky_training 1 + rocky_training 2 + rocky_training 3

/-- Theorem stating that Rocky ran 36 miles in total during the first three days of training -/
theorem rocky_ran_36_miles : total_miles = 36 := by
  sorry

end NUMINAMATH_CALUDE_rocky_ran_36_miles_l2970_297006


namespace NUMINAMATH_CALUDE_boys_camp_total_l2970_297030

theorem boys_camp_total (total : ℕ) : 
  (total : ℝ) * 0.2 * 0.7 = 21 →
  total = 150 := by
  sorry

end NUMINAMATH_CALUDE_boys_camp_total_l2970_297030


namespace NUMINAMATH_CALUDE_fraction_simplification_l2970_297059

theorem fraction_simplification (a : ℝ) (h : a ≠ 1) :
  a / (a - 1) + 1 / (1 - a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2970_297059
