import Mathlib

namespace NUMINAMATH_CALUDE_number_of_lineups_l915_91551

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of regular players in the starting lineup
def regular_players : ℕ := 11

-- Define the number of goalies in the starting lineup
def goalies : ℕ := 1

-- Theorem stating the number of different starting lineups
theorem number_of_lineups : 
  (total_players.choose goalies) * ((total_players - goalies).choose regular_players) = 222768 := by
  sorry

end NUMINAMATH_CALUDE_number_of_lineups_l915_91551


namespace NUMINAMATH_CALUDE_train_length_l915_91581

/-- The length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 82 →
  person_speed = 6 →
  passing_time = 4.499640028797696 →
  ∃ (length : ℝ), abs (length - 110) < 0.5 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l915_91581


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l915_91566

theorem quadratic_equation_equivalence : 
  (∀ x, x^2 - 2*(3*x - 2) + (x + 1) = 0 ↔ x^2 - 5*x + 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l915_91566


namespace NUMINAMATH_CALUDE_solve_for_b_l915_91535

theorem solve_for_b (y : ℝ) (b : ℝ) (h1 : y > 0) 
  (h2 : (6 * y) / b + (3 * y) / 10 = 0.60 * y) : b = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l915_91535


namespace NUMINAMATH_CALUDE_greatest_a_for_inequality_l915_91597

theorem greatest_a_for_inequality : 
  ∃ (a : ℝ), (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ a * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅)) ∧ 
  (∀ (b : ℝ), (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ b * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅)) → b ≤ a) ∧ 
  a = 2 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_greatest_a_for_inequality_l915_91597


namespace NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_wire_l915_91553

theorem min_sum_of_squares (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 4 → x^2 + y^2 ≥ 8 := by sorry

theorem min_sum_of_squares_wire :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 4 ∧ x^2 + y^2 = 8 := by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_min_sum_of_squares_wire_l915_91553


namespace NUMINAMATH_CALUDE_max_sum_with_length_constraint_l915_91546

-- Define the length function
def length (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem max_sum_with_length_constraint :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ length x + length y = 16 ∧ 
  ∀ (a b : ℕ), a > 1 → b > 1 → length a + length b = 16 → 
  a + 3 * b ≤ x + 3 * y ∧ x + 3 * y = 98306 :=
sorry

end NUMINAMATH_CALUDE_max_sum_with_length_constraint_l915_91546


namespace NUMINAMATH_CALUDE_investment_calculation_l915_91508

theorem investment_calculation (total : ℝ) (ratio : ℝ) (mutual_funds : ℝ) (bonds : ℝ) :
  total = 240000 ∧ 
  mutual_funds = ratio * bonds ∧ 
  ratio = 6 ∧ 
  total = mutual_funds + bonds →
  mutual_funds = 205714.29 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l915_91508


namespace NUMINAMATH_CALUDE_grasshopper_final_position_l915_91584

/-- The number of positions in the circular arrangement -/
def num_positions : ℕ := 6

/-- The number of jumps the grasshopper makes -/
def num_jumps : ℕ := 100

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The final position of the grasshopper after num_jumps -/
def final_position : ℕ := (sum_first_n num_jumps) % num_positions + 1

theorem grasshopper_final_position :
  final_position = 5 := by sorry

end NUMINAMATH_CALUDE_grasshopper_final_position_l915_91584


namespace NUMINAMATH_CALUDE_bobby_candy_count_l915_91575

/-- The number of candy pieces Bobby had initially -/
def initial_candy : ℕ := 22

/-- The number of candy pieces Bobby ate at the start -/
def eaten_start : ℕ := 9

/-- The number of additional candy pieces Bobby ate -/
def eaten_additional : ℕ := 5

/-- The number of candy pieces Bobby has left -/
def remaining_candy : ℕ := 8

/-- Theorem stating that Bobby's initial candy count is correct -/
theorem bobby_candy_count : 
  initial_candy = eaten_start + eaten_additional + remaining_candy :=
by sorry

end NUMINAMATH_CALUDE_bobby_candy_count_l915_91575


namespace NUMINAMATH_CALUDE_robin_seeds_count_robin_seeds_is_150_l915_91564

theorem robin_seeds_count : ℕ → ℕ → Prop :=
  fun (robin_bushes sparrow_bushes : ℕ) =>
    (robin_bushes = sparrow_bushes + 5) →
    (5 * robin_bushes = 6 * sparrow_bushes) →
    (5 * robin_bushes = 150)

/-- The number of seeds hidden by the robin is 150 -/
theorem robin_seeds_is_150 : ∃ (robin_bushes sparrow_bushes : ℕ),
  robin_seeds_count robin_bushes sparrow_bushes :=
by
  sorry

#check robin_seeds_is_150

end NUMINAMATH_CALUDE_robin_seeds_count_robin_seeds_is_150_l915_91564


namespace NUMINAMATH_CALUDE_inequality_theorem_l915_91502

open Real

theorem inequality_theorem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x > 0, f x > 0)
  (h2 : ∀ x, deriv f x < f x)
  (h3 : 0 < a ∧ a < 1) :
  3 * f 0 > f a ∧ f a > a * f 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l915_91502


namespace NUMINAMATH_CALUDE_conic_section_k_range_l915_91595

/-- Represents a conic section of the form x^2/2 + y^2/k = 1 -/
structure ConicSection (k : ℝ) where
  equation : ∀ (x y : ℝ), x^2/2 + y^2/k = 1

/-- Proposition p: The conic section is an ellipse with foci on the x-axis -/
def is_ellipse_x_foci (k : ℝ) : Prop :=
  0 < k ∧ k < 2

/-- Proposition q: The eccentricity of the conic section is in the interval (√2, √3) -/
def eccentricity_in_range (k : ℝ) : Prop :=
  let e := Real.sqrt ((2 - k) / 2)
  Real.sqrt 2 < e ∧ e < Real.sqrt 3

/-- The main theorem -/
theorem conic_section_k_range (k : ℝ) (E : ConicSection k) :
  (¬is_ellipse_x_foci k) ∧ eccentricity_in_range k → -4 < k ∧ k < -2 :=
sorry

end NUMINAMATH_CALUDE_conic_section_k_range_l915_91595


namespace NUMINAMATH_CALUDE_graph_relationship_l915_91560

theorem graph_relationship (x : ℝ) : |x^2 - 3/2*x + 3| ≥ x^2 + 3/2*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_graph_relationship_l915_91560


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_element5_l915_91585

theorem pascal_triangle_row20_element5 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_element5_l915_91585


namespace NUMINAMATH_CALUDE_cans_per_bag_l915_91511

theorem cans_per_bag (total_cans : ℕ) (num_bags : ℕ) (h1 : total_cans = 20) (h2 : num_bags = 4) :
  total_cans / num_bags = 5 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_bag_l915_91511


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l915_91573

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 - a*x - b < 0}) :
  Set.Ioo (-1/2 : ℝ) (-1/3) = {x : ℝ | b*x^2 - a*x - 1 > 0} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l915_91573


namespace NUMINAMATH_CALUDE_charles_speed_l915_91593

/-- Charles' stroll scenario -/
def charles_stroll (distance : ℝ) (time : ℝ) : Prop :=
  distance = 6 ∧ time = 2 ∧ distance / time = 3

theorem charles_speed : ∃ (distance time : ℝ), charles_stroll distance time :=
  sorry

end NUMINAMATH_CALUDE_charles_speed_l915_91593


namespace NUMINAMATH_CALUDE_marbles_exceed_200_l915_91557

def marbles (n : ℕ) : ℕ := 3 * 2^(n - 1)

theorem marbles_exceed_200 :
  ∀ k : ℕ, k < 9 → marbles k ≤ 200 ∧ marbles 9 > 200 :=
by sorry

end NUMINAMATH_CALUDE_marbles_exceed_200_l915_91557


namespace NUMINAMATH_CALUDE_multiple_properties_l915_91587

theorem multiple_properties (a b : ℤ) 
  (ha : 4 ∣ a) (hb : 8 ∣ b) : 
  (4 ∣ b) ∧ (4 ∣ (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_multiple_properties_l915_91587


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l915_91514

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : 
  (1 + 1 / (x + 1)) * ((x + 1) / (x^2 + 4)) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l915_91514


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l915_91532

/-- Given two points A and B in the plane, this theorem states that
    the equation 4x - 2y - 5 = 0 represents the perpendicular bisector
    of the line segment connecting A and B. -/
theorem perpendicular_bisector_equation (A B : ℝ × ℝ) :
  A = (1, 2) →
  B = (3, 1) →
  ∀ (x y : ℝ), (4 * x - 2 * y - 5 = 0) ↔
    (((x - 1)^2 + (y - 2)^2 = (x - 3)^2 + (y - 1)^2) ∧
     ((y - 2) * (3 - 1) = -(x - 1) * (1 - 2))) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l915_91532


namespace NUMINAMATH_CALUDE_inequality_theorem_l915_91583

-- Define the functions p and q
variable (p q : ℝ → ℝ)

-- Define the theorem
theorem inequality_theorem 
  (h1 : Differentiable ℝ p) 
  (h2 : Differentiable ℝ q)
  (h3 : p 0 = q 0)
  (h4 : p 0 > 0)
  (h5 : ∀ x ∈ Set.Icc 0 1, deriv p x * Real.sqrt (deriv q x) = Real.sqrt 2) :
  ∀ x ∈ Set.Icc 0 1, p x + 2 * q x > 3 * x := by
sorry


end NUMINAMATH_CALUDE_inequality_theorem_l915_91583


namespace NUMINAMATH_CALUDE_conference_arrangements_l915_91582

/-- The number of ways to arrange n distinct elements --/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n distinct elements with k pairs having a specific order requirement --/
def arrangementsWithOrderRequirements (n : ℕ) (k : ℕ) : ℕ :=
  arrangements n / (2^k)

/-- Theorem stating that arranging 7 lecturers with 2 order requirements results in 1260 possible arrangements --/
theorem conference_arrangements : arrangementsWithOrderRequirements 7 2 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_conference_arrangements_l915_91582


namespace NUMINAMATH_CALUDE_logarithm_simplification_l915_91519

theorem logarithm_simplification (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  Real.log (Real.cos x * Real.tan x + 1 - 2 * Real.sin (x / 2) ^ 2) +
  Real.log (Real.sqrt 2 * Real.cos (x - Real.pi / 4)) -
  Real.log (1 + Real.sin (2 * x)) = 0 := by sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l915_91519


namespace NUMINAMATH_CALUDE_f_zero_values_l915_91571

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = 2 * f x * f y

/-- The theorem stating the possible values of f(0) -/
theorem f_zero_values (f : ℝ → ℝ) (h : FunctionalEquation f) :
    f 0 = 0 ∨ f 0 = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_values_l915_91571


namespace NUMINAMATH_CALUDE_polynomial_simplification_l915_91541

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 3 * x - 15) = x^2 + 5 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l915_91541


namespace NUMINAMATH_CALUDE_common_area_rotated_squares_l915_91547

/-- The area of the region common to two squares with side length 2, 
    where one is rotated about a vertex by an angle θ such that cos θ = 3/5 -/
theorem common_area_rotated_squares (θ : Real) : 
  θ.cos = 3/5 → 
  (2 : Real) > 0 → 
  (4 * θ.cos * θ.sin : Real) = 48/25 := by
  sorry

end NUMINAMATH_CALUDE_common_area_rotated_squares_l915_91547


namespace NUMINAMATH_CALUDE_order_relation_l915_91599

theorem order_relation (a b c : ℝ) (ha : a = Real.log (1 + Real.exp 1))
    (hb : b = Real.sqrt (Real.exp 1)) (hc : c = (2 * Real.exp 1) / 3) :
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_order_relation_l915_91599


namespace NUMINAMATH_CALUDE_distance_time_relationship_l915_91534

/-- Represents the speed of a car in km/h -/
def speed : ℝ := 70

/-- Represents the distance traveled by the car in km -/
def distance (t : ℝ) : ℝ := speed * t

/-- Theorem stating the relationship between distance and time for the car -/
theorem distance_time_relationship (t : ℝ) : 
  distance t = speed * t ∧ 
  (∃ (S : ℝ → ℝ), S = distance ∧ (∀ x, S x = speed * x)) := by
  sorry

/-- The independent variable is time -/
def independent_variable : Type := ℝ

/-- The dependent variable is distance -/
def dependent_variable : ℝ → ℝ := distance

end NUMINAMATH_CALUDE_distance_time_relationship_l915_91534


namespace NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_36_l915_91549

theorem one_and_two_thirds_of_x_is_36 (x : ℝ) : (5/3) * x = 36 → x = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_36_l915_91549


namespace NUMINAMATH_CALUDE_textbook_weight_difference_l915_91530

theorem textbook_weight_difference :
  let chemistry_weight : Float := 7.12
  let geometry_weight : Float := 0.62
  (chemistry_weight - geometry_weight) = 6.50 := by
  sorry

end NUMINAMATH_CALUDE_textbook_weight_difference_l915_91530


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l915_91510

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l915_91510


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l915_91577

theorem arccos_equation_solution :
  ∃ x : ℝ, x = -1/3 ∧ Real.arccos (3*x) - Real.arccos (2*x) = π/6 :=
by sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l915_91577


namespace NUMINAMATH_CALUDE_ursula_annual_salary_l915_91580

/-- Calculates the annual salary given hourly wage, hours per day, and days per month -/
def annual_salary (hourly_wage : ℝ) (hours_per_day : ℝ) (days_per_month : ℝ) : ℝ :=
  hourly_wage * hours_per_day * days_per_month * 12

/-- Proves that Ursula's annual salary is $16,320 given her work conditions -/
theorem ursula_annual_salary :
  annual_salary 8.50 8 20 = 16320 := by
  sorry

end NUMINAMATH_CALUDE_ursula_annual_salary_l915_91580


namespace NUMINAMATH_CALUDE_correct_fraction_l915_91576

theorem correct_fraction (number : ℕ) (x y : ℕ) (h1 : number = 192) 
  (h2 : (5 : ℚ) / 6 * number = x / y * number + 100) : x / y = (5 : ℚ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_fraction_l915_91576


namespace NUMINAMATH_CALUDE_some_number_value_l915_91536

theorem some_number_value : ∃ (n : ℚ), n = 10/3 ∧ 
  (3 + 2 * (3/2 : ℚ))^5 = (1 + n * (3/2 : ℚ))^4 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l915_91536


namespace NUMINAMATH_CALUDE_f_of_3_equals_4_l915_91590

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2

-- State the theorem
theorem f_of_3_equals_4 : f 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_4_l915_91590


namespace NUMINAMATH_CALUDE_households_with_only_bike_l915_91579

theorem households_with_only_bike (total : ℕ) (without_car_or_bike : ℕ) (with_both : ℕ) (with_car : ℕ) :
  total = 90 →
  without_car_or_bike = 11 →
  with_both = 14 →
  with_car = 44 →
  ∃ (with_only_bike : ℕ), with_only_bike = 35 ∧
    total = without_car_or_bike + with_both + (with_car - with_both) + with_only_bike :=
by sorry

end NUMINAMATH_CALUDE_households_with_only_bike_l915_91579


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l915_91594

theorem cubic_equation_roots :
  ∃ (r₁ r₂ r₃ : ℝ), r₁ < 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
  ∀ x : ℝ, x^3 - 2*x^2 - 5*x + 6 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l915_91594


namespace NUMINAMATH_CALUDE_polynomial_sum_l915_91572

-- Define the polynomial P
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) :
  P a b c d 1 = 2000 →
  P a b c d 2 = 4000 →
  P a b c d 3 = 6000 →
  P a b c d 9 + P a b c d (-5) = 12704 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l915_91572


namespace NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l915_91545

-- Define the possible solid figures
inductive SolidFigure
  | Cone
  | Cylinder
  | TriangularPyramid
  | RectangularPrism

-- Define a predicate for having a quadrilateral front view
def has_quadrilateral_front_view (s : SolidFigure) : Prop :=
  match s with
  | SolidFigure.Cylinder => True
  | SolidFigure.RectangularPrism => True
  | _ => False

-- Theorem statement
theorem quadrilateral_front_view_solids (s : SolidFigure) :
  has_quadrilateral_front_view s ↔ (s = SolidFigure.Cylinder ∨ s = SolidFigure.RectangularPrism) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l915_91545


namespace NUMINAMATH_CALUDE_isosceles_triangle_conditions_l915_91528

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that each of the following conditions implies that the triangle is isosceles. -/
theorem isosceles_triangle_conditions (a b c A B C : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C)
  (h_angle_sum : A + B + C = Real.pi) : 
  (a * Real.cos B = b * Real.cos A → a = b ∨ b = c ∨ a = c) ∧ 
  (Real.cos B * Real.cos C = (1 - Real.cos A) / 2 → a = b ∨ b = c ∨ a = c) ∧
  (a / Real.sin B + b / Real.sin A ≤ 2 * c → a = b ∨ b = c ∨ a = c) := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_conditions_l915_91528


namespace NUMINAMATH_CALUDE_fraction_multiplication_addition_l915_91523

theorem fraction_multiplication_addition : (2 / 9 : ℚ) * (5 / 6 : ℚ) + (1 / 18 : ℚ) = (13 / 54 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_addition_l915_91523


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l915_91556

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 6th term of the arithmetic sequence is 11 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_a2 : a 2 = 3)
    (h_sum : a 3 + a 5 = 14) :
  a 6 = 11 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l915_91556


namespace NUMINAMATH_CALUDE_value_of_3x_minus_y_l915_91500

-- Define the augmented matrix
def augmented_matrix : Matrix (Fin 2) (Fin 3) ℚ := !![2, 1, 5; 1, -2, 0]

-- Define the system of equations
def system_equations (x y : ℚ) : Prop :=
  2 * x + y = 5 ∧ x - 2 * y = 0

-- Theorem statement
theorem value_of_3x_minus_y :
  ∃ x y : ℚ, system_equations x y → 3 * x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_3x_minus_y_l915_91500


namespace NUMINAMATH_CALUDE_bob_start_time_l915_91548

/-- Proves that Bob started walking 1 hour after Yolanda, given the conditions of the problem. -/
theorem bob_start_time (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ) (bob_distance : ℝ) :
  total_distance = 10 →
  yolanda_rate = 3 →
  bob_rate = 4 →
  bob_distance = 4 →
  ∃ (bob_start_time : ℝ),
    bob_start_time = 1 ∧
    bob_start_time * bob_rate + yolanda_rate * (bob_start_time + bob_distance / bob_rate) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_bob_start_time_l915_91548


namespace NUMINAMATH_CALUDE_bill_denomination_proof_l915_91552

/-- Given the cost of berries, cost of peaches, and the amount of change received,
    prove that the denomination of the bill used equals the sum of these three amounts. -/
theorem bill_denomination_proof 
  (cost_berries : ℚ) 
  (cost_peaches : ℚ) 
  (change_received : ℚ) 
  (h1 : cost_berries = 719/100)
  (h2 : cost_peaches = 683/100)
  (h3 : change_received = 598/100) :
  cost_berries + cost_peaches + change_received = 20 := by
  sorry

end NUMINAMATH_CALUDE_bill_denomination_proof_l915_91552


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_diff_l915_91554

theorem cube_sum_given_sum_and_diff (a b : ℝ) (h1 : a + b = 12) (h2 : a - b = 4) :
  a^3 + b^3 = 1344 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_diff_l915_91554


namespace NUMINAMATH_CALUDE_two_numbers_sum_diff_product_l915_91501

theorem two_numbers_sum_diff_product : ∃ (x y : ℝ), 
  x + y = 24 ∧ x - y = 8 ∧ x * y > 100 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_diff_product_l915_91501


namespace NUMINAMATH_CALUDE_sphere_only_all_circular_views_l915_91521

/-- Enumeration of common geometric bodies -/
inductive GeometricBody
  | Cone
  | Cylinder
  | Sphere
  | HollowCylinder

/-- Definition for a view of a geometric body being circular -/
def isCircularView (body : GeometricBody) (view : String) : Prop := sorry

/-- Theorem stating that only a sphere has all circular views -/
theorem sphere_only_all_circular_views (body : GeometricBody) :
  (isCircularView body "front" ∧ 
   isCircularView body "side" ∧ 
   isCircularView body "top") ↔ 
  body = GeometricBody.Sphere :=
sorry

end NUMINAMATH_CALUDE_sphere_only_all_circular_views_l915_91521


namespace NUMINAMATH_CALUDE_win_sector_area_l915_91522

/-- The area of the WIN sector on a circular spinner with given radius and win probability -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 8) (h_p : p = 3/7) :
  p * π * r^2 = (192 * π) / 7 := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l915_91522


namespace NUMINAMATH_CALUDE_potato_cost_proof_l915_91533

/-- The original cost of one bag of potatoes from the farmer -/
def original_cost : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase factor -/
def andrey_increase : ℝ := 2

/-- Boris's first price increase factor -/
def boris_first_increase : ℝ := 1.6

/-- Boris's second price increase factor -/
def boris_second_increase : ℝ := 1.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The extra profit Boris made compared to Andrey -/
def extra_profit : ℝ := 1200

theorem potato_cost_proof :
  bags_bought * original_cost * andrey_increase +
  extra_profit =
  boris_first_sale * original_cost * boris_first_increase +
  boris_second_sale * original_cost * boris_first_increase * boris_second_increase :=
by sorry

end NUMINAMATH_CALUDE_potato_cost_proof_l915_91533


namespace NUMINAMATH_CALUDE_remainder_is_18_l915_91567

/-- A cubic polynomial p(x) with coefficients a and b. -/
def p (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + a * x^2 + b * x + 12

/-- The theorem stating that the remainder when p(x) is divided by x-1 is 18. -/
theorem remainder_is_18 (a b : ℝ) :
  (x + 2 ∣ p a b x) → (x - 3 ∣ p a b x) → p a b 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_18_l915_91567


namespace NUMINAMATH_CALUDE_numerator_increase_percentage_l915_91526

theorem numerator_increase_percentage (P : ℝ) : 
  (5 * (1 + P / 100)) / (7 * (1 - 10 / 100)) = 20 / 21 → P = 20 := by
  sorry

end NUMINAMATH_CALUDE_numerator_increase_percentage_l915_91526


namespace NUMINAMATH_CALUDE_exactly_one_sick_probability_l915_91520

/-- The probability of an employee being sick on any given day -/
def prob_sick : ℚ := 1 / 40

/-- The probability of an employee not being sick on any given day -/
def prob_not_sick : ℚ := 1 - prob_sick

/-- The probability of exactly one out of three employees being sick -/
def prob_one_sick_out_of_three : ℚ :=
  3 * prob_sick * prob_not_sick * prob_not_sick

theorem exactly_one_sick_probability :
  prob_one_sick_out_of_three = 4563 / 64000 := by sorry

end NUMINAMATH_CALUDE_exactly_one_sick_probability_l915_91520


namespace NUMINAMATH_CALUDE_hiking_problem_l915_91540

/-- Hiking problem statement -/
theorem hiking_problem (endpoint_distance : ℝ) (speed_ratio : ℝ) (head_start : ℝ) (meet_time : ℝ) 
  (planned_time : ℝ) (early_arrival : ℝ) :
  endpoint_distance = 7.5 →
  speed_ratio = 1.5 →
  head_start = 0.75 →
  meet_time = 0.5 →
  planned_time = 1 →
  early_arrival = 1/6 →
  ∃ (speed_a speed_b actual_time : ℝ),
    speed_a = 4.5 ∧
    speed_b = 3 ∧
    actual_time = 4/3 ∧
    speed_a = speed_ratio * speed_b ∧
    (speed_a - speed_b) * meet_time = head_start ∧
    endpoint_distance / speed_b - early_arrival = planned_time + (endpoint_distance - speed_b * planned_time) / speed_a :=
by sorry


end NUMINAMATH_CALUDE_hiking_problem_l915_91540


namespace NUMINAMATH_CALUDE_expression_evaluation_l915_91504

theorem expression_evaluation : 
  let x := Real.sqrt ((9^9 + 3^12) / (9^5 + 3^13))
  ∃ ε > 0, abs (x - 15.3) < ε ∧ ε < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l915_91504


namespace NUMINAMATH_CALUDE_odd_integers_divisibility_l915_91592

theorem odd_integers_divisibility (a b : ℕ) : 
  Odd a → Odd b → a > 0 → b > 0 → (2 * a * b + 1) ∣ (a^2 + b^2 + 1) → a = b := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_divisibility_l915_91592


namespace NUMINAMATH_CALUDE_root_order_quadratic_equations_l915_91524

theorem root_order_quadratic_equations (m : ℝ) (a b c d : ℝ) 
  (hm : m > 0)
  (h1 : a^2 - m*a - 1 = 0)
  (h2 : b^2 - m*b - 1 = 0)
  (h3 : c^2 + m*c - 1 = 0)
  (h4 : d^2 + m*d - 1 = 0)
  (ha : a > 0)
  (hb : b < 0)
  (hc : c > 0)
  (hd : d < 0) :
  abs a > abs c ∧ abs c > abs b ∧ abs b > abs d :=
sorry

end NUMINAMATH_CALUDE_root_order_quadratic_equations_l915_91524


namespace NUMINAMATH_CALUDE_cube_root_negative_equals_negative_cube_root_l915_91506

theorem cube_root_negative_equals_negative_cube_root (x : ℝ) (h : x > 0) :
  ((-x : ℝ) ^ (1/3 : ℝ)) = -(x ^ (1/3 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_cube_root_negative_equals_negative_cube_root_l915_91506


namespace NUMINAMATH_CALUDE_angle_complement_l915_91596

theorem angle_complement (given_angle : ℝ) (straight_angle : ℝ) :
  given_angle = 13 →
  straight_angle = 180 →
  (straight_angle - (13 * given_angle)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_l915_91596


namespace NUMINAMATH_CALUDE_binomial_distribution_problem_l915_91589

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  ξ : ℝ

/-- The expectation of a binomial variable -/
def expectation (X : BinomialVariable) : ℝ := X.n * X.p

/-- The variance of a binomial variable -/
def variance (X : BinomialVariable) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_problem (X : BinomialVariable) 
  (h1 : expectation X = 300)
  (h2 : variance X = 200) :
  X.p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_problem_l915_91589


namespace NUMINAMATH_CALUDE_max_expensive_price_is_11000_l915_91505

/-- Represents a company's product line -/
structure ProductLine where
  num_products : ℕ
  average_price : ℝ
  min_price : ℝ
  num_below_threshold : ℕ
  price_threshold : ℝ

/-- The maximum possible price for the most expensive product -/
def max_expensive_price (pl : ProductLine) : ℝ :=
  let total_price := pl.num_products * pl.average_price
  let min_price_sum := pl.num_below_threshold * pl.min_price
  let remaining_price := total_price - min_price_sum
  let remaining_products := pl.num_products - pl.num_below_threshold
  remaining_price - (remaining_products - 1) * pl.price_threshold

/-- Theorem stating the maximum price of the most expensive product -/
theorem max_expensive_price_is_11000 (c : ProductLine) 
  (h1 : c.num_products = 20)
  (h2 : c.average_price = 1200)
  (h3 : c.min_price = 400)
  (h4 : c.num_below_threshold = 10)
  (h5 : c.price_threshold = 1000) :
  max_expensive_price c = 11000 := by
  sorry


end NUMINAMATH_CALUDE_max_expensive_price_is_11000_l915_91505


namespace NUMINAMATH_CALUDE_math_books_count_l915_91542

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) :
  total_books = 90 →
  math_cost = 4 →
  history_cost = 5 →
  total_price = 396 →
  ∃ (math_books : ℕ),
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧
    math_books = 54 :=
by sorry

end NUMINAMATH_CALUDE_math_books_count_l915_91542


namespace NUMINAMATH_CALUDE_absolute_value_equality_l915_91544

theorem absolute_value_equality (m : ℝ) : |m| = |-7| → m = 7 ∨ m = -7 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l915_91544


namespace NUMINAMATH_CALUDE_fraction_simplification_l915_91517

theorem fraction_simplification (x : ℝ) (h : x = 10) :
  (x^6 - 100*x^3 + 2500) / (x^3 - 50) = 950 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l915_91517


namespace NUMINAMATH_CALUDE_machine_worked_twelve_minutes_l915_91531

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  shirts_per_minute : ℕ
  shirts_made_today : ℕ

/-- Calculate the number of minutes the machine worked today -/
def minutes_worked_today (machine : ShirtMachine) : ℕ :=
  machine.shirts_made_today / machine.shirts_per_minute

/-- Theorem: The machine worked for 12 minutes today -/
theorem machine_worked_twelve_minutes 
  (machine : ShirtMachine) 
  (h1 : machine.shirts_per_minute = 6)
  (h2 : machine.shirts_made_today = 72) : 
  minutes_worked_today machine = 12 := by
  sorry

#eval minutes_worked_today ⟨6, 72⟩

end NUMINAMATH_CALUDE_machine_worked_twelve_minutes_l915_91531


namespace NUMINAMATH_CALUDE_fractional_equation_simplification_l915_91539

theorem fractional_equation_simplification (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : 
  (x / (x - 1) = 3 / (2 * x - 2) - 3) ↔ (2 * x = 3 - 6 * x + 6) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_simplification_l915_91539


namespace NUMINAMATH_CALUDE_tens_digit_equals_number_of_tens_l915_91537

theorem tens_digit_equals_number_of_tens (n : ℕ) (h : 10 ≤ n ∧ n ≤ 999) : 
  (n / 10) % 10 = n / 10 - (n / 100) * 10 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_equals_number_of_tens_l915_91537


namespace NUMINAMATH_CALUDE_complex_number_location_l915_91563

theorem complex_number_location (z : ℂ) (h : z * (1 + Complex.I)^2 = 1 - Complex.I) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l915_91563


namespace NUMINAMATH_CALUDE_chefs_wage_difference_l915_91562

/-- Proves that the difference between the total hourly wage of 3 managers
    and the total hourly wage of 3 chefs is $3.9375, given the specified conditions. -/
theorem chefs_wage_difference (manager_wage : ℝ) (num_chefs num_dishwashers : ℕ) :
  manager_wage = 8.5 →
  num_chefs = 3 →
  num_dishwashers = 4 →
  let first_dishwasher_wage := manager_wage / 2
  let dishwasher_wages := [
    first_dishwasher_wage,
    first_dishwasher_wage + 1.5,
    first_dishwasher_wage + 3,
    first_dishwasher_wage + 4.5
  ]
  let chef_wages := (List.take num_chefs dishwasher_wages).map (λ w => w * 1.25)
  (3 * manager_wage - chef_wages.sum) = 3.9375 := by
  sorry

end NUMINAMATH_CALUDE_chefs_wage_difference_l915_91562


namespace NUMINAMATH_CALUDE_alpha_beta_range_l915_91565

theorem alpha_beta_range (α β : ℝ) 
  (h1 : 0 < α - β) (h2 : α - β < π) 
  (h3 : 0 < α + 2*β) (h4 : α + 2*β < π) : 
  0 < α + β ∧ α + β < π :=
by sorry

end NUMINAMATH_CALUDE_alpha_beta_range_l915_91565


namespace NUMINAMATH_CALUDE_dog_park_ratio_l915_91515

theorem dog_park_ratio (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_ear_dogs : ℕ) :
  pointy_ear_dogs = total_dogs / 5 →
  pointy_ear_dogs = 6 →
  spotted_dogs = 15 →
  (spotted_dogs : ℚ) / total_dogs = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dog_park_ratio_l915_91515


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l915_91570

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l915_91570


namespace NUMINAMATH_CALUDE_absolute_value_equality_l915_91513

theorem absolute_value_equality : |5 - 3| = -(3 - 5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l915_91513


namespace NUMINAMATH_CALUDE_subtracted_value_l915_91588

theorem subtracted_value (chosen_number : ℕ) (x : ℚ) : 
  chosen_number = 120 → (chosen_number / 6 : ℚ) - x = 5 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l915_91588


namespace NUMINAMATH_CALUDE_student_position_l915_91527

theorem student_position (total_students : ℕ) (position_from_back : ℕ) (position_from_front : ℕ) :
  total_students = 27 →
  position_from_back = 13 →
  position_from_front = total_students - position_from_back + 1 →
  position_from_front = 15 :=
by sorry

end NUMINAMATH_CALUDE_student_position_l915_91527


namespace NUMINAMATH_CALUDE_positive_A_value_l915_91558

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- State the theorem
theorem positive_A_value (A : ℝ) (h : hash A 3 = 130) : A = 11 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l915_91558


namespace NUMINAMATH_CALUDE_fashion_show_duration_l915_91598

def fashion_show_time (num_models : ℕ) (bathing_suits_per_model : ℕ) (evening_wear_per_model : ℕ) (time_per_trip : ℕ) : ℕ :=
  num_models * (bathing_suits_per_model + evening_wear_per_model) * time_per_trip

theorem fashion_show_duration :
  fashion_show_time 6 2 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_fashion_show_duration_l915_91598


namespace NUMINAMATH_CALUDE_right_triangle_sin_A_l915_91574

theorem right_triangle_sin_A (A B C : ℝ) : 
  -- ABC is a right triangle
  A + B + C = Real.pi ∧ A = Real.pi / 2 →
  -- sin B = 3/5
  Real.sin B = 3 / 5 →
  -- sin C = 4/5
  Real.sin C = 4 / 5 →
  -- sin A = 1
  Real.sin A = 1 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_A_l915_91574


namespace NUMINAMATH_CALUDE_speed_at_40_degrees_l915_91586

-- Define the relationship between temperature and speed
def temperature (s : ℝ) : ℝ := 5 * s^2 + 20 * s + 15

-- Theorem statement
theorem speed_at_40_degrees : 
  ∃ (s₁ s₂ : ℝ), s₁ ≠ s₂ ∧ temperature s₁ = 40 ∧ temperature s₂ = 40 ∧ 
  ((s₁ = 1 ∧ s₂ = -5) ∨ (s₁ = -5 ∧ s₂ = 1)) := by
  sorry

end NUMINAMATH_CALUDE_speed_at_40_degrees_l915_91586


namespace NUMINAMATH_CALUDE_roberts_reading_capacity_l915_91512

def reading_speed : ℝ := 75
def book_length : ℝ := 300
def available_time : ℝ := 9

theorem roberts_reading_capacity :
  ⌊available_time / (book_length / reading_speed)⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_roberts_reading_capacity_l915_91512


namespace NUMINAMATH_CALUDE_bamboo_volume_proof_l915_91525

theorem bamboo_volume_proof (a : ℕ → ℚ) :
  (∀ i : ℕ, i < 8 → a (i + 1) - a i = a (i + 2) - a (i + 1)) →  -- arithmetic progression
  a 1 + a 2 + a 3 + a 4 = 3 →                                   -- sum of first 4 terms
  a 7 + a 8 + a 9 = 4 →                                         -- sum of last 3 terms
  a 5 + a 6 = 31/9 := by
sorry

end NUMINAMATH_CALUDE_bamboo_volume_proof_l915_91525


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l915_91543

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | 2*x - 3 > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 3/2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l915_91543


namespace NUMINAMATH_CALUDE_optimal_characterization_l915_91559

def Ω : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 2008}

def better (p q : ℝ × ℝ) : Prop := p.1 ≤ q.1 ∧ p.2 ≥ q.2

def optimal (q : ℝ × ℝ) : Prop :=
  q ∈ Ω ∧ ∀ p ∈ Ω, ¬(better p q ∧ p ≠ q)

theorem optimal_characterization (q : ℝ × ℝ) :
  optimal q ↔ q.1^2 + q.2^2 = 2008 ∧ q.1 ≤ 0 ∧ q.2 ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_optimal_characterization_l915_91559


namespace NUMINAMATH_CALUDE_compound_interest_approximation_l915_91568

/-- Approximation of compound interest using Binomial Theorem -/
theorem compound_interest_approximation
  (K : ℝ) (p : ℝ) (n : ℕ) :
  let r := p / 100
  let Kn := K * (1 + r)^n
  let approx := K * (1 + n*r + (n*(n-1)/2) * r^2 + (n*(n-1)*(n-2)/6) * r^3)
  ∃ (ε : ℝ), ε > 0 ∧ |Kn - approx| < ε * Kn :=
sorry

end NUMINAMATH_CALUDE_compound_interest_approximation_l915_91568


namespace NUMINAMATH_CALUDE_geometry_propositions_l915_91555

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) : 
  (p₁ ∧ p₄) ∧ ¬(p₁ ∧ p₂) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l915_91555


namespace NUMINAMATH_CALUDE_function_properties_l915_91538

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if x < c then c * x + 1 else 3 * x^4 + x^2 * c

theorem function_properties (c : ℝ) 
  (h1 : 0 < c) (h2 : c < 1) (h3 : f c c^2 = 9/8) :
  c = 1/2 ∧ ∀ x, f (1/2) x < 2 ↔ 0 < x ∧ x < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l915_91538


namespace NUMINAMATH_CALUDE_modulus_of_z_l915_91507

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - Complex.I) :
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l915_91507


namespace NUMINAMATH_CALUDE_three_in_A_even_not_in_A_l915_91569

-- Define the set A
def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}

-- Theorem statements
theorem three_in_A : 3 ∈ A := by sorry

theorem even_not_in_A : ∀ k : ℤ, (4*k - 2) ∉ A := by sorry

end NUMINAMATH_CALUDE_three_in_A_even_not_in_A_l915_91569


namespace NUMINAMATH_CALUDE_john_supermarket_spending_l915_91503

theorem john_supermarket_spending : 
  ∀ (total : ℚ),
  (1 / 2 : ℚ) * total + (1 / 3 : ℚ) * total + (1 / 10 : ℚ) * total + 5 = total →
  total = 75 := by
sorry

end NUMINAMATH_CALUDE_john_supermarket_spending_l915_91503


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l915_91561

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 1)

theorem vector_magnitude_proof : ‖(2 • a) + b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l915_91561


namespace NUMINAMATH_CALUDE_min_value_theorem_l915_91518

/-- Given that f(x) = a^x - b and g(x) = x + 1, where a > 0, a ≠ 1, and b ∈ ℝ,
    if f(x) * g(x) ≤ 0 for all real x, then the minimum value of 1/a + 4/b is 4 -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  (∀ x : ℝ, (a^x - b) * (x + 1) ≤ 0) →
  (∃ m : ℝ, m = 4 ∧ ∀ a b : ℝ, a > 0 → a ≠ 1 → (∀ x : ℝ, (a^x - b) * (x + 1) ≤ 0) → 1/a + 4/b ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l915_91518


namespace NUMINAMATH_CALUDE_machine_work_time_l915_91550

/-- Given machines A, B, and C, where B takes 3 hours and C takes 6 hours to complete a job,
    and all three machines together take 4/3 hours, prove that A takes 4 hours alone. -/
theorem machine_work_time (time_B time_C time_ABC : ℝ) (time_A : ℝ) : 
  time_B = 3 → 
  time_C = 6 → 
  time_ABC = 4/3 → 
  1/time_A + 1/time_B + 1/time_C = 1/time_ABC → 
  time_A = 4 := by
sorry

end NUMINAMATH_CALUDE_machine_work_time_l915_91550


namespace NUMINAMATH_CALUDE_spider_dressing_theorem_l915_91591

/-- The number of legs of the spider -/
def num_legs : ℕ := 10

/-- The number of items per leg -/
def items_per_leg : ℕ := 3

/-- The total number of items -/
def total_items : ℕ := num_legs * items_per_leg

/-- The number of possible arrangements of items for one leg -/
def arrangements_per_leg : ℕ := Nat.factorial items_per_leg

theorem spider_dressing_theorem :
  (Nat.factorial total_items) / (arrangements_per_leg ^ num_legs) = 
  (Nat.factorial (num_legs * items_per_leg)) / (Nat.factorial items_per_leg ^ num_legs) := by
  sorry

end NUMINAMATH_CALUDE_spider_dressing_theorem_l915_91591


namespace NUMINAMATH_CALUDE_max_value_a_l915_91509

theorem max_value_a : ∃ (a : ℝ) (b : ℤ), 
  (a * b^2) / (a + 2 * ↑b) = 2019 ∧ 
  ∀ (a' : ℝ) (b' : ℤ), (a' * b'^2) / (a' + 2 * ↑b') = 2019 → a' ≤ a :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l915_91509


namespace NUMINAMATH_CALUDE_washing_time_is_seven_hours_l915_91516

/-- Calculates the number of cycles needed for a given number of items and capacity per cycle -/
def cycles_needed (items : ℕ) (capacity : ℕ) : ℕ :=
  (items + capacity - 1) / capacity

/-- Calculates the total washing time in minutes -/
def total_washing_time (shirts pants sweaters jeans socks scarves : ℕ) 
  (regular_capacity sock_capacity scarf_capacity : ℕ)
  (regular_time sock_time scarf_time : ℕ) : ℕ :=
  let regular_cycles := cycles_needed shirts regular_capacity + 
                        cycles_needed pants regular_capacity + 
                        cycles_needed sweaters regular_capacity + 
                        cycles_needed jeans regular_capacity
  let sock_cycles := cycles_needed socks sock_capacity
  let scarf_cycles := cycles_needed scarves scarf_capacity
  regular_cycles * regular_time + sock_cycles * sock_time + scarf_cycles * scarf_time

theorem washing_time_is_seven_hours :
  total_washing_time 18 12 17 13 10 8 15 10 5 45 30 60 = 7 * 60 := by
  sorry

end NUMINAMATH_CALUDE_washing_time_is_seven_hours_l915_91516


namespace NUMINAMATH_CALUDE_remaining_water_fills_glasses_l915_91529

theorem remaining_water_fills_glasses (total_water : ℕ) (glass_5oz : ℕ) (glass_8oz : ℕ) (glass_4oz : ℕ) :
  total_water = 122 →
  glass_5oz = 6 →
  glass_8oz = 4 →
  glass_4oz * 4 = total_water - (glass_5oz * 5 + glass_8oz * 8) →
  glass_4oz = 15 := by
sorry

end NUMINAMATH_CALUDE_remaining_water_fills_glasses_l915_91529


namespace NUMINAMATH_CALUDE_problem_solution_l915_91578

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + 2 * y - 6 = 0
def equation2 (x y m : ℝ) : Prop := x - 2 * y + m * x + 5 = 0

theorem problem_solution :
  -- Part 1: Positive integer solutions
  (∀ x y : ℕ+, equation1 x y ↔ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 1)) ∧
  -- Part 2: Value of m when x + y = 0
  (∀ x y m : ℝ, x + y = 0 → equation1 x y → equation2 x y m → m = -13/6) ∧
  -- Part 3: Fixed solution regardless of m
  (∀ m : ℝ, equation2 0 (5/2) m) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l915_91578
