import Mathlib

namespace line_intersection_y_axis_l675_67529

/-- A line passing through two points intersects the y-axis at a specific point -/
theorem line_intersection_y_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_point1 : x₁ = 3 ∧ y₁ = 18) 
  (h_point2 : x₂ = -7 ∧ y₂ = -2) : 
  ∃ (y : ℝ), y = 12 ∧ 
  (y - y₁) / (0 - x₁) = (y₂ - y₁) / (x₂ - x₁) :=
sorry

end line_intersection_y_axis_l675_67529


namespace parabola_tangent_hyperbola_l675_67511

/-- A parabola is tangent to a hyperbola if and only if m is 4 or 8 -/
theorem parabola_tangent_hyperbola :
  ∀ m : ℝ,
  (∀ x y : ℝ, y = x^2 + 3 ∧ y^2 - m*x^2 = 4 →
    ∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y) ↔ (m = 4 ∨ m = 8) := by
  sorry

end parabola_tangent_hyperbola_l675_67511


namespace acid_dilution_l675_67522

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 40 ∧ 
  initial_concentration = 0.4 ∧ 
  water_added = 24 ∧ 
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by sorry

end acid_dilution_l675_67522


namespace reciprocal_inequality_l675_67567

theorem reciprocal_inequality (a b : ℝ) (ha : a < 0) (hb : b > 0) : 1 / a < 1 / b := by
  sorry

end reciprocal_inequality_l675_67567


namespace notebook_count_l675_67579

theorem notebook_count (n : ℕ) 
  (h1 : n > 0)
  (h2 : n^2 + 20 = (n + 1)^2 - 9) : 
  n^2 + 20 = 216 :=
by sorry

end notebook_count_l675_67579


namespace positive_integer_solutions_m_value_when_x_equals_y_fixed_solution_l675_67515

-- Define the system of equations
def equation1 (x y : ℤ) : Prop := 2*x + y - 6 = 0
def equation2 (x y m : ℤ) : Prop := 2*x - 2*y + m*y + 8 = 0

-- Theorem for part 1
theorem positive_integer_solutions :
  ∀ x y : ℤ, x > 0 ∧ y > 0 ∧ equation1 x y ↔ (x = 2 ∧ y = 2) ∨ (x = 1 ∧ y = 4) :=
sorry

-- Theorem for part 2
theorem m_value_when_x_equals_y :
  ∃ m : ℤ, ∀ x y : ℤ, x = y ∧ equation1 x y ∧ equation2 x y m → m = -4 :=
sorry

-- Theorem for part 3
theorem fixed_solution :
  ∀ m : ℤ, equation2 (-4) 0 m :=
sorry

end positive_integer_solutions_m_value_when_x_equals_y_fixed_solution_l675_67515


namespace restaurant_bill_change_l675_67502

theorem restaurant_bill_change (meal_cost drink_cost tip_percentage bill_amount : ℚ) : 
  meal_cost = 10 ∧ 
  drink_cost = 2.5 ∧ 
  tip_percentage = 0.2 ∧ 
  bill_amount = 20 → 
  bill_amount - (meal_cost + drink_cost + (meal_cost + drink_cost) * tip_percentage) = 5 := by
  sorry

end restaurant_bill_change_l675_67502


namespace typing_contest_orders_l675_67599

/-- The number of different possible orders for a given number of participants to finish a contest without ties. -/
def numberOfOrders (n : ℕ) : ℕ := Nat.factorial n

/-- The number of participants in the typing contest. -/
def numberOfParticipants : ℕ := 4

theorem typing_contest_orders :
  numberOfOrders numberOfParticipants = 24 := by
  sorry

end typing_contest_orders_l675_67599


namespace circle_area_ratio_false_l675_67597

theorem circle_area_ratio_false : 
  ¬ (∀ (r : ℝ), r > 0 → (π * r^2) / (π * (2*r)^2) = 1/2) := by
  sorry

end circle_area_ratio_false_l675_67597


namespace root_equation_problem_l675_67547

/-- Given two polynomial equations with constants c and d, prove that 100c + d = 359 -/
theorem root_equation_problem (c d : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    ((x + c) * (x + d) * (x + 10)) / ((x + 5) * (x + 5)) = 0 ∧
    ((y + c) * (y + d) * (y + 10)) / ((y + 5) * (y + 5)) = 0 ∧
    ((z + c) * (z + d) * (z + 10)) / ((z + 5) * (z + 5)) = 0) ∧
  (∃! w : ℝ, ((w + 2*c) * (w + 7) * (w + 9)) / ((w + d) * (w + 10)) = 0) →
  100 * c + d = 359 := by
sorry

end root_equation_problem_l675_67547


namespace system_solution_l675_67512

theorem system_solution (x : Fin 1995 → ℤ) 
  (h : ∀ i : Fin 1995, x i ^ 2 = 1 + x ((i + 1993) % 1995) * x ((i + 1994) % 1995)) :
  (∀ i : Fin 1995, i % 3 = 1 → x i = 0) ∧
  (∀ i : Fin 1995, i % 3 ≠ 1 → x i = 1 ∨ x i = -1) ∧
  (∀ i : Fin 1995, i % 3 ≠ 1 → x i = -x ((i + 1) % 1995)) :=
by sorry

end system_solution_l675_67512


namespace trigonometric_identity_l675_67535

theorem trigonometric_identity (x y z a : ℝ) 
  (h1 : (Real.cos x + Real.cos y + Real.cos z) / Real.cos (x + y + z) = a)
  (h2 : (Real.sin x + Real.sin y + Real.sin z) / Real.sin (x + y + z) = a) :
  Real.cos (x + y) + Real.cos (y + z) + Real.cos (z + x) = a :=
by sorry

end trigonometric_identity_l675_67535


namespace unique_divisible_by_seven_l675_67558

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 110000 ∧ n % 100 = 1 ∧ (n / 100) % 10 ≠ 0

theorem unique_divisible_by_seven :
  ∃! n : ℕ, is_valid_number n ∧ n % 7 = 0 :=
sorry

end unique_divisible_by_seven_l675_67558


namespace sum_of_digits_of_product_80_nines_80_sevens_l675_67543

/-- A function that returns a natural number consisting of n repetitions of a given digit --/
def repeatDigit (digit : Nat) (n : Nat) : Nat :=
  if n = 0 then 0 else digit + 10 * repeatDigit digit (n - 1)

/-- A function that calculates the sum of digits of a natural number --/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem --/
theorem sum_of_digits_of_product_80_nines_80_sevens :
  sumOfDigits (repeatDigit 9 80 * repeatDigit 7 80) = 720 := by
  sorry


end sum_of_digits_of_product_80_nines_80_sevens_l675_67543


namespace bailey_points_l675_67581

/-- 
Given four basketball players and their scoring relationships, 
prove that Bailey scored 14 points when the team's total score was 54.
-/
theorem bailey_points (bailey akiko michiko chandra : ℕ) : 
  chandra = 2 * akiko →
  akiko = michiko + 4 →
  michiko = bailey / 2 →
  bailey + akiko + michiko + chandra = 54 →
  bailey = 14 := by
sorry

end bailey_points_l675_67581


namespace rectangle_area_increase_l675_67542

theorem rectangle_area_increase (L W : ℝ) (h_positive : L > 0 ∧ W > 0) :
  let original_area := L * W
  let new_area := (1.1 * L) * (1.1 * W)
  (new_area - original_area) / original_area = 0.21 := by
  sorry

end rectangle_area_increase_l675_67542


namespace marys_friends_ages_sum_l675_67521

theorem marys_friends_ages_sum : 
  ∀ (a b c d : ℕ), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →  -- single-digit positive integers
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct
    ((a * b = 28 ∧ c * d = 45) ∨ (a * c = 28 ∧ b * d = 45) ∨ 
     (a * d = 28 ∧ b * c = 45) ∨ (b * c = 28 ∧ a * d = 45) ∨ 
     (b * d = 28 ∧ a * c = 45) ∨ (c * d = 28 ∧ a * b = 45)) →
    a + b + c + d = 25 := by
  sorry

end marys_friends_ages_sum_l675_67521


namespace fuel_consumption_population_l675_67595

/-- Represents a car model -/
structure CarModel where
  name : String

/-- Represents a car of a specific model -/
structure Car where
  model : CarModel

/-- Represents fuel consumption measurement -/
structure FuelConsumption where
  amount : ℝ
  distance : ℝ

/-- Represents a survey of fuel consumption -/
structure FuelConsumptionSurvey where
  model : CarModel
  sample_size : ℕ
  measurements : List FuelConsumption

/-- Definition of population for a fuel consumption survey -/
def survey_population (survey : FuelConsumptionSurvey) : Set FuelConsumption :=
  {fc | ∃ (car : Car), car.model = survey.model ∧ fc.distance = 100}

theorem fuel_consumption_population 
  (survey : FuelConsumptionSurvey) 
  (h1 : survey.sample_size = 20) 
  (h2 : ∀ fc ∈ survey.measurements, fc.distance = 100) :
  survey_population survey = 
    {fc | ∃ (car : Car), car.model = survey.model ∧ fc.distance = 100} := by
  sorry

end fuel_consumption_population_l675_67595


namespace sin_585_degrees_l675_67590

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_585_degrees_l675_67590


namespace toy_poodle_height_is_14_l675_67569

def standard_poodle_height : ℕ := 28

def height_difference_standard_miniature : ℕ := 8

def height_difference_miniature_toy : ℕ := 6

def toy_poodle_height : ℕ := standard_poodle_height - height_difference_standard_miniature - height_difference_miniature_toy

theorem toy_poodle_height_is_14 : toy_poodle_height = 14 := by
  sorry

end toy_poodle_height_is_14_l675_67569


namespace line_slope_one_implies_a_equals_one_l675_67532

/-- Given a line passing through points (-2, a) and (a, 4) with slope 1, prove that a = 1 -/
theorem line_slope_one_implies_a_equals_one (a : ℝ) :
  (4 - a) / (a + 2) = 1 → a = 1 := by
  sorry

end line_slope_one_implies_a_equals_one_l675_67532


namespace sqrt_expression_simplification_l675_67514

theorem sqrt_expression_simplification :
  2 * Real.sqrt 3 - (3 * Real.sqrt 2 + Real.sqrt 3) = Real.sqrt 3 - 3 * Real.sqrt 2 := by
  sorry

end sqrt_expression_simplification_l675_67514


namespace quadratic_function_property_l675_67504

theorem quadratic_function_property (a b c : ℝ) :
  (∀ x, (1 < x ∧ x < c) → (a * x^2 + b * x + c < 0)) →
  a = 1 := by
sorry

end quadratic_function_property_l675_67504


namespace right_triangle_perimeter_l675_67503

theorem right_triangle_perimeter : ∀ (a b c : ℕ),
  a > 0 → b > 0 → c > 0 →
  a = 11 →
  a * a + b * b = c * c →
  a + b + c = 132 :=
by
  sorry

end right_triangle_perimeter_l675_67503


namespace solve_equation_and_evaluate_l675_67513

theorem solve_equation_and_evaluate (x : ℚ) : 
  (4 * x - 3 = 13 * x + 12) → (5 * (x + 4) = 35 / 3) := by
  sorry

end solve_equation_and_evaluate_l675_67513


namespace soccer_ball_surface_area_l675_67563

/-- The surface area of a sphere with circumference 69 cm is 4761/π square cm. -/
theorem soccer_ball_surface_area :
  let circumference : ℝ := 69
  let radius : ℝ := circumference / (2 * Real.pi)
  let surface_area : ℝ := 4 * Real.pi * radius^2
  surface_area = 4761 / Real.pi :=
by sorry

end soccer_ball_surface_area_l675_67563


namespace terms_are_like_l675_67593

-- Define a structure for algebraic terms
structure AlgebraicTerm where
  coefficient : ℤ
  x_exponent : ℕ
  y_exponent : ℕ

-- Define a function to check if two terms are like terms
def are_like_terms (t1 t2 : AlgebraicTerm) : Prop :=
  t1.x_exponent = t2.x_exponent ∧ t1.y_exponent = t2.y_exponent

-- Define the two terms we want to compare
def term1 : AlgebraicTerm := { coefficient := -4, x_exponent := 1, y_exponent := 2 }
def term2 : AlgebraicTerm := { coefficient := 4, x_exponent := 1, y_exponent := 2 }

-- State the theorem
theorem terms_are_like : are_like_terms term1 term2 := by
  sorry

end terms_are_like_l675_67593


namespace no_polyhedron_with_area_2015_l675_67516

theorem no_polyhedron_with_area_2015 : ¬ ∃ (n k : ℕ), 6 * n - 2 * k = 2015 := by
  sorry

end no_polyhedron_with_area_2015_l675_67516


namespace cube_equal_angle_planes_l675_67528

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Calculates the angle between a plane and a line -/
def angle_plane_line (p : Plane) (l : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- Checks if a plane passes through a given point -/
def plane_through_point (p : Plane) (point : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- Theorem: There are exactly 4 planes through vertex A of a cube such that 
    the angles between each plane and the lines AB, AD, and AA₁ are all equal -/
theorem cube_equal_angle_planes (c : Cube) : 
  ∃! (planes : Finset Plane), 
    planes.card = 4 ∧ 
    ∀ p ∈ planes, 
      plane_through_point p (c.vertices 0) ∧
      ∃ θ : ℝ, 
        angle_plane_line p (c.vertices 0, c.vertices 1) = θ ∧
        angle_plane_line p (c.vertices 0, c.vertices 3) = θ ∧
        angle_plane_line p (c.vertices 0, c.vertices 4) = θ :=
  sorry

end cube_equal_angle_planes_l675_67528


namespace parallelogram_area_l675_67544

/-- The area of a parallelogram with base 15 and height 5 is 75 square feet. -/
theorem parallelogram_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 15 ∧ height = 5 → area = base * height → area = 75

/-- Proof of the parallelogram area theorem -/
lemma prove_parallelogram_area : parallelogram_area 15 5 75 := by
  sorry

end parallelogram_area_l675_67544


namespace geometric_series_common_ratio_l675_67526

/-- The first term of the geometric series -/
def a₁ : ℚ := 4 / 7

/-- The second term of the geometric series -/
def a₂ : ℚ := -16 / 21

/-- The third term of the geometric series -/
def a₃ : ℚ := -64 / 63

/-- The common ratio of the geometric series -/
def r : ℚ := -4 / 3

theorem geometric_series_common_ratio :
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by sorry

end geometric_series_common_ratio_l675_67526


namespace sum_remainder_by_six_l675_67545

theorem sum_remainder_by_six : (284917 + 517084) % 6 = 5 := by
  sorry

end sum_remainder_by_six_l675_67545


namespace factorial_fraction_simplification_l675_67565

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N - 1) * N^2) / Nat.factorial (N + 2) = N / (N + 1) := by
  sorry

end factorial_fraction_simplification_l675_67565


namespace f_2014_value_l675_67519

def N0 : Set ℕ := {n : ℕ | n ≥ 0}

def is_valid_f (f : ℕ → ℕ) : Prop :=
  f 2 = 0 ∧
  f 3 > 0 ∧
  f 6042 = 2014 ∧
  ∀ m n : ℕ, (f (m + n) - f m - f n) ∈ ({0, 1} : Set ℕ)

theorem f_2014_value (f : ℕ → ℕ) (h : is_valid_f f) : f 2014 = 671 := by
  sorry

end f_2014_value_l675_67519


namespace largest_common_term_l675_67555

def is_in_first_sequence (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k + 3

def is_in_second_sequence (n : ℕ) : Prop := ∃ m : ℕ, n = 7 * m + 5

theorem largest_common_term :
  (∃ n : ℕ, n < 1000 ∧ is_in_first_sequence n ∧ is_in_second_sequence n) ∧
  (∀ n : ℕ, n < 1000 ∧ is_in_first_sequence n ∧ is_in_second_sequence n → n ≤ 989) ∧
  (is_in_first_sequence 989 ∧ is_in_second_sequence 989) :=
by sorry

end largest_common_term_l675_67555


namespace midpoint_set_properties_l675_67534

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  perimeter : ℝ

/-- The set of midpoints of segments with one end in F and the other in G -/
def midpoint_set (F G : ConvexPolygon) : Set (ℝ × ℝ) := sorry

theorem midpoint_set_properties (F G : ConvexPolygon) :
  let H := midpoint_set F G
  ∃ (sides_H : ℕ) (perimeter_H : ℝ),
    (ConvexPolygon.sides F).max (ConvexPolygon.sides G) ≤ sides_H ∧
    sides_H ≤ (ConvexPolygon.sides F) + (ConvexPolygon.sides G) ∧
    perimeter_H = (ConvexPolygon.perimeter F + ConvexPolygon.perimeter G) / 2 ∧
    (∀ (x y : ℝ × ℝ), x ∈ H → y ∈ H → (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) • x + t • y ∈ H)) :=
by sorry

end midpoint_set_properties_l675_67534


namespace smallest_consecutive_integer_l675_67586

theorem smallest_consecutive_integer (a b c d e : ℤ) : 
  (a + b + c + d + e = 2015) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (d = c + 1) → 
  (e = d + 1) → 
  (a = 401) := by
  sorry

end smallest_consecutive_integer_l675_67586


namespace function_properties_l675_67559

noncomputable section

variable (I : Set ℝ)
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

theorem function_properties
  (h1 : ∀ x ∈ I, 0 < f' x ∧ f' x < 2)
  (h2 : ∀ x ∈ I, f' x ≠ 1)
  (h3 : ∃ c₁ ∈ I, f c₁ = c₁)
  (h4 : ∃ c₂ ∈ I, f c₂ = 2 * c₂)
  (h5 : ∀ a b, a ∈ I → b ∈ I → a ≤ b → ∃ x ∈ Set.Ioo a b, f b - f a = (b - a) * f' x) :
  (∀ x ∈ I, f x = x → x = Classical.choose h3) ∧
  (∀ x > Classical.choose h4, f x < 2 * x) :=
by sorry

end function_properties_l675_67559


namespace sum_reciprocals_equal_negative_two_l675_67551

theorem sum_reciprocals_equal_negative_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y + x * y = 0) : y / x + x / y = -2 := by
  sorry

end sum_reciprocals_equal_negative_two_l675_67551


namespace absolute_value_simplification_l675_67585

theorem absolute_value_simplification (x : ℝ) (h : x < 0) : |3*x + Real.sqrt (x^2)| = -2*x := by
  sorry

end absolute_value_simplification_l675_67585


namespace max_sum_of_squares_l675_67561

theorem max_sum_of_squares (x y z : ℕ+) 
  (h1 : x.val * y.val * z.val = (14 - x.val) * (14 - y.val) * (14 - z.val))
  (h2 : x.val + y.val + z.val < 28) :
  x.val^2 + y.val^2 + z.val^2 ≤ 219 := by
  sorry

end max_sum_of_squares_l675_67561


namespace integral_polynomial_l675_67550

variables (a b c p : ℝ) (x : ℝ)

theorem integral_polynomial (a b c p : ℝ) (x : ℝ) :
  deriv (fun x => (a/4) * x^4 + (b/3) * x^3 + (c/2) * x^2 + p * x) x
  = a * x^3 + b * x^2 + c * x + p :=
by sorry

end integral_polynomial_l675_67550


namespace tom_seashells_l675_67568

def seashells_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

theorem tom_seashells : seashells_remaining 5 2 = 3 := by
  sorry

end tom_seashells_l675_67568


namespace gcd_from_lcm_and_ratio_l675_67523

theorem gcd_from_lcm_and_ratio (X Y : ℕ+) :
  Nat.lcm X Y = 180 →
  (X : ℚ) / (Y : ℚ) = 2 / 5 →
  Nat.gcd X Y = 18 := by
sorry

end gcd_from_lcm_and_ratio_l675_67523


namespace smallest_group_size_l675_67572

theorem smallest_group_size : ∃ n : ℕ, n > 0 ∧ 
  (∃ m : ℕ, m > 2 ∧ n % m = 0) ∧ 
  n % 2 = 0 ∧
  (∀ k : ℕ, k > 0 ∧ (∃ l : ℕ, l > 2 ∧ k % l = 0) ∧ k % 2 = 0 → k ≥ n) ∧
  n = 6 := by
sorry

end smallest_group_size_l675_67572


namespace specific_lamp_arrangement_probability_l675_67530

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def num_lamps_on : ℕ := 4

def total_lamps : ℕ := num_red_lamps + num_blue_lamps

def probability_specific_arrangement : ℚ :=
  1 / 49

theorem specific_lamp_arrangement_probability :
  probability_specific_arrangement = 1 / 49 := by
  sorry

end specific_lamp_arrangement_probability_l675_67530


namespace article_selling_price_l675_67553

def cost_price : ℝ := 250
def profit_percentage : ℝ := 0.60

def selling_price : ℝ := cost_price + (profit_percentage * cost_price)

theorem article_selling_price : selling_price = 400 := by
  sorry

end article_selling_price_l675_67553


namespace prime_quadratic_equation_solution_l675_67524

theorem prime_quadratic_equation_solution (a b Q R : ℕ) : 
  Nat.Prime a → 
  Nat.Prime b → 
  a ≠ b → 
  a^2 - a*Q + R = 0 → 
  b^2 - b*Q + R = 0 → 
  R = 6 := by
sorry

end prime_quadratic_equation_solution_l675_67524


namespace binomial_expansion_coefficient_l675_67576

/-- The coefficient of x^r in the expansion of (1 + ax)^n -/
def binomialCoefficient (n : ℕ) (a : ℝ) (r : ℕ) : ℝ :=
  a^r * (n.choose r)

theorem binomial_expansion_coefficient (n : ℕ) :
  binomialCoefficient n 3 2 = 54 → n = 4 := by
  sorry

end binomial_expansion_coefficient_l675_67576


namespace job_completion_time_l675_67560

theorem job_completion_time (x : ℝ) : 
  x > 0 → 
  5 * (1/x + 1/20) = 1 - 0.41666666666666663 → 
  x = 15 := by
sorry

end job_completion_time_l675_67560


namespace linda_has_34_candies_l675_67549

/-- The number of candies Linda and Chloe have together -/
def total_candies : ℕ := 62

/-- The number of candies Chloe has -/
def chloe_candies : ℕ := 28

/-- The number of candies Linda has -/
def linda_candies : ℕ := total_candies - chloe_candies

theorem linda_has_34_candies : linda_candies = 34 := by
  sorry

end linda_has_34_candies_l675_67549


namespace prob_both_divisible_by_four_is_one_sixteenth_l675_67518

/-- Represents a fair 12-sided die -/
def TwelveSidedDie := Fin 12

/-- The probability of getting a number divisible by 4 on a 12-sided die -/
def prob_divisible_by_four (die : TwelveSidedDie) : ℚ :=
  3 / 12

/-- The probability of getting two numbers divisible by 4 when tossing two 12-sided dice -/
def prob_both_divisible_by_four (die1 die2 : TwelveSidedDie) : ℚ :=
  (prob_divisible_by_four die1) * (prob_divisible_by_four die2)

theorem prob_both_divisible_by_four_is_one_sixteenth :
  ∀ (die1 die2 : TwelveSidedDie), prob_both_divisible_by_four die1 die2 = 1 / 16 := by
  sorry

end prob_both_divisible_by_four_is_one_sixteenth_l675_67518


namespace inequality_solution_set_l675_67594

theorem inequality_solution_set (x : ℝ) :
  (5 * x - 2 ≤ 3 * (1 + x)) ↔ (x ≤ 5 / 2) :=
by sorry

end inequality_solution_set_l675_67594


namespace acute_angle_theorem_l675_67507

def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

theorem acute_angle_theorem (θ : ℝ) (h1 : is_acute_angle θ) 
  (h2 : 4 * (90 - θ) = (180 - θ) + 60) : θ = 40 := by
  sorry

end acute_angle_theorem_l675_67507


namespace sector_angle_l675_67584

/-- Given a circle with radius 12 meters and a sector with area 50.28571428571428 square meters,
    the angle at the center of the circle is 40 degrees. -/
theorem sector_angle (r : ℝ) (area : ℝ) (h1 : r = 12) (h2 : area = 50.28571428571428) :
  (area * 360) / (π * r^2) = 40 := by
  sorry

end sector_angle_l675_67584


namespace total_difference_is_90q_minus_250_l675_67548

/-- The total difference in money between Charles and Richard in cents -/
def total_difference (q : ℤ) : ℤ :=
  let charles_quarters := 6 * q + 2
  let charles_dimes := 3 * q - 2
  let richard_quarters := 2 * q + 10
  let richard_dimes := 4 * q + 3
  let quarter_value := 25
  let dime_value := 10
  (charles_quarters - richard_quarters) * quarter_value + 
  (charles_dimes - richard_dimes) * dime_value

theorem total_difference_is_90q_minus_250 (q : ℤ) : 
  total_difference q = 90 * q - 250 := by
  sorry

end total_difference_is_90q_minus_250_l675_67548


namespace tall_blonde_is_swedish_l675_67541

/-- Represents the nationality of a racer -/
inductive Nationality
| Italian
| Swedish

/-- Represents the physical characteristics of a racer -/
structure Characteristics where
  height : Bool  -- true for tall, false for short
  hair : Bool    -- true for blonde, false for brunette

/-- Represents a racer -/
structure Racer where
  nationality : Nationality
  characteristics : Characteristics

def is_tall_blonde (r : Racer) : Prop :=
  r.characteristics.height ∧ r.characteristics.hair

def is_short_brunette (r : Racer) : Prop :=
  ¬r.characteristics.height ∧ ¬r.characteristics.hair

theorem tall_blonde_is_swedish (racers : Finset Racer) : 
  (∀ r : Racer, r ∈ racers → (is_tall_blonde r → r.nationality = Nationality.Swedish)) :=
by
  sorry

#check tall_blonde_is_swedish

end tall_blonde_is_swedish_l675_67541


namespace polar_to_rectangular_l675_67592

theorem polar_to_rectangular (ρ : ℝ) (θ : ℝ) :
  ρ = 2 ∧ θ = π / 6 →
  ∃ x y : ℝ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ x = Real.sqrt 3 ∧ y = 1 := by
sorry


end polar_to_rectangular_l675_67592


namespace sqrt_two_div_sqrt_half_equals_two_l675_67574

theorem sqrt_two_div_sqrt_half_equals_two : 
  Real.sqrt 2 / Real.sqrt (1/2) = 2 := by sorry

end sqrt_two_div_sqrt_half_equals_two_l675_67574


namespace mary_shirts_problem_l675_67577

theorem mary_shirts_problem (blue_shirts : ℕ) (brown_shirts : ℕ) : 
  brown_shirts = 36 →
  blue_shirts / 2 + brown_shirts * 2 / 3 = 37 →
  blue_shirts = 26 := by
sorry

end mary_shirts_problem_l675_67577


namespace perpendicular_line_to_plane_perpendicular_planes_l675_67509

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- Statement 1
theorem perpendicular_line_to_plane 
  (α : Plane) (a l₁ l₂ : Line) :
  contains α l₁ → 
  contains α l₂ → 
  intersect l₁ l₂ → 
  perpendicular a l₁ → 
  perpendicular a l₂ → 
  perpendicularLP a α :=
sorry

-- Statement 6
theorem perpendicular_planes 
  (α β : Plane) (b : Line) :
  contains β b → 
  perpendicularLP b α → 
  perpendicularPP β α :=
sorry

end perpendicular_line_to_plane_perpendicular_planes_l675_67509


namespace library_books_count_l675_67531

theorem library_books_count : ∀ (total_books : ℕ), 
  (35 : ℚ) / 100 * total_books + 104 = total_books → total_books = 160 :=
by
  sorry

end library_books_count_l675_67531


namespace greatest_divisor_of_exponential_sum_l675_67575

theorem greatest_divisor_of_exponential_sum :
  ∃ (x : ℕ), x > 0 ∧
  (∀ (y : ℕ), y > 0 → (7^y + 12*y - 1) % x = 0) ∧
  (∀ (z : ℕ), z > x → ∃ (w : ℕ), w > 0 ∧ (7^w + 12*w - 1) % z ≠ 0) :=
by
  -- The proof goes here
  sorry

end greatest_divisor_of_exponential_sum_l675_67575


namespace bucket_fill_time_l675_67589

/-- Given that two-thirds of a bucket is filled in 100 seconds,
    prove that it takes 150 seconds to fill the bucket completely. -/
theorem bucket_fill_time :
  let partial_fill_time : ℝ := 100
  let partial_fill_fraction : ℝ := 2/3
  let complete_fill_time : ℝ := 150
  (partial_fill_fraction * complete_fill_time = partial_fill_time) →
  complete_fill_time = 150 :=
by sorry

end bucket_fill_time_l675_67589


namespace bedroom_set_final_price_l675_67598

/-- Calculates the final price of a bedroom set after discounts and gift card application --/
def final_price (initial_price gift_card first_discount second_discount : ℚ) : ℚ :=
  let price_after_first_discount := initial_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount - gift_card

/-- Theorem: The final price of the bedroom set is $1330 --/
theorem bedroom_set_final_price :
  final_price 2000 200 0.15 0.10 = 1330 := by
  sorry

end bedroom_set_final_price_l675_67598


namespace f_s_not_multiplicative_other_l675_67596

/-- r_s(n) is the number of solutions to x_1^2 + x_2^2 + ... + x_s^2 = n in integers x_1, x_2, ..., x_s -/
def r_s (s : ℕ) (n : ℕ) : ℕ := sorry

/-- f_s(n) = (2s)^(-1) * r_s(n) -/
def f_s (s : ℕ) (n : ℕ) : ℚ :=
  (2 * s : ℚ)⁻¹ * (r_s s n : ℚ)

/-- f_s is multiplicative for s = 1, 2, 4, 8 -/
axiom f_s_multiplicative_special (s : ℕ) (m n : ℕ) (h : s = 1 ∨ s = 2 ∨ s = 4 ∨ s = 8) :
  Nat.Coprime m n → f_s s (m * n) = f_s s m * f_s s n

/-- f_s is not multiplicative for any other value of s -/
theorem f_s_not_multiplicative_other (s : ℕ) (h : s ≠ 1 ∧ s ≠ 2 ∧ s ≠ 4 ∧ s ≠ 8) :
  ∃ m n : ℕ, Nat.Coprime m n ∧ f_s s (m * n) ≠ f_s s m * f_s s n := by
  sorry

end f_s_not_multiplicative_other_l675_67596


namespace second_set_is_twenty_feet_l675_67520

/-- The length of the first set of wood in feet -/
def first_set_length : ℝ := 4

/-- The factor by which the second set is longer than the first set -/
def length_factor : ℝ := 5

/-- The length of the second set of wood in feet -/
def second_set_length : ℝ := first_set_length * length_factor

/-- Theorem stating that the second set of wood is 20 feet long -/
theorem second_set_is_twenty_feet : second_set_length = 20 := by
  sorry

end second_set_is_twenty_feet_l675_67520


namespace square_difference_l675_67554

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end square_difference_l675_67554


namespace eraser_difference_l675_67583

/-- Proves that the difference between Rachel's erasers and one-half of Tanya's red erasers is 5 -/
theorem eraser_difference (tanya_total : ℕ) (tanya_red : ℕ) (rachel : ℕ) 
  (h1 : tanya_total = 20)
  (h2 : tanya_red = tanya_total / 2)
  (h3 : rachel = tanya_red) :
  rachel - tanya_red / 2 = 5 := by
  sorry

end eraser_difference_l675_67583


namespace solution_sets_l675_67564

theorem solution_sets (p q : ℝ) : 
  let A := {x : ℝ | 2 * x^2 + x + p = 0}
  let B := {x : ℝ | 2 * x^2 + q * x + 2 = 0}
  (A ∩ B = {1/2}) → 
  (A = {-1, 1/2} ∧ B = {2, 1/2} ∧ A ∪ B = {-1, 2, 1/2}) := by
  sorry

end solution_sets_l675_67564


namespace smaller_number_expression_l675_67562

theorem smaller_number_expression (m n t s : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : t > 1) 
  (h4 : m / n = t) 
  (h5 : m + n = s) : 
  n = s / (1 + t) := by
sorry

end smaller_number_expression_l675_67562


namespace unique_solution_implies_a_value_l675_67566

theorem unique_solution_implies_a_value (a : ℝ) :
  (∃! x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2) → (a = 1 ∨ a = 2) :=
by sorry

end unique_solution_implies_a_value_l675_67566


namespace schedule_ways_eq_840_l675_67533

/-- The number of periods in a day -/
def num_periods : ℕ := 8

/-- The number of mathematics courses -/
def num_courses : ℕ := 4

/-- The number of ways to schedule the mathematics courses -/
def schedule_ways : ℕ := (num_periods - 1).choose num_courses * num_courses.factorial

/-- Theorem stating that the number of ways to schedule the mathematics courses is 840 -/
theorem schedule_ways_eq_840 : schedule_ways = 840 := by sorry

end schedule_ways_eq_840_l675_67533


namespace divisor_35_power_l675_67588

theorem divisor_35_power (k : ℕ) : 35^k ∣ 1575320897 → 7^k - k^7 = 1 := by
  sorry

end divisor_35_power_l675_67588


namespace kenneth_theorem_l675_67506

def kenneth_problem (earnings : ℝ) (joystick_percentage : ℝ) : Prop :=
  let joystick_cost := earnings * (joystick_percentage / 100)
  let remaining := earnings - joystick_cost
  earnings = 450 ∧ joystick_percentage = 10 → remaining = 405

theorem kenneth_theorem : kenneth_problem 450 10 := by
  sorry

end kenneth_theorem_l675_67506


namespace circle_area_ratio_l675_67582

theorem circle_area_ratio : 
  ∀ (r₁ r₂ : ℝ), 
  r₁ > 0 → r₂ > 0 → r₂ = 2 * r₁ →
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 3 :=
by sorry

end circle_area_ratio_l675_67582


namespace divisibility_property_l675_67525

theorem divisibility_property (m : ℕ+) (x : ℝ) :
  ∃ k : ℝ, (x + 1)^(2 * m.val) - x^(2 * m.val) - 2*x - 1 = k * (x * (x + 1) * (2*x + 1)) := by
  sorry

end divisibility_property_l675_67525


namespace proportion_problem_l675_67587

theorem proportion_problem (x : ℝ) : x / 12 = 9 / 360 → x = 0.3 := by
  sorry

end proportion_problem_l675_67587


namespace julie_lawns_mowed_l675_67540

def bike_cost : ℕ := 2345
def initial_savings : ℕ := 1500
def newspapers_delivered : ℕ := 600
def newspaper_pay : ℚ := 0.4
def dogs_walked : ℕ := 24
def dog_walking_pay : ℕ := 15
def lawn_mowing_pay : ℕ := 20
def money_left : ℕ := 155

def total_earned (lawns_mowed : ℕ) : ℚ :=
  initial_savings + newspapers_delivered * newspaper_pay + dogs_walked * dog_walking_pay + lawns_mowed * lawn_mowing_pay

theorem julie_lawns_mowed :
  ∃ (lawns_mowed : ℕ), total_earned lawns_mowed = bike_cost + money_left ∧ lawns_mowed = 20 :=
sorry

end julie_lawns_mowed_l675_67540


namespace boat_speed_is_54_l675_67557

/-- Represents the speed of a boat in still water -/
def boat_speed (v : ℝ) : Prop :=
  ∃ (t : ℝ), t > 0 ∧
  (v - 18) * (2 * t) = (v + 18) * t

/-- Theorem: The speed of the boat in still water is 54 kmph -/
theorem boat_speed_is_54 : boat_speed 54 := by
  sorry

end boat_speed_is_54_l675_67557


namespace hyperbola_equation_l675_67573

/-- A hyperbola with center at the origin -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The equation of a hyperbola -/
def Hyperbola.equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of an asymptote of a hyperbola -/
def Hyperbola.asymptote_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (h.b / h.a) * x ∨ y = -(h.b / h.a) * x

theorem hyperbola_equation (h : Hyperbola) 
  (h_asymptote : h.asymptote_equation 4 3)
  (h_focus : h.a^2 - h.b^2 = 25) :
  h.equation = fun x y => x^2 / 16 - y^2 / 9 = 1 := by sorry

end hyperbola_equation_l675_67573


namespace golu_travel_distance_l675_67580

theorem golu_travel_distance (x : ℝ) :
  x > 0 ∧ x^2 + 6^2 = 10^2 → x = 8 := by
  sorry

end golu_travel_distance_l675_67580


namespace square_sum_of_product_and_sum_l675_67508

theorem square_sum_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
  sorry

end square_sum_of_product_and_sum_l675_67508


namespace ratio_and_equation_solution_l675_67571

/-- Given that x, y, and z are in the ratio 1:4:5, y = 15a - 5, and y = 60, prove that a = 13/3 -/
theorem ratio_and_equation_solution (x y z a : ℚ) 
  (h_ratio : x / y = 1 / 4 ∧ y / z = 4 / 5)
  (h_eq : y = 15 * a - 5)
  (h_y : y = 60) : 
  a = 13 / 3 := by
  sorry

end ratio_and_equation_solution_l675_67571


namespace translation_sum_l675_67556

/-- Given two points P and Q in a 2D plane, where P is translated m units left
    and n units up to obtain Q, prove that m + n = 4. -/
theorem translation_sum (P Q : ℝ × ℝ) (m n : ℝ) : 
  P = (-1, -3) → Q = (-2, 0) → Q.1 = P.1 - m → Q.2 = P.2 + n → m + n = 4 := by
  sorry

end translation_sum_l675_67556


namespace hyperbola_eccentricity_l675_67536

/-- Given a hyperbola with equation y²/16 - x²/m = 1 and eccentricity e = 2, prove that m = 48 -/
theorem hyperbola_eccentricity (m : ℝ) (e : ℝ) :
  (∀ x y : ℝ, y^2 / 16 - x^2 / m = 1) →
  e = 2 →
  m = 48 :=
by sorry

end hyperbola_eccentricity_l675_67536


namespace product_quality_probability_l675_67578

theorem product_quality_probability (p_B p_C : ℝ) 
  (h_B : p_B = 0.03) 
  (h_C : p_C = 0.02) : 
  1 - (p_B + p_C) = 0.95 := by
  sorry

end product_quality_probability_l675_67578


namespace triangle_properties_l675_67546

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  b * c * (Real.cos A) = 4 ∧
  a * c * (Real.sin B) = 8 * (Real.sin A) →
  A = π / 3 ∧ 
  0 < Real.sin A * Real.sin B * Real.sin C ∧ 
  Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8 :=
by sorry

end triangle_properties_l675_67546


namespace cube_root_simplification_l675_67591

theorem cube_root_simplification : 
  (50^3 + 60^3 + 70^3 : ℝ)^(1/3) = 10 * 684^(1/3) := by sorry

end cube_root_simplification_l675_67591


namespace metal_sheet_dimensions_l675_67505

theorem metal_sheet_dimensions (a : ℝ) :
  (a > 0) →
  (2*a > 6) →
  (a > 6) →
  (3 * (2*a - 6) * (a - 6) = 168) →
  (a = 10) := by
sorry

end metal_sheet_dimensions_l675_67505


namespace close_interval_is_two_to_three_l675_67552

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

-- Define the close function property
def is_close (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- Theorem statement
theorem close_interval_is_two_to_three :
  ∀ a b : ℝ, a ≤ 2 ∧ 3 ≤ b → (is_close f g a b ↔ a = 2 ∧ b = 3) :=
sorry

end close_interval_is_two_to_three_l675_67552


namespace competitive_exam_selection_difference_l675_67500

theorem competitive_exam_selection_difference (total_candidates : ℕ) 
  (selection_rate_A : ℚ) (selection_rate_B : ℚ) : 
  total_candidates = 7900 → 
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B - selection_rate_A) * total_candidates = 79 := by
sorry

end competitive_exam_selection_difference_l675_67500


namespace intersection_of_P_and_Q_l675_67537

def P : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def Q : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end intersection_of_P_and_Q_l675_67537


namespace sailboat_speed_proof_l675_67501

/-- The speed of a sailboat with two sails in knots -/
def speed_two_sails : ℝ := 50

/-- The conversion factor from nautical miles to land miles -/
def nautical_to_land : ℝ := 1.15

/-- The time spent sailing with one sail in hours -/
def time_one_sail : ℝ := 4

/-- The time spent sailing with two sails in hours -/
def time_two_sails : ℝ := 4

/-- The total distance traveled in land miles -/
def total_distance_land : ℝ := 345

/-- The speed of the sailboat with one sail in knots -/
def speed_one_sail : ℝ := 25

theorem sailboat_speed_proof :
  speed_one_sail * time_one_sail + speed_two_sails * time_two_sails =
  total_distance_land / nautical_to_land := by
  sorry

end sailboat_speed_proof_l675_67501


namespace unique_root_of_cubic_l675_67539

/-- The function f(x) = (x-3)(x^2+2x+3) has exactly one real root. -/
theorem unique_root_of_cubic (x : ℝ) : ∃! a : ℝ, (a - 3) * (a^2 + 2*a + 3) = 0 := by
  sorry

end unique_root_of_cubic_l675_67539


namespace ice_cream_arrangement_l675_67517

theorem ice_cream_arrangement (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 2) :
  (n! / k!) = 60 := by
  sorry

end ice_cream_arrangement_l675_67517


namespace cubic_function_property_l675_67510

/-- A cubic function with integer coefficients -/
def f (a b c : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Theorem: If f(a) = a^3 and f(b) = b^3, then c = 16 -/
theorem cubic_function_property (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h1 : f a b c a = a^3) (h2 : f a b c b = b^3) : c = 16 := by
  sorry

end cubic_function_property_l675_67510


namespace perfectSquareFactorsOf360_l675_67538

def perfectSquareFactors (n : ℕ) : ℕ := sorry

theorem perfectSquareFactorsOf360 : perfectSquareFactors 360 = 4 := by
  sorry

end perfectSquareFactorsOf360_l675_67538


namespace union_of_A_and_B_l675_67570

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 1 < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Ioo (-1) 2 := by sorry

end union_of_A_and_B_l675_67570


namespace odd_multiple_of_nine_is_multiple_of_three_l675_67527

theorem odd_multiple_of_nine_is_multiple_of_three (S : ℤ) :
  Odd S → (∃ k : ℤ, S = 9 * k) → (∃ m : ℤ, S = 3 * m) := by
  sorry

end odd_multiple_of_nine_is_multiple_of_three_l675_67527
