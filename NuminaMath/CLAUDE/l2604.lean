import Mathlib

namespace NUMINAMATH_CALUDE_sum_equals_60_l2604_260453

/-- An arithmetic sequence with specific terms. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  a3_eq_4 : a 3 = 4
  a101_eq_36 : a 101 = 36

/-- The sum of specific terms in the arithmetic sequence equals 60. -/
theorem sum_equals_60 (seq : ArithmeticSequence) : seq.a 9 + seq.a 52 + seq.a 95 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_60_l2604_260453


namespace NUMINAMATH_CALUDE_john_painting_area_l2604_260497

/-- The area John needs to paint on a wall -/
def areaToPaint (wallHeight wallLength paintingWidth paintingHeight : ℝ) : ℝ :=
  wallHeight * wallLength - paintingWidth * paintingHeight

/-- Theorem: John needs to paint 135 square feet -/
theorem john_painting_area :
  areaToPaint 10 15 3 5 = 135 := by
sorry

end NUMINAMATH_CALUDE_john_painting_area_l2604_260497


namespace NUMINAMATH_CALUDE_symmetric_points_coordinate_sum_l2604_260431

/-- Given two points P and Q symmetric with respect to the origin O in a Cartesian coordinate system, 
    prove that the sum of their x and y coordinates is -4. -/
theorem symmetric_points_coordinate_sum (p q : ℝ) : 
  (∃ (P Q : ℝ × ℝ), P = (p, -2) ∧ Q = (6, q) ∧ P.1 = -Q.1 ∧ P.2 = -Q.2) →
  p + q = -4 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_coordinate_sum_l2604_260431


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l2604_260487

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State the theorem
theorem perpendicular_transitivity 
  (h1 : parallel α β) 
  (h2 : parallel β γ) 
  (h3 : perpendicular m α) : 
  perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l2604_260487


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l2604_260479

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  2 * x + 1 / (x - 1) ≥ 2 + 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 1) :
  2 * x + 1 / (x - 1) = 2 + 2 * Real.sqrt 2 ↔ x = 1 + Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l2604_260479


namespace NUMINAMATH_CALUDE_boys_average_weight_l2604_260406

/-- Proves that the average weight of boys in a class is 48 kg given the specified conditions -/
theorem boys_average_weight (total_students : Nat) (num_boys : Nat) (num_girls : Nat)
  (class_avg_weight : ℝ) (girls_avg_weight : ℝ) :
  total_students = 25 →
  num_boys = 15 →
  num_girls = 10 →
  class_avg_weight = 45 →
  girls_avg_weight = 40.5 →
  (total_students * class_avg_weight - num_girls * girls_avg_weight) / num_boys = 48 := by
  sorry

end NUMINAMATH_CALUDE_boys_average_weight_l2604_260406


namespace NUMINAMATH_CALUDE_square_field_area_l2604_260447

/-- Calculates the area of a square field given the cost of barbed wire around it -/
theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) : 
  wire_cost_per_meter = 3 →
  total_cost = 1998 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ), 
    wire_cost_per_meter * (4 * side_length - num_gates * gate_width) = total_cost ∧
    side_length ^ 2 = 27889 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l2604_260447


namespace NUMINAMATH_CALUDE_geometric_progression_exists_l2604_260440

theorem geometric_progression_exists : ∃ (a b c d : ℚ) (e f g h : ℚ),
  (b = a * (-4) ∧ c = b * (-4) ∧ d = c * (-4) ∧
   b = a - 35 ∧ c = d + 560) ∧
  (f = e * 4 ∧ g = f * 4 ∧ h = g * 4 ∧
   f = e - 35 ∧ g = h + 560) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_exists_l2604_260440


namespace NUMINAMATH_CALUDE_five_digit_number_counts_l2604_260415

def digits := [0, 1, 2, 3, 4]

/-- Count of five-digit numbers without repeated digits using 0, 1, 2, 3, and 4
    that are greater than 21035 and even -/
def count_greater_than_21035_and_even : ℕ := 39

/-- Count of five-digit even numbers without repeated digits using 0, 1, 2, 3, and 4
    with the second and fourth digits from the left being odd numbers -/
def count_even_with_odd_second_and_fourth : ℕ := 8

/-- Theorem stating the counts of numbers satisfying the given conditions -/
theorem five_digit_number_counts :
  (count_greater_than_21035_and_even = 39) ∧
  (count_even_with_odd_second_and_fourth = 8) := by
  sorry

end NUMINAMATH_CALUDE_five_digit_number_counts_l2604_260415


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l2604_260488

/-- Given a circle with radius 3 cm tangent to three sides of a rectangle,
    and the area of the rectangle being three times the area of the circle,
    the length of the longer side of the rectangle is 4.5π cm. -/
theorem rectangle_longer_side (circle_radius : ℝ) (rectangle_area : ℝ) (circle_area : ℝ) :
  circle_radius = 3 →
  rectangle_area = 3 * circle_area →
  circle_area = π * circle_radius^2 →
  (4.5 * π : ℝ) * (2 * circle_radius) = rectangle_area :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l2604_260488


namespace NUMINAMATH_CALUDE_ellipse_equation_l2604_260441

/-- Given an ellipse with one focus at (√3, 0) and a = 2b, its standard equation is x²/4 + y² = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a = 2*b) (h2 : a^2 - b^2 = 3) :
  ∀ x y : ℝ, x^2/4 + y^2 = 1 ↔ x^2/(4*b^2) + y^2/b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2604_260441


namespace NUMINAMATH_CALUDE_polynomial_identity_l2604_260420

/-- 
Given a natural number n, define a bivariate polynomial P(x, y) 
that satisfies P(u+v, w) + P(v+w, u) + P(w+u, v) = 0 for all u, v, w.
-/
theorem polynomial_identity (n : ℕ) :
  ∃ P : ℝ → ℝ → ℝ, 
    (∀ x y : ℝ, P x y = (x + y)^(n-1) * (x - 2*y)) ∧
    (∀ u v w : ℝ, P (u+v) w + P (v+w) u + P (w+u) v = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2604_260420


namespace NUMINAMATH_CALUDE_wrappers_minus_caps_difference_l2604_260400

/-- Represents the number of bottle caps Danny found at the park. -/
def bottle_caps_found : ℕ := 11

/-- Represents the number of wrappers Danny found at the park. -/
def wrappers_found : ℕ := 28

/-- Represents the total number of bottle caps in Danny's collection. -/
def total_bottle_caps : ℕ := 68

/-- Represents the total number of wrappers in Danny's collection. -/
def total_wrappers : ℕ := 51

/-- Theorem stating the difference between wrappers and bottle caps found at the park. -/
theorem wrappers_minus_caps_difference : wrappers_found - bottle_caps_found = 17 := by
  sorry

end NUMINAMATH_CALUDE_wrappers_minus_caps_difference_l2604_260400


namespace NUMINAMATH_CALUDE_gcd_problem_l2604_260470

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 997) :
  Int.gcd (3 * b^2 + 34 * b + 102) (b + 21) = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2604_260470


namespace NUMINAMATH_CALUDE_fifth_rest_day_is_monday_l2604_260455

def day_of_week (n : ℕ) : ℕ := n % 7 + 1

def rest_day (n : ℕ) : ℕ := 4 * n - 2

theorem fifth_rest_day_is_monday :
  day_of_week (rest_day 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fifth_rest_day_is_monday_l2604_260455


namespace NUMINAMATH_CALUDE_problem_solution_l2604_260467

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : Real.sqrt x / Real.sqrt y - Real.sqrt y / Real.sqrt x = 7/12)
  (h2 : x - y = 7) :
  x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2604_260467


namespace NUMINAMATH_CALUDE_samson_utility_l2604_260442

/-- Represents the utility function for Samson's activities -/
def utility (math : ℝ) (frisbee : ℝ) : ℝ := math * frisbee

/-- Represents the conditions of the problem -/
theorem samson_utility (t : ℝ) : 
  utility (8 - t) t = utility (t + 3) (2 - t) → t = 2/3 := by
  sorry

#check samson_utility

end NUMINAMATH_CALUDE_samson_utility_l2604_260442


namespace NUMINAMATH_CALUDE_nested_fraction_equals_27_over_73_l2604_260471

theorem nested_fraction_equals_27_over_73 :
  1 / (3 - 1 / (3 + 1 / (3 - 1 / 3))) = 27 / 73 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equals_27_over_73_l2604_260471


namespace NUMINAMATH_CALUDE_composite_function_equality_l2604_260459

/-- Given two functions f and g, and a real number b, proves that if f(g(b)) = 3,
    then b = 1/2. -/
theorem composite_function_equality (f g : ℝ → ℝ) (b : ℝ) 
    (hf : ∀ x, f x = x / 4 + 2)
    (hg : ∀ x, g x = 5 - 2 * x)
    (h : f (g b) = 3) : b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_equality_l2604_260459


namespace NUMINAMATH_CALUDE_janet_running_average_l2604_260450

theorem janet_running_average (total_miles : ℕ) (num_days : ℕ) (miles_per_day : ℕ) : 
  total_miles = 72 → num_days = 9 → miles_per_day = total_miles / num_days → miles_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_janet_running_average_l2604_260450


namespace NUMINAMATH_CALUDE_base6_subtraction_431_254_l2604_260490

/-- Represents a number in base 6 using a list of digits -/
def Base6 : Type := List Nat

/-- Converts a base 6 number to its decimal representation -/
def to_decimal (n : Base6) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Subtracts two base 6 numbers -/
def base6_sub (a b : Base6) : Base6 :=
  sorry -- Implementation details omitted

theorem base6_subtraction_431_254 :
  base6_sub [1, 3, 4] [4, 5, 2] = [3, 3, 1] :=
by sorry

end NUMINAMATH_CALUDE_base6_subtraction_431_254_l2604_260490


namespace NUMINAMATH_CALUDE_factorization_equality_l2604_260452

theorem factorization_equality (a : ℝ) : 2 * a^2 - 2 * a + (1/2 : ℝ) = 2 * (a - 1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2604_260452


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2604_260409

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define proposition Q
def prop_Q (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, |f' x| < 2017

-- Define proposition P
def prop_P (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → |(f x₁ - f x₂) / (x₁ - x₂)| < 2017

-- State the theorem
theorem necessary_not_sufficient
  (hf : Differentiable ℝ f)
  (hf' : ∀ x, HasDerivAt f (f' x) x) :
  (prop_Q f' → prop_P f) ∧ ¬(prop_P f → prop_Q f') :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2604_260409


namespace NUMINAMATH_CALUDE_ru_length_l2604_260414

/-- Triangle PQR with given side lengths -/
structure Triangle (P Q R : ℝ × ℝ) where
  pq_length : dist P Q = 13
  qr_length : dist Q R = 30
  rp_length : dist R P = 26

/-- Point S on PR such that QS bisects angle PQR -/
def S (P Q R : ℝ × ℝ) (tri : Triangle P Q R) : ℝ × ℝ :=
  sorry

/-- Point T on the circumcircle of PQR, different from Q, such that QT bisects angle PQR -/
def T (P Q R : ℝ × ℝ) (tri : Triangle P Q R) : ℝ × ℝ :=
  sorry

/-- Point U on PQ, different from P, such that U is on the circumcircle of PTS -/
def U (P Q R : ℝ × ℝ) (tri : Triangle P Q R) : ℝ × ℝ :=
  sorry

/-- The main theorem stating that RU = 34 -/
theorem ru_length (P Q R : ℝ × ℝ) (tri : Triangle P Q R) :
  dist R (U P Q R tri) = 34 :=
sorry

end NUMINAMATH_CALUDE_ru_length_l2604_260414


namespace NUMINAMATH_CALUDE_eighth_number_in_list_l2604_260405

theorem eighth_number_in_list (numbers : List ℕ) : 
  numbers.length = 9 ∧ 
  (numbers.sum : ℚ) / numbers.length = 60 ∧
  numbers.count 54 = 1 ∧
  numbers.count 55 = 1 ∧
  numbers.count 57 = 1 ∧
  numbers.count 58 = 1 ∧
  numbers.count 59 = 1 ∧
  numbers.count 62 = 2 ∧
  numbers.count 65 = 2 →
  numbers.count 53 = 1 := by
sorry

end NUMINAMATH_CALUDE_eighth_number_in_list_l2604_260405


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l2604_260494

/-- The area of an equilateral triangle with perimeter 3p is (√3/4) * p^2. -/
theorem equilateral_triangle_area (p : ℝ) (p_pos : p > 0) : 
  let perimeter := 3 * p
  let side_length := perimeter / 3
  let area := (Real.sqrt 3 / 4) * side_length^2
  area = (Real.sqrt 3 / 4) * p^2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l2604_260494


namespace NUMINAMATH_CALUDE_ab_value_l2604_260401

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2604_260401


namespace NUMINAMATH_CALUDE_john_bought_36_rolls_l2604_260425

def price_per_dozen : ℕ := 5
def amount_spent : ℕ := 15
def rolls_per_dozen : ℕ := 12

theorem john_bought_36_rolls : 
  amount_spent / price_per_dozen * rolls_per_dozen = 36 := by
  sorry

end NUMINAMATH_CALUDE_john_bought_36_rolls_l2604_260425


namespace NUMINAMATH_CALUDE_inequality_solution_l2604_260461

theorem inequality_solution (x : ℝ) : 
  x / ((x + 3) * (x + 1)) > 0 ↔ x < -3 ∨ x > -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2604_260461


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l2604_260480

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l2604_260480


namespace NUMINAMATH_CALUDE_linear_regression_equation_l2604_260421

/-- Linear regression equation for given points -/
theorem linear_regression_equation (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁ = 3 ∧ y₁ = 10)
  (h₂ : x₂ = 7 ∧ y₂ = 20)
  (h₃ : x₃ = 11 ∧ y₃ = 24) :
  ∃ (a b : ℝ), a = 5.75 ∧ b = 1.75 ∧ 
    (∀ (x y : ℝ), (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃) → y = a + b * x) :=
sorry

end NUMINAMATH_CALUDE_linear_regression_equation_l2604_260421


namespace NUMINAMATH_CALUDE_delta_computation_l2604_260428

-- Define the delta operation
def delta (a b : ℕ) : ℕ := a^2 - b

-- State the theorem
theorem delta_computation :
  delta (5^(delta 6 17)) (2^(delta 7 11)) = 5^38 - 2^38 := by
  sorry

end NUMINAMATH_CALUDE_delta_computation_l2604_260428


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2604_260496

theorem fraction_to_decimal : (7 : ℚ) / 50 = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2604_260496


namespace NUMINAMATH_CALUDE_cylinder_sphere_area_ratio_l2604_260402

/-- A cylinder with a square cross-section and height equal to the diameter of a sphere -/
structure SquareCylinder where
  radius : ℝ
  height : ℝ
  isSquare : height = 2 * radius

/-- The sphere with diameter equal to the cylinder's height -/
structure MatchingSphere where
  radius : ℝ

/-- The ratio of the total surface area of the cylinder to the surface area of the sphere is 3:2 -/
theorem cylinder_sphere_area_ratio (c : SquareCylinder) (s : MatchingSphere) 
  (h : c.radius = s.radius) : 
  (2 * c.radius * c.radius + 4 * c.radius * c.height) / (4 * π * s.radius ^ 2) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_sphere_area_ratio_l2604_260402


namespace NUMINAMATH_CALUDE_no_groups_of_six_l2604_260481

theorem no_groups_of_six (x y z : ℕ) : 
  (2*x + 6*y + 10*z) / (x + y + z : ℚ) = 5 →
  (2*x + 30*y + 90*z) / (2*x + 6*y + 10*z : ℚ) = 7 →
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_no_groups_of_six_l2604_260481


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l2604_260460

-- Define the inequality function
def inequality (p q : Real) : Prop :=
  (5 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q

-- State the theorem
theorem inequality_holds_iff_p_in_interval :
  ∀ p : Real, p ≥ 0 →
  (∀ q : Real, q > 0 → inequality p q) ↔
  p ∈ Set.Icc 0 (355/100) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_in_interval_l2604_260460


namespace NUMINAMATH_CALUDE_integral_x_plus_sin_x_l2604_260410

theorem integral_x_plus_sin_x (x : ℝ) : 
  ∫ x in (0)..(π/2), (x + Real.sin x) = π^2/8 + 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_plus_sin_x_l2604_260410


namespace NUMINAMATH_CALUDE_probability_of_e_in_theorem_l2604_260434

theorem probability_of_e_in_theorem :
  let total_letters : ℕ := 7
  let number_of_e : ℕ := 2
  let probability : ℚ := number_of_e / total_letters
  probability = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_of_e_in_theorem_l2604_260434


namespace NUMINAMATH_CALUDE_club_size_after_four_years_l2604_260448

-- Define the club structure
structure Club where
  leaders : ℕ
  regular_members : ℕ

-- Define the initial state and yearly update function
def initial_club : Club := { leaders := 4, regular_members := 16 }

def update_club (c : Club) : Club :=
  { leaders := 4, regular_members := 4 * c.regular_members }

-- Define the club state after n years
def club_after_years (n : ℕ) : Club :=
  match n with
  | 0 => initial_club
  | n+1 => update_club (club_after_years n)

-- Theorem statement
theorem club_size_after_four_years :
  (club_after_years 4).leaders + (club_after_years 4).regular_members = 4100 := by
  sorry


end NUMINAMATH_CALUDE_club_size_after_four_years_l2604_260448


namespace NUMINAMATH_CALUDE_smallest_value_w_z_cubes_l2604_260495

theorem smallest_value_w_z_cubes (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 1)
  (h2 : Complex.abs (w^2 + z^2) = 14) :
  Complex.abs (w^3 + z^3) ≥ 41/2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_w_z_cubes_l2604_260495


namespace NUMINAMATH_CALUDE_max_value_theorem_l2604_260445

theorem max_value_theorem (x y : ℝ) :
  (2 * x + 3 * y + 2) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 17 ∧
  ∃ (x₀ y₀ : ℝ), (2 * x₀ + 3 * y₀ + 2) / Real.sqrt (x₀^2 + y₀^2 + 1) = Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2604_260445


namespace NUMINAMATH_CALUDE_number_of_men_in_first_scenario_l2604_260473

/-- Represents the number of men working in the first scenario -/
def M : ℕ := 15

/-- Represents the number of hours worked per day in the first scenario -/
def hours_per_day_1 : ℕ := 9

/-- Represents the number of days worked in the first scenario -/
def days_1 : ℕ := 16

/-- Represents the number of men working in the second scenario -/
def men_2 : ℕ := 18

/-- Represents the number of hours worked per day in the second scenario -/
def hours_per_day_2 : ℕ := 8

/-- Represents the number of days worked in the second scenario -/
def days_2 : ℕ := 15

/-- Theorem stating that the number of men in the first scenario is 15 -/
theorem number_of_men_in_first_scenario :
  M * hours_per_day_1 * days_1 = men_2 * hours_per_day_2 * days_2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_in_first_scenario_l2604_260473


namespace NUMINAMATH_CALUDE_max_value_of_expressions_l2604_260489

theorem max_value_of_expressions :
  let expr1 := 3 + 1 + 2 + 4
  let expr2 := 3 * 1 + 2 + 4
  let expr3 := 3 + 1 * 2 + 4
  let expr4 := 3 + 1 + 2 * 4
  let expr5 := 3 * 1 * 2 * 4
  max expr1 (max expr2 (max expr3 (max expr4 expr5))) = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expressions_l2604_260489


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l2604_260492

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- Definition of line l₁ -/
def l₁ (m : ℝ) (x y : ℝ) : Prop := 3 * m * x + (m + 2) * y + 1 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + (m + 2) * y + 2 = 0

theorem parallel_lines_m_values :
  ∀ m : ℝ, parallel (3 * m) (m + 2) (m - 2) (m + 2) → m = -1 ∨ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l2604_260492


namespace NUMINAMATH_CALUDE_total_cost_is_1340_l2604_260411

def number_of_vaccines : ℕ := 10
def vaccine_cost : ℚ := 45
def doctors_visit_cost : ℚ := 250
def insurance_coverage_rate : ℚ := 0.8
def trip_cost : ℚ := 1200

def total_cost : ℚ :=
  trip_cost + (1 - insurance_coverage_rate) * (number_of_vaccines * vaccine_cost + doctors_visit_cost)

theorem total_cost_is_1340 : total_cost = 1340 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_1340_l2604_260411


namespace NUMINAMATH_CALUDE_typing_difference_is_856800_l2604_260437

/-- The number of minutes in a week -/
def minutes_per_week : ℕ := 60 * 24 * 7

/-- Micah's typing speed in words per minute -/
def micah_speed : ℕ := 35

/-- Isaiah's typing speed in words per minute -/
def isaiah_speed : ℕ := 120

/-- The difference in words typed between Isaiah and Micah in a week -/
def typing_difference : ℕ := isaiah_speed * minutes_per_week - micah_speed * minutes_per_week

theorem typing_difference_is_856800 : typing_difference = 856800 := by
  sorry

end NUMINAMATH_CALUDE_typing_difference_is_856800_l2604_260437


namespace NUMINAMATH_CALUDE_min_c_value_l2604_260483

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b) (hbc : b < c) (hsum : a + b + c = 1503)
  (hunique : ∃! (x y : ℝ), 2 * x + y = 2008 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 496 ∧ ∃ (a' b' c' : ℕ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' < b' ∧ b' < c' ∧
    a' + b' + c' = 1503 ∧ c' = 496 ∧
    ∃! (x y : ℝ), 2 * x + y = 2008 ∧ y = |x - a'| + |x - b'| + |x - c'| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l2604_260483


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_zero_l2604_260432

theorem subset_implies_a_equals_zero (a : ℝ) : 
  let A : Set ℝ := {1, a - 1}
  let B : Set ℝ := {-1, 2*a - 3, 1 - 2*a}
  A ⊆ B → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_zero_l2604_260432


namespace NUMINAMATH_CALUDE_vector_equality_l2604_260430

/-- Given vectors a, b, and c in ℝ², prove that c = a - b -/
theorem vector_equality (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (2, -1)) 
  (hc : c = (-1, 2)) : 
  c = a - b := by sorry

end NUMINAMATH_CALUDE_vector_equality_l2604_260430


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_smallest_satisfying_number_value_l2604_260469

def is_perfect_power (n : ℕ) (k : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ k

def satisfies_conditions (N : ℕ) : Prop :=
  is_perfect_power (N / 2) 2 ∧
  is_perfect_power (N / 3) 3 ∧
  is_perfect_power (N / 5) 5

theorem smallest_satisfying_number :
  ∃ N : ℕ, satisfies_conditions N ∧
    ∀ M : ℕ, satisfies_conditions M → N ≤ M :=
by
  -- The proof goes here
  sorry

theorem smallest_satisfying_number_value :
  ∃ N : ℕ, N = 2^15 * 3^10 * 5^6 ∧ satisfies_conditions N ∧
    ∀ M : ℕ, satisfies_conditions M → N ≤ M :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_smallest_satisfying_number_value_l2604_260469


namespace NUMINAMATH_CALUDE_alberts_number_l2604_260446

theorem alberts_number (n : ℕ) : 
  (1 : ℚ) / n + (1 : ℚ) / 2 = (1 : ℚ) / 3 + (2 : ℚ) / (n + 1) ↔ n = 2 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_alberts_number_l2604_260446


namespace NUMINAMATH_CALUDE_x_in_terms_of_z_l2604_260444

/-- Given that x is 30% less than y and y = z + 50, prove that x = 0.70z + 35 -/
theorem x_in_terms_of_z (z : ℝ) :
  let y := z + 50
  let x := y - 0.30 * y
  x = 0.70 * z + 35 := by
  sorry

end NUMINAMATH_CALUDE_x_in_terms_of_z_l2604_260444


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2604_260435

theorem inequality_solution_set (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  {x : ℝ | -b < 1/x ∧ 1/x < a} = {x : ℝ | x < -1/b ∨ x > 1/a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2604_260435


namespace NUMINAMATH_CALUDE_question_arrangement_l2604_260485

/-- Represents the number of ways to arrange 6 questions -/
def arrangement_count : ℕ := 144

/-- The number of multiple-choice questions -/
def total_questions : ℕ := 6

/-- The number of easy questions -/
def easy_questions : ℕ := 2

/-- The number of medium questions -/
def medium_questions : ℕ := 2

/-- The number of difficult questions -/
def difficult_questions : ℕ := 2

theorem question_arrangement :
  (easy_questions = 2) →
  (medium_questions = 2) →
  (difficult_questions = 2) →
  (total_questions = easy_questions + medium_questions + difficult_questions) →
  arrangement_count = 144 := by
  sorry

end NUMINAMATH_CALUDE_question_arrangement_l2604_260485


namespace NUMINAMATH_CALUDE_mixture_composition_l2604_260424

theorem mixture_composition 
  (x_percent_a : Real) 
  (y_percent_a : Real) 
  (mixture_percent_a : Real) 
  (h1 : x_percent_a = 0.3) 
  (h2 : y_percent_a = 0.4) 
  (h3 : mixture_percent_a = 0.32) :
  ∃ (x_proportion : Real),
    x_proportion * x_percent_a + (1 - x_proportion) * y_percent_a = mixture_percent_a ∧ 
    x_proportion = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l2604_260424


namespace NUMINAMATH_CALUDE_chris_candy_distribution_l2604_260468

/-- Given that Chris gave each friend 12 pieces of candy and gave away a total of 420 pieces of candy,
    prove that the number of friends Chris gave candy to is 35. -/
theorem chris_candy_distribution (candy_per_friend : ℕ) (total_candy : ℕ) (num_friends : ℕ) :
  candy_per_friend = 12 →
  total_candy = 420 →
  num_friends * candy_per_friend = total_candy →
  num_friends = 35 := by
sorry

end NUMINAMATH_CALUDE_chris_candy_distribution_l2604_260468


namespace NUMINAMATH_CALUDE_photo_survey_result_l2604_260426

/-- Represents the number of students with each attitude towards photography -/
structure PhotoAttitudes where
  dislike : ℕ
  neutral : ℕ
  like : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (a : PhotoAttitudes) : Prop :=
  a.neutral = a.dislike + 12 ∧
  1 * (a.neutral + a.dislike + a.like) = 9 * a.dislike ∧
  3 * (a.neutral + a.dislike + a.like) = 9 * a.neutral ∧
  5 * (a.neutral + a.dislike + a.like) = 9 * a.like

/-- The theorem to be proved -/
theorem photo_survey_result :
  ∃ (a : PhotoAttitudes), satisfiesConditions a ∧ a.like = 30 := by
  sorry

end NUMINAMATH_CALUDE_photo_survey_result_l2604_260426


namespace NUMINAMATH_CALUDE_binary_11011011_equals_base4_3123_l2604_260476

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

theorem binary_11011011_equals_base4_3123 :
  decimal_to_base4 (binary_to_decimal [true, true, false, true, true, false, true, true]) = [3, 1, 2, 3] := by
  sorry

#eval binary_to_decimal [true, true, false, true, true, false, true, true]
#eval decimal_to_base4 (binary_to_decimal [true, true, false, true, true, false, true, true])

end NUMINAMATH_CALUDE_binary_11011011_equals_base4_3123_l2604_260476


namespace NUMINAMATH_CALUDE_andrews_cookies_l2604_260478

/-- Proves that Andrew purchased 3 cookies each day in May --/
theorem andrews_cookies (total_spent : ℕ) (cookie_cost : ℕ) (days_in_may : ℕ) 
  (h1 : total_spent = 1395)
  (h2 : cookie_cost = 15)
  (h3 : days_in_may = 31) :
  total_spent / (cookie_cost * days_in_may) = 3 :=
by
  sorry

#check andrews_cookies

end NUMINAMATH_CALUDE_andrews_cookies_l2604_260478


namespace NUMINAMATH_CALUDE_first_number_value_l2604_260438

theorem first_number_value (x y : ℝ) 
  (sum_condition : x + y = 33)
  (double_condition : y = 2 * x)
  (second_number_value : y = 22) :
  x = 11 := by
  sorry

end NUMINAMATH_CALUDE_first_number_value_l2604_260438


namespace NUMINAMATH_CALUDE_total_cans_donated_l2604_260456

/-- The number of homeless shelters -/
def num_shelters : ℕ := 6

/-- The number of people served by each shelter -/
def people_per_shelter : ℕ := 30

/-- The number of cans of soup bought per person -/
def cans_per_person : ℕ := 10

/-- Theorem: The total number of cans of soup Mark donates is 1800 -/
theorem total_cans_donated : 
  num_shelters * people_per_shelter * cans_per_person = 1800 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_donated_l2604_260456


namespace NUMINAMATH_CALUDE_shaded_area_circles_l2604_260416

theorem shaded_area_circles (R : ℝ) (h : R = 10) : 
  let r : ℝ := R / 3
  let larger_area : ℝ := π * R^2
  let smaller_area : ℝ := 2 * π * r^2
  let shaded_area : ℝ := larger_area - smaller_area
  shaded_area = (700 / 9) * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_circles_l2604_260416


namespace NUMINAMATH_CALUDE_product_even_if_sum_odd_l2604_260422

theorem product_even_if_sum_odd (a b : ℤ) : 
  Odd (a + b) → Even (a * b) := by
  sorry

end NUMINAMATH_CALUDE_product_even_if_sum_odd_l2604_260422


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2604_260449

theorem algebraic_expression_value (a b : ℝ) (h1 : a * b = 2) (h2 : a + b = 3) :
  2 * a^3 * b - 4 * a^2 * b^2 + 2 * a * b^3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2604_260449


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l2604_260466

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) : 
  x^2 + y^2 = 1057 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l2604_260466


namespace NUMINAMATH_CALUDE_wooden_box_width_l2604_260493

def wooden_box_length : Real := 8
def wooden_box_height : Real := 6
def small_box_length : Real := 0.04
def small_box_width : Real := 0.07
def small_box_height : Real := 0.06
def max_small_boxes : Nat := 2000000

theorem wooden_box_width :
  ∃ (W : Real),
    W * wooden_box_length * wooden_box_height =
      (small_box_length * small_box_width * small_box_height) * max_small_boxes ∧
    W = 7 := by
  sorry

end NUMINAMATH_CALUDE_wooden_box_width_l2604_260493


namespace NUMINAMATH_CALUDE_parabola_transformation_l2604_260451

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the transformation
def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 3) - 2

-- State the theorem
theorem parabola_transformation :
  ∀ x : ℝ, transform original_parabola x = (x + 1)^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l2604_260451


namespace NUMINAMATH_CALUDE_library_book_distribution_l2604_260403

def number_of_distributions (total_books : ℕ) (min_in_library : ℕ) (min_checked_out : ℕ) : ℕ :=
  (total_books - min_in_library - min_checked_out + 1)

theorem library_book_distribution :
  number_of_distributions 8 2 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_library_book_distribution_l2604_260403


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_l2604_260417

/-- Given a polynomial g(x) = 2x^7 - 3x^5 + px^2 + 2x - 6 where g(5) = 10, 
    prove that g(-5) = -301383 -/
theorem polynomial_value_at_negative (p : ℝ) : 
  (fun x : ℝ => 2*x^7 - 3*x^5 + p*x^2 + 2*x - 6) 5 = 10 → 
  (fun x : ℝ => 2*x^7 - 3*x^5 + p*x^2 + 2*x - 6) (-5) = -301383 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_l2604_260417


namespace NUMINAMATH_CALUDE_bowling_ball_surface_area_l2604_260484

theorem bowling_ball_surface_area :
  let diameter : ℝ := 9
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius^2
  surface_area = 81 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_surface_area_l2604_260484


namespace NUMINAMATH_CALUDE_prob_xi_equals_two_l2604_260465

/-- A random variable following a binomial distribution with n = 3 and p = 1/3 -/
def ξ : ℕ → ℝ := sorry

/-- The probability mass function for the binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1-p)^(n-k)

/-- Theorem: The probability that ξ equals 2 is 2/9 -/
theorem prob_xi_equals_two :
  binomial_pmf 3 (1/3) 2 = 2/9 := by sorry

end NUMINAMATH_CALUDE_prob_xi_equals_two_l2604_260465


namespace NUMINAMATH_CALUDE_count_common_divisors_60_108_l2604_260454

/-- The number of positive integers that are divisors of both 60 and 108 -/
def commonDivisorCount : ℕ := 
  (Finset.filter (fun n => n ∣ 60 ∧ n ∣ 108) (Finset.range 109)).card

theorem count_common_divisors_60_108 : commonDivisorCount = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_common_divisors_60_108_l2604_260454


namespace NUMINAMATH_CALUDE_system_solution_l2604_260457

theorem system_solution :
  ∃ (x y : ℚ), 
    (4 * x - 3 * y = -7) ∧ 
    (5 * x + 4 * y = -6) ∧ 
    (x = -46 / 31) ∧ 
    (y = 11 / 31) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2604_260457


namespace NUMINAMATH_CALUDE_track_team_size_l2604_260477

/-- The length of the relay race in meters -/
def relay_length : ℕ := 150

/-- The distance each team member runs in meters -/
def individual_distance : ℕ := 30

/-- The number of people on the track team -/
def team_size : ℕ := relay_length / individual_distance

theorem track_team_size : team_size = 5 := by
  sorry

end NUMINAMATH_CALUDE_track_team_size_l2604_260477


namespace NUMINAMATH_CALUDE_dad_steps_l2604_260412

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between Dad's and Masha's steps -/
def dad_masha_ratio (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha

/-- Defines the relationship between Masha's and Yasha's steps -/
def masha_yasha_ratio (s : Steps) : Prop :=
  5 * s.masha = 3 * s.yasha

/-- States that Masha and Yasha together took 400 steps -/
def total_masha_yasha (s : Steps) : Prop :=
  s.masha + s.yasha = 400

theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s)
  (h2 : masha_yasha_ratio s)
  (h3 : total_masha_yasha s) :
  s.dad = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_l2604_260412


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2604_260443

def U : Set ℝ := {x | x < 3}
def A : Set ℝ := {x | x < 1}

theorem complement_A_in_U : 
  U \ A = {x | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2604_260443


namespace NUMINAMATH_CALUDE_james_off_road_vehicles_l2604_260458

/-- The number of off-road vehicles James bought -/
def num_off_road_vehicles : ℕ := 4

/-- The cost of each dirt bike -/
def dirt_bike_cost : ℕ := 150

/-- The cost of each off-road vehicle -/
def off_road_vehicle_cost : ℕ := 300

/-- The registration cost for each vehicle -/
def registration_cost : ℕ := 25

/-- The number of dirt bikes James bought -/
def num_dirt_bikes : ℕ := 3

/-- The total amount James paid -/
def total_amount : ℕ := 1825

theorem james_off_road_vehicles :
  num_off_road_vehicles * (off_road_vehicle_cost + registration_cost) +
  num_dirt_bikes * (dirt_bike_cost + registration_cost) =
  total_amount :=
by sorry

end NUMINAMATH_CALUDE_james_off_road_vehicles_l2604_260458


namespace NUMINAMATH_CALUDE_another_number_with_remainder_three_l2604_260482

theorem another_number_with_remainder_three (n : ℕ) : 
  n = 1680 → 
  n % 9 = 0 → 
  ∃ m : ℕ, m ≠ n ∧ n % m = 3 → 
  n % 1677 = 3 :=
by sorry

end NUMINAMATH_CALUDE_another_number_with_remainder_three_l2604_260482


namespace NUMINAMATH_CALUDE_power_sum_equals_seventeen_l2604_260419

theorem power_sum_equals_seventeen : (-3 : ℤ)^4 + (-4 : ℤ)^3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_seventeen_l2604_260419


namespace NUMINAMATH_CALUDE_tan_three_expression_equals_zero_l2604_260439

theorem tan_three_expression_equals_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_expression_equals_zero_l2604_260439


namespace NUMINAMATH_CALUDE_seven_reverse_sum_squares_l2604_260472

/-- A function that reverses a two-digit number -/
def reverse_digits (n : Nat) : Nat :=
  (n % 10) * 10 + (n / 10)

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

/-- The main theorem stating that there are exactly 7 two-digit numbers
    (including 38) where the sum of the number and its reverse is a perfect square -/
theorem seven_reverse_sum_squares :
  ∃! (list : List Nat),
    list.length = 7 ∧
    (∀ n ∈ list, 10 ≤ n ∧ n < 100) ∧
    (∀ n ∈ list, is_perfect_square (n + reverse_digits n)) ∧
    (38 ∈ list) ∧
    (∀ m, 10 ≤ m ∧ m < 100 →
      is_perfect_square (m + reverse_digits m) →
      m ∈ list) :=
  sorry

end NUMINAMATH_CALUDE_seven_reverse_sum_squares_l2604_260472


namespace NUMINAMATH_CALUDE_log_equation_solution_l2604_260498

theorem log_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ x - 2 > 0 ∧ x + 2 > 0 ∧
  Real.log x + Real.log (x - 2) = Real.log 3 + Real.log (x + 2) ∧
  x = 6 :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2604_260498


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2604_260408

/-- Represents a tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  pq : ℝ
  pr : ℝ
  ps : ℝ
  qr : ℝ
  qs : ℝ
  rs : ℝ

/-- Calculates the volume of a tetrahedron -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 18 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    pq := 6,
    pr := 4,
    ps := 5,
    qr := 5,
    qs := 7,
    rs := 8
  }
  tetrahedronVolume t = 18 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2604_260408


namespace NUMINAMATH_CALUDE_expansion_terms_count_l2604_260491

/-- The number of terms in the expansion of a product of two sums -/
def num_terms_expansion (n m : ℕ) : ℕ := n * m

/-- Theorem: The expansion of a product of two sums with 4 and 5 terms respectively has 20 terms -/
theorem expansion_terms_count :
  num_terms_expansion 4 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l2604_260491


namespace NUMINAMATH_CALUDE_x_in_terms_of_y_l2604_260436

theorem x_in_terms_of_y (x y : ℚ) (h : 2 * x - 7 * y = 5) : x = (7 * y + 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_in_terms_of_y_l2604_260436


namespace NUMINAMATH_CALUDE_pencil_pen_choices_l2604_260462

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (m n : ℕ) : ℕ := m * n

/-- Theorem: Choosing one item from a set of 4 and one from a set of 6 results in 24 possibilities -/
theorem pencil_pen_choices : choose_one_from_each 4 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_choices_l2604_260462


namespace NUMINAMATH_CALUDE_group_size_is_seven_l2604_260433

/-- The number of boxes one person can lift -/
def boxes_per_person : ℕ := 2

/-- The total number of boxes the group can hold -/
def total_boxes : ℕ := 14

/-- The number of people in the group -/
def group_size : ℕ := total_boxes / boxes_per_person

theorem group_size_is_seven : group_size = 7 := by
  sorry

end NUMINAMATH_CALUDE_group_size_is_seven_l2604_260433


namespace NUMINAMATH_CALUDE_current_speed_is_4_l2604_260499

/-- Represents the speed of a boat in a river with a current -/
structure RiverBoat where
  boatSpeed : ℝ  -- Speed of the boat in still water
  currentSpeed : ℝ  -- Speed of the current

/-- Calculates the effective downstream speed -/
def downstreamSpeed (rb : RiverBoat) : ℝ :=
  rb.boatSpeed + rb.currentSpeed

/-- Calculates the effective upstream speed -/
def upstreamSpeed (rb : RiverBoat) : ℝ :=
  rb.boatSpeed - rb.currentSpeed

/-- Theorem stating the speed of the current given the problem conditions -/
theorem current_speed_is_4 (rb : RiverBoat) 
  (h1 : downstreamSpeed rb * 8 = 96)
  (h2 : upstreamSpeed rb * 2 = 8) :
  rb.currentSpeed = 4 := by
  sorry


end NUMINAMATH_CALUDE_current_speed_is_4_l2604_260499


namespace NUMINAMATH_CALUDE_triangle_area_l2604_260475

theorem triangle_area (a b c : ℝ) (h1 : c^2 = (a - b)^2 + 6) (h2 : c = π/3) :
  (1/2) * a * b * Real.sin (π/3) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2604_260475


namespace NUMINAMATH_CALUDE_product_of_numbers_l2604_260463

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 220) : x * y = 56 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2604_260463


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l2604_260486

/-- Represents the lengths of external diagonals of a right regular prism -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- Checks if given lengths can be external diagonals of a right regular prism -/
def isValidExternalDiagonals (d : ExternalDiagonals) : Prop :=
  d.a^2 + d.b^2 > d.c^2 ∧ d.b^2 + d.c^2 > d.a^2 ∧ d.a^2 + d.c^2 > d.b^2

theorem invalid_external_diagonals :
  ¬ isValidExternalDiagonals ⟨4, 5, 7, by norm_num, by norm_num, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l2604_260486


namespace NUMINAMATH_CALUDE_distance_between_vertices_l2604_260418

-- Define the equation of the parabolas
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 1) = 5

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -2)

-- Theorem stating the distance between vertices
theorem distance_between_vertices :
  let (x1, y1) := vertex1
  let (x2, y2) := vertex2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l2604_260418


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2604_260429

theorem rectangle_dimensions (l w : ℝ) :
  l = 3 * w ∧ l * w = 2 * (l + w) → w = 8 / 3 ∧ l = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2604_260429


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2604_260474

theorem concentric_circles_ratio (R r k : ℝ) (h1 : R > r) (h2 : k > 0) :
  (π * R^2 - π * r^2) = k * (π * r^2) → R / r = Real.sqrt (k + 1) :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l2604_260474


namespace NUMINAMATH_CALUDE_no_increasing_sequence_with_finite_primes_l2604_260407

theorem no_increasing_sequence_with_finite_primes :
  ¬ ∃ (a : ℕ → ℕ),
    (∀ n : ℕ, a n < a (n + 1)) ∧
    (∀ c : ℤ, ∃ N : ℕ, ∀ n ≥ N, ¬ (Prime (c + a n))) :=
by sorry

end NUMINAMATH_CALUDE_no_increasing_sequence_with_finite_primes_l2604_260407


namespace NUMINAMATH_CALUDE_guess_number_in_three_questions_l2604_260423

theorem guess_number_in_three_questions :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 8 →
  ∃ (q₁ q₂ q₃ : ℕ → Prop),
    ∀ m : ℕ, 1 ≤ m ∧ m ≤ 8 →
      (q₁ m = q₁ n ∧ q₂ m = q₂ n ∧ q₃ m = q₃ n) → m = n :=
by sorry

end NUMINAMATH_CALUDE_guess_number_in_three_questions_l2604_260423


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2604_260427

theorem complex_fraction_simplification :
  (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2604_260427


namespace NUMINAMATH_CALUDE_magnitude_of_z_squared_l2604_260404

theorem magnitude_of_z_squared (z : ℂ) : z = 5 + 2*I → Complex.abs (z^2) = 29 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_squared_l2604_260404


namespace NUMINAMATH_CALUDE_inequality_three_intervals_l2604_260413

theorem inequality_three_intervals (a : ℝ) (h : a > 1) :
  ∃ (I₁ I₂ I₃ : Set ℝ), 
    (∀ x : ℝ, (x^2 + (a+1)*x + a) / (x^2 + 5*x + 4) ≥ 0 ↔ x ∈ I₁ ∪ I₂ ∪ I₃) ∧
    (I₁.Nonempty ∧ I₂.Nonempty ∧ I₃.Nonempty) ∧
    (I₁ ∩ I₂ = ∅ ∧ I₁ ∩ I₃ = ∅ ∧ I₂ ∩ I₃ = ∅) :=
by sorry

end NUMINAMATH_CALUDE_inequality_three_intervals_l2604_260413


namespace NUMINAMATH_CALUDE_smallest_sum_of_products_l2604_260464

theorem smallest_sum_of_products (b : Fin 100 → Int) 
  (h : ∀ i, b i = 1 ∨ b i = -1) :
  22 = (Finset.range 100).sum (λ i => 
    (Finset.range 100).sum (λ j => 
      if i < j then b i * b j else 0)) ∧
  ∀ (c : Fin 100 → Int) (hc : ∀ i, c i = 1 ∨ c i = -1),
    0 < (Finset.range 100).sum (λ i => 
      (Finset.range 100).sum (λ j => 
        if i < j then c i * c j else 0)) →
    22 ≤ (Finset.range 100).sum (λ i => 
      (Finset.range 100).sum (λ j => 
        if i < j then c i * c j else 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_products_l2604_260464
