import Mathlib

namespace NUMINAMATH_CALUDE_cube_mean_inequality_l3512_351286

theorem cube_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 + b^3) / 2 ≥ ((a + b) / 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_mean_inequality_l3512_351286


namespace NUMINAMATH_CALUDE_integer_pair_divisibility_l3512_351229

theorem integer_pair_divisibility (a b : ℤ) (ha : a > 1) (hb : b > 1)
  (hab : a ∣ (b + 1)) (hba : b ∣ (a^3 - 1)) :
  (∃ s : ℤ, s ≥ 2 ∧ a = s ∧ b = s^3 - 1) ∨
  (∃ s : ℤ, s ≥ 3 ∧ a = s ∧ b = s - 1) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_divisibility_l3512_351229


namespace NUMINAMATH_CALUDE_fraction_problem_l3512_351271

theorem fraction_problem (x : ℚ) : x * 8 + 2 = 8 ↔ x = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l3512_351271


namespace NUMINAMATH_CALUDE_gardeners_mowing_time_l3512_351208

theorem gardeners_mowing_time (rate_A rate_B : ℚ) (h1 : rate_A = 1 / 3) (h2 : rate_B = 1 / 5) :
  1 / (rate_A + rate_B) = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_gardeners_mowing_time_l3512_351208


namespace NUMINAMATH_CALUDE_fuel_for_three_trips_l3512_351243

/-- Calculates the total fuel needed for a series of trips given a fuel consumption rate -/
def totalFuelNeeded (fuelRate : ℝ) (trips : List ℝ) : ℝ :=
  fuelRate * (trips.sum)

/-- Proves that the total fuel needed for three specific trips is 550 liters -/
theorem fuel_for_three_trips :
  let fuelRate : ℝ := 5
  let trips : List ℝ := [50, 35, 25]
  totalFuelNeeded fuelRate trips = 550 := by
  sorry

#check fuel_for_three_trips

end NUMINAMATH_CALUDE_fuel_for_three_trips_l3512_351243


namespace NUMINAMATH_CALUDE_three_not_in_range_of_g_l3512_351251

/-- The function g(x) defined as x^2 - bx + c -/
def g (b c x : ℝ) : ℝ := x^2 - b*x + c

/-- Theorem stating the conditions for 3 to not be in the range of g(x) -/
theorem three_not_in_range_of_g (b c : ℝ) :
  (∀ x, g b c x ≠ 3) ↔ (c ≥ 3 ∧ b > -Real.sqrt (4*c - 12) ∧ b < Real.sqrt (4*c - 12)) :=
sorry

end NUMINAMATH_CALUDE_three_not_in_range_of_g_l3512_351251


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l3512_351209

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of the shorter leg -/
  a : ℝ
  /-- The length of the longer leg -/
  b : ℝ
  /-- The length of the hypotenuse -/
  c : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- Ensure a is the shorter leg -/
  h_a_le_b : a ≤ b
  /-- Pythagorean theorem -/
  h_pythagorean : a^2 + b^2 = c^2
  /-- Formula for the radius of the inscribed circle -/
  h_radius : r = (a + b - c) / 2
  /-- Positivity conditions -/
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_c_pos : c > 0
  h_r_pos : r > 0

/-- The main theorem: the radius of the inscribed circle is less than one-third of the longer leg -/
theorem inscribed_circle_radius_bound (t : RightTriangleWithInscribedCircle) : t.r < t.b / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_bound_l3512_351209


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3512_351210

theorem fourth_root_equation_solutions :
  {x : ℝ | (57 - 2*x)^(1/4) + (45 + 2*x)^(1/4) = 4} = {27, -17} := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3512_351210


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l3512_351285

/-- Given that the total marks in physics, chemistry, and mathematics is 130 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 65. -/
theorem average_marks_chemistry_mathematics (P C M : ℕ) : 
  P + C + M = P + 130 → (C + M) / 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l3512_351285


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3512_351203

def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 - 48 * x - y^2 + 6 * y + 50 = 0

def vertex_distance (eq : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem hyperbola_vertex_distance :
  vertex_distance hyperbola_equation = 2 * Real.sqrt 85 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3512_351203


namespace NUMINAMATH_CALUDE_sin_2x_value_l3512_351287

theorem sin_2x_value (x : Real) (h : Real.sin (π / 4 - x) = 3 / 5) : 
  Real.sin (2 * x) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_value_l3512_351287


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l3512_351256

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2 * x - 1|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
theorem range_of_a :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, f a x ≤ |2 * x + 1|) → a ∈ Set.Icc (-1 : ℝ) (5/2) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l3512_351256


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3512_351227

theorem rectangle_dimensions (w : ℝ) (h : w > 0) :
  let l := 2 * w
  let area := w * l
  let perimeter := 2 * (w + l)
  area = 2 * perimeter → w = 6 ∧ l = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3512_351227


namespace NUMINAMATH_CALUDE_promotion_price_correct_l3512_351259

/-- The price of a medium pizza in the promotion -/
def promotion_price : ℚ := 5

/-- The regular price of a medium pizza -/
def regular_price : ℚ := 18

/-- The number of medium pizzas in the promotion -/
def promotion_quantity : ℕ := 3

/-- The total savings from the promotion -/
def total_savings : ℚ := 39

/-- Theorem stating that the promotion price satisfies the given conditions -/
theorem promotion_price_correct : 
  promotion_quantity * (regular_price - promotion_price) = total_savings :=
by sorry

end NUMINAMATH_CALUDE_promotion_price_correct_l3512_351259


namespace NUMINAMATH_CALUDE_set_B_equivalence_l3512_351260

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | f a b x - x = 0}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | f a b x - a*x = 0}

-- State the theorem
theorem set_B_equivalence (a b : ℝ) : 
  A a b = {1, -3} → B a b = {-2 - Real.sqrt 7, -2 + Real.sqrt 7} := by
  sorry

end NUMINAMATH_CALUDE_set_B_equivalence_l3512_351260


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l3512_351235

/-- The number of ways to arrange 4 passengers in 10 seats with exactly 5 consecutive empty seats -/
def arrangement_count : ℕ := 480

/-- The number of seats in the bus station -/
def total_seats : ℕ := 10

/-- The number of passengers -/
def num_passengers : ℕ := 4

/-- The number of consecutive empty seats required -/
def consecutive_empty_seats : ℕ := 5

/-- Theorem stating that the arrangement count is correct -/
theorem arrangement_count_correct : 
  arrangement_count = 
    (Nat.factorial num_passengers) * 
    (Nat.factorial 5 / (Nat.factorial 3)) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_correct_l3512_351235


namespace NUMINAMATH_CALUDE_sugar_calculation_l3512_351202

theorem sugar_calculation (num_packs : ℕ) (pack_weight : ℕ) (leftover : ℕ) :
  num_packs = 30 →
  pack_weight = 350 →
  leftover = 50 →
  num_packs * pack_weight + leftover = 10550 := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l3512_351202


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3512_351230

theorem simplify_sqrt_expression : Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3512_351230


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3512_351266

-- Define the hyperbola
structure Hyperbola where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  passes_through : ℝ × ℝ  -- point that the hyperbola passes through

-- Define the standard equation of a hyperbola
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) :
  h.a = 2 * Real.sqrt 5 →
  h.passes_through = (5, -2) →
  ∀ x y : ℝ, standard_equation h x y ↔ x^2 / 20 - y^2 / 16 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3512_351266


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l3512_351224

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; 2, 6]
  Matrix.det A = 24 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l3512_351224


namespace NUMINAMATH_CALUDE_smallest_class_size_l3512_351279

theorem smallest_class_size (b g : ℕ) : 
  b > 0 → g > 0 → 
  (3 * b) % 5 = 0 → 
  (2 * g) % 3 = 0 → 
  3 * b / 5 = 2 * (2 * g / 3) → 
  29 ≤ b + g ∧ 
  (∀ b' g' : ℕ, b' > 0 → g' > 0 → 
    (3 * b') % 5 = 0 → 
    (2 * g') % 3 = 0 → 
    3 * b' / 5 = 2 * (2 * g' / 3) → 
    b' + g' ≥ 29) :=
by sorry

#check smallest_class_size

end NUMINAMATH_CALUDE_smallest_class_size_l3512_351279


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3512_351213

theorem inequality_equivalence (x : ℝ) : (x - 1) / 3 > 2 ↔ x > 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3512_351213


namespace NUMINAMATH_CALUDE_distance_between_trees_l3512_351250

theorem distance_between_trees (total_length : ℝ) (num_trees : ℕ) :
  total_length = 600 →
  num_trees = 26 →
  (total_length / (num_trees - 1 : ℝ)) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3512_351250


namespace NUMINAMATH_CALUDE_students_without_A_l3512_351241

theorem students_without_A (total : ℕ) (lit_A : ℕ) (sci_A : ℕ) (both_A : ℕ) : 
  total - (lit_A + sci_A - both_A) = total - (lit_A + sci_A - both_A) :=
by sorry

#check students_without_A 40 10 18 6

end NUMINAMATH_CALUDE_students_without_A_l3512_351241


namespace NUMINAMATH_CALUDE_oil_drop_probability_l3512_351296

theorem oil_drop_probability (r : Real) (s : Real) (h1 : r = 1) (h2 : s = 0.5) : 
  (s^2) / (π * r^2) = 1 / (4 * π) :=
sorry

end NUMINAMATH_CALUDE_oil_drop_probability_l3512_351296


namespace NUMINAMATH_CALUDE_f_of_5_eq_92_l3512_351207

/-- Given a function f(x) = 2x^2 + y where f(2) = 50, prove that f(5) = 92 -/
theorem f_of_5_eq_92 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 50) :
  f 5 = 92 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_eq_92_l3512_351207


namespace NUMINAMATH_CALUDE_min_equation_implies_sum_l3512_351265

theorem min_equation_implies_sum (a b c d : ℝ) :
  (∀ x : ℝ, min (20 * x + 19) (19 * x + 20) = (a * x + b) - |c * x + d|) →
  a * b + c * d = 380 := by
  sorry

end NUMINAMATH_CALUDE_min_equation_implies_sum_l3512_351265


namespace NUMINAMATH_CALUDE_complement_of_M_relative_to_U_l3512_351292

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 4, 6}

theorem complement_of_M_relative_to_U :
  U \ M = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_relative_to_U_l3512_351292


namespace NUMINAMATH_CALUDE_agency_A_more_cost_effective_l3512_351270

/-- Represents the cost calculation for travel agencies A and B -/
def travel_cost (num_students : ℕ) : ℚ × ℚ :=
  let full_price : ℚ := 40
  let num_parents : ℕ := 10
  let cost_A : ℚ := full_price * num_parents.cast + (full_price / 2) * num_students.cast
  let cost_B : ℚ := full_price * (1 - 0.4) * (num_parents + num_students).cast
  (cost_A, cost_B)

/-- Theorem stating when travel agency A is more cost-effective -/
theorem agency_A_more_cost_effective (num_students : ℕ) :
  num_students > 40 → (travel_cost num_students).1 < (travel_cost num_students).2 := by
  sorry

#check agency_A_more_cost_effective

end NUMINAMATH_CALUDE_agency_A_more_cost_effective_l3512_351270


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l3512_351238

/-- Given two monomials -xy^(b+1) and (1/2)x^(a+2)y^3, if their sum is still a monomial, then a + b = 1 -/
theorem monomial_sum_condition (a b : ℤ) : 
  (∃ (k : ℚ), k * x * y^(b + 1) + (1/2) * x^(a + 2) * y^3 = c * x^m * y^n) → 
  a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l3512_351238


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3512_351214

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 4}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3512_351214


namespace NUMINAMATH_CALUDE_not_three_k_minus_one_l3512_351263

theorem not_three_k_minus_one (n : ℕ) : 
  (n * (n - 1) / 2) % 3 ≠ 2 ∧ (n^2) % 3 ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_not_three_k_minus_one_l3512_351263


namespace NUMINAMATH_CALUDE_smallest_fraction_l3512_351231

theorem smallest_fraction (S : Set ℚ) (h : S = {1/2, 2/3, 1/4, 5/6, 7/12}) :
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = 1/4 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l3512_351231


namespace NUMINAMATH_CALUDE_sangita_cross_country_hours_l3512_351272

/-- Calculates the cross-country flying hours completed by Sangita --/
def cross_country_hours (total_required : ℕ) (day_flying : ℕ) (night_flying : ℕ) 
  (hours_per_month : ℕ) (duration_months : ℕ) : ℕ :=
  total_required - (day_flying + night_flying)

/-- Theorem stating that Sangita's cross-country flying hours equal 1261 --/
theorem sangita_cross_country_hours : 
  cross_country_hours 1500 50 9 220 6 = 1261 := by
  sorry

#eval cross_country_hours 1500 50 9 220 6

end NUMINAMATH_CALUDE_sangita_cross_country_hours_l3512_351272


namespace NUMINAMATH_CALUDE_oranges_in_bowl_l3512_351204

def bowl_of_fruit (num_bananas : ℕ) (num_apples : ℕ) (num_oranges : ℕ) : Prop :=
  num_apples = 2 * num_bananas ∧
  num_bananas + num_apples + num_oranges = 12

theorem oranges_in_bowl :
  ∃ (num_oranges : ℕ), bowl_of_fruit 2 (2 * 2) num_oranges ∧ num_oranges = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_oranges_in_bowl_l3512_351204


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l3512_351268

-- Define the circles
def circle1 : ℝ × ℝ := (0, 0)
def circle2 : ℝ × ℝ := (20, 0)
def radius1 : ℝ := 3
def radius2 : ℝ := 9

-- Define the tangent line intersection point
def intersection_point : ℝ := 5

-- Theorem statement
theorem tangent_line_intersection :
  let d := circle2.1 - circle1.1  -- Distance between circle centers
  ∃ (t : ℝ), 
    t > 0 ∧ 
    intersection_point = circle1.1 + t * radius1 ∧
    intersection_point = circle2.1 - t * radius2 ∧
    t * (radius1 + radius2) = d :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l3512_351268


namespace NUMINAMATH_CALUDE_final_time_sum_l3512_351206

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the time after a given duration -/
def timeAfter (initial : Time) (duration : Time) : Time :=
  sorry

/-- Converts a Time to its representation on a 12-hour clock -/
def to12HourClock (t : Time) : Time :=
  sorry

theorem final_time_sum (initial : Time) (duration : Time) : 
  initial.hours = 15 ∧ initial.minutes = 0 ∧ initial.seconds = 0 →
  duration.hours = 158 ∧ duration.minutes = 55 ∧ duration.seconds = 32 →
  let finalTime := to12HourClock (timeAfter initial duration)
  finalTime.hours + finalTime.minutes + finalTime.seconds = 92 :=
sorry

end NUMINAMATH_CALUDE_final_time_sum_l3512_351206


namespace NUMINAMATH_CALUDE_intersection_A_B_l3512_351211

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3512_351211


namespace NUMINAMATH_CALUDE_second_number_is_90_l3512_351289

theorem second_number_is_90 (a b c : ℚ) : 
  a + b + c = 330 → 
  a = 2 * b → 
  c = (1/3) * a → 
  b = 90 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_90_l3512_351289


namespace NUMINAMATH_CALUDE_solution_range_l3512_351258

theorem solution_range (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 1/x + 4/y = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 4/y = 1 ∧ x + y/4 < m^2 - 3*m) ↔ m < -1 ∨ m > 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l3512_351258


namespace NUMINAMATH_CALUDE_christina_account_balance_l3512_351233

def initial_balance : ℕ := 27004
def transferred_amount : ℕ := 69
def remaining_balance : ℕ := 26935

theorem christina_account_balance :
  initial_balance - transferred_amount = remaining_balance :=
by sorry

end NUMINAMATH_CALUDE_christina_account_balance_l3512_351233


namespace NUMINAMATH_CALUDE_second_derivative_of_f_l3512_351261

/-- Given a function f(x) = α² - cos x, prove that its second derivative at α is sin α -/
theorem second_derivative_of_f (α : ℝ) : 
  let f : ℝ → ℝ := λ x => α^2 - Real.cos x
  (deriv (deriv f)) α = Real.sin α := by sorry

end NUMINAMATH_CALUDE_second_derivative_of_f_l3512_351261


namespace NUMINAMATH_CALUDE_domain_intersection_l3512_351242

-- Define the domain of y = √(4-x²)
def domain_sqrt (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- Define the domain of y = ln(1-x)
def domain_ln (x : ℝ) : Prop := x < 1

-- Theorem statement
theorem domain_intersection :
  {x : ℝ | domain_sqrt x ∧ domain_ln x} = {x : ℝ | -2 ≤ x ∧ x < 1} :=
sorry

end NUMINAMATH_CALUDE_domain_intersection_l3512_351242


namespace NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l3512_351232

theorem cos_theta_plus_pi_fourth (θ : ℝ) (h : Real.sin (θ - π/4) = 1/5) :
  Real.cos (θ + π/4) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l3512_351232


namespace NUMINAMATH_CALUDE_not_perfect_squares_l3512_351249

theorem not_perfect_squares : 
  (∃ x : ℕ, 7^2040 = x^2) ∧
  (¬∃ x : ℕ, 8^2041 = x^2) ∧
  (∃ x : ℕ, 9^2042 = x^2) ∧
  (¬∃ x : ℕ, 10^2043 = x^2) ∧
  (∃ x : ℕ, 11^2044 = x^2) := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_squares_l3512_351249


namespace NUMINAMATH_CALUDE_magnitude_a_minus_2b_equals_sqrt_21_l3512_351217

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define vectors a and b
variable (a b : V)

-- State the theorem
theorem magnitude_a_minus_2b_equals_sqrt_21 
  (h1 : ‖b‖ = 2 * ‖a‖) 
  (h2 : ‖b‖ = 2) 
  (h3 : inner a b = ‖a‖ * ‖b‖ * (-1/2)) : 
  ‖a - 2 • b‖ = Real.sqrt 21 := by
sorry


end NUMINAMATH_CALUDE_magnitude_a_minus_2b_equals_sqrt_21_l3512_351217


namespace NUMINAMATH_CALUDE_expression_simplification_l3512_351234

theorem expression_simplification : 
  let f (x : ℤ) := x^4 + 324
  ((f 12) * (f 26) * (f 38) * (f 50) * (f 62)) / 
  ((f 6) * (f 18) * (f 30) * (f 42) * (f 54)) = 3968 / 54 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3512_351234


namespace NUMINAMATH_CALUDE_pm25_scientific_notation_correct_l3512_351237

/-- PM2.5 diameter in meters -/
def pm25_diameter : ℝ := 0.0000025

/-- Scientific notation representation of PM2.5 diameter -/
def pm25_scientific : ℝ × ℤ := (2.5, -6)

/-- Theorem stating that the PM2.5 diameter is correctly expressed in scientific notation -/
theorem pm25_scientific_notation_correct :
  pm25_diameter = pm25_scientific.1 * (10 : ℝ) ^ pm25_scientific.2 :=
by sorry

end NUMINAMATH_CALUDE_pm25_scientific_notation_correct_l3512_351237


namespace NUMINAMATH_CALUDE_triangle_theorem_l3512_351267

noncomputable def triangle_proof (A B C : ℝ) (a b c : ℝ) : Prop :=
  let perimeter := a + b + c
  let area := (1/2) * b * c * Real.sin A
  perimeter = 4 * (Real.sqrt 2 + 1) ∧
  Real.sin B + Real.sin C = Real.sqrt 2 * Real.sin A ∧
  area = 3 * Real.sin A ∧
  a = 4 ∧
  A = Real.arccos (1/3)

theorem triangle_theorem :
  ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_proof A B C a b c :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3512_351267


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3512_351274

theorem complex_number_quadrant (z : ℂ) (h : z + z * Complex.I = 2 + 3 * Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3512_351274


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3512_351220

theorem complex_equation_solution (z : ℂ) :
  z / (z - Complex.I) = Complex.I → z = (1 : ℂ) / 2 + Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3512_351220


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_l3512_351281

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Proposition: 108 is the least positive integer with exactly 12 positive factors -/
theorem least_integer_with_12_factors :
  (∀ m : ℕ+, m < 108 → num_factors m ≠ 12) ∧ num_factors 108 = 12 := by sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_l3512_351281


namespace NUMINAMATH_CALUDE_quadratic_real_root_l3512_351252

theorem quadratic_real_root (a : ℝ) : 
  (∃ x : ℝ, (a * (1 + Complex.I)) * x^2 + (1 + a^2 * Complex.I) * x + (a^2 + Complex.I) = 0) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l3512_351252


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_m_is_one_l3512_351299

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Given that z = (m^2 - m) + mi is purely imaginary and m is real, prove that m = 1. -/
theorem purely_imaginary_implies_m_is_one (m : ℝ) :
  isPurelyImaginary ((m^2 - m : ℝ) + m * I) → m = 1 := by
  sorry


end NUMINAMATH_CALUDE_purely_imaginary_implies_m_is_one_l3512_351299


namespace NUMINAMATH_CALUDE_f_minimum_value_a_range_zeros_inequality_l3512_351275

noncomputable section

def f (x : ℝ) := x * Real.log (x + 1)

def g (a x : ℝ) := a * (x + 1 / (x + 1) - 1)

theorem f_minimum_value :
  ∃ (x_min : ℝ), f x_min = 0 ∧ ∀ x, f x ≥ f x_min :=
sorry

theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f x ≤ g a x) ↔ a ≥ 1 :=
sorry

theorem zeros_inequality (b : ℝ) (x₁ x₂ : ℝ) :
  f x₁ = b → f x₂ = b → 2 * |x₁ - x₂| > Real.sqrt (b^2 + 4*b) + 2 * Real.sqrt b - b :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_a_range_zeros_inequality_l3512_351275


namespace NUMINAMATH_CALUDE_ellipse_equation_triangle_area_line_equation_l3512_351295

/-- An ellipse passing through (-1, -1) with semi-focal distance c = √2b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : 1 / a^2 + 1 / b^2 = 1
  h4 : 2 * b^2 = (a^2 - b^2)

/-- Two points on the ellipse intersected by perpendicular lines through (-1, -1) -/
structure IntersectionPoints (e : Ellipse) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  h1 : M.1^2 / e.a^2 + M.2^2 / e.b^2 = 1
  h2 : N.1^2 / e.a^2 + N.2^2 / e.b^2 = 1
  h3 : (M.1 + 1) * (N.1 + 1) + (M.2 + 1) * (N.2 + 1) = 0

theorem ellipse_equation (e : Ellipse) : e.a^2 = 4 ∧ e.b^2 = 4/3 :=
sorry

theorem triangle_area (e : Ellipse) (p : IntersectionPoints e) 
  (h : p.M.2 = 0 ∧ p.N.1 = 1 ∧ p.N.2 = 1) : 
  abs ((p.M.1 + 1) * (p.N.2 + 1) - (p.N.1 + 1) * (p.M.2 + 1)) / 2 = 2 :=
sorry

theorem line_equation (e : Ellipse) (p : IntersectionPoints e) 
  (h : p.M.2 + p.N.2 = 0) :
  (p.M.2 = -p.M.1 ∧ p.N.2 = -p.N.1) ∨ 
  (p.M.1 + p.M.2 = 0 ∧ p.N.1 + p.N.2 = 0) ∨ 
  (p.M.1 = -1/2 ∧ p.N.1 = -1/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_triangle_area_line_equation_l3512_351295


namespace NUMINAMATH_CALUDE_half_circle_roll_center_path_length_l3512_351245

/-- The length of the path traveled by the center of a half-circle when rolled along a straight line -/
def half_circle_center_path_length (radius : ℝ) : ℝ := 2 * radius

/-- Theorem: The length of the path traveled by the center of a half-circle with radius 1 cm, 
    when rolled along a straight line until it completes a half rotation, is equal to 2 cm -/
theorem half_circle_roll_center_path_length :
  half_circle_center_path_length 1 = 2 := by sorry

end NUMINAMATH_CALUDE_half_circle_roll_center_path_length_l3512_351245


namespace NUMINAMATH_CALUDE_article_cost_l3512_351282

theorem article_cost (cost : ℝ) (selling_price : ℝ) : 
  selling_price = 1.25 * cost →
  (0.8 * cost + 0.3 * (0.8 * cost) = selling_price - 8.4) →
  cost = 40 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l3512_351282


namespace NUMINAMATH_CALUDE_extremum_condition_l3512_351219

/-- The function f(x) = ax³ + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- A function has an extremum if it has either a local maximum or a local minimum -/
def has_extremum (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, (∀ x : ℝ, f x ≤ f x₀) ∨ (∀ x : ℝ, f x ≥ f x₀)

/-- The necessary and sufficient condition for f(x) = ax³ + x + 1 to have an extremum -/
theorem extremum_condition (a : ℝ) :
  has_extremum (f a) ↔ a < 0 := by sorry

end NUMINAMATH_CALUDE_extremum_condition_l3512_351219


namespace NUMINAMATH_CALUDE_unique_solution_exists_l3512_351247

-- Define the system of equations
def system (a b c x y z : ℝ) : Prop :=
  Real.sqrt (y - a) + Real.sqrt (z - a) = 1 ∧
  Real.sqrt (z - b) + Real.sqrt (x - b) = 1 ∧
  Real.sqrt (x - c) + Real.sqrt (y - c) = 1

-- Theorem statement
theorem unique_solution_exists (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = Real.sqrt 3 / 2) :
  ∃! x y z : ℝ, system a b c x y z :=
sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l3512_351247


namespace NUMINAMATH_CALUDE_nora_paid_90_dimes_l3512_351205

/-- The number of dimes in one dollar -/
def dimes_per_dollar : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 9

/-- The number of dimes Nora paid for the watch -/
def dimes_paid : ℕ := watch_cost * dimes_per_dollar

theorem nora_paid_90_dimes : dimes_paid = 90 := by
  sorry

end NUMINAMATH_CALUDE_nora_paid_90_dimes_l3512_351205


namespace NUMINAMATH_CALUDE_cottage_village_price_l3512_351200

/-- The selling price of each house in a cottage village -/
def house_selling_price : ℕ := by sorry

/-- The number of houses in the village -/
def num_houses : ℕ := 15

/-- The total cost of construction for the entire village -/
def total_cost : ℕ := 150 + 105 + 225 + 45

/-- The markup percentage of the construction company -/
def markup_percentage : ℚ := 20 / 100

theorem cottage_village_price :
  (house_selling_price : ℚ) = (total_cost : ℚ) / num_houses * (1 + markup_percentage) ∧
  house_selling_price = 42 := by sorry

end NUMINAMATH_CALUDE_cottage_village_price_l3512_351200


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_two_digit_parts_l3512_351283

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def AB (n : ℕ) : ℕ := n / 10

def BC (n : ℕ) : ℕ := n % 100

theorem largest_three_digit_divisible_by_two_digit_parts :
  ∀ n : ℕ,
    is_three_digit n →
    is_two_digit (AB n) →
    is_two_digit (BC n) →
    n % (AB n) = 0 →
    n % (BC n) = 0 →
    n ≤ 990 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_two_digit_parts_l3512_351283


namespace NUMINAMATH_CALUDE_always_integer_l3512_351221

theorem always_integer (m : ℕ) : ∃ k : ℤ, (m : ℚ) / 3 + (m : ℚ)^2 / 2 + (m : ℚ)^3 / 6 = k := by
  sorry

end NUMINAMATH_CALUDE_always_integer_l3512_351221


namespace NUMINAMATH_CALUDE_nancy_total_games_l3512_351255

/-- The total number of football games Nancy would attend over three months -/
def total_games (this_month next_month last_month : ℕ) : ℕ :=
  this_month + next_month + last_month

/-- Theorem: Nancy would attend 24 games in total -/
theorem nancy_total_games : 
  total_games 9 7 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_nancy_total_games_l3512_351255


namespace NUMINAMATH_CALUDE_total_fish_is_23_l3512_351236

/-- The total number of fish caught by Brendan and his dad -/
def total_fish (morning_catch : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) : ℕ :=
  (morning_catch - thrown_back + afternoon_catch) + dad_catch

/-- Theorem stating that the total number of fish caught is 23 -/
theorem total_fish_is_23 :
  total_fish 8 3 5 13 = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_is_23_l3512_351236


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3512_351226

theorem algebraic_expression_value (k p : ℝ) :
  (∀ x : ℝ, (6 * x + 2) * (3 - x) = -6 * x^2 + k * x + p) →
  (k - p)^2 = 100 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3512_351226


namespace NUMINAMATH_CALUDE_bella_steps_l3512_351228

/-- The distance between houses in miles -/
def distance : ℝ := 3

/-- The waiting time for Ella in minutes -/
def wait_time : ℝ := 10

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℝ := 4

/-- The length of Bella's step in feet -/
def step_length : ℝ := 3

/-- The number of feet in a mile -/
def feet_per_mile : ℝ := 5280

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 1328

theorem bella_steps :
  ∃ (bella_speed : ℝ),
    bella_speed > 0 ∧
    (wait_time * bella_speed + 
     (distance * feet_per_mile - wait_time * bella_speed) / (bella_speed * (1 + speed_ratio))) * 
    bella_speed / step_length = steps_taken := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_l3512_351228


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l3512_351248

theorem factorization_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l3512_351248


namespace NUMINAMATH_CALUDE_basketball_probability_l3512_351218

/-- The probability of a basketball player scoring a basket -/
def p : ℚ := 2/3

/-- The number of attempts -/
def n : ℕ := 3

/-- The maximum number of successful baskets we're considering -/
def k : ℕ := 1

/-- The probability of scoring at most once in three attempts -/
def prob_at_most_one : ℚ := 7/27

theorem basketball_probability :
  (Finset.sum (Finset.range (k + 1)) (λ i => Nat.choose n i * p^i * (1 - p)^(n - i))) = prob_at_most_one :=
sorry

end NUMINAMATH_CALUDE_basketball_probability_l3512_351218


namespace NUMINAMATH_CALUDE_bead_arrangement_probability_l3512_351253

/-- Represents the number of beads of each color -/
structure BeadCounts where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- The probability of arranging beads with no adjacent same colors -/
def probability_no_adjacent_same_color (counts : BeadCounts) : Rat :=
  sorry

/-- The main theorem stating the probability for the given bead counts -/
theorem bead_arrangement_probability : 
  probability_no_adjacent_same_color ⟨4, 3, 2, 1⟩ = 1 / 252 := by
  sorry

end NUMINAMATH_CALUDE_bead_arrangement_probability_l3512_351253


namespace NUMINAMATH_CALUDE_exercise_books_count_l3512_351223

/-- Given a shop with pencils, pens, exercise books, and erasers in the ratio 10 : 2 : 3 : 4,
    where there are 150 pencils, prove that there are 45 exercise books. -/
theorem exercise_books_count (pencils : ℕ) (pens : ℕ) (exercise_books : ℕ) (erasers : ℕ) :
  pencils = 150 →
  10 * pens = 2 * pencils →
  10 * exercise_books = 3 * pencils →
  10 * erasers = 4 * pencils →
  exercise_books = 45 := by
  sorry

end NUMINAMATH_CALUDE_exercise_books_count_l3512_351223


namespace NUMINAMATH_CALUDE_lead_is_seventeen_l3512_351298

-- Define the scores of both teams
def chucks_team_score : ℕ := 72
def yellow_team_score : ℕ := 55

-- Define the lead as the difference between the scores
def lead : ℕ := chucks_team_score - yellow_team_score

-- Theorem stating that the lead is 17 points
theorem lead_is_seventeen : lead = 17 := by
  sorry

end NUMINAMATH_CALUDE_lead_is_seventeen_l3512_351298


namespace NUMINAMATH_CALUDE_trapezoid_xy_relation_l3512_351254

-- Define the trapezoid and its properties
structure Trapezoid where
  x : ℝ
  y : ℝ
  h : ℝ
  AC : ℝ
  BD : ℝ
  AB : ℝ
  CD : ℝ
  h_def : h = 5 * x * y
  area_relation : (1/2) * AC * BD = (15*Real.sqrt 3)/(36) * AB * CD
  xy_constraint : x^2 + y^2 = 1

-- State the theorem
theorem trapezoid_xy_relation (t : Trapezoid) : 5 * t.x * t.y = 4 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_xy_relation_l3512_351254


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l3512_351280

theorem solve_fraction_equation :
  ∀ y : ℚ, (3 / 4 : ℚ) - (5 / 8 : ℚ) = 1 / y → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l3512_351280


namespace NUMINAMATH_CALUDE_range_of_f_l3512_351212

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the domain
def domain : Set ℝ := {x | 1 ≤ x ∧ x < 5}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -4 ≤ y ∧ y < 5} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3512_351212


namespace NUMINAMATH_CALUDE_interval_equivalence_l3512_351244

def interval_condition (x : ℝ) : Prop :=
  1 < 3 * x ∧ 3 * x < 2 ∧ 1 < 5 * x ∧ 5 * x < 2

theorem interval_equivalence : 
  {x : ℝ | interval_condition x} = {x : ℝ | 1/3 < x ∧ x < 2/5} :=
by sorry

end NUMINAMATH_CALUDE_interval_equivalence_l3512_351244


namespace NUMINAMATH_CALUDE_cos_2018pi_minus_pi_sixth_l3512_351278

theorem cos_2018pi_minus_pi_sixth : 
  Real.cos (2018 * Real.pi - Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2018pi_minus_pi_sixth_l3512_351278


namespace NUMINAMATH_CALUDE_remainder_sum_l3512_351273

theorem remainder_sum (a b : ℤ) 
  (ha : a % 84 = 77) 
  (hb : b % 120 = 113) : 
  (a + b) % 42 = 22 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l3512_351273


namespace NUMINAMATH_CALUDE_shanna_garden_theorem_l3512_351297

/-- Calculates the number of vegetables per plant given the initial number of plants,
    the number of plants that died, and the total number of vegetables harvested. -/
def vegetables_per_plant (tomato_plants eggplant_plants pepper_plants : ℕ)
                         (dead_tomato_plants dead_pepper_plants : ℕ)
                         (total_vegetables : ℕ) : ℕ :=
  let surviving_plants := (tomato_plants - dead_tomato_plants) +
                          eggplant_plants +
                          (pepper_plants - dead_pepper_plants)
  total_vegetables / surviving_plants

/-- Theorem stating that given Shanna's garden conditions, each remaining plant gave 7 vegetables. -/
theorem shanna_garden_theorem :
  vegetables_per_plant 6 2 4 3 1 56 = 7 :=
by sorry

end NUMINAMATH_CALUDE_shanna_garden_theorem_l3512_351297


namespace NUMINAMATH_CALUDE_complex_magnitude_l3512_351277

theorem complex_magnitude (z : ℂ) (h : 2 + z = (2 - z) * Complex.I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3512_351277


namespace NUMINAMATH_CALUDE_proportion_solution_l3512_351239

theorem proportion_solution (x : ℝ) : (0.75 / x = 10 / 8) → x = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3512_351239


namespace NUMINAMATH_CALUDE_negative_four_is_square_root_of_sixteen_l3512_351276

-- Definition of square root
def is_square_root (x y : ℝ) : Prop := x * x = y

-- Theorem to prove
theorem negative_four_is_square_root_of_sixteen :
  is_square_root (-4) 16 := by
  sorry


end NUMINAMATH_CALUDE_negative_four_is_square_root_of_sixteen_l3512_351276


namespace NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_in_20_years_l3512_351257

/-- 
Given a sum of money that doubles itself in 20 years at simple interest,
this theorem proves that the rate percent per annum is 5%.
-/
theorem simple_interest_rate_for_doubling_in_20_years :
  ∀ (principal : ℝ) (rate : ℝ),
  principal > 0 →
  principal * (1 + rate * 20 / 100) = 2 * principal →
  rate = 5 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_in_20_years_l3512_351257


namespace NUMINAMATH_CALUDE_peters_speed_is_five_l3512_351294

/-- Peter's speed in miles per hour -/
def peter_speed : ℝ := sorry

/-- Juan's speed in miles per hour -/
def juan_speed : ℝ := peter_speed + 3

/-- Time traveled in hours -/
def time : ℝ := 1.5

/-- Total distance between Juan and Peter after traveling -/
def total_distance : ℝ := 19.5

/-- Theorem stating that Peter's speed is 5 miles per hour -/
theorem peters_speed_is_five :
  peter_speed = 5 :=
by
  have h1 : time * peter_speed + time * juan_speed = total_distance := sorry
  sorry

end NUMINAMATH_CALUDE_peters_speed_is_five_l3512_351294


namespace NUMINAMATH_CALUDE_cube_sum_fraction_l3512_351246

theorem cube_sum_fraction (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 8 = 219/8 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_fraction_l3512_351246


namespace NUMINAMATH_CALUDE_sum_of_m_values_l3512_351240

theorem sum_of_m_values (x y z m : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  x / (2 - y) = m ∧ y / (2 - z) = m ∧ z / (2 - x) = m →
  ∃ m₁ m₂ : ℝ, m₁ + m₂ = 2 ∧ (∀ m' : ℝ, m' = m₁ ∨ m' = m₂ ↔ 
    x / (2 - y) = m' ∧ y / (2 - z) = m' ∧ z / (2 - x) = m') :=
by sorry

end NUMINAMATH_CALUDE_sum_of_m_values_l3512_351240


namespace NUMINAMATH_CALUDE_sum_of_squares_l3512_351216

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 10)
  (eq2 : b^2 + 5*c = -10)
  (eq3 : c^2 + 7*a = -21) :
  a^2 + b^2 + c^2 = 20.75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3512_351216


namespace NUMINAMATH_CALUDE_simplify_expression_l3512_351293

theorem simplify_expression :
  (6^8 - 4^7) * (2^3 - (-2)^3)^10 = 1663232 * 16^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3512_351293


namespace NUMINAMATH_CALUDE_fraction_problem_l3512_351290

theorem fraction_problem (x : ℝ) (h : (5 / 9) * x = 60) : (1 / 4) * x = 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3512_351290


namespace NUMINAMATH_CALUDE_product_of_primes_l3512_351264

theorem product_of_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p * q = 69 →
  13 < q →
  q < 25 →
  15 < p * q →
  p * q < 70 →
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_l3512_351264


namespace NUMINAMATH_CALUDE_leak_empty_time_l3512_351201

/-- Given a pipe that can fill a tank in 6 hours, and with a leak it takes 8 hours to fill the tank,
    prove that the leak alone will empty the full tank in 24 hours. -/
theorem leak_empty_time (fill_rate : ℝ) (combined_rate : ℝ) (leak_rate : ℝ) : 
  fill_rate = 1 / 6 →
  combined_rate = 1 / 8 →
  combined_rate = fill_rate - leak_rate →
  1 / leak_rate = 24 := by
sorry

end NUMINAMATH_CALUDE_leak_empty_time_l3512_351201


namespace NUMINAMATH_CALUDE_candle_ratio_problem_l3512_351215

/-- Given the ratio of red candles to blue candles and the number of red candles,
    calculate the number of blue candles. -/
theorem candle_ratio_problem (red_candles : ℕ) (red_ratio blue_ratio : ℕ) 
    (h_red : red_candles = 45)
    (h_ratio : red_ratio = 5 ∧ blue_ratio = 3) :
    red_candles * blue_ratio = red_ratio * 27 :=
by sorry

end NUMINAMATH_CALUDE_candle_ratio_problem_l3512_351215


namespace NUMINAMATH_CALUDE_car_speed_proof_l3512_351222

/-- The speed of a car in km/h -/
def car_speed : ℝ := 48

/-- The reference speed in km/h -/
def reference_speed : ℝ := 60

/-- The additional time taken in seconds -/
def additional_time : ℝ := 15

/-- The distance traveled in km -/
def distance : ℝ := 1

theorem car_speed_proof :
  (distance / car_speed) * 3600 = (distance / reference_speed) * 3600 + additional_time :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l3512_351222


namespace NUMINAMATH_CALUDE_x_minus_y_value_l3512_351262

theorem x_minus_y_value (x y : ℝ) (h1 : 3 = 0.2 * x) (h2 : 3 = 0.4 * y) : x - y = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l3512_351262


namespace NUMINAMATH_CALUDE_line_through_focus_iff_b_eq_neg_one_l3512_351284

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line y = x + b -/
def line (x y b : ℝ) : Prop := y = x + b

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The line passes through the focus -/
def line_passes_through_focus (b : ℝ) : Prop :=
  line (focus.1) (focus.2) b

theorem line_through_focus_iff_b_eq_neg_one :
  ∀ b : ℝ, line_passes_through_focus b ↔ b = -1 :=
sorry

end NUMINAMATH_CALUDE_line_through_focus_iff_b_eq_neg_one_l3512_351284


namespace NUMINAMATH_CALUDE_average_of_combined_results_l3512_351288

theorem average_of_combined_results (n1 n2 : ℕ) (avg1 avg2 : ℚ) 
  (h1 : n1 = 45) (h2 : n2 = 25) (h3 : avg1 = 25) (h4 : avg2 = 45) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 2250 / 70 := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l3512_351288


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_neg_seven_l3512_351269

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line l1 -/
def l1 (m : ℝ) : Line :=
  { a := 3 + m, b := 4, c := 5 - 3*m }

/-- The second line l2 -/
def l2 (m : ℝ) : Line :=
  { a := 2, b := 5 + m, c := 8 }

/-- The theorem stating that l1 and l2 are parallel iff m = -7 -/
theorem lines_parallel_iff_m_eq_neg_seven :
  ∀ m : ℝ, parallel (l1 m) (l2 m) ↔ m = -7 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_neg_seven_l3512_351269


namespace NUMINAMATH_CALUDE_flat_transactions_gain_l3512_351291

/-- Calculates the overall gain from purchasing and selling three flats with given prices and taxes -/
def overall_gain (
  purchase1 sale1 purchase2 sale2 purchase3 sale3 : ℝ
) : ℝ :=
  let purchase_tax := 0.02
  let sale_tax := 0.01
  let gain1 := sale1 * (1 - sale_tax) - purchase1 * (1 + purchase_tax)
  let gain2 := sale2 * (1 - sale_tax) - purchase2 * (1 + purchase_tax)
  let gain3 := sale3 * (1 - sale_tax) - purchase3 * (1 + purchase_tax)
  gain1 + gain2 + gain3

/-- The overall gain from the three flat transactions is $87,762 -/
theorem flat_transactions_gain :
  overall_gain 675958 725000 848592 921500 940600 982000 = 87762 := by
  sorry

end NUMINAMATH_CALUDE_flat_transactions_gain_l3512_351291


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l3512_351225

/-- Represents the number of people in each age group -/
structure PopulationGroups where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Represents the number of people to be sampled from each age group -/
structure SampleGroups where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Calculates the total population -/
def totalPopulation (p : PopulationGroups) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Calculates the proportion of each group in the sample -/
def sampleProportion (p : PopulationGroups) (sampleSize : ℕ) : SampleGroups :=
  let total := totalPopulation p
  { elderly := (p.elderly * sampleSize + total - 1) / total,
    middleAged := (p.middleAged * sampleSize + total - 1) / total,
    young := (p.young * sampleSize + total - 1) / total }

/-- The main theorem to prove -/
theorem stratified_sampling_correct 
  (population : PopulationGroups)
  (sampleSize : ℕ) :
  population.elderly = 28 →
  population.middleAged = 54 →
  population.young = 81 →
  sampleSize = 36 →
  sampleProportion population sampleSize = { elderly := 6, middleAged := 12, young := 18 } := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_correct_l3512_351225
