import Mathlib

namespace NUMINAMATH_CALUDE_three_cyclic_equations_l2648_264830

theorem three_cyclic_equations (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    a = x + 1/y ∧ a = y + 1/z ∧ a = z + 1/x) ↔ 
  (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_three_cyclic_equations_l2648_264830


namespace NUMINAMATH_CALUDE_simple_random_sampling_probability_l2648_264824

-- Define the population size
def population_size : ℕ := 100

-- Define the sample size
def sample_size : ℕ := 5

-- Define the probability of an individual being drawn
def prob_individual_drawn (n : ℕ) (k : ℕ) : ℚ := k / n

-- Theorem statement
theorem simple_random_sampling_probability :
  prob_individual_drawn population_size sample_size = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_simple_random_sampling_probability_l2648_264824


namespace NUMINAMATH_CALUDE_gcd_90_450_l2648_264832

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_450_l2648_264832


namespace NUMINAMATH_CALUDE_wood_measurement_theorem_l2648_264869

/-- Represents the measurement of a piece of wood with a rope -/
structure WoodMeasurement where
  wood_length : ℝ
  rope_length : ℝ
  surplus : ℝ
  half_rope_shortage : ℝ

/-- The system of equations accurately represents the wood measurement situation -/
def accurate_representation (m : WoodMeasurement) : Prop :=
  (m.rope_length = m.wood_length + m.surplus) ∧
  (0.5 * m.rope_length = m.wood_length - m.half_rope_shortage)

/-- Theorem stating that the given conditions lead to the correct system of equations -/
theorem wood_measurement_theorem (m : WoodMeasurement) 
  (h1 : m.surplus = 4.5)
  (h2 : m.half_rope_shortage = 1) :
  accurate_representation m := by
  sorry

end NUMINAMATH_CALUDE_wood_measurement_theorem_l2648_264869


namespace NUMINAMATH_CALUDE_polynomial_coefficient_identity_l2648_264825

theorem polynomial_coefficient_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + Real.sqrt 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_identity_l2648_264825


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_achievable_486_largest_k_is_486_l2648_264816

theorem largest_consecutive_sum (k : ℕ) : 
  (∃ a : ℕ, (k * (2 * a + k - 1)) / 2 = 3^11) → k ≤ 486 :=
by
  sorry

theorem achievable_486 : 
  ∃ a : ℕ, (486 * (2 * a + 486 - 1)) / 2 = 3^11 :=
by
  sorry

theorem largest_k_is_486 : 
  (∃ k : ℕ, k > 486 ∧ ∃ a : ℕ, (k * (2 * a + k - 1)) / 2 = 3^11) → False :=
by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_achievable_486_largest_k_is_486_l2648_264816


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l2648_264851

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ (x y : ℝ), |f x - f y| ≤ (x - y)^2) :
  ∃ (c : ℝ), ∀ (x : ℝ), f x = c :=
sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l2648_264851


namespace NUMINAMATH_CALUDE_sum_of_ages_l2648_264844

-- Define the present ages of father and son
def father_age : ℚ := sorry
def son_age : ℚ := sorry

-- Define the conditions
def present_ratio : father_age / son_age = 7 / 4 := sorry
def future_ratio : (father_age + 10) / (son_age + 10) = 5 / 3 := sorry

-- Theorem to prove
theorem sum_of_ages : father_age + son_age = 220 := by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2648_264844


namespace NUMINAMATH_CALUDE_volunteer_selection_l2648_264812

/-- The number of ways to select 3 volunteers from 5, with at most one of A and B --/
def select_volunteers (total : ℕ) (to_select : ℕ) (special : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose (total - special) (to_select - special)

/-- Theorem stating that selecting 3 from 5 with at most one of two special volunteers results in 7 ways --/
theorem volunteer_selection :
  select_volunteers 5 3 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_l2648_264812


namespace NUMINAMATH_CALUDE_road_trip_driving_time_l2648_264855

/-- Calculates the total driving time for a road trip given the number of days and daily driving hours for two people. -/
def total_driving_time (days : ℕ) (person1_hours : ℕ) (person2_hours : ℕ) : ℕ :=
  days * (person1_hours + person2_hours)

/-- Theorem stating that for a 3-day road trip with given driving hours, the total driving time is 42 hours. -/
theorem road_trip_driving_time :
  total_driving_time 3 8 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_driving_time_l2648_264855


namespace NUMINAMATH_CALUDE_f_composition_value_l2648_264836

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin (Real.pi * x)
  else Real.cos (Real.pi * x / 2 + Real.pi / 3)

theorem f_composition_value : f (f (15/2)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l2648_264836


namespace NUMINAMATH_CALUDE_find_number_l2648_264814

theorem find_number : ∃! x : ℝ, ((((x - 74) * 15) / 5) + 16) - 15 = 58 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2648_264814


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2648_264865

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := 3 * (x + 1)^2 - 2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 6 * (x + 1)

theorem quadratic_function_properties :
  (f 1 = 10) ∧
  (f (-1) = -2) ∧
  (∀ x > -1, f' x > 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2648_264865


namespace NUMINAMATH_CALUDE_zero_of_f_l2648_264808

/-- The function f(x) = 4x - 2 -/
def f (x : ℝ) : ℝ := 4 * x - 2

/-- Theorem: The zero of the function f(x) = 4x - 2 is 1/2 -/
theorem zero_of_f : ∃ x : ℝ, f x = 0 ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_zero_of_f_l2648_264808


namespace NUMINAMATH_CALUDE_id_number_2520_l2648_264879

/-- A type representing a 7-digit identification number -/
def IdNumber := Fin 7 → Fin 7

/-- The set of all valid identification numbers -/
def ValidIdNumbers : Set IdNumber :=
  {id | Function.Injective id ∧ (∀ i, id i < 7)}

/-- Lexicographical order on identification numbers -/
def IdLexOrder (id1 id2 : IdNumber) : Prop :=
  ∃ k, (∀ i < k, id1 i = id2 i) ∧ id1 k < id2 k

/-- The nth identification number in lexicographical order -/
noncomputable def nthIdNumber (n : ℕ) : IdNumber :=
  sorry

/-- The main theorem: the 2520th identification number is 4376521 -/
theorem id_number_2520 :
  nthIdNumber 2520 = λ i =>
    match i with
    | 0 => 3  -- 4 (0-indexed)
    | 1 => 5  -- 6
    | 2 => 2  -- 3
    | 3 => 5  -- 6
    | 4 => 4  -- 5
    | 5 => 1  -- 2
    | 6 => 0  -- 1
  := by sorry

end NUMINAMATH_CALUDE_id_number_2520_l2648_264879


namespace NUMINAMATH_CALUDE_narrowest_strip_for_specific_figure_l2648_264835

/-- Represents a plane figure composed of an equilateral triangle and circular arcs --/
structure TriangleWithArcs where
  side_length : ℝ
  small_radius : ℝ
  large_radius : ℝ

/-- Calculates the narrowest strip width for a given TriangleWithArcs --/
def narrowest_strip_width (figure : TriangleWithArcs) : ℝ :=
  figure.small_radius + figure.large_radius

/-- Theorem stating that for the specific figure described, the narrowest strip width is 6 units --/
theorem narrowest_strip_for_specific_figure :
  let figure : TriangleWithArcs := {
    side_length := 4,
    small_radius := 1,
    large_radius := 5
  }
  narrowest_strip_width figure = 6 := by
  sorry

end NUMINAMATH_CALUDE_narrowest_strip_for_specific_figure_l2648_264835


namespace NUMINAMATH_CALUDE_square_sum_of_linear_equations_l2648_264882

theorem square_sum_of_linear_equations (x y : ℝ) 
  (eq1 : 3 * x + 4 * y = 30) 
  (eq2 : x + 2 * y = 13) : 
  x^2 + y^2 = 145/4 := by sorry

end NUMINAMATH_CALUDE_square_sum_of_linear_equations_l2648_264882


namespace NUMINAMATH_CALUDE_vector_sum_triangle_l2648_264800

-- Define a triangle ABC in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vector addition
def vectorAdd (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector subtraction (used to represent directed edges)
def vectorSub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Theorem statement
theorem vector_sum_triangle (t : Triangle) : 
  vectorAdd (vectorAdd (vectorSub t.B t.A) (vectorSub t.C t.B)) (vectorSub t.A t.C) = (0, 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_triangle_l2648_264800


namespace NUMINAMATH_CALUDE_maci_red_pens_l2648_264886

/-- The number of blue pens Maci needs -/
def blue_pens : ℕ := 10

/-- The cost of a blue pen in cents -/
def blue_pen_cost : ℕ := 10

/-- The cost of a red pen in cents -/
def red_pen_cost : ℕ := 2 * blue_pen_cost

/-- The total cost of all pens in cents -/
def total_cost : ℕ := 400

/-- The number of red pens Maci needs -/
def red_pens : ℕ := 15

theorem maci_red_pens :
  blue_pens * blue_pen_cost + red_pens * red_pen_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_maci_red_pens_l2648_264886


namespace NUMINAMATH_CALUDE_sum_inequality_l2648_264817

/-- Given real numbers x₁, x₂, x₃ such that the sum of any two is greater than the third,
    prove that (2/3) * (∑ xᵢ) * (∑ xᵢ²) > ∑ xᵢ³ + x₁x₂x₃ -/
theorem sum_inequality (x₁ x₂ x₃ : ℝ) 
    (h₁ : x₁ + x₂ > x₃) (h₂ : x₂ + x₃ > x₁) (h₃ : x₃ + x₁ > x₂) :
    2/3 * (x₁ + x₂ + x₃) * (x₁^2 + x₂^2 + x₃^2) > x₁^3 + x₂^3 + x₃^3 + x₁*x₂*x₃ := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2648_264817


namespace NUMINAMATH_CALUDE_polygon_with_20_diagonals_is_octagon_l2648_264806

theorem polygon_with_20_diagonals_is_octagon :
  ∀ n : ℕ, n > 2 → (n * (n - 3)) / 2 = 20 → n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_with_20_diagonals_is_octagon_l2648_264806


namespace NUMINAMATH_CALUDE_bijection_image_l2648_264826

def B : Set ℤ := {-3, 3, 5}

def f (x : ℤ) : ℤ := 2 * x - 1

theorem bijection_image (A : Set ℤ) :
  (Function.Bijective f) → (f '' A = B) → A = {-1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_bijection_image_l2648_264826


namespace NUMINAMATH_CALUDE_jason_read_all_books_l2648_264859

theorem jason_read_all_books 
  (jason_books : ℕ) 
  (mary_books : ℕ) 
  (total_books : ℕ) 
  (h1 : jason_books = 18) 
  (h2 : mary_books = 42) 
  (h3 : total_books = 60) 
  (h4 : jason_books + mary_books = total_books) : 
  jason_books = 18 := by
sorry

end NUMINAMATH_CALUDE_jason_read_all_books_l2648_264859


namespace NUMINAMATH_CALUDE_time_to_restaurant_is_10_minutes_l2648_264861

/-- Time in minutes to walk from Park Office to Hidden Lake -/
def time_to_hidden_lake : ℕ := 15

/-- Time in minutes to walk from Hidden Lake to Park Office -/
def time_from_hidden_lake : ℕ := 7

/-- Total time in minutes for the entire journey (including Lake Park restaurant) -/
def total_time : ℕ := 32

/-- Time in minutes to walk from Park Office to Lake Park restaurant -/
def time_to_restaurant : ℕ := total_time - (time_to_hidden_lake + time_from_hidden_lake)

theorem time_to_restaurant_is_10_minutes : time_to_restaurant = 10 := by
  sorry

end NUMINAMATH_CALUDE_time_to_restaurant_is_10_minutes_l2648_264861


namespace NUMINAMATH_CALUDE_cube_angle_sum_prove_cube_angle_sum_l2648_264860

/-- The sum of three right angles and one angle formed by a face diagonal in a cube is 330 degrees. -/
theorem cube_angle_sum : ℝ → Prop :=
  fun (cube_angle_sum : ℝ) =>
    let right_angle : ℝ := 90
    let face_diagonal_angle : ℝ := 60
    cube_angle_sum = 3 * right_angle + face_diagonal_angle ∧ cube_angle_sum = 330

/-- Proof of the theorem -/
theorem prove_cube_angle_sum : ∃ (x : ℝ), cube_angle_sum x :=
  sorry

end NUMINAMATH_CALUDE_cube_angle_sum_prove_cube_angle_sum_l2648_264860


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2648_264897

theorem fraction_subtraction : (15 : ℚ) / 45 - (1 + 2 / 9) = -8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2648_264897


namespace NUMINAMATH_CALUDE_range_positive_iff_l2648_264820

/-- The quadratic function f(x) = ax^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

/-- The range of f is a subset of positive real numbers -/
def range_subset_positive (a : ℝ) : Prop :=
  ∀ x, f a x > 0

/-- The necessary and sufficient condition for the range of f to be a subset of positive real numbers -/
theorem range_positive_iff (a : ℝ) :
  range_subset_positive a ↔ 0 ≤ a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_range_positive_iff_l2648_264820


namespace NUMINAMATH_CALUDE_equation_solution_l2648_264870

theorem equation_solution : 
  ∃ x : ℝ, (5 * 0.85) / x - (8 * 2.25) = 5.5 ∧ x = 4.25 / 23.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2648_264870


namespace NUMINAMATH_CALUDE_modular_inverse_97_mod_101_l2648_264842

theorem modular_inverse_97_mod_101 :
  ∃ x : ℕ, x < 101 ∧ (97 * x) % 101 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_modular_inverse_97_mod_101_l2648_264842


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l2648_264858

/-- Number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute 6 4 = 72 := by sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l2648_264858


namespace NUMINAMATH_CALUDE_negative_two_inequality_l2648_264896

theorem negative_two_inequality (a b : ℝ) (h : a < b) : -2*a > -2*b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_inequality_l2648_264896


namespace NUMINAMATH_CALUDE_complex_expression_value_l2648_264875

theorem complex_expression_value (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l2648_264875


namespace NUMINAMATH_CALUDE_circle_equation_l2648_264822

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := y = x - 1

-- Define the line l₂
def l₂ (x : ℝ) : Prop := x = -1

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6*x + 1 = 0

-- Define the circle equation
def circle_eq (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

theorem circle_equation 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : l₁ x₁ y₁) 
  (h₂ : l₁ x₂ y₂) 
  (h₃ : quadratic_eq x₁) 
  (h₄ : quadratic_eq x₂) :
  ∃ (a b r : ℝ), 
    (circle_eq x₁ y₁ a b r ∧ 
     circle_eq x₂ y₂ a b r ∧ 
     (a = 3 ∧ b = 2 ∧ r = 4) ∨ 
     (a = 11 ∧ b = -6 ∧ r = 12)) ∧
    ∀ (x : ℝ), l₂ x → (x - a)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2648_264822


namespace NUMINAMATH_CALUDE_problem_solution_l2648_264829

open Real

-- Define the given condition
def alpha_condition (α : ℝ) : Prop := 2 * sin α = cos α

-- Define that α is in the third quadrant
def third_quadrant (α : ℝ) : Prop := π < α ∧ α < 3 * π / 2

theorem problem_solution (α : ℝ) 
  (h1 : alpha_condition α) 
  (h2 : third_quadrant α) : 
  (cos (π - α) = 2 * sqrt 5 / 5) ∧ 
  ((1 + 2 * sin α * sin (π / 2 - α)) / (sin α ^ 2 - cos α ^ 2) = -3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2648_264829


namespace NUMINAMATH_CALUDE_pascal_triangle_50th_row_third_number_l2648_264828

theorem pascal_triangle_50th_row_third_number :
  Nat.choose 50 2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_50th_row_third_number_l2648_264828


namespace NUMINAMATH_CALUDE_ship_passengers_l2648_264839

theorem ship_passengers :
  ∀ (P : ℕ),
  (P / 20 : ℚ) + (P / 15 : ℚ) + (P / 10 : ℚ) + (P / 12 : ℚ) + (P / 30 : ℚ) + 60 = P →
  P = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l2648_264839


namespace NUMINAMATH_CALUDE_benjamin_car_insurance_expenditure_l2648_264867

/-- The annual expenditure on car insurance, given the total expenditure over a decade -/
def annual_expenditure (total_expenditure : ℕ) (years : ℕ) : ℕ :=
  total_expenditure / years

/-- Theorem stating that the annual expenditure is 3000 dollars given the conditions -/
theorem benjamin_car_insurance_expenditure :
  annual_expenditure 30000 10 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_benjamin_car_insurance_expenditure_l2648_264867


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2648_264868

def A : Set ℝ := {x | x - 1 < 5}
def B : Set ℝ := {x | -4*x + 8 < 0}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 6} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2648_264868


namespace NUMINAMATH_CALUDE_jonny_stairs_l2648_264809

theorem jonny_stairs :
  ∀ (j : ℕ),
  (j + (j / 3 - 7) = 1685) →
  j = 1521 := by
sorry

end NUMINAMATH_CALUDE_jonny_stairs_l2648_264809


namespace NUMINAMATH_CALUDE_marble_count_l2648_264833

theorem marble_count (n : ℕ) (left_pos right_pos : ℕ) 
  (h1 : left_pos = 5)
  (h2 : right_pos = 3)
  (h3 : n = left_pos + right_pos - 1) :
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l2648_264833


namespace NUMINAMATH_CALUDE_triangle_proof_l2648_264877

theorem triangle_proof (A B C : Real) (a b c : Real) (R : Real) :
  let D := (A + C) / 2  -- D is midpoint of AC
  (1/2) * Real.sin (2*B) * Real.cos C + Real.cos B ^ 2 * Real.sin C - Real.sin (A/2) * Real.cos (A/2) = 0 →
  R = Real.sqrt 3 →
  B = π/3 ∧ 
  Real.sqrt ((a^2 + c^2) * 2 - 9) / 2 = 
    Real.sqrt ((Real.sin A * R)^2 + (Real.sin C * R)^2 - (Real.sin B * R)^2 / 4) :=
by sorry


end NUMINAMATH_CALUDE_triangle_proof_l2648_264877


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_with_two_even_two_odd_l2648_264818

/-- A function that checks if a number has two even and two odd digits -/
def hasTwoEvenTwoOddDigits (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.filter (·.mod 2 = 0)).length = 2 ∧ 
  (digits.filter (·.mod 2 = 1)).length = 2

/-- The smallest positive four-digit number divisible by 9 with two even and two odd digits -/
def smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd : ℕ := 1089

theorem smallest_four_digit_divisible_by_9_with_two_even_two_odd :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n.mod 9 = 0 ∧ hasTwoEvenTwoOddDigits n →
    smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd ≤ n) ∧
  1000 ≤ smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd ∧
  smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd < 10000 ∧
  smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd.mod 9 = 0 ∧
  hasTwoEvenTwoOddDigits smallestFourDigitDivisibleBy9WithTwoEvenTwoOdd :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_9_with_two_even_two_odd_l2648_264818


namespace NUMINAMATH_CALUDE_train_length_calculation_l2648_264823

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (time_to_pass : ℝ) : 
  train_speed = 21 → 
  bridge_length = 130 → 
  time_to_pass = 142.2857142857143 → 
  ∃ (train_length : ℝ), (abs (train_length - 700) < 0.1) ∧ 
    (train_length + bridge_length = train_speed * (1000 / 3600) * time_to_pass) := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2648_264823


namespace NUMINAMATH_CALUDE_new_rectangle_area_l2648_264831

/-- Given a rectangle with sides a and b, construct a new rectangle and calculate its area -/
theorem new_rectangle_area (a b : ℝ) (ha : a = 3) (hb : b = 4) : 
  let d := Real.sqrt (a^2 + b^2)
  let new_length := d + min a b
  let new_breadth := d - max a b
  new_length * new_breadth = 8 := by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l2648_264831


namespace NUMINAMATH_CALUDE_frood_game_threshold_l2648_264843

theorem frood_game_threshold : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → m < n → (m * (m + 1)) / 2 ≤ 15 * m) ∧ (n * (n + 1)) / 2 > 15 * n := by
  sorry

end NUMINAMATH_CALUDE_frood_game_threshold_l2648_264843


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2648_264813

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := 27

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def total_diagonal_pairs (n : RegularNonagon) : ℕ := 351

/-- The number of ways to choose 4 vertices from the nonagon, 
    which correspond to intersecting diagonals -/
def intersecting_diagonal_pairs (n : RegularNonagon) : ℕ := 126

/-- The probability that two randomly chosen diagonals in a regular nonagon intersect -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  intersecting_diagonal_pairs n / total_diagonal_pairs n

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry


end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2648_264813


namespace NUMINAMATH_CALUDE_min_employees_needed_l2648_264810

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of working days for each employee per week -/
def working_days : ℕ := 5

/-- The number of rest days for each employee per week -/
def rest_days : ℕ := 2

/-- The minimum number of employees required on duty each day -/
def min_employees_per_day : ℕ := 45

/-- The minimum number of employees needed by the company -/
def min_total_employees : ℕ := 63

theorem min_employees_needed :
  ∀ (total_employees : ℕ),
    (∀ (day : Fin days_in_week),
      (total_employees * working_days) / days_in_week ≥ min_employees_per_day) →
    total_employees ≥ min_total_employees :=
by sorry

end NUMINAMATH_CALUDE_min_employees_needed_l2648_264810


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l2648_264838

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- For any geometric sequence, the average of the squares of the second and fourth terms
    is greater than or equal to the square of the third term. -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (h : IsGeometricSequence a) :
    (a 2)^2 / 2 + (a 4)^2 / 2 ≥ (a 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l2648_264838


namespace NUMINAMATH_CALUDE_only_fourth_statement_correct_l2648_264801

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the theorem
theorem only_fourth_statement_correct 
  (a b : Line) 
  (α β : Plane) 
  (distinct_lines : a ≠ b) 
  (distinct_planes : α ≠ β) :
  (∃ (a b : Line) (α β : Plane),
    perpendicular a b ∧ 
    perpendicular_plane a α ∧ 
    perpendicular_plane b β → 
    perpendicular_planes α β) ∧
  (¬∃ (a b : Line) (α : Plane),
    perpendicular a b ∧ 
    parallel a α → 
    parallel b α) ∧
  (¬∃ (a : Line) (α β : Plane),
    parallel a α ∧ 
    perpendicular_planes α β → 
    perpendicular_plane a β) ∧
  (¬∃ (a : Line) (α β : Plane),
    perpendicular_plane a β ∧ 
    perpendicular_planes α β → 
    parallel a α) :=
by sorry

end NUMINAMATH_CALUDE_only_fourth_statement_correct_l2648_264801


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2648_264850

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2648_264850


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2648_264805

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := e = 5/4

-- Define distance from focus to asymptote
def distance_focus_asymptote (d : ℝ) : Prop := d = 3

-- Theorem statement
theorem hyperbola_properties :
  ∃ (e d : ℝ), hyperbola x y ∧ eccentricity e ∧ distance_focus_asymptote d :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2648_264805


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l2648_264821

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l2648_264821


namespace NUMINAMATH_CALUDE_expression_evaluation_l2648_264892

theorem expression_evaluation : 150 * (150 - 4) - (150 * 150 - 6 + 2) = -596 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2648_264892


namespace NUMINAMATH_CALUDE_tan_triple_inequality_l2648_264893

theorem tan_triple_inequality (x y : Real) 
  (hx : 0 < x ∧ x < Real.pi / 2)
  (hy : 0 < y ∧ y < Real.pi / 2)
  (h_tan : Real.tan x = 3 * Real.tan y) :
  x - y ≤ Real.pi / 6 ∧
  (x - y = Real.pi / 6 ↔ x = Real.pi / 3 ∧ y = Real.pi / 6) := by
sorry

end NUMINAMATH_CALUDE_tan_triple_inequality_l2648_264893


namespace NUMINAMATH_CALUDE_zero_not_necessarily_in_2_5_l2648_264888

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of having a unique zero in an interval
def has_unique_zero_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

-- State the theorem
theorem zero_not_necessarily_in_2_5 :
  (has_unique_zero_in f 1 3) →
  (has_unique_zero_in f 1 4) →
  (has_unique_zero_in f 1 5) →
  ¬ (∀ g : ℝ → ℝ, (has_unique_zero_in g 1 3 ∧ has_unique_zero_in g 1 4 ∧ has_unique_zero_in g 1 5) → 
    (∃ x, 2 < x ∧ x < 5 ∧ g x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_zero_not_necessarily_in_2_5_l2648_264888


namespace NUMINAMATH_CALUDE_min_growth_rate_doubles_coverage_l2648_264804

-- Define the initial forest coverage area
variable (a : ℝ)
-- Define the natural growth rate
def natural_growth_rate : ℝ := 0.02
-- Define the time period in years
def years : ℕ := 10
-- Define the target multiplier for forest coverage
def target_multiplier : ℝ := 2

-- Define the function for forest coverage area after x years with natural growth
def forest_coverage (x : ℕ) : ℝ := a * (1 + natural_growth_rate) ^ x

-- Define the minimum required growth rate
def min_growth_rate : ℝ := 0.072

-- Theorem statement
theorem min_growth_rate_doubles_coverage :
  ∀ p : ℝ, p ≥ min_growth_rate →
  a * (1 + p) ^ years ≥ target_multiplier * a :=
by sorry

end NUMINAMATH_CALUDE_min_growth_rate_doubles_coverage_l2648_264804


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l2648_264880

/-- Calculates the final alcohol percentage in a solution after partial replacement -/
theorem alcohol_mixture_percentage 
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (drained_volume : ℝ)
  (replacement_percentage : ℝ)
  (h1 : initial_volume = 1)
  (h2 : initial_percentage = 0.75)
  (h3 : drained_volume = 0.4)
  (h4 : replacement_percentage = 0.5)
  (h5 : initial_volume - drained_volume + drained_volume = initial_volume) :
  let remaining_volume := initial_volume - drained_volume
  let remaining_alcohol := remaining_volume * initial_percentage
  let added_alcohol := drained_volume * replacement_percentage
  let total_alcohol := remaining_alcohol + added_alcohol
  let final_percentage := total_alcohol / initial_volume
  final_percentage = 0.65 := by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l2648_264880


namespace NUMINAMATH_CALUDE_trinomial_square_difference_l2648_264871

theorem trinomial_square_difference : (23 + 15 + 7)^2 - (23^2 + 15^2 + 7^2) = 1222 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_square_difference_l2648_264871


namespace NUMINAMATH_CALUDE_no_consecutive_integers_with_square_diff_2000_l2648_264827

theorem no_consecutive_integers_with_square_diff_2000 :
  ¬ ∃ (a : ℤ), (a + 1)^2 - a^2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_integers_with_square_diff_2000_l2648_264827


namespace NUMINAMATH_CALUDE_inequality_proof_l2648_264890

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 8) :
  (a - 2) / (a + 1) + (b - 2) / (b + 1) + (c - 2) / (c + 1) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2648_264890


namespace NUMINAMATH_CALUDE_other_pencil_length_is_12_l2648_264815

/-- The length of Isha's pencil in cubes -/
def ishas_pencil_length : ℕ := 12

/-- The total length of both pencils in cubes -/
def total_length : ℕ := 24

/-- The length of the other pencil in cubes -/
def other_pencil_length : ℕ := total_length - ishas_pencil_length

theorem other_pencil_length_is_12 : other_pencil_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_other_pencil_length_is_12_l2648_264815


namespace NUMINAMATH_CALUDE_max_carlson_jars_l2648_264857

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamJars where
  carlson_weights : List ℕ
  baby_weights : List ℕ

/-- Checks if the given JamJars satisfies the initial condition -/
def initial_condition (jars : JamJars) : Prop :=
  (jars.carlson_weights.sum = 13 * jars.baby_weights.sum) ∧
  (∀ w ∈ jars.carlson_weights, w > 0) ∧
  (∀ w ∈ jars.baby_weights, w > 0)

/-- Checks if the given JamJars satisfies the final condition after transfer -/
def final_condition (jars : JamJars) : Prop :=
  let smallest := jars.carlson_weights.minimum?
  match smallest with
  | some min =>
    ((jars.carlson_weights.sum - min) = 8 * (jars.baby_weights.sum + min)) ∧
    (∀ w ∈ jars.carlson_weights, w ≥ min)
  | none => False

/-- The main theorem stating the maximum number of jars Carlson could have initially had -/
theorem max_carlson_jars (jars : JamJars) :
  initial_condition jars → final_condition jars →
  jars.carlson_weights.length ≤ 23 :=
by sorry

#check max_carlson_jars

end NUMINAMATH_CALUDE_max_carlson_jars_l2648_264857


namespace NUMINAMATH_CALUDE_equation_solutions_l2648_264853

theorem equation_solutions :
  ∃! (x y : ℝ), y = (x + 2)^2 ∧ x * y + 2 * y = 2 ∧
  ∃ (a b c d : ℂ), a ≠ x ∧ c ≠ x ∧
    (a, b) ≠ (c, d) ∧
    b = (a + 2)^2 ∧ a * b + 2 * b = 2 ∧
    d = (c + 2)^2 ∧ c * d + 2 * d = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2648_264853


namespace NUMINAMATH_CALUDE_b_equals_484_l2648_264863

/-- Given two real numbers a and b satisfying certain conditions,
    prove that b equals 484. -/
theorem b_equals_484 (a b : ℝ) 
    (h1 : a + b = 1210)
    (h2 : (4/15) * a = (2/5) * b) : 
  b = 484 := by sorry

end NUMINAMATH_CALUDE_b_equals_484_l2648_264863


namespace NUMINAMATH_CALUDE_expression_evaluation_l2648_264802

theorem expression_evaluation : 80 + 5 * 12 / (180 / 3) = 81 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2648_264802


namespace NUMINAMATH_CALUDE_pats_family_size_l2648_264878

theorem pats_family_size (total_desserts : ℕ) (desserts_per_person : ℕ) 
  (h1 : total_desserts = 126)
  (h2 : desserts_per_person = 18) :
  total_desserts / desserts_per_person = 7 := by
  sorry

end NUMINAMATH_CALUDE_pats_family_size_l2648_264878


namespace NUMINAMATH_CALUDE_clock_cost_price_l2648_264841

/-- The cost price of each clock satisfies the given conditions -/
theorem clock_cost_price (total_clocks : ℕ) (sold_at_10_percent : ℕ) (sold_at_20_percent : ℕ) 
  (uniform_profit_percentage : ℚ) (price_difference : ℚ) :
  total_clocks = 90 →
  sold_at_10_percent = 40 →
  sold_at_20_percent = 50 →
  uniform_profit_percentage = 15 / 100 →
  price_difference = 40 →
  ∃ (cost_price : ℚ),
    cost_price = 80 ∧
    cost_price * (sold_at_10_percent * (1 + 10 / 100) + sold_at_20_percent * (1 + 20 / 100)) =
    cost_price * total_clocks * (1 + uniform_profit_percentage) + price_difference :=
by sorry

end NUMINAMATH_CALUDE_clock_cost_price_l2648_264841


namespace NUMINAMATH_CALUDE_rectangle_length_l2648_264874

theorem rectangle_length (width perimeter : ℝ) (h1 : width = 15) (h2 : perimeter = 70) :
  let length := (perimeter - 2 * width) / 2
  length = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l2648_264874


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l2648_264811

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 36) : x + y ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l2648_264811


namespace NUMINAMATH_CALUDE_craft_store_solution_l2648_264862

/-- Represents the craft store problem -/
structure CraftStore where
  markedPrice : ℝ
  costPrice : ℝ
  profitPerItem : ℝ
  discountedSales : ℕ
  discountPercentage : ℝ
  reducedPriceSales : ℕ
  priceReduction : ℝ
  dailySales : ℕ
  salesIncrease : ℕ
  priceDecreaseStep : ℝ

/-- The craft store problem statement -/
def craftStoreProblem (cs : CraftStore) : Prop :=
  -- Profit at marked price
  cs.profitPerItem = cs.markedPrice - cs.costPrice
  -- Equal profit for discounted and reduced price sales
  ∧ cs.discountedSales * (cs.markedPrice * cs.discountPercentage - cs.costPrice) =
    cs.reducedPriceSales * (cs.markedPrice - cs.priceReduction - cs.costPrice)
  -- Daily sales at marked price
  ∧ cs.dailySales * (cs.markedPrice - cs.costPrice) =
    (cs.dailySales + cs.salesIncrease) * (cs.markedPrice - cs.priceDecreaseStep - cs.costPrice)

/-- The theorem to be proved -/
theorem craft_store_solution (cs : CraftStore) 
  (h : craftStoreProblem cs) : 
  cs.costPrice = 155 
  ∧ cs.markedPrice = 200 
  ∧ (∃ optimalReduction maxProfit, 
      optimalReduction = 10 
      ∧ maxProfit = 4900 
      ∧ ∀ reduction, 
        cs.dailySales * (cs.markedPrice - reduction - cs.costPrice) 
        + (cs.salesIncrease * reduction / cs.priceDecreaseStep) 
          * (cs.markedPrice - reduction - cs.costPrice) 
        ≤ maxProfit) :=
sorry

end NUMINAMATH_CALUDE_craft_store_solution_l2648_264862


namespace NUMINAMATH_CALUDE_option_D_is_false_l2648_264898

-- Define the proposition p and q
variable (p q : Prop)

-- Define the statement for option D
def option_D : Prop := (p ∨ q) → (p ∧ q)

-- Theorem stating that option D is false
theorem option_D_is_false : ¬ (∀ p q, option_D p q) := by
  sorry

-- Note: We don't need to prove the other options are correct in this statement,
-- as the question only asks for the incorrect option.

end NUMINAMATH_CALUDE_option_D_is_false_l2648_264898


namespace NUMINAMATH_CALUDE_probability_both_asian_l2648_264848

def asian_countries : ℕ := 3
def european_countries : ℕ := 3
def total_countries : ℕ := asian_countries + european_countries
def countries_to_select : ℕ := 2

def total_outcomes : ℕ := (total_countries.choose countries_to_select)
def favorable_outcomes : ℕ := (asian_countries.choose countries_to_select)

theorem probability_both_asian :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_both_asian_l2648_264848


namespace NUMINAMATH_CALUDE_glue_drops_in_cube_l2648_264845

/-- 
For an n × n × n cube built from n³ unit cubes, where one drop of glue is used for each pair 
of touching faces between two cubes, the total number of glue drops used is 3n²(n-1).
-/
theorem glue_drops_in_cube (n : ℕ) : 
  n > 0 → 3 * n^2 * (n - 1) = 
    (n - 1) * n * n  -- drops for vertical contacts
    + (n - 1) * n * n  -- drops for horizontal contacts
    + (n - 1) * n * n  -- drops for depth contacts
  := by sorry

end NUMINAMATH_CALUDE_glue_drops_in_cube_l2648_264845


namespace NUMINAMATH_CALUDE_max_value_of_f_l2648_264895

/-- The quadratic function f(x) = -3x^2 + 6x + 2 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 6 * x + 2

/-- The maximum value of f(x) is 5 -/
theorem max_value_of_f : ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2648_264895


namespace NUMINAMATH_CALUDE_dinner_lunch_ratio_is_two_l2648_264846

/-- Represents the daily calorie intake of John -/
structure DailyCalories where
  breakfast : ℕ
  lunch : ℕ
  dinner : ℕ
  shakes : ℕ
  total : ℕ

/-- The ratio of dinner calories to lunch calories -/
def dinner_lunch_ratio (dc : DailyCalories) : ℚ :=
  dc.dinner / dc.lunch

/-- John's daily calorie intake satisfies the given conditions -/
def johns_calories : DailyCalories :=
  { breakfast := 500,
    lunch := 500 + (500 * 25 / 100),
    dinner := 3275 - (500 + (500 + (500 * 25 / 100)) + (3 * 300)),
    shakes := 3 * 300,
    total := 3275 }

theorem dinner_lunch_ratio_is_two : dinner_lunch_ratio johns_calories = 2 := by
  sorry

end NUMINAMATH_CALUDE_dinner_lunch_ratio_is_two_l2648_264846


namespace NUMINAMATH_CALUDE_point_difference_l2648_264807

-- Define the value of a touchdown
def touchdown_value : ℕ := 7

-- Define the number of touchdowns for each team
def brayden_gavin_touchdowns : ℕ := 7
def cole_freddy_touchdowns : ℕ := 9

-- Calculate the points for each team
def brayden_gavin_points : ℕ := brayden_gavin_touchdowns * touchdown_value
def cole_freddy_points : ℕ := cole_freddy_touchdowns * touchdown_value

-- State the theorem
theorem point_difference : cole_freddy_points - brayden_gavin_points = 14 := by
  sorry

end NUMINAMATH_CALUDE_point_difference_l2648_264807


namespace NUMINAMATH_CALUDE_specific_gathering_handshakes_l2648_264840

/-- Represents a gathering of married couples -/
structure Gathering where
  couples : ℕ
  people : ℕ
  circular : Bool
  shake_all : Bool
  no_spouse : Bool
  no_neighbors : Bool

/-- Calculates the number of handshakes in the gathering -/
def handshakes (g : Gathering) : ℕ :=
  (g.people * (g.people - 3)) / 2

/-- Theorem stating the number of handshakes for the specific gathering described in the problem -/
theorem specific_gathering_handshakes :
  let g : Gathering := {
    couples := 8,
    people := 16,
    circular := true,
    shake_all := true,
    no_spouse := true,
    no_neighbors := true
  }
  handshakes g = 96 := by
  sorry

end NUMINAMATH_CALUDE_specific_gathering_handshakes_l2648_264840


namespace NUMINAMATH_CALUDE_height_in_meters_l2648_264834

-- Define Xiaochao's height in meters and centimeters
def height_m : ℝ := 1
def height_cm : ℝ := 36

-- Theorem to prove
theorem height_in_meters : height_m + height_cm / 100 = 1.36 := by
  sorry

end NUMINAMATH_CALUDE_height_in_meters_l2648_264834


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2648_264873

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 2) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2648_264873


namespace NUMINAMATH_CALUDE_metallic_sheet_dimension_l2648_264837

/-- Given a rectangular metallic sheet, prove that the second dimension is 36 m -/
theorem metallic_sheet_dimension (sheet_length : ℝ) (sheet_width : ℝ) 
  (cut_length : ℝ) (box_volume : ℝ) :
  sheet_length = 46 →
  cut_length = 8 →
  box_volume = 4800 →
  box_volume = (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length →
  sheet_width = 36 := by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_dimension_l2648_264837


namespace NUMINAMATH_CALUDE_line_intersects_parabola_vertex_once_l2648_264856

theorem line_intersects_parabola_vertex_once :
  ∃! b : ℝ, ∀ x y : ℝ,
    (y = 2 * x + b) ∧ (y = x^2 + 2 * b * x) →
    (x = -b ∧ y = -b^2) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_vertex_once_l2648_264856


namespace NUMINAMATH_CALUDE_problem_proof_l2648_264866

theorem problem_proof : (-8: ℝ) ^ (1/3) + π^0 + Real.log 4 + Real.log 25 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2648_264866


namespace NUMINAMATH_CALUDE_little_john_friends_money_l2648_264899

/-- Calculates the amount given to each friend by Little John --/
theorem little_john_friends_money 
  (initial_amount : ℚ) 
  (sweets_cost : ℚ) 
  (num_friends : ℕ) 
  (remaining_amount : ℚ) 
  (h1 : initial_amount = 8.5)
  (h2 : sweets_cost = 1.25)
  (h3 : num_friends = 2)
  (h4 : remaining_amount = 4.85) :
  (initial_amount - remaining_amount - sweets_cost) / num_friends = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_little_john_friends_money_l2648_264899


namespace NUMINAMATH_CALUDE_parents_john_age_ratio_l2648_264889

/-- Given information about Mark, John, and their parents' ages, prove the ratio of parents' age to John's age -/
theorem parents_john_age_ratio :
  ∀ (mark_age john_age parents_age : ℕ),
    mark_age = 18 →
    john_age = mark_age - 10 →
    parents_age = 22 + mark_age →
    parents_age / john_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_parents_john_age_ratio_l2648_264889


namespace NUMINAMATH_CALUDE_equation_system_solvability_l2648_264885

theorem equation_system_solvability : ∃ (x y z : ℝ), 
  (2 * x + y = 4) ∧ 
  (x^2 + 3 * y = 5) ∧ 
  (3 * x - 1.5 * y + z = 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solvability_l2648_264885


namespace NUMINAMATH_CALUDE_fast_clock_accuracy_l2648_264883

/-- Represents time in minutes since midnight -/
def Time := ℕ

/-- Converts hours and minutes to total minutes -/
def toMinutes (hours minutes : ℕ) : Time :=
  hours * 60 + minutes

/-- A fast-running clock that gains time at a constant rate -/
structure FastClock where
  /-- The rate at which the clock gains time, represented as (gained_minutes, real_minutes) -/
  rate : ℕ × ℕ
  /-- The current time shown on the fast clock -/
  current_time : Time

/-- Calculates the actual time given a FastClock -/
def actualTime (clock : FastClock) (start_time : Time) : Time :=
  sorry

theorem fast_clock_accuracy (start_time : Time) (end_time : Time) :
  let initial_clock : FastClock := { rate := (15, 45), current_time := start_time }
  let final_clock : FastClock := { rate := (15, 45), current_time := end_time }
  start_time = toMinutes 15 0 →
  end_time = toMinutes 23 0 →
  actualTime final_clock start_time = toMinutes 23 15 :=
  sorry

end NUMINAMATH_CALUDE_fast_clock_accuracy_l2648_264883


namespace NUMINAMATH_CALUDE_polynomial_coefficient_problem_l2648_264881

theorem polynomial_coefficient_problem (a b : ℝ) : 
  (0 < a) → 
  (0 < b) → 
  (a + b = 1) → 
  (21 * a^10 * b^4 = 35 * a^8 * b^6) → 
  a = 5 / (5 + Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_problem_l2648_264881


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2648_264803

theorem rectangle_area_change (initial_area : ℝ) : 
  initial_area = 540 →
  (0.8 * 1.15) * initial_area = 496.8 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2648_264803


namespace NUMINAMATH_CALUDE_cersei_cotton_candies_l2648_264819

/-- The number of cotton candies Cersei initially bought -/
def initial_candies : ℕ := 40

/-- The number of cotton candies given to brother and sister -/
def given_to_siblings : ℕ := 10

/-- The fraction of remaining candies given to cousin -/
def fraction_to_cousin : ℚ := 1/4

/-- The number of cotton candies Cersei ate -/
def eaten_candies : ℕ := 12

/-- The number of cotton candies left at the end -/
def remaining_candies : ℕ := 18

theorem cersei_cotton_candies : 
  initial_candies = 40 ∧
  (initial_candies - given_to_siblings) * (1 - fraction_to_cousin) - eaten_candies = remaining_candies :=
by sorry

end NUMINAMATH_CALUDE_cersei_cotton_candies_l2648_264819


namespace NUMINAMATH_CALUDE_ice_cream_volume_l2648_264854

/-- The volume of ice cream on a cone -/
theorem ice_cream_volume (r h : ℝ) (hr : r = 3) (hh : h = 1) :
  (2/3) * π * r^3 + π * r^2 * h = 27 * π := by sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l2648_264854


namespace NUMINAMATH_CALUDE_pool_filling_time_l2648_264847

theorem pool_filling_time (pool_capacity : ℝ) (num_hoses : ℕ) (flow_rate : ℝ) : 
  pool_capacity = 24000 ∧ 
  num_hoses = 4 ∧ 
  flow_rate = 2.5 → 
  pool_capacity / (num_hoses * flow_rate * 60) = 40 := by
sorry

end NUMINAMATH_CALUDE_pool_filling_time_l2648_264847


namespace NUMINAMATH_CALUDE_number_of_smaller_cubes_l2648_264849

theorem number_of_smaller_cubes (surface_area : ℝ) (small_cube_volume : ℝ) : 
  surface_area = 5400 → small_cube_volume = 216 → 
  (surface_area / 6).sqrt ^ 3 / small_cube_volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_of_smaller_cubes_l2648_264849


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l2648_264864

theorem gcd_special_numbers : Nat.gcd 777777777 222222222222 = 999 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l2648_264864


namespace NUMINAMATH_CALUDE_total_weight_AlI3_is_3261_44_l2648_264884

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of aluminum atoms in a molecule of AlI3 -/
def num_Al_atoms : ℕ := 1

/-- The number of iodine atoms in a molecule of AlI3 -/
def num_I_atoms : ℕ := 3

/-- The number of moles of AlI3 -/
def num_moles : ℝ := 8

/-- The total weight of AlI3 in grams -/
def total_weight_AlI3 : ℝ :=
  num_moles * (num_Al_atoms * atomic_weight_Al + num_I_atoms * atomic_weight_I)

theorem total_weight_AlI3_is_3261_44 : total_weight_AlI3 = 3261.44 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_AlI3_is_3261_44_l2648_264884


namespace NUMINAMATH_CALUDE_existence_of_m_n_l2648_264891

theorem existence_of_m_n (p : ℕ) (hp : Prime p) (hp_gt_10 : p > 10) :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m + n < p ∧ (p ∣ (5^m * 7^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l2648_264891


namespace NUMINAMATH_CALUDE_quadrilateral_angle_l2648_264852

/-- 
Given a quadrilateral with angles α₁, α₂, α₃, α₄, α₅ satisfying:
1) α₁ + α₂ = 180°
2) α₃ = α₄
3) α₂ + α₅ = 180°
Prove that α₄ = 90°
-/
theorem quadrilateral_angle (α₁ α₂ α₃ α₄ α₅ : ℝ) 
  (h1 : α₁ + α₂ = 180)
  (h2 : α₃ = α₄)
  (h3 : α₂ + α₅ = 180)
  (h4 : α₁ + α₂ + α₃ + α₄ = 360) :  -- sum of angles in a quadrilateral
  α₄ = 90 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_l2648_264852


namespace NUMINAMATH_CALUDE_distance_center_to_point_l2648_264887

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 6*x - 8*y + 9

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, -4)

-- Define the given point
def given_point : ℝ × ℝ := (5, -3)

-- Statement to prove
theorem distance_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_point_l2648_264887


namespace NUMINAMATH_CALUDE_shirt_price_shopping_scenario_l2648_264894

/-- The price of a shirt given the shopping scenario --/
theorem shirt_price (total_paid : ℝ) (num_shorts : ℕ) (price_per_short : ℝ) 
  (num_shirts : ℕ) (senior_discount : ℝ) : ℝ :=
  let shorts_cost := num_shorts * price_per_short
  let discounted_shorts_cost := shorts_cost * (1 - senior_discount)
  let shirts_cost := total_paid - discounted_shorts_cost
  shirts_cost / num_shirts

/-- The price of each shirt in the given shopping scenario is $15.30 --/
theorem shopping_scenario : 
  shirt_price 117 3 15 5 0.1 = 15.3 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_shopping_scenario_l2648_264894


namespace NUMINAMATH_CALUDE_chord_circuit_l2648_264876

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two adjacent chords is 60°, then the minimum number of such chords
    needed to form a complete circuit is 3. -/
theorem chord_circuit (angle : ℝ) (n : ℕ) : angle = 60 → n * angle = 360 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_circuit_l2648_264876


namespace NUMINAMATH_CALUDE_min_cans_needed_l2648_264872

/-- The capacity of each can in ounces -/
def can_capacity : ℕ := 15

/-- The minimum amount of soda needed in ounces -/
def min_soda_amount : ℕ := 192

/-- The minimum number of cans needed -/
def min_cans : ℕ := 13

theorem min_cans_needed : 
  (∀ n : ℕ, n * can_capacity ≥ min_soda_amount → n ≥ min_cans) ∧ 
  (min_cans * can_capacity ≥ min_soda_amount) := by
  sorry

end NUMINAMATH_CALUDE_min_cans_needed_l2648_264872
