import Mathlib

namespace NUMINAMATH_CALUDE_james_balloons_l2760_276082

-- Define the number of balloons Amy has
def amy_balloons : ℕ := 513

-- Define the difference in balloons between James and Amy
def difference : ℕ := 208

-- Theorem statement
theorem james_balloons : amy_balloons + difference = 721 := by
  sorry

end NUMINAMATH_CALUDE_james_balloons_l2760_276082


namespace NUMINAMATH_CALUDE_probability_no_repetition_l2760_276042

def three_digit_numbers : ℕ := 3^3

def numbers_without_repetition : ℕ := 6

theorem probability_no_repetition :
  (numbers_without_repetition : ℚ) / three_digit_numbers = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_repetition_l2760_276042


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2760_276004

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 - 6 * x + m = 0) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2760_276004


namespace NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l2760_276074

theorem sin_x_squared_not_periodic : ¬∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), Real.sin ((x + p)^2) = Real.sin (x^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l2760_276074


namespace NUMINAMATH_CALUDE_rolling_cone_surface_area_l2760_276015

/-- The surface area described by the height of a rolling cone -/
theorem rolling_cone_surface_area (h l : ℝ) (h_pos : 0 < h) (l_pos : 0 < l) :
  let surface_area := π * h^3 / l
  surface_area = π * h^3 / l :=
by sorry

end NUMINAMATH_CALUDE_rolling_cone_surface_area_l2760_276015


namespace NUMINAMATH_CALUDE_uber_lyft_cost_difference_uber_lyft_cost_difference_proof_l2760_276000

/-- The cost difference between Uber and Lyft rides --/
theorem uber_lyft_cost_difference : ℝ :=
  let taxi_cost : ℝ := 15  -- Derived from the 20% tip condition
  let lyft_cost : ℝ := taxi_cost + 4
  let uber_cost : ℝ := 22
  uber_cost - lyft_cost

/-- Proof of the cost difference between Uber and Lyft rides --/
theorem uber_lyft_cost_difference_proof :
  uber_lyft_cost_difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_uber_lyft_cost_difference_uber_lyft_cost_difference_proof_l2760_276000


namespace NUMINAMATH_CALUDE_probability_theorem_l2760_276039

/-- Represents a club with members -/
structure Club where
  total : ℕ
  girls : ℕ
  boys : ℕ
  girls_under_18 : ℕ

/-- Calculates the probability of choosing two girls with at least one under 18 -/
def probability_two_girls_one_under_18 (club : Club) : ℚ :=
  let total_combinations := club.total.choose 2
  let girls_combinations := club.girls.choose 2
  let underaged_combinations := 
    club.girls_under_18 * (club.girls - club.girls_under_18) + club.girls_under_18.choose 2
  (underaged_combinations : ℚ) / total_combinations

/-- The main theorem to prove -/
theorem probability_theorem (club : Club) 
    (h1 : club.total = 15)
    (h2 : club.girls = 8)
    (h3 : club.boys = 7)
    (h4 : club.girls_under_18 = 3)
    (h5 : club.total = club.girls + club.boys) :
  probability_two_girls_one_under_18 club = 6/35 := by
  sorry

#eval probability_two_girls_one_under_18 ⟨15, 8, 7, 3⟩

end NUMINAMATH_CALUDE_probability_theorem_l2760_276039


namespace NUMINAMATH_CALUDE_school_population_l2760_276070

/-- Represents the total number of students in the school -/
def total_students : ℕ := 50

/-- Represents the number of students of 8 years of age -/
def students_8_years : ℕ := 24

/-- Represents the fraction of students below 8 years of age -/
def fraction_below_8 : ℚ := 1/5

/-- Represents the ratio of students above 8 years to students of 8 years -/
def ratio_above_to_8 : ℚ := 2/3

theorem school_population :
  (students_8_years : ℚ) + 
  (ratio_above_to_8 * students_8_years) + 
  (fraction_below_8 * total_students) = total_students := by sorry

end NUMINAMATH_CALUDE_school_population_l2760_276070


namespace NUMINAMATH_CALUDE_boys_distribution_l2760_276093

theorem boys_distribution (total_amount : ℕ) (additional_amount : ℕ) : 
  total_amount = 5040 →
  additional_amount = 80 →
  ∃ (x : ℕ), 
    x * (total_amount / 18 + additional_amount) = total_amount ∧
    x = 14 := by
  sorry

end NUMINAMATH_CALUDE_boys_distribution_l2760_276093


namespace NUMINAMATH_CALUDE_average_apples_per_day_l2760_276026

def boxes : ℕ := 12
def apples_per_box : ℕ := 25
def days : ℕ := 4

def total_apples : ℕ := boxes * apples_per_box

theorem average_apples_per_day : total_apples / days = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_apples_per_day_l2760_276026


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l2760_276019

theorem tic_tac_toe_tie_probability 
  (amy_win_prob : ℚ) 
  (lily_win_prob : ℚ) 
  (h1 : amy_win_prob = 4/9) 
  (h2 : lily_win_prob = 1/3) : 
  1 - (amy_win_prob + lily_win_prob) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l2760_276019


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2760_276024

theorem quadratic_coefficient (b : ℝ) :
  (b < 0) →
  (∃ m : ℝ, ∀ x : ℝ, x^2 + b*x + 1/4 = (x + m)^2 + 1/16) →
  b = -Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2760_276024


namespace NUMINAMATH_CALUDE_max_tennis_court_area_l2760_276083

/-- Represents the dimensions of a rectangular tennis court --/
structure CourtDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular tennis court --/
def area (d : CourtDimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular tennis court --/
def perimeter (d : CourtDimensions) : ℝ := 2 * (d.length + d.width)

/-- Checks if the court dimensions meet the minimum requirements --/
def meetsMinimumRequirements (d : CourtDimensions) : Prop :=
  d.length ≥ 85 ∧ d.width ≥ 45

/-- Theorem stating the maximum area of the tennis court --/
theorem max_tennis_court_area :
  ∃ (d : CourtDimensions),
    perimeter d = 320 ∧
    meetsMinimumRequirements d ∧
    area d = 6375 ∧
    ∀ (d' : CourtDimensions),
      perimeter d' = 320 ∧ meetsMinimumRequirements d' → area d' ≤ area d :=
by sorry

end NUMINAMATH_CALUDE_max_tennis_court_area_l2760_276083


namespace NUMINAMATH_CALUDE_cos_pi_plus_two_alpha_l2760_276088

/-- 
Given that the terminal side of angle α passes through point (3,4),
prove that cos(π+2α) = -7/25.
-/
theorem cos_pi_plus_two_alpha (α : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = 4) → 
  Real.cos (π + 2 * α) = -7/25 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_plus_two_alpha_l2760_276088


namespace NUMINAMATH_CALUDE_max_z_value_l2760_276022

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 0) (h2 : x*y + y*z + z*x = -3) :
  ∃ (max_z : ℝ), z ≤ max_z ∧ max_z = 2 := by
sorry

end NUMINAMATH_CALUDE_max_z_value_l2760_276022


namespace NUMINAMATH_CALUDE_score_calculation_l2760_276049

/-- Proves that given the average score and difference between subjects, we can determine the individual scores -/
theorem score_calculation (average : ℝ) (difference : ℝ) 
  (h_average : average = 96) 
  (h_difference : difference = 8) : 
  ∃ (chinese : ℝ) (math : ℝ), 
    chinese + math = 2 * average ∧ 
    math = chinese + difference ∧
    chinese = 92 ∧ 
    math = 100 := by
  sorry

end NUMINAMATH_CALUDE_score_calculation_l2760_276049


namespace NUMINAMATH_CALUDE_cat_litter_cost_210_days_l2760_276045

/-- Calculates the cost of cat litter for a given number of days -/
def catLitterCost (containerSize : ℕ) (containerPrice : ℕ) (litterBoxCapacity : ℕ) (changeDays : ℕ) (totalDays : ℕ) : ℕ :=
  let changes := totalDays / changeDays
  let totalLitter := changes * litterBoxCapacity
  let containers := (totalLitter + containerSize - 1) / containerSize  -- Ceiling division
  containers * containerPrice

/-- The cost of cat litter for 210 days is $210 -/
theorem cat_litter_cost_210_days :
  catLitterCost 45 21 15 7 210 = 210 := by sorry

end NUMINAMATH_CALUDE_cat_litter_cost_210_days_l2760_276045


namespace NUMINAMATH_CALUDE_parallel_to_x_axis_symmetric_in_third_quadrant_equal_distance_to_axes_l2760_276063

-- Define point P
def P (x : ℝ) : ℝ × ℝ := (2*x - 3, 3 - x)

-- Define point A
def A : ℝ × ℝ := (-3, 4)

-- Theorem 1: If AP is parallel to x-axis, then P(-5, 4)
theorem parallel_to_x_axis :
  (∃ x : ℝ, P x = (-5, 4)) ↔ (∃ x : ℝ, (P x).2 = A.2) :=
sorry

-- Theorem 2: If symmetric point is in third quadrant, then x < 3/2
theorem symmetric_in_third_quadrant :
  (∃ x : ℝ, (2*x - 3 < 0 ∧ x - 3 < 0)) ↔ (∃ x : ℝ, x < 3/2) :=
sorry

-- Theorem 3: If distances to axes are equal, then P(1,1) or P(-3,3)
theorem equal_distance_to_axes :
  (∃ x : ℝ, |2*x - 3| = |3 - x|) ↔ (P 2 = (1, 1) ∨ P 0 = (-3, 3)) :=
sorry

end NUMINAMATH_CALUDE_parallel_to_x_axis_symmetric_in_third_quadrant_equal_distance_to_axes_l2760_276063


namespace NUMINAMATH_CALUDE_intersecting_line_equation_l2760_276072

/-- A line passing through a point and intersecting both axes -/
structure IntersectingLine where
  P : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  passes_through_P : true  -- Placeholder for the condition that the line passes through P
  intersects_x_axis : A.2 = 0
  intersects_y_axis : B.1 = 0
  P_is_midpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- The equation of the line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

theorem intersecting_line_equation (l : IntersectingLine) (eq : LineEquation) :
  l.P = (-4, 6) →
  eq.a = 3 ∧ eq.b = -2 ∧ eq.c = 24 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_line_equation_l2760_276072


namespace NUMINAMATH_CALUDE_calculate_expression_quadratic_equation_roots_l2760_276098

-- Problem 1
theorem calculate_expression : 
  (Real.sqrt 2 - Real.sqrt 12 + Real.sqrt (1/2)) * Real.sqrt 3 = 3 * Real.sqrt 6 / 2 - 6 := by sorry

-- Problem 2
theorem quadratic_equation_roots (c : ℝ) (h : (2 + Real.sqrt 3)^2 - 4*(2 + Real.sqrt 3) + c = 0) :
  ∃ (x : ℝ), x^2 - 4*x + c = 0 ∧ x ≠ 2 + Real.sqrt 3 ∧ x = 2 - Real.sqrt 3 ∧ c = 1 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_quadratic_equation_roots_l2760_276098


namespace NUMINAMATH_CALUDE_intersection_of_polar_curves_l2760_276094

/-- The intersection point of two polar curves -/
theorem intersection_of_polar_curves (ρ θ : ℝ) :
  ρ ≥ 0 →
  0 ≤ θ →
  θ < π / 2 →
  ρ * Real.cos θ = 3 →
  ρ = 4 * Real.cos θ →
  (ρ = 2 * Real.sqrt 3 ∧ θ = π / 6) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_polar_curves_l2760_276094


namespace NUMINAMATH_CALUDE_triangle_side_length_l2760_276060

/-- Given a triangle ABC with the condition that cos(∠A - ∠B) + sin(∠A + ∠B) = 2 and AB = 4,
    prove that BC = 2√2 -/
theorem triangle_side_length (A B C : ℝ) (h1 : 0 < A ∧ A < π)
    (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π)
    (h5 : Real.cos (A - B) + Real.sin (A + B) = 2) (h6 : ∃ AB : ℝ, AB = 4) :
    ∃ BC : ℝ, BC = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2760_276060


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2760_276085

theorem simplify_trig_expression (x : ℝ) : 
  (Real.sqrt 3 / 2) * Real.sin x - (1 / 2) * Real.cos x = Real.sin (x - π / 6) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2760_276085


namespace NUMINAMATH_CALUDE_jack_coffee_batch_size_l2760_276001

/-- Proves that Jack makes 1.5 gallons of cold brew coffee in each batch given the conditions --/
theorem jack_coffee_batch_size :
  let coffee_per_2days : ℝ := 96  -- ounces
  let days : ℝ := 24
  let hours_per_batch : ℝ := 20
  let total_hours : ℝ := 120
  let ounces_per_gallon : ℝ := 128
  
  let total_coffee := (days / 2) * coffee_per_2days
  let total_gallons := total_coffee / ounces_per_gallon
  let num_batches := total_hours / hours_per_batch
  let gallons_per_batch := total_gallons / num_batches
  
  gallons_per_batch = 1.5 := by sorry

end NUMINAMATH_CALUDE_jack_coffee_batch_size_l2760_276001


namespace NUMINAMATH_CALUDE_two_thousandth_point_l2760_276007

/-- Represents a point in the first quadrant with integer coordinates -/
structure Point where
  x : Nat
  y : Nat

/-- The spiral numbering function that assigns a natural number to each point -/
def spiralNumber : Point → Nat := sorry

/-- The inverse function that finds the point corresponding to a given number -/
def spiralPoint : Nat → Point := sorry

/-- Theorem stating that the 2000th point in the spiral has coordinates (44, 24) -/
theorem two_thousandth_point : spiralPoint 2000 = Point.mk 44 24 := by sorry

end NUMINAMATH_CALUDE_two_thousandth_point_l2760_276007


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2760_276078

theorem inequality_equivalence (x : ℝ) :
  (3 * x - 5 ≥ 9 - 2 * x) ↔ (x ≥ 14 / 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2760_276078


namespace NUMINAMATH_CALUDE_distance_to_origin_is_sqrt2_l2760_276076

-- Define the ellipse parameters
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = 1/2

-- Define the right focus
def right_focus (a c : ℝ) : Prop := c^2 = a^2 / 4

-- Define the quadratic equation and its roots
def quadratic_roots (a b c x₁ x₂ : ℝ) : Prop :=
  a * x₁^2 + 2 * b * x₁ + c = 0 ∧
  a * x₂^2 + 2 * b * x₂ + c = 0

-- Theorem statement
theorem distance_to_origin_is_sqrt2
  (a b c x₁ x₂ : ℝ)
  (h_ellipse : ellipse a b x₁ x₂)
  (h_eccentricity : eccentricity (Real.sqrt (1 - b^2 / a^2)))
  (h_focus : right_focus a c)
  (h_roots : quadratic_roots a (Real.sqrt (a^2 - c^2)) c x₁ x₂) :
  Real.sqrt (x₁^2 + x₂^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_is_sqrt2_l2760_276076


namespace NUMINAMATH_CALUDE_largest_m_for_quadratic_function_l2760_276071

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem largest_m_for_quadratic_function (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b c (x - 4) = f a b c (2 - x)) →
  (∀ x, f a b c x ≥ x) →
  (∀ x ∈ Set.Ioo 0 2, f a b c x ≤ ((x + 1) / 2)^2) →
  (∃ x, ∀ y, f a b c y ≥ f a b c x) →
  (∃ x, f a b c x = 0) →
  (∃ m > 1, ∃ t, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) →
  (∀ m > 9, ¬∃ t, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_quadratic_function_l2760_276071


namespace NUMINAMATH_CALUDE_triangle_theorem_l2760_276069

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0)
  (h2 : t.a = 2)
  (h3 : 1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3) : 
  t.A = π/3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2760_276069


namespace NUMINAMATH_CALUDE_cost_difference_l2760_276013

def dan_money : ℕ := 5
def chocolate_cost : ℕ := 3
def candy_bar_cost : ℕ := 7

theorem cost_difference : candy_bar_cost - chocolate_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_l2760_276013


namespace NUMINAMATH_CALUDE_vector_computation_l2760_276014

theorem vector_computation :
  let v1 : Fin 3 → ℝ := ![3, -2, 5]
  let v2 : Fin 3 → ℝ := ![2, -3, 4]
  4 • v1 - 3 • v2 = ![6, 1, 8] :=
by
  sorry

end NUMINAMATH_CALUDE_vector_computation_l2760_276014


namespace NUMINAMATH_CALUDE_train_bridge_problem_l2760_276081

/-- Represents the problem of determining the carriage position of a person walking through a train on a bridge. -/
theorem train_bridge_problem
  (bridge_length : ℝ)
  (train_speed : ℝ)
  (person_speed : ℝ)
  (carriage_length : ℝ)
  (h_bridge : bridge_length = 1400)
  (h_train : train_speed = 54 * (1000 / 3600))
  (h_person : person_speed = 3.6 * (1000 / 3600))
  (h_carriage : carriage_length = 23)
  : ∃ (n : ℕ), 5 ≤ n ∧ n ≤ 6 ∧
    (n : ℝ) * carriage_length ≥
      person_speed * (bridge_length / (train_speed + person_speed)) ∧
    ((n + 1) : ℝ) * carriage_length >
      person_speed * (bridge_length / (train_speed + person_speed)) :=
by sorry

end NUMINAMATH_CALUDE_train_bridge_problem_l2760_276081


namespace NUMINAMATH_CALUDE_no_preimage_set_l2760_276028

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Theorem statement
theorem no_preimage_set (k : ℝ) :
  (∀ x : ℝ, f x ≠ k) ↔ k > 1 :=
sorry

end NUMINAMATH_CALUDE_no_preimage_set_l2760_276028


namespace NUMINAMATH_CALUDE_down_payment_percentage_l2760_276033

def house_price : ℝ := 100000
def parents_contribution_rate : ℝ := 0.30
def remaining_balance : ℝ := 56000

theorem down_payment_percentage :
  ∃ (down_payment_rate : ℝ),
    down_payment_rate * house_price +
    parents_contribution_rate * (house_price - down_payment_rate * house_price) +
    remaining_balance = house_price ∧
    down_payment_rate = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_percentage_l2760_276033


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2760_276057

theorem quadratic_equation_properties (a b : ℝ) (ha : a > 0) (hab : a^2 = 4*b) :
  (a^2 - b^2 ≤ 4) ∧
  (a^2 + 1/b ≥ 4) ∧
  (∀ c : ℝ, ∀ x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x + b < c ↔ x₁ < x ∧ x < x₂) → |x₁ - x₂| = 4 → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2760_276057


namespace NUMINAMATH_CALUDE_squats_on_fourth_day_l2760_276041

/-- Calculates the number of squats on a given day, given the initial number of squats and daily increase. -/
def squats_on_day (initial_squats : ℕ) (daily_increase : ℕ) (day : ℕ) : ℕ :=
  initial_squats + (day - 1) * daily_increase

/-- Theorem stating that given an initial number of 30 squats and a daily increase of 5,
    the number of squats on the fourth day will be 45. -/
theorem squats_on_fourth_day :
  squats_on_day 30 5 4 = 45 := by
  sorry

end NUMINAMATH_CALUDE_squats_on_fourth_day_l2760_276041


namespace NUMINAMATH_CALUDE_equal_slopes_imply_equal_angles_l2760_276065

/-- Theorem: For two lines with inclination angles in [0, π) and equal slopes, their inclination angles are equal. -/
theorem equal_slopes_imply_equal_angles (α₁ α₂ : Real) (k₁ k₂ : Real) :
  0 ≤ α₁ ∧ α₁ < π →
  0 ≤ α₂ ∧ α₂ < π →
  k₁ = Real.tan α₁ →
  k₂ = Real.tan α₂ →
  k₁ = k₂ →
  α₁ = α₂ := by
  sorry

end NUMINAMATH_CALUDE_equal_slopes_imply_equal_angles_l2760_276065


namespace NUMINAMATH_CALUDE_product_evaluation_l2760_276075

theorem product_evaluation : (6 - 5) * (6 - 4) * (6 - 3) * (6 - 2) * (6 - 1) * 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2760_276075


namespace NUMINAMATH_CALUDE_inequality_solution_l2760_276090

theorem inequality_solution (x : ℝ) : 
  1 / (x^2 + 4) > 5 / x + 21 / 10 ↔ -2 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2760_276090


namespace NUMINAMATH_CALUDE_sqrt_of_2_4_3_6_5_2_l2760_276044

theorem sqrt_of_2_4_3_6_5_2 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_2_4_3_6_5_2_l2760_276044


namespace NUMINAMATH_CALUDE_M_mod_1000_l2760_276061

def M : ℕ := Nat.choose 14 8

theorem M_mod_1000 : M % 1000 = 3 := by
  sorry

end NUMINAMATH_CALUDE_M_mod_1000_l2760_276061


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l2760_276032

theorem degree_to_radian_conversion (π : ℝ) (h : π > 0) :
  let degree_to_radian (d : ℝ) := d * (π / 180)
  degree_to_radian 15 = π / 12 := by
sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l2760_276032


namespace NUMINAMATH_CALUDE_binomial_coefficient_39_5_l2760_276099

theorem binomial_coefficient_39_5 : 
  let n : ℕ := 39
  let binomial := n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / (2 * 3 * 4 * 5)
  binomial = 575757 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_39_5_l2760_276099


namespace NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l2760_276018

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem second_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 1/5)
  (h_a3 : a 3 = 5) :
  a 2 = 1 ∨ a 2 = -1 :=
sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l2760_276018


namespace NUMINAMATH_CALUDE_marathon_run_solution_l2760_276029

/-- Represents the marathon run problem -/
def marathon_run (x : ℝ) : Prop :=
  let total_distance : ℝ := 95
  let total_time : ℝ := 15
  let speed1 : ℝ := 8
  let speed2 : ℝ := 6
  let speed3 : ℝ := 5
  (speed1 * x + speed2 * x + speed3 * (total_time - 2 * x) = total_distance) ∧
  (x ≥ 0) ∧ (x ≤ total_time / 2)

/-- Proves that the only solution to the marathon run problem is 5 hours at each speed -/
theorem marathon_run_solution :
  ∃! x : ℝ, marathon_run x ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_marathon_run_solution_l2760_276029


namespace NUMINAMATH_CALUDE_find_y_l2760_276053

theorem find_y (x : ℝ) (y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 24) : y = 96 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2760_276053


namespace NUMINAMATH_CALUDE_arrangements_with_pair_eq_10080_l2760_276012

/-- The number of ways to arrange n people in a line. -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 8 people in a line where two specific individuals must always stand next to each other. -/
def arrangements_with_pair : ℕ :=
  factorial 7 * factorial 2

theorem arrangements_with_pair_eq_10080 :
  arrangements_with_pair = 10080 := by sorry

end NUMINAMATH_CALUDE_arrangements_with_pair_eq_10080_l2760_276012


namespace NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angle_has_20_sides_l2760_276020

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  exterior_angle : ℝ
  regular : exterior_angle * (sides : ℝ) = 360

-- Theorem statement
theorem regular_polygon_with_18_degree_exterior_angle_has_20_sides :
  ∀ p : RegularPolygon, p.exterior_angle = 18 → p.sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_18_degree_exterior_angle_has_20_sides_l2760_276020


namespace NUMINAMATH_CALUDE_maria_name_rearrangement_time_l2760_276021

/-- The time in hours to write all rearrangements of a name -/
def time_to_write_rearrangements (name_length : ℕ) (repeated_letters : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  let total_rearrangements := (Nat.factorial name_length) / (Nat.factorial repeated_letters)
  let minutes_needed := total_rearrangements / rearrangements_per_minute
  (minutes_needed : ℚ) / 60

/-- Theorem stating that the time to write all rearrangements of Maria's name is 0.125 hours -/
theorem maria_name_rearrangement_time :
  time_to_write_rearrangements 5 1 8 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_maria_name_rearrangement_time_l2760_276021


namespace NUMINAMATH_CALUDE_expression_equals_zero_l2760_276002

theorem expression_equals_zero : (-3)^3 + (-3)^2 * 3^1 + 3^2 * (-3)^1 + 3^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l2760_276002


namespace NUMINAMATH_CALUDE_double_discount_price_l2760_276087

/-- Proves that if a price P is discounted twice by 25% and the final price is $15, then the original price P is equal to $26.67 -/
theorem double_discount_price (P : ℝ) : 
  (0.75 * (0.75 * P) = 15) → P = 26.67 := by
sorry

end NUMINAMATH_CALUDE_double_discount_price_l2760_276087


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2760_276091

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Collinearity of three points in 3D space -/
def collinear (p q r : Point3D) : Prop :=
  ∃ t s : ℝ, t ≠ s ∧ 
    q.x = (1 - t) * p.x + t * r.x ∧
    q.y = (1 - t) * p.y + t * r.y ∧
    q.z = (1 - t) * p.z + t * r.z ∧
    q.x = (1 - s) * p.x + s * r.x ∧
    q.y = (1 - s) * p.y + s * r.y ∧
    q.z = (1 - s) * p.z + s * r.z

theorem collinear_points_sum (x y z : ℝ) :
  collinear (Point3D.mk x 1 z) (Point3D.mk 2 y z) (Point3D.mk x y 3) →
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2760_276091


namespace NUMINAMATH_CALUDE_circle_radius_l2760_276050

/-- The radius of a circle defined by the equation x^2 + 2x + y^2 = 0 is 1 -/
theorem circle_radius (x y : ℝ) : x^2 + 2*x + y^2 = 0 → ∃ (c : ℝ × ℝ), (x - c.1)^2 + (y - c.2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2760_276050


namespace NUMINAMATH_CALUDE_vector_sum_equals_one_five_l2760_276092

/-- Given vectors a and b in R², prove that their sum is (1, 5) -/
theorem vector_sum_equals_one_five :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![-1, 2]
  (a + b) = ![1, 5] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_equals_one_five_l2760_276092


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l2760_276079

/-- Calculates the total cost of plastering a rectangular tank -/
def plasteringCost (length width depth rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

theorem tank_plastering_cost :
  let length : ℝ := 25
  let width : ℝ := 12
  let depth : ℝ := 6
  let rate : ℝ := 0.55  -- 55 paise converted to rupees
  plasteringCost length width depth rate = 409.2 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l2760_276079


namespace NUMINAMATH_CALUDE_pyramid_hemisphere_theorem_l2760_276067

/-- A triangular pyramid with an equilateral triangular base -/
structure TriangularPyramid where
  /-- The height of the pyramid -/
  height : ℝ
  /-- The side length of the equilateral triangular base -/
  base_side : ℝ

/-- A hemisphere placed inside the pyramid -/
structure Hemisphere where
  /-- The radius of the hemisphere -/
  radius : ℝ

/-- Predicate to check if the hemisphere is properly placed in the pyramid -/
def is_properly_placed (p : TriangularPyramid) (h : Hemisphere) : Prop :=
  h.radius = 3 ∧ 
  p.height = 9 ∧
  -- The hemisphere is tangent to all three faces and rests on the base
  -- (This condition is assumed to be true when the predicate is true)
  True

/-- The main theorem -/
theorem pyramid_hemisphere_theorem (p : TriangularPyramid) (h : Hemisphere) :
  is_properly_placed p h → p.base_side = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_hemisphere_theorem_l2760_276067


namespace NUMINAMATH_CALUDE_koala_fiber_intake_l2760_276038

/-- The absorption rate of fiber for koalas -/
def absorption_rate : ℝ := 0.35

/-- The amount of fiber absorbed on the first day -/
def fiber_absorbed_day1 : ℝ := 14.7

/-- The amount of fiber absorbed on the second day -/
def fiber_absorbed_day2 : ℝ := 9.8

/-- Theorem: The total amount of fiber eaten by the koala over two days is 70 ounces -/
theorem koala_fiber_intake :
  let fiber_eaten_day1 := fiber_absorbed_day1 / absorption_rate
  let fiber_eaten_day2 := fiber_absorbed_day2 / absorption_rate
  fiber_eaten_day1 + fiber_eaten_day2 = 70 := by sorry

end NUMINAMATH_CALUDE_koala_fiber_intake_l2760_276038


namespace NUMINAMATH_CALUDE_two_thirds_to_tenth_bounds_l2760_276030

theorem two_thirds_to_tenth_bounds : 1/100 < (2/3)^10 ∧ (2/3)^10 < 2/100 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_to_tenth_bounds_l2760_276030


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l2760_276055

theorem price_decrease_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 775)
  (h2 : new_price = 620) :
  (original_price - new_price) / original_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l2760_276055


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2760_276034

-- Problem 1
theorem factorization_problem_1 (a b : ℝ) :
  12 * a^3 * b - 12 * a^2 * b + 3 * a * b = 3 * a * b * (2*a - 1)^2 := by
  sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  9 - x^2 + 2*x*y - y^2 = (3 + x - y) * (3 - x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2760_276034


namespace NUMINAMATH_CALUDE_obtuse_triangle_area_bound_l2760_276086

theorem obtuse_triangle_area_bound (a b c : ℝ) (h_obtuse : 0 < a ∧ 0 < b ∧ 0 < c ∧ c^2 > a^2 + b^2) 
  (h_longest : c = 4) (h_shortest : a = 2) : 
  (1/2 * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)) ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_obtuse_triangle_area_bound_l2760_276086


namespace NUMINAMATH_CALUDE_negation_equivalence_l2760_276010

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ ≥ 0 ∧ 2 * x₀ = 3) ↔ (∀ x : ℝ, x ≥ 0 → 2 * x ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2760_276010


namespace NUMINAMATH_CALUDE_radish_basket_difference_l2760_276047

theorem radish_basket_difference (total : ℕ) (first_basket : ℕ) : total = 88 → first_basket = 37 → total - first_basket - first_basket = 14 := by
  sorry

end NUMINAMATH_CALUDE_radish_basket_difference_l2760_276047


namespace NUMINAMATH_CALUDE_only_rational_root_l2760_276006

def polynomial (x : ℚ) : ℚ := 6 * x^4 - 5 * x^3 - 17 * x^2 + 7 * x + 3

theorem only_rational_root :
  ∀ x : ℚ, polynomial x = 0 ↔ x = 1/2 := by sorry

end NUMINAMATH_CALUDE_only_rational_root_l2760_276006


namespace NUMINAMATH_CALUDE_triangle_inequality_l2760_276097

theorem triangle_inequality (a b c : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) 
  (h4 : a * b + b * c + c * a = 18) : 
  1 / (a - 1)^3 + 1 / (b - 1)^3 + 1 / (c - 1)^3 > 1 / (a + b + c - 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2760_276097


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2760_276089

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x < 0} = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2760_276089


namespace NUMINAMATH_CALUDE_min_draws_for_pair_of_each_color_l2760_276003

/-- Represents the number of items of a given color -/
structure ColorCount where
  count : Nat

/-- Represents the box containing hats and gloves -/
structure Box where
  red : ColorCount
  green : ColorCount
  orange : ColorCount

/-- Calculates the minimum number of draws required for a given color -/
def minDrawsForColor (c : ColorCount) : Nat :=
  c.count + 1

/-- Calculates the total minimum draws required for all colors -/
def totalMinDraws (box : Box) : Nat :=
  minDrawsForColor box.red + minDrawsForColor box.green + minDrawsForColor box.orange

/-- The main theorem to be proved -/
theorem min_draws_for_pair_of_each_color (box : Box) 
  (h_red : box.red.count = 41)
  (h_green : box.green.count = 23)
  (h_orange : box.orange.count = 11) :
  totalMinDraws box = 78 := by
  sorry

#eval totalMinDraws { red := { count := 41 }, green := { count := 23 }, orange := { count := 11 } }

end NUMINAMATH_CALUDE_min_draws_for_pair_of_each_color_l2760_276003


namespace NUMINAMATH_CALUDE_tyler_meal_combinations_l2760_276077

/-- The number of meat options available -/
def num_meats : ℕ := 4

/-- The number of vegetable options available -/
def num_vegetables : ℕ := 4

/-- The number of dessert options available -/
def num_desserts : ℕ := 5

/-- The number of bread options available -/
def num_breads : ℕ := 3

/-- The number of vegetables Tyler must choose -/
def vegetables_to_choose : ℕ := 2

/-- The number of breads Tyler must choose -/
def breads_to_choose : ℕ := 2

/-- Calculates the number of ways to choose k items from n items without replacement and without order -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of ways to choose k items from n items without replacement but with order -/
def permute (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The total number of meal combinations Tyler can choose -/
def total_combinations : ℕ := 
  num_meats * choose num_vegetables vegetables_to_choose * num_desserts * permute num_breads breads_to_choose

theorem tyler_meal_combinations : total_combinations = 720 := by
  sorry

end NUMINAMATH_CALUDE_tyler_meal_combinations_l2760_276077


namespace NUMINAMATH_CALUDE_x_axis_intersection_correct_y_coord_correct_l2760_276035

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 5 * y - 2 * x = 10

/-- The point where the line intersects the x-axis -/
def x_axis_intersection : ℝ × ℝ := (-5, 0)

/-- The y-coordinate when x = -5 -/
def y_coord_at_neg_five : ℝ := 0

/-- Theorem stating that x_axis_intersection is on the line and has y-coordinate 0 -/
theorem x_axis_intersection_correct :
  line_equation x_axis_intersection.1 x_axis_intersection.2 ∧ x_axis_intersection.2 = 0 := by sorry

/-- Theorem stating that when x = -5, the y-coordinate is y_coord_at_neg_five -/
theorem y_coord_correct : line_equation (-5) y_coord_at_neg_five := by sorry

end NUMINAMATH_CALUDE_x_axis_intersection_correct_y_coord_correct_l2760_276035


namespace NUMINAMATH_CALUDE_last_digit_tower3_5_l2760_276073

/-- The last digit of a number n -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Exponentiation modulo m -/
def powMod (base exp m : ℕ) : ℕ :=
  (base ^ exp) % m

/-- The tower of powers of 3 with height 5 -/
def tower3_5 : ℕ := 3^(3^(3^(3^3)))

/-- The last digit of the tower of powers of 3 with height 5 is 7 -/
theorem last_digit_tower3_5 : lastDigit tower3_5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_tower3_5_l2760_276073


namespace NUMINAMATH_CALUDE_identity_proof_l2760_276008

theorem identity_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hsum : a + b + c ≠ 0) (h : 1/a + 1/b + 1/c = 1/(a+b+c)) : 
  1/a^1999 + 1/b^1999 + 1/c^1999 = 1/(a^1999 + b^1999 + c^1999) := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l2760_276008


namespace NUMINAMATH_CALUDE_same_terminal_side_negative_420_and_660_l2760_276017

-- Define a function to represent angles with the same terminal side
def same_terminal_side (θ : ℝ) (φ : ℝ) : Prop :=
  ∃ n : ℤ, φ = θ + n * 360

-- Theorem statement
theorem same_terminal_side_negative_420_and_660 :
  same_terminal_side (-420) 660 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_negative_420_and_660_l2760_276017


namespace NUMINAMATH_CALUDE_phone_call_cost_per_minute_l2760_276052

/-- Proves that the cost per minute of each phone call is $0.05 given the specified conditions --/
theorem phone_call_cost_per_minute 
  (call_duration : ℝ) 
  (customers_per_week : ℕ) 
  (monthly_bill : ℝ) 
  (weeks_per_month : ℕ) : 
  call_duration = 1 →
  customers_per_week = 50 →
  monthly_bill = 600 →
  weeks_per_month = 4 →
  (monthly_bill / (customers_per_week * weeks_per_month * call_duration * 60)) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_phone_call_cost_per_minute_l2760_276052


namespace NUMINAMATH_CALUDE_no_bounded_ratio_interval_l2760_276080

theorem no_bounded_ratio_interval (a : ℝ) (ha : a > 0) :
  ¬∃ (b c : ℝ) (hbc : b < c),
    ∀ (x y : ℝ) (hx : b < x ∧ x < c) (hy : b < y ∧ y < c) (hxy : x ≠ y),
      |((x + y) / (x - y))| ≤ a :=
sorry

end NUMINAMATH_CALUDE_no_bounded_ratio_interval_l2760_276080


namespace NUMINAMATH_CALUDE_cubic_root_function_l2760_276027

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 64, 
    prove that y = 2√3 when x = 8 -/
theorem cubic_root_function (k : ℝ) :
  (∀ x, x > 0 → k * x^(1/3) = 4 * Real.sqrt 3 → x = 64) →
  k * 8^(1/3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_function_l2760_276027


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2760_276011

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2760_276011


namespace NUMINAMATH_CALUDE_circumcircle_equation_incircle_equation_l2760_276046

-- Define the Triangle type
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the Circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def Triangle1 : Triangle := { A := (5, 1), B := (7, -3), C := (2, -8) }
def Triangle2 : Triangle := { A := (0, 0), B := (5, 0), C := (0, 12) }

def CircumcircleEquation (t : Triangle) (c : Circle) : Prop :=
  ∀ (x y : ℝ), (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔
    (x = t.A.1 ∧ y = t.A.2) ∨ (x = t.B.1 ∧ y = t.B.2) ∨ (x = t.C.1 ∧ y = t.C.2)

def IncircleEquation (t : Triangle) (c : Circle) : Prop :=
  ∀ (x y : ℝ), (x - c.center.1)^2 + (y - c.center.2)^2 ≤ c.radius^2 ↔
    (y ≥ 0 ∧ y ≤ 12 ∧ x ≥ 0 ∧ 5*y + 12*x ≤ 60)

theorem circumcircle_equation (t : Triangle) (h : t = Triangle1) :
  CircumcircleEquation t { center := (2, -3), radius := 5 } := by sorry

theorem incircle_equation (t : Triangle) (h : t = Triangle2) :
  IncircleEquation t { center := (2, 2), radius := 2 } := by sorry

end NUMINAMATH_CALUDE_circumcircle_equation_incircle_equation_l2760_276046


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l2760_276036

/-- The parabola equation y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line equation y = k(x-2) + 1 passing through P(2,1) -/
def line (k x y : ℝ) : Prop := y = k*(x-2) + 1

/-- The number of common points between the line and the parabola -/
inductive CommonPoints
  | one
  | two
  | none

/-- Theorem stating the conditions for the number of common points -/
theorem line_parabola_intersection (k : ℝ) :
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2) ↔ k = 0 ∧
  ¬(∃ p q : ℝ × ℝ, p ≠ q ∧ parabola p.1 p.2 ∧ parabola q.1 q.2 ∧ line k p.1 p.2 ∧ line k q.1 q.2) ∧
  ¬(∀ p : ℝ × ℝ, parabola p.1 p.2 → ¬line k p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l2760_276036


namespace NUMINAMATH_CALUDE_self_inverse_fourth_power_congruence_l2760_276059

theorem self_inverse_fourth_power_congruence (n : ℕ+) (a : ℤ) 
  (h : a * a ≡ 1 [ZMOD n]) : 
  a^4 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_self_inverse_fourth_power_congruence_l2760_276059


namespace NUMINAMATH_CALUDE_opposite_sides_condition_l2760_276023

/-- 
Given a real number m, if the points (1, 2) and (1, 1) are on opposite sides of the line y - 3x - m = 0, 
then -2 < m < -1.
-/
theorem opposite_sides_condition (m : ℝ) : 
  (2 - 3 * 1 - m) * (1 - 3 * 1 - m) < 0 → -2 < m ∧ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_condition_l2760_276023


namespace NUMINAMATH_CALUDE_pretzels_theorem_l2760_276056

def pretzels_problem (initial_pretzels john_pretzels marcus_pretzels : ℕ) : Prop :=
  let alan_pretzels := initial_pretzels - john_pretzels - marcus_pretzels
  john_pretzels - alan_pretzels = 1

theorem pretzels_theorem :
  ∀ (initial_pretzels john_pretzels marcus_pretzels : ℕ),
    initial_pretzels = 95 →
    john_pretzels = 28 →
    marcus_pretzels = 40 →
    marcus_pretzels = john_pretzels + 12 →
    pretzels_problem initial_pretzels john_pretzels marcus_pretzels :=
by
  sorry

#check pretzels_theorem

end NUMINAMATH_CALUDE_pretzels_theorem_l2760_276056


namespace NUMINAMATH_CALUDE_spotted_cats_ratio_l2760_276025

/-- Proves that the ratio of spotted cats to total cats is 1:3 -/
theorem spotted_cats_ratio (total_cats : ℕ) (spotted_fluffy : ℕ) :
  total_cats = 120 →
  spotted_fluffy = 10 →
  (4 : ℚ) * spotted_fluffy = total_spotted →
  (total_spotted : ℚ) / total_cats = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_spotted_cats_ratio_l2760_276025


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2760_276068

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2760_276068


namespace NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l2760_276037

/-- Given two vectors a and b in ℝ², where a = (2,1) and b = (1-y, 2+y),
    and a is perpendicular to b, prove that |a - b| = 5√2. -/
theorem perpendicular_vectors_difference_magnitude :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ → ℝ × ℝ := λ y ↦ (1 - y, 2 + y)
  ∃ y : ℝ, (a.1 * (b y).1 + a.2 * (b y).2 = 0) →
    Real.sqrt ((a.1 - (b y).1)^2 + (a.2 - (b y).2)^2) = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l2760_276037


namespace NUMINAMATH_CALUDE_sequence_prime_divisor_l2760_276064

/-- Given a positive integer n > 1, prove that for all k ≥ 1, the k-th term of the sequence
    a_k = n^(n^(k-1)) - 1 has a prime divisor that does not divide any of the previous terms. -/
theorem sequence_prime_divisor (n : ℕ) (hn : n > 1) :
  ∀ k : ℕ, k ≥ 1 →
    ∃ p : ℕ, Nat.Prime p ∧ p ∣ (n^(n^(k-1)) - 1) ∧
      ∀ i : ℕ, 1 ≤ i ∧ i < k → ¬(p ∣ (n^(n^(i-1)) - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_prime_divisor_l2760_276064


namespace NUMINAMATH_CALUDE_area_of_inner_triangle_l2760_276096

/-- Given a triangle and points dividing its sides in a 1:2 ratio, 
    the area of the new triangle formed by these points is 1/9 of the original triangle's area. -/
theorem area_of_inner_triangle (T : ℝ) (h : T > 0) :
  ∃ (A : ℝ), A = T / 9 ∧ A > 0 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inner_triangle_l2760_276096


namespace NUMINAMATH_CALUDE_intersection_triangle_area_l2760_276009

/-- Given a regular tetrahedron with side length 2, cut by a plane parallel to one face
    at height 1 from the base, the area of the intersection triangle is (2√3 - 3) / 4 -/
theorem intersection_triangle_area (side_length : ℝ) (cut_height : ℝ) : 
  side_length = 2 → cut_height = 1 → 
  ∃ (area : ℝ), area = (2 * Real.sqrt 3 - 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_triangle_area_l2760_276009


namespace NUMINAMATH_CALUDE_cat_finishes_food_on_sunday_l2760_276043

/-- Represents days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the amount of food eaten by the cat -/
def cat_food_eaten (d : Day) : Rat :=
  match d with
  | Day.Monday => 5/6
  | Day.Tuesday => 10/6
  | Day.Wednesday => 15/6
  | Day.Thursday => 20/6
  | Day.Friday => 25/6
  | Day.Saturday => 30/6
  | Day.Sunday => 35/6

theorem cat_finishes_food_on_sunday :
  ∀ d : Day, cat_food_eaten d ≤ 9 ∧
  (d = Day.Sunday → cat_food_eaten d > 54/6) :=
by sorry

#check cat_finishes_food_on_sunday

end NUMINAMATH_CALUDE_cat_finishes_food_on_sunday_l2760_276043


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2760_276054

theorem divisibility_theorem (a b c : ℕ+) 
  (h1 : a ∣ b^5)
  (h2 : b ∣ c^5)
  (h3 : c ∣ a^5) :
  (a * b * c) ∣ (a + b + c)^31 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2760_276054


namespace NUMINAMATH_CALUDE_min_tiles_to_cover_l2760_276040

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Theorem stating the minimum number of tiles needed to cover the specified area -/
theorem min_tiles_to_cover (tileSize : Dimensions) (regionSize : Dimensions) (coveredSize : Dimensions) : 
  tileSize.length = 3 →
  tileSize.width = 4 →
  regionSize.length = feetToInches 3 →
  regionSize.width = feetToInches 6 →
  coveredSize.length = feetToInches 1 →
  coveredSize.width = feetToInches 1 →
  (area regionSize - area coveredSize) / area tileSize = 204 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_to_cover_l2760_276040


namespace NUMINAMATH_CALUDE_shortest_distance_to_start_l2760_276016

/-- Proof of the shortest distance between the third meeting point and the starting point on a circular track -/
theorem shortest_distance_to_start (track_length : ℝ) (time : ℝ) (speed_diff : ℝ) : 
  track_length = 400 →
  time = 8 * 60 →
  speed_diff = 0.1 →
  ∃ (speed_b : ℝ), 
    time * (speed_b + speed_b + speed_diff) = track_length * 3 ∧
    (time * speed_b) % track_length = 176 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_to_start_l2760_276016


namespace NUMINAMATH_CALUDE_one_third_percentage_l2760_276051

-- Define the given numbers
def total : ℚ := 1206
def divisor : ℚ := 3
def base : ℚ := 134

-- Define one-third of the total
def one_third : ℚ := total / divisor

-- Define the percentage calculation
def percentage : ℚ := (one_third / base) * 100

-- Theorem to prove
theorem one_third_percentage : percentage = 300 := by
  sorry

end NUMINAMATH_CALUDE_one_third_percentage_l2760_276051


namespace NUMINAMATH_CALUDE_total_money_for_76_members_l2760_276066

/-- Calculates the total money collected in rupees given the number of members in a group -/
def totalMoneyCollected (members : ℕ) : ℚ :=
  (members * members : ℕ) / 100

/-- Proves that for a group of 76 members, the total money collected is ₹57.76 -/
theorem total_money_for_76_members :
  totalMoneyCollected 76 = 57.76 := by
  sorry

end NUMINAMATH_CALUDE_total_money_for_76_members_l2760_276066


namespace NUMINAMATH_CALUDE_eugene_pencils_l2760_276048

/-- The number of pencils Eugene has after receiving more from Joyce -/
def total_pencils (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Eugene has 57 pencils in total -/
theorem eugene_pencils : total_pencils 51 6 = 57 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l2760_276048


namespace NUMINAMATH_CALUDE_circle_symmetry_l2760_276005

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 4*y + 19 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 5)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_circle x₁ y₁ →
    symmetric_circle x₂ y₂ →
    ∃ (x_mid y_mid : ℝ),
      symmetry_line x_mid y_mid ∧
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2760_276005


namespace NUMINAMATH_CALUDE_quadratic_roots_existence_l2760_276058

theorem quadratic_roots_existence (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    3 * x₁^2 - 2*(a+b+c)*x₁ + a*b + b*c + a*c = 0 ∧
    3 * x₂^2 - 2*(a+b+c)*x₂ + a*b + b*c + a*c = 0 ∧
    a < x₁ ∧ x₁ < b ∧ b < x₂ ∧ x₂ < c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_existence_l2760_276058


namespace NUMINAMATH_CALUDE_power_sum_equals_fourteen_l2760_276084

theorem power_sum_equals_fourteen : 2^3 + 2^2 + 2^1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_fourteen_l2760_276084


namespace NUMINAMATH_CALUDE_email_difference_l2760_276095

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 10

/-- The number of letters Jack received in the morning -/
def morning_letters : ℕ := 12

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of letters Jack received in the afternoon -/
def afternoon_letters : ℕ := 44

/-- The difference between the number of emails Jack received in the morning and afternoon -/
theorem email_difference : morning_emails - afternoon_emails = 7 := by
  sorry

end NUMINAMATH_CALUDE_email_difference_l2760_276095


namespace NUMINAMATH_CALUDE_quadratic_discriminant_positive_l2760_276031

theorem quadratic_discriminant_positive 
  (a b c : ℝ) 
  (h : (a + b + c) * c < 0) : 
  b^2 - 4*a*c > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_positive_l2760_276031


namespace NUMINAMATH_CALUDE_intersection_line_polar_equation_l2760_276062

/-- Given two circles in polar coordinates, find the polar equation of the line
    passing through their intersection points. -/
theorem intersection_line_polar_equation
  (O₁ : ℝ → ℝ → Prop) -- Circle O₁ in polar coordinates
  (O₂ : ℝ → ℝ → Prop) -- Circle O₂ in polar coordinates
  (h₁ : ∀ ρ θ, O₁ ρ θ ↔ ρ = 2)
  (h₂ : ∀ ρ θ, O₂ ρ θ ↔ ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos (θ - π/4) = 2) :
  ∀ ρ θ, (∃ θ₁ θ₂, O₁ ρ θ₁ ∧ O₂ ρ θ₂) →
    ρ * Real.sin (θ + π/4) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_polar_equation_l2760_276062
