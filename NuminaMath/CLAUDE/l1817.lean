import Mathlib

namespace NUMINAMATH_CALUDE_only_elevator_is_pure_translation_l1817_181705

/-- Represents a physical phenomenon --/
inductive Phenomenon
  | RollingSoccerBall
  | RotatingFanBlades
  | ElevatorGoingUp
  | MovingCarRearWheel

/-- Defines whether a phenomenon exhibits pure translation --/
def isPureTranslation (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.ElevatorGoingUp => True
  | _ => False

/-- The rolling soccer ball involves both rotation and translation --/
axiom rolling_soccer_ball_not_pure_translation :
  ¬ isPureTranslation Phenomenon.RollingSoccerBall

/-- Rotating fan blades involve rotation around a central axis --/
axiom rotating_fan_blades_not_pure_translation :
  ¬ isPureTranslation Phenomenon.RotatingFanBlades

/-- An elevator going up moves from one level to another without rotating --/
axiom elevator_going_up_is_pure_translation :
  isPureTranslation Phenomenon.ElevatorGoingUp

/-- A moving car rear wheel primarily exhibits rotation --/
axiom moving_car_rear_wheel_not_pure_translation :
  ¬ isPureTranslation Phenomenon.MovingCarRearWheel

/-- Theorem: Only the elevator going up exhibits pure translation --/
theorem only_elevator_is_pure_translation :
  ∀ p : Phenomenon, isPureTranslation p ↔ p = Phenomenon.ElevatorGoingUp :=
by sorry


end NUMINAMATH_CALUDE_only_elevator_is_pure_translation_l1817_181705


namespace NUMINAMATH_CALUDE_commodity_price_increase_l1817_181755

/-- The annual price increase of commodity Y -/
def y : ℝ := sorry

/-- The year we're interested in -/
def target_year : ℝ := 1999.18

/-- The reference year -/
def reference_year : ℝ := 2001

/-- The price of commodity X in the reference year -/
def price_x_reference : ℝ := 5.20

/-- The price of commodity Y in the reference year -/
def price_y_reference : ℝ := 7.30

/-- The annual price increase of commodity X -/
def x_increase : ℝ := 0.45

/-- The price difference between X and Y in the target year -/
def price_difference : ℝ := 0.90

/-- The number of years between the target year and the reference year -/
def years_difference : ℝ := reference_year - target_year

theorem commodity_price_increase : 
  abs (y - 0.021) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_commodity_price_increase_l1817_181755


namespace NUMINAMATH_CALUDE_total_puppies_l1817_181773

/-- Given an initial number of puppies and a number of additional puppies,
    prove that the total number of puppies is equal to the sum of the initial number
    and the additional number. -/
theorem total_puppies (initial_puppies additional_puppies : ℝ) :
  initial_puppies + additional_puppies = initial_puppies + additional_puppies :=
by sorry

end NUMINAMATH_CALUDE_total_puppies_l1817_181773


namespace NUMINAMATH_CALUDE_percent_of_y_l1817_181781

theorem percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 5 + (3 * y) / 10) / y = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l1817_181781


namespace NUMINAMATH_CALUDE_linear_system_solution_l1817_181789

/-- Solution to a system of linear equations -/
theorem linear_system_solution (a b c h : ℝ) :
  let x := (h - b) * (h - c) / ((a - b) * (a - c))
  let y := (h - a) * (h - c) / ((b - a) * (b - c))
  let z := (h - a) * (h - b) / ((c - a) * (c - b))
  x + y + z = 1 ∧
  a * x + b * y + c * z = h ∧
  a^2 * x + b^2 * y + c^2 * z = h^2 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1817_181789


namespace NUMINAMATH_CALUDE_food_product_shelf_life_l1817_181787

/-- Represents the shelf life function of a food product -/
noncomputable def shelf_life (k b x : ℝ) : ℝ := Real.exp (k * x + b)

/-- Theorem stating the shelf life at 30°C and the maximum temperature for 80 hours shelf life -/
theorem food_product_shelf_life 
  (k b : ℝ) 
  (h1 : shelf_life k b 0 = 160) 
  (h2 : shelf_life k b 20 = 40) : 
  shelf_life k b 30 = 20 ∧ 
  ∀ x, shelf_life k b x ≥ 80 → x ≤ 10 := by
sorry


end NUMINAMATH_CALUDE_food_product_shelf_life_l1817_181787


namespace NUMINAMATH_CALUDE_inequality_with_cosine_condition_l1817_181767

theorem inequality_with_cosine_condition (α β : ℝ) 
  (h : Real.cos α * Real.cos β > 0) : 
  -(Real.tan (α/2))^2 ≤ (Real.tan ((β-α)/2)) / (Real.tan ((β+α)/2)) ∧
  (Real.tan ((β-α)/2)) / (Real.tan ((β+α)/2)) ≤ (Real.tan (β/2))^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_with_cosine_condition_l1817_181767


namespace NUMINAMATH_CALUDE_fraction_simplification_l1817_181756

theorem fraction_simplification :
  (1 / 3 + 1 / 4) / (2 / 5 - 1 / 6) = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1817_181756


namespace NUMINAMATH_CALUDE_positive_x_y_l1817_181724

theorem positive_x_y (x y : ℝ) (h1 : x - y < x) (h2 : x + y > y) : x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_x_y_l1817_181724


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_range_l1817_181715

theorem geometric_sequence_fourth_term_range 
  (a : ℕ → ℝ) 
  (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : 0 < a 1 ∧ a 1 < 1)
  (h_a2 : 1 < a 2 ∧ a 2 < 2)
  (h_a3 : 2 < a 3 ∧ a 3 < 4) :
  2 * Real.sqrt 2 < a 4 ∧ a 4 < 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_range_l1817_181715


namespace NUMINAMATH_CALUDE_books_remaining_l1817_181720

theorem books_remaining (initial_books : ℕ) (given_away : ℕ) (sold : ℕ) : 
  initial_books = 108 → given_away = 35 → sold = 11 → 
  initial_books - given_away - sold = 62 := by
sorry

end NUMINAMATH_CALUDE_books_remaining_l1817_181720


namespace NUMINAMATH_CALUDE_area_ratio_PQRV_ABCD_l1817_181714

-- Define the squares and points
variable (A B C D P Q R V : ℝ × ℝ)

-- Define the properties of the squares
def is_square (A B C D : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    ‖B - A‖ = s ∧ ‖C - B‖ = s ∧ ‖D - C‖ = s ∧ ‖A - D‖ = s ∧
    (B - A) • (C - B) = 0

-- Define that P is on side AB
def P_on_AB (A B P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = A + t • (B - A)

-- Define the condition AP = 3 * PB
def AP_eq_3PB (A B P : ℝ × ℝ) : Prop :=
  ‖P - A‖ = 3 * ‖B - P‖

-- Define the area of a square
def area (A B C D : ℝ × ℝ) : ℝ :=
  ‖B - A‖^2

-- Theorem statement
theorem area_ratio_PQRV_ABCD 
  (h1 : is_square A B C D)
  (h2 : is_square P Q R V)
  (h3 : P_on_AB A B P)
  (h4 : AP_eq_3PB A B P) :
  area P Q R V / area A B C D = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_PQRV_ABCD_l1817_181714


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1817_181736

/-- The total surface area of a cylinder with diameter 9 and height 15 is 175.5π -/
theorem cylinder_surface_area :
  let d : ℝ := 9  -- diameter
  let h : ℝ := 15 -- height
  let r : ℝ := d / 2 -- radius
  let base_area : ℝ := π * r^2
  let lateral_area : ℝ := 2 * π * r * h
  let total_area : ℝ := 2 * base_area + lateral_area
  total_area = 175.5 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1817_181736


namespace NUMINAMATH_CALUDE_atmosphere_depth_for_specific_peak_l1817_181712

/-- Represents a cone-shaped peak on an alien planet -/
structure ConePeak where
  height : ℝ
  atmosphereVolumeFraction : ℝ

/-- Calculates the depth of the atmosphere at the base of a cone-shaped peak -/
def atmosphereDepth (peak : ConePeak) : ℝ :=
  peak.height * (1 - (peak.atmosphereVolumeFraction)^(1/3))

/-- Theorem stating the depth of the atmosphere for a specific cone-shaped peak -/
theorem atmosphere_depth_for_specific_peak :
  let peak : ConePeak := { height := 5000, atmosphereVolumeFraction := 4/5 }
  atmosphereDepth peak = 340 := by
  sorry

end NUMINAMATH_CALUDE_atmosphere_depth_for_specific_peak_l1817_181712


namespace NUMINAMATH_CALUDE_second_quadrant_m_negative_l1817_181710

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The theorem stating that if a point P(m, 2) is in the second quadrant, then m < 0 -/
theorem second_quadrant_m_negative (m : ℝ) :
  SecondQuadrant ⟨m, 2⟩ → m < 0 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_m_negative_l1817_181710


namespace NUMINAMATH_CALUDE_min_distinct_values_l1817_181745

theorem min_distinct_values (n : ℕ) (mode_count : ℕ) (total_count : ℕ) :
  n = 2017 →
  mode_count = 11 →
  total_count = n →
  (∃ (distinct_values : ℕ), 
    distinct_values ≥ 202 ∧
    ∀ (m : ℕ), m < 202 → 
      ¬(∃ (list : List ℕ),
        list.length = total_count ∧
        (∃ (mode : ℕ), list.count mode = mode_count ∧
          ∀ (x : ℕ), x ≠ mode → list.count x < mode_count) ∧
        list.toFinset.card = m)) :=
sorry

end NUMINAMATH_CALUDE_min_distinct_values_l1817_181745


namespace NUMINAMATH_CALUDE_simplify_fraction_l1817_181782

theorem simplify_fraction (a : ℝ) (ha : a ≠ 0) :
  (a - 1) / a / (a - 1 / a) = 1 / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1817_181782


namespace NUMINAMATH_CALUDE_polly_lunch_time_l1817_181746

/-- Represents the cooking time for a week -/
structure CookingTime where
  breakfast : ℕ  -- Time spent on breakfast daily
  dinner_short : ℕ  -- Time spent on dinner for short days
  dinner_long : ℕ  -- Time spent on dinner for long days
  short_days : ℕ  -- Number of days with short dinner time
  total : ℕ  -- Total cooking time for the week

/-- Calculates the time spent on lunch given the cooking time for other meals -/
def lunch_time (c : CookingTime) : ℕ :=
  c.total - (7 * c.breakfast + c.short_days * c.dinner_short + (7 - c.short_days) * c.dinner_long)

/-- Theorem stating that Polly spends 35 minutes cooking lunch -/
theorem polly_lunch_time :
  ∃ (c : CookingTime),
    c.breakfast = 20 ∧
    c.dinner_short = 10 ∧
    c.dinner_long = 30 ∧
    c.short_days = 4 ∧
    c.total = 305 ∧
    lunch_time c = 35 := by
  sorry

end NUMINAMATH_CALUDE_polly_lunch_time_l1817_181746


namespace NUMINAMATH_CALUDE_fabric_cutting_l1817_181775

theorem fabric_cutting (initial_length : ℚ) (desired_length : ℚ) :
  initial_length = 2/3 →
  desired_length = 1/2 →
  initial_length - (initial_length / 4) = desired_length :=
by
  sorry

end NUMINAMATH_CALUDE_fabric_cutting_l1817_181775


namespace NUMINAMATH_CALUDE_non_negative_integer_solutions_count_solution_count_equals_10626_l1817_181702

theorem non_negative_integer_solutions_count : Nat :=
  let n : Nat := 20
  let k : Nat := 5
  (n + k - 1).choose (k - 1)

theorem solution_count_equals_10626 : non_negative_integer_solutions_count = 10626 := by
  sorry

end NUMINAMATH_CALUDE_non_negative_integer_solutions_count_solution_count_equals_10626_l1817_181702


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1817_181783

-- Define the quadratic equation
def has_real_roots (m : ℕ) : Prop := ∃ x : ℝ, x^2 + x - m = 0

-- Define the original proposition
def original_prop (m : ℕ) : Prop := m > 0 → has_real_roots m

-- Define the contrapositive
def contrapositive (m : ℕ) : Prop := ¬(has_real_roots m) → m ≤ 0

-- Theorem statement
theorem contrapositive_equivalence :
  ∀ m : ℕ, m > 0 → (original_prop m ↔ contrapositive m) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1817_181783


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l1817_181708

theorem intersection_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  A ∩ B = {3} → a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l1817_181708


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_zero_zero_in_range_l1817_181778

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ 2) :
  (1 / (1 - x) + 1) / ((x^2 - 4*x + 4) / (x^2 - 1)) = (x + 1) / (x - 2) :=
by sorry

-- Evaluation at x = 0
theorem evaluate_at_zero :
  (1 / (1 - 0) + 1) / ((0^2 - 4*0 + 4) / (0^2 - 1)) = -1/2 :=
by sorry

-- Range constraint
def in_range (x : ℝ) : Prop := -2 < x ∧ x < 3

-- Proof that 0 is in the range
theorem zero_in_range : in_range 0 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_zero_zero_in_range_l1817_181778


namespace NUMINAMATH_CALUDE_blood_type_distribution_l1817_181771

theorem blood_type_distribution (total : ℕ) (type_a : ℕ) (type_b : ℕ) : 
  (2 : ℚ) / 9 * total = type_a →
  (2 : ℚ) / 5 * total = type_b →
  type_a = 10 →
  type_b = 18 := by
sorry

end NUMINAMATH_CALUDE_blood_type_distribution_l1817_181771


namespace NUMINAMATH_CALUDE_stones_combine_l1817_181757

/-- Definition of similar sizes -/
def similar (a b : ℕ) : Prop := a ≤ b ∧ b ≤ 2 * a

/-- A step in the combining process -/
inductive CombineStep (n : ℕ)
  | combine (i j : Fin n) (h : i.val < j.val) (hsimilar : similar (Fin.val i) (Fin.val j)) : CombineStep n

/-- A sequence of combining steps -/
def CombineSequence (n : ℕ) := List (CombineStep n)

/-- The final state after combining -/
def FinalState (n : ℕ) : Prop := ∃ (seq : CombineSequence n), 
  (∀ i : Fin n, i.val = 1) → (∃ j : Fin n, j.val = n ∧ ∀ k : Fin n, k ≠ j → k.val = 0)

/-- The main theorem -/
theorem stones_combine (n : ℕ) : FinalState n := by sorry

end NUMINAMATH_CALUDE_stones_combine_l1817_181757


namespace NUMINAMATH_CALUDE_sum_of_powers_l1817_181737

theorem sum_of_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 5)
  (h4 : a^4 + b^4 = 7) :
  a^10 + b^10 = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1817_181737


namespace NUMINAMATH_CALUDE_bag_contains_sixty_balls_l1817_181774

/-- The number of white balls in the bag -/
def white_balls : ℕ := 22

/-- The number of green balls in the bag -/
def green_balls : ℕ := 10

/-- The number of yellow balls in the bag -/
def yellow_balls : ℕ := 7

/-- The number of red balls in the bag -/
def red_balls : ℕ := 15

/-- The number of purple balls in the bag -/
def purple_balls : ℕ := 6

/-- The probability of choosing a ball that is neither red nor purple -/
def prob_not_red_or_purple : ℚ := 65/100

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + green_balls + yellow_balls + red_balls + purple_balls

theorem bag_contains_sixty_balls : total_balls = 60 := by
  sorry

end NUMINAMATH_CALUDE_bag_contains_sixty_balls_l1817_181774


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1817_181719

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 1 → a - b < a^2 - b^2) ∧
  (∃ a b : ℝ, a - b < a^2 - b^2 ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1817_181719


namespace NUMINAMATH_CALUDE_modulus_of_z_l1817_181742

-- Define the complex number z
def z : ℂ := (1 - Complex.I) * (1 + Complex.I)^2 + 1

-- Theorem statement
theorem modulus_of_z : Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1817_181742


namespace NUMINAMATH_CALUDE_power_multiply_l1817_181740

theorem power_multiply (x : ℝ) : x^2 * x^3 = x^5 := by sorry

end NUMINAMATH_CALUDE_power_multiply_l1817_181740


namespace NUMINAMATH_CALUDE_cell_phone_plan_comparison_l1817_181752

/-- Cellular phone plan comparison -/
theorem cell_phone_plan_comparison (F : ℝ) : 
  (∀ (minutes : ℝ), 
    F + max (minutes - 500) 0 * 0.35 = 
    75 + max (minutes - 1000) 0 * 0.45 → minutes = 2500) →
  F = 50 := by
sorry

end NUMINAMATH_CALUDE_cell_phone_plan_comparison_l1817_181752


namespace NUMINAMATH_CALUDE_quadratic_root_product_l1817_181734

theorem quadratic_root_product (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 : ℂ) + Complex.I ∈ {z : ℂ | z ^ 2 + p * z + q = 0} →
  p * q = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_product_l1817_181734


namespace NUMINAMATH_CALUDE_total_spent_on_toys_l1817_181727

def other_toys_cost : ℕ := 1000
def lightsaber_cost : ℕ := 2 * other_toys_cost

theorem total_spent_on_toys : other_toys_cost + lightsaber_cost = 3000 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_toys_l1817_181727


namespace NUMINAMATH_CALUDE_tangent_line_problem_l1817_181786

theorem tangent_line_problem (f : ℝ → ℝ) (h : ∀ x y, x = 2 ∧ f x = y → 2*x + y - 3 = 0) :
  f 2 + deriv f 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l1817_181786


namespace NUMINAMATH_CALUDE_parabola_and_line_equations_l1817_181762

-- Define the parabola E
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the focus of the parabola
def focus : Point := ⟨1, 0⟩

-- Define the midpoint M
def M : Point := ⟨2, 1⟩

-- Define the property of A and B being on the parabola E
def on_parabola (E : Parabola) (p : Point) : Prop :=
  E.equation p.x p.y

-- Define the property of M being the midpoint of AB
def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- Theorem statement
theorem parabola_and_line_equations 
  (E : Parabola) (A B : Point) 
  (h1 : on_parabola E A) 
  (h2 : on_parabola E B) 
  (h3 : A ≠ B) 
  (h4 : is_midpoint M A B) :
  (∀ (x y : ℝ), E.equation x y ↔ y^2 = 4*x) ∧ 
  (∀ (x y : ℝ), (y - M.y = 2*(x - M.x)) ↔ (2*x - y - 3 = 0)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_line_equations_l1817_181762


namespace NUMINAMATH_CALUDE_evaluate_expression_l1817_181706

theorem evaluate_expression : 8^6 * 27^6 * 8^15 * 27^15 = 216^21 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1817_181706


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1817_181753

/-- Given a parabola y = ax² - a (a ≠ 0) intersecting a line y = kx at points
    with sum of x-coordinates less than 0, prove that the line y = ax + k
    passes through the first and fourth quadrants. -/
theorem parabola_line_intersection (a k : ℝ) (ha : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, a * x₁^2 - a = k * x₁ ∧
               a * x₂^2 - a = k * x₂ ∧
               x₁ + x₂ < 0) →
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = a * x + k) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ y = a * x + k) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1817_181753


namespace NUMINAMATH_CALUDE_cos_45_degrees_l1817_181729

theorem cos_45_degrees : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_cos_45_degrees_l1817_181729


namespace NUMINAMATH_CALUDE_roof_area_l1817_181700

theorem roof_area (width length : ℝ) : 
  width > 0 → 
  length > 0 → 
  length = 4 * width → 
  length - width = 39 → 
  width * length = 676 := by
sorry

end NUMINAMATH_CALUDE_roof_area_l1817_181700


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l1817_181722

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, 2 * x^2 + 2 * x * y + y^2 - 2 * x + 2 * y + 4 ≥ -1 ∧
  ∃ x y : ℝ, 2 * x^2 + 2 * x * y + y^2 - 2 * x + 2 * y + 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l1817_181722


namespace NUMINAMATH_CALUDE_breath_holding_difference_l1817_181751

/-- 
Given that:
- Kelly held her breath for 3 minutes
- Brittany held her breath for 20 seconds less than Kelly
- Buffy held her breath for 120 seconds

Prove that Buffy held her breath for 40 seconds less than Brittany
-/
theorem breath_holding_difference : 
  let kelly_time := 3 * 60 -- Kelly's time in seconds
  let brittany_time := kelly_time - 20 -- Brittany's time in seconds
  let buffy_time := 120 -- Buffy's time in seconds
  brittany_time - buffy_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_breath_holding_difference_l1817_181751


namespace NUMINAMATH_CALUDE_coplanar_condition_l1817_181793

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define the origin and points
variable (O A B C D : V)

-- Define the scalar m
variable (m : ℝ)

-- Define the condition for coplanarity
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), (A - D) = a • (B - D) + b • (C - D) + c • (0 : V)

-- State the theorem
theorem coplanar_condition (h : ∀ A B C D : V, 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + m • (D - O) = 0 → coplanar A B C D) : 
  m = -7 := by sorry

end NUMINAMATH_CALUDE_coplanar_condition_l1817_181793


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l1817_181764

theorem quadratic_inequality_always_true (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + (a + 1) * x + 1 ≥ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l1817_181764


namespace NUMINAMATH_CALUDE_firing_time_per_minute_l1817_181704

/-- Calculates the time spent firing per minute given the firing interval and duration -/
def timeSpentFiring (secondsPerMinute : ℕ) (firingInterval : ℕ) (fireDuration : ℕ) : ℕ :=
  (secondsPerMinute / firingInterval) * fireDuration

/-- Proves that given the specified conditions, the time spent firing per minute is 20 seconds -/
theorem firing_time_per_minute :
  timeSpentFiring 60 15 5 = 20 := by sorry

end NUMINAMATH_CALUDE_firing_time_per_minute_l1817_181704


namespace NUMINAMATH_CALUDE_fly_path_distance_l1817_181707

theorem fly_path_distance (radius : ℝ) (third_segment : ℝ) :
  radius = 60 ∧ third_segment = 90 →
  ∃ (second_segment : ℝ),
    second_segment^2 + third_segment^2 = (2 * radius)^2 ∧
    (2 * radius) + third_segment + second_segment = 120 + 90 + 30 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_fly_path_distance_l1817_181707


namespace NUMINAMATH_CALUDE_volume_of_inscribed_cube_l1817_181797

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem volume_of_inscribed_cube (outer_cube_edge : ℝ) (h : outer_cube_edge = 16) :
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_edge : ℝ := sphere_diameter / Real.sqrt 3
  let inner_cube_volume : ℝ := inner_cube_edge ^ 3
  inner_cube_volume = 12288 * Real.sqrt 3 / 27 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_inscribed_cube_l1817_181797


namespace NUMINAMATH_CALUDE_circle_center_locus_l1817_181747

/-- Given a circle passing through A(0,a) with a chord of length 2a on the x-axis,
    prove that the locus of its center C(x,y) satisfies x^2 = 2ay -/
theorem circle_center_locus (a : ℝ) (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    (x^2 + (y - a)^2 = r^2) ∧  -- Circle passes through A(0,a)
    (y^2 + a^2 = r^2))         -- Chord on x-axis has length 2a
  → x^2 = 2*a*y := by sorry

end NUMINAMATH_CALUDE_circle_center_locus_l1817_181747


namespace NUMINAMATH_CALUDE_original_number_is_192_l1817_181765

theorem original_number_is_192 (N : ℚ) : 
  (((N / 8 + 8) - 30) * 6) = 12 → N = 192 := by
sorry

end NUMINAMATH_CALUDE_original_number_is_192_l1817_181765


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l1817_181703

/-- A quadratic polynomial satisfying specific conditions -/
def q (x : ℝ) : ℝ := -x^2 - 6*x + 27

/-- Theorem stating that q satisfies the required conditions -/
theorem q_satisfies_conditions :
  q (-9) = 0 ∧ q 3 = 0 ∧ q 6 = -45 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l1817_181703


namespace NUMINAMATH_CALUDE_organization_growth_l1817_181791

def population_growth (initial : ℕ) (years : ℕ) : ℕ :=
  match years with
  | 0 => initial
  | n + 1 => 3 * (population_growth initial n - 5) + 5

theorem organization_growth :
  population_growth 20 6 = 10895 := by
  sorry

end NUMINAMATH_CALUDE_organization_growth_l1817_181791


namespace NUMINAMATH_CALUDE_fish_eater_birds_count_l1817_181730

theorem fish_eater_birds_count (day1 day2 day3 : ℕ) : 
  day2 = 2 * day1 →
  day3 = day2 - 200 →
  day1 + day2 + day3 = 1300 →
  day1 = 300 := by
sorry

end NUMINAMATH_CALUDE_fish_eater_birds_count_l1817_181730


namespace NUMINAMATH_CALUDE_some_mythical_creatures_are_magical_beings_l1817_181784

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Dragon : U → Prop)
variable (MythicalCreature : U → Prop)
variable (MagicalBeing : U → Prop)

-- State the theorem
theorem some_mythical_creatures_are_magical_beings
  (h1 : ∀ x, Dragon x → MythicalCreature x)
  (h2 : ∃ x, MagicalBeing x ∧ Dragon x) :
  ∃ x, MythicalCreature x ∧ MagicalBeing x :=
by
  sorry


end NUMINAMATH_CALUDE_some_mythical_creatures_are_magical_beings_l1817_181784


namespace NUMINAMATH_CALUDE_third_day_temperature_l1817_181759

/-- Given three temperatures with a known average and two known values, 
    calculate the third temperature. -/
theorem third_day_temperature 
  (avg : ℚ) 
  (temp1 temp2 : ℚ) 
  (h_avg : avg = -7)
  (h_temp1 : temp1 = -14)
  (h_temp2 : temp2 = -8)
  (h_sum : 3 * avg = temp1 + temp2 + temp3) :
  temp3 = 1 := by
  sorry

#check third_day_temperature

end NUMINAMATH_CALUDE_third_day_temperature_l1817_181759


namespace NUMINAMATH_CALUDE_hyperbola_condition_ellipse_y_focus_condition_l1817_181790

-- Define the curve C
def C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (5 - t) + y^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) := (5 - t) * (t - 1) < 0

-- Define what it means for C to be an ellipse with focus on y-axis
def is_ellipse_y_focus (t : ℝ) := t - 1 > 5 - t ∧ t - 1 > 0 ∧ 5 - t > 0

-- Statement 1
theorem hyperbola_condition (t : ℝ) : 
  t < 1 → is_hyperbola t :=
sorry

-- Statement 2
theorem ellipse_y_focus_condition (t : ℝ) :
  is_ellipse_y_focus t → 3 < t ∧ t < 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_ellipse_y_focus_condition_l1817_181790


namespace NUMINAMATH_CALUDE_range_of_m_l1817_181733

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 7}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1817_181733


namespace NUMINAMATH_CALUDE_no_valid_n_for_ap_l1817_181788

theorem no_valid_n_for_ap : ¬∃ (n : ℕ), n > 1 ∧ 
  ∃ (a : ℤ), 136 = (n : ℤ) * (2 * a + (n - 1) * 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_for_ap_l1817_181788


namespace NUMINAMATH_CALUDE_correct_equation_transformation_l1817_181763

theorem correct_equation_transformation :
  ∀ x : ℚ, 3 * x = -7 ↔ x = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_transformation_l1817_181763


namespace NUMINAMATH_CALUDE_latest_departure_time_l1817_181754

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : Int :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

theorem latest_departure_time 
  (flight_time : Time)
  (check_in_time : Nat)
  (drive_time : Nat)
  (park_walk_time : Nat)
  (h1 : flight_time = ⟨20, 0⟩)  -- 8:00 pm
  (h2 : check_in_time = 120)    -- 2 hours
  (h3 : drive_time = 45)        -- 45 minutes
  (h4 : park_walk_time = 15)    -- 15 minutes
  : 
  let latest_departure := Time.mk 17 0  -- 5:00 pm
  timeDifferenceInMinutes flight_time latest_departure = 
    check_in_time + drive_time + park_walk_time :=
by sorry

end NUMINAMATH_CALUDE_latest_departure_time_l1817_181754


namespace NUMINAMATH_CALUDE_equation_roots_l1817_181772

-- Define the equation
def equation (x : ℝ) : Prop :=
  (21 / (x^2 - 9) - 3 / (x - 3) = 1)

-- Define the roots
def roots : Set ℝ := {-3, 7}

-- Theorem statement
theorem equation_roots :
  ∀ x : ℝ, x ∈ roots ↔ equation x ∧ x ≠ 3 ∧ x ≠ -3 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_l1817_181772


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1817_181716

theorem sqrt_equation_solution (x : ℝ) : (5 - 1/x)^(1/4) = -3 → x = -1/76 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1817_181716


namespace NUMINAMATH_CALUDE_same_grade_percentage_l1817_181796

/-- Represents the grade distribution for two assignments --/
structure GradeDistribution :=
  (aa ab ac ad : ℕ)
  (ba bb bc bd : ℕ)
  (ca cb cc cd : ℕ)
  (da db dc dd : ℕ)

/-- The total number of students --/
def totalStudents : ℕ := 40

/-- The grade distribution for the English class --/
def englishClassDistribution : GradeDistribution :=
  { aa := 3, ab := 2, ac := 1, ad := 0,
    ba := 1, bb := 6, bc := 3, bd := 1,
    ca := 0, cb := 2, cc := 7, cd := 2,
    da := 0, db := 1, dc := 2, dd := 2 }

/-- Calculates the number of students who received the same grade on both assignments --/
def sameGradeCount (dist : GradeDistribution) : ℕ :=
  dist.aa + dist.bb + dist.cc + dist.dd

/-- Theorem: The percentage of students who received the same grade on both assignments is 45% --/
theorem same_grade_percentage :
  (sameGradeCount englishClassDistribution : ℚ) / totalStudents * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l1817_181796


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1817_181749

def A : Set ℝ := {x | x * (x + 1) ≤ 0}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1817_181749


namespace NUMINAMATH_CALUDE_derek_walking_time_l1817_181779

/-- The time it takes Derek to walk a mile with his brother -/
def time_with_brother : ℝ := 12

/-- The time it takes Derek to walk a mile without his brother -/
def time_without_brother : ℝ := 9

/-- The additional time it takes to walk 20 miles with his brother -/
def additional_time : ℝ := 60

theorem derek_walking_time :
  time_with_brother = 12 ∧
  time_without_brother * 20 + additional_time = time_with_brother * 20 :=
sorry

end NUMINAMATH_CALUDE_derek_walking_time_l1817_181779


namespace NUMINAMATH_CALUDE_problem_solution_l1817_181760

theorem problem_solution (a b c : ℤ) : 
  a < b → b < c → 
  (a + b + c) / 3 = 4 * b → 
  c / b = 11 → 
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1817_181760


namespace NUMINAMATH_CALUDE_grandview_soccer_league_members_l1817_181743

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := 2 * sock_cost

/-- The total cost for one member's gear for both home and away games -/
def member_cost : ℕ := 2 * (sock_cost + tshirt_cost + cap_cost)

/-- The total cost for all members' gear -/
def total_cost : ℕ := 4410

/-- The number of members in the Grandview Soccer League -/
def num_members : ℕ := 70

theorem grandview_soccer_league_members :
  num_members * member_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_grandview_soccer_league_members_l1817_181743


namespace NUMINAMATH_CALUDE_intersection_M_N_l1817_181741

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_M_N : M ∩ N = {-1, 1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1817_181741


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1817_181780

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x > -a) ↔ a > -3/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1817_181780


namespace NUMINAMATH_CALUDE_arithmetic_seq_nth_term_l1817_181750

/-- Arithmetic sequence with first term 3 and common difference 2 -/
def arithmeticSeq (n : ℕ) : ℝ := 3 + 2 * (n - 1)

/-- Theorem: If the nth term of the arithmetic sequence is 25, then n is 12 -/
theorem arithmetic_seq_nth_term (n : ℕ) :
  arithmeticSeq n = 25 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_nth_term_l1817_181750


namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l1817_181768

def T : Finset ℕ := Finset.range 12

def m : ℕ := (3^12 - 2 * 2^12 + 1) / 2

theorem disjoint_subsets_remainder (T : Finset ℕ) (m : ℕ) :
  T = Finset.range 12 →
  m = (3^12 - 2 * 2^12 + 1) / 2 →
  m % 1000 = 625 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l1817_181768


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l1817_181701

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 * b2 = a2 * b1) ∧ (a1 ≠ 0 ∨ a2 ≠ 0)

/-- The problem statement -/
theorem parallel_lines_m_value (m : ℝ) :
  parallel_lines 1 (2*m) (-1) (m-2) (-m) 2 → m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l1817_181701


namespace NUMINAMATH_CALUDE_sqrt_x_plus_3_meaningful_l1817_181766

theorem sqrt_x_plus_3_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 3) ↔ x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_3_meaningful_l1817_181766


namespace NUMINAMATH_CALUDE_mr_a_speed_l1817_181723

/-- Proves that Mr. A's speed is 30 kmph given the problem conditions --/
theorem mr_a_speed (initial_distance : ℝ) (mrs_a_speed : ℝ) (bee_speed : ℝ) (bee_distance : ℝ)
  (h1 : initial_distance = 120)
  (h2 : mrs_a_speed = 10)
  (h3 : bee_speed = 60)
  (h4 : bee_distance = 180) :
  ∃ (mr_a_speed : ℝ), mr_a_speed = 30 ∧ 
    (bee_distance / bee_speed) * (mr_a_speed + mrs_a_speed) = initial_distance :=
by sorry

end NUMINAMATH_CALUDE_mr_a_speed_l1817_181723


namespace NUMINAMATH_CALUDE_range_of_a_l1817_181777

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ ({1, 2} : Set ℝ) → 3 * x^2 - a ≥ 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  (proposition_p a ∧ proposition_q a) → (a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1817_181777


namespace NUMINAMATH_CALUDE_soccer_tournament_points_l1817_181725

theorem soccer_tournament_points (n k : ℕ) (h1 : n ≥ 3) (h2 : 2 ≤ k) (h3 : k ≤ n - 1) :
  let min_points := 3 * n - (3 * k + 1) / 2 - 2
  ∀ (team_points : ℕ → ℕ),
    (∀ i, i < n → team_points i ≤ 3 * (n - 1)) →
    (∀ i j, i < n → j < n → i ≠ j → 
      team_points i + team_points j ≥ 1 ∧ team_points i + team_points j ≤ 4) →
    (∀ i, i < n → team_points i ≥ min_points) →
    ∃ (top_teams : Finset ℕ),
      top_teams.card ≤ k ∧
      ∀ j, j < n → j ∉ top_teams → team_points j < min_points :=
by sorry


end NUMINAMATH_CALUDE_soccer_tournament_points_l1817_181725


namespace NUMINAMATH_CALUDE_mary_eggs_problem_l1817_181731

theorem mary_eggs_problem (initial_eggs found_eggs final_eggs : ℕ) 
  (h1 : found_eggs = 4)
  (h2 : final_eggs = 31)
  (h3 : final_eggs = initial_eggs + found_eggs) :
  initial_eggs = 27 := by
  sorry

end NUMINAMATH_CALUDE_mary_eggs_problem_l1817_181731


namespace NUMINAMATH_CALUDE_gcd_2183_1947_l1817_181792

theorem gcd_2183_1947 : Nat.gcd 2183 1947 = 59 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2183_1947_l1817_181792


namespace NUMINAMATH_CALUDE_total_adjusted_income_equals_1219_72_l1817_181748

def initial_investment : ℝ := 6800
def stock_allocation : ℝ := 0.6
def bond_allocation : ℝ := 0.3
def cash_allocation : ℝ := 0.1
def inflation_rate : ℝ := 0.02

def cash_interest_rates : Fin 3 → ℝ
| 0 => 0.01
| 1 => 0.02
| 2 => 0.03

def stock_gains : Fin 3 → ℝ
| 0 => 0.08
| 1 => 0.04
| 2 => 0.10

def bond_returns : Fin 3 → ℝ
| 0 => 0.05
| 1 => 0.06
| 2 => 0.04

def adjusted_annual_income (i : Fin 3) : ℝ :=
  let stock_income := initial_investment * stock_allocation * stock_gains i
  let bond_income := initial_investment * bond_allocation * bond_returns i
  let cash_income := initial_investment * cash_allocation * cash_interest_rates i
  let total_income := stock_income + bond_income + cash_income
  total_income * (1 - inflation_rate)

theorem total_adjusted_income_equals_1219_72 :
  (adjusted_annual_income 0) + (adjusted_annual_income 1) + (adjusted_annual_income 2) = 1219.72 :=
sorry

end NUMINAMATH_CALUDE_total_adjusted_income_equals_1219_72_l1817_181748


namespace NUMINAMATH_CALUDE_probability_on_2x_is_one_twelfth_l1817_181739

/-- A die is a finite set of numbers from 1 to 6 -/
def Die : Finset ℕ := Finset.range 6

/-- The probability space of rolling a die twice -/
def DieRollSpace : Finset (ℕ × ℕ) := Die.product Die

/-- The event where (x, y) falls on y = 2x -/
def EventOn2x : Finset (ℕ × ℕ) := DieRollSpace.filter (fun (x, y) => y = 2 * x)

/-- The probability of the event -/
def ProbabilityOn2x : ℚ := (EventOn2x.card : ℚ) / (DieRollSpace.card : ℚ)

theorem probability_on_2x_is_one_twelfth : ProbabilityOn2x = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_on_2x_is_one_twelfth_l1817_181739


namespace NUMINAMATH_CALUDE_original_earnings_before_raise_l1817_181738

theorem original_earnings_before_raise (new_earnings : ℝ) (increase_percentage : ℝ) :
  new_earnings = 75 ∧ increase_percentage = 0.25 →
  ∃ original_earnings : ℝ,
    original_earnings * (1 + increase_percentage) = new_earnings ∧
    original_earnings = 60 :=
by sorry

end NUMINAMATH_CALUDE_original_earnings_before_raise_l1817_181738


namespace NUMINAMATH_CALUDE_m_three_sufficient_not_necessary_l1817_181726

def a (m : ℝ) : ℝ × ℝ := (-9, m^2)
def b : ℝ × ℝ := (1, -1)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem m_three_sufficient_not_necessary :
  (∃ (m : ℝ), m ≠ 3 ∧ parallel (a m) b) ∧
  (∀ (m : ℝ), m = 3 → parallel (a m) b) :=
sorry

end NUMINAMATH_CALUDE_m_three_sufficient_not_necessary_l1817_181726


namespace NUMINAMATH_CALUDE_average_speed_ratio_l1817_181761

/-- Given that Eddy travels 480 km in 3 hours and Freddy travels 300 km in 4 hours,
    prove that the ratio of their average speeds is 32:15. -/
theorem average_speed_ratio (eddy_distance : ℝ) (eddy_time : ℝ) (freddy_distance : ℝ) (freddy_time : ℝ)
    (h1 : eddy_distance = 480)
    (h2 : eddy_time = 3)
    (h3 : freddy_distance = 300)
    (h4 : freddy_time = 4) :
    (eddy_distance / eddy_time) / (freddy_distance / freddy_time) = 32 / 15 := by
  sorry

#check average_speed_ratio

end NUMINAMATH_CALUDE_average_speed_ratio_l1817_181761


namespace NUMINAMATH_CALUDE_company_fund_calculation_l1817_181794

theorem company_fund_calculation (n : ℕ) 
  (h1 : 80 * n - 15 = 70 * n + 155) : 
  80 * n - 15 = 1345 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_calculation_l1817_181794


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1817_181735

def P (n : ℕ) : ℕ := sorry

theorem unique_n_satisfying_conditions : 
  ∃! n : ℕ, n > 1 ∧ 
    P n = n - 8 ∧ 
    P (n + 60) = n + 52 :=
sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1817_181735


namespace NUMINAMATH_CALUDE_ryan_chinese_hours_l1817_181776

/-- Represents the daily learning schedule for Ryan -/
structure LearningSchedule where
  english_hours : ℕ
  chinese_hours : ℕ
  total_days : ℕ
  total_hours : ℕ

/-- Theorem: Ryan spends 7 hours each day on learning Chinese -/
theorem ryan_chinese_hours (schedule : LearningSchedule) 
  (h1 : schedule.english_hours = 6)
  (h2 : schedule.total_days = 5)
  (h3 : schedule.total_hours = 65) :
  schedule.chinese_hours = 7 := by
sorry

end NUMINAMATH_CALUDE_ryan_chinese_hours_l1817_181776


namespace NUMINAMATH_CALUDE_smallest_w_proof_l1817_181713

/-- The product of 1452 and the smallest positive integer w that results in a number 
    with 3^3 and 13^3 as factors -/
def smallest_w : ℕ := 19773

theorem smallest_w_proof :
  ∀ w : ℕ, w > 0 →
  (∃ k : ℕ, 1452 * w = k * 3^3 * 13^3) →
  w ≥ smallest_w :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_proof_l1817_181713


namespace NUMINAMATH_CALUDE_pipe_B_rate_correct_l1817_181769

/-- The rate at which pipe B fills the tank -/
def pipe_B_rate : ℝ := 30

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 750

/-- The rate at which pipe A fills the tank in liters per minute -/
def pipe_A_rate : ℝ := 40

/-- The rate at which pipe C drains the tank in liters per minute -/
def pipe_C_rate : ℝ := 20

/-- The time in minutes it takes to fill the tank -/
def fill_time : ℝ := 45

/-- The duration of each pipe's operation in a cycle in minutes -/
def cycle_duration : ℝ := 3

/-- Theorem stating that the calculated rate of pipe B is correct -/
theorem pipe_B_rate_correct : 
  pipe_B_rate = (tank_capacity - fill_time / cycle_duration * (pipe_A_rate - pipe_C_rate)) / 
                (fill_time / cycle_duration) :=
by sorry

end NUMINAMATH_CALUDE_pipe_B_rate_correct_l1817_181769


namespace NUMINAMATH_CALUDE_polar_to_cartesian_and_intersections_l1817_181717

/-- A circle in polar form -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

/-- A line in polar form -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Given a circle C₁ and a line C₂ in polar form, 
    prove their Cartesian equations and intersection points -/
theorem polar_to_cartesian_and_intersections 
  (C₁ : PolarCircle) 
  (C₂ : PolarLine) 
  (h₁ : C₁.equation = fun ρ θ ↦ ρ = 4 * Real.sin θ)
  (h₂ : C₂.equation = fun ρ θ ↦ ρ * Real.cos (θ - π/4) = 2 * Real.sqrt 2) :
  ∃ (f₁ f₂ : ℝ → ℝ → Prop) (p₁ p₂ : PolarPoint),
    (∀ x y, f₁ x y ↔ x^2 + (y-2)^2 = 4) ∧
    (∀ x y, f₂ x y ↔ x + y = 4) ∧
    p₁ = ⟨4, π/2⟩ ∧
    p₂ = ⟨2 * Real.sqrt 2, π/4⟩ ∧
    C₁.equation p₁.ρ p₁.θ ∧
    C₂.equation p₁.ρ p₁.θ ∧
    C₁.equation p₂.ρ p₂.θ ∧
    C₂.equation p₂.ρ p₂.θ := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_and_intersections_l1817_181717


namespace NUMINAMATH_CALUDE_circle_area_and_diameter_l1817_181709

theorem circle_area_and_diameter (C : ℝ) (h : C = 18 * Real.pi) : ∃ (A d : ℝ), A = 81 * Real.pi ∧ d = 18 ∧ A = Real.pi * (d / 2)^2 ∧ C = Real.pi * d := by
  sorry

end NUMINAMATH_CALUDE_circle_area_and_diameter_l1817_181709


namespace NUMINAMATH_CALUDE_calculation_proof_l1817_181744

theorem calculation_proof : (1/2)⁻¹ + Real.sqrt 12 - 4 * Real.sin (60 * π / 180) = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1817_181744


namespace NUMINAMATH_CALUDE_scissors_count_l1817_181732

/-- The number of scissors initially in the drawer -/
def initial_scissors : ℕ := 39

/-- The number of scissors Dan added to the drawer -/
def added_scissors : ℕ := 13

/-- The total number of scissors after Dan's addition -/
def total_scissors : ℕ := initial_scissors + added_scissors

/-- Theorem stating that the total number of scissors is 52 -/
theorem scissors_count : total_scissors = 52 := by
  sorry

end NUMINAMATH_CALUDE_scissors_count_l1817_181732


namespace NUMINAMATH_CALUDE_max_notebooks_is_11_l1817_181721

def single_notebook_cost : ℕ := 2
def pack_4_cost : ℕ := 6
def pack_7_cost : ℕ := 9
def total_money : ℕ := 15
def max_pack_7 : ℕ := 1

def notebooks_count (singles pack_4 pack_7 : ℕ) : ℕ :=
  singles + 4 * pack_4 + 7 * pack_7

def total_cost (singles pack_4 pack_7 : ℕ) : ℕ :=
  single_notebook_cost * singles + pack_4_cost * pack_4 + pack_7_cost * pack_7

theorem max_notebooks_is_11 :
  ∃ (singles pack_4 pack_7 : ℕ),
    notebooks_count singles pack_4 pack_7 = 11 ∧
    total_cost singles pack_4 pack_7 ≤ total_money ∧
    pack_7 ≤ max_pack_7 ∧
    ∀ (s p4 p7 : ℕ),
      total_cost s p4 p7 ≤ total_money →
      p7 ≤ max_pack_7 →
      notebooks_count s p4 p7 ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_is_11_l1817_181721


namespace NUMINAMATH_CALUDE_red_probability_both_jars_l1817_181711

/-- Represents a jar containing buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- Calculates the total number of buttons in a jar -/
def Jar.total (j : Jar) : ℕ := j.red + j.blue

/-- Calculates the probability of drawing a red button from a jar -/
def Jar.redProbability (j : Jar) : ℚ := j.red / j.total

/-- Represents the initial state of Jar A -/
def initialJarA : Jar := { red := 8, blue := 8 }

/-- Represents the transfer process -/
def transfer (j : Jar) : (Jar × Jar) :=
  let redTransfer := j.red / 3
  let blueTransfer := j.blue / 2
  let newJarA : Jar := { red := j.red - redTransfer, blue := j.blue - blueTransfer }
  let jarB : Jar := { red := redTransfer, blue := blueTransfer }
  (newJarA, jarB)

/-- The main theorem stating the probability of drawing red buttons from both jars -/
theorem red_probability_both_jars :
  let (jarA, jarB) := transfer initialJarA
  (jarA.redProbability * jarB.redProbability) = 5 / 21 := by
  sorry


end NUMINAMATH_CALUDE_red_probability_both_jars_l1817_181711


namespace NUMINAMATH_CALUDE_cycle_sale_gain_percent_l1817_181758

/-- Calculates the gain percent for a cycle sale given the original price, discount percentage, refurbishing cost, and selling price. -/
def cycleGainPercent (originalPrice discountPercent refurbishCost sellingPrice : ℚ) : ℚ :=
  let discountAmount := (discountPercent / 100) * originalPrice
  let purchasePrice := originalPrice - discountAmount
  let totalCostPrice := purchasePrice + refurbishCost
  let gain := sellingPrice - totalCostPrice
  (gain / totalCostPrice) * 100

/-- Theorem stating that the gain percent for the given cycle sale scenario is 62.5% -/
theorem cycle_sale_gain_percent :
  cycleGainPercent 1200 25 300 1950 = 62.5 := by
  sorry

#eval cycleGainPercent 1200 25 300 1950

end NUMINAMATH_CALUDE_cycle_sale_gain_percent_l1817_181758


namespace NUMINAMATH_CALUDE_man_to_boy_work_ratio_l1817_181785

/-- The daily work done by a man -/
def M : ℝ := sorry

/-- The daily work done by a boy -/
def B : ℝ := sorry

/-- The total amount of work to be done -/
def total_work : ℝ := sorry

/-- The first condition: 12 men and 16 boys can do the work in 5 days -/
axiom condition1 : 5 * (12 * M + 16 * B) = total_work

/-- The second condition: 13 men and 24 boys can do the work in 4 days -/
axiom condition2 : 4 * (13 * M + 24 * B) = total_work

/-- The theorem stating that the ratio of daily work done by a man to that of a boy is 2:1 -/
theorem man_to_boy_work_ratio : M / B = 2 := by sorry

end NUMINAMATH_CALUDE_man_to_boy_work_ratio_l1817_181785


namespace NUMINAMATH_CALUDE_probability_third_defective_correct_probability_correct_under_conditions_l1817_181795

/-- Represents the probability of drawing a defective item as the third draw
    given the conditions of the problem. -/
def probability_third_defective (total_items : ℕ) (defective_items : ℕ) (items_drawn : ℕ) : ℚ :=
  7 / 36

/-- Theorem stating the probability of drawing a defective item as the third draw
    under the given conditions. -/
theorem probability_third_defective_correct :
  probability_third_defective 10 3 3 = 7 / 36 := by
  sorry

/-- Checks if the conditions of the problem are met. -/
def valid_conditions (total_items : ℕ) (defective_items : ℕ) (items_drawn : ℕ) : Prop :=
  total_items = 10 ∧ defective_items = 3 ∧ items_drawn = 3

/-- Theorem stating that the probability is correct under the given conditions. -/
theorem probability_correct_under_conditions
  (total_items : ℕ) (defective_items : ℕ) (items_drawn : ℕ)
  (h : valid_conditions total_items defective_items items_drawn) :
  probability_third_defective total_items defective_items items_drawn = 7 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_third_defective_correct_probability_correct_under_conditions_l1817_181795


namespace NUMINAMATH_CALUDE_AD_length_l1817_181728

-- Define the points A, B, C, D, and M
variable (A B C D M : Point)

-- Define the length function
variable (length : Point → Point → ℝ)

-- State the conditions
axiom equal_segments : length A B = length B C ∧ length B C = length C D ∧ length C D = length A D / 4
axiom M_midpoint : length A M = length M D
axiom MC_length : length M C = 7

-- State the theorem
theorem AD_length : length A D = 56 / 3 := by sorry

end NUMINAMATH_CALUDE_AD_length_l1817_181728


namespace NUMINAMATH_CALUDE_max_value_fraction_l1817_181718

theorem max_value_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * (a + b + c) = b * c) :
  a / (b + c) ≤ (Real.sqrt 2 - 1) / 2 ∧
  ∃ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * (a + b + c) = b * c ∧ a / (b + c) = (Real.sqrt 2 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1817_181718


namespace NUMINAMATH_CALUDE_mark_additional_spending_l1817_181770

-- Define Mark's initial amount
def initial_amount : ℚ := 180

-- Define the amount spent in the first store
def first_store_spent (initial : ℚ) : ℚ := (1/2 * initial) + 14

-- Define the amount spent in the second store before the additional spending
def second_store_initial_spent (initial : ℚ) : ℚ := 1/3 * initial

-- Theorem to prove
theorem mark_additional_spending :
  initial_amount - first_store_spent initial_amount - second_store_initial_spent initial_amount = 16 := by
  sorry

end NUMINAMATH_CALUDE_mark_additional_spending_l1817_181770


namespace NUMINAMATH_CALUDE_min_additional_games_proof_l1817_181799

/-- The minimum number of additional games the Sharks need to win -/
def min_additional_games : ℕ := 145

/-- The initial number of games played -/
def initial_games : ℕ := 5

/-- The initial number of games won by the Sharks -/
def initial_sharks_wins : ℕ := 2

/-- Predicate to check if a given number of additional games satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  (initial_sharks_wins + n : ℚ) / (initial_games + n) ≥ 98 / 100

theorem min_additional_games_proof :
  satisfies_condition min_additional_games ∧
  ∀ m : ℕ, m < min_additional_games → ¬ satisfies_condition m :=
by sorry

end NUMINAMATH_CALUDE_min_additional_games_proof_l1817_181799


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1817_181798

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}

theorem sufficient_but_not_necessary : 
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1817_181798
