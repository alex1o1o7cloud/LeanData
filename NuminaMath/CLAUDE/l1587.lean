import Mathlib

namespace NUMINAMATH_CALUDE_number_of_hens_l1587_158760

/-- Represents the number of hens and cows a man has. -/
structure Animals where
  hens : ℕ
  cows : ℕ

/-- The total number of heads for the given animals. -/
def totalHeads (a : Animals) : ℕ := a.hens + a.cows

/-- The total number of feet for the given animals. -/
def totalFeet (a : Animals) : ℕ := 2 * a.hens + 4 * a.cows

/-- Theorem stating that given the conditions, the number of hens is 24. -/
theorem number_of_hens : 
  ∃ (a : Animals), totalHeads a = 48 ∧ totalFeet a = 144 ∧ a.hens = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_of_hens_l1587_158760


namespace NUMINAMATH_CALUDE_A_simplest_form_l1587_158706

/-- The complex expression A -/
def A : ℚ :=
  (0.375 * 2.6) / (2.5 * 1.2) +
  (0.625 * 1.6) / (3 * 1.2 * 4.1666666666666666) +
  6.666666666666667 * 0.12 +
  28 +
  (1 / 9) / 7 +
  0.2 / (9 * 22)

/-- Theorem stating that A, when expressed as a fraction in simplest form, has numerator 1901 and denominator 360 -/
theorem A_simplest_form :
  let (n, d) := (A.num, A.den)
  (n.gcd d = 1) ∧ (n = 1901) ∧ (d = 360) := by sorry

end NUMINAMATH_CALUDE_A_simplest_form_l1587_158706


namespace NUMINAMATH_CALUDE_units_digit_of_5_to_4_l1587_158717

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_5_to_4 : unitsDigit (5^4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_5_to_4_l1587_158717


namespace NUMINAMATH_CALUDE_house_of_cards_layers_l1587_158797

/-- Calculates the maximum number of layers in a house of cards --/
def maxLayers (decks : ℕ) (cardsPerDeck : ℕ) (cardsPerLayer : ℕ) : ℕ :=
  (decks * cardsPerDeck) / cardsPerLayer

/-- Theorem: Given 16 decks of 52 cards each, using 26 cards per layer,
    the maximum number of layers in a house of cards is 32 --/
theorem house_of_cards_layers :
  maxLayers 16 52 26 = 32 := by
  sorry

end NUMINAMATH_CALUDE_house_of_cards_layers_l1587_158797


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_four_l1587_158724

/-- Two-digit number represented as a pair of digits -/
def TwoDigitNumber := Nat × Nat

/-- Sum of digits of two two-digit numbers -/
def sumOfDigits (n1 n2 : TwoDigitNumber) : Nat :=
  n1.1 + n1.2 + n2.1 + n2.2

/-- Result of adding two two-digit numbers -/
def addTwoDigitNumbers (n1 n2 : TwoDigitNumber) : Nat × Nat × Nat :=
  let sum := n1.1 * 10 + n1.2 + n2.1 * 10 + n2.2
  (sum / 100, (sum / 10) % 10, sum % 10)

theorem sum_of_x_and_y_is_four (n1 n2 : TwoDigitNumber) :
  sumOfDigits n1 n2 = 22 →
  (addTwoDigitNumbers n1 n2).2.2 = 9 →
  (addTwoDigitNumbers n1 n2).1 + (addTwoDigitNumbers n1 n2).2.1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_four_l1587_158724


namespace NUMINAMATH_CALUDE_optimal_journey_solution_l1587_158747

/-- Represents the problem setup for the journey from M to N --/
structure JourneySetup where
  total_distance : ℝ
  walking_speed : ℝ
  cycling_speed : ℝ

/-- Represents the optimal solution for the journey --/
structure OptimalSolution where
  c_departure_time : ℝ
  walking_distance : ℝ
  cycling_distance : ℝ

/-- Theorem stating the optimal solution for the journey --/
theorem optimal_journey_solution (setup : JourneySetup) 
  (h1 : setup.total_distance = 15)
  (h2 : setup.walking_speed = 6)
  (h3 : setup.cycling_speed = 15) :
  ∃ (sol : OptimalSolution), 
    sol.c_departure_time = 3 / 11 ∧
    sol.walking_distance = 60 / 11 ∧
    sol.cycling_distance = 105 / 11 ∧
    (sol.walking_distance / setup.walking_speed + 
     sol.cycling_distance / setup.cycling_speed = 
     setup.total_distance / setup.cycling_speed + 
     sol.walking_distance / setup.walking_speed) ∧
    ∀ (other : OptimalSolution), 
      (other.walking_distance / setup.walking_speed + 
       other.cycling_distance / setup.cycling_speed ≥
       sol.walking_distance / setup.walking_speed + 
       sol.cycling_distance / setup.cycling_speed) :=
by sorry


end NUMINAMATH_CALUDE_optimal_journey_solution_l1587_158747


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l1587_158791

/-- A function f is increasing on an interval (a, +∞) if for all x₁, x₂ in the interval,
    x₁ < x₂ implies f x₁ < f x₂ -/
def IncreasingOn (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The quadratic function we're considering -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1 - a)*x + 2

theorem quadratic_increasing_condition (a : ℝ) :
  IncreasingOn (f a) 4 → a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_condition_l1587_158791


namespace NUMINAMATH_CALUDE_same_row_both_shows_l1587_158727

/-- Represents a seating arrangement for a show -/
def SeatingArrangement := Fin 50 → Fin 7

/-- The number of rows in the cinema -/
def num_rows : Nat := 7

/-- The number of children attending the shows -/
def num_children : Nat := 50

/-- Theorem: There exist at least two children who sat in the same row during both shows -/
theorem same_row_both_shows (morning_seating evening_seating : SeatingArrangement) :
  ∃ (i j : Fin 50), i ≠ j ∧
    morning_seating i = morning_seating j ∧
    evening_seating i = evening_seating j :=
sorry

end NUMINAMATH_CALUDE_same_row_both_shows_l1587_158727


namespace NUMINAMATH_CALUDE_five_people_arrangement_with_restriction_l1587_158742

/-- The number of ways to arrange n people in a line. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line where two specific people are always adjacent. -/
def adjacentArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of ways to arrange n people in a line where two specific people are not adjacent. -/
def nonAdjacentArrangements (n : ℕ) : ℕ := totalArrangements n - adjacentArrangements n

theorem five_people_arrangement_with_restriction :
  nonAdjacentArrangements 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_five_people_arrangement_with_restriction_l1587_158742


namespace NUMINAMATH_CALUDE_f_properties_l1587_158786

noncomputable section

def f (x : ℝ) := Real.log x - (x - 1)^2 / 2

theorem f_properties :
  let φ := (1 + Real.sqrt 5) / 2
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < φ → f x₁ < f x₂) ∧
  (∀ x, x > 1 → f x < x - 1) ∧
  (∀ k, k < 1 → ∃ x₀ > 1, ∀ x, 1 < x ∧ x < x₀ → f x > k * (x - 1)) ∧
  (∀ k, k ≥ 1 → ¬∃ x₀ > 1, ∀ x, 1 < x ∧ x < x₀ → f x > k * (x - 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_f_properties_l1587_158786


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1587_158752

/-- The inclination angle of a line is the angle it makes with the positive x-axis. -/
def inclination_angle (a b c : ℝ) : ℝ := sorry

/-- A line is represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem line_inclination_angle :
  let l : Line := { a := 1, b := -1, c := 1 }
  inclination_angle l.a l.b l.c = 45 * π / 180 := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1587_158752


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1587_158790

theorem polynomial_evaluation :
  let x : ℝ := -2
  x^4 + x^3 + x^2 + x + 2 = 12 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1587_158790


namespace NUMINAMATH_CALUDE_tony_water_consumption_l1587_158759

theorem tony_water_consumption (yesterday : ℝ) (two_days_ago : ℝ) 
  (h1 : yesterday = 48)
  (h2 : yesterday = two_days_ago - 0.04 * two_days_ago) :
  two_days_ago = 50 := by
  sorry

end NUMINAMATH_CALUDE_tony_water_consumption_l1587_158759


namespace NUMINAMATH_CALUDE_line_through_points_equation_l1587_158754

-- Define a line by two points
def Line (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • p1 + t • p2}

-- Define the equation of a line in the form ax + by + c = 0
def LineEquation (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Theorem statement
theorem line_through_points_equation :
  Line (3, 0) (0, 2) = LineEquation 2 3 (-6) := by sorry

end NUMINAMATH_CALUDE_line_through_points_equation_l1587_158754


namespace NUMINAMATH_CALUDE_option_C_equals_nine_l1587_158750

theorem option_C_equals_nine : 3 * 3 - 3 + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_option_C_equals_nine_l1587_158750


namespace NUMINAMATH_CALUDE_greatest_power_of_ten_dividing_twenty_factorial_l1587_158777

theorem greatest_power_of_ten_dividing_twenty_factorial : 
  (∃ m : ℕ, (20 : ℕ).factorial % (10 ^ m) = 0 ∧ 
    ∀ k : ℕ, k > m → (20 : ℕ).factorial % (10 ^ k) ≠ 0) → 
  (∃ m : ℕ, m = 4 ∧ (20 : ℕ).factorial % (10 ^ m) = 0 ∧ 
    ∀ k : ℕ, k > m → (20 : ℕ).factorial % (10 ^ k) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_ten_dividing_twenty_factorial_l1587_158777


namespace NUMINAMATH_CALUDE_roses_in_vase_l1587_158773

theorem roses_in_vase (total_flowers : ℕ) (carnations : ℕ) (roses : ℕ) : 
  total_flowers = 10 → carnations = 5 → total_flowers = roses + carnations → roses = 5 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l1587_158773


namespace NUMINAMATH_CALUDE_set_operation_result_l1587_158737

def A : Set Int := {-1, 0}
def B : Set Int := {0, 1}
def C : Set Int := {1, 2}

theorem set_operation_result : (A ∩ B) ∪ C = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l1587_158737


namespace NUMINAMATH_CALUDE_work_done_by_resultant_force_l1587_158722

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Calculates the dot product of two 2D vectors -/
def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Adds two 2D vectors -/
def add_vectors (v1 v2 : Vector2D) : Vector2D :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

theorem work_done_by_resultant_force : 
  let f1 : Vector2D := ⟨3, -4⟩
  let f2 : Vector2D := ⟨2, -5⟩
  let f3 : Vector2D := ⟨3, 1⟩
  let a : Vector2D := ⟨1, 1⟩
  let b : Vector2D := ⟨0, 5⟩
  let resultant_force := add_vectors (add_vectors f1 f2) f3
  let displacement := ⟨b.x - a.x, b.y - a.y⟩
  dot_product resultant_force displacement = -40 := by
  sorry

end NUMINAMATH_CALUDE_work_done_by_resultant_force_l1587_158722


namespace NUMINAMATH_CALUDE_paula_shopping_theorem_l1587_158766

/-- Calculates the remaining money after Paula's shopping trip -/
def remaining_money (initial_amount : ℕ) (num_shirts : ℕ) (shirt_price : ℕ) 
  (num_pants : ℕ) (pants_price : ℕ) : ℕ :=
  initial_amount - (num_shirts * shirt_price + num_pants * pants_price)

/-- Proves that Paula has $100 left after her shopping trip -/
theorem paula_shopping_theorem :
  remaining_money 250 5 15 3 25 = 100 := by
  sorry

end NUMINAMATH_CALUDE_paula_shopping_theorem_l1587_158766


namespace NUMINAMATH_CALUDE_modular_inverse_11_mod_1021_l1587_158715

theorem modular_inverse_11_mod_1021 : ∃ x : ℕ, x ∈ Finset.range 1021 ∧ (11 * x) % 1021 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_11_mod_1021_l1587_158715


namespace NUMINAMATH_CALUDE_trees_planted_l1587_158723

def road_length : ℕ := 2575
def tree_interval : ℕ := 25

theorem trees_planted (n : ℕ) : 
  n = road_length / tree_interval + 1 → n = 104 := by
  sorry

end NUMINAMATH_CALUDE_trees_planted_l1587_158723


namespace NUMINAMATH_CALUDE_triangle_properties_l1587_158738

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 + t.b^2 - t.c^2 = 4 * t.area) 
  (h2 : t.c = Real.sqrt 2) : 
  (t.C = Real.pi / 4) ∧ 
  (-1 < t.a - (Real.sqrt 2 / 2) * t.b) ∧ 
  (t.a - (Real.sqrt 2 / 2) * t.b < Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1587_158738


namespace NUMINAMATH_CALUDE_expand_expression_l1587_158745

-- Statement of the theorem
theorem expand_expression (x : ℝ) : (x + 3) * (6 * x - 12) = 6 * x^2 + 6 * x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1587_158745


namespace NUMINAMATH_CALUDE_salary_comparison_l1587_158755

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) :
  (b - a) / a * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_salary_comparison_l1587_158755


namespace NUMINAMATH_CALUDE_interior_triangle_area_l1587_158798

theorem interior_triangle_area (a b c : ℝ) (ha : a = 64) (hb : b = 225) (hc : c = 289)
  (h_right_triangle : a + b = c) : (1/2) * Real.sqrt a * Real.sqrt b = 60 := by
  sorry

end NUMINAMATH_CALUDE_interior_triangle_area_l1587_158798


namespace NUMINAMATH_CALUDE_inconsistent_school_population_l1587_158735

theorem inconsistent_school_population (total_students : Real) 
  (boy_percentage : Real) (representative_students : Nat) : 
  total_students = 113.38934190276818 → 
  boy_percentage = 0.70 → 
  representative_students = 90 → 
  (representative_students : Real) / (total_students * boy_percentage) > 1 := by
  sorry

end NUMINAMATH_CALUDE_inconsistent_school_population_l1587_158735


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l1587_158784

theorem angle_ABC_measure :
  ∀ (ABC ABD CBD : ℝ),
  CBD = 90 →
  ABC + ABD + CBD = 180 →
  ABD = 60 →
  ABC = 30 := by
sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l1587_158784


namespace NUMINAMATH_CALUDE_quadratic_completion_l1587_158753

theorem quadratic_completion (x : ℝ) : ∃ (a b : ℝ), x^2 - 6*x + 5 = 0 ↔ (x + a)^2 = b ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l1587_158753


namespace NUMINAMATH_CALUDE_largest_three_digit_integer_l1587_158789

theorem largest_three_digit_integer (n : ℕ) (a b c : ℕ) : 
  n = 100 * a + 10 * b + c →
  100 ≤ n → n < 1000 →
  2 ∣ a →
  3 ∣ (10 * a + b) →
  ¬(6 ∣ (10 * a + b)) →
  5 ∣ n →
  ¬(7 ∣ n) →
  n ≤ 870 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_integer_l1587_158789


namespace NUMINAMATH_CALUDE_parabola_directrix_l1587_158751

/-- The parabola defined by y = 8x^2 + 2 has a directrix y = 63/32 -/
theorem parabola_directrix : ∀ (x y : ℝ), y = 8 * x^2 + 2 → 
  ∃ (f d : ℝ), f = -d ∧ f - d = 1/16 ∧ d = -1/32 ∧ 
  (∀ (p : ℝ × ℝ), p.2 = 8 * p.1^2 + 2 → 
    (p.1^2 + (p.2 - (f + 2))^2 = (p.2 - (d + 2))^2)) ∧
  63/32 = d + 2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_directrix_l1587_158751


namespace NUMINAMATH_CALUDE_linear_func_not_in_M_exp_func_in_M_sin_func_in_M_iff_l1587_158714

-- Define the property for a function to be in set M
def in_set_M (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = T * f x

-- Part 1
theorem linear_func_not_in_M : ¬ in_set_M (λ x : ℝ ↦ x) := by sorry

-- Part 2
theorem exp_func_in_M (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  (∃ T : ℝ, T > 0 ∧ a^T = T) → in_set_M (λ x : ℝ ↦ a^x) := by sorry

-- Part 3
theorem sin_func_in_M_iff (k : ℝ) :
  in_set_M (λ x : ℝ ↦ Real.sin (k * x)) ↔ ∃ m : ℤ, k = m * Real.pi := by sorry

end NUMINAMATH_CALUDE_linear_func_not_in_M_exp_func_in_M_sin_func_in_M_iff_l1587_158714


namespace NUMINAMATH_CALUDE_triangle_area_is_one_l1587_158793

-- Define the complex number z
def z : ℂ := sorry

-- State the given conditions
axiom z_magnitude : Complex.abs z = Real.sqrt 2
axiom z_squared_imag : Complex.im (z ^ 2) = 2

-- Define the points A, B, and C
def A : ℂ := z
def B : ℂ := z ^ 2
def C : ℂ := z - z ^ 2

-- Define the area of the triangle
def triangle_area : ℝ := sorry

-- State the theorem to be proved
theorem triangle_area_is_one : triangle_area = 1 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_one_l1587_158793


namespace NUMINAMATH_CALUDE_charity_event_probability_l1587_158778

/-- The number of students participating in the charity event -/
def num_students : ℕ := 4

/-- The number of days students can choose from (Saturday and Sunday) -/
def num_days : ℕ := 2

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := num_days ^ num_students

/-- The number of outcomes where students participate on both days -/
def both_days_outcomes : ℕ := total_outcomes - num_days

/-- The probability of students participating on both days -/
def probability_both_days : ℚ := both_days_outcomes / total_outcomes

theorem charity_event_probability : probability_both_days = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_charity_event_probability_l1587_158778


namespace NUMINAMATH_CALUDE_surface_area_of_circumscribed_sphere_l1587_158726

/-- A regular tetrahedron with edge length √2 -/
structure RegularTetrahedron where
  edgeLength : ℝ
  isRegular : edgeLength = Real.sqrt 2

/-- A sphere circumscribing a regular tetrahedron -/
structure CircumscribedSphere (t : RegularTetrahedron) where
  radius : ℝ
  containsVertices : True  -- This is a placeholder for the condition that all vertices are on the sphere

/-- The surface area of a sphere circumscribing a regular tetrahedron with edge length √2 is 3π -/
theorem surface_area_of_circumscribed_sphere (t : RegularTetrahedron) (s : CircumscribedSphere t) :
  4 * Real.pi * s.radius ^ 2 = 3 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_circumscribed_sphere_l1587_158726


namespace NUMINAMATH_CALUDE_traveler_money_problem_l1587_158799

/-- Represents the amount of money a traveler has at the start of each day -/
def money_at_day (initial_money : ℚ) : ℕ → ℚ
  | 0 => initial_money
  | n + 1 => (money_at_day initial_money n / 2) - 1

theorem traveler_money_problem (initial_money : ℚ) :
  (money_at_day initial_money 0 > 0) ∧
  (money_at_day initial_money 1 > 0) ∧
  (money_at_day initial_money 2 > 0) ∧
  (money_at_day initial_money 3 = 0) →
  initial_money = 14 := by
sorry

end NUMINAMATH_CALUDE_traveler_money_problem_l1587_158799


namespace NUMINAMATH_CALUDE_circle_equation_implies_m_lt_5_l1587_158720

/-- A circle in the xy-plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- The equation of a circle given by x^2 + y^2 - 4x - 2y + m = 0 --/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + m = 0

/-- Theorem: If x^2 + y^2 - 4x - 2y + m = 0 represents a circle, then m < 5 --/
theorem circle_equation_implies_m_lt_5 :
  ∀ m : ℝ, (∃ c : Circle, ∀ x y : ℝ, circle_equation x y m ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) → m < 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_implies_m_lt_5_l1587_158720


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1587_158721

def A : Set ℕ := {0,1,2,3,4,6,7}
def B : Set ℕ := {1,2,4,8,0}

theorem intersection_of_A_and_B : A ∩ B = {1,2,4,0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1587_158721


namespace NUMINAMATH_CALUDE_difference_of_squares_l1587_158748

theorem difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1587_158748


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1587_158785

/-- An ellipse with foci at (7, 15) and (53, 65) that is tangent to the y-axis has a major axis of length 68. -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ : ℝ × ℝ),
    F₁ = (7, 15) →
    F₂ = (53, 65) →
    (∃ (y : ℝ), (0, y) ∈ E) →
    (∀ (P : ℝ × ℝ), P ∈ E ↔ 
      ∃ (k : ℝ), dist P F₁ + dist P F₂ = k ∧ 
      ∀ (Q : ℝ × ℝ), dist Q F₁ + dist Q F₂ ≤ k) →
    ∃ (a : ℝ), a = 68 ∧ 
      ∀ (P : ℝ × ℝ), P ∈ E → dist P F₁ + dist P F₂ = a :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1587_158785


namespace NUMINAMATH_CALUDE_portfolio_annual_yield_is_correct_l1587_158731

structure Security where
  quantity : ℕ
  initialPrice : ℝ
  priceAfter180Days : ℝ

def Portfolio : List Security := [
  ⟨1000, 95.3, 98.6⟩,
  ⟨1000, 89.5, 93.4⟩,
  ⟨1000, 92.1, 96.2⟩,
  ⟨1, 100000, 104300⟩,
  ⟨1, 200000, 209420⟩,
  ⟨40, 3700, 3900⟩,
  ⟨500, 137, 142⟩
]

def calculateAnnualYield (portfolio : List Security) : ℝ :=
  sorry

theorem portfolio_annual_yield_is_correct :
  abs (calculateAnnualYield Portfolio - 9.21) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_portfolio_annual_yield_is_correct_l1587_158731


namespace NUMINAMATH_CALUDE_absolute_value_sum_l1587_158780

theorem absolute_value_sum (m n p : ℤ) 
  (h : |m - n|^3 + |p - m|^5 = 1) : 
  |p - m| + |m - n| + 2 * |n - p| = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l1587_158780


namespace NUMINAMATH_CALUDE_balloon_problem_solution_l1587_158710

/-- The total number of balloons Brooke and Tracy have after Tracy pops half of hers -/
def total_balloons (brooke_initial : ℕ) (brooke_added : ℕ) (tracy_initial : ℕ) (tracy_added : ℕ) : ℕ :=
  let brooke_total := brooke_initial + brooke_added
  let tracy_before_popping := tracy_initial + tracy_added
  let tracy_after_popping := tracy_before_popping / 2
  brooke_total + tracy_after_popping

/-- Theorem stating that the total number of balloons is 35 given the problem conditions -/
theorem balloon_problem_solution :
  total_balloons 12 8 6 24 = 35 := by
  sorry

end NUMINAMATH_CALUDE_balloon_problem_solution_l1587_158710


namespace NUMINAMATH_CALUDE_polynomial_factors_l1587_158718

theorem polynomial_factors (x : ℝ) : 
  ∃ (a b c : ℝ), 8*x^3 + 14*x^2 - 17*x + 6 = (x + 1/2) * (x - 2) * (a*x + b) ∧ c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factors_l1587_158718


namespace NUMINAMATH_CALUDE_counterexample_exists_l1587_158787

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem counterexample_exists : ∃ n : ℕ, 
  ¬(is_prime n) ∧ ¬(is_prime (n - 5)) ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1587_158787


namespace NUMINAMATH_CALUDE_cos_alpha_minus_beta_l1587_158796

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : 2 * Real.cos α - Real.cos β = 3/2)
  (h2 : 2 * Real.sin α - Real.sin β = 2) :
  Real.cos (α - β) = -5/16 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_beta_l1587_158796


namespace NUMINAMATH_CALUDE_passing_marks_l1587_158711

/-- Given an exam with total marks T and passing marks P, prove that P = 240 -/
theorem passing_marks (T : ℝ) (P : ℝ) : 
  (0.30 * T = P - 60) →  -- Condition 1: 30% fails by 60 marks
  (0.45 * T = P + 30) →  -- Condition 2: 45% passes by 30 marks
  P = 240 := by
sorry

end NUMINAMATH_CALUDE_passing_marks_l1587_158711


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1587_158732

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1587_158732


namespace NUMINAMATH_CALUDE_unique_prime_product_l1587_158709

theorem unique_prime_product (p q r : Nat) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  p * q * r = 7802 ∧
  p + q + r = 1306 →
  ∀ (p1 p2 p3 : Nat), 
    Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    p1 * p2 * p3 ≠ 7802 ∧
    p1 + p2 + p3 = 1306 →
    False :=
by sorry

#check unique_prime_product

end NUMINAMATH_CALUDE_unique_prime_product_l1587_158709


namespace NUMINAMATH_CALUDE_rational_function_value_l1587_158781

-- Define f as a function from ℚ to ℚ (rational numbers)
variable (f : ℚ → ℚ)

-- State the main theorem
theorem rational_function_value : 
  (∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + 3 * f x / x = 2 * x^2) →
  f (-3) = 494 / 117 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l1587_158781


namespace NUMINAMATH_CALUDE_integer_solutions_equation_l1587_158795

theorem integer_solutions_equation :
  ∀ x y : ℤ, 2*x^2 + 8*y^2 = 17*x*y - 423 ↔ (x = 11 ∧ y = 19) ∨ (x = -11 ∧ y = -19) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_equation_l1587_158795


namespace NUMINAMATH_CALUDE_saree_discount_problem_l1587_158757

/-- Proves that given a saree with an original price of 600, after a 20% discount
    and a second discount resulting in a final price of 456, the second discount percentage is 5% -/
theorem saree_discount_problem (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ)
    (h1 : original_price = 600)
    (h2 : first_discount = 20)
    (h3 : final_price = 456) :
    let price_after_first_discount := original_price * (1 - first_discount / 100)
    let second_discount_amount := price_after_first_discount - final_price
    let second_discount_percentage := (second_discount_amount / price_after_first_discount) * 100
    second_discount_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_saree_discount_problem_l1587_158757


namespace NUMINAMATH_CALUDE_correct_equation_by_moving_digit_l1587_158705

theorem correct_equation_by_moving_digit : ∃ (a b c : ℕ), 
  (101 = 10^2 - 1 → False) ∧ 
  (101 = a * 10^2 + b * 10 + c - 1) ∧
  (a = 1 ∧ b = 0 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_correct_equation_by_moving_digit_l1587_158705


namespace NUMINAMATH_CALUDE_decimal_to_binary_111_octal_to_decimal_77_l1587_158749

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : ℕ) : List Bool := sorry

/-- Converts an octal number to its decimal representation -/
def octalToDecimal (n : ℕ) : ℕ := sorry

theorem decimal_to_binary_111 :
  decimalToBinary 111 = [true, true, false, true, true, true, true] := by sorry

theorem octal_to_decimal_77 :
  octalToDecimal 77 = 63 := by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_111_octal_to_decimal_77_l1587_158749


namespace NUMINAMATH_CALUDE_value_added_to_half_l1587_158713

theorem value_added_to_half : ∃ v : ℝ, (1/2 : ℝ) * 16 + v = 13 ∧ v = 5 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_half_l1587_158713


namespace NUMINAMATH_CALUDE_coffee_stock_problem_l1587_158774

/-- Represents the coffee stock problem --/
theorem coffee_stock_problem 
  (initial_stock : ℝ) 
  (additional_purchase : ℝ) 
  (decaf_percent_additional : ℝ) 
  (total_decaf_percent : ℝ) : 
  initial_stock = 400 ∧ 
  additional_purchase = 100 ∧ 
  decaf_percent_additional = 60 ∧ 
  total_decaf_percent = 32 → 
  (initial_stock * (25 / 100) + additional_purchase * (decaf_percent_additional / 100)) / 
  (initial_stock + additional_purchase) = total_decaf_percent / 100 := by
  sorry

#check coffee_stock_problem

end NUMINAMATH_CALUDE_coffee_stock_problem_l1587_158774


namespace NUMINAMATH_CALUDE_house_sale_profit_l1587_158739

theorem house_sale_profit (initial_value : ℝ) (first_sale_profit_percent : ℝ) (second_sale_loss_percent : ℝ) : 
  initial_value = 200000 ∧ 
  first_sale_profit_percent = 15 ∧ 
  second_sale_loss_percent = 20 → 
  (initial_value * (1 + first_sale_profit_percent / 100)) * (1 - second_sale_loss_percent / 100) - initial_value = 46000 :=
by sorry

end NUMINAMATH_CALUDE_house_sale_profit_l1587_158739


namespace NUMINAMATH_CALUDE_solve_dancers_earnings_l1587_158776

def dancers_earnings (total : ℚ) (d1 d2 d3 d4 : ℚ) : Prop :=
  d1 + d2 + d3 + d4 = total ∧
  d2 = d1 - 16 ∧
  d3 = d1 + d2 - 24 ∧
  d4 = d1 + d3

theorem solve_dancers_earnings :
  ∃ d1 d2 d3 d4 : ℚ,
    dancers_earnings 280 d1 d2 d3 d4 ∧
    d1 = 53 + 5/7 ∧
    d2 = 37 + 5/7 ∧
    d3 = 67 + 3/7 ∧
    d4 = 121 + 1/7 :=
by sorry

end NUMINAMATH_CALUDE_solve_dancers_earnings_l1587_158776


namespace NUMINAMATH_CALUDE_exists_same_type_quadratic_surd_with_three_l1587_158770

/-- Two square roots are of the same type of quadratic surd if one can be expressed as a rational multiple of the other. -/
def same_type_quadratic_surd (x y : ℝ) : Prop :=
  ∃ (q : ℚ), x = q * y ∨ y = q * x

theorem exists_same_type_quadratic_surd_with_three :
  ∃ (a : ℕ), a > 0 ∧ same_type_quadratic_surd (Real.sqrt a) (Real.sqrt 3) ∧ a = 12 := by
  sorry

end NUMINAMATH_CALUDE_exists_same_type_quadratic_surd_with_three_l1587_158770


namespace NUMINAMATH_CALUDE_john_cycling_distance_l1587_158741

def base_eight_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem john_cycling_distance : base_eight_to_decimal 6375 = 3325 := by
  sorry

end NUMINAMATH_CALUDE_john_cycling_distance_l1587_158741


namespace NUMINAMATH_CALUDE_parabola_max_area_l1587_158733

/-- A parabola with y-axis symmetry -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- The equation of a parabola -/
def Parabola.equation (p : Parabola) (x : ℝ) : ℝ := p.a * x^2 + p.c

/-- The condition that the parabola is concave up -/
def Parabola.concaveUp (p : Parabola) : Prop := p.a > 0

/-- The condition that the parabola touches the graph y = 1 - |x| -/
def Parabola.touchesGraph (p : Parabola) : Prop :=
  ∃ x₀ : ℝ, p.equation x₀ = 1 - |x₀| ∧ 
    (deriv p.equation) x₀ = if x₀ ≥ 0 then -1 else 1

/-- The area between the parabola and the x-axis -/
noncomputable def Parabola.area (p : Parabola) : ℝ :=
  ∫ x in (-Real.sqrt (1/p.a))..(Real.sqrt (1/p.a)), p.equation x

/-- The theorem statement -/
theorem parabola_max_area :
  ∀ p : Parabola, 
    p.concaveUp → 
    p.touchesGraph → 
    p.area ≤ Parabola.area ⟨1, 3/4⟩ :=
sorry

end NUMINAMATH_CALUDE_parabola_max_area_l1587_158733


namespace NUMINAMATH_CALUDE_a_less_than_one_l1587_158767

theorem a_less_than_one : 
  (0.99999 : ℝ)^(1.00001 : ℝ) * (1.00001 : ℝ)^(0.99999 : ℝ) < 1 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_one_l1587_158767


namespace NUMINAMATH_CALUDE_budget_remainder_l1587_158772

-- Define the given conditions
def weekly_budget : ℝ := 80
def fried_chicken_cost : ℝ := 12
def beef_pounds : ℝ := 4.5
def beef_price_per_pound : ℝ := 3
def soup_cans : ℕ := 3
def soup_cost_per_can : ℝ := 2
def milk_original_price : ℝ := 4
def milk_discount_percentage : ℝ := 0.1

-- Define the theorem
theorem budget_remainder : 
  let beef_cost := beef_pounds * beef_price_per_pound
  let soup_cost := (soup_cans - 1) * soup_cost_per_can
  let milk_cost := milk_original_price * (1 - milk_discount_percentage)
  let total_cost := fried_chicken_cost + beef_cost + soup_cost + milk_cost
  weekly_budget - total_cost = 46.90 := by
  sorry

end NUMINAMATH_CALUDE_budget_remainder_l1587_158772


namespace NUMINAMATH_CALUDE_no_two_digit_factorization_2109_l1587_158794

/-- A function that returns the number of ways to factor a positive integer
    as a product of two two-digit numbers -/
def count_two_digit_factorizations (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    10 ≤ p.1 ∧ p.1 < 100 ∧ 
    10 ≤ p.2 ∧ p.2 < 100 ∧ 
    p.1 * p.2 = n)
    (Finset.product (Finset.range 90) (Finset.range 90))).card / 2

/-- Theorem stating that 2109 cannot be factored as a product of two two-digit numbers -/
theorem no_two_digit_factorization_2109 : 
  count_two_digit_factorizations 2109 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_factorization_2109_l1587_158794


namespace NUMINAMATH_CALUDE_point_transformation_l1587_158712

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the initial point A
def A : Point2D := ⟨5, 4⟩

-- Define the transformation function
def transform (p : Point2D) : Point2D :=
  ⟨p.x - 4, p.y - 3⟩

-- State the theorem
theorem point_transformation :
  transform A = Point2D.mk 1 1 := by sorry

end NUMINAMATH_CALUDE_point_transformation_l1587_158712


namespace NUMINAMATH_CALUDE_round_and_convert_0_000359_l1587_158740

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ coefficient ∧ coefficient < 10

/-- Rounds a real number to a given number of significant figures -/
def round_to_sig_figs (x : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem round_and_convert_0_000359 :
  let rounded := round_to_sig_figs 0.000359 2
  let scientific := to_scientific_notation rounded
  scientific.coefficient = 3.6 ∧ scientific.exponent = -4 := by
  sorry

end NUMINAMATH_CALUDE_round_and_convert_0_000359_l1587_158740


namespace NUMINAMATH_CALUDE_percentage_problem_l1587_158792

theorem percentage_problem (P : ℝ) : P * 300 - 70 = 20 → P = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1587_158792


namespace NUMINAMATH_CALUDE_shortest_chord_length_l1587_158719

/-- Circle C with center (1,2) and radius 5 -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

/-- Line l passing through point M(3,1) -/
def line_l (x y m : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

/-- Point M(3,1) -/
def point_M : ℝ × ℝ := (3, 1)

/-- M is inside circle C -/
axiom M_inside_C : circle_C point_M.1 point_M.2

/-- The shortest chord theorem -/
theorem shortest_chord_length :
  ∃ (m : ℝ), line_l point_M.1 point_M.2 m →
  (∀ (x y : ℝ), line_l x y m → circle_C x y →
  ∃ (x' y' : ℝ), line_l x' y' m ∧ circle_C x' y' ∧
  ((x - x')^2 + (y - y')^2)^(1/2) ≤ 4 * 5^(1/2)) ∧
  (∃ (x y x' y' : ℝ), line_l x y m ∧ circle_C x y ∧
  line_l x' y' m ∧ circle_C x' y' ∧
  ((x - x')^2 + (y - y')^2)^(1/2) = 4 * 5^(1/2)) :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_length_l1587_158719


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1587_158762

theorem simplify_trig_expression (α : Real) 
  (h : -3 * Real.pi < α ∧ α < -(5/2) * Real.pi) : 
  Real.sqrt ((1 + Real.cos (α - 2018 * Real.pi)) / 2) = -Real.cos (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1587_158762


namespace NUMINAMATH_CALUDE_gcd_of_squared_sums_gcd_of_specific_squared_sums_l1587_158744

theorem gcd_of_squared_sums (a b c d e f : ℕ) : 
  Nat.gcd (a^2 + b^2 + c^2) (d^2 + e^2 + f^2) = 
  Nat.gcd ((a^2 + b^2 + c^2) - (d^2 + e^2 + f^2)) (d^2 + e^2 + f^2) :=
by sorry

theorem gcd_of_specific_squared_sums : 
  Nat.gcd (131^2 + 243^2 + 357^2) (130^2 + 242^2 + 358^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_squared_sums_gcd_of_specific_squared_sums_l1587_158744


namespace NUMINAMATH_CALUDE_page_lines_increase_percentage_l1587_158779

theorem page_lines_increase_percentage : 
  ∀ (original_lines : ℕ), 
  original_lines + 200 = 350 → 
  (200 : ℝ) / original_lines * 100 = 400 / 3 := by
sorry

end NUMINAMATH_CALUDE_page_lines_increase_percentage_l1587_158779


namespace NUMINAMATH_CALUDE_indexCardsPerStudentIs10_l1587_158768

/-- Calculates the number of index cards each student receives given the following conditions:
  * Carl teaches 6 periods a day
  * Each class has 30 students
  * A 50 pack of index cards costs $3
  * Carl spent $108 on index cards
-/
def indexCardsPerStudent (periods : Nat) (studentsPerClass : Nat) (cardsPerPack : Nat) 
  (costPerPack : Nat) (totalSpent : Nat) : Nat :=
  let totalPacks := totalSpent / costPerPack
  let totalCards := totalPacks * cardsPerPack
  let totalStudents := periods * studentsPerClass
  totalCards / totalStudents

theorem indexCardsPerStudentIs10 : 
  indexCardsPerStudent 6 30 50 3 108 = 10 := by
  sorry

end NUMINAMATH_CALUDE_indexCardsPerStudentIs10_l1587_158768


namespace NUMINAMATH_CALUDE_final_coin_count_l1587_158729

def coin_collection (initial : ℕ) (years : ℕ) : ℕ :=
  let year1 := initial * 2
  let year2 := year1 + 12 * 3
  let year3 := year2 + 12 / 3
  let year4 := year3 - year3 / 4
  year4

theorem final_coin_count : coin_collection 50 4 = 105 := by
  sorry

end NUMINAMATH_CALUDE_final_coin_count_l1587_158729


namespace NUMINAMATH_CALUDE_absolute_sum_inequality_l1587_158702

theorem absolute_sum_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x + 9| > a) → a < 8 := by
  sorry

end NUMINAMATH_CALUDE_absolute_sum_inequality_l1587_158702


namespace NUMINAMATH_CALUDE_problem_statement_l1587_158788

theorem problem_statement :
  (∀ x : ℝ, x < 0 → (2 : ℝ)^x > (3 : ℝ)^x) ∧
  (¬ ∃ x : ℝ, 0 < x ∧ x < Real.pi / 2 ∧ Real.sin x > x) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1587_158788


namespace NUMINAMATH_CALUDE_johns_donation_size_l1587_158775

/-- Represents the donation problem with given conditions -/
structure DonationProblem where
  num_previous_donations : ℕ
  new_average : ℚ
  increase_percentage : ℚ

/-- Calculates John's donation size based on the given conditions -/
def calculate_donation_size (problem : DonationProblem) : ℚ :=
  let previous_average := problem.new_average / (1 + problem.increase_percentage)
  let total_before := previous_average * problem.num_previous_donations
  let total_after := problem.new_average * (problem.num_previous_donations + 1)
  total_after - total_before

/-- Theorem stating that John's donation size is $225 given the problem conditions -/
theorem johns_donation_size (problem : DonationProblem) 
  (h1 : problem.num_previous_donations = 6)
  (h2 : problem.new_average = 75)
  (h3 : problem.increase_percentage = 1/2) :
  calculate_donation_size problem = 225 := by
  sorry

#eval calculate_donation_size { num_previous_donations := 6, new_average := 75, increase_percentage := 1/2 }

end NUMINAMATH_CALUDE_johns_donation_size_l1587_158775


namespace NUMINAMATH_CALUDE_sanda_minutes_per_day_l1587_158703

/-- The number of minutes Javier exercised per day -/
def javier_minutes_per_day : ℕ := 50

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of minutes Javier and Sanda exercised -/
def total_minutes : ℕ := 620

/-- The number of days Sanda exercised -/
def sanda_exercise_days : ℕ := 3

/-- Theorem stating that Sanda exercised 90 minutes each day -/
theorem sanda_minutes_per_day :
  (total_minutes - javier_minutes_per_day * days_in_week) / sanda_exercise_days = 90 := by
  sorry

end NUMINAMATH_CALUDE_sanda_minutes_per_day_l1587_158703


namespace NUMINAMATH_CALUDE_eight_teams_satisfy_conditions_l1587_158769

/-- The number of days in the tournament -/
def tournament_days : ℕ := 7

/-- The number of games scheduled per day -/
def games_per_day : ℕ := 4

/-- The total number of games in the tournament -/
def total_games : ℕ := tournament_days * games_per_day

/-- Function to calculate the number of games for a given number of teams -/
def games_for_teams (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that 8 teams satisfy the tournament conditions -/
theorem eight_teams_satisfy_conditions : 
  ∃ (n : ℕ), n > 0 ∧ games_for_teams n = total_games ∧ n = 8 :=
sorry

end NUMINAMATH_CALUDE_eight_teams_satisfy_conditions_l1587_158769


namespace NUMINAMATH_CALUDE_inequality_solution_l1587_158734

theorem inequality_solution (x : ℝ) :
  (x - 1) * (x - 4) * (x - 5) * (x - 7) / ((x - 3) * (x - 6) * (x - 8) * (x - 9)) > 0 →
  |x - 2| ≥ 1 →
  x ∈ Set.Ioo 3 4 ∪ Set.Ioo 6 7 ∪ Set.Ioo 8 9 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1587_158734


namespace NUMINAMATH_CALUDE_lisa_phone_expenses_l1587_158763

/-- Calculate the total cost of Lisa's phone and related expenses after three years -/
theorem lisa_phone_expenses :
  let iphone_cost : ℝ := 1000
  let monthly_contract : ℝ := 200
  let case_cost : ℝ := 0.2 * iphone_cost
  let headphones_cost : ℝ := 0.5 * case_cost
  let charger_cost : ℝ := 60
  let warranty_cost : ℝ := 150
  let discount_rate : ℝ := 0.1
  let years : ℝ := 3

  let discounted_case_cost : ℝ := case_cost * (1 - discount_rate)
  let discounted_headphones_cost : ℝ := headphones_cost * (1 - discount_rate)
  let total_contract_cost : ℝ := monthly_contract * 12 * years

  let total_cost : ℝ := iphone_cost + total_contract_cost + discounted_case_cost + 
                        discounted_headphones_cost + charger_cost + warranty_cost

  total_cost = 8680 := by sorry

end NUMINAMATH_CALUDE_lisa_phone_expenses_l1587_158763


namespace NUMINAMATH_CALUDE_A_equals_B_l1587_158761

-- Define set A
def A : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 2*y^2}

-- Define set B
def B : Set Int :=
  {n : Int | ∃ x y : Int, n = x^2 + 6*x*y + 11*y^2}

-- Theorem statement
theorem A_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l1587_158761


namespace NUMINAMATH_CALUDE_acute_angles_equal_l1587_158764

/-- A circle with a rhombus and an isosceles trapezoid inscribed around it -/
structure InscribedFigures where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Acute angle of the rhombus -/
  α : ℝ
  /-- Acute angle of the isosceles trapezoid -/
  β : ℝ
  /-- The rhombus and trapezoid are inscribed around the same circle -/
  inscribed : r > 0
  /-- The areas of the rhombus and trapezoid are equal -/
  equal_areas : (4 * r^2) / Real.sin α = (4 * r^2) / Real.sin β

/-- 
Given a rhombus and an isosceles trapezoid inscribed around the same circle with equal areas,
their acute angles are equal.
-/
theorem acute_angles_equal (fig : InscribedFigures) : fig.α = fig.β :=
  sorry

end NUMINAMATH_CALUDE_acute_angles_equal_l1587_158764


namespace NUMINAMATH_CALUDE_horner_method_v₂_l1587_158771

/-- Horner's method for a polynomial of degree 6 -/
def horner (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) (x : ℤ) : ℤ × ℤ × ℤ :=
  let v₀ := a₆
  let v₁ := v₀ * x + a₅
  let v₂ := v₁ * x + a₄
  (v₀, v₁, v₂)

/-- The polynomial f(x) = 208 + 9x² + 6x⁴ + x⁶ -/
def f (x : ℤ) : ℤ := 208 + 9*x^2 + 6*x^4 + x^6

theorem horner_method_v₂ : 
  (horner 208 0 9 0 6 0 1 (-4)).2.2 = 22 := by sorry

end NUMINAMATH_CALUDE_horner_method_v₂_l1587_158771


namespace NUMINAMATH_CALUDE_number_operations_equivalence_l1587_158701

theorem number_operations_equivalence (x : ℝ) : ((x * (5/6)) / (2/3)) - 2 = (x * (5/4)) - 2 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_equivalence_l1587_158701


namespace NUMINAMATH_CALUDE_speed_calculation_l1587_158708

/-- Given a distance of 240 km and a travel time of 6 hours, prove that the speed is 40 km/hr. -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 240) (h2 : time = 6) :
  distance / time = 40 := by
  sorry

end NUMINAMATH_CALUDE_speed_calculation_l1587_158708


namespace NUMINAMATH_CALUDE_nested_fraction_simplification_l1587_158783

theorem nested_fraction_simplification :
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_simplification_l1587_158783


namespace NUMINAMATH_CALUDE_sams_balloons_l1587_158704

theorem sams_balloons (fred_balloons : ℝ) (dan_destroyed : ℝ) (total_after : ℝ) 
  (h1 : fred_balloons = 10.0)
  (h2 : dan_destroyed = 16.0)
  (h3 : total_after = 40.0) :
  fred_balloons + (fred_balloons + dan_destroyed + total_after - fred_balloons) - dan_destroyed = total_after ∧
  fred_balloons + dan_destroyed + total_after - fred_balloons = 46.0 := by
  sorry

end NUMINAMATH_CALUDE_sams_balloons_l1587_158704


namespace NUMINAMATH_CALUDE_enemies_left_undefeated_video_game_enemies_l1587_158707

theorem enemies_left_undefeated 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (points_earned : ℕ) : ℕ :=
  let enemies_defeated := points_earned / points_per_enemy
  total_enemies - enemies_defeated

theorem video_game_enemies 
  (points_per_enemy : ℕ) 
  (total_enemies : ℕ) 
  (points_earned : ℕ) 
  (h1 : points_per_enemy = 8) 
  (h2 : total_enemies = 7) 
  (h3 : points_earned = 40) : 
  enemies_left_undefeated points_per_enemy total_enemies points_earned = 2 := by
  sorry

end NUMINAMATH_CALUDE_enemies_left_undefeated_video_game_enemies_l1587_158707


namespace NUMINAMATH_CALUDE_initial_players_count_l1587_158725

theorem initial_players_count (players_quit : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let initial_players := 8
  let remaining_players := initial_players - players_quit
  have h1 : players_quit = 3 := by sorry
  have h2 : lives_per_player = 3 := by sorry
  have h3 : total_lives = 15 := by sorry
  have h4 : remaining_players * lives_per_player = total_lives := by sorry
  initial_players

#check initial_players_count

end NUMINAMATH_CALUDE_initial_players_count_l1587_158725


namespace NUMINAMATH_CALUDE_sum_of_composite_functions_l1587_158765

def p (x : ℝ) : ℝ := |x + 1| - 3

def q (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composite_functions :
  (x_values.map (λ x => q (p x))).sum = -12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_composite_functions_l1587_158765


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_five_is_smallest_smallest_base_is_five_l1587_158756

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 4 → (∃ n : ℕ, 4 * b + 5 = n^2) → b ≥ 5 :=
by
  sorry

theorem five_is_smallest :
  ∃ n : ℕ, 4 * 5 + 5 = n^2 :=
by
  sorry

theorem smallest_base_is_five :
  ∀ b : ℕ, b > 4 ∧ (∃ n : ℕ, 4 * b + 5 = n^2) → b ≥ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_five_is_smallest_smallest_base_is_five_l1587_158756


namespace NUMINAMATH_CALUDE_jack_and_jill_probability_l1587_158746

/-- The probability of selecting both Jack and Jill when choosing 2 workers at random -/
def probability : ℚ := 1/6

/-- The number of other workers besides Jack and Jill -/
def other_workers : ℕ := 2

/-- The total number of workers -/
def total_workers : ℕ := other_workers + 2

theorem jack_and_jill_probability :
  (1 : ℚ) / (total_workers.choose 2) = probability → other_workers = 2 := by
  sorry

end NUMINAMATH_CALUDE_jack_and_jill_probability_l1587_158746


namespace NUMINAMATH_CALUDE_min_value_expression_l1587_158730

theorem min_value_expression (a : ℝ) (h : a > 1) :
  (4 / (a - 1)) + a ≥ 5 ∧ ((4 / (a - 1)) + a = 5 ↔ a = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1587_158730


namespace NUMINAMATH_CALUDE_response_rate_percentage_l1587_158782

theorem response_rate_percentage 
  (responses_needed : ℕ) 
  (questionnaires_mailed : ℕ) 
  (h1 : responses_needed = 240) 
  (h2 : questionnaires_mailed = 400) : 
  (responses_needed : ℝ) / questionnaires_mailed * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_response_rate_percentage_l1587_158782


namespace NUMINAMATH_CALUDE_john_stereo_trade_in_l1587_158728

/-- The cost of John's old stereo system -/
def old_system_cost : ℝ := 250

/-- The trade-in value as a percentage of the old system's cost -/
def trade_in_percentage : ℝ := 0.80

/-- The cost of the new stereo system before discount -/
def new_system_cost : ℝ := 600

/-- The discount percentage on the new system -/
def discount_percentage : ℝ := 0.25

/-- The amount John spent out of pocket -/
def out_of_pocket : ℝ := 250

theorem john_stereo_trade_in :
  old_system_cost * trade_in_percentage + out_of_pocket =
  new_system_cost * (1 - discount_percentage) :=
by sorry

end NUMINAMATH_CALUDE_john_stereo_trade_in_l1587_158728


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l1587_158743

def polynomial (x : ℂ) : ℂ := x^4 - 3*x^3 + 5*x^2 - 27*x - 36

theorem pure_imaginary_solutions :
  ∃ (k : ℝ), k > 0 ∧ 
  polynomial (k * Complex.I) = 0 ∧
  polynomial (-k * Complex.I) = 0 ∧
  ∀ (z : ℂ), polynomial z = 0 → z.re = 0 → z = k * Complex.I ∨ z = -k * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l1587_158743


namespace NUMINAMATH_CALUDE_min_editors_at_conference_l1587_158758

theorem min_editors_at_conference (total : Nat) (writers : Nat) (x : Nat) :
  total = 100 →
  writers = 45 →
  x ≤ 18 →
  total = writers + (55 + x) - x + 2 * x →
  55 + x ≥ 73 :=
by
  sorry

end NUMINAMATH_CALUDE_min_editors_at_conference_l1587_158758


namespace NUMINAMATH_CALUDE_marys_nickels_l1587_158736

/-- Given that Mary initially had 7 nickels and now has 12 nickels,
    prove that Mary's dad gave her 5 nickels. -/
theorem marys_nickels (initial : ℕ) (final : ℕ) (given : ℕ) :
  initial = 7 → final = 12 → given = final - initial → given = 5 := by
  sorry

end NUMINAMATH_CALUDE_marys_nickels_l1587_158736


namespace NUMINAMATH_CALUDE_interpolation_polynomial_existence_and_uniqueness_l1587_158700

theorem interpolation_polynomial_existence_and_uniqueness
  (n : ℕ) (x y : Fin n → ℝ) (h : ∀ i j : Fin n, i < j → x i < x j) :
  ∃! f : ℝ → ℝ,
    (∀ i : Fin n, f (x i) = y i) ∧
    ∃ p : Polynomial ℝ, (∀ t, f t = p.eval t) ∧ p.degree < n :=
sorry

end NUMINAMATH_CALUDE_interpolation_polynomial_existence_and_uniqueness_l1587_158700


namespace NUMINAMATH_CALUDE_hockey_league_games_l1587_158716

/-- The number of games played in a hockey league season -/
def number_of_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 19 teams, where each team faces every other team 10 times, 
    the total number of games played in the season is 1710 -/
theorem hockey_league_games : number_of_games 19 10 = 1710 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l1587_158716
