import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_equation_odd_degree_l786_78603

/-- A polynomial with real coefficients -/
def RealPolynomial := Polynomial ℝ

/-- The statement of the theorem -/
theorem polynomial_equation_odd_degree (d : ℕ) :
  (d > 0 ∧ ∃ (P Q : RealPolynomial), 
    (Polynomial.degree P = d) ∧ 
    (∀ x : ℝ, P.eval x ^ 2 + 1 = (x^2 + 1) * Q.eval x ^ 2)) ↔ 
  Odd d :=
sorry

end NUMINAMATH_CALUDE_polynomial_equation_odd_degree_l786_78603


namespace NUMINAMATH_CALUDE_parabola_chord_midpoint_tangent_intersection_l786_78631

/-- Given a parabola y² = 2px and a chord with endpoints P₁(x₁, y₁) and P₂(x₂, y₂),
    the line y = (y₁ + y₂)/2 passing through the midpoint M of the chord
    also passes through the intersection point of the tangents at P₁ and P₂. -/
theorem parabola_chord_midpoint_tangent_intersection
  (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : y₁^2 = 2*p*x₁)
  (h₂ : y₂^2 = 2*p*x₂)
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂)) :
  let midpoint_y := (y₁ + y₂) / 2
  let tangent₁ := fun x y ↦ y₁ * y = p * (x + x₁)
  let tangent₂ := fun x y ↦ y₂ * y = p * (x + x₂)
  let intersection := fun x y ↦ tangent₁ x y ∧ tangent₂ x y
  ∃ x, intersection x midpoint_y :=
sorry

end NUMINAMATH_CALUDE_parabola_chord_midpoint_tangent_intersection_l786_78631


namespace NUMINAMATH_CALUDE_composite_sum_of_power_l786_78634

theorem composite_sum_of_power (n : ℕ) (h : n ≥ 2) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4^n = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_sum_of_power_l786_78634


namespace NUMINAMATH_CALUDE_vaccine_cost_reduction_formula_correct_l786_78697

/-- Given an initial cost and an annual decrease rate, calculates the cost reduction of producing vaccines after two years. -/
def vaccine_cost_reduction (initial_cost : ℝ) (annual_decrease_rate : ℝ) : ℝ :=
  let cost_last_year := initial_cost * (1 - annual_decrease_rate)
  let cost_this_year := initial_cost * (1 - annual_decrease_rate)^2
  cost_last_year - cost_this_year

/-- Theorem stating that the vaccine cost reduction formula is correct for the given initial cost. -/
theorem vaccine_cost_reduction_formula_correct :
  ∀ (x : ℝ), vaccine_cost_reduction 5000 x = 5000 * x - 5000 * x^2 :=
by
  sorry

#eval vaccine_cost_reduction 5000 0.1

end NUMINAMATH_CALUDE_vaccine_cost_reduction_formula_correct_l786_78697


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l786_78624

theorem absolute_value_inequality (k : ℝ) :
  (∀ x : ℝ, |x + 1| - |x - 2| > k) → k ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l786_78624


namespace NUMINAMATH_CALUDE_expression_simplification_l786_78628

theorem expression_simplification (a : ℝ) (ha : a ≥ 0) :
  (((2 * (a + 1) + 2 * Real.sqrt (a^2 + 2*a)) / (3*a + 1 - 2 * Real.sqrt (a^2 + 2*a)))^(1/2 : ℝ)) -
  ((Real.sqrt (2*a + 1) - Real.sqrt a)⁻¹ * Real.sqrt (a + 2)) =
  Real.sqrt a / (Real.sqrt (2*a + 1) - Real.sqrt a) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l786_78628


namespace NUMINAMATH_CALUDE_task_selection_count_l786_78633

def num_males : ℕ := 3
def num_females : ℕ := 3
def total_students : ℕ := num_males + num_females
def num_selected : ℕ := 4

def num_single_person_tasks : ℕ := 2
def num_two_person_tasks : ℕ := 1

def selection_methods : ℕ := 144

theorem task_selection_count :
  (num_males = 3) →
  (num_females = 3) →
  (total_students = num_males + num_females) →
  (num_selected = 4) →
  (num_single_person_tasks = 2) →
  (num_two_person_tasks = 1) →
  selection_methods = 144 := by
  sorry

end NUMINAMATH_CALUDE_task_selection_count_l786_78633


namespace NUMINAMATH_CALUDE_solve_for_a_l786_78659

theorem solve_for_a : ∃ a : ℝ, (3 * 2 + 2 * a = 0) ∧ (a = -3) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l786_78659


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_135_l786_78606

/-- The coefficient of x^2 in the expansion of (3x-1)^6 -/
def coefficient_x_squared : ℕ :=
  let n : ℕ := 6
  let k : ℕ := 4
  let binomial_coefficient : ℕ := n.choose k
  let power_of_three : ℕ := 3^(n - k)
  binomial_coefficient * power_of_three

/-- Theorem stating that the coefficient of x^2 in (3x-1)^6 is 135 -/
theorem coefficient_x_squared_is_135 : coefficient_x_squared = 135 := by
  sorry

#eval coefficient_x_squared

end NUMINAMATH_CALUDE_coefficient_x_squared_is_135_l786_78606


namespace NUMINAMATH_CALUDE_carlos_laundry_loads_l786_78689

theorem carlos_laundry_loads (wash_time_per_load : ℕ) (dry_time : ℕ) (total_time : ℕ) 
  (h1 : wash_time_per_load = 45)
  (h2 : dry_time = 75)
  (h3 : total_time = 165) :
  ∃ n : ℕ, n * wash_time_per_load + dry_time = total_time ∧ n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_carlos_laundry_loads_l786_78689


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l786_78684

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 15) = 12 → x = 129 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l786_78684


namespace NUMINAMATH_CALUDE_longest_altitudes_sum_is_31_l786_78607

/-- A right triangle with sides 7, 24, and 25 -/
structure RightTriangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (side_a : a = 7)
  (side_b : b = 24)
  (side_c : c = 25)

/-- The sum of the lengths of the two longest altitudes in the right triangle -/
def longest_altitudes_sum (t : RightTriangle) : ℝ :=
  t.a + t.b

/-- Theorem: The sum of the lengths of the two longest altitudes in the given right triangle is 31 -/
theorem longest_altitudes_sum_is_31 (t : RightTriangle) :
  longest_altitudes_sum t = 31 := by
  sorry

end NUMINAMATH_CALUDE_longest_altitudes_sum_is_31_l786_78607


namespace NUMINAMATH_CALUDE_common_root_equations_l786_78637

theorem common_root_equations (p : ℝ) (h_p : p > 0) : 
  (∃ x : ℝ, 3 * x^2 - 4 * p * x + 9 = 0 ∧ x^2 - 2 * p * x + 5 = 0) ↔ p = 3 :=
by sorry

end NUMINAMATH_CALUDE_common_root_equations_l786_78637


namespace NUMINAMATH_CALUDE_consecutive_sum_26_l786_78653

theorem consecutive_sum_26 (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 26 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_26_l786_78653


namespace NUMINAMATH_CALUDE_sandy_paint_area_l786_78673

/-- The area to be painted on Sandy's bedroom wall -/
def areaToPaint (wallHeight wallLength window1Height window1Width window2Height window2Width : ℝ) : ℝ :=
  wallHeight * wallLength - (window1Height * window1Width + window2Height * window2Width)

/-- Theorem: The area Sandy needs to paint is 131 square feet -/
theorem sandy_paint_area :
  areaToPaint 10 15 3 5 2 2 = 131 := by
  sorry

end NUMINAMATH_CALUDE_sandy_paint_area_l786_78673


namespace NUMINAMATH_CALUDE_greatest_x_value_l786_78612

theorem greatest_x_value (x : ℤ) (h : (6.1 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 620) :
  x ≤ 2 ∧ ∃ y : ℤ, y > 2 → (6.1 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 620 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l786_78612


namespace NUMINAMATH_CALUDE_test_ways_count_l786_78616

/-- Represents the number of genuine items in the test. -/
def genuine_items : ℕ := 5

/-- Represents the number of defective items in the test. -/
def defective_items : ℕ := 4

/-- Represents the total number of tests conducted. -/
def total_tests : ℕ := 5

/-- Calculates the number of ways to conduct the test under the given conditions. -/
def test_ways : ℕ := sorry

/-- Theorem stating that the number of ways to conduct the test is 480. -/
theorem test_ways_count : test_ways = 480 := by sorry

end NUMINAMATH_CALUDE_test_ways_count_l786_78616


namespace NUMINAMATH_CALUDE_no_real_solution_system_l786_78688

theorem no_real_solution_system :
  ¬∃ (x y z : ℝ), (x + y - 2 - 4*x*y = 0) ∧ 
                  (y + z - 2 - 4*y*z = 0) ∧ 
                  (z + x - 2 - 4*z*x = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_system_l786_78688


namespace NUMINAMATH_CALUDE_special_integers_l786_78619

def is_special (n : ℕ) : Prop :=
  (∃ d1 d2 : ℕ, 1 < d1 ∧ d1 < n ∧ d1 ∣ n ∧
                1 < d2 ∧ d2 < n ∧ d2 ∣ n ∧
                d1 ≠ d2) ∧
  (∀ d1 d2 : ℕ, 1 < d1 ∧ d1 < n ∧ d1 ∣ n →
                1 < d2 ∧ d2 < n ∧ d2 ∣ n →
                (d1 - d2) ∣ n ∨ (d2 - d1) ∣ n)

theorem special_integers :
  ∀ n : ℕ, is_special n ↔ n = 6 ∨ n = 8 ∨ n = 12 :=
by sorry

end NUMINAMATH_CALUDE_special_integers_l786_78619


namespace NUMINAMATH_CALUDE_question_mark_value_l786_78670

theorem question_mark_value (x : ℝ) : (x * 74) / 30 = 1938.8 → x = 786 := by
  sorry

end NUMINAMATH_CALUDE_question_mark_value_l786_78670


namespace NUMINAMATH_CALUDE_regular_ticket_cost_l786_78671

theorem regular_ticket_cost (total_tickets : ℕ) (senior_ticket_cost : ℕ) (total_sales : ℕ) (regular_tickets_sold : ℕ) :
  total_tickets = 65 →
  senior_ticket_cost = 10 →
  total_sales = 855 →
  regular_tickets_sold = 41 →
  ∃ (regular_ticket_cost : ℕ),
    regular_ticket_cost * regular_tickets_sold + senior_ticket_cost * (total_tickets - regular_tickets_sold) = total_sales ∧
    regular_ticket_cost = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_ticket_cost_l786_78671


namespace NUMINAMATH_CALUDE_rectangle_dg_length_l786_78695

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

theorem rectangle_dg_length :
  ∀ (r1 r2 r3 : Rectangle),
  area r1 = area r2 ∧ area r2 = area r3 ∧   -- Equal areas
  r1.width = 23 ∧                           -- BC = 23
  r2.width = r1.height ∧                    -- DE = AB
  r3.width = r1.height - r2.height ∧        -- CE = AB - DE
  r3.height = r1.width →                    -- CH = BC
  r2.height = 552                           -- DG = 552
  := by sorry

end NUMINAMATH_CALUDE_rectangle_dg_length_l786_78695


namespace NUMINAMATH_CALUDE_josephine_milk_sales_l786_78601

/-- The total amount of milk sold by Josephine on Sunday morning -/
def total_milk_sold (container_2L : ℕ) (container_075L : ℕ) (container_05L : ℕ) : ℝ :=
  (container_2L * 2) + (container_075L * 0.75) + (container_05L * 0.5)

/-- Theorem stating that Josephine sold 10 liters of milk given the specified containers -/
theorem josephine_milk_sales : total_milk_sold 3 2 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_josephine_milk_sales_l786_78601


namespace NUMINAMATH_CALUDE_emily_small_gardens_l786_78690

/-- The number of small gardens Emily had -/
def num_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

/-- Proof that Emily had 3 small gardens -/
theorem emily_small_gardens :
  num_small_gardens 41 29 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_emily_small_gardens_l786_78690


namespace NUMINAMATH_CALUDE_matrix_condition_l786_78643

variable (a b c d : ℂ)

def N : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]]

theorem matrix_condition (h1 : N a b c d ^ 2 = 1) (h2 : a * b * c * d = 1) :
  a^4 + b^4 + c^4 + d^4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_matrix_condition_l786_78643


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l786_78617

theorem max_value_inequality (a b c d : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0)
  (hsum : a + b + c + d = 100) : 
  (a / (b + 7))^(1/3) + (b / (c + 7))^(1/3) + (c / (d + 7))^(1/3) + (d / (a + 7))^(1/3) ≤ 2 * 25^(1/3) :=
by sorry

theorem max_value_achievable : 
  ∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = 100 ∧
  (a / (b + 7))^(1/3) + (b / (c + 7))^(1/3) + (c / (d + 7))^(1/3) + (d / (a + 7))^(1/3) = 2 * 25^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l786_78617


namespace NUMINAMATH_CALUDE_complex_eighth_power_sum_l786_78678

theorem complex_eighth_power_sum : (((1 : ℂ) + Complex.I * Real.sqrt 3) / 2) ^ 8 + 
  (((1 : ℂ) - Complex.I * Real.sqrt 3) / 2) ^ 8 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_eighth_power_sum_l786_78678


namespace NUMINAMATH_CALUDE_smallest_common_factor_l786_78669

theorem smallest_common_factor : ∃ (n : ℕ), n > 0 ∧ n = 42 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    (∀ (k : ℕ), k > 1 → ¬(k ∣ (11 * m - 3) ∧ k ∣ (8 * m + 4)))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (11 * n - 3) ∧ k ∣ (8 * n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l786_78669


namespace NUMINAMATH_CALUDE_parallelogram_roots_l786_78608

theorem parallelogram_roots (b : ℝ) : 
  (∃ (z₁ z₂ z₃ z₄ : ℂ), 
    z₁^4 - 8*z₁^3 + 13*b*z₁^2 - 5*(2*b^2 + b - 2)*z₁ + 4 = 0 ∧
    z₂^4 - 8*z₂^3 + 13*b*z₂^2 - 5*(2*b^2 + b - 2)*z₂ + 4 = 0 ∧
    z₃^4 - 8*z₃^3 + 13*b*z₃^2 - 5*(2*b^2 + b - 2)*z₃ + 4 = 0 ∧
    z₄^4 - 8*z₄^3 + 13*b*z₄^2 - 5*(2*b^2 + b - 2)*z₄ + 4 = 0 ∧
    z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    (z₁ - z₂ = z₄ - z₃) ∧ (z₁ - z₃ = z₄ - z₂)) ↔ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_roots_l786_78608


namespace NUMINAMATH_CALUDE_right_triangle_distance_theorem_l786_78636

theorem right_triangle_distance_theorem (a b : ℝ) (ha : a = 9) (hb : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let area := a * b / 2
  let r := area / s
  let centroid_dist := 2 * c / 3
  1 = centroid_dist - r :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_distance_theorem_l786_78636


namespace NUMINAMATH_CALUDE_julie_age_is_fifteen_l786_78610

/-- Represents Julie's age and earnings during a four-month period --/
structure JulieData where
  hoursPerDay : ℕ
  hourlyRatePerAge : ℚ
  workDays : ℕ
  totalEarnings : ℚ

/-- Calculates Julie's age at the end of the four-month period --/
def calculateAge (data : JulieData) : ℕ :=
  sorry

/-- Theorem stating that Julie's age at the end of the period is 15 --/
theorem julie_age_is_fifteen (data : JulieData) 
  (h1 : data.hoursPerDay = 3)
  (h2 : data.hourlyRatePerAge = 3/4)
  (h3 : data.workDays = 60)
  (h4 : data.totalEarnings = 810) :
  calculateAge data = 15 := by
  sorry

end NUMINAMATH_CALUDE_julie_age_is_fifteen_l786_78610


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l786_78635

/-- Given a cuboid with edges x, 5, and 6, and volume 120, prove x = 4 -/
theorem cuboid_edge_length (x : ℝ) : x * 5 * 6 = 120 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l786_78635


namespace NUMINAMATH_CALUDE_equation_solutions_l786_78656

theorem equation_solutions :
  (∀ x : ℝ, 4 * x * (2 * x - 1) = 3 * (2 * x - 1) → x = 1/2 ∨ x = 3/4) ∧
  (∀ x : ℝ, x^2 + 2*x - 2 = 0 → x = -1 + Real.sqrt 3 ∨ x = -1 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l786_78656


namespace NUMINAMATH_CALUDE_valid_sets_count_l786_78676

/-- Represents a family tree with 4 generations -/
structure FamilyTree :=
  (root : Unit)
  (gen1 : Fin 3)
  (gen2 : Fin 6)
  (gen3 : Fin 6)

/-- Represents a set of women from the family tree -/
def WomenSet := FamilyTree → Bool

/-- Checks if a set is valid (no woman and her daughter are both in the set) -/
def is_valid_set (s : WomenSet) : Bool :=
  sorry

/-- Counts the number of valid sets -/
def count_valid_sets : Nat :=
  sorry

/-- The main theorem to prove -/
theorem valid_sets_count : count_valid_sets = 793 :=
  sorry

end NUMINAMATH_CALUDE_valid_sets_count_l786_78676


namespace NUMINAMATH_CALUDE_meaningful_iff_greater_than_one_l786_78662

-- Define the condition for the expression to be meaningful
def is_meaningful (x : ℝ) : Prop := x > 1

-- Theorem stating that the expression is meaningful if and only if x > 1
theorem meaningful_iff_greater_than_one (x : ℝ) :
  is_meaningful x ↔ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_iff_greater_than_one_l786_78662


namespace NUMINAMATH_CALUDE_tan_value_second_quadrant_l786_78683

/-- Given that α is an angle in the second quadrant and sin(π - α) = 3/5, prove that tan(α) = -3/4 -/
theorem tan_value_second_quadrant (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.sin (π - α) = 3/5) : 
  Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_second_quadrant_l786_78683


namespace NUMINAMATH_CALUDE_total_caffeine_consumption_l786_78682

/-- Calculates the total caffeine consumption given the specifications of three drinks and a pill -/
theorem total_caffeine_consumption
  (drink1_oz : ℝ)
  (drink1_caffeine : ℝ)
  (drink2_oz : ℝ)
  (drink2_caffeine_multiplier : ℝ)
  (drink3_caffeine_per_ml : ℝ)
  (drink3_ml_consumed : ℝ) :
  drink1_oz = 12 →
  drink1_caffeine = 250 →
  drink2_oz = 8 →
  drink2_caffeine_multiplier = 3 →
  drink3_caffeine_per_ml = 18 →
  drink3_ml_consumed = 150 →
  let drink2_caffeine := (drink1_caffeine / drink1_oz) * drink2_caffeine_multiplier * drink2_oz
  let drink3_caffeine := drink3_caffeine_per_ml * drink3_ml_consumed
  let pill_caffeine := drink1_caffeine + drink2_caffeine + drink3_caffeine
  drink1_caffeine + drink2_caffeine + drink3_caffeine + pill_caffeine = 6900 := by
  sorry


end NUMINAMATH_CALUDE_total_caffeine_consumption_l786_78682


namespace NUMINAMATH_CALUDE_tournament_teams_l786_78642

theorem tournament_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_tournament_teams_l786_78642


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l786_78611

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 20)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 168 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l786_78611


namespace NUMINAMATH_CALUDE_choir_composition_l786_78655

theorem choir_composition (initial_total : ℕ) : 
  let initial_girls : ℕ := (6 * initial_total) / 10
  let final_total : ℕ := initial_total + 6 - 4 - 2
  let final_girls : ℕ := initial_girls - 4
  (2 * final_girls = final_total) → initial_girls = 24 := by
sorry

end NUMINAMATH_CALUDE_choir_composition_l786_78655


namespace NUMINAMATH_CALUDE_f_sum_positive_l786_78699

def f (x : ℝ) : ℝ := x^2015

theorem f_sum_positive (a b : ℝ) (h1 : a + b > 0) (h2 : a * b < 0) : 
  f a + f b > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_positive_l786_78699


namespace NUMINAMATH_CALUDE_bowling_ball_surface_area_l786_78645

theorem bowling_ball_surface_area :
  let diameter : ℝ := 9
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 81 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_surface_area_l786_78645


namespace NUMINAMATH_CALUDE_domain_union_sqrt_ln_l786_78609

-- Define the domains M and N
def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x < 1}

-- State the theorem
theorem domain_union_sqrt_ln :
  M ∪ N = Set.Iio 1 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_domain_union_sqrt_ln_l786_78609


namespace NUMINAMATH_CALUDE_circle_equation_proof_l786_78666

/-- The circle C with equation x^2 + y^2 + 10x + 10y = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 10*x + 10*y = 0

/-- The point A with coordinates (0, 6) -/
def point_A : ℝ × ℝ := (0, 6)

/-- The desired circle passing through A and tangent to C at the origin -/
def desired_circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = 18

theorem circle_equation_proof :
  ∀ x y : ℝ,
  circle_C 0 0 ∧  -- C passes through the origin
  desired_circle (point_A.1) (point_A.2) ∧  -- Desired circle passes through A
  (∃ t : ℝ, t ≠ 0 ∧ 
    (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ 
      ∀ x' y' : ℝ, 
      (x' - 0)^2 + (y' - 0)^2 < δ^2 → 
      (circle_C x' y' ∧ desired_circle x' y') ∨ 
      (¬circle_C x' y' ∧ ¬desired_circle x' y'))) →  -- Tangency condition
  desired_circle x y  -- The equation of the desired circle
:= by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l786_78666


namespace NUMINAMATH_CALUDE_smallest_integer_y_l786_78618

theorem smallest_integer_y (y : ℤ) : (∀ z : ℤ, z < y → 3 * z - 6 ≥ 15) ∧ 3 * y - 6 < 15 ↔ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l786_78618


namespace NUMINAMATH_CALUDE_crayons_lost_theorem_l786_78640

/-- The number of crayons Paul lost or gave away -/
def crayons_lost_or_given_away (initial_crayons remaining_crayons : ℕ) : ℕ :=
  initial_crayons - remaining_crayons

/-- Theorem: The number of crayons lost or given away is equal to the difference between
    the initial number of crayons and the remaining number of crayons -/
theorem crayons_lost_theorem (initial_crayons remaining_crayons : ℕ) 
  (h : initial_crayons ≥ remaining_crayons) :
  crayons_lost_or_given_away initial_crayons remaining_crayons = initial_crayons - remaining_crayons :=
by
  sorry

#eval crayons_lost_or_given_away 479 134

end NUMINAMATH_CALUDE_crayons_lost_theorem_l786_78640


namespace NUMINAMATH_CALUDE_age_sum_proof_l786_78672

theorem age_sum_proof (p q : ℕ) : 
  (p : ℚ) / q = 3 / 4 →
  p - 8 = (q - 8) / 2 →
  p + q = 28 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l786_78672


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l786_78658

/-- The y-intercept of the line 2x - 3y = 6 is -2 -/
theorem y_intercept_of_line (x y : ℝ) : 2 * x - 3 * y = 6 → x = 0 → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l786_78658


namespace NUMINAMATH_CALUDE_domino_coverage_iff_even_uncoverable_boards_l786_78623

/-- Represents a checkerboard -/
structure Checkerboard where
  squares : ℕ

/-- Predicate to determine if a checkerboard can be fully covered by dominoes -/
def can_be_covered (board : Checkerboard) : Prop :=
  Even board.squares

theorem domino_coverage_iff_even (board : Checkerboard) :
  can_be_covered board ↔ Even board.squares :=
sorry

/-- 6x4 rectangular board -/
def board_6x4 : Checkerboard :=
  ⟨6 * 4⟩

/-- 5x5 square board -/
def board_5x5 : Checkerboard :=
  ⟨5 * 5⟩

/-- L-shaped board (5x5 with 2x2 removed) -/
def board_L : Checkerboard :=
  ⟨5 * 5 - 2 * 2⟩

/-- 3x7 rectangular board -/
def board_3x7 : Checkerboard :=
  ⟨3 * 7⟩

/-- Plus-shaped board (3x3 with 1x3 extension) -/
def board_plus : Checkerboard :=
  ⟨3 * 3 + 1 * 3⟩

theorem uncoverable_boards :
  ¬can_be_covered board_5x5 ∧
  ¬can_be_covered board_L ∧
  ¬can_be_covered board_3x7 :=
sorry

end NUMINAMATH_CALUDE_domino_coverage_iff_even_uncoverable_boards_l786_78623


namespace NUMINAMATH_CALUDE_caroline_lassis_l786_78646

/-- Represents the number of lassis that can be made with given ingredients -/
def max_lassis (initial_lassis initial_mangoes initial_coconuts available_mangoes available_coconuts : ℚ) : ℚ :=
  min 
    (available_mangoes * (initial_lassis / initial_mangoes))
    (available_coconuts * (initial_lassis / initial_coconuts))

/-- Theorem stating that Caroline can make 55 lassis with the given ingredients -/
theorem caroline_lassis : 
  max_lassis 11 2 4 12 20 = 55 := by
  sorry

#eval max_lassis 11 2 4 12 20

end NUMINAMATH_CALUDE_caroline_lassis_l786_78646


namespace NUMINAMATH_CALUDE_waiter_income_fraction_l786_78692

theorem waiter_income_fraction (salary : ℚ) (tips : ℚ) (income : ℚ) : 
  tips = (3 : ℚ) / 4 * salary →
  income = salary + tips →
  tips / income = (3 : ℚ) / 7 := by
  sorry

end NUMINAMATH_CALUDE_waiter_income_fraction_l786_78692


namespace NUMINAMATH_CALUDE_cos_alpha_value_l786_78661

theorem cos_alpha_value (α β : Real) 
  (h1 : -π/2 < α ∧ α < π/2)
  (h2 : 2 * Real.tan β = Real.tan (2 * α))
  (h3 : Real.tan (β - α) = -2 * Real.sqrt 2) :
  Real.cos α = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l786_78661


namespace NUMINAMATH_CALUDE_extreme_value_implies_n_eq_9_l786_78663

/-- The function f(x) = x^3 + 6x^2 + nx + 4 -/
def f (n : ℝ) (x : ℝ) : ℝ := x^3 + 6*x^2 + n*x + 4

/-- f has an extreme value at x = -1 -/
def has_extreme_value_at_neg_one (n : ℝ) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 
    x ≠ -1 ∧ |x + 1| < ε → (f n x - f n (-1)) * (x + 1) ≤ 0

theorem extreme_value_implies_n_eq_9 :
  ∀ (n : ℝ), has_extreme_value_at_neg_one n → n = 9 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_n_eq_9_l786_78663


namespace NUMINAMATH_CALUDE_abs_diff_of_sum_and_product_l786_78652

theorem abs_diff_of_sum_and_product (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (prod_eq : x * y = 221) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_abs_diff_of_sum_and_product_l786_78652


namespace NUMINAMATH_CALUDE_triangular_array_sum_recurrence_l786_78657

def triangular_array_sum (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n+1 => 2 * triangular_array_sum n + 2 * n

theorem triangular_array_sum_recurrence (n : ℕ) (h : n ≥ 2) :
  triangular_array_sum n = 2 * triangular_array_sum (n-1) + 2 * (n-1) :=
by sorry

#eval triangular_array_sum 20

end NUMINAMATH_CALUDE_triangular_array_sum_recurrence_l786_78657


namespace NUMINAMATH_CALUDE_variety_show_arrangements_l786_78674

def dance_song_count : ℕ := 3
def comedy_skit_count : ℕ := 2
def cross_talk_count : ℕ := 1

def non_adjacent_arrangements (ds c ct : ℕ) : ℕ :=
  ds.factorial * (2 * ds.factorial + c.choose 1 * (ds - 1).factorial * (ds - 1).factorial)

theorem variety_show_arrangements :
  non_adjacent_arrangements dance_song_count comedy_skit_count cross_talk_count = 120 := by
  sorry

end NUMINAMATH_CALUDE_variety_show_arrangements_l786_78674


namespace NUMINAMATH_CALUDE_parallel_condition_l786_78654

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The vector a as a function of k -/
def a (k : ℝ) : ℝ × ℝ := (k^2, k + 1)

/-- The vector b as a function of k -/
def b (k : ℝ) : ℝ × ℝ := (k, 4)

/-- Theorem stating the conditions for parallelism of vectors a and b -/
theorem parallel_condition (k : ℝ) : 
  are_parallel (a k) (b k) ↔ k = 0 ∨ k = 1/3 := by sorry

end NUMINAMATH_CALUDE_parallel_condition_l786_78654


namespace NUMINAMATH_CALUDE_rectangle_diagonal_triangle_area_l786_78691

/-- The area of a right triangle formed by the diagonal of a rectangle. -/
theorem rectangle_diagonal_triangle_area
  (length width : ℝ)
  (h_length : length = 35)
  (h_width : width = 48) :
  (1 / 2) * length * width = 840 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_triangle_area_l786_78691


namespace NUMINAMATH_CALUDE_choose_five_from_ten_l786_78675

theorem choose_five_from_ten : Nat.choose 10 5 = 252 := by sorry

end NUMINAMATH_CALUDE_choose_five_from_ten_l786_78675


namespace NUMINAMATH_CALUDE_probability_theorem_l786_78665

/-- The number of boys in the group -/
def num_boys : ℕ := 5

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The number of students to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly one girl -/
def prob_one_girl : ℚ := 15 / 28

/-- The probability of selecting exactly one girl given that at least one girl is selected -/
def prob_one_girl_given_at_least_one : ℚ := 5 / 6

/-- Theorem stating the probabilities for the given scenario -/
theorem probability_theorem :
  (prob_one_girl = 15 / 28) ∧
  (prob_one_girl_given_at_least_one = 5 / 6) :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l786_78665


namespace NUMINAMATH_CALUDE_swimmer_passes_l786_78687

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  delay : ℝ

/-- Calculates the number of times swimmers pass each other --/
def count_passes (pool_length : ℝ) (time : ℝ) (swimmer_a : Swimmer) (swimmer_b : Swimmer) : ℕ :=
  sorry

/-- Theorem stating the number of passes in the given scenario --/
theorem swimmer_passes :
  let pool_length : ℝ := 120
  let total_time : ℝ := 900
  let swimmer_a : Swimmer := { speed := 3, delay := 0 }
  let swimmer_b : Swimmer := { speed := 4, delay := 10 }
  count_passes pool_length total_time swimmer_a swimmer_b = 38 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_passes_l786_78687


namespace NUMINAMATH_CALUDE_sum_of_sqrt_odd_sums_equals_15_l786_78667

def odd_sum (n : ℕ) : ℕ := n^2

theorem sum_of_sqrt_odd_sums_equals_15 :
  Real.sqrt (odd_sum 1) + Real.sqrt (odd_sum 2) + Real.sqrt (odd_sum 3) + 
  Real.sqrt (odd_sum 4) + Real.sqrt (odd_sum 5) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_odd_sums_equals_15_l786_78667


namespace NUMINAMATH_CALUDE_marys_characters_l786_78648

theorem marys_characters (total : ℕ) (a b c d e f : ℕ) : 
  total = 120 →
  a = total / 3 →
  b = (total - a) / 4 →
  c = (total - a - b) / 5 →
  d + e + f = total - a - b - c →
  d = 3 * e →
  e = f / 2 →
  d = 24 := by sorry

end NUMINAMATH_CALUDE_marys_characters_l786_78648


namespace NUMINAMATH_CALUDE_rainbow_pencils_count_l786_78622

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of people who have the color box -/
def num_people : ℕ := 6

/-- The total number of pencils -/
def total_pencils : ℕ := rainbow_colors * num_people

theorem rainbow_pencils_count : total_pencils = 42 := by
  sorry

end NUMINAMATH_CALUDE_rainbow_pencils_count_l786_78622


namespace NUMINAMATH_CALUDE_bird_watching_average_l786_78614

theorem bird_watching_average : 
  let marcus_birds : ℕ := 7
  let humphrey_birds : ℕ := 11
  let darrel_birds : ℕ := 9
  let isabella_birds : ℕ := 15
  let total_birds : ℕ := marcus_birds + humphrey_birds + darrel_birds + isabella_birds
  let num_watchers : ℕ := 4
  (total_birds : ℚ) / num_watchers = 10.5 := by sorry

end NUMINAMATH_CALUDE_bird_watching_average_l786_78614


namespace NUMINAMATH_CALUDE_division_by_fraction_fifteen_divided_by_two_thirds_result_is_twentytwo_point_five_l786_78677

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b :=
by sorry

theorem fifteen_divided_by_two_thirds :
  15 / (2 / 3) = 45 / 2 :=
by sorry

theorem result_is_twentytwo_point_five :
  15 / (2 / 3) = 22.5 :=
by sorry

end NUMINAMATH_CALUDE_division_by_fraction_fifteen_divided_by_two_thirds_result_is_twentytwo_point_five_l786_78677


namespace NUMINAMATH_CALUDE_plane_intersection_line_properties_l786_78613

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (intersect_at : Plane → Plane → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (intersects : Line → Line → Point → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem plane_intersection_line_properties
  (α β : Plane) (l m : Line) (P : Point)
  (h1 : intersect_at α β l)
  (h2 : contains α m)
  (h3 : intersects m l P) :
  (∃ (n : Line), contains β n ∧ perpendicular m n) ∧
  (¬∃ (k : Line), contains β k ∧ parallel m k) :=
sorry

end NUMINAMATH_CALUDE_plane_intersection_line_properties_l786_78613


namespace NUMINAMATH_CALUDE_ball_game_attendance_l786_78626

/-- The number of children at a ball game -/
def num_children : ℕ :=
  let num_adults : ℕ := 10
  let adult_ticket_price : ℕ := 8
  let child_ticket_price : ℕ := 4
  let total_bill : ℕ := 124
  let adult_cost : ℕ := num_adults * adult_ticket_price
  let child_cost : ℕ := total_bill - adult_cost
  child_cost / child_ticket_price

theorem ball_game_attendance : num_children = 11 := by
  sorry

end NUMINAMATH_CALUDE_ball_game_attendance_l786_78626


namespace NUMINAMATH_CALUDE_right_triangle_area_l786_78685

theorem right_triangle_area (h : ℝ) (h_positive : h > 0) :
  let a := h * Real.sqrt 2
  let b := h * Real.sqrt 2
  let c := 2 * h * Real.sqrt 2
  h = 4 →
  (1 / 2 : ℝ) * c * h = 16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l786_78685


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l786_78639

theorem simplify_and_rationalize (x : ℝ) : 
  (1 : ℝ) / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l786_78639


namespace NUMINAMATH_CALUDE_min_x_over_y_for_system_l786_78627

/-- Given a system of equations, this theorem states that the minimum value of x/y
    for all solutions (x, y) is equal to (-1 - √217) / 12. -/
theorem min_x_over_y_for_system (x y : ℝ) :
  x^3 + 3*y^3 = 11 →
  x^2*y + x*y^2 = 6 →
  ∃ (min_val : ℝ), (∀ (x' y' : ℝ), x'^3 + 3*y'^3 = 11 → x'^2*y' + x'*y'^2 = 6 → x' / y' ≥ min_val) ∧
                   min_val = (-1 - Real.sqrt 217) / 12 :=
sorry

end NUMINAMATH_CALUDE_min_x_over_y_for_system_l786_78627


namespace NUMINAMATH_CALUDE_valid_factorization_l786_78647

theorem valid_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_valid_factorization_l786_78647


namespace NUMINAMATH_CALUDE_min_omega_value_l786_78602

theorem min_omega_value (f : ℝ → ℝ) (ω φ T : ℝ) : 
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < π →
  T > 0 →
  (∀ t > 0, (∀ x, f (x + t) = f x) → T ≤ t) →
  f T = Real.sqrt 3 / 2 →
  f (π / 9) = 0 →
  3 ≤ ω ∧ ∀ ω' ≥ 0, (
    (∀ x, Real.cos (ω' * x + φ) = Real.cos (ω * x + φ)) →
    (Real.cos (ω' * T + φ) = Real.sqrt 3 / 2) →
    (Real.cos (ω' * π / 9 + φ) = 0) →
    ω' ≥ 3
  ) := by sorry


end NUMINAMATH_CALUDE_min_omega_value_l786_78602


namespace NUMINAMATH_CALUDE_two_digit_division_problem_l786_78605

theorem two_digit_division_problem :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 
  (∃ q r : ℕ, q = 9 ∧ r = 6 ∧ n = q * (n % 10) + r) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_division_problem_l786_78605


namespace NUMINAMATH_CALUDE_problem_solution_l786_78668

theorem problem_solution (x y : ℝ) (hx : x = 12) (hy : y = 7) : 
  (x - y) * (x + y) = 95 ∧ (x + y)^2 = 361 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l786_78668


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l786_78621

theorem cubic_equation_roots (P : ℤ) : 
  (∃ x y z : ℤ, 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    x^3 - 10*x^2 + P*x - 30 = 0 ∧
    y^3 - 10*y^2 + P*y - 30 = 0 ∧
    z^3 - 10*z^2 + P*z - 30 = 0 ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  P = 31 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l786_78621


namespace NUMINAMATH_CALUDE_find_a_value_l786_78641

theorem find_a_value (x y a : ℝ) 
  (h1 : x = 2) 
  (h2 : y = 1) 
  (h3 : a * x - y = 3) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l786_78641


namespace NUMINAMATH_CALUDE_projection_of_a_onto_b_l786_78632

/-- Prove that the projection of vector a = (3, 4) onto vector b = (0, 1) results in the vector (0, 4) -/
theorem projection_of_a_onto_b :
  let a : Fin 2 → ℝ := ![3, 4]
  let b : Fin 2 → ℝ := ![0, 1]
  let proj := (a • b) / (b • b) • b
  proj = ![0, 4] := by sorry

end NUMINAMATH_CALUDE_projection_of_a_onto_b_l786_78632


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l786_78649

/-- Permutation function -/
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The problem statement -/
theorem permutation_equation_solution :
  ∃! (x : ℕ), x > 0 ∧ x ≤ 5 ∧ A 5 x = 2 * A 6 (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l786_78649


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l786_78615

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, -x^2 + c*x + 10 < 0 ↔ x < 2 ∨ x > 8) → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l786_78615


namespace NUMINAMATH_CALUDE_smallest_first_term_divisible_by_11_l786_78680

-- Define the arithmetic sequence
def arithmeticSequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

-- Define the sum of seven consecutive terms starting from k
def sumSevenTerms (k : ℕ) : ℤ := 
  (arithmeticSequence k) + 
  (arithmeticSequence (k + 1)) + 
  (arithmeticSequence (k + 2)) + 
  (arithmeticSequence (k + 3)) + 
  (arithmeticSequence (k + 4)) + 
  (arithmeticSequence (k + 5)) + 
  (arithmeticSequence (k + 6))

-- The theorem to prove
theorem smallest_first_term_divisible_by_11 :
  ∃ k : ℕ, (sumSevenTerms k) % 11 = 0 ∧ 
  ∀ m : ℕ, m < k → (sumSevenTerms m) % 11 ≠ 0 ∧
  arithmeticSequence k = 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_first_term_divisible_by_11_l786_78680


namespace NUMINAMATH_CALUDE_percentage_increase_l786_78620

theorem percentage_increase (x : ℝ) (h1 : x > 40) (h2 : x = 48) :
  (x - 40) / 40 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l786_78620


namespace NUMINAMATH_CALUDE_scientific_notation_of_43300000_l786_78679

theorem scientific_notation_of_43300000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 43300000 = a * (10 : ℝ) ^ n ∧ a = 4.33 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_43300000_l786_78679


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l786_78600

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles 
  (c1 : ℝ × ℝ → ℝ) 
  (c2 : ℝ × ℝ → ℝ) 
  (h1 : ∀ x y, c1 (x, y) = x^2 - 6*x + y^2 - 8*y + 4)
  (h2 : ∀ x y, c2 (x, y) = x^2 + 8*x + y^2 + 12*y + 36) :
  let d := Real.sqrt 149 - Real.sqrt 21 - 4
  ∃ p1 p2, c1 p1 = 0 ∧ c2 p2 = 0 ∧ 
    d = Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) - 
        (Real.sqrt 21 + 4) ∧
    ∀ q1 q2, c1 q1 = 0 → c2 q2 = 0 → 
      d ≤ Real.sqrt ((q1.1 - q2.1)^2 + (q1.2 - q2.2)^2) - 
          (Real.sqrt 21 + 4) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l786_78600


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l786_78694

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- M is defined as the square root of 36^49 * 49^36 -/
def M : ℕ := sorry

/-- Theorem stating that the sum of digits of M is 37 -/
theorem sum_of_digits_M : sum_of_digits M = 37 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l786_78694


namespace NUMINAMATH_CALUDE_cannot_eat_all_except_central_l786_78696

/-- Represents a 3D coordinate within the cheese cube -/
structure Coordinate where
  x : Fin 3
  y : Fin 3
  z : Fin 3

/-- Represents the color of a unit cube -/
inductive Color
  | White
  | Black

/-- The cheese cube -/
def CheeseCube := Fin 3 → Fin 3 → Fin 3 → Color

/-- Determines if two coordinates are adjacent (share a face) -/
def isAdjacent (c1 c2 : Coordinate) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ (c1.z = c2.z + 1 ∨ c1.z + 1 = c2.z)) ∨
  (c1.x = c2.x ∧ c1.z = c2.z ∧ (c1.y = c2.y + 1 ∨ c1.y + 1 = c2.y)) ∨
  (c1.y = c2.y ∧ c1.z = c2.z ∧ (c1.x = c2.x + 1 ∨ c1.x + 1 = c2.x))

/-- Assigns a color to each coordinate based on the sum of its components -/
def colorCube : CheeseCube :=
  fun x y z => if (x.val + y.val + z.val) % 2 = 0 then Color.White else Color.Black

/-- The central cube coordinate -/
def centralCube : Coordinate := ⟨1, 1, 1⟩

/-- Theorem stating that it's impossible to eat all cubes except the central one -/
theorem cannot_eat_all_except_central :
  ¬∃ (path : List Coordinate),
    path.Nodup ∧
    path.length = 26 ∧
    (∀ i, i ∈ path → i ≠ centralCube) ∧
    (∀ i j, i ∈ path → j ∈ path → i ≠ j → isAdjacent i j) :=
  sorry

end NUMINAMATH_CALUDE_cannot_eat_all_except_central_l786_78696


namespace NUMINAMATH_CALUDE_exam_scoring_l786_78644

theorem exam_scoring (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℕ) 
  (h1 : total_questions = 80)
  (h2 : correct_answers = 40)
  (h3 : total_marks = 120) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 := by
  sorry

end NUMINAMATH_CALUDE_exam_scoring_l786_78644


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l786_78638

/-- Proves that (x^4 + 12x^2 + 144)(x^2 - 12) = x^6 - 1728 for all real x. -/
theorem polynomial_multiplication (x : ℝ) : 
  (x^4 + 12*x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l786_78638


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l786_78629

theorem no_positive_integer_solutions
  (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 0) :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * (x + 1) = p^(2*n) * y * (y + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l786_78629


namespace NUMINAMATH_CALUDE_unique_n_reaches_two_l786_78664

def g (n : ℤ) : ℤ := 
  if n % 2 = 1 then n^2 - 2*n + 2 else 2*n

def iterateG (n : ℤ) (k : ℕ) : ℤ :=
  match k with
  | 0 => n
  | k+1 => g (iterateG n k)

theorem unique_n_reaches_two :
  ∃! n : ℤ, 1 ≤ n ∧ n ≤ 100 ∧ ∃ k : ℕ, iterateG n k = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_n_reaches_two_l786_78664


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_equals_4_l786_78625

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_k_equals_4
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = 1)
  (h_diff : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d)
  (h_k : ∃ k : ℕ, a k = 7) :
  ∃ k : ℕ, a k = 7 ∧ k = 4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_equals_4_l786_78625


namespace NUMINAMATH_CALUDE_half_of_five_bananas_worth_l786_78630

-- Define the worth of bananas in terms of oranges
def banana_orange_ratio : ℚ := 8 / (2/3 * 10)

-- Theorem statement
theorem half_of_five_bananas_worth (banana_orange_ratio : ℚ) :
  banana_orange_ratio = 8 / (2/3 * 10) →
  (1/2 * 5) * banana_orange_ratio = 3 := by
  sorry

end NUMINAMATH_CALUDE_half_of_five_bananas_worth_l786_78630


namespace NUMINAMATH_CALUDE_graduation_ceremony_chairs_l786_78660

/-- The number of graduates at a ceremony -/
def graduates : ℕ := 50

/-- The number of parents per graduate -/
def parents_per_graduate : ℕ := 2

/-- The number of teachers attending -/
def teachers : ℕ := 20

/-- The number of administrators attending -/
def administrators : ℕ := teachers / 2

/-- The total number of chairs available -/
def total_chairs : ℕ := 180

theorem graduation_ceremony_chairs :
  graduates + graduates * parents_per_graduate + teachers + administrators = total_chairs :=
sorry

end NUMINAMATH_CALUDE_graduation_ceremony_chairs_l786_78660


namespace NUMINAMATH_CALUDE_exponential_increasing_condition_l786_78698

theorem exponential_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → a^x < a^y) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_exponential_increasing_condition_l786_78698


namespace NUMINAMATH_CALUDE_circleplus_problem_l786_78604

-- Define the ⊕ operation
def circleplus (a b : ℚ) : ℚ := (a * b) / (a + b)

-- State the theorem
theorem circleplus_problem : 
  circleplus (circleplus 3 5) (circleplus 5 4) = 60 / 59 := by
  sorry

end NUMINAMATH_CALUDE_circleplus_problem_l786_78604


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l786_78681

/-- A geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 1 + a 2 + a 3 = 1 ∧
  a 2 + a 3 + a 4 = 2

/-- The sum of the 6th, 7th, and 8th terms equals 32 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 6 + a 7 + a 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l786_78681


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l786_78650

/-- The probability of drawing two chips of different colors from a bag containing
    7 blue chips and 5 yellow chips, with replacement after the first draw. -/
theorem different_color_chips_probability :
  let blue_chips : ℕ := 7
  let yellow_chips : ℕ := 5
  let total_chips : ℕ := blue_chips + yellow_chips
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_yellow : ℚ := yellow_chips / total_chips
  let prob_different_colors : ℚ := prob_blue * prob_yellow + prob_yellow * prob_blue
  prob_different_colors = 35 / 72 := by
sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l786_78650


namespace NUMINAMATH_CALUDE_perfume_bottle_size_l786_78651

def petals_per_ounce : ℕ := 320
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes_harvested : ℕ := 800
def bottles_to_make : ℕ := 20

theorem perfume_bottle_size :
  let total_petals := bushes_harvested * roses_per_bush * petals_per_rose
  let total_ounces := total_petals / petals_per_ounce
  let bottle_size := total_ounces / bottles_to_make
  bottle_size = 12 := by sorry

end NUMINAMATH_CALUDE_perfume_bottle_size_l786_78651


namespace NUMINAMATH_CALUDE_floor_difference_equals_eight_l786_78693

theorem floor_difference_equals_eight :
  ⌊(101^3 : ℝ) / (99 * 100) - (99^3 : ℝ) / (100 * 101)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_difference_equals_eight_l786_78693


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l786_78686

def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x < 2}

theorem union_of_A_and_B : A ∪ B = {x | -3 < x ∧ x < 6} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l786_78686
