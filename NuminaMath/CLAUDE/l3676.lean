import Mathlib

namespace NUMINAMATH_CALUDE_johns_investment_l3676_367629

theorem johns_investment (total_investment : ℝ) (alpha_rate beta_rate : ℝ) 
  (total_after_year : ℝ) (alpha_investment : ℝ) :
  total_investment = 1500 →
  alpha_rate = 0.04 →
  beta_rate = 0.06 →
  total_after_year = 1575 →
  alpha_investment = 750 →
  alpha_investment * (1 + alpha_rate) + 
    (total_investment - alpha_investment) * (1 + beta_rate) = total_after_year :=
by sorry

end NUMINAMATH_CALUDE_johns_investment_l3676_367629


namespace NUMINAMATH_CALUDE_problem_solution_l3676_367697

def f (x : ℝ) : ℝ := |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x < |x| + 1 ↔ 0 < x ∧ x < 2) ∧
  (∀ x y : ℝ, |x - y - 1| ≤ 1/3 ∧ |2*y + 1| ≤ 1/6 → f x < 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3676_367697


namespace NUMINAMATH_CALUDE_water_bottle_consumption_l3676_367606

/-- Proves that given a 24-pack of bottled water, if 1/3 is consumed on the first day
    and 1/2 of the remainder is consumed on the second day, then 8 bottles remain after 2 days. -/
theorem water_bottle_consumption (initial_bottles : ℕ) 
  (h1 : initial_bottles = 24)
  (first_day_consumption : ℚ) 
  (h2 : first_day_consumption = 1/3)
  (second_day_consumption : ℚ) 
  (h3 : second_day_consumption = 1/2) :
  initial_bottles - 
  (↑initial_bottles * first_day_consumption).floor - 
  ((↑initial_bottles - (↑initial_bottles * first_day_consumption).floor) * second_day_consumption).floor = 8 :=
by sorry

end NUMINAMATH_CALUDE_water_bottle_consumption_l3676_367606


namespace NUMINAMATH_CALUDE_min_value_theorem_l3676_367625

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  (x^2 + 3*x + 2) / x ≥ 2 * Real.sqrt 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3676_367625


namespace NUMINAMATH_CALUDE_multiplicative_inverse_207_mod_397_l3676_367683

theorem multiplicative_inverse_207_mod_397 :
  ∃ a : ℕ, a < 397 ∧ (207 * a) % 397 = 1 :=
by
  use 66
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_207_mod_397_l3676_367683


namespace NUMINAMATH_CALUDE_tangent_parabola_circle_l3676_367687

/-- Theorem: Tangent Line to Parabola Touching Circle -/
theorem tangent_parabola_circle (r : ℝ) (hr : r > 0) :
  ∃ (x y : ℝ),
    -- Point P(x, y) lies on the parabola
    y = (1/4) * x^2 ∧
    -- Point P(x, y) lies on the circle
    (x - 1)^2 + (y - 2)^2 = r^2 ∧
    -- The tangent line to the parabola at P touches the circle
    ∃ (m : ℝ),
      -- m is the slope of the tangent line to the parabola at P
      m = (1/2) * x ∧
      -- The tangent line touches the circle (perpendicular to radius)
      m * ((y - 2) / (x - 1)) = -1
  → r = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_parabola_circle_l3676_367687


namespace NUMINAMATH_CALUDE_min_common_perimeter_of_specific_isosceles_triangles_l3676_367644

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Checks if two isosceles triangles are noncongruent -/
def noncongruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.leg ≠ t2.leg ∨ t1.base ≠ t2.base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ :=
  2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

/-- Theorem stating the minimum common perimeter of two specific isosceles triangles -/
theorem min_common_perimeter_of_specific_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    noncongruent t1 t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t2.base = 4 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      noncongruent s1 s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s2.base = 4 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 1180 :=
  sorry

end NUMINAMATH_CALUDE_min_common_perimeter_of_specific_isosceles_triangles_l3676_367644


namespace NUMINAMATH_CALUDE_revenue_decrease_l3676_367686

theorem revenue_decrease (R : ℝ) (h1 : R > 0) : 
  let projected_revenue := 1.4 * R
  let actual_revenue := 0.5 * projected_revenue
  let percent_decrease := (R - actual_revenue) / R * 100
  percent_decrease = 30 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_l3676_367686


namespace NUMINAMATH_CALUDE_pats_family_size_l3676_367603

theorem pats_family_size (total_desserts : ℕ) (desserts_per_person : ℕ) 
  (h1 : total_desserts = 126)
  (h2 : desserts_per_person = 18) :
  total_desserts / desserts_per_person = 7 := by
  sorry

end NUMINAMATH_CALUDE_pats_family_size_l3676_367603


namespace NUMINAMATH_CALUDE_parabola_through_point_l3676_367690

def is_parabola_equation (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, f x y ↔ y^2 = 4*a*x

theorem parabola_through_point (f : ℝ → ℝ → Prop) : 
  is_parabola_equation f →
  (∀ x y : ℝ, f x y → y^2 = x) →
  f 4 (-2) →
  ∀ x y : ℝ, f x y ↔ y^2 = x :=
sorry

end NUMINAMATH_CALUDE_parabola_through_point_l3676_367690


namespace NUMINAMATH_CALUDE_windows_preference_count_survey_results_l3676_367617

/-- Represents the survey results of college students' computer brand preferences --/
structure SurveyResults where
  total : ℕ
  mac_preference : ℕ
  no_preference : ℕ
  both_preference : ℕ
  windows_preference : ℕ

/-- Theorem stating the number of students preferring Windows to Mac --/
theorem windows_preference_count (survey : SurveyResults) : 
  survey.total = 210 →
  survey.mac_preference = 60 →
  survey.no_preference = 90 →
  survey.both_preference = survey.mac_preference / 3 →
  survey.windows_preference = 40 := by
  sorry

/-- Main theorem proving the survey results --/
theorem survey_results : ∃ (survey : SurveyResults), 
  survey.total = 210 ∧
  survey.mac_preference = 60 ∧
  survey.no_preference = 90 ∧
  survey.both_preference = survey.mac_preference / 3 ∧
  survey.windows_preference = 40 := by
  sorry

end NUMINAMATH_CALUDE_windows_preference_count_survey_results_l3676_367617


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3676_367674

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 100) : 
  a * b = -3 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3676_367674


namespace NUMINAMATH_CALUDE_best_fitting_model_l3676_367609

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  name : String
  r_squared : Real

/-- Determines if a given model has the best fitting effect among a list of models -/
def has_best_fitting_effect (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, |1 - model.r_squared| ≤ |1 - m.r_squared|

theorem best_fitting_model (models : List RegressionModel) 
  (h1 : Model1 ∈ models ∧ Model1.r_squared = 0.98)
  (h2 : Model2 ∈ models ∧ Model2.r_squared = 0.80)
  (h3 : Model3 ∈ models ∧ Model3.r_squared = 0.50)
  (h4 : Model4 ∈ models ∧ Model4.r_squared = 0.25) :
  has_best_fitting_effect Model1 models :=
sorry

end NUMINAMATH_CALUDE_best_fitting_model_l3676_367609


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l3676_367641

theorem lcm_gcd_product (a b : ℕ) (ha : a = 28) (hb : b = 45) :
  (Nat.lcm a b) * (Nat.gcd a b) = a * b := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l3676_367641


namespace NUMINAMATH_CALUDE_relationship_between_variables_l3676_367637

-- Define variables
variable (a b c d : ℝ)
variable (x y q z : ℝ)

-- Define the theorem
theorem relationship_between_variables 
  (h1 : a^(3*x) = c^(2*q)) 
  (h2 : c^(2*q) = b)
  (h3 : c^(4*y) = a^(5*z))
  (h4 : a^(5*z) = d) :
  5*q*z = 6*x*y := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_variables_l3676_367637


namespace NUMINAMATH_CALUDE_three_propositions_l3676_367696

theorem three_propositions :
  (∀ a b : ℝ, |a - b| < 1 → |a| < |b| + 1) ∧
  (∀ a b : ℝ, |a + b| - 2 * |a| ≤ |a - b|) ∧
  (∀ x y : ℝ, |x| < 2 ∧ |y| > 3 → |x / y| < 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_three_propositions_l3676_367696


namespace NUMINAMATH_CALUDE_potential_parallel_necessary_not_sufficient_l3676_367600

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate for parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The condition for potential parallelism -/
def potential_parallel_condition (l1 l2 : Line) : Prop :=
  l1.a * l2.b - l2.a * l1.b = 0

/-- Theorem stating that the condition is necessary but not sufficient for parallelism -/
theorem potential_parallel_necessary_not_sufficient :
  (∀ l1 l2 : Line, parallel l1 l2 → potential_parallel_condition l1 l2) ∧
  ¬(∀ l1 l2 : Line, potential_parallel_condition l1 l2 → parallel l1 l2) :=
sorry

end NUMINAMATH_CALUDE_potential_parallel_necessary_not_sufficient_l3676_367600


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3676_367611

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, |x^2 + 2*a*x + a + 5| ≤ 3) ↔ (a = 4 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3676_367611


namespace NUMINAMATH_CALUDE_non_monotonic_interval_implies_k_range_l3676_367626

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of non-monotonicity in an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a < x ∧ x < y ∧ y < b ∧ (f x < f y ∧ ∃ z, x < z ∧ z < y ∧ f z < f x)

-- State the theorem
theorem non_monotonic_interval_implies_k_range (k : ℝ) :
  not_monotonic f (k - 1) (k + 1) → (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) :=
sorry

end NUMINAMATH_CALUDE_non_monotonic_interval_implies_k_range_l3676_367626


namespace NUMINAMATH_CALUDE_smallest_multiple_l3676_367659

theorem smallest_multiple (n : ℕ) : n = 2349 ↔ 
  n > 0 ∧ 
  29 ∣ n ∧ 
  n % 97 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 29 ∣ m → m % 97 = 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3676_367659


namespace NUMINAMATH_CALUDE_travel_distance_calculation_l3676_367642

theorem travel_distance_calculation (total_distance sea_distance : ℕ) 
  (h1 : total_distance = 601)
  (h2 : sea_distance = 150) :
  total_distance - sea_distance = 451 :=
by sorry

end NUMINAMATH_CALUDE_travel_distance_calculation_l3676_367642


namespace NUMINAMATH_CALUDE_sector_angle_l3676_367638

/-- Given a circle with radius 12 meters and a sector with area 45.25714285714286 square meters,
    the central angle of the sector is 36 degrees. -/
theorem sector_angle (r : ℝ) (area : ℝ) (h1 : r = 12) (h2 : area = 45.25714285714286) :
  (area / (π * r^2)) * 360 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l3676_367638


namespace NUMINAMATH_CALUDE_triangle_proof_l3676_367602

theorem triangle_proof (A B C : Real) (a b c : Real) (R : Real) :
  let D := (A + C) / 2  -- D is midpoint of AC
  (1/2) * Real.sin (2*B) * Real.cos C + Real.cos B ^ 2 * Real.sin C - Real.sin (A/2) * Real.cos (A/2) = 0 →
  R = Real.sqrt 3 →
  B = π/3 ∧ 
  Real.sqrt ((a^2 + c^2) * 2 - 9) / 2 = 
    Real.sqrt ((Real.sin A * R)^2 + (Real.sin C * R)^2 - (Real.sin B * R)^2 / 4) :=
by sorry


end NUMINAMATH_CALUDE_triangle_proof_l3676_367602


namespace NUMINAMATH_CALUDE_vacation_rental_families_l3676_367671

/-- The number of people in each family -/
def family_size : ℕ := 4

/-- The number of days of the vacation -/
def vacation_days : ℕ := 7

/-- The number of towels each person uses per day -/
def towels_per_person_per_day : ℕ := 1

/-- The capacity of the washing machine in towels -/
def washing_machine_capacity : ℕ := 14

/-- The number of loads needed to wash all towels -/
def total_loads : ℕ := 6

/-- The number of families sharing the vacation rental -/
def num_families : ℕ := 3

theorem vacation_rental_families :
  num_families * family_size * vacation_days * towels_per_person_per_day =
  total_loads * washing_machine_capacity := by sorry

end NUMINAMATH_CALUDE_vacation_rental_families_l3676_367671


namespace NUMINAMATH_CALUDE_unique_three_digit_pair_l3676_367613

theorem unique_three_digit_pair : 
  ∃! (a b : ℕ), 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 ∧ 1000 * a + b = 7 * a * b :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_pair_l3676_367613


namespace NUMINAMATH_CALUDE_smallest_x_squared_l3676_367651

/-- Represents a trapezoid ABCD with a circle tangent to its sides --/
structure TrapezoidWithCircle where
  AB : ℝ
  CD : ℝ
  x : ℝ
  circle_center_distance : ℝ

/-- The smallest possible value of x in the trapezoid configuration --/
def smallest_x (t : TrapezoidWithCircle) : ℝ := sorry

/-- Main theorem: The square of the smallest possible x is 256 --/
theorem smallest_x_squared (t : TrapezoidWithCircle) 
  (h1 : t.AB = 70)
  (h2 : t.CD = 25)
  (h3 : t.circle_center_distance = 10) :
  (smallest_x t)^2 = 256 := by sorry

end NUMINAMATH_CALUDE_smallest_x_squared_l3676_367651


namespace NUMINAMATH_CALUDE_power_calculation_l3676_367670

theorem power_calculation (a : ℝ) : (-a)^2 * (-a^5)^4 / a^12 * (-2 * a^4) = -2 * a^14 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l3676_367670


namespace NUMINAMATH_CALUDE_larry_substitution_l3676_367619

theorem larry_substitution (a b c d f : ℚ) : 
  a = 12 → b = 4 → c = 3 → d = 5 →
  (a / (b / (c * (d - f))) = 12 / 4 / 3 * 5 - f) → f = 5 := by
  sorry

end NUMINAMATH_CALUDE_larry_substitution_l3676_367619


namespace NUMINAMATH_CALUDE_sum_probability_is_thirteen_sixteenths_l3676_367653

/-- Represents an n-sided die -/
structure Die (n : ℕ) where
  sides : Fin n → ℕ
  valid : ∀ i, sides i ∈ Finset.range n.succ

/-- The 8-sided die -/
def eight_sided_die : Die 8 :=
  { sides := λ i => i.val + 1,
    valid := by sorry }

/-- The 6-sided die -/
def six_sided_die : Die 6 :=
  { sides := λ i => i.val + 1,
    valid := by sorry }

/-- The set of all possible outcomes when rolling two dice -/
def outcomes : Finset (Fin 8 × Fin 6) :=
  Finset.product (Finset.univ : Finset (Fin 8)) (Finset.univ : Finset (Fin 6))

/-- The set of favorable outcomes (sum ≤ 10) -/
def favorable_outcomes : Finset (Fin 8 × Fin 6) :=
  outcomes.filter (λ p => eight_sided_die.sides p.1 + six_sided_die.sides p.2 ≤ 10)

/-- The probability of the sum being less than or equal to 10 -/
def probability : ℚ :=
  favorable_outcomes.card / outcomes.card

theorem sum_probability_is_thirteen_sixteenths :
  probability = 13 / 16 := by sorry

end NUMINAMATH_CALUDE_sum_probability_is_thirteen_sixteenths_l3676_367653


namespace NUMINAMATH_CALUDE_find_k_l3676_367665

theorem find_k (x y z k : ℝ) 
  (h1 : 7 / (x + y) = k / (x + z)) 
  (h2 : k / (x + z) = 11 / (z - y)) : k = 18 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3676_367665


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l3676_367616

/-- A sequence with common difference -/
def ArithmeticSequence (a₁ d : ℝ) (n : ℕ) := fun i : ℕ => a₁ + d * (i : ℝ)

/-- Condition for a sequence to be geometric -/
def IsGeometric (s : ℕ → ℝ) := ∀ i j k, s i * s k = s j * s j

/-- Removing one term from a sequence -/
def RemoveTerm (s : ℕ → ℝ) (k : ℕ) := fun i : ℕ => if i < k then s i else s (i + 1)

theorem arithmetic_to_geometric_sequence 
  (n : ℕ) (a₁ d : ℝ) (hn : n ≥ 4) (hd : d ≠ 0) :
  (∃ k, IsGeometric (RemoveTerm (ArithmeticSequence a₁ d n) k)) ↔ 
  (n = 4 ∧ (a₁ / d = -4 ∨ a₁ / d = 1)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l3676_367616


namespace NUMINAMATH_CALUDE_cloth_sale_loss_per_metre_l3676_367660

/-- Given the following conditions for a cloth sale:
  * 500 metres of cloth sold
  * Total selling price of Rs. 15000
  * Cost price of Rs. 40 per metre
  Prove that the loss per metre of cloth sold is Rs. 10. -/
theorem cloth_sale_loss_per_metre 
  (total_metres : ℕ) 
  (total_selling_price : ℕ) 
  (cost_price_per_metre : ℕ) 
  (h1 : total_metres = 500)
  (h2 : total_selling_price = 15000)
  (h3 : cost_price_per_metre = 40) :
  (cost_price_per_metre * total_metres - total_selling_price) / total_metres = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_loss_per_metre_l3676_367660


namespace NUMINAMATH_CALUDE_loss_percentage_is_ten_percent_l3676_367633

def cost_price : ℚ := 1200
def selling_price : ℚ := 1080

def loss : ℚ := cost_price - selling_price

def percentage_loss : ℚ := (loss / cost_price) * 100

theorem loss_percentage_is_ten_percent : percentage_loss = 10 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_is_ten_percent_l3676_367633


namespace NUMINAMATH_CALUDE_sum_of_continuity_points_l3676_367615

/-- Piecewise function f(x) defined by n -/
noncomputable def f (n : ℝ) (x : ℝ) : ℝ :=
  if x < n then x^2 + 2*x + 3 else 3*x + 6

/-- Theorem stating that the sum of all values of n that make f(x) continuous is 2 -/
theorem sum_of_continuity_points (n : ℝ) :
  (∀ x : ℝ, ContinuousAt (f n) x) →
  (∃ n₁ n₂ : ℝ, n₁ ≠ n₂ ∧ 
    (∀ x : ℝ, ContinuousAt (f n₁) x) ∧ 
    (∀ x : ℝ, ContinuousAt (f n₂) x) ∧
    n₁ + n₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_continuity_points_l3676_367615


namespace NUMINAMATH_CALUDE_average_and_difference_l3676_367632

theorem average_and_difference (x : ℝ) : 
  (30 + x) / 2 = 34 → |x - 30| = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l3676_367632


namespace NUMINAMATH_CALUDE_negation_of_exists_negation_of_proposition_l3676_367622

theorem negation_of_exists (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 0, 2 * x + 3 ≤ 0) ↔ (∀ x > 0, 2 * x + 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_negation_of_proposition_l3676_367622


namespace NUMINAMATH_CALUDE_equal_area_intersection_sum_l3676_367608

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is on a line segment -/
def isOnLineSegment (P Q R : Point) : Prop :=
  ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧ R.x = P.x + t * (Q.x - P.x) ∧ R.y = P.y + t * (Q.y - P.y)

/-- Calculates the area of a quadrilateral -/
def quadrilateralArea (quad : Quadrilateral) : ℚ :=
  let A := quad.A
  let B := quad.B
  let C := quad.C
  let D := quad.D
  (1/2) * abs (A.x * (B.y - D.y) + B.x * (C.y - A.y) + C.x * (D.y - B.y) + D.x * (A.y - C.y))

/-- Theorem statement -/
theorem equal_area_intersection_sum (p q r s : ℕ) (quad : Quadrilateral)
  (hA : quad.A = ⟨0, 0⟩)
  (hB : quad.B = ⟨2, 3⟩)
  (hC : quad.C = ⟨6, 3⟩)
  (hD : quad.D = ⟨7, 0⟩)
  (hIntersection : isOnLineSegment quad.C quad.D ⟨p/q, r/s⟩)
  (hEqualArea : ∃ (l : Point → Point → Prop),
    l quad.A ⟨p/q, r/s⟩ ∧
    (quadrilateralArea ⟨quad.A, quad.B, ⟨p/q, r/s⟩, quad.D⟩ =
     quadrilateralArea ⟨quad.A, ⟨p/q, r/s⟩, quad.C, quad.B⟩))
  (hLowestTerms : Nat.gcd p q = 1 ∧ Nat.gcd r s = 1) :
  p + q + r + s = 11 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_intersection_sum_l3676_367608


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3676_367699

/-- The standard equation of a hyperbola with given foci and asymptotes -/
theorem hyperbola_standard_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) →
  (a^2 + b^2 = 10) →
  (b / a = 1 / 2) →
  (a^2 = 8 ∧ b^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3676_367699


namespace NUMINAMATH_CALUDE_union_P_S_when_m_2_S_subset_P_iff_m_in_zero_one_l3676_367624

-- Define the sets P and S
def P : Set ℝ := {x | 4 / (x + 2) ≥ 1}
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Part 1: P ∪ S when m = 2
theorem union_P_S_when_m_2 : 
  P ∪ S 2 = {x | -2 < x ∧ x ≤ 3} := by sorry

-- Part 2: S ⊆ P iff m ∈ [0, 1]
theorem S_subset_P_iff_m_in_zero_one (m : ℝ) : 
  S m ⊆ P ↔ 0 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_union_P_S_when_m_2_S_subset_P_iff_m_in_zero_one_l3676_367624


namespace NUMINAMATH_CALUDE_c_worked_four_days_l3676_367607

/-- Represents the number of days worked by person a -/
def days_a : ℕ := 6

/-- Represents the number of days worked by person b -/
def days_b : ℕ := 9

/-- Represents the daily wage of person c -/
def wage_c : ℕ := 125

/-- Represents the total earnings of all three people -/
def total_earnings : ℕ := 1850

/-- Represents the ratio of daily wages for a, b, and c -/
def wage_ratio : Fin 3 → ℕ
  | 0 => 3  -- a's ratio
  | 1 => 4  -- b's ratio
  | 2 => 5  -- c's ratio

/-- Calculates the daily wage for a given person based on c's wage and the ratio -/
def daily_wage (person : Fin 3) : ℕ :=
  wage_c * wage_ratio person / wage_ratio 2

/-- Theorem stating that person c worked for 4 days -/
theorem c_worked_four_days :
  ∃ (days_c : ℕ), 
    days_c * daily_wage 2 + 
    days_a * daily_wage 0 + 
    days_b * daily_wage 1 = total_earnings ∧
    days_c = 4 := by
  sorry

end NUMINAMATH_CALUDE_c_worked_four_days_l3676_367607


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3676_367677

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  sum_condition : a 3 + a 5 = 8
  product_condition : a 1 * a 5 = 4

/-- The ratio of the 13th term to the 9th term is 9 -/
theorem geometric_sequence_ratio
  (seq : GeometricSequence) :
  seq.a 13 / seq.a 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3676_367677


namespace NUMINAMATH_CALUDE_license_plate_theorem_l3676_367684

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions on the license plate -/
def letter_positions : ℕ := 6

/-- The number of digit positions on the license plate -/
def digit_positions : ℕ := 3

/-- The number of possible license plate combinations with 6 letters 
(where exactly two different letters are repeated once) followed by 3 non-repeating digits -/
def license_plate_combinations : ℕ :=
  (Nat.choose alphabet_size 2) *
  (Nat.choose letter_positions 2) *
  (Nat.choose (letter_positions - 2) 2) *
  (Nat.choose (alphabet_size - 2) 2) *
  (10 * 9 * 8)

theorem license_plate_theorem : license_plate_combinations = 84563400000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l3676_367684


namespace NUMINAMATH_CALUDE_morning_rowers_count_l3676_367688

/-- The number of campers who went rowing in the afternoon -/
def afternoon_rowers : ℕ := 7

/-- The total number of campers who went rowing that day -/
def total_rowers : ℕ := 60

/-- The number of campers who went rowing in the morning -/
def morning_rowers : ℕ := total_rowers - afternoon_rowers

theorem morning_rowers_count : morning_rowers = 53 := by
  sorry

end NUMINAMATH_CALUDE_morning_rowers_count_l3676_367688


namespace NUMINAMATH_CALUDE_unique_root_of_equation_l3676_367639

open Real

theorem unique_root_of_equation :
  ∃! x : ℝ, x > 0 ∧ 1 - x - x * log x = 0 :=
by
  -- Define the function
  let f : ℝ → ℝ := λ x ↦ 1 - x - x * log x

  -- Assume f is decreasing on (0, +∞)
  have h_decreasing : ∀ x y, 0 < x → 0 < y → x < y → f y < f x := sorry

  -- Prove there exists exactly one root
  sorry

end NUMINAMATH_CALUDE_unique_root_of_equation_l3676_367639


namespace NUMINAMATH_CALUDE_existence_of_parallel_plane_l3676_367661

/-- Two lines in space are non-intersecting (skew) -/
def NonIntersecting (a b : Line3) : Prop := sorry

/-- A line is parallel to a plane -/
def ParallelToPlane (l : Line3) (p : Plane3) : Prop := sorry

theorem existence_of_parallel_plane (a b : Line3) (h : NonIntersecting a b) :
  ∃ α : Plane3, ParallelToPlane a α ∧ ParallelToPlane b α := by sorry

end NUMINAMATH_CALUDE_existence_of_parallel_plane_l3676_367661


namespace NUMINAMATH_CALUDE_candy_box_price_increase_l3676_367647

theorem candy_box_price_increase (P : ℝ) : P + 0.25 * P = 10 → P = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_price_increase_l3676_367647


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3676_367654

theorem rectangle_dimension_change (L W : ℝ) (x : ℝ) (h : x > 0) :
  L * (1 + x / 100) * W * (1 - x / 100) = L * W * (1 + 4 / 100) →
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3676_367654


namespace NUMINAMATH_CALUDE_scrap_rate_cost_relationship_l3676_367649

/-- Represents the regression line equation for pig iron cost -/
def regression_line (x : ℝ) : ℝ := 256 + 3 * x

/-- Theorem stating the relationship between scrap rate increase and cost increase -/
theorem scrap_rate_cost_relationship (x : ℝ) :
  regression_line (x + 1) - regression_line x = 3 := by
  sorry

end NUMINAMATH_CALUDE_scrap_rate_cost_relationship_l3676_367649


namespace NUMINAMATH_CALUDE_hyperbola_and_k_range_l3676_367673

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + Real.sqrt 2

-- Define the dot product condition
def dot_product_condition (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 > 2

theorem hyperbola_and_k_range :
  ∃ (a b c : ℝ),
    (∀ x y, ellipse x y ↔ x^2 / a + y^2 / b = 1) ∧
    (c^2 = a / 2) ∧
    (∀ x y, hyperbola_C x y ↔ x^2 / 3 - y^2 = 1) ∧
    (∀ k,
      (∃ x1 y1 x2 y2,
        x1 ≠ x2 ∧
        hyperbola_C x1 y1 ∧
        hyperbola_C x2 y2 ∧
        line_l k x1 y1 ∧
        line_l k x2 y2 ∧
        dot_product_condition x1 y1 x2 y2) ↔
      (k ∈ Set.Ioo (-1 : ℝ) (-Real.sqrt 3 / 3) ∪ Set.Ioo (Real.sqrt 3 / 3) 1)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_and_k_range_l3676_367673


namespace NUMINAMATH_CALUDE_no_404_games_tournament_l3676_367645

theorem no_404_games_tournament : ¬ ∃ (n : ℕ), n > 0 ∧ n * (n - 4) / 2 = 404 := by sorry

end NUMINAMATH_CALUDE_no_404_games_tournament_l3676_367645


namespace NUMINAMATH_CALUDE_union_of_sets_l3676_367627

open Set

theorem union_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | -1 < x ∧ x < 3} → 
  B = {x : ℝ | x ≥ 1} → 
  A ∪ B = {x : ℝ | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3676_367627


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l3676_367636

theorem coefficient_x_squared (p q : Polynomial ℤ) : 
  p = X^3 - 4*X^2 + 6*X - 2 →
  q = 3*X^2 - 2*X + 5 →
  (p * q).coeff 2 = -38 := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_l3676_367636


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_y_value_l3676_367672

/-- Given two vectors a and b in ℝ², prove that if a = (2,5) and b = (1,y) are parallel, then y = 5/2 -/
theorem parallel_vectors_imply_y_value (a b : ℝ × ℝ) (y : ℝ) :
  a = (2, 5) →
  b = (1, y) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  y = 5/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_y_value_l3676_367672


namespace NUMINAMATH_CALUDE_cone_height_from_sector_l3676_367680

/-- Given a sector paper with radius 13 cm and area 65π cm², prove that when formed into a cone, the height of the cone is 12 cm. -/
theorem cone_height_from_sector (r : ℝ) (h : ℝ) :
  r = 13 →
  r * r * π / 2 = 65 * π →
  h = 12 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_from_sector_l3676_367680


namespace NUMINAMATH_CALUDE_prime_triplet_l3676_367662

theorem prime_triplet (p : ℕ) : 
  Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_triplet_l3676_367662


namespace NUMINAMATH_CALUDE_pizza_fraction_l3676_367655

theorem pizza_fraction (total_slices : ℕ) (whole_slices : ℕ) (shared_slice_fraction : ℚ) :
  total_slices = 16 →
  whole_slices = 2 →
  shared_slice_fraction = 1/6 →
  (whole_slices : ℚ) / total_slices + shared_slice_fraction / total_slices = 13/96 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_l3676_367655


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3676_367689

theorem tan_alpha_value (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3676_367689


namespace NUMINAMATH_CALUDE_range_of_even_quadratic_l3676_367635

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

-- Define the property of being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem range_of_even_quadratic (a b : ℝ) :
  (∀ x ∈ Set.Icc (-2*b) (3*b - 1), f a b x ∈ Set.Icc 1 5) ∧
  is_even (f a b) →
  Set.range (f a b) = Set.Icc 1 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_even_quadratic_l3676_367635


namespace NUMINAMATH_CALUDE_circle_symmetry_l3676_367634

-- Define the given circle
def given_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + (y + 1)^2 = 5

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y x' y' : ℝ),
    given_circle x y →
    symmetry_line ((x + x') / 2) ((y + y') / 2) →
    symmetric_circle x' y' :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3676_367634


namespace NUMINAMATH_CALUDE_complex_expression_value_l3676_367664

theorem complex_expression_value (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l3676_367664


namespace NUMINAMATH_CALUDE_total_students_in_line_l3676_367648

/-- The number of students in a line, given specific positions of Hoseok and Yoongi -/
def number_of_students (left_of_hoseok : ℕ) (between_hoseok_yoongi : ℕ) (right_of_yoongi : ℕ) : ℕ :=
  left_of_hoseok + 1 + between_hoseok_yoongi + 1 + right_of_yoongi

/-- Theorem stating that the total number of students in the line is 22 -/
theorem total_students_in_line : 
  number_of_students 9 5 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_students_in_line_l3676_367648


namespace NUMINAMATH_CALUDE_polygon_120_degree_angle_l3676_367657

/-- A triangular grid of equilateral triangles with unit sides -/
structure TriangularGrid where
  -- Add necessary fields here

/-- A non-self-intersecting polygon on a triangular grid -/
structure Polygon (grid : TriangularGrid) where
  vertices : List (ℕ × ℕ)
  is_non_self_intersecting : Bool
  perimeter : ℕ

/-- Checks if a polygon has a 120-degree angle (internal or external) -/
def has_120_degree_angle (grid : TriangularGrid) (p : Polygon grid) : Prop :=
  sorry

theorem polygon_120_degree_angle 
  (grid : TriangularGrid) 
  (p : Polygon grid) 
  (h1 : p.is_non_self_intersecting = true) 
  (h2 : p.perimeter = 1399) : 
  has_120_degree_angle grid p := by
  sorry

end NUMINAMATH_CALUDE_polygon_120_degree_angle_l3676_367657


namespace NUMINAMATH_CALUDE_absent_workers_l3676_367640

/-- Given a group of workers and their work schedule, calculate the number of absent workers. -/
theorem absent_workers 
  (total_workers : ℕ) 
  (original_days : ℕ) 
  (actual_days : ℕ) 
  (h1 : total_workers = 15)
  (h2 : original_days = 40)
  (h3 : actual_days = 60) :
  ∃ (absent : ℕ), 
    absent = 5 ∧ 
    (total_workers - absent) * actual_days = total_workers * original_days :=
by sorry


end NUMINAMATH_CALUDE_absent_workers_l3676_367640


namespace NUMINAMATH_CALUDE_win_by_fourth_round_prob_l3676_367643

/-- The probability of winning a single round in Rock, Paper, Scissors -/
def win_prob : ℚ := 1 / 3

/-- The number of rounds needed to win the game -/
def rounds_to_win : ℕ := 3

/-- The total number of rounds played -/
def total_rounds : ℕ := 4

/-- The probability of winning by the fourth round in a "best of five" Rock, Paper, Scissors game -/
theorem win_by_fourth_round_prob :
  (Nat.choose (total_rounds - 1) (rounds_to_win - 1) : ℚ) *
  win_prob ^ (rounds_to_win - 1) *
  (1 - win_prob) ^ (total_rounds - rounds_to_win) *
  win_prob = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_win_by_fourth_round_prob_l3676_367643


namespace NUMINAMATH_CALUDE_solution_negative_l3676_367678

-- Define the equation
def equation (x a : ℝ) : Prop :=
  (x - 1) / (x - 2) - (x - 2) / (x + 1) = (2 * x + a) / ((x - 2) * (x + 1))

-- Define the theorem
theorem solution_negative (a : ℝ) :
  (∃ x : ℝ, equation x a ∧ x < 0) ↔ (a < -5 ∧ a ≠ -7) :=
sorry

end NUMINAMATH_CALUDE_solution_negative_l3676_367678


namespace NUMINAMATH_CALUDE_annie_bus_ride_l3676_367610

/-- The number of blocks Annie walked from her house to the bus stop -/
def blocks_to_bus_stop : ℕ := 5

/-- The total number of blocks Annie traveled -/
def total_blocks : ℕ := 24

/-- The number of blocks Annie rode the bus to the coffee shop -/
def blocks_by_bus : ℕ := (total_blocks - 2 * blocks_to_bus_stop) / 2

theorem annie_bus_ride : blocks_by_bus = 7 := by
  sorry

end NUMINAMATH_CALUDE_annie_bus_ride_l3676_367610


namespace NUMINAMATH_CALUDE_b_4_lt_b_7_l3676_367668

def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1 + 1 / α 1
  | n+1 => 1 + 1 / (b n α + 1 / α (n+1))

theorem b_4_lt_b_7 (α : ℕ → ℕ) : b 4 α < b 7 α := by
  sorry

end NUMINAMATH_CALUDE_b_4_lt_b_7_l3676_367668


namespace NUMINAMATH_CALUDE_average_growth_rate_inequality_l3676_367694

theorem average_growth_rate_inequality (a p q x : ℝ) 
  (h1 : a > 0) 
  (h2 : p ≥ 0) 
  (h3 : q ≥ 0) 
  (h4 : a * (1 + p) * (1 + q) = a * (1 + x)^2) : 
  x ≤ (p + q) / 2 := by
sorry

end NUMINAMATH_CALUDE_average_growth_rate_inequality_l3676_367694


namespace NUMINAMATH_CALUDE_sqrt_90000_equals_300_l3676_367685

theorem sqrt_90000_equals_300 : Real.sqrt 90000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_90000_equals_300_l3676_367685


namespace NUMINAMATH_CALUDE_parallel_vectors_fraction_l3676_367681

theorem parallel_vectors_fraction (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, 3/2)
  let b : ℝ × ℝ := (Real.cos x, -1)
  (a.1 * b.2 = a.2 * b.1) →
  (2 * Real.sin x - Real.cos x) / (4 * Real.sin x + 3 * Real.cos x) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_fraction_l3676_367681


namespace NUMINAMATH_CALUDE_ellipse_equation_l3676_367693

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (e : ℝ) (P : ℝ × ℝ) : 
  e = Real.sqrt 5 / 5 →
  P = (-5, 4) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 45 + y^2 / 36 = 1) :=
by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ellipse_equation_l3676_367693


namespace NUMINAMATH_CALUDE_lucky_sock_pairs_l3676_367679

/-- The probability of all pairs being lucky given n pairs of socks --/
def prob_all_lucky (n : ℕ) : ℚ :=
  (2^n * n.factorial) / (2*n).factorial

/-- The expected number of lucky pairs given n pairs of socks --/
def expected_lucky_pairs (n : ℕ) : ℚ :=
  n / (2*n - 1)

/-- Theorem stating the properties of lucky sock pairs --/
theorem lucky_sock_pairs (n : ℕ) (h : n > 0) : 
  prob_all_lucky n = (2^n * n.factorial) / (2*n).factorial ∧ 
  expected_lucky_pairs n > 1/2 := by
  sorry

#check lucky_sock_pairs

end NUMINAMATH_CALUDE_lucky_sock_pairs_l3676_367679


namespace NUMINAMATH_CALUDE_expression_simplification_l3676_367618

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3 * x^2 + 2 * x) / ((x - 1) * (x + 3)) - (5 * x + 3) / ((x - 1) * (x + 3)) =
  3 * (x^2 - x - 1) / ((x - 1) * (x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3676_367618


namespace NUMINAMATH_CALUDE_only_cone_cannot_have_rectangular_cross_section_l3676_367658

-- Define the types of geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | PentagonalPrism
  | Cube

-- Define a function that determines if a solid can have a rectangular cross-section
def canHaveRectangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => true
  | GeometricSolid.Cone => false
  | GeometricSolid.PentagonalPrism => true
  | GeometricSolid.Cube => true

-- Theorem statement
theorem only_cone_cannot_have_rectangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveRectangularCrossSection solid) ↔ solid = GeometricSolid.Cone :=
by sorry

end NUMINAMATH_CALUDE_only_cone_cannot_have_rectangular_cross_section_l3676_367658


namespace NUMINAMATH_CALUDE_smallest_number_minus_three_divisible_by_fifteen_l3676_367691

theorem smallest_number_minus_three_divisible_by_fifteen : 
  ∃ N : ℕ, (N ≥ 18) ∧ (N - 3) % 15 = 0 ∧ ∀ M : ℕ, M < N → (M - 3) % 15 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_minus_three_divisible_by_fifteen_l3676_367691


namespace NUMINAMATH_CALUDE_smallest_to_large_square_area_ratio_l3676_367620

/-- Represents a square with a given side length -/
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.side_length ^ 2

theorem smallest_to_large_square_area_ratio :
  ∀ (large : Square),
  ∃ (middle smallest : Square),
  (middle.side_length = large.side_length / 2) ∧
  (smallest.side_length = middle.side_length / 2) →
  smallest.area / large.area = 1 / 16 :=
by
  sorry

#check smallest_to_large_square_area_ratio

end NUMINAMATH_CALUDE_smallest_to_large_square_area_ratio_l3676_367620


namespace NUMINAMATH_CALUDE_two_axes_implies_center_symmetry_l3676_367623

/-- A geometric figure in a 2D plane. -/
structure Figure where
  -- The implementation details of the figure are abstracted away

/-- An axis of symmetry for a figure. -/
structure AxisOfSymmetry where
  -- The implementation details of the axis are abstracted away

/-- A center of symmetry for a figure. -/
structure CenterOfSymmetry where
  -- The implementation details of the center are abstracted away

/-- Predicate to check if a figure has an axis of symmetry. -/
def hasAxisOfSymmetry (f : Figure) (a : AxisOfSymmetry) : Prop :=
  sorry

/-- Predicate to check if a figure has a center of symmetry. -/
def hasCenterOfSymmetry (f : Figure) (c : CenterOfSymmetry) : Prop :=
  sorry

/-- Theorem: If a figure has exactly two axes of symmetry, it must have a center of symmetry. -/
theorem two_axes_implies_center_symmetry (f : Figure) (a1 a2 : AxisOfSymmetry) :
  (hasAxisOfSymmetry f a1) ∧ 
  (hasAxisOfSymmetry f a2) ∧ 
  (a1 ≠ a2) ∧
  (∀ a : AxisOfSymmetry, hasAxisOfSymmetry f a → (a = a1 ∨ a = a2)) →
  ∃ c : CenterOfSymmetry, hasCenterOfSymmetry f c :=
sorry

end NUMINAMATH_CALUDE_two_axes_implies_center_symmetry_l3676_367623


namespace NUMINAMATH_CALUDE_cost_price_of_article_l3676_367682

/-- 
Given an article where the profit when selling it for Rs. 57 is equal to the loss 
when selling it for Rs. 43, prove that the cost price of the article is Rs. 50.
-/
theorem cost_price_of_article (cost_price : ℕ) : cost_price = 50 := by
  sorry

/--
Helper function to calculate profit
-/
def profit (selling_price cost_price : ℕ) : ℤ :=
  (selling_price : ℤ) - (cost_price : ℤ)

/--
Helper function to calculate loss
-/
def loss (cost_price selling_price : ℕ) : ℤ :=
  (cost_price : ℤ) - (selling_price : ℤ)

/--
Assumption that profit when selling at Rs. 57 equals loss when selling at Rs. 43
-/
axiom profit_loss_equality (cost_price : ℕ) :
  profit 57 cost_price = loss cost_price 43

end NUMINAMATH_CALUDE_cost_price_of_article_l3676_367682


namespace NUMINAMATH_CALUDE_pedestrian_meets_sixteen_buses_l3676_367612

/-- Represents the problem of a pedestrian meeting buses on a road --/
structure BusMeetingProblem where
  road_length : ℝ
  bus_speed : ℝ
  bus_interval : ℝ
  pedestrian_start_time : ℝ
  pedestrian_speed : ℝ

/-- Calculates the number of buses the pedestrian meets --/
def count_bus_meetings (problem : BusMeetingProblem) : ℕ :=
  sorry

/-- The main theorem stating that the pedestrian meets 16 buses --/
theorem pedestrian_meets_sixteen_buses :
  let problem : BusMeetingProblem := {
    road_length := 8,
    bus_speed := 12,
    bus_interval := 1/6,  -- 10 minutes in hours
    pedestrian_start_time := 81/4,  -- 8:15 AM in hours since midnight
    pedestrian_speed := 4
  }
  count_bus_meetings problem = 16 := by
  sorry

end NUMINAMATH_CALUDE_pedestrian_meets_sixteen_buses_l3676_367612


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3676_367605

theorem complex_fraction_simplification :
  let z : ℂ := (5 + 7*I) / (2 + 3*I)
  z = 31/13 - (1/13)*I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3676_367605


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3676_367692

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ (m : ℝ),
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (m, -1)
  are_parallel a b → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3676_367692


namespace NUMINAMATH_CALUDE_target_shopping_total_l3676_367621

/-- The total amount spent by Christy and Tanya at Target -/
def total_spent (face_moisturizer_price : ℕ) (body_lotion_price : ℕ) 
  (face_moisturizer_count : ℕ) (body_lotion_count : ℕ) : ℕ :=
  let tanya_spent := face_moisturizer_price * face_moisturizer_count + 
                     body_lotion_price * body_lotion_count
  2 * tanya_spent

/-- Theorem stating the total amount spent by Christy and Tanya -/
theorem target_shopping_total : 
  total_spent 50 60 2 4 = 1020 := by
  sorry

#eval total_spent 50 60 2 4

end NUMINAMATH_CALUDE_target_shopping_total_l3676_367621


namespace NUMINAMATH_CALUDE_tiles_required_for_room_l3676_367630

theorem tiles_required_for_room (room_length room_width tile_length tile_width : ℚ) :
  room_length = 10 →
  room_width = 15 →
  tile_length = 5 / 12 →
  tile_width = 2 / 3 →
  (room_length * room_width) / (tile_length * tile_width) = 540 :=
by
  sorry

end NUMINAMATH_CALUDE_tiles_required_for_room_l3676_367630


namespace NUMINAMATH_CALUDE_factor_expression_l3676_367675

theorem factor_expression (x y : ℝ) : 3 * x^2 - 75 * y^2 = 3 * (x + 5*y) * (x - 5*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3676_367675


namespace NUMINAMATH_CALUDE_cube_of_negative_two_x_l3676_367604

theorem cube_of_negative_two_x (x : ℝ) : (-2 * x)^3 = -8 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_two_x_l3676_367604


namespace NUMINAMATH_CALUDE_sin_cos_derivative_ratio_l3676_367698

theorem sin_cos_derivative_ratio (x : ℝ) (f : ℝ → ℝ) 
  (h1 : f = λ x => Real.sin x + Real.cos x)
  (h2 : deriv f = λ x => 3 * f x) :
  (Real.sin x)^2 - 3 / ((Real.cos x)^2 + 1) = -14/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_derivative_ratio_l3676_367698


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l3676_367631

theorem polar_to_rectangular (r θ : ℝ) :
  r = -3 ∧ θ = 5 * π / 6 →
  (r * Real.cos θ = 3 * Real.sqrt 3 / 2) ∧ (r * Real.sin θ = -3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l3676_367631


namespace NUMINAMATH_CALUDE_cookie_calories_l3676_367628

/-- Calculates the number of calories per cookie in a box of cookies. -/
def calories_per_cookie (cookies_per_bag : ℕ) (bags_per_box : ℕ) (total_calories : ℕ) : ℕ :=
  total_calories / (cookies_per_bag * bags_per_box)

/-- Theorem: Given a box of cookies with 4 bags, 20 cookies per bag, and a total of 1600 calories,
    each cookie contains 20 calories. -/
theorem cookie_calories :
  calories_per_cookie 20 4 1600 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cookie_calories_l3676_367628


namespace NUMINAMATH_CALUDE_round_trip_distance_approx_l3676_367614

/-- Represents the total distance traveled in John's round trip --/
def total_distance (city_speed outbound_highway_speed return_highway_speed : ℝ)
  (outbound_city_time outbound_highway_time return_highway_time1 return_highway_time2 return_city_time : ℝ) : ℝ :=
  let outbound_city_distance := city_speed * outbound_city_time
  let outbound_highway_distance := outbound_highway_speed * outbound_highway_time
  let return_highway_distance := return_highway_speed * (return_highway_time1 + return_highway_time2)
  let return_city_distance := city_speed * return_city_time
  outbound_city_distance + outbound_highway_distance + return_highway_distance + return_city_distance

/-- Theorem stating that the total round trip distance is approximately 166.67 km --/
theorem round_trip_distance_approx : 
  ∀ (ε : ℝ), ε > 0 → 
  ∃ (city_speed outbound_highway_speed return_highway_speed : ℝ)
    (outbound_city_time outbound_highway_time return_highway_time1 return_highway_time2 return_city_time : ℝ),
  city_speed = 40 ∧ 
  outbound_highway_speed = 80 ∧
  return_highway_speed = 100 ∧
  outbound_city_time = 1/3 ∧
  outbound_highway_time = 2/3 ∧
  return_highway_time1 = 1/2 ∧
  return_highway_time2 = 1/6 ∧
  return_city_time = 1/3 ∧
  |total_distance city_speed outbound_highway_speed return_highway_speed
    outbound_city_time outbound_highway_time return_highway_time1 return_highway_time2 return_city_time - 166.67| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_round_trip_distance_approx_l3676_367614


namespace NUMINAMATH_CALUDE_initial_balloons_eq_sum_l3676_367666

/-- The number of green balloons Fred initially had -/
def initial_balloons : ℕ := sorry

/-- The number of green balloons Fred gave to Sandy -/
def balloons_given : ℕ := 221

/-- The number of green balloons Fred has left -/
def balloons_left : ℕ := 488

/-- Theorem stating that the initial number of balloons is equal to the sum of balloons given away and balloons left -/
theorem initial_balloons_eq_sum : initial_balloons = balloons_given + balloons_left := by sorry

end NUMINAMATH_CALUDE_initial_balloons_eq_sum_l3676_367666


namespace NUMINAMATH_CALUDE_kevin_kangaroo_hops_l3676_367601

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Kevin's hopping problem -/
theorem kevin_kangaroo_hops :
  geometricSum (1/2) (1/2) 6 = 63/64 := by
  sorry

end NUMINAMATH_CALUDE_kevin_kangaroo_hops_l3676_367601


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l3676_367695

theorem triangle_cosine_theorem (X Y Z : Real) :
  -- Triangle XYZ
  X + Y + Z = Real.pi →
  -- sin X = 4/5
  Real.sin X = 4/5 →
  -- cos Y = 12/13
  Real.cos Y = 12/13 →
  -- Then cos Z = -16/65
  Real.cos Z = -16/65 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l3676_367695


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l3676_367646

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : a * b > 0) :
  (a > 0 ∧ b > 0 → 1 / a > 1 / b) ∧
  (a < 0 ∧ b < 0 → 1 / a < 1 / b) := by
sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l3676_367646


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_equality_condition_l3676_367663

theorem min_value_of_sum_of_roots (x : ℝ) : 
  Real.sqrt ((x - 2)^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 4 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) : 
  Real.sqrt ((x - 2)^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 4 * Real.sqrt 2 ↔ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_equality_condition_l3676_367663


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l3676_367650

/-- The number of five-digit odd numbers -/
def A : ℕ := 9 * 10 * 10 * 10 * 5

/-- The number of five-digit multiples of 5 that are also odd -/
def B : ℕ := 9 * 10 * 10 * 10 * 1

/-- The sum of A and B is equal to 45,000 -/
theorem sum_of_A_and_B : A + B = 45000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l3676_367650


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3676_367667

theorem cone_lateral_surface_area 
  (r : ℝ) 
  (V : ℝ) 
  (h : ℝ) 
  (l : ℝ) : 
  r = 6 →
  V = 30 * Real.pi →
  V = (1/3) * Real.pi * r^2 * h →
  l^2 = r^2 + h^2 →
  r * l * Real.pi = 39 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3676_367667


namespace NUMINAMATH_CALUDE_r_fraction_of_total_l3676_367656

theorem r_fraction_of_total (total : ℚ) (r_amount : ℚ) 
  (h1 : total = 4000)
  (h2 : r_amount = 1600) :
  r_amount / total = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_r_fraction_of_total_l3676_367656


namespace NUMINAMATH_CALUDE_specialArrangements_eq_480_l3676_367669

/-- The number of ways to arrange six distinct objects in a row,
    where two specific objects must be on the same side of a third specific object -/
def specialArrangements : ℕ :=
  let totalPositions := 6
  let fixedObjects := 3  -- A, B, and C
  let remainingObjects := 3  -- D, E, and F
  let positionsForC := totalPositions
  let waysToArrangeAB := 2  -- A and B can be swapped
  let waysToChooseSide := 2  -- A and B can be on either side of C
  let remainingArrangements := Nat.factorial remainingObjects

  positionsForC * waysToArrangeAB * waysToChooseSide * remainingArrangements

theorem specialArrangements_eq_480 : specialArrangements = 480 := by
  sorry

end NUMINAMATH_CALUDE_specialArrangements_eq_480_l3676_367669


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l3676_367652

/-- An isosceles triangle with given altitude and perimeter has area 75 -/
theorem isosceles_triangle_area (base altitudeToBase equalSide : ℝ) : 
  base > 0 → 
  altitudeToBase > 0 → 
  equalSide > 0 → 
  altitudeToBase = 10 → 
  base + 2 * equalSide = 40 → 
  (1/2) * base * altitudeToBase = 75 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l3676_367652


namespace NUMINAMATH_CALUDE_marker_cost_l3676_367676

theorem marker_cost (total_students : ℕ) (buyers : ℕ) (markers_per_student : ℕ) (marker_cost : ℕ) :
  total_students = 24 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  markers_per_student > 1 →
  marker_cost > markers_per_student →
  buyers * marker_cost * markers_per_student = 924 →
  marker_cost = 11 :=
by sorry

end NUMINAMATH_CALUDE_marker_cost_l3676_367676
