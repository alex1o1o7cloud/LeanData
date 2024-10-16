import Mathlib

namespace NUMINAMATH_CALUDE_travel_options_l1681_168193

/-- The number of train departures from City A to City B -/
def train_departures : ℕ := 10

/-- The number of flights from City A to City B -/
def flights : ℕ := 2

/-- The number of long-distance bus services from City A to City B -/
def bus_services : ℕ := 12

/-- The total number of ways Xiao Zhang can travel from City A to City B -/
def total_ways : ℕ := train_departures + flights + bus_services

theorem travel_options : total_ways = 24 := by sorry

end NUMINAMATH_CALUDE_travel_options_l1681_168193


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1681_168136

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (x - 2)^6 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1681_168136


namespace NUMINAMATH_CALUDE_factory_problem_l1681_168152

/-- Represents the production rates and working days of two factories -/
structure FactoryProduction where
  initial_rate_B : ℝ
  initial_rate_A : ℝ
  total_days : ℕ
  adjustment_days : ℕ

/-- The solution to the factory production problem -/
def factory_problem_solution (fp : FactoryProduction) : ℝ :=
  3

/-- Theorem stating the solution to the factory production problem -/
theorem factory_problem (fp : FactoryProduction) :
  fp.initial_rate_A = (4/3) * fp.initial_rate_B →
  fp.total_days = 6 →
  fp.adjustment_days = 1 →
  let days_before := fp.total_days - fp.adjustment_days - (factory_problem_solution fp)
  let production_A := fp.initial_rate_A * fp.total_days
  let production_B := fp.initial_rate_B * days_before + 2 * fp.initial_rate_B * (factory_problem_solution fp)
  production_A = production_B :=
by sorry

end NUMINAMATH_CALUDE_factory_problem_l1681_168152


namespace NUMINAMATH_CALUDE_jack_shoe_time_proof_l1681_168181

/-- The time it takes Jack to put on his shoes -/
def jack_shoe_time : ℝ := 4

/-- The time it takes Jack to help one toddler with their shoes -/
def toddler_shoe_time (j : ℝ) : ℝ := j + 3

/-- The total time for Jack and two toddlers to get ready -/
def total_time (j : ℝ) : ℝ := j + 2 * (toddler_shoe_time j)

theorem jack_shoe_time_proof :
  total_time jack_shoe_time = 18 :=
by sorry

end NUMINAMATH_CALUDE_jack_shoe_time_proof_l1681_168181


namespace NUMINAMATH_CALUDE_no_matrix_satisfies_condition_l1681_168142

theorem no_matrix_satisfies_condition : 
  ∀ (N : Matrix (Fin 2) (Fin 2) ℝ),
    (∀ (w x y z : ℝ), 
      N * !![w, x; y, z] = !![x, w; z, y]) → 
    N = 0 := by sorry

end NUMINAMATH_CALUDE_no_matrix_satisfies_condition_l1681_168142


namespace NUMINAMATH_CALUDE_x_value_implies_y_value_l1681_168117

theorem x_value_implies_y_value :
  let x := (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 48
  x^3 - 2*x^2 + Real.sin (2*Real.pi*x) - Real.cos (Real.pi*x) = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_value_implies_y_value_l1681_168117


namespace NUMINAMATH_CALUDE_johns_remaining_money_l1681_168166

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Calculates the remaining money after subtracting flight cost --/
def remaining_money (savings : ℕ) (flight_cost : ℕ) : ℕ :=
  octal_to_decimal savings - flight_cost

/-- Theorem stating that John's remaining money is 1725 in decimal --/
theorem johns_remaining_money :
  remaining_money 5555 1200 = 1725 := by sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l1681_168166


namespace NUMINAMATH_CALUDE_product_equality_l1681_168113

theorem product_equality (p q : ℤ) : 
  (∀ d : ℚ, (5 * d^2 - 4 * d + p) * (4 * d^2 + q * d - 5) = 20 * d^4 + 11 * d^3 - 45 * d^2 - 20 * d + 25) → 
  p = -5 ∧ q = 8 := by
sorry

end NUMINAMATH_CALUDE_product_equality_l1681_168113


namespace NUMINAMATH_CALUDE_max_contribution_l1681_168130

theorem max_contribution 
  (n : ℕ) 
  (total : ℚ) 
  (min_contribution : ℚ) 
  (h1 : n = 15)
  (h2 : total = 30)
  (h3 : min_contribution = 1)
  (h4 : ∀ i, i ∈ Finset.range n → ∃ c : ℚ, c ≥ min_contribution) :
  ∃ max_contribution : ℚ, 
    max_contribution ≤ total ∧ 
    (∀ i, i ∈ Finset.range n → ∃ c : ℚ, c ≤ max_contribution) ∧
    max_contribution = 16 :=
sorry

end NUMINAMATH_CALUDE_max_contribution_l1681_168130


namespace NUMINAMATH_CALUDE_complement_of_union_l1681_168159

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1681_168159


namespace NUMINAMATH_CALUDE_walking_time_equals_early_arrival_l1681_168138

/-- Represents the scenario of a man walking home and being picked up by his wife --/
structure WalkingScenario where
  D : ℝ  -- Total distance from station to home
  Vw : ℝ  -- Wife's driving speed
  Vm : ℝ  -- Man's walking speed
  T : ℝ  -- Usual time for wife to drive from station to home
  t : ℝ  -- Time man spent walking before being picked up
  early_arrival : ℝ  -- Time they arrived home earlier than usual

/-- The time the man spent walking is equal to the time they arrived home earlier --/
theorem walking_time_equals_early_arrival (scenario : WalkingScenario) 
  (h1 : scenario.D = scenario.Vw * scenario.T)
  (h2 : scenario.D - scenario.Vm * scenario.t = scenario.Vw * (scenario.T - scenario.t))
  (h3 : scenario.early_arrival = scenario.t) :
  scenario.t = scenario.early_arrival :=
by
  sorry

#check walking_time_equals_early_arrival

end NUMINAMATH_CALUDE_walking_time_equals_early_arrival_l1681_168138


namespace NUMINAMATH_CALUDE_shaded_square_area_fraction_l1681_168191

/-- The fraction of a 6x6 grid's area occupied by a square with vertices at midpoints of grid lines along the diagonal -/
theorem shaded_square_area_fraction (grid_size : ℕ) (shaded_square_side : ℝ) : 
  grid_size = 6 → 
  shaded_square_side = 1 / Real.sqrt 2 →
  (shaded_square_side^2) / (grid_size^2 : ℝ) = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_shaded_square_area_fraction_l1681_168191


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1681_168162

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a * x) / (x - 1) < 1 ↔ (x < 1 ∨ x > 2)) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1681_168162


namespace NUMINAMATH_CALUDE_monthly_fixed_costs_correct_l1681_168133

/-- Represents the monthly fixed costs for producing electronic components -/
def monthly_fixed_costs : ℝ := 16399.50

/-- Represents the cost to produce one electronic component -/
def production_cost : ℝ := 80

/-- Represents the shipping cost for one electronic component -/
def shipping_cost : ℝ := 4

/-- Represents the number of components produced and sold monthly -/
def monthly_sales : ℕ := 150

/-- Represents the lowest selling price per component without loss -/
def break_even_price : ℝ := 193.33

/-- Theorem stating that the monthly fixed costs are correct given the other parameters -/
theorem monthly_fixed_costs_correct :
  monthly_fixed_costs = 
    monthly_sales * break_even_price - 
    monthly_sales * (production_cost + shipping_cost) :=
by sorry

end NUMINAMATH_CALUDE_monthly_fixed_costs_correct_l1681_168133


namespace NUMINAMATH_CALUDE_second_knife_set_price_l1681_168127

/-- Calculates the price of the second set of knives based on given sales data --/
def price_of_second_knife_set (
  houses_per_day : ℕ)
  (buy_percentage : ℚ)
  (first_set_price : ℕ)
  (weekly_sales : ℕ)
  (work_days : ℕ) : ℚ :=
  let buyers_per_day : ℚ := houses_per_day * buy_percentage
  let first_set_buyers_per_day : ℚ := buyers_per_day / 2
  let first_set_sales_per_day : ℚ := first_set_buyers_per_day * first_set_price
  let first_set_sales_per_week : ℚ := first_set_sales_per_day * work_days
  let second_set_sales_per_week : ℚ := weekly_sales - first_set_sales_per_week
  let second_set_buyers_per_week : ℚ := first_set_buyers_per_day * work_days
  second_set_sales_per_week / second_set_buyers_per_week

/-- Theorem stating that the price of the second set of knives is $150 --/
theorem second_knife_set_price :
  price_of_second_knife_set 50 (1/5) 50 5000 5 = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_knife_set_price_l1681_168127


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1681_168106

theorem rectangle_ratio (w : ℝ) (h1 : w > 0) (h2 : 2 * w + 2 * 8 = 24) :
  w / 8 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1681_168106


namespace NUMINAMATH_CALUDE_irrational_and_rational_numbers_l1681_168187

theorem irrational_and_rational_numbers : 
  (¬ ∃ (p q : ℤ), π = (p : ℚ) / (q : ℚ)) ∧ 
  (∃ (p q : ℤ), (22 : ℚ) / (7 : ℚ) = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), (0 : ℚ) = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), (-2 : ℚ) = (p : ℚ) / (q : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_irrational_and_rational_numbers_l1681_168187


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l1681_168176

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def I : Set ℝ := Set.Icc (-3) 0

-- Statement of the theorem
theorem f_max_min_on_interval :
  (∃ x ∈ I, ∀ y ∈ I, f y ≤ f x) ∧
  (∃ x ∈ I, ∀ y ∈ I, f x ≤ f y) ∧
  (∃ x ∈ I, f x = 3) ∧
  (∃ x ∈ I, f x = -17) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l1681_168176


namespace NUMINAMATH_CALUDE_unique_p_for_three_natural_roots_l1681_168182

/-- The cubic equation with parameter p -/
def cubic_equation (p : ℝ) (x : ℝ) : ℝ :=
  5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1 - 66 * p

/-- Predicate to check if a number is a natural number -/
def is_natural (x : ℝ) : Prop := ∃ n : ℕ, x = n

/-- The theorem to be proved -/
theorem unique_p_for_three_natural_roots :
  ∃! p : ℝ, ∃ x y z : ℝ,
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    is_natural x ∧ is_natural y ∧ is_natural z ∧
    cubic_equation p x = 0 ∧ cubic_equation p y = 0 ∧ cubic_equation p z = 0 ∧
    p = 76 :=
sorry

end NUMINAMATH_CALUDE_unique_p_for_three_natural_roots_l1681_168182


namespace NUMINAMATH_CALUDE_relay_race_average_time_l1681_168150

/-- Calculates the average time per leg in a two-leg relay race -/
def average_time_per_leg (time_y time_z : ℕ) : ℚ :=
  (time_y + time_z : ℚ) / 2

/-- Theorem: The average time per leg for the given relay race is 42 seconds -/
theorem relay_race_average_time :
  average_time_per_leg 58 26 = 42 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_average_time_l1681_168150


namespace NUMINAMATH_CALUDE_square_area_calculation_square_area_proof_l1681_168109

theorem square_area_calculation (perimeter_B : ℝ) (probability : ℝ) : ℝ :=
  let side_B := perimeter_B / 4
  let area_B := side_B ^ 2
  let area_A := area_B / (1 - probability)
  area_A

theorem square_area_proof :
  square_area_calculation 16 0.7538461538461538 = 65 := by
  sorry

end NUMINAMATH_CALUDE_square_area_calculation_square_area_proof_l1681_168109


namespace NUMINAMATH_CALUDE_function_extremum_l1681_168111

/-- The function f(x) = (x-2)e^x has a minimum value of -e and no maximum value -/
theorem function_extremum :
  let f : ℝ → ℝ := λ x => (x - 2) * Real.exp x
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x) ∧
  f (Real.log 1) = -Real.exp 1 ∧
  ¬∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max :=
by sorry

end NUMINAMATH_CALUDE_function_extremum_l1681_168111


namespace NUMINAMATH_CALUDE_subtract_inequality_negative_l1681_168102

theorem subtract_inequality_negative (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_subtract_inequality_negative_l1681_168102


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l1681_168110

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -72 + 21*I ∧ z = 4 + 7*I → (-z)^2 = -72 + 21*I :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l1681_168110


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l1681_168168

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_4_meaningful_l1681_168168


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1681_168179

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 2}
def B : Set Int := {-1, 0, 1}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1681_168179


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1681_168163

theorem quadratic_factorization (x : ℝ) : x^2 + 14*x + 49 = (x + 7)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1681_168163


namespace NUMINAMATH_CALUDE_conference_left_handed_fraction_l1681_168100

/-- Represents the fraction of left-handed participants in a conference -/
def left_handed_fraction (total : ℕ) (red : ℕ) (blue : ℕ) (red_left : ℚ) (blue_left : ℚ) : ℚ :=
  (red_left * red + blue_left * blue) / total

/-- Theorem stating the fraction of left-handed participants in the conference -/
theorem conference_left_handed_fraction :
  ∀ (total : ℕ) (red : ℕ) (blue : ℕ),
  total > 0 →
  red + blue = total →
  red = blue →
  left_handed_fraction total red blue (1/3) (2/3) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_conference_left_handed_fraction_l1681_168100


namespace NUMINAMATH_CALUDE_cake_frosting_theorem_l1681_168154

/-- Represents a person who can frost cakes -/
structure FrostingPerson where
  name : String
  frostingTime : ℕ

/-- Represents the cake frosting problem -/
structure CakeFrostingProblem where
  people : List FrostingPerson
  numCakes : ℕ
  passingTime : ℕ

/-- Calculates the minimum time to frost all cakes -/
def minFrostingTime (problem : CakeFrostingProblem) : ℕ :=
  sorry

theorem cake_frosting_theorem (problem : CakeFrostingProblem) :
  problem.people = [
    { name := "Ann", frostingTime := 8 },
    { name := "Bob", frostingTime := 6 },
    { name := "Carol", frostingTime := 10 }
  ] ∧
  problem.numCakes = 10 ∧
  problem.passingTime = 1
  →
  minFrostingTime problem = 116 := by
  sorry

end NUMINAMATH_CALUDE_cake_frosting_theorem_l1681_168154


namespace NUMINAMATH_CALUDE_student_D_most_stable_smallest_variance_most_stable_l1681_168132

-- Define the variances for each student
def variance_A : ℝ := 6
def variance_B : ℝ := 5.5
def variance_C : ℝ := 10
def variance_D : ℝ := 3.8

-- Define a function to determine if a student has the most stable performance
def has_most_stable_performance (student_variance : ℝ) : Prop :=
  student_variance ≤ variance_A ∧
  student_variance ≤ variance_B ∧
  student_variance ≤ variance_C ∧
  student_variance ≤ variance_D

-- Theorem stating that student D has the most stable performance
theorem student_D_most_stable : has_most_stable_performance variance_D := by
  sorry

-- Theorem stating that the student with the smallest variance has the most stable performance
theorem smallest_variance_most_stable :
  ∀ (student_variance : ℝ),
    has_most_stable_performance student_variance →
    student_variance = min (min (min variance_A variance_B) variance_C) variance_D := by
  sorry

end NUMINAMATH_CALUDE_student_D_most_stable_smallest_variance_most_stable_l1681_168132


namespace NUMINAMATH_CALUDE_circular_path_area_l1681_168189

/-- The area of a circular path around a circular lawn -/
theorem circular_path_area (r : ℝ) (w : ℝ) (h_r : r > 0) (h_w : w > 0) :
  let R := r + w
  (π * R^2 - π * r^2) = π * (R^2 - r^2) := by sorry

#check circular_path_area

end NUMINAMATH_CALUDE_circular_path_area_l1681_168189


namespace NUMINAMATH_CALUDE_sin_double_angle_when_tan_is_half_l1681_168158

theorem sin_double_angle_when_tan_is_half (α : Real) (h : Real.tan α = 1/2) : 
  Real.sin (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_when_tan_is_half_l1681_168158


namespace NUMINAMATH_CALUDE_sum_of_perpendiculars_equals_height_l1681_168164

/-- An equilateral triangle with a point inside it -/
structure EquilateralTriangleWithPoint where
  /-- Side length of the equilateral triangle -/
  a : ℝ
  /-- Height of the equilateral triangle -/
  h : ℝ
  /-- Perpendicular distance from the point to side AB -/
  m₁ : ℝ
  /-- Perpendicular distance from the point to side BC -/
  m₂ : ℝ
  /-- Perpendicular distance from the point to side CA -/
  m₃ : ℝ
  /-- The point is inside the triangle -/
  point_inside : 0 < m₁ ∧ 0 < m₂ ∧ 0 < m₃
  /-- The triangle is equilateral -/
  equilateral : h = (Real.sqrt 3 / 2) * a
  /-- The height is positive -/
  height_positive : 0 < h

/-- 
The sum of perpendiculars from any point inside an equilateral triangle 
to its sides equals the triangle's height
-/
theorem sum_of_perpendiculars_equals_height (t : EquilateralTriangleWithPoint) : 
  t.m₁ + t.m₂ + t.m₃ = t.h := by
  sorry

end NUMINAMATH_CALUDE_sum_of_perpendiculars_equals_height_l1681_168164


namespace NUMINAMATH_CALUDE_wall_passing_skill_l1681_168107

theorem wall_passing_skill (n : ℕ) : 
  8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n)) → n = 63 :=
by sorry

end NUMINAMATH_CALUDE_wall_passing_skill_l1681_168107


namespace NUMINAMATH_CALUDE_intersection_M_N_l1681_168194

def M : Set ℝ := {x | 2*x - x^2 ≥ 0}
def N : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - x^2)}

theorem intersection_M_N : M ∩ N = Set.Icc 0 1 \ {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1681_168194


namespace NUMINAMATH_CALUDE_train_journey_time_l1681_168125

/-- Proves that if a train moving at 6/7 of its usual speed arrives 10 minutes late, then its usual journey time is 7 hours -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : (6 / 7 * usual_speed) * (usual_time + 10 / 60) = usual_speed * usual_time) : 
  usual_time = 7 := by
  sorry

#check train_journey_time

end NUMINAMATH_CALUDE_train_journey_time_l1681_168125


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1681_168185

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
  (x : ℤ) + y ≤ (a : ℤ) + b → 
  (x : ℤ) + y = 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l1681_168185


namespace NUMINAMATH_CALUDE_election_majority_l1681_168120

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 600 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = 240 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_l1681_168120


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l1681_168119

theorem simplify_product_of_square_roots (x : ℝ) :
  Real.sqrt (x^2 - 4*x + 4) * Real.sqrt (x^2 + 4*x + 4) = |x - 2| * |x + 2| := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l1681_168119


namespace NUMINAMATH_CALUDE_sum_of_squares_130_l1681_168134

theorem sum_of_squares_130 (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  a^2 + b^2 = 130 → 
  a + b = 16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_130_l1681_168134


namespace NUMINAMATH_CALUDE_unique_ambiguous_sum_l1681_168122

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 36

def sum_triple (a b c : ℕ) : ℕ := a + b + c

theorem unique_ambiguous_sum :
  ∃ (s : ℕ), 
    (∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℕ), 
      is_valid_triple a₁ b₁ c₁ ∧ 
      is_valid_triple a₂ b₂ c₂ ∧ 
      sum_triple a₁ b₁ c₁ = s ∧ 
      sum_triple a₂ b₂ c₂ = s ∧ 
      (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)) ∧
    (∀ (t : ℕ), 
      t ≠ s → 
      ∀ (x y z u v w : ℕ), 
        is_valid_triple x y z → 
        is_valid_triple u v w → 
        sum_triple x y z = t → 
        sum_triple u v w = t → 
        (x, y, z) = (u, v, w)) →
  s = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_ambiguous_sum_l1681_168122


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_y_l1681_168171

theorem gcd_polynomial_and_y (y : ℤ) (h : ∃ k : ℤ, y = 46896 * k) :
  let g := fun (y : ℤ) => (3*y+5)*(8*y+3)*(16*y+9)*(y+16)
  Nat.gcd (Int.natAbs (g y)) (Int.natAbs y) = 2160 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_y_l1681_168171


namespace NUMINAMATH_CALUDE_sum_of_squares_of_conjugates_l1681_168140

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_squares_of_conjugates : (1 + i)^2 + (1 - i)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_conjugates_l1681_168140


namespace NUMINAMATH_CALUDE_race_probability_l1681_168199

theorem race_probability (total_cars : ℕ) (prob_X prob_Y prob_total : ℝ) : 
  total_cars = 12 →
  prob_X = 1/6 →
  prob_Y = 1/10 →
  prob_total = 0.39166666666666666 →
  prob_total = prob_X + prob_Y + (0.125 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_race_probability_l1681_168199


namespace NUMINAMATH_CALUDE_smallest_positive_b_l1681_168105

/-- Circle w1 defined by the equation x^2+y^2+6x-8y-23=0 -/
def w1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y - 23 = 0

/-- Circle w2 defined by the equation x^2+y^2-6x-8y+65=0 -/
def w2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 65 = 0

/-- A circle is externally tangent to w2 -/
def externally_tangent_w2 (x y r : ℝ) : Prop := 
  r + 2 = Real.sqrt ((x - 3)^2 + (y - 4)^2)

/-- A circle is internally tangent to w1 -/
def internally_tangent_w1 (x y r : ℝ) : Prop := 
  6 - r = Real.sqrt ((x + 3)^2 + (y - 4)^2)

/-- The line y = bx contains the center (x, y) of the tangent circle -/
def center_on_line (x y b : ℝ) : Prop := y = b * x

theorem smallest_positive_b : 
  ∃ (b : ℝ), b > 0 ∧ 
  (∀ (b' : ℝ), b' > 0 → 
    (∃ (x y r : ℝ), externally_tangent_w2 x y r ∧ 
                    internally_tangent_w1 x y r ∧ 
                    center_on_line x y b') 
    → b ≤ b') ∧
  b = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_b_l1681_168105


namespace NUMINAMATH_CALUDE_nineteen_ninetyeight_impossible_l1681_168118

/-- The type of operations that can be performed on a number -/
inductive Operation
| Square : Operation
| AddOne : Operation

/-- Apply an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Square => n * n
  | Operation.AddOne => n + 1

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- Apply a sequence of operations to a number -/
def applySequence (n : ℕ) (seq : OperationSequence) : ℕ :=
  seq.foldl applyOperation n

/-- The theorem stating that 19 and 98 cannot be made equal with the same number of operations -/
theorem nineteen_ninetyeight_impossible :
  ∀ (seq : OperationSequence), applySequence 19 seq ≠ applySequence 98 seq :=
sorry

end NUMINAMATH_CALUDE_nineteen_ninetyeight_impossible_l1681_168118


namespace NUMINAMATH_CALUDE_power_zero_simplify_expression_l1681_168137

-- Theorem 1: For any real number x ≠ 0, x^0 = 1
theorem power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

-- Theorem 2: For any real numbers a and b, (-2a^2)^2 * 3ab^2 = 12a^5b^2
theorem simplify_expression (a b : ℝ) : (-2*a^2)^2 * 3*a*b^2 = 12*a^5*b^2 := by sorry

end NUMINAMATH_CALUDE_power_zero_simplify_expression_l1681_168137


namespace NUMINAMATH_CALUDE_prime_cube_minus_one_not_divisible_by_40_l1681_168103

theorem prime_cube_minus_one_not_divisible_by_40 (p : ℕ) (hp : Prime p) (hp_ge_7 : p ≥ 7) :
  ¬(40 ∣ p^3 - 1) :=
sorry

end NUMINAMATH_CALUDE_prime_cube_minus_one_not_divisible_by_40_l1681_168103


namespace NUMINAMATH_CALUDE_inequality_relation_l1681_168114

theorem inequality_relation (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end NUMINAMATH_CALUDE_inequality_relation_l1681_168114


namespace NUMINAMATH_CALUDE_scientific_notation_of_32100000_l1681_168115

theorem scientific_notation_of_32100000 : 
  32100000 = 3.21 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_32100000_l1681_168115


namespace NUMINAMATH_CALUDE_committee_count_l1681_168198

/-- Represents a department with male and female professors -/
structure Department where
  male_profs : Nat
  female_profs : Nat

/-- Represents the configuration of the science division -/
structure ScienceDivision where
  departments : Fin 3 → Department

/-- Represents a committee formation -/
structure Committee where
  members : Fin 6 → Nat
  department_count : Fin 3 → Nat
  male_count : Nat
  female_count : Nat

def is_valid_committee (sd : ScienceDivision) (c : Committee) : Prop :=
  c.male_count = 3 ∧ 
  c.female_count = 3 ∧ 
  (∀ d : Fin 3, c.department_count d = 2)

def count_valid_committees (sd : ScienceDivision) : Nat :=
  sorry

theorem committee_count (sd : ScienceDivision) : 
  (∀ d : Fin 3, sd.departments d = ⟨3, 3⟩) → 
  count_valid_committees sd = 1215 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_l1681_168198


namespace NUMINAMATH_CALUDE_right_triangle_square_areas_l1681_168188

theorem right_triangle_square_areas (P Q R : ℝ × ℝ) 
  (right_angle_Q : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0)
  (square_QR_area : (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = 144)
  (square_PR_area : (R.1 - P.1)^2 + (R.2 - P.2)^2 = 169) :
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_square_areas_l1681_168188


namespace NUMINAMATH_CALUDE_m_range_theorem_l1681_168184

/-- The range of values for m satisfying the given conditions -/
def m_range (m : ℝ) : Prop :=
  (-2 ≤ m ∧ m < 1) ∨ m > 2

/-- Condition p: The solution set of x^2 + mx + 1 < 0 is empty -/
def condition_p (m : ℝ) : Prop :=
  ∀ x, x^2 + m*x + 1 ≥ 0

/-- Condition q: The function 4x^2 + 4(m-1)x + 3 has no extreme value -/
def condition_q (m : ℝ) : Prop :=
  ∀ x, 8*x + 4*(m-1) ≠ 0

/-- Theorem stating the range of m given the conditions -/
theorem m_range_theorem (m : ℝ)
  (h1 : condition_p m ∨ condition_q m)
  (h2 : ¬(condition_p m ∧ condition_q m)) :
  m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l1681_168184


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1681_168129

theorem units_digit_sum_of_powers : (2016^2017 + 2017^2016) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1681_168129


namespace NUMINAMATH_CALUDE_three_planes_max_parts_l1681_168161

/-- The maximum number of parts that three planes can divide three-dimensional space into -/
def max_parts_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of parts that three planes can divide three-dimensional space into is 8 -/
theorem three_planes_max_parts :
  max_parts_three_planes = 8 := by sorry

end NUMINAMATH_CALUDE_three_planes_max_parts_l1681_168161


namespace NUMINAMATH_CALUDE_go_stones_problem_l1681_168177

theorem go_stones_problem (total : ℕ) (difference_result : ℕ) 
  (h_total : total = 6000)
  (h_difference : difference_result = 4800) :
  ∃ (white black : ℕ), 
    white + black = total ∧ 
    white > black ∧ 
    total - (white - black) = difference_result ∧
    white = 3600 := by
  sorry

end NUMINAMATH_CALUDE_go_stones_problem_l1681_168177


namespace NUMINAMATH_CALUDE_pauls_crayons_left_l1681_168147

/-- Represents the number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 336

/-- Represents the initial number of crayons Paul got -/
def initial_crayons : ℕ := 601

/-- Represents the number of erasers Paul got -/
def erasers : ℕ := 406

theorem pauls_crayons_left :
  crayons_left = 336 ∧
  initial_crayons = 601 ∧
  erasers = 406 ∧
  erasers = crayons_left + 70 :=
by sorry

end NUMINAMATH_CALUDE_pauls_crayons_left_l1681_168147


namespace NUMINAMATH_CALUDE_hamburger_cost_satisfies_conditions_l1681_168173

/-- The cost of a pack of hamburger meat that satisfies the given conditions -/
def hamburger_cost : ℝ :=
  let crackers : ℝ := 3.50
  let vegetables : ℝ := 4 * 2.00
  let cheese : ℝ := 3.50
  let discount_rate : ℝ := 0.10
  let total_after_discount : ℝ := 18.00
  5.00

/-- Theorem stating that the hamburger cost satisfies the given conditions -/
theorem hamburger_cost_satisfies_conditions :
  let crackers : ℝ := 3.50
  let vegetables : ℝ := 4 * 2.00
  let cheese : ℝ := 3.50
  let discount_rate : ℝ := 0.10
  let total_after_discount : ℝ := 18.00
  total_after_discount = (hamburger_cost + crackers + vegetables + cheese) * (1 - discount_rate) := by
  sorry

#eval hamburger_cost

end NUMINAMATH_CALUDE_hamburger_cost_satisfies_conditions_l1681_168173


namespace NUMINAMATH_CALUDE_second_number_proof_l1681_168153

theorem second_number_proof (x : ℕ) : 
  (∃ k : ℕ, 60 = 18 * k + 6) →
  (∃ m : ℕ, x = 18 * m + 10) →
  (∀ d : ℕ, d > 18 → (d ∣ 60 ∧ d ∣ x) → False) →
  x > 60 →
  x = 64 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l1681_168153


namespace NUMINAMATH_CALUDE_distinct_values_mod_p_l1681_168121

theorem distinct_values_mod_p (p : ℕ) (a b : Fin p) (hp : Nat.Prime p) (hab : a ≠ b) :
  let f : Fin p → ℕ := λ n => (Finset.range (p - 1)).sum (λ i => (i + 1) * n^(i + 1))
  ¬ (f a ≡ f b [MOD p]) := by
  sorry

end NUMINAMATH_CALUDE_distinct_values_mod_p_l1681_168121


namespace NUMINAMATH_CALUDE_george_hourly_rate_l1681_168172

/-- Calculates the hourly rate given total income and hours worked -/
def hourly_rate (total_income : ℚ) (total_hours : ℚ) : ℚ :=
  total_income / total_hours

theorem george_hourly_rate :
  let monday_hours : ℚ := 7
  let tuesday_hours : ℚ := 2
  let total_hours : ℚ := monday_hours + tuesday_hours
  let total_income : ℚ := 45
  hourly_rate total_income total_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_george_hourly_rate_l1681_168172


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1681_168144

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    for all n, a(n+1) = r * a(n) -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : IsGeometric a) 
    (h1 : a 1 * a 99 = 16) : a 20 * a 80 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1681_168144


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l1681_168156

/-- Calculates the surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Calculates the total cost of insulation for a rectangular tank -/
def insulation_cost (length width height cost_per_sqft : ℝ) : ℝ :=
  surface_area length width height * cost_per_sqft

/-- Theorem: The cost of insulating a 4x5x2 feet tank at $20 per square foot is $1520 -/
theorem tank_insulation_cost :
  insulation_cost 4 5 2 20 = 1520 := by
  sorry

#eval insulation_cost 4 5 2 20

end NUMINAMATH_CALUDE_tank_insulation_cost_l1681_168156


namespace NUMINAMATH_CALUDE_product_scaling_l1681_168157

theorem product_scaling (a b c : ℝ) (h : (a * 100) * (b * 100) = c) : 
  a * b = c / 10000 := by
  sorry

end NUMINAMATH_CALUDE_product_scaling_l1681_168157


namespace NUMINAMATH_CALUDE_probability_six_green_out_of_ten_l1681_168148

/-- The probability of drawing exactly k green marbles out of n draws with replacement from a bag containing g green marbles and b blue marbles. -/
def probability_k_green (g b n k : ℕ) : ℚ :=
  (n.choose k) * (g / (g + b))^k * (b / (g + b))^(n - k)

/-- The theorem stating the probability of drawing exactly 6 green marbles out of 10 draws with replacement from a bag containing 10 green marbles and 5 blue marbles. -/
theorem probability_six_green_out_of_ten : 
  probability_k_green 10 5 10 6 = 210 * (2/3)^6 * (1/3)^4 := by sorry

end NUMINAMATH_CALUDE_probability_six_green_out_of_ten_l1681_168148


namespace NUMINAMATH_CALUDE_factor_implies_k_value_l1681_168180

theorem factor_implies_k_value (k : ℚ) :
  (∀ x, (3 * x + 4) ∣ (9 * x^3 + k * x^2 + 16 * x + 64)) →
  k = -12 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_k_value_l1681_168180


namespace NUMINAMATH_CALUDE_division_of_fractions_l1681_168183

theorem division_of_fractions :
  (-4 / 5) / (8 / 25) = -5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l1681_168183


namespace NUMINAMATH_CALUDE_gcd_problem_l1681_168151

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = (2*k + 1) * 1183) :
  Int.gcd (2*a^2 + 29*a + 65) (a + 13) = 26 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1681_168151


namespace NUMINAMATH_CALUDE_least_m_satisfying_condition_l1681_168131

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The problem statement -/
theorem least_m_satisfying_condition : ∃ m : ℕ, 
  (m > 0) ∧ 
  (∀ k : ℕ, k > 0 → k < m → 
    ¬(∃ p : ℕ, trailingZeros k = p ∧ 
      trailingZeros (2 * k) = ⌊(5 * p : ℚ) / 2⌋)) ∧
  (∃ p : ℕ, trailingZeros m = p ∧ 
    trailingZeros (2 * m) = ⌊(5 * p : ℚ) / 2⌋) ∧
  m = 25 := by
  sorry

end NUMINAMATH_CALUDE_least_m_satisfying_condition_l1681_168131


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1681_168167

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.0000037 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.7 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1681_168167


namespace NUMINAMATH_CALUDE_square_function_is_even_l1681_168197

/-- The function f(x) = x^2 is an even function for all real numbers x. -/
theorem square_function_is_even : ∀ x : ℝ, (fun x => x^2) (-x) = (fun x => x^2) x := by
  sorry

end NUMINAMATH_CALUDE_square_function_is_even_l1681_168197


namespace NUMINAMATH_CALUDE_ellipse_area_irrational_l1681_168101

/-- The area of an ellipse with rational semi-major and semi-minor axes is irrational -/
theorem ellipse_area_irrational (a b : ℚ) (h_a : a > 0) (h_b : b > 0) : 
  Irrational (Real.pi * (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_area_irrational_l1681_168101


namespace NUMINAMATH_CALUDE_john_personal_payment_l1681_168186

def hearing_aid_cost : ℝ := 2500
def insurance_coverage_percentage : ℝ := 80
def number_of_hearing_aids : ℕ := 2

theorem john_personal_payment (total_cost : ℝ) (insurance_payment : ℝ) (john_payment : ℝ) :
  total_cost = number_of_hearing_aids * hearing_aid_cost →
  insurance_payment = (insurance_coverage_percentage / 100) * total_cost →
  john_payment = total_cost - insurance_payment →
  john_payment = 1000 := by
sorry

end NUMINAMATH_CALUDE_john_personal_payment_l1681_168186


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l1681_168165

/-- 
Proves that the number of years until a man's age is twice his son's age is 2,
given the initial conditions of their ages.
-/
theorem mans_age_twice_sons (man_age son_age : ℕ) (y : ℕ) : 
  man_age = son_age + 26 →
  son_age = 24 →
  man_age + y = 2 * (son_age + y) →
  y = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l1681_168165


namespace NUMINAMATH_CALUDE_amara_clothing_donation_l1681_168108

theorem amara_clothing_donation :
  ∀ (initial remaining thrown_away : ℕ) (first_donation : ℕ),
    initial = 100 →
    remaining = 65 →
    thrown_away = 15 →
    initial - remaining = first_donation + 3 * first_donation + thrown_away →
    first_donation = 5 := by
  sorry

end NUMINAMATH_CALUDE_amara_clothing_donation_l1681_168108


namespace NUMINAMATH_CALUDE_yoga_time_calculation_l1681_168112

/-- Represents the ratio of time spent on different activities -/
structure ActivityRatio where
  swimming : ℕ
  running : ℕ
  gym : ℕ
  biking : ℕ
  yoga : ℕ

/-- Calculates the time spent on yoga based on the given ratio and biking time -/
def yoga_time (ratio : ActivityRatio) (biking_time : ℕ) : ℕ :=
  (biking_time * ratio.yoga) / ratio.biking

/-- Theorem: Given the specified ratio and 30 minutes of biking, the yoga time is 24 minutes -/
theorem yoga_time_calculation (ratio : ActivityRatio)
    (h1 : ratio.swimming = 1)
    (h2 : ratio.running = 2)
    (h3 : ratio.gym = 3)
    (h4 : ratio.biking = 5)
    (h5 : ratio.yoga = 4)
    (h6 : yoga_time ratio 30 = 24) : True := by
  sorry

#check yoga_time_calculation

end NUMINAMATH_CALUDE_yoga_time_calculation_l1681_168112


namespace NUMINAMATH_CALUDE_total_yards_run_l1681_168141

def malik_yards : ℕ := 18
def malik_games : ℕ := 5

def josiah_yards : ℕ := 22
def josiah_games : ℕ := 7

def darnell_yards : ℕ := 11
def darnell_games : ℕ := 4

def kade_yards : ℕ := 15
def kade_games : ℕ := 6

theorem total_yards_run : 
  malik_yards * malik_games + 
  josiah_yards * josiah_games + 
  darnell_yards * darnell_games + 
  kade_yards * kade_games = 378 := by
sorry

end NUMINAMATH_CALUDE_total_yards_run_l1681_168141


namespace NUMINAMATH_CALUDE_units_digit_of_3_power_2020_l1681_168149

theorem units_digit_of_3_power_2020 : ∃ n : ℕ, 3^2020 ≡ 1 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_3_power_2020_l1681_168149


namespace NUMINAMATH_CALUDE_brians_pencils_l1681_168126

/-- Given Brian's initial pencil count, the number he gives away, and the number he buys,
    prove that his final pencil count is equal to the initial count minus the number given away
    plus the number bought. -/
theorem brians_pencils (initial : ℕ) (given_away : ℕ) (bought : ℕ) :
  initial - given_away + bought = initial - given_away + bought :=
by sorry

end NUMINAMATH_CALUDE_brians_pencils_l1681_168126


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficient_l1681_168104

theorem quadratic_equation_coefficient (p q : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + p) = x^2 + q*x + 12) → q = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficient_l1681_168104


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1681_168139

theorem unique_integer_solution :
  ∃! x : ℤ, (x - 3 : ℚ) ^ (27 - x^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1681_168139


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1681_168192

def vector_a : ℝ × ℝ := (2, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -6)

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), k ≠ 0 ∧ vector_a = k • vector_b x) → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1681_168192


namespace NUMINAMATH_CALUDE_sqrt_expressions_l1681_168175

theorem sqrt_expressions :
  (∀ x y z : ℝ, x = 8 ∧ y = 2 ∧ z = 18 → Real.sqrt x + Real.sqrt y - Real.sqrt z = 0) ∧
  (∀ a : ℝ, a = 3 → (Real.sqrt a - 2)^2 = 7 - 4 * Real.sqrt a) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_l1681_168175


namespace NUMINAMATH_CALUDE_linear_system_elimination_l1681_168116

theorem linear_system_elimination (x y : ℝ) : 
  (6 * x - 5 * y = 3) → 
  (3 * x + y = -15) → 
  (5 * (3 * x + y) + (6 * x - 5 * y) = 21 * x) ∧ 
  (5 * (-15) + 3 = -72) := by
  sorry

end NUMINAMATH_CALUDE_linear_system_elimination_l1681_168116


namespace NUMINAMATH_CALUDE_rachel_brownies_l1681_168145

theorem rachel_brownies (total : ℚ) : 
  (3 / 5 : ℚ) * total = 18 → total = 30 := by
  sorry

end NUMINAMATH_CALUDE_rachel_brownies_l1681_168145


namespace NUMINAMATH_CALUDE_exactly_one_success_probability_l1681_168160

/-- The probability of success in a single trial -/
def p : ℚ := 1/3

/-- The number of trials -/
def n : ℕ := 3

/-- The number of successes we're interested in -/
def k : ℕ := 1

/-- The binomial coefficient function -/
def binomial_coef (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

/-- The probability of exactly k successes in n trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coef n k * p^k * (1 - p)^(n - k)

theorem exactly_one_success_probability :
  binomial_probability n k p = 4/9 := by sorry

end NUMINAMATH_CALUDE_exactly_one_success_probability_l1681_168160


namespace NUMINAMATH_CALUDE_double_age_in_two_years_l1681_168135

/-- Calculates the number of years until a man's age is twice his son's age -/
def years_until_double_age (man_age_difference : ℕ) (son_current_age : ℕ) : ℕ :=
  let man_current_age := son_current_age + man_age_difference
  2 * son_current_age + 2 - man_current_age

theorem double_age_in_two_years 
  (man_age_difference : ℕ) 
  (son_current_age : ℕ) 
  (h1 : man_age_difference = 25) 
  (h2 : son_current_age = 23) : 
  years_until_double_age man_age_difference son_current_age = 2 := by
sorry

#eval years_until_double_age 25 23

end NUMINAMATH_CALUDE_double_age_in_two_years_l1681_168135


namespace NUMINAMATH_CALUDE_complement_of_M_l1681_168174

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

theorem complement_of_M (x : ℝ) : x ∈ (Set.compl M) ↔ x < -2 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l1681_168174


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1681_168146

/-- Given a hyperbola with eccentricity 2 and foci coinciding with those of a specific ellipse,
    prove that its equation is x²/4 - y²/12 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h : ℝ × ℝ → Prop) (e : ℝ × ℝ → Prop) :
  (∀ x y, h (x, y) ↔ x^2/a^2 - y^2/b^2 = 1) →
  (∀ x y, e (x, y) ↔ x^2/25 + y^2/9 = 1) →
  (∀ x y, h (x, y) → (x/a)^2 - (y/b)^2 = 1) →
  (∀ x, e (x, 0) → x = 4 ∨ x = -4) →
  (∀ x, h (x, 0) → x = 4 ∨ x = -4) →
  (a / Real.sqrt (a^2 - b^2) = 2) →
  (∀ x y, h (x, y) ↔ x^2/4 - y^2/12 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1681_168146


namespace NUMINAMATH_CALUDE_debate_team_group_size_l1681_168190

/-- The size of each group in a debate team -/
def group_size (num_boys num_girls num_groups : ℕ) : ℕ :=
  (num_boys + num_girls) / num_groups

/-- Theorem: The size of each group in the debate team is 7 -/
theorem debate_team_group_size :
  group_size 11 45 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_group_size_l1681_168190


namespace NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l1681_168124

def N : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, 8; 4, 6, -2; -9, -3, 5]

def i : Matrix (Fin 3) (Fin 1) ℝ := !![1; 0; 0]
def j : Matrix (Fin 3) (Fin 1) ℝ := !![0; 1; 0]
def k : Matrix (Fin 3) (Fin 1) ℝ := !![0; 0; 1]

theorem matrix_N_satisfies_conditions :
  N * i = !![3; 4; -9] ∧
  N * j = !![1; 6; -3] ∧
  N * k = !![8; -2; 5] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l1681_168124


namespace NUMINAMATH_CALUDE_d_value_l1681_168170

theorem d_value (d : ℚ) (h : 10 * d + 8 = 528) : 2 * d = 104 := by
  sorry

end NUMINAMATH_CALUDE_d_value_l1681_168170


namespace NUMINAMATH_CALUDE_ast_equation_solutions_l1681_168169

-- Define the operation ※
def ast (a b : ℝ) : ℝ := a + b^2

-- Theorem statement
theorem ast_equation_solutions :
  ∃! (s : Set ℝ), s = {x : ℝ | ast x (x + 1) = 5} ∧ s = {1, -4} :=
by sorry

end NUMINAMATH_CALUDE_ast_equation_solutions_l1681_168169


namespace NUMINAMATH_CALUDE_square_coins_problem_l1681_168155

theorem square_coins_problem (perimeter_coins : ℕ) (h : perimeter_coins = 240) :
  let side_length := (perimeter_coins + 4) / 4
  side_length * side_length = 3721 := by
  sorry

end NUMINAMATH_CALUDE_square_coins_problem_l1681_168155


namespace NUMINAMATH_CALUDE_complex_square_eq_neg_two_i_l1681_168143

theorem complex_square_eq_neg_two_i (z : ℂ) (a b : ℝ) :
  z = Complex.mk a b → z^2 = Complex.I * (-2) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_eq_neg_two_i_l1681_168143


namespace NUMINAMATH_CALUDE_existence_of_primes_with_gcd_one_l1681_168128

theorem existence_of_primes_with_gcd_one (n : ℕ) (h1 : n > 6) (h2 : Even n) :
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ Nat.gcd (n - p) (n - q) = 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_primes_with_gcd_one_l1681_168128


namespace NUMINAMATH_CALUDE_jan_beth_money_difference_l1681_168123

theorem jan_beth_money_difference (beth_money jan_money : ℕ) : 
  beth_money + 35 = 105 →
  beth_money + jan_money = 150 →
  jan_money - beth_money = 10 := by
sorry

end NUMINAMATH_CALUDE_jan_beth_money_difference_l1681_168123


namespace NUMINAMATH_CALUDE_career_preference_representation_l1681_168196

theorem career_preference_representation (total_students : ℕ) 
  (male_ratio female_ratio : ℕ) (male_preference female_preference : ℕ) : 
  total_students = 30 →
  male_ratio = 2 →
  female_ratio = 3 →
  male_preference = 2 →
  female_preference = 3 →
  (((male_preference + female_preference : ℝ) / total_students) * 360 : ℝ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_career_preference_representation_l1681_168196


namespace NUMINAMATH_CALUDE_coloring_arrangements_l1681_168178

/-- The number of ways to arrange n distinct objects into n distinct positions -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of parts to be colored -/
def num_parts : ℕ := 4

/-- The number of colors available -/
def num_colors : ℕ := 4

/-- Theorem: The number of ways to color 4 distinct parts with 4 distinct colors, 
    where each part must have a different color, is equal to 24 -/
theorem coloring_arrangements : permutations num_parts = 24 := by
  sorry

end NUMINAMATH_CALUDE_coloring_arrangements_l1681_168178


namespace NUMINAMATH_CALUDE_dasha_number_l1681_168195

-- Define a function to calculate the product of digits
def digitProduct (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digitProduct (n / 10)

-- Define a function to check if a number is single-digit
def isSingleDigit (n : ℕ) : Prop := n < 10

-- Theorem statement
theorem dasha_number (n : ℕ) :
  n ≤ digitProduct n → isSingleDigit n :=
by sorry

end NUMINAMATH_CALUDE_dasha_number_l1681_168195
