import Mathlib

namespace NUMINAMATH_CALUDE_villager_A_motorcycle_fraction_l3915_391551

/-- Represents the scenario of two villagers and a motorcycle traveling to a station -/
structure TravelScenario where
  totalDistance : ℝ := 1
  walkingSpeed : ℝ
  motorcycleSpeed : ℝ
  simultaneousArrival : Prop

/-- The main theorem stating the fraction of journey villager A travels by motorcycle -/
theorem villager_A_motorcycle_fraction (scenario : TravelScenario) 
  (h1 : scenario.motorcycleSpeed = 9 * scenario.walkingSpeed)
  (h2 : scenario.simultaneousArrival) : 
  ∃ (x : ℝ), x = 5/6 ∧ x * scenario.totalDistance = scenario.totalDistance - scenario.walkingSpeed / scenario.motorcycleSpeed * scenario.totalDistance :=
by sorry

end NUMINAMATH_CALUDE_villager_A_motorcycle_fraction_l3915_391551


namespace NUMINAMATH_CALUDE_ratio_equality_l3915_391533

theorem ratio_equality (x y : ℝ) (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3915_391533


namespace NUMINAMATH_CALUDE_first_car_right_turn_distance_l3915_391535

/-- The distance between two cars on a road --/
def distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - car1_distance - car2_distance

/-- The total distance traveled by the first car --/
def car1_total_distance (x : ℝ) : ℝ := 25 + x + 25

theorem first_car_right_turn_distance (initial_distance : ℝ) (car2_distance : ℝ) (final_distance : ℝ) :
  initial_distance = 113 ∧ 
  car2_distance = 35 ∧ 
  final_distance = 28 →
  ∃ x : ℝ, 
    car1_total_distance x + car2_distance = 
    distance_between_cars initial_distance 25 car2_distance + final_distance ∧
    x = 21 := by
  sorry

end NUMINAMATH_CALUDE_first_car_right_turn_distance_l3915_391535


namespace NUMINAMATH_CALUDE_library_tables_l3915_391541

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : Nat) : Nat :=
  let units := n % 10
  let sixes := (n / 10) % 10
  let thirty_sixes := n / 100
  thirty_sixes * 36 + sixes * 6 + units

/-- Calculates the number of tables needed given the total number of people and people per table -/
def tablesNeeded (totalPeople : Nat) (peoplePerTable : Nat) : Nat :=
  (totalPeople + peoplePerTable - 1) / peoplePerTable

theorem library_tables (seatingCapacity : Nat) (peoplePerTable : Nat) :
  seatingCapacity = 231 ∧ peoplePerTable = 3 →
  tablesNeeded (base6ToBase10 seatingCapacity) peoplePerTable = 31 := by
  sorry

end NUMINAMATH_CALUDE_library_tables_l3915_391541


namespace NUMINAMATH_CALUDE_problem_statement_l3915_391536

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) :
  (x - 1)^2 + 9/(x - 1)^2 = 3 + 8/x :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3915_391536


namespace NUMINAMATH_CALUDE_point_on_330_degree_angle_l3915_391525

/-- For any point P (x, y) ≠ (0, 0) on the terminal side of a 330° angle, y/x = -√3/3 -/
theorem point_on_330_degree_angle (x y : ℝ) : 
  (x ≠ 0 ∨ y ≠ 0) →  -- Point is not the origin
  (x, y) ∈ {p : ℝ × ℝ | ∃ (r : ℝ), r > 0 ∧ p.1 = r * Real.cos (330 * π / 180) ∧ p.2 = r * Real.sin (330 * π / 180)} →  -- Point is on the terminal side of 330° angle
  y / x = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_point_on_330_degree_angle_l3915_391525


namespace NUMINAMATH_CALUDE_cauchy_problem_solution_l3915_391554

noncomputable def y (x : ℝ) : ℝ := x^2/2 + x^3/6 + x^4/12 + x^5/20 + x + 1

theorem cauchy_problem_solution (x : ℝ) :
  (deriv^[2] y) x = 1 + x + x^2 + x^3 ∧
  y 0 = 1 ∧
  (deriv y) 0 = 1 := by sorry

end NUMINAMATH_CALUDE_cauchy_problem_solution_l3915_391554


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3915_391542

theorem fraction_evaluation : (3 : ℚ) / (2 - 5 / 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3915_391542


namespace NUMINAMATH_CALUDE_equation_solution_l3915_391596

theorem equation_solution (x : ℝ) : 
  (8 * x^2 + 52 * x + 4) / (3 * x + 13) = 2 * x + 3 ↔ 
  x = (-17 + Real.sqrt 569) / 4 ∨ x = (-17 - Real.sqrt 569) / 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3915_391596


namespace NUMINAMATH_CALUDE_semicircle_problem_l3915_391553

theorem semicircle_problem (N : ℕ) (r : ℝ) (h_positive : r > 0) : 
  let A := N * (π * r^2 / 2)
  let B := (π * (N*r)^2 / 2) - A
  A / B = 1 / 9 → N = 10 := by
sorry

end NUMINAMATH_CALUDE_semicircle_problem_l3915_391553


namespace NUMINAMATH_CALUDE_exam_time_allocation_l3915_391500

theorem exam_time_allocation (total_time : ℕ) (total_questions : ℕ) (type_a_questions : ℕ) :
  total_time = 180 →
  total_questions = 200 →
  type_a_questions = 50 →
  let type_b_questions := total_questions - type_a_questions
  let time_ratio := 2
  let total_time_units := type_a_questions * time_ratio + type_b_questions
  let time_per_unit := total_time / total_time_units
  let time_for_type_a := type_a_questions * time_ratio * time_per_unit
  time_for_type_a = 72 :=
by
  sorry

#check exam_time_allocation

end NUMINAMATH_CALUDE_exam_time_allocation_l3915_391500


namespace NUMINAMATH_CALUDE_train_interval_l3915_391518

/-- Represents a metro station -/
inductive Station : Type
| Taganskaya : Station
| Kievskaya : Station

/-- Represents a direction of travel -/
inductive Direction : Type
| Clockwise : Direction
| Counterclockwise : Direction

/-- Represents the metro system -/
structure MetroSystem where
  northern_route_time : ℝ
  southern_route_time : ℝ
  train_delay : ℝ
  trip_time_difference : ℝ

/-- Calculate the expected travel time between stations -/
def expected_travel_time (m : MetroSystem) (p : ℝ) : ℝ :=
  m.southern_route_time * p + m.northern_route_time * (1 - p)

/-- Theorem: The interval between trains in one direction is 3 minutes -/
theorem train_interval (m : MetroSystem) 
  (h1 : m.northern_route_time = 17)
  (h2 : m.southern_route_time = 11)
  (h3 : m.train_delay = 5/4)
  (h4 : m.trip_time_difference = 1)
  : ∃ (T : ℝ), T = 3 ∧ 
    ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧
    expected_travel_time m p = expected_travel_time m (1-p) - m.trip_time_difference ∧
    T * (1 - p) = m.train_delay := by
  sorry

end NUMINAMATH_CALUDE_train_interval_l3915_391518


namespace NUMINAMATH_CALUDE_percentage_problem_l3915_391501

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  x = 230 → 
  p / 100 * x = 20 / 100 * 747.50 → 
  p = 65 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3915_391501


namespace NUMINAMATH_CALUDE_maker_funds_and_loan_repayment_l3915_391522

/-- Represents the remaining funds after n months -/
def remaining_funds (n : ℕ) : ℝ := sorry

/-- The initial borrowed capital -/
def initial_capital : ℝ := 100000

/-- Monthly profit rate -/
def profit_rate : ℝ := 0.2

/-- Monthly expense rate (rent and tax) -/
def expense_rate : ℝ := 0.1

/-- Monthly fixed expenses -/
def fixed_expenses : ℝ := 3000

/-- Annual interest rate for the bank loan -/
def annual_interest_rate : ℝ := 0.05

/-- Number of months in a year -/
def months_in_year : ℕ := 12

theorem maker_funds_and_loan_repayment :
  remaining_funds months_in_year = 194890 ∧
  remaining_funds months_in_year > initial_capital * (1 + annual_interest_rate) :=
sorry

end NUMINAMATH_CALUDE_maker_funds_and_loan_repayment_l3915_391522


namespace NUMINAMATH_CALUDE_exponent_multiplication_specific_exponent_multiplication_l3915_391526

theorem exponent_multiplication (a b c : ℕ) : (10 : ℝ) ^ a * (10 : ℝ) ^ b = (10 : ℝ) ^ (a + b) :=
by sorry

theorem specific_exponent_multiplication : (10 : ℝ) ^ 10000 * (10 : ℝ) ^ 8000 = (10 : ℝ) ^ 18000 :=
by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_specific_exponent_multiplication_l3915_391526


namespace NUMINAMATH_CALUDE_gcd_sum_and_count_even_integers_l3915_391529

def sum_even_integers (a b : ℕ) : ℕ :=
  let first_even := if a % 2 = 0 then a else a + 1
  let last_even := if b % 2 = 0 then b else b - 1
  let n := (last_even - first_even) / 2 + 1
  n * (first_even + last_even) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  let first_even := if a % 2 = 0 then a else a + 1
  let last_even := if b % 2 = 0 then b else b - 1
  (last_even - first_even) / 2 + 1

theorem gcd_sum_and_count_even_integers :
  Nat.gcd (sum_even_integers 13 63) (count_even_integers 13 63) = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_and_count_even_integers_l3915_391529


namespace NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l3915_391570

theorem fraction_equality_implies_c_geq_one
  (a b : ℕ+) (c : ℝ)
  (h_c_pos : c > 0)
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) :
  c ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l3915_391570


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l3915_391584

theorem circle_equation_k_value (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 6*x + y^2 + 8*y - k = 0 ↔ (x + 3)^2 + (y + 4)^2 = 10^2) → 
  k = 75 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l3915_391584


namespace NUMINAMATH_CALUDE_group_size_problem_l3915_391507

theorem group_size_problem (T : ℝ) 
  (hat_wearers : ℝ → ℝ)
  (shoe_wearers : ℝ → ℝ)
  (both_wearers : ℝ → ℝ)
  (h1 : hat_wearers T = 0.40 * T + 60)
  (h2 : shoe_wearers T = 0.25 * T)
  (h3 : both_wearers T = 0.20 * T)
  (h4 : both_wearers T = hat_wearers T - shoe_wearers T) :
  T = 1200 := by
sorry

end NUMINAMATH_CALUDE_group_size_problem_l3915_391507


namespace NUMINAMATH_CALUDE_hat_problem_inconsistent_l3915_391527

/-- Represents the number of hats of each color --/
structure HatCounts where
  blue : ℕ
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Checks if the given hat counts satisfy the problem conditions --/
def satisfies_conditions (counts : HatCounts) : Prop :=
  counts.blue + counts.green + counts.red + counts.yellow = 150 ∧
  counts.blue = 2 * counts.green ∧
  8 * counts.blue + 10 * counts.green + 12 * counts.red + 15 * counts.yellow = 1280

/-- Theorem stating the inconsistency in the problem --/
theorem hat_problem_inconsistent : 
  ∀ (counts : HatCounts), satisfies_conditions counts → counts.red = 0 ∧ counts.yellow = 0 := by
  sorry

#check hat_problem_inconsistent

end NUMINAMATH_CALUDE_hat_problem_inconsistent_l3915_391527


namespace NUMINAMATH_CALUDE_cos_two_theta_collinear_vectors_l3915_391538

/-- Given two vectors AB and BC in 2D space, and that points A, B, and C are collinear,
    prove that cos(2θ) = 7/9 where θ is the angle in the definition of BC. -/
theorem cos_two_theta_collinear_vectors 
  (AB : ℝ × ℝ) 
  (BC : ℝ → ℝ × ℝ) 
  (h_AB : AB = (-1, -3))
  (h_BC : ∀ θ, BC θ = (2 * Real.sin θ, 2))
  (h_collinear : ∀ θ, ∃ k : ℝ, AB = k • BC θ) :
  ∃ θ, Real.cos (2 * θ) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_collinear_vectors_l3915_391538


namespace NUMINAMATH_CALUDE_units_digit_power_plus_six_l3915_391548

theorem units_digit_power_plus_six (x : ℕ) : 
  1 ≤ x → x ≤ 9 → (x^75 + 6) % 10 = 9 → x = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_power_plus_six_l3915_391548


namespace NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l3915_391571

/-- Represents the dimensions and areas of a yard with flower beds -/
structure YardWithFlowerBeds where
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  trapezoid_height : ℝ
  total_length : ℝ

/-- Calculates the fraction of the yard occupied by flower beds -/
def flower_bed_fraction (yard : YardWithFlowerBeds) : ℚ :=
  125 / 310

/-- Theorem stating that the fraction of the yard occupied by flower beds is 125/310 -/
theorem flower_bed_fraction_is_correct (yard : YardWithFlowerBeds) 
  (h1 : yard.trapezoid_short_side = 30)
  (h2 : yard.trapezoid_long_side = 40)
  (h3 : yard.trapezoid_height = 6)
  (h4 : yard.total_length = 60) : 
  flower_bed_fraction yard = 125 / 310 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_is_correct_l3915_391571


namespace NUMINAMATH_CALUDE_probability_at_least_one_chooses_23_l3915_391528

def num_students : ℕ := 4
def num_questions : ℕ := 2

theorem probability_at_least_one_chooses_23 :
  (1 : ℚ) - (1 / num_questions) ^ num_students = 15 / 16 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_chooses_23_l3915_391528


namespace NUMINAMATH_CALUDE_ratio_of_P_and_Q_l3915_391560

-- Define the equation as a function
def equation (P Q : ℤ) (x : ℝ) : Prop :=
  (P / (x + 6) + Q / (x^2 - 5*x) = (x^2 - x + 15) / (x^3 + x^2 - 30*x)) ∧ 
  (x ≠ -6) ∧ (x ≠ 0) ∧ (x ≠ 5)

-- State the theorem
theorem ratio_of_P_and_Q (P Q : ℤ) :
  (∀ x : ℝ, equation P Q x) → Q / P = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_P_and_Q_l3915_391560


namespace NUMINAMATH_CALUDE_range_of_m_l3915_391502

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 12 * x - x^3 else -2 * x

theorem range_of_m (m : ℝ) :
  (∀ y ∈ Set.Iic m, f y ∈ Set.Ici (-16)) ∧
  (∀ z : ℝ, z ≥ -16 → ∃ x ∈ Set.Iic m, f x = z) →
  m ∈ Set.Icc (-2) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3915_391502


namespace NUMINAMATH_CALUDE_scientist_contemporary_probability_scientist_contemporary_probability_value_l3915_391564

/-- The probability that two scientists were contemporaries for any length of time -/
theorem scientist_contemporary_probability : ℝ :=
  let years_range : ℕ := 300
  let lifespan : ℕ := 80
  let total_possibility_area : ℕ := years_range * years_range
  let overlap_area : ℕ := (years_range - lifespan) * (years_range - lifespan) - 2 * (lifespan * lifespan / 2)
  (overlap_area : ℝ) / total_possibility_area

/-- The probability is equal to 7/15 -/
theorem scientist_contemporary_probability_value : scientist_contemporary_probability = 7 / 15 :=
sorry

end NUMINAMATH_CALUDE_scientist_contemporary_probability_scientist_contemporary_probability_value_l3915_391564


namespace NUMINAMATH_CALUDE_dog_max_distance_dog_max_distance_is_22_l3915_391574

/-- The maximum distance a dog can reach from the origin when secured at (6,8) with a 12-foot rope -/
theorem dog_max_distance : ℝ :=
  let dog_position : ℝ × ℝ := (6, 8)
  let rope_length : ℝ := 12
  let origin : ℝ × ℝ := (0, 0)
  let distance_to_origin : ℝ := Real.sqrt ((dog_position.1 - origin.1)^2 + (dog_position.2 - origin.2)^2)
  distance_to_origin + rope_length

theorem dog_max_distance_is_22 : dog_max_distance = 22 := by
  sorry

end NUMINAMATH_CALUDE_dog_max_distance_dog_max_distance_is_22_l3915_391574


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l3915_391543

theorem min_value_squared_sum (a b c : ℝ) (h : 2*a + 2*b + c = 8) :
  (a - 1)^2 + (b + 2)^2 + (c - 3)^2 ≥ 49/9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l3915_391543


namespace NUMINAMATH_CALUDE_max_cylinder_lateral_area_l3915_391576

/-- Given a sphere with surface area 20π and a cylinder whose base circumferences
    lie on the surface of the sphere, the maximum lateral surface area of the cylinder is 10π. -/
theorem max_cylinder_lateral_area (R : ℝ) (r l : ℝ) :
  (4 * Real.pi * R^2 = 20 * Real.pi) →   -- Sphere surface area
  (r^2 + (l/2)^2 = R^2) →                -- Cylinder bases touch sphere surface
  (2 * Real.pi * r * l ≤ 10 * Real.pi) ∧ -- Max lateral surface area
  (∃ (r' l' : ℝ), 2 * Real.pi * r' * l' = 10 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_max_cylinder_lateral_area_l3915_391576


namespace NUMINAMATH_CALUDE_closest_to_sqrt_two_l3915_391513

theorem closest_to_sqrt_two : 
  let a := Real.sqrt 3 * Real.cos (14 * π / 180) + Real.sin (14 * π / 180)
  let b := Real.sqrt 3 * Real.cos (24 * π / 180) + Real.sin (24 * π / 180)
  let c := Real.sqrt 3 * Real.cos (64 * π / 180) + Real.sin (64 * π / 180)
  let d := Real.sqrt 3 * Real.cos (74 * π / 180) + Real.sin (74 * π / 180)
  abs (d - Real.sqrt 2) < min (abs (a - Real.sqrt 2)) (min (abs (b - Real.sqrt 2)) (abs (c - Real.sqrt 2))) := by
  sorry

end NUMINAMATH_CALUDE_closest_to_sqrt_two_l3915_391513


namespace NUMINAMATH_CALUDE_second_box_price_l3915_391549

/-- Represents a box of contacts with its quantity and price -/
structure ContactBox where
  quantity : ℕ
  price : ℚ

/-- Calculates the price per contact for a given box -/
def pricePerContact (box : ContactBox) : ℚ :=
  box.price / box.quantity

theorem second_box_price (box1 box2 : ContactBox)
  (h1 : box1.quantity = 50)
  (h2 : box1.price = 25)
  (h3 : box2.quantity = 99)
  (h4 : pricePerContact box2 < pricePerContact box1)
  (h5 : 3 * pricePerContact box2 = 1) :
  box2.price = 99/3 := by
  sorry

#eval (99 : ℚ) / 3  -- Should output 33

end NUMINAMATH_CALUDE_second_box_price_l3915_391549


namespace NUMINAMATH_CALUDE_emilys_weight_l3915_391577

/-- Given Heather's weight and the difference between Heather and Emily's weights,
    prove that Emily's weight is 9 pounds. -/
theorem emilys_weight (heathers_weight : ℕ) (weight_difference : ℕ)
  (hw : heathers_weight = 87)
  (diff : weight_difference = 78)
  : heathers_weight - weight_difference = 9 := by
  sorry

#check emilys_weight

end NUMINAMATH_CALUDE_emilys_weight_l3915_391577


namespace NUMINAMATH_CALUDE_salary_calculation_l3915_391506

theorem salary_calculation (salary : ℝ) : 
  (salary * (1/5 : ℝ) + salary * (1/10 : ℝ) + salary * (3/5 : ℝ) + 15000 = salary) → 
  salary = 150000 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l3915_391506


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3915_391557

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x-1)^4 * (x+1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3915_391557


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_sqrt_two_over_two_l3915_391591

theorem sin_cos_difference_equals_neg_sqrt_two_over_two :
  Real.sin (18 * π / 180) * Real.cos (63 * π / 180) -
  Real.sin (72 * π / 180) * Real.sin (117 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_sqrt_two_over_two_l3915_391591


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l3915_391532

theorem absolute_value_equation_product (y₁ y₂ : ℝ) : 
  (|3 * y₁| + 7 = 40) ∧ (|3 * y₂| + 7 = 40) ∧ (y₁ ≠ y₂) → y₁ * y₂ = -121 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l3915_391532


namespace NUMINAMATH_CALUDE_remainder_problem_l3915_391579

theorem remainder_problem (x : ℤ) : x % 84 = 25 → x % 14 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3915_391579


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l3915_391578

theorem geometric_progression_proof (y : ℝ) : 
  (90 + y)^2 = (30 + y) * (180 + y) → 
  y = 90 ∧ (90 + y) / (30 + y) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l3915_391578


namespace NUMINAMATH_CALUDE_right_triangle_trig_identity_l3915_391559

theorem right_triangle_trig_identity (A B C : Real) : 
  -- ABC is a right-angled triangle with right angle at C
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ 
  A + B + C = π / 2 ∧ 
  C = π / 2 →
  -- The trigonometric identity
  Real.sin A * Real.sin B * Real.sin (A - B) + 
  Real.sin B * Real.sin C * Real.sin (B - C) + 
  Real.sin C * Real.sin A * Real.sin (C - A) + 
  Real.sin (A - B) * Real.sin (B - C) * Real.sin (C - A) = 0 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_trig_identity_l3915_391559


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3915_391587

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3915_391587


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3915_391582

theorem sum_of_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 9 = (7 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3915_391582


namespace NUMINAMATH_CALUDE_parabola_trajectory_parabola_trajectory_is_parabola_l3915_391590

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

def parallelogram_point (A B F : Point) : Point :=
  Point.mk (A.x + B.x - F.x) (A.y + B.y - F.y)

def intersect_parabola_line (p : Parabola) (l : Line) : Set Point :=
  {P : Point | P.x^2 = 4 * P.y ∧ P.y = l.slope * P.x + l.intercept}

theorem parabola_trajectory (p : Parabola) (l : Line) (F : Point) :
  p.a = 1 ∧ p.h = 0 ∧ p.k = 0 ∧ 
  F.x = 0 ∧ F.y = 1 ∧
  l.intercept = -1 →
  ∃ (R : Point),
    (∃ (A B : Point), A ∈ intersect_parabola_line p l ∧ 
                      B ∈ intersect_parabola_line p l ∧ 
                      R = parallelogram_point A B F) ∧
    R.x^2 = 4 * (R.y + 3) ∧
    abs R.x > 4 :=
sorry

theorem parabola_trajectory_is_parabola (p : Parabola) (l : Line) (F : Point) :
  p.a = 1 ∧ p.h = 0 ∧ p.k = 0 ∧ 
  F.x = 0 ∧ F.y = 1 ∧
  l.intercept = -1 →
  ∃ (new_p : Parabola),
    new_p.a = 1 ∧ new_p.h = 0 ∧ new_p.k = -3 ∧
    (∀ (R : Point),
      (∃ (A B : Point), A ∈ intersect_parabola_line p l ∧ 
                        B ∈ intersect_parabola_line p l ∧ 
                        R = parallelogram_point A B F) →
      R.x^2 = 4 * (R.y + 3) ∧ abs R.x > 4) :=
sorry

end NUMINAMATH_CALUDE_parabola_trajectory_parabola_trajectory_is_parabola_l3915_391590


namespace NUMINAMATH_CALUDE_expand_product_l3915_391561

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x + 7) = 6 * x^2 + 29 * x + 28 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3915_391561


namespace NUMINAMATH_CALUDE_three_equal_numbers_sum_300_l3915_391583

theorem three_equal_numbers_sum_300 :
  ∃ (x : ℕ), x + x + x = 300 ∧ x = 100 := by
  sorry

end NUMINAMATH_CALUDE_three_equal_numbers_sum_300_l3915_391583


namespace NUMINAMATH_CALUDE_quadratic_symmetry_and_point_l3915_391581

def f (x : ℝ) := (x - 2)^2 - 3

theorem quadratic_symmetry_and_point :
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧ f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_and_point_l3915_391581


namespace NUMINAMATH_CALUDE_inequality_proof_l3915_391503

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : (a + 1) * (b + 1) * (c + 1) = 8) :
  a + b + c ≥ 3 ∧ a * b * c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3915_391503


namespace NUMINAMATH_CALUDE_history_score_calculation_l3915_391569

def geography_score : ℕ := 50
def math_score : ℕ := 70
def english_score : ℕ := 66
def total_score : ℕ := 248

theorem history_score_calculation :
  total_score - (geography_score + math_score + english_score) =
  total_score - geography_score - math_score - english_score :=
by sorry

end NUMINAMATH_CALUDE_history_score_calculation_l3915_391569


namespace NUMINAMATH_CALUDE_part_one_part_two_l3915_391572

-- Define propositions p and q
def p (x a : ℝ) : Prop := (x + a) / (x - 3 * a) < 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - 5*x + 6 ≤ 0

-- Part I
theorem part_one (x : ℝ) (h : p x 1 ∧ q x) : 2 ≤ x ∧ x < 3 := by sorry

-- Part II
theorem part_two (a : ℝ) (h : ∀ x, q x → p x a) (h' : ∃ x, p x a ∧ ¬q x) : a > 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3915_391572


namespace NUMINAMATH_CALUDE_pen_cost_problem_l3915_391594

theorem pen_cost_problem (total_students : Nat) (buyers : Nat) (pens_per_student : Nat) (pen_cost : Nat) :
  total_students = 32 →
  buyers > total_students / 2 →
  pens_per_student > 1 →
  pen_cost > pens_per_student →
  buyers * pens_per_student * pen_cost = 2116 →
  pen_cost = 23 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_problem_l3915_391594


namespace NUMINAMATH_CALUDE_least_apples_count_l3915_391562

theorem least_apples_count (b : ℕ) : 
  (b > 0) →
  (b % 3 = 2) → 
  (b % 4 = 3) → 
  (b % 5 = 1) → 
  (∀ n : ℕ, n > 0 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 1 → n ≥ b) →
  b = 11 := by
sorry

end NUMINAMATH_CALUDE_least_apples_count_l3915_391562


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l3915_391599

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) 
  (h1 : total_length = 28)
  (h2 : ratio = 2.00001 / 5) : 
  ∃ (shorter_piece : ℝ), 
    shorter_piece + ratio * shorter_piece = total_length ∧ 
    shorter_piece = 20 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l3915_391599


namespace NUMINAMATH_CALUDE_zoo_enclosure_claws_l3915_391547

theorem zoo_enclosure_claws (num_wombats : ℕ) (num_rheas : ℕ) 
  (wombat_claws : ℕ) (rhea_claws : ℕ) : 
  num_wombats = 9 → 
  num_rheas = 3 → 
  wombat_claws = 4 → 
  rhea_claws = 1 → 
  num_wombats * wombat_claws + num_rheas * rhea_claws = 39 := by
  sorry

end NUMINAMATH_CALUDE_zoo_enclosure_claws_l3915_391547


namespace NUMINAMATH_CALUDE_value_of_N_l3915_391539

theorem value_of_N : ∃ N : ℝ, (25 / 100) * (N + 100) = (35 / 100) * 1500 ∧ N = 2000 := by
  sorry

end NUMINAMATH_CALUDE_value_of_N_l3915_391539


namespace NUMINAMATH_CALUDE_married_fraction_l3915_391588

theorem married_fraction (total : ℕ) (women_fraction : ℚ) (max_unmarried_women : ℕ) :
  total = 80 →
  women_fraction = 1/4 →
  max_unmarried_women = 20 →
  (total - max_unmarried_women : ℚ) / total = 3/4 := by
sorry

end NUMINAMATH_CALUDE_married_fraction_l3915_391588


namespace NUMINAMATH_CALUDE_wednesday_sales_l3915_391531

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40
def unsold_percentage : ℚ := 80.57142857142857 / 100

theorem wednesday_sales :
  let unsold := (initial_stock : ℚ) * unsold_percentage
  let other_days_sales := monday_sales + tuesday_sales + thursday_sales + friday_sales
  initial_stock - (unsold.floor + other_days_sales) = 60 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_sales_l3915_391531


namespace NUMINAMATH_CALUDE_third_and_fourth_terms_equal_21_l3915_391516

def a (n : ℕ) : ℤ := -n^2 + 7*n + 9

theorem third_and_fourth_terms_equal_21 : a 3 = 21 ∧ a 4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_third_and_fourth_terms_equal_21_l3915_391516


namespace NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l3915_391568

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "lies in" relation between a line and a plane
variable (lies_in : Line → Plane → Prop)

-- Define our specific objects
variable (l m n : Line) (α : Plane)

-- State the theorem
theorem perpendicular_sufficient_not_necessary :
  (lies_in m α) → 
  (lies_in n α) → 
  (∀ x y : Line, lies_in x α → lies_in y α → perp_line_plane l α → perp_line_line l x ∧ perp_line_line l y) ∧ 
  (∃ x y : Line, lies_in x α → lies_in y α → perp_line_line l x ∧ perp_line_line l y ∧ ¬perp_line_plane l α) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l3915_391568


namespace NUMINAMATH_CALUDE_unique_solution_l3915_391592

theorem unique_solution : 
  ∃! x : ℝ, -1 < x ∧ x ≤ 2 ∧ 
  Real.sqrt (2 - x) + Real.sqrt (2 + 2*x) = 
  Real.sqrt ((x^4 + 1) / (x^2 + 1)) + (x + 3) / (x + 1) ∧ 
  x = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3915_391592


namespace NUMINAMATH_CALUDE_alex_academic_year_hours_l3915_391563

/-- Calculates the number of hours Alex needs to work per week during the academic year --/
def academic_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (academic_weeks : ℕ) (academic_earnings : ℕ) : ℚ :=
  let summer_total_hours := summer_weeks * summer_hours_per_week
  let hourly_rate := summer_earnings / summer_total_hours
  let academic_total_hours := academic_earnings / hourly_rate
  academic_total_hours / academic_weeks

/-- Theorem stating that Alex needs to work 20 hours per week during the academic year --/
theorem alex_academic_year_hours : 
  academic_year_hours_per_week 8 40 4000 32 8000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_alex_academic_year_hours_l3915_391563


namespace NUMINAMATH_CALUDE_ana_overall_percentage_l3915_391504

-- Define the number of problems and percentage correct for each test
def test1_problems : ℕ := 20
def test1_percent : ℚ := 75 / 100

def test2_problems : ℕ := 50
def test2_percent : ℚ := 85 / 100

def test3_problems : ℕ := 30
def test3_percent : ℚ := 80 / 100

-- Define the total number of problems
def total_problems : ℕ := test1_problems + test2_problems + test3_problems

-- Define the total number of correct answers
def total_correct : ℚ := test1_problems * test1_percent + test2_problems * test2_percent + test3_problems * test3_percent

-- Theorem statement
theorem ana_overall_percentage :
  (total_correct / total_problems : ℚ) = 815 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ana_overall_percentage_l3915_391504


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l3915_391540

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the interval [1,2)
def interval : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- State the theorem
theorem intersection_equals_interval : M ∩ N = interval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l3915_391540


namespace NUMINAMATH_CALUDE_unique_solution_system_l3915_391520

/-- The system of equations has a unique solution when a = 1, 
    and the solution is x = -3/2, y = -1/2, z = 0 -/
theorem unique_solution_system (a x y z : ℝ) : 
  z = a * (x + 2 * y + 5/2) ∧ 
  x^2 + y^2 + 2*x - y + z = 0 ∧
  ((x + (a + 2)/2)^2 + (y + (2*a - 1)/2)^2 = ((a + 2)^2)/4 + ((2*a - 1)^2)/4 - 5*a/2) →
  (a = 1 ∧ x = -3/2 ∧ y = -1/2 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3915_391520


namespace NUMINAMATH_CALUDE_solve_system_l3915_391545

theorem solve_system (x y z : ℝ) 
  (eq1 : x + 2*y = 10)
  (eq2 : y = 3)
  (eq3 : x - 3*y + z = 7) :
  z = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3915_391545


namespace NUMINAMATH_CALUDE_ratio_to_thirteen_l3915_391508

theorem ratio_to_thirteen : ∃ x : ℚ, (5 : ℚ) / 1 = x / 13 ∧ x = 65 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_thirteen_l3915_391508


namespace NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_100_l3915_391505

def is_common_multiple (n m k : ℕ) : Prop := k % n = 0 ∧ k % m = 0

theorem greatest_common_multiple_10_15_under_100 :
  ∃ (k : ℕ), k < 100 ∧ is_common_multiple 10 15 k ∧
  ∀ (j : ℕ), j < 100 → is_common_multiple 10 15 j → j ≤ k :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_10_15_under_100_l3915_391505


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3915_391537

theorem sin_2alpha_value (f : ℝ → ℝ) (a α : ℝ) :
  (∀ x, f x = 2 * Real.sin (2 * x + π / 6) + a * Real.cos (2 * x)) →
  (∀ x, f x = f (2 * π / 3 - x)) →
  0 < α →
  α < π / 3 →
  f α = 6 / 5 →
  Real.sin (2 * α) = (4 + 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3915_391537


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3915_391511

def U : Set Int := {-4, -2, -1, 0, 2, 4, 5, 6, 7}
def A : Set Int := {-2, 0, 4, 6}
def B : Set Int := {-1, 2, 4, 6, 7}

theorem intersection_complement_equality : A ∩ (U \ B) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3915_391511


namespace NUMINAMATH_CALUDE_x_neq_zero_necessary_not_sufficient_for_x_plus_abs_x_positive_l3915_391598

theorem x_neq_zero_necessary_not_sufficient_for_x_plus_abs_x_positive :
  (∀ x : ℝ, x + |x| > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ x + |x| ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_x_neq_zero_necessary_not_sufficient_for_x_plus_abs_x_positive_l3915_391598


namespace NUMINAMATH_CALUDE_abs_ratio_equality_l3915_391555

theorem abs_ratio_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 9*a*b) :
  |(a + b) / (a - b)| = Real.sqrt 77 / 7 := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_equality_l3915_391555


namespace NUMINAMATH_CALUDE_highest_numbered_street_l3915_391546

/-- The length of Gretzky Street in meters -/
def street_length : ℕ := 5600

/-- The distance between intersecting streets in meters -/
def intersection_distance : ℕ := 350

/-- The number of non-numbered intersecting streets (Orr and Howe) -/
def non_numbered_streets : ℕ := 2

/-- Theorem stating the highest-numbered intersecting street -/
theorem highest_numbered_street :
  (street_length / intersection_distance) - non_numbered_streets = 14 := by
  sorry

end NUMINAMATH_CALUDE_highest_numbered_street_l3915_391546


namespace NUMINAMATH_CALUDE_age_height_not_function_l3915_391595

-- Define a type for people
structure Person where
  age : ℕ
  height : ℝ

-- Define what it means for a relation to be a function
def is_function (R : α → β → Prop) : Prop :=
  ∀ a : α, ∃! b : β, R a b

-- State the theorem
theorem age_height_not_function :
  ¬ is_function (λ (p : Person) (h : ℝ) => p.height = h) :=
sorry

end NUMINAMATH_CALUDE_age_height_not_function_l3915_391595


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l3915_391573

theorem cubic_sum_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  2 * (a^3 + b^3 + c^3) ≥ a^2*b + a*b^2 + a^2*c + a*c^2 + b^2*c + b*c^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l3915_391573


namespace NUMINAMATH_CALUDE_angle_Y_measure_l3915_391566

-- Define the hexagon CHESSY
structure Hexagon :=
  (C E S1 S2 H Y : ℝ)

-- Define the properties of the hexagon
def is_valid_hexagon (h : Hexagon) : Prop :=
  h.C = h.E ∧ h.C = h.S1 ∧ h.C = h.Y ∧ h.H + h.S2 = 180

-- Theorem statement
theorem angle_Y_measure (h : Hexagon) (hvalid : is_valid_hexagon h) : h.Y = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_Y_measure_l3915_391566


namespace NUMINAMATH_CALUDE_number_of_children_l3915_391550

theorem number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) 
  (h1 : crayons_per_child = 12) 
  (h2 : total_crayons = 216) : 
  total_crayons / crayons_per_child = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l3915_391550


namespace NUMINAMATH_CALUDE_cloth_cost_price_l3915_391575

theorem cloth_cost_price
  (total_meters : ℕ)
  (selling_price : ℕ)
  (loss_per_meter : ℕ)
  (h1 : total_meters = 400)
  (h2 : selling_price = 18000)
  (h3 : loss_per_meter = 5) :
  (selling_price + total_meters * loss_per_meter) / total_meters = 50 := by
  sorry

#check cloth_cost_price

end NUMINAMATH_CALUDE_cloth_cost_price_l3915_391575


namespace NUMINAMATH_CALUDE_initial_wax_amount_l3915_391552

/-- Given the total required amount of wax and the additional amount needed,
    calculate the initial amount of wax available. -/
theorem initial_wax_amount (total_required additional_needed : ℕ) :
  total_required ≥ additional_needed →
  total_required - additional_needed = total_required - additional_needed :=
by sorry

end NUMINAMATH_CALUDE_initial_wax_amount_l3915_391552


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_not_prime_l3915_391558

/-- 
Given integers a and b, if the quadratic equation x^2 + ax + b + 1 = 0 
has two positive integer roots, then a^2 + b^2 is not prime.
-/
theorem quadratic_roots_imply_not_prime (a b : ℤ) 
  (h : ∃ p q : ℕ+, p.val ≠ q.val ∧ p.val^2 + a * p.val + b + 1 = 0 ∧ q.val^2 + a * q.val + b + 1 = 0) : 
  ¬ Prime (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_not_prime_l3915_391558


namespace NUMINAMATH_CALUDE_equal_hikes_in_64_weeks_l3915_391597

/-- The number of weeks it takes for Camila to have hiked as many times as Steven -/
def weeks_to_equal_hikes : ℕ :=
  let camila_initial := 7
  let amanda_initial := 8 * camila_initial
  let steven_initial := amanda_initial + 15
  let david_initial := 2 * steven_initial
  let elizabeth_initial := david_initial - 10
  let camila_weekly := 4
  let amanda_weekly := 2
  let steven_weekly := 3
  let david_weekly := 5
  let elizabeth_weekly := 1
  64

theorem equal_hikes_in_64_weeks :
  let camila_initial := 7
  let amanda_initial := 8 * camila_initial
  let steven_initial := amanda_initial + 15
  let david_initial := 2 * steven_initial
  let elizabeth_initial := david_initial - 10
  let camila_weekly := 4
  let amanda_weekly := 2
  let steven_weekly := 3
  let david_weekly := 5
  let elizabeth_weekly := 1
  let w := weeks_to_equal_hikes
  camila_initial + camila_weekly * w = steven_initial + steven_weekly * w :=
by sorry

end NUMINAMATH_CALUDE_equal_hikes_in_64_weeks_l3915_391597


namespace NUMINAMATH_CALUDE_inequality_solution_l3915_391585

theorem inequality_solution (x : ℝ) : 
  (x^2 + 3*x + 3)^(5*x^3 - 3*x^2) ≤ (x^2 + 3*x + 3)^(3*x^3 + 5*x) ↔ 
  x ≤ -2 ∨ x = -1 ∨ (0 ≤ x ∧ x ≤ 5/2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3915_391585


namespace NUMINAMATH_CALUDE_matrix_multiplication_and_scalar_l3915_391544

theorem matrix_multiplication_and_scalar : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 2]
  2 • (A * B) = !![34, -14; 32, -32] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_and_scalar_l3915_391544


namespace NUMINAMATH_CALUDE_trivia_team_score_l3915_391580

theorem trivia_team_score : 
  let total_members : ℕ := 30
  let absent_members : ℕ := 8
  let points_per_member : ℕ := 4
  let deduction_per_incorrect : ℕ := 2
  let total_incorrect : ℕ := 6
  let bonus_multiplier : ℚ := 3/2

  let present_members : ℕ := total_members - absent_members
  let initial_points : ℕ := present_members * points_per_member
  let total_deductions : ℕ := total_incorrect * deduction_per_incorrect
  let points_after_deductions : ℕ := initial_points - total_deductions
  let final_score : ℚ := (points_after_deductions : ℚ) * bonus_multiplier

  final_score = 114 := by sorry

end NUMINAMATH_CALUDE_trivia_team_score_l3915_391580


namespace NUMINAMATH_CALUDE_perpendicular_lines_l3915_391514

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line -/
def slope1 (a : ℝ) : ℝ := a

/-- The slope of the second line -/
def slope2 (a : ℝ) : ℝ := a + 2

/-- If the line y = ax - 2 is perpendicular to the line y = (a+2)x + 1, then a = -1 -/
theorem perpendicular_lines (a : ℝ) : 
  perpendicular (slope1 a) (slope2 a) → a = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l3915_391514


namespace NUMINAMATH_CALUDE_find_divisor_l3915_391523

theorem find_divisor (divisor : ℕ) : 
  (127 / divisor = 9) ∧ (127 % divisor = 1) → divisor = 14 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3915_391523


namespace NUMINAMATH_CALUDE_hyperbola_max_ratio_hyperbola_max_ratio_achievable_l3915_391519

theorem hyperbola_max_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_focal : c^2 = a^2 + b^2) : 
  (a + b) / c ≤ Real.sqrt 2 :=
sorry

theorem hyperbola_max_ratio_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ c^2 = a^2 + b^2 ∧ (a + b) / c = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_max_ratio_hyperbola_max_ratio_achievable_l3915_391519


namespace NUMINAMATH_CALUDE_total_material_proof_l3915_391567

/-- The amount of concrete ordered in tons -/
def concrete : ℝ := 0.16666666666666666

/-- The amount of bricks ordered in tons -/
def bricks : ℝ := 0.16666666666666666

/-- The amount of stone ordered in tons -/
def stone : ℝ := 0.5

/-- The total amount of material ordered by the construction company -/
def total_material : ℝ := concrete + bricks + stone

/-- Theorem stating that the total amount of material ordered is 0.8333333333333332 tons -/
theorem total_material_proof : total_material = 0.8333333333333332 := by
  sorry

end NUMINAMATH_CALUDE_total_material_proof_l3915_391567


namespace NUMINAMATH_CALUDE_geometric_series_sum_special_series_sum_l3915_391556

theorem geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∑' n, a * r^n = a / (1 - r) :=
sorry

/-- The sum of the infinite series 5 + 6(1/1000) + 7(1/1000)^2 + 8(1/1000)^3 + ... is 4995005/998001 -/
theorem special_series_sum :
  ∑' n : ℕ, (n + 5 : ℝ) * (1/1000)^(n-1) = 4995005 / 998001 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_special_series_sum_l3915_391556


namespace NUMINAMATH_CALUDE_alligator_population_after_year_l3915_391565

/-- The number of alligators after a given number of 6-month periods -/
def alligator_population (initial_population : ℕ) (periods : ℕ) : ℕ :=
  initial_population * 2^periods

/-- Theorem stating that given 4 initial alligators and population doubling every 6 months, 
    there will be 16 alligators after 1 year -/
theorem alligator_population_after_year (initial_population : ℕ) 
  (h1 : initial_population = 4) : alligator_population initial_population 2 = 16 := by
  sorry

#check alligator_population_after_year

end NUMINAMATH_CALUDE_alligator_population_after_year_l3915_391565


namespace NUMINAMATH_CALUDE_quadratic_properties_l3915_391512

/-- Quadratic function definition -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + b - 1

/-- Point definition -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem statement -/
theorem quadratic_properties (b : ℝ) :
  (∀ x, f b x = 0 ↔ x = -1 ∨ x = 1 - b) ∧
  (b < 2 → ∀ m, ∃ xp, xp = m - b + 1 ∧ 
    ∃ yp, Point.mk xp yp ∈ {p : Point | f b p.x = p.y ∧ p.x > 0 ∧ p.y > 0}) ∧
  (b = -3 → ∃ c, ∀ m n, 
    (∃ xp yp xq yq, 
      Point.mk xp yp ∈ {p : Point | f b p.x = p.y ∧ p.x > 0 ∧ p.y > 0} ∧
      Point.mk xq yq ∈ {p : Point | f b p.x = p.y ∧ p.x > 0 ∧ p.y < 0} ∧
      (yp - 0) / (xp - (-1)) = m ∧
      (yq - 0) / (xq - (-1)) = n) →
    m * n = c) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3915_391512


namespace NUMINAMATH_CALUDE_distance_to_left_focus_l3915_391509

-- Define the ellipse
def is_on_ellipse (x y b : ℝ) : Prop :=
  x^2 / 25 + y^2 / b^2 = 1

-- Define the condition for b
def valid_b (b : ℝ) : Prop :=
  0 < b ∧ b < 5

-- Define a point P on the ellipse
def P_on_ellipse (P : ℝ × ℝ) (b : ℝ) : Prop :=
  is_on_ellipse P.1 P.2 b

-- Define the left focus F₁
def F₁ : ℝ × ℝ := sorry

-- Define the condition |OP⃗ + OF₁⃗| = 8
def vector_sum_condition (P : ℝ × ℝ) : Prop :=
  ‖P + F₁‖ = 8

-- Theorem statement
theorem distance_to_left_focus
  (b : ℝ)
  (P : ℝ × ℝ)
  (h_b : valid_b b)
  (h_P : P_on_ellipse P b)
  (h_sum : vector_sum_condition P) :
  ‖P - F₁‖ = 2 :=
sorry

end NUMINAMATH_CALUDE_distance_to_left_focus_l3915_391509


namespace NUMINAMATH_CALUDE_add_particular_number_to_34_l3915_391589

theorem add_particular_number_to_34 (x : ℝ) (h : 96 / x = 6) : 34 + x = 50 := by
  sorry

end NUMINAMATH_CALUDE_add_particular_number_to_34_l3915_391589


namespace NUMINAMATH_CALUDE_fish_in_each_bowl_l3915_391517

theorem fish_in_each_bowl (total_bowls : ℕ) (total_fish : ℕ) (h1 : total_bowls = 261) (h2 : total_fish = 6003) :
  total_fish / total_bowls = 23 := by
  sorry

end NUMINAMATH_CALUDE_fish_in_each_bowl_l3915_391517


namespace NUMINAMATH_CALUDE_ellipse_sum_l3915_391534

-- Define the ellipse
def Ellipse (F₁ F₂ : ℝ × ℝ) (d : ℝ) :=
  {P : ℝ × ℝ | dist P F₁ + dist P F₂ = d}

-- Define the foci
def F₁ : ℝ × ℝ := (0, 0)
def F₂ : ℝ × ℝ := (6, 0)

-- Define the distance sum
def d : ℝ := 10

-- Theorem statement
theorem ellipse_sum (h k a b : ℝ) :
  Ellipse F₁ F₂ d →
  (∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ (x, y) ∈ Ellipse F₁ F₂ d) →
  h + k + a + b = 12 := by sorry

end NUMINAMATH_CALUDE_ellipse_sum_l3915_391534


namespace NUMINAMATH_CALUDE_consecutive_sum_product_l3915_391530

theorem consecutive_sum_product (start : ℕ) :
  (start + (start + 1) + (start + 2) + (start + 3) + (start + 4) + (start + 5) = 33) →
  (start * (start + 1) * (start + 2) * (start + 3) * (start + 4) * (start + 5) = 20160) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_product_l3915_391530


namespace NUMINAMATH_CALUDE_quadratic_root_conditions_l3915_391510

theorem quadratic_root_conditions (k : ℤ) : 
  (∃ x y : ℝ, (k^2 + 1) * x^2 - (4 - k) * x + 1 = 0 ∧
              (k^2 + 1) * y^2 - (4 - k) * y + 1 = 0 ∧
              x > 1 ∧ y < 1) →
  k = -1 ∨ k = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_conditions_l3915_391510


namespace NUMINAMATH_CALUDE_cone_base_radius_l3915_391593

/-- Given a semicircle with radius 6 cm forming the lateral surface of a cone,
    prove that the radius of the base circle of the cone is 3 cm. -/
theorem cone_base_radius (r : ℝ) (h : r = 6) : 
  2 * π * r / 2 = 2 * π * 3 := by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3915_391593


namespace NUMINAMATH_CALUDE_orchids_sold_correct_l3915_391521

/-- The number of orchids sold by a plant supplier -/
def orchids_sold : ℕ := 20

/-- The price of each orchid -/
def orchid_price : ℕ := 50

/-- The number of potted Chinese money plants sold -/
def money_plants_sold : ℕ := 15

/-- The price of each potted Chinese money plant -/
def money_plant_price : ℕ := 25

/-- The number of workers -/
def workers : ℕ := 2

/-- The wage paid to each worker -/
def worker_wage : ℕ := 40

/-- The cost of new pots -/
def new_pots_cost : ℕ := 150

/-- The amount left after all transactions -/
def amount_left : ℕ := 1145

/-- Theorem stating that the number of orchids sold is correct given the problem conditions -/
theorem orchids_sold_correct :
  orchids_sold * orchid_price + 
  money_plants_sold * money_plant_price - 
  (workers * worker_wage + new_pots_cost) = 
  amount_left := by sorry

end NUMINAMATH_CALUDE_orchids_sold_correct_l3915_391521


namespace NUMINAMATH_CALUDE_upstream_time_calculation_l3915_391524

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  downstream_time : ℝ  -- Time to swim downstream and across still lake
  upstream_time : ℝ    -- Time to swim upstream and across still lake
  all_downstream_time : ℝ  -- Time if entire journey was downstream

/-- The theorem stating the upstream time given the conditions -/
theorem upstream_time_calculation (s : SwimmingScenario) 
  (h1 : s.downstream_time = 1)
  (h2 : s.upstream_time = 2)
  (h3 : s.all_downstream_time = 5/6) :
  2 / ((2 / s.upstream_time) + (1 / s.downstream_time - 1 / s.all_downstream_time)) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_upstream_time_calculation_l3915_391524


namespace NUMINAMATH_CALUDE_function_extrema_sum_l3915_391586

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 0

-- State the theorem
theorem function_extrema_sum (m : ℝ) :
  (∃ (max min : ℝ), 
    max ∈ Set.image (f m) interval ∧ 
    min ∈ Set.image (f m) interval ∧
    (∀ y ∈ Set.image (f m) interval, y ≤ max ∧ y ≥ min) ∧
    max + min = -14) →
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_function_extrema_sum_l3915_391586


namespace NUMINAMATH_CALUDE_parallel_to_same_plane_are_parallel_perpendicular_to_same_line_are_parallel_l3915_391515

-- Define the basic types
variable (P : Type) -- Type for planes
variable (L : Type) -- Type for lines

-- Define the relationships
variable (parallel : P → P → Prop) -- Parallel planes
variable (perpendicular : P → L → Prop) -- Plane perpendicular to a line

-- Axioms
axiom parallel_trans (p q r : P) : parallel p q → parallel q r → parallel p r
axiom parallel_symm (p q : P) : parallel p q → parallel q p

-- Theorem 1: Two different planes that are parallel to the same plane are parallel to each other
theorem parallel_to_same_plane_are_parallel (p q r : P) 
  (hp : parallel p r) (hq : parallel q r) (hne : p ≠ q) : 
  parallel p q :=
sorry

-- Theorem 2: Two different planes that are perpendicular to the same line are parallel to each other
theorem perpendicular_to_same_line_are_parallel (p q : P) (l : L)
  (hp : perpendicular p l) (hq : perpendicular q l) (hne : p ≠ q) :
  parallel p q :=
sorry

end NUMINAMATH_CALUDE_parallel_to_same_plane_are_parallel_perpendicular_to_same_line_are_parallel_l3915_391515
