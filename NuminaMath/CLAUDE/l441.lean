import Mathlib

namespace NUMINAMATH_CALUDE_fuel_after_600km_distance_with_22L_left_l441_44136

-- Define the relationship between distance and remaining fuel
def fuel_remaining (s : ℝ) : ℝ := 50 - 0.08 * s

-- Theorem 1: When distance is 600 km, remaining fuel is 2 L
theorem fuel_after_600km : fuel_remaining 600 = 2 := by sorry

-- Theorem 2: When remaining fuel is 22 L, distance traveled is 350 km
theorem distance_with_22L_left : ∃ s : ℝ, fuel_remaining s = 22 ∧ s = 350 := by sorry

end NUMINAMATH_CALUDE_fuel_after_600km_distance_with_22L_left_l441_44136


namespace NUMINAMATH_CALUDE_paper_fold_distance_l441_44148

theorem paper_fold_distance (area : ℝ) (h_area : area = 18) : ∃ (distance : ℝ), distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_paper_fold_distance_l441_44148


namespace NUMINAMATH_CALUDE_unique_solution_system_l441_44187

theorem unique_solution_system (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  2 * x * Real.sqrt (x + 1) - y * (y + 1) = 1 ∧
  2 * y * Real.sqrt (y + 1) - z * (z + 1) = 1 ∧
  2 * z * Real.sqrt (z + 1) - x * (x + 1) = 1 →
  x = (1 + Real.sqrt 5) / 2 ∧
  y = (1 + Real.sqrt 5) / 2 ∧
  z = (1 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l441_44187


namespace NUMINAMATH_CALUDE_purchase_cost_l441_44142

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℕ := 4

/-- The cost of a milkshake in dollars -/
def milkshake_cost : ℕ := 3

/-- The number of hamburgers purchased -/
def num_hamburgers : ℕ := 7

/-- The number of milkshakes purchased -/
def num_milkshakes : ℕ := 6

/-- The total cost of the purchase -/
def total_cost : ℕ := hamburger_cost * num_hamburgers + milkshake_cost * num_milkshakes

theorem purchase_cost : total_cost = 46 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l441_44142


namespace NUMINAMATH_CALUDE_odometer_puzzle_l441_44152

/-- Represents the odometer reading as a triple of digits -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat

/-- Represents the trip details -/
structure TripDetails where
  initial : OdometerReading
  final : OdometerReading
  duration : Nat  -- in hours
  avgSpeed : Nat  -- in miles per hour

theorem odometer_puzzle (trip : TripDetails) :
  trip.initial.hundreds ≥ 2 ∧
  trip.initial.hundreds + trip.initial.tens + trip.initial.ones = 9 ∧
  trip.avgSpeed = 60 ∧
  trip.initial.hundreds = trip.final.ones ∧
  trip.initial.tens = trip.final.tens ∧
  trip.initial.ones = trip.final.hundreds →
  trip.initial.hundreds^2 + trip.initial.tens^2 + trip.initial.ones^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_odometer_puzzle_l441_44152


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l441_44129

/-- A fourth-degree polynomial with real coefficients -/
def fourth_degree_poly (g : ℝ → ℝ) : Prop :=
  ∃ a b c d e : ℝ, ∀ x, g x = a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem polynomial_value_theorem (g : ℝ → ℝ) 
  (h_poly : fourth_degree_poly g)
  (h_m1 : |g (-1)| = 15)
  (h_0 : |g 0| = 15)
  (h_2 : |g 2| = 15)
  (h_3 : |g 3| = 15)
  (h_4 : |g 4| = 15) :
  |g 1| = 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l441_44129


namespace NUMINAMATH_CALUDE_task_completion_condition_l441_44156

/-- Represents the completion of a task given the number of people working in two phases -/
def task_completion (x : ℝ) : Prop :=
  let total_time : ℝ := 40
  let phase1_time : ℝ := 4
  let phase2_time : ℝ := 8
  let phase1_people : ℝ := x
  let phase2_people : ℝ := x + 2
  (phase1_time * phase1_people) / total_time + (phase2_time * phase2_people) / total_time = 1

/-- Theorem stating the condition for task completion -/
theorem task_completion_condition (x : ℝ) :
  task_completion x ↔ 4 * x / 40 + 8 * (x + 2) / 40 = 1 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_condition_l441_44156


namespace NUMINAMATH_CALUDE_roll_distribution_probability_l441_44185

def total_rolls : ℕ := 9
def rolls_per_type : ℕ := 3
def num_guests : ℕ := 3

def total_arrangements : ℕ := (total_rolls.factorial) / ((rolls_per_type.factorial) ^ 3)

def favorable_outcomes : ℕ := (rolls_per_type.factorial) ^ num_guests

def probability : ℚ := favorable_outcomes / total_arrangements

theorem roll_distribution_probability :
  probability = 9 / 70 := by sorry

end NUMINAMATH_CALUDE_roll_distribution_probability_l441_44185


namespace NUMINAMATH_CALUDE_arc_length_sector_l441_44130

/-- The arc length of a sector with radius π cm and central angle 2π/3 radians is 2π²/3 cm. -/
theorem arc_length_sector (r : Real) (θ : Real) (l : Real) :
  r = π → θ = 2 * π / 3 → l = θ * r → l = 2 * π^2 / 3 := by
  sorry

#check arc_length_sector

end NUMINAMATH_CALUDE_arc_length_sector_l441_44130


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l441_44117

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 9*x^2 + 23*x - 15 < 0 ↔ x ∈ Set.Iio 1 ∪ Set.Ioo 3 5 :=
sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l441_44117


namespace NUMINAMATH_CALUDE_abc_over_def_value_l441_44151

theorem abc_over_def_value (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 10)
  (h_nonzero : b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0) : 
  a * b * c / (d * e * f) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_abc_over_def_value_l441_44151


namespace NUMINAMATH_CALUDE_rectangle_length_l441_44106

theorem rectangle_length (s : ℝ) (l : ℝ) : 
  s > 0 → l > 0 →
  s^2 = 5 * (l * 10) →
  4 * s = 200 →
  l = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l441_44106


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l441_44181

/-- Given a hyperbola and a parabola with specific properties, prove the value of p -/
theorem hyperbola_parabola_intersection (p : ℝ) : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / 12 = 1) →  -- Hyperbola equation
  (∀ x y : ℝ, x = 2 * p * y^2) →         -- Parabola equation
  (∃ e : ℝ, e = (4 : ℝ) / 2 ∧            -- Eccentricity of hyperbola
    (∀ y : ℝ, e = 2 * p * y^2)) →        -- Focus of parabola at (e, 0)
  p = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l441_44181


namespace NUMINAMATH_CALUDE_minimally_intersecting_triples_count_l441_44153

def Universe : Finset Nat := Finset.range 8

structure MinimallyIntersectingTriple (A B C : Finset Nat) : Prop where
  subset_universe : A ⊆ Universe ∧ B ⊆ Universe ∧ C ⊆ Universe
  intersection_size : (A ∩ B).card = 1 ∧ (B ∩ C).card = 1 ∧ (C ∩ A).card = 1
  empty_triple_intersection : (A ∩ B ∩ C).card = 0

def M : Nat := (Finset.powerset Universe).card

theorem minimally_intersecting_triples_count : M % 1000 = 344 := by
  sorry

end NUMINAMATH_CALUDE_minimally_intersecting_triples_count_l441_44153


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l441_44128

/-- Given a sinusoidal function with specific properties, prove its expression and range -/
theorem sinusoidal_function_properties (f : ℝ → ℝ) (A ω φ : ℝ)
  (h_def : ∀ x, f x = A * Real.sin (ω * x + φ))
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π)
  (h_symmetry : (π / 2) = π / ω)  -- Distance between adjacent axes of symmetry
  (h_lowest : f (2 * π / 3) = -1/2)  -- One of the lowest points
  : (∀ x, f x = (1/2) * Real.sin (2 * x + π/6)) ∧
    (∀ x, x ∈ Set.Icc (-π/6) (π/3) → f x ∈ Set.Icc (-1/4) (1/2)) := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l441_44128


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l441_44164

theorem arithmetic_geometric_sequence (x : ℝ) : 
  (∃ y : ℝ, 
    -- y is between 3 and x
    3 < y ∧ y < x ∧
    -- arithmetic sequence condition
    (y - 3 = x - y) ∧
    -- geometric sequence condition after subtracting 6 from the middle term
    ((y - 6) / 3 = x / (y - 6))) →
  (x = 3 ∨ x = 27) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l441_44164


namespace NUMINAMATH_CALUDE_square_even_implies_even_l441_44163

theorem square_even_implies_even (a : ℤ) (h : Even (a^2)) : Even a := by
  sorry

end NUMINAMATH_CALUDE_square_even_implies_even_l441_44163


namespace NUMINAMATH_CALUDE_range_of_m_l441_44111

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x - 6

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-10) (-6)) ∧
  (∀ y ∈ Set.Icc (-10) (-6), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l441_44111


namespace NUMINAMATH_CALUDE_angle_C_measure_l441_44123

-- Define the triangle ABC
variable (A B C : ℝ)

-- Define the conditions
axiom condition1 : 3 * Real.sin A + 4 * Real.cos B = 6
axiom condition2 : 3 * Real.cos A + 4 * Real.sin B = 1

-- Define that A, B, C form a triangle (their sum is π)
axiom triangle : A + B + C = Real.pi

-- The theorem to prove
theorem angle_C_measure : C = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_angle_C_measure_l441_44123


namespace NUMINAMATH_CALUDE_n_squared_divisible_by_144_l441_44132

theorem n_squared_divisible_by_144 (n : ℕ+) (h : ∀ d : ℕ+, d ∣ n → d ≤ 12) :
  144 ∣ n^2 := by
  sorry

end NUMINAMATH_CALUDE_n_squared_divisible_by_144_l441_44132


namespace NUMINAMATH_CALUDE_calculation_proofs_l441_44120

theorem calculation_proofs :
  (2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / Real.sqrt 2 = (3 * Real.sqrt 2) / 2) ∧
  ((Real.sqrt 3 - Real.sqrt 2)^2 + (Real.sqrt 8 - Real.sqrt 3) * (2 * Real.sqrt 2 + Real.sqrt 3) = 10 - 2 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l441_44120


namespace NUMINAMATH_CALUDE_walking_speed_problem_l441_44184

/-- Proves that given the conditions of the problem, A's walking speed is 10 kmph -/
theorem walking_speed_problem (v : ℝ) : 
  v > 0 → -- A's walking speed is positive
  v * (200 / v) = 20 * (200 / v - 10) → -- Distance equation
  v = 10 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l441_44184


namespace NUMINAMATH_CALUDE_taxi_service_distance_l441_44122

/-- A taxi service problem -/
theorem taxi_service_distance (initial_fee : ℝ) (charge_per_two_fifths : ℝ) (total_charge : ℝ) 
  (h1 : initial_fee = 2.35)
  (h2 : charge_per_two_fifths = 0.35)
  (h3 : total_charge = 5.50) :
  ∃ (distance : ℝ), distance = 3.6 ∧ 
    total_charge = initial_fee + (charge_per_two_fifths / (2/5)) * distance :=
by sorry

end NUMINAMATH_CALUDE_taxi_service_distance_l441_44122


namespace NUMINAMATH_CALUDE_same_solution_implies_k_equals_one_l441_44168

theorem same_solution_implies_k_equals_one :
  ∀ k : ℝ,
  (∀ x : ℝ, 4*x + 3*k = 2*x + 2 ↔ 2*x + k = 5*x + 2.5) →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_k_equals_one_l441_44168


namespace NUMINAMATH_CALUDE_intersection_point_sum_l441_44154

theorem intersection_point_sum (a b : ℝ) : 
  (∃ x y : ℝ, x = (1/3) * y + a ∧ y = (1/3) * x + b ∧ x = 3 ∧ y = 3) → 
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l441_44154


namespace NUMINAMATH_CALUDE_candy_sampling_probability_l441_44141

theorem candy_sampling_probability :
  let p_choose_A : ℝ := 0.40
  let p_choose_B : ℝ := 0.35
  let p_choose_C : ℝ := 0.25
  let p_sample_A : ℝ := 0.16 + 0.07
  let p_sample_B : ℝ := 0.24 + 0.15
  let p_sample_C : ℝ := 0.31 + 0.22
  let p_sample : ℝ := p_choose_A * p_sample_A + p_choose_B * p_sample_B + p_choose_C * p_sample_C
  p_sample = 0.361 :=
by sorry

end NUMINAMATH_CALUDE_candy_sampling_probability_l441_44141


namespace NUMINAMATH_CALUDE_difference_max_min_both_languages_l441_44179

/-- The number of students studying both Spanish and French -/
def students_both (S F : ℕ) : ℤ := S + F - 2001

theorem difference_max_min_both_languages :
  ∃ (S_min S_max F_min F_max : ℕ),
    1601 ≤ S_min ∧ S_max ≤ 1700 ∧
    601 ≤ F_min ∧ F_max ≤ 800 ∧
    (∀ S F, 1601 ≤ S ∧ S ≤ 1700 ∧ 601 ≤ F ∧ F ≤ 800 →
      students_both S_min F_min ≤ students_both S F ∧
      students_both S F ≤ students_both S_max F_max) ∧
    students_both S_max F_max - students_both S_min F_min = 298 := by
  sorry

end NUMINAMATH_CALUDE_difference_max_min_both_languages_l441_44179


namespace NUMINAMATH_CALUDE_exists_number_with_special_quotient_l441_44145

-- Define a function to check if a number contains all digits from 1 to 8 exactly once
def containsAllDigitsOnce (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ Finset.range 8 → (∃! i : ℕ, i < (Nat.digits 10 n).length ∧ (Nat.digits 10 n).get ⟨i, by sorry⟩ = d + 1)

-- Theorem statement
theorem exists_number_with_special_quotient :
  ∃ N d : ℕ, N > 0 ∧ d > 0 ∧ containsAllDigitsOnce (N / d) :=
sorry

end NUMINAMATH_CALUDE_exists_number_with_special_quotient_l441_44145


namespace NUMINAMATH_CALUDE_simplify_expression_l441_44155

theorem simplify_expression : 4 * (15 / 7) * (21 / (-45)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l441_44155


namespace NUMINAMATH_CALUDE_task_assignment_ways_l441_44135

def number_of_students : ℕ := 30
def number_of_tasks : ℕ := 3

def permutations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

theorem task_assignment_ways :
  permutations number_of_students number_of_tasks = 24360 := by
  sorry

end NUMINAMATH_CALUDE_task_assignment_ways_l441_44135


namespace NUMINAMATH_CALUDE_root_implies_m_minus_n_l441_44174

theorem root_implies_m_minus_n (m n : ℝ) : 
  ((-3)^2 + m*(-3) + 3*n = 0) → (m - n = 3) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_minus_n_l441_44174


namespace NUMINAMATH_CALUDE_root_implies_a_value_l441_44172

theorem root_implies_a_value (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x = 0 ∧ x = 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l441_44172


namespace NUMINAMATH_CALUDE_same_function_values_constant_bound_analytical_expression_l441_44137

/-- Definition of "same function" for quadratic and linear functions -/
def same_function (a b c m n : ℝ) : Prop :=
  Real.sqrt (a - m) + |b - n| = 0

/-- Theorem 1: Values of r and s for "same function" -/
theorem same_function_values (r s : ℝ) :
  same_function 3 r 1 s 1 → r = 3 ∧ s = 3 :=
sorry

/-- Theorem 2: Bound on constant term -/
theorem constant_bound (a b c m n : ℝ) :
  same_function a b c m n →
  (∀ x, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  |c| ≤ 1 :=
sorry

/-- Theorem 3: Analytical expression of y₁ -/
theorem analytical_expression (a b c m n : ℝ) :
  same_function a b c m n →
  (∀ x, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x, -1 ≤ x ∧ x ≤ 1 → m * x + n ≤ 2) →
  (a * x^2 + b * x + c = 2 * x^2 - 1 ∨ a * x^2 + b * x + c = -2 * x^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_same_function_values_constant_bound_analytical_expression_l441_44137


namespace NUMINAMATH_CALUDE_unique_solution_l441_44139

-- Define the equation
def equation (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 4 ∧ (3 * x^2 - 12 * x) / (x^2 - 4 * x) = x - 2

-- Theorem statement
theorem unique_solution : ∃! x : ℝ, equation x := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l441_44139


namespace NUMINAMATH_CALUDE_total_bags_delivered_l441_44131

-- Define the problem parameters
def bags_per_trip_light : ℕ := 15
def bags_per_trip_heavy : ℕ := 20
def total_days : ℕ := 7
def trips_per_day_light : ℕ := 25
def trips_per_day_heavy : ℕ := 18
def days_with_light_bags : ℕ := 3
def days_with_heavy_bags : ℕ := 4

-- Define the theorem
theorem total_bags_delivered : 
  (days_with_light_bags * trips_per_day_light * bags_per_trip_light) +
  (days_with_heavy_bags * trips_per_day_heavy * bags_per_trip_heavy) = 2565 :=
by sorry

end NUMINAMATH_CALUDE_total_bags_delivered_l441_44131


namespace NUMINAMATH_CALUDE_tangent_parallel_to_BC_l441_44193

/-- Two circles in a plane -/
structure TwoCircles where
  circle1 : Set (ℝ × ℝ)
  circle2 : Set (ℝ × ℝ)

/-- Points of intersection and other significant points -/
structure CirclePoints (tc : TwoCircles) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  on_circle1_P : P ∈ tc.circle1
  on_circle2_P : P ∈ tc.circle2
  on_circle1_Q : Q ∈ tc.circle1
  on_circle2_Q : Q ∈ tc.circle2
  on_circle1_A : A ∈ tc.circle1
  on_circle2_B : B ∈ tc.circle2
  on_circle2_C : C ∈ tc.circle2

/-- Line represented by two points -/
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Tangent line to a circle at a point -/
def TangentLine (circle : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Two lines are parallel -/
def Parallel (l1 l2 : Set (ℝ × ℝ)) : Prop := sorry

/-- Main theorem -/
theorem tangent_parallel_to_BC (tc : TwoCircles) (cp : CirclePoints tc) : 
  Parallel (TangentLine tc.circle1 cp.A) (Line cp.B cp.C) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_BC_l441_44193


namespace NUMINAMATH_CALUDE_r_plus_s_equals_six_l441_44167

theorem r_plus_s_equals_six (r s : ℕ) (h1 : 2^r = 16) (h2 : 5^s = 25) : r + s = 6 := by
  sorry

end NUMINAMATH_CALUDE_r_plus_s_equals_six_l441_44167


namespace NUMINAMATH_CALUDE_zero_in_M_l441_44133

theorem zero_in_M : 0 ∈ ({-1, 0, 1} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_zero_in_M_l441_44133


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l441_44178

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 15) 
  (h2 : 3 * x + y = 22) : 
  x^2 - y^2 = -120 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l441_44178


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l441_44127

theorem least_number_with_remainder (n : ℕ) : n = 184 ↔
  n > 0 ∧
  n % 5 = 4 ∧
  n % 9 = 4 ∧
  n % 12 = 4 ∧
  n % 18 = 4 ∧
  ∀ m : ℕ, m > 0 →
    m % 5 = 4 →
    m % 9 = 4 →
    m % 12 = 4 →
    m % 18 = 4 →
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l441_44127


namespace NUMINAMATH_CALUDE_average_weight_problem_l441_44194

theorem average_weight_problem (A B C : ℝ) : 
  (A + B) / 2 = 40 →
  (B + C) / 2 = 43 →
  B = 31 →
  (A + B + C) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l441_44194


namespace NUMINAMATH_CALUDE_club_officer_selection_l441_44100

/-- The number of ways to choose officers in a club with gender constraints -/
theorem club_officer_selection (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total = boys + girls)
  (h2 : boys = 14)
  (h3 : girls = 10) :
  (boys * (boys - 1) * girls + girls * (girls - 1) * boys) = 3080 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l441_44100


namespace NUMINAMATH_CALUDE_no_valid_x_l441_44144

theorem no_valid_x : ¬∃ (x : ℕ), x > 1 ∧ x ≠ 5 ∧ x ≠ 6 ∧ x ≠ 12 ∧ 
  184 % 5 = 4 ∧ 184 % 6 = 4 ∧ 184 % x = 4 ∧ 184 % 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_x_l441_44144


namespace NUMINAMATH_CALUDE_incorrect_calculation_ratio_l441_44159

theorem incorrect_calculation_ratio (N : ℝ) (h : N ≠ 0) : 
  (N * 16) / ((N / 16) / 8) = 2048 := by
sorry

end NUMINAMATH_CALUDE_incorrect_calculation_ratio_l441_44159


namespace NUMINAMATH_CALUDE_two_diggers_two_hours_l441_44197

/-- The rate at which diggers dig pits -/
def digging_rate (diggers : ℚ) (pits : ℚ) (hours : ℚ) : ℚ :=
  pits / (diggers * hours)

/-- The number of pits dug given a rate, number of diggers, and hours -/
def pits_dug (rate : ℚ) (diggers : ℚ) (hours : ℚ) : ℚ :=
  rate * diggers * hours

theorem two_diggers_two_hours 
  (h : digging_rate (3/2) (3/2) (3/2) = digging_rate 2 x 2) : x = 8/3 := by
  sorry

#check two_diggers_two_hours

end NUMINAMATH_CALUDE_two_diggers_two_hours_l441_44197


namespace NUMINAMATH_CALUDE_probability_theorem_l441_44109

/-- A regular hexagon --/
structure RegularHexagon where
  /-- The set of all sides and diagonals --/
  S : Finset ℝ
  /-- Number of sides --/
  num_sides : ℕ
  /-- Number of shorter diagonals --/
  num_shorter_diagonals : ℕ
  /-- Number of longer diagonals --/
  num_longer_diagonals : ℕ
  /-- Total number of segments --/
  total_segments : ℕ
  /-- Condition: num_sides = 6 --/
  sides_eq_six : num_sides = 6
  /-- Condition: num_shorter_diagonals = 6 --/
  shorter_diagonals_eq_six : num_shorter_diagonals = 6
  /-- Condition: num_longer_diagonals = 3 --/
  longer_diagonals_eq_three : num_longer_diagonals = 3
  /-- Condition: total_segments = num_sides + num_shorter_diagonals + num_longer_diagonals --/
  total_segments_eq_sum : total_segments = num_sides + num_shorter_diagonals + num_longer_diagonals

/-- The probability of selecting two segments of the same length --/
def probability_same_length (h : RegularHexagon) : ℚ :=
  33 / 105

/-- Theorem: The probability of selecting two segments of the same length is 33/105 --/
theorem probability_theorem (h : RegularHexagon) : 
  probability_same_length h = 33 / 105 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l441_44109


namespace NUMINAMATH_CALUDE_parabola_c_value_l441_44199

/-- A parabola with equation x = ay^2 + by + c, vertex (4, 1), and passing through point (-1, 3) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 4
  vertex_y : ℝ := 1
  point_x : ℝ := -1
  point_y : ℝ := 3
  eq_vertex : 4 = a * 1^2 + b * 1 + c
  eq_point : -1 = a * 3^2 + b * 3 + c

/-- The value of c for the given parabola is 11/4 -/
theorem parabola_c_value (p : Parabola) : p.c = 11/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l441_44199


namespace NUMINAMATH_CALUDE_radius_q3_is_one_point_five_l441_44160

/-- A triangle with an inscribed circle and two additional tangent circles -/
structure TripleCircleTriangle where
  /-- Side length AB of the triangle -/
  ab : ℝ
  /-- Side length BC of the triangle -/
  bc : ℝ
  /-- Side length AC of the triangle -/
  ac : ℝ
  /-- Radius of the inscribed circle Q1 -/
  r1 : ℝ
  /-- Radius of circle Q2, tangent to Q1 and sides AB and BC -/
  r2 : ℝ
  /-- Radius of circle Q3, tangent to Q2 and sides AB and BC -/
  r3 : ℝ
  /-- AB equals BC -/
  ab_eq_bc : ab = bc
  /-- AB equals 80 -/
  ab_eq_80 : ab = 80
  /-- AC equals 96 -/
  ac_eq_96 : ac = 96
  /-- Q1 is inscribed in the triangle -/
  q1_inscribed : r1 = (ab + bc + ac) / 2 - ab
  /-- Q2 is tangent to Q1 and sides AB and BC -/
  q2_tangent : r2 = r1 / 4
  /-- Q3 is tangent to Q2 and sides AB and BC -/
  q3_tangent : r3 = r2 / 4

/-- The radius of Q3 is 1.5 in the given triangle configuration -/
theorem radius_q3_is_one_point_five (t : TripleCircleTriangle) : t.r3 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_radius_q3_is_one_point_five_l441_44160


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l441_44157

/-- The function f(x) = a^(x+1) - 2 has a fixed point at (-1, -1) for a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 1) - 2
  f (-1) = -1 ∧ ∀ x : ℝ, f x = x → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l441_44157


namespace NUMINAMATH_CALUDE_man_work_days_l441_44113

theorem man_work_days (man_son_days : ℝ) (son_days : ℝ) (man_days : ℝ) : 
  man_son_days = 4 → son_days = 20 → man_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_man_work_days_l441_44113


namespace NUMINAMATH_CALUDE_star_properties_l441_44198

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

theorem star_properties :
  (∀ x y : ℝ, star x y = star y x) ∧
  (∃ x y z : ℝ, star x (y + z) ≠ star x y + star x z) ∧
  (∃ x : ℝ, star (x - 2) (x + 2) ≠ star x x - 2) ∧
  (¬ ∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x) ∧
  (∃ x y z : ℝ, star (star x y) z ≠ star x (star y z)) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l441_44198


namespace NUMINAMATH_CALUDE_max_books_borrowed_l441_44147

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 25)
  (h2 : zero_books = 3)
  (h3 : one_book = 11)
  (h4 : two_books = 6)
  (h5 : (total_students : ℚ) * 2 = (zero_books * 0 + one_book * 1 + two_books * 2 + 
    (total_students - zero_books - one_book - two_books) * 3 + 
    (total_students * 2 - zero_books * 0 - one_book * 1 - two_books * 2 - 
    (total_students - zero_books - one_book - two_books) * 3))) :
  ∃ (max_books : ℕ), max_books = 15 ∧ 
    max_books ≤ total_students * 2 - zero_books * 0 - one_book * 1 - two_books * 2 - 
    (total_students - zero_books - one_book - two_books - 1) * 3 :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l441_44147


namespace NUMINAMATH_CALUDE_cubic_factorization_l441_44119

theorem cubic_factorization (a : ℝ) : a^3 - 4*a^2 + 4*a = a*(a-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l441_44119


namespace NUMINAMATH_CALUDE_max_a_for_inequality_l441_44107

theorem max_a_for_inequality : ∃ (a : ℝ), ∀ (x : ℝ), |x - 2| + |x - 8| ≥ a ∧ ∀ (b : ℝ), (∀ (y : ℝ), |y - 2| + |y - 8| ≥ b) → b ≤ a :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_inequality_l441_44107


namespace NUMINAMATH_CALUDE_total_people_in_program_l441_44169

theorem total_people_in_program (parents pupils teachers : ℕ) 
  (h1 : parents = 73)
  (h2 : pupils = 724)
  (h3 : teachers = 744) :
  parents + pupils + teachers = 1541 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l441_44169


namespace NUMINAMATH_CALUDE_event_probability_l441_44188

theorem event_probability (p : ℝ) :
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^4 = 65/81) →
  p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_event_probability_l441_44188


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l441_44182

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 2 → x ≠ 4 →
  (8 * x + 1) / ((x - 4) * (x - 2)^2) =
  (33 / 4) / (x - 4) + (-19 / 4) / (x - 2) + (-17 / 2) / (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l441_44182


namespace NUMINAMATH_CALUDE_smallest_b_value_l441_44102

theorem smallest_b_value (a b : ℤ) (h1 : 29 < a ∧ a < 41) (h2 : b < 51) 
  (h3 : (40 : ℚ) / b - (30 : ℚ) / 50 = (2 : ℚ) / 5) : b ≥ 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_l441_44102


namespace NUMINAMATH_CALUDE_club_size_l441_44115

/-- A club with committees satisfying specific conditions -/
structure Club where
  /-- The number of committees in the club -/
  num_committees : Nat
  /-- The number of members in the club -/
  num_members : Nat
  /-- Each member belongs to exactly two committees -/
  member_in_two_committees : True
  /-- Each pair of committees has exactly one member in common -/
  one_common_member : True

/-- Theorem stating that a club with 4 committees satisfying the given conditions has 6 members -/
theorem club_size (c : Club) : c.num_committees = 4 → c.num_members = 6 := by
  sorry

end NUMINAMATH_CALUDE_club_size_l441_44115


namespace NUMINAMATH_CALUDE_abies_chips_l441_44116

theorem abies_chips (initial_bags : ℕ) (bought_bags : ℕ) (final_bags : ℕ) 
  (h1 : initial_bags = 20)
  (h2 : bought_bags = 6)
  (h3 : final_bags = 22) :
  initial_bags - (initial_bags - final_bags + bought_bags) = 4 :=
by sorry

end NUMINAMATH_CALUDE_abies_chips_l441_44116


namespace NUMINAMATH_CALUDE_regular_polygon_with_140_degree_interior_angles_l441_44173

theorem regular_polygon_with_140_degree_interior_angles (n : ℕ) 
  (h_regular : n ≥ 3) 
  (h_interior_angle : (n - 2) * 180 / n = 140) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_140_degree_interior_angles_l441_44173


namespace NUMINAMATH_CALUDE_twenty_percent_greater_than_80_l441_44125

theorem twenty_percent_greater_than_80 (x : ℝ) : 
  x = 80 * (1 + 0.2) → x = 96 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_greater_than_80_l441_44125


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l441_44108

def set_A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l441_44108


namespace NUMINAMATH_CALUDE_min_seedlings_to_plant_l441_44189

theorem min_seedlings_to_plant (min_survival : ℝ) (max_survival : ℝ) (target : ℕ) : 
  min_survival = 0.75 →
  max_survival = 0.8 →
  target = 1200 →
  ∃ n : ℕ, n ≥ 1500 ∧ ∀ m : ℕ, m < n → (m : ℝ) * max_survival < target := by
  sorry

end NUMINAMATH_CALUDE_min_seedlings_to_plant_l441_44189


namespace NUMINAMATH_CALUDE_lila_seventh_l441_44121

/-- Represents the finishing position of a racer -/
def Position : Type := Fin 12

/-- Represents a racer in the race -/
structure Racer :=
  (name : String)
  (position : Position)

/-- The race with given conditions -/
structure Race :=
  (racers : List Racer)
  (jessica_behind_esther : ∃ (j e : Racer), j.name = "Jessica" ∧ e.name = "Esther" ∧ j.position.val = e.position.val + 7)
  (ivan_behind_noel : ∃ (i n : Racer), i.name = "Ivan" ∧ n.name = "Noel" ∧ i.position.val = n.position.val + 2)
  (lila_behind_esther : ∃ (l e : Racer), l.name = "Lila" ∧ e.name = "Esther" ∧ l.position.val = e.position.val + 4)
  (noel_behind_omar : ∃ (n o : Racer), n.name = "Noel" ∧ o.name = "Omar" ∧ n.position.val = o.position.val + 4)
  (omar_behind_esther : ∃ (o e : Racer), o.name = "Omar" ∧ e.name = "Esther" ∧ o.position.val = e.position.val + 3)
  (ivan_fourth : ∃ (i : Racer), i.name = "Ivan" ∧ i.position.val = 4)

/-- Theorem stating that Lila finished in 7th place -/
theorem lila_seventh (race : Race) : ∃ (l : Racer), l.name = "Lila" ∧ l.position.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_lila_seventh_l441_44121


namespace NUMINAMATH_CALUDE_linda_rings_sold_l441_44196

/-- Proves that Linda sold 8 rings given the conditions of the problem -/
theorem linda_rings_sold :
  let necklaces_sold : ℕ := 4
  let total_sales : ℕ := 80
  let necklace_price : ℕ := 12
  let ring_price : ℕ := 4
  let rings_sold : ℕ := (total_sales - necklaces_sold * necklace_price) / ring_price
  rings_sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_linda_rings_sold_l441_44196


namespace NUMINAMATH_CALUDE_equation_one_l441_44177

theorem equation_one (x : ℚ) : 3 * (x + 8) - 5 = 6 * (2 * x - 1) → x = 25 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_l441_44177


namespace NUMINAMATH_CALUDE_cube_surface_area_l441_44134

/-- The surface area of a cube with edge length 3 cm is 54 square centimeters. -/
theorem cube_surface_area : 
  let edge_length : ℝ := 3
  let face_area : ℝ := edge_length ^ 2
  let surface_area : ℝ := 6 * face_area
  surface_area = 54 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l441_44134


namespace NUMINAMATH_CALUDE_lcm_812_3214_l441_44170

theorem lcm_812_3214 : Nat.lcm 812 3214 = 1303402 := by
  sorry

end NUMINAMATH_CALUDE_lcm_812_3214_l441_44170


namespace NUMINAMATH_CALUDE_square_cylinder_volume_l441_44195

/-- A cylinder with a square cross-section and lateral area 4π has volume 2π -/
theorem square_cylinder_volume (h : ℝ) (lateral_area : ℝ) (volume : ℝ) 
  (h_positive : h > 0)
  (lateral_area_eq : lateral_area = 4 * Real.pi)
  (lateral_area_def : lateral_area = h * h * Real.pi)
  (volume_def : volume = h * h * h / 4) : 
  volume = 2 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_square_cylinder_volume_l441_44195


namespace NUMINAMATH_CALUDE_king_arthur_table_seats_l441_44110

/-- Represents a circular seating arrangement -/
structure CircularArrangement where
  size : ℕ
  opposite : ℕ → ℕ
  opposite_symmetric : ∀ n, n ≤ size → opposite (opposite n) = n

/-- The specific circular arrangement described in the problem -/
def kingArthurTable : CircularArrangement where
  size := 38
  opposite := fun n => (n + 19) % 38
  opposite_symmetric := sorry

theorem king_arthur_table_seats :
  ∃ (t : CircularArrangement), t.size = 38 ∧ t.opposite 10 = 29 := by
  use kingArthurTable
  constructor
  · rfl
  · rfl

#check king_arthur_table_seats

end NUMINAMATH_CALUDE_king_arthur_table_seats_l441_44110


namespace NUMINAMATH_CALUDE_billboard_perimeter_l441_44138

/-- A rectangular billboard with given area and width has a specific perimeter -/
theorem billboard_perimeter (area : ℝ) (width : ℝ) (h1 : area = 117) (h2 : width = 9) :
  2 * (area / width) + 2 * width = 44 := by
  sorry

#check billboard_perimeter

end NUMINAMATH_CALUDE_billboard_perimeter_l441_44138


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l441_44140

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo (-3 : ℝ) 2 = {x : ℝ | a * x^2 - 5*x + b > 0}) :
  {x : ℝ | b * x^2 - 5*x + a > 0} = Set.Iic (-1/3 : ℝ) ∪ Set.Ici (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l441_44140


namespace NUMINAMATH_CALUDE_power_zero_plus_two_l441_44104

theorem power_zero_plus_two : (-2010)^0 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_plus_two_l441_44104


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l441_44162

/-- Proposition p: For all x ∈ [1, 2], x^2 - a ≥ 0 -/
def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

/-- Proposition q: The equation x^2 + 2ax + 2 - a = 0 has real roots -/
def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

/-- The proposition "¬p ∨ ¬q" is false -/
def not_p_or_not_q_is_false (a : ℝ) : Prop :=
  ¬(¬(prop_p a) ∨ ¬(prop_q a))

/-- The range of the real number a is a ≤ -2 or a = 1 -/
def range_of_a (a : ℝ) : Prop :=
  a ≤ -2 ∨ a = 1

theorem range_of_a_theorem (a : ℝ) :
  prop_p a ∧ prop_q a ∧ not_p_or_not_q_is_false a → range_of_a a :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l441_44162


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l441_44126

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

theorem min_value_reciprocal_sum_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ 
    (1 / a + 1 / b) < 4 + 2 * Real.sqrt 3 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l441_44126


namespace NUMINAMATH_CALUDE_parabola_m_value_l441_44149

/-- Theorem: For a parabola with equation x² = my, where m is a positive real number,
    if the distance from the vertex to the directrix is 1/2, then m = 2. -/
theorem parabola_m_value (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2 = m*y) →  -- Parabola equation
  (1/2 : ℝ) = (1/4 : ℝ) * m →  -- Distance from vertex to directrix is 1/2
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_m_value_l441_44149


namespace NUMINAMATH_CALUDE_min_value_of_b_over_a_l441_44114

theorem min_value_of_b_over_a (a b : ℝ) : 
  (∀ x > -1, Real.log (x + 1) - 1 ≤ a * x + b) → 
  (∃ c, c = 1 - Real.exp 1 ∧ ∀ a b, (∀ x > -1, Real.log (x + 1) - 1 ≤ a * x + b) → b / a ≥ c) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_b_over_a_l441_44114


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l441_44176

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 → q > 0 → 
  10 * p^9 * q = 45 * p^8 * q^2 →
  p + 2*q = 1 →
  p = 9/13 := by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l441_44176


namespace NUMINAMATH_CALUDE_total_writing_time_l441_44191

theorem total_writing_time :
  let woody_time : ℝ := 18 -- Woody's writing time in months
  let ivanka_time : ℝ := woody_time + 3 -- Ivanka's writing time
  let alice_time : ℝ := woody_time / 2 -- Alice's writing time
  let tom_time : ℝ := alice_time * 2 -- Tom's writing time
  ivanka_time + woody_time + alice_time + tom_time = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_writing_time_l441_44191


namespace NUMINAMATH_CALUDE_football_lineup_count_l441_44165

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_members : ℕ) (offensive_linemen : ℕ) : ℕ :=
  offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose a starting lineup for the given football team -/
theorem football_lineup_count :
  choose_lineup 15 5 = 109200 := by
  sorry

end NUMINAMATH_CALUDE_football_lineup_count_l441_44165


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l441_44143

/-- A quadratic function with specific properties -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x, p a b c x = a * x^2 + b * x + c) →   -- p is quadratic
  (p a b c 9 = 4) →                         -- p(9) = 4
  (∀ x, p a b c (18 - x) = p a b c x) →     -- axis of symmetry at x = 9
  (∃ n : ℤ, p a b c 0 = n) →                -- p(0) is an integer
  p a b c 18 = 1 :=                         -- prove p(18) = 1
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l441_44143


namespace NUMINAMATH_CALUDE_equation_solution_l441_44103

theorem equation_solution (a : ℕ) : 
  (∃ x y : ℕ, (x + y)^2 + 3*x + y = 2*a) ↔ a = 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l441_44103


namespace NUMINAMATH_CALUDE_supermarket_flour_import_l441_44161

theorem supermarket_flour_import (long_grain : ℚ) (glutinous : ℚ) (flour : ℚ) : 
  long_grain = 9/20 →
  glutinous = 7/20 →
  flour = long_grain + glutinous - 3/20 →
  flour = 13/20 := by
sorry

end NUMINAMATH_CALUDE_supermarket_flour_import_l441_44161


namespace NUMINAMATH_CALUDE_probability_three_men_l441_44112

/-- The probability of selecting 3 men out of 3 selections from a workshop with 7 men and 3 women -/
theorem probability_three_men (total : ℕ) (men : ℕ) (women : ℕ) (selections : ℕ) :
  total = men + women →
  total = 10 →
  men = 7 →
  women = 3 →
  selections = 3 →
  (men.choose selections : ℚ) / (total.choose selections) = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_men_l441_44112


namespace NUMINAMATH_CALUDE_first_number_value_l441_44124

theorem first_number_value (x : ℝ) : x * 6000 = 480 * (10 ^ 5) → x = 8000 := by
  sorry

end NUMINAMATH_CALUDE_first_number_value_l441_44124


namespace NUMINAMATH_CALUDE_malik_yards_per_game_l441_44190

-- Define the number of games
def num_games : ℕ := 4

-- Define Josiah's yards per game
def josiah_yards_per_game : ℕ := 22

-- Define Darnell's average yards per game
def darnell_avg_yards : ℕ := 11

-- Define the total yards run by all three athletes
def total_yards : ℕ := 204

-- Theorem to prove
theorem malik_yards_per_game :
  ∃ (malik_yards : ℕ),
    malik_yards * num_games + 
    josiah_yards_per_game * num_games + 
    darnell_avg_yards * num_games = 
    total_yards ∧ 
    malik_yards = 18 := by
  sorry

end NUMINAMATH_CALUDE_malik_yards_per_game_l441_44190


namespace NUMINAMATH_CALUDE_square_roots_of_2011_sum_l441_44186

theorem square_roots_of_2011_sum (x y : ℝ) : 
  x^2 = 2011 → y^2 = 2011 → x + y = 0 := by
sorry

end NUMINAMATH_CALUDE_square_roots_of_2011_sum_l441_44186


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l441_44175

/-- Given two hyperbolas with equations x²/9 - y²/16 = 1 and y²/25 - x²/N = 1,
    if they have the same asymptotes, then N = 225/16 -/
theorem hyperbolas_same_asymptotes (N : ℝ) :
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / N = 1) →
  N = 225 / 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l441_44175


namespace NUMINAMATH_CALUDE_function_properties_l441_44146

def f (x : ℝ) : ℝ := x^3 + x^2 + x + 1

theorem function_properties : 
  f 0 = 1 ∧ 
  f (-1) = 0 ∧ 
  ∃ ε > 0, |f 1 - 4| < ε := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l441_44146


namespace NUMINAMATH_CALUDE_complex_number_operation_l441_44180

theorem complex_number_operation : 
  let z₁ : ℂ := -2 + 5*I
  let z₂ : ℂ := 3*I
  3 * z₁ + z₂ = -6 + 18*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_operation_l441_44180


namespace NUMINAMATH_CALUDE_square_of_number_ending_in_five_l441_44166

theorem square_of_number_ending_in_five (n : ℕ) :
  ∃ k : ℕ, n = 10 * k + 5 →
    (n^2 % 100 = 25) ∧
    (n^2 = 100 * (k * (k + 1)) + 25) := by
  sorry

end NUMINAMATH_CALUDE_square_of_number_ending_in_five_l441_44166


namespace NUMINAMATH_CALUDE_june_bike_ride_l441_44118

/-- Given that June rides her bike at a constant rate and travels 2 miles in 6 minutes,
    prove that she will travel 5 miles in 15 minutes. -/
theorem june_bike_ride (rate : ℚ) : 
  (2 : ℚ) / (6 : ℚ) = rate → (5 : ℚ) / rate = (15 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_june_bike_ride_l441_44118


namespace NUMINAMATH_CALUDE_beast_of_war_runtime_l441_44183

/-- The running time of Millennium in hours -/
def millennium_runtime : ℝ := 2

/-- The difference in minutes between Millennium and Alpha Epsilon runtimes -/
def alpha_epsilon_diff : ℝ := 30

/-- The difference in minutes between Beast of War and Alpha Epsilon runtimes -/
def beast_of_war_diff : ℝ := 10

/-- Conversion factor from hours to minutes -/
def hours_to_minutes : ℝ := 60

/-- Theorem stating the runtime of Beast of War: Armoured Command -/
theorem beast_of_war_runtime : 
  millennium_runtime * hours_to_minutes - alpha_epsilon_diff + beast_of_war_diff = 100 := by
sorry

end NUMINAMATH_CALUDE_beast_of_war_runtime_l441_44183


namespace NUMINAMATH_CALUDE_quadratic_equations_solution_l441_44150

theorem quadratic_equations_solution :
  -- Part I
  let eq1 : ℝ → Prop := λ x ↦ x^2 + 6*x + 5 = 0
  ∃ x1 x2 : ℝ, eq1 x1 ∧ eq1 x2 ∧ x1 = -5 ∧ x2 = -1 ∧
  -- Part II
  ∀ k : ℝ,
    let eq2 : ℝ → Prop := λ x ↦ x^2 - 3*x + k = 0
    (∃ x1 x2 : ℝ, eq2 x1 ∧ eq2 x2 ∧ (x1 - 1) * (x2 - 1) = -6) →
    k = -4 ∧ ∃ x1 x2 : ℝ, eq2 x1 ∧ eq2 x2 ∧ x1 = 4 ∧ x2 = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solution_l441_44150


namespace NUMINAMATH_CALUDE_probability_for_given_scenario_l441_44101

/-- The probability that at least 4 people stay for the entire basketball game -/
def probability_at_least_4_stay (total_people : ℕ) (certain_stay : ℕ) (uncertain_stay : ℕ) 
  (prob_uncertain_stay : ℚ) : ℚ :=
  sorry

/-- Theorem stating the probability for the specific scenario -/
theorem probability_for_given_scenario : 
  probability_at_least_4_stay 8 3 5 (1/3) = 401/243 := by
  sorry

end NUMINAMATH_CALUDE_probability_for_given_scenario_l441_44101


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l441_44105

/-- A quadratic equation of the form kx^2 - 2x - 1 = 0 has two distinct real roots -/
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2*x - 1 = 0 ∧ k * y^2 - 2*y - 1 = 0

/-- The range of k for which the quadratic equation has two distinct real roots -/
theorem quadratic_roots_range :
  ∀ k : ℝ, has_two_distinct_real_roots k ↔ k > -1 ∧ k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l441_44105


namespace NUMINAMATH_CALUDE_range_of_2x_plus_y_min_value_of_c_l441_44192

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 2*y

-- Theorem 1: Range of 2x + y
theorem range_of_2x_plus_y :
  ∀ x y : ℝ, Circle x y → 1 - Real.sqrt 2 ≤ 2*x + y ∧ 2*x + y ≤ 1 + Real.sqrt 2 :=
sorry

-- Theorem 2: Minimum value of c
theorem min_value_of_c :
  (∃ c : ℝ, ∀ x y : ℝ, Circle x y → x + y + c > 0) ∧
  (∀ c' : ℝ, (∀ x y : ℝ, Circle x y → x + y + c' > 0) → c' ≥ -1) :=
sorry

end NUMINAMATH_CALUDE_range_of_2x_plus_y_min_value_of_c_l441_44192


namespace NUMINAMATH_CALUDE_tick_to_burr_ratio_l441_44158

/-- Given a dog with burrs and ticks in its fur, prove the ratio of ticks to burrs. -/
theorem tick_to_burr_ratio (num_burrs num_total : ℕ) (h1 : num_burrs = 12) (h2 : num_total = 84) :
  (num_total - num_burrs) / num_burrs = 6 := by
  sorry

end NUMINAMATH_CALUDE_tick_to_burr_ratio_l441_44158


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l441_44171

theorem quadratic_equation_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l441_44171
