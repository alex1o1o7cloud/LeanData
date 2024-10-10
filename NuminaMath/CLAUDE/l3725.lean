import Mathlib

namespace slower_train_speed_theorem_l3725_372557

/-- The speed of the faster train in km/h -/
def faster_train_speed : ℝ := 120

/-- The length of the first train in meters -/
def train_length_1 : ℝ := 500

/-- The length of the second train in meters -/
def train_length_2 : ℝ := 700

/-- The time taken for the trains to cross each other in seconds -/
def crossing_time : ℝ := 19.6347928529354

/-- The speed of the slower train in km/h -/
def slower_train_speed : ℝ := 100

theorem slower_train_speed_theorem :
  let total_length := train_length_1 + train_length_2
  let relative_speed := (slower_train_speed + faster_train_speed) * (1000 / 3600)
  total_length = relative_speed * crossing_time :=
by sorry

end slower_train_speed_theorem_l3725_372557


namespace exponent_division_l3725_372559

theorem exponent_division (x : ℝ) : x^8 / x^2 = x^6 := by
  sorry

end exponent_division_l3725_372559


namespace unique_function_property_l3725_372542

-- Define the property for the function
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y = f (x - y)

-- Define that the function is not identically zero
def not_zero_function (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x ≠ 0

-- Theorem statement
theorem unique_function_property :
  ∀ f : ℝ → ℝ, satisfies_property f → not_zero_function f →
  ∀ x : ℝ, f x = 1 :=
sorry

end unique_function_property_l3725_372542


namespace perpendicular_bisector_is_diameter_l3725_372526

/-- A circle in a plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A chord of a circle. -/
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- A line in a plane. -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Predicate to check if a line is perpendicular to a chord. -/
def isPerpendicular (l : Line) (ch : Chord c) : Prop := sorry

/-- Predicate to check if a line bisects a chord. -/
def bisectsChord (l : Line) (ch : Chord c) : Prop := sorry

/-- Predicate to check if a line bisects the arcs subtended by a chord. -/
def bisectsArcs (l : Line) (ch : Chord c) : Prop := sorry

/-- Predicate to check if a line is a diameter of a circle. -/
def isDiameter (l : Line) (c : Circle) : Prop := sorry

/-- Theorem: A line perpendicular to a chord that bisects the chord and the arcs
    subtended by the chord is a diameter of the circle. -/
theorem perpendicular_bisector_is_diameter
  (c : Circle) (ch : Chord c) (l : Line)
  (h1 : isPerpendicular l ch)
  (h2 : bisectsChord l ch)
  (h3 : bisectsArcs l ch) :
  isDiameter l c := by sorry

end perpendicular_bisector_is_diameter_l3725_372526


namespace arithmetic_expression_evaluation_l3725_372512

theorem arithmetic_expression_evaluation : 6 + 15 / 3 - 4^2 + 1 = -4 := by
  sorry

end arithmetic_expression_evaluation_l3725_372512


namespace h_function_iff_strictly_increasing_l3725_372593

def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ StrictMono f :=
sorry

end h_function_iff_strictly_increasing_l3725_372593


namespace janet_pill_count_l3725_372560

/-- Calculates the total number of pills Janet takes in a month -/
def total_pills (multivitamins_per_day : ℕ) (calcium_first_half : ℕ) (calcium_second_half : ℕ) : ℕ :=
  let days_in_month := 28
  let days_in_half := 14
  (multivitamins_per_day * days_in_month) + (calcium_first_half * days_in_half) + (calcium_second_half * days_in_half)

/-- Proves that Janet takes 112 pills in a month -/
theorem janet_pill_count : total_pills 2 3 1 = 112 := by
  sorry

end janet_pill_count_l3725_372560


namespace total_gas_spent_l3725_372522

/-- Calculates the total amount spent on gas by Jim in North Carolina and Virginia -/
theorem total_gas_spent (nc_gallons : ℝ) (nc_price : ℝ) (va_gallons : ℝ) (price_difference : ℝ) :
  nc_gallons = 10 ∧ 
  nc_price = 2 ∧ 
  va_gallons = 10 ∧ 
  price_difference = 1 →
  nc_gallons * nc_price + va_gallons * (nc_price + price_difference) = 50 := by
  sorry

#check total_gas_spent

end total_gas_spent_l3725_372522


namespace f_is_even_m_upper_bound_a_comparisons_l3725_372529

noncomputable def f (x : ℝ) := Real.exp x + Real.exp (-x)

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

theorem m_upper_bound (m : ℝ) : 
  (∀ x : ℝ, x > 0 → m * f x ≤ Real.exp (-x) + m - 1) → m ≤ -1/3 := by sorry

theorem a_comparisons (a : ℝ) (h : a > (Real.exp 1 + Real.exp (-1)) / 2) :
  (a < Real.exp 1 → Real.exp (a - 1) < a^(Real.exp 1 - 1)) ∧
  (a = Real.exp 1 → Real.exp (a - 1) = a^(Real.exp 1 - 1)) ∧
  (a > Real.exp 1 → Real.exp (a - 1) > a^(Real.exp 1 - 1)) := by sorry

end f_is_even_m_upper_bound_a_comparisons_l3725_372529


namespace part1_part2_l3725_372519

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + 3*x - 10 ≤ 0}

-- Define set B for part (1)
def B1 (m : ℝ) : Set ℝ := {x : ℝ | -2*m + 1 ≤ x ∧ x ≤ -m - 1}

-- Define set B for part (2)
def B2 (m : ℝ) : Set ℝ := {x : ℝ | -2*m + 1 ≤ x ∧ x ≤ -m - 1}

-- Theorem for part (1)
theorem part1 : ∀ m : ℝ, (A ∪ B1 m = A) → (2 < m ∧ m ≤ 3) := by sorry

-- Theorem for part (2)
theorem part2 : ∀ m : ℝ, (A ∪ B2 m = A) → m ≤ 3 := by sorry

end part1_part2_l3725_372519


namespace function_through_points_l3725_372587

/-- Given a function f(x) = a^x - k that passes through (1, 3) and (0, 2), 
    prove that f(x) = 2^x + 1 -/
theorem function_through_points 
  (f : ℝ → ℝ) 
  (a k : ℝ) 
  (h1 : ∀ x, f x = a^x - k) 
  (h2 : f 1 = 3) 
  (h3 : f 0 = 2) : 
  ∀ x, f x = 2^x + 1 := by
sorry

end function_through_points_l3725_372587


namespace coinciding_rest_days_count_l3725_372563

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 6

/-- Number of rest days in Al's cycle -/
def al_rest_days : ℕ := 2

/-- Barb's schedule cycle length -/
def barb_cycle : ℕ := 6

/-- Number of rest days in Barb's cycle -/
def barb_rest_days : ℕ := 1

/-- Total number of days -/
def total_days : ℕ := 1000

/-- The number of days both Al and Barb have rest-days on the same day -/
def coinciding_rest_days : ℕ := total_days / (al_cycle * barb_cycle / Nat.gcd al_cycle barb_cycle)

theorem coinciding_rest_days_count :
  coinciding_rest_days = 166 :=
by sorry

end coinciding_rest_days_count_l3725_372563


namespace negative_negative_one_plus_abs_negative_one_equals_two_l3725_372586

theorem negative_negative_one_plus_abs_negative_one_equals_two : 
  -(-1) + |-1| = 2 := by sorry

end negative_negative_one_plus_abs_negative_one_equals_two_l3725_372586


namespace catering_weight_calculation_l3725_372572

/-- Calculates the total weight of catering items for an event --/
theorem catering_weight_calculation (
  silverware_weight : ℕ)
  (silverware_per_setting : ℕ)
  (plate_weight : ℕ)
  (plates_per_setting : ℕ)
  (glass_weight : ℕ)
  (glasses_per_setting : ℕ)
  (decoration_weight : ℕ)
  (num_tables : ℕ)
  (settings_per_table : ℕ)
  (backup_settings : ℕ)
  (decoration_per_table : ℕ)
  (h1 : silverware_weight = 4)
  (h2 : silverware_per_setting = 3)
  (h3 : plate_weight = 12)
  (h4 : plates_per_setting = 2)
  (h5 : glass_weight = 8)
  (h6 : glasses_per_setting = 2)
  (h7 : decoration_weight = 16)
  (h8 : num_tables = 15)
  (h9 : settings_per_table = 8)
  (h10 : backup_settings = 20)
  (h11 : decoration_per_table = 1) :
  (num_tables * settings_per_table + backup_settings) *
    (silverware_weight * silverware_per_setting +
     plate_weight * plates_per_setting +
     glass_weight * glasses_per_setting) +
  num_tables * decoration_weight * decoration_per_table = 7520 := by
  sorry

end catering_weight_calculation_l3725_372572


namespace unique_perpendicular_plane_l3725_372588

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Perpendicularity of a plane to a line -/
def isPerpendicular (p : Plane3D) (l : Line3D) : Prop :=
  -- Definition of perpendicularity
  sorry

/-- A plane contains a point -/
def planeContainsPoint (p : Plane3D) (pt : Point3D) : Prop :=
  -- Definition of a plane containing a point
  sorry

theorem unique_perpendicular_plane 
  (M : Point3D) (h : Line3D) : 
  ∃! (p : Plane3D), planeContainsPoint p M ∧ isPerpendicular p h :=
sorry

end unique_perpendicular_plane_l3725_372588


namespace investment_interest_calculation_l3725_372546

/-- Calculate the total interest earned on an investment with compound interest -/
def totalInterestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- Theorem: The total interest earned on $2,000 at 8% annual interest rate after 5 years is approximately $938.66 -/
theorem investment_interest_calculation :
  let principal : ℝ := 2000
  let rate : ℝ := 0.08
  let years : ℕ := 5
  abs (totalInterestEarned principal rate years - 938.66) < 0.01 := by
  sorry

end investment_interest_calculation_l3725_372546


namespace largest_number_problem_l3725_372550

theorem largest_number_problem (a b c : ℝ) :
  a < b ∧ b < c →
  a + b + c = 82 →
  c - b = 8 →
  b - a = 4 →
  c = 34 := by
sorry

end largest_number_problem_l3725_372550


namespace sum_and_reciprocal_squared_l3725_372528

theorem sum_and_reciprocal_squared (x : ℝ) (h : x + 1/x = 6) : x^2 + 1/x^2 = 34 := by
  sorry

end sum_and_reciprocal_squared_l3725_372528


namespace log_2_3_proof_l3725_372561

theorem log_2_3_proof (a b : ℝ) (h1 : a = Real.log 6 / Real.log 2) (h2 : b = Real.log 20 / Real.log 2) :
  Real.log 3 / Real.log 2 = (a - b + 1) / (b - 1) := by
  sorry

end log_2_3_proof_l3725_372561


namespace constant_function_from_zero_derivative_l3725_372574

theorem constant_function_from_zero_derivative (f : ℝ → ℝ) (h : ∀ x, HasDerivAt f 0 x) :
  ∃ c, ∀ x, f x = c := by sorry

end constant_function_from_zero_derivative_l3725_372574


namespace circle_ray_angle_l3725_372553

/-- In a circle with twelve evenly spaced rays, where one ray points north,
    the smaller angle between the north-pointing ray and the southeast-pointing ray is 90°. -/
theorem circle_ray_angle (n : ℕ) (θ : ℝ) : 
  n = 12 → θ = 360 / n → θ * 3 = 90 :=
by
  sorry

end circle_ray_angle_l3725_372553


namespace problem_solution_l3725_372591

theorem problem_solution : 
  (Real.sqrt 75 + Real.sqrt 27 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 8 * Real.sqrt 3 + Real.sqrt 6) ∧
  ((Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) - (Real.sqrt 5 - 1)^2 = 2 * Real.sqrt 5 - 5) := by
  sorry

end problem_solution_l3725_372591


namespace quadratic_always_nonnegative_range_l3725_372544

theorem quadratic_always_nonnegative_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*m*x + m + 2 ≥ 0) ↔ m ∈ Set.Icc (-1) 2 :=
sorry

end quadratic_always_nonnegative_range_l3725_372544


namespace pyramid_volume_l3725_372564

/-- The volume of a pyramid with a rectangular base, lateral edges of length l,
    and angles α and β between the lateral edges and adjacent sides of the base. -/
theorem pyramid_volume (l α β : ℝ) (hl : l > 0) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ∃ V : ℝ, V = (4 / 3) * l^3 * Real.cos α * Real.cos β * Real.sqrt (-Real.cos (α + β) * Real.cos (α - β)) :=
by sorry

end pyramid_volume_l3725_372564


namespace restaurant_hamburgers_l3725_372525

/-- Represents the number of hamburgers in various states --/
structure HamburgerCount where
  served : ℕ
  leftOver : ℕ

/-- Calculates the total number of hamburgers initially made --/
def totalHamburgers (h : HamburgerCount) : ℕ :=
  h.served + h.leftOver

/-- The theorem stating that for the given values, the total hamburgers is 9 --/
theorem restaurant_hamburgers :
  let h : HamburgerCount := { served := 3, leftOver := 6 }
  totalHamburgers h = 9 := by
  sorry

end restaurant_hamburgers_l3725_372525


namespace fish_tank_problem_l3725_372567

theorem fish_tank_problem (initial_fish : ℕ) : 
  (initial_fish - 4 = 8) → (initial_fish + 8 = 20) := by
  sorry

end fish_tank_problem_l3725_372567


namespace system_solution_l3725_372502

theorem system_solution (x y z : ℝ) : 
  (x^2 + x - 1 = y ∧ y^2 + y - 1 = z ∧ z^2 + z - 1 = x) → 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1)) := by
  sorry

end system_solution_l3725_372502


namespace completing_square_equivalence_l3725_372549

theorem completing_square_equivalence :
  ∀ x : ℝ, 2 * x^2 + 4 * x - 3 = 0 ↔ (x + 1)^2 = 5/2 := by
  sorry

end completing_square_equivalence_l3725_372549


namespace empty_solution_set_inequality_l3725_372556

theorem empty_solution_set_inequality (a : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x + 3| ≥ a) → a ≤ 5 := by
  sorry

end empty_solution_set_inequality_l3725_372556


namespace function_decreasing_on_interval_l3725_372580

/-- Given function f(x) = e^x - ax - 1, prove that there exists a real number a ≥ e^3
    such that f(x) is monotonically decreasing on the interval (-2, 3). -/
theorem function_decreasing_on_interval (a : ℝ) :
  ∃ (a : ℝ), a ≥ Real.exp 3 ∧
  ∀ (x y : ℝ), -2 < x ∧ x < y ∧ y < 3 →
    (Real.exp x - a * x - 1) > (Real.exp y - a * y - 1) := by
  sorry

end function_decreasing_on_interval_l3725_372580


namespace max_sides_is_12_l3725_372576

/-- A convex polygon that can be divided into right triangles with acute angles of 30 and 60 degrees -/
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool
  divisible_into_right_triangles : Bool
  acute_angles : Set ℝ
  acute_angles_eq : acute_angles = {30, 60}

/-- The maximum number of sides for the described convex polygon -/
def max_sides : ℕ := 12

/-- Theorem stating that the maximum number of sides for the described convex polygon is 12 -/
theorem max_sides_is_12 (p : ConvexPolygon) : p.sides ≤ max_sides := by
  sorry

end max_sides_is_12_l3725_372576


namespace division_exponent_rule_l3725_372571

theorem division_exponent_rule (x : ℝ) : -6 * x^5 / (2 * x^3) = -3 * x^2 := by
  sorry

end division_exponent_rule_l3725_372571


namespace geometric_sequence_reciprocal_sum_l3725_372517

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_reciprocal_sum
  (a : ℕ → ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_geometric : is_geometric_sequence a)
  (h_sum : a 0 + a 1 + a 2 + a 3 = 9)
  (h_product : a 0 * a 1 * a 2 * a 3 = 81 / 4) :
  (1 / a 0) + (1 / a 1) + (1 / a 2) + (1 / a 3) = 2 :=
sorry

end geometric_sequence_reciprocal_sum_l3725_372517


namespace perpendicular_parallel_planes_l3725_372596

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_parallel_planes
  (m n : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : parallel_lines m n)
  (h3 : parallel_planes α β) :
  perpendicular n β :=
sorry

end perpendicular_parallel_planes_l3725_372596


namespace circle_change_l3725_372578

/-- Represents the properties of a circle before and after diameter increase -/
structure CircleChange where
  d : ℝ  -- Initial diameter
  Q : ℝ  -- Increase in circumference

/-- Theorem stating the increase in circumference and area when diameter increases by 2π -/
theorem circle_change (c : CircleChange) :
  c.Q = 2 * Real.pi ^ 2 ∧
  (π * ((c.d + 2 * π) / 2) ^ 2 - π * (c.d / 2) ^ 2) = π ^ 2 * c.d + π ^ 3 :=
by sorry

end circle_change_l3725_372578


namespace always_pair_with_difference_multiple_of_seven_l3725_372523

theorem always_pair_with_difference_multiple_of_seven :
  ∀ (S : Finset ℕ),
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 3000) →
  S.card = 8 →
  (∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b).mod 7 = 0) :=
by sorry

end always_pair_with_difference_multiple_of_seven_l3725_372523


namespace arithmetic_sequence_special_difference_l3725_372545

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  first_term : a 1 = 1
  arithmetic : ∀ n, a (n + 1) = a n + d
  geometric_mean : a 2 ^ 2 = a 1 * a 6

/-- The common difference of the arithmetic sequence is either 0 or 3 -/
theorem arithmetic_sequence_special_difference (seq : ArithmeticSequence) : 
  seq.d = 0 ∨ seq.d = 3 := by
  sorry

end arithmetic_sequence_special_difference_l3725_372545


namespace geometric_progression_constant_l3725_372509

theorem geometric_progression_constant (x : ℝ) : 
  (70 + x)^2 = (30 + x) * (150 + x) → x = 10 := by
  sorry

end geometric_progression_constant_l3725_372509


namespace sum_8th_10th_is_230_l3725_372539

/-- An arithmetic sequence with given 4th and 6th terms -/
structure ArithmeticSequence where
  term4 : ℤ
  term6 : ℤ
  is_arithmetic : ∃ (a d : ℤ), term4 = a + 3 * d ∧ term6 = a + 5 * d

/-- The sum of the 8th and 10th terms of the arithmetic sequence -/
def sum_8th_10th_terms (seq : ArithmeticSequence) : ℤ :=
  let a : ℤ := seq.term4 - 3 * ((seq.term6 - seq.term4) / 2)
  let d : ℤ := (seq.term6 - seq.term4) / 2
  (a + 7 * d) + (a + 9 * d)

/-- Theorem stating that the sum of the 8th and 10th terms is 230 -/
theorem sum_8th_10th_is_230 (seq : ArithmeticSequence) 
  (h1 : seq.term4 = 25) (h2 : seq.term6 = 61) : 
  sum_8th_10th_terms seq = 230 := by
  sorry

end sum_8th_10th_is_230_l3725_372539


namespace probability_two_male_finalists_l3725_372568

/-- The probability of selecting two male finalists from a group of 7 finalists (3 male, 4 female) -/
theorem probability_two_male_finalists (total : ℕ) (males : ℕ) (females : ℕ) 
  (h_total : total = 7)
  (h_males : males = 3)
  (h_females : females = 4)
  (h_sum : males + females = total) :
  (males.choose 2 : ℚ) / (total.choose 2) = 1 / 7 := by
  sorry

end probability_two_male_finalists_l3725_372568


namespace divisibility_theorem_l3725_372565

theorem divisibility_theorem (a b c : ℕ) (h1 : ∀ (p : ℕ), Nat.Prime p → c % (p^2) ≠ 0) 
  (h2 : (a^2) ∣ (b^2 * c)) : a ∣ b := by
  sorry

end divisibility_theorem_l3725_372565


namespace equal_area_triangles_l3725_372569

theorem equal_area_triangles (b c : ℝ) (h₁ : b > 0) (h₂ : c > 0) :
  let k : ℝ := (Real.sqrt 5 - 1) * c / 2
  let l : ℝ := (Real.sqrt 5 - 1) * b / 2
  let area_ABK : ℝ := b * k / 2
  let area_AKL : ℝ := l * c / 2
  let area_ADL : ℝ := (b * c - k * l) / 2
  area_ABK = area_AKL ∧ area_AKL = area_ADL := by sorry

end equal_area_triangles_l3725_372569


namespace log_sum_evaluation_l3725_372507

theorem log_sum_evaluation : 
  Real.log 16 / Real.log 2 + 3 * (Real.log 8 / Real.log 2) + 2 * (Real.log 4 / Real.log 2) - Real.log 64 / Real.log 2 = 11 := by
  sorry

end log_sum_evaluation_l3725_372507


namespace order_of_abc_l3725_372589

/-- Given a = 0.1e^(0.1), b = 1/9, and c = -ln(0.9), prove that c < a < b -/
theorem order_of_abc (a b c : ℝ) (ha : a = 0.1 * Real.exp 0.1) (hb : b = 1/9) (hc : c = -Real.log 0.9) :
  c < a ∧ a < b := by
  sorry

end order_of_abc_l3725_372589


namespace unique_real_root_l3725_372524

/-- A quadratic polynomial P(x) = x^2 - 2ax + b -/
def P (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

/-- The condition that P(0), P(1), and P(2) form a geometric progression -/
def geometric_progression (a b : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ P a b 1 = (P a b 0) * r ∧ P a b 2 = (P a b 0) * r^2

/-- The theorem stating that under given conditions, a = 1 is the only value for which P(x) = 0 has real roots -/
theorem unique_real_root (a b : ℝ) :
  geometric_progression a b ∧ P a b 0 * P a b 1 * P a b 2 ≠ 0 →
  (∃ x : ℝ, P a b x = 0) ↔ a = 1 :=
sorry

end unique_real_root_l3725_372524


namespace isosceles_area_sum_l3725_372543

/-- Represents a right isosceles triangle constructed on a side of a right triangle -/
structure RightIsoscelesTriangle where
  side : ℝ
  area : ℝ

/-- Represents a 5-12-13 right triangle with right isosceles triangles on its sides -/
structure TriangleWithIsosceles where
  short_side1 : RightIsoscelesTriangle
  short_side2 : RightIsoscelesTriangle
  hypotenuse : RightIsoscelesTriangle

/-- The theorem to be proved -/
theorem isosceles_area_sum (t : TriangleWithIsosceles) : 
  t.short_side1.side = 5 ∧ 
  t.short_side2.side = 12 ∧ 
  t.hypotenuse.side = 13 ∧
  t.short_side1.area = (1/2) * t.short_side1.side * t.short_side1.side ∧
  t.short_side2.area = (1/2) * t.short_side2.side * t.short_side2.side ∧
  t.hypotenuse.area = (1/2) * t.hypotenuse.side * t.hypotenuse.side →
  t.short_side1.area + t.short_side2.area = t.hypotenuse.area := by
  sorry

end isosceles_area_sum_l3725_372543


namespace smallest_integer_bound_l3725_372590

theorem smallest_integer_bound (a b c d e f : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f →  -- 6 different integers
  (a + b + c + d + e + f) / 6 = 85 →  -- average is 85
  f = 180 →  -- largest is 180
  e = 100 →  -- second largest is 100
  a ≥ -64 :=  -- smallest is not less than -64
by sorry

end smallest_integer_bound_l3725_372590


namespace paislee_calvin_ratio_l3725_372506

def calvin_points : ℕ := 500
def paislee_points : ℕ := 125

theorem paislee_calvin_ratio :
  (paislee_points : ℚ) / calvin_points = 1 / 4 := by
  sorry

end paislee_calvin_ratio_l3725_372506


namespace unique_reverse_multiple_of_nine_l3725_372501

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digits_reversed (n : ℕ) : ℕ :=
  let d₁ := n / 10000
  let d₂ := (n / 1000) % 10
  let d₃ := (n / 100) % 10
  let d₄ := (n / 10) % 10
  let d₅ := n % 10
  d₅ * 10000 + d₄ * 1000 + d₃ * 100 + d₂ * 10 + d₁

theorem unique_reverse_multiple_of_nine :
  ∃! n : ℕ, is_five_digit n ∧ 9 * n = digits_reversed n := by sorry

end unique_reverse_multiple_of_nine_l3725_372501


namespace leonardo_initial_money_l3725_372515

/-- The amount of money Leonardo had initially in his pocket -/
def initial_money : ℚ := 441 / 100

/-- The cost of the chocolate in dollars -/
def chocolate_cost : ℚ := 5

/-- The amount Leonardo borrowed from his friend in dollars -/
def borrowed_amount : ℚ := 59 / 100

/-- The additional amount Leonardo needs in dollars -/
def additional_needed : ℚ := 41 / 100

theorem leonardo_initial_money :
  chocolate_cost = initial_money + borrowed_amount + additional_needed :=
by sorry

end leonardo_initial_money_l3725_372515


namespace folded_strip_fits_l3725_372518

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the rectangular strip
structure RectangularStrip where
  width : ℝ
  length : ℝ

-- Define a folded strip
structure FoldedStrip where
  original : RectangularStrip
  fold_line : ℝ × ℝ → ℝ × ℝ → Prop

-- Define the property of fitting inside a circle
def fits_inside (s : RectangularStrip) (c : Circle) : Prop := sorry

-- Define the property of a folded strip fitting inside a circle
def folded_fits_inside (fs : FoldedStrip) (c : Circle) : Prop := sorry

-- Theorem statement
theorem folded_strip_fits (c : Circle) (s : RectangularStrip) (fs : FoldedStrip) :
  fits_inside s c → fs.original = s → folded_fits_inside fs c := by sorry

end folded_strip_fits_l3725_372518


namespace probability_cover_both_clubs_l3725_372592

/-- The probability of selecting two students that cover both clubs -/
theorem probability_cover_both_clubs 
  (total_students : Nat) 
  (robotics_members : Nat) 
  (science_members : Nat) 
  (h1 : total_students = 30)
  (h2 : robotics_members = 22)
  (h3 : science_members = 24) :
  (Nat.choose total_students 2 - (Nat.choose (robotics_members + science_members - total_students) 2 + 
   Nat.choose (robotics_members - (robotics_members + science_members - total_students)) 2 + 
   Nat.choose (science_members - (robotics_members + science_members - total_students)) 2)) / 
   Nat.choose total_students 2 = 392 / 435 := by
sorry

end probability_cover_both_clubs_l3725_372592


namespace mark_kate_difference_l3725_372581

/-- Project hours charged by four people --/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ
  sam : ℕ

/-- Conditions for the project hours --/
def valid_project_hours (h : ProjectHours) : Prop :=
  h.kate + h.pat + h.mark + h.sam = 198 ∧
  h.pat = 2 * h.kate ∧
  h.pat = h.mark / 3 ∧
  h.sam = (h.pat + h.mark) / 2

theorem mark_kate_difference (h : ProjectHours) (hvalid : valid_project_hours h) :
  h.mark - h.kate = 75 := by
  sorry

end mark_kate_difference_l3725_372581


namespace inequality_solution_set_l3725_372514

theorem inequality_solution_set (a : ℝ) (h : 2*a + 1 < 0) :
  {x : ℝ | x^2 - 4*a*x - 5*a^2 > 0} = {x : ℝ | x < 5*a ∨ x > -a} := by
  sorry

end inequality_solution_set_l3725_372514


namespace mean_of_number_set_l3725_372583

def number_set : List ℝ := [1, 22, 23, 24, 25, 26, 27, 2]

theorem mean_of_number_set :
  (number_set.sum / number_set.length : ℝ) = 18.75 := by sorry

end mean_of_number_set_l3725_372583


namespace fractional_equation_solution_range_l3725_372595

/-- Given that the equation m/(x-2) + 1 = x/(2-x) has a non-negative solution for x,
    prove that the range of values for m is m ≤ 2 and m ≠ -2. -/
theorem fractional_equation_solution_range (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ m / (x - 2) + 1 = x / (2 - x)) →
  m ≤ 2 ∧ m ≠ -2 :=
by sorry

end fractional_equation_solution_range_l3725_372595


namespace average_of_numbers_l3725_372577

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1530, 1200]

theorem average_of_numbers : (numbers.sum / numbers.length : ℚ) = 1380 := by
  sorry

end average_of_numbers_l3725_372577


namespace three_digit_number_l3725_372527

/-- Given a three-digit natural number where the hundreds digit is 5,
    the tens digit is 1, and the units digit is 3, prove that the number is 513. -/
theorem three_digit_number (n : ℕ) : 
  n ≥ 100 ∧ n < 1000 ∧ 
  (n / 100 = 5) ∧ 
  ((n / 10) % 10 = 1) ∧ 
  (n % 10 = 3) → 
  n = 513 := by
sorry

end three_digit_number_l3725_372527


namespace min_value_fraction_l3725_372511

theorem min_value_fraction (x : ℝ) (h : x < 2) :
  (5 - 4 * x + x^2) / (2 - x) ≥ 2 ∧
  ((5 - 4 * x + x^2) / (2 - x) = 2 ↔ x = 1) :=
sorry

end min_value_fraction_l3725_372511


namespace not_perpendicular_to_y_axis_can_be_perpendicular_to_x_axis_l3725_372551

/-- A line in the form x = ky + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Predicate for a line being perpendicular to the y-axis -/
def perpendicular_to_y_axis (l : Line) : Prop :=
  ∃ c : ℝ, ∀ y : ℝ, l.k * y + l.b = c

/-- Predicate for a line being perpendicular to the x-axis -/
def perpendicular_to_x_axis (l : Line) : Prop :=
  l.k = 0

/-- Theorem stating that lines in the form x = ky + b cannot be perpendicular to the y-axis -/
theorem not_perpendicular_to_y_axis :
  ¬ ∃ l : Line, perpendicular_to_y_axis l :=
sorry

/-- Theorem stating that lines in the form x = ky + b can be perpendicular to the x-axis -/
theorem can_be_perpendicular_to_x_axis :
  ∃ l : Line, perpendicular_to_x_axis l :=
sorry

end not_perpendicular_to_y_axis_can_be_perpendicular_to_x_axis_l3725_372551


namespace power_simplification_l3725_372597

theorem power_simplification : 2^6 * 8^3 * 2^12 * 8^6 = 2^45 := by sorry

end power_simplification_l3725_372597


namespace x_range_theorem_l3725_372598

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x + sin x

-- State the theorem
theorem x_range_theorem (h1 : ∀ x ∈ Set.Ioo (-1) 1, deriv f x = 3 + cos x)
                        (h2 : f 0 = 0)
                        (h3 : ∀ x, f (1 - x) + f (1 - x^2) < 0) :
  ∃ x ∈ Set.Ioo 1 (Real.sqrt 2), True :=
sorry

end x_range_theorem_l3725_372598


namespace plate_729_circulation_plate_363_circulation_plate_255_circulation_l3725_372594

def is_valid_plate (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 999

def monday_rule (n : ℕ) : Prop := n % 2 = 1

def tuesday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  (List.sum digits) ≥ 11

def wednesday_rule (n : ℕ) : Prop := n % 3 = 0

def thursday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  (List.sum digits) ≤ 14

def friday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  ∃ (i j : Fin 3), i ≠ j ∧ digits[i] = digits[j]

def saturday_rule (n : ℕ) : Prop := n < 500

def sunday_rule (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  ∀ d ∈ digits, d ≤ 5

theorem plate_729_circulation :
  is_valid_plate 729 ∧
  monday_rule 729 ∧
  tuesday_rule 729 ∧
  wednesday_rule 729 ∧
  ¬thursday_rule 729 ∧
  ¬friday_rule 729 ∧
  ¬saturday_rule 729 ∧
  ¬sunday_rule 729 := by sorry

theorem plate_363_circulation :
  is_valid_plate 363 ∧
  monday_rule 363 ∧
  tuesday_rule 363 ∧
  wednesday_rule 363 ∧
  thursday_rule 363 ∧
  friday_rule 363 ∧
  saturday_rule 363 ∧
  ¬sunday_rule 363 := by sorry

theorem plate_255_circulation :
  is_valid_plate 255 ∧
  monday_rule 255 ∧
  tuesday_rule 255 ∧
  wednesday_rule 255 ∧
  thursday_rule 255 ∧
  friday_rule 255 ∧
  saturday_rule 255 ∧
  sunday_rule 255 := by sorry

end plate_729_circulation_plate_363_circulation_plate_255_circulation_l3725_372594


namespace max_students_planting_trees_l3725_372552

theorem max_students_planting_trees :
  ∃ (a b : ℕ), 3 * a + 5 * b = 115 ∧
  ∀ (x y : ℕ), 3 * x + 5 * y = 115 → x + y ≤ a + b ∧
  a + b = 37 := by
sorry

end max_students_planting_trees_l3725_372552


namespace proposition_analysis_l3725_372582

theorem proposition_analysis :
  let P : ℝ → ℝ → Prop := λ a b => a^2 + b^2 = 0
  let Q : ℝ → ℝ → Prop := λ a b => a^2 - b^2 = 0
  let contrapositive : Prop := ∀ a b : ℝ, ¬(Q a b) → ¬(P a b)
  let inverse : Prop := ∀ a b : ℝ, ¬(P a b) → ¬(Q a b)
  let converse : Prop := ∀ a b : ℝ, Q a b → P a b
  (contrapositive ∧ ¬inverse ∧ ¬converse) ∨
  (¬contrapositive ∧ inverse ∧ ¬converse) ∨
  (¬contrapositive ∧ ¬inverse ∧ converse) :=
by sorry

end proposition_analysis_l3725_372582


namespace hyperbola_vertex_to_asymptote_distance_l3725_372505

/-- The distance from the vertex of the hyperbola x²/4 - y² = 1 to its asymptote -/
theorem hyperbola_vertex_to_asymptote_distance : 
  ∃ (d : ℝ), d = (2 * Real.sqrt 5) / 5 ∧ 
  ∀ (x y : ℝ), x^2/4 - y^2 = 1 → 
  ∃ (v : ℝ × ℝ) (a : ℝ → ℝ), 
    (v.1^2/4 - v.2^2 = 1) ∧  -- v is on the hyperbola
    (∀ (t : ℝ), (a t)^2/4 - t^2 = 1) ∧  -- a is the asymptote function
    d = dist v (a v.1, v.1) := by
  sorry


end hyperbola_vertex_to_asymptote_distance_l3725_372505


namespace bus_stop_time_l3725_372521

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : 
  speed_without_stops = 48 →
  speed_with_stops = 12 →
  (1 - speed_with_stops / speed_without_stops) * 60 = 45 := by
  sorry

#check bus_stop_time

end bus_stop_time_l3725_372521


namespace misread_weight_calculation_l3725_372579

theorem misread_weight_calculation (class_size : ℕ) (incorrect_avg : ℚ) (correct_avg : ℚ) (correct_weight : ℚ) :
  class_size = 20 →
  incorrect_avg = 58.4 →
  correct_avg = 58.7 →
  correct_weight = 62 →
  ∃ misread_weight : ℚ,
    class_size * correct_avg - class_size * incorrect_avg = correct_weight - misread_weight ∧
    misread_weight = 56 :=
by sorry

end misread_weight_calculation_l3725_372579


namespace problem_solution_l3725_372532

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 4) (h3 : z^2 / x = 8) :
  x = 2^(11/7) := by
sorry

end problem_solution_l3725_372532


namespace white_marbles_added_l3725_372513

/-- Proves that the number of white marbles added to a bag is 4, given the initial marble counts and the resulting probability of drawing a black or gold marble. -/
theorem white_marbles_added (black gold purple red : ℕ) 
  (h_black : black = 3)
  (h_gold : gold = 6)
  (h_purple : purple = 2)
  (h_red : red = 6)
  (h_prob : (black + gold : ℚ) / (black + gold + purple + red + w) = 3 / 7)
  : w = 4 := by
  sorry

end white_marbles_added_l3725_372513


namespace arithmetic_geometric_sequence_indices_l3725_372530

theorem arithmetic_geometric_sequence_indices 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (k : ℕ → ℕ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : ∃ r, ∀ n, a (k (n + 1)) = a (k n) * r)
  (h4 : k 1 = 1)
  (h5 : k 2 = 2)
  (h6 : k 3 = 6) :
  k 4 = 22 := by
sorry

end arithmetic_geometric_sequence_indices_l3725_372530


namespace absolute_value_equation_solution_difference_l3725_372520

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (|x₁ + 8| = Real.sqrt 256) ∧
  (|x₂ + 8| = Real.sqrt 256) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 32 :=
by sorry

end absolute_value_equation_solution_difference_l3725_372520


namespace diagonals_30_sided_polygon_l3725_372540

/-- The number of diagonals in a convex polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: The number of diagonals in a convex polygon with 30 sides is 375 -/
theorem diagonals_30_sided_polygon :
  numDiagonals 30 = 375 := by sorry

end diagonals_30_sided_polygon_l3725_372540


namespace inscribed_circle_radii_theorem_l3725_372508

/-- A regular pyramid with base ABCD and apex S -/
structure RegularPyramid where
  /-- The length of the base diagonal AC -/
  base_diagonal : ℝ
  /-- The cosine of the angle SBD -/
  cos_angle_sbd : ℝ
  /-- Assumption that the pyramid is regular -/
  regular : True
  /-- Assumption that the base diagonal AC = 1 -/
  base_diagonal_eq_one : base_diagonal = 1
  /-- Assumption that cos(∠SBD) = 2/3 -/
  cos_angle_sbd_eq_two_thirds : cos_angle_sbd = 2/3

/-- The set of possible radii for circles inscribed in planar sections of the pyramid -/
def inscribed_circle_radii (p : RegularPyramid) : Set ℝ :=
  {r : ℝ | (0 < r ∧ r ≤ 1/6) ∨ r = 1/3}

/-- Theorem stating the possible radii of inscribed circles in the regular pyramid -/
theorem inscribed_circle_radii_theorem (p : RegularPyramid) :
  ∀ r : ℝ, r ∈ inscribed_circle_radii p ↔ (0 < r ∧ r ≤ 1/6) ∨ r = 1/3 := by
  sorry


end inscribed_circle_radii_theorem_l3725_372508


namespace no_real_solution_range_l3725_372537

theorem no_real_solution_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + x - a = 0) ↔ a < -1/4 := by
sorry

end no_real_solution_range_l3725_372537


namespace luke_garage_sale_games_l3725_372555

/-- Represents the number of games Luke bought at the garage sale -/
def games_from_garage_sale : ℕ := sorry

/-- Represents the number of games Luke bought from a friend -/
def games_from_friend : ℕ := 2

/-- Represents the number of games that didn't work -/
def non_working_games : ℕ := 2

/-- Represents the number of good games Luke ended up with -/
def good_games : ℕ := 2

theorem luke_garage_sale_games :
  games_from_garage_sale = 2 :=
by
  sorry

end luke_garage_sale_games_l3725_372555


namespace bicentric_quadrilateral_segment_difference_l3725_372500

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (a b c d : ℝ)

-- Define the properties of the quadrilateral
def is_cyclic_bicentric (q : Quadrilateral) : Prop :=
  -- The quadrilateral is cyclic (inscribed in a circle)
  ∃ (r : ℝ), r > 0 ∧ 
  -- The quadrilateral has an incircle
  ∃ (s : ℝ), s > 0 ∧
  -- Additional conditions for cyclic and bicentric quadrilaterals
  -- (These are simplified representations and may need more detailed conditions)
  q.a + q.c = q.b + q.d

-- Define the theorem
theorem bicentric_quadrilateral_segment_difference 
  (q : Quadrilateral) 
  (h : is_cyclic_bicentric q) 
  (h_sides : q.a = 70 ∧ q.b = 90 ∧ q.c = 130 ∧ q.d = 110) : 
  ∃ (x y : ℝ), x + y = 130 ∧ |x - y| = 13 := by
  sorry

end bicentric_quadrilateral_segment_difference_l3725_372500


namespace cube_of_negative_double_l3725_372536

theorem cube_of_negative_double (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end cube_of_negative_double_l3725_372536


namespace max_value_of_f_l3725_372547

-- Define the function
def f (x : ℝ) : ℝ := -4 * x^2 + 10

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 10 :=
by sorry

end max_value_of_f_l3725_372547


namespace earl_floor_problem_l3725_372562

theorem earl_floor_problem (total_floors : ℕ) (start_floor : ℕ) 
  (up_first : ℕ) (down : ℕ) (up_second : ℕ) :
  total_floors = 20 →
  start_floor = 1 →
  up_first = 5 →
  down = 2 →
  up_second = 7 →
  total_floors - (start_floor + up_first - down + up_second) = 9 :=
by sorry

end earl_floor_problem_l3725_372562


namespace heather_shared_blocks_l3725_372534

/-- The number of blocks Heather shared with Jose -/
def blocks_shared (initial final : ℕ) : ℕ := initial - final

/-- Theorem: The number of blocks Heather shared is the difference between her initial and final blocks -/
theorem heather_shared_blocks (initial final : ℕ) (h1 : initial = 86) (h2 : final = 45) :
  blocks_shared initial final = 41 := by
  sorry

end heather_shared_blocks_l3725_372534


namespace book_pages_l3725_372531

theorem book_pages (pages_per_day : ℕ) (days : ℕ) (fraction_read : ℚ) : 
  pages_per_day = 8 → days = 12 → fraction_read = 2/3 →
  (pages_per_day * days : ℚ) / fraction_read = 144 := by
sorry

end book_pages_l3725_372531


namespace field_ratio_l3725_372558

theorem field_ratio (field_length : ℝ) (pond_side : ℝ) (pond_area_ratio : ℝ) :
  field_length = 96 →
  pond_side = 8 →
  pond_area_ratio = 1 / 72 →
  (pond_side * pond_side) * (1 / pond_area_ratio) = field_length * (field_length / 2) →
  field_length / (field_length / 2) = 2 := by
  sorry

end field_ratio_l3725_372558


namespace function_inequality_l3725_372548

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x ∈ Set.Ioo 0 (π/2), tan x * deriv f x < f x) : 
  f (π/6) * sin 1 > (1/2) * f 1 := by
  sorry

end function_inequality_l3725_372548


namespace banana_storage_l3725_372516

/-- The number of boxes needed to store bananas -/
def number_of_boxes (total_bananas : ℕ) (bananas_per_box : ℕ) : ℕ :=
  total_bananas / bananas_per_box

/-- Proof that 8 boxes are needed to store 40 bananas with 5 bananas per box -/
theorem banana_storage : number_of_boxes 40 5 = 8 := by
  sorry

end banana_storage_l3725_372516


namespace counterexamples_count_l3725_372570

/-- Definition of digit sum -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Definition to check if a number has zero as a digit -/
def hasZeroDigit (n : ℕ) : Prop := sorry

/-- Definition of a prime number -/
def isPrime (n : ℕ) : Prop := sorry

theorem counterexamples_count :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, n % 2 = 1 ∧ digitSum n = 4 ∧ ¬hasZeroDigit n ∧ ¬isPrime n) ∧
    s.card = 2 := by sorry

end counterexamples_count_l3725_372570


namespace binomial_plus_three_l3725_372584

theorem binomial_plus_three : (Nat.choose 13 11) + 3 = 81 := by
  sorry

end binomial_plus_three_l3725_372584


namespace gumball_difference_l3725_372573

/-- The number of gumballs Carl bought -/
def carl_gumballs : ℕ := 16

/-- The number of gumballs Lewis bought -/
def lewis_gumballs : ℕ := 12

/-- The minimum average number of gumballs -/
def min_average : ℚ := 19

/-- The maximum average number of gumballs -/
def max_average : ℚ := 25

/-- The number of people who bought gumballs -/
def num_people : ℕ := 3

theorem gumball_difference :
  ∃ (min_x max_x : ℕ),
    (∀ x : ℕ, 
      (min_average ≤ (carl_gumballs + lewis_gumballs + x : ℚ) / num_people ∧
       (carl_gumballs + lewis_gumballs + x : ℚ) / num_people ≤ max_average) →
      min_x ≤ x ∧ x ≤ max_x) ∧
    max_x - min_x = 18 := by
  sorry

end gumball_difference_l3725_372573


namespace greatest_third_side_proof_l3725_372575

/-- The greatest integer length for the third side of a triangle with sides 5 and 10 -/
def greatest_third_side : ℕ :=
  14

theorem greatest_third_side_proof :
  ∀ (c : ℕ),
  (c > greatest_third_side → ¬(5 < c + 10 ∧ 10 < c + 5 ∧ c < 5 + 10)) ∧
  (c ≤ greatest_third_side → (5 < c + 10 ∧ 10 < c + 5 ∧ c < 5 + 10)) :=
by sorry

end greatest_third_side_proof_l3725_372575


namespace range_of_a_l3725_372599

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a-1)^x < (a-1)^y

-- Define the theorem
theorem range_of_a : 
  (∃ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a))) → 
  (∃ a : ℝ, (-1 < a ∧ a ≤ 2) ∨ a ≥ 3) :=
by sorry

end range_of_a_l3725_372599


namespace ice_cream_preference_l3725_372535

theorem ice_cream_preference (total : ℕ) (vanilla : ℕ) (strawberry : ℕ) (neither : ℕ) 
  (h1 : total = 50)
  (h2 : vanilla = 23)
  (h3 : strawberry = 20)
  (h4 : neither = 14) :
  total - neither - (vanilla + strawberry - (total - neither)) = 7 := by
  sorry

end ice_cream_preference_l3725_372535


namespace car_distance_problem_l3725_372533

theorem car_distance_problem (V : ℝ) (D : ℝ) : 
  V = 50 →
  D / V - D / (V + 25) = 0.5 →
  D = 75 := by
sorry

end car_distance_problem_l3725_372533


namespace initial_stock_theorem_l3725_372541

/-- Represents the stock management of a bicycle shop over 3 months -/
structure BikeShop :=
  (mountain_weekly_add : Fin 3 → ℕ)
  (road_weekly_add : ℕ)
  (hybrid_weekly_add : Fin 3 → ℕ)
  (mountain_monthly_sell : ℕ)
  (road_monthly_sell : Fin 3 → ℕ)
  (hybrid_monthly_sell : ℕ)
  (helmet_initial : ℕ)
  (helmet_weekly_add : ℕ)
  (helmet_weekly_sell : ℕ)
  (lock_initial : ℕ)
  (lock_weekly_add : ℕ)
  (lock_weekly_sell : ℕ)
  (final_mountain : ℕ)
  (final_road : ℕ)
  (final_hybrid : ℕ)
  (final_helmet : ℕ)
  (final_lock : ℕ)

/-- The theorem stating the initial stock of bicycles -/
theorem initial_stock_theorem (shop : BikeShop) 
  (h_mountain : shop.mountain_weekly_add = ![6, 4, 3])
  (h_road : shop.road_weekly_add = 4)
  (h_hybrid : shop.hybrid_weekly_add = ![2, 2, 3])
  (h_mountain_sell : shop.mountain_monthly_sell = 12)
  (h_road_sell : shop.road_monthly_sell = ![16, 16, 24])
  (h_hybrid_sell : shop.hybrid_monthly_sell = 10)
  (h_helmet : shop.helmet_initial = 100 ∧ shop.helmet_weekly_add = 10 ∧ shop.helmet_weekly_sell = 15)
  (h_lock : shop.lock_initial = 50 ∧ shop.lock_weekly_add = 5 ∧ shop.lock_weekly_sell = 3)
  (h_final : shop.final_mountain = 75 ∧ shop.final_road = 80 ∧ shop.final_hybrid = 45 ∧ 
             shop.final_helmet = 115 ∧ shop.final_lock = 62) :
  ∃ (initial_mountain initial_road initial_hybrid : ℕ),
    initial_mountain = 59 ∧ 
    initial_road = 88 ∧ 
    initial_hybrid = 47 :=
by sorry

end initial_stock_theorem_l3725_372541


namespace johnnys_hourly_rate_l3725_372566

def hours_worked : ℝ := 8
def total_earnings : ℝ := 26

theorem johnnys_hourly_rate : total_earnings / hours_worked = 3.25 := by
  sorry

end johnnys_hourly_rate_l3725_372566


namespace abs_negative_2023_l3725_372504

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_negative_2023_l3725_372504


namespace correct_log_values_l3725_372554

/-- The logarithm base 10 function -/
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- Representation of logarithmic values in terms of a, b, and c -/
structure LogValues where
  a : ℝ
  b : ℝ
  c : ℝ

theorem correct_log_values (v : LogValues) :
  (log10 0.021 = 2 * v.a + v.b + v.c - 3) →
  (log10 0.27 = 6 * v.a - 3 * v.b - 2) →
  (log10 2.8 = 1 - 2 * v.a + 2 * v.b - v.c) →
  (log10 3 = 2 * v.a - v.b) →
  (log10 5 = v.a + v.c) →
  (log10 6 = 1 + v.a - v.b - v.c) →
  (log10 8 = 3 - 3 * v.a - 3 * v.c) →
  (log10 9 = 4 * v.a - 2 * v.b) →
  (log10 14 = 1 - v.c + 2 * v.b) →
  (log10 1.5 = 3 * v.a - v.b + v.c - 1) ∧
  (log10 7 = 2 * v.b + v.c) := by
  sorry


end correct_log_values_l3725_372554


namespace candidate_vote_percentage_l3725_372538

def total_votes : ℕ := 8000
def loss_margin : ℕ := 2400

theorem candidate_vote_percentage :
  ∃ (p : ℚ),
    p * total_votes = (total_votes - loss_margin) / 2 ∧
    p = 35 / 100 :=
by sorry

end candidate_vote_percentage_l3725_372538


namespace leftover_coin_value_l3725_372585

def quarters_per_roll : ℕ := 40
def dimes_per_roll : ℕ := 50
def half_dollars_per_roll : ℕ := 20

def james_quarters : ℕ := 120
def james_dimes : ℕ := 200
def james_half_dollars : ℕ := 90

def lindsay_quarters : ℕ := 150
def lindsay_dimes : ℕ := 310
def lindsay_half_dollars : ℕ := 160

def total_quarters : ℕ := james_quarters + lindsay_quarters
def total_dimes : ℕ := james_dimes + lindsay_dimes
def total_half_dollars : ℕ := james_half_dollars + lindsay_half_dollars

def leftover_quarters : ℕ := total_quarters % quarters_per_roll
def leftover_dimes : ℕ := total_dimes % dimes_per_roll
def leftover_half_dollars : ℕ := total_half_dollars % half_dollars_per_roll

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.1
def half_dollar_value : ℚ := 0.5

theorem leftover_coin_value :
  (leftover_quarters : ℚ) * quarter_value +
  (leftover_dimes : ℚ) * dime_value +
  (leftover_half_dollars : ℚ) * half_dollar_value = 13.5 := by
  sorry

end leftover_coin_value_l3725_372585


namespace sum_of_numbers_l3725_372503

theorem sum_of_numbers (a b : ℕ+) : 
  Nat.gcd a b = 3 →
  Nat.lcm a b = 100 →
  (1 : ℚ) / a + (1 : ℚ) / b = 103 / 300 →
  a + b = 36 := by
sorry

end sum_of_numbers_l3725_372503


namespace tangent_equation_solutions_l3725_372510

theorem tangent_equation_solutions (t : Real) : 
  (5.41 * Real.tan t = (Real.sin t ^ 2 + Real.sin (2 * t) - 1) / (Real.cos t ^ 2 - Real.sin (2 * t) + 1)) ↔ 
  (∃ k : ℤ, t = π / 4 + k * π ∨ 
            t = Real.arctan ((1 + Real.sqrt 5) / 2) + k * π ∨ 
            t = Real.arctan ((1 - Real.sqrt 5) / 2) + k * π) :=
by sorry

end tangent_equation_solutions_l3725_372510
