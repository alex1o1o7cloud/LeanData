import Mathlib

namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3297_329796

theorem regular_polygon_sides (D : ℕ) : D = 20 → ∃ n : ℕ, n > 2 ∧ D = n * (n - 3) / 2 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3297_329796


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3297_329790

theorem inequality_equivalence (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ↔ 
  a ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3297_329790


namespace NUMINAMATH_CALUDE_sum_first_ten_primes_with_units_digit_three_l3297_329714

/-- A function that returns true if a number has a units digit of 3 -/
def hasUnitsDigitThree (n : ℕ) : Bool :=
  n % 10 = 3

/-- The sequence of prime numbers with a units digit of 3 -/
def primesWithUnitsDigitThree : List ℕ :=
  (List.range 200).filter (λ n => n.Prime && hasUnitsDigitThree n)

/-- The sum of the first ten prime numbers with a units digit of 3 -/
def sumFirstTenPrimesWithUnitsDigitThree : ℕ :=
  (primesWithUnitsDigitThree.take 10).sum

theorem sum_first_ten_primes_with_units_digit_three :
  sumFirstTenPrimesWithUnitsDigitThree = 639 := by
  sorry


end NUMINAMATH_CALUDE_sum_first_ten_primes_with_units_digit_three_l3297_329714


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3297_329716

/-- If x² + 2x + m > 0 for all real x, then m > 1 -/
theorem quadratic_always_positive (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3297_329716


namespace NUMINAMATH_CALUDE_no_real_roots_implies_no_real_roots_composition_l3297_329785

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: If f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_real_roots_implies_no_real_roots_composition
  (a b c : ℝ) :
  (∀ x : ℝ, f a b c x ≠ x) →
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_implies_no_real_roots_composition_l3297_329785


namespace NUMINAMATH_CALUDE_cyclic_ratio_inequality_l3297_329738

theorem cyclic_ratio_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / b + b / c + c / d + d / a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_cyclic_ratio_inequality_l3297_329738


namespace NUMINAMATH_CALUDE_smallest_X_proof_l3297_329782

/-- A function that checks if a positive integer only contains 0s and 1s as digits -/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer X such that there exists a T satisfying the conditions -/
def smallestX : ℕ := 74

theorem smallest_X_proof :
  ∀ T : ℕ,
  T > 0 →
  onlyZerosAndOnes T →
  (∃ X : ℕ, T = 15 * X) →
  ∃ X : ℕ, T = 15 * X ∧ X ≥ smallestX :=
sorry

end NUMINAMATH_CALUDE_smallest_X_proof_l3297_329782


namespace NUMINAMATH_CALUDE_perspective_triangle_area_l3297_329718

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 1

/-- The perspective plane triangle of an equilateral triangle -/
structure PerspectiveTriangle (et : EquilateralTriangle) where

/-- The area of a triangle -/
def area (t : Type) : ℝ := sorry

/-- The theorem stating the area of the perspective plane triangle -/
theorem perspective_triangle_area (et : EquilateralTriangle) 
  (pt : PerspectiveTriangle et) : 
  area (PerspectiveTriangle et) = Real.sqrt 6 / 16 := by sorry

end NUMINAMATH_CALUDE_perspective_triangle_area_l3297_329718


namespace NUMINAMATH_CALUDE_f_max_value_l3297_329786

-- Define the function f(x) = x³ + 3x² - 4
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 4

-- State the theorem about the maximum value of f
theorem f_max_value :
  ∃ (M : ℝ), M = 0 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l3297_329786


namespace NUMINAMATH_CALUDE_B_2_1_eq_12_l3297_329737

def B : ℕ → ℕ → ℕ
  | 0, n => n + 2
  | m + 1, 0 => B m 2
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_1_eq_12 : B 2 1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_B_2_1_eq_12_l3297_329737


namespace NUMINAMATH_CALUDE_distribution_of_X_l3297_329751

/-- A discrete random variable with three possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  p₁ : ℝ
  p₂ : ℝ
  p₃ : ℝ
  x₁_lt_x₂ : x₁ < x₂
  x₂_lt_x₃ : x₂ < x₃
  prob_sum : p₁ + p₂ + p₃ = 1
  prob_nonneg : 0 ≤ p₁ ∧ 0 ≤ p₂ ∧ 0 ≤ p₃

/-- Expected value of a discrete random variable -/
def expectedValue (X : DiscreteRV) : ℝ :=
  X.x₁ * X.p₁ + X.x₂ * X.p₂ + X.x₃ * X.p₃

/-- Variance of a discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  X.x₁^2 * X.p₁ + X.x₂^2 * X.p₂ + X.x₃^2 * X.p₃ - (expectedValue X)^2

/-- Theorem stating the distribution of the random variable X -/
theorem distribution_of_X (X : DiscreteRV) 
  (h₁ : X.x₁ = 1)
  (h₂ : X.p₁ = 0.3)
  (h₃ : X.p₂ = 0.2)
  (h₄ : expectedValue X = 2.2)
  (h₅ : variance X = 0.76) :
  X.x₂ = 2 ∧ X.x₃ = 3 ∧ X.p₃ = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_distribution_of_X_l3297_329751


namespace NUMINAMATH_CALUDE_problem_solution_l3297_329773

def f (k : ℝ) (x : ℝ) : ℝ := k - |x - 3|

theorem problem_solution (k a b c : ℝ) :
  (∀ x, f k (x + 3) ≥ 0 ↔ x ∈ Set.Icc (-1) 1) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (1 / (k * a) + 1 / (2 * k * b) + 1 / (3 * k * c) = 1) →
  (k = 1 ∧ 1/9 * a + 2/9 * b + 3/9 * c ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3297_329773


namespace NUMINAMATH_CALUDE_shoes_count_l3297_329726

/-- The total number of pairs of shoes Ellie and Riley have together -/
def total_shoes (ellie_shoes : ℕ) (riley_difference : ℕ) : ℕ :=
  ellie_shoes + (ellie_shoes - riley_difference)

/-- Theorem stating that given Ellie has 8 pairs of shoes and Riley has 3 fewer pairs,
    they have 13 pairs of shoes in total -/
theorem shoes_count : total_shoes 8 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_shoes_count_l3297_329726


namespace NUMINAMATH_CALUDE_student_congress_sample_size_l3297_329768

/-- Given a school with classes and students, prove the sample size for a "Student Congress" -/
theorem student_congress_sample_size 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (selected_students : ℕ) 
  (h1 : num_classes = 40)
  (h2 : students_per_class = 50)
  (h3 : selected_students = 150) :
  selected_students = 150 := by
  sorry

end NUMINAMATH_CALUDE_student_congress_sample_size_l3297_329768


namespace NUMINAMATH_CALUDE_distance_before_collision_value_l3297_329757

/-- Two boats moving towards each other -/
structure BoatSystem where
  boat1_speed : ℝ
  boat2_speed : ℝ
  initial_distance : ℝ

/-- Calculate the distance between boats one minute before collision -/
def distance_before_collision (bs : BoatSystem) : ℝ :=
  sorry

/-- Theorem stating the distance between boats one minute before collision -/
theorem distance_before_collision_value (bs : BoatSystem)
  (h1 : bs.boat1_speed = 5)
  (h2 : bs.boat2_speed = 25)
  (h3 : bs.initial_distance = 20) :
  distance_before_collision bs = 0.5 :=
sorry

end NUMINAMATH_CALUDE_distance_before_collision_value_l3297_329757


namespace NUMINAMATH_CALUDE_prob_difference_increases_l3297_329723

/-- The probability of getting exactly 5 heads in 10 coin flips -/
def prob_five_heads : ℚ := 252 / 1024

/-- The probability of the absolute difference increasing given equal heads and tails -/
def prob_increase_equal : ℚ := 1

/-- The probability of the absolute difference increasing given unequal heads and tails -/
def prob_increase_unequal : ℚ := 1 / 2

/-- The probability of the absolute difference between heads and tails increasing after an 11th coin flip, given 10 initial flips -/
theorem prob_difference_increases : 
  prob_five_heads * prob_increase_equal + 
  (1 - prob_five_heads) * prob_increase_unequal = 319 / 512 := by
  sorry

end NUMINAMATH_CALUDE_prob_difference_increases_l3297_329723


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3297_329729

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- State the theorem
theorem line_perpendicular_to_plane 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_parallel : parallel m n)
  (h_perpendicular : perpendicular m β) :
  perpendicular n β := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3297_329729


namespace NUMINAMATH_CALUDE_percent_of_y_l3297_329748

theorem percent_of_y (y : ℝ) (h : y > 0) : ((6 * y) / 20 + (3 * y) / 10) / y = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l3297_329748


namespace NUMINAMATH_CALUDE_length_of_side_b_area_of_triangle_ABC_l3297_329769

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions for a valid triangle if necessary
  true

-- Given conditions
axiom side_a : ℝ
axiom side_a_value : side_a = 3 * Real.sqrt 3

axiom side_c : ℝ
axiom side_c_value : side_c = 2

axiom angle_B : ℝ
axiom angle_B_value : angle_B = 150 * Real.pi / 180

-- Theorem for the length of side b
theorem length_of_side_b (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C)
  (ha : a = side_a) (hc : c = side_c) (hB : B = angle_B) :
  b = 7 := by sorry

-- Theorem for the area of triangle ABC
theorem area_of_triangle_ABC (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C)
  (ha : a = side_a) (hc : c = side_c) (hB : B = angle_B) :
  (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_length_of_side_b_area_of_triangle_ABC_l3297_329769


namespace NUMINAMATH_CALUDE_all_statements_false_l3297_329727

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Define the theorem
theorem all_statements_false :
  ¬(∀ (m n : Line) (α : Plane), 
    parallel_line_plane m α → parallel_line_plane n α → parallel_lines m n) ∧
  ¬(∀ (m n : Line) (α : Plane), 
    perpendicular_line_plane m α → perpendicular_lines m n → parallel_line_plane n α) ∧
  ¬(∀ (m n : Line) (α β : Plane), 
    perpendicular_line_plane m α → perpendicular_line_plane n β → 
    perpendicular_lines m n → perpendicular_planes α β) ∧
  ¬(∀ (m : Line) (α β : Plane), 
    line_in_plane m β → parallel_planes α β → parallel_line_plane m α) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_false_l3297_329727


namespace NUMINAMATH_CALUDE_exists_student_with_eight_sessions_l3297_329713

/-- A structure representing a club with students and sessions. -/
structure Club where
  students : Finset Nat
  sessions : Finset Nat
  attended : Nat → Finset Nat
  meet_once : ∀ s₁ s₂, s₁ ∈ students → s₂ ∈ students → s₁ ≠ s₂ →
    ∃! session, session ∈ sessions ∧ s₁ ∈ attended session ∧ s₂ ∈ attended session
  not_all_in_one : ∀ session, session ∈ sessions → ∃ s, s ∈ students ∧ s ∉ attended session

/-- Theorem stating that in a club satisfying the given conditions,
    there exists a student who attended at least 8 sessions. -/
theorem exists_student_with_eight_sessions (c : Club) (h : c.students.card = 50) :
  ∃ s, s ∈ c.students ∧ (c.sessions.filter (fun session => s ∈ c.attended session)).card ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_exists_student_with_eight_sessions_l3297_329713


namespace NUMINAMATH_CALUDE_refrigerator_transport_cost_prove_transport_cost_l3297_329799

/-- Calculates the transport cost for a refrigerator purchase --/
theorem refrigerator_transport_cost 
  (purchase_price : ℝ) 
  (discount_rate : ℝ) 
  (installation_cost : ℝ) 
  (profit_rate : ℝ) 
  (selling_price : ℝ) : ℝ :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let total_cost := purchase_price + installation_cost
  let transport_cost := (selling_price / (1 + profit_rate)) - total_cost
  transport_cost

/-- Proves that the transport cost is 4000 given the problem conditions --/
theorem prove_transport_cost : 
  refrigerator_transport_cost 15500 0.2 250 0.1 21725 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_transport_cost_prove_transport_cost_l3297_329799


namespace NUMINAMATH_CALUDE_mandy_sister_age_difference_l3297_329774

/-- Represents the ages and relationships in Mandy's family -/
structure Family where
  mandy_age : ℕ
  brother_age : ℕ
  sister_age : ℕ
  brother_age_relation : brother_age = 4 * mandy_age
  sister_age_relation : sister_age = brother_age - 5

/-- Calculates the age difference between Mandy and her sister -/
def age_difference (f : Family) : ℕ :=
  f.sister_age - f.mandy_age

/-- Theorem stating the age difference between Mandy and her sister -/
theorem mandy_sister_age_difference (f : Family) (h : f.mandy_age = 3) :
  age_difference f = 4 := by
  sorry

#check mandy_sister_age_difference

end NUMINAMATH_CALUDE_mandy_sister_age_difference_l3297_329774


namespace NUMINAMATH_CALUDE_natural_number_pair_product_sum_gcd_lcm_l3297_329720

theorem natural_number_pair_product_sum_gcd_lcm : 
  ∀ a b : ℕ, 
    a > 0 ∧ b > 0 → 
    (a * b - (a + b) = Nat.gcd a b + Nat.lcm a b) ↔ 
    ((a = 6 ∧ b = 3) ∨ (a = 6 ∧ b = 4) ∨ (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 6)) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pair_product_sum_gcd_lcm_l3297_329720


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3297_329731

theorem complex_equation_solution (x y : ℝ) : 
  (2 * x - y + 1 : ℂ) + (y - 2 : ℂ) * I = 0 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3297_329731


namespace NUMINAMATH_CALUDE_gcd_bound_from_lcm_l3297_329793

theorem gcd_bound_from_lcm (a b : ℕ) : 
  a ≥ 1000000 ∧ a < 10000000 ∧ 
  b ≥ 1000000 ∧ b < 10000000 ∧ 
  Nat.lcm a b ≥ 100000000000 ∧ Nat.lcm a b < 1000000000000 →
  Nat.gcd a b < 1000 := by
sorry

end NUMINAMATH_CALUDE_gcd_bound_from_lcm_l3297_329793


namespace NUMINAMATH_CALUDE_largest_partition_size_l3297_329794

/-- A partition of the positive integers into k subsets -/
def Partition (k : ℕ) := Fin k → Set ℕ

/-- The property that every integer ≥ 15 can be represented as a sum of two distinct elements from each subset -/
def HasPropertyForAll (P : Partition k) : Prop :=
  ∀ (n : ℕ) (i : Fin k), n ≥ 15 → ∃ (x y : ℕ), x ≠ y ∧ x ∈ P i ∧ y ∈ P i ∧ x + y = n

/-- The main theorem statement -/
theorem largest_partition_size :
  ∃ (k : ℕ), k > 0 ∧ 
    (∃ (P : Partition k), HasPropertyForAll P) ∧ 
    (∀ (m : ℕ), m > k → ¬∃ (Q : Partition m), HasPropertyForAll Q) ∧
    k = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_partition_size_l3297_329794


namespace NUMINAMATH_CALUDE_tourist_tax_calculation_l3297_329722

theorem tourist_tax_calculation (tax_free_amount tax_rate total_tax : ℝ) 
  (h1 : tax_free_amount = 600)
  (h2 : tax_rate = 0.07)
  (h3 : total_tax = 78.4) : 
  ∃ (total_value : ℝ), 
    total_value > tax_free_amount ∧ 
    tax_rate * (total_value - tax_free_amount) = total_tax ∧ 
    total_value = 1720 := by
  sorry

end NUMINAMATH_CALUDE_tourist_tax_calculation_l3297_329722


namespace NUMINAMATH_CALUDE_strongest_signal_l3297_329741

def signal_strength (x : ℤ) : ℝ := |x|

def is_stronger (x y : ℤ) : Prop := signal_strength x < signal_strength y

theorem strongest_signal :
  let signals : List ℤ := [-50, -60, -70, -80]
  ∀ s ∈ signals, s ≠ -50 → is_stronger (-50) s :=
by sorry

end NUMINAMATH_CALUDE_strongest_signal_l3297_329741


namespace NUMINAMATH_CALUDE_parking_lot_problem_l3297_329763

theorem parking_lot_problem (initial cars_left cars_entered final : ℕ) : 
  cars_left = 13 →
  cars_entered = cars_left + 5 →
  final = 85 →
  final = initial - cars_left + cars_entered →
  initial = 80 := by sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l3297_329763


namespace NUMINAMATH_CALUDE_inequality_proof_l3297_329788

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a + b) + (b * c) / (b + c) + (c * a) / (c + a) ≤ 
  (3 * (a * b + b * c + c * a)) / (2 * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3297_329788


namespace NUMINAMATH_CALUDE_points_are_coplanar_l3297_329704

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem points_are_coplanar
  (h_not_collinear : ¬ ∃ (k : ℝ), e₂ = k • e₁)
  (h_AB : B - A = e₁ + e₂)
  (h_AC : C - A = 2 • e₁ + 8 • e₂)
  (h_AD : D - A = 3 • e₁ - 5 • e₂) :
  ∃ (x y : ℝ), D - A = x • (B - A) + y • (C - A) :=
sorry

end NUMINAMATH_CALUDE_points_are_coplanar_l3297_329704


namespace NUMINAMATH_CALUDE_rooftop_steps_l3297_329777

/-- The total number of stair steps to reach the rooftop -/
def total_steps (climbed : ℕ) (remaining : ℕ) : ℕ := climbed + remaining

/-- Theorem stating that the total number of steps is 96 -/
theorem rooftop_steps : total_steps 74 22 = 96 := by
  sorry

end NUMINAMATH_CALUDE_rooftop_steps_l3297_329777


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3297_329797

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_properties (a : ℕ → ℝ) :
  geometric_sequence a →
  increasing_sequence a →
  (∀ n : ℕ, a n = 2^n) →
  (a 5)^2 = a 10 ∧
  ∀ n : ℕ, 2 * (a n + a (n + 2)) = 5 * a (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3297_329797


namespace NUMINAMATH_CALUDE_function_properties_function_range_l3297_329743

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem function_properties (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, f a x ∈ Set.Icc 1 a) ∧
  (Set.range (f a) = Set.Icc 1 a) →
  a = 2 :=
sorry

theorem function_range (a : ℝ) :
  (∀ x ≤ 2, ∀ y ≤ 2, x < y → f a x > f a y) ∧
  (∀ x ∈ Set.Icc 1 (a + 1), ∀ y ∈ Set.Icc 1 (a + 1), |f a x - f a y| ≤ 4) →
  a ∈ Set.Icc 2 3 :=
sorry

end NUMINAMATH_CALUDE_function_properties_function_range_l3297_329743


namespace NUMINAMATH_CALUDE_sum_removal_proof_l3297_329761

theorem sum_removal_proof : 
  let original_sum := (1 : ℚ) / 3 + 1 / 5 + 1 / 7 + 1 / 9 + 1 / 11 + 1 / 13
  let removed_terms := 1 / 11 + 1 / 13
  original_sum - removed_terms = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_removal_proof_l3297_329761


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3297_329795

theorem min_value_theorem (x : ℝ) :
  (x^2 + 12) / Real.sqrt (x^2 + x + 5) ≥ 2 * Real.sqrt 7 :=
sorry

theorem min_value_achievable :
  ∃ x : ℝ, (x^2 + 12) / Real.sqrt (x^2 + x + 5) = 2 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3297_329795


namespace NUMINAMATH_CALUDE_alice_has_winning_strategy_l3297_329764

/-- Represents the state of the game with three piles of coins. -/
structure GameState :=
  (pile1 : Nat) (pile2 : Nat) (pile3 : Nat)

/-- Represents a player in the game. -/
inductive Player
  | Alice | Bob | Charlie

/-- Represents a move in the game. -/
structure Move :=
  (pile : Fin 3) (coins : Fin 3)

/-- Defines if a game state is terminal (no coins left). -/
def isTerminal (state : GameState) : Prop :=
  state.pile1 = 0 ∧ state.pile2 = 0 ∧ state.pile3 = 0

/-- Defines a valid move in the game. -/
def validMove (state : GameState) (move : Move) : Prop :=
  match move.pile with
  | 0 => state.pile1 ≥ move.coins
  | 1 => state.pile2 ≥ move.coins
  | 2 => state.pile3 ≥ move.coins

/-- Applies a move to a game state. -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move.pile with
  | 0 => { state with pile1 := state.pile1 - move.coins }
  | 1 => { state with pile2 := state.pile2 - move.coins }
  | 2 => { state with pile3 := state.pile3 - move.coins }

/-- Defines the next player in turn. -/
def nextPlayer : Player → Player
  | Player.Alice => Player.Bob
  | Player.Bob => Player.Charlie
  | Player.Charlie => Player.Alice

/-- Theorem: Alice has a winning strategy in the game starting with piles of 5, 7, and 8 coins. -/
theorem alice_has_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (game : GameState → Player → Prop),
      (∀ s p, game s p → ¬isTerminal s → ∃ m, validMove s m ∧ game (applyMove s m) (nextPlayer p)) →
      (∀ s, isTerminal s → game s Player.Charlie) →
      game { pile1 := 5, pile2 := 7, pile3 := 8 } Player.Alice :=
by sorry

end NUMINAMATH_CALUDE_alice_has_winning_strategy_l3297_329764


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l3297_329730

/-- Calculates the sampling interval for systematic sampling -/
def samplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The sampling interval for a population of 800 and sample size of 40 is 20 -/
theorem systematic_sampling_interval :
  samplingInterval 800 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l3297_329730


namespace NUMINAMATH_CALUDE_nunzio_pizza_consumption_l3297_329734

/-- Calculates the number of whole pizzas eaten given daily pieces, days, and pieces per pizza -/
def pizzas_eaten (daily_pieces : ℕ) (days : ℕ) (pieces_per_pizza : ℕ) : ℕ :=
  (daily_pieces * days) / pieces_per_pizza

theorem nunzio_pizza_consumption :
  pizzas_eaten 3 72 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nunzio_pizza_consumption_l3297_329734


namespace NUMINAMATH_CALUDE_function_value_theorem_l3297_329758

/-- Given a function f(x) = √(-x² + bx + c) with domain D, 
    and for any x in D, f(-1) ≤ f(x) ≤ f(1), 
    prove that b · c + f(3) = 6 -/
theorem function_value_theorem (b c : ℝ) (D : Set ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x ∈ D, f x = Real.sqrt (-x^2 + b*x + c))
    (h2 : ∀ x ∈ D, f (-1) ≤ f x ∧ f x ≤ f 1) :
    b * c + f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_function_value_theorem_l3297_329758


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3297_329749

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |1 + x + x^2 / 2| < 1} = {x : ℝ | -2 < x ∧ x < 0} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3297_329749


namespace NUMINAMATH_CALUDE_grasshopper_frog_jump_contest_l3297_329719

theorem grasshopper_frog_jump_contest (grasshopper_jump frog_jump : ℕ) 
  (h1 : grasshopper_jump = 31) 
  (h2 : frog_jump = 35) : 
  grasshopper_jump + frog_jump = 66 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_frog_jump_contest_l3297_329719


namespace NUMINAMATH_CALUDE_seating_theorem_l3297_329732

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  groups : Nat
  seats_per_group : Nat
  extra_pair : Nat
  total_seats : Nat
  max_customers : Nat

/-- Checks if a seating arrangement is valid --/
def is_valid_arrangement (arr : SeatingArrangement) : Prop :=
  arr.groups * arr.seats_per_group + arr.extra_pair = arr.total_seats ∧
  arr.max_customers ≤ arr.total_seats

/-- Checks if pairs can always be seated adjacently --/
def can_seat_pairs (arr : SeatingArrangement) : Prop :=
  ∀ n : Nat, n ≤ arr.max_customers → 
    (n / 2) * 2 ≤ arr.groups * 2 + arr.extra_pair

theorem seating_theorem (arr : SeatingArrangement) 
  (h1 : arr.groups = 7)
  (h2 : arr.seats_per_group = 3)
  (h3 : arr.extra_pair = 2)
  (h4 : arr.total_seats = 23)
  (h5 : arr.max_customers = 16)
  : is_valid_arrangement arr ∧ can_seat_pairs arr := by
  sorry

#check seating_theorem

end NUMINAMATH_CALUDE_seating_theorem_l3297_329732


namespace NUMINAMATH_CALUDE_range_of_a_l3297_329760

-- Define the propositions p and q
def p (x : ℝ) : Prop := 2 * x / (x - 1) < 1

def q (x a : ℝ) : Prop := (x + a) * (x - 3) > 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ a ∈ Set.Iic (-1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3297_329760


namespace NUMINAMATH_CALUDE_platform_length_specific_platform_length_l3297_329746

/-- The length of a platform given train parameters -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- Proof of the specific platform length problem -/
theorem specific_platform_length : 
  platform_length 360 45 48 = 840 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_specific_platform_length_l3297_329746


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l3297_329700

theorem cube_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 + 3*a*b*c ≥ a*b*(a+b) + b*c*(b+c) + c*a*(c+a) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l3297_329700


namespace NUMINAMATH_CALUDE_function_inequality_l3297_329776

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
    (h1 : ∀ x, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3297_329776


namespace NUMINAMATH_CALUDE_donut_selections_l3297_329744

theorem donut_selections (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  Nat.choose (n + k - 1) (k - 1) = 84 := by
  sorry

end NUMINAMATH_CALUDE_donut_selections_l3297_329744


namespace NUMINAMATH_CALUDE_second_player_cannot_lose_l3297_329779

/-- Represents a player in the game -/
inductive Player : Type
| First : Player
| Second : Player

/-- Represents a move in the game -/
structure Move where
  player : Player
  moveNumber : Nat

/-- Represents the state of the game -/
structure GameState where
  currentMove : Move
  isGameOver : Bool

/-- The game can only end on an even-numbered move -/
axiom game_ends_on_even_move : 
  ∀ (gs : GameState), gs.isGameOver → gs.currentMove.moveNumber % 2 = 0

/-- The first player makes even-numbered moves -/
axiom first_player_even_moves :
  ∀ (m : Move), m.player = Player.First → m.moveNumber % 2 = 0

/-- Theorem: The second player cannot lose -/
theorem second_player_cannot_lose :
  ∀ (gs : GameState), gs.isGameOver → gs.currentMove.player ≠ Player.Second :=
by sorry


end NUMINAMATH_CALUDE_second_player_cannot_lose_l3297_329779


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3297_329775

theorem hyperbola_equation (a b c : ℝ) : 
  (2 * c = 10) →  -- focal length is 10
  (b / a = 2) →   -- slope of asymptote is 2
  (a^2 + b^2 = c^2) →  -- relation between a, b, and c
  (a^2 = 5 ∧ b^2 = 20) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3297_329775


namespace NUMINAMATH_CALUDE_max_value_theorem_l3297_329778

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - |x - a|

-- State the theorem
theorem max_value_theorem (a b : ℝ) :
  (-1 ≤ a) →
  (a ≤ 1) →
  (∀ x ∈ Set.Icc 1 3, f a x + b * x ≤ 0) →
  (a^2 + 3 * b ≤ 10) ∧ 
  (∃ a₀ b₀, (-1 ≤ a₀) ∧ (a₀ ≤ 1) ∧ 
   (∀ x ∈ Set.Icc 1 3, f a₀ x + b₀ * x ≤ 0) ∧ 
   (a₀^2 + 3 * b₀ = 10)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3297_329778


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3297_329708

theorem decimal_to_fraction : 
  ∃ (n d : ℤ), d ≠ 0 ∧ 3.75 = (n : ℚ) / (d : ℚ) ∧ n = 15 ∧ d = 4 :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3297_329708


namespace NUMINAMATH_CALUDE_area_of_triangle_from_square_centers_l3297_329772

/-- The area of a triangle formed by the centers of three adjacent squares surrounding a central square --/
theorem area_of_triangle_from_square_centers (central_square_side : ℝ) (h : central_square_side = 2) :
  let outer_square_diagonal := central_square_side * Real.sqrt 2
  let triangle_side := outer_square_diagonal
  let triangle_area := Real.sqrt 3 / 4 * triangle_side^2
  triangle_area = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_from_square_centers_l3297_329772


namespace NUMINAMATH_CALUDE_solution_set_equality_l3297_329755

theorem solution_set_equality : {x : ℝ | x^2 - 2*x + 1 = 0} = {1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3297_329755


namespace NUMINAMATH_CALUDE_three_digit_sum_proof_l3297_329759

/-- Represents a three-digit number in the form xyz -/
def ThreeDigitNumber (x y z : Nat) : Nat :=
  100 * x + 10 * y + z

theorem three_digit_sum_proof (a b : Nat) :
  (ThreeDigitNumber 3 a 7) + 416 = (ThreeDigitNumber 7 b 3) ∧
  (ThreeDigitNumber 7 b 3) % 3 = 0 →
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_sum_proof_l3297_329759


namespace NUMINAMATH_CALUDE_max_soccer_balls_l3297_329770

/-- Represents the cost and quantity of soccer balls and basketballs -/
structure BallPurchase where
  soccer_cost : ℕ
  basketball_cost : ℕ
  total_balls : ℕ
  max_cost : ℕ

/-- Defines the conditions of the ball purchase problem -/
def ball_purchase_problem : BallPurchase where
  soccer_cost := 80
  basketball_cost := 60
  total_balls := 50
  max_cost := 3600

/-- Theorem stating the maximum number of soccer balls that can be purchased -/
theorem max_soccer_balls (bp : BallPurchase) : 
  bp.soccer_cost * 4 + bp.basketball_cost * 7 = 740 →
  bp.soccer_cost * 7 + bp.basketball_cost * 5 = 860 →
  ∃ (m : ℕ), m ≤ bp.total_balls ∧ 
             bp.soccer_cost * m + bp.basketball_cost * (bp.total_balls - m) ≤ bp.max_cost ∧
             ∀ (n : ℕ), n > m → 
               bp.soccer_cost * n + bp.basketball_cost * (bp.total_balls - n) > bp.max_cost :=
by sorry

#eval ball_purchase_problem.soccer_cost -- Expected output: 80
#eval ball_purchase_problem.basketball_cost -- Expected output: 60

end NUMINAMATH_CALUDE_max_soccer_balls_l3297_329770


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3297_329705

-- Define the displacement function
def h (t : ℝ) : ℝ := 15 * t - t^2

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 15 - 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 9 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l3297_329705


namespace NUMINAMATH_CALUDE_village_revenue_comparison_l3297_329724

def village_a : List ℝ := [5, 6, 6, 7, 8, 16]
def village_b : List ℝ := [4, 6, 8, 9, 10, 17]

theorem village_revenue_comparison :
  (village_a.sum / village_a.length) < (village_b.sum / village_b.length) := by
  sorry

end NUMINAMATH_CALUDE_village_revenue_comparison_l3297_329724


namespace NUMINAMATH_CALUDE_smallest_n_for_324_l3297_329735

/-- A geometric sequence (b_n) with given first three terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  b 1 = 2 ∧ b 2 = 6 ∧ b 3 = 18 ∧ ∀ n : ℕ, n ≥ 1 → b (n + 1) / b n = b 2 / b 1

/-- The smallest n for which b_n = 324 in the given geometric sequence is 5 -/
theorem smallest_n_for_324 (b : ℕ → ℝ) (h : geometric_sequence b) :
  (∃ n : ℕ, b n = 324) ∧ (∀ m : ℕ, b m = 324 → m ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_324_l3297_329735


namespace NUMINAMATH_CALUDE_store_purchase_combinations_l3297_329711

/-- The number of oreo flavors available -/
def num_oreo_flavors : ℕ := 6

/-- The number of milk flavors available -/
def num_milk_flavors : ℕ := 4

/-- The total number of items Alpha can choose from -/
def total_items : ℕ := num_oreo_flavors + num_milk_flavors

/-- The number of items they collectively buy -/
def total_purchased : ℕ := 3

/-- Represents the ways Alpha can choose items without repeats -/
def alpha_choices (k : ℕ) : ℕ := Nat.choose total_items k

/-- Represents the ways Beta can choose k oreos with possible repeats -/
def beta_choices (k : ℕ) : ℕ :=
  Nat.choose num_oreo_flavors k +  -- All different
  (if k ≥ 2 then num_oreo_flavors * (num_oreo_flavors - 1) else 0) +  -- Two same, one different (if k ≥ 2)
  (if k = 3 then num_oreo_flavors else 0)  -- All same (if k = 3)

/-- The total number of ways for Alpha and Beta to collectively buy 3 items -/
def total_ways : ℕ :=
  alpha_choices 3 +  -- Alpha buys 3, Beta 0
  alpha_choices 2 * num_oreo_flavors +  -- Alpha buys 2, Beta 1
  alpha_choices 1 * beta_choices 2 +  -- Alpha buys 1, Beta 2
  beta_choices 3  -- Alpha buys 0, Beta 3

theorem store_purchase_combinations :
  total_ways = 656 := by sorry

end NUMINAMATH_CALUDE_store_purchase_combinations_l3297_329711


namespace NUMINAMATH_CALUDE_sine_cosine_sum_simplification_l3297_329703

theorem sine_cosine_sum_simplification (x y : ℝ) : 
  Real.sin (x - 2*y) * Real.cos (3*y) + Real.cos (x - 2*y) * Real.sin (3*y) = Real.sin (x + y) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_simplification_l3297_329703


namespace NUMINAMATH_CALUDE_empire_state_building_total_height_l3297_329740

/-- The height of the Empire State Building -/
def empire_state_building_height (top_floor_height antenna_height : ℕ) : ℕ :=
  top_floor_height + antenna_height

/-- Theorem: The Empire State Building is 1454 feet tall -/
theorem empire_state_building_total_height :
  empire_state_building_height 1250 204 = 1454 := by
  sorry

end NUMINAMATH_CALUDE_empire_state_building_total_height_l3297_329740


namespace NUMINAMATH_CALUDE_mitch_family_milk_consumption_l3297_329767

/-- The amount of regular milk consumed by Mitch's family in 1 week -/
def regular_milk : ℚ := 1/2

/-- The amount of soy milk consumed by Mitch's family in 1 week -/
def soy_milk : ℚ := 1/10

/-- The total amount of milk consumed by Mitch's family in 1 week -/
def total_milk : ℚ := regular_milk + soy_milk

theorem mitch_family_milk_consumption :
  total_milk = 3/5 := by sorry

end NUMINAMATH_CALUDE_mitch_family_milk_consumption_l3297_329767


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achieved_l3297_329733

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  (a + 3*b + 5*c) * (a + b/3 + c/5) ≤ 9/5 := by
  sorry

theorem max_value_achieved (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀ + b₀ + c₀ = 1 ∧
    (a₀ + 3*b₀ + 5*c₀) * (a₀ + b₀/3 + c₀/5) = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achieved_l3297_329733


namespace NUMINAMATH_CALUDE_oscar_voting_theorem_l3297_329792

/-- Represents a vote for an actor and an actress -/
structure Vote where
  actor : ℕ
  actress : ℕ

/-- The problem statement -/
theorem oscar_voting_theorem 
  (votes : Finset Vote) 
  (vote_count : votes.card = 3366)
  (unique_counts : ∀ n : ℕ, 1 ≤ n → n ≤ 100 → 
    (∃ a : ℕ, (votes.filter (λ v => v.actor = a)).card = n) ∨ 
    (∃ b : ℕ, (votes.filter (λ v => v.actress = b)).card = n)) :
  ∃ v₁ v₂ : Vote, v₁ ∈ votes ∧ v₂ ∈ votes ∧ v₁ ≠ v₂ ∧ v₁.actor = v₂.actor ∧ v₁.actress = v₂.actress :=
by
  sorry

end NUMINAMATH_CALUDE_oscar_voting_theorem_l3297_329792


namespace NUMINAMATH_CALUDE_license_plate_count_l3297_329756

/-- Number of digits in the license plate -/
def num_digits : ℕ := 5

/-- Number of letters in the license plate -/
def num_letters : ℕ := 3

/-- Number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- Number of possible letters (A-Z) -/
def letter_choices : ℕ := 26

/-- Number of positions where the consecutive letters can be placed -/
def letter_positions : ℕ := num_digits + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := letter_positions * digit_choices^num_digits * letter_choices^num_letters

theorem license_plate_count : total_license_plates = 105456000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3297_329756


namespace NUMINAMATH_CALUDE_chalkboard_area_l3297_329753

/-- The area of a rectangular chalkboard with width 3 feet and length 2 times its width is 18 square feet. -/
theorem chalkboard_area (width : ℝ) (length : ℝ) : 
  width = 3 → length = 2 * width → width * length = 18 := by
  sorry

end NUMINAMATH_CALUDE_chalkboard_area_l3297_329753


namespace NUMINAMATH_CALUDE_overlapping_area_of_strips_l3297_329717

theorem overlapping_area_of_strips (total_length width : ℝ) 
  (left_length right_length : ℝ) (left_only_area right_only_area : ℝ) :
  total_length = 16 →
  left_length = 9 →
  right_length = 7 →
  left_length + right_length = total_length →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ),
    overlap_area = (left_length * width - left_only_area) ∧
    overlap_area = (right_length * width - right_only_area) ∧
    overlap_area = 13.5 :=
by sorry

end NUMINAMATH_CALUDE_overlapping_area_of_strips_l3297_329717


namespace NUMINAMATH_CALUDE_cubic_room_floor_perimeter_l3297_329747

/-- The perimeter of the floor of a cubic room -/
def floor_perimeter (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The perimeter of the floor of a cubic room with side length 5 meters is 20 meters -/
theorem cubic_room_floor_perimeter :
  floor_perimeter 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cubic_room_floor_perimeter_l3297_329747


namespace NUMINAMATH_CALUDE_brick_length_is_125_l3297_329787

/-- Represents the dimensions of a rectangular object in centimeters -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ :=
  d.length * d.width * d.height

/-- The dimensions of the wall in centimeters -/
def wall_dimensions : Dimensions :=
  { length := 800, width := 600, height := 22.5 }

/-- The partial dimensions of a brick in centimeters -/
def brick_partial_dimensions (x : ℝ) : Dimensions :=
  { length := x, width := 11.25, height := 6 }

/-- The number of bricks needed to build the wall -/
def number_of_bricks : ℕ := 1280

/-- Theorem stating that the length of each brick is 125 cm -/
theorem brick_length_is_125 : 
  ∃ x : ℝ, x = 125 ∧ 
  volume wall_dimensions = (number_of_bricks : ℝ) * volume (brick_partial_dimensions x) :=
sorry

end NUMINAMATH_CALUDE_brick_length_is_125_l3297_329787


namespace NUMINAMATH_CALUDE_triangle_area_l3297_329736

/-- A triangle with side lengths 6, 8, and 10 has an area of 24 square units. -/
theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3297_329736


namespace NUMINAMATH_CALUDE_total_pies_sold_is_29_l3297_329710

/-- Represents the number of pieces a shepherd's pie is cut into -/
def shepherds_pie_pieces : ℕ := 4

/-- Represents the number of pieces a chicken pot pie is cut into -/
def chicken_pot_pie_pieces : ℕ := 5

/-- Represents the number of customers who ordered slices of shepherd's pie -/
def shepherds_pie_orders : ℕ := 52

/-- Represents the number of customers who ordered slices of chicken pot pie -/
def chicken_pot_pie_orders : ℕ := 80

/-- Calculates the total number of pies sold by Chef Michel -/
def total_pies_sold : ℕ :=
  shepherds_pie_orders / shepherds_pie_pieces +
  chicken_pot_pie_orders / chicken_pot_pie_pieces

/-- Proves that the total number of pies sold is 29 -/
theorem total_pies_sold_is_29 : total_pies_sold = 29 := by
  sorry

end NUMINAMATH_CALUDE_total_pies_sold_is_29_l3297_329710


namespace NUMINAMATH_CALUDE_train_speed_l3297_329728

/-- Given a train that travels 80 km in 40 minutes, prove its speed is 120 kmph -/
theorem train_speed (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) : 
  distance = 80 ∧ time_minutes = 40 → speed = distance / (time_minutes / 60) → speed = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3297_329728


namespace NUMINAMATH_CALUDE_mixture_theorem_l3297_329789

/-- Represents a chemical solution with a given percentage of chemical a -/
structure Solution :=
  (percent_a : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution_x : Solution)
  (solution_y : Solution)
  (percent_x : ℝ)

/-- Calculates the percentage of chemical a in a mixture -/
def percent_a_in_mixture (m : Mixture) : ℝ :=
  m.solution_x.percent_a * m.percent_x + m.solution_y.percent_a * (1 - m.percent_x)

theorem mixture_theorem :
  let x : Solution := ⟨0.30⟩
  let y : Solution := ⟨0.40⟩
  let mixture : Mixture := ⟨x, y, 0.80⟩
  percent_a_in_mixture mixture = 0.32 := by
  sorry

end NUMINAMATH_CALUDE_mixture_theorem_l3297_329789


namespace NUMINAMATH_CALUDE_max_value_when_a_is_one_range_of_a_for_two_roots_l3297_329712

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 + 4 * x - 3 - a

-- Theorem for the maximum value of f when a = 1
theorem max_value_when_a_is_one :
  ∃ (max : ℝ), max = 2 ∧ ∀ x ∈ Set.Icc (-1) 1, f 1 x ≤ max :=
sorry

-- Theorem for the range of a when f has two distinct roots
theorem range_of_a_for_two_roots :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 
    a ∈ Set.Ioi 0 ∪ Set.Ioo (-1) 0 ∪ Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_max_value_when_a_is_one_range_of_a_for_two_roots_l3297_329712


namespace NUMINAMATH_CALUDE_cake_recipe_flour_calculation_l3297_329754

/-- Given a ratio of milk to flour and an amount of milk used, calculate the amount of flour needed. -/
def flour_needed (milk_ratio : ℚ) (flour_ratio : ℚ) (milk_used : ℚ) : ℚ :=
  (flour_ratio / milk_ratio) * milk_used

/-- The theorem states that given the specified ratio and milk amount, the flour needed is 1200 mL. -/
theorem cake_recipe_flour_calculation :
  let milk_ratio : ℚ := 60
  let flour_ratio : ℚ := 300
  let milk_used : ℚ := 240
  flour_needed milk_ratio flour_ratio milk_used = 1200 := by
sorry

#eval flour_needed 60 300 240

end NUMINAMATH_CALUDE_cake_recipe_flour_calculation_l3297_329754


namespace NUMINAMATH_CALUDE_cubic_equation_equivalence_l3297_329771

theorem cubic_equation_equivalence (x : ℝ) :
  x^3 + (x + 1)^4 + (x + 2)^3 = (x + 3)^4 ↔ 7 * (x^3 + 6 * x^2 + 13.14 * x + 10.29) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_equivalence_l3297_329771


namespace NUMINAMATH_CALUDE_circle_line_intersection_and_min_chord_l3297_329721

/-- Circle C: x^2 + y^2 - 4x - 2y - 20 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

/-- Line l: mx - y - m + 3 = 0 (m ∈ ℝ) -/
def line_l (m x y : ℝ) : Prop := m*x - y - m + 3 = 0

theorem circle_line_intersection_and_min_chord :
  (∀ m : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_l m x y) ∧
  (∃ min_length : ℝ, min_length = 4 * Real.sqrt 5 ∧
    ∀ m : ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ,
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l m x₁ y₁ ∧ line_l m x₂ y₂ →
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ min_length) ∧
  (∃ x y : ℝ, circle_C x y ∧ x - 2*y + 5 = 0 ∧
    ∀ x' y' : ℝ, circle_C x' y' ∧ x' - 2*y' + 5 = 0 →
      Real.sqrt ((x - x')^2 + (y - y')^2) = 4 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_and_min_chord_l3297_329721


namespace NUMINAMATH_CALUDE_escalator_speed_calculation_l3297_329707

/-- The speed of the escalator in feet per second. -/
def escalator_speed : ℝ := 12

/-- The length of the escalator in feet. -/
def escalator_length : ℝ := 160

/-- The walking speed of the person in feet per second. -/
def walking_speed : ℝ := 8

/-- The time taken to cover the entire length of the escalator in seconds. -/
def time_taken : ℝ := 8

theorem escalator_speed_calculation :
  (walking_speed + escalator_speed) * time_taken = escalator_length :=
by sorry

end NUMINAMATH_CALUDE_escalator_speed_calculation_l3297_329707


namespace NUMINAMATH_CALUDE_apple_basket_problem_l3297_329702

theorem apple_basket_problem (n : ℕ) (h1 : n > 1) : 
  (2 : ℝ) / n = (2 : ℝ) / 5 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l3297_329702


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3297_329745

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  is_arithmetic_sequence b →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 4 * b 5 = 10 →
  b 2 * b 7 = -224 ∨ b 2 * b 7 = -44 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3297_329745


namespace NUMINAMATH_CALUDE_g_243_equals_118_l3297_329742

/-- A function g with the property that g(a) + g(b) = m^3 when a + b = 3^m -/
def g_property (g : ℕ → ℝ) : Prop :=
  ∀ (a b m : ℕ), a > 0 → b > 0 → m > 0 → a + b = 3^m → g a + g b = (m : ℝ)^3

/-- The main theorem stating that g(243) = 118 -/
theorem g_243_equals_118 (g : ℕ → ℝ) (h : g_property g) : g 243 = 118 := by
  sorry


end NUMINAMATH_CALUDE_g_243_equals_118_l3297_329742


namespace NUMINAMATH_CALUDE_base5ToBinary_110_equals_11110_l3297_329765

-- Define a function to convert a number from base 5 to decimal
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define a function to convert a decimal number to binary
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec go (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else go (m / 2) ((m % 2) :: acc)
  go n []

-- Theorem statement
theorem base5ToBinary_110_equals_11110 :
  decimalToBinary (base5ToDecimal [0, 1, 1]) = [1, 1, 1, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_base5ToBinary_110_equals_11110_l3297_329765


namespace NUMINAMATH_CALUDE_concentration_reduction_proof_l3297_329762

def initial_concentration : ℝ := 0.9
def target_concentration : ℝ := 0.1
def concentration_reduction_factor : ℝ := 0.9

def minimum_operations : ℕ := 21

theorem concentration_reduction_proof :
  (∀ n : ℕ, n < minimum_operations → initial_concentration * concentration_reduction_factor ^ n ≥ target_concentration) ∧
  initial_concentration * concentration_reduction_factor ^ minimum_operations < target_concentration :=
by sorry

end NUMINAMATH_CALUDE_concentration_reduction_proof_l3297_329762


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_l3297_329709

/-- The position function of the object -/
def s (t : ℝ) : ℝ := 3 * t^2 + t + 4

/-- The velocity function of the object (derivative of s) -/
def v (t : ℝ) : ℝ := 6 * t + 1

theorem instantaneous_velocity_at_4 : v 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_l3297_329709


namespace NUMINAMATH_CALUDE_tangency_point_proof_l3297_329752

-- Define the two parabolas
def parabola1 (x y : ℚ) : Prop := y = x^2 + 20*x + 70
def parabola2 (x y : ℚ) : Prop := x = y^2 + 70*y + 1225

-- Define the point of tangency
def point_of_tangency : ℚ × ℚ := (-19/2, -69/2)

-- Theorem statement
theorem tangency_point_proof :
  let (x, y) := point_of_tangency
  parabola1 x y ∧ parabola2 x y ∧
  ∀ (x' y' : ℚ), x' ≠ x ∨ y' ≠ y →
    ¬(parabola1 x' y' ∧ parabola2 x' y') :=
by sorry

end NUMINAMATH_CALUDE_tangency_point_proof_l3297_329752


namespace NUMINAMATH_CALUDE_tan_double_angle_second_quadrant_l3297_329781

/-- Given an angle α in the second quadrant with sin(π + α) = -3/5, prove that tan(2α) = -24/7 -/
theorem tan_double_angle_second_quadrant (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin (π + α) = -3/5) : 
  Real.tan (2 * α) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_second_quadrant_l3297_329781


namespace NUMINAMATH_CALUDE_ball_throw_height_difference_l3297_329739

/-- A proof of the height difference between Janice's final throw and Christine's first throw -/
theorem ball_throw_height_difference :
  let christine_first : ℕ := 20
  let janice_first : ℕ := christine_first - 4
  let christine_second : ℕ := christine_first + 10
  let janice_second : ℕ := janice_first * 2
  let christine_third : ℕ := christine_second + 4
  let highest_throw : ℕ := 37
  let janice_third : ℕ := highest_throw
  janice_third - christine_first = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_throw_height_difference_l3297_329739


namespace NUMINAMATH_CALUDE_factorization_equality_l3297_329791

theorem factorization_equality (x : ℝ) : 
  x^2 * (x - 3) - 4 * (x - 3) = (x - 3) * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3297_329791


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l3297_329706

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {1, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l3297_329706


namespace NUMINAMATH_CALUDE_a_plus_b_value_l3297_329750

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem a_plus_b_value (a b : ℝ) : 
  A ∪ B a b = Set.univ ∧ A ∩ B a b = Set.Ioc 3 4 → a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l3297_329750


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l3297_329701

theorem power_mod_thirteen : 2^2010 ≡ 12 [ZMOD 13] := by sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l3297_329701


namespace NUMINAMATH_CALUDE_solution_existence_l3297_329766

theorem solution_existence (x y : ℝ) : 
  |x + 1| + (y - 8)^2 = 0 → x = -1 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_l3297_329766


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l3297_329725

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_k_value 
  (seq : ArithmeticSequence) 
  (k : ℕ) 
  (h1 : seq.S (k - 2) = -4)
  (h2 : seq.S k = 0)
  (h3 : seq.S (k + 2) = 8) :
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l3297_329725


namespace NUMINAMATH_CALUDE_average_percentage_decrease_l3297_329784

theorem average_percentage_decrease (initial_price final_price : ℝ) 
  (h1 : initial_price = 800)
  (h2 : final_price = 578)
  (h3 : final_price = initial_price * (1 - x)^2)
  : x = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_average_percentage_decrease_l3297_329784


namespace NUMINAMATH_CALUDE_james_truck_mpg_james_truck_mpg_proof_l3297_329780

/-- Proves that given the conditions of James's truck driving job, his truck's fuel efficiency is 20 miles per gallon. -/
theorem james_truck_mpg : ℝ → Prop :=
  λ mpg : ℝ =>
    let pay_per_mile : ℝ := 0.5
    let gas_cost_per_gallon : ℝ := 4
    let trip_distance : ℝ := 600
    let profit : ℝ := 180
    let earnings : ℝ := pay_per_mile * trip_distance
    let gas_cost : ℝ := (trip_distance / mpg) * gas_cost_per_gallon
    earnings - gas_cost = profit → mpg = 20

/-- The proof of james_truck_mpg. -/
theorem james_truck_mpg_proof : james_truck_mpg 20 := by
  sorry

end NUMINAMATH_CALUDE_james_truck_mpg_james_truck_mpg_proof_l3297_329780


namespace NUMINAMATH_CALUDE_divisible_by_25_l3297_329715

theorem divisible_by_25 (n : ℕ) : 25 ∣ (2^(n+2) * 3^n + 5*n - 4) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_25_l3297_329715


namespace NUMINAMATH_CALUDE_annes_cleaning_time_l3297_329798

/-- Represents the time (in hours) it takes Anne to clean the house by herself. -/
def annes_solo_time : ℝ := 12

/-- Represents Bruce's cleaning rate (portion of house cleaned per hour). -/
def bruce_rate : ℝ := sorry

/-- Represents Anne's cleaning rate (portion of house cleaned per hour). -/
def anne_rate : ℝ := sorry

/-- Theorem stating that Anne's solo cleaning time is 12 hours, given the conditions. -/
theorem annes_cleaning_time :
  (∀ (b a : ℝ),
    b = bruce_rate →
    a = anne_rate →
    (b + a) * 4 = 1 →
    (b + 2 * a) * 3 = 1 →
    a⁻¹ = annes_solo_time) :=
by sorry

end NUMINAMATH_CALUDE_annes_cleaning_time_l3297_329798


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l3297_329783

theorem cheryl_material_usage
  (material1_bought : ℚ)
  (material2_bought : ℚ)
  (material3_bought : ℚ)
  (material1_left : ℚ)
  (material2_left : ℚ)
  (material3_left : ℚ)
  (h1 : material1_bought = 4/9)
  (h2 : material2_bought = 2/3)
  (h3 : material3_bought = 5/6)
  (h4 : material1_left = 8/18)
  (h5 : material2_left = 3/9)
  (h6 : material3_left = 2/12) :
  (material1_bought - material1_left) + (material2_bought - material2_left) + (material3_bought - material3_left) = 1 := by
  sorry

#check cheryl_material_usage

end NUMINAMATH_CALUDE_cheryl_material_usage_l3297_329783
