import Mathlib

namespace NUMINAMATH_CALUDE_min_distance_sum_parabola_line_l24_2458

/-- The minimum distance sum from a point on the parabola y² = -4x to the y-axis and the line 2x + y - 4 = 0 -/
theorem min_distance_sum_parabola_line : 
  ∃ (min_sum : ℝ), 
    min_sum = (6 * Real.sqrt 5) / 5 - 1 ∧
    ∀ (x y : ℝ),
      y^2 = -4*x →  -- point (x,y) is on the parabola
      ∃ (m n : ℝ),
        m = |x| ∧   -- distance to y-axis
        n = |2*x + y - 4| / Real.sqrt 5 ∧  -- distance to line
        m + n ≥ min_sum :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_parabola_line_l24_2458


namespace NUMINAMATH_CALUDE_parabola_solution_l24_2400

/-- Parabola intersecting x-axis at two points -/
structure Parabola where
  a : ℝ
  intersectionA : ℝ × ℝ
  intersectionB : ℝ × ℝ

/-- The parabola y = a(x+1)^2 + 2 intersects the x-axis at A(-3, 0) and B -/
def parabola_problem (p : Parabola) : Prop :=
  p.intersectionA = (-3, 0) ∧
  p.intersectionA.2 = p.a * (p.intersectionA.1 + 1)^2 + 2 ∧
  p.intersectionB.2 = p.a * (p.intersectionB.1 + 1)^2 + 2 ∧
  p.intersectionA.2 = 0 ∧
  p.intersectionB.2 = 0

theorem parabola_solution (p : Parabola) (h : parabola_problem p) :
  p.a = -1/2 ∧ p.intersectionB = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_solution_l24_2400


namespace NUMINAMATH_CALUDE_flour_salt_difference_l24_2430

theorem flour_salt_difference (total_flour sugar total_salt flour_added : ℕ) : 
  total_flour = 12 → 
  sugar = 14 →
  total_salt = 7 → 
  flour_added = 2 → 
  (total_flour - flour_added) - total_salt = 3 := by
sorry

end NUMINAMATH_CALUDE_flour_salt_difference_l24_2430


namespace NUMINAMATH_CALUDE_tan_theta_range_l24_2479

-- Define the condition
def condition (θ : ℝ) : Prop := (Real.sin θ) / (Real.sqrt 3 * Real.cos θ + 1) > 1

-- Define the range of tan θ
def tan_range (x : ℝ) : Prop := x ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 3 / 3) (Real.sqrt 2)

-- Theorem statement
theorem tan_theta_range (θ : ℝ) : condition θ → tan_range (Real.tan θ) := by sorry

end NUMINAMATH_CALUDE_tan_theta_range_l24_2479


namespace NUMINAMATH_CALUDE_angle_D_measure_l24_2482

structure CyclicQuadrilateral where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_360 : A + B + C + D = 360
  ratio_ABC : ∃ (x : ℝ), A = 3*x ∧ B = 4*x ∧ C = 6*x

theorem angle_D_measure (q : CyclicQuadrilateral) : q.D = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l24_2482


namespace NUMINAMATH_CALUDE_impossibility_of_option_d_l24_2414

-- Define the basic rhombus shape
structure Rhombus :=
  (color : Bool)  -- True for white, False for gray

-- Define the operation of rotation
def rotate (r : Rhombus) : Rhombus := r

-- Define a larger shape as a collection of rhombuses
def LargerShape := List Rhombus

-- Define the four options
def option_a : LargerShape := sorry
def option_b : LargerShape := sorry
def option_c : LargerShape := sorry
def option_d : LargerShape := sorry

-- Define a function to check if a larger shape can be constructed
def can_construct (shape : LargerShape) : Prop := sorry

-- State the theorem
theorem impossibility_of_option_d :
  can_construct option_a ∧
  can_construct option_b ∧
  can_construct option_c ∧
  ¬ can_construct option_d :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_option_d_l24_2414


namespace NUMINAMATH_CALUDE_gabled_cuboid_theorem_l24_2461

/-- Represents a cuboid with gable-shaped figures on each face -/
structure GabledCuboid where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_bc : b > c

/-- Properties of the gabled cuboid -/
def GabledCuboidProperties (g : GabledCuboid) : Prop :=
  ∃ (num_faces num_edges num_vertices : ℕ) (volume : ℝ),
    num_faces = 12 ∧
    num_edges = 30 ∧
    num_vertices = 20 ∧
    volume = g.a * g.b * g.c + (1/2) * (g.a * g.b^2 + g.a * g.c^2 + g.b * g.c^2) - g.b^3/6 - g.c^3/3

theorem gabled_cuboid_theorem (g : GabledCuboid) : GabledCuboidProperties g := by
  sorry

end NUMINAMATH_CALUDE_gabled_cuboid_theorem_l24_2461


namespace NUMINAMATH_CALUDE_gel_pen_price_l24_2453

theorem gel_pen_price (x y b g : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : b > 0) (h4 : g > 0) : 
  ((x + y) * g = 4 * (x * b + y * g)) → 
  ((x + y) * b = (1/2) * (x * b + y * g)) → 
  g = 8 * b :=
by sorry

end NUMINAMATH_CALUDE_gel_pen_price_l24_2453


namespace NUMINAMATH_CALUDE_g_formula_and_domain_intersection_points_l24_2456

noncomputable section

-- Define the original function f
def f (x : ℝ) : ℝ := x + 1/x

-- Define the domain of f
def f_domain : Set ℝ := {x | x < 0 ∨ x > 0}

-- Define the symmetric function g
def g (x : ℝ) : ℝ := x - 2 + 1/(x-4)

-- Define the domain of g
def g_domain : Set ℝ := {x | x < 4 ∨ x > 4}

-- Define the symmetry point
def A : ℝ × ℝ := (2, 1)

-- Theorem for the correct formula and domain of g
theorem g_formula_and_domain :
  (∀ x ∈ g_domain, g x = x - 2 + 1/(x-4)) ∧
  (∀ x, x ∈ g_domain ↔ x < 4 ∨ x > 4) :=
sorry

-- Theorem for the intersection points
theorem intersection_points :
  (∀ b : ℝ, (∃! x, g x = b) ↔ b = 4 ∨ b = 0) ∧
  (g 5 = 4 ∧ g 3 = 0) :=
sorry

end

end NUMINAMATH_CALUDE_g_formula_and_domain_intersection_points_l24_2456


namespace NUMINAMATH_CALUDE_min_values_problem_l24_2495

theorem min_values_problem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a + b = 1) :
  (∀ x y : ℝ, x > y ∧ y > 0 ∧ x + y = 1 → a^2 + 2*b^2 ≤ x^2 + 2*y^2) ∧
  (∀ x y : ℝ, x > y ∧ y > 0 ∧ x + y = 1 → 4 / (a - b) + 1 / (2*b) ≤ 4 / (x - y) + 1 / (2*y)) ∧
  a^2 + 2*b^2 = 2/3 ∧
  4 / (a - b) + 1 / (2*b) = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_values_problem_l24_2495


namespace NUMINAMATH_CALUDE_factor_x8_minus_625_l24_2468

theorem factor_x8_minus_625 (x : ℝ) : 
  x^8 - 625 = (x^4 + 25) * (x^2 + 5) * (x + Real.sqrt 5) * (x - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_x8_minus_625_l24_2468


namespace NUMINAMATH_CALUDE_square_area_error_l24_2410

/-- Given a square with a side measurement error of 38% in excess,
    the percentage of error in the calculated area is 90.44%. -/
theorem square_area_error (S : ℝ) (S_pos : S > 0) :
  let measured_side := S * (1 + 0.38)
  let actual_area := S^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.9044 := by sorry

end NUMINAMATH_CALUDE_square_area_error_l24_2410


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l24_2450

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℕ) : ℚ :=
  (cost_price - selling_price : ℚ) / cost_price * 100

theorem radio_loss_percentage :
  let cost_price := 1500
  let selling_price := 1260
  loss_percentage cost_price selling_price = 16 := by
sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l24_2450


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l24_2423

theorem quadratic_roots_ratio (a b c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    (∀ x, a * x^2 - b * x + c = 0 ↔ x = x₁ ∨ x = x₂) ∧
    (∀ y, c * y^2 - a * y + b = 0 ↔ y = y₁ ∨ y = y₂) ∧
    (b / a ≥ 0) ∧
    (c / a = 9 * (a / c))) →
  (b / a) / (b / c) = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l24_2423


namespace NUMINAMATH_CALUDE_circle_plus_five_two_l24_2438

-- Define the ⊕ operation
def circle_plus (a b : ℝ) : ℝ := 5 * a + 2 * b

-- State the theorem
theorem circle_plus_five_two : circle_plus 5 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_five_two_l24_2438


namespace NUMINAMATH_CALUDE_perpendicular_condition_l24_2454

def is_perpendicular (a : ℝ) : Prop :=
  (a ≠ -1 ∧ a ≠ 0 ∧ -(a + 1) / (3 * a) * ((1 - a) / (a + 1)) = -1) ∨
  (a = -1)

theorem perpendicular_condition (a : ℝ) :
  (a = 1/4 → is_perpendicular a) ∧
  ¬(is_perpendicular a → a = 1/4) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l24_2454


namespace NUMINAMATH_CALUDE_largest_side_of_special_triangle_l24_2467

/-- Given a scalene triangle with sides x and y, and area Δ, satisfying the equation
    x + 2Δ/x = y + 2Δ/y, prove that when x = 60 and y = 63, the largest side is 87. -/
theorem largest_side_of_special_triangle (x y Δ : ℝ) 
  (hx : x = 60)
  (hy : y = 63)
  (h_eq : x + 2 * Δ / x = y + 2 * Δ / y)
  (h_scalene : x ≠ y)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_pos_Δ : Δ > 0) :
  max x (max y (Real.sqrt (x^2 + y^2))) = 87 :=
sorry

end NUMINAMATH_CALUDE_largest_side_of_special_triangle_l24_2467


namespace NUMINAMATH_CALUDE_manoj_lending_amount_l24_2424

/-- The amount Manoj borrowed from Anwar -/
def borrowed_amount : ℝ := 3900

/-- The interest rate for borrowing (as a decimal) -/
def borrowing_rate : ℝ := 0.06

/-- The interest rate for lending (as a decimal) -/
def lending_rate : ℝ := 0.09

/-- The duration of both borrowing and lending in years -/
def duration : ℝ := 3

/-- Manoj's total gain from the transaction -/
def total_gain : ℝ := 824.85

/-- Calculate simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The sum lent by Manoj to Ramu -/
def lent_amount : ℝ := 4355

theorem manoj_lending_amount :
  simple_interest lent_amount lending_rate duration -
  simple_interest borrowed_amount borrowing_rate duration =
  total_gain := by sorry

end NUMINAMATH_CALUDE_manoj_lending_amount_l24_2424


namespace NUMINAMATH_CALUDE_snail_wins_l24_2435

-- Define the race parameters
def race_distance : ℝ := 200

-- Define the animals' movements
structure Snail where
  speed : ℝ
  
structure Rabbit where
  initial_distance : ℝ
  speed : ℝ
  run_time1 : ℝ
  nap_time1 : ℝ
  run_time2 : ℝ
  nap_time2 : ℝ

-- Define the race conditions
def race_conditions (s : Snail) (r : Rabbit) : Prop :=
  s.speed > 0 ∧
  r.speed > 0 ∧
  r.initial_distance = 120 ∧
  r.run_time1 > 0 ∧
  r.nap_time1 > 0 ∧
  r.run_time2 > 0 ∧
  r.nap_time2 > 0 ∧
  r.initial_distance + r.speed * (r.run_time1 + r.run_time2) = race_distance

-- Theorem statement
theorem snail_wins (s : Snail) (r : Rabbit) 
  (h : race_conditions s r) : 
  s.speed * (r.run_time1 + r.nap_time1 + r.run_time2 + r.nap_time2) = race_distance :=
sorry

end NUMINAMATH_CALUDE_snail_wins_l24_2435


namespace NUMINAMATH_CALUDE_same_function_constant_one_and_x_power_zero_l24_2449

theorem same_function_constant_one_and_x_power_zero :
  ∀ x : ℝ, (1 : ℝ) = x^(0 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_same_function_constant_one_and_x_power_zero_l24_2449


namespace NUMINAMATH_CALUDE_b_range_l24_2418

def P : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}

def Q (b : ℝ) : Set ℝ := {x | x^2 - (b+2)*x + 2*b ≤ 0}

theorem b_range (b : ℝ) : P ⊇ Q b ↔ b ∈ Set.Icc 1 4 := by
  sorry

end NUMINAMATH_CALUDE_b_range_l24_2418


namespace NUMINAMATH_CALUDE_problem_I_problem_II_l24_2421

theorem problem_I (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin (π - α) - 2 * Real.cos (-α)) / (3 * Real.cos (π/2 - α) - 5 * Real.cos (π + α)) = 5/7 := by
  sorry

theorem problem_II (x : Real) (h1 : Real.sin x + Real.cos x = 1/5) (h2 : 0 < x) (h3 : x < π) :
  Real.sin x = 4/5 ∧ Real.cos x = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_I_problem_II_l24_2421


namespace NUMINAMATH_CALUDE_product_expansion_l24_2490

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * (7 / x^2 + 7*x - 7/x) = 3 / x^2 + 3*x - 3/x := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l24_2490


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l24_2475

theorem solve_fraction_equation :
  ∃ x : ℚ, (1 / 4 : ℚ) - (1 / 5 : ℚ) = 1 / x ∧ x = 20 := by
sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l24_2475


namespace NUMINAMATH_CALUDE_new_student_weight_l24_2492

/-- Calculates the weight of a new student given the initial and final conditions of a group of students. -/
theorem new_student_weight
  (initial_count : ℕ)
  (initial_avg : ℝ)
  (final_count : ℕ)
  (final_avg : ℝ)
  (h1 : initial_count = 19)
  (h2 : initial_avg = 15)
  (h3 : final_count = initial_count + 1)
  (h4 : final_avg = 14.6) :
  final_count * final_avg - initial_count * initial_avg = 7 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l24_2492


namespace NUMINAMATH_CALUDE_stream_speed_l24_2404

/-- Given that a canoe rows upstream at 9 km/hr and downstream at 12 km/hr,
    the speed of the stream is 1.5 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ)
  (h_upstream : upstream_speed = 9)
  (h_downstream : downstream_speed = 12) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l24_2404


namespace NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_l24_2406

theorem infinitely_many_primes_3_mod_4 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 4 = 3} := by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_l24_2406


namespace NUMINAMATH_CALUDE_shaded_region_characterization_l24_2439

def shaded_region (z : ℂ) : Prop :=
  Complex.abs z ≤ 1 ∧ Complex.im z ≥ (1/2 : ℝ)

theorem shaded_region_characterization :
  ∀ z : ℂ, z ∈ {z | shaded_region z} ↔ 
    (Complex.abs z ≤ 1 ∧ Complex.im z ≥ (1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_characterization_l24_2439


namespace NUMINAMATH_CALUDE_totalPaintingCost_l24_2447

/-- Calculates the sum of digits for a given range of an arithmetic sequence -/
def sumOfDigits (start : Nat) (diff : Nat) (count : Nat) : Nat :=
  sorry

/-- Calculates the total cost to paint house numbers on one side of the street -/
def sideCost (start : Nat) (diff : Nat) (count : Nat) : Nat :=
  sorry

/-- The total cost to paint all house numbers on the street -/
theorem totalPaintingCost : 
  let eastSideCost := sideCost 5 7 25
  let westSideCost := sideCost 6 8 25
  eastSideCost + westSideCost = 123 := by
  sorry

end NUMINAMATH_CALUDE_totalPaintingCost_l24_2447


namespace NUMINAMATH_CALUDE_inequality_solution_range_l24_2499

theorem inequality_solution_range (a : ℝ) : 
  (∃ x ∈ Set.Ioo 1 4, x^2 - 4*x - 2 - a > 0) → a ∈ Set.Ioi (-2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l24_2499


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l24_2485

theorem sum_of_solutions_quadratic (x : ℝ) :
  let a : ℝ := -3
  let b : ℝ := -18
  let c : ℝ := 81
  let sum_of_roots := -b / a
  (a * x^2 + b * x + c = 0) → sum_of_roots = -6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l24_2485


namespace NUMINAMATH_CALUDE_smallest_class_size_l24_2496

/-- Represents a class of students and their test scores. -/
structure TestClass where
  n : ℕ              -- Total number of students
  scores : Fin n → ℕ -- Scores of each student

/-- Conditions for the test class. -/
def validTestClass (c : TestClass) : Prop :=
  (∀ i, c.scores i ≥ 70 ∧ c.scores i ≤ 120) ∧
  (∃ s : Finset (Fin c.n), s.card = 7 ∧ ∀ i ∈ s, c.scores i = 120) ∧
  (Finset.sum (Finset.univ : Finset (Fin c.n)) c.scores / c.n = 85)

/-- The theorem stating the smallest possible number of students. -/
theorem smallest_class_size :
  ∀ c : TestClass, validTestClass c → c.n ≥ 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l24_2496


namespace NUMINAMATH_CALUDE_product_of_powers_equals_sum_l24_2484

theorem product_of_powers_equals_sum (w x y z k : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 11^k = 900 → 2*w + 3*x + 5*y + 7*z + 11*k = 20 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_sum_l24_2484


namespace NUMINAMATH_CALUDE_median_of_consecutive_integers_l24_2476

theorem median_of_consecutive_integers (n : ℕ) (sum : ℕ) (h1 : n = 36) (h2 : sum = 1296) :
  sum / n = 36 := by
  sorry

end NUMINAMATH_CALUDE_median_of_consecutive_integers_l24_2476


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l24_2408

theorem least_positive_integer_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  (∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 → m ≥ n) :=
by
  use 2519
  sorry

#eval 2519 % 3  -- Should output 2
#eval 2519 % 4  -- Should output 3
#eval 2519 % 5  -- Should output 4
#eval 2519 % 6  -- Should output 5
#eval 2519 % 7  -- Should output 6

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l24_2408


namespace NUMINAMATH_CALUDE_weekend_haircut_price_l24_2448

theorem weekend_haircut_price (weekday_price : ℝ) (weekend_markup : ℝ) : 
  weekday_price = 18 → weekend_markup = 0.5 → weekday_price * (1 + weekend_markup) = 27 := by
  sorry

end NUMINAMATH_CALUDE_weekend_haircut_price_l24_2448


namespace NUMINAMATH_CALUDE_cubic_root_sum_inverse_squares_l24_2487

theorem cubic_root_sum_inverse_squares (a b c : ℝ) : 
  a^3 - 8*a^2 + 6*a - 3 = 0 →
  b^3 - 8*b^2 + 6*b - 3 = 0 →
  c^3 - 8*c^2 + 6*c - 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_inverse_squares_l24_2487


namespace NUMINAMATH_CALUDE_inequality_requires_conditional_structure_l24_2440

-- Define the types of algorithms
inductive Algorithm
  | SolveInequality
  | CalculateAverage
  | CalculateCircleArea
  | FindRoots

-- Define a function to check if an algorithm requires a conditional structure
def requiresConditionalStructure (alg : Algorithm) : Prop :=
  match alg with
  | Algorithm.SolveInequality => true
  | _ => false

-- Theorem statement
theorem inequality_requires_conditional_structure :
  requiresConditionalStructure Algorithm.SolveInequality ∧
  ¬requiresConditionalStructure Algorithm.CalculateAverage ∧
  ¬requiresConditionalStructure Algorithm.CalculateCircleArea ∧
  ¬requiresConditionalStructure Algorithm.FindRoots :=
sorry

end NUMINAMATH_CALUDE_inequality_requires_conditional_structure_l24_2440


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_21_8_l24_2407

theorem product_of_fractions_equals_21_8 : 
  let f (n : ℕ) := (n^3 - 1) * (n - 2) / (n^3 + 1)
  f 3 * f 5 * f 7 * f 9 * f 11 = 21 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_21_8_l24_2407


namespace NUMINAMATH_CALUDE_junior_has_sixteen_rabbits_l24_2443

/-- The number of toys bought on Monday -/
def monday_toys : ℕ := 6

/-- The number of toys bought on Wednesday -/
def wednesday_toys : ℕ := 2 * monday_toys

/-- The number of toys bought on Friday -/
def friday_toys : ℕ := 4 * monday_toys

/-- The number of toys bought on Saturday -/
def saturday_toys : ℕ := wednesday_toys / 2

/-- The total number of toys -/
def total_toys : ℕ := monday_toys + wednesday_toys + friday_toys + saturday_toys

/-- The number of toys each rabbit receives -/
def toys_per_rabbit : ℕ := 3

/-- The number of rabbits Junior has -/
def num_rabbits : ℕ := total_toys / toys_per_rabbit

theorem junior_has_sixteen_rabbits : num_rabbits = 16 := by
  sorry

end NUMINAMATH_CALUDE_junior_has_sixteen_rabbits_l24_2443


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l24_2452

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 12 →
  (1/2) * a * b = 6 →
  a^2 + b^2 = c^2 →
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l24_2452


namespace NUMINAMATH_CALUDE_sequence_general_term_l24_2478

/-- Given a sequence {a_n} where S_n is the sum of its first n terms, 
    prove that if S_n + a_n = (n-1) / (n(n+1)) for n ≥ 1, 
    then a_n = 1/(2^n) - 1/(n(n+1)) for all n ≥ 1 -/
theorem sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n ≥ 1 → S n + a n = (n - 1 : ℚ) / (n * (n + 1))) →
  ∀ n : ℕ, n ≥ 1 → a n = 1 / (2 ^ n) - 1 / (n * (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l24_2478


namespace NUMINAMATH_CALUDE_line_through_P_with_opposite_sign_intercepts_l24_2459

-- Define the point P
def P : ℝ × ℝ := (3, -2)

-- Define the line equation types
inductive LineEquation
| Standard (a b c : ℝ) : LineEquation  -- ax + by + c = 0
| SlopeIntercept (m b : ℝ) : LineEquation  -- y = mx + b

-- Define a predicate for a line passing through a point
def passesThrough (eq : LineEquation) (p : ℝ × ℝ) : Prop :=
  match eq with
  | LineEquation.Standard a b c => a * p.1 + b * p.2 + c = 0
  | LineEquation.SlopeIntercept m b => p.2 = m * p.1 + b

-- Define a predicate for a line having intercepts of opposite signs
def hasOppositeSignIntercepts (eq : LineEquation) : Prop :=
  match eq with
  | LineEquation.Standard a b c =>
    (c / a) * (c / b) < 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
  | LineEquation.SlopeIntercept m b =>
    b * (b / m) < 0 ∧ m ≠ 0 ∧ b ≠ 0

-- The main theorem
theorem line_through_P_with_opposite_sign_intercepts :
  ∃ (eq : LineEquation),
    (eq = LineEquation.Standard 1 (-1) (-5) ∨ eq = LineEquation.SlopeIntercept (-2/3) 0) ∧
    passesThrough eq P ∧
    hasOppositeSignIntercepts eq :=
  sorry

end NUMINAMATH_CALUDE_line_through_P_with_opposite_sign_intercepts_l24_2459


namespace NUMINAMATH_CALUDE_identity_function_divisibility_l24_2442

theorem identity_function_divisibility (f : ℕ+ → ℕ+) :
  (∀ (a b : ℕ+), (a.val ^ 2 + (f a).val * (f b).val) % ((f a).val + b.val) = 0) →
  (∀ (n : ℕ+), f n = n) :=
by sorry

end NUMINAMATH_CALUDE_identity_function_divisibility_l24_2442


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l24_2455

theorem diophantine_equation_solution (x y : ℤ) :
  x^2 - 5*y^2 = 1 →
  ∃ n : ℕ, (x + y * Real.sqrt 5 = (9 + 4 * Real.sqrt 5)^n) ∨
           (x + y * Real.sqrt 5 = -(9 + 4 * Real.sqrt 5)^n) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l24_2455


namespace NUMINAMATH_CALUDE_range_of_m_l24_2419

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (2 - m) + y^2 / (m - 1) = 1 → m > 2) →
  (∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0) →
  ((m > 2) ∨ (1 < m ∧ m < 3)) →
  ¬(1 < m ∧ m < 3) →
  m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l24_2419


namespace NUMINAMATH_CALUDE_arrangements_with_restriction_l24_2446

def num_actors : ℕ := 6

-- Define a function to calculate the number of arrangements
def num_arrangements (n : ℕ) (restricted_positions : ℕ) : ℕ :=
  (n - restricted_positions) * (n - 1).factorial

-- Theorem statement
theorem arrangements_with_restriction :
  num_arrangements num_actors 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_restriction_l24_2446


namespace NUMINAMATH_CALUDE_historical_fiction_new_releases_l24_2434

theorem historical_fiction_new_releases 
  (total_inventory : ℝ)
  (historical_fiction_ratio : ℝ)
  (historical_fiction_new_release_ratio : ℝ)
  (other_new_release_ratio : ℝ)
  (h1 : historical_fiction_ratio = 0.3)
  (h2 : historical_fiction_new_release_ratio = 0.3)
  (h3 : other_new_release_ratio = 0.4)
  (h4 : total_inventory > 0) :
  let historical_fiction := total_inventory * historical_fiction_ratio
  let other_books := total_inventory * (1 - historical_fiction_ratio)
  let historical_fiction_new_releases := historical_fiction * historical_fiction_new_release_ratio
  let other_new_releases := other_books * other_new_release_ratio
  let total_new_releases := historical_fiction_new_releases + other_new_releases
  historical_fiction_new_releases / total_new_releases = 9 / 37 := by
    sorry

end NUMINAMATH_CALUDE_historical_fiction_new_releases_l24_2434


namespace NUMINAMATH_CALUDE_fraction_simplification_l24_2497

theorem fraction_simplification : (2020 : ℚ) / (20 * 20) = 5.05 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l24_2497


namespace NUMINAMATH_CALUDE_board_block_system_l24_2472

/-- A proof problem about forces and acceleration on a board and block system. -/
theorem board_block_system 
  (M : Real) (m : Real) (μ : Real) (g : Real) (a : Real)
  (hM : M = 4)
  (hm : m = 1)
  (hμ : μ = 0.2)
  (hg : g = 10)
  (ha : a = g / 5) :
  let T := m * (a + μ * g)
  let F := μ * g * (M + 2 * m) + M * a + T
  T = 4 ∧ F = 24 := by
  sorry


end NUMINAMATH_CALUDE_board_block_system_l24_2472


namespace NUMINAMATH_CALUDE_unique_n_solution_l24_2480

theorem unique_n_solution : ∃! (n : ℕ+), 
  Real.cos (π / (2 * n.val)) - Real.sin (π / (2 * n.val)) = Real.sqrt n.val / 2 :=
by
  -- The unique solution is n = 4
  use 4
  constructor
  -- Proof that n = 4 satisfies the equation
  sorry
  -- Proof of uniqueness
  sorry

end NUMINAMATH_CALUDE_unique_n_solution_l24_2480


namespace NUMINAMATH_CALUDE_equal_fractions_k_value_l24_2402

theorem equal_fractions_k_value 
  (x y z k : ℝ) 
  (h : (8 : ℝ) / (x + y + 1) = k / (x + z + 2) ∧ 
       k / (x + z + 2) = (12 : ℝ) / (z - y + 3)) : 
  k = 20 := by sorry

end NUMINAMATH_CALUDE_equal_fractions_k_value_l24_2402


namespace NUMINAMATH_CALUDE_valid_lineups_count_l24_2488

/-- The number of players in the team -/
def total_players : ℕ := 15

/-- The number of players in a starting lineup -/
def lineup_size : ℕ := 6

/-- The number of players who refuse to play together -/
def refusing_players : ℕ := 3

/-- Calculates the number of valid lineups -/
def valid_lineups : ℕ := 
  Nat.choose total_players lineup_size - Nat.choose (total_players - refusing_players) (lineup_size - refusing_players)

theorem valid_lineups_count : valid_lineups = 4785 := by sorry

end NUMINAMATH_CALUDE_valid_lineups_count_l24_2488


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l24_2429

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 4*x} = {0, 4} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l24_2429


namespace NUMINAMATH_CALUDE_cricket_team_throwers_l24_2481

theorem cricket_team_throwers (total_players : ℕ) (right_handed : ℕ) : 
  total_players = 61 → right_handed = 53 → ∃ (throwers : ℕ), 
    throwers = 37 ∧ 
    throwers ≤ right_handed ∧
    throwers ≤ total_players ∧
    3 * (right_handed - throwers) = 2 * (total_players - throwers) := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_throwers_l24_2481


namespace NUMINAMATH_CALUDE_terminal_side_point_y_value_l24_2477

theorem terminal_side_point_y_value (α : Real) (y : Real) :
  let P : Real × Real := (-Real.sqrt 3, y)
  (P.1^2 + P.2^2 ≠ 0) →  -- Ensure the point is not at the origin
  (Real.sin α = Real.sqrt 13 / 13) →
  y = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_terminal_side_point_y_value_l24_2477


namespace NUMINAMATH_CALUDE_sum_parity_from_cube_sum_parity_l24_2474

theorem sum_parity_from_cube_sum_parity (n m : ℤ) (h : Even (n^3 + m^3)) : Even (n + m) := by
  sorry

end NUMINAMATH_CALUDE_sum_parity_from_cube_sum_parity_l24_2474


namespace NUMINAMATH_CALUDE_bamboo_problem_l24_2401

def arithmetic_sequence (a : ℚ → ℚ) (n : ℕ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ k, a k = a₁ + (k - 1) * d

theorem bamboo_problem (a : ℚ → ℚ) :
  arithmetic_sequence a 9 →
  (a 1 + a 2 + a 3 + a 4 = 3) →
  (a 7 + a 8 + a 9 = 4) →
  a 5 = 67 / 66 := by
sorry

end NUMINAMATH_CALUDE_bamboo_problem_l24_2401


namespace NUMINAMATH_CALUDE_fifth_stair_area_fifth_stair_perimeter_twelfth_stair_area_twentyfifth_stair_perimeter_l24_2422

-- Define the stair structure
structure Stair :=
  (n : ℕ)

-- Define the area function
def area (s : Stair) : ℕ :=
  s.n * (s.n + 1) / 2

-- Define the perimeter function
def perimeter (s : Stair) : ℕ :=
  4 * s.n

-- Theorem statements
theorem fifth_stair_area :
  area { n := 5 } = 15 := by sorry

theorem fifth_stair_perimeter :
  perimeter { n := 5 } = 20 := by sorry

theorem twelfth_stair_area :
  area { n := 12 } = 78 := by sorry

theorem twentyfifth_stair_perimeter :
  perimeter { n := 25 } = 100 := by sorry

end NUMINAMATH_CALUDE_fifth_stair_area_fifth_stair_perimeter_twelfth_stair_area_twentyfifth_stair_perimeter_l24_2422


namespace NUMINAMATH_CALUDE_power_of_product_with_negative_l24_2483

theorem power_of_product_with_negative (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_with_negative_l24_2483


namespace NUMINAMATH_CALUDE_austin_starting_amount_l24_2444

def robot_cost : ℚ := 875 / 100
def discount_rate : ℚ := 1 / 10
def coupon_discount : ℚ := 5
def tax_rate : ℚ := 2 / 25
def total_tax : ℚ := 722 / 100
def shipping_fee : ℚ := 499 / 100
def gift_card : ℚ := 25
def change : ℚ := 1153 / 100

def total_robots : ℕ := 2 * 1 + 3 * 2 + 2 * 3

theorem austin_starting_amount (initial_amount : ℚ) :
  (∃ (discounted_price : ℚ),
    discounted_price = total_robots * robot_cost * (1 - discount_rate) - coupon_discount ∧
    total_tax = discounted_price * tax_rate ∧
    initial_amount = discounted_price + total_tax + shipping_fee - gift_card + change) →
  initial_amount = 7746 / 100 :=
by sorry

end NUMINAMATH_CALUDE_austin_starting_amount_l24_2444


namespace NUMINAMATH_CALUDE_only_group_D_forms_triangle_l24_2425

/-- Triangle inequality theorem for a set of three lengths -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Groups of line segments -/
def group_A : (ℝ × ℝ × ℝ) := (3, 8, 5)
def group_B : (ℝ × ℝ × ℝ) := (12, 5, 6)
def group_C : (ℝ × ℝ × ℝ) := (5, 5, 10)
def group_D : (ℝ × ℝ × ℝ) := (15, 10, 7)

/-- Theorem: Only group D can form a triangle -/
theorem only_group_D_forms_triangle :
  ¬(triangle_inequality group_A.1 group_A.2.1 group_A.2.2) ∧
  ¬(triangle_inequality group_B.1 group_B.2.1 group_B.2.2) ∧
  ¬(triangle_inequality group_C.1 group_C.2.1 group_C.2.2) ∧
  (triangle_inequality group_D.1 group_D.2.1 group_D.2.2) :=
by sorry

end NUMINAMATH_CALUDE_only_group_D_forms_triangle_l24_2425


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l24_2469

/-- Represents the price and volume of orangeade on a given day -/
structure OrangeadeDay where
  orange_juice : ℝ
  water : ℝ
  price : ℝ

/-- The orangeade scenario over two days -/
def OrangeadeScenario (day1 day2 : OrangeadeDay) : Prop :=
  day1.orange_juice > 0 ∧
  day1.water = day1.orange_juice ∧
  day2.orange_juice = day1.orange_juice ∧
  day2.water = 2 * day1.water ∧
  day1.price = 0.5 ∧
  (day1.orange_juice + day1.water) * day1.price = (day2.orange_juice + day2.water) * day2.price

theorem orangeade_price_day2 (day1 day2 : OrangeadeDay) 
  (h : OrangeadeScenario day1 day2) : day2.price = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l24_2469


namespace NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l24_2465

theorem smallest_integer_in_consecutive_set (n : ℤ) : 
  (n + 6 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7)) →
  n = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l24_2465


namespace NUMINAMATH_CALUDE_parabolic_triangle_area_l24_2405

theorem parabolic_triangle_area (n : ℕ) : 
  ∃ (a b : ℤ) (m : ℕ), 
    Odd m ∧ 
    (a * (b^2 - a^2) : ℤ) = (2^n * m)^2 := by
  sorry

end NUMINAMATH_CALUDE_parabolic_triangle_area_l24_2405


namespace NUMINAMATH_CALUDE_intersection_of_lines_l24_2451

theorem intersection_of_lines :
  ∃! p : ℝ × ℝ, 
    5 * p.1 - 3 * p.2 = 15 ∧ 
    6 * p.1 + 2 * p.2 = 14 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l24_2451


namespace NUMINAMATH_CALUDE_train_length_l24_2494

/-- The length of a train given its speed, bridge length, and crossing time -/
theorem train_length (speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  speed = 45 * 1000 / 3600 →
  bridge_length = 205 →
  crossing_time = 30 →
  speed * crossing_time - bridge_length = 170 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l24_2494


namespace NUMINAMATH_CALUDE_train_length_train_length_example_l24_2464

/-- The length of a train given its speed and time to cross a pole --/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ := by
  sorry

/-- Proof that a train with speed 60 km/hr crossing a pole in 4 seconds has a length of approximately 66.68 meters --/
theorem train_length_example : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_length 60 4 - 66.68| < ε := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_example_l24_2464


namespace NUMINAMATH_CALUDE_power_of_power_equals_ten_l24_2426

theorem power_of_power_equals_ten (m : ℝ) : (m^2)^5 = m^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_equals_ten_l24_2426


namespace NUMINAMATH_CALUDE_min_value_a_over_b_l24_2457

theorem min_value_a_over_b (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (h : 2 * Real.sqrt a + b = 1) :
  ∃ (x : ℝ), x = a / b ∧ ∀ (y : ℝ), y = a / b → x ≤ y ∧ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_over_b_l24_2457


namespace NUMINAMATH_CALUDE_x₂_1994th_place_l24_2427

-- Define the equation
def equation (x : ℝ) : Prop := x * Real.sqrt 8 + 1 / (x * Real.sqrt 8) = Real.sqrt 8

-- Define the two real solutions
axiom x₁ : ℝ
axiom x₂ : ℝ

-- Define that x₁ and x₂ satisfy the equation
axiom x₁_satisfies : equation x₁
axiom x₂_satisfies : equation x₂

-- Define the decimal place function (simplified for this problem)
def decimal_place (x : ℝ) (n : ℕ) : ℕ := sorry

-- Define that the 1994th decimal place of x₁ is 6
axiom x₁_1994th_place : decimal_place x₁ 1994 = 6

-- Theorem to prove
theorem x₂_1994th_place : decimal_place x₂ 1994 = 3 := by sorry

end NUMINAMATH_CALUDE_x₂_1994th_place_l24_2427


namespace NUMINAMATH_CALUDE_a_plus_b_squared_l24_2445

theorem a_plus_b_squared (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 2) (h3 : a < b) :
  (a + b)^2 = 1 ∨ (a + b)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_squared_l24_2445


namespace NUMINAMATH_CALUDE_exists_five_digit_number_with_digit_sum_31_divisible_by_31_l24_2428

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem exists_five_digit_number_with_digit_sum_31_divisible_by_31 :
  ∃ n : ℕ, is_five_digit n ∧ digit_sum n = 31 ∧ n % 31 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_five_digit_number_with_digit_sum_31_divisible_by_31_l24_2428


namespace NUMINAMATH_CALUDE_divisibility_problem_l24_2432

theorem divisibility_problem (p : ℕ) (hp : p.Prime) (hp_gt_7 : p > 7) 
  (hp_mod_6 : p % 6 = 1) (m : ℕ) (hm : m = 2^p - 1) : 
  (127 * m) ∣ (2^(m-1) - 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l24_2432


namespace NUMINAMATH_CALUDE_original_expression_proof_l24_2470

theorem original_expression_proof (X a b c : ℤ) : 
  X + (a*b - 2*b*c + 3*a*c) = 2*b*c - 3*a*c + 2*a*b → 
  X = 4*b*c - 6*a*c + a*b := by
sorry

end NUMINAMATH_CALUDE_original_expression_proof_l24_2470


namespace NUMINAMATH_CALUDE_equality_implications_l24_2433

theorem equality_implications (a b x y : ℝ) (h : a = b) : 
  (a - 3 = b - 3) ∧ 
  (3 * a = 3 * b) ∧ 
  ((a + 3) / 4 = (b + 3) / 4) ∧
  (∃ x y, a * x ≠ b * y) := by
sorry

end NUMINAMATH_CALUDE_equality_implications_l24_2433


namespace NUMINAMATH_CALUDE_problem_solution_l24_2486

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem problem_solution :
  let x := sum_integers 20 30
  let y := count_even_integers 20 30
  x + y = 281 → y = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l24_2486


namespace NUMINAMATH_CALUDE_polynomial_division_l24_2420

-- Define the dividend polynomial
def dividend (z : ℚ) : ℚ := 4*z^5 - 9*z^4 + 7*z^3 - 12*z^2 + 8*z - 3

-- Define the divisor polynomial
def divisor (z : ℚ) : ℚ := 2*z + 3

-- Define the quotient polynomial
def quotient (z : ℚ) : ℚ := 2*z^4 - 5*z^3 + 4*z^2 - (5/2)*z + 3/4

-- State the theorem
theorem polynomial_division :
  ∀ z : ℚ, dividend z / divisor z = quotient z :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_l24_2420


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l24_2437

/-- An increasing arithmetic sequence with specific initial conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) > a n) ∧  -- Increasing
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧  -- Arithmetic
  (a 1 = 1) ∧  -- Initial condition
  (a 3 = (a 2)^2 - 4)  -- Given relation

/-- The theorem stating the general term of the sequence -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) : 
  ∀ n : ℕ, a n = 2 * n - 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l24_2437


namespace NUMINAMATH_CALUDE_max_slope_no_lattice_points_l24_2416

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := True

-- Define the line equation
def on_line (m : ℚ) (x y : ℤ) : Prop := y = m * x + 2

-- Define the condition for no lattice points
def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 100 → is_lattice_point x y → ¬(on_line m x y)

-- State the theorem
theorem max_slope_no_lattice_points :
  (∀ m : ℚ, 1/2 < m → m < 50/99 → no_lattice_points m) ∧
  ¬(∀ m : ℚ, 1/2 < m → m < 50/99 + ε → no_lattice_points m) :=
sorry

end NUMINAMATH_CALUDE_max_slope_no_lattice_points_l24_2416


namespace NUMINAMATH_CALUDE_fraction_ordering_l24_2491

theorem fraction_ordering : 16/13 < 21/17 ∧ 21/17 < 20/15 := by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l24_2491


namespace NUMINAMATH_CALUDE_gcd_powers_of_two_minus_one_problem_4_l24_2436

theorem gcd_powers_of_two_minus_one (a b : Nat) :
  Nat.gcd (2^a - 1) (2^b - 1) = 2^(Nat.gcd a b) - 1 := by sorry

theorem problem_4 : Nat.gcd (2^6 - 1) (2^9 - 1) = 7 := by sorry

end NUMINAMATH_CALUDE_gcd_powers_of_two_minus_one_problem_4_l24_2436


namespace NUMINAMATH_CALUDE_powers_of_i_sum_l24_2409

def i : ℂ := Complex.I

theorem powers_of_i_sum (h1 : i^2 = -1) (h2 : i^4 = 1) :
  i^14 + i^19 + i^24 + i^29 + i^34 + i^39 = -1 - i :=
by sorry

end NUMINAMATH_CALUDE_powers_of_i_sum_l24_2409


namespace NUMINAMATH_CALUDE_parabola_focus_vertex_distance_l24_2493

theorem parabola_focus_vertex_distance (p : ℝ) (h_p : p > 0) : 
  let C : Set (ℝ × ℝ) := {(x, y) | y^2 = 2*p*x}
  let F : ℝ × ℝ := (p/2, 0)
  let l : Set (ℝ × ℝ) := {(x, y) | y = x - p/2}
  let chord_length : ℝ := 4
  let angle_with_axis : ℝ := π/4
  (∀ (x y : ℝ), (x, y) ∈ l → (x - F.1)^2 + (y - F.2)^2 = (x + F.1)^2 + (y - F.2)^2) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ C ∩ l ∧ (x₂, y₂) ∈ C ∩ l ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2) →
  (∀ (x y : ℝ), (x, y) ∈ l → y / x = Real.tan angle_with_axis) →
  F.1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_vertex_distance_l24_2493


namespace NUMINAMATH_CALUDE_numbers_with_2019_divisors_l24_2498

def has_2019_divisors (n : ℕ) : Prop :=
  (Finset.card (Nat.divisors n) = 2019)

theorem numbers_with_2019_divisors :
  {n : ℕ | n < 128^97 ∧ has_2019_divisors n} =
  {2^672 * 3^2, 2^672 * 5^2, 2^672 * 7^2, 2^672 * 11^2} :=
by sorry

end NUMINAMATH_CALUDE_numbers_with_2019_divisors_l24_2498


namespace NUMINAMATH_CALUDE_man_mass_from_boat_displacement_l24_2460

/-- Calculates the mass of a man based on the displacement of a boat -/
theorem man_mass_from_boat_displacement (boat_length boat_breadth boat_sink_height water_density : Real) 
  (h1 : boat_length = 3)
  (h2 : boat_breadth = 2)
  (h3 : boat_sink_height = 0.01)
  (h4 : water_density = 1000) : 
  boat_length * boat_breadth * boat_sink_height * water_density = 60 := by
  sorry

#check man_mass_from_boat_displacement

end NUMINAMATH_CALUDE_man_mass_from_boat_displacement_l24_2460


namespace NUMINAMATH_CALUDE_john_running_days_l24_2463

/-- The number of days John ran before getting injured -/
def days_ran (daily_distance : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / daily_distance

theorem john_running_days :
  days_ran 1700 10200 = 6 :=
by sorry

end NUMINAMATH_CALUDE_john_running_days_l24_2463


namespace NUMINAMATH_CALUDE_regression_line_equation_l24_2431

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear equation in the form y = mx + b -/
structure LinearEquation where
  slope : ℝ
  intercept : ℝ

/-- Check if a point lies on a given linear equation -/
def pointOnLine (p : Point) (eq : LinearEquation) : Prop :=
  p.y = eq.slope * p.x + eq.intercept

/-- The theorem to be proved -/
theorem regression_line_equation 
  (slope : ℝ) 
  (center : Point) 
  (h_slope : slope = 1.23)
  (h_center : center = ⟨4, 5⟩) :
  ∃ (eq : LinearEquation), 
    eq.slope = slope ∧ 
    pointOnLine center eq ∧ 
    eq = ⟨1.23, 0.08⟩ := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l24_2431


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l24_2412

theorem rectangle_circle_area_ratio (w l r : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 2 * Real.pi * r) :
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l24_2412


namespace NUMINAMATH_CALUDE_periodic_function_value_l24_2413

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β), if f(3) = 3, then f(2016) = -3 -/
theorem periodic_function_value (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)
  f 3 = 3 → f 2016 = -3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l24_2413


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l24_2471

/-- Given an original price p and two consecutive 10% discounts,
    the final selling price is 0.81p -/
theorem final_price_after_discounts (p : ℝ) : 
  let discount := 0.1
  let first_discount := p * (1 - discount)
  let second_discount := first_discount * (1 - discount)
  second_discount = 0.81 * p := by
sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_l24_2471


namespace NUMINAMATH_CALUDE_max_value_of_symmetric_f_l24_2473

/-- A function f that is symmetric about x = -2 and has the form (1-x^2)(x^2+ax+b) -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- The symmetry condition for f about x = -2 -/
def symmetric_about_neg_two (a b : ℝ) : Prop :=
  ∀ t, f a b (-2 + t) = f a b (-2 - t)

/-- The theorem stating that if f is symmetric about x = -2, its maximum value is 16 -/
theorem max_value_of_symmetric_f (a b : ℝ) 
  (h : symmetric_about_neg_two a b) : 
  ∃ x, f a b x = 16 ∧ ∀ y, f a b y ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_symmetric_f_l24_2473


namespace NUMINAMATH_CALUDE_complex_equation_solution_l24_2411

theorem complex_equation_solution (a : ℝ) :
  (2 + a * Complex.I) / (1 + Complex.I) = -2 * Complex.I → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l24_2411


namespace NUMINAMATH_CALUDE_self_common_tangents_l24_2466

-- Define the concept of a self-common tangent
def has_self_common_tangent (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₁ x₂ y m b : ℝ), x₁ ≠ x₂ ∧ 
    f x₁ y ∧ f x₂ y ∧
    (∀ x y, f x y → y = m * x + b)

-- Define the four curves
def curve1 (x y : ℝ) : Prop := x^2 - y^2 = 1
def curve2 (x y : ℝ) : Prop := y = x^2 - abs x
def curve3 (x y : ℝ) : Prop := y = 3 * Real.sin x + 4 * Real.cos x
def curve4 (x y : ℝ) : Prop := abs x + 1 = Real.sqrt (4 - y^2)

-- Theorem statement
theorem self_common_tangents :
  has_self_common_tangent curve2 ∧ 
  has_self_common_tangent curve3 ∧
  ¬has_self_common_tangent curve1 ∧
  ¬has_self_common_tangent curve4 :=
sorry

end NUMINAMATH_CALUDE_self_common_tangents_l24_2466


namespace NUMINAMATH_CALUDE_complex_in_first_quadrant_l24_2489

theorem complex_in_first_quadrant (a : ℝ) : 
  (((1 : ℂ) + a * Complex.I) / ((2 : ℂ) - Complex.I)).re > 0 ∧
  (((1 : ℂ) + a * Complex.I) / ((2 : ℂ) - Complex.I)).im > 0 →
  -1/2 < a ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_complex_in_first_quadrant_l24_2489


namespace NUMINAMATH_CALUDE_project_completion_time_l24_2441

theorem project_completion_time 
  (initial_team : ℕ) 
  (initial_work : ℚ) 
  (initial_time : ℕ) 
  (additional_members : ℕ) 
  (total_team : ℕ) :
  initial_team = 8 →
  initial_work = 1/3 →
  initial_time = 30 →
  additional_members = 4 →
  total_team = initial_team + additional_members →
  let work_efficiency := initial_work / (initial_team * initial_time)
  let remaining_work := 1 - initial_work
  let remaining_time := remaining_work / (total_team * work_efficiency)
  initial_time + remaining_time = 70 := by
sorry

end NUMINAMATH_CALUDE_project_completion_time_l24_2441


namespace NUMINAMATH_CALUDE_fraction_simplification_l24_2417

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l24_2417


namespace NUMINAMATH_CALUDE_inverse_matrices_values_l24_2462

def Matrix1 (a : ℚ) : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![a, 2],
    ![1, 4]]

def Matrix2 (b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![-2/7, 1/7],
    ![b, 3/14]]

theorem inverse_matrices_values (a b : ℚ) : 
  Matrix1 a * Matrix2 b = 1 → a = -3 ∧ b = 1/14 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_values_l24_2462


namespace NUMINAMATH_CALUDE_total_short_trees_after_planting_l24_2415

/-- Represents the types of trees in the park -/
inductive TreeType
  | Oak
  | Maple
  | Pine

/-- Represents the current state of trees in the park -/
structure ParkTrees where
  shortOak : ℕ
  shortMaple : ℕ
  shortPine : ℕ
  tallOak : ℕ
  tallMaple : ℕ
  tallPine : ℕ

/-- Calculates the new number of short trees after planting -/
def newShortTrees (park : ParkTrees) : ℕ :=
  let newOak := park.shortOak + 57
  let newMaple := park.shortMaple + (park.shortMaple * 3 / 10)  -- 30% increase
  let newPine := park.shortPine + (park.shortPine / 3)  -- 1/3 increase
  newOak + newMaple + newPine

/-- Theorem stating that the total number of short trees after planting is 153 -/
theorem total_short_trees_after_planting (park : ParkTrees) 
  (h1 : park.shortOak = 41)
  (h2 : park.shortMaple = 18)
  (h3 : park.shortPine = 24)
  (h4 : park.tallOak = 44)
  (h5 : park.tallMaple = 37)
  (h6 : park.tallPine = 17) :
  newShortTrees park = 153 := by
  sorry

end NUMINAMATH_CALUDE_total_short_trees_after_planting_l24_2415


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l24_2403

def A : Set ℝ := {x | x^2 - 3*x - 4 > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 5}

theorem intersection_of_A_and_B :
  A ∩ B = {x | (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l24_2403
