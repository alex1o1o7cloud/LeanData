import Mathlib

namespace NUMINAMATH_CALUDE_cos_angle_relation_l2035_203539

theorem cos_angle_relation (α : Real) (h : Real.cos (75 * Real.pi / 180 + α) = 1/2) :
  Real.cos (105 * Real.pi / 180 - α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_relation_l2035_203539


namespace NUMINAMATH_CALUDE_binary_1011001_to_base5_l2035_203591

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_1011001_to_base5 :
  decimal_to_base5 (binary_to_decimal [true, false, false, true, true, false, true]) = [3, 2, 4] :=
sorry

end NUMINAMATH_CALUDE_binary_1011001_to_base5_l2035_203591


namespace NUMINAMATH_CALUDE_physics_class_size_l2035_203580

theorem physics_class_size 
  (total_students : ℕ) 
  (math_only : ℕ) 
  (physics_only : ℕ) 
  (both : ℕ) :
  total_students = 53 →
  both = 7 →
  physics_only + both = 2 * (math_only + both) →
  total_students = math_only + physics_only + both →
  physics_only + both = 40 := by
  sorry

end NUMINAMATH_CALUDE_physics_class_size_l2035_203580


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2035_203597

/-- The x-intercept of the line 2x + 3y = 6 is 3 -/
theorem x_intercept_of_line (x y : ℝ) : 2 * x + 3 * y = 6 → y = 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2035_203597


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2035_203545

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence a_n where a_3 + a_5 = 20 and a_4 = 8, prove that a_2 + a_6 = 34 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geo : IsGeometricSequence a)
    (h_sum : a 3 + a 5 = 20) (h_fourth : a 4 = 8) : a 2 + a 6 = 34 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2035_203545


namespace NUMINAMATH_CALUDE_multiples_of_15_between_16_and_181_l2035_203568

theorem multiples_of_15_between_16_and_181 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ 16 < n ∧ n < 181) (Finset.range 181)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_16_and_181_l2035_203568


namespace NUMINAMATH_CALUDE_sum_in_first_quadrant_l2035_203518

/-- Given complex numbers z₁ and z₂, prove that their sum is in the first quadrant -/
theorem sum_in_first_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 1 + 2*I) (h₂ : z₂ = 1 - I) : 
  let z := z₁ + z₂
  (z.re > 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_sum_in_first_quadrant_l2035_203518


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l2035_203516

/-- A parallelogram in 2D space --/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Check if a given point is a valid fourth vertex of a parallelogram --/
def isValidFourthVertex (p : Parallelogram) (point : ℝ × ℝ) : Prop :=
  point = (11, 4) ∨ point = (-1, 12) ∨ point = (3, -12)

/-- The main theorem --/
theorem parallelogram_fourth_vertex 
  (p : Parallelogram) 
  (h1 : p.v1 = (1, 0)) 
  (h2 : p.v2 = (5, 8)) 
  (h3 : p.v3 = (7, -4)) : 
  isValidFourthVertex p p.v4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l2035_203516


namespace NUMINAMATH_CALUDE_diophantine_equation_equivalence_l2035_203547

/-- Given non-square integers a and b, the existence of a non-trivial integer solution
    to x^2 - ay^2 - bz^2 + abw^2 = 0 is equivalent to the existence of a non-trivial
    integer solution to x^2 - ay^2 - bz^2 = 0 -/
theorem diophantine_equation_equivalence (a b : ℤ) 
  (ha : ¬ ∃ (n : ℤ), n^2 = a) (hb : ¬ ∃ (n : ℤ), n^2 = b) :
  (∃ (x y z w : ℤ), x^2 - a*y^2 - b*z^2 + a*b*w^2 = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ w ≠ 0)) ↔
  (∃ (x y z : ℤ), x^2 - a*y^2 - b*z^2 = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry


end NUMINAMATH_CALUDE_diophantine_equation_equivalence_l2035_203547


namespace NUMINAMATH_CALUDE_ratio_equality_sometimes_l2035_203593

/-- An isosceles triangle with side lengths A and base B -/
structure IsoscelesTriangle where
  A : ℝ
  B : ℝ
  h : ℝ  -- height
  K₁ : ℝ  -- area
  β : ℝ  -- base angle
  h_eq : h = Real.sqrt (A^2 - (B/2)^2)
  K₁_eq : K₁ = (1/2) * B * h
  B_ne_A : B ≠ A

/-- An equilateral triangle with side length a -/
structure EquilateralTriangle where
  a : ℝ
  p : ℝ  -- perimeter
  k₁ : ℝ  -- area
  α : ℝ  -- angle
  p_eq : p = 3 * a
  k₁_eq : k₁ = (a^2 * Real.sqrt 3) / 4
  α_eq : α = π / 3

/-- The main theorem stating that the ratio equality holds sometimes but not always -/
theorem ratio_equality_sometimes (iso : IsoscelesTriangle) (equi : EquilateralTriangle)
    (h_eq : iso.A = equi.a) :
    ∃ (iso₁ : IsoscelesTriangle) (equi₁ : EquilateralTriangle),
      iso₁.h / equi₁.p = iso₁.K₁ / equi₁.k₁ ∧
    ∃ (iso₂ : IsoscelesTriangle) (equi₂ : EquilateralTriangle),
      iso₂.h / equi₂.p ≠ iso₂.K₁ / equi₂.k₁ := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_sometimes_l2035_203593


namespace NUMINAMATH_CALUDE_smallest_multiple_ending_in_three_l2035_203569

theorem smallest_multiple_ending_in_three : 
  ∀ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 5 = 0 → n ≥ 53 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_ending_in_three_l2035_203569


namespace NUMINAMATH_CALUDE_prime_pair_sum_is_106_l2035_203533

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def prime_pair_sum : ℕ → Prop
| S => ∃ (primes : Finset ℕ), 
    (∀ p ∈ primes, is_prime p ∧ is_prime (p + 2) ∧ p * (p + 2) ≤ 2007) ∧
    (∀ p : ℕ, is_prime p → is_prime (p + 2) → p * (p + 2) ≤ 2007 → p ∈ primes) ∧
    (Finset.sum primes id = S)

theorem prime_pair_sum_is_106 : prime_pair_sum 106 := by sorry

end NUMINAMATH_CALUDE_prime_pair_sum_is_106_l2035_203533


namespace NUMINAMATH_CALUDE_solutions_are_correct_l2035_203561

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x^2 - 5 * x - 2 = 0
def equation2 (x : ℝ) : Prop := x^2 - 1 = 2 * (x + 1)
def equation3 (x : ℝ) : Prop := 4 * x^2 + 4 * x + 1 = 3 * (3 - x)^2
def equation4 (x : ℝ) : Prop := (2 * x + 8) * (x - 2) = x^2 + 2 * x - 17

-- Theorem stating the solutions are correct
theorem solutions_are_correct :
  (equation1 (-1/3) ∧ equation1 2) ∧
  (equation2 (-1) ∧ equation2 3) ∧
  (equation3 (-11 + 7 * Real.sqrt 3) ∧ equation3 (-11 - 7 * Real.sqrt 3)) ∧
  (equation4 (-1)) := by
  sorry

#check solutions_are_correct

end NUMINAMATH_CALUDE_solutions_are_correct_l2035_203561


namespace NUMINAMATH_CALUDE_power_of_eleven_l2035_203511

/-- Given an expression (11)^n * (4)^11 * (7)^5 where the total number of prime factors is 29,
    prove that the value of n (the power of 11) is 2. -/
theorem power_of_eleven (n : ℕ) : 
  (n + 22 + 5 = 29) → n = 2 := by
sorry

end NUMINAMATH_CALUDE_power_of_eleven_l2035_203511


namespace NUMINAMATH_CALUDE_z₂_value_l2035_203588

-- Define the complex numbers
variable (z₁ z₂ : ℂ)

-- Define the conditions
axiom h₁ : (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I
axiom h₂ : z₂.im = 2
axiom h₃ : (z₁ * z₂).im = 0

-- Theorem statement
theorem z₂_value : z₂ = 4 + 2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_z₂_value_l2035_203588


namespace NUMINAMATH_CALUDE_pascal_triangle_43_numbers_l2035_203542

/-- The number of elements in a row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

/-- The second number in a row of Pascal's triangle -/
def pascal_second_number (n : ℕ) : ℕ := n

theorem pascal_triangle_43_numbers :
  ∃ n : ℕ, pascal_row_length n = 43 ∧ pascal_second_number n = 42 :=
sorry

end NUMINAMATH_CALUDE_pascal_triangle_43_numbers_l2035_203542


namespace NUMINAMATH_CALUDE_total_value_is_five_dollars_l2035_203525

/-- Represents the value of different coin types in dollars -/
def coin_value : Fin 4 → ℚ
  | 0 => 0.25  -- Quarter
  | 1 => 0.10  -- Dime
  | 2 => 0.05  -- Nickel
  | 3 => 0.01  -- Penny

/-- Represents the count of each coin type -/
def coin_count : Fin 4 → ℕ
  | 0 => 10   -- Quarters
  | 1 => 3    -- Dimes
  | 2 => 4    -- Nickels
  | 3 => 200  -- Pennies

/-- Calculates the total value of coins -/
def total_value : ℚ :=
  (Finset.sum Finset.univ (λ i => coin_value i * coin_count i))

/-- Theorem stating that the total value of coins is $5.00 -/
theorem total_value_is_five_dollars : total_value = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_value_is_five_dollars_l2035_203525


namespace NUMINAMATH_CALUDE_tan_alpha_neg_half_implies_expression_eq_neg_third_l2035_203500

theorem tan_alpha_neg_half_implies_expression_eq_neg_third (α : Real) 
  (h : Real.tan α = -1/2) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_neg_half_implies_expression_eq_neg_third_l2035_203500


namespace NUMINAMATH_CALUDE_group_collection_l2035_203564

/-- Calculates the total amount collected in rupees given the number of students in a group,
    where each student contributes as many paise as there are members. -/
def totalCollected (numStudents : ℕ) : ℚ :=
  (numStudents * numStudents : ℚ) / 100

/-- Theorem stating that for a group of 96 students, the total amount collected is 92.16 rupees. -/
theorem group_collection :
  totalCollected 96 = 92.16 := by
  sorry

end NUMINAMATH_CALUDE_group_collection_l2035_203564


namespace NUMINAMATH_CALUDE_lock_combinations_count_l2035_203562

/-- The number of digits on the lock -/
def n : ℕ := 4

/-- The number of possible digits (0 to 9) -/
def k : ℕ := 10

/-- The number of ways to select n digits from k possibilities in non-decreasing order -/
def lockCombinations : ℕ := (n + k - 1).choose (k - 1)

theorem lock_combinations_count : lockCombinations = 715 := by
  sorry

end NUMINAMATH_CALUDE_lock_combinations_count_l2035_203562


namespace NUMINAMATH_CALUDE_max_value_constraint_l2035_203557

theorem max_value_constraint (x y : ℝ) (h : 5 * x^2 + 4 * y^2 = 10 * x) :
  x^2 + y^2 ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), 5 * x₀^2 + 4 * y₀^2 = 10 * x₀ ∧ x₀^2 + y₀^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2035_203557


namespace NUMINAMATH_CALUDE_next_coincidence_l2035_203503

def factory_interval : ℕ := 18
def fire_station_interval : ℕ := 24
def town_hall_interval : ℕ := 30

theorem next_coincidence (start_time : ℕ) :
  ∃ (t : ℕ), t > start_time ∧ 
  t % factory_interval = 0 ∧
  t % fire_station_interval = 0 ∧
  t % town_hall_interval = 0 ∧
  t - start_time = 360 := by
sorry

end NUMINAMATH_CALUDE_next_coincidence_l2035_203503


namespace NUMINAMATH_CALUDE_floor_of_4_7_l2035_203575

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l2035_203575


namespace NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l2035_203576

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on other axles. -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

/-- Calculates the toll for a truck based on the number of axles. -/
def calculateToll (axles : ℕ) : ℚ :=
  1.5 + 1.5 * (axles - 2 : ℚ)

/-- Theorem stating that the toll for an 18-wheel truck with 2 wheels on the front axle
    and 4 wheels on each other axle is $6.00. -/
theorem eighteen_wheel_truck_toll :
  let axles := calculateAxles 18 2 4
  calculateToll axles = 6 := by
  sorry

#eval calculateAxles 18 2 4
#eval calculateToll (calculateAxles 18 2 4)

end NUMINAMATH_CALUDE_eighteen_wheel_truck_toll_l2035_203576


namespace NUMINAMATH_CALUDE_tracy_candies_l2035_203595

theorem tracy_candies (x : ℕ) (h1 : x % 4 = 0) 
  (h2 : x / 2 % 2 = 0) 
  (h3 : 4 ≤ x / 2 - 20) (h4 : x / 2 - 20 ≤ 8) 
  (h5 : ∃ (b : ℕ), 2 ≤ b ∧ b ≤ 6 ∧ x / 2 - 20 - b = 4) : x = 48 := by
  sorry

end NUMINAMATH_CALUDE_tracy_candies_l2035_203595


namespace NUMINAMATH_CALUDE_result_calculation_l2035_203504

/-- Definition of x as the solution to x = 2 + (√3 / (2 + (√3 / (2 + ...)))) -/
noncomputable def x : ℝ := 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / 2))))))

/-- Theorem stating the result of the calculation -/
theorem result_calculation : 1 / ((x + 2) * (x - 3)) = (5 + Real.sqrt 3) / -22 := by
  sorry

end NUMINAMATH_CALUDE_result_calculation_l2035_203504


namespace NUMINAMATH_CALUDE_winning_percentage_correct_l2035_203587

/-- Represents the percentage of votes secured by the winning candidate -/
def winning_percentage : ℝ := 70

/-- Represents the total number of valid votes -/
def total_votes : ℕ := 455

/-- Represents the majority of votes by which the winning candidate won -/
def vote_majority : ℕ := 182

/-- Theorem stating that the winning percentage is correct given the conditions -/
theorem winning_percentage_correct : 
  (winning_percentage / 100 * total_votes : ℝ) - 
  ((100 - winning_percentage) / 100 * total_votes : ℝ) = vote_majority := by
  sorry

end NUMINAMATH_CALUDE_winning_percentage_correct_l2035_203587


namespace NUMINAMATH_CALUDE_solve_system_for_y_l2035_203573

theorem solve_system_for_y (x y : ℚ) 
  (eq1 : 2 * x - y = 10) 
  (eq2 : x + 3 * y = 2) : 
  y = -6/7 := by sorry

end NUMINAMATH_CALUDE_solve_system_for_y_l2035_203573


namespace NUMINAMATH_CALUDE_problem_statement_l2035_203507

theorem problem_statement (a b : ℝ) (h : a + b = 1) :
  (a^3 + b^3 ≥ 1/4) ∧
  (∃ x : ℝ, |x - a| + |x - b| ≤ 5 → 0 ≤ 2*a + 3*b ∧ 2*a + 3*b ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2035_203507


namespace NUMINAMATH_CALUDE_derivative_at_pi_over_four_l2035_203509

theorem derivative_at_pi_over_four (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x * (Real.cos x + 1)) :
  deriv f (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_over_four_l2035_203509


namespace NUMINAMATH_CALUDE_monkey_doll_difference_l2035_203578

theorem monkey_doll_difference (total_budget : ℕ) (large_doll_cost : ℕ) (cost_difference : ℕ) : 
  total_budget = 300 → 
  large_doll_cost = 6 → 
  cost_difference = 2 → 
  (total_budget / (large_doll_cost - cost_difference) : ℕ) - (total_budget / large_doll_cost : ℕ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_monkey_doll_difference_l2035_203578


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2035_203538

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  parallel a b → m = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2035_203538


namespace NUMINAMATH_CALUDE_product_simplification_l2035_203514

theorem product_simplification (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((2*x + 2*y + 2*z)⁻¹) * (x⁻¹ + y⁻¹ + z⁻¹) * ((x*y + y*z + x*z)⁻¹) * 
  (2*(x*y)⁻¹ + 2*(y*z)⁻¹ + 2*(x*z)⁻¹) = (x^2 * y^2 * z^2)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_product_simplification_l2035_203514


namespace NUMINAMATH_CALUDE_bridge_length_l2035_203540

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 →
  train_speed_kmh = 42.3 →
  crossing_time = 40 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 320 := by
  sorry

#check bridge_length

end NUMINAMATH_CALUDE_bridge_length_l2035_203540


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2035_203521

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (0 ≤ x ∧ x < 1) ↔ (x + 2*a ≥ 4 ∧ (2*x - b) / 3 < 1)) →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2035_203521


namespace NUMINAMATH_CALUDE_range_of_k_l2035_203546

-- Define the condition function
def condition (x : ℝ) : Prop := 3 / (x + 1) < 1

-- Define the sufficient condition
def sufficient_condition (k : ℝ) : Prop := ∀ x, x > k → condition x

-- Define the not necessary condition
def not_necessary_condition (k : ℝ) : Prop := ∃ x, condition x ∧ x ≤ k

-- State the theorem
theorem range_of_k :
  ∀ k, (sufficient_condition k ∧ not_necessary_condition k) ↔ k ∈ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l2035_203546


namespace NUMINAMATH_CALUDE_rosalina_gifts_l2035_203532

theorem rosalina_gifts (emilio jorge pedro : ℕ) 
  (h1 : emilio = 11) 
  (h2 : jorge = 6) 
  (h3 : pedro = 4) : 
  emilio + jorge + pedro = 21 := by
  sorry

end NUMINAMATH_CALUDE_rosalina_gifts_l2035_203532


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2035_203552

theorem arithmetic_sequence_formula (x : ℝ) (a : ℕ → ℝ) :
  (a 1 = x - 1) →
  (a 2 = x + 1) →
  (a 3 = 2*x + 3) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = a 2 - a 1) →
  (∀ n : ℕ, n ≥ 1 → a n = 2*n - 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2035_203552


namespace NUMINAMATH_CALUDE_lilies_per_centerpiece_l2035_203536

/-- Proves that the number of lilies per centerpiece is 6 given the specified conditions -/
theorem lilies_per_centerpiece
  (num_centerpieces : ℕ)
  (roses_per_centerpiece : ℕ)
  (orchids_per_centerpiece : ℕ)
  (total_budget : ℚ)
  (flower_cost : ℚ)
  (h1 : num_centerpieces = 6)
  (h2 : roses_per_centerpiece = 8)
  (h3 : orchids_per_centerpiece = 2 * roses_per_centerpiece)
  (h4 : total_budget = 2700)
  (h5 : flower_cost = 15)
  : (total_budget / flower_cost / num_centerpieces : ℚ) - roses_per_centerpiece - orchids_per_centerpiece = 6 :=
sorry

end NUMINAMATH_CALUDE_lilies_per_centerpiece_l2035_203536


namespace NUMINAMATH_CALUDE_exists_function_double_composition_l2035_203549

theorem exists_function_double_composition :
  ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f (f n) = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_exists_function_double_composition_l2035_203549


namespace NUMINAMATH_CALUDE_water_difference_proof_l2035_203579

/-- The difference in initial water amounts between Ji-hoon and Hyo-joo, given the conditions of the problem -/
def water_difference (j h : ℕ) : Prop :=
  (j - 152 = h + 152 + 346) → (j - h = 650)

/-- Theorem stating the water difference problem -/
theorem water_difference_proof :
  ∀ j h : ℕ, water_difference j h :=
by
  sorry

end NUMINAMATH_CALUDE_water_difference_proof_l2035_203579


namespace NUMINAMATH_CALUDE_factorization_equality_l2035_203501

theorem factorization_equality (a b : ℝ) : a * b^2 - a = a * (b + 1) * (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2035_203501


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l2035_203581

/-- Given a polynomial function g(x) = px^4 + qx^3 + rx^2 + sx + t,
    if g(-3) = 9, then 16p - 8q + 4r - 2s + t = -9 -/
theorem polynomial_value_theorem (p q r s t : ℝ) :
  let g : ℝ → ℝ := λ x => p * x^4 + q * x^3 + r * x^2 + s * x + t
  g (-3) = 9 → 16 * p - 8 * q + 4 * r - 2 * s + t = -9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l2035_203581


namespace NUMINAMATH_CALUDE_max_first_term_arithmetic_progression_l2035_203548

def arithmetic_progression (a₁ : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n+1 => arithmetic_progression a₁ d n + d

def sum_arithmetic_progression (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem max_first_term_arithmetic_progression 
  (a₁ : ℚ) (d : ℚ) 
  (h₁ : ∃ (n : ℕ), sum_arithmetic_progression a₁ d 4 = n)
  (h₂ : ∃ (m : ℕ), sum_arithmetic_progression a₁ d 7 = m)
  (h₃ : a₁ ≤ 2/3) :
  a₁ ≤ 9/14 :=
sorry

end NUMINAMATH_CALUDE_max_first_term_arithmetic_progression_l2035_203548


namespace NUMINAMATH_CALUDE_greatest_multiple_of_five_cubed_less_than_8000_l2035_203567

theorem greatest_multiple_of_five_cubed_less_than_8000 :
  ∃ (y : ℕ), y > 0 ∧ 5 ∣ y ∧ y^3 < 8000 ∧ ∀ (z : ℕ), z > 0 → 5 ∣ z → z^3 < 8000 → z ≤ y :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_five_cubed_less_than_8000_l2035_203567


namespace NUMINAMATH_CALUDE_book_weight_l2035_203556

theorem book_weight (num_books : ℕ) (total_weight : ℝ) (bag_weight : ℝ) :
  num_books = 14 →
  total_weight = 11.14 →
  bag_weight = 0.5 →
  (total_weight - bag_weight) / num_books = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_book_weight_l2035_203556


namespace NUMINAMATH_CALUDE_savings_interest_rate_equation_l2035_203528

/-- Represents the annual interest rate calculation for a savings account --/
theorem savings_interest_rate_equation 
  (initial_amount : ℝ) 
  (final_amount : ℝ) 
  (years : ℕ) 
  (interest_rate : ℝ) 
  (h1 : initial_amount = 3000) 
  (h2 : final_amount = 3243) 
  (h3 : years = 3) :
  final_amount = initial_amount + initial_amount * years * (interest_rate / 100) :=
by sorry

end NUMINAMATH_CALUDE_savings_interest_rate_equation_l2035_203528


namespace NUMINAMATH_CALUDE_product_of_nonneg_quadratics_is_nonneg_l2035_203582

/-- Given two non-negative quadratic functions, their product is also non-negative. -/
theorem product_of_nonneg_quadratics_is_nonneg
  (a b c A B C : ℝ)
  (h1 : ∀ x : ℝ, a * x^2 + 2 * b * x + c ≥ 0)
  (h2 : ∀ x : ℝ, A * x^2 + 2 * B * x + C ≥ 0) :
  ∀ x : ℝ, a * A * x^2 + 2 * b * B * x + c * C ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_product_of_nonneg_quadratics_is_nonneg_l2035_203582


namespace NUMINAMATH_CALUDE_next_red_probability_l2035_203517

/-- Represents the count of balls of each color -/
structure BallCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Represents the result of pulling out balls -/
structure PullResult where
  pulled : ℕ
  redBlueDifference : ℤ

/-- Calculates the probability of pulling a red ball next -/
def probabilityNextRed (initial : BallCount) (result : PullResult) : ℚ :=
  9/26

theorem next_red_probability 
  (initial : BallCount)
  (result : PullResult)
  (h1 : initial.red = 50)
  (h2 : initial.blue = 50)
  (h3 : initial.yellow = 30)
  (h4 : result.pulled = 65)
  (h5 : result.redBlueDifference = 5) :
  probabilityNextRed initial result = 9/26 := by
  sorry

end NUMINAMATH_CALUDE_next_red_probability_l2035_203517


namespace NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_ten_l2035_203541

theorem product_seven_consecutive_divisible_by_ten (n : ℕ) : 
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) :=
by sorry

end NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_ten_l2035_203541


namespace NUMINAMATH_CALUDE_expression_evaluation_l2035_203526

theorem expression_evaluation :
  let a : ℤ := 1001
  let b : ℤ := 1002
  let c : ℤ := 1000
  b^3 - a*b^2 - a^2*b + a^3 - c^3 = 2009007 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2035_203526


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l2035_203558

theorem greatest_integer_with_gcd_six : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 → Nat.gcd m 18 = 6 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l2035_203558


namespace NUMINAMATH_CALUDE_decagon_triangles_l2035_203596

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- Theorem stating that the number of triangles formed by the vertices of a regular decagon is 120 -/
theorem decagon_triangles : trianglesInDecagon = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l2035_203596


namespace NUMINAMATH_CALUDE_hyperbola_distance_inequality_l2035_203599

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the left focus
def left_focus : ℝ × ℝ := sorry

-- Define a point on the right branch of the hyperbola
def right_branch_point (P : ℝ × ℝ) : Prop := 
  hyperbola P.1 P.2 ∧ P.1 > 0

-- State the theorem
theorem hyperbola_distance_inequality 
  (P₁ P₂ : ℝ × ℝ) 
  (h₁ : right_branch_point P₁) 
  (h₂ : right_branch_point P₂) : 
  dist left_focus P₁ + dist left_focus P₂ - dist P₁ P₂ ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_distance_inequality_l2035_203599


namespace NUMINAMATH_CALUDE_minimize_sum_with_constraint_l2035_203570

theorem minimize_sum_with_constraint :
  ∀ a b : ℕ+,
  (4 * a.val + b.val = 30) →
  (∀ x y : ℕ+, (4 * x.val + y.val = 30) → (a.val + b.val ≤ x.val + y.val)) →
  (a.val = 7 ∧ b.val = 2) :=
by sorry

end NUMINAMATH_CALUDE_minimize_sum_with_constraint_l2035_203570


namespace NUMINAMATH_CALUDE_cube_side_ratio_l2035_203555

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 49 / 1 → a / b = 7 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l2035_203555


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2035_203554

theorem smallest_number_with_given_remainders :
  ∃! x : ℕ,
    x > 0 ∧
    x % 5 = 2 ∧
    x % 4 = 2 ∧
    x % 6 = 3 ∧
    ∀ y : ℕ, y > 0 → y % 5 = 2 → y % 4 = 2 → y % 6 = 3 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2035_203554


namespace NUMINAMATH_CALUDE_jessy_jewelry_count_l2035_203506

def initial_necklaces : ℕ := 10
def initial_earrings : ℕ := 15
def bought_necklaces : ℕ := 10
def bought_earrings : ℕ := (2 * initial_earrings) / 3
def mother_gift_earrings : ℕ := bought_earrings / 5 + bought_earrings

def total_jewelry : ℕ := initial_necklaces + initial_earrings + bought_necklaces + bought_earrings + mother_gift_earrings

theorem jessy_jewelry_count : total_jewelry = 57 := by
  sorry

end NUMINAMATH_CALUDE_jessy_jewelry_count_l2035_203506


namespace NUMINAMATH_CALUDE_total_weight_BaF2_is_1051_956_l2035_203523

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 18.998

/-- The number of moles of BaF2 -/
def moles_BaF2 : ℝ := 6

/-- The molecular weight of BaF2 in g/mol -/
def molecular_weight_BaF2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_F

/-- The total weight of BaF2 in grams -/
def total_weight_BaF2 : ℝ := molecular_weight_BaF2 * moles_BaF2

/-- Theorem stating that the total weight of 6 moles of BaF2 is 1051.956 g -/
theorem total_weight_BaF2_is_1051_956 : 
  total_weight_BaF2 = 1051.956 := by sorry

end NUMINAMATH_CALUDE_total_weight_BaF2_is_1051_956_l2035_203523


namespace NUMINAMATH_CALUDE_B_k_closed_form_l2035_203505

/-- B_k(n) is the largest possible number of elements in a 2-separable k-configuration of a set with 2n elements -/
def B_k (k n : ℕ) : ℕ := Nat.choose (2*n) k - 2 * Nat.choose n k

/-- Theorem stating the closed-form expression for B_k(n) -/
theorem B_k_closed_form (k n : ℕ) (h1 : 2 ≤ k) (h2 : k ≤ n) :
  B_k k n = Nat.choose (2*n) k - 2 * Nat.choose n k := by
  sorry

end NUMINAMATH_CALUDE_B_k_closed_form_l2035_203505


namespace NUMINAMATH_CALUDE_circle_center_proof_l2035_203577

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation x² - 8x + y² - 4y = 4, prove that its center is (4, 2) -/
theorem circle_center_proof (eq : CircleEquation) 
    (h1 : eq.a = 1)
    (h2 : eq.b = -8)
    (h3 : eq.c = 1)
    (h4 : eq.d = -4)
    (h5 : eq.e = -4) :
    CircleCenter.mk 4 2 = CircleCenter.mk (-eq.b / (2 * eq.a)) (-eq.d / (2 * eq.c)) :=
  sorry

end NUMINAMATH_CALUDE_circle_center_proof_l2035_203577


namespace NUMINAMATH_CALUDE_simplify_radical_sum_l2035_203531

theorem simplify_radical_sum : Real.sqrt 98 + Real.sqrt 32 + (27 : Real).rpow (1/3) = 11 * Real.sqrt 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l2035_203531


namespace NUMINAMATH_CALUDE_dancing_and_math_intersection_l2035_203520

theorem dancing_and_math_intersection (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (a b : ℕ),
    b ≥ 1 ∧
    (a + b)^2 = (p + 1) * a + b ∧
    b = 1 :=
by sorry

end NUMINAMATH_CALUDE_dancing_and_math_intersection_l2035_203520


namespace NUMINAMATH_CALUDE_jack_keeps_half_deer_weight_l2035_203592

/-- Given Jack's hunting habits and the amount of deer he keeps, prove that he keeps half of the total deer weight caught each year. -/
theorem jack_keeps_half_deer_weight 
  (hunts_per_month : ℕ) 
  (hunting_season_months : ℕ) 
  (deers_per_hunt : ℕ) 
  (deer_weight : ℕ) 
  (weight_kept : ℕ) 
  (h1 : hunts_per_month = 6)
  (h2 : hunting_season_months = 3)
  (h3 : deers_per_hunt = 2)
  (h4 : deer_weight = 600)
  (h5 : weight_kept = 10800) : 
  weight_kept / (hunts_per_month * hunting_season_months * deers_per_hunt * deer_weight) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_jack_keeps_half_deer_weight_l2035_203592


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l2035_203519

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume
  (a : ℝ) -- base side length
  (γ : ℝ) -- angle between lateral faces
  (h : 0 < a ∧ 0 < γ ∧ γ < π) -- assumptions to ensure validity
  : ∃ V : ℝ, V = (a^3 * Real.sin (γ/2)) / (12 * Real.sqrt (3/4 - Real.sin (γ/2)^2)) :=
sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l2035_203519


namespace NUMINAMATH_CALUDE_sock_cost_l2035_203550

theorem sock_cost (total_cost shoes_cost : ℝ) 
  (h1 : total_cost = 111) 
  (h2 : shoes_cost = 92) : 
  (total_cost - shoes_cost) / 2 = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_sock_cost_l2035_203550


namespace NUMINAMATH_CALUDE_y_coord_at_neg_three_l2035_203594

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  max_value : ℝ
  max_x : ℝ
  point_zero : ℝ
  has_max : max_value = 7
  max_at : max_x = -2
  passes_zero : a * 0^2 + b * 0 + c = point_zero
  passes_zero_value : point_zero = -15

/-- The y-coordinate of a point on the quadratic function -/
def y_coord (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The theorem stating the y-coordinate at x = -3 is 1.5 -/
theorem y_coord_at_neg_three (f : QuadraticFunction) : y_coord f (-3) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_y_coord_at_neg_three_l2035_203594


namespace NUMINAMATH_CALUDE_floor_plus_self_equation_l2035_203563

theorem floor_plus_self_equation (r : ℝ) : ⌊r⌋ + r = 10.3 ↔ r = 5.3 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_equation_l2035_203563


namespace NUMINAMATH_CALUDE_total_time_for_ten_pictures_l2035_203535

/-- The total time spent on drawing and coloring pictures -/
def total_time (num_pictures : ℕ) (draw_time : ℝ) (color_time_reduction : ℝ) : ℝ :=
  let color_time := draw_time * (1 - color_time_reduction)
  num_pictures * (draw_time + color_time)

/-- Theorem: The total time spent on 10 pictures is 34 hours -/
theorem total_time_for_ten_pictures :
  total_time 10 2 0.3 = 34 := by
  sorry

#eval total_time 10 2 0.3

end NUMINAMATH_CALUDE_total_time_for_ten_pictures_l2035_203535


namespace NUMINAMATH_CALUDE_triangle_proof_l2035_203551

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_proof (t : Triangle) (h1 : t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0)
                       (h2 : t.a = 2)
                       (h3 : (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_proof_l2035_203551


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l2035_203510

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : 3 * a + b = a^2 + a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → 3 * x + y = x^2 + x * y → 2 * x + y ≥ 2 * Real.sqrt 2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l2035_203510


namespace NUMINAMATH_CALUDE_square_perimeter_side_length_l2035_203589

theorem square_perimeter_side_length (perimeter : ℝ) (side : ℝ) : 
  perimeter = 8 → side ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_side_length_l2035_203589


namespace NUMINAMATH_CALUDE_cylinder_volume_l2035_203584

/-- Given a sphere and a cylinder with specific properties, prove the volume of the cylinder --/
theorem cylinder_volume (sphere_volume : ℝ) (cylinder_base_diameter : ℝ) :
  sphere_volume = (500 * Real.pi) / 3 →
  cylinder_base_diameter = 8 →
  ∃ (cylinder_volume : ℝ),
    cylinder_volume = 96 * Real.pi ∧
    (∃ (sphere_radius : ℝ) (cylinder_height : ℝ),
      (4 / 3) * Real.pi * sphere_radius ^ 3 = sphere_volume ∧
      cylinder_height ^ 2 = sphere_radius ^ 2 - (cylinder_base_diameter / 2) ^ 2 ∧
      cylinder_volume = Real.pi * (cylinder_base_diameter / 2) ^ 2 * cylinder_height) :=
by sorry


end NUMINAMATH_CALUDE_cylinder_volume_l2035_203584


namespace NUMINAMATH_CALUDE_triangle_bottom_number_l2035_203508

/-- Define the triangle structure -/
def Triangle (n : ℕ) : Type :=
  Fin n → Fin n → ℕ

/-- The first row of the triangle contains numbers from 1 to 2000 -/
def first_row_condition (t : Triangle 2000) : Prop :=
  ∀ i : Fin 2000, t 0 i = i.val + 1

/-- Each subsequent number is the sum of the two numbers immediately above it -/
def sum_condition (t : Triangle 2000) : Prop :=
  ∀ i j : Fin 2000, i > 0 → t i j = t (i-1) j + t (i-1) (j+1)

/-- The theorem to be proved -/
theorem triangle_bottom_number (t : Triangle 2000) 
  (h1 : first_row_condition t) (h2 : sum_condition t) : 
  t 1999 0 = 2^1998 * 2001 := by
  sorry

end NUMINAMATH_CALUDE_triangle_bottom_number_l2035_203508


namespace NUMINAMATH_CALUDE_rodrigos_classroom_chairs_l2035_203565

/-- The number of chairs left in Rodrigo's classroom after Lisa borrows some chairs -/
def chairs_left (red_chairs yellow_chairs blue_chairs borrowed : ℕ) : ℕ :=
  red_chairs + yellow_chairs + blue_chairs - borrowed

/-- Theorem stating the number of chairs left in Rodrigo's classroom -/
theorem rodrigos_classroom_chairs :
  ∀ (red_chairs : ℕ),
  red_chairs = 4 →
  ∀ (yellow_chairs : ℕ),
  yellow_chairs = 2 * red_chairs →
  ∀ (blue_chairs : ℕ),
  blue_chairs = yellow_chairs - 2 →
  chairs_left red_chairs yellow_chairs blue_chairs 3 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_rodrigos_classroom_chairs_l2035_203565


namespace NUMINAMATH_CALUDE_fraction_value_zero_l2035_203566

theorem fraction_value_zero (B A P E H b K p J C O : ℕ) :
  (B ≠ A ∧ B ≠ P ∧ B ≠ E ∧ B ≠ H ∧ B ≠ b ∧ B ≠ K ∧ B ≠ p ∧ B ≠ J ∧ B ≠ C ∧ B ≠ O) ∧
  (A ≠ P ∧ A ≠ E ∧ A ≠ H ∧ A ≠ b ∧ A ≠ K ∧ A ≠ p ∧ A ≠ J ∧ A ≠ C ∧ A ≠ O) ∧
  (P ≠ E ∧ P ≠ H ∧ P ≠ b ∧ P ≠ K ∧ P ≠ p ∧ P ≠ J ∧ P ≠ C ∧ P ≠ O) ∧
  (E ≠ H ∧ E ≠ b ∧ E ≠ K ∧ E ≠ p ∧ E ≠ J ∧ E ≠ C ∧ E ≠ O) ∧
  (H ≠ b ∧ H ≠ K ∧ H ≠ p ∧ H ≠ J ∧ H ≠ C ∧ H ≠ O) ∧
  (b ≠ K ∧ b ≠ p ∧ b ≠ J ∧ b ≠ C ∧ b ≠ O) ∧
  (K ≠ p ∧ K ≠ J ∧ K ≠ C ∧ K ≠ O) ∧
  (p ≠ J ∧ p ≠ C ∧ p ≠ O) ∧
  (J ≠ C ∧ J ≠ O) ∧
  (C ≠ O) ∧
  (B < 10 ∧ A < 10 ∧ P < 10 ∧ E < 10 ∧ H < 10 ∧ b < 10 ∧ K < 10 ∧ p < 10 ∧ J < 10 ∧ C < 10 ∧ O < 10) →
  (B * A * P * E * H * b * E : ℚ) / (K * A * p * J * C * O * H : ℕ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_value_zero_l2035_203566


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2035_203524

theorem complex_fraction_equality : Complex.I / (1 + Complex.I) = (1 / 2 : ℂ) + (1 / 2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2035_203524


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2035_203529

theorem circumscribed_sphere_surface_area (a : ℝ) (h : a = 2 * Real.sqrt 3 / 3) :
  let R := Real.sqrt 3 * a / 2
  4 * Real.pi * R^2 = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2035_203529


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2035_203559

theorem diophantine_equation_solutions : 
  ∀ a b c : ℕ+, 
  (8 * a.val - 5 * b.val)^2 + (3 * b.val - 2 * c.val)^2 + (3 * c.val - 7 * a.val)^2 = 2 ↔ 
  ((a.val = 3 ∧ b.val = 5 ∧ c.val = 7) ∨ (a.val = 12 ∧ b.val = 19 ∧ c.val = 28)) :=
by sorry

#check diophantine_equation_solutions

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2035_203559


namespace NUMINAMATH_CALUDE_max_rectangle_area_l2035_203537

def perimeter : ℝ := 300
def min_length : ℝ := 80
def min_width : ℝ := 40

def rectangle_area (l w : ℝ) : ℝ := l * w

theorem max_rectangle_area :
  ∀ l w : ℝ,
  l ≥ min_length →
  w ≥ min_width →
  2 * l + 2 * w = perimeter →
  rectangle_area l w ≤ 5600 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l2035_203537


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l2035_203544

theorem sequence_gcd_property (a : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) →
  ∀ i : ℕ, a i = i := by
sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l2035_203544


namespace NUMINAMATH_CALUDE_catch_difference_l2035_203553

theorem catch_difference (joe_catches derek_catches tammy_catches : ℕ) : 
  joe_catches = 23 →
  derek_catches = 2 * joe_catches - 4 →
  tammy_catches = 30 →
  tammy_catches > derek_catches / 3 →
  tammy_catches - derek_catches / 3 = 16 := by
sorry

end NUMINAMATH_CALUDE_catch_difference_l2035_203553


namespace NUMINAMATH_CALUDE_equal_distribution_l2035_203560

theorem equal_distribution (total_amount : ℕ) (num_persons : ℕ) (amount_per_person : ℕ) :
  total_amount = 42900 →
  num_persons = 22 →
  amount_per_person = total_amount / num_persons →
  amount_per_person = 1950 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_l2035_203560


namespace NUMINAMATH_CALUDE_workers_per_block_l2035_203513

/-- Proves that given a total budget of $4000, a cost of $4 per gift, and 10 blocks in the company,
    the number of workers in each block is 100. -/
theorem workers_per_block (total_budget : ℕ) (cost_per_gift : ℕ) (num_blocks : ℕ)
  (h1 : total_budget = 4000)
  (h2 : cost_per_gift = 4)
  (h3 : num_blocks = 10) :
  (total_budget / cost_per_gift) / num_blocks = 100 := by
sorry

#eval (4000 / 4) / 10  -- Should output 100

end NUMINAMATH_CALUDE_workers_per_block_l2035_203513


namespace NUMINAMATH_CALUDE_percentage_difference_in_earnings_l2035_203571

def mike_hourly_rate : ℝ := 12
def phil_hourly_rate : ℝ := 6

theorem percentage_difference_in_earnings : 
  (mike_hourly_rate - phil_hourly_rate) / mike_hourly_rate * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_in_earnings_l2035_203571


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2035_203530

/-- In a right-angled triangle with legs a and b, hypotenuse c, and altitude m
    corresponding to the hypotenuse, m + c > a + b -/
theorem right_triangle_inequality (a b c m : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Pythagorean theorem
  (h_altitude : a * b = c * m) -- Area equality
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ m > 0) : m + c > a + b := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_inequality_l2035_203530


namespace NUMINAMATH_CALUDE_jerrys_books_count_l2035_203534

theorem jerrys_books_count :
  let initial_action_figures : ℕ := 3
  let added_action_figures : ℕ := 2
  let total_action_figures := initial_action_figures + added_action_figures
  let books_count := total_action_figures + 2
  books_count = 7 := by sorry

end NUMINAMATH_CALUDE_jerrys_books_count_l2035_203534


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2035_203543

theorem absolute_value_inequality (a b c : ℝ) : 
  |a - c| < |b| → |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2035_203543


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l2035_203586

theorem complex_magnitude_equation (t : ℝ) : 
  (t > 0 ∧ Complex.abs (t + 2 * Complex.I * Real.sqrt 3) * Complex.abs (6 - 4 * Complex.I) = 26) → t = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l2035_203586


namespace NUMINAMATH_CALUDE_circle_radius_from_chord_and_secant_l2035_203527

/-- Given a circle with a chord of length 10 and a secant parallel to the tangent at one end of the chord,
    where the internal segment of the secant is 12 units long, the radius of the circle is 10 units. -/
theorem circle_radius_from_chord_and_secant (C : ℝ → ℝ → Prop) (A B M : ℝ × ℝ) (r : ℝ) :
  (∀ x y, C x y ↔ (x - r)^2 + (y - r)^2 = r^2) →  -- C is a circle with center (r, r) and radius r
  C A.1 A.2 →  -- A is on the circle
  C B.1 B.2 →  -- B is on the circle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 100 →  -- AB is a chord of length 10
  (∃ t : ℝ, C (A.1 + t) (A.2 + t) ∧ (A.1 + t - B.1)^2 + (A.2 + t - B.2)^2 = 36) →  -- Secant parallel to tangent at A
  r = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_chord_and_secant_l2035_203527


namespace NUMINAMATH_CALUDE_oil_tank_explosion_theorem_l2035_203522

/-- The number of bullets available -/
def num_bullets : ℕ := 5

/-- The probability of hitting the target on each shot -/
def hit_probability : ℚ := 2/3

/-- The probability of the oil tank exploding -/
def explosion_probability : ℚ := 232/243

/-- The probability that the number of shots is not less than 4 -/
def shots_ge_4_probability : ℚ := 7/27

/-- Each shot is independent and the probability of hitting each time is 2/3.
    The first hit causes oil to flow out, and the second hit causes an explosion.
    Shooting stops when the oil tank explodes or bullets run out. -/
theorem oil_tank_explosion_theorem :
  (∀ (n : ℕ), n ≤ num_bullets → (hit_probability^n * (1 - hit_probability)^(num_bullets - n) : ℚ) = (2/3)^n * (1/3)^(num_bullets - n)) →
  explosion_probability = 232/243 ∧
  shots_ge_4_probability = 7/27 :=
sorry

end NUMINAMATH_CALUDE_oil_tank_explosion_theorem_l2035_203522


namespace NUMINAMATH_CALUDE_symmetric_points_product_l2035_203583

/-- Given two points A(-2, a) and B(b, -3) symmetric about the y-axis, prove that ab = -6 -/
theorem symmetric_points_product (a b : ℝ) : 
  ((-2 : ℝ) = -b) → (a = -3) → ab = -6 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_product_l2035_203583


namespace NUMINAMATH_CALUDE_raft_minimum_capacity_l2035_203572

/-- Represents an animal with its weight -/
structure Animal where
  weight : ℕ

/-- Represents the raft with its capacity -/
structure Raft where
  capacity : ℕ

/-- Checks if the raft can carry at least two of the lightest animals -/
def canCarryTwoLightest (r : Raft) (animals : List Animal) : Prop :=
  r.capacity ≥ 2 * (animals.map Animal.weight).minimum

/-- Checks if all animals can be transported using the given raft -/
def canTransportAll (r : Raft) (animals : List Animal) : Prop :=
  canCarryTwoLightest r animals

/-- The theorem to be proved -/
theorem raft_minimum_capacity 
  (mice : List Animal) 
  (moles : List Animal) 
  (hamsters : List Animal) 
  (h_mice : mice.length = 5 ∧ ∀ m ∈ mice, m.weight = 70)
  (h_moles : moles.length = 3 ∧ ∀ m ∈ moles, m.weight = 90)
  (h_hamsters : hamsters.length = 4 ∧ ∀ h ∈ hamsters, h.weight = 120)
  : ∃ (r : Raft), r.capacity = 140 ∧ canTransportAll r (mice ++ moles ++ hamsters) :=
sorry

end NUMINAMATH_CALUDE_raft_minimum_capacity_l2035_203572


namespace NUMINAMATH_CALUDE_no_extreme_points_l2035_203590

/-- The function f(x) = x^3 - 3x^2 + 3x has no extreme points. -/
theorem no_extreme_points (x : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^3 - 3*x^2 + 3*x
  (∀ a b, a < b → f a < f b) :=
by
  sorry

end NUMINAMATH_CALUDE_no_extreme_points_l2035_203590


namespace NUMINAMATH_CALUDE_metal_bar_weight_l2035_203502

/-- Represents the properties of a metal alloy bar --/
structure MetalBar where
  tin_weight : ℝ
  silver_weight : ℝ
  total_weight_loss : ℝ
  tin_loss_rate : ℝ
  silver_loss_rate : ℝ
  tin_silver_ratio : ℝ

/-- Theorem stating the weight of the metal bar given the conditions --/
theorem metal_bar_weight (bar : MetalBar)
  (h1 : bar.total_weight_loss = 6)
  (h2 : bar.tin_loss_rate = 1.375 / 10)
  (h3 : bar.silver_loss_rate = 0.375 / 5)
  (h4 : bar.tin_silver_ratio = 2 / 3)
  (h5 : bar.tin_weight * bar.tin_loss_rate + bar.silver_weight * bar.silver_loss_rate = bar.total_weight_loss)
  (h6 : bar.tin_weight / bar.silver_weight = bar.tin_silver_ratio) :
  bar.tin_weight + bar.silver_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_metal_bar_weight_l2035_203502


namespace NUMINAMATH_CALUDE_prob_red_base_is_half_l2035_203574

-- Define the total number of bases
def total_bases : ℕ := 4

-- Define the number of red educational bases
def red_bases : ℕ := 2

-- Define the probability of choosing a red educational base
def prob_red_base : ℚ := red_bases / total_bases

-- Theorem statement
theorem prob_red_base_is_half : prob_red_base = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_base_is_half_l2035_203574


namespace NUMINAMATH_CALUDE_specific_quilt_shaded_fraction_l2035_203598

/-- Represents a square quilt composed of smaller squares -/
structure Quilt :=
  (side_length : ℕ)
  (shaded_triangles : ℕ)
  (shaded_squares : ℕ)

/-- Calculates the fraction of a quilt that is shaded -/
def shaded_fraction (q : Quilt) : ℚ :=
  let total_area : ℚ := (q.side_length * q.side_length : ℚ)
  let shaded_area : ℚ := (q.shaded_squares : ℚ) + (q.shaded_triangles : ℚ) / 2
  shaded_area / total_area

/-- Theorem stating that for a specific quilt configuration, the shaded fraction is 5/18 -/
theorem specific_quilt_shaded_fraction :
  let q := Quilt.mk 3 3 1
  shaded_fraction q = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_specific_quilt_shaded_fraction_l2035_203598


namespace NUMINAMATH_CALUDE_number_of_broadcasting_methods_l2035_203515

/-- Represents the number of commercial advertisements -/
def num_commercial_ads : ℕ := 4

/-- Represents the number of public service advertisements -/
def num_public_service_ads : ℕ := 2

/-- Represents the total number of advertisements -/
def total_ads : ℕ := num_commercial_ads + num_public_service_ads

/-- Represents the fact that public service ads must be at the beginning and end -/
def public_service_ads_fixed : Prop := True

theorem number_of_broadcasting_methods : 
  (num_commercial_ads = 4 ∧ 
   num_public_service_ads = 2 ∧ 
   total_ads = 6 ∧ 
   public_service_ads_fixed) → 
  (Nat.factorial num_commercial_ads = 24) := by
  sorry

end NUMINAMATH_CALUDE_number_of_broadcasting_methods_l2035_203515


namespace NUMINAMATH_CALUDE_stars_permutations_l2035_203512

def word_length : ℕ := 5
def repeated_letter_count : ℕ := 2
def unique_letters_count : ℕ := 3

theorem stars_permutations :
  (word_length.factorial) / (repeated_letter_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_stars_permutations_l2035_203512


namespace NUMINAMATH_CALUDE_sequence_equality_l2035_203585

/-- Sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2018 / (n + 1) * a (n + 1) + a n

/-- Sequence b_n defined recursively -/
def b : ℕ → ℚ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2020 / (n + 1) * b (n + 1) + b n

/-- The main theorem to prove -/
theorem sequence_equality : a 1010 / 1010 = b 1009 / 1009 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l2035_203585
