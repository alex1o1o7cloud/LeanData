import Mathlib

namespace NUMINAMATH_CALUDE_extreme_values_of_f_range_of_a_for_f_greater_than_g_l715_71543

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - a * x + (a + 3) / x
def g (a : ℝ) (x : ℝ) : ℝ := 2 * Real.exp x - 4 * x + 2 * a

-- Theorem for part 1
theorem extreme_values_of_f (x : ℝ) (hx : x > 0) :
  let f_half := f (1/2)
  (∃ (x_min : ℝ), x_min > 0 ∧ ∀ y, y > 0 → f_half y ≥ f_half x_min) ∧
  (∃ (x_max : ℝ), x_max > 0 ∧ ∀ y, y > 0 → f_half y ≤ f_half x_max) ∧
  (∀ y, y > 0 → f_half y ≥ 3) ∧
  (∀ y, y > 0 → f_half y ≤ 4 * Real.log 7 - 3) :=
sorry

-- Theorem for part 2
theorem range_of_a_for_f_greater_than_g (a : ℝ) (ha : a ≥ 1) :
  (∃ (x₁ x₂ : ℝ), 1/2 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1/2 ≤ x₂ ∧ x₂ ≤ 2 ∧ f a x₁ > g a x₂) ↔
  (1 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_range_of_a_for_f_greater_than_g_l715_71543


namespace NUMINAMATH_CALUDE_largest_k_is_correct_l715_71587

/-- The largest natural number k for which there exists a natural number n 
    satisfying the inequality sin(n + 1) < sin(n + 2) < sin(n + 3) < ... < sin(n + k) -/
def largest_k : ℕ := 3

/-- Predicate that checks if the sine inequality holds for a given n and k -/
def sine_inequality (n k : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → Real.sin (n + i) < Real.sin (n + j)

theorem largest_k_is_correct :
  (∃ n : ℕ, sine_inequality n largest_k) ∧
  (∀ k : ℕ, k > largest_k → ¬∃ n : ℕ, sine_inequality n k) :=
sorry

end NUMINAMATH_CALUDE_largest_k_is_correct_l715_71587


namespace NUMINAMATH_CALUDE_marked_circle_triangles_l715_71500

/-- A circle with n equally spaced points on its circumference -/
structure MarkedCircle (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- Number of triangles that can be formed with n points -/
def num_triangles (c : MarkedCircle n) : ℕ := sorry

/-- Number of equilateral triangles that can be formed with n points -/
def num_equilateral_triangles (c : MarkedCircle n) : ℕ := sorry

/-- Number of right triangles that can be formed with n points -/
def num_right_triangles (c : MarkedCircle n) : ℕ := sorry

theorem marked_circle_triangles 
  (c4 : MarkedCircle 4) 
  (c5 : MarkedCircle 5) 
  (c6 : MarkedCircle 6) : 
  (num_triangles c4 = 4) ∧ 
  (num_equilateral_triangles c5 = 0) ∧ 
  (num_right_triangles c6 = 12) := by sorry

end NUMINAMATH_CALUDE_marked_circle_triangles_l715_71500


namespace NUMINAMATH_CALUDE_checkerboard_covering_l715_71576

/-- Represents a checkerboard -/
structure Checkerboard where
  size : ℕ
  removed_squares : Fin (4 * size * size) × Fin (4 * size * size)

/-- Represents a 2 × 1 domino -/
structure Domino

/-- Predicate to check if two squares are of opposite colors -/
def opposite_colors (c : Checkerboard) (s1 s2 : Fin (4 * c.size * c.size)) : Prop :=
  (s1.val + s2.val) % 2 = 1

/-- Predicate to check if a checkerboard can be covered by dominoes -/
def can_cover (c : Checkerboard) : Prop :=
  ∃ (covering : List (Fin (4 * c.size * c.size) × Fin (4 * c.size * c.size))),
    (∀ (square : Fin (4 * c.size * c.size)), 
      square ≠ c.removed_squares.1 ∧ square ≠ c.removed_squares.2 → 
      ∃ (domino : Fin (4 * c.size * c.size) × Fin (4 * c.size * c.size)), 
        domino ∈ covering ∧ (square = domino.1 ∨ square = domino.2)) ∧
    (∀ (domino : Fin (4 * c.size * c.size) × Fin (4 * c.size * c.size)), 
      domino ∈ covering → 
      (domino.1 ≠ c.removed_squares.1 ∧ domino.1 ≠ c.removed_squares.2) ∧
      (domino.2 ≠ c.removed_squares.1 ∧ domino.2 ≠ c.removed_squares.2) ∧
      (domino.1.val + 1 = domino.2.val ∨ domino.1.val + 2 * c.size = domino.2.val))

/-- Theorem stating that any 2k × 2k checkerboard with two squares of opposite colors removed can be covered by 2 × 1 dominoes -/
theorem checkerboard_covering (k : ℕ) (c : Checkerboard) 
  (h_size : c.size = 2 * k)
  (h_opposite : opposite_colors c c.removed_squares.1 c.removed_squares.2) :
  can_cover c :=
sorry

end NUMINAMATH_CALUDE_checkerboard_covering_l715_71576


namespace NUMINAMATH_CALUDE_first_child_born_1982_l715_71596

/-- Represents the year the first child was born -/
def first_child_birth_year : ℕ := sorry

/-- The year the couple got married -/
def marriage_year : ℕ := 1980

/-- The year the second child was born -/
def second_child_birth_year : ℕ := 1984

/-- The year when the combined ages of children equal the years of marriage -/
def reference_year : ℕ := 1986

theorem first_child_born_1982 :
  (reference_year - first_child_birth_year) + (reference_year - second_child_birth_year) = reference_year - marriage_year →
  first_child_birth_year = 1982 :=
by sorry

end NUMINAMATH_CALUDE_first_child_born_1982_l715_71596


namespace NUMINAMATH_CALUDE_equation_solution_l715_71578

theorem equation_solution (x : ℝ) : 
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 2) * (x - 1) / 
  ((x - 4) * (x - 2) * (x - 1)) = 1 →
  x = (9 + Real.sqrt 5) / 2 ∨ x = (9 - Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l715_71578


namespace NUMINAMATH_CALUDE_sum_of_multiples_l715_71547

def smallest_two_digit_multiple_of_7 : ℕ := sorry

def smallest_three_digit_multiple_of_5 : ℕ := sorry

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_7 + smallest_three_digit_multiple_of_5 = 114 := by sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l715_71547


namespace NUMINAMATH_CALUDE_base_conversion_equality_l715_71521

-- Define the base-5 number 132₅
def base_5_num : ℕ := 1 * 5^2 + 3 * 5^1 + 2 * 5^0

-- Define the base-b number 221ᵦ as a function of b
def base_b_num (b : ℝ) : ℝ := 2 * b^2 + 2 * b + 1

-- Theorem statement
theorem base_conversion_equality :
  ∃ b : ℝ, b > 0 ∧ base_5_num = base_b_num b ∧ b = (-1 + Real.sqrt 83) / 2 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l715_71521


namespace NUMINAMATH_CALUDE_remainder_scaling_l715_71530

theorem remainder_scaling (a b c r : ℤ) : 
  (a = b * c + r) → (0 ≤ r) → (r < b) → 
  ∃ (q : ℤ), (3 * a = 3 * b * q + 3 * r) ∧ (0 ≤ 3 * r) ∧ (3 * r < 3 * b) :=
by sorry

end NUMINAMATH_CALUDE_remainder_scaling_l715_71530


namespace NUMINAMATH_CALUDE_tangent_line_m_value_l715_71515

-- Define the curve
def curve (x m n : ℝ) : ℝ := x^3 + m*x + n

-- Define the line
def line (x : ℝ) : ℝ := 3*x + 1

-- State the theorem
theorem tangent_line_m_value :
  ∀ (m n : ℝ),
  (curve 1 m n = 4) →  -- The point (1, 4) lies on the curve
  (line 1 = 4) →       -- The point (1, 4) lies on the line
  (∀ x : ℝ, curve x m n ≤ line x) →  -- The line is tangent to the curve (no other intersection)
  (m = 0) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_m_value_l715_71515


namespace NUMINAMATH_CALUDE_train_length_proof_l715_71503

/-- Given a train that passes a pole in 10 seconds and a 1250m long platform in 60 seconds,
    prove that the length of the train is 250 meters. -/
theorem train_length_proof (pole_time : ℝ) (platform_time : ℝ) (platform_length : ℝ) 
    (h1 : pole_time = 10)
    (h2 : platform_time = 60)
    (h3 : platform_length = 1250) : 
  ∃ (train_length : ℝ), train_length = 250 ∧ 
    train_length / pole_time = (train_length + platform_length) / platform_time :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l715_71503


namespace NUMINAMATH_CALUDE_school_rewards_problem_l715_71572

/-- The price of a practical backpack -/
def backpack_price : ℝ := 60

/-- The price of a multi-functional pencil case -/
def pencil_case_price : ℝ := 40

/-- The total budget for purchases -/
def total_budget : ℝ := 1140

/-- The total number of items to be purchased -/
def total_items : ℕ := 25

/-- The maximum number of backpacks that can be purchased -/
def max_backpacks : ℕ := 7

theorem school_rewards_problem :
  (3 * backpack_price + 2 * pencil_case_price = 260) ∧
  (5 * backpack_price + 4 * pencil_case_price = 460) ∧
  (∀ m : ℕ, m ≤ total_items → 
    backpack_price * m + pencil_case_price * (total_items - m) ≤ total_budget) →
  max_backpacks = 7 := by sorry

end NUMINAMATH_CALUDE_school_rewards_problem_l715_71572


namespace NUMINAMATH_CALUDE_second_item_is_14_l715_71560

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  initial_selection : ℕ

/-- Calculates the second item in a systematic sample -/
def second_item (s : SystematicSampling) : ℕ :=
  s.initial_selection + (s.population_size / s.sample_size)

/-- Theorem: In the given systematic sampling scenario, the second item is 14 -/
theorem second_item_is_14 :
  let s : SystematicSampling := {
    population_size := 60,
    sample_size := 6,
    initial_selection := 4
  }
  second_item s = 14 := by
  sorry


end NUMINAMATH_CALUDE_second_item_is_14_l715_71560


namespace NUMINAMATH_CALUDE_regular_polygon_pentagon_l715_71553

/-- A regular polygon with side length 25 and perimeter divisible by 5 yielding the side length is a pentagon with perimeter 125. -/
theorem regular_polygon_pentagon (n : ℕ) (perimeter : ℝ) : 
  n > 0 → 
  perimeter > 0 → 
  perimeter / n = 25 → 
  perimeter / 5 = 25 → 
  n = 5 ∧ perimeter = 125 := by
  sorry

#check regular_polygon_pentagon

end NUMINAMATH_CALUDE_regular_polygon_pentagon_l715_71553


namespace NUMINAMATH_CALUDE_equation_holds_for_seven_halves_l715_71580

theorem equation_holds_for_seven_halves : 
  let x : ℚ := 7/2
  let y : ℚ := (x^2 - 9) / (x - 3)
  y = 3*x - 4 := by sorry

end NUMINAMATH_CALUDE_equation_holds_for_seven_halves_l715_71580


namespace NUMINAMATH_CALUDE_not_linear_in_M_exp_in_M_sin_in_M_iff_l715_71546

/-- The set M of functions satisfying f(x+T) = T⋅f(x) for some non-zero T -/
def M : Set (ℝ → ℝ) :=
  {f | ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = T * f x}

theorem not_linear_in_M :
    ∀ T : ℝ, T ≠ 0 → ∃ x : ℝ, x + T ≠ T * x := by sorry

theorem exp_in_M (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
    (∃ T : ℝ, T > 0 ∧ a^T = T) → (fun x ↦ a^x) ∈ M := by sorry

theorem sin_in_M_iff (k : ℝ) :
    (fun x ↦ Real.sin (k * x)) ∈ M ↔ ∃ m : ℤ, k = m * Real.pi := by sorry

end NUMINAMATH_CALUDE_not_linear_in_M_exp_in_M_sin_in_M_iff_l715_71546


namespace NUMINAMATH_CALUDE_intersection_area_is_4pi_l715_71525

-- Define the rectangle
def rectangle_vertices : List (ℝ × ℝ) := [(2, 3), (2, 15), (13, 3), (13, 15)]

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 13)^2 + (y - 3)^2 = 16

-- Define the area of intersection
def intersection_area (rect : List (ℝ × ℝ)) (circle : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem intersection_area_is_4pi :
  intersection_area rectangle_vertices circle_equation = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_4pi_l715_71525


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_neg_two_l715_71569

theorem fraction_zero_implies_x_neg_two (x : ℝ) :
  (abs x - 2) / (x^2 - 4*x + 4) = 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_neg_two_l715_71569


namespace NUMINAMATH_CALUDE_min_sum_dimensions_for_volume_2184_l715_71510

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the volume of a rectangular box -/
def volume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Calculates the sum of dimensions of a rectangular box -/
def sumDimensions (d : BoxDimensions) : ℕ := d.length + d.width + d.height

/-- Theorem: The minimum sum of dimensions for a box with volume 2184 is 36 -/
theorem min_sum_dimensions_for_volume_2184 :
  (∃ (d : BoxDimensions), volume d = 2184) →
  (∀ (d : BoxDimensions), volume d = 2184 → sumDimensions d ≥ 36) ∧
  (∃ (d : BoxDimensions), volume d = 2184 ∧ sumDimensions d = 36) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_for_volume_2184_l715_71510


namespace NUMINAMATH_CALUDE_min_time_for_eight_people_l715_71549

/-- Represents a group of people sharing information -/
structure InformationSharingGroup where
  numPeople : Nat
  callDuration : Nat
  initialInfo : Fin numPeople → Nat

/-- Represents the minimum time needed for complete information sharing -/
def minTimeForCompleteSharing (group : InformationSharingGroup) : Nat :=
  sorry

/-- Theorem stating the minimum time for the specific problem -/
theorem min_time_for_eight_people
  (group : InformationSharingGroup)
  (h1 : group.numPeople = 8)
  (h2 : group.callDuration = 3)
  (h3 : ∀ i j : Fin group.numPeople, i ≠ j → group.initialInfo i ≠ group.initialInfo j) :
  minTimeForCompleteSharing group = 9 :=
sorry

end NUMINAMATH_CALUDE_min_time_for_eight_people_l715_71549


namespace NUMINAMATH_CALUDE_sum_of_preceding_terms_l715_71555

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ d : ℕ, ∀ i : ℕ, i < n → a (i + 1) = a i + d

theorem sum_of_preceding_terms (a : ℕ → ℕ) (n : ℕ) :
  arithmetic_sequence a n →
  a 0 = 3 →
  a (n - 1) = 39 →
  n ≥ 3 →
  a (n - 2) + a (n - 3) = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_preceding_terms_l715_71555


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l715_71557

/-- The area of an equilateral triangle with altitude 2√3 is 4√3 -/
theorem equilateral_triangle_area (h : ℝ) (altitude : h = 2 * Real.sqrt 3) :
  let side := 2 * h / Real.sqrt 3
  let area := 1/2 * side * h
  area = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l715_71557


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l715_71554

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (Complex.I - 1) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l715_71554


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l715_71520

theorem number_of_divisors_36 : Nat.card {d : ℕ | d ∣ 36} = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l715_71520


namespace NUMINAMATH_CALUDE_doritos_ratio_l715_71563

theorem doritos_ratio (total_bags : ℕ) (doritos_piles : ℕ) (bags_per_pile : ℕ) 
  (h1 : total_bags = 80)
  (h2 : doritos_piles = 4)
  (h3 : bags_per_pile = 5) : 
  (doritos_piles * bags_per_pile : ℚ) / total_bags = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_doritos_ratio_l715_71563


namespace NUMINAMATH_CALUDE_min_balls_for_fifteen_in_box_l715_71532

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least 15 of a single color -/
def minBallsForFifteen (counts : BallCounts) : Nat :=
  sorry

/-- Theorem stating the minimum number of balls to draw for the given problem -/
theorem min_balls_for_fifteen_in_box :
  let counts : BallCounts := {
    red := 28, green := 20, yellow := 19,
    blue := 13, white := 11, black := 9
  }
  minBallsForFifteen counts = 76 := by sorry

end NUMINAMATH_CALUDE_min_balls_for_fifteen_in_box_l715_71532


namespace NUMINAMATH_CALUDE_equation_one_real_root_l715_71586

theorem equation_one_real_root :
  ∃! x : ℝ, (Real.sqrt (x^2 + 2*x - 63) + Real.sqrt (x + 9) - Real.sqrt (7 - x) + x + 13 = 0) ∧
             (x^2 + 2*x - 63 ≥ 0) ∧
             (x + 9 ≥ 0) ∧
             (7 - x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_one_real_root_l715_71586


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l715_71535

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l715_71535


namespace NUMINAMATH_CALUDE_discriminant_nonnegative_root_greater_than_three_implies_a_greater_than_four_l715_71504

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - a*x + a - 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) : ℝ := a^2 - 4*(a - 1)

-- Theorem 1: The discriminant is always non-negative
theorem discriminant_nonnegative (a : ℝ) : discriminant a ≥ 0 := by
  sorry

-- Theorem 2: When one root is greater than 3, a > 4
theorem root_greater_than_three_implies_a_greater_than_four (a : ℝ) :
  (∃ x, quadratic a x = 0 ∧ x > 3) → a > 4 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_nonnegative_root_greater_than_three_implies_a_greater_than_four_l715_71504


namespace NUMINAMATH_CALUDE_three_parallel_lines_theorem_l715_71502

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary properties for a line in 3D space
  -- This is a placeholder and may need to be adjusted based on Lean's standard library

-- Define a type for planes in 3D space
structure Plane3D where
  -- Add necessary properties for a plane in 3D space
  -- This is a placeholder and may need to be adjusted based on Lean's standard library

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines

-- Define a function to check if three lines are coplanar
def are_coplanar (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Definition of coplanar lines

-- Define a function to count the number of planes determined by three lines
def count_planes (l1 l2 l3 : Line3D) : Nat :=
  sorry -- Count the number of planes

-- Define a function to count the number of parts space is divided into by these planes
def count_space_parts (planes : List Plane3D) : Nat :=
  sorry -- Count the number of parts

-- Theorem statement
theorem three_parallel_lines_theorem (a b c : Line3D) 
  (h_parallel_ab : are_parallel a b)
  (h_parallel_bc : are_parallel b c)
  (h_parallel_ac : are_parallel a c)
  (h_not_coplanar : ¬ are_coplanar a b c) :
  (count_planes a b c = 3) ∧ 
  (count_space_parts (sorry : List Plane3D) = 7) := by
  sorry


end NUMINAMATH_CALUDE_three_parallel_lines_theorem_l715_71502


namespace NUMINAMATH_CALUDE_diego_paycheck_l715_71577

/-- Diego's monthly paycheck problem -/
theorem diego_paycheck (monthly_expenses : ℝ) (annual_savings : ℝ) (h1 : monthly_expenses = 4600) (h2 : annual_savings = 4800) :
  monthly_expenses + annual_savings / 12 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_diego_paycheck_l715_71577


namespace NUMINAMATH_CALUDE_base_ten_proof_l715_71567

/-- Given that in base b, the square of 31_b is 1021_b, prove that b = 10 -/
theorem base_ten_proof (b : ℕ) (h : b > 3) : 
  (3 * b + 1)^2 = b^3 + 2 * b + 1 → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_ten_proof_l715_71567


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l715_71592

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_common_difference 
  (a₁ d : ℝ) 
  (h1 : arithmetic_sequence a₁ d 5 = 8)
  (h2 : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 = 6) :
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l715_71592


namespace NUMINAMATH_CALUDE_fraction_of_c_grades_l715_71590

theorem fraction_of_c_grades 
  (total_students : ℕ) 
  (a_fraction : ℚ) 
  (b_fraction : ℚ) 
  (d_count : ℕ) 
  (h_total : total_students = 800)
  (h_a : a_fraction = 1 / 5)
  (h_b : b_fraction = 1 / 4)
  (h_d : d_count = 40) :
  (total_students - (a_fraction * total_students + b_fraction * total_students + d_count)) / total_students = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_c_grades_l715_71590


namespace NUMINAMATH_CALUDE_complex_sum_squared_l715_71507

noncomputable def i : ℂ := Complex.I

theorem complex_sum_squared 
  (a b c : ℂ) 
  (eq1 : a^2 + a*b + b^2 = 1 + i)
  (eq2 : b^2 + b*c + c^2 = -2)
  (eq3 : c^2 + c*a + a^2 = 1) :
  (a*b + b*c + c*a)^2 = (-11 - 4*i) / 3 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_squared_l715_71507


namespace NUMINAMATH_CALUDE_brownie_division_l715_71522

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a tray of brownies -/
structure BrownieTray where
  tray : Dimensions
  piece : Dimensions

/-- Calculates the number of brownie pieces that can be cut from the tray -/
def num_pieces (bt : BrownieTray) : ℕ :=
  (area bt.tray) / (area bt.piece)

/-- Theorem: A 24-inch by 20-inch tray can be divided into exactly 40 pieces of 3-inch by 4-inch brownies -/
theorem brownie_division :
  let bt : BrownieTray := {
    tray := { length := 24, width := 20 },
    piece := { length := 3, width := 4 }
  }
  num_pieces bt = 40 := by sorry

end NUMINAMATH_CALUDE_brownie_division_l715_71522


namespace NUMINAMATH_CALUDE_bernardo_always_less_than_silvia_l715_71571

def bernardo_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def silvia_set : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9}

def bernardo_number (a b c : ℕ) : ℕ := 
  100 * min a (min b c) + 10 * (a + b + c - max a (max b c) - min a (min b c)) + max a (max b c)

def silvia_number (x y z : ℕ) : ℕ := 
  100 * max x (max y z) + 10 * (x + y + z - max x (max y z) - min x (min y z)) + min x (min y z)

theorem bernardo_always_less_than_silvia :
  ∀ (a b c : ℕ) (x y z : ℕ),
    a ∈ bernardo_set → b ∈ bernardo_set → c ∈ bernardo_set →
    x ∈ silvia_set → y ∈ silvia_set → z ∈ silvia_set →
    a ≠ b → b ≠ c → a ≠ c →
    x ≠ y → y ≠ z → x ≠ z →
    bernardo_number a b c < silvia_number x y z := by
  sorry

end NUMINAMATH_CALUDE_bernardo_always_less_than_silvia_l715_71571


namespace NUMINAMATH_CALUDE_small_plate_diameter_l715_71564

theorem small_plate_diameter
  (big_plate_diameter : ℝ)
  (uncovered_fraction : ℝ)
  (h1 : big_plate_diameter = 12)
  (h2 : uncovered_fraction = 0.3055555555555555) :
  ∃ (small_plate_diameter : ℝ),
    small_plate_diameter = 10 ∧
    (1 - uncovered_fraction) * (π * big_plate_diameter^2 / 4) = π * small_plate_diameter^2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_small_plate_diameter_l715_71564


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_reciprocals_l715_71514

theorem cubic_roots_sum_of_cubes_reciprocals 
  (a b c d : ℂ) 
  (r s t : ℂ) 
  (h₁ : a ≠ 0) 
  (h₂ : d ≠ 0) 
  (h₃ : a * r^3 + b * r^2 + c * r + d = 0) 
  (h₄ : a * s^3 + b * s^2 + c * s + d = 0) 
  (h₅ : a * t^3 + b * t^2 + c * t + d = 0) : 
  1 / r^3 + 1 / s^3 + 1 / t^3 = c^3 / d^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_reciprocals_l715_71514


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l715_71513

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 14 ∧ (9679 - k) % 15 = 0 ∧ ∀ (m : ℕ), m < k → (9679 - m) % 15 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l715_71513


namespace NUMINAMATH_CALUDE_correct_second_sale_price_l715_71523

/-- The price of a single toothbrush in yuan -/
def toothbrush_price : ℝ := sorry

/-- The price of a single tube of toothpaste in yuan -/
def toothpaste_price : ℝ := sorry

/-- The total price of 26 toothbrushes and 14 tubes of toothpaste -/
def first_sale : ℝ := 26 * toothbrush_price + 14 * toothpaste_price

/-- The recorded total price of the first sale -/
def first_sale_record : ℝ := 264

theorem correct_second_sale_price :
  first_sale = first_sale_record →
  39 * toothbrush_price + 21 * toothpaste_price = 396 := by
  sorry

end NUMINAMATH_CALUDE_correct_second_sale_price_l715_71523


namespace NUMINAMATH_CALUDE_hash_2_3_1_5_equals_6_l715_71583

/-- The # operation for real numbers -/
def hash (a b c d : ℝ) : ℝ := b^2 - 4*a*c + d

/-- Theorem stating that #(2, 3, 1, 5) = 6 -/
theorem hash_2_3_1_5_equals_6 : hash 2 3 1 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_3_1_5_equals_6_l715_71583


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l715_71595

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (10 * π / 180) =
  (2 * Real.cos (40 * π / 180)) / (Real.cos (10 * π / 180) ^ 2 * Real.cos (30 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l715_71595


namespace NUMINAMATH_CALUDE_count_fives_in_S_l715_71597

/-- The sum of an arithmetic sequence with first term 1, common difference 9, and last term 10^2013 -/
def S : ℕ := (1 + 10^2013) * ((10^2013 + 8) / 18)

/-- Counts the occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

theorem count_fives_in_S : countDigit S 5 = 4022 := by sorry

end NUMINAMATH_CALUDE_count_fives_in_S_l715_71597


namespace NUMINAMATH_CALUDE_chord_AB_equation_tangent_circle_equation_l715_71581

-- Define the parabola E
def E (x y : ℝ) : Prop := x^2 = 4*y

-- Define point M
def M : ℝ × ℝ := (1, 4)

-- Define the chord AB passing through M as its midpoint
def chord_AB (x y : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    E x1 y1 ∧ E x2 y2 ∧
    (x, y) = ((x1 + x2)/2, (y1 + y2)/2) ∧
    (x, y) = M

-- Define the tangent line l
def tangent_line (x0 y0 b : ℝ) : Prop :=
  E x0 y0 ∧ y0 = x0 + b

-- Theorem for the equation of line AB
theorem chord_AB_equation :
  ∀ x y : ℝ, chord_AB x y → x - 2*y + 7 = 0 := sorry

-- Theorem for the equation of the circle
theorem tangent_circle_equation :
  ∀ x0 y0 b : ℝ,
    tangent_line x0 y0 b →
    ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 4 := sorry

end NUMINAMATH_CALUDE_chord_AB_equation_tangent_circle_equation_l715_71581


namespace NUMINAMATH_CALUDE_bill_face_value_l715_71539

/-- Face value of a bill given true discount and banker's discount -/
def face_value (true_discount : ℚ) (bankers_discount : ℚ) : ℚ :=
  true_discount * bankers_discount / (bankers_discount - true_discount)

/-- Theorem: Given the true discount and banker's discount, prove the face value is 2460 -/
theorem bill_face_value :
  face_value 360 421.7142857142857 = 2460 := by
  sorry

end NUMINAMATH_CALUDE_bill_face_value_l715_71539


namespace NUMINAMATH_CALUDE_reflection_xoz_coordinates_l715_71516

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflection of a point across the xOz plane -/
def reflectXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem reflection_xoz_coordinates :
  let P : Point3D := { x := 3, y := -2, z := 1 }
  reflectXOZ P = { x := 3, y := 2, z := 1 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_xoz_coordinates_l715_71516


namespace NUMINAMATH_CALUDE_empty_cube_exists_l715_71565

/-- Represents a 3D coordinate within the cube -/
structure Coord where
  x : Fin 5
  y : Fin 5
  z : Fin 5

/-- Represents the state of a unit cube -/
inductive CubeState
  | Occupied
  | Empty

/-- The type of the cube, mapping coordinates to cube states -/
def Cube := Coord → CubeState

/-- Checks if two coordinates are adjacent -/
def isAdjacent (c1 c2 : Coord) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ (c1.z = c2.z + 1 ∨ c1.z + 1 = c2.z)) ∨
  (c1.x = c2.x ∧ c1.z = c2.z ∧ (c1.y = c2.y + 1 ∨ c1.y + 1 = c2.y)) ∨
  (c1.y = c2.y ∧ c1.z = c2.z ∧ (c1.x = c2.x + 1 ∨ c1.x + 1 = c2.x))

/-- A function representing the movement of objects -/
def moveObjects (initial : Cube) : Cube :=
  sorry

theorem empty_cube_exists (initial : Cube) :
  (∀ c : Coord, initial c = CubeState.Occupied) →
  ∃ c : Coord, (moveObjects initial) c = CubeState.Empty :=
sorry

end NUMINAMATH_CALUDE_empty_cube_exists_l715_71565


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l715_71582

theorem cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) :
  selling_price = 720 →
  num_balls_sold = 15 →
  num_balls_loss = 5 →
  ∃ (cost_price : ℕ), 
    cost_price * num_balls_sold - cost_price * num_balls_loss = selling_price ∧
    cost_price = 72 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l715_71582


namespace NUMINAMATH_CALUDE_sin_2a_value_l715_71501

theorem sin_2a_value (a : Real) (h1 : a ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * a) = Real.sqrt 2 * Real.sin (π / 4 - a)) : 
  Real.sin (2 * a) = -8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2a_value_l715_71501


namespace NUMINAMATH_CALUDE_initial_apples_l715_71545

theorem initial_apples (minseok_ate jaeyoon_ate apples_left : ℕ) 
  (h1 : minseok_ate = 3)
  (h2 : jaeyoon_ate = 3)
  (h3 : apples_left = 2) : 
  minseok_ate + jaeyoon_ate + apples_left = 8 :=
by sorry

end NUMINAMATH_CALUDE_initial_apples_l715_71545


namespace NUMINAMATH_CALUDE_prob_two_green_balls_l715_71533

/-- The probability of drawing two green balls from a bag containing two green balls and one red ball when two balls are randomly drawn. -/
theorem prob_two_green_balls (total_balls : ℕ) (green_balls : ℕ) (red_balls : ℕ) : 
  total_balls = 3 → 
  green_balls = 2 → 
  red_balls = 1 → 
  (green_balls.choose 2 : ℚ) / (total_balls.choose 2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_prob_two_green_balls_l715_71533


namespace NUMINAMATH_CALUDE_point_in_region_l715_71574

theorem point_in_region (m : ℝ) :
  (2 * m + 3 < 4) → m < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_l715_71574


namespace NUMINAMATH_CALUDE_sum_of_squares_l715_71599

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_seven : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = -6/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l715_71599


namespace NUMINAMATH_CALUDE_constant_term_expansion_l715_71573

/-- The constant term in the expansion of (x - 1/(2x))^6 is -5/2 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := λ x => (x - 1/(2*x))^6
  ∃ (c : ℝ), (∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ c = -5/2 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l715_71573


namespace NUMINAMATH_CALUDE_decimal_equals_fraction_l715_71537

/-- The decimal representation of 0.1⁻35 as a real number -/
def decimal_rep : ℚ := 0.1 + (35 / 990)

/-- The fraction 67/495 as a rational number -/
def fraction : ℚ := 67 / 495

/-- Assertion that 67 and 495 are coprime -/
axiom coprime_67_495 : Nat.Coprime 67 495

theorem decimal_equals_fraction : decimal_rep = fraction := by sorry

end NUMINAMATH_CALUDE_decimal_equals_fraction_l715_71537


namespace NUMINAMATH_CALUDE_probability_inner_circle_l715_71593

/-- The probability of a random point from a circle with radius 3 falling within a concentric circle with radius 1.5 -/
theorem probability_inner_circle (outer_radius inner_radius : ℝ) 
  (h_outer : outer_radius = 3)
  (h_inner : inner_radius = 1.5) :
  (π * inner_radius^2) / (π * outer_radius^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_inner_circle_l715_71593


namespace NUMINAMATH_CALUDE_six_regions_three_colors_l715_71519

/-- The number of ways to color n regions using k colors --/
def colorings (n k : ℕ) : ℕ := k^n

/-- The number of ways to color n regions using exactly k colors --/
def exactColorings (n k : ℕ) : ℕ :=
  (Nat.choose k k) * k^n - (Nat.choose k (k-1)) * (k-1)^n + (Nat.choose k (k-2)) * (k-2)^n

theorem six_regions_three_colors :
  exactColorings 6 3 = 540 := by sorry

end NUMINAMATH_CALUDE_six_regions_three_colors_l715_71519


namespace NUMINAMATH_CALUDE_trailing_zeros_of_square_l715_71562

/-- The number of trailing zeros in (10^10 - 2)^2 is 17 -/
theorem trailing_zeros_of_square : ∃ n : ℕ, (10^10 - 2)^2 = n * 10^17 ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_square_l715_71562


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_588_l715_71585

/-- The area of a rectangle with an inscribed circle of radius 7 and length-to-width ratio of 3:1 -/
theorem rectangle_area_with_inscribed_circle : ℝ :=
  let circle_radius : ℝ := 7
  let length_to_width_ratio : ℝ := 3
  let rectangle_width : ℝ := 2 * circle_radius
  let rectangle_length : ℝ := length_to_width_ratio * rectangle_width
  rectangle_length * rectangle_width

/-- Proof that the area of the rectangle is 588 -/
theorem rectangle_area_is_588 : rectangle_area_with_inscribed_circle = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_rectangle_area_is_588_l715_71585


namespace NUMINAMATH_CALUDE_arrangement_theorem_l715_71594

def num_boys : ℕ := 3
def num_girls : ℕ := 4
def total_people : ℕ := num_boys + num_girls

def arrange_condition1 : ℕ := sorry

def arrange_condition2 : ℕ := sorry

def arrange_condition3 : ℕ := sorry

def arrange_condition4 : ℕ := sorry

theorem arrangement_theorem :
  arrange_condition1 = 2160 ∧
  arrange_condition2 = 720 ∧
  arrange_condition3 = 144 ∧
  arrange_condition4 = 720 :=
sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l715_71594


namespace NUMINAMATH_CALUDE_wings_temperature_l715_71527

/-- Given an initial oven temperature and a required temperature increase,
    calculate the final required temperature. -/
def required_temperature (initial_temp increase : ℕ) : ℕ :=
  initial_temp + increase

/-- Theorem: The required temperature for the wings is 546 degrees,
    given an initial temperature of 150 degrees and a needed increase of 396 degrees. -/
theorem wings_temperature : required_temperature 150 396 = 546 := by
  sorry

end NUMINAMATH_CALUDE_wings_temperature_l715_71527


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l715_71588

theorem no_solutions_for_equation : ¬∃ (x : Fin 8 → ℝ), 
  (2 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + 
  (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + (x 7)^2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l715_71588


namespace NUMINAMATH_CALUDE_vector_properties_l715_71552

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 1)

theorem vector_properties :
  let magnitude_sum := Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2)
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let x := -9/2
  (magnitude_sum = 5) ∧
  (angle = π/4) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ (x * a.1 + 3 * b.1, x * a.2 + 3 * b.2) = (k * (3 * a.1 - 2 * b.1), k * (3 * a.2 - 2 * b.2))) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l715_71552


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l715_71509

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 1080 → (n - 2) * 180 = angle_sum → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l715_71509


namespace NUMINAMATH_CALUDE_sports_enjoyment_misreporting_l715_71517

theorem sports_enjoyment_misreporting (total : ℝ) (total_pos : 0 < total) :
  let enjoy := 0.7 * total
  let not_enjoy := 0.3 * total
  let enjoy_say_enjoy := 0.75 * enjoy
  let enjoy_say_not := 0.25 * enjoy
  let not_enjoy_say_not := 0.85 * not_enjoy
  let not_enjoy_say_enjoy := 0.15 * not_enjoy
  enjoy_say_not / (enjoy_say_not + not_enjoy_say_not) = 7 / 17 := by
sorry

end NUMINAMATH_CALUDE_sports_enjoyment_misreporting_l715_71517


namespace NUMINAMATH_CALUDE_tram_speed_l715_71558

/-- Proves that a tram with constant speed passing an observer in 2 seconds
    and traversing a 96-meter tunnel in 10 seconds has a speed of 12 m/s. -/
theorem tram_speed (v : ℝ) 
  (h1 : v * 2 = v * 2)  -- Tram passes observer in 2 seconds
  (h2 : v * 10 = 96 + v * 2)  -- Tram traverses 96-meter tunnel in 10 seconds
  : v = 12 := by
  sorry

end NUMINAMATH_CALUDE_tram_speed_l715_71558


namespace NUMINAMATH_CALUDE_sqrt_102_between_consecutive_integers_l715_71591

theorem sqrt_102_between_consecutive_integers : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 102 ∧ 
  Real.sqrt 102 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 110 := by
sorry

end NUMINAMATH_CALUDE_sqrt_102_between_consecutive_integers_l715_71591


namespace NUMINAMATH_CALUDE_quadratic_sum_l715_71524

/-- A quadratic function passing through two given points -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  r : ℝ
  point1 : p + q + r = 5
  point2 : 4*p + 2*q + r = 3

/-- The theorem stating that p+q+2r equals 10 for the given quadratic function -/
theorem quadratic_sum (g : QuadraticFunction) : g.p + g.q + 2*g.r = 10 := by
  sorry

#check quadratic_sum

end NUMINAMATH_CALUDE_quadratic_sum_l715_71524


namespace NUMINAMATH_CALUDE_triangle_inequality_squared_l715_71584

/-- Given a triangle with sides a, b, and c, prove that (a^2 + b^2 + ab) / c^2 < 1 --/
theorem triangle_inequality_squared (a b c : ℝ) (h : 0 < c) (triangle : c < a + b) :
  (a^2 + b^2 + a*b) / c^2 < 1 := by
  sorry

#check triangle_inequality_squared

end NUMINAMATH_CALUDE_triangle_inequality_squared_l715_71584


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l715_71542

def N : Matrix (Fin 2) (Fin 2) ℚ := !![4, 0; 2, -6]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ c = 1/24 ∧ d = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l715_71542


namespace NUMINAMATH_CALUDE_investment_ratio_l715_71531

theorem investment_ratio (p q : ℝ) (h1 : p > 0) (h2 : q > 0) : 
  (p * 10) / (q * 20) = 7 / 10 → p / q = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l715_71531


namespace NUMINAMATH_CALUDE_union_of_sets_l715_71528

theorem union_of_sets : 
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l715_71528


namespace NUMINAMATH_CALUDE_vector_operation_result_l715_71548

def vector_operation : ℝ × ℝ := sorry

theorem vector_operation_result :
  vector_operation = (-3, 32) := by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l715_71548


namespace NUMINAMATH_CALUDE_positive_difference_theorem_l715_71506

theorem positive_difference_theorem : ∃ (x : ℝ), x > 0 ∧ x = |((8^2 * 8^2) / 8) - ((8^2 + 8^2) / 8)| ∧ x = 496 := by
  sorry

end NUMINAMATH_CALUDE_positive_difference_theorem_l715_71506


namespace NUMINAMATH_CALUDE_picnic_difference_l715_71566

/-- Proves that the difference between adults and children at a picnic is 20 --/
theorem picnic_difference (total : ℕ) (men : ℕ) (women : ℕ) (adults : ℕ) (children : ℕ) : 
  total = 200 → 
  men = women + 20 → 
  men = 65 → 
  total = adults + children → 
  adults = men + women → 
  adults - children = 20 := by
sorry

end NUMINAMATH_CALUDE_picnic_difference_l715_71566


namespace NUMINAMATH_CALUDE_no_perfect_square_131_base_n_l715_71579

theorem no_perfect_square_131_base_n : 
  ¬ ∃ (n : ℤ), 4 ≤ n ∧ n ≤ 12 ∧ ∃ (k : ℤ), n^2 + 3*n + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_131_base_n_l715_71579


namespace NUMINAMATH_CALUDE_platform_pillar_height_l715_71550

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular platform with pillars -/
structure Platform where
  P : Point3D
  Q : Point3D
  R : Point3D
  S : Point3D
  slopeAngle : ℝ
  pillarHeightP : ℝ
  pillarHeightQ : ℝ
  pillarHeightR : ℝ

/-- The height of the pillar at point S -/
def pillarHeightS (p : Platform) : ℝ :=
  sorry

theorem platform_pillar_height
  (p : Platform)
  (h_PQ : p.Q.x - p.P.x = 10)
  (h_PR : p.R.y - p.P.y = 15)
  (h_slope : p.slopeAngle = π / 6)
  (h_heightP : p.pillarHeightP = 7)
  (h_heightQ : p.pillarHeightQ = 10)
  (h_heightR : p.pillarHeightR = 12) :
  pillarHeightS p = 7.5 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_platform_pillar_height_l715_71550


namespace NUMINAMATH_CALUDE_first_question_percentage_l715_71512

theorem first_question_percentage
  (second_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : second_correct = 49)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 32)
  : ∃ first_correct : ℝ, first_correct = 63 := by
  sorry

end NUMINAMATH_CALUDE_first_question_percentage_l715_71512


namespace NUMINAMATH_CALUDE_total_wood_pieces_l715_71540

/-- The number of pieces of wood that can be contained in one sack -/
def sack_capacity : ℕ := 20

/-- The number of sacks filled with wood -/
def filled_sacks : ℕ := 4

/-- Theorem stating that the total number of wood pieces gathered is equal to
    the product of sack capacity and the number of filled sacks -/
theorem total_wood_pieces :
  sack_capacity * filled_sacks = 80 := by sorry

end NUMINAMATH_CALUDE_total_wood_pieces_l715_71540


namespace NUMINAMATH_CALUDE_isosceles_triangle_height_l715_71575

theorem isosceles_triangle_height (s : ℝ) (h : ℝ) : 
  (1/2 : ℝ) * s * h = 2 * s^2 → h = 4 * s := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_height_l715_71575


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l715_71570

/-- The volume of a right circular cone formed by rolling up a half-sector of a circle --/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let sector_arc_length := r * π
  let cone_base_radius := sector_arc_length / (2 * π)
  let cone_slant_height := r
  let cone_height := Real.sqrt (cone_slant_height^2 - cone_base_radius^2)
  let cone_volume := (1/3) * π * cone_base_radius^2 * cone_height
  cone_volume = 9 * π * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l715_71570


namespace NUMINAMATH_CALUDE_inequality_equivalence_l715_71556

-- Define the solution set
def solution_set : Set ℝ := {x | x < -1 ∨ x > 1}

-- Statement of the theorem
theorem inequality_equivalence (x : ℝ) : x > (1 / x) ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l715_71556


namespace NUMINAMATH_CALUDE_one_fifths_in_nine_fifths_l715_71526

theorem one_fifths_in_nine_fifths : (9 : ℚ) / 5 / (1 / 5) = 9 := by
  sorry

end NUMINAMATH_CALUDE_one_fifths_in_nine_fifths_l715_71526


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l715_71598

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l715_71598


namespace NUMINAMATH_CALUDE_equivalent_discount_l715_71529

theorem equivalent_discount (original_price : ℝ) (first_discount second_discount : ℝ) :
  first_discount = 0.2 →
  second_discount = 0.25 →
  original_price * (1 - first_discount) * (1 - second_discount) = original_price * (1 - 0.4) :=
by
  sorry

end NUMINAMATH_CALUDE_equivalent_discount_l715_71529


namespace NUMINAMATH_CALUDE_equilateral_triangle_properties_l715_71544

theorem equilateral_triangle_properties (side : ℝ) (h : side = 20) :
  let height := side * (Real.sqrt 3) / 2
  let half_side := side / 2
  height = 10 * Real.sqrt 3 ∧ half_side = 10 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_properties_l715_71544


namespace NUMINAMATH_CALUDE_rectangle_division_condition_l715_71568

/-- A rectangle can be divided into two unequal but similar rectangles if and only if its longer side is more than twice the length of its shorter side. -/
theorem rectangle_division_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≥ b) :
  (∃ x : ℝ, 0 < x ∧ x < a ∧ x * b = (a - x) * x) ↔ a > 2 * b :=
sorry

end NUMINAMATH_CALUDE_rectangle_division_condition_l715_71568


namespace NUMINAMATH_CALUDE_sin_cos_function_at_pi_12_l715_71505

theorem sin_cos_function_at_pi_12 :
  let f : ℝ → ℝ := λ x ↦ Real.sin x ^ 4 - Real.cos x ^ 4
  f (π / 12) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_function_at_pi_12_l715_71505


namespace NUMINAMATH_CALUDE_percent_decrease_l715_71589

theorem percent_decrease (original_price sale_price : ℝ) :
  original_price = 100 ∧ sale_price = 20 →
  (original_price - sale_price) / original_price * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_percent_decrease_l715_71589


namespace NUMINAMATH_CALUDE_quadratic_max_value_change_l715_71508

theorem quadratic_max_value_change (a b c : ℝ) (h_a : a < 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let max_value (a' : ℝ) := -b^2 / (4 * a') + c
  (max_value (a + 1) = max_value a + 27 / 2) →
  (max_value (a - 4) = max_value a - 9) →
  (max_value (a - 2) = max_value a - 27 / 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_change_l715_71508


namespace NUMINAMATH_CALUDE_seventh_term_is_24_l715_71551

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first_term : ℝ
  common_diff : ℝ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + seq.common_diff * (n - 1)

theorem seventh_term_is_24 (seq : ArithmeticSequence) 
  (h1 : seq.first_term = 12)
  (h2 : nth_term seq 4 = 18) :
  nth_term seq 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_24_l715_71551


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l715_71541

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in the form y = a(x - h)^2 + k -/
def Parabola.equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

/-- Shifts a parabola horizontally and vertically -/
def Parabola.shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

/-- The original parabola y = -2x^2 -/
def original_parabola : Parabola :=
  { a := -2, h := 0, k := 0 }

/-- Theorem stating that shifting the original parabola down 1 unit and right 3 units
    results in the equation y = -2(x - 3)^2 - 1 -/
theorem parabola_shift_theorem (x : ℝ) :
  (original_parabola.shift 3 (-1)).equation x = -2 * (x - 3)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l715_71541


namespace NUMINAMATH_CALUDE_total_painting_time_l715_71536

/-- Time to paint each type of flower in minutes -/
def lily_time : ℕ := 5
def rose_time : ℕ := 7
def orchid_time : ℕ := 3
def sunflower_time : ℕ := 10
def tulip_time : ℕ := 4
def vine_time : ℕ := 2
def peony_time : ℕ := 8

/-- Number of each type of flower to paint -/
def lily_count : ℕ := 23
def rose_count : ℕ := 15
def orchid_count : ℕ := 9
def sunflower_count : ℕ := 12
def tulip_count : ℕ := 18
def vine_count : ℕ := 30
def peony_count : ℕ := 27

/-- Theorem stating the total time to paint all flowers -/
theorem total_painting_time : 
  lily_time * lily_count + 
  rose_time * rose_count + 
  orchid_time * orchid_count + 
  sunflower_time * sunflower_count + 
  tulip_time * tulip_count + 
  vine_time * vine_count + 
  peony_time * peony_count = 715 := by
  sorry

end NUMINAMATH_CALUDE_total_painting_time_l715_71536


namespace NUMINAMATH_CALUDE_alice_minimum_score_l715_71559

def minimum_score (scores : List Float) (target_average : Float) (total_terms : Nat) : Float :=
  let sum_scores := scores.sum
  let remaining_terms := total_terms - scores.length
  (target_average * total_terms.toFloat - sum_scores) / remaining_terms.toFloat

theorem alice_minimum_score :
  let alice_scores := [84, 88, 82, 79]
  let target_average := 85
  let total_terms := 5
  minimum_score alice_scores target_average total_terms = 92 := by
  sorry

end NUMINAMATH_CALUDE_alice_minimum_score_l715_71559


namespace NUMINAMATH_CALUDE_intersection_of_sets_l715_71538

theorem intersection_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | x^2 - x < 0} →
  B = {x : ℝ | -2 < x ∧ x < 2} →
  A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l715_71538


namespace NUMINAMATH_CALUDE_algebraic_identities_l715_71561

theorem algebraic_identities (m n x y z : ℝ) : 
  ((m + 2*n) - (m - 2*n) = 4*n) ∧
  (2*(x - 3) - (-x + 4) = 3*x - 10) ∧
  (2*x - 3*(x - 2*y + 3*x) + 2*(3*x - 3*y + 2*z) = -4*x + 4*z) ∧
  (8*m^2 - (4*m^2 - 2*m - 4*(2*m^2 - 5*m)) = 12*m^2 - 18*m) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l715_71561


namespace NUMINAMATH_CALUDE_parabola_vertex_l715_71534

/-- The parabola is defined by the equation y = 2(x-3)^2 + 1 -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 + 1

/-- The vertex of the parabola has coordinates (3, 1) -/
theorem parabola_vertex : ∃ (x y : ℝ), parabola x y ∧ x = 3 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l715_71534


namespace NUMINAMATH_CALUDE_sprint_medal_awards_l715_71518

/-- The number of ways to award medals in the international sprinting event -/
def medal_award_ways (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) 
  (americans_winning : ℕ) : ℕ :=
  -- The actual calculation would go here
  216

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem sprint_medal_awards : 
  medal_award_ways 10 4 3 2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sprint_medal_awards_l715_71518


namespace NUMINAMATH_CALUDE_minimum_dice_value_l715_71511

theorem minimum_dice_value (X : ℕ) : (1 + 5 + X > 2 + 4 + 5) ↔ X ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_minimum_dice_value_l715_71511
