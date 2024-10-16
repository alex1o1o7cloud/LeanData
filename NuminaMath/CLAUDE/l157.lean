import Mathlib

namespace NUMINAMATH_CALUDE_price_difference_l157_15774

def original_price : ℚ := 150
def tax_rate : ℚ := 0.07
def discount_rate : ℚ := 0.25
def service_charge_rate : ℚ := 0.05

def ann_price : ℚ :=
  original_price * (1 + tax_rate) * (1 - discount_rate) * (1 + service_charge_rate)

def ben_price : ℚ :=
  original_price * (1 - discount_rate) * (1 + tax_rate)

theorem price_difference :
  ann_price - ben_price = 6.01875 := by sorry

end NUMINAMATH_CALUDE_price_difference_l157_15774


namespace NUMINAMATH_CALUDE_distance_A_P_main_theorem_l157_15765

/-- A rectangle with two equilateral triangles positioned on its sides -/
structure TrianglesOnRectangle where
  /-- The length of side YC of rectangle YQZC -/
  yc : ℝ
  /-- The length of side CZ of rectangle YQZC -/
  cz : ℝ
  /-- The side length of equilateral triangles ABC and PQR -/
  triangle_side : ℝ
  /-- Assumption that YC = 8 -/
  yc_eq : yc = 8
  /-- Assumption that CZ = 15 -/
  cz_eq : cz = 15
  /-- Assumption that the side length of triangles is 9 -/
  triangle_side_eq : triangle_side = 9

/-- The distance between points A and P is 10 -/
theorem distance_A_P (t : TrianglesOnRectangle) : ℝ :=
  10

#check distance_A_P

/-- The main theorem stating that the distance between A and P is 10 -/
theorem main_theorem (t : TrianglesOnRectangle) : distance_A_P t = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_P_main_theorem_l157_15765


namespace NUMINAMATH_CALUDE_higher_interest_rate_theorem_l157_15754

/-- Given a principal amount, two interest rates, and a time period,
    calculate the difference in interest earned between the two rates. -/
def interest_difference (principal : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) : ℝ :=
  principal * rate1 * time - principal * rate2 * time

theorem higher_interest_rate_theorem (R : ℝ) :
  interest_difference 5000 (R / 100) (12 / 100) 2 = 600 → R = 18 := by
  sorry

end NUMINAMATH_CALUDE_higher_interest_rate_theorem_l157_15754


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l157_15763

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (a + I) * (1 - 2*I) = b*I) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l157_15763


namespace NUMINAMATH_CALUDE_max_markable_nodes_6x6_l157_15779

/-- Represents a square grid -/
structure Grid :=
  (size : Nat)

/-- A node in the grid -/
structure Node :=
  (x : Nat)
  (y : Nat)

/-- Checks if a node is on the edge of the grid -/
def isEdgeNode (g : Grid) (n : Node) : Bool :=
  n.x = 0 || n.x = g.size || n.y = 0 || n.y = g.size

/-- Checks if a node is a corner node -/
def isCornerNode (g : Grid) (n : Node) : Bool :=
  (n.x = 0 || n.x = g.size) && (n.y = 0 || n.y = g.size)

/-- Counts the number of nodes in a grid -/
def nodeCount (g : Grid) : Nat :=
  (g.size + 1) * (g.size + 1)

/-- Theorem: The maximum number of markable nodes in a 6x6 grid is 45 -/
theorem max_markable_nodes_6x6 (g : Grid) (h : g.size = 6) :
  nodeCount g - (4 : Nat) = 45 := by
  sorry

#check max_markable_nodes_6x6

end NUMINAMATH_CALUDE_max_markable_nodes_6x6_l157_15779


namespace NUMINAMATH_CALUDE_angle_terminal_side_l157_15706

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α - β = k * 360

/-- The expression for angles with the same terminal side as -463° -/
def angle_expression (k : ℤ) : ℝ := k * 360 + 257

theorem angle_terminal_side :
  ∀ k : ℤ, same_terminal_side (angle_expression k) (-463) :=
by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l157_15706


namespace NUMINAMATH_CALUDE_range_of_a_l157_15735

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : DecreasingFunction f) 
  (h2 : f (1 - a) < f (2 * a - 1)) :
  0 < a ∧ a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l157_15735


namespace NUMINAMATH_CALUDE_fraction_equality_l157_15795

theorem fraction_equality : 
  (14 : ℚ) / 12 = 7 / 6 ∧
  (1 : ℚ) + 1 / 6 = 7 / 6 ∧
  (21 : ℚ) / 18 = 7 / 6 ∧
  (1 : ℚ) + 2 / 12 = 7 / 6 ∧
  (1 : ℚ) + 1 / 3 ≠ 7 / 6 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l157_15795


namespace NUMINAMATH_CALUDE_nested_radical_sqrt_18_l157_15731

theorem nested_radical_sqrt_18 :
  ∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x = (1 + Real.sqrt 73) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_sqrt_18_l157_15731


namespace NUMINAMATH_CALUDE_roots_sum_power_l157_15762

theorem roots_sum_power (c d : ℝ) : 
  c^2 - 5*c + 6 = 0 → d^2 - 5*d + 6 = 0 → c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_power_l157_15762


namespace NUMINAMATH_CALUDE_average_rate_of_change_f_on_0_2_l157_15715

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the interval [0, 2]
def a : ℝ := 0
def b : ℝ := 2

-- State the theorem
theorem average_rate_of_change_f_on_0_2 :
  (f b - f a) / (b - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_of_change_f_on_0_2_l157_15715


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l157_15733

theorem polynomial_evaluation : 
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 5
  x^2 + y^2 - z^2 + 3*x*y - z = -35 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l157_15733


namespace NUMINAMATH_CALUDE_road_completion_proof_l157_15773

def road_paving (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n+1 => road_paving n + 1 / road_paving n

theorem road_completion_proof :
  ∃ n : ℕ, n ≤ 5001 ∧ road_paving n ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_road_completion_proof_l157_15773


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l157_15790

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l157_15790


namespace NUMINAMATH_CALUDE_abs_is_even_and_increasing_l157_15742

def f (x : ℝ) := abs x

theorem abs_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_abs_is_even_and_increasing_l157_15742


namespace NUMINAMATH_CALUDE_vector_decomposition_l157_15767

/-- Given vectors in R^3 -/
def x : Fin 3 → ℝ := ![5, 15, 0]
def p : Fin 3 → ℝ := ![1, 0, 5]
def q : Fin 3 → ℝ := ![-1, 3, 2]
def r : Fin 3 → ℝ := ![0, -1, 1]

/-- The decomposition of x in terms of p, q, and r -/
theorem vector_decomposition :
  x = (4 : ℝ) • p - (1 : ℝ) • q - (18 : ℝ) • r := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l157_15767


namespace NUMINAMATH_CALUDE_reciprocal_difference_square_sum_product_difference_l157_15786

/-- The difference between the reciprocal of x and y is equal to 1/x - y, where x ≠ 0 -/
theorem reciprocal_difference (x y : ℝ) (h : x ≠ 0) :
  1 / x - y = 1 / x - y := by sorry

/-- The difference between the square of the sum of a and b and the product of a and b
    is equal to (a+b)^2 - ab -/
theorem square_sum_product_difference (a b : ℝ) :
  (a + b)^2 - a * b = (a + b)^2 - a * b := by sorry

end NUMINAMATH_CALUDE_reciprocal_difference_square_sum_product_difference_l157_15786


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l157_15746

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def yIntercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := -3, point := (0, 6) } →
  b.point = (3, -2) →
  yIntercept b = 7 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l157_15746


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l157_15769

theorem scientific_notation_equivalence : 26900000 = 2.69 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l157_15769


namespace NUMINAMATH_CALUDE_pentagon_point_reconstruction_l157_15724

/-- Given a pentagon ABCDE with extended sides, prove the relation between A and A', B', C', D' -/
theorem pentagon_point_reconstruction (A B C D E A' B' C' D' : ℝ × ℝ) : 
  A'B = 2 * AB → 
  B'C = BC → 
  C'D = CD → 
  D'E = 2 * DE → 
  A = (1/9 : ℝ) • A' + (2/9 : ℝ) • B' + (4/9 : ℝ) • C' + (8/9 : ℝ) • D' := by
  sorry


end NUMINAMATH_CALUDE_pentagon_point_reconstruction_l157_15724


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l157_15711

theorem tangent_line_to_parabola (x y : ℝ) :
  y = x^2 →                                    -- Condition: parabola equation
  (∃ k : ℝ, k * x - y + 4 = 0) →               -- Condition: parallel line exists
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧        -- Tangent line equation
               a / b = 2 ∧                     -- Parallel to given line
               (∃ x₀ y₀ : ℝ, y₀ = x₀^2 ∧       -- Point on parabola
                             a * x₀ + b * y₀ + c = 0 ∧  -- Point on tangent line
                             2 * x₀ = (y₀ - y) / (x₀ - x))) →  -- Derivative condition
  2 * x - y - 1 = 0 :=                         -- Conclusion: specific tangent line equation
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l157_15711


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l157_15729

theorem complex_equation_solutions :
  ∃! (s : Finset ℂ), (∀ z ∈ s, Complex.abs z < 15 ∧ Complex.exp z = (z - 2) / (z + 2)) ∧ Finset.card s = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l157_15729


namespace NUMINAMATH_CALUDE_apple_ratio_l157_15752

theorem apple_ratio (blue_apples : ℕ) (yellow_apples : ℕ) : 
  blue_apples = 5 →
  yellow_apples + blue_apples - (yellow_apples + blue_apples) / 5 = 12 →
  yellow_apples / blue_apples = 2 :=
by sorry

end NUMINAMATH_CALUDE_apple_ratio_l157_15752


namespace NUMINAMATH_CALUDE_best_washing_effect_and_full_capacity_l157_15716

-- Define the constants
def drum_capacity : Real := 25
def current_clothes : Real := 4.92
def current_detergent_scoops : Nat := 3
def scoop_weight : Real := 0.02
def water_per_scoop : Real := 5

-- Define the variables for additional detergent and water
def additional_detergent : Real := 0.02
def additional_water : Real := 20

-- Theorem statement
theorem best_washing_effect_and_full_capacity : 
  -- The total weight equals the drum capacity
  current_clothes + (current_detergent_scoops * scoop_weight) + additional_detergent + additional_water = drum_capacity ∧
  -- The ratio of water to detergent is correct for best washing effect
  (current_detergent_scoops * scoop_weight + additional_detergent) / 
    (additional_water + water_per_scoop * current_detergent_scoops) = 1 / water_per_scoop :=
by sorry

end NUMINAMATH_CALUDE_best_washing_effect_and_full_capacity_l157_15716


namespace NUMINAMATH_CALUDE_dan_marbles_count_l157_15725

/-- The total number of marbles Dan has after receiving red marbles from Mary -/
def total_marbles (violet_marbles red_marbles : ℕ) : ℕ :=
  violet_marbles + red_marbles

/-- Theorem stating that Dan has 78 marbles in total -/
theorem dan_marbles_count :
  total_marbles 64 14 = 78 := by
  sorry

end NUMINAMATH_CALUDE_dan_marbles_count_l157_15725


namespace NUMINAMATH_CALUDE_ball_trajectory_l157_15740

-- Define the quadratic function
def f (t : ℚ) : ℚ := -4.9 * t^2 + 7 * t + 10

-- State the theorem
theorem ball_trajectory :
  f (5/7) = 15 ∧
  f (10/7) = 0 ∧
  ∀ t : ℚ, 5/7 < t → t < 10/7 → f t ≠ 15 ∧ f t ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ball_trajectory_l157_15740


namespace NUMINAMATH_CALUDE_second_strategy_more_economical_l157_15753

/-- Proves that the second purchasing strategy (constant money spent) is more economical than
    the first strategy (constant quantity purchased) for two purchases of the same item. -/
theorem second_strategy_more_economical (p₁ p₂ x y : ℝ) 
    (hp₁ : p₁ > 0) (hp₂ : p₂ > 0) (hx : x > 0) (hy : y > 0) :
  (2 * p₁ * p₂) / (p₁ + p₂) ≤ (p₁ + p₂) / 2 := by
  sorry

#check second_strategy_more_economical

end NUMINAMATH_CALUDE_second_strategy_more_economical_l157_15753


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l157_15709

theorem least_number_divisible_by_five_primes : 
  ∃ n : ℕ, (n > 0) ∧ 
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ q₁ q₂ q₃ q₄ q₅ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0) → 
    m ≥ n) ∧
  n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l157_15709


namespace NUMINAMATH_CALUDE_dog_catch_sheep_time_dog_catch_sheep_problem_l157_15726

/-- The time it takes for a dog to catch a sheep under specific conditions -/
theorem dog_catch_sheep_time (sheep_speed dog_speed : ℝ) (initial_distance : ℝ) 
  (dog_run_distance : ℝ) (dog_rest_time : ℝ) : ℝ :=
by
  sorry

/-- The main theorem proving the time it takes for the dog to catch the sheep -/
theorem dog_catch_sheep_problem : 
  dog_catch_sheep_time 18 30 480 100 3 = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_dog_catch_sheep_time_dog_catch_sheep_problem_l157_15726


namespace NUMINAMATH_CALUDE_eighth_term_of_arithmetic_sequence_l157_15720

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the second term is 17 and the fifth term is 19,
    the eighth term is 21. -/
theorem eighth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_second_term : a 2 = 17)
  (h_fifth_term : a 5 = 19) :
  a 8 = 21 := by
  sorry


end NUMINAMATH_CALUDE_eighth_term_of_arithmetic_sequence_l157_15720


namespace NUMINAMATH_CALUDE_proportion_solution_l157_15743

theorem proportion_solution (x : ℝ) (h : (3/4) / x = 5/8) : x = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l157_15743


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l157_15793

-- Define the set M as the domain of y = log(1-x)
def M : Set ℝ := {x : ℝ | x < 1}

-- Define the set N = {y | y = e^x, x ∈ ℝ}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.exp x}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l157_15793


namespace NUMINAMATH_CALUDE_cost_of_pens_l157_15782

/-- Given that a box of 150 pens costs $45, prove that 4500 pens cost $1350 -/
theorem cost_of_pens (box_size : ℕ) (box_cost : ℚ) (num_pens : ℕ) :
  box_size = 150 →
  box_cost = 45 →
  num_pens = 4500 →
  (num_pens : ℚ) * (box_cost / box_size) = 1350 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_pens_l157_15782


namespace NUMINAMATH_CALUDE_selection_schemes_count_l157_15738

def num_students : ℕ := 6
def num_tasks : ℕ := 4
def num_restricted_students : ℕ := 2

theorem selection_schemes_count :
  (num_students.factorial / (num_students - num_tasks).factorial) -
  (num_restricted_students * (num_students - 1).factorial / (num_students - num_tasks).factorial) = 240 :=
by sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l157_15738


namespace NUMINAMATH_CALUDE_student_committee_candidates_l157_15778

theorem student_committee_candidates :
  ∃ n : ℕ, 
    n > 0 ∧ 
    n * (n - 1) = 132 ∧ 
    (∀ m : ℕ, m > 0 ∧ m * (m - 1) = 132 → m = n) ∧
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_student_committee_candidates_l157_15778


namespace NUMINAMATH_CALUDE_tracy_candies_l157_15701

theorem tracy_candies (x : ℕ) : x = 68 :=
  -- Initial number of candies
  have h1 : x > 0 := by sorry

  -- After eating 1/4, the remaining candies are divisible by 3 (for giving 1/3 to Rachel)
  have h2 : ∃ k : ℕ, 3 * k = 3 * x / 4 := by sorry

  -- After giving 1/3 to Rachel, the remaining candies are even (for Tracy and mom to eat 12 each)
  have h3 : ∃ m : ℕ, 2 * m = x / 2 := by sorry

  -- After Tracy and mom eat 12 each, the remaining candies are between 7 and 11
  have h4 : 7 ≤ x / 2 - 24 ∧ x / 2 - 24 ≤ 11 := by sorry

  -- Final number of candies is 5
  have h5 : ∃ b : ℕ, 2 ≤ b ∧ b ≤ 6 ∧ x / 2 - 24 - b = 5 := by sorry

  sorry

end NUMINAMATH_CALUDE_tracy_candies_l157_15701


namespace NUMINAMATH_CALUDE_triangle_inequality_l157_15777

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = Real.pi

-- Theorem statement
theorem triangle_inequality (t : Triangle) (x y z : Real) :
  x^2 + y^2 + z^2 ≥ 2*x*y*(Real.cos t.C) + 2*y*z*(Real.cos t.A) + 2*z*x*(Real.cos t.B) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l157_15777


namespace NUMINAMATH_CALUDE_y_share_l157_15788

theorem y_share (total : ℝ) (x_share y_share z_share : ℝ) : 
  total = 210 →
  y_share = 0.45 * x_share →
  z_share = 0.30 * x_share →
  total = x_share + y_share + z_share →
  y_share = 54 := by
  sorry

end NUMINAMATH_CALUDE_y_share_l157_15788


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l157_15744

theorem indefinite_integral_proof (x : ℝ) (h : x > 0) : 
  (deriv (λ x => -3 * (1 + x^(4/3))^(5/3) / (4 * x^(4/3))) x) = 
    (1 + x^(4/3))^(2/3) / (x^2 * x^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l157_15744


namespace NUMINAMATH_CALUDE_inequality_proof_l157_15791

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 1/b^3 - 1) * (b^3 + 1/c^3 - 1) * (c^3 + 1/a^3 - 1) ≤ (a*b*c + 1/(a*b*c) - 1)^3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l157_15791


namespace NUMINAMATH_CALUDE_plane_contains_points_and_satisfies_constraints_l157_15768

def point1 : ℝ × ℝ × ℝ := (2, -1, 3)
def point2 : ℝ × ℝ × ℝ := (0, -1, 5)
def point3 : ℝ × ℝ × ℝ := (-2, -3, 4)

def plane_equation (x y z : ℝ) : Prop := 2*x + 5*y - 2*z + 7 = 0

theorem plane_contains_points_and_satisfies_constraints :
  (plane_equation point1.1 point1.2.1 point1.2.2) ∧
  (plane_equation point2.1 point2.2.1 point2.2.2) ∧
  (plane_equation point3.1 point3.2.1 point3.2.2) ∧
  (2 > 0) ∧
  (Nat.gcd (Nat.gcd (Nat.gcd 2 5) 2) 7 = 1) :=
sorry

end NUMINAMATH_CALUDE_plane_contains_points_and_satisfies_constraints_l157_15768


namespace NUMINAMATH_CALUDE_rational_floor_equality_l157_15784

theorem rational_floor_equality :
  ∃ (c d : ℤ), d < 100 ∧ d > 0 ∧
    ∀ k : ℕ, k ∈ Finset.range 100 → k > 0 →
      ⌊k * (c : ℚ) / d⌋ = ⌊k * (73 : ℚ) / 100⌋ := by
  sorry

end NUMINAMATH_CALUDE_rational_floor_equality_l157_15784


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l157_15718

theorem arithmetic_geometric_mean_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h_mean : (a + b) / 2 = 3 * Real.sqrt (a * b)) : 
  ∃ (n : ℤ), n = 34 ∧ ∀ (m : ℤ), |a / b - n| ≤ |a / b - m| :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l157_15718


namespace NUMINAMATH_CALUDE_charity_donation_percentage_l157_15792

theorem charity_donation_percentage 
  (total_raised : ℝ)
  (num_organizations : ℕ)
  (amount_per_org : ℝ)
  (h1 : total_raised = 2500)
  (h2 : num_organizations = 8)
  (h3 : amount_per_org = 250) :
  (num_organizations * amount_per_org) / total_raised * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_charity_donation_percentage_l157_15792


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l157_15771

theorem right_triangle_hypotenuse : 
  ∀ (longer_side shorter_side hypotenuse : ℝ),
  longer_side > 0 →
  shorter_side > 0 →
  hypotenuse > 0 →
  hypotenuse = longer_side + 2 →
  shorter_side = longer_side - 7 →
  shorter_side ^ 2 + longer_side ^ 2 = hypotenuse ^ 2 →
  hypotenuse = 17 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l157_15771


namespace NUMINAMATH_CALUDE_abs_greater_than_two_necessary_not_sufficient_l157_15721

theorem abs_greater_than_two_necessary_not_sufficient :
  (∀ x : ℝ, x < -2 → |x| > 2) ∧
  ¬(∀ x : ℝ, |x| > 2 → x < -2) :=
by sorry

end NUMINAMATH_CALUDE_abs_greater_than_two_necessary_not_sufficient_l157_15721


namespace NUMINAMATH_CALUDE_daylight_duration_l157_15789

/-- Given a day with 24 hours and a daylight to nighttime ratio of 9:7, 
    the duration of daylight is 13.5 hours. -/
theorem daylight_duration (total_hours : ℝ) (daylight_ratio nighttime_ratio : ℕ) 
    (h1 : total_hours = 24)
    (h2 : daylight_ratio = 9)
    (h3 : nighttime_ratio = 7) :
  (daylight_ratio : ℝ) / (daylight_ratio + nighttime_ratio : ℝ) * total_hours = 13.5 := by
sorry

end NUMINAMATH_CALUDE_daylight_duration_l157_15789


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l157_15704

theorem smaller_root_of_equation :
  let f (x : ℚ) := (x - 7/8)^2 + (x - 1/4) * (x - 7/8)
  ∃ (r : ℚ), f r = 0 ∧ r < 9/16 ∧ f (9/16) = 0 :=
by sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l157_15704


namespace NUMINAMATH_CALUDE_root_sum_property_l157_15747

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 9*x^2 + 11*x - 1

-- Define the theorem
theorem root_sum_property (a b c : ℝ) (s : ℝ) : 
  f a = 0 → f b = 0 → f c = 0 → 
  s = Real.sqrt a + Real.sqrt b + Real.sqrt c → 
  s^4 - 18*s^2 - 8*s = -37 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_property_l157_15747


namespace NUMINAMATH_CALUDE_star_difference_equals_45_l157_15751

/-- The star operation defined as x ★ y = x^2y - 3x -/
def star (x y : ℝ) : ℝ := x^2 * y - 3 * x

/-- Theorem stating that (6 ★ 3) - (3 ★ 6) = 45 -/
theorem star_difference_equals_45 : star 6 3 - star 3 6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_equals_45_l157_15751


namespace NUMINAMATH_CALUDE_polynomial_degree_l157_15727

/-- The degree of the polynomial resulting from 
    (5x^3 + 2x^2 - x - 7)(2x^8 - 4x^6 + 3x^3 + 15) - (x^3 - 3)^4 is 12 -/
theorem polynomial_degree : ℕ := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_l157_15727


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l157_15710

theorem ac_squared_gt_bc_squared_implies_a_gt_b (a b c : ℝ) :
  a * c^2 > b * c^2 → a > b :=
by sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l157_15710


namespace NUMINAMATH_CALUDE_composition_gf_l157_15713

/-- Function f that transforms a point (m, n) to (m, -n) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Function g that transforms a point (m, n) to (-m, -n) -/
def g (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

/-- Theorem stating that g[f(-3, 2)] = (3, 2) -/
theorem composition_gf : g (f (-3, 2)) = (3, 2) := by sorry

end NUMINAMATH_CALUDE_composition_gf_l157_15713


namespace NUMINAMATH_CALUDE_no_integers_product_zeros_l157_15737

theorem no_integers_product_zeros : 
  ¬∃ (x y : ℤ), 
    (x % 10 ≠ 0) ∧ 
    (y % 10 ≠ 0) ∧ 
    (x * y = 100000) := by
  sorry

end NUMINAMATH_CALUDE_no_integers_product_zeros_l157_15737


namespace NUMINAMATH_CALUDE_triangle_property_l157_15705

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  a = 3 →
  b = 2 * Real.sqrt 6 →
  B = 2 * A →
  0 < A →
  A < π →
  0 < B →
  B < π →
  0 < C →
  C < π →
  A + B + C = π →
  a = 2 * (Real.sin (A / 2)) * (Real.sin (C / 2)) / Real.sin B →
  b = 2 * (Real.sin (B / 2)) * (Real.sin (C / 2)) / Real.sin A →
  c = 2 * (Real.sin (A / 2)) * (Real.sin (B / 2)) / Real.sin C →
  Real.cos A = Real.sqrt 6 / 3 ∧ c = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l157_15705


namespace NUMINAMATH_CALUDE_circle_radius_reduction_l157_15717

/-- Given a circle with an initial radius of 5 cm, if its area is reduced by 36%, the new radius will be 4 cm. -/
theorem circle_radius_reduction (π : ℝ) (h_π_pos : π > 0) : 
  let r₁ : ℝ := 5
  let A₁ : ℝ := π * r₁^2
  let A₂ : ℝ := 0.64 * A₁
  let r₂ : ℝ := Real.sqrt (A₂ / π)
  r₂ = 4 := by sorry

end NUMINAMATH_CALUDE_circle_radius_reduction_l157_15717


namespace NUMINAMATH_CALUDE_quadratic_root_meaningful_l157_15722

theorem quadratic_root_meaningful (x : ℝ) : 
  (∃ (y : ℝ), y = 2 / Real.sqrt (3 + x)) ↔ x > -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_meaningful_l157_15722


namespace NUMINAMATH_CALUDE_symmetry_probability_one_third_l157_15749

/-- A square grid with n^2 points -/
structure SquareGrid (n : ℕ) where
  points : Fin n → Fin n → Bool

/-- The center point of a square grid -/
def centerPoint (n : ℕ) : Fin n × Fin n :=
  (⟨n / 2, sorry⟩, ⟨n / 2, sorry⟩)

/-- A line of symmetry for a square grid -/
def isSymmetryLine (n : ℕ) (grid : SquareGrid n) (p q : Fin n × Fin n) : Prop :=
  sorry

/-- The number of symmetry lines through the center point -/
def numSymmetryLines (n : ℕ) (grid : SquareGrid n) : ℕ :=
  sorry

theorem symmetry_probability_one_third (grid : SquareGrid 11) :
  let center := centerPoint 11
  let totalPoints := 121
  let nonCenterPoints := totalPoints - 1
  let symmetryLines := numSymmetryLines 11 grid
  (symmetryLines : ℚ) / nonCenterPoints = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_symmetry_probability_one_third_l157_15749


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l157_15799

/-- The surface area of a sphere circumscribing a rectangular solid -/
theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (S : ℝ) :
  a = 3 →
  b = 4 →
  c = 5 →
  S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
  S = 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l157_15799


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l157_15750

theorem sufficient_not_necessary (p q : Prop) : 
  (¬(p ∨ q) → ¬p) ∧ ¬(∀ p q, ¬p → ¬(p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l157_15750


namespace NUMINAMATH_CALUDE_intersection_midpoint_l157_15780

/-- Given a straight line x - y = 2 intersecting a parabola y² = 4x at points A and B,
    the midpoint M of line segment AB has coordinates (4, 2). -/
theorem intersection_midpoint (A B M : ℝ × ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    A = (x₁, y₁) ∧ B = (x₂, y₂) ∧
    x₁ - y₁ = 2 ∧ x₂ - y₂ = 2 ∧
    y₁^2 = 4*x₁ ∧ y₂^2 = 4*x₂ ∧
    M = ((x₁ + x₂)/2, (y₁ + y₂)/2)) →
  M = (4, 2) := by
sorry


end NUMINAMATH_CALUDE_intersection_midpoint_l157_15780


namespace NUMINAMATH_CALUDE_cookie_days_count_l157_15748

-- Define the total number of school days
def total_days : ℕ := 5

-- Define the number of days with peanut butter sandwiches
def peanut_butter_days : ℕ := 2

-- Define the number of days with ham sandwiches
def ham_days : ℕ := 3

-- Define the number of days with cake
def cake_days : ℕ := 1

-- Define the probability of ham sandwich and cake on the same day
def ham_cake_prob : ℚ := 12 / 100

-- Theorem to prove
theorem cookie_days_count : 
  total_days - cake_days - peanut_butter_days = 2 :=
sorry

end NUMINAMATH_CALUDE_cookie_days_count_l157_15748


namespace NUMINAMATH_CALUDE_scientific_notation_of_12417_l157_15776

theorem scientific_notation_of_12417 : ∃ (a : ℝ) (n : ℤ), 
  12417 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.2417 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_12417_l157_15776


namespace NUMINAMATH_CALUDE_series_sum_l157_15745

theorem series_sum : ∑' n, (n : ℝ) / 5^n = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_series_sum_l157_15745


namespace NUMINAMATH_CALUDE_outfit_combinations_l157_15730

theorem outfit_combinations (tshirts pants hats : ℕ) 
  (h1 : tshirts = 8) 
  (h2 : pants = 6) 
  (h3 : hats = 3) : 
  tshirts * pants * hats = 144 := by
sorry

end NUMINAMATH_CALUDE_outfit_combinations_l157_15730


namespace NUMINAMATH_CALUDE_cow_calf_cost_problem_l157_15766

theorem cow_calf_cost_problem (total_cost calf_cost cow_cost : ℕ) : 
  total_cost = 990 →
  cow_cost = 8 * calf_cost →
  total_cost = cow_cost + calf_cost →
  cow_cost = 880 := by
sorry

end NUMINAMATH_CALUDE_cow_calf_cost_problem_l157_15766


namespace NUMINAMATH_CALUDE_range_of_a_l157_15761

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- Define the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingFunction f) 
  (h_inequality : f (1 - a) < f (2 * a - 1)) :
  0 < a ∧ a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l157_15761


namespace NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l157_15728

-- Define the function f(x) = 3x - x^3
def f (x : ℝ) : ℝ := 3*x - x^3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 - 3*x^2

-- Theorem for the tangent line at x = 2
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), m = -9 ∧ b = 18 ∧
  ∀ x, f x = m * (x - 2) + f 2 := by sorry

-- Theorem for monotonicity intervals
theorem monotonicity_intervals :
  (∀ x < -1, (f' x < 0)) ∧
  (∀ x ∈ Set.Ioo (-1) 1, (f' x > 0)) ∧
  (∀ x > 1, (f' x < 0)) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l157_15728


namespace NUMINAMATH_CALUDE_circle_tangent_at_origin_l157_15707

/-- A circle in the xy-plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- The equation of the circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  x^2 + y^2 + c.D * x + c.E * y + c.F = 0

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- A circle is tangent to the x-axis at the origin -/
def tangent_at_origin (c : Circle) : Prop :=
  c.equation origin.x origin.y ∧
  ∀ (p : Point), p.y = 0 → p = origin ∨ ¬c.equation p.x p.y

theorem circle_tangent_at_origin (c : Circle) :
  tangent_at_origin c → c.E ≠ 0 ∧ c.D = 0 ∧ c.F = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_at_origin_l157_15707


namespace NUMINAMATH_CALUDE_problem_statement_l157_15756

noncomputable def f (x : ℝ) : ℝ := (x / (x + 4)) * Real.exp (x + 2)

noncomputable def g (a x : ℝ) : ℝ := (Real.exp (x + 2) - a * x - 3 * a) / ((x + 2)^2)

theorem problem_statement :
  (∀ x > -2, x * Real.exp (x + 2) + x + 4 > 0) ∧
  (∀ a ∈ Set.Icc 0 1, ∃ min_x > -2, ∀ x > -2, g a min_x ≤ g a x) ∧
  (∃ h : ℝ → ℝ, (∀ a ∈ Set.Icc 0 1, ∃ min_x > -2, h a = g a min_x) ∧
    Set.range h = Set.Ioo (1/2) (Real.exp 2 / 4)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l157_15756


namespace NUMINAMATH_CALUDE_water_composition_ratio_l157_15797

theorem water_composition_ratio :
  ∀ (total_mass : ℝ) (hydrogen_mass : ℝ),
    total_mass = 117 →
    hydrogen_mass = 13 →
    (hydrogen_mass / (total_mass - hydrogen_mass) = 1 / 8) :=
by
  sorry

end NUMINAMATH_CALUDE_water_composition_ratio_l157_15797


namespace NUMINAMATH_CALUDE_ellipse_a_value_l157_15781

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) : Type :=
  (h_pos : 0 < b ∧ b < a)

/-- The foci of an ellipse -/
def foci (e : Ellipse a b) : ℝ × ℝ := sorry

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse a b) : Type :=
  (x y : ℝ)
  (on_ellipse : x^2 / a^2 + y^2 / b^2 = 1)

/-- The area of a triangle formed by a point on the ellipse and the foci -/
def triangle_area (e : Ellipse a b) (p : PointOnEllipse e) : ℝ := sorry

/-- The tangent of the angle PF₁F₂ -/
def tan_angle_PF1F2 (e : Ellipse a b) (p : PointOnEllipse e) : ℝ := sorry

/-- The tangent of the angle PF₂F₁ -/
def tan_angle_PF2F1 (e : Ellipse a b) (p : PointOnEllipse e) : ℝ := sorry

/-- Theorem: If there exists a point P on the ellipse satisfying the given conditions, 
    then the semi-major axis a equals √15/2 -/
theorem ellipse_a_value (a b : ℝ) (e : Ellipse a b) :
  (∃ p : PointOnEllipse e, 
    triangle_area e p = 1 ∧ 
    tan_angle_PF1F2 e p = 1/2 ∧ 
    tan_angle_PF2F1 e p = -2) →
  a = Real.sqrt 15 / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_a_value_l157_15781


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l157_15714

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l157_15714


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l157_15796

theorem algebraic_expression_value (a b : ℝ) (h : a - 2*b + 3 = 0) :
  5 + 2*b - a = 8 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l157_15796


namespace NUMINAMATH_CALUDE_parallelogram_area_26_14_l157_15760

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 26 cm and height 14 cm is 364 square centimeters -/
theorem parallelogram_area_26_14 : parallelogram_area 26 14 = 364 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_26_14_l157_15760


namespace NUMINAMATH_CALUDE_annual_earnings_difference_l157_15719

/-- Calculates the difference in annual earnings between a new job and an old job -/
theorem annual_earnings_difference
  (new_wage : ℝ)
  (new_hours : ℝ)
  (old_wage : ℝ)
  (old_hours : ℝ)
  (weeks_per_year : ℝ)
  (h1 : new_wage = 20)
  (h2 : new_hours = 40)
  (h3 : old_wage = 16)
  (h4 : old_hours = 25)
  (h5 : weeks_per_year = 52) :
  new_wage * new_hours * weeks_per_year - old_wage * old_hours * weeks_per_year = 20800 := by
  sorry

#check annual_earnings_difference

end NUMINAMATH_CALUDE_annual_earnings_difference_l157_15719


namespace NUMINAMATH_CALUDE_eliminate_denominators_l157_15785

theorem eliminate_denominators (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1) :
  (3 / (2 * x) = 1 / (x - 1)) ↔ (3 * x - 3 = 2 * x) :=
sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l157_15785


namespace NUMINAMATH_CALUDE_x_coordinate_difference_l157_15739

theorem x_coordinate_difference (m n k : ℝ) : 
  (m = 2*n + 5) → 
  (m + k = 2*(n + 2) + 5) → 
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_x_coordinate_difference_l157_15739


namespace NUMINAMATH_CALUDE_max_f_and_min_side_a_l157_15775

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 4 * Real.pi / 3) + 2 * (Real.cos x) ^ 2

theorem max_f_and_min_side_a :
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ 2) ∧
  (∀ (A B C a b c : ℝ),
    0 < A ∧ A < Real.pi →
    0 < B ∧ B < Real.pi →
    0 < C ∧ C < Real.pi →
    A + B + C = Real.pi →
    f (B + C) = 3 / 2 →
    b + c = 2 →
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
    a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_f_and_min_side_a_l157_15775


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l157_15755

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l157_15755


namespace NUMINAMATH_CALUDE_office_episodes_l157_15798

theorem office_episodes (total_episodes : ℕ) (wednesday_episodes : ℕ) (weeks : ℕ) 
  (h1 : total_episodes = 201)
  (h2 : wednesday_episodes = 2)
  (h3 : weeks = 67) :
  ∃ (monday_episodes : ℕ), 
    weeks * (monday_episodes + wednesday_episodes) = total_episodes ∧ 
    monday_episodes = 1 := by
  sorry

end NUMINAMATH_CALUDE_office_episodes_l157_15798


namespace NUMINAMATH_CALUDE_bryan_total_books_l157_15703

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 15

/-- The number of books in each bookshelf -/
def books_per_shelf : ℕ := 78

/-- The total number of books Bryan has -/
def total_books : ℕ := num_bookshelves * books_per_shelf

/-- Theorem stating that Bryan has 1170 books in total -/
theorem bryan_total_books : total_books = 1170 := by
  sorry

end NUMINAMATH_CALUDE_bryan_total_books_l157_15703


namespace NUMINAMATH_CALUDE_inequality_solution_set_l157_15770

theorem inequality_solution_set (x : ℝ) :
  (((1 - x) / (x + 1) ≤ 0) ∧ (x ≠ -1)) ↔ (x < -1 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l157_15770


namespace NUMINAMATH_CALUDE_horner_method_correct_l157_15708

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f_coeffs : List ℝ := [5, 4, 3, 2, 1, 0]

theorem horner_method_correct :
  horner_eval f_coeffs 3 = 1641 := by
  sorry

#eval horner_eval f_coeffs 3

end NUMINAMATH_CALUDE_horner_method_correct_l157_15708


namespace NUMINAMATH_CALUDE_max_value_of_f_l157_15723

def f (x : ℝ) : ℝ := -x^2 + 6*x - 10

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 4 → f x ≤ f c) ∧
  f c = -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l157_15723


namespace NUMINAMATH_CALUDE_first_level_teachers_selected_l157_15732

/-- Calculates the number of first-level teachers selected in a stratified sample -/
def stratifiedSampleFirstLevel (seniorTeachers firstLevelTeachers secondLevelTeachers sampleSize : ℕ) : ℕ :=
  (firstLevelTeachers * sampleSize) / (seniorTeachers + firstLevelTeachers + secondLevelTeachers)

/-- Theorem stating that the number of first-level teachers selected is 12 -/
theorem first_level_teachers_selected :
  stratifiedSampleFirstLevel 90 120 170 38 = 12 := by
  sorry

end NUMINAMATH_CALUDE_first_level_teachers_selected_l157_15732


namespace NUMINAMATH_CALUDE_ellipse_with_given_properties_l157_15783

/-- Represents an ellipse with center at the origin and foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  e : ℝ  -- Eccentricity

/-- The equation of an ellipse in standard form -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

theorem ellipse_with_given_properties :
  ∀ (E : Ellipse),
    E.b = 1 →  -- Half of minor axis length is 1
    E.e = Real.sqrt 2 / 2 →  -- Eccentricity is √2/2
    (∀ x y : ℝ, ellipse_equation E x y ↔ x^2 / 2 + y^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_with_given_properties_l157_15783


namespace NUMINAMATH_CALUDE_multiple_root_equation_l157_15734

/-- The equation x^4 + p^2*x + q = 0 has a multiple root if and only if p = 2 and q = 3, where p and q are positive prime numbers. -/
theorem multiple_root_equation (p q : ℕ) : 
  (Prime p ∧ Prime q ∧ 0 < p ∧ 0 < q) →
  (∃ (x : ℝ), (x^4 + p^2*x + q = 0 ∧ 
    ∃ (y : ℝ), y ≠ x ∧ y^4 + p^2*y + q = 0 ∧
    (∀ (z : ℝ), z^4 + p^2*z + q = 0 → z = x ∨ z = y))) ↔ 
  (p = 2 ∧ q = 3) :=
by sorry

end NUMINAMATH_CALUDE_multiple_root_equation_l157_15734


namespace NUMINAMATH_CALUDE_frog_jump_probability_l157_15772

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the square boundary -/
def square_boundary (p : Point) : Bool :=
  p.x = 0 ∨ p.x = 4 ∨ p.y = 0 ∨ p.y = 4

/-- Represents reaching a vertical side of the square -/
def vertical_side (p : Point) : Bool :=
  p.x = 0 ∨ p.x = 4

/-- Probability of ending on a vertical side when starting from a given point -/
noncomputable def prob_vertical_end (start : Point) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem frog_jump_probability :
  prob_vertical_end ⟨1, 2⟩ = 5/8 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l157_15772


namespace NUMINAMATH_CALUDE_circular_competition_rounds_l157_15741

theorem circular_competition_rounds (m : ℕ) (h : m ≥ 17) :
  ∃ (n : ℕ), n = m - 1 ∧
  (∀ (schedule : ℕ → Fin (2*m) → Fin (2*m) → Prop),
    (∀ (i : Fin (2*m)), ∀ (j : Fin (2*m)), i ≠ j → ∃ (k : Fin (2*m - 1)), schedule k i j) →
    (∀ (k : Fin (2*m - 1)), ∀ (i : Fin (2*m)), ∃! (j : Fin (2*m)), i ≠ j ∧ schedule k i j) →
    (∀ (a b c d : Fin (2*m)), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d →
      (∀ (k : Fin n), ¬(schedule k a b ∨ schedule k a c ∨ schedule k a d ∨ schedule k b c ∨ schedule k b d ∨ schedule k c d)) ∨
      (∃ (k₁ k₂ : Fin n), k₁ ≠ k₂ ∧
        ((schedule k₁ a b ∧ schedule k₂ c d) ∨
         (schedule k₁ a c ∧ schedule k₂ b d) ∨
         (schedule k₁ a d ∧ schedule k₂ b c))))) ∧
  (∀ (n' : ℕ), n' < n →
    ∃ (schedule : ℕ → Fin (2*m) → Fin (2*m) → Prop),
      (∀ (i : Fin (2*m)), ∀ (j : Fin (2*m)), i ≠ j → ∃ (k : Fin (2*m - 1)), schedule k i j) ∧
      (∀ (k : Fin (2*m - 1)), ∀ (i : Fin (2*m)), ∃! (j : Fin (2*m)), i ≠ j ∧ schedule k i j) ∧
      (∃ (a b c d : Fin (2*m)), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
        (∀ (k : Fin n'), ¬(schedule k a b ∨ schedule k a c ∨ schedule k a d ∨ schedule k b c ∨ schedule k b d ∨ schedule k c d)) ∧
        ¬(∃ (k₁ k₂ : Fin n'), k₁ ≠ k₂ ∧
          ((schedule k₁ a b ∧ schedule k₂ c d) ∨
           (schedule k₁ a c ∧ schedule k₂ b d) ∨
           (schedule k₁ a d ∧ schedule k₂ b c))))) :=
by
  sorry


end NUMINAMATH_CALUDE_circular_competition_rounds_l157_15741


namespace NUMINAMATH_CALUDE_watermelon_seeds_l157_15757

/-- Given a watermelon cut into 40 slices, with each slice having an equal number of black and white seeds,
    and a total of 1,600 seeds in the watermelon, prove that there are 20 black seeds in each slice. -/
theorem watermelon_seeds (slices : ℕ) (total_seeds : ℕ) (black_seeds_per_slice : ℕ) :
  slices = 40 →
  total_seeds = 1600 →
  total_seeds = 2 * slices * black_seeds_per_slice →
  black_seeds_per_slice = 20 := by
  sorry

#check watermelon_seeds

end NUMINAMATH_CALUDE_watermelon_seeds_l157_15757


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l157_15712

/-- Two lines are distinct if they are not equal -/
def distinct_lines (l m : Line) : Prop := l ≠ m

/-- Two planes are distinct if they are not equal -/
def distinct_planes (α β : Plane) : Prop := α ≠ β

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (α : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (α : Plane) : Prop := sorry

/-- Two planes are parallel -/
def planes_parallel (α β : Plane) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perpendicular (l m : Line) : Prop := sorry

theorem perpendicular_lines_from_parallel_planes 
  (l m : Line) (α β : Plane) 
  (h1 : distinct_lines l m)
  (h2 : distinct_planes α β)
  (h3 : planes_parallel α β)
  (h4 : line_perp_plane l α)
  (h5 : line_parallel_plane m β) :
  lines_perpendicular l m := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l157_15712


namespace NUMINAMATH_CALUDE_ellipse_k_range_l157_15794

/-- 
Given a real number k, if the equation x²/(9-k) + y²/(k-1) = 1 represents an ellipse 
with foci on the y-axis, then 5 < k < 9.
-/
theorem ellipse_k_range (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (9 - k) + y^2 / (k - 1) = 1) → -- equation represents an ellipse
  (9 - k > 0) →  -- condition for ellipse
  (k - 1 > 0) →  -- condition for ellipse
  (k - 1 > 9 - k) →  -- foci on y-axis condition
  (5 < k ∧ k < 9) := by
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l157_15794


namespace NUMINAMATH_CALUDE_alex_friends_count_l157_15787

def silk_problem (total_silk : ℕ) (silk_per_dress : ℕ) (dresses_made : ℕ) : ℕ :=
  let silk_used := dresses_made * silk_per_dress
  let silk_given := total_silk - silk_used
  silk_given / silk_per_dress

theorem alex_friends_count :
  silk_problem 600 5 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_alex_friends_count_l157_15787


namespace NUMINAMATH_CALUDE_no_18_pretty_below_1500_l157_15736

def is_m_pretty (n m : ℕ+) : Prop :=
  (Nat.divisors n).card = m ∧ m ∣ n

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem no_18_pretty_below_1500 :
  ∀ n : ℕ+,
  n < 1500 →
  (∃ (a b : ℕ) (k : ℕ+), 
    n = 2^a * 7^b * k ∧
    a ≥ 1 ∧
    b ≥ 1 ∧
    is_coprime k.val 14 ∧
    (Nat.divisors n.val).card = 18) →
  ¬(is_m_pretty n 18) :=
sorry

end NUMINAMATH_CALUDE_no_18_pretty_below_1500_l157_15736


namespace NUMINAMATH_CALUDE_f_lower_bound_l157_15758

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m / x + log x

theorem f_lower_bound (m : ℝ) (x : ℝ) (hm : m > 0) (hx : x > 0) :
  m * f m x ≥ 2 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_l157_15758


namespace NUMINAMATH_CALUDE_value_of_a_l157_15700

theorem value_of_a (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 7 * a^2 + 14 * a * b = a^3 + 2 * a^2 * b) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l157_15700


namespace NUMINAMATH_CALUDE_spelling_bee_points_l157_15764

theorem spelling_bee_points : 
  let max_points : ℝ := 7
  let dulce_points : ℝ := 5
  let val_points : ℝ := 4 * (max_points + dulce_points)
  let sarah_points : ℝ := 2 * dulce_points
  let steve_points : ℝ := 2.5 * (max_points + val_points)
  let team_points : ℝ := max_points + dulce_points + val_points + sarah_points + steve_points
  let opponents_points : ℝ := 200
  team_points - opponents_points = 7.5 := by sorry

end NUMINAMATH_CALUDE_spelling_bee_points_l157_15764


namespace NUMINAMATH_CALUDE_line_inclination_l157_15702

def line_equation (x y : ℝ) : Prop := y = x + 1

def angle_of_inclination (θ : ℝ) : Prop := θ = Real.arctan 1

theorem line_inclination :
  ∀ x y θ : ℝ, line_equation x y → angle_of_inclination θ → θ * (180 / Real.pi) = 45 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_l157_15702


namespace NUMINAMATH_CALUDE_friend_spent_more_l157_15759

theorem friend_spent_more (total : ℕ) (friend_spent : ℕ) (you_spent : ℕ) : 
  total = 11 → friend_spent = 7 → total = friend_spent + you_spent → friend_spent > you_spent →
  friend_spent - you_spent = 3 := by
sorry

end NUMINAMATH_CALUDE_friend_spent_more_l157_15759
