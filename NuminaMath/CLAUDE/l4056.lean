import Mathlib

namespace fixed_point_of_exponential_function_l4056_405645

/-- The function f(x) = a^(2x-1) + 1 passes through (1/2, 2) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(2*x - 1) + 1
  f (1/2) = 2 := by
  sorry

end fixed_point_of_exponential_function_l4056_405645


namespace sum_of_reciprocals_l4056_405693

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y - 1) :
  1 / x + 1 / y = 1 - 1 / (x * y) := by
  sorry

end sum_of_reciprocals_l4056_405693


namespace cosine_sum_range_inverse_tangent_sum_l4056_405631

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sum_angles : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (sine_law : a / Real.sin A = b / Real.sin B)
  (cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)

-- Part 1
theorem cosine_sum_range (t : Triangle) (h : t.B = π/3) :
  1/2 < Real.cos t.A + Real.cos t.C ∧ Real.cos t.A + Real.cos t.C ≤ 1 :=
sorry

-- Part 2
theorem inverse_tangent_sum (t : Triangle) 
  (h1 : t.b^2 = t.a * t.c) (h2 : Real.cos t.B = 4/5) :
  1 / Real.tan t.A + 1 / Real.tan t.C = 5/3 :=
sorry

end cosine_sum_range_inverse_tangent_sum_l4056_405631


namespace quadratic_equation_roots_ratio_l4056_405609

theorem quadratic_equation_roots_ratio (m : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r^2 - 4*r + m = 0 ∧ s^2 - 4*s + m = 0) → m = 3 := by
sorry

end quadratic_equation_roots_ratio_l4056_405609


namespace new_line_equation_l4056_405698

/-- Given a line y = mx + b, proves that a new line with half the slope
    and triple the y-intercept has the equation y = (m/2)x + 3b -/
theorem new_line_equation (m b : ℝ) :
  let original_line := fun x => m * x + b
  let new_line := fun x => (m / 2) * x + 3 * b
  ∀ x, new_line x = (m / 2) * x + 3 * b :=
by sorry

end new_line_equation_l4056_405698


namespace downstream_distance_is_16_l4056_405629

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  upstream_distance : ℝ
  swim_time : ℝ
  still_water_speed : ℝ

/-- Calculates the downstream distance given a swimming scenario -/
def downstream_distance (s : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the downstream distance is 16 km -/
theorem downstream_distance_is_16 (s : SwimmingScenario) 
  (h1 : s.upstream_distance = 10)
  (h2 : s.swim_time = 2)
  (h3 : s.still_water_speed = 6.5) :
  downstream_distance s = 16 := by
  sorry

end downstream_distance_is_16_l4056_405629


namespace snickers_cost_calculation_l4056_405658

/-- The cost of a single piece of Snickers -/
def snickers_cost : ℚ := 1.5

/-- The number of Snickers pieces Julia bought -/
def snickers_count : ℕ := 2

/-- The number of M&M's packs Julia bought -/
def mm_count : ℕ := 3

/-- The cost of a pack of M&M's in terms of Snickers pieces -/
def mm_cost_in_snickers : ℕ := 2

/-- The total amount Julia gave to the cashier -/
def total_paid : ℚ := 20

/-- The change Julia received -/
def change_received : ℚ := 8

theorem snickers_cost_calculation :
  snickers_cost * (snickers_count + mm_count * mm_cost_in_snickers) = total_paid - change_received :=
by sorry

end snickers_cost_calculation_l4056_405658


namespace jasons_commute_distance_l4056_405696

/-- Jason's commute to work problem -/
theorem jasons_commute_distance : ∀ (d1 d2 d3 d4 d5 : ℝ),
  d1 = 6 →                           -- Distance between first and second store
  d2 = d1 + (2/3 * d1) →             -- Distance between second and third store
  d3 = 4 →                           -- Distance from house to first store
  d4 = 4 →                           -- Distance from last store to work
  d5 = d1 + d2 + d3 + d4 →           -- Total commute distance
  d5 = 24 := by sorry

end jasons_commute_distance_l4056_405696


namespace complex_absolute_value_l4056_405699

theorem complex_absolute_value (t : ℝ) (h : t > 0) :
  Complex.abs (-5 + t * Complex.I) = 3 * Real.sqrt 13 → t = 2 * Real.sqrt 23 := by
  sorry

end complex_absolute_value_l4056_405699


namespace g_sum_zero_l4056_405682

def g (x : ℝ) : ℝ := x^2 - 2013*x

theorem g_sum_zero (a b : ℝ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 := by
  sorry

end g_sum_zero_l4056_405682


namespace system_solution_l4056_405688

theorem system_solution : ∃ (x y : ℚ), 
  (3 * x - 4 * y = -7) ∧ 
  (4 * x - 3 * y = 5) ∧ 
  (x = 41 / 7) ∧ 
  (y = 43 / 7) := by
  sorry

end system_solution_l4056_405688


namespace solution_set_correct_inequality_factorization_l4056_405646

/-- The solution set of the quadratic inequality ax^2 + (a-2)x - 2 ≤ 0 -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then Set.Ici (-1)
  else if a > 0 then Set.Icc (-1) (2/a)
  else if -2 < a ∧ a < 0 then Set.Iic (2/a) ∪ Set.Ici (-1)
  else if a < -2 then Set.Iic (-1) ∪ Set.Ici (2/a)
  else Set.univ

/-- Theorem stating that the solution_set function correctly solves the quadratic inequality -/
theorem solution_set_correct (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 + (a - 2) * x - 2 ≤ 0 := by
  sorry

/-- Theorem stating that the quadratic inequality can be rewritten as a product of linear factors -/
theorem inequality_factorization (a : ℝ) (x : ℝ) :
  a * x^2 + (a - 2) * x - 2 = (a * x - 2) * (x + 1) := by
  sorry

end solution_set_correct_inequality_factorization_l4056_405646


namespace polynomial_form_l4056_405674

/-- A real-coefficient polynomial function -/
def RealPolynomial := ℝ → ℝ

/-- The condition that needs to be satisfied by the polynomial -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), a * b + b * c + c * a = 0 →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem polynomial_form (P : RealPolynomial) 
    (h : SatisfiesCondition P) : 
    ∃ (α β : ℝ), ∀ (x : ℝ), P x = α * x^4 + β * x^2 := by
  sorry

end polynomial_form_l4056_405674


namespace derivative_sum_at_points_l4056_405686

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + 2*x + 5

-- Define the derivative of f
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + 2

-- Theorem statement
theorem derivative_sum_at_points (m : ℝ) : f' m 2 + f' m (-2) = 28 := by
  sorry

end derivative_sum_at_points_l4056_405686


namespace geometric_sequence_common_ratio_l4056_405665

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_prod : a 5 * a 11 = 3)
  (h_sum : a 3 + a 13 = 4) :
  ∃ r : ℝ, (r = 3 ∨ r = -3) ∧ ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

end geometric_sequence_common_ratio_l4056_405665


namespace trig_ratio_equality_l4056_405650

theorem trig_ratio_equality (α : Real) (h : Real.tan α = 2 * Real.tan (π / 5)) :
  Real.cos (α - 3 * π / 10) / Real.sin (α - π / 5) = 3 := by
  sorry

end trig_ratio_equality_l4056_405650


namespace product_of_primes_with_sum_85_l4056_405655

theorem product_of_primes_with_sum_85 (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 85 → p * q = 166 := by
sorry

end product_of_primes_with_sum_85_l4056_405655


namespace base8_12345_to_decimal_l4056_405633

/-- Converts a base-8 number to its decimal (base-10) equivalent -/
def base8_to_decimal (d1 d2 d3 d4 d5 : ℕ) : ℕ :=
  d1 * 8^4 + d2 * 8^3 + d3 * 8^2 + d4 * 8^1 + d5 * 8^0

/-- The decimal representation of 12345 in base-8 is 5349 -/
theorem base8_12345_to_decimal :
  base8_to_decimal 1 2 3 4 5 = 5349 := by
  sorry

end base8_12345_to_decimal_l4056_405633


namespace ben_remaining_money_l4056_405664

def calculate_remaining_money (initial amount : ℕ) (cheque debtor_payment maintenance_cost : ℕ) : ℕ :=
  initial - cheque + debtor_payment - maintenance_cost

theorem ben_remaining_money :
  calculate_remaining_money 2000 600 800 1200 = 1000 := by
  sorry

end ben_remaining_money_l4056_405664


namespace arithmetic_sequence_problem_l4056_405620

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 10) : 
  3 * a 5 + a 7 = 20 := by
sorry

end arithmetic_sequence_problem_l4056_405620


namespace wall_length_is_800_l4056_405654

-- Define the dimensions of a single brick
def brick_length : ℝ := 125
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the known dimensions of the wall
def wall_width : ℝ := 600
def wall_height : ℝ := 22.5

-- Define the number of bricks needed
def num_bricks : ℕ := 1280

-- Theorem statement
theorem wall_length_is_800 :
  ∃ (wall_length : ℝ),
    wall_length = 800 ∧
    (brick_length * brick_width * brick_height) * num_bricks =
    wall_length * wall_width * wall_height :=
by
  sorry


end wall_length_is_800_l4056_405654


namespace z_in_second_quadrant_l4056_405605

def z : ℂ := Complex.I + Complex.I^6

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end z_in_second_quadrant_l4056_405605


namespace sinusoidal_symmetry_center_l4056_405628

/-- Given a sinusoidal function with specific properties, prove that one of its symmetry centers has coordinates (-2π/3, 0) -/
theorem sinusoidal_symmetry_center 
  (f : ℝ → ℝ) 
  (ω φ : ℝ) 
  (h1 : ∀ x, f x = Real.sin (ω * x + φ))
  (h2 : ω > 0)
  (h3 : |φ| < π / 2)
  (h4 : ∀ x, f (x + 4 * π) = f x)
  (h5 : ∀ t, t > 0 → (∀ x, f (x + t) = f x) → t ≥ 4 * π)
  (h6 : f (π / 3) = 1) :
  ∃ k : ℤ, f (x + (-2 * π / 3)) = -f (-x + (-2 * π / 3)) := by
  sorry

end sinusoidal_symmetry_center_l4056_405628


namespace min_value_theorem_l4056_405611

theorem min_value_theorem (a b c d : ℝ) :
  (|b + a^2 - 4 * Real.log a| + |2 * c - d + 2| = 0) →
  ∃ (min_value : ℝ), (∀ (a' b' c' d' : ℝ), 
    (|b' + a'^2 - 4 * Real.log a'| + |2 * c' - d' + 2| = 0) →
    ((a' - c')^2 + (b' - d')^2 ≥ min_value)) ∧
  min_value = 5 :=
by sorry

end min_value_theorem_l4056_405611


namespace card_problem_l4056_405692

/-- Given the number of cards for Brenda, Janet, and Mara, calculate the certain number -/
def certainNumber (brenda : ℕ) (janet : ℕ) (mara : ℕ) : ℕ :=
  mara + 40

theorem card_problem (brenda : ℕ) :
  let janet := brenda + 9
  let mara := 2 * janet
  brenda + janet + mara = 211 →
  certainNumber brenda janet mara = 150 := by
  sorry

#check card_problem

end card_problem_l4056_405692


namespace function_behavior_l4056_405613

def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_prop : ∀ x, f x = -f (2 - x))
  (h_decr : is_decreasing_on f 1 2) :
  is_increasing_on f (-2) (-1) ∧ is_increasing_on f 3 4 := by
  sorry

end function_behavior_l4056_405613


namespace product_equals_99999919_l4056_405667

theorem product_equals_99999919 : 103 * 97 * 10009 = 99999919 := by
  sorry

end product_equals_99999919_l4056_405667


namespace simplify_fraction_l4056_405634

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by
  sorry

end simplify_fraction_l4056_405634


namespace perpendicular_vectors_parallel_vectors_l4056_405636

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (-2, 1)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Define scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Theorem 1: k*a - b is perpendicular to a + 3*b when k = -13/5
theorem perpendicular_vectors (k : ℝ) : 
  dot_product (vec_sub (scalar_mul k a) b) (vec_add a (scalar_mul 3 b)) = 0 ↔ k = -13/5 := by
  sorry

-- Theorem 2: k*a - b is parallel to a + 3*b when k = -1/3
theorem parallel_vectors (k : ℝ) : 
  ∃ (t : ℝ), vec_sub (scalar_mul k a) b = scalar_mul t (vec_add a (scalar_mul 3 b)) ↔ k = -1/3 := by
  sorry

end perpendicular_vectors_parallel_vectors_l4056_405636


namespace toy_cost_l4056_405653

theorem toy_cost (initial_money : ℕ) (game_cost : ℕ) (num_toys : ℕ) :
  initial_money = 57 →
  game_cost = 27 →
  num_toys = 5 →
  (initial_money - game_cost) % num_toys = 0 →
  (initial_money - game_cost) / num_toys = 6 :=
by
  sorry

end toy_cost_l4056_405653


namespace parabola_directrix_l4056_405610

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Theorem statement
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola x y → directrix (x - 2) :=
sorry

end parabola_directrix_l4056_405610


namespace cafeteria_problem_l4056_405603

theorem cafeteria_problem (n : ℕ) (h : n = 6) :
  (∃ (max_days : ℕ) (avg_dishes : ℚ),
    max_days = 2^n ∧
    avg_dishes = n / 2 ∧
    max_days = 64 ∧
    avg_dishes = 3) := by
  sorry

end cafeteria_problem_l4056_405603


namespace intersection_line_equation_l4056_405670

/-- Given two lines L₁ and L₂ in the plane, and a third line L that intersects both L₁ and L₂,
    if the midpoint of the line segment formed by these intersections is the origin,
    then L has the equation x + 6y = 0. -/
theorem intersection_line_equation (L₁ L₂ L : Set (ℝ × ℝ)) :
  L₁ = {p : ℝ × ℝ | 4 * p.1 + p.2 + 6 = 0} →
  L₂ = {p : ℝ × ℝ | 3 * p.1 - 5 * p.2 - 6 = 0} →
  (∃ A B : ℝ × ℝ, A ∈ L ∩ L₁ ∧ B ∈ L ∩ L₂ ∧ (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 0) →
  L = {p : ℝ × ℝ | p.1 + 6 * p.2 = 0} :=
by sorry

end intersection_line_equation_l4056_405670


namespace elena_operation_l4056_405602

theorem elena_operation (x : ℝ) : (((3 * x + 5) - 3) * 2) / 2 = 17 → x = 5 := by
  sorry

end elena_operation_l4056_405602


namespace x_times_one_minus_f_equals_one_l4056_405680

/-- Given x = (3 + √8)^1001, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 1 -/
theorem x_times_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 8) ^ 1001
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by
  sorry

end x_times_one_minus_f_equals_one_l4056_405680


namespace quadratic_factorization_l4056_405652

theorem quadratic_factorization (x : ℝ) : x^2 - 5*x + 6 = (x - 2) * (x - 3) := by
  sorry

end quadratic_factorization_l4056_405652


namespace jemma_grasshoppers_l4056_405626

/-- The number of grasshoppers Jemma saw on her African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found under the plant -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshoppers : total_grasshoppers = 31 := by
  sorry

end jemma_grasshoppers_l4056_405626


namespace emily_walks_farther_l4056_405635

/-- The distance Troy walks to school (in meters) -/
def troy_distance : ℕ := 75

/-- The distance Emily walks to school (in meters) -/
def emily_distance : ℕ := 98

/-- The number of days -/
def days : ℕ := 5

/-- The additional distance Emily walks compared to Troy over the given number of days -/
def additional_distance : ℕ := 
  days * (2 * emily_distance - 2 * troy_distance)

theorem emily_walks_farther : additional_distance = 230 := by
  sorry

end emily_walks_farther_l4056_405635


namespace female_leader_probability_l4056_405630

theorem female_leader_probability (female_count male_count : ℕ) 
  (h1 : female_count = 4) 
  (h2 : male_count = 6) : 
  (female_count : ℚ) / (female_count + male_count) = 2 / 5 := by
  sorry

end female_leader_probability_l4056_405630


namespace fixed_point_of_parabolas_l4056_405690

/-- The function f_m that defines the family of parabolas -/
def f_m (m : ℝ) (x : ℝ) : ℝ := (m^2 + m + 1) * x^2 - 2 * (m^2 + 1) * x + m^2 - m + 1

/-- Theorem stating that (1, 0) is the fixed common point of all parabolas -/
theorem fixed_point_of_parabolas :
  ∀ m : ℝ, f_m m 1 = 1 := by sorry

end fixed_point_of_parabolas_l4056_405690


namespace expression_simplification_and_evaluation_l4056_405627

theorem expression_simplification_and_evaluation :
  let x : ℚ := 4
  ((1 / (x + 2) + 1) / ((x^2 + 6*x + 9) / (x^2 - 4))) = 2/7 := by
  sorry

end expression_simplification_and_evaluation_l4056_405627


namespace select_twelve_students_l4056_405644

/-- Represents the number of students in each course -/
structure CourseDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the sampling information -/
structure SamplingInfo where
  total_students : ℕ
  selected_students : ℕ

/-- Checks if the course distribution forms an arithmetic sequence with the given common difference -/
def is_arithmetic_sequence (dist : CourseDistribution) (diff : ℤ) : Prop :=
  dist.second = dist.first - diff ∧ dist.third = dist.second - diff

/-- Calculates the number of students to be selected from the first course -/
def students_to_select (dist : CourseDistribution) (info : SamplingInfo) : ℕ :=
  (dist.first * info.selected_students) / info.total_students

/-- Main theorem: Given the conditions, prove that 12 students should be selected from the first course -/
theorem select_twelve_students 
  (dist : CourseDistribution)
  (info : SamplingInfo)
  (h1 : dist.first + dist.second + dist.third = info.total_students)
  (h2 : info.total_students = 600)
  (h3 : info.selected_students = 30)
  (h4 : is_arithmetic_sequence dist (-40)) :
  students_to_select dist info = 12 := by
  sorry

end select_twelve_students_l4056_405644


namespace root_square_condition_l4056_405622

theorem root_square_condition (q : ℝ) : 
  (∃ a b : ℝ, a^2 - 12*a + q = 0 ∧ b^2 - 12*b + q = 0 ∧ (a = b^2 ∨ b = a^2)) ↔ 
  (q = -64 ∨ q = 27) := by
sorry

end root_square_condition_l4056_405622


namespace largest_zip_code_l4056_405617

def phone_number : List Nat := [4, 6, 5, 3, 2, 7, 1]

def is_valid_zip_code (zip : List Nat) : Prop :=
  zip.length = 4 ∧ 
  zip.toFinset.card = 4 ∧
  zip.sum = phone_number.sum

def zip_code_value (zip : List Nat) : Nat :=
  zip.foldl (fun acc d => acc * 10 + d) 0

theorem largest_zip_code :
  ∀ zip : List Nat, is_valid_zip_code zip →
  zip_code_value zip ≤ 9865 :=
sorry

end largest_zip_code_l4056_405617


namespace sodium_chloride_moles_l4056_405669

-- Define the chemical reaction components
structure ChemicalReaction where
  NaCl : ℕ  -- moles of Sodium chloride
  HNO3 : ℕ  -- moles of Nitric acid
  NaNO3 : ℕ  -- moles of Sodium nitrate
  HCl : ℕ   -- moles of Hydrochloric acid

-- Define the theorem
theorem sodium_chloride_moles (reaction : ChemicalReaction) :
  reaction.NaNO3 = 2 →  -- Condition 1
  reaction.HCl = 2 →    -- Condition 2
  reaction.HNO3 = reaction.NaNO3 →  -- Condition 3
  reaction.NaCl = 2 :=  -- Conclusion
by
  sorry  -- Proof is omitted as per instructions

end sodium_chloride_moles_l4056_405669


namespace fifth_term_of_geometric_sequence_l4056_405648

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : is_geometric_sequence a)
  (h_third_term : a 3 = 16)
  (h_seventh_term : a 7 = 2) :
  a 5 = 2 := by
sorry

end fifth_term_of_geometric_sequence_l4056_405648


namespace exists_distinct_subsequences_l4056_405695

/-- A binary sequence is a function from ℕ to Bool -/
def BinarySequence := ℕ → Bool

/-- Cyclic index function to wrap around the sequence -/
def cyclicIndex (len : ℕ) (i : ℕ) : ℕ :=
  i % len

/-- Check if all n-length subsequences in a sequence of length 2^n are distinct -/
def allSubsequencesDistinct (n : ℕ) (seq : BinarySequence) : Prop :=
  ∀ i j, i < 2^n → j < 2^n → i ≠ j →
    (∃ k, k < n ∧ seq (cyclicIndex (2^n) (i + k)) ≠ seq (cyclicIndex (2^n) (j + k)))

/-- Main theorem: For any positive n, there exists a binary sequence of length 2^n
    where all n-length subsequences are distinct when considered cyclically -/
theorem exists_distinct_subsequences (n : ℕ) (hn : n > 0) :
  ∃ seq : BinarySequence, allSubsequencesDistinct n seq :=
sorry

end exists_distinct_subsequences_l4056_405695


namespace cryptarithm_solution_l4056_405638

def is_valid_solution (A R K : Nat) : Prop :=
  A ≠ R ∧ A ≠ K ∧ R ≠ K ∧
  A < 10 ∧ R < 10 ∧ K < 10 ∧
  1000 * A + 100 * R + 10 * K + A +
  100 * R + 10 * K + A +
  10 * K + A +
  A = 2014

theorem cryptarithm_solution :
  ∀ A R K : Nat, is_valid_solution A R K → A = 1 ∧ R = 4 ∧ K = 7 := by
  sorry

end cryptarithm_solution_l4056_405638


namespace unique_digits_for_multiple_of_99_l4056_405619

def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

theorem unique_digits_for_multiple_of_99 :
  ∀ α β : ℕ,
  0 ≤ α ∧ α ≤ 9 →
  0 ≤ β ∧ β ≤ 9 →
  is_divisible_by_99 (62 * 10000 + α * 1000 + β * 100 + 427) →
  α = 2 ∧ β = 4 := by
sorry

end unique_digits_for_multiple_of_99_l4056_405619


namespace number_of_boys_l4056_405661

/-- Given a school with a total of 1396 people, 315 girls, and 772 teachers,
    prove that there are 309 boys in the school. -/
theorem number_of_boys (total : ℕ) (girls : ℕ) (teachers : ℕ) 
    (h1 : total = 1396) 
    (h2 : girls = 315) 
    (h3 : teachers = 772) : 
  total - girls - teachers = 309 := by
  sorry


end number_of_boys_l4056_405661


namespace sunday_calorie_intake_theorem_l4056_405679

/-- Calculates John's calorie intake for Sunday given his meal structure and calorie content --/
def sunday_calorie_intake (breakfast_calories : ℝ) (morning_snack_addition : ℝ) 
  (lunch_percentage : ℝ) (afternoon_snack_reduction : ℝ) (dinner_multiplier : ℝ) 
  (energy_drink_calories : ℝ) : ℝ :=
  let lunch_calories := breakfast_calories * (1 + lunch_percentage)
  let afternoon_snack_calories := lunch_calories * (1 - afternoon_snack_reduction)
  let dinner_calories := lunch_calories * dinner_multiplier
  let weekday_calories := breakfast_calories + (breakfast_calories + morning_snack_addition) + 
                          lunch_calories + afternoon_snack_calories + dinner_calories
  let energy_drinks_calories := 2 * energy_drink_calories
  weekday_calories + energy_drinks_calories

theorem sunday_calorie_intake_theorem :
  sunday_calorie_intake 500 150 0.25 0.30 2 220 = 3402.5 := by
  sorry

end sunday_calorie_intake_theorem_l4056_405679


namespace exactly_two_valid_numbers_l4056_405681

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_number (n : ℕ) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧
  is_perfect_square (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) ∧
  is_perfect_square ((n / 10 % 10) + (n % 10)) ∧
  is_perfect_square ((n / 10 % 10) - (n % 10)) ∧
  is_perfect_square (n % 10) ∧
  is_perfect_square ((n / 100) % 100) ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) > 0) ∧
  ((n / 10 % 10) + (n % 10) > 0) ∧
  ((n / 10 % 10) - (n % 10) > 0) ∧
  (n % 10 > 0)

theorem exactly_two_valid_numbers :
  ∃! (s : Finset ℕ), s.card = 2 ∧ ∀ n ∈ s, valid_number n :=
sorry

end exactly_two_valid_numbers_l4056_405681


namespace similar_triangles_leg_length_l4056_405660

theorem similar_triangles_leg_length 
  (leg1 : ℝ) 
  (hyp1 : ℝ) 
  (hyp2 : ℝ) 
  (h1 : leg1 = 15) 
  (h2 : hyp1 = 17) 
  (h3 : hyp2 = 51) : 
  (leg1 * hyp2 / hyp1) = 45 :=
sorry

end similar_triangles_leg_length_l4056_405660


namespace fourth_root_equation_solutions_l4056_405637

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x ^ (1/4) = 18 / (9 - x ^ (1/4))) ↔ (x = 81 ∨ x = 1296) := by
  sorry

end fourth_root_equation_solutions_l4056_405637


namespace point_on_graph_l4056_405632

theorem point_on_graph (x y : ℝ) : 
  (x = 1 ∧ y = 4) → (y = 4 * x) := by sorry

end point_on_graph_l4056_405632


namespace two_books_from_shelves_l4056_405642

/-- The number of ways to choose two books of different subjects -/
def choose_two_books (chinese : ℕ) (math : ℕ) (english : ℕ) : ℕ :=
  chinese * math + chinese * english + math * english

/-- Theorem stating that choosing two books of different subjects from the given shelves results in 242 ways -/
theorem two_books_from_shelves :
  choose_two_books 10 9 8 = 242 := by
  sorry

end two_books_from_shelves_l4056_405642


namespace race_head_start_l4056_405683

/-- Proof of head start time in a race --/
theorem race_head_start 
  (race_distance : ℝ) 
  (cristina_speed : ℝ) 
  (nicky_speed : ℝ) 
  (catch_up_time : ℝ)
  (h1 : race_distance = 500)
  (h2 : cristina_speed = 5)
  (h3 : nicky_speed = 3)
  (h4 : catch_up_time = 30) :
  let distance_covered := nicky_speed * catch_up_time
  let cristina_time := distance_covered / cristina_speed
  let head_start := catch_up_time - cristina_time
  head_start = 12 := by sorry

end race_head_start_l4056_405683


namespace arithmetic_sequence_sum_property_l4056_405604

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_property
  (seq : ArithmeticSequence)
  (m : ℕ)
  (h_m_pos : m > 0)
  (h_sum_m : seq.S m = -2)
  (h_sum_m1 : seq.S (m + 1) = 0)
  (h_sum_m2 : seq.S (m + 2) = 3) :
  m = 4 := by
  sorry

end arithmetic_sequence_sum_property_l4056_405604


namespace correct_borrowing_process_l4056_405600

/-- Represents the steps in the book borrowing process -/
inductive BorrowingStep
  | StorageEntry
  | LocatingBook
  | Reading
  | Borrowing
  | StorageExit
  | Returning

/-- Defines the correct order of the book borrowing process -/
def correctBorrowingOrder : List BorrowingStep :=
  [BorrowingStep.StorageEntry, BorrowingStep.LocatingBook, BorrowingStep.Reading, 
   BorrowingStep.Borrowing, BorrowingStep.StorageExit, BorrowingStep.Returning]

/-- Theorem stating that the defined order is correct -/
theorem correct_borrowing_process :
  correctBorrowingOrder = [BorrowingStep.StorageEntry, BorrowingStep.LocatingBook, 
    BorrowingStep.Reading, BorrowingStep.Borrowing, BorrowingStep.StorageExit, 
    BorrowingStep.Returning] :=
by
  sorry


end correct_borrowing_process_l4056_405600


namespace perpendicular_line_to_plane_l4056_405697

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the relation for a line being contained in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (plane_intersection : Plane → Plane → Line)

-- Theorem statement
theorem perpendicular_line_to_plane 
  (α β : Plane) (l m : Line) 
  (h1 : perp_planes α β)
  (h2 : plane_intersection α β = l)
  (h3 : line_in_plane m α)
  (h4 : perp_lines m l) :
  perp_line_plane m β :=
sorry

end perpendicular_line_to_plane_l4056_405697


namespace horner_f_at_5_v2_eq_21_l4056_405618

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 - 5x^4 - 4x^3 + 3x^2 - 6x + 7 -/
def f : List ℝ := [2, -5, -4, 3, -6, 7]

/-- Theorem: Horner's method for f(x) at x = 5 yields v_2 = 21 -/
theorem horner_f_at_5_v2_eq_21 :
  let v := horner f 5
  let v0 := 2
  let v1 := v0 * 5 - 5
  let v2 := v1 * 5 - 4
  v2 = 21 := by sorry

end horner_f_at_5_v2_eq_21_l4056_405618


namespace number_of_divisors_32_l4056_405694

theorem number_of_divisors_32 : Finset.card (Nat.divisors 32) = 6 := by
  sorry

end number_of_divisors_32_l4056_405694


namespace ellipse_equation_l4056_405621

/-- Represents an ellipse with specific properties -/
structure Ellipse where
  /-- The sum of distances from any point on the ellipse to the two foci -/
  focal_distance_sum : ℝ
  /-- The eccentricity of the ellipse -/
  eccentricity : ℝ

/-- Theorem stating the equation of an ellipse with given properties -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.focal_distance_sum = 6)
  (h2 : e.eccentricity = 1/3) :
  ∃ (x y : ℝ), x^2/9 + y^2/8 = 1 :=
sorry

end ellipse_equation_l4056_405621


namespace total_weight_of_pets_l4056_405640

/-- The total weight of four pets given specific weight relationships -/
theorem total_weight_of_pets (evan_dog : ℝ) (ivan_dog : ℝ) (kara_cat : ℝ) (lisa_parrot : ℝ) 
  (h1 : evan_dog = 63)
  (h2 : evan_dog = 7 * ivan_dog)
  (h3 : kara_cat = 5 * (evan_dog + ivan_dog))
  (h4 : lisa_parrot = 3 * (evan_dog + ivan_dog + kara_cat)) :
  evan_dog + ivan_dog + kara_cat + lisa_parrot = 1728 := by
  sorry

#check total_weight_of_pets

end total_weight_of_pets_l4056_405640


namespace bookcase_length_inches_l4056_405651

/-- Conversion factor from feet to inches -/
def inches_per_foot : ℕ := 12

/-- Length of the bookcase in feet -/
def bookcase_length_feet : ℕ := 4

/-- Theorem stating that a 4-foot bookcase is 48 inches long -/
theorem bookcase_length_inches : 
  bookcase_length_feet * inches_per_foot = 48 := by
  sorry

end bookcase_length_inches_l4056_405651


namespace problem_statement_l4056_405647

theorem problem_statement (x y : ℤ) (hx : x = 1) (hy : y = 630) : 2019 * x - 3 * y - 9 = 120 := by
  sorry

end problem_statement_l4056_405647


namespace range_of_a_l4056_405673

theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - x - 2 ≥ 0 → x ≥ a) ∧ 
  (∃ x, x ≥ a ∧ x^2 - x - 2 < 0) → 
  a ∈ Set.Ici 2 :=
sorry

end range_of_a_l4056_405673


namespace factorization_equality_l4056_405641

theorem factorization_equality (x : ℝ) : x * (x - 3) - x + 3 = (x - 1) * (x - 3) := by
  sorry

end factorization_equality_l4056_405641


namespace circle_tangent_to_line_l4056_405691

/-- A circle with center (1,1) that is tangent to the line x + y = 4 -/
def TangentCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 2}

/-- The line x + y = 4 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 4}

theorem circle_tangent_to_line :
  (∃ (p : ℝ × ℝ), p ∈ TangentCircle ∧ p ∈ TangentLine) ∧
  (∀ (p : ℝ × ℝ), p ∈ TangentCircle → p ∈ TangentLine → 
    ∀ (q : ℝ × ℝ), q ∈ TangentCircle → q = p ∨ q ∉ TangentLine) :=
sorry

end circle_tangent_to_line_l4056_405691


namespace amanda_hiking_trip_l4056_405687

/-- Represents Amanda's hiking trip -/
def hiking_trip (total_distance : ℚ) : Prop :=
  let first_segment := total_distance / 4
  let forest_segment := 25
  let mountain_segment := total_distance / 6
  let plain_segment := 2 * forest_segment
  first_segment + forest_segment + mountain_segment + plain_segment = total_distance

theorem amanda_hiking_trip :
  ∃ (total_distance : ℚ), hiking_trip total_distance ∧ total_distance = 900 / 7 := by
  sorry

end amanda_hiking_trip_l4056_405687


namespace chocolate_eggs_weight_l4056_405675

theorem chocolate_eggs_weight (total_eggs : ℕ) (egg_weight : ℕ) (num_boxes : ℕ) (discarded_boxes : ℕ) : 
  total_eggs = 12 →
  egg_weight = 10 →
  num_boxes = 4 →
  discarded_boxes = 1 →
  (total_eggs / num_boxes) * egg_weight * (num_boxes - discarded_boxes) = 90 := by
sorry

end chocolate_eggs_weight_l4056_405675


namespace bike_rental_problem_l4056_405639

/-- Calculates the number of hours a bike was rented given the total amount paid,
    the initial charge, and the hourly rate. -/
def rental_hours (total_paid : ℚ) (initial_charge : ℚ) (hourly_rate : ℚ) : ℚ :=
  (total_paid - initial_charge) / hourly_rate

/-- Proves that given the specific rental conditions and total payment,
    the number of hours rented is 9. -/
theorem bike_rental_problem :
  let total_paid : ℚ := 80
  let initial_charge : ℚ := 17
  let hourly_rate : ℚ := 7
  rental_hours total_paid initial_charge hourly_rate = 9 := by
  sorry


end bike_rental_problem_l4056_405639


namespace intersection_sum_zero_l4056_405663

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x + 2)^2
def parabola2 (x y : ℝ) : Prop := x + 3 = (y - 2)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum_zero :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 0 :=
by
  sorry


end intersection_sum_zero_l4056_405663


namespace smallest_four_digit_not_dividing_l4056_405677

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def product_of_first_n (n : ℕ) : ℕ := Nat.factorial n

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_not_dividing :
  ∃ (n : ℕ), is_four_digit n ∧
    ¬(sum_of_first_n n ∣ product_of_first_n n) ∧
    (∀ m, is_four_digit m ∧ m < n →
      sum_of_first_n m ∣ product_of_first_n m) ∧
    n = 1002 :=
sorry

end smallest_four_digit_not_dividing_l4056_405677


namespace area_constant_circle_final_equation_minimum_distance_l4056_405608

noncomputable section

variable (t : ℝ)
variable (h : t ≠ 0)

def C : ℝ × ℝ := (t, 2/t)
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2*t, 0)
def B : ℝ × ℝ := (0, 4/t)

def circle_equation (x y : ℝ) : Prop :=
  (x - t)^2 + (y - 2/t)^2 = t^2 + 4/t^2

def line_equation (x y : ℝ) : Prop :=
  2*x + y - 4 = 0

def line_l_equation (x y : ℝ) : Prop :=
  x + y + 2 = 0

theorem area_constant :
  (1/2) * |2*t| * |4/t| = 4 :=
sorry

theorem circle_final_equation (x y : ℝ) :
  (∃ M N : ℝ × ℝ, 
    circle_equation t x y ∧ 
    line_equation (M.1) (M.2) ∧ 
    line_equation (N.1) (N.2) ∧
    (M.1 - O.1)^2 + (M.2 - O.2)^2 = (N.1 - O.1)^2 + (N.2 - O.2)^2) →
  (x - 2)^2 + (y - 1)^2 = 5 :=
sorry

theorem minimum_distance (h_pos : t > 0) :
  let B : ℝ × ℝ := (0, 2)
  ∃ P Q : ℝ × ℝ,
    line_l_equation P.1 P.2 ∧
    circle_equation t Q.1 Q.2 ∧
    (∀ P' Q' : ℝ × ℝ, 
      line_l_equation P'.1 P'.2 → 
      circle_equation t Q'.1 Q'.2 →
      Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) + Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤
      Real.sqrt ((P'.1 - B.1)^2 + (P'.2 - B.2)^2) + Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2)) ∧
    Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) + Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2 * Real.sqrt 5 ∧
    P.1 = -4/3 ∧ P.2 = -2/3 :=
sorry

end area_constant_circle_final_equation_minimum_distance_l4056_405608


namespace meaningful_expression_range_l4056_405689

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = (Real.sqrt (x + 2)) / (x - 1)) ↔ x ≥ -2 ∧ x ≠ 1 := by
  sorry

end meaningful_expression_range_l4056_405689


namespace line_inclination_angle_l4056_405668

/-- The inclination angle of a line with point-slope form y - 2 = -√3(x - 1) is π/3 -/
theorem line_inclination_angle (x y : ℝ) :
  y - 2 = -Real.sqrt 3 * (x - 1) → ∃ α : ℝ, α = π / 3 ∧ Real.tan α = -Real.sqrt 3 := by
  sorry

end line_inclination_angle_l4056_405668


namespace complex_equation_sum_l4056_405615

theorem complex_equation_sum (x y : ℝ) : 
  (x + y * Complex.I) / (1 + Complex.I) = (2 : ℂ) + Complex.I → x + y = 4 := by
  sorry

end complex_equation_sum_l4056_405615


namespace expression_defined_iff_l4056_405672

def expression_defined (x : ℝ) : Prop :=
  x > 2 ∧ x < 5

theorem expression_defined_iff (x : ℝ) :
  expression_defined x ↔ (∃ y : ℝ, y = (Real.log (5 - x)) / Real.sqrt (x - 2)) :=
by sorry

end expression_defined_iff_l4056_405672


namespace system_solution_l4056_405685

theorem system_solution :
  ∀ (a b c d n m : ℚ),
    a / 7 + b / 8 = n →
    b = 3 * a - 2 →
    c / 9 + d / 10 = m →
    d = 4 * c + 1 →
    a = 3 →
    c = 2 →
    n = 73 / 56 ∧ m = 101 / 90 := by
  sorry

end system_solution_l4056_405685


namespace min_value_expression_l4056_405623

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (5 * z) / (2 * x + y) + (5 * x) / (y + 2 * z) + (2 * y) / (x + z) + (x + y + z) / (x * y + y * z + z * x) ≥ 9 ∧
  ((5 * z) / (2 * x + y) + (5 * x) / (y + 2 * z) + (2 * y) / (x + z) + (x + y + z) / (x * y + y * z + z * x) = 9 ↔ x = y ∧ y = z) :=
by sorry

end min_value_expression_l4056_405623


namespace ice_cream_flavor_ratio_l4056_405614

def total_flavors : ℕ := 100
def flavors_two_years_ago : ℕ := total_flavors / 4
def flavors_remaining : ℕ := 25
def flavors_tried_total : ℕ := total_flavors - flavors_remaining
def flavors_last_year : ℕ := flavors_tried_total - flavors_two_years_ago

theorem ice_cream_flavor_ratio :
  (flavors_last_year : ℚ) / flavors_two_years_ago = 2 := by sorry

end ice_cream_flavor_ratio_l4056_405614


namespace banana_arrangements_l4056_405625

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  (total_letters! / (a_count! * n_count! * b_count!)) = 60 :=
by sorry

end banana_arrangements_l4056_405625


namespace ratio_ac_to_bd_l4056_405624

/-- Given points A, B, C, D, and E on a line in that order, with given distances between consecutive points,
    prove that the ratio of AC to BD is 7/6. -/
theorem ratio_ac_to_bd (A B C D E : ℝ) 
  (h_order : A < B ∧ B < C ∧ C < D ∧ D < E)
  (h_ab : B - A = 3)
  (h_bc : C - B = 4)
  (h_cd : D - C = 2)
  (h_de : E - D = 3) :
  (C - A) / (D - B) = 7 / 6 := by
  sorry

end ratio_ac_to_bd_l4056_405624


namespace max_value_on_ellipse_l4056_405657

theorem max_value_on_ellipse :
  ∃ (max : ℝ), max = 2 * Real.sqrt 10 ∧
  (∀ x y : ℝ, x^2/9 + y^2/4 = 1 → 2*x - y ≤ max) ∧
  (∃ x y : ℝ, x^2/9 + y^2/4 = 1 ∧ 2*x - y = max) := by
sorry

end max_value_on_ellipse_l4056_405657


namespace frustum_surface_area_l4056_405656

/-- The surface area of a frustum of a regular pyramid with square bases -/
theorem frustum_surface_area (top_side : ℝ) (bottom_side : ℝ) (slant_height : ℝ) :
  top_side = 2 →
  bottom_side = 4 →
  slant_height = 2 →
  let lateral_area := (top_side + bottom_side) * slant_height * 2
  let top_area := top_side ^ 2
  let bottom_area := bottom_side ^ 2
  lateral_area + top_area + bottom_area = 12 * Real.sqrt 3 + 20 := by
  sorry

end frustum_surface_area_l4056_405656


namespace cycle_gain_percent_l4056_405659

theorem cycle_gain_percent (cost_price selling_price : ℝ) : 
  cost_price = 675 →
  selling_price = 1080 →
  ((selling_price - cost_price) / cost_price) * 100 = 60 := by
sorry

end cycle_gain_percent_l4056_405659


namespace input_for_output_nine_l4056_405601

theorem input_for_output_nine (x : ℝ) (y : ℝ) : 
  (x < 0 → y = (x + 1)^2) ∧
  (x ≥ 0 → y = (x - 1)^2) ∧
  (y = 9) →
  (x = -4 ∨ x = 4) :=
by sorry

end input_for_output_nine_l4056_405601


namespace min_colors_for_grid_l4056_405643

-- Define the grid as a type alias for pairs of integers
def Grid := ℤ × ℤ

-- Define the distance function between two cells
def distance (a b : Grid) : ℕ :=
  max (Int.natAbs (a.1 - b.1)) (Int.natAbs (a.2 - b.2))

-- Define the color function
def color (cell : Grid) : Fin 4 :=
  Fin.ofNat ((cell.1 + cell.2).natAbs % 4)

-- State the theorem
theorem min_colors_for_grid : 
  (∀ a b : Grid, distance a b = 6 → color a ≠ color b) ∧
  (∀ n : ℕ, n < 4 → ∃ a b : Grid, distance a b = 6 ∧ 
    Fin.ofNat (n % 4) = color a ∧ Fin.ofNat (n % 4) = color b) :=
sorry

end min_colors_for_grid_l4056_405643


namespace ellipse_eccentricity_l4056_405684

theorem ellipse_eccentricity (b : ℝ) : 
  b > 0 → 
  (∀ x y : ℝ, x^2 + y^2 / (b^2 + 1) = 1 → 
    b / Real.sqrt (b^2 + 1) = Real.sqrt 10 / 10) → 
  b = 1/3 := by
sorry

end ellipse_eccentricity_l4056_405684


namespace arithmetic_mean_of_squares_formula_l4056_405678

/-- The arithmetic mean of the squares of the first n positive integers -/
def arithmetic_mean_of_squares (n : ℕ+) : ℚ :=
  (↑n.val * (↑n.val + 1) * (2 * ↑n.val + 1)) / (6 * ↑n.val)

theorem arithmetic_mean_of_squares_formula (n : ℕ+) :
  arithmetic_mean_of_squares n = ((↑n.val + 1) * (2 * ↑n.val + 1)) / 6 := by
  sorry

end arithmetic_mean_of_squares_formula_l4056_405678


namespace recliner_sales_increase_l4056_405649

/-- Proves that a 20% price reduction and 28% gross revenue increase results in a 60% increase in sales volume -/
theorem recliner_sales_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (new_price : ℝ) 
  (new_quantity : ℝ) 
  (h1 : new_price = 0.80 * original_price) 
  (h2 : new_price * new_quantity = 1.28 * (original_price * original_quantity)) : 
  (new_quantity - original_quantity) / original_quantity = 0.60 := by
sorry

end recliner_sales_increase_l4056_405649


namespace honey_water_percentage_l4056_405662

/-- Given that 1.7 kg of flower-nectar containing 50% water yields 1 kg of honey,
    prove that the percentage of water in the resulting honey is 15%. -/
theorem honey_water_percentage
  (nectar_weight : ℝ)
  (honey_weight : ℝ)
  (nectar_water_percentage : ℝ)
  (h1 : nectar_weight = 1.7)
  (h2 : honey_weight = 1)
  (h3 : nectar_water_percentage = 50)
  : (honey_weight - (nectar_weight * (1 - nectar_water_percentage / 100))) / honey_weight * 100 = 15 := by
  sorry

end honey_water_percentage_l4056_405662


namespace complex_in_fourth_quadrant_l4056_405607

theorem complex_in_fourth_quadrant (m : ℝ) (z : ℂ) 
  (h1 : m < 1) 
  (h2 : z = 2 + (m - 1) * Complex.I) : 
  z.re > 0 ∧ z.im < 0 := by
  sorry

end complex_in_fourth_quadrant_l4056_405607


namespace x_squared_y_plus_xy_squared_l4056_405612

theorem x_squared_y_plus_xy_squared (x y : ℝ) :
  x = Real.sqrt 3 + Real.sqrt 2 →
  y = Real.sqrt 3 - Real.sqrt 2 →
  x^2 * y + x * y^2 = 2 * Real.sqrt 3 := by
sorry

end x_squared_y_plus_xy_squared_l4056_405612


namespace greatest_average_speed_l4056_405616

/-- Checks if a number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The initial odometer reading -/
def initialReading : ℕ := 12321

/-- The duration of the drive in hours -/
def driveDuration : ℝ := 4

/-- The speed limit in miles per hour -/
def speedLimit : ℝ := 85

/-- The greatest possible average speed in miles per hour -/
def greatestAverageSpeed : ℝ := 75

/-- Theorem stating the greatest possible average speed given the conditions -/
theorem greatest_average_speed :
  isPalindrome initialReading →
  ∃ (finalReading : ℕ),
    isPalindrome finalReading ∧
    finalReading > initialReading ∧
    (finalReading - initialReading : ℝ) / driveDuration ≤ speedLimit ∧
    (finalReading - initialReading : ℝ) / driveDuration = greatestAverageSpeed :=
  sorry

end greatest_average_speed_l4056_405616


namespace multiplication_equality_l4056_405666

-- Define the digits as natural numbers
def A : ℕ := 6
def B : ℕ := 7
def C : ℕ := 4
def D : ℕ := 2
def E : ℕ := 5
def F : ℕ := 9
def H : ℕ := 3
def J : ℕ := 8

-- Define the numbers ABCD and EF
def ABCD : ℕ := A * 1000 + B * 100 + C * 10 + D
def EF : ℕ := E * 10 + F

-- Define the result HFBBBJ
def HFBBBJ : ℕ := H * 100000 + F * 10000 + B * 1000 + B * 100 + B * 10 + J

-- State the theorem
theorem multiplication_equality :
  ABCD * EF = HFBBBJ :=
sorry

end multiplication_equality_l4056_405666


namespace solution_set_of_inequality_l4056_405676

-- Define the properties of function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Main theorem
theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono : monotone_increasing_on_nonneg f)
  (h_f_neg_one : f (-1) = 0) :
  {x : ℝ | f (2 * x - 1) > 0} = {x : ℝ | x < 0 ∨ x > 1} :=
sorry

end solution_set_of_inequality_l4056_405676


namespace min_square_side_length_l4056_405671

theorem min_square_side_length (square_area_min : ℝ) (circle_area_min : ℝ) :
  square_area_min = 900 →
  circle_area_min = 100 →
  ∃ (s : ℝ),
    s^2 ≥ square_area_min ∧
    π * (s/2)^2 ≥ circle_area_min ∧
    ∀ (t : ℝ), (t^2 ≥ square_area_min ∧ π * (t/2)^2 ≥ circle_area_min) → s ≤ t :=
by
  sorry

#check min_square_side_length

end min_square_side_length_l4056_405671


namespace ellipse_major_axis_length_l4056_405606

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (percentage_longer : ℝ) : ℝ :=
  2 * cylinder_radius * (1 + percentage_longer)

/-- Theorem: The length of the major axis of the ellipse is 6.4 --/
theorem ellipse_major_axis_length :
  major_axis_length 2 0.6 = 6.4 := by
  sorry

end ellipse_major_axis_length_l4056_405606
