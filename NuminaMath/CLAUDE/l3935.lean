import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3935_393574

theorem complex_number_quadrant : ∃ (x y : ℝ), (Complex.mk x y = (2 - Complex.I)^2) ∧ (x > 0) ∧ (y < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3935_393574


namespace NUMINAMATH_CALUDE_lcm_5_7_10_21_l3935_393560

theorem lcm_5_7_10_21 : Nat.lcm 5 (Nat.lcm 7 (Nat.lcm 10 21)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_7_10_21_l3935_393560


namespace NUMINAMATH_CALUDE_trajectory_of_point_M_l3935_393533

/-- The trajectory of point M satisfying the given distance conditions -/
theorem trajectory_of_point_M (x y : ℝ) : 
  (∀ (x₀ y₀ : ℝ), (x₀ - 0)^2 + (y₀ - 4)^2 = (abs (y₀ + 5) - 1)^2 → x₀^2 = 16 * y₀) →
  x^2 + (y - 4)^2 = (abs (y + 5) - 1)^2 →
  x^2 = 16 * y := by
  sorry


end NUMINAMATH_CALUDE_trajectory_of_point_M_l3935_393533


namespace NUMINAMATH_CALUDE_hcf_of_12_and_15_l3935_393548

theorem hcf_of_12_and_15 : 
  ∀ (hcf lcm : ℕ), 
    lcm = 60 → 
    12 * 15 = lcm * hcf → 
    hcf = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_hcf_of_12_and_15_l3935_393548


namespace NUMINAMATH_CALUDE_solution_set_circle_plus_l3935_393571

/-- Custom operation ⊕ -/
def circle_plus (a b : ℝ) : ℝ := -2 * a + b

/-- Theorem stating the solution set of x ⊕ 4 > 0 -/
theorem solution_set_circle_plus (x : ℝ) :
  circle_plus x 4 > 0 ↔ x < 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_circle_plus_l3935_393571


namespace NUMINAMATH_CALUDE_tangent_line_of_odd_function_l3935_393500

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a - 2) * x^2 + 2 * x

-- State the theorem
theorem tangent_line_of_odd_function (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) →  -- f is an odd function
  (f a 1 = 3) →                 -- f(1) = 3
  ∃ m b : ℝ, m = 5 ∧ b = -2 ∧
    ∀ x, (f a x - f a 1) = m * (x - 1) + b :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_of_odd_function_l3935_393500


namespace NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l3935_393598

theorem smallest_integer_quadratic_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 36 ≤ 0 → n ≤ m) ∧ n^2 - 13*n + 36 ≤ 0 ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l3935_393598


namespace NUMINAMATH_CALUDE_adjacent_points_probability_l3935_393505

/-- The number of points around the square -/
def n : ℕ := 12

/-- The number of pairs of adjacent points -/
def adjacent_pairs : ℕ := 12

/-- The total number of ways to choose 2 points from n points -/
def total_combinations : ℕ := n * (n - 1) / 2

/-- The probability of choosing two adjacent points -/
def probability : ℚ := adjacent_pairs / total_combinations

theorem adjacent_points_probability : probability = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_points_probability_l3935_393505


namespace NUMINAMATH_CALUDE_triangle_side_length_l3935_393530

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  b = 6 ∧
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 ∧
  A = π/3 →
  a = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3935_393530


namespace NUMINAMATH_CALUDE_factorization_3m_squared_minus_12m_l3935_393515

theorem factorization_3m_squared_minus_12m (m : ℝ) : 3 * m^2 - 12 * m = 3 * m * (m - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3m_squared_minus_12m_l3935_393515


namespace NUMINAMATH_CALUDE_special_number_exists_l3935_393581

/-- Number of digits in a natural number -/
def number_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For every natural number a, there exists a natural number b and a non-negative integer k
    such that a * 10^k + b = a * (b * 10^(number_of_digits a) + a) -/
theorem special_number_exists (a : ℕ) : ∃ (b : ℕ) (k : ℕ), 
  a * 10^k + b = a * (b * 10^(number_of_digits a) + a) := by sorry

end NUMINAMATH_CALUDE_special_number_exists_l3935_393581


namespace NUMINAMATH_CALUDE_current_rate_calculation_l3935_393519

/-- Given a boat traveling downstream, calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ)             -- Speed of the boat in still water (km/hr)
  (distance : ℝ)               -- Distance traveled downstream (km)
  (time : ℝ)                   -- Time taken for the downstream journey (hr)
  (h1 : boat_speed = 20)       -- The boat's speed in still water is 20 km/hr
  (h2 : distance = 10)         -- The distance traveled downstream is 10 km
  (h3 : time = 24 / 60)        -- The time taken is 24 minutes, converted to hours
  : ∃ (current_rate : ℝ), 
    distance = (boat_speed + current_rate) * time ∧ 
    current_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l3935_393519


namespace NUMINAMATH_CALUDE_legs_exceed_twice_heads_by_30_l3935_393563

/-- Represents the number of ducks in the group -/
def num_ducks : ℕ := sorry

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 15

/-- Calculates the total number of legs in the group -/
def total_legs : ℕ := 2 * num_ducks + 4 * num_cows

/-- Calculates the total number of heads in the group -/
def total_heads : ℕ := num_ducks + num_cows

/-- Theorem stating that the number of legs exceeds twice the number of heads by 30 -/
theorem legs_exceed_twice_heads_by_30 : total_legs = 2 * total_heads + 30 := by
  sorry

end NUMINAMATH_CALUDE_legs_exceed_twice_heads_by_30_l3935_393563


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3935_393552

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  m : ℝ
  equation : (x : ℝ) → (y : ℝ) → Prop :=
    fun x y => x^2 / (m + 1) - y^2 / (3 - m) = 1

/-- Theorem statement for the hyperbola eccentricity problem -/
theorem hyperbola_eccentricity_range 
  (C : Hyperbola) 
  (F : Point) 
  (k : ℝ) 
  (A B P Q : Point) 
  (h1 : F.x < 0) -- F is the left focus
  (h2 : k ≥ Real.sqrt 3) -- Line slope condition
  (h3 : C.equation A.x A.y ∧ C.equation B.x B.y) -- A and B are on the hyperbola
  (h4 : P.x = (A.x + F.x) / 2 ∧ P.y = (A.y + F.y) / 2) -- P is midpoint of AF
  (h5 : Q.x = (B.x + F.x) / 2 ∧ Q.y = (B.y + F.y) / 2) -- Q is midpoint of BF
  (h6 : (P.y - 0) * (Q.y - 0) = -(P.x - 0) * (Q.x - 0)) -- OP ⊥ OQ
  : ∃ (e : ℝ), e ≥ Real.sqrt 3 + 1 ∧ 
    ∀ (e' : ℝ), e' ≥ Real.sqrt 3 + 1 → 
    ∃ (C' : Hyperbola), C'.m = C.m ∧ 
    (∃ (F' A' B' P' Q' : Point) (k' : ℝ), 
      F'.x < 0 ∧ 
      k' ≥ Real.sqrt 3 ∧
      C'.equation A'.x A'.y ∧ C'.equation B'.x B'.y ∧
      P'.x = (A'.x + F'.x) / 2 ∧ P'.y = (A'.y + F'.y) / 2 ∧
      Q'.x = (B'.x + F'.x) / 2 ∧ Q'.y = (B'.y + F'.y) / 2 ∧
      (P'.y - 0) * (Q'.y - 0) = -(P'.x - 0) * (Q'.x - 0) ∧
      e' = C'.m) := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3935_393552


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l3935_393559

/-- A quadratic equation in one variable -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ
  h_quadratic : a ≠ 0

/-- The general form of a quadratic equation -/
def general_form (eq : QuadraticEquation) : Prop :=
  eq.a * eq.x^2 + eq.b * eq.x + eq.c = 0

/-- Theorem: The general form of a quadratic equation in one variable is ax^2 + bx + c = 0 where a ≠ 0 -/
theorem quadratic_equation_general_form (eq : QuadraticEquation) :
  general_form eq :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l3935_393559


namespace NUMINAMATH_CALUDE_no_prime_between_30_40_congruent_7_mod_9_l3935_393575

theorem no_prime_between_30_40_congruent_7_mod_9 : ¬ ∃ (n : ℕ), Nat.Prime n ∧ 30 < n ∧ n < 40 ∧ n % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_between_30_40_congruent_7_mod_9_l3935_393575


namespace NUMINAMATH_CALUDE_thirty_percent_of_eighty_l3935_393590

theorem thirty_percent_of_eighty : ∃ x : ℝ, (30 / 100) * x = 24 ∧ x = 80 := by sorry

end NUMINAMATH_CALUDE_thirty_percent_of_eighty_l3935_393590


namespace NUMINAMATH_CALUDE_power_equation_solution_l3935_393541

theorem power_equation_solution (m : ℕ) : 8^2 = 4^2 * 2^m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3935_393541


namespace NUMINAMATH_CALUDE_distance_A_to_origin_l3935_393564

/-- The distance between point A(2, 3, 3) and the origin O(0, 0, 0) in a three-dimensional Cartesian coordinate system is √22. -/
theorem distance_A_to_origin : 
  let A : Fin 3 → ℝ := ![2, 3, 3]
  let O : Fin 3 → ℝ := ![0, 0, 0]
  Real.sqrt ((A 0 - O 0)^2 + (A 1 - O 1)^2 + (A 2 - O 2)^2) = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_to_origin_l3935_393564


namespace NUMINAMATH_CALUDE_both_shooters_hit_probability_l3935_393551

theorem both_shooters_hit_probability
  (prob_A : ℝ)
  (prob_B : ℝ)
  (h_prob_A : prob_A = 0.9)
  (h_prob_B : prob_B = 0.8)
  (h_independent : True)  -- Assumption of independence
  : prob_A * prob_B = 0.72 :=
by sorry

end NUMINAMATH_CALUDE_both_shooters_hit_probability_l3935_393551


namespace NUMINAMATH_CALUDE_max_reflections_theorem_l3935_393580

/-- The angle between the two reflecting lines in degrees -/
def angle : ℝ := 12

/-- The maximum angle of incidence in degrees -/
def max_incidence : ℝ := 90

/-- The maximum number of reflections possible -/
def max_reflections : ℕ := 7

/-- Theorem stating the maximum number of reflections possible given the angle between lines -/
theorem max_reflections_theorem : 
  ∀ n : ℕ, (n : ℝ) * angle ≤ max_incidence ↔ n ≤ max_reflections :=
by sorry

end NUMINAMATH_CALUDE_max_reflections_theorem_l3935_393580


namespace NUMINAMATH_CALUDE_square_sum_equals_five_l3935_393566

theorem square_sum_equals_five (a b c : ℝ) 
  (h : a + b + c + 3 = 2 * (Real.sqrt a + Real.sqrt (b + 1) + Real.sqrt (c - 1))) :
  a^2 + b^2 + c^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_five_l3935_393566


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l3935_393507

theorem arithmetic_equalities :
  (96 * 98 * 189 = 81 * 343 * 2^6) ∧
  (12^18 = 27^6 * 16^9) ∧
  (25^28 * 0.008^19 ≠ 0.25) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l3935_393507


namespace NUMINAMATH_CALUDE_nail_polish_difference_l3935_393504

theorem nail_polish_difference (kim heidi karen : ℕ) : 
  kim = 12 →
  heidi = kim + 5 →
  karen + heidi = 25 →
  karen < kim →
  kim - karen = 4 := by
sorry

end NUMINAMATH_CALUDE_nail_polish_difference_l3935_393504


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3935_393520

theorem arithmetic_calculations : 
  (12 - (-18) + (-7) + (-15) = 8) ∧ 
  (-2^3 + (-5)^2 * (2/5) - |(-3)| = -1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3935_393520


namespace NUMINAMATH_CALUDE_multiply_63_57_l3935_393582

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_CALUDE_multiply_63_57_l3935_393582


namespace NUMINAMATH_CALUDE_find_a_value_l3935_393565

theorem find_a_value (a : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + a) + |x - 3| ≤ 5) ∧ 
  (∃ x : ℝ, x > 3 → (x^2 - 4*x + a) + |x - 3| > 5) →
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l3935_393565


namespace NUMINAMATH_CALUDE_fibonacci_seventh_term_l3935_393511

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_seventh_term : fibonacci 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_seventh_term_l3935_393511


namespace NUMINAMATH_CALUDE_line_parameterization_l3935_393513

/-- Given a line y = 2x - 30 parameterized by (x,y) = (f(t), 20t - 14), prove that f(t) = 10t + 8 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 2 * f t - 30 = 20 * t - 14) → 
  (∀ t : ℝ, f t = 10 * t + 8) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3935_393513


namespace NUMINAMATH_CALUDE_school_boys_count_l3935_393592

theorem school_boys_count (girls : ℕ) (difference : ℕ) (boys : ℕ) : 
  girls = 739 → difference = 402 → girls = boys + difference → boys = 337 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l3935_393592


namespace NUMINAMATH_CALUDE_triangle_problem_l3935_393579

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧
  Real.sqrt 3 * a = 2 * c * Real.sin A ∧  -- Given condition
  c = Real.sqrt 7 ∧  -- Given condition
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →  -- Area condition
  C = π/3 ∧ a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3935_393579


namespace NUMINAMATH_CALUDE_radius_greater_than_distance_to_center_l3935_393540

/-- A circle with center O and a point P inside it -/
structure CircleWithInnerPoint where
  O : ℝ × ℝ  -- Center of the circle
  P : ℝ × ℝ  -- Point inside the circle
  r : ℝ      -- Radius of the circle
  h_inside : dist P O < r  -- P is inside the circle

/-- The theorem stating that if P is inside circle O and distance from P to O is 5,
    then the radius of circle O must be greater than 5 -/
theorem radius_greater_than_distance_to_center 
  (c : CircleWithInnerPoint) (h : dist c.P c.O = 5) : c.r > 5 := by
  sorry

end NUMINAMATH_CALUDE_radius_greater_than_distance_to_center_l3935_393540


namespace NUMINAMATH_CALUDE_even_function_extension_l3935_393583

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

/-- The given function f defined for x ≤ 0 -/
def f_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≤ 0 → f x = x^2 - 2*x

theorem even_function_extension :
  ∀ f : ℝ → ℝ, EvenFunction f → f_nonpositive f →
  ∀ x : ℝ, x > 0 → f x = x^2 + 2*x :=
sorry

end NUMINAMATH_CALUDE_even_function_extension_l3935_393583


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l3935_393594

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioo 0 2, x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l3935_393594


namespace NUMINAMATH_CALUDE_teachers_arrangements_count_l3935_393514

def num_students : ℕ := 5
def num_teachers : ℕ := 2

def arrangements (n_students : ℕ) (n_teachers : ℕ) : ℕ :=
  (Nat.factorial n_students) * (n_students - 1) * (Nat.factorial n_teachers)

theorem teachers_arrangements_count :
  arrangements num_students num_teachers = 960 := by
  sorry

end NUMINAMATH_CALUDE_teachers_arrangements_count_l3935_393514


namespace NUMINAMATH_CALUDE_print_output_l3935_393508

-- Define a simple output function to represent PRINT
def print (a : ℕ) (b : ℕ) : String :=
  s!"{a}, {b}"

-- Theorem statement
theorem print_output : print 3 (3 + 2) = "3, 5" := by
  sorry

end NUMINAMATH_CALUDE_print_output_l3935_393508


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l3935_393531

theorem sin_shift_equivalence :
  ∀ x : ℝ, Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l3935_393531


namespace NUMINAMATH_CALUDE_triangle_problem_l3935_393584

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  A + B + C = π →
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  -- Given conditions
  (Real.cos (B + C)) / (Real.cos C) = a / (2 * b + c) →
  b = 1 →
  Real.cos C = 2 * Real.sqrt 7 / 7 →
  -- Conclusions
  A = 2 * π / 3 ∧ a = Real.sqrt 7 ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3935_393584


namespace NUMINAMATH_CALUDE_factors_of_96_with_square_sum_208_l3935_393538

theorem factors_of_96_with_square_sum_208 : 
  ∃ (a b : ℕ+), (a * b = 96) ∧ (a^2 + b^2 = 208) := by sorry

end NUMINAMATH_CALUDE_factors_of_96_with_square_sum_208_l3935_393538


namespace NUMINAMATH_CALUDE_ellipse_properties_l3935_393535

/-- Ellipse structure -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line with slope 1 -/
structure Line where
  c : ℝ

/-- Theorem about ellipse properties -/
theorem ellipse_properties (E : Ellipse) (F₁ F₂ A B P : Point) (l : Line) :
  -- Line l passes through F₁ and has slope 1
  F₁.x = -E.a.sqrt^2 - E.b^2 ∧ F₁.y = 0 ∧ l.c = F₁.x →
  -- A and B are intersection points of l and E
  (A.x^2 / E.a^2 + A.y^2 / E.b^2 = 1) ∧ (B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1) ∧
  A.x = A.y - l.c ∧ B.x = B.y - l.c →
  -- |AF₂|, |AB|, |BF₂| form arithmetic sequence
  2 * ((A.x - B.x)^2 + (A.y - B.y)^2) = 
    ((A.x - F₂.x)^2 + (A.y - F₂.y)^2) + ((B.x - F₂.x)^2 + (B.y - F₂.y)^2) →
  -- P(0, -1) satisfies |PA| = |PB|
  P.x = 0 ∧ P.y = -1 ∧
  (P.x - A.x)^2 + (P.y - A.y)^2 = (P.x - B.x)^2 + (P.y - B.y)^2 →
  -- Eccentricity is √2/2 and equation is x^2/18 + y^2/9 = 1
  (E.a^2 - E.b^2) / E.a^2 = 1/2 ∧ E.a^2 = 18 ∧ E.b^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3935_393535


namespace NUMINAMATH_CALUDE_not_all_numbers_representable_l3935_393578

theorem not_all_numbers_representable :
  ∃ k : ℕ, k % 6 = 0 ∧ k > 1000 ∧
  ∀ m n : ℕ, k ≠ n * (n + 1) * (n + 2) * (n + 3) * (n + 4) - m * (m + 1) * (m + 2) :=
by sorry

end NUMINAMATH_CALUDE_not_all_numbers_representable_l3935_393578


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3935_393545

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : 1 / x + 1 / y = 5) 
  (h4 : 1 / x - 1 / y = -9) : 
  x + y = -5/14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l3935_393545


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_square_l3935_393516

theorem power_of_two_plus_one_square (m n : ℕ+) :
  2^(m : ℕ) + 1 = (n : ℕ)^2 ↔ m = 3 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_square_l3935_393516


namespace NUMINAMATH_CALUDE_pascal_triangle_entries_l3935_393510

/-- The number of entries in the n-th row of Pascal's Triangle -/
def entriesInRow (n : ℕ) : ℕ := n + 1

/-- The sum of entries in the first n rows of Pascal's Triangle -/
def sumOfEntries (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem pascal_triangle_entries : sumOfEntries 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_entries_l3935_393510


namespace NUMINAMATH_CALUDE_target_probability_l3935_393549

/-- The probability of hitting a target once -/
def p : ℝ := 0.6

/-- The number of shots -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in n shots -/
def prob_at_least_two (p : ℝ) (n : ℕ) : ℝ :=
  3 * p^2 * (1 - p) + p^3

theorem target_probability :
  prob_at_least_two p n = 0.648 :=
sorry

end NUMINAMATH_CALUDE_target_probability_l3935_393549


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3935_393557

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - (k + 1) * x + 2 * k = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - (k + 1) * y + 2 * k = 0 → y = x) ↔ 
  k = 11 + 10 * Real.sqrt 6 ∨ k = 11 - 10 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3935_393557


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l3935_393527

theorem multiplication_puzzle :
  ∀ A B C K : ℕ,
    A ∈ Finset.range 10 →
    B ∈ Finset.range 10 →
    C ∈ Finset.range 10 →
    K ∈ Finset.range 10 →
    A < B →
    A ≠ B ∧ A ≠ C ∧ A ≠ K ∧ B ≠ C ∧ B ≠ K ∧ C ≠ K →
    (10 * A + C) * (10 * B + C) = 111 * K →
    K * 111 = 100 * K + 10 * K + K →
    A = 2 ∧ B = 3 ∧ C = 7 ∧ K = 9 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l3935_393527


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3935_393517

/-- The radius of a circle inscribed in a sector that is one-third of a circle --/
theorem inscribed_circle_radius (R : ℝ) (h : R = 5) :
  let r := (5 * Real.sqrt 3 - 5) / 2
  r > 0 ∧ r + r * Real.sqrt 3 = R :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3935_393517


namespace NUMINAMATH_CALUDE_sequence_characterization_l3935_393596

def isValidSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, 0 ≤ a n) ∧
  (∀ n, a n ≤ a (n + 1)) ∧
  (∀ m n, a (m^2 + n^2) = (a m)^2 + (a n)^2)

theorem sequence_characterization (a : ℕ → ℝ) :
  isValidSequence a →
  ((∀ n, a n = 1/2) ∨ (∀ n, a n = 0) ∨ (∀ n, a n = n)) :=
sorry

end NUMINAMATH_CALUDE_sequence_characterization_l3935_393596


namespace NUMINAMATH_CALUDE_percentage_in_70to79_is_25_percent_l3935_393572

/-- Represents the score ranges in Ms. Hernandez's biology class -/
inductive ScoreRange
  | Above90
  | Range80to89
  | Range70to79
  | Range60to69
  | Below60

/-- The frequency of students in each score range -/
def frequency (range : ScoreRange) : ℕ :=
  match range with
  | ScoreRange.Above90 => 5
  | ScoreRange.Range80to89 => 9
  | ScoreRange.Range70to79 => 7
  | ScoreRange.Range60to69 => 4
  | ScoreRange.Below60 => 3

/-- The total number of students in the class -/
def totalStudents : ℕ := 
  frequency ScoreRange.Above90 +
  frequency ScoreRange.Range80to89 +
  frequency ScoreRange.Range70to79 +
  frequency ScoreRange.Range60to69 +
  frequency ScoreRange.Below60

/-- The percentage of students who scored in the 70%-79% range -/
def percentageIn70to79Range : ℚ :=
  (frequency ScoreRange.Range70to79 : ℚ) / (totalStudents : ℚ) * 100

theorem percentage_in_70to79_is_25_percent :
  percentageIn70to79Range = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_in_70to79_is_25_percent_l3935_393572


namespace NUMINAMATH_CALUDE_fermat_coprime_l3935_393543

/-- The n-th Fermat number -/
def fermat (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Fermat numbers are pairwise coprime -/
theorem fermat_coprime : ∀ i j : ℕ, i ≠ j → Nat.gcd (fermat i) (fermat j) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fermat_coprime_l3935_393543


namespace NUMINAMATH_CALUDE_quadratic_rational_solution_l3935_393577

theorem quadratic_rational_solution (a b : ℕ+) :
  (∃ x : ℚ, x^2 + (a + b : ℚ)^2 * x + 4 * (a : ℚ) * (b : ℚ) = 1) ↔ a = b :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rational_solution_l3935_393577


namespace NUMINAMATH_CALUDE_circle_equation_l3935_393561

-- Define the circle C
def circle_C (a r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = r^2}

-- Define the line 2x - y = 0
def line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - p.2 = 0}

theorem circle_equation : 
  ∃ (a r : ℝ), 
    a > 0 ∧ 
    (0, Real.sqrt 5) ∈ circle_C a r ∧ 
    (abs (2 * a) / Real.sqrt 5 = 4 * Real.sqrt 5 / 5) ∧
    circle_C a r = circle_C 2 3 :=
  sorry

end NUMINAMATH_CALUDE_circle_equation_l3935_393561


namespace NUMINAMATH_CALUDE_right_triangle_area_l3935_393532

theorem right_triangle_area (p : ℝ) (h : p > 0) : ∃ (x y z : ℝ),
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^2 + y^2 = z^2 ∧
  x + y + z = 3*p ∧
  x = z/2 ∧
  (1/2) * x * y = (p^2 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3935_393532


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3935_393587

theorem modulus_of_complex_number (z : ℂ) (h : z = Complex.I * (2 - Complex.I)) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3935_393587


namespace NUMINAMATH_CALUDE_x_values_l3935_393569

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) :
  x = 4 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l3935_393569


namespace NUMINAMATH_CALUDE_carpet_border_problem_l3935_393526

theorem carpet_border_problem :
  let count_valid_pairs := 
    (Finset.filter 
      (fun pair : ℕ × ℕ => 
        let (p, q) := pair
        q > p ∧ (p - 6) * (q - 6) = 48 ∧ p > 6 ∧ q > 6)
      (Finset.product (Finset.range 100) (Finset.range 100))).card
  count_valid_pairs = 5 := by
sorry

end NUMINAMATH_CALUDE_carpet_border_problem_l3935_393526


namespace NUMINAMATH_CALUDE_sqrt_domain_sqrt_nonneg_sqrt_undefined_neg_l3935_393553

-- Define the square root function for non-negative real numbers
noncomputable def sqrt (a : ℝ) : ℝ := Real.sqrt a

-- Theorem stating that the domain of the square root function is non-negative real numbers
theorem sqrt_domain (a : ℝ) : ∃ (x : ℝ), x ^ 2 = a → a ≥ 0 := by
  sorry

-- Theorem stating that the square root of a non-negative number is non-negative
theorem sqrt_nonneg (a : ℝ) (h : a ≥ 0) : sqrt a ≥ 0 := by
  sorry

-- Theorem stating that the square root function is undefined for negative numbers
theorem sqrt_undefined_neg (a : ℝ) : a < 0 → ¬∃ (x : ℝ), x ^ 2 = a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_domain_sqrt_nonneg_sqrt_undefined_neg_l3935_393553


namespace NUMINAMATH_CALUDE_dante_balloon_sharing_l3935_393536

theorem dante_balloon_sharing :
  ∀ (num_friends : ℕ),
    num_friends > 0 →
    250 / num_friends - 11 = 39 →
    num_friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_dante_balloon_sharing_l3935_393536


namespace NUMINAMATH_CALUDE_kanul_original_amount_l3935_393539

theorem kanul_original_amount (raw_materials machinery marketing : ℕ) 
  (h1 : raw_materials = 35000)
  (h2 : machinery = 40000)
  (h3 : marketing = 15000)
  (h4 : (raw_materials + machinery + marketing : ℚ) = 0.25 * 360000) :
  360000 = 360000 := by sorry

end NUMINAMATH_CALUDE_kanul_original_amount_l3935_393539


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_parameter_range_l3935_393586

/-- Given that f(x) = -x^3 + 2ax^2 - x - 3 is a monotonic function on ℝ, 
    prove that a ∈ [-√3/2, √3/2] -/
theorem monotonic_cubic_function_parameter_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + 2*a*x^2 - x - 3)) →
  a ∈ Set.Icc (-Real.sqrt 3 / 2) (Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_parameter_range_l3935_393586


namespace NUMINAMATH_CALUDE_residue_of_5_1234_mod_19_l3935_393573

theorem residue_of_5_1234_mod_19 : 
  (5 : ℤ)^1234 ≡ 7 [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_residue_of_5_1234_mod_19_l3935_393573


namespace NUMINAMATH_CALUDE_simple_interest_rate_l3935_393521

/-- Proves that given a principal of $600 lent at simple interest for 8 years,
    if the total interest is $360 less than the principal,
    then the annual interest rate is 5%. -/
theorem simple_interest_rate (principal : ℝ) (time : ℝ) (interest : ℝ) (rate : ℝ) :
  principal = 600 →
  time = 8 →
  interest = principal - 360 →
  interest = principal * rate * time →
  rate = 0.05 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l3935_393521


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l3935_393595

/-- The equation of a curve -/
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

/-- The point on the curve -/
def point : ℝ × ℝ := (1, 2)

/-- The equation of the tangent line -/
def tangent_line (x : ℝ) : ℝ := 3*x - 1

theorem tangent_line_at_point :
  let (x₀, y₀) := point
  (f x₀ = y₀) ∧ 
  (∀ x : ℝ, tangent_line x = f x₀ + (tangent_line x₀ - f x₀) * (x - x₀)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l3935_393595


namespace NUMINAMATH_CALUDE_distance_proof_l3935_393554

/- Define the speeds of A and B in meters per minute -/
def speed_A : ℝ := 60
def speed_B : ℝ := 80

/- Define the rest time of B in minutes -/
def rest_time : ℝ := 14

/- Define the distance between A and B -/
def distance_AB : ℝ := 1680

/- Theorem statement -/
theorem distance_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    speed_A * t + speed_B * t = distance_AB ∧
    (distance_AB / speed_A + distance_AB / speed_B + rest_time) / 2 = t :=
by sorry

#check distance_proof

end NUMINAMATH_CALUDE_distance_proof_l3935_393554


namespace NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l3935_393529

-- Problem 1
theorem factorization_problem1 (x y : ℝ) :
  x^3 + 2*x^2*y + x*y^2 = x*(x + y)^2 := by sorry

-- Problem 2
theorem factorization_problem2 (m n : ℝ) :
  4*m^2 - n^2 - 4*m + 1 = (2*m - 1 + n)*(2*m - 1 - n) := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l3935_393529


namespace NUMINAMATH_CALUDE_coconut_ratio_l3935_393570

theorem coconut_ratio (paolo_coconuts : ℕ) (dante_sold : ℕ) (dante_remaining : ℕ) :
  paolo_coconuts = 14 →
  dante_sold = 10 →
  dante_remaining = 32 →
  (dante_remaining : ℚ) / paolo_coconuts = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_coconut_ratio_l3935_393570


namespace NUMINAMATH_CALUDE_sum_of_digits_0_to_2012_l3935_393585

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers in a range -/
def sumOfDigitsInRange (start finish : ℕ) : ℕ :=
  (List.range (finish - start + 1)).map (fun i => sumOfDigits (start + i))
    |> List.sum

/-- The sum of the digits of all numbers from 0 to 2012 is 28077 -/
theorem sum_of_digits_0_to_2012 :
    sumOfDigitsInRange 0 2012 = 28077 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_0_to_2012_l3935_393585


namespace NUMINAMATH_CALUDE_disco_probabilities_l3935_393509

/-- Represents the content of a music case -/
structure MusicCase where
  disco : ℕ
  techno : ℕ

/-- The probability of selecting a disco tape from a given music case -/
def prob_disco (case : MusicCase) : ℚ :=
  case.disco / (case.disco + case.techno)

/-- The probability of selecting a second disco tape when the first is returned -/
def prob_disco_returned (case : MusicCase) : ℚ :=
  prob_disco case

/-- The probability of selecting a second disco tape when the first is not returned -/
def prob_disco_not_returned (case : MusicCase) : ℚ :=
  (case.disco - 1) / (case.disco + case.techno - 1)

/-- Theorem stating the probabilities for the given scenario -/
theorem disco_probabilities (case : MusicCase) (h : case = ⟨20, 10⟩) :
  prob_disco case = 2/3 ∧
  prob_disco_returned case = 2/3 ∧
  prob_disco_not_returned case = 19/29 := by
  sorry


end NUMINAMATH_CALUDE_disco_probabilities_l3935_393509


namespace NUMINAMATH_CALUDE_unit_rectangle_coverage_l3935_393544

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle parallel to the axes -/
structure Rectangle where
  left : ℝ
  bottom : ℝ
  width : ℝ
  height : ℝ

/-- The theorem stating that 1821 points can be arranged to cover all unit-area rectangles in a 15x15 square -/
theorem unit_rectangle_coverage : ∃ (points : Finset Point),
  (points.card = 1821) ∧ 
  (∀ p : Point, p ∈ points → p.x ≥ 0 ∧ p.x ≤ 15 ∧ p.y ≥ 0 ∧ p.y ≤ 15) ∧
  (∀ r : Rectangle, 
    r.left ≥ 0 ∧ r.left + r.width ≤ 15 ∧ 
    r.bottom ≥ 0 ∧ r.bottom + r.height ≤ 15 ∧
    r.width * r.height = 1 →
    ∃ p : Point, p ∈ points ∧ 
      p.x ≥ r.left ∧ p.x ≤ r.left + r.width ∧
      p.y ≥ r.bottom ∧ p.y ≤ r.bottom + r.height) := by
  sorry


end NUMINAMATH_CALUDE_unit_rectangle_coverage_l3935_393544


namespace NUMINAMATH_CALUDE_unreachable_one_if_not_div_three_l3935_393550

/-- The operation of adding 3 repeatedly until divisible by 5, then dividing by 5 -/
def operation (n : ℕ) : ℕ :=
  let m := n + 3 * (5 - n % 5) % 5
  m / 5

/-- Predicate to check if a number can reach 1 through repeated applications of the operation -/
def can_reach_one (n : ℕ) : Prop :=
  ∃ k : ℕ, (operation^[k] n) = 1

/-- Theorem stating that numbers not divisible by 3 cannot reach 1 through the given operations -/
theorem unreachable_one_if_not_div_three (n : ℕ) (h : ¬ 3 ∣ n) : ¬ can_reach_one n :=
sorry

end NUMINAMATH_CALUDE_unreachable_one_if_not_div_three_l3935_393550


namespace NUMINAMATH_CALUDE_ten_two_zero_one_composite_l3935_393597

theorem ten_two_zero_one_composite (n : ℕ) (h : n > 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 2*n^2 + 1 = a * b :=
sorry

end NUMINAMATH_CALUDE_ten_two_zero_one_composite_l3935_393597


namespace NUMINAMATH_CALUDE_fraction_comparison_l3935_393528

theorem fraction_comparison :
  (373737 : ℚ) / 777777 = 37 / 77 ∧ (41 : ℚ) / 61 < 411 / 611 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3935_393528


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3935_393546

/-- The equation of the tangent line to the curve y = x^3 - 3x^2 + 1 at the point (1, -1) is y = -3x + 2 -/
theorem tangent_line_equation (x y : ℝ) : 
  y = x^3 - 3*x^2 + 1 → -- curve equation
  (1 : ℝ)^3 - 3*(1 : ℝ)^2 + 1 = -1 → -- point (1, -1) satisfies the curve equation
  ∃ (m b : ℝ), 
    (∀ t, y = m*t + b → (t - 1)*(3*(1 : ℝ)^2 - 6*(1 : ℝ)) = y + 1) ∧ -- point-slope form of tangent line
    m = -3 ∧ b = 2 -- coefficients of the tangent line equation
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3935_393546


namespace NUMINAMATH_CALUDE_base_with_final_digit_two_l3935_393567

theorem base_with_final_digit_two : 
  ∃! b : ℕ, 2 ≤ b ∧ b ≤ 20 ∧ 625 % b = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_with_final_digit_two_l3935_393567


namespace NUMINAMATH_CALUDE_fishing_catch_difference_l3935_393518

theorem fishing_catch_difference (father_catch son_catch transfer : ℚ) : 
  (father_catch - transfer = son_catch + transfer) →
  (father_catch + transfer = 2 * (son_catch - transfer)) →
  (father_catch - son_catch) / son_catch = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_fishing_catch_difference_l3935_393518


namespace NUMINAMATH_CALUDE_cookies_distribution_l3935_393558

theorem cookies_distribution (cookies_per_person : ℝ) (total_cookies : ℕ) (h1 : cookies_per_person = 24.0) (h2 : total_cookies = 144) :
  (total_cookies : ℝ) / cookies_per_person = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookies_distribution_l3935_393558


namespace NUMINAMATH_CALUDE_mixed_number_less_than_decimal_l3935_393568

theorem mixed_number_less_than_decimal : -1 - (3 / 5 : ℚ) < -1.5 := by sorry

end NUMINAMATH_CALUDE_mixed_number_less_than_decimal_l3935_393568


namespace NUMINAMATH_CALUDE_mabel_steps_to_helen_l3935_393537

/-- The total number of steps Mabel walks to visit Helen -/
def total_steps (mabel_distance helen_fraction : ℕ) : ℕ :=
  mabel_distance + (helen_fraction * mabel_distance) / 4

/-- Proof that Mabel walks 7875 steps to visit Helen -/
theorem mabel_steps_to_helen :
  total_steps 4500 3 = 7875 := by
  sorry

end NUMINAMATH_CALUDE_mabel_steps_to_helen_l3935_393537


namespace NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l3935_393593

-- Define the possible solids
inductive Solid
| Cone
| Cylinder
| TriangularPyramid
| RectangularPrism

-- Define a property for having a quadrilateral front view
def has_quadrilateral_front_view (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => True
  | Solid.RectangularPrism => True
  | _ => False

-- Theorem statement
theorem quadrilateral_front_view_solids :
  ∀ s : Solid, has_quadrilateral_front_view s ↔ (s = Solid.Cylinder ∨ s = Solid.RectangularPrism) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_front_view_solids_l3935_393593


namespace NUMINAMATH_CALUDE_walnut_trees_before_planting_l3935_393576

theorem walnut_trees_before_planting 
  (initial : ℕ) 
  (planted : ℕ) 
  (final : ℕ) 
  (h1 : planted = 6) 
  (h2 : final = 10) 
  (h3 : final = initial + planted) : 
  initial = 4 :=
by sorry

end NUMINAMATH_CALUDE_walnut_trees_before_planting_l3935_393576


namespace NUMINAMATH_CALUDE_shems_earnings_l3935_393512

/-- Proves that Shem's earnings for an 8-hour workday is $80, given the conditions. -/
theorem shems_earnings (kem_hourly_rate : ℝ) (shem_multiplier : ℝ) (workday_hours : ℕ) :
  kem_hourly_rate = 4 →
  shem_multiplier = 2.5 →
  workday_hours = 8 →
  kem_hourly_rate * shem_multiplier * workday_hours = 80 := by
  sorry

#check shems_earnings

end NUMINAMATH_CALUDE_shems_earnings_l3935_393512


namespace NUMINAMATH_CALUDE_chocolate_bar_problem_l3935_393556

/-- The number of chocolate bars Min bought -/
def min_bars : ℕ := 67

/-- The initial number of chocolate bars in the store -/
def initial_bars : ℕ := 376

/-- The number of chocolate bars Max bought -/
def max_bars : ℕ := min_bars + 41

/-- The number of chocolate bars remaining in the store after purchases -/
def remaining_bars : ℕ := initial_bars - min_bars - max_bars

theorem chocolate_bar_problem :
  min_bars = 67 ∧
  initial_bars = 376 ∧
  max_bars = min_bars + 41 ∧
  remaining_bars = 3 * min_bars :=
sorry

end NUMINAMATH_CALUDE_chocolate_bar_problem_l3935_393556


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3935_393589

theorem unique_positive_solution : 
  ∃! x : ℝ, x > 0 ∧ (1/2) * (4 * x^2 - 4) = (x^2 - 40*x - 8) * (x^2 + 20*x + 4) ∧ x = 20 + Real.sqrt 410 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3935_393589


namespace NUMINAMATH_CALUDE_division_remainder_l3935_393542

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 271 →
  divisor = 30 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 1 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3935_393542


namespace NUMINAMATH_CALUDE_bus_stop_walk_time_l3935_393506

theorem bus_stop_walk_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h : usual_speed > 0) 
  (h1 : usual_time > 0)
  (h2 : (4/5 * usual_speed) * (usual_time + 6) = usual_speed * usual_time) : 
  usual_time = 30 := by
sorry

end NUMINAMATH_CALUDE_bus_stop_walk_time_l3935_393506


namespace NUMINAMATH_CALUDE_octahedron_triangle_count_l3935_393599

/-- The number of vertices in a regular octahedron -/
def octahedron_vertices : ℕ := 6

/-- The number of distinct triangles that can be formed by connecting three different vertices of a regular octahedron -/
def octahedron_triangles : ℕ := Nat.choose octahedron_vertices 3

theorem octahedron_triangle_count : octahedron_triangles = 20 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_triangle_count_l3935_393599


namespace NUMINAMATH_CALUDE_number_of_boys_l3935_393503

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 150 →
  boys + girls = total →
  girls = boys * total / 100 →
  boys = 60 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l3935_393503


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3935_393562

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 11 / 17) 
  (h2 : x - y = 1 / 143) : 
  x^2 - y^2 = 11 / 2431 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3935_393562


namespace NUMINAMATH_CALUDE_fraction_power_six_l3935_393522

theorem fraction_power_six :
  (5 / 3 : ℚ) ^ 6 = 15625 / 729 := by sorry

end NUMINAMATH_CALUDE_fraction_power_six_l3935_393522


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l3935_393502

/-- The number of distinct arrangements of the letters in BANANA -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in BANANA -/
def total_letters : ℕ := 6

/-- The number of occurrences of the letter A in BANANA -/
def num_a : ℕ := 3

/-- The number of occurrences of the letter N in BANANA -/
def num_n : ℕ := 2

/-- The number of occurrences of the letter B in BANANA -/
def num_b : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = (Nat.factorial total_letters) / ((Nat.factorial num_a) * (Nat.factorial num_n)) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l3935_393502


namespace NUMINAMATH_CALUDE_sophia_bus_time_l3935_393555

def sophia_schedule : Prop :=
  let leave_home : Nat := 8 * 60 + 15  -- 8:15 AM in minutes
  let catch_bus : Nat := 8 * 60 + 45   -- 8:45 AM in minutes
  let class_duration : Nat := 55
  let num_classes : Nat := 5
  let lunch_break : Nat := 45
  let club_activities : Nat := 3 * 60  -- 3 hours in minutes
  let arrive_home : Nat := 17 * 60 + 30  -- 5:30 PM in minutes

  let total_away_time : Nat := arrive_home - leave_home
  let school_activities_time : Nat := num_classes * class_duration + lunch_break + club_activities
  let bus_time : Nat := total_away_time - school_activities_time

  bus_time = 25

theorem sophia_bus_time : sophia_schedule := by
  sorry

end NUMINAMATH_CALUDE_sophia_bus_time_l3935_393555


namespace NUMINAMATH_CALUDE_brick_width_calculation_l3935_393525

/-- Calculates the width of a brick given the dimensions of a wall and the number of bricks needed. -/
theorem brick_width_calculation (wall_length wall_width wall_height : ℝ)
  (brick_length brick_height : ℝ) (num_bricks : ℝ) :
  wall_length = 9 →
  wall_width = 5 →
  wall_height = 18.5 →
  brick_length = 0.21 →
  brick_height = 0.08 →
  num_bricks = 4955.357142857142 →
  ∃ (brick_width : ℝ), abs (brick_width - 0.295) < 0.001 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry


end NUMINAMATH_CALUDE_brick_width_calculation_l3935_393525


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3935_393523

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {-3, -4} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3935_393523


namespace NUMINAMATH_CALUDE_middle_zero_product_l3935_393501

theorem middle_zero_product (a b c d : ℕ) : ∃ (x y z w : ℕ), 
  (x ≠ 0 ∧ z ≠ 0 ∧ y ≠ 0 ∧ w ≠ 0) ∧ 
  (100 * x + 0 * 10 + y) * z = 100 * a + 0 * 10 + b ∧
  (100 * x + 0 * 10 + y) * w = 100 * c + d * 10 + e ∧
  d ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_middle_zero_product_l3935_393501


namespace NUMINAMATH_CALUDE_equation_solution_l3935_393534

theorem equation_solution : ∃! x : ℝ, (27 : ℝ) ^ (x - 2) / (9 : ℝ) ^ (x - 1) = (81 : ℝ) ^ (3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3935_393534


namespace NUMINAMATH_CALUDE_alexis_shopping_problem_l3935_393547

/-- Alexis's shopping problem -/
theorem alexis_shopping_problem (budget initial_amount remaining_amount shirt_cost pants_cost socks_cost belt_cost shoes_cost : ℕ) 
  (h1 : initial_amount = 200)
  (h2 : shirt_cost = 30)
  (h3 : pants_cost = 46)
  (h4 : socks_cost = 11)
  (h5 : belt_cost = 18)
  (h6 : shoes_cost = 41)
  (h7 : remaining_amount = 16)
  (h8 : budget = initial_amount - remaining_amount) :
  budget - (shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost) = 38 := by
  sorry

#check alexis_shopping_problem

end NUMINAMATH_CALUDE_alexis_shopping_problem_l3935_393547


namespace NUMINAMATH_CALUDE_octal_2011_equals_base5_13113_l3935_393588

-- Define a function to convert from octal to decimal
def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldr (fun (i, digit) acc => acc + digit * (8 ^ i)) 0

-- Define a function to convert from decimal to base-5
def decimal_to_base5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

-- Theorem statement
theorem octal_2011_equals_base5_13113 :
  decimal_to_base5 (octal_to_decimal [1, 1, 0, 2]) = [3, 1, 1, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_octal_2011_equals_base5_13113_l3935_393588


namespace NUMINAMATH_CALUDE_rectangle_to_square_perimeter_l3935_393591

/-- Given a rectangle that forms a square when its width is doubled and length is halved,
    this theorem relates the perimeter of the resulting square to the original rectangle's perimeter. -/
theorem rectangle_to_square_perimeter (w l P : ℝ) 
  (h1 : w > 0) 
  (h2 : l > 0)
  (h3 : 2 * w = l / 2)  -- Condition for forming a square
  (h4 : P = 4 * (2 * w)) -- Perimeter of the square
  : 2 * (w + l) = 5/4 * P := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_perimeter_l3935_393591


namespace NUMINAMATH_CALUDE_max_value_yzx_l3935_393524

theorem max_value_yzx (x y z : ℝ) 
  (h1 : x^2 + z^2 = 1) 
  (h2 : y^2 + 2*y*(x + z) = 6) : 
  ∃ (M : ℝ), M = Real.sqrt 7 ∧ ∀ (x' y' z' : ℝ), 
    x'^2 + z'^2 = 1 → y'^2 + 2*y'*(x' + z') = 6 → 
    y'*(z' - x') ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_yzx_l3935_393524
