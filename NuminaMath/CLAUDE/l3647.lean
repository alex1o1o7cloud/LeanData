import Mathlib

namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l3647_364741

def f (x : ℝ) : ℝ := x^2 - 1

theorem inequality_holds_iff_m_in_range (m : ℝ) : 
  (∀ x ≥ 3, f (x / m) - 4 * m^2 * f x ≤ f (x - 1) + 4 * f m) ↔ 
  (m ≤ -Real.sqrt 2 / 2 ∨ m ≥ Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l3647_364741


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l3647_364733

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 5^n - 1 ∧ 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ 
    x = 2^(Nat.log 2 x) * 11 * p * q) →
  x = 3124 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l3647_364733


namespace NUMINAMATH_CALUDE_joan_balloons_l3647_364727

/-- The number of blue balloons Joan has after gaining more -/
def total_balloons (initial : ℕ) (gained : ℕ) : ℕ :=
  initial + gained

theorem joan_balloons :
  total_balloons 9 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l3647_364727


namespace NUMINAMATH_CALUDE_inequality_proof_l3647_364700

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_condition : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3647_364700


namespace NUMINAMATH_CALUDE_final_answer_calculation_l3647_364744

theorem final_answer_calculation (chosen_number : ℕ) (h : chosen_number = 800) : 
  (chosen_number / 5 : ℚ) - 154 = 6 := by
  sorry

end NUMINAMATH_CALUDE_final_answer_calculation_l3647_364744


namespace NUMINAMATH_CALUDE_find_number_l3647_364776

theorem find_number (G N : ℕ) (h1 : G = 129) (h2 : N % G = 9) (h3 : 2206 % G = 13) : N = 2202 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3647_364776


namespace NUMINAMATH_CALUDE_percentage_problem_l3647_364767

theorem percentage_problem (n : ℝ) (p : ℝ) : 
  n = 50 → 
  p / 100 * n = 30 / 100 * 10 + 27 → 
  p = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3647_364767


namespace NUMINAMATH_CALUDE_soccer_substitutions_l3647_364781

theorem soccer_substitutions (total_players : ℕ) (starting_players : ℕ) (non_playing_players : ℕ) :
  total_players = 24 →
  starting_players = 11 →
  non_playing_players = 7 →
  ∃ (first_half_subs : ℕ),
    first_half_subs = 2 ∧
    total_players = starting_players + first_half_subs + 2 * first_half_subs + non_playing_players :=
by sorry

end NUMINAMATH_CALUDE_soccer_substitutions_l3647_364781


namespace NUMINAMATH_CALUDE_black_balloons_problem_l3647_364783

theorem black_balloons_problem (x y : ℝ) (h1 : x = 4 * y) (h2 : x = 7.0) : y = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_black_balloons_problem_l3647_364783


namespace NUMINAMATH_CALUDE_line_plane_relationships_l3647_364794

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields

/-- Defines when a line is parallel to a plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- Defines when two lines are parallel -/
def lines_parallel (l1 l2 : Line3D) : Prop := sorry

/-- Defines when a line intersects a plane -/
def line_intersects_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- Theorem representing the four statements -/
theorem line_plane_relationships :
  (∀ (a b : Line3D) (α : Plane),
    line_parallel_to_plane a α → line_in_plane b α → lines_parallel a b) = False
  ∧
  (∀ (a b : Line3D) (α : Plane) (P : Point),
    line_intersects_plane a α → line_in_plane b α → ¬lines_parallel a b) = True
  ∧
  (∀ (a : Line3D) (α : Plane),
    ¬line_in_plane a α → line_parallel_to_plane a α) = False
  ∧
  (∀ (a b : Line3D) (α : Plane),
    line_parallel_to_plane a α → line_parallel_to_plane b α → lines_parallel a b) = False :=
by sorry

end NUMINAMATH_CALUDE_line_plane_relationships_l3647_364794


namespace NUMINAMATH_CALUDE_town_population_increase_l3647_364790

/-- Calculates the average percent increase of population per year given initial and final populations over a decade. -/
def average_percent_increase (initial_population final_population : ℕ) : ℚ :=
  let total_increase : ℕ := final_population - initial_population
  let average_annual_increase : ℚ := total_increase / 10
  (average_annual_increase / initial_population) * 100

/-- Theorem stating that the average percent increase of population per year is 7% for the given conditions. -/
theorem town_population_increase :
  average_percent_increase 175000 297500 = 7 := by
  sorry

#eval average_percent_increase 175000 297500

end NUMINAMATH_CALUDE_town_population_increase_l3647_364790


namespace NUMINAMATH_CALUDE_fraction_modification_l3647_364787

theorem fraction_modification (a b c d x y : ℚ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + y) = c / d) : 
  x = (b * c - a * d + y * c) / d := by
  sorry

end NUMINAMATH_CALUDE_fraction_modification_l3647_364787


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3647_364775

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3}

theorem union_of_M_and_N :
  M ∪ N = {1, 2, 3} :=
by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3647_364775


namespace NUMINAMATH_CALUDE_initial_playtime_l3647_364792

/-- Proof of initial daily playtime in a game scenario -/
theorem initial_playtime (initial_days : ℕ) (initial_completion_percent : ℚ)
  (remaining_days : ℕ) (remaining_hours_per_day : ℕ) :
  initial_days = 14 →
  initial_completion_percent = 2/5 →
  remaining_days = 12 →
  remaining_hours_per_day = 7 →
  ∃ (x : ℚ),
    x * initial_days = initial_completion_percent * (x * initial_days + remaining_days * remaining_hours_per_day) ∧
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_playtime_l3647_364792


namespace NUMINAMATH_CALUDE_chorus_students_l3647_364750

theorem chorus_students (total : Nat) (band : Nat) (both : Nat) (neither : Nat) :
  total = 50 →
  band = 26 →
  both = 2 →
  neither = 8 →
  ∃ chorus : Nat, chorus = 18 ∧ chorus + band - both = total - neither :=
by sorry

end NUMINAMATH_CALUDE_chorus_students_l3647_364750


namespace NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_9_and_tens_greater_l3647_364739

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that returns the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ :=
  n / 10

/-- A function that returns the units digit of a two-digit number -/
def units_digit (n : ℕ) : ℕ :=
  n % 10

/-- The main theorem -/
theorem no_two_digit_primes_with_digit_sum_9_and_tens_greater : 
  ∀ n : ℕ, 10 ≤ n → n < 100 → 
    (tens_digit n + units_digit n = 9 ∧ tens_digit n > units_digit n) → 
    ¬(is_prime n) :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_9_and_tens_greater_l3647_364739


namespace NUMINAMATH_CALUDE_homework_policy_for_25_points_l3647_364782

def homework_assignments (n : ℕ) : ℕ :=
  if n ≤ 3 then 0
  else ((n - 3 - 1) / 5 + 1)

def total_assignments (total_points : ℕ) : ℕ :=
  (List.range total_points).map homework_assignments |>.sum

theorem homework_policy_for_25_points :
  total_assignments 25 = 60 := by
  sorry

end NUMINAMATH_CALUDE_homework_policy_for_25_points_l3647_364782


namespace NUMINAMATH_CALUDE_special_matrix_exists_iff_even_l3647_364749

/-- A matrix with elements from {-1, 0, 1} -/
def SpecialMatrix (n : ℕ) := Matrix (Fin n) (Fin n) (Fin 3)

/-- The sum of elements in a row of a SpecialMatrix -/
def rowSum (A : SpecialMatrix n) (i : Fin n) : ℤ := sorry

/-- The sum of elements in a column of a SpecialMatrix -/
def colSum (A : SpecialMatrix n) (j : Fin n) : ℤ := sorry

/-- All row and column sums are distinct -/
def distinctSums (A : SpecialMatrix n) : Prop :=
  ∀ i j i' j', (i ≠ i' ∨ j ≠ j') → 
    (rowSum A i ≠ rowSum A i' ∧ 
     rowSum A i ≠ colSum A j' ∧ 
     colSum A j ≠ rowSum A i' ∧ 
     colSum A j ≠ colSum A j')

theorem special_matrix_exists_iff_even (n : ℕ) :
  (∃ A : SpecialMatrix n, distinctSums A) ↔ (∃ k : ℕ, n = 2 * k) :=
sorry

end NUMINAMATH_CALUDE_special_matrix_exists_iff_even_l3647_364749


namespace NUMINAMATH_CALUDE_fraction_begins_with_0_239_l3647_364710

/-- The infinite decimal a --/
def a : ℝ := 0.1234567891011

/-- The infinite decimal b --/
def b : ℝ := 0.51504948

/-- Theorem stating that the fraction a/b begins with 0.239 --/
theorem fraction_begins_with_0_239 (h1 : 0.515 < b) (h2 : b < 0.516) :
  0.239 * b ≤ a ∧ a < 0.24 * b := by sorry

end NUMINAMATH_CALUDE_fraction_begins_with_0_239_l3647_364710


namespace NUMINAMATH_CALUDE_min_a_value_l3647_364766

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x

-- Define the relationship between f, g, and 2^x
axiom f_g_sum : ∀ x ∈ Set.Icc 1 2, f x + g x = 2^x

-- Define the inequality condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, a * f x + g (2*x) ≥ 0

-- State the theorem
theorem min_a_value :
  ∃ a_min : ℝ, a_min = -17/6 ∧
  (∀ a, inequality_holds a ↔ a ≥ a_min) :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l3647_364766


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3647_364789

/-- A geometric sequence {a_n} satisfying the given conditions -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧
  a 6 = 8 * a 3 ∧
  a 6 = 8 * (a 2)^2

/-- The theorem stating the general term of the geometric sequence -/
theorem geometric_sequence_general_term {a : ℕ → ℝ} (h : geometric_sequence a) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3647_364789


namespace NUMINAMATH_CALUDE_minesweeper_configurations_l3647_364777

def valid_configuration (A B C D E : ℕ) : Prop :=
  A + B = 2 ∧ B + C + D = 1 ∧ D + E = 2

def count_configurations : ℕ := sorry

theorem minesweeper_configurations :
  count_configurations = 4545 := by sorry

end NUMINAMATH_CALUDE_minesweeper_configurations_l3647_364777


namespace NUMINAMATH_CALUDE_andrew_payment_l3647_364785

/-- The total amount Andrew paid to the shopkeeper -/
def total_amount (grape_price grape_quantity mango_price mango_quantity : ℕ) : ℕ :=
  grape_price * grape_quantity + mango_price * mango_quantity

/-- Theorem: Andrew paid 975 to the shopkeeper -/
theorem andrew_payment : total_amount 74 6 59 9 = 975 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l3647_364785


namespace NUMINAMATH_CALUDE_triangle_with_long_altitudes_l3647_364717

theorem triangle_with_long_altitudes (a b c : ℝ) (ma mb : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_altitudes : ma ≥ a ∧ mb ≥ b)
  (h_area : a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) / 2 = a * ma / 2)
  (h_area_alt : a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) / 2 = b * mb / 2) :
  a = b ∧ c^2 = 2 * a^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_long_altitudes_l3647_364717


namespace NUMINAMATH_CALUDE_find_number_multiplied_by_9999_l3647_364759

theorem find_number_multiplied_by_9999 :
  ∃! x : ℤ, x * 9999 = 724807415 :=
by
  sorry

end NUMINAMATH_CALUDE_find_number_multiplied_by_9999_l3647_364759


namespace NUMINAMATH_CALUDE_summer_mowing_times_l3647_364755

/-- The number of times Kale mowed his lawn in the summer -/
def summer_mowing : ℕ := 5

/-- The number of times Kale mowed his lawn in the spring -/
def spring_mowing : ℕ := 8

/-- The difference between spring and summer mowing times -/
def mowing_difference : ℕ := 3

theorem summer_mowing_times : 
  spring_mowing - summer_mowing = mowing_difference := by sorry

end NUMINAMATH_CALUDE_summer_mowing_times_l3647_364755


namespace NUMINAMATH_CALUDE_large_cube_pieces_l3647_364754

/-- The number of wire pieces needed for a cube framework -/
def wire_pieces (n : ℕ) : ℕ := 3 * (n + 1)^2 * n

/-- The fact that a 2 × 2 × 2 cube uses 54 wire pieces -/
axiom small_cube_pieces : wire_pieces 2 = 54

/-- Theorem: The number of wire pieces needed for a 10 × 10 × 10 cube is 3630 -/
theorem large_cube_pieces : wire_pieces 10 = 3630 := by
  sorry

end NUMINAMATH_CALUDE_large_cube_pieces_l3647_364754


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3647_364713

theorem absolute_value_equality (x : ℝ) : |x - 3| = |x - 5| → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3647_364713


namespace NUMINAMATH_CALUDE_sum_of_angles_l3647_364740

theorem sum_of_angles (angle1 angle2 angle3 angle4 angle5 angle6 angleA angleB angleC : ℝ) :
  angle1 + angle3 + angle5 = 180 →
  angle2 + angle4 + angle6 = 180 →
  angleA + angleB + angleC = 180 →
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angleA + angleB + angleC = 540 := by
sorry

end NUMINAMATH_CALUDE_sum_of_angles_l3647_364740


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l3647_364738

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * x - 7 * y = 35

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ := (0, -5)

/-- Theorem: The point (0, -5) is the intersection of the line 5x - 7y = 35 with the y-axis -/
theorem line_y_axis_intersection :
  line_equation intersection_point.1 intersection_point.2 ∧
  y_axis intersection_point.1 := by
  sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l3647_364738


namespace NUMINAMATH_CALUDE_class_size_l3647_364748

theorem class_size (total_average : ℝ) (group1_size : ℕ) (group1_average : ℝ)
                   (group2_size : ℕ) (group2_average : ℝ) (last_student_age : ℕ) :
  total_average = 15 →
  group1_size = 5 →
  group1_average = 12 →
  group2_size = 9 →
  group2_average = 16 →
  last_student_age = 21 →
  ∃ (n : ℕ), n = 15 ∧ n * total_average = group1_size * group1_average + group2_size * group2_average + last_student_age :=
by
  sorry

#check class_size

end NUMINAMATH_CALUDE_class_size_l3647_364748


namespace NUMINAMATH_CALUDE_distance_to_pole_for_given_point_l3647_364723

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The distance from a point to the pole in polar coordinates -/
def distanceToPole (p : PolarPoint) : ℝ := p.r

theorem distance_to_pole_for_given_point :
  let A : PolarPoint := { r := 3, θ := -4 }
  distanceToPole A = 3 := by sorry

end NUMINAMATH_CALUDE_distance_to_pole_for_given_point_l3647_364723


namespace NUMINAMATH_CALUDE_fermat_1000_units_digit_l3647_364721

/-- Fermat number F_n is defined as 2^(2^n) + 1 -/
def fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem fermat_1000_units_digit :
  units_digit (fermat_number 1000) = 7 := by sorry

end NUMINAMATH_CALUDE_fermat_1000_units_digit_l3647_364721


namespace NUMINAMATH_CALUDE_range_of_a_l3647_364702

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 3, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0

-- Define the set of a values that satisfy the conditions
def A : Set ℝ := {a | (p a ∨ q a) ∧ ¬(p a ∧ q a)}

-- Theorem statement
theorem range_of_a : A = Set.Icc (-1) 1 ∪ Set.Ioi 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3647_364702


namespace NUMINAMATH_CALUDE_two_part_number_problem_l3647_364742

theorem two_part_number_problem (x y k : ℕ) : 
  x + y = 24 →
  x = 13 →
  k * x + 5 * y = 146 →
  k = 7 :=
by sorry

end NUMINAMATH_CALUDE_two_part_number_problem_l3647_364742


namespace NUMINAMATH_CALUDE_total_scissors_is_86_l3647_364745

/-- Calculates the total number of scissors after changes in two drawers -/
def totalScissorsAfterChanges (
  initialScissors1 : ℕ) (initialScissors2 : ℕ) 
  (addedScissors1 : ℕ) (addedScissors2 : ℕ) : ℕ :=
  (initialScissors1 + addedScissors1) + (initialScissors2 + addedScissors2)

/-- Proves that the total number of scissors after changes is 86 -/
theorem total_scissors_is_86 :
  totalScissorsAfterChanges 39 27 13 7 = 86 := by
  sorry

end NUMINAMATH_CALUDE_total_scissors_is_86_l3647_364745


namespace NUMINAMATH_CALUDE_right_triangle_sum_of_legs_l3647_364771

theorem right_triangle_sum_of_legs (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 50 →           -- Length of hypotenuse
  (a * b) / 2 = 600 →  -- Area of the triangle
  a + b = 70 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sum_of_legs_l3647_364771


namespace NUMINAMATH_CALUDE_min_value_of_f_l3647_364708

/-- The quadratic function f(x) = 2x^2 + 8x + 7 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 7

/-- Theorem: The minimum value of f(x) = 2x^2 + 8x + 7 is -1 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -1 ∧ ∀ (x : ℝ), f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3647_364708


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_h_geq_one_l3647_364763

/-- A quadratic function of the form y = (x - h)^2 + 3 -/
def quadratic_function (h : ℝ) (x : ℝ) : ℝ := (x - h)^2 + 3

/-- The derivative of the quadratic function -/
def quadratic_derivative (h : ℝ) (x : ℝ) : ℝ := 2 * (x - h)

theorem quadratic_decreasing_implies_h_geq_one (h : ℝ) :
  (∀ x < 1, quadratic_derivative h x < 0) → h ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_h_geq_one_l3647_364763


namespace NUMINAMATH_CALUDE_hit_target_probability_l3647_364772

/-- The probability of hitting a target at least 2 times out of 3 independent shots,
    given that the probability of hitting the target in a single shot is 0.6 -/
theorem hit_target_probability :
  let p : ℝ := 0.6  -- Probability of hitting the target in a single shot
  let n : ℕ := 3    -- Total number of shots
  let k : ℕ := 2    -- Minimum number of successful hits

  -- Probability of hitting the target at least k times out of n shots
  (Finset.sum (Finset.range (n - k + 1)) (fun i =>
    (n.choose (k + i)) * p^(k + i) * (1 - p)^(n - k - i))) = 81 / 125 :=
by sorry

end NUMINAMATH_CALUDE_hit_target_probability_l3647_364772


namespace NUMINAMATH_CALUDE_complex_magnitude_l3647_364718

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = -1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3647_364718


namespace NUMINAMATH_CALUDE_food_drive_problem_l3647_364761

/-- Represents the food drive problem in Ms. Perez's class -/
theorem food_drive_problem (total_students : ℕ) (half_students_12_cans : ℕ) (students_4_cans : ℕ) (total_cans : ℕ) :
  total_students = 30 →
  half_students_12_cans = total_students / 2 →
  students_4_cans = 13 →
  total_cans = 232 →
  half_students_12_cans * 12 + students_4_cans * 4 = total_cans →
  total_students - (half_students_12_cans + students_4_cans) = 2 :=
by sorry

end NUMINAMATH_CALUDE_food_drive_problem_l3647_364761


namespace NUMINAMATH_CALUDE_initial_number_proof_l3647_364732

theorem initial_number_proof : ∃ x : ℤ, x - 10 * 2 * 5 = 10011 ∧ x = 10111 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l3647_364732


namespace NUMINAMATH_CALUDE_triangle_inequality_l3647_364798

theorem triangle_inequality (a b c S : ℝ) (n : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (hn : n ≥ 1) (hS : 2 * S = a + b + c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a^n)/(b+c) + (b^n)/(c+a) + (c^n)/(a+b) ≥ (2/3)^(n-2) * S^(n-1) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3647_364798


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3647_364756

theorem complex_equation_solution (z : ℂ) : z / (1 - 2*I) = I → z = 2 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3647_364756


namespace NUMINAMATH_CALUDE_percent_increase_l3647_364780

theorem percent_increase (N M P : ℝ) (h : M = N * (1 + P / 100)) :
  (M - N) / N * 100 = P := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_l3647_364780


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3647_364728

-- Define the conditions
def condition_p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 > 0

def condition_q (m : ℝ) : Prop := ∀ x y, x < y → (m+3)^x < (m+3)^y

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, condition_p m ∧ condition_q m) ∧
  (∃ m : ℝ, ¬condition_p m ∧ condition_q m) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3647_364728


namespace NUMINAMATH_CALUDE_circle_circumference_area_ratio_l3647_364751

/-- The ratio of circumference to area for a circle with radius 16 is 1/8 -/
theorem circle_circumference_area_ratio : 
  let r : ℝ := 16
  let circumference := 2 * Real.pi * r
  let area := Real.pi * r^2
  circumference / area = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_circle_circumference_area_ratio_l3647_364751


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_real_roots_l3647_364765

theorem quadratic_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_real_roots_l3647_364765


namespace NUMINAMATH_CALUDE_largest_common_term_l3647_364746

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem largest_common_term :
  ∃ (n m : ℕ),
    179 = arithmetic_sequence 3 8 n ∧
    179 = arithmetic_sequence 5 9 m ∧
    179 ≤ 200 ∧
    ∀ (k : ℕ), k > 179 →
      k ≤ 200 →
      (∀ (p q : ℕ), k ≠ arithmetic_sequence 3 8 p ∨ k ≠ arithmetic_sequence 5 9 q) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l3647_364746


namespace NUMINAMATH_CALUDE_never_sunday_date_l3647_364705

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a month of the year -/
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August
| September
| October
| November
| December

/-- Function to determine the day of the week for a given date in a month -/
def dayOfWeek (date : Nat) (month : Month) (isLeapYear : Bool) : DayOfWeek :=
  sorry

/-- Theorem stating that 31 is the only date that can never be a Sunday in any month of a year -/
theorem never_sunday_date :
  ∀ (date : Nat),
    (∀ (month : Month) (isLeapYear : Bool),
      dayOfWeek date month isLeapYear ≠ DayOfWeek.Sunday) ↔ date = 31 :=
by sorry

end NUMINAMATH_CALUDE_never_sunday_date_l3647_364705


namespace NUMINAMATH_CALUDE_matrix_determinant_l3647_364764

theorem matrix_determinant : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3/2; 2, 6]
  Matrix.det A = 27 := by
sorry

end NUMINAMATH_CALUDE_matrix_determinant_l3647_364764


namespace NUMINAMATH_CALUDE_jo_bob_max_height_l3647_364735

/-- Represents the state of a hot air balloon ride -/
structure BalloonRide where
  ascent_rate : ℝ
  descent_rate : ℝ
  first_pull_time : ℝ
  release_time : ℝ
  second_pull_time : ℝ

/-- Calculates the maximum height reached during a balloon ride -/
def max_height (ride : BalloonRide) : ℝ :=
  let first_ascent := ride.ascent_rate * ride.first_pull_time
  let descent := ride.descent_rate * ride.release_time
  let second_ascent := ride.ascent_rate * ride.second_pull_time
  first_ascent - descent + second_ascent

/-- Theorem stating the maximum height reached in Jo-Bob's balloon ride -/
theorem jo_bob_max_height :
  let ride : BalloonRide := {
    ascent_rate := 50,
    descent_rate := 10,
    first_pull_time := 15,
    release_time := 10,
    second_pull_time := 15
  }
  max_height ride = 1400 := by
  sorry


end NUMINAMATH_CALUDE_jo_bob_max_height_l3647_364735


namespace NUMINAMATH_CALUDE_number_of_divisors_l3647_364797

theorem number_of_divisors 
  (n : ℕ+) 
  (p₁ p₂ p₃ : ℕ) 
  (α β γ : ℕ+) 
  (h_prime : Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃)
  (h_distinct : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h_decomp : (n : ℕ) = p₁ ^ (α : ℕ) * p₂ ^ (β : ℕ) * p₃ ^ (γ : ℕ)) :
  (Finset.card (Nat.divisors (n : ℕ)) : ℕ) = (α + 1) * (β + 1) * (γ + 1) := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_l3647_364797


namespace NUMINAMATH_CALUDE_juice_boxes_for_school_year_l3647_364734

/-- Calculate the total number of juice boxes needed for a school year -/
def total_juice_boxes (num_children : ℕ) (school_days_per_week : ℕ) (weeks_in_school_year : ℕ) : ℕ :=
  num_children * school_days_per_week * weeks_in_school_year

/-- Theorem: Given the specific conditions, the total number of juice boxes needed is 375 -/
theorem juice_boxes_for_school_year :
  let num_children : ℕ := 3
  let school_days_per_week : ℕ := 5
  let weeks_in_school_year : ℕ := 25
  total_juice_boxes num_children school_days_per_week weeks_in_school_year = 375 := by
  sorry


end NUMINAMATH_CALUDE_juice_boxes_for_school_year_l3647_364734


namespace NUMINAMATH_CALUDE_min_a_plus_b_l3647_364703

theorem min_a_plus_b (x y a b : ℝ) : 
  2*x - y + 2 ≥ 0 →
  8*x - y - 4 ≤ 0 →
  x ≥ 0 →
  y ≥ 0 →
  a > 0 →
  b > 0 →
  (∀ x' y', 2*x' - y' + 2 ≥ 0 → 8*x' - y' - 4 ≤ 0 → x' ≥ 0 → y' ≥ 0 → a*x' + y' ≤ 8) →
  a*x + y = 8 →
  a + b ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_a_plus_b_l3647_364703


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l3647_364730

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit in one dimension -/
def maxTilesInDimension (floorDim tileADim tileBDim : ℕ) : ℕ :=
  max (floorDim / tileADim) (floorDim / tileBDim)

/-- Calculates the total number of tiles for a given orientation -/
def totalTiles (floor tile : Dimensions) : ℕ :=
  (maxTilesInDimension floor.length tile.length tile.width) *
  (maxTilesInDimension floor.width tile.length tile.width)

/-- The main theorem stating the maximum number of tiles that can be accommodated -/
theorem max_tiles_on_floor :
  let floor := Dimensions.mk 180 120
  let tile := Dimensions.mk 25 16
  (max (totalTiles floor tile) (totalTiles floor (Dimensions.mk tile.width tile.length))) = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l3647_364730


namespace NUMINAMATH_CALUDE_coin_bag_total_l3647_364731

theorem coin_bag_total (p : ℕ) : ∃ (p : ℕ), 
  (0.01 * p + 0.05 * (3 * p) + 0.10 * (4 * 3 * p) : ℚ) = 408 := by
  sorry

end NUMINAMATH_CALUDE_coin_bag_total_l3647_364731


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3647_364753

theorem ceiling_floor_difference : 
  ⌈(14 : ℚ) / 5 * (-31 : ℚ) / 3⌉ - ⌊(14 : ℚ) / 5 * ⌊(-31 : ℚ) / 3⌋⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3647_364753


namespace NUMINAMATH_CALUDE_investment_problem_l3647_364711

/-- Given a sum P invested at a rate R for 20 years, if investing at a rate (R + 10)%
    yields Rs. 3000 more in interest, then P = 1500. -/
theorem investment_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  P * (R + 10) * 20 / 100 = P * R * 20 / 100 + 3000 →
  P = 1500 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l3647_364711


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3647_364784

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 < 0 → x > a) ∧
  (∃ x : ℝ, x > a ∧ ¬(x^2 - 2*x - 3 < 0)) →
  a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3647_364784


namespace NUMINAMATH_CALUDE_square_sum_inequality_square_sum_equality_l3647_364752

theorem square_sum_inequality {a b c d : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
by sorry

theorem square_sum_equality {a b c d : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_square_sum_inequality_square_sum_equality_l3647_364752


namespace NUMINAMATH_CALUDE_max_value_and_sum_l3647_364770

theorem max_value_and_sum (x y z v w : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0 ∧ v > 0 ∧ w > 0) 
  (heq : 4 * x^2 + y^2 + z^2 + v^2 + w^2 = 8080) :
  (∃ (M : ℝ), ∀ (x' y' z' v' w' : ℝ), 
    x' > 0 → y' > 0 → z' > 0 → v' > 0 → w' > 0 →
    4 * x'^2 + y'^2 + z'^2 + v'^2 + w'^2 = 8080 →
    x' * z' + 4 * y' * z' + 6 * z' * v' + 14 * z' * w' ≤ M ∧
    M = 60480 * Real.sqrt 249) ∧
  (∃ (x_M y_M z_M v_M w_M : ℝ),
    x_M > 0 ∧ y_M > 0 ∧ z_M > 0 ∧ v_M > 0 ∧ w_M > 0 ∧
    4 * x_M^2 + y_M^2 + z_M^2 + v_M^2 + w_M^2 = 8080 ∧
    x_M * z_M + 4 * y_M * z_M + 6 * z_M * v_M + 14 * z_M * w_M = 60480 * Real.sqrt 249 ∧
    60480 * Real.sqrt 249 + x_M + y_M + z_M + v_M + w_M = 280 + 60600 * Real.sqrt 249) := by
  sorry

end NUMINAMATH_CALUDE_max_value_and_sum_l3647_364770


namespace NUMINAMATH_CALUDE_unique_number_exists_l3647_364799

def is_valid_number (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range 10, n % (k + 3) = k + 2

theorem unique_number_exists : 
  ∃! n : ℕ, is_valid_number n ∧ n = 27719 := by sorry

end NUMINAMATH_CALUDE_unique_number_exists_l3647_364799


namespace NUMINAMATH_CALUDE_rod_length_difference_l3647_364720

theorem rod_length_difference (L₁ L₂ : ℝ) : 
  L₁ + L₂ = 33 →
  (1 - 1/3) * L₁ = (1 - 1/5) * L₂ →
  L₁ - L₂ = 3 := by
sorry

end NUMINAMATH_CALUDE_rod_length_difference_l3647_364720


namespace NUMINAMATH_CALUDE_simplify_fraction_l3647_364712

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  (12 * x^2 * y^3) / (9 * x * y^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3647_364712


namespace NUMINAMATH_CALUDE_unique_solution_l3647_364795

def A (x : ℝ) : Set ℝ := {x^2, x+1, -3}
def B (x : ℝ) : Set ℝ := {x-5, 2*x-1, x^2+1}

theorem unique_solution : 
  ∃! x : ℝ, A x ∩ B x = {-3} ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3647_364795


namespace NUMINAMATH_CALUDE_parabola_directrix_l3647_364786

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Define the line containing the focus
def focus_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem parabola_directrix (a : ℝ) :
  (∃ x y, focus_line x y ∧ (x = 0 ∨ y = 0)) →
  (∃ x, ∀ y, y = parabola a x ↔ y + 1 = 2 * (parabola a (x/2))) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3647_364786


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3647_364769

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * s = 4 * Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3647_364769


namespace NUMINAMATH_CALUDE_t_minus_s_eq_negative_19_583_l3647_364726

/-- The number of students in the school -/
def num_students : ℕ := 120

/-- The number of teachers in the school -/
def num_teachers : ℕ := 6

/-- The list of class enrollments -/
def class_enrollments : List ℕ := [60, 30, 10, 10, 5, 5]

/-- The average number of students per teacher -/
def t : ℚ := (num_students : ℚ) / num_teachers

/-- The average number of students per student -/
noncomputable def s : ℚ := (class_enrollments.map (λ x => x * x)).sum / num_students

/-- The difference between t and s -/
theorem t_minus_s_eq_negative_19_583 : t - s = -19583 / 1000 := by sorry

end NUMINAMATH_CALUDE_t_minus_s_eq_negative_19_583_l3647_364726


namespace NUMINAMATH_CALUDE_factorial_division_l3647_364706

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_division : 
  factorial 8 / factorial (8 - 2) = 56 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3647_364706


namespace NUMINAMATH_CALUDE_largest_six_digit_divisible_by_41_l3647_364707

theorem largest_six_digit_divisible_by_41 : 
  ∀ n : ℕ, n ≤ 999999 ∧ n % 41 = 0 → n ≤ 999990 :=
by sorry

end NUMINAMATH_CALUDE_largest_six_digit_divisible_by_41_l3647_364707


namespace NUMINAMATH_CALUDE_park_shape_l3647_364719

theorem park_shape (total_cost : ℕ) (cost_per_side : ℕ) (h1 : total_cost = 224) (h2 : cost_per_side = 56) :
  total_cost / cost_per_side = 4 :=
by sorry

end NUMINAMATH_CALUDE_park_shape_l3647_364719


namespace NUMINAMATH_CALUDE_no_solution_exists_l3647_364724

theorem no_solution_exists : ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (3 / a + 4 / b = 12 / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3647_364724


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3647_364715

theorem triangle_perimeter (a b c : ℝ) : 
  a = 2 ∧ b = 5 ∧ 
  c^2 - 8*c + 12 = 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3647_364715


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3647_364704

/-- The eccentricity of a hyperbola with equation y²/4 - x² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  ∃ (e : ℝ), e = (Real.sqrt 5) / 2 ∧ 
  ∀ (x y : ℝ), y^2 / 4 - x^2 = 1 → 
  e = Real.sqrt ((y^2 / 4) + x^2) / (y / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3647_364704


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3647_364760

theorem bowling_ball_weight :
  ∀ (ball_weight canoe_weight : ℚ),
    9 * ball_weight = 4 * canoe_weight →
    3 * canoe_weight = 112 →
    ball_weight = 448 / 27 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3647_364760


namespace NUMINAMATH_CALUDE_smallest_block_size_l3647_364774

/-- Given a rectangular block of dimensions l × m × n formed by N unit cubes,
    where (l - 1) × (m - 1) × (n - 1) = 143, the smallest possible value of N is 336. -/
theorem smallest_block_size (l m n : ℕ) (h : (l - 1) * (m - 1) * (n - 1) = 143) :
  ∃ (N : ℕ), N = l * m * n ∧ N = 336 ∧ ∀ (l' m' n' : ℕ), 
    ((l' - 1) * (m' - 1) * (n' - 1) = 143) → l' * m' * n' ≥ N :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_size_l3647_364774


namespace NUMINAMATH_CALUDE_divisibility_by_37_l3647_364768

theorem divisibility_by_37 (a b c : ℕ) :
  (37 ∣ (100 * a + 10 * b + c)) →
  (37 ∣ (100 * b + 10 * c + a)) ∧
  (37 ∣ (100 * c + 10 * a + b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_37_l3647_364768


namespace NUMINAMATH_CALUDE_axis_of_symmetry_can_be_left_of_y_axis_l3647_364729

theorem axis_of_symmetry_can_be_left_of_y_axis :
  ∃ (a : ℝ), a > 0 ∧ ∃ (x : ℝ), x < 0 ∧
    x = -(1 - 2*a) / (2*a) ∧
    ∀ (y : ℝ), y = a*x^2 + (1 - 2*a)*x :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_can_be_left_of_y_axis_l3647_364729


namespace NUMINAMATH_CALUDE_eighth_pitch_frequency_l3647_364716

/-- Twelve-tone Equal Temperament system -/
structure TwelveToneEqualTemperament where
  /-- The frequency ratio between consecutive pitches -/
  ratio : ℝ
  /-- The ratio is the twelfth root of 2 -/
  ratio_def : ratio = Real.rpow 2 (1/12)

/-- The frequency of a pitch in the Twelve-tone Equal Temperament system -/
def frequency (system : TwelveToneEqualTemperament) (first_pitch : ℝ) (n : ℕ) : ℝ :=
  first_pitch * (system.ratio ^ (n - 1))

/-- Theorem: The frequency of the eighth pitch is the seventh root of 2 times the first pitch -/
theorem eighth_pitch_frequency (system : TwelveToneEqualTemperament) (f : ℝ) :
  frequency system f 8 = f * Real.rpow 2 (7/12) := by
  sorry

end NUMINAMATH_CALUDE_eighth_pitch_frequency_l3647_364716


namespace NUMINAMATH_CALUDE_x_equals_five_l3647_364737

/-- A composite rectangular figure with specific segment lengths -/
structure CompositeRectangle where
  top_left : ℝ
  top_middle : ℝ
  top_right : ℝ
  bottom_left : ℝ
  bottom_middle : ℝ
  bottom_right : ℝ

/-- The theorem stating that X equals 5 in the given composite rectangle -/
theorem x_equals_five (r : CompositeRectangle) 
  (h1 : r.top_left = 3)
  (h2 : r.top_right = 4)
  (h3 : r.bottom_left = 5)
  (h4 : r.bottom_middle = 7)
  (h5 : r.top_middle = r.bottom_right)
  (h6 : r.top_left + r.top_middle + r.top_right = r.bottom_left + r.bottom_middle + r.bottom_right) :
  r.top_middle = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_five_l3647_364737


namespace NUMINAMATH_CALUDE_gcd_lcm_392_count_l3647_364779

theorem gcd_lcm_392_count : 
  ∃! n : ℕ, n > 0 ∧ 
  (∃ S : Finset ℕ, S.card = n ∧
    ∀ d ∈ S, d > 0 ∧
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧
      Nat.gcd a b * Nat.lcm a b = 392 ∧
      Nat.gcd a b = d) ∧
    (∀ a b : ℕ, a > 0 → b > 0 →
      Nat.gcd a b * Nat.lcm a b = 392 →
      Nat.gcd a b ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_gcd_lcm_392_count_l3647_364779


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3647_364757

theorem hyperbola_sum (h k a b c : ℝ) : 
  (h = -2) →
  (k = 0) →
  (c = Real.sqrt 34) →
  (a = 3) →
  (c^2 = a^2 + b^2) →
  (h + k + a + b = 6) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3647_364757


namespace NUMINAMATH_CALUDE_range_of_m_l3647_364788

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : x + 2*y - x*y = 0) 
  (h_ineq : ∀ m : ℝ, x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3647_364788


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3647_364758

/-- An arithmetic sequence with first term 2 and sum of first 3 terms equal to 12 has its 6th term equal to 12 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term is 2
  a 1 + a 2 + a 3 = 12 →                -- sum of first 3 terms is 12
  a 6 = 12 := by                        -- 6th term is 12
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3647_364758


namespace NUMINAMATH_CALUDE_sam_has_six_balloons_l3647_364793

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := total_balloons - fred_balloons - mary_balloons

theorem sam_has_six_balloons : sam_balloons = 6 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_six_balloons_l3647_364793


namespace NUMINAMATH_CALUDE_basil_plant_selling_price_l3647_364743

/-- Proves that the selling price per basil plant is $5.00 given the costs and net profit --/
theorem basil_plant_selling_price 
  (seed_cost : ℝ) 
  (soil_cost : ℝ) 
  (num_plants : ℕ) 
  (net_profit : ℝ) 
  (h1 : seed_cost = 2)
  (h2 : soil_cost = 8)
  (h3 : num_plants = 20)
  (h4 : net_profit = 90) :
  (net_profit + seed_cost + soil_cost) / num_plants = 5 := by
  sorry

#check basil_plant_selling_price

end NUMINAMATH_CALUDE_basil_plant_selling_price_l3647_364743


namespace NUMINAMATH_CALUDE_cans_in_cat_package_l3647_364747

/-- Represents the number of cans in each package of cat food -/
def cans_per_cat_package : ℕ := sorry

/-- The number of packages of cat food Adam bought -/
def cat_packages : ℕ := 9

/-- The number of packages of dog food Adam bought -/
def dog_packages : ℕ := 7

/-- The number of cans in each package of dog food -/
def cans_per_dog_package : ℕ := 5

/-- The difference between the total number of cat food cans and dog food cans -/
def can_difference : ℕ := 55

theorem cans_in_cat_package : 
  cans_per_cat_package * cat_packages = 
  cans_per_dog_package * dog_packages + can_difference ∧ 
  cans_per_cat_package = 10 := by sorry

end NUMINAMATH_CALUDE_cans_in_cat_package_l3647_364747


namespace NUMINAMATH_CALUDE_net_cash_change_l3647_364791

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  ownsHouse : Bool

/-- Represents a house transaction -/
inductive Transaction
  | Rent : Transaction
  | BuyHouse : Int → Transaction
  | SellHouse : Int → Transaction

def initialValueA : Int := 15000
def initialValueB : Int := 20000
def initialHouseValue : Int := 15000
def rentAmount : Int := 2000

def applyTransaction (state : FinancialState) (transaction : Transaction) : FinancialState :=
  match transaction with
  | Transaction.Rent => 
      if state.ownsHouse then 
        { cash := state.cash + rentAmount, ownsHouse := state.ownsHouse }
      else 
        { cash := state.cash - rentAmount, ownsHouse := state.ownsHouse }
  | Transaction.BuyHouse price => 
      { cash := state.cash - price, ownsHouse := true }
  | Transaction.SellHouse price => 
      { cash := state.cash + price, ownsHouse := false }

def transactions : List Transaction := [
  Transaction.Rent,
  Transaction.SellHouse 18000,
  Transaction.BuyHouse 17000
]

theorem net_cash_change 
  (initialA : FinancialState) 
  (initialB : FinancialState) 
  (finalA : FinancialState) 
  (finalB : FinancialState) :
  initialA = { cash := initialValueA, ownsHouse := true } →
  initialB = { cash := initialValueB, ownsHouse := false } →
  finalA = transactions.foldl applyTransaction initialA →
  finalB = transactions.foldl applyTransaction initialB →
  finalA.cash - initialA.cash = 3000 ∧ 
  finalB.cash - initialB.cash = -3000 :=
sorry

end NUMINAMATH_CALUDE_net_cash_change_l3647_364791


namespace NUMINAMATH_CALUDE_time_to_reach_room_l3647_364709

theorem time_to_reach_room (total_time gate_time building_time : ℕ) 
  (h1 : total_time = 30)
  (h2 : gate_time = 15)
  (h3 : building_time = 6) :
  total_time - (gate_time + building_time) = 9 := by
  sorry

end NUMINAMATH_CALUDE_time_to_reach_room_l3647_364709


namespace NUMINAMATH_CALUDE_equation_solution_l3647_364736

theorem equation_solution (n : ℝ) (h : n = 3) :
  n^4 - 20*n + 1 = 22 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3647_364736


namespace NUMINAMATH_CALUDE_six_eight_x_ten_y_l3647_364701

theorem six_eight_x_ten_y (x y Q : ℝ) (h : 3 * (4 * x + 5 * y) = Q) : 
  6 * (8 * x + 10 * y) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_six_eight_x_ten_y_l3647_364701


namespace NUMINAMATH_CALUDE_smallest_x_value_l3647_364725

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (4 : ℚ) / 5 = y / (205 + x)) : 
  5 ≤ x ∧ ∃ (y' : ℕ+), (4 : ℚ) / 5 = y' / (205 + 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3647_364725


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l3647_364722

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2142 → n + (n + 1) = -93 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l3647_364722


namespace NUMINAMATH_CALUDE_arithmetic_progression_square_l3647_364762

/-- An arithmetic progression containing two natural numbers and the square of the smaller one also contains the square of the larger one. -/
theorem arithmetic_progression_square (a b : ℕ) (d : ℚ) (n m : ℤ) :
  a < b →
  b = a + n * d →
  a^2 = a + m * d →
  ∃ k : ℤ, b^2 = a + k * d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_square_l3647_364762


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3647_364778

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n * 15 < 500 → n * 15 ≤ 495 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3647_364778


namespace NUMINAMATH_CALUDE_left_handed_rock_lovers_l3647_364773

theorem left_handed_rock_lovers (total : ℕ) (left_handed : ℕ) (rock_lovers : ℕ) (right_handed_non_rock : ℕ) :
  total = 25 →
  left_handed = 10 →
  rock_lovers = 18 →
  right_handed_non_rock = 3 →
  left_handed + (total - left_handed) = total →
  ∃ (left_handed_rock : ℕ),
    left_handed_rock + (left_handed - left_handed_rock) + (rock_lovers - left_handed_rock) + right_handed_non_rock = total ∧
    left_handed_rock = 6 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_rock_lovers_l3647_364773


namespace NUMINAMATH_CALUDE_remainder_theorem_l3647_364796

-- Define the polynomial p(x) = x^4 - 2x^3 + x + 5
def p (x : ℝ) : ℝ := x^4 - 2*x^3 + x + 5

-- Theorem: The remainder when p(x) is divided by (x - 2) is 7
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = (x - 2) * q x + 7 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3647_364796


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l3647_364714

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two convex polygons in a plane -/
structure PolygonConfiguration where
  P1 : ConvexPolygon
  P2 : ConvexPolygon
  no_common_segment : Bool
  h : P1.sides ≤ P2.sides

/-- The maximum number of intersections between two convex polygons -/
def max_intersections (config : PolygonConfiguration) : ℕ :=
  config.P1.sides * config.P2.sides

/-- Theorem stating the maximum number of intersections between two convex polygons -/
theorem max_intersections_theorem (config : PolygonConfiguration) :
  config.P1.convex ∧ config.P2.convex ∧ config.no_common_segment →
  max_intersections config = config.P1.sides * config.P2.sides :=
by sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l3647_364714
