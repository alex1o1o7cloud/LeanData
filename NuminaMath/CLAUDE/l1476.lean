import Mathlib

namespace candy_bar_problem_l1476_147634

/-- Given the candy bar distribution problem, prove that 40% of Jacqueline's candy bars is 120 -/
theorem candy_bar_problem :
  let fred_candy : ℕ := 12
  let bob_candy : ℕ := fred_candy + 6
  let total_candy : ℕ := fred_candy + bob_candy
  let jacqueline_candy : ℕ := 10 * total_candy
  (40 : ℚ) / 100 * jacqueline_candy = 120 := by sorry

end candy_bar_problem_l1476_147634


namespace cube_root_difference_l1476_147672

theorem cube_root_difference (a b : ℝ) 
  (h1 : (a ^ (1/3) : ℝ) - (b ^ (1/3) : ℝ) = 12)
  (h2 : a * b = ((a + b + 8) / 6) ^ 3) :
  a - b = 468 := by
  sorry

end cube_root_difference_l1476_147672


namespace arithmetic_sequence_a12_l1476_147694

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 4 + a 5 = 3) 
  (h_a8 : a 8 = 8) : 
  a 12 = 15 := by
sorry

end arithmetic_sequence_a12_l1476_147694


namespace some_number_solution_l1476_147619

theorem some_number_solution : 
  ∃ x : ℚ, (1 / 2 : ℚ) + ((2 / 3 : ℚ) * (3 / 8 : ℚ) + 4) - x = (17 / 4 : ℚ) ∧ x = (1 / 2 : ℚ) := by
  sorry

end some_number_solution_l1476_147619


namespace angle_calculation_l1476_147644

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- Two angles are supplementary if their sum is 180 degrees -/
def supplementary (a b : ℝ) : Prop := a + b = 180

/-- Given that angle1 and angle2 are complementary, angle2 and angle3 are supplementary, 
    and angle1 is 20 degrees, prove that angle3 is 110 degrees -/
theorem angle_calculation (angle1 angle2 angle3 : ℝ) 
    (h1 : complementary angle1 angle2)
    (h2 : supplementary angle2 angle3)
    (h3 : angle1 = 20) : 
  angle3 = 110 := by sorry

end angle_calculation_l1476_147644


namespace hexagon_perimeter_is_24_l1476_147696

/-- Represents the side length of the larger equilateral triangle -/
def large_triangle_side : ℝ := 4

/-- Represents the side length of the regular hexagon -/
def hexagon_side : ℝ := large_triangle_side

/-- The number of sides in a regular hexagon -/
def hexagon_sides : ℕ := 6

/-- Calculates the perimeter of the regular hexagon -/
def hexagon_perimeter : ℝ := hexagon_side * hexagon_sides

theorem hexagon_perimeter_is_24 : hexagon_perimeter = 24 := by
  sorry

end hexagon_perimeter_is_24_l1476_147696


namespace baron_munchausen_crowd_size_l1476_147612

theorem baron_munchausen_crowd_size :
  ∃ (n : ℕ), n > 0 ∧
  (n / 2 + n / 3 + n / 5 ≤ n + 1) ∧
  (∀ m : ℕ, m > n → m / 2 + m / 3 + m / 5 > m + 1) ∧
  n = 37 := by
  sorry

end baron_munchausen_crowd_size_l1476_147612


namespace min_value_of_f_l1476_147678

/-- The function f(x) = x^2 + 8x + 25 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 25

/-- The minimum value of f(x) is 9 -/
theorem min_value_of_f : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 9 := by
  sorry

end min_value_of_f_l1476_147678


namespace complement_of_union_l1476_147671

def I : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {1, 2, 4, 5}
def N : Set ℕ := {0, 3, 5, 7}

theorem complement_of_union : (I \ (M ∪ N)) = {6, 8} := by sorry

end complement_of_union_l1476_147671


namespace factorial_sum_equals_4926_l1476_147697

theorem factorial_sum_equals_4926 : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 3 = 4926 := by
  sorry

end factorial_sum_equals_4926_l1476_147697


namespace special_line_equation_l1476_147664

/-- A line passing through (5, 2) with y-intercept twice the x-intercept -/
structure SpecialLine where
  -- Slope-intercept form: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (5, 2)
  point_condition : 2 = m * 5 + b
  -- The y-intercept is twice the x-intercept
  intercept_condition : b = -2 * (b / m)

/-- The equation of the special line is either 2x + y - 12 = 0 or 2x - 5y = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (∀ x y, 2 * x + y - 12 = 0 ↔ y = l.m * x + l.b) ∨
  (∀ x y, 2 * x - 5 * y = 0 ↔ y = l.m * x + l.b) :=
sorry

end special_line_equation_l1476_147664


namespace num_planes_is_one_or_three_l1476_147600

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  distinct : point1 ≠ point2

/-- Three pairwise parallel lines -/
structure ThreeParallelLines where
  line1 : Line3D
  line2 : Line3D
  line3 : Line3D
  parallel12 : line1.point2 - line1.point1 = line2.point2 - line2.point1
  parallel23 : line2.point2 - line2.point1 = line3.point2 - line3.point1
  parallel31 : line3.point2 - line3.point1 = line1.point2 - line1.point1

/-- The number of planes determined by three pairwise parallel lines -/
def num_planes_from_parallel_lines (lines : ThreeParallelLines) : Fin 4 :=
  sorry

/-- Theorem: The number of planes determined by three pairwise parallel lines is either 1 or 3 -/
theorem num_planes_is_one_or_three (lines : ThreeParallelLines) :
  (num_planes_from_parallel_lines lines = 1) ∨ (num_planes_from_parallel_lines lines = 3) :=
sorry

end num_planes_is_one_or_three_l1476_147600


namespace slope_of_line_l1476_147680

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 :=
by sorry

end slope_of_line_l1476_147680


namespace fourth_power_product_l1476_147674

theorem fourth_power_product : 
  (((2^4 - 1) / (2^4 + 1)) * ((3^4 - 1) / (3^4 + 1)) * 
   ((4^4 - 1) / (4^4 + 1)) * ((5^4 - 1) / (5^4 + 1))) = 432 / 1105 := by
  sorry

end fourth_power_product_l1476_147674


namespace line_slope_perpendicular_lines_b_value_l1476_147607

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (x₁ y₁ x₂ y₂ : ℝ), y₁ = m₁ * x₁ ∧ y₂ = m₂ * x₂ ∧ (y₂ - y₁) * (x₂ - x₁) = 0)

/-- The slope of a line ax + by + c = 0 where b ≠ 0 is -a/b -/
theorem line_slope (a b c : ℝ) (hb : b ≠ 0) :
  ∃ m : ℝ, m = -a / b ∧ ∀ x y : ℝ, a * x + b * y + c = 0 → y = m * x - c / b :=
sorry

theorem perpendicular_lines_b_value : 
  ∀ b : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ, 
    2 * x₁ + 3 * y₁ - 4 = 0 ∧ 
    b * x₂ + 3 * y₂ - 4 = 0 ∧ 
    (y₂ - y₁) * (x₂ - x₁) = 0) → 
  b = -9/2 :=
sorry

end line_slope_perpendicular_lines_b_value_l1476_147607


namespace max_xy_value_l1476_147628

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 6) :
  x*y ≤ 3/2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + 3*y₀ = 6 ∧ x₀*y₀ = 3/2 :=
sorry

end max_xy_value_l1476_147628


namespace isosceles_triangle_base_length_l1476_147681

/-- An isosceles triangle with congruent sides of 5 cm and perimeter of 17 cm has a base of 7 cm. -/
theorem isosceles_triangle_base_length : ∀ (base : ℝ),
  base > 0 →
  5 + 5 + base = 17 →
  base = 7 :=
by
  sorry

end isosceles_triangle_base_length_l1476_147681


namespace expanded_polynomial_has_four_nonzero_terms_l1476_147655

/-- The polynomial obtained from expanding (x+5)(3x^2+2x+4)-4(x^3-x^2+3x) -/
def expanded_polynomial (x : ℝ) : ℝ := -x^3 + 21*x^2 + 2*x + 20

/-- The number of nonzero terms in the expanded polynomial -/
def nonzero_term_count : ℕ := 4

theorem expanded_polynomial_has_four_nonzero_terms :
  (∃ a b c d : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    ∀ x : ℝ, expanded_polynomial x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ a b c d e : ℝ, ∀ i j k l m : ℕ,
    (i ≠ j ∨ a = 0) ∧ (i ≠ k ∨ a = 0) ∧ (i ≠ l ∨ a = 0) ∧ (i ≠ m ∨ a = 0) ∧
    (j ≠ k ∨ b = 0) ∧ (j ≠ l ∨ b = 0) ∧ (j ≠ m ∨ b = 0) ∧
    (k ≠ l ∨ c = 0) ∧ (k ≠ m ∨ c = 0) ∧
    (l ≠ m ∨ d = 0) →
    (∀ x : ℝ, expanded_polynomial x = a*x^i + b*x^j + c*x^k + d*x^l + e*x^m) →
    e = 0) :=
by sorry

#check expanded_polynomial_has_four_nonzero_terms

end expanded_polynomial_has_four_nonzero_terms_l1476_147655


namespace even_function_implies_m_zero_l1476_147673

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Theorem statement
theorem even_function_implies_m_zero (m : ℝ) :
  is_even (f m) → m = 0 := by sorry

end even_function_implies_m_zero_l1476_147673


namespace number_equation_solution_l1476_147613

theorem number_equation_solution : ∃ x : ℝ, (3 * x - 1 = 2 * x) ∧ (x = 1) := by
  sorry

end number_equation_solution_l1476_147613


namespace unit_digit_of_sum_factorials_100_l1476_147683

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem unit_digit_of_sum_factorials_100 :
  sum_factorials 100 % 10 = 3 := by
  sorry

end unit_digit_of_sum_factorials_100_l1476_147683


namespace sum_of_coefficients_l1476_147686

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₁ + a₂ + a₃ = -15 := by
sorry

end sum_of_coefficients_l1476_147686


namespace impossible_friendship_configuration_l1476_147656

/-- A graph representing friendships in a class -/
structure FriendshipGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  sym : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  irrefl : ∀ a, (a, a) ∉ edges

/-- The degree of a vertex in the graph -/
def degree (G : FriendshipGraph) (v : Nat) : Nat :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- Theorem: It's impossible to have a friendship graph with 30 students where
    9 have 3 friends, 11 have 4 friends, and 10 have 5 friends -/
theorem impossible_friendship_configuration (G : FriendshipGraph) :
  G.vertices.card = 30 →
  (∃ S₁ S₂ S₃ : Finset Nat,
    S₁.card = 9 ∧ S₂.card = 11 ∧ S₃.card = 10 ∧
    S₁ ∪ S₂ ∪ S₃ = G.vertices ∧
    (∀ v ∈ S₁, degree G v = 3) ∧
    (∀ v ∈ S₂, degree G v = 4) ∧
    (∀ v ∈ S₃, degree G v = 5)) →
  False := by
  sorry

end impossible_friendship_configuration_l1476_147656


namespace sasha_remainder_l1476_147658

theorem sasha_remainder (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  b < 102 ∧ 
  d < 103 ∧ 
  a + d = 20 → 
  b = 20 := by
sorry

end sasha_remainder_l1476_147658


namespace decreasing_power_function_m_values_l1476_147661

theorem decreasing_power_function_m_values (m : ℤ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → x₁^(m^2 - m - 2) > x₂^(m^2 - m - 2)) →
  m = 0 ∨ m = 1 := by
  sorry

end decreasing_power_function_m_values_l1476_147661


namespace boat_license_combinations_l1476_147626

/-- The number of possible letters for a boat license -/
def letter_choices : ℕ := 4

/-- The number of possible choices for the first digit of a boat license -/
def first_digit_choices : ℕ := 8

/-- The number of possible choices for each of the remaining digits of a boat license -/
def other_digit_choices : ℕ := 10

/-- The number of digits in a boat license after the letter -/
def num_digits : ℕ := 7

/-- Theorem: The number of possible boat license combinations is 32,000,000 -/
theorem boat_license_combinations :
  letter_choices * first_digit_choices * (other_digit_choices ^ (num_digits - 1)) = 32000000 := by
  sorry

end boat_license_combinations_l1476_147626


namespace locus_of_midpoint_of_tangent_l1476_147608

/-- Given two circles with centers at (0, 0) and (a, 0), prove that the locus of the midpoint
    of their common outer tangent is part of a specific circle. -/
theorem locus_of_midpoint_of_tangent (a c : ℝ) (h₁ : a > c) (h₂ : c > 0) :
  ∃ (x y : ℝ), 
    4 * x^2 + 4 * y^2 - 4 * a * x + a^2 = c^2 ∧ 
    (a^2 - c^2) / (2 * a) ≤ x ∧ 
    x ≤ (a^2 + c^2) / (2 * a) ∧ 
    y > 0 :=
sorry

end locus_of_midpoint_of_tangent_l1476_147608


namespace fraction_simplification_l1476_147611

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 4 / 6) (hy : y = 8 / 10) : 
  (6 * x + 8 * y) / (48 * x * y) = 13 / 32 := by
sorry

end fraction_simplification_l1476_147611


namespace polynomial_coefficient_sum_l1476_147614

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℝ, 
  (∀ x : ℝ, (x + 2) * (3 * x^2 - x + 5) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 21 := by
sorry

end polynomial_coefficient_sum_l1476_147614


namespace sin_cos_identity_l1476_147629

theorem sin_cos_identity (α : ℝ) : 
  Real.sin α ^ 6 + Real.cos α ^ 6 + 3 * (Real.sin α ^ 2) * (Real.cos α ^ 2) = 1 := by
  sorry

end sin_cos_identity_l1476_147629


namespace train_platform_lengths_l1476_147617

/-- Two trains with different constant velocities passing platforms -/
theorem train_platform_lengths 
  (V1 V2 L1 L2 T1 T2 : ℝ) 
  (h_diff_vel : V1 ≠ V2) 
  (h_pos_V1 : V1 > 0) 
  (h_pos_V2 : V2 > 0)
  (h_pos_T1 : T1 > 0)
  (h_pos_T2 : T2 > 0)
  (h_L1 : L1 = V1 * T1)
  (h_L2 : L2 = V2 * T2) :
  ∃ (P1 P2 : ℝ), 
    P1 = 3 * V1 * T1 ∧ 
    P2 = 2 * V2 * T2 ∧
    V1 * (4 * T1) = L1 + P1 ∧
    V2 * (3 * T2) = L2 + P2 := by
  sorry


end train_platform_lengths_l1476_147617


namespace four_students_same_group_probability_l1476_147603

/-- The number of groups -/
def num_groups : ℕ := 4

/-- The probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- The probability of four specific students being assigned to the same group -/
def prob_four_students_same_group : ℚ := prob_assigned_to_group ^ 3

theorem four_students_same_group_probability :
  prob_four_students_same_group = 1 / 64 := by sorry

end four_students_same_group_probability_l1476_147603


namespace divisor_of_infinite_set_l1476_147699

theorem divisor_of_infinite_set (A : Set ℕ+) 
  (h_infinite : Set.Infinite A)
  (h_finite_subset : ∀ B : Set ℕ+, B ⊆ A → Set.Finite B → 
    ∃ b : ℕ+, b > 1 ∧ ∀ x ∈ B, b ∣ x) :
  ∃ d : ℕ+, d > 1 ∧ ∀ x ∈ A, d ∣ x :=
sorry

end divisor_of_infinite_set_l1476_147699


namespace gcd_of_75_and_100_l1476_147636

theorem gcd_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end gcd_of_75_and_100_l1476_147636


namespace sum_of_squares_power_l1476_147604

theorem sum_of_squares_power (a p q : ℤ) (h : a = p^2 + q^2) :
  ∀ k : ℕ+, ∃ x y : ℤ, a^(k : ℕ) = x^2 + y^2 :=
by sorry

end sum_of_squares_power_l1476_147604


namespace Diamond_evaluation_l1476_147667

-- Define the Diamond operation
def Diamond (a b : ℝ) : ℝ := a * b^2 - b + 1

-- Theorem statement
theorem Diamond_evaluation : Diamond (Diamond 2 3) 4 = 253 := by
  sorry

end Diamond_evaluation_l1476_147667


namespace divisors_of_fermat_like_number_l1476_147620

-- Define a function to represent the product of the first n primes in a list
def primeProduct : List Nat → Nat
  | [] => 1
  | p::ps => p * primeProduct ps

-- Define the main theorem
theorem divisors_of_fermat_like_number (n : Nat) (primes : List Nat) 
  (h_distinct : List.Pairwise (·≠·) primes)
  (h_prime : ∀ p ∈ primes, Nat.Prime p)
  (h_greater_than_three : ∀ p ∈ primes, p > 3)
  (h_length : primes.length = n) :
  (Nat.divisors (2^(primeProduct primes) + 1)).card ≥ 4^n := by
  sorry

end divisors_of_fermat_like_number_l1476_147620


namespace equation_solution_and_condition_l1476_147659

theorem equation_solution_and_condition :
  ∃ x : ℝ,
    (8 * x^(1/3) - 4 * (x / x^(2/3)) = 12 + 2 * x^(1/3)) ∧
    (x ≥ Real.sqrt 144) ∧
    (x = 216) := by
  sorry

end equation_solution_and_condition_l1476_147659


namespace first_super_monday_l1476_147665

/-- Represents a date with a month and day -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of days in a given month -/
def daysInMonth (month : Nat) : Nat :=
  match month with
  | 3 => 31  -- March
  | 4 => 30  -- April
  | 5 => 31  -- May
  | _ => 30  -- Default (not used in this problem)

/-- Checks if a given date is a Monday -/
def isMonday (date : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Counts the number of Mondays in a given month -/
def countMondaysInMonth (month : Nat) (startDate : Date) (startDay : DayOfWeek) : Nat :=
  sorry

/-- Finds the date of the fifth Monday in a given month -/
def fifthMondayInMonth (month : Nat) (startDate : Date) (startDay : DayOfWeek) : Option Date :=
  sorry

/-- Theorem: The first Super Monday after school starts on Tuesday, March 1 is May 30 -/
theorem first_super_monday :
  let schoolStart : Date := ⟨3, 1⟩
  let firstSuperMonday := fifthMondayInMonth 5 schoolStart DayOfWeek.Tuesday
  firstSuperMonday = some ⟨5, 30⟩ :=
by
  sorry

#check first_super_monday

end first_super_monday_l1476_147665


namespace min_value_shifted_function_l1476_147676

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 4*x + 5 - c

theorem min_value_shifted_function 
  (c : ℝ) 
  (h : ∃ (m : ℝ), ∀ (x : ℝ), f x c ≥ m ∧ ∃ (x₀ : ℝ), f x₀ c = m) 
  (h_min : ∃ (x₀ : ℝ), f x₀ c = 2) :
  ∃ (m : ℝ), (∀ (x : ℝ), f (x - 3) c ≥ m) ∧ (∃ (x₁ : ℝ), f (x₁ - 3) c = m) ∧ m = 2 := by
sorry

end min_value_shifted_function_l1476_147676


namespace solve_equation_l1476_147657

theorem solve_equation : ∃ y : ℝ, (y - 3)^4 = (1/16)⁻¹ ∧ y = 5 := by sorry

end solve_equation_l1476_147657


namespace park_fencing_cost_l1476_147693

/-- The cost of fencing a rectangular park -/
theorem park_fencing_cost 
  (length width : ℝ) 
  (area : ℝ) 
  (fencing_cost_paise : ℝ) : 
  length / width = 3 / 2 →
  area = length * width →
  area = 2400 →
  fencing_cost_paise = 50 →
  2 * (length + width) * (fencing_cost_paise / 100) = 100 := by
  sorry


end park_fencing_cost_l1476_147693


namespace roots_of_cosine_equation_l1476_147618

theorem roots_of_cosine_equation :
  let f (t : ℝ) := 32 * t^5 - 40 * t^3 + 10 * t - Real.sqrt 3
  (f (Real.cos (6 * π / 180)) = 0) →
  (f (Real.cos (66 * π / 180)) = 0) ∧
  (f (Real.cos (78 * π / 180)) = 0) ∧
  (f (Real.cos (138 * π / 180)) = 0) ∧
  (f (Real.cos (150 * π / 180)) = 0) :=
by sorry

end roots_of_cosine_equation_l1476_147618


namespace same_odd_dice_probability_l1476_147695

/-- The number of faces on each die -/
def num_faces : ℕ := 8

/-- The number of odd faces on each die -/
def num_odd_faces : ℕ := 4

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability of rolling the same odd number on all dice -/
def prob_same_odd : ℚ := 1 / 1024

theorem same_odd_dice_probability :
  (num_odd_faces : ℚ) / num_faces * (1 / num_faces) ^ (num_dice - 1) = prob_same_odd :=
by sorry

end same_odd_dice_probability_l1476_147695


namespace quadratic_inequality_condition_l1476_147606

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 2)*x - k + 4 > 0) ↔ 
  k > -2 * Real.sqrt 3 ∧ k < 2 * Real.sqrt 3 :=
sorry

end quadratic_inequality_condition_l1476_147606


namespace spadesuit_inequality_not_always_true_l1476_147663

def spadesuit (x y : ℝ) : ℝ := x^2 - y^2

theorem spadesuit_inequality_not_always_true :
  ¬ (∀ x y : ℝ, x ≥ y → spadesuit x y ≥ 0) :=
sorry

end spadesuit_inequality_not_always_true_l1476_147663


namespace train_passes_jogger_l1476_147630

/-- Prove that a train passes a jogger in 35 seconds given the specified conditions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 230 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 35 := by
  sorry

end train_passes_jogger_l1476_147630


namespace firewood_sacks_l1476_147690

theorem firewood_sacks (total_wood : ℕ) (wood_per_sack : ℕ) (h1 : total_wood = 80) (h2 : wood_per_sack = 20) :
  total_wood / wood_per_sack = 4 :=
by sorry

end firewood_sacks_l1476_147690


namespace expansion_coefficient_l1476_147660

/-- The coefficient of x^5 in the expansion of (ax^2 + 1/√x)^5 -/
def coefficient_x5 (a : ℝ) : ℝ := 10 * a^3

theorem expansion_coefficient (a : ℝ) : 
  coefficient_x5 a = -80 → a = -2 := by
  sorry

end expansion_coefficient_l1476_147660


namespace cos_alpha_plus_17pi_12_l1476_147675

theorem cos_alpha_plus_17pi_12 (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 17 * π / 12) = 1 / 3 := by
  sorry

end cos_alpha_plus_17pi_12_l1476_147675


namespace determinant_value_l1476_147639

-- Define the operation
def determinant (a b c d : ℚ) : ℚ := a * d - b * c

-- Define the theorem
theorem determinant_value :
  let a : ℚ := -(1^2)
  let b : ℚ := (-2)^2 - 1
  let c : ℚ := -(3^2) + 5
  let d : ℚ := (3/4) / (-1/4)
  determinant a b c d = 15 := by
  sorry

end determinant_value_l1476_147639


namespace lollipops_left_over_l1476_147635

/-- The number of lollipops Winnie has left after distributing them evenly among her friends -/
theorem lollipops_left_over (total : ℕ) (friends : ℕ) (h1 : total = 505) (h2 : friends = 14) :
  total % friends = 1 := by
  sorry

end lollipops_left_over_l1476_147635


namespace derivative_f_at_3_l1476_147685

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem derivative_f_at_3 : 
  deriv f 3 = 1 / (3 * Real.log 3) := by sorry

end derivative_f_at_3_l1476_147685


namespace min_value_implies_m_range_l1476_147610

-- Define the piecewise function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then (x - m)^2 - 2 else 2*x^3 - 3*x^2

-- State the theorem
theorem min_value_implies_m_range (m : ℝ) :
  (∀ x, f m x ≥ -1) ∧ (∃ x, f m x = -1) → m ≥ 1 :=
by sorry

end min_value_implies_m_range_l1476_147610


namespace line_not_in_fourth_quadrant_l1476_147677

theorem line_not_in_fourth_quadrant 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b < 0) 
  (hc : c > 0) : 
  ∀ x y : ℝ, a * x + b * y + c = 0 → ¬(x > 0 ∧ y < 0) := by
sorry

end line_not_in_fourth_quadrant_l1476_147677


namespace olivia_change_olivia_change_proof_l1476_147670

/-- Calculates the change Olivia received after buying basketball and baseball cards -/
theorem olivia_change (basketball_packs : ℕ) (basketball_price : ℕ) 
  (baseball_decks : ℕ) (baseball_price : ℕ) (bill : ℕ) : ℕ :=
  let total_cost := basketball_packs * basketball_price + baseball_decks * baseball_price
  bill - total_cost

/-- Proves that Olivia received $24 in change -/
theorem olivia_change_proof :
  olivia_change 2 3 5 4 50 = 24 := by
  sorry

end olivia_change_olivia_change_proof_l1476_147670


namespace consecutive_integers_sum_l1476_147631

theorem consecutive_integers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2800 → n + (n + 1) = 105 := by
  sorry

end consecutive_integers_sum_l1476_147631


namespace power_inequality_l1476_147622

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  a^a < b^a := by sorry

end power_inequality_l1476_147622


namespace discriminant_of_5x2_minus_9x_plus_4_l1476_147616

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Proof that the discriminant of 5x^2 - 9x + 4 is 1 -/
theorem discriminant_of_5x2_minus_9x_plus_4 :
  discriminant 5 (-9) 4 = 1 := by
  sorry

end discriminant_of_5x2_minus_9x_plus_4_l1476_147616


namespace cattle_selling_price_l1476_147638

/-- Proves that the selling price per pound for cattle is $1.60 given the specified conditions --/
theorem cattle_selling_price
  (num_cattle : ℕ)
  (purchase_price : ℝ)
  (feed_cost_percentage : ℝ)
  (weight_per_cattle : ℝ)
  (profit : ℝ)
  (h1 : num_cattle = 100)
  (h2 : purchase_price = 40000)
  (h3 : feed_cost_percentage = 0.20)
  (h4 : weight_per_cattle = 1000)
  (h5 : profit = 112000)
  : ∃ (selling_price_per_pound : ℝ),
    selling_price_per_pound = 1.60 ∧
    selling_price_per_pound * (num_cattle * weight_per_cattle) =
      purchase_price + (feed_cost_percentage * purchase_price) + profit :=
by
  sorry

end cattle_selling_price_l1476_147638


namespace hamburger_cost_theorem_l1476_147646

def total_hamburgers : ℕ := 50
def single_burger_cost : ℚ := 1
def double_burger_cost : ℚ := 1.5
def double_burgers : ℕ := 41

theorem hamburger_cost_theorem :
  let single_burgers := total_hamburgers - double_burgers
  let total_cost := (single_burgers : ℚ) * single_burger_cost + (double_burgers : ℚ) * double_burger_cost
  total_cost = 70.5 := by sorry

end hamburger_cost_theorem_l1476_147646


namespace least_positive_integer_congruence_l1476_147625

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 5123) % 12 = 2900 % 12 ∧
  ∀ (y : ℕ), y > 0 → (y + 5123) % 12 = 2900 % 12 → x ≤ y :=
by sorry

end least_positive_integer_congruence_l1476_147625


namespace tangent_product_equality_l1476_147641

theorem tangent_product_equality : 
  (1 + Real.tan (20 * π / 180)) * (1 + Real.tan (25 * π / 180)) = 2 :=
by
  sorry

/- Proof hints:
   1. Use the fact that 45° = 20° + 25°
   2. Recall that tan(45°) = 1
   3. Apply the tangent sum formula: tan(A+B) = (tan A + tan B) / (1 - tan A * tan B)
   4. Algebraically manipulate the expressions
-/

end tangent_product_equality_l1476_147641


namespace phone_number_a_is_five_l1476_147624

/-- Represents a valid 10-digit telephone number -/
structure PhoneNumber where
  digits : Fin 10 → Fin 10
  all_different : ∀ i j, i ≠ j → digits i ≠ digits j
  decreasing_abc : digits 0 > digits 1 ∧ digits 1 > digits 2
  decreasing_def : digits 3 > digits 4 ∧ digits 4 > digits 5
  decreasing_ghij : digits 6 > digits 7 ∧ digits 7 > digits 8 ∧ digits 8 > digits 9
  consecutive_def : ∃ n : ℕ, digits 3 = n + 2 ∧ digits 4 = n + 1 ∧ digits 5 = n
  consecutive_ghij : ∃ n : ℕ, digits 6 = n + 3 ∧ digits 7 = n + 2 ∧ digits 8 = n + 1 ∧ digits 9 = n
  sum_abc : digits 0 + digits 1 + digits 2 = 10

theorem phone_number_a_is_five (p : PhoneNumber) : p.digits 0 = 5 := by
  sorry

end phone_number_a_is_five_l1476_147624


namespace quadratic_roots_close_existence_l1476_147669

theorem quadratic_roots_close_existence :
  ∃ (a b c : ℕ), a ≤ 2019 ∧ b ≤ 2019 ∧ c ≤ 2019 ∧
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (a * x₁^2 + b * x₁ + c = 0) ∧
  (a * x₂^2 + b * x₂ + c = 0) ∧
  |x₁ - x₂| < 0.01 :=
by sorry

end quadratic_roots_close_existence_l1476_147669


namespace journey_solution_l1476_147637

/-- Represents the problem of Xiaogang and Xiaoqiang's journey --/
structure JourneyProblem where
  total_distance : ℝ
  meeting_time : ℝ
  xiaogang_extra_distance : ℝ
  xiaogang_remaining_time : ℝ

/-- Represents the solution to the journey problem --/
structure JourneySolution where
  xiaogang_speed : ℝ
  xiaoqiang_speed : ℝ
  xiaoqiang_remaining_time : ℝ

/-- Theorem stating the correct solution for the given problem --/
theorem journey_solution (p : JourneyProblem) 
  (h1 : p.meeting_time = 2)
  (h2 : p.xiaogang_extra_distance = 24)
  (h3 : p.xiaogang_remaining_time = 0.5) :
  ∃ (s : JourneySolution),
    s.xiaogang_speed = 16 ∧
    s.xiaoqiang_speed = 4 ∧
    s.xiaoqiang_remaining_time = 8 ∧
    p.total_distance = s.xiaogang_speed * (p.meeting_time + p.xiaogang_remaining_time) ∧
    p.total_distance = (s.xiaogang_speed * p.meeting_time - p.xiaogang_extra_distance) + (s.xiaoqiang_speed * s.xiaoqiang_remaining_time) :=
by
  sorry


end journey_solution_l1476_147637


namespace first_hundred_contains_all_naturals_l1476_147615

/-- A sequence of 200 numbers partitioned into blue and red -/
def Sequence := Fin 200 → ℕ

/-- The property that blue numbers are in ascending order from 1 to 100 -/
def BlueAscending (s : Sequence) : Prop :=
  ∃ (blue : Fin 200 → Bool),
    (∀ i : Fin 200, blue i → s i ∈ Finset.range 101) ∧
    (∀ i j : Fin 200, i < j → blue i → blue j → s i < s j)

/-- The property that red numbers are in descending order from 100 to 1 -/
def RedDescending (s : Sequence) : Prop :=
  ∃ (red : Fin 200 → Bool),
    (∀ i : Fin 200, red i → s i ∈ Finset.range 101) ∧
    (∀ i j : Fin 200, i < j → red i → red j → s i > s j)

/-- The main theorem -/
theorem first_hundred_contains_all_naturals (s : Sequence)
    (h1 : BlueAscending s) (h2 : RedDescending s) :
    ∀ n : ℕ, n ∈ Finset.range 101 → ∃ i : Fin 100, s i = n :=
  sorry

end first_hundred_contains_all_naturals_l1476_147615


namespace max_volume_corner_cut_box_l1476_147602

/-- The maximum volume of an open-top box formed by cutting identical squares from the corners of a rectangular cardboard -/
theorem max_volume_corner_cut_box (a b : ℝ) (ha : a = 10) (hb : b = 16) :
  let V := fun x => (a - 2*x) * (b - 2*x) * x
  ∃ (x : ℝ), x > 0 ∧ x < a/2 ∧ x < b/2 ∧
    (∀ y, y > 0 → y < a/2 → y < b/2 → V y ≤ V x) ∧
    V x = 144 :=
sorry

end max_volume_corner_cut_box_l1476_147602


namespace power_multiplication_calculate_power_l1476_147668

theorem power_multiplication (a : ℕ) (m n : ℕ) : a * (a ^ n) = a ^ (n + 1) := by sorry

theorem calculate_power : 5000 * (5000 ^ 3000) = 5000 ^ 3001 := by sorry

end power_multiplication_calculate_power_l1476_147668


namespace expression_value_l1476_147643

theorem expression_value : (0.5 : ℝ)^3 - (0.1 : ℝ)^3 / (0.5 : ℝ)^2 + 0.05 + (0.1 : ℝ)^2 = 0.181 := by
  sorry

end expression_value_l1476_147643


namespace sqrt_product_equality_l1476_147688

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l1476_147688


namespace x_power_2187_minus_reciprocal_l1476_147651

theorem x_power_2187_minus_reciprocal (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^2187 - 1/x^2187 = Complex.I * Real.sqrt 3 := by
  sorry

end x_power_2187_minus_reciprocal_l1476_147651


namespace train_ticket_types_l1476_147609

/-- The number of ticket types needed for a train route -/
def ticket_types (stops_between : ℕ) : ℕ :=
  let total_stops := stops_between + 2
  total_stops * (total_stops - 1)

/-- Theorem: For a train route with 3 stops between two end cities, 
    the number of ticket types needed is 20 -/
theorem train_ticket_types : ticket_types 3 = 20 := by
  sorry

end train_ticket_types_l1476_147609


namespace equation_transformations_correct_l1476_147679

theorem equation_transformations_correct 
  (a b c x y : ℝ) : 
  (a = b → a * c = b * c) ∧ 
  (a * (x^2 + 1) = b * (x^2 + 1) → a = b) ∧ 
  (a = b → a / c^2 = b / c^2) ∧ 
  (x = y → x - 3 = y - 3) := by
  sorry

end equation_transformations_correct_l1476_147679


namespace f_composition_value_l1476_147682

def f (x : ℚ) : ℚ := x⁻¹ + x⁻¹ / (1 + x⁻¹)

theorem f_composition_value : f (f (-3)) = 24/5 := by
  sorry

end f_composition_value_l1476_147682


namespace function_inequality_l1476_147654

open Set

theorem function_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, x > 0 → f x ≥ 0) →
  (∀ x, x > 0 → HasDerivAt f (f x) x) →
  (∀ x, x > 0 → x * (deriv f x) + f x < 0) →
  0 < a → 0 < b → a < b →
  b * f b < a * f a := by
  sorry

end function_inequality_l1476_147654


namespace composition_equation_solution_l1476_147652

theorem composition_equation_solution (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x / 3 + 2)
  (hg : ∀ x, g x = 5 - 2 * x)
  (h : f (g a) = 4) :
  a = -1/2 := by sorry

end composition_equation_solution_l1476_147652


namespace rectangle_garden_length_l1476_147691

/-- Represents the perimeter of a rectangle. -/
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Proves that a rectangular garden with perimeter 950 m and breadth 100 m has a length of 375 m. -/
theorem rectangle_garden_length :
  ∀ (length : ℝ),
  perimeter length 100 = 950 →
  length = 375 := by
sorry

end rectangle_garden_length_l1476_147691


namespace tv_tower_height_l1476_147650

/-- The height of a TV tower given specific angle measurements and distances -/
theorem tv_tower_height (angle_A : Real) (angle_B : Real) (angle_southwest : Real) (distance_AB : Real) :
  angle_A = π / 3 →  -- 60 degrees in radians
  angle_B = π / 4 →  -- 45 degrees in radians
  angle_southwest = π / 6 →  -- 30 degrees in radians
  distance_AB = 35 →
  ∃ (height : Real), height = 5 * Real.sqrt 21 := by
  sorry

end tv_tower_height_l1476_147650


namespace angle_equality_l1476_147645

theorem angle_equality (a b c t : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < t) :
  let S := Real.sqrt (a^2 + b^2 + c^2)
  let ω1 := Real.arctan ((4*t) / (a^2 + b^2 + c^2))
  let ω2 := Real.arccos ((a^2 + b^2 + c^2) / S)
  ω1 = ω2 := by sorry

end angle_equality_l1476_147645


namespace everton_college_order_l1476_147666

/-- The number of scientific calculators ordered by Everton college -/
def scientific_calculators : ℕ := 20

/-- The number of graphing calculators ordered by Everton college -/
def graphing_calculators : ℕ := 45 - scientific_calculators

/-- The cost of a single scientific calculator -/
def scientific_calculator_cost : ℕ := 10

/-- The cost of a single graphing calculator -/
def graphing_calculator_cost : ℕ := 57

/-- The total cost of the order -/
def total_cost : ℕ := 1625

/-- The total number of calculators ordered -/
def total_calculators : ℕ := 45

theorem everton_college_order :
  scientific_calculators * scientific_calculator_cost +
  graphing_calculators * graphing_calculator_cost = total_cost ∧
  scientific_calculators + graphing_calculators = total_calculators :=
sorry

end everton_college_order_l1476_147666


namespace problem_solution_l1476_147692

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -15)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 6) :
  b / (a + b) + c / (b + c) + a / (c + a) = 12 := by
sorry

end problem_solution_l1476_147692


namespace zero_function_theorem_l1476_147642

-- Define the function type
def NonNegativeFunction := { f : ℝ → ℝ // ∀ x ≥ 0, f x ≥ 0 }

-- State the theorem
theorem zero_function_theorem (f : NonNegativeFunction) 
  (h_diff : Differentiable ℝ (fun x => f.val x))
  (h_initial : f.val 0 = 0)
  (h_deriv : ∀ x ≥ 0, (deriv f.val) (x^2) = f.val x) :
  ∀ x ≥ 0, f.val x = 0 := by
sorry

end zero_function_theorem_l1476_147642


namespace car_distance_theorem_l1476_147689

-- Define the length of the line of soldiers
def line_length : ℝ := 1

-- Define the distance each soldier marches
def soldier_distance : ℝ := 15

-- Define the speed ratio between the car and soldiers
def speed_ratio : ℝ := 2

-- Theorem statement
theorem car_distance_theorem :
  let car_distance := soldier_distance * speed_ratio * line_length
  car_distance = 30 := by sorry

end car_distance_theorem_l1476_147689


namespace equation_solution_l1476_147684

theorem equation_solution : 
  ∃ x : ℚ, (x ≠ 2) ∧ (7 * x / (x - 2) + 4 / (x - 2) = 6 / (x - 2)) → x = 2 / 7 := by
  sorry

end equation_solution_l1476_147684


namespace complex_magnitude_problem_l1476_147633

theorem complex_magnitude_problem (z : ℂ) : 
  (Complex.I / (1 - Complex.I)) * z = 1 → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l1476_147633


namespace sarah_initial_cupcakes_l1476_147632

theorem sarah_initial_cupcakes :
  ∀ (initial_cupcakes : ℕ),
    (initial_cupcakes / 3 : ℚ) + 5 = 11 - (2 * initial_cupcakes / 3 : ℚ) →
    initial_cupcakes = 9 := by
  sorry

end sarah_initial_cupcakes_l1476_147632


namespace sophia_lost_pawns_l1476_147647

theorem sophia_lost_pawns (initial_pawns : ℕ) (chloe_lost : ℕ) (pawns_left : ℕ) : 
  initial_pawns = 8 → chloe_lost = 1 → pawns_left = 10 → 
  initial_pawns - (pawns_left - (initial_pawns - chloe_lost)) = 5 := by
sorry

end sophia_lost_pawns_l1476_147647


namespace least_seven_digit_binary_l1476_147662

theorem least_seven_digit_binary : ∀ n : ℕ, n > 0 → (
  (64 ≤ n ∧ (Nat.log 2 n).succ = 7) ↔ 
  (∀ m : ℕ, m > 0 ∧ m < n → (Nat.log 2 m).succ < 7)
) := by sorry

end least_seven_digit_binary_l1476_147662


namespace yi_number_is_seven_eighths_l1476_147605

def card_numbers : Finset ℚ := {1/2, 3/4, 7/8, 15/16, 31/32}

def jia_statement (x : ℚ) : Prop :=
  x ∈ card_numbers ∧ x ≠ 1/2 ∧ x ≠ 31/32

def yi_statement (y : ℚ) : Prop :=
  y ∈ card_numbers ∧ y ≠ 3/4 ∧ y ≠ 15/16

theorem yi_number_is_seven_eighths :
  ∀ (x y : ℚ), x ∈ card_numbers → y ∈ card_numbers → x ≠ y →
  jia_statement x → yi_statement y → y = 7/8 := by
  sorry

end yi_number_is_seven_eighths_l1476_147605


namespace imoCandidate1988_l1476_147621

theorem imoCandidate1988 (d r : ℤ) : 
  d > 1 ∧ 
  (∃ k m n : ℤ, 1059 = k * d + r ∧ 
               1417 = m * d + r ∧ 
               2312 = n * d + r) →
  d - r = 15 := by sorry

end imoCandidate1988_l1476_147621


namespace sum_of_roots_l1476_147698

theorem sum_of_roots (k m : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂) 
  (h2 : 4 * x₁^2 - k * x₁ = m) 
  (h3 : 4 * x₂^2 - k * x₂ = m) : 
  x₁ + x₂ = k / 4 := by
sorry

end sum_of_roots_l1476_147698


namespace base_76_minus_b_multiple_of_17_l1476_147623

/-- The value of 528376415 in base 76 -/
def base_76_number : ℕ := 5 + 1*76 + 4*(76^2) + 6*(76^3) + 7*(76^4) + 3*(76^5) + 8*(76^6) + 2*(76^7) + 5*(76^8)

theorem base_76_minus_b_multiple_of_17 (b : ℤ) 
  (h1 : 0 ≤ b) (h2 : b ≤ 20) 
  (h3 : ∃ k : ℤ, base_76_number - b = 17 * k) :
  b = 0 ∨ b = 17 := by
sorry

end base_76_minus_b_multiple_of_17_l1476_147623


namespace modified_morse_code_symbols_l1476_147649

/-- The number of distinct symbols for a given sequence length -/
def symbolCount (n : Nat) : Nat :=
  2^n

/-- The total number of distinct symbols for sequences of length 1 to 5 -/
def totalSymbols : Nat :=
  (symbolCount 1) + (symbolCount 2) + (symbolCount 3) + (symbolCount 4) + (symbolCount 5)

/-- Theorem: The total number of distinct symbols in modified Morse code for sequences
    of length 1 to 5 is 62 -/
theorem modified_morse_code_symbols :
  totalSymbols = 62 := by
  sorry

end modified_morse_code_symbols_l1476_147649


namespace closest_integer_to_cube_root_of_250_l1476_147601

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, |n^3 - 250| ≥ |6^3 - 250| :=
by sorry

end closest_integer_to_cube_root_of_250_l1476_147601


namespace probability_five_green_marbles_l1476_147653

/-- The probability of drawing exactly k successes in n trials 
    with probability p for each success -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- The number of green marbles in the bag -/
def greenMarbles : ℕ := 9

/-- The number of purple marbles in the bag -/
def purpleMarbles : ℕ := 6

/-- The total number of marbles in the bag -/
def totalMarbles : ℕ := greenMarbles + purpleMarbles

/-- The probability of drawing a green marble -/
def pGreen : ℚ := greenMarbles / totalMarbles

/-- The number of marbles drawn -/
def numDraws : ℕ := 8

/-- The number of green marbles we want to draw -/
def numGreenDraws : ℕ := 5

theorem probability_five_green_marbles :
  binomialProbability numDraws numGreenDraws pGreen = 108864 / 390625 := by
  sorry

end probability_five_green_marbles_l1476_147653


namespace salary_increase_with_manager_l1476_147640

/-- Calculates the increase in average salary when a manager's salary is added to a group of employees. -/
theorem salary_increase_with_manager 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 18 → 
  avg_salary = 2000 → 
  manager_salary = 5800 → 
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 200 :=
by sorry

end salary_increase_with_manager_l1476_147640


namespace world_expo_allocation_schemes_l1476_147648

theorem world_expo_allocation_schemes :
  let n_volunteers : ℕ := 6
  let n_pavilions : ℕ := 4
  let n_groups_of_two : ℕ := 2
  let n_groups_of_one : ℕ := 2

  -- Number of ways to choose 2 groups of 2 people from 6 volunteers
  let ways_to_choose_groups_of_two : ℕ := Nat.choose n_volunteers 2 * Nat.choose (n_volunteers - 2) 2

  -- Number of ways to allocate the groups to pavilions
  let ways_to_allocate_pavilions : ℕ := Nat.factorial n_pavilions

  -- Total number of allocation schemes
  ways_to_choose_groups_of_two * Nat.choose n_pavilions n_groups_of_two * ways_to_allocate_pavilions = 1080 :=
by sorry

end world_expo_allocation_schemes_l1476_147648


namespace ratio_equality_solution_l1476_147687

theorem ratio_equality_solution : 
  ∃! x : ℝ, (3 * x + 1) / (5 * x + 2) = (6 * x + 4) / (10 * x + 7) :=
by
  sorry

end ratio_equality_solution_l1476_147687


namespace metallic_sheet_length_l1476_147627

/-- The length of a rectangular metallic sheet that forms an open box with given dimensions and volume -/
theorem metallic_sheet_length : ∃ (L : ℝ),
  (L > 0) ∧ 
  (L - 2 * 8) * (36 - 2 * 8) * 8 = 5120 ∧ 
  L = 48 := by
  sorry

end metallic_sheet_length_l1476_147627
