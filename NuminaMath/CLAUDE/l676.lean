import Mathlib

namespace NUMINAMATH_CALUDE_monotone_increasing_interval_l676_67696

/-- The function f(x) = sin(x/2) + cos(x/2) is monotonically increasing 
    on the intervals [4kπ - 3π/2, 4kπ + π/2] for all integer k. -/
theorem monotone_increasing_interval (k : ℤ) :
  StrictMonoOn (fun x => Real.sin (x/2) + Real.cos (x/2))
    (Set.Icc (4 * k * Real.pi - 3 * Real.pi / 2) (4 * k * Real.pi + Real.pi / 2)) :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_interval_l676_67696


namespace NUMINAMATH_CALUDE_simple_interest_problem_l676_67697

theorem simple_interest_problem (simple_interest rate time : ℝ) 
  (h1 : simple_interest = 100)
  (h2 : rate = 5)
  (h3 : time = 4) :
  simple_interest = (500 * rate * time) / 100 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l676_67697


namespace NUMINAMATH_CALUDE_book_distribution_l676_67674

theorem book_distribution (n m : ℕ) (hn : n = 7) (hm : m = 3) :
  (Nat.factorial n) / (Nat.factorial (n - m)) = 210 :=
sorry

end NUMINAMATH_CALUDE_book_distribution_l676_67674


namespace NUMINAMATH_CALUDE_brick_height_is_6cm_l676_67630

/-- Proves that the height of each brick is 6 cm given the wall dimensions,
    brick base dimensions, and the number of bricks needed. -/
theorem brick_height_is_6cm 
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
  (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℕ) :
  wall_length = 750 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 25 →
  brick_width = 11.25 →
  num_bricks = 6000 →
  ∃ (brick_height : ℝ), 
    wall_length * wall_width * wall_height = 
    (brick_length * brick_width * brick_height) * num_bricks ∧
    brick_height = 6 :=
by sorry

end NUMINAMATH_CALUDE_brick_height_is_6cm_l676_67630


namespace NUMINAMATH_CALUDE_p_one_eq_p_two_p_decreasing_l676_67605

/-- The number of items in the collection -/
def n : ℕ := 10

/-- The probability of finding any specific item in a randomly chosen container -/
def prob_item : ℝ := 0.1

/-- The probability that exactly k items are missing from the second collection when the first collection is completed -/
noncomputable def p (k : ℕ) : ℝ := sorry

/-- Theorem stating that p_1 equals p_2 -/
theorem p_one_eq_p_two : p 1 = p 2 := by sorry

/-- Theorem stating the strict decreasing order of probabilities -/
theorem p_decreasing {i j : ℕ} (h1 : 2 ≤ i) (h2 : i < j) (h3 : j ≤ n) : p i > p j := by sorry

end NUMINAMATH_CALUDE_p_one_eq_p_two_p_decreasing_l676_67605


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l676_67687

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + m = 0 → y = x) → 
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l676_67687


namespace NUMINAMATH_CALUDE_unique_solution_5a_7b_plus_4_eq_3c_l676_67664

theorem unique_solution_5a_7b_plus_4_eq_3c :
  ∀ a b c : ℕ, 5^a * 7^b + 4 = 3^c → a = 1 ∧ b = 0 ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_5a_7b_plus_4_eq_3c_l676_67664


namespace NUMINAMATH_CALUDE_angle_of_inclination_range_l676_67638

theorem angle_of_inclination_range (θ : ℝ) :
  let α := Real.arctan (1 / (Real.sin θ))
  (∃ x y, x - y * Real.sin θ + 1 = 0) →
  π / 4 ≤ α ∧ α ≤ 3 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_angle_of_inclination_range_l676_67638


namespace NUMINAMATH_CALUDE_remainder_789987_div_8_l676_67607

theorem remainder_789987_div_8 : 789987 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_789987_div_8_l676_67607


namespace NUMINAMATH_CALUDE_linear_system_fraction_l676_67698

theorem linear_system_fraction (x y m n : ℚ) 
  (eq1 : x + 2*y = 5)
  (eq2 : x + y = 7)
  (eq3 : x = -m)
  (eq4 : y = -n) :
  (3*m + 2*n) / (5*m - n) = 11/14 := by
sorry

end NUMINAMATH_CALUDE_linear_system_fraction_l676_67698


namespace NUMINAMATH_CALUDE_female_managers_count_female_managers_is_200_l676_67684

/-- The number of female managers in a company, given certain conditions. -/
theorem female_managers_count (total_employees : ℕ) : ℕ := by
  -- Define the total number of female employees
  let female_employees : ℕ := 500

  -- Define the ratio of managers to all employees
  let manager_ratio : ℚ := 2 / 5

  -- Define the ratio of male managers to male employees
  let male_manager_ratio : ℚ := 2 / 5

  -- The number of female managers
  let female_managers : ℕ := 200

  sorry

/-- The main theorem stating that the number of female managers is 200. -/
theorem female_managers_is_200 : female_managers_count = 200 := by
  sorry

end NUMINAMATH_CALUDE_female_managers_count_female_managers_is_200_l676_67684


namespace NUMINAMATH_CALUDE_probability_divisible_by_8_l676_67616

def is_valid_digit (d : ℕ) : Prop := d ∈ ({3, 58} : Set ℕ)

def form_number (x y : ℕ) : ℕ := 460000 + x * 1000 + y * 100 + 12

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

theorem probability_divisible_by_8 :
  ∀ x y : ℕ, is_valid_digit x → is_valid_digit y →
  (∃! y', is_valid_digit y' ∧ is_divisible_by_8 (form_number x y')) :=
sorry

end NUMINAMATH_CALUDE_probability_divisible_by_8_l676_67616


namespace NUMINAMATH_CALUDE_m_range_l676_67609

/-- Proposition p: For all x ∈ ℝ, |x| + x ≥ 0 -/
def prop_p : Prop := ∀ x : ℝ, |x| + x ≥ 0

/-- Proposition q: The equation x² + mx + 1 = 0 has real roots -/
def prop_q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m*x + 1 = 0

/-- The composite proposition "p ∧ q" is false -/
axiom p_and_q_false : ∀ m : ℝ, ¬(prop_p ∧ prop_q m)

/-- The main theorem: Given the conditions above, prove that -2 < m < 2 -/
theorem m_range : ∀ m : ℝ, (¬(prop_p ∧ prop_q m)) → -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l676_67609


namespace NUMINAMATH_CALUDE_solve_for_y_l676_67628

theorem solve_for_y (x y : ℤ) (h1 : x = 4) (h2 : x + y = 0) : y = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l676_67628


namespace NUMINAMATH_CALUDE_collinear_vectors_sum_l676_67650

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ m : ℝ, b = (m * a.1, m * a.2.1, m * a.2.2)

/-- The problem statement -/
theorem collinear_vectors_sum (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (x, 2, 2)
  let b : ℝ × ℝ × ℝ := (2, y, 4)
  collinear a b → x + y = 5 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_sum_l676_67650


namespace NUMINAMATH_CALUDE_max_omega_for_monotonic_sin_l676_67612

/-- The maximum value of ω for which f(x) = sin(ωx) is monotonic on (-π/4, π/4) -/
theorem max_omega_for_monotonic_sin (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.sin (ω * x)) →
  ω > 0 →
  (∀ x y, -π/4 < x ∧ x < y ∧ y < π/4 → (f x < f y ∨ f x > f y)) →
  ω ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_omega_for_monotonic_sin_l676_67612


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l676_67682

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x < 1}

theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x : ℝ | x < -2} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l676_67682


namespace NUMINAMATH_CALUDE_linear_function_property_l676_67690

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g) (h_diff : g 10 - g 5 = 20) :
  g 20 - g 5 = 60 := by
sorry

end NUMINAMATH_CALUDE_linear_function_property_l676_67690


namespace NUMINAMATH_CALUDE_orange_trees_count_l676_67646

theorem orange_trees_count (total_trees apple_trees : ℕ) 
  (h1 : total_trees = 74) 
  (h2 : apple_trees = 47) : 
  total_trees - apple_trees = 27 := by
  sorry

end NUMINAMATH_CALUDE_orange_trees_count_l676_67646


namespace NUMINAMATH_CALUDE_expand_product_l676_67636

theorem expand_product (y : ℝ) : (y + 3) * (y + 7) = y^2 + 10*y + 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l676_67636


namespace NUMINAMATH_CALUDE_class_average_problem_l676_67693

theorem class_average_problem (x : ℝ) :
  (0.2 * x + 0.5 * 60 + 0.3 * 40 = 58) →
  x = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l676_67693


namespace NUMINAMATH_CALUDE_select_three_from_boys_and_girls_l676_67657

theorem select_three_from_boys_and_girls :
  let num_boys : ℕ := 4
  let num_girls : ℕ := 3
  let total_to_select : ℕ := 3
  let ways_to_select : ℕ := 
    (num_boys.choose 2 * num_girls.choose 1) + 
    (num_boys.choose 1 * num_girls.choose 2)
  ways_to_select = 30 := by
sorry

end NUMINAMATH_CALUDE_select_three_from_boys_and_girls_l676_67657


namespace NUMINAMATH_CALUDE_least_square_tiles_l676_67651

/-- Given a rectangular room with length 624 cm and width 432 cm, 
    the least number of square tiles of equal size required to cover the entire floor is 117. -/
theorem least_square_tiles (length width : ℕ) (h1 : length = 624) (h2 : width = 432) : 
  (length / (Nat.gcd length width)) * (width / (Nat.gcd length width)) = 117 := by
  sorry

end NUMINAMATH_CALUDE_least_square_tiles_l676_67651


namespace NUMINAMATH_CALUDE_sequence_properties_l676_67602

/-- The sum of the first n terms of sequence a_n -/
def S (n : ℕ) : ℕ := 2^(n+1) - 2

/-- The n-th term of sequence a_n -/
def a (n : ℕ) : ℕ := 2^n

/-- The n-th term of sequence b_n -/
def b (n : ℕ) : ℕ := n * a n

/-- The sum of the first n terms of sequence b_n -/
def T (n : ℕ) : ℕ := (n-1) * 2^(n+1) + 2

theorem sequence_properties (n : ℕ) :
  (∀ k, S k = 2^(k+1) - 2) →
  (∀ k, a k = 2^k) ∧
  T n = (n-1) * 2^(n+1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l676_67602


namespace NUMINAMATH_CALUDE_complex_equation_solution_l676_67666

theorem complex_equation_solution (a : ℝ) (h : (1 + a * Complex.I) * Complex.I = 3 + Complex.I) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l676_67666


namespace NUMINAMATH_CALUDE_exists_complete_list_l676_67660

/-- Represents a tournament where each competitor meets every other competitor exactly once with no draws -/
structure Tournament (α : Type*) :=
  (competitors : Set α)
  (beats : α → α → Prop)
  (all_play_once : ∀ x y : α, x ≠ y → (beats x y ∨ beats y x) ∧ ¬(beats x y ∧ beats y x))

/-- The list of players beaten by a given player and those beaten by the players they've beaten -/
def extended_wins (T : Tournament α) (x : α) : Set α :=
  {y | T.beats x y ∨ ∃ z, T.beats x z ∧ T.beats z y}

/-- There exists a player whose extended wins list includes all other players -/
theorem exists_complete_list (T : Tournament α) :
  ∃ x : α, ∀ y : α, y ≠ x → y ∈ extended_wins T x := by
  sorry


end NUMINAMATH_CALUDE_exists_complete_list_l676_67660


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l676_67656

/-- The trajectory of the midpoint of a line segment with one endpoint fixed and the other moving on a circle -/
theorem midpoint_trajectory (A B M : ℝ × ℝ) : 
  (B = (4, 0)) →  -- B is fixed at (4, 0)
  (∀ t : ℝ, A.1^2 + A.2^2 = 4) →  -- A moves on the circle x^2 + y^2 = 4
  (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  (M.1 - 2)^2 + M.2^2 = 1 :=  -- The trajectory of M is (x-2)^2 + y^2 = 1
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l676_67656


namespace NUMINAMATH_CALUDE_binomial_expansion_unique_m_l676_67643

/-- Given constants b and y, and a natural number m, such that the second, third, and fourth terms
    in the expansion of (b + y)^m are 6, 24, and 60 respectively, prove that m = 11. -/
theorem binomial_expansion_unique_m (b y : ℝ) (m : ℕ) 
  (h1 : (m.choose 1) * b^(m-1) * y = 6)
  (h2 : (m.choose 2) * b^(m-2) * y^2 = 24)
  (h3 : (m.choose 3) * b^(m-3) * y^3 = 60) :
  m = 11 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_unique_m_l676_67643


namespace NUMINAMATH_CALUDE_irrational_approximation_l676_67673

-- Define α as an irrational real number
variable (α : ℝ) (h : ¬ IsRat α)

-- State the theorem
theorem irrational_approximation :
  ∃ (p q : ℤ), -(1 : ℝ) / (q : ℝ)^2 ≤ α - (p : ℝ) / (q : ℝ) ∧ α - (p : ℝ) / (q : ℝ) ≤ 1 / (q : ℝ)^2 :=
sorry

end NUMINAMATH_CALUDE_irrational_approximation_l676_67673


namespace NUMINAMATH_CALUDE_problem_solution_l676_67647

theorem problem_solution (a : ℝ) : 3 ∈ ({a + 3, 2 * a + 1, a^2 + a + 1} : Set ℝ) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l676_67647


namespace NUMINAMATH_CALUDE_range_of_m_l676_67639

/-- The function f(x) defined as 1 / √(mx² + mx + 1) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1 / Real.sqrt (m * x^2 + m * x + 1)

/-- The set of real numbers m for which f(x) has domain ℝ -/
def valid_m : Set ℝ := {m : ℝ | ∀ x : ℝ, m * x^2 + m * x + 1 > 0}

theorem range_of_m : valid_m = Set.Ici 0 ∩ Set.Iio 4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l676_67639


namespace NUMINAMATH_CALUDE_tan_ratio_sum_l676_67611

theorem tan_ratio_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_sum_l676_67611


namespace NUMINAMATH_CALUDE_function_inequality_l676_67699

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x < -1, (deriv f) x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l676_67699


namespace NUMINAMATH_CALUDE_nested_subtraction_simplification_l676_67678

theorem nested_subtraction_simplification (x : ℝ) : 1 - (2 - (3 - (4 - (5 - (6 - x))))) = x - 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_subtraction_simplification_l676_67678


namespace NUMINAMATH_CALUDE_expression_evaluation_l676_67623

theorem expression_evaluation : 
  (2015^3 - 2 * 2015^2 * 2016 + 3 * 2015 * 2016^2 - 2016^3 + 1) / (2015 * 2016) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l676_67623


namespace NUMINAMATH_CALUDE_gcf_of_75_and_100_l676_67642

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_100_l676_67642


namespace NUMINAMATH_CALUDE_francies_allowance_l676_67662

/-- Francie's allowance problem -/
theorem francies_allowance (x : ℚ) : 
  (∀ (total_saved half_spent remaining : ℚ),
    total_saved = 8 * x + 6 * 6 →
    half_spent = total_saved / 2 →
    remaining = half_spent - 35 →
    remaining = 3) →
  x = 5 := by sorry

end NUMINAMATH_CALUDE_francies_allowance_l676_67662


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l676_67675

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def mersenne_prime (p : ℕ) : Prop := is_prime p ∧ ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_under_500 :
  ∃ p : ℕ, mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, mersenne_prime q → q < 500 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l676_67675


namespace NUMINAMATH_CALUDE_fraction_equality_l676_67631

theorem fraction_equality : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l676_67631


namespace NUMINAMATH_CALUDE_sum_range_l676_67622

theorem sum_range (x y z : ℝ) 
  (eq1 : x + 2*y + 3*z = 1) 
  (eq2 : y*z + z*x + x*y = -1) : 
  (3 - 3*Real.sqrt 3) / 4 ≤ x + y + z ∧ x + y + z ≤ (3 + 3*Real.sqrt 3) / 4 :=
sorry

end NUMINAMATH_CALUDE_sum_range_l676_67622


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l676_67641

theorem ellipse_eccentricity (C2 : Set (ℝ × ℝ)) : 
  (∀ x y, (x = Real.sqrt 5 ∧ y = 0) ∨ (x = 0 ∧ y = 3) → (x, y) ∈ C2) →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ x y, (x, y) ∈ C2 ↔ x^2/a^2 + y^2/b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    c/a = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l676_67641


namespace NUMINAMATH_CALUDE_f_properties_l676_67683

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 - x)

theorem f_properties :
  (∃ (max_val : ℝ), max_val = -4 ∧ ∀ x ≠ 1, f x ≤ max_val) ∧
  (∀ x ≠ 1, f (1 - x) + f (1 + x) = -4) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 1 → x₂ > 1 → f ((x₁ + x₂) / 2) ≥ (f x₁ + f x₂) / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l676_67683


namespace NUMINAMATH_CALUDE_not_perfect_square_l676_67681

theorem not_perfect_square (n : ℕ) (h : n > 1) : ¬ ∃ (a : ℕ), 4 * 10^n + 9 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l676_67681


namespace NUMINAMATH_CALUDE_hispanic_west_percentage_l676_67610

/-- Represents the population data for a specific ethnic group across regions -/
structure PopulationData :=
  (ne : ℕ) (mw : ℕ) (south : ℕ) (west : ℕ)

/-- Calculates the total population across all regions -/
def total_population (data : PopulationData) : ℕ :=
  data.ne + data.mw + data.south + data.west

/-- Calculates the percentage of population in the West, rounded to the nearest percent -/
def west_percentage (data : PopulationData) : ℕ :=
  (data.west * 100 + (total_population data) / 2) / (total_population data)

/-- The given Hispanic population data for 1990 in millions -/
def hispanic_data : PopulationData :=
  { ne := 4, mw := 5, south := 12, west := 20 }

theorem hispanic_west_percentage :
  west_percentage hispanic_data = 49 := by sorry

end NUMINAMATH_CALUDE_hispanic_west_percentage_l676_67610


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l676_67600

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h4 : seq.S 4 = -4)
    (h6 : seq.S 6 = 6) :
    seq.S 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l676_67600


namespace NUMINAMATH_CALUDE_jimmy_lost_points_l676_67688

def jimmy_problem (points_to_pass : ℕ) (points_per_exam : ℕ) (num_exams : ℕ) (extra_points : ℕ) : Prop :=
  let total_exam_points := points_per_exam * num_exams
  let current_points := points_to_pass + extra_points
  let lost_points := total_exam_points - current_points
  lost_points = 5

theorem jimmy_lost_points :
  jimmy_problem 50 20 3 5 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_lost_points_l676_67688


namespace NUMINAMATH_CALUDE_polynomial_sum_l676_67667

-- Define the polynomials f and g
def f (a b x : ℝ) := x^2 + a*x + b
def g (c d x : ℝ) := x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) : 
  (∃ (x : ℝ), g c d x = 0 ∧ x = -a/2) →  -- x-coordinate of vertex of f is root of g
  (∃ (x : ℝ), f a b x = 0 ∧ x = -c/2) →  -- x-coordinate of vertex of g is root of f
  (f a b (-a/2) = -25) →                 -- minimum value of f is -25
  (g c d (-c/2) = -25) →                 -- minimum value of g is -25
  (f a b 50 = -50) →                     -- f and g intersect at (50, -50)
  (g c d 50 = -50) →                     -- f and g intersect at (50, -50)
  (a ≠ c ∨ b ≠ d) →                      -- f and g are distinct
  a + c = -200 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l676_67667


namespace NUMINAMATH_CALUDE_laura_garden_area_l676_67627

/-- Represents a rectangular garden with fence posts --/
structure Garden where
  total_posts : ℕ
  gap : ℕ
  longer_side_post_ratio : ℕ

/-- Calculates the area of the garden given its specifications --/
def garden_area (g : Garden) : ℕ :=
  let shorter_side_posts := (g.total_posts + 4) / (1 + g.longer_side_post_ratio)
  let longer_side_posts := shorter_side_posts * g.longer_side_post_ratio
  let shorter_side_length := (shorter_side_posts - 1) * g.gap
  let longer_side_length := (longer_side_posts - 1) * g.gap
  shorter_side_length * longer_side_length

theorem laura_garden_area :
  let g : Garden := { total_posts := 24, gap := 5, longer_side_post_ratio := 3 }
  garden_area g = 3000 := by
  sorry


end NUMINAMATH_CALUDE_laura_garden_area_l676_67627


namespace NUMINAMATH_CALUDE_phone_price_reduction_l676_67680

theorem phone_price_reduction (reduced_price : ℝ) (percentage : ℝ) 
  (h1 : reduced_price = 1800)
  (h2 : percentage = 90/100)
  (h3 : reduced_price = percentage * (reduced_price / percentage)) :
  reduced_price / percentage - reduced_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_phone_price_reduction_l676_67680


namespace NUMINAMATH_CALUDE_range_of_a_l676_67617

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}

-- State the theorem
theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l676_67617


namespace NUMINAMATH_CALUDE_winter_clothing_count_l676_67604

/-- The number of boxes of clothing -/
def num_boxes : ℕ := 6

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 5

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 5

/-- The total number of pieces of winter clothing -/
def total_clothing : ℕ := num_boxes * (scarves_per_box + mittens_per_box)

theorem winter_clothing_count : total_clothing = 60 := by
  sorry

end NUMINAMATH_CALUDE_winter_clothing_count_l676_67604


namespace NUMINAMATH_CALUDE_new_shipment_bears_l676_67608

/-- Calculates the number of bears in a new shipment given the initial stock,
    bears per shelf, and number of shelves used. -/
theorem new_shipment_bears (initial_stock : ℕ) (bears_per_shelf : ℕ) (shelves_used : ℕ) :
  (bears_per_shelf * shelves_used) - initial_stock =
  (bears_per_shelf * shelves_used) - initial_stock :=
by sorry

end NUMINAMATH_CALUDE_new_shipment_bears_l676_67608


namespace NUMINAMATH_CALUDE_last_digit_of_repeated_seven_exponentiation_l676_67648

def repeated_exponentiation (base : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => base
  | n + 1 => repeated_exponentiation (base^base) n

theorem last_digit_of_repeated_seven_exponentiation :
  repeated_exponentiation 7 1000 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_repeated_seven_exponentiation_l676_67648


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l676_67603

theorem polynomial_divisibility (m n p : ℕ) : 
  ∃ q : Polynomial ℤ, (X^2 + X + 1) * q = X^(3*m) + X^(n+1) + X^(p+2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l676_67603


namespace NUMINAMATH_CALUDE_fraction_squared_equality_l676_67629

theorem fraction_squared_equality : ((-123456789 : ℤ) / 246913578)^2 = (15241578750190521 : ℚ) / 60995928316126584 := by
  sorry

end NUMINAMATH_CALUDE_fraction_squared_equality_l676_67629


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l676_67668

theorem unique_root_quadratic (k : ℝ) :
  (∃! a : ℝ, (k^2 - 9) * a^2 - 2*(k + 1)*a + 1 = 0) →
  (k = 3 ∨ k = -3 ∨ k = -5) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_quadratic_l676_67668


namespace NUMINAMATH_CALUDE_convex_prism_right_iff_not_four_l676_67618

/-- A convex n-sided prism with congruent lateral faces -/
structure ConvexPrism (n : ℕ) where
  /-- The prism is convex -/
  convex : Bool
  /-- The prism has n sides -/
  sides : Fin n
  /-- All lateral faces are congruent -/
  congruentLateralFaces : Bool

/-- A prism is right if all its lateral edges are perpendicular to its bases -/
def isRight (p : ConvexPrism n) : Prop := sorry

/-- Main theorem: A convex n-sided prism with congruent lateral faces is necessarily right if and only if n ≠ 4 -/
theorem convex_prism_right_iff_not_four (n : ℕ) (p : ConvexPrism n) :
  p.convex ∧ p.congruentLateralFaces → (isRight p ↔ n ≠ 4) := by sorry

end NUMINAMATH_CALUDE_convex_prism_right_iff_not_four_l676_67618


namespace NUMINAMATH_CALUDE_total_bananas_used_l676_67659

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The number of loaves made on Monday -/
def monday_loaves : ℕ := 3

/-- The number of loaves made on Tuesday -/
def tuesday_loaves : ℕ := 2 * monday_loaves

/-- The total number of loaves made on both days -/
def total_loaves : ℕ := monday_loaves + tuesday_loaves

/-- Theorem: The total number of bananas used is 36 -/
theorem total_bananas_used : bananas_per_loaf * total_loaves = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_bananas_used_l676_67659


namespace NUMINAMATH_CALUDE_race_start_theorem_l676_67671

/-- Represents the start distance one runner can give another in a kilometer race -/
def start_distance (runner1 runner2 : ℕ) : ℝ := sorry

/-- The race distance in meters -/
def race_distance : ℝ := 1000

theorem race_start_theorem (A B C : ℕ) :
  start_distance A B = 50 →
  start_distance B C = 52.63157894736844 →
  start_distance A C = 100 := by
  sorry

end NUMINAMATH_CALUDE_race_start_theorem_l676_67671


namespace NUMINAMATH_CALUDE_residue_625_mod_17_l676_67654

theorem residue_625_mod_17 : 625 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_residue_625_mod_17_l676_67654


namespace NUMINAMATH_CALUDE_specific_triangle_count_is_32_l676_67635

/-- Represents the count of triangles at different levels in a structure --/
structure TriangleCount where
  smallest : Nat
  intermediate : Nat
  larger : Nat
  even_larger : Nat
  whole_structure : Nat

/-- Calculates the total number of triangles in the structure --/
def total_triangles (count : TriangleCount) : Nat :=
  count.smallest + count.intermediate + count.larger + count.even_larger + count.whole_structure

/-- Theorem stating that for a specific triangle count, the total number of triangles is 32 --/
theorem specific_triangle_count_is_32 :
  ∃ (count : TriangleCount),
    count.smallest = 2 ∧
    count.intermediate = 6 ∧
    count.larger = 6 ∧
    count.even_larger = 6 ∧
    count.whole_structure = 12 ∧
    total_triangles count = 32 := by
  sorry

#eval total_triangles { smallest := 2, intermediate := 6, larger := 6, even_larger := 6, whole_structure := 12 }

end NUMINAMATH_CALUDE_specific_triangle_count_is_32_l676_67635


namespace NUMINAMATH_CALUDE_angle_calculation_l676_67658

theorem angle_calculation (α : ℝ) (h : α = 30) : 
  2 * (90 - α) - (90 - (180 - α)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_calculation_l676_67658


namespace NUMINAMATH_CALUDE_function_value_l676_67621

theorem function_value (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 1) : f 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_function_value_l676_67621


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l676_67677

theorem mean_equality_implies_z_value :
  let x₁ : ℚ := 7
  let x₂ : ℚ := 11
  let x₃ : ℚ := 23
  let y₁ : ℚ := 15
  let mean_xyz : ℚ := (x₁ + x₂ + x₃) / 3
  let mean_yz : ℚ := (y₁ + z) / 2
  mean_xyz = mean_yz → z = 37 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l676_67677


namespace NUMINAMATH_CALUDE_tan_100_degrees_l676_67632

theorem tan_100_degrees (k : ℝ) (h : Real.sin (-(80 * π / 180)) = k) :
  Real.tan ((100 * π / 180)) = k / Real.sqrt (1 - k^2) := by
  sorry

end NUMINAMATH_CALUDE_tan_100_degrees_l676_67632


namespace NUMINAMATH_CALUDE_tomato_price_is_fifty_cents_l676_67624

/-- Represents the monthly sales data for Village Foods --/
structure VillageFoodsSales where
  customers : ℕ
  lettucePerCustomer : ℕ
  tomatoesPerCustomer : ℕ
  lettucePricePerHead : ℚ
  totalSales : ℚ

/-- Calculates the price of each tomato based on the sales data --/
def tomatoPrice (sales : VillageFoodsSales) : ℚ :=
  let lettuceSales := sales.customers * sales.lettucePerCustomer * sales.lettucePricePerHead
  let tomatoSales := sales.totalSales - lettuceSales
  let totalTomatoes := sales.customers * sales.tomatoesPerCustomer
  tomatoSales / totalTomatoes

/-- Theorem stating that the tomato price is $0.50 given the specific sales data --/
theorem tomato_price_is_fifty_cents 
  (sales : VillageFoodsSales)
  (h1 : sales.customers = 500)
  (h2 : sales.lettucePerCustomer = 2)
  (h3 : sales.tomatoesPerCustomer = 4)
  (h4 : sales.lettucePricePerHead = 1)
  (h5 : sales.totalSales = 2000) :
  tomatoPrice sales = 1/2 := by
  sorry

#eval tomatoPrice {
  customers := 500,
  lettucePerCustomer := 2,
  tomatoesPerCustomer := 4,
  lettucePricePerHead := 1,
  totalSales := 2000
}

end NUMINAMATH_CALUDE_tomato_price_is_fifty_cents_l676_67624


namespace NUMINAMATH_CALUDE_iains_pennies_l676_67645

theorem iains_pennies (initial_pennies : ℕ) (older_pennies : ℕ) (discard_percentage : ℚ) : 
  initial_pennies = 200 →
  older_pennies = 30 →
  discard_percentage = 1/5 →
  initial_pennies - older_pennies - (initial_pennies - older_pennies) * discard_percentage = 136 := by
  sorry

end NUMINAMATH_CALUDE_iains_pennies_l676_67645


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_measure_l676_67694

theorem right_triangle_acute_angle_measure (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b = 90) ∧ (a / b = 5 / 4) → min a b = 40 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_measure_l676_67694


namespace NUMINAMATH_CALUDE_thirtieth_sum_l676_67652

/-- Represents the sum of elements in the nth set of a sequence where each set starts one more than
    the last element of the preceding set and has one more element than the one before it. -/
def T (n : ℕ) : ℕ :=
  let first := 1 + (n * (n - 1)) / 2
  let last := first + n - 1
  n * (first + last) / 2

/-- The 30th sum in the sequence equals 13515. -/
theorem thirtieth_sum : T 30 = 13515 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_sum_l676_67652


namespace NUMINAMATH_CALUDE_winning_bet_amount_l676_67637

def initial_amount : ℕ := 400

def bet_multiplier : ℕ := 2

theorem winning_bet_amount (initial : ℕ) (multiplier : ℕ) :
  initial = initial_amount →
  multiplier = bet_multiplier →
  initial + (multiplier * initial) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_winning_bet_amount_l676_67637


namespace NUMINAMATH_CALUDE_circle_radius_through_three_points_l676_67653

/-- The radius of the circle passing through three given points is 5 -/
theorem circle_radius_through_three_points : ∃ (center : ℝ × ℝ) (r : ℝ),
  r = 5 ∧
  (center.1 - 1)^2 + (center.2 - 3)^2 = r^2 ∧
  (center.1 - 4)^2 + (center.2 - 2)^2 = r^2 ∧
  (center.1 - 1)^2 + (center.2 - (-7))^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_through_three_points_l676_67653


namespace NUMINAMATH_CALUDE_roberts_gre_preparation_time_l676_67620

/-- Represents the preparation time for each subject in the GRE examination -/
structure GREPreparation where
  vocabulary : Nat
  writing : Nat
  quantitative : Nat

/-- Calculates the total preparation time for the GRE examination -/
def totalPreparationTime (prep : GREPreparation) : Nat :=
  prep.vocabulary + prep.writing + prep.quantitative

/-- Theorem: The total preparation time for Robert's GRE examination is 8 months -/
theorem roberts_gre_preparation_time :
  let robert_prep : GREPreparation := ⟨3, 2, 3⟩
  totalPreparationTime robert_prep = 8 := by
  sorry

#check roberts_gre_preparation_time

end NUMINAMATH_CALUDE_roberts_gre_preparation_time_l676_67620


namespace NUMINAMATH_CALUDE_remaining_money_for_sharpeners_l676_67692

def total_money : ℕ := 100
def notebook_price : ℕ := 5
def notebooks_bought : ℕ := 4
def eraser_price : ℕ := 4
def erasers_bought : ℕ := 10
def highlighter_cost : ℕ := 30

def heaven_notebook_cost : ℕ := notebook_price * notebooks_bought
def brother_eraser_cost : ℕ := eraser_price * erasers_bought
def brother_total_cost : ℕ := brother_eraser_cost + highlighter_cost

theorem remaining_money_for_sharpeners :
  total_money - (heaven_notebook_cost + brother_total_cost) = 10 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_for_sharpeners_l676_67692


namespace NUMINAMATH_CALUDE_xiao_liang_arrival_time_l676_67691

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60,
    h_valid := by sorry
    m_valid := by sorry }

theorem xiao_liang_arrival_time :
  let departure_time : Time := ⟨7, 40, by sorry, by sorry⟩
  let journey_duration : Nat := 25
  let arrival_time : Time := addMinutes departure_time journey_duration
  arrival_time = ⟨8, 5, by sorry, by sorry⟩ := by sorry

end NUMINAMATH_CALUDE_xiao_liang_arrival_time_l676_67691


namespace NUMINAMATH_CALUDE_initial_persons_count_l676_67640

/-- The number of persons initially present -/
def N : ℕ := sorry

/-- The initial average weight -/
def initial_average : ℝ := sorry

/-- The weight increase when the new person replaces one person -/
def weight_increase : ℝ := 4

/-- The weight of the person being replaced -/
def replaced_weight : ℝ := 65

/-- The weight of the new person -/
def new_weight : ℝ := 97

theorem initial_persons_count : N = 8 := by sorry

end NUMINAMATH_CALUDE_initial_persons_count_l676_67640


namespace NUMINAMATH_CALUDE_paper_covers_cube_l676_67633

theorem paper_covers_cube (cube_edge : ℝ) (paper_side : ℝ) 
  (h1 : cube_edge = 1) (h2 : paper_side = 2.5) : 
  paper_side ^ 2 ≥ 6 * cube_edge ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_paper_covers_cube_l676_67633


namespace NUMINAMATH_CALUDE_distribute_six_interns_three_schools_l676_67626

/-- The number of ways to distribute n interns among k schools, where each intern is assigned to exactly one school and each school receives at least one intern. -/
def distribute_interns (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 540 ways to distribute 6 interns among 3 schools under the given conditions. -/
theorem distribute_six_interns_three_schools : distribute_interns 6 3 = 540 := by sorry

end NUMINAMATH_CALUDE_distribute_six_interns_three_schools_l676_67626


namespace NUMINAMATH_CALUDE_relationship_abc_l676_67625

theorem relationship_abc (a b c : ℝ) (ha : a = Real.sqrt 6 + Real.sqrt 7) 
  (hb : b = Real.sqrt 5 + Real.sqrt 8) (hc : c = 5) : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l676_67625


namespace NUMINAMATH_CALUDE_sarahs_coin_box_l676_67655

/-- The number of pennies in Sarah's box --/
def num_coins : ℕ := 36

/-- The total value of coins in cents --/
def total_value : ℕ := 2000

/-- Theorem stating that the number of each type of coin in Sarah's box is 36,
    given that the total value is $20 (2000 cents) and there are equal numbers
    of pennies, nickels, and half-dollars. --/
theorem sarahs_coin_box :
  (num_coins : ℚ) * (1 + 5 + 50) = total_value :=
sorry

end NUMINAMATH_CALUDE_sarahs_coin_box_l676_67655


namespace NUMINAMATH_CALUDE_rectangle_area_l676_67606

theorem rectangle_area (length width : ℝ) (h1 : length = 20) (h2 : length = 4 * width) : length * width = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l676_67606


namespace NUMINAMATH_CALUDE_river_width_l676_67663

/-- Proves that given a river with specified depth, flow rate, and discharge volume,
    the width of the river is 25 meters. -/
theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (discharge_volume : ℝ) :
  depth = 8 →
  flow_rate_kmph = 8 →
  discharge_volume = 26666.666666666668 →
  (discharge_volume / (depth * (flow_rate_kmph * 1000 / 60))) = 25 := by
  sorry


end NUMINAMATH_CALUDE_river_width_l676_67663


namespace NUMINAMATH_CALUDE_ice_cream_sundae_combinations_l676_67619

/-- The number of unique two-scoop sundae combinations given a total number of flavors and vanilla as a required flavor. -/
def sundae_combinations (total_flavors : ℕ) (vanilla_required : Bool) : ℕ :=
  if vanilla_required then total_flavors - 1 else 0

/-- Theorem: Given 8 ice cream flavors with vanilla required, the number of unique two-scoop sundae combinations is 7. -/
theorem ice_cream_sundae_combinations :
  sundae_combinations 8 true = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundae_combinations_l676_67619


namespace NUMINAMATH_CALUDE_hotel_reunions_l676_67661

theorem hotel_reunions (total_guests : ℕ) (oates_attendees : ℕ) (hall_attendees : ℕ)
  (h1 : total_guests = 100)
  (h2 : oates_attendees = 50)
  (h3 : hall_attendees = 62)
  (h4 : ∀ g, g ≤ total_guests → (g ≤ oates_attendees ∨ g ≤ hall_attendees)) :
  oates_attendees + hall_attendees - total_guests = 12 := by
  sorry

end NUMINAMATH_CALUDE_hotel_reunions_l676_67661


namespace NUMINAMATH_CALUDE_solution_set_x_abs_x_less_x_l676_67644

theorem solution_set_x_abs_x_less_x :
  {x : ℝ | x * |x| < x} = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x < -1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_x_abs_x_less_x_l676_67644


namespace NUMINAMATH_CALUDE_percentage_without_scholarship_l676_67685

/-- Represents the percentage of students who won't get a scholarship in a school with a given ratio of boys to girls and scholarship rates. -/
theorem percentage_without_scholarship
  (boy_girl_ratio : ℚ)
  (boy_scholarship_rate : ℚ)
  (girl_scholarship_rate : ℚ)
  (h1 : boy_girl_ratio = 5 / 6)
  (h2 : boy_scholarship_rate = 1 / 4)
  (h3 : girl_scholarship_rate = 1 / 5) :
  (1 - (boy_girl_ratio * boy_scholarship_rate + girl_scholarship_rate) / (boy_girl_ratio + 1)) * 100 =
  (1 - (1.25 + 1.2) / 11) * 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_without_scholarship_l676_67685


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l676_67665

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def P : Set Nat := {1, 2, 3, 4, 5}
def Q : Set Nat := {3, 4, 5, 6, 7}

theorem intersection_complement_equality : P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l676_67665


namespace NUMINAMATH_CALUDE_max_subsets_exists_444_subsets_l676_67676

/-- A structure representing a collection of 3-element subsets of a 1000-element set. -/
structure SubsetCollection where
  /-- The underlying 1000-element set -/
  base : Finset (Fin 1000)
  /-- The collection of 3-element subsets -/
  subsets : Finset (Finset (Fin 1000))
  /-- Each subset has exactly 3 elements -/
  three_element : ∀ s ∈ subsets, Finset.card s = 3
  /-- Each subset is a subset of the base set -/
  subset_of_base : ∀ s ∈ subsets, s ⊆ base
  /-- The union of any 5 subsets has at least 12 elements -/
  union_property : ∀ (five_subsets : Finset (Finset (Fin 1000))), 
    five_subsets ⊆ subsets → Finset.card five_subsets = 5 → 
    Finset.card (Finset.biUnion five_subsets id) ≥ 12

/-- The maximum number of three-element subsets satisfying the given conditions is 444. -/
theorem max_subsets (sc : SubsetCollection) : Finset.card sc.subsets ≤ 444 := by
  sorry

/-- There exists a collection of 444 three-element subsets satisfying the given conditions. -/
theorem exists_444_subsets : ∃ sc : SubsetCollection, Finset.card sc.subsets = 444 := by
  sorry

end NUMINAMATH_CALUDE_max_subsets_exists_444_subsets_l676_67676


namespace NUMINAMATH_CALUDE_angles_equal_necessary_not_sufficient_l676_67695

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the angle between a line and a plane
variable (angle : Line → Plane → ℝ)

-- State the theorem
theorem angles_equal_necessary_not_sufficient
  (m n : Line) (a : Plane) :
  (∀ (l₁ l₂ : Line), parallel l₁ l₂ → angle l₁ a = angle l₂ a) ∧
  ¬(∀ (l₁ l₂ : Line), angle l₁ a = angle l₂ a → parallel l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_angles_equal_necessary_not_sufficient_l676_67695


namespace NUMINAMATH_CALUDE_diagonals_bisect_implies_parallelogram_parallelogram_right_angle_implies_rectangle_l676_67614

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Define the property of diagonals bisecting each other
def diagonals_bisect (q : Quadrilateral) : Prop := sorry

-- Define the property of having a right angle
def has_right_angle (q : Quadrilateral) : Prop := sorry

-- Theorem 1: If a quadrilateral has diagonals that bisect each other, then it is a parallelogram
theorem diagonals_bisect_implies_parallelogram (q : Quadrilateral) :
  diagonals_bisect q → is_parallelogram q :=
sorry

-- Theorem 2: If a parallelogram has one right angle, then it is a rectangle
theorem parallelogram_right_angle_implies_rectangle (q : Quadrilateral) :
  is_parallelogram q → has_right_angle q → is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_diagonals_bisect_implies_parallelogram_parallelogram_right_angle_implies_rectangle_l676_67614


namespace NUMINAMATH_CALUDE_parking_probability_l676_67689

/-- The number of parking spaces -/
def total_spaces : ℕ := 20

/-- The number of cars parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent spaces required -/
def required_spaces : ℕ := 3

/-- The probability of finding the required adjacent empty spaces -/
def probability_of_parking : ℚ := 12501 / 15504

theorem parking_probability :
  (total_spaces : ℕ) = 20 →
  (parked_cars : ℕ) = 15 →
  (required_spaces : ℕ) = 3 →
  probability_of_parking = 12501 / 15504 := by
  sorry

end NUMINAMATH_CALUDE_parking_probability_l676_67689


namespace NUMINAMATH_CALUDE_nedy_crackers_total_l676_67615

/-- The number of packs of crackers Nedy eats per day from Monday to Thursday -/
def daily_crackers : ℕ := 8

/-- The number of days from Monday to Thursday -/
def weekdays : ℕ := 4

/-- The factor by which Nedy increases his cracker consumption on Friday -/
def friday_factor : ℕ := 2

/-- Theorem: Given Nedy eats 8 packs of crackers per day from Monday to Thursday
    and twice that amount on Friday, the total number of crackers Nedy eats
    from Monday to Friday is 48 packs. -/
theorem nedy_crackers_total :
  daily_crackers * weekdays + daily_crackers * friday_factor = 48 := by
  sorry

end NUMINAMATH_CALUDE_nedy_crackers_total_l676_67615


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l676_67679

def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)

theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l676_67679


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l676_67601

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if a point lies on a line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem perpendicular_line_through_point :
  let l1 : Line := { a := 2, b := 3, c := -4 }
  let l2 : Line := { a := 3, b := -2, c := 2 }
  perpendicular l1 l2 ∧ point_on_line 0 1 l2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l676_67601


namespace NUMINAMATH_CALUDE_roses_picked_second_correct_l676_67613

-- Define the problem parameters
def initial_roses : ℝ := 37.0
def first_picking : ℝ := 16.0
def final_total : ℕ := 72

-- Define the function to calculate roses picked in the second picking
def roses_picked_second (initial : ℝ) (first : ℝ) (total : ℕ) : ℝ :=
  (total : ℝ) - (initial + first)

-- Theorem statement
theorem roses_picked_second_correct :
  roses_picked_second initial_roses first_picking final_total = 19.0 := by
  sorry

end NUMINAMATH_CALUDE_roses_picked_second_correct_l676_67613


namespace NUMINAMATH_CALUDE_max_remainder_eleven_l676_67669

theorem max_remainder_eleven (x : ℕ+) : ∃ (q r : ℕ), x = 11 * q + r ∧ r ≤ 10 ∧ ∀ (r' : ℕ), x = 11 * q + r' → r' ≤ r :=
sorry

end NUMINAMATH_CALUDE_max_remainder_eleven_l676_67669


namespace NUMINAMATH_CALUDE_square_difference_263_257_l676_67672

theorem square_difference_263_257 : 263^2 - 257^2 = 3120 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_263_257_l676_67672


namespace NUMINAMATH_CALUDE_x_lt_1_necessary_not_sufficient_for_ln_x_lt_0_l676_67686

theorem x_lt_1_necessary_not_sufficient_for_ln_x_lt_0 :
  (∀ x : ℝ, Real.log x < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ Real.log x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_lt_1_necessary_not_sufficient_for_ln_x_lt_0_l676_67686


namespace NUMINAMATH_CALUDE_min_draw_for_eight_same_color_l676_67634

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat
  white : Nat

/-- The minimum number of balls to draw to ensure at least n of the same color -/
def minDrawToEnsure (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The theorem to prove -/
theorem min_draw_for_eight_same_color (counts : BallCounts)
    (h_red : counts.red = 15)
    (h_green : counts.green = 12)
    (h_blue : counts.blue = 10)
    (h_yellow : counts.yellow = 7)
    (h_white : counts.white = 6)
    (h_total : counts.red + counts.green + counts.blue + counts.yellow + counts.white = 50) :
    minDrawToEnsure counts 8 = 35 := by
  sorry

end NUMINAMATH_CALUDE_min_draw_for_eight_same_color_l676_67634


namespace NUMINAMATH_CALUDE_joan_football_games_l676_67649

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := sorry

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan went to -/
def total_games : ℕ := 13

theorem joan_football_games : games_this_year = 4 := by sorry

end NUMINAMATH_CALUDE_joan_football_games_l676_67649


namespace NUMINAMATH_CALUDE_johns_hats_cost_l676_67670

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- The total cost of John's hats -/
def total_cost : ℕ := weeks_of_different_hats * days_per_week * cost_per_hat

theorem johns_hats_cost :
  total_cost = 700 := by sorry

end NUMINAMATH_CALUDE_johns_hats_cost_l676_67670
