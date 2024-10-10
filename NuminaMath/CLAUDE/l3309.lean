import Mathlib

namespace find_unknown_number_l3309_330986

theorem find_unknown_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((60 + 35 + x) / 3) + 5 → x = 10 := by
  sorry

end find_unknown_number_l3309_330986


namespace complete_square_equivalence_l3309_330910

/-- Given a quadratic equation x^2 - 4x = 5, prove that it is equivalent to (x-2)^2 = 9 when completed square. -/
theorem complete_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x = 5 ↔ (x-2)^2 = 9 := by
  sorry

end complete_square_equivalence_l3309_330910


namespace equation_solution_l3309_330966

theorem equation_solution : ∃! x : ℚ, 5 * (x - 4) = 3 * (3 - 3 * x) + 6 ∧ x = 5 / 2 := by
  sorry

end equation_solution_l3309_330966


namespace limit_cosine_ratio_l3309_330998

theorem limit_cosine_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |(1 - Real.cos (2*x)) / (Real.cos (7*x) - Real.cos (3*x)) + (1/10)| < ε :=
by
  sorry

end limit_cosine_ratio_l3309_330998


namespace station_A_relay_ways_l3309_330977

/-- Represents a communication station -/
inductive Station : Type
| A | B | C | D

/-- The number of stations excluding A -/
def num_other_stations : Nat := 3

/-- The total number of ways station A can relay the message -/
def total_relay_ways : Nat := 16

/-- Theorem stating the number of ways station A can relay the message -/
theorem station_A_relay_ways :
  (∀ s₁ s₂ : Station, s₁ ≠ s₂ → (∃ t : Nat, t > 0)) →  -- Stations can communicate pairwise
  (∀ s : Station, ∃ t : Nat, t > 0) →  -- Space station can send to any station
  (∀ s : Station, ∀ n : Nat, n > 1 → ¬∃ t : Nat, t > 0) →  -- No simultaneous transmissions
  (∃ n : Nat, n = 3) →  -- Three transmissions occurred
  (∀ s : Station, ∃ m : Nat, m > 0) →  -- All stations received the message
  total_relay_ways = (2^num_other_stations - 1) + num_other_stations * 2^(num_other_stations - 1) :=
by sorry

end station_A_relay_ways_l3309_330977


namespace function_properties_l3309_330912

/-- Given a function f and a real number a, prove properties of f --/
theorem function_properties (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 * x + 1) = 3 * a^(x + 1) - 4) 
  (h2 : a > 0) 
  (h3 : a ≠ 1) :
  (∀ x, f x = 3 * a^((x + 1) / 2) - 4) ∧ 
  (f (-1) = -1) ∧
  (a > 1 → ∀ x, f (x - 3/4) ≥ 3 / a^(x^2 / 2) - 4) := by
  sorry

end function_properties_l3309_330912


namespace cos_A_minus_sin_C_range_l3309_330953

theorem cos_A_minus_sin_C_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧          -- Sum of angles in a triangle
  a = 2 * b * Real.sin A → -- Given condition
  -Real.sqrt 3 / 2 < Real.cos A - Real.sin C ∧ 
  Real.cos A - Real.sin C < 1 / 2 := by
sorry

end cos_A_minus_sin_C_range_l3309_330953


namespace ordering_of_exponential_and_logarithm_l3309_330904

/-- Given a = e^0.1 - 1, b = 0.1, and c = ln 1.1, prove that a > b > c -/
theorem ordering_of_exponential_and_logarithm :
  let a := Real.exp 0.1 - 1
  let b := 0.1
  let c := Real.log 1.1
  a > b ∧ b > c := by sorry

end ordering_of_exponential_and_logarithm_l3309_330904


namespace sufficient_but_not_necessary_l3309_330992

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end sufficient_but_not_necessary_l3309_330992


namespace midpoint_coordinate_sum_l3309_330911

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, -3) and (-4, 15) is 8 -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := -3
  let x₂ : ℝ := -4
  let y₂ : ℝ := 15
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 8 := by
  sorry

end midpoint_coordinate_sum_l3309_330911


namespace largest_n_with_unique_k_l3309_330933

theorem largest_n_with_unique_k : ∃ (k : ℤ), 
  (5 : ℚ)/11 < (359 : ℚ)/(359 + k) ∧ (359 : ℚ)/(359 + k) < (6 : ℚ)/11 ∧
  (∀ (n : ℕ) (k₁ k₂ : ℤ), n > 359 →
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₁) ∧ (n : ℚ)/(n + k₁) < (6 : ℚ)/11) ∧
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₂) ∧ (n : ℚ)/(n + k₂) < (6 : ℚ)/11) →
    k₁ = k₂) →
  (∃ (k₁ k₂ : ℤ), k₁ ≠ k₂ ∧
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₁) ∧ (n : ℚ)/(n + k₁) < (6 : ℚ)/11) ∧
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₂) ∧ (n : ℚ)/(n + k₂) < (6 : ℚ)/11)) :=
by sorry

end largest_n_with_unique_k_l3309_330933


namespace steven_more_peaches_l3309_330965

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 17

/-- The number of apples Steven has -/
def steven_apples : ℕ := 16

/-- Jake has 6 fewer peaches than Steven -/
def jake_peaches : ℕ := steven_peaches - 6

/-- Jake has 8 more apples than Steven -/
def jake_apples : ℕ := steven_apples + 8

/-- Theorem: Steven has 1 more peach than apples -/
theorem steven_more_peaches : steven_peaches - steven_apples = 1 := by
  sorry

end steven_more_peaches_l3309_330965


namespace total_tape_theorem_l3309_330961

/-- The amount of tape needed for a rectangular box -/
def tape_for_rect_box (length width : ℕ) : ℕ := 2 * width + length

/-- The amount of tape needed for a square box -/
def tape_for_square_box (side : ℕ) : ℕ := 3 * side

/-- The total amount of tape needed for multiple boxes -/
def total_tape_needed (rect_boxes square_boxes : ℕ) (rect_length rect_width square_side : ℕ) : ℕ :=
  rect_boxes * tape_for_rect_box rect_length rect_width +
  square_boxes * tape_for_square_box square_side

theorem total_tape_theorem :
  total_tape_needed 5 2 30 15 40 = 540 :=
by sorry

end total_tape_theorem_l3309_330961


namespace two_out_of_three_win_probability_l3309_330917

/-- The probability that exactly two out of three players win a game, given their individual probabilities of success. -/
theorem two_out_of_three_win_probability
  (p_alice : ℚ) (p_benjamin : ℚ) (p_carol : ℚ)
  (h_alice : p_alice = 1/5)
  (h_benjamin : p_benjamin = 3/8)
  (h_carol : p_carol = 2/7) :
  (p_alice * p_benjamin * (1 - p_carol)) +
  (p_alice * p_carol * (1 - p_benjamin)) +
  (p_benjamin * p_carol * (1 - p_alice)) = 49/280 := by
  sorry

end two_out_of_three_win_probability_l3309_330917


namespace floor_of_4_7_l3309_330957

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end floor_of_4_7_l3309_330957


namespace marie_age_proof_l3309_330997

/-- Marie's age in years -/
def marie_age : ℚ := 8/3

/-- Liam's age in years -/
def liam_age : ℚ := 4 * marie_age

/-- Oliver's age in years -/
def oliver_age : ℚ := marie_age + 8

theorem marie_age_proof :
  (liam_age = 4 * marie_age) ∧
  (oliver_age = marie_age + 8) ∧
  (liam_age = oliver_age) →
  marie_age = 8/3 := by
  sorry

end marie_age_proof_l3309_330997


namespace complex_equation_real_part_l3309_330938

-- Define complex number z as a + bi
def z (a b : ℝ) : ℂ := Complex.mk a b

-- State the theorem
theorem complex_equation_real_part 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : z a b ^ 3 + 2 * z a b ^ 2 * Complex.I - 2 * z a b * Complex.I - 8 = 1624 * Complex.I) : 
  a ^ 3 - 3 * a * b ^ 2 - 8 = 0 :=
by sorry

end complex_equation_real_part_l3309_330938


namespace truth_probability_l3309_330964

theorem truth_probability (pA pB pAB : ℝ) : 
  pA = 0.7 →
  pAB = 0.42 →
  pAB = pA * pB →
  pB = 0.6 :=
by
  sorry

end truth_probability_l3309_330964


namespace closest_to_neg_sqrt_two_l3309_330972

theorem closest_to_neg_sqrt_two :
  let options : List ℝ := [-2, -1, 0, 1]
  ∀ x ∈ options, |(-1) - (-Real.sqrt 2)| ≤ |x - (-Real.sqrt 2)| :=
by
  sorry

end closest_to_neg_sqrt_two_l3309_330972


namespace vector_perpendicular_condition_l3309_330962

/-- Given two vectors m and n in ℝ², if m + n is perpendicular to m, then the second component of n is -3. -/
theorem vector_perpendicular_condition (m n : ℝ × ℝ) :
  m = (1, 2) →
  n.1 = a →
  n.2 = -1 →
  (m + n) • m = 0 →
  a = -3 := by
  sorry

end vector_perpendicular_condition_l3309_330962


namespace f_value_at_2_l3309_330989

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
    (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end f_value_at_2_l3309_330989


namespace set_inclusion_iff_a_range_l3309_330925

/-- The set A -/
def A : Set ℝ := {x | -2 < x ∧ x < 3}

/-- The set B -/
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}

/-- The set C parameterized by a -/
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

/-- The main theorem stating the equivalence between C being a subset of (A ∩ ℝ\B) and the range of a -/
theorem set_inclusion_iff_a_range :
  ∀ a : ℝ, (C a ⊆ (A ∩ (Set.univ \ B))) ↔ (0 < a ∧ a ≤ 2/3) :=
sorry

end set_inclusion_iff_a_range_l3309_330925


namespace largest_n_value_l3309_330922

/-- Represents a digit in base 5 -/
def Base5Digit := Fin 5

/-- Represents a digit in base 9 -/
def Base9Digit := Fin 9

/-- Converts a three-digit number in base 5 to base 10 -/
def base5ToBase10 (x y z : Base5Digit) : ℕ :=
  25 * x.val + 5 * y.val + z.val

/-- Converts a three-digit number in base 9 to base 10 -/
def base9ToBase10 (z y x : Base9Digit) : ℕ :=
  81 * z.val + 9 * y.val + x.val

theorem largest_n_value (n : ℕ) 
  (h1 : ∃ (x y z : Base5Digit), n = base5ToBase10 x y z)
  (h2 : ∃ (x y z : Base9Digit), n = base9ToBase10 z y x) :
  n ≤ 121 ∧ ∃ (x y z : Base5Digit), 121 = base5ToBase10 x y z ∧ 
    ∃ (x y z : Base9Digit), 121 = base9ToBase10 z y x :=
by sorry

end largest_n_value_l3309_330922


namespace rectangle_length_l3309_330901

/-- Proves that a rectangle with given perimeter-to-breadth ratio and area has a specific length -/
theorem rectangle_length (P b l A : ℝ) : 
  P / b = 5 → 
  P = 2 * (l + b) → 
  A = l * b → 
  A = 216 → 
  l = 18 := by sorry

end rectangle_length_l3309_330901


namespace student_count_l3309_330973

theorem student_count : ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 4 = 1 ∧ n = 45 := by
  sorry

end student_count_l3309_330973


namespace inequality_never_satisfied_l3309_330905

theorem inequality_never_satisfied (m : ℝ) :
  (∀ x : ℝ, ¬(|x - 4| + |3 - x| < m)) → m ≤ 1 := by
  sorry

end inequality_never_satisfied_l3309_330905


namespace fraction_simplification_l3309_330963

theorem fraction_simplification (x y : ℚ) (hx : x = 4/6) (hy : y = 8/12) :
  (6*x + 8*y) / (48*x*y) = 7/16 := by
  sorry

end fraction_simplification_l3309_330963


namespace complex_number_properties_l3309_330909

/-- Given a complex number z = (a+i)(1-i)+bi where a and b are real, and the point
    corresponding to z in the complex plane lies on the graph of y = x - 3 -/
theorem complex_number_properties (a b : ℝ) (z : ℂ) 
  (h1 : z = (a + Complex.I) * (1 - Complex.I) + b * Complex.I)
  (h2 : z.im = z.re - 3) : 
  (2 * a > b) ∧ (Complex.abs z ≥ 3 * Real.sqrt 2 / 2) := by
  sorry

end complex_number_properties_l3309_330909


namespace unique_solution_system_l3309_330929

/-- The system of equations has only one real solution (0, 0, 0, 0) -/
theorem unique_solution_system :
  ∃! (x y z w : ℝ),
    x = z + w + Real.sqrt (z * w * x) ∧
    y = w + x + Real.sqrt (w * x * y) ∧
    z = x + y + Real.sqrt (x * y * z) ∧
    w = y + z + Real.sqrt (y * z * w) :=
by sorry

end unique_solution_system_l3309_330929


namespace smallest_prime_angle_in_right_triangle_l3309_330999

/-- Checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Theorem: Smallest angle b in a right triangle with prime angles -/
theorem smallest_prime_angle_in_right_triangle :
  ∀ a b : ℕ,
  (a : ℝ) + (b : ℝ) = 90 →
  isPrime a →
  isPrime b →
  (a : ℝ) > (b : ℝ) + 2 →
  b ≥ 7 :=
sorry

end smallest_prime_angle_in_right_triangle_l3309_330999


namespace function_f_theorem_l3309_330902

/-- A function f: ℝ → ℝ satisfying the given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∃ (S : Finset ℝ), ∀ x ≠ 0, ∃ c ∈ S, f x = c * x) ∧
  (∀ x, f (x - 1 - f x) = f x - 1 - x)

/-- The theorem stating that f(x) = x or f(x) = -x -/
theorem function_f_theorem (f : ℝ → ℝ) (h : FunctionF f) :
  (∀ x, f x = x) ∨ (∀ x, f x = -x) := by
  sorry

end function_f_theorem_l3309_330902


namespace largest_three_digit_multiple_of_six_l3309_330919

theorem largest_three_digit_multiple_of_six :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 6 = 0 → n ≤ 996 :=
sorry

end largest_three_digit_multiple_of_six_l3309_330919


namespace sequence_equality_l3309_330935

/-- Given a sequence a₀, a₁, a₂, ..., prove that aₙ = 10ⁿ for all natural numbers n,
    if the following equation holds for all real t:
    ∑_{n=0}^∞ aₙ * t^n / n! = (∑_{n=0}^∞ 2^n * t^n / n!)² * (∑_{n=0}^∞ 3^n * t^n / n!)² -/
theorem sequence_equality (a : ℕ → ℝ) :
  (∀ t : ℝ, ∑' n, a n * t^n / n.factorial = (∑' n, 2^n * t^n / n.factorial)^2 * (∑' n, 3^n * t^n / n.factorial)^2) →
  ∀ n : ℕ, a n = 10^n :=
by sorry

end sequence_equality_l3309_330935


namespace towels_given_to_mother_l3309_330920

theorem towels_given_to_mother (green_towels white_towels remaining_towels : ℕ) : 
  green_towels = 35 → 
  white_towels = 21 → 
  remaining_towels = 22 → 
  green_towels + white_towels - remaining_towels = 34 :=
by
  sorry

end towels_given_to_mother_l3309_330920


namespace parabola_directrix_coefficient_l3309_330946

/-- For a parabola with equation y = ax² and directrix y = 2, prove that a = -1/8 -/
theorem parabola_directrix_coefficient (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Parabola equation
  (∃ y : ℝ, y = 2 ∧ ∀ x : ℝ, y = -1 / (4 * a)) →  -- Directrix equation
  a = -1/8 := by
sorry

end parabola_directrix_coefficient_l3309_330946


namespace math_books_count_l3309_330983

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) :
  total_books = 90 →
  math_cost = 4 →
  history_cost = 5 →
  total_price = 396 →
  ∃ (math_books : ℕ), 
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧ 
    math_books = 54 := by
  sorry

end math_books_count_l3309_330983


namespace distance_2_5_distance_neg2_5_distance_x_3_solutions_abs_x_minus_1_int_solutions_sum_distances_min_value_sum_distances_l3309_330980

-- Define the distance function
def distance (a b : ℚ) : ℚ := |a - b|

-- Theorem 1: Distance between 2 and 5 is 3
theorem distance_2_5 : distance 2 5 = 3 := by sorry

-- Theorem 2: Distance between -2 and 5 is 7
theorem distance_neg2_5 : distance (-2) 5 = 7 := by sorry

-- Theorem 3: |x-3| represents the distance between x and 3
theorem distance_x_3 (x : ℚ) : |x - 3| = distance x 3 := by sorry

-- Theorem 4: Solutions of |x-1| = 3
theorem solutions_abs_x_minus_1 (x : ℚ) : |x - 1| = 3 ↔ x = 4 ∨ x = -2 := by sorry

-- Theorem 5: Integer solutions of |x-1| + |x+2| = 3
theorem int_solutions_sum_distances (x : ℤ) : 
  |x - 1| + |x + 2| = 3 ↔ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 := by sorry

-- Theorem 6: Minimum value of |x+8| + |x-3| + |x-6|
theorem min_value_sum_distances :
  ∃ (x : ℚ), ∀ (y : ℚ), |x + 8| + |x - 3| + |x - 6| ≤ |y + 8| + |y - 3| + |y - 6| ∧
  |x + 8| + |x - 3| + |x - 6| = 14 := by sorry

end distance_2_5_distance_neg2_5_distance_x_3_solutions_abs_x_minus_1_int_solutions_sum_distances_min_value_sum_distances_l3309_330980


namespace prime_pair_equation_solution_l3309_330956

theorem prime_pair_equation_solution :
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    p^3 - q^5 = (p + q)^2 → 
    (p = 7 ∧ q = 3) := by
  sorry

end prime_pair_equation_solution_l3309_330956


namespace x_percent_of_z_l3309_330948

-- Define the variables
variable (x y z : ℝ)

-- Define the conditions
def condition1 : Prop := x = 1.20 * y
def condition2 : Prop := y = 0.40 * z

-- State the theorem
theorem x_percent_of_z (h1 : condition1 x y) (h2 : condition2 y z) : x = 0.48 * z := by
  sorry

end x_percent_of_z_l3309_330948


namespace only_cat_owners_count_l3309_330950

/-- The number of people owning only cats in a pet ownership scenario. -/
def num_only_cat_owners : ℕ := 
  let total_pet_owners : ℕ := 59
  let only_dog_owners : ℕ := 15
  let cat_and_dog_owners : ℕ := 5
  let cat_dog_snake_owners : ℕ := 3
  total_pet_owners - (only_dog_owners + cat_and_dog_owners + cat_dog_snake_owners)

/-- Theorem stating that the number of people owning only cats is 36. -/
theorem only_cat_owners_count : num_only_cat_owners = 36 := by
  sorry

end only_cat_owners_count_l3309_330950


namespace angle_215_in_third_quadrant_l3309_330988

def angle_in_third_quadrant (angle : ℝ) : Prop :=
  180 < angle ∧ angle ≤ 270

theorem angle_215_in_third_quadrant :
  angle_in_third_quadrant 215 :=
sorry

end angle_215_in_third_quadrant_l3309_330988


namespace checker_arrangement_count_l3309_330974

/-- The number of ways to arrange white and black checkers on a chessboard -/
def checker_arrangements : ℕ := 
  let total_squares : ℕ := 32
  let white_checkers : ℕ := 12
  let black_checkers : ℕ := 12
  Nat.factorial total_squares / (Nat.factorial white_checkers * Nat.factorial black_checkers * Nat.factorial (total_squares - white_checkers - black_checkers))

/-- Theorem stating that the number of ways to arrange 12 white and 12 black checkers
    on 32 black squares of a chessboard is equal to (32! / (12! * 12! * 8!)) -/
theorem checker_arrangement_count : 
  checker_arrangements = Nat.factorial 32 / (Nat.factorial 12 * Nat.factorial 12 * Nat.factorial 8) :=
by sorry

end checker_arrangement_count_l3309_330974


namespace percentage_fraction_difference_l3309_330924

theorem percentage_fraction_difference : (75 / 100 * 40) - (4 / 5 * 25) = 10 := by
  sorry

end percentage_fraction_difference_l3309_330924


namespace female_democrats_count_l3309_330930

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) :
  total = 750 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = (total / 3 : ℚ) →
  female / 2 = 125 :=
by sorry

end female_democrats_count_l3309_330930


namespace parallel_lines_m_value_l3309_330923

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {a b c d : ℝ} : 
  (∀ x y : ℝ, a * x + b * y = 0 ↔ y = c * x + d) → b ≠ 0 → a / b = -c

/-- The value of m for which the lines 2x + my = 0 and y = 3x - 1 are parallel -/
theorem parallel_lines_m_value : 
  ∃ m : ℝ, (∀ x y : ℝ, 2 * x + m * y = 0 ↔ y = 3 * x - 1) ∧ m = -2/3 := by
  sorry

end parallel_lines_m_value_l3309_330923


namespace tank_capacity_is_21600_l3309_330969

/-- The capacity of a tank with specific inlet and outlet pipe properties -/
def tank_capacity : ℝ := by
  -- Define the time to empty the tank with only the outlet pipe open
  let outlet_time : ℝ := 10

  -- Define the inlet pipe rate in litres per minute
  let inlet_rate_per_minute : ℝ := 16

  -- Define the time to empty the tank with both pipes open
  let both_pipes_time : ℝ := 18

  -- Calculate the inlet rate in litres per hour
  let inlet_rate_per_hour : ℝ := inlet_rate_per_minute * 60

  -- The capacity of the tank
  exact 21600

/-- Theorem stating that the tank capacity is 21,600 litres -/
theorem tank_capacity_is_21600 : tank_capacity = 21600 := by
  sorry

end tank_capacity_is_21600_l3309_330969


namespace squares_below_specific_line_l3309_330968

/-- Represents a line in the coordinate plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of unit squares below a line in the first quadrant -/
def countSquaresBelowLine (l : Line) : ℕ :=
  sorry

theorem squares_below_specific_line :
  let l : Line := { a := 5, b := 195, c := 975 }
  countSquaresBelowLine l = 388 := by sorry

end squares_below_specific_line_l3309_330968


namespace abc_sum_888_l3309_330994

theorem abc_sum_888 : 
  ∃! (a b c : Nat), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    (100 * a + 10 * b + c) + (100 * a + 10 * b + c) + (100 * a + 10 * b + c) = 888 ∧
    100 * a + 10 * b + c = 296 :=
by sorry

end abc_sum_888_l3309_330994


namespace august_matches_l3309_330970

/-- Calculates the number of matches played in August given the initial and final winning percentages and the number of additional matches won. -/
def matches_in_august (initial_percentage : ℚ) (final_percentage : ℚ) (additional_wins : ℕ) : ℕ :=
  sorry

theorem august_matches :
  matches_in_august (22 / 100) (52 / 100) 75 = 120 :=
sorry

end august_matches_l3309_330970


namespace cubic_identity_l3309_330951

theorem cubic_identity (a b c : ℝ) : 
  (b + c - 2 * a)^3 + (c + a - 2 * b)^3 + (a + b - 2 * c)^3 = 
  (b + c - 2 * a) * (c + a - 2 * b) * (a + b - 2 * c) := by sorry

end cubic_identity_l3309_330951


namespace batsman_new_average_is_35_l3309_330914

/-- Represents a batsman's score history -/
structure Batsman where
  previousInnings : Nat
  previousTotalScore : Nat
  newInningScore : Nat
  averageIncrease : Nat

/-- Calculates the new average after the latest inning -/
def newAverage (b : Batsman) : Nat :=
  (b.previousTotalScore + b.newInningScore) / (b.previousInnings + 1)

/-- Theorem: Given the conditions, prove that the new average is 35 -/
theorem batsman_new_average_is_35 (b : Batsman)
  (h1 : b.previousInnings = 10)
  (h2 : b.newInningScore = 85)
  (h3 : b.averageIncrease = 5)
  (h4 : newAverage b = (b.previousTotalScore / b.previousInnings) + b.averageIncrease) :
  newAverage b = 35 := by
  sorry

#eval newAverage { previousInnings := 10, previousTotalScore := 300, newInningScore := 85, averageIncrease := 5 }

end batsman_new_average_is_35_l3309_330914


namespace distance_between_five_and_six_l3309_330916

/-- The distance to the nearest town in miles -/
def d : ℝ := sorry

/-- Alice's statement is false -/
axiom alice_false : ¬(d ≥ 6)

/-- Bob's statement is false -/
axiom bob_false : ¬(d ≤ 5)

/-- Charlie's statement is false -/
axiom charlie_false : ¬(d ≤ 4)

/-- Theorem: The distance to the nearest town is between 5 and 6 miles -/
theorem distance_between_five_and_six : 5 < d ∧ d < 6 := by sorry

end distance_between_five_and_six_l3309_330916


namespace centroid_property_l3309_330985

/-- Given a triangle PQR with vertices P(-2,4), Q(6,3), and R(2,-5),
    prove that if S(x,y) is the centroid of the triangle, then 7x + 3y = 16 -/
theorem centroid_property (P Q R S : ℝ × ℝ) (x y : ℝ) :
  P = (-2, 4) →
  Q = (6, 3) →
  R = (2, -5) →
  S = (x, y) →
  S = ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3) →
  7 * x + 3 * y = 16 := by
sorry

end centroid_property_l3309_330985


namespace evaluate_expression_l3309_330936

theorem evaluate_expression : 
  Real.sqrt ((16^6 + 8^8) / (16^3 + 8^9)) = (1 : ℝ) / 2 := by
  sorry

end evaluate_expression_l3309_330936


namespace intersection_of_A_and_B_l3309_330921

def A : Set ℝ := {Real.sin (90 * Real.pi / 180), Real.cos (180 * Real.pi / 180)}
def B : Set ℝ := {x : ℝ | x^2 + x = 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end intersection_of_A_and_B_l3309_330921


namespace chicken_rabbit_problem_l3309_330943

theorem chicken_rabbit_problem :
  ∀ (chickens rabbits : ℕ),
    chickens + rabbits = 100 →
    2 * chickens = 4 * rabbits + 26 →
    chickens = 71 :=
by
  sorry

end chicken_rabbit_problem_l3309_330943


namespace common_ratio_is_two_l3309_330932

/-- An increasing geometric sequence with specific conditions -/
def IncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q) ∧
  a 2 = 2 ∧
  a 4 - a 3 = 4

/-- The common ratio of the sequence is 2 -/
theorem common_ratio_is_two (a : ℕ → ℝ) (h : IncreasingGeometricSequence a) :
    ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 := by
  sorry

end common_ratio_is_two_l3309_330932


namespace vector_magnitude_l3309_330915

theorem vector_magnitude (a b : ℝ × ℝ) (m : ℝ) :
  a = (2, 1) →
  b = (3, m) →
  (∃ k : ℝ, (2 • a - b) = k • b) →
  ‖b‖ = (3 * Real.sqrt 5) / 2 :=
by sorry

end vector_magnitude_l3309_330915


namespace total_pens_l3309_330971

theorem total_pens (black_pens blue_pens : ℕ) : 
  black_pens = 4 → blue_pens = 4 → black_pens + blue_pens = 8 := by
  sorry

end total_pens_l3309_330971


namespace eulers_theorem_parallelepiped_l3309_330984

/-- Represents a parallelepiped with edges a, b, c meeting at a vertex,
    face diagonals d, e, f, and space diagonal g. -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  g : ℝ

/-- Euler's theorem for parallelepipeds:
    The sum of the squares of the edges and the space diagonal at one vertex
    is equal to the sum of the squares of the face diagonals. -/
theorem eulers_theorem_parallelepiped (p : Parallelepiped) :
  p.a^2 + p.b^2 + p.c^2 + p.g^2 = p.d^2 + p.e^2 + p.f^2 := by
  sorry

end eulers_theorem_parallelepiped_l3309_330984


namespace scaled_prism_volume_scaled_54_cubic_feet_prism_l3309_330918

/-- Theorem: Scaling a rectangular prism's volume -/
theorem scaled_prism_volume 
  (V : ℝ) 
  (a b c : ℝ) 
  (hV : V > 0) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a * b * c * V = (a * b * c) * V := by sorry

/-- Corollary: Specific case of scaling a 54 cubic feet prism -/
theorem scaled_54_cubic_feet_prism :
  let V : ℝ := 54
  let a : ℝ := 2
  let b : ℝ := 3
  let c : ℝ := 1.5
  a * b * c * V = 486 := by sorry

end scaled_prism_volume_scaled_54_cubic_feet_prism_l3309_330918


namespace basketball_cost_l3309_330987

/-- The cost of each basketball given the total cost and soccer ball cost -/
theorem basketball_cost (total_cost : ℕ) (soccer_cost : ℕ) : 
  total_cost = 920 ∧ soccer_cost = 65 → (total_cost - 8 * soccer_cost) / 5 = 80 := by
  sorry

#check basketball_cost

end basketball_cost_l3309_330987


namespace q_of_one_equals_five_l3309_330949

/-- Given a function q : ℝ → ℝ that passes through the point (1, 5), prove that q(1) = 5 -/
theorem q_of_one_equals_five (q : ℝ → ℝ) (h : q 1 = 5) : q 1 = 5 := by
  sorry

end q_of_one_equals_five_l3309_330949


namespace circle_area_ratio_l3309_330928

/-- Given two circles r and s, if the diameter of r is 50% of the diameter of s,
    then the area of r is 25% of the area of s. -/
theorem circle_area_ratio (r s : Real) (hr : r > 0) (hs : s > 0) 
  (h_diameter : 2 * r = 0.5 * (2 * s)) : 
  π * r^2 = 0.25 * (π * s^2) := by
  sorry

end circle_area_ratio_l3309_330928


namespace sector_area_l3309_330903

theorem sector_area (θ : Real) (L : Real) (A : Real) : 
  θ = π / 6 → 
  L = 2 * π / 3 → 
  A = 4 * π / 3 → 
  ∃ (r : Real), 
    L = r * θ ∧ 
    A = 1 / 2 * r^2 * θ := by
  sorry

end sector_area_l3309_330903


namespace sum_of_roots_equals_eight_pi_thirds_l3309_330937

noncomputable def f (x : Real) : Real := Real.sqrt 3 * Real.sin x + Real.cos x

theorem sum_of_roots_equals_eight_pi_thirds (a : Real) :
  0 < a → a < 1 → ∃ x₁ x₂ : Real, 
    x₁ ∈ Set.Icc 0 (2 * Real.pi) ∧ 
    x₂ ∈ Set.Icc 0 (2 * Real.pi) ∧ 
    f x₁ = a ∧ 
    f x₂ = a ∧ 
    x₁ + x₂ = 8 * Real.pi / 3 :=
sorry

end sum_of_roots_equals_eight_pi_thirds_l3309_330937


namespace circle_center_sum_l3309_330940

theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 10*x - 12*y + 40) → 
  ((x - 5)^2 + (y + 6)^2 = 101) → 
  x + y = -1 := by
sorry

end circle_center_sum_l3309_330940


namespace max_colors_upper_bound_l3309_330954

/-- 
Given a positive integer n ≥ 2, an n × n × n cube is divided into n³ unit cubes, 
each colored with one color. For each n × n × 1 rectangular prism (in 3 orientations), 
consider the set of colors appearing in this prism. For any color set in one group, 
it also appears in each of the other two groups. 
This theorem states the upper bound for the maximum number of colors.
-/
theorem max_colors_upper_bound (n : ℕ) (h : n ≥ 2) : 
  ∃ C : ℕ, C ≤ n * (n + 1) * (2 * n + 1) / 6 ∧ 
  (∀ D : ℕ, D ≤ n * (n + 1) * (2 * n + 1) / 6) :=
by sorry

end max_colors_upper_bound_l3309_330954


namespace remaining_money_after_ticket_l3309_330913

def octal_to_decimal (n : ℕ) : ℕ := sorry

theorem remaining_money_after_ticket : 
  let savings := octal_to_decimal 5376
  let ticket_cost := 1200
  savings - ticket_cost = 1614 := by
  sorry

end remaining_money_after_ticket_l3309_330913


namespace coefficient_d_nonzero_l3309_330991

/-- A polynomial of degree 5 with specific properties -/
def Q (a b c d f : ℝ) (x : ℝ) : ℝ :=
  x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + f

/-- Theorem: For a polynomial Q with five distinct x-intercepts, including (0,0) and (1,0), 
    the coefficient d must be non-zero -/
theorem coefficient_d_nonzero 
  (a b c d f : ℝ) 
  (h1 : Q a b c d f 0 = 0)
  (h2 : Q a b c d f 1 = 0)
  (h3 : ∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ 
       p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ p ≠ 1 ∧ q ≠ 1 ∧ r ≠ 1 ∧
       ∀ x : ℝ, Q a b c d f x = x * (x - 1) * (x - p) * (x - q) * (x - r)) : 
  d ≠ 0 := by
sorry

end coefficient_d_nonzero_l3309_330991


namespace value_of_expression_l3309_330990

theorem value_of_expression : 8 + 2 * (3^2) = 26 := by
  sorry

end value_of_expression_l3309_330990


namespace robins_gum_pieces_l3309_330939

/-- 
Given that Robin had an initial number of gum pieces, her brother gave her 26 more,
and now she has 44 pieces in total, prove that she initially had 18 pieces.
-/
theorem robins_gum_pieces (x : ℕ) : x + 26 = 44 → x = 18 := by
  sorry

end robins_gum_pieces_l3309_330939


namespace average_speed_calculation_l3309_330944

/-- Given a journey of 234 miles that takes 27/4 hours, prove that the average speed is 936/27 miles per hour. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 234) (h2 : time = 27/4) :
  distance / time = 936 / 27 := by
  sorry

end average_speed_calculation_l3309_330944


namespace ascendant_function_theorem_l3309_330995

/-- A function is ascendant if it is non-decreasing --/
def Ascendant (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem ascendant_function_theorem (f : ℝ → ℝ) 
  (h1 : Ascendant (fun x => f x - 3 * x))
  (h2 : Ascendant (fun x => f x - x^3)) :
  Ascendant (fun x => f x - x^2 - x) :=
sorry

end ascendant_function_theorem_l3309_330995


namespace inscribed_semicircle_radius_in_reflected_triangle_l3309_330908

/-- Represents a rectangle formed by reflecting an isosceles triangle over its base -/
structure ReflectedTriangleRectangle where
  base : ℝ
  height : ℝ
  inscribed_semicircle_radius : ℝ

/-- The theorem stating the radius of the inscribed semicircle in the specific rectangle -/
theorem inscribed_semicircle_radius_in_reflected_triangle
  (rect : ReflectedTriangleRectangle)
  (h_base : rect.base = 24)
  (h_height : rect.height = 10) :
  rect.inscribed_semicircle_radius = 60 / 11 := by
  sorry

end inscribed_semicircle_radius_in_reflected_triangle_l3309_330908


namespace gig_song_ratio_l3309_330981

/-- Proves that the ratio of the length of the last song to the length of the first two songs is 3:1 --/
theorem gig_song_ratio :
  let days_in_two_weeks : ℕ := 14
  let gigs_in_two_weeks : ℕ := days_in_two_weeks / 2
  let songs_per_gig : ℕ := 3
  let length_of_first_two_songs : ℕ := 2 * 5
  let total_playing_time : ℕ := 280
  let total_length_first_two_songs : ℕ := gigs_in_two_weeks * length_of_first_two_songs
  let total_length_third_song : ℕ := total_playing_time - total_length_first_two_songs
  let length_third_song_per_gig : ℕ := total_length_third_song / gigs_in_two_weeks
  length_third_song_per_gig / length_of_first_two_songs = 3 := by
  sorry

end gig_song_ratio_l3309_330981


namespace ratio_q_p_l3309_330942

def total_slips : ℕ := 60
def num_range : Set ℕ := Finset.range 10
def slips_per_num : ℕ := 6
def drawn_slips : ℕ := 4

def p : ℚ := (10 : ℚ) / Nat.choose total_slips drawn_slips
def q : ℚ := (5400 : ℚ) / Nat.choose total_slips drawn_slips

theorem ratio_q_p : q / p = 540 := by sorry

end ratio_q_p_l3309_330942


namespace abs_negative_2023_l3309_330955

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_negative_2023_l3309_330955


namespace power_fraction_simplification_l3309_330993

theorem power_fraction_simplification :
  (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by
sorry

end power_fraction_simplification_l3309_330993


namespace exam_maximum_marks_l3309_330927

/-- The maximum marks for an exam where:
  1. The passing mark is 33% of the maximum marks
  2. A student who got 175 marks failed by 56 marks
-/
theorem exam_maximum_marks : ∃ (M : ℕ), 
  (M * 33 / 100 = 175 + 56) ∧ 
  M = 700 := by
  sorry

end exam_maximum_marks_l3309_330927


namespace area_of_specific_region_l3309_330967

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bound by two circles and the x-axis -/
def areaOfRegion (c1 c2 : Circle) : ℝ :=
  sorry

theorem area_of_specific_region :
  let c1 : Circle := { center := (5, 5), radius := 5 }
  let c2 : Circle := { center := (10, 5), radius := 3 }
  areaOfRegion c1 c2 = 25 - 17 * Real.pi := by sorry

end area_of_specific_region_l3309_330967


namespace sqrt_equation_solution_l3309_330945

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 3) = 7 → x = 46 := by
  sorry

end sqrt_equation_solution_l3309_330945


namespace blood_expiry_time_l3309_330934

/-- Represents time as hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a date -/
structure Date where
  month : ℕ
  day : ℕ
  year : ℕ

/-- Represents a datetime -/
structure DateTime where
  date : Date
  time : Time

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def addSeconds (dt : DateTime) (seconds : ℕ) : DateTime :=
  sorry -- Implementation not required for the statement

theorem blood_expiry_time 
  (donation_time : DateTime)
  (expiry_seconds : ℕ)
  (h_donation_time : donation_time = ⟨⟨1, 1, 2023⟩, ⟨8, 0, sorry, sorry⟩⟩)
  (h_expiry_seconds : expiry_seconds = factorial 8) :
  addSeconds donation_time expiry_seconds = ⟨⟨1, 1, 2023⟩, ⟨19, 12, sorry, sorry⟩⟩ :=
sorry

end blood_expiry_time_l3309_330934


namespace symmetry_implies_a_value_l3309_330906

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are the same. -/
def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetry_implies_a_value :
  ∀ a : ℝ, symmetric_y_axis (a, 1) (-3, 1) → a = 3 :=
by sorry

end symmetry_implies_a_value_l3309_330906


namespace car_speed_second_hour_l3309_330900

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 98)
  (h2 : average_speed = 79) :
  let speed_second_hour := 2 * average_speed - speed_first_hour
  speed_second_hour = 60 := by
  sorry

end car_speed_second_hour_l3309_330900


namespace least_addition_for_divisibility_l3309_330982

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), (1056 + n) % 25 = 0 ∧ 
  ∀ (m : ℕ), m < n → (1056 + m) % 25 ≠ 0 :=
by
  use 19
  sorry

end least_addition_for_divisibility_l3309_330982


namespace kennel_problem_l3309_330976

theorem kennel_problem (total : ℕ) (long_fur : ℕ) (brown : ℕ) (neither : ℕ) 
  (h_total : total = 45)
  (h_long_fur : long_fur = 26)
  (h_brown : brown = 22)
  (h_neither : neither = 8) :
  long_fur + brown - (total - neither) = 11 :=
by sorry

end kennel_problem_l3309_330976


namespace cos_squared_minus_sin_squared_15_deg_l3309_330931

theorem cos_squared_minus_sin_squared_15_deg :
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end cos_squared_minus_sin_squared_15_deg_l3309_330931


namespace greatest_integer_gcd_eighteen_l3309_330975

theorem greatest_integer_gcd_eighteen : ∃ n : ℕ, n < 200 ∧ n.gcd 18 = 6 ∧ ∀ m : ℕ, m < 200 → m.gcd 18 = 6 → m ≤ n := by
  sorry

end greatest_integer_gcd_eighteen_l3309_330975


namespace f_formula_g_minimum_l3309_330926

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 7*x + 13

-- Define the function g
def g (a x : ℝ) : ℝ := f (x + a) - 7*x

-- Theorem for part (I)
theorem f_formula (x : ℝ) : f (2*x - 3) = 4*x^2 + 2*x + 1 := by sorry

-- Theorem for part (II)
theorem g_minimum (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, g a x ≥ 
    if a ≤ -3 then a^2 + 13*a + 22
    else if a < -1 then 7*a + 13
    else a^2 + 9*a + 14) ∧
  (∃ x ∈ Set.Icc 1 3, g a x = 
    if a ≤ -3 then a^2 + 13*a + 22
    else if a < -1 then 7*a + 13
    else a^2 + 9*a + 14) := by sorry

end f_formula_g_minimum_l3309_330926


namespace min_value_a_b_squared_l3309_330958

/-- Given that the ratio of the absolute values of the coefficients of x² and x³ terms
    in the expansion of (1/a + ax)⁵ - (1/b + bx)⁵ is 1:6, 
    the minimum value of a² + b² is 12 -/
theorem min_value_a_b_squared (a b : ℝ) (h : ∃ k : ℝ, k > 0 ∧ 
  |5 * (1/a^2 - 1/b^2)| = k ∧ |10 * (a - b)| = 6*k) : 
  a^2 + b^2 ≥ 12 := by
  sorry

end min_value_a_b_squared_l3309_330958


namespace volume_of_inscribed_sphere_l3309_330941

/-- The volume of a sphere inscribed in a cube -/
theorem volume_of_inscribed_sphere (cube_edge : ℝ) (sphere_volume : ℝ) : 
  cube_edge = 10 →
  sphere_volume = (4 / 3) * π * (cube_edge / 2)^3 →
  sphere_volume = (500 / 3) * π :=
by sorry

end volume_of_inscribed_sphere_l3309_330941


namespace ceiling_floor_sum_zero_l3309_330978

theorem ceiling_floor_sum_zero : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end ceiling_floor_sum_zero_l3309_330978


namespace space_filling_tetrahedrons_octahedrons_l3309_330996

/-- A regular tetrahedron -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- A regular octahedron -/
structure RegularOctahedron :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- A space-filling arrangement -/
structure SpaceFillingArrangement :=
  (tetrahedrons : Set RegularTetrahedron)
  (octahedrons : Set RegularOctahedron)

/-- No gaps or overlaps in the arrangement -/
def NoGapsOrOverlaps (arrangement : SpaceFillingArrangement) : Prop :=
  sorry

/-- All polyhedra in the arrangement are congruent and have equal edge lengths -/
def CongruentWithEqualEdges (arrangement : SpaceFillingArrangement) : Prop :=
  sorry

/-- The main theorem: There exists a space-filling arrangement of congruent regular tetrahedrons
    and regular octahedrons with equal edge lengths, without gaps or overlaps -/
theorem space_filling_tetrahedrons_octahedrons :
  ∃ (arrangement : SpaceFillingArrangement),
    CongruentWithEqualEdges arrangement ∧ NoGapsOrOverlaps arrangement :=
sorry

end space_filling_tetrahedrons_octahedrons_l3309_330996


namespace saltwater_solution_volume_l3309_330947

/-- Proves that the initial volume of a saltwater solution is 100 gallons, given the conditions stated in the problem. -/
theorem saltwater_solution_volume : ∃ (x : ℝ), 
  -- Initial salt concentration is 20%
  (0.2 * x = x * 0.2) ∧ 
  -- After evaporation, total volume is 3/4 of initial
  (3/4 * x = x * 3/4) ∧ 
  -- Final salt concentration is 33 1/3%
  ((0.2 * x + 10) / (3/4 * x + 15) = 1/3) ∧ 
  -- Initial volume is 100 gallons
  (x = 100) := by
  sorry

end saltwater_solution_volume_l3309_330947


namespace line_equations_l3309_330959

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ

-- Define the point A
def A : Point := (1, -3)

-- Define the reference line
def reference_line : Line := λ x y ↦ 2*x - y + 4

-- Define the properties of lines l and m
def parallel (l1 l2 : Line) : Prop := ∃ k : ℝ, ∀ x y, l1 x y = k * l2 x y
def perpendicular (l1 l2 : Line) : Prop := ∃ k : ℝ, ∀ x y, l1 x y * l2 x y = -k

-- Define the y-intercept of a line
def y_intercept (l : Line) : ℝ := l 0 1

-- State the theorem
theorem line_equations (l m : Line) : 
  (∃ k : ℝ, l A.fst A.snd = 0) →  -- l passes through A
  parallel l reference_line →     -- l is parallel to reference_line
  perpendicular l m →             -- m is perpendicular to l
  y_intercept m = 3 →             -- m has y-intercept 3
  (∀ x y, l x y = 2*x - y - 5) ∧  -- equation of l
  (∀ x y, m x y = x + 2*y - 6)    -- equation of m
  := by sorry

end line_equations_l3309_330959


namespace quadratic_inequality_solution_l3309_330907

-- Define the quadratic function
def f (x : ℝ) := x^2 + 3*x - 4

-- Define the solution set
def solution_set : Set ℝ := {x | -4 < x ∧ x < 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set := by sorry

end quadratic_inequality_solution_l3309_330907


namespace parallelogram_product_l3309_330960

structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ
  x : ℝ
  z : ℝ
  h_EF : EF = 46
  h_FG : FG z = 4 * z^3 + 1
  h_GH : GH x = 3 * x + 6
  h_HE : HE = 35
  h_opposite_sides_equal : EF = GH x ∧ FG z = HE

theorem parallelogram_product (p : Parallelogram) :
  p.x * p.z = (40/3) * Real.rpow 8.5 (1/3) := by sorry

end parallelogram_product_l3309_330960


namespace root_sum_reciprocals_l3309_330952

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 + 6*p^3 + 11*p^2 + 6*p + 3 = 0) →
  (q^4 + 6*q^3 + 11*q^2 + 6*q + 3 = 0) →
  (r^4 + 6*r^3 + 11*r^2 + 6*r + 3 = 0) →
  (s^4 + 6*s^3 + 11*s^2 + 6*s + 3 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3 := by
sorry

end root_sum_reciprocals_l3309_330952


namespace farm_animals_difference_l3309_330979

theorem farm_animals_difference (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 6 * initial_cows →
  (initial_horses - 30) = 4 * (initial_cows + 30) →
  (initial_horses - 30) - (initial_cows + 30) = 315 := by
sorry

end farm_animals_difference_l3309_330979
