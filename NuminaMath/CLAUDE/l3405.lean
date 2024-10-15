import Mathlib

namespace NUMINAMATH_CALUDE_cube_paper_expenditure_l3405_340556

-- Define the parameters
def paper_cost_per_kg : ℚ := 60
def cube_edge_length : ℚ := 10
def area_covered_per_kg : ℚ := 20

-- Define the function to calculate the expenditure
def calculate_expenditure (edge_length area_per_kg cost_per_kg : ℚ) : ℚ :=
  6 * edge_length^2 / area_per_kg * cost_per_kg

-- State the theorem
theorem cube_paper_expenditure :
  calculate_expenditure cube_edge_length area_covered_per_kg paper_cost_per_kg = 1800 := by
  sorry

end NUMINAMATH_CALUDE_cube_paper_expenditure_l3405_340556


namespace NUMINAMATH_CALUDE_oil_drilling_probability_l3405_340571

/-- The probability of hitting an oil layer when drilling in a sea area -/
theorem oil_drilling_probability 
  (total_area : ℝ) 
  (oil_area : ℝ) 
  (h1 : total_area = 10000) 
  (h2 : oil_area = 40) : 
  oil_area / total_area = 1 / 250 := by
sorry

end NUMINAMATH_CALUDE_oil_drilling_probability_l3405_340571


namespace NUMINAMATH_CALUDE_smallest_marble_count_l3405_340574

theorem smallest_marble_count : ∃ m : ℕ, 
  m > 0 ∧ 
  m % 9 = 1 ∧ 
  m % 7 = 3 ∧ 
  (∀ n : ℕ, n > 0 ∧ n % 9 = 1 ∧ n % 7 = 3 → m ≤ n) ∧ 
  m = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l3405_340574


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3405_340583

-- Define the sets A, B, and C
def A (x : ℝ) : Set ℝ := {2, -1, x^2 - x + 1}
def B (x y : ℝ) : Set ℝ := {2*y, -4, x + 4}
def C : Set ℝ := {-1}

-- State the theorem
theorem union_of_A_and_B (x y : ℝ) :
  (A x ∩ B x y = C) →
  (A x ∪ B x y = {2, -1, x^2 - x + 1, 2*y, -4, x + 4}) :=
by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3405_340583


namespace NUMINAMATH_CALUDE_corina_calculation_l3405_340512

theorem corina_calculation (P Q : ℤ) 
  (h1 : P + Q = 16) 
  (h2 : P - Q = 4) : 
  P = 10 := by
sorry

end NUMINAMATH_CALUDE_corina_calculation_l3405_340512


namespace NUMINAMATH_CALUDE_intersection_M_N_l3405_340509

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | Real.log (2 * x + 1) > 0}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3405_340509


namespace NUMINAMATH_CALUDE_grade_10_sample_size_l3405_340576

/-- Represents the number of students to be selected from a stratum in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (stratum_size : ℕ) (sample_size : ℕ) : ℕ :=
  (stratum_size * sample_size) / total_population

/-- The problem statement -/
theorem grade_10_sample_size :
  stratified_sample_size 4500 1200 150 = 40 := by
  sorry


end NUMINAMATH_CALUDE_grade_10_sample_size_l3405_340576


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l3405_340521

theorem unique_two_digit_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10 ≠ n % 10) ∧ 
  n^2 = (n / 10 + n % 10)^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l3405_340521


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3405_340505

theorem complex_magnitude_product : 
  Complex.abs ((5 - 3*Complex.I) * (7 + 24*Complex.I)) = 25 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3405_340505


namespace NUMINAMATH_CALUDE_dog_food_calculation_l3405_340530

/-- Calculates the total amount of dog food needed per day for a given list of dog weights -/
def totalDogFood (weights : List ℕ) : ℕ :=
  (weights.map (· / 10)).sum

/-- Theorem: Given five dogs with specific weights, the total dog food needed is 15 pounds -/
theorem dog_food_calculation :
  totalDogFood [20, 40, 10, 30, 50] = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_calculation_l3405_340530


namespace NUMINAMATH_CALUDE_election_winner_votes_l3405_340562

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) :
  winner_percentage = 55 / 100 →
  vote_difference = 100 →
  (winner_percentage * total_votes).num = 
    (1 - winner_percentage) * total_votes + vote_difference →
  (winner_percentage * total_votes).num = 550 := by
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3405_340562


namespace NUMINAMATH_CALUDE_remainder_relationship_l3405_340545

theorem remainder_relationship (P P' D R R' C : ℕ) (h1 : P > P') (h2 : P % D = R) (h3 : P' % D = R') : 
  ∃ (s r : ℕ), ((P + C) * P') % D = s ∧ (P * P') % D = r ∧ 
  (∃ (C1 D1 : ℕ), s > r) ∧ (∃ (C2 D2 : ℕ), s < r) :=
sorry

end NUMINAMATH_CALUDE_remainder_relationship_l3405_340545


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l3405_340569

def g (x : ℝ) : ℝ := 20 * x^4 - 21 * x^2 + 5

theorem greatest_root_of_g :
  ∃ (r : ℝ), g r = 0 ∧ r = 1 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l3405_340569


namespace NUMINAMATH_CALUDE_f_increasing_condition_f_extremum_at_3_f_max_value_f_min_value_l3405_340549

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - 3*x

-- Part 1: f(x) is increasing on [1, +∞) iff a ≤ 0
theorem f_increasing_condition (a : ℝ) :
  (∀ x ≥ 1, Monotone (f a)) ↔ a ≤ 0 := by sorry

-- Part 2: When x = 3 is an extremum point
theorem f_extremum_at_3 (a : ℝ) :
  (∃ x, HasDerivAt (f a) 0 x) → a = 6 := by sorry

-- Maximum value of f(x) on [1, 6] is -6
theorem f_max_value :
  ∃ x ∈ Set.Icc 1 6, ∀ y ∈ Set.Icc 1 6, f 6 y ≤ f 6 x ∧ f 6 x = -6 := by sorry

-- Minimum value of f(x) on [1, 6] is -18
theorem f_min_value :
  ∃ x ∈ Set.Icc 1 6, ∀ y ∈ Set.Icc 1 6, f 6 x ≤ f 6 y ∧ f 6 x = -18 := by sorry

end NUMINAMATH_CALUDE_f_increasing_condition_f_extremum_at_3_f_max_value_f_min_value_l3405_340549


namespace NUMINAMATH_CALUDE_max_distance_to_point_l3405_340542

/-- The maximum distance from a point on the curve y = √(2 - x^2) to (0, -1) -/
theorem max_distance_to_point (x : ℝ) : 
  let y : ℝ := Real.sqrt (2 - x^2)
  let d : ℝ := Real.sqrt (x^2 + (y + 1)^2)
  d ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_point_l3405_340542


namespace NUMINAMATH_CALUDE_cookies_left_l3405_340557

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens John bought -/
def dozens_bought : ℕ := 2

/-- The number of cookies John ate -/
def cookies_eaten : ℕ := 3

/-- Theorem: John has 21 cookies left -/
theorem cookies_left : dozens_bought * dozen - cookies_eaten = 21 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l3405_340557


namespace NUMINAMATH_CALUDE_functional_inequality_solution_l3405_340559

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x > 0, f x > 0}

/-- The functional inequality condition -/
def SatisfiesInequality (f : PositiveRealFunction) : Prop :=
  ∀ x y, x > 0 → y > 0 → f.val (x * y) ≤ (x * f.val y + y * f.val x) / 2

/-- The theorem statement -/
theorem functional_inequality_solution :
  ∀ f : PositiveRealFunction, SatisfiesInequality f →
  ∃ a : ℝ, a > 0 ∧ ∀ x > 0, f.val x = a * x :=
sorry

end NUMINAMATH_CALUDE_functional_inequality_solution_l3405_340559


namespace NUMINAMATH_CALUDE_three_digit_sum_problem_l3405_340593

theorem three_digit_sum_problem (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  122 * a + 212 * b + 221 * c = 2003 →
  100 * a + 10 * b + c = 345 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_problem_l3405_340593


namespace NUMINAMATH_CALUDE_max_of_expression_l3405_340552

theorem max_of_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 50) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ 18.124 ∧
  (x = 16 → Real.sqrt (x + 50) + Real.sqrt (20 - x) + 2 * Real.sqrt x = 18.124) :=
by sorry

end NUMINAMATH_CALUDE_max_of_expression_l3405_340552


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3405_340561

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b = 5) :
  (1/a + 1/b) ≥ 5 + 2*Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3405_340561


namespace NUMINAMATH_CALUDE_m_range_l3405_340595

-- Define the plane region
def plane_region (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |2 * p.1 + p.2 + m| < 3}

-- Theorem statement
theorem m_range (m : ℝ) :
  ((0, 0) ∈ plane_region m) ∧
  ((-1, 1) ∈ plane_region m) ↔
  -2 < m ∧ m < 3 :=
by
  sorry


end NUMINAMATH_CALUDE_m_range_l3405_340595


namespace NUMINAMATH_CALUDE_even_sum_probability_l3405_340589

/-- Represents a spinner with its possible outcomes -/
structure Spinner :=
  (outcomes : List ℕ)

/-- The probability of getting an even sum from spinning all three spinners -/
def probability_even_sum (s t u : Spinner) : ℚ :=
  sorry

/-- The spinners as defined in the problem -/
def spinner_s : Spinner := ⟨[1, 2, 4]⟩
def spinner_t : Spinner := ⟨[3, 3, 6]⟩
def spinner_u : Spinner := ⟨[2, 4, 6]⟩

/-- The main theorem to prove -/
theorem even_sum_probability :
  probability_even_sum spinner_s spinner_t spinner_u = 5/9 :=
sorry

end NUMINAMATH_CALUDE_even_sum_probability_l3405_340589


namespace NUMINAMATH_CALUDE_constant_triangle_sum_l3405_340565

/-- Given a rectangle ABCD with width 'a' and height 'b', and a line 'r' parallel to AB
    intersecting diagonal AC at point (x₀, y₀), the sum of the areas of the two triangles
    formed by 'r' is constant and equal to (a*b)/2, regardless of the position of 'r'. -/
theorem constant_triangle_sum (a b x₀ : ℝ) (ha : a > 0) (hb : b > 0) (hx : 0 ≤ x₀ ∧ x₀ ≤ a) :
  let y₀ := (b / a) * x₀
  let area₁ := (1 / 2) * b * x₀
  let area₂ := (1 / 2) * b * (a - x₀)
  area₁ + area₂ = (a * b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_constant_triangle_sum_l3405_340565


namespace NUMINAMATH_CALUDE_glitched_clock_correct_time_fraction_l3405_340591

/-- Represents a 12-hour digital clock with a glitch where '2' is displayed as '7' -/
structure GlitchedClock where
  /-- The number of hours in the clock cycle -/
  hours : Nat
  /-- The number of minutes in an hour -/
  minutes_per_hour : Nat
  /-- The digit that is erroneously displayed -/
  glitched_digit : Nat
  /-- The digit that replaces the glitched digit -/
  replacement_digit : Nat

/-- The fraction of the day that the glitched clock shows the correct time -/
def correct_time_fraction (clock : GlitchedClock) : ℚ :=
  sorry

/-- Theorem stating that the fraction of correct time for the given clock is 55/72 -/
theorem glitched_clock_correct_time_fraction :
  let clock : GlitchedClock := {
    hours := 12,
    minutes_per_hour := 60,
    glitched_digit := 2,
    replacement_digit := 7
  }
  correct_time_fraction clock = 55 / 72 := by
  sorry

end NUMINAMATH_CALUDE_glitched_clock_correct_time_fraction_l3405_340591


namespace NUMINAMATH_CALUDE_divisibility_property_l3405_340551

theorem divisibility_property (m n : ℕ) (h : 24 ∣ (m * n + 1)) : 24 ∣ (m + n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3405_340551


namespace NUMINAMATH_CALUDE_pr_less_than_qr_implies_p_less_than_q_l3405_340567

theorem pr_less_than_qr_implies_p_less_than_q
  (r p q : ℝ) 
  (h1 : r < 0) 
  (h2 : p * q ≠ 0) 
  (h3 : p * r < q * r) : 
  p < q :=
by sorry

end NUMINAMATH_CALUDE_pr_less_than_qr_implies_p_less_than_q_l3405_340567


namespace NUMINAMATH_CALUDE_square_side_length_l3405_340537

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 8)
  (h2 : rectangle_length = 2)
  (h3 : rectangle_width > 0)
  (h4 : rectangle_length > 0) :
  ∃ (square_side : ℝ), 
    square_side > 0 ∧ 
    square_side * square_side = rectangle_width * rectangle_length ∧
    square_side = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3405_340537


namespace NUMINAMATH_CALUDE_customers_without_tip_l3405_340513

theorem customers_without_tip (initial_customers : ℕ) (additional_customers : ℕ) (customers_with_tip : ℕ) : 
  initial_customers = 39 → additional_customers = 12 → customers_with_tip = 2 →
  initial_customers + additional_customers - customers_with_tip = 49 := by
sorry

end NUMINAMATH_CALUDE_customers_without_tip_l3405_340513


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3405_340544

/-- Given two natural numbers m and n, returns true if m has a units digit of 9 -/
def hasUnitsDigitOf9 (m : ℕ) : Prop :=
  m % 10 = 9

/-- Given a natural number n, returns its units digit -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 31^6) (h2 : hasUnitsDigitOf9 m) :
  unitsDigit n = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3405_340544


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l3405_340511

theorem complex_magnitude_equality (n : ℝ) (hn : n > 0) :
  Complex.abs (4 + 2 * n * Complex.I) = 4 * Real.sqrt 5 ↔ n = 4 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l3405_340511


namespace NUMINAMATH_CALUDE_platform_length_l3405_340564

/-- Given a train and platform with specific properties, prove the length of the platform. -/
theorem platform_length (train_length : ℝ) (time_cross_platform : ℝ) (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 40)
  (h3 : time_cross_pole = 18) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 367 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l3405_340564


namespace NUMINAMATH_CALUDE_cube_difference_l3405_340566

theorem cube_difference (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 - b^3 = 992 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l3405_340566


namespace NUMINAMATH_CALUDE_sqrt_six_star_sqrt_six_l3405_340540

-- Define the ¤ operation
def star (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt_six_star_sqrt_six : star (Real.sqrt 6) (Real.sqrt 6) = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_star_sqrt_six_l3405_340540


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l3405_340547

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l3405_340547


namespace NUMINAMATH_CALUDE_girls_on_playground_l3405_340520

theorem girls_on_playground (total_children boys : ℕ) 
  (h1 : total_children = 117)
  (h2 : boys = 40) :
  total_children - boys = 77 := by
sorry

end NUMINAMATH_CALUDE_girls_on_playground_l3405_340520


namespace NUMINAMATH_CALUDE_sum_of_divisors_is_96_l3405_340539

-- Define the property of n having exactly 8 divisors, including 1, n, 14, and 21
def has_eight_divisors_with_14_and_21 (n : ℕ) : Prop :=
  (∃ d : Finset ℕ, d.card = 8 ∧ 
    (∀ x, x ∈ d ↔ x ∣ n) ∧
    1 ∈ d ∧ n ∈ d ∧ 14 ∈ d ∧ 21 ∈ d)

-- Theorem stating that if n satisfies the above property, 
-- then the sum of its divisors is 96
theorem sum_of_divisors_is_96 (n : ℕ) 
  (h : has_eight_divisors_with_14_and_21 n) : 
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_is_96_l3405_340539


namespace NUMINAMATH_CALUDE_no_solution_eq1_unique_solution_eq2_l3405_340506

-- Problem 1
theorem no_solution_eq1 : ¬∃ x : ℝ, (1 / (x - 2) + 3 = (1 - x) / (2 - x)) := by sorry

-- Problem 2
theorem unique_solution_eq2 : ∃! x : ℝ, (x / (x - 1) - 1 = 3 / (x^2 - 1)) ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_no_solution_eq1_unique_solution_eq2_l3405_340506


namespace NUMINAMATH_CALUDE_ring_stack_height_l3405_340516

/-- Represents a stack of linked rings -/
structure RingStack where
  top_diameter : ℝ
  bottom_diameter : ℝ
  ring_thickness : ℝ

/-- Calculates the total height of the ring stack -/
def stack_height (stack : RingStack) : ℝ :=
  sorry

/-- Theorem: The height of the given ring stack is 72 cm -/
theorem ring_stack_height :
  let stack := RingStack.mk 20 4 2
  stack_height stack = 72 := by
  sorry

end NUMINAMATH_CALUDE_ring_stack_height_l3405_340516


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l3405_340510

/-- The set of complex numbers z satisfying |z-i|+|z+i|=3 forms an ellipse in the complex plane -/
theorem trajectory_is_ellipse (z : ℂ) : 
  (Set.range fun (z : ℂ) => Complex.abs (z - Complex.I) + Complex.abs (z + Complex.I) = 3) 
  IsEllipse :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l3405_340510


namespace NUMINAMATH_CALUDE_melanie_dimes_problem_l3405_340555

/-- Calculates the number of dimes Melanie's mother gave her -/
def mothers_dimes (initial : ℤ) (given_to_dad : ℤ) (final : ℤ) : ℤ :=
  final - (initial - given_to_dad)

theorem melanie_dimes_problem : mothers_dimes 7 8 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_problem_l3405_340555


namespace NUMINAMATH_CALUDE_total_money_l3405_340524

theorem total_money (john alice bob : ℚ) (h1 : john = 5/8) (h2 : alice = 7/20) (h3 : bob = 1/4) :
  john + alice + bob = 1.225 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3405_340524


namespace NUMINAMATH_CALUDE_sanchez_rope_purchase_sanchez_rope_purchase_l3405_340527

theorem sanchez_rope_purchase (inches_per_foot : ℕ) (this_week_inches : ℕ) : ℕ :=
  let last_week_feet := (this_week_inches / inches_per_foot) + 4
  last_week_feet

#check sanchez_rope_purchase 12 96 = 12

/- Proof
theorem sanchez_rope_purchase (inches_per_foot : ℕ) (this_week_inches : ℕ) : ℕ :=
  let last_week_feet := (this_week_inches / inches_per_foot) + 4
  last_week_feet
sorry
-/

end NUMINAMATH_CALUDE_sanchez_rope_purchase_sanchez_rope_purchase_l3405_340527


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1717_l3405_340533

theorem largest_prime_factor_of_1717 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 1717 ∧ ∀ (q : ℕ), q.Prime → q ∣ 1717 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1717_l3405_340533


namespace NUMINAMATH_CALUDE_sophia_book_reading_l3405_340573

theorem sophia_book_reading (total_pages : ℕ) (pages_read : ℕ) :
  total_pages = 90 →
  pages_read = (total_pages - pages_read) + 30 →
  pages_read = (2 : ℚ) / 3 * total_pages :=
by
  sorry

end NUMINAMATH_CALUDE_sophia_book_reading_l3405_340573


namespace NUMINAMATH_CALUDE_remainder_theorem_l3405_340586

theorem remainder_theorem : (7 * 11^24 + 2^24) % 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3405_340586


namespace NUMINAMATH_CALUDE_janette_breakfast_jerky_l3405_340528

/-- The number of days Janette went camping -/
def camping_days : ℕ := 5

/-- The initial number of beef jerky pieces Janette brought -/
def initial_jerky : ℕ := 40

/-- The number of beef jerky pieces Janette eats for lunch each day -/
def lunch_jerky : ℕ := 1

/-- The number of beef jerky pieces Janette eats for dinner each day -/
def dinner_jerky : ℕ := 2

/-- The number of beef jerky pieces Janette has left after giving half to her brother -/
def final_jerky : ℕ := 10

/-- The number of beef jerky pieces Janette eats for breakfast each day -/
def breakfast_jerky : ℕ := 1

theorem janette_breakfast_jerky :
  breakfast_jerky = 1 ∧
  camping_days * (breakfast_jerky + lunch_jerky + dinner_jerky) = initial_jerky - 2 * final_jerky :=
by sorry

end NUMINAMATH_CALUDE_janette_breakfast_jerky_l3405_340528


namespace NUMINAMATH_CALUDE_break_even_components_min_profitable_components_l3405_340508

/-- The number of components produced and sold monthly -/
def components : ℕ := 150

/-- Production cost per component -/
def production_cost : ℚ := 80

/-- Shipping cost per component -/
def shipping_cost : ℚ := 5

/-- Fixed monthly costs -/
def fixed_costs : ℚ := 16500

/-- Minimum selling price per component -/
def selling_price : ℚ := 195

/-- Theorem stating that the number of components produced and sold monthly
    is the break-even point where costs equal revenues -/
theorem break_even_components :
  (selling_price * components : ℚ) = 
  fixed_costs + (production_cost + shipping_cost) * components := by
  sorry

/-- Theorem stating that the number of components is the minimum
    where revenues are not less than costs -/
theorem min_profitable_components :
  ∀ n : ℕ, n < components → 
  (selling_price * n : ℚ) < fixed_costs + (production_cost + shipping_cost) * n := by
  sorry

end NUMINAMATH_CALUDE_break_even_components_min_profitable_components_l3405_340508


namespace NUMINAMATH_CALUDE_B_power_93_l3405_340532

def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 0; 0, 0, -1; 0, 1, 0]

theorem B_power_93 : B^93 = B := by sorry

end NUMINAMATH_CALUDE_B_power_93_l3405_340532


namespace NUMINAMATH_CALUDE_circle_points_count_l3405_340596

def number_of_triangles (n : ℕ) : ℕ := n.choose 4

theorem circle_points_count : ∃ (n : ℕ), n > 3 ∧ number_of_triangles n = 126 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_count_l3405_340596


namespace NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l3405_340500

theorem fraction_division (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : c ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction_division :
  (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l3405_340500


namespace NUMINAMATH_CALUDE_total_tea_gallons_l3405_340548

-- Define the number of containers
def num_containers : ℕ := 80

-- Define the relationship between containers and pints
def containers_to_pints : ℚ := 7 / (7/2)

-- Define the conversion rate from pints to gallons
def pints_per_gallon : ℕ := 8

-- Theorem stating the total amount of tea in gallons
theorem total_tea_gallons : 
  (↑num_containers * containers_to_pints) / ↑pints_per_gallon = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_tea_gallons_l3405_340548


namespace NUMINAMATH_CALUDE_odd_even_sum_theorem_l3405_340594

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_sum_theorem (f g : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_even : is_even_function g) 
  (h_diff : ∀ x, f x - g x = x^2 + 9*x + 12) : 
  ∀ x, f x + g x = -x^2 + 9*x - 12 :=
by sorry

end NUMINAMATH_CALUDE_odd_even_sum_theorem_l3405_340594


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3405_340535

theorem trigonometric_problem (α : Real) 
  (h1 : 3 * Real.pi / 4 < α ∧ α < Real.pi) 
  (h2 : Real.tan α + 1 / Real.tan α = -10/3) : 
  Real.tan α = -1/3 ∧ 
  (Real.sin (Real.pi + α))^2 + 2 * Real.sin α * Real.sin (Real.pi/2 + α) + 1 / 
  (3 * Real.sin α * Real.cos (Real.pi/2 - α) - 2 * Real.cos α * Real.cos (Real.pi - α)) = 5/21 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3405_340535


namespace NUMINAMATH_CALUDE_total_spider_legs_is_33_l3405_340579

/-- The total number of spider legs in a room with 5 spiders -/
def total_spider_legs : ℕ :=
  let spider1 := 6
  let spider2 := 7
  let spider3 := 8
  let spider4 := 5
  let spider5 := 7
  spider1 + spider2 + spider3 + spider4 + spider5

/-- Theorem stating that the total number of spider legs is 33 -/
theorem total_spider_legs_is_33 : total_spider_legs = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_is_33_l3405_340579


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3405_340519

theorem quadratic_inequality_empty_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) ↔ a ∈ Set.Icc (-4 : ℝ) 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l3405_340519


namespace NUMINAMATH_CALUDE_hot_dog_buns_per_package_l3405_340525

/-- Proves that the number of hot dog buns in one package is 8, given the conditions of the problem -/
theorem hot_dog_buns_per_package 
  (total_packages : ℕ) 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (buns_per_student : ℕ) 
  (h1 : total_packages = 30)
  (h2 : num_classes = 4)
  (h3 : students_per_class = 30)
  (h4 : buns_per_student = 2) : 
  (num_classes * students_per_class * buns_per_student) / total_packages = 8 := by
  sorry

#check hot_dog_buns_per_package

end NUMINAMATH_CALUDE_hot_dog_buns_per_package_l3405_340525


namespace NUMINAMATH_CALUDE_decimal_division_equivalence_l3405_340558

theorem decimal_division_equivalence : 
  ∀ (a b : ℚ), a = 11.7 ∧ b = 2.6 → 
    (a / b = 117 / 26) ∧ (a / b = 4.5) := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_equivalence_l3405_340558


namespace NUMINAMATH_CALUDE_red_bank_amount_when_equal_l3405_340568

/-- Proves that the amount in the red coin bank is 12,500 won when both banks have equal amounts -/
theorem red_bank_amount_when_equal (red_initial : ℕ) (yellow_initial : ℕ) 
  (red_daily : ℕ) (yellow_daily : ℕ) :
  red_initial = 8000 →
  yellow_initial = 5000 →
  red_daily = 300 →
  yellow_daily = 500 →
  ∃ d : ℕ, red_initial + d * red_daily = yellow_initial + d * yellow_daily ∧
          red_initial + d * red_daily = 12500 :=
by sorry

end NUMINAMATH_CALUDE_red_bank_amount_when_equal_l3405_340568


namespace NUMINAMATH_CALUDE_circle_on_parabola_circle_standard_equation_l3405_340585

def parabola (x y : ℝ) : Prop := y^2 = 16 * x

def circle_equation (h k r x y : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem circle_on_parabola (h k : ℝ) :
  parabola h k →
  first_quadrant h k →
  circle_equation h k 6 0 0 →
  circle_equation h k 6 4 0 →
  h = 2 ∧ k = 4 * Real.sqrt 2 :=
sorry

theorem circle_standard_equation (h k : ℝ) :
  h = 2 →
  k = 4 * Real.sqrt 2 →
  ∀ x y : ℝ, circle_equation h k 6 x y ↔ circle_equation 2 (4 * Real.sqrt 2) 6 x y :=
sorry

end NUMINAMATH_CALUDE_circle_on_parabola_circle_standard_equation_l3405_340585


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l3405_340550

theorem square_perimeter_ratio (a b : ℝ) (h : a > 0) (k : b > 0) (area_ratio : a^2 / b^2 = 49 / 64) :
  a / b = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l3405_340550


namespace NUMINAMATH_CALUDE_kim_gum_needs_l3405_340587

/-- The number of cousins Kim has -/
def num_cousins : ℕ := 4

/-- The number of gum pieces Kim wants to give to each cousin -/
def gum_per_cousin : ℕ := 5

/-- The total number of gum pieces Kim needs -/
def total_gum : ℕ := num_cousins * gum_per_cousin

/-- Theorem stating that the total number of gum pieces Kim needs is 20 -/
theorem kim_gum_needs : total_gum = 20 := by sorry

end NUMINAMATH_CALUDE_kim_gum_needs_l3405_340587


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3405_340597

theorem rahul_deepak_age_ratio : 
  ∀ (rahul_age deepak_age : ℕ),
    rahul_age + 6 = 18 →
    deepak_age = 9 →
    (rahul_age : ℚ) / deepak_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3405_340597


namespace NUMINAMATH_CALUDE_waiter_tip_problem_l3405_340536

theorem waiter_tip_problem (total_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) 
  (h1 : total_customers = 10)
  (h2 : tip_amount = 3)
  (h3 : total_tips = 15) :
  total_customers - (total_tips / tip_amount) = 5 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tip_problem_l3405_340536


namespace NUMINAMATH_CALUDE_sine_tangent_sum_greater_than_2pi_l3405_340541

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_angles : A + B + C = π

-- State the theorem
theorem sine_tangent_sum_greater_than_2pi (t : AcuteTriangle) :
  Real.sin t.A + Real.sin t.B + Real.sin t.C +
  Real.tan t.A + Real.tan t.B + Real.tan t.C > 2 * π :=
by sorry

end NUMINAMATH_CALUDE_sine_tangent_sum_greater_than_2pi_l3405_340541


namespace NUMINAMATH_CALUDE_two_equidistant_points_l3405_340531

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle with center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Configuration of a circle and two parallel lines -/
structure CircleLineConfiguration where
  circle : Circle
  line1 : Line
  line2 : Line
  d : ℝ
  h : d > circle.radius

/-- A point is equidistant from a circle and two parallel lines -/
def isEquidistant (p : Point) (config : CircleLineConfiguration) : Prop :=
  sorry

/-- The number of equidistant points -/
def numEquidistantPoints (config : CircleLineConfiguration) : ℕ :=
  sorry

/-- Theorem: There are exactly 2 equidistant points -/
theorem two_equidistant_points (config : CircleLineConfiguration) :
  numEquidistantPoints config = 2 :=
sorry

end NUMINAMATH_CALUDE_two_equidistant_points_l3405_340531


namespace NUMINAMATH_CALUDE_exactly_one_statement_true_l3405_340515

/-- Polynomials A, B, C, D, and E -/
def A (x : ℝ) : ℝ := 2 * x^2
def B (x : ℝ) : ℝ := x + 1
def C (x : ℝ) : ℝ := -2 * x
def D (y : ℝ) : ℝ := y^2
def E (x y : ℝ) : ℝ := 2 * x - y

/-- Statement 1: For all positive integer y, B*C + A + D + E > 0 -/
def statement1 : Prop :=
  ∀ (x : ℝ) (y : ℕ), (B x * C x + A x + D y + E x y) > 0

/-- Statement 2: There exist real numbers x and y such that A + D + 2E = -2 -/
def statement2 : Prop :=
  ∃ (x y : ℝ), A x + D y + 2 * E x y = -2

/-- Statement 3: For all real x, if 3(A-B) + m*B*C has no linear term in x
    (where m is a constant), then 3(A-B) + m*B*C > -3 -/
def statement3 : Prop :=
  ∀ (x m : ℝ),
    (∃ (k : ℝ), 3 * (A x - B x) + m * B x * C x = k * x^2 + (3 * (A 0 - B 0) + m * B 0 * C 0)) →
    3 * (A x - B x) + m * B x * C x > -3

theorem exactly_one_statement_true :
  (statement1 ∧ ¬statement2 ∧ ¬statement3) ∨
  (¬statement1 ∧ statement2 ∧ ¬statement3) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3) := by sorry

end NUMINAMATH_CALUDE_exactly_one_statement_true_l3405_340515


namespace NUMINAMATH_CALUDE_windows_installed_proof_l3405_340592

/-- Calculates the number of windows already installed given the total number of windows,
    time to install each window, and remaining installation time. -/
def windows_installed (total_windows : ℕ) (install_time_per_window : ℕ) (remaining_time : ℕ) : ℕ :=
  total_windows - (remaining_time / install_time_per_window)

/-- Proves that given the specific conditions, the number of windows already installed is 8. -/
theorem windows_installed_proof :
  windows_installed 14 8 48 = 8 := by
  sorry

end NUMINAMATH_CALUDE_windows_installed_proof_l3405_340592


namespace NUMINAMATH_CALUDE_f_lower_bound_f_inequality_solution_l3405_340588

def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

theorem f_lower_bound : ∀ x : ℝ, f x ≥ 4 := by sorry

theorem f_inequality_solution : 
  ∀ x : ℝ, f x ≥ x^2 - 2*x + 4 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_f_inequality_solution_l3405_340588


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3405_340554

theorem ratio_x_to_y (x y : ℝ) 
  (h1 : (3*x - 2*y) / (2*x + 3*y) = 5/4)
  (h2 : x + y = 5) : 
  x / y = 23/2 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3405_340554


namespace NUMINAMATH_CALUDE_max_min_x_plus_reciprocal_l3405_340546

theorem max_min_x_plus_reciprocal (x : ℝ) (h : 12 = x^2 + 1/x^2) :
  (∀ y : ℝ, y ≠ 0 → 12 = y^2 + 1/y^2 → x + 1/x ≤ Real.sqrt 14) ∧
  (∀ y : ℝ, y ≠ 0 → 12 = y^2 + 1/y^2 → -Real.sqrt 14 ≤ x + 1/x) :=
by sorry

end NUMINAMATH_CALUDE_max_min_x_plus_reciprocal_l3405_340546


namespace NUMINAMATH_CALUDE_greatest_common_factor_of_168_252_315_l3405_340599

theorem greatest_common_factor_of_168_252_315 : Nat.gcd 168 (Nat.gcd 252 315) = 21 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_of_168_252_315_l3405_340599


namespace NUMINAMATH_CALUDE_buddy_baseball_cards_l3405_340598

theorem buddy_baseball_cards (monday tuesday wednesday thursday : ℕ) : 
  tuesday = monday / 2 →
  wednesday = tuesday + 12 →
  thursday = wednesday + tuesday / 3 →
  thursday = 32 →
  monday = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_buddy_baseball_cards_l3405_340598


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_15_l3405_340572

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem infinitely_many_divisible_by_15 :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ 15 ∣ v (15 * k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_15_l3405_340572


namespace NUMINAMATH_CALUDE_unique_solution_xyz_l3405_340502

theorem unique_solution_xyz (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 12 ∧ y = 10 ∧ z = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_xyz_l3405_340502


namespace NUMINAMATH_CALUDE_cos_sum_inequality_l3405_340503

theorem cos_sum_inequality (x y : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 Real.pi →
  Real.cos (x + y) ≤ Real.cos x * Real.cos y :=
by sorry

end NUMINAMATH_CALUDE_cos_sum_inequality_l3405_340503


namespace NUMINAMATH_CALUDE_concert_seats_count_l3405_340534

/-- Represents the concert ticket sales scenario -/
structure ConcertSales where
  main_price : ℕ  -- Price of main seat tickets
  back_price : ℕ  -- Price of back seat tickets
  total_revenue : ℕ  -- Total revenue from ticket sales
  back_seats_sold : ℕ  -- Number of back seat tickets sold

/-- Calculates the total number of seats in the arena -/
def total_seats (cs : ConcertSales) : ℕ :=
  let main_seats := (cs.total_revenue - cs.back_price * cs.back_seats_sold) / cs.main_price
  main_seats + cs.back_seats_sold

/-- Theorem stating that the total number of seats is 20,000 -/
theorem concert_seats_count (cs : ConcertSales) 
  (h1 : cs.main_price = 55)
  (h2 : cs.back_price = 45)
  (h3 : cs.total_revenue = 955000)
  (h4 : cs.back_seats_sold = 14500) : 
  total_seats cs = 20000 := by
  sorry

#eval total_seats ⟨55, 45, 955000, 14500⟩

end NUMINAMATH_CALUDE_concert_seats_count_l3405_340534


namespace NUMINAMATH_CALUDE_cafeteria_problem_l3405_340529

/-- The cafeteria problem -/
theorem cafeteria_problem 
  (initial_apples : ℕ)
  (apple_cost orange_cost : ℚ)
  (total_earnings : ℚ)
  (apples_left oranges_left : ℕ)
  (h1 : initial_apples = 50)
  (h2 : apple_cost = 8/10)
  (h3 : orange_cost = 1/2)
  (h4 : total_earnings = 49)
  (h5 : apples_left = 10)
  (h6 : oranges_left = 6) :
  ∃ initial_oranges : ℕ, 
    initial_oranges = 40 ∧
    (initial_apples - apples_left) * apple_cost + 
    (initial_oranges - oranges_left) * orange_cost = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_problem_l3405_340529


namespace NUMINAMATH_CALUDE_division_problem_l3405_340553

theorem division_problem : (62976 : ℕ) / 512 = 123 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3405_340553


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l3405_340504

theorem complex_equation_real_solution (a : ℝ) : 
  (((a : ℂ) / (1 + Complex.I) + 1 + Complex.I).im = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l3405_340504


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3405_340581

theorem unique_solution_exponential_equation :
  ∃! (n : ℕ+), Real.exp (1 / n.val) + Real.exp (-1 / n.val) = Real.sqrt n.val :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3405_340581


namespace NUMINAMATH_CALUDE_consecutive_integers_perfect_square_product_specific_consecutive_integers_l3405_340580

theorem consecutive_integers_perfect_square_product :
  ∃ (n : ℤ), (n - 1) * n * (n + 1) * (n + 2) = (n^2 + n - 1)^2 - 1 ∧
  ∃ (k : ℤ), (n^2 + n - 1)^2 = k^2 + 1 ∧
  (n = 0 ∨ n = -1 ∨ n = 1 ∨ n = -2) :=
by sorry

theorem specific_consecutive_integers :
  (-1 : ℤ) * 0 * 1 * 2 = 0^2 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_perfect_square_product_specific_consecutive_integers_l3405_340580


namespace NUMINAMATH_CALUDE_triangle_to_square_area_ratio_l3405_340538

/-- The ratio of the area of a specific triangle to the area of a square -/
theorem triangle_to_square_area_ratio :
  let square_side : ℝ := 10
  let triangle_vertices : List (ℝ × ℝ) := [(2, 4), (4, 4), (4, 6)]
  let triangle_area := abs ((4 - 2) * (6 - 4) / 2)
  let square_area := square_side ^ 2
  triangle_area / square_area = 1 / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_to_square_area_ratio_l3405_340538


namespace NUMINAMATH_CALUDE_power_product_equality_l3405_340570

theorem power_product_equality : (81 : ℝ) ^ (1/4) * (81 : ℝ) ^ (1/5) = 3 * (3 ^ 4) ^ (1/5) := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l3405_340570


namespace NUMINAMATH_CALUDE_mistaken_divisor_problem_l3405_340582

theorem mistaken_divisor_problem (dividend : ℕ) (mistaken_divisor : ℕ) :
  dividend % 21 = 0 →
  dividend / 21 = 28 →
  dividend / mistaken_divisor = 49 →
  mistaken_divisor = 12 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_divisor_problem_l3405_340582


namespace NUMINAMATH_CALUDE_max_roses_for_680_l3405_340526

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : ℚ  -- Price of a single rose
  oneDozen : ℚ    -- Price of one dozen roses
  twoDozen : ℚ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased with a given amount -/
def maxRoses (pricing : RosePricing) (amount : ℚ) : ℕ :=
  sorry

/-- Theorem: Given the specific pricing, the maximum number of roses for $680 is 325 -/
theorem max_roses_for_680 (pricing : RosePricing) 
  (h1 : pricing.individual = 230 / 100)
  (h2 : pricing.oneDozen = 36)
  (h3 : pricing.twoDozen = 50) :
  maxRoses pricing 680 = 325 :=
sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l3405_340526


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3405_340517

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 1000 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 92 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3405_340517


namespace NUMINAMATH_CALUDE_four_digit_permutations_l3405_340584

/-- The number of distinct permutations of a multiset with repeated elements -/
def multinomial_coefficient (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.map Nat.factorial).prod

/-- The multiset representation of the given digits -/
def digit_multiset : List ℕ := [3, 3, 3, 5]

/-- The total number of digits -/
def total_digits : ℕ := digit_multiset.length

/-- The list of repetitions for each unique digit -/
def repetitions : List ℕ := [3, 1]

theorem four_digit_permutations :
  multinomial_coefficient total_digits repetitions = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_permutations_l3405_340584


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3405_340522

-- Define propositions p and q
variable (p q : Prop)

-- Define the statement "p or q is false" is sufficient for "not p is true"
def is_sufficient : Prop :=
  (¬(p ∨ q)) → (¬p)

-- Define the statement "p or q is false" is not necessary for "not p is true"
def is_not_necessary : Prop :=
  ∃ (p q : Prop), (¬p) ∧ ¬(¬(p ∨ q))

-- The main theorem stating that "p or q is false" is sufficient but not necessary for "not p is true"
theorem sufficient_but_not_necessary :
  (is_sufficient p q) ∧ is_not_necessary :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3405_340522


namespace NUMINAMATH_CALUDE_square_of_102_l3405_340563

theorem square_of_102 : 102 * 102 = 10404 := by
  sorry

end NUMINAMATH_CALUDE_square_of_102_l3405_340563


namespace NUMINAMATH_CALUDE_largest_negative_smallest_positive_smallest_abs_l3405_340560

theorem largest_negative_smallest_positive_smallest_abs (a b c : ℤ) : 
  (∀ x : ℤ, x < 0 → x ≤ a) →  -- a is the largest negative integer
  (∀ x : ℤ, x > 0 → b ≤ x) →  -- b is the smallest positive integer
  (∀ x : ℤ, |c| ≤ |x|) →      -- c has the smallest absolute value
  a + c - b = -2 := by
sorry

end NUMINAMATH_CALUDE_largest_negative_smallest_positive_smallest_abs_l3405_340560


namespace NUMINAMATH_CALUDE_smallest_multiple_l3405_340590

theorem smallest_multiple (n : ℕ) : n = 2015 ↔ 
  n > 0 ∧ 
  31 ∣ n ∧ 
  n % 97 = 6 ∧ 
  ∀ m : ℕ, m > 0 → 31 ∣ m → m % 97 = 6 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3405_340590


namespace NUMINAMATH_CALUDE_alexas_vacation_time_l3405_340507

/-- Proves that Alexa's vacation time is 9 days given the conditions of the problem. -/
theorem alexas_vacation_time (E : ℝ) 
  (ethan_time : E > 0)
  (alexa_time : ℝ)
  (joey_time : ℝ)
  (alexa_vacation : alexa_time = 3/4 * E)
  (joey_swimming : joey_time = 1/2 * E)
  (joey_days : joey_time = 6) : 
  alexa_time = 9 := by
  sorry

end NUMINAMATH_CALUDE_alexas_vacation_time_l3405_340507


namespace NUMINAMATH_CALUDE_outfits_count_l3405_340543

/-- The number of different outfits that can be formed by choosing one top and one pair of pants -/
def number_of_outfits (num_tops : ℕ) (num_pants : ℕ) : ℕ :=
  num_tops * num_pants

/-- Theorem stating that with 4 tops and 3 pants, the number of outfits is 12 -/
theorem outfits_count : number_of_outfits 4 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3405_340543


namespace NUMINAMATH_CALUDE_parallel_lines_a_values_l3405_340523

/-- Given two lines l₁ and l₂, if they are parallel, then a = -1 or a = 2 -/
theorem parallel_lines_a_values (a : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | (a - 1) * x + y + 3 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | 2 * x + a * y + 1 = 0}
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₂ → (a - 1) * (x₂ - x₁) = -(y₂ - y₁)) →
  a = -1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_values_l3405_340523


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3405_340518

/-- The sum of y-coordinates of points where a circle intersects the y-axis -/
theorem circle_y_axis_intersection_sum (c : ℝ × ℝ) (r : ℝ) : 
  c.1 = -8 → c.2 = 3 → r = 15 → 
  ∃ y₁ y₂ : ℝ, 
    (0 - c.1)^2 + (y₁ - c.2)^2 = r^2 ∧
    (0 - c.1)^2 + (y₂ - c.2)^2 = r^2 ∧
    y₁ + y₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3405_340518


namespace NUMINAMATH_CALUDE_shortest_distance_to_circle_l3405_340514

def circle_equation (x y : ℝ) : Prop := (x - 8)^2 + (y - 7)^2 = 25

def point : ℝ × ℝ := (1, -2)

def center : ℝ × ℝ := (8, 7)

def radius : ℝ := 5

theorem shortest_distance_to_circle :
  let d := Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2) - radius
  d = Real.sqrt 130 - 5 := by sorry

end NUMINAMATH_CALUDE_shortest_distance_to_circle_l3405_340514


namespace NUMINAMATH_CALUDE_min_value_at_eight_l3405_340575

theorem min_value_at_eight (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 24 / n ≥ 17 / 3 ∧
  ∃ (m : ℕ), m > 0 ∧ (m : ℝ) / 3 + 24 / m = 17 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_at_eight_l3405_340575


namespace NUMINAMATH_CALUDE_interesting_triple_existence_l3405_340578

/-- Definition of an interesting triple -/
def is_interesting (a b c : ℕ) : Prop :=
  (c^2 + 1) ∣ ((a^2 + 1) * (b^2 + 1)) ∧
  ¬((c^2 + 1) ∣ (a^2 + 1)) ∧
  ¬((c^2 + 1) ∣ (b^2 + 1))

theorem interesting_triple_existence 
  (a b c : ℕ) 
  (h : is_interesting a b c) : 
  ∃ u v : ℕ, is_interesting u v c ∧ u * v < c^3 := by
  sorry

end NUMINAMATH_CALUDE_interesting_triple_existence_l3405_340578


namespace NUMINAMATH_CALUDE_polynomial_factor_l3405_340577

theorem polynomial_factor (x : ℝ) : ∃ (q : ℝ → ℝ), x^2 - 1 = (x + 1) * q x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l3405_340577


namespace NUMINAMATH_CALUDE_james_earnings_difference_l3405_340501

theorem james_earnings_difference (january_earnings : ℕ) 
  (february_earnings : ℕ) (march_earnings : ℕ) (total_earnings : ℕ) :
  january_earnings = 4000 →
  february_earnings = 2 * january_earnings →
  march_earnings < february_earnings →
  total_earnings = january_earnings + february_earnings + march_earnings →
  total_earnings = 18000 →
  february_earnings - march_earnings = 2000 := by
sorry

end NUMINAMATH_CALUDE_james_earnings_difference_l3405_340501
