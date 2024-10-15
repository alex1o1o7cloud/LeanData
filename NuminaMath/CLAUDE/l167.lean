import Mathlib

namespace NUMINAMATH_CALUDE_clara_age_in_five_years_l167_16759

/-- Given the conditions about Alice and Clara's pens and ages, prove Clara's age in 5 years. -/
theorem clara_age_in_five_years
  (alice_pens : ℕ)
  (clara_pens_ratio : ℚ)
  (alice_age : ℕ)
  (clara_older : Prop)
  (pen_diff_equals_age_diff : Prop)
  (h1 : alice_pens = 60)
  (h2 : clara_pens_ratio = 2 / 5)
  (h3 : alice_age = 20)
  (h4 : clara_older)
  (h5 : pen_diff_equals_age_diff) :
  ∃ (clara_age : ℕ), clara_age + 5 = 61 :=
by sorry

end NUMINAMATH_CALUDE_clara_age_in_five_years_l167_16759


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l167_16717

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  2 * (2*b - 3*a) + 3 * (2*a - 3*b) = -5*b := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  4*a^2 + 2*(3*a*b - 2*a^2) - (7*a*b - 1) = -a*b + 1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l167_16717


namespace NUMINAMATH_CALUDE_original_wage_calculation_l167_16785

/-- The worker's original daily wage -/
def original_wage : ℝ := 242.83

/-- The new total weekly salary -/
def new_total_salary : ℝ := 1457

/-- The percentage increases for each day of the work week -/
def wage_increases : List ℝ := [0.20, 0.30, 0.40, 0.50, 0.60]

theorem original_wage_calculation :
  (wage_increases.map (λ i => (1 + i) * original_wage)).sum = new_total_salary :=
sorry

end NUMINAMATH_CALUDE_original_wage_calculation_l167_16785


namespace NUMINAMATH_CALUDE_parking_lot_car_difference_l167_16716

theorem parking_lot_car_difference (initial_cars : ℕ) (cars_left : ℕ) (current_cars : ℕ) : 
  initial_cars = 80 → cars_left = 13 → current_cars = 85 → 
  (current_cars - initial_cars) + cars_left = 18 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_car_difference_l167_16716


namespace NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l167_16720

/-- A polynomial over the complex numbers -/
def ComplexPolynomial := ℂ → ℂ

/-- Definition of an even polynomial -/
def IsEvenPolynomial (P : ComplexPolynomial) : Prop :=
  ∀ z : ℂ, P z = P (-z)

/-- The main theorem -/
theorem even_polynomial_iff_product_with_negation (P : ComplexPolynomial) :
  IsEvenPolynomial P ↔ ∃ Q : ComplexPolynomial, ∀ z : ℂ, P z = Q z * Q (-z) := by
  sorry

end NUMINAMATH_CALUDE_even_polynomial_iff_product_with_negation_l167_16720


namespace NUMINAMATH_CALUDE_determine_new_harvest_l167_16754

/-- Represents the harvest data for two plots of land before and after applying new agricultural techniques. -/
structure HarvestData where
  initial_total : ℝ
  yield_increase_plot1 : ℝ
  yield_increase_plot2 : ℝ
  new_total : ℝ

/-- Represents the harvest amounts for each plot after applying new techniques. -/
structure NewHarvest where
  plot1 : ℝ
  plot2 : ℝ

/-- Theorem stating that given the initial conditions, the new harvest amounts can be determined. -/
theorem determine_new_harvest (data : HarvestData) 
  (h1 : data.initial_total = 14.7)
  (h2 : data.yield_increase_plot1 = 0.8)
  (h3 : data.yield_increase_plot2 = 0.24)
  (h4 : data.new_total = 21.42) :
  ∃ (new_harvest : NewHarvest),
    new_harvest.plot1 = 10.26 ∧
    new_harvest.plot2 = 11.16 ∧
    new_harvest.plot1 + new_harvest.plot2 = data.new_total ∧
    new_harvest.plot1 / (1 + data.yield_increase_plot1) + 
    new_harvest.plot2 / (1 + data.yield_increase_plot2) = data.initial_total :=
  sorry

end NUMINAMATH_CALUDE_determine_new_harvest_l167_16754


namespace NUMINAMATH_CALUDE_a_bounded_by_two_l167_16709

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem a_bounded_by_two
  (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (a : ℝ)
  (h_ineq : ∀ x : ℝ, f (a * 2^x) - f (4^x + 1) ≤ 0) :
  -2 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_a_bounded_by_two_l167_16709


namespace NUMINAMATH_CALUDE_other_number_is_99_l167_16751

/-- Given two positive integers with specific HCF and LCM, prove one is 99 when the other is 48 -/
theorem other_number_is_99 (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 396) (h3 : a = 48) :
  b = 99 := by
  sorry

end NUMINAMATH_CALUDE_other_number_is_99_l167_16751


namespace NUMINAMATH_CALUDE_clock_hands_90_degree_times_l167_16735

/-- The angle (in degrees) that the minute hand moves per minute -/
def minute_hand_speed : ℚ := 6

/-- The angle (in degrees) that the hour hand moves per minute -/
def hour_hand_speed : ℚ := 1/2

/-- The relative speed (in degrees per minute) at which the minute hand moves compared to the hour hand -/
def relative_speed : ℚ := minute_hand_speed - hour_hand_speed

/-- The time (in minutes) when the clock hands first form a 90° angle after 12:00 -/
def first_90_degree_time : ℚ := 90 / relative_speed

/-- The time (in minutes) when the clock hands form a 90° angle for the second time after 12:00 -/
def second_90_degree_time : ℚ := 270 / relative_speed

theorem clock_hands_90_degree_times :
  (first_90_degree_time = 180/11) ∧ 
  (second_90_degree_time = 540/11) := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_90_degree_times_l167_16735


namespace NUMINAMATH_CALUDE_sum_distinct_prime_divisors_of_1728_l167_16791

theorem sum_distinct_prime_divisors_of_1728 : 
  (Finset.sum (Finset.filter Nat.Prime (Nat.divisors 1728)) id) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_divisors_of_1728_l167_16791


namespace NUMINAMATH_CALUDE_equation_solution_l167_16787

theorem equation_solution : 
  let x₁ : ℝ := (3 + Real.sqrt 17) / 2
  let x₂ : ℝ := (-3 - Real.sqrt 17) / 2
  (x₁^2 - 3 * |x₁| - 2 = 0) ∧ 
  (x₂^2 - 3 * |x₂| - 2 = 0) ∧ 
  (∀ x : ℝ, x^2 - 3 * |x| - 2 = 0 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l167_16787


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_l167_16794

theorem sqrt_sum_difference (x : ℝ) : 
  Real.sqrt ((5 - 4 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 4 * Real.sqrt 2) ^ 2) - Real.sqrt 16 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_l167_16794


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l167_16748

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- a_n is arithmetic with common difference d
  (d ≠ 0) →  -- nonzero common difference
  (∀ n, b (n + 1) = q * b n) →  -- b_n is geometric with common ratio q
  (b 1 = a 1 ^ 2) →  -- b₁ = a₁²
  (b 2 = a 2 ^ 2) →  -- b₂ = a₂²
  (b 3 = a 3 ^ 2) →  -- b₃ = a₃²
  (a 2 = -1) →  -- a₂ = -1
  (a 1 < a 2) →  -- a₁ < a₂
  (q = 3 - 2 * Real.sqrt 2 ∧ d = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l167_16748


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l167_16771

/-- Given a point M and its reflection N across the y-axis, and a point P on the y-axis,
    this theorem states that the line passing through P and N has the equation x - y + 1 = 0. -/
theorem reflected_ray_equation (M P N : ℝ × ℝ) : 
  M.1 = 3 ∧ M.2 = -2 ∧   -- M(3, -2)
  P.1 = 0 ∧ P.2 = 1 ∧    -- P(0, 1) on y-axis
  N.1 = -M.1 ∧ N.2 = M.2 -- N is reflection of M across y-axis
  → (∀ x y : ℝ, (x - y + 1 = 0) ↔ (∃ t : ℝ, x = N.1 * t + P.1 * (1 - t) ∧ y = N.2 * t + P.2 * (1 - t))) :=
by sorry


end NUMINAMATH_CALUDE_reflected_ray_equation_l167_16771


namespace NUMINAMATH_CALUDE_min_balloons_required_l167_16733

/-- Represents a balloon color -/
inductive Color
| A | B | C | D | E

/-- Represents a row of balloons -/
def BalloonRow := List Color

/-- Checks if two colors are adjacent in a balloon row -/
def areAdjacent (row : BalloonRow) (c1 c2 : Color) : Prop :=
  ∃ i, (row.get? i = some c1 ∧ row.get? (i+1) = some c2) ∨
       (row.get? i = some c2 ∧ row.get? (i+1) = some c1)

/-- Checks if all pairs of colors are adjacent in a balloon row -/
def allPairsAdjacent (row : BalloonRow) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → areAdjacent row c1 c2

/-- The main theorem: minimum number of balloons required is 11 -/
theorem min_balloons_required :
  ∀ row : BalloonRow,
    allPairsAdjacent row →
    row.length ≥ 11 ∧
    (∃ row' : BalloonRow, allPairsAdjacent row' ∧ row'.length = 11) :=
by sorry

end NUMINAMATH_CALUDE_min_balloons_required_l167_16733


namespace NUMINAMATH_CALUDE_factor_expression_l167_16729

theorem factor_expression (a b : ℝ) : 9 * a^2 * b - b = b * (3 * a + 1) * (3 * a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l167_16729


namespace NUMINAMATH_CALUDE_tinas_oranges_l167_16747

/-- The number of oranges in Tina's bag -/
def oranges : ℕ := sorry

/-- The number of apples in Tina's bag -/
def apples : ℕ := 9

/-- The number of tangerines in Tina's bag -/
def tangerines : ℕ := 17

/-- The number of oranges removed -/
def oranges_removed : ℕ := 2

/-- The number of tangerines removed -/
def tangerines_removed : ℕ := 10

/-- Theorem stating that the number of oranges in Tina's bag is 5 -/
theorem tinas_oranges : oranges = 5 := by
  have h1 : tangerines - tangerines_removed = (oranges - oranges_removed) + 4 := by sorry
  sorry

end NUMINAMATH_CALUDE_tinas_oranges_l167_16747


namespace NUMINAMATH_CALUDE_multiplication_error_correction_l167_16741

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem multiplication_error_correction 
  (c d : ℕ) 
  (h1 : is_two_digit c) 
  (h2 : (reverse_digits c) * d = 143) : 
  c * d = 341 := by
sorry

end NUMINAMATH_CALUDE_multiplication_error_correction_l167_16741


namespace NUMINAMATH_CALUDE_integral_rational_function_l167_16725

open Real

theorem integral_rational_function (x : ℝ) :
  deriv (fun x => (1/2) * log (x^2 + 2*x + 5) + (1/2) * arctan ((x + 1)/2)) x
  = (x + 2) / (x^2 + 2*x + 5) := by sorry

end NUMINAMATH_CALUDE_integral_rational_function_l167_16725


namespace NUMINAMATH_CALUDE_circle_op_range_theorem_l167_16782

/-- Custom operation ⊙ on real numbers -/
def circle_op (a b : ℝ) : ℝ := a * b - 2 * a - b

/-- Theorem stating the range of x for which x ⊙ (x+2) < 0 -/
theorem circle_op_range_theorem :
  ∀ x : ℝ, circle_op x (x + 2) < 0 ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_range_theorem_l167_16782


namespace NUMINAMATH_CALUDE_square_side_length_difference_l167_16758

theorem square_side_length_difference (area_A area_B : ℝ) 
  (h_A : area_A = 25) (h_B : area_B = 81) : 
  Real.sqrt area_B - Real.sqrt area_A = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_difference_l167_16758


namespace NUMINAMATH_CALUDE_sum_of_k_for_minimum_area_l167_16721

/-- The sum of k values that minimize the triangle area --/
def sum_of_k_values : ℤ := 24

/-- Point type --/
structure Point where
  x : ℚ
  y : ℚ

/-- Triangle defined by three points --/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Function to calculate the area of a triangle --/
def triangle_area (t : Triangle) : ℚ :=
  sorry

/-- Function to check if a triangle has minimum area --/
def has_minimum_area (t : Triangle) : Prop :=
  sorry

/-- Theorem stating the sum of k values that minimize the triangle area --/
theorem sum_of_k_for_minimum_area :
  ∃ (k1 k2 : ℤ),
    k1 ≠ k2 ∧
    has_minimum_area (Triangle.mk
      (Point.mk 2 9)
      (Point.mk 14 18)
      (Point.mk 6 k1)) ∧
    has_minimum_area (Triangle.mk
      (Point.mk 2 9)
      (Point.mk 14 18)
      (Point.mk 6 k2)) ∧
    k1 + k2 = sum_of_k_values :=
  sorry

end NUMINAMATH_CALUDE_sum_of_k_for_minimum_area_l167_16721


namespace NUMINAMATH_CALUDE_existence_of_unachievable_fraction_l167_16728

/-- Given an odd prime p, this theorem proves the existence of a specific fraction that cannot be achieved by any coloring of integers. -/
theorem existence_of_unachievable_fraction (p : Nat) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ a : Nat, 0 < a ∧ a < p ∧
  ∀ (coloring : Nat → Bool) (N : Nat),
    N = (p^3 - p) / 4 - 1 →
    ∀ n : Nat, 0 < n ∧ n ≤ N →
      (Finset.filter (fun i => coloring i) (Finset.range n)).card ≠ n * a / p :=
by sorry

end NUMINAMATH_CALUDE_existence_of_unachievable_fraction_l167_16728


namespace NUMINAMATH_CALUDE_function_value_at_five_l167_16757

theorem function_value_at_five (f : ℝ → ℝ) 
  (h : ∀ x, f x + 3 * f (1 - x) = 2 * x^2 + x) : 
  f 5 = 29/8 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_five_l167_16757


namespace NUMINAMATH_CALUDE_sphere_stack_ratio_l167_16798

theorem sphere_stack_ratio (n : ℕ) (sphere_volume_ratio : ℚ) 
  (h1 : n = 5)
  (h2 : sphere_volume_ratio = 2/3) : 
  (n : ℚ) * (1 - sphere_volume_ratio) / (n * sphere_volume_ratio) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_stack_ratio_l167_16798


namespace NUMINAMATH_CALUDE_line_segment_ratio_l167_16711

/-- Given five points P, Q, R, S, T on a line in that order, with specified distances between them,
    prove that the ratio of PR to ST is 9/10. -/
theorem line_segment_ratio (P Q R S T : ℝ) : 
  P < Q ∧ Q < R ∧ R < S ∧ S < T →  -- Points are in order
  Q - P = 3 →                      -- PQ = 3
  R - Q = 6 →                      -- QR = 6
  S - R = 4 →                      -- RS = 4
  T - S = 10 →                     -- ST = 10
  T - P = 30 →                     -- Total distance PT = 30
  (R - P) / (T - S) = 9 / 10 :=    -- Ratio of PR to ST
by
  sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l167_16711


namespace NUMINAMATH_CALUDE_race_time_difference_l167_16702

/-- Race parameters and result -/
theorem race_time_difference
  (malcolm_speed : ℝ) -- Malcolm's speed in minutes per mile
  (joshua_speed : ℝ)  -- Joshua's speed in minutes per mile
  (race_distance : ℝ) -- Race distance in miles
  (h1 : malcolm_speed = 7)
  (h2 : joshua_speed = 8)
  (h3 : race_distance = 12) :
  joshua_speed * race_distance - malcolm_speed * race_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l167_16702


namespace NUMINAMATH_CALUDE_proportion_ones_is_42_233_l167_16744

/-- The number of three-digit integers -/
def num_three_digit_ints : ℕ := 999 - 100 + 1

/-- The total number of digits in all three-digit integers -/
def total_digits : ℕ := num_three_digit_ints * 3

/-- The number of times each digit (1-9) appears in the three-digit integers -/
def digit_occurrences : ℕ := 100 + 90 + 90

/-- The number of times zero appears in the three-digit integers -/
def zero_occurrences : ℕ := 90 + 90

/-- The total number of digits after squaring -/
def total_squared_digits : ℕ := 
  (4 * digit_occurrences + zero_occurrences) + (6 * digit_occurrences * 2)

/-- The number of ones after squaring -/
def num_ones : ℕ := 3 * digit_occurrences

/-- The proportion of ones in the squared digits -/
def proportion_ones : ℚ := num_ones / total_squared_digits

theorem proportion_ones_is_42_233 : proportion_ones = 42 / 233 := by sorry

end NUMINAMATH_CALUDE_proportion_ones_is_42_233_l167_16744


namespace NUMINAMATH_CALUDE_male_students_count_l167_16780

theorem male_students_count (total_students sample_size female_in_sample : ℕ) 
  (h1 : total_students = 1000)
  (h2 : sample_size = 100)
  (h3 : female_in_sample = 51)
  (h4 : female_in_sample < sample_size) :
  (total_students : ℚ) * ((sample_size - female_in_sample) : ℚ) / (sample_size : ℚ) = 490 := by
  sorry

end NUMINAMATH_CALUDE_male_students_count_l167_16780


namespace NUMINAMATH_CALUDE_chebyshev_properties_l167_16731

/-- Chebyshev polynomial of the first kind -/
def T : ℕ → (Real → Real)
| 0 => λ _ => 1
| 1 => λ x => x
| (n + 2) => λ x => 2 * x * T (n + 1) x - T n x

/-- Chebyshev polynomial of the second kind -/
def U : ℕ → (Real → Real)
| 0 => λ _ => 1
| 1 => λ x => 2 * x
| (n + 2) => λ x => 2 * x * U (n + 1) x - U n x

/-- Theorem: Chebyshev polynomials satisfy their initial conditions and recurrence relations -/
theorem chebyshev_properties :
  (∀ x, T 0 x = 1) ∧
  (∀ x, T 1 x = x) ∧
  (∀ n x, T (n + 1) x = 2 * x * T n x - T (n - 1) x) ∧
  (∀ x, U 0 x = 1) ∧
  (∀ x, U 1 x = 2 * x) ∧
  (∀ n x, U (n + 1) x = 2 * x * U n x - U (n - 1) x) := by
  sorry

end NUMINAMATH_CALUDE_chebyshev_properties_l167_16731


namespace NUMINAMATH_CALUDE_first_player_win_prob_correct_l167_16783

/-- Represents the probability of winning for the first player in a three-player sequential game -/
def first_player_win_probability : ℚ :=
  729 / 5985

/-- The probability of a successful hit on any turn -/
def hit_probability : ℚ := 1 / 3

/-- The number of players in the game -/
def num_players : ℕ := 3

/-- Theorem stating the probability of the first player winning the game -/
theorem first_player_win_prob_correct :
  let p := hit_probability
  let n := num_players
  (p^2 * (1 - p^(2*n))⁻¹ : ℚ) = first_player_win_probability :=
by sorry

end NUMINAMATH_CALUDE_first_player_win_prob_correct_l167_16783


namespace NUMINAMATH_CALUDE_community_population_l167_16770

/-- Represents the number of people in each category of a community --/
structure Community where
  babies : ℝ
  seniors : ℝ
  children : ℝ
  teenagers : ℝ
  women : ℝ
  men : ℝ

/-- The total number of people in the community --/
def totalPeople (c : Community) : ℝ :=
  c.babies + c.seniors + c.children + c.teenagers + c.women + c.men

/-- Theorem stating the relationship between the number of babies and the total population --/
theorem community_population (c : Community) 
  (h1 : c.men = 1.5 * c.women)
  (h2 : c.women = 3 * c.teenagers)
  (h3 : c.teenagers = 2.5 * c.children)
  (h4 : c.children = 4 * c.seniors)
  (h5 : c.seniors = 3.5 * c.babies) :
  totalPeople c = 316 * c.babies := by
  sorry


end NUMINAMATH_CALUDE_community_population_l167_16770


namespace NUMINAMATH_CALUDE_smallest_m_proof_l167_16742

/-- The smallest positive integer m such that 15m - 3 is divisible by 11 -/
def smallest_m : ℕ := 9

theorem smallest_m_proof :
  smallest_m = 9 ∧
  ∀ k : ℕ, k > 0 → (15 * k - 3) % 11 = 0 → k ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_proof_l167_16742


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l167_16749

theorem floor_ceil_sum : ⌊(0.998 : ℝ)⌋ + ⌈(3.002 : ℝ)⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l167_16749


namespace NUMINAMATH_CALUDE_wendy_bought_four_tables_l167_16795

/-- The number of chairs Wendy bought -/
def num_chairs : ℕ := 4

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_piece : ℕ := 6

/-- The total assembly time (in minutes) -/
def total_time : ℕ := 48

/-- The number of tables Wendy bought -/
def num_tables : ℕ := (total_time - num_chairs * time_per_piece) / time_per_piece

theorem wendy_bought_four_tables : num_tables = 4 := by
  sorry

end NUMINAMATH_CALUDE_wendy_bought_four_tables_l167_16795


namespace NUMINAMATH_CALUDE_system_solution_l167_16773

theorem system_solution :
  let x : ℝ := (133 - Real.sqrt 73) / 48
  let y : ℝ := (-1 + Real.sqrt 73) / 12
  2 * x - 3 * y^2 = 4 ∧ 4 * x + y = 11 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l167_16773


namespace NUMINAMATH_CALUDE_picture_area_l167_16707

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h_frame_area : (2 * x + 3) * (y + 2) - x * y = 34) : x * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l167_16707


namespace NUMINAMATH_CALUDE_gilbert_crickets_l167_16766

/-- The number of crickets Gilbert eats per week at 90°F -/
def crickets_90 : ℕ := 4

/-- The number of crickets Gilbert eats per week at 100°F -/
def crickets_100 : ℕ := 2 * crickets_90

/-- The total number of weeks -/
def total_weeks : ℕ := 15

/-- The fraction of time at 90°F -/
def fraction_90 : ℚ := 4/5

/-- The fraction of time at 100°F -/
def fraction_100 : ℚ := 1 - fraction_90

theorem gilbert_crickets :
  (↑crickets_90 * (fraction_90 * total_weeks) +
   ↑crickets_100 * (fraction_100 * total_weeks)).floor = 72 := by
  sorry

end NUMINAMATH_CALUDE_gilbert_crickets_l167_16766


namespace NUMINAMATH_CALUDE_square_hexagon_area_l167_16767

theorem square_hexagon_area (s : ℝ) (square_area : ℝ) (hex_area : ℝ) : 
  square_area = Real.sqrt 3 →
  square_area = s^2 →
  hex_area = 3 * Real.sqrt 3 * s^2 / 2 →
  hex_area = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_square_hexagon_area_l167_16767


namespace NUMINAMATH_CALUDE_integer_decimal_parts_theorem_l167_16772

theorem integer_decimal_parts_theorem :
  ∀ (a b : ℝ),
  (a = ⌊7 - Real.sqrt 13⌋) →
  (b = 7 - Real.sqrt 13 - a) →
  (2 * a - b = 2 + Real.sqrt 13) := by
sorry

end NUMINAMATH_CALUDE_integer_decimal_parts_theorem_l167_16772


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_l167_16763

theorem simplify_sqrt_fraction : 
  (Real.sqrt ((7:ℝ)^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_l167_16763


namespace NUMINAMATH_CALUDE_probability_three_primes_in_six_rolls_l167_16724

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def count_primes_on_12_sided_die : ℕ := 5

def probability_prime_on_12_sided_die : ℚ := 5 / 12

def probability_not_prime_on_12_sided_die : ℚ := 7 / 12

def number_of_ways_to_choose_3_out_of_6 : ℕ := 20

theorem probability_three_primes_in_six_rolls : 
  (probability_prime_on_12_sided_die ^ 3 * 
   probability_not_prime_on_12_sided_die ^ 3 * 
   number_of_ways_to_choose_3_out_of_6 : ℚ) = 3575 / 124416 := by sorry

end NUMINAMATH_CALUDE_probability_three_primes_in_six_rolls_l167_16724


namespace NUMINAMATH_CALUDE_sin_squared_minus_2sin_range_l167_16769

theorem sin_squared_minus_2sin_range :
  ∀ x : ℝ, -1 ≤ Real.sin x ^ 2 - 2 * Real.sin x ∧ Real.sin x ^ 2 - 2 * Real.sin x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_minus_2sin_range_l167_16769


namespace NUMINAMATH_CALUDE_prob_girl_given_boy_specific_l167_16765

/-- Represents a club with members -/
structure Club where
  total_members : ℕ
  girls : ℕ
  boys : ℕ

/-- The probability of choosing a girl given that at least one boy is chosen -/
def prob_girl_given_boy (c : Club) : ℚ :=
  (c.girls * c.boys : ℚ) / ((c.girls * c.boys + (c.boys * (c.boys - 1)) / 2) : ℚ)

theorem prob_girl_given_boy_specific :
  let c : Club := { total_members := 12, girls := 7, boys := 5 }
  prob_girl_given_boy c = 7/9 := by
  sorry


end NUMINAMATH_CALUDE_prob_girl_given_boy_specific_l167_16765


namespace NUMINAMATH_CALUDE_no_strictly_increasing_sequence_with_addition_property_l167_16750

theorem no_strictly_increasing_sequence_with_addition_property :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ n m : ℕ, a (n * m) = a n + a m) ∧ 
    (∀ n : ℕ, a n < a (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_strictly_increasing_sequence_with_addition_property_l167_16750


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l167_16726

theorem factor_difference_of_squares (x : ℝ) : x^2 - 144 = (x - 12) * (x + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l167_16726


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l167_16718

/-- The value of m for a hyperbola with equation (y^2/16) - (x^2/9) = 1 and asymptotes y = ±mx -/
theorem hyperbola_asymptote_slope (m : ℝ) : m > 0 →
  (∀ x y : ℝ, y^2/16 - x^2/9 = 1 → (y = m*x ∨ y = -m*x)) → m = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l167_16718


namespace NUMINAMATH_CALUDE_rectangle_area_puzzle_l167_16734

/-- Given a rectangle divided into six smaller rectangles, if five of the rectangles
    have areas 126, 63, 161, 20, and 40, then the area of the remaining rectangle is 101. -/
theorem rectangle_area_puzzle (A B C D E F : ℝ) :
  A = 126 →
  B = 63 →
  C = 161 →
  D = 20 →
  E = 40 →
  A + B + C + D + E + F = (A + B) + C →
  F = 101 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_puzzle_l167_16734


namespace NUMINAMATH_CALUDE_sin_product_equality_l167_16756

theorem sin_product_equality : 
  Real.sin (π / 14) * Real.sin (3 * π / 14) * Real.sin (5 * π / 14) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equality_l167_16756


namespace NUMINAMATH_CALUDE_intersection_line_circle_chord_length_l167_16738

theorem intersection_line_circle_chord_length (k : ℝ) :
  (∃ M N : ℝ × ℝ, 
    (M.1^2 - 4*M.1 + M.2^2 = 0) ∧ 
    (N.1^2 - 4*N.1 + N.2^2 = 0) ∧
    (M.2 = k*M.1 + 1) ∧ 
    (N.2 = k*N.1 + 1) ∧
    ((M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12)) →
  -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_circle_chord_length_l167_16738


namespace NUMINAMATH_CALUDE_product_of_numbers_l167_16755

theorem product_of_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l167_16755


namespace NUMINAMATH_CALUDE_differential_equation_solution_l167_16737

open Real

/-- The differential equation (x^3 + xy^2) dx + (x^2y + y^3) dy = 0 has a solution F(x, y) = x^4 + 2(xy)^2 + y^4 -/
theorem differential_equation_solution (x y : ℝ) :
  let F : ℝ × ℝ → ℝ := fun (x, y) ↦ x^4 + 2*(x*y)^2 + y^4
  let dFdx : ℝ × ℝ → ℝ := fun (x, y) ↦ 4*x^3 + 4*x*y^2
  let dFdy : ℝ × ℝ → ℝ := fun (x, y) ↦ 4*x^2*y + 4*y^3
  (x^3 + x*y^2) * dFdx (x, y) + (x^2*y + y^3) * dFdy (x, y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l167_16737


namespace NUMINAMATH_CALUDE_gcd_linear_combination_l167_16727

theorem gcd_linear_combination (a b : ℤ) : Int.gcd (5*a + 3*b) (13*a + 8*b) = Int.gcd a b := by
  sorry

end NUMINAMATH_CALUDE_gcd_linear_combination_l167_16727


namespace NUMINAMATH_CALUDE_total_balls_in_box_l167_16713

theorem total_balls_in_box (white_balls black_balls : ℕ) : 
  white_balls = 6 * black_balls →
  black_balls = 8 →
  white_balls + black_balls = 56 := by
sorry

end NUMINAMATH_CALUDE_total_balls_in_box_l167_16713


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l167_16745

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l167_16745


namespace NUMINAMATH_CALUDE_muffins_per_box_l167_16736

theorem muffins_per_box (total_muffins : ℕ) (num_boxes : ℕ) 
  (h1 : total_muffins = 96) (h2 : num_boxes = 8) :
  total_muffins / num_boxes = 12 := by
  sorry

end NUMINAMATH_CALUDE_muffins_per_box_l167_16736


namespace NUMINAMATH_CALUDE_investment_rate_proof_l167_16705

/-- Calculates the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that the given investment scenario results in a 10% annual interest rate -/
theorem investment_rate_proof (principal : ℝ) (final_amount : ℝ) (time : ℕ) 
  (h1 : principal = 5000)
  (h2 : final_amount = 6050.000000000001)
  (h3 : time = 2) :
  ∃ (rate : ℝ), compound_interest principal rate time = final_amount ∧ rate = 0.1 := by
  sorry

#check investment_rate_proof

end NUMINAMATH_CALUDE_investment_rate_proof_l167_16705


namespace NUMINAMATH_CALUDE_opposite_to_turquoise_is_pink_l167_16790

/-- Represents the colors of the squares --/
inductive Color
  | Pink
  | Violet
  | Turquoise
  | Orange

/-- Represents a face of the cube --/
structure Face where
  color : Color

/-- Represents the cube formed by folding the squares --/
structure Cube where
  faces : List Face
  opposite : Face → Face

/-- The configuration of the cube --/
def cube_config : Cube :=
  { faces := [
      Face.mk Color.Pink,
      Face.mk Color.Pink,
      Face.mk Color.Pink,
      Face.mk Color.Violet,
      Face.mk Color.Violet,
      Face.mk Color.Turquoise,
      Face.mk Color.Orange
    ],
    opposite := sorry  -- The actual implementation of the opposite function
  }

/-- Theorem stating that the face opposite to Turquoise is Pink --/
theorem opposite_to_turquoise_is_pink :
  ∃ (f : Face), f ∈ cube_config.faces ∧ 
    f.color = Color.Turquoise ∧ 
    (cube_config.opposite f).color = Color.Pink :=
  sorry


end NUMINAMATH_CALUDE_opposite_to_turquoise_is_pink_l167_16790


namespace NUMINAMATH_CALUDE_power_multiplication_l167_16779

theorem power_multiplication (a b : ℕ) : (10 : ℝ) ^ a * (10 : ℝ) ^ b = (10 : ℝ) ^ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l167_16779


namespace NUMINAMATH_CALUDE_sports_club_overlap_l167_16710

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 30 →
  badminton = 17 →
  tennis = 19 →
  neither = 2 →
  ∃ (both : ℕ), both = 8 ∧
    total = badminton + tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l167_16710


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l167_16762

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The center of a circle passing through three given points -/
def circleCenterThroughThreePoints (A B C : Point) : Point :=
  sorry

/-- The three given points -/
def A : Point := ⟨2, 2⟩
def B : Point := ⟨6, 2⟩
def C : Point := ⟨4, 5⟩

/-- Theorem stating that the center of the circle passing through A, B, and C is (4, 17/6) -/
theorem circle_center_coordinates : 
  let center := circleCenterThroughThreePoints A B C
  center.x = 4 ∧ center.y = 17/6 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l167_16762


namespace NUMINAMATH_CALUDE_waiting_by_stump_is_random_event_l167_16746

-- Define the type for idioms
inductive Idiom
| WaitingByStump
| MarkingBoat
| ScoopingMoon
| MendingMirror

-- Define the property of being a random event
def isRandomEvent (i : Idiom) : Prop :=
  match i with
  | Idiom.WaitingByStump => true
  | _ => false

-- Theorem statement
theorem waiting_by_stump_is_random_event :
  isRandomEvent Idiom.WaitingByStump = true :=
by sorry

end NUMINAMATH_CALUDE_waiting_by_stump_is_random_event_l167_16746


namespace NUMINAMATH_CALUDE_horner_rule_equality_f_at_two_equals_62_l167_16761

/-- Horner's Rule representation of a polynomial -/
def horner_form (a b c d e : ℝ) (x : ℝ) : ℝ :=
  x * (x * (x * (a * x + b) + c) + d) + e

/-- Original polynomial function -/
def f (x : ℝ) : ℝ :=
  2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_rule_equality :
  ∀ x : ℝ, f x = horner_form 2 3 0 5 (-4) x :=
sorry

theorem f_at_two_equals_62 : f 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_equality_f_at_two_equals_62_l167_16761


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l167_16760

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon : num_diagonals 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l167_16760


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l167_16715

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- Distance from a point to a focus -/
noncomputable def distance_to_focus (x y : ℝ) (fx fy : ℝ) : ℝ :=
  Real.sqrt ((x - fx)^2 + (y - fy)^2)

/-- The statement to prove -/
theorem ellipse_focus_distance 
  (x y : ℝ) 
  (h_on_ellipse : is_on_ellipse x y) 
  (f1x f1y f2x f2y : ℝ) 
  (h_focus1 : distance_to_focus x y f1x f1y = 7) :
  distance_to_focus x y f2x f2y = 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l167_16715


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l167_16768

def sumOfDigits (n : ℕ) : ℕ := sorry

def isLucky (n : ℕ) : Prop := n % (sumOfDigits n) = 0

def isMultipleOf7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

theorem least_non_lucky_multiple_of_7 : 
  (isMultipleOf7 14) ∧ 
  ¬(isLucky 14) ∧ 
  ∀ n : ℕ, 0 < n ∧ n < 14 ∧ (isMultipleOf7 n) → (isLucky n) := by sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_7_l167_16768


namespace NUMINAMATH_CALUDE_triangle_ratio_l167_16712

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  A = Real.pi / 3 →
  b = 1 →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_l167_16712


namespace NUMINAMATH_CALUDE_min_intersection_at_45_deg_l167_16703

/-- A square in 2D space -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Rotation of a square around its center -/
def rotate_square (s : Square) (angle : ℝ) : Square :=
  { s with }  -- The internal structure remains the same after rotation

/-- The area of intersection between two squares -/
def intersection_area (s1 s2 : Square) : ℝ := sorry

/-- Theorem: The area of intersection between a square and its rotated version is minimized at 45 degrees -/
theorem min_intersection_at_45_deg (s : Square) :
  ∀ x : ℝ, 0 ≤ x → x ≤ 2 * π →
    intersection_area s (rotate_square s (π/4)) ≤ intersection_area s (rotate_square s x) := by
  sorry

#check min_intersection_at_45_deg

end NUMINAMATH_CALUDE_min_intersection_at_45_deg_l167_16703


namespace NUMINAMATH_CALUDE_election_winner_percentage_l167_16706

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 868 → 
  margin = 336 → 
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 62 / 100 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l167_16706


namespace NUMINAMATH_CALUDE_football_field_lap_time_l167_16701

-- Define the field dimensions
def field_length : ℝ := 100
def field_width : ℝ := 50

-- Define the number of laps and obstacles
def num_laps : ℕ := 6
def num_obstacles : ℕ := 2

-- Define the additional distance per obstacle
def obstacle_distance : ℝ := 20

-- Define the average speed of the player
def average_speed : ℝ := 4

-- Theorem to prove
theorem football_field_lap_time :
  let perimeter : ℝ := 2 * (field_length + field_width)
  let total_obstacle_distance : ℝ := num_obstacles * obstacle_distance
  let lap_distance : ℝ := perimeter + total_obstacle_distance
  let total_distance : ℝ := num_laps * lap_distance
  let time_taken : ℝ := total_distance / average_speed
  time_taken = 510 := by sorry

end NUMINAMATH_CALUDE_football_field_lap_time_l167_16701


namespace NUMINAMATH_CALUDE_determinant_equality_l167_16776

theorem determinant_equality (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = -3 →
  Matrix.det !![x + z, y + w; z, w] = -3 := by
sorry

end NUMINAMATH_CALUDE_determinant_equality_l167_16776


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l167_16740

theorem triangle_angle_sum (A B : Real) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2)
  (hsinA : Real.sin A = Real.sqrt 5 / 5) (hsinB : Real.sin B = Real.sqrt 10 / 10) :
  A + B = π/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l167_16740


namespace NUMINAMATH_CALUDE_tan_double_angle_l167_16792

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) : 
  Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l167_16792


namespace NUMINAMATH_CALUDE_wall_width_proof_l167_16781

/-- Proves that the width of a wall is 22.5 cm given specific dimensions and number of bricks -/
theorem wall_width_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_length : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_length = 800 →
  wall_height = 600 →
  num_bricks = 6400 →
  ∃ (wall_width : ℝ), wall_width = 22.5 ∧
    wall_length * wall_height * wall_width = 
    (brick_length * brick_width * brick_height * num_bricks) :=
by sorry

end NUMINAMATH_CALUDE_wall_width_proof_l167_16781


namespace NUMINAMATH_CALUDE_casino_solution_l167_16778

def casino_problem (money_A money_B money_C : ℕ) : Prop :=
  (money_B = 2 * money_C) ∧
  (money_A = 40) ∧
  (money_A + money_B + money_C = 220)

theorem casino_solution :
  ∀ money_A money_B money_C,
    casino_problem money_A money_B money_C →
    money_C - money_A = 20 := by
  sorry

end NUMINAMATH_CALUDE_casino_solution_l167_16778


namespace NUMINAMATH_CALUDE_square_of_negative_two_m_cubed_l167_16774

theorem square_of_negative_two_m_cubed (m : ℝ) : (-2 * m^3)^2 = 4 * m^6 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_m_cubed_l167_16774


namespace NUMINAMATH_CALUDE_current_rate_calculation_l167_16775

/-- Given a boat with speed in still water and its downstream travel details, 
    calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 20) 
  (h2 : downstream_distance = 5.2) 
  (h3 : downstream_time = 0.2) : 
  ∃ (current_rate : ℝ), 
    current_rate = 6 ∧ 
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l167_16775


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_range_l167_16797

theorem sin_cos_fourth_power_range :
  ∀ x : ℝ, (1/2 : ℝ) ≤ Real.sin x ^ 4 + Real.cos x ^ 4 ∧ Real.sin x ^ 4 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_range_l167_16797


namespace NUMINAMATH_CALUDE_marias_green_towels_l167_16732

theorem marias_green_towels :
  ∀ (green_towels : ℕ),
  (green_towels + 21 : ℕ) - 34 = 22 →
  green_towels = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_marias_green_towels_l167_16732


namespace NUMINAMATH_CALUDE_bake_sale_goal_l167_16764

def brownie_count : ℕ := 4
def brownie_price : ℕ := 3
def lemon_square_count : ℕ := 5
def lemon_square_price : ℕ := 2
def cookie_count : ℕ := 7
def cookie_price : ℕ := 4

def total_goal : ℕ := 50

theorem bake_sale_goal :
  brownie_count * brownie_price +
  lemon_square_count * lemon_square_price +
  cookie_count * cookie_price = total_goal :=
by
  sorry

end NUMINAMATH_CALUDE_bake_sale_goal_l167_16764


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l167_16789

/-- Given a line segment from (2, 5) to (x, 15) with length 13 and x > 0, prove x = 2 + √69 -/
theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((x - 2)^2 + 10^2)^(1/2 : ℝ) = 13 → 
  x = 2 + (69 : ℝ)^(1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l167_16789


namespace NUMINAMATH_CALUDE_distance_to_chord_equals_half_chord_l167_16739

-- Define the circle and points
structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- Define the function to calculate distance from a point to a line segment
def distancePointToSegment (p : Point) (a b : Point) : ℝ := sorry

-- Define the theorem
theorem distance_to_chord_equals_half_chord (O A B C D E : Point) (circle : Circle) :
  O = circle.center →
  distance A E = 2 * circle.radius →
  (∀ p ∈ [A, B, C, E], distance O p = circle.radius) →
  distancePointToSegment O A B = (distance C D) / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_chord_equals_half_chord_l167_16739


namespace NUMINAMATH_CALUDE_product_divisibility_l167_16723

theorem product_divisibility (a b c : ℤ) 
  (h1 : (a + b + c)^2 = -(a*b + a*c + b*c))
  (h2 : a + b ≠ 0)
  (h3 : b + c ≠ 0)
  (h4 : a + c ≠ 0) :
  (∃ k : ℤ, (a + b) * (a + c) = k * (b + c)) ∧
  (∃ k : ℤ, (a + b) * (b + c) = k * (a + c)) ∧
  (∃ k : ℤ, (a + c) * (b + c) = k * (a + b)) :=
sorry

end NUMINAMATH_CALUDE_product_divisibility_l167_16723


namespace NUMINAMATH_CALUDE_treats_calculation_l167_16700

/-- Calculates the number of treats per child per house -/
def treats_per_child_per_house (num_children : ℕ) (num_hours : ℕ) (houses_per_hour : ℕ) (total_treats : ℕ) : ℚ :=
  (total_treats : ℚ) / ((num_children : ℚ) * (num_hours * houses_per_hour))

/-- Theorem: Given the conditions from the problem, the number of treats per child per house is 3 -/
theorem treats_calculation :
  let num_children : ℕ := 3
  let num_hours : ℕ := 4
  let houses_per_hour : ℕ := 5
  let total_treats : ℕ := 180
  treats_per_child_per_house num_children num_hours houses_per_hour total_treats = 3 := by
  sorry

#eval treats_per_child_per_house 3 4 5 180

end NUMINAMATH_CALUDE_treats_calculation_l167_16700


namespace NUMINAMATH_CALUDE_exists_increasing_arithmetic_seq_exists_perm_without_long_increasing_seq_l167_16753

-- Define the set of natural numbers (positive integers)
def N : Set Nat := {n : Nat | n > 0}

-- Define a permutation of N
def isPerm (f : Nat → Nat) : Prop := Function.Bijective f ∧ ∀ n, f n ∈ N

-- Theorem 1
theorem exists_increasing_arithmetic_seq (f : Nat → Nat) (h : isPerm f) :
  ∃ a d : Nat, d > 0 ∧ a ∈ N ∧ (a + d) ∈ N ∧ (a + 2*d) ∈ N ∧
    f a < f (a + d) ∧ f (a + d) < f (a + 2*d) := by sorry

-- Theorem 2
theorem exists_perm_without_long_increasing_seq :
  ∃ f : Nat → Nat, isPerm f ∧
    ∀ a d : Nat, d > 0 → a ∈ N →
      ¬(∀ k : Nat, k ≤ 2003 → f (a + k*d) < f (a + (k+1)*d)) := by sorry

end NUMINAMATH_CALUDE_exists_increasing_arithmetic_seq_exists_perm_without_long_increasing_seq_l167_16753


namespace NUMINAMATH_CALUDE_unique_solution_mod_37_l167_16704

theorem unique_solution_mod_37 :
  ∃! (a b c d : ℤ),
    (a^2 + b*c) % 37 = a % 37 ∧
    (b*(a + d)) % 37 = b % 37 ∧
    (c*(a + d)) % 37 = c % 37 ∧
    (b*c + d^2) % 37 = d % 37 ∧
    (a*d - b*c) % 37 = 1 % 37 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mod_37_l167_16704


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l167_16788

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 + m - 2021 = 0) → (n^2 + n - 2021 = 0) → (m^2 + 2*m + n = 2020) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l167_16788


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l167_16752

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 60 →
  b = 100 →
  c^2 = a^2 + b^2 →
  c = 20 * Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l167_16752


namespace NUMINAMATH_CALUDE_binary_111_equals_7_l167_16722

def binary_to_decimal (b₂ b₁ b₀ : Nat) : Nat :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_111_equals_7 : binary_to_decimal 1 1 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_binary_111_equals_7_l167_16722


namespace NUMINAMATH_CALUDE_milk_sales_l167_16743

theorem milk_sales : 
  let morning_packets : ℕ := 150
  let morning_250ml : ℕ := 60
  let morning_300ml : ℕ := 40
  let morning_350ml : ℕ := morning_packets - morning_250ml - morning_300ml
  let evening_packets : ℕ := 100
  let evening_400ml : ℕ := evening_packets / 2
  let evening_500ml : ℕ := evening_packets / 4
  let evening_450ml : ℕ := evening_packets - evening_400ml - evening_500ml
  let ml_per_ounce : ℕ := 30
  let remaining_ml : ℕ := 42000
  let total_ml : ℕ := 
    morning_250ml * 250 + morning_300ml * 300 + morning_350ml * 350 +
    evening_400ml * 400 + evening_500ml * 500 + evening_450ml * 450
  let sold_ml : ℕ := total_ml - remaining_ml
  let sold_ounces : ℚ := sold_ml / ml_per_ounce
  sold_ounces = 1541.67 := by
    sorry

end NUMINAMATH_CALUDE_milk_sales_l167_16743


namespace NUMINAMATH_CALUDE_consecutive_sum_equals_fourteen_l167_16708

theorem consecutive_sum_equals_fourteen (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) = 14 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_equals_fourteen_l167_16708


namespace NUMINAMATH_CALUDE_function_value_at_alpha_l167_16714

theorem function_value_at_alpha (α : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.cos x ^ 4 + Real.sin x ^ 4
  Real.sin (2 * α) = 2 / 3 →
  f α = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_alpha_l167_16714


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l167_16719

/-- Given vectors a and b, if a + 2b is parallel to ma + b, then m = 1/2 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (ha : a = (2, 3)) 
  (hb : b = (1, 2)) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a + 2 • b = k • (m • a + b)) : 
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l167_16719


namespace NUMINAMATH_CALUDE_triangle_cutting_theorem_l167_16793

theorem triangle_cutting_theorem (x : ℝ) : 
  (∀ a b c : ℝ, a = 6 - x ∧ b = 8 - x ∧ c = 10 - x → a + b ≤ c) →
  x ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_cutting_theorem_l167_16793


namespace NUMINAMATH_CALUDE_first_number_of_sequence_l167_16784

/-- A sequence with specific properties -/
structure Sequence where
  second : ℕ
  increment : ℕ
  final : ℕ

/-- The first number in the sequence -/
def firstNumber (s : Sequence) : ℕ := s.second - s.increment

/-- Theorem stating the properties of the sequence and the first number -/
theorem first_number_of_sequence (s : Sequence) 
  (h1 : s.second = 45)
  (h2 : s.increment = 11)
  (h3 : s.final = 89) :
  firstNumber s = 34 := by
  sorry

#check first_number_of_sequence

end NUMINAMATH_CALUDE_first_number_of_sequence_l167_16784


namespace NUMINAMATH_CALUDE_arithmetic_triangle_theorem_l167_16799

/-- Triangle with sides a, b, c and angles A, B, C in arithmetic sequence --/
structure ArithmeticTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_arithmetic_sequence : True  -- represents that angles are in arithmetic sequence

/-- The theorem to be proved --/
theorem arithmetic_triangle_theorem (t : ArithmeticTriangle) : 
  1 / (t.a + t.b) + 1 / (t.b + t.c) = 3 / (t.a + t.b + t.c) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_triangle_theorem_l167_16799


namespace NUMINAMATH_CALUDE_opposite_of_2023_l167_16786

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l167_16786


namespace NUMINAMATH_CALUDE_max_intersections_circle_sine_l167_16730

/-- The maximum number of intersection points between a circle and sine curve --/
theorem max_intersections_circle_sine (h k : ℝ) : 
  (k ≥ -2 ∧ k ≤ 2) → 
  (∃ (n : ℕ), n ≤ 8 ∧ 
    (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = 4 ∧ y = Real.sin x → 
      (∃ (m : ℕ), m ≤ n ∧ 
        (∀ (p q : ℝ), (p - h)^2 + (q - k)^2 = 4 ∧ q = Real.sin p → 
          (x = p ∧ y = q) ∨ m > 1)))) ∧
  (∀ (m : ℕ), m > 8 → 
    (∃ (x y : ℝ), (x - h)^2 + (y - k)^2 = 4 ∧ y = Real.sin x ∧
      (∀ (p q : ℝ), (p - h)^2 + (q - k)^2 = 4 ∧ q = Real.sin p → 
        (x ≠ p ∨ y ≠ q)))) := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_circle_sine_l167_16730


namespace NUMINAMATH_CALUDE_every_second_sum_of_arithmetic_sequence_l167_16777

def sequence_sum (first : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * first + (n - 1)) / 2

def every_second_sum (first : ℚ) (n : ℕ) : ℚ :=
  sequence_sum first ((n + 1) / 2)

theorem every_second_sum_of_arithmetic_sequence 
  (first : ℚ) (n : ℕ) (h1 : n = 3015) (h2 : sequence_sum first n = 8010) :
  every_second_sum first (n - 1) = 3251.5 := by
  sorry

end NUMINAMATH_CALUDE_every_second_sum_of_arithmetic_sequence_l167_16777


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l167_16796

theorem imaginary_part_of_z (z : ℂ) (h : z * (Complex.I + 1) + Complex.I = 1 + 3 * Complex.I) : 
  z.im = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l167_16796
