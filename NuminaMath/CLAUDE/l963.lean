import Mathlib

namespace NUMINAMATH_CALUDE_max_a_value_l963_96329

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 4*x + 3 else -x^2 - 2*x + 3

theorem max_a_value (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 1), f (x + a) ≥ f (2*a - x)) →
  a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l963_96329


namespace NUMINAMATH_CALUDE_equation_solution_l963_96384

theorem equation_solution :
  ∃ x : ℚ, (2 / 3 + 1 / x = 7 / 9) ∧ (x = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l963_96384


namespace NUMINAMATH_CALUDE_freds_marbles_l963_96335

theorem freds_marbles (total : ℕ) (dark_blue red green : ℕ) : 
  dark_blue ≥ total / 3 →
  red = 38 →
  green = 4 →
  total = dark_blue + red + green →
  total ≥ 63 :=
by sorry

end NUMINAMATH_CALUDE_freds_marbles_l963_96335


namespace NUMINAMATH_CALUDE_unique_prime_in_range_l963_96344

/-- The only prime number in the range (200, 220) is 211 -/
theorem unique_prime_in_range : ∃! (n : ℕ), 200 < n ∧ n < 220 ∧ Nat.Prime n :=
  sorry

end NUMINAMATH_CALUDE_unique_prime_in_range_l963_96344


namespace NUMINAMATH_CALUDE_ellipse_equation_l963_96316

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- The foci of the ellipse -/
def f1 : Point := ⟨-4, 0⟩
def f2 : Point := ⟨4, 0⟩

/-- Distance between foci -/
def focalDistance : ℝ := 8

/-- Maximum area of triangle PF₁F₂ -/
def maxTriangleArea : ℝ := 12

/-- Theorem: Given an ellipse with foci at (-4,0) and (4,0), and maximum area of triangle PF₁F₂ is 12,
    the equation of the ellipse is x²/25 + y²/9 = 1 -/
theorem ellipse_equation (e : Ellipse) : 
  (focalDistance = 8) → 
  (maxTriangleArea = 12) → 
  (e.a^2 = 25 ∧ e.b^2 = 9) := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ellipse_equation_l963_96316


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l963_96396

theorem orthogonal_vectors (y : ℚ) : 
  ((-4 : ℚ) * 3 + 7 * y = 0) → y = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l963_96396


namespace NUMINAMATH_CALUDE_joyce_bananas_l963_96357

/-- Given a number of boxes and bananas per box, calculates the total number of bananas -/
def total_bananas (num_boxes : ℕ) (bananas_per_box : ℕ) : ℕ :=
  num_boxes * bananas_per_box

/-- Proves that 10 boxes with 4 bananas each results in 40 bananas total -/
theorem joyce_bananas : total_bananas 10 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_joyce_bananas_l963_96357


namespace NUMINAMATH_CALUDE_no_valid_n_exists_l963_96310

theorem no_valid_n_exists : ∀ n : ℕ, n ≥ 2 →
  ¬∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    ∀ (a : Fin n → ℕ), (∀ i j : Fin n, i.val < j.val → a i ≠ a j) →
      (∀ i j : Fin n, i.val ≤ j.val → (p ∣ a j - a i) ∨ (q ∣ a j - a i) ∨ (r ∣ a j - a i)) →
        ((∀ i j : Fin n, i.val < j.val → p ∣ a j - a i) ∨
         (∀ i j : Fin n, i.val < j.val → q ∣ a j - a i) ∨
         (∀ i j : Fin n, i.val < j.val → r ∣ a j - a i)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_exists_l963_96310


namespace NUMINAMATH_CALUDE_village_population_after_events_l963_96314

theorem village_population_after_events (initial_population : ℕ) : 
  initial_population = 7600 → 
  (initial_population - initial_population / 10 - 
   (initial_population - initial_population / 10) / 4) = 5130 := by
sorry

end NUMINAMATH_CALUDE_village_population_after_events_l963_96314


namespace NUMINAMATH_CALUDE_fn_equals_de_l963_96375

-- Define the circle
variable (O : Point) (A B : Point)
variable (circle : Circle O)

-- Define other points
variable (C D E F M N : Point)

-- Define the conditions
variable (h1 : C ∈ circle)
variable (h2 : Diameter circle A B)
variable (h3 : Perpendicular CD AB D)
variable (h4 : E ∈ Segment B D)
variable (h5 : AE = AC)
variable (h6 : Square D E F M)
variable (h7 : N ∈ circle ∩ Line A M)

-- State the theorem
theorem fn_equals_de : FN = DE := by
  sorry

end NUMINAMATH_CALUDE_fn_equals_de_l963_96375


namespace NUMINAMATH_CALUDE_wallace_existing_bags_l963_96379

/- Define the problem parameters -/
def batch_size : ℕ := 10
def order_size : ℕ := 60
def days_to_fulfill : ℕ := 4

/- Define the function to calculate the number of bags Wallace can make in given days -/
def bags_made_in_days (days : ℕ) : ℕ := days * batch_size

/- Theorem: Wallace has already made 20 bags of jerky -/
theorem wallace_existing_bags : 
  order_size - bags_made_in_days days_to_fulfill = 20 := by
  sorry

#eval order_size - bags_made_in_days days_to_fulfill

end NUMINAMATH_CALUDE_wallace_existing_bags_l963_96379


namespace NUMINAMATH_CALUDE_lcm_18_30_l963_96370

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l963_96370


namespace NUMINAMATH_CALUDE_dot_product_range_l963_96350

theorem dot_product_range (M N : ℝ × ℝ) (a : ℝ × ℝ) :
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  x₁ ≥ 0 ∧ y₁ ≥ 0 ∧ x₁ + 2 * y₁ ≤ 6 ∧ 3 * x₁ + y₁ ≤ 12 ∧
  x₂ ≥ 0 ∧ y₂ ≥ 0 ∧ x₂ + 2 * y₂ ≤ 6 ∧ 3 * x₂ + y₂ ≤ 12 ∧
  a = (1, -1) →
  -7 ≤ ((x₂ - x₁) * a.1 + (y₂ - y₁) * a.2) ∧ 
  ((x₂ - x₁) * a.1 + (y₂ - y₁) * a.2) ≤ 7 := by
sorry

end NUMINAMATH_CALUDE_dot_product_range_l963_96350


namespace NUMINAMATH_CALUDE_divisible_by_27_l963_96364

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, 2^(2*n - 1) - 9*n^2 + 21*n - 14 = 27*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_27_l963_96364


namespace NUMINAMATH_CALUDE_gift_payment_l963_96399

theorem gift_payment (a b c d : ℝ) 
  (h1 : a + b + c + d = 84)
  (h2 : a = (1/3) * (b + c + d))
  (h3 : b = (1/4) * (a + c + d))
  (h4 : c = (1/5) * (a + b + d))
  (h5 : a ≥ 0) (h6 : b ≥ 0) (h7 : c ≥ 0) (h8 : d ≥ 0) : 
  d = 40 := by
  sorry

end NUMINAMATH_CALUDE_gift_payment_l963_96399


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_slope_l963_96345

/-- A line that does not pass through the third quadrant has a non-positive slope -/
theorem line_not_in_third_quadrant_slope (k : ℝ) :
  (∀ x y : ℝ, y = k * x + 3 → ¬(x < 0 ∧ y < 0)) →
  k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_slope_l963_96345


namespace NUMINAMATH_CALUDE_multiplication_sum_equality_l963_96390

theorem multiplication_sum_equality : 45 * 58 + 45 * 42 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_sum_equality_l963_96390


namespace NUMINAMATH_CALUDE_squirrels_in_tree_l963_96356

theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) : 
  nuts = 2 → squirrels = nuts + 2 → squirrels = 4 := by
  sorry

end NUMINAMATH_CALUDE_squirrels_in_tree_l963_96356


namespace NUMINAMATH_CALUDE_greatest_7_power_divisor_l963_96347

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n is divisible by 7^k -/
def divides_by_7_pow (n : ℕ+) (k : ℕ) : Prop := sorry

theorem greatest_7_power_divisor (n : ℕ+) (h1 : num_divisors n = 30) (h2 : num_divisors (7 * n) = 42) :
  ∃ k : ℕ, divides_by_7_pow n k ∧ k = 1 ∧ ∀ m : ℕ, divides_by_7_pow n m → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_greatest_7_power_divisor_l963_96347


namespace NUMINAMATH_CALUDE_discount_comparison_l963_96392

theorem discount_comparison (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_price : x = 2 * y) :
  x + y = (3/2) * (0.6 * x + 0.8 * y) :=
by sorry

#check discount_comparison

end NUMINAMATH_CALUDE_discount_comparison_l963_96392


namespace NUMINAMATH_CALUDE_ellipse_implies_a_greater_than_one_l963_96397

/-- Represents the condition that the curve is an ellipse with foci on the x-axis -/
def is_ellipse (t : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (3 - t) + y^2 / (t + 1) = 1 → -1 < t ∧ t < 1

/-- Represents the inequality condition -/
def satisfies_inequality (t a : ℝ) : Prop :=
  t^2 - (a - 1) * t - a < 0

/-- The main theorem statement -/
theorem ellipse_implies_a_greater_than_one :
  (∀ t : ℝ, is_ellipse t → (∃ a : ℝ, satisfies_inequality t a)) ∧
  (∃ t a : ℝ, satisfies_inequality t a ∧ ¬is_ellipse t) →
  ∀ a : ℝ, (∃ t : ℝ, is_ellipse t → satisfies_inequality t a) → a > 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_implies_a_greater_than_one_l963_96397


namespace NUMINAMATH_CALUDE_squirrel_journey_time_l963_96336

/-- Proves that a squirrel traveling 0.5 miles at 6 mph and then 1.5 miles at 3 mph
    takes 35 minutes to complete a 2-mile journey. -/
theorem squirrel_journey_time :
  let total_distance : ℝ := 2
  let first_segment_distance : ℝ := 0.5
  let first_segment_speed : ℝ := 6
  let second_segment_distance : ℝ := 1.5
  let second_segment_speed : ℝ := 3
  let first_segment_time : ℝ := first_segment_distance / first_segment_speed
  let second_segment_time : ℝ := second_segment_distance / second_segment_speed
  let total_time_hours : ℝ := first_segment_time + second_segment_time
  let total_time_minutes : ℝ := total_time_hours * 60
  total_distance = first_segment_distance + second_segment_distance →
  total_time_minutes = 35 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_journey_time_l963_96336


namespace NUMINAMATH_CALUDE_problem_statement_l963_96371

theorem problem_statement (x y z : ℝ) 
  (h1 : x + y - 6 = 0) 
  (h2 : z^2 + 9 = x*y) : 
  x^2 + (1/3)*y^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l963_96371


namespace NUMINAMATH_CALUDE_exists_indivisible_treasure_l963_96346

/-- Represents a treasure of gold bars -/
structure Treasure where
  num_bars : ℕ
  total_value : ℕ
  bar_values : Fin num_bars → ℕ
  sum_constraint : (Finset.univ.sum bar_values) = total_value

/-- Represents an even division of a treasure among pirates -/
def EvenDivision (t : Treasure) (num_pirates : ℕ) : Prop :=
  ∃ (division : Fin t.num_bars → Fin num_pirates),
    ∀ p : Fin num_pirates,
      (Finset.univ.filter (λ i => division i = p)).sum t.bar_values =
        t.total_value / num_pirates

/-- The main theorem stating that there exists a treasure that cannot be evenly divided -/
theorem exists_indivisible_treasure :
  ∃ (t : Treasure),
    t.num_bars = 240 ∧
    t.total_value = 360 ∧
    (∀ i : Fin t.num_bars, t.bar_values i > 0) ∧
    ¬(EvenDivision t 3) := by
  sorry

end NUMINAMATH_CALUDE_exists_indivisible_treasure_l963_96346


namespace NUMINAMATH_CALUDE_infinitely_many_squares_l963_96355

theorem infinitely_many_squares (k : ℕ+) :
  ∀ (B : ℕ), ∃ (n m : ℕ), n > B ∧ m > B ∧ (2 * k.val * n - 7 = m^2) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_squares_l963_96355


namespace NUMINAMATH_CALUDE_sum_of_binary_numbers_l963_96378

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def num1 : List Bool := [true, true, true, true, true, true, true, true, true]
def num2 : List Bool := [true, false, false, false, false, false, true]

theorem sum_of_binary_numbers :
  binary_to_decimal num1 + binary_to_decimal num2 = 576 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_numbers_l963_96378


namespace NUMINAMATH_CALUDE_union_covers_reals_l963_96342

def set_A : Set ℝ := {x | x ≤ 2}
def set_B (a : ℝ) : Set ℝ := {x | x > a}

theorem union_covers_reals (a : ℝ) :
  set_A ∪ set_B a = Set.univ → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l963_96342


namespace NUMINAMATH_CALUDE_M_always_positive_l963_96307

theorem M_always_positive (x y : ℝ) : 3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13 > 0 := by
  sorry

end NUMINAMATH_CALUDE_M_always_positive_l963_96307


namespace NUMINAMATH_CALUDE_divisibility_problem_l963_96359

theorem divisibility_problem (a b c : ℤ) 
  (h1 : a ∣ b * c - 1) 
  (h2 : b ∣ c * a - 1) 
  (h3 : c ∣ a * b - 1) : 
  a * b * c ∣ a * b + b * c + c * a - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l963_96359


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l963_96306

/-- The height of a tree after a given number of years, given that it triples its height each year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height every year and reaches 243 feet after 5 years has a height of 9 feet after 2 years -/
theorem tree_height_after_two_years :
  ∃ (initial_height : ℝ),
    tree_height initial_height 5 = 243 ∧
    tree_height initial_height 2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l963_96306


namespace NUMINAMATH_CALUDE_equivalence_of_statements_l963_96366

theorem equivalence_of_statements (P Q : Prop) :
  (¬P → Q) ↔ (¬Q → P) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_of_statements_l963_96366


namespace NUMINAMATH_CALUDE_problem_solution_l963_96353

theorem problem_solution (a b : ℤ) (h : 6 * a * b = 9 * a - 10 * b + 303) : a + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l963_96353


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l963_96331

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l963_96331


namespace NUMINAMATH_CALUDE_linear_function_proof_l963_96308

/-- A linear function passing through (-2, -1) and parallel to y = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x + 3

theorem linear_function_proof :
  (∀ x y : ℝ, f y - f x = 2 * (y - x)) ∧  -- linearity and slope
  f (-2) = -1 ∧                           -- passes through (-2, -1)
  (∀ x : ℝ, f x - (2 * x - 3) = 3) :=     -- parallel to y = 2x - 3
by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l963_96308


namespace NUMINAMATH_CALUDE_lcm_problem_l963_96304

theorem lcm_problem (a b c : ℕ) : 
  lcm a b = 24 → lcm b c = 28 → 
  ∃ (m : ℕ), m = lcm a c ∧ ∀ (n : ℕ), n = lcm a c → m ≤ n := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l963_96304


namespace NUMINAMATH_CALUDE_square_difference_equals_360_l963_96343

theorem square_difference_equals_360 :
  (15 + 12)^2 - (12^2 + 15^2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_360_l963_96343


namespace NUMINAMATH_CALUDE_measure_45_minutes_l963_96321

/-- Represents a cord that can be burned --/
structure Cord :=
  (burn_time : ℝ)
  (burn_rate_uniform : Bool)

/-- Represents the state of burning a cord --/
inductive BurnState
  | Unlit
  | LitOneEnd (time : ℝ)
  | LitBothEnds (time : ℝ)
  | Burned

/-- Represents the measurement setup --/
structure MeasurementSetup :=
  (cord1 : Cord)
  (cord2 : Cord)
  (state1 : BurnState)
  (state2 : BurnState)

/-- The main theorem stating that 45 minutes can be measured --/
theorem measure_45_minutes 
  (c1 c2 : Cord) 
  (h1 : c1.burn_time = 60) 
  (h2 : c2.burn_time = 60) : 
  ∃ (process : List MeasurementSetup), 
    (∃ (t : ℝ), t = 45 ∧ 
      (∃ (final : MeasurementSetup), final ∈ process ∧ 
        final.state1 = BurnState.Burned ∧ 
        final.state2 = BurnState.Burned)) :=
sorry

end NUMINAMATH_CALUDE_measure_45_minutes_l963_96321


namespace NUMINAMATH_CALUDE_group_size_calculation_l963_96303

theorem group_size_calculation (initial_avg : ℝ) (final_avg : ℝ) (new_member1 : ℝ) (new_member2 : ℝ) :
  initial_avg = 48 →
  final_avg = 51 →
  new_member1 = 78 →
  new_member2 = 93 →
  ∃ n : ℕ, n * initial_avg + new_member1 + new_member2 = (n + 2) * final_avg ∧ n = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l963_96303


namespace NUMINAMATH_CALUDE_ship_passengers_theorem_l963_96341

theorem ship_passengers_theorem (total_passengers : ℝ) (round_trip_with_car : ℝ) 
  (round_trip_without_car : ℝ) (h1 : round_trip_with_car > 0) 
  (h2 : round_trip_with_car + round_trip_without_car ≤ total_passengers) 
  (h3 : round_trip_with_car / total_passengers = 0.3) :
  (round_trip_with_car + round_trip_without_car) / total_passengers = 
  round_trip_with_car / total_passengers := by
sorry

end NUMINAMATH_CALUDE_ship_passengers_theorem_l963_96341


namespace NUMINAMATH_CALUDE_smallest_number_of_purple_marbles_l963_96387

theorem smallest_number_of_purple_marbles :
  ∀ (n : ℕ),
  (n ≥ 10) →  -- Ensuring n is at least 10 to satisfy all conditions
  (n % 10 = 0) →  -- n must be a multiple of 10
  (n / 2 : ℕ) + (n / 5 : ℕ) + 7 < n →  -- Ensuring there's at least one purple marble
  (∃ (blue red green purple : ℕ),
    blue = n / 2 ∧
    red = n / 5 ∧
    green = 7 ∧
    purple = n - (blue + red + green) ∧
    purple > 0) →
  (∀ (m : ℕ),
    m < n →
    ¬(∃ (blue red green purple : ℕ),
      blue = m / 2 ∧
      red = m / 5 ∧
      green = 7 ∧
      purple = m - (blue + red + green) ∧
      purple > 0)) →
  (n - (n / 2 + n / 5 + 7) = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_purple_marbles_l963_96387


namespace NUMINAMATH_CALUDE_distance_gable_to_citadel_l963_96333

/-- The distance from the point (1600, 1200) to the origin (0, 0) on a complex plane is 2000. -/
theorem distance_gable_to_citadel : 
  Complex.abs (Complex.mk 1600 1200) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_distance_gable_to_citadel_l963_96333


namespace NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_for_x_squared_equals_one_l963_96365

theorem x_equals_one_sufficient_not_necessary_for_x_squared_equals_one :
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧
  (∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) →
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧
  ¬(∀ x : ℝ, x^2 = 1 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_x_equals_one_sufficient_not_necessary_for_x_squared_equals_one_l963_96365


namespace NUMINAMATH_CALUDE_jinas_mascots_l963_96326

/-- The number of mascots Jina has -/
def total_mascots (original_teddies bunny_to_teddy_ratio koala_bears additional_teddies_per_bunny : ℕ) : ℕ :=
  let bunnies := original_teddies * bunny_to_teddy_ratio
  let additional_teddies := bunnies * additional_teddies_per_bunny
  original_teddies + bunnies + koala_bears + additional_teddies

/-- Theorem stating the total number of mascots Jina has -/
theorem jinas_mascots : total_mascots 5 3 1 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_jinas_mascots_l963_96326


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l963_96320

theorem arithmetic_sequence_difference (a b c : ℚ) : 
  (∃ d : ℚ, d = (9 - 2) / 4 ∧ 
             a = 2 + d ∧ 
             b = 2 + 2*d ∧ 
             c = 2 + 3*d ∧ 
             9 = 2 + 4*d) → 
  c - a = 3.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l963_96320


namespace NUMINAMATH_CALUDE_total_running_time_l963_96322

/-- Represents the running data for a single day -/
structure DailyRun where
  distance : ℕ
  basePace : ℕ
  additionalTime : ℕ

/-- Calculates the total time for a single run -/
def runTime (run : DailyRun) : ℕ :=
  run.distance * (run.basePace + run.additionalTime)

/-- The running data for each day of the week -/
def weeklyRuns : List DailyRun :=
  [
    { distance := 3, basePace := 10, additionalTime := 1 },  -- Monday
    { distance := 4, basePace := 9,  additionalTime := 1 },  -- Tuesday
    { distance := 6, basePace := 12, additionalTime := 0 },  -- Wednesday
    { distance := 8, basePace := 8,  additionalTime := 2 },  -- Thursday
    { distance := 3, basePace := 10, additionalTime := 0 }   -- Friday
  ]

/-- The theorem stating that the total running time for the week is 255 minutes -/
theorem total_running_time :
  (weeklyRuns.map runTime).sum = 255 := by
  sorry


end NUMINAMATH_CALUDE_total_running_time_l963_96322


namespace NUMINAMATH_CALUDE_rick_ironing_time_l963_96386

/-- Represents the rate at which Rick irons dress shirts per hour -/
def shirts_per_hour : ℕ := 4

/-- Represents the rate at which Rick irons dress pants per hour -/
def pants_per_hour : ℕ := 3

/-- Represents the number of hours Rick spent ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- Represents the total number of pieces of clothing Rick ironed -/
def total_pieces : ℕ := 27

/-- Proves that Rick spent 3 hours ironing dress shirts given the conditions -/
theorem rick_ironing_time :
  ∃ (h : ℕ), h * shirts_per_hour + hours_ironing_pants * pants_per_hour = total_pieces ∧ h = 3 :=
by sorry

end NUMINAMATH_CALUDE_rick_ironing_time_l963_96386


namespace NUMINAMATH_CALUDE_calculator_squared_key_l963_96324

theorem calculator_squared_key (n : ℕ) : (5 ^ (2 ^ n) > 10000) ↔ n ≥ 3 :=
  sorry

end NUMINAMATH_CALUDE_calculator_squared_key_l963_96324


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l963_96358

/-- Three points are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

/-- The theorem states that if the points (4,7), (0,k), and (-8,5) are collinear, then k = 19/3. -/
theorem collinear_points_k_value :
  collinear (4, 7) (0, k) (-8, 5) → k = 19/3 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l963_96358


namespace NUMINAMATH_CALUDE_f_m_plus_one_positive_l963_96393

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + x + a

theorem f_m_plus_one_positive (a m : ℝ) (ha : a > 0) (hm : f a m < 0) : f a (m + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_m_plus_one_positive_l963_96393


namespace NUMINAMATH_CALUDE_perfect_squares_between_100_and_500_l963_96395

theorem perfect_squares_between_100_and_500 : 
  (Finset.filter (fun n => 100 < n^2 ∧ n^2 < 500) (Finset.range 23)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_100_and_500_l963_96395


namespace NUMINAMATH_CALUDE_negation_equivalence_l963_96354

theorem negation_equivalence (a b : ℝ) : 
  ¬(a * b = 0 → a = 0 ∨ b = 0) ↔ (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l963_96354


namespace NUMINAMATH_CALUDE_probability_x_plus_y_less_than_5_l963_96373

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A point inside the square --/
structure PointInSquare (s : Square) where
  point : ℝ × ℝ
  inside : point.1 ≥ s.bottomLeft.1 ∧ point.1 ≤ s.topRight.1 ∧
           point.2 ≥ s.bottomLeft.2 ∧ point.2 ≤ s.topRight.2

/-- The probability of an event for a uniformly distributed point in the square --/
def probability (s : Square) (event : PointInSquare s → Prop) : ℝ :=
  sorry

theorem probability_x_plus_y_less_than_5 :
  let s : Square := ⟨(0, 0), (4, 4)⟩
  probability s (fun p => p.point.1 + p.point.2 < 5) = 29 / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_less_than_5_l963_96373


namespace NUMINAMATH_CALUDE_office_age_problem_l963_96319

theorem office_age_problem (total_persons : ℕ) (avg_age_all : ℕ) 
  (group1_persons : ℕ) (avg_age_group1 : ℕ) (group2_persons : ℕ) 
  (age_15th_person : ℕ) : 
  total_persons = 16 → 
  avg_age_all = 15 → 
  group1_persons = 5 → 
  avg_age_group1 = 14 → 
  group2_persons = 9 → 
  age_15th_person = 26 → 
  (avg_age_all * total_persons - avg_age_group1 * group1_persons - age_15th_person) / group2_persons = 16 := by
sorry

end NUMINAMATH_CALUDE_office_age_problem_l963_96319


namespace NUMINAMATH_CALUDE_largest_initial_number_prove_largest_initial_number_l963_96312

theorem largest_initial_number : ℕ → Prop :=
  fun n => n = 189 ∧
    ∃ (a b c d e : ℕ),
      n + a + b + c + d + e = 200 ∧
      a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
      ¬(n % a = 0) ∧ ¬(n % b = 0) ∧ ¬(n % c = 0) ∧ ¬(n % d = 0) ∧ ¬(n % e = 0) ∧
      ∀ m : ℕ, m > n →
        ¬∃ (a' b' c' d' e' : ℕ),
          m + a' + b' + c' + d' + e' = 200 ∧
          a' ≥ 2 ∧ b' ≥ 2 ∧ c' ≥ 2 ∧ d' ≥ 2 ∧ e' ≥ 2 ∧
          ¬(m % a' = 0) ∧ ¬(m % b' = 0) ∧ ¬(m % c' = 0) ∧ ¬(m % d' = 0) ∧ ¬(m % e' = 0)

theorem prove_largest_initial_number : ∃ n : ℕ, largest_initial_number n := by
  sorry

end NUMINAMATH_CALUDE_largest_initial_number_prove_largest_initial_number_l963_96312


namespace NUMINAMATH_CALUDE_tetrahedron_volume_formula_l963_96376

/-- A tetrahedron with an inscribed sphere. -/
structure TetrahedronWithInscribedSphere where
  R : ℝ  -- Radius of the inscribed sphere
  S₁ : ℝ  -- Area of face 1
  S₂ : ℝ  -- Area of face 2
  S₃ : ℝ  -- Area of face 3
  S₄ : ℝ  -- Area of face 4

/-- The volume of a tetrahedron with an inscribed sphere. -/
def volume (t : TetrahedronWithInscribedSphere) : ℝ :=
  t.R * (t.S₁ + t.S₂ + t.S₃ + t.S₄)

/-- Theorem: The volume of a tetrahedron with an inscribed sphere
    is equal to the radius of the inscribed sphere multiplied by
    the sum of the areas of its four faces. -/
theorem tetrahedron_volume_formula (t : TetrahedronWithInscribedSphere) :
  volume t = t.R * (t.S₁ + t.S₂ + t.S₃ + t.S₄) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_formula_l963_96376


namespace NUMINAMATH_CALUDE_average_of_combined_sets_l963_96309

theorem average_of_combined_sets (m n : ℕ) (a b : ℝ) :
  let sum_m := m * a
  let sum_n := n * b
  (sum_m + sum_n) / (m + n) = (a * m + b * n) / (m + n) := by
  sorry

end NUMINAMATH_CALUDE_average_of_combined_sets_l963_96309


namespace NUMINAMATH_CALUDE_stickers_given_to_lucy_l963_96394

/-- Given that Gary initially had 99 stickers, gave 26 stickers to Alex, 
    and had 31 stickers left afterwards, prove that Gary gave 42 stickers to Lucy. -/
theorem stickers_given_to_lucy (initial_stickers : ℕ) (stickers_to_alex : ℕ) (stickers_left : ℕ) :
  initial_stickers = 99 →
  stickers_to_alex = 26 →
  stickers_left = 31 →
  initial_stickers - stickers_to_alex - stickers_left = 42 :=
by sorry

end NUMINAMATH_CALUDE_stickers_given_to_lucy_l963_96394


namespace NUMINAMATH_CALUDE_largest_y_coordinate_l963_96369

theorem largest_y_coordinate (x y : ℝ) :
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_l963_96369


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l963_96398

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℝ), ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (2 * x^2 - 5 * x + 6) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1) ∧
    P = -6 ∧ Q = 8 ∧ R = -5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l963_96398


namespace NUMINAMATH_CALUDE_oplus_k_oplus_k_l963_96383

-- Define the ⊕ operation
def oplus (x y : ℝ) : ℝ := x^3 - 2*y + x

-- Theorem statement
theorem oplus_k_oplus_k (k : ℝ) : oplus k (oplus k k) = -k^3 + 3*k := by
  sorry

end NUMINAMATH_CALUDE_oplus_k_oplus_k_l963_96383


namespace NUMINAMATH_CALUDE_parallelogram_sides_sum_l963_96360

theorem parallelogram_sides_sum (x y : ℝ) : 
  (5 * x - 7 = 14) → 
  (3 * y + 4 = 8 * y - 3) → 
  x + y = 5.6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_sides_sum_l963_96360


namespace NUMINAMATH_CALUDE_fisherman_catch_l963_96334

/-- The number of fish caught by a fisherman -/
def total_fish (num_boxes : ℕ) (fish_per_box : ℕ) (fish_outside : ℕ) : ℕ :=
  num_boxes * fish_per_box + fish_outside

/-- Theorem stating the total number of fish caught by the fisherman -/
theorem fisherman_catch :
  total_fish 15 20 6 = 306 := by
  sorry

end NUMINAMATH_CALUDE_fisherman_catch_l963_96334


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_third_term_l963_96301

/-- An arithmetic sequence with a positive first term and a_1 * a_2 = -2 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 > 0 ∧ a 1 * a 2 = -2 ∧ ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def CommonDifference (a : ℕ → ℝ) : ℝ :=
  (a 2) - (a 1)

/-- The third term of an arithmetic sequence -/
def ThirdTerm (a : ℕ → ℝ) : ℝ :=
  a 3

theorem arithmetic_sequence_max_third_term
  (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  (∀ d : ℝ, CommonDifference a ≤ d → ThirdTerm a ≤ ThirdTerm (fun n ↦ a 1 + (n - 1) * d)) →
  CommonDifference a = -3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_third_term_l963_96301


namespace NUMINAMATH_CALUDE_jonas_current_socks_l963_96374

-- Define the wardrobe items
def shoes : ℕ := 5
def pants : ℕ := 10
def tshirts : ℕ := 10
def socks_to_buy : ℕ := 35

-- Define the function to calculate individual items
def individual_items (socks : ℕ) : ℕ :=
  2 * shoes + 2 * pants + tshirts + 2 * socks

-- Theorem to prove
theorem jonas_current_socks :
  ∃ current_socks : ℕ,
    individual_items (current_socks + socks_to_buy) = 2 * individual_items current_socks ∧
    current_socks = 15 := by
  sorry


end NUMINAMATH_CALUDE_jonas_current_socks_l963_96374


namespace NUMINAMATH_CALUDE_train_speed_l963_96311

/-- Proves that a train of length 480 meters crossing a telegraph post in 16 seconds has a speed of 108 km/h -/
theorem train_speed (train_length : Real) (crossing_time : Real) (speed_kmh : Real) : 
  train_length = 480 ∧ 
  crossing_time = 16 ∧ 
  speed_kmh = (train_length / crossing_time) * 3.6 → 
  speed_kmh = 108 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l963_96311


namespace NUMINAMATH_CALUDE_expression_evaluation_l963_96362

theorem expression_evaluation : (60 + 5 * 12) / (Real.sqrt 180 / 3) ^ 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l963_96362


namespace NUMINAMATH_CALUDE_negation_of_perpendicular_plane_l963_96361

-- Define the concept of a line
variable (Line : Type)

-- Define the concept of a plane
variable (Plane : Type)

-- Define what it means for a plane to be perpendicular to a line
variable (perpendicular : Plane → Line → Prop)

-- State the theorem
theorem negation_of_perpendicular_plane :
  (¬ ∀ l : Line, ∃ α : Plane, perpendicular α l) ↔ 
  (∃ l : Line, ∀ α : Plane, ¬ perpendicular α l) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_perpendicular_plane_l963_96361


namespace NUMINAMATH_CALUDE_K2Cr2O7_molecular_weight_l963_96302

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (K_atoms Cr_atoms O_atoms : ℕ) (K_weight Cr_weight O_weight : ℝ) : ℝ :=
  K_atoms * K_weight + Cr_atoms * Cr_weight + O_atoms * O_weight

/-- The molecular weight of the compound K₂Cr₂O₇ is 294.192 g/mol -/
theorem K2Cr2O7_molecular_weight : 
  molecular_weight 2 2 7 39.10 51.996 16.00 = 294.192 := by
  sorry

#eval molecular_weight 2 2 7 39.10 51.996 16.00

end NUMINAMATH_CALUDE_K2Cr2O7_molecular_weight_l963_96302


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l963_96363

theorem jerrys_action_figures (initial : ℕ) : 
  (initial + 2 - 7 = 10) → initial = 15 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l963_96363


namespace NUMINAMATH_CALUDE_hash_3_8_l963_96389

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- Theorem to prove
theorem hash_3_8 : hash 3 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_hash_3_8_l963_96389


namespace NUMINAMATH_CALUDE_g_of_3_eq_15_l963_96325

/-- A function g satisfying the given conditions -/
def g (x : ℝ) : ℝ := sorry

/-- The theorem stating that g(3) = 15 -/
theorem g_of_3_eq_15 (h1 : g 1 = 7) (h2 : g 2 = 11) 
  (h3 : ∃ (c d : ℝ), ∀ x, g x = c * x + d * x + 3) : 
  g 3 = 15 := by sorry

end NUMINAMATH_CALUDE_g_of_3_eq_15_l963_96325


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l963_96385

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l963_96385


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l963_96388

theorem quadratic_equation_solution :
  ∀ x : ℝ, (x - 6) * (x + 2) = 0 ↔ x = 6 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l963_96388


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_inequality_l963_96337

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |4*x - a| + a^2 - 4*a

-- Define the function g
def g (x : ℝ) : ℝ := |x - 1|

-- Theorem for part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | -2 ≤ f 1 x ∧ f 1 x ≤ 4} = 
  {x : ℝ | -3/2 ≤ x ∧ x ≤ 0} ∪ {x : ℝ | 1/2 ≤ x ∧ x ≤ 2} :=
by sorry

-- Theorem for part 2
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x : ℝ, f a x - 4 * g x ≤ 6} = 
  {a : ℝ | (5 - Real.sqrt 33) / 2 ≤ a ∧ a ≤ 5} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_inequality_l963_96337


namespace NUMINAMATH_CALUDE_parabola_point_k_value_l963_96318

/-- Given that the point (3,0) lies on the parabola y = 2x^2 + (k+2)x - k, prove that k = -12 -/
theorem parabola_point_k_value :
  ∀ k : ℝ, (2 * 3^2 + (k + 2) * 3 - k = 0) → k = -12 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_k_value_l963_96318


namespace NUMINAMATH_CALUDE_fraction_subtraction_l963_96305

theorem fraction_subtraction : (5/6 : ℚ) + (1/4 : ℚ) - (2/3 : ℚ) = (5/12 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l963_96305


namespace NUMINAMATH_CALUDE_regular_triangle_on_hyperbola_coordinates_l963_96391

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define a point on the hyperbola
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

-- Define the branches of the hyperbola
def on_positive_branch (p : PointOnHyperbola) : Prop := p.x > 0
def on_negative_branch (p : PointOnHyperbola) : Prop := p.x < 0

-- Define a regular triangle on the hyperbola
structure RegularTriangleOnHyperbola where
  P : PointOnHyperbola
  Q : PointOnHyperbola
  R : PointOnHyperbola
  is_regular : True  -- We assume this property without proving it

-- Theorem statement
theorem regular_triangle_on_hyperbola_coordinates 
  (t : RegularTriangleOnHyperbola)
  (h_P : t.P.x = -1 ∧ t.P.y = 1)
  (h_P_branch : on_negative_branch t.P)
  (h_Q_branch : on_positive_branch t.Q)
  (h_R_branch : on_positive_branch t.R) :
  ((t.Q.x = 2 - Real.sqrt 3 ∧ t.Q.y = 2 + Real.sqrt 3) ∧
   (t.R.x = 2 + Real.sqrt 3 ∧ t.R.y = 2 - Real.sqrt 3)) ∨
  ((t.Q.x = 2 + Real.sqrt 3 ∧ t.Q.y = 2 - Real.sqrt 3) ∧
   (t.R.x = 2 - Real.sqrt 3 ∧ t.R.y = 2 + Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_regular_triangle_on_hyperbola_coordinates_l963_96391


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l963_96381

theorem quiz_competition_participants (total : ℕ) 
  (h1 : (total : ℝ) * (1 - 0.6) * 0.25 = 16) : total = 160 := by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l963_96381


namespace NUMINAMATH_CALUDE_circle_condition_l963_96338

theorem circle_condition (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*x + a = 0) →
  (∃ (h k r : ℝ), ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2) →
  a < 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l963_96338


namespace NUMINAMATH_CALUDE_positive_A_value_l963_96327

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h1 : hash A 7 = 130) (h2 : A > 0) : A = 9 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l963_96327


namespace NUMINAMATH_CALUDE_average_salary_proof_l963_96313

def salary_a : ℕ := 8000
def salary_b : ℕ := 5000
def salary_c : ℕ := 15000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

def total_salary : ℕ := salary_a + salary_b + salary_c + salary_d + salary_e
def num_individuals : ℕ := 5

theorem average_salary_proof :
  total_salary / num_individuals = 9000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_proof_l963_96313


namespace NUMINAMATH_CALUDE_function_properties_l963_96352

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - 9*x + 11

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x - 9

theorem function_properties :
  ∃ (a : ℝ),
    (f_derivative a 1 = -12) ∧
    (a = 3) ∧
    (∀ x, f a x ≤ 16) ∧
    (∃ x, f a x = 16) ∧
    (∀ x, f a x ≥ -16) ∧
    (∃ x, f a x = -16) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l963_96352


namespace NUMINAMATH_CALUDE_can_find_genuine_coin_l963_96300

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a group of coins -/
structure CoinGroup where
  size : Nat
  counterfeit : Nat

/-- Represents the state of coins -/
structure CoinState where
  total : Nat
  counterfeit : Nat

/-- Represents a weighing operation -/
def weigh (left right : CoinGroup) : WeighResult :=
  sorry

/-- Represents the process of finding a genuine coin -/
def findGenuineCoin (state : CoinState) : Prop :=
  ∃ (g1 g2 g3 : CoinGroup),
    g1.size + g2.size + g3.size = state.total ∧
    g1.counterfeit + g2.counterfeit + g3.counterfeit = state.counterfeit ∧
    (∃ (result : WeighResult),
      result = weigh g1 g2 ∧
      (result = WeighResult.Equal →
        ∃ (c1 c2 : CoinGroup),
          c1.size = 1 ∧ c2.size = 1 ∧
          c1.size + c2.size ≤ g3.size ∧
          (weigh c1 c2 = WeighResult.Equal ∨
           weigh c1 c2 = WeighResult.LeftHeavier ∨
           weigh c1 c2 = WeighResult.RightHeavier)) ∧
      ((result = WeighResult.LeftHeavier ∨ result = WeighResult.RightHeavier) →
        ∃ (c1 c2 : CoinGroup),
          c1.size = 1 ∧ c2.size = 1 ∧
          (c1.size ≤ g1.size ∧ c2.size ≤ g2.size) ∧
          (weigh c1 c2 = WeighResult.Equal ∨
           weigh c1 c2 = WeighResult.LeftHeavier ∨
           weigh c1 c2 = WeighResult.RightHeavier)))

theorem can_find_genuine_coin (state : CoinState)
  (h1 : state.total = 100)
  (h2 : state.counterfeit = 4)
  (h3 : state.counterfeit < state.total) :
  findGenuineCoin state :=
  sorry

end NUMINAMATH_CALUDE_can_find_genuine_coin_l963_96300


namespace NUMINAMATH_CALUDE_characterization_of_M_l963_96323

/-- S(n) represents the sum of digits of n -/
def S (n : ℕ) : ℕ := sorry

/-- M satisfies the given property -/
def satisfies_property (M : ℕ) : Prop :=
  M > 0 ∧ ∀ k : ℕ, 0 < k ∧ k ≤ M → S (M * k) = S M

/-- Main theorem -/
theorem characterization_of_M :
  ∀ M : ℕ, satisfies_property M ↔ ∃ n : ℕ, n > 0 ∧ M = 10^n - 1 :=
sorry

end NUMINAMATH_CALUDE_characterization_of_M_l963_96323


namespace NUMINAMATH_CALUDE_bill_bouquets_theorem_l963_96349

/-- Represents the number of roses in a bouquet Bill buys -/
def roses_per_bouquet_buy : ℕ := 7

/-- Represents the number of roses in a bouquet Bill sells -/
def roses_per_bouquet_sell : ℕ := 5

/-- Represents the price of a bouquet (both buying and selling) -/
def price_per_bouquet : ℕ := 20

/-- Represents the target profit -/
def target_profit : ℕ := 1000

/-- Calculates the number of bouquets Bill needs to buy to earn the target profit -/
def bouquets_to_buy : ℕ :=
  let bouquets_per_operation := roses_per_bouquet_sell
  let profit_per_operation := price_per_bouquet * bouquets_per_operation - price_per_bouquet * roses_per_bouquet_buy / roses_per_bouquet_sell
  let operations_needed := target_profit / profit_per_operation
  operations_needed * roses_per_bouquet_buy / roses_per_bouquet_sell

theorem bill_bouquets_theorem : bouquets_to_buy = 125 := by
  sorry

end NUMINAMATH_CALUDE_bill_bouquets_theorem_l963_96349


namespace NUMINAMATH_CALUDE_sequence_always_terminates_l963_96328

def last_digit (n : ℕ) : ℕ := n % 10

def remove_last_digit (n : ℕ) : ℕ := n / 10

def next_term (n : ℕ) : ℕ :=
  if n ≤ 5 then n
  else if last_digit n ≤ 5 then remove_last_digit n
  else 9 * n

def sequence_terminates (a₀ : ℕ) : Prop :=
  ∃ n : ℕ, (Nat.iterate next_term n a₀) ≤ 5

theorem sequence_always_terminates (a₀ : ℕ) : sequence_terminates a₀ := by
  sorry

#check sequence_always_terminates

end NUMINAMATH_CALUDE_sequence_always_terminates_l963_96328


namespace NUMINAMATH_CALUDE_pauls_money_duration_l963_96377

/-- 
Given Paul's earnings and weekly spending, prove how long the money will last.
-/
theorem pauls_money_duration (lawn_mowing : ℕ) (weed_eating : ℕ) (weekly_spending : ℕ) 
  (h1 : lawn_mowing = 44)
  (h2 : weed_eating = 28)
  (h3 : weekly_spending = 9) :
  (lawn_mowing + weed_eating) / weekly_spending = 8 := by
  sorry

end NUMINAMATH_CALUDE_pauls_money_duration_l963_96377


namespace NUMINAMATH_CALUDE_dogs_food_consumption_l963_96380

/-- The amount of dog food eaten by one dog per day -/
def dog_food_per_day : ℝ := 0.12

/-- The number of dogs -/
def num_dogs : ℕ := 2

/-- The total amount of dog food eaten by all dogs per day -/
def total_dog_food : ℝ := dog_food_per_day * num_dogs

theorem dogs_food_consumption :
  total_dog_food = 0.24 := by sorry

end NUMINAMATH_CALUDE_dogs_food_consumption_l963_96380


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_shifted_roots_l963_96367

theorem sum_of_reciprocals_shifted_roots (a b c : ℂ) : 
  (a^3 - 2*a + 4 = 0) → 
  (b^3 - 2*b + 4 = 0) → 
  (c^3 - 2*c + 4 = 0) → 
  (1/(a-2) + 1/(b-2) + 1/(c-2) = -5/4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_shifted_roots_l963_96367


namespace NUMINAMATH_CALUDE_jan_cable_purchase_l963_96348

theorem jan_cable_purchase (section_length : ℕ) (sections_on_hand : ℕ) : 
  section_length = 25 →
  sections_on_hand = 15 →
  (4 : ℚ) * sections_on_hand = 3 * (2 * sections_on_hand) →
  (4 : ℚ) * sections_on_hand * section_length = 1000 := by
  sorry

end NUMINAMATH_CALUDE_jan_cable_purchase_l963_96348


namespace NUMINAMATH_CALUDE_smallest_number_with_given_properties_l963_96351

theorem smallest_number_with_given_properties : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → ¬(8 ∣ m ∧ m % 2 = 1 ∧ m % 3 = 1 ∧ m % 4 = 1 ∧ m % 5 = 1 ∧ m % 7 = 1)) ∧ 
  (8 ∣ n) ∧ 
  (n % 2 = 1) ∧ 
  (n % 3 = 1) ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 7 = 1) ∧ 
  n = 7141 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_properties_l963_96351


namespace NUMINAMATH_CALUDE_complex_number_additive_inverse_parts_l963_96317

theorem complex_number_additive_inverse_parts (b : ℝ) : 
  let z := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_additive_inverse_parts_l963_96317


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l963_96372

theorem sum_of_a_and_b (a b : ℕ+) (h : a.val^2 - b.val^4 = 2009) : a.val + b.val = 47 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l963_96372


namespace NUMINAMATH_CALUDE_power_calculation_l963_96332

theorem power_calculation : 16^16 * 8^8 / 4^32 = 2^24 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l963_96332


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l963_96339

theorem correct_average_after_error_correction 
  (n : Nat) 
  (initial_average : ℚ) 
  (wrong_number correct_number : ℚ) :
  n = 10 →
  initial_average = 5 →
  wrong_number = 26 →
  correct_number = 36 →
  (n : ℚ) * initial_average + (correct_number - wrong_number) = n * 6 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l963_96339


namespace NUMINAMATH_CALUDE_train_length_proof_l963_96315

/-- Proves that the length of a train is 260 meters, given its speed and the time it takes to cross a platform of known length. -/
theorem train_length_proof (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (1000 / 3600) →
  platform_length = 260 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 260 :=
by sorry

end NUMINAMATH_CALUDE_train_length_proof_l963_96315


namespace NUMINAMATH_CALUDE_school_play_chairs_l963_96330

theorem school_play_chairs (chairs_per_row : ℕ) (unoccupied_seats : ℕ) (occupied_seats : ℕ) :
  chairs_per_row = 20 →
  unoccupied_seats = 10 →
  occupied_seats = 790 →
  (occupied_seats + unoccupied_seats) / chairs_per_row = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_school_play_chairs_l963_96330


namespace NUMINAMATH_CALUDE_triangular_number_formula_l963_96340

/-- The triangular number sequence -/
def triangular_number : ℕ → ℕ
| 0 => 0
| (n + 1) => triangular_number n + n + 1

/-- Theorem: The nth triangular number is equal to n(n+1)/2 -/
theorem triangular_number_formula (n : ℕ) :
  triangular_number n = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_formula_l963_96340


namespace NUMINAMATH_CALUDE_largest_equilateral_triangle_l963_96382

/-- Represents a square piece of paper -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- The folding process that creates the largest equilateral triangle from a square -/
noncomputable def foldLargestTriangle (s : Square) : EquilateralTriangle :=
  sorry

/-- Theorem stating that the triangle produced by foldLargestTriangle is the largest possible -/
theorem largest_equilateral_triangle (s : Square) :
  ∀ t : EquilateralTriangle, t.side ≤ (foldLargestTriangle s).side :=
  sorry

end NUMINAMATH_CALUDE_largest_equilateral_triangle_l963_96382


namespace NUMINAMATH_CALUDE_quadrilaterals_on_circle_l963_96368

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of points on the circle -/
def num_points : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def vertices_per_quad : ℕ := 4

/-- The number of points to choose from after fixing two points -/
def remaining_points : ℕ := num_points - 2

/-- The number of additional vertices needed after fixing two points -/
def additional_vertices : ℕ := vertices_per_quad - 2

theorem quadrilaterals_on_circle :
  choose num_points vertices_per_quad - choose remaining_points additional_vertices = 450 := by
  sorry

end NUMINAMATH_CALUDE_quadrilaterals_on_circle_l963_96368
