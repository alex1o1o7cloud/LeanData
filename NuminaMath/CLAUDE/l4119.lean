import Mathlib

namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l4119_411939

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l4119_411939


namespace NUMINAMATH_CALUDE_function_properties_l4119_411958

/-- Given a function f(x) = x + m/x where f(1) = 5, this theorem proves:
    1. The value of m
    2. The parity of f
    3. The monotonicity of f on (2, +∞) -/
theorem function_properties (f : ℝ → ℝ) (m : ℝ) 
    (h1 : ∀ x ≠ 0, f x = x + m / x)
    (h2 : f 1 = 5) :
    (m = 4) ∧ 
    (∀ x ≠ 0, f (-x) = -f x) ∧
    (∀ x₁ x₂, 2 < x₁ → x₁ < x₂ → f x₁ < f x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l4119_411958


namespace NUMINAMATH_CALUDE_pet_store_cats_sold_l4119_411984

theorem pet_store_cats_sold (dogs : ℕ) (cats : ℕ) : 
  cats = 3 * dogs →
  cats = 2 * (dogs + 8) →
  cats = 48 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cats_sold_l4119_411984


namespace NUMINAMATH_CALUDE_original_price_proof_l4119_411959

/-- Given an item sold at a 20% loss with a selling price of 1040, 
    prove that the original price of the item was 1300. -/
theorem original_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1040)
  (h2 : loss_percentage = 20) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 - loss_percentage / 100) ∧ 
    original_price = 1300 :=
by
  sorry

end NUMINAMATH_CALUDE_original_price_proof_l4119_411959


namespace NUMINAMATH_CALUDE_calvins_weight_loss_l4119_411963

/-- Calvin's weight loss problem -/
theorem calvins_weight_loss 
  (initial_weight : ℕ) 
  (weight_loss_per_month : ℕ) 
  (months : ℕ) 
  (h1 : initial_weight = 250)
  (h2 : weight_loss_per_month = 8)
  (h3 : months = 12) :
  initial_weight - (weight_loss_per_month * months) = 154 :=
by sorry

end NUMINAMATH_CALUDE_calvins_weight_loss_l4119_411963


namespace NUMINAMATH_CALUDE_star_difference_sum_l4119_411983

/-- The ⋆ operation for real numbers -/
def star (a b : ℝ) : ℝ := a^2 - b

/-- Theorem stating the result of (x - y) ⋆ (x + y) -/
theorem star_difference_sum (x y : ℝ) : 
  star (x - y) (x + y) = x^2 - x - 2*x*y + y^2 - y := by
  sorry

end NUMINAMATH_CALUDE_star_difference_sum_l4119_411983


namespace NUMINAMATH_CALUDE_sum_of_x_values_l4119_411947

theorem sum_of_x_values (N : ℝ) (h : N ≥ 0) : 
  ∃ x₁ x₂ : ℝ, |x₁ - 25| = N ∧ |x₂ - 25| = N ∧ x₁ + x₂ = 50 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_values_l4119_411947


namespace NUMINAMATH_CALUDE_smallest_power_l4119_411960

theorem smallest_power : 2^55 < 3^44 ∧ 2^55 < 5^33 ∧ 2^55 < 6^22 := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_l4119_411960


namespace NUMINAMATH_CALUDE_alternating_sum_squares_l4119_411981

/-- The sum of squares with alternating signs in pairs from 1 to 120 -/
def M : ℕ → ℕ
| 0 => 0
| (n + 1) => if n % 4 < 2
              then M n + (120 - n + 1)^2
              else M n - (120 - n + 1)^2

theorem alternating_sum_squares : M 120 = 14520 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_squares_l4119_411981


namespace NUMINAMATH_CALUDE_perpendicular_iff_m_eq_neg_one_l4119_411906

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line: x + y = 0 -/
def line1 : Line := ⟨1, 1, 0⟩

/-- The second line: x + my = 0 -/
def line2 (m : ℝ) : Line := ⟨1, m, 0⟩

/-- Theorem: The lines x+y=0 and x+my=0 are perpendicular if and only if m=-1 -/
theorem perpendicular_iff_m_eq_neg_one :
  ∀ m : ℝ, perpendicular line1 (line2 m) ↔ m = -1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_iff_m_eq_neg_one_l4119_411906


namespace NUMINAMATH_CALUDE_divisibility_property_l4119_411930

theorem divisibility_property (n : ℕ) : 
  (n - 1) ∣ (n^n - 7*n + 5*n^2024 + 3*n^2 - 2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l4119_411930


namespace NUMINAMATH_CALUDE_john_number_is_13_l4119_411986

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem john_number_is_13 :
  ∃! x : ℕ, is_two_digit x ∧
    92 ≤ switch_digits (4 * x + 17) ∧
    switch_digits (4 * x + 17) ≤ 96 ∧
    x = 13 :=
by sorry

end NUMINAMATH_CALUDE_john_number_is_13_l4119_411986


namespace NUMINAMATH_CALUDE_halloween_candy_problem_l4119_411900

theorem halloween_candy_problem (debby_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ) :
  debby_candy = 32 →
  eaten_candy = 35 →
  remaining_candy = 39 →
  ∃ (sister_candy : ℕ), 
    debby_candy + sister_candy = eaten_candy + remaining_candy ∧
    sister_candy = 42 :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_problem_l4119_411900


namespace NUMINAMATH_CALUDE_M_properties_M_remainder_l4119_411977

/-- The greatest integer multiple of 16 with no repeated digits -/
def M : ℕ :=
  sorry

/-- Predicate to check if a natural number has no repeated digits -/
def has_no_repeated_digits (n : ℕ) : Prop :=
  sorry

theorem M_properties :
  M % 16 = 0 ∧
  has_no_repeated_digits M ∧
  ∀ n : ℕ, n % 16 = 0 → has_no_repeated_digits n → n ≤ M :=
sorry

theorem M_remainder :
  M % 1000 = 864 :=
sorry

end NUMINAMATH_CALUDE_M_properties_M_remainder_l4119_411977


namespace NUMINAMATH_CALUDE_school_enrollment_problem_l4119_411972

theorem school_enrollment_problem (x y : ℝ) : 
  x + y = 4000 →
  0.07 * x - 0.03 * y = 40 →
  y = 2400 :=
by
  sorry

end NUMINAMATH_CALUDE_school_enrollment_problem_l4119_411972


namespace NUMINAMATH_CALUDE_probability_shortest_diagonal_decagon_l4119_411954

/-- The number of sides in a regular decagon -/
def n : ℕ := 10

/-- The total number of diagonals in a regular decagon -/
def total_diagonals : ℕ := n * (n - 3) / 2

/-- The number of shortest diagonals in a regular decagon -/
def shortest_diagonals : ℕ := n

/-- The probability of selecting one of the shortest diagonals -/
def probability : ℚ := shortest_diagonals / total_diagonals

theorem probability_shortest_diagonal_decagon :
  probability = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_shortest_diagonal_decagon_l4119_411954


namespace NUMINAMATH_CALUDE_pens_count_l4119_411905

/-- Given a ratio of pens to markers as 2:5 and 25 markers, prove that the number of pens is 10 -/
theorem pens_count (markers : ℕ) (h1 : markers = 25) : 
  (2 : ℚ) / 5 * markers = 10 := by
  sorry

#check pens_count

end NUMINAMATH_CALUDE_pens_count_l4119_411905


namespace NUMINAMATH_CALUDE_function_difference_l4119_411991

theorem function_difference (k : ℝ) : 
  let f (x : ℝ) := 4 * x^2 - 3 * x + 5
  let g (x : ℝ) := 2 * x^2 - k * x + 1
  (f 10 - g 10 = 20) → k = -21.4 := by
sorry

end NUMINAMATH_CALUDE_function_difference_l4119_411991


namespace NUMINAMATH_CALUDE_brass_price_is_correct_l4119_411945

/-- The price of copper in dollars per pound -/
def copper_price : ℚ := 65 / 100

/-- The price of zinc in dollars per pound -/
def zinc_price : ℚ := 30 / 100

/-- The total weight of brass in pounds -/
def total_weight : ℚ := 70

/-- The amount of copper used in pounds -/
def copper_weight : ℚ := 30

/-- The amount of zinc used in pounds -/
def zinc_weight : ℚ := total_weight - copper_weight

/-- The selling price of brass per pound -/
def brass_price : ℚ := (copper_price * copper_weight + zinc_price * zinc_weight) / total_weight

theorem brass_price_is_correct : brass_price = 45 / 100 := by
  sorry

end NUMINAMATH_CALUDE_brass_price_is_correct_l4119_411945


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l4119_411924

theorem complex_number_quadrant : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i)^2 / (1 + i)
  (z.re < 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l4119_411924


namespace NUMINAMATH_CALUDE_cos_2alpha_eq_neg_one_seventh_l4119_411993

theorem cos_2alpha_eq_neg_one_seventh (α : Real) 
  (h : 3 * Real.sin (α - Real.pi/6) = Real.sin (α + Real.pi/6)) : 
  Real.cos (2 * α) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_eq_neg_one_seventh_l4119_411993


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_l4119_411928

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a2 (a : ℕ → ℚ) :
  geometric_sequence a →
  a 1 = 1/4 →
  a 3 * a 5 = 4 * (a 4 - 1) →
  a 2 = 1/8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_l4119_411928


namespace NUMINAMATH_CALUDE_right_triangle_area_l4119_411938

/-- Given a right triangle with circumscribed circle radius R and inscribed circle radius r,
    prove that its area is r(2R + r). -/
theorem right_triangle_area (R r : ℝ) (h_positive_R : R > 0) (h_positive_r : r > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    c = 2 * R ∧
    r = (a + b - c) / 2 ∧
    a^2 + b^2 = c^2 ∧
    (1/2) * a * b = r * (2 * R + r) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l4119_411938


namespace NUMINAMATH_CALUDE_unique_base_twelve_l4119_411964

/-- Convert a base-b number to its decimal representation -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + b * acc) 0

/-- Check if all digits in a list are less than a given base -/
def valid_digits (digits : List Nat) (b : Nat) : Prop :=
  digits.all (· < b)

/-- The main theorem statement -/
theorem unique_base_twelve :
  ∃! b : Nat, 
    b > 1 ∧
    valid_digits [3, 0, 6] b ∧
    valid_digits [4, 2, 9] b ∧
    valid_digits [7, 4, 3] b ∧
    to_decimal [3, 0, 6] b + to_decimal [4, 2, 9] b = to_decimal [7, 4, 3] b :=
by
  sorry

end NUMINAMATH_CALUDE_unique_base_twelve_l4119_411964


namespace NUMINAMATH_CALUDE_profit_percentage_change_l4119_411985

def company_profits (revenue2008 : ℝ) : Prop :=
  let profit2008 := 0.1 * revenue2008
  let revenue2009 := 0.8 * revenue2008
  let profit2009 := 0.18 * revenue2009
  let revenue2010 := 1.25 * revenue2009
  let profit2010 := 0.15 * revenue2010
  let profit_change := (profit2010 - profit2008) / profit2008
  profit_change = 0.5

theorem profit_percentage_change (revenue2008 : ℝ) (h : revenue2008 > 0) :
  company_profits revenue2008 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_change_l4119_411985


namespace NUMINAMATH_CALUDE_jan_skips_after_training_l4119_411904

/-- The number of skips Jan does in 5 minutes after doubling her initial speed -/
def total_skips (initial_speed : ℕ) (time : ℕ) : ℕ :=
  2 * initial_speed * time

/-- Theorem stating that Jan does 700 skips in 5 minutes after doubling her initial speed of 70 skips per minute -/
theorem jan_skips_after_training :
  total_skips 70 5 = 700 := by
  sorry

end NUMINAMATH_CALUDE_jan_skips_after_training_l4119_411904


namespace NUMINAMATH_CALUDE_fraction_difference_equals_one_minus_two_over_x_l4119_411919

theorem fraction_difference_equals_one_minus_two_over_x 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x - 1 / y = 1 - 2 / x :=
by sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_one_minus_two_over_x_l4119_411919


namespace NUMINAMATH_CALUDE_two_sided_iced_cubes_count_l4119_411902

/-- Represents a cube cake with icing -/
structure IcedCake where
  size : Nat
  hasTopIcing : Bool
  hasSideIcing : Bool
  hasVerticalStrip : Bool

/-- Counts the number of 1x1x1 cubes with exactly two iced sides -/
def countTwoSidedIcedCubes (cake : IcedCake) : Nat :=
  sorry

/-- Theorem stating that a 5x5x5 cake with specified icing has 27 two-sided iced cubes -/
theorem two_sided_iced_cubes_count (cake : IcedCake) :
  cake.size = 5 ∧ cake.hasTopIcing ∧ cake.hasSideIcing ∧ cake.hasVerticalStrip →
  countTwoSidedIcedCubes cake = 27 := by
  sorry

end NUMINAMATH_CALUDE_two_sided_iced_cubes_count_l4119_411902


namespace NUMINAMATH_CALUDE_third_studio_students_l4119_411978

theorem third_studio_students (total : ℕ) (first : ℕ) (second : ℕ) 
  (h_total : total = 376)
  (h_first : first = 110)
  (h_second : second = 135) :
  total - first - second = 131 := by
  sorry

end NUMINAMATH_CALUDE_third_studio_students_l4119_411978


namespace NUMINAMATH_CALUDE_puppies_given_sandy_friend_puppies_l4119_411909

/-- Given the initial number of puppies and the total number of puppies after receiving more,
    calculate the number of puppies Sandy's friend gave her. -/
theorem puppies_given (initial : ℝ) (total : ℕ) : ℝ :=
  total - initial

/-- Prove that the number of puppies Sandy's friend gave her is 4. -/
theorem sandy_friend_puppies : puppies_given 8 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_puppies_given_sandy_friend_puppies_l4119_411909


namespace NUMINAMATH_CALUDE_square_51_and_39_l4119_411917

theorem square_51_and_39 : 51^2 = 2601 ∧ 39^2 = 1521 := by
  -- Given: (a ± b)² = a² ± 2ab + b²
  sorry


end NUMINAMATH_CALUDE_square_51_and_39_l4119_411917


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l4119_411968

theorem simplify_trig_expression :
  Real.sqrt (1 + 2 * Real.sin (π - 2) * Real.cos (π - 2)) = Real.sin 2 - Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l4119_411968


namespace NUMINAMATH_CALUDE_grocery_to_gym_speed_l4119_411969

-- Define the constants
def distance_home_to_grocery : ℝ := 840
def distance_grocery_to_gym : ℝ := 480
def time_difference : ℝ := 40

-- Define the variables
variable (speed_home_to_grocery : ℝ)
variable (speed_grocery_to_gym : ℝ)
variable (time_home_to_grocery : ℝ)
variable (time_grocery_to_gym : ℝ)

-- Define the theorem
theorem grocery_to_gym_speed :
  speed_grocery_to_gym = 2 * speed_home_to_grocery ∧
  time_home_to_grocery = distance_home_to_grocery / speed_home_to_grocery ∧
  time_grocery_to_gym = distance_grocery_to_gym / speed_grocery_to_gym ∧
  time_home_to_grocery = time_grocery_to_gym + time_difference ∧
  speed_home_to_grocery > 0 →
  speed_grocery_to_gym = 30 :=
by sorry

end NUMINAMATH_CALUDE_grocery_to_gym_speed_l4119_411969


namespace NUMINAMATH_CALUDE_unique_integer_solution_l4119_411937

theorem unique_integer_solution (m : ℤ) : 
  (∃! (x : ℤ), |2*x - m| ≤ 1 ∧ x = 2) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l4119_411937


namespace NUMINAMATH_CALUDE_manny_marbles_l4119_411935

theorem manny_marbles (total : ℕ) (mario_ratio manny_ratio given : ℕ) : 
  total = 36 →
  mario_ratio = 4 →
  manny_ratio = 5 →
  given = 2 →
  (manny_ratio * (total / (mario_ratio + manny_ratio))) - given = 18 :=
by sorry

end NUMINAMATH_CALUDE_manny_marbles_l4119_411935


namespace NUMINAMATH_CALUDE_polynomial_sqrt_value_l4119_411992

theorem polynomial_sqrt_value (a₃ a₂ a₁ a₀ : ℝ) :
  let P : ℝ → ℝ := fun x ↦ x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀
  (P 1 = 1) → (P 2 = 2) → (P 3 = 3) → (P 4 = 4) →
  Real.sqrt (P 13 - 12) = 109 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sqrt_value_l4119_411992


namespace NUMINAMATH_CALUDE_moon_distance_scientific_notation_l4119_411914

/-- The average distance between the Earth and the Moon in kilometers -/
def moon_distance : ℝ := 384000

/-- Theorem stating that the moon distance in scientific notation is 3.84 × 10^5 -/
theorem moon_distance_scientific_notation : moon_distance = 3.84 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_moon_distance_scientific_notation_l4119_411914


namespace NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l4119_411994

theorem sufficient_condition_for_f_less_than_one 
  (a : ℝ) (h_a : a > 1) :
  ∀ x : ℝ, -1 < x ∧ x < 0 → (a * x + 2 * x) < 1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l4119_411994


namespace NUMINAMATH_CALUDE_trains_at_start_after_2016_minutes_all_trains_at_start_after_2016_minutes_l4119_411975

/-- Represents a metro line with a given round trip time -/
structure MetroLine where
  roundTripTime : ℕ

/-- Represents the metro system of city N -/
structure MetroSystem where
  redLine : MetroLine
  blueLine : MetroLine
  greenLine : MetroLine

/-- Theorem stating that after 2016 minutes, all trains will be at their starting positions -/
theorem trains_at_start_after_2016_minutes (system : MetroSystem) 
  (h_red : system.redLine.roundTripTime = 14)
  (h_blue : system.blueLine.roundTripTime = 16)
  (h_green : system.greenLine.roundTripTime = 18) :
  2016 % system.redLine.roundTripTime = 0 ∧
  2016 % system.blueLine.roundTripTime = 0 ∧
  2016 % system.greenLine.roundTripTime = 0 := by
  sorry

/-- Function to check if a train is at its starting position after a given time -/
def isAtStartPosition (line : MetroLine) (time : ℕ) : Bool :=
  time % line.roundTripTime = 0

/-- Theorem stating that all trains are at their starting positions after 2016 minutes -/
theorem all_trains_at_start_after_2016_minutes (system : MetroSystem) 
  (h_red : system.redLine.roundTripTime = 14)
  (h_blue : system.blueLine.roundTripTime = 16)
  (h_green : system.greenLine.roundTripTime = 18) :
  isAtStartPosition system.redLine 2016 ∧
  isAtStartPosition system.blueLine 2016 ∧
  isAtStartPosition system.greenLine 2016 := by
  sorry

end NUMINAMATH_CALUDE_trains_at_start_after_2016_minutes_all_trains_at_start_after_2016_minutes_l4119_411975


namespace NUMINAMATH_CALUDE_investment_percentage_l4119_411980

/-- Proves that given a sum of 4000 Rs invested at 18% p.a. for two years yields 480 Rs more in interest
    than if it were invested at x% p.a. for the same period, x must equal 12%. -/
theorem investment_percentage (x : ℝ) : 
  (4000 * 18 * 2 / 100 - 4000 * x * 2 / 100 = 480) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_investment_percentage_l4119_411980


namespace NUMINAMATH_CALUDE_coin_flipping_theorem_l4119_411903

theorem coin_flipping_theorem :
  ∀ (initial_state : Fin 2015 → Bool),
  ∃! (final_state : Bool),
    (∀ (i : Fin 2015), final_state = initial_state i) ∨
    (∀ (i : Fin 2015), final_state ≠ initial_state i) :=
by
  sorry


end NUMINAMATH_CALUDE_coin_flipping_theorem_l4119_411903


namespace NUMINAMATH_CALUDE_largest_number_in_set_l4119_411943

theorem largest_number_in_set : 
  let S : Set ℝ := {0.01, 0.2, 0.03, 0.02, 0.1}
  ∀ x ∈ S, x ≤ 0.2 ∧ 0.2 ∈ S := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l4119_411943


namespace NUMINAMATH_CALUDE_sequence_prime_properties_l4119_411974

/-- The sequence a(n) = 3^(2^n) + 1 for n ≥ 1 -/
def a (n : ℕ) : ℕ := 3^(2^n) + 1

/-- The set of primes that do not divide any term of the sequence -/
def nondividing_primes : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∀ n, n ≥ 1 → ¬(p ∣ a n)}

/-- The set of primes that divide at least one term of the sequence -/
def dividing_primes : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ n, n ≥ 1 ∧ p ∣ a n}

theorem sequence_prime_properties :
  (Set.Infinite nondividing_primes) ∧ (Set.Infinite dividing_primes) := by
  sorry

end NUMINAMATH_CALUDE_sequence_prime_properties_l4119_411974


namespace NUMINAMATH_CALUDE_unknown_number_multiplication_l4119_411973

theorem unknown_number_multiplication (x : ℤ) : 
  55 = x + 45 - 62 → 7 * x = 504 := by
sorry

end NUMINAMATH_CALUDE_unknown_number_multiplication_l4119_411973


namespace NUMINAMATH_CALUDE_shells_in_morning_l4119_411942

theorem shells_in_morning (afternoon_shells : ℕ) (total_shells : ℕ) 
  (h1 : afternoon_shells = 324)
  (h2 : total_shells = 616) :
  total_shells - afternoon_shells = 292 := by
  sorry

end NUMINAMATH_CALUDE_shells_in_morning_l4119_411942


namespace NUMINAMATH_CALUDE_point_outside_circle_implies_a_range_l4119_411996

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 2 = 0

-- Define the condition for a point to be outside the circle
def point_outside_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 2 > 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -2 ∨ (2 < a ∧ a < 6)

-- Theorem statement
theorem point_outside_circle_implies_a_range :
  ∀ a : ℝ, point_outside_circle 1 1 a → a_range a :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_implies_a_range_l4119_411996


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l4119_411908

theorem consecutive_odd_numbers_sum (x : ℤ) : 
  (x % 2 = 1) →  -- x is odd
  ((x + 2) % 2 = 1) →  -- x + 2 is odd
  ((x + 4) % 2 = 1) →  -- x + 4 is odd
  (x + (x + 2) + (x + 4) = (x + 4) + 52) →  -- sum condition
  (x = 25) :=  -- conclusion
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l4119_411908


namespace NUMINAMATH_CALUDE_sqrt_x_plus_2_meaningful_l4119_411961

theorem sqrt_x_plus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 2) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_2_meaningful_l4119_411961


namespace NUMINAMATH_CALUDE_ratio_problem_l4119_411912

theorem ratio_problem (a b : ℝ) : 
  (a / b = 5 / 1) → (a = 45) → (b = 9) := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l4119_411912


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l4119_411927

def manuscript_cost (total_pages : ℕ) (once_revised : ℕ) (twice_revised : ℕ) (twice_revised_set : ℕ) 
                    (thrice_revised : ℕ) (thrice_revised_sets : ℕ) : ℕ :=
  let initial_cost := total_pages * 5
  let once_revised_cost := once_revised * 3
  let twice_revised_cost := (twice_revised - twice_revised_set) * 3 * 2 + twice_revised_set * 3 * 2 + 10
  let thrice_revised_cost := (thrice_revised - thrice_revised_sets * 10) * 3 * 3 + thrice_revised_sets * 15
  initial_cost + once_revised_cost + twice_revised_cost + thrice_revised_cost

theorem manuscript_typing_cost :
  manuscript_cost 200 50 70 10 40 2 = 1730 :=
by
  sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l4119_411927


namespace NUMINAMATH_CALUDE_square_perimeter_l4119_411944

/-- Given two squares I and II, where the diagonal of I is a+b and the area of II is twice the area of I, 
    the perimeter of II is 4(a+b) -/
theorem square_perimeter (a b : ℝ) : 
  let diagonal_I := a + b
  let area_I := (diagonal_I ^ 2) / 2
  let area_II := 2 * area_I
  let side_II := Real.sqrt area_II
  side_II * 4 = 4 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l4119_411944


namespace NUMINAMATH_CALUDE_population_ratio_l4119_411932

/-- The population ratio problem -/
theorem population_ratio 
  (pop_x pop_y pop_z : ℕ) 
  (h1 : pop_x = 7 * pop_y) 
  (h2 : pop_y = 2 * pop_z) : 
  pop_x / pop_z = 14 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_l4119_411932


namespace NUMINAMATH_CALUDE_complex_div_i_coords_l4119_411941

/-- The complex number (3+4i)/i corresponds to the point (4, -3) in the complex plane -/
theorem complex_div_i_coords : 
  let z : ℂ := (3 + 4*I) / I
  (z.re = 4 ∧ z.im = -3) :=
by sorry

end NUMINAMATH_CALUDE_complex_div_i_coords_l4119_411941


namespace NUMINAMATH_CALUDE_number_difference_l4119_411933

theorem number_difference (x y : ℕ) : 
  x + y = 34 → 
  y = 22 → 
  y - x = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_number_difference_l4119_411933


namespace NUMINAMATH_CALUDE_mitchell_unchewed_gum_l4119_411952

theorem mitchell_unchewed_gum (packets : ℕ) (pieces_per_packet : ℕ) (chewed_pieces : ℕ) 
  (h1 : packets = 8) 
  (h2 : pieces_per_packet = 7) 
  (h3 : chewed_pieces = 54) : 
  packets * pieces_per_packet - chewed_pieces = 2 := by
  sorry

end NUMINAMATH_CALUDE_mitchell_unchewed_gum_l4119_411952


namespace NUMINAMATH_CALUDE_square_difference_sqrt5_sqrt2_l4119_411951

theorem square_difference_sqrt5_sqrt2 :
  let x : ℝ := Real.sqrt 5
  let y : ℝ := Real.sqrt 2
  (x - y)^2 = 7 - 2 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_square_difference_sqrt5_sqrt2_l4119_411951


namespace NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achieved_l4119_411907

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 64) : x + y ≤ 8 := by
  sorry

theorem max_sum_achieved : ∃ (x y : ℤ), x^2 + y^2 = 64 ∧ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_max_sum_achieved_l4119_411907


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4119_411918

theorem min_value_quadratic (x : ℝ) :
  x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) →
  (x^2 + 2*x + 1) ≥ 0 ∧ ∃ y ∈ Set.Icc (-2 : ℝ) (2 : ℝ), y^2 + 2*y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4119_411918


namespace NUMINAMATH_CALUDE_band_arrangement_minimum_band_size_l4119_411929

theorem band_arrangement (n : ℕ) : n > 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0 → n ≥ 168 := by
  sorry

theorem minimum_band_size : ∃ n : ℕ, n > 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0 ∧ n = 168 := by
  sorry

end NUMINAMATH_CALUDE_band_arrangement_minimum_band_size_l4119_411929


namespace NUMINAMATH_CALUDE_simplify_absolute_expression_l4119_411920

theorem simplify_absolute_expression : |(-4^2 + 6^2 - 2)| = 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_expression_l4119_411920


namespace NUMINAMATH_CALUDE_line_properties_l4119_411967

/-- Represents a line in the form x = my + 1 --/
structure Line where
  m : ℝ

/-- The point (1, 0) is on the line --/
def point_on_line (l : Line) : Prop :=
  1 = l.m * 0 + 1

/-- The area of the triangle formed by the line and the axes when m = 2 --/
def triangle_area (l : Line) : Prop :=
  l.m = 2 → (1 / 2 : ℝ) * 1 * (1 / 2) = (1 / 4 : ℝ)

/-- Main theorem stating that both properties hold for any line of the form x = my + 1 --/
theorem line_properties (l : Line) : point_on_line l ∧ triangle_area l := by
  sorry

end NUMINAMATH_CALUDE_line_properties_l4119_411967


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_3_l4119_411925

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem statement
theorem subset_implies_a_geq_3 (a : ℝ) (h : A ⊆ B a) : a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_3_l4119_411925


namespace NUMINAMATH_CALUDE_complex_product_in_first_quadrant_l4119_411987

/-- The point corresponding to (1+3i)(3-i) is located in the first quadrant. -/
theorem complex_product_in_first_quadrant :
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_product_in_first_quadrant_l4119_411987


namespace NUMINAMATH_CALUDE_max_cross_section_area_l4119_411997

-- Define the prism
def prism_base_side_length : ℝ := 6

-- Define the cutting plane
def cutting_plane (x y z : ℝ) : Prop := 5 * x - 3 * y + 2 * z = 20

-- Define the cross-section area function
noncomputable def cross_section_area : ℝ := 9

-- Theorem statement
theorem max_cross_section_area :
  ∀ (area : ℝ),
    (∃ (x y z : ℝ), cutting_plane x y z ∧ 
      x^2 + y^2 ≤ (prism_base_side_length / 2)^2) →
    area ≤ cross_section_area :=
by sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l4119_411997


namespace NUMINAMATH_CALUDE_selection_theorem_l4119_411998

def number_of_students : ℕ := 10
def number_to_choose : ℕ := 3
def number_of_specific_students : ℕ := 2

def selection_ways : ℕ :=
  Nat.choose (number_of_students - 1) number_to_choose -
  Nat.choose (number_of_students - 1 - number_of_specific_students) number_to_choose

theorem selection_theorem :
  selection_ways = 49 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l4119_411998


namespace NUMINAMATH_CALUDE_cone_slant_height_l4119_411988

/-- Represents the properties of a cone --/
structure Cone where
  baseRadius : ℝ
  sectorAngle : ℝ
  slantHeight : ℝ

/-- Theorem: For a cone with base radius 6 cm and sector angle 240°, the slant height is 9 cm --/
theorem cone_slant_height (c : Cone) 
  (h1 : c.baseRadius = 6)
  (h2 : c.sectorAngle = 240) : 
  c.slantHeight = 9 := by
  sorry

#check cone_slant_height

end NUMINAMATH_CALUDE_cone_slant_height_l4119_411988


namespace NUMINAMATH_CALUDE_special_sequences_general_terms_l4119_411931

/-- Two sequences of positive real numbers satisfying specific conditions -/
structure SpecialSequences where
  a : ℕ → ℝ
  b : ℕ → ℝ
  a_pos : ∀ n, a n > 0
  b_pos : ∀ n, b n > 0
  arithmetic : ∀ n, 2 * b n = a n + a (n + 1)
  geometric : ∀ n, (a (n + 1))^2 = b n * b (n + 1)
  initial_a1 : a 1 = 1
  initial_b1 : b 1 = 2
  initial_a2 : a 2 = 3

/-- The general terms of the special sequences -/
theorem special_sequences_general_terms (s : SpecialSequences) :
    (∀ n, s.a n = n * (n + 1) / 2) ∧
    (∀ n, s.b n = (n + 1)^2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_special_sequences_general_terms_l4119_411931


namespace NUMINAMATH_CALUDE_hexagon_square_side_ratio_l4119_411922

/-- Given a regular hexagon and a square with the same perimeter,
    this theorem proves that the ratio of the side length of the hexagon
    to the side length of the square is 2/3. -/
theorem hexagon_square_side_ratio (perimeter : ℝ) (h s : ℝ)
  (hexagon_perimeter : 6 * h = perimeter)
  (square_perimeter : 4 * s = perimeter)
  (positive_perimeter : perimeter > 0) :
  h / s = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_square_side_ratio_l4119_411922


namespace NUMINAMATH_CALUDE_expression_simplification_l4119_411970

theorem expression_simplification (a x : ℝ) (h : a ≠ 3*x) :
  1.4 * (3*a^2 + 2*a*x - x^2) / ((3*x + a)*(a + x)) - 2 + 10 * (a*x - 3*x^2) / (a^2 + 9*x^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l4119_411970


namespace NUMINAMATH_CALUDE_lcm_problem_l4119_411911

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 18 m = 54) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 36 := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l4119_411911


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l4119_411948

/-- Given that a sum of money becomes 7/6 of itself in 2 years under simple interest,
    prove that the rate of interest per annum is 100/12. -/
theorem simple_interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  ∃ R : ℝ, R = 100 / 12 ∧ P * (1 + R * 2 / 100) = 7 / 6 * P :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l4119_411948


namespace NUMINAMATH_CALUDE_remaining_distance_l4119_411962

theorem remaining_distance (total_distance : ℝ) (father_fraction : ℝ) (mother_fraction : ℝ) :
  total_distance = 240 →
  father_fraction = 1/2 →
  mother_fraction = 3/8 →
  total_distance * (1 - father_fraction - mother_fraction) = 30 := by
sorry

end NUMINAMATH_CALUDE_remaining_distance_l4119_411962


namespace NUMINAMATH_CALUDE_lily_pad_coverage_l4119_411913

/-- Represents the number of days required for the lily pad patch to cover half the lake -/
def days_to_half_coverage : ℕ := 33

/-- Represents the number of days required for the lily pad patch to cover the entire lake -/
def days_to_full_coverage : ℕ := days_to_half_coverage + 1

/-- Theorem stating that the number of days to cover the entire lake is equal to
    the number of days to cover half the lake plus one -/
theorem lily_pad_coverage :
  days_to_full_coverage = days_to_half_coverage + 1 :=
by sorry

end NUMINAMATH_CALUDE_lily_pad_coverage_l4119_411913


namespace NUMINAMATH_CALUDE_bus_patrons_count_l4119_411934

/-- The number of patrons a golf cart can fit -/
def golf_cart_capacity : ℕ := 3

/-- The number of patrons who came in cars -/
def car_patrons : ℕ := 12

/-- The number of golf carts needed to transport all patrons -/
def golf_carts_needed : ℕ := 13

/-- The number of patrons who came from a bus -/
def bus_patrons : ℕ := golf_carts_needed * golf_cart_capacity - car_patrons

theorem bus_patrons_count : bus_patrons = 27 := by
  sorry

end NUMINAMATH_CALUDE_bus_patrons_count_l4119_411934


namespace NUMINAMATH_CALUDE_line_up_five_people_youngest_not_ends_l4119_411965

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def arrangements_with_youngest_at_ends (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem line_up_five_people_youngest_not_ends : 
  number_of_arrangements 5 - arrangements_with_youngest_at_ends 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_up_five_people_youngest_not_ends_l4119_411965


namespace NUMINAMATH_CALUDE_chocolate_boxes_price_l4119_411910

/-- The price of the small box of chocolates -/
def small_box_price : ℝ := 6

/-- The price of the large box of chocolates -/
def large_box_price : ℝ := small_box_price + 3

/-- The total cost of both boxes -/
def total_cost : ℝ := 15

theorem chocolate_boxes_price :
  small_box_price + large_box_price = total_cost ∧
  large_box_price = small_box_price + 3 ∧
  small_box_price = 6 := by
sorry

end NUMINAMATH_CALUDE_chocolate_boxes_price_l4119_411910


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l4119_411971

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) : 
  Real.sin (2 * α) = -17 / 18 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l4119_411971


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l4119_411946

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 - 2*m - 2025 = 0) → 
  (n^2 - 2*n - 2025 = 0) → 
  (m^2 - 3*m - n = 2023) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l4119_411946


namespace NUMINAMATH_CALUDE_triangle_area_l4119_411976

/-- Given a triangle with sides in ratio 5:12:13, perimeter 300 m, and angle 45° between shortest and middle sides, its area is 1500 * √2 m² -/
theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (5, 12, 13)) 
  (h_perimeter : a + b + c = 300) (h_angle : Real.cos (45 * π / 180) = b / (2 * a)) : 
  (1/2) * a * b * Real.sin (45 * π / 180) = 1500 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4119_411976


namespace NUMINAMATH_CALUDE_interval_constraint_l4119_411989

theorem interval_constraint (x : ℝ) : (2 < 4*x ∧ 4*x < 3 ∧ 2 < 5*x ∧ 5*x < 3) ↔ (1/2 < x ∧ x < 3/5) := by
  sorry

end NUMINAMATH_CALUDE_interval_constraint_l4119_411989


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l4119_411901

theorem polygon_interior_angles_sum (n : ℕ) : 
  (n - 2) * 180 = 900 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l4119_411901


namespace NUMINAMATH_CALUDE_equation_solution_l4119_411915

theorem equation_solution :
  ∃! x : ℚ, x ≠ -1 ∧ (x^2 + x + 1) / (x + 1) = x + 2 :=
by
  use (-1/2)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4119_411915


namespace NUMINAMATH_CALUDE_symmetric_decreasing_implies_l4119_411916

def is_symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def has_min_value_on (f : ℝ → ℝ) (a b : ℝ) (v : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → v ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = v)

theorem symmetric_decreasing_implies (f : ℝ → ℝ) :
  is_symmetric_about_origin f →
  is_decreasing_on f 1 5 →
  has_min_value_on f 1 5 3 →
  is_decreasing_on f (-5) (-1) ∧ has_min_value_on f (-5) (-1) (-3) :=
sorry

end NUMINAMATH_CALUDE_symmetric_decreasing_implies_l4119_411916


namespace NUMINAMATH_CALUDE_diff_sums_1500_l4119_411926

/-- Sum of the first n odd natural numbers -/
def sumOddNaturals (n : ℕ) : ℕ := n * n

/-- Sum of the first n even natural numbers -/
def sumEvenNaturals (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even natural numbers (each reduced by 3) 
    and the sum of the first n odd natural numbers -/
def diffSums (n : ℕ) : ℤ :=
  (sumEvenNaturals n - 3 * n : ℤ) - sumOddNaturals n

theorem diff_sums_1500 : diffSums 1500 = -2250 := by
  sorry

#eval diffSums 1500

end NUMINAMATH_CALUDE_diff_sums_1500_l4119_411926


namespace NUMINAMATH_CALUDE_barry_average_l4119_411936

def barry_yards : List ℕ := [98, 107, 85, 89, 91]

theorem barry_average : 
  (barry_yards.sum / barry_yards.length : ℚ) = 94 := by sorry

end NUMINAMATH_CALUDE_barry_average_l4119_411936


namespace NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l4119_411949

-- Define the expression under the cube root
def radicand (x y z : ℝ) : ℝ := 40 * x^5 * y^9 * z^14

-- Define the function to calculate the sum of exponents outside the radical
def sum_of_exponents_outside_radical (x y z : ℝ) : ℕ :=
  let simplified := (radicand x y z)^(1/3)
  -- The actual calculation of exponents would be implemented here
  -- For now, we'll use a placeholder
  8

-- Theorem statement
theorem sum_of_exponents_is_eight (x y z : ℝ) :
  sum_of_exponents_outside_radical x y z = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l4119_411949


namespace NUMINAMATH_CALUDE_honeycomb_briquettes_delivery_l4119_411979

theorem honeycomb_briquettes_delivery (total : ℕ) : 
  (3 * total) / 8 + 50 = (5 * ((total - ((3 * total) / 8 + 50)))) / 7 →
  total - ((3 * total) / 8 + 50) = 700 := by
  sorry

end NUMINAMATH_CALUDE_honeycomb_briquettes_delivery_l4119_411979


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l4119_411995

theorem polynomial_coefficient_sum (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (1 + m * x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = -63) →
  (m = 3 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l4119_411995


namespace NUMINAMATH_CALUDE_interest_rate_is_nine_percent_l4119_411940

/-- Calculates the simple interest rate given two loans and the total interest received. -/
def calculate_interest_rate (principal1 : ℚ) (time1 : ℚ) (principal2 : ℚ) (time2 : ℚ) (total_interest : ℚ) : ℚ :=
  (100 * total_interest) / (principal1 * time1 + principal2 * time2)

/-- Theorem stating that the interest rate is 9% for the given loan conditions. -/
theorem interest_rate_is_nine_percent :
  let principal1 : ℚ := 5000
  let time1 : ℚ := 2
  let principal2 : ℚ := 3000
  let time2 : ℚ := 4
  let total_interest : ℚ := 1980
  calculate_interest_rate principal1 time1 principal2 time2 total_interest = 9 := by
  sorry

#eval calculate_interest_rate 5000 2 3000 4 1980

end NUMINAMATH_CALUDE_interest_rate_is_nine_percent_l4119_411940


namespace NUMINAMATH_CALUDE_bucket_problem_l4119_411950

/-- Represents the capacity of a bucket --/
structure Bucket where
  capacity : ℝ
  sand : ℝ

/-- Proves that given the conditions of the bucket problem, 
    the initial fraction of Bucket B's capacity filled with sand is 3/8 --/
theorem bucket_problem (bucketA bucketB : Bucket) : 
  bucketA.sand = (1/4) * bucketA.capacity →
  bucketB.capacity = (1/2) * bucketA.capacity →
  bucketA.sand + bucketB.sand = (0.4375) * bucketA.capacity →
  bucketB.sand / bucketB.capacity = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_bucket_problem_l4119_411950


namespace NUMINAMATH_CALUDE_function_inequality_l4119_411999

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = f (x - 1)

def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_inequality (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : is_periodic_2 f)
  (h_monotone : monotone_increasing_on f 0 1) :
  f (-3/2) < f (4/3) ∧ f (4/3) < f 1 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l4119_411999


namespace NUMINAMATH_CALUDE_max_abs_z_on_circle_l4119_411921

open Complex

theorem max_abs_z_on_circle (z : ℂ) (h : abs (z - (3 + 4*I)) = 1) : 
  (∀ w : ℂ, abs (w - (3 + 4*I)) = 1 → abs w ≤ abs z) → abs z = 6 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_on_circle_l4119_411921


namespace NUMINAMATH_CALUDE_rectangle_max_area_l4119_411982

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → (l * w ≤ 100) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l4119_411982


namespace NUMINAMATH_CALUDE_square_digit_sequence_l4119_411955

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def form_number (x y : ℕ) (n : ℕ) : ℕ :=
  x * (10^(2*n+1) - 10^(n+1)) / 9 + 6 * 10^n + y * (10^n - 1) / 9

theorem square_digit_sequence (x y : ℕ) : x ≠ 0 →
  (∀ n : ℕ, n ≥ 1 → is_perfect_square (form_number x y n)) →
  ((x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = 0)) :=
sorry

end NUMINAMATH_CALUDE_square_digit_sequence_l4119_411955


namespace NUMINAMATH_CALUDE_two_digit_three_digit_sum_l4119_411956

theorem two_digit_three_digit_sum : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 100 ≤ y ∧ y < 1000 ∧ 
  100000 * x + y = 7 * x * y ∧ 
  x + y = 18 := by
sorry

end NUMINAMATH_CALUDE_two_digit_three_digit_sum_l4119_411956


namespace NUMINAMATH_CALUDE_yang_hui_field_equation_l4119_411923

theorem yang_hui_field_equation (area : ℕ) (length width : ℕ) :
  area = 650 ∧ width = length - 1 →
  length * (length - 1) = area :=
by sorry

end NUMINAMATH_CALUDE_yang_hui_field_equation_l4119_411923


namespace NUMINAMATH_CALUDE_smallest_n_for_g_with_large_digit_l4119_411953

/-- Sum of digits in base b representation of n -/
def digitSum (n : ℕ) (b : ℕ) : ℕ := sorry

/-- f(n) is the sum of digits in base-5 representation of n -/
def f (n : ℕ) : ℕ := digitSum n 5

/-- g(n) is the sum of digits in base-9 representation of f(n) -/
def g (n : ℕ) : ℕ := digitSum (f n) 9

/-- Checks if a number can be represented in base-17 using only digits 0-9 -/
def hasOnlyDigits0To9InBase17 (n : ℕ) : Prop := sorry

theorem smallest_n_for_g_with_large_digit : 
  (∀ m < 791, hasOnlyDigits0To9InBase17 (g m)) ∧ 
  ¬hasOnlyDigits0To9InBase17 (g 791) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_g_with_large_digit_l4119_411953


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l4119_411990

/-- The minimum value of a quadratic function -/
theorem quadratic_minimum_value 
  (p q r : ℝ) 
  (h1 : p > 0) 
  (h2 : q^2 - 4*p*r < 0) : 
  ∃ (x : ℝ), ∀ (y : ℝ), p*y^2 + q*y + r ≥ (4*p*r - q^2) / (4*p) :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l4119_411990


namespace NUMINAMATH_CALUDE_regular_pencil_price_correct_l4119_411957

/-- The price of a regular pencil in a stationery store --/
def regular_pencil_price : ℝ :=
  let pencil_with_eraser_price : ℝ := 0.8
  let short_pencil_price : ℝ := 0.4
  let pencils_with_eraser_sold : ℕ := 200
  let regular_pencils_sold : ℕ := 40
  let short_pencils_sold : ℕ := 35
  let total_sales : ℝ := 194
  0.5

/-- Theorem stating that the regular pencil price is correct --/
theorem regular_pencil_price_correct :
  let pencil_with_eraser_price : ℝ := 0.8
  let short_pencil_price : ℝ := 0.4
  let pencils_with_eraser_sold : ℕ := 200
  let regular_pencils_sold : ℕ := 40
  let short_pencils_sold : ℕ := 35
  let total_sales : ℝ := 194
  pencil_with_eraser_price * pencils_with_eraser_sold +
  regular_pencil_price * regular_pencils_sold +
  short_pencil_price * short_pencils_sold = total_sales :=
by
  sorry

end NUMINAMATH_CALUDE_regular_pencil_price_correct_l4119_411957


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l4119_411966

/-- The number of factors of 1200 that are perfect squares -/
def perfect_square_factors : ℕ :=
  let n := 1200
  let prime_factorization := (2, 4) :: (3, 1) :: (5, 2) :: []
  sorry

/-- Theorem stating that the number of factors of 1200 that are perfect squares is 6 -/
theorem count_perfect_square_factors :
  perfect_square_factors = 6 := by sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l4119_411966
