import Mathlib

namespace NUMINAMATH_CALUDE_maria_trip_portion_l2600_260078

theorem maria_trip_portion (total_distance : ℝ) (first_stop_fraction : ℝ) (remaining_distance : ℝ)
  (h1 : total_distance = 560)
  (h2 : first_stop_fraction = 1 / 2)
  (h3 : remaining_distance = 210) :
  (total_distance * (1 - first_stop_fraction) - remaining_distance) / (total_distance * (1 - first_stop_fraction)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_maria_trip_portion_l2600_260078


namespace NUMINAMATH_CALUDE_intersection_implies_C_value_l2600_260031

/-- Two lines intersect on the y-axis iff their intersection point has x-coordinate 0 -/
def intersect_on_y_axis (A C : ℝ) : Prop :=
  ∃ y : ℝ, A * 0 + 3 * y + C = 0 ∧ 2 * 0 - 3 * y + 4 = 0

/-- If the lines Ax + 3y + C = 0 and 2x - 3y + 4 = 0 intersect on the y-axis, then C = -4 -/
theorem intersection_implies_C_value (A : ℝ) :
  intersect_on_y_axis A C → C = -4 :=
sorry

end NUMINAMATH_CALUDE_intersection_implies_C_value_l2600_260031


namespace NUMINAMATH_CALUDE_gold_cube_profit_calculation_l2600_260011

/-- Calculates the profit from selling a gold cube -/
def goldCubeProfit (side : ℝ) (density : ℝ) (purchasePrice : ℝ) (markupFactor : ℝ) : ℝ :=
  let volume := side^3
  let mass := volume * density
  let cost := mass * purchasePrice
  let sellingPrice := cost * markupFactor
  sellingPrice - cost

/-- Theorem stating the profit from selling a specific gold cube -/
theorem gold_cube_profit_calculation :
  goldCubeProfit 6 19 60 1.5 = 123120 := by sorry

end NUMINAMATH_CALUDE_gold_cube_profit_calculation_l2600_260011


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2600_260084

theorem unique_solution_for_equation :
  ∃! (x y : ℝ), x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ∧ x = 1/3 ∧ y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2600_260084


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2600_260042

theorem sum_first_six_primes_mod_seventh_prime : 
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2600_260042


namespace NUMINAMATH_CALUDE_total_spider_legs_l2600_260030

/-- The number of spiders in the room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs is 32 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l2600_260030


namespace NUMINAMATH_CALUDE_max_value_x_y3_z4_l2600_260096

theorem max_value_x_y3_z4 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x + y^3 + z^4 ≤ 1 ∧ ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 ∧ x' + y'^3 + z'^4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_y3_z4_l2600_260096


namespace NUMINAMATH_CALUDE_expected_weight_of_disks_l2600_260012

/-- The expected weight of 100 disks with manufacturing errors -/
theorem expected_weight_of_disks (nominal_diameter : Real) (perfect_weight : Real) 
  (radius_std_dev : Real) (h1 : nominal_diameter = 1) (h2 : perfect_weight = 100) 
  (h3 : radius_std_dev = 0.01) : 
  ∃ (expected_weight : Real), 
    expected_weight = 10004 ∧ 
    expected_weight = 100 * perfect_weight * (1 + (radius_std_dev / (nominal_diameter / 2))^2) :=
by sorry

end NUMINAMATH_CALUDE_expected_weight_of_disks_l2600_260012


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l2600_260073

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 :=
by sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l2600_260073


namespace NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l2600_260097

theorem multiplicative_inverse_203_mod_301 :
  ∃ x : ℕ, x < 301 ∧ (7236 : ℤ) ≡ x [ZMOD 301] ∧ (203 * x) ≡ 1 [ZMOD 301] := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_301_l2600_260097


namespace NUMINAMATH_CALUDE_max_min_difference_z_l2600_260005

theorem max_min_difference_z (x y z : ℝ) 
  (sum_eq : x + y + z = 3) 
  (sum_squares_eq : x^2 + y^2 + z^2 = 18) : 
  ∃ (z_max z_min : ℝ), 
    (∀ w : ℝ, (∃ u v : ℝ, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 18) → w ≤ z_max) ∧
    (∀ w : ℝ, (∃ u v : ℝ, u + v + w = 3 ∧ u^2 + v^2 + w^2 = 18) → w ≥ z_min) ∧
    z_max - z_min = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l2600_260005


namespace NUMINAMATH_CALUDE_hexagon_sixth_angle_l2600_260081

/-- The sum of interior angles in a hexagon -/
def hexagon_angle_sum : ℝ := 720

/-- The five known angles in the hexagon -/
def known_angles : List ℝ := [108, 130, 142, 105, 120]

/-- Theorem: In a hexagon where five of the interior angles measure 108°, 130°, 142°, 105°, and 120°, the measure of the sixth angle is 115°. -/
theorem hexagon_sixth_angle :
  hexagon_angle_sum - (known_angles.sum) = 115 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_sixth_angle_l2600_260081


namespace NUMINAMATH_CALUDE_juan_reading_speed_l2600_260065

/-- Proves that Juan reads 250 pages per hour given the conditions of the problem -/
theorem juan_reading_speed (lunch_trip : ℝ) (book_pages : ℕ) (office_to_lunch : ℝ) 
  (h1 : lunch_trip = 2 * office_to_lunch)
  (h2 : book_pages = 4000)
  (h3 : office_to_lunch = 4)
  (h4 : lunch_trip = (book_pages : ℝ) / (250 : ℝ)) : 
  (book_pages : ℝ) / (2 * lunch_trip) = 250 := by
sorry

end NUMINAMATH_CALUDE_juan_reading_speed_l2600_260065


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2600_260024

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 144 = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2600_260024


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l2600_260023

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumOfDigitFactorials (n : ℕ) : ℕ :=
  (n.digits 10).map factorial |>.sum

def hasDigit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ n.digits 10

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = sumOfDigitFactorials n ∧ hasDigit n 3 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_sum_l2600_260023


namespace NUMINAMATH_CALUDE_central_angle_common_chord_l2600_260040

/-- The central angle corresponding to the common chord of two circles -/
theorem central_angle_common_chord (x y : ℝ) : 
  let circle1 := {(x, y) | (x - 2)^2 + y^2 = 4}
  let circle2 := {(x, y) | x^2 + (y - 2)^2 = 4}
  let center1 := (2, 0)
  let center2 := (0, 2)
  let radius := 2
  let center_distance := Real.sqrt ((2 - 0)^2 + (0 - 2)^2)
  let chord_distance := center_distance / 2
  let cos_half_angle := chord_distance / radius
  let central_angle := 2 * Real.arccos cos_half_angle
  central_angle = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_common_chord_l2600_260040


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l2600_260032

def m : ℕ := 2021^3 + 3^2021

theorem units_digit_of_m_squared_plus_three_to_m (m : ℕ := 2021^3 + 3^2021) :
  (m^2 + 3^m) % 10 = 7 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l2600_260032


namespace NUMINAMATH_CALUDE_cathy_win_probability_l2600_260043

/-- Represents a player in the die-rolling game -/
inductive Player : Type
| Ana : Player
| Bob : Player
| Cathy : Player

/-- The number of sides on the die -/
def dieSides : ℕ := 6

/-- The winning number on the die -/
def winningNumber : ℕ := 6

/-- The probability of rolling the winning number -/
def winProbability : ℚ := 1 / dieSides

/-- The probability of not rolling the winning number -/
def loseProbability : ℚ := 1 - winProbability

/-- The number of players before Cathy -/
def playersBeforeCathy : ℕ := 2

/-- Theorem stating the probability of Cathy winning -/
theorem cathy_win_probability :
  let p : ℚ := winProbability
  let q : ℚ := loseProbability
  (q^playersBeforeCathy * p) / (1 - q^3) = 25 / 91 := by sorry

end NUMINAMATH_CALUDE_cathy_win_probability_l2600_260043


namespace NUMINAMATH_CALUDE_nitrogen_atomic_weight_l2600_260000

/-- The atomic weight of nitrogen in a compound with given properties -/
theorem nitrogen_atomic_weight (molecular_weight : ℝ) (hydrogen_weight : ℝ) (bromine_weight : ℝ) :
  molecular_weight = 98 →
  hydrogen_weight = 1.008 →
  bromine_weight = 79.904 →
  molecular_weight = 4 * hydrogen_weight + bromine_weight + 14.064 :=
by sorry

end NUMINAMATH_CALUDE_nitrogen_atomic_weight_l2600_260000


namespace NUMINAMATH_CALUDE_value_of_120abc_l2600_260017

theorem value_of_120abc (a b c d : ℝ) 
  (h1 : 10 * a = 20) 
  (h2 : 6 * b = 20) 
  (h3 : c^2 + d^2 = 50) : 
  120 * a * b * c = 800 * Real.sqrt (50 - d^2) := by
  sorry

end NUMINAMATH_CALUDE_value_of_120abc_l2600_260017


namespace NUMINAMATH_CALUDE_f_at_four_is_zero_l2600_260071

/-- A function f satisfying the given property for all real x -/
def f : ℝ → ℝ := sorry

/-- The main property of the function f -/
axiom f_property : ∀ x : ℝ, x * f x = 2 * f (2 - x) + 1

/-- The theorem to be proved -/
theorem f_at_four_is_zero : f 4 = 0 := by sorry

end NUMINAMATH_CALUDE_f_at_four_is_zero_l2600_260071


namespace NUMINAMATH_CALUDE_multiplication_inequality_l2600_260034

theorem multiplication_inequality : 35 * 99 ≠ 35 * 100 + 35 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_inequality_l2600_260034


namespace NUMINAMATH_CALUDE_apple_in_B_l2600_260018

-- Define the boxes
inductive Box
| A
| B
| C

-- Define the location of the apple
def apple_location : Box := Box.B

-- Define the notes on the boxes
def note_A : Prop := apple_location = Box.A
def note_B : Prop := apple_location ≠ Box.B
def note_C : Prop := apple_location ≠ Box.A

-- Define the condition that only one note is true
def only_one_true : Prop :=
  (note_A ∧ ¬note_B ∧ ¬note_C) ∨
  (¬note_A ∧ note_B ∧ ¬note_C) ∨
  (¬note_A ∧ ¬note_B ∧ note_C)

-- Theorem to prove
theorem apple_in_B :
  only_one_true → apple_location = Box.B :=
by sorry

end NUMINAMATH_CALUDE_apple_in_B_l2600_260018


namespace NUMINAMATH_CALUDE_zero_additive_identity_for_integers_l2600_260009

theorem zero_additive_identity_for_integers : 
  ∃! y : ℤ, ∀ x : ℤ, y + x = x :=
by sorry

end NUMINAMATH_CALUDE_zero_additive_identity_for_integers_l2600_260009


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_four_l2600_260089

theorem sum_of_roots_equals_four :
  let f (x : ℝ) := (x^3 - 2*x^2 - 8*x) / (x + 2)
  (∃ a b : ℝ, (f a = 5 ∧ f b = 5 ∧ a ≠ b) ∧ a + b = 4) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_four_l2600_260089


namespace NUMINAMATH_CALUDE_ababa_binary_bits_l2600_260026

/-- The decimal representation of ABABA₁₆ -/
def ababa_decimal : ℕ := 701162

/-- The number of bits in the binary representation of ABABA₁₆ -/
def num_bits : ℕ := 20

theorem ababa_binary_bits :
  (2 ^ (num_bits - 1) : ℕ) ≤ ababa_decimal ∧ ababa_decimal < 2 ^ num_bits :=
by sorry

end NUMINAMATH_CALUDE_ababa_binary_bits_l2600_260026


namespace NUMINAMATH_CALUDE_rectangle_x_value_l2600_260058

/-- A rectangular construction with specified side lengths -/
structure RectConstruction where
  top_left : ℝ
  top_middle : ℝ
  top_right : ℝ
  bottom_left : ℝ
  bottom_middle : ℝ
  bottom_right : ℝ

/-- The theorem stating that X = 5 in the given rectangular construction -/
theorem rectangle_x_value (r : RectConstruction) 
  (h1 : r.top_left = 2)
  (h2 : r.top_right = 3)
  (h3 : r.bottom_left = 4)
  (h4 : r.bottom_middle = 1)
  (h5 : r.bottom_right = 5)
  (h6 : r.top_left + r.top_middle + r.top_right = r.bottom_left + r.bottom_middle + r.bottom_right) :
  r.top_middle = 5 := by
  sorry

#check rectangle_x_value

end NUMINAMATH_CALUDE_rectangle_x_value_l2600_260058


namespace NUMINAMATH_CALUDE_binomial_distribution_problem_l2600_260066

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The random variable X following a binomial distribution -/
def X (b : BinomialDistribution) : ℝ := sorry

/-- Expectation of a random variable -/
def expectation (X : ℝ) : ℝ := sorry

/-- Variance of a random variable -/
def variance (X : ℝ) : ℝ := sorry

theorem binomial_distribution_problem (b : BinomialDistribution) 
  (h2 : expectation (3 * X b - 9) = 27)
  (h3 : variance (3 * X b - 9) = 27) :
  b.n = 16 ∧ b.p = 3/4 := by sorry

end NUMINAMATH_CALUDE_binomial_distribution_problem_l2600_260066


namespace NUMINAMATH_CALUDE_bailey_towel_cost_l2600_260022

/-- Calculates the total cost of towel sets after discount -/
def towel_cost_after_discount (guest_sets : ℕ) (master_sets : ℕ) 
                               (guest_price : ℚ) (master_price : ℚ) 
                               (discount_percent : ℚ) : ℚ :=
  let total_cost := guest_sets * guest_price + master_sets * master_price
  let discount_amount := discount_percent * total_cost
  total_cost - discount_amount

/-- Theorem stating that Bailey's total cost for towel sets is $224.00 -/
theorem bailey_towel_cost :
  towel_cost_after_discount 2 4 40 50 (20 / 100) = 224 :=
by sorry

end NUMINAMATH_CALUDE_bailey_towel_cost_l2600_260022


namespace NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l2600_260067

theorem order_of_logarithmic_expressions :
  let a : ℝ := (Real.log (Real.sqrt 2)) / 2
  let b : ℝ := (Real.log 3) / 6
  let c : ℝ := 1 / (2 * Real.exp 1)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l2600_260067


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l2600_260044

theorem angle_measure_in_triangle (A B C : Real) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  A + B = 80 →       -- Given condition
  C = 100            -- Conclusion to prove
  := by sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l2600_260044


namespace NUMINAMATH_CALUDE_flu_infection_spread_l2600_260053

/-- The average number of people infected by one person in each round of infection -/
def average_infections : ℕ := 13

/-- The number of rounds of infection -/
def num_rounds : ℕ := 2

/-- The total number of people infected after two rounds -/
def total_infected : ℕ := 196

/-- The number of initially infected people -/
def initial_infected : ℕ := 1

theorem flu_infection_spread :
  (initial_infected + average_infections * initial_infected + 
   average_infections * (initial_infected + average_infections * initial_infected) = total_infected) ∧
  (average_infections > 0) := by
  sorry

end NUMINAMATH_CALUDE_flu_infection_spread_l2600_260053


namespace NUMINAMATH_CALUDE_remainder_91_power_91_mod_100_l2600_260050

/-- The remainder when 91^91 is divided by 100 is 91. -/
theorem remainder_91_power_91_mod_100 : 91^91 % 100 = 91 := by
  sorry

end NUMINAMATH_CALUDE_remainder_91_power_91_mod_100_l2600_260050


namespace NUMINAMATH_CALUDE_anthony_pencils_l2600_260072

def initial_pencils : ℕ := 56
def given_pencils : ℝ := 9.0
def remaining_pencils : ℕ := 47

theorem anthony_pencils : 
  (initial_pencils : ℝ) = given_pencils + remaining_pencils := by sorry

end NUMINAMATH_CALUDE_anthony_pencils_l2600_260072


namespace NUMINAMATH_CALUDE_f_increasing_after_3_l2600_260079

def f (x : ℝ) := 2 * (x - 3)^2 - 1

theorem f_increasing_after_3 :
  ∀ x₁ x₂ : ℝ, x₁ ≥ 3 → x₂ ≥ 3 → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_after_3_l2600_260079


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2600_260035

/-- The sum of coefficients of (1+ax)^6 -/
def sum_of_coefficients (a : ℝ) : ℝ := (1 + a)^6

/-- "a=1" is sufficient for the sum of coefficients to be 64 -/
theorem sufficient_condition : sum_of_coefficients 1 = 64 := by sorry

/-- "a=1" is not necessary for the sum of coefficients to be 64 -/
theorem not_necessary_condition : ∃ a : ℝ, a ≠ 1 ∧ sum_of_coefficients a = 64 := by sorry

/-- "a=1" is a sufficient but not necessary condition for the sum of coefficients to be 64 -/
theorem sufficient_but_not_necessary : 
  (sum_of_coefficients 1 = 64) ∧ (∃ a : ℝ, a ≠ 1 ∧ sum_of_coefficients a = 64) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2600_260035


namespace NUMINAMATH_CALUDE_greatest_integer_value_l2600_260069

theorem greatest_integer_value (x : ℤ) : (∀ y : ℤ, 3 * |y| - 1 ≤ 8 → y ≤ x) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_value_l2600_260069


namespace NUMINAMATH_CALUDE_cube_immersion_theorem_l2600_260074

/-- The edge length of a cube that, when immersed in a rectangular vessel,
    causes a specific rise in water level. -/
def cube_edge_length (vessel_length vessel_width water_rise : ℝ) : ℝ :=
  (vessel_length * vessel_width * water_rise) ^ (1/3)

/-- Theorem stating that a cube with edge length 16 cm, when immersed in a
    rectangular vessel with base 20 cm × 15 cm, causes a water level rise
    of 13.653333333333334 cm. -/
theorem cube_immersion_theorem :
  cube_edge_length 20 15 13.653333333333334 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cube_immersion_theorem_l2600_260074


namespace NUMINAMATH_CALUDE_unique_intersection_implies_r_equals_three_l2600_260092

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def B (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- State the theorem
theorem unique_intersection_implies_r_equals_three 
  (r : ℝ) 
  (h_r_pos : r > 0) 
  (h_unique : ∃! p, p ∈ A ∩ B r) : 
  r = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_intersection_implies_r_equals_three_l2600_260092


namespace NUMINAMATH_CALUDE_point_order_on_line_l2600_260004

theorem point_order_on_line (m n b : ℝ) : 
  (2 * (-1/2) + b = m) → (2 * 2 + b = n) → m < n := by sorry

end NUMINAMATH_CALUDE_point_order_on_line_l2600_260004


namespace NUMINAMATH_CALUDE_divisible_by_five_l2600_260080

theorem divisible_by_five (a b : ℕ) : 
  (5 ∣ a * b) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l2600_260080


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_implies_m_range_l2600_260076

/-- A point in the Cartesian plane is in the second quadrant if and only if its x-coordinate is negative and its y-coordinate is positive. -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Given a real number m, prove that if the point P(m-3, m+1) is in the second quadrant,
    then -1 < m and m < 3. -/
theorem point_in_second_quadrant_implies_m_range (m : ℝ) :
  is_in_second_quadrant (m - 3) (m + 1) → -1 < m ∧ m < 3 := by
  sorry


end NUMINAMATH_CALUDE_point_in_second_quadrant_implies_m_range_l2600_260076


namespace NUMINAMATH_CALUDE_wallpaper_overlap_area_l2600_260094

/-- Given the total area of wallpaper and areas covered by exactly two and three layers,
    calculate the actual area of the wall covered by overlapping wallpapers. -/
theorem wallpaper_overlap_area (total_area double_layer triple_layer : ℝ) 
    (h1 : total_area = 300)
    (h2 : double_layer = 30)
    (h3 : triple_layer = 45) :
    total_area - (2 * double_layer - double_layer) - (3 * triple_layer - triple_layer) = 180 := by
  sorry


end NUMINAMATH_CALUDE_wallpaper_overlap_area_l2600_260094


namespace NUMINAMATH_CALUDE_beetle_projection_theorem_l2600_260075

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a beetle moving on a line --/
structure Beetle where
  line : Line
  speed : ℝ
  initialPosition : ℝ

/-- Theorem: If two beetles move on intersecting lines with constant speeds,
    and their projections on the OX axis never coincide,
    then their projections on the OY axis must either coincide or have coincided in the past --/
theorem beetle_projection_theorem (L1 L2 : Line) (b1 b2 : Beetle)
    (h_intersect : L1 ≠ L2)
    (h_b1_on_L1 : b1.line = L1)
    (h_b2_on_L2 : b2.line = L2)
    (h_constant_speed : b1.speed ≠ 0 ∧ b2.speed ≠ 0)
    (h_x_proj_never_coincide : ∀ t : ℝ, 
      b1.initialPosition + b1.speed * t ≠ b2.initialPosition + b2.speed * t) :
    ∃ t : ℝ, 
      L1.slope * (b1.initialPosition + b1.speed * t) + L1.intercept = 
      L2.slope * (b2.initialPosition + b2.speed * t) + L2.intercept :=
sorry

end NUMINAMATH_CALUDE_beetle_projection_theorem_l2600_260075


namespace NUMINAMATH_CALUDE_fish_rice_trade_l2600_260038

/-- Represents the value of one fish in terms of bags of rice -/
def fish_value (fish bread apple rice : ℚ) : Prop :=
  (5 * fish = 3 * bread) ∧
  (bread = 6 * apple) ∧
  (2 * apple = rice) →
  fish = 9/5 * rice

theorem fish_rice_trade : ∀ (fish bread apple rice : ℚ),
  fish_value fish bread apple rice :=
by
  sorry

end NUMINAMATH_CALUDE_fish_rice_trade_l2600_260038


namespace NUMINAMATH_CALUDE_beth_crayons_l2600_260037

theorem beth_crayons (initial : ℕ) (given_away : ℕ) (left : ℕ) : 
  given_away = 54 → left = 52 → initial = given_away + left → initial = 106 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_l2600_260037


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2600_260036

/-- If |a-4|+(b+3)^2=0, then a > 0 and b < 0 -/
theorem point_in_fourth_quadrant (a b : ℝ) (h : |a - 4| + (b + 3)^2 = 0) : a > 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2600_260036


namespace NUMINAMATH_CALUDE_estimate_city_standards_l2600_260057

/-- Estimates the number of students meeting standards in a population based on a sample. -/
def estimate_meeting_standards (sample_size : ℕ) (sample_meeting : ℕ) (total_population : ℕ) : ℕ :=
  (total_population * sample_meeting) / sample_size

/-- Theorem stating the estimated number of students meeting standards in the city -/
theorem estimate_city_standards : 
  let sample_size := 1000
  let sample_meeting := 950
  let total_population := 1200000
  estimate_meeting_standards sample_size sample_meeting total_population = 1140000 := by
  sorry

end NUMINAMATH_CALUDE_estimate_city_standards_l2600_260057


namespace NUMINAMATH_CALUDE_A_intersect_B_l2600_260029

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem A_intersect_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2600_260029


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l2600_260062

/-- Circle type with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the line passing through the intersection points of two circles -/
def intersection_line_equation (c1 c2 : Circle) : ℝ × ℝ → Prop :=
  fun p => p.1 + p.2 = -2

/-- Theorem stating that the line passing through the intersection points of the given circles has the equation x + y = -2 -/
theorem intersection_line_of_circles :
  let c1 : Circle := { center := (-4, -10), radius := 15 }
  let c2 : Circle := { center := (8, 6), radius := Real.sqrt 104 }
  ∀ p, p ∈ { p | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 } ∩
           { p | (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2 } →
  intersection_line_equation c1 c2 p :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l2600_260062


namespace NUMINAMATH_CALUDE_more_cats_than_dogs_l2600_260064

theorem more_cats_than_dogs : 
  let num_dogs : ℕ := 9
  let num_cats : ℕ := 23
  num_cats - num_dogs = 14 := by sorry

end NUMINAMATH_CALUDE_more_cats_than_dogs_l2600_260064


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2600_260049

/-- The number of dots on each side of the square array -/
def n : ℕ := 5

/-- The number of different rectangles with sides parallel to the grid
    that can be formed by connecting four dots in an n×n square array of dots -/
def num_rectangles (n : ℕ) : ℕ :=
  (n.choose 2) * (n.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_5x5_grid :
  num_rectangles n = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2600_260049


namespace NUMINAMATH_CALUDE_equation_solution_difference_l2600_260095

theorem equation_solution_difference : ∃ (s₁ s₂ : ℝ),
  (s₁^2 - 5*s₁ - 24) / (s₁ + 3) = 3*s₁ + 10 ∧
  (s₂^2 - 5*s₂ - 24) / (s₂ + 3) = 3*s₂ + 10 ∧
  s₁ ≠ s₂ ∧
  |s₁ - s₂| = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l2600_260095


namespace NUMINAMATH_CALUDE_fraction_inequality_l2600_260014

theorem fraction_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) :
  (x₁ + 1) / (x₂ + 1) > x₁ / x₂ := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2600_260014


namespace NUMINAMATH_CALUDE_sum_of_roots_l2600_260033

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a - 17 = 0)
  (hb : b^3 - 3*b^2 + 5*b + 11 = 0) : 
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2600_260033


namespace NUMINAMATH_CALUDE_expected_draws_eq_sixteen_thirds_l2600_260088

/-- The number of red balls in the bag -/
def num_red : ℕ := 2

/-- The number of black balls in the bag -/
def num_black : ℕ := 5

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_black

/-- The set of possible numbers of draws -/
def possible_draws : Finset ℕ := Finset.range (total_balls + 1) \ Finset.range num_red

/-- The probability of drawing a specific number of balls -/
noncomputable def prob_draw (n : ℕ) : ℚ :=
  if n ∈ possible_draws then
    -- This is a placeholder for the actual probability calculation
    1 / possible_draws.card
  else
    0

/-- The expected number of draws -/
noncomputable def expected_draws : ℚ :=
  Finset.sum possible_draws (λ n => n * prob_draw n)

theorem expected_draws_eq_sixteen_thirds :
  expected_draws = 16 / 3 := by sorry

end NUMINAMATH_CALUDE_expected_draws_eq_sixteen_thirds_l2600_260088


namespace NUMINAMATH_CALUDE_isosceles_base_angle_l2600_260056

-- Define an isosceles triangle with a 30° vertex angle
def IsoscelesTriangle (α β γ : ℝ) : Prop :=
  α = 30 ∧ β = γ ∧ α + β + γ = 180

-- Theorem: In an isosceles triangle with a 30° vertex angle, each base angle is 75°
theorem isosceles_base_angle (α β γ : ℝ) (h : IsoscelesTriangle α β γ) : β = 75 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_base_angle_l2600_260056


namespace NUMINAMATH_CALUDE_polynomial_equality_l2600_260059

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x - 2) = x^2 + b*x - 6) → (a = 3 ∧ b = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2600_260059


namespace NUMINAMATH_CALUDE_petya_counterexample_l2600_260045

theorem petya_counterexample : ∃ (a b : ℕ), 
  (a^5 % b^2 = 0) ∧ (a^2 % b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_petya_counterexample_l2600_260045


namespace NUMINAMATH_CALUDE_largest_root_of_cubic_l2600_260003

theorem largest_root_of_cubic (p q r : ℝ) : 
  p + q + r = 3 → 
  p * q + p * r + q * r = -6 → 
  p * q * r = -8 → 
  ∃ (largest : ℝ), largest = (1 + Real.sqrt 17) / 2 ∧ 
    largest ≥ p ∧ largest ≥ q ∧ largest ≥ r ∧
    largest^3 - 3 * largest^2 - 6 * largest + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_root_of_cubic_l2600_260003


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2600_260098

theorem trigonometric_inequality (α β γ : Real) 
  (h1 : 0 ≤ α ∧ α ≤ Real.pi / 2)
  (h2 : 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h3 : 0 ≤ γ ∧ γ ≤ Real.pi / 2)
  (h4 : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  2 ≤ (1 + Real.cos α ^ 2) ^ 2 * Real.sin α ^ 4 + 
      (1 + Real.cos β ^ 2) ^ 2 * Real.sin β ^ 4 + 
      (1 + Real.cos γ ^ 2) ^ 2 * Real.sin γ ^ 4 ∧
  (1 + Real.cos α ^ 2) ^ 2 * Real.sin α ^ 4 + 
  (1 + Real.cos β ^ 2) ^ 2 * Real.sin β ^ 4 + 
  (1 + Real.cos γ ^ 2) ^ 2 * Real.sin γ ^ 4 ≤ 
  (1 + Real.cos α ^ 2) * (1 + Real.cos β ^ 2) * (1 + Real.cos γ ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2600_260098


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_4_l2600_260051

theorem greatest_integer_with_gcf_4 : ∃ n : ℕ, 
  n < 200 ∧ 
  Nat.gcd n 24 = 4 ∧ 
  ∀ m : ℕ, m < 200 → Nat.gcd m 24 = 4 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_4_l2600_260051


namespace NUMINAMATH_CALUDE_polynomial_sum_l2600_260060

-- Define the polynomial P
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) :
  P a b c d 1 = 2000 →
  P a b c d 2 = 4000 →
  P a b c d 3 = 6000 →
  P a b c d 9 + P a b c d (-5) = 12704 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2600_260060


namespace NUMINAMATH_CALUDE_max_value_of_f_l2600_260046

def f (x : ℝ) : ℝ := -2 * x^2 + 8

theorem max_value_of_f :
  ∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M ∧ M = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2600_260046


namespace NUMINAMATH_CALUDE_max_digits_distinct_divisible_l2600_260085

/-- A function that checks if all digits in a natural number are different -/
def hasDistinctDigits (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is divisible by all of its digits -/
def isDivisibleByAllDigits (n : ℕ) : Prop := sorry

/-- A function that returns the number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the maximum number of digits in a natural number
    with distinct digits and divisible by all its digits is 7 -/
theorem max_digits_distinct_divisible :
  ∃ (n : ℕ), hasDistinctDigits n ∧ isDivisibleByAllDigits n ∧ numDigits n = 7 ∧
  ∀ (m : ℕ), hasDistinctDigits m → isDivisibleByAllDigits m → numDigits m ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_digits_distinct_divisible_l2600_260085


namespace NUMINAMATH_CALUDE_max_product_roots_quadratic_l2600_260086

/-- Given a quadratic equation 6x^2 - 12x + m = 0 with real roots,
    the maximum value of m that maximizes the product of the roots is 6. -/
theorem max_product_roots_quadratic :
  ∀ m : ℝ,
  (∃ x y : ℝ, 6 * x^2 - 12 * x + m = 0 ∧ 6 * y^2 - 12 * y + m = 0 ∧ x ≠ y) →
  (∀ k : ℝ, (∃ x y : ℝ, 6 * x^2 - 12 * x + k = 0 ∧ 6 * y^2 - 12 * y + k = 0 ∧ x ≠ y) →
    m / 6 ≥ k / 6) →
  m = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_product_roots_quadratic_l2600_260086


namespace NUMINAMATH_CALUDE_substitution_remainder_l2600_260010

/-- Represents the number of available players -/
def total_players : ℕ := 15

/-- Represents the number of starting players -/
def starting_players : ℕ := 5

/-- Represents the maximum number of substitutions allowed -/
def max_substitutions : ℕ := 4

/-- Calculates the number of ways to make substitutions for a given number of substitutions -/
def substitution_ways (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then starting_players * (total_players - starting_players)
  else starting_players * (total_players - starting_players - n + 2) * substitution_ways (n - 1)

/-- Calculates the total number of ways to make substitutions -/
def total_substitution_ways : ℕ :=
  (List.range (max_substitutions + 1)).map substitution_ways |>.sum

/-- The main theorem stating that the remainder of total substitution ways divided by 1000 is 301 -/
theorem substitution_remainder :
  total_substitution_ways % 1000 = 301 := by sorry

end NUMINAMATH_CALUDE_substitution_remainder_l2600_260010


namespace NUMINAMATH_CALUDE_cube_edge_length_l2600_260015

theorem cube_edge_length (V : ℝ) (h : V = 32 * Real.pi / 3) :
  ∃ s : ℝ, s = 4 * Real.sqrt 3 / 3 ∧ V = 4 * Real.pi * (s * Real.sqrt 3 / 2)^3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2600_260015


namespace NUMINAMATH_CALUDE_percentage_spent_l2600_260008

theorem percentage_spent (initial_amount remaining_amount : ℝ) 
  (h1 : initial_amount = 5000)
  (h2 : remaining_amount = 3500) :
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_spent_l2600_260008


namespace NUMINAMATH_CALUDE_custom_operation_theorem_l2600_260013

def custom_operation (M N : Set ℕ) : Set ℕ :=
  {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

def M : Set ℕ := {0, 2, 4, 6, 8, 10}
def N : Set ℕ := {0, 3, 6, 9, 12, 15}

theorem custom_operation_theorem :
  custom_operation (custom_operation M N) M = N := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_theorem_l2600_260013


namespace NUMINAMATH_CALUDE_blithe_lost_toys_l2600_260048

theorem blithe_lost_toys (initial_toys : ℕ) (found_toys : ℕ) (final_toys : ℕ) 
  (h1 : initial_toys = 40)
  (h2 : found_toys = 9)
  (h3 : final_toys = 43)
  : initial_toys - (final_toys - found_toys) = 9 := by
  sorry

end NUMINAMATH_CALUDE_blithe_lost_toys_l2600_260048


namespace NUMINAMATH_CALUDE_stratified_sampling_first_grade_l2600_260061

theorem stratified_sampling_first_grade (total_students : ℕ) (sample_size : ℕ) 
  (grade_1_ratio grade_2_ratio grade_3_ratio : ℕ) :
  total_students = 2400 →
  sample_size = 120 →
  grade_1_ratio = 5 →
  grade_2_ratio = 4 →
  grade_3_ratio = 3 →
  (grade_1_ratio * sample_size) / (grade_1_ratio + grade_2_ratio + grade_3_ratio) = 50 := by
  sorry

#check stratified_sampling_first_grade

end NUMINAMATH_CALUDE_stratified_sampling_first_grade_l2600_260061


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l2600_260087

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 21 = 40 →
  arithmetic_sequence a₁ d 22 = 44 →
  arithmetic_sequence a₁ d 5 = -24 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l2600_260087


namespace NUMINAMATH_CALUDE_female_officers_count_l2600_260055

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 204 →
  female_on_duty_ratio = 1/2 →
  female_ratio = 17/100 →
  ∃ (total_female : ℕ), total_female = 600 ∧ 
    (female_ratio * total_female : ℚ) = (female_on_duty_ratio * total_on_duty : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l2600_260055


namespace NUMINAMATH_CALUDE_match_probabilities_l2600_260019

/-- A best-of-5 match where the probability of winning each game is 3/5 -/
structure Match :=
  (p : ℝ)
  (h_p : p = 3/5)

/-- The probability of winning 3 consecutive games -/
def prob_win_3_0 (m : Match) : ℝ := m.p^3

/-- The probability of winning the match after losing the first game -/
def prob_win_after_loss (m : Match) : ℝ :=
  m.p^3 + 3 * m.p^3 * (1 - m.p)

/-- The expected number of games played when losing the first game -/
def expected_games_after_loss (m : Match) : ℝ :=
  3 * (1 - m.p)^2 + 4 * (2 * m.p * (1 - m.p)^2 + m.p^3) + 5 * (3 * m.p^2 * (1 - m.p)^2 + m.p^3 * (1 - m.p))

theorem match_probabilities (m : Match) :
  prob_win_3_0 m = 27/125 ∧
  prob_win_after_loss m = 297/625 ∧
  expected_games_after_loss m = 534/125 :=
by sorry

end NUMINAMATH_CALUDE_match_probabilities_l2600_260019


namespace NUMINAMATH_CALUDE_find_number_l2600_260047

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 57 :=
  sorry

end NUMINAMATH_CALUDE_find_number_l2600_260047


namespace NUMINAMATH_CALUDE_sector_area_l2600_260099

theorem sector_area (r : Real) (θ : Real) (h1 : r = Real.pi) (h2 : θ = 2 * Real.pi / 3) :
  (1 / 2) * r * r * θ = Real.pi^3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2600_260099


namespace NUMINAMATH_CALUDE_gina_remaining_money_l2600_260063

def initial_amount : ℚ := 400

def mom_fraction : ℚ := 1/4
def clothes_fraction : ℚ := 1/8
def charity_fraction : ℚ := 1/5

def remaining_amount : ℚ := initial_amount * (1 - mom_fraction - clothes_fraction - charity_fraction)

theorem gina_remaining_money :
  remaining_amount = 170 := by sorry

end NUMINAMATH_CALUDE_gina_remaining_money_l2600_260063


namespace NUMINAMATH_CALUDE_input_statement_incorrect_l2600_260054

-- Define a type for program statements
inductive ProgramStatement
| Input (prompt : String) (value : String)
| Print (prompt : String) (value : String)
| Assignment (left : String) (right : String)

-- Define a function to check if an input statement is valid
def isValidInputStatement (stmt : ProgramStatement) : Prop :=
  match stmt with
  | ProgramStatement.Input _ value => ¬ (value.contains '+' ∨ value.contains '-' ∨ value.contains '*' ∨ value.contains '/')
  | _ => True

-- Theorem to prove
theorem input_statement_incorrect :
  let stmt := ProgramStatement.Input "MATH=" "a+b+c"
  ¬ (isValidInputStatement stmt) := by
sorry

end NUMINAMATH_CALUDE_input_statement_incorrect_l2600_260054


namespace NUMINAMATH_CALUDE_max_togs_value_l2600_260077

def tag_price : ℕ := 3
def tig_price : ℕ := 4
def tog_price : ℕ := 8
def total_budget : ℕ := 100

def max_togs (x y z : ℕ) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧
  x * tag_price + y * tig_price + z * tog_price = total_budget ∧
  ∀ (a b c : ℕ), a ≥ 1 → b ≥ 1 → c ≥ 1 →
    a * tag_price + b * tig_price + c * tog_price = total_budget →
    c ≤ z

theorem max_togs_value : ∃ (x y : ℕ), max_togs x y 11 := by
  sorry

end NUMINAMATH_CALUDE_max_togs_value_l2600_260077


namespace NUMINAMATH_CALUDE_grocer_coffee_stock_l2600_260093

/-- The amount of coffee initially in stock -/
def initial_stock : ℝ := 400

/-- The percentage of decaffeinated coffee in the initial stock -/
def initial_decaf_percent : ℝ := 0.20

/-- The amount of additional coffee purchased -/
def additional_coffee : ℝ := 100

/-- The percentage of decaffeinated coffee in the additional purchase -/
def additional_decaf_percent : ℝ := 0.60

/-- The final percentage of decaffeinated coffee after the purchase -/
def final_decaf_percent : ℝ := 0.28000000000000004

theorem grocer_coffee_stock :
  (initial_decaf_percent * initial_stock + additional_decaf_percent * additional_coffee) / 
  (initial_stock + additional_coffee) = final_decaf_percent := by
  sorry

end NUMINAMATH_CALUDE_grocer_coffee_stock_l2600_260093


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l2600_260091

theorem lcm_gcf_problem (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 4) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l2600_260091


namespace NUMINAMATH_CALUDE_equation_root_constraint_l2600_260006

theorem equation_root_constraint (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = a * x + 1) ∧ 
  (∀ x : ℝ, x > 0 → |x| ≠ a * x + 1) → 
  -1 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_equation_root_constraint_l2600_260006


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l2600_260001

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) :
  ∃ (q' r' : ℕ+), Nat.gcd q'.val r'.val = 10 ∧
    ∀ (q'' r'' : ℕ+), Nat.gcd q''.val r''.val < 10 →
      ¬(Nat.gcd p q''.val = 210 ∧ Nat.gcd p r''.val = 770) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l2600_260001


namespace NUMINAMATH_CALUDE_tim_grew_44_cantaloupes_l2600_260016

/-- The number of cantaloupes Fred grew -/
def fred_cantaloupes : ℕ := 38

/-- The total number of cantaloupes Fred and Tim grew together -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Tim grew -/
def tim_cantaloupes : ℕ := total_cantaloupes - fred_cantaloupes

theorem tim_grew_44_cantaloupes : tim_cantaloupes = 44 := by
  sorry

end NUMINAMATH_CALUDE_tim_grew_44_cantaloupes_l2600_260016


namespace NUMINAMATH_CALUDE_oranges_discarded_per_day_l2600_260090

theorem oranges_discarded_per_day 
  (harvest_per_day : ℕ) 
  (days : ℕ) 
  (remaining_sacks : ℕ) 
  (h1 : harvest_per_day = 74)
  (h2 : days = 51)
  (h3 : remaining_sacks = 153) :
  (harvest_per_day * days - remaining_sacks) / days = 71 := by
  sorry

end NUMINAMATH_CALUDE_oranges_discarded_per_day_l2600_260090


namespace NUMINAMATH_CALUDE_quality_difference_confidence_l2600_260002

/-- Production data for two machines -/
structure ProductionData :=
  (machine_a_first : ℕ)
  (machine_a_second : ℕ)
  (machine_b_first : ℕ)
  (machine_b_second : ℕ)

/-- Calculate K^2 statistic -/
def calculate_k_squared (data : ProductionData) : ℚ :=
  let n := data.machine_a_first + data.machine_a_second + data.machine_b_first + data.machine_b_second
  let a := data.machine_a_first
  let b := data.machine_a_second
  let c := data.machine_b_first
  let d := data.machine_b_second
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Critical values for K^2 test -/
def critical_value_99_percent : ℚ := 6635 / 1000
def critical_value_999_percent : ℚ := 10828 / 1000

/-- Theorem stating the confidence level for the difference in quality -/
theorem quality_difference_confidence (data : ProductionData) 
  (h1 : data.machine_a_first = 150)
  (h2 : data.machine_a_second = 50)
  (h3 : data.machine_b_first = 120)
  (h4 : data.machine_b_second = 80) :
  critical_value_99_percent < calculate_k_squared data ∧ 
  calculate_k_squared data < critical_value_999_percent :=
sorry

end NUMINAMATH_CALUDE_quality_difference_confidence_l2600_260002


namespace NUMINAMATH_CALUDE_right_building_shorter_l2600_260039

def middle_height : ℝ := 100
def left_height : ℝ := 0.8 * middle_height
def total_height : ℝ := 340

theorem right_building_shorter : 
  (middle_height + left_height) - (total_height - (middle_height + left_height)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_building_shorter_l2600_260039


namespace NUMINAMATH_CALUDE_yarn_parts_count_l2600_260068

/-- Given a yarn of 10 meters cut into equal parts, where 3 parts equal 6 meters,
    prove that the yarn was cut into 5 parts. -/
theorem yarn_parts_count (total_length : ℝ) (used_parts : ℕ) (used_length : ℝ) :
  total_length = 10 →
  used_parts = 3 →
  used_length = 6 →
  (total_length / (used_length / used_parts : ℝ) : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_yarn_parts_count_l2600_260068


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2600_260070

/-- The polynomial function f(x) = x^8 + 6x^7 + 12x^6 + 2027x^5 - 1586x^4 -/
def f (x : ℝ) : ℝ := x^8 + 6*x^7 + 12*x^6 + 2027*x^5 - 1586*x^4

/-- Theorem: The equation f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! x : ℝ, x > 0 ∧ f x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2600_260070


namespace NUMINAMATH_CALUDE_john_running_distance_l2600_260027

def monday_distance : ℕ := 1700
def tuesday_distance : ℕ := monday_distance + 200
def wednesday_distance : ℕ := (7 * tuesday_distance) / 10
def thursday_distance : ℕ := 2 * wednesday_distance
def friday_distance : ℕ := 3500

def total_distance : ℕ := monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance

theorem john_running_distance : total_distance = 10090 := by
  sorry

end NUMINAMATH_CALUDE_john_running_distance_l2600_260027


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l2600_260028

def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_sequence_condition (a b c : ℝ) :
  (is_geometric_sequence a b c → b^2 = a*c) ∧
  ¬(b^2 = a*c → is_geometric_sequence a b c) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l2600_260028


namespace NUMINAMATH_CALUDE_product_of_one_plus_tangents_sine_double_angle_l2600_260082

-- Part I
theorem product_of_one_plus_tangents (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) (h3 : α + β = π/4) : 
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

-- Part II
theorem sine_double_angle (α β : Real) 
  (h1 : π/2 < β ∧ β < α ∧ α < 3*π/4) 
  (h2 : Real.cos (α - β) = 12/13) 
  (h3 : Real.sin (α + β) = -3/5) : 
  Real.sin (2 * α) = -56/65 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_tangents_sine_double_angle_l2600_260082


namespace NUMINAMATH_CALUDE_five_letter_words_count_l2600_260020

def alphabet_size : ℕ := 26
def vowel_count : ℕ := 5

theorem five_letter_words_count : 
  (alphabet_size^3 * vowel_count : ℕ) = 87880 := by
sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l2600_260020


namespace NUMINAMATH_CALUDE_parabola_tangent_theorem_l2600_260021

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line that point A is on
def line_A (x y : ℝ) : Prop := x - 2*y + 13 = 0

-- Define that A is not on the y-axis
def A_not_on_y_axis (x y : ℝ) : Prop := x ≠ 0

-- Define points M and N as tangent points on the parabola
def M_N_tangent_points (xm ym xn yn : ℝ) : Prop :=
  parabola xm ym ∧ parabola xn yn

-- Define B and C as intersection points of AM and AN with y-axis
def B_C_intersection_points (xb yb xc yc : ℝ) : Prop :=
  xb = 0 ∧ xc = 0

-- Theorem statement
theorem parabola_tangent_theorem
  (xa ya xm ym xn yn xb yb xc yc : ℝ)
  (h1 : line_A xa ya)
  (h2 : A_not_on_y_axis xa ya)
  (h3 : M_N_tangent_points xm ym xn yn)
  (h4 : B_C_intersection_points xb yb xc yc) :
  -- 1. Line MN passes through (13, 8)
  ∃ (t : ℝ), xm + t * (xn - xm) = 13 ∧ ym + t * (yn - ym) = 8 ∧
  -- 2. Circumcircle of ABC passes through (2, 0)
  (xa - 2)^2 + ya^2 = (xb - 2)^2 + yb^2 ∧ (xa - 2)^2 + ya^2 = (xc - 2)^2 + yc^2 ∧
  -- 3. Minimum radius of circumcircle is (3√5)/2
  ∃ (r : ℝ), r ≥ (3 * Real.sqrt 5) / 2 ∧
    (xa - 2)^2 + ya^2 = 4 * r^2 ∧ (xb - 2)^2 + yb^2 = 4 * r^2 ∧ (xc - 2)^2 + yc^2 = 4 * r^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_theorem_l2600_260021


namespace NUMINAMATH_CALUDE_three_digit_reverse_subtraction_l2600_260052

theorem three_digit_reverse_subtraction (b c : ℕ) : 
  (0 < c) ∧ (c < 10) ∧ (b < 10) → 
  (101*c + 10*b + 300) - (101*c + 10*b + 3) = 297 := by
  sorry

#check three_digit_reverse_subtraction

end NUMINAMATH_CALUDE_three_digit_reverse_subtraction_l2600_260052


namespace NUMINAMATH_CALUDE_sixteen_points_divide_square_into_ten_equal_triangles_l2600_260007

/-- Represents a point inside a unit square -/
structure PointInSquare where
  x : Real
  y : Real
  inside : 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1

/-- Represents the areas of the four triangles formed by a point and the square's sides -/
structure TriangleAreas where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  a₄ : ℕ
  sum_is_ten : a₁ + a₂ + a₃ + a₄ = 10
  all_positive : 1 ≤ a₁ ∧ 1 ≤ a₂ ∧ 1 ≤ a₃ ∧ 1 ≤ a₄
  all_at_most_four : a₁ ≤ 4 ∧ a₂ ≤ 4 ∧ a₃ ≤ 4 ∧ a₄ ≤ 4

/-- The main theorem stating that there are exactly 16 points satisfying the condition -/
theorem sixteen_points_divide_square_into_ten_equal_triangles :
  ∃ (points : Finset PointInSquare),
    points.card = 16 ∧
    (∀ p ∈ points, ∃ (areas : TriangleAreas), True) ∧
    (∀ p : PointInSquare, p ∉ points → ¬∃ (areas : TriangleAreas), True) := by
  sorry


end NUMINAMATH_CALUDE_sixteen_points_divide_square_into_ten_equal_triangles_l2600_260007


namespace NUMINAMATH_CALUDE_percentage_relation_l2600_260041

theorem percentage_relation (j p t m n : ℕ+) (r : ℚ) : 
  (j : ℚ) = 0.75 * p ∧
  (j : ℚ) = 0.80 * t ∧
  (t : ℚ) = p - (r / 100) * p ∧
  (m : ℚ) = 1.10 * p ∧
  (n : ℚ) = 0.70 * m ∧
  (j : ℚ) + p + t = m * n →
  r = 6.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l2600_260041


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l2600_260025

/-- Represents the price of type A Kiwi in yuan -/
def a : ℝ := 35

/-- Represents the price of type B Kiwi in yuan -/
def b : ℝ := 50

/-- The cost of 2 type A and 1 type B Kiwi is 120 yuan -/
axiom cost_equation_1 : 2 * a + b = 120

/-- The cost of 3 type A and 2 type B Kiwi is 205 yuan -/
axiom cost_equation_2 : 3 * a + 2 * b = 205

/-- The cost price of each type B Kiwi is 40 yuan -/
def cost_B : ℝ := 40

/-- Daily sales of type B Kiwi at price b -/
def initial_sales : ℝ := 100

/-- Decrease in sales for each yuan increase in price -/
def sales_decrease : ℝ := 5

/-- Daily profit function for type B Kiwi -/
def profit (x : ℝ) : ℝ := (x - cost_B) * (initial_sales - sales_decrease * (x - b))

/-- The optimal selling price for type B Kiwi -/
def optimal_price : ℝ := 55

/-- The maximum daily profit for type B Kiwi -/
def max_profit : ℝ := 1125

/-- Theorem stating that the optimal price maximizes the profit -/
theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit x ≤ profit optimal_price ∧ profit optimal_price = max_profit :=
sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l2600_260025


namespace NUMINAMATH_CALUDE_jay_and_paul_distance_l2600_260083

/-- Calculates the distance traveled given a speed and time --/
def distance (speed : ℚ) (time : ℚ) : ℚ := speed * time

/-- Proves that Jay and Paul will be 20 miles apart after walking in opposite directions for 2 hours --/
theorem jay_and_paul_distance : 
  let jay_speed : ℚ := 1 / 15  -- 1 mile per 15 minutes
  let paul_speed : ℚ := 3 / 30 -- 3 miles per 30 minutes
  let time : ℚ := 2 * 60      -- 2 hours in minutes
  distance jay_speed time + distance paul_speed time = 20 := by
sorry

end NUMINAMATH_CALUDE_jay_and_paul_distance_l2600_260083
