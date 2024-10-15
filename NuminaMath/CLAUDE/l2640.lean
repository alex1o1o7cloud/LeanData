import Mathlib

namespace NUMINAMATH_CALUDE_determine_x_value_l2640_264026

theorem determine_x_value (w y z x : ℤ) 
  (hw : w = 65)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 7) : 
  x = 112 := by
  sorry

end NUMINAMATH_CALUDE_determine_x_value_l2640_264026


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l2640_264070

theorem arithmetic_sequence_max_product (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 8 + a 9 + a 10 = 24 →           -- given sum condition
  ∃ m : ℝ, m = 2 ∧ ∀ d' : ℝ, a 1 * d' ≤ m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l2640_264070


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l2640_264098

theorem system_of_equations_solutions :
  -- First system of equations
  (∃ x y : ℝ, 2 * x - 3 * y = -2 ∧ 5 * x + 3 * y = 37 ∧ x = 5 ∧ y = 4) ∧
  -- Second system of equations
  (∃ x y : ℝ, 3 * x + 2 * y = 5 ∧ 4 * x - y = 3 ∧ x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l2640_264098


namespace NUMINAMATH_CALUDE_sum_of_gcd_and_lcm_l2640_264090

def A : ℕ := Nat.gcd 18 (Nat.gcd 24 36)
def B : ℕ := Nat.lcm 18 (Nat.lcm 24 36)

theorem sum_of_gcd_and_lcm : A + B = 78 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_gcd_and_lcm_l2640_264090


namespace NUMINAMATH_CALUDE_cheryl_same_color_probability_l2640_264069

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 3

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the total number of marbles -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- Represents the number of marbles each person draws -/
def marbles_drawn : ℕ := 3

/-- Calculates the probability of Cheryl getting 3 marbles of the same color -/
theorem cheryl_same_color_probability :
  (num_colors * (Nat.choose (total_marbles - 2 * marbles_drawn) marbles_drawn)) /
  (Nat.choose total_marbles marbles_drawn * Nat.choose (total_marbles - marbles_drawn) marbles_drawn) = 1 / 28 :=
by sorry

end NUMINAMATH_CALUDE_cheryl_same_color_probability_l2640_264069


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2640_264023

/-- The ellipse equation -/
def ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0

/-- The origin is inside the ellipse -/
def origin_inside (k : ℝ) : Prop :=
  ∃ ε > 0, ∀ x y : ℝ, x^2 + y^2 < ε^2 → ellipse k x y

/-- The theorem stating the range of k -/
theorem ellipse_k_range :
  ∀ k : ℝ, origin_inside k → 0 < |k| ∧ |k| < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2640_264023


namespace NUMINAMATH_CALUDE_vector_subtraction_l2640_264062

def a : Fin 3 → ℝ := ![(-5 : ℝ), 1, 3]
def b : Fin 3 → ℝ := ![(3 : ℝ), -1, 2]

theorem vector_subtraction :
  a - 2 • b = ![(-11 : ℝ), 3, -1] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2640_264062


namespace NUMINAMATH_CALUDE_log2_derivative_l2640_264053

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
sorry

end NUMINAMATH_CALUDE_log2_derivative_l2640_264053


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l2640_264089

theorem two_numbers_sum_and_difference (x y : ℝ) : 
  x + y = 18 ∧ x - y = 6 → x = 12 ∧ y = 6 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l2640_264089


namespace NUMINAMATH_CALUDE_c_share_l2640_264022

/-- Represents the share of money for each person -/
structure Share where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The theorem stating C's share given the conditions -/
theorem c_share (s : Share) 
  (h1 : s.a / s.b = 5 / 3)
  (h2 : s.b / s.c = 3 / 2)
  (h3 : s.c / s.d = 2 / 3)
  (h4 : s.a = s.b + 1000) :
  s.c = 1000 := by
sorry

end NUMINAMATH_CALUDE_c_share_l2640_264022


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l2640_264077

/-- 
Theorem: The largest value of n for which 2x^2 + nx + 50 can be factored 
as the product of two linear factors with integer coefficients is 101.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (m : ℤ), 
    (∃ (a b : ℤ), 2 * X^2 + n * X + 50 = (2 * X + a) * (X + b)) → 
    m ≤ n) ∧ 
  (∃ (a b : ℤ), 2 * X^2 + 101 * X + 50 = (2 * X + a) * (X + b)) :=
by sorry


end NUMINAMATH_CALUDE_largest_n_for_factorization_l2640_264077


namespace NUMINAMATH_CALUDE_fraction_equality_l2640_264047

theorem fraction_equality (x y : ℝ) (h : x / y = 2) : (x - y) / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2640_264047


namespace NUMINAMATH_CALUDE_fraction_simplification_l2640_264083

theorem fraction_simplification : (4 / 252 : ℚ) + (17 / 36 : ℚ) = 41 / 84 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2640_264083


namespace NUMINAMATH_CALUDE_cost_AB_flight_l2640_264075

-- Define the distances
def distance_AC : ℝ := 3000
def distance_AB : ℝ := 3250

-- Define the cost structure
def bus_cost_per_km : ℝ := 0.15
def plane_cost_per_km : ℝ := 0.10
def plane_booking_fee : ℝ := 100

-- Define the function to calculate flight cost
def flight_cost (distance : ℝ) : ℝ :=
  distance * plane_cost_per_km + plane_booking_fee

-- Theorem to prove
theorem cost_AB_flight : flight_cost distance_AB = 425 := by
  sorry

end NUMINAMATH_CALUDE_cost_AB_flight_l2640_264075


namespace NUMINAMATH_CALUDE_time_per_video_l2640_264019

-- Define the parameters
def setup_time : ℝ := 1
def cleanup_time : ℝ := 1
def painting_time_per_video : ℝ := 1
def editing_time_per_video : ℝ := 1.5
def num_videos : ℕ := 4

-- Define the theorem
theorem time_per_video : 
  (setup_time + cleanup_time + num_videos * painting_time_per_video + num_videos * editing_time_per_video) / num_videos = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_per_video_l2640_264019


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l2640_264042

theorem abs_inequality_equivalence (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 8) ↔ ((-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11)) :=
by sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l2640_264042


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2640_264074

/-- Given a rectangular tank with length 3√3, width 1, and height 2√2,
    the surface area of its circumscribed sphere is 36π. -/
theorem circumscribed_sphere_surface_area 
  (length : ℝ) (width : ℝ) (height : ℝ)
  (h_length : length = 3 * Real.sqrt 3)
  (h_width : width = 1)
  (h_height : height = 2 * Real.sqrt 2) :
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 36 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2640_264074


namespace NUMINAMATH_CALUDE_coincident_foci_and_vertices_m_range_l2640_264006

-- Define the ellipse equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (9 - m) + y^2 / (2 * m) = 1 ∧ 9 - m > 2 * m ∧ 2 * m > 0

-- Define the hyperbola equation and its eccentricity condition
def hyperbola_eccentricity_condition (m : ℝ) : Prop :=
  ∃ (e : ℝ), e > Real.sqrt 6 / 2 ∧ e < Real.sqrt 2 ∧
  e^2 = (5 + m) / 5

-- Theorem for part (I)
theorem coincident_foci_and_vertices (m : ℝ) 
  (h1 : is_ellipse m) (h2 : hyperbola_eccentricity_condition m) :
  (∃ (x : ℝ), x^2 / (9 - m) + 0^2 / (2 * m) = 1 ∧ 
              x^2 / 5 - 0^2 / m = 1) → m = 4 / 3 :=
sorry

-- Theorem for part (II)
theorem m_range (m : ℝ) 
  (h1 : is_ellipse m) (h2 : hyperbola_eccentricity_condition m) :
  5 / 2 < m ∧ m < 3 :=
sorry

end NUMINAMATH_CALUDE_coincident_foci_and_vertices_m_range_l2640_264006


namespace NUMINAMATH_CALUDE_integer_roots_count_l2640_264003

theorem integer_roots_count : 
  let lower_bound := -5 - Real.sqrt 42
  let upper_bound := -5 + Real.sqrt 42
  let is_valid_root (x : ℤ) := 
    (Real.cos (2 * π * ↑x) + Real.cos (π * ↑x) = Real.sin (3 * π * ↑x) + Real.sin (π * ↑x)) ∧
    (lower_bound < x) ∧ (x < upper_bound)
  ∃! (roots : Finset ℤ), (Finset.card roots = 7) ∧ (∀ x, x ∈ roots ↔ is_valid_root x) :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_count_l2640_264003


namespace NUMINAMATH_CALUDE_red_balls_count_l2640_264035

theorem red_balls_count (total : ℕ) (prob : ℚ) : 
  total = 15 →
  prob = 1 / 21 →
  ∃ (r : ℕ), r ≤ total ∧ 
    (r : ℚ) / total * ((r : ℚ) - 1) / (total - 1 : ℚ) = prob ∧
    r = 5 :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l2640_264035


namespace NUMINAMATH_CALUDE_no_real_solutions_l2640_264048

theorem no_real_solutions :
  ∀ x : ℝ, (2*x - 3*x + 7)^2 + 4 ≠ -|2*x| :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2640_264048


namespace NUMINAMATH_CALUDE_fraction_simplification_l2640_264009

/-- Given a, b, c, x, y, z are real numbers, prove that the given complex fraction 
    is equal to the simplified form. -/
theorem fraction_simplification (a b c x y z : ℝ) :
  (c * z * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + 
   b * z * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (c * z + b * z) = 
  a^3 * x^3 + c^3 * z^3 + (3 * c * z * a^3 * y^3 + 3 * b * z * c^3 * x^3) / (c * z + b * z) :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2640_264009


namespace NUMINAMATH_CALUDE_divide_n_plus_one_l2640_264007

theorem divide_n_plus_one (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divide_n_plus_one_l2640_264007


namespace NUMINAMATH_CALUDE_basketball_practice_average_l2640_264012

/-- The average practice time per day for a basketball player over a week -/
theorem basketball_practice_average (weekday_hours : ℝ) (weekend_hours : ℝ) (weekdays : ℕ) (weekend_days : ℕ) :
  weekday_hours = 2 →
  weekend_hours = 11 →
  weekdays = 5 →
  weekend_days = 2 →
  (weekday_hours * weekdays + weekend_hours) / (weekdays + weekend_days) = 3 := by
  sorry

end NUMINAMATH_CALUDE_basketball_practice_average_l2640_264012


namespace NUMINAMATH_CALUDE_product_remainder_mod_25_l2640_264000

theorem product_remainder_mod_25 : (1523 * 1857 * 1919 * 2012) % 25 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_25_l2640_264000


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_4_8_l2640_264066

theorem gcf_lcm_sum_4_8 : Nat.gcd 4 8 + Nat.lcm 4 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_4_8_l2640_264066


namespace NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l2640_264040

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 30
  let hole_depth : ℝ := 8
  let small_hole_diameter : ℝ := 2
  let large_hole_diameter : ℝ := 3
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2) ^ 3
  let small_hole_volume := π * (small_hole_diameter / 2) ^ 2 * hole_depth
  let large_hole_volume := π * (large_hole_diameter / 2) ^ 2 * hole_depth
  sphere_volume - 2 * small_hole_volume - large_hole_volume = 4466 * π :=
by sorry

end NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l2640_264040


namespace NUMINAMATH_CALUDE_hotel_stay_cost_l2640_264081

/-- The total cost for a group staying at a hotel. -/
def total_cost (cost_per_night_per_person : ℕ) (num_people : ℕ) (num_nights : ℕ) : ℕ :=
  cost_per_night_per_person * num_people * num_nights

/-- Theorem: The total cost for 3 people staying 3 nights at $40 per night per person is $360. -/
theorem hotel_stay_cost : total_cost 40 3 3 = 360 := by
  sorry

end NUMINAMATH_CALUDE_hotel_stay_cost_l2640_264081


namespace NUMINAMATH_CALUDE_magician_balls_l2640_264016

/-- Represents the number of balls in the box after each operation -/
def BallCount : ℕ → ℕ
  | 0 => 7  -- Initial count
  | n + 1 => BallCount n + 6 * (BallCount n - 1)  -- After each operation

/-- The form that the ball count must follow -/
def ValidForm (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k + 7

theorem magician_balls :
  (∀ n : ℕ, ValidForm (BallCount n)) ∧
  ValidForm 1993 ∧
  ¬ValidForm 1990 ∧
  ¬ValidForm 1991 ∧
  ¬ValidForm 1992 := by sorry

end NUMINAMATH_CALUDE_magician_balls_l2640_264016


namespace NUMINAMATH_CALUDE_fraction_calculation_l2640_264029

theorem fraction_calculation : 
  (3/7 + 2/3) / (5/12 + 1/4) = 23/14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2640_264029


namespace NUMINAMATH_CALUDE_isabella_currency_exchange_l2640_264063

theorem isabella_currency_exchange :
  ∃ d : ℕ+, 
    (10 : ℚ) / 7 * d.val - 60 = d.val ∧ 
    (d.val / 100 + (d.val / 10) % 10 + d.val % 10 = 5) := by
  sorry

end NUMINAMATH_CALUDE_isabella_currency_exchange_l2640_264063


namespace NUMINAMATH_CALUDE_road_repair_groups_equivalent_l2640_264028

/-- The number of persons in the second group repairing the road -/
def second_group_size : ℕ := 30

/-- The number of persons in the first group -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 10

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

/-- The total man-hours required to complete the road repair -/
def total_man_hours : ℕ := first_group_size * first_group_days * first_group_hours_per_day

theorem road_repair_groups_equivalent :
  second_group_size * second_group_days * second_group_hours_per_day = total_man_hours :=
by sorry

end NUMINAMATH_CALUDE_road_repair_groups_equivalent_l2640_264028


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2640_264068

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to x-axis --/
def symmetricXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- Theorem: If A(3,a) is symmetric to B(b,4) with respect to x-axis, then a + b = -1 --/
theorem symmetric_points_sum (a b : ℝ) : 
  let A : Point := ⟨3, a⟩
  let B : Point := ⟨b, 4⟩
  symmetricXAxis A B → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2640_264068


namespace NUMINAMATH_CALUDE_wheel_radii_theorem_l2640_264067

/-- Given two wheels A and B with radii R and r respectively, 
    if the ratio of their rotational speeds is 4:5 and 
    the distance between their centers is 9, 
    then R = 2.5 and r = 2. -/
theorem wheel_radii_theorem (R r : ℝ) : 
  (4 : ℝ) / 5 = 1200 / 1500 →  -- ratio of rotational speeds
  2 * (R + r) = 9 →            -- distance between centers
  R = 2.5 ∧ r = 2 := by
  sorry


end NUMINAMATH_CALUDE_wheel_radii_theorem_l2640_264067


namespace NUMINAMATH_CALUDE_smallest_palindromic_prime_l2640_264010

/-- A function that checks if a number is a three-digit palindrome with hundreds digit 2 -/
def isValidPalindrome (n : ℕ) : Prop :=
  n ≥ 200 ∧ n ≤ 299 ∧ (n / 100 = 2) ∧ (n % 10 = n / 100)

/-- The theorem stating that 232 is the smallest three-digit palindromic prime with hundreds digit 2 -/
theorem smallest_palindromic_prime :
  isValidPalindrome 232 ∧ Nat.Prime 232 ∧
  ∀ n < 232, isValidPalindrome n → ¬Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_smallest_palindromic_prime_l2640_264010


namespace NUMINAMATH_CALUDE_optimal_chip_purchase_l2640_264079

/-- Represents the purchase of chips with given constraints -/
structure ChipPurchase where
  priceA : ℕ  -- Unit price of type A chips
  priceB : ℕ  -- Unit price of type B chips
  quantityA : ℕ  -- Quantity of type A chips
  quantityB : ℕ  -- Quantity of type B chips
  total_cost : ℕ  -- Total cost of the purchase

/-- Theorem stating the optimal purchase and minimum cost -/
theorem optimal_chip_purchase :
  ∃ (purchase : ChipPurchase),
    -- Conditions
    purchase.priceB = purchase.priceA + 9 ∧
    purchase.quantityA * purchase.priceA = 3120 ∧
    purchase.quantityB * purchase.priceB = 4200 ∧
    purchase.quantityA = purchase.quantityB ∧
    purchase.quantityA + purchase.quantityB = 200 ∧
    4 * purchase.quantityA ≥ purchase.quantityB ∧
    3 * purchase.quantityA ≤ purchase.quantityB ∧
    -- Correct answer
    purchase.priceA = 26 ∧
    purchase.priceB = 35 ∧
    purchase.quantityA = 50 ∧
    purchase.quantityB = 150 ∧
    purchase.total_cost = 6550 ∧
    -- Minimum cost property
    (∀ (other : ChipPurchase),
      other.priceB = other.priceA + 9 →
      other.quantityA + other.quantityB = 200 →
      4 * other.quantityA ≥ other.quantityB →
      3 * other.quantityA ≤ other.quantityB →
      other.total_cost ≥ purchase.total_cost) :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_chip_purchase_l2640_264079


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l2640_264099

theorem sum_of_five_consecutive_even_integers (m : ℤ) : 
  (m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l2640_264099


namespace NUMINAMATH_CALUDE_hundreds_digit_of_8_pow_1234_l2640_264065

-- Define a function to get the last three digits of 8^n
def lastThreeDigits (n : ℕ) : ℕ := 8^n % 1000

-- Define the cycle length of the last three digits of 8^n
def cycleLengthOf8 : ℕ := 20

-- Theorem statement
theorem hundreds_digit_of_8_pow_1234 :
  (lastThreeDigits 1234) / 100 = 1 :=
sorry

end NUMINAMATH_CALUDE_hundreds_digit_of_8_pow_1234_l2640_264065


namespace NUMINAMATH_CALUDE_candy_necklace_problem_l2640_264039

/-- Candy necklace problem -/
theorem candy_necklace_problem 
  (pieces_per_necklace : ℕ) 
  (pieces_per_block : ℕ) 
  (blocks_used : ℕ) 
  (h1 : pieces_per_necklace = 10)
  (h2 : pieces_per_block = 30)
  (h3 : blocks_used = 3)
  : (blocks_used * pieces_per_block) / pieces_per_necklace - 1 = 8 := by
  sorry

#check candy_necklace_problem

end NUMINAMATH_CALUDE_candy_necklace_problem_l2640_264039


namespace NUMINAMATH_CALUDE_angle_E_measure_l2640_264014

/-- A quadrilateral with specific angle relationships -/
structure SpecialQuadrilateral where
  E : ℝ  -- Angle E in degrees
  F : ℝ  -- Angle F in degrees
  G : ℝ  -- Angle G in degrees
  H : ℝ  -- Angle H in degrees
  angle_sum : E + F + G + H = 360  -- Sum of angles in a quadrilateral
  E_eq_5H : E = 5 * H  -- Relationship between E and H
  E_eq_4G : E = 4 * G  -- Relationship between E and G
  E_eq_5div3F : E = 5 / 3 * F  -- Relationship between E and F

/-- The measure of angle E in the special quadrilateral -/
theorem angle_E_measure (q : SpecialQuadrilateral) : q.E = 1440 / 11 := by
  sorry


end NUMINAMATH_CALUDE_angle_E_measure_l2640_264014


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l2640_264078

theorem sum_of_x_solutions_is_zero (y : ℝ) (h1 : y = 9) (h2 : ∃ x : ℝ, x^2 + y^2 = 169) :
  ∃ x₁ x₂ : ℝ, x₁^2 + y^2 = 169 ∧ x₂^2 + y^2 = 169 ∧ x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l2640_264078


namespace NUMINAMATH_CALUDE_ripe_fruits_weight_l2640_264005

/-- Given the following conditions:
    - Total fruits: 14 apples, 10 pears, 5 lemons
    - Average weights of ripe fruits: apples 150g, pears 200g, lemons 100g
    - Average weights of unripe fruits: apples 120g, pears 180g, lemons 80g
    - Unripe fruits: 6 apples, 4 pears, 2 lemons
    Prove that the total weight of ripe fruits is 2700 grams -/
theorem ripe_fruits_weight (
  total_apples : ℕ) (total_pears : ℕ) (total_lemons : ℕ)
  (ripe_apple_weight : ℕ) (ripe_pear_weight : ℕ) (ripe_lemon_weight : ℕ)
  (unripe_apple_weight : ℕ) (unripe_pear_weight : ℕ) (unripe_lemon_weight : ℕ)
  (unripe_apples : ℕ) (unripe_pears : ℕ) (unripe_lemons : ℕ)
  (h1 : total_apples = 14)
  (h2 : total_pears = 10)
  (h3 : total_lemons = 5)
  (h4 : ripe_apple_weight = 150)
  (h5 : ripe_pear_weight = 200)
  (h6 : ripe_lemon_weight = 100)
  (h7 : unripe_apple_weight = 120)
  (h8 : unripe_pear_weight = 180)
  (h9 : unripe_lemon_weight = 80)
  (h10 : unripe_apples = 6)
  (h11 : unripe_pears = 4)
  (h12 : unripe_lemons = 2) :
  (total_apples - unripe_apples) * ripe_apple_weight +
  (total_pears - unripe_pears) * ripe_pear_weight +
  (total_lemons - unripe_lemons) * ripe_lemon_weight = 2700 := by
  sorry

end NUMINAMATH_CALUDE_ripe_fruits_weight_l2640_264005


namespace NUMINAMATH_CALUDE_ammunition_depot_explosion_probability_l2640_264052

theorem ammunition_depot_explosion_probability 
  (p_first : ℝ) 
  (p_others : ℝ) 
  (h1 : p_first = 0.025) 
  (h2 : p_others = 0.1) : 
  1 - (1 - p_first) * (1 - p_others) * (1 - p_others) = 0.21025 := by
  sorry

end NUMINAMATH_CALUDE_ammunition_depot_explosion_probability_l2640_264052


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l2640_264015

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube (x : ℕ) (h : x = 11 * 36 * 54) :
  ∃! y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 → is_perfect_cube (x * z) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l2640_264015


namespace NUMINAMATH_CALUDE_solve_for_a_l2640_264087

theorem solve_for_a : 
  ∀ a : ℝ, (∃ x : ℝ, x = 3 ∧ a * x - 5 = x + 1) → a = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2640_264087


namespace NUMINAMATH_CALUDE_player_b_winning_strategy_l2640_264050

/-- Represents the game state with two players on a line -/
structure GameState where
  L : ℕ+  -- Distance between initial positions (positive integer)
  a : ℕ+  -- Move distance for player A (positive integer)
  b : ℕ+  -- Move distance for player B (positive integer)
  h : a < b  -- Condition that a is less than b

/-- Winning condition for player B -/
def winning_condition (g : GameState) : Prop :=
  g.b = 2 * g.a ∧ ∃ k : ℕ, g.L = k * g.a

/-- Theorem stating the necessary and sufficient conditions for player B to have a winning strategy -/
theorem player_b_winning_strategy (g : GameState) :
  winning_condition g ↔ ∃ (strategy : Unit), True  -- Replace True with actual strategy type when implementing
:= by sorry

end NUMINAMATH_CALUDE_player_b_winning_strategy_l2640_264050


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2640_264072

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ (1 - Real.sqrt 2 * i) / (Real.sqrt 2 + 1) = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2640_264072


namespace NUMINAMATH_CALUDE_layla_nahima_score_difference_l2640_264013

theorem layla_nahima_score_difference :
  ∀ (layla_score nahima_score : ℕ),
    layla_score = 70 →
    layla_score + nahima_score = 112 →
    layla_score - nahima_score = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_layla_nahima_score_difference_l2640_264013


namespace NUMINAMATH_CALUDE_frank_final_position_l2640_264037

/-- Represents Frank's position relative to his starting point -/
def dance_position (back1 forward1 back2 : ℤ) : ℤ :=
  -back1 + forward1 - back2 + 2 * back2

/-- Proves that Frank's final position is 7 steps forward from his starting point -/
theorem frank_final_position :
  dance_position 5 10 2 = 7 := by sorry

end NUMINAMATH_CALUDE_frank_final_position_l2640_264037


namespace NUMINAMATH_CALUDE_profit_division_ratio_l2640_264097

/-- Represents the capital contribution and duration for a business partner -/
structure Contribution where
  capital : ℕ
  months : ℕ

/-- Calculates the total capital contribution over time -/
def totalContribution (c : Contribution) : ℕ := c.capital * c.months

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplifyRatio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

/-- The main theorem stating the profit division ratio -/
theorem profit_division_ratio 
  (a_initial : ℕ) 
  (b_capital : ℕ) 
  (total_months : ℕ) 
  (b_join_month : ℕ) 
  (h1 : a_initial = 3500)
  (h2 : b_capital = 31500)
  (h3 : total_months = 12)
  (h4 : b_join_month = 10) :
  simplifyRatio 
    (totalContribution { capital := a_initial, months := total_months })
    (totalContribution { capital := b_capital, months := total_months - b_join_month }) 
  = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_profit_division_ratio_l2640_264097


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2640_264038

theorem tan_alpha_value (α : Real)
  (h1 : α ∈ Set.Ioo (π / 2) π)
  (h2 : Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) = (1 - 2 * Real.cos α) / (2 * Real.sin (α / 2)^2 - 1)) :
  Real.tan α = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2640_264038


namespace NUMINAMATH_CALUDE_range_of_a_l2640_264036

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5-2*a))^x > (-(5-2*a))^y

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Iic (-2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2640_264036


namespace NUMINAMATH_CALUDE_probability_three_black_face_cards_l2640_264024

theorem probability_three_black_face_cards (total_cards : ℕ) (drawn_cards : ℕ) 
  (black_face_cards : ℕ) (non_black_face_cards : ℕ) :
  total_cards = 36 →
  drawn_cards = 6 →
  black_face_cards = 8 →
  non_black_face_cards = 28 →
  (Nat.choose black_face_cards 3 * Nat.choose non_black_face_cards 3) / 
  Nat.choose total_cards drawn_cards = 11466 / 121737 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_black_face_cards_l2640_264024


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l2640_264057

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 8) :
  ∃ (k : ℕ+), k ≥ 32 ∧ Nat.gcd (8 * m) (12 * n) = k ∧
  ∀ (l : ℕ+), Nat.gcd (8 * m) (12 * n) = l → l ≥ 32 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l2640_264057


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l2640_264071

-- Define the quadrilateral ABCD
def Quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let DA := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  AB = 13 ∧ BC = 10 ∧ CD = 8 ∧ DA = 11

-- Define the inscribed circle
def InscribedCircle (A B C D : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) : Prop :=
  Quadrilateral A B C D ∧
  ∀ P : ℝ × ℝ, (P ∈ Set.range (fun t => (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2)) ∨
                 P ∈ Set.range (fun t => (t * C.1 + (1 - t) * B.1, t * C.2 + (1 - t) * B.2)) ∨
                 P ∈ Set.range (fun t => (t * D.1 + (1 - t) * C.1, t * D.2 + (1 - t) * C.2)) ∨
                 P ∈ Set.range (fun t => (t * A.1 + (1 - t) * D.1, t * A.2 + (1 - t) * D.2))) →
                Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) ≥ r

-- Theorem statement
theorem largest_inscribed_circle_radius :
  ∀ A B C D O : ℝ × ℝ,
  ∀ r : ℝ,
  InscribedCircle A B C D O r →
  r ≤ 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l2640_264071


namespace NUMINAMATH_CALUDE_book_pages_count_l2640_264073

/-- The total number of pages in Isabella's book -/
def total_pages : ℕ := 288

/-- The number of days Isabella took to read the book -/
def total_days : ℕ := 8

/-- The average number of pages Isabella read daily for the first four days -/
def first_four_days_avg : ℕ := 28

/-- The average number of pages Isabella read daily for the next three days -/
def next_three_days_avg : ℕ := 52

/-- The number of pages Isabella read on the final day -/
def final_day_pages : ℕ := 20

/-- Theorem stating that the total number of pages in the book is 288 -/
theorem book_pages_count : 
  (4 * first_four_days_avg + 3 * next_three_days_avg + final_day_pages = total_pages) ∧
  (total_days = 8) := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l2640_264073


namespace NUMINAMATH_CALUDE_division_multiplication_result_l2640_264018

theorem division_multiplication_result : (-1 : ℚ) / (-5 : ℚ) * (-1/5 : ℚ) = -1/25 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l2640_264018


namespace NUMINAMATH_CALUDE_set_A_determination_l2640_264056

def U : Set ℕ := {0, 1, 2, 4}

theorem set_A_determination (A : Set ℕ) (h : (U \ A) = {1, 2}) : A = {0, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_A_determination_l2640_264056


namespace NUMINAMATH_CALUDE_function_derivative_value_l2640_264017

theorem function_derivative_value (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 * (deriv f (π/3)) + Real.sin x) :
  deriv f (π/3) = 3 / (6 - 4*π) := by sorry

end NUMINAMATH_CALUDE_function_derivative_value_l2640_264017


namespace NUMINAMATH_CALUDE_f_difference_at_three_l2640_264076

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 3*x^3 + 2*x^2 + 7*x

-- Theorem statement
theorem f_difference_at_three : f 3 - f (-3) = 690 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_three_l2640_264076


namespace NUMINAMATH_CALUDE_student_takehome_pay_l2640_264084

/-- Calculates the take-home pay for a well-performing student at a fast-food chain --/
def takehomePay (baseSalary bonus taxRate : ℚ) : ℚ :=
  let totalEarnings := baseSalary + bonus
  let incomeTax := totalEarnings * taxRate
  totalEarnings - incomeTax

/-- Theorem: The take-home pay for a well-performing student is 26,100 rubles --/
theorem student_takehome_pay :
  takehomePay 25000 5000 (13/100) = 26100 := by
  sorry

#eval takehomePay 25000 5000 (13/100)

end NUMINAMATH_CALUDE_student_takehome_pay_l2640_264084


namespace NUMINAMATH_CALUDE_triangle_side_length_l2640_264025

theorem triangle_side_length (A B C : ℝ) (angleA angleB : ℝ) (sideAC : ℝ) :
  angleA = π / 4 →
  angleB = 5 * π / 12 →
  sideAC = 6 →
  ∃ (sideBC : ℝ), sideBC = 6 * (Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2640_264025


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l2640_264043

def f (x : ℝ) := x^3 - 3*x - 3

theorem f_has_root_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_f_has_root_in_interval_l2640_264043


namespace NUMINAMATH_CALUDE_max_distance_on_curve_l2640_264059

/-- The maximum distance between a point on the curve y² = 4 - 2x² and the point (0, -√2) -/
theorem max_distance_on_curve : ∃ (max_dist : ℝ),
  max_dist = 2 + Real.sqrt 2 ∧
  ∀ (x y : ℝ),
    y^2 = 4 - 2*x^2 →
    Real.sqrt ((x - 0)^2 + (y - (-Real.sqrt 2))^2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_on_curve_l2640_264059


namespace NUMINAMATH_CALUDE_max_integer_value_of_expression_l2640_264054

theorem max_integer_value_of_expression (x : ℝ) : 
  (4 * x^2 + 8 * x + 21) / (4 * x^2 + 8 * x + 9) ≤ 7/3 ∧ 
  ∃ (y : ℝ), (4 * y^2 + 8 * y + 21) / (4 * y^2 + 8 * y + 9) > 2 ∧
  ∀ (z : ℝ), (4 * z^2 + 8 * z + 21) / (4 * z^2 + 8 * z + 9) < 3 := by
  sorry

end NUMINAMATH_CALUDE_max_integer_value_of_expression_l2640_264054


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2640_264096

theorem sum_of_fractions : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (5 / 10 : ℚ) + 
  (6 / 10 : ℚ) + (7 / 10 : ℚ) + (8 / 10 : ℚ) + (10 / 10 : ℚ) + (60 / 10 : ℚ) = 
  (106 : ℚ) / 10 := by
  sorry

#eval (106 : ℚ) / 10  -- This should evaluate to 10.6

end NUMINAMATH_CALUDE_sum_of_fractions_l2640_264096


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2640_264032

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 3| + |x - 6| - |3*x - 9|

-- Define the domain of x
def domain (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 12

-- Theorem statement
theorem sum_of_max_min_g :
  ∃ (max min : ℝ), 
    (∀ x, domain x → g x ≤ max) ∧
    (∃ x, domain x ∧ g x = max) ∧
    (∀ x, domain x → min ≤ g x) ∧
    (∃ x, domain x ∧ g x = min) ∧
    max + min = -6 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2640_264032


namespace NUMINAMATH_CALUDE_problem_solution_l2640_264092

theorem problem_solution (p q : ℚ) (h1 : 3 / p = 6) (h2 : 3 / q = 18) : p - q = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2640_264092


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_polar_axis_l2640_264021

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  -- The equation of the line in the form ρ sin θ = k
  k : ℝ

/-- Checks if a point lies on a given polar line -/
def isOnLine (p : PolarPoint) (l : PolarLine) : Prop :=
  p.ρ * Real.sin p.θ = l.k

theorem line_through_point_parallel_to_polar_axis 
  (p : PolarPoint) (l : PolarLine) 
  (h1 : p.ρ = 1) 
  (h2 : p.θ = Real.pi / 2) 
  (h3 : l.k = 1) : 
  ∀ q : PolarPoint, isOnLine q l ↔ q.ρ * Real.sin q.θ = 1 := by
  sorry

#check line_through_point_parallel_to_polar_axis

end NUMINAMATH_CALUDE_line_through_point_parallel_to_polar_axis_l2640_264021


namespace NUMINAMATH_CALUDE_inverse_sqrt_problem_l2640_264061

-- Define the relationship between x and y
def inverse_sqrt_relation (x y : ℝ) (k : ℝ) : Prop :=
  y * Real.sqrt x = k

-- Define the theorem
theorem inverse_sqrt_problem (x y k : ℝ) :
  inverse_sqrt_relation x y k →
  inverse_sqrt_relation 2 4 k →
  y = 1 →
  x = 32 := by sorry

end NUMINAMATH_CALUDE_inverse_sqrt_problem_l2640_264061


namespace NUMINAMATH_CALUDE_min_editors_conference_l2640_264011

theorem min_editors_conference (total : ℕ) (writers : ℕ) (max_both : ℕ) :
  total = 100 →
  writers = 35 →
  max_both = 26 →
  ∃ (editors : ℕ) (both : ℕ),
    both ≤ max_both ∧
    editors ≥ 39 ∧
    total = writers + editors - both + 2 * both :=
by
  sorry

end NUMINAMATH_CALUDE_min_editors_conference_l2640_264011


namespace NUMINAMATH_CALUDE_optimal_pricing_strategy_l2640_264086

/-- Represents the pricing strategy of a merchant -/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  sale_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price given the list price and purchase discount -/
def purchase_price (m : MerchantPricing) : ℝ :=
  m.list_price * (1 - m.purchase_discount)

/-- Calculates the selling price given the marked price and sale discount -/
def selling_price (m : MerchantPricing) : ℝ :=
  m.marked_price * (1 - m.sale_discount)

/-- Calculates the profit given the selling price and purchase price -/
def profit (m : MerchantPricing) : ℝ :=
  selling_price m - purchase_price m

/-- Theorem stating the optimal pricing strategy for the merchant -/
theorem optimal_pricing_strategy (m : MerchantPricing) 
  (h1 : m.purchase_discount = 0.25)
  (h2 : m.sale_discount = 0.20)
  (h3 : m.profit_margin = 0.25)
  (h4 : profit m = m.profit_margin * selling_price m) :
  m.marked_price = 1.25 * m.list_price := by
  sorry

end NUMINAMATH_CALUDE_optimal_pricing_strategy_l2640_264086


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l2640_264085

theorem max_value_of_sum_products (a b c d : ℕ) : 
  ({a, b, c, d} : Finset ℕ) = {1, 2, 4, 5} →
  a * b + b * c + c * d + d * a ≤ 36 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l2640_264085


namespace NUMINAMATH_CALUDE_distribute_5_3_l2640_264064

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    where each box must contain at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 150 ways to distribute 5 distinct objects into 3 distinct boxes,
    where each box must contain at least one object. -/
theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l2640_264064


namespace NUMINAMATH_CALUDE_prob_queen_then_diamond_correct_l2640_264093

/-- The probability of drawing a Queen first and a diamond second from a standard 52-card deck, without replacement -/
def prob_queen_then_diamond : ℚ := 18 / 221

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The number of diamond cards in a standard deck -/
def num_diamonds : ℕ := 13

/-- The number of non-diamond Queens in a standard deck -/
def num_non_diamond_queens : ℕ := 3

theorem prob_queen_then_diamond_correct :
  prob_queen_then_diamond = 
    (1 / deck_size * num_diamonds / (deck_size - 1)) + 
    (num_non_diamond_queens / deck_size * num_diamonds / (deck_size - 1)) := by
  sorry

end NUMINAMATH_CALUDE_prob_queen_then_diamond_correct_l2640_264093


namespace NUMINAMATH_CALUDE_sales_price_calculation_l2640_264094

theorem sales_price_calculation (C S : ℝ) 
  (h1 : 1.7 * C = 51)  -- Gross profit is 170% of cost and equals $51
  (h2 : S = C + 1.7 * C)  -- Sales price is cost plus gross profit
  : S = 81 := by
  sorry

end NUMINAMATH_CALUDE_sales_price_calculation_l2640_264094


namespace NUMINAMATH_CALUDE_typist_salary_calculation_l2640_264049

theorem typist_salary_calculation (original_salary : ℝ) (raise_percentage : ℝ) (reduction_percentage : ℝ) : 
  original_salary = 2000 ∧ 
  raise_percentage = 10 ∧ 
  reduction_percentage = 5 → 
  original_salary * (1 + raise_percentage / 100) * (1 - reduction_percentage / 100) = 2090 :=
by sorry

end NUMINAMATH_CALUDE_typist_salary_calculation_l2640_264049


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2640_264080

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2640_264080


namespace NUMINAMATH_CALUDE_random_events_identification_l2640_264004

structure Event where
  description : String
  is_random : Bool

def event1 : Event := { 
  description := "An object will fall freely under the influence of gravity alone",
  is_random := false 
}

def event2 : Event := { 
  description := "The equation x^2 + 2x + 8 = 0 has two real roots",
  is_random := false 
}

def event3 : Event := { 
  description := "A certain information desk receives more than 10 requests for information consultation during a certain period of the day",
  is_random := true 
}

def event4 : Event := { 
  description := "It will rain next Saturday",
  is_random := true 
}

def events : List Event := [event1, event2, event3, event4]

theorem random_events_identification : 
  (events.filter (λ e => e.is_random)).map (λ e => e.description) = 
  [event3.description, event4.description] := by sorry

end NUMINAMATH_CALUDE_random_events_identification_l2640_264004


namespace NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l2640_264045

theorem largest_integer_for_negative_quadratic : 
  ∀ x : ℤ, x^2 - 11*x + 28 < 0 → x ≤ 6 :=
by sorry

theorem six_satisfies_inequality : 
  (6 : ℤ)^2 - 11*6 + 28 < 0 :=
by sorry

theorem seven_does_not_satisfy_inequality : 
  (7 : ℤ)^2 - 11*7 + 28 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_six_satisfies_inequality_seven_does_not_satisfy_inequality_l2640_264045


namespace NUMINAMATH_CALUDE_runner_problem_l2640_264030

theorem runner_problem (v : ℝ) (h : v > 0) : 
  (40 / v = 20 / v + 11) → (40 / (v / 2) = 22) :=
by
  sorry

end NUMINAMATH_CALUDE_runner_problem_l2640_264030


namespace NUMINAMATH_CALUDE_tom_swim_time_l2640_264041

/-- Proves that Tom swam for 2 hours given the conditions of the problem -/
theorem tom_swim_time (swim_speed : ℝ) (run_speed_multiplier : ℝ) (total_distance : ℝ) :
  swim_speed = 2 →
  run_speed_multiplier = 4 →
  total_distance = 12 →
  ∃ (swim_time : ℝ),
    swim_time * swim_speed + (swim_time / 2) * (run_speed_multiplier * swim_speed) = total_distance ∧
    swim_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_tom_swim_time_l2640_264041


namespace NUMINAMATH_CALUDE_rational_sqrt5_zero_quadratic_roots_sum_difference_l2640_264055

theorem rational_sqrt5_zero (a b : ℚ) (h : a + b * Real.sqrt 5 = 0) : a = 0 ∧ b = 0 := by
  sorry

theorem quadratic_roots_sum_difference (k : ℝ) :
  k ≠ 0 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    4 * k * x₁^2 - 4 * k * x₁ + k + 1 = 0 ∧
    4 * k * x₂^2 - 4 * k * x₂ + k + 1 = 0 ∧
    x₁^2 + x₂^2 - 2 * x₁ * x₂ = 1/2) →
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt5_zero_quadratic_roots_sum_difference_l2640_264055


namespace NUMINAMATH_CALUDE_max_chocolates_buyable_l2640_264033

def total_money : ℚ := 24.50
def chocolate_price : ℚ := 2.20

theorem max_chocolates_buyable : 
  ⌊total_money / chocolate_price⌋ = 11 := by sorry

end NUMINAMATH_CALUDE_max_chocolates_buyable_l2640_264033


namespace NUMINAMATH_CALUDE_log_sum_equality_l2640_264044

theorem log_sum_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l2640_264044


namespace NUMINAMATH_CALUDE_leo_has_largest_answer_l2640_264001

def starting_number : ℕ := 12

def rodrigo_process (n : ℕ) : ℕ := ((n - 3)^2 + 4)

def samantha_process (n : ℕ) : ℕ := (n^2 - 5 + 4)

def leo_process (n : ℕ) : ℕ := ((n - 3 + 4)^2)

theorem leo_has_largest_answer :
  leo_process starting_number > rodrigo_process starting_number ∧
  leo_process starting_number > samantha_process starting_number :=
sorry

end NUMINAMATH_CALUDE_leo_has_largest_answer_l2640_264001


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2640_264058

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2*x + 1, 3)
  let b : ℝ × ℝ := (2 - x, 1)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2640_264058


namespace NUMINAMATH_CALUDE_seventh_term_value_l2640_264088

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first five terms
  sum_first_five : ℚ
  -- The sixth term
  sixth_term : ℚ
  -- Property: The sum of the first five terms is 15
  sum_property : sum_first_five = 15
  -- Property: The sixth term is 7
  sixth_property : sixth_term = 7

/-- The seventh term of the arithmetic sequence -/
def seventh_term (seq : ArithmeticSequence) : ℚ := 25/3

/-- Theorem: The seventh term of the arithmetic sequence is 25/3 -/
theorem seventh_term_value (seq : ArithmeticSequence) :
  seventh_term seq = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_value_l2640_264088


namespace NUMINAMATH_CALUDE_square_inequality_l2640_264020

theorem square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l2640_264020


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2640_264002

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (1 / Real.sqrt (x + 2 * y + 6) + 1 / Real.sqrt (y + 2 * z + 6) + 1 / Real.sqrt (z + 2 * x + 6)) ≤
  (x / Real.sqrt (x^2 + 4 * Real.sqrt y + 4 * Real.sqrt z) +
   y / Real.sqrt (y^2 + 4 * Real.sqrt z + 4 * Real.sqrt x) +
   z / Real.sqrt (z^2 + 4 * Real.sqrt x + 4 * Real.sqrt y)) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2640_264002


namespace NUMINAMATH_CALUDE_positive_integer_solutions_for_equation_l2640_264008

theorem positive_integer_solutions_for_equation :
  ∀ m n : ℕ+,
  m^2 = n^2 + m + n + 2018 ↔ (m = 1010 ∧ n = 1008) ∨ (m = 506 ∧ n = 503) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_for_equation_l2640_264008


namespace NUMINAMATH_CALUDE_stream_speed_l2640_264034

/-- Proves that the speed of a stream is 3 km/h given specific rowing conditions. -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (trip_time : ℝ) 
  (h1 : downstream_distance = 75)
  (h2 : upstream_distance = 45)
  (h3 : trip_time = 5) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = trip_time * (boat_speed + stream_speed) ∧
    upstream_distance = trip_time * (boat_speed - stream_speed) ∧
    stream_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2640_264034


namespace NUMINAMATH_CALUDE_equation_solutions_l2640_264031

theorem equation_solutions : 
  (∃ (s : Set ℝ), s = {0, 3} ∧ ∀ x ∈ s, 4 * x^2 = 12 * x) ∧
  (∃ (t : Set ℝ), t = {-3, -1} ∧ ∀ x ∈ t, x^2 + 4 * x + 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2640_264031


namespace NUMINAMATH_CALUDE_olivia_spent_l2640_264046

/-- Calculates the amount spent given initial amount, amount collected, and amount left after shopping. -/
def amount_spent (initial : ℕ) (collected : ℕ) (left : ℕ) : ℕ :=
  initial + collected - left

/-- Proves that Olivia spent 89 dollars given the problem conditions. -/
theorem olivia_spent (initial : ℕ) (collected : ℕ) (left : ℕ)
  (h1 : initial = 100)
  (h2 : collected = 148)
  (h3 : left = 159) :
  amount_spent initial collected left = 89 := by
  sorry

#eval amount_spent 100 148 159

end NUMINAMATH_CALUDE_olivia_spent_l2640_264046


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l2640_264082

theorem cube_plus_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 3) : 
  r^3 + 1/r^3 = 0 := by sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l2640_264082


namespace NUMINAMATH_CALUDE_fraction_simplification_l2640_264091

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2640_264091


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2640_264095

theorem simplify_and_evaluate (x : ℝ) (h : x = 4) :
  (x - 1 - 3 / (x + 1)) / ((x^2 + 2*x) / (x + 1)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2640_264095


namespace NUMINAMATH_CALUDE_expression_simplification_l2640_264060

theorem expression_simplification (x y : ℚ) (hx : x = 4) (hy : y = -1/4) :
  ((x + y) * (3 * x - y) + y^2) / (-x) = -23/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2640_264060


namespace NUMINAMATH_CALUDE_exact_time_proof_l2640_264027

def minutes_after_3 (h m : ℕ) : ℝ := 60 * (h - 3 : ℝ) + m

def minute_hand_position (t : ℝ) : ℝ := 6 * t

def hour_hand_position (t : ℝ) : ℝ := 90 + 0.5 * t

theorem exact_time_proof :
  ∃ (h m : ℕ), h = 3 ∧ m < 60 ∧
  let t := minutes_after_3 h m
  abs (minute_hand_position (t + 5) - hour_hand_position (t - 4)) = 178 ∧
  h = 3 ∧ m = 43 := by
  sorry

end NUMINAMATH_CALUDE_exact_time_proof_l2640_264027


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2640_264051

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 10 % 10 : ℚ) = ((n / 100) + (n % 10)) / 2 ∧
  n % 10 = 2 * (n / 100)

theorem count_valid_numbers : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid_number n) ∧ S.card = 2 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2640_264051
