import Mathlib

namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l3219_321994

def total_players : ℕ := 15
def quadruplets : ℕ := 4
def starters : ℕ := 5
def quadruplets_in_lineup : ℕ := 2

theorem basketball_lineup_combinations :
  (Nat.choose quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 990 := by
sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l3219_321994


namespace NUMINAMATH_CALUDE_tens_digit_of_2013_squared_minus_2013_l3219_321951

theorem tens_digit_of_2013_squared_minus_2013 : (2013^2 - 2013) % 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2013_squared_minus_2013_l3219_321951


namespace NUMINAMATH_CALUDE_pizza_problem_l3219_321908

theorem pizza_problem : ∃! (m d : ℕ), 
  m > 0 ∧ d > 0 ∧ 
  7 * m + 2 * d > 36 ∧
  8 * m + 4 * d < 48 := by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l3219_321908


namespace NUMINAMATH_CALUDE_northton_capsule_depth_l3219_321914

/-- The depth of Southton's time capsule in feet -/
def southton_depth : ℝ := 15

/-- The depth of Northton's time capsule in feet -/
def northton_depth : ℝ := 4 * southton_depth - 12

/-- Theorem stating the depth of Northton's time capsule -/
theorem northton_capsule_depth : northton_depth = 48 := by
  sorry

end NUMINAMATH_CALUDE_northton_capsule_depth_l3219_321914


namespace NUMINAMATH_CALUDE_inequality_proof_l3219_321996

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3219_321996


namespace NUMINAMATH_CALUDE_factorization_equality_l3219_321933

theorem factorization_equality (a b : ℝ) :
  (a - b)^4 + (a + b)^4 + (a + b)^2 * (a - b)^2 = (3*a^2 + b^2) * (a^2 + 3*b^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3219_321933


namespace NUMINAMATH_CALUDE_yoongi_total_carrots_l3219_321956

/-- The number of carrots Yoongi has -/
def yoongi_carrots (initial : ℕ) (from_sister : ℕ) : ℕ :=
  initial + from_sister

theorem yoongi_total_carrots :
  yoongi_carrots 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_total_carrots_l3219_321956


namespace NUMINAMATH_CALUDE_smallest_number_with_prime_property_l3219_321903

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def remove_first_digit (n : ℕ) : ℕ := n % 1000

theorem smallest_number_with_prime_property : 
  ∃! n : ℕ, 
    (∀ m : ℕ, m < n → 
      ¬(∃ p q : ℕ, 
        is_prime p ∧ 
        is_prime q ∧ 
        remove_first_digit m = 4 * p ∧ 
        remove_first_digit m + 1 = 5 * q)) ∧
    (∃ p q : ℕ, 
      is_prime p ∧ 
      is_prime q ∧ 
      remove_first_digit n = 4 * p ∧ 
      remove_first_digit n + 1 = 5 * q) ∧
    n = 1964 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_prime_property_l3219_321903


namespace NUMINAMATH_CALUDE_highest_power_of_seven_in_100_factorial_l3219_321967

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem highest_power_of_seven_in_100_factorial :
  ∃ (k : ℕ), factorial 100 % (7^16) = 0 ∧ factorial 100 % (7^17) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_highest_power_of_seven_in_100_factorial_l3219_321967


namespace NUMINAMATH_CALUDE_total_triangles_is_68_l3219_321993

/-- Represents a rectangle divided into triangles as described in the problem -/
structure DividedRectangle where
  -- The rectangle is divided into 4 quarters
  quarters : Nat
  -- Number of smallest triangles in each quarter
  smallestTrianglesPerQuarter : Nat
  -- Number of half-inner rectangles from smaller triangles
  halfInnerRectangles : Nat
  -- Number of central and side isosceles triangles
  isoscelesTriangles : Nat
  -- Number of large right triangles covering half the rectangle
  largeRightTriangles : Nat
  -- Number of largest isosceles triangles
  largestIsoscelesTriangles : Nat

/-- Calculates the total number of triangles in the divided rectangle -/
def totalTriangles (r : DividedRectangle) : Nat :=
  r.smallestTrianglesPerQuarter * r.quarters +
  r.halfInnerRectangles +
  r.isoscelesTriangles +
  r.largeRightTriangles +
  r.largestIsoscelesTriangles

/-- The specific rectangle configuration from the problem -/
def problemRectangle : DividedRectangle where
  quarters := 4
  smallestTrianglesPerQuarter := 8
  halfInnerRectangles := 16
  isoscelesTriangles := 8
  largeRightTriangles := 8
  largestIsoscelesTriangles := 4

theorem total_triangles_is_68 : totalTriangles problemRectangle = 68 := by
  sorry


end NUMINAMATH_CALUDE_total_triangles_is_68_l3219_321993


namespace NUMINAMATH_CALUDE_angle_between_clock_hands_at_8_30_angle_between_clock_hands_at_8_30_is_75_l3219_321995

/-- The angle between clock hands at 8:30 -/
theorem angle_between_clock_hands_at_8_30 : ℝ :=
  let minute_hand_angle : ℝ := 30 * 6
  let hour_hand_angle : ℝ := 30 * 8 + 30 * 0.5
  |hour_hand_angle - minute_hand_angle|

/-- The angle between clock hands at 8:30 is 75 degrees -/
theorem angle_between_clock_hands_at_8_30_is_75 :
  angle_between_clock_hands_at_8_30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_clock_hands_at_8_30_angle_between_clock_hands_at_8_30_is_75_l3219_321995


namespace NUMINAMATH_CALUDE_rational_equation_solution_l3219_321990

theorem rational_equation_solution : 
  ∀ x : ℝ, x ≠ 2 → 
  ((3 * x - 9) / (x^2 - 6*x + 8) = (x + 1) / (x - 2)) ↔ 
  (x = 1 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l3219_321990


namespace NUMINAMATH_CALUDE_club_has_25_seniors_l3219_321989

/-- Represents a high school club with juniors and seniors -/
structure Club where
  juniors : ℕ
  seniors : ℕ
  project_juniors : ℕ
  project_seniors : ℕ

/-- The conditions of the problem -/
def club_conditions (c : Club) : Prop :=
  c.juniors + c.seniors = 50 ∧
  c.project_juniors = (40 * c.juniors) / 100 ∧
  c.project_seniors = (20 * c.seniors) / 100 ∧
  c.project_juniors = 2 * c.project_seniors

/-- The theorem stating that a club satisfying the conditions has 25 seniors -/
theorem club_has_25_seniors (c : Club) (h : club_conditions c) : c.seniors = 25 := by
  sorry


end NUMINAMATH_CALUDE_club_has_25_seniors_l3219_321989


namespace NUMINAMATH_CALUDE_prob_eight_rolls_divisible_by_four_l3219_321922

/-- The probability that a single die roll is even -/
def p_even : ℚ := 1/2

/-- The number of dice rolls -/
def n : ℕ := 8

/-- The probability mass function of the binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability that the product of n dice rolls is divisible by 4 -/
def prob_divisible_by_four (n : ℕ) (p : ℚ) : ℚ :=
  1 - (binomial_pmf n p 0 + binomial_pmf n p 1)

theorem prob_eight_rolls_divisible_by_four :
  prob_divisible_by_four n p_even = 247/256 := by
  sorry

#eval prob_divisible_by_four n p_even

end NUMINAMATH_CALUDE_prob_eight_rolls_divisible_by_four_l3219_321922


namespace NUMINAMATH_CALUDE_remainder_thirteen_six_twelve_seven_eleven_eight_mod_five_l3219_321909

theorem remainder_thirteen_six_twelve_seven_eleven_eight_mod_five :
  (13^6 + 12^7 + 11^8) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_thirteen_six_twelve_seven_eleven_eight_mod_five_l3219_321909


namespace NUMINAMATH_CALUDE_sine_sum_gt_cosine_sum_in_acute_triangle_l3219_321906

/-- In any acute-angled triangle ABC, the sum of the sines of its angles is greater than the sum of the cosines of its angles. -/
theorem sine_sum_gt_cosine_sum_in_acute_triangle (A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_triangle : A + B + C = π) : 
  Real.sin A + Real.sin B + Real.sin C > Real.cos A + Real.cos B + Real.cos C := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_gt_cosine_sum_in_acute_triangle_l3219_321906


namespace NUMINAMATH_CALUDE_product_divisible_by_twelve_l3219_321925

theorem product_divisible_by_twelve (a b c d : ℤ) :
  ∃ k : ℤ, (b - a) * (c - a) * (d - a) * (d - c) * (d - b) * (c - b) = 12 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_twelve_l3219_321925


namespace NUMINAMATH_CALUDE_second_divisor_is_24_l3219_321904

theorem second_divisor_is_24 (m n : ℕ) (h1 : m % 288 = 47) (h2 : m % n = 23) (h3 : n > 23) : n = 24 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_is_24_l3219_321904


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l3219_321949

theorem complete_square_quadratic (x : ℝ) :
  ∃ (a b : ℝ), (x^2 + 10*x - 3 = 0) ↔ ((x + a)^2 = b) ∧ b = 28 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l3219_321949


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3219_321948

theorem quadratic_minimum (x : ℝ) : x^2 + 8*x + 3 ≥ -13 ∧ 
  (x^2 + 8*x + 3 = -13 ↔ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3219_321948


namespace NUMINAMATH_CALUDE_inverse_variation_solution_l3219_321928

/-- The constant k in the inverse variation relationship -/
def k (a b : ℝ) : ℝ := a^3 * b^2

/-- The inverse variation relationship between a and b -/
def inverse_variation (a b : ℝ) : Prop := k a b = k 5 2

theorem inverse_variation_solution :
  ∀ a b : ℝ,
  inverse_variation a b →
  (a = 5 ∧ b = 2) ∨ (a = 2.5 ∧ b = 8) :=
sorry

end NUMINAMATH_CALUDE_inverse_variation_solution_l3219_321928


namespace NUMINAMATH_CALUDE_books_before_adding_l3219_321902

theorem books_before_adding (total_after : ℕ) (added : ℕ) (h1 : total_after = 19) (h2 : added = 10) :
  total_after - added = 9 := by
  sorry

end NUMINAMATH_CALUDE_books_before_adding_l3219_321902


namespace NUMINAMATH_CALUDE_value_of_expression_l3219_321955

theorem value_of_expression : (-0.125)^2009 * (-8)^2010 = -8 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3219_321955


namespace NUMINAMATH_CALUDE_initial_daily_steps_is_1000_l3219_321939

/-- Calculates the total steps logged over 4 weeks given the initial daily step count -/
def totalSteps (initialDailySteps : ℕ) : ℕ :=
  7 * initialDailySteps +
  7 * (initialDailySteps + 1000) +
  7 * (initialDailySteps + 2000) +
  7 * (initialDailySteps + 3000)

/-- Proves that the initial daily step count is 1000 given the problem conditions -/
theorem initial_daily_steps_is_1000 :
  ∃ (initialDailySteps : ℕ),
    totalSteps initialDailySteps = 100000 - 30000 ∧
    initialDailySteps = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_daily_steps_is_1000_l3219_321939


namespace NUMINAMATH_CALUDE_ice_cream_scoops_l3219_321988

/-- The number of scoops in a single cone of ice cream -/
def single_cone_scoops : ℕ := sorry

/-- The number of scoops in a banana split -/
def banana_split_scoops : ℕ := 3 * single_cone_scoops

/-- The number of scoops in a waffle bowl -/
def waffle_bowl_scoops : ℕ := banana_split_scoops + 1

/-- The number of scoops in a double cone -/
def double_cone_scoops : ℕ := 2 * single_cone_scoops

/-- The total number of scoops served -/
def total_scoops : ℕ := 10

theorem ice_cream_scoops : 
  single_cone_scoops + banana_split_scoops + waffle_bowl_scoops + double_cone_scoops = total_scoops ∧
  single_cone_scoops = 1 := by sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_l3219_321988


namespace NUMINAMATH_CALUDE_river_width_river_width_example_l3219_321913

/-- Calculates the width of a river given its depth, flow rate, and discharge volume. -/
theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (discharge_volume : ℝ) : ℝ :=
  let flow_rate_mpm := flow_rate_kmph * 1000 / 60
  let width := discharge_volume / (flow_rate_mpm * depth)
  width

/-- The width of a river with given parameters is 45 meters. -/
theorem river_width_example : river_width 2 6 9000 = 45 := by
  sorry

end NUMINAMATH_CALUDE_river_width_river_width_example_l3219_321913


namespace NUMINAMATH_CALUDE_seating_probability_l3219_321930

/-- The number of people seated at the round table -/
def total_people : ℕ := 12

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 4

/-- The number of biology majors -/
def biology_majors : ℕ := 3

/-- The probability of the desired seating arrangement -/
def desired_probability : ℚ := 18/175

theorem seating_probability :
  let total_arrangements := (total_people - 1).factorial
  let math_block_arrangements := total_people * (math_majors - 1).factorial
  let physics_arrangements := physics_majors.factorial
  let biology_arrangements := (physics_majors + 1).choose biology_majors * biology_majors.factorial
  let favorable_arrangements := math_block_arrangements * physics_arrangements * biology_arrangements
  (favorable_arrangements : ℚ) / total_arrangements = desired_probability := by
  sorry

end NUMINAMATH_CALUDE_seating_probability_l3219_321930


namespace NUMINAMATH_CALUDE_equal_values_at_fixed_distance_l3219_321952

theorem equal_values_at_fixed_distance (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 1, ContinuousAt f x) →
  f 0 = 0 →
  f 1 = 0 →
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 0 1 ∧ |x₁ - x₂| = 0.1 ∧ f x₁ = f x₂ := by
  sorry


end NUMINAMATH_CALUDE_equal_values_at_fixed_distance_l3219_321952


namespace NUMINAMATH_CALUDE_percentage_married_employees_l3219_321938

/-- The percentage of married employees in a company given specific conditions -/
theorem percentage_married_employees :
  let percent_women : ℝ := 0.61
  let percent_married_women : ℝ := 0.7704918032786885
  let percent_single_men : ℝ := 2/3
  
  let percent_men : ℝ := 1 - percent_women
  let percent_married_men : ℝ := 1 - percent_single_men
  
  let married_women : ℝ := percent_women * percent_married_women
  let married_men : ℝ := percent_men * percent_married_men
  
  let total_married : ℝ := married_women + married_men
  
  total_married = 0.60020016000000005
  := by sorry

end NUMINAMATH_CALUDE_percentage_married_employees_l3219_321938


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3219_321969

theorem largest_constant_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (K : ℝ), K = Real.sqrt 3 ∧ 
  (∀ (K' : ℝ), (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0), 
    Real.sqrt (x * y / z) + Real.sqrt (y * z / x) + Real.sqrt (x * z / y) ≥ K' * Real.sqrt (x + y + z)) → 
  K' ≤ K) ∧
  Real.sqrt (a * b / c) + Real.sqrt (b * c / a) + Real.sqrt (a * c / b) ≥ K * Real.sqrt (a + b + c) :=
sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3219_321969


namespace NUMINAMATH_CALUDE_ellipse_properties_l3219_321959

def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * a = 4 * Real.sqrt 3) 
  (h4 : (2^2 / a^2) + ((Real.sqrt 2)^2 / b^2) = 1) 
  (h5 : ∃ (A B : ℝ × ℝ), A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧ 
    (A.1 + B.1) / 2 = -8/5 ∧ (A.2 + B.2) / 2 = 2/5) :
  (a^2 = 12 ∧ b^2 = 3) ∧ 
  (∀ (A B : ℝ × ℝ), A ∈ Ellipse a b → B ∈ Ellipse a b → 
    (A.1 + B.1) / 2 = -8/5 → (A.2 + B.2) / 2 = 2/5 → 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 22 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3219_321959


namespace NUMINAMATH_CALUDE_notebook_purchase_cost_l3219_321974

def pen_cost : ℝ := 1.50
def notebook_cost : ℝ := 3 * pen_cost
def number_of_notebooks : ℕ := 4

theorem notebook_purchase_cost : 
  number_of_notebooks * notebook_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_notebook_purchase_cost_l3219_321974


namespace NUMINAMATH_CALUDE_white_chips_percentage_l3219_321905

theorem white_chips_percentage
  (total : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (h1 : blue = 3)
  (h2 : blue = total / 10)
  (h3 : green = 12) :
  (total - blue - green) * 100 / total = 50 :=
sorry

end NUMINAMATH_CALUDE_white_chips_percentage_l3219_321905


namespace NUMINAMATH_CALUDE_dvd_sales_proof_l3219_321911

theorem dvd_sales_proof (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * cd →
  dvd + cd = 273 →
  dvd = 168 := by
sorry

end NUMINAMATH_CALUDE_dvd_sales_proof_l3219_321911


namespace NUMINAMATH_CALUDE_second_person_speed_l3219_321942

/-- Given two people walking in the same direction, this theorem proves
    the speed of the second person given the conditions of the problem. -/
theorem second_person_speed
  (time : ℝ)
  (distance : ℝ)
  (speed1 : ℝ)
  (h1 : time = 9.5)
  (h2 : distance = 9.5)
  (h3 : speed1 = 4.5)
  : ∃ (speed2 : ℝ), speed2 = 5.5 ∧ distance = (speed2 - speed1) * time :=
by
  sorry

#check second_person_speed

end NUMINAMATH_CALUDE_second_person_speed_l3219_321942


namespace NUMINAMATH_CALUDE_range_of_power_function_l3219_321965

/-- The range of g(x) = x^m for m > 0 on the interval (0, 1) is (0, 1) -/
theorem range_of_power_function (m : ℝ) (hm : m > 0) :
  Set.range (fun x : ℝ => x ^ m) ∩ Set.Ioo 0 1 = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_power_function_l3219_321965


namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_eq_side_length_l3219_321918

/-- A regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  /-- The side length of the base -/
  baseSideLength : ℝ
  /-- The height of the pyramid -/
  height : ℝ

/-- The radius of a sphere circumscribed around a regular hexagonal pyramid -/
def circumscribedSphereRadius (p : RegularHexagonalPyramid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed sphere of a regular hexagonal pyramid
    with base side length a and height a is equal to a -/
theorem circumscribed_sphere_radius_eq_side_length
    (p : RegularHexagonalPyramid)
    (h1 : p.baseSideLength = p.height)
    (h2 : p.baseSideLength > 0) :
    circumscribedSphereRadius p = p.baseSideLength :=
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_eq_side_length_l3219_321918


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3219_321985

theorem rationalize_denominator : 
  (7 : ℝ) / (Real.sqrt 175 - Real.sqrt 75) = 7 * (Real.sqrt 7 + Real.sqrt 3) / 20 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3219_321985


namespace NUMINAMATH_CALUDE_magic_square_sum_div_by_3_l3219_321977

/-- Definition of a 3x3 magic square -/
def is_magic_square (a : Fin 9 → ℕ) (S : ℕ) : Prop :=
  -- Row sums
  (a 0 + a 1 + a 2 = S) ∧
  (a 3 + a 4 + a 5 = S) ∧
  (a 6 + a 7 + a 8 = S) ∧
  -- Column sums
  (a 0 + a 3 + a 6 = S) ∧
  (a 1 + a 4 + a 7 = S) ∧
  (a 2 + a 5 + a 8 = S) ∧
  -- Diagonal sums
  (a 0 + a 4 + a 8 = S) ∧
  (a 2 + a 4 + a 6 = S)

/-- Theorem: The sum of a third-order magic square is divisible by 3 -/
theorem magic_square_sum_div_by_3 (a : Fin 9 → ℕ) (S : ℕ) 
  (h : is_magic_square a S) : 
  3 ∣ S :=
by sorry

end NUMINAMATH_CALUDE_magic_square_sum_div_by_3_l3219_321977


namespace NUMINAMATH_CALUDE_factors_of_prime_factorization_l3219_321945

def prime_factorization := 2^3 * 3^5 * 5^4 * 7^2 * 11^6

def number_of_factors (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (3 + 1) * (5 + 1) * (4 + 1) * (2 + 1) * (6 + 1)

theorem factors_of_prime_factorization :
  number_of_factors prime_factorization = 2520 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_prime_factorization_l3219_321945


namespace NUMINAMATH_CALUDE_decision_symbol_is_diamond_l3219_321931

-- Define the type for flowchart symbols
inductive FlowchartSymbol
  | Diamond
  | Rectangle
  | Oval
  | Parallelogram

-- Define the function that determines if a symbol represents a decision
def representsDecision (symbol : FlowchartSymbol) : Prop :=
  symbol = FlowchartSymbol.Diamond

-- Theorem: The symbol that represents a decision in a flowchart is a diamond-shaped box
theorem decision_symbol_is_diamond :
  ∃ (symbol : FlowchartSymbol), representsDecision symbol :=
sorry

end NUMINAMATH_CALUDE_decision_symbol_is_diamond_l3219_321931


namespace NUMINAMATH_CALUDE_congruent_count_l3219_321919

theorem congruent_count (n : ℕ) : 
  (Finset.filter (fun x => x % 7 = 3) (Finset.range 300)).card = 43 :=
by sorry

end NUMINAMATH_CALUDE_congruent_count_l3219_321919


namespace NUMINAMATH_CALUDE_pieces_left_l3219_321916

/-- The number of medieval art pieces Alicia originally had -/
def original_pieces : ℕ := 70

/-- The number of medieval art pieces Alicia donated -/
def donated_pieces : ℕ := 46

/-- Theorem: The number of medieval art pieces Alicia has left is 24 -/
theorem pieces_left : original_pieces - donated_pieces = 24 := by
  sorry

end NUMINAMATH_CALUDE_pieces_left_l3219_321916


namespace NUMINAMATH_CALUDE_min_distance_squared_l3219_321975

/-- Given real numbers a, b, c, d satisfying |b+a^2-4ln a|+|2c-d+2|=0,
    the minimum value of (a-c)^2+(b-d)^2 is 5. -/
theorem min_distance_squared (a b c d : ℝ) 
    (h : |b + a^2 - 4*Real.log a| + |2*c - d + 2| = 0) : 
  (∀ x y z w : ℝ, |w + x^2 - 4*Real.log x| + |2*y - z + 2| = 0 →
    (a - c)^2 + (b - d)^2 ≤ (x - y)^2 + (w - z)^2) ∧
  (∃ x y z w : ℝ, |w + x^2 - 4*Real.log x| + |2*y - z + 2| = 0 ∧
    (a - c)^2 + (b - d)^2 = (x - y)^2 + (w - z)^2) ∧
  (a - c)^2 + (b - d)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_squared_l3219_321975


namespace NUMINAMATH_CALUDE_square_arrangement_exists_l3219_321973

-- Define the structure of the square
structure Square where
  bottomLeft : ℕ
  topRight : ℕ
  bottomRight : ℕ
  topLeft : ℕ
  center : ℕ

-- Define the property of having a common divisor greater than 1
def hasCommonDivisorGreaterThanOne (m n : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ k ∣ m ∧ k ∣ n

-- Define the property of being relatively prime
def isRelativelyPrime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

-- Main theorem
theorem square_arrangement_exists : ∃ (a b c d : ℕ), ∃ (s : Square),
  s.bottomLeft = a * b ∧
  s.topRight = c * d ∧
  s.bottomRight = a * d ∧
  s.topLeft = b * c ∧
  s.center = a * b * c * d ∧
  (hasCommonDivisorGreaterThanOne s.bottomLeft s.center) ∧
  (hasCommonDivisorGreaterThanOne s.topRight s.center) ∧
  (hasCommonDivisorGreaterThanOne s.bottomRight s.center) ∧
  (hasCommonDivisorGreaterThanOne s.topLeft s.center) ∧
  (isRelativelyPrime s.bottomLeft s.topRight) ∧
  (isRelativelyPrime s.bottomRight s.topLeft) :=
sorry

end NUMINAMATH_CALUDE_square_arrangement_exists_l3219_321973


namespace NUMINAMATH_CALUDE_calculation_proof_equation_solution_l3219_321929

-- Part 1
theorem calculation_proof :
  (Real.sqrt (25 / 9) + (Real.log 5 / Real.log 10) ^ 0 + (27 / 64) ^ (-(1/3 : ℝ))) = 4 := by
  sorry

-- Part 2
theorem equation_solution :
  ∀ x : ℝ, (Real.log (6^x - 9) / Real.log 3) = 3 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_solution_l3219_321929


namespace NUMINAMATH_CALUDE_functional_equation_problem_l3219_321981

/-- The functional equation problem -/
theorem functional_equation_problem (α : ℝ) (hα : α ≠ 0) :
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, f (f (x + y)) = f (x + y) + f x * f y + α * x * y) ↔
  (α = -1 ∧ ∃! f : ℝ → ℝ, ∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l3219_321981


namespace NUMINAMATH_CALUDE_square_greater_than_l3219_321962

theorem square_greater_than (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_l3219_321962


namespace NUMINAMATH_CALUDE_largest_increase_2006_2007_l3219_321999

def students : Fin 6 → ℕ
  | 0 => 50  -- 2003
  | 1 => 55  -- 2004
  | 2 => 60  -- 2005
  | 3 => 65  -- 2006
  | 4 => 75  -- 2007
  | 5 => 80  -- 2008

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreasePair : Fin 5 := sorry

theorem largest_increase_2006_2007 :
  largestIncreasePair = 3 ∧
  ∀ i : Fin 5, percentageIncrease (students i) (students (i + 1)) ≤
    percentageIncrease (students 3) (students 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_2006_2007_l3219_321999


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3219_321907

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3219_321907


namespace NUMINAMATH_CALUDE_symmetric_expressions_l3219_321920

-- Define what it means for an expression to be completely symmetric
def is_completely_symmetric (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), f a b c = f b a c ∧ f a b c = f a c b ∧ f a b c = f c b a

-- Define the three expressions
def expr1 (a b c : ℝ) : ℝ := (a - b)^2
def expr2 (a b c : ℝ) : ℝ := a * b + b * c + c * a
def expr3 (a b c : ℝ) : ℝ := a^2 * b + b^2 * c + c^2 * a

-- State the theorem
theorem symmetric_expressions :
  is_completely_symmetric expr1 ∧
  is_completely_symmetric expr2 ∧
  ¬ is_completely_symmetric expr3 := by sorry

end NUMINAMATH_CALUDE_symmetric_expressions_l3219_321920


namespace NUMINAMATH_CALUDE_bucket_capacity_proof_l3219_321997

theorem bucket_capacity_proof (capacity : ℝ) : 
  (12 * capacity = 108 * 9) → capacity = 81 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_proof_l3219_321997


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l3219_321950

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (x - 1 : ℚ) / 2 < x / 3 ∧ (2 * x - 5 : ℤ) ≤ 3 * (x - 2)}
  S = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l3219_321950


namespace NUMINAMATH_CALUDE_acid_solution_replacement_l3219_321921

theorem acid_solution_replacement (x : ℝ) :
  x ≥ 0 ∧ x ≤ 1 →
  0.5 * (1 - x) + 0.3 * x = 0.4 →
  x = 1/2 := by sorry

end NUMINAMATH_CALUDE_acid_solution_replacement_l3219_321921


namespace NUMINAMATH_CALUDE_data_median_and_variance_l3219_321972

def data : List ℝ := [2, 3, 3, 3, 6, 6, 4, 5]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_median_and_variance :
  median data = 3.5 ∧ variance data = 2 := by sorry

end NUMINAMATH_CALUDE_data_median_and_variance_l3219_321972


namespace NUMINAMATH_CALUDE_traffic_light_is_random_l3219_321986

-- Define the concept of a random event
def is_random_event (event : String) : Prop := sorry

-- Define the phenomena
def water_boiling : String := "Under standard atmospheric pressure, water will boil when heated to 100°C"
def traffic_light : String := "Encountering a red light when walking to a crossroads"
def rectangle_area : String := "The area of a rectangle with length and width a and b respectively is a × b"
def linear_equation : String := "A linear equation with real coefficients must have a real root"

-- Theorem to prove
theorem traffic_light_is_random : is_random_event traffic_light :=
by sorry

end NUMINAMATH_CALUDE_traffic_light_is_random_l3219_321986


namespace NUMINAMATH_CALUDE_inverse_sum_equals_six_l3219_321979

-- Define the function f
def f (x : ℝ) : ℝ := x * |x|^2

-- State the theorem
theorem inverse_sum_equals_six :
  ∃ (a b : ℝ), f a = 8 ∧ f b = -64 ∧ a + b = 6 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_six_l3219_321979


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l3219_321953

theorem sqrt_sum_equals_seven (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) : 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l3219_321953


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l3219_321932

theorem solution_set_abs_inequality :
  Set.Icc (1 : ℝ) 2 = {x : ℝ | |2*x - 3| ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l3219_321932


namespace NUMINAMATH_CALUDE_motorboat_travel_time_l3219_321940

/-- The time taken for a motorboat to travel from pier X to pier Y downstream,
    given the conditions of the river journey problem. -/
theorem motorboat_travel_time (s r : ℝ) (h₁ : s > 0) (h₂ : r > 0) (h₃ : s > r) : 
  ∃ t : ℝ, t = (12 * (s - r)) / (s + r) ∧ 
    (s + r) * t + (s - r) * (12 - t) = 12 * r := by
  sorry

end NUMINAMATH_CALUDE_motorboat_travel_time_l3219_321940


namespace NUMINAMATH_CALUDE_chain_store_sales_theorem_l3219_321992

-- Define the basic parameters
def cost_price : ℝ := 60
def initial_selling_price : ℝ := 80
def new_selling_price : ℝ := 100

-- Define the sales functions
def y₁ (x : ℝ) : ℝ := x^2 - 8*x + 56
def y₂ (x : ℝ) : ℝ := 2*x + 8

-- Define the gross profit function for sales > 60
def W (x : ℝ) : ℝ := 8*x^2 - 96*x - 512

-- Theorem statement
theorem chain_store_sales_theorem :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 10 → y₁ x ≥ 0) ∧
  (y₁ 4 = 40) ∧
  (y₁ 6 = 44) ∧
  (∀ x : ℝ, 11 ≤ x ∧ x ≤ 31 → y₂ x ≥ 0) ∧
  ((initial_selling_price - cost_price) * (y₁ 8) = 1120) ∧
  (∀ x : ℝ, 26 < x ∧ x ≤ 31 → W x = (new_selling_price - (cost_price - 2*(y₂ x - 60))) * y₂ x) := by
  sorry


end NUMINAMATH_CALUDE_chain_store_sales_theorem_l3219_321992


namespace NUMINAMATH_CALUDE_second_to_last_digit_of_power_of_three_is_even_l3219_321927

theorem second_to_last_digit_of_power_of_three_is_even (n : ℕ) :
  ∃ (k : ℕ), 3^n ≡ 20 * k + 2 * (3^n / 10 % 10) [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_second_to_last_digit_of_power_of_three_is_even_l3219_321927


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l3219_321991

theorem pencil_pen_cost (pencil_cost pen_cost : ℝ) 
  (h1 : 5 * pencil_cost + pen_cost = 2.50)
  (h2 : pencil_cost + 2 * pen_cost = 1.85) :
  2 * pencil_cost + pen_cost = 1.45 := by
sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l3219_321991


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3219_321937

/-- Given a sum of money P put at simple interest for 7 years at rate R%,
    if increasing the rate by 2% results in 140 more interest, then P = 1000. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * (R + 2) * 7) / 100 = (P * R * 7) / 100 + 140 → P = 1000 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3219_321937


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l3219_321910

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (population : ℕ) (sampleSize : ℕ) : ℕ :=
  population / sampleSize

/-- Theorem: The systematic sampling interval for 1000 students with a sample size of 50 is 20 -/
theorem systematic_sampling_interval_example :
  systematicSamplingInterval 1000 50 = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l3219_321910


namespace NUMINAMATH_CALUDE_stating_cube_coloring_theorem_l3219_321978

/-- Represents the number of available colors -/
def num_colors : ℕ := 5

/-- Represents the number of faces in a cube -/
def num_faces : ℕ := 6

/-- Represents the number of faces already painted -/
def painted_faces : ℕ := 3

/-- Represents the number of remaining faces to be painted -/
def remaining_faces : ℕ := num_faces - painted_faces

/-- 
  Represents the number of valid coloring schemes for the remaining faces of a cube,
  given that three adjacent faces are already painted with different colors,
  and no two adjacent faces can have the same color.
-/
def valid_coloring_schemes : ℕ := 13

/-- 
  Theorem stating that the number of valid coloring schemes for the remaining faces
  of a cube is equal to 13, given the specified conditions.
-/
theorem cube_coloring_theorem :
  valid_coloring_schemes = 13 :=
sorry

end NUMINAMATH_CALUDE_stating_cube_coloring_theorem_l3219_321978


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_with_quarter_circles_l3219_321943

/-- The area of the shaded region in a rectangle with quarter circles at corners -/
theorem shaded_area_rectangle_with_quarter_circles 
  (length : ℝ) (width : ℝ) (radius : ℝ) 
  (h_length : length = 12) 
  (h_width : width = 8) 
  (h_radius : radius = 4) : 
  length * width - π * radius^2 = 96 - 16 * π := by
sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_with_quarter_circles_l3219_321943


namespace NUMINAMATH_CALUDE_sweets_expenditure_correct_l3219_321912

/-- Calculates the amount spent on sweets given the initial amount and the amount given to each friend -/
def amount_spent_on_sweets (initial_amount : ℚ) (amount_per_friend : ℚ) : ℚ :=
  initial_amount - 2 * amount_per_friend

/-- Proves that the amount spent on sweets is correct for the given problem -/
theorem sweets_expenditure_correct (initial_amount : ℚ) (amount_per_friend : ℚ) 
  (h1 : initial_amount = 10.50)
  (h2 : amount_per_friend = 3.40) : 
  amount_spent_on_sweets initial_amount amount_per_friend = 3.70 := by
  sorry

#eval amount_spent_on_sweets 10.50 3.40

end NUMINAMATH_CALUDE_sweets_expenditure_correct_l3219_321912


namespace NUMINAMATH_CALUDE_beacon_school_earnings_l3219_321980

/-- Represents a school's participation in the community project -/
structure School where
  name : String
  students : ℕ
  weekdays : ℕ
  weekendDays : ℕ

/-- Calculates the total earnings for a school given the daily rates -/
def schoolEarnings (s : School) (weekdayRate weekendRate : ℚ) : ℚ :=
  s.students * (s.weekdays * weekdayRate + s.weekendDays * weekendRate)

/-- The main theorem stating that Beacon school's earnings are $336.00 -/
theorem beacon_school_earnings :
  let apex : School := ⟨"Apex", 9, 4, 2⟩
  let beacon : School := ⟨"Beacon", 6, 6, 1⟩
  let citadel : School := ⟨"Citadel", 7, 8, 3⟩
  let schools : List School := [apex, beacon, citadel]
  let totalPaid : ℚ := 1470
  ∃ (weekdayRate : ℚ),
    weekdayRate > 0 ∧
    (schools.map (fun s => schoolEarnings s weekdayRate (2 * weekdayRate))).sum = totalPaid ∧
    schoolEarnings beacon weekdayRate (2 * weekdayRate) = 336 := by
  sorry

end NUMINAMATH_CALUDE_beacon_school_earnings_l3219_321980


namespace NUMINAMATH_CALUDE_total_toys_proof_l3219_321924

/-- The number of toys Kamari has -/
def kamari_toys : ℝ := 65

/-- The number of toys Anais has -/
def anais_toys : ℝ := kamari_toys + 30.5

/-- The number of toys Lucien has -/
def lucien_toys : ℝ := 2 * kamari_toys

/-- The total number of toys Anais and Kamari have together -/
def anais_kamari_total : ℝ := 160.5

theorem total_toys_proof :
  kamari_toys + anais_toys + lucien_toys = 290.5 ∧
  anais_toys = kamari_toys + 30.5 ∧
  lucien_toys = 2 * kamari_toys ∧
  anais_toys + kamari_toys = anais_kamari_total :=
by sorry

end NUMINAMATH_CALUDE_total_toys_proof_l3219_321924


namespace NUMINAMATH_CALUDE_smallest_b_value_l3219_321941

theorem smallest_b_value (a b : ℕ) : 
  (a ≥ 1000) → (a ≤ 9999) → (b ≥ 100000) → (b ≤ 999999) → 
  (1 : ℚ) / 2006 = 1 / a + 1 / b → 
  ∀ b' ≥ 100000, b' ≤ 999999 → 
    ∃ a' ≥ 1000, a' ≤ 9999 → (1 : ℚ) / 2006 = 1 / a' + 1 / b' → 
      b ≤ b' → b = 120360 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3219_321941


namespace NUMINAMATH_CALUDE_marked_elements_not_distinct_l3219_321998

theorem marked_elements_not_distinct (marked : Fin 10 → Fin 10) : 
  (∀ i j, i ≠ j → marked i ≠ marked j) → False :=
by
  intro h
  -- The proof goes here
  sorry

#check marked_elements_not_distinct

end NUMINAMATH_CALUDE_marked_elements_not_distinct_l3219_321998


namespace NUMINAMATH_CALUDE_lunch_calories_calculation_l3219_321944

def daily_calorie_allowance : ℕ := 2200
def breakfast_calories : ℕ := 353
def snack_calories : ℕ := 130
def dinner_calories_left : ℕ := 832

theorem lunch_calories_calculation : 
  daily_calorie_allowance - breakfast_calories - snack_calories - dinner_calories_left = 885 := by
  sorry

end NUMINAMATH_CALUDE_lunch_calories_calculation_l3219_321944


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_factorial_l3219_321982

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_product (n : Nat) : Nat :=
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

def is_five_digit (n : Nat) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_with_product_factorial :
  ∃ (n : Nat), is_five_digit n ∧
               digit_product n = factorial 8 ∧
               ∀ (m : Nat), is_five_digit m ∧ digit_product m = factorial 8 → m ≤ n :=
by
  use 98752
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_factorial_l3219_321982


namespace NUMINAMATH_CALUDE_exponential_solution_l3219_321923

/-- A function satisfying f(x+1) = 2f(x) for all real x -/
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = 2 * f x

/-- The theorem stating that if f satisfies the functional equation,
    then f(x) = C * 2^x for some constant C -/
theorem exponential_solution (f : ℝ → ℝ) (h : functional_equation f) :
  ∃ C, ∀ x, f x = C * 2^x := by
  sorry

end NUMINAMATH_CALUDE_exponential_solution_l3219_321923


namespace NUMINAMATH_CALUDE_reduce_to_single_digit_l3219_321926

/-- Represents the operation of splitting digits and summing -/
def digit_split_sum (n : ℕ) : ℕ → ℕ → ℕ := sorry

/-- Predicate for a number being single-digit -/
def is_single_digit (n : ℕ) : Prop := n < 10

/-- Theorem stating that any natural number can be reduced to a single digit in at most 15 steps -/
theorem reduce_to_single_digit (N : ℕ) :
  ∃ (sequence : Fin 16 → ℕ),
    sequence 0 = N ∧
    (∀ i : Fin 15, ∃ a b : ℕ, sequence (i.succ) = digit_split_sum (sequence i) a b) ∧
    is_single_digit (sequence 15) :=
  sorry

end NUMINAMATH_CALUDE_reduce_to_single_digit_l3219_321926


namespace NUMINAMATH_CALUDE_he_has_21_apples_l3219_321954

/-- The number of apples Adam and Jackie have together -/
def total_adam_jackie : ℕ := 12

/-- The number of additional apples He has compared to Adam and Jackie together -/
def additional_apples : ℕ := 9

/-- The number of additional apples Adam has compared to Jackie -/
def adam_more_than_jackie : ℕ := 8

/-- The number of apples He has -/
def he_apples : ℕ := total_adam_jackie + additional_apples

theorem he_has_21_apples : he_apples = 21 := by
  sorry

end NUMINAMATH_CALUDE_he_has_21_apples_l3219_321954


namespace NUMINAMATH_CALUDE_complex_expression_equality_l3219_321917

theorem complex_expression_equality : -(-1 - (-2*(-3-4) - 5 - 6*(-7-80))) - 9 = 523 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l3219_321917


namespace NUMINAMATH_CALUDE_line_in_three_quadrants_coeff_products_l3219_321934

/-- A line passing through the first, second, and third quadrants -/
structure LineInThreeQuadrants where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_first_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + b * y + c = 0
  passes_second_quadrant : ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ a * x + b * y + c = 0
  passes_third_quadrant : ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0

/-- Theorem: If a line passes through the first, second, and third quadrants, 
    then the product of its coefficients satisfies ab < 0 and bc < 0 -/
theorem line_in_three_quadrants_coeff_products (line : LineInThreeQuadrants) :
  line.a * line.b < 0 ∧ line.b * line.c < 0 := by
  sorry

end NUMINAMATH_CALUDE_line_in_three_quadrants_coeff_products_l3219_321934


namespace NUMINAMATH_CALUDE_four_digit_divisor_characterization_l3219_321966

/-- Represents a four-digit number in decimal notation -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- Converts a FourDigitNumber to its decimal value -/
def to_decimal (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Checks if one FourDigitNumber divides another -/
def divides (m n : FourDigitNumber) : Prop :=
  ∃ k : Nat, k * (to_decimal m) = to_decimal n

/-- Main theorem: Characterization of four-digit numbers that divide their rotations -/
theorem four_digit_divisor_characterization (n : FourDigitNumber) :
  (divides n {a := n.b, b := n.c, c := n.d, d := n.a, 
              a_nonzero := sorry, b_digit := n.c_digit, c_digit := n.d_digit, d_digit := sorry}) ∨
  (divides n {a := n.c, b := n.d, c := n.a, d := n.b, 
              a_nonzero := sorry, b_digit := n.d_digit, c_digit := sorry, d_digit := n.b_digit}) ∨
  (divides n {a := n.d, b := n.a, c := n.b, d := n.c, 
              a_nonzero := sorry, b_digit := sorry, c_digit := n.b_digit, d_digit := n.c_digit})
  ↔
  n.a = n.c ∧ n.b = n.d ∧ n.b ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_four_digit_divisor_characterization_l3219_321966


namespace NUMINAMATH_CALUDE_inequality_theorem_range_theorem_l3219_321935

-- Theorem 1
theorem inequality_theorem (x y : ℝ) : x^2 + 2*y^2 ≥ 2*x*y + 2*y - 1 := by
  sorry

-- Theorem 2
theorem range_theorem (a b : ℝ) (h1 : -2 < a ∧ a ≤ 3) (h2 : 1 ≤ b ∧ b < 2) :
  -1 < a + b ∧ a + b < 5 ∧ -10 < 2*a - 3*b ∧ 2*a - 3*b ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_range_theorem_l3219_321935


namespace NUMINAMATH_CALUDE_inverse_function_inequality_solution_set_l3219_321946

/-- Given two functions f and g that intersect at two points, 
    prove the solution set of the inequality between their inverse functions. -/
theorem inverse_function_inequality_solution_set 
  (f g : ℝ → ℝ)
  (h_f : ∃ k b : ℝ, ∀ x, f x = k * x + b)
  (h_g : ∀ x, g x = 2^x + 1)
  (h_intersect : ∃ x₁ x₂ : ℝ, 
    f x₁ = g x₁ ∧ f x₁ = 2 ∧ 
    f x₂ = g x₂ ∧ f x₂ = 4 ∧
    x₁ < x₂)
  (f_inv g_inv : ℝ → ℝ)
  (h_f_inv : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x)
  (h_g_inv : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x)
  : {x : ℝ | f_inv x ≥ g_inv x} = Set.Ici 4 ∪ Set.Ioc 1 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_inequality_solution_set_l3219_321946


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l3219_321961

theorem degree_to_radian_conversion (π : ℝ) :
  (1 : ℝ) * π / 180 = π / 180 →
  (-150 : ℝ) * π / 180 = -5 * π / 6 :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l3219_321961


namespace NUMINAMATH_CALUDE_van_helsing_werewolf_removal_percentage_l3219_321936

def vampire_price : ℕ := 5
def werewolf_price : ℕ := 10
def total_earnings : ℕ := 105
def werewolves_removed : ℕ := 8
def werewolf_vampire_ratio : ℕ := 4

theorem van_helsing_werewolf_removal_percentage 
  (vampires : ℕ) (werewolves : ℕ) : 
  vampire_price * (vampires / 2) + werewolf_price * werewolves_removed = total_earnings →
  werewolves = werewolf_vampire_ratio * vampires →
  (werewolves_removed : ℚ) / werewolves * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_van_helsing_werewolf_removal_percentage_l3219_321936


namespace NUMINAMATH_CALUDE_zero_product_implies_zero_factor_l3219_321971

theorem zero_product_implies_zero_factor (x y : ℝ) : 
  x * y = 0 → x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_product_implies_zero_factor_l3219_321971


namespace NUMINAMATH_CALUDE_probability_product_multiple_of_four_l3219_321987

def dodecahedral_die : Finset ℕ := Finset.range 12
def eight_sided_die : Finset ℕ := Finset.range 8

def is_multiple_of_four (n : ℕ) : Bool := n % 4 = 0

theorem probability_product_multiple_of_four :
  let outcomes := dodecahedral_die.product eight_sided_die
  let favorable_outcomes := outcomes.filter (fun (x, y) => is_multiple_of_four (x * y))
  (favorable_outcomes.card : ℚ) / outcomes.card = 7 / 16 := by sorry

end NUMINAMATH_CALUDE_probability_product_multiple_of_four_l3219_321987


namespace NUMINAMATH_CALUDE_age_difference_ratio_l3219_321968

/-- Represents the ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.roy = ages.julia + 8 ∧
  ages.roy + 4 = 2 * (ages.julia + 4) ∧
  (ages.roy + 4) * (ages.kelly + 4) = 192

/-- The theorem to prove -/
theorem age_difference_ratio (ages : Ages) :
  satisfiesConditions ages →
  (ages.roy - ages.julia) / (ages.roy - ages.kelly) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_ratio_l3219_321968


namespace NUMINAMATH_CALUDE_n_equals_six_l3219_321976

/-- The number of coins flipped simultaneously -/
def n : ℕ := sorry

/-- The probability of exactly two tails when flipping n coins -/
def prob_two_tails (n : ℕ) : ℚ := n * (n - 1) / (2^(n + 1))

/-- Theorem stating that n equals 6 when the probability of two tails is 5/32 -/
theorem n_equals_six : 
  (prob_two_tails n = 5/32) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_n_equals_six_l3219_321976


namespace NUMINAMATH_CALUDE_broomstick_charge_theorem_l3219_321964

/-- Represents the state of the broomstick at a given time -/
structure BroomState where
  minutes : Nat  -- Minutes since midnight
  charge : Nat   -- Current charge (0-100)

/-- Calculates the charge of the broomstick given the number of minutes since midnight -/
def calculateCharge (minutes : Nat) : Nat :=
  100 - minutes / 6

/-- Checks if the given time (in minutes since midnight) is a solution -/
def isSolution (minutes : Nat) : Bool :=
  let charge := calculateCharge minutes
  let minutesPastHour := minutes % 60
  charge == minutesPastHour

/-- The set of solution times -/
def solutionTimes : List BroomState :=
  [
    { minutes := 292, charge := 52 },  -- 04:52
    { minutes := 343, charge := 43 },  -- 05:43
    { minutes := 395, charge := 35 },  -- 06:35
    { minutes := 446, charge := 26 },  -- 07:26
    { minutes := 549, charge := 9 }    -- 09:09
  ]

/-- Main theorem: The given solution times are correct and complete -/
theorem broomstick_charge_theorem :
  (∀ t ∈ solutionTimes, isSolution t.minutes) ∧
  (∀ m, 0 ≤ m ∧ m < 600 → isSolution m → (∃ t ∈ solutionTimes, t.minutes = m)) :=
sorry


end NUMINAMATH_CALUDE_broomstick_charge_theorem_l3219_321964


namespace NUMINAMATH_CALUDE_guest_payment_divisibility_l3219_321947

theorem guest_payment_divisibility (A : Nat) (h1 : A < 10) : 
  (100 + 10 * A + 2) % 11 = 0 ↔ A = 3 := by
  sorry

end NUMINAMATH_CALUDE_guest_payment_divisibility_l3219_321947


namespace NUMINAMATH_CALUDE_spade_nested_operation_l3219_321983

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_nested_operation : spade 5 (spade 3 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spade_nested_operation_l3219_321983


namespace NUMINAMATH_CALUDE_total_savings_after_tax_l3219_321984

def total_income : ℝ := 18000

def income_ratio_a : ℝ := 3
def income_ratio_b : ℝ := 2
def income_ratio_c : ℝ := 1

def tax_rate_a : ℝ := 0.1
def tax_rate_b : ℝ := 0.15
def tax_rate_c : ℝ := 0

def expenditure_ratio : ℝ := 5
def income_ratio : ℝ := 9

theorem total_savings_after_tax :
  let income_a := (income_ratio_a / (income_ratio_a + income_ratio_b + income_ratio_c)) * total_income
  let income_b := (income_ratio_b / (income_ratio_a + income_ratio_b + income_ratio_c)) * total_income
  let income_c := (income_ratio_c / (income_ratio_a + income_ratio_b + income_ratio_c)) * total_income
  let tax_a := tax_rate_a * income_a
  let tax_b := tax_rate_b * income_b
  let tax_c := tax_rate_c * income_c
  let total_tax := tax_a + tax_b + tax_c
  let income_after_tax := total_income - total_tax
  let expenditure := (expenditure_ratio / income_ratio) * total_income
  let savings := income_after_tax - expenditure
  savings = 6200 := by sorry

end NUMINAMATH_CALUDE_total_savings_after_tax_l3219_321984


namespace NUMINAMATH_CALUDE_necklace_diamond_count_l3219_321957

theorem necklace_diamond_count (total_necklaces : ℕ) (total_diamonds : ℕ) : 
  total_necklaces = 20 →
  total_diamonds = 79 →
  ∃ (two_diamond_necklaces five_diamond_necklaces : ℕ),
    two_diamond_necklaces + five_diamond_necklaces = total_necklaces ∧
    2 * two_diamond_necklaces + 5 * five_diamond_necklaces = total_diamonds ∧
    five_diamond_necklaces = 13 := by
  sorry

end NUMINAMATH_CALUDE_necklace_diamond_count_l3219_321957


namespace NUMINAMATH_CALUDE_multiplication_mistake_l3219_321901

theorem multiplication_mistake (x : ℤ) : 
  (43 * x - 34 * x = 1206) → x = 134 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l3219_321901


namespace NUMINAMATH_CALUDE_shark_teeth_multiple_l3219_321900

theorem shark_teeth_multiple : 
  let tiger_teeth : ℕ := 180
  let hammerhead_teeth : ℕ := tiger_teeth / 6
  let sum_teeth : ℕ := tiger_teeth + hammerhead_teeth
  let great_white_teeth : ℕ := 420
  great_white_teeth / sum_teeth = 2 := by
  sorry

end NUMINAMATH_CALUDE_shark_teeth_multiple_l3219_321900


namespace NUMINAMATH_CALUDE_total_books_count_l3219_321915

/-- The total number of Iesha's books -/
def total_books : ℕ := sorry

/-- The number of Iesha's school books -/
def school_books : ℕ := 19

/-- The number of Iesha's sports books -/
def sports_books : ℕ := 39

/-- Theorem: The total number of Iesha's books is 58 -/
theorem total_books_count : total_books = school_books + sports_books ∧ total_books = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l3219_321915


namespace NUMINAMATH_CALUDE_arccos_negative_half_l3219_321958

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_half_l3219_321958


namespace NUMINAMATH_CALUDE_passing_grade_fraction_l3219_321960

theorem passing_grade_fraction (a b c d f : ℚ) : 
  a = 1/4 → b = 1/2 → c = 1/8 → d = 1/12 → f = 1/24 → a + b + c = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_passing_grade_fraction_l3219_321960


namespace NUMINAMATH_CALUDE_toy_store_solution_l3219_321963

/-- Represents the selling and cost prices of toys, and the optimal purchase strategy -/
structure ToyStore where
  sell_price_A : ℝ
  sell_price_B : ℝ
  cost_price_A : ℝ
  cost_price_B : ℝ
  optimal_purchase_A : ℕ
  optimal_purchase_B : ℕ

/-- The toy store problem with given conditions -/
def toy_store_problem : Prop :=
  ∃ (store : ToyStore),
    -- Selling price condition
    store.sell_price_B - store.sell_price_A = 30 ∧
    -- Total sales condition
    2 * store.sell_price_A + 3 * store.sell_price_B = 740 ∧
    -- Cost prices
    store.cost_price_A = 90 ∧
    store.cost_price_B = 110 ∧
    -- Total purchase constraint
    store.optimal_purchase_A + store.optimal_purchase_B = 80 ∧
    -- Total cost constraint
    store.cost_price_A * store.optimal_purchase_A + store.cost_price_B * store.optimal_purchase_B ≤ 8400 ∧
    -- Correct selling prices
    store.sell_price_A = 130 ∧
    store.sell_price_B = 160 ∧
    -- Optimal purchase strategy
    store.optimal_purchase_A = 20 ∧
    store.optimal_purchase_B = 60 ∧
    -- Profit maximization (implied by the optimal strategy)
    ∀ (m : ℕ), m + (80 - m) = 80 →
      (store.sell_price_A - store.cost_price_A) * store.optimal_purchase_A +
      (store.sell_price_B - store.cost_price_B) * store.optimal_purchase_B ≥
      (store.sell_price_A - store.cost_price_A) * m +
      (store.sell_price_B - store.cost_price_B) * (80 - m)

theorem toy_store_solution : toy_store_problem := by
  sorry


end NUMINAMATH_CALUDE_toy_store_solution_l3219_321963


namespace NUMINAMATH_CALUDE_food_duration_l3219_321970

theorem food_duration (initial_cows : ℕ) (days_passed : ℕ) (cows_left : ℕ) : 
  initial_cows = 1000 →
  days_passed = 10 →
  cows_left = 800 →
  (initial_cows * x - initial_cows * days_passed = cows_left * x) →
  x = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_food_duration_l3219_321970
