import Mathlib

namespace NUMINAMATH_CALUDE_profit_1200_optimal_price_reduction_l4109_410987

/-- Represents the shirt sales scenario --/
structure ShirtSales where
  baseSales : ℕ := 20
  baseProfit : ℕ := 40
  salesIncrease : ℕ := 2
  priceReduction : ℚ

/-- Calculates the daily profit for a given price reduction --/
def dailyProfit (s : ShirtSales) : ℚ :=
  (s.baseProfit - s.priceReduction) * (s.baseSales + s.salesIncrease * s.priceReduction)

/-- Theorem for the price reductions that result in a daily profit of 1200 yuan --/
theorem profit_1200 (s : ShirtSales) :
  dailyProfit s = 1200 ↔ s.priceReduction = 10 ∨ s.priceReduction = 20 := by sorry

/-- Theorem for the optimal price reduction and maximum profit --/
theorem optimal_price_reduction (s : ShirtSales) :
  (∀ x, dailyProfit { s with priceReduction := x } ≤ dailyProfit { s with priceReduction := 15 }) ∧
  dailyProfit { s with priceReduction := 15 } = 1250 := by sorry

end NUMINAMATH_CALUDE_profit_1200_optimal_price_reduction_l4109_410987


namespace NUMINAMATH_CALUDE_win_sector_area_l4109_410990

/-- The area of the WIN sector on a circular spinner with given radius and winning probability -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 10) (h_p : p = 3/7) :
  p * π * r^2 = 300 * π / 7 := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l4109_410990


namespace NUMINAMATH_CALUDE_circle_radius_sqrt34_l4109_410961

/-- Given a circle with center on the x-axis passing through points (0,5) and (2,3),
    prove that its radius is √34. -/
theorem circle_radius_sqrt34 :
  ∀ x : ℝ,
  (x^2 + 5^2 = (x-2)^2 + 3^2) →  -- condition that (x,0) is equidistant from (0,5) and (2,3)
  ∃ r : ℝ,
  r^2 = 34 ∧                    -- r is the radius
  r^2 = x^2 + 5^2               -- distance formula from center to (0,5)
  :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_sqrt34_l4109_410961


namespace NUMINAMATH_CALUDE_ninety_nine_squared_l4109_410916

theorem ninety_nine_squared : 99 * 99 = 9801 := by
  sorry

end NUMINAMATH_CALUDE_ninety_nine_squared_l4109_410916


namespace NUMINAMATH_CALUDE_fliers_remaining_l4109_410995

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ)
  (h1 : total = 1000)
  (h2 : morning_fraction = 1 / 5)
  (h3 : afternoon_fraction = 1 / 4) :
  total - (morning_fraction * total).num - (afternoon_fraction * (total - (morning_fraction * total).num)).num = 600 :=
by sorry

end NUMINAMATH_CALUDE_fliers_remaining_l4109_410995


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_46_l4109_410957

theorem no_primes_divisible_by_46 : ∀ p : ℕ, Nat.Prime p → ¬(46 ∣ p) := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_46_l4109_410957


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4109_410981

theorem sum_of_coefficients (k : ℝ) (h : k ≠ 0) : ∃ (a b c d : ℤ),
  (8 * k + 9 + 10 * k^2 - 3 * k^3) + (4 * k + 6 + k^2 + k^3) = 
  (a : ℝ) * k^3 + (b : ℝ) * k^2 + (c : ℝ) * k + (d : ℝ) ∧ 
  a + b + c + d = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4109_410981


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l4109_410926

theorem no_prime_roots_for_quadratic : 
  ¬∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    p ≠ q ∧
    (p : ℤ) + q = 72 ∧ 
    (p : ℤ) * q = k ∧
    ∀ (x : ℤ), x^2 - 72*x + k = 0 ↔ x = p ∨ x = q :=
by sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l4109_410926


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l4109_410912

/-- The x-intercept of the line 6x + 7y = 35 is (35/6, 0) -/
theorem x_intercept_of_line (x y : ℚ) : 
  (6 * x + 7 * y = 35) → (x = 35 / 6 ∧ y = 0) → (6 * (35 / 6) + 7 * 0 = 35) := by
  sorry

#check x_intercept_of_line

end NUMINAMATH_CALUDE_x_intercept_of_line_l4109_410912


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_l4109_410920

/-- The intersection points of a hyperbola and a circle -/
theorem hyperbola_circle_intersection :
  ∀ x y : ℝ, x^2 - 9*y^2 = 36 ∧ x^2 + y^2 = 36 → (x = 6 ∧ y = 0) ∨ (x = -6 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_l4109_410920


namespace NUMINAMATH_CALUDE_cos_ninety_degrees_l4109_410944

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_ninety_degrees_l4109_410944


namespace NUMINAMATH_CALUDE_henan_population_scientific_notation_l4109_410927

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem henan_population_scientific_notation :
  toScientificNotation (98.83 * 1000000) = ScientificNotation.mk 9.883 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_henan_population_scientific_notation_l4109_410927


namespace NUMINAMATH_CALUDE_order_of_equations_l4109_410918

def order_of_diff_eq (eq : String) : ℕ :=
  match eq with
  | "y' + 2x = 0" => 1
  | "y'' + 3y' - 4 = 0" => 2
  | "2dy - 3x dx = 0" => 1
  | "y'' = cos x" => 2
  | _ => 0

theorem order_of_equations :
  (order_of_diff_eq "y' + 2x = 0" = 1) ∧
  (order_of_diff_eq "y'' + 3y' - 4 = 0" = 2) ∧
  (order_of_diff_eq "2dy - 3x dx = 0" = 1) ∧
  (order_of_diff_eq "y'' = cos x" = 2) := by
  sorry

end NUMINAMATH_CALUDE_order_of_equations_l4109_410918


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4109_410997

theorem inequality_system_solution (x : ℝ) :
  (x / 3 + 2 > 0 ∧ 2 * x + 5 ≥ 3) ↔ x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4109_410997


namespace NUMINAMATH_CALUDE_slope_implies_y_coordinate_l4109_410904

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is -3/2, then the y-coordinate of Q is -2. -/
theorem slope_implies_y_coordinate (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = -2 →
  y₁ = 7 →
  x₂ = 4 →
  (y₂ - y₁) / (x₂ - x₁) = -3/2 →
  y₂ = -2 :=
by sorry

end NUMINAMATH_CALUDE_slope_implies_y_coordinate_l4109_410904


namespace NUMINAMATH_CALUDE_midpoint_is_inferior_exists_n_satisfying_conditions_l4109_410985

/-- Definition of a superior point in the first quadrant -/
def is_superior_point (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a / b > c / d

/-- Definition of an inferior point in the first quadrant -/
def is_inferior_point (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a / b < c / d

/-- Theorem: The midpoint of a superior point and an inferior point is inferior to the superior point -/
theorem midpoint_is_inferior (a b c d : ℝ) :
  is_superior_point a b c d →
  is_inferior_point ((a + c) / 2) ((b + d) / 2) a b :=
sorry

/-- Definition of the set of integers from 1 to 2021 -/
def S : Set ℤ := {m | 0 < m ∧ m < 2022}

/-- Theorem: There exists an integer n satisfying the given conditions -/
theorem exists_n_satisfying_conditions :
  ∃ n : ℤ, ∀ m ∈ S,
    (is_inferior_point n (2 * m + 1) 2022 m) ∧
    (is_superior_point n (2 * m + 1) 2023 (m + 1)) :=
sorry

end NUMINAMATH_CALUDE_midpoint_is_inferior_exists_n_satisfying_conditions_l4109_410985


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l4109_410911

theorem smallest_number_with_properties : 
  ∃ (n : ℕ), n = 153846 ∧ 
  (∀ m : ℕ, m < n → 
    (m % 10 = 6 ∧ 
     ∃ k : ℕ, 6 * 10^k + (m - 6) / 10 = 4 * m) → False) ∧
  n % 10 = 6 ∧
  ∃ k : ℕ, 6 * 10^k + (n - 6) / 10 = 4 * n :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_properties_l4109_410911


namespace NUMINAMATH_CALUDE_number_of_unique_lines_l4109_410900

/-- The set of possible coefficients for A and B -/
def S : Finset ℕ := {0, 1, 2, 3, 5}

/-- A line is represented by a pair of distinct coefficients (A, B) -/
def Line : Type := { p : ℕ × ℕ // p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2 }

/-- The set of all possible lines -/
def AllLines : Finset Line := sorry

theorem number_of_unique_lines : Finset.card AllLines = 14 := by
  sorry

end NUMINAMATH_CALUDE_number_of_unique_lines_l4109_410900


namespace NUMINAMATH_CALUDE_hidden_dots_four_dice_l4109_410902

/-- The sum of dots on a standard six-sided die -/
def standard_die_sum : ℕ := 21

/-- The total number of dots on four standard six-sided dice -/
def total_dots (n : ℕ) : ℕ := n * standard_die_sum

/-- The sum of visible dots on the stacked dice -/
def visible_dots : ℕ := 1 + 2 + 2 + 3 + 4 + 5 + 6 + 6

/-- The number of hidden dots on four stacked dice -/
def hidden_dots (n : ℕ) : ℕ := total_dots n - visible_dots

theorem hidden_dots_four_dice : 
  hidden_dots 4 = 55 := by sorry

end NUMINAMATH_CALUDE_hidden_dots_four_dice_l4109_410902


namespace NUMINAMATH_CALUDE_disk_count_l4109_410954

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- The ratio of blue:yellow:green disks is 3:7:8 -/
def ratio_condition (bag : DiskBag) : Prop :=
  ∃ (x : ℕ), bag.blue = 3 * x ∧ bag.yellow = 7 * x ∧ bag.green = 8 * x

/-- There are 20 more green disks than blue disks -/
def green_blue_difference (bag : DiskBag) : Prop :=
  bag.green = bag.blue + 20

/-- The total number of disks in the bag -/
def total_disks (bag : DiskBag) : ℕ :=
  bag.blue + bag.yellow + bag.green

/-- Theorem: The total number of disks in the bag is 72 -/
theorem disk_count (bag : DiskBag) 
  (h1 : ratio_condition bag) 
  (h2 : green_blue_difference bag) : 
  total_disks bag = 72 := by
  sorry


end NUMINAMATH_CALUDE_disk_count_l4109_410954


namespace NUMINAMATH_CALUDE_charity_race_dropouts_l4109_410996

/-- The number of people who dropped out of a bicycle charity race --/
def dropouts (initial_racers : ℕ) (joined_racers : ℕ) (finishers : ℕ) : ℕ :=
  (initial_racers + joined_racers) * 2 - finishers

theorem charity_race_dropouts : dropouts 50 30 130 = 30 := by
  sorry

end NUMINAMATH_CALUDE_charity_race_dropouts_l4109_410996


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l4109_410935

/-- The cost of one dozen pens given the ratio of pen to pencil cost and the total cost of 3 pens and 5 pencils -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℕ) : 
  pen_cost = 5 * pencil_cost →  -- Condition 1: pen cost is 5 times pencil cost
  3 * pen_cost + 5 * pencil_cost = 240 →  -- Condition 2: total cost of 3 pens and 5 pencils
  12 * pen_cost = 720 :=  -- Conclusion: cost of one dozen pens
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l4109_410935


namespace NUMINAMATH_CALUDE_polygon_with_45_degree_exterior_angles_has_8_sides_l4109_410947

/-- A polygon with exterior angles measuring 45° has 8 sides. -/
theorem polygon_with_45_degree_exterior_angles_has_8_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 45 →
    (n : ℝ) * exterior_angle = 360 →
    n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_45_degree_exterior_angles_has_8_sides_l4109_410947


namespace NUMINAMATH_CALUDE_binomial_max_probability_l4109_410909

/-- The number of trials in the binomial distribution -/
def n : ℕ := 10

/-- The probability of success in each trial -/
def p : ℝ := 0.8

/-- The probability mass function of the binomial distribution -/
def binomialPMF (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The value of k that maximizes the binomial PMF -/
def kMax : ℕ := 8

theorem binomial_max_probability :
  ∀ k : ℕ, k ≠ kMax → binomialPMF k ≤ binomialPMF kMax :=
sorry

end NUMINAMATH_CALUDE_binomial_max_probability_l4109_410909


namespace NUMINAMATH_CALUDE_candidate_total_score_l4109_410977

/-- Calculates the total score of a candidate based on their written test and interview scores -/
def totalScore (writtenScore : ℝ) (interviewScore : ℝ) : ℝ :=
  0.70 * writtenScore + 0.30 * interviewScore

/-- Theorem stating that the total score of a candidate with given scores is 87 -/
theorem candidate_total_score :
  let writtenScore : ℝ := 90
  let interviewScore : ℝ := 80
  totalScore writtenScore interviewScore = 87 := by
  sorry

#eval totalScore 90 80

end NUMINAMATH_CALUDE_candidate_total_score_l4109_410977


namespace NUMINAMATH_CALUDE_power_of_power_l4109_410968

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l4109_410968


namespace NUMINAMATH_CALUDE_fraction_order_l4109_410988

theorem fraction_order : (24 : ℚ) / 19 < 23 / 17 ∧ 23 / 17 < 11 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l4109_410988


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4109_410940

theorem complex_magnitude_problem (z : ℂ) (h : (1 - Complex.I * Real.sqrt 3) * z = Complex.I) :
  Complex.abs z = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4109_410940


namespace NUMINAMATH_CALUDE_right_triangle_inequality_right_triangle_inequality_equality_l4109_410917

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  3*a + 4*b ≤ 5*c :=
by sorry

theorem right_triangle_inequality_equality (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  3*a + 4*b = 5*c ↔ ∃ (k : ℝ), k > 0 ∧ a = 3*k ∧ b = 4*k ∧ c = 5*k :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_right_triangle_inequality_equality_l4109_410917


namespace NUMINAMATH_CALUDE_new_person_weight_l4109_410924

def group_size : ℕ := 8
def average_weight_increase : ℝ := 2.5
def replaced_person_weight : ℝ := 65

theorem new_person_weight (new_weight : ℝ) :
  (group_size : ℝ) * average_weight_increase = new_weight - replaced_person_weight →
  new_weight = 85 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l4109_410924


namespace NUMINAMATH_CALUDE_eulers_partition_theorem_l4109_410966

/-- The number of partitions of a natural number into distinct parts -/
def d (n : ℕ) : ℕ := sorry

/-- The number of partitions of a natural number into odd parts -/
def l (n : ℕ) : ℕ := sorry

/-- Euler's partition theorem: The number of partitions of a natural number
    into distinct parts is equal to the number of partitions into odd parts -/
theorem eulers_partition_theorem : ∀ n : ℕ, d n = l n := by sorry

end NUMINAMATH_CALUDE_eulers_partition_theorem_l4109_410966


namespace NUMINAMATH_CALUDE_max_value_sum_l4109_410929

theorem max_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + 2*a*b + 4*b^2 = 6) :
  a + 2*b ≤ 2 * Real.sqrt 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀^2 + 2*a₀*b₀ + 4*b₀^2 = 6 ∧ a₀ + 2*b₀ = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_l4109_410929


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l4109_410914

theorem quadratic_roots_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    |x₁ - x₂| = 2) →
  p = 2 * Real.sqrt (q + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l4109_410914


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l4109_410931

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter 
  (total : Rectangle)
  (small : Rectangle)
  (shaded_area : ℝ)
  (h1 : total.width = 12)
  (h2 : total.height = 10)
  (h3 : small.width = 4)
  (h4 : small.height = 3)
  (h5 : shaded_area = 120) :
  perimeter { width := total.width - (total.width - small.width),
              height := total.height - small.height } = 23 := by
sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l4109_410931


namespace NUMINAMATH_CALUDE_attendant_claimed_two_shirts_l4109_410936

-- Define the given conditions
def trousers : ℕ := 10
def total_bill : ℕ := 140
def shirt_cost : ℕ := 5
def trouser_cost : ℕ := 9
def missing_shirts : ℕ := 8

-- Define the function to calculate the number of shirts the attendant initially claimed
def attendant_claim : ℕ :=
  let trouser_total : ℕ := trousers * trouser_cost
  let shirt_total : ℕ := total_bill - trouser_total
  let actual_shirts : ℕ := shirt_total / shirt_cost
  actual_shirts - missing_shirts

-- Theorem statement
theorem attendant_claimed_two_shirts :
  attendant_claim = 2 := by sorry

end NUMINAMATH_CALUDE_attendant_claimed_two_shirts_l4109_410936


namespace NUMINAMATH_CALUDE_two_distinct_roots_iff_a_in_A_l4109_410994

/-- The equation has exactly two distinct roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁ - a)^2 - 1 = 2 * (x₁ + |x₁|) ∧
  (x₂ - a)^2 - 1 = 2 * (x₂ + |x₂|) ∧
  ∀ x : ℝ, (x - a)^2 - 1 = 2 * (x + |x|) → x = x₁ ∨ x = x₂

/-- The set of values for a -/
def A : Set ℝ := Set.Ioi 1 ∪ Set.Ioo (-1) 1 ∪ Set.Iic (-5/4)

theorem two_distinct_roots_iff_a_in_A :
  ∀ a : ℝ, has_two_distinct_roots a ↔ a ∈ A :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_roots_iff_a_in_A_l4109_410994


namespace NUMINAMATH_CALUDE_product_of_roots_l4109_410963

theorem product_of_roots (y₁ y₂ : ℝ) : 
  y₁ + 16 / y₁ = 12 → 
  y₂ + 16 / y₂ = 12 → 
  y₁ * y₂ = 16 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l4109_410963


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4109_410973

theorem quadratic_inequality_solution_set 
  (a b c x₁ x₂ : ℝ) 
  (h₁ : x₁ < x₂) 
  (h₂ : a < 0) 
  (h₃ : ∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :
  ∀ x, a * x^2 + b * x + c > 0 ↔ x₁ < x ∧ x < x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4109_410973


namespace NUMINAMATH_CALUDE_baker_total_cookies_l4109_410984

/-- Represents the number of cookies in a batch for each type of cookie --/
structure CookieBatch where
  chocolate_chip : Nat
  oatmeal : Nat
  sugar : Nat
  double_chocolate : Nat

/-- Represents the number of batches for each type of cookie --/
structure BatchCount where
  chocolate_chip : Nat
  oatmeal : Nat
  sugar : Nat
  double_chocolate : Nat

/-- Calculates the total number of cookies made by the baker --/
def total_cookies (batch : CookieBatch) (count : BatchCount) : Nat :=
  batch.chocolate_chip * count.chocolate_chip +
  batch.oatmeal * count.oatmeal +
  batch.sugar * count.sugar +
  batch.double_chocolate * count.double_chocolate

/-- Theorem stating that the baker made 77 cookies in total --/
theorem baker_total_cookies :
  let batch := CookieBatch.mk 8 7 10 6
  let count := BatchCount.mk 5 3 1 1
  total_cookies batch count = 77 := by
  sorry

end NUMINAMATH_CALUDE_baker_total_cookies_l4109_410984


namespace NUMINAMATH_CALUDE_window_purchase_savings_l4109_410970

def window_price : ℕ := 150
def alice_windows : ℕ := 9
def bob_windows : ℕ := 10

def discount (n : ℕ) : ℕ :=
  (n / 6) * window_price

def cost (n : ℕ) : ℕ :=
  n * window_price - discount n

def total_separate_cost : ℕ :=
  cost alice_windows + cost bob_windows

def joint_windows : ℕ :=
  alice_windows + bob_windows

def joint_cost : ℕ :=
  cost joint_windows

def savings : ℕ :=
  total_separate_cost - joint_cost

theorem window_purchase_savings :
  savings = 150 := by sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l4109_410970


namespace NUMINAMATH_CALUDE_cone_central_angle_l4109_410921

/-- Represents a cone with its surface areas and central angle. -/
structure Cone where
  base_area : ℝ
  total_surface_area : ℝ
  lateral_surface_area : ℝ
  central_angle : ℝ

/-- The theorem stating the relationship between the cone's surface areas and its central angle. -/
theorem cone_central_angle (c : Cone) 
  (h1 : c.total_surface_area = 3 * c.base_area)
  (h2 : c.lateral_surface_area = 2 * c.base_area)
  (h3 : c.lateral_surface_area = (c.central_angle / 360) * (2 * π * c.base_area)) :
  c.central_angle = 240 := by
  sorry


end NUMINAMATH_CALUDE_cone_central_angle_l4109_410921


namespace NUMINAMATH_CALUDE_segment_length_l4109_410941

/-- Given three points on a line, prove that the length of AC is either 7 or 1 -/
theorem segment_length (A B C : ℝ) : 
  (B - A = 4) → (C - B = 3) → (C - A = 7 ∨ C - A = 1) := by sorry

end NUMINAMATH_CALUDE_segment_length_l4109_410941


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_equals_10_l4109_410905

-- Define the equation
def equation (a : ℝ) (x : ℝ) : Prop :=
  (x * Real.log a ^ 2 - 1) / (x + Real.log a) = x

-- Theorem statement
theorem unique_solution_implies_a_equals_10 :
  (∃! x : ℝ, equation a x) → a = 10 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_equals_10_l4109_410905


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l4109_410949

theorem cubic_root_ratio (a b c d : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 1/2 ∨ x = 4) →
  c / d = 9/4 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l4109_410949


namespace NUMINAMATH_CALUDE_flower_shop_sales_ratio_l4109_410937

/-- Proves that the ratio of Tuesday's sales to Monday's sales is 3:1 given the conditions of the flower shop's three-day sale. -/
theorem flower_shop_sales_ratio : 
  ∀ (tuesday_sales : ℕ),
  12 + tuesday_sales + tuesday_sales / 3 = 60 →
  tuesday_sales / 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_sales_ratio_l4109_410937


namespace NUMINAMATH_CALUDE_sqrt_2x_plus_3_eq_x_solution_l4109_410962

theorem sqrt_2x_plus_3_eq_x_solution :
  ∃! x : ℝ, Real.sqrt (2 * x + 3) = x :=
by
  -- The unique solution is x = 3
  use 3
  constructor
  · -- Prove that x = 3 satisfies the equation
    sorry
  · -- Prove that any solution must be equal to 3
    sorry

#check sqrt_2x_plus_3_eq_x_solution

end NUMINAMATH_CALUDE_sqrt_2x_plus_3_eq_x_solution_l4109_410962


namespace NUMINAMATH_CALUDE_completing_square_result_l4109_410955

theorem completing_square_result (x : ℝ) :
  x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_completing_square_result_l4109_410955


namespace NUMINAMATH_CALUDE_prob_three_red_standard_deck_l4109_410999

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = 52)
  (h_red : red_cards = 26)

/-- The probability of drawing three red cards from a standard deck -/
def prob_three_red (d : Deck) : ℚ :=
  (d.red_cards * (d.red_cards - 1) * (d.red_cards - 2)) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- Theorem stating the probability of drawing three red cards from a standard deck -/
theorem prob_three_red_standard_deck :
  ∃ (d : Deck), prob_three_red d = 200 / 1701 :=
sorry

end NUMINAMATH_CALUDE_prob_three_red_standard_deck_l4109_410999


namespace NUMINAMATH_CALUDE_select_five_from_eight_l4109_410978

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l4109_410978


namespace NUMINAMATH_CALUDE_max_value_of_f_l4109_410964

def f (a b : ℕ) : ℚ :=
  (a : ℚ) / (10 * b + a) + (b : ℚ) / (10 * a + b)

theorem max_value_of_f :
  ∀ a b : ℕ,
  a ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ) →
  b ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ) →
  f a b ≤ 89 / 287 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4109_410964


namespace NUMINAMATH_CALUDE_grocer_banana_purchase_l4109_410965

/-- Proves that the grocer purchased 792 pounds of bananas given the conditions -/
theorem grocer_banana_purchase :
  ∀ (pounds : ℝ),
  (pounds / 3 * 0.50 = pounds / 4 * 1.00 - 11.00) →
  pounds = 792 := by
sorry

end NUMINAMATH_CALUDE_grocer_banana_purchase_l4109_410965


namespace NUMINAMATH_CALUDE_tennis_ball_cost_l4109_410974

theorem tennis_ball_cost (num_packs : ℕ) (total_cost : ℚ) (balls_per_pack : ℕ) 
  (h1 : num_packs = 4)
  (h2 : total_cost = 24)
  (h3 : balls_per_pack = 3) :
  total_cost / (num_packs * balls_per_pack) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tennis_ball_cost_l4109_410974


namespace NUMINAMATH_CALUDE_clara_cookie_sales_l4109_410922

/-- Proves the number of boxes of the third type of cookies Clara sells -/
theorem clara_cookie_sales (cookies_per_box1 cookies_per_box2 cookies_per_box3 : ℕ)
  (boxes_sold1 boxes_sold2 : ℕ) (total_cookies : ℕ)
  (h1 : cookies_per_box1 = 12)
  (h2 : cookies_per_box2 = 20)
  (h3 : cookies_per_box3 = 16)
  (h4 : boxes_sold1 = 50)
  (h5 : boxes_sold2 = 80)
  (h6 : total_cookies = 3320)
  (h7 : total_cookies = cookies_per_box1 * boxes_sold1 + cookies_per_box2 * boxes_sold2 + cookies_per_box3 * boxes_sold3) :
  boxes_sold3 = 70 := by
  sorry

end NUMINAMATH_CALUDE_clara_cookie_sales_l4109_410922


namespace NUMINAMATH_CALUDE_vector_BC_l4109_410956

/-- Given points A and B, and vector AC, prove that vector BC is (-3, 2) -/
theorem vector_BC (A B C : ℝ × ℝ) : 
  A = (-1, 1) → B = (0, 2) → (C.1 - A.1, C.2 - A.2) = (-2, 3) → 
  (C.1 - B.1, C.2 - B.2) = (-3, 2) := by sorry

end NUMINAMATH_CALUDE_vector_BC_l4109_410956


namespace NUMINAMATH_CALUDE_f_composition_value_l4109_410919

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else 2 * Real.sqrt 2 * Real.cos x

theorem f_composition_value : f (f (-Real.pi/4)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l4109_410919


namespace NUMINAMATH_CALUDE_function_properties_l4109_410910

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- Define the specific function h
def h (m : ℝ) (x : ℝ) : ℝ := f 2 9 x - m + 1

-- Theorem statement
theorem function_properties :
  (∃ (a b : ℝ), f a b (-1) = 0 ∧ 
   (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b x ≥ f a b (-1)) ∧
   (∀ x : ℝ, f a b x = x^3 + 6*x^2 + 9*x + 4)) ∧
  (∀ m : ℝ, (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    h m x₁ = 0 ∧ h m x₂ = 0 ∧ h m x₃ = 0) ↔ 1 < m ∧ m < 5) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4109_410910


namespace NUMINAMATH_CALUDE_base8_to_base10_77_l4109_410928

/-- Converts a two-digit number in base 8 to base 10 -/
def base8_to_base10 (a b : Nat) : Nat :=
  a * 8 + b

/-- The given number in base 8 -/
def number_base8 : Nat × Nat := (7, 7)

theorem base8_to_base10_77 :
  base8_to_base10 number_base8.1 number_base8.2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_77_l4109_410928


namespace NUMINAMATH_CALUDE_asian_games_ticket_scientific_notation_l4109_410943

theorem asian_games_ticket_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), 1 ≤ a ∧ a < 10 ∧ 113700 = a * (10 : ℝ) ^ b ∧ a = 1.137 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_asian_games_ticket_scientific_notation_l4109_410943


namespace NUMINAMATH_CALUDE_carpet_border_problem_l4109_410993

theorem carpet_border_problem :
  let count_valid_pairs := 
    (Finset.filter 
      (fun pair : ℕ × ℕ => 
        let (p, q) := pair
        q > p ∧ (p - 6) * (q - 6) = 48 ∧ p > 6 ∧ q > 6)
      (Finset.product (Finset.range 100) (Finset.range 100))).card
  count_valid_pairs = 5 := by
sorry

end NUMINAMATH_CALUDE_carpet_border_problem_l4109_410993


namespace NUMINAMATH_CALUDE_construction_cost_difference_equals_profit_l4109_410979

/-- Represents the construction and sale details of houses in an area --/
structure HouseData where
  other_sale_price : ℕ
  certain_sale_multiplier : ℚ
  profit : ℕ

/-- Calculates the difference in construction cost between a certain house and other houses --/
def construction_cost_difference (data : HouseData) : ℕ :=
  data.profit

theorem construction_cost_difference_equals_profit (data : HouseData)
  (h1 : data.other_sale_price = 320000)
  (h2 : data.certain_sale_multiplier = 3/2)
  (h3 : data.profit = 60000) :
  construction_cost_difference data = data.profit := by
  sorry

#eval construction_cost_difference { other_sale_price := 320000, certain_sale_multiplier := 3/2, profit := 60000 }

end NUMINAMATH_CALUDE_construction_cost_difference_equals_profit_l4109_410979


namespace NUMINAMATH_CALUDE_johnnys_jogging_speed_l4109_410972

/-- Proves that given the specified conditions, Johnny's jogging speed to school is approximately 9.333333333333334 miles per hour -/
theorem johnnys_jogging_speed 
  (total_time : ℝ) 
  (distance : ℝ) 
  (bus_speed : ℝ) 
  (h1 : total_time = 1) 
  (h2 : distance = 6.461538461538462) 
  (h3 : bus_speed = 21) : 
  ∃ (jogging_speed : ℝ), 
    (distance / jogging_speed + distance / bus_speed = total_time) ∧ 
    (abs (jogging_speed - 9.333333333333334) < 0.000001) := by
  sorry

end NUMINAMATH_CALUDE_johnnys_jogging_speed_l4109_410972


namespace NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l4109_410938

-- Problem 1
theorem factorization_problem1 (x y : ℝ) :
  x^3 + 2*x^2*y + x*y^2 = x*(x + y)^2 := by sorry

-- Problem 2
theorem factorization_problem2 (m n : ℝ) :
  4*m^2 - n^2 - 4*m + 1 = (2*m - 1 + n)*(2*m - 1 - n) := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l4109_410938


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l4109_410953

/-- Given that α and β are the roots of x^2 + x - 1 = 0, prove that 2α^5 + β^3 = -13 ± 4√5 -/
theorem quadratic_roots_sum (α β : ℝ) : 
  α^2 + α - 1 = 0 → β^2 + β - 1 = 0 → 
  2 * α^5 + β^3 = -13 + 4 * Real.sqrt 5 ∨ 2 * α^5 + β^3 = -13 - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l4109_410953


namespace NUMINAMATH_CALUDE_function_value_theorem_l4109_410971

theorem function_value_theorem (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x + 3) :
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_function_value_theorem_l4109_410971


namespace NUMINAMATH_CALUDE_inscribed_rectangle_coefficient_l4109_410982

/-- Triangle ABC with inscribed rectangle PQRS --/
structure TriangleWithRectangle where
  /-- Side length AB --/
  ab : ℝ
  /-- Side length BC --/
  bc : ℝ
  /-- Side length CA --/
  ca : ℝ
  /-- Width of the inscribed rectangle (PQ) --/
  ω : ℝ
  /-- Coefficient α in the area formula --/
  α : ℝ
  /-- Coefficient β in the area formula --/
  β : ℝ
  /-- P is on AB, Q on AC, R and S on BC --/
  rectangle_inscribed : Bool
  /-- Area formula for rectangle PQRS --/
  area_formula : ℝ → ℝ := fun ω => α * ω - β * ω^2

/-- The main theorem --/
theorem inscribed_rectangle_coefficient
  (t : TriangleWithRectangle)
  (h1 : t.ab = 15)
  (h2 : t.bc = 26)
  (h3 : t.ca = 25)
  (h4 : t.rectangle_inscribed = true) :
  t.β = 33 / 28 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_coefficient_l4109_410982


namespace NUMINAMATH_CALUDE_only_2017_is_prime_l4109_410952

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem only_2017_is_prime :
  ¬(is_prime 2015) ∧
  ¬(is_prime 2016) ∧
  is_prime 2017 ∧
  ¬(is_prime 2018) ∧
  ¬(is_prime 2019) :=
sorry

end NUMINAMATH_CALUDE_only_2017_is_prime_l4109_410952


namespace NUMINAMATH_CALUDE_paco_cookies_proof_l4109_410975

/-- The number of cookies Paco had initially -/
def initial_cookies : ℕ := 2

theorem paco_cookies_proof :
  (∃ (x : ℕ), 
    (x - 2 + 36 = 2 + 34) ∧ 
    (x = initial_cookies)) := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_proof_l4109_410975


namespace NUMINAMATH_CALUDE_angle_D_measure_l4109_410958

-- Define the hexagon and its angles
def Hexagon (A B C D E F : ℝ) : Prop :=
  -- Convexity condition (sum of angles = 720°)
  A + B + C + D + E + F = 720 ∧
  -- Angle congruence conditions
  A = B ∧ B = C ∧
  D = E ∧
  F = 2 * D ∧
  -- Relationship between angles A and D
  A + 30 = D

-- Theorem statement
theorem angle_D_measure (A B C D E F : ℝ) :
  Hexagon A B C D E F → D = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l4109_410958


namespace NUMINAMATH_CALUDE_van_capacity_l4109_410960

theorem van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) (h1 : students = 2) (h2 : adults = 6) (h3 : vans = 2) :
  (students + adults) / vans = 4 := by
  sorry

end NUMINAMATH_CALUDE_van_capacity_l4109_410960


namespace NUMINAMATH_CALUDE_scientific_notation_of_20160_l4109_410934

theorem scientific_notation_of_20160 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 20160 = a * (10 : ℝ) ^ n ∧ a = 2.016 ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_20160_l4109_410934


namespace NUMINAMATH_CALUDE_total_weekend_hours_l4109_410903

-- Define the working hours for Saturday and Sunday
def saturday_hours : ℕ := 6
def sunday_hours : ℕ := 4

-- Theorem to prove
theorem total_weekend_hours :
  saturday_hours + sunday_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_weekend_hours_l4109_410903


namespace NUMINAMATH_CALUDE_brick_width_calculation_l4109_410992

/-- Calculates the width of a brick given the dimensions of a wall and the number of bricks needed. -/
theorem brick_width_calculation (wall_length wall_width wall_height : ℝ)
  (brick_length brick_height : ℝ) (num_bricks : ℝ) :
  wall_length = 9 →
  wall_width = 5 →
  wall_height = 18.5 →
  brick_length = 0.21 →
  brick_height = 0.08 →
  num_bricks = 4955.357142857142 →
  ∃ (brick_width : ℝ), abs (brick_width - 0.295) < 0.001 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry


end NUMINAMATH_CALUDE_brick_width_calculation_l4109_410992


namespace NUMINAMATH_CALUDE_expression_evaluation_l4109_410930

theorem expression_evaluation : 2 * (5 * 9) + 3 * (4 * 11) + (2^3 * 7) + 6 * (3 * 5) = 368 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4109_410930


namespace NUMINAMATH_CALUDE_line_not_in_first_quadrant_l4109_410976

/-- A line y = -3x + b that does not pass through the first quadrant has b ≤ 0 -/
theorem line_not_in_first_quadrant (b : ℝ) : 
  (∀ x y : ℝ, y = -3 * x + b → ¬(x > 0 ∧ y > 0)) → b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_first_quadrant_l4109_410976


namespace NUMINAMATH_CALUDE_expand_expression_l4109_410925

theorem expand_expression (x : ℝ) : (x - 3) * (x + 6) = x^2 + 3*x - 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l4109_410925


namespace NUMINAMATH_CALUDE_min_ceiling_height_for_illumination_l4109_410991

/-- The minimum ceiling height for complete illumination of a rectangular field. -/
theorem min_ceiling_height_for_illumination (length width : ℝ) 
  (h : ℝ) (multiple : ℝ) : 
  length = 100 →
  width = 80 →
  multiple = 0.1 →
  (∃ (n : ℕ), h = n * multiple) →
  (2 * h ≥ Real.sqrt (length^2 + width^2)) →
  (∀ (h' : ℝ), (∃ (n : ℕ), h' = n * multiple) → 
    (2 * h' ≥ Real.sqrt (length^2 + width^2)) → h' ≥ h) →
  h = 32.1 :=
by sorry

end NUMINAMATH_CALUDE_min_ceiling_height_for_illumination_l4109_410991


namespace NUMINAMATH_CALUDE_power_five_addition_l4109_410906

theorem power_five_addition (a : ℝ) : a^5 + a^5 = 2*a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_five_addition_l4109_410906


namespace NUMINAMATH_CALUDE_only_set_C_forms_triangle_l4109_410907

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- The given sets of line segments -/
def set_A : (ℝ × ℝ × ℝ) := (3, 5, 8)
def set_B : (ℝ × ℝ × ℝ) := (8, 8, 18)
def set_C : (ℝ × ℝ × ℝ) := (1, 1, 1)
def set_D : (ℝ × ℝ × ℝ) := (3, 4, 8)

/-- Theorem: Among the given sets, only set C can form a triangle -/
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A.1 set_A.2.1 set_A.2.2) ∧
  ¬(can_form_triangle set_B.1 set_B.2.1 set_B.2.2) ∧
  can_form_triangle set_C.1 set_C.2.1 set_C.2.2 ∧
  ¬(can_form_triangle set_D.1 set_D.2.1 set_D.2.2) :=
by sorry

end NUMINAMATH_CALUDE_only_set_C_forms_triangle_l4109_410907


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l4109_410901

def U : Set ℝ := Set.univ
def A : Set ℝ := {-3, -2, -1, 0, 1, 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {-3, -2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l4109_410901


namespace NUMINAMATH_CALUDE_maize_stolen_l4109_410932

def months_in_year : ℕ := 12
def years : ℕ := 2
def maize_per_month : ℕ := 1
def donation : ℕ := 8
def final_amount : ℕ := 27

theorem maize_stolen : 
  (months_in_year * years * maize_per_month + donation) - final_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_maize_stolen_l4109_410932


namespace NUMINAMATH_CALUDE_quadratic_root_value_l4109_410939

theorem quadratic_root_value (r s : ℝ) : 
  (∃ x : ℂ, 2 * x^2 + r * x + s = 0 ∧ x = 3 + 2*I) → s = 26 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l4109_410939


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l4109_410989

theorem two_numbers_with_given_means : ∃ a b : ℝ, 
  a > 0 ∧ b > 0 ∧ 
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  2 / (1/a + 1/b) = 5/3 ∧
  a = (15 + Real.sqrt 145) / 4 ∧
  b = (15 - Real.sqrt 145) / 4 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l4109_410989


namespace NUMINAMATH_CALUDE_biased_coin_expected_value_l4109_410908

/-- The expected value of winnings for a biased coin flip -/
theorem biased_coin_expected_value :
  let p_heads : ℚ := 1/4  -- Probability of heads
  let p_tails : ℚ := 3/4  -- Probability of tails
  let win_heads : ℚ := 4  -- Amount won for heads
  let lose_tails : ℚ := 3 -- Amount lost for tails
  p_heads * win_heads - p_tails * lose_tails = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_expected_value_l4109_410908


namespace NUMINAMATH_CALUDE_sum_of_fractions_l4109_410951

theorem sum_of_fractions : 
  (4 : ℚ) / 3 + 8 / 9 + 18 / 27 + 40 / 81 + 88 / 243 - 5 = -305 / 243 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l4109_410951


namespace NUMINAMATH_CALUDE_train_speed_l4109_410942

/-- Prove that given the conditions, the train's speed is 20 miles per hour. -/
theorem train_speed (distance_to_work : ℝ) (walking_speed : ℝ) (additional_train_time : ℝ) 
  (walking_vs_train_time_diff : ℝ) :
  distance_to_work = 1.5 →
  walking_speed = 3 →
  additional_train_time = 10.5 / 60 →
  walking_vs_train_time_diff = 15 / 60 →
  ∃ (train_speed : ℝ), 
    train_speed = 20 ∧
    distance_to_work / walking_speed = 
      distance_to_work / train_speed + additional_train_time + walking_vs_train_time_diff :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l4109_410942


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l4109_410948

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → a 5 * a 14 = 5 → a 8 * a 9 * a 10 * a 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l4109_410948


namespace NUMINAMATH_CALUDE_intersection_with_complement_example_l4109_410967

open Set

theorem intersection_with_complement_example : 
  let U : Set ℕ := {1, 3, 5, 7, 9}
  let A : Set ℕ := {3, 7, 9}
  let B : Set ℕ := {1, 9}
  A ∩ (U \ B) = {3, 7} := by
sorry

end NUMINAMATH_CALUDE_intersection_with_complement_example_l4109_410967


namespace NUMINAMATH_CALUDE_y_coordinate_range_l4109_410969

/-- The parabola equation y^2 = x + 4 -/
def parabola (x y : ℝ) : Prop := y^2 = x + 4

/-- Point A is at (0,2) -/
def point_A : ℝ × ℝ := (0, 2)

/-- B is on the parabola -/
def B_on_parabola (B : ℝ × ℝ) : Prop := parabola B.1 B.2

/-- C is on the parabola -/
def C_on_parabola (C : ℝ × ℝ) : Prop := parabola C.1 C.2

/-- AB is perpendicular to BC -/
def AB_perp_BC (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.2 - B.2) = -(B.1 - A.1) * (C.1 - B.1)

/-- The main theorem -/
theorem y_coordinate_range (B C : ℝ × ℝ) :
  B_on_parabola B → C_on_parabola C → AB_perp_BC point_A B C →
  C.2 ≤ 0 ∨ C.2 ≥ 4 := by sorry

end NUMINAMATH_CALUDE_y_coordinate_range_l4109_410969


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l4109_410946

theorem polynomial_identity_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ + a₂ + a₀ = 41 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l4109_410946


namespace NUMINAMATH_CALUDE_cube_root_simplification_l4109_410980

theorem cube_root_simplification :
  let x : ℝ := 5488000
  let y : ℝ := 2744
  let z : ℝ := 343
  (1000 = 10^3) →
  (y = 2^3 * z) →
  (z = 7^3) →
  x^(1/3) = 140 * 2^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l4109_410980


namespace NUMINAMATH_CALUDE_percentage_calculation_l4109_410998

theorem percentage_calculation : 
  (0.47 * 1442 - 0.36 * 1412) + 66 = 235.42 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l4109_410998


namespace NUMINAMATH_CALUDE_sine_function_shifted_symmetric_l4109_410915

/-- Given a function f(x) = sin(ωx + φ), prove that under certain conditions, φ = π/6 -/
theorem sine_function_shifted_symmetric (ω φ : Real) : 
  ω > 0 → 
  0 < φ → 
  φ < Real.pi / 2 → 
  (fun x ↦ Real.sin (ω * x + φ)) 0 = -(fun x ↦ Real.sin (ω * x + φ)) (Real.pi / 2) →
  (∀ x, Real.sin (ω * (x + Real.pi / 12) + φ) = -Real.sin (ω * (-x + Real.pi / 12) + φ)) →
  φ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_shifted_symmetric_l4109_410915


namespace NUMINAMATH_CALUDE_circle_theorem_l4109_410923

-- Define the set of complex numbers satisfying the condition
def S : Set ℂ := {z : ℂ | Complex.abs (z - 3 * Complex.I) = 10}

-- State the theorem
theorem circle_theorem : 
  S = {z : ℂ | Complex.abs (z - Complex.ofReal 0 - Complex.I * 3) = 10} := by
sorry

end NUMINAMATH_CALUDE_circle_theorem_l4109_410923


namespace NUMINAMATH_CALUDE_smallest_n_for_fraction_inequality_l4109_410950

theorem smallest_n_for_fraction_inequality : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∀ (m : ℤ), 0 < m → m < 2004 → 
    ∃ (k : ℤ), (m : ℚ) / 2004 < (k : ℚ) / n ∧ (k : ℚ) / n < ((m + 1) : ℚ) / 2005) ∧
  (∀ (n' : ℕ), 0 < n' → n' < n → 
    ∃ (m : ℤ), 0 < m ∧ m < 2004 ∧
      ∀ (k : ℤ), ¬((m : ℚ) / 2004 < (k : ℚ) / n' ∧ (k : ℚ) / n' < ((m + 1) : ℚ) / 2005)) ∧
  n = 4009 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_fraction_inequality_l4109_410950


namespace NUMINAMATH_CALUDE_ski_price_after_discounts_l4109_410933

def original_price : ℝ := 200
def first_discount : ℝ := 0.4
def second_discount : ℝ := 0.2

theorem ski_price_after_discounts :
  let price_after_first := original_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  final_price = 96 := by sorry

end NUMINAMATH_CALUDE_ski_price_after_discounts_l4109_410933


namespace NUMINAMATH_CALUDE_youngest_brother_age_l4109_410913

theorem youngest_brother_age (a b c : ℕ) : 
  (a + b + c = 96) → 
  (b = a + 1) → 
  (c = a + 2) → 
  a = 31 := by
sorry

end NUMINAMATH_CALUDE_youngest_brother_age_l4109_410913


namespace NUMINAMATH_CALUDE_probability_theorem_l4109_410983

def num_attractions : ℕ := 5

def P_AB (n : ℕ) : ℚ := 8 / (n * n)

def P_B (n : ℕ) : ℚ := (n * (n - 1)) / (n * n)

theorem probability_theorem (n : ℕ) (h : n = num_attractions) :
  P_AB n / P_B n = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l4109_410983


namespace NUMINAMATH_CALUDE_complex_vector_properties_l4109_410959

open Complex

theorem complex_vector_properties (x y : ℝ) : 
  let z₁ : ℂ := (1 + I) / I
  let z₂ : ℂ := x + y * I
  true → 
  (∃ (k : ℝ), z₁.re * k = z₂.re ∧ z₁.im * k = z₂.im → x + y = 0) ∧
  (z₁.re * z₂.re + z₁.im * z₂.im = 0 → abs (z₁ + z₂) = abs (z₁ - z₂)) := by
  sorry

end NUMINAMATH_CALUDE_complex_vector_properties_l4109_410959


namespace NUMINAMATH_CALUDE_triangle_properties_l4109_410986

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def altitude_equation (t : Triangle) : ℝ → ℝ → Prop :=
  fun x y => x + 5 * y - 3 = 0

def side_BC_equation (t : Triangle) : ℝ → ℝ → Prop :=
  fun x y => x + 2 * y - 10 = 0

theorem triangle_properties (t : Triangle) :
  t.A = (-2, 1) →
  t.B = (4, 3) →
  (t.C = (3, -2) → altitude_equation t t.A.1 t.A.2) ∧
  (∃ M : ℝ × ℝ, M = (3, 1) ∧ M.1 = (t.A.1 + t.C.1) / 2 ∧ M.2 = (t.A.2 + t.C.2) / 2 →
    side_BC_equation t t.B.1 t.B.2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4109_410986


namespace NUMINAMATH_CALUDE_fraction_bounds_l4109_410945

theorem fraction_bounds (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  0 ≤ (|x + y|^2) / (|x|^2 + |y|^2) ∧ (|x + y|^2) / (|x|^2 + |y|^2) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_bounds_l4109_410945
