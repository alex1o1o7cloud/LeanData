import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inequality_from_inequality_l1288_128878

theorem triangle_inequality_from_inequality (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) :
  c < a + b ∧ a < b + c ∧ b < c + a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_inequality_l1288_128878


namespace NUMINAMATH_CALUDE_average_of_five_l1288_128886

/-- Given five real numbers x₁, x₂, x₃, x₄, x₅, if the average of x₁ and x₂ is 2
    and the average of x₃, x₄, and x₅ is 4, then the average of all five numbers is 3.2. -/
theorem average_of_five (x₁ x₂ x₃ x₄ x₅ : ℝ) 
    (h₁ : (x₁ + x₂) / 2 = 2)
    (h₂ : (x₃ + x₄ + x₅) / 3 = 4) :
    (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_l1288_128886


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l1288_128895

theorem circle_line_intersection_range (r : ℝ) (h_r_pos : r > 0) :
  (∀ m : ℝ, ∃ A B : ℝ × ℝ,
    (A.1^2 + A.2^2 = r^2) ∧
    (B.1^2 + B.2^2 = r^2) ∧
    (m * A.1 - A.2 + 1 = 0) ∧
    (m * B.1 - B.2 + 1 = 0) ∧
    A ≠ B) ∧
  (∃ m : ℝ, ∀ A B : ℝ × ℝ,
    (A.1^2 + A.2^2 = r^2) ∧
    (B.1^2 + B.2^2 = r^2) ∧
    (m * A.1 - A.2 + 1 = 0) ∧
    (m * B.1 - B.2 + 1 = 0) →
    (A.1 + B.1)^2 + (A.2 + B.2)^2 ≥ (B.1 - A.1)^2 + (B.2 - A.2)^2) →
  1 < r ∧ r ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l1288_128895


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1288_128807

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 6*m*x + 6

-- Define what it means for f to be decreasing on the interval (-∞, 3]
def is_decreasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 3 → f m x > f m y

-- State the theorem
theorem sufficient_not_necessary_condition :
  (m = 1 → is_decreasing_on_interval m) ∧
  ¬(is_decreasing_on_interval m → m = 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1288_128807


namespace NUMINAMATH_CALUDE_faster_speed_calculation_l1288_128874

/-- Proves that the faster speed is 20 km/hr given the conditions of the problem -/
theorem faster_speed_calculation (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 20)
  (h2 : actual_speed = 10)
  (h3 : additional_distance = 20) :
  let time := actual_distance / actual_speed
  let total_distance := actual_distance + additional_distance
  let faster_speed := total_distance / time
  faster_speed = 20 := by sorry

end NUMINAMATH_CALUDE_faster_speed_calculation_l1288_128874


namespace NUMINAMATH_CALUDE_nephews_difference_l1288_128824

theorem nephews_difference (alden_past : ℕ) (total : ℕ) : 
  alden_past = 50 →
  total = 260 →
  ∃ (alden_now vihaan : ℕ),
    alden_now = 2 * alden_past ∧
    vihaan > alden_now ∧
    alden_now + vihaan = total ∧
    vihaan - alden_now = 60 :=
by sorry

end NUMINAMATH_CALUDE_nephews_difference_l1288_128824


namespace NUMINAMATH_CALUDE_heart_ratio_l1288_128892

def heart (n m : ℝ) : ℝ := n^4 * m^3

theorem heart_ratio : (heart 3 5) / (heart 5 3) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_l1288_128892


namespace NUMINAMATH_CALUDE_opening_night_customers_count_l1288_128848

/-- Represents the revenue and customer data for a movie theater on a specific day. -/
structure TheaterData where
  matineePrice : ℕ
  eveningPrice : ℕ
  openingNightPrice : ℕ
  popcornPrice : ℕ
  matineeCustomers : ℕ
  eveningCustomers : ℕ
  totalRevenue : ℕ

/-- Calculates the number of opening night customers given theater data. -/
def openingNightCustomers (data : TheaterData) : ℕ :=
  let totalCustomers := data.matineeCustomers + data.eveningCustomers + (data.totalRevenue - 
    (data.matineePrice * data.matineeCustomers + 
     data.eveningPrice * data.eveningCustomers + 
     (data.popcornPrice * (data.matineeCustomers + data.eveningCustomers)) / 2)) / data.openingNightPrice
  (data.totalRevenue - 
   (data.matineePrice * data.matineeCustomers + 
    data.eveningPrice * data.eveningCustomers + 
    data.popcornPrice * totalCustomers / 2)) / data.openingNightPrice

theorem opening_night_customers_count (data : TheaterData) 
  (h1 : data.matineePrice = 5)
  (h2 : data.eveningPrice = 7)
  (h3 : data.openingNightPrice = 10)
  (h4 : data.popcornPrice = 10)
  (h5 : data.matineeCustomers = 32)
  (h6 : data.eveningCustomers = 40)
  (h7 : data.totalRevenue = 1670) :
  openingNightCustomers data = 58 := by
  sorry

#eval openingNightCustomers {
  matineePrice := 5,
  eveningPrice := 7,
  openingNightPrice := 10,
  popcornPrice := 10,
  matineeCustomers := 32,
  eveningCustomers := 40,
  totalRevenue := 1670
}

end NUMINAMATH_CALUDE_opening_night_customers_count_l1288_128848


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l1288_128891

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  ∀ t : ℕ, t > 0 → t ∣ n → t ≤ 12 ∧ 12 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l1288_128891


namespace NUMINAMATH_CALUDE_point_coordinates_l1288_128808

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The third quadrant of the 2D plane -/
def ThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point in the third quadrant with specific distances to axes has coordinates (-5, -2) -/
theorem point_coordinates (p : Point) :
  ThirdQuadrant p →
  DistanceToXAxis p = 2 →
  DistanceToYAxis p = 5 →
  p = Point.mk (-5) (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1288_128808


namespace NUMINAMATH_CALUDE_min_value_theorem_l1288_128800

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / x + 1 / y = 1) :
  3 * x + 4 * y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), 3 * x₀ + 4 * y₀ = 25 ∧ 3 / x₀ + 1 / y₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1288_128800


namespace NUMINAMATH_CALUDE_xy_value_l1288_128899

theorem xy_value (x y : ℝ) 
  (eq1 : (4 : ℝ)^x / (2 : ℝ)^(x + y) = 8)
  (eq2 : (9 : ℝ)^(x + y) / (3 : ℝ)^(5 * y) = 243) :
  x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1288_128899


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1288_128888

-- Problem 1
theorem problem_1 : (-36 : ℚ) * (5/4 - 5/6 - 11/12) = 18 := by sorry

-- Problem 2
theorem problem_2 : (-2)^2 - 3 * (-1)^3 + 0 * (-2)^3 = 7 := by sorry

-- Problem 3
theorem problem_3 (x y : ℚ) (hx : x = -2) (hy : y = 1/2) :
  3 * x^2 * y - 2 * x * y^2 - 3/2 * (x^2 * y - 2 * x * y^2) = 5/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1288_128888


namespace NUMINAMATH_CALUDE_matches_left_after_2022_l1288_128841

/-- The number of matchsticks needed to form a digit --/
def matchsticks_for_digit (d : Nat) : Nat :=
  if d = 2 then 5
  else if d = 0 then 6
  else 0  -- We only care about 2 and 0 for this problem

/-- The number of matchsticks needed to form the year 2022 --/
def matchsticks_for_2022 : Nat :=
  matchsticks_for_digit 2 * 3 + matchsticks_for_digit 0

/-- The initial number of matches in the box --/
def initial_matches : Nat := 30

/-- Theorem: After forming 2022 with matchsticks, 9 matches will be left --/
theorem matches_left_after_2022 :
  initial_matches - matchsticks_for_2022 = 9 := by
  sorry


end NUMINAMATH_CALUDE_matches_left_after_2022_l1288_128841


namespace NUMINAMATH_CALUDE_product_positive_l1288_128885

theorem product_positive (x y z t : ℝ) 
  (h1 : x > y^3) 
  (h2 : y > z^3) 
  (h3 : z > t^3) 
  (h4 : t > x^3) : 
  x * y * z * t > 0 := by
sorry

end NUMINAMATH_CALUDE_product_positive_l1288_128885


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1288_128831

theorem complex_number_quadrant : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i)^2 / (1 + i)
  (z.re < 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1288_128831


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_39_l1288_128803

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define the factorization of 39
def factorization_39 : ℕ × ℕ := (3, 13)

-- Theorem statement
theorem no_primes_divisible_by_39 : 
  ¬∃ p : ℕ, isPrime p ∧ 39 ∣ p :=
sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_39_l1288_128803


namespace NUMINAMATH_CALUDE_farm_animal_pricing_l1288_128838

theorem farm_animal_pricing (num_cows : ℕ) (num_pigs : ℕ) (price_per_pig : ℕ) (total_earnings : ℕ) :
  num_cows = 20 →
  num_pigs = 4 * num_cows →
  price_per_pig = 400 →
  total_earnings = 48000 →
  (total_earnings - num_pigs * price_per_pig) / num_cows = 800 :=
by sorry

end NUMINAMATH_CALUDE_farm_animal_pricing_l1288_128838


namespace NUMINAMATH_CALUDE_vertical_shift_theorem_l1288_128853

/-- The original line function -/
def original_line (x : ℝ) : ℝ := 2 * x

/-- The vertical shift amount -/
def shift : ℝ := 5

/-- The resulting line after vertical shift -/
def shifted_line (x : ℝ) : ℝ := original_line x + shift

theorem vertical_shift_theorem :
  ∀ x : ℝ, shifted_line x = 2 * x + 5 := by sorry

end NUMINAMATH_CALUDE_vertical_shift_theorem_l1288_128853


namespace NUMINAMATH_CALUDE_factor_polynomial_l1288_128897

theorem factor_polynomial (x : ℝ) : 75 * x^3 - 250 * x^7 = 25 * x^3 * (3 - 10 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1288_128897


namespace NUMINAMATH_CALUDE_first_triangle_isosceles_l1288_128864

theorem first_triangle_isosceles (α β γ : Real) (θ₁ θ₂ : Real) : 
  α + β + γ = π → 
  α + β = θ₁ → 
  α + γ = θ₂ → 
  θ₁ + θ₂ < π →
  β = γ := by
sorry

end NUMINAMATH_CALUDE_first_triangle_isosceles_l1288_128864


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l1288_128837

theorem binomial_coefficient_divisibility (p n : ℕ) : 
  Prime p → n ≥ p → ∃ k : ℕ, Nat.choose n p - n / p = k * p := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l1288_128837


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1288_128805

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The theorem to prove -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 4 + a 6 + a 8 + a 10 = 80 →
  a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1288_128805


namespace NUMINAMATH_CALUDE_equal_angles_l1288_128898

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the right focus of the ellipse
def right_focus (F : ℝ × ℝ) : Prop := F.1 > 0 ∧ F.1^2 / 2 + F.2^2 = 1

-- Define a line passing through a point
def line_through (l : ℝ → ℝ) (p : ℝ × ℝ) : Prop := l p.1 = p.2

-- Define the intersection points of the line and the ellipse
def intersection_points (A B : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line_through l A ∧ line_through l B ∧ A ≠ B

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem equal_angles (F : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  right_focus F →
  line_through l F →
  intersection_points A B l →
  angle (0, 0) (2, 0) A = angle (0, 0) (2, 0) B :=
sorry

end NUMINAMATH_CALUDE_equal_angles_l1288_128898


namespace NUMINAMATH_CALUDE_total_ladybugs_count_l1288_128816

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem total_ladybugs_count : total_ladybugs = 67082 := by
  sorry

end NUMINAMATH_CALUDE_total_ladybugs_count_l1288_128816


namespace NUMINAMATH_CALUDE_fifteen_plus_neg_twentythree_l1288_128863

-- Define the operation for adding a positive and negative rational number
def add_pos_neg (a b : ℚ) : ℚ := -(b - a)

-- Theorem statement
theorem fifteen_plus_neg_twentythree :
  15 + (-23) = add_pos_neg 15 23 :=
sorry

end NUMINAMATH_CALUDE_fifteen_plus_neg_twentythree_l1288_128863


namespace NUMINAMATH_CALUDE_matrix_equation_holds_l1288_128893

def M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 2, 4]

theorem matrix_equation_holds :
  M^3 - 2 • M^2 + (-12) • M = 3 • !![1, 2; 2, 4] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_holds_l1288_128893


namespace NUMINAMATH_CALUDE_kyles_weight_lifting_ratio_l1288_128810

/-- 
Given:
- Kyle can lift 60 more pounds this year
- He can now lift 80 pounds in total
Prove that the ratio of the additional weight to the weight he could lift last year is 3
-/
theorem kyles_weight_lifting_ratio : 
  ∀ (last_year_weight additional_weight total_weight : ℕ),
  additional_weight = 60 →
  total_weight = 80 →
  total_weight = last_year_weight + additional_weight →
  (additional_weight : ℚ) / last_year_weight = 3 := by
sorry

end NUMINAMATH_CALUDE_kyles_weight_lifting_ratio_l1288_128810


namespace NUMINAMATH_CALUDE_simplify_expression_l1288_128852

theorem simplify_expression (x : ℝ) (h : -1 < x ∧ x < 3) :
  Real.sqrt ((x - 3)^2) + |x + 1| = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1288_128852


namespace NUMINAMATH_CALUDE_ellipse_line_slope_l1288_128865

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (2, 0)

-- Define a line passing through a point with a given slope
def line_through_point (m : ℝ) (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

-- Define a circle passing through three points
def circle_through_points (p1 p2 p3 : ℝ × ℝ) (center : ℝ × ℝ) : Prop :=
  (p1.1 - center.1)^2 + (p1.2 - center.2)^2 =
  (p2.1 - center.1)^2 + (p2.2 - center.2)^2 ∧
  (p1.1 - center.1)^2 + (p1.2 - center.2)^2 =
  (p3.1 - center.1)^2 + (p3.2 - center.2)^2

-- Theorem statement
theorem ellipse_line_slope :
  ∀ (A B : ℝ × ℝ) (m : ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    line_through_point m right_focus A.1 A.2 ∧
    line_through_point m right_focus B.1 B.2 ∧
    (∃ (t : ℝ), circle_through_points A B (-Real.sqrt 7, 0) (0, t)) →
    m = Real.sqrt 2 / 2 ∨ m = -Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_l1288_128865


namespace NUMINAMATH_CALUDE_canadian_olympiad_2008_l1288_128833

theorem canadian_olympiad_2008 (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  (a - b*c)/(a + b*c) + (b - c*a)/(b + c*a) + (c - a*b)/(c + a*b) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_canadian_olympiad_2008_l1288_128833


namespace NUMINAMATH_CALUDE_binary_to_base4_correct_l1288_128840

/-- Converts a binary number to base 4 --/
def binary_to_base4 (b : ℕ) : ℕ := sorry

/-- The binary representation of the number --/
def binary_num : ℕ := 110110100

/-- The base 4 representation of the number --/
def base4_num : ℕ := 31220

/-- Theorem stating that the conversion of the binary number to base 4 is correct --/
theorem binary_to_base4_correct : binary_to_base4 binary_num = base4_num := by sorry

end NUMINAMATH_CALUDE_binary_to_base4_correct_l1288_128840


namespace NUMINAMATH_CALUDE_polynomial_value_at_2_l1288_128860

-- Define the polynomial coefficients
def a₃ : ℝ := 7
def a₂ : ℝ := 3
def a₁ : ℝ := -5
def a₀ : ℝ := 11

-- Define the point at which to evaluate the polynomial
def x : ℝ := 2

-- Define Horner's method for a cubic polynomial
def horner_cubic (a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₃ * x + a₂) * x + a₁) * x + a₀

-- Theorem statement
theorem polynomial_value_at_2 :
  horner_cubic a₃ a₂ a₁ a₀ x = 69 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_2_l1288_128860


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l1288_128870

/-- The number of ways to place n distinguishable balls into k distinguishable boxes -/
def placeBalls (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to place 5 distinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : placeBalls 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l1288_128870


namespace NUMINAMATH_CALUDE_triangle_area_l1288_128842

/-- Given a triangle with sides 6, 8, and 10, its area is 24 -/
theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1288_128842


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1288_128829

theorem largest_angle_in_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 4/3 of a right angle
  a + b = 4/3 * 90
  -- One angle is 36° larger than the other
  → b = a + 36
  -- All angles are non-negative
  → a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0
  -- Sum of all angles in a triangle is 180°
  → a + b + c = 180
  -- The largest angle is 78°
  → max a (max b c) = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1288_128829


namespace NUMINAMATH_CALUDE_problem_solution_l1288_128883

theorem problem_solution (a b : ℕ+) 
  (sum_constraint : a + b = 30)
  (equation_constraint : 2 * a * b + 12 * a = 3 * b + 270) :
  a * b = 216 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1288_128883


namespace NUMINAMATH_CALUDE_max_value_of_f_l1288_128827

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 5 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 4, f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1288_128827


namespace NUMINAMATH_CALUDE_john_work_hours_john_total_hours_l1288_128804

theorem john_work_hours : ℕ → ℕ → ℕ → ℕ
  | hours_per_day, start_day, end_day =>
    (end_day - start_day + 1) * hours_per_day

theorem john_total_hours : john_work_hours 8 3 7 = 40 := by
  sorry

end NUMINAMATH_CALUDE_john_work_hours_john_total_hours_l1288_128804


namespace NUMINAMATH_CALUDE_coin_problem_l1288_128850

theorem coin_problem (total_coins : ℕ) (total_value : ℕ) 
  (pennies nickels dimes quarters : ℕ) :
  total_coins = 11 →
  total_value = 165 →
  pennies ≥ 1 →
  nickels ≥ 1 →
  dimes ≥ 1 →
  quarters ≥ 1 →
  total_coins = pennies + nickels + dimes + quarters →
  total_value = pennies + 5 * nickels + 10 * dimes + 25 * quarters →
  quarters = 4 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l1288_128850


namespace NUMINAMATH_CALUDE_lowest_sale_price_percentage_l1288_128844

theorem lowest_sale_price_percentage (list_price : ℝ) (max_regular_discount : ℝ) (summer_discount : ℝ) :
  list_price = 80 →
  max_regular_discount = 0.7 →
  summer_discount = 0.2 →
  (list_price * (1 - max_regular_discount) - list_price * summer_discount) / list_price = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_lowest_sale_price_percentage_l1288_128844


namespace NUMINAMATH_CALUDE_tom_monthly_fluid_intake_l1288_128869

/-- Represents Tom's daily fluid intake --/
structure DailyFluidIntake where
  soda : Nat
  water : Nat
  juice : Nat
  sports_drink : Nat

/-- Represents Tom's additional weekend fluid intake --/
structure WeekendExtraFluidIntake where
  smoothie : Nat

/-- Represents the structure of a month --/
structure Month where
  weeks : Nat
  days_per_week : Nat
  weekdays_per_week : Nat
  weekend_days_per_week : Nat

def weekday_intake (d : DailyFluidIntake) : Nat :=
  d.soda * 12 + d.water + d.juice * 8 + d.sports_drink * 16

def weekend_intake (d : DailyFluidIntake) (w : WeekendExtraFluidIntake) : Nat :=
  weekday_intake d + w.smoothie

def total_monthly_intake (d : DailyFluidIntake) (w : WeekendExtraFluidIntake) (m : Month) : Nat :=
  (weekday_intake d * m.weekdays_per_week * m.weeks) +
  (weekend_intake d w * m.weekend_days_per_week * m.weeks)

theorem tom_monthly_fluid_intake :
  let tom_daily := DailyFluidIntake.mk 5 64 3 2
  let tom_weekend := WeekendExtraFluidIntake.mk 32
  let month := Month.mk 4 7 5 2
  total_monthly_intake tom_daily tom_weekend month = 5296 := by
  sorry

end NUMINAMATH_CALUDE_tom_monthly_fluid_intake_l1288_128869


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1288_128879

theorem division_remainder_proof (dividend : Nat) (divisor : Nat) (quotient : Nat) (h1 : dividend = 109) (h2 : divisor = 12) (h3 : quotient = 9) :
  dividend % divisor = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1288_128879


namespace NUMINAMATH_CALUDE_fence_perimeter_is_177_l1288_128876

/-- Calculates the outer perimeter of a rectangular fence with specified conditions -/
def fence_perimeter (num_posts : ℕ) (post_width : ℚ) (gap_width : ℚ) : ℚ :=
  let width_posts := num_posts / 4
  let length_posts := width_posts * 2
  let width := (width_posts - 1) * gap_width + width_posts * post_width
  let length := (length_posts - 1) * gap_width + length_posts * post_width
  2 * (width + length)

/-- The outer perimeter of the fence is 177 feet -/
theorem fence_perimeter_is_177 :
  fence_perimeter 36 (1/2) 3 = 177 := by sorry

end NUMINAMATH_CALUDE_fence_perimeter_is_177_l1288_128876


namespace NUMINAMATH_CALUDE_original_element_l1288_128849

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

/-- Theorem: If f(x, y) = (3, 1), then (x, y) = (1, 1) -/
theorem original_element (x y : ℝ) (h : f (x, y) = (3, 1)) : (x, y) = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_original_element_l1288_128849


namespace NUMINAMATH_CALUDE_difference_of_numbers_l1288_128834

theorem difference_of_numbers (x y : ℝ) 
  (sum_condition : x + y = 36) 
  (product_condition : x * y = 320) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l1288_128834


namespace NUMINAMATH_CALUDE_project_hours_l1288_128832

theorem project_hours (x y z : ℕ) (h1 : y = (5 * x) / 3) (h2 : z = 2 * x) (h3 : z = x + 30) :
  x + y + z = 140 :=
by sorry

end NUMINAMATH_CALUDE_project_hours_l1288_128832


namespace NUMINAMATH_CALUDE_quadratic_root_and_coefficient_l1288_128861

def quadratic_polynomial (x : ℂ) : ℂ := 3 * x^2 - 24 * x + 60

theorem quadratic_root_and_coefficient : 
  (quadratic_polynomial (4 + 2*Complex.I) = 0) ∧ 
  (∃ (a b : ℝ), ∀ x, quadratic_polynomial x = 3 * x^2 + a * x + b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_and_coefficient_l1288_128861


namespace NUMINAMATH_CALUDE_recess_time_calculation_l1288_128822

/-- Calculates the total recess time based on grade distribution -/
def total_recess_time (normal_recess : ℕ) 
  (extra_time_A extra_time_B extra_time_C extra_time_D extra_time_E extra_time_F : ℤ)
  (num_A num_B num_C num_D num_E num_F : ℕ) : ℤ :=
  normal_recess + 
  extra_time_A * num_A + 
  extra_time_B * num_B + 
  extra_time_C * num_C + 
  extra_time_D * num_D + 
  extra_time_E * num_E + 
  extra_time_F * num_F

theorem recess_time_calculation :
  total_recess_time 20 4 3 2 1 (-1) (-2) 10 12 14 5 3 2 = 122 := by
  sorry

end NUMINAMATH_CALUDE_recess_time_calculation_l1288_128822


namespace NUMINAMATH_CALUDE_table_length_proof_l1288_128859

/-- Proves that the length of the table is 77 cm given the conditions of the paper placement problem. -/
theorem table_length_proof (table_width : ℕ) (sheet_width sheet_height : ℕ) (x : ℕ) :
  table_width = 80 ∧
  sheet_width = 8 ∧
  sheet_height = 5 ∧
  (x - sheet_height : ℤ) = (table_width - sheet_width : ℤ) →
  x = 77 := by
  sorry

end NUMINAMATH_CALUDE_table_length_proof_l1288_128859


namespace NUMINAMATH_CALUDE_complement_of_25_36_l1288_128881

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 180 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_25_36 :
  complement ⟨25, 36⟩ = ⟨154, 24⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_25_36_l1288_128881


namespace NUMINAMATH_CALUDE_complex_number_proof_l1288_128826

theorem complex_number_proof (a : ℝ) (h_a : a > 0) (z : ℂ) (h_z : z = a - Complex.I) 
  (h_real : (z + 2 / z).im = 0) :
  z = 1 - Complex.I ∧ 
  ∀ m : ℝ, (((m : ℂ) - z)^2).re < 0 ∧ (((m : ℂ) - z)^2).im > 0 ↔ 1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_proof_l1288_128826


namespace NUMINAMATH_CALUDE_sum_of_products_l1288_128875

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 941) 
  (h2 : a + b + c = 31) : 
  a*b + b*c + c*a = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l1288_128875


namespace NUMINAMATH_CALUDE_eleven_divides_difference_l1288_128821

theorem eleven_divides_difference (A B C : ℕ) : 
  A ≠ C →
  A < 10 → B < 10 → C < 10 →
  ∃ k : ℤ, (100 * A + 10 * B + C) - (100 * C + 10 * B + A) = 11 * k :=
by sorry

end NUMINAMATH_CALUDE_eleven_divides_difference_l1288_128821


namespace NUMINAMATH_CALUDE_roja_speed_calculation_l1288_128868

/-- Roja's speed in km/hr -/
def rojaSpeed : ℝ := 8

/-- Pooja's speed in km/hr -/
def poojaSpeed : ℝ := 3

/-- Time elapsed in hours -/
def timeElapsed : ℝ := 4

/-- Distance between Roja and Pooja after the elapsed time in km -/
def distanceBetween : ℝ := 44

theorem roja_speed_calculation :
  rojaSpeed = 8 ∧
  poojaSpeed = 3 ∧
  timeElapsed = 4 ∧
  distanceBetween = 44 ∧
  distanceBetween = (rojaSpeed + poojaSpeed) * timeElapsed :=
by sorry

end NUMINAMATH_CALUDE_roja_speed_calculation_l1288_128868


namespace NUMINAMATH_CALUDE_otimes_example_l1288_128858

/-- Custom operation ⊗ defined as a ⊗ b = a² - ab -/
def otimes (a b : ℤ) : ℤ := a^2 - a * b

/-- Theorem stating that 4 ⊗ [2 ⊗ (-5)] = -40 -/
theorem otimes_example : otimes 4 (otimes 2 (-5)) = -40 := by
  sorry

end NUMINAMATH_CALUDE_otimes_example_l1288_128858


namespace NUMINAMATH_CALUDE_modified_arithmetic_sum_l1288_128843

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

theorem modified_arithmetic_sum :
  3 * (arithmetic_sum 110 119 10) = 3435 :=
by sorry

end NUMINAMATH_CALUDE_modified_arithmetic_sum_l1288_128843


namespace NUMINAMATH_CALUDE_triangle_side_length_l1288_128894

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    if c = 2a, b = 4, and cos B = 1/4, then c = 4 -/
theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : c = 2 * a) 
  (h2 : b = 4) 
  (h3 : Real.cos B = 1 / 4) : 
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1288_128894


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_geq_3_l1288_128871

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 6

theorem decreasing_function_implies_a_geq_3 :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 3 → f a x₁ > f a x₂) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_geq_3_l1288_128871


namespace NUMINAMATH_CALUDE_fraction_simplification_l1288_128880

theorem fraction_simplification :
  ((3^2008)^2 - (3^2006)^2) / ((3^2007)^2 - (3^2005)^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1288_128880


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_capacity_l1288_128887

/-- The number of people that can ride a Ferris wheel simultaneously -/
def ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) : ℕ :=
  num_seats * people_per_seat

/-- Theorem: A Ferris wheel with 14 seats, each holding 6 people, can accommodate 84 people -/
theorem paradise_park_ferris_wheel_capacity :
  ferris_wheel_capacity 14 6 = 84 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_capacity_l1288_128887


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1288_128896

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 4*a^2 + 6*a - 3 = 0) →
  (b^3 - 4*b^2 + 6*b - 3 = 0) →
  (c^3 - 4*c^2 + 6*c - 3 = 0) →
  (a/(b*c + 2) + b/(a*c + 2) + c/(a*b + 2) = 4/5) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1288_128896


namespace NUMINAMATH_CALUDE_valid_quadrilateral_set_l1288_128813

/-- A function that checks if a set of four line segments can form a valid quadrilateral. -/
def is_valid_quadrilateral (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a

/-- Theorem stating that among the given sets, only (2,2,2) forms a valid quadrilateral with side length 5. -/
theorem valid_quadrilateral_set :
  ¬ is_valid_quadrilateral 1 1 1 5 ∧
  ¬ is_valid_quadrilateral 1 2 2 5 ∧
  ¬ is_valid_quadrilateral 1 1 7 5 ∧
  is_valid_quadrilateral 2 2 2 5 :=
by sorry

end NUMINAMATH_CALUDE_valid_quadrilateral_set_l1288_128813


namespace NUMINAMATH_CALUDE_strawberries_picked_l1288_128811

theorem strawberries_picked (initial : ℕ) (final : ℕ) (picked : ℕ) : 
  initial = 42 → final = 120 → final = initial + picked → picked = 78 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_picked_l1288_128811


namespace NUMINAMATH_CALUDE_x_plus_reciprocal_two_implies_x_twelve_one_l1288_128825

theorem x_plus_reciprocal_two_implies_x_twelve_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_reciprocal_two_implies_x_twelve_one_l1288_128825


namespace NUMINAMATH_CALUDE_perfect_correlation_l1288_128851

/-- A sample point in a 2D plane -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The correlation coefficient -/
def correlationCoefficient (points : List SamplePoint) : ℝ :=
  sorry

/-- Theorem: If all sample points lie on a straight line with non-zero slope, 
    then the correlation coefficient R^2 is 1 -/
theorem perfect_correlation 
  (points : List SamplePoint) 
  (line : Line) 
  (h1 : line.slope ≠ 0) 
  (h2 : ∀ p ∈ points, p.y = line.slope * p.x + line.intercept) : 
  correlationCoefficient points = 1 :=
sorry

end NUMINAMATH_CALUDE_perfect_correlation_l1288_128851


namespace NUMINAMATH_CALUDE_single_rooms_count_l1288_128802

/-- The number of rooms for couples -/
def couple_rooms : ℕ := 13

/-- The amount of bubble bath needed for each bath in milliliters -/
def bath_amount : ℕ := 10

/-- The total amount of bubble bath needed when all rooms are at maximum capacity in milliliters -/
def total_bath_amount : ℕ := 400

/-- The number of single rooms in the hotel -/
def single_rooms : ℕ := total_bath_amount / bath_amount - 2 * couple_rooms

theorem single_rooms_count : single_rooms = 14 := by
  sorry

end NUMINAMATH_CALUDE_single_rooms_count_l1288_128802


namespace NUMINAMATH_CALUDE_grocery_store_diet_soda_l1288_128889

/-- The number of bottles of diet soda in a grocery store -/
def diet_soda_bottles (regular_soda_bottles : ℕ) (difference : ℕ) : ℕ :=
  regular_soda_bottles - difference

/-- Theorem: The grocery store has 4 bottles of diet soda -/
theorem grocery_store_diet_soda :
  diet_soda_bottles 83 79 = 4 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_diet_soda_l1288_128889


namespace NUMINAMATH_CALUDE_john_juice_bottles_l1288_128835

/-- The number of fluid ounces John needs -/
def required_oz : ℝ := 60

/-- The size of each bottle in milliliters -/
def bottle_size_ml : ℝ := 150

/-- The number of fluid ounces in 1 liter -/
def oz_per_liter : ℝ := 34

/-- The number of milliliters in 1 liter -/
def ml_per_liter : ℝ := 1000

/-- The smallest number of bottles John should buy -/
def min_bottles : ℕ := 12

theorem john_juice_bottles : 
  ∃ (n : ℕ), n = min_bottles ∧ 
  (n : ℝ) * bottle_size_ml / ml_per_liter * oz_per_liter ≥ required_oz ∧
  ∀ (m : ℕ), m < n → (m : ℝ) * bottle_size_ml / ml_per_liter * oz_per_liter < required_oz :=
by sorry

end NUMINAMATH_CALUDE_john_juice_bottles_l1288_128835


namespace NUMINAMATH_CALUDE_house_glass_panels_l1288_128839

/-- The number of glass panels in a house -/
def total_glass_panels (panels_per_window : ℕ) (double_windows : ℕ) (single_windows : ℕ) : ℕ :=
  panels_per_window * (2 * double_windows + single_windows)

/-- Theorem stating the total number of glass panels in the house -/
theorem house_glass_panels :
  let panels_per_window := 4
  let double_windows := 6
  let single_windows := 8
  total_glass_panels panels_per_window double_windows single_windows = 80 := by
sorry

end NUMINAMATH_CALUDE_house_glass_panels_l1288_128839


namespace NUMINAMATH_CALUDE_cost_of_one_each_l1288_128809

/-- The cost of goods A, B, and C -/
structure GoodsCost where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (cost : GoodsCost) : Prop :=
  cost.A + 2 * cost.B + 3 * cost.C = 136 ∧
  3 * cost.A + 2 * cost.B + cost.C = 240

/-- The theorem to be proved -/
theorem cost_of_one_each (cost : GoodsCost) 
  (h : satisfies_conditions cost) : 
  cost.A + cost.B + cost.C = 94 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_each_l1288_128809


namespace NUMINAMATH_CALUDE_exists_config_with_more_than_20_components_l1288_128845

/-- A configuration of diagonals on an 8x8 grid -/
def DiagonalConfiguration := Fin 8 → Fin 8 → Bool

/-- A point on the 8x8 grid -/
structure GridPoint where
  x : Fin 8
  y : Fin 8

/-- Two points are connected if they are in the same cell or adjacent cells with connecting diagonals -/
def connected (config : DiagonalConfiguration) (p1 p2 : GridPoint) : Prop :=
  sorry

/-- A connected component is a maximal set of connected points -/
def ConnectedComponent (config : DiagonalConfiguration) := Set GridPoint

/-- The number of connected components in a configuration -/
def numComponents (config : DiagonalConfiguration) : ℕ :=
  sorry

/-- There exists a configuration with more than 20 connected components -/
theorem exists_config_with_more_than_20_components :
  ∃ (config : DiagonalConfiguration), numComponents config > 20 :=
sorry

end NUMINAMATH_CALUDE_exists_config_with_more_than_20_components_l1288_128845


namespace NUMINAMATH_CALUDE_range_of_f_l1288_128884

noncomputable def f (x : ℝ) : ℝ := |Real.cos x| / Real.cos x + |Real.sin x| / Real.sin x

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, Real.cos x ≠ 0 ∧ Real.sin x ≠ 0 ∧ f x = y) ↔ y ∈ ({-2, 0, 2} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1288_128884


namespace NUMINAMATH_CALUDE_max_distance_between_circles_l1288_128819

/-- The maximum distance between the centers of two circles with 6-inch diameters
    placed within a 12-inch by 14-inch rectangle without extending beyond it. -/
def max_circle_centers_distance : ℝ := 10

/-- The width of the rectangle -/
def rectangle_width : ℝ := 12

/-- The height of the rectangle -/
def rectangle_height : ℝ := 14

/-- The diameter of each circle -/
def circle_diameter : ℝ := 6

/-- Theorem stating that the maximum distance between the centers of the circles is 10 inches -/
theorem max_distance_between_circles :
  ∀ (center1 center2 : ℝ × ℝ),
  (0 ≤ center1.1 ∧ center1.1 ≤ rectangle_width) →
  (0 ≤ center1.2 ∧ center1.2 ≤ rectangle_height) →
  (0 ≤ center2.1 ∧ center2.1 ≤ rectangle_width) →
  (0 ≤ center2.2 ∧ center2.2 ≤ rectangle_height) →
  (∀ (x y : ℝ), (x - center1.1)^2 + (y - center1.2)^2 ≤ (circle_diameter / 2)^2 →
    0 ≤ x ∧ x ≤ rectangle_width ∧ 0 ≤ y ∧ y ≤ rectangle_height) →
  (∀ (x y : ℝ), (x - center2.1)^2 + (y - center2.2)^2 ≤ (circle_diameter / 2)^2 →
    0 ≤ x ∧ x ≤ rectangle_width ∧ 0 ≤ y ∧ y ≤ rectangle_height) →
  (center1.1 - center2.1)^2 + (center1.2 - center2.2)^2 ≤ max_circle_centers_distance^2 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_circles_l1288_128819


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l1288_128801

theorem base_conversion_theorem (n : ℕ+) (A B : ℕ) : 
  (A < 8 ∧ B < 5) →
  (8 * A + B = n) →
  (5 * B + A = n) →
  (n : ℕ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l1288_128801


namespace NUMINAMATH_CALUDE_sheet_difference_l1288_128806

theorem sheet_difference : ∀ (tommy jimmy : ℕ), 
  jimmy = 32 →
  jimmy + 40 = tommy + 30 →
  tommy - jimmy = 10 := by
    sorry

end NUMINAMATH_CALUDE_sheet_difference_l1288_128806


namespace NUMINAMATH_CALUDE_x_younger_than_w_l1288_128855

-- Define the ages of the individuals
variable (w_years x_years y_years z_years : ℤ)

-- Define the conditions
axiom sum_condition : w_years + x_years = y_years + z_years + 15
axiom difference_condition : |w_years - x_years| = 2 * |y_years - z_years|
axiom w_z_relation : w_years = z_years + 30

-- Theorem to prove
theorem x_younger_than_w : x_years = w_years - 45 := by
  sorry

end NUMINAMATH_CALUDE_x_younger_than_w_l1288_128855


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1288_128812

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1288_128812


namespace NUMINAMATH_CALUDE_bus_trip_distance_l1288_128836

/-- The distance of a bus trip in miles. -/
def trip_distance : ℝ := 280

/-- The actual average speed of the bus in miles per hour. -/
def actual_speed : ℝ := 35

/-- The increased speed of the bus in miles per hour. -/
def increased_speed : ℝ := 40

/-- Theorem stating that the trip distance is 280 miles given the conditions. -/
theorem bus_trip_distance :
  (trip_distance / actual_speed - trip_distance / increased_speed = 1) →
  trip_distance = 280 := by
  sorry


end NUMINAMATH_CALUDE_bus_trip_distance_l1288_128836


namespace NUMINAMATH_CALUDE_min_values_theorem_l1288_128856

theorem min_values_theorem (a b c : ℕ+) (h : a^2 + b^2 - c^2 = 2018) :
  (∀ x y z : ℕ+, x^2 + y^2 - z^2 = 2018 → a + b - c ≤ x + y - z) ∧
  (∀ x y z : ℕ+, x^2 + y^2 - z^2 = 2018 → a + b + c ≤ x + y + z) ∧
  a + b - c = 2 ∧ a + b + c = 52 :=
sorry

end NUMINAMATH_CALUDE_min_values_theorem_l1288_128856


namespace NUMINAMATH_CALUDE_not_monotone_decreasing_periodic_function_l1288_128873

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Theorem 1: If f(1) > f(-1), then f is not monotonically decreasing on ℝ
theorem not_monotone_decreasing (h : f 1 > f (-1)) : 
  ¬ (∀ x y : ℝ, x ≤ y → f x ≥ f y) := by sorry

-- Theorem 2: If f(1+x) = f(x-1) for all x ∈ ℝ, then f is periodic
theorem periodic_function (h : ∀ x : ℝ, f (1 + x) = f (x - 1)) : 
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x := by sorry

end NUMINAMATH_CALUDE_not_monotone_decreasing_periodic_function_l1288_128873


namespace NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l1288_128882

theorem circle_area_equilateral_triangle (s : ℝ) (h : s = 12) :
  let R := s / Real.sqrt 3
  (π * R^2) = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l1288_128882


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l1288_128815

theorem cos_2alpha_value (α : ℝ) (h : Real.sin (α - 3 * Real.pi / 2) = 3 / 5) : 
  Real.cos (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l1288_128815


namespace NUMINAMATH_CALUDE_juans_number_problem_l1288_128877

theorem juans_number_problem (n : ℝ) : 
  (((n + 3) * 2 - 2) / 2 = 8) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_juans_number_problem_l1288_128877


namespace NUMINAMATH_CALUDE_average_tape_length_l1288_128820

def tape_lengths : List ℝ := [35, 29, 35.5, 36, 30.5]

theorem average_tape_length :
  (tape_lengths.sum / tape_lengths.length : ℝ) = 33.2 := by
  sorry

end NUMINAMATH_CALUDE_average_tape_length_l1288_128820


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1288_128867

theorem trigonometric_simplification :
  (Real.tan (20 * π / 180) + Real.tan (70 * π / 180) + Real.tan (80 * π / 180)) / Real.cos (30 * π / 180) =
  (1 + Real.cos (10 * π / 180) * Real.cos (20 * π / 180)) / (Real.cos (20 * π / 180) * Real.cos (70 * π / 180) * Real.cos (30 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1288_128867


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l1288_128866

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  geometric_sequence a → a 1 = 1 → a 5 = 9 → a 3 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l1288_128866


namespace NUMINAMATH_CALUDE_cristinas_croissants_l1288_128814

theorem cristinas_croissants (total_croissants : ℕ) (num_guests : ℕ) 
  (h1 : total_croissants = 17) 
  (h2 : num_guests = 7) : 
  total_croissants % num_guests = 3 := by
  sorry

end NUMINAMATH_CALUDE_cristinas_croissants_l1288_128814


namespace NUMINAMATH_CALUDE_log_equality_l1288_128817

theorem log_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l1288_128817


namespace NUMINAMATH_CALUDE_ab_max_and_inequality_l1288_128890

theorem ab_max_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + a = 15 - b) :
  (∃ (max : ℝ), max = 9 ∧ a * b ≤ max) ∧ b ≥ 6 - a := by
  sorry

end NUMINAMATH_CALUDE_ab_max_and_inequality_l1288_128890


namespace NUMINAMATH_CALUDE_inequality_proof_l1288_128857

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a / b + b / c + c / a)^2 ≥ 3 * (a / c + c / b + b / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1288_128857


namespace NUMINAMATH_CALUDE_problem_solution_l1288_128846

theorem problem_solution (A B : ℝ) (h1 : B + A + B = 814.8) (h2 : A = 10 * B) : A - B = 611.1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1288_128846


namespace NUMINAMATH_CALUDE_money_in_pond_is_637_l1288_128862

/-- The amount of money in cents left in the pond after all calculations -/
def moneyInPond : ℕ :=
  let dimeValue : ℕ := 10
  let quarterValue : ℕ := 25
  let halfDollarValue : ℕ := 50
  let dollarValue : ℕ := 100
  let nickelValue : ℕ := 5
  let pennyValue : ℕ := 1
  let foreignCoinValue : ℕ := 25

  let cindyMoney : ℕ := 5 * dimeValue + 3 * halfDollarValue
  let ericMoney : ℕ := 3 * quarterValue + 2 * dollarValue + halfDollarValue
  let garrickMoney : ℕ := 8 * nickelValue + 7 * pennyValue
  let ivyMoney : ℕ := 60 * pennyValue + 5 * foreignCoinValue

  let totalBefore : ℕ := cindyMoney + ericMoney + garrickMoney + ivyMoney

  let beaumontRemoval : ℕ := 2 * dimeValue + 3 * nickelValue + 10 * pennyValue
  let ericRemoval : ℕ := quarterValue + halfDollarValue

  totalBefore - beaumontRemoval - ericRemoval

theorem money_in_pond_is_637 : moneyInPond = 637 := by
  sorry

end NUMINAMATH_CALUDE_money_in_pond_is_637_l1288_128862


namespace NUMINAMATH_CALUDE_vector_problem_l1288_128828

def vector_a : Fin 2 → ℝ := ![3, -4]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![2, x]
def vector_c (y : ℝ) : Fin 2 → ℝ := ![2, y]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i, v i = k * u i

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem vector_problem (x y : ℝ) 
  (h1 : parallel vector_a (vector_b x))
  (h2 : perpendicular vector_a (vector_c y)) :
  (x = -8/3 ∧ y = 3/2) ∧ 
  perpendicular (vector_b (-8/3)) (vector_c (3/2)) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l1288_128828


namespace NUMINAMATH_CALUDE_two_digit_addition_equation_l1288_128818

theorem two_digit_addition_equation (A B : ℕ) : 
  A ≠ B →
  A < 10 →
  B < 10 →
  6 * A + 10 * B + 2 = 77 →
  B = 1 := by sorry

end NUMINAMATH_CALUDE_two_digit_addition_equation_l1288_128818


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1288_128823

/-- Given a line segment with midpoint (-3, 2) and one endpoint (-7, 6), 
    prove that the other endpoint is (1, -2). -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (-3, 2) → endpoint1 = (-7, 6) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1288_128823


namespace NUMINAMATH_CALUDE_yang_hui_field_equation_l1288_128830

theorem yang_hui_field_equation (area : ℕ) (length width : ℕ) :
  area = 650 ∧ width = length - 1 →
  length * (length - 1) = area :=
by sorry

end NUMINAMATH_CALUDE_yang_hui_field_equation_l1288_128830


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l1288_128872

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def ones_digit (n : ℕ) : ℕ := n % 10

def arithmetic_sequence (a b c d : ℕ) : Prop :=
  ∃ (r : ℕ), r > 0 ∧ b = a + r ∧ c = b + r ∧ d = c + r

theorem prime_arithmetic_sequence_ones_digit
  (p q r s : ℕ)
  (h_prime : is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s)
  (h_seq : arithmetic_sequence p q r s)
  (h_p_gt_5 : p > 5)
  (h_diff : ∃ (d : ℕ), d = 10 ∧ q = p + d ∧ r = q + d ∧ s = r + d) :
  ones_digit p = 1 ∨ ones_digit p = 3 ∨ ones_digit p = 7 ∨ ones_digit p = 9 :=
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l1288_128872


namespace NUMINAMATH_CALUDE_base7_divisible_by_19_l1288_128847

/-- Given a digit y, returns the decimal representation of 52y3 in base 7 -/
def base7ToDecimal (y : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + y * 7 + 3

/-- Theorem stating that when 52y3 in base 7 is divisible by 19, y must be 8 -/
theorem base7_divisible_by_19 :
  ∃ y : ℕ, y < 7 ∧ (base7ToDecimal y) % 19 = 0 → y = 8 :=
by sorry

end NUMINAMATH_CALUDE_base7_divisible_by_19_l1288_128847


namespace NUMINAMATH_CALUDE_fraction_sum_integer_l1288_128854

theorem fraction_sum_integer (n : ℕ) (h1 : n > 0) 
  (h2 : ∃ k : ℤ, (1 / 2 : ℚ) + (1 / 3 : ℚ) + (1 / 11 : ℚ) + (1 / n : ℚ) = k) :
  n = 66 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_l1288_128854
