import Mathlib

namespace NUMINAMATH_CALUDE_total_annual_earnings_l2999_299996

def months_in_year : Nat := 12

def orange_harvest_frequency : Nat := 2
def orange_harvest_price : Nat := 50

def apple_harvest_frequency : Nat := 3
def apple_harvest_price : Nat := 30

def peach_harvest_frequency : Nat := 4
def peach_harvest_price : Nat := 45

def blackberry_harvest_frequency : Nat := 6
def blackberry_harvest_price : Nat := 70

def annual_earnings (frequency : Nat) (price : Nat) : Nat :=
  (months_in_year / frequency) * price

theorem total_annual_earnings : 
  annual_earnings orange_harvest_frequency orange_harvest_price +
  annual_earnings apple_harvest_frequency apple_harvest_price +
  annual_earnings peach_harvest_frequency peach_harvest_price +
  annual_earnings blackberry_harvest_frequency blackberry_harvest_price = 695 := by
  sorry

end NUMINAMATH_CALUDE_total_annual_earnings_l2999_299996


namespace NUMINAMATH_CALUDE_apps_files_difference_l2999_299913

/-- Represents the contents of Dave's phone -/
structure PhoneContents where
  apps : ℕ
  files : ℕ

/-- The initial state of Dave's phone -/
def initial : PhoneContents := { apps := 24, files := 9 }

/-- The final state of Dave's phone -/
def final : PhoneContents := { apps := 12, files := 5 }

/-- The theorem stating the difference between apps and files in the final state -/
theorem apps_files_difference : final.apps - final.files = 7 := by
  sorry

end NUMINAMATH_CALUDE_apps_files_difference_l2999_299913


namespace NUMINAMATH_CALUDE_a_4_equals_zero_l2999_299995

def a (n : ℕ+) : ℤ := n^2 - 3*n - 4

theorem a_4_equals_zero : a 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_zero_l2999_299995


namespace NUMINAMATH_CALUDE_fraction_simplification_l2999_299908

theorem fraction_simplification :
  (1 : ℚ) / 330 + 19 / 30 = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2999_299908


namespace NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l2999_299984

-- Define the parallel lines
def line1 (x y : ℝ) : Prop := x + 3 * y - 5 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y - 3 = 0

-- Define the line containing the center of the circle
def centerLine (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the circle equation
def circleEquation (x y : ℝ) : Prop := (x + 13/5)^2 + (y - 11/5)^2 = 1/10

-- Theorem stating the circle equation given the conditions
theorem circle_tangent_to_parallel_lines :
  ∀ (C : Set (ℝ × ℝ)),
  (∃ (x₁ y₁ : ℝ), (x₁, y₁) ∈ C ∧ line1 x₁ y₁) ∧
  (∃ (x₂ y₂ : ℝ), (x₂, y₂) ∈ C ∧ line2 x₂ y₂) ∧
  (∃ (x₀ y₀ : ℝ), (x₀, y₀) ∈ C ∧ centerLine x₀ y₀) →
  ∀ (x y : ℝ), (x, y) ∈ C ↔ circleEquation x y :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l2999_299984


namespace NUMINAMATH_CALUDE_centroid_trajectory_l2999_299948

/-- The trajectory of the centroid of a triangle ABC, where A and B are fixed points
    and C moves on a hyperbola. -/
theorem centroid_trajectory
  (A B C : ℝ × ℝ)  -- Vertices of the triangle
  (x y : ℝ)        -- Coordinates of the centroid
  (h1 : A = (0, 0))
  (h2 : B = (6, 0))
  (h3 : (C.1^2 / 16) - (C.2^2 / 9) = 1)  -- C moves on the hyperbola
  (h4 : x = (A.1 + B.1 + C.1) / 3)       -- Centroid x-coordinate
  (h5 : y = (A.2 + B.2 + C.2) / 3)       -- Centroid y-coordinate
  (h6 : y ≠ 0) :
  9 * (x - 2)^2 / 16 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_centroid_trajectory_l2999_299948


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2999_299963

/-- The sum of an infinite geometric series with first term 1 and common ratio 1/4 is 4/3 -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/4
  let S := ∑' n, a * r^n
  S = 4/3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2999_299963


namespace NUMINAMATH_CALUDE_park_area_l2999_299926

theorem park_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  width > 0 → 
  length > 0 → 
  length = 3 * width → 
  perimeter = 2 * (width + length) → 
  perimeter = 72 → 
  area = width * length → 
  area = 243 := by sorry

end NUMINAMATH_CALUDE_park_area_l2999_299926


namespace NUMINAMATH_CALUDE_antonio_age_is_51_months_l2999_299927

/- Define Isabella's current age in months -/
def isabella_age_months : ℕ := 10 * 12 - 18

/- Define the relationship between Isabella's and Antonio's ages -/
def antonio_age_months : ℕ := isabella_age_months / 2

/- Theorem stating Antonio's age in months -/
theorem antonio_age_is_51_months : antonio_age_months = 51 := by
  sorry

end NUMINAMATH_CALUDE_antonio_age_is_51_months_l2999_299927


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2999_299962

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 7*x + 6 < 0 ↔ -6 < x ∧ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2999_299962


namespace NUMINAMATH_CALUDE_circle_ratio_after_radius_increase_l2999_299949

theorem circle_ratio_after_radius_increase (r : ℝ) : 
  let new_radius : ℝ := r + 2
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_radius_increase_l2999_299949


namespace NUMINAMATH_CALUDE_bacteria_growth_l2999_299943

theorem bacteria_growth (n : ℕ) : (∀ k < n, 4 * 3^k ≤ 500) ∧ 4 * 3^n > 500 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l2999_299943


namespace NUMINAMATH_CALUDE_selling_price_is_50_l2999_299980

/-- Represents the manufacturing and sales data for horseshoe sets -/
structure HorseshoeData where
  initial_outlay : ℕ
  cost_per_set : ℕ
  sets_sold : ℕ
  profit : ℕ
  selling_price : ℕ

/-- Calculates the total manufacturing cost -/
def total_manufacturing_cost (data : HorseshoeData) : ℕ :=
  data.initial_outlay + data.cost_per_set * data.sets_sold

/-- Calculates the total revenue -/
def total_revenue (data : HorseshoeData) : ℕ :=
  data.selling_price * data.sets_sold

/-- Theorem stating that the selling price is $50 given the conditions -/
theorem selling_price_is_50 (data : HorseshoeData) 
    (h1 : data.initial_outlay = 10000)
    (h2 : data.cost_per_set = 20)
    (h3 : data.sets_sold = 500)
    (h4 : data.profit = 5000)
    (h5 : data.profit = total_revenue data - total_manufacturing_cost data) :
  data.selling_price = 50 := by
  sorry


end NUMINAMATH_CALUDE_selling_price_is_50_l2999_299980


namespace NUMINAMATH_CALUDE_range_of_n_minus_m_l2999_299903

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x - 1 else (3/2) * x + 1

theorem range_of_n_minus_m (m n : ℝ) (h1 : m < n) (h2 : f m = f n) :
  2/3 < n - m ∧ n - m ≤ Real.log (3/2) + 1/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_n_minus_m_l2999_299903


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_l2999_299946

theorem infinitely_many_pairs (c : ℝ) : 
  (c > 0) → 
  (∀ k : ℕ, ∃ n m : ℕ, 
    n > 0 ∧ m > 0 ∧
    (n : ℝ) ≥ (m : ℝ) + c * Real.sqrt ((m : ℝ) - 1) + 1 ∧
    ∀ i ∈ Finset.range (2 * n - m - n + 1), ¬ ∃ j : ℕ, (n + i : ℝ) = (j : ℝ) ^ 2) ↔ 
  c ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_l2999_299946


namespace NUMINAMATH_CALUDE_melanie_dimes_l2999_299920

/-- The number of dimes Melanie has after receiving dimes from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Proof that Melanie has 19 dimes after receiving dimes from her parents -/
theorem melanie_dimes : total_dimes 7 8 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l2999_299920


namespace NUMINAMATH_CALUDE_sum_of_a_and_t_is_71_l2999_299941

/-- Given a natural number n, this function represents the equation
    √(n+1 + (n+1)/((n+1)²-1)) = (n+1)√((n+1)/((n+1)²-1)) -/
def equation_pattern (n : ℕ) : Prop :=
  Real.sqrt ((n + 1 : ℝ) + (n + 1) / ((n + 1)^2 - 1)) = (n + 1 : ℝ) * Real.sqrt ((n + 1) / ((n + 1)^2 - 1))

/-- The main theorem stating that given the pattern for n = 1 to 7,
    the sum of a and t in the equation √(8 + a/t) = 8√(a/t) is 71 -/
theorem sum_of_a_and_t_is_71 
  (h1 : equation_pattern 1)
  (h2 : equation_pattern 2)
  (h3 : equation_pattern 3)
  (h4 : equation_pattern 4)
  (h5 : equation_pattern 5)
  (h6 : equation_pattern 6)
  (h7 : equation_pattern 7)
  (a t : ℝ)
  (ha : a > 0)
  (ht : t > 0)
  (h : Real.sqrt (8 + a/t) = 8 * Real.sqrt (a/t)) :
  a + t = 71 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_t_is_71_l2999_299941


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2999_299934

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (1 - 4 * x) = 5 → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2999_299934


namespace NUMINAMATH_CALUDE_willson_work_hours_l2999_299997

theorem willson_work_hours : 
  let monday : ℚ := 3/4
  let tuesday : ℚ := 1/2
  let wednesday : ℚ := 2/3
  let thursday : ℚ := 5/6
  let friday : ℚ := 75/60
  monday + tuesday + wednesday + thursday + friday = 4 := by
sorry

end NUMINAMATH_CALUDE_willson_work_hours_l2999_299997


namespace NUMINAMATH_CALUDE_cube_painting_cost_l2999_299923

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem cube_painting_cost 
  (paint_cost : ℝ) 
  (paint_coverage : ℝ) 
  (cube_side : ℝ) 
  (h1 : paint_cost = 40) 
  (h2 : paint_coverage = 20) 
  (h3 : cube_side = 10) : 
  paint_cost * (6 * cube_side^2 / paint_coverage) = 1200 := by
  sorry

#check cube_painting_cost

end NUMINAMATH_CALUDE_cube_painting_cost_l2999_299923


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_M_l2999_299966

-- Define the sum of divisors function
def sumOfDivisors (n : ℕ) : ℕ := sorry

-- Define M as the sum of divisors of 300
def M : ℕ := sumOfDivisors 300

-- Define a function to get the largest prime factor of a number
def largestPrimeFactor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_M :
  largestPrimeFactor M = 31 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_M_l2999_299966


namespace NUMINAMATH_CALUDE_total_turnips_proof_l2999_299912

/-- The number of turnips grown by Sally -/
def sally_turnips : ℕ := 113

/-- The number of turnips grown by Mary -/
def mary_turnips : ℕ := 129

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := sally_turnips + mary_turnips

theorem total_turnips_proof : total_turnips = 242 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_proof_l2999_299912


namespace NUMINAMATH_CALUDE_system_is_linear_l2999_299956

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A system of two linear equations -/
structure LinearSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system of equations we want to prove is linear -/
def system : LinearSystem := {
  eq1 := { a := 1, b := 0, c := 1 }  -- Represents x = 1
  eq2 := { a := 3, b := -2, c := 6 } -- Represents 3x - 2y = 6
}

/-- Theorem stating that our system is indeed a system of two linear equations -/
theorem system_is_linear : ∃ (s : LinearSystem), s = system := by sorry

end NUMINAMATH_CALUDE_system_is_linear_l2999_299956


namespace NUMINAMATH_CALUDE_sin_240_degrees_l2999_299994

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l2999_299994


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2999_299990

theorem quadratic_equation_solution :
  let f (x : ℝ) := (2*x + 1)^2 - (2*x + 1)*(x - 1)
  ∀ x : ℝ, f x = 0 ↔ x = -1/2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2999_299990


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_cosine_condition_l2999_299902

/-- Given a triangle ABC where a*cos(B) = b*cos(A), prove that the triangle is isosceles -/
theorem isosceles_triangle_from_cosine_condition (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_cosine : a * Real.cos B = b * Real.cos A) : 
  a = b ∨ b = c ∨ a = c :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_cosine_condition_l2999_299902


namespace NUMINAMATH_CALUDE_circle_equation_correct_l2999_299931

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the equation of a circle
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation_correct :
  let c : Circle := { center := (-1, 3), radius := 2 }
  ∀ x y : ℝ, circleEquation c x y ↔ (x + 1)^2 + (y - 3)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l2999_299931


namespace NUMINAMATH_CALUDE_prob_non_white_ball_l2999_299983

/-- The probability of drawing a non-white ball from a bag -/
theorem prob_non_white_ball (white yellow red : ℕ) (h : white = 6 ∧ yellow = 5 ∧ red = 4) :
  (yellow + red) / (white + yellow + red) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_non_white_ball_l2999_299983


namespace NUMINAMATH_CALUDE_base_conversion_proof_l2999_299930

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

theorem base_conversion_proof :
  let base_5_101 := to_base_10 [1, 0, 1] 5
  let base_7_1234 := to_base_10 [4, 3, 2, 1] 7
  let base_9_3456 := to_base_10 [6, 5, 4, 3] 9
  2468 / base_5_101 * base_7_1234 - base_9_3456 = 41708 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_proof_l2999_299930


namespace NUMINAMATH_CALUDE_overlap_difference_l2999_299936

def total_students : ℕ := 232
def geometry_students : ℕ := 144
def biology_students : ℕ := 119

theorem overlap_difference :
  (min geometry_students biology_students) - 
  (geometry_students + biology_students - total_students) = 88 :=
by sorry

end NUMINAMATH_CALUDE_overlap_difference_l2999_299936


namespace NUMINAMATH_CALUDE_inequality_proof_l2999_299924

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z ≥ 1/x + 1/y + 1/z) :
  x/y + y/z + z/x ≥ 1/(x*y) + 1/(y*z) + 1/(z*x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2999_299924


namespace NUMINAMATH_CALUDE_m_greater_equal_nine_l2999_299950

-- Define the conditions p and q
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Define what it means for p to be a sufficient but not necessary condition for q
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, p x → q x m) ∧ ¬(∀ x, q x m → p x)

-- Theorem statement
theorem m_greater_equal_nine (m : ℝ) :
  sufficient_not_necessary m → m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_m_greater_equal_nine_l2999_299950


namespace NUMINAMATH_CALUDE_inequality_theorem_l2999_299957

theorem inequality_theorem (x y : ℝ) 
  (h1 : y ≥ 0) 
  (h2 : y * (y + 1) ≤ (x + 1)^2) 
  (h3 : y * (y - 1) ≤ x^2) : 
  y * (y - 1) ≤ x^2 ∧ y * (y + 1) ≤ (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2999_299957


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2999_299945

theorem polynomial_factorization (x : ℤ) :
  3 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 2 * x^2 =
  (3 * x^2 + 35 * x + 72) * (x + 3) * (x + 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2999_299945


namespace NUMINAMATH_CALUDE_not_prime_5n_plus_3_l2999_299986

theorem not_prime_5n_plus_3 (n : ℕ) (h1 : ∃ a : ℕ, 2 * n + 1 = a ^ 2) (h2 : ∃ b : ℕ, 3 * n + 1 = b ^ 2) : 
  ¬ Nat.Prime (5 * n + 3) := by
sorry

end NUMINAMATH_CALUDE_not_prime_5n_plus_3_l2999_299986


namespace NUMINAMATH_CALUDE_movies_needed_for_even_distribution_movie_store_problem_l2999_299969

theorem movies_needed_for_even_distribution (total_movies : Nat) (num_shelves : Nat) : Nat :=
  let movies_per_shelf := total_movies / num_shelves
  let movies_needed := (movies_per_shelf + 1) * num_shelves - total_movies
  movies_needed

theorem movie_store_problem : movies_needed_for_even_distribution 2763 17 = 155 := by
  sorry

end NUMINAMATH_CALUDE_movies_needed_for_even_distribution_movie_store_problem_l2999_299969


namespace NUMINAMATH_CALUDE_triangle_constructibility_l2999_299991

/-- Given two sides of a triangle and the median to the third side,
    this theorem proves the condition for the triangle's constructibility. -/
theorem triangle_constructibility 
  (a b s : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hs : s > 0) :
  ((a - b) / 2 < s ∧ s < (a + b) / 2) ↔ 
  ∃ (c : ℝ), c > 0 ∧ 
    (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
    s^2 = (2 * (a^2 + b^2) - c^2) / 4 :=
by sorry


end NUMINAMATH_CALUDE_triangle_constructibility_l2999_299991


namespace NUMINAMATH_CALUDE_sector_central_angle_l2999_299973

/-- Given a sector with radius R and circumference 3R, its central angle is 1 radian -/
theorem sector_central_angle (R : ℝ) (h : R > 0) : 
  let circumference := 3 * R
  let arc_length := circumference - 2 * R
  let central_angle := arc_length / R
  central_angle = 1 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2999_299973


namespace NUMINAMATH_CALUDE_garage_wheel_count_l2999_299998

/-- Calculates the total number of wheels in a garage given the quantities of various vehicles --/
def total_wheels (bicycles cars tricycles single_axle_trailers double_axle_trailers eighteen_wheelers : ℕ) : ℕ :=
  bicycles * 2 + cars * 4 + tricycles * 3 + single_axle_trailers * 2 + double_axle_trailers * 4 + eighteen_wheelers * 18

/-- Proves that the total number of wheels in the garage is 97 --/
theorem garage_wheel_count :
  total_wheels 5 12 3 2 2 1 = 97 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheel_count_l2999_299998


namespace NUMINAMATH_CALUDE_difference_has_7_in_thousands_l2999_299979

/-- Given a number with 3 in the ten-thousands place (28943712) and its local value (30000) -/
def local_value_of_3 : ℕ := 30000

/-- The difference between an unknown number and the local value of 3 -/
def difference (x : ℕ) : ℕ := x - local_value_of_3

/-- Check if a number has 7 in the thousands place -/
def has_7_in_thousands (n : ℕ) : Prop :=
  (n / 1000) % 10 = 7

/-- The local value of 7 in the thousands place -/
def local_value_of_7_in_thousands : ℕ := 7000

/-- Theorem: If the difference has 7 in the thousands place, 
    then the local value of 7 in the difference is 7000 -/
theorem difference_has_7_in_thousands (x : ℕ) :
  has_7_in_thousands (difference x) →
  (difference x / 1000) % 10 * 1000 = local_value_of_7_in_thousands :=
by
  sorry

end NUMINAMATH_CALUDE_difference_has_7_in_thousands_l2999_299979


namespace NUMINAMATH_CALUDE_inequality_proof_l2999_299909

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt (a^2 + 1/a) + Real.sqrt (b^2 + 1/b) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2999_299909


namespace NUMINAMATH_CALUDE_gvidon_descendants_l2999_299944

/-- The number of sons King Gvidon had -/
def kings_sons : ℕ := 5

/-- The number of descendants who had exactly 3 sons each -/
def descendants_with_sons : ℕ := 100

/-- The number of sons each fertile descendant had -/
def sons_per_descendant : ℕ := 3

/-- The total number of descendants of King Gvidon -/
def total_descendants : ℕ := kings_sons + descendants_with_sons * sons_per_descendant

theorem gvidon_descendants :
  total_descendants = 305 :=
sorry

end NUMINAMATH_CALUDE_gvidon_descendants_l2999_299944


namespace NUMINAMATH_CALUDE_dot_product_sum_l2999_299999

/-- Given vectors in ℝ², prove that the dot product of (a + b) and c equals 6 -/
theorem dot_product_sum (a b c : ℝ × ℝ) (ha : a = (1, -2)) (hb : b = (3, 4)) (hc : c = (2, -1)) :
  ((a.1 + b.1, a.2 + b.2) • c) = 6 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_sum_l2999_299999


namespace NUMINAMATH_CALUDE_cube_edge_sum_l2999_299906

/-- Given a cube with surface area 486 square centimeters, 
    prove that the sum of the lengths of all its edges is 108 centimeters. -/
theorem cube_edge_sum (surface_area : ℝ) (h : surface_area = 486) : 
  ∃ (edge_length : ℝ), 
    surface_area = 6 * edge_length^2 ∧ 
    12 * edge_length = 108 :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_sum_l2999_299906


namespace NUMINAMATH_CALUDE_digit_sum_equals_78331_l2999_299940

/-- A function that generates all possible natural numbers from a given list of digits,
    where each digit can be used no more than once. -/
def generateNumbers (digits : List Nat) : List Nat :=
  sorry

/-- The sum of all numbers generated from the digits 2, 0, 1, 8. -/
def digitSum : Nat :=
  (generateNumbers [2, 0, 1, 8]).sum

/-- Theorem stating that the sum of all possible natural numbers formed from digits 2, 0, 1, 8,
    where each digit is used no more than once, is equal to 78331. -/
theorem digit_sum_equals_78331 : digitSum = 78331 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_equals_78331_l2999_299940


namespace NUMINAMATH_CALUDE_figure_perimeter_is_33_l2999_299915

/-- The perimeter of a figure composed of a 4x4 square with a 2x1 rectangle protruding from one side -/
def figurePerimeter (unitSquareSideLength : ℝ) : ℝ :=
  let largeSquareSide := 4 * unitSquareSideLength
  let rectangleWidth := 2 * unitSquareSideLength
  let rectangleHeight := unitSquareSideLength
  2 * largeSquareSide + rectangleWidth + rectangleHeight

theorem figure_perimeter_is_33 :
  figurePerimeter 2 = 33 := by
  sorry


end NUMINAMATH_CALUDE_figure_perimeter_is_33_l2999_299915


namespace NUMINAMATH_CALUDE_ball_attendance_l2999_299929

theorem ball_attendance :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by sorry

end NUMINAMATH_CALUDE_ball_attendance_l2999_299929


namespace NUMINAMATH_CALUDE_binary_arithmetic_problem_l2999_299989

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

theorem binary_arithmetic_problem :
  let a := [true, false, true, false, true]  -- 10101₂
  let b := [true, true, false, true, true]   -- 11011₂
  let c := [true, false, true, false]        -- 1010₂
  let result := [false, true, true, false, true, true]  -- 110110₂
  binary_to_nat (nat_to_binary ((binary_to_nat a + binary_to_nat b) - binary_to_nat c)) = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_problem_l2999_299989


namespace NUMINAMATH_CALUDE_matrix_power_equals_fibonacci_l2999_299964

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 2; 1, 1]

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem matrix_power_equals_fibonacci (n : ℕ) :
  A^n = !![fib (2*n + 1), fib (2*n + 2); fib (2*n), fib (2*n + 1)] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_equals_fibonacci_l2999_299964


namespace NUMINAMATH_CALUDE_line_intersects_parabola_at_one_point_l2999_299985

/-- The parabola function -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

/-- The condition for the line to intersect the parabola at one point -/
def intersects_at_one_point (k : ℝ) : Prop :=
  ∃! y : ℝ, parabola y = k

/-- The theorem stating the value of k for which the line intersects the parabola at one point -/
theorem line_intersects_parabola_at_one_point :
  ∃! k : ℝ, intersects_at_one_point k ∧ k = 25/3 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_at_one_point_l2999_299985


namespace NUMINAMATH_CALUDE_fibonacci_periodicity_l2999_299918

-- Define p-arithmetic system
class PArithmetic (p : ℕ) where
  sqrt5_extractable : ∃ x, x^2 = 5
  fermat_little : ∀ a : ℤ, a ≠ 0 → a^(p-1) ≡ 1 [ZMOD p]

-- Define Fibonacci sequence
def fibonacci (v₀ v₁ : ℤ) : ℕ → ℤ
| 0 => v₀
| 1 => v₁
| (n+2) => fibonacci v₀ v₁ n + fibonacci v₀ v₁ (n+1)

-- Theorem statement
theorem fibonacci_periodicity {p : ℕ} [PArithmetic p] (v₀ v₁ : ℤ) :
  ∀ k : ℕ, fibonacci v₀ v₁ (k + p - 1) = fibonacci v₀ v₁ k :=
sorry

end NUMINAMATH_CALUDE_fibonacci_periodicity_l2999_299918


namespace NUMINAMATH_CALUDE_intersection_points_l2999_299904

theorem intersection_points (a : ℝ) : 
  (∃! p : ℝ × ℝ, (p.2 = a * p.1 + a ∧ p.2 = p.1 ∧ p.2 = 2 - 2 * a * p.1)) ↔ 
  (a = 1/2 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_l2999_299904


namespace NUMINAMATH_CALUDE_tan_315_degrees_l2999_299978

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l2999_299978


namespace NUMINAMATH_CALUDE_max_b_squared_l2999_299982

theorem max_b_squared (a b : ℤ) : 
  (a + b)^2 + a*(a + b) + b = 0 → b^2 ≤ 81 :=
by sorry

end NUMINAMATH_CALUDE_max_b_squared_l2999_299982


namespace NUMINAMATH_CALUDE_tj_race_second_half_time_l2999_299993

/-- Represents a race with given parameters -/
structure Race where
  totalDistance : ℝ
  firstHalfTime : ℝ
  averagePace : ℝ

/-- Calculates the time for the second half of the race -/
def secondHalfTime (race : Race) : ℝ :=
  race.averagePace * race.totalDistance - race.firstHalfTime

/-- Theorem stating that for a 10K race with given conditions, 
    the second half time is 30 minutes -/
theorem tj_race_second_half_time :
  let race : Race := {
    totalDistance := 10,
    firstHalfTime := 20,
    averagePace := 5
  }
  secondHalfTime race = 30 := by
  sorry


end NUMINAMATH_CALUDE_tj_race_second_half_time_l2999_299993


namespace NUMINAMATH_CALUDE_x_intercept_thrice_y_intercept_implies_a_eq_neg_two_l2999_299951

/-- A line with equation ax - 6y - 12a = 0 where a ≠ 0 -/
structure Line where
  a : ℝ
  eq : ∀ x y : ℝ, a * x - 6 * y - 12 * a = 0
  a_neq_zero : a ≠ 0

/-- The x-intercept of the line -/
def x_intercept (l : Line) : ℝ := 12

/-- The y-intercept of the line -/
def y_intercept (l : Line) : ℝ := -2 * l.a

/-- Theorem stating that if the x-intercept is three times the y-intercept, then a = -2 -/
theorem x_intercept_thrice_y_intercept_implies_a_eq_neg_two (l : Line) 
  (h : x_intercept l = 3 * y_intercept l) : l.a = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_thrice_y_intercept_implies_a_eq_neg_two_l2999_299951


namespace NUMINAMATH_CALUDE_vectors_parallel_opposite_direction_l2999_299960

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -4)

theorem vectors_parallel_opposite_direction :
  ∃ k : ℝ, k < 0 ∧ b = (k • a.1, k • a.2) :=
sorry

end NUMINAMATH_CALUDE_vectors_parallel_opposite_direction_l2999_299960


namespace NUMINAMATH_CALUDE_x_value_after_z_doubled_l2999_299925

theorem x_value_after_z_doubled (x y z_original z_doubled : ℚ) : 
  x = (1 / 3) * y →
  y = (1 / 4) * z_doubled →
  z_original = 48 →
  z_doubled = 2 * z_original →
  x = 8 := by sorry

end NUMINAMATH_CALUDE_x_value_after_z_doubled_l2999_299925


namespace NUMINAMATH_CALUDE_coloring_scheme_satisfies_conditions_l2999_299917

/-- Represents the three colors used in the coloring scheme. -/
inductive Color
  | White
  | Red
  | Blue

/-- The coloring function that assigns a color to each integral point in the plane. -/
def f : ℤ × ℤ → Color :=
  sorry

/-- Represents an infinite set of integers. -/
def InfiniteSet (s : Set ℤ) : Prop :=
  ∀ n : ℤ, ∃ m ∈ s, m > n

theorem coloring_scheme_satisfies_conditions :
  (∀ c : Color, InfiniteSet {k : ℤ | InfiniteSet {n : ℤ | f (n, k) = c}}) ∧
  (∀ a b c : ℤ × ℤ,
    f a = Color.White → f b = Color.Red → f c = Color.Blue →
    ∃ d : ℤ × ℤ, f d = Color.Red ∧ d = (a.1 + c.1 - b.1, a.2 + c.2 - b.2)) :=
by
  sorry

end NUMINAMATH_CALUDE_coloring_scheme_satisfies_conditions_l2999_299917


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2999_299992

/-- Given an arithmetic sequence {a_n} with the following properties:
  1) a_4 = 7
  2) a_3 + a_6 = 16
  3) a_n = 31
  This theorem states that n = 16 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ k m : ℕ, a (k + m) - a k = m * (a 2 - a 1))  -- arithmetic sequence property
  (h2 : a 4 = 7)
  (h3 : a 3 + a 6 = 16)
  (h4 : a n = 31) :
  n = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2999_299992


namespace NUMINAMATH_CALUDE_cat_toy_cost_l2999_299914

theorem cat_toy_cost (total_cost cage_cost : ℚ) (h1 : total_cost = 21.95) (h2 : cage_cost = 11.73) :
  total_cost - cage_cost = 10.22 := by
  sorry

end NUMINAMATH_CALUDE_cat_toy_cost_l2999_299914


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l2999_299916

-- Define the nonagon's interior angle
def nonagon_interior_angle : ℝ := 140

-- Define the nonagon's exterior angle
def nonagon_exterior_angle : ℝ := 360 - nonagon_interior_angle

-- Define the square's interior angle
def square_interior_angle : ℝ := 90

-- Theorem statement
theorem exterior_angle_measure :
  nonagon_exterior_angle - square_interior_angle = 130 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l2999_299916


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2999_299900

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (6 * x₁^2 + 5 * x₁ - 4 = 0) → 
  (6 * x₂^2 + 5 * x₂ - 4 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 73/36) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2999_299900


namespace NUMINAMATH_CALUDE_min_price_with_profit_margin_l2999_299935

theorem min_price_with_profit_margin (marked_price : ℝ) (markup_percentage : ℝ) (min_profit_margin : ℝ) : 
  marked_price = 240 →
  markup_percentage = 0.6 →
  min_profit_margin = 0.1 →
  let cost_price := marked_price / (1 + markup_percentage)
  let min_reduced_price := cost_price * (1 + min_profit_margin)
  min_reduced_price = 165 :=
by sorry

end NUMINAMATH_CALUDE_min_price_with_profit_margin_l2999_299935


namespace NUMINAMATH_CALUDE_log_50_between_consecutive_integers_l2999_299977

theorem log_50_between_consecutive_integers :
  ∃ (a b : ℤ), (a + 1 = b) ∧ (a < Real.log 50 / Real.log 10) ∧ (Real.log 50 / Real.log 10 < b) ∧ (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_log_50_between_consecutive_integers_l2999_299977


namespace NUMINAMATH_CALUDE_three_heads_in_a_row_probability_l2999_299942

def coin_flips : ℕ := 6

def favorable_outcomes : ℕ := 12

def total_outcomes : ℕ := 2^coin_flips

def probability : ℚ := favorable_outcomes / total_outcomes

theorem three_heads_in_a_row_probability :
  probability = 3/16 := by sorry

end NUMINAMATH_CALUDE_three_heads_in_a_row_probability_l2999_299942


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2999_299947

/-- Given a hyperbola with equation x²/a² - y²/(4a-2) = 1 and eccentricity √3, prove that a = 1 -/
theorem hyperbola_eccentricity (a : ℝ) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / (4*a - 2) = 1) →
  (∃ b : ℝ, b^2 = 4*a - 2 ∧ b^2 / a^2 = 2) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2999_299947


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2999_299932

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := 3 * a * b - a + b

-- Theorem statement
theorem diamond_equation_solution :
  ∃ x : ℝ, diamond 3 x = 24 ∧ x = 2.7 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2999_299932


namespace NUMINAMATH_CALUDE_pet_shop_dogs_l2999_299933

theorem pet_shop_dogs (dogs cats bunnies : ℕ) : 
  dogs + cats + bunnies > 0 →
  dogs * 7 = cats * 4 →
  dogs * 9 = bunnies * 4 →
  dogs + bunnies = 364 →
  dogs = 112 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_l2999_299933


namespace NUMINAMATH_CALUDE_shark_sightings_problem_l2999_299953

/-- Shark sightings problem -/
theorem shark_sightings_problem 
  (daytona : ℕ) 
  (cape_may long_beach santa_cruz : ℕ) :
  daytona = 26 ∧
  daytona = 3 * cape_may + 5 ∧
  long_beach = 2 * cape_may ∧
  long_beach = daytona - 4 ∧
  santa_cruz = cape_may + long_beach + 3 ∧
  santa_cruz = daytona - 9 →
  cape_may = 7 ∧ long_beach = 22 ∧ santa_cruz = 32 := by
sorry

end NUMINAMATH_CALUDE_shark_sightings_problem_l2999_299953


namespace NUMINAMATH_CALUDE_f_properties_l2999_299974

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem f_properties :
  (∃ (x : ℝ), f x = 0 ∧ x = 1) ∧
  (f 0 * f 2 > 0) ∧
  (∃ (x : ℝ), x > 0 ∧ x < 2 ∧ f x = 0) ∧
  (¬ ∀ (x y : ℝ), x < y ∧ y < 0 → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2999_299974


namespace NUMINAMATH_CALUDE_not_concurrent_deduction_l2999_299952

/-- Represents a method of direct proof -/
inductive ProofMethod
| Synthetic
| Analytic

/-- Represents the direction of deduction in a proof method -/
inductive DeductionDirection
| CauseToEffect
| EffectToCause

/-- Maps a proof method to its deduction direction -/
def methodDirection (m : ProofMethod) : DeductionDirection :=
  match m with
  | ProofMethod.Synthetic => DeductionDirection.CauseToEffect
  | ProofMethod.Analytic => DeductionDirection.EffectToCause

/-- Theorem stating that synthetic and analytic methods do not concurrently deduce cause and effect -/
theorem not_concurrent_deduction :
  ∀ (m : ProofMethod), methodDirection m ≠ DeductionDirection.CauseToEffect ∨
                       methodDirection m ≠ DeductionDirection.EffectToCause :=
by
  sorry


end NUMINAMATH_CALUDE_not_concurrent_deduction_l2999_299952


namespace NUMINAMATH_CALUDE_x_value_l2999_299981

theorem x_value (x : ℝ) (h : x ∈ ({1, x^2} : Set ℝ)) : x = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2999_299981


namespace NUMINAMATH_CALUDE_safe_lock_configuration_l2999_299971

/-- The number of commission members -/
def n : ℕ := 9

/-- The minimum number of members required to access the safe -/
def k : ℕ := 6

/-- The number of keys for each lock -/
def keys_per_lock : ℕ := n - k + 1

/-- The number of locks needed for the safe -/
def num_locks : ℕ := Nat.choose n (n - k + 1)

theorem safe_lock_configuration :
  num_locks = 126 ∧ keys_per_lock = 4 :=
sorry

end NUMINAMATH_CALUDE_safe_lock_configuration_l2999_299971


namespace NUMINAMATH_CALUDE_inverse_of_inverse_f_l2999_299955

-- Define the original function f
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the inverse of f^(-1)(x+1)
def g (x : ℝ) : ℝ := 2 * x + 2

-- Theorem statement
theorem inverse_of_inverse_f (x : ℝ) : 
  g (f⁻¹ (x + 1)) = x ∧ f⁻¹ (g x + 1) = x := by
  sorry


end NUMINAMATH_CALUDE_inverse_of_inverse_f_l2999_299955


namespace NUMINAMATH_CALUDE_frieda_prob_reach_edge_l2999_299910

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 4)

/-- Defines the center position -/
def center : Position := ⟨1, 1⟩

/-- Checks if a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.row = 0 || p.row = 3 || p.col = 0 || p.col = 3

/-- Defines the possible moves -/
inductive Move
  | up
  | down
  | left
  | right

/-- Applies a move to a position -/
def applyMove (p : Position) (m : Move) : Position :=
  match m with
  | Move.up    => ⟨(p.row + 1) % 4, p.col⟩
  | Move.down  => ⟨(p.row - 1 + 4) % 4, p.col⟩
  | Move.left  => ⟨p.row, (p.col - 1 + 4) % 4⟩
  | Move.right => ⟨p.row, (p.col + 1) % 4⟩

/-- Calculates the probability of reaching an edge within n hops -/
def probReachEdge (n : Nat) : ℚ :=
  sorry

theorem frieda_prob_reach_edge :
  probReachEdge 3 = 5/8 :=
sorry

end NUMINAMATH_CALUDE_frieda_prob_reach_edge_l2999_299910


namespace NUMINAMATH_CALUDE_employee_pay_solution_exists_and_unique_l2999_299958

/-- Represents the weekly pay of employees X, Y, and Z -/
structure EmployeePay where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Conditions for the employee pay problem -/
def satisfiesConditions (pay : EmployeePay) : Prop :=
  pay.x = 1.2 * pay.y ∧
  pay.z = 0.75 * pay.x ∧
  pay.x + pay.y + pay.z = 1540

/-- Theorem stating the existence and uniqueness of the solution -/
theorem employee_pay_solution_exists_and_unique :
  ∃! pay : EmployeePay, satisfiesConditions pay :=
sorry

end NUMINAMATH_CALUDE_employee_pay_solution_exists_and_unique_l2999_299958


namespace NUMINAMATH_CALUDE_equation_simplification_l2999_299907

theorem equation_simplification :
  (Real.sqrt ((7 : ℝ)^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end NUMINAMATH_CALUDE_equation_simplification_l2999_299907


namespace NUMINAMATH_CALUDE_hyperbola_n_range_l2999_299968

-- Define the hyperbola equation
def hyperbola_equation (x y m n : ℝ) : Prop :=
  x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

-- Define the distance between foci
def foci_distance : ℝ := 4

-- Theorem statement
theorem hyperbola_n_range (x y m n : ℝ) :
  hyperbola_equation x y m n ∧ 
  (∃ (a b : ℝ), (a - b)^2 = foci_distance^2) →
  -1 < n ∧ n < 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_n_range_l2999_299968


namespace NUMINAMATH_CALUDE_solve_equation_l2999_299961

theorem solve_equation (t x : ℝ) (h1 : (5 + x) / (t + x) = 2 / 3) (h2 : t = 13) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2999_299961


namespace NUMINAMATH_CALUDE_line_AB_parallel_to_xOz_plane_l2999_299911

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
def Vector3D : Type := ℝ × ℝ × ℝ

/-- Calculate the vector from point A to point B -/
def vectorBetweenPoints (A B : Point3D) : Vector3D :=
  (B.x - A.x, B.y - A.y, B.z - A.z)

/-- Check if a vector is parallel to the xOz plane -/
def isParallelToXOZ (v : Vector3D) : Prop :=
  v.2 = 0

/-- The main theorem: Line AB is parallel to xOz plane -/
theorem line_AB_parallel_to_xOz_plane :
  let A : Point3D := ⟨1, 3, 0⟩
  let B : Point3D := ⟨0, 3, -1⟩
  let AB : Vector3D := vectorBetweenPoints A B
  isParallelToXOZ AB := by sorry

end NUMINAMATH_CALUDE_line_AB_parallel_to_xOz_plane_l2999_299911


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2999_299988

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 / ((2*x + y) * (2*x + z)) + y^2 / ((2*y + x) * (2*y + z)) + z^2 / ((2*z + x) * (2*z + y)) ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2999_299988


namespace NUMINAMATH_CALUDE_binary_1101110_equals_3131_base4_l2999_299975

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldr (λ b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The binary representation of 1101110 -/
def binary_1101110 : List Bool := [true, true, false, true, true, true, false]

theorem binary_1101110_equals_3131_base4 :
  decimal_to_base4 (binary_to_decimal binary_1101110) = [3, 1, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_1101110_equals_3131_base4_l2999_299975


namespace NUMINAMATH_CALUDE_marbles_distribution_l2999_299919

theorem marbles_distribution (total_marbles : ℕ) (kept_marbles : ℕ) (best_friends : ℕ) (marbles_per_best_friend : ℕ) (neighborhood_friends : ℕ) :
  total_marbles = 1125 →
  kept_marbles = 100 →
  best_friends = 2 →
  marbles_per_best_friend = 50 →
  neighborhood_friends = 7 →
  (total_marbles - kept_marbles - best_friends * marbles_per_best_friend) / neighborhood_friends = 132 :=
by sorry

end NUMINAMATH_CALUDE_marbles_distribution_l2999_299919


namespace NUMINAMATH_CALUDE_flight_passenger_distribution_l2999_299939

/-- Proof of the flight passenger distribution problem -/
theorem flight_passenger_distribution
  (total_passengers : ℕ)
  (female_percentage : ℚ)
  (first_class_male_ratio : ℚ)
  (coach_females : ℕ)
  (h1 : total_passengers = 120)
  (h2 : female_percentage = 30 / 100)
  (h3 : first_class_male_ratio = 1 / 3)
  (h4 : coach_females = 28)
  : ∃ (first_class_percentage : ℚ), first_class_percentage = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_flight_passenger_distribution_l2999_299939


namespace NUMINAMATH_CALUDE_vector_equation_solution_l2999_299972

theorem vector_equation_solution :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![1, -2]
  ∀ m n : ℝ, (m • a + n • b = ![9, -8]) → (m - n = -3) := by
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l2999_299972


namespace NUMINAMATH_CALUDE_inequality_solution_l2999_299967

theorem inequality_solution (x : ℝ) : 
  (x - 2) / (x - 1) > (4 * x - 1) / (3 * x + 8) ↔ 
  (x > -3 ∧ x < -2) ∨ (x > -8/3 ∧ x < 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2999_299967


namespace NUMINAMATH_CALUDE_smallest_numbers_with_special_property_l2999_299954

theorem smallest_numbers_with_special_property :
  ∃ (a b : ℕ), a > b ∧ 
    (∃ (k : ℕ), a^2 - b^2 = k^3) ∧
    (∃ (m : ℕ), a^3 - b^3 = m^2) ∧
    (∀ (x y : ℕ), x > y → 
      (∃ (k : ℕ), x^2 - y^2 = k^3) →
      (∃ (m : ℕ), x^3 - y^3 = m^2) →
      (x > a ∨ (x = a ∧ y ≥ b))) ∧
    a = 10 ∧ b = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_numbers_with_special_property_l2999_299954


namespace NUMINAMATH_CALUDE_event_probability_l2999_299928

theorem event_probability (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (1 - (1 - p)^3 = 63/64) →
  (3 * p * (1 - p)^2 = 9/64) :=
by
  sorry

end NUMINAMATH_CALUDE_event_probability_l2999_299928


namespace NUMINAMATH_CALUDE_divisibility_property_l2999_299921

theorem divisibility_property (a b : ℤ) : (7 ∣ a^2 + b^2) → (7 ∣ a) ∧ (7 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2999_299921


namespace NUMINAMATH_CALUDE_ipod_ratio_l2999_299922

-- Define the initial number of iPods Emmy has
def emmy_initial : ℕ := 14

-- Define the number of iPods Emmy loses
def emmy_lost : ℕ := 6

-- Define the total number of iPods Emmy and Rosa have together
def total_ipods : ℕ := 12

-- Define Emmy's remaining iPods
def emmy_remaining : ℕ := emmy_initial - emmy_lost

-- Define Rosa's iPods
def rosa_ipods : ℕ := total_ipods - emmy_remaining

-- Theorem statement
theorem ipod_ratio : 
  emmy_remaining * 1 = rosa_ipods * 2 := by
  sorry

end NUMINAMATH_CALUDE_ipod_ratio_l2999_299922


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2999_299937

theorem arithmetic_equality : 3889 + 12.808 - 47.80600000000004 = 3854.002 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2999_299937


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2999_299959

def set_A : Set ℝ := {x | x^2 - 2*x ≤ 0}
def set_B : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2999_299959


namespace NUMINAMATH_CALUDE_andy_max_demerits_l2999_299987

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def maxDemerits : ℕ := by sorry

/-- The number of demerits Andy gets per instance of showing up late -/
def demeritsPerLateInstance : ℕ := 2

/-- The number of times Andy showed up late -/
def lateInstances : ℕ := 6

/-- The number of demerits Andy got for making an inappropriate joke -/
def demeritsForJoke : ℕ := 15

/-- The number of additional demerits Andy can get before getting fired -/
def remainingDemerits : ℕ := 23

theorem andy_max_demerits :
  maxDemerits = demeritsPerLateInstance * lateInstances + demeritsForJoke + remainingDemerits := by
  sorry

end NUMINAMATH_CALUDE_andy_max_demerits_l2999_299987


namespace NUMINAMATH_CALUDE_polyhedron_inequalities_l2999_299905

/-- A simply connected polyhedron -/
structure SimplyConnectedPolyhedron where
  B : ℕ  -- number of vertices
  P : ℕ  -- number of edges
  G : ℕ  -- number of faces
  euler : B - P + G = 2  -- Euler's formula
  edge_face : P ≥ 3 * G / 2  -- each face has at least 3 edges, each edge is shared by 2 faces
  edge_vertex : P ≥ 3 * B / 2  -- each vertex is connected to at least 3 edges

/-- Theorem stating the inequalities for a simply connected polyhedron -/
theorem polyhedron_inequalities (poly : SimplyConnectedPolyhedron) :
  (3 / 2 : ℝ) ≤ (poly.P : ℝ) / poly.B ∧ (poly.P : ℝ) / poly.B < 3 ∧
  (3 / 2 : ℝ) ≤ (poly.P : ℝ) / poly.G ∧ (poly.P : ℝ) / poly.G < 3 :=
by sorry

end NUMINAMATH_CALUDE_polyhedron_inequalities_l2999_299905


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l2999_299938

theorem min_value_of_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (2 / x + 1 / y) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l2999_299938


namespace NUMINAMATH_CALUDE_pi_approximation_l2999_299970

theorem pi_approximation (π : Real) (h : π = 4 * Real.sin (52 * π / 180)) :
  (1 - 2 * (Real.cos (7 * π / 180))^2) / (π * Real.sqrt (16 - π^2)) = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_pi_approximation_l2999_299970


namespace NUMINAMATH_CALUDE_original_car_cost_l2999_299976

/-- Proves that the original cost of a car is 39200 given the specified conditions -/
theorem original_car_cost (C : ℝ) : 
  C > 0 →  -- Ensure the cost is positive
  (68400 - (C + 8000)) / C * 100 = 54.054054054054056 →
  C = 39200 := by
  sorry

end NUMINAMATH_CALUDE_original_car_cost_l2999_299976


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l2999_299965

theorem smallest_sum_of_squares (x y : ℕ) : x^2 - y^2 = 221 → x^2 + y^2 ≥ 229 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l2999_299965


namespace NUMINAMATH_CALUDE_bird_migration_difference_l2999_299901

theorem bird_migration_difference (migrating_families : ℕ) (remaining_families : ℕ)
  (avg_birds_migrating : ℕ) (avg_birds_remaining : ℕ)
  (h1 : migrating_families = 86)
  (h2 : remaining_families = 45)
  (h3 : avg_birds_migrating = 12)
  (h4 : avg_birds_remaining = 8) :
  migrating_families * avg_birds_migrating - remaining_families * avg_birds_remaining = 672 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_difference_l2999_299901
