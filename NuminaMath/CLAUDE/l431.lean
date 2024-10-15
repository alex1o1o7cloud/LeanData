import Mathlib

namespace NUMINAMATH_CALUDE_specific_hexagon_perimeter_l431_43192

/-- A hexagon with specific side lengths and right angles -/
structure RightAngleHexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  FA : ℝ
  right_angles : Bool

/-- The perimeter of a hexagon -/
def perimeter (h : RightAngleHexagon) : ℝ :=
  h.AB + h.BC + h.CD + h.DE + h.EF + h.FA

/-- Theorem: The perimeter of the specific hexagon is 6 -/
theorem specific_hexagon_perimeter :
  ∃ (h : RightAngleHexagon),
    h.AB = 1 ∧ h.BC = 1 ∧ h.CD = 2 ∧ h.DE = 1 ∧ h.EF = 1 ∧ h.right_angles = true ∧
    perimeter h = 6 := by
  sorry

end NUMINAMATH_CALUDE_specific_hexagon_perimeter_l431_43192


namespace NUMINAMATH_CALUDE_divisible_by_ten_l431_43150

theorem divisible_by_ten : ∃ k : ℕ, 11^11 + 12^12 + 13^13 = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_ten_l431_43150


namespace NUMINAMATH_CALUDE_amicable_pairs_l431_43173

/-- Sum of proper divisors of a natural number -/
def sumProperDivisors (n : ℕ) : ℕ := sorry

/-- Two numbers are amicable if the sum of proper divisors of each equals the other -/
def areAmicable (a b : ℕ) : Prop :=
  sumProperDivisors a = b ∧ sumProperDivisors b = a

theorem amicable_pairs :
  (areAmicable 284 220) ∧ (areAmicable 76084 63020) := by sorry

end NUMINAMATH_CALUDE_amicable_pairs_l431_43173


namespace NUMINAMATH_CALUDE_hospital_bed_charge_l431_43136

theorem hospital_bed_charge 
  (days_in_hospital : ℕ) 
  (specialist_hourly_rate : ℚ) 
  (specialist_time : ℚ) 
  (num_specialists : ℕ) 
  (ambulance_cost : ℚ) 
  (total_bill : ℚ) :
  days_in_hospital = 3 →
  specialist_hourly_rate = 250 →
  specialist_time = 1/4 →
  num_specialists = 2 →
  ambulance_cost = 1800 →
  total_bill = 4625 →
  let daily_bed_charge := (total_bill - num_specialists * specialist_hourly_rate * specialist_time - ambulance_cost) / days_in_hospital
  daily_bed_charge = 900 := by
sorry

end NUMINAMATH_CALUDE_hospital_bed_charge_l431_43136


namespace NUMINAMATH_CALUDE_square_of_sum_85_7_l431_43154

theorem square_of_sum_85_7 : (85 + 7)^2 = 8464 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_85_7_l431_43154


namespace NUMINAMATH_CALUDE_james_tennis_balls_l431_43155

/-- Given that James buys 100 tennis balls, gives half away, and distributes the remaining balls 
    equally among 5 containers, prove that each container will have 10 tennis balls. -/
theorem james_tennis_balls (total_balls : ℕ) (containers : ℕ) : 
  total_balls = 100 → 
  containers = 5 → 
  (total_balls / 2) / containers = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_tennis_balls_l431_43155


namespace NUMINAMATH_CALUDE_potato_price_proof_l431_43148

/-- The original price of one bag of potatoes in rubles -/
def original_price : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase percentage -/
def andrey_increase : ℝ := 100

/-- Boris's first price increase percentage -/
def boris_first_increase : ℝ := 60

/-- Boris's second price increase percentage -/
def boris_second_increase : ℝ := 40

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey in rubles -/
def earnings_difference : ℝ := 1200

theorem potato_price_proof :
  let andrey_earnings := bags_bought * original_price * (1 + andrey_increase / 100)
  let boris_first_earnings := boris_first_sale * original_price * (1 + boris_first_increase / 100)
  let boris_second_earnings := boris_second_sale * original_price * (1 + boris_first_increase / 100) * (1 + boris_second_increase / 100)
  boris_first_earnings + boris_second_earnings - andrey_earnings = earnings_difference :=
by sorry

end NUMINAMATH_CALUDE_potato_price_proof_l431_43148


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l431_43158

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the centers of the circles
def center_C1 : ℝ × ℝ := (0, 0)
def center_C2 : ℝ × ℝ := (2, 0)

-- Define the radii of the circles
def radius_C1 : ℝ := 1
def radius_C2 : ℝ := 1

-- Define the distance between centers
def distance_between_centers : ℝ := 2

-- Theorem: The circles are externally tangent
theorem circles_externally_tangent :
  distance_between_centers = radius_C1 + radius_C2 :=
by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l431_43158


namespace NUMINAMATH_CALUDE_base_conversion_problem_l431_43159

/-- Convert a number from base 6 to base 10 -/
def base6To10 (n : Nat) : Nat :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Check if a number is a valid base-10 digit -/
def isBase10Digit (n : Nat) : Prop := n < 10

theorem base_conversion_problem :
  ∀ c d : Nat,
  isBase10Digit c →
  isBase10Digit d →
  base6To10 524 = 2 * (10 * c + d) →
  (c * d : ℚ) / 12 = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l431_43159


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l431_43196

/-- Given a normal distribution with mean 51 and 3 standard deviations below the mean greater than 44,
    prove that the standard deviation is less than 2.33 -/
theorem normal_distribution_std_dev (σ : ℝ) (h1 : 51 - 3 * σ > 44) : σ < 2.33 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l431_43196


namespace NUMINAMATH_CALUDE_chess_board_pawn_placement_l431_43113

theorem chess_board_pawn_placement :
  let board_size : ℕ := 5
  let num_pawns : ℕ := 5
  let ways_to_place_in_rows : ℕ := Nat.factorial board_size
  let ways_to_arrange_pawns : ℕ := Nat.factorial num_pawns
  ways_to_place_in_rows * ways_to_arrange_pawns = 14400 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_board_pawn_placement_l431_43113


namespace NUMINAMATH_CALUDE_calculate_expression_l431_43195

theorem calculate_expression : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l431_43195


namespace NUMINAMATH_CALUDE_weight_of_packet_a_l431_43140

theorem weight_of_packet_a (a b c d e f : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 3 →
  (b + c + d + e) / 4 = 79 →
  f = (a + e) / 2 →
  (b + c + d + e + f) / 5 = 81 →
  a = 75 := by
sorry

end NUMINAMATH_CALUDE_weight_of_packet_a_l431_43140


namespace NUMINAMATH_CALUDE_sin_225_degrees_l431_43123

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l431_43123


namespace NUMINAMATH_CALUDE_rachel_homework_pages_l431_43147

/-- The number of pages of math homework Rachel has to complete -/
def math_homework : ℕ := 8

/-- The number of pages of biology homework Rachel has to complete -/
def biology_homework : ℕ := 3

/-- The total number of pages of math and biology homework Rachel has to complete -/
def total_homework : ℕ := math_homework + biology_homework

theorem rachel_homework_pages :
  total_homework = 11 :=
by sorry

end NUMINAMATH_CALUDE_rachel_homework_pages_l431_43147


namespace NUMINAMATH_CALUDE_seventh_observation_value_l431_43108

theorem seventh_observation_value
  (n : ℕ) -- number of initial observations
  (initial_avg : ℚ) -- initial average
  (new_avg : ℚ) -- new average after adding one observation
  (h1 : n = 6) -- there are 6 initial observations
  (h2 : initial_avg = 16) -- the initial average is 16
  (h3 : new_avg = initial_avg - 1) -- the new average is decreased by 1
  : (n + 1) * new_avg - n * initial_avg = 9 := by
  sorry

end NUMINAMATH_CALUDE_seventh_observation_value_l431_43108


namespace NUMINAMATH_CALUDE_solution_equality_l431_43179

-- Define the function F
def F (a b c : ℚ) : ℚ := a * b^3 + c

-- Theorem statement
theorem solution_equality :
  ∃ a : ℚ, F a 3 4 = F a 5 8 ∧ a = -2/49 := by sorry

end NUMINAMATH_CALUDE_solution_equality_l431_43179


namespace NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l431_43104

/-- Represents a lottery with a total number of tickets and a winning rate. -/
structure Lottery where
  totalTickets : ℕ
  winningRate : ℝ
  winningRate_pos : winningRate > 0
  winningRate_le_one : winningRate ≤ 1

/-- The probability of not winning with a single ticket. -/
def Lottery.loseProb (l : Lottery) : ℝ := 1 - l.winningRate

/-- The probability of not winning with n tickets. -/
def Lottery.loseProbN (l : Lottery) (n : ℕ) : ℝ := (l.loseProb) ^ n

theorem lottery_not_guaranteed_win (l : Lottery) (h1 : l.totalTickets = 1000000) (h2 : l.winningRate = 0.001) :
  l.loseProbN 1000 > 0 := by sorry

end NUMINAMATH_CALUDE_lottery_not_guaranteed_win_l431_43104


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l431_43181

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents the sample size for each age group -/
structure SampleSize where
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Checks if the sample sizes are proportional to the population sizes -/
def isProportionalSample (employees : EmployeeCount) (sample : SampleSize) (totalSampleSize : ℕ) : Prop :=
  sample.young * employees.total = employees.young * totalSampleSize ∧
  sample.middleAged * employees.total = employees.middleAged * totalSampleSize ∧
  sample.elderly * employees.total = employees.elderly * totalSampleSize

/-- The main theorem to prove -/
theorem stratified_sampling_theorem (employees : EmployeeCount) (sample : SampleSize) :
  employees.total = 750 →
  employees.young = 350 →
  employees.middleAged = 250 →
  employees.elderly = 150 →
  sample.young = 7 →
  sample.middleAged = 5 →
  sample.elderly = 3 →
  isProportionalSample employees sample 15 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l431_43181


namespace NUMINAMATH_CALUDE_complex_equation_solution_l431_43188

theorem complex_equation_solution : ∃ (z : ℂ), (Complex.I + 1) * z = Complex.abs (2 * Complex.I) ∧ z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l431_43188


namespace NUMINAMATH_CALUDE_waiter_problem_l431_43178

/-- Given a waiter's section with initial customers, some leaving customers, and a number of tables,
    calculate the number of people at each table after the customers left. -/
def people_per_table (initial_customers leaving_customers tables : ℕ) : ℕ :=
  (initial_customers - leaving_customers) / tables

/-- Theorem stating that with 44 initial customers, 12 leaving customers, and 4 tables,
    the number of people at each table after the customers left is 8. -/
theorem waiter_problem :
  people_per_table 44 12 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_waiter_problem_l431_43178


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_120_perfect_square_l431_43142

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem smallest_multiplier_for_120_perfect_square :
  ∃! n : ℕ, n > 0 ∧ is_perfect_square (120 * n) ∧ 
  ∀ m : ℕ, m > 0 → is_perfect_square (120 * m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_120_perfect_square_l431_43142


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l431_43193

/-- A rectangle with perimeter 60 feet and area 130 square feet has a shorter side of approximately 5 feet -/
theorem rectangle_shorter_side (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≥ b)
  (h_perimeter : 2*a + 2*b = 60) (h_area : a*b = 130) :
  ∃ ε > 0, abs (b - 5) < ε :=
sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l431_43193


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l431_43197

def z : ℂ := Complex.I + Complex.I^2

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l431_43197


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l431_43151

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 12) = 10 → x = 88 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l431_43151


namespace NUMINAMATH_CALUDE_largest_number_problem_l431_43132

theorem largest_number_problem (a b c d e : ℝ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →
  a + b = 32 →
  a + c = 36 →
  b + c = 37 →
  c + e = 48 →
  d + e = 51 →
  e = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_problem_l431_43132


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l431_43107

theorem simultaneous_equations_solution (n : ℕ+) (u v : ℝ) :
  (∃ (a b c : ℕ+),
    (a^2 + b^2 + c^2 : ℝ) = 169 * (n : ℝ)^2 ∧
    (a^2 * (u * a^2 + v * b^2) + b^2 * (u * b^2 + v * c^2) + c^2 * (u * c^2 + v * a^2) : ℝ) = 
      ((2 * u + v) * (13 * (n : ℝ))^4) / 4) ↔
  v = 2 * u :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l431_43107


namespace NUMINAMATH_CALUDE_cost_price_calculation_l431_43100

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 150 ∧ profit_percentage = 25 →
  ∃ (cost_price : ℝ), cost_price = 120 ∧
    selling_price = cost_price * (1 + profit_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l431_43100


namespace NUMINAMATH_CALUDE_problem_statement_l431_43121

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, x^2 + (a-1)*x + a^2 > 0

def q : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (2*a^2 - a)^x₁ < (2*a^2 - a)^x₂

theorem problem_statement : (p a ∨ q a) → (a < -1/2 ∨ a > 1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l431_43121


namespace NUMINAMATH_CALUDE_least_number_of_cans_l431_43110

def maaza_liters : ℕ := 157
def pepsi_liters : ℕ := 173
def sprite_liters : ℕ := 389

def total_cans : ℕ := maaza_liters + pepsi_liters + sprite_liters

theorem least_number_of_cans :
  ∀ (can_size : ℕ),
    can_size > 0 →
    can_size ∣ maaza_liters →
    can_size ∣ pepsi_liters →
    can_size ∣ sprite_liters →
    total_cans ≤ maaza_liters / can_size + pepsi_liters / can_size + sprite_liters / can_size :=
by sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l431_43110


namespace NUMINAMATH_CALUDE_pet_store_problem_l431_43122

/-- The number of ways to distribute pets among Alice, Bob, and Charlie -/
def pet_distribution_ways (num_puppies num_kittens num_hamsters : ℕ) : ℕ :=
  num_kittens * num_hamsters + num_hamsters * num_kittens

/-- Theorem stating the number of ways Alice, Bob, and Charlie can buy pets -/
theorem pet_store_problem :
  let num_puppies : ℕ := 20
  let num_kittens : ℕ := 4
  let num_hamsters : ℕ := 8
  pet_distribution_ways num_puppies num_kittens num_hamsters = 64 :=
by
  sorry

#eval pet_distribution_ways 20 4 8

end NUMINAMATH_CALUDE_pet_store_problem_l431_43122


namespace NUMINAMATH_CALUDE_selling_price_with_loss_l431_43103

theorem selling_price_with_loss (cost_price : ℝ) (loss_percent : ℝ) (selling_price : ℝ) :
  cost_price = 600 →
  loss_percent = 8.333333333333329 →
  selling_price = cost_price * (1 - loss_percent / 100) →
  selling_price = 550 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_with_loss_l431_43103


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_prime_l431_43153

theorem infinitely_many_divisible_by_prime (p : Nat) (hp : Prime p) :
  ∃ f : ℕ → ℕ, ∀ k : ℕ, p ∣ (2^(f k) - f k) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_prime_l431_43153


namespace NUMINAMATH_CALUDE_intersection_with_complement_l431_43111

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_with_complement :
  A ∩ (U \ B) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l431_43111


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l431_43198

/-- The total surface area of a cube with edge length 3 meters and square holes of edge length 1 meter drilled through each face. -/
def total_surface_area (cube_edge : ℝ) (hole_edge : ℝ) : ℝ :=
  let exterior_area := 6 * cube_edge^2 - 6 * hole_edge^2
  let interior_area := 24 * cube_edge * hole_edge
  exterior_area + interior_area

/-- Theorem stating that the total surface area of the described cube is 120 square meters. -/
theorem cube_with_holes_surface_area :
  total_surface_area 3 1 = 120 := by
  sorry

#eval total_surface_area 3 1

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l431_43198


namespace NUMINAMATH_CALUDE_sector_area_l431_43185

/-- The area of a circular sector with central angle π/3 and radius 4 is 8π/3 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 4) :
  (1 / 2) * r * r * θ = (8 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l431_43185


namespace NUMINAMATH_CALUDE_irregular_decagon_angle_l431_43120

/-- Theorem: In a 10-sided polygon where the sum of all interior angles is 1470°,
    and 9 of the angles are equal, the measure of the non-equal angle is 174°. -/
theorem irregular_decagon_angle (n : ℕ) (sum : ℝ) (regular_angle : ℝ) (irregular_angle : ℝ) :
  n = 10 ∧ 
  sum = 1470 ∧
  (n - 1) * regular_angle + irregular_angle = sum ∧
  (n - 1) * regular_angle = (n - 2) * 180 →
  irregular_angle = 174 := by
  sorry

end NUMINAMATH_CALUDE_irregular_decagon_angle_l431_43120


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_19_12_l431_43182

theorem sum_of_solutions_eq_19_12 : ∃ (x₁ x₂ : ℝ), 
  (4 * x₁ + 7) * (3 * x₁ - 10) = 0 ∧
  (4 * x₂ + 7) * (3 * x₂ - 10) = 0 ∧
  x₁ ≠ x₂ ∧
  x₁ + x₂ = 19 / 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_19_12_l431_43182


namespace NUMINAMATH_CALUDE_trigonometric_calculation_l431_43176

theorem trigonometric_calculation : ((-2)^2 : ℝ) + 2 * Real.sin (π/3) - Real.tan (π/3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_calculation_l431_43176


namespace NUMINAMATH_CALUDE_complement_of_M_wrt_U_l431_43184

open Set

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}

theorem complement_of_M_wrt_U : (U \ M) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_wrt_U_l431_43184


namespace NUMINAMATH_CALUDE_positive_A_value_l431_43175

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) : 
  (hash A 5 = 169) → (A > 0) → (A = 12) := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l431_43175


namespace NUMINAMATH_CALUDE_sams_age_two_years_ago_l431_43129

/-- Given the ages of John and Sam, prove Sam's age two years ago -/
theorem sams_age_two_years_ago (john_age sam_age : ℕ) : 
  john_age = 3 * sam_age →
  john_age + 9 = 2 * (sam_age + 9) →
  sam_age - 2 = 7 := by
  sorry

#check sams_age_two_years_ago

end NUMINAMATH_CALUDE_sams_age_two_years_ago_l431_43129


namespace NUMINAMATH_CALUDE_area_left_of_y_axis_is_half_l431_43167

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Calculates the area of a parallelogram left of the y-axis -/
def areaLeftOfYAxis (p : Parallelogram) : ℝ := sorry

/-- The main theorem stating that the area left of the y-axis is half the total area -/
theorem area_left_of_y_axis_is_half (p : Parallelogram) 
  (h1 : p.A = ⟨3, 4⟩) 
  (h2 : p.B = ⟨-2, 1⟩) 
  (h3 : p.C = ⟨-5, -2⟩) 
  (h4 : p.D = ⟨0, 1⟩) : 
  areaLeftOfYAxis p = (1 / 2) * area p := by
  sorry

end NUMINAMATH_CALUDE_area_left_of_y_axis_is_half_l431_43167


namespace NUMINAMATH_CALUDE_total_pencils_l431_43164

/-- Given that each child has 2 pencils and there are 9 children, 
    prove that the total number of pencils is 18. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) (h2 : num_children = 9) : 
  pencils_per_child * num_children = 18 := by
sorry

end NUMINAMATH_CALUDE_total_pencils_l431_43164


namespace NUMINAMATH_CALUDE_george_number_l431_43125

/-- Checks if a number is skipped by a student given their position in the sequence -/
def isSkipped (num : ℕ) (studentPosition : ℕ) : Prop :=
  ∃ k : ℕ, num = 5^studentPosition * (5 * k - 1) - 1

/-- Checks if a number is the sum of squares of two consecutive integers -/
def isSumOfConsecutiveSquares (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 + (k+1)^2

theorem george_number :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 1005 ∧
  (∀ i : ℕ, i ≥ 1 ∧ i ≤ 6 → ¬isSkipped n i) ∧
  isSumOfConsecutiveSquares n ∧
  n = 25 := by sorry

end NUMINAMATH_CALUDE_george_number_l431_43125


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l431_43187

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) :
  Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l431_43187


namespace NUMINAMATH_CALUDE_sum_equals_zero_l431_43171

theorem sum_equals_zero (m n p : ℝ) 
  (h1 : m * n + p^2 + 4 = 0) 
  (h2 : m - n = 4) : 
  m + n = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_zero_l431_43171


namespace NUMINAMATH_CALUDE_lily_milk_problem_l431_43191

theorem lily_milk_problem (initial_milk : ℚ) (given_milk : ℚ) (remaining_milk : ℚ) :
  initial_milk = 5 ∧ given_milk = 18 / 7 ∧ remaining_milk = initial_milk - given_milk →
  remaining_milk = 17 / 7 := by
  sorry

end NUMINAMATH_CALUDE_lily_milk_problem_l431_43191


namespace NUMINAMATH_CALUDE_watson_class_size_l431_43101

/-- The number of students in Ms. Watson's class -/
def total_students (kindergartners first_graders second_graders : ℕ) : ℕ :=
  kindergartners + first_graders + second_graders

/-- Theorem stating the total number of students in Ms. Watson's class -/
theorem watson_class_size :
  total_students 14 24 4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_watson_class_size_l431_43101


namespace NUMINAMATH_CALUDE_interest_rate_equation_l431_43130

/-- Given a principal that doubles in 10 years with quarterly compound interest,
    prove that the annual interest rate satisfies the equation 2 = (1 + r/4)^40 -/
theorem interest_rate_equation (r : ℝ) : 2 = (1 + r/4)^40 ↔ 
  ∀ (P : ℝ), P > 0 → 2*P = P * (1 + r/4)^40 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equation_l431_43130


namespace NUMINAMATH_CALUDE_count_sevens_20_to_119_l431_43161

/-- Count of digit 7 in a number -/
def countSevens (n : ℕ) : ℕ := sorry

/-- Sum of countSevens for a range of natural numbers -/
def sumCountSevens (start finish : ℕ) : ℕ := sorry

theorem count_sevens_20_to_119 : sumCountSevens 20 119 = 19 := by sorry

end NUMINAMATH_CALUDE_count_sevens_20_to_119_l431_43161


namespace NUMINAMATH_CALUDE_midpoint_chain_l431_43137

/-- Given points A, B, C, D, E, F on a line segment, where:
    C is the midpoint of AB,
    D is the midpoint of AC,
    E is the midpoint of AD,
    F is the midpoint of AE,
    and AB = 64,
    prove that AF = 4. -/
theorem midpoint_chain (A B C D E F : ℝ) : 
  C = (A + B) / 2 →
  D = (A + C) / 2 →
  E = (A + D) / 2 →
  F = (A + E) / 2 →
  B - A = 64 →
  F - A = 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_chain_l431_43137


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l431_43199

theorem quadratic_form_equivalence : ∃ (a b c : ℝ), 
  (∀ x, x * (x + 2) = 5 * (x - 2) ↔ a * x^2 + b * x + c = 0) ∧ 
  a = 1 ∧ b = -3 ∧ c = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l431_43199


namespace NUMINAMATH_CALUDE_rosies_pies_l431_43124

/-- Given that Rosie can make 3 pies from 12 apples, this theorem proves
    how many pies she can make from 36 apples. -/
theorem rosies_pies (apples_per_three_pies : ℕ) (total_apples : ℕ) 
  (h1 : apples_per_three_pies = 12)
  (h2 : total_apples = 36) :
  (total_apples / apples_per_three_pies) * 3 = 9 :=
sorry

end NUMINAMATH_CALUDE_rosies_pies_l431_43124


namespace NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_one_l431_43189

/-- The function f(x) = (1/3)x³ + x² + ax + 1 is monotonically increasing in the interval [-2, a] -/
def is_monotone_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ a → f x < f y

/-- The main theorem stating that if f(x) = (1/3)x³ + x² + ax + 1 is monotonically increasing 
    in the interval [-2, a], then a ≥ 1 -/
theorem monotone_increasing_implies_a_geq_one (a : ℝ) :
  is_monotone_increasing (fun x => (1/3) * x^3 + x^2 + a*x + 1) a → a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_one_l431_43189


namespace NUMINAMATH_CALUDE_smallest_sum_of_pairwise_sums_l431_43106

theorem smallest_sum_of_pairwise_sums (a b c d : ℝ) (y : ℝ) : 
  let sums := {a + b, a + c, a + d, b + c, b + d, c + d}
  ({170, 305, 270, 255, 320, y} : Set ℝ) = sums →
  (320 ∈ sums) →
  (∀ z ∈ sums, 320 + y ≤ z + y) →
  320 + y = 255 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_pairwise_sums_l431_43106


namespace NUMINAMATH_CALUDE_remaining_walk_time_l431_43168

theorem remaining_walk_time (total_distance : ℝ) (speed : ℝ) (walked_distance : ℝ) : 
  total_distance = 2.5 → 
  speed = 1 / 20 → 
  walked_distance = 1 → 
  (total_distance - walked_distance) / speed = 30 := by
sorry

end NUMINAMATH_CALUDE_remaining_walk_time_l431_43168


namespace NUMINAMATH_CALUDE_inequality_subtraction_l431_43141

theorem inequality_subtraction (a b : ℝ) : a < b → a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_subtraction_l431_43141


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l431_43115

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_students : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_students : Nat) 
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_students = 5)
  (h4 : avg_age_group1 = 14)
  (h5 : group2_students = 9)
  (h6 : avg_age_group2 = 16) :
  (total_students : ℝ) * avg_age_all - 
  ((group1_students : ℝ) * avg_age_group1 + (group2_students : ℝ) * avg_age_group2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l431_43115


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l431_43194

theorem smallest_b_for_factorization : ∃ (b : ℕ), 
  (∀ (x p q : ℤ), (x^2 + b*x + 1800 = (x + p) * (x + q)) → (p > 0 ∧ q > 0)) ∧
  (∀ (b' : ℕ), b' < b → ¬∃ (p q : ℤ), (p > 0 ∧ q > 0 ∧ x^2 + b'*x + 1800 = (x + p) * (x + q))) ∧
  b = 85 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l431_43194


namespace NUMINAMATH_CALUDE_multiply_by_seven_l431_43146

theorem multiply_by_seven (x : ℝ) : 7 * x = 50.68 → x = 7.24 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_seven_l431_43146


namespace NUMINAMATH_CALUDE_correct_propositions_l431_43128

theorem correct_propositions :
  let prop1 := (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔ (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0)
  let prop2 := ∀ p q : Prop, (p ∨ q) → (p ∧ q)
  let prop3 := ∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q)
  let prop4 := (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0)
  prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 := by sorry

end NUMINAMATH_CALUDE_correct_propositions_l431_43128


namespace NUMINAMATH_CALUDE_cookies_left_after_six_days_l431_43152

/-- Represents the number of cookies baked and eaten over six days -/
structure CookieCount where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  parentEaten : ℕ
  neighborEaten : ℕ

/-- Calculates the total number of cookies left after six days -/
def totalCookiesLeft (c : CookieCount) : ℕ :=
  c.monday + c.tuesday + c.wednesday + c.thursday + c.friday + c.saturday - (c.parentEaten + c.neighborEaten)

/-- Theorem stating the number of cookies left after six days -/
theorem cookies_left_after_six_days :
  ∃ (c : CookieCount),
    c.monday = 32 ∧
    c.tuesday = c.monday / 2 ∧
    c.wednesday = (c.tuesday * 3) - 4 ∧
    c.thursday = (c.monday * 2) - 10 ∧
    c.friday = (c.tuesday * 3) - 6 ∧
    c.saturday = c.monday + c.friday ∧
    c.parentEaten = 2 * 6 ∧
    c.neighborEaten = 8 ∧
    totalCookiesLeft c = 242 := by
  sorry


end NUMINAMATH_CALUDE_cookies_left_after_six_days_l431_43152


namespace NUMINAMATH_CALUDE_number_puzzle_l431_43162

theorem number_puzzle : ∃ x : ℝ, 47 - 3 * x = 14 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l431_43162


namespace NUMINAMATH_CALUDE_saber_toothed_frog_tails_l431_43112

/-- Represents the number of tadpoles of each type -/
structure TadpoleCount where
  triassic : ℕ
  saber : ℕ

/-- Represents the characteristics of each tadpole type -/
structure TadpoleType where
  legs : ℕ
  tails : ℕ

/-- The main theorem to prove -/
theorem saber_toothed_frog_tails 
  (triassic : TadpoleType)
  (saber : TadpoleType)
  (count : TadpoleCount)
  (h1 : triassic.legs = 5)
  (h2 : triassic.tails = 1)
  (h3 : saber.legs = 4)
  (h4 : count.triassic * triassic.legs + count.saber * saber.legs = 100)
  (h5 : count.triassic * triassic.tails + count.saber * saber.tails = 64) :
  saber.tails = 3 := by
  sorry

end NUMINAMATH_CALUDE_saber_toothed_frog_tails_l431_43112


namespace NUMINAMATH_CALUDE_extreme_value_condition_l431_43105

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem extreme_value_condition (a b : ℝ) :
  f a b 1 = 10 ∧ f_derivative a b 1 = 0 → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l431_43105


namespace NUMINAMATH_CALUDE_greatest_of_five_consecutive_integers_sum_cube_l431_43135

theorem greatest_of_five_consecutive_integers_sum_cube (n : ℤ) (m : ℤ) : 
  (5 * n + 10 = m^3) → 
  (∀ k : ℤ, k > n → 5 * k + 10 ≠ m^3) → 
  202 = n + 4 :=
by sorry

end NUMINAMATH_CALUDE_greatest_of_five_consecutive_integers_sum_cube_l431_43135


namespace NUMINAMATH_CALUDE_cubic_equation_geometric_progression_l431_43114

theorem cubic_equation_geometric_progression (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
   x^3 + 16*x^2 + a*x + 64 = 0 ∧
   y^3 + 16*y^2 + a*y + 64 = 0 ∧
   z^3 + 16*z^2 + a*z + 64 = 0 ∧
   ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ y = x*q ∧ z = y*q) →
  a = 64 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_geometric_progression_l431_43114


namespace NUMINAMATH_CALUDE_odd_function_inequality_l431_43109

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x / 3 - 2^x
  else if x < 0 then x / 3 + 2^(-x)
  else 0

-- State the theorem
theorem odd_function_inequality (k : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ t, f (t^2 - 2*t) + f (2*t^2 - k) < 0) →
  k < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l431_43109


namespace NUMINAMATH_CALUDE_smallest_six_digit_negative_congruent_to_5_mod_17_l431_43165

theorem smallest_six_digit_negative_congruent_to_5_mod_17 :
  ∀ n : ℤ, -999999 ≤ n ∧ n < -99999 ∧ n ≡ 5 [ZMOD 17] → n ≥ -100011 :=
by sorry

end NUMINAMATH_CALUDE_smallest_six_digit_negative_congruent_to_5_mod_17_l431_43165


namespace NUMINAMATH_CALUDE_problem_8_l431_43119

theorem problem_8 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + b^2 + c^2 = 63)
  (h2 : 2*a + 3*b + 6*c = 21*Real.sqrt 7) :
  (a/c)^(a/b) = (1/3)^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_8_l431_43119


namespace NUMINAMATH_CALUDE_min_values_theorem_l431_43127

theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a * b = a + 3 * b) :
  (∀ x y, x > 0 → y > 0 → 3 * x * y = x + 3 * y → 3 * a + b ≤ 3 * x + y) ∧
  (∀ x y, x > 0 → y > 0 → 3 * x * y = x + 3 * y → a * b ≤ x * y) ∧
  (∀ x y, x > 0 → y > 0 → 3 * x * y = x + 3 * y → a^2 + 9 * b^2 ≤ x^2 + 9 * y^2) ∧
  (3 * a + b = 16 / 3 ∨ a * b = 4 / 3 ∨ a^2 + 9 * b^2 = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_values_theorem_l431_43127


namespace NUMINAMATH_CALUDE_probability_nine_correct_zero_l431_43133

/-- Represents a matching problem with n pairs -/
structure MatchingProblem (n : ℕ) where
  /-- The number of pairs to match -/
  pairs : ℕ
  /-- Assumption that the number of pairs is positive -/
  positive : 0 < pairs
  /-- Assumption that the number of pairs is equal to n -/
  eq_n : pairs = n

/-- The probability of randomly matching exactly k pairs correctly in a matching problem with n pairs -/
def probability_exact_match (n k : ℕ) (prob : MatchingProblem n) : ℚ :=
  sorry

/-- Theorem stating that the probability of randomly matching exactly 9 pairs correctly in a matching problem with 10 pairs is 0 -/
theorem probability_nine_correct_zero : 
  ∀ (prob : MatchingProblem 10), probability_exact_match 10 9 prob = 0 :=
sorry

end NUMINAMATH_CALUDE_probability_nine_correct_zero_l431_43133


namespace NUMINAMATH_CALUDE_fence_rods_count_l431_43186

/-- Calculates the total number of metal rods needed for a fence --/
def total_rods (panels : ℕ) (sheets_per_panel : ℕ) (beams_per_panel : ℕ) 
                (rods_per_sheet : ℕ) (rods_per_beam : ℕ) : ℕ :=
  panels * (sheets_per_panel * rods_per_sheet + beams_per_panel * rods_per_beam)

/-- Proves that the total number of metal rods needed for the fence is 380 --/
theorem fence_rods_count : total_rods 10 3 2 10 4 = 380 := by
  sorry

end NUMINAMATH_CALUDE_fence_rods_count_l431_43186


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_three_l431_43169

theorem gcd_of_powers_of_three :
  Nat.gcd (3^1007 - 1) (3^1018 - 1) = 3^11 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_three_l431_43169


namespace NUMINAMATH_CALUDE_range_of_x_minus_cos_y_l431_43118

theorem range_of_x_minus_cos_y :
  ∀ x y : ℝ, x^2 + 2 * Real.cos y = 1 →
  ∃ z : ℝ, z = x - Real.cos y ∧ -1 ≤ z ∧ z ≤ Real.sqrt 3 + 1 ∧
  (∃ x₁ y₁ : ℝ, x₁^2 + 2 * Real.cos y₁ = 1 ∧ x₁ - Real.cos y₁ = -1) ∧
  (∃ x₂ y₂ : ℝ, x₂^2 + 2 * Real.cos y₂ = 1 ∧ x₂ - Real.cos y₂ = Real.sqrt 3 + 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_minus_cos_y_l431_43118


namespace NUMINAMATH_CALUDE_bill_share_proof_l431_43138

def total_bill : ℝ := 139.00
def num_people : ℕ := 9
def tip_percentage : ℝ := 0.10

theorem bill_share_proof :
  let tip := total_bill * tip_percentage
  let total_with_tip := total_bill + tip
  let share_per_person := total_with_tip / num_people
  ∃ ε > 0, |share_per_person - 16.99| < ε :=
by sorry

end NUMINAMATH_CALUDE_bill_share_proof_l431_43138


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l431_43145

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles :
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 12*x + y^2 - 6*y + 9 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 10*x + y^2 + 8*y + 34 = 0}
  (shortest_distance : ℝ) →
  shortest_distance = Real.sqrt 170 - 3 - Real.sqrt 7 ∧
  ∀ (p1 : ℝ × ℝ) (p2 : ℝ × ℝ),
    p1 ∈ circle1 → p2 ∈ circle2 →
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) ≥ shortest_distance :=
by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_between_circles_l431_43145


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l431_43160

theorem arithmetic_calculation : 5020 - (1004 / 20.08) = 4970 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l431_43160


namespace NUMINAMATH_CALUDE_square_corner_distance_l431_43126

theorem square_corner_distance (small_perimeter large_area : ℝ) 
  (h_small : small_perimeter = 8)
  (h_large : large_area = 36) : ∃ (distance : ℝ), distance = Real.sqrt 32 :=
by
  sorry

end NUMINAMATH_CALUDE_square_corner_distance_l431_43126


namespace NUMINAMATH_CALUDE_factor_calculation_l431_43144

theorem factor_calculation (n : ℝ) (f : ℝ) (h1 : n = 155) (h2 : n * f - 200 = 110) : f = 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l431_43144


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l431_43177

theorem inverse_proportion_problem (x y : ℝ → ℝ) (k : ℝ) :
  (∀ t, x t * y t = k) →  -- x and y are inversely proportional
  x 15 = 3 →              -- x = 3 when y = 15
  y 15 = 15 →             -- y = 15 when x = 3
  y (-30) = -30 →         -- y = -30
  x (-30) = -3/2 :=       -- x = -3/2 when y = -30
by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l431_43177


namespace NUMINAMATH_CALUDE_gcd_306_522_l431_43156

theorem gcd_306_522 : Nat.gcd 306 522 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_306_522_l431_43156


namespace NUMINAMATH_CALUDE_two_heads_in_three_tosses_l431_43134

/-- The probability of getting exactly k successes in n trials with probability p of success on each trial. -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of getting exactly 2 heads when a fair coin is tossed 3 times is 0.375 -/
theorem two_heads_in_three_tosses :
  binomialProbability 3 2 (1/2) = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_two_heads_in_three_tosses_l431_43134


namespace NUMINAMATH_CALUDE_max_value_on_unit_circle_l431_43190

def unitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem max_value_on_unit_circle (x₁ y₁ x₂ y₂ : ℝ) :
  unitCircle x₁ y₁ →
  unitCircle x₂ y₂ →
  (x₁, y₁) ≠ (x₂, y₂) →
  x₁ * y₂ = x₂ * y₁ →
  ∀ t, 2*x₁ + x₂ + 2*y₁ + y₂ ≤ t →
  t = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_unit_circle_l431_43190


namespace NUMINAMATH_CALUDE_triangle_height_l431_43163

/-- Proves that a triangle with area 46 cm² and base 10 cm has a height of 9.2 cm -/
theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 46 → base = 10 → area = (base * height) / 2 → height = 9.2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l431_43163


namespace NUMINAMATH_CALUDE_largest_power_l431_43131

theorem largest_power : 
  3^4000 > 2^5000 ∧ 
  3^4000 > 4^3000 ∧ 
  3^4000 > 5^2000 ∧ 
  3^4000 > 6^1000 := by sorry

end NUMINAMATH_CALUDE_largest_power_l431_43131


namespace NUMINAMATH_CALUDE_initial_number_count_l431_43172

theorem initial_number_count (n : ℕ) (S : ℝ) : 
  S / n = 20 →
  (S - 100) / (n - 2) = 18.75 →
  n = 110 := by
sorry

end NUMINAMATH_CALUDE_initial_number_count_l431_43172


namespace NUMINAMATH_CALUDE_olaf_sailing_speed_l431_43117

/-- Given the conditions of Olaf's sailing trip, prove the boat's daily travel distance. -/
theorem olaf_sailing_speed :
  -- Total distance to travel
  ∀ (total_distance : ℝ)
  -- Total number of men
  (total_men : ℕ)
  -- Water consumption per man per day (in gallons)
  (water_per_man_per_day : ℝ)
  -- Total water available (in gallons)
  (total_water : ℝ),
  total_distance = 4000 →
  total_men = 25 →
  water_per_man_per_day = 1/2 →
  total_water = 250 →
  -- The boat can travel this many miles per day
  (total_distance / (total_water / (total_men * water_per_man_per_day))) = 200 :=
by
  sorry


end NUMINAMATH_CALUDE_olaf_sailing_speed_l431_43117


namespace NUMINAMATH_CALUDE_projections_proportional_to_squares_l431_43157

/-- In a right triangle, the projections of the legs onto the hypotenuse are proportional to the squares of the legs. -/
theorem projections_proportional_to_squares 
  {a b c a1 b1 : ℝ} 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (right_triangle : a^2 + b^2 = c^2)
  (proj_a : a1 = (a^2) / c)
  (proj_b : b1 = (b^2) / c) :
  a1 / b1 = a^2 / b^2 := by
  sorry

end NUMINAMATH_CALUDE_projections_proportional_to_squares_l431_43157


namespace NUMINAMATH_CALUDE_meal_combinations_count_l431_43183

/-- The number of items on the menu -/
def menu_items : ℕ := 12

/-- The number of dishes each person orders -/
def dishes_per_person : ℕ := 1

/-- The number of special dishes shared -/
def shared_special_dishes : ℕ := 1

/-- The number of remaining dishes after choosing the special dish -/
def remaining_dishes : ℕ := menu_items - shared_special_dishes

/-- The number of different meal combinations for Yann and Camille -/
def meal_combinations : ℕ := remaining_dishes * remaining_dishes

theorem meal_combinations_count : meal_combinations = 121 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_count_l431_43183


namespace NUMINAMATH_CALUDE_rajan_profit_share_l431_43116

/-- Calculates the share of profit for a partner in a business --/
def calculate_profit_share (
  rajan_investment : ℕ) (rajan_duration : ℕ)
  (rakesh_investment : ℕ) (rakesh_duration : ℕ)
  (mukesh_investment : ℕ) (mukesh_duration : ℕ)
  (total_profit : ℕ) : ℕ :=
  let rajan_ratio := rajan_investment * rajan_duration
  let rakesh_ratio := rakesh_investment * rakesh_duration
  let mukesh_ratio := mukesh_investment * mukesh_duration
  let total_ratio := rajan_ratio + rakesh_ratio + mukesh_ratio
  (rajan_ratio * total_profit) / total_ratio

/-- Theorem stating that Rajan's share of the profit is 2400 --/
theorem rajan_profit_share :
  calculate_profit_share 20000 12 25000 4 15000 8 4600 = 2400 :=
by sorry

end NUMINAMATH_CALUDE_rajan_profit_share_l431_43116


namespace NUMINAMATH_CALUDE_watermelon_count_l431_43139

theorem watermelon_count (seeds_per_watermelon : ℕ) (total_seeds : ℕ) (h1 : seeds_per_watermelon = 100) (h2 : total_seeds = 400) :
  total_seeds / seeds_per_watermelon = 4 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_count_l431_43139


namespace NUMINAMATH_CALUDE_simplify_expression_l431_43180

theorem simplify_expression (x y : ℝ) :
  (3 * x^2 + 4 * x + 6 * y - 9) - (x^2 - 2 * x + 3 * y + 15) = 2 * x^2 + 6 * x + 3 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l431_43180


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l431_43143

/-- Given that -1, a, b, c, -9 form an arithmetic sequence, prove that b = -5 and ac = 21 -/
theorem arithmetic_sequence_problem (a b c : ℝ) 
  (h1 : ∃ (d : ℝ), a - (-1) = d ∧ b - a = d ∧ c - b = d ∧ (-9) - c = d) : 
  b = -5 ∧ a * c = 21 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l431_43143


namespace NUMINAMATH_CALUDE_set_equiv_interval_l431_43174

-- Define the set S as {x | x ≤ -1}
def S : Set ℝ := {x | x ≤ -1}

-- Define the interval (-∞, -1]
def I : Set ℝ := Set.Iic (-1)

-- Theorem: S is equivalent to I
theorem set_equiv_interval : S = I := by sorry

end NUMINAMATH_CALUDE_set_equiv_interval_l431_43174


namespace NUMINAMATH_CALUDE_abs_one_minus_x_gt_one_solution_set_l431_43102

theorem abs_one_minus_x_gt_one_solution_set :
  {x : ℝ | |1 - x| > 1} = Set.Ioi 2 ∪ Set.Iic 0 := by sorry

end NUMINAMATH_CALUDE_abs_one_minus_x_gt_one_solution_set_l431_43102


namespace NUMINAMATH_CALUDE_giraffe_height_difference_l431_43149

/-- The height of the tallest giraffe in inches -/
def tallest_giraffe : ℕ := 96

/-- The height of the shortest giraffe in inches -/
def shortest_giraffe : ℕ := 68

/-- The number of adult giraffes at the zoo -/
def num_giraffes : ℕ := 14

/-- The difference in height between the tallest and shortest giraffe -/
def height_difference : ℕ := tallest_giraffe - shortest_giraffe

theorem giraffe_height_difference :
  height_difference = 28 :=
sorry

end NUMINAMATH_CALUDE_giraffe_height_difference_l431_43149


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l431_43170

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a + 2 * b + 1 = 3) :
  -4 * a - 2 * b + 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l431_43170


namespace NUMINAMATH_CALUDE_sum_of_fractions_l431_43166

theorem sum_of_fractions : (7 : ℚ) / 12 + (3 : ℚ) / 8 = (23 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l431_43166
