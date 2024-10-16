import Mathlib

namespace NUMINAMATH_CALUDE_geometric_progression_solution_l2517_251717

theorem geometric_progression_solution (b₁ q : ℝ) : 
  b₁ * (1 + q + q^2) = 21 ∧ 
  b₁^2 * (1 + q^2 + q^4) = 189 → 
  ((b₁ = 3 ∧ q = 2) ∨ (b₁ = 12 ∧ q = 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l2517_251717


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_l2517_251790

/-- The number of seats in a row -/
def num_seats : ℕ := 7

/-- The number of persons to be seated -/
def num_persons : ℕ := 2

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.factorial) / ((n - k).factorial)

/-- Theorem stating the number of valid seating arrangements -/
theorem valid_seating_arrangements :
  seating_arrangements num_seats num_persons - 
  (num_seats - 1) * seating_arrangements 2 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_l2517_251790


namespace NUMINAMATH_CALUDE_coconut_grove_yield_l2517_251749

/-- Given a coconut grove with the following conditions:
    - (x + 2) trees yield 30 nuts per year each
    - x trees yield 120 nuts per year each
    - (x - 2) trees yield 180 nuts per year each
    - The average yield per year per tree is 100
    Prove that x = 10 -/
theorem coconut_grove_yield (x : ℝ) : 
  ((x + 2) * 30 + x * 120 + (x - 2) * 180) / (3 * x) = 100 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_yield_l2517_251749


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2517_251765

/-- Calculates the speed of a train given the lengths of two trains, the speed of the second train, and the time taken for the first train to pass the second train. -/
theorem train_speed_calculation (length1 length2 : ℝ) (speed2 : ℝ) (time : ℝ) :
  length1 = 250 →
  length2 = 300 →
  speed2 = 36 * (1000 / 3600) →
  time = 54.995600351971845 →
  ∃ (speed1 : ℝ), speed1 = 72 * (1000 / 3600) ∧
    (length1 + length2) / time = speed1 - speed2 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2517_251765


namespace NUMINAMATH_CALUDE_carly_lollipops_l2517_251737

/-- The number of grape lollipops -/
def grape_lollipops : ℕ := 7

/-- The total number of lollipops Carly has -/
def total_lollipops : ℕ := 42

/-- The number of non-cherry lollipop flavors -/
def non_cherry_flavors : ℕ := 3

theorem carly_lollipops :
  (total_lollipops / 2 = total_lollipops - total_lollipops / 2) ∧
  ((total_lollipops - total_lollipops / 2) / non_cherry_flavors = grape_lollipops) ∧
  (total_lollipops = 42) := by
  sorry

end NUMINAMATH_CALUDE_carly_lollipops_l2517_251737


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2517_251783

/-- The trajectory of the midpoint of a line segment with one end fixed and the other on a circle -/
theorem midpoint_trajectory (m n x y : ℝ) : 
  (m + 1)^2 + n^2 = 4 →  -- B(m, n) is on the circle (x+1)^2 + y^2 = 4
  x = (m + 4) / 2 →      -- x-coordinate of midpoint M
  y = (n - 3) / 2 →      -- y-coordinate of midpoint M
  (x - 3/2)^2 + (y + 3/2)^2 = 1 := by
sorry


end NUMINAMATH_CALUDE_midpoint_trajectory_l2517_251783


namespace NUMINAMATH_CALUDE_jacoby_needs_3214_l2517_251738

/-- The amount Jacoby needs for his trip to Brickville -/
def tripCost : ℕ := 5000

/-- Jacoby's hourly wage -/
def hourlyWage : ℕ := 20

/-- Hours Jacoby worked -/
def hoursWorked : ℕ := 10

/-- Price of each cookie -/
def cookiePrice : ℕ := 4

/-- Number of cookies sold -/
def cookiesSold : ℕ := 24

/-- Cost of lottery ticket -/
def lotteryCost : ℕ := 10

/-- Lottery winnings -/
def lotteryWin : ℕ := 500

/-- Gift amount from each sister -/
def sisterGift : ℕ := 500

/-- Number of sisters who gave gifts -/
def numSisters : ℕ := 2

/-- Calculate the remaining amount Jacoby needs for his trip -/
def remainingAmount : ℕ :=
  tripCost - (
    hourlyWage * hoursWorked +
    cookiePrice * cookiesSold +
    lotteryWin +
    sisterGift * numSisters -
    lotteryCost
  )

theorem jacoby_needs_3214 : remainingAmount = 3214 := by
  sorry

end NUMINAMATH_CALUDE_jacoby_needs_3214_l2517_251738


namespace NUMINAMATH_CALUDE_equation_solution_l2517_251703

theorem equation_solution : 
  let f (x : ℝ) := (x^3 + x^2 + x + 1) / (x + 1)
  let g (x : ℝ) := x^2 + 4*x + 4
  ∀ x : ℝ, f x = g x ↔ x = -3/4 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2517_251703


namespace NUMINAMATH_CALUDE_congruent_triangles_exist_l2517_251723

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  (n_ge_4 : n ≥ 4)

/-- A subset of vertices of a regular polygon -/
structure VertexSubset (n : ℕ) where
  (polygon : RegularPolygon n)
  (r : ℕ)
  (vertices : Finset (Fin n))
  (subset_size : vertices.card = r)

/-- Two triangles in a regular polygon -/
structure PolygonTrianglePair (n : ℕ) where
  (polygon : RegularPolygon n)
  (t1 t2 : Fin n → Fin n → Fin n → Prop)

/-- Congruence of two triangles in a regular polygon -/
def CongruentTriangles (n : ℕ) (pair : PolygonTrianglePair n) : Prop :=
  sorry

/-- The main theorem -/
theorem congruent_triangles_exist (n : ℕ) (V : VertexSubset n) 
  (h : V.r * (V.r - 3) ≥ n) : 
  ∃ (pair : PolygonTrianglePair n), 
    (∀ i j k, pair.t1 i j k → i ∈ V.vertices ∧ j ∈ V.vertices ∧ k ∈ V.vertices) ∧
    (∀ i j k, pair.t2 i j k → i ∈ V.vertices ∧ j ∈ V.vertices ∧ k ∈ V.vertices) ∧
    CongruentTriangles n pair :=
sorry

end NUMINAMATH_CALUDE_congruent_triangles_exist_l2517_251723


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2517_251788

theorem sum_of_squares_of_roots (a b c d : ℝ) : 
  (a^4 - 15*a^2 + 56 = 0) ∧ 
  (b^4 - 15*b^2 + 56 = 0) ∧ 
  (c^4 - 15*c^2 + 56 = 0) ∧ 
  (d^4 - 15*d^2 + 56 = 0) ∧ 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) →
  a^2 + b^2 + c^2 + d^2 = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2517_251788


namespace NUMINAMATH_CALUDE_race_participants_l2517_251792

theorem race_participants (total : ℕ) (finished : ℕ) : 
  finished = 52 →
  (3/4 : ℚ) * total * (1/3 : ℚ) + 
  (3/4 : ℚ) * total * (2/3 : ℚ) * (4/5 : ℚ) = finished →
  total = 130 := by
  sorry

end NUMINAMATH_CALUDE_race_participants_l2517_251792


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2517_251784

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 2 + a 8 = 12) :
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2517_251784


namespace NUMINAMATH_CALUDE_caterer_order_l2517_251771

/-- The number of ice-cream bars ordered by a caterer -/
def num_ice_cream_bars : ℕ := 225

/-- The total price of the order in cents -/
def total_price : ℕ := 20000

/-- The price of each ice-cream bar in cents -/
def price_ice_cream_bar : ℕ := 60

/-- The price of each sundae in cents -/
def price_sundae : ℕ := 52

/-- The number of sundaes ordered -/
def num_sundaes : ℕ := 125

theorem caterer_order :
  num_ice_cream_bars * price_ice_cream_bar + num_sundaes * price_sundae = total_price :=
by sorry

end NUMINAMATH_CALUDE_caterer_order_l2517_251771


namespace NUMINAMATH_CALUDE_equation_solution_l2517_251759

theorem equation_solution (x : ℝ) : x ≠ -2 → (-2 * x^2 = (4 * x + 2) / (x + 2)) ↔ (x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2517_251759


namespace NUMINAMATH_CALUDE_james_pizza_fraction_l2517_251716

theorem james_pizza_fraction (num_pizzas : ℕ) (slices_per_pizza : ℕ) (james_slices : ℕ) :
  num_pizzas = 2 →
  slices_per_pizza = 6 →
  james_slices = 8 →
  (james_slices : ℚ) / (num_pizzas * slices_per_pizza : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_james_pizza_fraction_l2517_251716


namespace NUMINAMATH_CALUDE_number_relationships_l2517_251797

theorem number_relationships : 
  (¬(∀ (x y : ℝ), x = 2 * y)) ∧ 
  (∀ (x : ℝ), ∃ (y : ℝ), x = 2 * y) ∧ 
  (∀ (y : ℝ), ∃ (x : ℝ), x = 2 * y) ∧ 
  (¬(∃ (x : ℝ), ∀ (y : ℝ), x = 2 * y)) ∧ 
  (¬(∃ (y : ℝ), ∀ (x : ℝ), x = 2 * y)) ∧ 
  (∃ (x y : ℝ), x = 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_number_relationships_l2517_251797


namespace NUMINAMATH_CALUDE_collinearity_iff_sum_one_l2517_251770

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define points
variable (O A B P : V)

-- Define real numbers m and n
variable (m n : ℝ)

-- Define the condition that O, A, B are not collinear
def not_collinear (O A B : V) : Prop := 
  ∀ (t : ℝ), (B - O) ≠ t • (A - O)

-- Define the vector equation
def vector_equation (O A B P : V) (m n : ℝ) : Prop :=
  (P - O) = m • (A - O) + n • (B - O)

-- Define collinearity of points A, P, B
def collinear (A P B : V) : Prop :=
  ∃ (t : ℝ), (P - A) = t • (B - A)

-- State the theorem
theorem collinearity_iff_sum_one
  (h₁ : not_collinear O A B)
  (h₂ : vector_equation O A B P m n) :
  collinear A P B ↔ m + n = 1 := by sorry

end NUMINAMATH_CALUDE_collinearity_iff_sum_one_l2517_251770


namespace NUMINAMATH_CALUDE_monotonic_decreasing_intervals_l2517_251747

/-- The function f(x) = (x + 1) / x is monotonically decreasing on (-∞, 0) and (0, +∞) -/
theorem monotonic_decreasing_intervals (f : ℝ → ℝ) :
  (∀ x ≠ 0, f x = (x + 1) / x) →
  (StrictMonoOn f (Set.Iio 0) ∧ StrictMonoOn f (Set.Ioi 0)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_intervals_l2517_251747


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l2517_251722

/-- The surface area of a sphere that circumscribes a cube with edge length 4 -/
theorem sphere_surface_area_with_inscribed_cube : 
  ∀ (cube_edge_length : ℝ) (sphere_radius : ℝ),
    cube_edge_length = 4 →
    sphere_radius = 2 * Real.sqrt 3 →
    4 * Real.pi * sphere_radius^2 = 48 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l2517_251722


namespace NUMINAMATH_CALUDE_cube_face_perimeter_l2517_251727

/-- Given a cube with volume 1000 cm³, prove that the perimeter of one of its faces is 40 cm -/
theorem cube_face_perimeter (V : ℝ) (h : V = 1000) : 
  4 * (V ^ (1/3 : ℝ)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_perimeter_l2517_251727


namespace NUMINAMATH_CALUDE_fraction_integer_iff_p_6_or_28_l2517_251705

theorem fraction_integer_iff_p_6_or_28 (p : ℕ+) :
  (∃ (n : ℕ+), (4 * p + 28 : ℚ) / (3 * p - 7) = n) ↔ p = 6 ∨ p = 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_p_6_or_28_l2517_251705


namespace NUMINAMATH_CALUDE_quartic_roots_difference_l2517_251758

/-- A quartic polynomial with roots forming an arithmetic sequence -/
def quartic_with_arithmetic_roots (a : ℝ) (x : ℝ) : ℝ := 
  a * (x^4 - 10*x^2 + 9)

/-- The derivative of the quartic polynomial -/
def quartic_derivative (a : ℝ) (x : ℝ) : ℝ := 
  4 * a * x * (x^2 - 5)

theorem quartic_roots_difference (a : ℝ) (h : a ≠ 0) :
  let f := quartic_with_arithmetic_roots a
  let f' := quartic_derivative a
  let max_root := Real.sqrt 5
  let min_root := -Real.sqrt 5
  (∀ x, f' x = 0 → x ≤ max_root) ∧ 
  (∀ x, f' x = 0 → x ≥ min_root) ∧
  (max_root - min_root = 2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_quartic_roots_difference_l2517_251758


namespace NUMINAMATH_CALUDE_vishal_investment_percentage_l2517_251764

def total_investment : ℝ := 6358
def raghu_investment : ℝ := 2200
def trishul_investment_percentage : ℝ := 90  -- 100% - 10%

theorem vishal_investment_percentage (vishal_investment trishul_investment : ℝ) : 
  vishal_investment + trishul_investment + raghu_investment = total_investment →
  trishul_investment = raghu_investment * trishul_investment_percentage / 100 →
  (vishal_investment - trishul_investment) / trishul_investment * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_vishal_investment_percentage_l2517_251764


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2517_251763

theorem polynomial_expansion (z : ℝ) : 
  (3 * z^2 + 2 * z - 4) * (4 * z^2 - 3) = 18 * z^4 + 4 * z^3 - 20 * z^2 - 8 * z + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2517_251763


namespace NUMINAMATH_CALUDE_triangle_problem_l2517_251761

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a > b →
  a = 5 →
  c = 6 →
  Real.sin B = 3/5 →
  b = Real.sqrt 13 ∧
  Real.sin A = 3 * Real.sqrt 13 / 13 ∧
  Real.sin (2 * A + π/4) = 7 * Real.sqrt 2 / 26 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2517_251761


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2517_251701

def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem necessary_not_sufficient_condition :
  (∀ a : ℕ → ℝ, IsIncreasing a → ∀ n, |a (n + 1)| > a n) ∧
  (∃ a : ℕ → ℝ, (∀ n, |a (n + 1)| > a n) ∧ ¬IsIncreasing a) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2517_251701


namespace NUMINAMATH_CALUDE_compare_expressions_l2517_251726

theorem compare_expressions (x : ℝ) : x^2 - x > x - 2 := by sorry

end NUMINAMATH_CALUDE_compare_expressions_l2517_251726


namespace NUMINAMATH_CALUDE_largest_prime_factor_l2517_251743

theorem largest_prime_factor : 
  let n := 20^3 + 15^4 - 10^5 + 2*25^3
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n ∧ ∀ q, Nat.Prime q → q ∣ n → q ≤ p ∧ p = 11 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l2517_251743


namespace NUMINAMATH_CALUDE_christian_future_age_l2517_251796

def brian_future_age : ℕ := 40
def years_to_future : ℕ := 8
def christian_current_age : ℕ := 72

theorem christian_future_age :
  christian_current_age + years_to_future = 80 :=
by sorry

end NUMINAMATH_CALUDE_christian_future_age_l2517_251796


namespace NUMINAMATH_CALUDE_automobile_repair_cost_l2517_251735

/-- The cost of fixing Leila's automobile given her supermarket expenses and total spending -/
def cost_to_fix_automobile (supermarket_expense : ℝ) (total_spent : ℝ) : ℝ :=
  3 * supermarket_expense + 50

/-- Theorem: Given the conditions, the cost to fix Leila's automobile is $350 -/
theorem automobile_repair_cost :
  ∃ (supermarket_expense : ℝ),
    cost_to_fix_automobile supermarket_expense 450 + supermarket_expense = 450 ∧
    cost_to_fix_automobile supermarket_expense 450 = 350 := by
  sorry

end NUMINAMATH_CALUDE_automobile_repair_cost_l2517_251735


namespace NUMINAMATH_CALUDE_beautiful_arrangements_theorem_l2517_251750

/-- A beautiful arrangement of numbers 0 to n is a circular arrangement where 
    for any four distinct numbers a, b, c, d with a + c = b + d, 
    the chord joining a and c does not intersect the chord joining b and d -/
def is_beautiful_arrangement (n : ℕ) (arrangement : List ℕ) : Prop :=
  sorry

/-- M is the number of beautiful arrangements of numbers 0 to n -/
def M (n : ℕ) : ℕ :=
  sorry

/-- N is the number of pairs (x, y) of positive integers such that x + y ≤ n and gcd(x, y) = 1 -/
def N (n : ℕ) : ℕ :=
  sorry

/-- For any integer n ≥ 2, M(n) = N(n) + 1 -/
theorem beautiful_arrangements_theorem (n : ℕ) (h : n ≥ 2) : M n = N n + 1 :=
  sorry

end NUMINAMATH_CALUDE_beautiful_arrangements_theorem_l2517_251750


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2517_251706

theorem sum_of_coefficients (C D : ℝ) :
  (∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x - 2) = (5 * x^2 - 8 * x - 6) / (x - 3)) →
  C + D = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2517_251706


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l2517_251774

/-- Calculates the corrected mean of a set of observations after fixing an error -/
def corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * initial_mean - incorrect_value + correct_value) / n

/-- Theorem stating the corrected mean for the given problem -/
theorem corrected_mean_problem :
  let n : ℕ := 50
  let initial_mean : ℚ := 36
  let incorrect_value : ℚ := 23
  let correct_value : ℚ := 60
  corrected_mean n initial_mean incorrect_value correct_value = 36.74 := by
sorry

#eval corrected_mean 50 36 23 60

end NUMINAMATH_CALUDE_corrected_mean_problem_l2517_251774


namespace NUMINAMATH_CALUDE_even_sum_not_both_odd_l2517_251733

theorem even_sum_not_both_odd (n m : ℤ) :
  Even (n^2 + m^2 + n*m) → ¬(Odd n ∧ Odd m) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_not_both_odd_l2517_251733


namespace NUMINAMATH_CALUDE_john_profit_l2517_251789

/-- Calculates John's profit from selling woodburnings, metal sculptures, and paintings. -/
theorem john_profit : 
  let woodburnings_count : ℕ := 20
  let woodburnings_price : ℚ := 15
  let metal_sculptures_count : ℕ := 15
  let metal_sculptures_price : ℚ := 25
  let paintings_count : ℕ := 10
  let paintings_price : ℚ := 40
  let wood_cost : ℚ := 100
  let metal_cost : ℚ := 150
  let paint_cost : ℚ := 120
  let woodburnings_discount : ℚ := 0.1
  let sales_tax : ℚ := 0.05

  let woodburnings_revenue := woodburnings_count * woodburnings_price * (1 - woodburnings_discount)
  let metal_sculptures_revenue := metal_sculptures_count * metal_sculptures_price
  let paintings_revenue := paintings_count * paintings_price
  let total_revenue := woodburnings_revenue + metal_sculptures_revenue + paintings_revenue
  let total_revenue_with_tax := total_revenue * (1 + sales_tax)
  let total_cost := wood_cost + metal_cost + paint_cost
  let profit := total_revenue_with_tax - total_cost

  profit = 727.25
:= by sorry

end NUMINAMATH_CALUDE_john_profit_l2517_251789


namespace NUMINAMATH_CALUDE_two_distinct_roots_iff_a_in_open_interval_l2517_251769

-- Define the logarithmic function
noncomputable def log_base (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the main theorem
theorem two_distinct_roots_iff_a_in_open_interval (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁ > 0 ∧ x₁ + a > 0 ∧ x₁ + a ≠ 1 ∧
    x₂ > 0 ∧ x₂ + a > 0 ∧ x₂ + a ≠ 1 ∧
    log_base (x₁ + a) (2 * x₁) = 2 ∧
    log_base (x₂ + a) (2 * x₂) = 2) ↔
  (0 < a ∧ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_iff_a_in_open_interval_l2517_251769


namespace NUMINAMATH_CALUDE_perimeter_difference_rectangles_l2517_251791

/-- Calculate the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Calculate the positive difference between two natural numbers -/
def positiveDifference (a b : ℕ) : ℕ :=
  max a b - min a b

theorem perimeter_difference_rectangles :
  positiveDifference (rectanglePerimeter 3 4) (rectanglePerimeter 1 8) = 4 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_rectangles_l2517_251791


namespace NUMINAMATH_CALUDE_base_k_theorem_l2517_251718

theorem base_k_theorem (k : ℕ) (h : k > 0) : 
  (1 * k^2 + 3 * k + 2 = 30) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_k_theorem_l2517_251718


namespace NUMINAMATH_CALUDE_line_equation_proof_l2517_251795

/-- Given a line that passes through the point (-2, 5) with a slope of -3/4,
    prove that its equation is 3x + 4y - 14 = 0 -/
theorem line_equation_proof (x y : ℝ) : 
  (y - 5 = -(3/4) * (x + 2)) ↔ (3*x + 4*y - 14 = 0) := by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2517_251795


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l2517_251778

theorem round_trip_average_speed (n : ℝ) : 
  let distance := n / 1000 -- distance in km
  let time_west := n / 30000 -- time for westward journey in hours
  let time_east := n / 3000 -- time for eastward journey in hours
  let time_wait := 0.5 -- waiting time in hours
  let total_distance := 2 * distance -- total round trip distance
  let total_time := time_west + time_east + time_wait -- total time for round trip
  total_distance / total_time = (60 * n) / (11 * n + 150000) := by
sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l2517_251778


namespace NUMINAMATH_CALUDE_unique_solution_equation_l2517_251700

theorem unique_solution_equation : ∃! x : ℝ, 3 * x + 3 * 12 + 3 * 13 + 11 = 134 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l2517_251700


namespace NUMINAMATH_CALUDE_propositions_analysis_l2517_251707

theorem propositions_analysis :
  (∃ (a b c : ℝ), a > b ∧ b > 0 ∧ a * c^2 ≤ b * c^2) ∧
  (∃ (a b : ℝ), a < b ∧ 1/a ≤ 1/b) ∧
  (∀ (a b : ℝ), a > b ∧ b > 0 → a^2 > a*b ∧ a*b > b^2) ∧
  (∀ (a b : ℝ), a > abs b → a^2 > b^2) :=
by sorry

end NUMINAMATH_CALUDE_propositions_analysis_l2517_251707


namespace NUMINAMATH_CALUDE_prob_B_winning_l2517_251756

/-- The probability of B winning in a chess game between A and B -/
theorem prob_B_winning (p_A_win p_draw : ℝ) 
  (h1 : p_A_win = 0.2)
  (h2 : p_draw = 0.5)
  (h3 : p_A_win + p_draw + (1 - p_A_win - p_draw) = 1) :
  1 - p_A_win - p_draw = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_prob_B_winning_l2517_251756


namespace NUMINAMATH_CALUDE_alpha_value_at_negative_four_l2517_251746

/-- Given that α is inversely proportional to β², prove that α = 5/4 when β = -4, 
    given that α = 5 when β = 2. -/
theorem alpha_value_at_negative_four (α β : ℝ) (k : ℝ) 
  (h1 : ∀ β, α * β^2 = k)  -- α is inversely proportional to β²
  (h2 : α = 5 ∧ β = 2 → k = 20)  -- α = 5 when β = 2
  : β = -4 → α = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_at_negative_four_l2517_251746


namespace NUMINAMATH_CALUDE_emily_quiz_score_l2517_251772

theorem emily_quiz_score (scores : List ℝ) (target_mean : ℝ) : 
  scores = [92, 95, 87, 89, 100] →
  target_mean = 93 →
  let new_score := 95
  let all_scores := scores ++ [new_score]
  (all_scores.sum / all_scores.length : ℝ) = target_mean := by
sorry


end NUMINAMATH_CALUDE_emily_quiz_score_l2517_251772


namespace NUMINAMATH_CALUDE_horner_evaluation_exclude_l2517_251728

def horner_polynomial (x : ℤ) : ℤ :=
  ((7 * x + 3) * x - 5) * x + 11

def horner_step1 (x : ℤ) : ℤ :=
  7 * x + 3

def horner_step2 (x : ℤ) : ℤ :=
  (7 * x + 3) * x - 5

theorem horner_evaluation_exclude (x : ℤ) :
  x = 23 →
  horner_polynomial x ≠ 85169 ∧
  horner_step1 x ≠ 85169 ∧
  horner_step2 x ≠ 85169 :=
by sorry

end NUMINAMATH_CALUDE_horner_evaluation_exclude_l2517_251728


namespace NUMINAMATH_CALUDE_unfactorable_polynomial_l2517_251786

theorem unfactorable_polynomial (b c d : ℤ) (h : Odd (b * d + c * d)) :
  ¬ ∃ (p q r : ℤ), ∀ (x : ℤ), x^3 + b*x^2 + c*x + d = (x + p) * (x^2 + q*x + r) :=
sorry

end NUMINAMATH_CALUDE_unfactorable_polynomial_l2517_251786


namespace NUMINAMATH_CALUDE_mindy_emails_l2517_251731

theorem mindy_emails (phone_messages : ℕ) (emails : ℕ) : 
  emails = 9 * phone_messages - 7 →
  emails + phone_messages = 93 →
  emails = 83 := by
sorry

end NUMINAMATH_CALUDE_mindy_emails_l2517_251731


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2517_251794

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Given a line with slope 4 passing through (-2, 5), prove m + b = 17 -/
theorem line_slope_intercept_sum (L : Line) 
  (slope_is_4 : L.m = 4)
  (passes_through : 5 = 4 * (-2) + L.b) : 
  L.m + L.b = 17 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2517_251794


namespace NUMINAMATH_CALUDE_prob_change_approx_point_54_l2517_251709

/-- The number of banks in the country of Alpha -/
def num_banks : ℕ := 5

/-- The initial probability of a bank closing -/
def initial_prob : ℝ := 0.05

/-- The probability of a bank closing after the crisis -/
def crisis_prob : ℝ := 0.25

/-- The probability that at least one bank will close -/
def prob_at_least_one_close (p : ℝ) : ℝ := 1 - (1 - p) ^ num_banks

/-- The change in probability of at least one bank closing -/
def prob_change : ℝ :=
  |prob_at_least_one_close crisis_prob - prob_at_least_one_close initial_prob|

/-- Theorem stating that the change in probability is approximately 0.54 -/
theorem prob_change_approx_point_54 :
  ∃ ε > 0, ε < 0.005 ∧ |prob_change - 0.54| < ε :=
sorry

end NUMINAMATH_CALUDE_prob_change_approx_point_54_l2517_251709


namespace NUMINAMATH_CALUDE_power_of_five_mod_thousand_l2517_251755

theorem power_of_five_mod_thousand : 5^1993 % 1000 = 125 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_mod_thousand_l2517_251755


namespace NUMINAMATH_CALUDE_birdseed_supply_l2517_251777

/-- Represents a box of birdseed -/
structure BirdseedBox where
  totalAmount : ℕ
  typeAAmount : ℕ
  typeBAmount : ℕ

/-- Represents a bird's weekly seed consumption -/
structure BirdConsumption where
  totalAmount : ℕ
  typeAPercentage : ℚ
  typeBPercentage : ℚ

/-- The problem statement -/
theorem birdseed_supply (pantryBoxes : List BirdseedBox)
  (parrot cockatiel canary : BirdConsumption) :
  pantryBoxes.length = 5 →
  (pantryBoxes.map (·.typeAAmount)).sum ≥ 650 →
  (pantryBoxes.map (·.typeBAmount)).sum ≥ 675 →
  parrot.totalAmount = 100 ∧ parrot.typeAPercentage = 3/5 ∧ parrot.typeBPercentage = 2/5 →
  cockatiel.totalAmount = 50 ∧ cockatiel.typeAPercentage = 1/2 ∧ cockatiel.typeBPercentage = 1/2 →
  canary.totalAmount = 25 ∧ canary.typeAPercentage = 2/5 ∧ canary.typeBPercentage = 3/5 →
  ∃ (weeks : ℕ), weeks ≥ 6 ∧
    (pantryBoxes.map (·.typeAAmount)).sum ≥ weeks * (parrot.totalAmount * parrot.typeAPercentage +
      cockatiel.totalAmount * cockatiel.typeAPercentage +
      canary.totalAmount * canary.typeAPercentage) ∧
    (pantryBoxes.map (·.typeBAmount)).sum ≥ weeks * (parrot.totalAmount * parrot.typeBPercentage +
      cockatiel.totalAmount * cockatiel.typeBPercentage +
      canary.totalAmount * canary.typeBPercentage) := by
  sorry


end NUMINAMATH_CALUDE_birdseed_supply_l2517_251777


namespace NUMINAMATH_CALUDE_butanoic_acid_molecular_weight_l2517_251780

/-- The molecular weight of one mole of Butanoic acid. -/
def molecular_weight_one_mole : ℝ := 88

/-- The number of moles given in the problem. -/
def num_moles : ℝ := 9

/-- The total molecular weight of the given number of moles. -/
def total_molecular_weight : ℝ := 792

/-- Theorem stating that the molecular weight of one mole of Butanoic acid is 88 g/mol,
    given that the molecular weight of 9 moles is 792. -/
theorem butanoic_acid_molecular_weight :
  molecular_weight_one_mole = total_molecular_weight / num_moles :=
by sorry

end NUMINAMATH_CALUDE_butanoic_acid_molecular_weight_l2517_251780


namespace NUMINAMATH_CALUDE_sandys_scooter_gain_percent_l2517_251740

/-- Calculates the gain percent for a transaction given purchase price, repair cost, and selling price -/
def gainPercent (purchasePrice repairCost sellingPrice : ℚ) : ℚ :=
  let totalCost := purchasePrice + repairCost
  let gain := sellingPrice - totalCost
  (gain / totalCost) * 100

/-- Theorem: The gain percent for Sandy's scooter transaction is 10% -/
theorem sandys_scooter_gain_percent :
  gainPercent 900 300 1320 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sandys_scooter_gain_percent_l2517_251740


namespace NUMINAMATH_CALUDE_power_product_result_l2517_251739

theorem power_product_result : (-1.5) ^ 2021 * (2/3) ^ 2023 = -(4/9) := by sorry

end NUMINAMATH_CALUDE_power_product_result_l2517_251739


namespace NUMINAMATH_CALUDE_polynomial_roots_arithmetic_progression_l2517_251745

/-- If a polynomial x^4 + jx^2 + kx + 256 has four distinct real roots in arithmetic progression, then j = -80 -/
theorem polynomial_roots_arithmetic_progression (j k : ℝ) : 
  (∃ (a b c d : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ 
    (∀ (x : ℝ), x^4 + j*x^2 + k*x + 256 = (x - a) * (x - b) * (x - c) * (x - d)) ∧
    (b - a = c - b) ∧ (c - b = d - c)) →
  j = -80 := by sorry

end NUMINAMATH_CALUDE_polynomial_roots_arithmetic_progression_l2517_251745


namespace NUMINAMATH_CALUDE_xyz_product_one_l2517_251734

theorem xyz_product_one (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5) (eq2 : y + 1/z = 2) (eq3 : z + 1/x = 3) :
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_one_l2517_251734


namespace NUMINAMATH_CALUDE_original_price_from_reduced_l2517_251704

/-- Given a shirt with a reduced price that is 25% of its original price,
    prove that if the reduced price is $6, then the original price was $24. -/
theorem original_price_from_reduced (reduced_price : ℝ) (original_price : ℝ) : 
  reduced_price = 6 → reduced_price = 0.25 * original_price → original_price = 24 := by
  sorry

end NUMINAMATH_CALUDE_original_price_from_reduced_l2517_251704


namespace NUMINAMATH_CALUDE_class_size_is_36_l2517_251714

/-- The number of students in a class, given boat seating conditions. -/
def number_of_students (b : ℕ) : Prop :=
  ∃ n : ℕ,
    n = 6 * (b + 1) ∧
    n = 9 * (b - 1)

/-- Theorem stating that the number of students is 36. -/
theorem class_size_is_36 :
  ∃ b : ℕ, number_of_students b ∧ (6 * (b + 1) = 36) :=
sorry

end NUMINAMATH_CALUDE_class_size_is_36_l2517_251714


namespace NUMINAMATH_CALUDE_pascal_triangle_34th_row_23rd_number_l2517_251767

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The 34th row of Pascal's triangle has 35 numbers -/
def row_length : ℕ := 35

/-- The row number (0-indexed) corresponding to a row with 35 numbers -/
def row_number : ℕ := row_length - 1

/-- The position (0-indexed) of the number we're looking for -/
def position : ℕ := 22

theorem pascal_triangle_34th_row_23rd_number :
  binomial row_number position = 64512240 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_34th_row_23rd_number_l2517_251767


namespace NUMINAMATH_CALUDE_function_maximum_value_l2517_251785

theorem function_maximum_value (x : ℝ) (h : x < 1/2) : 2*x + 1/(2*x - 1) ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_function_maximum_value_l2517_251785


namespace NUMINAMATH_CALUDE_mary_flour_added_l2517_251757

def recipe_flour : ℕ := 12
def recipe_salt : ℕ := 7
def extra_flour : ℕ := 3

theorem mary_flour_added (flour_added : ℕ) : 
  flour_added = recipe_flour - (recipe_salt + extra_flour) → flour_added = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_added_l2517_251757


namespace NUMINAMATH_CALUDE_balloon_distribution_l2517_251782

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 215) (h2 : num_friends = 9) :
  total_balloons % num_friends = 8 := by
sorry

end NUMINAMATH_CALUDE_balloon_distribution_l2517_251782


namespace NUMINAMATH_CALUDE_polynomial_pell_equation_l2517_251732

theorem polynomial_pell_equation (a b : ℤ) :
  (∃ (p q : ℝ → ℝ), ∀ x : ℝ, 
    (∃ (cp cq : ℤ → ℝ), (∀ n : ℤ, p x = cp n * x^n) ∧ (∀ n : ℤ, q x = cq n * x^n)) ∧ 
    q ≠ 0 ∧ 
    (p x)^2 - (x^2 + a*x + b) * (q x)^2 = 1) ↔ 
  (a % 2 = 1 ∧ b = (a^2 - 1) / 4) ∨ 
  (a % 2 = 0 ∧ ∃ k : ℤ, b = a^2 / 4 + k ∧ 2 % k = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_pell_equation_l2517_251732


namespace NUMINAMATH_CALUDE_equation_equivalence_l2517_251779

theorem equation_equivalence (x : ℝ) : x^2 - 6*x + 5 = 0 ↔ (x - 3)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2517_251779


namespace NUMINAMATH_CALUDE_vectors_perpendicular_l2517_251762

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 1)

theorem vectors_perpendicular : a.1 * b.1 + a.2 * b.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_l2517_251762


namespace NUMINAMATH_CALUDE_real_number_inequalities_l2517_251720

-- Define the propositions
theorem real_number_inequalities (a b c : ℝ) : 
  -- Proposition A
  ((a * c^2 > b * c^2) → (a > b)) ∧ 
  -- Proposition B (negation)
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) ∧ 
  -- Proposition C (negation)
  (∃ a b : ℝ, a > b ∧ 1/a ≥ 1/b) ∧ 
  -- Proposition D
  ((a > b ∧ b > 0) → (a^2 > a*b ∧ a*b > b^2)) :=
by sorry

end NUMINAMATH_CALUDE_real_number_inequalities_l2517_251720


namespace NUMINAMATH_CALUDE_graduation_photo_arrangements_l2517_251742

/-- The number of students in the class -/
def num_students : ℕ := 6

/-- The total number of people (students + teacher) -/
def total_people : ℕ := num_students + 1

/-- The number of arrangements with the teacher in the middle -/
def total_arrangements : ℕ := (num_students.factorial)

/-- The number of arrangements with the teacher in the middle and students A and B adjacent -/
def adjacent_arrangements : ℕ := 4 * 2 * ((num_students - 2).factorial)

/-- The number of valid arrangements -/
def valid_arrangements : ℕ := total_arrangements - adjacent_arrangements

theorem graduation_photo_arrangements :
  valid_arrangements = 528 := by sorry

end NUMINAMATH_CALUDE_graduation_photo_arrangements_l2517_251742


namespace NUMINAMATH_CALUDE_scooter_initial_cost_l2517_251719

/-- Proves that the initial cost of a scooter is $900 given the conditions of the problem -/
theorem scooter_initial_cost (initial_cost : ℝ) : 
  (∃ (total_cost : ℝ), 
    total_cost = initial_cost + 300 ∧ 
    1500 = 1.25 * total_cost) → 
  initial_cost = 900 :=
by sorry

end NUMINAMATH_CALUDE_scooter_initial_cost_l2517_251719


namespace NUMINAMATH_CALUDE_log_exponent_simplification_l2517_251729

theorem log_exponent_simplification :
  Real.log 2 + Real.log 5 - 42 * (8 ^ (1/4 : ℝ)) - (2017 ^ (0 : ℝ)) = -2 :=
by sorry

end NUMINAMATH_CALUDE_log_exponent_simplification_l2517_251729


namespace NUMINAMATH_CALUDE_house_rent_expenditure_l2517_251713

/-- Given a person's income and expenditure pattern, calculate their house rent expense -/
theorem house_rent_expenditure (income : ℝ) (petrol_expense : ℝ) :
  petrol_expense = 0.3 * income →
  petrol_expense = 300 →
  let remaining_income := income - petrol_expense
  let house_rent := 0.2 * remaining_income
  house_rent = 140 := by
  sorry

end NUMINAMATH_CALUDE_house_rent_expenditure_l2517_251713


namespace NUMINAMATH_CALUDE_philip_paintings_l2517_251799

/-- Calculates the total number of paintings Philip will have after a given number of days -/
def total_paintings (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + paintings_per_day * days

/-- Theorem: Philip will have 80 paintings after 30 days -/
theorem philip_paintings :
  total_paintings 2 20 30 = 80 := by
  sorry

end NUMINAMATH_CALUDE_philip_paintings_l2517_251799


namespace NUMINAMATH_CALUDE_derivative_zero_in_interval_l2517_251781

theorem derivative_zero_in_interval (n : ℕ) (f : ℝ → ℝ) 
  (h_diff : ContDiff ℝ (n + 1) f)
  (h_f_zero : f 1 = 0 ∧ f 0 = 0)
  (h_derivatives_zero : ∀ k : ℕ, k ≤ n → (deriv^[k] f) 0 = 0) :
  ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ (deriv^[n + 1] f) x = 0 := by
sorry

end NUMINAMATH_CALUDE_derivative_zero_in_interval_l2517_251781


namespace NUMINAMATH_CALUDE_special_function_properties_l2517_251721

/-- A function satisfying specific properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, ∀ y > 0, f (x * y) = f x + f y) ∧
  (∀ x > 1, f x < 0) ∧
  (f 3 = -1)

theorem special_function_properties
  (f : ℝ → ℝ)
  (hf : SpecialFunction f) :
  f 1 = 0 ∧
  f (1/9) = 2 ∧
  (∀ x y, x > 0 → y > 0 → x < y → f y < f x) ∧
  (∀ x, f x + f (2 - x) < 2 ↔ 1 - 2 * Real.sqrt 2 / 3 < x ∧ x < 1 + 2 * Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l2517_251721


namespace NUMINAMATH_CALUDE_range_of_k_l2517_251776

theorem range_of_k (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + c^2 = 16) (h2 : b^2 + c^2 = 25) :
  let k := a^2 + b^2
  9 < k ∧ k < 41 := by sorry

end NUMINAMATH_CALUDE_range_of_k_l2517_251776


namespace NUMINAMATH_CALUDE_sqrt_difference_comparison_l2517_251712

theorem sqrt_difference_comparison 
  (a b m : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hm : m > 0) 
  (hab : a > b) : 
  Real.sqrt (b + m) - Real.sqrt b > Real.sqrt (a + m) - Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_comparison_l2517_251712


namespace NUMINAMATH_CALUDE_ellipse_equation_from_shared_focus_l2517_251710

/-- Given a parabola and an ellipse with a shared focus, prove the equation of the ellipse -/
theorem ellipse_equation_from_shared_focus (a : ℝ) (h_a : a > 0) :
  (∃ (x y : ℝ), y^2 = 8*x ∧ x^2/a^2 + y^2 = 1 ∧ x = 2) →
  (∀ (x y : ℝ), x^2/8 + y^2/4 = 1 ↔ x^2/a^2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_shared_focus_l2517_251710


namespace NUMINAMATH_CALUDE_kids_difference_l2517_251754

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 18) 
  (h2 : tuesday = 10) : 
  monday - tuesday = 8 := by
sorry

end NUMINAMATH_CALUDE_kids_difference_l2517_251754


namespace NUMINAMATH_CALUDE_cherry_pitting_time_l2517_251725

/-- Proves that it takes 2 hours to pit cherries for a pie given the specified conditions -/
theorem cherry_pitting_time :
  ∀ (pounds_needed : ℕ) 
    (cherries_per_pound : ℕ) 
    (cherries_per_set : ℕ) 
    (minutes_per_set : ℕ),
  pounds_needed = 3 →
  cherries_per_pound = 80 →
  cherries_per_set = 20 →
  minutes_per_set = 10 →
  (pounds_needed * cherries_per_pound * minutes_per_set) / 
  (cherries_per_set * 60) = 2 := by
sorry

end NUMINAMATH_CALUDE_cherry_pitting_time_l2517_251725


namespace NUMINAMATH_CALUDE_gcd_153_119_l2517_251748

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_153_119_l2517_251748


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2517_251741

theorem fraction_equivalence : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 :=
by
  use 13 / 2
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2517_251741


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l2517_251711

/-- A quadratic function with a negative leading coefficient -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_neg : a < 0

/-- The function f(x) -/
def f (qf : QuadraticFunction) (x : ℝ) : ℝ := qf.a * x^2 + qf.b * x + qf.c

/-- The condition that 1 and 3 are roots of y = f(x) + 2x -/
def roots_condition (qf : QuadraticFunction) : Prop :=
  f qf 1 + 2 * 1 = 0 ∧ f qf 3 + 2 * 3 = 0

/-- The condition that f(x) + 6a = 0 has two equal roots -/
def equal_roots_condition (qf : QuadraticFunction) : Prop :=
  ∃ (x : ℝ), f qf x + 6 * qf.a = 0 ∧ 
  ∀ (y : ℝ), f qf y + 6 * qf.a = 0 → y = x

/-- The theorem statement -/
theorem unique_quadratic_function :
  ∃! (qf : QuadraticFunction),
    roots_condition qf ∧
    equal_roots_condition qf ∧
    qf.a = -1/4 ∧ qf.b = -1 ∧ qf.c = -3/4 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_l2517_251711


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l2517_251775

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N + 2) * (N + 1)) / Nat.factorial (N + 3) = (N + 1) / (N + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l2517_251775


namespace NUMINAMATH_CALUDE_proportion_solution_l2517_251708

theorem proportion_solution (x : ℝ) (h : (3/4) / x = 5/6) : x = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2517_251708


namespace NUMINAMATH_CALUDE_product_of_reals_l2517_251752

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 8) (cube_sum_eq : a^3 + b^3 = 170) : a * b = 21.375 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reals_l2517_251752


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2517_251766

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + y' = 1 → 1/(2*x') + 1/y' ≥ 1/(2*x) + 1/y) →
  1/(2*x) + 1/y = 3/2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2517_251766


namespace NUMINAMATH_CALUDE_book_cost_price_l2517_251793

/-- Given a book sold for Rs 90 with a profit rate of 80%, prove that the cost price is Rs 50. -/
theorem book_cost_price (selling_price : ℝ) (profit_rate : ℝ) (h1 : selling_price = 90) (h2 : profit_rate = 80) :
  ∃ (cost_price : ℝ), cost_price = 50 ∧ profit_rate / 100 = (selling_price - cost_price) / cost_price :=
by sorry

end NUMINAMATH_CALUDE_book_cost_price_l2517_251793


namespace NUMINAMATH_CALUDE_dara_waiting_time_l2517_251702

/-- Represents a person's age and employment status -/
structure Person where
  age : ℕ
  employed : Bool

/-- The minimum age required for employment -/
def min_employment_age : ℕ := 25

/-- Calculates the age of a person after a given number of years -/
def age_after (p : Person) (years : ℕ) : ℕ := p.age + years

/-- Jane's current state -/
def jane : Person := { age := 28, employed := true }

/-- Dara's current age -/
def dara_age : ℕ := jane.age + 6 - 2 * (jane.age + 6 - min_employment_age)

/-- Time Dara needs to wait to reach the minimum employment age -/
def waiting_time : ℕ := min_employment_age - dara_age

theorem dara_waiting_time :
  waiting_time = 14 := by sorry

end NUMINAMATH_CALUDE_dara_waiting_time_l2517_251702


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2517_251787

theorem fraction_multiplication : (1/2 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * 72 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2517_251787


namespace NUMINAMATH_CALUDE_positive_operation_l2517_251773

theorem positive_operation : 
  ((-1 : ℝ)^2 > 0) ∧ 
  (-(|-2|) ≤ 0) ∧ 
  (0 * (-3) = 0) ∧ 
  (-(3^2) < 0) := by
sorry

end NUMINAMATH_CALUDE_positive_operation_l2517_251773


namespace NUMINAMATH_CALUDE_expand_difference_of_squares_l2517_251744

theorem expand_difference_of_squares (a : ℝ) : (a + 2) * (2 - a) = 4 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_difference_of_squares_l2517_251744


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l2517_251736

/-- The speed of a man in still water, given his downstream and upstream swimming times and distances. -/
theorem mans_speed_in_still_water 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (h1 : downstream_distance = 48) 
  (h2 : upstream_distance = 18) 
  (h3 : downstream_time = 3) 
  (h4 : upstream_time = 3) : 
  ∃ (speed : ℝ), speed = 11 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l2517_251736


namespace NUMINAMATH_CALUDE_steven_pears_count_l2517_251760

/-- The number of seeds Steven needs to collect -/
def total_seeds : ℕ := 60

/-- The average number of seeds in an apple -/
def apple_seeds : ℕ := 6

/-- The average number of seeds in a pear -/
def pear_seeds : ℕ := 2

/-- The average number of seeds in a grape -/
def grape_seeds : ℕ := 3

/-- The number of apples Steven has set aside -/
def apples_set_aside : ℕ := 4

/-- The number of grapes Steven has set aside -/
def grapes_set_aside : ℕ := 9

/-- The number of additional seeds Steven needs -/
def additional_seeds_needed : ℕ := 3

/-- The number of pears Steven has set aside -/
def pears_set_aside : ℕ := 3

theorem steven_pears_count :
  pears_set_aside * pear_seeds + 
  apples_set_aside * apple_seeds + 
  grapes_set_aside * grape_seeds = 
  total_seeds - additional_seeds_needed :=
by sorry

end NUMINAMATH_CALUDE_steven_pears_count_l2517_251760


namespace NUMINAMATH_CALUDE_nested_f_result_l2517_251768

def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

theorem nested_f_result (p q : ℝ) :
  (∀ x ∈ Set.Icc 1 3, |f p q x| ≤ 1/2) →
  (f p q)^[2017] ((3 + Real.sqrt 7) / 2) = (3 - Real.sqrt 7) / 2 :=
by sorry

end NUMINAMATH_CALUDE_nested_f_result_l2517_251768


namespace NUMINAMATH_CALUDE_equality_of_coefficients_l2517_251715

theorem equality_of_coefficients (a b c : ℝ) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c ≥ b * x^2 + c * x + a ∧ 
                b * x^2 + c * x + a ≥ c * x^2 + a * x + b) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equality_of_coefficients_l2517_251715


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l2517_251798

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → 
    ¬(∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m)) ∧
  n = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l2517_251798


namespace NUMINAMATH_CALUDE_max_boxes_in_wooden_box_l2517_251753

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ :=
  m * 100

theorem max_boxes_in_wooden_box :
  let largeBox : BoxDimensions := {
    length := metersToCentimeters 8,
    width := metersToCentimeters 10,
    height := metersToCentimeters 6
  }
  let smallBox : BoxDimensions := {
    length := 4,
    width := 5,
    height := 6
  }
  (boxVolume largeBox) / (boxVolume smallBox) = 4000000 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_in_wooden_box_l2517_251753


namespace NUMINAMATH_CALUDE_odd_factors_of_360_is_6_l2517_251730

/-- The number of odd factors of 360 -/
def odd_factors_of_360 : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem: The number of odd factors of 360 is 6 -/
theorem odd_factors_of_360_is_6 : odd_factors_of_360 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_factors_of_360_is_6_l2517_251730


namespace NUMINAMATH_CALUDE_min_packs_for_120_cans_l2517_251724

/-- Represents the available pack sizes for soda cans -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans in a given pack size -/
def cansInPack (size : PackSize) : ℕ :=
  match size with
  | .small => 8
  | .medium => 16
  | .large => 32

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (combo : PackCombination) : ℕ :=
  combo.small * cansInPack PackSize.small +
  combo.medium * cansInPack PackSize.medium +
  combo.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a pack combination -/
def totalPacks (combo : PackCombination) : ℕ :=
  combo.small + combo.medium + combo.large

/-- Checks if a pack combination is valid for the given total cans -/
def isValidCombination (combo : PackCombination) (totalCansNeeded : ℕ) : Prop :=
  totalCans combo = totalCansNeeded

/-- Theorem: The minimum number of packs needed to buy exactly 120 cans of soda is 5 -/
theorem min_packs_for_120_cans :
  ∃ (minCombo : PackCombination),
    isValidCombination minCombo 120 ∧
    totalPacks minCombo = 5 ∧
    ∀ (combo : PackCombination),
      isValidCombination combo 120 → totalPacks combo ≥ totalPacks minCombo := by
  sorry

end NUMINAMATH_CALUDE_min_packs_for_120_cans_l2517_251724


namespace NUMINAMATH_CALUDE_system_solution_l2517_251751

theorem system_solution (x y z : ℝ) : 
  (x + y + z = 1 ∧ x^3 + y^3 + z^3 = 1 ∧ x*y*z = -16) ↔ 
  ((x = 1 ∧ y = 4 ∧ z = -4) ∨
   (x = 1 ∧ y = -4 ∧ z = 4) ∨
   (x = 4 ∧ y = 1 ∧ z = -4) ∨
   (x = 4 ∧ y = -4 ∧ z = 1) ∨
   (x = -4 ∧ y = 1 ∧ z = 4) ∨
   (x = -4 ∧ y = 4 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2517_251751
