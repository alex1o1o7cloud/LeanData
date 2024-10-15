import Mathlib

namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2052_205268

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (a*x + b)^2) ↔ k = 100 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2052_205268


namespace NUMINAMATH_CALUDE_gift_purchase_probability_is_correct_l2052_205295

/-- The probability of purchasing gifts from all three stores and still having money left -/
def gift_purchase_probability : ℚ :=
  let initial_amount : ℕ := 5000
  let num_stores : ℕ := 3
  let prices : List ℕ := [1000, 1500, 2000]
  let total_combinations : ℕ := 3^num_stores
  let favorable_cases : ℕ := 17
  favorable_cases / total_combinations

/-- Theorem stating the probability of successful gift purchases -/
theorem gift_purchase_probability_is_correct :
  gift_purchase_probability = 17 / 27 := by sorry

end NUMINAMATH_CALUDE_gift_purchase_probability_is_correct_l2052_205295


namespace NUMINAMATH_CALUDE_school_dance_attendance_l2052_205236

theorem school_dance_attendance (P : ℕ) : 
  (P * 10 / 100 = P / 10) →  -- 10% of P are faculty and staff
  (P * 90 / 100 = P * 9 / 10) →  -- 90% of P are students
  ((P * 9 / 10) * 2 / 3 = (P * 9 / 10) - 30) →  -- Two-thirds of students are girls
  ((P * 9 / 10) * 1 / 3 = 30) →  -- One-third of students are boys
  P = 100 := by sorry

end NUMINAMATH_CALUDE_school_dance_attendance_l2052_205236


namespace NUMINAMATH_CALUDE_cylinder_in_hemisphere_height_l2052_205271

theorem cylinder_in_hemisphere_height (r c h : ℝ) : 
  r > 0 ∧ c > 0 ∧ r > c ∧ r = 8 ∧ c = 3 → h = Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_in_hemisphere_height_l2052_205271


namespace NUMINAMATH_CALUDE_square_sum_nonzero_iff_exists_nonzero_l2052_205260

theorem square_sum_nonzero_iff_exists_nonzero (a b : ℝ) : 
  a^2 + b^2 ≠ 0 ↔ a ≠ 0 ∨ b ≠ 0 := by sorry

end NUMINAMATH_CALUDE_square_sum_nonzero_iff_exists_nonzero_l2052_205260


namespace NUMINAMATH_CALUDE_distribute_5_3_l2052_205281

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^5 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 3^5 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l2052_205281


namespace NUMINAMATH_CALUDE_profit_starts_in_third_year_option1_more_cost_effective_l2052_205280

-- Define the constants
def initial_cost : ℕ := 980000
def first_year_expenses : ℕ := 120000
def yearly_expense_increase : ℕ := 40000
def annual_income : ℕ := 500000

-- Define a function to calculate expenses for a given year
def expenses (year : ℕ) : ℕ :=
  first_year_expenses + (year - 1) * yearly_expense_increase

-- Define a function to calculate cumulative profit for a given year
def cumulative_profit (year : ℕ) : ℤ :=
  year * annual_income - (initial_cost + (Finset.range year).sum (λ i => expenses (i + 1)))

-- Define a function to calculate average profit for a given year
def average_profit (year : ℕ) : ℚ :=
  (cumulative_profit year : ℚ) / year

-- Theorem 1: The company starts to make a profit in the third year
theorem profit_starts_in_third_year :
  cumulative_profit 3 > 0 ∧ ∀ y : ℕ, y < 3 → cumulative_profit y ≤ 0 := by sorry

-- Define the selling prices for the two options
def option1_price : ℕ := 260000
def option2_price : ℕ := 80000

-- Theorem 2: Option 1 is more cost-effective than Option 2
theorem option1_more_cost_effective :
  ∃ y1 y2 : ℕ,
    (∀ y : ℕ, average_profit y ≤ average_profit y1) ∧
    (∀ y : ℕ, cumulative_profit y ≤ cumulative_profit y2) ∧
    option1_price + cumulative_profit y1 > option2_price + cumulative_profit y2 := by sorry

end NUMINAMATH_CALUDE_profit_starts_in_third_year_option1_more_cost_effective_l2052_205280


namespace NUMINAMATH_CALUDE_equation_solution_l2052_205279

theorem equation_solution :
  ∀ y : ℝ, (45 : ℝ) / 75 = Real.sqrt (y / 25) → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2052_205279


namespace NUMINAMATH_CALUDE_bent_polygon_total_angle_l2052_205257

/-- For a regular polygon with n sides (n > 4), if each side is bent inward at an angle θ = 360°/(2n),
    then the total angle formed by all the bends is 180°. -/
theorem bent_polygon_total_angle (n : ℕ) (h : n > 4) :
  let θ : ℝ := 360 / (2 * n)
  n * θ = 180 := by sorry

end NUMINAMATH_CALUDE_bent_polygon_total_angle_l2052_205257


namespace NUMINAMATH_CALUDE_quadrilaterals_in_100gon_l2052_205285

/-- A regular polygon with n vertices -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A coloring of vertices of a regular polygon -/
def Coloring (n : ℕ) := Fin n → Bool

/-- A convex quadrilateral formed by four vertices of a regular polygon -/
structure Quadrilateral (n : ℕ) where
  v1 : Fin n
  v2 : Fin n
  v3 : Fin n
  v4 : Fin n

/-- Check if two quadrilaterals are disjoint -/
def are_disjoint (n : ℕ) (q1 q2 : Quadrilateral n) : Prop :=
  q1.v1 ≠ q2.v1 ∧ q1.v1 ≠ q2.v2 ∧ q1.v1 ≠ q2.v3 ∧ q1.v1 ≠ q2.v4 ∧
  q1.v2 ≠ q2.v1 ∧ q1.v2 ≠ q2.v2 ∧ q1.v2 ≠ q2.v3 ∧ q1.v2 ≠ q2.v4 ∧
  q1.v3 ≠ q2.v1 ∧ q1.v3 ≠ q2.v2 ∧ q1.v3 ≠ q2.v3 ∧ q1.v3 ≠ q2.v4 ∧
  q1.v4 ≠ q2.v1 ∧ q1.v4 ≠ q2.v2 ∧ q1.v4 ≠ q2.v3 ∧ q1.v4 ≠ q2.v4

/-- Check if a quadrilateral has three corners of one color and one of the other -/
def has_three_one_coloring (n : ℕ) (q : Quadrilateral n) (c : Coloring n) : Prop :=
  (c q.v1 = c q.v2 ∧ c q.v2 = c q.v3 ∧ c q.v3 ≠ c q.v4) ∨
  (c q.v1 = c q.v2 ∧ c q.v2 = c q.v4 ∧ c q.v4 ≠ c q.v3) ∨
  (c q.v1 = c q.v3 ∧ c q.v3 = c q.v4 ∧ c q.v4 ≠ c q.v2) ∨
  (c q.v2 = c q.v3 ∧ c q.v3 = c q.v4 ∧ c q.v4 ≠ c q.v1)

/-- The main theorem -/
theorem quadrilaterals_in_100gon :
  ∃ (p : RegularPolygon 100) (c : Coloring 100) (qs : Fin 24 → Quadrilateral 100),
    (∀ i : Fin 100, c i = true → (∃ j : Fin 41, true)) ∧  -- 41 black vertices
    (∀ i : Fin 100, c i = false → (∃ j : Fin 59, true)) ∧  -- 59 white vertices
    (∀ i j : Fin 24, i ≠ j → are_disjoint 100 (qs i) (qs j)) ∧
    (∀ i : Fin 24, has_three_one_coloring 100 (qs i) c) :=
by sorry

end NUMINAMATH_CALUDE_quadrilaterals_in_100gon_l2052_205285


namespace NUMINAMATH_CALUDE_initial_orchids_is_three_l2052_205266

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  initialRoses : ℕ
  finalRoses : ℕ
  finalOrchids : ℕ
  orchidsCut : ℕ

/-- Calculates the initial number of orchids in the vase -/
def initialOrchids (v : FlowerVase) : ℕ :=
  v.finalOrchids - v.orchidsCut

/-- Theorem stating that the initial number of orchids is 3 -/
theorem initial_orchids_is_three (v : FlowerVase) 
  (h1 : v.initialRoses = 16)
  (h2 : v.finalRoses = 13)
  (h3 : v.finalOrchids = 7)
  (h4 : v.orchidsCut = 4) : 
  initialOrchids v = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_orchids_is_three_l2052_205266


namespace NUMINAMATH_CALUDE_probability_under_20_is_7_16_l2052_205223

/-- Represents a group of people with age categories --/
structure AgeGroup where
  total : ℕ
  over30 : ℕ
  under20 : ℕ
  h1 : over30 + under20 = total

/-- The probability of selecting a person under 20 years old --/
def probabilityUnder20 (group : AgeGroup) : ℚ :=
  group.under20 / group.total

theorem probability_under_20_is_7_16 (group : AgeGroup) 
  (h2 : group.total = 160) 
  (h3 : group.over30 = 90) : 
  probabilityUnder20 group = 7 / 16 := by
  sorry

#check probability_under_20_is_7_16

end NUMINAMATH_CALUDE_probability_under_20_is_7_16_l2052_205223


namespace NUMINAMATH_CALUDE_solution_set_f_geq_6_range_of_a_for_nonempty_solution_l2052_205255

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 2|

-- Theorem for the first part of the problem
theorem solution_set_f_geq_6 :
  {x : ℝ | f x ≥ 6} = {x : ℝ | x ≤ -3 ∨ x ≥ 3} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_nonempty_solution :
  ∀ a : ℝ, (∃ x : ℝ, f x < a + x) ↔ a > 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_6_range_of_a_for_nonempty_solution_l2052_205255


namespace NUMINAMATH_CALUDE_probability_monotonic_increasing_l2052_205251

def cube_faces : Finset ℤ := {-2, -1, 0, 1, 2, 3}

def is_monotonic_increasing (a b : ℤ) : Prop :=
  a ≥ 0 ∧ b ≥ 0

def favorable_outcomes : Finset (ℤ × ℤ) :=
  (cube_faces.filter (λ x => x ≥ 0)).product (cube_faces.filter (λ x => x ≥ 0))

def total_outcomes : Finset (ℤ × ℤ) :=
  cube_faces.product cube_faces

theorem probability_monotonic_increasing :
  (favorable_outcomes.card : ℚ) / (total_outcomes.card : ℚ) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_monotonic_increasing_l2052_205251


namespace NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l2052_205283

theorem factorization_of_2a_squared_minus_8 (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l2052_205283


namespace NUMINAMATH_CALUDE_quadrilateral_angle_sum_l2052_205208

structure Quadrilateral where
  diagonals_intersect : Bool
  intersection_not_on_side : Bool

def sum_of_angles (q : Quadrilateral) : ℝ :=
  if q.diagonals_intersect ∧ q.intersection_not_on_side then 720 else 0

theorem quadrilateral_angle_sum (q : Quadrilateral) 
  (h1 : q.diagonals_intersect = true) 
  (h2 : q.intersection_not_on_side = true) : 
  sum_of_angles q = 720 := by
  sorry

#check quadrilateral_angle_sum

end NUMINAMATH_CALUDE_quadrilateral_angle_sum_l2052_205208


namespace NUMINAMATH_CALUDE_valid_numbers_l2052_205222

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  let tens := n / 10
  let ones := n % 10
  2 * (tens + ones) = tens * ones

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {63, 44, 36} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l2052_205222


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l2052_205202

theorem difference_of_squares_special_case : (2 + Real.sqrt 2) * (2 - Real.sqrt 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l2052_205202


namespace NUMINAMATH_CALUDE_raven_age_is_55_l2052_205209

-- Define the current ages
def phoebe_age : ℕ := 10
def raven_age : ℕ := 55

-- Define the conditions
def condition1 : Prop := raven_age + 5 = 4 * (phoebe_age + 5)
def condition2 : Prop := phoebe_age = 10
def condition3 : Prop := ∃ sam_age : ℕ, sam_age = 2 * ((raven_age + 3) - (phoebe_age + 3))

-- Theorem statement
theorem raven_age_is_55 : 
  condition1 ∧ condition2 ∧ condition3 → raven_age = 55 :=
by sorry

end NUMINAMATH_CALUDE_raven_age_is_55_l2052_205209


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2052_205299

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 82 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 16)
  (h2 : a * b + c + d = 81)
  (h3 : a * d + b * c = 168)
  (h4 : c * d = 100) :
  ∀ (w x y z : ℝ), 
  (w + x = 16) → 
  (w * x + y + z = 81) → 
  (w * z + x * y = 168) → 
  (y * z = 100) → 
  a^2 + b^2 + c^2 + d^2 ≥ w^2 + x^2 + y^2 + z^2 ∧
  a^2 + b^2 + c^2 + d^2 ≤ 82 :=
by
  sorry

#check max_sum_of_squares

end NUMINAMATH_CALUDE_max_sum_of_squares_l2052_205299


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2052_205293

-- Define the polynomials
def p (z : ℝ) : ℝ := 3 * z^2 - 4 * z + 1
def q (z : ℝ) : ℝ := 4 * z^3 + z^2 - 5 * z + 3

-- State the theorem
theorem polynomial_multiplication :
  ∀ z : ℝ, p z * q z = 12 * z^5 + 3 * z^4 + 32 * z^3 + z^2 - 7 * z + 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2052_205293


namespace NUMINAMATH_CALUDE_solve_for_a_l2052_205248

-- Define the operation *
def star (a b : ℝ) : ℝ := 2 * a - b^2

-- Theorem statement
theorem solve_for_a : ∃ a : ℝ, star a 3 = 15 ∧ a = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2052_205248


namespace NUMINAMATH_CALUDE_total_pizza_cost_l2052_205206

def pizza_cost : ℕ := 8
def number_of_pizzas : ℕ := 3

theorem total_pizza_cost : pizza_cost * number_of_pizzas = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_cost_l2052_205206


namespace NUMINAMATH_CALUDE_marlas_grid_squares_per_row_l2052_205264

/-- Represents a grid with colored squares -/
structure ColoredGrid where
  rows : ℕ
  squaresPerRow : ℕ
  redSquares : ℕ
  blueRows : ℕ
  greenSquares : ℕ

/-- The number of squares in each row of Marla's grid -/
def marlasGridSquaresPerRow : ℕ := 15

/-- Theorem stating that Marla's grid has 15 squares per row -/
theorem marlas_grid_squares_per_row :
  ∃ (g : ColoredGrid),
    g.rows = 10 ∧
    g.redSquares = 24 ∧
    g.blueRows = 4 ∧
    g.greenSquares = 66 ∧
    g.squaresPerRow = marlasGridSquaresPerRow :=
by sorry


end NUMINAMATH_CALUDE_marlas_grid_squares_per_row_l2052_205264


namespace NUMINAMATH_CALUDE_members_not_playing_specific_club_l2052_205243

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  badminton : ℕ
  tennis : ℕ
  both : ℕ

/-- The number of members who don't play either badminton or tennis -/
def members_not_playing (club : SportsClub) : ℕ :=
  club.total - (club.badminton + club.tennis - club.both)

/-- Theorem stating the number of members not playing either sport in the given scenario -/
theorem members_not_playing_specific_club :
  let club : SportsClub := {
    total := 30,
    badminton := 17,
    tennis := 19,
    both := 8
  }
  members_not_playing club = 2 := by
  sorry

end NUMINAMATH_CALUDE_members_not_playing_specific_club_l2052_205243


namespace NUMINAMATH_CALUDE_dogwood_trees_in_park_l2052_205242

theorem dogwood_trees_in_park (current_trees : ℕ) : current_trees = 34 :=
  by
  have h1 : current_trees + 49 = 83 := by sorry
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_in_park_l2052_205242


namespace NUMINAMATH_CALUDE_square_sum_implies_product_l2052_205237

theorem square_sum_implies_product (m : ℝ) 
  (h : (m - 2023)^2 + (2024 - m)^2 = 2025) : 
  (m - 2023) * (2024 - m) = -1012 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_implies_product_l2052_205237


namespace NUMINAMATH_CALUDE_ratio_of_fifth_terms_in_arithmetic_sequences_l2052_205286

/-- Given two arithmetic sequences, prove the ratio of their 5th terms -/
theorem ratio_of_fifth_terms_in_arithmetic_sequences 
  (a b : ℕ → ℚ) 
  (h_arithmetic_a : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_arithmetic_b : ∀ n, b (n + 1) - b n = b 1 - b 0)
  (h_ratio : ∀ n, (n * (a 0 + a n)) / (n * (b 0 + b n)) = (3 * n) / (2 * n + 9)) :
  a 5 / b 5 = 15 / 19 := by
sorry


end NUMINAMATH_CALUDE_ratio_of_fifth_terms_in_arithmetic_sequences_l2052_205286


namespace NUMINAMATH_CALUDE_distinct_z_values_l2052_205259

/-- Given two integers x and y where:
    1. 200 ≤ x ≤ 999
    2. 100 ≤ y ≤ 999
    3. y is the number formed by reversing the digits of x
    4. z = x + y
    This theorem states that there are exactly 1878 distinct possible values for z. -/
theorem distinct_z_values (x y z : ℕ) 
  (hx : 200 ≤ x ∧ x ≤ 999)
  (hy : 100 ≤ y ∧ y ≤ 999)
  (hrev : y = 100 * (x % 10) + 10 * ((x / 10) % 10) + (x / 100))
  (hz : z = x + y) :
  ∃! (s : Finset ℕ), s = {z | ∃ (x y : ℕ), 
    200 ≤ x ∧ x ≤ 999 ∧
    100 ≤ y ∧ y ≤ 999 ∧
    y = 100 * (x % 10) + 10 * ((x / 10) % 10) + (x / 100) ∧
    z = x + y} ∧ 
  Finset.card s = 1878 :=
by sorry

end NUMINAMATH_CALUDE_distinct_z_values_l2052_205259


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2052_205231

def M : Set ℕ := {0, 1}
def N : Set ℕ := {1, 2}

theorem union_of_M_and_N : M ∪ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2052_205231


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2052_205284

theorem arithmetic_mean_of_fractions : 
  (3/8 + 5/9) / 2 = 67/144 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2052_205284


namespace NUMINAMATH_CALUDE_william_car_wash_body_time_l2052_205227

/-- Represents the time William spends washing vehicles -/
def WilliamCarWash :=
  {time_body : ℕ //
    ∃ (time_normal time_suv : ℕ),
      time_normal = time_body + 17 ∧
      time_suv = 2 * time_normal ∧
      2 * time_normal + time_suv = 96}

/-- Theorem stating that William spends 7 minutes washing the car body -/
theorem william_car_wash_body_time :
  ∀ w : WilliamCarWash, w.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_william_car_wash_body_time_l2052_205227


namespace NUMINAMATH_CALUDE_triangle_equation_no_real_roots_l2052_205246

theorem triangle_equation_no_real_roots 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) : 
  ∀ x : ℝ, a^2 * x^2 - (c^2 - a^2 - b^2) * x + b^2 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_equation_no_real_roots_l2052_205246


namespace NUMINAMATH_CALUDE_largest_b_value_l2052_205217

theorem largest_b_value (b : ℚ) (h : (3 * b + 7) * (b - 2) = 8 * b) : 
  b ≤ 7 / 2 ∧ ∃ (b₀ : ℚ), (3 * b₀ + 7) * (b₀ - 2) = 8 * b₀ ∧ b₀ = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_b_value_l2052_205217


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2052_205245

/-- Given a curve y = x^3 + ax + b and a line y = kx + 1 that is tangent to this curve at the point (1, 3), 
    the value of a - b is equal to -4. -/
theorem tangent_line_problem (a b k : ℝ) : 
  (∀ x, x^3 + a*x + b = k*x + 1 → x = 1) →  -- The line is tangent to the curve
  3^3 + a*3 + b = k*3 + 1 →                 -- The point (1, 3) lies on the curve
  3 = k*1 + 1 →                             -- The point (1, 3) lies on the line
  a - b = -4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l2052_205245


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l2052_205204

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_coefficient_problem (m : ℕ+) :
  let a := binomial (2 * m) m
  let b := binomial (2 * m + 1) m
  13 * a = 7 * b → m = 6 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l2052_205204


namespace NUMINAMATH_CALUDE_f_pi_third_is_nonnegative_reals_l2052_205240

-- Define the set f(x)
def f (φ : Real) : Set Real :=
  {x : Real | x ≥ 0}

-- Theorem statement
theorem f_pi_third_is_nonnegative_reals :
  f (π / 3) = {x : Real | x ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_f_pi_third_is_nonnegative_reals_l2052_205240


namespace NUMINAMATH_CALUDE_percentage_difference_l2052_205289

theorem percentage_difference (A B C y : ℝ) : 
  C > A ∧ A > B ∧ B > 0 → 
  C = 2 * B → 
  A = C * (1 - y / 100) → 
  y = 100 - 50 * (A / B) :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2052_205289


namespace NUMINAMATH_CALUDE_combined_average_score_l2052_205224

theorem combined_average_score (score1 score2 : ℝ) (ratio1 ratio2 : ℕ) : 
  score1 = 88 →
  score2 = 75 →
  ratio1 = 2 →
  ratio2 = 3 →
  (ratio1 * score1 + ratio2 * score2) / (ratio1 + ratio2) = 80 :=
by sorry

end NUMINAMATH_CALUDE_combined_average_score_l2052_205224


namespace NUMINAMATH_CALUDE_circus_performers_standing_time_l2052_205269

/-- The combined time that Pulsar, Polly, and Petra stand on their back legs is 45 minutes. -/
theorem circus_performers_standing_time : 
  let pulsar_time : ℕ := 10
  let polly_time : ℕ := 3 * pulsar_time
  let petra_time : ℕ := polly_time / 6
  pulsar_time + polly_time + petra_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_circus_performers_standing_time_l2052_205269


namespace NUMINAMATH_CALUDE_percentage_increase_l2052_205287

theorem percentage_increase (x : ℝ) (base : ℝ) (percentage : ℝ) : 
  x = base + (percentage / 100) * base →
  x = 110 →
  base = 88 →
  percentage = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l2052_205287


namespace NUMINAMATH_CALUDE_circle_radius_is_three_l2052_205232

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + m*y - 4 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x + y = 0

-- Define symmetry with respect to a line
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_equation ((x₁ + x₂)/2) ((y₁ + y₂)/2)

-- Theorem statement
theorem circle_radius_is_three (m : ℝ) 
  (h₁ : ∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ m ∧ 
    circle_equation x₂ y₂ m ∧ 
    symmetric_points x₁ y₁ x₂ y₂) :
  (let center_x := 1
   let center_y := -m/2
   let radius := Real.sqrt ((center_x - 0)^2 + (center_y - 0)^2)
   radius = 3) := by sorry

end NUMINAMATH_CALUDE_circle_radius_is_three_l2052_205232


namespace NUMINAMATH_CALUDE_remaining_nails_l2052_205273

def initial_nails : ℕ := 400

def kitchen_repair (n : ℕ) : ℕ := n - (n * 35 / 100)

def fence_repair (n : ℕ) : ℕ := n - (n * 75 / 100)

def table_repair (n : ℕ) : ℕ := n - (n * 55 / 100)

def floorboard_repair (n : ℕ) : ℕ := n - (n * 30 / 100)

theorem remaining_nails :
  floorboard_repair (table_repair (fence_repair (kitchen_repair initial_nails))) = 21 :=
by sorry

end NUMINAMATH_CALUDE_remaining_nails_l2052_205273


namespace NUMINAMATH_CALUDE_smallest_product_of_factors_l2052_205296

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_product_of_factors (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  is_factor a 48 → 
  is_factor b 48 → 
  ¬ is_factor (a * b) 48 → 
  (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → is_factor x 48 → is_factor y 48 → 
    ¬ is_factor (x * y) 48 → a * b ≤ x * y) → 
  a * b = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_of_factors_l2052_205296


namespace NUMINAMATH_CALUDE_circle_triangle_construction_l2052_205294

theorem circle_triangle_construction (R r : ℝ) (h : R > r) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    a^2 = 2 * (R^2 + r^2) ∧
    b^2 = 2 * (R^2 - r^2) ∧
    (π * a^2 / 4 + π * b^2 / 4 = π * R^2) ∧
    (π * a^2 / 4 - π * b^2 / 4 = π * r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_construction_l2052_205294


namespace NUMINAMATH_CALUDE_line_inclination_45_degrees_l2052_205215

/-- Proves that for a line passing through points (1, 2) and (3, m) with an inclination angle of 45°, m = 4 -/
theorem line_inclination_45_degrees (m : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    (1, 2) ∈ line ∧ 
    (3, m) ∈ line ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → (y - 2) = (x - 1))) → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_line_inclination_45_degrees_l2052_205215


namespace NUMINAMATH_CALUDE_sum_of_squares_rational_l2052_205278

theorem sum_of_squares_rational (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ q : ℚ, a + b = q) → 
  (∃ r : ℚ, a^3 + b^3 = r) → 
  (∃ s : ℚ, a^2 + b^2 = s) ∧ 
  ¬(∀ t u : ℚ, a = t ∧ b = u) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_rational_l2052_205278


namespace NUMINAMATH_CALUDE_sam_football_games_l2052_205212

theorem sam_football_games (games_this_year games_last_year : ℕ) 
  (h1 : games_this_year = 14)
  (h2 : games_last_year = 29) :
  games_this_year + games_last_year = 43 := by
  sorry

end NUMINAMATH_CALUDE_sam_football_games_l2052_205212


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_cube_positive_l2052_205233

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) :=
by sorry

theorem negation_of_cube_positive :
  (¬ ∀ x > 0, x^3 > 0) ↔ (∃ x > 0, x^3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_cube_positive_l2052_205233


namespace NUMINAMATH_CALUDE_percentage_equality_theorem_l2052_205228

theorem percentage_equality_theorem (x : ℚ) : 
  (30 : ℚ) / 100 * x = (25 : ℚ) / 100 * 40 → x = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_theorem_l2052_205228


namespace NUMINAMATH_CALUDE_christine_distance_l2052_205261

theorem christine_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 20 → time = 4 → distance = speed * time → distance = 80 := by
  sorry

end NUMINAMATH_CALUDE_christine_distance_l2052_205261


namespace NUMINAMATH_CALUDE_test_coincidences_l2052_205270

theorem test_coincidences (n : ℕ) (p_vasya p_misha : ℝ) 
  (hn : n = 20) 
  (hv : p_vasya = 6 / 20) 
  (hm : p_misha = 8 / 20) : 
  n * (p_vasya * p_misha + (1 - p_vasya) * (1 - p_misha)) = 10.8 := by
  sorry

end NUMINAMATH_CALUDE_test_coincidences_l2052_205270


namespace NUMINAMATH_CALUDE_distance_sum_bounds_l2052_205276

/-- Given three mutually perpendicular segments with lengths a, b, and c,
    this theorem proves the bounds for the sum of distances from the endpoints
    to any line passing through the origin. -/
theorem distance_sum_bounds
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_order : a ≤ b ∧ b ≤ c) :
  ∀ (α β γ : ℝ),
    (α^2 + β^2 + γ^2 = 1) →
    (a * α + b * β + c * γ ≥ a + b) ∧
    (a * α + b * β + c * γ ≤ c + Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_distance_sum_bounds_l2052_205276


namespace NUMINAMATH_CALUDE_ad_probability_is_one_third_l2052_205265

/-- The duration of advertisements per hour in minutes -/
def ad_duration : ℕ := 20

/-- The total duration of an hour in minutes -/
def hour_duration : ℕ := 60

/-- The probability of seeing an advertisement when turning on the TV -/
def ad_probability : ℚ := ad_duration / hour_duration

theorem ad_probability_is_one_third : ad_probability = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ad_probability_is_one_third_l2052_205265


namespace NUMINAMATH_CALUDE_parallelogram_area_l2052_205239

/-- Parallelogram EFGH with given side lengths and diagonal -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ
  EH : ℝ
  is_parallelogram : EF > 0 ∧ FG > 0 ∧ EH > 0

/-- The area of the parallelogram EFGH -/
def area (p : Parallelogram) : ℝ :=
  p.EF * p.FG

/-- Theorem: The area of parallelogram EFGH is 1200 -/
theorem parallelogram_area (p : Parallelogram) 
  (h1 : p.EF = 40) 
  (h2 : p.FG = 30) 
  (h3 : p.EH = 50) : 
  area p = 1200 := by
  sorry

#check parallelogram_area

end NUMINAMATH_CALUDE_parallelogram_area_l2052_205239


namespace NUMINAMATH_CALUDE_police_officers_on_duty_l2052_205267

theorem police_officers_on_duty 
  (total_female_officers : ℕ)
  (female_duty_percentage : ℚ)
  (female_duty_ratio : ℚ)
  (h1 : total_female_officers = 600)
  (h2 : female_duty_percentage = 17 / 100)
  (h3 : female_duty_ratio = 1 / 2) :
  ∃ (officers_on_duty : ℕ), 
    officers_on_duty = 204 ∧ 
    (officers_on_duty : ℚ) * female_duty_ratio = (total_female_officers : ℚ) * female_duty_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_police_officers_on_duty_l2052_205267


namespace NUMINAMATH_CALUDE_range_of_a_l2052_205253

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2052_205253


namespace NUMINAMATH_CALUDE_limit_of_a_l2052_205210

def a (n : ℕ) : ℚ := (2 * n + 3) / (n + 5)

theorem limit_of_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_of_a_l2052_205210


namespace NUMINAMATH_CALUDE_parabola_sum_is_line_l2052_205214

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflects a parabola about the x-axis -/
def reflect (p : Parabola) : Parabola :=
  { a := -p.a, b := -p.b, c := -p.c }

/-- Translates a parabola horizontally by a given amount -/
def translate (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b - 2 * p.a * d, c := p.c + p.a * d^2 - p.b * d }

/-- The sum of two parabolas -/
def add (p q : Parabola) : Parabola :=
  { a := p.a + q.a, b := p.b + q.b, c := p.c + q.c }

theorem parabola_sum_is_line (p : Parabola) :
  let p1 := translate p 5
  let p2 := translate (reflect p) (-5)
  let sum := add p1 p2
  sum.a = 0 ∧ sum.b ≠ 0 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_is_line_l2052_205214


namespace NUMINAMATH_CALUDE_simplify_complex_radical_expression_l2052_205298

theorem simplify_complex_radical_expression :
  (3 * (Real.sqrt 5 + Real.sqrt 7)) / (4 * Real.sqrt (3 + Real.sqrt 5)) =
  Real.sqrt (414 - 98 * Real.sqrt 35) / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_radical_expression_l2052_205298


namespace NUMINAMATH_CALUDE_double_reflection_of_D_l2052_205290

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)  -- Translate down by 1
  let reflected := (-p'.2, -p'.1)  -- Reflect across y = -x
  (reflected.1, reflected.2 + 1)  -- Translate up by 1

def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_line (reflect_x p)

theorem double_reflection_of_D :
  double_reflection (4, 1) = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_of_D_l2052_205290


namespace NUMINAMATH_CALUDE_terror_arrangements_count_l2052_205200

/-- The number of unique arrangements of the letters in "TERROR" -/
def terror_arrangements : ℕ := 180

/-- The total number of letters in "TERROR" -/
def total_letters : ℕ := 6

/-- The number of R's in "TERROR" -/
def num_r : ℕ := 2

/-- The number of E's in "TERROR" -/
def num_e : ℕ := 2

/-- Theorem stating that the number of unique arrangements of the letters in "TERROR" is 180 -/
theorem terror_arrangements_count : 
  terror_arrangements = (Nat.factorial total_letters) / ((Nat.factorial num_r) * (Nat.factorial num_e)) :=
by sorry

end NUMINAMATH_CALUDE_terror_arrangements_count_l2052_205200


namespace NUMINAMATH_CALUDE_largest_divisor_of_four_consecutive_integers_l2052_205213

theorem largest_divisor_of_four_consecutive_integers (n : ℕ) :
  (∀ m : ℕ, (m * (m + 1) * (m + 2) * (m + 3)) % 24 = 0) ∧
  (∃ k : ℕ, (k * (k + 1) * (k + 2) * (k + 3)) % 25 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_four_consecutive_integers_l2052_205213


namespace NUMINAMATH_CALUDE_system_solution_l2052_205256

def system_equations (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  (x₃ + x₄ + x₅)^5 = 3*x₁ ∧
  (x₄ + x₅ + x₁)^5 = 3*x₂ ∧
  (x₅ + x₁ + x₂)^5 = 3*x₃ ∧
  (x₁ + x₂ + x₃)^5 = 3*x₄ ∧
  (x₂ + x₃ + x₄)^5 = 3*x₅

theorem system_solution :
  ∀ x₁ x₂ x₃ x₄ x₅ : ℝ,
  system_equations x₁ x₂ x₃ x₄ x₅ →
  ((x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∨
   (x₁ = 1/3 ∧ x₂ = 1/3 ∧ x₃ = 1/3 ∧ x₄ = 1/3 ∧ x₅ = 1/3) ∨
   (x₁ = -1/3 ∧ x₂ = -1/3 ∧ x₃ = -1/3 ∧ x₄ = -1/3 ∧ x₅ = -1/3)) :=
by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l2052_205256


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2052_205207

/-- Proves that a train with given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 80)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 295) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2052_205207


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l2052_205205

theorem circle_area_with_diameter_10 (π : ℝ) :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l2052_205205


namespace NUMINAMATH_CALUDE_simplification_proofs_l2052_205216

theorem simplification_proofs :
  (3.5 * 101 = 353.5) ∧
  (11 * 5.9 - 5.9 = 59) ∧
  (88 - 17.5 - 12.5 = 58) := by
  sorry

end NUMINAMATH_CALUDE_simplification_proofs_l2052_205216


namespace NUMINAMATH_CALUDE_triangle_side_length_l2052_205221

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 5 →
  c = 8 →
  B = Real.pi / 3 →  -- 60° in radians
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  b = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2052_205221


namespace NUMINAMATH_CALUDE_grandchildren_probability_l2052_205274

def num_children : ℕ := 12

theorem grandchildren_probability :
  let total_outcomes := 2^num_children
  let equal_boys_girls := Nat.choose num_children (num_children / 2)
  let all_same_gender := 2
  (total_outcomes - (equal_boys_girls + all_same_gender)) / total_outcomes = 3170 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_grandchildren_probability_l2052_205274


namespace NUMINAMATH_CALUDE_square_sum_from_linear_equations_l2052_205229

theorem square_sum_from_linear_equations (x y : ℝ) 
  (eq1 : x + y = 12) 
  (eq2 : 3 * x + y = 20) : 
  x^2 + y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_linear_equations_l2052_205229


namespace NUMINAMATH_CALUDE_original_lemon_price_was_eight_l2052_205218

/-- The problem of determining the original lemon price --/
def lemon_price_problem (original_lemon_price : ℚ) : Prop :=
  let lemon_price_increase : ℚ := 4
  let grape_price_increase : ℚ := lemon_price_increase / 2
  let original_grape_price : ℚ := 7
  let num_lemons : ℕ := 80
  let num_grapes : ℕ := 140
  let total_revenue : ℚ := 2220
  let new_lemon_price : ℚ := original_lemon_price + lemon_price_increase
  let new_grape_price : ℚ := original_grape_price + grape_price_increase
  (num_lemons : ℚ) * new_lemon_price + (num_grapes : ℚ) * new_grape_price = total_revenue

/-- Theorem stating that the original lemon price was 8 --/
theorem original_lemon_price_was_eight :
  lemon_price_problem 8 := by
  sorry

end NUMINAMATH_CALUDE_original_lemon_price_was_eight_l2052_205218


namespace NUMINAMATH_CALUDE_polynomial_problem_l2052_205238

theorem polynomial_problem (f : ℝ → ℝ) :
  (∃ (a b c d e : ℤ), ∀ x, f x = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
  f (1 + Real.rpow 3 (1/3)) = 1 + Real.rpow 3 (1/3) →
  f (1 + Real.sqrt 3) = 7 + Real.sqrt 3 →
  ∀ x, f x = x^4 - 3*x^3 + 3*x^2 - 3*x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_problem_l2052_205238


namespace NUMINAMATH_CALUDE_museum_visitors_l2052_205262

theorem museum_visitors (V : ℕ) 
  (h1 : V = (3/4 : ℚ) * V + 130)
  (h2 : ∃ E U : ℕ, E = U ∧ E = (3/4 : ℚ) * V) : 
  V = 520 := by
  sorry

end NUMINAMATH_CALUDE_museum_visitors_l2052_205262


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2052_205263

theorem quadratic_inequality_range (p : ℝ) (α : ℝ) (h1 : p = 4 * (Real.sin α) ^ 4)
  (h2 : α ∈ Set.Icc (π / 6) (5 * π / 6)) :
  (∀ x : ℝ, x^2 + p*x + 1 > 2*x + p) ↔ (∀ x : ℝ, x > 1 ∨ x < -3) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2052_205263


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2052_205201

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2052_205201


namespace NUMINAMATH_CALUDE_prism_pyramid_sum_l2052_205297

/-- A shape formed by adding a pyramid to one face of a rectangular prism -/
structure PrismPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_faces : ℕ
  pyramid_edges : ℕ
  pyramid_vertex : ℕ

/-- The total number of exterior faces in the combined shape -/
def total_faces (pp : PrismPyramid) : ℕ := pp.prism_faces - 1 + pp.pyramid_faces

/-- The total number of edges in the combined shape -/
def total_edges (pp : PrismPyramid) : ℕ := pp.prism_edges + pp.pyramid_edges

/-- The total number of vertices in the combined shape -/
def total_vertices (pp : PrismPyramid) : ℕ := pp.prism_vertices + pp.pyramid_vertex

/-- The sum of exterior faces, edges, and vertices in the combined shape -/
def total_sum (pp : PrismPyramid) : ℕ := total_faces pp + total_edges pp + total_vertices pp

theorem prism_pyramid_sum :
  ∃ (pp : PrismPyramid), total_sum pp = 34 ∧
  ∀ (pp' : PrismPyramid), total_sum pp' ≤ total_sum pp :=
sorry

end NUMINAMATH_CALUDE_prism_pyramid_sum_l2052_205297


namespace NUMINAMATH_CALUDE_josephine_milk_sales_l2052_205226

/-- Given the conditions of Josephine's milk sales, prove that the amount in each of the two unknown containers is 0.75 liters. -/
theorem josephine_milk_sales (total_milk : ℝ) (big_containers : ℕ) (small_containers : ℕ) (unknown_containers : ℕ)
  (big_container_capacity : ℝ) (small_container_capacity : ℝ)
  (h_total : total_milk = 10)
  (h_big : big_containers = 3)
  (h_small : small_containers = 5)
  (h_unknown : unknown_containers = 2)
  (h_big_capacity : big_container_capacity = 2)
  (h_small_capacity : small_container_capacity = 0.5) :
  (total_milk - (big_containers * big_container_capacity + small_containers * small_container_capacity)) / unknown_containers = 0.75 := by
sorry

end NUMINAMATH_CALUDE_josephine_milk_sales_l2052_205226


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2052_205219

open Set

theorem intersection_of_M_and_N :
  let U : Type := ℝ
  let M : Set U := {x | x < 1}
  let N : Set U := {x | 0 < x ∧ x < 2}
  M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2052_205219


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2052_205211

/-- Given a > 0 and a ≠ 1, prove that if f(x) = a^x is decreasing on ℝ, 
    then g(x) = (2-a)x^3 is increasing on ℝ, but not necessarily vice versa. -/
theorem sufficient_but_not_necessary 
  (a : ℝ) 
  (ha_pos : a > 0) 
  (ha_neq_one : a ≠ 1) 
  (f : ℝ → ℝ) 
  (hf : f = fun x ↦ a^x) 
  (g : ℝ → ℝ) 
  (hg : g = fun x ↦ (2-a)*x^3) : 
  (∀ x y, x < y → f x > f y) → 
  (∀ x y, x < y → g x < g y) ∧ 
  ¬(∀ x y, x < y → g x < g y → ∀ x y, x < y → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2052_205211


namespace NUMINAMATH_CALUDE_exists_rectangle_six_pieces_l2052_205247

/-- A rectangle inscribed in an isosceles right triangle --/
structure InscribedRectangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  h_positive : side1 > 0 ∧ side2 > 0 ∧ hypotenuse > 0
  h_inscribed : side1 + side2 < hypotenuse

/-- Two straight lines that divide a rectangle --/
structure DividingLines where
  line1 : ℝ × ℝ → ℝ × ℝ → Prop
  line2 : ℝ × ℝ → ℝ × ℝ → Prop

/-- The number of pieces a rectangle is divided into by two straight lines --/
def numPieces (r : InscribedRectangle) (d : DividingLines) : ℕ :=
  sorry

/-- Theorem stating the existence of a rectangle that can be divided into 6 pieces --/
theorem exists_rectangle_six_pieces :
  ∃ (r : InscribedRectangle) (d : DividingLines), numPieces r d = 6 :=
sorry

end NUMINAMATH_CALUDE_exists_rectangle_six_pieces_l2052_205247


namespace NUMINAMATH_CALUDE_janet_earnings_l2052_205234

/-- Calculates the total earnings of Janet based on her exterminator work and sculpture sales. -/
theorem janet_earnings (
  hourly_rate : ℝ)
  (sculpture_price_per_pound : ℝ)
  (hours_worked : ℝ)
  (sculpture1_weight : ℝ)
  (sculpture2_weight : ℝ)
  (h1 : hourly_rate = 70)
  (h2 : sculpture_price_per_pound = 20)
  (h3 : hours_worked = 20)
  (h4 : sculpture1_weight = 5)
  (h5 : sculpture2_weight = 7) :
  hourly_rate * hours_worked + sculpture_price_per_pound * (sculpture1_weight + sculpture2_weight) = 1640 :=
by
  sorry


end NUMINAMATH_CALUDE_janet_earnings_l2052_205234


namespace NUMINAMATH_CALUDE_factorization_identity_l2052_205244

theorem factorization_identity (a b : ℝ) : a^2 - 2*a*b = a*(a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l2052_205244


namespace NUMINAMATH_CALUDE_percentage_problem_l2052_205230

theorem percentage_problem (y : ℝ) (h1 : y > 0) (h2 : (y / 100) * y = 16) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2052_205230


namespace NUMINAMATH_CALUDE_g_sum_neg_one_l2052_205275

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the functional equation
axiom func_eq : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y

-- State the condition f(-2) = f(1) ≠ 0
axiom f_cond : f (-2) = f 1 ∧ f 1 ≠ 0

-- Theorem to prove
theorem g_sum_neg_one : g 1 + g (-1) = -1 :=
sorry

end NUMINAMATH_CALUDE_g_sum_neg_one_l2052_205275


namespace NUMINAMATH_CALUDE_smallest_divisor_l2052_205282

theorem smallest_divisor : 
  let n : ℕ := 1012
  let m : ℕ := n - 4
  let divisors : List ℕ := [16, 18, 21, 28]
  (∀ d ∈ divisors, m % d = 0) ∧ 
  (∀ d ∈ divisors, d ≥ 16) ∧
  16 ∈ divisors →
  16 = (divisors.filter (λ d => m % d = 0)).minimum?.getD 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_l2052_205282


namespace NUMINAMATH_CALUDE_earth_circumference_scientific_notation_l2052_205258

theorem earth_circumference_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ |a| ∧ |a| < 10 ∧
    n = 6 ∧
    4010000 = a * (10 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_earth_circumference_scientific_notation_l2052_205258


namespace NUMINAMATH_CALUDE_tadd_3000th_number_l2052_205250

/-- Represents the counting game with Tadd, Todd, and Tucker --/
structure CountingGame where
  max_count : Nat
  tadd_start : Nat
  todd_initial_count : Nat
  tucker_initial_count : Nat
  increment : Nat

/-- Calculates Tadd's nth number in the game --/
def tadd_nth_number (game : CountingGame) (n : Nat) : Nat :=
  sorry

/-- The main theorem stating that Tadd's 3000th number is X --/
theorem tadd_3000th_number (game : CountingGame) 
  (h1 : game.max_count = 15000)
  (h2 : game.tadd_start = 1)
  (h3 : game.todd_initial_count = 3)
  (h4 : game.tucker_initial_count = 5)
  (h5 : game.increment = 2) :
  tadd_nth_number game 3000 = X :=
  sorry

end NUMINAMATH_CALUDE_tadd_3000th_number_l2052_205250


namespace NUMINAMATH_CALUDE_problem_solution_l2052_205235

-- Define the function f(x) = |x-a| + 3x
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

theorem problem_solution :
  (∀ x : ℝ, f 1 x > 3 * x + 2 ↔ (x > 3 ∨ x < -1)) ∧
  (∀ a : ℝ, a > 0 → (∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) → a = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2052_205235


namespace NUMINAMATH_CALUDE_perimeter_of_square_III_l2052_205252

/-- Given three squares I, II, and III, prove that the perimeter of III is 36. -/
theorem perimeter_of_square_III (I II III : Real) : 
  (I > 0) →  -- I is positive (side length of a square)
  (II > 0) → -- II is positive (side length of a square)
  (4 * I = 12) → -- Perimeter of I is 12
  (4 * II = 24) → -- Perimeter of II is 24
  (III = I + II) → -- Side length of III is sum of side lengths of I and II
  (4 * III = 36) := by -- Perimeter of III is 36
sorry

end NUMINAMATH_CALUDE_perimeter_of_square_III_l2052_205252


namespace NUMINAMATH_CALUDE_geometric_series_sum_times_four_fifths_l2052_205291

theorem geometric_series_sum_times_four_fifths :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 5
  let S := (a * (1 - r^n)) / (1 - r)
  (S * 4/5 : ℚ) = 21/80 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_times_four_fifths_l2052_205291


namespace NUMINAMATH_CALUDE_product_difference_theorem_l2052_205241

theorem product_difference_theorem (number value : ℕ) (h1 : number = 15) (h2 : value = 13) :
  number * value - number = 180 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_theorem_l2052_205241


namespace NUMINAMATH_CALUDE_final_sum_is_eight_times_original_l2052_205225

theorem final_sum_is_eight_times_original (S a b : ℝ) (h : a + b = S) :
  (2 * (4 * a)) + (2 * (4 * b)) = 8 * S := by
  sorry

end NUMINAMATH_CALUDE_final_sum_is_eight_times_original_l2052_205225


namespace NUMINAMATH_CALUDE_extremum_implies_f2_value_l2052_205203

/-- A function f with an extremum at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f2_value (a b : ℝ) :
  f' a b 1 = 0 → f a b 1 = 10 → f a b 2 = 11 ∨ f a b 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_f2_value_l2052_205203


namespace NUMINAMATH_CALUDE_max_revenue_at_22_l2052_205272

def cinema_revenue (price : ℕ) : ℤ :=
  if price ≤ 10 then
    1000 * price - 5750
  else
    -30 * price * price + 1300 * price - 5750

def valid_price (price : ℕ) : Prop :=
  (6 ≤ price) ∧ (price ≤ 38)

theorem max_revenue_at_22 :
  (∀ p, valid_price p → cinema_revenue p ≤ cinema_revenue 22) ∧
  cinema_revenue 22 = 8330 :=
sorry

end NUMINAMATH_CALUDE_max_revenue_at_22_l2052_205272


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2052_205220

theorem greatest_of_three_consecutive_integers (n : ℤ) :
  n + 2 = 8 → (n < n + 1 ∧ n + 1 < n + 2) → n + 2 = max n (max (n + 1) (n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2052_205220


namespace NUMINAMATH_CALUDE_printing_presses_count_l2052_205254

/-- The number of papers printed -/
def num_papers : ℕ := 500000

/-- The time taken in the first scenario (in hours) -/
def time1 : ℝ := 12

/-- The time taken in the second scenario (in hours) -/
def time2 : ℝ := 13.999999999999998

/-- The number of printing presses in the second scenario -/
def presses2 : ℕ := 30

/-- The number of printing presses in the first scenario -/
def presses1 : ℕ := 26

theorem printing_presses_count :
  (num_papers : ℝ) / time1 / (num_papers / time2) = presses1 / presses2 :=
sorry

end NUMINAMATH_CALUDE_printing_presses_count_l2052_205254


namespace NUMINAMATH_CALUDE_angle_c_is_60_degrees_l2052_205292

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the concept of an angle in a quadrilateral
def angle (q : Quadrilateral) (v : Fin 4) : ℝ := sorry

-- State the theorem
theorem angle_c_is_60_degrees (q : Quadrilateral) :
  angle q 0 + 60 = angle q 1 →  -- Angle A is 60° smaller than angle B
  angle q 2 = 60 := by  -- Angle C is 60°
  sorry

end NUMINAMATH_CALUDE_angle_c_is_60_degrees_l2052_205292


namespace NUMINAMATH_CALUDE_calorie_difference_l2052_205249

-- Define the number of squirrels and rabbits caught per hour
def squirrels_per_hour : ℕ := 6
def rabbits_per_hour : ℕ := 2

-- Define the calorie content of each animal
def calories_per_squirrel : ℕ := 300
def calories_per_rabbit : ℕ := 800

-- Define the total calories from squirrels and rabbits
def total_calories_squirrels : ℕ := squirrels_per_hour * calories_per_squirrel
def total_calories_rabbits : ℕ := rabbits_per_hour * calories_per_rabbit

-- Theorem to prove
theorem calorie_difference : total_calories_squirrels - total_calories_rabbits = 200 := by
  sorry

end NUMINAMATH_CALUDE_calorie_difference_l2052_205249


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l2052_205277

/-- A perfect sphere inscribed in a cube -/
structure InscribedSphere where
  cube_side_length : ℝ
  touches_face_centers : Bool
  radius : ℝ

/-- Theorem: The radius of a perfect sphere inscribed in a cube with side length 2,
    such that it touches the center of each face, is equal to 1 -/
theorem inscribed_sphere_radius
  (s : InscribedSphere)
  (h1 : s.cube_side_length = 2)
  (h2 : s.touches_face_centers = true) :
  s.radius = 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l2052_205277


namespace NUMINAMATH_CALUDE_social_science_papers_selected_l2052_205288

/-- Proves the number of social science papers selected in stratified sampling -/
theorem social_science_papers_selected
  (total_papers : ℕ)
  (social_science_papers : ℕ)
  (selected_papers : ℕ)
  (h1 : total_papers = 153)
  (h2 : social_science_papers = 54)
  (h3 : selected_papers = 51)
  : (social_science_papers * selected_papers) / total_papers = 18 := by
  sorry

end NUMINAMATH_CALUDE_social_science_papers_selected_l2052_205288
