import Mathlib

namespace NUMINAMATH_CALUDE_fruit_purchase_total_l3362_336229

/-- Calculate the total amount paid for fruits given their quantities and rates --/
theorem fruit_purchase_total (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) :
  grape_quantity = 9 →
  grape_rate = 70 →
  mango_quantity = 9 →
  mango_rate = 55 →
  grape_quantity * grape_rate + mango_quantity * mango_rate = 1125 := by
sorry

end NUMINAMATH_CALUDE_fruit_purchase_total_l3362_336229


namespace NUMINAMATH_CALUDE_percentage_problem_l3362_336276

theorem percentage_problem (x : ℝ) (h : 24 = 75 / 100 * x) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3362_336276


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l3362_336245

-- Define the parabola function
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x - 7

-- Define the theorem
theorem parabola_point_relationship (a : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h_opens_down : a < 0)
  (h_y₁ : y₁ = parabola a (-4))
  (h_y₂ : y₂ = parabola a 2)
  (h_y₃ : y₃ = parabola a 3) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l3362_336245


namespace NUMINAMATH_CALUDE_equation_root_implies_m_value_l3362_336291

theorem equation_root_implies_m_value (x m : ℝ) :
  (∃ x, (x - 1) / (x - 4) = m / (x - 4)) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_root_implies_m_value_l3362_336291


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3362_336243

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 1 ≤ x^2 ∧ x^2 ≤ 16) ∧ 
  (∃ x : ℝ, 1 ≤ x^2 ∧ x^2 ≤ 16 ∧ ¬(1 ≤ x ∧ x ≤ 4)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3362_336243


namespace NUMINAMATH_CALUDE_beach_house_rent_l3362_336212

/-- The total amount paid for rent by a group of people -/
def total_rent (num_people : ℕ) (rent_per_person : ℚ) : ℚ :=
  num_people * rent_per_person

/-- Proof that 7 people paying $70.00 each results in a total of $490.00 -/
theorem beach_house_rent :
  total_rent 7 70 = 490 := by
  sorry

end NUMINAMATH_CALUDE_beach_house_rent_l3362_336212


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3362_336277

-- Define an arithmetic sequence and its sum
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence a →
  sum_of_arithmetic_sequence a S →
  m > 0 →
  S (m - 1) = -2 →
  S m = 0 →
  S (m + 1) = 3 →
  m = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3362_336277


namespace NUMINAMATH_CALUDE_cube_cutting_l3362_336279

theorem cube_cutting (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^3 = 98 + b^3) : b = 3 :=
sorry

end NUMINAMATH_CALUDE_cube_cutting_l3362_336279


namespace NUMINAMATH_CALUDE_interest_rate_problem_l3362_336273

/-- Given a sum P at simple interest rate R for 5 years, if increasing the rate by 5% 
    results in Rs. 250 more interest, then P = 1000 -/
theorem interest_rate_problem (P R : ℝ) (h : P > 0) (k : R > 0) :
  (P * (R + 5) * 5) / 100 - (P * R * 5) / 100 = 250 → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l3362_336273


namespace NUMINAMATH_CALUDE_total_ways_to_place_balls_l3362_336285

/-- The number of ways to place four distinct colored balls into two boxes -/
def place_balls : ℕ :=
  let box1_with_1_ball := Nat.choose 4 1
  let box1_with_2_balls := Nat.choose 4 2
  box1_with_1_ball + box1_with_2_balls

/-- Theorem stating that there are 10 ways to place the balls -/
theorem total_ways_to_place_balls : place_balls = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_to_place_balls_l3362_336285


namespace NUMINAMATH_CALUDE_rectangles_count_l3362_336257

/-- The number of rectangles in a strip of height 1 and width n --/
def rectanglesInStrip (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The total number of rectangles in the given grid --/
def totalRectangles : ℕ :=
  rectanglesInStrip 5 + rectanglesInStrip 4 - 1

theorem rectangles_count : totalRectangles = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_count_l3362_336257


namespace NUMINAMATH_CALUDE_probability_gears_from_algebras_l3362_336218

/-- The set of letters in "ALGEBRAS" -/
def algebras : Finset Char := {'A', 'L', 'G', 'E', 'B', 'R', 'S'}

/-- The set of letters in "GEARS" -/
def gears : Finset Char := {'G', 'E', 'A', 'R', 'S'}

/-- The probability of selecting a tile with a letter from "GEARS" out of the tiles from "ALGEBRAS" -/
theorem probability_gears_from_algebras :
  (algebras.filter (λ c => c ∈ gears)).card / algebras.card = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_gears_from_algebras_l3362_336218


namespace NUMINAMATH_CALUDE_complex_fraction_real_implies_a_equals_two_l3362_336207

theorem complex_fraction_real_implies_a_equals_two (a : ℝ) :
  (((a : ℂ) + 2 * Complex.I) / (1 + Complex.I)).im = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_implies_a_equals_two_l3362_336207


namespace NUMINAMATH_CALUDE_pencil_distribution_solution_l3362_336208

/-- Represents the pencil distribution problem --/
def PencilDistribution (initial_pencils : ℕ) (initial_containers : ℕ) (first_addition : ℕ) (second_addition : ℕ) (final_containers : ℕ) : Prop :=
  let total_pencils := initial_pencils + first_addition + second_addition
  ∃ (distributed_pencils : ℕ), 
    distributed_pencils ≤ total_pencils ∧
    distributed_pencils % final_containers = 0 ∧
    ∀ (n : ℕ), n > distributed_pencils → n % final_containers ≠ 0 ∨ n > total_pencils

/-- Theorem stating the solution to the pencil distribution problem --/
theorem pencil_distribution_solution :
  PencilDistribution 150 5 30 47 6 → 
  ∃ (distributed_pencils : ℕ), distributed_pencils = 222 ∧ distributed_pencils % 6 = 0 :=
sorry

end NUMINAMATH_CALUDE_pencil_distribution_solution_l3362_336208


namespace NUMINAMATH_CALUDE_cost_for_holly_fence_l3362_336233

/-- The total cost to plant trees along a fence --/
def total_cost (fence_length_yards : ℕ) (tree_width_feet : ℚ) (cost_per_tree : ℚ) : ℚ :=
  let fence_length_feet := fence_length_yards * 3
  let num_trees := fence_length_feet / tree_width_feet
  num_trees * cost_per_tree

/-- Proof that the total cost to plant trees along a 25-yard fence,
    where each tree is 1.5 feet wide and costs $8.00, is $400.00 --/
theorem cost_for_holly_fence :
  total_cost 25 (3/2) 8 = 400 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_holly_fence_l3362_336233


namespace NUMINAMATH_CALUDE_assignment_plans_eq_48_l3362_336281

/-- Represents the number of umpires from each country -/
def umpires_per_country : ℕ := 2

/-- Represents the number of countries -/
def num_countries : ℕ := 3

/-- Represents the number of venues -/
def num_venues : ℕ := 3

/-- Calculates the number of ways to assign umpires to venues -/
def assignment_plans : ℕ := sorry

/-- Theorem stating that the number of assignment plans is 48 -/
theorem assignment_plans_eq_48 : assignment_plans = 48 := by sorry

end NUMINAMATH_CALUDE_assignment_plans_eq_48_l3362_336281


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3362_336213

theorem arithmetic_expression_equality : 4 * 6 + 8 * 3 - 28 / 2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3362_336213


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_four_and_five_l3362_336278

theorem smallest_four_digit_divisible_by_four_and_five : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 ∧ n % 5 = 0 → n ≥ 1020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_four_and_five_l3362_336278


namespace NUMINAMATH_CALUDE_smallest_factor_of_32_with_sum_3_l3362_336231

theorem smallest_factor_of_32_with_sum_3 :
  ∃ (a b c : ℤ),
    a * b * c = 32 ∧
    a + b + c = 3 ∧
    (∀ (x y z : ℤ), x * y * z = 32 → x + y + z = 3 → min a (min b c) ≤ min x (min y z)) ∧
    min a (min b c) = -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_of_32_with_sum_3_l3362_336231


namespace NUMINAMATH_CALUDE_toy_bridge_weight_l3362_336289

/-- The weight that a toy bridge must support -/
theorem toy_bridge_weight (full_cans : Nat) (soda_per_can : Nat) (empty_can_weight : Nat) (additional_empty_cans : Nat) : 
  full_cans * (soda_per_can + empty_can_weight) + additional_empty_cans * empty_can_weight = 88 :=
by
  sorry

#check toy_bridge_weight 6 12 2 2

end NUMINAMATH_CALUDE_toy_bridge_weight_l3362_336289


namespace NUMINAMATH_CALUDE_prob_at_least_one_heart_or_joker_correct_l3362_336217

/-- The number of cards in a standard deck plus two jokers -/
def total_cards : ℕ := 54

/-- The number of hearts and jokers combined -/
def heart_or_joker : ℕ := 15

/-- The probability of drawing at least one heart or joker in two draws with replacement -/
def prob_at_least_one_heart_or_joker : ℚ := 155 / 324

theorem prob_at_least_one_heart_or_joker_correct :
  (1 : ℚ) - (1 - heart_or_joker / total_cards) ^ 2 = prob_at_least_one_heart_or_joker :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_heart_or_joker_correct_l3362_336217


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3362_336263

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a scalene triangle -/
def isScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ 
  b = a + 2 ∧ c = b + 2 ∧
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1

theorem smallest_prime_perimeter_scalene_triangle : 
  ∃ (a b c : ℕ), 
    isScaleneTriangle a b c ∧ 
    areConsecutiveOddPrimes a b c ∧ 
    isPrime (a + b + c) ∧
    ∀ (x y z : ℕ), 
      isScaleneTriangle x y z → 
      areConsecutiveOddPrimes x y z → 
      isPrime (x + y + z) → 
      a + b + c ≤ x + y + z ∧
    a + b + c = 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3362_336263


namespace NUMINAMATH_CALUDE_bus_driver_rate_l3362_336274

theorem bus_driver_rate (hours_worked : ℕ) (total_compensation : ℚ) : 
  hours_worked = 50 →
  total_compensation = 920 →
  ∃ (regular_rate : ℚ),
    (40 * regular_rate + (hours_worked - 40) * (1.75 * regular_rate) = total_compensation) ∧
    regular_rate = 16 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_rate_l3362_336274


namespace NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l3362_336225

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_sided_polygon_diagonals_l3362_336225


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3362_336214

theorem quadratic_equation_solutions (x : ℝ) :
  (x^2 + 2*x + 1 = 4) ↔ (x = 1 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3362_336214


namespace NUMINAMATH_CALUDE_circle_ratio_theorem_l3362_336295

noncomputable def circle_ratio (r : ℝ) (A B C : ℝ × ℝ) : Prop :=
  let O := (0, 0)  -- Center of the circle
  ∃ (θ : ℝ),
    -- Points A, B, and C are on a circle of radius r
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧
    -- AB = AC
    dist A B = dist A C ∧
    -- AB > r
    dist A B > r ∧
    -- Length of minor arc BC is r
    θ = 1 ∧
    -- Ratio AB/BC
    dist A B / dist B C = (1/2) * (1 / Real.sin (1/4))

theorem circle_ratio_theorem (r : ℝ) (A B C : ℝ × ℝ) 
  (h : circle_ratio r A B C) : 
  ∃ (θ : ℝ), dist A B / dist B C = (1/2) * (1 / Real.sin (1/4)) :=
sorry

end NUMINAMATH_CALUDE_circle_ratio_theorem_l3362_336295


namespace NUMINAMATH_CALUDE_number_of_divisors_l3362_336269

theorem number_of_divisors 
  (p q r : Nat) 
  (m : Nat) 
  (h_p_prime : Nat.Prime p) 
  (h_q_prime : Nat.Prime q) 
  (h_r_prime : Nat.Prime r) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) 
  (h_m_pos : m > 0) 
  (h_n_def : n = 7^m * p^2 * q * r) : 
  Nat.card (Nat.divisors n) = 12 * (m + 1) := by
sorry

end NUMINAMATH_CALUDE_number_of_divisors_l3362_336269


namespace NUMINAMATH_CALUDE_intersection_point_Q_l3362_336235

-- Define the circles
def circle1 (x y r : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = r^2
def circle2 (x y R : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = R^2

-- Define the intersection points
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (-2, -1)

-- Theorem statement
theorem intersection_point_Q :
  ∀ (r R : ℝ),
  (∃ (x y : ℝ), circle1 x y r ∧ circle2 x y R) →  -- Circles intersect
  circle1 P.1 P.2 r →                            -- P is on circle1
  circle2 P.1 P.2 R →                            -- P is on circle2
  circle1 Q.1 Q.2 r ∧ circle2 Q.1 Q.2 R          -- Q is on both circles
  := by sorry

end NUMINAMATH_CALUDE_intersection_point_Q_l3362_336235


namespace NUMINAMATH_CALUDE_emily_candy_from_neighbors_l3362_336230

/-- The number of candy pieces Emily received from her sister -/
def candy_from_sister : ℕ := 13

/-- The number of candy pieces Emily ate per day -/
def candy_eaten_per_day : ℕ := 9

/-- The number of days Emily's candy lasted -/
def days_candy_lasted : ℕ := 2

/-- The number of candy pieces Emily received from neighbors -/
def candy_from_neighbors : ℕ := (candy_eaten_per_day * days_candy_lasted) - candy_from_sister

theorem emily_candy_from_neighbors : candy_from_neighbors = 5 := by
  sorry

end NUMINAMATH_CALUDE_emily_candy_from_neighbors_l3362_336230


namespace NUMINAMATH_CALUDE_ginger_water_usage_l3362_336260

/-- Calculates the total cups of water used by Ginger in her garden --/
def total_water_used (hours_worked : ℕ) (cups_per_bottle : ℕ) (bottles_for_plants : ℕ) : ℕ :=
  (hours_worked * cups_per_bottle) + (bottles_for_plants * cups_per_bottle)

/-- Theorem stating that given the conditions, Ginger used 26 cups of water --/
theorem ginger_water_usage :
  let hours_worked : ℕ := 8
  let cups_per_bottle : ℕ := 2
  let bottles_for_plants : ℕ := 5
  total_water_used hours_worked cups_per_bottle bottles_for_plants = 26 := by
  sorry

end NUMINAMATH_CALUDE_ginger_water_usage_l3362_336260


namespace NUMINAMATH_CALUDE_derivative_x_squared_cos_x_l3362_336200

theorem derivative_x_squared_cos_x :
  let y : ℝ → ℝ := λ x ↦ x^2 * Real.cos x
  deriv y = λ x ↦ 2 * x * Real.cos x - x^2 * Real.sin x := by
sorry

end NUMINAMATH_CALUDE_derivative_x_squared_cos_x_l3362_336200


namespace NUMINAMATH_CALUDE_divisor_problem_l3362_336293

theorem divisor_problem (x : ℝ) (d : ℝ) : 
  x = 22.142857142857142 →
  (7 * (x + 5)) / d - 5 = 33 →
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l3362_336293


namespace NUMINAMATH_CALUDE_expand_polynomial_l3362_336211

theorem expand_polynomial (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4*x + 4) = x^4 + 4*x^3 - 16*x - 16 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3362_336211


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l3362_336248

-- Define the vectors
def vec_a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def vec_b (y : ℝ) : Fin 2 → ℝ := ![1, y]
def vec_c : Fin 2 → ℝ := ![2, -6]

-- Define the conditions
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (i : Fin 2), u i = k * v i

-- Theorem statement
theorem magnitude_of_sum (x y : ℝ) :
  perpendicular (vec_a x) vec_c →
  parallel (vec_b y) vec_c →
  ‖vec_a x + vec_b y‖ = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_magnitude_of_sum_l3362_336248


namespace NUMINAMATH_CALUDE_ellipse_theorem_l3362_336209

-- Define the ellipse M
def ellipse_M (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1 ∧ a > 0

-- Define the right focus F
def right_focus (a c : ℝ) : Prop :=
  c > 0 ∧ a^2 = 3 + c^2

-- Define the symmetric property
def symmetric_property (c : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse_M (2*c) (-x+2*c) y ∧ x^2 + y^2 = 0

-- Main theorem
theorem ellipse_theorem (a c : ℝ) 
  (h1 : ellipse_M a 0 0)
  (h2 : right_focus a c)
  (h3 : symmetric_property c) :
  a^2 = 4 ∧ c = 1 ∧
  ∀ (k x₁ y₁ x₂ y₂ : ℝ),
    (ellipse_M a x₁ y₁ ∧ ellipse_M a x₂ y₂ ∧
     y₁ = k*(x₁ - 4) ∧ y₂ = k*(x₂ - 4) ∧ k ≠ 0) →
    ∃ (t : ℝ), t*(y₁ + y₂) + x₁ = 1 ∧ t*(x₁ - x₂) + y₁ = 0 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l3362_336209


namespace NUMINAMATH_CALUDE_halfway_between_one_third_and_one_eighth_l3362_336287

theorem halfway_between_one_third_and_one_eighth :
  (1 / 3 : ℚ) / 2 + (1 / 8 : ℚ) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_third_and_one_eighth_l3362_336287


namespace NUMINAMATH_CALUDE_recycling_point_calculation_l3362_336240

/-- The number of pounds needed to recycle to earn one point -/
def pounds_per_point (zoe_pounds : ℕ) (friends_pounds : ℕ) (total_points : ℕ) : ℚ :=
  (zoe_pounds + friends_pounds : ℚ) / total_points

theorem recycling_point_calculation :
  pounds_per_point 25 23 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_recycling_point_calculation_l3362_336240


namespace NUMINAMATH_CALUDE_rectangle_hexagon_pqr_sum_l3362_336242

/-- A hexagon formed by three rectangles intersecting three straight lines -/
structure RectangleHexagon where
  -- External angles at S, T, U
  s : ℝ
  t : ℝ
  u : ℝ
  -- External angles at P, Q, R
  p : ℝ
  q : ℝ
  r : ℝ
  -- Conditions
  angle_s : s = 55
  angle_t : t = 60
  angle_u : u = 65
  sum_external : p + q + r + s + t + u = 360

/-- The sum of external angles at P, Q, and R in the RectangleHexagon is 180° -/
theorem rectangle_hexagon_pqr_sum (h : RectangleHexagon) : h.p + h.q + h.r = 180 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_hexagon_pqr_sum_l3362_336242


namespace NUMINAMATH_CALUDE_daniel_noodles_remaining_l3362_336262

/-- The number of noodles Daniel has now, given his initial count and the number he gave away. -/
def noodles_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Daniel has 54 noodles remaining. -/
theorem daniel_noodles_remaining :
  noodles_remaining 66 12 = 54 := by
  sorry

end NUMINAMATH_CALUDE_daniel_noodles_remaining_l3362_336262


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3362_336284

/-- Given a function f : ℝ → ℝ that satisfies the functional equation
    3f(x) + 2f(1-x) = 4x for all x, prove that f(x) = 4x - 8/5 for all x. -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x, 3 * f x + 2 * f (1 - x) = 4 * x) :
  ∀ x, f x = 4 * x - 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3362_336284


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3362_336215

theorem regular_polygon_sides (interior_angle : ℝ) (sum_except_one : ℝ) : 
  interior_angle = 160 → sum_except_one = 3600 → 
  (sum_except_one + interior_angle) / interior_angle = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3362_336215


namespace NUMINAMATH_CALUDE_lemonade_lemons_l3362_336237

/-- Given that each glass of lemonade requires 2 lemons and Jane can make 9 glasses,
    prove that the total number of lemons is 18. -/
theorem lemonade_lemons :
  ∀ (lemons_per_glass : ℕ) (glasses : ℕ) (total_lemons : ℕ),
    lemons_per_glass = 2 →
    glasses = 9 →
    total_lemons = lemons_per_glass * glasses →
    total_lemons = 18 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_lemons_l3362_336237


namespace NUMINAMATH_CALUDE_aunt_may_milk_problem_l3362_336220

theorem aunt_may_milk_problem (morning_milk evening_milk sold_milk leftover_milk : ℕ) 
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk - sold_milk + leftover_milk = 148 :=
by sorry

end NUMINAMATH_CALUDE_aunt_may_milk_problem_l3362_336220


namespace NUMINAMATH_CALUDE_noah_has_largest_result_l3362_336299

def starting_number : ℕ := 15

def liam_result : ℕ := ((starting_number - 2) * 3) + 3
def mia_result : ℕ := ((starting_number * 3) - 4) + 3
def noah_result : ℕ := ((starting_number - 3) + 4) * 3

theorem noah_has_largest_result :
  noah_result > liam_result ∧ noah_result > mia_result :=
by sorry

end NUMINAMATH_CALUDE_noah_has_largest_result_l3362_336299


namespace NUMINAMATH_CALUDE_rational_sum_l3362_336268

theorem rational_sum (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Finset ℚ) = 
    {-24, -2, -3/2, -1/8, 1, 3} →
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
sorry

end NUMINAMATH_CALUDE_rational_sum_l3362_336268


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3362_336272

theorem system_of_equations_solution (x y : ℚ) 
  (eq1 : 3 * x - 2 * y = 7)
  (eq2 : 2 * x + 3 * y = 8) : 
  x = 37 / 13 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3362_336272


namespace NUMINAMATH_CALUDE_sector_max_area_l3362_336201

/-- Given a sector with perimeter 60 cm, its maximum area is 225 cm² -/
theorem sector_max_area (r : ℝ) (l : ℝ) (S : ℝ → ℝ) :
  (0 < r) → (r < 30) →
  (l + 2 * r = 60) →
  (S = λ r => (1 / 2) * l * r) →
  (∀ r', S r' ≤ 225) ∧ (∃ r', S r' = 225) :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l3362_336201


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l3362_336241

/-- The coefficient of x^2 in the expansion of (3x^3 + 5x^2 - 4x + 1)(2x^2 - 9x + 3) -/
def coefficient_x_squared : ℤ := 51

/-- The first polynomial in the product -/
def poly1 (x : ℚ) : ℚ := 3 * x^3 + 5 * x^2 - 4 * x + 1

/-- The second polynomial in the product -/
def poly2 (x : ℚ) : ℚ := 2 * x^2 - 9 * x + 3

/-- Theorem stating that the coefficient of x^2 in the expansion of (poly1 * poly2) is equal to coefficient_x_squared -/
theorem coefficient_x_squared_expansion :
  ∃ (a b c d e : ℚ), (poly1 * poly2) = (λ x => a * x^4 + b * x^3 + coefficient_x_squared * x^2 + d * x + e) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l3362_336241


namespace NUMINAMATH_CALUDE_proportion_check_l3362_336216

/-- A set of four positive real numbers forms a proportion if the product of the first and last
    numbers equals the product of the middle two numbers. -/
def IsProportional (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * d = b * c

theorem proportion_check :
  IsProportional 5 15 3 9 ∧
  ¬IsProportional 4 5 6 7 ∧
  ¬IsProportional 3 4 5 8 ∧
  ¬IsProportional 8 4 1 3 :=
by sorry

end NUMINAMATH_CALUDE_proportion_check_l3362_336216


namespace NUMINAMATH_CALUDE_jamie_speed_equals_alex_speed_l3362_336254

/-- Given the cycling speeds of Alex, Sam, and Jamie, prove that Jamie's speed equals Alex's speed. -/
theorem jamie_speed_equals_alex_speed (alex_speed : ℝ) (sam_speed : ℝ) (jamie_speed : ℝ)
  (h1 : alex_speed = 6)
  (h2 : sam_speed = 3/4 * alex_speed)
  (h3 : jamie_speed = 4/3 * sam_speed) :
  jamie_speed = alex_speed :=
by sorry

end NUMINAMATH_CALUDE_jamie_speed_equals_alex_speed_l3362_336254


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l3362_336253

/-- The cost per page for revisions -/
def revision_cost : ℝ := 4

/-- The total number of pages in the manuscript -/
def total_pages : ℕ := 100

/-- The number of pages revised once -/
def pages_revised_once : ℕ := 35

/-- The number of pages revised twice -/
def pages_revised_twice : ℕ := 15

/-- The total cost of typing the manuscript -/
def total_cost : ℝ := 860

/-- The cost per page for the first time a page is typed -/
def first_time_cost : ℝ := 6

theorem manuscript_typing_cost :
  first_time_cost * total_pages +
  revision_cost * pages_revised_once +
  2 * revision_cost * pages_revised_twice = total_cost :=
by sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l3362_336253


namespace NUMINAMATH_CALUDE_base_conversion_sum_equality_l3362_336210

def base_to_decimal (digits : List Nat) (base : Nat) : Rat :=
  (digits.reverse.enum.map (λ (i, d) => d * base ^ i)).sum

theorem base_conversion_sum_equality : 
  let a := base_to_decimal [2, 5, 4] 8
  let b := base_to_decimal [1, 2] 4
  let c := base_to_decimal [1, 3, 2] 5
  let d := base_to_decimal [2, 2] 3
  a / b + c / d = 33.9167 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_equality_l3362_336210


namespace NUMINAMATH_CALUDE_laura_debt_l3362_336227

/-- Calculates the total amount owed after applying simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

/-- Proves that Laura owes $36.40 after one year -/
theorem laura_debt : 
  let principal : ℝ := 35
  let rate : ℝ := 0.04
  let time : ℝ := 1
  total_amount_owed principal rate time = 36.40 := by
  sorry

end NUMINAMATH_CALUDE_laura_debt_l3362_336227


namespace NUMINAMATH_CALUDE_age_ratio_in_one_year_l3362_336250

/-- Mike's current age -/
def m : ℕ := sorry

/-- Sarah's current age -/
def s : ℕ := sorry

/-- The condition that 3 years ago, Mike was twice as old as Sarah -/
axiom three_years_ago : m - 3 = 2 * (s - 3)

/-- The condition that 5 years ago, Mike was three times as old as Sarah -/
axiom five_years_ago : m - 5 = 3 * (s - 5)

/-- The number of years until the ratio of their ages is 3:2 -/
def years_until_ratio : ℕ := sorry

/-- The theorem stating that the number of years until the ratio of their ages is 3:2 is 1 -/
theorem age_ratio_in_one_year : 
  years_until_ratio = 1 ∧ (m + years_until_ratio) / (s + years_until_ratio) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_age_ratio_in_one_year_l3362_336250


namespace NUMINAMATH_CALUDE_divisibility_sequence_eventually_periodic_l3362_336282

/-- A sequence of positive integers satisfying the given divisibility property -/
def DivisibilitySequence (a : ℕ → ℕ+) : Prop :=
  ∀ n m : ℕ, (a (n + 2*m)).val ∣ (a n).val + (a (n + m)).val

/-- The sequence is eventually periodic -/
def EventuallyPeriodic (a : ℕ → ℕ+) : Prop :=
  ∃ N d : ℕ, d > 0 ∧ ∀ n : ℕ, n > N → a n = a (n + d)

/-- Main theorem: A sequence satisfying the divisibility property is eventually periodic -/
theorem divisibility_sequence_eventually_periodic (a : ℕ → ℕ+) :
  DivisibilitySequence a → EventuallyPeriodic a := by
  sorry

end NUMINAMATH_CALUDE_divisibility_sequence_eventually_periodic_l3362_336282


namespace NUMINAMATH_CALUDE_remainder_98765432_mod_25_l3362_336226

theorem remainder_98765432_mod_25 : 98765432 % 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98765432_mod_25_l3362_336226


namespace NUMINAMATH_CALUDE_max_sum_at_11_l3362_336234

/-- Arithmetic sequence with first term 21 and common difference -2 -/
def arithmetic_sequence (n : ℕ) : ℚ := 21 - 2 * (n - 1)

/-- Sum of first n terms of the arithmetic sequence -/
def sequence_sum (n : ℕ) : ℚ := (n : ℚ) * (21 + arithmetic_sequence n) / 2

/-- The sum reaches its maximum value when n = 11 -/
theorem max_sum_at_11 : 
  ∀ k : ℕ, k ≠ 0 → sequence_sum 11 ≥ sequence_sum k :=
sorry

end NUMINAMATH_CALUDE_max_sum_at_11_l3362_336234


namespace NUMINAMATH_CALUDE_derivative_of_fraction_l3362_336244

theorem derivative_of_fraction (x : ℝ) :
  let y : ℝ → ℝ := λ x => (1 - Real.cos (2 * x)) / (1 + Real.cos (2 * x))
  HasDerivAt y (4 * Real.sin (2 * x) / (1 + Real.cos (2 * x))^2) x :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_of_fraction_l3362_336244


namespace NUMINAMATH_CALUDE_specific_stack_logs_l3362_336264

/-- Represents a triangular stack of logs. -/
structure LogStack where
  bottom_logs : ℕ  -- Number of logs in the bottom row
  decrement : ℕ    -- Number of logs decreased in each row
  top_logs : ℕ     -- Number of logs in the top row

/-- Calculates the number of rows in a log stack. -/
def num_rows (stack : LogStack) : ℕ :=
  (stack.bottom_logs - stack.top_logs) / stack.decrement + 1

/-- Calculates the total number of logs in a stack. -/
def total_logs (stack : LogStack) : ℕ :=
  let n := num_rows stack
  n * (stack.bottom_logs + stack.top_logs) / 2

/-- Theorem stating the total number of logs in the specific stack. -/
theorem specific_stack_logs :
  let stack : LogStack := ⟨15, 2, 1⟩
  total_logs stack = 64 := by
  sorry


end NUMINAMATH_CALUDE_specific_stack_logs_l3362_336264


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3362_336296

theorem triangle_angle_C (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  Real.sin B + Real.sin A * (Real.sin C - Real.cos C) = 0 →
  a = 2 →
  c = Real.sqrt 2 →
  a / Real.sin A = c / Real.sin C →
  C = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3362_336296


namespace NUMINAMATH_CALUDE_calculate_expression_l3362_336247

theorem calculate_expression : 5 + 4 * (4 - 9)^3 = -495 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3362_336247


namespace NUMINAMATH_CALUDE_ourSystem_is_linear_l3362_336288

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  toFun : ℝ → ℝ → Prop := fun x y => a * x + b * y = c

/-- A system of two equations -/
structure SystemOfTwoEquations where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system we want to prove is linear -/
def ourSystem : SystemOfTwoEquations where
  eq1 := { a := 1, b := 1, c := 5 }
  eq2 := { a := 0, b := 1, c := 2 }

/-- Predicate to check if a system is linear -/
def isLinearSystem (system : SystemOfTwoEquations) : Prop :=
  system.eq1.a ≠ 0 ∨ system.eq1.b ≠ 0 ∧
  system.eq2.a ≠ 0 ∨ system.eq2.b ≠ 0

theorem ourSystem_is_linear : isLinearSystem ourSystem := by
  sorry

end NUMINAMATH_CALUDE_ourSystem_is_linear_l3362_336288


namespace NUMINAMATH_CALUDE_kyles_presents_cost_difference_kyles_presents_cost_difference_is_11_l3362_336283

/-- The cost difference between the first and third present given Kyle's purchases. -/
theorem kyles_presents_cost_difference : ℕ → Prop :=
  fun difference =>
    ∀ (cost_1 cost_2 cost_3 : ℕ),
      cost_1 = 18 →
      cost_2 = cost_1 + 7 →
      cost_3 < cost_1 →
      cost_1 + cost_2 + cost_3 = 50 →
      difference = cost_1 - cost_3

/-- The cost difference between the first and third present is 11. -/
theorem kyles_presents_cost_difference_is_11 : kyles_presents_cost_difference 11 := by
  sorry

end NUMINAMATH_CALUDE_kyles_presents_cost_difference_kyles_presents_cost_difference_is_11_l3362_336283


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3362_336256

theorem quadratic_equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - 5*x₁ + 6 = 0 ∧ x₂^2 - 5*x₂ + 6 = 0 ∧ x₁ = 2 ∧ x₂ = 3) ∧
  (∃ x₁ x₂ : ℝ, 2*x₁^2 - 4*x₁ - 1 = 0 ∧ 2*x₂^2 - 4*x₂ - 1 = 0 ∧ 
    x₁ = (4 + Real.sqrt 24) / 4 ∧ x₂ = (4 - Real.sqrt 24) / 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3362_336256


namespace NUMINAMATH_CALUDE_range_of_alpha_minus_half_beta_l3362_336258

theorem range_of_alpha_minus_half_beta (α β : Real) 
  (h_α : 0 ≤ α ∧ α ≤ π/2) 
  (h_β : π/2 ≤ β ∧ β ≤ π) : 
  ∃ (x : Real), x = α - β/2 ∧ -π/2 ≤ x ∧ x ≤ π/4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_alpha_minus_half_beta_l3362_336258


namespace NUMINAMATH_CALUDE_cheapest_solution_for_1096_days_l3362_336228

/-- Represents the cost and coverage of a ticket type -/
structure Ticket where
  days : ℕ
  cost : ℚ

/-- Finds the minimum cost to cover at least a given number of days using two types of tickets -/
def minCost (ticket1 ticket2 : Ticket) (totalDays : ℕ) : ℚ :=
  sorry

theorem cheapest_solution_for_1096_days :
  let sevenDayTicket : Ticket := ⟨7, 703/100⟩
  let thirtyDayTicket : Ticket := ⟨30, 30⟩
  minCost sevenDayTicket thirtyDayTicket 1096 = 140134/100 := by sorry

end NUMINAMATH_CALUDE_cheapest_solution_for_1096_days_l3362_336228


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l3362_336286

-- Define the streets
inductive Street
| Batman
| Robin
| Joker

-- Define the properties for each street
def termite_ridden_fraction (s : Street) : ℚ :=
  match s with
  | Street.Batman => 1/3
  | Street.Robin => 3/7
  | Street.Joker => 1/2

def collapsing_fraction (s : Street) : ℚ :=
  match s with
  | Street.Batman => 7/10
  | Street.Robin => 4/5
  | Street.Joker => 3/8

-- Theorem to prove
theorem termite_ridden_not_collapsing (s : Street) :
  (termite_ridden_fraction s) * (1 - collapsing_fraction s) =
    match s with
    | Street.Batman => 1/10
    | Street.Robin => 3/35
    | Street.Joker => 5/16
    := by sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l3362_336286


namespace NUMINAMATH_CALUDE_petrov_insurance_cost_l3362_336206

/-- Calculate the total insurance cost for the Petrov family's mortgage --/
def calculate_insurance_cost (apartment_cost loan_amount interest_rate property_rate
                              woman_rate man_rate title_rate maria_share vasily_share : ℝ) : ℝ :=
  let total_loan := loan_amount * (1 + interest_rate)
  let property_cost := total_loan * property_rate
  let title_cost := total_loan * title_rate
  let maria_cost := total_loan * maria_share * woman_rate
  let vasily_cost := total_loan * vasily_share * man_rate
  property_cost + title_cost + maria_cost + vasily_cost

/-- The total insurance cost for the Petrov family's mortgage is 47481.2 rubles --/
theorem petrov_insurance_cost :
  calculate_insurance_cost 13000000 8000000 0.095 0.0009 0.0017 0.0019 0.0027 0.4 0.6 = 47481.2 := by
  sorry

end NUMINAMATH_CALUDE_petrov_insurance_cost_l3362_336206


namespace NUMINAMATH_CALUDE_floor_length_calculation_l3362_336275

/-- Given a rectangular floor with width 8 m, covered by a square carpet with 4 m sides,
    leaving 64 square meters uncovered, the length of the floor is 10 m. -/
theorem floor_length_calculation (floor_width : ℝ) (carpet_side : ℝ) (uncovered_area : ℝ) :
  floor_width = 8 →
  carpet_side = 4 →
  uncovered_area = 64 →
  (floor_width * (carpet_side ^ 2 + uncovered_area) / floor_width) = 10 :=
by
  sorry

#check floor_length_calculation

end NUMINAMATH_CALUDE_floor_length_calculation_l3362_336275


namespace NUMINAMATH_CALUDE_range_of_f_l3362_336297

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 7

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -2} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3362_336297


namespace NUMINAMATH_CALUDE_davids_trip_expenses_l3362_336238

theorem davids_trip_expenses (initial_amount spent_amount remaining_amount : ℕ) : 
  initial_amount = 1800 →
  remaining_amount = 500 →
  spent_amount = initial_amount - remaining_amount →
  spent_amount - remaining_amount = 800 := by
  sorry

end NUMINAMATH_CALUDE_davids_trip_expenses_l3362_336238


namespace NUMINAMATH_CALUDE_aardvark_path_length_l3362_336246

/- Define the radii of the circles -/
def small_radius : ℝ := 15
def large_radius : ℝ := 30

/- Define pi as a real number -/
noncomputable def π : ℝ := Real.pi

/- Theorem statement -/
theorem aardvark_path_length :
  let small_arc := π * small_radius
  let large_arc := π * large_radius
  let radial_segment := large_radius - small_radius
  small_arc + large_arc + 2 * radial_segment = 45 * π + 30 := by
  sorry

#check aardvark_path_length

end NUMINAMATH_CALUDE_aardvark_path_length_l3362_336246


namespace NUMINAMATH_CALUDE_sammy_score_l3362_336255

theorem sammy_score (sammy_score : ℕ) (gab_score : ℕ) (cher_score : ℕ) (opponent_score : ℕ) :
  gab_score = 2 * sammy_score →
  cher_score = 2 * gab_score →
  opponent_score = 85 →
  sammy_score + gab_score + cher_score = opponent_score + 55 →
  sammy_score = 20 := by
sorry

end NUMINAMATH_CALUDE_sammy_score_l3362_336255


namespace NUMINAMATH_CALUDE_book_pages_l3362_336222

theorem book_pages : ∀ (total : ℕ), 
  (total : ℚ) * (1 - 2/5) * (1 - 5/8) = 36 → 
  total = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l3362_336222


namespace NUMINAMATH_CALUDE_total_handshakes_l3362_336249

-- Define the number of twin sets and triplet sets
def twin_sets : ℕ := 12
def triplet_sets : ℕ := 8

-- Define the total number of twins and triplets
def total_twins : ℕ := twin_sets * 2
def total_triplets : ℕ := triplet_sets * 3

-- Define the number of handshakes for each twin and triplet
def twin_handshakes : ℕ := (total_twins - 2) + (total_triplets * 3 / 4)
def triplet_handshakes : ℕ := (total_triplets - 3) + (total_twins * 1 / 4)

-- Theorem to prove
theorem total_handshakes : 
  (total_twins * twin_handshakes + total_triplets * triplet_handshakes) / 2 = 804 := by
  sorry

end NUMINAMATH_CALUDE_total_handshakes_l3362_336249


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3362_336252

theorem quadratic_function_range (x : ℝ) :
  let y := x^2 - 4*x + 3
  y < 0 ↔ 1 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3362_336252


namespace NUMINAMATH_CALUDE_soap_boxes_in_carton_l3362_336221

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the maximum number of smaller boxes that can fit in a larger box -/
def maxBoxesFit (largeBox smallBox : BoxDimensions) : ℕ :=
  (boxVolume largeBox) / (boxVolume smallBox)

theorem soap_boxes_in_carton :
  let carton : BoxDimensions := ⟨25, 42, 60⟩
  let soapBox : BoxDimensions := ⟨7, 12, 5⟩
  maxBoxesFit carton soapBox = 150 := by
  sorry

end NUMINAMATH_CALUDE_soap_boxes_in_carton_l3362_336221


namespace NUMINAMATH_CALUDE_tangent_line_at_2_2_tangent_lines_through_origin_l3362_336219

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x + 1

-- Theorem for part (I)
theorem tangent_line_at_2_2 :
  ∃ (m b : ℝ), (∀ x y, y = m*x + b ↔ 5*x - y - 8 = 0) ∧
               f 2 = 2 ∧
               f' 2 = m :=
sorry

-- Theorem for part (II)
theorem tangent_lines_through_origin :
  ∃ (x₁ x₂ : ℝ),
    (f x₁ = 0 ∧ f' x₁ = 1 ∧ x₁ ≠ x₂) ∧
    (f x₂ = 0 ∧ f' x₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_2_tangent_lines_through_origin_l3362_336219


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l3362_336259

-- Define the slopes and y-intercept
def m₁ : ℝ := 12
def m₂ : ℝ := 8
def b : ℝ := sorry

-- Define the x-intercepts
def u : ℝ := sorry
def v : ℝ := sorry

-- Define the lines
def line₁ (x : ℝ) : ℝ := m₁ * x + b
def line₂ (x : ℝ) : ℝ := m₂ * x + b

-- State the theorem
theorem x_intercept_ratio : 
  b ≠ 0 ∧ 
  line₁ u = 0 ∧ 
  line₂ v = 0 → 
  u / v = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l3362_336259


namespace NUMINAMATH_CALUDE_last_remaining_number_l3362_336236

def josephus_variant (n : ℕ) : ℕ :=
  let rec aux (k m : ℕ) : ℕ :=
    if k ≤ 1 then m
    else
      let m' := (m + 1) % k
      aux (k - 1) (2 * m' + 1)
  aux n 0

theorem last_remaining_number :
  josephus_variant 150 = 73 := by sorry

end NUMINAMATH_CALUDE_last_remaining_number_l3362_336236


namespace NUMINAMATH_CALUDE_simplify_fraction_l3362_336266

theorem simplify_fraction : (126 : ℚ) / 11088 = 1 / 88 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3362_336266


namespace NUMINAMATH_CALUDE_quadratic_root_implies_p_value_l3362_336223

theorem quadratic_root_implies_p_value (q p : ℝ) (h : Complex.I * Complex.I = -1) :
  (3 : ℂ) * (4 + Complex.I)^2 - q * (4 + Complex.I) + p = 0 → p = 51 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_p_value_l3362_336223


namespace NUMINAMATH_CALUDE_fraction_simplification_l3362_336251

theorem fraction_simplification :
  (16 : ℚ) / 54 * 27 / 8 * 64 / 81 = 64 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3362_336251


namespace NUMINAMATH_CALUDE_equatorial_circumference_scientific_notation_l3362_336280

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ a
  h2 : a < 10

/-- Check if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (x : ℝ) : Prop :=
  sn.a * (10 : ℝ) ^ sn.n = x

/-- The equatorial circumference in meters -/
def equatorialCircumference : ℝ := 40000000

/-- Theorem stating that 4 × 10^7 is the correct scientific notation for the equatorial circumference -/
theorem equatorial_circumference_scientific_notation :
  ∃ sn : ScientificNotation, sn.a = 4 ∧ sn.n = 7 ∧ represents sn equatorialCircumference :=
sorry

end NUMINAMATH_CALUDE_equatorial_circumference_scientific_notation_l3362_336280


namespace NUMINAMATH_CALUDE_solve_x_equation_l3362_336292

theorem solve_x_equation : ∃ x : ℝ, (0.6 * x = x / 3 + 110) ∧ x = 412.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_x_equation_l3362_336292


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3362_336204

theorem inequality_solution_set (a b : ℝ) : 
  {x : ℝ | a * x > b} ≠ Set.Iio (-b/a) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3362_336204


namespace NUMINAMATH_CALUDE_sum_of_common_divisors_l3362_336261

def number_list : List Int := [24, 48, -18, 108, 72]

def is_common_divisor (d : Nat) : Bool :=
  number_list.all (fun n => n % d == 0)

def common_divisors : List Nat :=
  (List.range 108).filter is_common_divisor

theorem sum_of_common_divisors : (common_divisors.sum) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_divisors_l3362_336261


namespace NUMINAMATH_CALUDE_equation_solution_l3362_336202

theorem equation_solution : 
  let eq := fun x : ℝ => 81 * (1 - x)^2 - 64
  ∃ (x1 x2 : ℝ), x1 = 1/9 ∧ x2 = 17/9 ∧ eq x1 = 0 ∧ eq x2 = 0 ∧
  ∀ (x : ℝ), eq x = 0 → x = x1 ∨ x = x2 :=
by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3362_336202


namespace NUMINAMATH_CALUDE_equivalent_operations_l3362_336265

theorem equivalent_operations (x : ℝ) : 
  (x * (2/5)) / (4/7) = x * (7/10) := by
sorry

end NUMINAMATH_CALUDE_equivalent_operations_l3362_336265


namespace NUMINAMATH_CALUDE_square_of_complex_l3362_336294

theorem square_of_complex (z : ℂ) (i : ℂ) : z = 5 + 2 * i → i ^ 2 = -1 → z ^ 2 = 21 + 20 * i := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_l3362_336294


namespace NUMINAMATH_CALUDE_intersection_point_modulo_9_l3362_336232

theorem intersection_point_modulo_9 :
  ∀ x : ℕ, (3 * x + 6) % 9 = (7 * x + 3) % 9 → x % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_modulo_9_l3362_336232


namespace NUMINAMATH_CALUDE_cereal_servings_l3362_336267

theorem cereal_servings (cups_per_serving : ℝ) (total_cups_needed : ℝ) 
  (h1 : cups_per_serving = 2.0)
  (h2 : total_cups_needed = 36) :
  total_cups_needed / cups_per_serving = 18 := by
  sorry

end NUMINAMATH_CALUDE_cereal_servings_l3362_336267


namespace NUMINAMATH_CALUDE_cloth_selling_price_l3362_336224

/-- Calculates the total selling price of cloth given the quantity sold, profit per meter, and cost price per meter. -/
def total_selling_price (quantity : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  quantity * (profit_per_meter + cost_price_per_meter)

/-- Proves that the total selling price of 66 meters of cloth with a profit of Rs. 5 per meter and a cost price of Rs. 5 per meter is Rs. 660. -/
theorem cloth_selling_price :
  total_selling_price 66 5 5 = 660 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l3362_336224


namespace NUMINAMATH_CALUDE_johns_allowance_l3362_336203

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℝ) : A = 2.40 :=
  let arcade_spent := (3 : ℝ) / 5 * A
  let remaining_after_arcade := A - arcade_spent
  let toy_store_spent := (1 : ℝ) / 3 * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spent
  by
    have h1 : remaining_after_toy_store = 0.64
    sorry
    -- Proof goes here
    sorry

#check johns_allowance

end NUMINAMATH_CALUDE_johns_allowance_l3362_336203


namespace NUMINAMATH_CALUDE_prime_divisibility_equivalence_l3362_336290

theorem prime_divisibility_equivalence (p : ℕ) (hp : Nat.Prime p) :
  (∃ x : ℤ, ∃ d₁ : ℤ, x^2 - x + 3 = d₁ * p) ↔ 
  (∃ y : ℤ, ∃ d₂ : ℤ, y^2 - y + 25 = d₂ * p) := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_equivalence_l3362_336290


namespace NUMINAMATH_CALUDE_roots_on_circle_l3362_336298

theorem roots_on_circle : ∃ (r : ℝ), r = 1 / Real.sqrt 3 ∧
  ∀ (z : ℂ), (z - 1)^3 = 8*z^3 → Complex.abs (z + 1/3) = r := by
  sorry

end NUMINAMATH_CALUDE_roots_on_circle_l3362_336298


namespace NUMINAMATH_CALUDE_hyperbola_center_trajectory_l3362_336239

/-- The hyperbola equation with parameter m -/
def hyperbola (x y m : ℝ) : Prop :=
  x^2 - y^2 - 6*m*x - 4*m*y + 5*m^2 - 1 = 0

/-- The trajectory equation of the center -/
def trajectory_equation (x y : ℝ) : Prop :=
  2*x + 3*y = 0

/-- Theorem stating that the trajectory equation of the center of the hyperbola
    is 2x + 3y = 0 for all real m -/
theorem hyperbola_center_trajectory :
  ∀ m : ℝ, ∃ x y : ℝ, hyperbola x y m ∧ trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_trajectory_l3362_336239


namespace NUMINAMATH_CALUDE_sqrt_ab_max_value_l3362_336205

theorem sqrt_ab_max_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  ∃ (m : ℝ), m = 1/2 ∧ ∀ x, x = Real.sqrt (a * b) → x ≤ m :=
sorry

end NUMINAMATH_CALUDE_sqrt_ab_max_value_l3362_336205


namespace NUMINAMATH_CALUDE_seating_arrangement_l3362_336270

theorem seating_arrangement (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l3362_336270


namespace NUMINAMATH_CALUDE_min_beacons_proof_l3362_336271

/-- Represents a room in the maze --/
structure Room :=
  (x : Nat) (y : Nat)

/-- Represents the maze structure --/
def Maze := List Room

/-- Calculates the distance between two rooms --/
def distance (r1 r2 : Room) : Nat :=
  sorry

/-- Checks if a set of beacon positions allows unambiguous location determination --/
def is_unambiguous (maze : Maze) (beacons : List Room) : Prop :=
  sorry

/-- The minimum number of beacons required --/
def min_beacons : Nat := 3

/-- The specific beacon positions that work --/
def beacon_positions : List Room :=
  [⟨1, 1⟩, ⟨4, 3⟩, ⟨1, 5⟩]  -- Representing a1, d3, a5

theorem min_beacons_proof (maze : Maze) :
  (∀ beacons : List Room, beacons.length < min_beacons → ¬ is_unambiguous maze beacons) ∧
  is_unambiguous maze beacon_positions :=
sorry

end NUMINAMATH_CALUDE_min_beacons_proof_l3362_336271
