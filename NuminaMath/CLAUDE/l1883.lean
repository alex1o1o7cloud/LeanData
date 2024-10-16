import Mathlib

namespace NUMINAMATH_CALUDE_line_problem_l1883_188374

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_problem (a b : ℝ) :
  let l0 : Line := ⟨1, -1, 1⟩
  let l1 : Line := ⟨a, -2, 1⟩
  let l2 : Line := ⟨1, b, 3⟩
  perpendicular l0 l1 → parallel l0 l2 → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_problem_l1883_188374


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1883_188301

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n, a (n + 1) = (a n : ℚ) * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℕ)
  (h_geometric : is_geometric_sequence a)
  (h_first : a 1 = 5)
  (h_fifth : a 5 = 1280) :
  a 4 = 320 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1883_188301


namespace NUMINAMATH_CALUDE_burgers_per_day_l1883_188315

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The cost of each burger in dollars -/
def burger_cost : ℕ := 13

/-- The total amount Alice spent on burgers in June in dollars -/
def total_spent : ℕ := 1560

/-- Alice bought burgers every day in June -/
axiom bought_daily : ∀ d : ℕ, d ≤ june_days → ∃ b : ℕ, b > 0

/-- Theorem: Alice purchased 4 burgers per day in June -/
theorem burgers_per_day : 
  (total_spent / burger_cost) / june_days = 4 := by sorry

end NUMINAMATH_CALUDE_burgers_per_day_l1883_188315


namespace NUMINAMATH_CALUDE_sum_of_sequence_equals_63_over_19_l1883_188349

def A : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n + 2) => 2 * A (n + 1) + A n

theorem sum_of_sequence_equals_63_over_19 :
  ∑' n, A n / 5^n = 63 / 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_sequence_equals_63_over_19_l1883_188349


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l1883_188376

theorem largest_five_digit_divisible_by_8 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 8 = 0 → n ≤ 99992 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l1883_188376


namespace NUMINAMATH_CALUDE_product_of_geometric_terms_l1883_188370

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem product_of_geometric_terms
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_sum : a 3 + a 11 = 8)
  (h_equal : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_geometric_terms_l1883_188370


namespace NUMINAMATH_CALUDE_inequality_solutions_l1883_188335

def inequality (a x : ℝ) := a * x^2 - (a + 2) * x + 2 < 0

theorem inequality_solutions :
  ∀ a : ℝ,
    (a = -1 → {x : ℝ | inequality a x} = {x : ℝ | x < -2 ∨ x > 1}) ∧
    (a = 0 → {x : ℝ | inequality a x} = {x : ℝ | x > 1}) ∧
    (a < 0 → {x : ℝ | inequality a x} = {x : ℝ | x < 2/a ∨ x > 1}) ∧
    (0 < a ∧ a < 2 → {x : ℝ | inequality a x} = {x : ℝ | 1 < x ∧ x < 2/a}) ∧
    (a = 2 → {x : ℝ | inequality a x} = ∅) ∧
    (a > 2 → {x : ℝ | inequality a x} = {x : ℝ | 2/a < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l1883_188335


namespace NUMINAMATH_CALUDE_spider_web_production_l1883_188310

def spider_webs (num_spiders : ℕ) (num_webs : ℕ) (days : ℕ) : Prop :=
  num_spiders = num_webs ∧ days > 0

theorem spider_web_production 
  (h1 : spider_webs 7 7 (7 : ℕ)) 
  (h2 : spider_webs 1 1 7) : 
  ∀ s, s ≤ 7 → spider_webs 1 1 7 :=
sorry

end NUMINAMATH_CALUDE_spider_web_production_l1883_188310


namespace NUMINAMATH_CALUDE_no_valid_p_exists_l1883_188320

theorem no_valid_p_exists (p M : ℝ) (hp : 0 < p) (hM : 0 < M) (hp2 : p < 2) : 
  ¬∃ p, M * (1 + p / 100) * (1 - 50 * p / 100) > M :=
by
  sorry

end NUMINAMATH_CALUDE_no_valid_p_exists_l1883_188320


namespace NUMINAMATH_CALUDE_inequality_solution_l1883_188361

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 3/4) ↔ (-2 < x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1883_188361


namespace NUMINAMATH_CALUDE_chimney_bricks_chimney_bricks_proof_l1883_188346

theorem chimney_bricks : ℕ → Prop :=
  fun n =>
    let brenda_rate := n / 12
    let brandon_rate := n / 15
    let combined_rate := n / 12 + n / 15 - 15
    6 * combined_rate = n →
    n = 900

-- The proof is omitted
theorem chimney_bricks_proof : chimney_bricks 900 := by sorry

end NUMINAMATH_CALUDE_chimney_bricks_chimney_bricks_proof_l1883_188346


namespace NUMINAMATH_CALUDE_max_free_squares_l1883_188305

/-- Represents a chessboard with bugs -/
structure BugChessboard (n : ℕ) where
  size : ℕ
  size_eq : size = n

/-- Represents a valid move of bugs on the chessboard -/
def ValidMove (board : BugChessboard n) : Prop :=
  ∀ i j : ℕ, i < n ∧ j < n →
    ∃ (i₁ j₁ i₂ j₂ : ℕ), 
      (i₁ < n ∧ j₁ < n ∧ i₂ < n ∧ j₂ < n) ∧
      ((i₁ = i ∧ (j₁ = j + 1 ∨ j₁ = j - 1)) ∨ (j₁ = j ∧ (i₁ = i + 1 ∨ i₁ = i - 1))) ∧
      ((i₂ = i ∧ (j₂ = j + 1 ∨ j₂ = j - 1)) ∨ (j₂ = j ∧ (i₂ = i + 1 ∨ i₂ = i - 1))) ∧
      (i₁ ≠ i₂ ∨ j₁ ≠ j₂)

/-- The number of free squares after a valid move -/
def FreeSquares (board : BugChessboard n) (move : ValidMove board) : ℕ := sorry

/-- The main theorem: the maximal number of free squares after one move is n^2 -/
theorem max_free_squares (n : ℕ) (board : BugChessboard n) :
  ∃ (move : ValidMove board), FreeSquares board move = n^2 :=
sorry

end NUMINAMATH_CALUDE_max_free_squares_l1883_188305


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_29_l1883_188347

theorem modular_inverse_of_3_mod_29 : ∃ x : ℕ, x < 29 ∧ (3 * x) % 29 = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_29_l1883_188347


namespace NUMINAMATH_CALUDE_triangle_inradius_l1883_188380

/-- Given a triangle with perimeter 20 cm and area 25 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) :
  p = 20 →
  A = 25 →
  A = r * p / 2 →
  r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l1883_188380


namespace NUMINAMATH_CALUDE_tom_average_speed_l1883_188387

theorem tom_average_speed (karen_speed : ℝ) (karen_delay : ℝ) (win_margin : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 →
  karen_delay = 4 / 60 →
  win_margin = 4 →
  tom_distance = 24 →
  ∃ (tom_speed : ℝ), tom_speed = 300 / 7 ∧
    karen_speed * (tom_distance / karen_speed) = 
    tom_speed * (tom_distance / karen_speed + karen_delay) - win_margin :=
by sorry

end NUMINAMATH_CALUDE_tom_average_speed_l1883_188387


namespace NUMINAMATH_CALUDE_five_consecutive_integers_product_not_square_l1883_188325

theorem five_consecutive_integers_product_not_square (n : ℕ+) :
  ∃ m : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) : ℕ) ≠ m^2 := by
  sorry

end NUMINAMATH_CALUDE_five_consecutive_integers_product_not_square_l1883_188325


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1883_188306

theorem fraction_evaluation : (((5 * 4) + 6) : ℝ) / 10 = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1883_188306


namespace NUMINAMATH_CALUDE_f_min_value_a_range_l1883_188322

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x - 2) - x + 5

-- Theorem for the minimum value of f
theorem f_min_value :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 3 :=
sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∀ x, |x - a| + |x + 2| ≥ 3) → (a ≤ -5 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_f_min_value_a_range_l1883_188322


namespace NUMINAMATH_CALUDE_mixture_weight_l1883_188373

-- Define the constants
def cashew_price : ℝ := 5.00
def peanut_price : ℝ := 2.00
def total_price : ℝ := 92.00
def cashew_weight : ℝ := 11.00

-- Define the theorem
theorem mixture_weight :
  ∃ (peanut_weight : ℝ),
    cashew_price * cashew_weight + peanut_price * peanut_weight = total_price ∧
    cashew_weight + peanut_weight = 29.5 := by
  sorry

end NUMINAMATH_CALUDE_mixture_weight_l1883_188373


namespace NUMINAMATH_CALUDE_zac_strawberries_l1883_188359

def strawberry_problem (total : ℕ) (jonathan_matthew : ℕ) (matthew_zac : ℕ) : Prop :=
  ∃ (jonathan matthew zac : ℕ),
    jonathan + matthew + zac = total ∧
    jonathan + matthew = jonathan_matthew ∧
    matthew + zac = matthew_zac ∧
    zac = 200

theorem zac_strawberries :
  strawberry_problem 550 350 250 :=
sorry

end NUMINAMATH_CALUDE_zac_strawberries_l1883_188359


namespace NUMINAMATH_CALUDE_work_break_difference_l1883_188352

/-- Calculates the difference between water breaks and sitting breaks
    given work duration and break intervals. -/
def break_difference (work_duration : ℕ) (water_interval : ℕ) (sitting_interval : ℕ) : ℕ :=
  (work_duration / water_interval) - (work_duration / sitting_interval)

/-- Proves that for 240 minutes of work, with water breaks every 20 minutes
    and sitting breaks every 120 minutes, there are 10 more water breaks than sitting breaks. -/
theorem work_break_difference :
  break_difference 240 20 120 = 10 := by
  sorry

end NUMINAMATH_CALUDE_work_break_difference_l1883_188352


namespace NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l1883_188317

def OddUnitsDigits : Set Nat := {1, 3, 5, 7, 9}

def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem smallest_non_odd_units_digit :
  ∃ (d : Nat), d ∈ Digits ∧ d ∉ OddUnitsDigits ∧ ∀ (x : Nat), x ∈ Digits ∧ x ∉ OddUnitsDigits → d ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l1883_188317


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1883_188344

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 6 * x - 8 = 2 * (x - 4) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1883_188344


namespace NUMINAMATH_CALUDE_train_distance_problem_l1883_188396

theorem train_distance_problem (speed1 speed2 distance_difference : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 25)
  (h3 : distance_difference = 65)
  : speed1 * (distance_difference / (speed2 - speed1)) + 
    speed2 * (distance_difference / (speed2 - speed1)) = 585 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l1883_188396


namespace NUMINAMATH_CALUDE_trace_equation_equiv_equal_distance_l1883_188343

/-- 
For any point P(x, y) in a 2D coordinate system, the trace equation y = |x| 
is equivalent to the condition that the distance from P to the x-axis 
is equal to the distance from P to the y-axis.
-/
theorem trace_equation_equiv_equal_distance (x y : ℝ) : 
  y = |x| ↔ |y| = |x| :=
sorry

end NUMINAMATH_CALUDE_trace_equation_equiv_equal_distance_l1883_188343


namespace NUMINAMATH_CALUDE_solution_difference_l1883_188333

theorem solution_difference (p q : ℝ) : 
  ((6 * p - 18) / (p^2 + 3*p - 18) = p + 3) →
  ((6 * q - 18) / (q^2 + 3*q - 18) = q + 3) →
  p ≠ q →
  p > q →
  p - q = 9 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1883_188333


namespace NUMINAMATH_CALUDE_product_of_roots_l1883_188345

theorem product_of_roots (x : ℂ) :
  2 * x^3 - 3 * x^2 - 10 * x + 14 = 0 →
  ∃ r₁ r₂ r₃ : ℂ, (x - r₁) * (x - r₂) * (x - r₃) = 2 * x^3 - 3 * x^2 - 10 * x + 14 ∧ r₁ * r₂ * r₃ = -7 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1883_188345


namespace NUMINAMATH_CALUDE_second_smallest_prime_perimeter_l1883_188326

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers can form a triangle -/
def isTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are distinct -/
def areDistinct (a b c : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ b ≠ c

/-- The main theorem stating that the second smallest perimeter of a scalene triangle
    with distinct prime sides and a prime perimeter is 29 -/
theorem second_smallest_prime_perimeter :
  ∃ (a b c : ℕ),
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    areDistinct a b c ∧
    isTriangle a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 29) ∧
    (∀ (x y z : ℕ),
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      areDistinct x y z ∧
      isTriangle x y z ∧
      isPrime (x + y + z) ∧
      (x + y + z < 29) →
      (x + y + z = 23)) :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_prime_perimeter_l1883_188326


namespace NUMINAMATH_CALUDE_pen_price_is_14_l1883_188303

/-- The price of a pen in yuan -/
def pen_price : ℝ := 14

/-- The price of a ballpoint pen in yuan -/
def ballpoint_price : ℝ := 7

/-- The total cost of the pens in yuan -/
def total_cost : ℝ := 49

theorem pen_price_is_14 :
  (2 * pen_price + 3 * ballpoint_price = total_cost) ∧
  (3 * pen_price + ballpoint_price = total_cost) →
  pen_price = 14 := by
sorry

end NUMINAMATH_CALUDE_pen_price_is_14_l1883_188303


namespace NUMINAMATH_CALUDE_all_diagonal_triangles_multiplicative_l1883_188392

/-- A regular polygon with n sides, all of length 1 -/
structure RegularPolygon (n : ℕ) where
  (n_ge_3 : n ≥ 3)
  (side_length : ℝ := 1)

/-- A triangle formed by diagonals in a regular polygon -/
structure DiagonalTriangle (n : ℕ) (p : RegularPolygon n) where
  (vertex1 : ℝ × ℝ)
  (vertex2 : ℝ × ℝ)
  (vertex3 : ℝ × ℝ)

/-- A triangle is multiplicative if the product of the lengths of two sides equals the length of the third side -/
def is_multiplicative (t : DiagonalTriangle n p) : Prop :=
  ∀ (i j k : Fin 3), i ≠ j → j ≠ k → i ≠ k →
    let sides := [dist t.vertex1 t.vertex2, dist t.vertex2 t.vertex3, dist t.vertex3 t.vertex1]
    sides[i] * sides[j] = sides[k]

/-- The main theorem: all triangles formed by diagonals in a regular polygon are multiplicative -/
theorem all_diagonal_triangles_multiplicative (n : ℕ) (p : RegularPolygon n) :
  ∀ t : DiagonalTriangle n p, is_multiplicative t :=
sorry

end NUMINAMATH_CALUDE_all_diagonal_triangles_multiplicative_l1883_188392


namespace NUMINAMATH_CALUDE_celebrity_baby_matching_probability_l1883_188321

theorem celebrity_baby_matching_probability :
  let n : ℕ := 4
  let total_arrangements := n.factorial
  let correct_arrangements : ℕ := 1
  (correct_arrangements : ℚ) / total_arrangements = 1 / 24 :=
by sorry

end NUMINAMATH_CALUDE_celebrity_baby_matching_probability_l1883_188321


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1883_188307

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + m = 0 ∧ x = 2) → 
  (∃ y : ℝ, y^2 - 6*y + m = 0 ∧ y = 4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1883_188307


namespace NUMINAMATH_CALUDE_jellybean_count_l1883_188316

/-- The number of jellybeans in a dozen -/
def dozen : ℕ := 12

/-- Caleb's number of jellybeans -/
def caleb_jellybeans : ℕ := 3 * dozen

/-- Sophie's number of jellybeans -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2

/-- The total number of jellybeans Caleb and Sophie have -/
def total_jellybeans : ℕ := caleb_jellybeans + sophie_jellybeans

theorem jellybean_count : total_jellybeans = 54 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l1883_188316


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_350_l1883_188395

theorem closest_perfect_square_to_350 :
  ∀ n : ℕ, n ≠ 19 → (n ^ 2 : ℝ) ≠ 361 → |350 - (19 ^ 2 : ℝ)| < |350 - (n ^ 2 : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_350_l1883_188395


namespace NUMINAMATH_CALUDE_apple_count_in_bowl_l1883_188323

theorem apple_count_in_bowl (initial_oranges : ℕ) (removed_oranges : ℕ) (apple_percentage : ℚ) :
  initial_oranges = 25 →
  removed_oranges = 19 →
  apple_percentage = 70 / 100 →
  ∃ (apples : ℕ),
    apples = 14 ∧
    apples / (apples + (initial_oranges - removed_oranges)) = apple_percentage :=
by sorry

end NUMINAMATH_CALUDE_apple_count_in_bowl_l1883_188323


namespace NUMINAMATH_CALUDE_min_value_problem_l1883_188382

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1/a + 2) * (1/b + 2) ≥ 16 ∧
  ((1/a + 2) * (1/b + 2) = 16 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1883_188382


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1883_188318

theorem right_triangle_segment_ratio (a b c r s : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 →  -- Positive lengths
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  r * s = a^2 →  -- Geometric mean theorem for r
  r * s = b^2 →  -- Geometric mean theorem for s
  r + s = c →  -- r and s form the hypotenuse
  a / b = 1 / 4 →  -- Given ratio
  r / s = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1883_188318


namespace NUMINAMATH_CALUDE_water_depth_relation_l1883_188394

/-- Represents a cylindrical water tank -/
structure WaterTank where
  height : ℝ
  baseDiameter : ℝ

/-- Calculates the water depth when the tank is upright -/
def uprightDepth (tank : WaterTank) (horizontalDepth : ℝ) : ℝ :=
  sorry

/-- Theorem stating the relation between horizontal and upright water depths -/
theorem water_depth_relation (tank : WaterTank) (horizontalDepth : ℝ) :
  tank.height = 20 →
  tank.baseDiameter = 5 →
  horizontalDepth = 4 →
  abs (uprightDepth tank horizontalDepth - 8.1) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_relation_l1883_188394


namespace NUMINAMATH_CALUDE_star_calculation_l1883_188337

-- Define the ★ operation
def star (a b : ℚ) : ℚ := (a + b) / 3

-- Theorem statement
theorem star_calculation : star (star 7 15) 10 = 52 / 9 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l1883_188337


namespace NUMINAMATH_CALUDE_distance_sum_difference_bound_l1883_188336

-- Define a convex dodecagon
def ConvexDodecagon : Type := Unit

-- Define a point inside the dodecagon
def Point : Type := Unit

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define the vertices of the dodecagon
def vertices (d : ConvexDodecagon) : Finset Point := sorry

-- Define the sum of distances from a point to all vertices
def sum_distances (p : Point) (d : ConvexDodecagon) : ℝ :=
  (vertices d).sum (λ v => distance p v)

-- The main theorem
theorem distance_sum_difference_bound
  (d : ConvexDodecagon) (p q : Point)
  (h : distance p q = 10) :
  |sum_distances p d - sum_distances q d| < 100 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_difference_bound_l1883_188336


namespace NUMINAMATH_CALUDE_unique_solution_l1883_188367

-- Define the properties of p, q, and r
def is_valid_solution (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Nat.Prime r ∧
  q - p = r ∧
  5 < p ∧ p < 15 ∧
  q < 15

-- Theorem statement
theorem unique_solution : 
  ∃! q : ℕ, ∃ (p r : ℕ), is_valid_solution p q r ∧ q = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1883_188367


namespace NUMINAMATH_CALUDE_souvenir_problem_l1883_188324

/-- Represents the cost and quantity of souvenirs --/
structure SouvenirPlan where
  costA : ℕ
  costB : ℕ
  quantityA : ℕ
  quantityB : ℕ

/-- Checks if a souvenir plan satisfies all conditions --/
def isValidPlan (plan : SouvenirPlan) : Prop :=
  plan.costA + 20 = plan.costB ∧
  9 * plan.costA = 7 * plan.costB ∧
  plan.quantityB = 2 * plan.quantityA + 5 ∧
  plan.quantityA ≥ 18 ∧
  plan.costA * plan.quantityA + plan.costB * plan.quantityB ≤ 5450

/-- The correct costs and possible purchasing plans --/
def correctSolution : Prop :=
  ∃ (plan : SouvenirPlan),
    isValidPlan plan ∧
    plan.costA = 70 ∧
    plan.costB = 90 ∧
    (plan.quantityA = 18 ∧ plan.quantityB = 41) ∨
    (plan.quantityA = 19 ∧ plan.quantityB = 43) ∨
    (plan.quantityA = 20 ∧ plan.quantityB = 45)

theorem souvenir_problem : correctSolution := by
  sorry

end NUMINAMATH_CALUDE_souvenir_problem_l1883_188324


namespace NUMINAMATH_CALUDE_equation_is_linear_in_two_vars_l1883_188351

/-- A linear equation in two variables -/
structure LinearEquation2Var where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop
  is_linear : ∀ x y, eq x y ↔ a * x + b * y + c = 0

/-- The equation y - x = 1 -/
def equation : ℝ → ℝ → Prop :=
  fun x y => y - x = 1

theorem equation_is_linear_in_two_vars :
  ∃ le : LinearEquation2Var, le.eq = equation :=
sorry

end NUMINAMATH_CALUDE_equation_is_linear_in_two_vars_l1883_188351


namespace NUMINAMATH_CALUDE_max_remaining_area_l1883_188399

/-- Represents the dimensions of a small rectangle cut from the corner --/
structure SmallRectangle where
  length : ℕ
  width : ℕ

/-- The original cardboard dimensions --/
def original_length : ℕ := 12
def original_width : ℕ := 11

/-- The list of small rectangles cut from the corners --/
def cut_rectangles : List SmallRectangle := [
  ⟨1, 4⟩, ⟨2, 2⟩, ⟨2, 3⟩, ⟨2, 3⟩
]

/-- Calculate the area of a rectangle --/
def area (length width : ℕ) : ℕ := length * width

/-- Calculate the total area cut out --/
def total_area_cut (rectangles : List SmallRectangle) : ℕ :=
  rectangles.foldl (fun acc rect => acc + area rect.length rect.width) 0

/-- The theorem to be proved --/
theorem max_remaining_area :
  area original_length original_width - total_area_cut cut_rectangles = 112 := by
  sorry


end NUMINAMATH_CALUDE_max_remaining_area_l1883_188399


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inv_sqrt_x_l1883_188348

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inv_sqrt_x_l1883_188348


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1883_188300

theorem right_triangle_hypotenuse : ∀ (short_leg long_leg hypotenuse : ℝ),
  short_leg > 0 →
  long_leg > 0 →
  hypotenuse > 0 →
  long_leg = 2 * short_leg - 1 →
  (1 / 2) * short_leg * long_leg = 60 →
  short_leg ^ 2 + long_leg ^ 2 = hypotenuse ^ 2 →
  hypotenuse = 17 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1883_188300


namespace NUMINAMATH_CALUDE_cards_lost_l1883_188371

theorem cards_lost (initial_cards : ℝ) (final_cards : ℕ) : 
  initial_cards = 47.0 → final_cards = 40 → initial_cards - final_cards = 7 := by
  sorry

end NUMINAMATH_CALUDE_cards_lost_l1883_188371


namespace NUMINAMATH_CALUDE_largest_c_for_negative_four_in_range_l1883_188364

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + c

-- State the theorem
theorem largest_c_for_negative_four_in_range :
  (∃ (c : ℝ), ∀ (c' : ℝ), 
    (∃ (x : ℝ), f c' x = -4) → c' ≤ c) ∧
  (∃ (x : ℝ), f (9/4) x = -4) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_four_in_range_l1883_188364


namespace NUMINAMATH_CALUDE_work_completion_time_l1883_188334

/-- Given Johnson's and Vincent's individual work rates, calculates the time required for them to complete the work together -/
theorem work_completion_time (johnson_rate vincent_rate : ℚ) 
  (h1 : johnson_rate = 1 / 10)
  (h2 : vincent_rate = 1 / 40) :
  1 / (johnson_rate + vincent_rate) = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1883_188334


namespace NUMINAMATH_CALUDE_penguin_count_l1883_188390

theorem penguin_count (zebras tigers zookeepers : ℕ) 
  (h1 : zebras = 22)
  (h2 : tigers = 8)
  (h3 : zookeepers = 12)
  (h4 : ∀ (penguins : ℕ), 
    (penguins + zebras + tigers + zookeepers) + 132 = 
    4 * penguins + 4 * zebras + 4 * tigers + 2 * zookeepers) :
  ∃ (penguins : ℕ), penguins = 10 := by
sorry

end NUMINAMATH_CALUDE_penguin_count_l1883_188390


namespace NUMINAMATH_CALUDE_blue_balls_drawn_first_probability_l1883_188385

def num_blue_balls : ℕ := 4
def num_yellow_balls : ℕ := 3
def total_balls : ℕ := num_blue_balls + num_yellow_balls

def favorable_outcomes : ℕ := Nat.choose (total_balls - 1) num_yellow_balls
def total_outcomes : ℕ := Nat.choose total_balls num_yellow_balls

theorem blue_balls_drawn_first_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_drawn_first_probability_l1883_188385


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_l1883_188340

theorem tan_pi_minus_alpha (α : Real) (h : 3 * Real.sin α = Real.cos α) :
  Real.tan (π - α) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_l1883_188340


namespace NUMINAMATH_CALUDE_min_c_value_l1883_188314

theorem min_c_value (a b c : ℕ+) (h1 : a ≤ b) (h2 : b < c)
  (h3 : ∃! p : ℝ × ℝ, (2 * p.1 + p.2 = 2023) ∧
    (p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - c.val|)) :
  c.val ≥ 2022 ∧ ∃ a b : ℕ+, a ≤ b ∧ b < 2022 ∧
    ∃! p : ℝ × ℝ, (2 * p.1 + p.2 = 2023) ∧
      (p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - 2022|) := by
  sorry

end NUMINAMATH_CALUDE_min_c_value_l1883_188314


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l1883_188342

/-- The percentage of water in fresh grapes -/
def water_percentage_fresh : ℝ := 90

/-- The percentage of water in dried grapes -/
def water_percentage_dried : ℝ := 20

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 5

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 0.625

theorem water_percentage_in_fresh_grapes :
  (100 - water_percentage_fresh) / 100 * fresh_weight = 
  (100 - water_percentage_dried) / 100 * dried_weight := by sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l1883_188342


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l1883_188332

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- Statement: f is an even function and increasing on (0, +∞)
theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l1883_188332


namespace NUMINAMATH_CALUDE_f_positive_iff_f_plus_3abs_min_f_plus_3abs_min_value_l1883_188363

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for part (1)
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x > 1 ∨ x < -5 := by sorry

-- Theorem for part (2)
theorem f_plus_3abs_min (x : ℝ) : f x + 3 * |x - 4| ≥ 9 := by sorry

-- Theorem for the minimum value
theorem f_plus_3abs_min_value : ∃ x : ℝ, f x + 3 * |x - 4| = 9 := by sorry

end NUMINAMATH_CALUDE_f_positive_iff_f_plus_3abs_min_f_plus_3abs_min_value_l1883_188363


namespace NUMINAMATH_CALUDE_circle_radii_theorem_l1883_188398

/-- The configuration of circles as described in the problem -/
structure CircleConfiguration where
  r : ℝ  -- radius of white circles
  red_radius : ℝ  -- radius of Adam's red circle
  green_radius : ℝ  -- radius of Eva's green circle

/-- The theorem stating the radii of the red and green circles -/
theorem circle_radii_theorem (config : CircleConfiguration) :
  config.red_radius = (Real.sqrt 2 - 1) * config.r ∧
  config.green_radius = (2 * Real.sqrt 3 - 3) / 3 * config.r :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_theorem_l1883_188398


namespace NUMINAMATH_CALUDE_no_solution_condition_l1883_188378

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (m * x - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_condition_l1883_188378


namespace NUMINAMATH_CALUDE_cos_sin_thirty_squared_difference_l1883_188304

theorem cos_sin_thirty_squared_difference :
  let cos_thirty : ℝ := Real.sqrt 3 / 2
  let sin_thirty : ℝ := 1 / 2
  cos_thirty ^ 2 - sin_thirty ^ 2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_thirty_squared_difference_l1883_188304


namespace NUMINAMATH_CALUDE_emma_hit_eleven_l1883_188386

-- Define the set of players
inductive Player : Type
| Alice : Player
| Ben : Player
| Cindy : Player
| Dave : Player
| Emma : Player
| Felix : Player

-- Define the score function
def score : Player → Nat
| Player.Alice => 21
| Player.Ben => 10
| Player.Cindy => 18
| Player.Dave => 15
| Player.Emma => 30
| Player.Felix => 22

-- Define the set of possible target values
def target_values : Finset Nat := Finset.range 12 \ {0}

-- Define a function to check if a player's score can be made up of three distinct values from the target
def valid_score (p : Player) : Prop :=
  ∃ (a b c : Nat), a ∈ target_values ∧ b ∈ target_values ∧ c ∈ target_values ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = score p

-- Theorem: Emma is the only player who could have hit the region worth 11 points
theorem emma_hit_eleven :
  ∀ (p : Player), p ≠ Player.Emma → 
    (valid_score p → ¬∃ (a b : Nat), a ∈ target_values ∧ b ∈ target_values ∧ a ≠ b ∧ a + b + 11 = score p) ∧
    (valid_score Player.Emma → ∃ (a b : Nat), a ∈ target_values ∧ b ∈ target_values ∧ a ≠ b ∧ a + b + 11 = score Player.Emma) :=
by sorry

end NUMINAMATH_CALUDE_emma_hit_eleven_l1883_188386


namespace NUMINAMATH_CALUDE_complex_sum_equivalence_l1883_188355

theorem complex_sum_equivalence :
  let z₁ := 12 * Complex.exp (Complex.I * (3 * Real.pi / 13))
  let z₂ := 12 * Complex.exp (Complex.I * (7 * Real.pi / 26))
  let r := 12 * Real.sqrt (2 + Real.sqrt 2)
  let θ := 3.25 * Real.pi / 13
  z₁ + z₂ = r * Complex.exp (Complex.I * θ) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equivalence_l1883_188355


namespace NUMINAMATH_CALUDE_ratio_equality_l1883_188319

theorem ratio_equality (a b c : ℝ) 
  (h1 : a / b = 4 / 3) 
  (h2 : a + c / b - c = 5 / 2) : 
  (3 * a + 2 * b) / (3 * a - 2 * b) = 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1883_188319


namespace NUMINAMATH_CALUDE_geometric_sequences_properties_l1883_188354

/-- A sequence is geometric if there exists a constant r such that the ratio of any two consecutive terms is r. -/
def IsGeometric (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_sequences_properties
  (a b : ℕ → ℝ) (ha : IsGeometric a) (hb : IsGeometric b) :
  (IsGeometric (fun n ↦ a n * b n)) ∧
  (∃ a' b' : ℕ → ℝ, IsGeometric a' ∧ IsGeometric b' ∧ ¬IsGeometric (fun n ↦ a' n + b' n)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequences_properties_l1883_188354


namespace NUMINAMATH_CALUDE_special_rectangle_area_l1883_188372

/-- Rectangle with a special circle configuration -/
structure SpecialRectangle where
  -- The radius of the inscribed circle
  r : ℝ
  -- The width of the rectangle (length of side AB)
  w : ℝ
  -- The height of the rectangle (length of side AD)
  h : ℝ
  -- The circle is tangent to sides AD and BC
  tangent_sides : h = 2 * r
  -- The circle is tangent internally to the semicircle with diameter AB
  tangent_semicircle : w = 6 * r
  -- The circle passes through the midpoint of AB
  passes_midpoint : w / 2 = 3 * r

/-- The area of the special rectangle is 12r^2 -/
theorem special_rectangle_area (rect : SpecialRectangle) :
  rect.w * rect.h = 12 * rect.r^2 := by
  sorry


end NUMINAMATH_CALUDE_special_rectangle_area_l1883_188372


namespace NUMINAMATH_CALUDE_total_cost_star_wars_toys_l1883_188360

/-- The total cost of Star Wars toys, including a lightsaber, given the cost of other toys -/
theorem total_cost_star_wars_toys (other_toys_cost : ℕ) : 
  other_toys_cost = 1000 → 
  (2 * other_toys_cost + other_toys_cost) = 3 * other_toys_cost := by
  sorry

#check total_cost_star_wars_toys

end NUMINAMATH_CALUDE_total_cost_star_wars_toys_l1883_188360


namespace NUMINAMATH_CALUDE_parallelogram_distance_l1883_188383

/-- Given a parallelogram with the following properties:
    - One side has length 20 feet
    - The perpendicular distance between that side and its opposite side is 60 feet
    - The other two parallel sides are each 50 feet long
    Prove that the perpendicular distance between the 50-foot sides is 24 feet. -/
theorem parallelogram_distance (base : ℝ) (height : ℝ) (side : ℝ) (h1 : base = 20) 
    (h2 : height = 60) (h3 : side = 50) : 
  (base * height) / side = 24 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_distance_l1883_188383


namespace NUMINAMATH_CALUDE_prime_power_equality_l1883_188357

theorem prime_power_equality (p : ℕ) (k : ℕ) (hp : Prime p) (hk : k > 1) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (m, n) ≠ (1, 1) ∧ 
    (m^p + n^p) / 2 = ((m + n) / 2)^k) ↔ k = p :=
by sorry

end NUMINAMATH_CALUDE_prime_power_equality_l1883_188357


namespace NUMINAMATH_CALUDE_triangle_isosceles_l1883_188330

/-- A triangle with sides a, b, and c satisfying the given equation is isosceles with c as the base -/
theorem triangle_isosceles (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : 1/a - 1/b + 1/c = 1/(a-b+c)) : a = c :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l1883_188330


namespace NUMINAMATH_CALUDE_place_digit_two_equals_formula_l1883_188369

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_bound : hundreds < 10
  tens_bound : tens < 10
  units_bound : units < 10

/-- Converts a ThreeDigitNumber to its integer value -/
def ThreeDigitNumber.toInt (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Places the digit 2 before a three-digit number -/
def placeDigitTwo (n : ThreeDigitNumber) : ℕ :=
  2000 + 10 * n.toInt

theorem place_digit_two_equals_formula (n : ThreeDigitNumber) :
  placeDigitTwo n = 1000 * (n.hundreds + 2) + 100 * n.tens + 10 * n.units := by
  sorry

end NUMINAMATH_CALUDE_place_digit_two_equals_formula_l1883_188369


namespace NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l1883_188308

theorem sweets_neither_red_nor_green 
  (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 285) 
  (h2 : red = 49) 
  (h3 : green = 59) : 
  total - (red + green) = 177 := by
  sorry

end NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l1883_188308


namespace NUMINAMATH_CALUDE_chess_team_arrangements_l1883_188388

/-- Represents the number of boys in the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the chess team -/
def num_girls : ℕ := 4

/-- Represents the total number of team members -/
def total_members : ℕ := num_boys + num_girls

/-- Represents the requirement that two specific girls must sit together -/
def specific_girls_together : Prop := True

/-- Represents the requirement that a boy must sit at each end -/
def boy_at_each_end : Prop := True

/-- The number of possible arrangements of the chess team -/
def num_arrangements : ℕ := 72

/-- Theorem stating that the number of arrangements is 72 -/
theorem chess_team_arrangements :
  num_boys = 3 →
  num_girls = 4 →
  specific_girls_together →
  boy_at_each_end →
  num_arrangements = 72 := by sorry

end NUMINAMATH_CALUDE_chess_team_arrangements_l1883_188388


namespace NUMINAMATH_CALUDE_minimum_heat_for_piston_ejection_l1883_188397

/-- The minimum amount of heat required to shoot a piston out of a cylinder -/
theorem minimum_heat_for_piston_ejection
  (l₁ : Real) (l₂ : Real) (M : Real) (S : Real) (v : Real) (p₀ : Real) (g : Real)
  (h₁ : l₁ = 0.1) -- 10 cm in meters
  (h₂ : l₂ = 0.15) -- 15 cm in meters
  (h₃ : M = 10) -- 10 kg
  (h₄ : S = 0.001) -- 10 cm² in m²
  (h₅ : v = 1) -- 1 mole
  (h₆ : p₀ = 10^5) -- 10⁵ Pa
  (h₇ : g = 10) -- 10 m/s²
  : ∃ Q : Real, Q = 127.5 ∧ Q ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_minimum_heat_for_piston_ejection_l1883_188397


namespace NUMINAMATH_CALUDE_solve_for_a_l1883_188331

theorem solve_for_a : ∀ a : ℝ, (2 * 2 + a - 5 = 0) → a = 1 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l1883_188331


namespace NUMINAMATH_CALUDE_library_visitors_theorem_l1883_188362

/-- Represents the average number of visitors on a given day type -/
structure VisitorAverage where
  sunday : ℕ
  other : ℕ

/-- Represents a month with visitor data -/
structure Month where
  days : ℕ
  startsWithSunday : Bool
  avgVisitorsPerDay : ℕ
  visitorAvg : VisitorAverage

/-- Calculates the number of Sundays in a month -/
def numSundays (m : Month) : ℕ :=
  if m.startsWithSunday then
    (m.days + 6) / 7
  else
    m.days / 7

/-- Theorem: Given the conditions, prove that the average number of visitors on non-Sunday days is 80 -/
theorem library_visitors_theorem (m : Month) 
    (h1 : m.days = 30)
    (h2 : m.startsWithSunday = true)
    (h3 : m.visitorAvg.sunday = 140)
    (h4 : m.avgVisitorsPerDay = 90) :
    m.visitorAvg.other = 80 := by
  sorry


end NUMINAMATH_CALUDE_library_visitors_theorem_l1883_188362


namespace NUMINAMATH_CALUDE_number_thought_of_l1883_188302

theorem number_thought_of (x : ℝ) : (x / 4) + 9 = 15 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l1883_188302


namespace NUMINAMATH_CALUDE_marble_problem_solution_l1883_188341

/-- Represents a jar of marbles -/
structure Jar where
  blue : ℕ
  green : ℕ

/-- The problem setup -/
def marble_problem : Prop :=
  ∃ (jar1 jar2 : Jar),
    -- Both jars have the same total number of marbles
    jar1.blue + jar1.green = jar2.blue + jar2.green
    -- Ratio of blue to green in Jar 1 is 9:1
    ∧ 9 * jar1.green = jar1.blue
    -- Ratio of blue to green in Jar 2 is 7:2
    ∧ 7 * jar2.green = 2 * jar2.blue
    -- Total number of green marbles is 108
    ∧ jar1.green + jar2.green = 108
    -- The difference in blue marbles between Jar 1 and Jar 2 is 38
    ∧ jar1.blue - jar2.blue = 38

/-- The theorem to prove -/
theorem marble_problem_solution : marble_problem := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_solution_l1883_188341


namespace NUMINAMATH_CALUDE_original_earnings_l1883_188384

theorem original_earnings (new_earnings : ℝ) (percentage_increase : ℝ) 
  (h1 : new_earnings = 84)
  (h2 : percentage_increase = 40) :
  let original_earnings := new_earnings / (1 + percentage_increase / 100)
  original_earnings = 60 := by
sorry

end NUMINAMATH_CALUDE_original_earnings_l1883_188384


namespace NUMINAMATH_CALUDE_inequality_solution_l1883_188393

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  {x | (a + 2) * x - 4 ≤ 2 * (x - 1)}

theorem inequality_solution (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 2 → solution_set a = {x | 1 < x ∧ x ≤ 2/a}) ∧
  (a = 2 → solution_set a = ∅) ∧
  (a > 2 → solution_set a = {x | 2/a ≤ x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1883_188393


namespace NUMINAMATH_CALUDE_james_twitch_income_l1883_188313

/-- Calculates the monthly income from Twitch subscriptions given the subscriber counts, costs, and revenue percentages for each tier. -/
def monthly_twitch_income (tier1_subs tier2_subs tier3_subs : ℕ) 
                          (tier1_cost tier2_cost tier3_cost : ℚ) 
                          (tier1_percent tier2_percent tier3_percent : ℚ) : ℚ :=
  tier1_subs * tier1_cost * tier1_percent +
  tier2_subs * tier2_cost * tier2_percent +
  tier3_subs * tier3_cost * tier3_percent

/-- Proves that James' monthly income from Twitch subscriptions is $2065.41 given the specified conditions. -/
theorem james_twitch_income : 
  monthly_twitch_income 130 75 45 (499/100) (999/100) (2499/100) (70/100) (80/100) (90/100) = 206541/100 := by
  sorry

end NUMINAMATH_CALUDE_james_twitch_income_l1883_188313


namespace NUMINAMATH_CALUDE_total_tickets_sold_l1883_188377

/-- The total number of tickets sold given the ticket prices, total revenue, and number of adult tickets. -/
theorem total_tickets_sold
  (child_price : ℕ)
  (adult_price : ℕ)
  (total_revenue : ℕ)
  (adult_tickets : ℕ)
  (h1 : child_price = 6)
  (h2 : adult_price = 9)
  (h3 : total_revenue = 1875)
  (h4 : adult_tickets = 175) :
  child_price * (total_revenue - adult_price * adult_tickets) / child_price + adult_tickets = 225 :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l1883_188377


namespace NUMINAMATH_CALUDE_expression_factorization_l1883_188350

theorem expression_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (-(x*y + x*z + y*z)) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1883_188350


namespace NUMINAMATH_CALUDE_cake_division_l1883_188312

theorem cake_division (total_cake : ℚ) (num_people : ℕ) :
  total_cake = 7/8 ∧ num_people = 4 →
  total_cake / num_people = 7/32 := by
sorry

end NUMINAMATH_CALUDE_cake_division_l1883_188312


namespace NUMINAMATH_CALUDE_find_b_l1883_188339

/-- The circle's equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 2*y - 2 = 0

/-- The line's equation -/
def line_eq (x y b : ℝ) : Prop := y = x + b

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (4, -1)

/-- The line bisects the circle's circumference -/
axiom bisects : ∃ b : ℝ, ∀ x y : ℝ, circle_eq x y → line_eq x y b

/-- The theorem to prove -/
theorem find_b : ∃ b : ℝ, b = -5 ∧ 
  (∀ x y : ℝ, circle_eq x y → line_eq x y b) ∧
  line_eq (circle_center.1) (circle_center.2) b :=
sorry

end NUMINAMATH_CALUDE_find_b_l1883_188339


namespace NUMINAMATH_CALUDE_subtract_squares_l1883_188368

theorem subtract_squares (a : ℝ) : 3 * a^2 - a^2 = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_squares_l1883_188368


namespace NUMINAMATH_CALUDE_project_speedup_l1883_188389

/-- Calculates the number of days saved when additional workers join a project -/
def days_saved (original_workers : ℕ) (original_days : ℕ) (additional_workers : ℕ) : ℕ :=
  original_days - (original_workers * original_days) / (original_workers + additional_workers)

/-- Theorem stating that 10 additional workers save 6 days on a 12-day project with 10 original workers -/
theorem project_speedup :
  days_saved 10 12 10 = 6 := by sorry

end NUMINAMATH_CALUDE_project_speedup_l1883_188389


namespace NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_l1883_188338

theorem smallest_positive_integer_3003m_55555n :
  ∃ (k : ℕ), k > 0 ∧ (∀ (j : ℕ), j > 0 → (∃ (m n : ℤ), j = 3003 * m + 55555 * n) → k ≤ j) ∧
  (∃ (m n : ℤ), k = 3003 * m + 55555 * n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_l1883_188338


namespace NUMINAMATH_CALUDE_complement_union_eq_inter_complements_l1883_188358

variable {Ω : Type*} [MeasurableSpace Ω]
variable (A B : Set Ω)

theorem complement_union_eq_inter_complements :
  (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ := by sorry

end NUMINAMATH_CALUDE_complement_union_eq_inter_complements_l1883_188358


namespace NUMINAMATH_CALUDE_gasohol_mixture_proof_l1883_188366

/-- Proves that the initial percentage of gasoline in the gasohol mixture is 95% --/
theorem gasohol_mixture_proof (initial_volume : ℝ) (initial_ethanol_percent : ℝ) 
  (desired_ethanol_percent : ℝ) (added_ethanol : ℝ) :
  initial_volume = 45 →
  initial_ethanol_percent = 5 →
  desired_ethanol_percent = 10 →
  added_ethanol = 2.5 →
  (100 - initial_ethanol_percent) = 95 :=
by
  sorry

#check gasohol_mixture_proof

end NUMINAMATH_CALUDE_gasohol_mixture_proof_l1883_188366


namespace NUMINAMATH_CALUDE_diagonal_less_than_half_perimeter_l1883_188328

-- Define a quadrilateral with sides a, b, c, d and diagonal x
structure Quadrilateral :=
  (a b c d x : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (positive_diagonal : 0 < x)

-- Theorem: The diagonal is less than half the perimeter
theorem diagonal_less_than_half_perimeter (q : Quadrilateral) :
  q.x < (q.a + q.b + q.c + q.d) / 2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_less_than_half_perimeter_l1883_188328


namespace NUMINAMATH_CALUDE_total_strawberries_l1883_188353

/-- The number of strawberries picked by Jonathan and Matthew together -/
def jonathan_matthew_total : ℕ := 350

/-- The number of strawberries picked by Matthew and Zac together -/
def matthew_zac_total : ℕ := 250

/-- The number of strawberries picked by Zac alone -/
def zac_alone : ℕ := 200

/-- Theorem stating that the total number of strawberries picked is 550 -/
theorem total_strawberries : 
  ∃ (j m z : ℕ), 
    j + m = jonathan_matthew_total ∧ 
    m + z = matthew_zac_total ∧ 
    z = zac_alone ∧ 
    j + m + z = 550 := by
  sorry

end NUMINAMATH_CALUDE_total_strawberries_l1883_188353


namespace NUMINAMATH_CALUDE_sophie_coin_distribution_l1883_188379

/-- The minimum number of additional coins needed for Sophie's distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins Sophie needs. -/
theorem sophie_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 10) (h2 : initial_coins = 40) : 
  min_additional_coins num_friends initial_coins = 15 := by
  sorry

#eval min_additional_coins 10 40

end NUMINAMATH_CALUDE_sophie_coin_distribution_l1883_188379


namespace NUMINAMATH_CALUDE_common_area_is_32_l1883_188381

/-- Represents a circle with an inscribed square and an intersecting rectangle -/
structure GeometricSetup where
  -- Radius of the circle
  radius : ℝ
  -- Side length of the inscribed square
  square_side : ℝ
  -- Width of the intersecting rectangle
  rect_width : ℝ
  -- Height of the intersecting rectangle
  rect_height : ℝ
  -- The square is inscribed in the circle
  h_inscribed : radius = square_side * Real.sqrt 2 / 2
  -- The rectangle intersects the circle
  h_intersects : rect_width > 2 * radius ∧ rect_height ≤ 2 * radius

/-- The area common to both the rectangle and the circle -/
def commonArea (setup : GeometricSetup) : ℝ :=
  setup.rect_height * setup.rect_width

/-- The theorem stating the common area is 32 square units -/
theorem common_area_is_32 (setup : GeometricSetup) 
    (h_square : setup.square_side = 8)
    (h_rect : setup.rect_width = 10 ∧ setup.rect_height = 4) :
    commonArea setup = 32 := by
  sorry


end NUMINAMATH_CALUDE_common_area_is_32_l1883_188381


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1883_188329

theorem quadratic_roots_relation (A B C : ℝ) (r s p q : ℝ) : 
  (A * r^2 + B * r + C = 0) →
  (A * s^2 + B * s + C = 0) →
  (r^2)^2 + p * r^2 + q = 0 →
  (s^2)^2 + p * s^2 + q = 0 →
  p = (2 * A * C - B^2) / A^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1883_188329


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l1883_188375

theorem sin_thirteen_pi_sixths : Real.sin (13 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_sixths_l1883_188375


namespace NUMINAMATH_CALUDE_parabola_symmetry_axis_l1883_188391

/-- A parabola defined by y = 2x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The theorem stating that for a parabola y = 2x^2 + bx + c passing through
    points A(2,5) and B(4,5), the axis of symmetry is x = 3 -/
theorem parabola_symmetry_axis (p : Parabola) (A B : Point)
    (hA : A.y = 2 * A.x^2 + p.b * A.x + p.c)
    (hB : B.y = 2 * B.x^2 + p.b * B.x + p.c)
    (hAx : A.x = 2) (hAy : A.y = 5)
    (hBx : B.x = 4) (hBy : B.y = 5) :
    (A.x + B.x) / 2 = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_axis_l1883_188391


namespace NUMINAMATH_CALUDE_population_increase_rate_example_l1883_188365

/-- Given an initial population and a final population after one year,
    calculate the population increase rate as a percentage. -/
def population_increase_rate (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Theorem stating that for an initial population of 220 and
    a final population of 242, the increase rate is 10%. -/
theorem population_increase_rate_example :
  population_increase_rate 220 242 = 10 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_example_l1883_188365


namespace NUMINAMATH_CALUDE_amit_work_days_l1883_188356

theorem amit_work_days (ananthu_days : ℝ) (amit_worked : ℝ) (total_days : ℝ) :
  ananthu_days = 90 ∧ amit_worked = 3 ∧ total_days = 75 →
  ∃ x : ℝ, 
    x > 0 ∧
    (3 / x) + ((total_days - amit_worked) / ananthu_days) = 1 ∧
    x = 15 := by
  sorry

end NUMINAMATH_CALUDE_amit_work_days_l1883_188356


namespace NUMINAMATH_CALUDE_water_bottles_left_l1883_188327

theorem water_bottles_left (initial_bottles : ℕ) (bottles_drunk : ℕ) : 
  initial_bottles = 301 → bottles_drunk = 144 → initial_bottles - bottles_drunk = 157 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_left_l1883_188327


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_neg_five_thirds_l1883_188311

theorem trigonometric_expression_equals_neg_five_thirds :
  (Real.tan (30 * π / 180))^2 - (Real.cos (30 * π / 180))^2
  / ((Real.tan (30 * π / 180))^2 * (Real.cos (30 * π / 180))^2) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_neg_five_thirds_l1883_188311


namespace NUMINAMATH_CALUDE_deepak_age_l1883_188309

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 18 →
  deepak_age = 9 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1883_188309
