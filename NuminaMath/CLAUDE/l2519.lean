import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2519_251921

-- Define the radius of the cylinder
def cylinder_radius : ℝ := 2

-- Define the relationship between major and minor axes
def major_axis_ratio : ℝ := 1.75

-- Theorem statement
theorem ellipse_major_axis_length :
  let minor_axis := 2 * cylinder_radius
  let major_axis := major_axis_ratio * minor_axis
  major_axis = 7 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2519_251921


namespace NUMINAMATH_CALUDE_wire_cutting_l2519_251904

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 80 →
  ratio = 3 / 5 →
  shorter_piece + ratio * shorter_piece = total_length →
  shorter_piece = 50 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l2519_251904


namespace NUMINAMATH_CALUDE_island_ratio_l2519_251979

theorem island_ratio (centipedes humans sheep : ℕ) : 
  centipedes = 2 * humans →
  centipedes = 100 →
  sheep + humans = 75 →
  sheep.gcd humans = 25 →
  (sheep / 25 : ℚ) / (humans / 25 : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_island_ratio_l2519_251979


namespace NUMINAMATH_CALUDE_cindy_lisa_marble_difference_l2519_251968

theorem cindy_lisa_marble_difference :
  ∀ (lisa_initial : ℕ),
  let cindy_initial : ℕ := 20
  let cindy_after : ℕ := cindy_initial - 12
  let lisa_after : ℕ := lisa_initial + 12
  lisa_after = cindy_after + 19 →
  cindy_initial - lisa_initial = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cindy_lisa_marble_difference_l2519_251968


namespace NUMINAMATH_CALUDE_ordered_triples_satisfying_equation_l2519_251987

theorem ordered_triples_satisfying_equation :
  ∀ m n p : ℕ,
    m > 0 ∧ n > 0 ∧ Nat.Prime p ∧ p^n + 144 = m^2 →
    ((m = 13 ∧ n = 2 ∧ p = 5) ∨
     (m = 20 ∧ n = 8 ∧ p = 2) ∨
     (m = 15 ∧ n = 4 ∧ p = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ordered_triples_satisfying_equation_l2519_251987


namespace NUMINAMATH_CALUDE_darry_total_steps_l2519_251925

/-- The number of steps Darry climbed in total -/
def total_steps (full_ladder_steps : ℕ) (full_ladder_climbs : ℕ) 
                (small_ladder_steps : ℕ) (small_ladder_climbs : ℕ) : ℕ :=
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs

/-- Proof that Darry climbed 152 steps in total -/
theorem darry_total_steps : 
  total_steps 11 10 6 7 = 152 := by
  sorry

end NUMINAMATH_CALUDE_darry_total_steps_l2519_251925


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2519_251964

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_mono : is_monotonically_increasing a)
  (h_prod : a 1 * a 9 = 64)
  (h_sum : a 3 + a 7 = 20) :
  a 11 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2519_251964


namespace NUMINAMATH_CALUDE_correct_equation_for_john_scenario_l2519_251983

/-- Represents a driving scenario with a stop -/
structure DrivingScenario where
  speed_before_stop : ℝ
  stop_duration : ℝ
  speed_after_stop : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The given driving scenario -/
def john_scenario : DrivingScenario :=
  { speed_before_stop := 60
  , stop_duration := 0.5
  , speed_after_stop := 80
  , total_distance := 200
  , total_time := 4 }

/-- The equation representing the driving scenario -/
def scenario_equation (s : DrivingScenario) (t : ℝ) : Prop :=
  s.speed_before_stop * t + s.speed_after_stop * (s.total_time - s.stop_duration - t) = s.total_distance

/-- Theorem stating that the equation correctly represents John's driving scenario -/
theorem correct_equation_for_john_scenario :
  ∀ t, scenario_equation john_scenario t ↔ 60 * t + 80 * (7/2 - t) = 200 :=
sorry

end NUMINAMATH_CALUDE_correct_equation_for_john_scenario_l2519_251983


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2519_251945

theorem right_triangle_sides : ∃ (a b c : ℝ), 
  a = 8 ∧ b = 15 ∧ c = 17 ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l2519_251945


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l2519_251962

theorem ellipse_hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e_ellipse := Real.sqrt 3 / 2
  let c := e_ellipse * a
  let e_hyperbola := Real.sqrt ((a^2 + b^2) / a^2)
  (a^2 = b^2 + c^2) → e_hyperbola = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l2519_251962


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2519_251929

theorem fraction_multiplication : 
  (7 / 8 : ℚ) * (1 / 3 : ℚ) * (3 / 7 : ℚ) = 0.12499999999999997 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2519_251929


namespace NUMINAMATH_CALUDE_vessel_base_length_l2519_251952

/-- The length of the base of a vessel given specific conditions -/
theorem vessel_base_length : ∀ (breadth rise cube_edge : ℝ),
  breadth = 30 →
  rise = 15 →
  cube_edge = 30 →
  (cube_edge ^ 3) = breadth * rise * 60 :=
by
  sorry

end NUMINAMATH_CALUDE_vessel_base_length_l2519_251952


namespace NUMINAMATH_CALUDE_solve_system_l2519_251994

theorem solve_system (c d : ℤ) 
  (eq1 : 5 + c = 7 - d) 
  (eq2 : 6 + d = 10 + c) : 
  5 - c = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l2519_251994


namespace NUMINAMATH_CALUDE_log_equation_solution_l2519_251938

theorem log_equation_solution :
  ∃ y : ℝ, y > 0 ∧ 2 * Real.log y - 4 * Real.log 2 = 2 → y = 160 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2519_251938


namespace NUMINAMATH_CALUDE_tree_spacing_l2519_251901

theorem tree_spacing (yard_length : ℕ) (num_trees : ℕ) (distance : ℕ) : 
  yard_length = 273 ∧ num_trees = 14 → distance * (num_trees - 1) = yard_length → distance = 21 := by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l2519_251901


namespace NUMINAMATH_CALUDE_atheris_population_2080_l2519_251905

def population_growth (initial_population : ℕ) (years : ℕ) : ℕ :=
  initial_population * (4 ^ (years / 30))

theorem atheris_population_2080 :
  population_growth 250 80 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_atheris_population_2080_l2519_251905


namespace NUMINAMATH_CALUDE_factorial_divides_power_difference_l2519_251934

theorem factorial_divides_power_difference (n : ℕ) : 
  (n.factorial : ℤ) ∣ (2^(2*n.factorial) - 2^n.factorial) :=
sorry

end NUMINAMATH_CALUDE_factorial_divides_power_difference_l2519_251934


namespace NUMINAMATH_CALUDE_circle_intersection_l2519_251930

/-- The equation of the circle C -/
def C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

/-- The equation of the line l -/
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

/-- Theorem stating when C represents a circle and the value of m when C intersects l -/
theorem circle_intersection :
  (∃ (m : ℝ), ∀ (x y : ℝ), C x y m → m < 5) ∧
  (∃ (m : ℝ), ∀ (x y : ℝ), C x y m → l x y → 
    ∃ (M N : ℝ × ℝ), C M.1 M.2 m ∧ C N.1 N.2 m ∧ l M.1 M.2 ∧ l N.1 N.2 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = (4 / Real.sqrt 5)^2 → m = 4) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_l2519_251930


namespace NUMINAMATH_CALUDE_brand_z_percentage_approx_l2519_251917

/-- Represents the capacity of the fuel tank -/
def tank_capacity : ℚ := 12

/-- Represents the amount of brand Z gasoline after the final filling -/
def final_brand_z : ℚ := 10

/-- Represents the total amount of gasoline after the final filling -/
def final_total : ℚ := 12

/-- Calculates the percentage of a part relative to the whole -/
def percentage (part whole : ℚ) : ℚ := (part / whole) * 100

/-- Theorem stating that the percentage of brand Z gasoline is approximately 83.33% -/
theorem brand_z_percentage_approx : 
  abs (percentage final_brand_z final_total - 83.33) < 0.01 := by
  sorry

#eval percentage final_brand_z final_total

end NUMINAMATH_CALUDE_brand_z_percentage_approx_l2519_251917


namespace NUMINAMATH_CALUDE_classroom_a_fundraising_l2519_251950

/-- The fundraising goal for each classroom -/
def goal : ℕ := 200

/-- The amount raised from two families at $20 each -/
def amount_20 : ℕ := 2 * 20

/-- The amount raised from eight families at $10 each -/
def amount_10 : ℕ := 8 * 10

/-- The amount raised from ten families at $5 each -/
def amount_5 : ℕ := 10 * 5

/-- The total amount raised by Classroom A -/
def total_raised : ℕ := amount_20 + amount_10 + amount_5

/-- The additional amount needed to reach the goal -/
def additional_amount_needed : ℕ := goal - total_raised

theorem classroom_a_fundraising :
  additional_amount_needed = 30 :=
by sorry

end NUMINAMATH_CALUDE_classroom_a_fundraising_l2519_251950


namespace NUMINAMATH_CALUDE_draw_probability_standard_deck_l2519_251948

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (hearts : Nat)
  (clubs : Nat)
  (spades : Nat)

/-- A standard 52-card deck -/
def standardDeck : Deck :=
  { cards := 52,
    hearts := 13,
    clubs := 13,
    spades := 13 }

/-- The probability of drawing a heart, then a club, then a spade from a standard deck -/
def drawProbability (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.cards *
  (d.clubs : ℚ) / (d.cards - 1) *
  (d.spades : ℚ) / (d.cards - 2)

theorem draw_probability_standard_deck :
  drawProbability standardDeck = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_draw_probability_standard_deck_l2519_251948


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l2519_251941

variable (x : ℝ) (y : ℝ → ℝ) (C : ℝ)

noncomputable def solution (x : ℝ) (y : ℝ → ℝ) (C : ℝ) : Prop :=
  x * y x - 1 / (x * y x) - 2 * Real.log (abs (y x)) = C

def differential_equation (x : ℝ) (y : ℝ → ℝ) : Prop :=
  (1 + x^2 * (y x)^2) * y x + (x * y x - 1)^2 * x * (deriv y x) = 0

theorem solution_satisfies_equation :
  solution x y C → differential_equation x y :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l2519_251941


namespace NUMINAMATH_CALUDE_book_selection_problem_l2519_251922

theorem book_selection_problem (n m k : ℕ) (h1 : n = 8) (h2 : m = 5) (h3 : k = 4) :
  (Nat.choose (n - 1) k) = (Nat.choose (n - 1) (m - 1)) :=
sorry

end NUMINAMATH_CALUDE_book_selection_problem_l2519_251922


namespace NUMINAMATH_CALUDE_website_visitors_ratio_l2519_251966

/-- Proves that the ratio of visitors on the last day to the total visitors on the first 6 days is 2:1 -/
theorem website_visitors_ratio (daily_visitors : ℕ) (constant_days : ℕ) (revenue_per_visit : ℚ) (total_revenue : ℚ) 
  (h1 : daily_visitors = 100)
  (h2 : constant_days = 6)
  (h3 : revenue_per_visit = 1 / 100)
  (h4 : total_revenue = 18) :
  (total_revenue / revenue_per_visit - daily_visitors * constant_days) / (daily_visitors * constant_days) = 2 := by
sorry

end NUMINAMATH_CALUDE_website_visitors_ratio_l2519_251966


namespace NUMINAMATH_CALUDE_uncle_bruce_chocolate_cookies_l2519_251955

theorem uncle_bruce_chocolate_cookies (total_dough : ℝ) (chocolate_percentage : ℝ) (leftover_chocolate : ℝ) :
  total_dough = 36 ∧ 
  chocolate_percentage = 0.20 ∧ 
  leftover_chocolate = 4 →
  ∃ initial_chocolate : ℝ,
    initial_chocolate = 13 ∧
    chocolate_percentage * (total_dough + initial_chocolate - leftover_chocolate) = initial_chocolate - leftover_chocolate :=
by sorry

end NUMINAMATH_CALUDE_uncle_bruce_chocolate_cookies_l2519_251955


namespace NUMINAMATH_CALUDE_find_genuine_coins_l2519_251985

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a coin -/
inductive Coin
  | Genuine : Coin
  | Counterfeit : Coin

/-- Represents a set of coins -/
def CoinSet := Fin 7 → Coin

/-- A weighing function that compares two sets of coins -/
def weigh (coins : CoinSet) (left right : List (Fin 7)) : WeighResult :=
  sorry

/-- Checks if a given set of coins contains exactly 3 genuine coins -/
def isValidResult (coins : CoinSet) (result : List (Fin 7)) : Prop :=
  result.length = 3 ∧ ∀ i ∈ result.toFinset, coins i = Coin.Genuine

/-- The main theorem stating that it's possible to find 3 genuine coins in two weighings -/
theorem find_genuine_coins 
  (coins : CoinSet) 
  (h1 : ∃ (i j : Fin 7), i ≠ j ∧ coins i = Coin.Counterfeit ∧ coins j = Coin.Counterfeit)
  (h2 : ∀ (i : Fin 7), coins i ≠ Coin.Counterfeit → coins i = Coin.Genuine)
  : ∃ (w1 w2 : List (Fin 7) × List (Fin 7)) (result : List (Fin 7)),
    isValidResult coins result ∧ 
    (∀ (c1 c2 : CoinSet), 
      (∀ (i : Fin 7), coins i = Coin.Genuine ↔ c1 i = Coin.Genuine) →
      (∀ (i : Fin 7), coins i = Coin.Genuine ↔ c2 i = Coin.Genuine) →
      weigh c1 w1.1 w1.2 = weigh c2 w1.1 w1.2 →
      weigh c1 w2.1 w2.2 = weigh c2 w2.1 w2.2 →
      (∀ (i : Fin 7), i ∈ result → c1 i = Coin.Genuine)) :=
sorry

end NUMINAMATH_CALUDE_find_genuine_coins_l2519_251985


namespace NUMINAMATH_CALUDE_linear_function_k_range_l2519_251975

theorem linear_function_k_range (k b : ℝ) :
  k ≠ 0 →
  (2 * k + b = -3) →
  (0 < b ∧ b < 1) →
  (-2 < k ∧ k < -3/2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_k_range_l2519_251975


namespace NUMINAMATH_CALUDE_fraction_equality_l2519_251936

theorem fraction_equality (a b : ℝ) (h : a / 5 = b / 3) : (a - b) / (3 * a) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2519_251936


namespace NUMINAMATH_CALUDE_product_325_3_base7_l2519_251932

-- Define a function to convert from base 7 to base 10
def base7ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 7
def base10ToBase7 (n : ℕ) : ℕ := sorry

-- Define the multiplication operation in base 7
def multBase7 (a b : ℕ) : ℕ := 
  base10ToBase7 (base7ToBase10 a * base7ToBase10 b)

-- State the theorem
theorem product_325_3_base7 : 
  multBase7 325 3 = 3111 := by sorry

end NUMINAMATH_CALUDE_product_325_3_base7_l2519_251932


namespace NUMINAMATH_CALUDE_race_permutations_eq_24_l2519_251918

/-- The number of different possible orders for a race with 4 distinct participants and no ties -/
def race_permutations : ℕ := 24

/-- The number of participants in the race -/
def num_participants : ℕ := 4

/-- Theorem: The number of different possible orders for a race with 4 distinct participants and no ties is 24 -/
theorem race_permutations_eq_24 : race_permutations = Nat.factorial num_participants := by
  sorry

end NUMINAMATH_CALUDE_race_permutations_eq_24_l2519_251918


namespace NUMINAMATH_CALUDE_choose_four_from_fifteen_l2519_251977

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : n = 15 ∧ k = 4 → Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_fifteen_l2519_251977


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l2519_251989

theorem rectangle_width_decrease (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_L := 1.5 * L
  let new_W := W * (L / new_L)
  (W - new_W) / W = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l2519_251989


namespace NUMINAMATH_CALUDE_extended_triangle_theorem_l2519_251972

-- Define the triangle ABC
variable (A B C : Point) (ABC : Triangle A B C)

-- Define the condition BC = 2AC
variable (h1 : BC = 2 * AC)

-- Define point D such that AD = 1/3 * AB
variable (D : Point)
variable (h2 : AD = (1/3) * AB)

-- Theorem statement
theorem extended_triangle_theorem : CD = 2 * AD := by
  sorry

end NUMINAMATH_CALUDE_extended_triangle_theorem_l2519_251972


namespace NUMINAMATH_CALUDE_third_trapezoid_largest_area_l2519_251940

-- Define the lengths of the segments
def a : ℝ := 2.12
def b : ℝ := 2.71
def c : ℝ := 3.53

-- Define the area calculation function for a trapezoid
def trapezoidArea (top bottom height : ℝ) : ℝ := (top + bottom) * height

-- Define the three possible trapezoids
def trapezoid1 : ℝ := trapezoidArea a c b
def trapezoid2 : ℝ := trapezoidArea b c a
def trapezoid3 : ℝ := trapezoidArea a b c

-- Theorem statement
theorem third_trapezoid_largest_area :
  trapezoid3 > trapezoid1 ∧ trapezoid3 > trapezoid2 :=
by sorry

end NUMINAMATH_CALUDE_third_trapezoid_largest_area_l2519_251940


namespace NUMINAMATH_CALUDE_set_equality_implies_a_squared_minus_b_zero_l2519_251959

theorem set_equality_implies_a_squared_minus_b_zero (a b : ℝ) 
  (h : ({1, a + b, a} : Set ℝ) = {0, b / a, b}) : a^2 - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_squared_minus_b_zero_l2519_251959


namespace NUMINAMATH_CALUDE_system_solution_proof_l2519_251913

theorem system_solution_proof :
  ∃ (x y z : ℝ),
    (1/x + 1/(y+z) = 6/5) ∧
    (1/y + 1/(x+z) = 3/4) ∧
    (1/z + 1/(x+y) = 2/3) ∧
    (x = 2) ∧ (y = 3) ∧ (z = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_proof_l2519_251913


namespace NUMINAMATH_CALUDE_house_distance_ratio_l2519_251920

/-- Given three points on a road representing houses, proves the ratio of distances -/
theorem house_distance_ratio (K D M : ℝ) : 
  let KD := |K - D|
  let DM := |D - M|
  KD = 4 → KD + DM + DM + KD = 12 → KD / DM = 2 := by
  sorry

end NUMINAMATH_CALUDE_house_distance_ratio_l2519_251920


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l2519_251928

theorem circle_line_distance_range (a : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4}
  let line := {(x, y) : ℝ × ℝ | x + y = a}
  let distance_to_line (p : ℝ × ℝ) := |p.1 + p.2 - a| / Real.sqrt 2
  (∃ p1 p2 : ℝ × ℝ, p1 ∈ circle ∧ p2 ∈ circle ∧ p1 ≠ p2 ∧ 
    distance_to_line p1 = 1 ∧ distance_to_line p2 = 1) →
  a ∈ Set.Ioo (-3 * Real.sqrt 2) (3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l2519_251928


namespace NUMINAMATH_CALUDE_range_of_expression_l2519_251997

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = 5 * Real.arcsin x - 2 * Real.arccos y ∧ 
  -7/2 * Real.pi ≤ z ∧ z ≤ 3/2 * Real.pi ∧
  (∀ ε > 0, ∃ (x' y' : ℝ), x'^2 + y'^2 = 1 ∧
    (5 * Real.arcsin x' - 2 * Real.arccos y' < -7/2 * Real.pi + ε ∨
     5 * Real.arcsin x' - 2 * Real.arccos y' > 3/2 * Real.pi - ε)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l2519_251997


namespace NUMINAMATH_CALUDE_reciprocal_roots_l2519_251992

theorem reciprocal_roots (a b c : ℝ) (x y : ℝ) : 
  (a * x^2 + b * x + c = 0 ↔ c * (1/x)^2 + b * (1/x) + a = 0) ∧ 
  (c * y^2 + b * y + a = 0 ↔ a * (1/y)^2 + b * (1/y) + c = 0) := by
sorry

end NUMINAMATH_CALUDE_reciprocal_roots_l2519_251992


namespace NUMINAMATH_CALUDE_geometric_progression_special_ratio_l2519_251965

/-- A geometric progression with positive terms where each term is the average of the next two terms plus 2 has a common ratio of 1. -/
theorem geometric_progression_special_ratio :
  ∀ (a : ℝ) (r : ℝ),
  (a > 0) →  -- First term is positive
  (r > 0) →  -- Common ratio is positive
  (∀ n : ℕ, a * r^n = (a * r^(n+1) + a * r^(n+2)) / 2 + 2) →  -- Condition on terms
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_special_ratio_l2519_251965


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2519_251903

/-- The area of a rectangular field with length 1.2 meters and width three-fourths of the length is 1.08 square meters. -/
theorem rectangular_field_area : 
  let length : ℝ := 1.2
  let width : ℝ := (3/4) * length
  let area : ℝ := length * width
  area = 1.08 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2519_251903


namespace NUMINAMATH_CALUDE_revolver_game_theorem_l2519_251914

/-- The probability that player A fires the bullet in the revolver game -/
def revolver_game_prob : ℚ :=
  let p : ℚ := 1/6  -- probability of firing on a single shot
  6/11

/-- The revolver game theorem -/
theorem revolver_game_theorem :
  let p : ℚ := 1/6  -- probability of firing on a single shot
  let q : ℚ := 1 - p  -- probability of not firing on a single shot
  revolver_game_prob = p / (1 - q^2) :=
by sorry

#eval revolver_game_prob

end NUMINAMATH_CALUDE_revolver_game_theorem_l2519_251914


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l2519_251943

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents the resulting structure after modifications -/
structure ModifiedCube where
  original : Cube 9
  small_cubes : ℕ := 27
  removed_corners : ℕ := 8

/-- Calculates the surface area of the modified cube structure -/
def surface_area (mc : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that the surface area of the modified cube is 1056 -/
theorem modified_cube_surface_area :
  ∀ (mc : ModifiedCube), surface_area mc = 1056 :=
sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l2519_251943


namespace NUMINAMATH_CALUDE_estate_area_calculation_l2519_251944

-- Define the scale conversion factor
def scale : ℝ := 500

-- Define the map dimensions
def map_width : ℝ := 5
def map_height : ℝ := 3

-- Define the actual dimensions
def actual_width : ℝ := scale * map_width
def actual_height : ℝ := scale * map_height

-- Define the actual area
def actual_area : ℝ := actual_width * actual_height

-- Theorem to prove
theorem estate_area_calculation :
  actual_area = 3750000 := by
  sorry

end NUMINAMATH_CALUDE_estate_area_calculation_l2519_251944


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2519_251969

theorem unknown_number_proof : 
  ∃ x : ℝ, (45 * x = 0.4 * 900) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2519_251969


namespace NUMINAMATH_CALUDE_two_from_same_class_three_from_same_class_l2519_251939

/-- A function representing the distribution of students among classes -/
def Distribution (n : ℕ) := Fin 3 → ℕ

/-- The sum of students in all classes equals the total number of students -/
def valid_distribution (n : ℕ) (d : Distribution n) : Prop :=
  (d 0) + (d 1) + (d 2) = n

/-- There exists a class with at least k students -/
def exists_class_with_k_students (n k : ℕ) (d : Distribution n) : Prop :=
  ∃ i : Fin 3, d i ≥ k

theorem two_from_same_class (n : ℕ) (h : n ≥ 4) :
  ∀ d : Distribution n, valid_distribution n d → exists_class_with_k_students n 2 d :=
sorry

theorem three_from_same_class (n : ℕ) (h : n ≥ 7) :
  ∀ d : Distribution n, valid_distribution n d → exists_class_with_k_students n 3 d :=
sorry

end NUMINAMATH_CALUDE_two_from_same_class_three_from_same_class_l2519_251939


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l2519_251951

theorem difference_of_squares_factorization (x : ℝ) : 
  x^2 - 4 = (x + 2) * (x - 2) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l2519_251951


namespace NUMINAMATH_CALUDE_shirt_price_theorem_l2519_251947

/-- The price of a shirt when the total cost of the shirt and a coat is $600,
    and the shirt costs one-third the price of the coat. -/
def shirt_price : ℝ := 150

/-- The price of a coat when the total cost of the shirt and the coat is $600,
    and the shirt costs one-third the price of the coat. -/
def coat_price : ℝ := 3 * shirt_price

theorem shirt_price_theorem :
  shirt_price + coat_price = 600 ∧ shirt_price = (1/3) * coat_price →
  shirt_price = 150 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_theorem_l2519_251947


namespace NUMINAMATH_CALUDE_tan_equality_in_range_l2519_251980

theorem tan_equality_in_range : ∃ (n : ℤ), -150 < n ∧ n < 150 ∧ Real.tan (n * π / 180) = Real.tan (286 * π / 180) ∧ n = -74 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_in_range_l2519_251980


namespace NUMINAMATH_CALUDE_fair_remaining_money_l2519_251986

/-- Calculates the remaining money after purchases at a fair --/
theorem fair_remaining_money 
  (initial_amount : ℝ) 
  (toy_cost : ℝ) 
  (hot_dog_cost : ℝ) 
  (candy_apple_cost : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : initial_amount = 15)
  (h2 : toy_cost = 2)
  (h3 : hot_dog_cost = 3.5)
  (h4 : candy_apple_cost = 1.5)
  (h5 : discount_percentage = 0.5)
  (h6 : hot_dog_cost ≥ toy_cost ∧ hot_dog_cost ≥ candy_apple_cost) :
  initial_amount - (toy_cost + hot_dog_cost * (1 - discount_percentage) + candy_apple_cost) = 9.75 := by
  sorry


end NUMINAMATH_CALUDE_fair_remaining_money_l2519_251986


namespace NUMINAMATH_CALUDE_geometric_sequence_divisibility_l2519_251926

theorem geometric_sequence_divisibility (a₁ a₂ : ℚ) (n : ℕ) : 
  a₁ = 5/8 → a₂ = 25 → 
  (∃ k : ℕ, k > 0 ∧ (a₂/a₁)^(k-1) * a₁ % 2000000 = 0) →
  (∀ m : ℕ, m > 0 ∧ m < n → (a₂/a₁)^(m-1) * a₁ % 2000000 ≠ 0) →
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_divisibility_l2519_251926


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2519_251976

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 15 ∧ 
  (∀ (m : ℕ), m < n → ¬(23 ∣ (4499 * 17 + m))) ∧
  (23 ∣ (4499 * 17 + n)) := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2519_251976


namespace NUMINAMATH_CALUDE_expression_evaluation_l2519_251908

theorem expression_evaluation : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2519_251908


namespace NUMINAMATH_CALUDE_inequality_properties_l2519_251958

theorem inequality_properties (m n : ℝ) : 
  (∀ a : ℝ, a > 0 → m * a^2 < n * a^2 → m < n) ∧
  (m < n → n < 0 → n / m < 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_properties_l2519_251958


namespace NUMINAMATH_CALUDE_x₂_integer_part_sum_of_arctans_l2519_251927

-- Define the cubic equation
def cubic_equation (x : ℝ) : ℝ := x^3 - 17*x - 18

-- Define the roots and their properties
axiom x₁ : ℝ
axiom x₂ : ℝ
axiom x₃ : ℝ
axiom x₁_range : -4 < x₁ ∧ x₁ < -3
axiom x₃_range : 4 < x₃ ∧ x₃ < 5
axiom roots_property : cubic_equation x₁ = 0 ∧ cubic_equation x₂ = 0 ∧ cubic_equation x₃ = 0

-- Theorem for the integer part of x₂
theorem x₂_integer_part : ⌊x₂⌋ = -2 := by sorry

-- Theorem for the sum of arctangents
theorem sum_of_arctans : Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = -π/4 := by sorry

end NUMINAMATH_CALUDE_x₂_integer_part_sum_of_arctans_l2519_251927


namespace NUMINAMATH_CALUDE_tangent_line_intersection_at_minus_one_range_of_a_l2519_251911

/-- The function f(x) = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x^2 + a, where a is a parameter -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The derivative of g(x) -/
def g_derivative (x : ℝ) : ℝ := 2 * x

/-- Theorem stating that when x₁ = -1, a = 3 -/
theorem tangent_line_intersection_at_minus_one (a : ℝ) :
  (∃ x₂ : ℝ, f_derivative (-1) = g_derivative x₂ ∧ 
    f (-1) - f_derivative (-1) * (-1) = g a x₂ - g_derivative x₂ * x₂) →
  a = 3 :=
sorry

/-- Theorem stating that a ≥ -1 -/
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, f_derivative x₁ = g_derivative x₂ ∧ 
    f x₁ - f_derivative x₁ * x₁ = g a x₂ - g_derivative x₂ * x₂) →
  a ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_at_minus_one_range_of_a_l2519_251911


namespace NUMINAMATH_CALUDE_waitress_income_fraction_l2519_251931

theorem waitress_income_fraction (salary : ℝ) (tips : ℝ) (income : ℝ) : 
  salary > 0 →
  tips = (7 / 4) * salary →
  income = salary + tips →
  tips / income = 7 / 11 := by
sorry

end NUMINAMATH_CALUDE_waitress_income_fraction_l2519_251931


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l2519_251954

theorem subtraction_of_fractions : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l2519_251954


namespace NUMINAMATH_CALUDE_unique_q_13_l2519_251995

-- Define the cubic polynomial q(x)
def q (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem unique_q_13 (a b c d : ℝ) :
  (∀ x : ℝ, (q a b c d x)^3 - x = 0 → x = 2 ∨ x = -2 ∨ x = 5) →
  q a b c d 2 = 2 →
  q a b c d (-2) = -2 →
  q a b c d 5 = 3 →
  ∃! y : ℝ, q a b c d 13 = y :=
sorry

end NUMINAMATH_CALUDE_unique_q_13_l2519_251995


namespace NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l2519_251999

theorem baby_sea_turtles_on_sand (total : ℕ) (swept_fraction : ℚ) (remaining : ℕ) : 
  total = 42 → 
  swept_fraction = 1/3 → 
  remaining = total - (total * swept_fraction).floor → 
  remaining = 28 := by
sorry

end NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l2519_251999


namespace NUMINAMATH_CALUDE_change_received_l2519_251953

/-- Represents the cost of a basic calculator in dollars -/
def basic_cost : ℕ := 8

/-- Represents the total amount of money the teacher had in dollars -/
def total_money : ℕ := 100

/-- Calculates the cost of a scientific calculator -/
def scientific_cost : ℕ := 2 * basic_cost

/-- Calculates the cost of a graphing calculator -/
def graphing_cost : ℕ := 3 * scientific_cost

/-- Calculates the total cost of buying one of each calculator -/
def total_cost : ℕ := basic_cost + scientific_cost + graphing_cost

/-- Theorem stating that the change received is $28 -/
theorem change_received : total_money - total_cost = 28 := by
  sorry

end NUMINAMATH_CALUDE_change_received_l2519_251953


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l2519_251998

theorem chord_length_concentric_circles (R r : ℝ) (h : R > r) :
  (R^2 - r^2 = 20) →
  ∃ c : ℝ, c > 0 ∧ c^2 / 4 + r^2 = R^2 ∧ c = 4 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l2519_251998


namespace NUMINAMATH_CALUDE_expression_evaluation_l2519_251988

theorem expression_evaluation : 
  let a : ℤ := -2
  (a - 1)^2 - a*(a + 3) + 2*(a + 2)*(a - 2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2519_251988


namespace NUMINAMATH_CALUDE_q_minimized_at_2_l2519_251981

/-- The quadratic function q in terms of x -/
def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

/-- The value of x that minimizes q -/
def minimizing_x : ℝ := 2

theorem q_minimized_at_2 :
  ∀ x : ℝ, q x ≥ q minimizing_x :=
sorry

end NUMINAMATH_CALUDE_q_minimized_at_2_l2519_251981


namespace NUMINAMATH_CALUDE_min_value_product_l2519_251967

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : (2/x) + (3/y) + (1/z) = 12) : 
  x^2 * y^3 * z ≥ (1/64) :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l2519_251967


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l2519_251910

/-- The equation (x+y)^2 = x^2 + y^2 + 2 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (x y : ℝ), (x + y)^2 = x^2 + y^2 + 2 ↔ x * y = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l2519_251910


namespace NUMINAMATH_CALUDE_sqrt_operations_l2519_251946

theorem sqrt_operations :
  (∀ x y : ℝ, x > 0 → y > 0 → (Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y)) ∧
  (Real.sqrt 12 / Real.sqrt 3 = 2) ∧
  (Real.sqrt 8 = 2 * Real.sqrt 2) ∧
  (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) ∧
  (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_operations_l2519_251946


namespace NUMINAMATH_CALUDE_square_root_identity_polynomial_identity_square_root_polynomial_l2519_251963

theorem square_root_identity (n : ℕ) : 
  Real.sqrt ((n - 1) * (n + 1) + 1) = n :=
sorry

theorem polynomial_identity (n : ℕ) : 
  (n * (n + 3) + 1)^2 = n * (n + 1) * (n + 2) * (n + 3) + 1 :=
sorry

theorem square_root_polynomial (n : ℕ) : 
  Real.sqrt (n * (n + 1) * (n + 2) * (n + 3) + 1) = n * (n + 3) :=
sorry

end NUMINAMATH_CALUDE_square_root_identity_polynomial_identity_square_root_polynomial_l2519_251963


namespace NUMINAMATH_CALUDE_conference_handshakes_l2519_251937

/-- Represents a conference with two groups of people -/
structure Conference where
  total_people : ℕ
  group_x : ℕ
  group_y : ℕ
  known_people : ℕ
  h_total : total_people = group_x + group_y
  h_group_x : group_x = 25
  h_group_y : group_y = 15
  h_known : known_people = 5

/-- Calculates the number of handshakes in the conference -/
def handshakes (c : Conference) : ℕ :=
  let between_groups := c.group_x * c.group_y
  let within_x := (c.group_x * (c.group_x - 1 - c.known_people)) / 2
  let within_y := (c.group_y * (c.group_y - 1)) / 2
  between_groups + within_x + within_y

/-- Theorem stating that the number of handshakes in the given conference is 717 -/
theorem conference_handshakes :
    ∃ (c : Conference), handshakes c = 717 :=
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l2519_251937


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l2519_251942

theorem hexagon_perimeter (AB BC CD DE EF : ℝ) (AC AD AE AF : ℝ) : 
  AB = 1 →
  BC = 1 →
  CD = 1 →
  DE = 2 →
  EF = 1 →
  AC^2 = AB^2 + BC^2 →
  AD^2 = AC^2 + CD^2 →
  AE^2 = AD^2 + DE^2 →
  AF^2 = AE^2 + EF^2 →
  AB + BC + CD + DE + EF + AF = 6 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l2519_251942


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l2519_251900

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l2519_251900


namespace NUMINAMATH_CALUDE_emily_roses_purchase_l2519_251993

theorem emily_roses_purchase (flower_cost : ℕ) (total_spent : ℕ) : 
  flower_cost = 3 →
  total_spent = 12 →
  ∃ (roses : ℕ), roses * 2 * flower_cost = total_spent ∧ roses = 2 :=
by sorry

end NUMINAMATH_CALUDE_emily_roses_purchase_l2519_251993


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l2519_251902

/-- The degree measure of an interior angle of a regular n-gon -/
def interior_angle (n : ℕ) : ℚ := 180 * (n - 2) / n

theorem exterior_angle_measure :
  let square_angle : ℚ := 90
  let heptagon_angle : ℚ := interior_angle 7
  let exterior_angle : ℚ := 360 - heptagon_angle - square_angle
  exterior_angle = 990 / 7 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l2519_251902


namespace NUMINAMATH_CALUDE_football_field_area_l2519_251996

theorem football_field_area (total_fertilizer : ℝ) (partial_fertilizer : ℝ) (partial_area : ℝ) :
  total_fertilizer = 1200 →
  partial_fertilizer = 400 →
  partial_area = 3600 →
  (total_fertilizer / (partial_fertilizer / partial_area)) = 10800 := by
  sorry

end NUMINAMATH_CALUDE_football_field_area_l2519_251996


namespace NUMINAMATH_CALUDE_circle_equation_l2519_251909

/-- A circle with center on the y-axis, radius 1, passing through (1,2) has equation x^2 + (y-2)^2 = 1 -/
theorem circle_equation : ∃ (b : ℝ), 
  (∀ (x y : ℝ), x^2 + (y - b)^2 = 1 ↔ ((x = 1 ∧ y = 2) ∨ (x^2 + (y - 2)^2 = 1))) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2519_251909


namespace NUMINAMATH_CALUDE_unsold_books_percentage_l2519_251924

def initial_stock : ℕ := 700
def monday_sales : ℕ := 50
def tuesday_sales : ℕ := 82
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

theorem unsold_books_percentage :
  let total_sales := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales
  let unsold_books := initial_stock - total_sales
  (unsold_books : ℚ) / initial_stock * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_percentage_l2519_251924


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2519_251990

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x + k^2 - 1 = 0) ↔ k ∈ Set.Icc (-2*Real.sqrt 3/3) (2*Real.sqrt 3/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2519_251990


namespace NUMINAMATH_CALUDE_souvenir_walk_distance_l2519_251949

theorem souvenir_walk_distance (total : ℝ) (hotel_to_postcard : ℝ) (postcard_to_tshirt : ℝ)
  (h1 : total = 0.89)
  (h2 : hotel_to_postcard = 0.11)
  (h3 : postcard_to_tshirt = 0.11) :
  total - (hotel_to_postcard + postcard_to_tshirt) = 0.67 := by
sorry

end NUMINAMATH_CALUDE_souvenir_walk_distance_l2519_251949


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2519_251919

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x^2 + x - 2 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2519_251919


namespace NUMINAMATH_CALUDE_peggy_initial_dolls_l2519_251960

theorem peggy_initial_dolls :
  ∀ (initial_dolls : ℕ) 
    (grandmother_dolls : ℕ) 
    (birthday_christmas_dolls : ℕ) 
    (total_dolls : ℕ),
  grandmother_dolls = 30 →
  birthday_christmas_dolls = grandmother_dolls / 2 →
  total_dolls = 51 →
  initial_dolls + grandmother_dolls + birthday_christmas_dolls = total_dolls →
  initial_dolls = 6 := by
sorry

end NUMINAMATH_CALUDE_peggy_initial_dolls_l2519_251960


namespace NUMINAMATH_CALUDE_nina_not_taller_than_lena_l2519_251935

-- Define the set of friends
inductive Friend : Type
| Masha : Friend
| Nina : Friend
| Lena : Friend
| Olya : Friend

-- Define a height comparison relation
def TallerThan : Friend → Friend → Prop := sorry

-- State the theorem
theorem nina_not_taller_than_lena 
  (h1 : TallerThan Friend.Masha Friend.Nina)
  (h2 : TallerThan Friend.Lena Friend.Olya)
  (h3 : ∀ (f1 f2 : Friend), f1 ≠ f2 → (TallerThan f1 f2 ∨ TallerThan f2 f1))
  (h4 : ∀ (f1 f2 f3 : Friend), TallerThan f1 f2 → TallerThan f2 f3 → TallerThan f1 f3) :
  ¬(TallerThan Friend.Nina Friend.Lena) :=
sorry

end NUMINAMATH_CALUDE_nina_not_taller_than_lena_l2519_251935


namespace NUMINAMATH_CALUDE_steve_pie_difference_l2519_251907

/-- Represents a baker's weekly pie production --/
structure BakerProduction where
  pies_per_day : ℕ
  apple_pie_days : ℕ
  cherry_pie_days : ℕ

/-- Calculates the difference between apple pies and cherry pies baked in a week --/
def pie_difference (bp : BakerProduction) : ℕ :=
  bp.pies_per_day * bp.apple_pie_days - bp.pies_per_day * bp.cherry_pie_days

/-- Theorem stating the difference in pie production for Steve's bakery --/
theorem steve_pie_difference :
  ∀ (bp : BakerProduction),
    bp.pies_per_day = 12 →
    bp.apple_pie_days = 3 →
    bp.cherry_pie_days = 2 →
    pie_difference bp = 12 := by
  sorry

end NUMINAMATH_CALUDE_steve_pie_difference_l2519_251907


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2519_251974

theorem greatest_divisor_with_remainders : 
  let a := 6215 - 23
  let b := 7373 - 29
  let c := 8927 - 35
  Nat.gcd a (Nat.gcd b c) = 36 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2519_251974


namespace NUMINAMATH_CALUDE_trig_expression_equals_zero_l2519_251912

theorem trig_expression_equals_zero :
  Real.cos (π / 3) - Real.tan (π / 4) + (3 / 4) * (Real.tan (π / 6))^2 - Real.sin (π / 6) + (Real.cos (π / 6))^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_zero_l2519_251912


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l2519_251915

/-- The y-coordinate of the vertex of the parabola y = 2x^2 + 16x + 35 is 3 -/
theorem parabola_vertex_y_coordinate :
  let f (x : ℝ) := 2 * x^2 + 16 * x + 35
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l2519_251915


namespace NUMINAMATH_CALUDE_fayes_coloring_books_l2519_251923

theorem fayes_coloring_books : 
  ∀ (initial_books : ℕ), 
  (initial_books - 3 + 48 = 79) → initial_books = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_fayes_coloring_books_l2519_251923


namespace NUMINAMATH_CALUDE_expectation_linear_transform_binomial_probability_normal_probability_l2519_251971

/-- The expectation of a random variable -/
noncomputable def expectation (X : Real → Real) : Real := sorry

/-- The variance of a random variable -/
noncomputable def variance (X : Real → Real) : Real := sorry

/-- The probability mass function for a binomial distribution -/
noncomputable def binomial_pmf (n : Nat) (p : Real) (k : Nat) : Real := sorry

/-- The cumulative distribution function for a normal distribution -/
noncomputable def normal_cdf (μ σ : Real) (x : Real) : Real := sorry

theorem expectation_linear_transform (X : Real → Real) :
  expectation (fun x => 2 * x + 3) = 2 * expectation X + 3 := by sorry

theorem binomial_probability (X : Real → Real) :
  binomial_pmf 6 (1/2) 3 = 5/16 := by sorry

theorem normal_probability (X : Real → Real) (σ : Real) :
  normal_cdf 2 σ 4 = 0.9 →
  normal_cdf 2 σ 2 - normal_cdf 2 σ 0 = 0.4 := by sorry

end NUMINAMATH_CALUDE_expectation_linear_transform_binomial_probability_normal_probability_l2519_251971


namespace NUMINAMATH_CALUDE_birds_on_fence_l2519_251956

/-- The number of birds that fly away -/
def birds_flown : ℝ := 8.0

/-- The number of birds left on the fence -/
def birds_left : ℕ := 4

/-- The initial number of birds on the fence -/
def initial_birds : ℝ := birds_flown + birds_left

theorem birds_on_fence : initial_birds = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2519_251956


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_below_neg_85_l2519_251978

theorem largest_multiple_of_seven_below_neg_85 :
  ∀ n : ℤ, 7 ∣ n ∧ n < -85 → n ≤ -91 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_below_neg_85_l2519_251978


namespace NUMINAMATH_CALUDE_basketball_team_size_l2519_251982

theorem basketball_team_size 
  (total_score : ℕ) 
  (min_score : ℕ) 
  (max_score : ℕ) 
  (h1 : total_score = 100) 
  (h2 : min_score = 7) 
  (h3 : max_score = 23) :
  ∃ (team_size : ℕ), 
    team_size * min_score ≤ total_score ∧ 
    total_score ≤ (team_size - 1) * min_score + max_score ∧
    team_size = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_team_size_l2519_251982


namespace NUMINAMATH_CALUDE_circle_symmetry_l2519_251933

/-- Given a circle C1 with equation (x+1)^2 + (y-1)^2 = 1, 
    prove that the circle C2 with equation (x-2)^2 + (y+2)^2 = 1 
    is symmetric to C1 with respect to the line x - y - 1 = 0 -/
theorem circle_symmetry (x y : ℝ) : 
  (∀ x y, (x + 1)^2 + (y - 1)^2 = 1 → 
    ∃ x' y', x' - y' = -(x - y) ∧ (x' + 1)^2 + (y' - 1)^2 = 1) → 
  (x - 2)^2 + (y + 2)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2519_251933


namespace NUMINAMATH_CALUDE_min_period_cosine_l2519_251916

/-- The minimum positive period of the cosine function Y = 3cos(2/5x - π/6) is 5π. -/
theorem min_period_cosine (x : ℝ) : 
  let Y : ℝ → ℝ := λ x => 3 * Real.cos ((2/5) * x - π/6)
  ∃ (T : ℝ), T > 0 ∧ (∀ t, Y (t + T) = Y t) ∧ (∀ S, S > 0 ∧ (∀ t, Y (t + S) = Y t) → T ≤ S) ∧ T = 5 * π :=
by sorry

end NUMINAMATH_CALUDE_min_period_cosine_l2519_251916


namespace NUMINAMATH_CALUDE_tanning_salon_revenue_l2519_251984

/-- Calculate the revenue of a tanning salon for a month --/
theorem tanning_salon_revenue :
  let first_visit_charge : ℚ := 10
  let subsequent_visit_charge : ℚ := 8
  let discount_rate : ℚ := 0.1
  let premium_service_charge : ℚ := 15
  let premium_service_rate : ℚ := 0.2
  let total_customers : ℕ := 150
  let second_visit_customers : ℕ := 40
  let third_visit_customers : ℕ := 15
  let fourth_visit_customers : ℕ := 5

  let first_visit_revenue : ℚ := 
    (premium_service_rate * total_customers.cast) * premium_service_charge +
    ((1 - premium_service_rate) * total_customers.cast) * first_visit_charge
  let second_visit_revenue : ℚ := second_visit_customers.cast * subsequent_visit_charge
  let discounted_visit_charge : ℚ := subsequent_visit_charge * (1 - discount_rate)
  let third_visit_revenue : ℚ := third_visit_customers.cast * discounted_visit_charge
  let fourth_visit_revenue : ℚ := fourth_visit_customers.cast * discounted_visit_charge

  let total_revenue : ℚ := 
    first_visit_revenue + second_visit_revenue + third_visit_revenue + fourth_visit_revenue

  total_revenue = 2114 := by sorry

end NUMINAMATH_CALUDE_tanning_salon_revenue_l2519_251984


namespace NUMINAMATH_CALUDE_mortdecai_charity_donation_l2519_251957

/-- Represents the number of eggs in a dozen --/
def dozen : ℕ := 12

/-- Represents the number of days Mortdecai collects eggs --/
def collection_days : ℕ := 2

/-- Represents the number of dozens of eggs Mortdecai collects per day --/
def collected_dozens_per_day : ℕ := 8

/-- Represents the number of dozens of eggs Mortdecai delivers to the market --/
def market_delivery : ℕ := 3

/-- Represents the number of dozens of eggs Mortdecai delivers to the mall --/
def mall_delivery : ℕ := 5

/-- Represents the number of dozens of eggs Mortdecai uses for pie --/
def pie_dozens : ℕ := 4

/-- Theorem stating that Mortdecai donates 48 eggs to charity --/
theorem mortdecai_charity_donation : 
  (collection_days * collected_dozens_per_day - (market_delivery + mall_delivery) - pie_dozens) * dozen = 48 := by
  sorry

end NUMINAMATH_CALUDE_mortdecai_charity_donation_l2519_251957


namespace NUMINAMATH_CALUDE_average_permutation_sum_l2519_251961

def permutation_sum (b : Fin 8 → Fin 8) : ℕ :=
  |b 0 - b 1| + |b 2 - b 3| + |b 4 - b 5| + |b 6 - b 7|

def all_permutations : Finset (Fin 8 → Fin 8) :=
  Finset.univ.filter (λ b ↦ Function.Bijective b)

theorem average_permutation_sum :
  (Finset.sum all_permutations permutation_sum) / all_permutations.card = 672 := by
  sorry

end NUMINAMATH_CALUDE_average_permutation_sum_l2519_251961


namespace NUMINAMATH_CALUDE_polynomial_roots_l2519_251906

theorem polynomial_roots (AT TB : ℝ) (h1 : AT + TB = 15) (h2 : AT * TB = 36) :
  ∃ (p : ℝ → ℝ), p = (fun x ↦ x^2 - 20*x + 75) ∧ 
  p (AT + 5) = 0 ∧ p TB = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2519_251906


namespace NUMINAMATH_CALUDE_sqrt_greater_than_3x_iff_less_than_one_ninth_l2519_251973

theorem sqrt_greater_than_3x_iff_less_than_one_ninth 
  (x : ℝ) (hx : x > 0) : 
  Real.sqrt x > 3 * x ↔ x < 1/9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_greater_than_3x_iff_less_than_one_ninth_l2519_251973


namespace NUMINAMATH_CALUDE_chess_tournament_director_games_l2519_251970

theorem chess_tournament_director_games (total_games : ℕ) (h : total_games = 325) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_games ∧ 
  ∀ (k : ℕ), n * (n - 1) / 2 + k = total_games → k ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_director_games_l2519_251970


namespace NUMINAMATH_CALUDE_inequality_of_distinct_positives_l2519_251991

theorem inequality_of_distinct_positives (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_distinct_ab : a ≠ b) (h_distinct_ac : a ≠ c) (h_distinct_bc : b ≠ c) :
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_distinct_positives_l2519_251991
