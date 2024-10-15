import Mathlib

namespace NUMINAMATH_CALUDE_three_digit_square_end_same_l3710_371050

theorem three_digit_square_end_same (A : ℕ) : 
  (100 ≤ A ∧ A < 1000) ∧ (A^2 % 1000 = A) ↔ (A = 376 ∨ A = 625) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_square_end_same_l3710_371050


namespace NUMINAMATH_CALUDE_log_length_l3710_371071

/-- Represents the properties of a log that has been cut in half -/
structure LogCut where
  weight_per_foot : ℝ
  weight_of_piece : ℝ
  original_length : ℝ

/-- Theorem stating that given the conditions, the original log length is 20 feet -/
theorem log_length (log : LogCut) 
  (h1 : log.weight_per_foot = 150)
  (h2 : log.weight_of_piece = 1500) :
  log.original_length = 20 := by
  sorry

#check log_length

end NUMINAMATH_CALUDE_log_length_l3710_371071


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_l3710_371084

/-- Given two points A and B, if the line AB is parallel to the x-axis, then m = -1 --/
theorem line_parallel_to_x_axis (m : ℝ) : 
  let A : ℝ × ℝ := (m + 1, -2)
  let B : ℝ × ℝ := (3, m - 1)
  (A.2 = B.2) → m = -1 := by sorry

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_l3710_371084


namespace NUMINAMATH_CALUDE_probability_of_selecting_boy_l3710_371008

/-- Given a class with 60 students where 24 are girls, the probability of selecting a boy is 0.6 -/
theorem probability_of_selecting_boy (total_students : ℕ) (num_girls : ℕ) 
  (h1 : total_students = 60) 
  (h2 : num_girls = 24) : 
  (total_students - num_girls : ℚ) / total_students = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_boy_l3710_371008


namespace NUMINAMATH_CALUDE_max_min_difference_c_l3710_371073

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 2) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 12) : 
  ∃ (c_max c_min : ℝ), 
    (∀ c', (∃ a' b', a' + b' + c' = 2 ∧ a'^2 + b'^2 + c'^2 = 12) → c' ≤ c_max) ∧
    (∀ c', (∃ a' b', a' + b' + c' = 2 ∧ a'^2 + b'^2 + c'^2 = 12) → c' ≥ c_min) ∧
    c_max - c_min = 16/3 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l3710_371073


namespace NUMINAMATH_CALUDE_fourth_triangle_exists_l3710_371049

/-- Given four positive real numbers that can form three different triangles,
    prove that they can form a fourth triangle. -/
theorem fourth_triangle_exists (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab_c : a + b > c ∧ a + c > b ∧ b + c > a)
  (hab_d : a + b > d ∧ a + d > b ∧ b + d > a)
  (acd : a + c > d ∧ a + d > c ∧ c + d > a) :
  b + c > d ∧ b + d > c ∧ c + d > b := by
  sorry

end NUMINAMATH_CALUDE_fourth_triangle_exists_l3710_371049


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3710_371063

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 20) 
  (eq2 : x + 4 * y = 26) : 
  17 * x^2 + 20 * x * y + 17 * y^2 = 1076 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3710_371063


namespace NUMINAMATH_CALUDE_log_product_equals_one_l3710_371082

theorem log_product_equals_one : Real.log 2 / Real.log 5 * (2 * Real.log 5 / (2 * Real.log 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_one_l3710_371082


namespace NUMINAMATH_CALUDE_unique_root_of_sum_with_shift_l3710_371060

/-- Given a monic quadratic polynomial with two distinct roots, 
    prove that f(x) + f(x - √D) = 0 has exactly one root. -/
theorem unique_root_of_sum_with_shift 
  (b c : ℝ) 
  (h_distinct : ∃ (x y : ℝ), x ≠ y ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0) :
  ∃! x : ℝ, (x^2 + b*x + c) + ((x - Real.sqrt (b^2 - 4*c))^2 + b*(x - Real.sqrt (b^2 - 4*c)) + c) = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_root_of_sum_with_shift_l3710_371060


namespace NUMINAMATH_CALUDE_prime_power_equality_l3710_371062

theorem prime_power_equality (p : ℕ) (k : ℕ) (hp : Prime p) (hk : k > 1) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (m, n) ≠ (1, 1) ∧ 
    (m^p + n^p) / 2 = ((m + n) / 2)^k) ↔ k = p :=
by sorry

end NUMINAMATH_CALUDE_prime_power_equality_l3710_371062


namespace NUMINAMATH_CALUDE_integer_root_of_cubic_l3710_371092

theorem integer_root_of_cubic (b c : ℚ) :
  (∃ x : ℝ, x^3 + b*x + c = 0 ∧ x = 5 - Real.sqrt 21) →
  (∃ n : ℤ, n^3 + b*n + c = 0) →
  (∃ n : ℤ, n^3 + b*n + c = 0 ∧ n = -10) := by
sorry

end NUMINAMATH_CALUDE_integer_root_of_cubic_l3710_371092


namespace NUMINAMATH_CALUDE_coprimality_preserving_polynomials_l3710_371086

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The property that a polynomial preserves coprimality -/
def PreservesCoprimality (P : IntPolynomial) : Prop :=
  ∀ a b : ℤ, Int.gcd a b = 1 → Int.gcd (P.eval a) (P.eval b) = 1

/-- Characterization of polynomials that preserve coprimality -/
theorem coprimality_preserving_polynomials :
  ∀ P : IntPolynomial,
  PreservesCoprimality P ↔
  (∃ n : ℕ, P = Polynomial.monomial n 1) ∨
  (∃ n : ℕ, P = Polynomial.monomial n (-1)) :=
sorry

end NUMINAMATH_CALUDE_coprimality_preserving_polynomials_l3710_371086


namespace NUMINAMATH_CALUDE_interview_scores_properties_l3710_371048

def scores : List ℝ := [70, 85, 86, 88, 90, 90, 92, 94, 95, 100]

def sixtieth_percentile (l : List ℝ) : ℝ := sorry

def average (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def remove_extremes (l : List ℝ) : List ℝ := sorry

theorem interview_scores_properties :
  let s := scores
  let s_without_extremes := remove_extremes s
  (sixtieth_percentile s = 91) ∧
  (average s_without_extremes > average s) ∧
  (variance s_without_extremes < variance s) ∧
  (¬ (9 / 45 = 1 / 10)) ∧
  (¬ (average s > median s)) := by sorry

end NUMINAMATH_CALUDE_interview_scores_properties_l3710_371048


namespace NUMINAMATH_CALUDE_triangle_inradius_l3710_371036

/-- Given a triangle with perimeter 20 cm and area 25 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) :
  p = 20 →
  A = 25 →
  A = r * p / 2 →
  r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l3710_371036


namespace NUMINAMATH_CALUDE_green_balls_count_l3710_371006

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob_not_red_purple : ℚ) 
  (h_total : total = 60)
  (h_white : white = 22)
  (h_yellow : yellow = 8)
  (h_red : red = 5)
  (h_purple : purple = 7)
  (h_prob : prob_not_red_purple = 4/5) :
  ∃ green : ℕ, 
    green = total - (white + yellow + red + purple) ∧ 
    (white + green + yellow : ℚ) / total = prob_not_red_purple :=
by sorry

end NUMINAMATH_CALUDE_green_balls_count_l3710_371006


namespace NUMINAMATH_CALUDE_large_cheese_block_volume_l3710_371056

/-- The volume of a large cheese block is 32 cubic feet -/
theorem large_cheese_block_volume :
  ∀ (w d l : ℝ),
  w * d * l = 4 →
  (2 * w) * (2 * d) * (2 * l) = 32 :=
by sorry

end NUMINAMATH_CALUDE_large_cheese_block_volume_l3710_371056


namespace NUMINAMATH_CALUDE_money_left_after_tickets_l3710_371014

/-- The amount of money Olivia and Nigel have left after buying tickets -/
def money_left (olivia_money : ℕ) (nigel_money : ℕ) (num_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (olivia_money + nigel_money) - (num_tickets * ticket_price)

/-- Theorem stating the amount of money left after buying tickets -/
theorem money_left_after_tickets :
  money_left 112 139 6 28 = 83 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_tickets_l3710_371014


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l3710_371055

/-- The volume of a rectangular solid with specific face areas and sum of dimensions -/
theorem rectangular_solid_volume
  (a b c : ℝ)
  (side_area : a * b = 15)
  (front_area : b * c = 10)
  (bottom_area : c * a = 6)
  (sum_dimensions : a + b + c = 11)
  : a * b * c = 90 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l3710_371055


namespace NUMINAMATH_CALUDE_x_minus_y_values_l3710_371020

theorem x_minus_y_values (x y : ℝ) (hx : |x| = 5) (hy : |y| = 3) (hxy : y > x) :
  x - y = -8 ∨ x - y = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l3710_371020


namespace NUMINAMATH_CALUDE_anthony_pencils_l3710_371064

/-- The number of pencils Anthony has after giving some to Kathryn -/
def pencils_remaining (initial : Float) (given : Float) : Float :=
  initial - given

/-- Theorem: Anthony has 47.0 pencils after giving some to Kathryn -/
theorem anthony_pencils :
  pencils_remaining 56.0 9.0 = 47.0 := by
  sorry

end NUMINAMATH_CALUDE_anthony_pencils_l3710_371064


namespace NUMINAMATH_CALUDE_circle_equation_l3710_371085

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line on which the center of the required circle lies
def centerLine (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

-- Define the equation of the required circle
def requiredCircle (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 13

-- Theorem statement
theorem circle_equation : 
  ∀ x y : ℝ, 
  (circle1 x y ∧ circle2 x y) → 
  (∃ a b : ℝ, centerLine a b ∧ 
    ((x - a)^2 + (y - b)^2 = (x + 1)^2 + (y - 1)^2)) → 
  requiredCircle x y :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3710_371085


namespace NUMINAMATH_CALUDE_no_solution_condition_l3710_371094

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (m * x - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_condition_l3710_371094


namespace NUMINAMATH_CALUDE_emma_hit_eleven_l3710_371023

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

end NUMINAMATH_CALUDE_emma_hit_eleven_l3710_371023


namespace NUMINAMATH_CALUDE_polygon_with_45_degree_exterior_angles_has_8_sides_l3710_371045

/-- A polygon with exterior angles of 45° has 8 sides. -/
theorem polygon_with_45_degree_exterior_angles_has_8_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
  n > 0 →
  exterior_angle = 45 →
  (n : ℝ) * exterior_angle = 360 →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_with_45_degree_exterior_angles_has_8_sides_l3710_371045


namespace NUMINAMATH_CALUDE_pells_equation_unique_solution_l3710_371001

-- Define the fundamental solution
def fundamental_solution (x₀ y₀ : ℤ) : Prop :=
  x₀^2 - 2003 * y₀^2 = 1 ∧ x₀ > 0 ∧ y₀ > 0

-- Define the property that all prime factors of x divide x₀
def all_prime_factors_divide (x x₀ : ℤ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p : ℤ) ∣ x → (p : ℤ) ∣ x₀

-- Main theorem
theorem pells_equation_unique_solution
  (x₀ y₀ x y : ℤ)
  (h_fund : fundamental_solution x₀ y₀)
  (h_sol : x^2 - 2003 * y^2 = 1)
  (h_pos : x > 0 ∧ y > 0)
  (h_divide : all_prime_factors_divide x x₀) :
  x = x₀ ∧ y = y₀ :=
sorry

end NUMINAMATH_CALUDE_pells_equation_unique_solution_l3710_371001


namespace NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l3710_371047

theorem order_of_logarithmic_expressions :
  let a := 2 * Real.log 0.99
  let b := Real.log 0.98
  let c := Real.sqrt 0.96 - 1
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l3710_371047


namespace NUMINAMATH_CALUDE_maintenance_time_is_three_minutes_l3710_371051

/-- Represents the passage scenario with order maintenance --/
structure PassageScenario where
  normal_rate : ℕ
  congested_rate : ℕ
  people_waiting : ℕ
  time_saved : ℕ

/-- Calculates the time spent maintaining order --/
def maintenance_time (scenario : PassageScenario) : ℕ :=
  let total_wait_time := scenario.people_waiting / scenario.congested_rate
  let actual_wait_time := total_wait_time - scenario.time_saved
  actual_wait_time

/-- Theorem stating that the maintenance time is 3 minutes for the given scenario --/
theorem maintenance_time_is_three_minutes 
  (scenario : PassageScenario)
  (h1 : scenario.normal_rate = 9)
  (h2 : scenario.congested_rate = 3)
  (h3 : scenario.people_waiting = 36)
  (h4 : scenario.time_saved = 6) :
  maintenance_time scenario = 3 := by
  sorry

#eval maintenance_time { normal_rate := 9, congested_rate := 3, people_waiting := 36, time_saved := 6 }

end NUMINAMATH_CALUDE_maintenance_time_is_three_minutes_l3710_371051


namespace NUMINAMATH_CALUDE_x_gt_y_iff_x_minus_y_plus_sin_gt_zero_l3710_371081

theorem x_gt_y_iff_x_minus_y_plus_sin_gt_zero (x y : ℝ) :
  x > y ↔ x - y + Real.sin (x - y) > 0 := by sorry

end NUMINAMATH_CALUDE_x_gt_y_iff_x_minus_y_plus_sin_gt_zero_l3710_371081


namespace NUMINAMATH_CALUDE_evaluation_ratio_l3710_371069

def relevance_percentage : ℚ := 45 / 100
def language_percentage : ℚ := 25 / 100
def structure_percentage : ℚ := 30 / 100

theorem evaluation_ratio :
  let r := relevance_percentage
  let l := language_percentage
  let s := structure_percentage
  let gcd := (r * 100).num.gcd ((l * 100).num.gcd (s * 100).num)
  ((r * 100).num / gcd, (l * 100).num / gcd, (s * 100).num / gcd) = (9, 5, 6) := by
  sorry

end NUMINAMATH_CALUDE_evaluation_ratio_l3710_371069


namespace NUMINAMATH_CALUDE_purple_balls_count_l3710_371018

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 ∧
  white = 50 ∧
  green = 20 ∧
  yellow = 10 ∧
  red = 17 ∧
  prob_not_red_purple = 4/5 →
  total - (white + green + yellow + red) = 3 := by
sorry

end NUMINAMATH_CALUDE_purple_balls_count_l3710_371018


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l3710_371007

theorem probability_of_black_ball 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (prob_white : ℚ) :
  total_balls = 100 →
  red_balls = 45 →
  prob_white = 23/100 →
  (total_balls - red_balls - (total_balls * prob_white).floor) / total_balls = 32/100 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l3710_371007


namespace NUMINAMATH_CALUDE_twelve_integer_chords_l3710_371042

/-- Represents a circle with a point inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceToCenter : ℝ

/-- Counts the number of integer-length chords through a point in a circle -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- Theorem stating that for a circle with radius 15 and a point 9 units from the center,
    there are exactly 12 integer-length chords through that point -/
theorem twelve_integer_chords :
  let c : CircleWithPoint := { radius := 15, distanceToCenter := 9 }
  countIntegerChords c = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_integer_chords_l3710_371042


namespace NUMINAMATH_CALUDE_oranges_per_tree_l3710_371027

/-- Represents the number of oranges picked by Betty -/
def betty_oranges : ℕ := 15

/-- Represents the number of oranges picked by Bill -/
def bill_oranges : ℕ := 12

/-- Represents the number of oranges picked by Frank -/
def frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)

/-- Represents the number of seeds Frank planted -/
def seeds_planted : ℕ := 2 * frank_oranges

/-- Represents the total number of oranges Philip can pick -/
def philip_total_oranges : ℕ := 810

/-- Theorem stating that the number of oranges per tree for Philip to pick is 5 -/
theorem oranges_per_tree :
  philip_total_oranges / seeds_planted = 5 := by sorry

end NUMINAMATH_CALUDE_oranges_per_tree_l3710_371027


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l3710_371033

/-- The area of a square with adjacent vertices at (1,3) and (5,6) is 25 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (5, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l3710_371033


namespace NUMINAMATH_CALUDE_distance_to_origin_l3710_371079

/-- The distance from point P (-2, 4) to the origin (0, 0) is 2√5 -/
theorem distance_to_origin : 
  let P : ℝ × ℝ := (-2, 4)
  let O : ℝ × ℝ := (0, 0)
  Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3710_371079


namespace NUMINAMATH_CALUDE_marbles_shared_proof_l3710_371026

/-- The number of marbles Jack starts with -/
def initial_marbles : ℕ := 62

/-- The number of marbles Jack ends with -/
def final_marbles : ℕ := 29

/-- The number of marbles Jack shared with Rebecca -/
def shared_marbles : ℕ := initial_marbles - final_marbles

theorem marbles_shared_proof : shared_marbles = 33 := by
  sorry

end NUMINAMATH_CALUDE_marbles_shared_proof_l3710_371026


namespace NUMINAMATH_CALUDE_triangle_properties_l3710_371000

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: If b - (1/2)c = a cos C, 4(b + c) = 3bc, and a = 2√3 in a triangle ABC,
    then angle A = 60° and the area of the triangle is 2√3 --/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b - (1/2) * t.c = t.a * Real.cos t.C)
  (h2 : 4 * (t.b + t.c) = 3 * t.b * t.c)
  (h3 : t.a = 2 * Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ 
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3710_371000


namespace NUMINAMATH_CALUDE_integer_midpoint_exists_l3710_371057

def Point := ℤ × ℤ

theorem integer_midpoint_exists (P : Fin 5 → Point) :
  ∃ i j : Fin 5, i ≠ j ∧ 
    let (xi, yi) := P i
    let (xj, yj) := P j
    (xi + xj) % 2 = 0 ∧ (yi + yj) % 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_integer_midpoint_exists_l3710_371057


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_expression_l3710_371072

theorem largest_power_of_two_dividing_expression : ∃ k : ℕ, 
  (2^k : ℤ) ∣ (15^4 - 7^4 - 8) ∧ 
  ∀ m : ℕ, (2^m : ℤ) ∣ (15^4 - 7^4 - 8) → m ≤ k ∧
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_expression_l3710_371072


namespace NUMINAMATH_CALUDE_parallelogram_area_l3710_371077

structure Vector2D where
  x : ℝ
  y : ℝ

def angle (v w : Vector2D) : ℝ := sorry

def norm (v : Vector2D) : ℝ := sorry

def cross (v w : Vector2D) : ℝ := sorry

theorem parallelogram_area (p q : Vector2D) : 
  let a := Vector2D.mk (6 * p.x - q.x) (6 * p.y - q.y)
  let b := Vector2D.mk (5 * q.x + p.x) (5 * q.y + p.y)
  norm p = 1/2 →
  norm q = 4 →
  angle p q = 5 * π / 6 →
  abs (cross a b) = 31 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3710_371077


namespace NUMINAMATH_CALUDE_benjamin_total_steps_l3710_371041

/-- Calculates the total distance traveled in steps given various modes of transportation -/
def total_steps_traveled (steps_per_mile : ℕ) (initial_walk : ℕ) (subway_miles : ℕ) (second_walk : ℕ) (cab_miles : ℕ) : ℕ :=
  initial_walk + (subway_miles * steps_per_mile) + second_walk + (cab_miles * steps_per_mile)

/-- The total steps traveled by Benjamin is 24000 -/
theorem benjamin_total_steps :
  total_steps_traveled 2000 2000 7 3000 3 = 24000 := by
  sorry


end NUMINAMATH_CALUDE_benjamin_total_steps_l3710_371041


namespace NUMINAMATH_CALUDE_park_entry_exit_choices_l3710_371074

def num_gates : ℕ := 5

theorem park_entry_exit_choices :
  (num_gates * (num_gates - 1) : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_park_entry_exit_choices_l3710_371074


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_l3710_371068

theorem tan_pi_minus_alpha (α : Real) (h : 3 * Real.sin α = Real.cos α) :
  Real.tan (π - α) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_l3710_371068


namespace NUMINAMATH_CALUDE_count_words_with_e_l3710_371040

/-- The number of letters in our alphabet -/
def n : ℕ := 5

/-- The length of the words we're creating -/
def k : ℕ := 4

/-- The number of letters in our alphabet excluding E -/
def m : ℕ := 4

/-- The number of 4-letter words that can be made from 5 letters (A, B, C, D, E) with repetition allowed -/
def total_words : ℕ := n ^ k

/-- The number of 4-letter words that can be made from 4 letters (A, B, C, D) with repetition allowed -/
def words_without_e : ℕ := m ^ k

/-- The number of 4-letter words that can be made from 5 letters (A, B, C, D, E) with repetition allowed and using E at least once -/
def words_with_e : ℕ := total_words - words_without_e

theorem count_words_with_e : words_with_e = 369 := by
  sorry

end NUMINAMATH_CALUDE_count_words_with_e_l3710_371040


namespace NUMINAMATH_CALUDE_alice_wins_iff_zero_l3710_371061

/-- Alice's winning condition in the quadratic equation game -/
theorem alice_wins_iff_zero (a b c : ℝ) : 
  (∀ d : ℝ, ¬(∃ x y : ℝ, x ≠ y ∧ 
    ((a + d) * x^2 + (b + d) * x + (c + d) = 0) ∧ 
    ((a + d) * y^2 + (b + d) * y + (c + d) = 0)))
  ↔ 
  (a = 0 ∧ b = 0 ∧ c = 0) :=
sorry

end NUMINAMATH_CALUDE_alice_wins_iff_zero_l3710_371061


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3710_371097

theorem min_value_trig_expression (x : ℝ) :
  (Real.sin x)^8 + 16 * (Real.cos x)^8 + 1 ≥ 
  4.7692 * ((Real.sin x)^6 + 4 * (Real.cos x)^6 + 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3710_371097


namespace NUMINAMATH_CALUDE_sphere_intersection_l3710_371043

/-- Sphere intersection problem -/
theorem sphere_intersection (center : ℝ × ℝ × ℝ) (R : ℝ) :
  let (x₀, y₀, z₀) := center
  -- Conditions
  (x₀ = 3 ∧ y₀ = -2 ∧ z₀ = 5) →  -- Sphere center
  (R^2 = 29) →  -- Sphere radius
  -- xy-plane intersection
  ((3 - x₀)^2 + (-2 - y₀)^2 = 2^2) →
  -- yz-plane intersection
  ((0 - x₀)^2 + (5 - z₀)^2 = 3^2) →
  -- xz-plane intersection
  (∃ (x z : ℝ), (x - x₀)^2 + (z - z₀)^2 = 8 ∧ z = -x + 3) →
  -- Conclusion
  (3^2 = 3^2 ∧ 8 = (2 * Real.sqrt 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_sphere_intersection_l3710_371043


namespace NUMINAMATH_CALUDE_max_lcm_20_and_others_l3710_371029

theorem max_lcm_20_and_others : 
  let lcm_list := [Nat.lcm 20 2, Nat.lcm 20 4, Nat.lcm 20 6, Nat.lcm 20 8, Nat.lcm 20 10, Nat.lcm 20 12]
  List.maximum lcm_list = some 60 := by sorry

end NUMINAMATH_CALUDE_max_lcm_20_and_others_l3710_371029


namespace NUMINAMATH_CALUDE_find_b_l3710_371088

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

end NUMINAMATH_CALUDE_find_b_l3710_371088


namespace NUMINAMATH_CALUDE_wire_length_for_square_field_l3710_371058

-- Define the area of the square field
def field_area : ℝ := 53824

-- Define the number of times the wire goes around the field
def num_rounds : ℕ := 10

-- Theorem statement
theorem wire_length_for_square_field :
  ∃ (side_length : ℝ),
    side_length * side_length = field_area ∧
    (4 * side_length * num_rounds : ℝ) = 9280 :=
by sorry

end NUMINAMATH_CALUDE_wire_length_for_square_field_l3710_371058


namespace NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_l3710_371087

theorem smallest_positive_integer_3003m_55555n :
  ∃ (k : ℕ), k > 0 ∧ (∀ (j : ℕ), j > 0 → (∃ (m n : ℤ), j = 3003 * m + 55555 * n) → k ≤ j) ∧
  (∃ (m n : ℤ), k = 3003 * m + 55555 * n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_l3710_371087


namespace NUMINAMATH_CALUDE_a_squared_gt_b_squared_sufficient_not_necessary_l3710_371039

theorem a_squared_gt_b_squared_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a^2 > b^2 → abs a > b) ∧
  (∃ a b : ℝ, abs a > b ∧ a^2 ≤ b^2) :=
by sorry

end NUMINAMATH_CALUDE_a_squared_gt_b_squared_sufficient_not_necessary_l3710_371039


namespace NUMINAMATH_CALUDE_total_tickets_sold_l3710_371093

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

end NUMINAMATH_CALUDE_total_tickets_sold_l3710_371093


namespace NUMINAMATH_CALUDE_math_competition_solution_l3710_371034

/-- Represents the number of contestants from each school -/
structure ContestantCounts where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ

/-- The conditions of the math competition -/
def ValidContestantCounts (counts : ContestantCounts) : Prop :=
  counts.A + counts.B = 16 ∧
  counts.B + counts.C = 20 ∧
  counts.C + counts.D = 34 ∧
  counts.A < counts.B ∧
  counts.B < counts.C ∧
  counts.C < counts.D

/-- The theorem to prove -/
theorem math_competition_solution :
  ∃ (counts : ContestantCounts), ValidContestantCounts counts ∧
    counts.A = 7 ∧ counts.B = 9 ∧ counts.C = 11 ∧ counts.D = 23 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_solution_l3710_371034


namespace NUMINAMATH_CALUDE_power_product_equals_negative_one_l3710_371078

theorem power_product_equals_negative_one : 
  (4 : ℝ)^7 * (-0.25 : ℝ)^7 = -1 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_negative_one_l3710_371078


namespace NUMINAMATH_CALUDE_high_sulfur_oil_count_l3710_371099

/-- Represents the properties of an oil sample set -/
structure OilSampleSet where
  total_samples : Nat
  heavy_oil_prob : Rat
  light_low_sulfur_prob : Rat

/-- Theorem stating the number of high-sulfur oil samples in a given set -/
theorem high_sulfur_oil_count (s : OilSampleSet)
  (h1 : s.total_samples % 7 = 0)
  (h2 : s.total_samples ≤ 100 ∧ ∀ n, n % 7 = 0 → n ≤ 100 → s.total_samples ≥ n)
  (h3 : s.heavy_oil_prob = 1 / 7)
  (h4 : s.light_low_sulfur_prob = 9 / 14) :
  (s.total_samples : Rat) * s.heavy_oil_prob +
  (s.total_samples : Rat) * (1 - s.heavy_oil_prob) * (1 - s.light_low_sulfur_prob) = 44 := by
  sorry

end NUMINAMATH_CALUDE_high_sulfur_oil_count_l3710_371099


namespace NUMINAMATH_CALUDE_common_area_is_32_l3710_371037

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


end NUMINAMATH_CALUDE_common_area_is_32_l3710_371037


namespace NUMINAMATH_CALUDE_no_solutions_for_cos_and_odd_multiples_of_90_l3710_371083

theorem no_solutions_for_cos_and_odd_multiples_of_90 :
  ¬ ∃ x : ℝ, 0 ≤ x ∧ x < 720 ∧ Real.cos (x * π / 180) = -0.6 ∧ ∃ n : ℕ, x = (2 * n + 1) * 90 :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_for_cos_and_odd_multiples_of_90_l3710_371083


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3710_371052

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  S seq 3 = seq.a 2 + 10 * seq.a 1 →
  seq.a 5 = 9 →
  seq.a 1 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3710_371052


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l3710_371015

/-- Represents a cube with diagonal stripes on each face --/
structure StripedCube where
  faces : Fin 6 → Bool  -- True for one diagonal orientation, False for the other

/-- The probability of a continuous stripe loop on a cube --/
def probability_continuous_loop : ℚ :=
  2 / 64

theorem continuous_stripe_probability :
  probability_continuous_loop = 1 / 32 := by
  sorry

#check continuous_stripe_probability

end NUMINAMATH_CALUDE_continuous_stripe_probability_l3710_371015


namespace NUMINAMATH_CALUDE_correct_average_l3710_371002

theorem correct_average (n : ℕ) (initial_avg : ℚ) (correction1 : ℚ) (wrong2 : ℚ) (correct2 : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  correction1 = 19 →
  wrong2 = 13 →
  correct2 = 31 →
  let initial_sum := n * initial_avg
  let corrected_sum := initial_sum - correction1 - wrong2 + correct2
  let corrected_avg := corrected_sum / n
  corrected_avg = 40.1 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l3710_371002


namespace NUMINAMATH_CALUDE_apple_selling_price_l3710_371053

theorem apple_selling_price (cost_price : ℝ) (loss_fraction : ℝ) (selling_price : ℝ) : 
  cost_price = 21 →
  loss_fraction = 1/6 →
  selling_price = cost_price * (1 - loss_fraction) →
  selling_price = 17.50 := by
sorry

end NUMINAMATH_CALUDE_apple_selling_price_l3710_371053


namespace NUMINAMATH_CALUDE_third_team_pies_l3710_371021

/-- Given a catering job requiring 750 mini meat pies to be made by 3 teams,
    where the first team made 235 pies and the second team made 275 pies,
    prove that the third team should make 240 pies. -/
theorem third_team_pies (total : ℕ) (teams : ℕ) (first : ℕ) (second : ℕ) 
    (h1 : total = 750)
    (h2 : teams = 3)
    (h3 : first = 235)
    (h4 : second = 275) :
  total - first - second = 240 := by
  sorry

end NUMINAMATH_CALUDE_third_team_pies_l3710_371021


namespace NUMINAMATH_CALUDE_greater_than_theorem_l3710_371089

theorem greater_than_theorem (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : (a - b) * (b - c) * (c - a) > 0) : 
  a > c := by sorry

end NUMINAMATH_CALUDE_greater_than_theorem_l3710_371089


namespace NUMINAMATH_CALUDE_area_inside_rectangle_outside_circles_l3710_371070

/-- The area of the region inside a rectangle but outside three quarter circles --/
theorem area_inside_rectangle_outside_circles (π : ℝ) :
  let rectangle_area : ℝ := 4 * 6
  let circle_e_area : ℝ := π * 2^2
  let circle_f_area : ℝ := π * 3^2
  let circle_g_area : ℝ := π * 4^2
  let quarter_circles_area : ℝ := (circle_e_area + circle_f_area + circle_g_area) / 4
  rectangle_area - quarter_circles_area = 24 - (29 * π) / 4 :=
by sorry

end NUMINAMATH_CALUDE_area_inside_rectangle_outside_circles_l3710_371070


namespace NUMINAMATH_CALUDE_tom_average_speed_l3710_371024

theorem tom_average_speed (karen_speed : ℝ) (karen_delay : ℝ) (win_margin : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 →
  karen_delay = 4 / 60 →
  win_margin = 4 →
  tom_distance = 24 →
  ∃ (tom_speed : ℝ), tom_speed = 300 / 7 ∧
    karen_speed * (tom_distance / karen_speed) = 
    tom_speed * (tom_distance / karen_speed + karen_delay) - win_margin :=
by sorry

end NUMINAMATH_CALUDE_tom_average_speed_l3710_371024


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l3710_371031

def U : Finset Int := {-1, 0, 1, 2, 3}
def A : Finset Int := {-1, 0}
def B : Finset Int := {0, 1, 2}

theorem complement_intersection_equals_set : (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l3710_371031


namespace NUMINAMATH_CALUDE_max_servings_emily_l3710_371009

/-- Represents the recipe requirements for 4 servings --/
structure Recipe :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (berries : ℕ)
  (almond_milk : ℕ)

/-- Represents Emily's available ingredients --/
structure Available :=
  (bananas : ℕ)
  (yogurt : ℕ)
  (berries : ℕ)
  (almond_milk : ℕ)

/-- Calculates the maximum number of servings possible --/
def max_servings (recipe : Recipe) (available : Available) : ℕ :=
  min
    (available.bananas * 4 / recipe.bananas)
    (min
      (available.yogurt * 4 / recipe.yogurt)
      (min
        (available.berries * 4 / recipe.berries)
        (available.almond_milk * 4 / recipe.almond_milk)))

/-- The theorem to be proved --/
theorem max_servings_emily :
  let recipe := Recipe.mk 3 2 1 1
  let available := Available.mk 9 5 3 4
  max_servings recipe available = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_emily_l3710_371009


namespace NUMINAMATH_CALUDE_solve_for_c_l3710_371091

theorem solve_for_c (h1 : ∀ a b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
                    (h2 : 6 * 15 * c = 4) : c = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l3710_371091


namespace NUMINAMATH_CALUDE_x1_value_l3710_371032

theorem x1_value (x₁ x₂ x₃ : ℝ) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h2 : (1-x₁)^3 + (x₁-x₂)^3 + (x₂-x₃)^3 + x₃^3 = 1/8) : 
  x₁ = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_x1_value_l3710_371032


namespace NUMINAMATH_CALUDE_unique_digit_divisibility_l3710_371067

theorem unique_digit_divisibility : 
  ∃! (A : ℕ), A < 10 ∧ 70 % A = 0 ∧ (546200 + 10 * A + 4) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_divisibility_l3710_371067


namespace NUMINAMATH_CALUDE_third_player_games_l3710_371098

/-- Represents a table tennis game with three players -/
structure TableTennisGame where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The rules of the game ensure that the total games is the sum of games played by any two players -/
def valid_game (g : TableTennisGame) : Prop :=
  g.total_games = g.player1_games + g.player2_games ∧
  g.total_games = g.player1_games + g.player3_games ∧
  g.total_games = g.player2_games + g.player3_games

/-- The theorem to be proved -/
theorem third_player_games (g : TableTennisGame) 
  (h1 : g.player1_games = 10)
  (h2 : g.player2_games = 21)
  (h3 : valid_game g) :
  g.player3_games = 11 := by
  sorry

end NUMINAMATH_CALUDE_third_player_games_l3710_371098


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3710_371017

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 6*x + 5) + (x^2 + 3*x - 18) = (x^2 + 10*x + 64) * (x^2 + 10*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3710_371017


namespace NUMINAMATH_CALUDE_dance_class_permutations_l3710_371059

theorem dance_class_permutations :
  Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_dance_class_permutations_l3710_371059


namespace NUMINAMATH_CALUDE_unity_digit_of_n_l3710_371010

theorem unity_digit_of_n (n : ℕ) (h : 3 * n = 999^1000) : n % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_unity_digit_of_n_l3710_371010


namespace NUMINAMATH_CALUDE_subset_implies_a_value_l3710_371066

theorem subset_implies_a_value (A B : Set ℤ) (a : ℤ) 
  (h1 : A = {0, 1}) 
  (h2 : B = {-1, 0, a+3}) 
  (h3 : A ⊆ B) : 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_value_l3710_371066


namespace NUMINAMATH_CALUDE_seventieth_number_with_remainder_five_seventieth_number_is_557_l3710_371075

theorem seventieth_number_with_remainder_five : ℕ → Prop :=
  fun n => ∃ k : ℕ, n = 8 * k + 5 ∧ n > 0

theorem seventieth_number_is_557 :
  ∃! n : ℕ, seventieth_number_with_remainder_five n ∧ (∃ m : ℕ, m = 70 ∧
    (∀ k < n, seventieth_number_with_remainder_five k →
      (∃ i : ℕ, i < m ∧ (∀ j < k, seventieth_number_with_remainder_five j → ∃ l : ℕ, l < i)))) ∧
  n = 557 :=
by sorry

end NUMINAMATH_CALUDE_seventieth_number_with_remainder_five_seventieth_number_is_557_l3710_371075


namespace NUMINAMATH_CALUDE_prime_property_l3710_371019

theorem prime_property (p : ℕ) : 
  Prime p → (∃ q : ℕ, Prime q ∧ q = 2^(p+1) + p^3 - p^2 - p) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_property_l3710_371019


namespace NUMINAMATH_CALUDE_cost_price_percentage_l3710_371025

theorem cost_price_percentage (selling_price cost_price : ℝ) 
  (h_profit_percent : (selling_price - cost_price) / cost_price = 1/3) :
  cost_price / selling_price = 3/4 := by
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l3710_371025


namespace NUMINAMATH_CALUDE_equation_is_linear_in_two_vars_l3710_371035

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

end NUMINAMATH_CALUDE_equation_is_linear_in_two_vars_l3710_371035


namespace NUMINAMATH_CALUDE_pyramid_volume_l3710_371005

/-- The volume of a pyramid with specific properties -/
theorem pyramid_volume (base_angle : Real) (lateral_edge : Real) (inclination : Real) : 
  base_angle = π/8 →
  lateral_edge = Real.sqrt 6 →
  inclination = 5*π/13 →
  ∃ (volume : Real), 
    volume = Real.sqrt 3 * Real.sin (10*π/13) * Real.cos (5*π/13) ∧
    volume = (1/3) * 
             ((lateral_edge * Real.cos inclination)^2 * Real.sin (2*base_angle)) * 
             (lateral_edge * Real.sin inclination) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3710_371005


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3710_371076

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - t|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 2 x > 2} = {x : ℝ | x < 1/2 ∨ x > 5/2} :=
by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ t ∈ Set.Icc 1 2,
    (∀ x ∈ Set.Icc (-1) 3, ∃ a : ℝ, f t x ≥ a + x) →
    ∃ a : ℝ, a ≤ -1 ∧ ∀ x ∈ Set.Icc (-1) 3, f t x ≥ a + x :=
by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3710_371076


namespace NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l3710_371090

theorem negation_of_or_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l3710_371090


namespace NUMINAMATH_CALUDE_blue_balls_drawn_first_probability_l3710_371022

def num_blue_balls : ℕ := 4
def num_yellow_balls : ℕ := 3
def total_balls : ℕ := num_blue_balls + num_yellow_balls

def favorable_outcomes : ℕ := Nat.choose (total_balls - 1) num_yellow_balls
def total_outcomes : ℕ := Nat.choose total_balls num_yellow_balls

theorem blue_balls_drawn_first_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_drawn_first_probability_l3710_371022


namespace NUMINAMATH_CALUDE_farm_animals_l3710_371095

/-- Given a farm with chickens and buffalos, prove the number of chickens -/
theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) (chicken_legs : ℕ) (buffalo_legs : ℕ) 
  (h_total_animals : total_animals = 13)
  (h_total_legs : total_legs = 44)
  (h_chicken_legs : chicken_legs = 2)
  (h_buffalo_legs : buffalo_legs = 4) :
  ∃ (chickens : ℕ) (buffalos : ℕ),
    chickens + buffalos = total_animals ∧
    chickens * chicken_legs + buffalos * buffalo_legs = total_legs ∧
    chickens = 4 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l3710_371095


namespace NUMINAMATH_CALUDE_smaller_cube_weight_l3710_371028

/-- Represents the weight of a cube given its side length -/
def cube_weight (side_length : ℝ) : ℝ := sorry

theorem smaller_cube_weight :
  let small_side : ℝ := 1
  let large_side : ℝ := 2 * small_side
  let large_weight : ℝ := 56
  cube_weight small_side = 7 ∧ 
  cube_weight large_side = large_weight ∧
  cube_weight large_side = 8 * cube_weight small_side :=
by sorry

end NUMINAMATH_CALUDE_smaller_cube_weight_l3710_371028


namespace NUMINAMATH_CALUDE_min_value_condition_inequality_condition_l3710_371065

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |x + m|

-- Theorem for part 1
theorem min_value_condition (m : ℝ) :
  (∃ (x : ℝ), f x m = 2 ∧ ∀ (y : ℝ), f y m ≥ 2) ↔ (m = 3 ∨ m = -1) :=
sorry

-- Theorem for part 2
theorem inequality_condition (m : ℝ) :
  (∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f x m ≤ 2 * x + 3) ↔ (0 ≤ m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_condition_inequality_condition_l3710_371065


namespace NUMINAMATH_CALUDE_team_ratio_is_correct_l3710_371054

/-- Represents a co-ed softball team -/
structure Team where
  total_players : ℕ
  men : ℕ
  women : ℕ
  h_total : men + women = total_players
  h_ratio : ∀ (group : ℕ), group * 3 ≤ total_players → group * 2 = women - men

/-- The specific team in the problem -/
def problem_team : Team where
  total_players := 25
  men := 8
  women := 17
  h_total := by sorry
  h_ratio := by sorry

theorem team_ratio_is_correct (team : Team) (h : team = problem_team) :
  team.men = 8 ∧ team.women = 17 := by sorry

end NUMINAMATH_CALUDE_team_ratio_is_correct_l3710_371054


namespace NUMINAMATH_CALUDE_field_trip_van_occupancy_l3710_371012

theorem field_trip_van_occupancy :
  let num_vans : ℕ := 2
  let num_buses : ℕ := 3
  let people_per_bus : ℕ := 20
  let total_people : ℕ := 76
  let people_in_vans : ℕ := total_people - (num_buses * people_per_bus)
  people_in_vans / num_vans = 8 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_van_occupancy_l3710_371012


namespace NUMINAMATH_CALUDE_max_speed_theorem_l3710_371096

/-- Represents a set of observations for machine speed and defective items produced. -/
structure Observation where
  speed : ℝ
  defects : ℝ

/-- Calculates the slope of the linear regression line. -/
def calculateSlope (observations : List Observation) : ℝ :=
  sorry

/-- Calculates the y-intercept of the linear regression line. -/
def calculateIntercept (observations : List Observation) (slope : ℝ) : ℝ :=
  sorry

/-- Theorem: The maximum speed at which the machine can operate while producing
    no more than 10 defective items per hour is 15 revolutions per second. -/
theorem max_speed_theorem (observations : List Observation)
    (h1 : observations = [⟨8, 5⟩, ⟨12, 8⟩, ⟨14, 9⟩, ⟨16, 11⟩])
    (h2 : ∀ obs ∈ observations, obs.speed > 0 ∧ obs.defects > 0)
    (h3 : calculateSlope observations > 0) : 
    let slope := calculateSlope observations
    let intercept := calculateIntercept observations slope
    Int.floor ((10 - intercept) / slope) = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_speed_theorem_l3710_371096


namespace NUMINAMATH_CALUDE_power_function_alpha_l3710_371004

/-- Given a power function y = mx^α where m and α are real numbers,
    if the graph passes through the point (8, 1/4), then α equals -2/3. -/
theorem power_function_alpha (m α : ℝ) :
  (∃ (x y : ℝ), x = 8 ∧ y = 1/4 ∧ y = m * x^α) → α = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_alpha_l3710_371004


namespace NUMINAMATH_CALUDE_parallelogram_area_32_14_l3710_371030

/-- The area of a parallelogram given its base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 cm and height 14 cm is 448 square centimeters -/
theorem parallelogram_area_32_14 : parallelogram_area 32 14 = 448 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_32_14_l3710_371030


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3710_371046

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 812) : 
  x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3710_371046


namespace NUMINAMATH_CALUDE_monkey_banana_problem_l3710_371011

/-- The number of monkeys in the initial scenario -/
def initial_monkeys : ℕ := 8

/-- The time taken to eat bananas in minutes -/
def eating_time : ℕ := 8

/-- The number of bananas eaten in the initial scenario -/
def initial_bananas : ℕ := 8

/-- The number of monkeys in the second scenario -/
def second_monkeys : ℕ := 3

/-- The number of bananas eaten in the second scenario -/
def second_bananas : ℕ := 3

theorem monkey_banana_problem :
  (initial_monkeys * eating_time = initial_bananas * eating_time) ∧
  (second_monkeys * eating_time = second_bananas * eating_time) →
  initial_monkeys = initial_bananas :=
by sorry

end NUMINAMATH_CALUDE_monkey_banana_problem_l3710_371011


namespace NUMINAMATH_CALUDE_rectangle_cut_theorem_l3710_371080

/-- Represents a figure cut from the rectangle -/
structure Figure where
  area : ℕ
  perimeter : ℕ

/-- The problem statement -/
theorem rectangle_cut_theorem :
  ∃ (figures : List Figure),
    figures.length = 5 ∧
    (figures.map Figure.area).sum = 30 ∧
    (∀ f ∈ figures, f.perimeter = 2 * f.area) ∧
    (∃ x : ℕ, figures.map Figure.area = [x, x+1, x+2, x+3, x+4]) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_cut_theorem_l3710_371080


namespace NUMINAMATH_CALUDE_total_questions_on_math_test_l3710_371016

/-- The number of questions on a math test -/
def math_test_questions (word_problems subtraction_problems answered_questions blank_questions : ℕ) : Prop :=
  word_problems + subtraction_problems = answered_questions + blank_questions

/-- Theorem: There are 45 questions on the math test -/
theorem total_questions_on_math_test :
  ∃ (word_problems subtraction_problems answered_questions blank_questions : ℕ),
    word_problems = 17 ∧
    subtraction_problems = 28 ∧
    answered_questions = 38 ∧
    blank_questions = 7 ∧
    math_test_questions word_problems subtraction_problems answered_questions blank_questions ∧
    answered_questions + blank_questions = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_total_questions_on_math_test_l3710_371016


namespace NUMINAMATH_CALUDE_jugglers_balls_l3710_371013

theorem jugglers_balls (num_jugglers : ℕ) (total_balls : ℕ) 
  (h1 : num_jugglers = 378) 
  (h2 : total_balls = 2268) : 
  total_balls / num_jugglers = 6 := by
  sorry

end NUMINAMATH_CALUDE_jugglers_balls_l3710_371013


namespace NUMINAMATH_CALUDE_total_fish_count_l3710_371044

-- Define the number of fish for each person
def billy_fish : ℕ := 10
def tony_fish : ℕ := 3 * billy_fish
def sarah_fish : ℕ := tony_fish + 5
def bobby_fish : ℕ := 2 * sarah_fish

-- Define the total number of fish
def total_fish : ℕ := billy_fish + tony_fish + sarah_fish + bobby_fish

-- Theorem statement
theorem total_fish_count : total_fish = 145 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l3710_371044


namespace NUMINAMATH_CALUDE_employment_percentage_l3710_371003

theorem employment_percentage (total_population : ℝ) (employed_population : ℝ) 
  (h1 : employed_population > 0) 
  (h2 : employed_population ≤ total_population)
  (h3 : employed_population * 0.7 + employed_population * 0.3 = employed_population)
  (h4 : employed_population * 0.3 = total_population * 0.21) : 
  employed_population / total_population = 0.7 := by
sorry

end NUMINAMATH_CALUDE_employment_percentage_l3710_371003


namespace NUMINAMATH_CALUDE_cube_side_ratio_l3710_371038

/-- Given two cubes of the same material, this theorem proves that if their weights are in the ratio of 32:4, then their side lengths are in the ratio of 2:1. -/
theorem cube_side_ratio (s₁ s₂ : ℝ) (w₁ w₂ : ℝ) (h₁ : w₁ = 4) (h₂ : w₂ = 32) :
  w₁ * s₂^3 = w₂ * s₁^3 → s₂ / s₁ = 2 := by sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l3710_371038
