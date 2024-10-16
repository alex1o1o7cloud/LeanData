import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_three_squared_l3799_379913

theorem sqrt_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l3799_379913


namespace NUMINAMATH_CALUDE_min_value_3x_minus_2y_l3799_379994

theorem min_value_3x_minus_2y (x y : ℝ) (h : 4 * (x^2 + y^2 + x*y) = 2 * (x + y)) :
  ∃ (m : ℝ), m = -1 ∧ ∀ (a b : ℝ), 4 * (a^2 + b^2 + a*b) = 2 * (a + b) → 3*a - 2*b ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_3x_minus_2y_l3799_379994


namespace NUMINAMATH_CALUDE_set_equals_naturals_l3799_379990

def is_closed_under_multiplication_by_four (X : Set ℕ) : Prop :=
  ∀ x ∈ X, (4 * x) ∈ X

def is_closed_under_floor_sqrt (X : Set ℕ) : Prop :=
  ∀ x ∈ X, Nat.sqrt x ∈ X

theorem set_equals_naturals (X : Set ℕ) 
  (h_nonempty : X.Nonempty)
  (h_mul_four : is_closed_under_multiplication_by_four X)
  (h_floor_sqrt : is_closed_under_floor_sqrt X) : 
  X = Set.univ :=
sorry

end NUMINAMATH_CALUDE_set_equals_naturals_l3799_379990


namespace NUMINAMATH_CALUDE_triangle_ratio_sum_l3799_379911

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_ratio_sum (t : Triangle) (h : t.B = 60) :
  (t.c / (t.a + t.b)) + (t.a / (t.b + t.c)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_sum_l3799_379911


namespace NUMINAMATH_CALUDE_father_and_sons_ages_l3799_379974

/-- Given a father and his three sons, prove that the ages of the middle and oldest sons are 3 and 4 years respectively. -/
theorem father_and_sons_ages (father_age : ℕ) (youngest_son_age : ℕ) (middle_son_age : ℕ) (oldest_son_age : ℕ) :
  father_age = 33 →
  youngest_son_age = 2 →
  father_age + 12 = (youngest_son_age + 12) + (middle_son_age + 12) + (oldest_son_age + 12) →
  (middle_son_age = 3 ∧ oldest_son_age = 4) ∨ (middle_son_age = 4 ∧ oldest_son_age = 3) :=
by sorry

end NUMINAMATH_CALUDE_father_and_sons_ages_l3799_379974


namespace NUMINAMATH_CALUDE_plane_determining_pairs_count_l3799_379977

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  /-- The number of edges in a regular tetrahedron -/
  num_edges : ℕ
  /-- The number of edges that intersect with each edge -/
  intersecting_edges : ℕ
  /-- Property that the number of edges is 6 -/
  edge_count : num_edges = 6
  /-- Property that each edge intersects with 2 other edges -/
  intersect_count : intersecting_edges = 2
  /-- Property that there are no skew edges -/
  no_skew_edges : True

/-- The number of unordered pairs of edges that determine a plane in a regular tetrahedron -/
def plane_determining_pairs (t : RegularTetrahedron) : ℕ :=
  t.num_edges * t.intersecting_edges / 2

/-- Theorem stating that the number of unordered pairs of edges that determine a plane in a regular tetrahedron is 6 -/
theorem plane_determining_pairs_count (t : RegularTetrahedron) :
  plane_determining_pairs t = 6 := by
  sorry

end NUMINAMATH_CALUDE_plane_determining_pairs_count_l3799_379977


namespace NUMINAMATH_CALUDE_polynomial_identity_l3799_379980

theorem polynomial_identity 
  (a b c x : ℝ) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  a^2 * ((x-b)*(x-c)) / ((a-b)*(a-c)) + 
  b^2 * ((x-c)*(x-a)) / ((b-c)*(b-a)) + 
  c^2 * ((x-a)*(x-b)) / ((c-a)*(c-b)) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3799_379980


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3799_379951

def I : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 4, 5}
def B : Set Nat := {1, 4}

theorem intersection_A_complement_B :
  A ∩ (I \ B) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3799_379951


namespace NUMINAMATH_CALUDE_max_comic_books_l3799_379995

/-- The cost function for buying comic books -/
def cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 1.2 * n else 12 + 1.1 * (n - 10)

/-- Jason's budget -/
def budget : ℚ := 15

/-- Predicate to check if a number of books is affordable -/
def is_affordable (n : ℕ) : Prop :=
  cost n ≤ budget

/-- The maximum number of comic books Jason can buy -/
def max_books : ℕ := 12

theorem max_comic_books : 
  (∀ n : ℕ, is_affordable n → n ≤ max_books) ∧ 
  is_affordable max_books :=
sorry

end NUMINAMATH_CALUDE_max_comic_books_l3799_379995


namespace NUMINAMATH_CALUDE_five_cubes_volume_l3799_379989

/-- The volume of a cube with edge length s -/
def cubeVolume (s : ℝ) : ℝ := s ^ 3

/-- The total volume of n cubes, each with edge length s -/
def totalVolume (n : ℕ) (s : ℝ) : ℝ := n * cubeVolume s

/-- Theorem: The total volume of five cubes with edge length 6 feet is 1080 cubic feet -/
theorem five_cubes_volume : totalVolume 5 6 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_five_cubes_volume_l3799_379989


namespace NUMINAMATH_CALUDE_sum_arithmetic_series_base8_l3799_379954

/-- Conversion from base 8 to base 10 -/
def base8ToBase10 (x : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 8 -/
def base10ToBase8 (x : ℕ) : ℕ := sorry

/-- Sum of arithmetic series in base 8 -/
def sumArithmeticSeriesBase8 (n a l : ℕ) : ℕ :=
  base10ToBase8 ((n * (base8ToBase10 a + base8ToBase10 l)) / 2)

theorem sum_arithmetic_series_base8 :
  sumArithmeticSeriesBase8 36 1 36 = 1056 := by sorry

end NUMINAMATH_CALUDE_sum_arithmetic_series_base8_l3799_379954


namespace NUMINAMATH_CALUDE_amy_game_score_l3799_379978

theorem amy_game_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ) : 
  points_per_treasure = 4 →
  treasures_level1 = 6 →
  treasures_level2 = 2 →
  points_per_treasure * treasures_level1 + points_per_treasure * treasures_level2 = 32 := by
sorry

end NUMINAMATH_CALUDE_amy_game_score_l3799_379978


namespace NUMINAMATH_CALUDE_alla_boris_meeting_l3799_379953

/-- Represents the meeting point of Alla and Boris along a straight alley with lampposts -/
def meeting_point (total_lampposts : ℕ) (alla_position : ℕ) (boris_position : ℕ) : ℕ :=
  alla_position + (total_lampposts - alla_position - boris_position + 1) / 2

/-- Theorem stating that Alla and Boris meet at lamppost 163 under the given conditions -/
theorem alla_boris_meeting :
  let total_lampposts : ℕ := 400
  let alla_start : ℕ := 1
  let boris_start : ℕ := total_lampposts
  let alla_position : ℕ := 55
  let boris_position : ℕ := 321
  meeting_point total_lampposts alla_position boris_position = 163 :=
by sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_l3799_379953


namespace NUMINAMATH_CALUDE_A_share_is_175_l3799_379907

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_share_A (initial_A initial_B change_A change_B total_profit : ℚ) : ℚ :=
  let investment_months_A := initial_A * 8 + (initial_A + change_A) * 4
  let investment_months_B := initial_B * 8 + (initial_B + change_B) * 4
  let total_investment_months := investment_months_A + investment_months_B
  (investment_months_A / total_investment_months) * total_profit

/-- Theorem stating that A's share of the profit is 175 given the specified conditions --/
theorem A_share_is_175 :
  calculate_share_A 2000 4000 (-1000) 1000 630 = 175 := by
  sorry

end NUMINAMATH_CALUDE_A_share_is_175_l3799_379907


namespace NUMINAMATH_CALUDE_right_triangle_condition_l3799_379998

theorem right_triangle_condition (a b : ℝ) (α β : Real) :
  a > 0 → b > 0 →
  a ≠ b →
  (a / b) ^ 2 = (Real.tan α) / (Real.tan β) →
  α + β = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l3799_379998


namespace NUMINAMATH_CALUDE_grocery_store_salary_l3799_379901

/-- Calculates the total daily salary of all employees in a grocery store -/
def total_daily_salary (owner_salary : ℕ) (manager_salary : ℕ) (cashier_salary : ℕ) 
  (clerk_salary : ℕ) (bagger_salary : ℕ) (num_owners : ℕ) (num_managers : ℕ) 
  (num_cashiers : ℕ) (num_clerks : ℕ) (num_baggers : ℕ) : ℕ :=
  owner_salary * num_owners + manager_salary * num_managers + 
  cashier_salary * num_cashiers + clerk_salary * num_clerks + 
  bagger_salary * num_baggers

theorem grocery_store_salary : 
  total_daily_salary 20 15 10 5 3 1 3 5 7 9 = 177 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_salary_l3799_379901


namespace NUMINAMATH_CALUDE_inequality_proof_l3799_379920

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_condition : x * y + y * z + z * x + 2 * x * y * z = 1) : 
  4 * x + y + z ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3799_379920


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3799_379940

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3799_379940


namespace NUMINAMATH_CALUDE_book_cost_problem_l3799_379972

theorem book_cost_problem : ∃ (s b c : ℕ+), 
  s > 18 ∧ 
  b > 1 ∧ 
  c > b ∧ 
  s * b * c = 3203 ∧ 
  c = 11 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l3799_379972


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3799_379945

theorem fraction_multiplication : (3 : ℚ) / 4 * 5 / 7 * 11 / 13 = 165 / 364 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3799_379945


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_equal_cost_3999_largest_equal_cost_is_3999_l3799_379955

/-- Function to calculate the sum of squares of decimal digits -/
def sumSquaresDecimal (n : Nat) : Nat :=
  sorry

/-- Function to calculate the sum of squares of binary digits -/
def sumSquaresBinary (n : Nat) : Nat :=
  sorry

/-- Check if a number has equal costs for both options -/
def hasEqualCosts (n : Nat) : Prop :=
  sumSquaresDecimal n = sumSquaresBinary n

theorem largest_equal_cost_number :
  ∀ n : Nat, n < 5000 → n > 3999 → ¬(hasEqualCosts n) :=
by sorry

theorem equal_cost_3999 : hasEqualCosts 3999 :=
by sorry

theorem largest_equal_cost_is_3999 :
  ∃ n : Nat, n < 5000 ∧ hasEqualCosts n ∧ ∀ m : Nat, m < 5000 → m > n → ¬(hasEqualCosts m) :=
by sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_equal_cost_3999_largest_equal_cost_is_3999_l3799_379955


namespace NUMINAMATH_CALUDE_tan_seven_pi_fourths_l3799_379926

theorem tan_seven_pi_fourths : Real.tan (7 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_fourths_l3799_379926


namespace NUMINAMATH_CALUDE_point_on_inverse_proportion_in_first_quadrant_l3799_379934

/-- Given that point M(3,m) lies on the graph of y = 6/x, prove that M is in the first quadrant -/
theorem point_on_inverse_proportion_in_first_quadrant (m : ℝ) : 
  m = 6 / 3 → m > 0 := by sorry

end NUMINAMATH_CALUDE_point_on_inverse_proportion_in_first_quadrant_l3799_379934


namespace NUMINAMATH_CALUDE_first_earthquake_collapse_l3799_379916

/-- Represents the number of buildings collapsed in the first earthquake -/
def first_collapse : ℕ := sorry

/-- Represents the total number of collapsed buildings after four earthquakes -/
def total_collapse : ℕ := 60

/-- Theorem stating that the number of buildings collapsed in the first earthquake is 4 -/
theorem first_earthquake_collapse : 
  (first_collapse + 2 * first_collapse + 4 * first_collapse + 8 * first_collapse = total_collapse) → 
  first_collapse = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_earthquake_collapse_l3799_379916


namespace NUMINAMATH_CALUDE_highest_power_of_two_dividing_P_l3799_379996

def P : ℕ → ℕ := λ n => (List.range n).foldl (λ acc i => acc * (3^(i+1) + 1)) 1

theorem highest_power_of_two_dividing_P :
  ∃ (k : ℕ), (2^3030 ∣ P 2020) ∧ ¬(2^(3030 + 1) ∣ P 2020) := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_two_dividing_P_l3799_379996


namespace NUMINAMATH_CALUDE_impossible_all_defective_l3799_379997

theorem impossible_all_defective (total : Nat) (defective : Nat) (selected : Nat)
  (h1 : total = 10)
  (h2 : defective = 2)
  (h3 : selected = 3)
  (h4 : defective ≤ total)
  (h5 : selected ≤ total) :
  Nat.choose defective selected / Nat.choose total selected = 0 :=
sorry

end NUMINAMATH_CALUDE_impossible_all_defective_l3799_379997


namespace NUMINAMATH_CALUDE_complex_number_solution_l3799_379964

theorem complex_number_solution (z : ℂ) (h : z + Complex.abs z = 2 + 8 * I) : z = -15 + 8 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_solution_l3799_379964


namespace NUMINAMATH_CALUDE_bus_journey_distance_l3799_379963

/-- Represents the bus journey with an obstruction --/
structure BusJourney where
  initialSpeed : ℝ
  totalDistance : ℝ
  obstructionTime : ℝ
  delayTime : ℝ
  speedReductionFactor : ℝ
  lateArrivalTime : ℝ
  alternativeObstructionDistance : ℝ
  alternativeLateArrivalTime : ℝ

/-- Theorem stating that given the conditions, the total distance of the journey is 570 miles --/
theorem bus_journey_distance (j : BusJourney) 
  (h1 : j.obstructionTime = 2)
  (h2 : j.delayTime = 2/3)
  (h3 : j.speedReductionFactor = 5/6)
  (h4 : j.lateArrivalTime = 2.75)
  (h5 : j.alternativeObstructionDistance = 50)
  (h6 : j.alternativeLateArrivalTime = 2 + 1/3)
  : j.totalDistance = 570 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_distance_l3799_379963


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3799_379925

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_prod : x * y = 16) (h_recip : 1 / x = 3 / y) : 
  x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3799_379925


namespace NUMINAMATH_CALUDE_min_value_x2_plus_2y2_l3799_379956

theorem min_value_x2_plus_2y2 (x y : ℝ) (h : x^2 - x*y + y^2 = 1) :
  ∃ (m : ℝ), m = (6 - 2*Real.sqrt 3) / 3 ∧ ∀ (a b : ℝ), a^2 - a*b + b^2 = 1 → x^2 + 2*y^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_x2_plus_2y2_l3799_379956


namespace NUMINAMATH_CALUDE_f_properties_l3799_379918

def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

theorem f_properties (a : ℝ) :
  (∀ x, f a x = f a (-x) ↔ a = 0) ∧
  (∀ x, f a x ≥ 
    if a ≤ -1/2 then 3/4 - a
    else if a ≤ 1/2 then a^2 + 1
    else 3/4 + a) ∧
  (∃ x, f a x = 
    if a ≤ -1/2 then 3/4 - a
    else if a ≤ 1/2 then a^2 + 1
    else 3/4 + a) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3799_379918


namespace NUMINAMATH_CALUDE_tan_theta_value_l3799_379929

theorem tan_theta_value (θ : Real) (a : Real) 
  (h1 : (4, a) ∈ {p : ℝ × ℝ | ∃ (r : ℝ), p.1 = r * Real.cos θ ∧ p.2 = r * Real.sin θ})
  (h2 : Real.sin (θ - π) = 3/5) : 
  Real.tan θ = -3/4 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l3799_379929


namespace NUMINAMATH_CALUDE_rand_code_is_1236_l3799_379933

/-- Represents a coding system for words -/
structure CodeSystem where
  range_code : Nat
  random_code : Nat

/-- Extracts the code for a given letter based on its position in a word -/
def extract_code (n : Nat) (code : Nat) : Nat :=
  (code / (10 ^ (5 - n))) % 10

/-- Determines the code for "rand" based on the given coding system -/
def rand_code (cs : CodeSystem) : Nat :=
  let r := extract_code 1 cs.range_code
  let a := extract_code 2 cs.range_code
  let n := extract_code 3 cs.range_code
  let d := extract_code 4 cs.random_code
  r * 1000 + a * 100 + n * 10 + d

/-- Theorem stating that the code for "rand" is 1236 given the specified coding system -/
theorem rand_code_is_1236 (cs : CodeSystem) 
    (h1 : cs.range_code = 12345) 
    (h2 : cs.random_code = 123678) : 
  rand_code cs = 1236 := by
  sorry

end NUMINAMATH_CALUDE_rand_code_is_1236_l3799_379933


namespace NUMINAMATH_CALUDE_ali_circles_l3799_379909

theorem ali_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ernie_circles : ℕ) : 
  total_boxes = 80 → 
  ali_boxes_per_circle = 8 → 
  ernie_boxes_per_circle = 10 → 
  ernie_circles = 4 → 
  (total_boxes - ernie_boxes_per_circle * ernie_circles) / ali_boxes_per_circle = 5 := by
sorry

end NUMINAMATH_CALUDE_ali_circles_l3799_379909


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l3799_379965

theorem sin_50_plus_sqrt3_tan_10_equals_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l3799_379965


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l3799_379904

theorem sum_reciprocals_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  2 ≤ (1 / a + 1 / b) ∧ ∀ x ≥ 2, ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ 1 / a + 1 / b = x :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l3799_379904


namespace NUMINAMATH_CALUDE_simplify_expression_l3799_379957

theorem simplify_expression (s t : ℝ) : 105 * s - 37 * s + 18 * t = 68 * s + 18 * t := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3799_379957


namespace NUMINAMATH_CALUDE_only_36_is_perfect_square_l3799_379988

theorem only_36_is_perfect_square : 
  (∃ n : ℤ, n * n = 36) ∧ 
  (∀ m : ℤ, m * m ≠ 32) ∧ 
  (∀ m : ℤ, m * m ≠ 33) ∧ 
  (∀ m : ℤ, m * m ≠ 34) ∧ 
  (∀ m : ℤ, m * m ≠ 35) :=
by sorry

end NUMINAMATH_CALUDE_only_36_is_perfect_square_l3799_379988


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3799_379906

/-- A function f(x) = ax + 3 has a zero point in the interval (-1, 2) -/
def has_zero_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 < x ∧ x < 2 ∧ a * x + 3 = 0

/-- The condition a < -3 is sufficient but not necessary for f(x) = ax + 3 
    to have a zero point in (-1, 2) -/
theorem sufficient_not_necessary_condition :
  (∀ a : ℝ, a < -3 → has_zero_in_interval a) ∧
  ¬(∀ a : ℝ, has_zero_in_interval a → a < -3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3799_379906


namespace NUMINAMATH_CALUDE_concentrate_water_ratio_is_one_to_three_l3799_379910

/-- The ratio of concentrate to water for orange juice -/
def concentrate_to_water_ratio : ℚ := 1 / 3

/-- The number of cans of concentrate used -/
def concentrate_cans : ℕ := 40

/-- The number of cans of water per can of concentrate -/
def water_cans_per_concentrate : ℕ := 3

/-- Theorem: The ratio of cans of concentrate to cans of water is 1:3 -/
theorem concentrate_water_ratio_is_one_to_three :
  concentrate_to_water_ratio = 1 / (water_cans_per_concentrate : ℚ) ∧
  concentrate_to_water_ratio = (concentrate_cans : ℚ) / ((water_cans_per_concentrate * concentrate_cans) : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_concentrate_water_ratio_is_one_to_three_l3799_379910


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l3799_379946

/-- A bag containing red and black balls -/
structure Bag where
  red : ℕ
  black : ℕ

/-- The result of drawing two balls from the bag -/
inductive Draw
  | TwoRed
  | OneRedOneBlack
  | TwoBlack

/-- Define the events -/
def exactlyOneBlack (d : Draw) : Prop :=
  d = Draw.OneRedOneBlack

def exactlyTwoBlack (d : Draw) : Prop :=
  d = Draw.TwoBlack

/-- The probability of a draw given a bag -/
def prob (b : Bag) (d : Draw) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem mutually_exclusive_not_contradictory (b : Bag) 
  (h1 : b.red = 2) (h2 : b.black = 2) : 
  (∀ d, ¬(exactlyOneBlack d ∧ exactlyTwoBlack d)) ∧ 
  (∃ d, exactlyOneBlack d ∨ exactlyTwoBlack d) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l3799_379946


namespace NUMINAMATH_CALUDE_geometric_sequence_term_number_l3799_379943

theorem geometric_sequence_term_number (a₁ q aₙ : ℚ) (n : ℕ) :
  a₁ = 1/2 →
  q = 1/2 →
  aₙ = 1/64 →
  aₙ = a₁ * q^(n - 1) →
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_term_number_l3799_379943


namespace NUMINAMATH_CALUDE_color_preference_theorem_l3799_379922

theorem color_preference_theorem (total_students : ℕ) 
  (blue_percentage : ℚ) (red_percentage : ℚ) :
  total_students = 200 →
  blue_percentage = 30 / 100 →
  red_percentage = 40 / 100 →
  ∃ (blue_students red_students yellow_students : ℕ),
    blue_students = (blue_percentage * total_students).floor ∧
    red_students = (red_percentage * (total_students - blue_students)).floor ∧
    yellow_students = total_students - blue_students - red_students ∧
    blue_students + yellow_students = 144 :=
by sorry

end NUMINAMATH_CALUDE_color_preference_theorem_l3799_379922


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_even_l3799_379903

/-- Sum of first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := 2 * sum_first_n n

/-- Sum of five consecutive even integers -/
def sum_five_consecutive_even (n : ℕ) : ℕ := 5 * n - 20

theorem largest_of_five_consecutive_even :
  ∃ n : ℕ, sum_first_n_even 30 = sum_five_consecutive_even n ∧ n = 190 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_even_l3799_379903


namespace NUMINAMATH_CALUDE_carnival_snack_booth_sales_ratio_l3799_379979

-- Define the constants from the problem
def daily_popcorn_sales : ℚ := 50
def num_days : ℕ := 5
def rent : ℚ := 30
def ingredient_cost : ℚ := 75
def total_earnings : ℚ := 895

-- Define the theorem
theorem carnival_snack_booth_sales_ratio :
  ∃ (daily_cotton_candy_sales : ℚ),
    (daily_cotton_candy_sales * num_days + daily_popcorn_sales * num_days - (rent + ingredient_cost) = total_earnings) ∧
    (daily_cotton_candy_sales / daily_popcorn_sales = 3 / 1) := by
  sorry

end NUMINAMATH_CALUDE_carnival_snack_booth_sales_ratio_l3799_379979


namespace NUMINAMATH_CALUDE_last_digit_congruence_l3799_379905

theorem last_digit_congruence (N : ℕ) : ∃ (a b : ℕ), N = 10 * a + b ∧ b < 10 →
  (N ≡ b [ZMOD 10]) ∧ (N ≡ b [ZMOD 2]) ∧ (N ≡ b [ZMOD 5]) := by
  sorry

end NUMINAMATH_CALUDE_last_digit_congruence_l3799_379905


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l3799_379928

theorem units_digit_of_fraction : ∃ n : ℕ, n % 10 = 4 ∧ (30 * 31 * 32 * 33 * 34) / 400 = n := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l3799_379928


namespace NUMINAMATH_CALUDE_tuition_cost_l3799_379931

/-- Proves that the tuition cost per semester is $22,000 given the specified conditions --/
theorem tuition_cost (T : ℝ) : 
  (T / 2)                     -- Parents' contribution
  + 3000                      -- Scholarship
  + (2 * 3000)                -- Student loan (twice scholarship amount)
  + (200 * 10)                -- Work earnings (200 hours at $10/hour)
  = T                         -- Total equals tuition cost
  → T = 22000 := by
  sorry

end NUMINAMATH_CALUDE_tuition_cost_l3799_379931


namespace NUMINAMATH_CALUDE_range_of_a_l3799_379921

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ [0, 1] ∧ 
    x₀ + (Real.exp 2 - 1) * Real.log a ≥ (2 * a / Real.exp x₀) + Real.exp 2 * x₀ - 2) →
  a ∈ Set.Icc 1 (Real.exp 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3799_379921


namespace NUMINAMATH_CALUDE_max_vector_sum_value_l3799_379941

/-- The maximum value of |OA + OB + OP| given the specified conditions -/
theorem max_vector_sum_value : ∃ (max : ℝ),
  max = 6 ∧
  ∀ (P : ℝ × ℝ),
  (P.1 - 3)^2 + P.2^2 = 1 →
  ‖(1, 0) + (0, 3) + P‖ ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_vector_sum_value_l3799_379941


namespace NUMINAMATH_CALUDE_pseudo_symmetry_point_l3799_379961

noncomputable def f (x : ℝ) : ℝ := x^2 - 6*x + 4*Real.log x

noncomputable def g (x₀ x : ℝ) : ℝ := 
  (2*x₀ + 4/x₀ - 6)*(x - x₀) + x₀^2 - 6*x₀ + 4*Real.log x₀

theorem pseudo_symmetry_point :
  ∃! x₀ : ℝ, x₀ > 0 ∧ 
  ∀ x, x > 0 → x ≠ x₀ → (f x - g x₀ x) / (x - x₀) > 0 :=
sorry

end NUMINAMATH_CALUDE_pseudo_symmetry_point_l3799_379961


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l3799_379914

theorem unique_solution_to_equation :
  ∃! x : ℝ, x ≠ -1 ∧ x ≠ -3 ∧
  (x^3 + 3*x^2 - x) / (x^2 + 4*x + 3) + x = -7 ∧
  x = -5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l3799_379914


namespace NUMINAMATH_CALUDE_twenty_bananas_equal_twelve_pears_l3799_379983

/-- The cost relationship between bananas, apples, and pears at Hector's Healthy Habits -/
structure FruitCosts where
  banana : ℚ
  apple : ℚ
  pear : ℚ
  banana_apple_ratio : 4 * banana = 3 * apple
  apple_pear_ratio : 5 * apple = 4 * pear

/-- Theorem stating that 20 bananas cost the same as 12 pears -/
theorem twenty_bananas_equal_twelve_pears (c : FruitCosts) : 20 * c.banana = 12 * c.pear := by
  sorry

end NUMINAMATH_CALUDE_twenty_bananas_equal_twelve_pears_l3799_379983


namespace NUMINAMATH_CALUDE_range_of_k_l3799_379935

def proposition_p (k : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + k*x + 2*k + 5 ≥ 0

def proposition_q (k : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ x y : ℝ, x^2 / (4-k) + y^2 / (k-1) = 1 ↔ (x/a)^2 + (y/b)^2 = 1

theorem range_of_k (k : ℝ) :
  (proposition_q k ↔ k ∈ Set.Ioo 1 (5/2)) ∧
  ((proposition_p k ∨ proposition_q k) ∧ ¬(proposition_p k ∧ proposition_q k) ↔
   k ∈ Set.Icc (-2) 1 ∪ Set.Icc (5/2) 10) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l3799_379935


namespace NUMINAMATH_CALUDE_rice_dumpling_suitable_for_sampling_only_rice_dumpling_suitable_for_sampling_l3799_379959

/-- Represents an event that could potentially be surveyed. -/
inductive Event
  | AirplaneSecurity
  | SpacecraftInspection
  | TeacherRecruitment
  | RiceDumplingQuality

/-- Characteristics that make an event suitable for sampling survey. -/
structure SamplingSurveyCharacteristics where
  large_population : Bool
  impractical_full_inspection : Bool
  representative_sample_possible : Bool

/-- Defines the characteristics of a sampling survey for each event. -/
def event_characteristics : Event → SamplingSurveyCharacteristics
  | Event.AirplaneSecurity => ⟨false, false, false⟩
  | Event.SpacecraftInspection => ⟨false, false, false⟩
  | Event.TeacherRecruitment => ⟨false, false, false⟩
  | Event.RiceDumplingQuality => ⟨true, true, true⟩

/-- Determines if an event is suitable for a sampling survey based on its characteristics. -/
def is_suitable_for_sampling (e : Event) : Prop :=
  let c := event_characteristics e
  c.large_population ∧ c.impractical_full_inspection ∧ c.representative_sample_possible

/-- Theorem stating that the rice dumpling quality investigation is suitable for a sampling survey. -/
theorem rice_dumpling_suitable_for_sampling :
  is_suitable_for_sampling Event.RiceDumplingQuality :=
by
  sorry

/-- Theorem stating that the rice dumpling quality investigation is the only event suitable for a sampling survey. -/
theorem only_rice_dumpling_suitable_for_sampling :
  ∀ e : Event, is_suitable_for_sampling e ↔ e = Event.RiceDumplingQuality :=
by
  sorry

end NUMINAMATH_CALUDE_rice_dumpling_suitable_for_sampling_only_rice_dumpling_suitable_for_sampling_l3799_379959


namespace NUMINAMATH_CALUDE_probability_in_B_l3799_379912

-- Define set A
def A : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 ≤ 2}

-- Define set B
def B : Set (ℝ × ℝ) := {p ∈ A | p.2 ≤ p.1^2}

-- Define the area function
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem probability_in_B : (area B) / (area A) = 17 / 24 := by sorry

end NUMINAMATH_CALUDE_probability_in_B_l3799_379912


namespace NUMINAMATH_CALUDE_product_and_squared_sum_l3799_379915

theorem product_and_squared_sum (x y : ℝ) 
  (sum_eq : x + y = 60) 
  (diff_eq : x - y = 10) : 
  x * y = 875 ∧ (x + y)^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_product_and_squared_sum_l3799_379915


namespace NUMINAMATH_CALUDE_leo_and_kendra_combined_weight_l3799_379908

/-- The combined weight of Leo and Kendra is 150 pounds -/
theorem leo_and_kendra_combined_weight :
  let leo_weight : ℝ := 86
  let kendra_weight : ℝ := (leo_weight + 10) / 1.5
  leo_weight + kendra_weight = 150 :=
by sorry

end NUMINAMATH_CALUDE_leo_and_kendra_combined_weight_l3799_379908


namespace NUMINAMATH_CALUDE_unique_a_value_l3799_379944

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem unique_a_value : ∃! a : ℝ, A a ∩ B a = {9} := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l3799_379944


namespace NUMINAMATH_CALUDE_common_chord_length_l3799_379984

/-- The length of the common chord of two overlapping circles -/
theorem common_chord_length (r : ℝ) (h : r = 15) : 
  let chord_length := 2 * r * Real.sqrt 3 / 2
  chord_length = 15 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_common_chord_length_l3799_379984


namespace NUMINAMATH_CALUDE_car_trip_duration_l3799_379923

/-- Proves that a car trip with given conditions has a total duration of 15 hours -/
theorem car_trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) : 
  initial_speed = 30 →
  initial_time = 5 →
  additional_speed = 42 →
  average_speed = 38 →
  (initial_speed * initial_time + additional_speed * (15 - initial_time)) / 15 = average_speed :=
by
  sorry

#check car_trip_duration

end NUMINAMATH_CALUDE_car_trip_duration_l3799_379923


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3799_379982

theorem triangle_angle_problem (A B C : ℝ) : 
  A = B + 21 →
  C = B + 36 →
  A + B + C = 180 →
  B = 41 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3799_379982


namespace NUMINAMATH_CALUDE_robotics_club_theorem_l3799_379917

theorem robotics_club_theorem (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ)
  (h_total : total = 75)
  (h_cs : cs = 44)
  (h_elec : elec = 40)
  (h_both : both = 25) :
  total - (cs + elec - both) = 16 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_theorem_l3799_379917


namespace NUMINAMATH_CALUDE_tournament_points_l3799_379968

def number_of_teams : ℕ := 16
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def total_draws : ℕ := 30
def max_losses_per_team : ℕ := 2

def total_games : ℕ := number_of_teams * (number_of_teams - 1) / 2

theorem tournament_points :
  let total_wins : ℕ := total_games - total_draws
  let points_from_wins : ℕ := total_wins * points_for_win
  let points_from_draws : ℕ := total_draws * points_for_draw * 2
  points_from_wins + points_from_draws = 330 :=
by sorry

end NUMINAMATH_CALUDE_tournament_points_l3799_379968


namespace NUMINAMATH_CALUDE_kellys_sister_visit_l3799_379958

def vacation_length : ℕ := 3 * 7

def travel_days : ℕ := 1 + 1 + 2 + 2

def grandparents_days : ℕ := 5

def brother_days : ℕ := 5

theorem kellys_sister_visit (sister_days : ℕ) : 
  sister_days = vacation_length - (travel_days + grandparents_days + brother_days) → 
  sister_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_kellys_sister_visit_l3799_379958


namespace NUMINAMATH_CALUDE_calc_difference_l3799_379950

-- Define the correct calculation (Mark's method)
def correct_calc : ℤ := 12 - (3 + 6)

-- Define the incorrect calculation (Jane's method)
def incorrect_calc : ℤ := 12 - 3 + 6

-- Theorem statement
theorem calc_difference : correct_calc - incorrect_calc = -12 := by
  sorry

end NUMINAMATH_CALUDE_calc_difference_l3799_379950


namespace NUMINAMATH_CALUDE_journey_distance_l3799_379932

theorem journey_distance (total_journey : ℕ) (remaining : ℕ) (driven : ℕ) : 
  total_journey = 1200 → remaining = 277 → driven = total_journey - remaining → driven = 923 := by
sorry

end NUMINAMATH_CALUDE_journey_distance_l3799_379932


namespace NUMINAMATH_CALUDE_second_discount_percentage_l3799_379971

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 350 →
  first_discount = 20 →
  final_price = 266 →
  (original_price * (1 - first_discount / 100) * (1 - (original_price * (1 - first_discount / 100) - final_price) / (original_price * (1 - first_discount / 100))) = final_price) →
  (original_price * (1 - first_discount / 100) - final_price) / (original_price * (1 - first_discount / 100)) * 100 = 5 :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l3799_379971


namespace NUMINAMATH_CALUDE_distance_traveled_l3799_379927

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Given a speed of 100 km/hr and a time of 5 hours, the distance traveled is 500 km -/
theorem distance_traveled (speed : ℝ) (time : ℝ) 
  (h_speed : speed = 100) 
  (h_time : time = 5) : 
  distance speed time = 500 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3799_379927


namespace NUMINAMATH_CALUDE_min_degree_for_connected_system_l3799_379902

/-- A graph representing a road system in a kingdom --/
structure RoadSystem where
  cities : Finset Nat
  roads : Finset (Nat × Nat)
  city_count : cities.card = 8
  road_symmetry : ∀ a b, (a, b) ∈ roads → (b, a) ∈ roads

/-- The maximum number of roads leading out from any city --/
def max_degree (g : RoadSystem) : Nat :=
  g.cities.sup (λ c => (g.roads.filter (λ r => r.1 = c)).card)

/-- A path between two cities with at most one intermediate city --/
def has_short_path (g : RoadSystem) (a b : Nat) : Prop :=
  (a, b) ∈ g.roads ∨ ∃ c, (a, c) ∈ g.roads ∧ (c, b) ∈ g.roads

/-- The property that any two cities are connected by a short path --/
def all_cities_connected (g : RoadSystem) : Prop :=
  ∀ a b, a ∈ g.cities → b ∈ g.cities → a ≠ b → has_short_path g a b

/-- The main theorem: the minimum degree for a connected road system is greater than 2 --/
theorem min_degree_for_connected_system (g : RoadSystem) (h : all_cities_connected g) :
  max_degree g > 2 := by
  sorry


end NUMINAMATH_CALUDE_min_degree_for_connected_system_l3799_379902


namespace NUMINAMATH_CALUDE_chopped_cube_height_l3799_379976

/-- The height of a cube with a chopped corner -/
theorem chopped_cube_height (s : ℝ) (h_s : s = 2) : 
  let diagonal := s * Real.sqrt 3
  let triangle_side := Real.sqrt (2 * s^2)
  let triangle_area := (Real.sqrt 3 / 4) * triangle_side^2
  let pyramid_volume := (1 / 6) * s^3
  let pyramid_height := 3 * pyramid_volume / triangle_area
  s - pyramid_height = (2 * Real.sqrt 3 - 1) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chopped_cube_height_l3799_379976


namespace NUMINAMATH_CALUDE_assembly_line_arrangements_eq_120_l3799_379930

/-- Represents the number of tasks in the assembly line -/
def num_tasks : ℕ := 6

/-- Represents the number of independent arrangements -/
def num_independent_arrangements : ℕ := 5

/-- The number of ways to arrange the assembly line -/
def assembly_line_arrangements : ℕ := Nat.factorial num_independent_arrangements

/-- Theorem stating that the number of ways to arrange the assembly line is 120 -/
theorem assembly_line_arrangements_eq_120 : 
  assembly_line_arrangements = 120 := by sorry

end NUMINAMATH_CALUDE_assembly_line_arrangements_eq_120_l3799_379930


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l3799_379992

theorem smallest_number_of_students (n : ℕ) : 
  (∃ x : ℕ, 
    n = 5 * x + 3 ∧ 
    n > 50 ∧ 
    ∀ m : ℕ, m > 50 → (∃ y : ℕ, m = 5 * y + 3) → m ≥ n) → 
  n = 53 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l3799_379992


namespace NUMINAMATH_CALUDE_angle_measure_l3799_379948

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3799_379948


namespace NUMINAMATH_CALUDE_first_group_factories_l3799_379937

theorem first_group_factories (total : ℕ) (second_group : ℕ) (remaining : ℕ) 
  (h1 : total = 169)
  (h2 : second_group = 52)
  (h3 : remaining = 48) :
  total - second_group - remaining = 69 :=
by sorry

end NUMINAMATH_CALUDE_first_group_factories_l3799_379937


namespace NUMINAMATH_CALUDE_find_a_l3799_379993

-- Define the sets U and A as functions of a
def U (a : ℝ) : Set ℝ := {1, 3*a+5, a^2+1}
def A (a : ℝ) : Set ℝ := {1, a+1}

-- Define the complement of A in U
def C_U_A (a : ℝ) : Set ℝ := U a \ A a

-- Theorem statement
theorem find_a : ∃ a : ℝ, 
  (U a = {1, 3*a+5, a^2+1}) ∧ 
  (A a = {1, a+1}) ∧ 
  (C_U_A a = {5}) ∧ 
  (a = -2) := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3799_379993


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3799_379939

/-- Given a quadratic equation 5x^2 + kx = 8 with one root equal to 2, 
    prove that the other root is -4/5 -/
theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 5 * x^2 + k * x = 8) ∧ (2 : ℝ) ∈ {x : ℝ | 5 * x^2 + k * x = 8} →
  (-4/5 : ℝ) ∈ {x : ℝ | 5 * x^2 + k * x = 8} :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3799_379939


namespace NUMINAMATH_CALUDE_a_4_equals_7_l3799_379985

-- Define the sequence sum function
def S (n : ℕ) : ℤ := n^2 - 1

-- Define the sequence term function
def a (n : ℕ) : ℤ := S n - S (n-1)

-- Theorem statement
theorem a_4_equals_7 : a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_7_l3799_379985


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3799_379938

/-- Simple interest calculation problem -/
theorem simple_interest_problem (P : ℚ) : 
  (P * 4 * 5) / 100 = P - 2000 → P = 2500 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3799_379938


namespace NUMINAMATH_CALUDE_dog_catches_fox_dog_catches_fox_at_120m_l3799_379986

/-- The distance at which a dog catches a fox given their jump lengths and frequencies -/
theorem dog_catches_fox (initial_distance : ℝ) (dog_jump : ℝ) (fox_jump : ℝ) 
  (dog_jumps_per_unit : ℕ) (fox_jumps_per_unit : ℕ) : ℝ :=
  let dog_distance_per_unit := dog_jump * dog_jumps_per_unit
  let fox_distance_per_unit := fox_jump * fox_jumps_per_unit
  let net_gain_per_unit := dog_distance_per_unit - fox_distance_per_unit
  let time_units := initial_distance / net_gain_per_unit
  dog_distance_per_unit * time_units

/-- Proof that the dog catches the fox at 120 meters from the starting point -/
theorem dog_catches_fox_at_120m : 
  dog_catches_fox 30 2 1 2 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_dog_catches_fox_dog_catches_fox_at_120m_l3799_379986


namespace NUMINAMATH_CALUDE_eighth_root_two_power_l3799_379919

theorem eighth_root_two_power (n : ℝ) : (8 : ℝ)^(1/3) = 2^n → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_eighth_root_two_power_l3799_379919


namespace NUMINAMATH_CALUDE_ceilings_left_to_paint_l3799_379967

theorem ceilings_left_to_paint (total : ℕ) (this_week : ℕ) (next_week_fraction : ℚ) : 
  total = 28 → 
  this_week = 12 → 
  next_week_fraction = 1/4 →
  total - (this_week + next_week_fraction * this_week) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ceilings_left_to_paint_l3799_379967


namespace NUMINAMATH_CALUDE_number_of_lineups_l3799_379970

def team_size : ℕ := 15
def lineup_size : ℕ := 5

def cannot_play_together : Prop := true
def at_least_one_must_play : Prop := true

theorem number_of_lineups : 
  ∃ (n : ℕ), n = Nat.choose (team_size - 2) (lineup_size - 1) * 2 + 
             Nat.choose (team_size - 3) (lineup_size - 2) ∧
  n = 1210 := by
  sorry

end NUMINAMATH_CALUDE_number_of_lineups_l3799_379970


namespace NUMINAMATH_CALUDE_puzzle_solution_l3799_379960

-- Define the types of characters
inductive Character
| Human
| Ape

-- Define the types of statements
inductive StatementType
| Truthful
| Lie

-- Define a structure for a person
structure Person where
  species : Character
  statementType : StatementType

-- Define the statements made by A and B
def statement_A (b : Person) (a : Person) : Prop :=
  b.statementType = StatementType.Lie ∧ 
  b.species = Character.Ape ∧ 
  a.species = Character.Human

def statement_B (a : Person) : Prop :=
  a.statementType = StatementType.Truthful

-- Theorem stating the conclusion
theorem puzzle_solution :
  ∃ (a b : Person),
    (statement_A b a = (a.statementType = StatementType.Lie)) ∧
    (statement_B a = (b.statementType = StatementType.Lie)) ∧
    a.species = Character.Ape ∧
    a.statementType = StatementType.Lie ∧
    b.species = Character.Human ∧
    b.statementType = StatementType.Lie :=
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l3799_379960


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3799_379991

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 36 is 8√2 -/
theorem ellipse_foci_distance : 
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 36}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ 
    ∀ p ∈ ellipse, dist p f₁ + dist p f₂ = 2 * 6 ∧
    dist f₁ f₂ = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3799_379991


namespace NUMINAMATH_CALUDE_system_equation_solution_l3799_379966

theorem system_equation_solution :
  ∃ (x y : ℝ),
    (4 * x + y = 15) ∧
    (x + 4 * y = 18) ∧
    (13 * x^2 + 14 * x * y + 13 * y^2 = 438.6) := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l3799_379966


namespace NUMINAMATH_CALUDE_bananas_cantaloupe_eggs_cost_l3799_379947

/-- Represents the cost of groceries with given conditions -/
def grocery_cost (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = 30 ∧
  d = 3 * a ∧
  c = a - b ∧
  e = b + c

/-- Theorem stating the cost of bananas, cantaloupe, and eggs -/
theorem bananas_cantaloupe_eggs_cost (a b c d e : ℝ) :
  grocery_cost a b c d e → b + c + e = 10 := by
  sorry

#check bananas_cantaloupe_eggs_cost

end NUMINAMATH_CALUDE_bananas_cantaloupe_eggs_cost_l3799_379947


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3799_379942

-- Define the slopes of the two lines
def slope1 : ℚ := 1/2
def slope2 (b : ℚ) : ℚ := -b/5

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_b_value :
  perpendicular slope1 (slope2 b) → b = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3799_379942


namespace NUMINAMATH_CALUDE_total_eyes_count_l3799_379962

theorem total_eyes_count (num_boys : ℕ) (eyes_per_boy : ℕ) (h1 : num_boys = 23) (h2 : eyes_per_boy = 2) :
  num_boys * eyes_per_boy = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_eyes_count_l3799_379962


namespace NUMINAMATH_CALUDE_value_added_to_numbers_l3799_379936

theorem value_added_to_numbers (n : ℕ) (initial_avg final_avg x : ℚ) : 
  n = 15 → 
  initial_avg = 40 → 
  final_avg = 51 → 
  (n : ℚ) * initial_avg + n * x = n * final_avg → 
  x = 11 := by sorry

end NUMINAMATH_CALUDE_value_added_to_numbers_l3799_379936


namespace NUMINAMATH_CALUDE_regular_polygon_45_degree_exterior_angle_is_octagon_l3799_379987

/-- A regular polygon with exterior angles of 45° is a regular octagon -/
theorem regular_polygon_45_degree_exterior_angle_is_octagon :
  ∀ (n : ℕ), n > 2 →
  (360 / n : ℚ) = 45 →
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_45_degree_exterior_angle_is_octagon_l3799_379987


namespace NUMINAMATH_CALUDE_sum_of_coefficients_cubic_factorization_l3799_379900

theorem sum_of_coefficients_cubic_factorization :
  ∀ (a b c d e : ℝ),
  (∀ x, 512 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_cubic_factorization_l3799_379900


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3799_379969

/-- Given a circle with equation x^2 + y^2 - 4x = 0, prove that its center is (2, 0) and its radius is 2 -/
theorem circle_center_and_radius :
  let equation := fun (x y : ℝ) => x^2 + y^2 - 4*x
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, equation x y = 0 ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    center = (2, 0) ∧
    radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3799_379969


namespace NUMINAMATH_CALUDE_fraction_inequality_counterexample_l3799_379975

theorem fraction_inequality_counterexample : 
  ∃ (a b c d : ℝ), (a / b > c / d) ∧ (b / a ≥ d / c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_counterexample_l3799_379975


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_under_sqrt_400_l3799_379952

theorem greatest_multiple_of_four_under_sqrt_400 :
  ∀ x : ℕ, 
    x > 0 → 
    (∃ k : ℕ, x = 4 * k) → 
    x^2 < 400 → 
    x ≤ 16 ∧ 
    (∀ y : ℕ, y > 0 → (∃ m : ℕ, y = 4 * m) → y^2 < 400 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_under_sqrt_400_l3799_379952


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3799_379973

theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^3 - 2*b*x^2 + b*x + b^2 - 2 = 0) ↔ (b = 0 ∨ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3799_379973


namespace NUMINAMATH_CALUDE_page_added_thrice_l3799_379949

/-- Given a book with pages numbered from 2 to n, if one page number p
    is added three times instead of once, resulting in a total sum of 4090,
    then p = 43. -/
theorem page_added_thrice (n : ℕ) (p : ℕ) (h1 : n ≥ 2) 
    (h2 : n * (n + 1) / 2 - 1 + 2 * p = 4090) : p = 43 := by
  sorry

end NUMINAMATH_CALUDE_page_added_thrice_l3799_379949


namespace NUMINAMATH_CALUDE_shirts_washed_l3799_379924

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (not_washed : ℕ) 
  (h1 : short_sleeve = 40)
  (h2 : long_sleeve = 23)
  (h3 : not_washed = 34) :
  short_sleeve + long_sleeve - not_washed = 29 := by
  sorry

end NUMINAMATH_CALUDE_shirts_washed_l3799_379924


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_l3799_379981

/-- Represents a point on a hyperbola -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / 4 - y^2 = 1

/-- Theorem: For a hyperbola defined by x^2/4 - y^2 = 1, if the distance from a point
    on the hyperbola to one focus is 12, then the distance to the other focus is either 16 or 8 -/
theorem hyperbola_focus_distance (p : HyperbolaPoint) (d1 : ℝ) (d2 : ℝ)
  (h1 : d1 = 12) -- Distance to one focus is 12
  (h2 : d2 = |d1 - 4| ∨ d2 = |d1 + 4|) -- Distance to other focus based on hyperbola properties
  : d2 = 16 ∨ d2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_focus_distance_l3799_379981


namespace NUMINAMATH_CALUDE_bedroom_set_price_l3799_379999

theorem bedroom_set_price (P : ℝ) : 
  (P * 0.85 * 0.9 - 200 = 1330) → P = 2000 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_set_price_l3799_379999
