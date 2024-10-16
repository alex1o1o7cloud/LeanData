import Mathlib

namespace NUMINAMATH_CALUDE_max_perimeter_right_triangle_l1550_155081

theorem max_perimeter_right_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^2 + b^2 = 36) : 
  a + b + 6 ≤ 6 * Real.sqrt 2 + 6 :=
sorry

end NUMINAMATH_CALUDE_max_perimeter_right_triangle_l1550_155081


namespace NUMINAMATH_CALUDE_sqrt_36_divided_by_itself_is_one_l1550_155049

theorem sqrt_36_divided_by_itself_is_one : 
  (Real.sqrt 36) / (Real.sqrt 36) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_36_divided_by_itself_is_one_l1550_155049


namespace NUMINAMATH_CALUDE_even_sum_odd_vertices_l1550_155031

/-- Represents a country on the spherical map -/
structure Country where
  color : Fin 4  -- 0: red, 1: yellow, 2: blue, 3: green
  vertices : ℕ

/-- Represents the spherical map -/
structure SphericalMap where
  countries : List Country
  neighbor_relation : Country → Country → Prop

/-- The number of countries with odd vertices for a given color -/
def num_odd_vertices (m : SphericalMap) (c : Fin 4) : ℕ :=
  (m.countries.filter (λ country => country.color = c ∧ country.vertices % 2 = 1)).length

theorem even_sum_odd_vertices (m : SphericalMap) :
  (num_odd_vertices m 0 + num_odd_vertices m 2) % 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_even_sum_odd_vertices_l1550_155031


namespace NUMINAMATH_CALUDE_pattern_continuation_l1550_155022

theorem pattern_continuation (h1 : 1 = 6) (h2 : 2 = 12) (h3 : 3 = 18) (h4 : 4 = 24) (h5 : 5 = 30) : 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_pattern_continuation_l1550_155022


namespace NUMINAMATH_CALUDE_cowboy_shortest_path_l1550_155095

/-- The shortest path for a cowboy to travel from his position to a stream and then to his cabin -/
theorem cowboy_shortest_path (cowboy_pos cabin_pos : ℝ × ℝ) (stream_y : ℝ) :
  cowboy_pos = (0, -5) →
  cabin_pos = (6, 4) →
  stream_y = 0 →
  let dist_to_stream := |cowboy_pos.2 - stream_y|
  let dist_stream_to_cabin := Real.sqrt ((cabin_pos.1 - cowboy_pos.1)^2 + (cabin_pos.2 - stream_y)^2)
  dist_to_stream + dist_stream_to_cabin = 5 + 2 * Real.sqrt 58 :=
by sorry

end NUMINAMATH_CALUDE_cowboy_shortest_path_l1550_155095


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l1550_155091

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l1550_155091


namespace NUMINAMATH_CALUDE_profit_maximized_optimal_selling_price_l1550_155045

/-- Profit function given the increase in selling price -/
def profit (x : ℝ) : ℝ := (2 + x) * (200 - 20 * x)

/-- The optimal price increase that maximizes profit -/
def optimal_price_increase : ℝ := 4

/-- The maximum profit achievable -/
def max_profit : ℝ := 720

/-- Theorem stating that the profit function reaches its maximum at the optimal price increase -/
theorem profit_maximized :
  (∀ x : ℝ, profit x ≤ profit optimal_price_increase) ∧
  profit optimal_price_increase = max_profit :=
sorry

/-- The initial selling price -/
def initial_price : ℝ := 10

/-- Theorem stating the optimal selling price -/
theorem optimal_selling_price :
  initial_price + optimal_price_increase = 14 :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_optimal_selling_price_l1550_155045


namespace NUMINAMATH_CALUDE_kelly_baking_powder_l1550_155033

/-- The amount of baking powder Kelly has today in boxes -/
def today_amount : ℝ := 0.3

/-- The difference in baking powder between yesterday and today in boxes -/
def difference : ℝ := 0.1

/-- The amount of baking powder Kelly had yesterday in boxes -/
def yesterday_amount : ℝ := today_amount + difference

theorem kelly_baking_powder : yesterday_amount = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_kelly_baking_powder_l1550_155033


namespace NUMINAMATH_CALUDE_A_intersect_B_l1550_155041

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}

-- Define set B
def B : Set ℝ := {2, 3, 4, 5}

-- Theorem statement
theorem A_intersect_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1550_155041


namespace NUMINAMATH_CALUDE_robotics_club_mentors_average_age_l1550_155007

theorem robotics_club_mentors_average_age 
  (total_members : ℕ) 
  (avg_age_all : ℕ) 
  (num_girls : ℕ) 
  (num_boys : ℕ) 
  (num_mentors : ℕ) 
  (avg_age_girls : ℕ) 
  (avg_age_boys : ℕ) 
  (h1 : total_members = 50)
  (h2 : avg_age_all = 20)
  (h3 : num_girls = 25)
  (h4 : num_boys = 20)
  (h5 : num_mentors = 5)
  (h6 : avg_age_girls = 18)
  (h7 : avg_age_boys = 19)
  (h8 : total_members = num_girls + num_boys + num_mentors) :
  (total_members * avg_age_all - num_girls * avg_age_girls - num_boys * avg_age_boys) / num_mentors = 34 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_mentors_average_age_l1550_155007


namespace NUMINAMATH_CALUDE_rational_division_and_linear_combination_l1550_155021

theorem rational_division_and_linear_combination (m a b c d k : ℤ) : 
  (∀ (x : ℤ), (x ∣ 5*m + 6 ∧ x ∣ 8*m + 7) ↔ (x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13)) ∧
  ((k ∣ a*m + b ∧ k ∣ c*m + d) → k ∣ a*d - b*c) := by
  sorry

end NUMINAMATH_CALUDE_rational_division_and_linear_combination_l1550_155021


namespace NUMINAMATH_CALUDE_max_person_money_100_2000_380_l1550_155012

/-- Given a group of people and their money distribution, 
    calculate the maximum amount one person can have. -/
def maxPersonMoney (n : ℕ) (total : ℕ) (maxTen : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum amount one person can have 
    under the given conditions. -/
theorem max_person_money_100_2000_380 : 
  maxPersonMoney 100 2000 380 = 218 := by sorry

end NUMINAMATH_CALUDE_max_person_money_100_2000_380_l1550_155012


namespace NUMINAMATH_CALUDE_john_coffee_consumption_l1550_155036

/-- Represents the number of fluid ounces in a gallon -/
def gallonToOunces : ℚ := 128

/-- Represents the number of fluid ounces in a standard cup -/
def cupToOunces : ℚ := 8

/-- Represents the number of days between John's coffee purchases -/
def purchaseInterval : ℚ := 4

/-- Represents the fraction of a gallon John buys each time -/
def purchaseAmount : ℚ := 1/2

/-- Theorem stating that John drinks 2 cups of coffee per day -/
theorem john_coffee_consumption :
  let cupsPerPurchase := purchaseAmount * gallonToOunces / cupToOunces
  cupsPerPurchase / purchaseInterval = 2 := by sorry

end NUMINAMATH_CALUDE_john_coffee_consumption_l1550_155036


namespace NUMINAMATH_CALUDE_geometric_sequence_min_a5_l1550_155070

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_min_a5 (a : ℕ → ℝ) (h1 : GeometricSequence a) 
    (h2 : ∀ n, a n > 0) (h3 : a 3 - a 1 = 2) :
  ∃ m : ℝ, m = 8 ∧ ∀ x : ℝ, (∃ q : ℝ, q > 0 ∧ x = (2 * q^4) / (q^2 - 1)) → x ≥ m :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_a5_l1550_155070


namespace NUMINAMATH_CALUDE_total_match_sequences_l1550_155009

/-- Represents the number of players in each team -/
def n : ℕ := 7

/-- Calculates the number of possible match sequences for one team winning -/
def sequences_for_one_team_winning : ℕ := Nat.choose (2 * n - 1) (n - 1)

/-- Theorem stating the total number of possible match sequences -/
theorem total_match_sequences : 2 * sequences_for_one_team_winning = 3432 := by
  sorry

end NUMINAMATH_CALUDE_total_match_sequences_l1550_155009


namespace NUMINAMATH_CALUDE_set_operation_result_l1550_155014

def A : Set ℕ := {0, 1, 2, 4, 5, 7}
def B : Set ℕ := {1, 3, 6, 8, 9}
def C : Set ℕ := {3, 7, 8}

theorem set_operation_result : (A ∩ B) ∪ C = {1, 3, 7, 8} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l1550_155014


namespace NUMINAMATH_CALUDE_biology_class_percentage_l1550_155072

theorem biology_class_percentage (total_students : ℕ) (not_enrolled : ℕ) :
  total_students = 880 →
  not_enrolled = 572 →
  (((total_students - not_enrolled : ℚ) / total_students) * 100 : ℚ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_biology_class_percentage_l1550_155072


namespace NUMINAMATH_CALUDE_bear_hunting_problem_l1550_155076

theorem bear_hunting_problem (bear_need : ℕ) (cub_need : ℕ) (num_cubs : ℕ) (animals_per_day : ℕ) : 
  bear_need = 210 →
  cub_need = 35 →
  num_cubs = 4 →
  animals_per_day = 10 →
  (bear_need + cub_need * num_cubs) / 7 / animals_per_day = 5 := by
sorry

end NUMINAMATH_CALUDE_bear_hunting_problem_l1550_155076


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l1550_155028

theorem hexagon_angle_measure :
  ∀ (a b c d e f : ℝ),
    a = 135 ∧ b = 120 ∧ c = 105 ∧ d = 150 ∧ e = 110 →
    (a + b + c + d + e + f = 720) →
    f = 100 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l1550_155028


namespace NUMINAMATH_CALUDE_no_nonzero_integer_solutions_l1550_155069

theorem no_nonzero_integer_solutions :
  ∀ x y z : ℤ, x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_nonzero_integer_solutions_l1550_155069


namespace NUMINAMATH_CALUDE_kays_sibling_age_fraction_l1550_155086

/-- Proves that the fraction of Kay's age relating to the youngest sibling's age is 1/2 -/
theorem kays_sibling_age_fraction :
  ∀ (kay_age youngest_age oldest_age : ℕ) (f : ℚ),
    kay_age = 32 →
    youngest_age = f * kay_age - 5 →
    oldest_age = 4 * youngest_age →
    oldest_age = 44 →
    f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_kays_sibling_age_fraction_l1550_155086


namespace NUMINAMATH_CALUDE_simplify_expression_l1550_155011

theorem simplify_expression (x : ℝ) :
  3 * x^3 + 5 * x + 16 * x^2 + 15 - (7 - 3 * x^3 - 5 * x - 16 * x^2) = 
  6 * x^3 + 32 * x^2 + 10 * x + 8 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1550_155011


namespace NUMINAMATH_CALUDE_five_by_five_decomposition_l1550_155052

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square grid -/
structure Grid where
  size : ℕ

/-- Checks if a list of rectangles can fit in a grid -/
def canFitInGrid (grid : Grid) (rectangles : List Rectangle) : Prop :=
  (rectangles.map (λ r => r.width * r.height)).sum = grid.size * grid.size

/-- Theorem: A 5x5 grid can be decomposed into 1x3 and 1x4 rectangles -/
theorem five_by_five_decomposition :
  ∃ (rectangles : List Rectangle),
    (∀ r ∈ rectangles, r.width = 1 ∧ (r.height = 3 ∨ r.height = 4)) ∧
    canFitInGrid { size := 5 } rectangles :=
  sorry

end NUMINAMATH_CALUDE_five_by_five_decomposition_l1550_155052


namespace NUMINAMATH_CALUDE_digit_sum_at_positions_l1550_155067

def sequence_generator (n : ℕ) : ℕ :=
  (n - 1) % 6 + 1

def remove_nth (n : ℕ) (seq : ℕ → ℕ) : ℕ → ℕ :=
  Function.comp seq (λ m => m + m / (n - 1))

def final_sequence : ℕ → ℕ :=
  remove_nth 7 (remove_nth 5 sequence_generator)

theorem digit_sum_at_positions : 
  final_sequence 3031 + final_sequence 3032 + final_sequence 3033 = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_at_positions_l1550_155067


namespace NUMINAMATH_CALUDE_unread_pages_after_two_weeks_l1550_155061

theorem unread_pages_after_two_weeks (total_pages : ℕ) (pages_per_day : ℕ) (days : ℕ) (unread_pages : ℕ) : 
  total_pages = 200 →
  pages_per_day = 12 →
  days = 14 →
  unread_pages = total_pages - (pages_per_day * days) →
  unread_pages = 32 := by
sorry

end NUMINAMATH_CALUDE_unread_pages_after_two_weeks_l1550_155061


namespace NUMINAMATH_CALUDE_abs_neg_two_eq_two_l1550_155018

theorem abs_neg_two_eq_two : |(-2 : ℝ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_eq_two_l1550_155018


namespace NUMINAMATH_CALUDE_total_sheets_required_l1550_155089

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0 to 9) -/
def digit_count : ℕ := 10

/-- The number of sheets required for one character -/
def sheets_per_char : ℕ := 1

/-- Theorem: The total number of sheets required to write all uppercase and lowercase 
    English alphabets and digits from 0 to 9 is 62 -/
theorem total_sheets_required : 
  sheets_per_char * (2 * alphabet_size + digit_count) = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_sheets_required_l1550_155089


namespace NUMINAMATH_CALUDE_problem_statement_l1550_155026

theorem problem_statement : 
  (∃ x : ℝ, x - 2 > 0) ∧ ¬(∀ x : ℝ, x ≥ 0 → Real.sqrt x < x) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1550_155026


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l1550_155002

def num_islands : ℕ := 7
def num_treasure_islands : ℕ := 4

def prob_treasure : ℚ := 1/5
def prob_trap : ℚ := 1/10
def prob_neither : ℚ := 7/10

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  12005/625000 := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l1550_155002


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l1550_155088

theorem max_sum_on_circle (x y : ℝ) (h : (x - 3)^2 + (y - 3)^2 = 8) :
  ∃ (max : ℝ), ∀ (a b : ℝ), (a - 3)^2 + (b - 3)^2 = 8 → a + b ≤ max ∧ ∃ (u v : ℝ), (u - 3)^2 + (v - 3)^2 = 8 ∧ u + v = max :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l1550_155088


namespace NUMINAMATH_CALUDE_eeyore_triangle_problem_l1550_155099

/-- A type representing a stick with a length -/
structure Stick :=
  (length : ℝ)

/-- A function to check if three sticks can form a triangle -/
def canFormTriangle (s1 s2 s3 : Stick) : Prop :=
  s1.length + s2.length > s3.length ∧
  s1.length + s3.length > s2.length ∧
  s2.length + s3.length > s1.length

/-- A function to split six sticks into two sets of three, with the three shortest in one set -/
def splitSticks (sticks : Fin 6 → Stick) : (Fin 3 → Stick) × (Fin 3 → Stick) :=
  sorry

theorem eeyore_triangle_problem :
  ∃ (sticks : Fin 6 → Stick),
    (∃ (t1 t2 t3 t4 t5 t6 : Fin 6), canFormTriangle (sticks t1) (sticks t2) (sticks t3) ∧
                                    canFormTriangle (sticks t4) (sticks t5) (sticks t6)) ∧
    let (yellow, green) := splitSticks sticks
    ¬(canFormTriangle (yellow 0) (yellow 1) (yellow 2) ∧
      canFormTriangle (green 0) (green 1) (green 2)) :=
  sorry

end NUMINAMATH_CALUDE_eeyore_triangle_problem_l1550_155099


namespace NUMINAMATH_CALUDE_second_polygon_sides_l1550_155078

/-- Given two regular polygons with the same perimeter, where the first polygon has 45 sides
    and a side length three times as long as the second, prove that the second polygon has 135 sides. -/
theorem second_polygon_sides (s : ℝ) (sides_second : ℕ) : 
  s > 0 →  -- Assume positive side length
  45 * (3 * s) = sides_second * s →  -- Same perimeter condition
  sides_second = 135 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l1550_155078


namespace NUMINAMATH_CALUDE_max_distance_sum_l1550_155094

/-- Given m ∈ R, for points A on the line x + my = 0 and B on the line mx - y - m + 3 = 0,
    where these lines intersect at point P, the maximum value of |PA| + |PB| is 2√5. -/
theorem max_distance_sum (m : ℝ) : 
  ∃ (A B P : ℝ × ℝ), 
    (A.1 + m * A.2 = 0) ∧ 
    (m * B.1 - B.2 - m + 3 = 0) ∧ 
    (P.1 + m * P.2 = 0) ∧ 
    (m * P.1 - P.2 - m + 3 = 0) ∧
    (∀ (A' B' : ℝ × ℝ), 
      (A'.1 + m * A'.2 = 0) → 
      (m * B'.1 - B'.2 - m + 3 = 0) → 
      Real.sqrt ((P.1 - A'.1)^2 + (P.2 - A'.2)^2) + Real.sqrt ((P.1 - B'.1)^2 + (P.2 - B'.2)^2) ≤ 2 * Real.sqrt 5) ∧
    (Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_sum_l1550_155094


namespace NUMINAMATH_CALUDE_meeting_handshakes_l1550_155001

/-- The number of people in the meeting -/
def total_people : ℕ := 40

/-- The number of people who know each other -/
def group_a : ℕ := 25

/-- The number of people who don't know anyone -/
def group_b : ℕ := 15

/-- Calculate the number of handshakes between two groups -/
def inter_group_handshakes (g1 g2 : ℕ) : ℕ := g1 * g2

/-- Calculate the number of handshakes within a group where no one knows each other -/
def intra_group_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of handshakes in the meeting -/
def total_handshakes : ℕ := 
  inter_group_handshakes group_a group_b + intra_group_handshakes group_b

theorem meeting_handshakes : 
  total_people = group_a + group_b → total_handshakes = 480 := by
  sorry

end NUMINAMATH_CALUDE_meeting_handshakes_l1550_155001


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l1550_155065

theorem no_function_satisfies_conditions : ¬∃ f : ℝ → ℝ, 
  (∀ x : ℝ, f (x^2) - f x ^ 2 ≥ (1/4 : ℝ)) ∧ 
  Function.Injective f := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l1550_155065


namespace NUMINAMATH_CALUDE_salt_added_amount_l1550_155013

/-- Represents the salt solution problem --/
structure SaltSolution where
  initial_volume : ℝ
  initial_salt_concentration : ℝ
  evaporation_fraction : ℝ
  water_added : ℝ
  final_salt_concentration : ℝ

/-- Calculates the amount of salt added to the solution --/
def salt_added (s : SaltSolution) : ℝ :=
  let initial_salt := s.initial_volume * s.initial_salt_concentration
  let water_evaporated := s.initial_volume * s.evaporation_fraction
  let remaining_volume := s.initial_volume - water_evaporated
  let new_volume := remaining_volume + s.water_added
  let final_salt := new_volume * s.final_salt_concentration
  final_salt - initial_salt

/-- The theorem stating the amount of salt added --/
theorem salt_added_amount (s : SaltSolution) 
  (h1 : s.initial_volume = 149.99999999999994)
  (h2 : s.initial_salt_concentration = 0.20)
  (h3 : s.evaporation_fraction = 0.25)
  (h4 : s.water_added = 10)
  (h5 : s.final_salt_concentration = 1/3) :
  ∃ ε > 0, |salt_added s - 10.83| < ε :=
sorry

end NUMINAMATH_CALUDE_salt_added_amount_l1550_155013


namespace NUMINAMATH_CALUDE_max_distance_unit_circle_l1550_155029

/-- The maximum distance between any two points on the unit circle is 2 -/
theorem max_distance_unit_circle : 
  ∀ (α β : ℝ), 
  let P := (Real.cos α, Real.sin α)
  let Q := (Real.cos β, Real.sin β)
  ∃ (maxDist : ℝ), maxDist = 2 ∧ 
    ∀ (α' β' : ℝ), 
    let P' := (Real.cos α', Real.sin α')
    let Q' := (Real.cos β', Real.sin β')
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≤ maxDist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_unit_circle_l1550_155029


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1550_155092

theorem inequality_solution_set (x : ℝ) :
  (x + 3) * (x - 2) < 0 ↔ -3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1550_155092


namespace NUMINAMATH_CALUDE_x_zero_value_l1550_155006

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem x_zero_value (x₀ : ℝ) (h : (deriv f) x₀ = 3) :
  x₀ = 1 ∨ x₀ = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_zero_value_l1550_155006


namespace NUMINAMATH_CALUDE_reflection_line_l1550_155050

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (10, 9), then m + b = 38/3 -/
theorem reflection_line (m b : ℝ) : 
  (∃ (x y : ℝ), x = 10 ∧ y = 9 ∧ 
    (x - 2)^2 + (y - 3)^2 = ((x - 2) * m + (y - 3))^2 / (m^2 + 1) ∧
    y - 3 = -m * (x - 2)) →
  m + b = 38/3 :=
by sorry

end NUMINAMATH_CALUDE_reflection_line_l1550_155050


namespace NUMINAMATH_CALUDE_shannon_bracelets_l1550_155027

/-- Given Shannon has 48 heart-shaped stones and each bracelet requires 8 stones,
    prove that she can make 6 bracelets. -/
theorem shannon_bracelets :
  let total_stones : ℕ := 48
  let stones_per_bracelet : ℕ := 8
  let num_bracelets : ℕ := total_stones / stones_per_bracelet
  num_bracelets = 6 := by
sorry

end NUMINAMATH_CALUDE_shannon_bracelets_l1550_155027


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l1550_155020

/-- The distance between two people walking in opposite directions for a given time -/
def distance_apart (maya_speed : ℚ) (lucas_speed : ℚ) (time : ℚ) : ℚ :=
  maya_speed * time + lucas_speed * time

/-- Theorem stating the distance apart after 2 hours -/
theorem distance_after_two_hours :
  let maya_speed : ℚ := 1 / 20 -- miles per minute
  let lucas_speed : ℚ := 3 / 40 -- miles per minute
  let time : ℚ := 120 -- 2 hours in minutes
  distance_apart maya_speed lucas_speed time = 15 := by
  sorry

#eval distance_apart (1/20) (3/40) 120

end NUMINAMATH_CALUDE_distance_after_two_hours_l1550_155020


namespace NUMINAMATH_CALUDE_monika_total_expense_l1550_155068

def mall_expense : ℝ := 250
def movie_cost : ℝ := 24
def movie_count : ℕ := 3
def bean_bag_cost : ℝ := 1.25
def bean_bag_count : ℕ := 20

theorem monika_total_expense : 
  mall_expense + movie_cost * movie_count + bean_bag_cost * bean_bag_count = 347 := by
  sorry

end NUMINAMATH_CALUDE_monika_total_expense_l1550_155068


namespace NUMINAMATH_CALUDE_white_balls_count_l1550_155093

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 ∧
  green = 30 ∧
  yellow = 10 ∧
  red = 47 ∧
  purple = 3 ∧
  prob_not_red_purple = 1/2 →
  total - (green + yellow + red + purple) = 10 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l1550_155093


namespace NUMINAMATH_CALUDE_side_length_equation_l1550_155098

/-- Rectangle ABCD with equilateral triangles AEF and XYZ -/
structure SpecialRectangle where
  /-- Length of rectangle ABCD -/
  length : ℝ
  /-- Width of rectangle ABCD -/
  width : ℝ
  /-- Point E on BC such that BE = EC -/
  E : ℝ × ℝ
  /-- Point F on CD -/
  F : ℝ × ℝ
  /-- Side length of equilateral triangle XYZ -/
  s : ℝ
  /-- Rectangle ABCD has length 2 and width 1 -/
  length_eq : length = 2
  /-- Rectangle ABCD has length 2 and width 1 -/
  width_eq : width = 1
  /-- BE = EC = 1 -/
  BE_eq_EC : E.1 = 1
  /-- Angle AEF is 60 degrees -/
  angle_AEF : Real.cos (60 * π / 180) = 1 / 2
  /-- Triangle AEF is equilateral -/
  AEF_equilateral : (E.1 - 0)^2 + (E.2 - 0)^2 = (F.1 - E.1)^2 + (F.2 - E.2)^2
  /-- XY is parallel to AB -/
  XY_parallel_AB : s ≤ width

theorem side_length_equation (r : SpecialRectangle) :
  r.s^2 + 4 * r.s - 8 / Real.sqrt 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_side_length_equation_l1550_155098


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l1550_155075

theorem stratified_sampling_medium_supermarkets 
  (total_large : ℕ) (total_medium : ℕ) (total_small : ℕ) (sample_size : ℕ) :
  total_large = 200 →
  total_medium = 400 →
  total_small = 1400 →
  sample_size = 100 →
  (total_large + total_medium + total_small) * (sample_size / (total_large + total_medium + total_small)) = sample_size →
  total_medium * (sample_size / (total_large + total_medium + total_small)) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l1550_155075


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1550_155082

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) 
  (h_sum : a 2013 + a 2015 = ∫ x in (0:ℝ)..2, Real.sqrt (4 - x^2)) :
  a 2014 * (a 2012 + 2 * a 2014 + a 2016) = π^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1550_155082


namespace NUMINAMATH_CALUDE_incorrect_number_calculation_l1550_155035

theorem incorrect_number_calculation (n : ℕ) (initial_avg correct_avg correct_num : ℝ) :
  n = 10 ∧ initial_avg = 19 ∧ correct_avg = 24 ∧ correct_num = 76 →
  ∃ incorrect_num : ℝ,
    incorrect_num = correct_num - (n * correct_avg - n * initial_avg) ∧
    incorrect_num = 26 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_number_calculation_l1550_155035


namespace NUMINAMATH_CALUDE_additional_profit_special_house_l1550_155085

/-- The selling price of standard houses in the area -/
def standard_house_price : ℝ := 320000

/-- The additional cost to build the special house -/
def additional_build_cost : ℝ := 100000

/-- The factor by which the special house sells compared to standard houses -/
def special_house_price_factor : ℝ := 1.5

/-- Theorem stating the additional profit made by building the special house -/
theorem additional_profit_special_house : 
  (special_house_price_factor * standard_house_price - standard_house_price) - additional_build_cost = 60000 := by
  sorry

end NUMINAMATH_CALUDE_additional_profit_special_house_l1550_155085


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l1550_155071

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 231 → 
  ∀ a b : ℕ, a^2 - b^2 = 231 → x^2 + y^2 ≤ a^2 + b^2 → 
  x^2 + y^2 = 281 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l1550_155071


namespace NUMINAMATH_CALUDE_pats_running_speed_l1550_155039

/-- Proves that given a 20-mile course, where a person bicycles at 30 mph for 12 minutes
and then runs the rest of the distance, taking a total of 117 minutes to complete the course,
the person's average running speed is 8 mph. -/
theorem pats_running_speed (total_distance : ℝ) (bicycle_speed : ℝ) (bicycle_time : ℝ) (total_time : ℝ)
  (h1 : total_distance = 20)
  (h2 : bicycle_speed = 30)
  (h3 : bicycle_time = 12 / 60)
  (h4 : total_time = 117 / 60) :
  let bicycle_distance := bicycle_speed * bicycle_time
  let run_distance := total_distance - bicycle_distance
  let run_time := total_time - bicycle_time
  run_distance / run_time = 8 := by sorry

end NUMINAMATH_CALUDE_pats_running_speed_l1550_155039


namespace NUMINAMATH_CALUDE_x_cubed_remainder_l1550_155032

theorem x_cubed_remainder (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^3 ≡ 8 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_remainder_l1550_155032


namespace NUMINAMATH_CALUDE_function_difference_theorem_l1550_155096

theorem function_difference_theorem (m : ℚ) : 
  let f : ℚ → ℚ := λ x => 4 * x^2 - 3 * x + 5
  let g : ℚ → ℚ := λ x => 2 * x^2 - m * x + 8
  (f 5 - g 5 = 15) → m = -17/5 := by
sorry

end NUMINAMATH_CALUDE_function_difference_theorem_l1550_155096


namespace NUMINAMATH_CALUDE_album_ratio_proof_l1550_155024

/-- Prove that given the conditions, the ratio of Katrina's albums to Bridget's albums is 6:1 -/
theorem album_ratio_proof (miriam katrina bridget adele : ℕ) 
  (h1 : miriam = 5 * katrina)
  (h2 : ∃ n : ℕ, katrina = n * bridget)
  (h3 : bridget = adele - 15)
  (h4 : miriam + katrina + bridget + adele = 585)
  (h5 : adele = 30) :
  katrina / bridget = 6 := by
sorry

end NUMINAMATH_CALUDE_album_ratio_proof_l1550_155024


namespace NUMINAMATH_CALUDE_sum_medians_gt_four_times_circumradius_l1550_155010

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define properties of a triangle
def Triangle.isNonObtuse (t : Triangle) : Prop := sorry

def Triangle.medians (t : Triangle) : ℝ × ℝ × ℝ := sorry

def Triangle.circumradius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem sum_medians_gt_four_times_circumradius 
  (t : Triangle) (h : t.isNonObtuse) : 
  let (m₁, m₂, m₃) := t.medians
  m₁ + m₂ + m₃ > 4 * t.circumradius :=
by
  sorry

end NUMINAMATH_CALUDE_sum_medians_gt_four_times_circumradius_l1550_155010


namespace NUMINAMATH_CALUDE_incorrect_division_result_l1550_155053

theorem incorrect_division_result (D : ℕ) (h : D / 36 = 58) : 
  Int.floor (D / 87 : ℚ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_division_result_l1550_155053


namespace NUMINAMATH_CALUDE_no_solution_for_floor_equation_l1550_155023

theorem no_solution_for_floor_equation :
  ∀ x : ℝ, ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ ≠ 12345 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_floor_equation_l1550_155023


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_ratio_l1550_155097

theorem quadratic_roots_imply_ratio (a b : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x = -1/2 ∧ y = 1/3 ∧ a * x^2 + b * x + 2 = 0 ∧ a * y^2 + b * y + 2 = 0) →
  (a - b) / a = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_ratio_l1550_155097


namespace NUMINAMATH_CALUDE_sequence_theorem_l1550_155059

/-- A positive sequence satisfying the given condition -/
def PositiveSequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n > 0

/-- The sum of the first n terms of the sequence -/
def S (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (fun i => a ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem -/
theorem sequence_theorem (a : ℕ+ → ℝ) (h_pos : PositiveSequence a)
    (h_cond : ∀ n : ℕ+, 2 * S a n = a n ^ 2 + a n) :
    ∀ n : ℕ+, a n = n := by
  sorry

end NUMINAMATH_CALUDE_sequence_theorem_l1550_155059


namespace NUMINAMATH_CALUDE_continuous_function_inequality_l1550_155077

theorem continuous_function_inequality (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ x : ℝ, (x - 1) * (deriv f x) < 0) : f 0 + f 2 < 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_inequality_l1550_155077


namespace NUMINAMATH_CALUDE_abc_divisibility_problem_l1550_155083

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end NUMINAMATH_CALUDE_abc_divisibility_problem_l1550_155083


namespace NUMINAMATH_CALUDE_swimmer_speed_l1550_155073

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  man : ℝ  -- Speed of the man in still water (km/h)
  stream : ℝ  -- Speed of the stream (km/h)

/-- Calculates the effective speed of the swimmer. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.man + s.stream else s.man - s.stream

/-- Theorem stating that given the conditions, the man's speed in still water is 15.5 km/h. -/
theorem swimmer_speed (s : SwimmerSpeeds) 
  (h1 : effectiveSpeed s true * 2 = 36)  -- Downstream condition
  (h2 : effectiveSpeed s false * 2 = 26) -- Upstream condition
  : s.man = 15.5 := by
  sorry

#check swimmer_speed

end NUMINAMATH_CALUDE_swimmer_speed_l1550_155073


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_find_circle_parameter_l1550_155000

-- Part 1
theorem circle_equation_from_diameter (P₁ P₂ : ℝ × ℝ) (h : P₁ = (4, 9) ∧ P₂ = (6, 3)) :
  ∃ C : ℝ × ℝ, ∃ r : ℝ, ∀ x y : ℝ,
    (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ (x - 5)^2 + (y - 6)^2 = 10 :=
sorry

-- Part 2
theorem find_circle_parameter (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x - y + 3 = 0 ∧ (x - a)^2 + (y - 2)^2 = 4) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ - y₁ + 3 = 0 ∧ x₂ - y₂ + 3 = 0 ∧
    (x₁ - a)^2 + (y₁ - 2)^2 = 4 ∧ (x₂ - a)^2 + (y₂ - 2)^2 = 4 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_find_circle_parameter_l1550_155000


namespace NUMINAMATH_CALUDE_total_bowling_balls_l1550_155066

theorem total_bowling_balls (red_balls : ℕ) (green_extra : ℕ) : 
  red_balls = 30 → green_extra = 6 → red_balls + (red_balls + green_extra) = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_bowling_balls_l1550_155066


namespace NUMINAMATH_CALUDE_minimum_travel_time_l1550_155034

/-- The minimum time for a person to travel from point A to point B -/
theorem minimum_travel_time (BC : ℝ) (angle_BAC : ℝ) (swimming_speed : ℝ) 
  (h1 : BC = 30)
  (h2 : angle_BAC = 15 * π / 180)
  (h3 : swimming_speed = 3) :
  ∃ t : ℝ, t = 20 ∧ 
  ∀ t' : ℝ, t' ≥ t ∧ 
  ∃ d : ℝ, t' = d / (swimming_speed * Real.sqrt 2) + Real.sqrt (d^2 - BC^2) / swimming_speed :=
by sorry

end NUMINAMATH_CALUDE_minimum_travel_time_l1550_155034


namespace NUMINAMATH_CALUDE_function_zero_range_l1550_155056

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

def has_exactly_one_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0

theorem function_zero_range (a : ℝ) :
  has_exactly_one_zero (f a) 1 (Real.exp 2) →
  a ∈ Set.Iic (-(Real.exp 4) / 2) ∪ {-2 * Real.exp 1} :=
sorry

end NUMINAMATH_CALUDE_function_zero_range_l1550_155056


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1550_155087

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {y | y ≥ 1}

-- State the theorem
theorem intersection_A_complement_B : A ∩ Bᶜ = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1550_155087


namespace NUMINAMATH_CALUDE_problem_solution_l1550_155060

theorem problem_solution : (0.15 : ℝ)^3 - (0.06 : ℝ)^3 / (0.15 : ℝ)^2 + 0.009 + (0.06 : ℝ)^2 = 0.006375 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1550_155060


namespace NUMINAMATH_CALUDE_mabel_shark_count_l1550_155017

-- Define the percentage of sharks and other fish
def shark_percentage : ℚ := 25 / 100
def other_fish_percentage : ℚ := 75 / 100

-- Define the number of fish counted on day one
def day_one_count : ℕ := 15

-- Define the multiplier for day two
def day_two_multiplier : ℕ := 3

-- Theorem statement
theorem mabel_shark_count :
  let day_two_count := day_one_count * day_two_multiplier
  let total_fish := day_one_count + day_two_count
  let shark_count := (total_fish : ℚ) * shark_percentage
  shark_count = 15 := by sorry

end NUMINAMATH_CALUDE_mabel_shark_count_l1550_155017


namespace NUMINAMATH_CALUDE_expand_expression_l1550_155030

theorem expand_expression (x y : ℝ) : (2 * x - 5) * (3 * y + 15) = 6 * x * y + 30 * x - 15 * y - 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1550_155030


namespace NUMINAMATH_CALUDE_a2_4_sufficient_not_necessary_for_a3_16_l1550_155016

/-- A geometric sequence with first term 1 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The property that a₂ = 4 is sufficient but not necessary for a₃ = 16 -/
theorem a2_4_sufficient_not_necessary_for_a3_16 :
  ∀ a : ℕ → ℝ, GeometricSequence a →
    (∀ a : ℕ → ℝ, GeometricSequence a → a 2 = 4 → a 3 = 16) ∧
    ¬(∀ a : ℕ → ℝ, GeometricSequence a → a 3 = 16 → a 2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_a2_4_sufficient_not_necessary_for_a3_16_l1550_155016


namespace NUMINAMATH_CALUDE_jungkook_persimmons_jungkook_picked_8_persimmons_l1550_155019

theorem jungkook_persimmons : ℕ → Prop :=
  fun j : ℕ =>
    let h := 35  -- Hoseok's persimmons
    h = 4 * j + 3 → j = 8

-- Proof
theorem jungkook_picked_8_persimmons : jungkook_persimmons 8 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_persimmons_jungkook_picked_8_persimmons_l1550_155019


namespace NUMINAMATH_CALUDE_equation_represents_three_lines_not_all_intersecting_l1550_155084

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^2 * (x + y + 2) = y^2 * (x + y + 2)

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = -x
def line2 (x y : ℝ) : Prop := y = x
def line3 (x y : ℝ) : Prop := y = -x - 2

-- Theorem statement
theorem equation_represents_three_lines_not_all_intersecting :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    (∀ x y, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y)) ∧
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧
    (¬ (line1 p3.1 p3.2 ∧ line2 p3.1 p3.2 ∧ line3 p3.1 p3.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_represents_three_lines_not_all_intersecting_l1550_155084


namespace NUMINAMATH_CALUDE_quiz_passing_requirement_l1550_155008

theorem quiz_passing_requirement (total_questions : ℕ) 
  (chemistry_questions biology_questions physics_questions : ℕ)
  (chemistry_correct_percent biology_correct_percent physics_correct_percent : ℚ)
  (passing_grade : ℚ) :
  total_questions = 100 →
  chemistry_questions = 20 →
  biology_questions = 40 →
  physics_questions = 40 →
  chemistry_correct_percent = 80 / 100 →
  biology_correct_percent = 50 / 100 →
  physics_correct_percent = 55 / 100 →
  passing_grade = 65 / 100 →
  (passing_grade * total_questions : ℚ).ceil - 
  (chemistry_correct_percent * chemistry_questions +
   biology_correct_percent * biology_questions +
   physics_correct_percent * physics_questions : ℚ).floor = 7 := by
  sorry

end NUMINAMATH_CALUDE_quiz_passing_requirement_l1550_155008


namespace NUMINAMATH_CALUDE_max_value_implies_a_l1550_155042

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + a

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 4, f a x ≤ 3) ∧ (∃ x ∈ Set.Icc 0 4, f a x = 3) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l1550_155042


namespace NUMINAMATH_CALUDE_greta_worked_40_hours_l1550_155051

/-- Greta's hourly rate in dollars -/
def greta_rate : ℝ := 12

/-- Lisa's hourly rate in dollars -/
def lisa_rate : ℝ := 15

/-- Number of hours Lisa would need to work to equal Greta's earnings -/
def lisa_hours : ℝ := 32

/-- Theorem stating that Greta worked 40 hours -/
theorem greta_worked_40_hours : 
  ∃ (greta_hours : ℝ), greta_hours * greta_rate = lisa_hours * lisa_rate ∧ greta_hours = 40 := by
  sorry

end NUMINAMATH_CALUDE_greta_worked_40_hours_l1550_155051


namespace NUMINAMATH_CALUDE_calculation_error_exists_l1550_155040

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_valid_expression (expr : List (Bool × ℕ)) : Prop :=
  expr.map (Prod.snd) = numbers

def evaluate_expression (expr : List (Bool × ℕ)) : ℤ :=
  expr.foldl (λ acc (op, n) => if op then acc + n else acc - n) 0

theorem calculation_error_exists 
  (expr1 expr2 : List (Bool × ℕ)) 
  (h1 : is_valid_expression expr1)
  (h2 : is_valid_expression expr2)
  (h3 : Odd (evaluate_expression expr1))
  (h4 : Even (evaluate_expression expr2)) :
  ∃ expr, expr ∈ [expr1, expr2] ∧ evaluate_expression expr ≠ 33 ∧ evaluate_expression expr ≠ 32 := by
  sorry

end NUMINAMATH_CALUDE_calculation_error_exists_l1550_155040


namespace NUMINAMATH_CALUDE_sqrt_of_square_l1550_155054

theorem sqrt_of_square (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_of_square_l1550_155054


namespace NUMINAMATH_CALUDE_sports_enthusiasts_l1550_155025

theorem sports_enthusiasts (I A B : Finset ℕ) : 
  Finset.card I = 100 → 
  Finset.card A = 63 → 
  Finset.card B = 75 → 
  38 ≤ Finset.card (A ∩ B) ∧ Finset.card (A ∩ B) ≤ 63 := by
  sorry

end NUMINAMATH_CALUDE_sports_enthusiasts_l1550_155025


namespace NUMINAMATH_CALUDE_even_function_extension_l1550_155090

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_neg : ∀ x < 0, f x = x - x^4) :
  ∀ x > 0, f x = -x^4 - x :=
by sorry

end NUMINAMATH_CALUDE_even_function_extension_l1550_155090


namespace NUMINAMATH_CALUDE_ellipse_condition_l1550_155080

/-- The equation of the curve -/
def curve_equation (x y c : ℝ) : Prop :=
  9 * x^2 + y^2 + 54 * x - 8 * y = c

/-- Definition of a non-degenerate ellipse -/
def is_non_degenerate_ellipse (c : ℝ) : Prop :=
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧
  ∀ x y : ℝ, curve_equation x y c ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- Theorem: The curve is a non-degenerate ellipse if and only if c > -97 -/
theorem ellipse_condition (c : ℝ) :
  is_non_degenerate_ellipse c ↔ c > -97 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1550_155080


namespace NUMINAMATH_CALUDE_min_h_10_l1550_155005

/-- A function is expansive if f(x) + f(y) > x^2 + y^2 for all positive integers x and y -/
def Expansive (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y > (x.val : ℤ)^2 + (y.val : ℤ)^2

/-- The sum of h(1) to h(15) -/
def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (λ i => h ⟨i + 1, by linarith⟩)

/-- The theorem statement -/
theorem min_h_10 (h : ℕ+ → ℤ) (hExpansive : Expansive h) (hMinSum : ∀ g : ℕ+ → ℤ, Expansive g → SumH g ≥ SumH h) :
  h ⟨10, by norm_num⟩ ≥ 125 := by
  sorry

end NUMINAMATH_CALUDE_min_h_10_l1550_155005


namespace NUMINAMATH_CALUDE_height_relation_holds_for_data_height_relation_generalizes_l1550_155055

/-- Represents the height of a ball falling and rebounding -/
structure BallHeight where
  x : ℝ  -- height of ball falling
  h : ℝ  -- height of ball after landing

/-- The set of observed data points -/
def observedData : Set BallHeight := {
  ⟨10, 5⟩, ⟨30, 15⟩, ⟨50, 25⟩, ⟨70, 35⟩
}

/-- The proposed relationship between x and h -/
def heightRelation (bh : BallHeight) : Prop :=
  bh.h = (1/2) * bh.x

/-- Theorem stating that the proposed relationship holds for all observed data points -/
theorem height_relation_holds_for_data : 
  ∀ bh ∈ observedData, heightRelation bh :=
sorry

/-- Theorem stating that the relationship generalizes to any height -/
theorem height_relation_generalizes (x : ℝ) : 
  ∃ h : ℝ, heightRelation ⟨x, h⟩ :=
sorry

end NUMINAMATH_CALUDE_height_relation_holds_for_data_height_relation_generalizes_l1550_155055


namespace NUMINAMATH_CALUDE_percentage_less_than_500000_l1550_155057

-- Define the population categories
structure PopulationCategory where
  name : String
  percentage : ℝ

-- Define the theorem
theorem percentage_less_than_500000 (categories : List PopulationCategory)
  (h1 : categories.length = 3)
  (h2 : ∃ c ∈ categories, c.name = "less than 200,000" ∧ c.percentage = 35)
  (h3 : ∃ c ∈ categories, c.name = "200,000 to 499,999" ∧ c.percentage = 40)
  (h4 : ∃ c ∈ categories, c.name = "500,000 or more" ∧ c.percentage = 25)
  : (categories.filter (λ c => c.name = "less than 200,000" ∨ c.name = "200,000 to 499,999")).foldl (λ acc c => acc + c.percentage) 0 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_less_than_500000_l1550_155057


namespace NUMINAMATH_CALUDE_video_game_cost_l1550_155043

/-- If two identical video games cost $50 in total, then seven of these video games will cost $175. -/
theorem video_game_cost (cost_of_two : ℝ) (h : cost_of_two = 50) :
  7 * (cost_of_two / 2) = 175 := by
  sorry

end NUMINAMATH_CALUDE_video_game_cost_l1550_155043


namespace NUMINAMATH_CALUDE_simplify_expression_l1550_155003

theorem simplify_expression (x w : ℝ) : 3*x + 4*w - 2*x + 6 - 5*w - 5 = x - w + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1550_155003


namespace NUMINAMATH_CALUDE_keaton_apple_earnings_l1550_155074

/-- Represents Keaton's farm earnings -/
structure FarmEarnings where
  orangeHarvestFrequency : ℕ  -- Number of orange harvests per year
  orangeHarvestValue : ℕ      -- Value of each orange harvest in dollars
  totalAnnualEarnings : ℕ     -- Total annual earnings in dollars

/-- Calculates the annual earnings from apple harvest -/
def appleEarnings (f : FarmEarnings) : ℕ :=
  f.totalAnnualEarnings - (f.orangeHarvestFrequency * f.orangeHarvestValue)

/-- Theorem: Keaton's annual earnings from apple harvest is $120 -/
theorem keaton_apple_earnings :
  ∃ (f : FarmEarnings),
    f.orangeHarvestFrequency = 6 ∧
    f.orangeHarvestValue = 50 ∧
    f.totalAnnualEarnings = 420 ∧
    appleEarnings f = 120 := by
  sorry

end NUMINAMATH_CALUDE_keaton_apple_earnings_l1550_155074


namespace NUMINAMATH_CALUDE_trivia_game_score_l1550_155004

/-- Calculates the final score in a trivia game given the specified conditions -/
def calculateFinalScore (firstHalfCorrect secondHalfCorrect : ℕ) 
  (firstHalfOddPoints firstHalfEvenPoints : ℕ)
  (secondHalfOddPoints secondHalfEvenPoints : ℕ)
  (bonusPoints : ℕ) : ℕ :=
  let firstHalfOdd := firstHalfCorrect / 2 + firstHalfCorrect % 2
  let firstHalfEven := firstHalfCorrect / 2
  let secondHalfOdd := secondHalfCorrect / 2 + secondHalfCorrect % 2
  let secondHalfEven := secondHalfCorrect / 2
  let firstHalfMultiplesOf3 := (firstHalfCorrect + 2) / 3
  let secondHalfMultiplesOf3 := (secondHalfCorrect + 1) / 3
  (firstHalfOdd * firstHalfOddPoints + firstHalfEven * firstHalfEvenPoints +
   secondHalfOdd * secondHalfOddPoints + secondHalfEven * secondHalfEvenPoints +
   (firstHalfMultiplesOf3 + secondHalfMultiplesOf3) * bonusPoints)

theorem trivia_game_score :
  calculateFinalScore 10 12 2 4 3 5 5 = 113 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_score_l1550_155004


namespace NUMINAMATH_CALUDE_product_of_x_values_l1550_155058

theorem product_of_x_values (x : ℝ) : 
  (|15 / x - 2| = 3) → (∃ y : ℝ, (|15 / y - 2| = 3) ∧ x * y = -45) :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l1550_155058


namespace NUMINAMATH_CALUDE_john_pennies_l1550_155079

/-- Given that Kate has 223 pennies and John has 165 more pennies than Kate,
    prove that John has 388 pennies. -/
theorem john_pennies (kate_pennies : ℕ) (john_extra : ℕ) 
    (h1 : kate_pennies = 223)
    (h2 : john_extra = 165) :
    kate_pennies + john_extra = 388 := by
  sorry

end NUMINAMATH_CALUDE_john_pennies_l1550_155079


namespace NUMINAMATH_CALUDE_eighth_pentagon_shaded_fraction_l1550_155063

/-- Triangular number sequence -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Pentagonal number sequence -/
def pentagonal (n : ℕ) : ℕ := (3 * n^2 - n) / 2

/-- Total sections in the nth pentagon -/
def total_sections (n : ℕ) : ℕ := n^2

/-- Shaded sections in the nth pentagon -/
def shaded_sections (n : ℕ) : ℕ :=
  if n % 2 = 1 then triangular (n / 2 + 1)
  else pentagonal (n / 2)

theorem eighth_pentagon_shaded_fraction :
  (shaded_sections 8 : ℚ) / (total_sections 8 : ℚ) = 11 / 32 := by
  sorry

end NUMINAMATH_CALUDE_eighth_pentagon_shaded_fraction_l1550_155063


namespace NUMINAMATH_CALUDE_min_value_of_f_l1550_155064

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - 4 * x^3

-- Define the closed interval [-1, 0]
def I : Set ℝ := {x | -1 ≤ x ∧ x ≤ 0}

-- Theorem statement
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ I ∧ f x = -1 ∧ ∀ (y : ℝ), y ∈ I → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1550_155064


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_similar_triangle_perimeter_proof_l1550_155048

/-- Given two similar triangles where the smaller triangle has sides 15, 15, and 24,
    and the larger triangle has its longest side measuring 72,
    the perimeter of the larger triangle is 162. -/
theorem similar_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c d p =>
    (a = 15 ∧ b = 15 ∧ c = 24) →  -- Dimensions of smaller triangle
    (d = 72) →                    -- Longest side of larger triangle
    (d / c = b / a) →             -- Triangles are similar
    (p = 3 * a + d) →             -- Perimeter of larger triangle
    p = 162

theorem similar_triangle_perimeter_proof : similar_triangle_perimeter 15 15 24 72 162 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_similar_triangle_perimeter_proof_l1550_155048


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1550_155038

theorem modulus_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.abs (2 * i / (1 - i)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1550_155038


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l1550_155015

theorem rectangle_area_diagonal_relation :
  ∀ (length width diagonal : ℝ),
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 5 / 2 →
  diagonal^2 = length^2 + width^2 →
  diagonal = 13 →
  ∃ (k : ℝ), length * width = k * diagonal^2 ∧ k = 10 / 29 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l1550_155015


namespace NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l1550_155044

/-- The length of a bridge that a train can cross, given the train's length, speed, and crossing time. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof that a 500m train traveling at 42 km/h crosses a bridge of approximately 200.2m in 60 seconds. -/
theorem specific_bridge_length : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |bridge_length 500 42 60 - 200.2| < ε :=
sorry

end NUMINAMATH_CALUDE_bridge_length_specific_bridge_length_l1550_155044


namespace NUMINAMATH_CALUDE_simplify_expression_l1550_155037

theorem simplify_expression (x : ℝ) : x * (4 * x^2 - 3) - 6 * (x^2 - 3*x + 8) = 4 * x^3 - 6 * x^2 + 15 * x - 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1550_155037


namespace NUMINAMATH_CALUDE_fuel_tank_capacities_solve_problem_l1550_155046

/-- Represents the fuel tank capacities and prices for two cars -/
structure CarFuelData where
  small_capacity : ℝ
  large_capacity : ℝ
  small_fill_cost : ℝ
  large_fill_cost : ℝ
  price_difference : ℝ

/-- The theorem to be proved -/
theorem fuel_tank_capacities (data : CarFuelData) : 
  data.small_capacity = 30 ∧ data.large_capacity = 40 :=
by
  have total_capacity : data.small_capacity + data.large_capacity = 70 := by sorry
  have small_fill_equation : data.small_capacity * (data.large_fill_cost / data.large_capacity - data.price_difference) = data.small_fill_cost := by sorry
  have large_fill_equation : data.large_capacity * (data.large_fill_cost / data.large_capacity) = data.large_fill_cost := by sorry
  have price_relation : data.large_fill_cost / data.large_capacity = data.small_fill_cost / data.small_capacity + data.price_difference := by sorry
  
  sorry -- The proof would go here

/-- The specific instance of CarFuelData for our problem -/
def problem_data : CarFuelData := {
  small_capacity := 30,  -- to be proved
  large_capacity := 40,  -- to be proved
  small_fill_cost := 45,
  large_fill_cost := 68,
  price_difference := 0.29
}

/-- The main theorem applied to our specific problem -/
theorem solve_problem : 
  problem_data.small_capacity = 30 ∧ problem_data.large_capacity = 40 :=
fuel_tank_capacities problem_data

end NUMINAMATH_CALUDE_fuel_tank_capacities_solve_problem_l1550_155046


namespace NUMINAMATH_CALUDE_fraction_value_l1550_155047

theorem fraction_value : (2200 - 2096)^2 / 121 = 89 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l1550_155047


namespace NUMINAMATH_CALUDE_cube_sum_problem_l1550_155062

theorem cube_sum_problem (x y : ℝ) 
  (h1 : 1/x + 1/y = 4)
  (h2 : x*y + x^2 + y^2 = 17) :
  x^3 + y^3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l1550_155062
