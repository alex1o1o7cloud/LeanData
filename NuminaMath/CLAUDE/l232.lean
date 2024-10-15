import Mathlib

namespace NUMINAMATH_CALUDE_order_of_abc_l232_23209

theorem order_of_abc (a b c : ℝ) 
  (ha : a = 0.1 * Real.exp 0.1)
  (hb : b = 1/9)
  (hc : c = -Real.log 0.9) : 
  c < a ∧ a < b := by
sorry

end NUMINAMATH_CALUDE_order_of_abc_l232_23209


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l232_23246

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (∀ x : ℂ, (3 : ℂ) * x^2 + (a : ℂ) * x + (b : ℂ) = 0 ↔ x = 4 + 2*I ∨ x = 4 - 2*I) ∧
    c = 3 ∧
    a = -24 ∧
    b = 60 :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l232_23246


namespace NUMINAMATH_CALUDE_count_divisible_numbers_l232_23215

theorem count_divisible_numbers : 
  let upper_bound := 242400
  let divisor := 303
  (Finset.filter 
    (fun k => (k^2 + 2*k) % divisor = 0) 
    (Finset.range (upper_bound + 1))).card = 3200 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_l232_23215


namespace NUMINAMATH_CALUDE_tan_22_5_degrees_l232_23295

theorem tan_22_5_degrees :
  Real.tan (22.5 * π / 180) = Real.sqrt 8 - Real.sqrt 0 - 2 := by sorry

end NUMINAMATH_CALUDE_tan_22_5_degrees_l232_23295


namespace NUMINAMATH_CALUDE_angle_sum_proof_l232_23220

theorem angle_sum_proof (x y : Real) (h1 : 0 < x ∧ x < π/2) (h2 : 0 < y ∧ y < π/2)
  (h3 : 4 * (Real.cos x)^2 + 3 * (Real.cos y)^2 = 1)
  (h4 : 4 * Real.sin (2*x) + 3 * Real.sin (2*y) = 0) :
  x + 2*y = π/6*5 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l232_23220


namespace NUMINAMATH_CALUDE_parallelogram_intersection_ratio_l232_23270

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (W : Point) (X : Point) (Y : Point) (Z : Point)

/-- The theorem statement -/
theorem parallelogram_intersection_ratio 
  (WXYZ : Parallelogram) 
  (M : Point) (N : Point) (P : Point) :
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ M = Point.mk ((1 - t) * WXYZ.W.x + t * WXYZ.Z.x) ((1 - t) * WXYZ.W.y + t * WXYZ.Z.y) ∧ t = 3/100) →
  (∃ s : ℝ, s ∈ Set.Icc 0 1 ∧ N = Point.mk ((1 - s) * WXYZ.W.x + s * WXYZ.Y.x) ((1 - s) * WXYZ.W.y + s * WXYZ.Y.y) ∧ s = 3/251) →
  (∃ r : ℝ, r ∈ Set.Icc 0 1 ∧ P = Point.mk ((1 - r) * WXYZ.W.x + r * WXYZ.Y.x) ((1 - r) * WXYZ.W.y + r * WXYZ.Y.y)) →
  (∃ q : ℝ, q ∈ Set.Icc 0 1 ∧ P = Point.mk ((1 - q) * M.x + q * N.x) ((1 - q) * M.y + q * N.y)) →
  (WXYZ.Y.x - WXYZ.W.x) / (P.x - WXYZ.W.x) = 2 ∧ (WXYZ.Y.y - WXYZ.W.y) / (P.y - WXYZ.W.y) = 2 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_intersection_ratio_l232_23270


namespace NUMINAMATH_CALUDE_chocolates_difference_l232_23202

theorem chocolates_difference (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 7)
  (h2 : nickel_chocolates = 3) :
  robert_chocolates - nickel_chocolates = 4 := by
sorry

end NUMINAMATH_CALUDE_chocolates_difference_l232_23202


namespace NUMINAMATH_CALUDE_expression_meaning_l232_23229

theorem expression_meaning (a : ℝ) : 2 * (a - 3)^2 = 2 * (a - 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_meaning_l232_23229


namespace NUMINAMATH_CALUDE_at_least_two_first_grade_products_l232_23234

theorem at_least_two_first_grade_products (total : Nat) (first_grade : Nat) (second_grade : Nat) (third_grade : Nat) (drawn : Nat) 
  (h1 : total = 9)
  (h2 : first_grade = 4)
  (h3 : second_grade = 3)
  (h4 : third_grade = 2)
  (h5 : drawn = 4)
  (h6 : total = first_grade + second_grade + third_grade) :
  (Nat.choose total drawn) - (Nat.choose (second_grade + third_grade) drawn) - 
  (Nat.choose first_grade 1 * Nat.choose (second_grade + third_grade) (drawn - 1)) = 81 := by
sorry

end NUMINAMATH_CALUDE_at_least_two_first_grade_products_l232_23234


namespace NUMINAMATH_CALUDE_fibonacci_last_four_zeros_exist_l232_23271

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem fibonacci_last_four_zeros_exist :
  ∃ n, n < 100000001 ∧ last_four_digits (fibonacci n) = 0 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_last_four_zeros_exist_l232_23271


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l232_23278

theorem rectangular_solid_diagonal (x y z : ℝ) : 
  (2 * (x * y + y * z + z * x) = 62) →
  (4 * (x + y + z) = 48) →
  (x^2 + y^2 + z^2 : ℝ) = 82 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l232_23278


namespace NUMINAMATH_CALUDE_homework_students_l232_23238

theorem homework_students (total : ℕ) (silent_reading : ℚ) (board_games : ℚ) (group_discussions : ℚ)
  (h_total : total = 120)
  (h_silent : silent_reading = 2 / 5)
  (h_board : board_games = 3 / 10)
  (h_group : group_discussions = 1 / 8) :
  total - (silent_reading * total + board_games * total + group_discussions * total).floor = 21 :=
by sorry

end NUMINAMATH_CALUDE_homework_students_l232_23238


namespace NUMINAMATH_CALUDE_expensive_feed_cost_l232_23265

/-- Prove that the cost per pound of the more expensive dog feed is $0.53 --/
theorem expensive_feed_cost 
  (total_mix : ℝ) 
  (target_cost : ℝ) 
  (cheap_feed_cost : ℝ) 
  (cheap_feed_amount : ℝ) 
  (h1 : total_mix = 35)
  (h2 : target_cost = 0.36)
  (h3 : cheap_feed_cost = 0.18)
  (h4 : cheap_feed_amount = 17)
  : ∃ expensive_feed_cost : ℝ, 
    expensive_feed_cost = 0.53 ∧
    expensive_feed_cost * (total_mix - cheap_feed_amount) + 
    cheap_feed_cost * cheap_feed_amount = 
    target_cost * total_mix := by
  sorry

end NUMINAMATH_CALUDE_expensive_feed_cost_l232_23265


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l232_23200

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  Real.sqrt 3 * a * Real.sin C - c * (2 + Real.cos A) = 0 →
  a = Real.sqrt 13 →
  Real.sin C = 3 * Real.sin B →
  a ≥ b ∧ a ≥ c →
  (A = 2 * π / 3 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l232_23200


namespace NUMINAMATH_CALUDE_total_players_on_ground_l232_23289

theorem total_players_on_ground (cricket_players hockey_players football_players softball_players : ℕ) 
  (h1 : cricket_players = 10)
  (h2 : hockey_players = 12)
  (h3 : football_players = 16)
  (h4 : softball_players = 13) :
  cricket_players + hockey_players + football_players + softball_players = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l232_23289


namespace NUMINAMATH_CALUDE_chess_group_games_l232_23287

/-- Represents a chess group with alternating even-odd opponent play --/
structure ChessGroup where
  total_players : ℕ
  even_players : ℕ
  odd_players : ℕ
  alternating_play : Bool

/-- Calculates the total number of games played in the chess group --/
def total_games (cg : ChessGroup) : ℕ :=
  (cg.total_players * cg.even_players) / 2

/-- Theorem stating the total number of games played in a specific chess group setup --/
theorem chess_group_games :
  ∀ (cg : ChessGroup),
    cg.total_players = 12 ∧
    cg.even_players = 6 ∧
    cg.odd_players = 6 ∧
    cg.alternating_play = true →
    total_games cg = 36 := by
  sorry

end NUMINAMATH_CALUDE_chess_group_games_l232_23287


namespace NUMINAMATH_CALUDE_sheep_ratio_l232_23217

theorem sheep_ratio (total : ℕ) (beth_sheep : ℕ) (h1 : total = 608) (h2 : beth_sheep = 76) :
  (total - beth_sheep) / beth_sheep = 133 / 19 := by
  sorry

end NUMINAMATH_CALUDE_sheep_ratio_l232_23217


namespace NUMINAMATH_CALUDE_trigonometric_identities_l232_23297

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) : 
  ((2 * sin α - 3 * cos α) / (4 * sin α - 9 * cos α) = -1) ∧ 
  (4 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = 1) := by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l232_23297


namespace NUMINAMATH_CALUDE_find_number_l232_23232

theorem find_number : ∃! x : ℝ, 7 * x + 21.28 = 50.68 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l232_23232


namespace NUMINAMATH_CALUDE_inverse_sum_mod_31_l232_23257

theorem inverse_sum_mod_31 : ∃ (a b : ℤ), (5 * a) % 31 = 1 ∧ (5^2 * b) % 31 = 1 ∧ (a + b) % 31 = 26 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_31_l232_23257


namespace NUMINAMATH_CALUDE_jack_combinations_eq_44_l232_23235

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of combinations of rolls Jack could purchase. -/
def jack_combinations : ℕ := distribute 10 4

theorem jack_combinations_eq_44 : jack_combinations = 44 := by sorry

end NUMINAMATH_CALUDE_jack_combinations_eq_44_l232_23235


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l232_23203

theorem equilateral_triangle_area (perimeter : ℝ) (area : ℝ) : 
  perimeter = 30 → area = 25 * Real.sqrt 3 → 
  ∃ (side : ℝ), side > 0 ∧ 3 * side = perimeter ∧ area = (Real.sqrt 3 / 4) * side^2 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l232_23203


namespace NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l232_23208

theorem contrapositive_square_sum_zero (a b : ℝ) :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ ((a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l232_23208


namespace NUMINAMATH_CALUDE_weed_pulling_l232_23261

theorem weed_pulling (day1 : ℕ) : 
  let day2 := 3 * day1
  let day3 := day2 / 5
  let day4 := day3 - 10
  day1 + day2 + day3 + day4 = 120 →
  day1 = 25 := by
sorry

end NUMINAMATH_CALUDE_weed_pulling_l232_23261


namespace NUMINAMATH_CALUDE_swimmer_speed_is_5_l232_23243

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmingScenario where
  swimmer_speed : ℝ
  stream_speed : ℝ

/-- Calculates the effective speed when swimming downstream. -/
def downstream_speed (s : SwimmingScenario) : ℝ :=
  s.swimmer_speed + s.stream_speed

/-- Calculates the effective speed when swimming upstream. -/
def upstream_speed (s : SwimmingScenario) : ℝ :=
  s.swimmer_speed - s.stream_speed

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 5 km/h. -/
theorem swimmer_speed_is_5 (s : SwimmingScenario) 
    (h_downstream : downstream_speed s * 6 = 54)
    (h_upstream : upstream_speed s * 6 = 6) : 
    s.swimmer_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_swimmer_speed_is_5_l232_23243


namespace NUMINAMATH_CALUDE_tims_lunch_cost_l232_23225

theorem tims_lunch_cost (tip_percentage : ℝ) (total_spent : ℝ) (lunch_cost : ℝ) : 
  tip_percentage = 0.20 → 
  total_spent = 72.6 → 
  lunch_cost * (1 + tip_percentage) = total_spent → 
  lunch_cost = 60.5 := by
sorry

end NUMINAMATH_CALUDE_tims_lunch_cost_l232_23225


namespace NUMINAMATH_CALUDE_emily_subtraction_l232_23266

theorem emily_subtraction (h : 51^2 = 50^2 + 101) : 50^2 - 49^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_emily_subtraction_l232_23266


namespace NUMINAMATH_CALUDE_prob_ten_then_spade_value_l232_23210

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the event of drawing a 10 as the first card -/
def first_card_is_ten (d : Deck) : Prop :=
  ∃ c ∈ d.cards, c < 4

/-- Represents the event of drawing a spade as the second card -/
def second_card_is_spade (d : Deck) : Prop :=
  ∃ c ∈ d.cards, 39 ≤ c ∧ c < 52

/-- The probability of drawing a 10 as the first card -/
def prob_first_ten (d : Deck) : ℚ :=
  4 / 52

/-- The probability of drawing a spade as the second card -/
def prob_second_spade (d : Deck) : ℚ :=
  13 / 51

/-- The probability of drawing a 10 as the first card and a spade as the second card -/
def prob_ten_then_spade (d : Deck) : ℚ :=
  prob_first_ten d * prob_second_spade d

theorem prob_ten_then_spade_value (d : Deck) :
  prob_ten_then_spade d = 12 / 663 :=
sorry

end NUMINAMATH_CALUDE_prob_ten_then_spade_value_l232_23210


namespace NUMINAMATH_CALUDE_triangle_trig_inequality_l232_23281

theorem triangle_trig_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A + Real.cos B * Real.cos C ≤ (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_inequality_l232_23281


namespace NUMINAMATH_CALUDE_robert_books_read_l232_23262

def reading_speed : ℕ := 120
def book_length : ℕ := 360
def available_time : ℕ := 8

def books_read (speed pages time : ℕ) : ℕ :=
  (speed * time) / pages

theorem robert_books_read :
  books_read reading_speed book_length available_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_books_read_l232_23262


namespace NUMINAMATH_CALUDE_hamilton_marching_band_max_members_l232_23221

theorem hamilton_marching_band_max_members :
  ∃ (m : ℕ),
    (30 * m) % 31 = 5 ∧
    30 * m < 1500 ∧
    ∀ (n : ℕ), (30 * n) % 31 = 5 ∧ 30 * n < 1500 → 30 * n ≤ 30 * m :=
by sorry

end NUMINAMATH_CALUDE_hamilton_marching_band_max_members_l232_23221


namespace NUMINAMATH_CALUDE_min_students_satisfying_conditions_l232_23299

/-- Represents the number of students in a classroom. -/
structure Classroom where
  boys : ℕ
  girls : ℕ

/-- Checks if the given classroom satisfies all conditions. -/
def satisfiesConditions (c : Classroom) : Prop :=
  ∃ (passed_boys passed_girls : ℕ),
    passed_boys = passed_girls ∧
    passed_boys = (3 * c.boys) / 5 ∧
    passed_girls = (2 * c.girls) / 3 ∧
    (c.boys + c.girls) % 10 = 0

/-- The theorem stating the minimum number of students satisfying all conditions. -/
theorem min_students_satisfying_conditions :
  ∀ c : Classroom, satisfiesConditions c →
    ∀ c' : Classroom, satisfiesConditions c' →
      c.boys + c.girls ≤ c'.boys + c'.girls →
        c.boys + c.girls = 38 := by
  sorry

#check min_students_satisfying_conditions

end NUMINAMATH_CALUDE_min_students_satisfying_conditions_l232_23299


namespace NUMINAMATH_CALUDE_no_convex_equal_sided_all_obtuse_polygon_l232_23256

/-- A polygon is represented as a list of points in 2D space -/
def Polygon := List (Real × Real)

/-- A polygon is convex if for any three consecutive vertices, the turn is always in the same direction -/
def is_convex (p : Polygon) : Prop := sorry

/-- All sides of a polygon have equal length -/
def has_equal_sides (p : Polygon) : Prop := sorry

/-- Three points form an obtuse triangle if one of its angles is greater than 90 degrees -/
def is_obtuse_triangle (a b c : Real × Real) : Prop := sorry

/-- Any three vertices of the polygon form an obtuse triangle -/
def all_triangles_obtuse (p : Polygon) : Prop := sorry

theorem no_convex_equal_sided_all_obtuse_polygon :
  ¬∃ (p : Polygon), is_convex p ∧ has_equal_sides p ∧ all_triangles_obtuse p := by
  sorry

end NUMINAMATH_CALUDE_no_convex_equal_sided_all_obtuse_polygon_l232_23256


namespace NUMINAMATH_CALUDE_camerons_list_count_camerons_list_count_is_871_l232_23204

theorem camerons_list_count : ℕ → Prop :=
  fun count =>
    let smallest_square := 900
    let smallest_cube := 27000
    (∀ k : ℕ, k < smallest_square → ¬∃ m : ℕ, k = 30 * m * m) ∧
    (∀ k : ℕ, k < smallest_cube → ¬∃ m : ℕ, k = 30 * m * m * m) ∧
    count = (smallest_cube / 30 - smallest_square / 30 + 1)

theorem camerons_list_count_is_871 : camerons_list_count 871 := by
  sorry

end NUMINAMATH_CALUDE_camerons_list_count_camerons_list_count_is_871_l232_23204


namespace NUMINAMATH_CALUDE_equation_solution_l232_23254

theorem equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  (2 / (x - 1) - (x + 2) / (x^2 - x) = 0) ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l232_23254


namespace NUMINAMATH_CALUDE_prism_pyramid_sum_l232_23291

/-- A pentagonal prism with a pyramid attached to one of its pentagonal faces -/
structure PrismWithPyramid where
  prism_faces : Nat
  prism_vertices : Nat
  prism_edges : Nat
  pyramid_faces : Nat
  pyramid_vertex : Nat
  pyramid_edges : Nat

/-- The total number of exterior elements (faces, vertices, edges) of the combined solid -/
def total_elements (solid : PrismWithPyramid) : Nat :=
  (solid.prism_faces - 1 + solid.pyramid_faces) +
  (solid.prism_vertices + solid.pyramid_vertex) +
  (solid.prism_edges + solid.pyramid_edges)

/-- Theorem stating that the sum of exterior faces, vertices, and edges is 42 -/
theorem prism_pyramid_sum (solid : PrismWithPyramid)
  (h1 : solid.prism_faces = 7)
  (h2 : solid.prism_vertices = 10)
  (h3 : solid.prism_edges = 15)
  (h4 : solid.pyramid_faces = 5)
  (h5 : solid.pyramid_vertex = 1)
  (h6 : solid.pyramid_edges = 5) :
  total_elements solid = 42 := by
  sorry

end NUMINAMATH_CALUDE_prism_pyramid_sum_l232_23291


namespace NUMINAMATH_CALUDE_square_perimeter_l232_23214

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (5 / 2 * s = 40) → (4 * s = 64) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l232_23214


namespace NUMINAMATH_CALUDE_max_value_f_l232_23211

/-- The function f defined on ℝ -/
def f (a b x : ℝ) : ℝ := 2 * a * x + b

/-- The theorem stating the conditions and the result -/
theorem max_value_f (a b : ℝ) : 
  b > 0 ∧ 
  (∀ x ∈ Set.Icc (-1/2) (1/2), |f a b x| ≤ 2) ∧
  (∀ a' b' : ℝ, b' > 0 ∧ (∀ x ∈ Set.Icc (-1/2) (1/2), |f a' b' x| ≤ 2) → a * b ≥ a' * b') →
  f a b 2017 = 4035 := by
sorry

end NUMINAMATH_CALUDE_max_value_f_l232_23211


namespace NUMINAMATH_CALUDE_current_rate_l232_23218

/-- Proves that given a man who can row 3.3 km/hr in still water, and it takes him twice as long
    to row upstream as to row downstream, the rate of the current is 1.1 km/hr. -/
theorem current_rate (still_water_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) :
  still_water_speed = 3.3 ∧ upstream_time = 2 * downstream_time →
  ∃ current_rate : ℝ,
    current_rate = 1.1 ∧
    (still_water_speed + current_rate) * downstream_time =
    (still_water_speed - current_rate) * upstream_time :=
by sorry

end NUMINAMATH_CALUDE_current_rate_l232_23218


namespace NUMINAMATH_CALUDE_chicken_bucket_price_l232_23233

/-- Represents the price of a chicken bucket with sides -/
def bucket_price (people_per_bucket : ℕ) (total_people : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (total_people / people_per_bucket)

/-- Proves that the price of each chicken bucket with sides is $12 -/
theorem chicken_bucket_price :
  bucket_price 6 36 72 = 12 := by
  sorry

end NUMINAMATH_CALUDE_chicken_bucket_price_l232_23233


namespace NUMINAMATH_CALUDE_leah_bird_feeding_l232_23285

/-- The number of weeks Leah can feed her birds without going back to the store -/
def weeks_of_feed (boxes_bought : ℕ) (boxes_in_pantry : ℕ) (parrot_consumption : ℕ) (cockatiel_consumption : ℕ) (grams_per_box : ℕ) : ℕ :=
  let total_boxes := boxes_bought + boxes_in_pantry
  let total_grams := total_boxes * grams_per_box
  let weekly_consumption := parrot_consumption + cockatiel_consumption
  total_grams / weekly_consumption

/-- Theorem stating that Leah can feed her birds for 12 weeks -/
theorem leah_bird_feeding :
  weeks_of_feed 3 5 100 50 225 = 12 := by
sorry

end NUMINAMATH_CALUDE_leah_bird_feeding_l232_23285


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l232_23275

/-- Given a quadratic equation ax^2 + 16x + c = 0 with exactly one solution,
    where a + c = 25 and a < c, prove that a = 3 and c = 22 -/
theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 16 * x + c = 0) →  -- exactly one solution
  (a + c = 25) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 3 ∧ c = 22) :=                 -- conclusion
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l232_23275


namespace NUMINAMATH_CALUDE_cube_root_and_square_root_problem_l232_23205

theorem cube_root_and_square_root_problem :
  ∀ (a b : ℝ),
  (5 * a + 2) ^ (1/3 : ℝ) = 3 →
  (3 * a + b - 1) ^ (1/2 : ℝ) = 4 →
  a = 5 ∧ b = 2 ∧ (3 * a - b + 3) ^ (1/2 : ℝ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_and_square_root_problem_l232_23205


namespace NUMINAMATH_CALUDE_equation_solution_l232_23283

theorem equation_solution (a : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ a * x + 3 = 4 * x + 1) ↔ (a = 2 ∨ a = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l232_23283


namespace NUMINAMATH_CALUDE_prob_A_B_together_is_two_thirds_l232_23236

/-- The number of ways to arrange 3 students in a row -/
def total_arrangements : ℕ := 6

/-- The number of arrangements where A and B are together -/
def favorable_arrangements : ℕ := 4

/-- The probability that A and B stand together -/
def prob_A_B_together : ℚ := favorable_arrangements / total_arrangements

theorem prob_A_B_together_is_two_thirds : 
  prob_A_B_together = 2/3 := by sorry

end NUMINAMATH_CALUDE_prob_A_B_together_is_two_thirds_l232_23236


namespace NUMINAMATH_CALUDE_stock_price_calculation_l232_23277

theorem stock_price_calculation (closing_price : ℝ) (percent_increase : ℝ) (opening_price : ℝ) : 
  closing_price = 16 → 
  percent_increase = 6.666666666666665 → 
  closing_price = opening_price * (1 + percent_increase / 100) →
  opening_price = 15 := by
sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l232_23277


namespace NUMINAMATH_CALUDE_complete_square_result_l232_23258

/-- Given a quadratic equation x^2 + 6x - 3 = 0, prove that when completing the square, 
    the resulting equation (x + a)^2 = b has b = 12 -/
theorem complete_square_result (x : ℝ) : 
  (∃ a b : ℝ, x^2 + 6*x - 3 = 0 ↔ (x + a)^2 = b) → 
  (∃ a : ℝ, (x + a)^2 = 12) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_result_l232_23258


namespace NUMINAMATH_CALUDE_sufficient_condition_l232_23276

theorem sufficient_condition (θ P₁ P₂ : Prop) 
  (h1 : P₁ → θ) 
  (h2 : P₂ → P₁) : 
  P₂ → θ := by
sorry

end NUMINAMATH_CALUDE_sufficient_condition_l232_23276


namespace NUMINAMATH_CALUDE_smallest_n_for_324_l232_23253

/-- A geometric sequence (b_n) with given first three terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  b 1 = 2 ∧ b 2 = 6 ∧ b 3 = 18 ∧ ∀ n : ℕ, n ≥ 1 → b (n + 1) / b n = b 2 / b 1

/-- The smallest n for which b_n = 324 in the given geometric sequence is 5 -/
theorem smallest_n_for_324 (b : ℕ → ℝ) (h : geometric_sequence b) :
  (∃ n : ℕ, b n = 324) ∧ (∀ m : ℕ, b m = 324 → m ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_324_l232_23253


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l232_23240

theorem complex_magnitude_equation (n : ℝ) : 
  n > 0 ∧ Complex.abs (3 + n * Complex.I) = 3 * Real.sqrt 10 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l232_23240


namespace NUMINAMATH_CALUDE_farm_heads_count_l232_23263

/-- Represents a farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of feet on the farm -/
def totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- The total number of heads (animals) on the farm -/
def totalHeads (f : Farm) : ℕ := f.hens + f.cows

/-- Theorem: Given a farm with 24 hens and 144 total feet, the total number of heads is 48 -/
theorem farm_heads_count (f : Farm) 
  (hen_count : f.hens = 24) 
  (feet_count : totalFeet f = 144) : 
  totalHeads f = 48 := by
  sorry


end NUMINAMATH_CALUDE_farm_heads_count_l232_23263


namespace NUMINAMATH_CALUDE_linear_function_increasing_l232_23252

/-- A linear function y = mx + b where m = k - 2 and b = 3 -/
def linearFunction (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + 3

/-- The property that y increases as x increases -/
def increasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem linear_function_increasing (k : ℝ) :
  increasingFunction (linearFunction k) → k > 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_increasing_l232_23252


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l232_23207

theorem degree_to_radian_conversion (θ_deg : ℝ) (θ_rad : ℝ) :
  θ_deg = 150 ∧ θ_rad = θ_deg * (π / 180) → θ_rad = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l232_23207


namespace NUMINAMATH_CALUDE_jenna_stamps_problem_l232_23279

theorem jenna_stamps_problem (a b c : ℕ) 
  (ha : a = 924) (hb : b = 1260) (hc : c = 1386) : 
  Nat.gcd a (Nat.gcd b c) = 42 := by
  sorry

end NUMINAMATH_CALUDE_jenna_stamps_problem_l232_23279


namespace NUMINAMATH_CALUDE_triangle_sine_property_l232_23264

theorem triangle_sine_property (A B C : ℝ) (h : 3 * Real.sin B ^ 2 + 7 * Real.sin C ^ 2 = 2 * Real.sin A * Real.sin B * Real.sin C + 2 * Real.sin A ^ 2) :
  Real.sin (A + π / 4) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_property_l232_23264


namespace NUMINAMATH_CALUDE_base_8_addition_l232_23231

/-- Addition in base 8 -/
def add_base_8 (a b : ℕ) : ℕ := 
  (a + b) % 8

/-- Conversion from base 8 to base 10 -/
def base_8_to_10 (n : ℕ) : ℕ := 
  (n / 10) * 8 + (n % 10)

theorem base_8_addition : 
  add_base_8 (base_8_to_10 5) (base_8_to_10 13) = base_8_to_10 20 := by
  sorry

end NUMINAMATH_CALUDE_base_8_addition_l232_23231


namespace NUMINAMATH_CALUDE_number_of_divisors_180_l232_23241

theorem number_of_divisors_180 : Nat.card {d : ℕ | d > 0 ∧ 180 % d = 0} = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_180_l232_23241


namespace NUMINAMATH_CALUDE_area_change_possibilities_l232_23255

/-- Represents the change in area of a rectangle when one side is increased by 3 cm
    and the other is decreased by 3 cm. --/
def areaChange (a b : ℝ) : ℝ := 3 * (a - b - 3)

/-- Theorem stating that the area change can be positive, negative, or zero. --/
theorem area_change_possibilities (a b : ℝ) :
  ∃ (x y z : ℝ), x > 0 ∧ y < 0 ∧ z = 0 ∧
  (areaChange x b = z ∨ areaChange a x = z) ∧
  (areaChange y b > 0 ∨ areaChange a y > 0) ∧
  (areaChange z b < 0 ∨ areaChange a z < 0) := by
  sorry

end NUMINAMATH_CALUDE_area_change_possibilities_l232_23255


namespace NUMINAMATH_CALUDE_sqrt_neg_five_squared_l232_23273

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_five_squared_l232_23273


namespace NUMINAMATH_CALUDE_function_inequality_l232_23237

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : Differentiable ℝ f)
  (h2 : ∀ x, (x - 1) * (deriv (deriv f) x) ≤ 0) :
  f 0 + f 2 ≤ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l232_23237


namespace NUMINAMATH_CALUDE_wide_flags_count_l232_23293

/-- Represents the flag-making scenario with given parameters -/
structure FlagScenario where
  totalFabric : ℕ
  squareFlagSide : ℕ
  wideRectFlagWidth : ℕ
  wideRectFlagHeight : ℕ
  tallRectFlagWidth : ℕ
  tallRectFlagHeight : ℕ
  squareFlagsMade : ℕ
  tallFlagsMade : ℕ
  fabricLeft : ℕ

/-- Calculates the number of wide rectangular flags made -/
def wideFlagsMade (scenario : FlagScenario) : ℕ :=
  let squareFlagArea := scenario.squareFlagSide * scenario.squareFlagSide
  let wideFlagArea := scenario.wideRectFlagWidth * scenario.wideRectFlagHeight
  let tallFlagArea := scenario.tallRectFlagWidth * scenario.tallRectFlagHeight
  let usedFabric := scenario.totalFabric - scenario.fabricLeft
  let squareAndTallFlagsArea := scenario.squareFlagsMade * squareFlagArea + scenario.tallFlagsMade * tallFlagArea
  let wideFlagsArea := usedFabric - squareAndTallFlagsArea
  wideFlagsArea / wideFlagArea

/-- Theorem stating that the number of wide flags made is 20 -/
theorem wide_flags_count (scenario : FlagScenario) 
  (h1 : scenario.totalFabric = 1000)
  (h2 : scenario.squareFlagSide = 4)
  (h3 : scenario.wideRectFlagWidth = 5)
  (h4 : scenario.wideRectFlagHeight = 3)
  (h5 : scenario.tallRectFlagWidth = 3)
  (h6 : scenario.tallRectFlagHeight = 5)
  (h7 : scenario.squareFlagsMade = 16)
  (h8 : scenario.tallFlagsMade = 10)
  (h9 : scenario.fabricLeft = 294) :
  wideFlagsMade scenario = 20 := by
  sorry


end NUMINAMATH_CALUDE_wide_flags_count_l232_23293


namespace NUMINAMATH_CALUDE_more_diamonds_than_rubies_l232_23222

theorem more_diamonds_than_rubies (diamonds : ℕ) (rubies : ℕ) 
  (h1 : diamonds = 421) (h2 : rubies = 377) : 
  diamonds - rubies = 44 := by sorry

end NUMINAMATH_CALUDE_more_diamonds_than_rubies_l232_23222


namespace NUMINAMATH_CALUDE_correct_calculation_l232_23298

theorem correct_calculation (original : ℤ) (incorrect_result : ℤ) 
  (incorrect_addition : ℤ) (correct_addition : ℤ) : 
  incorrect_result = original + incorrect_addition →
  original + correct_addition = 97 :=
by
  intro h
  sorry

#check correct_calculation 35 61 26 62

end NUMINAMATH_CALUDE_correct_calculation_l232_23298


namespace NUMINAMATH_CALUDE_max_silver_tokens_l232_23267

/-- Represents the state of tokens --/
structure TokenState :=
  (red : ℕ)
  (blue : ℕ)
  (silver : ℕ)

/-- Represents an exchange at a booth --/
inductive Exchange
  | RedToSilver
  | BlueToSilver

/-- Applies an exchange to the current state --/
def applyExchange (state : TokenState) (ex : Exchange) : TokenState :=
  match ex with
  | Exchange.RedToSilver => 
      if state.red ≥ 3 then
        TokenState.mk (state.red - 3) (state.blue + 2) (state.silver + 1)
      else
        state
  | Exchange.BlueToSilver => 
      if state.blue ≥ 4 then
        TokenState.mk (state.red + 2) (state.blue - 4) (state.silver + 1)
      else
        state

/-- Checks if any exchange is possible --/
def canExchange (state : TokenState) : Bool :=
  state.red ≥ 3 ∨ state.blue ≥ 4

/-- Theorem: The maximum number of silver tokens Alex can obtain is 131 --/
theorem max_silver_tokens : 
  ∃ (exchanges : List Exchange), 
    let finalState := exchanges.foldl applyExchange (TokenState.mk 100 100 0)
    ¬(canExchange finalState) ∧ finalState.silver = 131 := by
  sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l232_23267


namespace NUMINAMATH_CALUDE_final_salary_calculation_l232_23282

/-- Calculates the final salary after two salary changes --/
theorem final_salary_calculation (initial_salary : ℝ) (first_year_raise : ℝ) (second_year_cut : ℝ) :
  initial_salary = 10 →
  first_year_raise = 0.2 →
  second_year_cut = 0.75 →
  initial_salary * (1 + first_year_raise) * second_year_cut = 9 := by
  sorry

end NUMINAMATH_CALUDE_final_salary_calculation_l232_23282


namespace NUMINAMATH_CALUDE_value_range_cos_x_tan_x_l232_23213

-- Define the function f(x) = cos x tan x
noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.tan x

-- Theorem statement
theorem value_range_cos_x_tan_x :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ Real.pi / 2 + Real.pi * ↑k ∧ f x = y) ↔ -1 < y ∧ y < 1 :=
by sorry

end NUMINAMATH_CALUDE_value_range_cos_x_tan_x_l232_23213


namespace NUMINAMATH_CALUDE_exists_diameter_points_l232_23274

/-- A circle divided into 3k arcs by 3k points -/
structure CircleDivision (k : ℕ) where
  points : Fin (3 * k) → ℝ × ℝ
  is_on_circle : ∀ i, (points i).1^2 + (points i).2^2 = 1
  arc_lengths : Fin (3 * k) → ℝ
  unit_arcs : ∃ (S : Finset (Fin (3 * k))), S.card = k ∧ ∀ i ∈ S, arc_lengths i = 1
  double_unit_arcs : ∃ (S : Finset (Fin (3 * k))), S.card = k ∧ ∀ i ∈ S, arc_lengths i = 2
  triple_unit_arcs : ∃ (S : Finset (Fin (3 * k))), S.card = k ∧ ∀ i ∈ S, arc_lengths i = 3
  total_length : (Finset.univ.sum arc_lengths) = 2 * Real.pi

/-- Two points determine a diameter if they are opposite each other on the circle -/
def is_diameter {k : ℕ} (cd : CircleDivision k) (i j : Fin (3 * k)) : Prop :=
  (cd.points i).1 = -(cd.points j).1 ∧ (cd.points i).2 = -(cd.points j).2

/-- There exist two division points that determine a diameter -/
theorem exists_diameter_points {k : ℕ} (cd : CircleDivision k) :
  ∃ (i j : Fin (3 * k)), is_diameter cd i j :=
sorry

end NUMINAMATH_CALUDE_exists_diameter_points_l232_23274


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l232_23228

/-- Given an arithmetic sequence, prove that if the sum of the first n terms is 54
    and the sum of the first 2n terms is 72, then the sum of the first 3n terms is 78. -/
theorem arithmetic_sequence_sum (n : ℕ) (S : ℕ → ℝ) : 
  S n = 54 → S (2*n) = 72 → S (3*n) = 78 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l232_23228


namespace NUMINAMATH_CALUDE_no_distinct_positive_roots_l232_23201

theorem no_distinct_positive_roots :
  ∀ (b c : ℤ), 0 ≤ b ∧ b ≤ 5 ∧ -10 ≤ c ∧ c ≤ 10 →
  ¬∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  x₁^2 + b * x₁ + c = 0 ∧ x₂^2 + b * x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_positive_roots_l232_23201


namespace NUMINAMATH_CALUDE_difference_of_squares_l232_23259

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l232_23259


namespace NUMINAMATH_CALUDE_dihedral_angle_of_inscribed_spheres_l232_23294

theorem dihedral_angle_of_inscribed_spheres (r R : ℝ) (θ : ℝ) : 
  r > 0 → 
  R = 3 * r → 
  (R + r) * Real.cos θ = (R + r) * (1/2) → 
  Real.cos (θ) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_dihedral_angle_of_inscribed_spheres_l232_23294


namespace NUMINAMATH_CALUDE_andy_cookies_count_l232_23223

/-- The number of cookies Andy ate -/
def andy_ate : Nat := 3

/-- The number of cookies Andy gave to his brother -/
def brother_cookies : Nat := 5

/-- The number of players in Andy's basketball team -/
def team_size : Nat := 8

/-- The sequence of cookies taken by each team member -/
def team_sequence (n : Nat) : Nat := 2 * n - 1

/-- The total number of cookies Andy had at the start -/
def total_cookies : Nat := andy_ate + brother_cookies + (Finset.sum (Finset.range team_size) team_sequence)

theorem andy_cookies_count : total_cookies = 72 := by sorry

end NUMINAMATH_CALUDE_andy_cookies_count_l232_23223


namespace NUMINAMATH_CALUDE_dollars_to_dozen_quarters_l232_23239

theorem dollars_to_dozen_quarters (dollars : ℕ) (quarters_per_dollar : ℕ) (items_per_dozen : ℕ) :
  dollars = 9 →
  quarters_per_dollar = 4 →
  items_per_dozen = 12 →
  (dollars * quarters_per_dollar) / items_per_dozen = 3 :=
by sorry

end NUMINAMATH_CALUDE_dollars_to_dozen_quarters_l232_23239


namespace NUMINAMATH_CALUDE_tempo_original_value_l232_23227

/-- The original value of the tempo -/
def original_value : ℝ := 11083.33

/-- The insurance coverage percentage for all three years -/
def coverage_percentage : ℚ := 5/7

/-- The premium rate for the first year -/
def premium_rate_year1 : ℚ := 3/100

/-- The premium rate for the second year -/
def premium_rate_year2 : ℚ := 4/100

/-- The premium rate for the third year -/
def premium_rate_year3 : ℚ := 5/100

/-- The total premium paid for all three years -/
def total_premium : ℝ := 950

/-- Theorem stating that the original value of the tempo satisfies the given conditions -/
theorem tempo_original_value :
  (coverage_percentage * premium_rate_year1 * original_value +
   coverage_percentage * premium_rate_year2 * original_value +
   coverage_percentage * premium_rate_year3 * original_value) = total_premium := by
  sorry

end NUMINAMATH_CALUDE_tempo_original_value_l232_23227


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l232_23224

/-- Given a right triangle PQR with legs PQ and PR, points M and N on PQ and PR respectively
    such that PM:MQ = PN:NQ = 1:3, QN = 18, and MR = 30, prove that QR = 8√18 -/
theorem right_triangle_hypotenuse (P Q R M N : ℝ × ℝ) 
  (h_right : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0)
  (h_M_on_PQ : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2))
  (h_N_on_PR : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ N = (s * P.1 + (1 - s) * R.1, s * P.2 + (1 - s) * R.2))
  (h_ratio_M : dist P M = 1/4 * dist P Q)
  (h_ratio_N : dist P N = 1/4 * dist P R)
  (h_QN : dist Q N = 18)
  (h_MR : dist M R = 30) :
  dist Q R = 8 * Real.sqrt 18 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l232_23224


namespace NUMINAMATH_CALUDE_perfect_square_digit_sum_l232_23290

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem perfect_square_digit_sum :
  (¬ ∃ n : ℕ, ∃ m : ℕ, n = m^2 ∧ sum_of_digits n = 20) ∧
  (∃ n : ℕ, ∃ m : ℕ, n = m^2 ∧ sum_of_digits n = 10) := by sorry

end NUMINAMATH_CALUDE_perfect_square_digit_sum_l232_23290


namespace NUMINAMATH_CALUDE_max_sum_with_reciprocal_constraint_l232_23245

theorem max_sum_with_reciprocal_constraint (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_eq : 1/a + 9/b = 1) : 
  a + b ≤ 16 := by
sorry

end NUMINAMATH_CALUDE_max_sum_with_reciprocal_constraint_l232_23245


namespace NUMINAMATH_CALUDE_cube_volume_l232_23216

/-- The volume of a cube with total edge length 48 cm is 64 cm³. -/
theorem cube_volume (edge_sum : ℝ) (h : edge_sum = 48) : 
  (edge_sum / 12)^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l232_23216


namespace NUMINAMATH_CALUDE_stock_exchange_problem_l232_23260

/-- The number of stocks that closed higher today -/
def higher_stocks : ℕ := 1080

/-- The number of stocks that closed lower today -/
def lower_stocks : ℕ := 900

/-- The total number of stocks on the stock exchange -/
def total_stocks : ℕ := higher_stocks + lower_stocks

theorem stock_exchange_problem :
  (higher_stocks = lower_stocks * 120 / 100) →
  (total_stocks = 1980) := by
  sorry

end NUMINAMATH_CALUDE_stock_exchange_problem_l232_23260


namespace NUMINAMATH_CALUDE_probability_A_wins_after_four_games_l232_23230

def probability_A_wins : ℚ := 3 / 5
def probability_B_wins : ℚ := 2 / 5
def number_of_games : ℕ := 4
def number_of_wins_needed : ℕ := 3

theorem probability_A_wins_after_four_games :
  (Nat.choose number_of_games number_of_wins_needed : ℚ) * 
  probability_A_wins ^ number_of_wins_needed * 
  probability_B_wins ^ (number_of_games - number_of_wins_needed) =
  (Nat.choose number_of_games number_of_wins_needed : ℚ) * 
  (3 / 5) ^ 3 * (2 / 5) := by
  sorry

end NUMINAMATH_CALUDE_probability_A_wins_after_four_games_l232_23230


namespace NUMINAMATH_CALUDE_max_sin_cos_product_l232_23212

theorem max_sin_cos_product (x : Real) : 
  (∃ (y : Real), y = Real.sin x * Real.cos x ∧ ∀ (z : Real), z = Real.sin x * Real.cos x → z ≤ y) → 
  (∃ (max : Real), max = (1 : Real) / 2 ∧ ∀ (y : Real), y = Real.sin x * Real.cos x → y ≤ max) :=
sorry

end NUMINAMATH_CALUDE_max_sin_cos_product_l232_23212


namespace NUMINAMATH_CALUDE_inequality_system_solution_l232_23206

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x - 1 ≥ a ∧ 2*x - b < 3) ↔ (3 ≤ x ∧ x < 5)) → 
  a + b = 9 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l232_23206


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l232_23296

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 6 < 0} = Set.Ioo (-3 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l232_23296


namespace NUMINAMATH_CALUDE_carries_tshirt_purchase_l232_23268

/-- The cost of a single t-shirt in dollars -/
def cost_per_shirt : ℝ := 9.95

/-- The number of t-shirts Carrie bought -/
def num_shirts : ℕ := 25

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := cost_per_shirt * (num_shirts : ℝ)

theorem carries_tshirt_purchase :
  total_cost = 248.75 := by sorry

end NUMINAMATH_CALUDE_carries_tshirt_purchase_l232_23268


namespace NUMINAMATH_CALUDE_irreducible_fraction_l232_23250

theorem irreducible_fraction (n : ℕ+) : 
  (Nat.gcd (3 * n + 1) (5 * n + 2) = 1) := by sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l232_23250


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_3_union_equality_iff_m_range_l232_23288

-- Define the sets A and B
def A : Set ℝ := {x | |x| ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 2*m + 1}

-- Define the complement of A
def C_U_A : Set ℝ := {x | |x| > 3}

-- Theorem 1
theorem intersection_complement_when_m_3 :
  (C_U_A ∩ B 3) = {x | 3 < x ∧ x < 7} := by sorry

-- Theorem 2
theorem union_equality_iff_m_range :
  ∀ m : ℝ, (A ∪ B m = A) ↔ (-2 ≤ m ∧ m ≤ 1) := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_3_union_equality_iff_m_range_l232_23288


namespace NUMINAMATH_CALUDE_additional_amount_proof_l232_23251

theorem additional_amount_proof (n : ℕ) (h : n = 3) : 7 * n - 3 * n = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_amount_proof_l232_23251


namespace NUMINAMATH_CALUDE_problem_statement_l232_23244

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  ((a - 1) * (b - 1) = 1) ∧
  (∀ a b, a > 0 → b > 0 → a + b = a * b → a + 4 * b ≥ 9) ∧
  (∀ a b, a > 0 → b > 0 → a + b = a * b → 1 / a^2 + 2 / b^2 ≥ 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l232_23244


namespace NUMINAMATH_CALUDE_cube_root_21600_l232_23242

theorem cube_root_21600 : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (∀ (a' b' : ℕ), a' > 0 → b' > 0 → a'^3 * b' = 21600 → b ≤ b') ∧ a^3 * b = 21600 ∧ a + b = 106 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_21600_l232_23242


namespace NUMINAMATH_CALUDE_special_circle_equation_l232_23286

/-- A circle with center on the line 2x - y - 3 = 0 passing through (5, 2) and (3, -2) -/
def special_circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - (2*a - 3))^2 = ((5 - a)^2 + (2 - (2*a - 3))^2)}

/-- The equation of the circle is (x-2)^2 + (y-1)^2 = 10 -/
theorem special_circle_equation :
  ∃ a : ℝ, special_circle a = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 10} ∧
    (5, 2) ∈ special_circle a ∧ (3, -2) ∈ special_circle a :=
by
  sorry

end NUMINAMATH_CALUDE_special_circle_equation_l232_23286


namespace NUMINAMATH_CALUDE_cubic_coefficient_b_value_l232_23284

/-- A cubic function passing through specific points -/
def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- Theorem stating that for a cubic function passing through (2,0), (-1,0), and (1,4), b = 6 -/
theorem cubic_coefficient_b_value 
  (a b c d : ℝ) 
  (h1 : g a b c d 2 = 0)
  (h2 : g a b c d (-1) = 0)
  (h3 : g a b c d 1 = 4) :
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_coefficient_b_value_l232_23284


namespace NUMINAMATH_CALUDE_arthurs_wallet_l232_23292

theorem arthurs_wallet (initial_amount : ℝ) : 
  (1 / 5 : ℝ) * initial_amount = 40 → initial_amount = 200 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_wallet_l232_23292


namespace NUMINAMATH_CALUDE_circumcircle_tangency_l232_23226

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary functions and relations
variable (circumcircle : Point → Point → Point → Circle)
variable (on_arc : Point → Point → Point → Point → Prop)
variable (incircle_center : Point → Point → Point → Point)
variable (touches : Circle → Circle → Prop)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem circumcircle_tangency
  (A B C D I_A I_B : Point) (k : Circle) :
  k = circumcircle A B C →
  on_arc A B C D →
  I_A = incircle_center A D C →
  I_B = incircle_center B D C →
  touches (circumcircle I_A I_B C) k ↔
    distance A D / distance B D =
    (distance A C + distance C D) / (distance B C + distance C D) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_tangency_l232_23226


namespace NUMINAMATH_CALUDE_set_intersection_equality_l232_23272

def A : Set ℝ := {x | x ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem set_intersection_equality : A ∩ B = {x | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l232_23272


namespace NUMINAMATH_CALUDE_cheburashka_krakozyabra_relation_num_cheburashkas_is_eleven_l232_23249

/-- Represents the number of Cheburashkas in Katya's drawing -/
def num_cheburashkas : ℕ := 11

/-- Represents the total number of Krakozyabras in the final drawing -/
def total_krakozyabras : ℕ := 29

/-- Represents the number of rows in Katya's drawing -/
def num_rows : ℕ := 2

/-- Theorem stating the relationship between Cheburashkas and Krakozyabras -/
theorem cheburashka_krakozyabra_relation :
  num_cheburashkas = (total_krakozyabras + num_rows) / 2 := by
  sorry

/-- Theorem proving that the number of Cheburashkas is 11 -/
theorem num_cheburashkas_is_eleven :
  num_cheburashkas = 11 := by
  sorry

end NUMINAMATH_CALUDE_cheburashka_krakozyabra_relation_num_cheburashkas_is_eleven_l232_23249


namespace NUMINAMATH_CALUDE_male_students_count_l232_23269

/-- Given a school with a total of 1200 students, where a sample of 200 students
    contains 85 females, prove that the number of male students in the school is 690. -/
theorem male_students_count (total : ℕ) (sample : ℕ) (females_in_sample : ℕ)
    (h1 : total = 1200)
    (h2 : sample = 200)
    (h3 : females_in_sample = 85) :
    total - (females_in_sample * (total / sample)) = 690 := by
  sorry

end NUMINAMATH_CALUDE_male_students_count_l232_23269


namespace NUMINAMATH_CALUDE_expand_cube_difference_l232_23280

theorem expand_cube_difference (x y : ℝ) : (x + y) * (x^2 - x*y + y^2) = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_cube_difference_l232_23280


namespace NUMINAMATH_CALUDE_polynomial_value_relation_l232_23219

theorem polynomial_value_relation (y : ℝ) : 
  4 * y^2 - 2 * y + 5 = 7 → 2 * y^2 - y + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_relation_l232_23219


namespace NUMINAMATH_CALUDE_average_salary_combined_l232_23247

theorem average_salary_combined (num_supervisors : ℕ) (num_laborers : ℕ) 
  (avg_salary_supervisors : ℚ) (avg_salary_laborers : ℚ) :
  num_supervisors = 6 →
  num_laborers = 42 →
  avg_salary_supervisors = 2450 →
  avg_salary_laborers = 950 →
  let total_salary := num_supervisors * avg_salary_supervisors + num_laborers * avg_salary_laborers
  let total_workers := num_supervisors + num_laborers
  (total_salary / total_workers : ℚ) = 1137.5 := by
sorry

end NUMINAMATH_CALUDE_average_salary_combined_l232_23247


namespace NUMINAMATH_CALUDE_science_book_pages_l232_23248

/-- Given a history book, novel, and science book, prove that the science book has 600 pages. -/
theorem science_book_pages
  (history_book novel science_book : ℕ) -- Define the books as natural numbers
  (h1 : novel = history_book / 2) -- The novel has half as many pages as the history book
  (h2 : science_book = 4 * novel) -- The science book has 4 times the amount of pages as the novel
  (h3 : history_book = 300) -- The history book has 300 pages
  : science_book = 600 := by
  sorry

end NUMINAMATH_CALUDE_science_book_pages_l232_23248
