import Mathlib

namespace NUMINAMATH_CALUDE_derivative_exponential_plus_sine_l1564_156402

theorem derivative_exponential_plus_sine (x : ℝ) :
  let y := fun x => Real.exp x + Real.sin x
  HasDerivAt y (Real.exp x + Real.cos x) x :=
by sorry

end NUMINAMATH_CALUDE_derivative_exponential_plus_sine_l1564_156402


namespace NUMINAMATH_CALUDE_reader_one_hour_ago_page_l1564_156483

/-- A reader who reads at a constant rate -/
structure Reader where
  rate : ℕ  -- pages per hour
  total_pages : ℕ
  current_page : ℕ
  remaining_hours : ℕ

/-- Calculates the page a reader was on one hour ago -/
def page_one_hour_ago (r : Reader) : ℕ :=
  r.current_page - r.rate

/-- Theorem: Given the specified conditions, the reader was on page 60 one hour ago -/
theorem reader_one_hour_ago_page :
  ∀ (r : Reader),
  r.total_pages = 210 →
  r.current_page = 90 →
  r.remaining_hours = 4 →
  (r.total_pages - r.current_page) = (r.rate * r.remaining_hours) →
  page_one_hour_ago r = 60 := by
  sorry


end NUMINAMATH_CALUDE_reader_one_hour_ago_page_l1564_156483


namespace NUMINAMATH_CALUDE_scarves_per_box_l1564_156441

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_pieces : ℕ) : 
  num_boxes = 3 → 
  mittens_per_box = 4 → 
  total_pieces = 21 → 
  (total_pieces - num_boxes * mittens_per_box) / num_boxes = 3 :=
by sorry

end NUMINAMATH_CALUDE_scarves_per_box_l1564_156441


namespace NUMINAMATH_CALUDE_sixth_term_value_l1564_156464

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the roots of the quadratic equation
def roots_of_equation (a : ℕ → ℝ) : Prop :=
  3 * (a 3)^2 - 11 * (a 3) + 9 = 0 ∧ 3 * (a 9)^2 - 11 * (a 9) + 9 = 0

-- Theorem statement
theorem sixth_term_value (a : ℕ → ℝ) :
  geometric_sequence a → roots_of_equation a → (a 6)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_sixth_term_value_l1564_156464


namespace NUMINAMATH_CALUDE_at_most_one_root_l1564_156498

-- Define a monotonically increasing function on an interval
def MonoIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Theorem statement
theorem at_most_one_root (f : ℝ → ℝ) (a b : ℝ) (h : MonoIncreasing f a b) :
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_at_most_one_root_l1564_156498


namespace NUMINAMATH_CALUDE_marathon_speed_fraction_l1564_156417

theorem marathon_speed_fraction (t₃ t₆ : ℝ) (h₁ : t₃ > 0) (h₂ : t₆ > 0) : 
  (3 * t₃ + 6 * t₆) / (t₃ + t₆) = 5 → t₃ / (t₃ + t₆) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_marathon_speed_fraction_l1564_156417


namespace NUMINAMATH_CALUDE_line_inclination_45_degrees_l1564_156493

/-- Proves that for a line passing through points (1, 2) and (3, m) with an inclination angle of 45°, m = 4 -/
theorem line_inclination_45_degrees (m : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    (1, 2) ∈ line ∧ 
    (3, m) ∈ line ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → (y - 2) = (x - 1))) → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_line_inclination_45_degrees_l1564_156493


namespace NUMINAMATH_CALUDE_rectangular_hall_dimensions_l1564_156418

theorem rectangular_hall_dimensions (length width area : ℝ) : 
  width = (1/2) * length →
  area = length * width →
  area = 200 →
  length - width = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimensions_l1564_156418


namespace NUMINAMATH_CALUDE_min_distance_curve_to_line_l1564_156458

/-- The minimum distance from a point on y = e^(2x) to the line 2x - y - 4 = 0 -/
theorem min_distance_curve_to_line : 
  let f : ℝ → ℝ := fun x ↦ Real.exp (2 * x)
  let l : ℝ → ℝ → ℝ := fun x y ↦ 2 * x - y - 4
  let d : ℝ → ℝ := fun x ↦ |l x (f x)| / Real.sqrt 5
  ∃ (x_min : ℝ), ∀ (x : ℝ), d x_min ≤ d x ∧ d x_min = 4 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_min_distance_curve_to_line_l1564_156458


namespace NUMINAMATH_CALUDE_opponent_score_l1564_156426

/-- Given UF's previous game scores and championship game performance, 
    calculate their opponent's score. -/
theorem opponent_score (total_points : ℕ) (num_games : ℕ) (half_reduction : ℕ) (point_difference : ℕ) : 
  total_points = 720 →
  num_games = 24 →
  half_reduction = 2 →
  point_difference = 2 →
  (total_points / num_games / 2 - half_reduction) - point_difference = 11 := by
  sorry


end NUMINAMATH_CALUDE_opponent_score_l1564_156426


namespace NUMINAMATH_CALUDE_lower_right_is_one_l1564_156462

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → ℕ

/-- Checks if a number appears exactly once in each row -/
def unique_in_rows (g : Grid) : Prop :=
  ∀ i n, (∃! j, g i j = n) ∧ (1 ≤ n ∧ n ≤ 5)

/-- Checks if a number appears exactly once in each column -/
def unique_in_columns (g : Grid) : Prop :=
  ∀ j n, (∃! i, g i j = n) ∧ (1 ≤ n ∧ n ≤ 5)

/-- Initial grid configuration -/
def initial_grid : Grid :=
  fun i j =>
    if i = 0 ∧ j = 0 then 1
    else if i = 0 ∧ j = 2 then 2
    else if i = 1 ∧ j = 0 then 2
    else if i = 1 ∧ j = 1 then 4
    else if i = 2 ∧ j = 3 then 5
    else if i = 3 ∧ j = 1 then 5
    else 0  -- placeholder for empty cells

/-- The main theorem -/
theorem lower_right_is_one :
  ∀ g : Grid,
    (∀ i j, initial_grid i j ≠ 0 → g i j = initial_grid i j) →
    unique_in_rows g →
    unique_in_columns g →
    g 4 4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_lower_right_is_one_l1564_156462


namespace NUMINAMATH_CALUDE_simplification_proofs_l1564_156494

theorem simplification_proofs :
  (3.5 * 101 = 353.5) ∧
  (11 * 5.9 - 5.9 = 59) ∧
  (88 - 17.5 - 12.5 = 58) := by
  sorry

end NUMINAMATH_CALUDE_simplification_proofs_l1564_156494


namespace NUMINAMATH_CALUDE_common_root_of_polynomials_l1564_156460

theorem common_root_of_polynomials (a b c d e f g : ℚ) :
  ∃ k : ℚ, k < 0 ∧ k ≠ ⌊k⌋ ∧
  (90 * k^4 + a * k^3 + b * k^2 + c * k + 18 = 0) ∧
  (18 * k^5 + d * k^4 + e * k^3 + f * k^2 + g * k + 90 = 0) :=
by
  use -1/6
  sorry

end NUMINAMATH_CALUDE_common_root_of_polynomials_l1564_156460


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l1564_156472

theorem square_plus_inverse_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) : x^2 + 1/x^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l1564_156472


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1564_156470

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 2*x + y = 5) 
  (eq2 : x + 2*y = 6) : 
  7*x^2 + 10*x*y + 7*y^2 = 85 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1564_156470


namespace NUMINAMATH_CALUDE_volume_maximized_at_height_1_2_l1564_156490

/-- Represents the dimensions of a rectangular container frame -/
structure ContainerFrame where
  shortSide : ℝ
  longSide : ℝ
  height : ℝ

/-- Calculates the volume of a container given its dimensions -/
def volume (frame : ContainerFrame) : ℝ :=
  frame.shortSide * frame.longSide * frame.height

/-- Calculates the perimeter of a container given its dimensions -/
def perimeter (frame : ContainerFrame) : ℝ :=
  2 * (frame.shortSide + frame.longSide + frame.height)

/-- Theorem: The volume of the container is maximized when the height is 1.2 m -/
theorem volume_maximized_at_height_1_2 :
  ∃ (frame : ContainerFrame),
    frame.longSide = frame.shortSide + 0.5 ∧
    perimeter frame = 14.8 ∧
    ∀ (other : ContainerFrame),
      other.longSide = other.shortSide + 0.5 →
      perimeter other = 14.8 →
      volume other ≤ volume frame ∧
      frame.height = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_volume_maximized_at_height_1_2_l1564_156490


namespace NUMINAMATH_CALUDE_avg_price_goat_l1564_156427

def num_goats : ℕ := 5
def num_hens : ℕ := 10
def total_cost : ℕ := 2500
def avg_price_hen : ℕ := 50

theorem avg_price_goat :
  (total_cost - num_hens * avg_price_hen) / num_goats = 400 :=
sorry

end NUMINAMATH_CALUDE_avg_price_goat_l1564_156427


namespace NUMINAMATH_CALUDE_tram_speed_l1564_156421

/-- The speed of a tram given observation times and tunnel length -/
theorem tram_speed (t_pass : ℝ) (t_tunnel : ℝ) (tunnel_length : ℝ) 
  (h_pass : t_pass = 3)
  (h_tunnel : t_tunnel = 13)
  (h_length : tunnel_length = 100)
  (h_positive : t_pass > 0 ∧ t_tunnel > 0 ∧ tunnel_length > 0) :
  tunnel_length / (t_tunnel - t_pass) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tram_speed_l1564_156421


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l1564_156451

/-- A regular polygon with an interior angle of 150° has 12 sides -/
theorem regular_polygon_with_150_degree_interior_angle_has_12_sides :
  ∀ (n : ℕ), n > 2 →
  (∃ (angle : ℝ), angle = 150 ∧ angle * n = 180 * (n - 2)) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l1564_156451


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l1564_156413

/-- The time it takes for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 130)
  (h2 : bridge_length = 150)
  (h3 : train_speed_kmph = 36) : 
  (train_length + bridge_length) / (train_speed_kmph * (5/18)) = 28 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l1564_156413


namespace NUMINAMATH_CALUDE_two_negative_factors_l1564_156489

theorem two_negative_factors
  (a b c : ℚ)
  (h : a * b * c > 0) :
  (a < 0 ∧ b < 0 ∧ c > 0) ∨
  (a < 0 ∧ b > 0 ∧ c < 0) ∨
  (a > 0 ∧ b < 0 ∧ c < 0) :=
sorry

end NUMINAMATH_CALUDE_two_negative_factors_l1564_156489


namespace NUMINAMATH_CALUDE_largest_non_expressible_l1564_156414

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ (a : ℕ) (b : ℕ), n = 48 * a + b ∧ is_composite b ∧ 0 < b

theorem largest_non_expressible :
  (∀ n > 95, is_expressible n) ∧
  ¬is_expressible 95 :=
sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l1564_156414


namespace NUMINAMATH_CALUDE_prob_two_queens_or_at_least_one_jack_l1564_156459

def standard_deck_size : ℕ := 52
def jack_count : ℕ := 4
def queen_count : ℕ := 4

def probability_two_queens_or_at_least_one_jack : ℚ :=
  217 / 882

theorem prob_two_queens_or_at_least_one_jack :
  probability_two_queens_or_at_least_one_jack = 
    (Nat.choose queen_count 2 * (standard_deck_size - queen_count) + 
     (standard_deck_size - jack_count).choose 2 * jack_count + 
     (standard_deck_size - jack_count).choose 1 * Nat.choose jack_count 2 + 
     Nat.choose jack_count 3) / 
    Nat.choose standard_deck_size 3 :=
by
  sorry

#eval probability_two_queens_or_at_least_one_jack

end NUMINAMATH_CALUDE_prob_two_queens_or_at_least_one_jack_l1564_156459


namespace NUMINAMATH_CALUDE_distance_after_skating_l1564_156400

/-- Calculates the distance between two skaters moving in opposite directions -/
def distance_between_skaters (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem: The distance between Ann and Glenda after skating for 3 hours -/
theorem distance_after_skating :
  let ann_speed : ℝ := 6
  let glenda_speed : ℝ := 8
  let skating_time : ℝ := 3
  distance_between_skaters ann_speed glenda_speed skating_time = 42 := by
  sorry

#check distance_after_skating

end NUMINAMATH_CALUDE_distance_after_skating_l1564_156400


namespace NUMINAMATH_CALUDE_pentagon_area_given_equal_perimeter_square_l1564_156442

theorem pentagon_area_given_equal_perimeter_square (s : ℝ) (p : ℝ) : 
  s > 0 →
  p > 0 →
  4 * s = 5 * p →
  s^2 = 16 →
  abs ((5 * p^2 * Real.tan (3 * Real.pi / 10)) / 4 - 15.26) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_given_equal_perimeter_square_l1564_156442


namespace NUMINAMATH_CALUDE_lawn_length_l1564_156479

/-- Given a rectangular lawn with specified conditions, prove its length is 70 meters -/
theorem lawn_length (width : ℝ) (road_width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) : 
  width = 60 → 
  road_width = 10 → 
  total_cost = 3600 → 
  cost_per_sqm = 3 → 
  ∃ (length : ℝ), 
    (road_width * length + road_width * (width - road_width)) * cost_per_sqm = total_cost ∧ 
    length = 70 := by
  sorry

end NUMINAMATH_CALUDE_lawn_length_l1564_156479


namespace NUMINAMATH_CALUDE_point_on_linear_graph_l1564_156452

/-- 
Given a point P(a,b) on the graph of y = 4x + 3, 
prove that the value of 4a - b - 2 is -5
-/
theorem point_on_linear_graph (a b : ℝ) : 
  b = 4 * a + 3 → 4 * a - b - 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_graph_l1564_156452


namespace NUMINAMATH_CALUDE_rounded_number_accuracy_l1564_156473

/-- Given a number 5.60 × 10^5 rounded to the nearest whole number,
    prove that it is accurate to the thousandth place. -/
theorem rounded_number_accuracy (n : ℝ) (h : n = 5.60 * 10^5) :
  ∃ (m : ℕ), |n - m| ≤ 5 * 10^2 :=
sorry

end NUMINAMATH_CALUDE_rounded_number_accuracy_l1564_156473


namespace NUMINAMATH_CALUDE_journey_matches_graph_l1564_156439

/-- Represents a segment of a journey --/
inductive JourneySegment
  | SlowAway
  | FastAway
  | Stationary
  | FastTowards
  | SlowTowards

/-- Represents a complete journey --/
def Journey := List JourneySegment

/-- Represents the shape of a graph segment --/
inductive GraphSegment
  | GradualIncline
  | SteepIncline
  | FlatLine
  | SteepDecline
  | GradualDecline

/-- Represents a complete graph --/
def Graph := List GraphSegment

/-- The journey we're analyzing --/
def janesJourney : Journey :=
  [JourneySegment.SlowAway, JourneySegment.FastAway, JourneySegment.Stationary,
   JourneySegment.FastTowards, JourneySegment.SlowTowards]

/-- The correct graph representation --/
def correctGraph : Graph :=
  [GraphSegment.GradualIncline, GraphSegment.SteepIncline, GraphSegment.FlatLine,
   GraphSegment.SteepDecline, GraphSegment.GradualDecline]

/-- Function to convert a journey to its graph representation --/
def journeyToGraph (j : Journey) : Graph :=
  sorry

/-- Theorem stating that the journey converts to the correct graph --/
theorem journey_matches_graph : journeyToGraph janesJourney = correctGraph := by
  sorry

end NUMINAMATH_CALUDE_journey_matches_graph_l1564_156439


namespace NUMINAMATH_CALUDE_boat_distribution_equation_l1564_156449

/-- Represents the boat distribution problem from "Nine Chapters on the Mathematical Art" --/
theorem boat_distribution_equation :
  ∀ (x : ℕ),
  (x ≤ 8) →
  (4 * x + 6 * (8 - x) = 38) ↔
  (x = number_of_small_boats ∧ 
   8 - x = number_of_large_boats ∧
   4 * x = people_in_small_boats ∧
   6 * (8 - x) = people_in_large_boats ∧
   people_in_small_boats + people_in_large_boats = 38) :=
by
  sorry

/-- Total number of boats --/
def total_boats : ℕ := 8

/-- Capacity of a large boat --/
def large_boat_capacity : ℕ := 6

/-- Capacity of a small boat --/
def small_boat_capacity : ℕ := 4

/-- Total number of students --/
def total_students : ℕ := 38

/-- Number of small boats (to be solved) --/
def number_of_small_boats : ℕ := sorry

/-- Number of large boats (to be solved) --/
def number_of_large_boats : ℕ := sorry

/-- Number of people in small boats --/
def people_in_small_boats : ℕ := sorry

/-- Number of people in large boats --/
def people_in_large_boats : ℕ := sorry

end NUMINAMATH_CALUDE_boat_distribution_equation_l1564_156449


namespace NUMINAMATH_CALUDE_parabola_and_line_equations_l1564_156440

/-- Parabola with focus F and point (3,m) on it -/
structure Parabola where
  p : ℝ
  m : ℝ
  h_p_pos : p > 0
  h_on_parabola : m^2 = 2 * p * 3
  h_distance_to_focus : Real.sqrt ((3 - p/2)^2 + m^2) = 4

/-- Line passing through focus F and intersecting parabola at A and B -/
structure IntersectingLine (E : Parabola) where
  k : ℝ  -- slope of the line
  h_midpoint : ∃ (y_A y_B : ℝ), y_A^2 = 4 * (k * y_A + 1) ∧
                                 y_B^2 = 4 * (k * y_B + 1) ∧
                                 (y_A + y_B) / 2 = -1

/-- Main theorem -/
theorem parabola_and_line_equations (E : Parabola) (l : IntersectingLine E) :
  (E.p = 2 ∧ ∀ x y, y^2 = 2 * E.p * x ↔ y^2 = 4 * x) ∧
  (l.k = -1/2 ∧ ∀ x y, y = l.k * (x - 1) ↔ 2 * x + y - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_equations_l1564_156440


namespace NUMINAMATH_CALUDE_water_moles_in_reaction_l1564_156422

-- Define the chemical reaction
structure ChemicalReaction where
  lithium_nitride : ℚ
  water : ℚ
  lithium_hydroxide : ℚ
  ammonia : ℚ

-- Define the balanced equation
def balanced_equation (r : ChemicalReaction) : Prop :=
  r.lithium_nitride = r.water / 3 ∧ 
  r.lithium_hydroxide = r.water ∧ 
  r.ammonia = r.water / 3

-- Theorem statement
theorem water_moles_in_reaction 
  (r : ChemicalReaction) 
  (h1 : r.lithium_nitride = 1) 
  (h2 : r.lithium_hydroxide = 3) 
  (h3 : balanced_equation r) : 
  r.water = 3 := by sorry

end NUMINAMATH_CALUDE_water_moles_in_reaction_l1564_156422


namespace NUMINAMATH_CALUDE_complex_sum_real_part_l1564_156405

theorem complex_sum_real_part (z₁ z₂ z₃ : ℂ) (r : ℝ) 
  (h₁ : Complex.abs z₁ = 1) 
  (h₂ : Complex.abs z₂ = 1) 
  (h₃ : Complex.abs z₃ = 1) 
  (h₄ : Complex.abs (z₁ + z₂ + z₃) = r) : 
  (z₁ / z₂ + z₂ / z₃ + z₃ / z₁).re = (r^2 - 3) / 2 := by
  sorry

#check complex_sum_real_part

end NUMINAMATH_CALUDE_complex_sum_real_part_l1564_156405


namespace NUMINAMATH_CALUDE_license_plate_difference_l1564_156401

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits available -/
def num_digits : ℕ := 10

/-- The number of possible license plates for Alpha state -/
def alpha_plates : ℕ := num_letters^3 * num_digits^4

/-- The number of possible license plates for Beta state -/
def beta_plates : ℕ := num_letters^4 * num_digits^3

/-- The difference in the number of possible license plates between Beta and Alpha -/
def plate_difference : ℕ := beta_plates - alpha_plates

theorem license_plate_difference : plate_difference = 281216000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l1564_156401


namespace NUMINAMATH_CALUDE_smallest_consecutive_non_primes_l1564_156433

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_non_primes (start : ℕ) : Prop :=
  ∀ i : ℕ, i < 5 → ¬(is_prime (start + i))

theorem smallest_consecutive_non_primes :
  ∃ (n : ℕ), n > 90 ∧ n < 96 ∧ consecutive_non_primes n ∧
  ∀ m : ℕ, m > 90 ∧ m < 96 ∧ consecutive_non_primes m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_non_primes_l1564_156433


namespace NUMINAMATH_CALUDE_largest_common_divisor_l1564_156429

theorem largest_common_divisor : ∃ (n : ℕ), n = 45 ∧ 
  n ∣ 540 ∧ n < 60 ∧ n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 ∧ m < 60 ∧ m ∣ 180 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l1564_156429


namespace NUMINAMATH_CALUDE_smallest_n_cubic_minus_n_divisibility_l1564_156455

theorem smallest_n_cubic_minus_n_divisibility : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), 0 < m ∧ m < n → 
    ∀ (k : ℕ), 1 ≤ k ∧ k ≤ m + 2 → (m^3 - m) % k = 0) ∧
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n + 2 ∧ (n^3 - n) % k ≠ 0) ∧
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_cubic_minus_n_divisibility_l1564_156455


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1564_156424

def set_A : Set ℝ := {x | x + 2 = 0}
def set_B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1564_156424


namespace NUMINAMATH_CALUDE_expenses_notation_l1564_156434

def income : ℤ := 5
def expenses : ℤ := 5

theorem expenses_notation (h : income = 5) : expenses = -5 := by
  sorry

end NUMINAMATH_CALUDE_expenses_notation_l1564_156434


namespace NUMINAMATH_CALUDE_election_votes_l1564_156495

/-- Represents the total number of votes in an election --/
def total_votes : ℕ := sorry

/-- Represents the percentage of votes for candidate A --/
def percent_A : ℚ := 45/100

/-- Represents the percentage of votes for candidate B --/
def percent_B : ℚ := 35/100

/-- Represents the percentage of votes for candidate C --/
def percent_C : ℚ := 1 - percent_A - percent_B

/-- The difference between votes for A and B --/
def diff_AB : ℕ := 500

/-- The difference between votes for B and C --/
def diff_BC : ℕ := 350

theorem election_votes : 
  (percent_A * total_votes : ℚ) - (percent_B * total_votes : ℚ) = diff_AB ∧
  (percent_B * total_votes : ℚ) - (percent_C * total_votes : ℚ) = diff_BC →
  total_votes = 5000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l1564_156495


namespace NUMINAMATH_CALUDE_negation_equivalence_l1564_156467

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < Real.sin x ∨ x > Real.tan x) ↔
  (∀ x : ℝ, x ≥ Real.sin x ∧ x ≤ Real.tan x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1564_156467


namespace NUMINAMATH_CALUDE_distance_between_points_l1564_156448

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (4, -6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1564_156448


namespace NUMINAMATH_CALUDE_total_saltwater_animals_l1564_156491

/-- The number of aquariums -/
def num_aquariums : ℕ := 20

/-- The number of animals per aquarium -/
def animals_per_aquarium : ℕ := 2

/-- Theorem stating the total number of saltwater animals -/
theorem total_saltwater_animals : 
  num_aquariums * animals_per_aquarium = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_saltwater_animals_l1564_156491


namespace NUMINAMATH_CALUDE_sum_exterior_angles_quadrilateral_l1564_156412

/-- A quadrilateral is a polygon with four sides. -/
def Quadrilateral : Type := Unit  -- Placeholder definition

/-- The sum of exterior angles of a polygon. -/
def sum_exterior_angles (p : Type) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of a quadrilateral is 360 degrees. -/
theorem sum_exterior_angles_quadrilateral :
  sum_exterior_angles Quadrilateral = 360 := by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_quadrilateral_l1564_156412


namespace NUMINAMATH_CALUDE_books_sold_l1564_156484

theorem books_sold (initial_books : ℕ) (added_books : ℕ) (final_books : ℕ) : 
  initial_books = 4 → added_books = 10 → final_books = 11 → 
  initial_books - (final_books - added_books) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_books_sold_l1564_156484


namespace NUMINAMATH_CALUDE_abes_age_problem_l1564_156477

/-- Abe's age problem -/
theorem abes_age_problem (present_age : ℕ) (x : ℕ) 
  (h1 : present_age = 28)
  (h2 : present_age + (present_age - x) = 35) :
  present_age + x = 49 := by
  sorry

end NUMINAMATH_CALUDE_abes_age_problem_l1564_156477


namespace NUMINAMATH_CALUDE_product_expansion_l1564_156486

theorem product_expansion (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l1564_156486


namespace NUMINAMATH_CALUDE_garrett_granola_bars_l1564_156485

/-- Proves that Garrett bought 6 oatmeal raisin granola bars -/
theorem garrett_granola_bars :
  ∀ (total peanut oatmeal_raisin : ℕ),
    total = 14 →
    peanut = 8 →
    total = peanut + oatmeal_raisin →
    oatmeal_raisin = 6 := by
  sorry

end NUMINAMATH_CALUDE_garrett_granola_bars_l1564_156485


namespace NUMINAMATH_CALUDE_f_property_f_1001_eq_1_f_1002_eq_1_l1564_156444

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def has_prime_divisor (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ p ∣ n

theorem f_property (f : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 1 →
    ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

theorem f_1001_eq_1 (f : ℕ → ℤ) : Prop := f 1001 = 1

theorem f_1002_eq_1 (f : ℕ → ℤ) : f_property f → f_1001_eq_1 f → f 1002 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_property_f_1001_eq_1_f_1002_eq_1_l1564_156444


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1564_156445

theorem point_in_fourth_quadrant :
  let P : ℝ × ℝ := (Real.tan (2015 * π / 180), Real.cos (2015 * π / 180))
  (0 < P.1) ∧ (P.2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1564_156445


namespace NUMINAMATH_CALUDE_seeds_planted_equals_85_l1564_156443

/-- Calculates the total number of seeds planted given the number of seeds per bed,
    flowers per bed, and total flowers grown. -/
def total_seeds_planted (seeds_per_bed : ℕ) (flowers_per_bed : ℕ) (total_flowers : ℕ) : ℕ :=
  let full_beds := total_flowers / flowers_per_bed
  let seeds_in_full_beds := full_beds * seeds_per_bed
  let flowers_in_partial_bed := total_flowers % flowers_per_bed
  seeds_in_full_beds + flowers_in_partial_bed

/-- Theorem stating that given the specific conditions, the total seeds planted is 85. -/
theorem seeds_planted_equals_85 :
  total_seeds_planted 15 60 220 = 85 := by
  sorry

end NUMINAMATH_CALUDE_seeds_planted_equals_85_l1564_156443


namespace NUMINAMATH_CALUDE_max_floor_product_sum_l1564_156499

theorem max_floor_product_sum (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → x + y + z = 1399 →
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1399 →
  ⌊x⌋ * y + ⌊y⌋ * z + ⌊z⌋ * x ≤ ⌊a⌋ * b + ⌊b⌋ * c + ⌊c⌋ * a →
  ⌊a⌋ * b + ⌊b⌋ * c + ⌊c⌋ * a ≤ 652400 :=
by sorry

#check max_floor_product_sum

end NUMINAMATH_CALUDE_max_floor_product_sum_l1564_156499


namespace NUMINAMATH_CALUDE_degrees_to_radians_conversion_l1564_156482

theorem degrees_to_radians_conversion :
  ∀ (degrees : ℝ) (radians : ℝ),
  degrees * (π / 180) = radians →
  -630 * (π / 180) = -7 * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_degrees_to_radians_conversion_l1564_156482


namespace NUMINAMATH_CALUDE_logarithm_problem_l1564_156480

noncomputable def a : ℝ := Real.log 55 / Real.log 50
noncomputable def b : ℝ := Real.log 20 / Real.log 55

theorem logarithm_problem (a b : ℝ) (h1 : a = Real.log 55 / Real.log 50) (h2 : b = Real.log 20 / Real.log 55) :
  Real.log (2662 * Real.sqrt 10) / Real.log 250 = (18 * a + 11 * a * b - 13) / (10 - 2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_problem_l1564_156480


namespace NUMINAMATH_CALUDE_special_number_property_l1564_156469

/-- The greatest integer less than 100 for which the greatest common factor with 18 is 3 -/
def special_number : ℕ := 93

/-- Theorem stating that special_number satisfies the required conditions -/
theorem special_number_property : 
  special_number < 100 ∧ 
  Nat.gcd special_number 18 = 3 ∧ 
  ∀ n : ℕ, n < 100 → Nat.gcd n 18 = 3 → n ≤ special_number := by
  sorry

end NUMINAMATH_CALUDE_special_number_property_l1564_156469


namespace NUMINAMATH_CALUDE_display_window_configurations_l1564_156416

/-- The number of permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The number of configurations for a single window with n books -/
def window_configurations (n : ℕ) : ℕ := factorial n

/-- The total number of configurations for two windows -/
def total_configurations (left_window : ℕ) (right_window : ℕ) : ℕ :=
  window_configurations left_window * window_configurations right_window

theorem display_window_configurations :
  total_configurations 3 3 = 36 :=
by sorry

end NUMINAMATH_CALUDE_display_window_configurations_l1564_156416


namespace NUMINAMATH_CALUDE_chemical_mixture_percentage_l1564_156436

/-- Given two solutions x and y with different compositions of chemicals a and b,
    and a mixture of these solutions, prove that the percentage of chemical a
    in the mixture is 12%. -/
theorem chemical_mixture_percentage : 
  let x_percent_a : ℝ := 10  -- Percentage of chemical a in solution x
  let x_percent_b : ℝ := 90  -- Percentage of chemical b in solution x
  let y_percent_a : ℝ := 20  -- Percentage of chemical a in solution y
  let y_percent_b : ℝ := 80  -- Percentage of chemical b in solution y
  let mixture_percent_x : ℝ := 80  -- Percentage of solution x in the mixture
  let mixture_percent_y : ℝ := 20  -- Percentage of solution y in the mixture

  -- Ensure percentages add up to 100%
  x_percent_a + x_percent_b = 100 →
  y_percent_a + y_percent_b = 100 →
  mixture_percent_x + mixture_percent_y = 100 →

  -- Calculate the percentage of chemical a in the mixture
  (mixture_percent_x * x_percent_a + mixture_percent_y * y_percent_a) / 100 = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_chemical_mixture_percentage_l1564_156436


namespace NUMINAMATH_CALUDE_profit_for_450_pieces_l1564_156471

/-- The price function for the clothing factory -/
def price (x : ℕ) : ℚ :=
  if x ≤ 100 then 60
  else 62 - x / 50

/-- The profit function for the clothing factory -/
def profit (x : ℕ) : ℚ :=
  (price x - 40) * x

/-- The theorem stating the profit for an order of 450 pieces -/
theorem profit_for_450_pieces :
  0 < 450 ∧ 450 ≤ 500 → profit 450 = 5850 := by sorry

end NUMINAMATH_CALUDE_profit_for_450_pieces_l1564_156471


namespace NUMINAMATH_CALUDE_students_not_participating_l1564_156463

theorem students_not_participating (total : ℕ) (football : ℕ) (tennis : ℕ) (basketball : ℕ)
  (football_tennis : ℕ) (football_basketball : ℕ) (tennis_basketball : ℕ) (all_three : ℕ) :
  total = 50 →
  football = 30 →
  tennis = 25 →
  basketball = 18 →
  football_tennis = 12 →
  football_basketball = 10 →
  tennis_basketball = 8 →
  all_three = 5 →
  total - (football + tennis + basketball - football_tennis - football_basketball - tennis_basketball + all_three) = 2 :=
by sorry

end NUMINAMATH_CALUDE_students_not_participating_l1564_156463


namespace NUMINAMATH_CALUDE_grid_d4_is_5_l1564_156465

/-- Represents a 5x5 grid of numbers -/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Checks if a row contains all different numbers -/
def row_all_different (g : Grid) (r : Fin 5) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g r i ≠ g r j

/-- Checks if a column contains all different numbers -/
def col_all_different (g : Grid) (c : Fin 5) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i c ≠ g j c

/-- Checks if all rows and columns contain different numbers -/
def all_different (g : Grid) : Prop :=
  (∀ r : Fin 5, row_all_different g r) ∧ (∀ c : Fin 5, col_all_different g c)

/-- Checks if the sum of numbers in the 4th column is 9 -/
def fourth_column_sum_9 (g : Grid) : Prop :=
  (g 1 3).val + (g 3 3).val = 9

/-- Checks if the sum of numbers in white cells of row C is 7 -/
def row_c_white_sum_7 (g : Grid) : Prop :=
  (g 2 0).val + (g 2 2).val + (g 2 4).val = 7

/-- Checks if the sum of numbers in white cells of 2nd column is 8 -/
def second_column_white_sum_8 (g : Grid) : Prop :=
  (g 0 1).val + (g 2 1).val + (g 4 1).val = 8

/-- Checks if the sum of numbers in white cells of row B is less than row D -/
def row_b_less_than_row_d (g : Grid) : Prop :=
  (g 1 1).val + (g 1 3).val < (g 3 1).val + (g 3 3).val

theorem grid_d4_is_5 (g : Grid) 
  (h1 : all_different g)
  (h2 : fourth_column_sum_9 g)
  (h3 : row_c_white_sum_7 g)
  (h4 : second_column_white_sum_8 g)
  (h5 : row_b_less_than_row_d g) :
  g 3 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_grid_d4_is_5_l1564_156465


namespace NUMINAMATH_CALUDE_renovation_project_materials_l1564_156408

/-- The total number of truck-loads of material needed for a renovation project -/
theorem renovation_project_materials :
  let sand := 0.16666666666666666 * Real.pi
  let dirt := 0.3333333333333333 * Real.exp 1
  let cement := 0.16666666666666666 * Real.sqrt 2
  let gravel := 0.25 * Real.log 5
  abs ((sand + dirt + cement + gravel) - 1.8401374808985008) < 1e-10 := by
sorry

end NUMINAMATH_CALUDE_renovation_project_materials_l1564_156408


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1564_156438

theorem complex_equation_solution :
  let z : ℂ := -3 * I / 4
  2 - 3 * I * z = -4 + 5 * I * z :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1564_156438


namespace NUMINAMATH_CALUDE_square_sum_from_linear_and_product_l1564_156456

theorem square_sum_from_linear_and_product (x y : ℝ) 
  (h1 : x + 3 * y = 3) (h2 : x * y = -6) : 
  x^2 + 9 * y^2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_linear_and_product_l1564_156456


namespace NUMINAMATH_CALUDE_simplify_fraction_l1564_156478

theorem simplify_fraction (x : ℝ) (hx : x ≠ 0) : (x + 1) / x - 1 / x = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1564_156478


namespace NUMINAMATH_CALUDE_algebraic_expression_change_l1564_156435

theorem algebraic_expression_change (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (3/4 * x) * ((3/4 * y)^2) * (5/4 * z) = (135/256) * x * y^2 * z :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_change_l1564_156435


namespace NUMINAMATH_CALUDE_touchdown_points_l1564_156461

theorem touchdown_points (total_points : ℕ) (num_touchdowns : ℕ) (points_per_touchdown : ℕ) :
  total_points = 21 →
  num_touchdowns = 3 →
  total_points = num_touchdowns * points_per_touchdown →
  points_per_touchdown = 7 := by
  sorry

end NUMINAMATH_CALUDE_touchdown_points_l1564_156461


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1564_156487

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 1) ↔ (∃ x₀ : ℝ, x₀^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1564_156487


namespace NUMINAMATH_CALUDE_sqrt_8_is_quadratic_radical_l1564_156409

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), y ≥ 0 ∧ x = Real.sqrt y

-- Theorem statement
theorem sqrt_8_is_quadratic_radical :
  is_quadratic_radical (Real.sqrt 8) ∧
  ¬(∀ x : ℝ, is_quadratic_radical (Real.sqrt x)) ∧
  ¬(∀ m n : ℝ, is_quadratic_radical (Real.sqrt (m + n))) :=
sorry

end NUMINAMATH_CALUDE_sqrt_8_is_quadratic_radical_l1564_156409


namespace NUMINAMATH_CALUDE_median_and_mode_are_23_l1564_156411

/-- Represents the shoe size distribution of a class --/
structure ShoeSizeDistribution where
  sizes : List Nat
  frequencies : List Nat
  total_students : Nat
  h_sizes_freq : sizes.length = frequencies.length
  h_total : total_students = frequencies.sum

/-- Calculates the median of a shoe size distribution --/
def median (d : ShoeSizeDistribution) : Nat :=
  sorry

/-- Calculates the mode of a shoe size distribution --/
def mode (d : ShoeSizeDistribution) : Nat :=
  sorry

/-- The shoe size distribution for the class in the problem --/
def class_distribution : ShoeSizeDistribution :=
  { sizes := [20, 21, 22, 23, 24],
    frequencies := [2, 8, 9, 19, 2],
    total_students := 40,
    h_sizes_freq := by rfl,
    h_total := by rfl }

theorem median_and_mode_are_23 :
  median class_distribution = 23 ∧ mode class_distribution = 23 :=
sorry

end NUMINAMATH_CALUDE_median_and_mode_are_23_l1564_156411


namespace NUMINAMATH_CALUDE_ali_class_size_l1564_156415

/-- Calculates the total number of students in a class given a student's rank from top and bottom -/
def class_size (rank_from_top : ℕ) (rank_from_bottom : ℕ) : ℕ :=
  rank_from_top + rank_from_bottom - 1

/-- Theorem: In a class where a student ranks 40th from both the top and bottom, the total number of students is 79 -/
theorem ali_class_size :
  class_size 40 40 = 79 := by
  sorry

#eval class_size 40 40

end NUMINAMATH_CALUDE_ali_class_size_l1564_156415


namespace NUMINAMATH_CALUDE_even_factors_count_l1564_156420

/-- The number of even positive factors of 2^4 * 3^2 * 5 * 7 -/
def num_even_factors (n : ℕ) : ℕ :=
  if n = 2^4 * 3^2 * 5 * 7 then
    (4 * 3 * 2 * 2)  -- 4 choices for 2's exponent (1 to 4), 3 for 3's (0 to 2), 2 for 5's (0 to 1), 2 for 7's (0 to 1)
  else
    0  -- Return 0 if n is not equal to 2^4 * 3^2 * 5 * 7

theorem even_factors_count (n : ℕ) :
  n = 2^4 * 3^2 * 5 * 7 → num_even_factors n = 48 := by
  sorry

end NUMINAMATH_CALUDE_even_factors_count_l1564_156420


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l1564_156454

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 49}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l1564_156454


namespace NUMINAMATH_CALUDE_unique_triplet_l1564_156447

theorem unique_triplet : 
  ∃! (a b c : ℕ), 2 ≤ a ∧ a < b ∧ b < c ∧ 
  (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) ∧
  a = 4 ∧ b = 5 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_triplet_l1564_156447


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1564_156432

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 189)
  (h3 : downstream_time = 7)
  : ∃ (boat_speed : ℝ), boat_speed = 22 ∧ 
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1564_156432


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1564_156446

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1564_156446


namespace NUMINAMATH_CALUDE_sqrt_three_difference_of_squares_l1564_156404

theorem sqrt_three_difference_of_squares : (Real.sqrt 3 - 1) * (Real.sqrt 3 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_difference_of_squares_l1564_156404


namespace NUMINAMATH_CALUDE_third_person_weight_l1564_156468

/-- The weight of the third person entering an elevator given specific average weight changes --/
theorem third_person_weight (initial_people : ℕ) (initial_avg : ℝ) 
  (avg_after_first : ℝ) (avg_after_second : ℝ) (avg_after_third : ℝ) :
  initial_people = 6 →
  initial_avg = 156 →
  avg_after_first = 159 →
  avg_after_second = 162 →
  avg_after_third = 161 →
  ∃ (w1 w2 w3 : ℝ),
    w1 = (initial_people + 1) * avg_after_first - initial_people * initial_avg ∧
    w2 = (initial_people + 2) * avg_after_second - (initial_people + 1) * avg_after_first ∧
    w3 = (initial_people + 3) * avg_after_third - (initial_people + 2) * avg_after_second ∧
    w3 = 163 :=
by sorry

end NUMINAMATH_CALUDE_third_person_weight_l1564_156468


namespace NUMINAMATH_CALUDE_inference_is_analogical_l1564_156428

/-- Inductive reasoning is the process of reasoning from specific instances to a general conclusion. -/
def inductive_reasoning : Prop := sorry

/-- Deductive reasoning is the process of reasoning from a general premise to a specific conclusion. -/
def deductive_reasoning : Prop := sorry

/-- Analogical reasoning is the process of reasoning from one specific instance to another specific instance. -/
def analogical_reasoning : Prop := sorry

/-- The inference from "If a > b, then a + c > b + c" to "If a > b, then ac > bc" -/
def inference : Prop := sorry

/-- The inference is an example of analogical reasoning -/
theorem inference_is_analogical : inference → analogical_reasoning := by sorry

end NUMINAMATH_CALUDE_inference_is_analogical_l1564_156428


namespace NUMINAMATH_CALUDE_consecutive_numbers_with_divisible_digit_sums_l1564_156419

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exist two consecutive natural numbers whose sum of digits are both divisible by 7 -/
theorem consecutive_numbers_with_divisible_digit_sums :
  ∃ n : ℕ, 7 ∣ sumOfDigits n ∧ 7 ∣ sumOfDigits (n + 1) := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_with_divisible_digit_sums_l1564_156419


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l1564_156488

theorem stratified_sampling_problem (total_population : ℕ) 
  (stratum_size : ℕ) (stratum_sample : ℕ) (h1 : total_population = 55) 
  (h2 : stratum_size = 15) (h3 : stratum_sample = 3) :
  (stratum_sample : ℚ) * total_population / stratum_size = 11 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l1564_156488


namespace NUMINAMATH_CALUDE_pentagon_ink_length_l1564_156496

/-- Ink length of a regular pentagon with side length n -/
def inkLength (n : ℕ) : ℕ := 5 * n

theorem pentagon_ink_length :
  (inkLength 4 = 20) ∧
  (inkLength 9 - inkLength 8 = 5) ∧
  (inkLength 100 = 500) := by
  sorry

end NUMINAMATH_CALUDE_pentagon_ink_length_l1564_156496


namespace NUMINAMATH_CALUDE_sum_of_squares_l1564_156453

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (power_eq : a^7 + b^7 + c^7 = a^9 + b^9 + c^9) :
  a^2 + b^2 + c^2 = 14/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1564_156453


namespace NUMINAMATH_CALUDE_rhombus_60_min_rotation_l1564_156492

/-- A rhombus with a 60° angle -/
structure Rhombus60 where
  /-- The rhombus has a 60° angle -/
  angle_60 : ∃ θ, θ = 60

/-- Minimum rotation for a Rhombus60 to coincide with its original position -/
def min_rotation (r : Rhombus60) : ℝ :=
  180

/-- Theorem: The minimum rotation for a Rhombus60 to coincide with its original position is 180° -/
theorem rhombus_60_min_rotation (r : Rhombus60) :
  min_rotation r = 180 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_60_min_rotation_l1564_156492


namespace NUMINAMATH_CALUDE_candidates_per_state_l1564_156457

theorem candidates_per_state (total : ℕ) (selected_A selected_B : ℕ) 
  (h1 : selected_A = total * 6 / 100)
  (h2 : selected_B = total * 7 / 100)
  (h3 : selected_B = selected_A + 79) :
  total = 7900 := by
  sorry

end NUMINAMATH_CALUDE_candidates_per_state_l1564_156457


namespace NUMINAMATH_CALUDE_factor_expression_l1564_156431

theorem factor_expression (x : ℝ) : x * (x + 3) - 2 * (x + 3) = (x + 3) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1564_156431


namespace NUMINAMATH_CALUDE_range_of_c_l1564_156437

-- Define the triangular pyramid
structure TriangularPyramid where
  -- Base edges
  base_edge1 : ℝ
  base_edge2 : ℝ
  base_edge3 : ℝ
  -- Side edges opposite to base edges
  side_edge1 : ℝ
  side_edge2 : ℝ
  side_edge3 : ℝ

-- Define the specific triangular pyramid from the problem
def specificPyramid (c : ℝ) : TriangularPyramid :=
  { base_edge1 := 1
  , base_edge2 := 1
  , base_edge3 := c
  , side_edge1 := 1
  , side_edge2 := c
  , side_edge3 := c }

-- Theorem stating the range of c
theorem range_of_c :
  ∀ c : ℝ, (∃ p : TriangularPyramid, p = specificPyramid c) →
  (Real.sqrt 5 - 1) / 2 < c ∧ c < (Real.sqrt 5 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_c_l1564_156437


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1564_156407

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x > 3 → x > 2) ∧
  (∃ x : ℝ, x > 2 ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1564_156407


namespace NUMINAMATH_CALUDE_specific_pyramid_properties_l1564_156450

/-- Represents a straight pyramid with an equilateral triangular base -/
structure EquilateralPyramid where
  height : ℝ
  side_face_area : ℝ

/-- Calculates the base edge length of the pyramid -/
def base_edge_length (p : EquilateralPyramid) : ℝ := sorry

/-- Calculates the volume of the pyramid -/
def volume (p : EquilateralPyramid) : ℝ := sorry

/-- Theorem stating the properties of the specific pyramid -/
theorem specific_pyramid_properties :
  let p : EquilateralPyramid := { height := 11, side_face_area := 210 }
  base_edge_length p = 30 ∧ volume p = 825 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_properties_l1564_156450


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1564_156481

/-- A symmetric trapezoid with an inscribed and circumscribed circle -/
structure SymmetricTrapezoid where
  -- The lengths of the parallel sides
  a : ℝ
  b : ℝ
  -- The radius of the circumscribed circle
  R : ℝ
  -- The radius of the inscribed circle
  ρ : ℝ
  -- Conditions
  h_symmetric : a ≥ b
  h_R : R = 1
  h_inscribed : ρ > 0
  h_center_bisects : ∃ (K : ℝ × ℝ), K.1^2 + K.2^2 = (R/2)^2

/-- The radius of the inscribed circle in the symmetric trapezoid -/
theorem inscribed_circle_radius (T : SymmetricTrapezoid) : T.ρ = Real.sqrt (9/40) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1564_156481


namespace NUMINAMATH_CALUDE_two_consecutive_count_l1564_156425

/-- Represents the number of balls in the box -/
def n : ℕ := 5

/-- Represents the number of people drawing balls -/
def k : ℕ := 3

/-- Counts the number of ways to draw balls with exactly two consecutive numbers -/
def count_two_consecutive (n k : ℕ) : ℕ :=
  sorry

theorem two_consecutive_count :
  count_two_consecutive n k = 36 := by
  sorry

end NUMINAMATH_CALUDE_two_consecutive_count_l1564_156425


namespace NUMINAMATH_CALUDE_vector_sum_proof_l1564_156466

/-- Given two vectors in a plane, prove that their sum with specific coefficients equals a certain vector. -/
theorem vector_sum_proof (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (a • b = 0) → 
  (2 • a + 3 • b : Fin 2 → ℝ) = ![-4, 7] :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l1564_156466


namespace NUMINAMATH_CALUDE_tv_price_increase_l1564_156430

theorem tv_price_increase (initial_price : ℝ) (first_increase : ℝ) : 
  first_increase > 0 →
  (initial_price * (1 + first_increase / 100) * 1.4 = initial_price * 1.82) →
  first_increase = 30 := by
sorry

end NUMINAMATH_CALUDE_tv_price_increase_l1564_156430


namespace NUMINAMATH_CALUDE_gina_money_to_mom_l1564_156476

theorem gina_money_to_mom (total : ℝ) (clothes_fraction : ℝ) (charity_fraction : ℝ) (kept : ℝ) :
  total = 400 →
  clothes_fraction = 1/8 →
  charity_fraction = 1/5 →
  kept = 170 →
  ∃ (mom_fraction : ℝ), 
    mom_fraction * total + clothes_fraction * total + charity_fraction * total + kept = total ∧
    mom_fraction = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_gina_money_to_mom_l1564_156476


namespace NUMINAMATH_CALUDE_spider_human_leg_ratio_l1564_156406

/-- The number of legs a spider has -/
def spider_legs : ℕ := 8

/-- The number of legs a human has -/
def human_legs : ℕ := 2

/-- The ratio of spider legs to human legs -/
def leg_ratio : ℚ := spider_legs / human_legs

/-- Theorem: The ratio of spider legs to human legs is 4 -/
theorem spider_human_leg_ratio : leg_ratio = 4 := by
  sorry

end NUMINAMATH_CALUDE_spider_human_leg_ratio_l1564_156406


namespace NUMINAMATH_CALUDE_special_sequence_theorem_l1564_156403

/-- A sequence satisfying certain properties -/
def SpecialSequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  c > 1 ∧
  a 1 = 1 ∧
  a 2 = 2 ∧
  (∀ m n, a (m * n) = a m * a n) ∧
  (∀ m n, a (m + n) ≤ c * (a m + a n))

/-- The main theorem: if a sequence satisfies the SpecialSequence properties,
    then a_n = n for all natural numbers n -/
theorem special_sequence_theorem (a : ℕ → ℝ) (c : ℝ) 
    (h : SpecialSequence a c) : ∀ n : ℕ, a n = n := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_theorem_l1564_156403


namespace NUMINAMATH_CALUDE_overlapping_rectangles_perimeter_l1564_156474

/-- The perimeter of a shape formed by two overlapping rectangles -/
theorem overlapping_rectangles_perimeter :
  ∀ (length width : ℝ),
  length = 7 →
  width = 3 →
  (2 * (length + width)) * 2 - 2 * width = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_overlapping_rectangles_perimeter_l1564_156474


namespace NUMINAMATH_CALUDE_smallest_pair_with_six_coins_l1564_156497

/-- Represents the value of a coin in half-pennies -/
inductive Coin : Nat → Type where
  | halfpenny : Coin 1
  | penny : Coin 2
  | threepence : Coin 6
  | fourpence : Coin 8
  | sixpence : Coin 12
  | shilling : Coin 24

/-- Checks if an amount can be represented with exactly 6 coins -/
def representableWithSixCoins (amount : Nat) : Prop :=
  ∃ (c₁ c₂ c₃ c₄ c₅ c₆ : Nat),
    (∃ (coin₁ : Coin c₁) (coin₂ : Coin c₂) (coin₃ : Coin c₃)
        (coin₄ : Coin c₄) (coin₅ : Coin c₅) (coin₆ : Coin c₆),
      c₁ + c₂ + c₃ + c₄ + c₅ + c₆ = amount)

/-- The main theorem to prove -/
theorem smallest_pair_with_six_coins :
  ∀ (a b : Nat),
    a < 60 ∧ b < 60 ∧ a < b ∧
    representableWithSixCoins a ∧
    representableWithSixCoins b ∧
    representableWithSixCoins (a + b) →
    a ≥ 23 ∧ b ≥ 47 :=
sorry

end NUMINAMATH_CALUDE_smallest_pair_with_six_coins_l1564_156497


namespace NUMINAMATH_CALUDE_max_hiking_time_l1564_156423

/-- Calculates the maximum hiking time for Violet and her dog given their water consumption rates and the total water carried. -/
theorem max_hiking_time (violet_rate : ℝ) (dog_rate : ℝ) (total_water : ℝ) :
  violet_rate = 800 →
  dog_rate = 400 →
  total_water = 4800 →
  (total_water / (violet_rate + dog_rate) : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_hiking_time_l1564_156423


namespace NUMINAMATH_CALUDE_g_five_times_one_l1564_156410

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x + 2 else 3 * x + 1

theorem g_five_times_one : g (g (g (g (g 1)))) = 12 := by
  sorry

end NUMINAMATH_CALUDE_g_five_times_one_l1564_156410


namespace NUMINAMATH_CALUDE_chandler_can_buy_bike_l1564_156475

/-- The cost of the mountain bike in dollars -/
def bike_cost : ℕ := 500

/-- The total birthday money Chandler received in dollars -/
def birthday_money : ℕ := 50 + 35 + 15

/-- Chandler's weekly earnings from the paper route in dollars -/
def weekly_earnings : ℕ := 16

/-- The number of weeks required to save enough money for the bike -/
def weeks_to_save : ℕ := 25

/-- Theorem stating that Chandler can buy the bike after saving for 25 weeks -/
theorem chandler_can_buy_bike : 
  birthday_money + weekly_earnings * weeks_to_save = bike_cost :=
sorry

end NUMINAMATH_CALUDE_chandler_can_buy_bike_l1564_156475
