import Mathlib

namespace NUMINAMATH_CALUDE_distance_between_points_l2603_260316

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 3)
  let p2 : ℝ × ℝ := (-2, -2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2603_260316


namespace NUMINAMATH_CALUDE_books_loaned_out_l2603_260300

/-- Proves that the number of books loaned out is 50 given the initial and final book counts and return rate -/
theorem books_loaned_out (initial_books : ℕ) (final_books : ℕ) (return_rate : ℚ) : 
  initial_books = 75 → final_books = 65 → return_rate = 4/5 → 
  (initial_books - final_books) / (1 - return_rate) = 50 := by
  sorry

end NUMINAMATH_CALUDE_books_loaned_out_l2603_260300


namespace NUMINAMATH_CALUDE_intersection_empty_union_real_l2603_260331

-- Define sets A and B
def A (a : ℝ) := {x : ℝ | 2*a ≤ x ∧ x ≤ a + 3}
def B := {x : ℝ | x < -1 ∨ x > 1}

-- Theorem for part I
theorem intersection_empty (a : ℝ) : A a ∩ B = ∅ ↔ a > 3 := by sorry

-- Theorem for part II
theorem union_real (a : ℝ) : A a ∪ B = Set.univ ↔ -2 ≤ a ∧ a ≤ -1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_union_real_l2603_260331


namespace NUMINAMATH_CALUDE_quadratic_always_two_roots_l2603_260306

theorem quadratic_always_two_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (∀ x : ℝ, x^2 - m*x + m - 2 = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_two_roots_l2603_260306


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l2603_260366

theorem solve_quadratic_equation (B : ℝ) :
  5 * B^2 + 5 = 30 → B = Real.sqrt 5 ∨ B = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l2603_260366


namespace NUMINAMATH_CALUDE_power_function_through_point_l2603_260320

theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^a) → f 2 = 8 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2603_260320


namespace NUMINAMATH_CALUDE_losing_ticket_probability_l2603_260339

/-- Given the odds of drawing a winning ticket are 5:8, 
    the probability of drawing a losing ticket is 8/13 -/
theorem losing_ticket_probability (winning_odds : Rat) 
  (h : winning_odds = 5 / 8) : 
  (1 : Rat) - winning_odds * (13 : Rat) / ((5 : Rat) + (8 : Rat)) = 8 / 13 :=
sorry

end NUMINAMATH_CALUDE_losing_ticket_probability_l2603_260339


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2603_260360

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation z(1+i³) = i
def equation (z : ℂ) : Prop := z * (1 + i^3) = i

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2603_260360


namespace NUMINAMATH_CALUDE_power_calculation_l2603_260335

theorem power_calculation : (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2603_260335


namespace NUMINAMATH_CALUDE_polynomial_roots_l2603_260374

theorem polynomial_roots (a b c : ℝ) : 
  (∀ x : ℝ, x^5 + 4*x^4 + a*x = b*x^2 + 4*c ↔ x = 2 ∨ x = -2) ↔ 
  (a = -16 ∧ b = 48 ∧ c = -32) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2603_260374


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l2603_260390

/-- A boat travels on a river with a current. -/
structure Boat :=
  (speed : ℝ)  -- Speed of the boat in still water in km/h

/-- A river with a current. -/
structure River :=
  (current : ℝ)  -- Speed of the river current in km/h

/-- The distance traveled by a boat on a river in one hour. -/
def distanceTraveled (b : Boat) (r : River) (withCurrent : Bool) : ℝ :=
  if withCurrent then b.speed + r.current else b.speed - r.current

theorem boat_distance_along_stream 
  (b : Boat) 
  (r : River) 
  (h1 : b.speed = 8) 
  (h2 : distanceTraveled b r false = 5) : 
  distanceTraveled b r true = 11 := by
  sorry

#check boat_distance_along_stream

end NUMINAMATH_CALUDE_boat_distance_along_stream_l2603_260390


namespace NUMINAMATH_CALUDE_total_stickers_l2603_260376

theorem total_stickers (stickers_per_page : ℕ) (total_pages : ℕ) : 
  stickers_per_page = 10 → total_pages = 22 → stickers_per_page * total_pages = 220 := by
sorry

end NUMINAMATH_CALUDE_total_stickers_l2603_260376


namespace NUMINAMATH_CALUDE_a_divides_iff_k_divides_l2603_260373

/-- Definition of a_n as the integer consisting of n repetitions of the digit 1 in base 10 -/
def a (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Theorem stating that a_k divides a_l if and only if k divides l -/
theorem a_divides_iff_k_divides (k l : ℕ) (h : k ≥ 1) :
  (a k ∣ a l) ↔ k ∣ l :=
by sorry

end NUMINAMATH_CALUDE_a_divides_iff_k_divides_l2603_260373


namespace NUMINAMATH_CALUDE_triangle_inequality_l2603_260317

/-- A complete graph K_n with n vertices, where each edge is colored either red, green, or blue. -/
structure ColoredCompleteGraph (n : ℕ) where
  n_ge_3 : n ≥ 3

/-- The number of triangles in K_n with all edges of the same color. -/
def monochromatic_triangles (G : ColoredCompleteGraph n) : ℕ := sorry

/-- The number of triangles in K_n with all edges of different colors. -/
def trichromatic_triangles (G : ColoredCompleteGraph n) : ℕ := sorry

/-- Theorem stating the relationship between monochromatic and trichromatic triangles. -/
theorem triangle_inequality (G : ColoredCompleteGraph n) :
  trichromatic_triangles G ≤ 2 * monochromatic_triangles G + n * (n - 1) / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2603_260317


namespace NUMINAMATH_CALUDE_jerry_butterflies_l2603_260355

/-- The number of butterflies Jerry originally had -/
def original_butterflies : ℕ := 93

/-- The number of butterflies Jerry let go -/
def butterflies_let_go : ℕ := 11

/-- The number of butterflies Jerry has left -/
def butterflies_left : ℕ := 82

/-- Theorem: Jerry originally had 93 butterflies -/
theorem jerry_butterflies : original_butterflies = butterflies_let_go + butterflies_left := by
  sorry

end NUMINAMATH_CALUDE_jerry_butterflies_l2603_260355


namespace NUMINAMATH_CALUDE_work_completion_time_l2603_260367

theorem work_completion_time (original_laborers : ℕ) (absent_laborers : ℕ) (actual_days : ℕ) : 
  original_laborers = 20 → 
  absent_laborers = 5 → 
  actual_days = 20 → 
  ∃ (original_days : ℕ), 
    original_days * original_laborers = actual_days * (original_laborers - absent_laborers) ∧ 
    original_days = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2603_260367


namespace NUMINAMATH_CALUDE_total_sheets_l2603_260356

-- Define the number of brown and yellow sheets
def brown_sheets : ℕ := 28
def yellow_sheets : ℕ := 27

-- Theorem to prove
theorem total_sheets : brown_sheets + yellow_sheets = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_sheets_l2603_260356


namespace NUMINAMATH_CALUDE_horner_method_v2_l2603_260314

def f (x : ℝ) : ℝ := 4*x^4 + 3*x^3 - 6*x^2 + x - 1

def horner_v0 : ℝ := 4

def horner_v1 (x : ℝ) : ℝ := horner_v0 * x + 3

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x - 6

theorem horner_method_v2 : horner_v2 (-1) = -5 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v2_l2603_260314


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2603_260389

theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  marked_price = 65 ∧ 
  discount_rate = 0.05 ∧ 
  profit_rate = 0.30 →
  ∃ (cost_price : ℝ),
    cost_price = 47.50 ∧
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2603_260389


namespace NUMINAMATH_CALUDE_equation_solutions_l2603_260377

theorem equation_solutions :
  (∃ x : ℝ, 7 * x + 2 * (3 * x - 3) = 20 ∧ x = 2) ∧
  (∃ x : ℝ, (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 3 ∧ x = 67 / 23) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2603_260377


namespace NUMINAMATH_CALUDE_rectangle_cut_perimeter_l2603_260302

/-- Given a rectangle with perimeter 10, prove that when cut twice parallel to its
    length and width to form 9 smaller rectangles, the total perimeter of these
    9 rectangles is 30. -/
theorem rectangle_cut_perimeter (a b : ℝ) : 
  (2 * (a + b) = 10) →  -- Perimeter of original rectangle
  (∃ x y z w : ℝ, 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧  -- Cuts are positive
    x + y + z = a ∧ w + y + z = b) →  -- Cuts divide length and width
  (2 * (a + b) + 4 * (a + b) = 30) :=  -- Total perimeter after cuts
by sorry

end NUMINAMATH_CALUDE_rectangle_cut_perimeter_l2603_260302


namespace NUMINAMATH_CALUDE_distance_ratio_of_cars_l2603_260329

-- Define the speeds and travel times for both cars
def speed_A : ℝ := 50
def time_A : ℝ := 8
def speed_B : ℝ := 25
def time_B : ℝ := 4

-- Define a function to calculate distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem distance_ratio_of_cars :
  (distance speed_A time_A) / (distance speed_B time_B) = 4 := by
  sorry


end NUMINAMATH_CALUDE_distance_ratio_of_cars_l2603_260329


namespace NUMINAMATH_CALUDE_marie_erasers_l2603_260399

theorem marie_erasers (initial final lost : ℕ) 
  (h1 : lost = 42)
  (h2 : final = 53)
  (h3 : initial = final + lost) : initial = 95 := by
  sorry

end NUMINAMATH_CALUDE_marie_erasers_l2603_260399


namespace NUMINAMATH_CALUDE_min_races_for_fifty_horses_l2603_260351

/-- Represents the minimum number of races needed to find the top k fastest horses
    from a total of n horses, racing at most m horses at a time. -/
def min_races (n m k : ℕ) : ℕ :=
  sorry

/-- The theorem stating that for 50 horses, racing 3 at a time,
    19 races are needed to find the top 5 fastest horses. -/
theorem min_races_for_fifty_horses :
  min_races 50 3 5 = 19 := by sorry

end NUMINAMATH_CALUDE_min_races_for_fifty_horses_l2603_260351


namespace NUMINAMATH_CALUDE_inequality_proof_l2603_260353

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2603_260353


namespace NUMINAMATH_CALUDE_football_cards_per_box_l2603_260347

theorem football_cards_per_box (basketball_boxes : ℕ) (basketball_cards_per_box : ℕ) (total_cards : ℕ) :
  basketball_boxes = 9 →
  basketball_cards_per_box = 15 →
  total_cards = 255 →
  let football_boxes := basketball_boxes - 3
  let basketball_cards := basketball_boxes * basketball_cards_per_box
  let football_cards := total_cards - basketball_cards
  football_cards / football_boxes = 20 := by
sorry

end NUMINAMATH_CALUDE_football_cards_per_box_l2603_260347


namespace NUMINAMATH_CALUDE_price_crossover_year_l2603_260371

def price_X (year : ℕ) : ℚ :=
  4.20 + 0.45 * (year - 2001 : ℚ)

def price_Y (year : ℕ) : ℚ :=
  6.30 + 0.20 * (year - 2001 : ℚ)

theorem price_crossover_year : 
  (∀ y : ℕ, y < 2010 → price_X y ≤ price_Y y) ∧ 
  price_X 2010 > price_Y 2010 :=
by sorry

end NUMINAMATH_CALUDE_price_crossover_year_l2603_260371


namespace NUMINAMATH_CALUDE_missing_number_implies_next_prime_l2603_260328

/-- Definition of the table entry function -/
def table_entry (r s : ℕ) : ℕ := r * s - (r + s)

/-- Theorem: If n > 3 is not in the table, then n + 1 is prime -/
theorem missing_number_implies_next_prime (n : ℕ) (h1 : n > 3) 
  (h2 : ∀ r s, r ≥ 3 → s ≥ 3 → table_entry r s ≠ n) : 
  Nat.Prime (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_implies_next_prime_l2603_260328


namespace NUMINAMATH_CALUDE_point_movement_l2603_260388

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Move a point left by a given number of units -/
def moveLeft (p : Point) (units : ℝ) : Point :=
  ⟨p.x - units, p.y⟩

/-- Move a point up by a given number of units -/
def moveUp (p : Point) (units : ℝ) : Point :=
  ⟨p.x, p.y + units⟩

theorem point_movement :
  let A : Point := ⟨2, -1⟩
  let B : Point := moveUp (moveLeft A 3) 4
  B.x = -1 ∧ B.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_l2603_260388


namespace NUMINAMATH_CALUDE_function_parity_l2603_260345

-- Define the property of the function
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

-- Define even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem function_parity (f : ℝ → ℝ) (h : satisfies_property f) :
  (is_even f ∨ is_odd f) :=
sorry

end NUMINAMATH_CALUDE_function_parity_l2603_260345


namespace NUMINAMATH_CALUDE_weaving_increase_proof_l2603_260362

/-- Represents the daily increase in cloth production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the number of days in a month -/
def days_in_month : ℕ := 30

/-- Represents the initial daily production in meters -/
def initial_production : ℚ := 5

/-- Represents the total production in meters for the month -/
def total_production : ℚ := 390

/-- Theorem stating that given the initial conditions, the daily increase in production is 16/29 meters -/
theorem weaving_increase_proof :
  initial_production * days_in_month + 
  (days_in_month * (days_in_month - 1) / 2) * daily_increase = 
  total_production :=
by
  sorry


end NUMINAMATH_CALUDE_weaving_increase_proof_l2603_260362


namespace NUMINAMATH_CALUDE_min_value_expression_l2603_260326

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a^2 + 4*a + 2)*(b^2 + 4*b + 2)*(c^2 + 4*c + 2)) / (a*b*c) ≥ 512 ∧
  (∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    ((a₀^2 + 4*a₀ + 2)*(b₀^2 + 4*b₀ + 2)*(c₀^2 + 4*c₀ + 2)) / (a₀*b₀*c₀) = 512) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2603_260326


namespace NUMINAMATH_CALUDE_maria_towels_l2603_260350

/-- The number of towels Maria ended up with after shopping and giving some to her mother. -/
def towels_maria_kept (green_towels white_towels towels_given : ℕ) : ℕ :=
  green_towels + white_towels - towels_given

/-- Theorem stating that Maria ended up with 22 towels -/
theorem maria_towels :
  towels_maria_kept 35 21 34 = 22 := by
  sorry

end NUMINAMATH_CALUDE_maria_towels_l2603_260350


namespace NUMINAMATH_CALUDE_randy_practice_hours_l2603_260321

/-- Calculates the number of hours per day Randy needs to practice piano to become an expert --/
def hours_per_day_to_expert (current_age : ℕ) (target_age : ℕ) (practice_days_per_week : ℕ) (vacation_weeks : ℕ) (hours_to_expert : ℕ) : ℚ :=
  let years_to_practice := target_age - current_age
  let weeks_per_year := 52
  let practice_weeks := weeks_per_year - vacation_weeks
  let practice_days_per_year := practice_weeks * practice_days_per_week
  let total_practice_days := years_to_practice * practice_days_per_year
  hours_to_expert / total_practice_days

/-- Theorem stating that Randy needs to practice 5 hours per day to become a piano expert --/
theorem randy_practice_hours :
  hours_per_day_to_expert 12 20 5 2 10000 = 5 := by
  sorry

end NUMINAMATH_CALUDE_randy_practice_hours_l2603_260321


namespace NUMINAMATH_CALUDE_ratio_problem_l2603_260313

theorem ratio_problem (w x y z : ℝ) 
  (h1 : w / x = 1 / 3) 
  (h2 : w / y = 2 / 3) 
  (h3 : w / z = 3 / 5) 
  (hw : w ≠ 0) : 
  (x + y) / z = 27 / 10 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2603_260313


namespace NUMINAMATH_CALUDE_lucy_crayons_count_l2603_260368

/-- The number of crayons Willy has -/
def willys_crayons : ℕ := 5092

/-- The difference between Willy's and Lucy's crayons -/
def difference : ℕ := 1121

/-- The number of crayons Lucy has -/
def lucys_crayons : ℕ := willys_crayons - difference

theorem lucy_crayons_count : lucys_crayons = 3971 := by
  sorry

end NUMINAMATH_CALUDE_lucy_crayons_count_l2603_260368


namespace NUMINAMATH_CALUDE_no_21_length2_segments_in_10x10_grid_l2603_260398

/-- Represents a grid skeleton -/
structure GridSkeleton :=
  (size : ℕ)

/-- Represents the division of a grid skeleton into angle pieces and segments of length 2 -/
structure GridDivision :=
  (grid : GridSkeleton)
  (length2_segments : ℕ)

/-- Theorem stating that a 10x10 grid skeleton cannot have exactly 21 segments of length 2 -/
theorem no_21_length2_segments_in_10x10_grid :
  ∀ (d : GridDivision), d.grid.size = 10 → d.length2_segments ≠ 21 := by
  sorry

end NUMINAMATH_CALUDE_no_21_length2_segments_in_10x10_grid_l2603_260398


namespace NUMINAMATH_CALUDE_max_missed_percentage_is_five_percent_l2603_260346

/-- The maximum percentage of school days a senior can miss and still skip final exams -/
def max_missed_percentage (total_days : ℕ) (missed_days : ℕ) (additional_days : ℕ) : ℚ :=
  (missed_days + additional_days : ℚ) / total_days * 100

/-- Proof that the maximum percentage of school days a senior can miss is 5% -/
theorem max_missed_percentage_is_five_percent
  (total_days : ℕ) (missed_days : ℕ) (additional_days : ℕ)
  (h1 : total_days = 180)
  (h2 : missed_days = 6)
  (h3 : additional_days = 3) :
  max_missed_percentage total_days missed_days additional_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_missed_percentage_is_five_percent_l2603_260346


namespace NUMINAMATH_CALUDE_jewels_gain_is_3_25_l2603_260385

/-- Calculates Jewel's total gain from selling magazines --/
def jewels_gain (num_magazines : ℕ) 
                (cost_per_magazine : ℚ) 
                (regular_price : ℚ) 
                (discount_percent : ℚ) 
                (num_regular_price : ℕ) : ℚ :=
  let total_cost := num_magazines * cost_per_magazine
  let revenue_regular := num_regular_price * regular_price
  let discounted_price := regular_price * (1 - discount_percent)
  let revenue_discounted := (num_magazines - num_regular_price) * discounted_price
  let total_revenue := revenue_regular + revenue_discounted
  total_revenue - total_cost

theorem jewels_gain_is_3_25 : 
  jewels_gain 10 3 (7/2) (1/10) 5 = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_jewels_gain_is_3_25_l2603_260385


namespace NUMINAMATH_CALUDE_chord_equation_l2603_260318

/-- Given a circle and a chord, prove the equation of the chord --/
theorem chord_equation (P Q : ℝ × ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 9 → (x - 1)^2 + (y - 2)^2 = (x - P.1)^2 + (y - P.2)^2) →
  (∀ (x y : ℝ), x^2 + y^2 = 9 → (x - 1)^2 + (y - 2)^2 = (x - Q.1)^2 + (y - Q.2)^2) →
  (P.1 + Q.1) / 2 = 1 →
  (P.2 + Q.2) / 2 = 2 →
  ∃ (k : ℝ), ∀ (x y : ℝ), (y - P.2) = k * (x - P.1) ∧ (y - Q.2) = k * (x - Q.1) →
    x + 2*y - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l2603_260318


namespace NUMINAMATH_CALUDE_sixth_face_configuration_l2603_260357

structure Cube where
  size : Nat
  black_cubes : Nat
  white_cubes : Nat

structure Face where
  center_white : Nat
  edge_white : Nat
  corner_white : Nat

def valid_face (f : Face) : Prop :=
  f.center_white = 1 ∧ f.edge_white = 2 ∧ f.corner_white = 1

def cube_configuration (c : Cube) (known_faces : List Face) : Prop :=
  c.size = 3 ∧
  c.black_cubes = 15 ∧
  c.white_cubes = 12 ∧
  known_faces.length = 5

theorem sixth_face_configuration
  (c : Cube)
  (known_faces : List Face)
  (h_config : cube_configuration c known_faces) :
  ∃ (sixth_face : Face), valid_face sixth_face :=
by sorry

end NUMINAMATH_CALUDE_sixth_face_configuration_l2603_260357


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2603_260332

/-- Represents a repeating decimal with a two-digit repeating sequence -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

/-- The repeating decimal 0.474747... -/
def x : ℚ := RepeatingDecimal 4 7

/-- The sum of the numerator and denominator of a fraction -/
def sumNumeratorDenominator (q : ℚ) : ℕ :=
  q.num.natAbs + q.den

theorem repeating_decimal_sum : sumNumeratorDenominator x = 146 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2603_260332


namespace NUMINAMATH_CALUDE_airplane_distance_difference_l2603_260330

/-- Theorem: Distance difference for an airplane flying with and against wind -/
theorem airplane_distance_difference (a : ℝ) : 
  let windless_speed : ℝ := a
  let wind_speed : ℝ := 20
  let time_without_wind : ℝ := 4
  let time_against_wind : ℝ := 3
  windless_speed * time_without_wind - (windless_speed - wind_speed) * time_against_wind = a + 60 := by
  sorry

end NUMINAMATH_CALUDE_airplane_distance_difference_l2603_260330


namespace NUMINAMATH_CALUDE_z_properties_l2603_260327

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := 2 * m + (4 - m^2) * Complex.I

/-- z lies on the imaginary axis -/
def on_imaginary_axis (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- z lies in the first or third quadrant -/
def in_first_or_third_quadrant (z : ℂ) : Prop := z.re * z.im < 0

theorem z_properties (m : ℝ) :
  (on_imaginary_axis (z m) ↔ m = 0) ∧
  (in_first_or_third_quadrant (z m) ↔ m > 2 ∨ (-2 < m ∧ m < 0)) := by
  sorry

end NUMINAMATH_CALUDE_z_properties_l2603_260327


namespace NUMINAMATH_CALUDE_simplify_expression_l2603_260381

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x - 2) = 4*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2603_260381


namespace NUMINAMATH_CALUDE_train_crossing_time_l2603_260337

/-- Proves that the time taken for a train to cross a bridge is 20 seconds -/
theorem train_crossing_time (bridge_length : ℝ) (train_length : ℝ) (train_speed : ℝ) :
  bridge_length = 180 →
  train_length = 120 →
  train_speed = 15 →
  (bridge_length + train_length) / train_speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2603_260337


namespace NUMINAMATH_CALUDE_ice_melting_problem_l2603_260308

theorem ice_melting_problem (V : ℝ) : 
  V > 0 → 
  ((1 - 3/4) * (1 - 3/4) * V = 0.75) → 
  V = 12 := by
  sorry

end NUMINAMATH_CALUDE_ice_melting_problem_l2603_260308


namespace NUMINAMATH_CALUDE_auston_taller_than_emma_l2603_260340

def inch_to_cm (inches : ℝ) : ℝ := inches * 2.54

def height_difference_cm (auston_height_inch : ℝ) (emma_height_inch : ℝ) : ℝ :=
  inch_to_cm auston_height_inch - inch_to_cm emma_height_inch

theorem auston_taller_than_emma : 
  height_difference_cm 60 54 = 15.24 := by sorry

end NUMINAMATH_CALUDE_auston_taller_than_emma_l2603_260340


namespace NUMINAMATH_CALUDE_gcf_of_40_and_14_l2603_260333

theorem gcf_of_40_and_14 :
  let n : ℕ := 40
  let m : ℕ := 14
  let lcm_nm : ℕ := 56
  Nat.lcm n m = lcm_nm →
  Nat.gcd n m = 10 := by
sorry

end NUMINAMATH_CALUDE_gcf_of_40_and_14_l2603_260333


namespace NUMINAMATH_CALUDE_h_neither_even_nor_odd_l2603_260303

-- Define an even function
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- Define the function h
def h (g : ℝ → ℝ) (x : ℝ) : ℝ := g (g x + x)

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem h_neither_even_nor_odd (g : ℝ → ℝ) (hg : even_function g) :
  ¬(is_even (h g)) ∧ ¬(is_odd (h g)) :=
sorry

end NUMINAMATH_CALUDE_h_neither_even_nor_odd_l2603_260303


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2603_260372

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 = 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = x + 3}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2603_260372


namespace NUMINAMATH_CALUDE_solution_difference_l2603_260304

theorem solution_difference (m n : ℝ) : 
  (m - 4) * (m + 4) = 24 * m - 96 →
  (n - 4) * (n + 4) = 24 * n - 96 →
  m ≠ n →
  m > n →
  m - n = 16 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l2603_260304


namespace NUMINAMATH_CALUDE_triangle_construction_valid_l2603_260322

/-- A triangle can be constructed with perimeter k, one side c, and angle difference δ
    between angles opposite the other two sides if and only if 2c < k. -/
theorem triangle_construction_valid (k c : ℝ) (δ : ℝ) :
  (∃ (a b : ℝ) (α β γ : ℝ),
    a + b + c = k ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    α + β + γ = π ∧
    α - β = δ ∧
    0 < α ∧ α < π ∧
    0 < β ∧ β < π ∧
    0 < γ ∧ γ < π) ↔
  2 * c < k :=
by sorry


end NUMINAMATH_CALUDE_triangle_construction_valid_l2603_260322


namespace NUMINAMATH_CALUDE_rain_probability_l2603_260359

theorem rain_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 5) :
  1 - (1 - p)^n = 1023/1024 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l2603_260359


namespace NUMINAMATH_CALUDE_hyperbola_focus_parameter_l2603_260396

/-- Given a hyperbola with equation y²/m - x²/9 = 1 and a focus at (0, 5),
    prove that m = 16. -/
theorem hyperbola_focus_parameter (m : ℝ) : 
  (∀ x y : ℝ, y^2/m - x^2/9 = 1 → (x = 0 ∧ y = 5) → m = 16) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_parameter_l2603_260396


namespace NUMINAMATH_CALUDE_initial_customers_l2603_260384

theorem initial_customers (remaining : ℕ) (left : ℕ) (initial : ℕ) : 
  remaining = 12 → left = 9 → initial = remaining + left → initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_customers_l2603_260384


namespace NUMINAMATH_CALUDE_intersection_A_B_l2603_260363

-- Define set A
def A : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (x + 1)}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2603_260363


namespace NUMINAMATH_CALUDE_max_a_value_l2603_260378

def is_lattice_point (x y : ℤ) : Prop := True

def passes_through_lattice_point (m : ℚ) (b : ℤ) : Prop :=
  ∃ x y : ℤ, is_lattice_point x y ∧ 0 < x ∧ x ≤ 200 ∧ y = m * x + b

theorem max_a_value :
  let a : ℚ := 68 / 201
  ∀ m : ℚ, 1/3 < m → m < a →
    ¬(passes_through_lattice_point m 3 ∨ passes_through_lattice_point m 1) ∧
    ∀ a' : ℚ, a < a' →
      ∃ m : ℚ, 1/3 < m ∧ m < a' ∧
        (passes_through_lattice_point m 3 ∨ passes_through_lattice_point m 1) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l2603_260378


namespace NUMINAMATH_CALUDE_inequality_proof_l2603_260391

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2603_260391


namespace NUMINAMATH_CALUDE_remainder_thirteen_power_fiftyone_mod_five_l2603_260394

theorem remainder_thirteen_power_fiftyone_mod_five :
  13^51 % 5 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_thirteen_power_fiftyone_mod_five_l2603_260394


namespace NUMINAMATH_CALUDE_red_window_exchange_equations_l2603_260301

/-- Represents the relationship between online and offline booth transactions -/
theorem red_window_exchange_equations 
  (x y : ℝ)  -- Total transaction amounts for online (x) and offline (y) booths
  (online_booths : ℕ := 44)  -- Number of online booths
  (offline_booths : ℕ := 71)  -- Number of offline booths
  (h1 : y - 7 * x = 1.8)  -- Relationship between total transaction amounts
  (h2 : y / offline_booths - x / online_booths = 0.3)  -- Difference in average transaction amounts
  : ∃ (system : ℝ × ℝ → Prop), 
    system (x, y) ∧ 
    (∀ (a b : ℝ), system (a, b) ↔ (b - 7 * a = 1.8 ∧ b / offline_booths - a / online_booths = 0.3)) :=
by
  sorry


end NUMINAMATH_CALUDE_red_window_exchange_equations_l2603_260301


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l2603_260310

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.tan x = 1) ↔ (∀ x : ℝ, Real.tan x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l2603_260310


namespace NUMINAMATH_CALUDE_charity_boxes_theorem_l2603_260375

/-- Calculates the total number of boxes a charity can pack given initial conditions --/
theorem charity_boxes_theorem (initial_boxes : ℕ) (food_cost : ℕ) (supplies_cost : ℕ) (donation_multiplier : ℕ) : 
  initial_boxes = 400 → 
  food_cost = 80 → 
  supplies_cost = 165 → 
  donation_multiplier = 4 → 
  (initial_boxes + (donation_multiplier * initial_boxes * (food_cost + supplies_cost)) / (food_cost + supplies_cost) : ℕ) = 2000 := by
  sorry

#check charity_boxes_theorem

end NUMINAMATH_CALUDE_charity_boxes_theorem_l2603_260375


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2603_260349

/-- The imaginary part of (3-2i)/(1-i) is 1/2 -/
theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (3 - 2*I) / (1 - I)
  Complex.im z = 1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2603_260349


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2603_260382

theorem concentric_circles_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 5 * (π * a^2) → a / b = 1 / Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l2603_260382


namespace NUMINAMATH_CALUDE_milk_can_problem_l2603_260342

theorem milk_can_problem :
  ∃! (x y : ℕ), 10 * x + 17 * y = 206 :=
by sorry

end NUMINAMATH_CALUDE_milk_can_problem_l2603_260342


namespace NUMINAMATH_CALUDE_function_inequality_constraint_l2603_260392

theorem function_inequality_constraint (x a : ℝ) : 
  x > 0 → (2 * x + 1 > a * x) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_constraint_l2603_260392


namespace NUMINAMATH_CALUDE_lakers_win_series_in_five_l2603_260311

def probability_win_series_in_five (p : ℚ) : ℚ :=
  let q := 1 - p
  6 * q^2 * p^2 * q

theorem lakers_win_series_in_five :
  probability_win_series_in_five (3/4) = 27/512 := by
  sorry

end NUMINAMATH_CALUDE_lakers_win_series_in_five_l2603_260311


namespace NUMINAMATH_CALUDE_power_calculation_l2603_260315

theorem power_calculation : 4^2011 * (-0.25)^2010 - 1 = 3 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l2603_260315


namespace NUMINAMATH_CALUDE_spring_math_camp_inconsistency_l2603_260380

theorem spring_math_camp_inconsistency : 
  ¬ ∃ (b g : ℕ), 11 * b + 7 * g = 4046 := by
  sorry

end NUMINAMATH_CALUDE_spring_math_camp_inconsistency_l2603_260380


namespace NUMINAMATH_CALUDE_parabola_p_value_l2603_260341

/-- A parabola with equation y^2 = 2px and directrix x = -2 has p = 4 -/
theorem parabola_p_value (y x p : ℝ) : 
  (∀ y x, y^2 = 2*p*x) →  -- Condition 1: Parabola equation
  (x = -2)               -- Condition 2: Directrix equation
  → p = 4 :=             -- Conclusion: p = 4
by sorry

end NUMINAMATH_CALUDE_parabola_p_value_l2603_260341


namespace NUMINAMATH_CALUDE_limit_S_2_pow_n_to_infinity_l2603_260365

/-- S(n) represents the sum of digits of n in base 10 -/
def S (n : ℕ) : ℕ := sorry

/-- Main theorem: The limit of S(2^n) as n approaches infinity is infinity -/
theorem limit_S_2_pow_n_to_infinity :
  ∀ M : ℕ, ∃ N : ℕ, ∀ n : ℕ, n ≥ N → S (2^n) > M :=
sorry

end NUMINAMATH_CALUDE_limit_S_2_pow_n_to_infinity_l2603_260365


namespace NUMINAMATH_CALUDE_coffee_consumption_theorem_l2603_260307

/-- Represents the relationship between coffee consumption, sleep, and work intensity -/
def coffee_relation (sleep : ℝ) (work : ℝ) (coffee : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ coffee * sleep * work = k

/-- Theorem stating the relationship between coffee consumption on two different days -/
theorem coffee_consumption_theorem (sleep_mon sleep_tue work_mon work_tue coffee_mon : ℝ) :
  sleep_mon = 8 →
  work_mon = 4 →
  coffee_mon = 1 →
  sleep_tue = 5 →
  work_tue = 7 →
  coffee_relation sleep_mon work_mon coffee_mon →
  coffee_relation sleep_tue work_tue ((32 : ℝ) / 35) :=
by sorry

end NUMINAMATH_CALUDE_coffee_consumption_theorem_l2603_260307


namespace NUMINAMATH_CALUDE_abc_inequality_l2603_260386

theorem abc_inequality (a b c : ℝ) (ha : -1 ≤ a ∧ a ≤ 2) (hb : -1 ≤ b ∧ b ≤ 2) (hc : -1 ≤ c ∧ c ≤ 2) :
  a * b * c + 4 ≥ a * b + b * c + c * a := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l2603_260386


namespace NUMINAMATH_CALUDE_average_difference_l2603_260336

theorem average_difference (x : ℝ) : 
  (10 + 30 + 50) / 3 = (20 + 40 + x) / 3 + 8 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l2603_260336


namespace NUMINAMATH_CALUDE_hike_length_l2603_260309

/-- Represents a four-day hike with given conditions -/
structure FourDayHike where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  first_two_days : day1 + day2 = 24
  second_third_avg : (day2 + day3) / 2 = 15
  last_two_days : day3 + day4 = 32
  first_third_days : day1 + day3 = 28

/-- The total length of the hike is 56 miles -/
theorem hike_length (h : FourDayHike) : h.day1 + h.day2 + h.day3 + h.day4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_hike_length_l2603_260309


namespace NUMINAMATH_CALUDE_number_square_problem_l2603_260334

theorem number_square_problem : ∃! x : ℝ, x^2 + 95 = (x - 19)^2 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_square_problem_l2603_260334


namespace NUMINAMATH_CALUDE_indigo_restaurant_rating_l2603_260323

/-- Calculates the average star rating for a restaurant given the number of reviews for each star rating. -/
def averageStarRating (five_star : ℕ) (four_star : ℕ) (three_star : ℕ) (two_star : ℕ) : ℚ :=
  let total_stars := 5 * five_star + 4 * four_star + 3 * three_star + 2 * two_star
  let total_reviews := five_star + four_star + three_star + two_star
  (total_stars : ℚ) / total_reviews

/-- The average star rating for Indigo Restaurant is 4 stars. -/
theorem indigo_restaurant_rating :
  averageStarRating 6 7 4 1 = 4 := by
  sorry


end NUMINAMATH_CALUDE_indigo_restaurant_rating_l2603_260323


namespace NUMINAMATH_CALUDE_prob_more_heads_12_coins_l2603_260379

/-- The number of coins flipped -/
def n : ℕ := 12

/-- The probability of getting more heads than tails when flipping n coins -/
def prob_more_heads (n : ℕ) : ℚ :=
  1 / 2 - (n.choose (n / 2)) / (2 ^ n)

theorem prob_more_heads_12_coins : 
  prob_more_heads n = 793 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_prob_more_heads_12_coins_l2603_260379


namespace NUMINAMATH_CALUDE_smallest_with_16_divisors_exactly_16_divisors_210_smallest_positive_integer_with_16_divisors_l2603_260319

def number_of_divisors (n : ℕ) : ℕ :=
  (Nat.divisors n).card

theorem smallest_with_16_divisors : 
  ∀ n : ℕ, n > 0 → number_of_divisors n = 16 → n ≥ 210 :=
by
  sorry

theorem exactly_16_divisors_210 : number_of_divisors 210 = 16 :=
by
  sorry

theorem smallest_positive_integer_with_16_divisors : 
  ∀ n : ℕ, n > 0 → number_of_divisors n = 16 → n ≥ 210 ∧ number_of_divisors 210 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_with_16_divisors_exactly_16_divisors_210_smallest_positive_integer_with_16_divisors_l2603_260319


namespace NUMINAMATH_CALUDE_quadratic_equation_completing_square_l2603_260343

theorem quadratic_equation_completing_square (x : ℝ) :
  ∃ (q t : ℝ), (16 * x^2 - 32 * x - 512 = 0) ↔ ((x + q)^2 = t) ∧ t = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_completing_square_l2603_260343


namespace NUMINAMATH_CALUDE_triangle_one_two_two_l2603_260348

/-- Triangle inequality theorem for three sides --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_one_two_two :
  can_form_triangle 1 2 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_one_two_two_l2603_260348


namespace NUMINAMATH_CALUDE_triangle_side_length_l2603_260312

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  a = 2 →
  b = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2603_260312


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l2603_260393

-- Define the types for lines and planes
variable (L : Type) (P : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : L → P → Prop)
variable (parallel : L → P → Prop)
variable (planePerpendicular : P → P → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (l : L) (α β : P) 
  (h1 : perpendicular l α)
  (h2 : parallel l β) :
  planePerpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l2603_260393


namespace NUMINAMATH_CALUDE_vector_operation_l2603_260370

theorem vector_operation (a b : ℝ × ℝ) :
  a = (2, 2) → b = (-1, 3) → 2 • a - b = (5, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l2603_260370


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l2603_260383

theorem quadratic_form_h_value (a b c : ℝ) :
  (∃ (n k : ℝ), ∀ x, 5 * (a * x^2 + b * x + c) = n * (x - 5)^2 + k) →
  (∀ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l2603_260383


namespace NUMINAMATH_CALUDE_cubic_polynomial_negative_one_bound_l2603_260305

/-- A polynomial of degree 3 with three distinct positive roots -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  roots : Fin 3 → ℝ
  roots_positive : ∀ i, roots i > 0
  roots_distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  is_root : ∀ i, (roots i)^3 + a*(roots i)^2 + b*(roots i) - 1 = 0

/-- The polynomial P(x) = x^3 + ax^2 + bx - 1 -/
def P (poly : CubicPolynomial) (x : ℝ) : ℝ :=
  x^3 + poly.a * x^2 + poly.b * x - 1

theorem cubic_polynomial_negative_one_bound (poly : CubicPolynomial) : P poly (-1) < -8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_negative_one_bound_l2603_260305


namespace NUMINAMATH_CALUDE_division_problem_l2603_260344

theorem division_problem (dividend quotient remainder : ℕ) (divisor : ℚ) : 
  dividend = 12 → quotient = 9 → remainder = 8 → 
  dividend = (divisor * quotient) + remainder → 
  divisor = 4/9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2603_260344


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l2603_260369

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ∣ (2^n - 1) ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l2603_260369


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2603_260352

theorem cylinder_lateral_surface_area 
  (r h : ℝ) 
  (hr : r = 2) 
  (hh : h = 2) : 
  2 * Real.pi * r * h = 8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2603_260352


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2603_260361

theorem decimal_sum_to_fraction :
  (0.3 : ℚ) + 0.04 + 0.005 + 0.0006 + 0.00007 = 34567 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2603_260361


namespace NUMINAMATH_CALUDE_smallest_max_sum_l2603_260325

theorem smallest_max_sum (a b c d e : ℕ+) (h : a + b + c + d + e = 2020) :
  (∃ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (d + e))) ∧
   ∀ M' : ℕ, (∃ a' b' c' d' e' : ℕ+, 
     a' + b' + c' + d' + e' = 2020 ∧
     M' = max (a' + b') (max (b' + c') (max (c' + d') (d' + e')))) → M' ≥ M) ∧
  (∀ M : ℕ, (∃ a' b' c' d' e' : ℕ+, 
    a' + b' + c' + d' + e' = 2020 ∧
    M = max (a' + b') (max (b' + c') (max (c' + d') (d' + e')))) → M ≥ 674) :=
sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l2603_260325


namespace NUMINAMATH_CALUDE_annie_televisions_correct_l2603_260358

/-- The number of televisions Annie bought at a liquidation sale -/
def num_televisions : ℕ := 5

/-- The cost of each television -/
def television_cost : ℕ := 50

/-- The number of figurines Annie bought -/
def num_figurines : ℕ := 10

/-- The cost of each figurine -/
def figurine_cost : ℕ := 1

/-- The total amount Annie spent -/
def total_spent : ℕ := 260

/-- Theorem stating that the number of televisions Annie bought is correct -/
theorem annie_televisions_correct : 
  num_televisions * television_cost + num_figurines * figurine_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_annie_televisions_correct_l2603_260358


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2603_260354

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_nonzero : d ≠ 0
  a_1_eq_2 : a 1 = 2
  geometric : a 1 * a 4 = a 2 * a 2  -- a_1, a_2, a_4 form a geometric sequence

/-- The theorem stating the general term of the sequence -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2603_260354


namespace NUMINAMATH_CALUDE_license_plate_count_l2603_260324

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The number of possible letters -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 5

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 3

/-- The number of possible positions for the letter block -/
def block_positions : ℕ := digits_in_plate + 1

/-- The total number of distinct license plates -/
def total_plates : ℕ := block_positions * (num_digits ^ digits_in_plate) * (num_letters ^ letters_in_plate)

theorem license_plate_count : total_plates = 105456000 := by sorry

end NUMINAMATH_CALUDE_license_plate_count_l2603_260324


namespace NUMINAMATH_CALUDE_left_handed_women_percentage_l2603_260364

/-- Represents the population distribution in Smithtown -/
structure SmithtownPopulation where
  right_handed : ℕ
  left_handed : ℕ
  men : ℕ
  women : ℕ

/-- Conditions for Smithtown's population distribution -/
def valid_distribution (p : SmithtownPopulation) : Prop :=
  p.right_handed = 3 * p.left_handed ∧
  p.men = 3 * p.women / 2 ∧
  p.right_handed + p.left_handed = p.men + p.women ∧
  p.right_handed ≥ p.men

/-- Theorem: In a valid Smithtown population distribution, 
    left-handed women constitute 25% of the total population -/
theorem left_handed_women_percentage 
  (p : SmithtownPopulation) 
  (h : valid_distribution p) : 
  (p.left_handed : ℚ) / (p.right_handed + p.left_handed : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_women_percentage_l2603_260364


namespace NUMINAMATH_CALUDE_marks_score_is_46_l2603_260338

def highest_score : ℕ := 98
def score_range : ℕ := 75

def least_score : ℕ := highest_score - score_range

def marks_score : ℕ := 2 * least_score

theorem marks_score_is_46 : marks_score = 46 := by
  sorry

end NUMINAMATH_CALUDE_marks_score_is_46_l2603_260338


namespace NUMINAMATH_CALUDE_largest_n_for_arithmetic_sequences_l2603_260397

theorem largest_n_for_arithmetic_sequences (a b : ℕ → ℕ) : 
  (∀ n, ∃ x y : ℤ, a n = 1 + (n - 1) * x ∧ b n = 1 + (n - 1) * y) →  -- arithmetic sequences
  (a 1 = 1 ∧ b 1 = 1) →  -- first terms are 1
  (a 2 ≤ b 2) →  -- a_2 ≤ b_2
  (∃ n, a n * b n = 1540) →  -- product condition
  (∀ n, a n * b n = 1540 → n ≤ 512) ∧  -- 512 is an upper bound
  (∃ n, a n * b n = 1540 ∧ n = 512) -- 512 is achievable
  := by sorry

end NUMINAMATH_CALUDE_largest_n_for_arithmetic_sequences_l2603_260397


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2603_260395

-- Define the coefficients of the quadratic equation
variable (a b c : ℝ)

-- Define the condition that ax^2 + bx + c can be expressed as 3(x - 5)^2 + 7
def quadratic_condition (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 3 * (x - 5)^2 + 7

-- Define the expanded form of 4ax^2 + 4bx + 4c
def expanded_quadratic (x : ℝ) : ℝ :=
  4 * a * x^2 + 4 * b * x + 4 * c

-- Theorem statement
theorem quadratic_transformation (h : ∀ x, quadratic_condition a b c x) :
  ∃ (n k : ℝ), ∀ x, expanded_quadratic a b c x = n * (x - 5)^2 + k :=
sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2603_260395


namespace NUMINAMATH_CALUDE_intersection_point_unique_l2603_260387

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 1) / 8 = (y - 8) / (-5) ∧ (x - 1) / 8 = (z + 5) / 12

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  x - 2*y - 3*z + 18 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (9, 3, 7)

theorem intersection_point_unique :
  ∃! p : ℝ × ℝ × ℝ, line p.1 p.2.1 p.2.2 ∧ plane p.1 p.2.1 p.2.2 ∧ p = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l2603_260387
