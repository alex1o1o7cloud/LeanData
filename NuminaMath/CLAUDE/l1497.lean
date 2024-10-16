import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1497_149700

theorem quadratic_equation_solution :
  let x₁ : ℝ := (1 + Real.sqrt 3) / 2
  let x₂ : ℝ := (1 - Real.sqrt 3) / 2
  2 * x₁^2 - 2 * x₁ - 1 = 0 ∧ 2 * x₂^2 - 2 * x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1497_149700


namespace NUMINAMATH_CALUDE_people_in_room_l1497_149734

theorem people_in_room (total_chairs : ℕ) (people : ℕ) : 
  (3 * people : ℚ) / 5 = (5 * total_chairs : ℚ) / 6 →  -- Three-fifths of people are seated in five-sixths of chairs
  total_chairs - (5 * total_chairs) / 6 = 10 →         -- 10 chairs are empty
  people = 83 := by
sorry

end NUMINAMATH_CALUDE_people_in_room_l1497_149734


namespace NUMINAMATH_CALUDE_pen_cost_l1497_149729

-- Define the number of pens bought by Robert
def robert_pens : ℕ := 4

-- Define the number of pens bought by Julia in terms of Robert's
def julia_pens : ℕ := 3 * robert_pens

-- Define the number of pens bought by Dorothy in terms of Julia's
def dorothy_pens : ℕ := julia_pens / 2

-- Define the total amount spent
def total_spent : ℚ := 33

-- Define the total number of pens bought
def total_pens : ℕ := dorothy_pens + julia_pens + robert_pens

-- Theorem: The cost of one pen is $1.50
theorem pen_cost : total_spent / total_pens = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l1497_149729


namespace NUMINAMATH_CALUDE_shiny_igneous_rocks_l1497_149799

theorem shiny_igneous_rocks (total : ℕ) (sedimentary : ℕ) (igneous : ℕ) :
  total = 270 →
  igneous = sedimentary / 2 →
  total = sedimentary + igneous →
  (igneous / 3 : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_shiny_igneous_rocks_l1497_149799


namespace NUMINAMATH_CALUDE_spheres_in_unit_cube_radius_l1497_149757

/-- A configuration of spheres in a unit cube -/
structure SpheresInCube where
  /-- The number of spheres in the cube -/
  num_spheres : ℕ
  /-- The radius of each sphere -/
  radius : ℝ
  /-- One sphere is at a vertex of the cube -/
  vertex_sphere : Prop
  /-- Each of the remaining spheres is tangent to the vertex sphere and three faces of the cube -/
  remaining_spheres_tangent : Prop

/-- The theorem stating the radius of spheres in the given configuration -/
theorem spheres_in_unit_cube_radius (config : SpheresInCube) :
  config.num_spheres = 12 →
  config.vertex_sphere →
  config.remaining_spheres_tangent →
  config.radius = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_spheres_in_unit_cube_radius_l1497_149757


namespace NUMINAMATH_CALUDE_quadratic_sum_l1497_149773

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * x^2 + b * x + c) →
  (QuadraticFunction a b c 1 = 64) →
  (QuadraticFunction a b c (-2) = 0) →
  (QuadraticFunction a b c 4 = 0) →
  a + b + c = 64 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1497_149773


namespace NUMINAMATH_CALUDE_orange_juice_bottles_l1497_149709

/-- Represents the number of bottles of each juice type -/
structure JuiceBottles where
  orange : ℕ
  apple : ℕ
  grape : ℕ

/-- Represents the cost in cents of each juice type -/
structure JuiceCosts where
  orange : ℕ
  apple : ℕ
  grape : ℕ

/-- The main theorem to prove -/
theorem orange_juice_bottles (b : JuiceBottles) (c : JuiceCosts) : 
  c.orange = 70 ∧ 
  c.apple = 60 ∧ 
  c.grape = 80 ∧ 
  b.orange + b.apple + b.grape = 100 ∧ 
  c.orange * b.orange + c.apple * b.apple + c.grape * b.grape = 7250 ∧
  b.apple = b.grape ∧
  b.orange = 2 * b.apple →
  b.orange = 50 := by
sorry

end NUMINAMATH_CALUDE_orange_juice_bottles_l1497_149709


namespace NUMINAMATH_CALUDE_gomez_students_sum_l1497_149707

theorem gomez_students_sum (x y : ℕ+) (h1 : x - y = 4) (h2 : x * y = 132) : x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_gomez_students_sum_l1497_149707


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1497_149745

theorem polygon_sides_count : ∀ n : ℕ,
  (n ≥ 3) →
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 5 :=
by
  sorry

#check polygon_sides_count

end NUMINAMATH_CALUDE_polygon_sides_count_l1497_149745


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1497_149750

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, -6)
  let b : ℝ × ℝ := (-1, m)
  are_parallel a b → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1497_149750


namespace NUMINAMATH_CALUDE_infinitely_many_minimal_points_l1497_149795

/-- Distance function from origin to point (x, y) -/
def distance (x y : ℝ) : ℝ := |x| + |y|

/-- The set of points (x, y) on the line y = x + 1 that minimize the distance from the origin -/
def minimal_distance_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 1 ∧ ∀ q : ℝ × ℝ, q.2 = q.1 + 1 → distance p.1 p.2 ≤ distance q.1 q.2}

/-- Theorem stating that there are infinitely many points that minimize the distance -/
theorem infinitely_many_minimal_points : Set.Infinite minimal_distance_points := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_minimal_points_l1497_149795


namespace NUMINAMATH_CALUDE_kyle_earnings_theorem_l1497_149792

/-- Calculates the money Kyle makes from selling his remaining baked goods --/
def kyle_earnings (initial_cookies : ℕ) (initial_brownies : ℕ) 
                  (kyle_cookies_eaten : ℕ) (kyle_brownies_eaten : ℕ)
                  (mom_cookies_eaten : ℕ) (mom_brownies_eaten : ℕ)
                  (cookie_price : ℚ) (brownie_price : ℚ) : ℚ :=
  let remaining_cookies := initial_cookies - (kyle_cookies_eaten + mom_cookies_eaten)
  let remaining_brownies := initial_brownies - (kyle_brownies_eaten + mom_brownies_eaten)
  (remaining_cookies : ℚ) * cookie_price + (remaining_brownies : ℚ) * brownie_price

/-- Theorem stating Kyle's earnings from selling all remaining baked goods --/
theorem kyle_earnings_theorem : 
  kyle_earnings 60 32 2 2 1 2 1 (3/2) = 99 := by
  sorry

end NUMINAMATH_CALUDE_kyle_earnings_theorem_l1497_149792


namespace NUMINAMATH_CALUDE_kevin_run_distance_l1497_149731

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents Kevin's run with three segments -/
structure KevinRun where
  flat_speed : ℝ
  flat_time : ℝ
  uphill_speed : ℝ
  uphill_time : ℝ
  downhill_speed : ℝ
  downhill_time : ℝ

/-- Calculates the total distance of Kevin's run -/
def total_distance (run : KevinRun) : ℝ :=
  distance run.flat_speed run.flat_time +
  distance run.uphill_speed run.uphill_time +
  distance run.downhill_speed run.downhill_time

/-- Theorem stating that Kevin's total run distance is 17 miles -/
theorem kevin_run_distance :
  let run : KevinRun := {
    flat_speed := 10,
    flat_time := 0.5,
    uphill_speed := 20,
    uphill_time := 0.5,
    downhill_speed := 8,
    downhill_time := 0.25
  }
  total_distance run = 17 := by sorry

end NUMINAMATH_CALUDE_kevin_run_distance_l1497_149731


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1497_149762

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_q : q = 2)
  (h_a2 : a 2 = 8) :
  a 6 = 128 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1497_149762


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1497_149798

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 16 = -6 ∧ a 2 * a 16 = 2) →
  (a 2 * a 16) / a 9 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1497_149798


namespace NUMINAMATH_CALUDE_value_of_expression_l1497_149727

theorem value_of_expression (a b x y : ℝ) 
  (h1 : a * x + b * y = 3) 
  (h2 : a * y - b * x = 5) : 
  (a^2 + b^2) * (x^2 + y^2) = 34 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l1497_149727


namespace NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l1497_149790

theorem min_product_of_reciprocal_sum (x y : ℕ+) : 
  (1 : ℚ) / x + 1 / (3 * y) = 1 / 9 → 
  ∃ (a b : ℕ+), (1 : ℚ) / a + 1 / (3 * b) = 1 / 9 ∧ a * b = 108 ∧ 
  ∀ (c d : ℕ+), (1 : ℚ) / c + 1 / (3 * d) = 1 / 9 → c * d ≥ 108 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l1497_149790


namespace NUMINAMATH_CALUDE_area_triangle_GCD_l1497_149704

/-- Given a square ABCD with area 256, point E on BC dividing it 3:1, 
    F and G midpoints of AE and DE, and area of BEGF is 48, 
    prove that the area of triangle GCD is 48. -/
theorem area_triangle_GCD (A B C D E F G : ℝ × ℝ) : 
  -- Square ABCD has area 256
  (B.1 - A.1) * (C.2 - B.2) = 256 →
  -- E divides BC in 3:1 ratio
  E.1 - B.1 = 3/4 * (C.1 - B.1) →
  E.2 = B.2 →
  -- F is midpoint of AE
  F = ((A.1 + E.1)/2, (A.2 + E.2)/2) →
  -- G is midpoint of DE
  G = ((D.1 + E.1)/2, (D.2 + E.2)/2) →
  -- Area of quadrilateral BEGF is 48
  abs ((B.1*E.2 + E.1*G.2 + G.1*F.2 + F.1*B.2) - 
       (E.1*B.2 + G.1*E.2 + F.1*G.2 + B.1*F.2)) / 2 = 48 →
  -- Then the area of triangle GCD is 48
  abs ((G.1*C.2 + C.1*D.2 + D.1*G.2) - 
       (C.1*G.2 + D.1*C.2 + G.1*D.2)) / 2 = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_area_triangle_GCD_l1497_149704


namespace NUMINAMATH_CALUDE_second_player_wins_l1497_149774

/-- Represents the game board -/
def Board := Fin 4 → Fin 2017 → Bool

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents a position on the board -/
structure Position :=
  (row : Fin 4)
  (col : Fin 2017)

/-- Checks if a rook at the given position attacks an even number of other rooks -/
def attacksEven (board : Board) (pos : Position) : Bool :=
  sorry

/-- Checks if a rook at the given position attacks an odd number of other rooks -/
def attacksOdd (board : Board) (pos : Position) : Bool :=
  sorry

/-- Checks if the given move is valid for the current player -/
def isValidMove (board : Board) (player : Player) (pos : Position) : Prop :=
  match player with
  | Player.First => attacksEven board pos
  | Player.Second => attacksOdd board pos

/-- Represents a winning strategy for the second player -/
def secondPlayerStrategy (board : Board) (firstPlayerMove : Position) : Position :=
  sorry

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∀ (board : Board),
  ∀ (firstPlayerMove : Position),
  isValidMove board Player.First firstPlayerMove →
  isValidMove board Player.Second (secondPlayerStrategy board firstPlayerMove) :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l1497_149774


namespace NUMINAMATH_CALUDE_kata_friends_and_money_l1497_149797

/-- The number of friends Káta has -/
def n : ℕ := sorry

/-- The amount of money Káta has for gifts (in Kč) -/
def x : ℕ := sorry

/-- The cost of a hair clip (in Kč) -/
def hair_clip_cost : ℕ := 28

/-- The cost of a teddy bear (in Kč) -/
def teddy_bear_cost : ℕ := 42

/-- The amount left after buying hair clips (in Kč) -/
def hair_clip_remainder : ℕ := 29

/-- The amount short after buying teddy bears (in Kč) -/
def teddy_bear_shortage : ℕ := 13

theorem kata_friends_and_money :
  (n * hair_clip_cost + hair_clip_remainder = x) ∧
  (n * teddy_bear_cost - teddy_bear_shortage = x) →
  n = 3 ∧ x = 113 := by
  sorry

end NUMINAMATH_CALUDE_kata_friends_and_money_l1497_149797


namespace NUMINAMATH_CALUDE_amc8_paths_count_l1497_149768

/-- Represents a position on the grid --/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a letter on the grid --/
inductive Letter
  | A | M | C | Eight

/-- Defines the grid layout --/
def grid : Position → Letter := sorry

/-- Checks if two positions are adjacent --/
def isAdjacent (p1 p2 : Position) : Bool := sorry

/-- Defines a valid path on the grid --/
def ValidPath : List Position → Prop := sorry

/-- Counts the number of valid paths spelling AMC8 --/
def countAMC8Paths : Nat := sorry

/-- Theorem stating that the number of valid AMC8 paths is 24 --/
theorem amc8_paths_count : countAMC8Paths = 24 := by sorry

end NUMINAMATH_CALUDE_amc8_paths_count_l1497_149768


namespace NUMINAMATH_CALUDE_equation_solution_l1497_149741

theorem equation_solution : ∃ x : ℚ, (4/7 : ℚ) * (2/5 : ℚ) * x = 8 ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1497_149741


namespace NUMINAMATH_CALUDE_camp_total_is_250_l1497_149763

/-- Represents the distribution of students in a boys camp --/
structure CampDistribution where
  total : ℕ
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ
  schoolAScience : ℕ
  schoolAMath : ℕ
  schoolALiterature : ℕ
  schoolBScience : ℕ
  schoolBMath : ℕ
  schoolBLiterature : ℕ
  schoolCScience : ℕ
  schoolCMath : ℕ
  schoolCLiterature : ℕ

/-- The camp distribution satisfies the given conditions --/
def isValidDistribution (d : CampDistribution) : Prop :=
  d.schoolA = d.total / 5 ∧
  d.schoolB = 3 * d.total / 10 ∧
  d.schoolC = d.total / 2 ∧
  d.schoolAScience = 3 * d.schoolA / 10 ∧
  d.schoolAMath = 2 * d.schoolA / 5 ∧
  d.schoolALiterature = 3 * d.schoolA / 10 ∧
  d.schoolBScience = d.schoolB / 4 ∧
  d.schoolBMath = 7 * d.schoolB / 20 ∧
  d.schoolBLiterature = 2 * d.schoolB / 5 ∧
  d.schoolCScience = 3 * d.schoolC / 20 ∧
  d.schoolCMath = d.schoolC / 2 ∧
  d.schoolCLiterature = 7 * d.schoolC / 20 ∧
  d.schoolA - d.schoolAScience = 35 ∧
  d.schoolBMath = 20

/-- Theorem: Given the conditions, the total number of boys in the camp is 250 --/
theorem camp_total_is_250 (d : CampDistribution) (h : isValidDistribution d) : d.total = 250 := by
  sorry


end NUMINAMATH_CALUDE_camp_total_is_250_l1497_149763


namespace NUMINAMATH_CALUDE_framing_needed_l1497_149761

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered photograph. -/
theorem framing_needed (orig_width orig_height border_width : ℕ) : 
  orig_width = 5 →
  orig_height = 7 →
  border_width = 3 →
  let enlarged_width := 2 * orig_width
  let enlarged_height := 2 * orig_height
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (framed_width + framed_height)
  let feet := (perimeter + 11) / 12  -- Ceiling division to get the next whole foot
  feet = 6 := by
  sorry

#check framing_needed

end NUMINAMATH_CALUDE_framing_needed_l1497_149761


namespace NUMINAMATH_CALUDE_f_properties_l1497_149711

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sin x + 1

theorem f_properties :
  (∃ (x : ℝ), f x = -2) ∧ 
  (∀ (x : ℝ), f (Real.pi / 2 + x) = f (Real.pi / 2 - x)) ∧
  (∀ (x : ℝ), x > 0 → x < Real.pi / 2 → (deriv f) x < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1497_149711


namespace NUMINAMATH_CALUDE_no_extrema_iff_a_nonpositive_l1497_149748

/-- The function f(x) = x^2 - a * ln(x) has no extrema if and only if a ≤ 0 -/
theorem no_extrema_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y > 0 ∧ (x^2 - a * Real.log x < y^2 - a * Real.log y ∨ 
                                      x^2 - a * Real.log x > y^2 - a * Real.log y)) ↔ 
  a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_no_extrema_iff_a_nonpositive_l1497_149748


namespace NUMINAMATH_CALUDE_complex_inequality_complex_inequality_equality_complex_inequality_equality_at_one_l1497_149736

theorem complex_inequality (z : ℂ) : Complex.abs z ^ 2 + 2 * Complex.abs (z - 1) ≥ 1 :=
by sorry

theorem complex_inequality_equality : ∃ z : ℂ, Complex.abs z ^ 2 + 2 * Complex.abs (z - 1) = 1 :=
by sorry

theorem complex_inequality_equality_at_one : Complex.abs (1 : ℂ) ^ 2 + 2 * Complex.abs (1 - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_complex_inequality_equality_complex_inequality_equality_at_one_l1497_149736


namespace NUMINAMATH_CALUDE_system_has_three_solutions_l1497_149793

/-- The system of equations has exactly 3 distinct real solutions -/
theorem system_has_three_solutions :
  ∃! (S : Set (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (a b c d : ℝ), (a, b, c, d) ∈ S ↔ 
      (a = (b + c + d)^3 ∧
       b = (a + c + d)^3 ∧
       c = (a + b + d)^3 ∧
       d = (a + b + c)^3)) ∧
    S.ncard = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_has_three_solutions_l1497_149793


namespace NUMINAMATH_CALUDE_number_B_value_l1497_149776

theorem number_B_value (A B : ℕ) (h1 : A = 612) (h2 : B = 3 * A) : B = 1836 := by
  sorry

end NUMINAMATH_CALUDE_number_B_value_l1497_149776


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l1497_149714

theorem division_multiplication_problem : ((-128) / (-16)) * 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l1497_149714


namespace NUMINAMATH_CALUDE_smallest_year_after_2000_with_digit_sum_15_l1497_149780

def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem smallest_year_after_2000_with_digit_sum_15 :
  (∀ y : ℕ, 2000 < y ∧ y < 2049 → sumOfDigits y ≠ 15) ∧ 
  2000 < 2049 ∧ 
  sumOfDigits 2049 = 15 := by
sorry

end NUMINAMATH_CALUDE_smallest_year_after_2000_with_digit_sum_15_l1497_149780


namespace NUMINAMATH_CALUDE_percentage_not_sold_approx_l1497_149701

def initial_stock : ℕ := 1100
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem percentage_not_sold_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentage_not_sold - 63.45| < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_not_sold_approx_l1497_149701


namespace NUMINAMATH_CALUDE_train_length_l1497_149710

/-- The length of a train given its speed and the time it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (5/18) → 
  platform_length = 270 → 
  crossing_time = 26 → 
  (train_speed * crossing_time) - platform_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1497_149710


namespace NUMINAMATH_CALUDE_line_segment_proportion_l1497_149771

theorem line_segment_proportion (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_proportion_l1497_149771


namespace NUMINAMATH_CALUDE_remaining_drawings_l1497_149791

-- Define the given parameters
def total_markers : ℕ := 12
def drawings_per_marker : ℚ := 3/2
def drawings_already_made : ℕ := 8

-- State the theorem
theorem remaining_drawings : 
  ⌊(total_markers : ℚ) * drawings_per_marker⌋ - drawings_already_made = 10 := by
  sorry

end NUMINAMATH_CALUDE_remaining_drawings_l1497_149791


namespace NUMINAMATH_CALUDE_balloon_permutations_count_l1497_149794

/-- The number of distinct permutations of a 7-letter word with two pairs of repeated letters -/
def balloon_permutations : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "balloon" is 1260 -/
theorem balloon_permutations_count : balloon_permutations = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_permutations_count_l1497_149794


namespace NUMINAMATH_CALUDE_square_cards_puzzle_l1497_149749

theorem square_cards_puzzle (n : ℕ) (h : n > 0) (eq : n^2 + 36 = (n + 1)^2 + 3) :
  n^2 + 36 = 292 := by
  sorry

end NUMINAMATH_CALUDE_square_cards_puzzle_l1497_149749


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l1497_149742

/-- Given points A, B, and C in a 2D plane satisfying specific conditions, 
    prove that the sum of the coordinates of A is 24. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (dist A C / dist A B = 1/3) →
  (dist B C / dist A B = 1/3) →
  B = (2, 6) →
  C = (4, 12) →
  A.1 + A.2 = 24 := by
  sorry

#check sum_of_coordinates_A

end NUMINAMATH_CALUDE_sum_of_coordinates_A_l1497_149742


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1497_149733

theorem diophantine_equation_solutions (p : ℕ) (h_prime : Nat.Prime p) :
  ∀ x y n : ℕ, x > 0 ∧ y > 0 ∧ n > 0 →
  p^n = x^3 + y^3 ↔
  (p = 2 ∧ ∃ k : ℕ, x = 2^k ∧ y = 2^k ∧ n = 3*k + 1) ∨
  (p = 3 ∧ ∃ k : ℕ, (x = 3^k ∧ y = 2 * 3^k ∧ n = 3*k + 2) ∨
                    (x = 2 * 3^k ∧ y = 3^k ∧ n = 3*k + 2)) :=
by sorry


end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1497_149733


namespace NUMINAMATH_CALUDE_initial_population_is_4144_l1497_149743

/-- Represents the population changes in a village --/
def village_population (initial : ℕ) : ℕ :=
  let after_bombardment := initial * 90 / 100
  let after_departure := after_bombardment * 85 / 100
  let after_refugees := after_departure + 50
  let after_births := after_refugees * 105 / 100
  let after_employment := after_births * 92 / 100
  after_employment + 100

/-- Theorem stating that the initial population of 4144 results in a final population of 3213 --/
theorem initial_population_is_4144 : village_population 4144 = 3213 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_is_4144_l1497_149743


namespace NUMINAMATH_CALUDE_inequality_proof_l1497_149786

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a + b + c + d + 8 / (a * b + b * c + c * d + d * a) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1497_149786


namespace NUMINAMATH_CALUDE_exists_nonnegative_coeff_multiplier_l1497_149719

/-- A polynomial with real coefficients that is positive for all nonnegative real numbers. -/
structure PositivePolynomial where
  P : Polynomial ℝ
  pos : ∀ x : ℝ, x ≥ 0 → P.eval x > 0

/-- The theorem stating that for any positive polynomial, there exists a positive integer n
    such that (1 + x)^n * P(x) has nonnegative coefficients. -/
theorem exists_nonnegative_coeff_multiplier (p : PositivePolynomial) :
  ∃ n : ℕ+, ∀ i : ℕ, ((1 + X : Polynomial ℝ)^(n : ℕ) * p.P).coeff i ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_nonnegative_coeff_multiplier_l1497_149719


namespace NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l1497_149706

theorem sin_product_equals_one_eighth : 
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * 
  Real.sin (72 * π / 180) * Real.sin (84 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_eighth_l1497_149706


namespace NUMINAMATH_CALUDE_sector_radius_l1497_149708

theorem sector_radius (arc_length : ℝ) (area : ℝ) (radius : ℝ) : 
  arc_length = 2 → area = 4 → area = (1/2) * radius * arc_length → radius = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l1497_149708


namespace NUMINAMATH_CALUDE_total_weekly_meals_l1497_149796

/-- The number of days in a week -/
def daysInWeek : ℕ := 7

/-- The number of meals served daily by the first restaurant -/
def restaurant1Meals : ℕ := 20

/-- The number of meals served daily by the second restaurant -/
def restaurant2Meals : ℕ := 40

/-- The number of meals served daily by the third restaurant -/
def restaurant3Meals : ℕ := 50

/-- Theorem stating that the total number of meals served per week by the three restaurants is 770 -/
theorem total_weekly_meals :
  (restaurant1Meals * daysInWeek) + (restaurant2Meals * daysInWeek) + (restaurant3Meals * daysInWeek) = 770 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_meals_l1497_149796


namespace NUMINAMATH_CALUDE_sine_is_odd_and_has_zero_point_l1497_149767

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to have a zero point
def has_zero_point (f : ℝ → ℝ) : Prop := ∃ x, f x = 0

theorem sine_is_odd_and_has_zero_point :
  is_odd Real.sin ∧ has_zero_point Real.sin :=
sorry

end NUMINAMATH_CALUDE_sine_is_odd_and_has_zero_point_l1497_149767


namespace NUMINAMATH_CALUDE_fixed_point_existence_l1497_149765

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two line segments have equal length -/
def equal_length (a b c d : Point) : Prop :=
  (a.x - b.x)^2 + (a.y - b.y)^2 = (c.x - d.x)^2 + (c.y - d.y)^2

/-- Check if an angle is 90 degrees -/
def is_right_angle (a b c : Point) : Prop :=
  (b.x - a.x) * (b.x - c.x) + (b.y - a.y) * (b.y - c.y) = 0

/-- Check if a quadrilateral is convex -/
def is_convex (a b c d : Point) : Prop := sorry

/-- Check if two points are on the same side of a line -/
def same_side (p q : Point) (l : Line) : Prop := sorry

theorem fixed_point_existence (a b : Point) :
  ∃ p : Point,
    ∀ c d : Point,
      is_convex a b c d →
      equal_length a b b c →
      equal_length a d d c →
      is_right_angle a d c →
      same_side c d (Line.mk (b.y - a.y) (a.x - b.x) (a.y * b.x - a.x * b.y)) →
      ∃ l : Line, d.on_line l ∧ c.on_line l ∧ p.on_line l :=
sorry

end NUMINAMATH_CALUDE_fixed_point_existence_l1497_149765


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1497_149781

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 462 →
  B = 150 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1497_149781


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l1497_149772

theorem davids_chemistry_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (h1 : english_marks = 72)
  (h2 : math_marks = 60)
  (h3 : physics_marks = 35)
  (h4 : biology_marks = 84)
  (h5 : average_marks = 62.6)
  (h6 : (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks : ℚ) / 5 = average_marks) :
  chemistry_marks = 62 :=
by sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l1497_149772


namespace NUMINAMATH_CALUDE_candy_store_spending_l1497_149784

def weekly_allowance : ℚ := 4.5

def arcade_spending (allowance : ℚ) : ℚ := (3 / 5) * allowance

def remaining_after_arcade (allowance : ℚ) : ℚ := allowance - arcade_spending allowance

def toy_store_spending (remaining : ℚ) : ℚ := (1 / 3) * remaining

def remaining_after_toy_store (remaining : ℚ) : ℚ := remaining - toy_store_spending remaining

theorem candy_store_spending :
  remaining_after_toy_store (remaining_after_arcade weekly_allowance) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_candy_store_spending_l1497_149784


namespace NUMINAMATH_CALUDE_expected_interval_is_three_minutes_l1497_149724

/-- Represents the train system with given conditions -/
structure TrainSystem where
  northern_route_time : ℝ
  southern_route_time : ℝ
  arrival_time_difference : ℝ
  travel_time_difference : ℝ

/-- The expected interval between trains in one direction -/
def expected_interval (ts : TrainSystem) : ℝ :=
  3

/-- Theorem stating that the expected interval is 3 minutes -/
theorem expected_interval_is_three_minutes (ts : TrainSystem) 
  (h1 : ts.northern_route_time = 17)
  (h2 : ts.southern_route_time = 11)
  (h3 : ts.arrival_time_difference = 1.25)
  (h4 : ts.travel_time_difference = 1) :
  expected_interval ts = 3 := by
  sorry

#check expected_interval_is_three_minutes

end NUMINAMATH_CALUDE_expected_interval_is_three_minutes_l1497_149724


namespace NUMINAMATH_CALUDE_asymptote_sum_l1497_149715

/-- Given a function g(x) = (x+5) / (x^2 + cx + d) with vertical asymptotes at x = 2 and x = -3,
    prove that the sum of c and d is -5. -/
theorem asymptote_sum (c d : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -3 → 
    (x + 5) / (x^2 + c*x + d) = (x + 5) / ((x - 2) * (x + 3))) →
  c + d = -5 := by
sorry

end NUMINAMATH_CALUDE_asymptote_sum_l1497_149715


namespace NUMINAMATH_CALUDE_sparrow_population_decrease_l1497_149732

def initial_population : ℕ := 1200
def decrease_rate : ℚ := 0.7
def target_percentage : ℚ := 0.15

def population (year : ℕ) : ℚ :=
  (initial_population : ℚ) * decrease_rate ^ (year - 2010)

theorem sparrow_population_decrease (year : ℕ) :
  year = 2016 ↔ 
    (population year < (initial_population : ℚ) * target_percentage ∧
     ∀ y, 2010 ≤ y ∧ y < 2016 → population y ≥ (initial_population : ℚ) * target_percentage) :=
by sorry

end NUMINAMATH_CALUDE_sparrow_population_decrease_l1497_149732


namespace NUMINAMATH_CALUDE_senior_count_l1497_149770

/-- Represents the count of students in each grade level -/
structure StudentCounts where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- Given the conditions of the student sample, proves the number of seniors -/
theorem senior_count (total : ℕ) (counts : StudentCounts) : 
  total = 800 ∧ 
  counts.juniors = (23 * total) / 100 ∧ 
  counts.sophomores = (25 * total) / 100 ∧ 
  counts.freshmen = counts.sophomores + 56 ∧ 
  total = counts.freshmen + counts.sophomores + counts.juniors + counts.seniors → 
  counts.seniors = 160 := by
sorry


end NUMINAMATH_CALUDE_senior_count_l1497_149770


namespace NUMINAMATH_CALUDE_cutlery_count_l1497_149760

/-- Calculates the total number of cutlery pieces after purchases -/
def totalCutlery (initialKnives : ℕ) : ℕ :=
  let initialTeaspoons := 2 * initialKnives
  let additionalKnives := initialKnives / 3
  let additionalTeaspoons := (2 * initialTeaspoons) / 3
  (initialKnives + additionalKnives) + (initialTeaspoons + additionalTeaspoons)

/-- Theorem stating that given the initial conditions, the total cutlery after purchases is 112 -/
theorem cutlery_count : totalCutlery 24 = 112 := by
  sorry

#eval totalCutlery 24

end NUMINAMATH_CALUDE_cutlery_count_l1497_149760


namespace NUMINAMATH_CALUDE_gcd_a_b_is_one_or_three_l1497_149737

def a (n : ℤ) : ℤ := n^5 + 6*n^3 + 8*n
def b (n : ℤ) : ℤ := n^4 + 4*n^2 + 3

theorem gcd_a_b_is_one_or_three (n : ℤ) : Nat.gcd (Int.natAbs (a n)) (Int.natAbs (b n)) = 1 ∨ Nat.gcd (Int.natAbs (a n)) (Int.natAbs (b n)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_a_b_is_one_or_three_l1497_149737


namespace NUMINAMATH_CALUDE_bus_journey_distance_l1497_149723

/-- Given a bus journey with two different speeds, prove the distance covered at the lower speed. -/
theorem bus_journey_distance (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5)
  (h5 : total_time = (distance1 / speed1) + ((total_distance - distance1) / speed2)) :
  distance1 = 100 := by sorry


end NUMINAMATH_CALUDE_bus_journey_distance_l1497_149723


namespace NUMINAMATH_CALUDE_function_is_linear_l1497_149713

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

/-- The main theorem stating that any function satisfying the equation is linear -/
theorem function_is_linear (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
by sorry

end NUMINAMATH_CALUDE_function_is_linear_l1497_149713


namespace NUMINAMATH_CALUDE_number_of_valid_choices_is_84_l1497_149785

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- The number of ways to choose three different digits a, b, c from 1 to 9 such that a < b < c -/
def NumberOfValidChoices : ℕ := sorry

/-- The theorem stating that the number of valid choices is 84 -/
theorem number_of_valid_choices_is_84 : NumberOfValidChoices = 84 := by sorry

end NUMINAMATH_CALUDE_number_of_valid_choices_is_84_l1497_149785


namespace NUMINAMATH_CALUDE_magical_stack_size_is_470_l1497_149726

/-- A stack of cards is magical if it satisfies certain conditions --/
structure MagicalStack :=
  (n : ℕ)
  (total_cards : ℕ := 2 * n)
  (retains_position : ℕ := 157)
  (is_magical : Prop)

/-- The number of cards in a magical stack where card 157 retains its position --/
def magical_stack_size (stack : MagicalStack) : ℕ := stack.total_cards

/-- Theorem stating the size of the magical stack --/
theorem magical_stack_size_is_470 (stack : MagicalStack) : 
  stack.retains_position = 157 → magical_stack_size stack = 470 := by
  sorry

#check magical_stack_size_is_470

end NUMINAMATH_CALUDE_magical_stack_size_is_470_l1497_149726


namespace NUMINAMATH_CALUDE_percentage_difference_l1497_149766

theorem percentage_difference (p t j : ℝ) : 
  t = 0.9375 * p →  -- t is 6.25% less than p
  j = 0.8 * t →     -- j is 20% less than t
  j = 0.75 * p :=   -- j is 25% less than p
by sorry

end NUMINAMATH_CALUDE_percentage_difference_l1497_149766


namespace NUMINAMATH_CALUDE_bicycle_problem_l1497_149764

/-- Prove that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
  (h1 : distance = 12)
  (h2 : speed_ratio = 1.2)
  (h3 : time_difference = 1/6) :
  ∃ (speed_B : ℝ), 
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference ∧ 
    speed_B = 12 := by
sorry

end NUMINAMATH_CALUDE_bicycle_problem_l1497_149764


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1497_149782

theorem sufficient_not_necessary (x : ℝ) :
  (x > 1/2 → (1 - 2*x) * (x + 1) < 0) ∧
  ¬(∀ x : ℝ, (1 - 2*x) * (x + 1) < 0 → x > 1/2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1497_149782


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l1497_149746

theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 3 / 4) (h2 : b / d = 3 / 4) :
  (a * b) / (c * d) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l1497_149746


namespace NUMINAMATH_CALUDE_history_books_shelved_l1497_149775

theorem history_books_shelved (total_books : ℕ) (romance_books : ℕ) (poetry_books : ℕ)
  (western_books : ℕ) (biography_books : ℕ) :
  total_books = 46 →
  romance_books = 8 →
  poetry_books = 4 →
  western_books = 5 →
  biography_books = 6 →
  ∃ (history_books : ℕ) (mystery_books : ℕ),
    history_books = 12 ∧
    mystery_books = western_books + biography_books ∧
    total_books = history_books + romance_books + poetry_books + western_books + biography_books + mystery_books :=
by
  sorry

end NUMINAMATH_CALUDE_history_books_shelved_l1497_149775


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_four_squared_plus_six_l1497_149712

theorem absolute_value_of_negative_four_squared_plus_six : 
  |(-4^2 + 6)| = 10 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_four_squared_plus_six_l1497_149712


namespace NUMINAMATH_CALUDE_keith_cards_proof_l1497_149744

/-- The number of cards Keith started with -/
def initial_cards : ℕ := 84

/-- The number of cards Keith has left after the incident -/
def remaining_cards : ℕ := 46

/-- The number of cards Keith bought -/
def bought_cards : ℕ := 8

theorem keith_cards_proof :
  ∃ (total : ℕ), total = initial_cards + bought_cards ∧ 
                 remaining_cards * 2 = total := by
  sorry

end NUMINAMATH_CALUDE_keith_cards_proof_l1497_149744


namespace NUMINAMATH_CALUDE_cats_problem_l1497_149779

/-- The number of cats owned by the certain person -/
def person_cats (melanie_cats : ℕ) (annie_cats : ℕ) : ℕ :=
  3 * annie_cats

theorem cats_problem (melanie_cats : ℕ) (annie_cats : ℕ) 
  (h1 : melanie_cats = 2 * annie_cats)
  (h2 : melanie_cats = 60) :
  person_cats melanie_cats annie_cats = 90 := by
  sorry

end NUMINAMATH_CALUDE_cats_problem_l1497_149779


namespace NUMINAMATH_CALUDE_work_completion_l1497_149752

/-- The number of men in the first group -/
def first_group : ℕ := 18

/-- The number of days for the first group to complete the work -/
def first_days : ℕ := 30

/-- The number of days for the second group to complete the work -/
def second_days : ℕ := 36

/-- The number of men in the second group -/
def second_group : ℕ := (first_group * first_days) / second_days

theorem work_completion :
  second_group = 15 := by sorry

end NUMINAMATH_CALUDE_work_completion_l1497_149752


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1497_149738

theorem square_area_from_diagonal (d : ℝ) (h : d = 16 * Real.sqrt 2) : 
  (d / Real.sqrt 2) ^ 2 = 256 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1497_149738


namespace NUMINAMATH_CALUDE_inequality_proof_l1497_149751

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a^3 / (b^2 - 1)) + (b^3 / (c^2 - 1)) + (c^3 / (a^2 - 1)) ≥ (9 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1497_149751


namespace NUMINAMATH_CALUDE_star_def_star_diff_neg_star_special_case_l1497_149777

-- Define the ☆ operation
def star (a b : ℝ) : ℝ := 3 * a + b

-- Theorem 1: Definition of ☆ operation
theorem star_def (a b : ℝ) : star a b = 3 * a + b := by sorry

-- Theorem 2: If a < b, then a☆b - b☆a < 0
theorem star_diff_neg {a b : ℝ} (h : a < b) : star a b - star b a < 0 := by sorry

-- Theorem 3: If a☆(-2b) = 4, then [3(a-b)]☆(3a+b) = 16
theorem star_special_case {a b : ℝ} (h : star a (-2*b) = 4) : 
  star (3*(a-b)) (3*a+b) = 16 := by sorry

end NUMINAMATH_CALUDE_star_def_star_diff_neg_star_special_case_l1497_149777


namespace NUMINAMATH_CALUDE_triangle_expression_l1497_149756

theorem triangle_expression (A B C : ℝ) : 
  A = 15 * π / 180 →
  A + B + C = π →
  Real.sqrt 3 * Real.sin A - Real.cos (B + C) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_l1497_149756


namespace NUMINAMATH_CALUDE_water_consumption_per_person_per_hour_l1497_149787

theorem water_consumption_per_person_per_hour 
  (num_people : ℕ) 
  (total_hours : ℕ) 
  (total_bottles : ℕ) 
  (h1 : num_people = 4) 
  (h2 : total_hours = 16) 
  (h3 : total_bottles = 32) : 
  (total_bottles : ℚ) / total_hours / num_people = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_per_person_per_hour_l1497_149787


namespace NUMINAMATH_CALUDE_book_original_price_l1497_149728

theorem book_original_price (discounted_price original_price : ℝ) : 
  discounted_price = 5 → 
  discounted_price = (1 / 10) * original_price → 
  original_price = 50 := by
sorry

end NUMINAMATH_CALUDE_book_original_price_l1497_149728


namespace NUMINAMATH_CALUDE_largest_positive_integer_theorem_l1497_149721

/-- Binary operation @ defined as n @ n = n - (n * 5) -/
def binary_op (n : ℤ) : ℤ := n - (n * 5)

/-- Proposition: 1 is the largest positive integer n such that n @ n < 10 -/
theorem largest_positive_integer_theorem :
  ∀ n : ℕ, n > 1 → binary_op n ≥ 10 ∧ binary_op 1 < 10 := by
  sorry

end NUMINAMATH_CALUDE_largest_positive_integer_theorem_l1497_149721


namespace NUMINAMATH_CALUDE_video_game_lives_l1497_149789

theorem video_game_lives (initial_lives next_level_lives total_lives : ℝ) 
  (h1 : initial_lives = 43.0)
  (h2 : next_level_lives = 27.0)
  (h3 : total_lives = 84) :
  ∃ hard_part_lives : ℝ, 
    hard_part_lives = 14.0 ∧ 
    initial_lives + hard_part_lives + next_level_lives = total_lives :=
by sorry

end NUMINAMATH_CALUDE_video_game_lives_l1497_149789


namespace NUMINAMATH_CALUDE_inverse_of_12_mod_1009_l1497_149702

theorem inverse_of_12_mod_1009 : ∃! x : ℕ, x < 1009 ∧ (12 * x) % 1009 = 1 :=
by
  use 925
  sorry

end NUMINAMATH_CALUDE_inverse_of_12_mod_1009_l1497_149702


namespace NUMINAMATH_CALUDE_only_cri_du_chat_chromosomal_variation_l1497_149718

-- Define the types of genetic causes
inductive GeneticCause
| GeneMutation
| ChromosomalVariation

-- Define the genetic diseases
inductive GeneticDisease
| Albinism
| Hemophilia
| CriDuChatSyndrome
| SickleCellAnemia

-- Define a function that assigns a cause to each disease
def diseaseCause : GeneticDisease → GeneticCause
| GeneticDisease.Albinism => GeneticCause.GeneMutation
| GeneticDisease.Hemophilia => GeneticCause.GeneMutation
| GeneticDisease.CriDuChatSyndrome => GeneticCause.ChromosomalVariation
| GeneticDisease.SickleCellAnemia => GeneticCause.GeneMutation

-- Theorem stating that only Cri-du-chat syndrome is caused by chromosomal variation
theorem only_cri_du_chat_chromosomal_variation :
  ∀ (d : GeneticDisease), diseaseCause d = GeneticCause.ChromosomalVariation ↔ d = GeneticDisease.CriDuChatSyndrome :=
by sorry


end NUMINAMATH_CALUDE_only_cri_du_chat_chromosomal_variation_l1497_149718


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l1497_149717

theorem polynomial_product_expansion :
  let p₁ : Polynomial ℝ := 3 * X^2 + 4 * X - 5
  let p₂ : Polynomial ℝ := 4 * X^3 - 3 * X^2 + 2 * X - 1
  p₁ * p₂ = 12 * X^5 + 25 * X^4 - 41 * X^3 - 14 * X^2 + 28 * X - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l1497_149717


namespace NUMINAMATH_CALUDE_spade_calculation_l1497_149722

def spade (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem spade_calculation : spade 2 (spade 3 (spade 1 2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l1497_149722


namespace NUMINAMATH_CALUDE_keychain_arrangements_l1497_149788

def number_of_keychains : ℕ := 5

def total_permutations (n : ℕ) : ℕ := n.factorial

def adjacent_permutations (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem keychain_arrangements :
  total_permutations number_of_keychains - adjacent_permutations number_of_keychains = 72 :=
sorry

end NUMINAMATH_CALUDE_keychain_arrangements_l1497_149788


namespace NUMINAMATH_CALUDE_horner_method_value_l1497_149769

def horner_polynomial (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℤ) : ℤ :=
  horner_polynomial [3, 5, 6, 79, -8, 35, 12] x

theorem horner_method_value :
  f (-4) = 220 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_value_l1497_149769


namespace NUMINAMATH_CALUDE_circle_x_axis_intersection_l1497_149755

/-- Given a circle with diameter endpoints (0,0) and (10,10), 
    the x-coordinate of the second intersection point with the x-axis is 10 -/
theorem circle_x_axis_intersection :
  ∀ (C : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - 5)^2 + (y - 5)^2 = 50) →
    (0, 0) ∈ C →
    (10, 10) ∈ C →
    ∃ (x : ℝ), x ≠ 0 ∧ (x, 0) ∈ C ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_x_axis_intersection_l1497_149755


namespace NUMINAMATH_CALUDE_b_2048_value_l1497_149740

/-- A sequence of real numbers satisfying the given conditions -/
def special_sequence (b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≥ 2 → b n = b (n - 1) * b (n + 1)) ∧
  (b 1 = 3 + 2 * Real.sqrt 5) ∧
  (b 2023 = 23 + 10 * Real.sqrt 5)

/-- The theorem stating the value of b_2048 -/
theorem b_2048_value (b : ℕ → ℝ) (h : special_sequence b) :
  b 2048 = 19 + 6 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_b_2048_value_l1497_149740


namespace NUMINAMATH_CALUDE_cubic_function_extrema_condition_l1497_149730

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a + 6)

/-- Theorem: If f has both a maximum and a minimum value, then a < -3 or a > 6 -/
theorem cubic_function_extrema_condition (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_condition_l1497_149730


namespace NUMINAMATH_CALUDE_lexie_age_difference_l1497_149754

/-- Given information about Lexie, her sister, and her brother's ages, prove the age difference between Lexie and her brother. -/
theorem lexie_age_difference (lexie_age : ℕ) (sister_age : ℕ) (brother_age : ℕ)
  (h1 : lexie_age = 8)
  (h2 : sister_age = 2 * lexie_age)
  (h3 : sister_age - brother_age = 14) :
  lexie_age - brother_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_lexie_age_difference_l1497_149754


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1497_149716

theorem parabola_shift_theorem (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁ * x₂ = x₁ + x₂ + 49 →
  x₁ * x₂ = -6 * m →
  x₁ + x₂ = 2 * m - 1 →
  min (abs x₁) (abs x₂) = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1497_149716


namespace NUMINAMATH_CALUDE_max_subjects_per_teacher_l1497_149705

theorem max_subjects_per_teacher 
  (total_subjects : Nat) 
  (min_teachers : Nat) 
  (maths_teachers : Nat) 
  (physics_teachers : Nat) 
  (chemistry_teachers : Nat) 
  (h1 : total_subjects = maths_teachers + physics_teachers + chemistry_teachers)
  (h2 : maths_teachers = 4)
  (h3 : physics_teachers = 3)
  (h4 : chemistry_teachers = 3)
  (h5 : min_teachers = 5)
  : (total_subjects / min_teachers : Nat) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_subjects_per_teacher_l1497_149705


namespace NUMINAMATH_CALUDE_largest_constant_inequality_largest_constant_is_three_equality_condition_l1497_149739

theorem largest_constant_inequality (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) :=
sorry

theorem largest_constant_is_three :
  ∀ C > 3, ∃ x₁ x₂ x₃ x₄ x₅ x₆ : ℝ,
    (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 < C * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) :=
sorry

theorem equality_condition (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 = 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) ↔
  (x₁ + x₄ = x₂ + x₅) ∧ (x₂ + x₅ = x₃ + x₆) :=
sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_largest_constant_is_three_equality_condition_l1497_149739


namespace NUMINAMATH_CALUDE_james_sodas_per_day_l1497_149703

/-- Calculates the number of sodas James drinks per day -/
def sodas_per_day (packs : ℕ) (sodas_per_pack : ℕ) (initial_sodas : ℕ) (days : ℕ) : ℕ :=
  (packs * sodas_per_pack + initial_sodas) / days

/-- Theorem: James drinks 10 sodas per day -/
theorem james_sodas_per_day :
  sodas_per_day 5 12 10 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_sodas_per_day_l1497_149703


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1497_149758

theorem complex_expression_simplification :
  let z₁ : ℂ := 4 + 2*I
  let z₂ : ℂ := 4 - 2*I
  (z₁ / z₂) + (z₂ / z₁) + ((4*I) / z₂) - ((4*I) / z₁) = (2 : ℂ) / 5 :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1497_149758


namespace NUMINAMATH_CALUDE_fence_sections_count_l1497_149747

/-- The number of posts in the nth section -/
def posts_in_section (n : ℕ) : ℕ := 2 * n + 1

/-- The total number of posts used for n sections -/
def total_posts (n : ℕ) : ℕ := n^2

/-- The total number of posts available -/
def available_posts : ℕ := 435

theorem fence_sections_count :
  ∃ (n : ℕ), total_posts n = available_posts ∧ n = 21 :=
sorry

end NUMINAMATH_CALUDE_fence_sections_count_l1497_149747


namespace NUMINAMATH_CALUDE_monitoring_system_odd_agents_l1497_149720

/-- Represents a cyclic monitoring system of agents -/
structure MonitoringSystem (n : ℕ) where
  -- The number of agents is positive
  agents_exist : 0 < n
  -- The monitoring function
  monitor : Fin n → Fin n
  -- The monitoring is cyclic
  cyclic : ∀ i : Fin n, monitor (monitor i) = i.succ

/-- Theorem: In a cyclic monitoring system, the number of agents is odd -/
theorem monitoring_system_odd_agents (n : ℕ) (sys : MonitoringSystem n) : 
  Odd n := by
  sorry


end NUMINAMATH_CALUDE_monitoring_system_odd_agents_l1497_149720


namespace NUMINAMATH_CALUDE_club_average_age_l1497_149783

/-- Represents the average age of a group of people -/
def average_age (total_age : ℕ) (num_people : ℕ) : ℚ :=
  (total_age : ℚ) / (num_people : ℚ)

/-- Represents the total age of a group of people -/
def total_age (avg_age : ℕ) (num_people : ℕ) : ℕ :=
  avg_age * num_people

theorem club_average_age 
  (num_women : ℕ) (women_avg_age : ℕ) 
  (num_men : ℕ) (men_avg_age : ℕ) 
  (num_children : ℕ) (children_avg_age : ℕ) :
  num_women = 12 → 
  women_avg_age = 32 → 
  num_men = 18 → 
  men_avg_age = 36 → 
  num_children = 20 → 
  children_avg_age = 10 → 
  average_age 
    (total_age women_avg_age num_women + 
     total_age men_avg_age num_men + 
     total_age children_avg_age num_children)
    (num_women + num_men + num_children) = 24 := by
  sorry

end NUMINAMATH_CALUDE_club_average_age_l1497_149783


namespace NUMINAMATH_CALUDE_stock_purchase_problem_l1497_149725

/-- Mr. Wise's stock purchase problem -/
theorem stock_purchase_problem (total_value : ℝ) (price_type1 : ℝ) (total_shares : ℕ) (shares_type1 : ℕ) :
  total_value = 1950 →
  price_type1 = 3 →
  total_shares = 450 →
  shares_type1 = 400 →
  ∃ (price_type2 : ℝ),
    price_type2 * (total_shares - shares_type1) + price_type1 * shares_type1 = total_value ∧
    price_type2 = 15 :=
by sorry

end NUMINAMATH_CALUDE_stock_purchase_problem_l1497_149725


namespace NUMINAMATH_CALUDE_new_person_weight_is_129_l1497_149759

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (replaced_weight : ℝ) (average_increase : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating the weight of the new person under the given conditions -/
theorem new_person_weight_is_129 :
  weight_of_new_person 4 95 8.5 = 129 := by
  sorry

#eval weight_of_new_person 4 95 8.5

end NUMINAMATH_CALUDE_new_person_weight_is_129_l1497_149759


namespace NUMINAMATH_CALUDE_lcm_inequality_l1497_149778

/-- For any two positive integers n and m where n > m, 
    the sum of the least common multiples of (m,n) and (m+1,n+1) 
    is greater than or equal to (2nm)/√(n-m). -/
theorem lcm_inequality (m n : ℕ) (h1 : 0 < m) (h2 : m < n) :
  Nat.lcm m n + Nat.lcm (m + 1) (n + 1) ≥ 
  (2 * n * m : ℝ) / Real.sqrt (n - m : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_lcm_inequality_l1497_149778


namespace NUMINAMATH_CALUDE_calculation_proof_l1497_149735

theorem calculation_proof : (-2)^0 - Real.sqrt 8 - abs (-5) + 4 * Real.sin (π/4) = -4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1497_149735


namespace NUMINAMATH_CALUDE_john_weekly_earnings_l1497_149753

/-- Calculates John's weekly earnings from crab fishing -/
def weekly_earnings (small_baskets medium_baskets large_baskets jumbo_baskets : ℕ)
  (small_per_basket medium_per_basket large_per_basket jumbo_per_basket : ℕ)
  (small_price medium_price large_price jumbo_price : ℕ) : ℕ :=
  (small_baskets * small_per_basket * small_price) +
  (medium_baskets * medium_per_basket * medium_price) +
  (large_baskets * large_per_basket * large_price) +
  (jumbo_baskets * jumbo_per_basket * jumbo_price)

theorem john_weekly_earnings :
  weekly_earnings 3 2 4 1 4 3 5 2 3 4 5 7 = 174 := by
  sorry

end NUMINAMATH_CALUDE_john_weekly_earnings_l1497_149753
