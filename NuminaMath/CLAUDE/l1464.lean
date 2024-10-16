import Mathlib

namespace NUMINAMATH_CALUDE_max_value_product_sum_l1464_146419

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l1464_146419


namespace NUMINAMATH_CALUDE_rectangular_solid_spheres_l1464_146405

/-- A rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

/-- A sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Predicate for a sphere being circumscribed around a rectangular solid -/
def isCircumscribed (s : Sphere) (r : RectangularSolid) : Prop :=
  sorry

/-- Predicate for a sphere being inscribed in a rectangular solid -/
def isInscribed (s : Sphere) (r : RectangularSolid) : Prop :=
  sorry

/-- Theorem: A rectangular solid with a circumscribed sphere does not necessarily have an inscribed sphere -/
theorem rectangular_solid_spheres (r : RectangularSolid) (s : Sphere) :
  isCircumscribed s r → ¬∀ (s' : Sphere), isInscribed s' r :=
sorry

end NUMINAMATH_CALUDE_rectangular_solid_spheres_l1464_146405


namespace NUMINAMATH_CALUDE_heather_bicycle_speed_l1464_146447

/-- Heather's bicycle problem -/
theorem heather_bicycle_speed (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 40 ∧ time = 5 ∧ speed = distance / time → speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_heather_bicycle_speed_l1464_146447


namespace NUMINAMATH_CALUDE_smallest_N_is_255_l1464_146413

/-- Represents a team in the basketball championship --/
structure Team where
  id : ℕ
  isCalifornian : Bool
  wins : ℕ

/-- Represents the basketball championship --/
structure Championship where
  N : ℕ
  teams : Finset Team
  games : Finset (Team × Team)

/-- The conditions of the championship --/
def ChampionshipConditions (c : Championship) : Prop :=
  -- Total number of teams is 5N
  c.teams.card = 5 * c.N
  -- Every two teams played exactly one game
  ∧ c.games.card = (c.teams.card * (c.teams.card - 1)) / 2
  -- 251 teams are from California
  ∧ (c.teams.filter (λ t => t.isCalifornian)).card = 251
  -- Alcatraz is a Californian team
  ∧ ∃ alcatraz ∈ c.teams, alcatraz.isCalifornian
    -- Alcatraz is the unique Californian champion
    ∧ ∀ t ∈ c.teams, t.isCalifornian → t.wins ≤ alcatraz.wins
    -- Alcatraz is the unique loser of the tournament
    ∧ ∀ t ∈ c.teams, t.id ≠ alcatraz.id → alcatraz.wins < t.wins

/-- The theorem stating that the smallest possible value of N is 255 --/
theorem smallest_N_is_255 :
  ∀ c : Championship, ChampionshipConditions c → c.N ≥ 255 :=
sorry

end NUMINAMATH_CALUDE_smallest_N_is_255_l1464_146413


namespace NUMINAMATH_CALUDE_mia_tv_watching_time_l1464_146423

def minutes_in_day : ℕ := 1440

def studying_minutes : ℕ := 288

theorem mia_tv_watching_time :
  ∃ (x : ℚ), 
    x > 0 ∧ 
    x < 1 ∧ 
    (1 / 4 : ℚ) * (1 - x) * minutes_in_day = studying_minutes ∧
    x = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_mia_tv_watching_time_l1464_146423


namespace NUMINAMATH_CALUDE_m_plus_abs_m_nonnegative_l1464_146495

theorem m_plus_abs_m_nonnegative (m : ℚ) : m + |m| ≥ 0 := by sorry

end NUMINAMATH_CALUDE_m_plus_abs_m_nonnegative_l1464_146495


namespace NUMINAMATH_CALUDE_inequality_proof_l1464_146492

theorem inequality_proof (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 ∧
  ∀ m : ℝ, (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a / (b + c + d + e)) + 
    Real.sqrt (b / (a + c + d + e)) + 
    Real.sqrt (c / (a + b + d + e)) + 
    Real.sqrt (d / (a + b + c + e)) + 
    Real.sqrt (e / (a + b + c + d)) > m) → 
  m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1464_146492


namespace NUMINAMATH_CALUDE_max_value_of_x_l1464_146414

-- Define the tangent function for degrees
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- Define the condition from the problem
def condition (x y : ℝ) : Prop :=
  tan_deg x - tan_deg y = 1 + tan_deg x * tan_deg y

-- State the theorem
theorem max_value_of_x :
  ∃ (x : ℝ), condition x 0 ∧ (∀ (y : ℝ), condition y 0 → y ≤ x) ∧ x = 98721 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_x_l1464_146414


namespace NUMINAMATH_CALUDE_toy_box_problem_l1464_146459

/-- The time taken to put all toys in the box -/
def time_to_fill_box (total_toys : ℕ) (toys_in_per_cycle : ℕ) (toys_out_per_cycle : ℕ) (cycle_time : ℕ) : ℕ := 
  sorry

/-- The problem statement -/
theorem toy_box_problem :
  let total_toys : ℕ := 50
  let toys_in_per_cycle : ℕ := 5
  let toys_out_per_cycle : ℕ := 3
  let cycle_time_seconds : ℕ := 45
  let minutes_per_hour : ℕ := 60
  time_to_fill_box total_toys toys_in_per_cycle toys_out_per_cycle cycle_time_seconds = 18 * minutes_per_hour :=
by sorry

end NUMINAMATH_CALUDE_toy_box_problem_l1464_146459


namespace NUMINAMATH_CALUDE_quadratic_intersection_and_root_distance_l1464_146482

theorem quadratic_intersection_and_root_distance 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2*b * x₁ + c = 0 ∧ a * x₂^2 + 2*b * x₂ + c = 0) ∧
  (∀ x₁ x₂ : ℝ, a * x₁^2 + 2*b * x₁ + c = 0 → a * x₂^2 + 2*b * x₂ + c = 0 → 
    Real.sqrt 3 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_and_root_distance_l1464_146482


namespace NUMINAMATH_CALUDE_larger_number_is_26_l1464_146415

theorem larger_number_is_26 (x y z : ℝ) (sum_eq : x + y = 40) 
  (diff_eq : x - y = 6 * z) (z_eq : z = 2) : x = 26 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_26_l1464_146415


namespace NUMINAMATH_CALUDE_unique_configuration_l1464_146457

/-- A configuration of n points in the plane with associated real numbers -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  r : Fin n → ℝ

/-- The area of a triangle given by three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Predicate for non-collinearity of three points -/
def non_collinear (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

/-- The main theorem: only n = 4 satisfies the conditions -/
theorem unique_configuration :
  ∀ n : ℕ, n > 3 →
  (∃ (config : PointConfiguration n),
    (∀ i j k : Fin n, i < j → j < k →
      non_collinear (config.points i) (config.points j) (config.points k)) ∧
    (∀ i j k : Fin n, i < j → j < k →
      triangle_area (config.points i) (config.points j) (config.points k) =
        config.r i + config.r j + config.r k)) →
  n = 4 := by sorry

end NUMINAMATH_CALUDE_unique_configuration_l1464_146457


namespace NUMINAMATH_CALUDE_tiger_catch_deer_distance_tiger_catch_deer_distance_is_800_l1464_146472

/-- The distance a tiger runs to catch a deer under specific conditions -/
theorem tiger_catch_deer_distance (tiger_leaps_behind : ℕ) 
  (tiger_leaps_per_minute : ℕ) (deer_leaps_per_minute : ℕ)
  (tiger_meters_per_leap : ℕ) (deer_meters_per_leap : ℕ) : ℕ :=
  let initial_distance := tiger_leaps_behind * tiger_meters_per_leap
  let tiger_distance_per_minute := tiger_leaps_per_minute * tiger_meters_per_leap
  let deer_distance_per_minute := deer_leaps_per_minute * deer_meters_per_leap
  let gain_per_minute := tiger_distance_per_minute - deer_distance_per_minute
  let time_to_catch := initial_distance / gain_per_minute
  time_to_catch * tiger_distance_per_minute

/-- The distance a tiger runs to catch a deer is 800 meters under the given conditions -/
theorem tiger_catch_deer_distance_is_800 : 
  tiger_catch_deer_distance 50 5 4 8 5 = 800 := by
  sorry

end NUMINAMATH_CALUDE_tiger_catch_deer_distance_tiger_catch_deer_distance_is_800_l1464_146472


namespace NUMINAMATH_CALUDE_max_large_chips_l1464_146448

theorem max_large_chips (total : ℕ) (small large : ℕ → ℕ) (p : ℕ → ℕ) :
  total = 70 →
  (∀ n, total = small n + large n) →
  (∀ n, Prime (p n)) →
  (∀ n, small n = large n + p n) →
  (∀ n, large n ≤ 34) ∧ (∃ n, large n = 34) :=
by sorry

end NUMINAMATH_CALUDE_max_large_chips_l1464_146448


namespace NUMINAMATH_CALUDE_infinite_congruent_sum_digits_l1464_146477

/-- Sum of digits function -/
def S (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the existence of infinitely many n such that S(n) ≡ n (mod p) for any prime p -/
theorem infinite_congruent_sum_digits (p : ℕ) (hp : Nat.Prime p) :
  ∃ (f : ℕ → ℕ), StrictMono f ∧ ∀ (k : ℕ), S (f k) ≡ f k [MOD p] :=
sorry

end NUMINAMATH_CALUDE_infinite_congruent_sum_digits_l1464_146477


namespace NUMINAMATH_CALUDE_mary_nickels_l1464_146431

/-- The number of nickels Mary has after receiving some from her dad -/
def total_nickels (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Proof that Mary has 12 nickels after receiving 5 from her dad -/
theorem mary_nickels : total_nickels 7 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_l1464_146431


namespace NUMINAMATH_CALUDE_unsold_books_l1464_146484

def initial_stock : ℕ := 800
def monday_sales : ℕ := 60
def tuesday_sales : ℕ := 10
def wednesday_sales : ℕ := 20
def thursday_sales : ℕ := 44
def friday_sales : ℕ := 66

theorem unsold_books :
  initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales) = 600 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_l1464_146484


namespace NUMINAMATH_CALUDE_gcd_654321_543210_l1464_146475

theorem gcd_654321_543210 : Nat.gcd 654321 543210 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_654321_543210_l1464_146475


namespace NUMINAMATH_CALUDE_star_transformation_l1464_146427

theorem star_transformation (a b c d : ℕ) :
  a ∈ Finset.range 17 → b ∈ Finset.range 17 → c ∈ Finset.range 17 → d ∈ Finset.range 17 →
  a + b + c + d = 34 →
  (17 - a) + (17 - b) + (17 - c) + (17 - d) = 34 := by
sorry

end NUMINAMATH_CALUDE_star_transformation_l1464_146427


namespace NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l1464_146400

theorem disjunction_false_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_false_implies_both_false_l1464_146400


namespace NUMINAMATH_CALUDE_x_value_l1464_146434

-- Define the problem statement
theorem x_value (x : ℝ) : x = 70 * (1 + 0.12) → x = 78.4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1464_146434


namespace NUMINAMATH_CALUDE_no_solution_equation_l1464_146435

theorem no_solution_equation :
  ¬ ∃ (x : ℝ), 6 + 3.5 * x = 2.5 * x - 30 + x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l1464_146435


namespace NUMINAMATH_CALUDE_base12_remainder_theorem_l1464_146488

/-- Converts a base-12 integer to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of 2543₁₂ --/
def base12Number : List Nat := [2, 5, 4, 3]

/-- The theorem stating that the remainder of 2543₁₂ divided by 9 is 8 --/
theorem base12_remainder_theorem :
  (base12ToDecimal base12Number) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_base12_remainder_theorem_l1464_146488


namespace NUMINAMATH_CALUDE_similar_triangles_shortest_side_l1464_146451

theorem similar_triangles_shortest_side 
  (a b c : ℝ) -- sides of the first triangle
  (d e f : ℝ) -- sides of the second triangle
  (h1 : a^2 + b^2 = c^2) -- first triangle is right-angled
  (h2 : d^2 + e^2 = f^2) -- second triangle is right-angled
  (h3 : a = 24) -- first condition on first triangle
  (h4 : b = 32) -- second condition on first triangle
  (h5 : f = 80) -- condition on second triangle's hypotenuse
  (h6 : a / d = b / e) -- triangles are similar
  (h7 : b / e = c / f) -- triangles are similar
  : d = 48 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_shortest_side_l1464_146451


namespace NUMINAMATH_CALUDE_tire_circumference_l1464_146476

/-- Given a tire rotating at 400 revolutions per minute on a car traveling at 120 km/h,
    the circumference of the tire is 5 meters. -/
theorem tire_circumference (rpm : ℝ) (speed : ℝ) (circ : ℝ) : 
  rpm = 400 → speed = 120 → circ * rpm = speed * 1000 / 60 → circ = 5 := by
  sorry

#check tire_circumference

end NUMINAMATH_CALUDE_tire_circumference_l1464_146476


namespace NUMINAMATH_CALUDE_project_completion_time_l1464_146418

theorem project_completion_time
  (a b c d e : ℝ)
  (h1 : 1/a + 1/b + 1/c + 1/d = 1/6)
  (h2 : 1/b + 1/c + 1/d + 1/e = 1/8)
  (h3 : 1/a + 1/e = 1/12)
  : e = 48 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l1464_146418


namespace NUMINAMATH_CALUDE_x_equals_plus_minus_fifteen_l1464_146493

theorem x_equals_plus_minus_fifteen (x : ℝ) :
  (x / 5) / 3 = 3 / (x / 5) → x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_plus_minus_fifteen_l1464_146493


namespace NUMINAMATH_CALUDE_sarah_friends_count_l1464_146450

/-- The number of friends Sarah brought into the bedroom -/
def friends_with_sarah (total_people bedroom_people living_room_people : ℕ) : ℕ :=
  total_people - (bedroom_people + living_room_people)

theorem sarah_friends_count :
  ∀ (total_people bedroom_people living_room_people : ℕ),
  total_people = 15 →
  bedroom_people = 3 →
  living_room_people = 8 →
  friends_with_sarah total_people bedroom_people living_room_people = 4 := by
sorry

end NUMINAMATH_CALUDE_sarah_friends_count_l1464_146450


namespace NUMINAMATH_CALUDE_line_AC_equation_circumcircle_equation_l1464_146468

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, 4)

-- Define the line l
def l (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the symmetry condition
def symmetric_about_l (p₁ p₂ : ℝ × ℝ) : Prop :=
  let m := (p₁.1 + p₂.1) / 2
  let n := (p₁.2 + p₂.2) / 2
  l m n

-- Define point C
def C : ℝ × ℝ := (-1, 3)

-- Theorem for the equation of line AC
theorem line_AC_equation (x y : ℝ) : x + y - 2 = 0 ↔ 
  (∃ t : ℝ, x = A.1 + t * (C.1 - A.1) ∧ y = A.2 + t * (C.2 - A.2)) :=
sorry

-- Theorem for the equation of the circumcircle
theorem circumcircle_equation (x y : ℝ) : 
  x^2 + y^2 - 3/2*x + 11/2*y - 17 = 0 ↔ 
  (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
  (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2 :=
sorry

end NUMINAMATH_CALUDE_line_AC_equation_circumcircle_equation_l1464_146468


namespace NUMINAMATH_CALUDE_greatest_divisor_of_sum_first_12_terms_l1464_146485

-- Define an arithmetic sequence of positive integers
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ (x c : ℕ), ∀ n, a n = x + n * c

-- Define the sum of the first 12 terms
def SumFirst12Terms (a : ℕ → ℕ) : ℕ :=
  (List.range 12).map a |>.sum

-- Theorem statement
theorem greatest_divisor_of_sum_first_12_terms :
  ∀ a : ℕ → ℕ, ArithmeticSequence a →
  (∃ k : ℕ, k > 6 ∧ k ∣ SumFirst12Terms a) → False :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_sum_first_12_terms_l1464_146485


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l1464_146470

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, 2007 * x^3 + 2006 * x^2 + 2005 * x = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l1464_146470


namespace NUMINAMATH_CALUDE_student_calculation_l1464_146417

theorem student_calculation (x : ℕ) (h : x = 121) : 2 * x - 140 = 102 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l1464_146417


namespace NUMINAMATH_CALUDE_square_difference_divided_l1464_146425

theorem square_difference_divided : (147^2 - 133^2) / 14 = 280 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_l1464_146425


namespace NUMINAMATH_CALUDE_lulu_blueberry_pies_count_l1464_146432

/-- The number of blueberry pies Lulu baked -/
def lulu_blueberry_pies : ℕ := 73 - (13 + 10 + 8 + 16 + 12)

theorem lulu_blueberry_pies_count :
  lulu_blueberry_pies = 14 := by
  sorry

end NUMINAMATH_CALUDE_lulu_blueberry_pies_count_l1464_146432


namespace NUMINAMATH_CALUDE_distance_to_point_l1464_146466

theorem distance_to_point : ∀ (x y : ℝ), x = 7 ∧ y = -24 →
  Real.sqrt (x^2 + y^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_l1464_146466


namespace NUMINAMATH_CALUDE_dark_tile_fraction_is_16_81_l1464_146412

/-- Represents a tiling system with darker tiles along diagonals -/
structure TilingSystem where
  size : Nat
  corner_size : Nat
  dark_tiles_per_corner : Nat

/-- The fraction of darker tiles in the entire floor -/
def dark_tile_fraction (ts : TilingSystem) : Rat :=
  (4 * ts.dark_tiles_per_corner : Rat) / (ts.size^2 : Rat)

/-- The specific tiling system described in the problem -/
def floor_tiling : TilingSystem :=
  { size := 9
  , corner_size := 4
  , dark_tiles_per_corner := 4 }

/-- Theorem: The fraction of darker tiles in the floor is 16/81 -/
theorem dark_tile_fraction_is_16_81 :
  dark_tile_fraction floor_tiling = 16 / 81 := by
  sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_is_16_81_l1464_146412


namespace NUMINAMATH_CALUDE_rectangle_area_l1464_146489

/-- The length of the shorter side of the smaller rectangles -/
def short_side : ℝ := 7

/-- The length of the longer side of the smaller rectangles -/
def long_side : ℝ := 3 * short_side

/-- The width of the larger rectangle EFGH -/
def width : ℝ := long_side

/-- The length of the larger rectangle EFGH -/
def length : ℝ := long_side + short_side

/-- The area of the larger rectangle EFGH -/
def area : ℝ := length * width

theorem rectangle_area : area = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1464_146489


namespace NUMINAMATH_CALUDE_negation_of_negation_l1464_146411

theorem negation_of_negation : -(-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_negation_l1464_146411


namespace NUMINAMATH_CALUDE_minimum_teams_l1464_146480

theorem minimum_teams (total_players : Nat) (max_team_size : Nat) : total_players = 30 → max_team_size = 8 → ∃ (num_teams : Nat), num_teams = 5 ∧ 
  (∃ (players_per_team : Nat), 
    players_per_team ≤ max_team_size ∧ 
    total_players = num_teams * players_per_team ∧
    ∀ (x : Nat), x < num_teams → 
      total_players % x ≠ 0 ∨ (total_players / x) > max_team_size) := by
  sorry

end NUMINAMATH_CALUDE_minimum_teams_l1464_146480


namespace NUMINAMATH_CALUDE_xiaopang_score_l1464_146491

theorem xiaopang_score (father_score : ℕ) (xiaopang_score : ℕ) : 
  father_score = 48 → 
  xiaopang_score = father_score / 2 - 8 → 
  xiaopang_score = 16 := by
sorry

end NUMINAMATH_CALUDE_xiaopang_score_l1464_146491


namespace NUMINAMATH_CALUDE_five_dice_probability_l1464_146467

/-- A die is represented as a number from 1 to 6 -/
def Die := Fin 6

/-- A roll of five dice -/
def FiveDiceRoll := Fin 5 → Die

/-- The probability space of rolling five fair six-sided dice -/
def Ω : Type := FiveDiceRoll

/-- The probability measure on Ω -/
noncomputable def P : Set Ω → ℝ := sorry

/-- The event that at least three dice show the same value -/
def AtLeastThreeSame (roll : Ω) : Prop := sorry

/-- The sum of the values shown on all dice -/
def DiceSum (roll : Ω) : ℕ := sorry

/-- The event that the sum of all dice is greater than 20 -/
def SumGreaterThan20 (roll : Ω) : Prop := DiceSum roll > 20

/-- The main theorem to be proved -/
theorem five_dice_probability : 
  P {roll : Ω | AtLeastThreeSame roll ∧ SumGreaterThan20 roll} = 31 / 432 := by sorry

end NUMINAMATH_CALUDE_five_dice_probability_l1464_146467


namespace NUMINAMATH_CALUDE_gcd_50421_35343_l1464_146440

theorem gcd_50421_35343 : Nat.gcd 50421 35343 = 23 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50421_35343_l1464_146440


namespace NUMINAMATH_CALUDE_power_product_equality_l1464_146401

theorem power_product_equality : (-0.125)^2022 * 8^2023 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1464_146401


namespace NUMINAMATH_CALUDE_largest_subset_size_l1464_146454

/-- A function that returns the size of the largest subset of {1,2,...,n} where no two elements differ by 5 or 8 -/
def maxSubsetSize (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the largest subset of {1,2,3,...,2023} where no two elements differ by 5 or 8 has 780 elements -/
theorem largest_subset_size :
  maxSubsetSize 2023 = 780 :=
sorry

end NUMINAMATH_CALUDE_largest_subset_size_l1464_146454


namespace NUMINAMATH_CALUDE_blue_notes_under_red_l1464_146497

theorem blue_notes_under_red (red_rows : Nat) (red_per_row : Nat) (additional_blue : Nat) (total_notes : Nat) : Nat :=
  let total_red := red_rows * red_per_row
  let total_blue := total_notes - total_red
  let blue_under_red := (total_blue - additional_blue) / total_red
  blue_under_red

#check blue_notes_under_red 5 6 10 100

end NUMINAMATH_CALUDE_blue_notes_under_red_l1464_146497


namespace NUMINAMATH_CALUDE_complex_sum_zero_l1464_146465

theorem complex_sum_zero : (1 - Complex.I) ^ 10 + (1 + Complex.I) ^ 10 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l1464_146465


namespace NUMINAMATH_CALUDE_problem_solution_l1464_146462

theorem problem_solution (x y z : ℤ) 
  (h1 : x + y = 74)
  (h2 : (x + y) + y + z = 164)
  (h3 : z - y = 16) :
  x = 37 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1464_146462


namespace NUMINAMATH_CALUDE_quadratic_roots_subset_l1464_146486

/-- Set A is defined as the solution set of x^2 + ax + b = 0 -/
def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

/-- Set B is defined as {1, 2} -/
def B : Set ℝ := {1, 2}

/-- The theorem states that given the conditions, (a, b) must be one of the three specified pairs -/
theorem quadratic_roots_subset (a b : ℝ) : 
  A a b ⊆ B ∧ A a b ≠ ∅ → 
  ((a = -2 ∧ b = 1) ∨ (a = -4 ∧ b = 4) ∨ (a = -3 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_subset_l1464_146486


namespace NUMINAMATH_CALUDE_max_z_value_l1464_146494

theorem max_z_value (x y z : ℝ) (sum_eq : x + y + z = 5) (prod_eq : x*y + y*z + z*x = 3) :
  z ≤ 13/3 := by
  sorry

end NUMINAMATH_CALUDE_max_z_value_l1464_146494


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1464_146444

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x => 4 * x^2 - 9 = 0
  let eq2 : ℝ → Prop := λ x => 2 * x^2 - 3 * x - 5 = 0
  let solutions1 : Set ℝ := {3/2, -3/2}
  let solutions2 : Set ℝ := {1, 5/2}
  (∀ x : ℝ, eq1 x ↔ x ∈ solutions1) ∧
  (∀ x : ℝ, eq2 x ↔ x ∈ solutions2) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1464_146444


namespace NUMINAMATH_CALUDE_inequality_implication_l1464_146436

theorem inequality_implication (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^19 / b^19 + b^19 / c^19 + c^19 / a^19 ≤ a^19 / c^19 + b^19 / a^19 + c^19 / b^19) :
  a^20 / b^20 + b^20 / c^20 + c^20 / a^20 ≤ a^20 / c^20 + b^20 / a^20 + c^20 / b^20 :=
by sorry

end NUMINAMATH_CALUDE_inequality_implication_l1464_146436


namespace NUMINAMATH_CALUDE_chelsea_sugar_division_l1464_146420

theorem chelsea_sugar_division (initial_sugar : ℕ) (remaining_sugar : ℕ) 
  (h1 : initial_sugar = 24)
  (h2 : remaining_sugar = 21) :
  ∃ (n : ℕ), n > 0 ∧ 
    (initial_sugar / n) * (n - 1) + (initial_sugar / n / 2) = remaining_sugar ∧
    n = 4 := by
  sorry

end NUMINAMATH_CALUDE_chelsea_sugar_division_l1464_146420


namespace NUMINAMATH_CALUDE_figure_rearrangeable_to_square_l1464_146498

/-- A figure on graph paper can be rearranged into a square if and only if 
    its area (in unit squares) is a perfect square. -/
theorem figure_rearrangeable_to_square (n : ℕ) : 
  (∃ (k : ℕ), n = k^2) ↔ (∃ (s : ℕ), s^2 = n) :=
sorry

end NUMINAMATH_CALUDE_figure_rearrangeable_to_square_l1464_146498


namespace NUMINAMATH_CALUDE_x_percent_of_x_squared_is_nine_l1464_146460

theorem x_percent_of_x_squared_is_nine (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x^2 = 9) :
  ∃ (y : ℝ), abs (x - y) < 0.01 ∧ y^3 = 900 ∧ 
  ∀ (z : ℤ), abs (x - ↑z) ≥ abs (x - 10) :=
sorry

end NUMINAMATH_CALUDE_x_percent_of_x_squared_is_nine_l1464_146460


namespace NUMINAMATH_CALUDE_remainder_double_n_l1464_146449

theorem remainder_double_n (n : ℤ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_double_n_l1464_146449


namespace NUMINAMATH_CALUDE_train_distance_problem_l1464_146424

/-- Theorem: Distance between two stations given train speeds and catch-up point -/
theorem train_distance_problem (v₁ v₂ d : ℝ) :
  v₁ = 1/2 →                   -- passenger train speed in km/min
  v₂ = 1 →                     -- express train speed in km/min
  d = 244/9 →                  -- distance before express train catches up in km
  ∃ x : ℝ,                     -- x is the total distance between stations
    x > 0 ∧                    -- distance is positive
    2/3 * x + 1/2 * (1/3 * x - d) = x - d ∧  -- distance equation
    x = 528                    -- solution
  := by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l1464_146424


namespace NUMINAMATH_CALUDE_equation_equivalence_l1464_146406

theorem equation_equivalence :
  ∀ x : ℝ, x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1464_146406


namespace NUMINAMATH_CALUDE_position_of_three_fifths_l1464_146471

def sequence_sum (n : ℕ) : ℕ := n - 1

def position_in_group (n m : ℕ) : ℕ := 
  (sequence_sum n * (sequence_sum n + 1)) / 2 + m

theorem position_of_three_fifths : 
  position_in_group 8 3 = 24 := by sorry

end NUMINAMATH_CALUDE_position_of_three_fifths_l1464_146471


namespace NUMINAMATH_CALUDE_cube_volume_from_circumscribed_sphere_l1464_146479

/-- Given a cube with a circumscribed sphere of volume 32π/3, the volume of the cube is 64√3/9 -/
theorem cube_volume_from_circumscribed_sphere (V_sphere : ℝ) (V_cube : ℝ) :
  V_sphere = 32 / 3 * Real.pi → V_cube = 64 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_circumscribed_sphere_l1464_146479


namespace NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l1464_146458

theorem sunglasses_and_caps_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_cap_and_sunglasses : ℚ) :
  total_sunglasses = 60 →
  total_caps = 40 →
  prob_cap_and_sunglasses = 2/5 →
  (prob_cap_and_sunglasses * total_caps) / total_sunglasses = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l1464_146458


namespace NUMINAMATH_CALUDE_possible_values_of_x_l1464_146496

theorem possible_values_of_x (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 225)
  (h3 : x < 30) :
  x = 18 ∨ x = 27 :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_x_l1464_146496


namespace NUMINAMATH_CALUDE_school_purchase_cost_l1464_146445

/-- The total cost of purchasing sweaters and sports shirts -/
def total_cost (sweater_price : ℕ) (sweater_quantity : ℕ) 
               (shirt_price : ℕ) (shirt_quantity : ℕ) : ℕ :=
  sweater_price * sweater_quantity + shirt_price * shirt_quantity

/-- Theorem stating that the total cost for the given quantities and prices is 5400 yuan -/
theorem school_purchase_cost : 
  total_cost 98 25 59 50 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_school_purchase_cost_l1464_146445


namespace NUMINAMATH_CALUDE_triangle_side_range_l1464_146483

theorem triangle_side_range (x : ℝ) : 
  x > 0 → 
  (4 + 5 > x ∧ 4 + x > 5 ∧ 5 + x > 4) → 
  1 < x ∧ x < 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1464_146483


namespace NUMINAMATH_CALUDE_chips_bought_l1464_146410

/-- Given three friends paying $5 each for bags of chips costing $3 per bag,
    prove that they can buy 5 bags of chips. -/
theorem chips_bought (num_friends : ℕ) (payment_per_friend : ℕ) (cost_per_bag : ℕ) :
  num_friends = 3 →
  payment_per_friend = 5 →
  cost_per_bag = 3 →
  (num_friends * payment_per_friend) / cost_per_bag = 5 :=
by sorry

end NUMINAMATH_CALUDE_chips_bought_l1464_146410


namespace NUMINAMATH_CALUDE_speaking_sequences_count_l1464_146430

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def totalStudents : ℕ := 6

/-- The number of speakers to be selected -/
def speakersToSelect : ℕ := 4

/-- The number of specific students (A and B) -/
def specificStudents : ℕ := 2

theorem speaking_sequences_count :
  (choose specificStudents 1 * choose (totalStudents - specificStudents) (speakersToSelect - 1) * arrange speakersToSelect speakersToSelect) +
  (choose specificStudents 2 * choose (totalStudents - specificStudents) (speakersToSelect - 2) * arrange speakersToSelect speakersToSelect) = 336 :=
by sorry

end NUMINAMATH_CALUDE_speaking_sequences_count_l1464_146430


namespace NUMINAMATH_CALUDE_percentage_of_juniors_l1464_146463

theorem percentage_of_juniors (total_students : ℕ) (juniors_in_sports : ℕ) 
  (sports_percentage : ℚ) (h1 : total_students = 500) 
  (h2 : juniors_in_sports = 140) (h3 : sports_percentage = 70 / 100) :
  (juniors_in_sports / sports_percentage) / total_students = 40 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_juniors_l1464_146463


namespace NUMINAMATH_CALUDE_fraction_simplification_l1464_146416

theorem fraction_simplification (a b : ℝ) : (9 * b) / (6 * a + 3) = (3 * b) / (2 * a + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1464_146416


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1464_146487

theorem gcd_of_three_numbers : Nat.gcd 9240 (Nat.gcd 12240 33720) = 240 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1464_146487


namespace NUMINAMATH_CALUDE_watch_time_calculation_l1464_146426

/-- The total watching time for two shows, where the second is 4 times longer than the first -/
def total_watching_time (first_show_duration : ℕ) : ℕ :=
  first_show_duration + 4 * first_show_duration

/-- Theorem stating that given a 30-minute show and another 4 times longer, the total watching time is 150 minutes -/
theorem watch_time_calculation : total_watching_time 30 = 150 := by
  sorry

end NUMINAMATH_CALUDE_watch_time_calculation_l1464_146426


namespace NUMINAMATH_CALUDE_arc_length_example_l1464_146404

/-- The length of an arc in a circle, given the radius and central angle -/
def arcLength (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  radius * centralAngle

theorem arc_length_example :
  let radius : ℝ := 10
  let centralAngle : ℝ := 2 * Real.pi / 3
  arcLength radius centralAngle = 20 * Real.pi / 3 := by
sorry


end NUMINAMATH_CALUDE_arc_length_example_l1464_146404


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l1464_146446

theorem binomial_coefficient_problem (a : ℝ) : 
  (Nat.choose 7 6 : ℝ) * a = 7 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l1464_146446


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l1464_146437

theorem constant_ratio_problem (x y : ℝ) (k : ℝ) : 
  ((5 * x - 3) / (2 * y + 10) = k) →  -- The ratio is constant
  (y = 2 → x = 3) →                   -- When y = 2, x = 3
  (y = 5 → x = 47 / 5) :=             -- When y = 5, x = 47/5
by
  sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l1464_146437


namespace NUMINAMATH_CALUDE_f_one_upper_bound_l1464_146461

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - m * x + 5

-- State the theorem
theorem f_one_upper_bound (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ -2 → f m x₁ > f m x₂) →
  f m 1 ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_f_one_upper_bound_l1464_146461


namespace NUMINAMATH_CALUDE_tv_selection_combinations_l1464_146428

def num_type_a : ℕ := 4
def num_type_b : ℕ := 5
def num_to_choose : ℕ := 3

def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem tv_selection_combinations : 
  (combinations num_type_a 2 * combinations num_type_b 1) + 
  (combinations num_type_a 1 * combinations num_type_b 2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_tv_selection_combinations_l1464_146428


namespace NUMINAMATH_CALUDE_valid_schedule_count_is_twelve_l1464_146481

/-- Represents the four subjects in the class schedule -/
inductive Subject
| Chinese
| Mathematics
| English
| PhysicalEducation

/-- Represents a schedule of four periods -/
def Schedule := Fin 4 → Subject

/-- Checks if a schedule is valid (PE is not in first or fourth period) -/
def isValidSchedule (s : Schedule) : Prop :=
  s 0 ≠ Subject.PhysicalEducation ∧ s 3 ≠ Subject.PhysicalEducation

/-- The number of valid schedules -/
def validScheduleCount : ℕ := sorry

/-- Theorem stating that the number of valid schedules is 12 -/
theorem valid_schedule_count_is_twelve : validScheduleCount = 12 := by sorry

end NUMINAMATH_CALUDE_valid_schedule_count_is_twelve_l1464_146481


namespace NUMINAMATH_CALUDE_not_geometric_complement_sequence_l1464_146422

/-- Given a geometric sequence a, b, c with common ratio q ≠ 1,
    prove that 1-a, 1-b, 1-c cannot form a geometric sequence. -/
theorem not_geometric_complement_sequence 
  (a b c q : ℝ) 
  (h1 : b = a * q) 
  (h2 : c = b * q) 
  (h3 : q ≠ 1) : 
  ¬ ∃ r : ℝ, (1 - b = (1 - a) * r ∧ 1 - c = (1 - b) * r) :=
sorry

end NUMINAMATH_CALUDE_not_geometric_complement_sequence_l1464_146422


namespace NUMINAMATH_CALUDE_rectangle_area_l1464_146409

/-- Given a wire of length 32 cm bent into a rectangle with a length-to-width ratio of 5:3,
    the area of the resulting rectangle is 60 cm². -/
theorem rectangle_area (wire_length : ℝ) (length : ℝ) (width : ℝ) : 
  wire_length = 32 →
  length / width = 5 / 3 →
  2 * (length + width) = wire_length →
  length * width = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1464_146409


namespace NUMINAMATH_CALUDE_slope_of_line_AB_l1464_146499

/-- Given points A(2, 0) and B(3, √3), prove that the slope of line AB is √3 -/
theorem slope_of_line_AB (A B : ℝ × ℝ) : 
  A = (2, 0) → B = (3, Real.sqrt 3) → (B.2 - A.2) / (B.1 - A.1) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_AB_l1464_146499


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1464_146478

theorem complex_equation_solution (i : ℂ) (h_i : i^2 = -1) :
  ∃ z : ℂ, (2 + i) * z = 2 - i ∧ z = 3/5 - 4/5 * i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1464_146478


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1464_146441

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The main theorem -/
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_a3 : a 3 = 2) 
  (h_a4a6 : a 4 * a 6 = 16) : 
  (a 9 - a 11) / (a 5 - a 7) = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1464_146441


namespace NUMINAMATH_CALUDE_extreme_value_conditions_max_min_values_l1464_146407

/-- The function f(x) = x^3 + 3ax^2 + bx -/
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x

/-- The derivative of f(x) -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + b

theorem extreme_value_conditions (a b : ℝ) :
  f a b (-1) = 0 ∧ f_deriv a b (-1) = 0 →
  a = 2/3 ∧ b = 1 :=
sorry

theorem max_min_values (a b : ℝ) :
  a = 2/3 ∧ b = 1 →
  (∀ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x ≤ 0) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x = 0) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) (-1/4), f a b x = -2) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_conditions_max_min_values_l1464_146407


namespace NUMINAMATH_CALUDE_two_digit_swap_difference_l1464_146456

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  tens_valid : tens < 10
  units_valid : units < 10

/-- Calculates the value of a two-digit number -/
def value (n : TwoDigitNumber) : ℕ := 10 * n.tens + n.units

/-- Swaps the digits of a two-digit number -/
def swap_digits (n : TwoDigitNumber) : TwoDigitNumber := {
  tens := n.units,
  units := n.tens,
  tens_valid := n.units_valid,
  units_valid := n.tens_valid
}

/-- 
Theorem: The difference between a two-digit number with its digits swapped
and the original number is equal to -9x + 9y, where x is the tens digit
and y is the units digit of the original number.
-/
theorem two_digit_swap_difference (n : TwoDigitNumber) :
  (value (swap_digits n) : ℤ) - (value n : ℤ) = -9 * (n.tens : ℤ) + 9 * (n.units : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_swap_difference_l1464_146456


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l1464_146442

/-- A hyperbola with one known asymptote and foci on a vertical line -/
structure Hyperbola where
  /-- The slope of the known asymptote -/
  known_asymptote_slope : ℝ
  /-- The x-coordinate of the line containing the foci -/
  foci_x : ℝ

/-- The equation of the other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => y = (-h.known_asymptote_slope) * x + (h.known_asymptote_slope + 1) * h.foci_x * 2

theorem other_asymptote_equation (h : Hyperbola) 
    (h_slope : h.known_asymptote_slope = 4) 
    (h_foci : h.foci_x = 3) :
    other_asymptote h = fun x y => y = -4 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_other_asymptote_equation_l1464_146442


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1464_146455

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

-- Define the intersection condition
def intersection_condition (m : ℝ) : Prop := A m ∩ B = {4}

-- Define sufficiency
def is_sufficient (m : ℝ) : Prop := m = -2 → intersection_condition m

-- Define non-necessity
def is_not_necessary (m : ℝ) : Prop := ∃ x : ℝ, x ≠ -2 ∧ intersection_condition x

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ m : ℝ, is_sufficient m) ∧ (∃ m : ℝ, is_not_necessary m) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1464_146455


namespace NUMINAMATH_CALUDE_angle_approximation_l1464_146464

/-- Regular polygon with n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  center : ℝ × ℝ
  radius : ℝ
  vertices : Fin n → ℝ × ℝ

/-- Construct points B, C, D, E as described in the problem -/
def constructPoints (p : RegularPolygon 19) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := sorry

/-- Length of chord DE -/
def chordLength (p : RegularPolygon 19) : ℝ := sorry

/-- Angle formed by radii after 19 sequential measurements -/
def angleAfterMeasurements (p : RegularPolygon 19) : ℝ := sorry

/-- Main theorem: The angle formed after 19 measurements is approximately 4°57' -/
theorem angle_approximation (p : RegularPolygon 19) : 
  ∃ ε > 0, abs (angleAfterMeasurements p - (4 + 57/60) * π / 180) < ε :=
sorry

end NUMINAMATH_CALUDE_angle_approximation_l1464_146464


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1464_146439

theorem arithmetic_calculation : (21 / (6 + 1 - 4)) * 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1464_146439


namespace NUMINAMATH_CALUDE_peter_twice_harriet_age_l1464_146403

/- Define the current ages and time span -/
def mother_age : ℕ := 60
def harriet_age : ℕ := 13
def years_passed : ℕ := 4

/- Define Peter's current age based on the given condition -/
def peter_age : ℕ := mother_age / 2

/- Define future ages -/
def peter_future_age : ℕ := peter_age + years_passed
def harriet_future_age : ℕ := harriet_age + years_passed

/- Theorem to prove -/
theorem peter_twice_harriet_age : 
  peter_future_age = 2 * harriet_future_age := by
  sorry


end NUMINAMATH_CALUDE_peter_twice_harriet_age_l1464_146403


namespace NUMINAMATH_CALUDE_trip_distance_proof_l1464_146453

/-- Represents the total length of the trip in miles. -/
def total_distance : ℝ := 150

/-- Represents the distance traveled on battery power in miles. -/
def battery_distance : ℝ := 50

/-- Represents the fuel consumption rate in gallons per mile. -/
def fuel_rate : ℝ := 0.03

/-- Represents the average fuel efficiency for the entire trip in miles per gallon. -/
def avg_efficiency : ℝ := 50

theorem trip_distance_proof :
  (total_distance / (fuel_rate * (total_distance - battery_distance)) = avg_efficiency) ∧
  (total_distance > battery_distance) :=
by sorry

end NUMINAMATH_CALUDE_trip_distance_proof_l1464_146453


namespace NUMINAMATH_CALUDE_problem_solution_l1464_146421

theorem problem_solution : (6 * 5 * 4 * 3 * 2 * 1 - 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 25 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1464_146421


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l1464_146433

def M : Set ℝ := {x : ℝ | x ≥ 1}
def N : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}

theorem complement_intersection_M_N :
  (M ∩ N)ᶜ = {x : ℝ | x < 1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l1464_146433


namespace NUMINAMATH_CALUDE_investment_growth_l1464_146438

/-- The monthly interest rate for an investment that grows from $300 to $363 in 2 months -/
def monthly_interest_rate : ℝ :=
  0.1

theorem investment_growth (initial_investment : ℝ) (final_amount : ℝ) (months : ℕ) :
  initial_investment = 300 →
  final_amount = 363 →
  months = 2 →
  final_amount = initial_investment * (1 + monthly_interest_rate) ^ months :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l1464_146438


namespace NUMINAMATH_CALUDE_all_statements_incorrect_l1464_146408

-- Define the types for functions and properties
def Function := ℝ → ℝ
def Periodic (f : Function) : Prop := ∃ T > 0, ∀ x, f (x + T) = f x
def Monotonic (f : Function) : Prop := ∀ x y, x < y → f x < f y

-- Define the original proposition
def OriginalProposition : Prop := ∀ f : Function, Periodic f → ¬(Monotonic f)

-- Define the given statements
def GivenConverse : Prop := ∀ f : Function, Monotonic f → ¬(Periodic f)
def GivenNegation : Prop := ∀ f : Function, Periodic f → Monotonic f
def GivenContrapositive : Prop := ∀ f : Function, Monotonic f → Periodic f

-- Theorem stating that none of the given statements are correct
theorem all_statements_incorrect : 
  (GivenConverse ≠ (¬OriginalProposition → OriginalProposition)) ∧
  (GivenNegation ≠ ¬OriginalProposition) ∧
  (GivenContrapositive ≠ (¬¬OriginalProposition → ¬OriginalProposition)) :=
sorry

end NUMINAMATH_CALUDE_all_statements_incorrect_l1464_146408


namespace NUMINAMATH_CALUDE_min_value_of_f_l1464_146402

/-- The quadratic function f(x) = x^2 + 12x + 5 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 5

/-- The minimum value of f(x) is -31 -/
theorem min_value_of_f : ∀ x : ℝ, f x ≥ -31 ∧ ∃ y : ℝ, f y = -31 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1464_146402


namespace NUMINAMATH_CALUDE_female_math_only_result_l1464_146473

/-- The number of female students who participated in the math competition but not in the English competition -/
def female_math_only (male_math female_math female_eng male_eng total male_both : ℕ) : ℕ :=
  let male_total := male_math + male_eng - male_both
  let female_total := total - male_total
  let female_both := female_math + female_eng - female_total
  female_math - female_both

/-- Theorem stating the result of the problem -/
theorem female_math_only_result : 
  female_math_only 120 80 120 80 260 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_female_math_only_result_l1464_146473


namespace NUMINAMATH_CALUDE_right_focus_coordinates_l1464_146490

/-- An ellipse with parametric equations x = 2cos(θ) and y = √3sin(θ) -/
structure Ellipse where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ θ, x θ = 2 * Real.cos θ
  h_y : ∀ θ, y θ = Real.sqrt 3 * Real.sin θ

/-- The right focus of an ellipse -/
def right_focus (e : Ellipse) : ℝ × ℝ :=
  (1, 0)

/-- Theorem: The right focus of the given ellipse has coordinates (1, 0) -/
theorem right_focus_coordinates (e : Ellipse) : right_focus e = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_right_focus_coordinates_l1464_146490


namespace NUMINAMATH_CALUDE_coin_toss_probability_l1464_146443

/-- The probability of getting a specific sequence of heads and tails in 10 coin tosses -/
theorem coin_toss_probability : 
  let n : ℕ := 10  -- number of tosses
  let p : ℚ := 1/2  -- probability of heads (or tails) in a single toss
  (p ^ n : ℚ) = 1/1024 := by sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l1464_146443


namespace NUMINAMATH_CALUDE_trig_identity_l1464_146429

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.sin (π/3 - x))^2 = 19/16 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1464_146429


namespace NUMINAMATH_CALUDE_units_digit_of_j_squared_plus_three_to_j_l1464_146474

theorem units_digit_of_j_squared_plus_three_to_j (j : ℕ) : 
  j = 2023^3 + 3^2023 + 2023 → (j^2 + 3^j) % 10 = 6 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_j_squared_plus_three_to_j_l1464_146474


namespace NUMINAMATH_CALUDE_two_digit_numbers_product_gcd_l1464_146469

theorem two_digit_numbers_product_gcd (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 1728 ∧ 
  Nat.gcd a b = 12 →
  (a = 36 ∧ b = 48) ∨ (a = 48 ∧ b = 36) := by
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_product_gcd_l1464_146469


namespace NUMINAMATH_CALUDE_solve_equation_l1464_146452

theorem solve_equation (x : ℚ) : (3 * x - 7) / 4 = 15 → x = 67 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1464_146452
