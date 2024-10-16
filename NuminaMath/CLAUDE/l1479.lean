import Mathlib

namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l1479_147943

theorem bucket_capacity_reduction (original_buckets reduced_buckets : ℚ) 
  (h1 : original_buckets = 25)
  (h2 : reduced_buckets = 62.5)
  (h3 : original_buckets * original_capacity = reduced_buckets * reduced_capacity) :
  reduced_capacity / original_capacity = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l1479_147943


namespace NUMINAMATH_CALUDE_profit_share_difference_theorem_l1479_147987

/-- Calculates the difference between profit shares of two partners given their investments and a known profit share of the third partner. -/
def profit_share_difference (invest_a invest_b invest_c b_profit : ℕ) : ℕ :=
  let total_investment := invest_a + invest_b + invest_c
  let total_profit := b_profit * total_investment / invest_b
  let a_profit := total_profit * invest_a / total_investment
  let c_profit := total_profit * invest_c / total_investment
  c_profit - a_profit

/-- The difference between profit shares of a and c is 600 given their investments and b's profit share. -/
theorem profit_share_difference_theorem :
  profit_share_difference 8000 10000 12000 1500 = 600 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_theorem_l1479_147987


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l1479_147901

/-- The number of entries in the nth row of Pascal's Triangle -/
def pascalRowEntries (n : ℕ) : ℕ := n + 1

/-- The sum of entries in the first n rows of Pascal's Triangle -/
def pascalTriangleSum (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem pascal_triangle_30_rows_sum :
  pascalTriangleSum 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l1479_147901


namespace NUMINAMATH_CALUDE_line_perp_plane_from_conditions_l1479_147972

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and lines
variable (perp_plane_line : Plane → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_from_conditions 
  (α β : Plane) (m n : Line) 
  (h1 : perp_plane_line α n) 
  (h2 : perp_plane_line β n) 
  (h3 : perp_plane_line α m) : 
  perp_plane_line β m :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_from_conditions_l1479_147972


namespace NUMINAMATH_CALUDE_odd_number_induction_l1479_147989

theorem odd_number_induction (P : ℕ → Prop) 
  (base : P 1) 
  (step : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) : 
  ∀ n : ℕ, n % 2 = 1 → P n := by
  sorry

end NUMINAMATH_CALUDE_odd_number_induction_l1479_147989


namespace NUMINAMATH_CALUDE_min_packs_for_120_cans_l1479_147993

/-- Represents the available pack sizes for soda cans -/
def PackSizes : List Nat := [6, 12, 24, 30]

/-- The total number of cans needed -/
def TotalCans : Nat := 120

/-- A function that checks if a combination of packs can exactly make the total number of cans -/
def canMakeTotalCans (packs : List Nat) : Bool :=
  (packs.map (fun size => size * (packs.count size))).sum = TotalCans

/-- Theorem stating that the minimum number of packs needed to buy exactly 120 cans is 4 -/
theorem min_packs_for_120_cans :
  ∃ (packs : List Nat),
    packs.all (PackSizes.contains ·) ∧
    canMakeTotalCans packs ∧
    packs.length = 4 ∧
    (∀ (other_packs : List Nat),
      other_packs.all (PackSizes.contains ·) →
      canMakeTotalCans other_packs →
      other_packs.length ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_min_packs_for_120_cans_l1479_147993


namespace NUMINAMATH_CALUDE_points_earned_l1479_147940

def points_per_enemy : ℕ := 3
def total_enemies : ℕ := 6
def enemies_not_defeated : ℕ := 2

theorem points_earned : 
  (total_enemies - enemies_not_defeated) * points_per_enemy = 12 := by
  sorry

end NUMINAMATH_CALUDE_points_earned_l1479_147940


namespace NUMINAMATH_CALUDE_log_stacks_total_l1479_147923

def first_stack_start : ℕ := 15
def first_stack_end : ℕ := 4
def second_stack_start : ℕ := 5
def second_stack_end : ℕ := 10

def total_logs : ℕ := 159

theorem log_stacks_total :
  (first_stack_start - first_stack_end + 1) * (first_stack_start + first_stack_end) / 2 +
  (second_stack_end - second_stack_start + 1) * (second_stack_start + second_stack_end) / 2 =
  total_logs := by
  sorry

end NUMINAMATH_CALUDE_log_stacks_total_l1479_147923


namespace NUMINAMATH_CALUDE_fraction_equality_l1479_147933

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1479_147933


namespace NUMINAMATH_CALUDE_total_weekly_calories_l1479_147981

/-- Represents the number of each type of burger consumed on a given day -/
structure DailyConsumption where
  burgerA : ℕ
  burgerB : ℕ
  burgerC : ℕ

/-- Calculates the total calories for a given daily consumption -/
def dailyCalories (d : DailyConsumption) : ℕ :=
  d.burgerA * 350 + d.burgerB * 450 + d.burgerC * 550

/-- Represents Dimitri's burger consumption for the week -/
def weeklyConsumption : List DailyConsumption :=
  [
    ⟨2, 1, 0⟩,  -- Day 1
    ⟨1, 2, 1⟩,  -- Day 2
    ⟨1, 1, 2⟩,  -- Day 3
    ⟨0, 3, 0⟩,  -- Day 4
    ⟨1, 1, 1⟩,  -- Day 5
    ⟨2, 0, 3⟩,  -- Day 6
    ⟨0, 1, 2⟩   -- Day 7
  ]

/-- Theorem: The total calories consumed by Dimitri in a week is 11,450 -/
theorem total_weekly_calories : 
  (weeklyConsumption.map dailyCalories).sum = 11450 := by
  sorry


end NUMINAMATH_CALUDE_total_weekly_calories_l1479_147981


namespace NUMINAMATH_CALUDE_fraction_value_when_a_equals_4b_l1479_147905

theorem fraction_value_when_a_equals_4b (a b : ℝ) (h : a = 4 * b) :
  (a^2 + b^2) / (a * b) = 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_when_a_equals_4b_l1479_147905


namespace NUMINAMATH_CALUDE_min_value_expression_l1479_147955

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^4 / (y - 1)) + (y^4 / (x - 1)) ≥ 12 ∧
  ((x^4 / (y - 1)) + (y^4 / (x - 1)) = 12 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1479_147955


namespace NUMINAMATH_CALUDE_same_terminal_side_l1479_147900

theorem same_terminal_side : ∀ (k : ℤ), 95 = -265 + k * 360 → 95 ≡ -265 [ZMOD 360] := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l1479_147900


namespace NUMINAMATH_CALUDE_moray_eel_eats_twenty_l1479_147950

/-- The number of guppies eaten by a moray eel per day, given the total number of guppies needed,
    the number of betta fish, and the number of guppies eaten by each betta fish per day. -/
def moray_eel_guppies (total_guppies : ℕ) (num_betta : ℕ) (betta_guppies : ℕ) : ℕ :=
  total_guppies - (num_betta * betta_guppies)

/-- Theorem stating that the number of guppies eaten by the moray eel is 20,
    given the conditions in the problem. -/
theorem moray_eel_eats_twenty :
  moray_eel_guppies 55 5 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_moray_eel_eats_twenty_l1479_147950


namespace NUMINAMATH_CALUDE_six_by_six_grid_squares_l1479_147942

/-- The number of squares of a given size in a grid --/
def count_squares (grid_size : ℕ) (square_size : ℕ) : ℕ :=
  (grid_size + 1 - square_size) ^ 2

/-- The total number of squares in a 6x6 grid --/
def total_squares (grid_size : ℕ) : ℕ :=
  (count_squares grid_size 1) + (count_squares grid_size 2) +
  (count_squares grid_size 3) + (count_squares grid_size 4)

/-- Theorem: The total number of squares in a 6x6 grid is 86 --/
theorem six_by_six_grid_squares :
  total_squares 6 = 86 := by
  sorry


end NUMINAMATH_CALUDE_six_by_six_grid_squares_l1479_147942


namespace NUMINAMATH_CALUDE_decagon_triangles_l1479_147946

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def num_triangles_in_decagon : ℕ := 120

/-- Theorem: The number of triangles that can be formed using the vertices of a regular decagon is 120 -/
theorem decagon_triangles : num_triangles_in_decagon = 120 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangles_l1479_147946


namespace NUMINAMATH_CALUDE_disk_difference_l1479_147986

/-- Given a bag of disks with blue, yellow, and green colors, prove the difference between green and blue disks -/
theorem disk_difference (total : ℕ) (blue_ratio yellow_ratio green_ratio : ℕ) : 
  total = 126 →
  blue_ratio = 3 →
  yellow_ratio = 7 →
  green_ratio = 8 →
  (green_ratio * (total / (blue_ratio + yellow_ratio + green_ratio))) - 
  (blue_ratio * (total / (blue_ratio + yellow_ratio + green_ratio))) = 35 := by
  sorry

end NUMINAMATH_CALUDE_disk_difference_l1479_147986


namespace NUMINAMATH_CALUDE_expected_potato_yield_l1479_147911

/-- Calculates the expected potato yield from a rectangular garden --/
theorem expected_potato_yield
  (length_steps : ℕ)
  (width_steps : ℕ)
  (step_length : ℝ)
  (yield_per_sqft : ℝ)
  (h1 : length_steps = 18)
  (h2 : width_steps = 25)
  (h3 : step_length = 3)
  (h4 : yield_per_sqft = 0.75)
  : ↑length_steps * step_length * (↑width_steps * step_length) * yield_per_sqft = 3037.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_potato_yield_l1479_147911


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1479_147975

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {(1, 1), (-1, 1)} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1479_147975


namespace NUMINAMATH_CALUDE_stating_basketball_tournament_wins_l1479_147937

/-- Represents the number of games won in a basketball tournament -/
def games_won (total_games : ℕ) (total_points : ℕ) : ℕ :=
  total_games - (2 * total_games - total_points)

/-- 
Theorem stating that given 8 games where wins earn 2 points and losses earn 1 point, 
if the total points earned is 13, then the number of games won is 5.
-/
theorem basketball_tournament_wins :
  games_won 8 13 = 5 := by
  sorry

#eval games_won 8 13  -- Should output 5

end NUMINAMATH_CALUDE_stating_basketball_tournament_wins_l1479_147937


namespace NUMINAMATH_CALUDE_film_review_analysis_l1479_147984

structure FilmReviewData where
  total_sample : ℕ
  male_count : ℕ
  female_count : ℕ
  male_negative : ℕ
  female_positive : ℕ
  significance_level : ℝ
  chi_square_critical : ℝ
  stratified_sample_size : ℕ
  coupon_recipients : ℕ

def chi_square_statistic (data : FilmReviewData) : ℝ := sorry

def is_associated (data : FilmReviewData) : Prop :=
  chi_square_statistic data > data.chi_square_critical

def probability_distribution (x : ℕ) : ℝ := sorry

def expected_value : ℝ := sorry

theorem film_review_analysis (data : FilmReviewData) 
  (h1 : data.total_sample = 220)
  (h2 : data.male_count = 110)
  (h3 : data.female_count = 110)
  (h4 : data.male_negative = 70)
  (h5 : data.female_positive = 60)
  (h6 : data.significance_level = 0.010)
  (h7 : data.chi_square_critical = 6.635)
  (h8 : data.stratified_sample_size = 10)
  (h9 : data.coupon_recipients = 3) :
  is_associated data ∧ 
  probability_distribution 0 = 1/30 ∧
  probability_distribution 1 = 3/10 ∧
  probability_distribution 2 = 1/2 ∧
  probability_distribution 3 = 1/6 ∧
  expected_value = 9/5 := by sorry

end NUMINAMATH_CALUDE_film_review_analysis_l1479_147984


namespace NUMINAMATH_CALUDE_inequality_theorem_l1479_147927

theorem inequality_theorem (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  c / a > c / b :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1479_147927


namespace NUMINAMATH_CALUDE_inequality_proof_l1479_147945

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x + y ≤ (y^2 / x) + (x^2 / y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1479_147945


namespace NUMINAMATH_CALUDE_workers_required_l1479_147982

/-- Given a craft factory that needs to produce 60 units per day, 
    and each worker can produce x units per day, 
    prove that the number of workers required y is equal to 60/x -/
theorem workers_required (x : ℝ) (h : x > 0) : 
  ∃ y : ℝ, y * x = 60 ∧ y = 60 / x := by
  sorry

end NUMINAMATH_CALUDE_workers_required_l1479_147982


namespace NUMINAMATH_CALUDE_wilsons_theorem_l1479_147968

theorem wilsons_theorem (p : ℕ) (hp : Prime p) : (Nat.factorial (p - 1)) % p = p - 1 := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l1479_147968


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1479_147980

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 4 * x + y = 12) (h2 : x + 4 * y = 16) : 
  17 * x^2 + 18 * x * y + 17 * y^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1479_147980


namespace NUMINAMATH_CALUDE_function_equation_implies_identity_l1479_147925

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x + y) * (f (x - y) + 1) = f (f (x * f (x + 1)) - y * f (y - 1))) :
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_identity_l1479_147925


namespace NUMINAMATH_CALUDE_vector_AB_after_translation_l1479_147951

def point_A : ℝ × ℝ := (3, 7)
def point_B : ℝ × ℝ := (5, 2)
def vector_a : ℝ × ℝ := (1, 2)

def vector_AB : ℝ × ℝ := (point_B.1 - point_A.1, point_B.2 - point_A.2)

theorem vector_AB_after_translation :
  vector_AB = (2, -5) := by sorry

end NUMINAMATH_CALUDE_vector_AB_after_translation_l1479_147951


namespace NUMINAMATH_CALUDE_min_magnitude_sum_vectors_l1479_147931

/-- Given two vectors a and b in a real inner product space, with magnitudes 8 and 12 respectively,
    the minimum value of the magnitude of their sum is 4. -/
theorem min_magnitude_sum_vectors {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (ha : ‖a‖ = 8) (hb : ‖b‖ = 12) : 
  ∃ (sum : V), sum = a + b ∧ ‖sum‖ = 4 ∧ ∀ (x : V), x = a + b → ‖x‖ ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_magnitude_sum_vectors_l1479_147931


namespace NUMINAMATH_CALUDE_haji_mother_sales_l1479_147913

/-- Calculate the total sales for Haji's mother given the following conditions:
  - Tough week sales: $800
  - Tough week sales are half of good week sales
  - Number of good weeks: 5
  - Number of tough weeks: 3
-/
theorem haji_mother_sales (tough_week_sales : ℕ) (good_weeks : ℕ) (tough_weeks : ℕ)
  (h1 : tough_week_sales = 800)
  (h2 : good_weeks = 5)
  (h3 : tough_weeks = 3) :
  tough_week_sales * 2 * good_weeks + tough_week_sales * tough_weeks = 10400 :=
by sorry

end NUMINAMATH_CALUDE_haji_mother_sales_l1479_147913


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l1479_147970

/-- Represents a circular seating arrangement -/
structure CircularSeating where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if a seating arrangement is valid -/
def is_valid_seating (s : CircularSeating) : Prop :=
  s.seated_people > 0 ∧ 
  s.seated_people ≤ s.total_chairs ∧
  ∀ (new_seat : ℕ), new_seat < s.total_chairs → 
    ∃ (adjacent : ℕ), adjacent < s.total_chairs ∧ 
      (new_seat + 1) % s.total_chairs = adjacent ∨ 
      (new_seat + s.total_chairs - 1) % s.total_chairs = adjacent

/-- The main theorem to prove -/
theorem smallest_valid_seating :
  ∀ (s : CircularSeating), 
    s.total_chairs = 72 → 
    (is_valid_seating s ↔ s.seated_people ≥ 18) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l1479_147970


namespace NUMINAMATH_CALUDE_tourist_cyclist_speed_problem_l1479_147936

/-- Represents the problem of finding the maximum speed of a tourist and the corresponding speed of a cyclist --/
theorem tourist_cyclist_speed_problem 
  (distance : ℝ) 
  (min_cyclist_time : ℝ) 
  (cyclist_speed_increase : ℝ) 
  (meet_time : ℝ) :
  distance = 8 ∧ 
  min_cyclist_time = 0.5 ∧ 
  cyclist_speed_increase = 0.25 ∧
  meet_time = 1/6 →
  ∃ (tourist_speed cyclist_speed : ℝ),
    tourist_speed = 7 ∧
    cyclist_speed = 16 ∧
    (∀ x : ℕ, x > tourist_speed → 
      ¬(∃ y : ℝ, 
        distance / y ≥ min_cyclist_time ∧
        x * (distance / y + meet_time) + y * meet_time * (1 + cyclist_speed_increase) = distance)) :=
by sorry

end NUMINAMATH_CALUDE_tourist_cyclist_speed_problem_l1479_147936


namespace NUMINAMATH_CALUDE_least_positive_angle_theorem_l1479_147919

/-- The least positive angle θ (in degrees) satisfying sin 15° = cos 40° + cos θ is 115° -/
theorem least_positive_angle_theorem : 
  let θ : ℝ := 115
  ∀ φ : ℝ, 0 < φ ∧ φ < θ → 
    Real.sin (15 * π / 180) ≠ Real.cos (40 * π / 180) + Real.cos (φ * π / 180) ∧
    Real.sin (15 * π / 180) = Real.cos (40 * π / 180) + Real.cos (θ * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_angle_theorem_l1479_147919


namespace NUMINAMATH_CALUDE_point_order_l1479_147915

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the line y = -7x + 14 -/
def lies_on_line (p : Point) : Prop :=
  p.y = -7 * p.x + 14

theorem point_order (A B C : Point) 
  (hA : lies_on_line A) 
  (hB : lies_on_line B) 
  (hC : lies_on_line C) 
  (hx : A.x > C.x ∧ C.x > B.x) : 
  A.y < C.y ∧ C.y < B.y := by
  sorry

end NUMINAMATH_CALUDE_point_order_l1479_147915


namespace NUMINAMATH_CALUDE_quadratic_root_in_interval_l1479_147956

theorem quadratic_root_in_interval (a b : ℝ) (hb : b > 0) 
  (h_distinct : ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁^2 + a*r₁ + b = 0 ∧ r₂^2 + a*r₂ + b = 0)
  (h_one_in_interval : ∃! r : ℝ, r^2 + a*r + b = 0 ∧ r ∈ Set.Icc (-1) 1) :
  ∃ r : ℝ, r^2 + a*r + b = 0 ∧ r ∈ Set.Ioo (-b) b :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_in_interval_l1479_147956


namespace NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l1479_147962

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 7

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x + y = 4

-- State the theorem
theorem shortest_distance_circle_to_line :
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 - Real.sqrt 7 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ → C₂ x₂ y₂ →
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_circle_to_line_l1479_147962


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l1479_147966

theorem rectangle_area_diagonal (length width diagonal : ℝ) (h1 : length / width = 5 / 2) 
  (h2 : length^2 + width^2 = diagonal^2) : 
  length * width = (10/29) * diagonal^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l1479_147966


namespace NUMINAMATH_CALUDE_store_profit_percentage_l1479_147983

/-- Proves that the profit percentage is 30% given the conditions of the problem -/
theorem store_profit_percentage (cost_price : ℝ) (sale_price : ℝ) :
  cost_price = 20 →
  sale_price = 13 →
  ∃ (selling_price : ℝ),
    selling_price = cost_price * (1 + 30 / 100) ∧
    sale_price = selling_price / 2 :=
by sorry

end NUMINAMATH_CALUDE_store_profit_percentage_l1479_147983


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1479_147903

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 :=
by
  -- The unique positive solution is 5/3
  use 5/3
  constructor
  · constructor
    · -- Prove 5/3 > 0
      sorry
    · -- Prove 3 * (5/3)^2 + 7 * (5/3) - 20 = 0
      sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1479_147903


namespace NUMINAMATH_CALUDE_literate_employees_count_l1479_147967

/-- The number of literate employees in an NGO --/
def number_of_literate_employees : ℕ :=
  let illiterate_employees : ℕ := 35
  let wage_decrease_per_illiterate : ℕ := 25
  let total_wage_decrease : ℕ := illiterate_employees * wage_decrease_per_illiterate
  let average_salary_decrease : ℕ := 15
  let total_employees : ℕ := (total_wage_decrease + average_salary_decrease - 1) / average_salary_decrease
  total_employees - illiterate_employees

/-- Theorem stating that the number of literate employees is 23 --/
theorem literate_employees_count :
  number_of_literate_employees = 23 := by
  sorry

end NUMINAMATH_CALUDE_literate_employees_count_l1479_147967


namespace NUMINAMATH_CALUDE_cube_root_of_four_fifth_powers_l1479_147904

theorem cube_root_of_four_fifth_powers (x : ℝ) :
  x = (5^6 + 5^6 + 5^6 + 5^6)^(1/3) → x = 25 * 2^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_four_fifth_powers_l1479_147904


namespace NUMINAMATH_CALUDE_birds_on_fence_l1479_147953

theorem birds_on_fence (initial_birds : ℕ) (storks_joined : ℕ) (stork_bird_difference : ℕ) :
  initial_birds = 3 →
  storks_joined = 6 →
  stork_bird_difference = 1 →
  ∃ (birds_joined : ℕ), birds_joined = 2 ∧
    storks_joined = initial_birds + birds_joined + stork_bird_difference :=
by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1479_147953


namespace NUMINAMATH_CALUDE_polygon_perimeter_l1479_147990

/-- The perimeter of a polygon formed by cutting a right triangle from a rectangle --/
theorem polygon_perimeter (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c < 10) : 
  2 * (10 + (10 - b)) - a = 29 :=
by sorry

end NUMINAMATH_CALUDE_polygon_perimeter_l1479_147990


namespace NUMINAMATH_CALUDE_smallest_nonzero_place_12000_l1479_147954

/-- The smallest place value with a non-zero digit in 12000 is the hundreds place -/
theorem smallest_nonzero_place_12000 : 
  ∀ n : ℕ, n > 0 ∧ n < 1000 → (12000 / 10^n) % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonzero_place_12000_l1479_147954


namespace NUMINAMATH_CALUDE_function_min_max_values_l1479_147924

/-- The function f(x) = x^3 - 3x + m has a minimum value of -1 and a maximum value of 3 -/
theorem function_min_max_values (m : ℝ) : 
  (∃ x₀ : ℝ, ∀ x : ℝ, x^3 - 3*x + m ≥ x₀^3 - 3*x₀ + m ∧ x₀^3 - 3*x₀ + m = -1) →
  (∃ x₁ : ℝ, ∀ x : ℝ, x^3 - 3*x + m ≤ x₁^3 - 3*x₁ + m ∧ x₁^3 - 3*x₁ + m = 3) :=
by sorry

end NUMINAMATH_CALUDE_function_min_max_values_l1479_147924


namespace NUMINAMATH_CALUDE_g_inverse_composition_l1479_147958

def g : Fin 5 → Fin 5
| 0 => 3  -- Representing g(1) = 4
| 1 => 2  -- Representing g(2) = 3
| 2 => 0  -- Representing g(3) = 1
| 3 => 4  -- Representing g(4) = 5
| 4 => 1  -- Representing g(5) = 2

theorem g_inverse_composition (h : Function.Bijective g) :
  (Function.invFun g) ((Function.invFun g) ((Function.invFun g) 2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_inverse_composition_l1479_147958


namespace NUMINAMATH_CALUDE_acute_triangle_angle_b_l1479_147938

theorem acute_triangle_angle_b (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- Acute triangle condition
  A + B + C = π →  -- Sum of angles in a triangle
  Real.sqrt 3 * a = 2 * b * Real.sin B * Real.cos C + 2 * b * Real.sin C * Real.cos B →
  B = π/3 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_angle_b_l1479_147938


namespace NUMINAMATH_CALUDE_imaginary_part_of_minus_one_plus_i_squared_l1479_147914

theorem imaginary_part_of_minus_one_plus_i_squared :
  Complex.im ((-1 + Complex.I) ^ 2) = -2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_minus_one_plus_i_squared_l1479_147914


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1479_147977

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1479_147977


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l1479_147965

theorem subset_implies_a_values (a : ℝ) : 
  let M : Set ℝ := {x | x^2 = 1}
  let N : Set ℝ := {x | a * x = 1}
  N ⊆ M → a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l1479_147965


namespace NUMINAMATH_CALUDE_all_propositions_false_l1479_147976

-- Define the type for lines in space
def Line : Type := ℝ → ℝ → ℝ → Prop

-- Define the relations between lines
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def coplanar (l1 l2 : Line) : Prop := sorry

-- Define the propositions
def proposition1 (a b c : Line) : Prop :=
  (perpendicular a b ∧ perpendicular b c) → parallel a c

def proposition2 (a b c : Line) : Prop :=
  (skew a b ∧ skew b c) → skew a c

def proposition3 (a b c : Line) : Prop :=
  (intersect a b ∧ intersect b c) → intersect a c

def proposition4 (a b c : Line) : Prop :=
  (coplanar a b ∧ coplanar b c) → coplanar a c

-- Theorem stating that all propositions are false
theorem all_propositions_false (a b c : Line) :
  ¬ proposition1 a b c ∧
  ¬ proposition2 a b c ∧
  ¬ proposition3 a b c ∧
  ¬ proposition4 a b c :=
sorry

end NUMINAMATH_CALUDE_all_propositions_false_l1479_147976


namespace NUMINAMATH_CALUDE_all_three_classes_l1479_147979

/-- Represents the number of students in each class combination --/
structure ClassCombinations where
  yoga : ℕ
  bridge : ℕ
  painting : ℕ
  yogaBridge : ℕ
  yogaPainting : ℕ
  bridgePainting : ℕ
  allThree : ℕ

/-- Represents the given conditions of the problem --/
def problem_conditions (c : ClassCombinations) : Prop :=
  c.yoga + c.bridge + c.painting + c.yogaBridge + c.yogaPainting + c.bridgePainting + c.allThree = 20 ∧
  c.yoga + c.yogaBridge + c.yogaPainting + c.allThree = 10 ∧
  c.bridge + c.yogaBridge + c.bridgePainting + c.allThree = 13 ∧
  c.painting + c.yogaPainting + c.bridgePainting + c.allThree = 9 ∧
  c.yogaBridge + c.yogaPainting + c.bridgePainting + c.allThree = 9

theorem all_three_classes (c : ClassCombinations) :
  problem_conditions c → c.allThree = 3 := by
  sorry

end NUMINAMATH_CALUDE_all_three_classes_l1479_147979


namespace NUMINAMATH_CALUDE_xiaohuo_has_448_books_l1479_147949

/-- The number of books Xiaohuo, Xiaoyan, and Xiaoyi have collectively -/
def total_books : ℕ := 1248

/-- The number of books Xiaohuo has -/
def xiaohuo_books : ℕ := sorry

/-- The number of books Xiaoyan has -/
def xiaoyan_books : ℕ := sorry

/-- The number of books Xiaoyi has -/
def xiaoyi_books : ℕ := sorry

/-- Xiaohuo has 64 more books than Xiaoyan -/
axiom xiaohuo_more_than_xiaoyan : xiaohuo_books = xiaoyan_books + 64

/-- Xiaoyan has 32 fewer books than Xiaoyi -/
axiom xiaoyan_fewer_than_xiaoyi : xiaoyan_books = xiaoyi_books - 32

/-- The total number of books is the sum of books owned by each person -/
axiom total_books_sum : total_books = xiaohuo_books + xiaoyan_books + xiaoyi_books

/-- Theorem: Xiaohuo has 448 books -/
theorem xiaohuo_has_448_books : xiaohuo_books = 448 := by sorry

end NUMINAMATH_CALUDE_xiaohuo_has_448_books_l1479_147949


namespace NUMINAMATH_CALUDE_subtract_negative_numbers_l1479_147957

theorem subtract_negative_numbers : -5 - 9 = -14 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_numbers_l1479_147957


namespace NUMINAMATH_CALUDE_bottles_ratio_l1479_147969

/-- The number of bottles Paul drinks per day -/
def paul_bottles : ℕ := 3

/-- The number of bottles Donald drinks per day -/
def donald_bottles : ℕ := 9

/-- Donald drinks more than twice the number of bottles Paul drinks -/
axiom donald_drinks_more : donald_bottles > 2 * paul_bottles

/-- The ratio of bottles Donald drinks to bottles Paul drinks is 3:1 -/
theorem bottles_ratio : (donald_bottles : ℚ) / paul_bottles = 3 := by
  sorry

end NUMINAMATH_CALUDE_bottles_ratio_l1479_147969


namespace NUMINAMATH_CALUDE_negation_equivalence_l1479_147997

open Real

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, (2 / x₀) + log x₀ ≤ 0) ↔ (∀ x : ℝ, (2 / x) + log x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1479_147997


namespace NUMINAMATH_CALUDE_count_positive_numbers_l1479_147947

theorem count_positive_numbers : 
  let numbers : List ℚ := [-3, -1, 1/3, 0, -3/7, 2017]
  (numbers.filter (λ x => x > 0)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_positive_numbers_l1479_147947


namespace NUMINAMATH_CALUDE_fred_total_games_l1479_147988

/-- The total number of basketball games Fred attended over two years -/
def total_games (games_this_year games_last_year : ℕ) : ℕ :=
  games_this_year + games_last_year

/-- Theorem stating that Fred attended 85 games in total -/
theorem fred_total_games : 
  total_games 60 25 = 85 := by
  sorry

end NUMINAMATH_CALUDE_fred_total_games_l1479_147988


namespace NUMINAMATH_CALUDE_wood_sawing_problem_l1479_147961

theorem wood_sawing_problem (original_length final_length : ℝ) 
  (h1 : original_length = 8.9)
  (h2 : final_length = 6.6) :
  original_length - final_length = 2.3 := by
  sorry

end NUMINAMATH_CALUDE_wood_sawing_problem_l1479_147961


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_eq_six_l1479_147910

/-- Represents a hyperbola with equation x²/m - y²/6 = 1 -/
structure Hyperbola (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / m - y^2 / 6 = 1

/-- Represents an asymptote of a hyperbola -/
structure Asymptote (m : ℝ) where
  slope : ℝ
  eq : ∀ (x y : ℝ), y = slope * x

/-- 
If a hyperbola with equation x²/m - y²/6 = 1 has an asymptote y = x,
then m = 6
-/
theorem hyperbola_asymptote_implies_m_eq_six (m : ℝ) 
  (h : Hyperbola m) 
  (a : Asymptote m) 
  (ha : a.slope = 1) : m = 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_eq_six_l1479_147910


namespace NUMINAMATH_CALUDE_tanker_filling_rate_l1479_147999

/-- Proves that the filling rate of 3 barrels per minute is equivalent to 28.62 m³/hour -/
theorem tanker_filling_rate 
  (barrel_rate : ℝ) 
  (liters_per_barrel : ℝ) 
  (h1 : barrel_rate = 3) 
  (h2 : liters_per_barrel = 159) : 
  (barrel_rate * liters_per_barrel * 60) / 1000 = 28.62 := by
  sorry

end NUMINAMATH_CALUDE_tanker_filling_rate_l1479_147999


namespace NUMINAMATH_CALUDE_problem_solution_l1479_147944

-- Define the quadratic equation
def quadratic_equation (a : ℝ) (t : ℝ) : Prop :=
  2 * a * t^2 + 12 * t + 9 = 0

-- Define parallel lines
def parallel_lines (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x + b * y = 1 ∧ 4 * x + 18 * y = 3

-- Define the b-th prime number
def nth_prime (b : ℕ) (p : ℕ) : Prop :=
  p.Prime ∧ (Finset.filter Nat.Prime (Finset.range p)).card = b

-- Define the trigonometric equation
def trig_equation (k θ : ℝ) : Prop :=
  k = (4 * Real.sin θ + 3 * Real.cos θ) / (2 * Real.sin θ - Real.cos θ) ∧
  Real.tan θ = 3

theorem problem_solution :
  (∃ a : ℝ, quadratic_equation a has_equal_roots) →
  (∃ b : ℝ, parallel_lines 2 b) →
  (∃ p : ℕ, nth_prime 9 p) →
  (∃ k θ : ℝ, trig_equation k θ) →
  ∃ (a b : ℝ) (p : ℕ) (k : ℝ),
    a = 2 ∧ b = 9 ∧ p = 23 ∧ k = 3 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1479_147944


namespace NUMINAMATH_CALUDE_card_area_theorem_l1479_147929

/-- Represents a rectangular card with length and width in inches -/
structure Card where
  length : ℝ
  width : ℝ

/-- Calculates the area of a card in square inches -/
def area (c : Card) : ℝ := c.length * c.width

/-- The original card -/
def original_card : Card := { length := 5, width := 7 }

/-- Theorem: If shortening one side of the original 5x7 card by 2 inches
    results in an area of 21 square inches, then shortening the other side
    by 1 inch results in an area of 30 square inches -/
theorem card_area_theorem :
  (∃ (c : Card), (c.length = original_card.length - 2 ∨ c.width = original_card.width - 2) ∧
                 area c = 21) →
  area { length := original_card.length,
         width := original_card.width - 1 } = 30 := by
  sorry

end NUMINAMATH_CALUDE_card_area_theorem_l1479_147929


namespace NUMINAMATH_CALUDE_ellipse_chord_slopes_product_l1479_147973

/-- Theorem: Product of slopes for chord through center of ellipse -/
theorem ellipse_chord_slopes_product (a b x₀ y₀ x₁ y₁ : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hP : x₀^2 / a^2 + y₀^2 / b^2 = 1)  -- P is on the ellipse
  (hP₁ : x₁^2 / a^2 + y₁^2 / b^2 = 1)  -- P₁ is on the ellipse
  (hP₂ : (-x₁)^2 / a^2 + (-y₁)^2 / b^2 = 1)  -- P₂ is on the ellipse
  (k₁ : ℝ) (hk₁ : k₁ = (y₀ - y₁) / (x₀ - x₁))  -- Slope of PP₁
  (k₂ : ℝ) (hk₂ : k₂ = (y₀ - (-y₁)) / (x₀ - (-x₁)))  -- Slope of PP₂
  : k₁ * k₂ = -b^2 / a^2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_slopes_product_l1479_147973


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1479_147991

/-- A geometric sequence with specific properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 1) →
  (a 3 + a 4 = 2) →
  (a 9 + a 10 = 16) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1479_147991


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l1479_147916

/-- Calculates the net rate of pay for a driver given specific conditions --/
theorem driver_net_pay_rate (hours : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (pay_per_mile : ℝ) (gas_price : ℝ) :
  hours = 3 →
  speed = 50 →
  fuel_efficiency = 25 →
  pay_per_mile = 0.60 →
  gas_price = 2.50 →
  let distance := hours * speed
  let gas_used := distance / fuel_efficiency
  let earnings := distance * pay_per_mile
  let gas_cost := gas_used * gas_price
  let net_earnings := earnings - gas_cost
  net_earnings / hours = 25 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l1479_147916


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_unchanged_l1479_147926

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : CartesianPoint := ⟨0, 0⟩

/-- The coordinates of a point with respect to the origin -/
def coordinatesWrtOrigin (p : CartesianPoint) : CartesianPoint := p

theorem coordinates_wrt_origin_unchanged (p : CartesianPoint) :
  coordinatesWrtOrigin p = p := by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_unchanged_l1479_147926


namespace NUMINAMATH_CALUDE_durand_more_likely_to_win_l1479_147902

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of ways to roll a sum of 7 with two dice -/
def ways_to_roll_7 : ℕ := 6

/-- The number of ways to roll a sum of 8 with two dice -/
def ways_to_roll_8 : ℕ := 5

/-- The probability of rolling a sum of 7 with two dice -/
def prob_7 : ℚ := ways_to_roll_7 / total_outcomes

/-- The probability of rolling a sum of 8 with two dice -/
def prob_8 : ℚ := ways_to_roll_8 / total_outcomes

theorem durand_more_likely_to_win : prob_7 > prob_8 := by
  sorry

end NUMINAMATH_CALUDE_durand_more_likely_to_win_l1479_147902


namespace NUMINAMATH_CALUDE_unique_solution_l1479_147964

/-- The equation from the problem -/
def equation (x y : ℝ) : Prop :=
  11 * x^2 + 2 * x * y + 9 * y^2 + 8 * x - 12 * y + 6 = 0

/-- There exists exactly one pair of real numbers (x, y) that satisfies the equation -/
theorem unique_solution : ∃! p : ℝ × ℝ, equation p.1 p.2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1479_147964


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l1479_147994

theorem least_number_divisible_by_five_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧ 
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧ 
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0) → 
    m ≥ n) ∧
  n = 2310 := by
  sorry

#check least_number_divisible_by_five_primes

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l1479_147994


namespace NUMINAMATH_CALUDE_leapYears105_l1479_147971

/-- Calculates the maximum number of leap years in a given period under a system where
    leap years occur every 4 years and every 5th year. -/
def maxLeapYears (period : ℕ) : ℕ :=
  (period / 4) + (period / 5) - (period / 20)

/-- Theorem stating that in a 105-year period, the maximum number of leap years is 42
    under the given leap year system. -/
theorem leapYears105 : maxLeapYears 105 = 42 := by
  sorry

#eval maxLeapYears 105  -- Should output 42

end NUMINAMATH_CALUDE_leapYears105_l1479_147971


namespace NUMINAMATH_CALUDE_artworks_per_quarter_is_two_l1479_147932

/-- The number of students in the art club -/
def num_students : ℕ := 15

/-- The number of quarters in a school year -/
def quarters_per_year : ℕ := 4

/-- The total number of artworks collected in two school years -/
def total_artworks : ℕ := 240

/-- The number of artworks each student makes by the end of each quarter -/
def artworks_per_student_per_quarter : ℕ := 2

/-- Theorem stating that the number of artworks each student makes by the end of each quarter is 2 -/
theorem artworks_per_quarter_is_two :
  artworks_per_student_per_quarter * num_students * quarters_per_year * 2 = total_artworks :=
by sorry

end NUMINAMATH_CALUDE_artworks_per_quarter_is_two_l1479_147932


namespace NUMINAMATH_CALUDE_battery_life_comparison_l1479_147918

-- Define the battery characteristics
def tablet_standby : ℚ := 18
def tablet_continuous : ℚ := 6
def smartphone_standby : ℚ := 30
def smartphone_continuous : ℚ := 4

-- Define the usage
def tablet_total_time : ℚ := 14
def tablet_usage_time : ℚ := 2
def smartphone_total_time : ℚ := 20
def smartphone_usage_time : ℚ := 3

-- Define the battery consumption rates
def tablet_standby_rate : ℚ := 1 / tablet_standby
def tablet_usage_rate : ℚ := 1 / tablet_continuous
def smartphone_standby_rate : ℚ := 1 / smartphone_standby
def smartphone_usage_rate : ℚ := 1 / smartphone_continuous

-- Define the theorem
theorem battery_life_comparison : 
  let tablet_battery_used := (tablet_total_time - tablet_usage_time) * tablet_standby_rate + tablet_usage_time * tablet_usage_rate
  let smartphone_battery_used := (smartphone_total_time - smartphone_usage_time) * smartphone_standby_rate + smartphone_usage_time * smartphone_usage_rate
  let smartphone_battery_remaining := 1 - smartphone_battery_used
  tablet_battery_used ≥ 1 ∧ 
  smartphone_battery_remaining / smartphone_standby_rate = 9 :=
by sorry

end NUMINAMATH_CALUDE_battery_life_comparison_l1479_147918


namespace NUMINAMATH_CALUDE_toms_journey_ratio_l1479_147959

/-- Proves that the ratio of running time to swimming time is 1:2 given the conditions of Tom's journey --/
theorem toms_journey_ratio (swim_speed swim_time run_speed total_distance : ℝ) : 
  swim_speed = 2 →
  swim_time = 2 →
  run_speed = 4 * swim_speed →
  total_distance = 12 →
  total_distance = swim_speed * swim_time + run_speed * (total_distance - swim_speed * swim_time) / run_speed →
  (total_distance - swim_speed * swim_time) / run_speed / swim_time = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_toms_journey_ratio_l1479_147959


namespace NUMINAMATH_CALUDE_unique_a_value_l1479_147909

/-- Converts a number from base 53 to base 10 -/
def base53ToBase10 (n : ℕ) : ℕ := sorry

/-- Theorem: If a is an integer between 0 and 20 (inclusive) and 4254253₅₃ - a is a multiple of 17, then a = 3 -/
theorem unique_a_value (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 20) 
  (h3 : (base53ToBase10 4254253 - a) % 17 = 0) : a = 3 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l1479_147909


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1479_147992

theorem simplify_complex_fraction : 
  (1 / ((3 / (Real.sqrt 5 + 2)) - (1 / (Real.sqrt 4 + 1)))) = ((27 * Real.sqrt 5 + 57) / 40) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1479_147992


namespace NUMINAMATH_CALUDE_prime_triple_divisibility_l1479_147907

theorem prime_triple_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  p ∣ (q + r) ∧ q ∣ (r + 2*p) ∧ r ∣ (p + 3*q) →
  ((p = 5 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = 11 ∧ r = 7) ∨ 
   (p = 2 ∧ q = 3 ∧ r = 11)) :=
by sorry

#check prime_triple_divisibility

end NUMINAMATH_CALUDE_prime_triple_divisibility_l1479_147907


namespace NUMINAMATH_CALUDE_units_digit_of_square_l1479_147998

/-- 
Given an integer n, if the tens digit of n^2 is 7, 
then the units digit of n^2 is 6.
-/
theorem units_digit_of_square (n : ℤ) : 
  (n^2 % 100 / 10 = 7) → (n^2 % 10 = 6) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_square_l1479_147998


namespace NUMINAMATH_CALUDE_three_A_plus_six_B_m_value_when_independent_l1479_147920

-- Define A and B as functions of x and m
def A (x m : ℝ) : ℝ := 2*x^2 + 3*m*x - 2*x - 1
def B (x m : ℝ) : ℝ := -x^2 + m*x - 1

-- Theorem 1: 3A + 6B = (15m-6)x - 9
theorem three_A_plus_six_B (x m : ℝ) : 
  3 * A x m + 6 * B x m = (15*m - 6)*x - 9 := by sorry

-- Theorem 2: When 3A + 6B is independent of x, m = 2/5
theorem m_value_when_independent (m : ℝ) :
  (∀ x : ℝ, 3 * A x m + 6 * B x m = (15*m - 6)*x - 9) →
  (∀ x y : ℝ, 3 * A x m + 6 * B x m = 3 * A y m + 6 * B y m) →
  m = 2/5 := by sorry

end NUMINAMATH_CALUDE_three_A_plus_six_B_m_value_when_independent_l1479_147920


namespace NUMINAMATH_CALUDE_extreme_values_l1479_147922

/-- A quadratic function passing through four points with specific properties. -/
structure QuadraticFunction where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  y₄ : ℝ
  h₁ : y₂ < y₃
  h₂ : y₃ = y₄

/-- Theorem stating that y₁ is the smallest and y₃ is the largest among y₁, y₂, and y₃ -/
theorem extreme_values (f : QuadraticFunction) : 
  f.y₁ ≤ f.y₂ ∧ f.y₁ ≤ f.y₃ ∧ f.y₂ < f.y₃ := by
  sorry

#check extreme_values

end NUMINAMATH_CALUDE_extreme_values_l1479_147922


namespace NUMINAMATH_CALUDE_remaining_watch_time_l1479_147974

/-- Represents a time duration in hours and minutes -/
structure Duration where
  hours : ℕ
  minutes : ℕ

/-- Converts a Duration to minutes -/
def Duration.toMinutes (d : Duration) : ℕ :=
  d.hours * 60 + d.minutes

/-- The total duration of the series -/
def seriesDuration : Duration := { hours := 6, minutes := 0 }

/-- The durations of Hannah's watching periods -/
def watchingPeriods : List Duration := [
  { hours := 2, minutes := 24 },
  { hours := 1, minutes := 25 },
  { hours := 0, minutes := 55 }
]

/-- Theorem stating the remaining time to watch the series -/
theorem remaining_watch_time :
  seriesDuration.toMinutes - (watchingPeriods.map Duration.toMinutes).sum = 76 := by
  sorry

end NUMINAMATH_CALUDE_remaining_watch_time_l1479_147974


namespace NUMINAMATH_CALUDE_kelly_games_theorem_l1479_147934

/-- The number of games Kelly needs to give away to reach her desired number of games -/
def games_to_give_away (initial_games desired_games : ℕ) : ℕ :=
  initial_games - desired_games

theorem kelly_games_theorem (initial_games desired_games : ℕ) 
  (h1 : initial_games = 120) (h2 : desired_games = 20) : 
  games_to_give_away initial_games desired_games = 100 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_theorem_l1479_147934


namespace NUMINAMATH_CALUDE_chris_age_l1479_147941

theorem chris_age (c m : ℕ) : c = 3 * m - 22 → c + m = 70 → c = 47 := by
  sorry

end NUMINAMATH_CALUDE_chris_age_l1479_147941


namespace NUMINAMATH_CALUDE_museum_pictures_l1479_147952

theorem museum_pictures (zoo_pics : ℕ) (deleted_pics : ℕ) (remaining_pics : ℕ) (museum_pics : ℕ) : 
  zoo_pics = 15 → 
  deleted_pics = 31 → 
  remaining_pics = 2 → 
  zoo_pics + museum_pics - deleted_pics = remaining_pics → 
  museum_pics = 18 := by
sorry

end NUMINAMATH_CALUDE_museum_pictures_l1479_147952


namespace NUMINAMATH_CALUDE_cube_symmetry_properties_change_l1479_147921

/-- Represents the symmetrical properties of a geometric object -/
structure SymmetryProperties where
  planes : ℕ
  axes : ℕ
  center : Bool

/-- Represents the different painting configurations of a cube -/
inductive CubePainting
  | Unpainted
  | OneFace
  | TwoFacesParallel
  | TwoFacesAdjacent
  | ThreeFacesMeetingAtVertex
  | ThreeFacesNotMeetingAtVertex

/-- Returns the symmetry properties for a given cube painting configuration -/
def symmetryPropertiesForCube (painting : CubePainting) : SymmetryProperties :=
  match painting with
  | .Unpainted => { planes := 9, axes := 9, center := true }
  | .OneFace => { planes := 4, axes := 1, center := false }
  | .TwoFacesParallel => { planes := 5, axes := 3, center := true }
  | .TwoFacesAdjacent => { planes := 2, axes := 1, center := false }
  | .ThreeFacesMeetingAtVertex => { planes := 3, axes := 0, center := false }
  | .ThreeFacesNotMeetingAtVertex => { planes := 2, axes := 1, center := false }

theorem cube_symmetry_properties_change (painting : CubePainting) :
  symmetryPropertiesForCube painting ≠ symmetryPropertiesForCube CubePainting.Unpainted :=
by sorry

end NUMINAMATH_CALUDE_cube_symmetry_properties_change_l1479_147921


namespace NUMINAMATH_CALUDE_prism_volume_with_inscribed_sphere_l1479_147995

/-- The volume of a regular triangular prism with an inscribed sphere -/
theorem prism_volume_with_inscribed_sphere (r : ℝ) (h : r > 0) :
  let sphere_volume : ℝ := (4 / 3) * Real.pi * r^3
  let prism_side : ℝ := 2 * Real.sqrt 3 * r
  let prism_height : ℝ := 2 * r
  let prism_volume : ℝ := (Real.sqrt 3 / 4) * prism_side^2 * prism_height
  sphere_volume = 36 * Real.pi → prism_volume = 162 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_with_inscribed_sphere_l1479_147995


namespace NUMINAMATH_CALUDE_melanie_book_count_l1479_147948

/-- The total number of books Melanie has after buying more books is equal to the sum of her initial book count and the number of books she bought. -/
theorem melanie_book_count (initial_books new_books : ℝ) :
  let total_books := initial_books + new_books
  total_books = initial_books + new_books :=
by sorry

end NUMINAMATH_CALUDE_melanie_book_count_l1479_147948


namespace NUMINAMATH_CALUDE_base7_multiplication_l1479_147935

/-- Converts a number from base 7 to base 10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Represents a number in base 7 --/
structure Base7 where
  value : ℕ

theorem base7_multiplication :
  let a : Base7 := ⟨345⟩
  let b : Base7 := ⟨3⟩
  let result : Base7 := ⟨1401⟩
  toBase7 (toBase10 a.value * toBase10 b.value) = result.value := by sorry

end NUMINAMATH_CALUDE_base7_multiplication_l1479_147935


namespace NUMINAMATH_CALUDE_staircase_perimeter_l1479_147978

/-- A staircase-shaped region with specific properties -/
structure StaircaseRegion where
  congruentSides : ℕ
  sideLength : ℝ
  area : ℝ

/-- Calculate the perimeter of the staircase region -/
def perimeter (s : StaircaseRegion) : ℝ :=
  7 + 11 + 3 + 7 + s.congruentSides * s.sideLength

/-- Theorem: The perimeter of the specific staircase region is 39 feet -/
theorem staircase_perimeter :
  ∀ s : StaircaseRegion,
    s.congruentSides = 10 ∧
    s.sideLength = 1 ∧
    s.area = 74 →
    perimeter s = 39 := by
  sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l1479_147978


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1479_147917

theorem quadratic_always_positive (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1479_147917


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l1479_147996

theorem parakeets_per_cage (num_cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ) 
  (h1 : num_cages = 6)
  (h2 : parrots_per_cage = 2)
  (h3 : total_birds = 54) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 7 := by
  sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l1479_147996


namespace NUMINAMATH_CALUDE_roy_pens_total_l1479_147930

theorem roy_pens_total (blue : ℕ) (black : ℕ) (red : ℕ) : 
  blue = 2 → 
  black = 2 * blue → 
  red = 2 * black - 2 → 
  blue + black + red = 12 := by
  sorry

end NUMINAMATH_CALUDE_roy_pens_total_l1479_147930


namespace NUMINAMATH_CALUDE_strawberry_picking_problem_l1479_147912

/-- The strawberry picking problem -/
theorem strawberry_picking_problem 
  (betty_strawberries : ℕ)
  (matthew_strawberries : ℕ)
  (natalie_strawberries : ℕ)
  (strawberries_per_jar : ℕ)
  (price_per_jar : ℕ)
  (total_money_made : ℕ)
  (h1 : betty_strawberries = 16)
  (h2 : matthew_strawberries > betty_strawberries)
  (h3 : matthew_strawberries = 2 * natalie_strawberries)
  (h4 : strawberries_per_jar = 7)
  (h5 : price_per_jar = 4)
  (h6 : total_money_made = 40)
  (h7 : betty_strawberries + matthew_strawberries + natalie_strawberries = 
        (total_money_made / price_per_jar) * strawberries_per_jar) :
  matthew_strawberries - betty_strawberries = 20 := by
sorry

end NUMINAMATH_CALUDE_strawberry_picking_problem_l1479_147912


namespace NUMINAMATH_CALUDE_point_order_on_line_l1479_147928

/-- Given points on a line, prove their y-coordinates are ordered. -/
theorem point_order_on_line (b : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 3 * (-3) - b)
  (h₂ : y₂ = 3 * 1 - b)
  (h₃ : y₃ = 3 * (-1) - b) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end NUMINAMATH_CALUDE_point_order_on_line_l1479_147928


namespace NUMINAMATH_CALUDE_remaining_work_time_for_a_l1479_147985

/-- The problem of calculating the remaining work time for person a -/
theorem remaining_work_time_for_a (a b c : ℝ) (h1 : a = 1 / 9) (h2 : b = 1 / 15) (h3 : c = 1 / 20) : 
  (1 - (10 * b + 5 * c)) / a = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_work_time_for_a_l1479_147985


namespace NUMINAMATH_CALUDE_no_right_obtuse_triangle_l1479_147908

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of triangles
def Triangle.isValid (t : Triangle) : Prop :=
  t.angle1 > 0 ∧ t.angle2 > 0 ∧ t.angle3 > 0 ∧ t.angle1 + t.angle2 + t.angle3 = 180

def Triangle.hasRightAngle (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

def Triangle.hasObtuseAngle (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: A valid triangle cannot be both right-angled and obtuse
theorem no_right_obtuse_triangle :
  ∀ t : Triangle, t.isValid → ¬(t.hasRightAngle ∧ t.hasObtuseAngle) :=
by
  sorry


end NUMINAMATH_CALUDE_no_right_obtuse_triangle_l1479_147908


namespace NUMINAMATH_CALUDE_starting_number_of_range_l1479_147963

theorem starting_number_of_range (n : ℕ) (h1 : n ≤ 31) (h2 : n % 3 = 0) 
  (h3 : ∀ k, n - 18 ≤ k ∧ k ≤ n → k % 3 = 0) : n - 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_starting_number_of_range_l1479_147963


namespace NUMINAMATH_CALUDE_expression_value_l1479_147960

theorem expression_value (a b c d x y : ℤ) :
  (a + b = 0) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (abs x = 3) →  -- absolute value of x is 3
  (y = -1) →     -- y is the largest negative integer
  (2*x - c*d + 6*(a + b) - abs y = 4) ∨ (2*x - c*d + 6*(a + b) - abs y = -8) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l1479_147960


namespace NUMINAMATH_CALUDE_equation_solutions_count_l1479_147939

theorem equation_solutions_count :
  let f : ℝ → ℝ := λ x => 2 * Real.sqrt 2 * (Real.sin (π * x / 4))^3 - Real.cos (π * (1 - x) / 4)
  ∃! (solutions : Finset ℝ),
    (∀ x ∈ solutions, f x = 0 ∧ 0 ≤ x ∧ x ≤ 2020) ∧
    (∀ x, f x = 0 ∧ 0 ≤ x ∧ x ≤ 2020 → x ∈ solutions) ∧
    Finset.card solutions = 505 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l1479_147939


namespace NUMINAMATH_CALUDE_min_cooking_time_is_15_l1479_147906

/-- Represents the time required for each step in the noodle cooking process -/
structure CookingTimes where
  washPot : ℕ
  washVegetables : ℕ
  prepareIngredients : ℕ
  boilWater : ℕ
  cookNoodles : ℕ

/-- Calculates the minimum time to cook noodles given the cooking times -/
def minCookingTime (times : CookingTimes) : ℕ :=
  let simultaneousTime := max times.washVegetables times.prepareIngredients
  times.washPot + simultaneousTime + times.cookNoodles

/-- Theorem stating that the minimum cooking time is 15 minutes -/
theorem min_cooking_time_is_15 (times : CookingTimes) 
  (h1 : times.washPot = 2)
  (h2 : times.washVegetables = 6)
  (h3 : times.prepareIngredients = 2)
  (h4 : times.boilWater = 10)
  (h5 : times.cookNoodles = 3) :
  minCookingTime times = 15 := by
  sorry

#eval minCookingTime ⟨2, 6, 2, 10, 3⟩

end NUMINAMATH_CALUDE_min_cooking_time_is_15_l1479_147906
