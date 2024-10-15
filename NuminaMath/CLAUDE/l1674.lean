import Mathlib

namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1674_167484

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ) (horse_food_per_day total_food : ℝ),
    sheep * 7 = horses * 6 →
    horse_food_per_day = 230 →
    total_food = 12880 →
    horses * horse_food_per_day = total_food →
    sheep = 48 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l1674_167484


namespace NUMINAMATH_CALUDE_number_puzzle_l1674_167407

theorem number_puzzle (x : ℝ) : 3 * (2 * x + 9) = 51 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1674_167407


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1674_167430

/-- The set of available numbers -/
def S : Finset Int := {-9, -6, -4, -1, 3, 5, 7, 12}

/-- The expression to be minimized -/
def f (p q r s t u v w : Int) : ℚ :=
  ((p + q + r + s : ℚ) ^ 2 + (t + u + v + w : ℚ) ^ 2 : ℚ)

/-- The theorem stating the minimum value of the expression -/
theorem min_value_of_expression :
  ∀ p q r s t u v w : Int,
    p ∈ S → q ∈ S → r ∈ S → s ∈ S → t ∈ S → u ∈ S → v ∈ S → w ∈ S →
    p ≠ q → p ≠ r → p ≠ s → p ≠ t → p ≠ u → p ≠ v → p ≠ w →
    q ≠ r → q ≠ s → q ≠ t → q ≠ u → q ≠ v → q ≠ w →
    r ≠ s → r ≠ t → r ≠ u → r ≠ v → r ≠ w →
    s ≠ t → s ≠ u → s ≠ v → s ≠ w →
    t ≠ u → t ≠ v → t ≠ w →
    u ≠ v → u ≠ w →
    v ≠ w →
    f p q r s t u v w ≥ 26.5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1674_167430


namespace NUMINAMATH_CALUDE_age_difference_proof_l1674_167467

theorem age_difference_proof (people : Fin 5 → ℕ) 
  (h1 : people 0 = people 1 + 1)
  (h2 : people 2 = people 3 + 2)
  (h3 : people 4 = people 5 + 3)
  (h4 : people 6 = people 7 + 4) :
  people 9 = people 8 + 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1674_167467


namespace NUMINAMATH_CALUDE_max_k_value_l1674_167408

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 13) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l1674_167408


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1674_167455

/-- Given a geometric sequence {a_n} with common ratio q = 4 and S_3 = 21,
    prove that the general term formula is a_n = 4^(n-1) -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) -- The sequence
  (q : ℝ) -- Common ratio
  (S₃ : ℝ) -- Sum of first 3 terms
  (h1 : ∀ n, a (n + 1) = q * a n) -- Definition of geometric sequence
  (h2 : q = 4) -- Given common ratio
  (h3 : S₃ = 21) -- Given sum of first 3 terms
  (h4 : S₃ = a 1 + a 2 + a 3) -- Definition of S₃
  : ∀ n : ℕ, a n = 4^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1674_167455


namespace NUMINAMATH_CALUDE_book_survey_difference_l1674_167422

/-- Represents the survey results of students reading books A and B -/
structure BookSurvey where
  total : ℕ
  only_a : ℕ
  only_b : ℕ
  both : ℕ
  h_total : total = only_a + only_b + both
  h_a_both : both = (only_a + both) / 5
  h_b_both : both = (only_b + both) / 4

/-- The difference between students who read only book A and only book B is 75 -/
theorem book_survey_difference (s : BookSurvey) (h_total : s.total = 600) :
  s.only_a - s.only_b = 75 := by
  sorry

end NUMINAMATH_CALUDE_book_survey_difference_l1674_167422


namespace NUMINAMATH_CALUDE_last_three_average_l1674_167443

theorem last_three_average (list : List ℝ) : 
  list.length = 6 →
  list.sum / list.length = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 3 = 65 := by
sorry

end NUMINAMATH_CALUDE_last_three_average_l1674_167443


namespace NUMINAMATH_CALUDE_quadratic_point_range_l1674_167411

/-- Given a quadratic function y = ax² + 4ax + c with a ≠ 0, and points A, B, C on its graph,
    prove that m < -3 under certain conditions. -/
theorem quadratic_point_range (a c m y₁ y₂ x₀ y₀ : ℝ) : 
  a ≠ 0 →
  y₁ = a * m^2 + 4 * a * m + c →
  y₂ = a * (m + 2)^2 + 4 * a * (m + 2) + c →
  y₀ = a * x₀^2 + 4 * a * x₀ + c →
  x₀ = -2 →
  y₀ ≥ y₂ →
  y₂ > y₁ →
  m < -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_point_range_l1674_167411


namespace NUMINAMATH_CALUDE_ellipse_range_l1674_167426

theorem ellipse_range (x y : ℝ) (h : x^2/4 + y^2 = 1) :
  ∃ (z : ℝ), z = 2*x + y ∧ -Real.sqrt 17 ≤ z ∧ z ≤ Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_range_l1674_167426


namespace NUMINAMATH_CALUDE_common_divisor_nineteen_l1674_167462

theorem common_divisor_nineteen (a : ℤ) : Int.gcd (35 * a + 57) (45 * a + 76) = 19 := by
  sorry

end NUMINAMATH_CALUDE_common_divisor_nineteen_l1674_167462


namespace NUMINAMATH_CALUDE_team_a_wins_l1674_167404

/-- Represents the outcome of a match for a team -/
inductive MatchResult
  | Win
  | Draw
  | Loss

/-- Calculates points for a given match result -/
def pointsForResult (result : MatchResult) : Nat :=
  match result with
  | MatchResult.Win => 3
  | MatchResult.Draw => 1
  | MatchResult.Loss => 0

/-- Represents the results of a series of matches for a team -/
structure TeamResults where
  wins : Nat
  draws : Nat
  losses : Nat

/-- Calculates total points for a team's results -/
def totalPoints (results : TeamResults) : Nat :=
  results.wins * (pointsForResult MatchResult.Win) +
  results.draws * (pointsForResult MatchResult.Draw) +
  results.losses * (pointsForResult MatchResult.Loss)

theorem team_a_wins (total_matches : Nat) (team_a_points : Nat)
    (h1 : total_matches = 10)
    (h2 : team_a_points = 22)
    (h3 : ∀ (r : TeamResults), 
      r.wins + r.draws = total_matches → 
      r.losses = 0 → 
      totalPoints r = team_a_points → 
      r.wins = 6) :
  ∃ (r : TeamResults), r.wins + r.draws = total_matches ∧ 
                       r.losses = 0 ∧ 
                       totalPoints r = team_a_points ∧ 
                       r.wins = 6 := by
  sorry

#check team_a_wins

end NUMINAMATH_CALUDE_team_a_wins_l1674_167404


namespace NUMINAMATH_CALUDE_trouser_original_price_l1674_167418

theorem trouser_original_price (sale_price : ℝ) (discount_percent : ℝ) : 
  sale_price = 55 → discount_percent = 45 → 
  ∃ (original_price : ℝ), original_price = 100 ∧ sale_price = original_price * (1 - discount_percent / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_trouser_original_price_l1674_167418


namespace NUMINAMATH_CALUDE_arccos_equation_solution_l1674_167470

theorem arccos_equation_solution :
  ∀ x : ℝ, (Real.arccos (3 * x) - Real.arccos x = π / 6) ↔ (x = 1/12 ∨ x = -1/12) :=
by sorry

end NUMINAMATH_CALUDE_arccos_equation_solution_l1674_167470


namespace NUMINAMATH_CALUDE_sequence_formulas_l1674_167453

/-- Given a sequence {a_n} with sum of first n terms S_n satisfying S_n = 2 - a_n,
    and sequence {b_n} satisfying b_1 = 1 and b_{n+1} = b_n + a_n,
    prove the general term formulas for both sequences. -/
theorem sequence_formulas (a b : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n = 2 - a n) →
  b 1 = 1 →
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = b n + a n) →
  (∀ n : ℕ, n ≥ 1 → a n = (1/2)^(n-1)) ∧
  (∀ n : ℕ, n ≥ 2 → b n = 3 - 1/(2^(n-2))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formulas_l1674_167453


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1674_167486

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = (Real.sqrt (x^12 + 7 * x^6 + 1)) / (3 * x^3) :=
sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1674_167486


namespace NUMINAMATH_CALUDE_red_ball_probability_l1674_167464

theorem red_ball_probability (x : ℕ) : 
  (8 : ℝ) / (x + 8 : ℝ) = 0.2 → x = 32 := by
sorry

end NUMINAMATH_CALUDE_red_ball_probability_l1674_167464


namespace NUMINAMATH_CALUDE_investment_rate_problem_l1674_167417

/-- Proves that given the conditions of the investment problem, the rate of the first investment is 10% -/
theorem investment_rate_problem (total_investment : ℝ) (second_investment : ℝ) (second_rate : ℝ) (income_difference : ℝ) :
  total_investment = 2000 →
  second_investment = 750 →
  second_rate = 0.08 →
  income_difference = 65 →
  let first_investment := total_investment - second_investment
  let first_rate := (income_difference + second_investment * second_rate) / first_investment
  first_rate = 0.1 := by
  sorry

#check investment_rate_problem

end NUMINAMATH_CALUDE_investment_rate_problem_l1674_167417


namespace NUMINAMATH_CALUDE_shift_arrangements_count_l1674_167465

def total_volunteers : ℕ := 14
def shifts_per_day : ℕ := 3
def people_per_shift : ℕ := 4

def shift_arrangements : ℕ := (total_volunteers.choose people_per_shift) * 
                               ((total_volunteers - people_per_shift).choose people_per_shift) * 
                               ((total_volunteers - 2 * people_per_shift).choose people_per_shift)

theorem shift_arrangements_count : shift_arrangements = 3153150 := by
  sorry

end NUMINAMATH_CALUDE_shift_arrangements_count_l1674_167465


namespace NUMINAMATH_CALUDE_well_depth_proof_l1674_167423

/-- The depth of the well in feet -/
def depth : ℝ := 918.09

/-- The total time from dropping the stone to hearing it hit the bottom, in seconds -/
def total_time : ℝ := 8.5

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1100

/-- The function describing the distance fallen by the stone after t seconds -/
def stone_fall (t : ℝ) : ℝ := 16 * t^2

theorem well_depth_proof :
  ∃ (t_fall : ℝ), 
    t_fall > 0 ∧
    stone_fall t_fall = depth ∧
    t_fall + depth / sound_speed = total_time :=
sorry

end NUMINAMATH_CALUDE_well_depth_proof_l1674_167423


namespace NUMINAMATH_CALUDE_trapezoid_area_l1674_167425

/-- A trapezoid with given dimensions -/
structure Trapezoid where
  AD : ℝ  -- Length of longer base
  BC : ℝ  -- Length of shorter base
  AC : ℝ  -- Length of one diagonal
  BD : ℝ  -- Length of other diagonal

/-- The area of a trapezoid with the given dimensions is 80 -/
theorem trapezoid_area (T : Trapezoid)
    (h1 : T.AD = 24)
    (h2 : T.BC = 8)
    (h3 : T.AC = 13)
    (h4 : T.BD = 5 * Real.sqrt 17) :
    (T.AD + T.BC) * Real.sqrt (T.AC ^ 2 - ((T.AD - T.BC) / 2 + T.BC) ^ 2) / 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1674_167425


namespace NUMINAMATH_CALUDE_sum_of_coordinates_P_l1674_167476

/-- Given three points P, Q, and R in a plane such that PR/PQ = RQ/PQ = 1/2,
    Q = (2, 5), and R = (0, -10), prove that the sum of coordinates of P is -27. -/
theorem sum_of_coordinates_P (P Q R : ℝ × ℝ) : 
  (dist P R / dist P Q = 1/2) →
  (dist R Q / dist P Q = 1/2) →
  Q = (2, 5) →
  R = (0, -10) →
  P.1 + P.2 = -27 := by
  sorry

#check sum_of_coordinates_P

end NUMINAMATH_CALUDE_sum_of_coordinates_P_l1674_167476


namespace NUMINAMATH_CALUDE_inequality_and_system_solution_l1674_167416

theorem inequality_and_system_solution :
  (∀ x : ℝ, (2*x - 3)/3 > (3*x + 1)/6 - 1 ↔ x > 1) ∧
  (∀ x : ℝ, x ≤ 3*x - 6 ∧ 3*x + 1 > 2*(x - 1) ↔ x ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_system_solution_l1674_167416


namespace NUMINAMATH_CALUDE_scavenger_hunt_items_l1674_167449

theorem scavenger_hunt_items (tanya samantha lewis james : ℕ) : 
  tanya = 4 ∧ 
  samantha = 4 * tanya ∧ 
  lewis = samantha + 4 ∧ 
  james = 2 * lewis →
  lewis = 20 := by
sorry

end NUMINAMATH_CALUDE_scavenger_hunt_items_l1674_167449


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l1674_167474

theorem quadratic_solution_property (k : ℝ) : 
  (∃ a b : ℝ, 6 * a^2 + 5 * a + k = 0 ∧ 
              6 * b^2 + 5 * b + k = 0 ∧ 
              a ≠ b ∧
              |a - b| = 3 * (a^2 + b^2)) ↔ 
  (k = 1 ∨ k = -17900 / 864) :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l1674_167474


namespace NUMINAMATH_CALUDE_distance_set_exists_l1674_167437

/-- A set of points in the plane satisfying the distance condition -/
def DistanceSet (m : ℕ) (S : Set (ℝ × ℝ)) : Prop :=
  (∀ A ∈ S, ∃! (points : Finset (ℝ × ℝ)), 
    points.card = m ∧ 
    (∀ B ∈ points, B ∈ S ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1))

/-- The existence of a finite set satisfying the distance condition for any m ≥ 1 -/
theorem distance_set_exists (m : ℕ) (hm : m ≥ 1) : 
  ∃ S : Set (ℝ × ℝ), S.Finite ∧ DistanceSet m S :=
sorry

end NUMINAMATH_CALUDE_distance_set_exists_l1674_167437


namespace NUMINAMATH_CALUDE_smallest_m_has_n_14_l1674_167409

def is_valid_m (m : ℕ) : Prop :=
  ∃ (n : ℕ) (r : ℝ), 
    n > 0 ∧ 
    r > 0 ∧ 
    r < 1/10000 ∧ 
    m^(1/4 : ℝ) = n + r

theorem smallest_m_has_n_14 : 
  ∃ (m : ℕ), is_valid_m m ∧ 
  (∀ (k : ℕ), k < m → ¬is_valid_m k) ∧
  (∃ (r : ℝ), m^(1/4 : ℝ) = 14 + r ∧ r > 0 ∧ r < 1/10000) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_has_n_14_l1674_167409


namespace NUMINAMATH_CALUDE_problems_per_page_problems_per_page_is_four_l1674_167424

theorem problems_per_page : ℕ → Prop :=
  fun p =>
    let math_pages : ℕ := 4
    let reading_pages : ℕ := 6
    let total_pages : ℕ := math_pages + reading_pages
    let total_problems : ℕ := 40
    total_pages * p = total_problems → p = 4

-- The proof is omitted
theorem problems_per_page_is_four : problems_per_page 4 := by sorry

end NUMINAMATH_CALUDE_problems_per_page_problems_per_page_is_four_l1674_167424


namespace NUMINAMATH_CALUDE_new_year_markup_l1674_167473

theorem new_year_markup (initial_markup : ℝ) (february_discount : ℝ) (final_profit : ℝ) :
  initial_markup = 0.20 →
  february_discount = 0.12 →
  final_profit = 0.32 →
  ∃ (new_year_markup : ℝ),
    (1 + initial_markup) * (1 + new_year_markup) * (1 - february_discount) = 1 + final_profit ∧
    new_year_markup = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_new_year_markup_l1674_167473


namespace NUMINAMATH_CALUDE_rohan_salary_l1674_167442

def monthly_salary (food_percent : ℚ) (rent_percent : ℚ) (entertainment_percent : ℚ) (conveyance_percent : ℚ) (savings : ℕ) : ℕ :=
  sorry

theorem rohan_salary :
  let food_percent : ℚ := 40 / 100
  let rent_percent : ℚ := 20 / 100
  let entertainment_percent : ℚ := 10 / 100
  let conveyance_percent : ℚ := 10 / 100
  let savings : ℕ := 1000
  monthly_salary food_percent rent_percent entertainment_percent conveyance_percent savings = 5000 := by
  sorry

end NUMINAMATH_CALUDE_rohan_salary_l1674_167442


namespace NUMINAMATH_CALUDE_min_radius_of_circle_l1674_167405

theorem min_radius_of_circle (r a b : ℝ) : 
  ((a - (r + 1))^2 + b^2 = r^2) →  -- Point (a, b) is on the circle
  (b^2 ≥ 4*a) →                    -- Condition b^2 ≥ 4a
  (r ≥ 0) →                        -- Radius is non-negative
  (r ≥ 4) :=                       -- Minimum value of r is 4
by sorry

end NUMINAMATH_CALUDE_min_radius_of_circle_l1674_167405


namespace NUMINAMATH_CALUDE_yuna_has_most_points_l1674_167440

-- Define the point totals for each person
def yoongi_points : ℕ := 7
def jungkook_points : ℕ := 6
def yuna_points : ℕ := 9
def yoojung_points : ℕ := 8

-- Theorem stating that Yuna has the largest number of points
theorem yuna_has_most_points :
  yuna_points ≥ yoongi_points ∧
  yuna_points ≥ jungkook_points ∧
  yuna_points ≥ yoojung_points :=
by sorry

end NUMINAMATH_CALUDE_yuna_has_most_points_l1674_167440


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l1674_167493

/-- Given a baker who made 167 cakes and sold 108 cakes, prove that the number of cakes remaining is 59. -/
theorem baker_remaining_cakes (cakes_made : ℕ) (cakes_sold : ℕ) 
  (h1 : cakes_made = 167) (h2 : cakes_sold = 108) : 
  cakes_made - cakes_sold = 59 := by
  sorry

#check baker_remaining_cakes

end NUMINAMATH_CALUDE_baker_remaining_cakes_l1674_167493


namespace NUMINAMATH_CALUDE_average_of_special_squares_l1674_167477

/-- Represents a 4x4 grid filled with numbers 1, 3, 5, and 7 -/
def Grid := Fin 4 → Fin 4 → Fin 4

/-- Checks if a row contains different numbers -/
def row_valid (g : Grid) (i : Fin 4) : Prop :=
  ∀ j k : Fin 4, j ≠ k → g i j ≠ g i k

/-- Checks if a column contains different numbers -/
def col_valid (g : Grid) (j : Fin 4) : Prop :=
  ∀ i k : Fin 4, i ≠ k → g i j ≠ g k j

/-- Checks if a 2x2 board contains different numbers -/
def board_valid (g : Grid) (i j : Fin 2) : Prop :=
  ∀ x y z w : Fin 2, (x, y) ≠ (z, w) → g (i + x) (j + y) ≠ g (i + z) (j + w)

/-- Checks if the entire grid is valid -/
def grid_valid (g : Grid) : Prop :=
  (∀ i : Fin 4, row_valid g i) ∧
  (∀ j : Fin 4, col_valid g j) ∧
  (∀ i j : Fin 2, board_valid g i j)

/-- The set of valid numbers in the grid -/
def valid_numbers : Finset (Fin 4) :=
  {0, 1, 2, 3}

/-- Maps Fin 4 to the actual numbers used in the grid -/
def to_actual_number (n : Fin 4) : ℕ :=
  2 * n + 1

/-- Theorem: The average of numbers in squares A, B, C, D is 4 -/
theorem average_of_special_squares (g : Grid) (hg : grid_valid g) :
  (to_actual_number (g 0 0) + to_actual_number (g 0 3) +
   to_actual_number (g 3 0) + to_actual_number (g 3 3)) / 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_special_squares_l1674_167477


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l1674_167490

/-- Given a parabola with equation y² = 4x, its latus rectum has the equation x = -1 -/
theorem latus_rectum_of_parabola :
  ∀ (x y : ℝ), y^2 = 4*x → (∃ (x₀ : ℝ), x₀ = -1 ∧ ∀ (y₀ : ℝ), (y₀^2 = 4*x₀ → x₀ = -1)) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l1674_167490


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l1674_167478

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (10 * bowling_ball_weight = 5 * canoe_weight) →
    (3 * canoe_weight = 120) →
    bowling_ball_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l1674_167478


namespace NUMINAMATH_CALUDE_recycling_drive_target_l1674_167498

/-- Calculates the target amount of kilos for a recycling drive given the number of sections,
    amount collected per section in two weeks, and additional amount needed. -/
def recycling_target (sections : ℕ) (kilos_per_section_two_weeks : ℕ) (additional_kilos : ℕ) : ℕ :=
  let kilos_per_section_per_week := kilos_per_section_two_weeks / 2
  let kilos_per_section_three_weeks := kilos_per_section_per_week * 3
  let total_collected := kilos_per_section_three_weeks * sections
  total_collected + additional_kilos

/-- The recycling drive target matches the calculated amount. -/
theorem recycling_drive_target :
  recycling_target 6 280 320 = 2840 := by
  sorry

end NUMINAMATH_CALUDE_recycling_drive_target_l1674_167498


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1674_167488

theorem max_value_of_expression (t : ℝ) :
  (∃ (max : ℝ), max = (1 / 8) ∧
    ∀ (t : ℝ), ((3^t - 2*t^2)*t) / (9^t) ≤ max ∧
    ∃ (t_max : ℝ), ((3^t_max - 2*t_max^2)*t_max) / (9^t_max) = max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1674_167488


namespace NUMINAMATH_CALUDE_ab_geq_4_and_a_plus_b_geq_4_relationship_l1674_167445

theorem ab_geq_4_and_a_plus_b_geq_4_relationship (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, a > 0 → b > 0 → a * b ≥ 4 → a + b ≥ 4) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b ≥ 4 ∧ a * b < 4) := by
  sorry

end NUMINAMATH_CALUDE_ab_geq_4_and_a_plus_b_geq_4_relationship_l1674_167445


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_N_div_100_l1674_167480

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

def sum_of_fractions : ℚ :=
  1 / (factorial 2 * factorial 17) +
  1 / (factorial 3 * factorial 16) +
  1 / (factorial 4 * factorial 15) +
  1 / (factorial 5 * factorial 14) +
  1 / (factorial 6 * factorial 13) +
  1 / (factorial 7 * factorial 12) +
  1 / (factorial 8 * factorial 11) +
  1 / (factorial 9 * factorial 10)

def N : ℚ := sum_of_fractions * factorial 18

theorem greatest_integer_less_than_N_div_100 :
  ⌊N / 100⌋ = 137 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_N_div_100_l1674_167480


namespace NUMINAMATH_CALUDE_boat_current_rate_l1674_167452

/-- Proves that the rate of current is 5 km/hr given the conditions of the boat problem -/
theorem boat_current_rate (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 5 →
  downstream_time = 1/5 →
  ∃ (current_rate : ℝ), 
    downstream_distance = (boat_speed + current_rate) * downstream_time ∧
    current_rate = 5 :=
by sorry

end NUMINAMATH_CALUDE_boat_current_rate_l1674_167452


namespace NUMINAMATH_CALUDE_initial_students_count_l1674_167419

/-- The number of students initially on the bus -/
def initial_students : ℕ := sorry

/-- The number of students who got on at the first stop -/
def students_who_got_on : ℕ := 3

/-- The total number of students on the bus after the first stop -/
def total_students : ℕ := 13

/-- Theorem stating that the initial number of students was 10 -/
theorem initial_students_count : initial_students = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_students_count_l1674_167419


namespace NUMINAMATH_CALUDE_longest_interval_l1674_167458

-- Define the conversion factors
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 24

-- Define the time intervals
def interval_a : ℕ := 1500  -- in minutes
def interval_b : ℕ := 10    -- in hours
def interval_c : ℕ := 1     -- in days

-- Theorem to prove
theorem longest_interval :
  (interval_a : ℝ) > (interval_b * minutes_per_hour : ℝ) ∧
  (interval_a : ℝ) > (interval_c * hours_per_day * minutes_per_hour : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_longest_interval_l1674_167458


namespace NUMINAMATH_CALUDE_rotated_region_volume_is_19pi_l1674_167472

/-- The volume of a solid formed by rotating a region about the y-axis. The region consists of:
    1. A vertical strip of 7 unit squares high and 1 unit wide along the y-axis.
    2. A horizontal strip of 3 unit squares wide and 2 units high along the x-axis, 
       starting from the top of the vertical strip. -/
def rotated_region_volume : ℝ := sorry

/-- The theorem states that the volume of the rotated region is equal to 19π cubic units. -/
theorem rotated_region_volume_is_19pi : rotated_region_volume = 19 * Real.pi := by sorry

end NUMINAMATH_CALUDE_rotated_region_volume_is_19pi_l1674_167472


namespace NUMINAMATH_CALUDE_fathers_age_l1674_167496

theorem fathers_age (son_age father_age : ℕ) : 
  father_age = 3 * son_age + 3 →
  father_age + 3 = 2 * (son_age + 3) + 8 →
  father_age = 27 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l1674_167496


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1674_167446

theorem cubic_equation_roots (p q : ℝ) :
  let x₁ := (-p + Real.sqrt (p^2 - 4*q)) / 2
  let x₂ := (-p - Real.sqrt (p^2 - 4*q)) / 2
  let cubic := fun y : ℝ => y^3 - (p^2 - q)*y^2 + (p^2*q - q^2)*y - q^3
  (cubic x₁^2 = 0) ∧ (cubic (x₁*x₂) = 0) ∧ (cubic x₂^2 = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1674_167446


namespace NUMINAMATH_CALUDE_value_of_expression_l1674_167469

theorem value_of_expression (x : ℝ) (h : x = 2) : 3^x - x^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1674_167469


namespace NUMINAMATH_CALUDE_curve_symmetry_condition_l1674_167485

/-- Given a curve y = x + p/x where p ≠ 0, this theorem states that the condition for two distinct
points on the curve to be symmetric with respect to the line y = x is satisfied if and only if p < 0 -/
theorem curve_symmetry_condition (p : ℝ) (hp : p ≠ 0) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
   x₁ + p / x₁ = x₂ + p / x₂ ∧
   x₁ + p / x₁ + x₂ + p / x₂ = x₁ + x₂) ↔ 
  p < 0 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetry_condition_l1674_167485


namespace NUMINAMATH_CALUDE_price_reduction_equation_correct_l1674_167433

/-- Represents the price reduction scenario -/
structure PriceReduction where
  initial_price : ℝ
  final_price : ℝ
  num_reductions : ℕ
  
/-- The price reduction equation is correct for the given scenario -/
theorem price_reduction_equation_correct (pr : PriceReduction) 
  (h1 : pr.initial_price = 560)
  (h2 : pr.final_price = 315)
  (h3 : pr.num_reductions = 2) :
  ∃ x : ℝ, pr.initial_price * (1 - x)^pr.num_reductions = pr.final_price :=
sorry

end NUMINAMATH_CALUDE_price_reduction_equation_correct_l1674_167433


namespace NUMINAMATH_CALUDE_problem_solution_l1674_167412

theorem problem_solution (n : ℝ) : 32 - 16 = n * 4 → (n / 4) + 16 = 17 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1674_167412


namespace NUMINAMATH_CALUDE_binomial_sum_formula_l1674_167461

def binomial_sum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun k => (k + 1) * (k + 2) * Nat.choose n (k + 1))

theorem binomial_sum_formula (n : ℕ) (h : n ≥ 4) :
  binomial_sum n = n * (n + 3) * 2^(n - 2) :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_formula_l1674_167461


namespace NUMINAMATH_CALUDE_set_propositions_equivalence_l1674_167436

theorem set_propositions_equivalence (A B : Set α) :
  (((A ∪ B ≠ B) → (A ∩ B ≠ A)) ∧
   ((A ∩ B ≠ A) → (A ∪ B ≠ B)) ∧
   ((A ∪ B = B) → (A ∩ B = A)) ∧
   ((A ∩ B = A) → (A ∪ B = B))) := by
  sorry

end NUMINAMATH_CALUDE_set_propositions_equivalence_l1674_167436


namespace NUMINAMATH_CALUDE_wall_bricks_count_l1674_167447

theorem wall_bricks_count :
  ∀ (x : ℕ),
  (∃ (rate1 rate2 : ℚ),
    rate1 = x / 9 ∧
    rate2 = x / 10 ∧
    5 * (rate1 + rate2 - 10) = x) →
  x = 900 := by
sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l1674_167447


namespace NUMINAMATH_CALUDE_cubic_polynomial_third_root_l1674_167475

theorem cubic_polynomial_third_root 
  (a b : ℚ) 
  (h1 : a * 1^3 + (a + 3*b) * 1^2 + (b - 4*a) * 1 + (6 - a) = 0)
  (h2 : a * (-3)^3 + (a + 3*b) * (-3)^2 + (b - 4*a) * (-3) + (6 - a) = 0) :
  ∃ (x : ℚ), x = 7/13 ∧ a * x^3 + (a + 3*b) * x^2 + (b - 4*a) * x + (6 - a) = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_third_root_l1674_167475


namespace NUMINAMATH_CALUDE_brown_eyes_ratio_l1674_167421

/-- Represents the number of people with different eye colors in a theater. -/
structure TheaterEyeColors where
  total : ℕ
  blue : ℕ
  black : ℕ
  green : ℕ
  brown : ℕ

/-- Theorem stating the ratio of people with brown eyes to total people in the theater. -/
theorem brown_eyes_ratio (t : TheaterEyeColors) :
  t.total = 100 ∧ 
  t.blue = 19 ∧ 
  t.black = t.total / 4 ∧ 
  t.green = 6 ∧ 
  t.brown = t.total - (t.blue + t.black + t.green) →
  2 * t.brown = t.total := by
  sorry

#check brown_eyes_ratio

end NUMINAMATH_CALUDE_brown_eyes_ratio_l1674_167421


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1674_167456

theorem complex_modulus_problem (z : ℂ) : z = (2 * Complex.I) / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1674_167456


namespace NUMINAMATH_CALUDE_min_chords_for_circle_l1674_167427

/-- Given a circle with chords subtending a central angle of 120°, 
    prove that the minimum number of such chords to complete the circle is 3. -/
theorem min_chords_for_circle (n : ℕ) : n > 0 → (120 * n = 360 * m) → n ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_chords_for_circle_l1674_167427


namespace NUMINAMATH_CALUDE_certain_number_proof_l1674_167415

theorem certain_number_proof : ∃ x : ℝ, 45 * x = 0.35 * 900 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1674_167415


namespace NUMINAMATH_CALUDE_university_diploma_percentage_l1674_167489

theorem university_diploma_percentage
  (no_diploma_with_job : Real)
  (diploma_without_job : Real)
  (job_of_choice : Real)
  (h1 : no_diploma_with_job = 0.18)
  (h2 : diploma_without_job = 0.25)
  (h3 : job_of_choice = 0.4) :
  (job_of_choice - no_diploma_with_job) + (diploma_without_job * (1 - job_of_choice)) = 0.37 := by
sorry

end NUMINAMATH_CALUDE_university_diploma_percentage_l1674_167489


namespace NUMINAMATH_CALUDE_simplify_expression_l1674_167482

theorem simplify_expression (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) : 
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1674_167482


namespace NUMINAMATH_CALUDE_cody_lost_tickets_l1674_167429

theorem cody_lost_tickets (initial : Real) (spent : Real) (left : Real) : 
  initial = 49.0 → spent = 25.0 → left = 18 → initial - spent - left = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_cody_lost_tickets_l1674_167429


namespace NUMINAMATH_CALUDE_harry_seashells_count_l1674_167491

theorem harry_seashells_count :
  ∀ (seashells : ℕ),
    -- Initial collection
    34 + seashells + 29 = 34 + seashells + 29 →
    -- Total items lost
    25 = 25 →
    -- Items left at the end
    59 = 59 →
    -- Proof that seashells = 21
    seashells = 21 := by
  sorry

end NUMINAMATH_CALUDE_harry_seashells_count_l1674_167491


namespace NUMINAMATH_CALUDE_lcm_5_6_8_9_l1674_167499

theorem lcm_5_6_8_9 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_6_8_9_l1674_167499


namespace NUMINAMATH_CALUDE_lisa_marble_difference_l1674_167487

/-- Proves that Lisa has 19 more marbles than Cindy after the marble exchange -/
theorem lisa_marble_difference (cindy_initial : ℕ) (lisa_initial : ℕ) (marbles_given : ℕ) : 
  cindy_initial = 20 →
  cindy_initial = lisa_initial + 5 →
  marbles_given = 12 →
  (lisa_initial + marbles_given) - (cindy_initial - marbles_given) = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_lisa_marble_difference_l1674_167487


namespace NUMINAMATH_CALUDE_sugar_profit_percentage_l1674_167497

theorem sugar_profit_percentage 
  (total_sugar : ℝ) 
  (sugar_at_18_percent : ℝ) 
  (overall_profit_percentage : ℝ) :
  total_sugar = 1000 →
  sugar_at_18_percent = 600 →
  overall_profit_percentage = 14 →
  ∃ (unknown_profit_percentage : ℝ),
    unknown_profit_percentage = 80 ∧
    sugar_at_18_percent * (18 / 100) + 
    (total_sugar - sugar_at_18_percent) * (unknown_profit_percentage / 100) = 
    total_sugar * (overall_profit_percentage / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_sugar_profit_percentage_l1674_167497


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1674_167466

theorem triangle_angle_proof (A B C : Real) (a b c : Real) :
  -- Conditions
  (a = 2) →
  (b = Real.sqrt 3) →
  (B = π / 3) →
  -- Triangle definition (implicitly assuming it's a valid triangle)
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  -- Sine law
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion
  A = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1674_167466


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1674_167444

-- Define the original number
def original_number : ℕ := 262883000000

-- Define the scientific notation components
def significand : ℚ := 2.62883
def exponent : ℕ := 11

-- Theorem statement
theorem scientific_notation_correct : 
  (significand * (10 : ℚ) ^ exponent) = original_number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1674_167444


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l1674_167448

theorem stratified_sampling_sample_size 
  (total_employees : ℕ) 
  (young_workers : ℕ) 
  (sample_young : ℕ) 
  (h1 : total_employees = 750) 
  (h2 : young_workers = 350) 
  (h3 : sample_young = 7) : 
  ∃ (sample_size : ℕ), 
    sample_size * young_workers = sample_young * total_employees ∧ 
    sample_size = 15 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l1674_167448


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l1674_167451

theorem simplify_radical_expression :
  (Real.sqrt 6 + 4 * Real.sqrt 3 + 3 * Real.sqrt 2) / 
  ((Real.sqrt 6 + Real.sqrt 3) * (Real.sqrt 3 + Real.sqrt 2)) = 
  Real.sqrt 6 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l1674_167451


namespace NUMINAMATH_CALUDE_tea_consumption_l1674_167483

theorem tea_consumption (total : ℕ) (days : ℕ) (diff : ℕ) : 
  total = 120 → days = 6 → diff = 4 → 
  ∃ (first : ℕ), 
    (first + 3 * diff = 22) ∧ 
    (days * (2 * first + (days - 1) * diff) / 2 = total) := by
  sorry

end NUMINAMATH_CALUDE_tea_consumption_l1674_167483


namespace NUMINAMATH_CALUDE_sum_of_squares_149_l1674_167402

theorem sum_of_squares_149 : ∃ (a b c : ℕ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = 21 ∧
  a^2 + b^2 + c^2 = 149 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_149_l1674_167402


namespace NUMINAMATH_CALUDE_christopher_karen_difference_l1674_167468

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of quarters Karen has -/
def karen_quarters : ℕ := 32

/-- The number of quarters Christopher has -/
def christopher_quarters : ℕ := 64

/-- The difference in money between Christopher and Karen -/
def money_difference : ℚ := (christopher_quarters - karen_quarters) * quarter_value

theorem christopher_karen_difference : money_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_christopher_karen_difference_l1674_167468


namespace NUMINAMATH_CALUDE_surfers_count_l1674_167410

/-- The number of surfers on Santa Monica beach -/
def santa_monica_surfers : ℕ := 20

/-- The number of surfers on Malibu beach -/
def malibu_surfers : ℕ := 2 * santa_monica_surfers

/-- The total number of surfers on both beaches -/
def total_surfers : ℕ := malibu_surfers + santa_monica_surfers

theorem surfers_count : total_surfers = 60 := by
  sorry

end NUMINAMATH_CALUDE_surfers_count_l1674_167410


namespace NUMINAMATH_CALUDE_no_real_roots_iff_k_gt_9_l1674_167459

theorem no_real_roots_iff_k_gt_9 (k : ℝ) : 
  (∀ x : ℝ, x^2 + k ≠ 6*x) ↔ k > 9 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_k_gt_9_l1674_167459


namespace NUMINAMATH_CALUDE_concrete_wall_width_l1674_167432

theorem concrete_wall_width
  (r : ℝ)  -- radius of the pool
  (w : ℝ)  -- width of the concrete wall
  (h1 : r = 20)  -- radius of the pool is 20 ft
  (h2 : π * ((r + w)^2 - r^2) = (11/25) * (π * r^2))  -- area of wall is 11/25 of pool area
  : w = 4 := by
  sorry

end NUMINAMATH_CALUDE_concrete_wall_width_l1674_167432


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1674_167406

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) :
  3 * a 9 - a 11 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1674_167406


namespace NUMINAMATH_CALUDE_cubic_line_bounded_area_l1674_167439

/-- The area bounded by a cubic function and a line -/
noncomputable def boundedArea (a b c d p q α β γ : ℝ) : ℝ :=
  |a| / 12 * (γ - α)^3 * |2*β - γ - α|

/-- Theorem stating the area bounded by a cubic function and a line -/
theorem cubic_line_bounded_area
  (a b c d p q α β γ : ℝ)
  (h_a : a ≠ 0)
  (h_order : α < β ∧ β < γ)
  (h_cubic : ∀ x, a*x^3 + b*x^2 + c*x + d = p*x + q → x = α ∨ x = β ∨ x = γ) :
  ∃ A, A = boundedArea a b c d p q α β γ ∧
    A = |∫ (x : ℝ) in α..γ, (a*x^3 + b*x^2 + c*x + d) - (p*x + q)| :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_line_bounded_area_l1674_167439


namespace NUMINAMATH_CALUDE_find_z_l1674_167481

theorem find_z (M N : Set ℂ) (i : ℂ) (z : ℂ) : 
  M = {1, 2, z * i} →
  N = {3, 4} →
  M ∩ N = {4} →
  i * i = -1 →
  z = 4 * i :=
by sorry

end NUMINAMATH_CALUDE_find_z_l1674_167481


namespace NUMINAMATH_CALUDE_triangle_inequality_cosine_law_l1674_167450

theorem triangle_inequality_cosine_law (x y z α β γ : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_angle_range : 0 ≤ α ∧ α < π ∧ 0 ≤ β ∧ β < π ∧ 0 ≤ γ ∧ γ < π)
  (h_angle_sum : α + β > γ ∧ β + γ > α ∧ γ + α > β) :
  Real.sqrt (x^2 + y^2 - 2*x*y*(Real.cos α)) + Real.sqrt (y^2 + z^2 - 2*y*z*(Real.cos β)) 
  ≥ Real.sqrt (z^2 + x^2 - 2*z*x*(Real.cos γ)) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_cosine_law_l1674_167450


namespace NUMINAMATH_CALUDE_spam_price_theorem_l1674_167414

-- Define the constants from the problem
def peanut_butter_price : ℝ := 5
def bread_price : ℝ := 2
def spam_cans : ℕ := 12
def peanut_butter_jars : ℕ := 3
def bread_loaves : ℕ := 4
def total_paid : ℝ := 59

-- Define the theorem
theorem spam_price_theorem :
  ∃ (spam_price : ℝ),
    spam_price * spam_cans +
    peanut_butter_price * peanut_butter_jars +
    bread_price * bread_loaves = total_paid ∧
    spam_price = 3 :=
by sorry

end NUMINAMATH_CALUDE_spam_price_theorem_l1674_167414


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l1674_167401

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 9)) / x ≤ 3 - Real.sqrt 6 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, x > 0 ∧ (x^2 + 3 - Real.sqrt (x^4 + 9)) / x = 3 - Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l1674_167401


namespace NUMINAMATH_CALUDE_plane_hit_probability_l1674_167434

theorem plane_hit_probability (p_A p_B : ℝ) (h_A : p_A = 0.3) (h_B : p_B = 0.5) :
  1 - (1 - p_A) * (1 - p_B) = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_plane_hit_probability_l1674_167434


namespace NUMINAMATH_CALUDE_stratified_sampling_management_l1674_167479

theorem stratified_sampling_management (total_employees : ℕ) (management : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 150)
  (h2 : management = 15)
  (h3 : sample_size = 30) :
  (management * sample_size) / total_employees = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_management_l1674_167479


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l1674_167403

/-- The function f(x) = ax^2 - 2x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 2

/-- The function F(x) = |f(x)| -/
def F (a : ℝ) (x : ℝ) : ℝ := |f a x|

/-- The theorem statement -/
theorem f_increasing_on_interval (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → x₁ ≠ x₂ →
    (F a x₁ - F a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Iic 0 ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l1674_167403


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l1674_167431

/-- Represents a repeating decimal with a whole number part and a repeating part -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def repeating_decimal_to_rational (d : RepeatingDecimal) : ℚ :=
  d.whole + d.repeating / (99 : ℚ)

/-- The theorem stating that the division of two specific repeating decimals equals 3/10 -/
theorem repeating_decimal_division :
  let d1 := RepeatingDecimal.mk 0 81
  let d2 := RepeatingDecimal.mk 2 72
  (repeating_decimal_to_rational d1) / (repeating_decimal_to_rational d2) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l1674_167431


namespace NUMINAMATH_CALUDE_paulas_travel_time_fraction_l1674_167441

theorem paulas_travel_time_fraction (luke_bus_time : ℕ) (total_travel_time : ℕ) 
  (h1 : luke_bus_time = 70)
  (h2 : total_travel_time = 504) :
  ∃ f : ℚ, 
    f = 3/5 ∧ 
    (luke_bus_time + 5 * luke_bus_time + 2 * (f * luke_bus_time) : ℚ) = total_travel_time :=
by sorry

end NUMINAMATH_CALUDE_paulas_travel_time_fraction_l1674_167441


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1674_167454

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (-1 + Complex.I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1674_167454


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l1674_167435

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l1674_167435


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1674_167460

theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) / Complex.I = 1 - z) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1674_167460


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1674_167492

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(12 ∣ (427398 - y))) ∧
  (12 ∣ (427398 - x)) := by
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1674_167492


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1674_167471

theorem train_bridge_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) :
  train_length = 130 →
  bridge_length = 150 →
  train_speed_kmph = 36 →
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 28 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1674_167471


namespace NUMINAMATH_CALUDE_books_together_l1674_167400

-- Define the number of books Tim and Mike have
def tim_books : ℕ := 22
def mike_books : ℕ := 20

-- Define the total number of books
def total_books : ℕ := tim_books + mike_books

-- Theorem to prove
theorem books_together : total_books = 42 := by
  sorry

end NUMINAMATH_CALUDE_books_together_l1674_167400


namespace NUMINAMATH_CALUDE_cloth_length_problem_l1674_167428

theorem cloth_length_problem (initial_length : ℝ) 
  (h1 : initial_length > 32)
  (h2 : initial_length > 20) :
  (initial_length - 32) * 3 = initial_length - 20 →
  initial_length = 38 := by
  sorry

end NUMINAMATH_CALUDE_cloth_length_problem_l1674_167428


namespace NUMINAMATH_CALUDE_specific_l_shape_perimeter_l1674_167495

/-- An L-shaped figure formed by squares -/
structure LShapedFigure where
  squareSideLength : ℕ
  baseSquares : ℕ
  stackedSquares : ℕ

/-- Calculate the perimeter of an L-shaped figure -/
def perimeter (figure : LShapedFigure) : ℕ :=
  2 * figure.squareSideLength * (figure.baseSquares + figure.stackedSquares + 1)

/-- Theorem: The perimeter of the specific L-shaped figure is 14 units -/
theorem specific_l_shape_perimeter :
  let figure : LShapedFigure := ⟨2, 3, 2⟩
  perimeter figure = 14 := by
  sorry

end NUMINAMATH_CALUDE_specific_l_shape_perimeter_l1674_167495


namespace NUMINAMATH_CALUDE_binary_sum_is_eleven_l1674_167438

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 101₂ -/
def binary1 : List Bool := [true, false, true]

/-- The second binary number 110₂ -/
def binary2 : List Bool := [false, true, true]

/-- The sum of binary1 and binary2 in decimal form -/
def sum_decimal : ℕ := binary_to_decimal binary1 + binary_to_decimal binary2

theorem binary_sum_is_eleven : sum_decimal = 11 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_is_eleven_l1674_167438


namespace NUMINAMATH_CALUDE_range_of_a_l1674_167420

theorem range_of_a (a x : ℝ) : 
  (∀ x, (a - 4 < x ∧ x < a + 4) → (x - 2) * (x - 3) > 0) →
  (a ≤ -2 ∨ a ≥ 7) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1674_167420


namespace NUMINAMATH_CALUDE_euler_totient_properties_l1674_167494

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Definition: p is prime -/
def is_prime (p : ℕ) : Prop := sorry

theorem euler_totient_properties (p : ℕ) (α : ℕ) (h : is_prime p) (h' : α > 0) :
  (phi 17 = 16) ∧
  (phi p = p - 1) ∧
  (phi (p^2) = p * (p - 1)) ∧
  (phi (p^α) = p^(α-1) * (p - 1)) :=
sorry

end NUMINAMATH_CALUDE_euler_totient_properties_l1674_167494


namespace NUMINAMATH_CALUDE_average_equation_solution_l1674_167413

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((x + 8) + (5*x + 3) + (3*x + 4)) = 4*x + 1 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l1674_167413


namespace NUMINAMATH_CALUDE_newspaper_cost_8_weeks_l1674_167463

/-- The cost of newspapers over a period of weeks -/
def newspaper_cost (weekday_price : ℚ) (sunday_price : ℚ) (num_weeks : ℕ) : ℚ :=
  (3 * weekday_price + sunday_price) * num_weeks

/-- Proof that the total cost of newspapers for 8 weeks is $28.00 -/
theorem newspaper_cost_8_weeks :
  newspaper_cost 0.5 2 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_cost_8_weeks_l1674_167463


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1674_167457

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1674_167457
