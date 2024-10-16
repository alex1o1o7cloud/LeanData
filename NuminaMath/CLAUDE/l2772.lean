import Mathlib

namespace NUMINAMATH_CALUDE_total_ways_eq_64_l2772_277207

/-- The number of sports available to choose from -/
def num_sports : ℕ := 4

/-- The number of people choosing sports -/
def num_people : ℕ := 3

/-- The total number of different ways to choose sports -/
def total_ways : ℕ := num_sports ^ num_people

/-- Theorem stating that the total number of ways to choose sports is 64 -/
theorem total_ways_eq_64 : total_ways = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_eq_64_l2772_277207


namespace NUMINAMATH_CALUDE_pizza_distribution_l2772_277259

/-- Given the number of brothers, slices in small and large pizzas, and the number of each type of pizza ordered, 
    calculate the number of slices each brother can eat. -/
def slices_per_brother (num_brothers : ℕ) (slices_small : ℕ) (slices_large : ℕ) 
                       (num_small : ℕ) (num_large : ℕ) : ℕ :=
  (num_small * slices_small + num_large * slices_large) / num_brothers

/-- Theorem stating that under the given conditions, each brother can eat 12 slices of pizza. -/
theorem pizza_distribution :
  slices_per_brother 3 8 14 1 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_distribution_l2772_277259


namespace NUMINAMATH_CALUDE_min_box_value_l2772_277262

theorem min_box_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a*x + b) * (b*x + 2*a) = 36*x^2 + box*x + 72) →
  a ≠ b ∧ b ≠ box ∧ a ≠ box →
  (∃ a' b' box', 
    (∀ x, (a'*x + b') * (b'*x + 2*a') = 36*x^2 + box'*x + 72) ∧
    a' ≠ b' ∧ b' ≠ box' ∧ a' ≠ box' ∧
    box' < box) →
  box ≥ 332 :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l2772_277262


namespace NUMINAMATH_CALUDE_intersecting_circles_equal_chords_l2772_277209

/-- Given two intersecting circles with radii 10 and 8 units, whose centers are 15 units apart,
    if a line is drawn through their intersection point P such that it creates equal chords QP and PR,
    then the square of the length of chord QP is 250. -/
theorem intersecting_circles_equal_chords (r₁ r₂ d : ℝ) (P Q R : ℝ × ℝ) :
  r₁ = 10 →
  r₂ = 8 →
  d = 15 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1 - R.1)^2 + (P.2 - R.2)^2 →
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 250 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_circles_equal_chords_l2772_277209


namespace NUMINAMATH_CALUDE_equation_solution_l2772_277217

theorem equation_solution : ∃ x : ℝ, (x - 6) ^ 4 = (1 / 16)⁻¹ ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2772_277217


namespace NUMINAMATH_CALUDE_h_closed_form_l2772_277258

def h : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * h n + 2 * n

theorem h_closed_form (n : ℕ) : h n = 2^n + n^2 - n := by
  sorry

end NUMINAMATH_CALUDE_h_closed_form_l2772_277258


namespace NUMINAMATH_CALUDE_only_negative_five_smaller_than_negative_three_l2772_277280

theorem only_negative_five_smaller_than_negative_three :
  let numbers : List ℚ := [0, -1, -5, -1/2]
  ∀ x ∈ numbers, x < -3 ↔ x = -5 := by
sorry

end NUMINAMATH_CALUDE_only_negative_five_smaller_than_negative_three_l2772_277280


namespace NUMINAMATH_CALUDE_probability_two_white_balls_l2772_277211

/-- The probability of drawing two white balls sequentially without replacement from a box containing 7 white balls and 8 black balls is 1/5. -/
theorem probability_two_white_balls (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) :
  total_balls = white_balls + black_balls →
  white_balls = 7 →
  black_balls = 8 →
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_white_balls_l2772_277211


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2772_277232

theorem simplify_trig_expression (θ : ℝ) (h : θ ∈ Set.Ioo 0 (π / 4)) :
  Real.sqrt (1 - 2 * Real.sin (π + θ) * Real.sin ((3 * π) / 2 - θ)) = Real.cos θ - Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2772_277232


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l2772_277284

theorem added_number_after_doubling (initial_number : ℕ) (x : ℕ) : 
  initial_number = 8 → 
  3 * (2 * initial_number + x) = 75 → 
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l2772_277284


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2772_277260

theorem complex_expression_evaluation (z : ℂ) (h : z = 1 + I) :
  2 / z + z^2 = 1 + I := by sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2772_277260


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l2772_277206

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) : 
  cube_edge = 5 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l2772_277206


namespace NUMINAMATH_CALUDE_combination_sum_equals_seven_l2772_277205

theorem combination_sum_equals_seven (n : ℕ) 
  (h1 : 0 ≤ 5 - n ∧ 5 - n ≤ n) 
  (h2 : 0 ≤ 10 - n ∧ 10 - n ≤ n + 1) : 
  Nat.choose n (5 - n) + Nat.choose (n + 1) (10 - n) = 7 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_seven_l2772_277205


namespace NUMINAMATH_CALUDE_chairs_to_remove_l2772_277293

theorem chairs_to_remove (initial_chairs : ℕ) (chairs_per_row : ℕ) (expected_participants : ℕ)
  (h1 : initial_chairs = 196)
  (h2 : chairs_per_row = 14)
  (h3 : expected_participants = 120)
  (h4 : chairs_per_row > 0) :
  let remaining_chairs := ((expected_participants + chairs_per_row - 1) / chairs_per_row) * chairs_per_row
  initial_chairs - remaining_chairs = 70 := by
sorry

end NUMINAMATH_CALUDE_chairs_to_remove_l2772_277293


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2772_277254

/-- Given a geometric sequence {a_n} with common ratio q = 1/2, prove that S_4 / a_2 = 15/4,
    where S_n is the sum of the first n terms. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = (1 / 2) * a n) →  -- Common ratio q = 1/2
  (∀ n, S n = a 1 * (1 - (1 / 2)^n) / (1 - (1 / 2))) →  -- Definition of S_n
  S 4 / a 2 = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2772_277254


namespace NUMINAMATH_CALUDE_abs_of_complex_fraction_l2772_277255

open Complex

theorem abs_of_complex_fraction : 
  let z : ℂ := (4 - 2*I) / (1 + I)
  abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_abs_of_complex_fraction_l2772_277255


namespace NUMINAMATH_CALUDE_two_sin_sixty_degrees_l2772_277265

theorem two_sin_sixty_degrees : 2 * Real.sin (π / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_two_sin_sixty_degrees_l2772_277265


namespace NUMINAMATH_CALUDE_profit_percentage_invariant_l2772_277210

/-- Represents the profit percentage as a real number between 0 and 1 -/
def ProfitPercentage := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- Represents the discount percentage as a real number between 0 and 1 -/
def DiscountPercentage := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- The profit percentage remains the same regardless of the discount -/
theorem profit_percentage_invariant (profit_with_discount : ProfitPercentage) 
  (discount : DiscountPercentage) :
  ∃ (profit_without_discount : ProfitPercentage), 
  profit_without_discount = profit_with_discount :=
sorry

end NUMINAMATH_CALUDE_profit_percentage_invariant_l2772_277210


namespace NUMINAMATH_CALUDE_xy_sum_l2772_277220

theorem xy_sum (x y : ℤ) (h : 2*x*y + x + y = 83) : x + y = 83 ∨ x + y = -85 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_l2772_277220


namespace NUMINAMATH_CALUDE_snow_probability_first_week_january_l2772_277242

theorem snow_probability_first_week_january : 
  let p1 : ℚ := 1/2  -- Probability of snow for each of the first 3 days
  let p2 : ℚ := 1/3  -- Probability of snow for each of the next 4 days
  let days1 : ℕ := 3 -- Number of days with probability p1
  let days2 : ℕ := 4 -- Number of days with probability p2
  1 - (1 - p1) ^ days1 * (1 - p2) ^ days2 = 79/81 :=
by sorry

end NUMINAMATH_CALUDE_snow_probability_first_week_january_l2772_277242


namespace NUMINAMATH_CALUDE_valid_combinations_count_l2772_277273

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of valid four-letter word combinations where the word begins and ends with the same letter, and the second letter is a vowel -/
def valid_combinations : ℕ := alphabet_size * vowel_count * alphabet_size

theorem valid_combinations_count : valid_combinations = 3380 := by
  sorry

end NUMINAMATH_CALUDE_valid_combinations_count_l2772_277273


namespace NUMINAMATH_CALUDE_iris_shopping_l2772_277202

theorem iris_shopping (jacket_price shorts_price pants_price total_spent : ℕ)
  (jacket_quantity pants_quantity : ℕ) :
  jacket_price = 10 →
  shorts_price = 6 →
  pants_price = 12 →
  jacket_quantity = 3 →
  pants_quantity = 4 →
  total_spent = 90 →
  ∃ shorts_quantity : ℕ, 
    total_spent = jacket_price * jacket_quantity + 
                  shorts_price * shorts_quantity + 
                  pants_price * pants_quantity ∧
    shorts_quantity = 2 :=
by sorry

end NUMINAMATH_CALUDE_iris_shopping_l2772_277202


namespace NUMINAMATH_CALUDE_annual_decrease_rate_l2772_277236

/-- Proves that the annual decrease rate is 20% for a town with given population changes. -/
theorem annual_decrease_rate (initial_population : ℝ) (population_after_two_years : ℝ) 
  (h1 : initial_population = 15000)
  (h2 : population_after_two_years = 9600) :
  ∃ (r : ℝ), r = 20 ∧ population_after_two_years = initial_population * (1 - r / 100)^2 := by
  sorry

end NUMINAMATH_CALUDE_annual_decrease_rate_l2772_277236


namespace NUMINAMATH_CALUDE_mary_work_hours_l2772_277237

/-- Mary's weekly work schedule and earnings -/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ
  tue_thu_hours : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Theorem stating Mary's work hours on Monday, Wednesday, and Friday -/
theorem mary_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.tue_thu_hours = 5)
  (h2 : schedule.weekly_earnings = 407)
  (h3 : schedule.hourly_rate = 11)
  (h4 : schedule.hourly_rate * (3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours) = schedule.weekly_earnings) :
  schedule.mon_wed_fri_hours = 9 := by
  sorry


end NUMINAMATH_CALUDE_mary_work_hours_l2772_277237


namespace NUMINAMATH_CALUDE_f_has_unique_zero_a_lower_bound_l2772_277201

noncomputable section

def f (x : ℝ) : ℝ := -1/2 * Real.log x + 2/(x+1)

theorem f_has_unique_zero :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

theorem a_lower_bound (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (1/Real.exp 1) 1 →
    ∀ t : ℝ, t ∈ Set.Icc (1/2) 2 →
      f x ≥ t^3 - t^2 - 2*a*t + 2) →
  a ≥ 5/4 :=
sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_a_lower_bound_l2772_277201


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_bound_l2772_277240

/-- A tetrahedron with an inscribed sphere -/
structure Tetrahedron where
  /-- Length of one pair of opposite edges -/
  a : ℝ
  /-- Length of the other pair of opposite edges -/
  b : ℝ
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- Ensure a and b are positive -/
  ha : 0 < a
  hb : 0 < b
  /-- Ensure r is positive -/
  hr : 0 < r

/-- The radius of the inscribed sphere is less than ab/(2(a+b)) -/
theorem inscribed_sphere_radius_bound (t : Tetrahedron) : t.r < (t.a * t.b) / (2 * (t.a + t.b)) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_bound_l2772_277240


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_condition_l2772_277221

theorem smallest_n_for_probability_condition : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (((m : ℝ) - 4)^3 / m^3 > 1/2) → m ≥ n) ∧
  ((n : ℝ) - 4)^3 / n^3 > 1/2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_condition_l2772_277221


namespace NUMINAMATH_CALUDE_bank_profit_optimization_l2772_277224

/-- Represents the bank's profit optimization problem -/
def BankProfitOptimization (initialEmployees : ℕ) (initialProfitPerEmployee : ℝ) 
  (profitIncreasePerLayoff : ℝ) (costPerLayoff : ℝ) (minEmployeeRatio : ℝ) : Prop :=
  ∃ (optimalLayoffs : ℕ),
    -- Optimal number of layoffs is within the allowed range
    (optimalLayoffs : ℝ) ≤ initialEmployees * (1 - minEmployeeRatio) ∧
    -- Profit function
    let profitFunction := λ (x : ℝ) => 
      (initialEmployees - x) * (initialProfitPerEmployee + profitIncreasePerLayoff * x) - costPerLayoff * x
    -- Optimal layoffs maximize the profit
    ∀ (x : ℝ), 0 ≤ x ∧ x ≤ initialEmployees * (1 - minEmployeeRatio) →
      profitFunction x ≤ profitFunction optimalLayoffs

/-- The specific instance of the bank's profit optimization problem -/
theorem bank_profit_optimization :
  BankProfitOptimization 320 200000 20000 60000 (3/4) ∧
  (∃ (optimalLayoffs : ℕ), optimalLayoffs = 80) :=
sorry

end NUMINAMATH_CALUDE_bank_profit_optimization_l2772_277224


namespace NUMINAMATH_CALUDE_quadrilateral_comparison_l2772_277298

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral :=
  (a b c d : Point)

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Calculates the perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℝ := sorry

/-- Quadrilateral I defined by its vertices -/
def quadI : Quadrilateral :=
  { a := {x := 0, y := 0},
    b := {x := 3, y := 0},
    c := {x := 3, y := 3},
    d := {x := 0, y := 2} }

/-- Quadrilateral II defined by its vertices -/
def quadII : Quadrilateral :=
  { a := {x := 0, y := 0},
    b := {x := 3, y := 0},
    c := {x := 3, y := 2},
    d := {x := 0, y := 3} }

theorem quadrilateral_comparison :
  (area quadI = 7.5 ∧ area quadII = 7.5) ∧
  perimeter quadI > perimeter quadII := by sorry

end NUMINAMATH_CALUDE_quadrilateral_comparison_l2772_277298


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_zero_l2772_277246

theorem sum_of_powers_equals_zero : 
  -1^2010 + (-1)^2013 + 1^2014 + (-1)^2016 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_zero_l2772_277246


namespace NUMINAMATH_CALUDE_original_number_is_27_l2772_277219

theorem original_number_is_27 :
  ∃ (n : ℕ), 
    (Odd (3 * n)) ∧ 
    (∃ (k : ℕ), k > 1 ∧ (3 * n) % k = 0) ∧ 
    (4 * n = 108) ∧
    n = 27 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_27_l2772_277219


namespace NUMINAMATH_CALUDE_completing_square_correct_l2772_277250

-- Define the original equation
def original_equation (x : ℝ) : Prop := x^2 - 4*x - 22 = 0

-- Define the result of completing the square
def completed_square_result (x : ℝ) : Prop := (x - 2)^2 = 26

-- Theorem statement
theorem completing_square_correct :
  ∀ x : ℝ, original_equation x ↔ completed_square_result x :=
by sorry

end NUMINAMATH_CALUDE_completing_square_correct_l2772_277250


namespace NUMINAMATH_CALUDE_wedding_ring_cost_l2772_277251

/-- Proves that the cost of the first wedding ring is $10,000 given the problem conditions --/
theorem wedding_ring_cost (first_ring_cost : ℝ) : 
  (3 * first_ring_cost - first_ring_cost / 2 = 25000) → 
  first_ring_cost = 10000 := by
  sorry

#check wedding_ring_cost

end NUMINAMATH_CALUDE_wedding_ring_cost_l2772_277251


namespace NUMINAMATH_CALUDE_factorization_equality_l2772_277275

theorem factorization_equality (x a : ℝ) : 4*x - x*a^2 = x*(2-a)*(2+a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2772_277275


namespace NUMINAMATH_CALUDE_art_club_artworks_l2772_277253

/-- The number of artworks collected by an art club over multiple school years -/
def artworks_collected (num_students : ℕ) (artworks_per_quarter : ℕ) (quarters_per_year : ℕ) (num_years : ℕ) : ℕ :=
  num_students * artworks_per_quarter * quarters_per_year * num_years

/-- Theorem: The art club collects 900 artworks in 3 school years -/
theorem art_club_artworks :
  artworks_collected 25 3 4 3 = 900 := by
  sorry

end NUMINAMATH_CALUDE_art_club_artworks_l2772_277253


namespace NUMINAMATH_CALUDE_first_valid_year_is_1979_l2772_277268

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 1970 ∧ sum_of_digits year = 15

theorem first_valid_year_is_1979 :
  (∀ y : ℕ, y < 1979 → ¬(is_valid_year y)) ∧ is_valid_year 1979 := by
  sorry

end NUMINAMATH_CALUDE_first_valid_year_is_1979_l2772_277268


namespace NUMINAMATH_CALUDE_nine_team_league_games_l2772_277279

/-- The number of games played in a league where each team plays every other team once -/
def numGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 9 teams, where each team plays every other team exactly once,
    the total number of games played is 36. -/
theorem nine_team_league_games :
  numGames 9 = 36 := by
  sorry


end NUMINAMATH_CALUDE_nine_team_league_games_l2772_277279


namespace NUMINAMATH_CALUDE_parallel_line_m_value_l2772_277231

/-- A line passing through two points is parallel to another line -/
def is_parallel_line (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  (y2 - y1) / (x2 - x1) = -a / b

/-- The value of m for which the line through (-2, m) and (m, 4) is parallel to 2x + y - 1 = 0 -/
theorem parallel_line_m_value :
  ∀ m : ℝ, is_parallel_line (-2) m m 4 2 1 (-1) → m = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_m_value_l2772_277231


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l2772_277229

theorem inequality_not_always_true (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x^2 > y^2) (hz : z ≠ 0) :
  ¬ ∀ (x y z : ℝ), x > 0 → y > 0 → x^2 > y^2 → z ≠ 0 → x * z^3 > y * z^3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l2772_277229


namespace NUMINAMATH_CALUDE_video_call_cost_proof_l2772_277287

/-- Calculates the cost of a video call given the charge rate and duration. -/
def video_call_cost (charge_rate : ℕ) (charge_interval : ℕ) (duration : ℕ) : ℕ :=
  (duration / charge_interval) * charge_rate

/-- Proves that a 2 minute and 40 second video call costs 480 won at a rate of 30 won per 10 seconds. -/
theorem video_call_cost_proof :
  let charge_rate : ℕ := 30
  let charge_interval : ℕ := 10
  let duration : ℕ := 2 * 60 + 40
  video_call_cost charge_rate charge_interval duration = 480 := by
  sorry

#eval video_call_cost 30 10 (2 * 60 + 40)

end NUMINAMATH_CALUDE_video_call_cost_proof_l2772_277287


namespace NUMINAMATH_CALUDE_race_probability_l2772_277271

theorem race_probability (p_x p_y p_z : ℚ) : 
  p_x = 1/8 →
  p_y = 1/12 →
  p_x + p_y + p_z = 375/1000 →
  p_z = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_race_probability_l2772_277271


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2772_277215

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => a₁ + d * i)

def count_terms (a₁ : ℤ) (d : ℤ) (aₙ : ℤ) : ℕ :=
  ((aₙ - a₁) / d).toNat + 1

def sum_multiples_of_10 (lst : List ℤ) : ℤ :=
  lst.filter (λ x => x % 10 = 0) |>.sum

theorem arithmetic_sequence_properties :
  let a₁ := -45
  let d := 7
  let aₙ := 98
  let n := count_terms a₁ d aₙ
  let seq := arithmetic_sequence a₁ d n
  n = 21 ∧ sum_multiples_of_10 seq = 60 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2772_277215


namespace NUMINAMATH_CALUDE_jean_gives_480_l2772_277234

/-- The amount Jean gives away to her grandchildren in a year -/
def total_amount_given (num_grandchildren : ℕ) (cards_per_grandchild : ℕ) (amount_per_card : ℕ) : ℕ :=
  num_grandchildren * cards_per_grandchild * amount_per_card

/-- Proof that Jean gives away $480 to her grandchildren in a year -/
theorem jean_gives_480 :
  total_amount_given 3 2 80 = 480 :=
by sorry

end NUMINAMATH_CALUDE_jean_gives_480_l2772_277234


namespace NUMINAMATH_CALUDE_not_solution_and_solutions_l2772_277223

def is_solution (x y : Int) : Prop :=
  85 * x - 324 * y = 101

theorem not_solution_and_solutions : 
  ¬(is_solution 978 256) ∧ 
  is_solution 5 1 ∧ 
  is_solution 329 86 ∧ 
  is_solution 653 171 ∧ 
  is_solution 1301 341 := by sorry

end NUMINAMATH_CALUDE_not_solution_and_solutions_l2772_277223


namespace NUMINAMATH_CALUDE_complex_point_location_l2772_277241

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- State the theorem
theorem complex_point_location :
  (2 : ℂ) / z + z^2 = 1 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_point_location_l2772_277241


namespace NUMINAMATH_CALUDE_billy_spits_30_inches_l2772_277277

/-- The distance Billy can spit a watermelon seed -/
def billy_distance : ℝ := sorry

/-- The distance Madison can spit a watermelon seed -/
def madison_distance : ℝ := sorry

/-- The distance Ryan can spit a watermelon seed -/
def ryan_distance : ℝ := 18

/-- Madison spits 20% farther than Billy -/
axiom madison_farther : madison_distance = billy_distance * 1.2

/-- Ryan spits 50% shorter than Madison -/
axiom ryan_shorter : ryan_distance = madison_distance * 0.5

theorem billy_spits_30_inches : billy_distance = 30 := by sorry

end NUMINAMATH_CALUDE_billy_spits_30_inches_l2772_277277


namespace NUMINAMATH_CALUDE_ant_return_probability_l2772_277288

/-- A modified lattice with an extra horizontal connection --/
structure ModifiedLattice :=
  (extra_connection : Bool)

/-- An ant on the modified lattice --/
structure Ant :=
  (position : ℤ × ℤ)
  (moves : ℕ)

/-- The probability of the ant returning to its starting point --/
def return_probability (l : ModifiedLattice) (a : Ant) : ℚ :=
  sorry

/-- Theorem stating the probability of returning to the starting point after 6 moves --/
theorem ant_return_probability (l : ModifiedLattice) (a : Ant) : 
  l.extra_connection = true →
  a.moves = 6 →
  return_probability l a = 1 / 64 :=
sorry

end NUMINAMATH_CALUDE_ant_return_probability_l2772_277288


namespace NUMINAMATH_CALUDE_max_min_equation_characterization_l2772_277261

theorem max_min_equation_characterization (x y : ℝ) : 
  max x (x^2) + min y (y^2) = 1 ↔ 
    (y = 1 - x^2 ∧ y ≤ 0) ∨
    (x^2 + y^2 = 1 ∧ ((x ≤ -1 ∨ x > 0) ∧ 0 < y ∧ y < 1)) ∨
    (y^2 = 1 - x ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1) :=
by sorry

end NUMINAMATH_CALUDE_max_min_equation_characterization_l2772_277261


namespace NUMINAMATH_CALUDE_spelling_bee_points_behind_l2772_277227

/-- The spelling bee problem -/
theorem spelling_bee_points_behind
  (max_points : ℕ)
  (dulce_points : ℕ)
  (val_points : ℕ)
  (opponents_points : ℕ)
  (h1 : max_points = 7)
  (h2 : dulce_points = 5)
  (h3 : val_points = 4 * (max_points + dulce_points))
  (h4 : opponents_points = 80) :
  opponents_points - (max_points + dulce_points + val_points) = 20 := by
sorry

end NUMINAMATH_CALUDE_spelling_bee_points_behind_l2772_277227


namespace NUMINAMATH_CALUDE_expression_defined_iff_l2772_277299

-- Define the set of real numbers for which the expression is defined
def valid_x : Set ℝ := {x | x ∈ Set.Ioo (-Real.sqrt 5) 1 ∪ Set.Ioo 3 (Real.sqrt 5)}

-- Define the conditions for the expression to be defined
def conditions (x : ℝ) : Prop :=
  x^2 - 4*x + 3 > 0 ∧ 5 - x^2 > 0

-- Theorem statement
theorem expression_defined_iff (x : ℝ) :
  conditions x ↔ x ∈ valid_x := by sorry

end NUMINAMATH_CALUDE_expression_defined_iff_l2772_277299


namespace NUMINAMATH_CALUDE_residue_mod_29_l2772_277296

theorem residue_mod_29 : ∃ k : ℤ, -1237 = 29 * k + 10 := by sorry

end NUMINAMATH_CALUDE_residue_mod_29_l2772_277296


namespace NUMINAMATH_CALUDE_min_value_of_f_l2772_277295

def f (x : ℕ+) : ℚ := (x.val^2 + 33) / x.val

theorem min_value_of_f :
  (∀ x : ℕ+, f x ≥ 23/2) ∧ (∃ x : ℕ+, f x = 23/2) := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2772_277295


namespace NUMINAMATH_CALUDE_min_sum_given_log_sum_l2772_277204

theorem min_sum_given_log_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : Real.log m / Real.log 3 + Real.log n / Real.log 3 = 4) : 
  m + n ≥ 18 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 
    Real.log m₀ / Real.log 3 + Real.log n₀ / Real.log 3 = 4 ∧ m₀ + n₀ = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_given_log_sum_l2772_277204


namespace NUMINAMATH_CALUDE_son_age_problem_l2772_277203

theorem son_age_problem (son_age father_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end NUMINAMATH_CALUDE_son_age_problem_l2772_277203


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2772_277264

/-- Given that z = m²(1+i) - m(3+6i) is a pure imaginary number, 
    prove that m = 3 is the only real solution. -/
theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := m^2 * (1 + Complex.I) - m * (3 + 6 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2772_277264


namespace NUMINAMATH_CALUDE_log_equation_solution_l2772_277294

theorem log_equation_solution :
  ∃ t : ℝ, t > 0 ∧ 4 * (Real.log t / Real.log 3) = Real.log (4 * t) / Real.log 3 → t = (4 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2772_277294


namespace NUMINAMATH_CALUDE_expression_equality_l2772_277289

theorem expression_equality : 
  Real.sqrt 12 + |Real.sqrt 3 - 2| + 3 - (Real.pi - 3.14)^0 = Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2772_277289


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_five_l2772_277235

/-- Given a function f(x) = 4x^3 - ax^2 - 2x + b with an extremum at x = 1, prove that a = 5 --/
theorem extremum_implies_a_equals_five (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => 4*x^3 - a*x^2 - 2*x + b
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_five_l2772_277235


namespace NUMINAMATH_CALUDE_sum_unchanged_l2772_277290

theorem sum_unchanged (a b c : ℤ) (h : a + b + c = 1281) :
  (a - 329) + (b + 401) + (c - 72) = 1281 := by
sorry

end NUMINAMATH_CALUDE_sum_unchanged_l2772_277290


namespace NUMINAMATH_CALUDE_total_harvest_l2772_277272

/-- The number of sacks of oranges harvested per day -/
def daily_harvest : ℕ := 83

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- Theorem: The total number of sacks of oranges harvested after 6 days is 498 -/
theorem total_harvest : daily_harvest * harvest_days = 498 := by
  sorry

end NUMINAMATH_CALUDE_total_harvest_l2772_277272


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_l2772_277248

theorem quadratic_rational_solutions (d : ℕ+) : 
  (∃ x : ℚ, 7 * x^2 + 13 * x + d.val = 0) ↔ d = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_l2772_277248


namespace NUMINAMATH_CALUDE_equation_solution_l2772_277266

theorem equation_solution (x y : ℝ) : 
  x / (x - 2) = (y^3 + 3*y - 2) / (y^3 + 3*y - 5) → 
  x = (2*y^3 + 6*y - 4) / 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2772_277266


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2772_277238

theorem geometric_sequence_problem (x : ℝ) : 
  x > 0 → 
  (∃ r : ℝ, r > 0 ∧ x = 40 * r ∧ (10/3) = x * r) → 
  x = (20 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2772_277238


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2772_277270

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 - a*x - b < 0}) : 
  {x : ℝ | b*x^2 - a*x - 1 > 0} = Set.Ioo (-1/2) (-1/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2772_277270


namespace NUMINAMATH_CALUDE_min_cost_45_ropes_l2772_277216

/-- Represents the cost and quantity of ropes --/
structure RopePurchase where
  costA : ℝ  -- Cost of one rope A
  costB : ℝ  -- Cost of one rope B
  quantA : ℕ -- Quantity of rope A
  quantB : ℕ -- Quantity of rope B

/-- Calculates the total cost of a rope purchase --/
def totalCost (p : RopePurchase) : ℝ :=
  p.costA * p.quantA + p.costB * p.quantB

/-- Theorem stating the minimum cost for purchasing 45 ropes --/
theorem min_cost_45_ropes (p : RopePurchase) :
  p.quantA + p.quantB = 45 →
  10 * 10 + 5 * 15 = 175 →
  15 * 10 + 10 * 15 = 300 →
  548 ≤ totalCost p →
  totalCost p ≤ 560 →
  ∃ (q : RopePurchase), 
    q.costA = 10 ∧ 
    q.costB = 15 ∧ 
    q.quantA = 25 ∧ 
    q.quantB = 20 ∧ 
    totalCost q = 550 ∧ 
    totalCost q ≤ totalCost p :=
by
  sorry


end NUMINAMATH_CALUDE_min_cost_45_ropes_l2772_277216


namespace NUMINAMATH_CALUDE_one_slice_remains_l2772_277263

/-- Calculates the number of slices of bread remaining after eating and making toast. -/
def remaining_slices (initial_slices : ℕ) (eaten_twice : ℕ) (slices_per_toast : ℕ) (toast_made : ℕ) : ℕ :=
  initial_slices - (2 * eaten_twice) - (slices_per_toast * toast_made)

/-- Theorem stating that given the specific conditions, 1 slice of bread remains. -/
theorem one_slice_remains : remaining_slices 27 3 2 10 = 1 := by
  sorry

#eval remaining_slices 27 3 2 10

end NUMINAMATH_CALUDE_one_slice_remains_l2772_277263


namespace NUMINAMATH_CALUDE_min_operations_to_check_square_l2772_277200

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define an operation (either measurement or comparison)
inductive Operation
  | Measure : Point → Point → Operation
  | Compare : ℝ → ℝ → Operation

-- Define a function to check if a quadrilateral is a square
def isSquare (q : Quadrilateral) : Prop := sorry

-- Define a function that returns the list of operations needed to check if a quadrilateral is a square
def operationsToCheckSquare (q : Quadrilateral) : List Operation := sorry

-- Theorem statement
theorem min_operations_to_check_square (q : Quadrilateral) :
  (isSquare q ↔ operationsToCheckSquare q = [
    Operation.Measure q.A q.B,
    Operation.Measure q.B q.C,
    Operation.Measure q.C q.D,
    Operation.Measure q.D q.A,
    Operation.Measure q.A q.C,
    Operation.Measure q.B q.D,
    Operation.Compare (q.A.x - q.B.x) (q.B.x - q.C.x),
    Operation.Compare (q.B.x - q.C.x) (q.C.x - q.D.x),
    Operation.Compare (q.C.x - q.D.x) (q.D.x - q.A.x),
    Operation.Compare (q.A.x - q.C.x) (q.B.x - q.D.x)
  ]) :=
sorry

end NUMINAMATH_CALUDE_min_operations_to_check_square_l2772_277200


namespace NUMINAMATH_CALUDE_expression_evaluation_l2772_277267

theorem expression_evaluation (x y z : ℝ) :
  let P := x + y + z
  let Q := x - y - z
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (x^2 - y^2 - 2*y*z - z^2) / (x*(y + z)) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2772_277267


namespace NUMINAMATH_CALUDE_bicycle_discount_proof_l2772_277218

theorem bicycle_discount_proof (original_price : ℝ) 
  (discount1 discount2 discount3 : ℝ) : 
  original_price = 200 →
  discount1 = 0.60 →
  discount2 = 0.20 →
  discount3 = 0.10 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 57.60 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_discount_proof_l2772_277218


namespace NUMINAMATH_CALUDE_sequence_a_property_sequence_a_positive_sequence_a_first_two_terms_sequence_a_bounds_sequence_a_decreasing_l2772_277249

def sequence_a (n : ℕ+) : ℝ := sorry

theorem sequence_a_property (n : ℕ+) : 
  sequence_a n + n * sequence_a n - 1 = 0 := sorry

theorem sequence_a_positive (n : ℕ+) : 
  sequence_a n > 0 := sorry

theorem sequence_a_first_two_terms : 
  sequence_a 1 = 1/2 ∧ sequence_a 2 = 1/4 := sorry

theorem sequence_a_bounds (n : ℕ+) : 
  0 < sequence_a n ∧ sequence_a n < 1 := sorry

theorem sequence_a_decreasing (n : ℕ+) : 
  sequence_a n > sequence_a (n + 1) := sorry

end NUMINAMATH_CALUDE_sequence_a_property_sequence_a_positive_sequence_a_first_two_terms_sequence_a_bounds_sequence_a_decreasing_l2772_277249


namespace NUMINAMATH_CALUDE_bookstore_sales_ratio_l2772_277274

theorem bookstore_sales_ratio :
  -- Initial conditions
  let initial_inventory : ℕ := 743
  let saturday_instore : ℕ := 37
  let saturday_online : ℕ := 128
  let sunday_online_increase : ℕ := 34
  let shipment : ℕ := 160
  let final_inventory : ℕ := 502

  -- Define Sunday in-store sales
  let sunday_instore : ℕ := initial_inventory - final_inventory + shipment - 
    (saturday_instore + saturday_online + sunday_online_increase)

  -- Theorem statement
  (sunday_instore : ℚ) / (saturday_instore : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_ratio_l2772_277274


namespace NUMINAMATH_CALUDE_max_perfect_squares_l2772_277228

/-- The sequence (a_n) defined recursively -/
def a : ℕ → ℕ → ℕ
  | m, 0 => m
  | m, n + 1 => (a m n)^5 + 487

/-- Proposition: m = 9 is the unique positive integer that maximizes perfect squares in the sequence -/
theorem max_perfect_squares (m : ℕ) : m > 0 → (∀ k : ℕ, k > 0 → (∀ n : ℕ, ∃ i : ℕ, i ≤ n ∧ ∃ j : ℕ, a k i = j^2) → 
  (∀ n : ℕ, ∃ i : ℕ, i ≤ n ∧ ∃ j : ℕ, a m i = j^2)) → m = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_perfect_squares_l2772_277228


namespace NUMINAMATH_CALUDE_total_present_age_l2772_277278

-- Define the present ages of p and q
def p : ℕ := sorry
def q : ℕ := sorry

-- Define the conditions
axiom age_relation : p - 12 = (q - 12) / 2
axiom present_ratio : p * 4 = q * 3

-- Theorem to prove
theorem total_present_age : p + q = 42 := by sorry

end NUMINAMATH_CALUDE_total_present_age_l2772_277278


namespace NUMINAMATH_CALUDE_power_sum_value_l2772_277225

theorem power_sum_value (a m n : ℝ) (h1 : a^m = 5) (h2 : a^n = 3) :
  a^(m + n) = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_value_l2772_277225


namespace NUMINAMATH_CALUDE_student_number_problem_l2772_277297

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 102 → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2772_277297


namespace NUMINAMATH_CALUDE_n_sum_of_squares_l2772_277291

theorem n_sum_of_squares (n : ℕ) (h1 : n > 2) 
  (h2 : ∃ (x : ℕ), n^2 = (x + 1)^3 - x^3) : 
  (∃ (a b : ℕ), n = a^2 + b^2) ∧ 
  (∃ (m : ℕ), m > 2 ∧ (∃ (y : ℕ), m^2 = (y + 1)^3 - y^3) ∧ (∃ (c d : ℕ), m = c^2 + d^2)) :=
by sorry

end NUMINAMATH_CALUDE_n_sum_of_squares_l2772_277291


namespace NUMINAMATH_CALUDE_factorial_ratio_equality_l2772_277214

theorem factorial_ratio_equality : (Nat.factorial 9)^2 / (Nat.factorial 4 * Nat.factorial 5) = 45760000 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equality_l2772_277214


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2772_277230

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, y < 3 * y - 15 → y ≥ 8 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2772_277230


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2772_277243

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 7 ∧ 
  a 5 = a 2 + 6

/-- The general term formula for the arithmetic sequence -/
def GeneralTerm (n : ℕ) : ℝ := 2 * n + 1

/-- Theorem stating that the general term formula is correct for the given arithmetic sequence -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  ∀ n : ℕ, a n = GeneralTerm n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2772_277243


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l2772_277276

theorem factorization_of_quadratic (x : ℝ) : x^2 - 3*x = x*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l2772_277276


namespace NUMINAMATH_CALUDE_quadratic_sum_of_squares_l2772_277213

theorem quadratic_sum_of_squares (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  (∃ (x y z : ℝ),
    (x^2 + a*x + b = 0 ∧ y^2 + b*y + c = 0 ∧ x = y) ∧
    (y^2 + b*y + c = 0 ∧ z^2 + c*z + a = 0 ∧ y = z) ∧
    (z^2 + c*z + a = 0 ∧ x^2 + a*x + b = 0 ∧ z = x)) →
  a^2 + b^2 + c^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_squares_l2772_277213


namespace NUMINAMATH_CALUDE_find_m_l2772_277244

/-- Given two functions f and g, prove that if 3f(5) = g(5), then m = 10 -/
theorem find_m (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x + m
  let g : ℝ → ℝ := λ x ↦ x^2 - 3*x + 5*m
  3 * f 5 = g 5 → m = 10 := by
sorry

end NUMINAMATH_CALUDE_find_m_l2772_277244


namespace NUMINAMATH_CALUDE_joes_gym_people_l2772_277283

/-- The number of people in Joe's Gym during Bethany's shift --/
theorem joes_gym_people (W A : ℕ) : 
  W + A + 5 + 2 - 3 - 4 + 2 = 20 → W + A = 18 := by
  sorry

#check joes_gym_people

end NUMINAMATH_CALUDE_joes_gym_people_l2772_277283


namespace NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l2772_277281

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2, 3}

-- Define set N
def N : Set ℕ := {2, 3, 4}

-- State the theorem
theorem complement_intersection_equals_specific_set :
  (M ∩ N)ᶜ = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l2772_277281


namespace NUMINAMATH_CALUDE_four_children_probability_l2772_277285

theorem four_children_probability (p_boy p_girl : ℚ) : 
  p_boy = 2/3 → 
  p_girl = 1/3 → 
  (1 - (p_boy^4 + p_girl^4)) = 64/81 := by
sorry

end NUMINAMATH_CALUDE_four_children_probability_l2772_277285


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l2772_277269

theorem arithmetic_sequence_sum_divisibility :
  ∀ (a d : ℕ+), 
  ∃ (k : ℕ), (15 : ℕ) * (a + 7 * d) = k := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l2772_277269


namespace NUMINAMATH_CALUDE_senior_discount_percentage_l2772_277222

def original_cost : ℚ := 7.5
def coupon_discount : ℚ := 2.5
def final_payment : ℚ := 4

def cost_after_coupon : ℚ := original_cost - coupon_discount
def senior_discount_amount : ℚ := cost_after_coupon - final_payment

theorem senior_discount_percentage :
  (senior_discount_amount / cost_after_coupon) * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_senior_discount_percentage_l2772_277222


namespace NUMINAMATH_CALUDE_product_lmn_equals_one_l2772_277286

theorem product_lmn_equals_one 
  (p q r l m n : ℂ)
  (distinct_pqr : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (distinct_lmn : l ≠ m ∧ m ≠ n ∧ l ≠ n)
  (nonzero_lmn : l ≠ 0 ∧ m ≠ 0 ∧ n ≠ 0)
  (eq1 : p / (1 - q) = l)
  (eq2 : q / (1 - r) = m)
  (eq3 : r / (1 - p) = n) :
  l * m * n = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_lmn_equals_one_l2772_277286


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2772_277282

/-- Triangle ABC with given conditions -/
structure TriangleABC where
  -- Vertex A coordinates
  A : ℝ × ℝ
  -- Equation of line containing median CM on side AB
  median_CM_eq : ℝ → ℝ → ℝ
  -- Equation of line containing altitude BH on side AC
  altitude_BH_eq : ℝ → ℝ → ℝ
  -- Conditions
  h_A : A = (5, 1)
  h_median_CM : ∀ x y, median_CM_eq x y = 2*x - y - 5
  h_altitude_BH : ∀ x y, altitude_BH_eq x y = x - 2*y - 5

/-- Main theorem about Triangle ABC -/
theorem triangle_abc_properties (t : TriangleABC) :
  -- 1. Coordinates of vertex C
  ∃ C : ℝ × ℝ, C = (4, 3) ∧
  -- 2. Length of AC
  Real.sqrt ((C.1 - t.A.1)^2 + (C.2 - t.A.2)^2) = Real.sqrt 5 ∧
  -- 3. Equation of line BC
  ∃ BC_eq : ℝ → ℝ → ℝ, (∀ x y, BC_eq x y = 6*x - 5*y - 9) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2772_277282


namespace NUMINAMATH_CALUDE_fare_ratio_proof_l2772_277212

theorem fare_ratio_proof (passenger_ratio : ℚ) (total_amount : ℕ) (second_class_amount : ℕ) :
  passenger_ratio = 1 / 50 →
  total_amount = 1325 →
  second_class_amount = 1250 →
  ∃ (first_class_fare second_class_fare : ℕ),
    first_class_fare / second_class_fare = 3 :=
by sorry

end NUMINAMATH_CALUDE_fare_ratio_proof_l2772_277212


namespace NUMINAMATH_CALUDE_exact_one_second_class_probability_l2772_277208

/-- The probability of selecting exactly one second-class product when randomly
    selecting three products from a batch of 100 products containing 90 first-class
    and 10 second-class products. -/
theorem exact_one_second_class_probability
  (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ)
  (h_total : total = 100)
  (h_first : first_class = 90)
  (h_second : second_class = 10)
  (h_selected : selected = 3)
  (h_sum : first_class + second_class = total) :
  (Nat.choose first_class 2 * Nat.choose second_class 1) / Nat.choose total selected = 267 / 1078 :=
sorry

end NUMINAMATH_CALUDE_exact_one_second_class_probability_l2772_277208


namespace NUMINAMATH_CALUDE_cubic_equation_one_root_implies_a_range_l2772_277292

theorem cubic_equation_one_root_implies_a_range (a : ℝ) : 
  (∃! x : ℝ, x^3 + (1-a)*x^2 - 2*a*x + a^2 = 0) → a < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_one_root_implies_a_range_l2772_277292


namespace NUMINAMATH_CALUDE_factorial_equation_l2772_277252

theorem factorial_equation (x : ℕ) : 6 * 8 * 3 * x = Nat.factorial 10 → x = 75600 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l2772_277252


namespace NUMINAMATH_CALUDE_factorial_ratio_l2772_277226

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 47 = 117600 := by sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2772_277226


namespace NUMINAMATH_CALUDE_min_distinct_values_l2772_277245

theorem min_distinct_values (total : ℕ) (mode_count : ℕ) (h1 : total = 2023) (h2 : mode_count = 15) :
  ∃ (distinct : ℕ), distinct = 145 ∧ 
  (∀ d : ℕ, d < 145 → 
    ¬∃ (l : List ℕ), l.length = total ∧ 
    (∃! x : ℕ, x ∈ l ∧ l.count x = mode_count) ∧
    l.toFinset.card = d) :=
by sorry

end NUMINAMATH_CALUDE_min_distinct_values_l2772_277245


namespace NUMINAMATH_CALUDE_volume_of_sphere_wedge_l2772_277247

/-- The volume of a wedge from a sphere -/
theorem volume_of_sphere_wedge (c : ℝ) (n : ℕ) (h1 : c = 18 * Real.pi) (h2 : n = 6) :
  (4 / 3 * Real.pi * (c / (2 * Real.pi))^3) / n = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_volume_of_sphere_wedge_l2772_277247


namespace NUMINAMATH_CALUDE_unique_provider_choices_l2772_277239

theorem unique_provider_choices (n m : ℕ) (hn : n = 23) (hm : m = 4) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) = 213840 := by
  sorry

end NUMINAMATH_CALUDE_unique_provider_choices_l2772_277239


namespace NUMINAMATH_CALUDE_zachary_purchase_l2772_277256

/-- The cost of items at a store -/
structure StorePrices where
  pencil : ℕ
  notebook : ℕ
  eraser : ℕ

/-- The conditions of the problem -/
def store_conditions (p : StorePrices) : Prop :=
  p.pencil + p.notebook = 80 ∧
  p.notebook + p.eraser = 85 ∧
  3 * p.pencil + 3 * p.notebook + 3 * p.eraser = 315

/-- The theorem to prove -/
theorem zachary_purchase (p : StorePrices) (h : store_conditions p) : 
  p.pencil + p.eraser = 45 := by
  sorry

end NUMINAMATH_CALUDE_zachary_purchase_l2772_277256


namespace NUMINAMATH_CALUDE_solve_sqrt_equation_l2772_277233

theorem solve_sqrt_equation (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_sqrt_equation_l2772_277233


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2772_277257

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a5 (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2772_277257
