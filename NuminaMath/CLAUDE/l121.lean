import Mathlib

namespace NUMINAMATH_CALUDE_salt_merchant_problem_l121_12146

/-- The salt merchant problem -/
theorem salt_merchant_problem (x y : ℝ) (a : ℝ) 
  (h1 : a * (y - x) = 100)  -- Profit from first transaction
  (h2 : a * y * (y / x - 1) = 120)  -- Profit from second transaction
  (h3 : x > 0)  -- Price in Tver is positive
  (h4 : y > x)  -- Price in Moscow is higher than in Tver
  : a * x = 500 := by
  sorry

end NUMINAMATH_CALUDE_salt_merchant_problem_l121_12146


namespace NUMINAMATH_CALUDE_solution_verification_l121_12124

-- Define the differential equation
def differential_equation (x : ℝ) (y : ℝ → ℝ) : Prop :=
  x * (deriv y x) = y x - 1

-- Define the first function
def f₁ (x : ℝ) : ℝ := 3 * x + 1

-- Define the second function (C is a real constant)
def f₂ (C : ℝ) (x : ℝ) : ℝ := C * x + 1

-- Theorem statement
theorem solution_verification :
  (∀ x, x ≠ 0 → differential_equation x f₁) ∧
  (∀ C, ∀ x, x ≠ 0 → differential_equation x (f₂ C)) :=
sorry

end NUMINAMATH_CALUDE_solution_verification_l121_12124


namespace NUMINAMATH_CALUDE_yellow_balls_count_l121_12114

/-- Represents a bag containing red and yellow balls -/
structure BallBag where
  redBalls : ℕ
  yellowBalls : ℕ

/-- Calculates the probability of drawing a red ball from the bag -/
def redProbability (bag : BallBag) : ℚ :=
  bag.redBalls / (bag.redBalls + bag.yellowBalls)

/-- Theorem: Given the conditions, the number of yellow balls is 25 -/
theorem yellow_balls_count (bag : BallBag) 
  (h1 : bag.redBalls = 10)
  (h2 : redProbability bag = 2/7) :
  bag.yellowBalls = 25 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l121_12114


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l121_12165

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the equation of the common chord
def common_chord (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord x y :=
sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l121_12165


namespace NUMINAMATH_CALUDE_line_intersects_circle_l121_12137

/-- A circle with center O and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Relationship between a line and a circle -/
inductive LineCircleRelation
  | Disjoint
  | Tangent
  | Intersect

theorem line_intersects_circle (O : Circle) (l : Line) :
  O.radius = 4 →
  distancePointToLine O.center l = 3 →
  LineCircleRelation.Intersect = 
    match O.radius, distancePointToLine O.center l with
    | r, d => if r > d then LineCircleRelation.Intersect
              else if r = d then LineCircleRelation.Tangent
              else LineCircleRelation.Disjoint :=
by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l121_12137


namespace NUMINAMATH_CALUDE_nail_color_percentage_difference_l121_12109

/-- Given a total of 20 nails, with 6 painted purple and 8 painted blue, 
    prove that the difference in percentage points between blue and striped nails is 10. -/
theorem nail_color_percentage_difference : 
  let total_nails : ℕ := 20
  let purple_nails : ℕ := 6
  let blue_nails : ℕ := 8
  let striped_nails : ℕ := total_nails - purple_nails - blue_nails
  let blue_percentage : ℚ := blue_nails * 100 / total_nails
  let striped_percentage : ℚ := striped_nails * 100 / total_nails
  blue_percentage - striped_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_nail_color_percentage_difference_l121_12109


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l121_12102

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2) ∧
  (∃ a b : ℝ, a + b > 2 ∧ ¬(a > 1 ∧ b > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l121_12102


namespace NUMINAMATH_CALUDE_base8_addition_l121_12171

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

/-- Addition in base-8 --/
def add_base8 (a b c : ℕ) : ℕ :=
  base10_to_base8 (base8_to_base10 a + base8_to_base10 b + base8_to_base10 c)

theorem base8_addition :
  add_base8 246 573 62 = 1123 := by sorry

end NUMINAMATH_CALUDE_base8_addition_l121_12171


namespace NUMINAMATH_CALUDE_layla_fish_food_total_l121_12141

/-- The total amount of food Layla needs to give her fish -/
def total_fish_food (goldfish_count : ℕ) (goldfish_food : ℚ) 
                    (swordtail_count : ℕ) (swordtail_food : ℚ) 
                    (guppy_count : ℕ) (guppy_food : ℚ) : ℚ :=
  goldfish_count * goldfish_food + swordtail_count * swordtail_food + guppy_count * guppy_food

/-- Theorem stating the total amount of food Layla needs to give her fish -/
theorem layla_fish_food_total : 
  total_fish_food 2 1 3 2 8 (1/2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_layla_fish_food_total_l121_12141


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l121_12149

/-- The quadratic equation x^2 + 2x√3 + 3 = 0 has real and equal roots given that its discriminant is zero -/
theorem quadratic_roots_real_and_equal :
  let a : ℝ := 1
  let b : ℝ := 2 * Real.sqrt 3
  let c : ℝ := 3
  let discriminant := b^2 - 4*a*c
  discriminant = 0 →
  ∃! x : ℝ, a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l121_12149


namespace NUMINAMATH_CALUDE_circle_radius_c_value_l121_12117

theorem circle_radius_c_value (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 10*x + y^2 + 2*y + c = 0 ↔ (x+5)^2 + (y+1)^2 = 25) → 
  c = 51 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_c_value_l121_12117


namespace NUMINAMATH_CALUDE_sum_difference_theorem_l121_12101

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def jo_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def lisa_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => round_to_nearest_five (i + 1))

theorem sum_difference_theorem :
  jo_sum 60 - lisa_sum 60 = 240 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_theorem_l121_12101


namespace NUMINAMATH_CALUDE_triangle_properties_l121_12147

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = Real.sqrt 3) 
  (h2 : t.C = 5 * Real.pi / 6) 
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2) 
  (h4 : t.B = Real.pi / 3) : 
  (t.c = Real.sqrt 13) ∧ 
  (-Real.sqrt 3 < 2 * t.c - t.a) ∧ 
  (2 * t.c - t.a < 2 * Real.sqrt 3) := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l121_12147


namespace NUMINAMATH_CALUDE_discount_calculation_l121_12187

/-- Calculates the total discount percentage given initial, member, and special promotion discounts -/
def total_discount (initial_discount : ℝ) (member_discount : ℝ) (special_discount : ℝ) : ℝ :=
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_member := remaining_after_initial * (1 - member_discount)
  let final_remaining := remaining_after_member * (1 - special_discount)
  (1 - final_remaining) * 100

/-- Theorem stating that the total discount is 65.8% given the specific discounts -/
theorem discount_calculation :
  total_discount 0.6 0.1 0.05 = 65.8 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l121_12187


namespace NUMINAMATH_CALUDE_lenas_collage_glue_drops_l121_12173

/-- Calculates the total number of glue drops needed for a collage -/
def totalGlueDrops (clippings : List Nat) (gluePerClipping : Nat) : Nat :=
  (clippings.sum) * gluePerClipping

/-- Proves that the total number of glue drops for Lena's collage is 240 -/
theorem lenas_collage_glue_drops :
  let clippings := [4, 7, 5, 3, 5, 8, 2, 6]
  let gluePerClipping := 6
  totalGlueDrops clippings gluePerClipping = 240 := by
  sorry

#eval totalGlueDrops [4, 7, 5, 3, 5, 8, 2, 6] 6

end NUMINAMATH_CALUDE_lenas_collage_glue_drops_l121_12173


namespace NUMINAMATH_CALUDE_rectangle_area_l121_12167

/-- Given a rectangle where the length is five times the width and the perimeter is 180 cm,
    prove that its area is 1125 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 5 * w
  2 * l + 2 * w = 180 → l * w = 1125 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l121_12167


namespace NUMINAMATH_CALUDE_brads_cookies_brads_cookies_solution_l121_12152

theorem brads_cookies (total_cookies : ℕ) (greg_ate : ℕ) (leftover : ℕ) : ℕ :=
  let total_halves := total_cookies * 2
  let after_greg := total_halves - greg_ate
  after_greg - leftover

theorem brads_cookies_solution :
  brads_cookies 14 4 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_brads_cookies_brads_cookies_solution_l121_12152


namespace NUMINAMATH_CALUDE_floor_difference_l121_12120

theorem floor_difference : ⌊(-2.7 : ℝ)⌋ - ⌊(4.5 : ℝ)⌋ = -7 := by
  sorry

end NUMINAMATH_CALUDE_floor_difference_l121_12120


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l121_12159

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2*p.2, 2*p.1 - p.2)

-- Theorem statement
theorem preimage_of_3_1 :
  ∃ (p : ℝ × ℝ), f p = (3, 1) ∧ p = (1, 1) :=
sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l121_12159


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l121_12195

/-- A function that checks if a natural number consists only of 2's and 7's in its decimal representation -/
def only_2_and_7 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 7

/-- A function that checks if a natural number has at least one 2 and one 7 in its decimal representation -/
def has_2_and_7 (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 7 ∈ n.digits 10

/-- The theorem stating the properties of the smallest number satisfying the given conditions -/
theorem smallest_number_with_conditions (m : ℕ) : 
  (∀ n : ℕ, n < m → ¬(n % 5 = 0 ∧ n % 8 = 0 ∧ only_2_and_7 n ∧ has_2_and_7 n)) →
  m % 5 = 0 ∧ m % 8 = 0 ∧ only_2_and_7 m ∧ has_2_and_7 m →
  m % 10000 = 7272 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l121_12195


namespace NUMINAMATH_CALUDE_max_min_xy_constraint_l121_12198

theorem max_min_xy_constraint (x y : ℝ) : 
  x^2 + x*y + y^2 ≤ 1 → 
  (∃ (max min : ℝ), 
    (∀ z, x - y + 2*x*y ≤ z → z ≤ max) ∧ 
    (∀ w, min ≤ w → w ≤ x - y + 2*x*y) ∧
    max = 25/24 ∧ min = -4) := by
  sorry

end NUMINAMATH_CALUDE_max_min_xy_constraint_l121_12198


namespace NUMINAMATH_CALUDE_line_BC_equation_l121_12197

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem conditions
def problem_conditions (triangle : Triangle) (altitude1 altitude2 : Line) : Prop :=
  -- First altitude: x + y = 0
  altitude1.a = 1 ∧ altitude1.b = 1 ∧ altitude1.c = 0 ∧
  -- Second altitude: 2x - 3y + 1 = 0
  altitude2.a = 2 ∧ altitude2.b = -3 ∧ altitude2.c = 1 ∧
  -- Point A is (1, 2)
  triangle.A = (1, 2)

-- Theorem statement
theorem line_BC_equation (triangle : Triangle) (altitude1 altitude2 : Line) :
  problem_conditions triangle altitude1 altitude2 →
  ∃ (line_BC : Line), line_BC.a = 2 ∧ line_BC.b = 3 ∧ line_BC.c = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_line_BC_equation_l121_12197


namespace NUMINAMATH_CALUDE_hallie_earnings_l121_12108

/-- Calculates the total earnings for a waitress over three days given her hourly wage, hours worked, and tips for each day. -/
def total_earnings (hourly_wage : ℝ) (hours_day1 hours_day2 hours_day3 : ℝ) (tips_day1 tips_day2 tips_day3 : ℝ) : ℝ :=
  (hourly_wage * hours_day1 + tips_day1) +
  (hourly_wage * hours_day2 + tips_day2) +
  (hourly_wage * hours_day3 + tips_day3)

/-- Theorem stating that Hallie's total earnings over three days equal $240 given her work schedule and tips. -/
theorem hallie_earnings :
  total_earnings 10 7 5 7 18 12 20 = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_hallie_earnings_l121_12108


namespace NUMINAMATH_CALUDE_favorite_song_probability_l121_12158

/-- Represents a digital music player with a collection of songs. -/
structure MusicPlayer where
  numSongs : Nat
  shortestSongDuration : Nat
  durationIncrement : Nat
  favoriteSongDuration : Nat
  playbackDuration : Nat

/-- Calculates the probability of not hearing the favorite song in full 
    within the given playback duration. -/
def probabilityNoFavoriteSong (player : MusicPlayer) : Rat :=
  sorry

/-- Theorem stating the probability of not hearing the favorite song in full
    for the specific music player configuration. -/
theorem favorite_song_probability (player : MusicPlayer) 
  (h1 : player.numSongs = 12)
  (h2 : player.shortestSongDuration = 40)
  (h3 : player.durationIncrement = 40)
  (h4 : player.favoriteSongDuration = 300)
  (h5 : player.playbackDuration = 360) :
  probabilityNoFavoriteSong player = 43 / 48 := by
  sorry

end NUMINAMATH_CALUDE_favorite_song_probability_l121_12158


namespace NUMINAMATH_CALUDE_total_trophies_in_three_years_l121_12119

theorem total_trophies_in_three_years :
  let michael_current_trophies : ℕ := 30
  let michael_trophy_increase : ℕ := 100
  let jack_trophy_multiplier : ℕ := 10
  let michael_future_trophies : ℕ := michael_current_trophies + michael_trophy_increase
  let jack_future_trophies : ℕ := jack_trophy_multiplier * michael_current_trophies
  michael_future_trophies + jack_future_trophies = 430 := by
sorry

end NUMINAMATH_CALUDE_total_trophies_in_three_years_l121_12119


namespace NUMINAMATH_CALUDE_min_value_inequality_l121_12183

theorem min_value_inequality (a : ℝ) (h : a > 1) :
  a + 2 / (a - 1) ≥ 1 + 2 * Real.sqrt 2 ∧
  ∃ a₀ > 1, a₀ + 2 / (a₀ - 1) = 1 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l121_12183


namespace NUMINAMATH_CALUDE_first_friend_cookies_l121_12129

theorem first_friend_cookies (initial : ℕ) (eaten : ℕ) (brother : ℕ) (second : ℕ) (third : ℕ) (remaining : ℕ) : 
  initial = 22 → 
  eaten = 2 → 
  brother = 1 → 
  second = 5 → 
  third = 5 → 
  remaining = 6 → 
  initial - eaten - brother - second - third - remaining = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_friend_cookies_l121_12129


namespace NUMINAMATH_CALUDE_chip_paper_usage_l121_12131

/-- Calculates the number of packs of paper Chip will use during the semester --/
def calculate_packs_of_paper (pages_per_pack : ℕ) (regular_weeks : ℕ) (short_weeks : ℕ) 
  (pages_per_regular_week : ℕ) (pages_per_short_week : ℕ) : ℕ :=
  let total_pages := regular_weeks * pages_per_regular_week + short_weeks * pages_per_short_week
  ((total_pages + pages_per_pack - 1) / pages_per_pack : ℕ)

/-- Theorem stating that Chip will use 6 packs of paper during the semester --/
theorem chip_paper_usage : 
  calculate_packs_of_paper 100 13 3 40 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_chip_paper_usage_l121_12131


namespace NUMINAMATH_CALUDE_janet_income_difference_l121_12169

/-- Calculates the difference in monthly income between freelancing and current job for Janet --/
theorem janet_income_difference :
  let hours_per_week : ℕ := 40
  let weeks_per_month : ℕ := 4
  let current_hourly_rate : ℚ := 30
  let freelance_hourly_rate : ℚ := 40
  let extra_fica_per_week : ℚ := 25
  let healthcare_premium_per_month : ℚ := 400

  let current_monthly_income : ℚ := hours_per_week * weeks_per_month * current_hourly_rate
  let freelance_gross_monthly_income : ℚ := hours_per_week * weeks_per_month * freelance_hourly_rate
  let additional_monthly_costs : ℚ := extra_fica_per_week * weeks_per_month + healthcare_premium_per_month
  let freelance_net_monthly_income : ℚ := freelance_gross_monthly_income - additional_monthly_costs

  freelance_net_monthly_income - current_monthly_income = 1100 := by
  sorry

end NUMINAMATH_CALUDE_janet_income_difference_l121_12169


namespace NUMINAMATH_CALUDE_largest_y_coordinate_l121_12162

theorem largest_y_coordinate (x y : ℝ) :
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_l121_12162


namespace NUMINAMATH_CALUDE_worker_count_l121_12110

/-- Represents the number of workers in the factory -/
def num_workers : ℕ := sorry

/-- The initial average monthly salary of workers and supervisor -/
def initial_average_salary : ℚ := 430

/-- The initial supervisor's monthly salary -/
def initial_supervisor_salary : ℚ := 870

/-- The new average monthly salary after supervisor change -/
def new_average_salary : ℚ := 410

/-- The new supervisor's monthly salary -/
def new_supervisor_salary : ℚ := 690

/-- The total number of people (workers + new supervisor) after the change -/
def total_people : ℕ := 9

theorem worker_count :
  (num_workers + 1) * initial_average_salary - initial_supervisor_salary =
  total_people * new_average_salary - new_supervisor_salary ∧
  num_workers = 8 := by sorry

end NUMINAMATH_CALUDE_worker_count_l121_12110


namespace NUMINAMATH_CALUDE_park_area_l121_12185

/-- Calculates the area of a rectangular park given cycling conditions --/
theorem park_area (speed : ℝ) (time : ℝ) : 
  speed = 12 → -- speed in km/hr
  time = 8 / 60 → -- time in hours (8 minutes)
  ∃ (length breadth : ℝ),
    length > 0 ∧ 
    breadth > 0 ∧
    length = breadth / 3 ∧ -- ratio condition
    2 * (length + breadth) = speed * time * 1000 ∧ -- perimeter in meters
    length * breadth = 120000 -- area in square meters
  := by sorry

end NUMINAMATH_CALUDE_park_area_l121_12185


namespace NUMINAMATH_CALUDE_solve_batting_problem_l121_12191

def batting_problem (pitches_per_token : ℕ) (macy_tokens : ℕ) (piper_tokens : ℕ) 
  (macy_hits : ℕ) (total_misses : ℕ) : Prop :=
  let total_pitches := pitches_per_token * (macy_tokens + piper_tokens)
  let total_hits := total_pitches - total_misses
  let piper_hits := total_hits - macy_hits
  piper_hits = 55

theorem solve_batting_problem :
  batting_problem 15 11 17 50 315 := by
  sorry

end NUMINAMATH_CALUDE_solve_batting_problem_l121_12191


namespace NUMINAMATH_CALUDE_geese_to_ducks_ratio_l121_12107

theorem geese_to_ducks_ratio (initial_ducks : ℕ) (arriving_ducks : ℕ) (leaving_geese : ℕ) (initial_geese : ℕ) :
  initial_ducks = 25 →
  arriving_ducks = 4 →
  leaving_geese = 10 →
  initial_geese - leaving_geese = initial_ducks + arriving_ducks + 1 →
  (initial_geese : ℚ) / initial_ducks = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_geese_to_ducks_ratio_l121_12107


namespace NUMINAMATH_CALUDE_number_division_remainders_l121_12184

theorem number_division_remainders (N : ℤ) (h : N % 1554 = 131) : 
  (N % 37 = 20) ∧ (N % 73 = 58) := by
  sorry

end NUMINAMATH_CALUDE_number_division_remainders_l121_12184


namespace NUMINAMATH_CALUDE_percentage_of_160_l121_12143

theorem percentage_of_160 : (3 / 8 : ℚ) / 100 * 160 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_percentage_of_160_l121_12143


namespace NUMINAMATH_CALUDE_work_completion_time_increase_l121_12130

theorem work_completion_time_increase 
  (initial_men : ℕ) 
  (initial_days : ℕ) 
  (removed_men : ℕ) 
  (h1 : initial_men = 100) 
  (h2 : initial_days = 20) 
  (h3 : removed_men = 50) : 
  (initial_men * initial_days) / (initial_men - removed_men) - initial_days = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_increase_l121_12130


namespace NUMINAMATH_CALUDE_inscribed_polygon_has_larger_area_l121_12134

/-- A polygon is a set of points in the plane --/
def Polygon : Type := Set (ℝ × ℝ)

/-- A convex polygon is a polygon where all interior angles are less than or equal to 180 degrees --/
def ConvexPolygon (P : Polygon) : Prop := sorry

/-- A polygon is inscribed in a circle if all its vertices lie on the circle's circumference --/
def InscribedInCircle (P : Polygon) : Prop := sorry

/-- The area of a polygon --/
def PolygonArea (P : Polygon) : ℝ := sorry

/-- The side lengths of a polygon --/
def SideLengths (P : Polygon) : List ℝ := sorry

/-- Two polygons have the same side lengths --/
def SameSideLengths (P Q : Polygon) : Prop :=
  SideLengths P = SideLengths Q

theorem inscribed_polygon_has_larger_area 
  (N M : Polygon) 
  (h1 : ConvexPolygon N) 
  (h2 : ConvexPolygon M) 
  (h3 : InscribedInCircle N) 
  (h4 : SameSideLengths N M) :
  PolygonArea N > PolygonArea M :=
sorry

end NUMINAMATH_CALUDE_inscribed_polygon_has_larger_area_l121_12134


namespace NUMINAMATH_CALUDE_line_equation_l121_12188

/-- Circle with center (3, 5) and radius √5 -/
def C : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 - 5)^2 = 5}

/-- Line passing through the center of circle C -/
structure Line where
  k : ℝ
  eq : ℝ × ℝ → Prop := fun p => p.2 - 5 = k * (p.1 - 3)

/-- Point where the line intersects the y-axis -/
def P (l : Line) : ℝ × ℝ := (0, 5 - 3 * l.k)

/-- Intersection points of the line and the circle -/
def intersectionPoints (l : Line) : Set (ℝ × ℝ) :=
  {p ∈ C | l.eq p}

/-- A is the midpoint of PB -/
def isMidpoint (A B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  A.1 = (P.1 + B.1) / 2 ∧ A.2 = (P.2 + B.2) / 2

theorem line_equation (l : Line) (A B : ℝ × ℝ) 
  (hA : A ∈ intersectionPoints l) (hB : B ∈ intersectionPoints l)
  (hMid : isMidpoint A B (P l)) :
  (l.k = 2 ∧ l.eq = fun p => 2 * p.1 - p.2 - 1 = 0) ∨
  (l.k = -2 ∧ l.eq = fun p => 2 * p.1 + p.2 + 11 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l121_12188


namespace NUMINAMATH_CALUDE_money_duration_l121_12121

def mowing_earnings : ℕ := 5
def weed_eating_earnings : ℕ := 58
def weekly_spending : ℕ := 7

theorem money_duration : 
  (mowing_earnings + weed_eating_earnings) / weekly_spending = 9 := by
  sorry

end NUMINAMATH_CALUDE_money_duration_l121_12121


namespace NUMINAMATH_CALUDE_outfit_combinations_l121_12135

def num_shirts : ℕ := 8
def num_pants : ℕ := 5
def num_jacket_options : ℕ := 3

theorem outfit_combinations :
  num_shirts * num_pants * num_jacket_options = 120 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l121_12135


namespace NUMINAMATH_CALUDE_oranges_count_l121_12157

/-- The number of oranges initially in Tom's fruit bowl -/
def initial_oranges : ℕ := 3

/-- The number of lemons initially in Tom's fruit bowl -/
def initial_lemons : ℕ := 6

/-- The number of fruits Tom eats -/
def fruits_eaten : ℕ := 3

/-- The number of fruits remaining after Tom eats -/
def remaining_fruits : ℕ := 6

/-- Theorem stating that the number of oranges initially in the fruit bowl is 3 -/
theorem oranges_count : initial_oranges = 3 :=
  by
    have h1 : initial_oranges + initial_lemons = remaining_fruits + fruits_eaten :=
      by sorry
    have h2 : initial_oranges + 6 = 6 + 3 :=
      by sorry
    show initial_oranges = 3
    sorry

#check oranges_count

end NUMINAMATH_CALUDE_oranges_count_l121_12157


namespace NUMINAMATH_CALUDE_n_times_n_plus_one_divisible_by_two_l121_12105

theorem n_times_n_plus_one_divisible_by_two (n : ℤ) (h : 1 ≤ n ∧ n ≤ 99) : 
  2 ∣ (n * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_n_times_n_plus_one_divisible_by_two_l121_12105


namespace NUMINAMATH_CALUDE_reservoir_crossing_time_l121_12193

/-- The time it takes to cross a reservoir under specific conditions -/
theorem reservoir_crossing_time
  (b : ℝ)  -- width of the reservoir in km
  (v : ℝ)  -- swimming speed of A and C in km/h
  (h1 : 0 < b)  -- reservoir width is positive
  (h2 : 0 < v)  -- swimming speed is positive
  : ∃ (t : ℝ), t = (31 * b) / (130 * v) ∧ 
    (∃ (x d : ℝ),
      0 < x ∧ 0 < d ∧
      x = (9 * b) / 13 ∧
      d = (b - x) / 2 ∧
      2 * d + x = b ∧
      (b + 3 * x) / 2 / (10 * v) = d / v ∧
      t = ((b + 2 * x) / (10 * v))) :=
sorry

end NUMINAMATH_CALUDE_reservoir_crossing_time_l121_12193


namespace NUMINAMATH_CALUDE_parabola_horizontal_shift_l121_12176

/-- A parabola is defined by its coefficient and vertex -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in vertex form is y = a(x-h)^2 + k -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

theorem parabola_horizontal_shift 
  (p1 p2 : Parabola) 
  (h1 : p1.a = p2.a) 
  (h2 : p1.k = p2.k) 
  (h3 : p1.h = p2.h + 3) : 
  ∀ x, parabola_equation p1 x = parabola_equation p2 (x - 3) :=
sorry

end NUMINAMATH_CALUDE_parabola_horizontal_shift_l121_12176


namespace NUMINAMATH_CALUDE_shortest_side_right_triangle_l121_12199

theorem shortest_side_right_triangle (a b c : ℝ) : 
  a = 7 → b = 10 → c^2 = a^2 + b^2 → min a (min b c) = 7 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_right_triangle_l121_12199


namespace NUMINAMATH_CALUDE_negative_four_cubed_equality_l121_12166

theorem negative_four_cubed_equality : (-4)^3 = -4^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_four_cubed_equality_l121_12166


namespace NUMINAMATH_CALUDE_marie_erasers_l121_12126

/-- Given that Marie starts with 95.0 erasers and buys 42.0 erasers, 
    prove that she ends up with 137.0 erasers. -/
theorem marie_erasers : 
  let initial_erasers : ℝ := 95.0
  let bought_erasers : ℝ := 42.0
  let final_erasers : ℝ := initial_erasers + bought_erasers
  final_erasers = 137.0 := by
  sorry

end NUMINAMATH_CALUDE_marie_erasers_l121_12126


namespace NUMINAMATH_CALUDE_isosceles_triangulation_condition_l121_12100

/-- A regular convex polygon with n sides -/
structure RegularConvexPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- A triangulation of a polygon -/
structure Triangulation (P : RegularConvexPolygon) where
  isosceles : Bool

/-- Theorem: If a regular convex polygon with n sides has a triangulation
    consisting of only isosceles triangles, then n can be written as 2^(a+1) + 2^b
    for some non-negative integers a and b -/
theorem isosceles_triangulation_condition (P : RegularConvexPolygon)
  (T : Triangulation P) (h : T.isosceles = true) :
  ∃ (a b : ℕ), P.n = 2^(a+1) + 2^b :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangulation_condition_l121_12100


namespace NUMINAMATH_CALUDE_chemistry_books_count_l121_12192

def number_of_biology_books : ℕ := 13
def total_combinations : ℕ := 2184

theorem chemistry_books_count (C : ℕ) : 
  (number_of_biology_books.choose 2) * (C.choose 2) = total_combinations → C = 8 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_books_count_l121_12192


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_less_than_one_l121_12174

theorem negation_of_existence_squared_less_than_one :
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_less_than_one_l121_12174


namespace NUMINAMATH_CALUDE_bells_toll_together_l121_12155

def bell_intervals : List Nat := [2, 4, 6, 8, 10, 12]
def period_minutes : Nat := 30
def period_seconds : Nat := period_minutes * 60

def lcm_list (list : List Nat) : Nat :=
  list.foldl Nat.lcm 1

theorem bells_toll_together : 
  (period_seconds / lcm_list bell_intervals) + 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bells_toll_together_l121_12155


namespace NUMINAMATH_CALUDE_abc_sum_l121_12189

theorem abc_sum (a b c : ℕ+) (h : (139 : ℚ) / 22 = a + 1 / (b + 1 / c)) : 
  (a : ℕ) + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_l121_12189


namespace NUMINAMATH_CALUDE_birds_joining_fence_l121_12151

theorem birds_joining_fence (initial_storks initial_birds joining_birds : ℕ) : 
  initial_storks = 6 →
  initial_birds = 2 →
  initial_storks = initial_birds + joining_birds + 1 →
  joining_birds = 3 := by
sorry

end NUMINAMATH_CALUDE_birds_joining_fence_l121_12151


namespace NUMINAMATH_CALUDE_symmetry_f_and_f_inv_symmetry_f_and_f_swap_same_curve_f_and_f_inv_l121_12175

-- Define a function f and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverses of each other
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Statement 1
theorem symmetry_f_and_f_inv :
  ∀ x y, y = f x ↔ x = f_inv y :=
sorry

-- Statement 2
theorem symmetry_f_and_f_swap :
  ∀ x y, y = f x ↔ x = f y :=
sorry

-- Statement 4
theorem same_curve_f_and_f_inv :
  ∀ x y, y = f x ↔ x = f_inv y :=
sorry

end NUMINAMATH_CALUDE_symmetry_f_and_f_inv_symmetry_f_and_f_swap_same_curve_f_and_f_inv_l121_12175


namespace NUMINAMATH_CALUDE_fraction_comparison_and_inequality_l121_12178

theorem fraction_comparison_and_inequality : 
  (37 : ℚ) / 29 < 41 / 31 ∧ 
  41 / 31 < 31 / 23 ∧ 
  37 / 29 ≠ 4 / 3 ∧ 
  41 / 31 ≠ 4 / 3 ∧ 
  31 / 23 ≠ 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_and_inequality_l121_12178


namespace NUMINAMATH_CALUDE_amount_to_hand_in_l121_12168

/-- Represents the denominations of bills in US currency --/
inductive Denomination
  | Hundred
  | Fifty
  | Twenty
  | Ten
  | Five
  | One

/-- Represents the quantity of each denomination in the till --/
def till_contents : Denomination → ℕ
  | Denomination.Hundred => 2
  | Denomination.Fifty => 1
  | Denomination.Twenty => 5
  | Denomination.Ten => 3
  | Denomination.Five => 7
  | Denomination.One => 27

/-- The value of each denomination in dollars --/
def denomination_value : Denomination → ℕ
  | Denomination.Hundred => 100
  | Denomination.Fifty => 50
  | Denomination.Twenty => 20
  | Denomination.Ten => 10
  | Denomination.Five => 5
  | Denomination.One => 1

/-- The amount to be left in the till --/
def amount_to_leave : ℕ := 300

/-- Calculates the total value of bills in the till --/
def total_value : ℕ := sorry

/-- Theorem: The amount Jack will hand in is $142 --/
theorem amount_to_hand_in :
  total_value - amount_to_leave = 142 := by sorry

end NUMINAMATH_CALUDE_amount_to_hand_in_l121_12168


namespace NUMINAMATH_CALUDE_append_five_to_two_digit_number_l121_12161

theorem append_five_to_two_digit_number (t u : ℕ) (h1 : t ≤ 9) (h2 : u ≤ 9) :
  let original := 10 * t + u
  100 * t + 10 * u + 5 = original * 10 + 5 := by
  sorry

end NUMINAMATH_CALUDE_append_five_to_two_digit_number_l121_12161


namespace NUMINAMATH_CALUDE_aartis_work_time_l121_12103

/-- Given that Aarti completes three times a piece of work in 27 days,
    prove that she can complete one piece of work in 9 days. -/
theorem aartis_work_time :
  ∀ (work_time : ℕ),
  (3 * work_time = 27) →
  (work_time = 9) :=
by sorry

end NUMINAMATH_CALUDE_aartis_work_time_l121_12103


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l121_12140

theorem fraction_inequality_solution_set :
  {x : ℝ | (x + 1) / (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < -1} :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l121_12140


namespace NUMINAMATH_CALUDE_all_numbers_equal_l121_12179

/-- Represents a grid of positive integers -/
def Grid := ℕ → ℕ → ℕ+

/-- Checks if two polygons are congruent -/
def CongruentPolygons (p q : Set (ℕ × ℕ)) : Prop := sorry

/-- Calculates the sum of numbers in a polygon -/
def PolygonSum (g : Grid) (p : Set (ℕ × ℕ)) : ℕ := sorry

/-- Calculates the area of a polygon -/
def PolygonArea (p : Set (ℕ × ℕ)) : ℕ := sorry

/-- Main theorem -/
theorem all_numbers_equal (g : Grid) (n : ℕ) (h_n : n > 2) :
  (∀ p q : Set (ℕ × ℕ), CongruentPolygons p q → PolygonArea p = n → PolygonArea q = n →
    PolygonSum g p = PolygonSum g q) →
  ∀ i j k l : ℕ, g i j = g k l :=
sorry

end NUMINAMATH_CALUDE_all_numbers_equal_l121_12179


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l121_12111

def complex_number_quadrant (z : ℂ) : Prop :=
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (3 + 2 * Complex.I) / Complex.I
  complex_number_quadrant z :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l121_12111


namespace NUMINAMATH_CALUDE_sum_a_3000_l121_12115

def a (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 18 = 0 then 15
  else if n % 18 = 0 ∧ n % 17 = 0 then 18
  else if n % 15 = 0 ∧ n % 17 = 0 then 21
  else 0

theorem sum_a_3000 :
  (Finset.range 3000).sum (fun n => a (n + 1)) = 888 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_3000_l121_12115


namespace NUMINAMATH_CALUDE_max_triangles_from_lines_l121_12116

/-- Given 2017 lines separated into three sets such that lines in the same set are parallel to each other,
    prove that the largest possible number of triangles that can be formed with vertices on these lines
    is 673 * 672^2. -/
theorem max_triangles_from_lines (total_lines : ℕ) (set1 set2 set3 : ℕ) :
  total_lines = 2017 →
  set1 + set2 + set3 = total_lines →
  set1 ≥ set2 →
  set2 ≥ set3 →
  set1 * set2 * set3 ≤ 673 * 672 * 672 :=
by sorry

end NUMINAMATH_CALUDE_max_triangles_from_lines_l121_12116


namespace NUMINAMATH_CALUDE_marble_selection_ways_l121_12156

def total_marbles : ℕ := 15
def red_marbles : ℕ := 2
def green_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def marbles_to_choose : ℕ := 5

theorem marble_selection_ways :
  (red_marbles.choose 1) * (green_marbles.choose 1) *
  ((total_marbles - red_marbles - green_marbles + 2).choose (marbles_to_choose - 2)) = 660 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l121_12156


namespace NUMINAMATH_CALUDE_gcd_6724_13104_l121_12128

theorem gcd_6724_13104 : Nat.gcd 6724 13104 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_6724_13104_l121_12128


namespace NUMINAMATH_CALUDE_solution_to_system_l121_12186

theorem solution_to_system : ∃ (x y : ℝ), 
  x^2 * y - x * y^2 - 3*x + 3*y + 1 = 0 ∧
  x^3 * y - x * y^3 - 3*x^2 + 3*y^2 + 3 = 0 ∧
  x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l121_12186


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l121_12154

theorem simplify_and_evaluate (x : ℝ) (h : x = 1 / (3 + 2 * Real.sqrt 2)) :
  ((1 - x)^2 / (x - 1)) + (Real.sqrt (x^2 + 4 - 4*x) / (x - 2)) = 1 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l121_12154


namespace NUMINAMATH_CALUDE_correct_years_until_twice_as_old_l121_12113

/-- Represents the current ages of the three brothers -/
structure BrothersAges where
  david : ℕ
  richard : ℕ
  scott : ℕ

/-- Calculates the number of years until Richard is twice as old as Scott -/
def yearsUntilTwiceAsOld (ages : BrothersAges) : ℕ :=
  sorry

theorem correct_years_until_twice_as_old : 
  ∀ (ages : BrothersAges),
    ages.david = 14 →
    ages.richard = ages.david + 6 →
    ages.scott = ages.david - 8 →
    yearsUntilTwiceAsOld ages = 8 :=
  sorry

end NUMINAMATH_CALUDE_correct_years_until_twice_as_old_l121_12113


namespace NUMINAMATH_CALUDE_percentage_difference_l121_12127

theorem percentage_difference (x : ℝ) : x = 35 → (0.8 * 170) - (x / 100 * 300) = 31 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l121_12127


namespace NUMINAMATH_CALUDE_distance_between_points_l121_12177

theorem distance_between_points (x₁ x₂ y₁ y₂ : ℝ) :
  x₁^2 + y₁^2 = 29 →
  x₂^2 + y₂^2 = 29 →
  x₁ + y₁ = 11 →
  x₂ + y₂ = 11 →
  x₁ ≠ x₂ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l121_12177


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l121_12112

/-- Tangent line to a circle -/
theorem tangent_line_to_circle (r x_0 y_0 : ℝ) (h : x_0^2 + y_0^2 = r^2) :
  ∀ x y : ℝ, (x^2 + y^2 = r^2) → ((x - x_0)^2 + (y - y_0)^2 = 0 ∨ x_0*x + y_0*y = r^2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l121_12112


namespace NUMINAMATH_CALUDE_expression_evaluation_l121_12194

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^y + 2 * y^x = 533 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l121_12194


namespace NUMINAMATH_CALUDE_dress_price_difference_l121_12153

theorem dress_price_difference (P : ℝ) (h : P - 0.15 * P = 68) :
  P - (68 + 0.25 * 68) = -5 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_difference_l121_12153


namespace NUMINAMATH_CALUDE_art_project_marker_distribution_l121_12123

theorem art_project_marker_distribution (total_students : ℕ) (total_boxes : ℕ) (markers_per_box : ℕ)
  (group1_students : ℕ) (group1_markers : ℕ) (group2_students : ℕ) (group2_markers : ℕ) :
  total_students = 30 →
  total_boxes = 22 →
  markers_per_box = 5 →
  group1_students = 10 →
  group1_markers = 2 →
  group2_students = 15 →
  group2_markers = 4 →
  (total_students - group1_students - group2_students) > 0 →
  (total_boxes * markers_per_box - group1_students * group1_markers - group2_students * group2_markers) /
    (total_students - group1_students - group2_students) = 6 :=
by sorry

end NUMINAMATH_CALUDE_art_project_marker_distribution_l121_12123


namespace NUMINAMATH_CALUDE_smoothie_size_l121_12190

-- Define the constants from the problem
def packet_size : ℝ := 3
def water_per_packet : ℝ := 15
def total_smoothies : ℝ := 150
def total_packets : ℝ := 180

-- Define the theorem
theorem smoothie_size :
  let packets_per_smoothie := total_packets / total_smoothies
  let mix_per_smoothie := packets_per_smoothie * packet_size
  let water_per_smoothie := packets_per_smoothie * water_per_packet
  mix_per_smoothie + water_per_smoothie = 21.6 := by
sorry

end NUMINAMATH_CALUDE_smoothie_size_l121_12190


namespace NUMINAMATH_CALUDE_stock_price_increase_l121_12181

/-- Proves that if a stock's price increases by 50% and closes at $15, then its opening price was $10. -/
theorem stock_price_increase (opening_price closing_price : ℝ) :
  closing_price = 15 ∧ closing_price = opening_price * 1.5 → opening_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l121_12181


namespace NUMINAMATH_CALUDE_median_name_length_is_five_l121_12144

/-- Represents the distribution of name lengths -/
structure NameLengthDistribution where
  fourLetters : Nat
  fiveLetters : Nat
  sixLetters : Nat
  sevenLetters : Nat

/-- Calculates the median of a list of numbers -/
def median (list : List Nat) : Rat :=
  sorry

/-- Generates a list of name lengths based on the distribution -/
def generateNameLengthList (dist : NameLengthDistribution) : List Nat :=
  sorry

theorem median_name_length_is_five (dist : NameLengthDistribution) : 
  dist.fourLetters = 9 ∧ 
  dist.fiveLetters = 6 ∧ 
  dist.sixLetters = 2 ∧ 
  dist.sevenLetters = 7 → 
  median (generateNameLengthList dist) = 5 := by
  sorry

end NUMINAMATH_CALUDE_median_name_length_is_five_l121_12144


namespace NUMINAMATH_CALUDE_mary_pokemon_cards_l121_12133

theorem mary_pokemon_cards (x : ℕ) : 
  x + 23 - 6 = 56 → x = 39 := by
sorry

end NUMINAMATH_CALUDE_mary_pokemon_cards_l121_12133


namespace NUMINAMATH_CALUDE_boys_in_school_l121_12139

/-- The number of boys in a school, given the initial number of girls, 
    the number of new girls who joined, and the total number of pupils after new girls joined. -/
def number_of_boys (initial_girls new_girls total_pupils : ℕ) : ℕ :=
  total_pupils - (initial_girls + new_girls)

/-- Theorem stating that the number of boys in the school is 222 -/
theorem boys_in_school : number_of_boys 706 418 1346 = 222 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_school_l121_12139


namespace NUMINAMATH_CALUDE_order_of_special_values_l121_12196

/-- Given a = √(1.01), b = e^(0.01) / 1.01, and c = ln(1.01e), prove that b < a < c. -/
theorem order_of_special_values :
  let a : ℝ := Real.sqrt 1.01
  let b : ℝ := Real.exp 0.01 / 1.01
  let c : ℝ := Real.log (1.01 * Real.exp 1)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_special_values_l121_12196


namespace NUMINAMATH_CALUDE_square_of_difference_l121_12106

theorem square_of_difference (a b : ℝ) : (a - b)^2 = a^2 - 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l121_12106


namespace NUMINAMATH_CALUDE_min_trips_for_given_weights_l121_12136

-- Define the list of people's weights
def weights : List ℕ := [130, 60, 61, 65, 68, 70, 79, 81, 83, 87, 90, 91, 95]

-- Define the elevator capacity
def capacity : ℕ := 175

-- Function to calculate the minimum number of trips
def min_trips (weights : List ℕ) (capacity : ℕ) : ℕ := sorry

-- Theorem stating that the minimum number of trips is 7
theorem min_trips_for_given_weights :
  min_trips weights capacity = 7 := by sorry

end NUMINAMATH_CALUDE_min_trips_for_given_weights_l121_12136


namespace NUMINAMATH_CALUDE_roi_is_25_percent_l121_12118

/-- Calculates the return on investment (ROI) percentage for an investor given the dividend rate, face value, and purchase price of shares. -/
def calculate_roi (dividend_rate : ℚ) (face_value : ℚ) (purchase_price : ℚ) : ℚ :=
  (dividend_rate * face_value / purchase_price) * 100

/-- Theorem stating that for the given conditions, the ROI is 25%. -/
theorem roi_is_25_percent :
  let dividend_rate : ℚ := 125 / 1000  -- 12.5%
  let face_value : ℚ := 50
  let purchase_price : ℚ := 25
  calculate_roi dividend_rate face_value purchase_price = 25 := by
  sorry

#eval calculate_roi (125/1000) 50 25  -- This should evaluate to 25

end NUMINAMATH_CALUDE_roi_is_25_percent_l121_12118


namespace NUMINAMATH_CALUDE_integer_sqrt_15_l121_12150

theorem integer_sqrt_15 (a : ℝ) : 
  (∃ m n : ℤ, (a + Real.sqrt 15 = m) ∧ (1 / (a - Real.sqrt 15) = n)) →
  (a = 4 + Real.sqrt 15 ∨ a = -(4 + Real.sqrt 15)) :=
by sorry

end NUMINAMATH_CALUDE_integer_sqrt_15_l121_12150


namespace NUMINAMATH_CALUDE_xy_value_l121_12160

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x+y) = 16)
  (h2 : (16 : ℝ)^(x+y) / (4 : ℝ)^(7*y) = 256) : 
  x * y = 48 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l121_12160


namespace NUMINAMATH_CALUDE_triangle_side_range_l121_12142

theorem triangle_side_range (a b c : ℝ) : 
  c = 4 → -- Given condition: one side has length 4
  a > 0 → -- Positive length
  b > 0 → -- Positive length
  a ≤ b → -- Assume a is the shorter of the two variable sides
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  a < 4 * Real.sqrt 2 -- Upper bound of the range
  ∧ a > 0 -- Lower bound of the range
  := by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l121_12142


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l121_12122

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l121_12122


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l121_12172

theorem diophantine_equation_solution :
  ∃ (m n : ℕ), 26019 * m - 649 * n = 118 ∧ m = 2 ∧ n = 80 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l121_12172


namespace NUMINAMATH_CALUDE_existence_of_c_l121_12163

theorem existence_of_c (n : ℕ) (a b : Fin n → ℝ) 
  (h_n : n ≥ 2)
  (h_pos : ∀ i, a i > 0 ∧ b i > 0)
  (h_less : ∀ i, a i < b i)
  (h_sum : (Finset.sum Finset.univ b) < 1 + (Finset.sum Finset.univ a)) :
  ∃ c : ℝ, ∀ (i : Fin n) (k : ℤ), (a i + c + k) * (b i + c + k) > 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_c_l121_12163


namespace NUMINAMATH_CALUDE_rupert_age_rupert_candles_l121_12132

-- Define Peter's age
def peter_age : ℕ := 10

-- Define the ratio of Rupert's age to Peter's age
def age_ratio : ℚ := 7/2

-- Theorem to prove Rupert's age
theorem rupert_age : ℕ := by
  -- The proof goes here
  sorry

-- Theorem to prove the number of candles on Rupert's cake
theorem rupert_candles : ℕ := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_rupert_age_rupert_candles_l121_12132


namespace NUMINAMATH_CALUDE_white_ball_estimate_l121_12180

/-- Represents the result of drawing balls from a bag -/
structure BagDrawResult where
  totalBalls : ℕ
  totalDraws : ℕ
  whiteDraws : ℕ

/-- Calculates the estimated number of white balls in the bag -/
def estimateWhiteBalls (result : BagDrawResult) : ℚ :=
  result.totalBalls * (result.whiteDraws : ℚ) / result.totalDraws

theorem white_ball_estimate (result : BagDrawResult) 
  (h1 : result.totalBalls = 20)
  (h2 : result.totalDraws = 100)
  (h3 : result.whiteDraws = 40) :
  estimateWhiteBalls result = 8 := by
  sorry

#eval estimateWhiteBalls { totalBalls := 20, totalDraws := 100, whiteDraws := 40 }

end NUMINAMATH_CALUDE_white_ball_estimate_l121_12180


namespace NUMINAMATH_CALUDE_trigonometric_values_signs_l121_12164

theorem trigonometric_values_signs :
  (∃ x, x = Real.sin (-1000 * π / 180) ∧ x > 0) ∧
  (∃ y, y = Real.cos (-2200 * π / 180) ∧ y > 0) ∧
  (∃ z, z = Real.tan (-10) ∧ z < 0) ∧
  (∃ w, w = (Real.sin (7 * π / 10) * Real.cos π) / Real.tan (17 * π / 9) ∧ w > 0) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_values_signs_l121_12164


namespace NUMINAMATH_CALUDE_area_isosceles_right_triangle_l121_12182

/-- Given a right triangle ABC with AB = 12 and AC = 24, and points D on AC and E on BC
    forming an isosceles right triangle BDE, prove that the area of BDE is 80. -/
theorem area_isosceles_right_triangle (A B C D E : ℝ × ℝ) : 
  -- Right triangle ABC
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) →
  -- AB = 12
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 12 →
  -- AC = 24
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 24 →
  -- D is on AC
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2)) →
  -- E is on BC
  ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (B.1 + s * (C.1 - B.1), B.2 + s * (C.2 - B.2)) →
  -- BDE is an isosceles right triangle
  (D.1 - B.1) * (E.1 - B.1) + (D.2 - B.2) * (E.2 - B.2) = 0 ∧
  (D.1 - B.1)^2 + (D.2 - B.2)^2 = (E.1 - B.1)^2 + (E.2 - B.2)^2 →
  -- Area of BDE is 80
  (1/2) * Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) * Real.sqrt ((E.1 - B.1)^2 + (E.2 - B.2)^2) = 80 :=
by sorry


end NUMINAMATH_CALUDE_area_isosceles_right_triangle_l121_12182


namespace NUMINAMATH_CALUDE_fixed_points_of_quadratic_l121_12125

/-- A quadratic function of the form f(x) = mx^2 - 2mx + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 3

/-- Theorem stating that (0, 3) and (2, 3) are fixed points of f for all non-zero m -/
theorem fixed_points_of_quadratic (m : ℝ) (h : m ≠ 0) :
  (f m 0 = 3) ∧ (f m 2 = 3) := by
  sorry

#check fixed_points_of_quadratic

end NUMINAMATH_CALUDE_fixed_points_of_quadratic_l121_12125


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l121_12138

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 18) :
  ∃ (k : ℕ+), k = Nat.gcd (12 * m) (20 * n) ∧ 
  (∀ (l : ℕ+), l = Nat.gcd (12 * m) (20 * n) → k ≤ l) ∧
  k = 72 := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l121_12138


namespace NUMINAMATH_CALUDE_distance_between_stations_l121_12145

/-- The distance between two stations given three cars with different speeds --/
theorem distance_between_stations (speed_A speed_B speed_C : ℝ) (time_diff : ℝ) : 
  speed_A = 90 →
  speed_B = 80 →
  speed_C = 60 →
  time_diff = 1/3 →
  (speed_A + speed_B) * ((speed_A + speed_C) * time_diff / (speed_B - speed_C)) = 425 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_stations_l121_12145


namespace NUMINAMATH_CALUDE_sequence_identity_l121_12148

def StrictlyIncreasing (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem sequence_identity (a : ℕ → ℕ) 
  (h_increasing : StrictlyIncreasing a)
  (h_upper_bound : ∀ n : ℕ, a n ≤ n + 2020)
  (h_divisibility : ∀ n : ℕ, (a (n + 1)) ∣ (n^3 * (a n) - 1)) :
  ∀ n : ℕ, a n = n :=
sorry

end NUMINAMATH_CALUDE_sequence_identity_l121_12148


namespace NUMINAMATH_CALUDE_cube_coverage_l121_12170

/-- Represents a rectangular strip of size 1 × 2 -/
structure Rectangle where
  length : Nat
  width : Nat

/-- Represents a cube of size n × n × n -/
structure Cube where
  size : Nat

/-- Predicate to check if a rectangle abuts exactly five others -/
def abutsFiveOthers (r : Rectangle) : Prop :=
  sorry

/-- Predicate to check if a cube's surface can be covered with rectangles -/
def canBeCovered (c : Cube) (r : Rectangle) : Prop :=
  sorry

theorem cube_coverage (n : Nat) :
  (∃ c : Cube, c.size = n ∧ ∃ r : Rectangle, r.length = 2 ∧ r.width = 1 ∧
    canBeCovered c r ∧ abutsFiveOthers r) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_cube_coverage_l121_12170


namespace NUMINAMATH_CALUDE_perfect_square_product_iff_factors_l121_12104

theorem perfect_square_product_iff_factors (x y z : ℕ+) :
  ∃ (n : ℕ), (x * y + 1) * (y * z + 1) * (z * x + 1) = n ^ 2 ↔
  (∃ (a b c : ℕ), (x * y + 1 = a ^ 2) ∧ (y * z + 1 = b ^ 2) ∧ (z * x + 1 = c ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_product_iff_factors_l121_12104
