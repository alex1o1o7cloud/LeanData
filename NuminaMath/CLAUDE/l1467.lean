import Mathlib

namespace total_amount_theorem_l1467_146736

/-- The total amount spent on cows and goats -/
def total_amount_spent (num_cows num_goats avg_price_cow avg_price_goat : ℕ) : ℕ :=
  num_cows * avg_price_cow + num_goats * avg_price_goat

/-- Theorem: The total amount spent on 2 cows and 10 goats is 1500 rupees -/
theorem total_amount_theorem :
  total_amount_spent 2 10 400 70 = 1500 := by
  sorry

end total_amount_theorem_l1467_146736


namespace hexagon_theorem_l1467_146784

/-- Regular hexagon with side length 4 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ := 4)

/-- Intersection point of diagonals CE and DF -/
def L (hex : RegularHexagon) : ℝ × ℝ := sorry

/-- Point K defined by vector equation -/
def K (hex : RegularHexagon) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Predicate for a point being outside a hexagon -/
def is_outside (p : ℝ × ℝ) (hex : RegularHexagon) : Prop := sorry

theorem hexagon_theorem (hex : RegularHexagon) :
  is_outside (K hex) hex ∧ distance (K hex) hex.A = (4 * Real.sqrt 3) / 3 := by sorry

end hexagon_theorem_l1467_146784


namespace popcorn_yield_two_tablespoons_yield_l1467_146791

/-- Represents the ratio of cups of popcorn to tablespoons of kernels -/
def popcorn_ratio (cups : ℚ) (tablespoons : ℚ) : Prop :=
  cups / tablespoons = 2

theorem popcorn_yield (cups : ℚ) (tablespoons : ℚ) 
  (h : popcorn_ratio 16 8) : 
  popcorn_ratio cups tablespoons → cups = 2 * tablespoons :=
by
  sorry

/-- Shows that 2 tablespoons of kernels make 4 cups of popcorn -/
theorem two_tablespoons_yield (h : popcorn_ratio 16 8) : 
  popcorn_ratio 4 2 :=
by
  sorry

end popcorn_yield_two_tablespoons_yield_l1467_146791


namespace cindy_friends_l1467_146757

/-- Calculates the number of friends Cindy gives envelopes to -/
def num_friends (initial_envelopes : ℕ) (envelopes_per_friend : ℕ) (remaining_envelopes : ℕ) : ℕ :=
  (initial_envelopes - remaining_envelopes) / envelopes_per_friend

/-- Proves that Cindy gives envelopes to 5 friends -/
theorem cindy_friends : num_friends 37 3 22 = 5 := by
  sorry

end cindy_friends_l1467_146757


namespace mouse_cheese_distance_sum_l1467_146738

/-- The point where the mouse begins moving away from the cheese -/
def mouse_turn_point (c d : ℝ) : Prop :=
  ∃ (k : ℝ), 
    d = -3 * c + 18 ∧  -- Mouse path
    d - 5 = k * (c - 20) ∧  -- Perpendicular line
    -3 * k = -1  -- Perpendicular condition

/-- The theorem stating the sum of coordinates where the mouse turns -/
theorem mouse_cheese_distance_sum : 
  ∃ (c d : ℝ), mouse_turn_point c d ∧ c + d = 9.4 :=
sorry

end mouse_cheese_distance_sum_l1467_146738


namespace final_lives_correct_tiffany_final_lives_l1467_146772

/-- Given an initial number of lives, lives lost, and a bonus multiplier,
    calculate the final number of lives after completing the bonus stage. -/
def finalLives (initialLives lostLives bonusMultiplier : ℕ) : ℕ :=
  let remainingLives := initialLives - lostLives
  remainingLives + bonusMultiplier * remainingLives

/-- Theorem: The final number of lives after the bonus stage is correct. -/
theorem final_lives_correct (initialLives lostLives bonusMultiplier : ℕ) 
    (h : lostLives ≤ initialLives) :
    finalLives initialLives lostLives bonusMultiplier = 
    (initialLives - lostLives) + bonusMultiplier * (initialLives - lostLives) := by
  sorry

/-- Corollary: For the specific case in the problem. -/
theorem tiffany_final_lives : 
    finalLives 250 58 3 = 768 := by
  sorry

end final_lives_correct_tiffany_final_lives_l1467_146772


namespace ball_hits_ground_at_calculated_time_l1467_146713

/-- The time when a ball hits the ground, given its height equation -/
def ball_ground_time : ℝ :=
  let initial_height : ℝ := 180
  let initial_velocity : ℝ := -32  -- negative because it's downward
  let release_delay : ℝ := 1
  let height (t : ℝ) : ℝ := -16 * (t - release_delay)^2 - 32 * (t - release_delay) + initial_height
  3.5

/-- Theorem stating that the ball hits the ground at the calculated time -/
theorem ball_hits_ground_at_calculated_time :
  let height (t : ℝ) : ℝ := -16 * (t - 1)^2 - 32 * (t - 1) + 180
  height ball_ground_time = 0 := by
  sorry

end ball_hits_ground_at_calculated_time_l1467_146713


namespace prime_odd_sum_l1467_146731

theorem prime_odd_sum (x y : ℕ) : 
  Nat.Prime x → 
  Odd y → 
  x^2 + y = 2009 → 
  x + y = 2007 := by
sorry

end prime_odd_sum_l1467_146731


namespace problem_solution_l1467_146779

theorem problem_solution (m n : ℕ+) 
  (h1 : m.val + 12 < n.val + 3)
  (h2 : (m.val + (m.val + 6) + (m.val + 12) + (n.val + 3) + (n.val + 6) + 3 * n.val) / 6 = n.val + 3)
  (h3 : (m.val + 12 + n.val + 3) / 2 = n.val + 3) : 
  m.val + n.val = 57 := by
sorry

end problem_solution_l1467_146779


namespace min_value_of_f_l1467_146748

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem min_value_of_f (a : ℝ) (h1 : a > 2) 
  (h2 : ∀ x > 2, f x ≥ f a) : a = 3 := by
  sorry

end min_value_of_f_l1467_146748


namespace yoga_time_calculation_l1467_146727

/-- Calculates the yoga time given exercise ratios and bicycle riding time -/
theorem yoga_time_calculation (bicycle_time : ℚ) : 
  bicycle_time = 12 → (40 : ℚ) / 3 = 
    2 * (2 * bicycle_time / 3 + bicycle_time) / 3 := by
  sorry

#eval (40 : ℚ) / 3

end yoga_time_calculation_l1467_146727


namespace division_remainder_l1467_146715

theorem division_remainder (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 31 = 5 := by
  sorry

end division_remainder_l1467_146715


namespace average_visitors_theorem_l1467_146744

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (daysInMonth : Nat) (sundayVisitors : Nat) (otherDayVisitors : Nat) : Nat :=
  let numSundays := (daysInMonth + 6) / 7
  let numOtherDays := daysInMonth - numSundays
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / daysInMonth

theorem average_visitors_theorem :
  averageVisitors 30 660 240 = 296 := by
  sorry

end average_visitors_theorem_l1467_146744


namespace smallest_independent_after_reorganization_l1467_146754

/-- Represents a faction of deputies -/
structure Faction :=
  (size : ℕ)

/-- Represents the parliament configuration -/
structure Parliament :=
  (factions : List Faction)
  (independent : ℕ)

def initialParliament : Parliament :=
  { factions := [5, 6, 7, 8, 9, 10, 11, 12, 13, 14].map (λ n => ⟨n⟩),
    independent := 0 }

def totalDeputies (p : Parliament) : ℕ :=
  p.factions.foldl (λ acc f => acc + f.size) p.independent

def isValidReorganization (initial final : Parliament) : Prop :=
  totalDeputies initial = totalDeputies final ∧
  final.factions.all (λ f => f.size ≤ initial.factions.length) ∧
  final.factions.all (λ f => f.size ≥ 5)

theorem smallest_independent_after_reorganization :
  ∀ (final : Parliament),
    isValidReorganization initialParliament final →
    final.independent ≥ 50 :=
sorry

end smallest_independent_after_reorganization_l1467_146754


namespace bridge_length_l1467_146793

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 → 
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 225 := by
  sorry

end bridge_length_l1467_146793


namespace area_ratio_of_inner_triangle_l1467_146722

/-- Given a triangle with area T, if we divide each side of the triangle in the ratio of 1:2
    (starting from each vertex) and form a new triangle by connecting these points,
    the area of the new triangle S is related to the area of the original triangle T
    by the equation: S / T = 1 / 9 -/
theorem area_ratio_of_inner_triangle (T : ℝ) (S : ℝ) (h : T > 0) :
  (∀ (side : ℝ), ∃ (new_side : ℝ), new_side = side / 3) →
  S / T = 1 / 9 := by
  sorry

end area_ratio_of_inner_triangle_l1467_146722


namespace problem_solution_l1467_146798

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 5*x + 5/x + 1/x^2 = 42)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by sorry

end problem_solution_l1467_146798


namespace right_triangle_special_case_l1467_146708

theorem right_triangle_special_case (a b c : ℝ) :
  a > 0 →  -- AB is positive
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  c + b = 2*a →  -- Given condition
  b = 3/4 * a ∧ c = 5/4 * a := by sorry

end right_triangle_special_case_l1467_146708


namespace production_rates_l1467_146778

/-- The rate at which A makes parts per hour -/
def rate_A : ℝ := sorry

/-- The rate at which B makes parts per hour -/
def rate_B : ℝ := sorry

/-- The time it takes for A to make 90 parts equals the time for B to make 120 parts -/
axiom time_ratio : (90 / rate_A) = (120 / rate_B)

/-- A and B together make 35 parts per hour -/
axiom total_rate : rate_A + rate_B = 35

theorem production_rates : rate_A = 15 ∧ rate_B = 20 := by
  sorry

end production_rates_l1467_146778


namespace melissa_bananas_l1467_146768

/-- Calculates the number of bananas Melissa has left after sharing some. -/
def bananas_left (initial : ℕ) (shared : ℕ) : ℕ :=
  initial - shared

/-- Proves that Melissa has 84 bananas left after sharing 4 out of her initial 88 bananas. -/
theorem melissa_bananas : bananas_left 88 4 = 84 := by
  sorry

end melissa_bananas_l1467_146768


namespace reading_difference_l1467_146775

/-- The number of pages Janet reads per day -/
def janet_pages_per_day : ℕ := 80

/-- The number of pages Belinda reads per day -/
def belinda_pages_per_day : ℕ := 30

/-- The number of weeks in the reading period -/
def reading_weeks : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem reading_difference :
  (janet_pages_per_day - belinda_pages_per_day) * (reading_weeks * days_per_week) = 2100 := by
  sorry

end reading_difference_l1467_146775


namespace cookie_ratio_anna_to_tim_l1467_146767

/-- Represents the distribution of cookies among recipients --/
structure CookieDistribution where
  total : Nat
  tim : Nat
  mike : Nat
  fridge : Nat

/-- Calculates the number of cookies given to Anna --/
def cookiesForAnna (d : CookieDistribution) : Nat :=
  d.total - (d.tim + d.mike + d.fridge)

/-- Represents a ratio as a pair of natural numbers --/
structure Ratio where
  numerator : Nat
  denominator : Nat

/-- Theorem stating the ratio of cookies given to Anna to cookies given to Tim --/
theorem cookie_ratio_anna_to_tim (d : CookieDistribution)
  (h1 : d.total = 256)
  (h2 : d.tim = 15)
  (h3 : d.mike = 23)
  (h4 : d.fridge = 188) :
  Ratio.mk (cookiesForAnna d) d.tim = Ratio.mk 2 1 := by
  sorry

#check cookie_ratio_anna_to_tim

end cookie_ratio_anna_to_tim_l1467_146767


namespace morse_high_school_students_l1467_146721

/-- The number of seniors at Morse High School -/
def num_seniors : ℕ := 300

/-- The percentage of seniors with cars -/
def senior_car_percentage : ℚ := 40 / 100

/-- The percentage of seniors with motorcycles -/
def senior_motorcycle_percentage : ℚ := 5 / 100

/-- The percentage of lower grade students with cars -/
def lower_car_percentage : ℚ := 10 / 100

/-- The percentage of lower grade students with motorcycles -/
def lower_motorcycle_percentage : ℚ := 3 / 100

/-- The percentage of all students with either a car or a motorcycle -/
def total_vehicle_percentage : ℚ := 20 / 100

/-- The number of students in the lower grades -/
def num_lower_grades : ℕ := 1071

theorem morse_high_school_students :
  ∃ (total_students : ℕ),
    (num_seniors + num_lower_grades = total_students) ∧
    (↑num_seniors * senior_car_percentage + 
     ↑num_seniors * senior_motorcycle_percentage +
     ↑num_lower_grades * lower_car_percentage + 
     ↑num_lower_grades * lower_motorcycle_percentage : ℚ) = 
    ↑total_students * total_vehicle_percentage :=
by sorry

end morse_high_school_students_l1467_146721


namespace contractor_absence_l1467_146739

theorem contractor_absence (total_days : ℕ) (daily_pay : ℚ) (daily_fine : ℚ) (total_amount : ℚ) :
  total_days = 30 ∧
  daily_pay = 25 ∧
  daily_fine = (15/2) ∧
  total_amount = 685 →
  ∃ (days_worked days_absent : ℕ),
    days_worked + days_absent = total_days ∧
    daily_pay * days_worked - daily_fine * days_absent = total_amount ∧
    days_absent = 2 :=
by sorry

end contractor_absence_l1467_146739


namespace algebra_test_average_l1467_146765

theorem algebra_test_average (total_average : ℝ) (male_count : ℕ) (female_average : ℝ) (female_count : ℕ) :
  total_average = 90 →
  male_count = 8 →
  female_average = 92 →
  female_count = 28 →
  (total_average * (male_count + female_count) - female_average * female_count) / male_count = 83 := by
  sorry

end algebra_test_average_l1467_146765


namespace hamburgers_left_over_l1467_146741

-- Define the initial quantities and served items
def initial_hamburgers : ℕ := 25
def first_hour_hamburgers : ℕ := 12
def second_hour_hamburgers : ℕ := 6

-- Define the function to calculate remaining hamburgers
def remaining_hamburgers : ℕ := 
  initial_hamburgers - (first_hour_hamburgers + second_hour_hamburgers)

-- Theorem statement
theorem hamburgers_left_over : 
  remaining_hamburgers = 7 :=
by sorry

end hamburgers_left_over_l1467_146741


namespace teena_loe_distance_l1467_146764

theorem teena_loe_distance (teena_speed loe_speed : ℝ) (time : ℝ) (ahead_distance : ℝ) :
  teena_speed = 55 →
  loe_speed = 40 →
  time = 1.5 →
  ahead_distance = 15 →
  ∃ initial_distance : ℝ,
    initial_distance = (teena_speed * time - loe_speed * time - ahead_distance) ∧
    initial_distance = 7.5 := by
  sorry

end teena_loe_distance_l1467_146764


namespace current_batting_average_l1467_146728

/-- Represents a cricket player's batting statistics -/
structure BattingStats where
  matches_played : ℕ
  total_runs : ℕ

/-- Calculates the batting average -/
def batting_average (stats : BattingStats) : ℚ :=
  stats.total_runs / stats.matches_played

/-- The theorem statement -/
theorem current_batting_average 
  (current_stats : BattingStats)
  (next_match_runs : ℕ)
  (new_average : ℚ)
  (h1 : current_stats.matches_played = 6)
  (h2 : batting_average 
    ⟨current_stats.matches_played + 1, current_stats.total_runs + next_match_runs⟩ = new_average)
  (h3 : next_match_runs = 78)
  (h4 : new_average = 54)
  : batting_average current_stats = 50 := by
  sorry

end current_batting_average_l1467_146728


namespace limit_of_a_is_one_l1467_146752

def a : ℕ+ → ℚ
  | n => if n < 10000 then (2^(n.val+1)) / (2^n.val+1) else ((n.val+1)^2) / (n.val^2+1)

theorem limit_of_a_is_one :
  ∀ ε > 0, ∃ N : ℕ+, ∀ n ≥ N, |a n - 1| < ε :=
by sorry

end limit_of_a_is_one_l1467_146752


namespace sum_of_products_l1467_146710

theorem sum_of_products (x y z : ℝ) 
  (sum_condition : x + y + z = 20) 
  (sum_squares_condition : x^2 + y^2 + z^2 = 200) : 
  x*y + x*z + y*z = 100 := by sorry

end sum_of_products_l1467_146710


namespace perfect_square_factors_of_360_l1467_146714

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem perfect_square_factors_of_360 :
  let factorization := prime_factorization 360
  (factorization = [(2, 3), (3, 2), (5, 1)]) →
  count_perfect_square_factors 360 = 4 := by sorry

end perfect_square_factors_of_360_l1467_146714


namespace cubic_equation_sum_of_cubes_l1467_146729

theorem cubic_equation_sum_of_cubes (a b c : ℝ) : 
  (a - Real.rpow 20 (1/3 : ℝ)) * (a - Real.rpow 70 (1/3 : ℝ)) * (a - Real.rpow 170 (1/3 : ℝ)) = 1/2 →
  (b - Real.rpow 20 (1/3 : ℝ)) * (b - Real.rpow 70 (1/3 : ℝ)) * (b - Real.rpow 170 (1/3 : ℝ)) = 1/2 →
  (c - Real.rpow 20 (1/3 : ℝ)) * (c - Real.rpow 70 (1/3 : ℝ)) * (c - Real.rpow 170 (1/3 : ℝ)) = 1/2 →
  a ≠ b → b ≠ c → a ≠ c →
  a^3 + b^3 + c^3 = 260.5 := by
  sorry

end cubic_equation_sum_of_cubes_l1467_146729


namespace max_triangle_area_in_three_squares_l1467_146732

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in the plane -/
structure Square where
  center : Point
  side : ℝ

/-- Definition of a unit square -/
def isUnitSquare (s : Square) : Prop := s.side = 1

/-- Definition of a point being contained in a square -/
def isContainedIn (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side / 2 ∧ abs (p.y - s.center.y) ≤ s.side / 2

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (A B C : Point) : ℝ :=
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2

/-- The main theorem -/
theorem max_triangle_area_in_three_squares 
  (s₁ s₂ s₃ : Square) 
  (h₁ : isUnitSquare s₁) 
  (h₂ : isUnitSquare s₂) 
  (h₃ : isUnitSquare s₃) 
  (X : Point) 
  (hX₁ : isContainedIn X s₁) 
  (hX₂ : isContainedIn X s₂) 
  (hX₃ : isContainedIn X s₃) 
  (A B C : Point) 
  (hA : isContainedIn A s₁ ∨ isContainedIn A s₂ ∨ isContainedIn A s₃)
  (hB : isContainedIn B s₁ ∨ isContainedIn B s₂ ∨ isContainedIn B s₃)
  (hC : isContainedIn C s₁ ∨ isContainedIn C s₂ ∨ isContainedIn C s₃) :
  triangleArea A B C ≤ 3 * Real.sqrt 3 / 2 := by
  sorry

end max_triangle_area_in_three_squares_l1467_146732


namespace even_cube_diff_iff_even_sum_l1467_146776

theorem even_cube_diff_iff_even_sum (p q : ℕ) : 
  Even (p^3 - q^3) ↔ Even (p + q) := by sorry

end even_cube_diff_iff_even_sum_l1467_146776


namespace boxes_per_carton_l1467_146742

/-- Given a number of cartons per case, prove that there is 1 box per carton -/
theorem boxes_per_carton (c : ℕ) (h1 : c > 0) : ∃ (b : ℕ), b = 1 ∧ b * c * 400 = 400 := by
  sorry

end boxes_per_carton_l1467_146742


namespace zeros_properties_l1467_146707

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

-- State the theorem
theorem zeros_properties (k α β : ℝ) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < 2)
  (h4 : f k α = 0) (h5 : f k β = 0) :
  (-7/2 < k ∧ k < -1) ∧ (1/α + 1/β < 4) := by
  sorry

end zeros_properties_l1467_146707


namespace coordinates_wrt_x_axis_l1467_146719

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Theorem: The coordinates of point A(2,3) with respect to the x-axis are (2,-3) -/
theorem coordinates_wrt_x_axis :
  let A : Point := ⟨2, 3⟩
  reflectAcrossXAxis A = ⟨2, -3⟩ := by
  sorry

end coordinates_wrt_x_axis_l1467_146719


namespace smaller_number_value_l1467_146771

theorem smaller_number_value (s l : ℤ) : 
  (l - s = 28) → 
  (l + 13 = 2 * (s + 13)) → 
  s = 15 := by
sorry

end smaller_number_value_l1467_146771


namespace geometric_mean_minimum_l1467_146725

theorem geometric_mean_minimum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^x * 4^y)) :
  x^2 + 2*y^2 ≥ 1/3 :=
sorry

end geometric_mean_minimum_l1467_146725


namespace age_difference_l1467_146743

theorem age_difference (c d : ℕ) (hc : c < 10) (hd : d < 10) 
  (h : 10 * c + d + 10 = 3 * (10 * d + c + 10)) :
  (10 * c + d) - (10 * d + c) = 54 := by
  sorry

end age_difference_l1467_146743


namespace modified_grid_perimeter_l1467_146770

/-- Represents a square grid with a hole and an additional row on top. -/
structure ModifiedGrid :=
  (side : ℕ)
  (hole_size : ℕ)
  (top_row : ℕ)

/-- Calculates the perimeter of the modified grid. -/
def perimeter (grid : ModifiedGrid) : ℕ :=
  2 * (grid.side + grid.top_row) + 2 * grid.side - 2 * grid.hole_size

/-- Theorem stating that the perimeter of the specific modified 3x3 grid is 9. -/
theorem modified_grid_perimeter :
  ∃ (grid : ModifiedGrid), grid.side = 3 ∧ grid.hole_size = 1 ∧ grid.top_row = 3 ∧ perimeter grid = 9 :=
sorry

end modified_grid_perimeter_l1467_146770


namespace leap_year_1996_l1467_146717

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

theorem leap_year_1996 : is_leap_year 1996 := by
  sorry

end leap_year_1996_l1467_146717


namespace rectangle_area_l1467_146759

theorem rectangle_area (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end rectangle_area_l1467_146759


namespace f_zero_equals_two_l1467_146700

def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ x₃ x₄ x₅ : ℝ, f (x₁ + x₂ + x₃ + x₄ + x₅) = f x₁ + f x₂ + f x₃ + f x₄ + f x₅ - 8

theorem f_zero_equals_two (f : ℝ → ℝ) (h : f_property f) : f 0 = 2 := by
  sorry

end f_zero_equals_two_l1467_146700


namespace special_square_area_l1467_146709

/-- Square ABCD with points E on AD and F on BC, where BE = EF = FD = 20,
    AE = 2 * ED, and BF = 2 * FC -/
structure SpecialSquare where
  -- Define the side length of the square
  side : ℝ
  -- Define points E and F
  e : ℝ -- distance AE
  f : ℝ -- distance BF
  -- Conditions
  e_on_side : 0 < e ∧ e < side
  f_on_side : 0 < f ∧ f < side
  be_ef_fd : side - f + e = 20 -- BE + EF = 20
  ef_fd : e + side - f = 40 -- EF + FD = 40
  ae_twice_ed : e = 2 * (side - e)
  bf_twice_fc : f = 2 * (side - f)

/-- The area of the SpecialSquare is 720 -/
theorem special_square_area (sq : SpecialSquare) : sq.side ^ 2 = 720 :=
  sorry

end special_square_area_l1467_146709


namespace intersection_condition_implies_m_leq_neg_one_l1467_146749

/-- Given sets A and B, prove that if A ∩ B = A, then m ≤ -1 -/
theorem intersection_condition_implies_m_leq_neg_one (m : ℝ) : 
  let A : Set ℝ := {x | |x - 1| < 2}
  let B : Set ℝ := {x | x ≥ m}
  A ∩ B = A → m ≤ -1 := by
  sorry

end intersection_condition_implies_m_leq_neg_one_l1467_146749


namespace remainder_of_double_division_l1467_146705

theorem remainder_of_double_division (x : ℝ) : 
  let q₃ := (x^10 - 1) / (x - 1)
  let r₃ := x^10 - (x - 1) * q₃
  let q₄ := (q₃ - r₃) / (x - 1)
  let r₄ := q₃ - (x - 1) * q₄
  r₄ = 10 := by sorry

end remainder_of_double_division_l1467_146705


namespace minimum_balloons_l1467_146706

theorem minimum_balloons (red blue burst_red burst_blue : ℕ) : 
  red = 7 * blue →
  burst_red * 3 = burst_blue →
  burst_red ≥ 1 →
  burst_blue ≥ 1 →
  red + blue ≥ 24 :=
by sorry

end minimum_balloons_l1467_146706


namespace bookshelf_problem_l1467_146747

/-- Bookshelf purchasing problem -/
theorem bookshelf_problem (price_A price_B : ℕ) 
  (h1 : 3 * price_A + 2 * price_B = 1020)
  (h2 : 4 * price_A + 3 * price_B = 1440)
  (total_bookshelves : ℕ) (h3 : total_bookshelves = 20)
  (max_budget : ℕ) (h4 : max_budget = 4320) :
  (price_A = 180 ∧ price_B = 240) ∧ 
  (∃ (m : ℕ), 
    (m = 8 ∨ m = 9 ∨ m = 10) ∧
    (total_bookshelves - m ≥ m) ∧
    (price_A * m + price_B * (total_bookshelves - m) ≤ max_budget)) := by
  sorry

end bookshelf_problem_l1467_146747


namespace set_equality_implies_sum_of_powers_l1467_146766

theorem set_equality_implies_sum_of_powers (x y : ℝ) : 
  ({x, x * y, x + y} : Set ℝ) = ({0, |x|, y} : Set ℝ) → x^2018 + y^2018 = 2 := by
  sorry

end set_equality_implies_sum_of_powers_l1467_146766


namespace inequality_proof_l1467_146720

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a / c < b / c := by
  sorry

end inequality_proof_l1467_146720


namespace triangle_equilateral_l1467_146735

theorem triangle_equilateral (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides opposite to angles
  a = 2 * Real.sin (A / 2) ∧
  b = 2 * Real.sin (B / 2) ∧
  c = 2 * Real.sin (C / 2) →
  -- Arithmetic sequence condition
  2 * b = a + c →
  -- Geometric sequence condition
  (Real.sin B)^2 = (Real.sin A) * (Real.sin C) →
  -- Conclusion: triangle is equilateral
  a = b ∧ b = c := by
sorry

end triangle_equilateral_l1467_146735


namespace sin_beta_value_l1467_146751

theorem sin_beta_value (α β : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (α - β) = -5 / 13)
  (h4 : Real.sin α = 4 / 5) :
  Real.sin β = -56 / 65 := by
sorry

end sin_beta_value_l1467_146751


namespace area_of_region_is_5_25_l1467_146701

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The region defined by the given inequalities -/
def Region : Set Point :=
  {p : Point | p.y > 3 * p.x ∧ p.y > 5 - 2 * p.x ∧ p.y < 6}

/-- The area of the region -/
noncomputable def areaOfRegion : ℝ := sorry

/-- Theorem stating that the area of the region is 5.25 square units -/
theorem area_of_region_is_5_25 : areaOfRegion = 5.25 := by sorry

end area_of_region_is_5_25_l1467_146701


namespace november_december_revenue_ratio_l1467_146785

/-- Proves that the revenue in November is 2/5 of the revenue in December given the conditions --/
theorem november_december_revenue_ratio
  (revenue : Fin 3 → ℝ)  -- revenue function for 3 months (0: November, 1: December, 2: January)
  (h1 : revenue 2 = (1/5) * revenue 0)  -- January revenue is 1/5 of November revenue
  (h2 : revenue 1 = (25/6) * ((revenue 0 + revenue 2) / 2))  -- December revenue condition
  : revenue 0 = (2/5) * revenue 1 := by
  sorry

#check november_december_revenue_ratio

end november_december_revenue_ratio_l1467_146785


namespace square_less_than_triple_l1467_146777

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end square_less_than_triple_l1467_146777


namespace alternating_squares_sum_l1467_146733

theorem alternating_squares_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 242 := by
  sorry

end alternating_squares_sum_l1467_146733


namespace negation_equivalence_l1467_146796

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 + 2*x + 2 ≤ 0) ↔ 
  (∀ x : ℝ, x > 1 → x^2 + 2*x + 2 > 0) :=
by sorry

end negation_equivalence_l1467_146796


namespace unique_number_with_special_divisor_property_l1467_146750

theorem unique_number_with_special_divisor_property : 
  ∃! (N : ℕ), 
    N > 0 ∧ 
    (∃ (k : ℕ), 
      N + (Nat.factors N).foldl Nat.lcm 1 = 10^k) := by
  sorry

end unique_number_with_special_divisor_property_l1467_146750


namespace simplify_fraction_l1467_146746

theorem simplify_fraction : 18 * (8 / 15) * (1 / 12) = 18 / 5 := by
  sorry

end simplify_fraction_l1467_146746


namespace sum_of_squares_of_roots_l1467_146756

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 + 2 * p^2 - 5 * p - 8 = 0) →
  (3 * q^3 + 2 * q^2 - 5 * q - 8 = 0) →
  (3 * r^3 + 2 * r^2 - 5 * r - 8 = 0) →
  p^2 + q^2 + r^2 = 34/9 := by
  sorry

end sum_of_squares_of_roots_l1467_146756


namespace park_visitors_l1467_146769

theorem park_visitors (hikers bike_riders total : ℕ) : 
  hikers = 427 →
  hikers = bike_riders + 178 →
  total = hikers + bike_riders →
  total = 676 := by
sorry

end park_visitors_l1467_146769


namespace multiplication_puzzle_l1467_146702

theorem multiplication_puzzle :
  ∀ (A B C D : ℕ),
    A < 10 → B < 10 → C < 10 → D < 10 →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    C ≠ 0 → D ≠ 0 →
    100 * A + 10 * B + 1 = (10 * C + D) * (100 * C + D) →
    A + B = 3 := by
  sorry

end multiplication_puzzle_l1467_146702


namespace football_game_attendance_l1467_146787

/-- Football game attendance problem -/
theorem football_game_attendance 
  (saturday_attendance : ℕ)
  (monday_attendance : ℕ)
  (wednesday_attendance : ℕ)
  (friday_attendance : ℕ)
  (expected_total : ℕ)
  (actual_total : ℕ)
  (h1 : saturday_attendance = 80)
  (h2 : monday_attendance = saturday_attendance - 20)
  (h3 : wednesday_attendance > monday_attendance)
  (h4 : friday_attendance = saturday_attendance + monday_attendance)
  (h5 : expected_total = 350)
  (h6 : actual_total = expected_total + 40)
  (h7 : actual_total = saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance) :
  wednesday_attendance - monday_attendance = 50 := by
  sorry

end football_game_attendance_l1467_146787


namespace office_canteen_round_tables_l1467_146788

theorem office_canteen_round_tables :
  let rectangular_tables : ℕ := 2
  let chairs_per_round_table : ℕ := 6
  let chairs_per_rectangular_table : ℕ := 7
  let total_chairs : ℕ := 26
  
  ∃ (round_tables : ℕ),
    round_tables * chairs_per_round_table +
    rectangular_tables * chairs_per_rectangular_table = total_chairs ∧
    round_tables = 2 :=
by sorry

end office_canteen_round_tables_l1467_146788


namespace replaced_men_age_sum_l1467_146795

/-- Given a group of 8 men where replacing two of them with two women increases the average age by 2 years,
    and the average age of the women is 32 years, prove that the combined age of the two replaced men is 48 years. -/
theorem replaced_men_age_sum (n : ℕ) (A : ℝ) (women_avg_age : ℝ) :
  n = 8 ∧ women_avg_age = 32 →
  ∃ (older_man_age younger_man_age : ℝ),
    n * (A + 2) = (n - 2) * A + 2 * women_avg_age ∧
    older_man_age + younger_man_age = 48 :=
by sorry

end replaced_men_age_sum_l1467_146795


namespace roots_negative_of_each_other_l1467_146783

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots r and s, 
    if r = -s, then b = 0 -/
theorem roots_negative_of_each_other 
  (a b c r s : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * r^2 + b * r + c = 0) 
  (h3 : a * s^2 + b * s + c = 0) 
  (h4 : r = -s) : 
  b = 0 := by
sorry

end roots_negative_of_each_other_l1467_146783


namespace gcd_840_1764_l1467_146790

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l1467_146790


namespace inequality_solution_l1467_146789

def inequality (a x : ℝ) : Prop :=
  (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

theorem inequality_solution :
  (∀ a : ℝ, inequality a x) ↔ x = -2 ∨ x = 0 :=
sorry

end inequality_solution_l1467_146789


namespace percentage_of_a_l1467_146758

-- Define the four numbers
variable (a b c d : ℝ)

-- Define the conditions
def condition1 : Prop := a = 0.12 * b
def condition2 : Prop := b = 0.40 * c
def condition3 : Prop := c = 0.75 * d
def condition4 : Prop := d = 1.50 * (a + b)

-- Define the theorem
theorem percentage_of_a (h1 : condition1 a b) (h2 : condition2 b c) 
                        (h3 : condition3 c d) (h4 : condition4 a b d) :
  (a / (b + c + d)) * 100 = (1 / 43.166) * 100 := by
  sorry

end percentage_of_a_l1467_146758


namespace cannot_reach_54_from_12_l1467_146745

def Operation := Nat → Nat

def isValidOperation (op : Operation) : Prop :=
  ∀ n, (op n = 2 * n) ∨ (op n = 3 * n) ∨ (op n = n / 2) ∨ (op n = n / 3)

def applyOperations (ops : List Operation) (start : Nat) : Nat :=
  ops.foldl (λ acc op => op acc) start

theorem cannot_reach_54_from_12 :
  ¬ ∃ (ops : List Operation),
    (ops.length = 60) ∧
    (∀ op ∈ ops, isValidOperation op) ∧
    (applyOperations ops 12 = 54) :=
sorry

end cannot_reach_54_from_12_l1467_146745


namespace right_triangle_probability_l1467_146704

/-- A 3x3 grid with 16 vertices -/
structure Grid :=
  (vertices : Finset (ℕ × ℕ))
  (is_3x3 : vertices.card = 16)

/-- Three vertices from the grid -/
structure TripleOfVertices (g : Grid) :=
  (v₁ v₂ v₃ : ℕ × ℕ)
  (v₁_in : v₁ ∈ g.vertices)
  (v₂_in : v₂ ∈ g.vertices)
  (v₃_in : v₃ ∈ g.vertices)
  (distinct : v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃)

/-- Predicate to check if three vertices form a right triangle -/
def is_right_triangle (t : TripleOfVertices g) : Prop :=
  sorry

/-- The probability of forming a right triangle -/
def probability_right_triangle (g : Grid) : ℚ :=
  sorry

/-- The main theorem -/
theorem right_triangle_probability (g : Grid) :
  probability_right_triangle g = 9 / 35 :=
sorry

end right_triangle_probability_l1467_146704


namespace bridge_length_l1467_146774

/-- Given a train of length 120 meters traveling at 45 km/hr that crosses a bridge in 30 seconds,
    the length of the bridge is 255 meters. -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 120 ∧ train_speed_kmh = 45 ∧ crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 255 := by
  sorry

end bridge_length_l1467_146774


namespace increasing_odd_sum_nonpositive_l1467_146737

/-- A function f: ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- A function f: ℝ → ℝ is odd if for all x ∈ ℝ, f(-x) = -f(x) -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- Theorem: If f is an increasing and odd function on ℝ, and a and b are real numbers
    such that a + b ≤ 0, then f(a) + f(b) ≤ 0 -/
theorem increasing_odd_sum_nonpositive
  (f : ℝ → ℝ) (hf_inc : IsIncreasing f) (hf_odd : IsOdd f)
  (a b : ℝ) (hab : a + b ≤ 0) :
  f a + f b ≤ 0 := by
  sorry


end increasing_odd_sum_nonpositive_l1467_146737


namespace smallest_angle_is_180_div_7_l1467_146761

/-- An isosceles triangle that can be cut into two isosceles triangles -/
structure CuttableIsoscelesTriangle where
  /-- The measure of one of the equal angles in the original triangle -/
  α : ℝ
  /-- The original triangle is isosceles -/
  isosceles : α ≤ 90
  /-- The triangle can be cut into two isosceles triangles -/
  cuttable : ∃ (β γ : ℝ), (β = α ∧ γ = (180 - α) / 2) ∨ 
                           (β = (180 - α) / 2 ∧ γ = (180 - α) / 2) ∨
                           (β = 90 - α / 2 ∧ γ = α)

/-- The smallest angle in a CuttableIsoscelesTriangle is 180/7 -/
theorem smallest_angle_is_180_div_7 : 
  ∀ (t : CuttableIsoscelesTriangle), 
    min t.α (180 - 2 * t.α) ≥ 180 / 7 ∧ 
    ∃ (t : CuttableIsoscelesTriangle), min t.α (180 - 2 * t.α) = 180 / 7 := by
  sorry

end smallest_angle_is_180_div_7_l1467_146761


namespace function_symmetry_l1467_146786

theorem function_symmetry (f : ℝ → ℝ) (h : f 0 = 1) : f (4 - 4) = 1 := by
  sorry

end function_symmetry_l1467_146786


namespace quadratic_inequality_unique_solution_l1467_146740

theorem quadratic_inequality_unique_solution (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 + a*x + 5 ∧ x^2 + a*x + 5 ≤ 4) ↔ (a = 2 ∨ a = -2) := by
  sorry

end quadratic_inequality_unique_solution_l1467_146740


namespace n_has_24_digits_l1467_146755

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 12 -/
axiom n_div_12 : 12 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n^4 is a perfect fifth power -/
axiom n_fourth_fifth : ∃ k : ℕ, n^4 = k^5

/-- n is the smallest positive integer satisfying all conditions -/
axiom n_smallest : ∀ m : ℕ, m > 0 → (12 ∣ m) → (∃ k : ℕ, m^2 = k^3) → 
  (∃ k : ℕ, m^3 = k^2) → (∃ k : ℕ, m^4 = k^5) → m ≥ n

/-- Function to count the number of digits in a natural number -/
def digit_count (x : ℕ) : ℕ := sorry

/-- Theorem stating that n has 24 digits -/
theorem n_has_24_digits : digit_count n = 24 := sorry

end n_has_24_digits_l1467_146755


namespace article_profit_percentage_l1467_146780

theorem article_profit_percentage (CP : ℝ) (G : ℝ) : 
  CP = 800 →
  (CP * 0.95) * 1.1 = CP * (1 + G / 100) - 4 →
  G = 5 := by
sorry

end article_profit_percentage_l1467_146780


namespace bookstore_max_revenue_l1467_146712

/-- The revenue function for the bookstore -/
def revenue (p : ℝ) : ℝ := p * (150 - 6 * p)

/-- The maximum price allowed -/
def max_price : ℝ := 30

theorem bookstore_max_revenue :
  ∃ (p : ℝ), p ≤ max_price ∧
    ∀ (q : ℝ), q ≤ max_price → revenue q ≤ revenue p ∧
    p = 12.5 := by
  sorry

end bookstore_max_revenue_l1467_146712


namespace fred_final_card_count_l1467_146797

/-- Calculates the final number of baseball cards Fred has after a series of transactions. -/
def final_card_count (initial : ℕ) (given_away : ℕ) (traded : ℕ) (received : ℕ) : ℕ :=
  initial - given_away + received

/-- Proves that Fred ends up with 6 baseball cards after the given transactions. -/
theorem fred_final_card_count :
  final_card_count 5 2 1 3 = 6 := by
  sorry

end fred_final_card_count_l1467_146797


namespace sum_of_digits_theorem_l1467_146711

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- State the theorem
theorem sum_of_digits_theorem (n : ℕ) :
  sum_of_digits n = 351 → sum_of_digits (n + 1) = 352 := by sorry

end sum_of_digits_theorem_l1467_146711


namespace eastbound_plane_speed_l1467_146763

/-- Given two planes traveling in opposite directions, this theorem proves
    the speed of the eastbound plane given the conditions of the problem. -/
theorem eastbound_plane_speed
  (time : ℝ)
  (westbound_speed : ℝ)
  (total_distance : ℝ)
  (h_time : time = 3.5)
  (h_westbound : westbound_speed = 275)
  (h_distance : total_distance = 2100) :
  ∃ (eastbound_speed : ℝ),
    eastbound_speed = 325 ∧
    (eastbound_speed + westbound_speed) * time = total_distance :=
by sorry

end eastbound_plane_speed_l1467_146763


namespace hyperbola_asymptotes_l1467_146753

/-- Given a hyperbola with equation x²/8 - y²/2 = 1, its asymptotes have the equations y = ±(1/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 8 - y^2 / 2 = 1 →
  ∃ (k : ℝ), k = 1/2 ∧ (y = k * x ∨ y = -k * x) :=
by sorry

end hyperbola_asymptotes_l1467_146753


namespace no_solution_iff_v_eq_neg_one_l1467_146730

/-- The system of equations has no solution if and only if v = -1 -/
theorem no_solution_iff_v_eq_neg_one (v : ℝ) :
  (∀ x y z : ℝ, (x + y + z = v ∧ x + v*y + z = v ∧ x + y + v^2*z = v^2) → False) ↔ v = -1 :=
sorry

end no_solution_iff_v_eq_neg_one_l1467_146730


namespace discount_order_difference_l1467_146716

/-- Calculates the difference in final price when applying discounts in different orders -/
theorem discount_order_difference : 
  let original_price : ℚ := 30
  let fixed_discount : ℚ := 5
  let percentage_discount : ℚ := 0.25
  let scenario1 := (original_price - fixed_discount) * (1 - percentage_discount)
  let scenario2 := (original_price * (1 - percentage_discount)) - fixed_discount
  (scenario2 - scenario1) * 100 = 125 := by sorry

end discount_order_difference_l1467_146716


namespace tangent_line_at_one_tangent_lines_through_one_l1467_146718

noncomputable section

-- Define the function f(x) = x^3 + a*ln(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * Real.log x

-- Part I
theorem tangent_line_at_one (a : ℝ) (h : a = 1) :
  ∃ (m b : ℝ), m * 1 - f a 1 + b = 0 ∧
    ∀ x, m * x - (f a x) + b = 0 ↔ 4 * x - (f a x) - 3 = 0 :=
sorry

-- Part II
theorem tangent_lines_through_one (a : ℝ) (h : a = 0) :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (m₁ * 1 - f a 1 + b₁ = 0 ∧ ∀ x, m₁ * x - (f a x) + b₁ = 0 ↔ 3 * x - (f a x) - 2 = 0) ∧
    (m₂ * 1 - f a 1 + b₂ = 0 ∧ ∀ x, m₂ * x - (f a x) + b₂ = 0 ↔ 3 * x - 4 * (f a x) + 1 = 0) :=
sorry

end tangent_line_at_one_tangent_lines_through_one_l1467_146718


namespace digital_signal_probability_l1467_146723

theorem digital_signal_probability (p_receive_0_given_send_0 : ℝ) 
                                   (p_receive_1_given_send_0 : ℝ)
                                   (p_receive_1_given_send_1 : ℝ)
                                   (p_receive_0_given_send_1 : ℝ)
                                   (p_send_0 : ℝ)
                                   (p_send_1 : ℝ)
                                   (h1 : p_receive_0_given_send_0 = 0.9)
                                   (h2 : p_receive_1_given_send_0 = 0.1)
                                   (h3 : p_receive_1_given_send_1 = 0.95)
                                   (h4 : p_receive_0_given_send_1 = 0.05)
                                   (h5 : p_send_0 = 0.5)
                                   (h6 : p_send_1 = 0.5) :
  p_send_0 * p_receive_1_given_send_0 + p_send_1 * p_receive_1_given_send_1 = 0.525 :=
by sorry

end digital_signal_probability_l1467_146723


namespace coin_experiment_results_l1467_146762

/-- A fair coin is flipped 100 times with 48 heads observed. -/
structure CoinExperiment where
  total_flips : ℕ
  heads_count : ℕ
  is_fair : Bool
  h_total : total_flips = 100
  h_heads : heads_count = 48
  h_fair : is_fair = true

/-- The frequency of heads in a coin experiment. -/
def frequency (e : CoinExperiment) : ℚ :=
  e.heads_count / e.total_flips

/-- The theoretical probability of heads for a fair coin. -/
def fair_coin_probability : ℚ := 1 / 2

theorem coin_experiment_results (e : CoinExperiment) :
  frequency e = 48 / 100 ∧ fair_coin_probability = 1 / 2 := by
  sorry

#eval (48 : ℚ) / 100  -- To show that 48/100 evaluates to 0.48
#eval (1 : ℚ) / 2     -- To show that 1/2 evaluates to 0.5

end coin_experiment_results_l1467_146762


namespace expected_ones_three_dice_l1467_146703

/-- A standard die with 6 sides -/
def StandardDie : Type := Fin 6

/-- The probability of rolling a 1 on a standard die -/
def probOne : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a standard die -/
def probNotOne : ℚ := 5 / 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expectedOnes : ℚ := 1 / 2

/-- Theorem stating that the expected number of 1's when rolling three standard dice is 1/2 -/
theorem expected_ones_three_dice :
  (numDice : ℚ) * probOne = expectedOnes :=
sorry

end expected_ones_three_dice_l1467_146703


namespace rhombus_properties_l1467_146773

structure Rhombus where
  points : Fin 4 → ℝ × ℝ
  is_rhombus : ∀ i j : Fin 4, i ≠ j → dist (points i) (points j) = dist (points ((i+1) % 4)) (points ((j+1) % 4))

def diagonal1 (r : Rhombus) : ℝ × ℝ := r.points 0 - r.points 2
def diagonal2 (r : Rhombus) : ℝ × ℝ := r.points 1 - r.points 3

theorem rhombus_properties (r : Rhombus) :
  (∃ m : ℝ, diagonal1 r = m • (diagonal2 r)) ∧ 
  (diagonal1 r • diagonal2 r = 0) ∧
  (∀ i : Fin 4, dist (r.points i) (r.points ((i+1) % 4)) = dist (r.points ((i+1) % 4)) (r.points ((i+2) % 4))) ∧
  (¬ ∀ r : Rhombus, ‖diagonal1 r‖ = ‖diagonal2 r‖) := by
  sorry

#check rhombus_properties

end rhombus_properties_l1467_146773


namespace largest_divisor_of_p_squared_minus_q_squared_l1467_146781

theorem largest_divisor_of_p_squared_minus_q_squared (p q : ℤ) 
  (h_p_gt_q : p > q) 
  (h_p_odd : Odd p) 
  (h_q_even : Even q) : 
  (∀ (d : ℤ), d ∣ (p^2 - q^2) → d = 1 ∨ d = -1) :=
by sorry

end largest_divisor_of_p_squared_minus_q_squared_l1467_146781


namespace day_care_ratio_l1467_146724

/-- Proves that the initial ratio of toddlers to infants is 7:3 given the conditions of the day care problem. -/
theorem day_care_ratio (toddlers initial_infants : ℕ) : 
  toddlers = 42 →
  (toddlers : ℚ) / (initial_infants + 12 : ℚ) = 7 / 5 →
  (toddlers : ℚ) / (initial_infants : ℚ) = 7 / 3 :=
by sorry

end day_care_ratio_l1467_146724


namespace cabbage_production_increase_l1467_146726

theorem cabbage_production_increase (garden_size : ℕ) (this_year_production : ℕ) : 
  garden_size * garden_size = this_year_production →
  this_year_production = 9409 →
  this_year_production - (garden_size - 1) * (garden_size - 1) = 193 := by
sorry

end cabbage_production_increase_l1467_146726


namespace boys_between_rajan_and_vinay_l1467_146782

theorem boys_between_rajan_and_vinay (total_boys : ℕ) (rajan_position : ℕ) (vinay_position : ℕ)
  (h1 : total_boys = 24)
  (h2 : rajan_position = 6)
  (h3 : vinay_position = 10) :
  total_boys - (rajan_position - 1 + vinay_position - 1 + 2) = 8 :=
by sorry

end boys_between_rajan_and_vinay_l1467_146782


namespace m_range_l1467_146794

-- Define the propositions p and q
def p (m : ℝ) : Prop := m + 1 ≤ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

-- Theorem statement
theorem m_range (m : ℝ) : 
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (m ≤ -2 ∨ (-1 < m ∧ m < 2)) :=
sorry

end m_range_l1467_146794


namespace ceiling_neg_seven_fourths_squared_l1467_146734

theorem ceiling_neg_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end ceiling_neg_seven_fourths_squared_l1467_146734


namespace clients_using_radio_l1467_146792

/-- The number of clients using radio in an advertising agency with given client distribution. -/
theorem clients_using_radio (total : ℕ) (tv : ℕ) (mag : ℕ) (tv_mag : ℕ) (tv_radio : ℕ) (radio_mag : ℕ) (all_three : ℕ) :
  total = 180 →
  tv = 115 →
  mag = 130 →
  tv_mag = 85 →
  tv_radio = 75 →
  radio_mag = 95 →
  all_three = 80 →
  ∃ radio : ℕ, radio = 30 ∧ 
    total = tv + radio + mag - tv_mag - tv_radio - radio_mag + all_three :=
by
  sorry


end clients_using_radio_l1467_146792


namespace inequality_proof_l1467_146799

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b / (a + b)^2 + b * c / (b + c)^2 + c * a / (c + a)^2) + 
  3 * (a^2 + b^2 + c^2) / (a + b + c)^2 ≥ 7/4 := by
  sorry

end inequality_proof_l1467_146799


namespace mrs_brown_other_bills_value_l1467_146760

/-- Represents the utility bill payment scenario for Mrs. Brown -/
structure UtilityPayment where
  totalBill : ℕ
  tenDollarBillsUsed : ℕ
  tenDollarBillValue : ℕ

/-- Calculates the value of other bills used in the utility payment -/
def otherBillsValue (payment : UtilityPayment) : ℕ :=
  payment.totalBill - (payment.tenDollarBillsUsed * payment.tenDollarBillValue)

/-- Theorem stating that the value of other bills used by Mrs. Brown is $150 -/
theorem mrs_brown_other_bills_value :
  let payment : UtilityPayment := {
    totalBill := 170,
    tenDollarBillsUsed := 2,
    tenDollarBillValue := 10
  }
  otherBillsValue payment = 150 := by
  sorry

end mrs_brown_other_bills_value_l1467_146760
