import Mathlib

namespace NUMINAMATH_CALUDE_smartphone_price_difference_l2403_240372

def store_a_full_price : ℚ := 125
def store_a_discount : ℚ := 8 / 100
def store_b_full_price : ℚ := 130
def store_b_discount : ℚ := 10 / 100

theorem smartphone_price_difference :
  store_b_full_price * (1 - store_b_discount) - store_a_full_price * (1 - store_a_discount) = 2 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_difference_l2403_240372


namespace NUMINAMATH_CALUDE_cone_slant_height_l2403_240322

/-- Given a cone with base radius 2 cm and an unfolded side forming a sector
    with central angle 120°, prove that its slant height is 6 cm. -/
theorem cone_slant_height (r : ℝ) (θ : ℝ) (x : ℝ) 
    (h_r : r = 2)
    (h_θ : θ = 120)
    (h_arc_length : θ / 360 * (2 * Real.pi * x) = 2 * Real.pi * r) :
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l2403_240322


namespace NUMINAMATH_CALUDE_mean_sales_is_five_point_five_l2403_240344

def monday_sales : ℕ := 8
def tuesday_sales : ℕ := 3
def wednesday_sales : ℕ := 10
def thursday_sales : ℕ := 4
def friday_sales : ℕ := 4
def saturday_sales : ℕ := 4

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales
def number_of_days : ℕ := 6

theorem mean_sales_is_five_point_five :
  (total_sales : ℚ) / (number_of_days : ℚ) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_sales_is_five_point_five_l2403_240344


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l2403_240307

theorem largest_four_digit_congruent_to_17_mod_26 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n ≡ 17 [ZMOD 26] → n ≤ 9972 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l2403_240307


namespace NUMINAMATH_CALUDE_steve_initial_boxes_l2403_240380

/-- The number of boxes Steve had initially -/
def initial_boxes (pencils_per_box : ℕ) (pencils_to_lauren : ℕ) (pencils_to_matt_diff : ℕ) (pencils_left : ℕ) : ℕ :=
  (pencils_to_lauren + (pencils_to_lauren + pencils_to_matt_diff) + pencils_left) / pencils_per_box

theorem steve_initial_boxes :
  initial_boxes 12 6 3 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_steve_initial_boxes_l2403_240380


namespace NUMINAMATH_CALUDE_pencil_pen_difference_l2403_240330

/-- Given a ratio of pens to pencils and the number of pencils, 
    calculate the difference between pencils and pens. -/
theorem pencil_pen_difference 
  (ratio_pens ratio_pencils num_pencils : ℕ) 
  (h_ratio : ratio_pens < ratio_pencils)
  (h_pencils : num_pencils = 42)
  (h_prop : ratio_pens * num_pencils = ratio_pencils * (num_pencils - 7)) :
  num_pencils - (num_pencils - 7) = 7 := by
  sorry

#check pencil_pen_difference 5 6 42

end NUMINAMATH_CALUDE_pencil_pen_difference_l2403_240330


namespace NUMINAMATH_CALUDE_cricket_game_overs_l2403_240368

theorem cricket_game_overs (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) (remaining_overs : ℝ) :
  target = 282 →
  initial_rate = 3.2 →
  required_rate = 6.25 →
  remaining_overs = 40 →
  ∃ x : ℝ, x = 10 ∧ initial_rate * x + required_rate * remaining_overs = target :=
by sorry

end NUMINAMATH_CALUDE_cricket_game_overs_l2403_240368


namespace NUMINAMATH_CALUDE_office_network_connections_l2403_240362

/-- Represents a computer network with switches and connections -/
structure ComputerNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ
  num_crucial_switches : ℕ

/-- Calculates the total number of connections in the network -/
def total_connections (network : ComputerNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2 + network.num_crucial_switches

/-- Theorem: The total number of connections in the given network is 65 -/
theorem office_network_connections :
  let network : ComputerNetwork := {
    num_switches := 30,
    connections_per_switch := 4,
    num_crucial_switches := 5
  }
  total_connections network = 65 := by
  sorry

end NUMINAMATH_CALUDE_office_network_connections_l2403_240362


namespace NUMINAMATH_CALUDE_second_apartment_rent_l2403_240392

/-- Calculates the total monthly cost for an apartment --/
def total_monthly_cost (rent : ℚ) (utilities : ℚ) (miles_per_day : ℚ) (work_days : ℚ) (cost_per_mile : ℚ) : ℚ :=
  rent + utilities + (miles_per_day * work_days * cost_per_mile)

/-- The problem statement --/
theorem second_apartment_rent :
  let first_rent : ℚ := 800
  let first_utilities : ℚ := 260
  let first_miles : ℚ := 31
  let second_utilities : ℚ := 200
  let second_miles : ℚ := 21
  let work_days : ℚ := 20
  let cost_per_mile : ℚ := 58 / 100
  let cost_difference : ℚ := 76
  ∃ second_rent : ℚ,
    second_rent = 900 ∧
    total_monthly_cost first_rent first_utilities first_miles work_days cost_per_mile -
    total_monthly_cost second_rent second_utilities second_miles work_days cost_per_mile = cost_difference :=
by
  sorry


end NUMINAMATH_CALUDE_second_apartment_rent_l2403_240392


namespace NUMINAMATH_CALUDE_right_angle_sufficiency_not_necessity_l2403_240319

theorem right_angle_sufficiency_not_necessity (A B C : ℝ) :
  -- Triangle ABC exists
  (0 < A) ∧ (0 < B) ∧ (0 < C) ∧ (A + B + C = π) →
  -- 1. If angle C is 90°, then cos A + sin A = cos B + sin B
  (C = π / 2 → Real.cos A + Real.sin A = Real.cos B + Real.sin B) ∧
  -- 2. There exists a triangle where cos A + sin A = cos B + sin B, but angle C ≠ 90°
  ∃ (A' B' C' : ℝ), (0 < A') ∧ (0 < B') ∧ (0 < C') ∧ (A' + B' + C' = π) ∧
    (Real.cos A' + Real.sin A' = Real.cos B' + Real.sin B') ∧ (C' ≠ π / 2) :=
by sorry

end NUMINAMATH_CALUDE_right_angle_sufficiency_not_necessity_l2403_240319


namespace NUMINAMATH_CALUDE_can_form_triangle_l2403_240300

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a 
    triangle must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given lengths 5, 3, and 4 can form a triangle. -/
theorem can_form_triangle : triangle_inequality 5 3 4 := by
  sorry

end NUMINAMATH_CALUDE_can_form_triangle_l2403_240300


namespace NUMINAMATH_CALUDE_betta_fish_guppies_l2403_240359

/-- The number of guppies eaten by each betta fish per day -/
def guppies_per_betta : ℕ := sorry

/-- The number of guppies eaten by the moray eel per day -/
def moray_guppies : ℕ := 20

/-- The number of betta fish -/
def num_bettas : ℕ := 5

/-- The total number of guppies needed per day -/
def total_guppies : ℕ := 55

theorem betta_fish_guppies :
  guppies_per_betta = 7 ∧
  moray_guppies + num_bettas * guppies_per_betta = total_guppies :=
sorry

end NUMINAMATH_CALUDE_betta_fish_guppies_l2403_240359


namespace NUMINAMATH_CALUDE_fraction_value_l2403_240335

theorem fraction_value (x y : ℝ) (h1 : 4 < (2*x - 3*y) / (2*x + 3*y)) 
  (h2 : (2*x - 3*y) / (2*x + 3*y) < 8) (h3 : ∃ (n : ℤ), x/y = n) : 
  x/y = -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l2403_240335


namespace NUMINAMATH_CALUDE_log_equation_solution_l2403_240357

theorem log_equation_solution (a : ℝ) (h : a > 0) :
  Real.log a / Real.log 2 - 2 * Real.log 2 / Real.log a = 1 →
  a = 4 ∨ a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2403_240357


namespace NUMINAMATH_CALUDE_total_houses_l2403_240397

theorem total_houses (dogs : ℕ) (cats : ℕ) (both : ℕ) (h1 : dogs = 40) (h2 : cats = 30) (h3 : both = 10) :
  dogs + cats - both = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_houses_l2403_240397


namespace NUMINAMATH_CALUDE_no_four_primes_product_11_times_sum_l2403_240382

theorem no_four_primes_product_11_times_sum : 
  ¬ ∃ (a b c d : ℕ), 
    Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧
    (a * b * c * d = 11 * (a + b + c + d)) ∧
    ((a + b + c + d = 46) ∨ (a + b + c + d = 47) ∨ (a + b + c + d = 48)) :=
sorry

end NUMINAMATH_CALUDE_no_four_primes_product_11_times_sum_l2403_240382


namespace NUMINAMATH_CALUDE_cinema_hall_capacity_l2403_240389

/-- Represents a cinema hall with a given number of rows and seats per row -/
structure CinemaHall where
  rows : ℕ
  seatsPerRow : ℕ

/-- Calculates the approximate seating capacity of a cinema hall -/
def approximateCapacity (hall : CinemaHall) : ℕ :=
  900

/-- Calculates the actual seating capacity of a cinema hall -/
def actualCapacity (hall : CinemaHall) : ℕ :=
  hall.rows * hall.seatsPerRow

theorem cinema_hall_capacity (hall : CinemaHall) 
  (h1 : hall.rows = 28) 
  (h2 : hall.seatsPerRow = 31) : 
  approximateCapacity hall = 900 ∧ actualCapacity hall = 868 := by
  sorry

end NUMINAMATH_CALUDE_cinema_hall_capacity_l2403_240389


namespace NUMINAMATH_CALUDE_walk_a_thon_miles_difference_l2403_240321

theorem walk_a_thon_miles_difference 
  (last_year_rate : ℝ) 
  (this_year_rate : ℝ) 
  (last_year_total : ℝ) 
  (h1 : last_year_rate = 4)
  (h2 : this_year_rate = 2.75)
  (h3 : last_year_total = 44) :
  (last_year_total / this_year_rate) - (last_year_total / last_year_rate) = 5 :=
by sorry

end NUMINAMATH_CALUDE_walk_a_thon_miles_difference_l2403_240321


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l2403_240396

/-- Represents the number of ways to distribute books between the library and checked-out status. -/
def book_distribution_ways (n₁ n₂ : ℕ) : ℕ :=
  (n₁ - 1) * (n₂ - 1)

/-- Theorem stating the number of ways to distribute books between the library and checked-out status. -/
theorem book_distribution_theorem :
  let n₁ : ℕ := 8  -- number of copies of the first type of book
  let n₂ : ℕ := 4  -- number of copies of the second type of book
  book_distribution_ways n₁ n₂ = 21 :=
by
  sorry

#eval book_distribution_ways 8 4

end NUMINAMATH_CALUDE_book_distribution_theorem_l2403_240396


namespace NUMINAMATH_CALUDE_hexagonal_prism_diagonals_truncated_cube_diagonals_l2403_240304

/-- Calculate the number of diagonals in a polyhedron -/
def count_diagonals (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

theorem hexagonal_prism_diagonals :
  count_diagonals 12 3 = 18 := by sorry

theorem truncated_cube_diagonals :
  count_diagonals 24 10 = 120 := by sorry

end NUMINAMATH_CALUDE_hexagonal_prism_diagonals_truncated_cube_diagonals_l2403_240304


namespace NUMINAMATH_CALUDE_mice_on_bottom_path_l2403_240367

/-- Represents the number of mice in each house --/
structure MouseDistribution where
  left : ℕ
  top : ℕ
  right : ℕ

/-- The problem setup --/
def initial_distribution : MouseDistribution := ⟨8, 3, 7⟩
def final_distribution : MouseDistribution := ⟨5, 4, 9⟩

/-- The theorem to prove --/
theorem mice_on_bottom_path :
  let bottom_path_mice := 
    (initial_distribution.left + initial_distribution.right) -
    (final_distribution.left + final_distribution.right)
  bottom_path_mice = 11 := by
  sorry


end NUMINAMATH_CALUDE_mice_on_bottom_path_l2403_240367


namespace NUMINAMATH_CALUDE_age_problem_l2403_240390

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  b = 2 * c →  -- b is twice as old as c
  a + b + c = 17 →  -- The total of the ages of a, b, and c is 17
  b = 6 :=  -- Prove that b is 6 years old
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l2403_240390


namespace NUMINAMATH_CALUDE_eva_test_probability_l2403_240301

theorem eva_test_probability (p_history : ℝ) (p_geography : ℝ) 
  (h_history : p_history = 5/9)
  (h_geography : p_geography = 1/3)
  (h_independent : True) -- We don't need to define independence formally for this statement
  : (1 - p_history) * (1 - p_geography) = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_eva_test_probability_l2403_240301


namespace NUMINAMATH_CALUDE_prob_at_most_one_red_l2403_240331

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def red_balls : ℕ := 2
def drawn_balls : ℕ := 3

theorem prob_at_most_one_red :
  (1 : ℚ) - (Nat.choose white_balls 1 * Nat.choose red_balls 2 : ℚ) / Nat.choose total_balls drawn_balls = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_red_l2403_240331


namespace NUMINAMATH_CALUDE_team_performance_l2403_240352

theorem team_performance (total_games : ℕ) (total_points : ℕ) 
  (wins : ℕ) (draws : ℕ) (losses : ℕ) : 
  total_games = 38 →
  total_points = 80 →
  wins + draws + losses = total_games →
  3 * wins + draws = total_points →
  wins > 2 * draws →
  wins > 5 * losses →
  draws = 11 := by
sorry

end NUMINAMATH_CALUDE_team_performance_l2403_240352


namespace NUMINAMATH_CALUDE_prob_sum_7_or_11_l2403_240313

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The set of possible sums we're interested in -/
def target_sums : Set ℕ := {7, 11}

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of ways to get a sum of 7 or 11 -/
def favorable_outcomes : ℕ := 8

/-- The probability of rolling a sum of 7 or 11 with two dice -/
def probability_sum_7_or_11 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_7_or_11 : probability_sum_7_or_11 = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_7_or_11_l2403_240313


namespace NUMINAMATH_CALUDE_total_distance_is_11500_l2403_240314

/-- A right-angled triangle with sides XY, YZ, and ZX -/
structure RightTriangle where
  XY : ℝ
  ZX : ℝ
  YZ : ℝ
  right_angle : YZ^2 + ZX^2 = XY^2

/-- The total distance traveled in the triangle -/
def total_distance (t : RightTriangle) : ℝ :=
  t.XY + t.YZ + t.ZX

/-- Theorem: The total distance traveled in the given triangle is 11500 km -/
theorem total_distance_is_11500 :
  ∃ t : RightTriangle, t.XY = 5000 ∧ t.ZX = 4000 ∧ total_distance t = 11500 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_11500_l2403_240314


namespace NUMINAMATH_CALUDE_system_solution_l2403_240388

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 3 * y = -7) ∧ (5 * x + 6 * y = -20) ∧ 
  (x = -34/13) ∧ (y = -15/13) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2403_240388


namespace NUMINAMATH_CALUDE_short_stack_pancakes_l2403_240379

/-- The number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- The number of customers who ordered short stack -/
def short_stack_orders : ℕ := 9

/-- The number of customers who ordered big stack -/
def big_stack_orders : ℕ := 6

/-- The total number of pancakes needed -/
def total_pancakes : ℕ := 57

/-- The number of pancakes in a short stack -/
def short_stack : ℕ := 3

theorem short_stack_pancakes :
  short_stack * short_stack_orders + big_stack * big_stack_orders = total_pancakes :=
by sorry

end NUMINAMATH_CALUDE_short_stack_pancakes_l2403_240379


namespace NUMINAMATH_CALUDE_log_relation_l2403_240371

theorem log_relation (p q : ℝ) : 
  p = Real.log 192 / Real.log 5 → 
  q = Real.log 12 / Real.log 3 → 
  p = (q * (Real.log 12 / Real.log 3 + 8/3)) / (Real.log 5 / Real.log 3) := by
sorry

end NUMINAMATH_CALUDE_log_relation_l2403_240371


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2403_240327

def k : ℕ := 2010^2 + 2^2010

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := k) :
  (k^2 + 2^k) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2403_240327


namespace NUMINAMATH_CALUDE_pages_read_initially_l2403_240318

def book_chapters : ℕ := 8
def book_pages : ℕ := 95
def pages_read_later : ℕ := 25
def total_pages_read : ℕ := 62

theorem pages_read_initially : 
  total_pages_read - pages_read_later = 37 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_initially_l2403_240318


namespace NUMINAMATH_CALUDE_f_two_eq_zero_iff_r_eq_neg_38_l2403_240373

/-- The function f(x) as defined in the problem -/
def f (x r : ℝ) : ℝ := 2 * x^4 + x^3 + x^2 - 3 * x + r

/-- Theorem stating that f(2) = 0 if and only if r = -38 -/
theorem f_two_eq_zero_iff_r_eq_neg_38 : ∀ r : ℝ, f 2 r = 0 ↔ r = -38 := by sorry

end NUMINAMATH_CALUDE_f_two_eq_zero_iff_r_eq_neg_38_l2403_240373


namespace NUMINAMATH_CALUDE_five_by_five_uncoverable_l2403_240342

/-- Represents a rectangular board -/
structure Board where
  rows : ℕ
  cols : ℕ

/-- Represents a domino -/
structure Domino where
  width : ℕ
  height : ℕ

/-- Checks if a board can be completely covered by a given domino -/
def is_coverable (b : Board) (d : Domino) : Prop :=
  (b.rows * b.cols) % (d.width * d.height) = 0

/-- Theorem stating that a 5x5 board cannot be covered by 1x2 dominoes -/
theorem five_by_five_uncoverable :
  ¬ is_coverable (Board.mk 5 5) (Domino.mk 2 1) := by
  sorry

end NUMINAMATH_CALUDE_five_by_five_uncoverable_l2403_240342


namespace NUMINAMATH_CALUDE_a_2018_value_l2403_240364

def triangle_sequence (A : ℕ → ℝ) : ℕ → ℝ := λ n => A (n + 1) - A n

theorem a_2018_value (A : ℕ → ℝ) 
  (h1 : ∀ n, triangle_sequence (triangle_sequence A) n = 1)
  (h2 : A 18 = 0)
  (h3 : A 2017 = 0) :
  A 2018 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_a_2018_value_l2403_240364


namespace NUMINAMATH_CALUDE_triangle_number_assignment_l2403_240398

theorem triangle_number_assignment :
  ∀ (A B C D E F : ℕ),
    ({A, B, C, D, E, F} : Finset ℕ) = {1, 2, 3, 4, 5, 6} →
    B + D + E = 14 →
    C + E + F = 12 →
    A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_number_assignment_l2403_240398


namespace NUMINAMATH_CALUDE_sine_function_problem_l2403_240309

theorem sine_function_problem (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin x + b * x + c
  (f 0 = -2) → (f (Real.pi / 2) = 1) → (f (-Real.pi / 2) = -5) := by
  sorry

end NUMINAMATH_CALUDE_sine_function_problem_l2403_240309


namespace NUMINAMATH_CALUDE_katies_new_friends_games_l2403_240343

/-- The number of games Katie's new friends have -/
def new_friends_games (total_friends_games old_friends_games : ℕ) : ℕ :=
  total_friends_games - old_friends_games

/-- Theorem: Katie's new friends have 88 games -/
theorem katies_new_friends_games :
  new_friends_games 141 53 = 88 := by
  sorry

end NUMINAMATH_CALUDE_katies_new_friends_games_l2403_240343


namespace NUMINAMATH_CALUDE_alice_weekly_distance_l2403_240320

/-- Represents the walking distances for a single day --/
structure DailyWalk where
  morning : ℕ
  evening : ℕ

/-- Alice's walking schedule for the week --/
def aliceSchedule : List DailyWalk := [
  ⟨21, 0⟩,  -- Monday
  ⟨14, 0⟩,  -- Tuesday
  ⟨22, 0⟩,  -- Wednesday
  ⟨19, 0⟩,  -- Thursday
  ⟨20, 0⟩   -- Friday
]

/-- Calculates the total walking distance for a day --/
def totalDailyDistance (day : DailyWalk) : ℕ :=
  day.morning + day.evening

/-- Calculates the total walking distance for the week --/
def totalWeeklyDistance (schedule : List DailyWalk) : ℕ :=
  schedule.map totalDailyDistance |>.sum

/-- Theorem: Alice's total walking distance for the week is 96 miles --/
theorem alice_weekly_distance :
  totalWeeklyDistance aliceSchedule = 96 := by
  sorry

end NUMINAMATH_CALUDE_alice_weekly_distance_l2403_240320


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l2403_240306

theorem modulo_eleven_residue : (255 + 6 * 41 + 8 * 154 + 5 * 18) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l2403_240306


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l2403_240395

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel 
  (a b : Line) (α : Plane) 
  (h1 : perp a α) (h2 : perp b α) : 
  para a b := by sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l2403_240395


namespace NUMINAMATH_CALUDE_unit_vectors_collinear_with_AB_l2403_240354

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

def unitVectorAB : Set (ℝ × ℝ) := {v | ∃ k : ℝ, k ≠ 0 ∧ v = (k * AB.1, k * AB.2) ∧ v.1^2 + v.2^2 = 1}

theorem unit_vectors_collinear_with_AB :
  unitVectorAB = {(3/5, -4/5), (-3/5, 4/5)} :=
by sorry

end NUMINAMATH_CALUDE_unit_vectors_collinear_with_AB_l2403_240354


namespace NUMINAMATH_CALUDE_special_square_pt_length_l2403_240305

/-- Square with side length √2 and special folding property -/
structure SpecialSquare where
  -- Square side length
  side : ℝ
  side_eq : side = Real.sqrt 2
  -- Points T and U on sides PQ and RQ
  t : ℝ
  u : ℝ
  t_range : 0 < t ∧ t < side
  u_range : 0 < u ∧ u < side
  -- PT = QU
  pt_eq_qu : t = u
  -- Folding property: PS and RS coincide on diagonal QS
  folding : t * Real.sqrt 2 = side

/-- The length of PT in a SpecialSquare can be expressed as √8 - 2 -/
theorem special_square_pt_length (s : SpecialSquare) : s.t = Real.sqrt 8 - 2 := by
  sorry

#check special_square_pt_length

end NUMINAMATH_CALUDE_special_square_pt_length_l2403_240305


namespace NUMINAMATH_CALUDE_goldfish_fed_by_four_scoops_l2403_240363

/-- The number of goldfish that can be fed by one scoop of fish food -/
def goldfish_per_scoop : ℕ := 8

/-- The number of scoops of fish food -/
def number_of_scoops : ℕ := 4

/-- Theorem: 4 scoops of fish food can feed 32 goldfish -/
theorem goldfish_fed_by_four_scoops : 
  number_of_scoops * goldfish_per_scoop = 32 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_fed_by_four_scoops_l2403_240363


namespace NUMINAMATH_CALUDE_pencil_packs_l2403_240375

theorem pencil_packs (pencils_per_pack : ℕ) (pencils_per_row : ℕ) (num_rows : ℕ) : 
  pencils_per_pack = 24 →
  pencils_per_row = 16 →
  num_rows = 42 →
  (num_rows * pencils_per_row) / pencils_per_pack = 28 := by
sorry

end NUMINAMATH_CALUDE_pencil_packs_l2403_240375


namespace NUMINAMATH_CALUDE_problem_solution_l2403_240338

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 6)
  (h_eq2 : y + 1 / x = 30) :
  z + 1 / y = 38 / 179 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2403_240338


namespace NUMINAMATH_CALUDE_alexas_weight_l2403_240393

/-- Given the combined weight of two people and the weight of one person, 
    calculate the weight of the other person. -/
theorem alexas_weight (total_weight katerina_weight : ℕ) 
  (h1 : total_weight = 95)
  (h2 : katerina_weight = 49) :
  total_weight - katerina_weight = 46 := by
  sorry

end NUMINAMATH_CALUDE_alexas_weight_l2403_240393


namespace NUMINAMATH_CALUDE_invertible_function_property_l2403_240386

/-- Given an invertible function f, if f(b) = 3 and f(a) = b, then a - b = 2 -/
theorem invertible_function_property (f : ℝ → ℝ) (a b : ℝ) 
  (h_inv : Function.Injective f) 
  (h_fb : f b = 3) 
  (h_fa : f a = b) : 
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_invertible_function_property_l2403_240386


namespace NUMINAMATH_CALUDE_interchanged_digits_theorem_l2403_240303

/-- 
Given a two-digit number n = 10a + b, where n = 3(a + b),
prove that the number formed by interchanging its digits (10b + a) 
is equal to 8 times the sum of its digits (8(a + b)).
-/
theorem interchanged_digits_theorem (a b : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0)
  (h4 : 10 * a + b = 3 * (a + b)) :
  10 * b + a = 8 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_interchanged_digits_theorem_l2403_240303


namespace NUMINAMATH_CALUDE_gabes_original_seat_l2403_240326

/-- Represents the seats in the movie theater --/
inductive Seat
| one
| two
| three
| four
| five
| six
| seven

/-- Represents the friends --/
inductive Friend
| gabe
| flo
| dan
| cal
| bea
| eva
| hal

/-- Represents the seating arrangement --/
def Arrangement := Friend → Seat

/-- Returns the seat to the right of the given seat --/
def seatToRight (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.two
  | Seat.two => Seat.three
  | Seat.three => Seat.four
  | Seat.four => Seat.five
  | Seat.five => Seat.six
  | Seat.six => Seat.seven
  | Seat.seven => Seat.seven

/-- Returns the seat to the left of the given seat --/
def seatToLeft (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.one
  | Seat.two => Seat.one
  | Seat.three => Seat.two
  | Seat.four => Seat.three
  | Seat.five => Seat.four
  | Seat.six => Seat.five
  | Seat.seven => Seat.six

/-- Theorem stating Gabe's original seat --/
theorem gabes_original_seat (initial : Arrangement) (final : Arrangement) :
  (∀ (f : Friend), initial f ≠ initial Friend.gabe) →
  (final Friend.flo = seatToRight (seatToRight (seatToRight (initial Friend.flo)))) →
  (final Friend.dan = seatToLeft (initial Friend.dan)) →
  (final Friend.cal = initial Friend.cal) →
  (final Friend.bea = initial Friend.eva ∧ final Friend.eva = initial Friend.bea) →
  (final Friend.hal = seatToRight (initial Friend.gabe)) →
  (final Friend.gabe = Seat.one ∨ final Friend.gabe = Seat.seven) →
  initial Friend.gabe = Seat.three :=
by sorry


end NUMINAMATH_CALUDE_gabes_original_seat_l2403_240326


namespace NUMINAMATH_CALUDE_total_frog_eyes_l2403_240308

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 6

/-- The number of eyes each frog has -/
def eyes_per_frog : ℕ := 2

/-- Theorem: The total number of frog eyes in the pond is equal to the product of the number of frogs and the number of eyes per frog -/
theorem total_frog_eyes : num_frogs * eyes_per_frog = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_frog_eyes_l2403_240308


namespace NUMINAMATH_CALUDE_odd_as_difference_of_squares_l2403_240377

theorem odd_as_difference_of_squares (n : ℕ) : 2 * n + 1 = (n + 1)^2 - n^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_as_difference_of_squares_l2403_240377


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2403_240315

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  ∃ (min : ℝ), min = (3 / 2 + Real.sqrt 2) ∧ ∀ z w, z > 0 → w > 0 → 2 * z + w = 2 → 1 / z + 1 / w ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2403_240315


namespace NUMINAMATH_CALUDE_mean_of_four_numbers_with_given_variance_l2403_240324

/-- Given a set of four positive real numbers with a specific variance, prove that their mean is 2. -/
theorem mean_of_four_numbers_with_given_variance 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (pos₁ : 0 < x₁) (pos₂ : 0 < x₂) (pos₃ : 0 < x₃) (pos₄ : 0 < x₄)
  (variance_eq : (1/4) * (x₁^2 + x₂^2 + x₃^2 + x₄^2 - 16) = 
                 (1/4) * ((x₁ - (x₁ + x₂ + x₃ + x₄)/4)^2 + 
                          (x₂ - (x₁ + x₂ + x₃ + x₄)/4)^2 + 
                          (x₃ - (x₁ + x₂ + x₃ + x₄)/4)^2 + 
                          (x₄ - (x₁ + x₂ + x₃ + x₄)/4)^2)) :
  (x₁ + x₂ + x₃ + x₄) / 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_four_numbers_with_given_variance_l2403_240324


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2403_240334

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (a b c : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), b - a = t • (c - a) ∨ c - a = t • (b - a)

/-- The theorem states that if the given points are collinear, then p + q = 6. -/
theorem collinear_points_sum (p q : ℝ) :
  collinear (2, p, q) (p, 3, q) (p, q, 4) → p + q = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2403_240334


namespace NUMINAMATH_CALUDE_walnut_trees_in_park_l2403_240329

theorem walnut_trees_in_park (current : ℕ) (planted : ℕ) (total : ℕ) :
  current + planted = total →
  planted = 55 →
  total = 77 →
  current = 22 := by
sorry

end NUMINAMATH_CALUDE_walnut_trees_in_park_l2403_240329


namespace NUMINAMATH_CALUDE_intersection_when_a_is_3_range_of_a_when_intersection_empty_l2403_240339

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

-- Theorem for part 1
theorem intersection_when_a_is_3 :
  A 3 ∩ B = {x | -1 ≤ x ∧ x ≤ 1} ∪ {x | 4 ≤ x ∧ x ≤ 5} :=
by sorry

-- Theorem for part 2
theorem range_of_a_when_intersection_empty :
  ∀ a : ℝ, a > 0 → (A a ∩ B = ∅) → (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_3_range_of_a_when_intersection_empty_l2403_240339


namespace NUMINAMATH_CALUDE_garrison_reinforcement_departure_reinforcement_left_after_27_days_l2403_240358

/-- Represents the problem of determining when reinforcements left a garrison --/
theorem garrison_reinforcement_departure (initial_men : ℕ) (initial_days : ℕ) 
  (departed_men : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_provisions := initial_men * initial_days
  let remaining_men := initial_men - departed_men
  let x := (total_provisions - remaining_men * remaining_days) / initial_men
  x

/-- Proves that the reinforcements left after 27 days given the problem conditions --/
theorem reinforcement_left_after_27_days : 
  garrison_reinforcement_departure 400 31 200 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_garrison_reinforcement_departure_reinforcement_left_after_27_days_l2403_240358


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2403_240346

/-- Given a circle D with equation x^2 + 14x + y^2 - 8y = -64,
    prove that the sum of its center coordinates and radius is -2 -/
theorem circle_center_radius_sum :
  ∀ (c d s : ℝ),
  (∀ (x y : ℝ), x^2 + 14*x + y^2 - 8*y = -64 ↔ (x - c)^2 + (y - d)^2 = s^2) →
  c + d + s = -2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2403_240346


namespace NUMINAMATH_CALUDE_projection_a_onto_b_l2403_240361

def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (-2, 1)

theorem projection_a_onto_b :
  let proj_magnitude := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  proj_magnitude = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_projection_a_onto_b_l2403_240361


namespace NUMINAMATH_CALUDE_ellipse_focal_property_l2403_240316

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- The left focus of the ellipse -/
def leftFocus : ℝ × ℝ := (-3, 0)

/-- The left directrix of the ellipse -/
def leftDirectrix (x : ℝ) : Prop := x = -25/3

/-- A line passing through the left focus -/
def lineThroughFocus (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 3)

/-- Point D is to the right of the left focus -/
def pointD (a θ : ℝ) : Prop := a > -3

/-- The circle with MN as diameter passes through F₁ -/
def circleCondition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + 3) * (x₂ + 3) + y₁ * y₂ = 0

/-- The main theorem -/
theorem ellipse_focal_property (k a θ : ℝ) (x₁ y₁ x₂ y₂ xₘ xₙ yₘ yₙ : ℝ) :
  ellipse x₁ y₁ →
  ellipse x₂ y₂ →
  lineThroughFocus k x₁ y₁ →
  lineThroughFocus k x₂ y₂ →
  pointD a θ →
  leftDirectrix xₘ →
  leftDirectrix xₙ →
  circleCondition xₘ yₘ xₙ yₙ →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focal_property_l2403_240316


namespace NUMINAMATH_CALUDE_abc_mod_seven_l2403_240378

theorem abc_mod_seven (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
  (h1 : (a + 3*b + 2*c) % 7 = 2)
  (h2 : (2*a + b + 3*c) % 7 = 3)
  (h3 : (3*a + 2*b + c) % 7 = 5) :
  (a * b * c) % 7 = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_mod_seven_l2403_240378


namespace NUMINAMATH_CALUDE_mod_inverse_sum_l2403_240365

theorem mod_inverse_sum : ∃ (a b : ℤ), 
  (5 * a) % 35 = 1 ∧ 
  (15 * b) % 35 = 1 ∧ 
  (a + b) % 35 = 21 := by
  sorry

end NUMINAMATH_CALUDE_mod_inverse_sum_l2403_240365


namespace NUMINAMATH_CALUDE_smallest_integer_with_divisibility_condition_l2403_240391

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_integer_with_divisibility_condition : 
  ∀ n : ℕ, n > 0 →
  (∀ i ∈ Finset.range 31, i ≠ 23 ∧ i ≠ 24 → is_divisible n i) →
  ¬(is_divisible n 23) →
  ¬(is_divisible n 24) →
  n ≥ 2230928700 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_divisibility_condition_l2403_240391


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_line_not_perpendicular_no_perpendicular_line_perpendicular_intersection_perpendicular_l2403_240384

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (on_plane : Point → Plane → Prop)
variable (on_line : Point → Line → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line → Prop)

-- Define the given conditions
variable (l : Line) (α β γ : Plane)
variable (h_diff : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Statement 1
theorem perpendicular_planes_parallel_line :
  perpendicular α β → ∃ m : Line, line_in_plane m α ∧ parallel m β := by sorry

-- Statement 2
theorem not_perpendicular_no_perpendicular_line :
  ¬perpendicular α β → ¬∃ m : Line, line_in_plane m α ∧ line_perpendicular_plane m β := by sorry

-- Statement 3
theorem perpendicular_intersection_perpendicular :
  perpendicular α γ → perpendicular β γ → intersection α β l → line_perpendicular_plane l γ := by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_line_not_perpendicular_no_perpendicular_line_perpendicular_intersection_perpendicular_l2403_240384


namespace NUMINAMATH_CALUDE_xy_value_l2403_240356

theorem xy_value (x y : ℝ) (h : (x + 22) / y + 290 / (x * y) = (26 - y) / x) :
  x * y = -143 := by sorry

end NUMINAMATH_CALUDE_xy_value_l2403_240356


namespace NUMINAMATH_CALUDE_necklaces_theorem_l2403_240350

/-- The number of necklaces Charlene made initially -/
def total_necklaces : ℕ := sorry

/-- The number of necklaces Charlene sold at the craft fair -/
def sold_necklaces : ℕ := 16

/-- The number of necklaces Charlene gave to her friends -/
def given_necklaces : ℕ := 18

/-- The number of necklaces Charlene has left -/
def left_necklaces : ℕ := 26

/-- Theorem stating that the total number of necklaces is equal to the sum of sold, given, and left necklaces -/
theorem necklaces_theorem : total_necklaces = sold_necklaces + given_necklaces + left_necklaces := by
  sorry

end NUMINAMATH_CALUDE_necklaces_theorem_l2403_240350


namespace NUMINAMATH_CALUDE_certain_number_value_l2403_240345

theorem certain_number_value : ∀ (t k certain_number : ℝ),
  t = 5 / 9 * (k - certain_number) →
  t = 75 →
  k = 167 →
  certain_number = 32 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_l2403_240345


namespace NUMINAMATH_CALUDE_inequality_proof_l2403_240325

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (1 + 2 * a)) + (1 / (1 + 2 * b)) + (1 / (1 + 2 * c)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2403_240325


namespace NUMINAMATH_CALUDE_cube_construction_count_l2403_240310

/-- Represents a rotation of a cube -/
structure CubeRotation where
  fixedConfigurations : ℕ

/-- The group of rotations for a cube -/
def rotationGroup : Finset CubeRotation := sorry

/-- The number of distinct ways to construct the cube -/
def distinctConstructions : ℕ := sorry

theorem cube_construction_count :
  distinctConstructions = 54 := by sorry

end NUMINAMATH_CALUDE_cube_construction_count_l2403_240310


namespace NUMINAMATH_CALUDE_inequality_proof_l2403_240332

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 4)
  (h2 : c^2 + d^2 = 16) : 
  a*c + b*d ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2403_240332


namespace NUMINAMATH_CALUDE_fraction_meaningfulness_l2403_240385

def is_meaningful (x : ℝ) : Prop := x + 3 ≠ 0

theorem fraction_meaningfulness (x : ℝ) : is_meaningful x ↔ x ≠ -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningfulness_l2403_240385


namespace NUMINAMATH_CALUDE_fourth_root_of_fourth_power_l2403_240369

theorem fourth_root_of_fourth_power (a : ℝ) (h : a < 2) :
  (((a - 2) ^ 4) ^ (1/4 : ℝ)) = 2 - a := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_fourth_power_l2403_240369


namespace NUMINAMATH_CALUDE_pentagon_area_is_14_l2403_240347

/-- Represents a trapezoid segmented into two triangles and a pentagon -/
structure SegmentedTrapezoid where
  triangle1_area : ℝ
  triangle2_area : ℝ
  base_ratio : ℝ
  total_area : ℝ

/-- The area of the pentagon in a segmented trapezoid -/
def pentagon_area (t : SegmentedTrapezoid) : ℝ :=
  t.total_area - t.triangle1_area - t.triangle2_area

/-- Theorem stating that the area of the pentagon is 14 under given conditions -/
theorem pentagon_area_is_14 (t : SegmentedTrapezoid) 
  (h1 : t.triangle1_area = 8)
  (h2 : t.triangle2_area = 18)
  (h3 : t.base_ratio = 2)
  (h4 : t.total_area = 40) :
  pentagon_area t = 14 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_area_is_14_l2403_240347


namespace NUMINAMATH_CALUDE_binomial_distribution_p_value_l2403_240340

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: For a binomial distribution with E(ξ) = 7 and D(ξ) = 6, p = 1/7 -/
theorem binomial_distribution_p_value (ξ : BinomialDistribution) 
  (h_exp : expectedValue ξ = 7)
  (h_var : variance ξ = 6) : 
  ξ.p = 1/7 := by
  sorry


end NUMINAMATH_CALUDE_binomial_distribution_p_value_l2403_240340


namespace NUMINAMATH_CALUDE_function_bounds_l2403_240317

theorem function_bounds (k : ℕ) (f : ℕ → ℕ) 
  (h_increasing : ∀ m n, m < n → f m < f n)
  (h_composition : ∀ n, f (f n) = k * n) :
  ∀ n : ℕ, (2 * k : ℚ) / (k + 1 : ℚ) * n ≤ f n ∧ (f n : ℚ) ≤ (k + 1 : ℚ) / 2 * n :=
by sorry

end NUMINAMATH_CALUDE_function_bounds_l2403_240317


namespace NUMINAMATH_CALUDE_factorization_equality_l2403_240351

theorem factorization_equality (x₁ x₂ : ℝ) :
  x₁^3 - 2*x₁^2*x₂ - x₁ + 2*x₂ = (x₁ - 1) * (x₁ + 1) * (x₁ - 2*x₂) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2403_240351


namespace NUMINAMATH_CALUDE_four_blocks_in_six_by_six_grid_l2403_240312

theorem four_blocks_in_six_by_six_grid : 
  let n : ℕ := 6
  let k : ℕ := 4
  let grid_size := n * n
  let combinations := (n.choose k) * (n.choose k) * (k.factorial)
  combinations = 5400 := by
  sorry

end NUMINAMATH_CALUDE_four_blocks_in_six_by_six_grid_l2403_240312


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_two_l2403_240349

theorem circle_area_with_diameter_two (π : Real) : Real :=
  let diameter : Real := 2
  let radius : Real := diameter / 2
  let area : Real := π * radius^2
  area

#check circle_area_with_diameter_two

end NUMINAMATH_CALUDE_circle_area_with_diameter_two_l2403_240349


namespace NUMINAMATH_CALUDE_total_spears_l2403_240383

/-- The number of spears that can be made from a sapling -/
def spears_per_sapling : ℕ := 3

/-- The number of spears that can be made from a log -/
def spears_per_log : ℕ := 9

/-- The number of spears that can be made from a bundle of branches -/
def spears_per_bundle : ℕ := 7

/-- The number of spears that can be made from a large tree trunk -/
def spears_per_trunk : ℕ := 15

/-- The number of saplings Marcy has -/
def num_saplings : ℕ := 6

/-- The number of logs Marcy has -/
def num_logs : ℕ := 1

/-- The number of bundles of branches Marcy has -/
def num_bundles : ℕ := 3

/-- The number of large tree trunks Marcy has -/
def num_trunks : ℕ := 2

/-- Theorem stating the total number of spears Marcy can make -/
theorem total_spears : 
  num_saplings * spears_per_sapling + 
  num_logs * spears_per_log + 
  num_bundles * spears_per_bundle + 
  num_trunks * spears_per_trunk = 78 := by
  sorry


end NUMINAMATH_CALUDE_total_spears_l2403_240383


namespace NUMINAMATH_CALUDE_condition_relationship_l2403_240323

theorem condition_relationship (x : ℝ) :
  (∀ x, x > 2 → x^2 > 4) ∧ 
  (∃ x, x^2 > 4 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2403_240323


namespace NUMINAMATH_CALUDE_remainder_problem_l2403_240302

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1023 % d = r) (h3 : 1386 % d = r) (h4 : 2151 % d = r) : 
  d - r = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2403_240302


namespace NUMINAMATH_CALUDE_alice_bob_calculation_l2403_240355

theorem alice_bob_calculation (x : ℕ) : 
  let alice_result := ((x + 2) * 2 + 3)
  2 * (alice_result + 3) = 4 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_calculation_l2403_240355


namespace NUMINAMATH_CALUDE_divisibility_of_factorial_l2403_240387

def f (n : ℕ) : ℕ := (n / 7) + (n / 49) + (n / 343) + (n / 2401)

theorem divisibility_of_factorial (n : ℕ) :
  (∀ m : ℕ, m > 0 → ¬(7^399 ∣ m.factorial ∧ ¬(7^400 ∣ m.factorial))) ∧
  ({m : ℕ | 7^400 ∣ m.factorial ∧ ¬(7^401 ∣ m.factorial)} = {2401, 2402, 2403, 2404, 2405, 2406, 2407}) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_factorial_l2403_240387


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2403_240376

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (1 + 3 * Complex.I) / (3 - Complex.I)
  Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2403_240376


namespace NUMINAMATH_CALUDE_polar_equation_is_parabola_l2403_240311

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.sin θ)

-- Define the Cartesian equation of a parabola
def is_parabola (x y : ℝ) : Prop :=
  ∃ (a : ℝ), x^2 = 2 * a * y

-- Theorem statement
theorem polar_equation_is_parabola :
  ∀ (x y : ℝ), (∃ (r θ : ℝ), polar_equation r θ ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  is_parabola x y :=
sorry

end NUMINAMATH_CALUDE_polar_equation_is_parabola_l2403_240311


namespace NUMINAMATH_CALUDE_ticket_difference_l2403_240336

/-- Represents the number of tickets sold for each category -/
structure TicketSales where
  vip : ℕ
  premium : ℕ
  general : ℕ

/-- Checks if the given ticket sales satisfy the problem conditions -/
def satisfiesConditions (sales : TicketSales) : Prop :=
  sales.vip + sales.premium + sales.general = 420 ∧
  50 * sales.vip + 30 * sales.premium + 10 * sales.general = 12000

/-- Theorem stating the difference between general admission and VIP tickets -/
theorem ticket_difference (sales : TicketSales) 
  (h : satisfiesConditions sales) : 
  sales.general - sales.vip = 30 := by
  sorry

end NUMINAMATH_CALUDE_ticket_difference_l2403_240336


namespace NUMINAMATH_CALUDE_semicircle_sum_limit_l2403_240333

/-- The sum of areas of n semicircles on a diameter approaches 0 as n approaches infinity -/
theorem semicircle_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * D^2 / (8 * n^2)) - 0| < ε :=
sorry

end NUMINAMATH_CALUDE_semicircle_sum_limit_l2403_240333


namespace NUMINAMATH_CALUDE_haploid_corn_triploid_watermelon_heritable_variation_l2403_240341

-- Define the sources of heritable variations
inductive HeritableVariationSource
  | GeneMutation
  | ChromosomalVariation
  | GeneRecombination

-- Define a structure for crop variations
structure CropVariation where
  name : String
  isChromosomalVariation : Bool

-- Define the property of being a heritable variation
def isHeritableVariation (source : HeritableVariationSource) : Prop :=
  match source with
  | HeritableVariationSource.GeneMutation => True
  | HeritableVariationSource.ChromosomalVariation => True
  | HeritableVariationSource.GeneRecombination => True

-- Theorem statement
theorem haploid_corn_triploid_watermelon_heritable_variation 
  (haploidCorn triploidWatermelon : CropVariation)
  (haploidCornChromosomal : haploidCorn.isChromosomalVariation = true)
  (triploidWatermelonChromosomal : triploidWatermelon.isChromosomalVariation = true) :
  isHeritableVariation HeritableVariationSource.ChromosomalVariation := by
  sorry


end NUMINAMATH_CALUDE_haploid_corn_triploid_watermelon_heritable_variation_l2403_240341


namespace NUMINAMATH_CALUDE_alfonso_daily_earnings_l2403_240370

def helmet_cost : ℕ := 340
def savings : ℕ := 40
def days_per_week : ℕ := 5
def total_weeks : ℕ := 10

def total_working_days : ℕ := days_per_week * total_weeks

def additional_savings_needed : ℕ := helmet_cost - savings

theorem alfonso_daily_earnings :
  additional_savings_needed / total_working_days = 6 :=
by sorry

end NUMINAMATH_CALUDE_alfonso_daily_earnings_l2403_240370


namespace NUMINAMATH_CALUDE_tom_drives_12_miles_l2403_240337

/-- A car race between Karen and Tom -/
structure CarRace where
  karen_speed : ℝ  -- Karen's speed in mph
  tom_speed : ℝ    -- Tom's speed in mph
  karen_delay : ℝ  -- Karen's delay in minutes
  win_margin : ℝ   -- Karen's winning margin in miles

/-- Calculate the distance Tom drives before Karen wins -/
def distance_tom_drives (race : CarRace) : ℝ :=
  sorry

/-- Theorem stating that Tom drives 12 miles before Karen wins -/
theorem tom_drives_12_miles (race : CarRace) 
  (h1 : race.karen_speed = 60)
  (h2 : race.tom_speed = 45)
  (h3 : race.karen_delay = 4)
  (h4 : race.win_margin = 4) :
  distance_tom_drives race = 12 :=
sorry

end NUMINAMATH_CALUDE_tom_drives_12_miles_l2403_240337


namespace NUMINAMATH_CALUDE_exists_n_divisors_n_factorial_divisible_by_2019_l2403_240353

theorem exists_n_divisors_n_factorial_divisible_by_2019 :
  ∃ n : ℕ+, (2019 : ℕ) ∣ (Nat.card (Nat.divisors (Nat.factorial n))) := by
  sorry

end NUMINAMATH_CALUDE_exists_n_divisors_n_factorial_divisible_by_2019_l2403_240353


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l2403_240381

/-- The number of ways to distribute students into classes -/
def distribute_students (total_students : ℕ) (num_classes : ℕ) (must_be_together : ℕ) : ℕ :=
  sorry

/-- The theorem to prove -/
theorem student_distribution_theorem :
  distribute_students 5 3 2 = 36 :=
sorry

end NUMINAMATH_CALUDE_student_distribution_theorem_l2403_240381


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l2403_240374

theorem base_2_representation_of_123 : 
  (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l2403_240374


namespace NUMINAMATH_CALUDE_total_playing_hours_l2403_240328

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of hours Nathan plays per day -/
def nathan_hours_per_day : ℕ := 3

/-- The number of weeks Nathan plays -/
def nathan_weeks : ℕ := 2

/-- The number of hours Tobias plays per day -/
def tobias_hours_per_day : ℕ := 5

/-- The number of weeks Tobias plays -/
def tobias_weeks : ℕ := 1

/-- The total number of hours Nathan and Tobias played -/
def total_hours : ℕ := 
  nathan_hours_per_day * days_per_week * nathan_weeks + 
  tobias_hours_per_day * days_per_week * tobias_weeks

theorem total_playing_hours : total_hours = 77 := by
  sorry

end NUMINAMATH_CALUDE_total_playing_hours_l2403_240328


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l2403_240399

-- Define the function f
def f (x : ℝ) : ℝ := x

-- Theorem stating that f satisfies all conditions
theorem f_satisfies_conditions :
  (∀ x : ℝ, f (-x) + f x = 0) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0) :=
by sorry

#check f_satisfies_conditions

end NUMINAMATH_CALUDE_f_satisfies_conditions_l2403_240399


namespace NUMINAMATH_CALUDE_trees_died_l2403_240394

/-- Proof that 15 trees died in the park --/
theorem trees_died (initial : ℕ) (cut : ℕ) (remaining : ℕ) (died : ℕ) : 
  initial = 86 → cut = 23 → remaining = 48 → died = initial - cut - remaining → died = 15 := by
  sorry

end NUMINAMATH_CALUDE_trees_died_l2403_240394


namespace NUMINAMATH_CALUDE_only_solutions_for_equation_l2403_240366

theorem only_solutions_for_equation (x p n : ℕ) : 
  Prime p → 2 * x * (x + 5) = p^n + 3 * (x - 1) → 
  ((x = 2 ∧ p = 5 ∧ n = 2) ∨ (x = 0 ∧ p = 3 ∧ n = 1)) := by
  sorry

end NUMINAMATH_CALUDE_only_solutions_for_equation_l2403_240366


namespace NUMINAMATH_CALUDE_triangle_side_length_l2403_240360

theorem triangle_side_length 
  (A B C : ℝ) 
  (hBC : Real.cos C = -Real.sqrt 2 / 2) 
  (hAC : Real.sin A / Real.sin B = 1 / (2 * Real.cos (A + B))) 
  (hBA : B * A = 2 * Real.sqrt 2) : 
  Real.sqrt ((Real.sin A)^2 + (Real.sin B)^2 - 2 * Real.sin A * Real.sin B * Real.cos C) = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2403_240360


namespace NUMINAMATH_CALUDE_cubic_integer_roots_l2403_240348

/-- Represents a cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ

/-- Counts the number of integer roots of a cubic polynomial, including multiplicity -/
def count_integer_roots (p : CubicPolynomial) : ℕ := sorry

/-- Theorem stating that the number of integer roots of a cubic polynomial with integer coefficients is 0, 1, 2, or 3 -/
theorem cubic_integer_roots (p : CubicPolynomial) :
  count_integer_roots p = 0 ∨ count_integer_roots p = 1 ∨ count_integer_roots p = 2 ∨ count_integer_roots p = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_integer_roots_l2403_240348
