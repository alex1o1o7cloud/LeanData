import Mathlib

namespace certain_value_problem_l319_31947

theorem certain_value_problem (x y : ℝ) : x = 69 ∧ x - 18 = 3 * (y - x) → y = 86 := by
  sorry

end certain_value_problem_l319_31947


namespace one_match_among_withdrawn_l319_31995

/-- Represents a table tennis tournament with special conditions -/
structure TableTennisTournament where
  n : ℕ  -- Total number of players
  total_matches : ℕ  -- Total number of matches played
  withdrawn_players : ℕ  -- Number of players who withdrew
  matches_per_withdrawn : ℕ  -- Number of matches each withdrawn player played
  hwithdrawncond : withdrawn_players = 3
  hmatchescond : matches_per_withdrawn = 2
  htotalcond : total_matches = 50

/-- The number of matches played among the withdrawn players -/
def matches_among_withdrawn (t : TableTennisTournament) : ℕ := 
  (t.withdrawn_players * t.matches_per_withdrawn - 
   t.total_matches + (t.n - t.withdrawn_players).choose 2) / 2

/-- Theorem stating that exactly one match was played among the withdrawn players -/
theorem one_match_among_withdrawn (t : TableTennisTournament) : 
  matches_among_withdrawn t = 1 := by
  sorry

end one_match_among_withdrawn_l319_31995


namespace suzanna_bike_ride_l319_31994

/-- Suzanna's bike ride problem -/
theorem suzanna_bike_ride (speed : ℝ) (total_time : ℝ) (break_time : ℝ) (distance : ℝ) : 
  speed = 2 / 10 → 
  total_time = 30 → 
  break_time = 5 → 
  distance = speed * (total_time - break_time) → 
  distance = 5 := by
  sorry

end suzanna_bike_ride_l319_31994


namespace jasons_points_theorem_l319_31993

/-- Calculates the total points Jason has from seashells and starfish -/
def jasons_total_points (initial_seashells : ℕ) (initial_starfish : ℕ) 
  (seashell_points : ℕ) (starfish_points : ℕ)
  (seashells_given_tim : ℕ) (seashells_given_lily : ℕ)
  (seashells_found : ℕ) (seashells_lost : ℕ) : ℕ :=
  let initial_points := initial_seashells * seashell_points + initial_starfish * starfish_points
  let points_given_away := (seashells_given_tim + seashells_given_lily) * seashell_points
  let net_points_found_lost := (seashells_found - seashells_lost) * seashell_points
  initial_points - points_given_away + net_points_found_lost

theorem jasons_points_theorem :
  jasons_total_points 49 48 2 3 13 7 15 5 = 222 := by
  sorry

end jasons_points_theorem_l319_31993


namespace solution_set_f_greater_than_5_range_of_a_no_solution_l319_31928

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |2*x + 1|

-- Theorem for the solution of f(x) > 5
theorem solution_set_f_greater_than_5 :
  {x : ℝ | f x > 5} = Set.Iio (-4/3) ∪ Set.Ioi 2 := by sorry

-- Theorem for the range of a when 1/(f(x)-4) = a has no solution
theorem range_of_a_no_solution :
  {a : ℝ | ∀ x, 1/(f x - 4) ≠ a} = Set.Ioo (-2/3) 0 := by sorry

end solution_set_f_greater_than_5_range_of_a_no_solution_l319_31928


namespace root_difference_of_quadratic_l319_31913

theorem root_difference_of_quadratic (r₁ r₂ : ℝ) : 
  r₁^2 - 9*r₁ + 14 = 0 → 
  r₂^2 - 9*r₂ + 14 = 0 → 
  r₁ + r₂ = r₁ * r₂ → 
  |r₁ - r₂| = 5 := by
sorry

end root_difference_of_quadratic_l319_31913


namespace population_difference_specific_population_difference_l319_31977

/-- The population difference between two cities with different volumes, given a constant population density. -/
theorem population_difference (density : ℕ) (volume1 volume2 : ℕ) :
  density * (volume1 - volume2) = density * volume1 - density * volume2 := by
  sorry

/-- The population difference between two specific cities. -/
theorem specific_population_difference :
  let density : ℕ := 80
  let volume1 : ℕ := 9000
  let volume2 : ℕ := 6400
  density * (volume1 - volume2) = 208000 := by
  sorry

end population_difference_specific_population_difference_l319_31977


namespace line_transformation_l319_31905

/-- The analytical expression of a line after transformation -/
def transformed_line (a b : ℝ) (dx dy : ℝ) : ℝ → ℝ := fun x ↦ a * (x + dx) + b + dy

/-- The original line y = 2x - 1 -/
def original_line : ℝ → ℝ := fun x ↦ 2 * x - 1

theorem line_transformation :
  transformed_line 2 (-1) 1 (-2) = original_line := by sorry

end line_transformation_l319_31905


namespace thirty_factorial_trailing_zeros_l319_31942

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The factorial of n -/
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem thirty_factorial_trailing_zeros :
  trailingZeros 30 = 7 :=
by sorry

end thirty_factorial_trailing_zeros_l319_31942


namespace perfect_square_condition_l319_31967

/-- If 100x^2 - kxy + 49y^2 is a perfect square, then k = ±140 -/
theorem perfect_square_condition (x y k : ℝ) :
  (∃ (z : ℝ), 100 * x^2 - k * x * y + 49 * y^2 = z^2) →
  (k = 140 ∨ k = -140) := by
sorry

end perfect_square_condition_l319_31967


namespace min_games_for_condition_l319_31985

/-- Represents a football championship. -/
structure Championship where
  teams : Nat
  games_played : Nat

/-- Calculates the total number of possible games in a championship. -/
def total_possible_games (c : Championship) : Nat :=
  c.teams * (c.teams - 1) / 2

/-- Defines the property that among any three teams, at least two have played against each other. -/
def satisfies_condition (c : Championship) : Prop :=
  ∀ (a b d : Fin c.teams), a ≠ b ∧ b ≠ d ∧ a ≠ d →
    ∃ (x y : Fin c.teams), (x = a ∧ y = b) ∨ (x = b ∧ y = d) ∨ (x = a ∧ y = d)

/-- The main theorem to be proved. -/
theorem min_games_for_condition (c : Championship) 
  (h1 : c.teams = 20)
  (h2 : c.games_played ≥ 90)
  (h3 : ∀ (c' : Championship), c'.teams = 20 ∧ c'.games_played < 90 → ¬satisfies_condition c') :
  satisfies_condition c :=
sorry

end min_games_for_condition_l319_31985


namespace max_value_theorem_l319_31984

theorem max_value_theorem (x y z : ℝ) 
  (hx : 0 < x ∧ x < Real.sqrt 5) 
  (hy : 0 < y ∧ y < Real.sqrt 5) 
  (hz : 0 < z ∧ z < Real.sqrt 5) 
  (h_sum : x^4 + y^4 + z^4 ≥ 27) :
  (x / (x^2 - 5)) + (y / (y^2 - 5)) + (z / (z^2 - 5)) ≤ -3 * Real.sqrt 3 / 2 := by
  sorry

end max_value_theorem_l319_31984


namespace A_intersect_B_l319_31972

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}

def B : Set ℝ := {x | ∃ k : ℤ, x = 2*k}

theorem A_intersect_B : A ∩ B = {-2, 0} := by sorry

end A_intersect_B_l319_31972


namespace plates_arrangement_theorem_l319_31981

-- Define the number of plates of each color
def yellow_plates : ℕ := 4
def blue_plates : ℕ := 3
def red_plates : ℕ := 2
def purple_plates : ℕ := 1

-- Define the total number of plates
def total_plates : ℕ := yellow_plates + blue_plates + red_plates + purple_plates

-- Function to calculate circular arrangements
def circular_arrangements (n : ℕ) : ℕ :=
  (Nat.factorial n) / n

-- Function to calculate arrangements with restrictions
def arrangements_with_restrictions (total : ℕ) (y : ℕ) (b : ℕ) (r : ℕ) (p : ℕ) : ℕ :=
  circular_arrangements total - circular_arrangements (total - 1)

-- Theorem statement
theorem plates_arrangement_theorem :
  arrangements_with_restrictions total_plates yellow_plates blue_plates red_plates purple_plates = 980 := by
  sorry

end plates_arrangement_theorem_l319_31981


namespace tournament_games_played_23_teams_l319_31944

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  no_ties : Bool

/-- Calculates the number of games played in a single-elimination tournament. -/
def games_played (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties,
    22 games must be played before a winner can be declared. -/
theorem tournament_games_played_23_teams :
  ∀ (t : Tournament), t.num_teams = 23 → t.no_ties = true →
  games_played t = 22 := by
  sorry

end tournament_games_played_23_teams_l319_31944


namespace quadratic_inequality_theorem_l319_31969

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ := {x | x ≤ -4 ∨ x ≥ 3}

-- State the theorem
theorem quadratic_inequality_theorem (a b c : ℝ) 
  (h : ∀ x, f a b c x ≤ 0 ↔ x ∈ solution_set a b c) : 
  (a + b + c > 0) ∧ (∀ x, b * x + c > 0 ↔ x < 12) := by
  sorry

end quadratic_inequality_theorem_l319_31969


namespace colored_area_half_l319_31989

/-- Triangle ABC with side AB divided into n parts and AC into n+1 parts -/
structure DividedTriangle where
  ABC : Triangle
  n : ℕ

/-- The ratio of the sum of areas of colored triangles to the area of ABC -/
def coloredAreaRatio (dt : DividedTriangle) : ℚ :=
  sorry

/-- Theorem: The colored area ratio is always 1/2 -/
theorem colored_area_half (dt : DividedTriangle) : coloredAreaRatio dt = 1/2 :=
  sorry

end colored_area_half_l319_31989


namespace arithmetic_sequence_sum_l319_31921

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 7 = 38 →
  a 4 + a 5 + a 6 = 93 := by
  sorry

end arithmetic_sequence_sum_l319_31921


namespace sunday_newspaper_delivery_l319_31910

theorem sunday_newspaper_delivery (total : ℕ) (difference : ℕ) 
  (h1 : total = 110)
  (h2 : difference = 20) :
  ∃ (saturday sunday : ℕ), 
    saturday + sunday = total ∧ 
    sunday = saturday + difference ∧ 
    sunday = 65 := by
  sorry

end sunday_newspaper_delivery_l319_31910


namespace equation_sum_zero_l319_31917

theorem equation_sum_zero (a b c : ℝ) 
  (h1 : a + b / c = 1) 
  (h2 : b + c / a = 1) 
  (h3 : c + a / b = 1) : 
  a * b + b * c + c * a = 0 := by
sorry

end equation_sum_zero_l319_31917


namespace sqrt_x_plus_reciprocal_l319_31997

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end sqrt_x_plus_reciprocal_l319_31997


namespace triangle_side_ratio_l319_31956

/-- Given a triangle ABC with heights ha, hb, hc corresponding to sides a, b, c respectively,
    prove that if ha = 6, hb = 4, and hc = 3, then a : b : c = 2 : 3 : 4 -/
theorem triangle_side_ratio (a b c ha hb hc : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_heights : ha = 6 ∧ hb = 4 ∧ hc = 3) 
  (h_area : a * ha = b * hb ∧ b * hb = c * hc) : 
  ∃ (k : ℝ), k > 0 ∧ a = 2 * k ∧ b = 3 * k ∧ c = 4 * k := by
  sorry

end triangle_side_ratio_l319_31956


namespace exists_sum_of_digits_div_13_l319_31964

def sumOfDigits (n : ℕ) : ℕ := sorry

theorem exists_sum_of_digits_div_13 (n : ℕ) : 
  ∃ k ∈ Finset.range 79, (sumOfDigits (n + k)) % 13 = 0 := by sorry

end exists_sum_of_digits_div_13_l319_31964


namespace min_value_a_l319_31908

theorem min_value_a (a : ℝ) : 
  (∀ x > a, x + 4 / (x - a) ≥ 5) → a ≥ 1 := by sorry

end min_value_a_l319_31908


namespace intersection_implies_m_equals_one_l319_31960

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N (m : ℝ) : Set ℝ := {x | x^2 - m*x < 0}

-- State the theorem
theorem intersection_implies_m_equals_one :
  ∀ m : ℝ, (M ∩ N m) = {x | 0 < x ∧ x < 1} → m = 1 := by
  sorry

end intersection_implies_m_equals_one_l319_31960


namespace crosswalk_distance_l319_31982

/-- Given a parallelogram with the following properties:
  * One side has length 22 feet
  * An adjacent side has length 65 feet
  * The altitude perpendicular to the 22-foot side is 60 feet
  Then the altitude perpendicular to the 65-foot side is 264/13 feet. -/
theorem crosswalk_distance (a b h₁ h₂ : ℝ) 
  (ha : a = 22) 
  (hb : b = 65) 
  (hh₁ : h₁ = 60) : 
  a * h₁ = b * h₂ → h₂ = 264 / 13 := by
  sorry

#check crosswalk_distance

end crosswalk_distance_l319_31982


namespace cubic_equation_has_real_root_l319_31931

theorem cubic_equation_has_real_root :
  ∃ (x : ℝ), x^3 + 2 = 0 :=
sorry

end cubic_equation_has_real_root_l319_31931


namespace cylinder_not_triangular_front_view_l319_31922

/-- A solid geometry object --/
inductive Solid
  | Cylinder
  | Cone
  | Tetrahedron
  | TriangularPrism

/-- The shape of a view (projection) of a solid --/
inductive ViewShape
  | Triangle
  | Rectangle

/-- The front view of a solid --/
def frontView (s : Solid) : ViewShape :=
  match s with
  | Solid.Cylinder => ViewShape.Rectangle
  | _ => ViewShape.Triangle  -- We only care about the cylinder case for this problem

/-- Theorem: A cylinder cannot have a triangular front view --/
theorem cylinder_not_triangular_front_view :
  ∀ s : Solid, s = Solid.Cylinder → frontView s ≠ ViewShape.Triangle :=
by
  sorry

end cylinder_not_triangular_front_view_l319_31922


namespace expression_evaluation_l319_31950

theorem expression_evaluation : 6^3 - 4 * 6^2 + 4 * 6 + 2 = 98 := by
  sorry

end expression_evaluation_l319_31950


namespace arc_length_sector_l319_31971

/-- The arc length of a sector with central angle 36° and radius 15 is 3π. -/
theorem arc_length_sector (angle : ℝ) (radius : ℝ) : 
  angle = 36 → radius = 15 → (angle * π * radius) / 180 = 3 * π := by
  sorry

end arc_length_sector_l319_31971


namespace chocolate_distribution_l319_31929

theorem chocolate_distribution (pieces_per_bar : ℕ) 
  (girls_consumption_case1 girls_consumption_case2 : ℕ) 
  (boys_consumption_case1 boys_consumption_case2 : ℕ) 
  (bars_case1 bars_case2 : ℕ) : 
  pieces_per_bar = 12 →
  girls_consumption_case1 = 7 →
  boys_consumption_case1 = 2 →
  bars_case1 = 3 →
  girls_consumption_case2 = 8 →
  boys_consumption_case2 = 4 →
  bars_case2 = 4 →
  ∃ (girls boys : ℕ),
    girls_consumption_case1 * girls + boys_consumption_case1 * boys > pieces_per_bar * bars_case1 ∧
    girls_consumption_case2 * girls + boys_consumption_case2 * boys < pieces_per_bar * bars_case2 ∧
    girls = 5 ∧
    boys = 1 :=
by sorry

end chocolate_distribution_l319_31929


namespace zachary_pushups_count_l319_31924

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := sorry

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 58

/-- Zachary did 12 more crunches than push-ups -/
axiom crunches_pushups_difference : zachary_crunches = zachary_pushups + 12

theorem zachary_pushups_count : zachary_pushups = 46 := by sorry

end zachary_pushups_count_l319_31924


namespace annie_walk_distance_l319_31988

/-- The number of blocks Annie walked from her house to the bus stop -/
def annie_walk : ℕ := sorry

/-- The number of blocks Annie rode on the bus each way -/
def bus_ride : ℕ := 7

/-- The total number of blocks Annie traveled -/
def total_distance : ℕ := 24

theorem annie_walk_distance : annie_walk = 5 := by
  have h1 : 2 * annie_walk + 2 * bus_ride = total_distance := sorry
  sorry

end annie_walk_distance_l319_31988


namespace sin_cos_shift_l319_31991

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x) = Real.cos (2 * (x - π / 3) + π / 6) := by sorry

end sin_cos_shift_l319_31991


namespace prob_at_least_one_man_l319_31904

/-- The probability of selecting at least one man when choosing 5 people at random from a group of 12 men and 8 women -/
theorem prob_at_least_one_man (total_people : ℕ) (men : ℕ) (women : ℕ) (selection_size : ℕ) :
  total_people = men + women →
  men = 12 →
  women = 8 →
  selection_size = 5 →
  (1 : ℚ) - (women.choose selection_size : ℚ) / (total_people.choose selection_size : ℚ) = 687 / 692 :=
by sorry

end prob_at_least_one_man_l319_31904


namespace employee_relocation_l319_31962

theorem employee_relocation (E : ℝ) 
  (prefer_Y : ℝ) (prefer_X : ℝ) (max_preferred : ℝ) 
  (h1 : prefer_Y = 0.4 * E)
  (h2 : prefer_X = 0.6 * E)
  (h3 : max_preferred = 140)
  (h4 : prefer_Y + prefer_X = max_preferred) :
  prefer_X / E = 0.6 := by
sorry

end employee_relocation_l319_31962


namespace hyperbola_equation_l319_31907

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (f : ℝ × ℝ) (M N : ℝ × ℝ) :
  (∃ c : ℝ, c > 0 ∧ f = (c, 0) ∧ ∀ x y : ℝ, y^2 = 4 * Real.sqrt 7 * x → (x - c)^2 + y^2 = c^2) →  -- focus coincides with parabola focus
  (M.1 - 1 = M.2 ∧ N.1 - 1 = N.2) →  -- M and N are on the line y = x - 1
  ((M.1 + N.1) / 2 = -2/3) →  -- x-coordinate of midpoint is -2/3
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ 
    ((x - f.1)^2 / (a^2 + b^2) + (y - f.2)^2 / (a^2 + b^2) = 1 ∧
     (x + f.1)^2 / (a^2 + b^2) + (y - f.2)^2 / (a^2 + b^2) = 1)) →
  ∃ x y : ℝ, x^2/2 - y^2/5 = 1 ↔
    ((x - f.1)^2 / 7 + (y - f.2)^2 / 7 = 1 ∧
     (x + f.1)^2 / 7 + (y - f.2)^2 / 7 = 1) :=
by sorry

end hyperbola_equation_l319_31907


namespace equation_solution_l319_31965

theorem equation_solution : ∃ x : ℚ, (3/4 : ℚ) + 1/x = (7/8 : ℚ) ∧ x = 8 := by
  sorry

end equation_solution_l319_31965


namespace only_dog_owners_l319_31949

/-- The number of people who own only dogs -/
def D : ℕ := sorry

/-- The number of people who own only cats -/
def C : ℕ := 10

/-- The number of people who own only snakes -/
def S : ℕ := sorry

/-- The number of people who own only cats and dogs -/
def CD : ℕ := 5

/-- The number of people who own only cats and snakes -/
def CS : ℕ := sorry

/-- The number of people who own only dogs and snakes -/
def DS : ℕ := sorry

/-- The number of people who own cats, dogs, and snakes -/
def CDS : ℕ := 3

/-- The total number of pet owners -/
def total_pet_owners : ℕ := 59

/-- The total number of snake owners -/
def total_snake_owners : ℕ := 29

theorem only_dog_owners : D = 15 := by
  have h1 : D + C + S + CD + CS + DS + CDS = total_pet_owners := by sorry
  have h2 : S + CS + DS + CDS = total_snake_owners := by sorry
  sorry

end only_dog_owners_l319_31949


namespace complex_equation_solution_l319_31968

theorem complex_equation_solution (z : ℂ) : z * (1 + 2*I) = 3 + I → z = 1 - I := by
  sorry

end complex_equation_solution_l319_31968


namespace units_digit_17_2025_l319_31943

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The property that 17^n and 7^n have the same units digit for all n -/
axiom units_digit_17_7 (n : ℕ) : unitsDigit (17^n) = unitsDigit (7^n)

/-- The main theorem: the units digit of 17^2025 is 7 -/
theorem units_digit_17_2025 : unitsDigit (17^2025) = 7 := by
  sorry

end units_digit_17_2025_l319_31943


namespace cube_decomposition_smallest_term_l319_31935

theorem cube_decomposition_smallest_term (m : ℕ) (h1 : m ≥ 2) : 
  (m^2 - m + 1 = 73) → m = 9 := by
  sorry

end cube_decomposition_smallest_term_l319_31935


namespace tshirt_packages_l319_31938

theorem tshirt_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) (h1 : total_tshirts = 426) (h2 : tshirts_per_package = 6) :
  total_tshirts / tshirts_per_package = 71 := by
  sorry

end tshirt_packages_l319_31938


namespace student_weight_average_l319_31934

theorem student_weight_average (girls_avg : ℝ) (boys_avg : ℝ) 
  (h1 : girls_avg = 45) 
  (h2 : boys_avg = 55) : 
  (5 * girls_avg + 5 * boys_avg) / 10 = 50 := by
  sorry

end student_weight_average_l319_31934


namespace sum_three_numbers_l319_31992

theorem sum_three_numbers (a b c N : ℝ) : 
  a + b + c = 72 →
  a - 7 = N →
  b + 7 = N →
  2 * c = N →
  N = 28.8 := by
sorry

end sum_three_numbers_l319_31992


namespace expression_evaluation_l319_31940

theorem expression_evaluation :
  (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end expression_evaluation_l319_31940


namespace sara_letters_count_l319_31980

/-- The number of letters Sara sent in January. -/
def january_letters : ℕ := 6

/-- The number of letters Sara sent in February. -/
def february_letters : ℕ := 9

/-- The number of letters Sara sent in March. -/
def march_letters : ℕ := 3 * january_letters

/-- The total number of letters Sara sent over three months. -/
def total_letters : ℕ := january_letters + february_letters + march_letters

theorem sara_letters_count : total_letters = 33 := by
  sorry

end sara_letters_count_l319_31980


namespace largest_divisor_of_consecutive_odd_integers_l319_31936

theorem largest_divisor_of_consecutive_odd_integers (n : ℕ) :
  ∃ (Q : ℕ), Q = (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3) ∧
  15 ∣ Q ∧
  ∀ (k : ℕ), k > 15 → ¬(∀ (m : ℕ), k ∣ ((2*m - 3) * (2*m - 1) * (2*m + 1) * (2*m + 3))) :=
by sorry

end largest_divisor_of_consecutive_odd_integers_l319_31936


namespace celia_receives_171_spiders_l319_31948

/-- Represents the number of stickers of each type Célia has -/
structure StickerCount where
  butterfly : ℕ
  shark : ℕ
  snake : ℕ
  parakeet : ℕ
  monkey : ℕ

/-- Represents the conversion rates between different types of stickers -/
structure ConversionRates where
  butterfly_to_shark : ℕ
  snake_to_parakeet : ℕ
  monkey_to_spider : ℕ
  parakeet_to_spider : ℕ
  shark_to_parakeet : ℕ

/-- Calculates the total number of spider stickers Célia can receive -/
def total_spider_stickers (count : StickerCount) (rates : ConversionRates) : ℕ :=
  sorry

/-- Theorem stating that Célia can receive 171 spider stickers -/
theorem celia_receives_171_spiders (count : StickerCount) (rates : ConversionRates) 
    (h1 : count.butterfly = 4)
    (h2 : count.shark = 5)
    (h3 : count.snake = 3)
    (h4 : count.parakeet = 6)
    (h5 : count.monkey = 6)
    (h6 : rates.butterfly_to_shark = 3)
    (h7 : rates.snake_to_parakeet = 3)
    (h8 : rates.monkey_to_spider = 4)
    (h9 : rates.parakeet_to_spider = 3)
    (h10 : rates.shark_to_parakeet = 2) :
    total_spider_stickers count rates = 171 :=
  sorry

end celia_receives_171_spiders_l319_31948


namespace probability_yellow_ball_l319_31911

def num_red_balls : ℕ := 4
def num_yellow_balls : ℕ := 7

def total_balls : ℕ := num_red_balls + num_yellow_balls

theorem probability_yellow_ball :
  (num_yellow_balls : ℚ) / (total_balls : ℚ) = 7 / 11 := by
  sorry

end probability_yellow_ball_l319_31911


namespace distance_from_origin_to_12_5_l319_31987

/-- The distance from the origin to the point (12, 5) in a rectangular coordinate system is 13 units. -/
theorem distance_from_origin_to_12_5 : 
  Real.sqrt (12^2 + 5^2) = 13 := by sorry

end distance_from_origin_to_12_5_l319_31987


namespace scheme1_higher_sale_price_l319_31918

def original_price : ℝ := 15000

def scheme1_price (p : ℝ) : ℝ :=
  p * (1 - 0.25) * (1 - 0.15) * (1 - 0.05) * (1 + 0.30)

def scheme2_price (p : ℝ) : ℝ :=
  p * (1 - 0.40) * (1 + 0.30)

theorem scheme1_higher_sale_price :
  scheme1_price original_price > scheme2_price original_price :=
by sorry

end scheme1_higher_sale_price_l319_31918


namespace exam_marks_lost_l319_31906

theorem exam_marks_lost (total_questions : ℕ) (marks_per_correct : ℕ) (total_marks : ℕ) (correct_answers : ℕ)
  (h1 : total_questions = 80)
  (h2 : marks_per_correct = 4)
  (h3 : total_marks = 120)
  (h4 : correct_answers = 40) :
  (marks_per_correct * correct_answers - total_marks) / (total_questions - correct_answers) = 1 := by
sorry

end exam_marks_lost_l319_31906


namespace triangle_theorem_l319_31957

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.cos B = 3 →
  b * Real.cos A = 1 →
  A - B = π / 6 →
  c = 4 ∧ B = π / 6 := by
sorry

end triangle_theorem_l319_31957


namespace total_percentage_increase_l319_31915

/-- Calculates the total percentage increase in a purchase of three items given their initial and final prices. -/
theorem total_percentage_increase
  (book_initial : ℝ) (book_final : ℝ)
  (album_initial : ℝ) (album_final : ℝ)
  (poster_initial : ℝ) (poster_final : ℝ)
  (h1 : book_initial = 300)
  (h2 : book_final = 480)
  (h3 : album_initial = 15)
  (h4 : album_final = 20)
  (h5 : poster_initial = 5)
  (h6 : poster_final = 10) :
  (((book_final + album_final + poster_final) - (book_initial + album_initial + poster_initial)) / (book_initial + album_initial + poster_initial)) * 100 = 59.375 := by
  sorry

end total_percentage_increase_l319_31915


namespace product_of_three_numbers_l319_31976

theorem product_of_three_numbers (x y z : ℝ) : 
  x + y + z = 30 →
  x = 3 * ((y + z) - 2) →
  y = 4 * z - 1 →
  x * y * z = 294 := by
sorry

end product_of_three_numbers_l319_31976


namespace tax_difference_equals_0_625_l319_31902

/-- The price of an item before tax -/
def price : ℝ := 50

/-- The higher tax rate -/
def high_rate : ℝ := 0.075

/-- The lower tax rate -/
def low_rate : ℝ := 0.0625

/-- The difference between the two tax amounts -/
def tax_difference : ℝ := price * high_rate - price * low_rate

theorem tax_difference_equals_0_625 : tax_difference = 0.625 := by
  sorry

end tax_difference_equals_0_625_l319_31902


namespace decimal_digits_of_fraction_l319_31990

/-- The number of digits to the right of the decimal point when 5^7 / (10^5 * 125) is expressed as a decimal is 5. -/
theorem decimal_digits_of_fraction : ∃ (n : ℕ) (d : ℕ+) (k : ℕ),
  5^7 / (10^5 * 125) = n / d ∧
  10^k ≤ d ∧ d < 10^(k+1) ∧
  k = 5 := by sorry

end decimal_digits_of_fraction_l319_31990


namespace factor_tree_X_value_l319_31966

/-- Represents a node in the factor tree -/
structure TreeNode where
  value : ℕ

/-- Represents the factor tree structure -/
structure FactorTree where
  X : TreeNode
  F : TreeNode
  G : TreeNode
  H : TreeNode

/-- The main theorem to prove -/
theorem factor_tree_X_value (tree : FactorTree) : tree.X.value = 6776 :=
  sorry

/-- Axioms representing the given conditions -/
axiom F_value (tree : FactorTree) : tree.F.value = 7 * 4

axiom G_value (tree : FactorTree) : tree.G.value = 11 * tree.H.value

axiom H_value (tree : FactorTree) : tree.H.value = 11 * 2

axiom X_value (tree : FactorTree) : tree.X.value = tree.F.value * tree.G.value

end factor_tree_X_value_l319_31966


namespace log_equality_implies_ratio_l319_31903

theorem log_equality_implies_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 4 = Real.log q / Real.log 18) ∧
  (Real.log p / Real.log 4 = Real.log (p + q) / Real.log 25) →
  q / p = 2 - 2/5 := by
  sorry

end log_equality_implies_ratio_l319_31903


namespace projectile_trajectory_l319_31933

/-- Represents the trajectory of a projectile --/
def trajectory (c g : ℝ) (x y : ℝ) : Prop :=
  x^2 = (2 * c^2 / g) * y

/-- Theorem stating that a projectile follows a parabolic trajectory --/
theorem projectile_trajectory (c g : ℝ) (hc : c > 0) (hg : g > 0) :
  ∀ x y : ℝ, trajectory c g x y ↔ x^2 = (2 * c^2 / g) * y :=
sorry

end projectile_trajectory_l319_31933


namespace valid_assignments_count_l319_31953

/-- Represents the set of mascots -/
inductive Mascot
| AXiang
| AHe
| ARu
| AYi
| LeYangyang

/-- Represents the set of volunteers -/
inductive Volunteer
| A
| B
| C
| D
| E

/-- A function that assigns mascots to volunteers -/
def Assignment := Volunteer → Mascot

/-- Predicate that checks if an assignment satisfies the given conditions -/
def ValidAssignment (f : Assignment) : Prop :=
  (f Volunteer.A = Mascot.AXiang ∨ f Volunteer.B = Mascot.AXiang) ∧
  f Volunteer.C ≠ Mascot.LeYangyang

/-- The number of valid assignments -/
def NumValidAssignments : ℕ := sorry

/-- Theorem stating that the number of valid assignments is 36 -/
theorem valid_assignments_count : NumValidAssignments = 36 := by sorry

end valid_assignments_count_l319_31953


namespace factor_x_squared_minus_81_l319_31963

theorem factor_x_squared_minus_81 (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := by
  sorry

end factor_x_squared_minus_81_l319_31963


namespace quadratic_intersection_theorem_l319_31986

/-- Represents the "graph number" of a quadratic function y = ax^2 + bx + c -/
structure GraphNumber where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- A quadratic function intersects the x-axis at only one point if and only if its discriminant is zero -/
def intersects_x_axis_once (g : GraphNumber) : Prop :=
  (g.b ^ 2) - (4 * g.a * g.c) = 0

theorem quadratic_intersection_theorem (m : ℝ) (hm : m ≠ 0) :
  let g := GraphNumber.mk m (2 * m + 4) (2 * m + 4) hm
  intersects_x_axis_once g → m = 2 ∨ m = -2 := by
  sorry

end quadratic_intersection_theorem_l319_31986


namespace max_expression_value_l319_31946

-- Define a digit as a natural number between 0 and 9
def Digit := {n : ℕ // n ≤ 9}

-- Define the expression as a function of three digits
def Expression (x y z : Digit) : ℕ := 
  100 * x.val + 10 * y.val + z.val + 
  10 * x.val + z.val + 
  x.val

-- Theorem statement
theorem max_expression_value :
  ∃ (x y z : Digit), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Expression x y z = 992 ∧
    ∀ (a b c : Digit), a ≠ b ∧ b ≠ c ∧ a ≠ c →
      Expression a b c ≤ 992 :=
sorry

end max_expression_value_l319_31946


namespace quadratic_equation_single_solution_l319_31974

theorem quadratic_equation_single_solution :
  ∀ b : ℝ, b ≠ 0 →
  (∃! x : ℝ, b * x^2 - 24 * x + 6 = 0) →
  (∃ x : ℝ, b * x^2 - 24 * x + 6 = 0 ∧ x = 1/2) :=
by sorry

end quadratic_equation_single_solution_l319_31974


namespace existence_of_lower_bound_upper_bound_l319_31958

/-- The number of coefficients in (x+1)^a(x+2)^(n-a) divisible by 3 -/
def f (n a : ℕ) : ℕ :=
  sorry

/-- The minimum of f(n,a) for all valid a -/
def F (n : ℕ) : ℕ :=
  sorry

/-- There exist infinitely many positive integers n such that F(n) ≥ (n-1)/3 -/
theorem existence_of_lower_bound : ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, F n ≥ (n - 1) / 3 :=
  sorry

/-- For any positive integer n, F(n) ≤ (n-1)/3 -/
theorem upper_bound (n : ℕ) (hn : n > 0) : F n ≤ (n - 1) / 3 :=
  sorry

end existence_of_lower_bound_upper_bound_l319_31958


namespace rational_sums_and_products_l319_31945

-- Define the property of being rational
def IsRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Main theorem
theorem rational_sums_and_products (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : IsRational (x * y))
  (hyz : IsRational (y * z))
  (hzx : IsRational (z * x)) :
  (IsRational (x^2 + y^2 + z^2)) ∧
  (IsRational (x^3 + y^3 + z^3) → IsRational x ∧ IsRational y ∧ IsRational z) := by
  sorry


end rational_sums_and_products_l319_31945


namespace sphere_tetrahedron_intersection_length_l319_31925

/-- The total length of the intersection between a sphere and a regular tetrahedron -/
theorem sphere_tetrahedron_intersection_length 
  (edge_length : ℝ) 
  (sphere_radius : ℝ) 
  (h_edge : edge_length = 2 * Real.sqrt 6) 
  (h_radius : sphere_radius = Real.sqrt 3) : 
  ∃ (intersection_length : ℝ), 
    intersection_length = 8 * Real.sqrt 2 * Real.pi := by
  sorry

end sphere_tetrahedron_intersection_length_l319_31925


namespace range_of_a_l319_31955

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0

theorem range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  (a ∈ Set.Icc (-1) 1) ∨ (a > 3) := by
  sorry

end range_of_a_l319_31955


namespace polynomial_evaluation_l319_31937

theorem polynomial_evaluation (y : ℝ) (h : y = 2) : y^4 + y^3 + y^2 + y + 1 = 31 := by
  sorry

end polynomial_evaluation_l319_31937


namespace quadratic_inequality_range_l319_31926

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0) → (-1 < m ∧ m < 0) :=
by sorry

end quadratic_inequality_range_l319_31926


namespace problem_statement_l319_31996

theorem problem_statement (x y a b c : ℝ) : 
  (x = -y) → 
  (a * b = 1) → 
  (|c| = 2) → 
  ((((x + y) / 2)^2023) - ((-a * b)^2023) + c^3 = 9 ∨ 
   (((x + y) / 2)^2023) - ((-a * b)^2023) + c^3 = -7) :=
by sorry

end problem_statement_l319_31996


namespace f_2018_eq_l319_31927

open Real

/-- Sequence of functions defined recursively --/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ x => sin x - cos x
  | n + 1 => λ x => deriv (f n) x

/-- The 2018th function in the sequence equals -sin(x) + cos(x) --/
theorem f_2018_eq (x : ℝ) : f 2018 x = -sin x + cos x := by
  sorry

end f_2018_eq_l319_31927


namespace divisors_of_40_and_72_l319_31932

theorem divisors_of_40_and_72 : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ 40 % n = 0 ∧ 72 % n = 0) ∧ 
  (∀ n : ℕ, n > 0 ∧ 40 % n = 0 ∧ 72 % n = 0 → n ∈ S) ∧
  Finset.card S = 4 := by
sorry

end divisors_of_40_and_72_l319_31932


namespace function_inequality_l319_31999

theorem function_inequality (f : ℝ → ℝ) (h₁ : Differentiable ℝ f) 
  (h₂ : ∀ x, deriv f x < f x) : 
  f 1 < Real.exp 1 * f 0 ∧ f 2017 < Real.exp 2017 * f 0 := by
  sorry

end function_inequality_l319_31999


namespace arrangements_count_is_24_l319_31939

/-- The number of ways to arrange 5 people in a row with specific adjacency constraints -/
def arrangements_count : ℕ :=
  let total_people : ℕ := 5
  let adjacent_pair : ℕ := 2  -- A and B
  let non_adjacent : ℕ := 1   -- C
  (adjacent_pair.choose 1) * (adjacent_pair.factorial) * ((total_people - adjacent_pair).factorial)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count_is_24 : arrangements_count = 24 := by
  sorry

end arrangements_count_is_24_l319_31939


namespace minimum_value_theorem_l319_31959

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.log x + 1)

def interval : Set ℝ := {x | 1 / Real.exp 2 ≤ x ∧ x ≤ 1}

theorem minimum_value_theorem (m : ℝ) (hm : ∀ x ∈ interval, f x ≥ m) :
  Real.log (abs m) = 1 / Real.exp 2 := by
  sorry

end minimum_value_theorem_l319_31959


namespace octagon_triangle_angle_sum_l319_31912

theorem octagon_triangle_angle_sum :
  ∀ (ABC ABD : ℝ),
  (∃ (n : ℕ), n = 8 ∧ ABC = 180 * (n - 2) / n) →
  (∃ (m : ℕ), m = 3 ∧ ABD = 180 * (m - 2) / m) →
  ABC + ABD = 195 := by
sorry

end octagon_triangle_angle_sum_l319_31912


namespace total_amount_after_ten_years_l319_31916

/-- Calculates the total amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem: The total amount after 10 years with 5% annual interest rate -/
theorem total_amount_after_ten_years :
  let initial_deposit : ℝ := 100000
  let interest_rate : ℝ := 0.05
  let years : ℕ := 10
  compound_interest initial_deposit interest_rate years = initial_deposit * (1 + interest_rate) ^ years :=
by sorry

end total_amount_after_ten_years_l319_31916


namespace rotten_bananas_percentage_l319_31961

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percent : ℚ)
  (good_fruits_percent : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percent = 15 / 100)
  (h4 : good_fruits_percent = 89 / 100)
  : (1 : ℚ) - (good_fruits_percent * (total_oranges + total_bananas : ℚ) - (1 - rotten_oranges_percent) * total_oranges) / total_bananas = 5 / 100 := by
  sorry

end rotten_bananas_percentage_l319_31961


namespace teenage_group_size_l319_31923

theorem teenage_group_size (total_bill : ℝ) (individual_cost : ℝ) (gratuity_rate : ℝ) :
  total_bill = 840 →
  individual_cost = 100 →
  gratuity_rate = 0.2 →
  ∃ n : ℕ, n = 7 ∧ total_bill = (individual_cost * n) * (1 + gratuity_rate) :=
by
  sorry

end teenage_group_size_l319_31923


namespace fraction_classification_l319_31901

-- Define a fraction as a pair of integers (numerator, denominator)
def Fraction := ℤ × ℤ

-- Define proper fractions
def ProperFraction (f : Fraction) : Prop := f.1.natAbs < f.2.natAbs ∧ f.2 ≠ 0

-- Define improper fractions
def ImproperFraction (f : Fraction) : Prop := f.1.natAbs ≥ f.2.natAbs ∧ f.2 ≠ 0

-- Theorem stating that all fractions are either proper or improper
theorem fraction_classification (f : Fraction) : f.2 ≠ 0 → ProperFraction f ∨ ImproperFraction f :=
sorry

end fraction_classification_l319_31901


namespace work_completion_time_l319_31998

/-- 
Given a group of people who can complete a task in 12 days, 
prove that twice that number of people can complete half the task in 3 days.
-/
theorem work_completion_time 
  (people : ℕ) 
  (work : ℝ) 
  (h : people > 0) 
  (complete_time : ℝ → ℝ → ℝ → ℝ) 
  (h_complete : complete_time people work 12 = 1) :
  complete_time (2 * people) (work / 2) 3 = 1 := by
  sorry

#check work_completion_time

end work_completion_time_l319_31998


namespace santa_claus_candy_distribution_l319_31954

theorem santa_claus_candy_distribution :
  ∃ (n b g c m : ℕ),
    n = b + g ∧
    n > 0 ∧
    b * c + g * (c + 1) = 47 ∧
    b * (m + 1) + g * m = 74 ∧
    n = 11 :=
by sorry

end santa_claus_candy_distribution_l319_31954


namespace number_division_multiplication_l319_31952

theorem number_division_multiplication (x : ℚ) : x = 5.5 → (x / 6) * 12 = 11 := by
  sorry

end number_division_multiplication_l319_31952


namespace geometric_roots_poly_n_value_l319_31978

/-- A polynomial of degree 4 with four distinct real roots in geometric progression -/
structure GeometricRootsPoly where
  m : ℝ
  n : ℝ
  p : ℝ
  roots : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  geometric : ∃ (a r : ℝ), ∀ i, roots i = a * r ^ i.val
  is_root : ∀ i, roots i ^ 4 + m * roots i ^ 3 + n * roots i ^ 2 + p * roots i + 256 = 0

/-- The theorem stating that n = -32 for such polynomials -/
theorem geometric_roots_poly_n_value (poly : GeometricRootsPoly) : poly.n = -32 := by
  sorry

end geometric_roots_poly_n_value_l319_31978


namespace sine_cosine_inequality_l319_31979

theorem sine_cosine_inequality (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c := by
  sorry

end sine_cosine_inequality_l319_31979


namespace third_altitude_values_l319_31951

/-- Triangle with two known altitudes and an integer third altitude -/
structure TriangleWithAltitudes where
  /-- First known altitude -/
  h₁ : ℝ
  /-- Second known altitude -/
  h₂ : ℝ
  /-- Third altitude (integer) -/
  h₃ : ℤ
  /-- Condition that first altitude is 4 -/
  h₁_eq : h₁ = 4
  /-- Condition that second altitude is 12 -/
  h₂_eq : h₂ = 12

/-- Theorem stating the possible values of the third altitude -/
theorem third_altitude_values (t : TriangleWithAltitudes) :
  t.h₃ = 4 ∨ t.h₃ = 5 :=
sorry

end third_altitude_values_l319_31951


namespace point_in_second_quadrant_l319_31920

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: A point M with coordinates (a, b), where a < 0 and b > 0, is in the second quadrant -/
theorem point_in_second_quadrant (a b : ℝ) (ha : a < 0) (hb : b > 0) :
  SecondQuadrant ⟨a, b⟩ := by
  sorry

end point_in_second_quadrant_l319_31920


namespace solution_sum_l319_31930

theorem solution_sum (a b x y : ℝ) : 
  x = 2 ∧ y = -1 ∧ 
  a * x - 2 * y = 4 ∧ 
  3 * x + b * y = -7 →
  a + b = 14 := by
sorry

end solution_sum_l319_31930


namespace farmer_cows_problem_l319_31909

theorem farmer_cows_problem (initial_cows : ℕ) (final_cows : ℕ) (new_cows : ℕ) : 
  initial_cows = 51 →
  final_cows = 42 →
  (3 : ℚ) / 4 * (initial_cows + new_cows) = final_cows →
  new_cows = 5 := by
sorry

end farmer_cows_problem_l319_31909


namespace immersed_cone_specific_gravity_l319_31973

/-- Represents an equilateral cone immersed in water -/
structure ImmersedCone where
  -- Radius of the base of the cone
  baseRadius : ℝ
  -- Height of the cone
  height : ℝ
  -- Height of the cone above water
  heightAboveWater : ℝ
  -- Specific gravity of the cone material
  specificGravity : ℝ
  -- The cone is equilateral
  equilateral : height = baseRadius * Real.sqrt 3
  -- The area of the water surface circle is one-third of the base area
  waterSurfaceArea : π * (heightAboveWater / 3)^2 = π * baseRadius^2 / 3
  -- The angle between water surface and cone side is 120°
  waterSurfaceAngle : Real.cos (2 * π / 3) = heightAboveWater / (2 * baseRadius)

/-- Theorem stating the specific gravity of the cone -/
theorem immersed_cone_specific_gravity (c : ImmersedCone) :
  c.specificGravity = 1 - Real.sqrt 3 / 9 := by
  sorry

end immersed_cone_specific_gravity_l319_31973


namespace expression_simplification_and_evaluation_l319_31919

-- Define the original expression
def original_expr (a b : ℝ) : ℝ := 3*a^2*b - (2*a^2*b - (2*a*b - a^2*b) - 4*a^2) - a*b

-- Define the simplified expression
def simplified_expr (a b : ℝ) : ℝ := a*b + 4*a^2

-- Theorem statement
theorem expression_simplification_and_evaluation :
  (∀ a b : ℝ, original_expr a b = simplified_expr a b) ∧
  (original_expr (-3) (-2) = 22) :=
by sorry

end expression_simplification_and_evaluation_l319_31919


namespace intersection_of_P_and_Q_l319_31900

-- Define the set P
def P : Set ℝ := {x | x^2 - 7*x + 10 < 0}

-- Define the set Q
def Q : Set ℝ := {y | ∃ x ∈ P, y = x^2 - 8*x + 19}

-- Theorem statement
theorem intersection_of_P_and_Q : P ∩ Q = Set.Icc 3 5 := by
  sorry

end intersection_of_P_and_Q_l319_31900


namespace no_integer_solution_l319_31970

theorem no_integer_solution :
  ∀ (x y z : ℤ), x ≠ 0 → 2 * x^4 + 2 * x^2 * y^2 + y^4 ≠ z^2 := by
  sorry

end no_integer_solution_l319_31970


namespace complex_equation_solution_l319_31914

-- Define the complex number z
variable (z : ℂ)

-- State the theorem
theorem complex_equation_solution :
  (3 - 4*I + z)*I = 2 + I → z = -2 + 2*I := by
  sorry

end complex_equation_solution_l319_31914


namespace tom_annual_cost_l319_31975

/-- Calculates the total annual cost for Tom's sleep medication and doctor visits -/
def annual_cost (daily_pills : ℕ) (pill_cost : ℚ) (insurance_coverage : ℚ) 
                (yearly_doctor_visits : ℕ) (doctor_visit_cost : ℚ) : ℚ :=
  let daily_medication_cost := daily_pills * pill_cost
  let daily_out_of_pocket := daily_medication_cost * (1 - insurance_coverage)
  let annual_medication_cost := daily_out_of_pocket * 365
  let annual_doctor_cost := yearly_doctor_visits * doctor_visit_cost
  annual_medication_cost + annual_doctor_cost

/-- Theorem stating that Tom's annual cost for sleep medication and doctor visits is $1530 -/
theorem tom_annual_cost : 
  annual_cost 2 5 (4/5) 2 400 = 1530 := by
  sorry

end tom_annual_cost_l319_31975


namespace cube_root_of_unity_in_finite_field_l319_31941

theorem cube_root_of_unity_in_finite_field (p : ℕ) (hp : p.Prime) (hp3 : p > 3) :
  let F := ZMod p
  (∃ x : F, x^2 = -3) →
    (∃! (a b c : F), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^3 = 1 ∧ b^3 = 1 ∧ c^3 = 1) ∧
  (¬∃ x : F, x^2 = -3) →
    (∃! a : F, a^3 = 1) :=
sorry

end cube_root_of_unity_in_finite_field_l319_31941


namespace sum_divides_10n_count_l319_31983

theorem sum_divides_10n_count : 
  (∃ (S : Finset ℕ), S.card = 5 ∧ 
    (∀ n : ℕ, n > 0 → (n ∈ S ↔ (10 * n) % ((n * (n + 1)) / 2) = 0))) :=
sorry

end sum_divides_10n_count_l319_31983
