import Mathlib

namespace find_k_l3527_352719

theorem find_k : ∃ k : ℕ, 3 * 10 * 4 * k = Nat.factorial 9 ∧ k = 15120 := by
  sorry

end find_k_l3527_352719


namespace students_left_l3527_352720

theorem students_left (initial_students new_students final_students : ℕ) :
  initial_students = 8 →
  new_students = 8 →
  final_students = 11 →
  initial_students + new_students - final_students = 5 := by
sorry

end students_left_l3527_352720


namespace sum_of_fractions_l3527_352733

theorem sum_of_fractions : (3 : ℚ) / 10 + (3 : ℚ) / 1000 = 303 / 1000 := by
  sorry

end sum_of_fractions_l3527_352733


namespace farmer_land_calculation_l3527_352756

theorem farmer_land_calculation (total_land : ℝ) : 
  0.2 * 0.5 * 0.9 * total_land = 252 → total_land = 2800 := by
  sorry

end farmer_land_calculation_l3527_352756


namespace number_of_divisors_of_m_l3527_352726

def m : ℕ := 2^5 * 3^6 * 5^7 * 7^8

theorem number_of_divisors_of_m : 
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 3024 :=
sorry

end number_of_divisors_of_m_l3527_352726


namespace half_plus_five_equals_eleven_l3527_352777

theorem half_plus_five_equals_eleven (n : ℝ) : (1/2 : ℝ) * n + 5 = 11 → n = 12 := by
  sorry

end half_plus_five_equals_eleven_l3527_352777


namespace local_max_implies_a_less_than_neg_one_l3527_352798

/-- Given a real number a and a function y = e^x + ax with a local maximum point greater than zero, prove that a < -1 -/
theorem local_max_implies_a_less_than_neg_one (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ IsLocalMax (fun x => Real.exp x + a * x) x) → a < -1 :=
by sorry

end local_max_implies_a_less_than_neg_one_l3527_352798


namespace negation_equivalence_l3527_352725

theorem negation_equivalence :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) := by
  sorry

end negation_equivalence_l3527_352725


namespace plywood_cut_perimeter_difference_l3527_352703

/-- Represents a rectangular piece of plywood -/
structure Plywood where
  length : ℝ
  width : ℝ

/-- Represents a cut of the plywood into congruent rectangles -/
structure Cut where
  num_pieces : ℕ
  piece_length : ℝ
  piece_width : ℝ

/-- Calculate the perimeter of a rectangular piece -/
def perimeter (l w : ℝ) : ℝ := 2 * (l + w)

/-- Check if a cut is valid for a given plywood -/
def is_valid_cut (p : Plywood) (c : Cut) : Prop :=
  c.num_pieces * c.piece_length = p.length ∧ 
  c.num_pieces * c.piece_width = p.width

/-- The main theorem -/
theorem plywood_cut_perimeter_difference 
  (p : Plywood) 
  (h1 : p.length = 10 ∧ p.width = 5) 
  (h2 : ∃ c : Cut, is_valid_cut p c ∧ c.num_pieces = 5) :
  ∃ (max_perim min_perim : ℝ),
    (∀ c : Cut, is_valid_cut p c ∧ c.num_pieces = 5 → 
      perimeter c.piece_length c.piece_width ≤ max_perim) ∧
    (∀ c : Cut, is_valid_cut p c ∧ c.num_pieces = 5 → 
      perimeter c.piece_length c.piece_width ≥ min_perim) ∧
    max_perim - min_perim = 8 := by
  sorry

end plywood_cut_perimeter_difference_l3527_352703


namespace kelly_games_left_l3527_352704

/-- Given that Kelly has 106 Nintendo games initially and gives away 64 games,
    prove that she will have 42 games left. -/
theorem kelly_games_left (initial_games : ℕ) (games_given_away : ℕ) 
    (h1 : initial_games = 106) (h2 : games_given_away = 64) : 
    initial_games - games_given_away = 42 := by
  sorry

end kelly_games_left_l3527_352704


namespace contrapositive_equivalence_l3527_352793

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ ((x ≤ -1 ∨ x ≥ 1) → x^2 ≥ 1) := by
  sorry

end contrapositive_equivalence_l3527_352793


namespace score_statistics_l3527_352748

def scores : List ℝ := [80, 85, 90, 95]
def frequencies : List ℕ := [4, 6, 8, 2]

def total_students : ℕ := frequencies.sum

def median (s : List ℝ) (f : List ℕ) : ℝ := sorry

def mode (s : List ℝ) (f : List ℕ) : ℝ := sorry

theorem score_statistics :
  median scores frequencies = 87.5 ∧ mode scores frequencies = 90 := by sorry

end score_statistics_l3527_352748


namespace fishing_trip_total_l3527_352768

/-- The total number of fish caught by Brendan and his dad -/
def total_fish (morning_catch : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) : ℕ :=
  (morning_catch + afternoon_catch - thrown_back) + dad_catch

/-- Theorem stating that the total number of fish caught is 23 -/
theorem fishing_trip_total : 
  total_fish 8 3 5 13 = 23 := by sorry

end fishing_trip_total_l3527_352768


namespace fruit_seller_apples_l3527_352701

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * (1 - 0.4) = 420 → initial_apples = 700 :=
by sorry

end fruit_seller_apples_l3527_352701


namespace cylinder_sphere_surface_area_l3527_352722

theorem cylinder_sphere_surface_area (r : ℝ) (h : ℝ) :
  h = 2 * r →
  (4 / 3) * Real.pi * r^3 = 4 * Real.sqrt 3 * Real.pi →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 18 * Real.pi :=
by sorry

end cylinder_sphere_surface_area_l3527_352722


namespace unique_prime_square_sum_l3527_352782

theorem unique_prime_square_sum (p q : ℕ) : 
  Prime p → Prime q → ∃ (n : ℕ), p^(q+1) + q^(p+1) = n^2 → p = 2 ∧ q = 2 := by
  sorry

end unique_prime_square_sum_l3527_352782


namespace sin_sum_to_product_l3527_352740

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end sin_sum_to_product_l3527_352740


namespace triangle_construction_uniqueness_l3527_352729

/-- A point in the Euclidean plane -/
structure Point :=
  (x y : ℝ)

/-- A triangle in the Euclidean plane -/
structure Triangle :=
  (A B C : Point)

/-- The centroid of a triangle -/
def centroid (t : Triangle) : Point :=
  sorry

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry

/-- The touch point of the incircle on a side of a triangle -/
def touchPoint (t : Triangle) : Point :=
  sorry

/-- Theorem: Given the centroid, incenter, and touch point of the incircle on a side,
    a unique triangle can be constructed -/
theorem triangle_construction_uniqueness 
  (M I Q_a : Point) : 
  ∃! t : Triangle, 
    centroid t = M ∧ 
    incenter t = I ∧ 
    touchPoint t = Q_a :=
  sorry

end triangle_construction_uniqueness_l3527_352729


namespace sam_travel_distance_l3527_352761

/-- Given that Marguerite drove 150 miles in 3 hours, and Sam increased his speed by 20% and drove for 4 hours, prove that Sam traveled 240 miles. -/
theorem sam_travel_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) 
  (sam_speed_increase : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 →
  marguerite_time = 3 →
  sam_speed_increase = 0.2 →
  sam_time = 4 →
  (marguerite_distance / marguerite_time) * (1 + sam_speed_increase) * sam_time = 240 := by
  sorry

end sam_travel_distance_l3527_352761


namespace martyrs_cemetery_distance_l3527_352787

/-- The distance from the school to the Martyrs' Cemetery in kilometers -/
def distance : ℝ := 216

/-- The scheduled time for the journey in minutes -/
def scheduledTime : ℝ := 180

/-- The time saved in minutes when increasing speed by one-fifth after 1 hour -/
def timeSaved1 : ℝ := 20

/-- The time saved in minutes when increasing speed by one-third after 72km -/
def timeSaved2 : ℝ := 30

/-- The distance traveled at original speed before increasing by one-third -/
def initialDistance : ℝ := 72

theorem martyrs_cemetery_distance :
  (distance = 216) ∧
  (scheduledTime * (1 - 5/6) = timeSaved1) ∧
  (scheduledTime * (1 - 3/4) > timeSaved2) ∧
  (initialDistance / (1 - 2/3) = distance) :=
sorry

end martyrs_cemetery_distance_l3527_352787


namespace largest_n_with_unique_k_l3527_352735

theorem largest_n_with_unique_k : ∀ n : ℕ, n > 24 →
  ¬(∃! k : ℤ, (3 : ℚ)/7 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/19) ∧
  (∃! k : ℤ, (3 : ℚ)/7 < (24 : ℚ)/(24 + k) ∧ (24 : ℚ)/(24 + k) < 8/19) :=
by sorry

end largest_n_with_unique_k_l3527_352735


namespace broker_commission_rate_change_l3527_352706

/-- Proves that the new commission rate is 5% given the conditions of the problem -/
theorem broker_commission_rate_change
  (original_rate : ℝ)
  (business_slump : ℝ)
  (new_rate : ℝ)
  (h1 : original_rate = 0.04)
  (h2 : business_slump = 0.20000000000000007)
  (h3 : original_rate * (1 - business_slump) = new_rate) :
  new_rate = 0.05 := by
  sorry

#eval (0.04 / 0.7999999999999999 : Float)

end broker_commission_rate_change_l3527_352706


namespace factorization_equality_l3527_352774

theorem factorization_equality (a b : ℝ) : 2*a*b - a^2 - b^2 + 4 = (2 + a - b)*(2 - a + b) := by
  sorry

end factorization_equality_l3527_352774


namespace inner_triangle_area_l3527_352712

/-- Given a triangle ABC with sides a, b, c, and lines parallel to the sides drawn at a distance d from them,
    the area of the resulting inner triangle is (t - ds)^2 / t, where t is the area of ABC and s is its semi-perimeter. -/
theorem inner_triangle_area (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let s := (a + b + c) / 2
  let t := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inner_area := (t - d * s)^2 / t
  ∃ (inner_triangle_area : ℝ), inner_triangle_area = inner_area :=
by sorry

end inner_triangle_area_l3527_352712


namespace cartesian_coordinates_l3527_352767

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define planes and axes
def yOz_plane (p : Point3D) : Prop := p.x = 0
def z_axis (p : Point3D) : Prop := p.x = 0 ∧ p.y = 0
def xOz_plane (p : Point3D) : Prop := p.y = 0

-- Theorem statement
theorem cartesian_coordinates :
  (∃ (p : Point3D), yOz_plane p ∧ ∃ (b c : ℝ), p.y = b ∧ p.z = c) ∧
  (∃ (p : Point3D), z_axis p ∧ ∃ (c : ℝ), p.z = c) ∧
  (∃ (p : Point3D), xOz_plane p ∧ ∃ (a c : ℝ), p.x = a ∧ p.z = c) :=
by sorry

end cartesian_coordinates_l3527_352767


namespace incorrect_rectangle_l3527_352781

/-- Represents a 3x3 grid of rectangle perimeters --/
structure PerimeterGrid :=
  (top_row : Fin 3 → ℕ)
  (middle_row : Fin 3 → ℕ)
  (bottom_row : Fin 3 → ℕ)

/-- The given grid of perimeters --/
def given_grid : PerimeterGrid :=
  { top_row := ![14, 16, 12],
    middle_row := ![18, 18, 2],
    bottom_row := ![16, 18, 14] }

/-- Predicate to check if a perimeter grid is valid --/
def is_valid_grid (grid : PerimeterGrid) : Prop :=
  ∀ i j, i < 3 → j < 3 → 
    (grid.top_row i > 0) ∧ 
    (grid.middle_row i > 0) ∧ 
    (grid.bottom_row i > 0)

/-- Theorem stating that the rectangle with perimeter 2 is incorrect --/
theorem incorrect_rectangle (grid : PerimeterGrid) 
  (h : is_valid_grid grid) : 
  ∃ i j, grid.middle_row j = 2 ∧ 
    (i = 1 ∨ j = 2) ∧ 
    ¬(∀ k l, k ≠ i ∨ l ≠ j → grid.middle_row l > 2) :=
sorry


end incorrect_rectangle_l3527_352781


namespace area_enclosed_by_midpoints_l3527_352743

/-- The area enclosed by midpoints of line segments with length 3 and endpoints on adjacent sides of a square with side length 3 -/
theorem area_enclosed_by_midpoints (square_side : ℝ) (segment_length : ℝ) : square_side = 3 → segment_length = 3 → 
  ∃ (area : ℝ), area = 9 - (9 * Real.pi / 16) := by
  sorry

end area_enclosed_by_midpoints_l3527_352743


namespace sarah_initial_trucks_l3527_352711

/-- The number of trucks Sarah gave to Jeff -/
def trucks_given_to_jeff : ℕ := 13

/-- The number of trucks Sarah has left -/
def trucks_left : ℕ := 38

/-- The initial number of trucks Sarah had -/
def initial_trucks : ℕ := trucks_given_to_jeff + trucks_left

theorem sarah_initial_trucks : initial_trucks = 51 := by sorry

end sarah_initial_trucks_l3527_352711


namespace sum_inequality_l3527_352700

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1) :
  (1/a^2 + 1/b^2 + 1/c^2) ≥ ((4*b*c/(a^2 + 1) + 4*a*c/(b^2 + 1) + 4*a*b/(c^2 + 1)))^2 := by
  sorry

end sum_inequality_l3527_352700


namespace player_A_wins_l3527_352759

/-- Represents a game state with three piles of matches -/
structure GameState where
  pile1 : Nat
  pile2 : Nat
  pile3 : Nat

/-- Represents a player in the game -/
inductive Player
  | A
  | B

/-- Defines a valid move in the game -/
def ValidMove (state : GameState) (newState : GameState) : Prop :=
  ∃ (i j : Fin 3) (k : Nat),
    i ≠ j ∧
    k > 0 ∧
    k < state.pile1 + state.pile2 + state.pile3 ∧
    newState.pile1 + newState.pile2 + newState.pile3 = state.pile1 + state.pile2 + state.pile3 - k

/-- Defines the winning condition for a player -/
def Wins (player : Player) (initialState : GameState) : Prop :=
  ∀ (state : GameState),
    state = initialState →
    ∃ (strategy : GameState → GameState),
      (∀ (s : GameState), ValidMove s (strategy s)) ∧
      (∀ (opponent : Player → GameState → GameState),
        (∀ (s : GameState), ValidMove s (opponent player s)) →
        ∃ (n : Nat), ¬ValidMove (Nat.iterate (λ s => opponent player (strategy s)) n initialState) (opponent player (Nat.iterate (λ s => opponent player (strategy s)) n initialState)))

/-- The main theorem stating that Player A has a winning strategy -/
theorem player_A_wins :
  Wins Player.A ⟨100, 200, 300⟩ := by
  sorry


end player_A_wins_l3527_352759


namespace no_negative_log_base_exists_positive_fraction_log_base_l3527_352716

-- Define the property of being a valid logarithm base
def IsValidLogBase (b : ℝ) : Prop := b > 0 ∧ b ≠ 1

-- Theorem 1: No negative number can be a valid logarithm base
theorem no_negative_log_base :
  ∀ b : ℝ, b < 0 → ¬(IsValidLogBase b) :=
sorry

-- Theorem 2: There exists a positive fraction that is a valid logarithm base
theorem exists_positive_fraction_log_base :
  ∃ b : ℝ, 0 < b ∧ b < 1 ∧ IsValidLogBase b :=
sorry

end no_negative_log_base_exists_positive_fraction_log_base_l3527_352716


namespace fan_daily_usage_l3527_352779

/-- Calculates the daily usage of an electric fan given its power, monthly energy consumption, and days in a month -/
theorem fan_daily_usage 
  (fan_power : ℝ) 
  (monthly_energy : ℝ) 
  (days_in_month : ℕ) 
  (h1 : fan_power = 75) 
  (h2 : monthly_energy = 18) 
  (h3 : days_in_month = 30) : 
  (monthly_energy * 1000) / (fan_power * days_in_month) = 8 := by
  sorry

end fan_daily_usage_l3527_352779


namespace star_symmetry_l3527_352710

/-- The star operation for real numbers -/
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

/-- Theorem: For all real x and y, (x² - y²) ⋆ (y² - x²) = 0 -/
theorem star_symmetry (x y : ℝ) : star (x^2 - y^2) (y^2 - x^2) = 0 := by
  sorry

end star_symmetry_l3527_352710


namespace second_mechanic_rate_calculation_l3527_352786

/-- Represents the hourly rate of the second mechanic -/
def second_mechanic_rate : ℝ := sorry

/-- The first mechanic's hourly rate -/
def first_mechanic_rate : ℝ := 45

/-- Total combined work hours -/
def total_hours : ℝ := 20

/-- Total charge for both mechanics -/
def total_charge : ℝ := 1100

/-- Hours worked by the second mechanic -/
def second_mechanic_hours : ℝ := 5

theorem second_mechanic_rate_calculation : 
  second_mechanic_rate = 85 :=
by
  sorry

#check second_mechanic_rate_calculation

end second_mechanic_rate_calculation_l3527_352786


namespace three_lines_intersection_l3527_352794

/-- Three lines intersect at the same point if and only if m = -9 -/
theorem three_lines_intersection (m : ℝ) : 
  (∃ (x y : ℝ), y = 2*x ∧ x + y = 3 ∧ m*x + 2*y + 5 = 0) ↔ m = -9 := by
  sorry

end three_lines_intersection_l3527_352794


namespace subtract_negative_add_l3527_352736

theorem subtract_negative_add : 3 - (-5) + 7 = 15 := by
  sorry

end subtract_negative_add_l3527_352736


namespace simplify_power_l3527_352715

theorem simplify_power (y : ℝ) : (3 * y^4)^4 = 81 * y^16 := by
  sorry

end simplify_power_l3527_352715


namespace max_distance_complex_l3527_352755

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  (⨆ (z : ℂ), Complex.abs ((1 + 2*Complex.I)*z^4 - z^6)) = 81 * (9 + Real.sqrt 5) := by
  sorry

end max_distance_complex_l3527_352755


namespace triangle_properties_l3527_352790

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.b * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.B ∧
  t.b = 3 ∧
  Real.sin t.C = 2 * Real.sin t.A

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 3 ∧ (1 / 2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2 :=
sorry

end triangle_properties_l3527_352790


namespace largest_divisor_of_m_l3527_352723

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 39 ∣ m^2) :
  39 = Nat.gcd 39 m := by sorry

end largest_divisor_of_m_l3527_352723


namespace infinite_geometric_series_first_term_l3527_352727

theorem infinite_geometric_series_first_term 
  (r : ℚ) 
  (S : ℚ) 
  (h1 : r = 1 / 4) 
  (h2 : S = 40) 
  (h3 : S = a / (1 - r)) : 
  a = 30 := by
  sorry

end infinite_geometric_series_first_term_l3527_352727


namespace batter_distribution_l3527_352744

/-- Given two trays of batter where the second tray holds 20 cups less than the first,
    and the total amount is 500 cups, prove that the second tray holds 240 cups. -/
theorem batter_distribution (first_tray second_tray : ℕ) : 
  first_tray = second_tray + 20 →
  first_tray + second_tray = 500 →
  second_tray = 240 := by
sorry

end batter_distribution_l3527_352744


namespace example_rearrangements_l3527_352730

def word : String := "EXAMPLE"

def vowels : List Char := ['E', 'E', 'A']
def consonants : List Char := ['X', 'M', 'P', 'L']

def vowel_arrangements : ℕ := 3
def consonant_arrangements : ℕ := 24

theorem example_rearrangements :
  (vowel_arrangements * consonant_arrangements) = 72 :=
by sorry

end example_rearrangements_l3527_352730


namespace investment_ratio_problem_l3527_352718

theorem investment_ratio_problem (profit_ratio_p profit_ratio_q : ℚ) 
  (investment_time_p investment_time_q : ℚ) 
  (investment_ratio_p investment_ratio_q : ℚ) : 
  profit_ratio_p / profit_ratio_q = 7 / 11 →
  investment_time_p = 5 →
  investment_time_q = 10.999999999999998 →
  (investment_ratio_p * investment_time_p) / (investment_ratio_q * investment_time_q) = profit_ratio_p / profit_ratio_q →
  investment_ratio_p / investment_ratio_q = 7 / 5 := by
sorry

end investment_ratio_problem_l3527_352718


namespace dartboard_probability_l3527_352709

/-- The probability of hitting a specific region on a square dartboard -/
theorem dartboard_probability : 
  ∃ (square_side_length : ℝ) (region_area : ℝ → ℝ),
    square_side_length = 2 ∧
    (∀ x, x > 0 → region_area x = (π * x^2) / 4 - x^2 / 2) ∧
    region_area square_side_length / square_side_length^2 = (π - 2) / 4 := by
  sorry

end dartboard_probability_l3527_352709


namespace M_when_a_is_one_M_union_N_equals_N_l3527_352754

-- Define the set M as a function of a
def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}

-- Define the set N
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem 1: When a = 1, M is the open interval (0, 2)
theorem M_when_a_is_one : M 1 = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem 2: M ∪ N = N if and only if a ∈ [-1, 2]
theorem M_union_N_equals_N (a : ℝ) : M a ∪ N = N ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

end M_when_a_is_one_M_union_N_equals_N_l3527_352754


namespace tan_theta_value_l3527_352788

theorem tan_theta_value (θ : Real) 
  (h : 2 * Real.cos (θ - π/3) = 3 * Real.cos θ) : 
  Real.tan θ = 2 * Real.sqrt 3 / 3 := by
  sorry

end tan_theta_value_l3527_352788


namespace negative_numbers_roots_l3527_352746

theorem negative_numbers_roots :
  (∀ x : ℝ, x < 0 → ¬∃ y : ℝ, y ^ 2 = x) ∧
  (∀ x : ℝ, x < 0 → ∃ y : ℝ, y ^ 3 = x) :=
by sorry

end negative_numbers_roots_l3527_352746


namespace digit_D_is_nine_l3527_352750

/-- Represents a two-digit number --/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  tens_is_digit : tens < 10
  ones_is_digit : ones < 10

def value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem digit_D_is_nine
  (A B C D : Nat)
  (A_is_digit : A < 10)
  (B_is_digit : B < 10)
  (C_is_digit : C < 10)
  (D_is_digit : D < 10)
  (addition : value ⟨A, B, A_is_digit, B_is_digit⟩ + value ⟨C, B, C_is_digit, B_is_digit⟩ = value ⟨D, A, D_is_digit, A_is_digit⟩)
  (subtraction : value ⟨A, B, A_is_digit, B_is_digit⟩ - value ⟨C, B, C_is_digit, B_is_digit⟩ = A) :
  D = 9 := by
  sorry

end digit_D_is_nine_l3527_352750


namespace pencils_multiple_of_fifty_l3527_352769

/-- Given a number of students, pens, and pencils, we define a valid distribution --/
def ValidDistribution (S P : ℕ) : Prop :=
  S > 0 ∧ S ≤ 50 ∧ 100 % S = 0 ∧ P % S = 0

/-- Theorem stating that the number of pencils must be a multiple of 50 --/
theorem pencils_multiple_of_fifty (P : ℕ) :
  (∃ S : ℕ, ValidDistribution S P) → P % 50 = 0 := by
  sorry

end pencils_multiple_of_fifty_l3527_352769


namespace triangle_rectangle_equal_area_l3527_352702

theorem triangle_rectangle_equal_area (h : ℝ) (h_pos : h > 0) :
  let triangle_base : ℝ := 24
  let triangle_area : ℝ := (1 / 2) * triangle_base * h
  let rectangle_base : ℝ := (1 / 2) * triangle_base
  let rectangle_area : ℝ := rectangle_base * h
  triangle_area = rectangle_area →
  rectangle_base = 12 := by
sorry


end triangle_rectangle_equal_area_l3527_352702


namespace problem1_solution_problem2_solution_l3527_352783

-- Problem 1
def problem1 (a b : ℕ) : Prop :=
  a ≠ b ∧
  ∃ p k : ℕ, Prime p ∧ b^2 + a = p^k ∧
  (b^2 + a) ∣ (a^2 + b)

theorem problem1_solution :
  ∀ a b : ℕ, problem1 a b ↔ (a = 5 ∧ b = 2) :=
sorry

-- Problem 2
def problem2 (a b : ℕ) : Prop :=
  a > 1 ∧ b > 1 ∧ a ≠ b ∧
  (b^2 + a - 1) ∣ (a^2 + b - 1)

theorem problem2_solution :
  ∀ a b : ℕ, problem2 a b →
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ (b^2 + a - 1) ∧ q ∣ (b^2 + a - 1) :=
sorry

end problem1_solution_problem2_solution_l3527_352783


namespace quadratic_equation_roots_l3527_352780

/-- A quadratic equation with roots -1 and 3 -/
theorem quadratic_equation_roots (x : ℝ) : 
  (x^2 - 2*x - 3 = 0) ↔ (x = -1 ∨ x = 3) := by
  sorry

end quadratic_equation_roots_l3527_352780


namespace mink_skins_per_coat_l3527_352739

theorem mink_skins_per_coat 
  (initial_minks : ℕ) 
  (babies_per_mink : ℕ) 
  (fraction_set_free : ℚ) 
  (coats_made : ℕ) :
  initial_minks = 30 →
  babies_per_mink = 6 →
  fraction_set_free = 1/2 →
  coats_made = 7 →
  (initial_minks * (1 + babies_per_mink) * (1 - fraction_set_free)) / coats_made = 15 := by
sorry

end mink_skins_per_coat_l3527_352739


namespace sarah_toad_count_l3527_352758

/-- The number of toads each person has -/
structure ToadCount where
  tim : ℕ
  jim : ℕ
  sarah : ℕ

/-- Given conditions about toad counts -/
def toad_conditions (tc : ToadCount) : Prop :=
  tc.tim = 30 ∧ 
  tc.jim = tc.tim + 20 ∧ 
  tc.sarah = 2 * tc.jim

/-- Theorem stating Sarah has 100 toads under given conditions -/
theorem sarah_toad_count (tc : ToadCount) (h : toad_conditions tc) : tc.sarah = 100 := by
  sorry

end sarah_toad_count_l3527_352758


namespace teacher_student_probability_teacher_student_probability_correct_l3527_352770

/-- The probability that neither teacher stands at either end when 2 teachers and 2 students
    stand in a row for a group photo. -/
theorem teacher_student_probability : ℚ :=
  let num_teachers : ℕ := 2
  let num_students : ℕ := 2
  let total_arrangements : ℕ := Nat.factorial 4
  let favorable_arrangements : ℕ := Nat.factorial 2 * Nat.factorial 2
  1 / 6

/-- Proof that the probability is correct. -/
theorem teacher_student_probability_correct : teacher_student_probability = 1 / 6 := by
  sorry

end teacher_student_probability_teacher_student_probability_correct_l3527_352770


namespace right_triangle_3_4_5_l3527_352753

/-- A triangle with side lengths 3, 4, and 5 is a right triangle. -/
theorem right_triangle_3_4_5 :
  ∀ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 →
  a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_3_4_5_l3527_352753


namespace tony_age_l3527_352771

/-- Given that Tony and Belinda have a combined age of 56, and Belinda is 40 years old,
    prove that Tony is 16 years old. -/
theorem tony_age (total_age : ℕ) (belinda_age : ℕ) (h1 : total_age = 56) (h2 : belinda_age = 40) :
  total_age - belinda_age = 16 := by
  sorry

end tony_age_l3527_352771


namespace student_age_problem_l3527_352717

theorem student_age_problem (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ) 
  (total_avg_age : ℕ) (group1_avg_age : ℕ) (group2_avg_age : ℕ) :
  total_students = 20 →
  group1_students = 9 →
  group2_students = 10 →
  total_avg_age = 20 →
  group1_avg_age = 11 →
  group2_avg_age = 24 →
  (total_students * total_avg_age) - (group1_students * group1_avg_age + group2_students * group2_avg_age) = 61 :=
by sorry

end student_age_problem_l3527_352717


namespace train_platform_problem_l3527_352791

/-- The length of a train in meters. -/
def train_length : ℝ := 110

/-- The time taken to cross the first platform in seconds. -/
def time_first : ℝ := 15

/-- The time taken to cross the second platform in seconds. -/
def time_second : ℝ := 20

/-- The length of the second platform in meters. -/
def second_platform_length : ℝ := 250

/-- The length of the first platform in meters. -/
def first_platform_length : ℝ := 160

theorem train_platform_problem :
  (train_length + first_platform_length) / time_first =
  (train_length + second_platform_length) / time_second :=
sorry

end train_platform_problem_l3527_352791


namespace remainder_sum_l3527_352728

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) :
  (n % 2) + (n % 3) + (n % 6) + (n % 9) = 10 := by
  sorry

end remainder_sum_l3527_352728


namespace min_value_fraction_sum_l3527_352773

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 9 / b ≥ 8) ∧
  (1 / a + 9 / b = 8 ↔ a = 1/2 ∧ b = 3/2) :=
by sorry

end min_value_fraction_sum_l3527_352773


namespace committee_arrangement_count_l3527_352785

/-- The number of ways to arrange n indistinguishable objects of type A
    and m indistinguishable objects of type B in a row of (n+m) positions -/
def arrangement_count (n m : ℕ) : ℕ :=
  Nat.choose (n + m) m

/-- Theorem stating that there are 120 ways to arrange 7 indistinguishable objects
    and 3 indistinguishable objects in a row of 10 positions -/
theorem committee_arrangement_count :
  arrangement_count 7 3 = 120 := by
  sorry

end committee_arrangement_count_l3527_352785


namespace chord_length_circle_line_intersection_chord_length_proof_l3527_352721

/-- The chord length cut by the line y = x from the circle x^2 + (y+2)^2 = 4 is 2√2 -/
theorem chord_length_circle_line_intersection : Real → Prop :=
  λ chord_length =>
    let circle := λ x y => x^2 + (y+2)^2 = 4
    let line := λ x y => y = x
    ∃ x₁ y₁ x₂ y₂,
      circle x₁ y₁ ∧ circle x₂ y₂ ∧
      line x₁ y₁ ∧ line x₂ y₂ ∧
      chord_length = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ∧
      chord_length = 2 * Real.sqrt 2

/-- Proof of the chord length theorem -/
theorem chord_length_proof : chord_length_circle_line_intersection (2 * Real.sqrt 2) := by
  sorry

end chord_length_circle_line_intersection_chord_length_proof_l3527_352721


namespace division_not_imply_multiple_and_factor_l3527_352742

theorem division_not_imply_multiple_and_factor :
  ¬ (∀ a b : ℝ, a / b = 5 → (∃ k : ℤ, a = b * k) ∧ (∃ k : ℤ, b * k = a)) := by
  sorry

end division_not_imply_multiple_and_factor_l3527_352742


namespace prob_three_red_cards_standard_deck_l3527_352737

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- A standard deck of cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- The probability of drawing three red cards in succession from a standard deck -/
def prob_three_red_cards (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit : ℚ) / d.total_cards *
  ((d.red_suits * d.cards_per_suit - 1) : ℚ) / (d.total_cards - 1) *
  ((d.red_suits * d.cards_per_suit - 2) : ℚ) / (d.total_cards - 2)

/-- Theorem: The probability of drawing three red cards in succession from a standard deck is 2/17 -/
theorem prob_three_red_cards_standard_deck :
  prob_three_red_cards standard_deck = 2 / 17 := by
  sorry

end prob_three_red_cards_standard_deck_l3527_352737


namespace technicians_count_l3527_352752

/-- Represents the workshop scenario with workers and salaries -/
structure Workshop where
  total_workers : ℕ
  avg_salary : ℚ
  technician_salary : ℚ
  other_salary : ℚ

/-- Calculates the number of technicians in the workshop -/
def num_technicians (w : Workshop) : ℚ :=
  ((w.avg_salary - w.other_salary) * w.total_workers) / (w.technician_salary - w.other_salary)

/-- The given workshop scenario -/
def given_workshop : Workshop :=
  { total_workers := 22
    avg_salary := 850
    technician_salary := 1000
    other_salary := 780 }

/-- Theorem stating that the number of technicians in the given workshop is 7 -/
theorem technicians_count :
  num_technicians given_workshop = 7 := by
  sorry


end technicians_count_l3527_352752


namespace b_cubed_is_zero_l3527_352799

theorem b_cubed_is_zero (B : Matrix (Fin 3) (Fin 3) ℝ) (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end b_cubed_is_zero_l3527_352799


namespace triangle_parallelogram_area_relation_l3527_352708

theorem triangle_parallelogram_area_relation (x : ℝ) :
  let triangle_base := x - 2
  let triangle_height := x - 2
  let parallelogram_base := x - 3
  let parallelogram_height := x + 4
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let parallelogram_area := parallelogram_base * parallelogram_height
  parallelogram_area = 3 * triangle_area →
  (∀ y : ℝ, (y - 8) * (y - 3) = 0 ↔ y = x) →
  8 + 3 = 11 :=
by sorry

end triangle_parallelogram_area_relation_l3527_352708


namespace arithmetic_mean_problem_l3527_352784

theorem arithmetic_mean_problem (m n : ℝ) 
  (h1 : (m + 2*n) / 2 = 4) 
  (h2 : (2*m + n) / 2 = 5) : 
  (m + n) / 2 = 3 := by
sorry

end arithmetic_mean_problem_l3527_352784


namespace future_age_difference_l3527_352765

/-- Proves that the number of years in the future when the father's age will be 20 years more than twice the son's age is 4, given the conditions stated in the problem. -/
theorem future_age_difference (father_age son_age x : ℕ) : 
  father_age = 44 →
  father_age = 4 * son_age + 4 →
  father_age + x = 2 * (son_age + x) + 20 →
  x = 4 := by
sorry

end future_age_difference_l3527_352765


namespace quadratic_equation_sum_product_l3527_352714

theorem quadratic_equation_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9) →
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x * y = 20) →
  p + q = 87 := by
sorry

end quadratic_equation_sum_product_l3527_352714


namespace unique_grid_solution_l3527_352731

-- Define the grid type
def Grid := List (List Nat)

-- Define the visibility type
def Visibility := List Nat

-- Function to check if a grid is valid
def is_valid_grid (g : Grid) : Prop := sorry

-- Function to check if visibility conditions are met
def meets_visibility (g : Grid) (v : Visibility) : Prop := sorry

-- Function to extract the four-digit number from the grid
def extract_number (g : Grid) : Nat := sorry

-- Theorem statement
theorem unique_grid_solution :
  ∀ (g : Grid) (v : Visibility),
    is_valid_grid g ∧ meets_visibility g v →
    extract_number g = 2213 := by sorry

end unique_grid_solution_l3527_352731


namespace cubic_function_properties_l3527_352772

/-- A cubic function f(x) = ax³ - bx² + c with a > 0 -/
def f (a b c x : ℝ) : ℝ := a * x^3 - b * x^2 + c

/-- The derivative of f with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := 3 * a * x^2 - 2 * b * x

theorem cubic_function_properties
  (a b c : ℝ)
  (ha : a > 0) :
  -- 1. Extreme points when b = 3a
  (b = 3 * a → (∀ x : ℝ, f_deriv a b x = 0 ↔ x = 0 ∨ x = 2)) ∧
  -- 2. Range of b when a = 1 and x²ln(x) ≥ f(x) - 2x - c for x ∈ [3,4]
  (a = 1 → (∀ x : ℝ, x ∈ Set.Icc 3 4 → x^2 * Real.log x ≥ f 1 b c x - 2*x - c) →
    b ≥ 7/2 - Real.log 4) ∧
  -- 3. Existence of three tangent lines
  (b = 3 * a → 5 * a < c → c < 6 * a →
    ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      f a b c x₁ + f_deriv a b x₁ * (2 - x₁) = a ∧
      f a b c x₂ + f_deriv a b x₂ * (2 - x₂) = a ∧
      f a b c x₃ + f_deriv a b x₃ * (2 - x₃) = a) :=
by sorry

end cubic_function_properties_l3527_352772


namespace constant_term_expansion_l3527_352749

/-- The constant term in the expansion of (3+x)(x+1/x)^6 -/
def constant_term : ℕ := 60

/-- Theorem: The constant term in the expansion of (3+x)(x+1/x)^6 is 60 -/
theorem constant_term_expansion :
  constant_term = 60 := by sorry

end constant_term_expansion_l3527_352749


namespace divisibility_condition_l3527_352760

theorem divisibility_condition (m n : ℕ) :
  (1 + (m + n) * m) ∣ ((n + 1) * (m + n) - 1) ↔ (m = 0 ∨ m = 1) := by
  sorry

end divisibility_condition_l3527_352760


namespace system_solution_l3527_352778

theorem system_solution : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 → 
  (Real.log x / Real.log 4 - Real.log y / Real.log 2 = 0) ∧ 
  (x^2 - 5*y^2 + 4 = 0) → 
  ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) :=
by sorry

end system_solution_l3527_352778


namespace prob_heads_tails_heads_l3527_352797

/-- A fair coin has equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of a specific sequence of n independent events -/
def prob_sequence (p : ℝ) (n : ℕ) : ℝ := p ^ n

/-- The probability of getting heads, then tails, then heads when flipping a fair coin three times -/
theorem prob_heads_tails_heads (p : ℝ) (h_fair : fair_coin p) : 
  prob_sequence p 3 = 1 / 8 := by sorry

end prob_heads_tails_heads_l3527_352797


namespace unique_solution_iff_m_eq_49_div_12_l3527_352741

/-- For a quadratic equation ax^2 + bx + c = 0 to have exactly one solution,
    its discriminant (b^2 - 4ac) must be zero. -/
def has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The quadratic equation 3x^2 - 7x + m = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  3*x^2 - 7*x + m = 0

/-- Theorem: The quadratic equation 3x^2 - 7x + m = 0 has exactly one solution
    if and only if m = 49/12 -/
theorem unique_solution_iff_m_eq_49_div_12 :
  (∃! x, quadratic_equation x m) ↔ m = 49/12 :=
sorry

end unique_solution_iff_m_eq_49_div_12_l3527_352741


namespace push_up_sets_l3527_352734

/-- Represents the number of push-ups done by each person -/
structure PushUps where
  zachary : ℕ
  david : ℕ
  emily : ℕ

/-- Calculates the number of complete sets of push-ups done together -/
def completeSets (p : PushUps) : ℕ :=
  1

theorem push_up_sets (p : PushUps) 
  (h1 : p.zachary = 47)
  (h2 : p.david = p.zachary + 15)
  (h3 : p.emily = 2 * p.david) :
  completeSets p = 1 := by
  sorry

#check push_up_sets

end push_up_sets_l3527_352734


namespace quarters_to_nickels_difference_l3527_352724

/-- The difference in money (in nickels) between two people given their quarter amounts -/
def money_difference_in_nickels (charles_quarters richard_quarters : ℕ) : ℤ :=
  5 * (charles_quarters - richard_quarters)

theorem quarters_to_nickels_difference (q : ℕ) :
  money_difference_in_nickels (5 * q + 3) (q + 7) = 20 * (q - 1) := by
  sorry

end quarters_to_nickels_difference_l3527_352724


namespace rhombus_side_length_l3527_352707

/-- For a rhombus with area K and one diagonal three times the length of the other,
    the side length s is equal to √(5K/3). -/
theorem rhombus_side_length (K : ℝ) (h : K > 0) :
  ∃ (d : ℝ), d > 0 ∧ 
  let s := Real.sqrt ((5 * K) / 3)
  let area := (1/2) * d * (3*d)
  area = K ∧ 
  s^2 = (d/2)^2 + (3*d/2)^2 := by
  sorry

end rhombus_side_length_l3527_352707


namespace square_ratio_side_lengths_l3527_352796

theorem square_ratio_side_lengths :
  let area_ratio : ℚ := 8 / 125
  let side_ratio : ℝ := Real.sqrt (area_ratio)
  let rationalized_ratio : ℝ := side_ratio * Real.sqrt 5 / Real.sqrt 5
  rationalized_ratio = 2 * Real.sqrt 10 / 25 := by
  sorry

end square_ratio_side_lengths_l3527_352796


namespace absolute_difference_26th_terms_l3527_352763

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + d * (n - 1)

theorem absolute_difference_26th_terms : 
  let C := arithmetic_sequence 50 15
  let D := arithmetic_sequence 85 (-20)
  |C 26 - D 26| = 840 := by
sorry

end absolute_difference_26th_terms_l3527_352763


namespace f_pi_sixth_l3527_352789

/-- The function f(x) = sin x + a cos x, where a < 0 and max f(x) = 2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + a * Real.cos x

/-- Theorem stating that f(π/6) = -1 under given conditions -/
theorem f_pi_sixth (a : ℝ) (h1 : a < 0) (h2 : ∀ x, f a x ≤ 2) (h3 : ∃ x, f a x = 2) :
  f a (Real.pi / 6) = -1 := by
  sorry

end f_pi_sixth_l3527_352789


namespace tea_hot_chocolate_difference_mo_drink_difference_l3527_352776

/-- Represents the drinking habits and week data for Mo --/
structure MoDrinkingHabits where
  n : ℕ  -- Number of hot chocolate cups on rainy days
  total_cups : ℕ  -- Total cups drunk in a week
  rainy_days : ℕ  -- Number of rainy days in a week

/-- Theorem stating the difference between tea and hot chocolate cups --/
theorem tea_hot_chocolate_difference (mo : MoDrinkingHabits) 
  (h1 : mo.total_cups = 26)
  (h2 : mo.rainy_days = 1) :
  3 * (7 - mo.rainy_days) - mo.n * mo.rainy_days = 10 := by
  sorry

/-- Main theorem proving the difference is 10 --/
theorem mo_drink_difference : ∃ mo : MoDrinkingHabits, 
  mo.total_cups = 26 ∧ 
  mo.rainy_days = 1 ∧ 
  3 * (7 - mo.rainy_days) - mo.n * mo.rainy_days = 10 := by
  sorry

end tea_hot_chocolate_difference_mo_drink_difference_l3527_352776


namespace megan_initial_markers_l3527_352705

/-- The number of markers Megan initially had -/
def initial_markers : ℕ := sorry

/-- The number of markers Robert gave to Megan -/
def roberts_markers : ℕ := 109

/-- The total number of markers Megan has after receiving markers from Robert -/
def total_markers : ℕ := 326

/-- Theorem stating that the initial number of markers Megan had is 217 -/
theorem megan_initial_markers : initial_markers = 217 := by
  sorry

end megan_initial_markers_l3527_352705


namespace one_isosceles_triangle_l3527_352762

-- Define a point in 2D space
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a triangle by its three vertices
structure Triangle :=
  (v1 : Point)
  (v2 : Point)
  (v3 : Point)

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := squaredDistance t.v1 t.v2
  let d2 := squaredDistance t.v2 t.v3
  let d3 := squaredDistance t.v3 t.v1
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the four triangles
def triangle1 : Triangle := ⟨⟨0, 7⟩, ⟨3, 7⟩, ⟨1, 5⟩⟩
def triangle2 : Triangle := ⟨⟨4, 5⟩, ⟨4, 7⟩, ⟨6, 5⟩⟩
def triangle3 : Triangle := ⟨⟨0, 2⟩, ⟨3, 3⟩, ⟨7, 2⟩⟩
def triangle4 : Triangle := ⟨⟨11, 5⟩, ⟨10, 7⟩, ⟨12, 5⟩⟩

-- Theorem: Exactly one of the four triangles is isosceles
theorem one_isosceles_triangle :
  (isIsosceles triangle1 ∨ isIsosceles triangle2 ∨ isIsosceles triangle3 ∨ isIsosceles triangle4) ∧
  ¬(isIsosceles triangle1 ∧ isIsosceles triangle2) ∧
  ¬(isIsosceles triangle1 ∧ isIsosceles triangle3) ∧
  ¬(isIsosceles triangle1 ∧ isIsosceles triangle4) ∧
  ¬(isIsosceles triangle2 ∧ isIsosceles triangle3) ∧
  ¬(isIsosceles triangle2 ∧ isIsosceles triangle4) ∧
  ¬(isIsosceles triangle3 ∧ isIsosceles triangle4) :=
sorry

end one_isosceles_triangle_l3527_352762


namespace bird_nest_twigs_l3527_352766

theorem bird_nest_twigs (circle_twigs : ℕ) (found_fraction : ℚ) (remaining_twigs : ℕ) :
  circle_twigs = 12 →
  found_fraction = 1 / 3 →
  remaining_twigs = 48 →
  (circle_twigs : ℚ) * (1 - found_fraction) * (circle_twigs : ℚ) = (remaining_twigs : ℚ) →
  circle_twigs * found_fraction * (circle_twigs : ℚ) + (remaining_twigs : ℚ) = 18 * (circle_twigs : ℚ) :=
by sorry

end bird_nest_twigs_l3527_352766


namespace frac_less_one_necessary_not_sufficient_l3527_352751

theorem frac_less_one_necessary_not_sufficient (a : ℝ) :
  (∀ a, a > 1 → 1/a < 1) ∧ 
  (∃ a, 1/a < 1 ∧ ¬(a > 1)) :=
sorry

end frac_less_one_necessary_not_sufficient_l3527_352751


namespace quadratic_roots_and_exponential_inequality_l3527_352757

theorem quadratic_roots_and_exponential_inequality (a : ℝ) :
  (∃ x : ℝ, x^2 + 8*x + a^2 = 0) ∧ 
  (∀ x : ℝ, Real.exp x + 1 / Real.exp x > a) →
  -4 ≤ a ∧ a < 2 :=
by sorry

end quadratic_roots_and_exponential_inequality_l3527_352757


namespace total_uniform_cost_is_355_l3527_352795

/-- Calculates the total cost of school uniforms for a student --/
def uniform_cost (num_uniforms : ℕ) (pants_cost : ℚ) (sock_cost : ℚ) : ℚ :=
  let shirt_cost := 2 * pants_cost
  let tie_cost := (1 / 5) * shirt_cost
  let single_uniform_cost := pants_cost + shirt_cost + tie_cost + sock_cost
  num_uniforms * single_uniform_cost

/-- Proves that the total cost of school uniforms for a student is $355 --/
theorem total_uniform_cost_is_355 :
  uniform_cost 5 20 3 = 355 := by
  sorry

#eval uniform_cost 5 20 3

end total_uniform_cost_is_355_l3527_352795


namespace absolute_value_zero_l3527_352713

theorem absolute_value_zero (x : ℚ) : |4*x + 6| = 0 ↔ x = -3/2 := by sorry

end absolute_value_zero_l3527_352713


namespace roots_problem_l3527_352747

theorem roots_problem :
  (∀ x : ℝ, x ^ 2 = 0 → x = 0) ∧
  (∃ x : ℝ, x ≥ 0 ∧ x ^ 2 = 9 ∧ ∀ y : ℝ, y ≥ 0 ∧ y ^ 2 = 9 → x = y) ∧
  (∃ x : ℝ, x ^ 3 = (64 : ℝ).sqrt ∧ ∀ y : ℝ, y ^ 3 = (64 : ℝ).sqrt → x = y) :=
by sorry

end roots_problem_l3527_352747


namespace ryan_owns_eleven_twentyfourths_l3527_352738

/-- The fraction of the total amount that Ryan owns -/
def ryan_fraction (total : ℚ) (leo_final : ℚ) (ryan_debt : ℚ) (leo_debt : ℚ) : ℚ :=
  1 - (leo_final + leo_debt - ryan_debt) / total

theorem ryan_owns_eleven_twentyfourths :
  let total : ℚ := 48
  let leo_final : ℚ := 19
  let ryan_debt : ℚ := 10
  let leo_debt : ℚ := 7
  ryan_fraction total leo_final ryan_debt leo_debt = 11 / 24 := by
sorry

end ryan_owns_eleven_twentyfourths_l3527_352738


namespace supplement_of_complement_of_36_degrees_l3527_352745

def angle_measure : ℝ := 36

def complement (x : ℝ) : ℝ := 90 - x

def supplement (x : ℝ) : ℝ := 180 - x

theorem supplement_of_complement_of_36_degrees : 
  supplement (complement angle_measure) = 126 := by
  sorry

end supplement_of_complement_of_36_degrees_l3527_352745


namespace infinite_series_sum_l3527_352792

/-- The sum of the infinite series ∑(n=1 to ∞) (5n-2)/(3^n) is equal to 11/4 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (5 * n - 2 : ℝ) / (3 ^ n)) = 11 / 4 := by
  sorry

end infinite_series_sum_l3527_352792


namespace equation_is_linear_l3527_352764

/-- A linear equation with two variables is of the form ax + by = c, where a, b, and c are constants, and a and b are not both zero. -/
def IsLinearEquationWithTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0), ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation 2x = 3y + 1 -/
def Equation (x y : ℝ) : Prop := 2 * x = 3 * y + 1

theorem equation_is_linear : IsLinearEquationWithTwoVariables Equation := by
  sorry

end equation_is_linear_l3527_352764


namespace machine_production_time_l3527_352732

/-- Given a machine that produces 150 items in 2 hours, 
    prove that it takes 0.8 minutes to produce one item. -/
theorem machine_production_time : 
  let total_items : ℕ := 150
  let total_hours : ℝ := 2
  let minutes_per_hour : ℝ := 60
  let total_minutes : ℝ := total_hours * minutes_per_hour
  total_minutes / total_items = 0.8 := by sorry

end machine_production_time_l3527_352732


namespace cube_surface_area_l3527_352775

theorem cube_surface_area (volume : ℝ) (h : volume = 64) : 
  (6 : ℝ) * (volume ^ (1/3 : ℝ))^2 = 96 := by
  sorry

end cube_surface_area_l3527_352775
