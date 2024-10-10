import Mathlib

namespace dilative_rotation_commutes_l2845_284548

/-- A transformation consisting of a rotation and scaling -/
structure DilativeRotation where
  center : ℝ × ℝ
  angle : ℝ
  scale : ℝ

/-- A triangle represented by its three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Apply a dilative rotation to a point -/
def applyDilativeRotation (t : DilativeRotation) (p : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Apply a dilative rotation to a triangle -/
def applyDilativeRotationToTriangle (t : DilativeRotation) (tri : Triangle) : Triangle :=
  sorry

/-- Theorem stating that the order of rotation and scaling is interchangeable -/
theorem dilative_rotation_commutes (t : DilativeRotation) (tri : Triangle) :
  let t1 := DilativeRotation.mk t.center t.angle 1
  let t2 := DilativeRotation.mk t.center 0 t.scale
  applyDilativeRotationToTriangle t2 (applyDilativeRotationToTriangle t1 tri) =
  applyDilativeRotationToTriangle t1 (applyDilativeRotationToTriangle t2 tri) :=
  sorry

end dilative_rotation_commutes_l2845_284548


namespace integral_absolute_value_l2845_284554

theorem integral_absolute_value : 
  ∫ x in (0 : ℝ)..2, (2 - |1 - x|) = 3 := by sorry

end integral_absolute_value_l2845_284554


namespace red_peach_count_l2845_284574

/-- Represents the count of peaches of different colors in a basket -/
structure PeachBasket where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Given a basket of peaches with 8 green peaches and 1 more green peach than red peaches,
    prove that there are 7 red peaches -/
theorem red_peach_count (basket : PeachBasket) 
    (green_count : basket.green = 8)
    (green_red_diff : basket.green = basket.red + 1) : 
  basket.red = 7 := by
  sorry

end red_peach_count_l2845_284574


namespace ball_box_distribution_l2845_284567

def num_balls : ℕ := 5
def num_boxes : ℕ := 5

/-- The number of ways to put all balls into boxes -/
def total_ways : ℕ := num_boxes ^ num_balls

/-- The number of ways to put balls into boxes with exactly one box left empty -/
def one_empty : ℕ := Nat.choose num_boxes 2 * Nat.factorial (num_balls - 1)

/-- The number of ways to put balls into boxes with exactly two boxes left empty -/
def two_empty : ℕ := 
  (Nat.choose num_boxes 2 * Nat.choose 3 2 * Nat.factorial (num_balls - 2) +
   Nat.choose num_boxes 3 * Nat.choose 2 1 * Nat.factorial (num_balls - 2)) * 
  Nat.factorial num_boxes / (Nat.factorial 2)

theorem ball_box_distribution :
  total_ways = 3125 ∧ one_empty = 1200 ∧ two_empty = 1500 := by
  sorry

end ball_box_distribution_l2845_284567


namespace only_setA_cannot_form_triangle_l2845_284546

-- Define a function to check if three line segments can form a triangle
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the sets of line segments
def setA : List ℝ := [4, 4, 9]
def setB : List ℝ := [3, 5, 6]
def setC : List ℝ := [6, 8, 10]
def setD : List ℝ := [5, 12, 13]

-- State the theorem
theorem only_setA_cannot_form_triangle :
  (¬ canFormTriangle setA[0] setA[1] setA[2]) ∧
  (canFormTriangle setB[0] setB[1] setB[2]) ∧
  (canFormTriangle setC[0] setC[1] setC[2]) ∧
  (canFormTriangle setD[0] setD[1] setD[2]) := by
  sorry

end only_setA_cannot_form_triangle_l2845_284546


namespace mork_mindy_tax_rate_l2845_284584

/-- Calculates the combined tax rate for Mork and Mindy -/
theorem mork_mindy_tax_rate (mork_income : ℝ) (mork_tax_rate : ℝ) (mindy_tax_rate : ℝ) :
  mork_tax_rate = 0.45 →
  mindy_tax_rate = 0.15 →
  let mindy_income := 4 * mork_income
  let combined_tax := mork_tax_rate * mork_income + mindy_tax_rate * mindy_income
  let combined_income := mork_income + mindy_income
  combined_tax / combined_income = 0.21 :=
by
  sorry

#check mork_mindy_tax_rate

end mork_mindy_tax_rate_l2845_284584


namespace math_only_count_l2845_284577

def brainiac_survey (total : ℕ) (rebus math logic : ℕ) 
  (rebus_math rebus_logic math_logic all_three neither : ℕ) : Prop :=
  total = 500 ∧
  rebus = 2 * math ∧
  logic = math ∧
  rebus_math = 72 ∧
  rebus_logic = 40 ∧
  math_logic = 36 ∧
  all_three = 10 ∧
  neither = 20

theorem math_only_count 
  (total rebus math logic rebus_math rebus_logic math_logic all_three neither : ℕ) :
  brainiac_survey total rebus math logic rebus_math rebus_logic math_logic all_three neither →
  math - rebus_math - math_logic + all_three = 54 :=
by sorry

end math_only_count_l2845_284577


namespace tire_usage_theorem_l2845_284519

/-- Represents the usage of tires on a car --/
structure TireUsage where
  total_tires : ℕ
  road_tires : ℕ
  total_miles : ℕ

/-- Calculates the miles each tire was used given equal usage --/
def miles_per_tire (usage : TireUsage) : ℕ :=
  (usage.total_miles * usage.road_tires) / usage.total_tires

/-- Theorem stating that for the given car configuration and mileage, each tire was used for 33333 miles --/
theorem tire_usage_theorem (usage : TireUsage) 
  (h1 : usage.total_tires = 6)
  (h2 : usage.road_tires = 4)
  (h3 : usage.total_miles = 50000) :
  miles_per_tire usage = 33333 := by
  sorry

end tire_usage_theorem_l2845_284519


namespace expression_equals_one_l2845_284564

theorem expression_equals_one (b : ℝ) (hb : b ≠ 0) :
  ∀ x : ℝ, x ≠ b ∧ x ≠ -b →
    (b / (b - x) - x / (b + x)) / (b / (b + x) + x / (b - x)) = 1 :=
by sorry

end expression_equals_one_l2845_284564


namespace cube_volume_equals_surface_area_l2845_284552

/-- For a cube with side length s, if the volume is equal to the surface area, then s = 6. -/
theorem cube_volume_equals_surface_area (s : ℝ) (h : s > 0) :
  s^3 = 6 * s^2 → s = 6 := by
  sorry

end cube_volume_equals_surface_area_l2845_284552


namespace value_after_seven_years_l2845_284513

/-- Calculates the value after n years given initial value, annual increase rate, inflation rate, and tax rate -/
def value_after_years (initial_value : ℝ) (increase_rate : ℝ) (inflation_rate : ℝ) (tax_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * ((1 - tax_rate) * (1 - inflation_rate) * (1 + increase_rate)) ^ years

/-- Theorem stating that the value after 7 years is approximately 126469.75 -/
theorem value_after_seven_years :
  let initial_value : ℝ := 59000
  let increase_rate : ℝ := 1/8
  let inflation_rate : ℝ := 0.03
  let tax_rate : ℝ := 0.07
  let years : ℕ := 7
  abs (value_after_years initial_value increase_rate inflation_rate tax_rate years - 126469.75) < 0.01 := by
  sorry

end value_after_seven_years_l2845_284513


namespace fraction_cubed_l2845_284595

theorem fraction_cubed : (2 / 3 : ℚ) ^ 3 = 8 / 27 := by
  sorry

end fraction_cubed_l2845_284595


namespace roots_are_imaginary_l2845_284506

theorem roots_are_imaginary (k : ℝ) : 
  let quadratic (x : ℝ) := x^2 - 3*k*x + 2*k^2 - 1
  ∀ r₁ r₂ : ℝ, quadratic r₁ = 0 ∧ quadratic r₂ = 0 → r₁ * r₂ = 8 →
  ∃ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ 
    (∀ x : ℝ, quadratic x = 0 ↔ x = Complex.mk a b ∨ x = Complex.mk a (-b)) :=
by sorry

end roots_are_imaginary_l2845_284506


namespace cucumber_packing_l2845_284596

theorem cucumber_packing (total_cucumbers : ℕ) (basket_capacity : ℕ) 
  (h1 : total_cucumbers = 216)
  (h2 : basket_capacity = 23) :
  ∃ (filled_baskets : ℕ) (remaining_cucumbers : ℕ),
    filled_baskets * basket_capacity + remaining_cucumbers = total_cucumbers ∧
    filled_baskets = 9 ∧
    remaining_cucumbers = 9 := by
  sorry

end cucumber_packing_l2845_284596


namespace max_value_fraction_l2845_284580

theorem max_value_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : b^2 + 2*(a + c)*b - a*c = 0) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → y^2 + 2*(x + z)*y - x*z = 0 → 
  y / (x + z) ≤ b / (a + c) → b / (a + c) ≤ (Real.sqrt 5 - 2) / 2 :=
by sorry

end max_value_fraction_l2845_284580


namespace concert_ticket_ratio_l2845_284594

theorem concert_ticket_ratio (initial_amount : ℚ) (motorcycle_cost : ℚ) (final_amount : ℚ)
  (h1 : initial_amount = 5000)
  (h2 : motorcycle_cost = 2800)
  (h3 : final_amount = 825)
  (h4 : ∃ (concert_cost : ℚ),
    final_amount = (initial_amount - motorcycle_cost - concert_cost) * (3/4)) :
  ∃ (concert_cost : ℚ),
    concert_cost / (initial_amount - motorcycle_cost) = 1 / 2 := by
  sorry

end concert_ticket_ratio_l2845_284594


namespace min_value_theorem_l2845_284525

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ min :=
sorry

end min_value_theorem_l2845_284525


namespace necessary_not_sufficient_condition_l2845_284562

theorem necessary_not_sufficient_condition (x : ℝ) : 
  (x < 4 → x < 0) ∧ ¬(x < 0 → x < 4) := by sorry

end necessary_not_sufficient_condition_l2845_284562


namespace exists_non_increasing_exponential_l2845_284563

theorem exists_non_increasing_exponential : 
  ∃ (a : ℝ), a > 0 ∧ ¬(∀ x y : ℝ, x < y → (a^(-x) : ℝ) < a^(-y)) :=
sorry

end exists_non_increasing_exponential_l2845_284563


namespace no_double_square_sum_l2845_284557

theorem no_double_square_sum (x y : ℕ) : 
  ¬(∃ (a b : ℕ), a^2 = x^2 + y ∧ b^2 = y^2 + x) := by
  sorry

end no_double_square_sum_l2845_284557


namespace baseball_game_total_baseball_game_total_is_643_l2845_284590

/-- Represents the statistics of a baseball team for a single day -/
structure DayStats where
  misses : ℕ
  hits : ℕ
  singles : ℕ
  doubles : ℕ
  triples : ℕ
  homeRuns : ℕ

/-- Represents the statistics of a baseball team for three days -/
structure TeamStats where
  day1 : DayStats
  day2 : DayStats
  day3 : DayStats

theorem baseball_game_total (teamA teamB : TeamStats) : ℕ :=
  let totalMisses := teamA.day1.misses + teamA.day2.misses + teamA.day3.misses +
                     teamB.day1.misses + teamB.day2.misses + teamB.day3.misses
  let totalSingles := teamA.day1.singles + teamA.day2.singles + teamA.day3.singles +
                      teamB.day1.singles + teamB.day2.singles + teamB.day3.singles
  let totalDoubles := teamA.day1.doubles + teamA.day2.doubles + teamA.day3.doubles +
                      teamB.day1.doubles + teamB.day2.doubles + teamB.day3.doubles
  let totalTriples := teamA.day1.triples + teamA.day2.triples + teamA.day3.triples +
                      teamB.day1.triples + teamB.day2.triples + teamB.day3.triples
  let totalHomeRuns := teamA.day1.homeRuns + teamA.day2.homeRuns + teamA.day3.homeRuns +
                       teamB.day1.homeRuns + teamB.day2.homeRuns + teamB.day3.homeRuns
  totalMisses + totalSingles + totalDoubles + totalTriples + totalHomeRuns

theorem baseball_game_total_is_643 :
  let teamA : TeamStats := {
    day1 := { misses := 60, hits := 30, singles := 15, doubles := 0, triples := 0, homeRuns := 15 },
    day2 := { misses := 68, hits := 17, singles := 11, doubles := 6, triples := 0, homeRuns := 0 },
    day3 := { misses := 100, hits := 20, singles := 10, doubles := 0, triples := 5, homeRuns := 5 }
  }
  let teamB : TeamStats := {
    day1 := { misses := 90, hits := 30, singles := 15, doubles := 0, triples := 0, homeRuns := 15 },
    day2 := { misses := 56, hits := 28, singles := 19, doubles := 9, triples := 0, homeRuns := 0 },
    day3 := { misses := 120, hits := 24, singles := 12, doubles := 0, triples := 6, homeRuns := 6 }
  }
  baseball_game_total teamA teamB = 643 := by
  sorry

#check baseball_game_total_is_643

end baseball_game_total_baseball_game_total_is_643_l2845_284590


namespace players_joined_equals_two_l2845_284507

/-- The number of players who joined an online game --/
def players_joined (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  (total_lives / lives_per_player) - initial_players

/-- Theorem: The number of players who joined the game is 2 --/
theorem players_joined_equals_two :
  players_joined 2 6 24 = 2 := by
  sorry

end players_joined_equals_two_l2845_284507


namespace fraction_equality_l2845_284578

theorem fraction_equality (a b : ℚ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end fraction_equality_l2845_284578


namespace lioness_age_l2845_284575

theorem lioness_age (hyena_age lioness_age : ℕ) : 
  lioness_age = 2 * hyena_age →
  (hyena_age / 2 + 5) + (lioness_age / 2 + 5) = 19 →
  lioness_age = 12 := by
  sorry

end lioness_age_l2845_284575


namespace floor_ceiling_sum_l2845_284527

theorem floor_ceiling_sum : ⌊(3.67 : ℝ)⌋ + ⌈(-14.2 : ℝ)⌉ = -11 := by sorry

end floor_ceiling_sum_l2845_284527


namespace product_pure_imaginary_implies_magnitude_l2845_284533

open Complex

theorem product_pure_imaginary_implies_magnitude (b : ℝ) :
  (((2 : ℂ) + b * I) * ((1 : ℂ) - I)).re = 0 ∧
  (((2 : ℂ) + b * I) * ((1 : ℂ) - I)).im ≠ 0 →
  abs ((1 : ℂ) + b * I) = Real.sqrt 5 := by
  sorry

end product_pure_imaginary_implies_magnitude_l2845_284533


namespace quadratic_function_range_l2845_284585

/-- A quadratic function with two distinct zeros -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  has_distinct_zeros : ∃ (x y : ℝ), x ≠ y ∧ x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0

/-- Four distinct roots in arithmetic progression -/
structure FourRoots where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  distinct : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄
  arithmetic : ∃ (d : ℝ), x₂ - x₁ = d ∧ x₃ - x₂ = d ∧ x₄ - x₃ = d

/-- The main theorem -/
theorem quadratic_function_range (f : QuadraticFunction) (roots : FourRoots) 
  (h : ∀ x, (x^2 + 2*x - 1)^2 + f.a*(x^2 + 2*x - 1) + f.b = 0 ↔ 
           x = roots.x₁ ∨ x = roots.x₂ ∨ x = roots.x₃ ∨ x = roots.x₄) :
  ∀ x, x ≤ 25/9 ∧ (∃ y, f.a - f.b = y) :=
sorry

end quadratic_function_range_l2845_284585


namespace combined_fuel_efficiency_l2845_284524

theorem combined_fuel_efficiency
  (m : ℝ) -- distance driven by each car
  (h_pos : m > 0) -- ensure distance is positive
  (efficiency_linda : ℝ := 30) -- Linda's car efficiency
  (efficiency_joe : ℝ := 15) -- Joe's car efficiency
  (efficiency_anne : ℝ := 20) -- Anne's car efficiency
  : (3 * m) / (m / efficiency_linda + m / efficiency_joe + m / efficiency_anne) = 20 :=
by sorry

end combined_fuel_efficiency_l2845_284524


namespace factorization_equality_l2845_284579

theorem factorization_equality (m n : ℝ) : m^2 * n - 16 * n = n * (m + 4) * (m - 4) := by
  sorry

end factorization_equality_l2845_284579


namespace taylor_family_reunion_tables_l2845_284565

theorem taylor_family_reunion_tables (num_kids : ℕ) (num_adults : ℕ) (people_per_table : ℕ) : 
  num_kids = 45 → num_adults = 123 → people_per_table = 12 → 
  (num_kids + num_adults) / people_per_table = 14 := by
sorry

end taylor_family_reunion_tables_l2845_284565


namespace min_value_of_f_l2845_284538

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - Real.sqrt 3 * abs x + 1) + Real.sqrt (x^2 + Real.sqrt 3 * abs x + 3)

theorem min_value_of_f :
  (∀ x : ℝ, f x ≥ Real.sqrt 7) ∧
  f (Real.sqrt 3 / 4) = Real.sqrt 7 ∧
  f (-Real.sqrt 3 / 4) = Real.sqrt 7 :=
sorry

end min_value_of_f_l2845_284538


namespace floor_length_calculation_l2845_284526

theorem floor_length_calculation (floor_width : ℝ) (strip_width : ℝ) (rug_area : ℝ) :
  floor_width = 20 →
  strip_width = 4 →
  rug_area = 204 →
  (floor_width - 2 * strip_width) * (floor_length - 2 * strip_width) = rug_area →
  floor_length = 25 :=
by
  sorry

end floor_length_calculation_l2845_284526


namespace circle_equation_given_conditions_l2845_284598

/-- A circle C in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle given its center and radius -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- A circle is tangent to the y-axis if its center's x-coordinate equals its radius -/
def tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius ∨ c.center.1 = -c.radius

/-- A point (x, y) lies on the line x - 3y = 0 -/
def on_line (x y : ℝ) : Prop :=
  x - 3*y = 0

theorem circle_equation_given_conditions :
  ∀ (C : Circle),
    tangent_to_y_axis C →
    C.radius = 4 →
    on_line C.center.1 C.center.2 →
    ∀ (x y : ℝ),
      circle_equation C x y ↔ 
        ((x - 4)^2 + (y - 4/3)^2 = 16 ∨ (x + 4)^2 + (y + 4/3)^2 = 16) :=
by sorry

end circle_equation_given_conditions_l2845_284598


namespace amount_after_two_years_l2845_284511

/-- The final amount after compound interest --/
def final_amount (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate) ^ years

/-- The problem statement --/
theorem amount_after_two_years :
  let initial := 2880
  let rate := 1 / 8
  let years := 2
  final_amount initial rate years = 3645 := by
sorry

end amount_after_two_years_l2845_284511


namespace pizza_diameter_increase_l2845_284559

theorem pizza_diameter_increase (d : ℝ) (D : ℝ) (h : d > 0) (h' : D > 0) :
  (π * (D / 2)^2 = 1.96 * π * (d / 2)^2) →
  (D = 1.4 * d) := by
sorry

end pizza_diameter_increase_l2845_284559


namespace savings_difference_l2845_284592

def initial_amount : ℝ := 10000

def option1_discounts : List ℝ := [0.20, 0.20, 0.10]
def option2_discounts : List ℝ := [0.40, 0.05, 0.05]

def apply_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ acc d => acc * (1 - d)) amount

theorem savings_difference : 
  apply_discounts initial_amount option1_discounts - 
  apply_discounts initial_amount option2_discounts = 345 := by
  sorry

end savings_difference_l2845_284592


namespace inequality_range_l2845_284588

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ m ∈ Set.Ioo (-10) 2 ∪ {2} :=
sorry

end inequality_range_l2845_284588


namespace exponent_simplification_l2845_284556

theorem exponent_simplification :
  3000 * (3000 ^ 2500) * 2 = 2 * 3000 ^ 2501 := by
  sorry

end exponent_simplification_l2845_284556


namespace video_game_shelves_l2845_284535

/-- Calculates the minimum number of shelves needed to display video games -/
def minimum_shelves_needed (total_games : ℕ) (action_games : ℕ) (adventure_games : ℕ) (simulation_games : ℕ) (shelf_capacity : ℕ) (special_display_per_genre : ℕ) : ℕ :=
  let remaining_action := action_games - special_display_per_genre
  let remaining_adventure := adventure_games - special_display_per_genre
  let remaining_simulation := simulation_games - special_display_per_genre
  let action_shelves := (remaining_action + shelf_capacity - 1) / shelf_capacity
  let adventure_shelves := (remaining_adventure + shelf_capacity - 1) / shelf_capacity
  let simulation_shelves := (remaining_simulation + shelf_capacity - 1) / shelf_capacity
  action_shelves + adventure_shelves + simulation_shelves + 1

theorem video_game_shelves :
  minimum_shelves_needed 163 73 51 39 84 10 = 4 := by
  sorry

end video_game_shelves_l2845_284535


namespace quadratic_real_solutions_l2845_284549

theorem quadratic_real_solutions (m : ℝ) : 
  (∀ x : ℝ, x^2 + x + m = 0 → ∃ y : ℝ, y^2 + y + m = 0) ∧ 
  (∃ n : ℝ, n ≥ 1/4 ∧ ∃ z : ℝ, z^2 + z + n = 0) :=
by sorry

end quadratic_real_solutions_l2845_284549


namespace gunther_typing_capacity_l2845_284582

/-- Gunther's typing rate in words per 3 minutes -/
def typing_rate : ℕ := 160

/-- Number of minutes in 3 minutes -/
def minutes_per_unit : ℕ := 3

/-- Number of minutes Gunther works per day -/
def working_minutes : ℕ := 480

/-- Number of words Gunther can type in a working day -/
def words_per_day : ℕ := 25598

theorem gunther_typing_capacity :
  (typing_rate : ℚ) / minutes_per_unit * working_minutes = words_per_day := by
  sorry

end gunther_typing_capacity_l2845_284582


namespace company_fund_problem_l2845_284502

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  initial_fund = 60 * n - 10 →  -- The fund initially contained $10 less than needed for $60 bonuses
  initial_fund = 55 * n + 120 → -- Each employee received $55, and $120 remained
  initial_fund = 1550 :=        -- The initial fund amount was $1550
by sorry

end company_fund_problem_l2845_284502


namespace problem_solid_surface_area_l2845_284545

/-- Represents a solid constructed from unit cubes -/
structure CubeSolid where
  bottomRow : ℕ
  middleColumn : ℕ
  leftColumns : ℕ
  leftColumnHeight : ℕ

/-- Calculates the surface area of the CubeSolid -/
def surfaceArea (solid : CubeSolid) : ℕ :=
  let bottomArea := solid.bottomRow + 2 * (solid.bottomRow + 1)
  let middleColumnArea := 4 + (solid.middleColumn - 1)
  let leftColumnsArea := 2 * (2 * solid.leftColumnHeight + 1)
  bottomArea + middleColumnArea + leftColumnsArea

/-- The specific solid described in the problem -/
def problemSolid : CubeSolid :=
  { bottomRow := 5
  , middleColumn := 5
  , leftColumns := 2
  , leftColumnHeight := 3 }

theorem problem_solid_surface_area :
  surfaceArea problemSolid = 34 := by
  sorry

#eval surfaceArea problemSolid

end problem_solid_surface_area_l2845_284545


namespace fraction_meaningful_range_l2845_284529

theorem fraction_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y = 2 / (x + 3)) → x ≠ -3 := by
  sorry

end fraction_meaningful_range_l2845_284529


namespace similar_triangle_point_coordinates_l2845_284558

structure Triangle :=
  (O : ℝ × ℝ)
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)

def similar_triangle (T1 T2 : Triangle) (ratio : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), 
    (T2.A.1 - center.1 = ratio * (T1.A.1 - center.1)) ∧
    (T2.A.2 - center.2 = ratio * (T1.A.2 - center.2)) ∧
    (T2.B.1 - center.1 = ratio * (T1.B.1 - center.1)) ∧
    (T2.B.2 - center.2 = ratio * (T1.B.2 - center.2))

theorem similar_triangle_point_coordinates 
  (a : ℝ) 
  (OAB : Triangle) 
  (OCD : Triangle) 
  (h1 : OAB.O = (0, 0)) 
  (h2 : OAB.A = (4, 3)) 
  (h3 : OAB.B = (3, a)) 
  (h4 : similar_triangle OAB OCD (1/3)) 
  (h5 : OCD.O = (0, 0)) :
  OCD.A = (4/3, 1) ∨ OCD.A = (-4/3, -1) :=
sorry

end similar_triangle_point_coordinates_l2845_284558


namespace tangent_triangle_angle_theorem_l2845_284583

-- Define the circle
variable (O : Point)

-- Define the triangle
variable (P A B : Point)

-- Define the property that PAB is formed by tangents to circle O
def is_tangent_triangle (O P A B : Point) : Prop := sorry

-- Define the measure of an angle
def angle_measure (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem tangent_triangle_angle_theorem 
  (h_tangent : is_tangent_triangle O P A B)
  (h_angle : angle_measure A P B = 50) :
  angle_measure A O B = 65 := by sorry

end tangent_triangle_angle_theorem_l2845_284583


namespace ellipse_equation_l2845_284505

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    if its eccentricity is √3/2 and the distance from one endpoint of
    the minor axis to the right focus is 2, then its equation is x²/4 + y² = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := (Real.sqrt 3) / 2
  let d := 2
  (e = Real.sqrt (1 - b^2 / a^2) ∧ d = a) →
  a^2 = 4 ∧ b^2 = 1 := by sorry

end ellipse_equation_l2845_284505


namespace sign_selection_theorem_l2845_284568

theorem sign_selection_theorem (n : ℕ) (a : ℕ → ℕ) 
  (h_n : n ≥ 2)
  (h_a : ∀ k ∈ Finset.range n, 0 < a k ∧ a k ≤ k + 1)
  (h_even : Even (Finset.sum (Finset.range n) a)) :
  ∃ f : ℕ → Int, (∀ k, f k = 1 ∨ f k = -1) ∧ 
    Finset.sum (Finset.range n) (λ k => (f k) * (a k)) = 0 := by
  sorry

end sign_selection_theorem_l2845_284568


namespace full_time_one_year_count_l2845_284520

/-- Represents the number of employees in different categories at company x -/
structure CompanyEmployees where
  total : ℕ
  fullTime : ℕ
  atLeastOneYear : ℕ
  neitherFullTimeNorOneYear : ℕ

/-- The function to calculate the number of full-time employees who have worked at least one year -/
def fullTimeAndOneYear (e : CompanyEmployees) : ℕ :=
  e.total - (e.fullTime + e.atLeastOneYear - e.neitherFullTimeNorOneYear)

/-- Theorem stating the number of full-time employees who have worked at least one year -/
theorem full_time_one_year_count (e : CompanyEmployees) 
  (h1 : e.total = 130)
  (h2 : e.fullTime = 80)
  (h3 : e.atLeastOneYear = 100)
  (h4 : e.neitherFullTimeNorOneYear = 20) :
  fullTimeAndOneYear e = 90 := by
  sorry

end full_time_one_year_count_l2845_284520


namespace min_orange_weight_l2845_284500

theorem min_orange_weight (a o : ℝ) 
  (h1 : a ≥ 8 + 3 * o) 
  (h2 : a ≤ 4 * o) : 
  o ≥ 8 := by
  sorry

end min_orange_weight_l2845_284500


namespace intersection_condition_l2845_284509

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1^2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - a)^2 ≤ 1}

-- State the theorem
theorem intersection_condition (a : ℝ) :
  M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end intersection_condition_l2845_284509


namespace inequality_solution_set_l2845_284515

theorem inequality_solution_set :
  {x : ℝ | -1/3 * x + 1 ≤ -5} = {x : ℝ | x ≥ 18} := by
  sorry

end inequality_solution_set_l2845_284515


namespace porter_monthly_earnings_l2845_284591

/-- Porter's daily wage in dollars -/
def daily_wage : ℕ := 8

/-- Number of regular working days per week -/
def regular_days : ℕ := 5

/-- Overtime bonus rate as a percentage -/
def overtime_bonus_rate : ℕ := 50

/-- Number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Calculate Porter's monthly earnings with overtime every week -/
def monthly_earnings_with_overtime : ℕ :=
  let regular_weekly_earnings := daily_wage * regular_days
  let overtime_daily_earnings := daily_wage + (daily_wage * overtime_bonus_rate / 100)
  let weekly_earnings_with_overtime := regular_weekly_earnings + overtime_daily_earnings
  weekly_earnings_with_overtime * weeks_per_month

theorem porter_monthly_earnings :
  monthly_earnings_with_overtime = 208 := by
  sorry

end porter_monthly_earnings_l2845_284591


namespace max_value_on_circle_l2845_284593

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 16*x + 8*y + 8 →
  4*x + 3*y ≤ 63 :=
by sorry

end max_value_on_circle_l2845_284593


namespace exists_arrangement_with_more_than_five_holes_l2845_284541

/-- Represents a strange ring, which is a circle with a square hole in the middle. -/
structure StrangeRing where
  circle_radius : ℝ
  square_side : ℝ
  center : ℝ × ℝ
  h_square_fits : square_side ≤ 2 * circle_radius

/-- Represents an arrangement of two strange rings on a table. -/
structure StrangeRingArrangement where
  ring1 : StrangeRing
  ring2 : StrangeRing
  placement : ℝ × ℝ  -- Relative placement of ring2 with respect to ring1

/-- Counts the number of holes in a given arrangement of strange rings. -/
def count_holes (arrangement : StrangeRingArrangement) : ℕ :=
  sorry

/-- Theorem stating that there exists an arrangement of two strange rings
    that results in more than 5 holes. -/
theorem exists_arrangement_with_more_than_five_holes :
  ∃ (arrangement : StrangeRingArrangement), count_holes arrangement > 5 :=
sorry

end exists_arrangement_with_more_than_five_holes_l2845_284541


namespace dress_designs_count_l2845_284514

/-- The number of color choices available for a dress design. -/
def num_colors : ℕ := 5

/-- The number of pattern choices available for a dress design. -/
def num_patterns : ℕ := 4

/-- The number of accessory choices available for a dress design. -/
def num_accessories : ℕ := 2

/-- The total number of possible dress designs. -/
def total_designs : ℕ := num_colors * num_patterns * num_accessories

/-- Theorem stating that the total number of possible dress designs is 40. -/
theorem dress_designs_count : total_designs = 40 := by
  sorry

end dress_designs_count_l2845_284514


namespace chocolate_milk_consumption_l2845_284587

theorem chocolate_milk_consumption (milk_per_glass : ℝ) (syrup_per_glass : ℝ) 
  (total_milk : ℝ) (total_syrup : ℝ) : 
  milk_per_glass = 6.5 → 
  syrup_per_glass = 1.5 → 
  total_milk = 130 → 
  total_syrup = 60 → 
  let glasses_from_milk := total_milk / milk_per_glass
  let glasses_from_syrup := total_syrup / syrup_per_glass
  let glasses_made := min glasses_from_milk glasses_from_syrup
  let total_consumption := glasses_made * (milk_per_glass + syrup_per_glass)
  total_consumption = 160 := by
  sorry

#check chocolate_milk_consumption

end chocolate_milk_consumption_l2845_284587


namespace smallest_five_digit_divisible_by_53_l2845_284512

theorem smallest_five_digit_divisible_by_53 : ∀ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) → -- five-digit number condition
  n % 53 = 0 → -- divisibility by 53 condition
  n ≥ 10017 := by
sorry

end smallest_five_digit_divisible_by_53_l2845_284512


namespace root_implies_m_value_l2845_284503

theorem root_implies_m_value (x m : ℝ) : 
  x = 2 → x^2 - m*x + 6 = 0 → m = 5 := by
  sorry

end root_implies_m_value_l2845_284503


namespace stream_speed_l2845_284521

/-- Given upstream and downstream speeds of a canoe, calculate the speed of the stream. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 9)
  (h_downstream : downstream_speed = 12) :
  (downstream_speed - upstream_speed) / 2 = 1.5 := by
  sorry

end stream_speed_l2845_284521


namespace min_value_of_expression_l2845_284561

theorem min_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ x y : ℝ, 2*a*x - b*y + 2 = 0 ∧ x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∃ x1 y1 x2 y2 : ℝ, 2*a*x1 - b*y1 + 2 = 0 ∧ x1^2 + y1^2 + 2*x1 - 4*y1 + 1 = 0 ∧
                      2*a*x2 - b*y2 + 2 = 0 ∧ x2^2 + y2^2 + 2*x2 - 4*y2 + 1 = 0 ∧
                      (x1 - x2)^2 + (y1 - y2)^2 = 16) →
  (4/a + 1/b ≥ 9 ∧ ∃ a0 b0 : ℝ, a0 > 0 ∧ b0 > 0 ∧ 4/a0 + 1/b0 = 9) :=
by sorry

end min_value_of_expression_l2845_284561


namespace sum_of_coefficients_l2845_284523

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (5*x - 4)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 25 := by
sorry

end sum_of_coefficients_l2845_284523


namespace factorization_equality_l2845_284589

theorem factorization_equality (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

end factorization_equality_l2845_284589


namespace sin_75_degrees_l2845_284539

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end sin_75_degrees_l2845_284539


namespace quaternary_201_equals_33_l2845_284597

/-- Converts a quaternary (base-4) number to its decimal (base-10) equivalent -/
def quaternary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

theorem quaternary_201_equals_33 :
  quaternary_to_decimal [1, 0, 2] = 33 := by
  sorry

end quaternary_201_equals_33_l2845_284597


namespace repeating_decimal_135_equals_5_37_l2845_284570

def repeating_decimal (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

theorem repeating_decimal_135_equals_5_37 :
  repeating_decimal 1 3 5 = 5 / 37 := by
  sorry

end repeating_decimal_135_equals_5_37_l2845_284570


namespace sqrt_27_plus_sqrt_75_l2845_284528

theorem sqrt_27_plus_sqrt_75 : Real.sqrt 27 + Real.sqrt 75 = 8 * Real.sqrt 3 := by
  sorry

end sqrt_27_plus_sqrt_75_l2845_284528


namespace divisible_by_eleven_l2845_284543

theorem divisible_by_eleven (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∃ k : ℤ, (n^2 + 4^n + 7^n : ℤ) = k * n) : 
  ∃ m : ℤ, (n^2 + 4^n + 7^n : ℤ) / n = 11 * m := by
  sorry

end divisible_by_eleven_l2845_284543


namespace three_classes_five_spots_l2845_284531

/-- The number of ways for classes to choose scenic spots -/
def num_selection_methods (num_classes : ℕ) (num_spots : ℕ) : ℕ :=
  num_spots ^ num_classes

/-- Theorem: Three classes choosing from five scenic spots results in 5^3 selection methods -/
theorem three_classes_five_spots : num_selection_methods 3 5 = 5^3 := by
  sorry

end three_classes_five_spots_l2845_284531


namespace intersection_solution_set_l2845_284566

theorem intersection_solution_set (a b : ℝ) : 
  (∀ x, x^2 + a*x + b < 0 ↔ (x^2 - 2*x - 3 < 0 ∧ x^2 + x - 6 < 0)) →
  a + b = -3 := by
sorry

end intersection_solution_set_l2845_284566


namespace box_side_length_l2845_284540

/-- Given the total volume needed, cost per box, and minimum total cost,
    calculate the length of one side of a cubic box. -/
theorem box_side_length 
  (total_volume : ℝ) 
  (cost_per_box : ℝ) 
  (min_total_cost : ℝ) 
  (h1 : total_volume = 1920000) 
  (h2 : cost_per_box = 0.5) 
  (h3 : min_total_cost = 200) : 
  ∃ (side_length : ℝ), abs (side_length - 16.89) < 0.01 := by
  sorry

#check box_side_length

end box_side_length_l2845_284540


namespace exists_far_reaching_quadrilateral_with_bounded_area_l2845_284501

/-- A point in the 2D plane with integer coordinates. -/
structure Point where
  x : ℤ
  y : ℤ

/-- A rectangle defined by its width and height. -/
structure Rectangle where
  width : ℤ
  height : ℤ

/-- A quadrilateral defined by its four vertices. -/
structure Quadrilateral where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Predicate to check if a point is on or inside a rectangle. -/
def pointInRectangle (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

/-- Predicate to check if a quadrilateral is far-reaching in a rectangle. -/
def isFarReaching (q : Quadrilateral) (r : Rectangle) : Prop :=
  (pointInRectangle q.v1 r ∧ pointInRectangle q.v2 r ∧ pointInRectangle q.v3 r ∧ pointInRectangle q.v4 r) ∧
  (q.v1.x = 0 ∨ q.v2.x = 0 ∨ q.v3.x = 0 ∨ q.v4.x = 0) ∧
  (q.v1.y = 0 ∨ q.v2.y = 0 ∨ q.v3.y = 0 ∨ q.v4.y = 0) ∧
  (q.v1.x = r.width ∨ q.v2.x = r.width ∨ q.v3.x = r.width ∨ q.v4.x = r.width) ∧
  (q.v1.y = r.height ∨ q.v2.y = r.height ∨ q.v3.y = r.height ∨ q.v4.y = r.height)

/-- Calculate the area of a quadrilateral. -/
def quadrilateralArea (q : Quadrilateral) : ℚ :=
  sorry  -- The actual area calculation would go here

/-- The main theorem to be proved. -/
theorem exists_far_reaching_quadrilateral_with_bounded_area
  (n m : ℕ) (hn : n ≤ 10^10) (hm : m ≤ 10^10) :
  ∃ (q : Quadrilateral), isFarReaching q (Rectangle.mk n m) ∧ quadrilateralArea q ≤ 10^6 := by
  sorry

end exists_far_reaching_quadrilateral_with_bounded_area_l2845_284501


namespace modulus_of_z_l2845_284508

theorem modulus_of_z (i : ℂ) (h : i^2 = -1) : 
  let z : ℂ := i / (1 + i)
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end modulus_of_z_l2845_284508


namespace equal_distance_to_axes_l2845_284551

theorem equal_distance_to_axes (m : ℝ) : 
  let P : ℝ × ℝ := (3*m + 1, 2*m - 5)
  (|P.1| = |P.2|) ↔ (m = -6 ∨ m = 4/5) := by
  sorry

end equal_distance_to_axes_l2845_284551


namespace quinn_free_donuts_l2845_284542

/-- Calculates the number of free donuts earned in a summer reading challenge -/
def free_donuts (books_per_week : ℕ) (weeks : ℕ) (books_per_coupon : ℕ) : ℕ :=
  (books_per_week * weeks) / books_per_coupon

/-- Proves that Quinn is eligible for 4 free donuts -/
theorem quinn_free_donuts : free_donuts 2 10 5 = 4 := by
  sorry

end quinn_free_donuts_l2845_284542


namespace bowling_team_weight_specific_bowling_problem_l2845_284537

/-- Given a bowling team with initial players and weights, prove the weight of a new player --/
theorem bowling_team_weight (initial_players : ℕ) (initial_avg_weight : ℝ) 
  (new_player1_weight : ℝ) (new_avg_weight : ℝ) : ℝ :=
  let total_initial_weight := initial_players * initial_avg_weight
  let new_total_players := initial_players + 2
  let new_total_weight := new_total_players * new_avg_weight
  let new_players_total_weight := new_total_weight - total_initial_weight
  let new_player2_weight := new_players_total_weight - new_player1_weight
  new_player2_weight

/-- The specific bowling team problem --/
theorem specific_bowling_problem : 
  bowling_team_weight 7 76 110 78 = 60 := by
  sorry

end bowling_team_weight_specific_bowling_problem_l2845_284537


namespace symmetric_points_count_l2845_284581

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if three points are distinct and non-collinear -/
def distinct_non_collinear (M G T : Point2D) : Prop :=
  M ≠ G ∧ M ≠ T ∧ G ≠ T ∧
  (G.x - M.x) * (T.y - M.y) ≠ (T.x - M.x) * (G.y - M.y)

/-- Check if a figure has an axis of symmetry -/
def has_axis_of_symmetry (points : List Point2D) : Prop :=
  sorry  -- Definition omitted for brevity

/-- Count the number of distinct points U that create a figure with symmetry -/
def count_symmetric_points (M G T : Point2D) : ℕ :=
  sorry  -- Definition omitted for brevity

/-- The main theorem -/
theorem symmetric_points_count 
  (M G T : Point2D) 
  (h1 : distinct_non_collinear M G T) 
  (h2 : ¬ has_axis_of_symmetry [M, G, T]) :
  count_symmetric_points M G T = 5 ∨ count_symmetric_points M G T = 6 :=
sorry

end symmetric_points_count_l2845_284581


namespace base_8_of_2023_l2845_284534

/-- Converts a base-10 number to its base-8 representation -/
def toBase8 (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The base-8 representation of 2023 (base 10) is 3747 -/
theorem base_8_of_2023 : toBase8 2023 = 3747 := by
  sorry

end base_8_of_2023_l2845_284534


namespace recurrence_sequence_eventually_periodic_l2845_284550

/-- A sequence of integers satisfying the given recurrence relation -/
def RecurrenceSequence (u : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 4 → u n = (u (n-1) + u (n-2) + u (n-3) * u (n-4)) / (u (n-1) * u (n-2) + u (n-3) + u (n-4))

/-- A sequence is bounded if there exist m and M such that m ≤ u_n ≤ M for all n -/
def IsBounded (u : ℕ → ℤ) : Prop :=
  ∃ m M : ℤ, ∀ n : ℕ, m ≤ u n ∧ u n ≤ M

/-- A sequence is eventually periodic if there exist N and p such that u_{n+p} = u_n for all n ≥ N -/
def EventuallyPeriodic (u : ℕ → ℤ) : Prop :=
  ∃ N p : ℕ, p > 0 ∧ ∀ n : ℕ, n ≥ N → u (n + p) = u n

/-- The main theorem: a bounded recurrence sequence is eventually periodic -/
theorem recurrence_sequence_eventually_periodic (u : ℕ → ℤ) 
  (h_recurrence : RecurrenceSequence u) (h_bounded : IsBounded u) : 
  EventuallyPeriodic u :=
sorry

end recurrence_sequence_eventually_periodic_l2845_284550


namespace lawrence_county_kids_l2845_284553

/-- The number of kids from Lawrence county going to camp -/
def kids_camp : ℕ := 610769

/-- The number of kids from Lawrence county staying home -/
def kids_home : ℕ := 590796

/-- The total number of kids in Lawrence county -/
def total_kids : ℕ := kids_camp + kids_home

/-- Theorem stating that the total number of kids in Lawrence county
    is equal to the sum of kids going to camp and kids staying home -/
theorem lawrence_county_kids : total_kids = 1201565 := by sorry

end lawrence_county_kids_l2845_284553


namespace oscar_review_questions_l2845_284517

/-- Calculates the total number of questions Professor Oscar must review. -/
def total_questions (questions_per_exam : ℕ) (num_classes : ℕ) (students_per_class : ℕ) : ℕ :=
  questions_per_exam * num_classes * students_per_class

/-- Proves that Professor Oscar must review 1750 questions in total. -/
theorem oscar_review_questions :
  total_questions 10 5 35 = 1750 := by
  sorry

end oscar_review_questions_l2845_284517


namespace circle_circumference_radius_increase_l2845_284516

/-- If the circumference of a circle increases by 0.628 cm, then its radius increases by 0.1 cm. -/
theorem circle_circumference_radius_increase : 
  ∀ (r : ℝ) (Δr : ℝ), 
  2 * Real.pi * Δr = 0.628 → Δr = 0.1 :=
by sorry

end circle_circumference_radius_increase_l2845_284516


namespace common_tangents_of_specific_circles_l2845_284544

/-- The number of common tangents to two intersecting circles -/
def num_common_tangents (c1_center : ℝ × ℝ) (c1_radius : ℝ) (c2_center : ℝ × ℝ) (c2_radius : ℝ) : ℕ :=
  sorry

/-- The theorem stating that the number of common tangents to the given circles is 2 -/
theorem common_tangents_of_specific_circles : 
  num_common_tangents (2, 1) 2 (-1, 2) 3 = 2 := by
  sorry

end common_tangents_of_specific_circles_l2845_284544


namespace joe_marshmallow_fraction_l2845_284532

theorem joe_marshmallow_fraction :
  let dad_marshmallows : ℕ := 21
  let joe_marshmallows : ℕ := 4 * dad_marshmallows
  let dad_roasted : ℕ := dad_marshmallows / 3
  let total_roasted : ℕ := 49
  let joe_roasted : ℕ := total_roasted - dad_roasted
  joe_roasted / joe_marshmallows = 1 / 2 := by
sorry

end joe_marshmallow_fraction_l2845_284532


namespace inequality_interval_length_l2845_284586

/-- Given an inequality a ≤ 3x + 4 ≤ b, if the length of the interval of solutions is 8, then b - a = 24 -/
theorem inequality_interval_length (a b : ℝ) : 
  (∃ (l : ℝ), l = 8 ∧ l = (b - 4) / 3 - (a - 4) / 3) → b - a = 24 :=
by sorry

end inequality_interval_length_l2845_284586


namespace steve_commute_time_l2845_284569

-- Define the parameters
def distance_to_work : ℝ := 35
def speed_back : ℝ := 17.5

-- Define the theorem
theorem steve_commute_time :
  let speed_to_work : ℝ := speed_back / 2
  let time_to_work : ℝ := distance_to_work / speed_to_work
  let time_from_work : ℝ := distance_to_work / speed_back
  let total_time : ℝ := time_to_work + time_from_work
  total_time = 6 := by sorry

end steve_commute_time_l2845_284569


namespace marys_next_birthday_age_l2845_284530

/-- Proves that Mary's age on her next birthday is 11 years, given the conditions of the problem. -/
theorem marys_next_birthday_age :
  ∀ (m s d : ℝ),
  m = 1.3 * s →  -- Mary is 30% older than Sally
  s = 0.75 * d →  -- Sally is 25% younger than Danielle
  m + s + d = 30 →  -- Sum of their ages is 30 years
  ⌈m⌉ = 11  -- Mary's age on her next birthday (ceiling of her current age)
  := by sorry

end marys_next_birthday_age_l2845_284530


namespace no_natural_number_divisible_by_100_l2845_284522

theorem no_natural_number_divisible_by_100 : ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by
  sorry

end no_natural_number_divisible_by_100_l2845_284522


namespace purely_imaginary_complex_number_l2845_284547

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ).re = 0 ∧ (Complex.I * (a - 1) : ℂ).im ≠ 0 → a = 2 := by
  sorry

end purely_imaginary_complex_number_l2845_284547


namespace expression_evaluation_l2845_284518

theorem expression_evaluation : (25 - (3010 - 260)) * (1500 - (100 - 25)) = -3885625 := by
  sorry

end expression_evaluation_l2845_284518


namespace rectangle_width_equals_square_side_l2845_284510

/-- The width of a rectangle with length 4 cm and area equal to a square with sides 4 cm is 4 cm. -/
theorem rectangle_width_equals_square_side {width : ℝ} (h : width > 0) : 
  4 * width = 4 * 4 → width = 4 := by
  sorry

#check rectangle_width_equals_square_side

end rectangle_width_equals_square_side_l2845_284510


namespace circle_problem_l2845_284573

noncomputable section

-- Define the line l: y = kx
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define circle C₁: (x-1)² + y² = 1
def circle_C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define point M
def point_M : ℝ × ℝ := (3, Real.sqrt 3)

-- Define the tangency condition for C₂ and l at M
def tangent_C₂_l (k : ℝ) : Prop := line_l k 3 (Real.sqrt 3)

-- Define the external tangency condition for C₁ and C₂
def external_tangent_C₁_C₂ (m n R : ℝ) : Prop :=
  (m - 1)^2 + n^2 = (1 + R)^2

-- Main theorem
theorem circle_problem (k : ℝ) :
  (∃ m n R, external_tangent_C₁_C₂ m n R ∧ tangent_C₂_l k) →
  (k = Real.sqrt 3 / 3) ∧
  (∃ A B : ℝ × ℝ, circle_C₁ A.1 A.2 ∧ circle_C₁ B.1 B.2 ∧ line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 3)) ∧
  (∃ m n : ℝ, ((m = 4 ∧ n = 0) ∨ (m = 0 ∧ n = 4 * Real.sqrt 3)) ∧
    (∀ x y : ℝ, (x - m)^2 + (y - n)^2 = (if m = 4 then 4 else 36))) :=
sorry

end

end circle_problem_l2845_284573


namespace circulation_ratio_l2845_284599

/-- Represents the circulation of a magazine over time -/
structure MagazineCirculation where
  /-- Circulation in 1962 -/
  C_1962 : ℝ
  /-- Growth rate per year (as a decimal) -/
  r : ℝ
  /-- Average yearly circulation from 1962 to 1970 -/
  A : ℝ

/-- Theorem stating the ratio of circulation in 1961 to total circulation 1961-1970 -/
theorem circulation_ratio (P : MagazineCirculation) :
  /- Circulation in 1961 is 4 times the average from 1962-1970 -/
  (4 * P.A) / 
  /- Total circulation from 1961-1970 -/
  (4 * P.A + 9 * P.A) = 4 / 13 := by
  sorry

end circulation_ratio_l2845_284599


namespace roof_collapse_leaves_l2845_284571

theorem roof_collapse_leaves (roof_capacity : ℕ) (leaves_per_pound : ℕ) (days_to_collapse : ℕ) :
  roof_capacity = 500 →
  leaves_per_pound = 1000 →
  days_to_collapse = 5000 →
  (roof_capacity * leaves_per_pound) / days_to_collapse = 100 :=
by sorry

end roof_collapse_leaves_l2845_284571


namespace power_result_l2845_284572

theorem power_result (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : a^(2*m - 3*n) = 9/8 := by
  sorry

end power_result_l2845_284572


namespace f_properties_l2845_284560

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - a / (3^x + 1)

theorem f_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 2 ∧ ∀ x y, x < y → f 2 x < f 2 y) :=
by sorry

end f_properties_l2845_284560


namespace volunteer_hours_theorem_l2845_284576

/-- Calculates the total hours volunteered per year given the frequency per month and hours per session -/
def total_volunteer_hours_per_year (sessions_per_month : ℕ) (hours_per_session : ℕ) : ℕ :=
  sessions_per_month * 12 * hours_per_session

/-- Proves that volunteering twice a month for 3 hours each time results in 72 hours per year -/
theorem volunteer_hours_theorem :
  total_volunteer_hours_per_year 2 3 = 72 := by
  sorry

end volunteer_hours_theorem_l2845_284576


namespace april_roses_unsold_l2845_284536

/-- The number of roses left unsold after a sale --/
def roses_left_unsold (initial_roses : ℕ) (price_per_rose : ℕ) (total_earned : ℕ) : ℕ :=
  initial_roses - (total_earned / price_per_rose)

/-- Theorem: Given the conditions of April's rose sale, prove that 4 roses were left unsold --/
theorem april_roses_unsold : roses_left_unsold 13 4 36 = 4 := by
  sorry

end april_roses_unsold_l2845_284536


namespace inscribed_triangle_area_l2845_284555

theorem inscribed_triangle_area (r : ℝ) (a b c : ℝ) (h_radius : r = 5) 
  (h_ratio : ∃ (k : ℝ), a = 4*k ∧ b = 5*k ∧ c = 6*k) 
  (h_inscribed : c = 2*r) : 
  (1/2 : ℝ) * a * b = 250/9 := by
sorry

end inscribed_triangle_area_l2845_284555


namespace logarithm_expression_evaluation_l2845_284504

theorem logarithm_expression_evaluation :
  (Real.log 50 / Real.log 4) / (Real.log 4 / Real.log 25) -
  (Real.log 100 / Real.log 4) / (Real.log 4 / Real.log 50) = -1/2 := by
  sorry

end logarithm_expression_evaluation_l2845_284504
