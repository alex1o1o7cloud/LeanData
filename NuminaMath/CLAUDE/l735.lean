import Mathlib

namespace NUMINAMATH_CALUDE_M_superset_P_l735_73537

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x^2 - 4}

-- Define the set P
def P : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}

-- Define the transformation function
def f (x : ℝ) : ℝ := x^2 - 4

-- Theorem statement
theorem M_superset_P : M ⊇ f '' P := by sorry

end NUMINAMATH_CALUDE_M_superset_P_l735_73537


namespace NUMINAMATH_CALUDE_final_season_premiere_l735_73551

/-- The number of days needed to watch all episodes of a TV series -/
def days_to_watch (seasons : ℕ) (episodes_per_season : ℕ) (episodes_per_day : ℕ) : ℕ :=
  (seasons * episodes_per_season) / episodes_per_day

/-- Proof that it takes 10 days to watch all episodes -/
theorem final_season_premiere :
  days_to_watch 4 15 6 = 10 := by
  sorry

#eval days_to_watch 4 15 6

end NUMINAMATH_CALUDE_final_season_premiere_l735_73551


namespace NUMINAMATH_CALUDE_laundry_time_difference_l735_73552

theorem laundry_time_difference : ∀ (clothes_time towels_time sheets_time : ℕ),
  clothes_time = 30 →
  towels_time = 2 * clothes_time →
  clothes_time + towels_time + sheets_time = 135 →
  towels_time - sheets_time = 15 := by
sorry

end NUMINAMATH_CALUDE_laundry_time_difference_l735_73552


namespace NUMINAMATH_CALUDE_constant_sum_property_l735_73508

/-- Represents a triangle with numbers assigned to its vertices -/
structure NumberedTriangle where
  x : ℝ  -- Number assigned to vertex A
  y : ℝ  -- Number assigned to vertex B
  z : ℝ  -- Number assigned to vertex C

/-- The sum of a vertex number and the opposite side sum is constant -/
theorem constant_sum_property (t : NumberedTriangle) :
  t.x + (t.y + t.z) = t.y + (t.z + t.x) ∧
  t.y + (t.z + t.x) = t.z + (t.x + t.y) ∧
  t.z + (t.x + t.y) = t.x + t.y + t.z :=
sorry

end NUMINAMATH_CALUDE_constant_sum_property_l735_73508


namespace NUMINAMATH_CALUDE_arrangement_count_l735_73553

/-- The number of ways to arrange young and elderly people in a line with specific conditions -/
def arrangements (n r : ℕ) : ℕ :=
  (n.factorial * (n - r).factorial) / (n - 2*r).factorial

/-- Theorem stating the number of arrangements for young and elderly people -/
theorem arrangement_count (n r : ℕ) (h : n > 2*r) :
  arrangements n r = (n.factorial * (n - r).factorial) / (n - 2*r).factorial :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l735_73553


namespace NUMINAMATH_CALUDE_candy_count_l735_73597

/-- The number of candies initially in the pile -/
def initial_candies : ℕ := 6

/-- The number of candies added to the pile -/
def added_candies : ℕ := 4

/-- The total number of candies after adding -/
def total_candies : ℕ := initial_candies + added_candies

theorem candy_count : total_candies = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l735_73597


namespace NUMINAMATH_CALUDE_binomial_square_exclusion_l735_73595

theorem binomial_square_exclusion (y : ℝ) :
  ∃ (a b : ℝ), (a + b)^2 = (a + b) * (a - b) ∧
  ∃ (x : ℝ), (-x + 1) * (-x - 1) = -(x + 1)^2 + 2 * (x + 1) ∧
  ∀ (k : ℝ), (y + 1) * (-y - 1) ≠ k * ((y + 1)^2) ∧
  ∃ (m : ℝ), (m - 1) * (-1 - m) = -(m + 1)^2 + 4 * m := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_exclusion_l735_73595


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_q_gt_one_l735_73590

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = q * a n)

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem geometric_sequence_increasing_iff_q_gt_one (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q → (IncreasingSequence a ↔ q > 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_q_gt_one_l735_73590


namespace NUMINAMATH_CALUDE_milk_distribution_l735_73580

/-- The milk distribution problem -/
theorem milk_distribution (container_a capacity_a quantity_b quantity_c : ℚ) : 
  capacity_a = 1264 →
  quantity_b + quantity_c = capacity_a →
  quantity_b + 158 = quantity_c - 158 →
  (capacity_a - (quantity_b + 158)) / capacity_a * 100 = 50 := by
  sorry

#check milk_distribution

end NUMINAMATH_CALUDE_milk_distribution_l735_73580


namespace NUMINAMATH_CALUDE_sum_of_roots_l735_73568

theorem sum_of_roots (k d : ℝ) (x₁ x₂ : ℝ) (h₁ : 4 * x₁^2 - k * x₁ = d)
    (h₂ : 4 * x₂^2 - k * x₂ = d) (h₃ : x₁ ≠ x₂) (h₄ : d ≠ 0) :
  x₁ + x₂ = k / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l735_73568


namespace NUMINAMATH_CALUDE_rectangular_triangle_condition_l735_73581

theorem rectangular_triangle_condition (A B C : Real) 
  (h : (Real.sin A)^2 + (Real.sin B)^2 + (Real.sin C)^2 = 
       2 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2)) 
  (triangle_angles : A + B + C = Real.pi) :
  A = Real.pi/2 ∨ B = Real.pi/2 ∨ C = Real.pi/2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_triangle_condition_l735_73581


namespace NUMINAMATH_CALUDE_hockey_league_games_l735_73506

theorem hockey_league_games (n : ℕ) (k : ℕ) (h1 : n = 18) (h2 : k = 10) :
  (n * (n - 1) / 2) * k = 1530 :=
by sorry

end NUMINAMATH_CALUDE_hockey_league_games_l735_73506


namespace NUMINAMATH_CALUDE_vasya_fool_count_l735_73540

/-- Represents the number of times a player was left as the "fool" -/
structure FoolCount where
  count : ℕ
  positive : count > 0

/-- The game "Fool" with four players -/
structure FoolGame where
  misha : FoolCount
  petya : FoolCount
  kolya : FoolCount
  vasya : FoolCount
  total_games : misha.count + petya.count + kolya.count + vasya.count = 16
  misha_most : misha.count > petya.count ∧ misha.count > kolya.count ∧ misha.count > vasya.count
  petya_kolya_sum : petya.count + kolya.count = 9

theorem vasya_fool_count (game : FoolGame) : game.vasya.count = 1 := by
  sorry

end NUMINAMATH_CALUDE_vasya_fool_count_l735_73540


namespace NUMINAMATH_CALUDE_number_equality_l735_73518

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (16/216) * (1/x)) : x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l735_73518


namespace NUMINAMATH_CALUDE_loss_ratio_is_one_third_l735_73577

/-- Represents a baseball team's season statistics -/
structure BaseballSeason where
  total_games : ℕ
  away_games : ℕ
  home_game_wins : ℕ
  away_game_wins : ℕ
  home_game_wins_extra_innings : ℕ

/-- Calculates the ratio of away game losses to home game losses not in extra innings -/
def loss_ratio (season : BaseballSeason) : Rat :=
  let home_games := season.total_games - season.away_games
  let home_game_losses := home_games - season.home_game_wins
  let away_game_losses := season.away_games - season.away_game_wins
  away_game_losses / home_game_losses

/-- Theorem stating the loss ratio for the given season is 1/3 -/
theorem loss_ratio_is_one_third (season : BaseballSeason) 
  (h1 : season.total_games = 45)
  (h2 : season.away_games = 15)
  (h3 : season.home_game_wins = 6)
  (h4 : season.away_game_wins = 7)
  (h5 : season.home_game_wins_extra_innings = 3) :
  loss_ratio season = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_loss_ratio_is_one_third_l735_73577


namespace NUMINAMATH_CALUDE_rectangle_fold_area_l735_73522

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (P Q R S : Point)

/-- Represents the folded configuration -/
structure FoldedConfig :=
  (rect : Rectangle)
  (T : Point)
  (U : Point)
  (Q' : Point)
  (R' : Point)

/-- Checks if a point is on a line segment -/
def isOnSegment (A B C : Point) : Prop :=
  sorry

/-- Calculates the angle between three points -/
def angle (A B C : Point) : ℝ :=
  sorry

/-- Calculates the distance between two points -/
def distance (A B : Point) : ℝ :=
  sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (rect : Rectangle) : ℝ :=
  sorry

theorem rectangle_fold_area (config : FoldedConfig) :
  isOnSegment config.rect.P config.rect.Q config.T →
  isOnSegment config.rect.R config.rect.S config.U →
  distance config.rect.Q config.T < distance config.rect.R config.U →
  isOnSegment config.rect.P config.rect.S config.R' →
  angle config.rect.P config.Q' config.R' = angle config.Q' config.T config.rect.P →
  distance config.rect.P config.Q' = 7 →
  distance config.rect.Q config.T = 27 →
  rectangleArea config.rect = 378 * Real.sqrt 21 + 243 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_fold_area_l735_73522


namespace NUMINAMATH_CALUDE_school_boys_count_l735_73567

theorem school_boys_count (muslim_percent : ℝ) (hindu_percent : ℝ) (sikh_percent : ℝ) (other_count : ℕ) : 
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  other_count = 126 →
  ∃ (total : ℕ), 
    (muslim_percent + hindu_percent + sikh_percent + (other_count : ℝ) / total = 1) ∧
    total = 700 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l735_73567


namespace NUMINAMATH_CALUDE_fractional_exponent_simplification_l735_73549

theorem fractional_exponent_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a ^ (2 * b ^ (1/4))) / (((a * (b ^ (1/2))) ^ (1/2))) = a ^ (1/2) := by
  sorry

end NUMINAMATH_CALUDE_fractional_exponent_simplification_l735_73549


namespace NUMINAMATH_CALUDE_third_basket_apples_l735_73513

/-- The number of apples originally in the third basket -/
def apples_in_third_basket : ℕ := 655

theorem third_basket_apples :
  ∀ (x y : ℕ),
  -- Total number of apples in all baskets
  (x + 2*y) + (x + 49) + (x + y) = 2014 →
  -- Number of apples left in first basket is twice the number left in third
  2*y = 2*(x + y - apples_in_third_basket) →
  -- The original number of apples in the third basket
  apples_in_third_basket = x + y :=
by
  sorry

#check third_basket_apples

end NUMINAMATH_CALUDE_third_basket_apples_l735_73513


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l735_73535

/-- Represents the recipe and Mary's baking progress -/
structure Recipe :=
  (total_sugar : ℕ)
  (flour_added : ℕ)
  (sugar_added : ℕ)
  (sugar_to_add : ℕ)

/-- The amount of flour required is independent of the amount of sugar -/
axiom flour_independent_of_sugar (r : Recipe) : 
  r.flour_added = r.flour_added

/-- Theorem: The recipe calls for 10 cups of flour -/
theorem recipe_flour_amount (r : Recipe) 
  (h1 : r.total_sugar = 14)
  (h2 : r.flour_added = 10)
  (h3 : r.sugar_added = 2)
  (h4 : r.sugar_to_add = 12) :
  r.flour_added = 10 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l735_73535


namespace NUMINAMATH_CALUDE_smallest_n_with_square_sums_l735_73515

theorem smallest_n_with_square_sums : ∃ (a b c : ℕ), 
  (a < b ∧ b < c) ∧ 
  (∃ (x y z : ℕ), a + b = x^2 ∧ a + c = y^2 ∧ b + c = z^2) ∧
  a + b + c = 55 ∧
  (∀ (n : ℕ), n < 55 → 
    ¬∃ (a' b' c' : ℕ), (a' < b' ∧ b' < c') ∧ 
    (∃ (x' y' z' : ℕ), a' + b' = x'^2 ∧ a' + c' = y'^2 ∧ b' + c' = z'^2) ∧
    a' + b' + c' = n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_square_sums_l735_73515


namespace NUMINAMATH_CALUDE_marble_count_l735_73575

theorem marble_count (r g b : ℕ) : 
  g + b = 6 →
  r + b = 8 →
  r + g = 4 →
  r + g + b = 9 := by sorry

end NUMINAMATH_CALUDE_marble_count_l735_73575


namespace NUMINAMATH_CALUDE_apples_bought_l735_73547

theorem apples_bought (initial : ℕ) (used : ℕ) (final : ℕ) (bought : ℕ) : 
  initial ≥ used →
  bought = final - (initial - used) := by
  sorry

end NUMINAMATH_CALUDE_apples_bought_l735_73547


namespace NUMINAMATH_CALUDE_gcd_n_squared_plus_four_n_plus_three_l735_73511

theorem gcd_n_squared_plus_four_n_plus_three (n : ℕ) (h : n > 4) :
  Nat.gcd (n^2 + 4) (n + 3) = if (n + 3) % 13 = 0 then 13 else 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_squared_plus_four_n_plus_three_l735_73511


namespace NUMINAMATH_CALUDE_angle_between_points_l735_73520

/-- The angle between two points on a spherical Earth given their coordinates -/
def angleOnSphere (latA longA latB longB : Real) : Real :=
  360 - longA - longB

/-- Point A's coordinates -/
def pointA : (Real × Real) := (0, 100)

/-- Point B's coordinates -/
def pointB : (Real × Real) := (45, -115)

theorem angle_between_points :
  angleOnSphere pointA.1 pointA.2 pointB.1 pointB.2 = 145 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_points_l735_73520


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l735_73544

theorem complex_magnitude_problem (z : ℂ) : 
  z = (Complex.I : ℂ) / (1 + Complex.I) → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l735_73544


namespace NUMINAMATH_CALUDE_online_store_sales_analysis_l735_73550

/-- Represents the daily sales volume as a function of selling price -/
def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 180

/-- Represents the daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 60) * (daily_sales_volume x)

/-- The original selling price -/
def original_price : ℝ := 80

/-- The cost price of each item -/
def cost_price : ℝ := 60

/-- The valid range for the selling price -/
def valid_price_range (x : ℝ) : Prop := 60 ≤ x ∧ x ≤ 80

theorem online_store_sales_analysis 
  (x : ℝ) 
  (h : valid_price_range x) :
  (daily_sales_volume x = -2 * x + 180) ∧
  (∃ x₁, daily_profit x₁ = 432 ∧ x₁ = 72) ∧
  (∃ x₂, ∀ y, valid_price_range y → daily_profit x₂ ≥ daily_profit y ∧ x₂ = 75) := by
  sorry

end NUMINAMATH_CALUDE_online_store_sales_analysis_l735_73550


namespace NUMINAMATH_CALUDE_clock_strike_time_l735_73523

/-- Given a clock that takes 6 seconds to strike 3 times, prove that it takes 33 seconds to strike 12 times. -/
theorem clock_strike_time (strike_time : ℕ → ℕ) (h1 : strike_time 3 = 6) :
  strike_time 12 = 33 := by
  sorry

end NUMINAMATH_CALUDE_clock_strike_time_l735_73523


namespace NUMINAMATH_CALUDE_calculate_principal_amount_l735_73539

/-- Given simple interest, time, and rate, calculate the principal amount -/
theorem calculate_principal_amount
  (simple_interest : ℝ)
  (time : ℝ)
  (rate : ℝ)
  (h1 : simple_interest = 180)
  (h2 : time = 2)
  (h3 : rate = 22.5) :
  simple_interest / (rate * time / 100) = 400 := by
  sorry

end NUMINAMATH_CALUDE_calculate_principal_amount_l735_73539


namespace NUMINAMATH_CALUDE_order_of_abc_l735_73562

theorem order_of_abc (a b c : ℝ) : 
  a = (Real.exp 1)⁻¹ → 
  b = (Real.log 3) / 3 → 
  c = (Real.log 4) / 4 → 
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l735_73562


namespace NUMINAMATH_CALUDE_annual_compound_interest_rate_exists_l735_73598

-- Define the initial principal
def initial_principal : ℝ := 780

-- Define the final amount
def final_amount : ℝ := 1300

-- Define the time period in years
def time_period : ℕ := 4

-- Define the compound interest equation
def compound_interest_equation (r : ℝ) : Prop :=
  final_amount = initial_principal * (1 + r) ^ time_period

-- Theorem statement
theorem annual_compound_interest_rate_exists :
  ∃ r : ℝ, compound_interest_equation r ∧ r > 0 ∧ r < 1 :=
sorry

end NUMINAMATH_CALUDE_annual_compound_interest_rate_exists_l735_73598


namespace NUMINAMATH_CALUDE_pencil_distribution_problem_l735_73561

/-- The number of ways to distribute pencils among friends -/
def distribute_pencils (total_pencils : ℕ) (num_friends : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of distributions for the given problem -/
theorem pencil_distribution_problem :
  distribute_pencils 10 4 = 58 :=
sorry

end NUMINAMATH_CALUDE_pencil_distribution_problem_l735_73561


namespace NUMINAMATH_CALUDE_cube_sum_formula_l735_73531

theorem cube_sum_formula (x y z c d : ℝ) 
  (h1 : x * y * z = c)
  (h2 : 1 / x^3 + 1 / y^3 + 1 / z^3 = d) :
  (x + y + z)^3 = d * c^3 + 3 * c - 3 * c * d :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_formula_l735_73531


namespace NUMINAMATH_CALUDE_max_number_bound_l735_73570

/-- Represents an arc on the circle with two natural numbers -/
structure Arc where
  a : ℕ
  b : ℕ

/-- Represents the circle with 1000 arcs -/
def Circle := Fin 1000 → Arc

/-- The condition that the sum of numbers on each arc is divisible by the product of numbers on the next arc -/
def valid_circle (c : Circle) : Prop :=
  ∀ i : Fin 1000, (c i).a + (c i).b ∣ (c (i + 1)).a * (c (i + 1)).b

/-- The theorem stating that the maximum number on any arc is at most 2001 -/
theorem max_number_bound (c : Circle) (h : valid_circle c) :
  ∀ i : Fin 1000, (c i).a ≤ 2001 ∧ (c i).b ≤ 2001 :=
sorry

end NUMINAMATH_CALUDE_max_number_bound_l735_73570


namespace NUMINAMATH_CALUDE_pascal_ratio_row_l735_73500

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Checks if three consecutive entries in a row have the ratio 2:3:4 -/
def has_ratio_2_3_4 (n : ℕ) : Prop :=
  ∃ r : ℕ, 
    (pascal n r : ℚ) / (pascal n (r + 1)) = 2 / 3 ∧
    (pascal n (r + 1) : ℚ) / (pascal n (r + 2)) = 3 / 4

theorem pascal_ratio_row : 
  ∃ n : ℕ, has_ratio_2_3_4 n ∧ ∀ m : ℕ, m < n → ¬has_ratio_2_3_4 m :=
by sorry

end NUMINAMATH_CALUDE_pascal_ratio_row_l735_73500


namespace NUMINAMATH_CALUDE_ratio_of_fractions_l735_73574

theorem ratio_of_fractions (x y : ℝ) (h1 : 5 * x = 6 * y) (h2 : x * y ≠ 0) :
  (1 / 3 * x) / (1 / 5 * y) = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_fractions_l735_73574


namespace NUMINAMATH_CALUDE_canoe_kayak_ratio_l735_73521

theorem canoe_kayak_ratio : 
  ∀ (C K : ℕ),
  (9 * C + 12 * K = 432) →  -- Total revenue
  (C = K + 6) →             -- 6 more canoes than kayaks
  (∃ (n : ℕ), C = 3 * n * K) →  -- Canoes are a multiple of 3 times kayaks
  (C : ℚ) / K = 4 / 3 :=    -- Ratio of canoes to kayaks is 4:3
by
  sorry

#check canoe_kayak_ratio

end NUMINAMATH_CALUDE_canoe_kayak_ratio_l735_73521


namespace NUMINAMATH_CALUDE_existence_of_point_l735_73534

theorem existence_of_point (f : ℝ → ℝ) (h_pos : ∀ x, f x > 0) (h_nondec : ∀ x y, x ≤ y → f x ≤ f y) :
  ∃ a : ℝ, f (a + 1 / f a) < 2 * f a := by
  sorry

end NUMINAMATH_CALUDE_existence_of_point_l735_73534


namespace NUMINAMATH_CALUDE_population_increase_rate_is_10_percent_l735_73585

/-- The population increase rate given initial and final populations -/
def population_increase_rate (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Theorem: The population increase rate is 10% given the conditions -/
theorem population_increase_rate_is_10_percent :
  let initial_population := 260
  let final_population := 286
  population_increase_rate initial_population final_population = 10 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_is_10_percent_l735_73585


namespace NUMINAMATH_CALUDE_max_inradii_difference_l735_73554

noncomputable def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

def Q₁ (P : ℝ × ℝ) : ℝ × ℝ := sorry
def Q₂ (P : ℝ × ℝ) : ℝ × ℝ := sorry

def r₁ (P : ℝ × ℝ) : ℝ := sorry
def r₂ (P : ℝ × ℝ) : ℝ := sorry

theorem max_inradii_difference :
  ∃ (max : ℝ), max = 1/3 ∧
  ∀ (P : ℝ × ℝ), on_ellipse P → first_quadrant P.1 P.2 →
  r₁ P - r₂ P ≤ max ∧
  ∃ (P' : ℝ × ℝ), on_ellipse P' ∧ first_quadrant P'.1 P'.2 ∧ r₁ P' - r₂ P' = max :=
sorry

end NUMINAMATH_CALUDE_max_inradii_difference_l735_73554


namespace NUMINAMATH_CALUDE_father_reaches_mom_age_in_three_years_l735_73599

/-- Represents the ages and time in the problem -/
structure AgesProblem where
  talia_future_age : ℕ      -- Talia's age in 7 years
  talia_future_years : ℕ    -- Years until Talia reaches future_age
  father_current_age : ℕ    -- Talia's father's current age
  mom_age_ratio : ℕ         -- Ratio of mom's age to Talia's current age

/-- Calculates the years until Talia's father reaches Talia's mom's current age -/
def years_until_father_reaches_mom_age (p : AgesProblem) : ℕ :=
  let talia_current_age := p.talia_future_age - p.talia_future_years
  let mom_current_age := talia_current_age * p.mom_age_ratio
  mom_current_age - p.father_current_age

/-- Theorem stating the solution to the problem -/
theorem father_reaches_mom_age_in_three_years (p : AgesProblem) 
    (h1 : p.talia_future_age = 20)
    (h2 : p.talia_future_years = 7)
    (h3 : p.father_current_age = 36)
    (h4 : p.mom_age_ratio = 3) :
  years_until_father_reaches_mom_age p = 3 := by
  sorry


end NUMINAMATH_CALUDE_father_reaches_mom_age_in_three_years_l735_73599


namespace NUMINAMATH_CALUDE_residue_of_7_2050_mod_19_l735_73583

theorem residue_of_7_2050_mod_19 : 7^2050 % 19 = 11 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_7_2050_mod_19_l735_73583


namespace NUMINAMATH_CALUDE_lcm_18_20_l735_73565

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_20_l735_73565


namespace NUMINAMATH_CALUDE_remaining_files_l735_73538

theorem remaining_files (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 13)
  (h2 : video_files = 30)
  (h3 : deleted_files = 10) :
  music_files + video_files - deleted_files = 33 := by
  sorry

end NUMINAMATH_CALUDE_remaining_files_l735_73538


namespace NUMINAMATH_CALUDE_continuity_reciprocal_quadratic_plus_four_l735_73501

theorem continuity_reciprocal_quadratic_plus_four (x : ℝ) :
  Continuous (fun x => 1 / (x^2 + 4)) :=
sorry

end NUMINAMATH_CALUDE_continuity_reciprocal_quadratic_plus_four_l735_73501


namespace NUMINAMATH_CALUDE_toy_store_revenue_l735_73569

theorem toy_store_revenue (D : ℝ) (h1 : D > 0) : 
  let nov := (2 / 5 : ℝ) * D
  let jan := (1 / 2 : ℝ) * nov
  let avg := (nov + jan) / 2
  D / avg = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_toy_store_revenue_l735_73569


namespace NUMINAMATH_CALUDE_lowest_price_theorem_l735_73514

/-- Calculates the lowest price per component to break even --/
def lowest_price_per_component (production_cost shipping_cost : ℚ) 
  (fixed_costs : ℚ) (num_components : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / num_components)

theorem lowest_price_theorem (production_cost shipping_cost : ℚ) 
  (fixed_costs : ℚ) (num_components : ℕ) :
  lowest_price_per_component production_cost shipping_cost fixed_costs num_components =
  (production_cost * num_components + shipping_cost * num_components + fixed_costs) / num_components :=
by sorry

#eval lowest_price_per_component 80 2 16200 150

end NUMINAMATH_CALUDE_lowest_price_theorem_l735_73514


namespace NUMINAMATH_CALUDE_negative_sum_distribution_l735_73589

theorem negative_sum_distribution (x y : ℝ) : -(x + y) = -x - y := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_distribution_l735_73589


namespace NUMINAMATH_CALUDE_strawberry_pies_l735_73555

/-- The number of pies that can be made from strawberries picked by Christine and Rachel -/
def number_of_pies (christine_picked : ℕ) (rachel_factor : ℕ) (pounds_per_pie : ℕ) : ℕ :=
  (christine_picked + christine_picked * rachel_factor) / pounds_per_pie

/-- Theorem stating that Christine and Rachel can make 10 pies -/
theorem strawberry_pies :
  number_of_pies 10 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_pies_l735_73555


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l735_73505

-- Define the hourly rates and total hours
def ordinary_rate : ℚ := 60 / 100
def overtime_rate : ℚ := 90 / 100
def total_hours : ℕ := 50

-- Define the total pay in dollars
def total_pay : ℚ := 3240 / 100

-- Theorem statement
theorem overtime_hours_calculation :
  ∃ (ordinary_hours overtime_hours : ℕ),
    ordinary_hours + overtime_hours = total_hours ∧
    ordinary_rate * ordinary_hours + overtime_rate * overtime_hours = total_pay ∧
    overtime_hours = 8 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l735_73505


namespace NUMINAMATH_CALUDE_value_of_expression_l735_73564

theorem value_of_expression (x y : ℝ) 
  (h1 : x - 2*y = -5) 
  (h2 : x*y = -2) : 
  2*x^2*y - 4*x*y^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l735_73564


namespace NUMINAMATH_CALUDE_unique_factorization_l735_73546

/-- The set of all positive integers that cannot be written as a sum of an arithmetic progression with difference d, having at least two terms and consisting of positive integers. -/
def M (d : ℕ) : Set ℕ :=
  {n : ℕ | ∀ (a k : ℕ), k ≥ 2 → n ≠ (k * (2 * a + (k - 1) * d)) / 2}

/-- A is the set M₁ -/
def A : Set ℕ := M 1

/-- B is the set M₂ without the element 2 -/
def B : Set ℕ := M 2 \ {2}

/-- C is the set M₃ -/
def C : Set ℕ := M 3

/-- Every element in C can be uniquely expressed as a product of an element from A and an element from B -/
theorem unique_factorization (c : ℕ) (hc : c ∈ C) :
  ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ c = a * b :=
sorry

end NUMINAMATH_CALUDE_unique_factorization_l735_73546


namespace NUMINAMATH_CALUDE_distance_roja_pooja_distance_sooraj_pole_angle_roja_pooja_l735_73576

-- Define the speeds and time
def roja_speed : ℝ := 5
def pooja_speed : ℝ := 3
def sooraj_speed : ℝ := 4
def time : ℝ := 4

-- Define the distances traveled
def roja_distance : ℝ := roja_speed * time
def pooja_distance : ℝ := pooja_speed * time
def sooraj_distance : ℝ := sooraj_speed * time

-- Theorem for the distance between Roja and Pooja
theorem distance_roja_pooja : 
  Real.sqrt (roja_distance ^ 2 + pooja_distance ^ 2) = Real.sqrt 544 :=
sorry

-- Theorem for the distance between Sooraj and the pole
theorem distance_sooraj_pole : sooraj_distance = 16 :=
sorry

-- Theorem for the angle between Roja and Pooja's directions
theorem angle_roja_pooja : ∃ (angle : ℝ), angle = 90 :=
sorry

end NUMINAMATH_CALUDE_distance_roja_pooja_distance_sooraj_pole_angle_roja_pooja_l735_73576


namespace NUMINAMATH_CALUDE_checkerboard_probability_l735_73519

/-- The size of one side of the square checkerboard -/
def boardSize : ℕ := 10

/-- The total number of squares on the checkerboard -/
def totalSquares : ℕ := boardSize * boardSize

/-- The number of squares on the perimeter of the checkerboard -/
def perimeterSquares : ℕ := 4 * boardSize - 4

/-- The number of squares not touching the outer edge -/
def innerSquares : ℕ := totalSquares - perimeterSquares

/-- The probability of choosing a square not touching the outer edge -/
def innerProbability : ℚ := innerSquares / totalSquares

theorem checkerboard_probability :
  innerProbability = 16 / 25 := by sorry

end NUMINAMATH_CALUDE_checkerboard_probability_l735_73519


namespace NUMINAMATH_CALUDE_joan_initial_money_l735_73517

/-- The amount of money Joan had initially -/
def initial_money : ℕ := 60

/-- The cost of one container of hummus -/
def hummus_cost : ℕ := 5

/-- The number of hummus containers Joan buys -/
def hummus_quantity : ℕ := 2

/-- The cost of chicken -/
def chicken_cost : ℕ := 20

/-- The cost of bacon -/
def bacon_cost : ℕ := 10

/-- The cost of vegetables -/
def vegetable_cost : ℕ := 10

/-- The cost of one apple -/
def apple_cost : ℕ := 2

/-- The number of apples Joan can buy with remaining money -/
def apple_quantity : ℕ := 5

theorem joan_initial_money :
  initial_money = 
    hummus_cost * hummus_quantity + 
    chicken_cost + 
    bacon_cost + 
    vegetable_cost + 
    apple_cost * apple_quantity := by
  sorry

end NUMINAMATH_CALUDE_joan_initial_money_l735_73517


namespace NUMINAMATH_CALUDE_boxes_left_is_three_l735_73566

/-- The number of boxes of apples Merry had on Saturday -/
def saturday_boxes : ℕ := 50

/-- The number of boxes of apples Merry had on Sunday -/
def sunday_boxes : ℕ := 25

/-- The number of apples in each box -/
def apples_per_box : ℕ := 10

/-- The total number of apples Merry sold on Saturday and Sunday -/
def apples_sold : ℕ := 720

/-- Calculate the number of boxes of apples left -/
def boxes_left : ℕ := 
  (saturday_boxes * apples_per_box + sunday_boxes * apples_per_box - apples_sold) / apples_per_box

/-- Theorem stating that the number of boxes left is 3 -/
theorem boxes_left_is_three : boxes_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_boxes_left_is_three_l735_73566


namespace NUMINAMATH_CALUDE_number_of_zeros_l735_73558

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 2 else Real.log x - 1

-- Define what it means for x to be a zero of f
def isZero (x : ℝ) : Prop := f x = 0

-- State the theorem
theorem number_of_zeros : ∃ (a b : ℝ), a ≠ b ∧ isZero a ∧ isZero b ∧ ∀ c, isZero c → c = a ∨ c = b := by
  sorry

end NUMINAMATH_CALUDE_number_of_zeros_l735_73558


namespace NUMINAMATH_CALUDE_order_of_fractions_with_exponents_l735_73542

theorem order_of_fractions_with_exponents :
  (1/5 : ℝ)^(2/3) < (1/2 : ℝ)^(2/3) ∧ (1/2 : ℝ)^(2/3) < (1/2 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_order_of_fractions_with_exponents_l735_73542


namespace NUMINAMATH_CALUDE_flight_cost_proof_l735_73528

theorem flight_cost_proof (initial_cost : ℝ) : 
  (∃ (cost_per_person_4 cost_per_person_5 : ℝ),
    cost_per_person_4 = initial_cost / 4 ∧
    cost_per_person_5 = initial_cost / 5 ∧
    cost_per_person_4 - cost_per_person_5 = 30) →
  initial_cost = 600 := by
sorry

end NUMINAMATH_CALUDE_flight_cost_proof_l735_73528


namespace NUMINAMATH_CALUDE_binomial_coefficient_25_7_l735_73525

theorem binomial_coefficient_25_7 
  (h1 : Nat.choose 23 5 = 33649)
  (h2 : Nat.choose 23 6 = 42504)
  (h3 : Nat.choose 23 7 = 33649) : 
  Nat.choose 25 7 = 152306 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_25_7_l735_73525


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_101_l735_73516

theorem gcd_of_powers_of_101 (h : Prime 101) :
  Nat.gcd (101^5 + 1) (101^5 + 101^3 + 101 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_101_l735_73516


namespace NUMINAMATH_CALUDE_fifth_term_equals_eight_l735_73502

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = 2ⁿ⁻¹, prove that a₅ = 8 -/
theorem fifth_term_equals_eight (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = 2^(n - 1)) : a 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_equals_eight_l735_73502


namespace NUMINAMATH_CALUDE_total_turnips_count_l735_73582

/-- The number of turnips grown by Sally -/
def sally_turnips : ℕ := 113

/-- The number of turnips grown by Mary -/
def mary_turnips : ℕ := 129

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := sally_turnips + mary_turnips

theorem total_turnips_count : total_turnips = 242 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_count_l735_73582


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l735_73532

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, x^3 - 6*x^2 + 9*x - 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l735_73532


namespace NUMINAMATH_CALUDE_ring_worth_proof_l735_73504

theorem ring_worth_proof (total_worth car_cost : ℕ) (h1 : total_worth = 14000) (h2 : car_cost = 2000) :
  ∃ (ring_cost : ℕ), 
    ring_cost + car_cost + 2 * ring_cost = total_worth ∧ 
    ring_cost = 4000 := by
  sorry

end NUMINAMATH_CALUDE_ring_worth_proof_l735_73504


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l735_73573

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l735_73573


namespace NUMINAMATH_CALUDE_product_inequality_l735_73572

theorem product_inequality (a b c d : ℝ) 
  (sum_zero : a + b + c = 0)
  (d_def : d = max (abs a) (max (abs b) (abs c))) :
  abs ((1 + a) * (1 + b) * (1 + c)) ≥ 1 - d^2 :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l735_73572


namespace NUMINAMATH_CALUDE_chess_game_probability_l735_73530

/-- The probability of player A winning a chess game -/
def prob_A_win : ℝ := 0.3

/-- The probability of the chess game ending in a draw -/
def prob_draw : ℝ := 0.5

/-- The probability of player B not losing the chess game -/
def prob_B_not_lose : ℝ := 1 - prob_A_win

theorem chess_game_probability : prob_B_not_lose = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l735_73530


namespace NUMINAMATH_CALUDE_vector_problem_l735_73548

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m - 2)

def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem vector_problem :
  (∀ m : ℝ, parallel (vector_a m) (vector_b m) → m = 3 ∨ m = -1) ∧
  (∀ m : ℝ, perpendicular (vector_a m) (vector_b m) →
    let a := vector_a m
    let b := vector_b m
    dot_product (a.1 + 2 * b.1, a.2 + 2 * b.2) (2 * a.1 - b.1, 2 * a.2 - b.2) = 20) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l735_73548


namespace NUMINAMATH_CALUDE_absent_men_calculation_l735_73536

/-- Represents the number of men who became absent -/
def absentMen (totalMen originalDays actualDays : ℕ) : ℕ :=
  totalMen - (totalMen * originalDays) / actualDays

theorem absent_men_calculation (totalMen originalDays actualDays : ℕ) 
  (h1 : totalMen = 15)
  (h2 : originalDays = 8)
  (h3 : actualDays = 10)
  (h4 : totalMen > 0)
  (h5 : originalDays > 0)
  (h6 : actualDays > 0)
  (h7 : (totalMen * originalDays) % actualDays = 0) :
  absentMen totalMen originalDays actualDays = 3 := by
  sorry

#eval absentMen 15 8 10

end NUMINAMATH_CALUDE_absent_men_calculation_l735_73536


namespace NUMINAMATH_CALUDE_box_side_length_l735_73559

/-- The length of one side of a cubic box given total volume, cost per box, and total cost -/
theorem box_side_length 
  (total_volume : ℝ) 
  (cost_per_box : ℝ) 
  (total_cost : ℝ) 
  (h1 : total_volume = 2.16e6)  -- 2.16 million cubic inches
  (h2 : cost_per_box = 0.5)     -- $0.50 per box
  (h3 : total_cost = 225)       -- $225 total cost
  : ∃ (side_length : ℝ), abs (side_length - 16.89) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_box_side_length_l735_73559


namespace NUMINAMATH_CALUDE_hyperbola_equation_l735_73541

-- Define the hyperbola C
structure Hyperbola where
  -- The equation of the hyperbola in the form ax² + by² = c
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions of the problem
def hyperbola_conditions (C : Hyperbola) : Prop :=
  -- Center at origin (implied by the standard form)
  -- Asymptote y = √2x
  C.a / C.b = -2 ∧
  -- Point P(2√2, -√2) lies on C
  C.a * (2 * Real.sqrt 2)^2 + C.b * (-Real.sqrt 2)^2 = C.c

-- The theorem to prove
theorem hyperbola_equation (C : Hyperbola) :
  hyperbola_conditions C →
  C.a = 1/7 ∧ C.b = -1/14 ∧ C.c = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l735_73541


namespace NUMINAMATH_CALUDE_salesperson_earnings_theorem_l735_73579

/-- Represents the earnings of a salesperson based on their sales. -/
structure SalespersonEarnings where
  sales : ℕ
  earnings : ℝ

/-- Represents the direct proportionality between sales and earnings. -/
def directlyProportional (e1 e2 : SalespersonEarnings) : Prop :=
  e1.sales * e2.earnings = e2.sales * e1.earnings

/-- Theorem: If earnings are directly proportional to sales, and a salesperson
    earns $180 for 15 sales, then they will earn $240 for 20 sales. -/
theorem salesperson_earnings_theorem
  (e1 e2 : SalespersonEarnings)
  (h1 : directlyProportional e1 e2)
  (h2 : e1.sales = 15)
  (h3 : e1.earnings = 180)
  (h4 : e2.sales = 20) :
  e2.earnings = 240 := by
  sorry


end NUMINAMATH_CALUDE_salesperson_earnings_theorem_l735_73579


namespace NUMINAMATH_CALUDE_sandy_sums_theorem_l735_73524

theorem sandy_sums_theorem (correct_marks : ℕ → ℕ) (incorrect_marks : ℕ → ℕ) 
  (total_marks : ℕ) (correct_sums : ℕ) :
  (correct_marks = λ n => 3 * n) →
  (incorrect_marks = λ n => 2 * n) →
  (total_marks = 50) →
  (correct_sums = 22) →
  (∃ (total_sums : ℕ), 
    total_sums = correct_sums + (total_sums - correct_sums) ∧
    total_marks = correct_marks correct_sums - incorrect_marks (total_sums - correct_sums) ∧
    total_sums = 30) :=
by sorry

end NUMINAMATH_CALUDE_sandy_sums_theorem_l735_73524


namespace NUMINAMATH_CALUDE_interior_angle_measure_l735_73509

/-- Given a triangle with an interior angle, if the measures of the three triangle angles are known,
    then the measure of the interior angle can be determined. -/
theorem interior_angle_measure (m1 m2 m3 m4 : ℝ) : 
  m1 = 62 → m2 = 36 → m3 = 24 → 
  m1 + m2 + m3 + m4 < 360 →
  m4 = 122 := by
  sorry

#check interior_angle_measure

end NUMINAMATH_CALUDE_interior_angle_measure_l735_73509


namespace NUMINAMATH_CALUDE_garden_length_difference_l735_73527

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ

/-- Properties of the rectangular garden -/
def GardenProperties (garden : RectangularGarden) : Prop :=
  garden.length > 3 * garden.width ∧
  2 * (garden.length + garden.width) = 100 ∧
  garden.length = 38

theorem garden_length_difference (garden : RectangularGarden) 
  (h : GardenProperties garden) : 
  garden.length - 3 * garden.width = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_difference_l735_73527


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l735_73594

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (a_val b_val : ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a5 : a 5 = a_val)
  (h_a10 : a 10 = b_val) :
  a 15 = 2 * b_val - a_val :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l735_73594


namespace NUMINAMATH_CALUDE_count_green_curlers_l735_73545

/-- Given a total number of curlers and relationships between different types,
    prove the number of large green curlers. -/
theorem count_green_curlers (total : ℕ) (pink : ℕ) (blue : ℕ) (green : ℕ)
  (h1 : total = 16)
  (h2 : pink = total / 4)
  (h3 : blue = 2 * pink)
  (h4 : green = total - pink - blue) :
  green = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_green_curlers_l735_73545


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l735_73507

theorem arithmetic_calculation : (8 * 4) + 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l735_73507


namespace NUMINAMATH_CALUDE_fraction_equality_l735_73533

theorem fraction_equality (a b : ℝ) (h : a / b = 3 / 5) : (2 * a + 3 * b) / a = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l735_73533


namespace NUMINAMATH_CALUDE_binomial_20_18_l735_73596

theorem binomial_20_18 : Nat.choose 20 18 = 190 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_18_l735_73596


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l735_73578

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  t : ℝ
  h1 : 0 < d
  h2 : a 1 = 1
  h3 : ∀ n, 2 * (a n * a (n + 1) + 1) = t * n * (1 + a n)
  h4 : ∀ n, a (n + 1) = a n + d

/-- The general term of the arithmetic sequence is 2n - 1 -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, n > 0 → seq.a n = 2 * n - 1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l735_73578


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l735_73591

theorem smallest_positive_integer_satisfying_congruences : 
  ∃ (x : ℕ), x > 0 ∧ 
  (42 * x + 14) % 26 = 4 ∧ 
  x % 5 = 3 ∧
  (∀ (y : ℕ), y > 0 ∧ (42 * y + 14) % 26 = 4 ∧ y % 5 = 3 → x ≤ y) ∧
  x = 38 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_congruences_l735_73591


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l735_73510

theorem least_positive_integer_with_remainders (M : ℕ) : 
  (M % 11 = 10 ∧ M % 12 = 11 ∧ M % 13 = 12 ∧ M % 14 = 13) → 
  (∀ n : ℕ, n > 0 ∧ n % 11 = 10 ∧ n % 12 = 11 ∧ n % 13 = 12 ∧ n % 14 = 13 → M ≤ n) → 
  M = 30029 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l735_73510


namespace NUMINAMATH_CALUDE_saltwater_concentration_l735_73512

/-- Represents a saltwater solution -/
structure SaltWaterSolution where
  salt : ℝ
  water : ℝ
  concentration : ℝ
  concentration_def : concentration = salt / (salt + water) * 100

/-- The condition that adding 200g of water halves the concentration -/
def half_concentration (s : SaltWaterSolution) : Prop :=
  s.salt / (s.salt + s.water + 200) = s.concentration / 2

/-- The condition that adding 25g of salt doubles the concentration -/
def double_concentration (s : SaltWaterSolution) : Prop :=
  (s.salt + 25) / (s.salt + s.water + 25) = 2 * s.concentration / 100

/-- The main theorem to prove -/
theorem saltwater_concentration 
  (s : SaltWaterSolution) 
  (h1 : half_concentration s) 
  (h2 : double_concentration s) : 
  s.concentration = 10 := by
  sorry


end NUMINAMATH_CALUDE_saltwater_concentration_l735_73512


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l735_73584

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  3230000 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.23 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l735_73584


namespace NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_33_l735_73503

theorem remainder_11_pow_2023_mod_33 : 11^2023 % 33 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_33_l735_73503


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l735_73556

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 1 ∧ x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l735_73556


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l735_73557

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 17 + (2 * x) + 15 + (2 * x + 6) + (3 * x - 5)) / 6 = 30 → x = 137 / 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l735_73557


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l735_73587

theorem complex_modulus_problem (z : ℂ) : z = (Complex.I : ℂ) / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l735_73587


namespace NUMINAMATH_CALUDE_maggie_earnings_l735_73592

def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_next_door : ℕ := 2
def subscriptions_to_another : ℕ := 4

def base_pay : ℚ := 5
def family_bonus : ℚ := 2
def neighbor_bonus : ℚ := 1
def additional_bonus_base : ℚ := 10
def additional_bonus_per_extra : ℚ := 0.5

def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_next_door + subscriptions_to_another

def family_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather
def neighbor_subscriptions : ℕ := subscriptions_to_next_door + subscriptions_to_another

def base_earnings : ℚ := base_pay * total_subscriptions
def family_bonus_earnings : ℚ := family_bonus * family_subscriptions
def neighbor_bonus_earnings : ℚ := neighbor_bonus * neighbor_subscriptions

def additional_bonus : ℚ :=
  if total_subscriptions > 10
  then additional_bonus_base + additional_bonus_per_extra * (total_subscriptions - 10)
  else 0

def total_earnings : ℚ := base_earnings + family_bonus_earnings + neighbor_bonus_earnings + additional_bonus

theorem maggie_earnings : total_earnings = 81.5 := by sorry

end NUMINAMATH_CALUDE_maggie_earnings_l735_73592


namespace NUMINAMATH_CALUDE_decagon_division_impossible_l735_73529

/-- Represents a division of a polygon into colored triangles -/
structure TriangleDivision (n : ℕ) where
  black_sides : ℕ
  white_sides : ℕ
  valid_division : black_sides - white_sides = n

/-- Checks if a number is divisible by 3 -/
def divisible_by_three (n : ℕ) : Prop := ∃ k, n = 3 * k

theorem decagon_division_impossible :
  ¬ ∃ (d : TriangleDivision 10),
    divisible_by_three d.black_sides ∧
    divisible_by_three d.white_sides :=
sorry

end NUMINAMATH_CALUDE_decagon_division_impossible_l735_73529


namespace NUMINAMATH_CALUDE_polynomial_non_negative_l735_73593

theorem polynomial_non_negative (p q : ℝ) (h : q > p^2) :
  ∀ x : ℝ, x^2 + 2*p*x + q ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_non_negative_l735_73593


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l735_73588

theorem abs_sum_minimum (x : ℝ) : 
  |x + 3| + |x + 5| + |x + 6| + |x + 7| ≥ 5 ∧ 
  ∃ y : ℝ, |y + 3| + |y + 5| + |y + 6| + |y + 7| = 5 := by
  sorry

#check abs_sum_minimum

end NUMINAMATH_CALUDE_abs_sum_minimum_l735_73588


namespace NUMINAMATH_CALUDE_correct_answers_is_120_l735_73563

/-- Represents an exam scoring system -/
structure ExamScoring where
  totalScore : Int
  totalQuestions : Nat
  correctScore : Int
  wrongPenalty : Int

/-- Calculates the number of correct answers in an exam -/
def calculateCorrectAnswers (exam : ExamScoring) : Int :=
  (exam.totalScore + 2 * exam.totalQuestions) / (exam.correctScore - exam.wrongPenalty)

/-- Theorem: Given the exam conditions, the number of correct answers is 120 -/
theorem correct_answers_is_120 (exam : ExamScoring) 
  (h1 : exam.totalScore = 420)
  (h2 : exam.totalQuestions = 150)
  (h3 : exam.correctScore = 4)
  (h4 : exam.wrongPenalty = 2) :
  calculateCorrectAnswers exam = 120 := by
  sorry

#eval calculateCorrectAnswers { totalScore := 420, totalQuestions := 150, correctScore := 4, wrongPenalty := 2 }

end NUMINAMATH_CALUDE_correct_answers_is_120_l735_73563


namespace NUMINAMATH_CALUDE_swim_meet_transport_theorem_l735_73571

/-- Represents the transportation setup for the swimming club's trip --/
structure SwimMeetTransport where
  num_cars : Nat
  num_vans : Nat
  people_per_car : Nat
  max_people_per_car : Nat
  max_people_per_van : Nat
  additional_capacity : Nat

/-- Calculates the number of people in each van --/
def people_per_van (t : SwimMeetTransport) : Nat :=
  let total_capacity := t.num_cars * t.max_people_per_car + t.num_vans * t.max_people_per_van
  let actual_people := total_capacity - t.additional_capacity
  let people_in_cars := t.num_cars * t.people_per_car
  let people_in_vans := actual_people - people_in_cars
  people_in_vans / t.num_vans

/-- Theorem stating that the number of people in each van is 3 --/
theorem swim_meet_transport_theorem (t : SwimMeetTransport) 
  (h1 : t.num_cars = 2)
  (h2 : t.num_vans = 3)
  (h3 : t.people_per_car = 5)
  (h4 : t.max_people_per_car = 6)
  (h5 : t.max_people_per_van = 8)
  (h6 : t.additional_capacity = 17) :
  people_per_van t = 3 := by
  sorry

#eval people_per_van { 
  num_cars := 2, 
  num_vans := 3, 
  people_per_car := 5, 
  max_people_per_car := 6, 
  max_people_per_van := 8, 
  additional_capacity := 17 
}

end NUMINAMATH_CALUDE_swim_meet_transport_theorem_l735_73571


namespace NUMINAMATH_CALUDE_quadratic_problem_l735_73560

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_problem (a b c : ℝ) (f : ℝ → ℝ) (h_f : f = QuadraticFunction a b c) :
  (∀ x, f x ≤ 4) ∧ -- The maximum value of f(x) is 4
  (f 2 = 4) ∧ -- The maximum occurs at x = 2
  (f 0 = -20) ∧ -- The graph passes through (0, -20)
  (∃ m, f 5 = m) -- The graph passes through (5, m)
  → f 5 = -50 := by sorry

end NUMINAMATH_CALUDE_quadratic_problem_l735_73560


namespace NUMINAMATH_CALUDE_tax_rate_percentage_l735_73526

/-- Given a tax rate of $82 per $100.00, prove that it is equivalent to 82% -/
theorem tax_rate_percentage : 
  let tax_amount : ℚ := 82
  let base_amount : ℚ := 100
  (tax_amount / base_amount) * 100 = 82 := by sorry

end NUMINAMATH_CALUDE_tax_rate_percentage_l735_73526


namespace NUMINAMATH_CALUDE_paula_shopping_remaining_l735_73543

/-- Given an initial amount, cost of shirts, number of shirts, and cost of pants,
    calculate the remaining amount after purchases. -/
def remaining_amount (initial : ℕ) (shirt_cost : ℕ) (num_shirts : ℕ) (pants_cost : ℕ) : ℕ :=
  initial - (shirt_cost * num_shirts + pants_cost)

/-- Theorem stating that given the specific values from the problem,
    the remaining amount is 74. -/
theorem paula_shopping_remaining : remaining_amount 109 11 2 13 = 74 := by
  sorry

end NUMINAMATH_CALUDE_paula_shopping_remaining_l735_73543


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_m_part2_l735_73586

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |x - 2| - m

-- Part 1: Solution set for f(x) > 0 when m = 5
theorem solution_set_part1 : 
  {x : ℝ | f x 5 > 0} = {x : ℝ | x < -2 ∨ x > 3} := by sorry

-- Part 2: Range of m for which f(x) ≥ 2 has solution set ℝ
theorem range_of_m_part2 : 
  {m : ℝ | ∀ x, f x m ≥ 2} = {m : ℝ | m ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_m_part2_l735_73586
