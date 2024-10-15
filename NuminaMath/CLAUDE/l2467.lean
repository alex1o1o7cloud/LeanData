import Mathlib

namespace NUMINAMATH_CALUDE_min_value_quadratic_l2467_246796

theorem min_value_quadratic (x y : ℝ) :
  x^2 + y^2 - 10*x + 6*y + 25 ≥ -9 ∧
  ∃ (a b : ℝ), a^2 + b^2 - 10*a + 6*b + 25 = -9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2467_246796


namespace NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l2467_246754

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x^4 + 1/x^4 = 2398) : 
  x^2 + 1/x^2 = 20 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l2467_246754


namespace NUMINAMATH_CALUDE_power_three_mod_eleven_l2467_246767

theorem power_three_mod_eleven : 3^221 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_eleven_l2467_246767


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_reciprocal_l2467_246726

theorem cubic_roots_sum_of_cubes_reciprocal (a b c d r s : ℝ) :
  a ≠ 0 →
  c ≠ 0 →
  a * r^3 + b * r^2 + c * r + d = 0 →
  a * s^3 + b * s^2 + c * s + d = 0 →
  r ≠ 0 →
  s ≠ 0 →
  (1 / r^3) + (1 / s^3) = (b^3 - 3 * a * b * c) / c^3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_cubes_reciprocal_l2467_246726


namespace NUMINAMATH_CALUDE_divides_totient_power_two_minus_one_l2467_246753

theorem divides_totient_power_two_minus_one (n : ℕ) (hn : n > 0) : 
  n ∣ Nat.totient (2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_divides_totient_power_two_minus_one_l2467_246753


namespace NUMINAMATH_CALUDE_nested_bracket_equals_two_l2467_246707

/-- Define the operation [a, b, c] as (a + b) / c, where c ≠ 0 -/
def bracket (a b c : ℚ) : ℚ :=
  if c ≠ 0 then (a + b) / c else 0

/-- The main theorem to prove -/
theorem nested_bracket_equals_two :
  bracket (bracket 50 50 100) (bracket 3 6 9) (bracket 20 30 50) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_bracket_equals_two_l2467_246707


namespace NUMINAMATH_CALUDE_flour_info_doesnt_determine_sugar_l2467_246778

/-- Represents a cake recipe --/
structure Recipe where
  flour : ℕ
  sugar : ℕ

/-- Represents the state of Jessica's baking process --/
structure BakingProcess where
  flour_added : ℕ
  flour_needed : ℕ

/-- Given information about flour doesn't determine sugar amount --/
theorem flour_info_doesnt_determine_sugar 
  (recipe : Recipe) 
  (baking : BakingProcess) 
  (h1 : recipe.flour = 8)
  (h2 : baking.flour_added = 4)
  (h3 : baking.flour_needed = 4)
  (h4 : baking.flour_added + baking.flour_needed = recipe.flour) :
  ∃ (r1 r2 : Recipe), r1.flour = r2.flour ∧ r1.sugar ≠ r2.sugar :=
sorry

end NUMINAMATH_CALUDE_flour_info_doesnt_determine_sugar_l2467_246778


namespace NUMINAMATH_CALUDE_triangle_coloring_theorem_l2467_246714

/-- The number of ways to color 6 circles in a fixed triangular arrangement with 4 blue, 1 green, and 1 red circle -/
def triangle_coloring_ways : ℕ := 30

/-- Theorem stating that the number of ways to color the triangular arrangement is 30 -/
theorem triangle_coloring_theorem : triangle_coloring_ways = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_coloring_theorem_l2467_246714


namespace NUMINAMATH_CALUDE_next_three_same_calendar_years_l2467_246759

/-- A function that determines if a given year is a leap year -/
def isLeapYear (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

/-- The number of years between consecutive years with the same calendar -/
def calendarCycle : ℕ := 28

/-- The base year from which we start calculating -/
def baseYear : ℕ := 2024

/-- A function that calculates the nth year with the same calendar as the base year -/
def nthSameCalendarYear (n : ℕ) : ℕ :=
  baseYear + n * calendarCycle

/-- Theorem stating that the next three years following 2024 with the same calendar
    are 2052, 2080, and 2108 -/
theorem next_three_same_calendar_years :
  (nthSameCalendarYear 1 = 2052) ∧
  (nthSameCalendarYear 2 = 2080) ∧
  (nthSameCalendarYear 3 = 2108) ∧
  (isLeapYear baseYear) ∧
  (∀ n : ℕ, isLeapYear (nthSameCalendarYear n)) :=
sorry

end NUMINAMATH_CALUDE_next_three_same_calendar_years_l2467_246759


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2467_246732

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : Real.sin (α - π / 6) = 3 / 5) : 
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2467_246732


namespace NUMINAMATH_CALUDE_range_of_m_l2467_246774

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) : 
  (1 / x + 4 / y = 1) → 
  (∃ x y, x > 0 ∧ y > 0 ∧ 1 / x + 4 / y = 1 ∧ x + y / 4 < m^2 + 3*m) ↔ 
  (m < -4 ∨ m > 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2467_246774


namespace NUMINAMATH_CALUDE_unique_three_digit_integer_l2467_246751

theorem unique_three_digit_integer : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  n % 7 = 3 ∧
  n % 8 = 4 ∧
  n % 13 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_integer_l2467_246751


namespace NUMINAMATH_CALUDE_difference_of_squares_l2467_246706

theorem difference_of_squares (x y : ℝ) (h_sum : x + y = 10) (h_diff : x - y = 19) :
  x^2 - y^2 = 190 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2467_246706


namespace NUMINAMATH_CALUDE_original_number_proof_l2467_246737

theorem original_number_proof (q : ℝ) : 
  (q + 0.125 * q) - (q - 0.25 * q) = 30 → q = 80 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l2467_246737


namespace NUMINAMATH_CALUDE_car_distance_traveled_l2467_246786

/-- Calculates the distance traveled by a car given its speed and time -/
def distanceTraveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- The actual speed of the car in km/h -/
def actualSpeed : ℚ := 35

/-- The fraction of the actual speed at which the car is traveling -/
def speedFraction : ℚ := 5 / 7

/-- The time the car travels in hours -/
def travelTime : ℚ := 126 / 75

/-- The theorem stating the distance traveled by the car -/
theorem car_distance_traveled :
  distanceTraveled (speedFraction * actualSpeed) travelTime = 42 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l2467_246786


namespace NUMINAMATH_CALUDE_seating_theorem_l2467_246780

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  total_people : ℕ
  rows_with_seven : ℕ
  rows_with_six : ℕ

/-- Checks if a seating arrangement is valid --/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = 53 ∧
  s.rows_with_seven * 7 + s.rows_with_six * 6 = s.total_people

/-- The theorem to be proved --/
theorem seating_theorem :
  ∃ (s : SeatingArrangement), is_valid_arrangement s ∧ s.rows_with_seven = 5 :=
sorry

end NUMINAMATH_CALUDE_seating_theorem_l2467_246780


namespace NUMINAMATH_CALUDE_positive_rational_number_l2467_246772

theorem positive_rational_number : ∃! x : ℚ, (x > 0) ∧
  (x = 1/2 ∨ x = Real.sqrt 2 * (-1) ∨ x = 0 ∨ x = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_positive_rational_number_l2467_246772


namespace NUMINAMATH_CALUDE_ternary_121_equals_16_l2467_246784

/-- Converts a ternary (base-3) number to decimal (base-10) --/
def ternary_to_decimal (a b c : ℕ) : ℕ :=
  a * 3^2 + b * 3^1 + c * 3^0

/-- Proves that the ternary number 121₃ is equal to the decimal number 16 --/
theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_equals_16_l2467_246784


namespace NUMINAMATH_CALUDE_coefficient_x_plus_one_squared_in_x_to_tenth_l2467_246700

theorem coefficient_x_plus_one_squared_in_x_to_tenth : ∃ (a₀ a₁ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  ∀ x : ℝ, x^10 = a₀ + a₁*(x+1) + 45*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
            a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_plus_one_squared_in_x_to_tenth_l2467_246700


namespace NUMINAMATH_CALUDE_first_digit_1025_base12_l2467_246728

/-- The first digit of a number in a given base -/
def firstDigitInBase (n : ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Theorem: The first digit of 1025 (base 10) in base 12 is 7 -/
theorem first_digit_1025_base12 : firstDigitInBase 1025 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_1025_base12_l2467_246728


namespace NUMINAMATH_CALUDE_lattice_points_5_11_to_35_221_l2467_246756

/-- The number of lattice points on a line segment --/
def lattice_points_on_segment (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (5, 11) to (35, 221) is 31 --/
theorem lattice_points_5_11_to_35_221 :
  lattice_points_on_segment 5 11 35 221 = 31 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_5_11_to_35_221_l2467_246756


namespace NUMINAMATH_CALUDE_total_seashells_is_fifty_l2467_246731

/-- The number of seashells Tim found -/
def tim_seashells : ℕ := 37

/-- The number of seashells Sally found -/
def sally_seashells : ℕ := 13

/-- The total number of seashells found by Tim and Sally -/
def total_seashells : ℕ := tim_seashells + sally_seashells

/-- Theorem stating that the total number of seashells found is 50 -/
theorem total_seashells_is_fifty : total_seashells = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_is_fifty_l2467_246731


namespace NUMINAMATH_CALUDE_total_games_is_62_l2467_246710

/-- Represents a baseball league with its characteristics and calculates the total number of games played -/
structure BaseballLeague where
  teams : Nat
  games_per_team_per_month : Nat
  season_months : Nat
  playoff_rounds : Nat
  games_per_playoff_round : Nat

/-- Calculates the total number of games played in the season, including playoffs -/
def BaseballLeague.total_games (league : BaseballLeague) : Nat :=
  let regular_season_games := (league.teams / 2) * league.games_per_team_per_month * league.season_months
  let playoff_games := league.playoff_rounds * league.games_per_playoff_round
  regular_season_games + playoff_games

/-- The specific baseball league described in the problem -/
def specific_league : BaseballLeague :=
  { teams := 8
  , games_per_team_per_month := 7
  , season_months := 2
  , playoff_rounds := 3
  , games_per_playoff_round := 2
  }

/-- Theorem stating that the total number of games in the specific league is 62 -/
theorem total_games_is_62 : specific_league.total_games = 62 := by
  sorry


end NUMINAMATH_CALUDE_total_games_is_62_l2467_246710


namespace NUMINAMATH_CALUDE_sum_base8_equals_1207_l2467_246795

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimalToBase8 (n / 8)

/-- The sum of 527₈, 165₈, and 273₈ in base 8 is equal to 1207₈ -/
theorem sum_base8_equals_1207 :
  let a := base8ToDecimal [7, 2, 5]
  let b := base8ToDecimal [5, 6, 1]
  let c := base8ToDecimal [3, 7, 2]
  decimalToBase8 (a + b + c) = [7, 0, 2, 1] := by
  sorry

end NUMINAMATH_CALUDE_sum_base8_equals_1207_l2467_246795


namespace NUMINAMATH_CALUDE_root_equation_problem_l2467_246727

theorem root_equation_problem (c d : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    ((x + c) * (x + d) * (x + 10)) / ((x + 5)^2) = 0 ∧
    ((y + c) * (y + d) * (y + 10)) / ((y + 5)^2) = 0 ∧
    ((z + c) * (z + d) * (z + 10)) / ((z + 5)^2) = 0) ∧
  (∃! w : ℝ, ((w + 3*c) * (w + 2) * (w + 4)) / ((w + d) * (w + 10)) = 0) →
  50 * c + 10 * d = 310 / 3 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l2467_246727


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2467_246701

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2467_246701


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l2467_246779

theorem fraction_product_theorem (fractions : Finset (ℕ × ℕ)) : 
  (fractions.card = 48) →
  (∀ (n : ℕ), n ∈ fractions.image Prod.fst → 2 ≤ n ∧ n ≤ 49) →
  (∀ (d : ℕ), d ∈ fractions.image Prod.snd → 2 ≤ d ∧ d ≤ 49) →
  (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 49 → (fractions.filter (λ f => f.fst = k)).card = 1) →
  (∀ (k : ℕ), 2 ≤ k ∧ k ≤ 49 → (fractions.filter (λ f => f.snd = k)).card = 1) →
  (∃ (f : ℕ × ℕ), f ∈ fractions ∧ f.fst % f.snd = 0) ∨
  (∃ (subset : Finset (ℕ × ℕ)), subset ⊆ fractions ∧ subset.card ≤ 25 ∧ 
    (subset.prod (λ f => f.fst) % subset.prod (λ f => f.snd) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l2467_246779


namespace NUMINAMATH_CALUDE_exhibit_fish_count_l2467_246740

/-- The number of pufferfish in the exhibit -/
def num_pufferfish : ℕ := 15

/-- The ratio of swordfish to pufferfish -/
def swordfish_ratio : ℕ := 5

/-- The total number of fish in the exhibit -/
def total_fish : ℕ := num_pufferfish + swordfish_ratio * num_pufferfish

theorem exhibit_fish_count : total_fish = 90 := by
  sorry

end NUMINAMATH_CALUDE_exhibit_fish_count_l2467_246740


namespace NUMINAMATH_CALUDE_pyramid_volume_integer_heights_l2467_246793

theorem pyramid_volume_integer_heights (base_side : ℕ) (height : ℕ) :
  base_side = 640 →
  height = 1024 →
  (∃ (n : ℕ), n = 85 ∧
    (∀ h : ℕ, h < height →
      (25 * (height - h)^3) % 192 = 0 ↔ h ∈ Finset.range (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_integer_heights_l2467_246793


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_800_by_110_percent_l2467_246770

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) : 
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_800_by_110_percent : 
  800 * (1 + 110 / 100) = 1680 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_800_by_110_percent_l2467_246770


namespace NUMINAMATH_CALUDE_negation_existence_sufficient_not_necessary_sufficient_necessary_relationship_quadratic_inequality_condition_l2467_246712

-- 1. Negation of existence statement
theorem negation_existence : 
  (¬ ∃ x : ℝ, x ≥ 1 ∧ x^2 > 1) ↔ (∀ x : ℝ, x ≥ 1 → x^2 ≤ 1) := by sorry

-- 2. Sufficient but not necessary condition
theorem sufficient_not_necessary :
  (∃ x : ℝ, x ≠ 1 ∧ x^2 + 2*x - 3 = 0) ∧
  (∀ x : ℝ, x = 1 → x^2 + 2*x - 3 = 0) := by sorry

-- 3. Relationship between sufficient and necessary conditions
theorem sufficient_necessary_relationship (p q s : Prop) :
  ((p → q) ∧ (q → s)) → (p → s) := by sorry

-- 4. Conditions for quadratic inequality
theorem quadratic_inequality_condition (m : ℝ) :
  (¬ ∃ x : ℝ, m*x^2 + m*x + 1 < 0) → (0 ≤ m ∧ m ≤ 4) := by sorry

end NUMINAMATH_CALUDE_negation_existence_sufficient_not_necessary_sufficient_necessary_relationship_quadratic_inequality_condition_l2467_246712


namespace NUMINAMATH_CALUDE_johns_investment_l2467_246745

theorem johns_investment (total_investment : ℝ) (alpha_rate beta_rate : ℝ) 
  (total_after_year : ℝ) (alpha_investment : ℝ) :
  total_investment = 1500 →
  alpha_rate = 0.04 →
  beta_rate = 0.06 →
  total_after_year = 1575 →
  alpha_investment = 750 →
  alpha_investment * (1 + alpha_rate) + 
    (total_investment - alpha_investment) * (1 + beta_rate) = total_after_year :=
by sorry

end NUMINAMATH_CALUDE_johns_investment_l2467_246745


namespace NUMINAMATH_CALUDE_smallest_root_of_g_l2467_246792

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 7

-- State the theorem
theorem smallest_root_of_g :
  ∃ (r : ℝ), r = -Real.sqrt (7/5) ∧
  (∀ x : ℝ, g x = 0 → r ≤ x) ∧
  g r = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_root_of_g_l2467_246792


namespace NUMINAMATH_CALUDE_triangle_properties_l2467_246709

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (A + B + C = π) →
  -- Side lengths are positive
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  -- Law of cosines
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →
  -- Prove the three properties
  ((A > B ↔ Real.sin A > Real.sin B) ∧
   (B = π/3 ∧ b^2 = a*c → A = π/3 ∧ B = π/3 ∧ C = π/3) ∧
   (b = a * Real.cos C + c * Real.sin A → A = π/4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2467_246709


namespace NUMINAMATH_CALUDE_quadratic_root_l2467_246788

/- Given a quadratic equation x^2 - (m+n)x + mn - p = 0 with roots α and β -/
theorem quadratic_root (m n p : ℤ) (α β : ℝ) 
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_roots : ∀ x : ℝ, x^2 - (m+n)*x + mn - p = 0 ↔ x = α ∨ x = β)
  (h_alpha : α = 3) :
  β = m + n - 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_l2467_246788


namespace NUMINAMATH_CALUDE_cookie_calories_l2467_246744

/-- Calculates the number of calories per cookie in a box of cookies. -/
def calories_per_cookie (cookies_per_bag : ℕ) (bags_per_box : ℕ) (total_calories : ℕ) : ℕ :=
  total_calories / (cookies_per_bag * bags_per_box)

/-- Theorem: Given a box of cookies with 4 bags, 20 cookies per bag, and a total of 1600 calories,
    each cookie contains 20 calories. -/
theorem cookie_calories :
  calories_per_cookie 20 4 1600 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cookie_calories_l2467_246744


namespace NUMINAMATH_CALUDE_function_inequality_l2467_246787

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2467_246787


namespace NUMINAMATH_CALUDE_division_of_fractions_l2467_246799

theorem division_of_fractions : (4 : ℚ) / (8 / 13) = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l2467_246799


namespace NUMINAMATH_CALUDE_exponent_division_l2467_246791

theorem exponent_division (a : ℝ) : a^3 / a^2 = a := by sorry

end NUMINAMATH_CALUDE_exponent_division_l2467_246791


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_fractional_equation_solution_l2467_246762

-- Problem 1
theorem quadratic_equation_solution (x : ℝ) :
  x^2 + 6*x - 1 = 0 ↔ x = Real.sqrt 10 - 3 ∨ x = -Real.sqrt 10 - 3 :=
sorry

-- Problem 2
theorem fractional_equation_solution (x : ℝ) :
  x ≠ -2 ∧ x ≠ 1 →
  (x / (x + 2) = 2 / (x - 1) + 1 ↔ x = -1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_fractional_equation_solution_l2467_246762


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2467_246721

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_two : x + y + z = 2) : 
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2467_246721


namespace NUMINAMATH_CALUDE_m_range_l2467_246704

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x, -m * x^2 + 2*x - m > 0

def q (m : ℝ) : Prop := ∀ x > 0, (4/x + x - (m - 1)) > 2

-- Define the theorem
theorem m_range :
  (∃ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (∃ m : ℝ, m ≥ -1 ∧ m < 3) ∧ (∀ m : ℝ, m < -1 ∨ m ≥ 3 → ¬(p m ∨ q m) ∨ (p m ∧ q m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l2467_246704


namespace NUMINAMATH_CALUDE_ballsInBoxes_eq_36_l2467_246771

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
def ballsInBoxes : ℕ := starsAndBars 7 3

theorem ballsInBoxes_eq_36 : ballsInBoxes = 36 := by
  sorry

end NUMINAMATH_CALUDE_ballsInBoxes_eq_36_l2467_246771


namespace NUMINAMATH_CALUDE_touching_spheres_radii_l2467_246724

/-- Given four spheres of radius r, where each sphere touches the other three,
    calculate the radii of spheres that touch all four spheres internally and externally. -/
theorem touching_spheres_radii (r : ℝ) (r_pos : r > 0) :
  ∃ (p R : ℝ),
    (p = r * (Real.sqrt 6 / 2 - 1)) ∧
    (R = r * (Real.sqrt 6 / 2 + 1)) ∧
    (p > 0) ∧ (R > 0) :=
by sorry

end NUMINAMATH_CALUDE_touching_spheres_radii_l2467_246724


namespace NUMINAMATH_CALUDE_team_e_not_played_b_l2467_246743

/-- Represents a soccer team in the tournament -/
inductive Team : Type
  | A | B | C | D | E | F

/-- The number of matches played by each team at a certain point -/
def matches_played (t : Team) : ℕ :=
  match t with
  | Team.A => 5
  | Team.B => 4
  | Team.C => 3
  | Team.D => 2
  | Team.E => 1
  | Team.F => 0

/-- Predicate to check if two teams have played against each other -/
def has_played_against (t1 t2 : Team) : Prop :=
  sorry

/-- The total number of teams in the tournament -/
def total_teams : ℕ := 6

/-- The maximum number of matches a team can play in a round-robin tournament -/
def max_matches : ℕ := total_teams - 1

theorem team_e_not_played_b :
  matches_played Team.A = max_matches ∧
  matches_played Team.E = 1 →
  ¬ has_played_against Team.E Team.B :=
by sorry

end NUMINAMATH_CALUDE_team_e_not_played_b_l2467_246743


namespace NUMINAMATH_CALUDE_equation_equality_l2467_246720

theorem equation_equality : ∀ x y : ℝ, 9*x*y - 6*x*y = 3*x*y := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l2467_246720


namespace NUMINAMATH_CALUDE_chess_tournament_l2467_246735

theorem chess_tournament (W M : ℕ) 
  (h1 : W * (W - 1) / 2 = 45)  -- Number of games with both women
  (h2 : W * M = 200)           -- Number of games with one man and one woman
  : M * (M - 1) / 2 = 190 :=   -- Number of games with both men
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_l2467_246735


namespace NUMINAMATH_CALUDE_not_right_triangle_l2467_246703

theorem not_right_triangle (A B C : ℝ) (h : A + B + C = 180) 
  (h_ratio : A / 3 = B / 4 ∧ B / 4 = C / 5) : 
  A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l2467_246703


namespace NUMINAMATH_CALUDE_min_ops_to_500_l2467_246760

def calculator_ops (n : ℕ) : ℕ → ℕ
| 0     => n
| (k+1) => calculator_ops (min (2*n) (n+1)) k

theorem min_ops_to_500 : ∃ k, calculator_ops 1 k = 500 ∧ ∀ j, j < k → calculator_ops 1 j ≠ 500 :=
  sorry

end NUMINAMATH_CALUDE_min_ops_to_500_l2467_246760


namespace NUMINAMATH_CALUDE_paint_usage_l2467_246722

theorem paint_usage (initial_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ)
  (h1 : initial_paint = 360)
  (h2 : first_week_fraction = 1/6)
  (h3 : second_week_fraction = 1/5) :
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 120 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_l2467_246722


namespace NUMINAMATH_CALUDE_count_adjacent_arrangements_l2467_246783

/-- The number of distinct arrangements of the letters in "КАРАКАТИЦА" where 'Р' and 'Ц' are adjacent -/
def adjacent_arrangements : ℕ := 15120

/-- The word from which we are forming arrangements -/
def word : String := "КАРАКАТИЦА"

/-- The length of the word -/
def word_length : ℕ := word.length

/-- The number of 'А's in the word -/
def count_A : ℕ := (word.toList.filter (· = 'А')).length

/-- The number of 'К's in the word -/
def count_K : ℕ := (word.toList.filter (· = 'К')).length

/-- Theorem stating that the number of distinct arrangements of the letters in "КАРАКАТИЦА" 
    where 'Р' and 'Ц' are adjacent is equal to adjacent_arrangements -/
theorem count_adjacent_arrangements :
  adjacent_arrangements = 
    2 * (Nat.factorial (word_length - 1)) / 
    (Nat.factorial count_A * Nat.factorial count_K) :=
by sorry

end NUMINAMATH_CALUDE_count_adjacent_arrangements_l2467_246783


namespace NUMINAMATH_CALUDE_freshman_count_l2467_246764

theorem freshman_count (total : ℕ) (f s j r : ℕ) : 
  total = 2158 →
  5 * s = 4 * f →
  8 * s = 7 * j →
  7 * j = 9 * r →
  total = f + s + j + r →
  f = 630 := by
sorry

end NUMINAMATH_CALUDE_freshman_count_l2467_246764


namespace NUMINAMATH_CALUDE_school_survey_most_suitable_for_census_l2467_246775

/-- Represents a survey type --/
inductive SurveyType
  | CityResidents
  | CarBatch
  | LightTubeBatch
  | SchoolStudents

/-- Determines if a survey type is suitable for a census --/
def isSuitableForCensus (s : SurveyType) : Prop :=
  match s with
  | .SchoolStudents => True
  | _ => False

/-- Theorem stating that the school students survey is the most suitable for a census --/
theorem school_survey_most_suitable_for_census :
  ∀ s : SurveyType, isSuitableForCensus s ↔ s = SurveyType.SchoolStudents :=
by sorry

end NUMINAMATH_CALUDE_school_survey_most_suitable_for_census_l2467_246775


namespace NUMINAMATH_CALUDE_helmet_safety_analysis_l2467_246733

/-- Data for people not wearing helmets over 4 years -/
def helmet_data : List (Nat × Nat) := [(1, 1250), (2, 1050), (3, 1000), (4, 900)]

/-- Contingency table for helmet wearing and casualties -/
def contingency_table : Matrix (Fin 2) (Fin 2) Nat :=
  ![![7, 3],
    ![13, 27]]

/-- Calculate the regression line equation coefficients -/
def regression_line (data : List (Nat × Nat)) : ℝ × ℝ :=
  sorry

/-- Estimate the number of people not wearing helmets for a given year -/
def estimate_no_helmet (coef : ℝ × ℝ) (year : Nat) : ℝ :=
  sorry

/-- Calculate the K^2 statistic for a 2x2 contingency table -/
def k_squared (table : Matrix (Fin 2) (Fin 2) Nat) : ℝ :=
  sorry

theorem helmet_safety_analysis :
  let (b, a) := regression_line helmet_data
  (b = -110 ∧ a = 1325) ∧
  estimate_no_helmet (b, a) 5 = 775 ∧
  k_squared contingency_table > 3.841 :=
sorry

end NUMINAMATH_CALUDE_helmet_safety_analysis_l2467_246733


namespace NUMINAMATH_CALUDE_total_players_l2467_246777

theorem total_players (kabaddi : ℕ) (kho_kho_only : ℕ) (both : ℕ) 
  (h1 : kabaddi = 10) 
  (h2 : kho_kho_only = 40) 
  (h3 : both = 5) : 
  kabaddi + kho_kho_only - both = 50 := by
  sorry

#check total_players

end NUMINAMATH_CALUDE_total_players_l2467_246777


namespace NUMINAMATH_CALUDE_complement_of_union_l2467_246765

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {3, 4, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2467_246765


namespace NUMINAMATH_CALUDE_remainder_problem_l2467_246776

theorem remainder_problem (N : ℕ) (R : ℕ) (h1 : R < 100) (h2 : ∃ k : ℕ, N = 100 * k + R) (h3 : ∃ m : ℕ, N = R * m + 1) : R = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2467_246776


namespace NUMINAMATH_CALUDE_spade_equation_solution_l2467_246705

/-- Definition of the spade operation -/
def spade (A B : ℝ) : ℝ := 4 * A + 3 * B + 6

/-- Theorem stating that 9.5 is the unique solution to A ♠ 5 = 59 -/
theorem spade_equation_solution :
  ∃! A : ℝ, spade A 5 = 59 ∧ A = 9.5 := by sorry

end NUMINAMATH_CALUDE_spade_equation_solution_l2467_246705


namespace NUMINAMATH_CALUDE_megans_acorns_l2467_246798

/-- The initial number of acorns Megan had -/
def T : ℝ := 20

/-- Theorem stating the conditions and the correct answer for Megan's acorn problem -/
theorem megans_acorns :
  (0.35 * T = 7) ∧ (0.45 * T = 9) ∧ (T = 20) := by
  sorry

#check megans_acorns

end NUMINAMATH_CALUDE_megans_acorns_l2467_246798


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2467_246723

theorem min_value_quadratic (x y : ℝ) :
  y = x^2 + 16*x + 20 → ∀ z : ℝ, y ≥ -44 ∧ (∃ x₀ : ℝ, x₀^2 + 16*x₀ + 20 = -44) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2467_246723


namespace NUMINAMATH_CALUDE_percentage_difference_l2467_246702

theorem percentage_difference : (40 * 0.8) - (25 * (2/5)) = 22 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2467_246702


namespace NUMINAMATH_CALUDE_negative_one_to_zero_power_l2467_246752

theorem negative_one_to_zero_power : ((-1 : ℤ) ^ (0 : ℕ)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_to_zero_power_l2467_246752


namespace NUMINAMATH_CALUDE_triangle_angle_B_l2467_246781

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) : 
  a = Real.sqrt 3 →
  b = Real.sqrt 6 →
  A = π / 6 →
  (B = π / 4 ∨ B = 3 * π / 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l2467_246781


namespace NUMINAMATH_CALUDE_negation_equivalence_l2467_246794

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2467_246794


namespace NUMINAMATH_CALUDE_line_through_point_l2467_246742

/-- The value of b for which the line bx + (b-1)y = b+3 passes through the point (3, -7) -/
theorem line_through_point (b : ℚ) : 
  (b * 3 + (b - 1) * (-7) = b + 3) → b = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2467_246742


namespace NUMINAMATH_CALUDE_y_equals_x_cubed_l2467_246739

/-- Represents a pair of x and y values from the table -/
structure XYPair where
  x : ℕ
  y : ℕ

/-- The set of (x, y) pairs from the given table -/
def xyTable : List XYPair := [
  ⟨1, 1⟩,
  ⟨2, 8⟩,
  ⟨3, 27⟩,
  ⟨4, 64⟩,
  ⟨5, 125⟩
]

/-- Theorem stating that y = x^3 holds for all pairs in the table -/
theorem y_equals_x_cubed (pair : XYPair) (h : pair ∈ xyTable) : pair.y = pair.x ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_y_equals_x_cubed_l2467_246739


namespace NUMINAMATH_CALUDE_ship_passengers_asia_fraction_l2467_246718

theorem ship_passengers_asia_fraction (total : ℕ) 
  (north_america : ℚ) (europe : ℚ) (africa : ℚ) (other : ℕ) :
  total = 108 →
  north_america = 1 / 12 →
  europe = 1 / 4 →
  africa = 1 / 9 →
  other = 42 →
  (north_america + europe + africa + (other : ℚ) / total + 1 / 6 : ℚ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ship_passengers_asia_fraction_l2467_246718


namespace NUMINAMATH_CALUDE_shirts_washed_l2467_246716

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (not_washed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 27 → not_washed = 16 →
  short_sleeve + long_sleeve - not_washed = 20 := by
  sorry

end NUMINAMATH_CALUDE_shirts_washed_l2467_246716


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l2467_246749

theorem dogwood_tree_count (current_trees new_trees : ℕ) 
  (h1 : current_trees = 34)
  (h2 : new_trees = 49) :
  current_trees + new_trees = 83 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l2467_246749


namespace NUMINAMATH_CALUDE_polynomial_inequality_l2467_246797

theorem polynomial_inequality (a b c : ℝ) 
  (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l2467_246797


namespace NUMINAMATH_CALUDE_lunch_choices_l2467_246782

theorem lunch_choices (chicken_types : ℕ) (drink_types : ℕ) 
  (h1 : chicken_types = 3) (h2 : drink_types = 2) : 
  chicken_types * drink_types = 6 := by
sorry

end NUMINAMATH_CALUDE_lunch_choices_l2467_246782


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l2467_246763

/-- The profit percentage of a dishonest dealer who uses a weight of 600 grams per kg while selling at the professed cost price. -/
theorem dishonest_dealer_profit_percentage :
  let actual_weight : ℝ := 600  -- grams
  let claimed_weight : ℝ := 1000  -- grams (1 kg)
  let profit_ratio := (claimed_weight - actual_weight) / actual_weight
  profit_ratio * 100 = 200 / 3 :=
by sorry

end NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l2467_246763


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2467_246746

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2467_246746


namespace NUMINAMATH_CALUDE_ship_passengers_heads_l2467_246730

/-- Represents the number of heads and legs on a ship with cats, crew, and a one-legged captain. -/
structure ShipPassengers where
  cats : ℕ
  crew : ℕ
  captain : ℕ := 1

/-- Calculates the total number of heads on the ship. -/
def totalHeads (p : ShipPassengers) : ℕ :=
  p.cats + p.crew + p.captain

/-- Calculates the total number of legs on the ship. -/
def totalLegs (p : ShipPassengers) : ℕ :=
  p.cats * 4 + p.crew * 2 + 1

/-- Theorem stating that given the conditions, the total number of heads on the ship is 14. -/
theorem ship_passengers_heads :
  ∃ (p : ShipPassengers),
    p.cats = 7 ∧
    totalLegs p = 41 ∧
    totalHeads p = 14 :=
sorry

end NUMINAMATH_CALUDE_ship_passengers_heads_l2467_246730


namespace NUMINAMATH_CALUDE_force_resultant_arithmetic_mean_l2467_246773

/-- Given two forces p₁ and p₂ forming an angle α, if their resultant is equal to their arithmetic mean, 
    then the angle α is between 120° and 180°, and the ratio of the forces is between 1/3 and 3. -/
theorem force_resultant_arithmetic_mean 
  (p₁ p₂ : ℝ) 
  (α : Real) 
  (h_positive : p₁ > 0 ∧ p₂ > 0) 
  (h_resultant : Real.sqrt (p₁^2 + p₂^2 + 2*p₁*p₂*(Real.cos α)) = (p₁ + p₂)/2) : 
  (2*π/3 ≤ α ∧ α ≤ π) ∧ (1/3 ≤ p₁/p₂ ∧ p₁/p₂ ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_force_resultant_arithmetic_mean_l2467_246773


namespace NUMINAMATH_CALUDE_march_starts_on_friday_l2467_246715

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with its properties -/
structure Month where
  days : Nat
  first_day : Weekday
  monday_count : Nat
  thursday_count : Nat

/-- The specific March we're considering -/
def march : Month :=
  { days := 31
  , first_day := Weekday.Friday  -- This is what we want to prove
  , monday_count := 5
  , thursday_count := 5 }

/-- Main theorem: If March has 31 days, 5 Mondays, and 5 Thursdays, then it starts on a Friday -/
theorem march_starts_on_friday :
  march.days = 31 ∧ march.monday_count = 5 ∧ march.thursday_count = 5 →
  march.first_day = Weekday.Friday :=
sorry

end NUMINAMATH_CALUDE_march_starts_on_friday_l2467_246715


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2467_246725

/-- Calculates the weight of a new person in a group replacement scenario. -/
def newPersonWeight (groupSize : ℕ) (avgWeightIncrease : ℝ) (replacedPersonWeight : ℝ) : ℝ :=
  replacedPersonWeight + groupSize * avgWeightIncrease

/-- Proves that the weight of the new person is 108 kg given the specified conditions. -/
theorem weight_of_new_person :
  let groupSize : ℕ := 15
  let avgWeightIncrease : ℝ := 2.2
  let replacedPersonWeight : ℝ := 75
  newPersonWeight groupSize avgWeightIncrease replacedPersonWeight = 108 := by
  sorry

#eval newPersonWeight 15 2.2 75

end NUMINAMATH_CALUDE_weight_of_new_person_l2467_246725


namespace NUMINAMATH_CALUDE_scientific_notation_50300_l2467_246750

theorem scientific_notation_50300 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 50300 = a * (10 : ℝ) ^ n ∧ a = 5.03 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_50300_l2467_246750


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2467_246785

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2467_246785


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l2467_246741

theorem mango_rate_calculation (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (total_paid : ℕ) :
  grape_quantity = 3 →
  grape_rate = 70 →
  mango_quantity = 9 →
  total_paid = 705 →
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l2467_246741


namespace NUMINAMATH_CALUDE_cube_painting_cost_l2467_246789

/-- The cost to paint a cube with given dimensions and paint properties -/
theorem cube_painting_cost
  (side_length : ℝ)
  (paint_cost_per_kg : ℝ)
  (paint_coverage_per_kg : ℝ)
  (h_side : side_length = 10)
  (h_cost : paint_cost_per_kg = 60)
  (h_coverage : paint_coverage_per_kg = 20) :
  side_length ^ 2 * 6 / paint_coverage_per_kg * paint_cost_per_kg = 1800 :=
by sorry

end NUMINAMATH_CALUDE_cube_painting_cost_l2467_246789


namespace NUMINAMATH_CALUDE_eighth_term_of_specific_geometric_sequence_l2467_246736

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

-- Theorem statement
theorem eighth_term_of_specific_geometric_sequence :
  geometric_sequence 8 2 8 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_specific_geometric_sequence_l2467_246736


namespace NUMINAMATH_CALUDE_total_monthly_payment_l2467_246719

/-- Calculates the total monthly payment for employees after new hires --/
theorem total_monthly_payment
  (initial_employees : ℕ)
  (hourly_rate : ℚ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (additional_hires : ℕ)
  (h1 : initial_employees = 500)
  (h2 : hourly_rate = 12)
  (h3 : hours_per_day = 10)
  (h4 : days_per_week = 5)
  (h5 : weeks_per_month = 4)
  (h6 : additional_hires = 200) :
  (initial_employees + additional_hires) *
  (hourly_rate * hours_per_day * days_per_week * weeks_per_month) = 1680000 := by
  sorry

#eval (500 + 200) * (12 * 10 * 5 * 4)

end NUMINAMATH_CALUDE_total_monthly_payment_l2467_246719


namespace NUMINAMATH_CALUDE_local_politics_coverage_l2467_246717

/-- The percentage of reporters covering politics -/
def politics_coverage : ℝ := 100 - 92.85714285714286

/-- The percentage of reporters covering local politics among those covering politics -/
def local_coverage_ratio : ℝ := 100 - 30

theorem local_politics_coverage :
  (local_coverage_ratio * politics_coverage / 100) = 5 := by sorry

end NUMINAMATH_CALUDE_local_politics_coverage_l2467_246717


namespace NUMINAMATH_CALUDE_retained_pits_problem_l2467_246761

/-- The maximum number of pits that can be retained on a road --/
def max_retained_pits (road_length : ℕ) (initial_spacing : ℕ) (revised_spacing : ℕ) : ℕ :=
  2 * (road_length / (initial_spacing * revised_spacing) + 1)

/-- Theorem stating the maximum number of retained pits for the given problem --/
theorem retained_pits_problem :
  max_retained_pits 120 3 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_retained_pits_problem_l2467_246761


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l2467_246734

/-- The cost price of one meter of cloth given the selling price, quantity, and profit per meter -/
theorem cost_price_per_meter
  (selling_price : ℕ)
  (quantity : ℕ)
  (profit_per_meter : ℕ)
  (h1 : selling_price = 8925)
  (h2 : quantity = 85)
  (h3 : profit_per_meter = 20) :
  (selling_price - quantity * profit_per_meter) / quantity = 85 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l2467_246734


namespace NUMINAMATH_CALUDE_vector_scalar_mult_and_add_l2467_246711

theorem vector_scalar_mult_and_add :
  (3 : ℝ) • ((-3 : ℝ), (2 : ℝ), (-5 : ℝ)) + ((1 : ℝ), (7 : ℝ), (-3 : ℝ)) = ((-8 : ℝ), (13 : ℝ), (-18 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_vector_scalar_mult_and_add_l2467_246711


namespace NUMINAMATH_CALUDE_thomas_total_training_hours_l2467_246769

/-- Calculates the total training hours for Thomas given his training schedule --/
def total_training_hours : ℕ :=
  let first_phase_days : ℕ := 15
  let first_phase_hours_per_day : ℕ := 5
  let second_phase_days : ℕ := 15
  let second_phase_rest_days : ℕ := 3
  let third_phase_days : ℕ := 12
  let third_phase_rest_days : ℕ := 2
  let new_schedule_morning_hours : ℕ := 4
  let new_schedule_evening_hours : ℕ := 3

  let first_phase_total := first_phase_days * first_phase_hours_per_day
  let second_phase_total := (second_phase_days - second_phase_rest_days) * (new_schedule_morning_hours + new_schedule_evening_hours)
  let third_phase_total := (third_phase_days - third_phase_rest_days) * (new_schedule_morning_hours + new_schedule_evening_hours)

  first_phase_total + second_phase_total + third_phase_total

/-- Theorem stating that Thomas' total training hours is 229 --/
theorem thomas_total_training_hours :
  total_training_hours = 229 := by
  sorry

end NUMINAMATH_CALUDE_thomas_total_training_hours_l2467_246769


namespace NUMINAMATH_CALUDE_no_77_cents_combination_l2467_246758

/-- Represents the set of available coin values in cents -/
def CoinValues : Set ℕ := {1, 5, 10, 50}

/-- Represents a selection of exactly three coins -/
def CoinSelection := Fin 3 → ℕ

/-- The sum of a coin selection -/
def sum_coins (selection : CoinSelection) : ℕ :=
  (selection 0) + (selection 1) + (selection 2)

/-- Predicate to check if a selection is valid (all coins are from CoinValues) -/
def valid_selection (selection : CoinSelection) : Prop :=
  ∀ i, selection i ∈ CoinValues

theorem no_77_cents_combination :
  ¬∃ (selection : CoinSelection), valid_selection selection ∧ sum_coins selection = 77 := by
  sorry

#check no_77_cents_combination

end NUMINAMATH_CALUDE_no_77_cents_combination_l2467_246758


namespace NUMINAMATH_CALUDE_cube_pyramid_volume_equality_l2467_246729

theorem cube_pyramid_volume_equality (h : ℝ) : 
  let cube_edge : ℝ := 6
  let pyramid_base : ℝ := 12
  let cube_volume : ℝ := cube_edge^3
  let pyramid_volume : ℝ := (1/3) * pyramid_base^2 * h
  cube_volume = pyramid_volume → h = 4.5 := by
sorry

end NUMINAMATH_CALUDE_cube_pyramid_volume_equality_l2467_246729


namespace NUMINAMATH_CALUDE_pick_shoes_five_pairs_l2467_246757

/-- The number of ways to pick 4 shoes from 5 pairs such that exactly one pair is among them -/
def pick_shoes (num_pairs : ℕ) : ℕ := 
  num_pairs * (Nat.choose (num_pairs - 1) 2) * 2 * 2

/-- Theorem stating that picking 4 shoes from 5 pairs with exactly one pair among them can be done in 120 ways -/
theorem pick_shoes_five_pairs : pick_shoes 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_pick_shoes_five_pairs_l2467_246757


namespace NUMINAMATH_CALUDE_probability_red_is_half_l2467_246748

def bag_contents : ℕ × ℕ := (3, 3)

def probability_red (contents : ℕ × ℕ) : ℚ :=
  contents.1 / (contents.1 + contents.2)

theorem probability_red_is_half : 
  probability_red bag_contents = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_red_is_half_l2467_246748


namespace NUMINAMATH_CALUDE_coefficient_x2y3_in_binomial_expansion_l2467_246766

theorem coefficient_x2y3_in_binomial_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k) * x^k * y^(5-k)) =
  10 * x^2 * y^3 + (Finset.range 6).sum (fun k => if k ≠ 2 then (Nat.choose 5 k) * x^k * y^(5-k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x2y3_in_binomial_expansion_l2467_246766


namespace NUMINAMATH_CALUDE_leila_money_left_l2467_246755

def money_left_after_shopping (initial_money sweater_cost jewelry_cost : ℕ) : ℕ :=
  initial_money - (sweater_cost + jewelry_cost)

theorem leila_money_left :
  ∀ (sweater_cost : ℕ),
    sweater_cost = 40 →
    ∀ (initial_money : ℕ),
      initial_money = 4 * sweater_cost →
      ∀ (jewelry_cost : ℕ),
        jewelry_cost = sweater_cost + 60 →
        money_left_after_shopping initial_money sweater_cost jewelry_cost = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_leila_money_left_l2467_246755


namespace NUMINAMATH_CALUDE_first_solution_concentration_l2467_246738

-- Define the variables
def total_volume : ℝ := 630
def final_concentration : ℝ := 50
def first_solution_volume : ℝ := 420
def second_solution_concentration : ℝ := 30

-- Define the theorem
theorem first_solution_concentration :
  ∃ (x : ℝ),
    x * first_solution_volume / 100 +
    second_solution_concentration * (total_volume - first_solution_volume) / 100 =
    final_concentration * total_volume / 100 ∧
    x = 60 := by
  sorry

end NUMINAMATH_CALUDE_first_solution_concentration_l2467_246738


namespace NUMINAMATH_CALUDE_train_speed_l2467_246713

/-- The speed of a train given its length and time to pass a stationary point. -/
theorem train_speed (length time : ℝ) (h1 : length = 300) (h2 : time = 6) :
  length / time = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2467_246713


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2467_246747

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  -- Given conditions
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A ∧
  b = 3 ∧
  c = 2 →
  -- Conclusions
  A = π / 3 ∧ a = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2467_246747


namespace NUMINAMATH_CALUDE_intersection_M_N_l2467_246790

-- Define the sets M and N
def M : Set ℝ := {s | |s| < 4}
def N : Set ℝ := {x | 3 * x ≥ -1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | -1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2467_246790


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2467_246768

open Real

theorem sufficient_not_necessary_condition : 
  (∀ α : ℝ, ∃ k : ℤ, α = π / 6 + 2 * k * π → cos (2 * α) = 1 / 2) ∧ 
  (∃ α : ℝ, cos (2 * α) = 1 / 2 ∧ ∀ k : ℤ, α ≠ π / 6 + 2 * k * π) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2467_246768


namespace NUMINAMATH_CALUDE_high_school_total_students_l2467_246708

/-- Represents a high school with three grades -/
structure HighSchool where
  freshman : ℕ
  sophomore : ℕ
  senior : ℕ

/-- Represents a stratified sample from the high school -/
structure StratifiedSample where
  freshman : ℕ
  sophomore : ℕ
  senior : ℕ
  total : ℕ

theorem high_school_total_students (hs : HighSchool) (sample : StratifiedSample) : 
  hs.senior = 1000 →
  sample.freshman = 75 →
  sample.sophomore = 60 →
  sample.total = 185 →
  hs.freshman + hs.sophomore + hs.senior = 3700 := by
  sorry

#check high_school_total_students

end NUMINAMATH_CALUDE_high_school_total_students_l2467_246708
