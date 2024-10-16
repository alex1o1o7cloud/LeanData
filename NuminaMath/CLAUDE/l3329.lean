import Mathlib

namespace NUMINAMATH_CALUDE_oliver_candy_to_janet_l3329_332940

theorem oliver_candy_to_janet (initial_candy : ℕ) (remaining_candy : ℕ) : 
  initial_candy = 78 → remaining_candy = 68 → initial_candy - remaining_candy = 10 := by
  sorry

end NUMINAMATH_CALUDE_oliver_candy_to_janet_l3329_332940


namespace NUMINAMATH_CALUDE_yacht_distance_squared_bounds_l3329_332907

theorem yacht_distance_squared_bounds (θ : Real) 
  (h1 : 30 * Real.pi / 180 ≤ θ) 
  (h2 : θ ≤ 75 * Real.pi / 180) : 
  ∃ (AC : Real), 200 ≤ AC^2 ∧ AC^2 ≤ 656 := by
  sorry

end NUMINAMATH_CALUDE_yacht_distance_squared_bounds_l3329_332907


namespace NUMINAMATH_CALUDE_power_calculation_l3329_332938

theorem power_calculation : 2^300 + 9^3 / 9^2 - 3^4 = 2^300 - 72 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l3329_332938


namespace NUMINAMATH_CALUDE_number_thought_of_l3329_332923

theorem number_thought_of (x : ℝ) : (x / 5 + 10 = 21) → x = 55 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l3329_332923


namespace NUMINAMATH_CALUDE_quadratic_equation_root_value_l3329_332910

theorem quadratic_equation_root_value (a b : ℝ) : 
  (∀ x, a * x^2 + b * x = 6) → -- The quadratic equation
  (a * 2^2 + b * 2 = 6) →     -- x = 2 is a root
  4 * a + 2 * b = 6 :=        -- The value of 4a + 2b
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_value_l3329_332910


namespace NUMINAMATH_CALUDE_cube_distance_to_plane_l3329_332984

theorem cube_distance_to_plane (cube_side : ℝ) (height1 height2 height3 : ℝ) 
  (r s t : ℕ+) (d : ℝ) :
  cube_side = 15 →
  height1 = 15 ∧ height2 = 16 ∧ height3 = 17 →
  d = (r : ℝ) - Real.sqrt s / (t : ℝ) →
  d = (48 - Real.sqrt 224) / 3 →
  r + s + t = 275 := by
sorry

end NUMINAMATH_CALUDE_cube_distance_to_plane_l3329_332984


namespace NUMINAMATH_CALUDE_dougs_age_l3329_332917

theorem dougs_age (betty_age : ℕ) (doug_age : ℕ) (pack_cost : ℕ) :
  2 * betty_age = pack_cost →
  betty_age + doug_age = 90 →
  20 * pack_cost = 2000 →
  doug_age = 40 := by
sorry

end NUMINAMATH_CALUDE_dougs_age_l3329_332917


namespace NUMINAMATH_CALUDE_conference_season_games_l3329_332916

/-- Calculates the number of games in a complete season for a sports conference. -/
def games_in_season (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let divisions := total_teams / teams_per_division
  let intra_division_total := divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_division_total := (total_teams / 2) * teams_per_division * inter_division_games
  intra_division_total + inter_division_total

/-- Theorem stating the number of games in a complete season for the given conference structure. -/
theorem conference_season_games : 
  games_in_season 14 7 3 1 = 175 := by
  sorry

end NUMINAMATH_CALUDE_conference_season_games_l3329_332916


namespace NUMINAMATH_CALUDE_system_two_solutions_l3329_332905

/-- The system of equations has exactly two solutions if and only if a ∈ {49, 289} -/
theorem system_two_solutions (a : ℝ) : 
  (∃! x y z w : ℝ, 
    (abs (y + x + 8) + abs (y - x + 8) = 16 ∧
     (abs x - 15)^2 + (abs y - 8)^2 = a) ∧
    (abs (z + w + 8) + abs (z - w + 8) = 16 ∧
     (abs w - 15)^2 + (abs z - 8)^2 = a) ∧
    (x ≠ w ∨ y ≠ z)) ↔ 
  (a = 49 ∨ a = 289) :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l3329_332905


namespace NUMINAMATH_CALUDE_marys_average_speed_l3329_332983

/-- Mary's round trip walking problem -/
theorem marys_average_speed (distance_up distance_down : ℝ) (time_up time_down : ℝ) 
  (h1 : distance_up = 1.5)
  (h2 : distance_down = 1.5)
  (h3 : time_up = 45 / 60)
  (h4 : time_down = 15 / 60) :
  (distance_up + distance_down) / (time_up + time_down) = 3 := by
  sorry

end NUMINAMATH_CALUDE_marys_average_speed_l3329_332983


namespace NUMINAMATH_CALUDE_basic_astrophysics_degrees_l3329_332975

/-- Represents the allocation of a research and development budget in a circle graph -/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  industrial_lubricants : ℝ

/-- The theorem stating that the remaining sector (basic astrophysics) occupies 90 degrees of the circle -/
theorem basic_astrophysics_degrees (budget : BudgetAllocation) : 
  budget.microphotonics = 14 ∧ 
  budget.home_electronics = 19 ∧ 
  budget.food_additives = 10 ∧ 
  budget.genetically_modified_microorganisms = 24 ∧ 
  budget.industrial_lubricants = 8 → 
  (100 - (budget.microphotonics + budget.home_electronics + budget.food_additives + 
          budget.genetically_modified_microorganisms + budget.industrial_lubricants)) / 100 * 360 = 90 :=
by sorry

end NUMINAMATH_CALUDE_basic_astrophysics_degrees_l3329_332975


namespace NUMINAMATH_CALUDE_unique_valid_number_l3329_332978

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n / 10 + n % 10 = 9) ∧
  (10 * (n % 10) + (n / 10) = n + 9)

theorem unique_valid_number : ∃! n : ℕ, is_valid_number n :=
  sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3329_332978


namespace NUMINAMATH_CALUDE_original_number_proof_l3329_332948

theorem original_number_proof (final_number : ℝ) (increase_percentage : ℝ) 
  (h1 : final_number = 105)
  (h2 : increase_percentage = 50) : 
  ∃ (original_number : ℝ), original_number * (1 + increase_percentage / 100) = final_number ∧ 
                            original_number = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3329_332948


namespace NUMINAMATH_CALUDE_students_playing_all_sports_l3329_332969

/-- The number of students playing all three sports in a school with given sport participation data -/
theorem students_playing_all_sports (total : ℕ) (football cricket basketball : ℕ) 
  (neither : ℕ) (football_cricket football_basketball cricket_basketball : ℕ) :
  total = 580 →
  football = 300 →
  cricket = 250 →
  basketball = 180 →
  neither = 60 →
  football_cricket = 120 →
  football_basketball = 80 →
  cricket_basketball = 70 →
  ∃ (all_sports : ℕ), 
    all_sports = 140 ∧
    total = football + cricket + basketball - football_cricket - football_basketball - cricket_basketball + all_sports + neither :=
by sorry

end NUMINAMATH_CALUDE_students_playing_all_sports_l3329_332969


namespace NUMINAMATH_CALUDE_unit_circle_trig_values_l3329_332999

theorem unit_circle_trig_values :
  ∀ y : ℝ,
  ((-Real.sqrt 3 / 2) ^ 2 + y ^ 2 = 1) →
  ∃ θ : ℝ,
  (0 < θ ∧ θ < 2 * Real.pi) ∧
  (Real.sin θ = y ∧ Real.cos θ = -Real.sqrt 3 / 2) ∧
  (y = 1 / 2 ∨ y = -1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_unit_circle_trig_values_l3329_332999


namespace NUMINAMATH_CALUDE_probability_same_length_segments_l3329_332955

def regular_pentagon_segments : Finset ℕ := sorry

theorem probability_same_length_segments :
  let S := regular_pentagon_segments
  let total_segments := S.card
  let same_type_segments := (total_segments / 2) - 1
  (same_type_segments : ℚ) / ((total_segments - 1) : ℚ) = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_same_length_segments_l3329_332955


namespace NUMINAMATH_CALUDE_four_term_expression_l3329_332939

theorem four_term_expression (x : ℝ) : 
  ∃ (a b c d : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    (x^3 - 2)^2 + (x^2 + 2*x)^2 = a*x^n₁ + b*x^n₂ + c*x^n₃ + d 
    ∧ n₁ > n₂ ∧ n₂ > n₃ ∧ n₃ > 0
    ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_four_term_expression_l3329_332939


namespace NUMINAMATH_CALUDE_fake_coin_determinable_l3329_332960

/-- Represents the result of a weighing on a two-pan balance scale -/
inductive WeighingResult
  | Left : WeighingResult  -- Left pan is heavier
  | Right : WeighingResult -- Right pan is heavier
  | Equal : WeighingResult -- Pans are balanced

/-- Represents the state of a coin -/
inductive CoinState
  | Normal : CoinState
  | Heavier : CoinState
  | Lighter : CoinState

/-- Represents a weighing on a two-pan balance scale -/
def Weighing := (Fin 25 → Bool) → WeighingResult

/-- Represents the strategy for determining the state of the fake coin -/
def Strategy := Weighing → Weighing → CoinState

/-- Theorem stating that it's possible to determine whether the fake coin
    is lighter or heavier using only two weighings -/
theorem fake_coin_determinable :
  ∃ (s : Strategy),
    ∀ (fake : Fin 25) (state : CoinState),
      state ≠ CoinState.Normal →
        ∀ (w₁ w₂ : Weighing),
          (∀ (f : Fin 25 → Bool),
            w₁ f = WeighingResult.Left ↔ (state = CoinState.Heavier ∧ f fake) ∨
                                         (state = CoinState.Lighter ∧ ¬f fake)) →
          (∀ (f : Fin 25 → Bool),
            w₂ f = WeighingResult.Left ↔ (state = CoinState.Heavier ∧ f fake) ∨
                                         (state = CoinState.Lighter ∧ ¬f fake)) →
          s w₁ w₂ = state :=
sorry

end NUMINAMATH_CALUDE_fake_coin_determinable_l3329_332960


namespace NUMINAMATH_CALUDE_inequalities_for_negative_a_l3329_332945

theorem inequalities_for_negative_a (a b : ℝ) (ha : a < 0) :
  (a < b) ∧ (a^2 + b^2 > 2) ∧ 
  (∃ b, ¬(a + b < a*b)) ∧ (∃ b, ¬(|a| > |b|)) :=
sorry

end NUMINAMATH_CALUDE_inequalities_for_negative_a_l3329_332945


namespace NUMINAMATH_CALUDE_floor_sqrt_26_squared_l3329_332959

theorem floor_sqrt_26_squared : ⌊Real.sqrt 26⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_26_squared_l3329_332959


namespace NUMINAMATH_CALUDE_largest_valid_number_l3329_332935

def is_valid_number (a b c d e : Nat) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
  d > e ∧
  c > d + e ∧
  b > c + d + e ∧
  a > b + c + d + e

def number_value (a b c d e : Nat) : Nat :=
  a * 10000 + b * 1000 + c * 100 + d * 10 + e

theorem largest_valid_number :
  ∀ a b c d e : Nat,
    is_valid_number a b c d e →
    number_value a b c d e ≤ 95210 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3329_332935


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l3329_332982

/-- Given a quadrilateral with one diagonal of 20 cm, one offset of 4 cm, and an area of 90 square cm,
    the length of the other offset is 5 cm. -/
theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 20 →
  offset1 = 4 →
  area = 90 →
  area = (diagonal * (offset1 + 5)) / 2 →
  ∃ (offset2 : ℝ), offset2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_offset_l3329_332982


namespace NUMINAMATH_CALUDE_melody_reading_pages_l3329_332989

def english_pages : ℕ := 20
def science_pages : ℕ := 16
def civics_pages : ℕ := 8
def total_pages_tomorrow : ℕ := 14

def chinese_pages : ℕ := 12

theorem melody_reading_pages : 
  (english_pages / 4 + science_pages / 4 + civics_pages / 4 + chinese_pages / 4 = total_pages_tomorrow) ∧
  (chinese_pages ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_melody_reading_pages_l3329_332989


namespace NUMINAMATH_CALUDE_line_through_point_l3329_332971

/-- Given a line equation ax + (a+4)y = a + 5 passing through (5, -10), prove a = -7.5 -/
theorem line_through_point (a : ℝ) : 
  (∀ x y : ℝ, a * x + (a + 4) * y = a + 5 → x = 5 ∧ y = -10) → 
  a = -7.5 := by sorry

end NUMINAMATH_CALUDE_line_through_point_l3329_332971


namespace NUMINAMATH_CALUDE_sine_shifted_is_even_l3329_332931

/-- A function that reaches its maximum at x = 1 -/
def reaches_max_at_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f x ≤ f 1

/-- Definition of an even function -/
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Main theorem -/
theorem sine_shifted_is_even
    (A ω φ : ℝ)
    (hA : A > 0)
    (hω : ω > 0)
    (h_max : reaches_max_at_one (fun x ↦ A * Real.sin (ω * x + φ))) :
    is_even (fun x ↦ A * Real.sin (ω * (x + 1) + φ)) := by
  sorry

end NUMINAMATH_CALUDE_sine_shifted_is_even_l3329_332931


namespace NUMINAMATH_CALUDE_corn_preference_result_l3329_332952

/-- The percentage of children preferring corn in Carolyn's daycare -/
def corn_preference_percentage (total_children : ℕ) (corn_preference : ℕ) : ℚ :=
  (corn_preference : ℚ) / (total_children : ℚ) * 100

/-- Theorem stating that the percentage of children preferring corn is 17.5% -/
theorem corn_preference_result : 
  corn_preference_percentage 40 7 = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_corn_preference_result_l3329_332952


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3329_332920

theorem unique_solution_for_equation : 
  ∃! (x y : ℕ+), 2 * (x : ℕ) ^ (y : ℕ) - (y : ℕ) = 2005 ∧ x = 1003 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3329_332920


namespace NUMINAMATH_CALUDE_number_333_less_than_600_l3329_332957

theorem number_333_less_than_600 : 600 - 333 = 267 := by sorry

end NUMINAMATH_CALUDE_number_333_less_than_600_l3329_332957


namespace NUMINAMATH_CALUDE_distance_satisfies_conditions_l3329_332970

/-- The distance traveled by both the train and the ship -/
def distance : ℝ := 480

/-- The speed of the train in km/h -/
def train_speed : ℝ := 48

/-- The speed of the ship in km/h -/
def ship_speed : ℝ := 60

/-- The time difference between the train and ship journeys in hours -/
def time_difference : ℝ := 2

/-- Theorem stating that the given distance satisfies the problem conditions -/
theorem distance_satisfies_conditions :
  distance / train_speed = distance / ship_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_distance_satisfies_conditions_l3329_332970


namespace NUMINAMATH_CALUDE_gary_earnings_l3329_332951

def total_flour : ℚ := 6
def flour_for_cakes : ℚ := 4
def flour_per_cake : ℚ := 1/2
def flour_for_cupcakes : ℚ := 2
def flour_per_cupcake : ℚ := 1/5
def price_per_cake : ℚ := 5/2
def price_per_cupcake : ℚ := 1

def num_cakes : ℚ := flour_for_cakes / flour_per_cake
def num_cupcakes : ℚ := flour_for_cupcakes / flour_per_cupcake

def earnings_from_cakes : ℚ := num_cakes * price_per_cake
def earnings_from_cupcakes : ℚ := num_cupcakes * price_per_cupcake

theorem gary_earnings :
  earnings_from_cakes + earnings_from_cupcakes = 30 :=
by sorry

end NUMINAMATH_CALUDE_gary_earnings_l3329_332951


namespace NUMINAMATH_CALUDE_truck_speed_truck_speed_is_52_l3329_332963

/-- The speed of the truck given two cars with different speeds meeting it at different times --/
theorem truck_speed (speed_A speed_B : ℝ) (time_A time_B : ℝ) : ℝ :=
  let distance_A := speed_A * time_A
  let distance_B := speed_B * time_B
  (distance_A - distance_B) / (time_B - time_A)

/-- Proof that the truck's speed is 52 km/h given the problem conditions --/
theorem truck_speed_is_52 :
  truck_speed 102 80 6 7 = 52 := by
  sorry

end NUMINAMATH_CALUDE_truck_speed_truck_speed_is_52_l3329_332963


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_a_minus_b_eq_one_l3329_332981

/-- The polynomial in question -/
def P (a b x y : ℝ) : ℝ := x^2 + a*x*y + b*y^2 - 5*x + y + 6

/-- The factor of the polynomial -/
def F (x y : ℝ) : ℝ := x + y - 2

theorem polynomial_factor_implies_a_minus_b_eq_one (a b : ℝ) :
  (∀ x y : ℝ, ∃ k : ℝ, P a b x y = F x y * k) →
  a - b = 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_a_minus_b_eq_one_l3329_332981


namespace NUMINAMATH_CALUDE_smallest_three_digit_with_digit_product_8_l3329_332953

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem smallest_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 118 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_with_digit_product_8_l3329_332953


namespace NUMINAMATH_CALUDE_sqrt_112_between_consecutive_integers_l3329_332926

theorem sqrt_112_between_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ n^2 < 112 ∧ (n + 1)^2 > 112 ∧ n * (n + 1) = 110 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_112_between_consecutive_integers_l3329_332926


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l3329_332993

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 7 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l3329_332993


namespace NUMINAMATH_CALUDE_arctan_sum_right_triangle_l3329_332936

theorem arctan_sum_right_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a = 2 * b) :
  Real.arctan (b / a) + Real.arctan (a / b) = π / 2 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_right_triangle_l3329_332936


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fifths_l3329_332973

/-- A fraction with two-digit numerator and denominator -/
structure TwoDigitFraction where
  numerator : ℕ
  denominator : ℕ
  num_two_digit : 10 ≤ numerator ∧ numerator ≤ 99
  den_two_digit : 10 ≤ denominator ∧ denominator ≤ 99

/-- The property of being greater than 3/5 -/
def greater_than_three_fifths (f : TwoDigitFraction) : Prop :=
  (f.numerator : ℚ) / f.denominator > 3 / 5

/-- The theorem stating that 59/98 is the smallest fraction greater than 3/5 with two-digit numerator and denominator -/
theorem smallest_fraction_greater_than_three_fifths :
  ∀ f : TwoDigitFraction, greater_than_three_fifths f →
    (59 : ℚ) / 98 ≤ (f.numerator : ℚ) / f.denominator :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fifths_l3329_332973


namespace NUMINAMATH_CALUDE_timothy_read_300_pages_l3329_332998

/-- The total number of pages Timothy read in a week -/
def total_pages_read : ℕ :=
  let monday_tuesday := 2 * 45
  let wednesday := 50
  let thursday_to_saturday := 3 * 40
  let sunday := 25 + 15
  monday_tuesday + wednesday + thursday_to_saturday + sunday

/-- Theorem stating that Timothy read 300 pages in total -/
theorem timothy_read_300_pages : total_pages_read = 300 := by
  sorry

end NUMINAMATH_CALUDE_timothy_read_300_pages_l3329_332998


namespace NUMINAMATH_CALUDE_sticker_pages_l3329_332908

theorem sticker_pages (stickers_per_page : ℕ) (total_stickers : ℕ) (pages : ℕ) : 
  stickers_per_page = 10 →
  total_stickers = 220 →
  pages * stickers_per_page = total_stickers →
  pages = 22 := by
sorry

end NUMINAMATH_CALUDE_sticker_pages_l3329_332908


namespace NUMINAMATH_CALUDE_function_sum_at_one_l3329_332932

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem function_sum_at_one 
  (h1 : is_even f) 
  (h2 : is_odd g) 
  (h3 : ∀ x, f x - g x = x^3 + x^2 + 1) : 
  f 1 + g 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_function_sum_at_one_l3329_332932


namespace NUMINAMATH_CALUDE_cindy_marbles_l3329_332921

/-- Given Cindy's initial marbles and distribution to friends, calculate five times her remaining marbles -/
theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ)
  (h1 : initial_marbles = 800)
  (h2 : friends = 6)
  (h3 : marbles_per_friend = 120) :
  5 * (initial_marbles - friends * marbles_per_friend) = 400 := by
  sorry

end NUMINAMATH_CALUDE_cindy_marbles_l3329_332921


namespace NUMINAMATH_CALUDE_no_integer_solution_for_equation_l3329_332918

theorem no_integer_solution_for_equation :
  ∀ x y : ℤ, x^2 - 3*y^2 ≠ 17 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_equation_l3329_332918


namespace NUMINAMATH_CALUDE_cos_five_pi_sixth_plus_alpha_l3329_332965

theorem cos_five_pi_sixth_plus_alpha (α : ℝ) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) = -(Real.sqrt 3 / 3) := by
sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixth_plus_alpha_l3329_332965


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3329_332962

/-- An equilateral triangle is a triangle with all sides of equal length -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The perimeter of a triangle is the sum of its side lengths -/
def perimeter (triangle : EquilateralTriangle) : ℝ :=
  3 * triangle.side_length

/-- Theorem: The perimeter of an equilateral triangle with side length 'a' is 3a -/
theorem equilateral_triangle_perimeter (a : ℝ) (ha : a > 0) :
  let triangle : EquilateralTriangle := ⟨a, ha⟩
  perimeter triangle = 3 * a := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3329_332962


namespace NUMINAMATH_CALUDE_pigeonhole_sum_to_ten_l3329_332974

theorem pigeonhole_sum_to_ten :
  ∀ (S : Finset ℕ), 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 10) → 
    S.card ≥ 7 → 
    ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = 10 :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_sum_to_ten_l3329_332974


namespace NUMINAMATH_CALUDE_division_chain_l3329_332934

theorem division_chain : (180 / 6) / 3 = 10 := by sorry

end NUMINAMATH_CALUDE_division_chain_l3329_332934


namespace NUMINAMATH_CALUDE_average_income_problem_l3329_332994

/-- Given the average incomes of different pairs of individuals and the income of one individual,
    prove that the average income of a specific pair is as stated. -/
theorem average_income_problem (M N O : ℕ) : 
  (M + N) / 2 = 5050 →
  (M + O) / 2 = 5200 →
  M = 4000 →
  (N + O) / 2 = 6250 := by
sorry

end NUMINAMATH_CALUDE_average_income_problem_l3329_332994


namespace NUMINAMATH_CALUDE_dice_probability_l3329_332988

/-- The number of sides on each die -/
def n : ℕ := 4025

/-- The threshold for the first die -/
def k : ℕ := 2012

/-- The probability that the first die is less than or equal to k,
    given that it's greater than or equal to the second die -/
def prob : ℚ :=
  (k * (k + 1)) / (n * (n + 1))

theorem dice_probability :
  prob = 1006 / 4025 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l3329_332988


namespace NUMINAMATH_CALUDE_f_extremum_range_l3329_332909

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - a * x^2 + x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * a * x + 1

-- Define the condition for exactly one extremum point in (-1, 0)
def has_one_extremum (a : ℝ) : Prop :=
  ∃! x, x ∈ Set.Ioo (-1) 0 ∧ f' a x = 0

-- State the theorem
theorem f_extremum_range :
  ∀ a : ℝ, has_one_extremum a ↔ a ∈ Set.Ioi (-1/5) ∪ {-1} :=
sorry

end NUMINAMATH_CALUDE_f_extremum_range_l3329_332909


namespace NUMINAMATH_CALUDE_matrix_multiplication_l3329_332922

theorem matrix_multiplication (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec ![3, -2] = ![4, 1])
  (h2 : N.mulVec ![-4, 6] = ![-2, 0]) :
  N.mulVec ![7, 0] = ![14, 4.2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_l3329_332922


namespace NUMINAMATH_CALUDE_smallest_c_value_l3329_332924

/-- Given a cosine function y = a cos(bx + c) with positive constants a, b, c,
    and maximum at x = 1, the smallest possible value of c is 0. -/
theorem smallest_c_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : ∀ x : ℝ, a * Real.cos (b * x + c) ≤ a * Real.cos (b * 1 + c)) :
    ∃ c' : ℝ, c' ≥ 0 ∧ c' ≤ c ∧ ∀ c'' : ℝ, c'' ≥ 0 → c'' ≤ c → c' ≤ c'' := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l3329_332924


namespace NUMINAMATH_CALUDE_fourth_term_binomial_expansion_l3329_332941

theorem fourth_term_binomial_expansion 
  (a x : ℝ) (ha : a ≠ 0) (hx : x > 0) :
  let binomial := (2*a/Real.sqrt x - Real.sqrt x/(2*a^2))^8
  let fourth_term := (Nat.choose 8 3) * (2*a/Real.sqrt x)^5 * (-Real.sqrt x/(2*a^2))^3
  fourth_term = -4/(a*x) :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_binomial_expansion_l3329_332941


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_143_l3329_332954

def sum_of_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_prime_factors_143 : sum_of_prime_factors 143 = 24 := by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_143_l3329_332954


namespace NUMINAMATH_CALUDE_complex_number_equality_l3329_332972

theorem complex_number_equality : (((1 + Complex.I)^4) / (1 - Complex.I)) + 2 = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3329_332972


namespace NUMINAMATH_CALUDE_bagel_cut_theorem_l3329_332903

/-- Number of pieces resulting from cutting a torus-shaped object -/
def torusPieces (cuts : ℕ) : ℕ := cuts + 1

/-- Theorem: Cutting a torus-shaped object (bagel) with 10 cuts results in 11 pieces -/
theorem bagel_cut_theorem :
  torusPieces 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bagel_cut_theorem_l3329_332903


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3329_332987

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x - m = 0 ∧ y^2 - 2*y - m = 0) → m ≥ -1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3329_332987


namespace NUMINAMATH_CALUDE_negative_300_coterminal_with_60_l3329_332961

/-- Two angles are coterminal if their difference is a multiple of 360° -/
def coterminal (a b : ℝ) : Prop := ∃ k : ℤ, a - b = 360 * k

/-- The theorem states that -300° is coterminal with 60° -/
theorem negative_300_coterminal_with_60 : coterminal (-300 : ℝ) 60 := by
  sorry

end NUMINAMATH_CALUDE_negative_300_coterminal_with_60_l3329_332961


namespace NUMINAMATH_CALUDE_power_equation_solution_l3329_332912

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^26 → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3329_332912


namespace NUMINAMATH_CALUDE_wall_volume_l3329_332966

/-- The volume of a rectangular wall with specific proportions -/
theorem wall_volume (width : ℝ) (height : ℝ) (length : ℝ) : 
  width = 4 → 
  height = 6 * width → 
  length = 7 * height → 
  width * height * length = 16128 := by
sorry

end NUMINAMATH_CALUDE_wall_volume_l3329_332966


namespace NUMINAMATH_CALUDE_evaluate_expression_l3329_332913

theorem evaluate_expression : (10010 - 12 * 3) * 2 ^ 3 = 79792 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3329_332913


namespace NUMINAMATH_CALUDE_candy_distribution_l3329_332991

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 858 →
  num_bags = 26 →
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 33 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3329_332991


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l3329_332995

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l3329_332995


namespace NUMINAMATH_CALUDE_debby_dvds_left_l3329_332902

/-- The number of DVDs Debby has left after selling some -/
def dvds_left (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

/-- Theorem: If Debby owned thirteen DVDs and sold six of them, she would have 7 DVDs left -/
theorem debby_dvds_left : dvds_left 13 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_debby_dvds_left_l3329_332902


namespace NUMINAMATH_CALUDE_sum_remainder_mod_9_l3329_332968

theorem sum_remainder_mod_9 : (9237 + 9238 + 9239 + 9240 + 9241) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_9_l3329_332968


namespace NUMINAMATH_CALUDE_ariel_age_l3329_332911

/-- Ariel's present age in years -/
def present_age : ℕ := 5

/-- The number of years in the future -/
def years_future : ℕ := 15

/-- Theorem stating that Ariel's present age is 5, given the condition -/
theorem ariel_age : 
  (present_age + years_future = 4 * present_age) → present_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_ariel_age_l3329_332911


namespace NUMINAMATH_CALUDE_P_inf_zero_no_minimum_l3329_332947

/-- The function P from ℝ² to ℝ -/
def P : ℝ × ℝ → ℝ := fun (x₁, x₂) ↦ x₁^2 + (1 - x₁ * x₂)^2

theorem P_inf_zero_no_minimum :
  (∀ ε > 0, ∃ x : ℝ × ℝ, P x < ε) ∧
  ¬∃ x : ℝ × ℝ, ∀ y : ℝ × ℝ, P x ≤ P y :=
by sorry

end NUMINAMATH_CALUDE_P_inf_zero_no_minimum_l3329_332947


namespace NUMINAMATH_CALUDE_min_value_polynomial_min_value_achieved_l3329_332964

theorem min_value_polynomial (x : ℝ) : (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 ≥ 2034 := by
  sorry

theorem min_value_achieved : ∃ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2050 = 2034 := by
  sorry

end NUMINAMATH_CALUDE_min_value_polynomial_min_value_achieved_l3329_332964


namespace NUMINAMATH_CALUDE_no_harmonic_point_reciprocal_unique_harmonic_point_range_of_m_l3329_332914

-- Definition of a harmonic point
def is_harmonic_point (x y : ℝ) : Prop := x = y

-- Part 1: No harmonic point for y = -4/x
theorem no_harmonic_point_reciprocal : ¬∃ x : ℝ, x ≠ 0 ∧ is_harmonic_point x (-4/x) := by sorry

-- Part 2: Quadratic function with one harmonic point
theorem unique_harmonic_point (a c : ℝ) :
  a ≠ 0 ∧ 
  (∃! x : ℝ, is_harmonic_point x (a * x^2 + 6 * x + c)) ∧
  is_harmonic_point (5/2) (a * (5/2)^2 + 6 * (5/2) + c) →
  a = -1 ∧ c = -25/4 := by sorry

-- Part 3: Range of m for quadratic function with given min and max
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 ≥ -1) ∧
  (∀ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 ≤ 3) ∧
  (∃ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 = -1) ∧
  (∃ x ∈ Set.Icc 1 m, -x^2 + 6*x - 6 = 3) →
  3 ≤ m ∧ m ≤ 5 := by sorry

end NUMINAMATH_CALUDE_no_harmonic_point_reciprocal_unique_harmonic_point_range_of_m_l3329_332914


namespace NUMINAMATH_CALUDE_min_value_theorem_l3329_332996

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b)/c + (2*a + c)/b + (2*b + c)/a ≥ 6 ∧
  ((2*a + b)/c + (2*a + c)/b + (2*b + c)/a = 6 ↔ 2*a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3329_332996


namespace NUMINAMATH_CALUDE_johns_leisure_travel_l3329_332946

/-- Calculates the leisure travel distance for John given his car's efficiency,
    work commute details, and total gas consumption. -/
theorem johns_leisure_travel
  (efficiency : ℝ)  -- Car efficiency in miles per gallon
  (work_distance : ℝ)  -- One-way distance to work in miles
  (work_days : ℕ)  -- Number of work days per week
  (total_gas : ℝ)  -- Total gas used per week in gallons
  (h1 : efficiency = 30)  -- Car efficiency is 30 mpg
  (h2 : work_distance = 20)  -- Distance to work is 20 miles each way
  (h3 : work_days = 5)  -- Works 5 days a week
  (h4 : total_gas = 8)  -- Uses 8 gallons of gas per week
  : ℝ :=
  total_gas * efficiency - 2 * work_distance * work_days

#check johns_leisure_travel

end NUMINAMATH_CALUDE_johns_leisure_travel_l3329_332946


namespace NUMINAMATH_CALUDE_coin_probability_theorem_l3329_332930

theorem coin_probability_theorem (p q : ℝ) : 
  p + q = 1 →
  0 ≤ p ∧ p ≤ 1 →
  0 ≤ q ∧ q ≤ 1 →
  (Nat.choose 10 5 : ℝ) * p^5 * q^5 = (Nat.choose 10 6 : ℝ) * p^6 * q^4 →
  p = 6/11 :=
by sorry

end NUMINAMATH_CALUDE_coin_probability_theorem_l3329_332930


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3329_332956

open Real

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  3 * cos θ + 1 / sin θ + 2 * tan θ ≥ 3 * Real.rpow 6 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3329_332956


namespace NUMINAMATH_CALUDE_strawberry_picking_problem_l3329_332933

/-- Calculates the number of pounds of strawberries picked given the total paid, 
    number of pickers, entrance fee per person, and price per pound of strawberries -/
def strawberries_picked (total_paid : ℚ) (num_pickers : ℕ) (entrance_fee : ℚ) (price_per_pound : ℚ) : ℚ :=
  (total_paid + num_pickers * entrance_fee) / price_per_pound

/-- Theorem stating that under the given conditions, the number of pounds of strawberries picked is 7 -/
theorem strawberry_picking_problem :
  let total_paid : ℚ := 128
  let num_pickers : ℕ := 3
  let entrance_fee : ℚ := 4
  let price_per_pound : ℚ := 20
  strawberries_picked total_paid num_pickers entrance_fee price_per_pound = 7 := by
  sorry


end NUMINAMATH_CALUDE_strawberry_picking_problem_l3329_332933


namespace NUMINAMATH_CALUDE_ratio_of_divisors_sums_l3329_332919

def M : ℕ := 75 * 75 * 140 * 343

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisors_sums_l3329_332919


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3329_332992

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧ 
  n % 8 = 5 ∧ 
  (∀ (m : ℕ), m > 20 ∧ m % 6 = 4 ∧ m % 7 = 3 ∧ m % 8 = 5 → m ≥ n) ∧
  n = 136 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3329_332992


namespace NUMINAMATH_CALUDE_sum_200th_row_l3329_332929

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ := sorry

/-- The triangular array has the following properties:
    1. The sides contain numbers 0, 1, 2, 3, ...
    2. Each interior number is the sum of two adjacent numbers in the previous row -/
axiom array_properties : True

/-- The sum of numbers in the nth row follows the recurrence relation:
    f(n) = 2 * f(n-1) + 2 for n ≥ 2 -/
axiom recurrence_relation (n : ℕ) (h : n ≥ 2) : f n = 2 * f (n-1) + 2

/-- The sum of numbers in the 200th row of the triangular array is 2^200 - 2 -/
theorem sum_200th_row : f 200 = 2^200 - 2 := by sorry

end NUMINAMATH_CALUDE_sum_200th_row_l3329_332929


namespace NUMINAMATH_CALUDE_all_gp_lines_pass_through_origin_l3329_332915

/-- A line in 2D space defined by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three real numbers form a geometric progression -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The set of all lines where a, b, c form a geometric progression -/
def GPLines : Set Line :=
  {l : Line | isGeometricProgression l.a l.b l.c}

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

theorem all_gp_lines_pass_through_origin :
  ∀ l ∈ GPLines, pointOnLine ⟨0, 0⟩ l :=
sorry

end NUMINAMATH_CALUDE_all_gp_lines_pass_through_origin_l3329_332915


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3329_332985

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3329_332985


namespace NUMINAMATH_CALUDE_largest_increase_1993_l3329_332942

/-- Profit margin percentages for each year from 1990 to 1999 -/
def profitMargins : Fin 10 → ℝ
  | 0 => 10
  | 1 => 20
  | 2 => 30
  | 3 => 60
  | 4 => 70
  | 5 => 75
  | 6 => 80
  | 7 => 82
  | 8 => 86
  | 9 => 70

/-- Calculate the percentage increase between two years -/
def percentageIncrease (year1 year2 : Fin 10) : ℝ :=
  profitMargins year2 - profitMargins year1

/-- The year with the largest percentage increase -/
def yearWithLargestIncrease : Fin 10 :=
  3  -- Representing 1993 (index 3 corresponds to 1993)

/-- Theorem stating that 1993 (index 3) has the largest percentage increase -/
theorem largest_increase_1993 :
  ∀ (year : Fin 9), percentageIncrease year (year + 1) ≤ percentageIncrease 2 3 :=
sorry

end NUMINAMATH_CALUDE_largest_increase_1993_l3329_332942


namespace NUMINAMATH_CALUDE_proportion_solution_l3329_332925

theorem proportion_solution (n : ℝ) : n / 1.2 = 5 / 8 → n = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3329_332925


namespace NUMINAMATH_CALUDE_square_of_prime_quadratic_l3329_332927

def is_square_of_prime (x : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ x = p^2

theorem square_of_prime_quadratic :
  ∀ n : ℕ, (is_square_of_prime (2*n^2 + 3*n - 35)) ↔ (n = 4 ∨ n = 12) :=
sorry

end NUMINAMATH_CALUDE_square_of_prime_quadratic_l3329_332927


namespace NUMINAMATH_CALUDE_remainder_of_product_mod_12_l3329_332990

theorem remainder_of_product_mod_12 : (1425 * 1427 * 1429) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_mod_12_l3329_332990


namespace NUMINAMATH_CALUDE_square_root_difference_product_l3329_332928

theorem square_root_difference_product : (Real.sqrt 100 + Real.sqrt 9) * (Real.sqrt 100 - Real.sqrt 9) = 91 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_product_l3329_332928


namespace NUMINAMATH_CALUDE_total_blisters_l3329_332967

/-- Given a person with 60 blisters on each arm and 80 blisters on the rest of their body,
    the total number of blisters is 200. -/
theorem total_blisters (blisters_per_arm : ℕ) (blisters_rest : ℕ) :
  blisters_per_arm = 60 →
  blisters_rest = 80 →
  blisters_per_arm * 2 + blisters_rest = 200 :=
by sorry

end NUMINAMATH_CALUDE_total_blisters_l3329_332967


namespace NUMINAMATH_CALUDE_intersecting_plane_theorem_l3329_332944

/-- Represents a 3D cube composed of unit cubes -/
structure Cube where
  side_length : ℕ
  total_units : ℕ

/-- Represents a plane intersecting the cube -/
structure IntersectingPlane where
  perpendicular_to_diagonal : Bool
  distance_ratio : ℚ

/-- Calculates the number of unit cubes intersected by the plane -/
def intersected_cubes (c : Cube) (p : IntersectingPlane) : ℕ :=
  sorry

/-- Theorem stating that a plane intersecting a 4x4x4 cube at 1/4 of its diagonal intersects 36 unit cubes -/
theorem intersecting_plane_theorem (c : Cube) (p : IntersectingPlane) :
  c.side_length = 4 ∧ c.total_units = 64 ∧ p.perpendicular_to_diagonal = true ∧ p.distance_ratio = 1/4 →
  intersected_cubes c p = 36 :=
sorry

end NUMINAMATH_CALUDE_intersecting_plane_theorem_l3329_332944


namespace NUMINAMATH_CALUDE_jack_sugar_today_l3329_332980

/-- The amount of sugar Jack has today -/
def S : ℕ := by sorry

/-- Theorem: Jack has 65 pounds of sugar today -/
theorem jack_sugar_today : S = 65 := by
  have h1 : S - 18 + 50 = 97 := by sorry
  sorry


end NUMINAMATH_CALUDE_jack_sugar_today_l3329_332980


namespace NUMINAMATH_CALUDE_movie_choices_eq_nine_l3329_332901

/-- The number of ways two people can choose one movie each from three movies -/
def movie_choices : ℕ :=
  let number_of_people : ℕ := 2
  let number_of_movies : ℕ := 3
  number_of_movies ^ number_of_people

/-- Theorem stating that the number of ways two people can choose one movie each from three movies is 9 -/
theorem movie_choices_eq_nine : movie_choices = 9 := by
  sorry

end NUMINAMATH_CALUDE_movie_choices_eq_nine_l3329_332901


namespace NUMINAMATH_CALUDE_range_of_m_l3329_332977

def p (x : ℝ) : Prop := |x - 4| ≤ 6

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m :
  (∀ x m : ℝ, (¬(p x) → ¬(q x m)) ∧ ∃ x, ¬(p x) ∧ (q x m)) →
  ∀ m : ℝ, -3 ≤ m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3329_332977


namespace NUMINAMATH_CALUDE_gcd_plus_binary_sum_l3329_332906

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem gcd_plus_binary_sum : 
  let a := Nat.gcd 98 63
  let b := binary_to_decimal [true, true, false, false, true, true]
  a + b = 58 := by sorry

end NUMINAMATH_CALUDE_gcd_plus_binary_sum_l3329_332906


namespace NUMINAMATH_CALUDE_jerry_recycling_time_l3329_332986

/-- The time it takes Jerry to throw away all the cans -/
def total_time (num_cans : ℕ) (cans_per_trip : ℕ) (drain_time : ℕ) (walk_time : ℕ) : ℕ :=
  let num_trips := (num_cans + cans_per_trip - 1) / cans_per_trip
  let round_trip_time := 2 * walk_time
  let time_per_trip := round_trip_time + drain_time
  num_trips * time_per_trip

/-- Theorem stating that under the given conditions, it takes Jerry 350 seconds to throw away all the cans -/
theorem jerry_recycling_time :
  total_time 28 4 30 10 = 350 := by
  sorry

end NUMINAMATH_CALUDE_jerry_recycling_time_l3329_332986


namespace NUMINAMATH_CALUDE_ellipse_intersection_equidistant_point_range_l3329_332904

/-- Ellipse G with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h : a > 0
  i : b > 0
  j : a > b
  k : e = Real.sqrt 3 / 3
  l : a = Real.sqrt 3

/-- Line intersecting the ellipse -/
structure IntersectingLine where
  k : ℝ
  m : ℝ := 1

/-- Point on x-axis equidistant from intersection points -/
structure EquidistantPoint where
  x : ℝ

/-- Main theorem -/
theorem ellipse_intersection_equidistant_point_range
  (G : Ellipse)
  (l : IntersectingLine)
  (M : EquidistantPoint) :
  (∃ A B : ℝ × ℝ,
    (A.1^2 / G.a^2 + A.2^2 / G.b^2 = 1) ∧
    (B.1^2 / G.a^2 + B.2^2 / G.b^2 = 1) ∧
    (A.2 = l.k * A.1 + l.m) ∧
    (B.2 = l.k * B.1 + l.m) ∧
    ((A.1 - M.x)^2 + A.2^2 = (B.1 - M.x)^2 + B.2^2) ∧
    (M.x ≠ A.1) ∧ (M.x ≠ B.1)) →
  -Real.sqrt 6 / 12 ≤ M.x ∧ M.x ≤ Real.sqrt 6 / 12 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_equidistant_point_range_l3329_332904


namespace NUMINAMATH_CALUDE_inequality_proof_l3329_332997

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  Real.sqrt (16 * a^2 + 9) + Real.sqrt (16 * b^2 + 9) + Real.sqrt (16 * c^2 + 9) ≥ 3 + 4 * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3329_332997


namespace NUMINAMATH_CALUDE_amaya_viewing_time_l3329_332900

/-- Represents the total time Amaya spent watching the movie -/
def total_viewing_time (
  segment1 segment2 segment3 segment4 segment5 : ℕ
) (rewind1 rewind2 rewind3 rewind4 : ℕ) : ℕ :=
  segment1 + segment2 + segment3 + segment4 + segment5 +
  rewind1 + rewind2 + rewind3 + rewind4

/-- Theorem stating that the total viewing time is 170 minutes -/
theorem amaya_viewing_time :
  total_viewing_time 35 45 25 15 20 5 7 10 8 = 170 := by
  sorry

end NUMINAMATH_CALUDE_amaya_viewing_time_l3329_332900


namespace NUMINAMATH_CALUDE_min_four_digit_with_different_remainders_l3329_332958

theorem min_four_digit_with_different_remainders :
  ∃ (n : ℕ),
    1000 ≤ n ∧ n ≤ 9999 ∧
    (∀ i j, i ≠ j → n % (i + 2) ≠ n % (j + 2)) ∧
    (∀ i, n % (i + 2) ≠ 0) ∧
    (∀ m, 1000 ≤ m ∧ m < n →
      ¬(∀ i j, i ≠ j → m % (i + 2) ≠ m % (j + 2)) ∨
      ¬(∀ i, m % (i + 2) ≠ 0)) ∧
    n = 1259 :=
by sorry

end NUMINAMATH_CALUDE_min_four_digit_with_different_remainders_l3329_332958


namespace NUMINAMATH_CALUDE_square_area_increase_l3329_332949

theorem square_area_increase (s : ℝ) (h1 : s^2 = 256) (h2 : s > 0) :
  (s + 2)^2 - s^2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l3329_332949


namespace NUMINAMATH_CALUDE_shirts_to_wash_l3329_332943

theorem shirts_to_wash (total_shirts : ℕ) (rewash_shirts : ℕ) (correctly_washed : ℕ) : 
  total_shirts = 63 → rewash_shirts = 12 → correctly_washed = 29 →
  total_shirts - correctly_washed + rewash_shirts = 46 := by
  sorry

end NUMINAMATH_CALUDE_shirts_to_wash_l3329_332943


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l3329_332937

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l3329_332937


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l3329_332950

theorem cloth_sale_calculation (total_selling_price : ℝ) (profit_per_meter : ℝ) (cost_price_per_meter : ℝ)
  (h1 : total_selling_price = 9890)
  (h2 : profit_per_meter = 24)
  (h3 : cost_price_per_meter = 83.5) :
  (total_selling_price / (cost_price_per_meter + profit_per_meter)) = 92 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_calculation_l3329_332950


namespace NUMINAMATH_CALUDE_correlation_identification_l3329_332976

-- Define the relationships
def age_wealth_relation : Type := Unit
def point_coordinates_relation : Type := Unit
def apple_climate_relation : Type := Unit
def tree_diameter_height_relation : Type := Unit

-- Define the concept of correlation
def has_correlation (relation : Type) : Prop := sorry

-- Define the concept of deterministic relationship
def is_deterministic (relation : Type) : Prop := sorry

-- Theorem statement
theorem correlation_identification :
  (has_correlation age_wealth_relation) ∧
  (has_correlation apple_climate_relation) ∧
  (has_correlation tree_diameter_height_relation) ∧
  (is_deterministic point_coordinates_relation) ∧
  (¬ has_correlation point_coordinates_relation) := by sorry

end NUMINAMATH_CALUDE_correlation_identification_l3329_332976


namespace NUMINAMATH_CALUDE_ariella_interest_rate_l3329_332979

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem ariella_interest_rate :
  ∀ (daniella_initial ariella_initial ariella_final : ℝ),
  daniella_initial = 400 →
  ariella_initial = daniella_initial + 200 →
  ariella_final = 720 →
  ∃ (rate : ℝ), 
    simple_interest ariella_initial rate 2 = ariella_final ∧
    rate = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ariella_interest_rate_l3329_332979
