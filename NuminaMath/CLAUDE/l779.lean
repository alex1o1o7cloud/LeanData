import Mathlib

namespace NUMINAMATH_CALUDE_triangle_abc_proof_l779_77956

theorem triangle_abc_proof (a b c : ℝ) (A B : ℝ) :
  a = 2 * Real.sqrt 3 →
  c = Real.sqrt 6 + Real.sqrt 2 →
  B = 45 * (π / 180) →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) →
  b = 2 * Real.sqrt 2 ∧ A = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_proof_l779_77956


namespace NUMINAMATH_CALUDE_square_ratios_area_ratio_diagonal_ratio_l779_77958

/-- Given two squares where the perimeter of one is 4 times the other, 
    prove the ratios of their areas and diagonals -/
theorem square_ratios (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : 
  a ^ 2 = 16 * b ^ 2 ∧ a * Real.sqrt 2 = 4 * (b * Real.sqrt 2) := by
  sorry

/-- The area of the larger square is 16 times the area of the smaller square -/
theorem area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : 
  a ^ 2 = 16 * b ^ 2 := by
  sorry

/-- The diagonal of the larger square is 4 times the diagonal of the smaller square -/
theorem diagonal_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : 
  a * Real.sqrt 2 = 4 * (b * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_square_ratios_area_ratio_diagonal_ratio_l779_77958


namespace NUMINAMATH_CALUDE_bills_milk_problem_l779_77908

/-- Represents the problem of determining the amount of milk Bill got from his cow --/
theorem bills_milk_problem (M : ℝ) : 
  M > 0 ∧ 
  (M / 16) * 5 + (M / 8) * 6 + (M / 2) * 3 = 41 → 
  M = 16 :=
by sorry

end NUMINAMATH_CALUDE_bills_milk_problem_l779_77908


namespace NUMINAMATH_CALUDE_polynomial_positive_reals_l779_77931

def P (x y : ℝ) : ℝ := x^2 + (x*y + 1)^2

theorem polynomial_positive_reals :
  (∀ x y : ℝ, P x y > 0) ∧
  (∀ c : ℝ, c > 0 → ∃ x y : ℝ, P x y = c) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_positive_reals_l779_77931


namespace NUMINAMATH_CALUDE_brick_width_calculation_l779_77913

theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 18 →
  courtyard_width = 12 →
  brick_length = 0.12 →
  total_bricks = 30000 →
  ∃ (brick_width : ℝ),
    brick_width = 0.06 ∧
    courtyard_length * courtyard_width * 100 * 100 = total_bricks * brick_length * brick_width * 10000 :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l779_77913


namespace NUMINAMATH_CALUDE_n_equals_ten_l779_77906

/-- The number of sides in a regular polygon satisfying the given condition -/
def n : ℕ := sorry

/-- The measure of the internal angle in a regular polygon with k sides -/
def internal_angle (k : ℕ) : ℚ := (k - 2) * 180 / k

/-- The condition that the internal angle of an n-sided polygon is 12° less
    than that of a polygon with n/4 fewer sides -/
axiom angle_condition : internal_angle n = internal_angle (3 * n / 4) - 12

/-- Theorem stating that n = 10 -/
theorem n_equals_ten : n = 10 := by sorry

end NUMINAMATH_CALUDE_n_equals_ten_l779_77906


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l779_77916

/-- Given a boat that travels 13 km along a stream and 5 km against the same stream
    in one hour each, its speed in still water is 9 km/hr. -/
theorem boat_speed_in_still_water
  (along_stream : ℝ) (against_stream : ℝ)
  (h_along : along_stream = 13)
  (h_against : against_stream = 5)
  (h_time : along_stream = (boat_speed + stream_speed) * 1 ∧
            against_stream = (boat_speed - stream_speed) * 1)
  : boat_speed = 9 :=
by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l779_77916


namespace NUMINAMATH_CALUDE_linear_function_proof_l779_77982

/-- A linear function y = kx + 3 passing through (1, -2) with negative slope -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 3

theorem linear_function_proof (k : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → linear_function k x₁ > linear_function k x₂) ∧
  linear_function k 1 = -2 →
  k = -5 := by
sorry

end NUMINAMATH_CALUDE_linear_function_proof_l779_77982


namespace NUMINAMATH_CALUDE_summer_jolly_degree_difference_l779_77907

theorem summer_jolly_degree_difference :
  ∀ (summer_degrees jolly_degrees : ℕ),
    summer_degrees = 150 →
    summer_degrees + jolly_degrees = 295 →
    summer_degrees - jolly_degrees = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_summer_jolly_degree_difference_l779_77907


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l779_77953

theorem factor_implies_b_value (a b : ℤ) :
  (∃ c : ℤ, ∀ x : ℝ, (x^2 - 2*x - 1) * (c*x - 1) = a*x^3 + b*x^2 + 1) →
  b = -3 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l779_77953


namespace NUMINAMATH_CALUDE_card_distribution_l779_77954

theorem card_distribution (total_cards : Nat) (num_players : Nat) 
  (h1 : total_cards = 57) (h2 : num_players = 4) :
  ∃ (cards_per_player : Nat) (unassigned_cards : Nat),
    cards_per_player * num_players + unassigned_cards = total_cards ∧
    cards_per_player = 14 ∧
    unassigned_cards = 1 := by
  sorry

end NUMINAMATH_CALUDE_card_distribution_l779_77954


namespace NUMINAMATH_CALUDE_triangle_side_length_l779_77970

theorem triangle_side_length
  (a b c : ℝ)
  (A : ℝ)
  (area : ℝ)
  (h1 : a + b + c = 20)
  (h2 : area = 10 * Real.sqrt 3)
  (h3 : A = π / 3) :
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l779_77970


namespace NUMINAMATH_CALUDE_census_suitability_l779_77937

-- Define the type for survey options
inductive SurveyOption
| A : SurveyOption  -- Favorite TV programs of middle school students
| B : SurveyOption  -- Printing errors on a certain exam paper
| C : SurveyOption  -- Survey on the service life of batteries
| D : SurveyOption  -- Internet usage of middle school students

-- Define what it means for a survey to be suitable for a census
def suitableForCensus (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.B => True
  | _ => False

-- Define the property of examining every item in a population
def examinesEveryItem (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.B => True
  | _ => False

-- Theorem statement
theorem census_suitability :
  ∀ s : SurveyOption, suitableForCensus s ↔ examinesEveryItem s :=
by sorry

end NUMINAMATH_CALUDE_census_suitability_l779_77937


namespace NUMINAMATH_CALUDE_climb_nine_flights_l779_77987

/-- Calculates the number of steps climbed given the number of flights, height per flight, and height per step. -/
def steps_climbed (flights : ℕ) (feet_per_flight : ℕ) (inches_per_step : ℕ) : ℕ :=
  (flights * feet_per_flight * 12) / inches_per_step

/-- Proves that climbing 9 flights of 10-foot stairs with 18-inch steps results in 60 steps. -/
theorem climb_nine_flights : steps_climbed 9 10 18 = 60 := by
  sorry

end NUMINAMATH_CALUDE_climb_nine_flights_l779_77987


namespace NUMINAMATH_CALUDE_multiplicative_inverse_800_mod_7801_l779_77910

theorem multiplicative_inverse_800_mod_7801 (h : 28^2 + 195^2 = 197^2) :
  ∃ n : ℕ, n < 7801 ∧ (800 * n) % 7801 = 1 :=
by
  use 3125
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_800_mod_7801_l779_77910


namespace NUMINAMATH_CALUDE_wood_carvings_per_shelf_example_l779_77929

/-- Given a total number of wood carvings and a number of shelves,
    calculate the number of wood carvings per shelf. -/
def woodCarvingsPerShelf (totalCarvings : ℕ) (numShelves : ℕ) : ℕ :=
  totalCarvings / numShelves

/-- Theorem stating that with 56 total wood carvings and 7 shelves,
    each shelf contains 8 wood carvings. -/
theorem wood_carvings_per_shelf_example :
  woodCarvingsPerShelf 56 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_wood_carvings_per_shelf_example_l779_77929


namespace NUMINAMATH_CALUDE_neg_white_is_black_sum_black_is_white_zero_is_red_nonzero_black_or_white_neg_opposite_color_l779_77995

-- Define the color type
inductive Color : Type
  | Black : Color
  | Red : Color
  | White : Color

-- Define the coloring function
def coloring : ℤ → Color := sorry

-- Define the coloring rules
axiom neg_black_is_white : ∀ n : ℤ, coloring n = Color.Black → coloring (-n) = Color.White
axiom sum_white_is_black : ∀ a b : ℤ, coloring a = Color.White → coloring b = Color.White → coloring (a + b) = Color.Black

-- Theorems to prove
theorem neg_white_is_black : ∀ n : ℤ, coloring n = Color.White → coloring (-n) = Color.Black := sorry

theorem sum_black_is_white : ∀ a b : ℤ, coloring a = Color.Black → coloring b = Color.Black → coloring (a + b) = Color.White := sorry

theorem zero_is_red : coloring 0 = Color.Red := sorry

theorem nonzero_black_or_white : ∀ n : ℤ, n ≠ 0 → (coloring n = Color.Black ∨ coloring n = Color.White) := sorry

theorem neg_opposite_color : ∀ n : ℤ, n ≠ 0 → 
  (coloring n = Color.Black → coloring (-n) = Color.White) ∧ 
  (coloring n = Color.White → coloring (-n) = Color.Black) := sorry

end NUMINAMATH_CALUDE_neg_white_is_black_sum_black_is_white_zero_is_red_nonzero_black_or_white_neg_opposite_color_l779_77995


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_l779_77930

/-- For a parabola y = ax^2 with a > 0, if the length of its latus rectum is 4 units, then a = 1/4 -/
theorem parabola_latus_rectum (a : ℝ) (h1 : a > 0) :
  (1 / a = 4) → a = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_l779_77930


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_origin_l779_77944

/-- Given a line with equation y = 2x + 1, its symmetric line with respect to the origin
    has the equation y = 2x - 1. -/
theorem symmetric_line_wrt_origin :
  ∀ (x y : ℝ), y = 2*x + 1 → ∃ (x' y' : ℝ), y' = 2*x' - 1 ∧ x' = -x ∧ y' = -y :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_origin_l779_77944


namespace NUMINAMATH_CALUDE_question_types_sum_steve_answerable_relation_l779_77925

/-- Represents a math test with different types of questions -/
structure MathTest where
  total : ℕ
  word : ℕ
  addition_subtraction : ℕ
  geometry : ℕ

/-- Defines the properties of a valid math test -/
def is_valid_test (test : MathTest) : Prop :=
  test.word = test.total / 2 ∧
  test.addition_subtraction = test.total / 3 ∧
  test.geometry = test.total - test.word - test.addition_subtraction

/-- Theorem stating the relationship between question types and total questions -/
theorem question_types_sum (test : MathTest) (h : is_valid_test test) :
  test.word + test.addition_subtraction + test.geometry = test.total := by
  sorry

/-- Function representing the number of questions Steve can answer -/
def steve_answerable (total : ℕ) : ℕ :=
  total / 2 - 4

/-- Theorem stating the relationship between Steve's answerable questions and total questions -/
theorem steve_answerable_relation (test : MathTest) (h : is_valid_test test) :
  steve_answerable test.total = test.total / 2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_question_types_sum_steve_answerable_relation_l779_77925


namespace NUMINAMATH_CALUDE_theater_seats_count_l779_77912

/-- Represents a theater with a specific seating arrangement. -/
structure Theater where
  rows : ℕ
  first_row_seats : ℕ
  seat_increment : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater. -/
def total_seats (t : Theater) : ℕ :=
  (t.rows * (2 * t.first_row_seats + (t.rows - 1) * t.seat_increment)) / 2

/-- Theorem stating that a theater with the given properties has 720 seats. -/
theorem theater_seats_count :
  ∀ (t : Theater),
    t.rows = 15 →
    t.first_row_seats = 20 →
    t.seat_increment = 4 →
    t.last_row_seats = 76 →
    total_seats t = 720 :=
by
  sorry


end NUMINAMATH_CALUDE_theater_seats_count_l779_77912


namespace NUMINAMATH_CALUDE_concatenated_number_divisible_by_1980_l779_77941

def concatenated_number : ℕ := sorry

theorem concatenated_number_divisible_by_1980 : 
  ∃ k : ℕ, concatenated_number = 1980 * k := by sorry

end NUMINAMATH_CALUDE_concatenated_number_divisible_by_1980_l779_77941


namespace NUMINAMATH_CALUDE_emily_beads_count_l779_77998

theorem emily_beads_count (necklaces : ℕ) (beads_per_necklace : ℕ) 
  (h1 : necklaces = 11) (h2 : beads_per_necklace = 28) : 
  necklaces * beads_per_necklace = 308 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l779_77998


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_sixths_l779_77968

theorem tan_thirteen_pi_sixths : Real.tan (13 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_sixths_l779_77968


namespace NUMINAMATH_CALUDE_sum_of_even_positive_integers_less_than_100_l779_77988

theorem sum_of_even_positive_integers_less_than_100 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ n > 0) (Finset.range 100)).sum id = 2450 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_even_positive_integers_less_than_100_l779_77988


namespace NUMINAMATH_CALUDE_arkansas_game_sales_l779_77975

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts : ℕ := 172

/-- The number of t-shirts sold during the Texas Tech game -/
def texas_tech_shirts : ℕ := 186 - arkansas_shirts

/-- The revenue per t-shirt in dollars -/
def revenue_per_shirt : ℕ := 78

/-- The total number of t-shirts sold during both games -/
def total_shirts : ℕ := 186

/-- The revenue from the Texas Tech game in dollars -/
def texas_tech_revenue : ℕ := 1092

theorem arkansas_game_sales : 
  arkansas_shirts = 172 ∧ 
  texas_tech_shirts + arkansas_shirts = total_shirts ∧
  texas_tech_shirts * revenue_per_shirt = texas_tech_revenue :=
sorry

end NUMINAMATH_CALUDE_arkansas_game_sales_l779_77975


namespace NUMINAMATH_CALUDE_planet_colonization_combinations_l779_77933

/-- Represents the number of planets of each type -/
structure PlanetCounts where
  venusLike : Nat
  jupiterLike : Nat

/-- Represents the colonization units required for each planet type -/
structure ColonizationUnits where
  venusLike : Nat
  jupiterLike : Nat

/-- Calculates the number of ways to choose planets given the constraints -/
def countPlanetCombinations (totalPlanets : PlanetCounts) (units : ColonizationUnits) (totalUnits : Nat) : Nat :=
  sorry

/-- The main theorem stating the number of combinations for the given problem -/
theorem planet_colonization_combinations :
  let totalPlanets := PlanetCounts.mk 7 5
  let units := ColonizationUnits.mk 3 1
  let totalUnits := 15
  countPlanetCombinations totalPlanets units totalUnits = 435 := by
  sorry

end NUMINAMATH_CALUDE_planet_colonization_combinations_l779_77933


namespace NUMINAMATH_CALUDE_nestedRadical_eq_six_l779_77938

/-- The value of the infinite nested radical sqrt(18 + sqrt(18 + sqrt(18 + ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (18 + Real.sqrt (18 + Real.sqrt (18 + Real.sqrt (18 + Real.sqrt 18))))

/-- Theorem stating that the value of the nested radical is 6 -/
theorem nestedRadical_eq_six : nestedRadical = 6 := by
  sorry

end NUMINAMATH_CALUDE_nestedRadical_eq_six_l779_77938


namespace NUMINAMATH_CALUDE_quarterback_sacks_l779_77919

theorem quarterback_sacks (total_attempts : ℕ) (no_throw_percentage : ℚ) (sack_ratio : ℚ) :
  total_attempts = 80 →
  no_throw_percentage = 30 / 100 →
  sack_ratio = 1 / 2 →
  ↑(total_attempts : ℕ) * no_throw_percentage * sack_ratio = 12 :=
by sorry

end NUMINAMATH_CALUDE_quarterback_sacks_l779_77919


namespace NUMINAMATH_CALUDE_z_is_real_z_is_pure_imaginary_z_in_third_quadrant_l779_77900

-- Define the complex number z as a function of real m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 3*m) (m^2 - m - 6)

-- Theorem for when z is a real number
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = 3 ∨ m = -2 := by sorry

-- Theorem for when z is a pure imaginary number
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 0 := by sorry

-- Theorem for when z is in the third quadrant
theorem z_in_third_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im < 0 ↔ 0 < m ∧ m < 3 := by sorry

end NUMINAMATH_CALUDE_z_is_real_z_is_pure_imaginary_z_in_third_quadrant_l779_77900


namespace NUMINAMATH_CALUDE_hunting_season_quarter_year_l779_77976

/-- Represents the hunting scenario -/
structure HuntingScenario where
  hunts_per_month : ℕ
  deers_per_hunt : ℕ
  deer_weight : ℕ
  kept_fraction : ℚ
  kept_weight : ℕ

/-- Calculates the fraction of the year the hunting season lasts -/
def hunting_season_fraction (scenario : HuntingScenario) : ℚ :=
  let total_catch := scenario.kept_weight / scenario.kept_fraction
  let catch_per_hunt := scenario.deers_per_hunt * scenario.deer_weight
  let hunts_per_year := total_catch / catch_per_hunt
  let months_of_hunting := hunts_per_year / scenario.hunts_per_month
  months_of_hunting / 12

/-- Theorem stating that for the given scenario, the hunting season lasts 1/4 of the year -/
theorem hunting_season_quarter_year (scenario : HuntingScenario) 
  (h1 : scenario.hunts_per_month = 6)
  (h2 : scenario.deers_per_hunt = 2)
  (h3 : scenario.deer_weight = 600)
  (h4 : scenario.kept_fraction = 1/2)
  (h5 : scenario.kept_weight = 10800) :
  hunting_season_fraction scenario = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_hunting_season_quarter_year_l779_77976


namespace NUMINAMATH_CALUDE_vector_relations_l779_77948

/-- Given two vectors a and b in R², prove the following:
    1. The condition for a to be parallel to b
    2. The condition for a to be perpendicular to b
    3. The minimum value of a certain expression when a is perpendicular to b -/
theorem vector_relations (m : ℝ) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1/2, Real.sqrt 3 / 2)
  
  -- 1. a is parallel to b when m = √3/3
  (a.1 / b.1 = a.2 / b.2 ↔ m = Real.sqrt 3 / 3) ∧
  
  -- 2. a is perpendicular to b when m = √3
  (a.1 * b.1 + a.2 * b.2 = 0 ↔ m = Real.sqrt 3) ∧
  
  -- 3. When a ⊥ b, the minimum value of (k + t²)/t is -7/4
  (a.1 * b.1 + a.2 * b.2 = 0 →
    ∀ k t : ℝ, t ≠ 0 →
      (a.1 + t^2 * (-3) * b.1) * (-k * a.1 + t * b.1) +
      (a.2 + t^2 * (-3) * b.2) * (-k * a.2 + t * b.2) = 0 →
        (∀ x : ℝ, (k + t^2) / t ≥ -7/4) ∧
        (∃ x : ℝ, (k + t^2) / t = -7/4)) :=
by sorry


end NUMINAMATH_CALUDE_vector_relations_l779_77948


namespace NUMINAMATH_CALUDE_parallel_perpendicular_transitivity_l779_77961

-- Define the space
variable (S : Type*) [MetricSpace S]

-- Define lines and planes
variable (Line Plane : Type*)

-- Define the lines m and n, and the plane α
variable (m n : Line) (α : Plane)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the distinct relation for lines
variable (distinct : Line → Line → Prop)

-- Theorem statement
theorem parallel_perpendicular_transitivity 
  (h_distinct : distinct m n)
  (h_parallel : parallel m n)
  (h_perpendicular : perpendicular n α) :
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_transitivity_l779_77961


namespace NUMINAMATH_CALUDE_prob_less_than_5_eq_half_l779_77923

/-- A fair 8-sided die -/
def fair_8_sided_die : Finset ℕ := Finset.range 8

/-- The probability of an event occurring when rolling a fair 8-sided die -/
def prob (event : Finset ℕ) : ℚ := (event.card : ℚ) / (fair_8_sided_die.card : ℚ)

/-- The event of rolling a number less than 5 -/
def less_than_5 : Finset ℕ := Finset.filter (λ x => x < 5) fair_8_sided_die

theorem prob_less_than_5_eq_half : 
  prob less_than_5 = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_less_than_5_eq_half_l779_77923


namespace NUMINAMATH_CALUDE_right_triangle_area_l779_77921

/-- The area of a right triangle with legs of 60 feet and 80 feet is 345600 square inches -/
theorem right_triangle_area : 
  let leg1_feet : ℝ := 60
  let leg2_feet : ℝ := 80
  let inches_per_foot : ℝ := 12
  let leg1_inches : ℝ := leg1_feet * inches_per_foot
  let leg2_inches : ℝ := leg2_feet * inches_per_foot
  let area : ℝ := (1/2) * leg1_inches * leg2_inches
  area = 345600 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l779_77921


namespace NUMINAMATH_CALUDE_smallest_odd_n_for_product_l779_77983

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

theorem smallest_odd_n_for_product : 
  ∃ (n : ℕ), is_odd n ∧ 
    (∀ m : ℕ, is_odd m → m < n → (2 : ℝ)^((m+1)^2/7) ≤ 1000) ∧ 
    (2 : ℝ)^((n+1)^2/7) > 1000 ∧ 
    n = 9 := by sorry

end NUMINAMATH_CALUDE_smallest_odd_n_for_product_l779_77983


namespace NUMINAMATH_CALUDE_alternating_sequence_sum_l779_77914

def alternating_sequence (first last step : ℕ) : List ℤ :=
  let n := (first - last) / step + 1
  List.range n |> List.map (λ i => first - i * step) |> List.map (λ x => if x % (2 * step) = 0 then x else -x)

theorem alternating_sequence_sum (first last step : ℕ) :
  first > last ∧ step > 0 ∧ (first - last) % step = 0 →
  List.sum (alternating_sequence first last step) = 520 :=
by
  sorry

#eval List.sum (alternating_sequence 1050 20 20)

end NUMINAMATH_CALUDE_alternating_sequence_sum_l779_77914


namespace NUMINAMATH_CALUDE_star_three_neg_two_thirds_l779_77915

-- Define the ☆ operation
def star (a b : ℚ) : ℚ := a^2 + a*b - 5

-- Theorem statement
theorem star_three_neg_two_thirds : star 3 (-2/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_star_three_neg_two_thirds_l779_77915


namespace NUMINAMATH_CALUDE_smallest_value_in_range_l779_77972

theorem smallest_value_in_range (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < y ∧ y^3 < 3*y ∧ y^3 < y^(1/3) ∧ y^3 < 1/y := by
  sorry

#check smallest_value_in_range

end NUMINAMATH_CALUDE_smallest_value_in_range_l779_77972


namespace NUMINAMATH_CALUDE_power_of_two_equality_l779_77945

theorem power_of_two_equality (x : ℤ) : (1 / 8 : ℚ) * (2 : ℚ)^40 = (2 : ℚ)^x → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l779_77945


namespace NUMINAMATH_CALUDE_coat_original_price_l779_77989

/-- Proves that if a coat is sold for 135 yuan after a 25% discount, its original price was 180 yuan -/
theorem coat_original_price (discounted_price : ℝ) (discount_percent : ℝ) 
  (h1 : discounted_price = 135)
  (h2 : discount_percent = 25) : 
  discounted_price / (1 - discount_percent / 100) = 180 := by
sorry

end NUMINAMATH_CALUDE_coat_original_price_l779_77989


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l779_77918

/-- The length of the path traveled by point F when rolling a quarter-circle region -/
theorem quarter_circle_roll_path_length 
  (EF : ℝ) -- Length of EF (radius of the quarter-circle)
  (h_EF : EF = 3 / Real.pi) -- Given condition that EF = 3/π cm
  : (2 * Real.pi * EF) = 6 := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l779_77918


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l779_77920

theorem constant_term_of_expansion (x : ℝ) (x_pos : x > 0) :
  ∃ (c : ℝ), (∀ (ε : ℝ), ε > 0 → 
    ∃ (δ : ℝ), δ > 0 ∧ 
    ∀ (y : ℝ), abs (y - x) < δ → 
    abs ((y.sqrt + 3 / y)^10 - c) < ε) ∧
  c = 59049 := by
sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l779_77920


namespace NUMINAMATH_CALUDE_mechanism_composition_l779_77957

/-- Represents a mechanism with small and large parts. -/
structure Mechanism where
  total_parts : ℕ
  small_parts : ℕ
  large_parts : ℕ
  total_eq : total_parts = small_parts + large_parts

/-- Property: Among any 12 parts, there is at least one small part. -/
def has_small_in_12 (m : Mechanism) : Prop :=
  ∀ (subset : Finset ℕ), subset.card = 12 → (∃ (x : ℕ), x ∈ subset ∧ x ≤ m.small_parts)

/-- Property: Among any 15 parts, there is at least one large part. -/
def has_large_in_15 (m : Mechanism) : Prop :=
  ∀ (subset : Finset ℕ), subset.card = 15 → (∃ (x : ℕ), x ∈ subset ∧ x > m.small_parts)

/-- Main theorem: If a mechanism satisfies the given conditions, it has 11 large parts and 14 small parts. -/
theorem mechanism_composition (m : Mechanism) 
    (h_total : m.total_parts = 25)
    (h_small : has_small_in_12 m)
    (h_large : has_large_in_15 m) : 
    m.large_parts = 11 ∧ m.small_parts = 14 := by
  sorry


end NUMINAMATH_CALUDE_mechanism_composition_l779_77957


namespace NUMINAMATH_CALUDE_multiply_divide_sqrt_l779_77990

theorem multiply_divide_sqrt (x : ℝ) (y : ℝ) (h1 : x = 0.42857142857142855) (h2 : x ≠ 0) :
  Real.sqrt ((x * y) / 7) = x → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_sqrt_l779_77990


namespace NUMINAMATH_CALUDE_min_sum_floor_l779_77974

theorem min_sum_floor (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ⌊(a + b) / c⌋ + ⌊(b + c) / d⌋ + ⌊(c + a) / b⌋ + ⌊(d + a) / c⌋ ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_floor_l779_77974


namespace NUMINAMATH_CALUDE_clubs_distribution_l779_77964

-- Define the set of cards
inductive Card : Type
| Hearts : Card
| Spades : Card
| Diamonds : Card
| Clubs : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "A gets the clubs"
def A_gets_clubs (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "B gets the clubs"
def B_gets_clubs (d : Distribution) : Prop := d Person.B = Card.Clubs

-- State the theorem
theorem clubs_distribution :
  (∀ d : Distribution, ¬(A_gets_clubs d ∧ B_gets_clubs d)) ∧ 
  (∃ d : Distribution, ¬A_gets_clubs d ∧ ¬B_gets_clubs d) :=
sorry

end NUMINAMATH_CALUDE_clubs_distribution_l779_77964


namespace NUMINAMATH_CALUDE_steven_shirt_count_l779_77991

def brian_shirts : ℕ := 3

def andrew_shirts : ℕ := 6 * brian_shirts

def steven_shirts : ℕ := 4 * andrew_shirts

theorem steven_shirt_count : steven_shirts = 72 := by
  sorry

end NUMINAMATH_CALUDE_steven_shirt_count_l779_77991


namespace NUMINAMATH_CALUDE_original_mean_calculation_l779_77904

theorem original_mean_calculation (n : ℕ) (decrease : ℝ) (updated_mean : ℝ) (h1 : n = 50) (h2 : decrease = 9) (h3 : updated_mean = 191) :
  ∃ (original_mean : ℝ), 
    original_mean * n = updated_mean * n + decrease * n ∧ 
    original_mean = 200 :=
by sorry

end NUMINAMATH_CALUDE_original_mean_calculation_l779_77904


namespace NUMINAMATH_CALUDE_symmetry_about_origin_l779_77950

/-- Given a point (x, y) in R^2, its symmetrical point about the origin is (-x, -y) -/
def symmetrical_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (2, -3)

/-- The proposed symmetrical point -/
def proposed_symmetrical_point : ℝ × ℝ := (-2, 3)

theorem symmetry_about_origin :
  symmetrical_point original_point = proposed_symmetrical_point :=
by sorry

end NUMINAMATH_CALUDE_symmetry_about_origin_l779_77950


namespace NUMINAMATH_CALUDE_same_walking_speed_l779_77901

-- Define Jack's speed function
def jack_speed (x : ℝ) : ℝ := x^2 - 13*x - 48

-- Define Jill's distance function
def jill_distance (x : ℝ) : ℝ := x^2 - 5*x - 84

-- Define Jill's time function
def jill_time (x : ℝ) : ℝ := x + 8

theorem same_walking_speed : 
  ∃ x : ℝ, x > 0 ∧ 
    jack_speed x = jill_distance x / jill_time x ∧ 
    jack_speed x = 6 := by
  sorry

end NUMINAMATH_CALUDE_same_walking_speed_l779_77901


namespace NUMINAMATH_CALUDE_sqrt_three_times_five_to_fourth_l779_77978

theorem sqrt_three_times_five_to_fourth (x : ℝ) : 
  x = Real.sqrt (5^4 + 5^4 + 5^4) → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_times_five_to_fourth_l779_77978


namespace NUMINAMATH_CALUDE_larger_number_l779_77986

theorem larger_number (a b : ℝ) (sum : a + b = 40) (diff : a - b = 10) : max a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l779_77986


namespace NUMINAMATH_CALUDE_distribute_five_teachers_to_three_schools_l779_77949

/-- The number of ways to distribute n teachers to m schools with at least one teacher per school -/
def distribute_teachers (n m : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 teachers to 3 schools with at least one teacher per school -/
theorem distribute_five_teachers_to_three_schools :
  distribute_teachers 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_teachers_to_three_schools_l779_77949


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l779_77935

theorem units_digit_of_expression : 
  (3 * 19 * 1981 - 3^4) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l779_77935


namespace NUMINAMATH_CALUDE_initial_passengers_l779_77999

theorem initial_passengers (P : ℕ) : 
  P % 2 = 0 ∧ 
  (P : ℝ) + 0.08 * (P : ℝ) ≤ 70 ∧ 
  P % 25 = 0 → 
  P = 50 := by
sorry

end NUMINAMATH_CALUDE_initial_passengers_l779_77999


namespace NUMINAMATH_CALUDE_impossible_wire_arrangement_l779_77924

-- Define a regular heptagon with columns
structure RegularHeptagonWithColumns where
  vertices : Fin 7 → ℝ
  is_regular : True  -- Placeholder for regularity condition

-- Define the connection between vertices
def second_nearest_neighbors (i : Fin 7) : Fin 7 × Fin 7 :=
  ((i + 2) % 7, (i + 5) % 7)

-- Define the intersection of wires
def wire_intersections (h : RegularHeptagonWithColumns) (i j : Fin 7) : Prop :=
  let (a, b) := second_nearest_neighbors i
  let (c, d) := second_nearest_neighbors j
  (a = c ∧ b ≠ d) ∨ (a = d ∧ b ≠ c) ∨ (b = c ∧ a ≠ d) ∨ (b = d ∧ a ≠ c)

-- Define the condition for wire arrangement
def valid_wire_arrangement (h : RegularHeptagonWithColumns) : Prop :=
  ∀ i j k : Fin 7, wire_intersections h i j → wire_intersections h i k →
    (h.vertices i < h.vertices j ∧ h.vertices i > h.vertices k) ∨
    (h.vertices i > h.vertices j ∧ h.vertices i < h.vertices k)

-- Theorem statement
theorem impossible_wire_arrangement :
  ¬∃ (h : RegularHeptagonWithColumns), valid_wire_arrangement h :=
sorry

end NUMINAMATH_CALUDE_impossible_wire_arrangement_l779_77924


namespace NUMINAMATH_CALUDE_total_revenue_is_21040_l779_77905

/-- Represents the seating capacity and ticket price for a section of the circus tent. -/
structure Section where
  capacity : ℕ
  price : ℕ

/-- Calculates the revenue for a given section when all seats are occupied. -/
def sectionRevenue (s : Section) : ℕ := s.capacity * s.price

/-- Represents the seating arrangement of the circus tent. -/
def circusTent : List Section :=
  [{ capacity := 246, price := 15 },
   { capacity := 246, price := 15 },
   { capacity := 246, price := 15 },
   { capacity := 246, price := 15 },
   { capacity := 314, price := 20 }]

/-- Calculates the total revenue when all seats are occupied. -/
def totalRevenue : ℕ := (circusTent.map sectionRevenue).sum

/-- Theorem stating that the total revenue when all seats are occupied is $21,040. -/
theorem total_revenue_is_21040 : totalRevenue = 21040 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_21040_l779_77905


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l779_77985

/-- The average speed of a round trip given different speeds for each direction -/
theorem round_trip_average_speed (speed_to_school speed_from_school : ℝ) :
  speed_to_school > 0 →
  speed_from_school > 0 →
  let average_speed := 2 / (1 / speed_to_school + 1 / speed_from_school)
  average_speed = 4.8 ↔ speed_to_school = 6 ∧ speed_from_school = 4 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l779_77985


namespace NUMINAMATH_CALUDE_inequality_implies_b_leq_c_l779_77960

theorem inequality_implies_b_leq_c
  (a b c x y : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (pos_x : 0 < x)
  (pos_y : 0 < y)
  (h : a * x + b * y ≤ b * x + c * y ∧ b * x + c * y ≤ c * x + a * y) :
  b ≤ c :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_b_leq_c_l779_77960


namespace NUMINAMATH_CALUDE_grunters_win_probability_l779_77963

/-- The probability of a team winning a single game -/
def win_probability : ℚ := 4/5

/-- The number of games in the series -/
def num_games : ℕ := 5

/-- The probability of winning all games in the series -/
def win_all_probability : ℚ := (4/5)^5

theorem grunters_win_probability :
  win_all_probability = 1024/3125 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l779_77963


namespace NUMINAMATH_CALUDE_point_O_is_circumcenter_l779_77926

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the triangle ABC
def Triangle (A B C : Point3D) : Prop := True

-- Define the plane α containing the triangle
def PlaneContainsTriangle (α : Plane) (A B C : Point3D) : Prop := True

-- Define a point being outside a plane
def PointOutsidePlane (P : Point3D) (α : Plane) : Prop := True

-- Define perpendicularity between a line and a plane
def PerpendicularToPlane (P O : Point3D) (α : Plane) : Prop := True

-- Define the foot of a perpendicular
def FootOfPerpendicular (O : Point3D) (P : Point3D) (α : Plane) : Prop := True

-- Define equality of distances
def EqualDistances (P A B C : Point3D) : Prop := True

-- Define circumcenter
def Circumcenter (O : Point3D) (A B C : Point3D) : Prop := True

theorem point_O_is_circumcenter 
  (α : Plane) (A B C P O : Point3D)
  (h1 : Triangle A B C)
  (h2 : PlaneContainsTriangle α A B C)
  (h3 : PointOutsidePlane P α)
  (h4 : PerpendicularToPlane P O α)
  (h5 : FootOfPerpendicular O P α)
  (h6 : EqualDistances P A B C) :
  Circumcenter O A B C := by
  sorry


end NUMINAMATH_CALUDE_point_O_is_circumcenter_l779_77926


namespace NUMINAMATH_CALUDE_trader_profit_above_goal_l779_77951

theorem trader_profit_above_goal 
  (profit : ℝ) 
  (required_amount : ℝ) 
  (donation : ℝ) 
  (half_profit : ℝ) 
  (h1 : profit = 960) 
  (h2 : required_amount = 610) 
  (h3 : donation = 310) 
  (h4 : half_profit = profit / 2) : 
  half_profit + donation - required_amount = 180 := by
sorry

end NUMINAMATH_CALUDE_trader_profit_above_goal_l779_77951


namespace NUMINAMATH_CALUDE_folk_song_competition_probability_l779_77977

/-- The number of provinces in the competition -/
def num_provinces : ℕ := 6

/-- The number of singers per province -/
def singers_per_province : ℕ := 2

/-- The total number of singers in the competition -/
def total_singers : ℕ := num_provinces * singers_per_province

/-- The number of winners to be selected -/
def num_winners : ℕ := 4

/-- The probability of selecting 4 winners such that exactly two of them are from the same province -/
theorem folk_song_competition_probability :
  (num_provinces.choose 1 * singers_per_province.choose 2 * (total_singers - singers_per_province).choose 1 * (num_provinces - 1).choose 1) / total_singers.choose num_winners = 16 / 33 := by
  sorry

end NUMINAMATH_CALUDE_folk_song_competition_probability_l779_77977


namespace NUMINAMATH_CALUDE_material_mix_ratio_l779_77917

theorem material_mix_ratio (x y : ℝ) 
  (h1 : 50 * x + 40 * y = 50 * (1 + 0.1) * x + 40 * (1 - 0.15) * y) : 
  x / y = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_material_mix_ratio_l779_77917


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l779_77946

theorem polynomial_coefficient_sum : 
  ∀ (A B C D : ℝ), 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 6 - x) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 28 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l779_77946


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l779_77911

theorem sqrt_equation_solution (y : ℝ) :
  (y > 2) →
  (Real.sqrt (7 * y) / Real.sqrt (4 * (y - 2)) = 3) →
  y = 72 / 29 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l779_77911


namespace NUMINAMATH_CALUDE_no_real_solutions_to_equation_l779_77993

theorem no_real_solutions_to_equation : 
  ¬ ∃ (x : ℝ), x > 0 ∧ x^(Real.log x / Real.log 10) = x^3 / 1000 :=
sorry

end NUMINAMATH_CALUDE_no_real_solutions_to_equation_l779_77993


namespace NUMINAMATH_CALUDE_min_value_theorem_l779_77932

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a * b = 1) :
  (a^2 + b^2) / (a - b) ≥ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l779_77932


namespace NUMINAMATH_CALUDE_haman_initial_trays_l779_77934

/-- Represents the number of eggs in a standard tray -/
def eggs_per_tray : ℕ := 30

/-- Represents the number of trays Haman dropped -/
def dropped_trays : ℕ := 2

/-- Represents the number of trays added after the accident -/
def added_trays : ℕ := 7

/-- Represents the total number of eggs sold -/
def total_eggs_sold : ℕ := 540

/-- Theorem stating that Haman initially collected 13 trays of eggs -/
theorem haman_initial_trays :
  (total_eggs_sold / eggs_per_tray - added_trays + dropped_trays : ℕ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_haman_initial_trays_l779_77934


namespace NUMINAMATH_CALUDE_expression_evaluation_l779_77955

theorem expression_evaluation : 3^(0^(2^5)) + ((3^0)^2)^5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l779_77955


namespace NUMINAMATH_CALUDE_percentage_error_division_vs_multiplication_l779_77902

theorem percentage_error_division_vs_multiplication (x : ℝ) (h : x > 0) : 
  (|4 * x - x / 4| / (4 * x)) * 100 = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_error_division_vs_multiplication_l779_77902


namespace NUMINAMATH_CALUDE_cone_volume_and_surface_area_l779_77966

/-- Cone with given slant height and height -/
structure Cone where
  slant_height : ℝ
  height : ℝ

/-- Volume of a cone -/
def volume (c : Cone) : ℝ := sorry

/-- Surface area of a cone -/
def surface_area (c : Cone) : ℝ := sorry

/-- Theorem stating the volume and surface area of a specific cone -/
theorem cone_volume_and_surface_area :
  let c : Cone := { slant_height := 17, height := 15 }
  (volume c = 320 * Real.pi) ∧ (surface_area c = 200 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_and_surface_area_l779_77966


namespace NUMINAMATH_CALUDE_probability_square_factor_l779_77940

/-- A standard 6-sided die -/
def StandardDie : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- The number of dice rolled -/
def NumDice : Nat := 6

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, n = m * m

/-- The probability of rolling a product containing a square factor -/
def probabilitySquareFactor : ℚ := 665 / 729

/-- Theorem stating the probability of rolling a product containing a square factor -/
theorem probability_square_factor :
  (1 : ℚ) - (2 / 3) ^ NumDice = probabilitySquareFactor := by sorry

end NUMINAMATH_CALUDE_probability_square_factor_l779_77940


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l779_77994

/-- Given a train and platform with specific lengths, prove the time taken to cross the platform -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (signal_pole_crossing_time : ℝ) 
  (h1 : train_length = 600)
  (h2 : platform_length = 700)
  (h3 : signal_pole_crossing_time = 18) :
  (train_length + platform_length) / (train_length / signal_pole_crossing_time) = 39 := by
  sorry

#check train_platform_crossing_time

end NUMINAMATH_CALUDE_train_platform_crossing_time_l779_77994


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_17_l779_77992

theorem least_five_digit_congruent_to_6_mod_17 :
  ∃ (n : ℕ), (n ≥ 10000 ∧ n < 100000) ∧ 
             (n % 17 = 6) ∧ 
             (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m) ∧
             n = 10002 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_17_l779_77992


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l779_77959

/-- Coefficient of the r-th term in the binomial expansion of (x + 1/(2√x))^n -/
def coeff (n : ℕ) (r : ℕ) : ℚ :=
  (1 / 2^r) * (n.choose r)

/-- The expansion of (x + 1/(2√x))^n has coefficients forming an arithmetic sequence
    for the first three terms -/
def arithmetic_sequence (n : ℕ) : Prop :=
  coeff n 0 - coeff n 1 = coeff n 1 - coeff n 2

/-- The r-th term has the maximum coefficient in the expansion -/
def max_coeff (n : ℕ) (r : ℕ) : Prop :=
  ∀ k, k ≠ r → coeff n r ≥ coeff n k

theorem binomial_expansion_properties :
  ∃ n : ℕ,
    arithmetic_sequence n ∧
    max_coeff n 2 ∧
    max_coeff n 3 ∧
    ∀ r, r ≠ 2 ∧ r ≠ 3 → ¬(max_coeff n r) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l779_77959


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l779_77928

theorem cube_sum_from_sum_and_product (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l779_77928


namespace NUMINAMATH_CALUDE_right_triangle_area_l779_77936

theorem right_triangle_area (AB AC : ℝ) (h1 : AB = 12) (h2 : AC = 5) :
  let BC : ℝ := Real.sqrt (AB^2 - AC^2)
  (1 / 2) * AC * BC = (5 * Real.sqrt 119) / 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l779_77936


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l779_77939

theorem pure_imaginary_condition (m : ℝ) : 
  (Complex.I * Complex.I = -1) →
  (Complex.mk (m + 2) (-1) = Complex.mk 0 (Complex.im (Complex.mk (m + 2) (-1)))) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l779_77939


namespace NUMINAMATH_CALUDE_jerry_collection_cost_l779_77922

/-- The amount of money Jerry needs to finish his action figure collection. -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem stating the amount Jerry needs to finish his collection. -/
theorem jerry_collection_cost :
  money_needed 7 25 12 = 216 := by
  sorry

end NUMINAMATH_CALUDE_jerry_collection_cost_l779_77922


namespace NUMINAMATH_CALUDE_probability_draw_green_or_white_l779_77947

/-- The probability of drawing either a green or white marble from a bag -/
def probability_green_or_white (green white black : ℕ) : ℚ :=
  (green + white) / (green + white + black)

/-- Theorem: The probability of drawing either a green or white marble
    from a bag containing 4 green, 3 white, and 8 black marbles is 7/15 -/
theorem probability_draw_green_or_white :
  probability_green_or_white 4 3 8 = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_draw_green_or_white_l779_77947


namespace NUMINAMATH_CALUDE_conic_sections_properties_l779_77952

-- Define the equations for the conic sections
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1
def ellipse_eq (x y : ℝ) : Prop := x^2 / 35 + y^2 = 1
def parabola_eq (x y p : ℝ) : Prop := y^2 = 2 * p * x

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 - 5 * x + 2 = 0

-- Define the theorem
theorem conic_sections_properties :
  -- Proposition ②
  (∃ e₁ e₂ : ℝ, quadratic_eq e₁ ∧ quadratic_eq e₂ ∧ 0 < e₁ ∧ e₁ < 1 ∧ e₂ > 1) ∧
  -- Proposition ③
  (∃ c : ℝ, (∀ x y : ℝ, hyperbola_eq x y → x^2 - c^2 = 25) ∧
            (∀ x y : ℝ, ellipse_eq x y → x^2 + c^2 = 35)) ∧
  -- Proposition ④
  (∀ p : ℝ, p > 0 →
    ∃ x₀ y₀ r : ℝ,
      -- Circle equation
      (∀ x y : ℝ, (x - x₀)^2 + (y - y₀)^2 = r^2 →
        -- Tangent to directrix
        x = -p ∨
        -- Passes through focus
        (x₀ = p/2 ∧ y₀ = 0 ∧ r = p/2))) :=
sorry

end NUMINAMATH_CALUDE_conic_sections_properties_l779_77952


namespace NUMINAMATH_CALUDE_inequality_proof_l779_77943

theorem inequality_proof (x : ℝ) : 
  -2 < (x^2 - 10*x + 9) / (x^2 - 4*x + 8) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 8) < 2 ↔ 
  1/3 < x ∧ x < 14/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l779_77943


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l779_77909

/-- An isosceles triangle with base 10 and equal sides 7 has perimeter 24 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun base equal_side perimeter =>
    base = 10 ∧ equal_side = 7 ∧ perimeter = base + 2 * equal_side → perimeter = 24

#check isosceles_triangle_perimeter

theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 10 7 24 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l779_77909


namespace NUMINAMATH_CALUDE_market_fruit_count_l779_77979

theorem market_fruit_count (apples oranges bananas : ℕ) 
  (h1 : apples = oranges + 27)
  (h2 : oranges = bananas + 11)
  (h3 : apples + oranges + bananas = 301) :
  apples = 122 := by
sorry

end NUMINAMATH_CALUDE_market_fruit_count_l779_77979


namespace NUMINAMATH_CALUDE_olivia_initial_money_l779_77997

/-- The amount of money Olivia and Nigel spent on tickets -/
def ticket_cost : ℕ := 6 * 28

/-- The amount of money Nigel had initially -/
def nigel_money : ℕ := 139

/-- The amount of money Olivia and Nigel have left after buying tickets -/
def remaining_money : ℕ := 83

/-- The amount of money Olivia had initially -/
def olivia_money : ℕ := (ticket_cost + remaining_money) - nigel_money

theorem olivia_initial_money : olivia_money = 112 := by
  sorry

end NUMINAMATH_CALUDE_olivia_initial_money_l779_77997


namespace NUMINAMATH_CALUDE_sandy_earnings_l779_77984

/-- Calculates the total earnings for a given hourly rate and hours worked over three days -/
def total_earnings (hourly_rate : ℝ) (hours_day1 hours_day2 hours_day3 : ℝ) : ℝ :=
  hourly_rate * (hours_day1 + hours_day2 + hours_day3)

/-- Sandy's earnings problem -/
theorem sandy_earnings : 
  let hourly_rate : ℝ := 15
  let hours_friday : ℝ := 10
  let hours_saturday : ℝ := 6
  let hours_sunday : ℝ := 14
  total_earnings hourly_rate hours_friday hours_saturday hours_sunday = 450 := by
  sorry

end NUMINAMATH_CALUDE_sandy_earnings_l779_77984


namespace NUMINAMATH_CALUDE_percentage_calculation_l779_77927

theorem percentage_calculation (n : ℝ) : n = 4000 → (0.15 * (0.30 * (0.50 * n))) = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l779_77927


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l779_77967

theorem sufficient_not_necessary : 
  (∃ a : ℝ, a^2 > 16 ∧ a ≤ 4) ∧ 
  (∀ a : ℝ, a > 4 → a^2 > 16) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l779_77967


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_l779_77973

/-- The sum of distinct powers of 2 that equals 72 -/
def sum_of_powers (a b c : ℕ) : Prop :=
  2^a + 2^b + 2^c = 72 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The least sum of exponents when expressing 72 as a sum of at least three distinct powers of 2 -/
theorem least_sum_of_exponents :
  ∃ (a b c : ℕ), sum_of_powers a b c ∧
    ∀ (x y z : ℕ), sum_of_powers x y z → a + b + c ≤ x + y + z ∧ a + b + c = 9 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_l779_77973


namespace NUMINAMATH_CALUDE_theater_admission_revenue_l779_77965

/-- Calculates the total amount collected from theater admission tickets. -/
theorem theater_admission_revenue
  (total_persons : ℕ)
  (num_children : ℕ)
  (adult_price : ℚ)
  (child_price : ℚ)
  (h1 : total_persons = 280)
  (h2 : num_children = 80)
  (h3 : adult_price = 60 / 100)
  (h4 : child_price = 25 / 100) :
  (total_persons - num_children) * adult_price + num_children * child_price = 140 / 100 := by
  sorry

end NUMINAMATH_CALUDE_theater_admission_revenue_l779_77965


namespace NUMINAMATH_CALUDE_weed_ratio_l779_77903

/-- Represents the number of weeds pulled on each day --/
structure WeedCount where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Defines the conditions of the weed-pulling problem --/
def weed_problem (w : WeedCount) : Prop :=
  w.tuesday = 25 ∧
  w.thursday = w.wednesday / 5 ∧
  w.friday = w.thursday - 10 ∧
  w.tuesday + w.wednesday + w.thursday + w.friday = 120

/-- The theorem to be proved --/
theorem weed_ratio (w : WeedCount) (h : weed_problem w) : 
  w.wednesday = 3 * w.tuesday :=
sorry

end NUMINAMATH_CALUDE_weed_ratio_l779_77903


namespace NUMINAMATH_CALUDE_g_extreme_points_l779_77962

noncomputable def f (x : ℝ) : ℝ := Real.log x - x - 1

noncomputable def g (x : ℝ) : ℝ := x * f x + (1/2) * x^2 + 2 * x

noncomputable def g' (x : ℝ) : ℝ := f x + 3

theorem g_extreme_points :
  ∃ (x₁ x₂ : ℝ), 
    0 < x₁ ∧ x₁ < 1 ∧
    3 < x₂ ∧ x₂ < 4 ∧
    g' x₁ = 0 ∧ g' x₂ = 0 ∧
    (∀ x ∈ Set.Ioo 0 1, x ≠ x₁ → g' x ≠ 0) ∧
    (∀ x ∈ Set.Ioo 3 4, x ≠ x₂ → g' x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_g_extreme_points_l779_77962


namespace NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l779_77942

theorem cube_sum_divisible_by_nine (n : ℕ+) :
  ∃ k : ℤ, n^3 + (n + 1)^3 + (n + 2)^3 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divisible_by_nine_l779_77942


namespace NUMINAMATH_CALUDE_number_operations_l779_77996

theorem number_operations (x : ℝ) : ((x + 5) * 5 - 5) / 5 = 5 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l779_77996


namespace NUMINAMATH_CALUDE_alpha_plus_beta_l779_77981

theorem alpha_plus_beta (α β : ℝ) :
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 90*x + 1980) / (x^2 + 70*x - 3570)) →
  α + β = 123 := by
  sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_l779_77981


namespace NUMINAMATH_CALUDE_percentage_problem_l779_77969

theorem percentage_problem (P : ℝ) : 
  0 ≤ P ∧ P ≤ 100 → P * 700 = (60 / 100) * 150 + 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l779_77969


namespace NUMINAMATH_CALUDE_credit_remaining_proof_l779_77980

def credit_problem (credit_limit : ℕ) (payment1 : ℕ) (payment2 : ℕ) : ℕ :=
  credit_limit - payment1 - payment2

theorem credit_remaining_proof :
  credit_problem 100 15 23 = 62 := by
  sorry

end NUMINAMATH_CALUDE_credit_remaining_proof_l779_77980


namespace NUMINAMATH_CALUDE_age_ratio_after_two_years_l779_77971

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the specified conditions. -/
theorem age_ratio_after_two_years (son_age : ℕ) (man_age : ℕ) : 
  son_age = 27 → 
  man_age = son_age + 29 → 
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_after_two_years_l779_77971
