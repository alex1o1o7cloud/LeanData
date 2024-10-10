import Mathlib

namespace height_prediction_at_10_l1944_194451

/-- Represents a linear regression model for height based on age -/
structure HeightModel where
  slope : ℝ
  intercept : ℝ

/-- Predicts the height for a given age using the model -/
def predict_height (model : HeightModel) (age : ℝ) : ℝ :=
  model.slope * age + model.intercept

/-- Theorem stating that the predicted height at age 10 is approximately 145.83cm -/
theorem height_prediction_at_10 (model : HeightModel)
  (h_slope : model.slope = 7.19)
  (h_intercept : model.intercept = 73.93) :
  ∃ ε > 0, |predict_height model 10 - 145.83| < ε :=
sorry

#check height_prediction_at_10

end height_prediction_at_10_l1944_194451


namespace profit_margin_properties_l1944_194403

/-- Profit margin calculation --/
theorem profit_margin_properties
  (B E : ℝ)  -- Purchase price and selling price
  (hE : E > B)  -- Condition: selling price is greater than purchase price
  (a : ℝ := 100 * (E - B) / B)  -- Profit margin from bottom up
  (f : ℝ := 100 * (E - B) / E)  -- Profit margin from top down
  : 
  (f = 100 * a / (a + 100) ∧ a = 100 * f / (100 - f)) ∧  -- Conversion formulas
  (a - f = a * f / 100)  -- Difference property
  := by sorry

end profit_margin_properties_l1944_194403


namespace min_seats_for_adjacent_seating_l1944_194489

/-- Represents a seating arrangement in a row of seats. -/
structure SeatingArrangement where
  total_seats : ℕ
  min_gap : ℕ
  occupied_seats : ℕ

/-- Checks if a seating arrangement is valid according to the rules. -/
def is_valid_arrangement (sa : SeatingArrangement) : Prop :=
  sa.total_seats = 150 ∧ sa.min_gap = 2 ∧ sa.occupied_seats ≤ sa.total_seats

/-- Checks if the next person must sit next to someone in the given arrangement. -/
def forces_adjacent_seating (sa : SeatingArrangement) : Prop :=
  ∀ (new_seat : ℕ), new_seat ≤ sa.total_seats →
    (∃ (occupied : ℕ), occupied ≤ sa.total_seats ∧
      (new_seat = occupied + 1 ∨ new_seat = occupied - 1))

/-- The main theorem stating the minimum number of occupied seats. -/
theorem min_seats_for_adjacent_seating :
  ∃ (sa : SeatingArrangement),
    is_valid_arrangement sa ∧
    forces_adjacent_seating sa ∧
    sa.occupied_seats = 74 ∧
    (∀ (sa' : SeatingArrangement),
      is_valid_arrangement sa' ∧
      forces_adjacent_seating sa' →
      sa'.occupied_seats ≥ 74) :=
  sorry

end min_seats_for_adjacent_seating_l1944_194489


namespace modified_cube_edge_count_l1944_194492

/-- Represents a modified cube with smaller cubes removed from corners and sliced by a plane -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ
  slicePlane : Bool

/-- Calculates the number of edges in the modified cube -/
def edgeCount (cube : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that a cube with side length 5, smaller cubes of side length 2 removed from corners,
    and sliced by a plane, has 40 edges -/
theorem modified_cube_edge_count :
  let cube : ModifiedCube := {
    originalSideLength := 5,
    removedCubeSideLength := 2,
    slicePlane := true
  }
  edgeCount cube = 40 := by sorry

end modified_cube_edge_count_l1944_194492


namespace sisters_get_five_bars_l1944_194491

/-- Calculates the number of granola bars each sister receives when splitting the remaining bars evenly -/
def granola_bars_per_sister (total : ℕ) (set_aside : ℕ) (traded : ℕ) (num_sisters : ℕ) : ℕ :=
  (total - set_aside - traded) / num_sisters

/-- Proves that given the specific conditions, each sister receives 5 granola bars -/
theorem sisters_get_five_bars :
  let total := 20
  let set_aside := 7
  let traded := 3
  let num_sisters := 2
  granola_bars_per_sister total set_aside traded num_sisters = 5 := by
  sorry

#eval granola_bars_per_sister 20 7 3 2

end sisters_get_five_bars_l1944_194491


namespace workers_required_l1944_194400

/-- Given a craft factory that needs to produce 60 units per day, 
    and each worker can produce x units per day, 
    prove that the number of workers required y is equal to 60/x -/
theorem workers_required (x : ℝ) (h : x > 0) : 
  ∃ y : ℝ, y * x = 60 ∧ y = 60 / x := by
  sorry

end workers_required_l1944_194400


namespace volleyball_team_selection_l1944_194473

theorem volleyball_team_selection (total_players : Nat) (quadruplets : Nat) (starters : Nat) :
  total_players = 18 →
  quadruplets = 4 →
  starters = 8 →
  (total_players.choose (starters - quadruplets)) = 1001 := by
  sorry

end volleyball_team_selection_l1944_194473


namespace water_cooler_capacity_l1944_194423

/-- Represents the capacity of the water cooler in ounces -/
def cooler_capacity : ℕ := 126

/-- Number of linemen on the team -/
def num_linemen : ℕ := 12

/-- Number of skill position players on the team -/
def num_skill_players : ℕ := 10

/-- Amount of water each lineman drinks in ounces -/
def lineman_water : ℕ := 8

/-- Amount of water each skill position player drinks in ounces -/
def skill_player_water : ℕ := 6

/-- Number of skill position players who can drink before refill -/
def skill_players_before_refill : ℕ := 5

theorem water_cooler_capacity : 
  cooler_capacity = num_linemen * lineman_water + skill_players_before_refill * skill_player_water :=
by sorry

end water_cooler_capacity_l1944_194423


namespace contrapositive_equivalence_l1944_194429

theorem contrapositive_equivalence (p q : Prop) :
  (p → q) ↔ (¬q → ¬p) := by
  sorry

end contrapositive_equivalence_l1944_194429


namespace friends_team_assignment_l1944_194444

theorem friends_team_assignment : 
  let n : ℕ := 8  -- number of friends
  let k : ℕ := 4  -- number of teams
  k ^ n = 65536 := by sorry

end friends_team_assignment_l1944_194444


namespace min_socks_theorem_l1944_194446

/-- Represents a collection of socks with at least 5 different colors -/
structure SockCollection where
  colors : Nat
  min_socks_per_color : Nat
  colors_ge_5 : colors ≥ 5
  min_socks_ge_40 : min_socks_per_color ≥ 40

/-- The smallest number of socks that must be selected to guarantee at least 15 pairs -/
def min_socks_for_15_pairs (sc : SockCollection) : Nat :=
  38

theorem min_socks_theorem (sc : SockCollection) :
  min_socks_for_15_pairs sc = 38 := by
  sorry

#check min_socks_theorem

end min_socks_theorem_l1944_194446


namespace section_area_is_28_sqrt_34_l1944_194407

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  edge_length : ℝ
  origin : Point3D

/-- Calculates the area of the section cut by a plane in a cube -/
noncomputable def sectionArea (cube : Cube) (plane : Plane) : ℝ :=
  sorry

/-- Theorem: The area of the section cut by plane α in the given cube is 28√34 -/
theorem section_area_is_28_sqrt_34 :
  let cube : Cube := { edge_length := 12, origin := { x := 0, y := 0, z := 0 } }
  let A : Point3D := cube.origin
  let E : Point3D := { x := 12, y := 0, z := 9 }
  let F : Point3D := { x := 0, y := 12, z := 9 }
  let plane : Plane := { a := 1, b := 1, c := -3/4, d := 0 }
  sectionArea cube plane = 28 * Real.sqrt 34 := by
  sorry

end section_area_is_28_sqrt_34_l1944_194407


namespace londolozi_lion_cubs_per_month_l1944_194482

/-- The number of lion cubs born per month in Londolozi -/
def lion_cubs_per_month (initial_population final_population : ℕ) (months death_rate : ℕ) : ℕ :=
  (final_population - initial_population + months * death_rate) / months

/-- Theorem stating the number of lion cubs born per month in Londolozi -/
theorem londolozi_lion_cubs_per_month :
  lion_cubs_per_month 100 148 12 1 = 5 := by
  sorry

end londolozi_lion_cubs_per_month_l1944_194482


namespace no_integer_solutions_l1944_194411

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^3 + 3 = 4*y*(y + 1) := by
  sorry

end no_integer_solutions_l1944_194411


namespace pick_two_different_suits_standard_deck_l1944_194469

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    h_total := rfl }

/-- The number of ways to pick two cards from different suits -/
def pick_two_different_suits (d : Deck) : Nat :=
  d.total_cards * (d.total_cards - d.cards_per_suit)

theorem pick_two_different_suits_standard_deck :
  pick_two_different_suits standard_deck = 2028 := by
  sorry

#eval pick_two_different_suits standard_deck

end pick_two_different_suits_standard_deck_l1944_194469


namespace kanul_total_amount_l1944_194438

/-- The total amount Kanul had -/
def total_amount : ℝ := 137500

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 80000

/-- The amount spent on machinery -/
def machinery : ℝ := 30000

/-- The percentage of total amount spent as cash -/
def cash_percentage : ℝ := 0.20

theorem kanul_total_amount : 
  total_amount = raw_materials + machinery + cash_percentage * total_amount := by
  sorry

end kanul_total_amount_l1944_194438


namespace profitable_iff_price_ge_132_l1944_194415

/-- The transaction fee rate for stock trading in China -/
def fee_rate : ℚ := 75 / 10000

/-- The number of shares traded -/
def num_shares : ℕ := 1000

/-- The price increase per share -/
def price_increase : ℚ := 2

/-- Determines if a stock transaction is profitable given the initial price -/
def is_profitable (x : ℚ) : Prop :=
  (x + price_increase) * (1 - fee_rate) * num_shares ≥ (1 + fee_rate) * num_shares * x

/-- Theorem: The transaction is profitable if and only if the initial share price is at least 132 yuan -/
theorem profitable_iff_price_ge_132 (x : ℚ) : is_profitable x ↔ x ≥ 132 := by
  sorry

end profitable_iff_price_ge_132_l1944_194415


namespace survey_result_l1944_194494

theorem survey_result (total : ℕ) (thought_diseases : ℕ) (said_rabies : ℕ) :
  (thought_diseases : ℚ) / total = 3 / 4 →
  (said_rabies : ℚ) / thought_diseases = 1 / 2 →
  said_rabies = 18 →
  total = 48 :=
by sorry

end survey_result_l1944_194494


namespace exists_element_with_mass_percentage_l1944_194412

/-- Molar mass of Hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- Molar mass of Bromine in g/mol -/
def molar_mass_Br : ℝ := 79.90

/-- Molar mass of Oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Molar mass of HBrO3 in g/mol -/
def molar_mass_HBrO3 : ℝ := molar_mass_H + molar_mass_Br + 3 * molar_mass_O

/-- Mass percentage of a certain element in HBrO3 -/
def target_mass_percentage : ℝ := 0.78

theorem exists_element_with_mass_percentage :
  ∃ (element_mass : ℝ), 
    0 < element_mass ∧ 
    element_mass ≤ molar_mass_HBrO3 ∧
    (element_mass / molar_mass_HBrO3) * 100 = target_mass_percentage :=
by sorry

end exists_element_with_mass_percentage_l1944_194412


namespace birthday_problem_l1944_194478

/-- The number of trainees -/
def n : ℕ := 62

/-- The number of days in a year -/
def d : ℕ := 365

/-- The probability of at least two trainees sharing a birthday -/
noncomputable def prob_shared_birthday : ℝ :=
  1 - (d.factorial / (d - n).factorial : ℝ) / d ^ n

theorem birthday_problem :
  ∃ (p : ℝ), prob_shared_birthday = p ∧ p > 0.9959095 ∧ p < 0.9959096 :=
sorry

end birthday_problem_l1944_194478


namespace dime_count_l1944_194439

/-- Given a collection of coins consisting of dimes and nickels, 
    this theorem proves the number of dimes given the total number 
    of coins and their total value. -/
theorem dime_count 
  (total_coins : ℕ) 
  (total_value : ℚ) 
  (h_total : total_coins = 36) 
  (h_value : total_value = 31/10) : 
  ∃ (dimes nickels : ℕ),
    dimes + nickels = total_coins ∧ 
    (dimes : ℚ) / 10 + (nickels : ℚ) / 20 = total_value ∧
    dimes = 26 := by
  sorry

end dime_count_l1944_194439


namespace sum_of_reciprocals_l1944_194484

theorem sum_of_reciprocals (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^4 = 1 →
  ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) : ℂ) = 4 / ω^2 →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) : ℝ) = -1 :=
by sorry

end sum_of_reciprocals_l1944_194484


namespace jean_money_l1944_194445

theorem jean_money (jane : ℕ) (jean : ℕ) : 
  jean = 3 * jane → 
  jean + jane = 76 → 
  jean = 57 := by
sorry

end jean_money_l1944_194445


namespace min_sum_factors_l1944_194479

theorem min_sum_factors (n : ℕ) (hn : n = 2025) :
  ∃ (a b : ℕ), a * b = n ∧ a > 0 ∧ b > 0 ∧
  (∀ (x y : ℕ), x * y = n → x > 0 → y > 0 → a + b ≤ x + y) ∧
  a + b = 90 :=
by sorry

end min_sum_factors_l1944_194479


namespace num_motorcycles_in_parking_lot_l1944_194430

-- Define the number of wheels for each vehicle type
def car_wheels : ℕ := 5
def motorcycle_wheels : ℕ := 2
def tricycle_wheels : ℕ := 3

-- Define the number of cars and tricycles
def num_cars : ℕ := 19
def num_tricycles : ℕ := 11

-- Define the total number of wheels
def total_wheels : ℕ := 184

-- Theorem to prove
theorem num_motorcycles_in_parking_lot :
  ∃ (num_motorcycles : ℕ),
    num_motorcycles = 28 ∧
    num_motorcycles * motorcycle_wheels +
    num_cars * car_wheels +
    num_tricycles * tricycle_wheels = total_wheels :=
by sorry

end num_motorcycles_in_parking_lot_l1944_194430


namespace problem_solution_l1944_194488

theorem problem_solution : 
  (-(3^3) * (-1/3) + |(-2)| / ((-1/2)^2) = 17) ∧ 
  (7 - 12 * (2/3 - 3/4 + 5/6) = -2) := by sorry

end problem_solution_l1944_194488


namespace circle_area_from_polar_equation_l1944_194431

/-- The area of the circle represented by the polar equation r = 4cosθ - 3sinθ -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := λ θ => 4 * Real.cos θ - 3 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, (r θ * Real.cos θ - center.1)^2 + (r θ * Real.sin θ - center.2)^2 = radius^2) ∧
    π * radius^2 = 25 * π / 4 :=
by sorry

end circle_area_from_polar_equation_l1944_194431


namespace six_digit_number_divisibility_l1944_194442

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Represents the six-digit number formed by appending double of a three-digit number -/
def makeSixDigitNumber (n : ThreeDigitNumber) : Nat :=
  1000 * n.toNat + 2 * n.toNat

theorem six_digit_number_divisibility (n : ThreeDigitNumber) :
  (∃ k : Nat, makeSixDigitNumber n = 2 * k) ∧
  (∃ m : Nat, makeSixDigitNumber n = 3 * m ↔ ∃ l : Nat, n.toNat = 3 * l) :=
sorry

end six_digit_number_divisibility_l1944_194442


namespace parabola_coefficients_l1944_194452

/-- A parabola with vertex at (-2, 5) passing through (2, 9) has coefficients a = 1/4, b = 1, c = 6 -/
theorem parabola_coefficients :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (5 = a * (-2)^2 + b * (-2) + c) →
  (∀ x : ℝ, a * (x + 2)^2 + 5 = a * x^2 + b * x + c) →
  (9 = a * 2^2 + b * 2 + c) →
  (a = 1/4 ∧ b = 1 ∧ c = 6) :=
by sorry

end parabola_coefficients_l1944_194452


namespace quadratic_roots_to_coefficients_l1944_194485

theorem quadratic_roots_to_coefficients :
  ∀ (p q : ℝ),
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I : ℂ) ^ 2 + p * (2 + Complex.I) + q = 0 →
  (2 - Complex.I : ℂ) ^ 2 + p * (2 - Complex.I) + q = 0 →
  p = -4 ∧ q = 5 := by
  sorry

end quadratic_roots_to_coefficients_l1944_194485


namespace sleepy_squirrel_stockpile_l1944_194499

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts stockpiled per day by each busy squirrel -/
def nuts_per_busy_squirrel : ℕ := 30

/-- The number of days the squirrels have been stockpiling -/
def days_stockpiling : ℕ := 40

/-- The total number of nuts in Mason's car -/
def total_nuts : ℕ := 3200

/-- The number of nuts stockpiled per day by the sleepy squirrel -/
def sleepy_squirrel_nuts : ℕ := 20

theorem sleepy_squirrel_stockpile :
  busy_squirrels * nuts_per_busy_squirrel * days_stockpiling + 
  sleepy_squirrel_nuts * days_stockpiling = total_nuts :=
by sorry

end sleepy_squirrel_stockpile_l1944_194499


namespace choose_formula_l1944_194417

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k ≤ n then
    Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  else
    0

/-- Theorem: The number of ways to choose k items from n items is given by n! / (k!(n-k)!) -/
theorem choose_formula (n k : ℕ) (h : k ≤ n) :
  choose n k = Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) := by
  sorry

end choose_formula_l1944_194417


namespace g_sum_equal_164_l1944_194465

def g (x : ℝ) : ℝ := 2 * x^6 - 5 * x^4 + 7 * x^2 + 6

theorem g_sum_equal_164 (h : g 15 = 82) : g 15 + g (-15) = 164 := by
  sorry

end g_sum_equal_164_l1944_194465


namespace flower_bed_perimeter_reduction_l1944_194441

/-- Represents a rectangular flower bed with length and width -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular flower bed -/
def perimeter (fb : FlowerBed) : ℝ := 2 * (fb.length + fb.width)

/-- Theorem: The perimeter of a rectangular flower bed decreases by 17.5% 
    after reducing the length by 28% and the width by 28% -/
theorem flower_bed_perimeter_reduction (fb : FlowerBed) :
  let reduced_fb := FlowerBed.mk (fb.length * 0.72) (fb.width * 0.72)
  (perimeter fb - perimeter reduced_fb) / perimeter fb = 0.175 := by
  sorry

end flower_bed_perimeter_reduction_l1944_194441


namespace first_week_pushups_l1944_194434

theorem first_week_pushups (initial_pushups : ℕ) (daily_increase : ℕ) (workout_days : ℕ) : 
  initial_pushups = 10 →
  daily_increase = 5 →
  workout_days = 3 →
  (initial_pushups + (initial_pushups + daily_increase) + (initial_pushups + 2 * daily_increase)) = 45 := by
  sorry

end first_week_pushups_l1944_194434


namespace point_on_y_axis_l1944_194459

/-- A point M with coordinates (a+2, 2a-5) lies on the y-axis. -/
theorem point_on_y_axis (a : ℝ) : (a + 2 = 0) → a = -2 := by
  sorry

end point_on_y_axis_l1944_194459


namespace angle_between_diagonals_is_133_l1944_194461

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the angles of the quadrilateral
def angle_ABC (q : Quadrilateral) : ℝ := 116
def angle_ADC (q : Quadrilateral) : ℝ := 64
def angle_CAB (q : Quadrilateral) : ℝ := 35
def angle_CAD (q : Quadrilateral) : ℝ := 52

-- Define the angle between diagonals subtended by side AB
def angle_between_diagonals (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem angle_between_diagonals_is_133 (q : Quadrilateral) : 
  angle_between_diagonals q = 133 := by sorry

end angle_between_diagonals_is_133_l1944_194461


namespace peanuts_in_jar_l1944_194419

theorem peanuts_in_jar (initial_peanuts : ℕ) (brock_fraction : ℚ) (bonita_fraction : ℚ) (carlos_peanuts : ℕ) : 
  initial_peanuts = 220 →
  brock_fraction = 1/4 →
  bonita_fraction = 2/5 →
  carlos_peanuts = 17 →
  initial_peanuts - 
    (initial_peanuts * brock_fraction).floor - 
    ((initial_peanuts - (initial_peanuts * brock_fraction).floor) * bonita_fraction).floor - 
    carlos_peanuts = 82 := by
  sorry

end peanuts_in_jar_l1944_194419


namespace total_cards_packed_l1944_194495

/-- The number of cards in a standard playing card deck -/
def playing_cards_per_deck : ℕ := 52

/-- The number of cards in a Pinochle deck -/
def pinochle_cards_per_deck : ℕ := 48

/-- The number of cards in a Tarot deck -/
def tarot_cards_per_deck : ℕ := 78

/-- The number of cards in an Uno deck -/
def uno_cards_per_deck : ℕ := 108

/-- The number of playing card decks Elijah packed -/
def playing_card_decks : ℕ := 6

/-- The number of Pinochle decks Elijah packed -/
def pinochle_decks : ℕ := 4

/-- The number of Tarot decks Elijah packed -/
def tarot_decks : ℕ := 2

/-- The number of Uno decks Elijah packed -/
def uno_decks : ℕ := 3

/-- Theorem stating the total number of cards Elijah packed -/
theorem total_cards_packed : 
  playing_card_decks * playing_cards_per_deck + 
  pinochle_decks * pinochle_cards_per_deck + 
  tarot_decks * tarot_cards_per_deck + 
  uno_decks * uno_cards_per_deck = 984 := by
  sorry

end total_cards_packed_l1944_194495


namespace ceiling_sum_sqrt_l1944_194470

theorem ceiling_sum_sqrt : ⌈Real.sqrt 5⌉ + ⌈Real.sqrt 50⌉ + ⌈Real.sqrt 500⌉ + ⌈Real.sqrt 1000⌉ = 66 := by
  sorry

end ceiling_sum_sqrt_l1944_194470


namespace marble_distribution_l1944_194402

theorem marble_distribution (n : ℕ) (x : ℕ) :
  (∀ i : ℕ, i ≤ n → i + (n * x - (i * (i + 1)) / 2) / 10 = x) →
  (n * x = n * x - (n * (n + 1)) / 2) →
  (n = 9 ∧ x = 9) := by
  sorry

end marble_distribution_l1944_194402


namespace log_problem_l1944_194447

theorem log_problem (x k : ℝ) 
  (h1 : Real.log x * (Real.log 10 / Real.log k) = 4)
  (h2 : k^2 = 100) : 
  x = 10000 := by
  sorry

end log_problem_l1944_194447


namespace volume_to_surface_area_ratio_l1944_194416

/-- Represents a shape formed by unit cubes -/
structure CubeShape where
  cubes : ℕ
  central_cube : Bool
  surrounding_cubes : ℕ

/-- Calculates the volume of the shape -/
def volume (shape : CubeShape) : ℕ := shape.cubes

/-- Calculates the surface area of the shape -/
def surface_area (shape : CubeShape) : ℕ :=
  shape.surrounding_cubes * 5

/-- The specific shape described in the problem -/
def problem_shape : CubeShape :=
  { cubes := 8
  , central_cube := true
  , surrounding_cubes := 7 }

theorem volume_to_surface_area_ratio :
  (volume problem_shape : ℚ) / (surface_area problem_shape : ℚ) = 8 / 35 := by
  sorry

end volume_to_surface_area_ratio_l1944_194416


namespace p_necessary_not_sufficient_for_q_l1944_194490

-- Define the conditions
def p (m : ℝ) : Prop := -2 < m ∧ m < -1

def q (m : ℝ) : Prop := 
  ∃ (x y : ℝ), x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
  2 + m > 0 ∧ m + 1 < 0

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ m : ℝ, q m → p m) ∧ 
  (∃ m : ℝ, p m ∧ ¬q m) :=
sorry

end p_necessary_not_sufficient_for_q_l1944_194490


namespace fixed_point_on_all_parabolas_l1944_194483

/-- A parabola of the form y = 4x^2 + 2tx - 3t, where t is a real parameter -/
def parabola (t : ℝ) (x : ℝ) : ℝ := 4 * x^2 + 2 * t * x - 3 * t

/-- The fixed point through which all parabolas pass -/
def fixed_point : ℝ × ℝ := (1, 4)

/-- Theorem stating that the fixed point lies on all parabolas -/
theorem fixed_point_on_all_parabolas :
  ∀ t : ℝ, parabola t (fixed_point.1) = fixed_point.2 := by sorry

end fixed_point_on_all_parabolas_l1944_194483


namespace union_of_M_and_N_l1944_194409

def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {1, 3, 4}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3, 4} := by sorry

end union_of_M_and_N_l1944_194409


namespace parabola_point_coordinates_l1944_194413

/-- A point on a parabola with a specific distance to its directrix -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2*x
  distance_to_directrix : x + 1/2 = 2

/-- The coordinates of the point are (3/2, ±√3) -/
theorem parabola_point_coordinates (p : ParabolaPoint) : 
  p.x = 3/2 ∧ (p.y = Real.sqrt 3 ∨ p.y = -Real.sqrt 3) := by
  sorry

end parabola_point_coordinates_l1944_194413


namespace chord_length_l1944_194425

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r ^ 2 - d ^ 2)
  chord_length = 6 := by sorry

end chord_length_l1944_194425


namespace donut_calculation_l1944_194421

def total_donuts (initial_friends : ℕ) (additional_friends : ℕ) (donuts_per_friend : ℕ) (extra_donuts : ℕ) : ℕ :=
  let total_friends := initial_friends + additional_friends
  let donuts_for_friends := total_friends * (donuts_per_friend + extra_donuts)
  let donuts_for_andrew := donuts_per_friend + extra_donuts
  donuts_for_friends + donuts_for_andrew

theorem donut_calculation :
  total_donuts 2 2 3 1 = 20 := by
  sorry

end donut_calculation_l1944_194421


namespace prime_value_theorem_l1944_194466

theorem prime_value_theorem (n : ℕ+) (h : Nat.Prime (n^4 - 16*n^2 + 100)) : n = 3 := by
  sorry

end prime_value_theorem_l1944_194466


namespace smallest_c_for_all_real_domain_l1944_194454

theorem smallest_c_for_all_real_domain : ∃ c : ℤ, 
  (∀ x : ℝ, (x^2 + c*x + 15 ≠ 0)) ∧ 
  (∀ k : ℤ, k < c → ∃ x : ℝ, x^2 + k*x + 15 = 0) ∧
  c = -7 :=
sorry

end smallest_c_for_all_real_domain_l1944_194454


namespace power_of_three_equality_l1944_194477

theorem power_of_three_equality : 3^1999 - 3^1998 - 3^1997 + 3^1996 = 16 * 3^1996 := by
  sorry

end power_of_three_equality_l1944_194477


namespace quadrilateral_sum_of_squares_bounds_l1944_194476

/-- Represents a point on the side of a rectangle -/
structure SidePoint where
  side : Fin 4  -- 0: top, 1: right, 2: bottom, 3: left
  position : ℝ
  h_position : 0 ≤ position ∧ position ≤ match side with
    | 0 | 2 => 3  -- top and bottom sides
    | 1 | 3 => 4  -- right and left sides

/-- The quadrilateral formed by four points on the sides of a 3x4 rectangle -/
def Quadrilateral (p₁ p₂ p₃ p₄ : SidePoint) : Prop :=
  p₁.side ≠ p₂.side ∧ p₂.side ≠ p₃.side ∧ p₃.side ≠ p₄.side ∧ p₄.side ≠ p₁.side

/-- The side length of the quadrilateral between two points -/
def sideLength (p₁ p₂ : SidePoint) : ℝ :=
  sorry  -- Definition of side length calculation

/-- The sum of squares of side lengths of the quadrilateral -/
def sumOfSquares (p₁ p₂ p₃ p₄ : SidePoint) : ℝ :=
  (sideLength p₁ p₂)^2 + (sideLength p₂ p₃)^2 + (sideLength p₃ p₄)^2 + (sideLength p₄ p₁)^2

/-- The main theorem -/
theorem quadrilateral_sum_of_squares_bounds
  (p₁ p₂ p₃ p₄ : SidePoint)
  (h : Quadrilateral p₁ p₂ p₃ p₄) :
  25 ≤ sumOfSquares p₁ p₂ p₃ p₄ ∧ sumOfSquares p₁ p₂ p₃ p₄ ≤ 50 := by
  sorry

end quadrilateral_sum_of_squares_bounds_l1944_194476


namespace min_m_for_log_triangle_l1944_194486

/-- The minimum value of M such that for any right-angled triangle with sides a, b, c > M,
    the logarithms of these sides also form a triangle. -/
theorem min_m_for_log_triangle : ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ (a b c : ℝ), a > M → b > M → c > M →
    a^2 + b^2 = c^2 →
    Real.log a + Real.log b > Real.log c) ∧
  (∀ (M' : ℝ), M' < M →
    ∃ (a b c : ℝ), a > M' ∧ b > M' ∧ c > M' ∧
      a^2 + b^2 = c^2 ∧
      Real.log a + Real.log b ≤ Real.log c) :=
by sorry

end min_m_for_log_triangle_l1944_194486


namespace smallest_n_not_divisible_l1944_194475

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_n_not_divisible : ∃ (n : ℕ), n = 124 ∧ 
  ¬(factorial 1999 ∣ 34^n * factorial n) ∧ 
  ∀ (m : ℕ), m < n → (factorial 1999 ∣ 34^m * factorial m) :=
sorry

end smallest_n_not_divisible_l1944_194475


namespace f_range_and_max_value_l1944_194458

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1)*x - 3

theorem f_range_and_max_value :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f 2 x ∈ Set.Icc (-21/4 : ℝ) 15) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc (-1 : ℝ) 3, f a x ≤ 1) ∧ 
            (∃ x ∈ Set.Icc (-1 : ℝ) 3, f a x = 1) →
            a = -1/3 ∨ a = -1) :=
by sorry

end f_range_and_max_value_l1944_194458


namespace failed_students_l1944_194481

/-- The number of students who failed an examination, given the total number of students and the percentage who passed. -/
theorem failed_students (total : ℕ) (pass_percent : ℚ) : 
  total = 700 → pass_percent = 35 / 100 → 
  (total : ℚ) * (1 - pass_percent) = 455 := by
  sorry

end failed_students_l1944_194481


namespace paper_cutting_theorem_l1944_194463

/-- Represents a polygon --/
structure Polygon where
  vertices : ℕ

/-- Represents the state of the paper after cuts --/
structure PaperState where
  polygons : List Polygon
  totalVertices : ℕ

/-- Initial state of the rectangular paper --/
def initialState : PaperState :=
  { polygons := [{ vertices := 4 }], totalVertices := 4 }

/-- Perform a single cut on the paper state --/
def performCut (state : PaperState) : PaperState :=
  { polygons := state.polygons ++ [{ vertices := 3 }],
    totalVertices := state.totalVertices + 2 }

/-- Perform n cuts on the paper state --/
def performCuts (n : ℕ) (state : PaperState) : PaperState :=
  match n with
  | 0 => state
  | n + 1 => performCuts n (performCut state)

/-- The main theorem to prove --/
theorem paper_cutting_theorem :
  (performCuts 100 initialState).totalVertices ≠ 302 :=
by sorry

end paper_cutting_theorem_l1944_194463


namespace digit_sum_congruence_l1944_194468

/-- The digit sum of n in base r -/
noncomputable def digit_sum (r n : ℕ) : ℕ := sorry

theorem digit_sum_congruence :
  (∀ r > 2, ∃ p : ℕ, Nat.Prime p ∧ ∀ n > 0, digit_sum r n ≡ n [MOD p]) ∧
  (∀ r > 1, ∀ p : ℕ, Nat.Prime p → ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, digit_sum r n ≡ n [MOD p]) :=
sorry

end digit_sum_congruence_l1944_194468


namespace circle_area_from_polar_equation_l1944_194436

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop :=
  r = 4 * Real.cos θ - 3 * Real.sin θ

-- State the theorem
theorem circle_area_from_polar_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (r θ : ℝ), polar_equation r θ ↔ 
      (r * Real.cos θ - center.1)^2 + (r * Real.sin θ - center.2)^2 = radius^2) ∧
    (π * radius^2 = 25 * π / 4) :=
sorry

end circle_area_from_polar_equation_l1944_194436


namespace vector_on_line_and_parallel_l1944_194472

def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 1, 2 * t + 3)

def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_on_line_and_parallel : 
  ∃ t : ℝ, line_param t = (16, 32/3) ∧ 
  is_parallel (16, 32/3) (3, 2) := by
  sorry

end vector_on_line_and_parallel_l1944_194472


namespace complex_product_one_plus_i_one_minus_i_l1944_194448

theorem complex_product_one_plus_i_one_minus_i : (1 + Complex.I) * (1 - Complex.I) = 2 := by
  sorry

end complex_product_one_plus_i_one_minus_i_l1944_194448


namespace largest_package_size_l1944_194474

theorem largest_package_size (mary_markers luis_markers ali_markers : ℕ) 
  (h1 : mary_markers = 36)
  (h2 : luis_markers = 45)
  (h3 : ali_markers = 75) :
  Nat.gcd mary_markers (Nat.gcd luis_markers ali_markers) = 3 := by
  sorry

end largest_package_size_l1944_194474


namespace line_through_parabola_vertex_l1944_194464

-- Define the parabola
def parabola (a x : ℝ) : ℝ := x^3 - 3*a*x + a^3

-- Define the line
def line (a x : ℝ) : ℝ := x + 2*a

-- Define the derivative of the parabola with respect to x
def parabola_derivative (a x : ℝ) : ℝ := 3*x^2 - 3*a

-- Theorem statement
theorem line_through_parabola_vertex :
  ∃! (a : ℝ), ∃ (x : ℝ),
    (parabola_derivative a x = 0) ∧
    (line a x = parabola a x) :=
sorry

end line_through_parabola_vertex_l1944_194464


namespace green_shirt_pairs_l1944_194408

theorem green_shirt_pairs (total_students : ℕ) (red_students : ℕ) (green_students : ℕ) 
  (total_pairs : ℕ) (red_pairs : ℕ) :
  total_students = 148 →
  red_students = 65 →
  green_students = 83 →
  total_pairs = 74 →
  red_pairs = 27 →
  red_students + green_students = total_students →
  2 * total_pairs = total_students →
  ∃ (green_pairs : ℕ), green_pairs = 36 ∧ 
    red_pairs + green_pairs + (total_students - 2 * (red_pairs + green_pairs)) / 2 = total_pairs :=
by sorry

end green_shirt_pairs_l1944_194408


namespace ratio_problem_l1944_194422

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 4) : 
  (a + b) / (b + c) = 4/15 := by sorry

end ratio_problem_l1944_194422


namespace max_quarters_problem_l1944_194404

theorem max_quarters_problem :
  ∃! q : ℕ, 8 < q ∧ q < 60 ∧
  q % 4 = 2 ∧
  q % 7 = 3 ∧
  q % 9 = 2 ∧
  q = 38 :=
by sorry

end max_quarters_problem_l1944_194404


namespace point_D_coordinates_l1944_194420

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℚ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

/-- Checks if a point is on a line segment -/
def isOnSegment (p : Point) (a b : Point) : Prop :=
  distance a p + distance p b = distance a b

theorem point_D_coordinates :
  let X : Point := ⟨-2, 1⟩
  let Y : Point := ⟨4, 9⟩
  ∀ D : Point,
    isOnSegment D X Y →
    distance X D = 2 * distance Y D →
    D.x = 2 ∧ D.y = 19 / 3 := by
  sorry

end point_D_coordinates_l1944_194420


namespace annie_distance_equals_22_l1944_194497

def base_fare : ℚ := 2.5
def per_mile_rate : ℚ := 0.25
def mike_distance : ℚ := 42
def annie_toll : ℚ := 5

theorem annie_distance_equals_22 :
  ∃ (annie_distance : ℚ),
    base_fare + per_mile_rate * mike_distance =
    base_fare + annie_toll + per_mile_rate * annie_distance ∧
    annie_distance = 22 := by
  sorry

end annie_distance_equals_22_l1944_194497


namespace ashleys_friends_ages_sum_l1944_194440

theorem ashleys_friends_ages_sum :
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    0 < d ∧ d < 10 →
    (a * b = 36 ∧ c * d = 30) ∨ (a * c = 36 ∧ b * d = 30) ∨ (a * d = 36 ∧ b * c = 30) →
    a + b + c + d = 24 :=
by sorry

end ashleys_friends_ages_sum_l1944_194440


namespace overtime_threshold_is_40_l1944_194455

/-- Represents Janet's work and financial situation -/
structure JanetWorkSituation where
  regularRate : ℝ  -- Regular hourly rate
  weeklyHours : ℝ  -- Total weekly work hours
  overtimeMultiplier : ℝ  -- Overtime pay multiplier
  carCost : ℝ  -- Cost of the car
  weeksToSave : ℝ  -- Number of weeks to save for the car

/-- Calculates the weekly earnings given a threshold for overtime hours -/
def weeklyEarnings (j : JanetWorkSituation) (threshold : ℝ) : ℝ :=
  threshold * j.regularRate + (j.weeklyHours - threshold) * j.regularRate * j.overtimeMultiplier

/-- Theorem stating that the overtime threshold is 40 hours -/
theorem overtime_threshold_is_40 (j : JanetWorkSituation) 
    (h1 : j.regularRate = 20)
    (h2 : j.weeklyHours = 52)
    (h3 : j.overtimeMultiplier = 1.5)
    (h4 : j.carCost = 4640)
    (h5 : j.weeksToSave = 4)
    : ∃ (threshold : ℝ), 
      threshold = 40 ∧ 
      weeklyEarnings j threshold ≥ j.carCost / j.weeksToSave ∧
      ∀ t, t > threshold → weeklyEarnings j t < j.carCost / j.weeksToSave :=
by
  sorry

end overtime_threshold_is_40_l1944_194455


namespace bag_of_balls_l1944_194493

theorem bag_of_balls (white green yellow red purple : ℕ) 
  (h1 : white = 20)
  (h2 : green = 30)
  (h3 : yellow = 10)
  (h4 : red = 37)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 3/5) :
  white + green + yellow + red + purple = 100 := by
  sorry

end bag_of_balls_l1944_194493


namespace opposite_of_negative_three_fourths_l1944_194462

theorem opposite_of_negative_three_fourths :
  ∀ x : ℚ, x + (-3/4) = 0 → x = 3/4 := by
  sorry

end opposite_of_negative_three_fourths_l1944_194462


namespace combination_equality_l1944_194428

theorem combination_equality (n : ℕ) : 
  Nat.choose n 14 = Nat.choose n 4 → n = 18 := by
  sorry

end combination_equality_l1944_194428


namespace store_profit_percentage_l1944_194401

/-- Proves that the profit percentage is 30% given the conditions of the problem -/
theorem store_profit_percentage (cost_price : ℝ) (sale_price : ℝ) :
  cost_price = 20 →
  sale_price = 13 →
  ∃ (selling_price : ℝ),
    selling_price = cost_price * (1 + 30 / 100) ∧
    sale_price = selling_price / 2 :=
by sorry

end store_profit_percentage_l1944_194401


namespace shaded_area_square_with_circles_l1944_194453

/-- The area of the shaded region in a square with circles at its vertices -/
theorem shaded_area_square_with_circles (s : ℝ) (r : ℝ) (h1 : s = 12) (h2 : r = 2) :
  s^2 - (4 * (π / 2 * r^2) + 4 * (r^2 / 2)) = 136 - 2 * π := by sorry

end shaded_area_square_with_circles_l1944_194453


namespace vector_sum_magnitude_l1944_194424

theorem vector_sum_magnitude : 
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (0, 1)
  ‖a + 2 • b‖ = Real.sqrt 5 := by
  sorry

end vector_sum_magnitude_l1944_194424


namespace jack_morning_emails_l1944_194450

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- The difference between afternoon and morning emails -/
def email_difference : ℕ := 2

theorem jack_morning_emails : 
  morning_emails = 6 ∧ 
  afternoon_emails = morning_emails + email_difference := by
  sorry

end jack_morning_emails_l1944_194450


namespace log_50_between_integers_l1944_194467

theorem log_50_between_integers : ∃ c d : ℤ, (c : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < (d : ℝ) ∧ c + d = 3 := by
  sorry

end log_50_between_integers_l1944_194467


namespace polynomial_evaluation_l1944_194498

theorem polynomial_evaluation :
  let x : ℤ := -2
  x^4 + x^3 + x^2 + x + 1 = 11 := by
  sorry

end polynomial_evaluation_l1944_194498


namespace bakery_problem_proof_l1944_194427

/-- Given the total number of muffins, muffins per box, and available boxes, 
    calculate the number of additional boxes needed --/
def additional_boxes_needed (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) : ℕ :=
  (total_muffins / muffins_per_box) - available_boxes

/-- Proof that 9 additional boxes are needed for the given bakery problem --/
theorem bakery_problem_proof : 
  additional_boxes_needed 95 5 10 = 9 := by
  sorry

end bakery_problem_proof_l1944_194427


namespace fuel_mixture_problem_l1944_194410

/-- Proves that given a 204-gallon tank filled partially with fuel A (12% ethanol)
    and then to capacity with fuel B (16% ethanol), if the full tank contains
    30 gallons of ethanol, then the volume of fuel A added is 66 gallons. -/
theorem fuel_mixture_problem (x : ℝ) : 
  (0.12 * x + 0.16 * (204 - x) = 30) → x = 66 := by
  sorry

end fuel_mixture_problem_l1944_194410


namespace equation_solution_l1944_194480

theorem equation_solution (x : ℝ) :
  (5.31 * Real.tan (6 * x) * Real.cos (2 * x) - Real.sin (2 * x) - 2 * Real.sin (4 * x) = 0) ↔
  (∃ k : ℤ, x = k * π / 2) ∨ (∃ k : ℤ, x = π / 18 * (6 * k + 1) ∨ x = π / 18 * (6 * k - 1)) :=
by sorry

end equation_solution_l1944_194480


namespace three_true_propositions_l1944_194457

theorem three_true_propositions : 
  (¬∀ (a b c : ℝ), a > b → a * c < b * c) ∧ 
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b : ℝ), a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) ∧
  (∀ (a b : ℝ), a > b ∧ 1 / a > 1 / b → a > 0 ∧ b < 0) :=
by sorry

end three_true_propositions_l1944_194457


namespace inequality_solution_l1944_194496

theorem inequality_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 3) (h4 : x ≠ 4) :
  (1 / (x - 1) - 3 / (x - 2) + 3 / (x - 3) - 1 / (x - 4) < 1 / 24) ↔
  (x < -2 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 6)) :=
by sorry


end inequality_solution_l1944_194496


namespace root_sum_reciprocal_l1944_194405

def f (x : ℝ) := x^3 + 3*x - 1

theorem root_sum_reciprocal (a b c : ℝ) (m n : ℕ) :
  f a = 0 → f b = 0 → f c = 0 →
  (1 / (a^3 + b^3) + 1 / (b^3 + c^3) + 1 / (c^3 + a^3) : ℝ) = m / n →
  m > 0 → n > 0 →
  Nat.gcd m n = 1 →
  100 * m + n = 3989 := by
sorry

end root_sum_reciprocal_l1944_194405


namespace problem_solution_l1944_194426

theorem problem_solution : ∃ x : ℝ, 
  ((35 * x)^2 / 100) * x = (23/18) / 100 * 9500 - 175 ∧ 
  abs (x + 0.62857) < 0.00001 := by
sorry

end problem_solution_l1944_194426


namespace rope_cutting_problem_l1944_194414

theorem rope_cutting_problem : Nat.gcd 42 (Nat.gcd 56 (Nat.gcd 63 77)) = 7 := by
  sorry

end rope_cutting_problem_l1944_194414


namespace guppies_count_l1944_194435

-- Define the number of guppies each person has
def haylee_guppies : ℕ := 3 * 12 -- 3 dozen
def jose_guppies : ℕ := haylee_guppies / 2
def charliz_guppies : ℕ := jose_guppies / 3
def nicolai_guppies : ℕ := charliz_guppies * 4

-- Define the total number of guppies
def total_guppies : ℕ := haylee_guppies + jose_guppies + charliz_guppies + nicolai_guppies

-- Theorem to prove
theorem guppies_count : total_guppies = 84 := by
  sorry

end guppies_count_l1944_194435


namespace james_printing_sheets_l1944_194456

theorem james_printing_sheets (num_books : ℕ) (pages_per_book : ℕ) (pages_per_side : ℕ) :
  num_books = 2 →
  pages_per_book = 600 →
  pages_per_side = 4 →
  (num_books * pages_per_book) / (2 * pages_per_side) = 150 := by
  sorry

end james_printing_sheets_l1944_194456


namespace top_square_is_14_l1944_194449

/-- Represents a position in the 4x4 grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the 4x4 grid -/
def Grid := Fin 4 → Fin 4 → ℕ

/-- Initial configuration of the grid -/
def initialGrid : Grid :=
  λ i j => i.val * 4 + j.val + 1

/-- Fold right half over left half -/
def foldRight (g : Grid) : Grid :=
  λ i j => g i (3 - j)

/-- Fold bottom half over top half -/
def foldBottom (g : Grid) : Grid :=
  λ i j => g (3 - i) j

/-- Apply all folding operations -/
def applyFolds (g : Grid) : Grid :=
  foldRight (foldBottom (foldRight g))

/-- The position of the top square after folding -/
def topPosition : Position :=
  ⟨0, 0⟩

/-- Theorem: After folding, the top square was originally numbered 14 -/
theorem top_square_is_14 :
  applyFolds initialGrid topPosition.row topPosition.col = 14 := by
  sorry

end top_square_is_14_l1944_194449


namespace company_average_salary_l1944_194432

/-- Calculates the average salary for a company given the number of managers,
    number of associates, average salary of managers, and average salary of associates. -/
def average_company_salary (num_managers : ℕ) (num_associates : ℕ) (avg_salary_managers : ℚ) (avg_salary_associates : ℚ) : ℚ :=
  let total_employees := num_managers + num_associates
  let total_salary := num_managers * avg_salary_managers + num_associates * avg_salary_associates
  total_salary / total_employees

/-- Theorem stating that the average salary for the company is $40,000 -/
theorem company_average_salary :
  average_company_salary 15 75 90000 30000 = 40000 := by
  sorry

end company_average_salary_l1944_194432


namespace fraction_subtraction_l1944_194433

theorem fraction_subtraction : (8 : ℚ) / 24 - (5 : ℚ) / 40 = (5 : ℚ) / 24 := by
  sorry

end fraction_subtraction_l1944_194433


namespace number_thought_of_l1944_194437

theorem number_thought_of (x : ℚ) : (6 * x) / 2 - 5 = 25 → x = 10 := by
  sorry

end number_thought_of_l1944_194437


namespace product_ratio_theorem_l1944_194406

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1/4 := by
  sorry

end product_ratio_theorem_l1944_194406


namespace sin_72_cos_18_plus_cos_72_sin_18_l1944_194471

theorem sin_72_cos_18_plus_cos_72_sin_18 : 
  Real.sin (72 * π / 180) * Real.cos (18 * π / 180) + 
  Real.cos (72 * π / 180) * Real.sin (18 * π / 180) = 1 := by
  sorry

end sin_72_cos_18_plus_cos_72_sin_18_l1944_194471


namespace dinner_cakes_l1944_194443

def total_cakes : ℕ := 15
def lunch_cakes : ℕ := 6

theorem dinner_cakes : total_cakes - lunch_cakes = 9 := by
  sorry

end dinner_cakes_l1944_194443


namespace fourth_term_coefficient_implies_n_l1944_194487

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the r-th term in the binomial expansion of (√x - 1/2x)^n -/
def coefficient (n r : ℕ) : ℚ :=
  (binomial n r : ℚ) * (-1/2)^r

theorem fourth_term_coefficient_implies_n (n : ℕ) :
  coefficient n 3 = -7 → n = 8 := by sorry

end fourth_term_coefficient_implies_n_l1944_194487


namespace friends_fireworks_count_l1944_194460

/-- The number of fireworks Henry bought -/
def henrys_fireworks : ℕ := 2

/-- The number of fireworks saved from last year -/
def saved_fireworks : ℕ := 6

/-- The total number of fireworks they have now -/
def total_fireworks : ℕ := 11

/-- The number of fireworks Henry's friend bought -/
def friends_fireworks : ℕ := total_fireworks - (henrys_fireworks + saved_fireworks)

theorem friends_fireworks_count : friends_fireworks = 3 := by
  sorry

end friends_fireworks_count_l1944_194460


namespace inequality_proof_l1944_194418

theorem inequality_proof (b a : ℝ) : 
  (4 * b^2 * (b^3 - 1) - 3 * (1 - 2 * b^2) > 4 * (b^5 - 1)) ∧ 
  (a - a * |(-a^2 - 1)| < 1 - a^2 * (a - 1)) := by
  sorry

end inequality_proof_l1944_194418
