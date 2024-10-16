import Mathlib

namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l4033_403369

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def total_cost : ℚ := 33.56

theorem jacket_cost_calculation :
  ∃ (jacket_cost : ℚ), 
    jacket_cost = total_cost - (shorts_cost + shirt_cost) ∧
    jacket_cost = 7.43 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l4033_403369


namespace NUMINAMATH_CALUDE_select_and_arrange_theorem_l4033_403399

/-- The number of ways to select k items from n items -/
def combination (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange k items -/
def permutation (k : ℕ) : ℕ := Nat.factorial k

/-- The total number of people -/
def total_people : ℕ := 9

/-- The number of people to be selected and arranged -/
def selected_people : ℕ := 3

/-- The number of ways to select and arrange people -/
def ways_to_select_and_arrange : ℕ := combination total_people selected_people * permutation selected_people

theorem select_and_arrange_theorem : ways_to_select_and_arrange = 504 := by
  sorry

end NUMINAMATH_CALUDE_select_and_arrange_theorem_l4033_403399


namespace NUMINAMATH_CALUDE_coefficient_proof_l4033_403379

theorem coefficient_proof (n : ℤ) :
  (∃! (count : ℕ), count = 25 ∧
    count = (Finset.filter (fun i => 1 < 4 * i + 7 ∧ 4 * i + 7 < 100) (Finset.range 200)).card) →
  ∃ (a : ℤ), ∀ (x : ℤ), (a * x + 7 = 4 * x + 7) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_proof_l4033_403379


namespace NUMINAMATH_CALUDE_coffee_shop_tables_l4033_403361

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- The number of people that can be seated in the coffee shop (in base 7) -/
def seatingCapacityBase7 : Nat := 321

/-- The number of people that sit at one table -/
def peoplePerTable : Nat := 3

theorem coffee_shop_tables :
  (base7ToBase10 seatingCapacityBase7) / peoplePerTable = 54 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_tables_l4033_403361


namespace NUMINAMATH_CALUDE_smallest_possible_b_l4033_403322

theorem smallest_possible_b : ∃ (b : ℝ), b = 2 ∧ 
  (∀ (a : ℝ), (2 < a ∧ a < b) → 
    (2 + a ≤ b ∧ 1/a + 1/b ≤ 1)) ∧
  (∀ (b' : ℝ), 2 < b' → 
    (∃ (a : ℝ), (2 < a ∧ a < b') ∧ 
      (2 + a > b' ∨ 1/a + 1/b' > 1))) :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l4033_403322


namespace NUMINAMATH_CALUDE_artistic_parents_l4033_403356

theorem artistic_parents (total : ℕ) (dad : ℕ) (mom : ℕ) (both : ℕ) : 
  total = 40 → dad = 18 → mom = 20 → both = 11 →
  total - (dad + mom - both) = 13 := by
sorry

end NUMINAMATH_CALUDE_artistic_parents_l4033_403356


namespace NUMINAMATH_CALUDE_twelveRowTriangle_l4033_403330

/-- Calculates the sum of an arithmetic progression -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Represents the triangle construction -/
structure TriangleConstruction where
  rows : ℕ
  firstRowRods : ℕ
  rodIncreasePerRow : ℕ

/-- Calculates the total number of pieces in the triangle construction -/
def totalPieces (t : TriangleConstruction) : ℕ :=
  let rodSum := arithmeticSum t.firstRowRods t.rodIncreasePerRow t.rows
  let connectorSum := arithmeticSum 1 1 (t.rows + 1)
  rodSum + connectorSum

/-- Theorem statement for the 12-row triangle construction -/
theorem twelveRowTriangle :
  totalPieces { rows := 12, firstRowRods := 3, rodIncreasePerRow := 3 } = 325 := by
  sorry


end NUMINAMATH_CALUDE_twelveRowTriangle_l4033_403330


namespace NUMINAMATH_CALUDE_quadratic_equation_from_root_properties_l4033_403328

theorem quadratic_equation_from_root_properties (a b c : ℝ) (h_sum : b / a = -4) (h_product : c / a = 3) :
  ∃ (k : ℝ), k ≠ 0 ∧ k * (a * X^2 + b * X + c) = X^2 - 4*X + 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_root_properties_l4033_403328


namespace NUMINAMATH_CALUDE_inequality_solutions_l4033_403394

/-- The solution set of the inequality 2x^2 + x - 3 < 0 -/
def solution_set_1 : Set ℝ := { x | -3/2 < x ∧ x < 1 }

/-- The solution set of the inequality x(9 - x) > 0 -/
def solution_set_2 : Set ℝ := { x | 0 < x ∧ x < 9 }

theorem inequality_solutions :
  (∀ x : ℝ, x ∈ solution_set_1 ↔ 2*x^2 + x - 3 < 0) ∧
  (∀ x : ℝ, x ∈ solution_set_2 ↔ x*(9 - x) > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l4033_403394


namespace NUMINAMATH_CALUDE_factor_implication_l4033_403344

theorem factor_implication (m n : ℝ) : 
  (∃ a b : ℝ, 3 * X^3 - m * X + n = a * (X - 3) * (X + 1) * X) →
  |3 * m + n| = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_factor_implication_l4033_403344


namespace NUMINAMATH_CALUDE_tax_base_amount_theorem_l4033_403377

/-- Calculates the base amount given the tax rate and tax amount -/
def calculate_base_amount (tax_rate : ℚ) (tax_amount : ℚ) : ℚ :=
  tax_amount / (tax_rate / 100)

/-- Theorem: Given a tax rate of 65% and a tax amount of $65, the base amount is $100 -/
theorem tax_base_amount_theorem :
  let tax_rate : ℚ := 65
  let tax_amount : ℚ := 65
  calculate_base_amount tax_rate tax_amount = 100 := by
  sorry

#eval calculate_base_amount 65 65

end NUMINAMATH_CALUDE_tax_base_amount_theorem_l4033_403377


namespace NUMINAMATH_CALUDE_ball_box_probability_l4033_403360

/-- The number of ways to place 5 balls in 4 boxes with no empty box -/
def total_arrangements : ℕ := 240

/-- The number of ways to place 5 balls in 4 boxes with no empty box and no ball in a box with the same label -/
def valid_arrangements : ℕ := 84

/-- The probability of placing 5 balls in 4 boxes with no empty box and no ball in a box with the same label -/
def probability : ℚ := valid_arrangements / total_arrangements

theorem ball_box_probability : probability = 7 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ball_box_probability_l4033_403360


namespace NUMINAMATH_CALUDE_hundredth_card_is_ninth_l4033_403333

/-- Represents the cyclic order of cards in a standard deck --/
def cardCycle : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

/-- The number of cards in a cycle --/
def cycleLength : ℕ := 13

/-- The position we want to find --/
def targetPosition : ℕ := 100

/-- Function to get the equivalent position in the cycle --/
def cyclicPosition (n : ℕ) : ℕ :=
  (n - 1) % cycleLength + 1

theorem hundredth_card_is_ninth (h : targetPosition = 100) :
  cyclicPosition targetPosition = 9 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_card_is_ninth_l4033_403333


namespace NUMINAMATH_CALUDE_second_number_in_first_set_l4033_403317

theorem second_number_in_first_set (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 70 + 16) / 3 + 8 ↔ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_number_in_first_set_l4033_403317


namespace NUMINAMATH_CALUDE_complement_union_theorem_l4033_403353

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l4033_403353


namespace NUMINAMATH_CALUDE_no_real_solution_l4033_403308

theorem no_real_solution : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x^2 = 1 + 1/y^2 ∧ y^2 = 1 + 1/x^2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l4033_403308


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l4033_403352

theorem pentagon_area_sum (a b : ℤ) (h1 : 0 < b) (h2 : b < a) :
  let F := (a, b)
  let G := (b, a)
  let H := (-b, a)
  let I := (-b, -a)
  let J := (-a, -b)
  let pentagon_area := (a^2 : ℝ) + 3 * (a * b)
  pentagon_area = 550 → a + b = 24 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l4033_403352


namespace NUMINAMATH_CALUDE_two_digit_powers_of_three_l4033_403325

theorem two_digit_powers_of_three :
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99)) ∧
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99) ∧ Finset.card s = 2) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_powers_of_three_l4033_403325


namespace NUMINAMATH_CALUDE_coefficient_of_z_in_equation1_l4033_403318

-- Define the system of equations
def equation1 (x y z : ℚ) : Prop := 6 * x - 5 * y + z = 22 / 3
def equation2 (x y z : ℚ) : Prop := 4 * x + 8 * y - 11 * z = 7
def equation3 (x y z : ℚ) : Prop := 5 * x - 6 * y + 2 * z = 12

-- Define the sum condition
def sum_condition (x y z : ℚ) : Prop := x + y + z = 10

-- Theorem statement
theorem coefficient_of_z_in_equation1 (x y z : ℚ) 
  (eq1 : equation1 x y z) (eq2 : equation2 x y z) (eq3 : equation3 x y z) 
  (sum : sum_condition x y z) : 
  ∃ (a b c : ℚ), equation1 x y z ↔ a * x + b * y + 1 * z = 22 / 3 :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_z_in_equation1_l4033_403318


namespace NUMINAMATH_CALUDE_exists_participation_to_invalidate_forecast_l4033_403339

/-- Represents a voter in the election -/
structure Voter :=
  (id : Nat)
  (isCandidate : Bool)
  (friends : Set Nat)

/-- Represents a forecast for the election -/
def Forecast := Nat → Nat

/-- Represents the actual votes cast in the election -/
def ActualVotes := Nat → Nat

/-- Determines if a voter participates in the election -/
def VoterParticipation := Nat → Bool

/-- Calculates the actual votes based on voter participation -/
def calculateActualVotes (voters : List Voter) (participation : VoterParticipation) : ActualVotes :=
  sorry

/-- Checks if a forecast is good (correct for at least one candidate) -/
def isGoodForecast (forecast : Forecast) (actualVotes : ActualVotes) : Bool :=
  sorry

/-- Main theorem: For any forecast, there exists a voter participation that makes the forecast not good -/
theorem exists_participation_to_invalidate_forecast (voters : List Voter) (forecast : Forecast) :
  ∃ (participation : VoterParticipation),
    ¬(isGoodForecast forecast (calculateActualVotes voters participation)) :=
  sorry

end NUMINAMATH_CALUDE_exists_participation_to_invalidate_forecast_l4033_403339


namespace NUMINAMATH_CALUDE_max_digit_sum_in_range_l4033_403302

def is_valid_time (h m s : ℕ) : Prop :=
  13 ≤ h ∧ h ≤ 23 ∧ m < 60 ∧ s < 60

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def time_digit_sum (h m s : ℕ) : ℕ :=
  digit_sum h + digit_sum m + digit_sum s

theorem max_digit_sum_in_range :
  ∃ (h m s : ℕ), is_valid_time h m s ∧
    ∀ (h' m' s' : ℕ), is_valid_time h' m' s' →
      time_digit_sum h' m' s' ≤ time_digit_sum h m s ∧
      time_digit_sum h m s = 33 :=
sorry

end NUMINAMATH_CALUDE_max_digit_sum_in_range_l4033_403302


namespace NUMINAMATH_CALUDE_exchange_calculation_l4033_403362

/-- Exchange rate between Canadian and American dollars -/
def exchange_rate : ℚ := 120 / 80

/-- Amount of American dollars to be exchanged -/
def american_dollars : ℚ := 50

/-- Function to calculate Canadian dollars given American dollars -/
def exchange (usd : ℚ) : ℚ := usd * exchange_rate

theorem exchange_calculation :
  exchange american_dollars = 75 := by
  sorry

end NUMINAMATH_CALUDE_exchange_calculation_l4033_403362


namespace NUMINAMATH_CALUDE_range_of_g_l4033_403387

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_g :
  Set.range g = {-π/3, π/3} := by sorry

end NUMINAMATH_CALUDE_range_of_g_l4033_403387


namespace NUMINAMATH_CALUDE_consecutive_sum_product_l4033_403319

theorem consecutive_sum_product (n : ℕ) (h : n > 100) :
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (3 * (n + 1) = a * b * c ∨ 3 * (n + 2) = a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_product_l4033_403319


namespace NUMINAMATH_CALUDE_arithmetic_operations_equal_reciprocal_2016_l4033_403327

theorem arithmetic_operations_equal_reciprocal_2016 :
  (1 / 8 * 1 / 9 * 1 / 28 : ℚ) = 1 / 2016 ∧ 
  ((1 / 8 - 1 / 9) * 1 / 28 : ℚ) = 1 / 2016 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_equal_reciprocal_2016_l4033_403327


namespace NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_16_l4033_403320

theorem cube_root_27_times_fourth_root_16 : (27 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_16_l4033_403320


namespace NUMINAMATH_CALUDE_wavelength_in_scientific_notation_l4033_403370

/-- Converts nanometers to meters -/
def nm_to_m (nm : ℝ) : ℝ := nm * 0.000000001

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem wavelength_in_scientific_notation :
  let wavelength_nm : ℝ := 688
  let wavelength_m : ℝ := nm_to_m wavelength_nm
  let scientific : ScientificNotation := to_scientific_notation wavelength_m
  scientific.coefficient = 6.88 ∧ scientific.exponent = -7 :=
sorry

end NUMINAMATH_CALUDE_wavelength_in_scientific_notation_l4033_403370


namespace NUMINAMATH_CALUDE_distinct_elements_condition_l4033_403386

theorem distinct_elements_condition (x : ℝ) : 
  ({1, x, x^2 - x} : Set ℝ).ncard = 3 ↔ 
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ x ≠ (1 + Real.sqrt 5) / 2 ∧ x ≠ (1 - Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_elements_condition_l4033_403386


namespace NUMINAMATH_CALUDE_farm_area_l4033_403351

/-- Proves that a rectangular farm with given conditions has an area of 1200 square meters -/
theorem farm_area (short_side : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) : 
  short_side = 30 →
  cost_per_meter = 14 →
  total_cost = 1680 →
  ∃ (long_side : ℝ),
    long_side > 0 ∧
    cost_per_meter * (long_side + short_side + Real.sqrt (long_side^2 + short_side^2)) = total_cost ∧
    long_side * short_side = 1200 :=
by
  sorry

#check farm_area

end NUMINAMATH_CALUDE_farm_area_l4033_403351


namespace NUMINAMATH_CALUDE_sequence_properties_l4033_403347

def sequence_a (n : ℕ+) : ℚ := (1 / 3) ^ n.val

def sum_S (n : ℕ+) : ℚ := (1 / 2) * (1 - (1 / 3) ^ n.val)

def arithmetic_sequence_condition (t : ℚ) : Prop :=
  let S₁ := sum_S 1
  let S₂ := sum_S 2
  let S₃ := sum_S 3
  S₁ + 3 * (S₂ + S₃) = 2 * (S₁ + S₂) * t

theorem sequence_properties :
  (∀ n : ℕ+, sum_S (n + 1) - sum_S n = (1 / 3) ^ (n + 1).val) →
  (∀ n : ℕ+, sequence_a n = (1 / 3) ^ n.val) ∧
  (∀ n : ℕ+, sum_S n = (1 / 2) * (1 - (1 / 3) ^ n.val)) ∧
  (∃ t : ℚ, arithmetic_sequence_condition t ∧ t = 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l4033_403347


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4033_403323

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z - 3) = -1 + 3 * Complex.I) : 
  z.im = 1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4033_403323


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l4033_403372

/-- Represents the repeating decimal 0.333... -/
def repeating_third : ℚ := 1 / 3

/-- Proves that 8 divided by 0.333... equals 24 -/
theorem eight_divided_by_repeating_third : 8 / repeating_third = 24 := by sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l4033_403372


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4033_403363

theorem complex_equation_solution (z : ℂ) : 
  z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) →
  z = (1/2 : ℂ) - Complex.I * ((Real.sqrt 3)/2) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4033_403363


namespace NUMINAMATH_CALUDE_exponent_multiplication_l4033_403341

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l4033_403341


namespace NUMINAMATH_CALUDE_irwin_family_hike_distance_l4033_403332

/-- The total distance hiked by Irwin's family during their camping trip. -/
def total_distance_hiked (car_to_stream stream_to_meadow meadow_to_campsite : ℝ) : ℝ :=
  car_to_stream + stream_to_meadow + meadow_to_campsite

/-- Theorem stating that the total distance hiked by Irwin's family is 0.7 miles. -/
theorem irwin_family_hike_distance :
  total_distance_hiked 0.2 0.4 0.1 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_irwin_family_hike_distance_l4033_403332


namespace NUMINAMATH_CALUDE_chess_club_mixed_groups_l4033_403342

/-- Represents the chess club and its game statistics -/
structure ChessClub where
  total_children : ℕ
  total_groups : ℕ
  children_per_group : ℕ
  boy_vs_boy_games : ℕ
  girl_vs_girl_games : ℕ

/-- Calculates the number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ := 
  let total_games := club.total_groups * (club.children_per_group.choose 2)
  let mixed_games := total_games - club.boy_vs_boy_games - club.girl_vs_girl_games
  mixed_games / 2

/-- The main theorem stating the number of mixed groups -/
theorem chess_club_mixed_groups :
  let club : ChessClub := {
    total_children := 90,
    total_groups := 30,
    children_per_group := 3,
    boy_vs_boy_games := 30,
    girl_vs_girl_games := 14
  }
  mixed_groups club = 23 := by sorry

end NUMINAMATH_CALUDE_chess_club_mixed_groups_l4033_403342


namespace NUMINAMATH_CALUDE_doctor_selection_count_l4033_403304

def numInternists : ℕ := 5
def numSurgeons : ℕ := 6
def teamSize : ℕ := 4

theorem doctor_selection_count :
  (Nat.choose (numInternists + numSurgeons) teamSize) -
  (Nat.choose numInternists teamSize + Nat.choose numSurgeons teamSize) = 310 := by
  sorry

end NUMINAMATH_CALUDE_doctor_selection_count_l4033_403304


namespace NUMINAMATH_CALUDE_polynomial_value_l4033_403390

theorem polynomial_value (x : ℝ) (h : x^2 + 2*x - 2 = 0) : 4 - 2*x - x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l4033_403390


namespace NUMINAMATH_CALUDE_rhombus_area_from_quadratic_roots_l4033_403365

/-- Given a quadratic equation x^2 - 10x + 24 = 0, if its roots are the lengths of the diagonals
    of a rhombus, then the area of the rhombus is 12. -/
theorem rhombus_area_from_quadratic_roots : 
  ∀ (d₁ d₂ : ℝ), d₁ * d₂ = 24 → d₁ + d₂ = 10 → (1/2) * d₁ * d₂ = 12 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_from_quadratic_roots_l4033_403365


namespace NUMINAMATH_CALUDE_hockey_team_ties_l4033_403368

theorem hockey_team_ties (total_points : ℕ) (win_tie_difference : ℕ) : 
  total_points = 60 → win_tie_difference = 12 → 
  ∃ (ties wins : ℕ), 
    ties + wins = total_points ∧ 
    wins = ties + win_tie_difference ∧
    2 * wins + ties = total_points ∧
    ties = 12 := by
  sorry

end NUMINAMATH_CALUDE_hockey_team_ties_l4033_403368


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l4033_403316

/-- Proves that a train with given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 150)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 225) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l4033_403316


namespace NUMINAMATH_CALUDE_unit_circle_representation_l4033_403350

theorem unit_circle_representation (x y : ℝ) (n : ℤ) :
  (Real.arcsin x + Real.arccos y = n * π) →
  ((n = 0 → x^2 + y^2 = 1 ∧ x ≤ 0 ∧ y ≥ 0) ∧
   (n = 1 → x^2 + y^2 = 1 ∧ x ≥ 0 ∧ y ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_unit_circle_representation_l4033_403350


namespace NUMINAMATH_CALUDE_farmland_cleanup_theorem_l4033_403301

/-- Calculates the remaining area to be cleaned given the total area and cleaned areas -/
def remaining_area (total : Float) (lizzie : Float) (hilltown : Float) (green_valley : Float) : Float :=
  total - (lizzie + hilltown + green_valley)

/-- Theorem stating that the remaining area to be cleaned is 2442.38 square feet -/
theorem farmland_cleanup_theorem :
  remaining_area 9500.0 2534.1 2675.95 1847.57 = 2442.38 := by
  sorry

end NUMINAMATH_CALUDE_farmland_cleanup_theorem_l4033_403301


namespace NUMINAMATH_CALUDE_cubic_factorization_l4033_403388

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l4033_403388


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l4033_403343

theorem imaginary_part_of_one_plus_i_to_fifth (i : ℂ) : i * i = -1 → Complex.im ((1 + i) ^ 5) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l4033_403343


namespace NUMINAMATH_CALUDE_smallest_multiple_l4033_403380

theorem smallest_multiple (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k) ∧ 
  n ≡ 3 [ZMOD 71] ∧ 
  (∀ m : ℕ, m < n → ¬((∃ k : ℕ, m = 17 * k) ∧ m ≡ 3 [ZMOD 71])) → 
  n = 1139 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l4033_403380


namespace NUMINAMATH_CALUDE_seonwoo_change_l4033_403331

/-- Calculates the change Seonwoo received after buying bubblegum and ramen. -/
theorem seonwoo_change
  (initial_amount : ℕ)
  (bubblegum_cost : ℕ)
  (bubblegum_count : ℕ)
  (ramen_cost_per_two : ℕ)
  (ramen_count : ℕ)
  (h1 : initial_amount = 10000)
  (h2 : bubblegum_cost = 600)
  (h3 : bubblegum_count = 2)
  (h4 : ramen_cost_per_two = 1600)
  (h5 : ramen_count = 9) :
  initial_amount - (bubblegum_cost * bubblegum_count + 
    (ramen_cost_per_two * (ramen_count / 2)) + 
    (ramen_cost_per_two / 2 * (ramen_count % 2))) = 1600 :=
by sorry

end NUMINAMATH_CALUDE_seonwoo_change_l4033_403331


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l4033_403326

/-- Given a line l: 3x + 4y - 12 = 0, prove that 3x + 4y - 9 = 0 is the equation of the line
    that passes through the point (-1, 3) and has the same slope as line l. -/
theorem parallel_line_through_point (x y : ℝ) : 
  let l : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 12 = 0
  let m : ℝ := -3 / 4  -- slope of line l
  let new_line : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 9 = 0
  (∀ x y, l x y → (y - 0) = m * (x - 0)) →  -- l has slope m
  new_line (-1) 3 →  -- new line passes through (-1, 3)
  (∀ x y, new_line x y → (y - 3) = m * (x - (-1))) →  -- new line has slope m
  ∀ x y, new_line x y ↔ 3 * x + 4 * y - 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l4033_403326


namespace NUMINAMATH_CALUDE_student_a_final_score_l4033_403373

/-- Calculate the final score for a test -/
def finalScore (totalQuestions : ℕ) (correctAnswers : ℕ) : ℕ :=
  let incorrectAnswers := totalQuestions - correctAnswers
  correctAnswers - 2 * incorrectAnswers

/-- Theorem: The final score for a test with 100 questions and 92 correct answers is 76 -/
theorem student_a_final_score :
  finalScore 100 92 = 76 := by
  sorry

end NUMINAMATH_CALUDE_student_a_final_score_l4033_403373


namespace NUMINAMATH_CALUDE_jenn_bike_purchase_l4033_403307

/-- Calculates the amount left over after buying a bike, given the number of jars of quarters,
    quarters per jar, and the cost of the bike. -/
def money_left_over (num_jars : ℕ) (quarters_per_jar : ℕ) (bike_cost : ℚ) : ℚ :=
  (num_jars * quarters_per_jar * (1/4 : ℚ)) - bike_cost

/-- Proves that given 5 jars of quarters with 160 quarters per jar, and a bike costing $180,
    the amount left over after buying the bike is $20. -/
theorem jenn_bike_purchase : money_left_over 5 160 180 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jenn_bike_purchase_l4033_403307


namespace NUMINAMATH_CALUDE_larger_number_ratio_l4033_403391

theorem larger_number_ratio (m v : ℝ) (h1 : m < v) (h2 : v - m/4 = 5*(3*m/4)) : v = 4*m := by
  sorry

end NUMINAMATH_CALUDE_larger_number_ratio_l4033_403391


namespace NUMINAMATH_CALUDE_train_speed_problem_l4033_403366

/-- Proves that for a journey of 70 km, if a train traveling at 35 kmph
    arrives 15 minutes late compared to its on-time speed,
    then the on-time speed is 40 kmph. -/
theorem train_speed_problem (v : ℝ) : 
  (70 / v + 0.25 = 70 / 35) → v = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l4033_403366


namespace NUMINAMATH_CALUDE_probability_of_cooking_sum_of_probabilities_l4033_403349

/-- Represents the set of available courses. -/
inductive Course
| Planting
| Cooking
| Pottery
| Carpentry

/-- The probability of selecting a specific course from the available courses. -/
def probability_of_course (course : Course) : ℚ :=
  1 / 4

/-- Theorem stating that the probability of selecting "cooking" is 1/4. -/
theorem probability_of_cooking :
  probability_of_course Course.Cooking = 1 / 4 := by
  sorry

/-- Theorem stating that the sum of probabilities for all courses is 1. -/
theorem sum_of_probabilities :
  (probability_of_course Course.Planting) +
  (probability_of_course Course.Cooking) +
  (probability_of_course Course.Pottery) +
  (probability_of_course Course.Carpentry) = 1 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_cooking_sum_of_probabilities_l4033_403349


namespace NUMINAMATH_CALUDE_sum_interior_angles_theorem_l4033_403397

/-- The sum of interior angles of an n-gon -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of the interior angles of any n-gon is (n-2) * 180° -/
theorem sum_interior_angles_theorem (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2) * 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_theorem_l4033_403397


namespace NUMINAMATH_CALUDE_x_value_for_y_4_l4033_403309

/-- The relationship between x, y, and z -/
def x_relation (x y z k : ℚ) : Prop :=
  x = k * (z / y^2)

/-- The function defining z in terms of y -/
def z_function (y : ℚ) : ℚ :=
  2 * y - 1

theorem x_value_for_y_4 (k : ℚ) :
  (∃ x₀ : ℚ, x_relation x₀ 3 (z_function 3) k ∧ x₀ = 1) →
  (∃ x : ℚ, x_relation x 4 (z_function 4) k ∧ x = 63/80) :=
by sorry

end NUMINAMATH_CALUDE_x_value_for_y_4_l4033_403309


namespace NUMINAMATH_CALUDE_square_area_on_circle_and_tangent_l4033_403393

/-- Given a circle with radius 5 and a square with two vertices on the circle
    and two vertices on a tangent to the circle, the area of the square is 64. -/
theorem square_area_on_circle_and_tangent :
  ∀ (circle : ℝ → ℝ → Prop) (square : ℝ → ℝ → Prop) (r : ℝ),
  (r = 5) →  -- The radius of the circle is 5
  (∃ (A B C D : ℝ × ℝ),
    -- Two vertices of the square lie on the circle
    circle A.1 A.2 ∧ circle C.1 C.2 ∧
    -- The other two vertices lie on a tangent to the circle
    (∃ (t : ℝ → ℝ → Prop), t B.1 B.2 ∧ t D.1 D.2) ∧
    -- A, B, C, D form a square
    square A.1 A.2 ∧ square B.1 B.2 ∧ square C.1 C.2 ∧ square D.1 D.2) →
  (∃ (area : ℝ), area = 64) :=
by sorry

end NUMINAMATH_CALUDE_square_area_on_circle_and_tangent_l4033_403393


namespace NUMINAMATH_CALUDE_max_x_minus_y_is_sqrt5_l4033_403334

theorem max_x_minus_y_is_sqrt5 (x y : ℝ) (h : x^2 + 2*x*y + y^2 + 4*x^2*y^2 = 4) :
  ∃ (max : ℝ), (∀ (a b : ℝ), a^2 + 2*a*b + b^2 + 4*a^2*b^2 = 4 → a - b ≤ max) ∧ max = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_is_sqrt5_l4033_403334


namespace NUMINAMATH_CALUDE_rectangle_diagonal_after_expansion_l4033_403371

/-- Given a rectangle with width 10 meters and area 150 square meters,
    if its length is increased such that the new area is 3.7 times the original area,
    then the length of the diagonal of the new rectangle is approximately 56.39 meters. -/
theorem rectangle_diagonal_after_expansion (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (original_length new_length diagonal : ℝ),
    original_length > 0 ∧
    new_length > 0 ∧
    diagonal > 0 ∧
    10 * original_length = 150 ∧
    10 * new_length = 3.7 * 150 ∧
    diagonal ^ 2 = 10 ^ 2 + new_length ^ 2 ∧
    |diagonal - 56.39| < ε :=
by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_after_expansion_l4033_403371


namespace NUMINAMATH_CALUDE_last_two_digits_problem_l4033_403355

theorem last_two_digits_problem (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 2/13) :
  (x.val ^ y.val + y.val ^ x.val) % 100 = 74 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_problem_l4033_403355


namespace NUMINAMATH_CALUDE_square_and_cube_sum_l4033_403321

theorem square_and_cube_sum (p q : ℝ) 
  (h1 : p * q = 15) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 34 ∧ p^3 + q^3 = 152 := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_sum_l4033_403321


namespace NUMINAMATH_CALUDE_work_completion_solution_l4033_403381

/-- Represents the work completion problem -/
structure WorkCompletion where
  initial_workers : ℕ
  initial_days : ℕ
  worked_days : ℕ
  added_workers : ℕ

/-- Calculates the total days to complete the work -/
def total_days (w : WorkCompletion) : ℚ :=
  let initial_work_rate : ℚ := 1 / (w.initial_workers * w.initial_days)
  let work_done : ℚ := w.worked_days * w.initial_workers * initial_work_rate
  let remaining_work : ℚ := 1 - work_done
  let new_work_rate : ℚ := (w.initial_workers + w.added_workers) * initial_work_rate
  w.worked_days + remaining_work / new_work_rate

/-- Theorem stating the solution to the work completion problem -/
theorem work_completion_solution :
  ∀ w : WorkCompletion,
    w.initial_workers = 12 →
    w.initial_days = 18 →
    w.worked_days = 6 →
    w.added_workers = 4 →
    total_days w = 24 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_solution_l4033_403381


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_5_minimum_m_for_intersection_l4033_403358

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2*|x + 1|

-- Part I
theorem solution_set_when_m_is_5 :
  {x : ℝ | f 5 x > 2} = {x : ℝ | -4/3 < x ∧ x < 0} := by sorry

-- Part II
-- Define the quadratic function
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem minimum_m_for_intersection :
  ∀ m : ℝ, (∀ x : ℝ, ∃ y : ℝ, f m y = g y) ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_5_minimum_m_for_intersection_l4033_403358


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l4033_403348

theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_of_interior_angles : ℝ := (n - 2) * 180
  let interior_angle : ℝ := sum_of_interior_angles / n
  135

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l4033_403348


namespace NUMINAMATH_CALUDE_cubic_inequality_l4033_403385

theorem cubic_inequality (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) : a^3 < a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l4033_403385


namespace NUMINAMATH_CALUDE_deepak_present_age_l4033_403337

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age. -/
theorem deepak_present_age 
  (rahul_deepak_ratio : ℚ) 
  (rahul_future_age : ℕ) 
  (years_difference : ℕ) :
  rahul_deepak_ratio = 4 / 3 →
  rahul_future_age = 42 →
  years_difference = 6 →
  ∃ (deepak_age : ℕ), deepak_age = 27 :=
by sorry

end NUMINAMATH_CALUDE_deepak_present_age_l4033_403337


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4033_403311

/-- Given a geometric sequence, prove that if the sum of the first n terms is 48
    and the sum of the first 2n terms is 60, then the sum of the first 3n terms is 63. -/
theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) : 
  S n = 48 → S (2*n) = 60 → S (3*n) = 63 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4033_403311


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l4033_403395

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) := by sorry

theorem negation_of_inequality :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l4033_403395


namespace NUMINAMATH_CALUDE_nine_b_value_l4033_403345

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) : 9 * b = 216 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nine_b_value_l4033_403345


namespace NUMINAMATH_CALUDE_solve_p_q_system_l4033_403375

theorem solve_p_q_system (p q : ℝ) (hp : p > 1) (hq : q > 1)
  (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_p_q_system_l4033_403375


namespace NUMINAMATH_CALUDE_fraction_undefined_l4033_403313

theorem fraction_undefined (x : ℝ) : (x + 1 = 0) ↔ (∀ y : ℝ, y ≠ 1 / (x + 1)) :=
  sorry

end NUMINAMATH_CALUDE_fraction_undefined_l4033_403313


namespace NUMINAMATH_CALUDE_stating_perfect_match_equation_l4033_403306

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 22

/-- Represents the number of screws a worker can produce per day -/
def screws_per_worker : ℕ := 1200

/-- Represents the number of nuts a worker can produce per day -/
def nuts_per_worker : ℕ := 2000

/-- Represents the number of nuts needed for each screw -/
def nuts_per_screw : ℕ := 2

/-- 
Theorem stating that for a perfect match of products, 
the equation 2 × 1200(22 - x) = 2000x must hold, 
where x is the number of workers assigned to produce nuts
-/
theorem perfect_match_equation (x : ℕ) (h : x ≤ total_workers) : 
  2 * screws_per_worker * (total_workers - x) = nuts_per_worker * x := by
  sorry

end NUMINAMATH_CALUDE_stating_perfect_match_equation_l4033_403306


namespace NUMINAMATH_CALUDE_quadratic_sum_l4033_403384

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∃ (x_max : ℝ), ∀ x, QuadraticFunction a b c x ≤ QuadraticFunction a b c x_max ∧
    QuadraticFunction a b c x_max = 72) →
  QuadraticFunction a b c 0 = -1 →
  QuadraticFunction a b c 6 = -1 →
  a + b + c = 356 / 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l4033_403384


namespace NUMINAMATH_CALUDE_cupcakes_per_package_l4033_403303

theorem cupcakes_per_package (packages : ℕ) (eaten : ℕ) (left : ℕ) :
  packages = 3 →
  eaten = 5 →
  left = 7 →
  ∃ cupcakes_per_package : ℕ,
    cupcakes_per_package * packages = eaten + left ∧
    cupcakes_per_package = 4 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_per_package_l4033_403303


namespace NUMINAMATH_CALUDE_distance_is_sqrt_51_l4033_403340

def point : ℝ × ℝ × ℝ := (3, 5, -1)
def line_point : ℝ × ℝ × ℝ := (2, 4, 6)
def line_direction : ℝ × ℝ × ℝ := (4, 3, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_sqrt_51 :
  distance_to_line point line_point line_direction = Real.sqrt 51 :=
sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_51_l4033_403340


namespace NUMINAMATH_CALUDE_inverse_f_at_8_l4033_403354

def f (x : ℝ) : ℝ := 1 - 3*(x - 1) + 3*(x - 1)^2 - (x - 1)^3

theorem inverse_f_at_8 : f 0 = 8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_8_l4033_403354


namespace NUMINAMATH_CALUDE_tan_two_theta_minus_pi_six_l4033_403382

theorem tan_two_theta_minus_pi_six (θ : Real) 
  (h : 4 * Real.cos (θ + π/3) * Real.cos (θ - π/6) = Real.sin (2*θ)) : 
  Real.tan (2*θ - π/6) = Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_theta_minus_pi_six_l4033_403382


namespace NUMINAMATH_CALUDE_package_weight_is_five_l4033_403329

/-- Calculates the weight of a package given the total shipping cost, flat fee, and cost per pound. -/
def package_weight (total_cost flat_fee cost_per_pound : ℚ) : ℚ :=
  (total_cost - flat_fee) / cost_per_pound

/-- Theorem stating that given the specific shipping costs, the package weighs 5 pounds. -/
theorem package_weight_is_five :
  package_weight 9 5 (4/5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_package_weight_is_five_l4033_403329


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l4033_403367

theorem polynomial_root_implies_coefficients : ∀ (a b : ℝ), 
  (Complex.I : ℂ) ^ 3 + a * (Complex.I : ℂ) ^ 2 + 2 * (Complex.I : ℂ) + b = (2 - 3 * Complex.I : ℂ) ^ 3 →
  a = -5/4 ∧ b = 143/4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l4033_403367


namespace NUMINAMATH_CALUDE_triangle_sine_law_l4033_403300

theorem triangle_sine_law (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  A = π / 3 →
  a = Real.sqrt 3 →
  c / Real.sin C = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_law_l4033_403300


namespace NUMINAMATH_CALUDE_solution_set_of_inequalities_l4033_403396

theorem solution_set_of_inequalities :
  let S := { x : ℝ | x - 2 > 1 ∧ x < 4 }
  S = { x : ℝ | 3 < x ∧ x < 4 } := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequalities_l4033_403396


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_quadratic_l4033_403312

theorem negation_of_universal_positive_quadratic (p : Prop) :
  (p ↔ ∀ x : ℝ, x^2 - x + 1 > 0) →
  (¬p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_quadratic_l4033_403312


namespace NUMINAMATH_CALUDE_cubic_factorization_l4033_403338

theorem cubic_factorization (x : ℝ) : x^3 + 5*x^2 + 6*x = x*(x+2)*(x+3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l4033_403338


namespace NUMINAMATH_CALUDE_band_members_proof_l4033_403324

/-- Represents the price per set of costumes based on the quantity purchased -/
def price_per_set (quantity : ℕ) : ℕ :=
  if quantity ≤ 39 then 80
  else if quantity ≤ 79 then 70
  else 60

theorem band_members_proof :
  ∀ (x y : ℕ),
    x + y = 75 →
    x ≥ 40 →
    price_per_set x * x + price_per_set y * y = 5600 →
    x = 40 ∧ y = 35 := by
  sorry

end NUMINAMATH_CALUDE_band_members_proof_l4033_403324


namespace NUMINAMATH_CALUDE_a_in_M_l4033_403364

def M : Set ℝ := { x | x ≤ 5 }

def a : ℝ := 2

theorem a_in_M : a ∈ M := by sorry

end NUMINAMATH_CALUDE_a_in_M_l4033_403364


namespace NUMINAMATH_CALUDE_matrix_det_plus_five_l4033_403346

theorem matrix_det_plus_five (M : Matrix (Fin 2) (Fin 2) ℤ) :
  M = ![![7, -2], ![-3, 6]] →
  M.det + 5 = 41 := by
sorry

end NUMINAMATH_CALUDE_matrix_det_plus_five_l4033_403346


namespace NUMINAMATH_CALUDE_volume_ratio_theorem_l4033_403374

/-- A right rectangular prism with edge lengths 2, 3, and 5 -/
structure RectPrism where
  length : ℝ := 2
  width : ℝ := 3
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : RectPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume_S (B : RectPrism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume function -/
structure VolumeCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d

theorem volume_ratio_theorem (B : RectPrism) (coeffs : VolumeCoeffs) 
  (h : ∀ r : ℝ, volume_S B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d) :
  coeffs.b * coeffs.c / (coeffs.a * coeffs.d) = 15.5 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_theorem_l4033_403374


namespace NUMINAMATH_CALUDE_exponent_multiplication_l4033_403389

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l4033_403389


namespace NUMINAMATH_CALUDE_third_square_is_G_l4033_403359

/-- Represents a 2x2 square -/
structure Square :=
  (label : Char)

/-- Represents the visibility of a square -/
inductive Visibility
  | Full
  | Partial

/-- Represents the position of a square in the 4x4 grid -/
structure Position :=
  (row : Fin 2)
  (col : Fin 2)

/-- Represents the state of the 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Option Square

/-- Represents the sequence of square placements -/
def PlacementSequence := List Square

/-- Determines if a square is in a corner position -/
def isCorner (pos : Position) : Bool :=
  (pos.row = 0 ∨ pos.row = 1) ∧ (pos.col = 0 ∨ pos.col = 1)

/-- The main theorem to prove -/
theorem third_square_is_G 
  (squares : List Square)
  (grid : Grid)
  (sequence : PlacementSequence)
  (visibility : Square → Visibility)
  (position : Square → Position) :
  squares.length = 8 ∧
  (∃ s ∈ squares, s.label = 'E') ∧
  visibility (Square.mk 'E') = Visibility.Full ∧
  (∀ s ∈ squares, s.label ≠ 'E' → visibility s = Visibility.Partial) ∧
  (∃ s ∈ squares, isCorner (position s) ∧ s.label = 'G') ∧
  sequence.length = 8 ∧
  sequence.getLast? = some (Square.mk 'E') →
  (sequence.get? 2 = some (Square.mk 'G')) :=
by sorry

end NUMINAMATH_CALUDE_third_square_is_G_l4033_403359


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l4033_403376

theorem unique_two_digit_integer (s : ℕ) : 
  (∃! s, 10 ≤ s ∧ s < 100 ∧ 13 * s % 100 = 52) :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l4033_403376


namespace NUMINAMATH_CALUDE_sausage_problem_l4033_403378

theorem sausage_problem (initial_sausages : ℕ) (remaining_sausages : ℕ) 
  (h1 : initial_sausages = 600)
  (h2 : remaining_sausages = 45) :
  ∃ (x : ℚ), 
    0 < x ∧ x < 1 ∧
    remaining_sausages = (1/4 : ℚ) * (1/2 : ℚ) * (1 - x) * initial_sausages ∧
    x = (2/5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sausage_problem_l4033_403378


namespace NUMINAMATH_CALUDE_positive_numbers_relation_l4033_403315

theorem positive_numbers_relation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 / b = 5) (h2 : b^2 / c = 3) (h3 : c^2 / a = 7) : a = 15 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_relation_l4033_403315


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4033_403335

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 2 ∧ b = 4 ∧ c = 4 ∧  -- Two sides are equal (isosceles) and one side is 2
  a + b > c ∧ b + c > a ∧ a + c > b →  -- Triangle inequality
  a + b + c = 10 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4033_403335


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l4033_403357

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 7
def quadruplets_in_lineup : ℕ := 3

theorem volleyball_team_selection :
  (Nat.choose quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 1980 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l4033_403357


namespace NUMINAMATH_CALUDE_system_solution_l4033_403392

theorem system_solution : ∃ (x y : ℝ), (2 * x + y = 7) ∧ (4 * x + 5 * y = 11) ∧ (x = 4) ∧ (y = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4033_403392


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l4033_403314

theorem quadratic_two_roots
  (a b c d e : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > d)
  (h4 : e ≠ -1) :
  ∃ (x y : ℝ), x ≠ y ∧
  (e + 1) * x^2 - (a + c + b*e + d*e) * x + a*c + e*b*d = 0 ∧
  (e + 1) * y^2 - (a + c + b*e + d*e) * y + a*c + e*b*d = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l4033_403314


namespace NUMINAMATH_CALUDE_vector_collinearity_l4033_403336

/-- Given vectors a, b, and c in R², prove that if a = (-2, 0), b = (2, 1), c = (x, 1),
    and 3a + b is collinear with c, then x = -4. -/
theorem vector_collinearity (a b c : ℝ × ℝ) (x : ℝ) :
  a = (-2, 0) →
  b = (2, 1) →
  c = (x, 1) →
  ∃ (k : ℝ), k ≠ 0 ∧ (3 • a + b) = k • c →
  x = -4 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_l4033_403336


namespace NUMINAMATH_CALUDE_tv_monthly_payment_l4033_403305

/-- Calculates the monthly payment for a discounted television purchase with installments -/
theorem tv_monthly_payment 
  (original_price : ℝ) 
  (discount_rate : ℝ) 
  (first_installment : ℝ) 
  (num_installments : ℕ) 
  (h1 : original_price = 480) 
  (h2 : discount_rate = 0.05) 
  (h3 : first_installment = 150) 
  (h4 : num_installments = 3) : 
  ∃ (monthly_payment : ℝ), 
    monthly_payment = (original_price * (1 - discount_rate) - first_installment) / num_installments ∧ 
    monthly_payment = 102 := by
  sorry

end NUMINAMATH_CALUDE_tv_monthly_payment_l4033_403305


namespace NUMINAMATH_CALUDE_product_of_D_coordinates_l4033_403398

-- Define the points
def C : ℝ × ℝ := (6, -1)
def N : ℝ × ℝ := (4, 3)

-- Define D as a variable point
variable (D : ℝ × ℝ)

-- Define the midpoint condition
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- State the theorem
theorem product_of_D_coordinates :
  is_midpoint N C D → D.1 * D.2 = 14 := by sorry

end NUMINAMATH_CALUDE_product_of_D_coordinates_l4033_403398


namespace NUMINAMATH_CALUDE_root_sum_fraction_l4033_403310

/-- Given a, b, c are roots of x^3 - 20x^2 + 22, prove bc/a^2 + ac/b^2 + ab/c^2 = -40 -/
theorem root_sum_fraction (a b c : ℝ) : 
  (a^3 - 20*a^2 + 22 = 0) → 
  (b^3 - 20*b^2 + 22 = 0) → 
  (c^3 - 20*c^2 + 22 = 0) → 
  (b*c/a^2 + a*c/b^2 + a*b/c^2 = -40) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l4033_403310


namespace NUMINAMATH_CALUDE_min_throws_for_three_occurrences_l4033_403383

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice thrown each time -/
def num_dice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def min_sum : ℕ := num_dice

/-- The maximum possible sum when rolling the dice -/
def max_sum : ℕ := num_dice * sides

/-- The number of possible different sums -/
def num_sums : ℕ := max_sum - min_sum + 1

/-- The minimum number of throws required -/
def min_throws : ℕ := num_sums * 2 + 1

theorem min_throws_for_three_occurrences :
  min_throws = 43 :=
sorry

end NUMINAMATH_CALUDE_min_throws_for_three_occurrences_l4033_403383
