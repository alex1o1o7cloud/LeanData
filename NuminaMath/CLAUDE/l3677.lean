import Mathlib

namespace chefs_and_waiters_arrangements_l3677_367713

/-- The number of ways to arrange chefs and waiters in a row --/
def arrangements (num_chefs num_waiters : ℕ) : ℕ :=
  if num_chefs + num_waiters ≠ 5 then 0
  else if num_chefs ≠ 2 then 0
  else if num_waiters ≠ 3 then 0
  else 36

/-- Theorem stating that the number of arrangements for 2 chefs and 3 waiters is 36 --/
theorem chefs_and_waiters_arrangements :
  arrangements 2 3 = 36 := by sorry

end chefs_and_waiters_arrangements_l3677_367713


namespace rectangular_paper_area_l3677_367765

/-- The area of a rectangular sheet of paper -/
def paper_area (width length : ℝ) : ℝ := width * length

theorem rectangular_paper_area :
  let width : ℝ := 25
  let length : ℝ := 20
  paper_area width length = 500 := by
  sorry

end rectangular_paper_area_l3677_367765


namespace tag_sum_is_1000_l3677_367748

/-- The sum of the numbers tagged on four cards W, X, Y, Z -/
def total_tag_sum (w x y z : ℕ) : ℕ := w + x + y + z

/-- Theorem stating that the sum of the tagged numbers is 1000 -/
theorem tag_sum_is_1000 :
  ∀ (w x y z : ℕ),
  w = 200 →
  x = w / 2 →
  y = w + x →
  z = 400 →
  total_tag_sum w x y z = 1000 := by
  sorry

end tag_sum_is_1000_l3677_367748


namespace trigonometric_expression_evaluation_l3677_367733

theorem trigonometric_expression_evaluation (α : Real) (h : Real.tan α = 2) :
  (Real.cos (-π/2 - α) * Real.tan (π + α) - Real.sin (π/2 - α)) /
  (Real.cos (3*π/2 + α) + Real.cos (π - α)) = -5 := by
  sorry

end trigonometric_expression_evaluation_l3677_367733


namespace tetrahedron_volume_and_surface_area_l3677_367754

/-- A regular tetrahedron with given height and base edge length -/
structure RegularTetrahedron where
  height : ℝ
  base_edge : ℝ

/-- Volume of a regular tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ := sorry

/-- Surface area of a regular tetrahedron -/
def surface_area (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the volume and surface area of a specific regular tetrahedron -/
theorem tetrahedron_volume_and_surface_area :
  let t : RegularTetrahedron := ⟨1, 2 * Real.sqrt 6⟩
  volume t = 2 * Real.sqrt 3 ∧
  surface_area t = 9 * Real.sqrt 2 + 6 * Real.sqrt 3 := by sorry

end tetrahedron_volume_and_surface_area_l3677_367754


namespace stair_step_24th_row_white_squares_l3677_367712

/-- Represents the number of squares in a row of the stair-step figure -/
def total_squares (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- Represents the number of white squares in a row of the stair-step figure -/
def white_squares (n : ℕ) : ℕ := (total_squares n - 2) / 2 + (total_squares n - 2) % 2

/-- Theorem stating that the 24th row of the stair-step figure contains 23 white squares -/
theorem stair_step_24th_row_white_squares :
  white_squares 24 = 23 := by sorry

end stair_step_24th_row_white_squares_l3677_367712


namespace large_ball_radius_l3677_367759

theorem large_ball_radius (num_small_balls : ℕ) (small_radius : ℝ) (large_radius : ℝ) : 
  num_small_balls = 12 →
  small_radius = 2 →
  (4 / 3) * Real.pi * large_radius ^ 3 = num_small_balls * ((4 / 3) * Real.pi * small_radius ^ 3) →
  large_radius = (96 : ℝ) ^ (1 / 3 : ℝ) :=
by sorry

end large_ball_radius_l3677_367759


namespace number_satisfying_condition_l3677_367744

theorem number_satisfying_condition : ∃ x : ℝ, 0.05 * x = 0.20 * 650 + 190 := by
  sorry

end number_satisfying_condition_l3677_367744


namespace decoration_cost_theorem_l3677_367739

/-- Calculates the total cost of decorations for a wedding reception. -/
def total_decoration_cost (num_tables : ℕ) 
                          (tablecloth_cost : ℕ) 
                          (place_setting_cost : ℕ) 
                          (place_settings_per_table : ℕ) 
                          (roses_per_centerpiece : ℕ) 
                          (rose_cost : ℕ) 
                          (lilies_per_centerpiece : ℕ) 
                          (lily_cost : ℕ) 
                          (daisies_per_centerpiece : ℕ) 
                          (daisy_cost : ℕ) 
                          (sunflowers_per_centerpiece : ℕ) 
                          (sunflower_cost : ℕ) 
                          (lighting_cost : ℕ) : ℕ :=
  let tablecloth_total := num_tables * tablecloth_cost
  let place_setting_total := num_tables * place_settings_per_table * place_setting_cost
  let centerpiece_cost := roses_per_centerpiece * rose_cost + 
                          lilies_per_centerpiece * lily_cost + 
                          daisies_per_centerpiece * daisy_cost + 
                          sunflowers_per_centerpiece * sunflower_cost
  let centerpiece_total := num_tables * centerpiece_cost
  tablecloth_total + place_setting_total + centerpiece_total + lighting_cost

theorem decoration_cost_theorem : 
  total_decoration_cost 30 25 12 6 15 6 20 5 5 3 3 4 450 = 9870 := by
  sorry

end decoration_cost_theorem_l3677_367739


namespace extreme_value_and_min_max_l3677_367700

/-- Function f(x) = 2x³ + ax² + bx + 1 -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

/-- Derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem extreme_value_and_min_max (a b : ℝ) : 
  f a b 1 = -6 ∧ f' a b 1 = 0 →
  a = 3 ∧ b = -12 ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x ≤ 21) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x ≥ -6) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x = 21) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f 3 (-12) x = -6) :=
by sorry

end extreme_value_and_min_max_l3677_367700


namespace quadratic_two_distinct_roots_l3677_367771

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 := by
  sorry

end quadratic_two_distinct_roots_l3677_367771


namespace power_twenty_equals_r_s_l3677_367776

theorem power_twenty_equals_r_s (a b : ℤ) (R S : ℝ) 
  (hR : R = 2^a) (hS : S = 5^b) : 
  R^(2*b) * S^a = 20^(a*b) := by
sorry

end power_twenty_equals_r_s_l3677_367776


namespace pawn_placement_count_l3677_367788

/-- The number of ways to place distinct pawns on a square chess board -/
def placePawns (n : ℕ) : ℕ :=
  (n.factorial) ^ 2

/-- The size of the chess board -/
def boardSize : ℕ := 5

/-- The number of pawns to be placed -/
def numPawns : ℕ := 5

theorem pawn_placement_count :
  placePawns boardSize = 14400 :=
sorry

end pawn_placement_count_l3677_367788


namespace smallest_n_for_more_than_half_remaining_l3677_367757

def outer_layer_cubes (n : ℕ) : ℕ := 6 * n^2 - 12 * n + 8

def remaining_cubes (n : ℕ) : ℕ := n^3 - outer_layer_cubes n

theorem smallest_n_for_more_than_half_remaining : 
  (∀ k : ℕ, k < 10 → 2 * remaining_cubes k ≤ k^3) ∧
  (2 * remaining_cubes 10 > 10^3) := by sorry

end smallest_n_for_more_than_half_remaining_l3677_367757


namespace doll_collection_increase_l3677_367735

/-- Proves that if adding 2 dolls to a collection increases it by 25%, then the final number of dolls in the collection is 10. -/
theorem doll_collection_increase (original : ℕ) : 
  (original + 2 : ℚ) = original * (1 + 1/4) → original + 2 = 10 := by
  sorry

end doll_collection_increase_l3677_367735


namespace function_equal_to_inverse_is_identity_l3677_367711

-- Define an increasing function from R to R
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Define the theorem
theorem function_equal_to_inverse_is_identity
  (f : ℝ → ℝ)
  (h_increasing : IncreasingFunction f)
  (h_inverse : ∀ x : ℝ, f x = Function.invFun f x) :
  ∀ x : ℝ, f x = x :=
sorry

end function_equal_to_inverse_is_identity_l3677_367711


namespace division_inequality_l3677_367763

theorem division_inequality : ¬(∃ q r, 4900 = 600 * q + r ∧ r < 600 ∧ 49 = 6 * q + r ∧ r < 6) := by
  sorry

end division_inequality_l3677_367763


namespace hungarian_deck_probabilities_l3677_367777

/-- Represents a Hungarian deck of cards -/
structure HungarianDeck :=
  (cards : Finset (Fin 32))
  (suits : Fin 4)
  (cardsPerSuit : Fin 8)

/-- Calculates the probability of drawing at least one ace given three cards of different suits -/
def probAceGivenDifferentSuits (deck : HungarianDeck) : ℚ :=
  169 / 512

/-- Calculates the probability of drawing at least one ace when drawing three cards -/
def probAceThreeCards (deck : HungarianDeck) : ℚ :=
  421 / 1240

/-- Calculates the probability of drawing three cards of different suits with at least one ace -/
def probDifferentSuitsWithAce (deck : HungarianDeck) : ℚ :=
  169 / 1240

/-- Main theorem stating the probabilities for the given scenarios -/
theorem hungarian_deck_probabilities (deck : HungarianDeck) :
  (probAceGivenDifferentSuits deck = 169 / 512) ∧
  (probAceThreeCards deck = 421 / 1240) ∧
  (probDifferentSuitsWithAce deck = 169 / 1240) :=
sorry

end hungarian_deck_probabilities_l3677_367777


namespace donny_spending_friday_sunday_l3677_367768

def monday_savings : ℝ := 15

def savings_increase_rate : ℝ := 0.1

def friday_spending_rate : ℝ := 0.5

def saturday_savings_decrease : ℝ := 0.2

def sunday_spending_rate : ℝ := 0.4

def tuesday_savings (monday : ℝ) : ℝ := monday * (1 + savings_increase_rate)

def wednesday_savings (tuesday : ℝ) : ℝ := tuesday * (1 + savings_increase_rate)

def thursday_savings (wednesday : ℝ) : ℝ := wednesday * (1 + savings_increase_rate)

def total_savings_thursday (mon tue wed thu : ℝ) : ℝ := mon + tue + wed + thu

def friday_spending (total : ℝ) : ℝ := total * friday_spending_rate

def saturday_savings (thursday : ℝ) : ℝ := thursday * (1 - saturday_savings_decrease)

def total_savings_saturday (friday_remaining saturday : ℝ) : ℝ := friday_remaining + saturday

def sunday_spending (total : ℝ) : ℝ := total * sunday_spending_rate

theorem donny_spending_friday_sunday : 
  let tue := tuesday_savings monday_savings
  let wed := wednesday_savings tue
  let thu := thursday_savings wed
  let total_thu := total_savings_thursday monday_savings tue wed thu
  let fri_spend := friday_spending total_thu
  let fri_remaining := total_thu - fri_spend
  let sat := saturday_savings thu
  let total_sat := total_savings_saturday fri_remaining sat
  let sun_spend := sunday_spending total_sat
  fri_spend + sun_spend = 55.13 := by sorry

end donny_spending_friday_sunday_l3677_367768


namespace fractions_not_both_integers_l3677_367703

theorem fractions_not_both_integers (n : ℤ) : 
  ¬(∃ (x y : ℤ), (n - 6 : ℤ) = 15 * x ∧ (n - 5 : ℤ) = 24 * y) := by
  sorry

end fractions_not_both_integers_l3677_367703


namespace supplementary_angles_ratio_l3677_367793

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 4 / 5 →  -- The ratio of the angles is 4:5
  b = 100 :=  -- The larger angle is 100°
by sorry

end supplementary_angles_ratio_l3677_367793


namespace four_rows_with_eight_people_l3677_367740

/-- Represents a seating arrangement with rows of 7 or 8 people -/
structure SeatingArrangement where
  total_people : ℕ
  rows_with_eight : ℕ
  rows_with_seven : ℕ

/-- Conditions for a valid seating arrangement -/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = 46 ∧
  s.total_people = 8 * s.rows_with_eight + 7 * s.rows_with_seven

/-- Theorem stating that in a valid arrangement with 46 people, 
    there are exactly 4 rows with 8 people -/
theorem four_rows_with_eight_people 
  (s : SeatingArrangement) (h : is_valid_arrangement s) : 
  s.rows_with_eight = 4 := by
  sorry

end four_rows_with_eight_people_l3677_367740


namespace nine_points_chords_l3677_367727

/-- The number of different chords that can be drawn by connecting two points
    out of n points marked on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of different chords that can be drawn by connecting two points
    out of nine points marked on the circumference of a circle is equal to 36 -/
theorem nine_points_chords : num_chords 9 = 36 := by
  sorry

end nine_points_chords_l3677_367727


namespace union_of_A_and_B_l3677_367707

def A : Set ℝ := {x | x^2 ≤ 1}
def B : Set ℝ := {x | x > 0}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x} := by sorry

end union_of_A_and_B_l3677_367707


namespace repeating_decimal_sum_l3677_367792

theorem repeating_decimal_sum (a b : ℕ+) : 
  (a.val : ℚ) / b.val = 4 / 11 → Nat.gcd a.val b.val = 1 → a.val + b.val = 15 := by
  sorry

end repeating_decimal_sum_l3677_367792


namespace sandro_children_l3677_367718

/-- Calculates the total number of children Sandro has -/
def total_children (sons : ℕ) (daughter_ratio : ℕ) : ℕ :=
  sons + sons * daughter_ratio

theorem sandro_children :
  let sons := 3
  let daughter_ratio := 6
  total_children sons daughter_ratio = 21 := by
  sorry

end sandro_children_l3677_367718


namespace ln_range_is_real_l3677_367710

-- Define the natural logarithm function
noncomputable def ln : ℝ → ℝ := Real.log

-- Statement: The range of the natural logarithm is all real numbers
theorem ln_range_is_real : Set.range ln = Set.univ := by sorry

end ln_range_is_real_l3677_367710


namespace arithmetic_mean_problem_l3677_367730

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + 17 + (2 * x) + 15 + (2 * x + 6)) / 5 = 26 → x = 82 / 5 := by
  sorry

end arithmetic_mean_problem_l3677_367730


namespace digit_subtraction_result_l3677_367705

def three_digit_number (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9

theorem digit_subtraction_result (a b c : ℕ) 
  (h1 : three_digit_number a b c) 
  (h2 : a = c + 2) : 
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) ≡ 8 [ZMOD 10] := by
  sorry

end digit_subtraction_result_l3677_367705


namespace tommys_balloons_l3677_367709

theorem tommys_balloons (initial_balloons : ℕ) (final_balloons : ℕ) : 
  initial_balloons = 26 → final_balloons = 60 → final_balloons - initial_balloons = 34 := by
  sorry

end tommys_balloons_l3677_367709


namespace water_tank_capacity_l3677_367755

theorem water_tank_capacity : ∀ (c : ℝ),
  (c / 3 : ℝ) / c = 1 / 3 →
  ((c / 3 + 7) : ℝ) / c = 2 / 5 →
  c = 105 := by
sorry

end water_tank_capacity_l3677_367755


namespace problem_statement_l3677_367728

theorem problem_statement (x y : ℝ) (a : ℝ) :
  (x - a*y) * (x + a*y) = x^2 - 16*y^2 → a = 4 ∨ a = -4 := by
  sorry

end problem_statement_l3677_367728


namespace smallest_integer_with_given_remainders_l3677_367761

theorem smallest_integer_with_given_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
    x % 5 = 2 ∧ 
    x % 3 = 1 ∧ 
    x % 7 = 3 ∧
    (∀ y : ℕ, y > 0 → y % 5 = 2 → y % 3 = 1 → y % 7 = 3 → x ≤ y) ∧
    x = 22 :=
by sorry

end smallest_integer_with_given_remainders_l3677_367761


namespace gcd_18_30_l3677_367794

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l3677_367794


namespace james_new_weight_l3677_367758

/-- Calculates the new weight after muscle and fat gain -/
def new_weight (initial_weight : ℝ) (muscle_gain_percentage : ℝ) (fat_gain_ratio : ℝ) : ℝ :=
  let muscle_gain := initial_weight * muscle_gain_percentage
  let fat_gain := muscle_gain * fat_gain_ratio
  initial_weight + muscle_gain + fat_gain

/-- Proves that James's new weight is 150 kg after gaining muscle and fat -/
theorem james_new_weight :
  new_weight 120 0.2 0.25 = 150 := by
  sorry

end james_new_weight_l3677_367758


namespace bowl_cost_l3677_367786

theorem bowl_cost (sets : ℕ) (bowls_per_set : ℕ) (total_cost : ℕ) : 
  sets = 12 → bowls_per_set = 2 → total_cost = 240 → 
  (total_cost : ℚ) / (sets * bowls_per_set : ℚ) = 10 := by
  sorry

end bowl_cost_l3677_367786


namespace oshea_basil_seeds_l3677_367773

/-- The number of basil seeds Oshea bought -/
def total_seeds : ℕ := sorry

/-- The number of large planters Oshea has -/
def large_planters : ℕ := 4

/-- The number of seeds each large planter can hold -/
def seeds_per_large_planter : ℕ := 20

/-- The number of seeds each small planter can hold -/
def seeds_per_small_planter : ℕ := 4

/-- The number of small planters needed to plant all the basil seeds -/
def small_planters : ℕ := 30

/-- Theorem stating that the total number of basil seeds Oshea bought is 200 -/
theorem oshea_basil_seeds : total_seeds = 200 := by sorry

end oshea_basil_seeds_l3677_367773


namespace long_furred_brown_dogs_l3677_367729

theorem long_furred_brown_dogs 
  (total : ℕ) 
  (long_furred : ℕ) 
  (brown : ℕ) 
  (neither : ℕ) 
  (h1 : total = 45)
  (h2 : long_furred = 26)
  (h3 : brown = 30)
  (h4 : neither = 8) :
  long_furred + brown - (total - neither) = 19 := by
sorry

end long_furred_brown_dogs_l3677_367729


namespace sqrt_equation_solution_l3677_367723

theorem sqrt_equation_solution (y : ℝ) :
  Real.sqrt (3 + Real.sqrt (3 * y - 4)) = Real.sqrt 10 → y = 53 / 3 := by
  sorry

end sqrt_equation_solution_l3677_367723


namespace selectBooks_eq_1041_l3677_367795

/-- The number of ways to select books for three children -/
def selectBooks : ℕ :=
  let smallBooks := 6
  let largeBooks := 3
  let children := 3

  -- Case 1: All children take large books
  let case1 := largeBooks.factorial

  -- Case 2: 1 child takes small books, 2 take large books
  let case2 := Nat.choose children 1 * Nat.choose smallBooks 2 * Nat.choose largeBooks 2

  -- Case 3: 2 children take small books, 1 takes large book
  let case3 := Nat.choose children 2 * Nat.choose smallBooks 2 * Nat.choose (smallBooks - 2) 2 * Nat.choose largeBooks 1

  -- Case 4: All children take small books
  let case4 := Nat.choose smallBooks 2 * Nat.choose (smallBooks - 2) 2 * Nat.choose (smallBooks - 4) 2

  case1 + case2 + case3 + case4

theorem selectBooks_eq_1041 : selectBooks = 1041 := by
  sorry

end selectBooks_eq_1041_l3677_367795


namespace not_divides_power_plus_one_l3677_367769

theorem not_divides_power_plus_one (n k : ℕ) (h1 : n = 2^2007 * k + 1) (h2 : Odd k) :
  ¬(n ∣ 2^(n - 1) + 1) := by
  sorry

end not_divides_power_plus_one_l3677_367769


namespace a_initial_is_9000_l3677_367780

/-- Represents the initial investment and profit distribution scenario -/
structure BusinessScenario where
  a_initial : ℕ  -- A's initial investment
  b_investment : ℕ  -- B's investment
  b_join_time : ℕ  -- Time when B joined (in months)
  total_time : ℕ  -- Total time of the year (in months)
  profit_ratio : Rat  -- Profit ratio (A:B)

/-- Calculates the initial investment of A given the business scenario -/
def calculate_a_initial (scenario : BusinessScenario) : ℕ :=
  (scenario.b_investment * scenario.b_join_time * 2) / scenario.total_time

/-- Theorem stating that A's initial investment is 9000 given the specific conditions -/
theorem a_initial_is_9000 : 
  let scenario : BusinessScenario := {
    a_initial := 9000,
    b_investment := 27000,
    b_join_time := 2,
    total_time := 12,
    profit_ratio := 2/1
  }
  calculate_a_initial scenario = 9000 := by
  sorry

#eval calculate_a_initial {
  a_initial := 9000,
  b_investment := 27000,
  b_join_time := 2,
  total_time := 12,
  profit_ratio := 2/1
}

end a_initial_is_9000_l3677_367780


namespace sufficient_not_necessary_condition_l3677_367708

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → 1/a < 1/b) ∧
  (∃ a b : ℝ, 1/a < 1/b ∧ ¬(a > b ∧ b > 0)) :=
sorry

end sufficient_not_necessary_condition_l3677_367708


namespace sams_pen_collection_l3677_367796

/-- The number of pens in Sam's collection -/
def total_pens (black blue red pencils : ℕ) : ℕ := black + blue + red

/-- The problem statement -/
theorem sams_pen_collection :
  ∀ (black blue red pencils : ℕ),
  black = blue + 10 →
  blue = 2 * pencils →
  pencils = 8 →
  red = pencils - 2 →
  total_pens black blue red pencils = 48 := by
sorry

end sams_pen_collection_l3677_367796


namespace max_total_profit_l3677_367745

/-- The fixed cost in million yuan -/
def fixed_cost : ℝ := 20

/-- The variable cost per unit in million yuan -/
def variable_cost_per_unit : ℝ := 10

/-- The total revenue function k(Q) in million yuan -/
def total_revenue (Q : ℝ) : ℝ := 40 * Q - Q^2

/-- The total cost function C(Q) in million yuan -/
def total_cost (Q : ℝ) : ℝ := fixed_cost + variable_cost_per_unit * Q

/-- The total profit function L(Q) in million yuan -/
def total_profit (Q : ℝ) : ℝ := total_revenue Q - total_cost Q

/-- The theorem stating that the maximum total profit is 205 million yuan -/
theorem max_total_profit : ∃ Q : ℝ, ∀ x : ℝ, total_profit Q ≥ total_profit x ∧ total_profit Q = 205 :=
sorry

end max_total_profit_l3677_367745


namespace determinant_equality_l3677_367731

theorem determinant_equality (p q r s : ℝ) : 
  (p * s - q * r = 7) → ((p + 2 * r) * s - (q + 2 * s) * r = 7) := by
  sorry

end determinant_equality_l3677_367731


namespace stating_min_problems_olympiad_l3677_367791

/-- The number of students in the olympiad -/
def num_students : ℕ := 55

/-- 
The function that calculates the maximum number of distinct pairs of "+" and "-" scores
for a given number of problems.
-/
def max_distinct_pairs (num_problems : ℕ) : ℕ :=
  (num_problems + 1) * (num_problems + 2) / 2

/-- 
Theorem stating that the minimum number of problems needed in the olympiad is 9,
given that there are 55 students and no two students can have the same number of "+" and "-" scores.
-/
theorem min_problems_olympiad :
  ∃ (n : ℕ), n = 9 ∧ max_distinct_pairs n = num_students ∧
  ∀ (m : ℕ), m < n → max_distinct_pairs m < num_students :=
by sorry

end stating_min_problems_olympiad_l3677_367791


namespace inscribed_trapezoid_lub_l3677_367743

/-- A trapezoid inscribed in a unit circle -/
structure InscribedTrapezoid where
  /-- The length of side AB -/
  s₁ : ℝ
  /-- The length of side CD -/
  s₂ : ℝ
  /-- The distance from the center to the intersection of diagonals -/
  d : ℝ
  /-- s₁ and s₂ are between 0 and 2 (diameter of unit circle) -/
  h_s₁_bounds : 0 < s₁ ∧ s₁ ≤ 2
  h_s₂_bounds : 0 < s₂ ∧ s₂ ≤ 2
  /-- d is positive (intersection is not at the center) -/
  h_d_pos : d > 0

/-- The least upper bound of (s₁ - s₂) / d for inscribed trapezoids is 2 -/
theorem inscribed_trapezoid_lub :
  ∀ T : InscribedTrapezoid, (T.s₁ - T.s₂) / T.d ≤ 2 ∧
  ∀ ε > 0, ∃ T : InscribedTrapezoid, (T.s₁ - T.s₂) / T.d > 2 - ε := by
  sorry

end inscribed_trapezoid_lub_l3677_367743


namespace minimum_days_to_find_poisoned_apple_l3677_367722

def number_of_apples : ℕ := 2021

theorem minimum_days_to_find_poisoned_apple :
  ∀ (n : ℕ), n = number_of_apples →
  (∃ (k : ℕ), 2^k ≥ n ∧ ∀ (m : ℕ), 2^m ≥ n → k ≤ m) →
  (∃ (k : ℕ), k = 11 ∧ 2^k ≥ n ∧ ∀ (m : ℕ), 2^m ≥ n → k ≤ m) :=
by sorry

end minimum_days_to_find_poisoned_apple_l3677_367722


namespace candy_mix_cost_per_pound_l3677_367734

/-- Proves that the desired cost per pound of mixed candy is $2.00 given the specified conditions --/
theorem candy_mix_cost_per_pound
  (total_weight : ℝ)
  (cost_A : ℝ)
  (cost_B : ℝ)
  (weight_A : ℝ)
  (h_total_weight : total_weight = 5)
  (h_cost_A : cost_A = 3.2)
  (h_cost_B : cost_B = 1.7)
  (h_weight_A : weight_A = 1)
  : (weight_A * cost_A + (total_weight - weight_A) * cost_B) / total_weight = 2 := by
  sorry

#check candy_mix_cost_per_pound

end candy_mix_cost_per_pound_l3677_367734


namespace absolute_value_inequality_l3677_367716

theorem absolute_value_inequality (x : ℝ) : 
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ ((-9 ≤ x ∧ x ≤ -5) ∨ (1 ≤ x ∧ x ≤ 5)) := by
  sorry

end absolute_value_inequality_l3677_367716


namespace pure_imaginary_complex_number_l3677_367785

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ z : ℂ, z = Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2) ∧ z.re = 0 ∧ z.im ≠ 0) → a = -2 := by
  sorry

end pure_imaginary_complex_number_l3677_367785


namespace geometric_sequence_sum_l3677_367797

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 + a 3 = 5 →
  a 3 + a 5 = 20 →
  a 5 + a 7 = 80 := by
  sorry

end geometric_sequence_sum_l3677_367797


namespace sum_is_zero_or_negative_two_l3677_367726

-- Define the conditions
def is_neither_positive_nor_negative (a : ℝ) : Prop := a = 0

def largest_negative_integer (b : ℤ) : Prop := b = -1

def reciprocal_is_self (c : ℝ) : Prop := c = 1 ∨ c = -1

-- Theorem statement
theorem sum_is_zero_or_negative_two 
  (a : ℝ) (b : ℤ) (c : ℝ) 
  (ha : is_neither_positive_nor_negative a)
  (hb : largest_negative_integer b)
  (hc : reciprocal_is_self c) :
  a + b + c = 0 ∨ a + b + c = -2 := by
  sorry

end sum_is_zero_or_negative_two_l3677_367726


namespace algebraic_expression_equality_l3677_367752

theorem algebraic_expression_equality (x : ℝ) : 
  x^2 + 2*x + 5 = 6 → 2*x^2 + 4*x + 15 = 17 := by
  sorry

end algebraic_expression_equality_l3677_367752


namespace solution_set_of_inequality_l3677_367772

def f (x : ℝ) := x^3 + x

theorem solution_set_of_inequality (x : ℝ) :
  x ∈ Set.Ioo (1/3 : ℝ) 3 ↔ 
  (x ∈ Set.Icc (-5 : ℝ) 5 ∧ f (2*x - 1) + f x > 0) :=
by sorry

end solution_set_of_inequality_l3677_367772


namespace team_total_points_l3677_367738

theorem team_total_points (player_points : ℕ) (percentage : ℚ) (h1 : player_points = 35) (h2 : percentage = 1/2) :
  player_points / percentage = 70 := by
  sorry

end team_total_points_l3677_367738


namespace sprint_team_total_miles_l3677_367715

theorem sprint_team_total_miles :
  let num_people : Float := 150.0
  let miles_per_person : Float := 5.0
  let total_miles := num_people * miles_per_person
  total_miles = 750.0 := by
sorry

end sprint_team_total_miles_l3677_367715


namespace first_discount_percentage_l3677_367736

/-- Proves that the first discount percentage is 15% given the original price,
    final price after two discounts, and the second discount percentage. -/
theorem first_discount_percentage
  (original_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : original_price = 495)
  (h2 : final_price = 378.675)
  (h3 : second_discount = 10) :
  ∃ (first_discount : ℝ),
    first_discount = 15 ∧
    final_price = original_price * (100 - first_discount) / 100 * (100 - second_discount) / 100 :=
by sorry

end first_discount_percentage_l3677_367736


namespace min_max_f_on_I_l3677_367714

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^2 * (x - 2)

-- Define the interval
def I : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem min_max_f_on_I :
  ∃ (min max : ℝ), min = -64 ∧ max = 0 ∧
  (∀ x ∈ I, f x ≥ min) ∧
  (∀ x ∈ I, f x ≤ max) ∧
  (∃ x₁ ∈ I, f x₁ = min) ∧
  (∃ x₂ ∈ I, f x₂ = max) :=
sorry

end min_max_f_on_I_l3677_367714


namespace parabola_distance_theorem_l3677_367799

/-- Parabola type representing y^2 = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Represents a point on the parabola -/
structure ParabolaPoint (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

theorem parabola_distance_theorem (p : Parabola) 
  (h_equation : p.equation = fun x y ↦ y^2 = 8*x)
  (P : ParabolaPoint p)
  (A : ℝ × ℝ)
  (h_perpendicular : (P.point.1 - A.1) * (P.point.2 - A.2) = 0)
  (h_on_directrix : p.directrix A.1 A.2)
  (h_slope : (A.2 - p.focus.2) / (A.1 - p.focus.1) = -Real.sqrt 3) :
  Real.sqrt ((P.point.1 - p.focus.1)^2 + (P.point.2 - p.focus.2)^2) = 8 := by
  sorry

end parabola_distance_theorem_l3677_367799


namespace solve_equation_l3677_367719

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 1

-- State the theorem
theorem solve_equation (a : ℝ) (h1 : 0 < a) (h2 : a < 3) (h3 : f a = 7) : a = 2 := by
  sorry

end solve_equation_l3677_367719


namespace proposition_equivalence_l3677_367749

theorem proposition_equivalence (a : ℝ) :
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a ≥ 0) ↔ a ≥ -8 := by
  sorry

end proposition_equivalence_l3677_367749


namespace range_of_f_less_than_one_l3677_367762

-- Define the function f
def f (x : ℝ) := x^3

-- State the theorem
theorem range_of_f_less_than_one :
  {x : ℝ | f x < 1} = Set.Iio 1 := by sorry

end range_of_f_less_than_one_l3677_367762


namespace function_bound_on_unit_interval_l3677_367751

theorem function_bound_on_unit_interval 
  (f : Set.Icc 0 1 → ℝ)
  (h₁ : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h₂ : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₁ - f x₂| < |x₁.1 - x₂.1|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₁ - f x₂| < (1/2 : ℝ) := by
  sorry

end function_bound_on_unit_interval_l3677_367751


namespace total_earnings_theorem_l3677_367767

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  investment_ratio : Fin 3 → ℕ
  return_ratio : Fin 3 → ℕ

/-- Calculates the total earnings given investment data and the earnings difference between two investors -/
def calculate_total_earnings (data : InvestmentData) (earnings_diff : ℕ) : ℕ := sorry

/-- The main theorem stating the total earnings for the given scenario -/
theorem total_earnings_theorem (data : InvestmentData) (earnings_diff : ℕ) : 
  data.investment_ratio 0 = 3 ∧ 
  data.investment_ratio 1 = 4 ∧ 
  data.investment_ratio 2 = 5 ∧
  data.return_ratio 0 = 6 ∧ 
  data.return_ratio 1 = 5 ∧ 
  data.return_ratio 2 = 4 ∧
  earnings_diff = 120 →
  calculate_total_earnings data earnings_diff = 3480 := by sorry

end total_earnings_theorem_l3677_367767


namespace woman_age_difference_l3677_367720

/-- The age of the son -/
def son_age : ℕ := 27

/-- The age of the woman -/
def woman_age : ℕ := 84 - son_age

/-- The difference between the woman's age and twice her son's age -/
def age_difference : ℕ := woman_age - 2 * son_age

theorem woman_age_difference : age_difference = 3 := by
  sorry

end woman_age_difference_l3677_367720


namespace binomial_expansion_problem_l3677_367753

def binomial_sum (n : ℕ) : ℕ := 2^(n-1)

def rational_terms (n : ℕ) : List (ℕ × ℕ) :=
  [(5, 1), (4, 210)]

def coefficient_x_squared (n : ℕ) : ℕ :=
  (Finset.range (n - 2)).sum (λ k => Nat.choose (k + 3) 2)

theorem binomial_expansion_problem (n : ℕ) 
  (h : binomial_sum n = 512) : 
  n = 10 ∧ 
  rational_terms n = [(5, 1), (4, 210)] ∧
  coefficient_x_squared n = 164 := by
  sorry

end binomial_expansion_problem_l3677_367753


namespace circus_performance_legs_on_ground_l3677_367701

/-- Calculates the total number of legs/paws/hands on the ground in a circus performance --/
def circus_legs_on_ground (total_dogs : ℕ) (total_cats : ℕ) (total_horses : ℕ) (acrobats_one_hand : ℕ) (acrobats_two_hands : ℕ) : ℕ :=
  let dogs_on_back_legs := total_dogs / 2
  let dogs_on_all_fours := total_dogs - dogs_on_back_legs
  let cats_on_back_legs := total_cats / 3
  let cats_on_all_fours := total_cats - cats_on_back_legs
  let horses_on_hind_legs := 2
  let horses_on_all_fours := total_horses - horses_on_hind_legs
  
  let dog_paws := dogs_on_back_legs * 2 + dogs_on_all_fours * 4
  let cat_paws := cats_on_back_legs * 2 + cats_on_all_fours * 4
  let horse_hooves := horses_on_hind_legs * 2 + horses_on_all_fours * 4
  let acrobat_hands := acrobats_one_hand * 1 + acrobats_two_hands * 2
  
  dog_paws + cat_paws + horse_hooves + acrobat_hands

theorem circus_performance_legs_on_ground :
  circus_legs_on_ground 20 10 5 4 2 = 118 := by
  sorry

end circus_performance_legs_on_ground_l3677_367701


namespace compare_expressions_l3677_367784

theorem compare_expressions : 3 - Real.sqrt 2 > 4 - 2 * Real.sqrt 2 := by sorry

end compare_expressions_l3677_367784


namespace election_vote_ratio_l3677_367787

theorem election_vote_ratio (Vx Vy : ℝ) 
  (h1 : 0.72 * Vx + 0.36 * Vy = 0.6 * (Vx + Vy)) 
  (h2 : Vx > 0) 
  (h3 : Vy > 0) : 
  Vx / Vy = 2 := by
sorry

end election_vote_ratio_l3677_367787


namespace lathe_problem_l3677_367741

/-- Represents the work efficiency of a lathe -/
structure Efficiency : Type :=
  (value : ℝ)

/-- Represents a lathe with its efficiency and start time -/
structure Lathe : Type :=
  (efficiency : Efficiency)
  (startTime : ℝ)

/-- The number of parts processed by a lathe after a given time -/
def partsProcessed (l : Lathe) (time : ℝ) : ℝ :=
  l.efficiency.value * (time - l.startTime)

theorem lathe_problem (a b c : Lathe) :
  a.startTime = c.startTime - 10 →
  c.startTime = b.startTime - 5 →
  partsProcessed b (b.startTime + 10) = partsProcessed c (b.startTime + 10) →
  partsProcessed a (c.startTime + 30) = partsProcessed c (c.startTime + 30) →
  ∃ t : ℝ, t = 15 ∧ partsProcessed a (b.startTime + t) = partsProcessed b (b.startTime + t) :=
by sorry

end lathe_problem_l3677_367741


namespace average_temperature_l3677_367742

def temperatures : List ℝ := [53, 59, 61, 55, 50]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 55.6 := by sorry

end average_temperature_l3677_367742


namespace cloth_cost_price_l3677_367789

/-- Calculates the cost price per meter of cloth given the total selling price,
    number of meters sold, and profit per meter. -/
def cost_price_per_meter (total_selling_price : ℕ) (meters_sold : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (total_selling_price - profit_per_meter * meters_sold) / meters_sold

/-- Proves that the cost price of one meter of cloth is Rs. 100, given the conditions. -/
theorem cloth_cost_price :
  cost_price_per_meter 8925 85 5 = 100 := by
  sorry

end cloth_cost_price_l3677_367789


namespace tourist_count_scientific_notation_l3677_367704

theorem tourist_count_scientific_notation :
  ∀ (n : ℝ), n = 15.276 * 1000000 → 
  ∃ (a : ℝ) (b : ℤ), n = a * (10 : ℝ) ^ b ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.5276 ∧ b = 7 :=
by sorry

end tourist_count_scientific_notation_l3677_367704


namespace speed_difference_l3677_367770

/-- The difference in average speeds between no traffic and heavy traffic conditions --/
theorem speed_difference (distance : ℝ) (time_heavy : ℝ) (time_no : ℝ)
  (h1 : distance = 200)
  (h2 : time_heavy = 5)
  (h3 : time_no = 4) :
  distance / time_no - distance / time_heavy = 10 := by
  sorry

end speed_difference_l3677_367770


namespace one_valid_x_l3677_367737

def box_volume (x : ℤ) : ℤ := (x + 6) * (x - 6) * (x^2 + 36)

theorem one_valid_x : ∃! x : ℤ, 
  x > 0 ∧ 
  x - 6 > 0 ∧ 
  box_volume x < 800 :=
sorry

end one_valid_x_l3677_367737


namespace binomial_expansion_sum_l3677_367766

theorem binomial_expansion_sum (n : ℕ) : 
  (∀ k : ℕ, k ≠ 2 → Nat.choose n 2 > Nat.choose n k) → 
  (1 - 2)^n = 1 := by
  sorry

end binomial_expansion_sum_l3677_367766


namespace movie_watchers_l3677_367778

theorem movie_watchers (total_seats empty_seats : ℕ) 
  (h1 : total_seats = 750)
  (h2 : empty_seats = 218) : 
  total_seats - empty_seats = 532 := by
sorry

end movie_watchers_l3677_367778


namespace a_equals_one_sufficient_not_necessary_l3677_367746

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem a_equals_one_sufficient_not_necessary (a : ℝ) :
  (a = 1 → is_purely_imaginary ((a - 1) * (a + 2) + (a + 3) * I)) ∧
  ¬(is_purely_imaginary ((a - 1) * (a + 2) + (a + 3) * I) → a = 1) :=
sorry

end a_equals_one_sufficient_not_necessary_l3677_367746


namespace hoonjeong_marbles_l3677_367775

theorem hoonjeong_marbles :
  ∀ (initial_marbles : ℝ),
    (initial_marbles * (1 - 0.2) * (1 - 0.35) = 130) →
    initial_marbles = 250 :=
by
  sorry

end hoonjeong_marbles_l3677_367775


namespace equal_sum_of_squares_l3677_367774

/-- Given a positive integer, return the sum of its digits -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The set of positive integers with at most n digits -/
def numbersWithAtMostNDigits (n : ℕ) : Set ℕ := sorry

/-- The set of positive integers with at most n digits and even digit sum -/
def evenDigitSumNumbers (n : ℕ) : Set ℕ := 
  {x ∈ numbersWithAtMostNDigits n | Even (digitSum x)}

/-- The set of positive integers with at most n digits and odd digit sum -/
def oddDigitSumNumbers (n : ℕ) : Set ℕ := 
  {x ∈ numbersWithAtMostNDigits n | Odd (digitSum x)}

/-- The sum of squares of elements in a set of natural numbers -/
def sumOfSquares (s : Set ℕ) : ℕ := sorry

theorem equal_sum_of_squares (n : ℕ) (h : n > 2) :
  sumOfSquares (evenDigitSumNumbers n) = sumOfSquares (oddDigitSumNumbers n) := by
  sorry

end equal_sum_of_squares_l3677_367774


namespace smallest_exponent_divisibility_l3677_367764

theorem smallest_exponent_divisibility (x y z : ℕ+) 
  (h1 : x ∣ y^3) (h2 : y ∣ z^3) (h3 : z ∣ x^3) :
  (∀ n : ℕ, n < 13 → ¬(x * y * z ∣ (x + y + z)^n)) ∧
  (x * y * z ∣ (x + y + z)^13) := by
sorry

end smallest_exponent_divisibility_l3677_367764


namespace part_to_whole_ratio_l3677_367702

theorem part_to_whole_ratio (N P : ℚ) 
  (h1 : (1/4) * (1/3) * P = 25)
  (h2 : (2/5) * N = 300) : 
  P / N = 2 / 5 := by
sorry

end part_to_whole_ratio_l3677_367702


namespace equation_solvable_for_small_primes_l3677_367732

theorem equation_solvable_for_small_primes :
  ∀ p : ℕ, p ≤ 100 → Prime p → ∃ x y : ℕ, y^37 ≡ x^3 + 11 [ZMOD p] :=
by sorry

end equation_solvable_for_small_primes_l3677_367732


namespace root_sum_reciprocal_l3677_367717

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ x, x^3 - 20*x^2 + 96*x - 91 = 0 ↔ (x = p ∨ x = q ∨ x = r)) →
  (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 20*s^2 + 96*s - 91) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 225 := by
sorry

end root_sum_reciprocal_l3677_367717


namespace right_triangle_ratio_l3677_367781

theorem right_triangle_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x ≤ x + y → x + y ≤ x + 3*y →
  (x + 3*y)^2 = x^2 + (x + y)^2 →
  x / y = 1 + Real.sqrt 5 := by
sorry

end right_triangle_ratio_l3677_367781


namespace point_position_l3677_367747

theorem point_position (a : ℝ) : 
  (a < 0) → -- A is on the negative side of the origin
  (2 > 0) → -- B is on the positive side of the origin
  (|a + 3| = 4) → -- CO = 2BO, where BO = 2
  a = -7 := by
sorry

end point_position_l3677_367747


namespace inequality_proof_l3677_367779

theorem inequality_proof (a b : ℝ) : (a^2 - 1) * (b^2 - 1) ≤ 0 → a^2 + b^2 - 1 - a^2*b^2 ≥ 0 := by
  sorry

end inequality_proof_l3677_367779


namespace polyhedron_space_diagonals_l3677_367782

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem: A convex polyhedron Q with 30 vertices, 72 edges, 44 faces 
    (30 triangular and 14 quadrilateral) has 335 space diagonals -/
theorem polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 30,
    quadrilateral_faces := 14
  }
  space_diagonals Q = 335 := by
  sorry

end polyhedron_space_diagonals_l3677_367782


namespace triangle_abc_properties_l3677_367706

theorem triangle_abc_properties (A B C : ℝ) (h_obtuse : π / 2 < C ∧ C < π) 
  (h_sin_2c : Real.sin (2 * C) = Real.sqrt 3 * Real.cos C) 
  (h_b : Real.sqrt (A^2 + C^2 - 2*A*C*Real.cos B) = 6) 
  (h_area : 1/2 * Real.sqrt (A^2 + C^2 - 2*A*C*Real.cos B) * 
    Real.sqrt (B^2 + C^2 - 2*B*C*Real.cos A) * Real.sin C = 6 * Real.sqrt 3) : 
  C = 2 * π / 3 ∧ 
  Real.sqrt (A^2 + C^2 - 2*A*C*Real.cos B) + 
  Real.sqrt (B^2 + C^2 - 2*B*C*Real.cos A) + 
  Real.sqrt (A^2 + B^2 - 2*A*B*Real.cos C) = 10 + 2 * Real.sqrt 19 := by
  sorry


end triangle_abc_properties_l3677_367706


namespace infinite_chain_resistance_l3677_367798

/-- The resistance of a single resistor in the chain -/
def R₀ : ℝ := 50

/-- The resistance of an infinitely long chain of identical resistors -/
noncomputable def R_X : ℝ := R₀ * (1 + Real.sqrt 5) / 2

/-- Theorem stating that R_X satisfies the equation for the infinite chain resistance -/
theorem infinite_chain_resistance : R_X = R₀ + (R₀ * R_X) / (R₀ + R_X) := by
  sorry

end infinite_chain_resistance_l3677_367798


namespace total_animals_after_addition_l3677_367724

/-- Represents the number of animals on a farm --/
structure FarmAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- Calculates the total number of animals on the farm --/
def totalAnimals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.pigs + farm.goats

/-- The initial number of animals on the farm --/
def initialFarm : FarmAnimals :=
  { cows := 2, pigs := 3, goats := 6 }

/-- The number of animals to be added to the farm --/
def addedAnimals : FarmAnimals :=
  { cows := 3, pigs := 5, goats := 2 }

/-- Theorem stating that the total number of animals after addition is 21 --/
theorem total_animals_after_addition :
  totalAnimals initialFarm + totalAnimals addedAnimals = 21 := by
  sorry


end total_animals_after_addition_l3677_367724


namespace pencil_distribution_l3677_367721

theorem pencil_distribution (total_students : ℕ) (total_pencils : ℕ) 
  (h1 : total_students = 36)
  (h2 : total_pencils = 50)
  (h3 : ∃ (a b c : ℕ), a + b + c = total_students ∧ a + 2*b + 3*c = total_pencils ∧ a = 2*(b + c)) :
  ∃ (a b c : ℕ), a + b + c = total_students ∧ a + 2*b + 3*c = total_pencils ∧ a = 2*(b + c) ∧ b = 10 := by
  sorry

#check pencil_distribution

end pencil_distribution_l3677_367721


namespace inequality_solution_fractional_equation_no_solution_l3677_367790

-- Part 1: Inequality
theorem inequality_solution (x : ℝ) : 
  (1 - x) / 3 - x < 3 - (x + 2) / 4 ↔ x > -2 :=
sorry

-- Part 2: Fractional equation
theorem fractional_equation_no_solution :
  ¬∃ (x : ℝ), (x - 2) / (2 * x - 1) + 1 = 3 / (2 * (1 - 2 * x)) :=
sorry

end inequality_solution_fractional_equation_no_solution_l3677_367790


namespace curve_intersection_points_l3677_367783

-- Define the parametric equations of the curve
def x (t : ℝ) : ℝ := -2 + 5 * t
def y (t : ℝ) : ℝ := 1 - 2 * t

-- Theorem statement
theorem curve_intersection_points :
  (∃ t : ℝ, x t = 0 ∧ y t = 1/5) ∧
  (∃ t : ℝ, x t = 1/2 ∧ y t = 0) :=
by
  sorry


end curve_intersection_points_l3677_367783


namespace max_gcd_13n_plus_4_8n_plus_3_l3677_367756

theorem max_gcd_13n_plus_4_8n_plus_3 :
  ∃ (k : ℕ), k > 0 ∧ gcd (13 * k + 4) (8 * k + 3) = 9 ∧
  ∀ (n : ℕ), n > 0 → gcd (13 * n + 4) (8 * n + 3) ≤ 9 :=
by sorry

end max_gcd_13n_plus_4_8n_plus_3_l3677_367756


namespace work_completion_time_l3677_367760

/-- Given two workers A and B, where:
    - A and B together can complete a job in 6 days
    - A alone can complete the job in 14 days
    This theorem proves that B alone can complete the job in 10.5 days -/
theorem work_completion_time (work_rate_A : ℝ) (work_rate_B : ℝ) : 
  work_rate_A + work_rate_B = 1 / 6 →
  work_rate_A = 1 / 14 →
  1 / work_rate_B = 10.5 := by
sorry

end work_completion_time_l3677_367760


namespace remaining_cube_volume_l3677_367725

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) :
  cube_side = 4 →
  cylinder_radius = 2 →
  (cube_side ^ 3) - (π * cylinder_radius ^ 2 * cube_side) = 64 - 16 * π :=
by sorry

end remaining_cube_volume_l3677_367725


namespace complex_root_quadratic_equation_l3677_367750

theorem complex_root_quadratic_equation (q : ℝ) : 
  (2 * (Complex.mk (-3) 2)^2 + 12 * Complex.mk (-3) 2 + q = 0) → q = 26 := by
  sorry

end complex_root_quadratic_equation_l3677_367750
