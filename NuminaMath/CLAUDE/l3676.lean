import Mathlib

namespace charlie_feathers_l3676_367619

theorem charlie_feathers (total_needed : ℕ) (still_needed : ℕ) 
  (h1 : total_needed = 900)
  (h2 : still_needed = 513) :
  total_needed - still_needed = 387 := by
  sorry

end charlie_feathers_l3676_367619


namespace sum_of_decimals_l3676_367622

theorem sum_of_decimals : (5.47 + 4.96 : ℝ) = 10.43 := by
  sorry

end sum_of_decimals_l3676_367622


namespace weighted_distances_sum_l3676_367614

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  V : ℝ  -- Volume
  S : Fin 4 → ℝ  -- Face areas
  d : Fin 4 → ℝ  -- Distances from a point to each face
  k : ℝ  -- Constant ratio
  h_positive : V > 0
  S_positive : ∀ i, S i > 0
  d_positive : ∀ i, d i > 0
  k_positive : k > 0
  h_ratio : ∀ i : Fin 4, S i / (i.val + 1 : ℝ) = k

/-- The sum of weighted distances equals three times the volume divided by k -/
theorem weighted_distances_sum (p : TriangularPyramid) :
  (p.d 0) + 2 * (p.d 1) + 3 * (p.d 2) + 4 * (p.d 3) = 3 * p.V / p.k := by
  sorry

end weighted_distances_sum_l3676_367614


namespace population_change_l3676_367638

theorem population_change (P : ℝ) : 
  P > 0 →
  (P * 1.25 * 0.75 = 18750) →
  P = 20000 := by
sorry

end population_change_l3676_367638


namespace temperature_stats_l3676_367699

def temperatures : List ℝ := [12, 9, 10, 6, 11, 12, 17]

def median (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

theorem temperature_stats :
  median temperatures = 11 ∧ range temperatures = 11 := by sorry

end temperature_stats_l3676_367699


namespace number_of_friends_l3676_367649

-- Define the total number of stickers
def total_stickers : ℕ := 72

-- Define the number of stickers each friend receives
def stickers_per_friend : ℕ := 8

-- Theorem to prove the number of friends receiving stickers
theorem number_of_friends : total_stickers / stickers_per_friend = 9 := by
  sorry

end number_of_friends_l3676_367649


namespace square_area_from_diagonal_l3676_367617

/-- Given a square with diagonal length 40, prove its area is 800 -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 40) : d^2 / 2 = 800 := by
  sorry

end square_area_from_diagonal_l3676_367617


namespace candy_bar_division_l3676_367646

theorem candy_bar_division (total_candy : ℝ) (num_bags : ℕ) 
  (h1 : total_candy = 15.5) 
  (h2 : num_bags = 5) : 
  total_candy / (num_bags : ℝ) = 3.1 := by
  sorry

end candy_bar_division_l3676_367646


namespace jessicas_mothers_death_years_jessicas_mothers_death_years_proof_l3676_367690

/-- Prove that the number of years passed since Jessica's mother's death is 10 -/
theorem jessicas_mothers_death_years : ℕ :=
  let jessica_current_age : ℕ := 40
  let mother_hypothetical_age : ℕ := 70
  let years_passed : ℕ → Prop := λ x =>
    -- Jessica was half her mother's age when her mother died
    2 * (jessica_current_age - x) = jessica_current_age - x + x ∧
    -- Jessica's mother would be 70 if she were alive now
    jessica_current_age - x + x = mother_hypothetical_age
  10

theorem jessicas_mothers_death_years_proof :
  jessicas_mothers_death_years = 10 := by sorry

end jessicas_mothers_death_years_jessicas_mothers_death_years_proof_l3676_367690


namespace paint_usage_fraction_l3676_367610

theorem paint_usage_fraction (initial_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) :
  initial_paint = 360 →
  first_week_fraction = 1/4 →
  total_used = 180 →
  let remaining_after_first_week := initial_paint - first_week_fraction * initial_paint
  let used_second_week := total_used - first_week_fraction * initial_paint
  used_second_week / remaining_after_first_week = 1/3 := by
  sorry

end paint_usage_fraction_l3676_367610


namespace author_earnings_calculation_l3676_367628

def author_earnings (paperback_copies : Nat) (paperback_price : Real)
                    (hardcover_copies : Nat) (hardcover_price : Real)
                    (ebook_copies : Nat) (ebook_price : Real)
                    (audiobook_copies : Nat) (audiobook_price : Real) : Real :=
  let paperback_sales := paperback_copies * paperback_price
  let hardcover_sales := hardcover_copies * hardcover_price
  let ebook_sales := ebook_copies * ebook_price
  let audiobook_sales := audiobook_copies * audiobook_price
  0.06 * paperback_sales + 0.12 * hardcover_sales + 0.08 * ebook_sales + 0.10 * audiobook_sales

theorem author_earnings_calculation :
  author_earnings 32000 0.20 15000 0.40 10000 0.15 5000 0.50 = 1474 :=
by sorry

end author_earnings_calculation_l3676_367628


namespace quadrilateral_division_theorem_l3676_367661

/-- Represents a convex quadrilateral with areas of its four parts --/
structure ConvexQuadrilateral :=
  (area1 : ℝ)
  (area2 : ℝ)
  (area3 : ℝ)
  (area4 : ℝ)

/-- The theorem stating the relationship between the areas of the four parts --/
theorem quadrilateral_division_theorem (q : ConvexQuadrilateral) 
  (h1 : q.area1 = 360)
  (h2 : q.area2 = 720)
  (h3 : q.area3 = 900) :
  q.area4 = 540 := by
  sorry

#check quadrilateral_division_theorem

end quadrilateral_division_theorem_l3676_367661


namespace point_c_coordinates_l3676_367612

/-- Given two points A and B in ℝ², if vector BC is half of vector BA, 
    then the coordinates of point C are (0, 3/2) -/
theorem point_c_coordinates 
  (A B : ℝ × ℝ)
  (h_A : A = (1, 1))
  (h_B : B = (-1, 2))
  (h_BC : ∃ (C : ℝ × ℝ), C - B = (1/2) • (A - B)) :
  ∃ (C : ℝ × ℝ), C = (0, 3/2) := by
sorry

end point_c_coordinates_l3676_367612


namespace certain_amount_proof_l3676_367639

theorem certain_amount_proof (A : ℝ) : 
  (0.20 * 1050 = 0.15 * 1500 - A) → A = 15 := by
  sorry

end certain_amount_proof_l3676_367639


namespace point_four_units_from_one_l3676_367620

theorem point_four_units_from_one (x : ℝ) : 
  (x = 1 + 4 ∨ x = 1 - 4) ↔ (x = 5 ∨ x = -3) :=
by sorry

end point_four_units_from_one_l3676_367620


namespace correct_ring_arrangements_l3676_367657

/-- The number of ways to arrange rings on fingers -/
def ring_arrangements (total_rings : ℕ) (rings_to_arrange : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings rings_to_arrange *
  Nat.choose (rings_to_arrange + fingers - 1) (fingers - 1) *
  Nat.factorial rings_to_arrange

/-- Theorem stating the correct number of ring arrangements -/
theorem correct_ring_arrangements :
  ring_arrangements 7 6 4 = 423360 := by
  sorry

end correct_ring_arrangements_l3676_367657


namespace select_three_roles_from_25_l3676_367659

/-- The number of ways to select three distinct roles from a squad of players. -/
def selectThreeRoles (squadSize : ℕ) : ℕ :=
  squadSize * (squadSize - 1) * (squadSize - 2)

/-- Theorem: The number of ways to select a captain, vice-captain, and goalkeeper
    from a squad of 25 players, where no player can occupy more than one role, is 13800. -/
theorem select_three_roles_from_25 : selectThreeRoles 25 = 13800 := by
  sorry

end select_three_roles_from_25_l3676_367659


namespace twelve_person_tournament_matches_l3676_367668

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a 12-person round-robin tournament, the number of matches is 66 -/
theorem twelve_person_tournament_matches : num_matches 12 = 66 := by
  sorry

end twelve_person_tournament_matches_l3676_367668


namespace alternating_sum_of_coefficients_l3676_367632

theorem alternating_sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  |a₀| - |a₁| + |a₂| - |a₃| + |a₄| - |a₅| = 1 := by
  sorry

end alternating_sum_of_coefficients_l3676_367632


namespace davids_remaining_money_l3676_367696

/-- The amount of money David has left after mowing lawns, buying shoes, and giving money to his mom. -/
def davidsRemainingMoney (hourlyRate : ℚ) (hoursPerDay : ℚ) (daysPerWeek : ℕ) : ℚ :=
  let totalEarned := hourlyRate * hoursPerDay * daysPerWeek
  let afterShoes := totalEarned / 2
  afterShoes / 2

theorem davids_remaining_money :
  davidsRemainingMoney 14 2 7 = 49 := by
  sorry

#eval davidsRemainingMoney 14 2 7

end davids_remaining_money_l3676_367696


namespace fraction_less_than_decimal_l3676_367688

theorem fraction_less_than_decimal : (7 : ℚ) / 24 < (3 : ℚ) / 10 := by
  sorry

end fraction_less_than_decimal_l3676_367688


namespace volleyball_team_combinations_l3676_367642

def total_players : ℕ := 16
def num_triplets : ℕ := 3
def num_twins : ℕ := 2
def starters : ℕ := 6

def choose_two_triplets : ℕ := Nat.choose num_triplets 2
def remaining_after_triplets : ℕ := total_players - num_triplets + 1
def choose_rest_with_triplets : ℕ := Nat.choose remaining_after_triplets (starters - 2)

def choose_twins : ℕ := 1
def remaining_after_twins : ℕ := total_players - num_twins
def choose_rest_with_twins : ℕ := Nat.choose remaining_after_twins (starters - 2)

theorem volleyball_team_combinations :
  choose_two_triplets * choose_rest_with_triplets + choose_twins * choose_rest_with_twins = 3146 :=
sorry

end volleyball_team_combinations_l3676_367642


namespace sum_of_squares_l3676_367698

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 10)
  (eq2 : b^2 + 5*c = 0)
  (eq3 : c^2 + 7*a = -21) :
  a^2 + b^2 + c^2 = 83/4 := by
sorry

end sum_of_squares_l3676_367698


namespace fraction_calls_team_B_value_l3676_367697

/-- Represents the fraction of calls processed by team B in a call center scenario -/
def fraction_calls_team_B (num_agents_A num_agents_B : ℚ) 
  (calls_per_agent_A calls_per_agent_B : ℚ) : ℚ :=
  (num_agents_B * calls_per_agent_B) / 
  (num_agents_A * calls_per_agent_A + num_agents_B * calls_per_agent_B)

/-- Theorem stating the fraction of calls processed by team B -/
theorem fraction_calls_team_B_value 
  (num_agents_A num_agents_B : ℚ) 
  (calls_per_agent_A calls_per_agent_B : ℚ) 
  (h1 : num_agents_A = (5 / 8) * num_agents_B)
  (h2 : calls_per_agent_A = (6 / 5) * calls_per_agent_B) :
  fraction_calls_team_B num_agents_A num_agents_B calls_per_agent_A calls_per_agent_B = 4 / 7 := by
  sorry


end fraction_calls_team_B_value_l3676_367697


namespace range_of_m_range_of_x_l3676_367609

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x + 1 ∧ x + 1 < 1 + m ∧ m > 0

-- Theorem 1
theorem range_of_m (m : ℝ) :
  (∀ x, ¬(p x) → ¬(q x m)) →
  0 < m ∧ m ≤ 1 :=
sorry

-- Theorem 2
theorem range_of_x (x : ℝ) :
  (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) →
  (-5 ≤ x ∧ x < -1) ∨ x = 5 :=
sorry

end range_of_m_range_of_x_l3676_367609


namespace min_value_implies_a_l3676_367666

/-- Given a function f(x) = 4x + a/x where x > 0 and a > 0, 
    if f attains its minimum value at x = 2, then a = 16 -/
theorem min_value_implies_a (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, 4 * x + a / x ≥ 4 * 2 + a / 2) →
  (∃ x > 0, 4 * x + a / x = 4 * 2 + a / 2) →
  a = 16 := by
  sorry

end min_value_implies_a_l3676_367666


namespace number_calculation_l3676_367685

theorem number_calculation (x : ℚ) : (x - 2) / 13 = 4 → (x - 5) / 7 = 7 := by
  sorry

end number_calculation_l3676_367685


namespace average_payment_is_460_l3676_367647

/-- The total number of installments -/
def total_installments : ℕ := 52

/-- The number of initial payments -/
def initial_payments : ℕ := 12

/-- The amount of each initial payment -/
def initial_payment_amount : ℚ := 410

/-- The additional amount for each remaining payment -/
def additional_amount : ℚ := 65

/-- The amount of each remaining payment -/
def remaining_payment_amount : ℚ := initial_payment_amount + additional_amount

/-- The number of remaining payments -/
def remaining_payments : ℕ := total_installments - initial_payments

theorem average_payment_is_460 :
  (initial_payments * initial_payment_amount + remaining_payments * remaining_payment_amount) / total_installments = 460 := by
  sorry

end average_payment_is_460_l3676_367647


namespace age_problem_l3676_367684

theorem age_problem (p q : ℕ) : 
  (p - 8 = (q - 8) / 2) →  -- 8 years ago, p was half of q's age
  (p * 4 = q * 3) →        -- The ratio of their present ages is 3:4
  (p + q = 28) :=          -- The total of their present ages is 28
by sorry

end age_problem_l3676_367684


namespace apple_distribution_l3676_367653

theorem apple_distribution (total_apples : ℕ) (alice_min : ℕ) (becky_min : ℕ) (chris_min : ℕ)
  (h1 : total_apples = 30)
  (h2 : alice_min = 3)
  (h3 : becky_min = 2)
  (h4 : chris_min = 2) :
  (Nat.choose (total_apples - alice_min - becky_min - chris_min + 2) 2) = 300 := by
  sorry

end apple_distribution_l3676_367653


namespace point_on_curve_l3676_367671

def curve (x y : ℝ) : Prop := x^2 - x*y + 2*y + 1 = 0

theorem point_on_curve :
  curve 0 (-1/2) ∧
  ¬ curve 0 0 ∧
  ¬ curve 1 (-1) ∧
  ¬ curve 1 1 := by
  sorry

end point_on_curve_l3676_367671


namespace sin_sum_zero_l3676_367600

theorem sin_sum_zero : 
  Real.sin (-1071 * π / 180) * Real.sin (99 * π / 180) + 
  Real.sin (-171 * π / 180) * Real.sin (-261 * π / 180) = 0 := by
  sorry

end sin_sum_zero_l3676_367600


namespace cunningham_lambs_count_l3676_367689

/-- Represents the total number of lambs owned by farmer Cunningham -/
def total_lambs : ℕ := 6048

/-- Represents the number of white lambs -/
def white_lambs : ℕ := 193

/-- Represents the number of black lambs -/
def black_lambs : ℕ := 5855

/-- Theorem stating that the total number of lambs is the sum of white and black lambs -/
theorem cunningham_lambs_count : total_lambs = white_lambs + black_lambs := by
  sorry

end cunningham_lambs_count_l3676_367689


namespace fraction_simplification_l3676_367604

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by
  sorry

end fraction_simplification_l3676_367604


namespace max_soap_boxes_in_carton_l3676_367625

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dims : BoxDimensions) : ℕ :=
  dims.length * dims.width * dims.height

/-- Represents the carton dimensions -/
def cartonDims : BoxDimensions :=
  { length := 25, width := 35, height := 50 }

/-- Represents the soap box dimensions -/
def soapBoxDims : BoxDimensions :=
  { length := 8, width := 7, height := 6 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDims) / (boxVolume soapBoxDims) = 130 := by
  sorry

end max_soap_boxes_in_carton_l3676_367625


namespace inequality_and_min_value_l3676_367656

theorem inequality_and_min_value (a b x y : ℝ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) (h4 : x > 0) (h5 : y > 0) :
  (a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y)) ∧
  (a^2 / x + b^2 / y = (a + b)^2 / (x + y) ↔ x / y = a / b) ∧
  (∀ x ∈ Set.Ioo 0 (1/2), 2/x + 9/(1-2*x) ≥ 25) ∧
  (2/(1/5) + 9/(1-2*(1/5)) = 25) := by
  sorry

end inequality_and_min_value_l3676_367656


namespace sector_angle_l3676_367680

theorem sector_angle (r : ℝ) (α : ℝ) 
  (h1 : 2 * r + α * r = 4)  -- circumference of sector is 4
  (h2 : (1 / 2) * α * r^2 = 1)  -- area of sector is 1
  : α = 2 := by
  sorry

end sector_angle_l3676_367680


namespace profit_percentage_previous_year_l3676_367634

theorem profit_percentage_previous_year
  (revenue_prev : ℝ)
  (profit_prev : ℝ)
  (revenue_decrease : ℝ)
  (profit_percentage_2009 : ℝ)
  (profit_increase : ℝ)
  (h1 : revenue_decrease = 0.2)
  (h2 : profit_percentage_2009 = 0.15)
  (h3 : profit_increase = 1.5)
  (h4 : profit_prev > 0)
  (h5 : revenue_prev > 0) :
  profit_prev / revenue_prev = 0.08 := by
sorry

end profit_percentage_previous_year_l3676_367634


namespace fourth_power_sum_l3676_367669

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (square_sum_condition : a^2 + b^2 + c^2 = 3)
  (cube_sum_condition : a^3 + b^3 + c^3 = 4) :
  a^4 + b^4 + c^4 = 7.833 := by
  sorry

end fourth_power_sum_l3676_367669


namespace average_geometric_sequence_l3676_367602

theorem average_geometric_sequence (z : ℝ) : 
  (z + 3*z + 9*z + 27*z + 81*z) / 5 = 24.2 * z := by
  sorry

end average_geometric_sequence_l3676_367602


namespace necessary_condition_propositions_l3676_367629

-- Definition for necessary condition
def is_necessary_condition (p q : Prop) : Prop :=
  q → p

-- Proposition A
def prop_a (x y : ℝ) : Prop :=
  is_necessary_condition (x^2 > y^2) (x > y)

-- Proposition B
def prop_b (x : ℝ) : Prop :=
  is_necessary_condition (x > 5) (x > 10)

-- Proposition C
def prop_c (a b c : ℝ) : Prop :=
  is_necessary_condition (a * c = b * c) (a = b)

-- Proposition D
def prop_d (x y : ℝ) : Prop :=
  is_necessary_condition (2 * x + 1 = 2 * y + 1) (x = y)

-- Theorem stating which propositions have p as a necessary condition for q
theorem necessary_condition_propositions :
  (∃ x y : ℝ, ¬(prop_a x y)) ∧
  (∀ x : ℝ, prop_b x) ∧
  (∀ a b c : ℝ, c ≠ 0 → prop_c a b c) ∧
  (∀ x y : ℝ, prop_d x y) :=
sorry

end necessary_condition_propositions_l3676_367629


namespace boxes_with_neither_l3676_367613

-- Define the set universe
def U : Set Nat := {n | n ≤ 15}

-- Define the set of boxes with crayons
def C : Set Nat := {n ∈ U | n ≤ 9}

-- Define the set of boxes with markers
def M : Set Nat := {n ∈ U | n ≤ 5}

-- Define the set of boxes with both crayons and markers
def B : Set Nat := {n ∈ U | n ≤ 4}

theorem boxes_with_neither (hU : Fintype U) (hC : Fintype C) (hM : Fintype M) (hB : Fintype B) :
  Fintype.card U - (Fintype.card C + Fintype.card M - Fintype.card B) = 5 := by
  sorry


end boxes_with_neither_l3676_367613


namespace distance_between_points_l3676_367648

/-- The distance between two points given round trip time and speed -/
theorem distance_between_points (speed : ℝ) (time : ℝ) (h1 : speed > 0) (h2 : time > 0) :
  let total_distance := speed * time
  let distance_between := total_distance / 2
  distance_between = 120 :=
by
  sorry

#check distance_between_points 60 4

end distance_between_points_l3676_367648


namespace common_tangents_count_l3676_367673

/-- Circle C₁ with equation x² + y² + 2x + 8y + 16 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 8*p.2 + 16 = 0}

/-- Circle C₂ with equation x² + y² - 4x - 4y - 1 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 - 1 = 0}

/-- The number of common tangents to circles C₁ and C₂ -/
def numCommonTangents : ℕ := 4

/-- Theorem stating that the number of common tangents to C₁ and C₂ is 4 -/
theorem common_tangents_count :
  numCommonTangents = 4 :=
sorry

end common_tangents_count_l3676_367673


namespace survey_participants_l3676_367683

theorem survey_participants (sample : ℕ) (percentage : ℚ) (total : ℕ) 
  (h1 : sample = 40)
  (h2 : percentage = 20 / 100)
  (h3 : sample = percentage * total) :
  total = 200 := by
sorry

end survey_participants_l3676_367683


namespace skylar_donation_l3676_367654

/-- Calculates the total donation amount given starting age, current age, and annual donation. -/
def total_donation (start_age : ℕ) (current_age : ℕ) (annual_donation : ℕ) : ℕ :=
  (current_age - start_age) * annual_donation

/-- Proves that Skylar's total donation is $432,000 given the specified conditions. -/
theorem skylar_donation :
  let start_age : ℕ := 17
  let current_age : ℕ := 71
  let annual_donation : ℕ := 8000
  total_donation start_age current_age annual_donation = 432000 := by
  sorry

end skylar_donation_l3676_367654


namespace product_of_numbers_l3676_367678

theorem product_of_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 := by
  sorry

end product_of_numbers_l3676_367678


namespace slices_per_banana_l3676_367624

/-- Given information about yogurt preparation and banana usage, 
    calculate the number of slices per banana. -/
theorem slices_per_banana 
  (slices_per_yogurt : ℕ) 
  (yogurts_to_make : ℕ) 
  (bananas_needed : ℕ) 
  (h1 : slices_per_yogurt = 8) 
  (h2 : yogurts_to_make = 5) 
  (h3 : bananas_needed = 4) : 
  (slices_per_yogurt * yogurts_to_make) / bananas_needed = 10 := by
sorry

end slices_per_banana_l3676_367624


namespace fraction_zero_implies_x_equals_one_l3676_367606

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  (x - 1) / (x + 1) = 0 → x = 1 := by
sorry

end fraction_zero_implies_x_equals_one_l3676_367606


namespace city_distance_l3676_367691

/-- The distance between Hallelujah City and San Pedro -/
def distance : ℝ := 1074

/-- The distance from San Pedro where the planes first meet -/
def first_meeting : ℝ := 437

/-- The distance from Hallelujah City where the planes meet on the return journey -/
def second_meeting : ℝ := 237

/-- The theorem stating the distance between the cities -/
theorem city_distance : 
  ∃ (v1 v2 : ℝ), v1 > v2 ∧ v1 > 0 ∧ v2 > 0 →
  first_meeting = v2 * (distance / (v1 + v2)) ∧
  second_meeting = v1 * (distance / (v1 + v2)) ∧
  distance = 1074 := by
sorry


end city_distance_l3676_367691


namespace arrangement_is_correct_l3676_367630

-- Define the metals and safes
inductive Metal
| Gold | Silver | Bronze | Platinum | Nickel

inductive Safe
| One | Two | Three | Four | Five

-- Define the arrangement as a function from Safe to Metal
def Arrangement := Safe → Metal

-- Define the statements on the safes
def statement1 (a : Arrangement) : Prop :=
  a Safe.Two = Metal.Gold ∨ a Safe.Three = Metal.Gold

def statement2 (a : Arrangement) : Prop :=
  a Safe.One = Metal.Silver

def statement3 (a : Arrangement) : Prop :=
  a Safe.Three ≠ Metal.Bronze

def statement4 (a : Arrangement) : Prop :=
  (a Safe.One = Metal.Nickel ∧ a Safe.Two = Metal.Gold) ∨
  (a Safe.Two = Metal.Nickel ∧ a Safe.Three = Metal.Gold) ∨
  (a Safe.Three = Metal.Nickel ∧ a Safe.Four = Metal.Gold) ∨
  (a Safe.Four = Metal.Nickel ∧ a Safe.Five = Metal.Gold)

def statement5 (a : Arrangement) : Prop :=
  (a Safe.One = Metal.Bronze ∧ a Safe.Two = Metal.Platinum) ∨
  (a Safe.Two = Metal.Bronze ∧ a Safe.Three = Metal.Platinum) ∨
  (a Safe.Three = Metal.Bronze ∧ a Safe.Four = Metal.Platinum) ∨
  (a Safe.Four = Metal.Bronze ∧ a Safe.Five = Metal.Platinum)

-- Define the correct arrangement
def correctArrangement : Arrangement :=
  fun s => match s with
  | Safe.One => Metal.Nickel
  | Safe.Two => Metal.Silver
  | Safe.Three => Metal.Bronze
  | Safe.Four => Metal.Platinum
  | Safe.Five => Metal.Gold

-- Theorem statement
theorem arrangement_is_correct (a : Arrangement) :
  (∃! s, a s = Metal.Gold ∧
    (s = Safe.One → statement1 a) ∧
    (s = Safe.Two → statement2 a) ∧
    (s = Safe.Three → statement3 a) ∧
    (s = Safe.Four → statement4 a) ∧
    (s = Safe.Five → statement5 a)) →
  (∀ s, a s = correctArrangement s) :=
sorry

end arrangement_is_correct_l3676_367630


namespace product_of_valid_bases_l3676_367679

theorem product_of_valid_bases : ∃ (S : Finset ℕ), 
  (∀ b ∈ S, b ≥ 2 ∧ 
    (∃ (P : Finset ℕ), (∀ p ∈ P, Nat.Prime p) ∧ 
      Finset.card P = b ∧
      (b^6 - 1) / (b - 1) = Finset.prod P id)) ∧
  Finset.prod S id = 12 := by
  sorry

end product_of_valid_bases_l3676_367679


namespace f_properties_l3676_367695

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x y : ℝ, x > -1 → y > -1 → x < y → f a x < f a y) ∧
  (∀ x : ℝ, x < 0 → f a x ≠ 0) := by
  sorry

end f_properties_l3676_367695


namespace subset_implies_m_values_l3676_367693

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

-- State the theorem
theorem subset_implies_m_values (m : ℝ) : B m ⊆ A → m = 1 ∨ m = 2 := by
  sorry

end subset_implies_m_values_l3676_367693


namespace min_value_a_l3676_367677

theorem min_value_a (m n : ℝ) (h1 : 0 < n) (h2 : n < m) (h3 : m < 1/a) 
  (h4 : (n^(1/m)) / (m^(1/n)) > (n^a) / (m^a)) : 
  ∀ ε > 0, ∃ a : ℝ, a ≥ 1 ∧ a < 1 + ε := by
  sorry

end min_value_a_l3676_367677


namespace problem_solution_l3676_367686

/-- A function satisfying the given property for all real numbers -/
def satisfies_property (g : ℝ → ℝ) : Prop :=
  ∀ a c : ℝ, c^3 * g a = a^3 * g c

theorem problem_solution (g : ℝ → ℝ) (h1 : satisfies_property g) (h2 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := by
  sorry

end problem_solution_l3676_367686


namespace system_of_inequalities_solution_l3676_367615

theorem system_of_inequalities_solution (x : ℝ) :
  (2 * x + 1 > x ∧ x < -3 * x + 8) ↔ (-1 < x ∧ x < 2) := by
  sorry

end system_of_inequalities_solution_l3676_367615


namespace cubic_identity_l3676_367631

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end cubic_identity_l3676_367631


namespace concentric_circles_radii_difference_l3676_367675

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : (R^2 / r^2) = 4) :
  R - r = r :=
sorry

end concentric_circles_radii_difference_l3676_367675


namespace circles_intersect_l3676_367687

/-- Definition of circle C1 -/
def C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 3*y + 2 = 0

/-- Definition of circle C2 -/
def C2 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 3*y + 1 = 0

/-- Theorem stating that C1 and C2 are intersecting -/
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y :=
sorry

end circles_intersect_l3676_367687


namespace quadratic_factorization_sum_l3676_367621

theorem quadratic_factorization_sum (a b c : ℤ) :
  (∀ x, x^2 + 14*x + 45 = (x + a)*(x + b)) →
  (∀ x, x^2 - 19*x + 90 = (x - b)*(x - c)) →
  a + b + c = 24 := by
sorry

end quadratic_factorization_sum_l3676_367621


namespace polar_bear_fish_consumption_l3676_367601

/-- Calculates the total number of fish buckets required for three polar bears for a week -/
theorem polar_bear_fish_consumption 
  (bear1_trout bear1_salmon : ℝ)
  (bear2_trout bear2_salmon : ℝ)
  (bear3_trout bear3_salmon : ℝ)
  (h1 : bear1_trout = 0.2)
  (h2 : bear1_salmon = 0.4)
  (h3 : bear2_trout = 0.3)
  (h4 : bear2_salmon = 0.5)
  (h5 : bear3_trout = 0.25)
  (h6 : bear3_salmon = 0.45)
  : (bear1_trout + bear1_salmon + bear2_trout + bear2_salmon + bear3_trout + bear3_salmon) * 7 = 14.7 := by
  sorry

#check polar_bear_fish_consumption

end polar_bear_fish_consumption_l3676_367601


namespace counterexample_exists_l3676_367650

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a * b > 0 ∧ 1/a ≥ 1/b := by
  sorry

end counterexample_exists_l3676_367650


namespace first_player_wins_l3676_367641

/-- Represents a position on the rectangular table -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the game state -/
structure GameState :=
  (table : Set Position)
  (occupied : Set Position)
  (currentPlayer : Nat)

/-- Defines a valid move in the game -/
def validMove (state : GameState) (pos : Position) : Prop :=
  pos ∈ state.table ∧ pos ∉ state.occupied

/-- Defines the winning condition for a player -/
def winningStrategy (player : Nat) : Prop :=
  ∀ (state : GameState), 
    state.currentPlayer = player → 
    ∃ (move : Position), validMove state move

/-- The main theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Position), 
    winningStrategy 1 ∧ 
    (∀ (state : GameState), 
      state.currentPlayer = 1 → 
      validMove state (strategy state)) :=
sorry

end first_player_wins_l3676_367641


namespace loan_duration_is_seven_years_l3676_367635

/-- Calculates the duration of a loan given the principal, interest rate, and interest paid. -/
def loanDuration (principal interestPaid interestRate : ℚ) : ℚ :=
  (interestPaid * 100) / (principal * interestRate)

/-- Theorem stating that for the given loan conditions, the duration is 7 years. -/
theorem loan_duration_is_seven_years 
  (principal : ℚ) 
  (interestPaid : ℚ) 
  (interestRate : ℚ) 
  (h1 : principal = 1500)
  (h2 : interestPaid = 735)
  (h3 : interestRate = 7) :
  loanDuration principal interestPaid interestRate = 7 := by
  sorry

#eval loanDuration 1500 735 7

end loan_duration_is_seven_years_l3676_367635


namespace coefficient_a_is_zero_l3676_367626

-- Define the quadratic equation
def quadratic_equation (a b c p : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c + p = 0

-- Define the condition that all roots are real and positive
def all_roots_real_positive (a b c : ℝ) : Prop :=
  ∀ p > 0, ∀ x, quadratic_equation a b c p x → x > 0

-- Theorem statement
theorem coefficient_a_is_zero (a b c : ℝ) :
  all_roots_real_positive a b c → a = 0 := by
  sorry

end coefficient_a_is_zero_l3676_367626


namespace intersection_of_A_and_B_l3676_367627

def A : Set ℝ := {-2, -1, 2, 3}
def B : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by
  sorry

end intersection_of_A_and_B_l3676_367627


namespace miranda_stuffs_six_pillows_l3676_367663

/-- The number of pillows Miranda can stuff given the conditions -/
def miranda_pillows : ℕ :=
  let feathers_per_pound : ℕ := 300
  let goose_feathers : ℕ := 3600
  let pounds_per_pillow : ℕ := 2
  let total_pounds : ℕ := goose_feathers / feathers_per_pound
  total_pounds / pounds_per_pillow

/-- Proof that Miranda can stuff 6 pillows -/
theorem miranda_stuffs_six_pillows : miranda_pillows = 6 := by
  sorry

end miranda_stuffs_six_pillows_l3676_367663


namespace arccos_cos_ten_l3676_367670

open Real

-- Define the problem statement
theorem arccos_cos_ten :
  let x := 10
  let y := arccos (cos x)
  0 ≤ y ∧ y ≤ π →
  y = x - 2 * π :=
by sorry

end arccos_cos_ten_l3676_367670


namespace total_gold_stars_l3676_367692

def gold_stars_yesterday : ℕ := 4
def gold_stars_today : ℕ := 3

theorem total_gold_stars : gold_stars_yesterday + gold_stars_today = 7 := by
  sorry

end total_gold_stars_l3676_367692


namespace third_month_sale_l3676_367605

def average_sale : ℕ := 5500
def number_of_months : ℕ := 6
def sales : List ℕ := [5435, 5927, 6230, 5562, 3991]

theorem third_month_sale :
  (average_sale * number_of_months - sales.sum) = 5855 := by
  sorry

end third_month_sale_l3676_367605


namespace x_range_proof_l3676_367682

def S (n : ℕ) : ℝ := sorry

def a : ℕ → ℝ := sorry

theorem x_range_proof :
  (∀ n : ℕ, n ≥ 2 → S (n - 1) + S n = 2 * n^2 + 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) > a n) →
  a 1 = x →
  2 < x ∧ x < 3 :=
by sorry

end x_range_proof_l3676_367682


namespace binary_to_quaternary_conversion_l3676_367636

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The binary representation of 1101101₂ -/
def binary_num : List Bool := [true, true, false, true, true, false, true]

/-- The expected quaternary representation -/
def expected_quaternary : List Nat := [3, 1, 2, 1]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_num) = expected_quaternary :=
by sorry

end binary_to_quaternary_conversion_l3676_367636


namespace perpendicular_line_equation_l3676_367611

/-- The equation of a line perpendicular to another line and passing through a given point. -/
theorem perpendicular_line_equation (m : ℚ) (b : ℚ) (x₀ : ℚ) (y₀ : ℚ) :
  let l₁ : ℚ → ℚ := λ x => m * x + b
  let m₂ : ℚ := -1 / m
  let l₂ : ℚ → ℚ := λ x => m₂ * (x - x₀) + y₀
  (∀ x, x - 2 * l₁ x + 3 = 0) →
  (∀ x, 2 * x + l₂ x - 1 = 0) := by
sorry

end perpendicular_line_equation_l3676_367611


namespace line_through_ellipse_midpoint_l3676_367637

/-- Given an ellipse and a line passing through its midpoint, prove the line's equation -/
theorem line_through_ellipse_midpoint (A B : ℝ × ℝ) :
  let M : ℝ × ℝ := (1, 1)
  let ellipse (p : ℝ × ℝ) := p.1^2 / 4 + p.2^2 / 3 = 1
  ellipse A ∧ ellipse B ∧  -- A and B are on the ellipse
  (∃ (k m : ℝ), ∀ (x y : ℝ), y = k * x + m ↔ ((x, y) = A ∨ (x, y) = B ∨ (x, y) = M)) ∧  -- A, B, and M are collinear
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  ∃ (k m : ℝ), k = 3 ∧ m = -7 ∧ ∀ (x y : ℝ), y = k * x + m ↔ ((x, y) = A ∨ (x, y) = B ∨ (x, y) = M) :=
by sorry


end line_through_ellipse_midpoint_l3676_367637


namespace factorization_equality_l3676_367667

theorem factorization_equality (a b x y : ℝ) :
  (a*x - b*y)^2 + (a*y + b*x)^2 = (x^2 + y^2) * (a^2 + b^2) := by
  sorry

end factorization_equality_l3676_367667


namespace new_students_l3676_367643

theorem new_students (initial : ℕ) (left : ℕ) (final : ℕ) : 
  initial = 33 → left = 18 → final = 29 → final - (initial - left) = 14 := by
  sorry

end new_students_l3676_367643


namespace pavan_travel_distance_l3676_367633

theorem pavan_travel_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h_total_time : total_time = 11)
  (h_speed1 : speed1 = 30)
  (h_speed2 : speed2 = 25)
  (h_half_distance : ∀ d : ℝ, d / 2 / speed1 + d / 2 / speed2 = total_time) :
  ∃ d : ℝ, d = 150 ∧ d / 2 / speed1 + d / 2 / speed2 = total_time :=
by sorry

end pavan_travel_distance_l3676_367633


namespace jasmine_solution_problem_l3676_367672

theorem jasmine_solution_problem (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_jasmine : ℝ) (final_concentration : ℝ) (x : ℝ) : 
  initial_volume = 90 →
  initial_concentration = 0.05 →
  added_jasmine = 8 →
  final_concentration = 0.125 →
  initial_volume * initial_concentration + added_jasmine = 
    (initial_volume + added_jasmine + x) * final_concentration →
  x = 2 := by
sorry

end jasmine_solution_problem_l3676_367672


namespace second_term_is_seven_l3676_367608

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ
  num_terms : ℕ
  sum_first_eight_eq_last_four : ℝ → Prop

/-- The theorem statement -/
theorem second_term_is_seven
  (seq : ArithmeticSequence)
  (h1 : seq.num_terms = 12)
  (h2 : seq.common_difference = 2)
  (h3 : seq.sum_first_eight_eq_last_four seq.first_term) :
  seq.first_term + seq.common_difference = 7 := by
  sorry

end second_term_is_seven_l3676_367608


namespace min_value_trig_expression_l3676_367674

theorem min_value_trig_expression (α β : Real) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 215 := by
  sorry

end min_value_trig_expression_l3676_367674


namespace total_oranges_in_box_l3676_367607

def initial_oranges : ℝ := 55.0
def added_oranges : ℝ := 35.0

theorem total_oranges_in_box : initial_oranges + added_oranges = 90.0 := by
  sorry

end total_oranges_in_box_l3676_367607


namespace least_positive_integer_to_multiple_of_four_l3676_367644

theorem least_positive_integer_to_multiple_of_four (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬((530 + m) % 4 = 0)) ∧ ((530 + n) % 4 = 0) → n = 2 := by
  sorry

end least_positive_integer_to_multiple_of_four_l3676_367644


namespace solution_pairs_count_l3676_367623

theorem solution_pairs_count : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    4 * p.1 + 7 * p.2 = 548 ∧ p.1 > 0 ∧ p.2 > 0) 
    (Finset.product (Finset.range 548) (Finset.range 548))).card ∧ n = 19 := by
  sorry

end solution_pairs_count_l3676_367623


namespace complement_A_union_B_l3676_367681

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

-- State the theorem
theorem complement_A_union_B : 
  (Set.univ \ A) ∪ B = {x : ℝ | x ≥ -2} := by sorry

end complement_A_union_B_l3676_367681


namespace sum_of_solutions_squared_equation_l3676_367662

theorem sum_of_solutions_squared_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ + 6)^2 = 49 ∧ (x₂ + 6)^2 = 49 ∧ x₁ + x₂ = -12 :=
by sorry

end sum_of_solutions_squared_equation_l3676_367662


namespace translation_problem_l3676_367645

def translation (z w : ℂ) : ℂ := z + w

theorem translation_problem (t : ℂ → ℂ) :
  (∃ w : ℂ, ∀ z, t z = translation z w) →
  t (1 + 3*I) = 5 + 7*I →
  t (2 - 2*I) = 6 + 2*I :=
by sorry

end translation_problem_l3676_367645


namespace complex_sum_problem_l3676_367694

theorem complex_sum_problem (x y z w u v : ℝ) : 
  y = 2 ∧ 
  x = -z - u ∧ 
  (x + z + u) + (y + w + v) * I = 3 - 4 * I → 
  w + v = -6 := by sorry

end complex_sum_problem_l3676_367694


namespace seven_balls_three_boxes_l3676_367664

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 8 ways to distribute 7 indistinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 8 := by
  sorry

end seven_balls_three_boxes_l3676_367664


namespace probability_at_least_one_correct_l3676_367618

theorem probability_at_least_one_correct (total_questions : Nat) (options_per_question : Nat) (guessed_questions : Nat) : 
  total_questions = 30 → 
  options_per_question = 6 → 
  guessed_questions = 5 → 
  (1 - (options_per_question - 1 : ℚ) / options_per_question ^ guessed_questions) = 4651 / 7776 := by
sorry

end probability_at_least_one_correct_l3676_367618


namespace greatest_common_divisor_under_60_l3676_367652

theorem greatest_common_divisor_under_60 : ∃ (d : ℕ), d = 36 ∧ 
  d ∣ 468 ∧ d ∣ 108 ∧ d < 60 ∧ 
  ∀ (x : ℕ), x ∣ 468 ∧ x ∣ 108 ∧ x < 60 → x ≤ d :=
by sorry

end greatest_common_divisor_under_60_l3676_367652


namespace existence_of_equal_point_l3676_367603

theorem existence_of_equal_point
  (f g : ℝ → ℝ)
  (hf : Continuous f)
  (hg : Continuous g)
  (hg_diff : Differentiable ℝ g)
  (h_condition : (f 0 - deriv g 0) * (deriv g 1 - f 1) > 0) :
  ∃ c ∈ (Set.Ioo 0 1), f c = deriv g c :=
sorry

end existence_of_equal_point_l3676_367603


namespace range_of_a_valid_a_set_is_closed_l3676_367676

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the set of valid a values
def valid_a_set : Set ℝ := {a | a ≤ -2 ∨ a = 1}

-- Theorem statement
theorem range_of_a (a : ℝ) (h1 : a ≥ 0) (h2 : p a ∧ q a) : a ∈ valid_a_set := by
  sorry

-- Additional helper theorem to show the set is closed
theorem valid_a_set_is_closed : IsClosed valid_a_set := by
  sorry

end range_of_a_valid_a_set_is_closed_l3676_367676


namespace quadratic_roots_property_l3676_367651

theorem quadratic_roots_property (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ x₁ - x₂ = 5 ∧ x₁^3 - x₂^3 = 35) →
  ((p = 1 ∧ q = -6) ∨ (p = -1 ∧ q = -6)) := by
sorry

end quadratic_roots_property_l3676_367651


namespace hyperbola_and_line_properties_l3676_367640

/-- Hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_focus : 2 = Real.sqrt (a^2 + b^2)
  h_eccentricity : 2 = Real.sqrt (a^2 + b^2) / a

/-- Line intersecting the hyperbola -/
structure IntersectingLine where
  k : ℝ
  m : ℝ
  h_slope : k = 1
  h_distinct : ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + m ∧ y₂ = k * x₂ + m ∧
    x₁^2 - y₁^2/3 = 1 ∧ x₂^2 - y₂^2/3 = 1
  h_area : ∃ (x₀ y₀ : ℝ), 
    x₀ = (k * m) / (3 - k^2) ∧
    y₀ = (3 * m) / (3 - k^2) ∧
    1/2 * |4 * k * m / (3 - k^2)| * |4 * m / (3 - k^2)| = 4

/-- Main theorem -/
theorem hyperbola_and_line_properties (C : Hyperbola) (l : IntersectingLine) :
  (C.a = 1 ∧ C.b = Real.sqrt 3) ∧ 
  (l.m = Real.sqrt 2 ∨ l.m = -Real.sqrt 2) := by
  sorry

end hyperbola_and_line_properties_l3676_367640


namespace jackson_spending_money_l3676_367660

/-- The amount of money earned per hour of chores -/
def money_per_hour : ℝ := 5

/-- The time spent vacuuming (in hours) -/
def vacuuming_time : ℝ := 2 * 2

/-- The time spent washing dishes (in hours) -/
def dish_washing_time : ℝ := 0.5

/-- The time spent cleaning the bathroom (in hours) -/
def bathroom_cleaning_time : ℝ := 3 * dish_washing_time

/-- The total time spent on chores (in hours) -/
def total_chore_time : ℝ := vacuuming_time + dish_washing_time + bathroom_cleaning_time

/-- The theorem stating that Jackson's earned spending money is $30 -/
theorem jackson_spending_money : money_per_hour * total_chore_time = 30 := by
  sorry

end jackson_spending_money_l3676_367660


namespace dessert_division_l3676_367655

/-- Represents the number of dessert items -/
structure DessertItems where
  cinnamon_swirls : ℕ
  brownie_bites : ℕ
  fruit_tartlets : ℕ

/-- Represents the number of people sharing the desserts -/
def num_people : ℕ := 8

/-- The actual dessert items from the problem -/
def desserts : DessertItems := {
  cinnamon_swirls := 15,
  brownie_bites := 24,
  fruit_tartlets := 18
}

/-- Theorem stating that brownie bites can be equally divided, while others cannot -/
theorem dessert_division (d : DessertItems) (p : ℕ) (h_p : p = num_people) :
  d.brownie_bites / p = 3 ∧
  ¬(∃ (n : ℕ), n * p = d.cinnamon_swirls) ∧
  ¬(∃ (m : ℕ), m * p = d.fruit_tartlets) :=
sorry

end dessert_division_l3676_367655


namespace equation_solution_l3676_367616

theorem equation_solution : ∃ x : ℝ, (Real.sqrt (x + 42) + Real.sqrt (x + 10) = 16) ∧ (x = 39) := by
  sorry

end equation_solution_l3676_367616


namespace line_through_midpoint_parallel_to_PR_l3676_367665

/-- Given points P, Q, R in a 2D plane, prove that if a line y = mx + b is parallel to PR
    and passes through the midpoint of QR, then b = -4. -/
theorem line_through_midpoint_parallel_to_PR (P Q R : ℝ × ℝ) (m b : ℝ) : 
  P = (0, 0) →
  Q = (4, 0) →
  R = (1, 2) →
  (∀ x y : ℝ, y = m * x + b ↔ (∃ t : ℝ, (x, y) = ((1 - t) * P.1 + t * R.1, (1 - t) * P.2 + t * R.2))) →
  (m * ((Q.1 + R.1) / 2) + b = (Q.2 + R.2) / 2) →
  b = -4 := by
  sorry


end line_through_midpoint_parallel_to_PR_l3676_367665


namespace union_covers_reals_implies_a_leq_neg_one_l3676_367658

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < 4}

-- State the theorem
theorem union_covers_reals_implies_a_leq_neg_one (a : ℝ) :
  A ∪ B a = Set.univ → a ≤ -1 := by
  sorry

end union_covers_reals_implies_a_leq_neg_one_l3676_367658
