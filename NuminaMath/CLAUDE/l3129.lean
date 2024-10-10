import Mathlib

namespace smallest_share_is_five_thirds_l3129_312939

/-- Represents the shares of bread in an arithmetic sequence -/
structure BreadShares where
  a : ℚ  -- The middle term of the arithmetic sequence
  d : ℚ  -- The common difference of the arithmetic sequence
  sum_equals_100 : 5 * a = 100
  larger_three_seventh_smaller_two : 3 * a + 3 * d = 7 * (2 * a - 3 * d)
  d_positive : d > 0

/-- The smallest share of bread -/
def smallest_share (shares : BreadShares) : ℚ :=
  shares.a - 2 * shares.d

/-- Theorem stating that the smallest share is 5/3 -/
theorem smallest_share_is_five_thirds (shares : BreadShares) :
  smallest_share shares = 5 / 3 := by
  sorry

end smallest_share_is_five_thirds_l3129_312939


namespace notebook_purchase_savings_l3129_312941

theorem notebook_purchase_savings (s : ℚ) (n : ℚ) (p : ℚ) 
  (h1 : s > 0) (h2 : n > 0) (h3 : p > 0) 
  (h4 : (1/4) * s = (1/2) * n * p) : 
  s - n * p = (1/2) * s := by
sorry

end notebook_purchase_savings_l3129_312941


namespace solve_equation_l3129_312938

theorem solve_equation : ∃ m : ℤ, 2^4 - 3 = 3^3 + m ∧ m = -14 := by
  sorry

end solve_equation_l3129_312938


namespace ceiling_floor_sum_l3129_312957

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end ceiling_floor_sum_l3129_312957


namespace largest_four_digit_perfect_cube_largest_four_digit_perfect_cube_is_9261_l3129_312981

theorem largest_four_digit_perfect_cube : ℕ → Prop :=
  fun n => (1000 ≤ n ∧ n ≤ 9999) ∧  -- n is a four-digit number
            (∃ m : ℕ, n = m^3) ∧    -- n is a perfect cube
            (∀ k : ℕ, (1000 ≤ k ∧ k ≤ 9999 ∧ ∃ m : ℕ, k = m^3) → k ≤ n)  -- n is the largest such number

theorem largest_four_digit_perfect_cube_is_9261 :
  largest_four_digit_perfect_cube 9261 := by
  sorry

end largest_four_digit_perfect_cube_largest_four_digit_perfect_cube_is_9261_l3129_312981


namespace inscribed_semicircle_radius_l3129_312924

/-- An isosceles triangle with a semicircle inscribed in its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The diameter of the semicircle is equal to the base of the triangle -/
  diameter_eq_base : 2 * radius = base

/-- The radius of the inscribed semicircle in the given isosceles triangle -/
theorem inscribed_semicircle_radius 
    (t : IsoscelesTriangleWithSemicircle) 
    (h1 : t.base = 20) 
    (h2 : t.height = 21) : 
    t.radius = 210 / Real.sqrt 541 := by
  sorry


end inscribed_semicircle_radius_l3129_312924


namespace tenth_stays_tenth_probability_sum_of_numerator_and_denominator_l3129_312985

/-- Represents a sequence of distinct numbers -/
def DistinctSequence (n : ℕ) := { s : Fin n → ℕ // Function.Injective s }

/-- The probability of the 10th element staying in the 10th position after one bubble pass -/
def probabilityTenthStaysTenth (n : ℕ) : ℚ :=
  if n < 12 then 0 else 1 / (12 * 11)

theorem tenth_stays_tenth_probability :
  probabilityTenthStaysTenth 20 = 1 / 132 := by sorry

#eval Nat.gcd 1 132  -- Should output 1, confirming 1/132 is in lowest terms

theorem sum_of_numerator_and_denominator :
  let p := 1
  let q := 132
  p + q = 133 := by sorry

end tenth_stays_tenth_probability_sum_of_numerator_and_denominator_l3129_312985


namespace sunzi_wood_measurement_l3129_312949

theorem sunzi_wood_measurement 
  (x y : ℝ) 
  (h1 : y - x = 4.5) 
  (h2 : (1/2) * y < x) 
  (h3 : x < (1/2) * y + 1) : 
  y - x = 4.5 ∧ (1/2) * y = x - 1 := by
  sorry

end sunzi_wood_measurement_l3129_312949


namespace intersection_point_sum_l3129_312937

/-- Given two lines that intersect at (3, 5), prove that a + b = 86/15 -/
theorem intersection_point_sum (a b : ℚ) : 
  (3 = (1/3) * 5 + a) → 
  (5 = (1/5) * 3 + b) → 
  a + b = 86/15 := by
  sorry

end intersection_point_sum_l3129_312937


namespace sin_15_and_tan_75_l3129_312993

theorem sin_15_and_tan_75 :
  (Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4) ∧
  (Real.tan (75 * π / 180) = 2 + Real.sqrt 3) := by
  sorry

end sin_15_and_tan_75_l3129_312993


namespace seating_arrangements_with_restriction_l3129_312984

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n people in a row where 2 specific people must sit together -/
def restrictedArrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

/-- The number of ways to arrange n people in a row where 2 specific people cannot sit together -/
def acceptableArrangements (n : ℕ) : ℕ := totalArrangements n - restrictedArrangements n

theorem seating_arrangements_with_restriction (n : ℕ) (hn : n = 9) :
  acceptableArrangements n = 282240 := by
  sorry

#eval acceptableArrangements 9

end seating_arrangements_with_restriction_l3129_312984


namespace arithmetic_sequence_average_l3129_312970

/-- The average value of an arithmetic sequence with 5 terms, starting at 0 and with a common difference of 3x, is 6x. -/
theorem arithmetic_sequence_average (x : ℝ) : 
  let sequence := [0, 3*x, 6*x, 9*x, 12*x]
  (sequence.sum / sequence.length : ℝ) = 6*x := by
sorry

end arithmetic_sequence_average_l3129_312970


namespace adjustment_schemes_no_adjacent_boys_arrangements_specific_position_arrangements_l3129_312995

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 4
def total_people : ℕ := num_boys + num_girls

-- Statement A
theorem adjustment_schemes :
  (Nat.choose total_people 3) * 2 = 70 := by sorry

-- Statement B
theorem no_adjacent_boys_arrangements :
  (Nat.factorial num_girls) * (Nat.factorial (num_girls + 1) / Nat.factorial (num_girls + 1 - num_boys)) = 1440 := by sorry

-- Statement D
theorem specific_position_arrangements :
  Nat.factorial total_people - 2 * Nat.factorial (total_people - 1) + Nat.factorial (total_people - 2) = 3720 := by sorry

end adjustment_schemes_no_adjacent_boys_arrangements_specific_position_arrangements_l3129_312995


namespace joseph_drives_one_mile_more_than_kyle_l3129_312986

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Joseph's driving speed in mph -/
def joseph_speed : ℝ := 50

/-- Joseph's driving time in hours -/
def joseph_time : ℝ := 2.5

/-- Kyle's driving speed in mph -/
def kyle_speed : ℝ := 62

/-- Kyle's driving time in hours -/
def kyle_time : ℝ := 2

theorem joseph_drives_one_mile_more_than_kyle :
  distance joseph_speed joseph_time - distance kyle_speed kyle_time = 1 := by
  sorry

end joseph_drives_one_mile_more_than_kyle_l3129_312986


namespace y_in_terms_of_x_l3129_312953

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end y_in_terms_of_x_l3129_312953


namespace total_cost_for_six_people_l3129_312976

/-- The total cost of buying soda and pizza for a group -/
def total_cost (num_people : ℕ) (soda_price pizza_price : ℚ) : ℚ :=
  num_people * (soda_price + pizza_price)

/-- Theorem: The total cost for 6 people is $9.00 -/
theorem total_cost_for_six_people :
  total_cost 6 (1/2) 1 = 9 :=
by sorry

end total_cost_for_six_people_l3129_312976


namespace bills_age_l3129_312950

theorem bills_age (caroline_age : ℝ) 
  (h1 : caroline_age + (2 * caroline_age - 1) + (caroline_age - 4) = 45) : 
  2 * caroline_age - 1 = 24 := by
  sorry

#check bills_age

end bills_age_l3129_312950


namespace fare_comparison_l3129_312987

/-- Fare calculation for city A -/
def fareA (x : ℝ) : ℝ := 10 + 2 * (x - 3)

/-- Fare calculation for city B -/
def fareB (x : ℝ) : ℝ := 8 + 2.5 * (x - 3)

/-- Theorem stating the condition for city A's fare to be higher than city B's -/
theorem fare_comparison (x : ℝ) :
  x > 3 → (fareA x > fareB x ↔ 3 < x ∧ x < 7) := by sorry

end fare_comparison_l3129_312987


namespace geometric_sum_first_six_terms_l3129_312956

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six_terms :
  geometric_sum (1/4) (1/4) 6 = 4095/12288 := by
  sorry

end geometric_sum_first_six_terms_l3129_312956


namespace exam_average_marks_l3129_312999

theorem exam_average_marks (num_papers : ℕ) (geography_increase : ℕ) (history_increase : ℕ) (new_average : ℕ) :
  num_papers = 11 →
  geography_increase = 20 →
  history_increase = 2 →
  new_average = 65 →
  (num_papers * new_average - geography_increase - history_increase) / num_papers = 63 :=
by sorry

end exam_average_marks_l3129_312999


namespace no_rectangular_prism_exists_l3129_312972

theorem no_rectangular_prism_exists : ¬∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 12 ∧ 
  a * b + b * c + c * a = 1 ∧ 
  a * b * c = 12 := by
  sorry

end no_rectangular_prism_exists_l3129_312972


namespace negation_of_universal_statement_l3129_312997

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by
  sorry

end negation_of_universal_statement_l3129_312997


namespace unit_circle_dot_product_l3129_312906

theorem unit_circle_dot_product 
  (x₁ y₁ x₂ y₂ θ : ℝ) 
  (h₁ : x₁^2 + y₁^2 = 1) 
  (h₂ : x₂^2 + y₂^2 = 1)
  (h₃ : π/2 < θ ∧ θ < π)
  (h₄ : Real.sin (θ + π/4) = 3/5) : 
  x₁ * x₂ + y₁ * y₂ = -Real.sqrt 2 / 10 := by
sorry

end unit_circle_dot_product_l3129_312906


namespace enterprise_tax_comparison_l3129_312980

theorem enterprise_tax_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) : 
  let x := (b - a) / 2
  let y := Real.sqrt (b / a) - 1
  b * (1 + y) > b + x := by sorry

end enterprise_tax_comparison_l3129_312980


namespace two_digit_subtraction_l3129_312942

/-- Given two different natural numbers A and B that satisfy the two-digit subtraction equation 6A - B2 = 36, prove that A - B = 5 -/
theorem two_digit_subtraction (A B : ℕ) (h1 : A ≠ B) (h2 : 10 ≤ A) (h3 : A < 100) (h4 : 10 ≤ B) (h5 : B < 100) (h6 : 60 + A - (10 * B + 2) = 36) : A - B = 5 := by
  sorry

end two_digit_subtraction_l3129_312942


namespace group_size_proof_l3129_312926

theorem group_size_proof (n : ℕ) (k : ℕ) : 
  k = 7 → 
  (n : ℚ) - k ≠ 0 → 
  ((n - k) / n - k / n : ℚ) = 0.30000000000000004 → 
  n = 20 := by
sorry

end group_size_proof_l3129_312926


namespace inscribed_circles_radius_l3129_312933

/-- Given a circle segment with radius R and central angle α, 
    this theorem proves the radius of two equal inscribed circles 
    that touch each other, the arc, and the chord. -/
theorem inscribed_circles_radius 
  (R : ℝ) 
  (α : ℝ) 
  (h_α_pos : 0 < α) 
  (h_α_lt_pi : α < π) : 
  ∃ x : ℝ, 
    x = R * Real.sin (α / 4) ^ 2 ∧ 
    x > 0 ∧
    (∀ y : ℝ, y = R * Real.sin (α / 4) ^ 2 → y = x) :=
sorry

end inscribed_circles_radius_l3129_312933


namespace factorization_equality_l3129_312909

theorem factorization_equality (x : ℝ) : 
  (x - 3) * (x - 1) * (x - 2) * (x + 4) + 24 = (x - 2) * (x + 3) * (x^2 + x - 8) := by
  sorry

end factorization_equality_l3129_312909


namespace triangle_stack_impossibility_l3129_312996

theorem triangle_stack_impossibility : ¬ ∃ (n : ℕ), 6 * n = 165 := by
  sorry

end triangle_stack_impossibility_l3129_312996


namespace firetruck_reachable_area_l3129_312988

/-- Represents the speed of the firetruck in different terrains --/
structure FiretruckSpeed where
  road : ℝ
  field : ℝ

/-- Calculates the area reachable by a firetruck given its speed and time --/
def reachable_area (speed : FiretruckSpeed) (time : ℝ) : ℝ :=
  sorry

/-- Theorem stating the area reachable by the firetruck in 15 minutes --/
theorem firetruck_reachable_area :
  let speed := FiretruckSpeed.mk 60 18
  let time := 15 / 60  -- 15 minutes in hours
  reachable_area speed time = 1194.75 := by
  sorry

end firetruck_reachable_area_l3129_312988


namespace probability_even_or_greater_than_4_l3129_312905

/-- A fair six-sided die. -/
structure Die :=
  (faces : Finset ℕ)
  (fair : faces.card = 6)
  (labeled : faces = {1, 2, 3, 4, 5, 6})

/-- The event "the number facing up is even or greater than 4". -/
def EventEvenOrGreaterThan4 (d : Die) : Finset ℕ :=
  d.faces.filter (λ x => x % 2 = 0 ∨ x > 4)

/-- The probability of an event for a fair die. -/
def Probability (d : Die) (event : Finset ℕ) : ℚ :=
  event.card / d.faces.card

theorem probability_even_or_greater_than_4 (d : Die) :
  Probability d (EventEvenOrGreaterThan4 d) = 2/3 := by
  sorry

end probability_even_or_greater_than_4_l3129_312905


namespace students_without_favorite_l3129_312975

theorem students_without_favorite (total : ℕ) (math_frac english_frac history_frac science_frac : ℚ) : 
  total = 120 →
  math_frac = 3 / 10 →
  english_frac = 5 / 12 →
  history_frac = 1 / 8 →
  science_frac = 3 / 20 →
  total - (↑total * math_frac).floor - (↑total * english_frac).floor - 
  (↑total * history_frac).floor - (↑total * science_frac).floor = 1 := by
  sorry

end students_without_favorite_l3129_312975


namespace base_five_last_digit_l3129_312992

theorem base_five_last_digit (n : ℕ) (h : n = 89) : n % 5 = 4 := by
  sorry

end base_five_last_digit_l3129_312992


namespace comparison_of_expressions_l3129_312961

theorem comparison_of_expressions : 2 + Real.log 6 / Real.log 2 > 2 * Real.sqrt 5 := by
  sorry

end comparison_of_expressions_l3129_312961


namespace carnival_ticket_cost_l3129_312977

/-- The cost of carnival tickets -/
theorem carnival_ticket_cost :
  ∀ (cost_12 : ℚ) (cost_4 : ℚ),
  cost_12 = 3 →
  12 * cost_4 = cost_12 →
  cost_4 = 1 := by
  sorry

end carnival_ticket_cost_l3129_312977


namespace train_length_calculation_l3129_312952

theorem train_length_calculation (speed_kmh : ℝ) (crossing_time : ℝ) : 
  speed_kmh = 18 → crossing_time = 20 → speed_kmh * (1000 / 3600) * crossing_time = 100 := by
  sorry

end train_length_calculation_l3129_312952


namespace rotation_of_P_l3129_312944

/-- Rotate a point 180 degrees counterclockwise about the origin -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotation_of_P :
  let P : ℝ × ℝ := (-3, 2)
  rotate180 P = (3, -2) := by
sorry

end rotation_of_P_l3129_312944


namespace lcm_24_90_128_l3129_312963

theorem lcm_24_90_128 : Nat.lcm (Nat.lcm 24 90) 128 = 2880 := by
  sorry

end lcm_24_90_128_l3129_312963


namespace system_solution_ratio_l3129_312982

theorem system_solution_ratio (x y a b : ℝ) (h1 : 2 * x - y = a) (h2 : 3 * y - 6 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 3 := by
  sorry

end system_solution_ratio_l3129_312982


namespace constant_term_expansion_l3129_312966

def binomial_coeff (n k : ℕ) : ℕ := sorry

def general_term (r : ℕ) : ℤ :=
  (binomial_coeff 5 r : ℤ) * (-1)^r

theorem constant_term_expansion :
  (general_term 1) + (general_term 3) + (general_term 5) = -51 := by
  sorry

end constant_term_expansion_l3129_312966


namespace equation_equivalence_l3129_312998

theorem equation_equivalence (y : ℝ) :
  (8 * y^2 + 90 * y + 5) / (3 * y^2 + 4 * y + 49) = 4 * y + 1 →
  12 * y^3 + 11 * y^2 + 110 * y + 44 = 0 := by
  sorry

end equation_equivalence_l3129_312998


namespace tiles_in_row_l3129_312931

theorem tiles_in_row (area : ℝ) (length : ℝ) (tile_size : ℝ) : 
  area = 320 → length = 16 → tile_size = 1 → 
  (area / length) / tile_size = 20 := by sorry

end tiles_in_row_l3129_312931


namespace officer_selection_theorem_l3129_312923

/-- Represents a club with members of two genders -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ

/-- Calculates the number of ways to choose officers from a single gender -/
def waysToChooseOfficers (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- Calculates the total number of ways to choose officers in the club -/
def totalWaysToChooseOfficers (club : Club) : ℕ :=
  waysToChooseOfficers club.boys + waysToChooseOfficers club.girls

/-- The main theorem stating the number of ways to choose officers -/
theorem officer_selection_theorem (club : Club) 
    (h1 : club.total_members = 30)
    (h2 : club.boys = 18)
    (h3 : club.girls = 12) :
    totalWaysToChooseOfficers club = 6216 := by
  sorry


end officer_selection_theorem_l3129_312923


namespace great_eight_teams_l3129_312913

/-- The number of teams in the GREAT EIGHT conference -/
def num_teams : ℕ := 9

/-- The total number of games played in the conference -/
def total_games : ℕ := 36

/-- The number of games played by one team -/
def games_per_team : ℕ := 8

/-- Calculates the number of games in a round-robin tournament -/
def round_robin_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem great_eight_teams :
  (round_robin_games num_teams = total_games) ∧
  (num_teams - 1 = games_per_team) := by
  sorry

end great_eight_teams_l3129_312913


namespace framed_painting_ratio_l3129_312940

theorem framed_painting_ratio : 
  let painting_width : ℝ := 28
  let painting_height : ℝ := 32
  let frame_side_width : ℝ := 10/3
  let frame_top_bottom_width : ℝ := 3 * frame_side_width
  let framed_width : ℝ := painting_width + 2 * frame_side_width
  let framed_height : ℝ := painting_height + 2 * frame_top_bottom_width
  let frame_area : ℝ := framed_width * framed_height - painting_width * painting_height
  frame_area = painting_width * painting_height →
  framed_width / framed_height = 26 / 35 :=
by
  sorry

end framed_painting_ratio_l3129_312940


namespace max_prob_with_highest_prob_second_l3129_312921

/-- Represents a chess player with a given win probability -/
structure Player where
  winProb : ℝ

/-- Represents the chess player's opponents -/
structure Opponents where
  A : Player
  B : Player
  C : Player

/-- Calculates the probability of winning two consecutive games given the order of opponents -/
def probTwoConsecutiveWins (opponents : Opponents) (first second : Player) : ℝ :=
  2 * (first.winProb * second.winProb - 2 * opponents.A.winProb * opponents.B.winProb * opponents.C.winProb)

/-- Theorem stating that playing against the opponent with the highest win probability in the second game maximizes the probability of winning two consecutive games -/
theorem max_prob_with_highest_prob_second (opponents : Opponents)
    (h1 : opponents.C.winProb > opponents.B.winProb)
    (h2 : opponents.B.winProb > opponents.A.winProb)
    (h3 : opponents.A.winProb > 0) :
    ∀ (first : Player),
      probTwoConsecutiveWins opponents first opponents.C ≥ probTwoConsecutiveWins opponents first opponents.B ∧
      probTwoConsecutiveWins opponents first opponents.C ≥ probTwoConsecutiveWins opponents first opponents.A :=
  sorry


end max_prob_with_highest_prob_second_l3129_312921


namespace nicki_total_distance_l3129_312945

/-- Represents Nicki's exercise regime for a year --/
structure ExerciseRegime where
  running_miles_first_3_months : ℕ
  running_miles_next_3_months : ℕ
  running_miles_last_6_months : ℕ
  swimming_miles_first_6_months : ℕ
  hiking_miles_per_rest_week : ℕ
  weeks_in_year : ℕ
  weeks_per_month : ℕ

/-- Calculates the total distance covered in all exercises during the year --/
def totalDistance (regime : ExerciseRegime) : ℕ :=
  let running_weeks_per_month := regime.weeks_per_month - 1
  let running_miles := 
    (running_weeks_per_month * 3 * regime.running_miles_first_3_months) +
    (running_weeks_per_month * 3 * regime.running_miles_next_3_months) +
    (running_weeks_per_month * 6 * regime.running_miles_last_6_months)
  let swimming_miles := running_weeks_per_month * 6 * regime.swimming_miles_first_6_months
  let rest_weeks := regime.weeks_in_year / 4
  let hiking_miles := rest_weeks * regime.hiking_miles_per_rest_week
  running_miles + swimming_miles + hiking_miles

/-- Theorem stating that Nicki's total distance is 1095 miles --/
theorem nicki_total_distance :
  ∃ (regime : ExerciseRegime),
    regime.running_miles_first_3_months = 10 ∧
    regime.running_miles_next_3_months = 20 ∧
    regime.running_miles_last_6_months = 30 ∧
    regime.swimming_miles_first_6_months = 5 ∧
    regime.hiking_miles_per_rest_week = 15 ∧
    regime.weeks_in_year = 52 ∧
    regime.weeks_per_month = 4 ∧
    totalDistance regime = 1095 := by
  sorry

end nicki_total_distance_l3129_312945


namespace expression_simplification_l3129_312962

theorem expression_simplification :
  let a : ℝ := Real.sqrt 3
  let b : ℝ := Real.sqrt 2
  let c : ℝ := Real.sqrt 5
  6 * 37 * (a + b) ^ (2 * (Real.log c / Real.log (a - b))) = 1110 :=
by sorry

end expression_simplification_l3129_312962


namespace rowing_current_rate_l3129_312907

/-- Proves that the rate of the current is 1.4 km/hr given the conditions of the rowing problem -/
theorem rowing_current_rate (rowing_speed : ℝ) (upstream_time downstream_time : ℝ) : 
  rowing_speed = 4.2 →
  upstream_time = 2 * downstream_time →
  let current_rate := (rowing_speed / 3 : ℝ)
  current_rate = 1.4 := by sorry

end rowing_current_rate_l3129_312907


namespace sqrt_product_equation_l3129_312954

theorem sqrt_product_equation (x : ℝ) (hx : x > 0) :
  Real.sqrt (16 * x) * Real.sqrt (25 * x) * Real.sqrt (5 * x) * Real.sqrt (20 * x) = 40 →
  x = 1 / Real.sqrt 5 := by
sorry

end sqrt_product_equation_l3129_312954


namespace quadratic_equation_coefficients_l3129_312965

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, a * x^2 + b * x + c = 0 ↔ 3 * x^2 - 4 * x + 1 = 0) →
  a = 3 ∧ b = -4 ∧ c = 1 := by
sorry

end quadratic_equation_coefficients_l3129_312965


namespace alcohol_in_mixture_l3129_312927

/-- Proves that the amount of alcohol in a mixture is 7.5 liters given specific conditions -/
theorem alcohol_in_mixture :
  ∀ (A W : ℝ), 
    (A / W = 4 / 3) →  -- Initial ratio of alcohol to water
    (A / (W + 5) = 4 / 5) →  -- Ratio after adding 5 liters of water
    A = 7.5 := by
  sorry

end alcohol_in_mixture_l3129_312927


namespace z_in_fourth_quadrant_l3129_312959

/-- A complex number is in the fourth quadrant if its real part is positive and its imaginary part is negative. -/
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

/-- Given complex number z -/
def z : ℂ := 1 - 2 * Complex.I

/-- Theorem: The complex number z = 1 - 2i is in the fourth quadrant -/
theorem z_in_fourth_quadrant : in_fourth_quadrant z := by
  sorry

end z_in_fourth_quadrant_l3129_312959


namespace max_cars_in_parking_lot_l3129_312971

/-- Represents a parking lot configuration --/
structure ParkingLot :=
  (grid : Fin 7 → Fin 7 → Bool)
  (gate : Fin 7 × Fin 7)

/-- Checks if a car can exit from its position --/
def canExit (lot : ParkingLot) (pos : Fin 7 × Fin 7) : Prop :=
  sorry

/-- Counts the number of cars in the parking lot --/
def carCount (lot : ParkingLot) : Nat :=
  sorry

/-- Checks if the parking lot configuration is valid --/
def isValidConfig (lot : ParkingLot) : Prop :=
  ∀ pos, lot.grid pos.1 pos.2 → canExit lot pos

/-- The main theorem --/
theorem max_cars_in_parking_lot :
  ∃ (lot : ParkingLot),
    isValidConfig lot ∧
    carCount lot = 28 ∧
    ∀ (other : ParkingLot), isValidConfig other → carCount other ≤ 28 :=
  sorry

end max_cars_in_parking_lot_l3129_312971


namespace log_expression_equals_half_l3129_312994

theorem log_expression_equals_half :
  (1/2) * (Real.log 12 / Real.log 6) - (Real.log (Real.sqrt 2) / Real.log 6) = 1/2 := by
  sorry

end log_expression_equals_half_l3129_312994


namespace rational_expression_value_l3129_312901

theorem rational_expression_value (a b c d m : ℚ) : 
  a ≠ 0 ∧ 
  a + b = 0 ∧ 
  c * d = 1 ∧ 
  (m = -5 ∨ m = 1) → 
  |m| - a/b + (a+b)/2020 - c*d = 1 ∨ |m| - a/b + (a+b)/2020 - c*d = 5 :=
by sorry

end rational_expression_value_l3129_312901


namespace solve_linear_equation_l3129_312930

theorem solve_linear_equation (x : ℚ) :
  3 * x - 5 * x + 6 * x = 150 → x = 37.5 := by
  sorry

end solve_linear_equation_l3129_312930


namespace red_cars_count_l3129_312979

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 90 → ratio_red = 3 → ratio_black = 8 → 
  ∃ red_cars : ℕ, red_cars * ratio_black = black_cars * ratio_red ∧ red_cars = 33 :=
by
  sorry

end red_cars_count_l3129_312979


namespace f_three_zeros_range_l3129_312946

/-- The function f(x) = x^2 * exp(x) - a -/
noncomputable def f (x a : ℝ) : ℝ := x^2 * Real.exp x - a

/-- The statement that f has exactly three zeros -/
def has_exactly_three_zeros (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧ 
    (f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0) ∧
    (∀ x : ℝ, f x a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)

/-- The theorem stating the range of 'a' for which f has exactly three zeros -/
theorem f_three_zeros_range :
  ∀ a : ℝ, has_exactly_three_zeros a ↔ 0 < a ∧ a < 4 / Real.exp 2 := by
  sorry

end f_three_zeros_range_l3129_312946


namespace chocolate_box_problem_l3129_312928

theorem chocolate_box_problem (day1 day2 day3 day4 remaining : ℕ) :
  day1 = 4 →
  day2 = 2 * day1 - 3 →
  day3 = day1 - 2 →
  day4 = day3 - 1 →
  remaining = 12 →
  day1 + day2 + day3 + day4 + remaining = 24 := by
  sorry

end chocolate_box_problem_l3129_312928


namespace sum_of_roots_quadratic_l3129_312964

theorem sum_of_roots_quadratic (x : ℝ) : 
  (∃ r1 r2 : ℝ, r1 + r2 = 5 ∧ x^2 - 5*x + 6 = (x - r1) * (x - r2)) :=
by sorry

end sum_of_roots_quadratic_l3129_312964


namespace exam_scores_l3129_312968

theorem exam_scores (total_items : Nat) (lowella_percentage : Nat) (pamela_increase : Nat) :
  total_items = 100 →
  lowella_percentage = 35 →
  pamela_increase = 20 →
  let lowella_score := total_items * lowella_percentage / 100
  let pamela_score := lowella_score + lowella_score * pamela_increase / 100
  let mandy_score := 2 * pamela_score
  mandy_score = 84 := by sorry

end exam_scores_l3129_312968


namespace factorization_equality_l3129_312922

theorem factorization_equality (x : ℝ) : -3*x^3 + 12*x^2 - 12*x = -3*x*(x-2)^2 := by
  sorry

end factorization_equality_l3129_312922


namespace simplify_trig_expression_simplify_trig_expression_second_quadrant_l3129_312918

-- Problem 1
theorem simplify_trig_expression : 
  (Real.sqrt (1 - 2 * Real.sin (130 * π / 180) * Real.cos (130 * π / 180))) / 
  (Real.sin (130 * π / 180) + Real.sqrt (1 - Real.sin (130 * π / 180) ^ 2)) = 1 := by sorry

-- Problem 2
theorem simplify_trig_expression_second_quadrant (α : Real) 
  (h : π / 2 < α ∧ α < π) : 
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) + 
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  Real.sin α - Real.cos α := by sorry

end simplify_trig_expression_simplify_trig_expression_second_quadrant_l3129_312918


namespace dilution_proof_l3129_312900

def initial_volume : ℝ := 12
def initial_concentration : ℝ := 0.60
def final_concentration : ℝ := 0.40
def water_added : ℝ := 6

theorem dilution_proof :
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  initial_alcohol / final_volume = final_concentration :=
by sorry

end dilution_proof_l3129_312900


namespace notebook_cost_l3129_312948

/-- Given a notebook and its cover with a total pre-tax cost of $3.00,
    where the notebook costs $2 more than its cover,
    prove that the pre-tax cost of the notebook is $2.50. -/
theorem notebook_cost (notebook_cost cover_cost : ℝ) : 
  notebook_cost + cover_cost = 3 →
  notebook_cost = cover_cost + 2 →
  notebook_cost = 2.5 := by
  sorry

end notebook_cost_l3129_312948


namespace even_mono_decreasing_relation_l3129_312991

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f is monotonically decreasing on [0, +∞) if
    for all x, y ≥ 0, x < y implies f(x) > f(y) -/
def IsMonoDecreasingOnNonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x > f y

theorem even_mono_decreasing_relation
    (f : ℝ → ℝ)
    (h_even : IsEven f)
    (h_mono : IsMonoDecreasingOnNonnegatives f) :
    f 1 > f (-6) := by
  sorry

end even_mono_decreasing_relation_l3129_312991


namespace equation_solution_l3129_312974

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), (x₁ = 5 ∧ x₂ = -1) ∧ 
  (∀ x : ℝ, (x - 2)^2 = 9 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end equation_solution_l3129_312974


namespace circle_center_is_two_one_l3129_312990

/-- A line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A circle defined by its center and a point on its circumference -/
structure Circle where
  center : ℝ × ℝ
  point_on_circle : ℝ × ℝ

/-- The line l passing through (2, 1) and (6, 3) -/
def l : Line := { point1 := (2, 1), point2 := (6, 3) }

/-- The circle C with center on line l and tangent to x-axis at (2, 0) -/
noncomputable def C : Circle :=
  { center := sorry,  -- To be proved
    point_on_circle := (2, 0) }

theorem circle_center_is_two_one :
  C.center = (2, 1) := by sorry

end circle_center_is_two_one_l3129_312990


namespace smallest_integer_in_ratio_l3129_312911

theorem smallest_integer_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive integers
  a + b + c = 90 →         -- sum is 90
  b = 3 * a ∧ c = 5 * a →  -- ratio 2:3:5
  a = 10 :=                -- smallest integer is 10
by sorry

end smallest_integer_in_ratio_l3129_312911


namespace tan_alpha_value_l3129_312951

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3/5) 
  (h2 : Real.tan β = 1/3) : 
  Real.tan α = 2/9 := by
  sorry

end tan_alpha_value_l3129_312951


namespace calculation_difference_l3129_312947

theorem calculation_difference : 
  let correct_calc := 8 - (2 + 5)
  let incorrect_calc := 8 - 2 + 5
  correct_calc - incorrect_calc = -10 := by
sorry

end calculation_difference_l3129_312947


namespace probability_white_ball_l3129_312925

/-- The probability of drawing a white ball from a bag containing 2 red balls and 1 white ball is 1/3. -/
theorem probability_white_ball (red_balls white_balls total_balls : ℕ) : 
  red_balls = 2 → white_balls = 1 → total_balls = red_balls + white_balls →
  (white_balls : ℚ) / total_balls = 1 / 3 := by
  sorry

end probability_white_ball_l3129_312925


namespace lara_has_largest_result_l3129_312912

def starting_number : ℕ := 12

def john_result : ℕ := ((starting_number + 3) * 2) - 4
def lara_result : ℕ := (starting_number * 3 + 5) - 6
def miguel_result : ℕ := (starting_number * 2 - 2) + 2

theorem lara_has_largest_result :
  lara_result > john_result ∧ lara_result > miguel_result := by
  sorry

end lara_has_largest_result_l3129_312912


namespace min_value_sum_l3129_312978

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y * z = 27 → a + 3 * b + 9 * c ≤ x + 3 * y + 9 * z :=
by sorry

end min_value_sum_l3129_312978


namespace problem_solution_l3129_312902

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2*(a-1) + b^2*(b-1) + c^2*(c-1) = a*(a-1) + b*(b-1) + c*(c-1)) :
  1956*a^2 + 1986*b^2 + 2016*c^2 = 5958 := by
sorry

end problem_solution_l3129_312902


namespace max_band_members_l3129_312917

/-- Represents a band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ
  totalMembers : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (bf : BandFormation) : Prop :=
  bf.totalMembers < 120 ∧
  bf.totalMembers = bf.rows * bf.membersPerRow + 3 ∧
  bf.totalMembers = (bf.rows + 1) * (bf.membersPerRow + 2)

/-- Theorem stating the maximum number of band members --/
theorem max_band_members :
  ∀ bf : BandFormation, isValidFormation bf → bf.totalMembers ≤ 119 :=
by sorry

end max_band_members_l3129_312917


namespace largest_integer_not_exceeding_a_n_l3129_312915

/-- Sequence a_n defined recursively -/
def a (a₀ : ℕ) : ℕ → ℚ
  | 0 => a₀
  | n + 1 => (a a₀ n)^2 / ((a a₀ n) + 1)

/-- Theorem stating the largest integer not exceeding a_n is a - n -/
theorem largest_integer_not_exceeding_a_n (a₀ : ℕ) (n : ℕ) 
  (h : n ≤ a₀/2 + 1) : 
  ⌊a a₀ n⌋ = a₀ - n := by sorry

end largest_integer_not_exceeding_a_n_l3129_312915


namespace gum_cost_theorem_l3129_312989

/-- Calculates the discounted cost in dollars for a bulk purchase of gum -/
def discounted_gum_cost (quantity : ℕ) (price_per_piece : ℚ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let total_cost := quantity * price_per_piece
  let discount := if quantity > discount_threshold then discount_rate * total_cost else 0
  (total_cost - discount) / 100

theorem gum_cost_theorem :
  discounted_gum_cost 1500 2 (1/10) 1000 = 27 := by
  sorry

end gum_cost_theorem_l3129_312989


namespace largest_common_term_l3129_312904

def isInFirstSequence (n : ℕ) : Prop := ∃ k : ℕ, n = 3 + 8 * k

def isInSecondSequence (n : ℕ) : Prop := ∃ m : ℕ, n = 5 + 9 * m

theorem largest_common_term : 
  (∀ n : ℕ, n > 59 ∧ n ≤ 90 → ¬(isInFirstSequence n ∧ isInSecondSequence n)) ∧ 
  isInFirstSequence 59 ∧ 
  isInSecondSequence 59 :=
sorry

end largest_common_term_l3129_312904


namespace absolute_value_equation_solution_l3129_312934

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |x - 5| = 3*x + 6 :=
by
  -- The unique solution is x = -1/4
  use (-1/4 : ℚ)
  sorry

end absolute_value_equation_solution_l3129_312934


namespace f_of_f_2_l3129_312969

def f (x : ℝ) : ℝ := 2 * x^3 - 4 * x^2 + 3 * x - 1

theorem f_of_f_2 : f (f 2) = 164 := by sorry

end f_of_f_2_l3129_312969


namespace store_display_arrangement_l3129_312973

def stripe_arrangement (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | 1 => 2
  | 2 => 2
  | (n + 3) => stripe_arrangement (n + 1) + stripe_arrangement (n + 2)

theorem store_display_arrangement : 
  stripe_arrangement 10 = 110 := by sorry

end store_display_arrangement_l3129_312973


namespace polynomial_factorization_l3129_312955

theorem polynomial_factorization (x : ℝ) : 
  x^9 + x^6 + x^3 + 1 = (x^3 + 1) * (x^6 - x^3 + 1) := by
  sorry

end polynomial_factorization_l3129_312955


namespace initial_dumbbell_count_l3129_312920

theorem initial_dumbbell_count (dumbbell_weight : ℕ) (added_dumbbells : ℕ) (total_weight : ℕ) : 
  dumbbell_weight = 20 →
  added_dumbbells = 2 →
  total_weight = 120 →
  (total_weight / dumbbell_weight) - added_dumbbells = 4 := by
  sorry

end initial_dumbbell_count_l3129_312920


namespace equation_equivalence_l3129_312919

theorem equation_equivalence (x y : ℝ) :
  (x - 60) / 3 = (4 - 3 * x) / 6 + y ↔ x = (124 + 6 * y) / 5 := by
  sorry

end equation_equivalence_l3129_312919


namespace circles_externally_tangent_l3129_312958

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

def center1 : ℝ × ℝ := (0, 3)
def center2 : ℝ × ℝ := (4, 0)

def radius1 : ℝ := 3
def radius2 : ℝ := 2

theorem circles_externally_tangent :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = radius1 + radius2 := by sorry

end circles_externally_tangent_l3129_312958


namespace sum_of_coefficients_l3129_312910

theorem sum_of_coefficients (a b : ℝ) : 
  (Nat.choose 6 4) * a^4 * b^2 = 135 →
  (Nat.choose 6 5) * a^5 * b = -18 →
  (a + b)^6 = 64 := by
sorry

end sum_of_coefficients_l3129_312910


namespace max_sum_of_squares_l3129_312967

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 →
  n ∈ Finset.range 1982 →
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end max_sum_of_squares_l3129_312967


namespace a_squared_plus_b_minus_c_in_M_l3129_312983

def P : Set ℤ := {x | ∃ k, x = 3*k + 1}
def Q : Set ℤ := {x | ∃ k, x = 3*k - 1}
def M : Set ℤ := {x | ∃ k, x = 3*k}

theorem a_squared_plus_b_minus_c_in_M (a b c : ℤ) 
  (ha : a ∈ P) (hb : b ∈ Q) (hc : c ∈ M) : 
  a^2 + b - c ∈ M :=
by sorry

end a_squared_plus_b_minus_c_in_M_l3129_312983


namespace intersection_point_coords_l3129_312943

/-- A line in a 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-axis, represented as a vertical line passing through (0, 0). -/
def yAxis : Line := { slope := 0, point := (0, 0) }

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- The point where a line intersects the y-axis. -/
def yAxisIntersection (l : Line) : ℝ × ℝ :=
  (0, l.point.2 + l.slope * (0 - l.point.1))

theorem intersection_point_coords (l1 l2 : Line) (P : ℝ × ℝ) :
  l1.slope = 2 →
  parallel l1 l2 →
  l2.point = (-1, 1) →
  P = yAxisIntersection l2 →
  P = (0, 3) := by sorry

end intersection_point_coords_l3129_312943


namespace chef_cakes_problem_l3129_312929

def chef_cakes (total_eggs : ℕ) (fridge_eggs : ℕ) (eggs_per_cake : ℕ) : ℕ :=
  (total_eggs - fridge_eggs) / eggs_per_cake

theorem chef_cakes_problem :
  chef_cakes 60 10 5 = 10 := by
  sorry

end chef_cakes_problem_l3129_312929


namespace gcd_lcm_product_l3129_312903

theorem gcd_lcm_product (a b : ℕ) (h : a = 140 ∧ b = 175) : 
  (Nat.gcd a b) * (Nat.lcm a b) = 24500 := by
  sorry

end gcd_lcm_product_l3129_312903


namespace nine_b_value_l3129_312960

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) :
  9 * b = 216 / 11 := by
  sorry

end nine_b_value_l3129_312960


namespace parabola_point_coordinates_l3129_312914

/-- Given a parabola y^2 = 4x, point A(3,0), and a point P on the parabola,
    if a line through P intersects perpendicularly with x = -1 at B,
    and |PB| = |PA|, then the x-coordinate of P is 2. -/
theorem parabola_point_coordinates (P : ℝ × ℝ) :
  P.2^2 = 4 * P.1 →  -- P is on the parabola y^2 = 4x
  ∃ B : ℝ × ℝ, 
    B.1 = -1 ∧  -- B is on the line x = -1
    (P.2 - B.2) * (P.1 - B.1) = -1 ∧  -- PB is perpendicular to x = -1
    (P.1 - B.1)^2 + (P.2 - B.2)^2 = (P.1 - 3)^2 + P.2^2 →  -- |PB| = |PA|
  P.1 = 2 :=
sorry

end parabola_point_coordinates_l3129_312914


namespace sufficient_not_necessary_condition_l3129_312916

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a < b ∧ b < 0 → 1/a > 1/b) ∧
  (∃ a b, 1/a > 1/b ∧ ¬(a < b ∧ b < 0)) :=
sorry

end sufficient_not_necessary_condition_l3129_312916


namespace count_valid_n_l3129_312908

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, y * y = x

def valid_n (n : ℕ) : Prop :=
  n > 0 ∧
  is_perfect_square (1 * 4 + 2112) ∧
  is_perfect_square (1 * n + 2112) ∧
  is_perfect_square (4 * n + 2112)

theorem count_valid_n :
  ∃ (S : Finset ℕ), (∀ n ∈ S, valid_n n) ∧ S.card = 7 ∧ (∀ n, valid_n n → n ∈ S) :=
sorry

end count_valid_n_l3129_312908


namespace geometric_sequence_sum_l3129_312935

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  q > 1 →
  (4 * (a 2016)^2 - 8 * (a 2016) + 3 = 0) →
  (4 * (a 2017)^2 - 8 * (a 2017) + 3 = 0) →
  a 2018 + a 2019 = 18 :=
by
  sorry

end geometric_sequence_sum_l3129_312935


namespace time_saved_weekly_l3129_312936

/-- Time saved weekly by eliminating a daily habit -/
theorem time_saved_weekly (search_time complain_time : ℕ) (days_per_week : ℕ) : 
  search_time = 8 → complain_time = 3 → days_per_week = 7 →
  (search_time + complain_time) * days_per_week = 77 :=
by sorry

end time_saved_weekly_l3129_312936


namespace vector_sum_magnitude_l3129_312932

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a + 2 • b))^2 + (Prod.snd (a + 2 • b))^2) = 2 * Real.sqrt 3 := by
    sorry

end vector_sum_magnitude_l3129_312932
