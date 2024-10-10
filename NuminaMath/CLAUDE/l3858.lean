import Mathlib

namespace traditionalist_fraction_l3858_385884

theorem traditionalist_fraction (num_provinces : ℕ) (num_progressives : ℕ) (num_traditionalists_per_province : ℕ) :
  num_provinces = 4 →
  num_traditionalists_per_province * 12 = num_progressives →
  (num_traditionalists_per_province * num_provinces) / (num_progressives + num_traditionalists_per_province * num_provinces) = 1 / 4 :=
by sorry

end traditionalist_fraction_l3858_385884


namespace rhombus_area_triple_diagonals_l3858_385864

/-- The area of a rhombus with diagonals that are 3 times longer than a rhombus
    with diagonals 6 cm and 4 cm is 108 cm². -/
theorem rhombus_area_triple_diagonals (d1 d2 : ℝ) : 
  d1 = 6 → d2 = 4 → (3 * d1 * 3 * d2) / 2 = 108 := by
  sorry

end rhombus_area_triple_diagonals_l3858_385864


namespace speech_arrangement_count_l3858_385813

theorem speech_arrangement_count :
  let total_male : ℕ := 4
  let total_female : ℕ := 3
  let selected_male : ℕ := 3
  let selected_female : ℕ := 2
  let total_selected : ℕ := selected_male + selected_female

  (Nat.choose total_male selected_male) *
  (Nat.choose total_female selected_female) *
  (Nat.factorial selected_male) *
  (Nat.factorial (total_selected - 1)) = 864 :=
by sorry

end speech_arrangement_count_l3858_385813


namespace origin_midpoint_coordinates_l3858_385892

/-- Given two points A and B in a 2D Cartesian coordinate system, 
    this function returns true if the origin (0, 0) is the midpoint of AB. -/
def isOriginMidpoint (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 0

/-- Theorem stating that if the origin is the midpoint of AB and 
    A has coordinates (-1, 2), then B has coordinates (1, -2). -/
theorem origin_midpoint_coordinates :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (1, -2)
  isOriginMidpoint A B → B = (1, -2) := by
  sorry


end origin_midpoint_coordinates_l3858_385892


namespace reciprocal_comparison_l3858_385865

theorem reciprocal_comparison : 
  ((-1/3 : ℚ) < (-3 : ℚ) → False) ∧
  ((-3/2 : ℚ) < (-2/3 : ℚ)) ∧
  ((1/4 : ℚ) < (4 : ℚ)) ∧
  ((3/4 : ℚ) < (4/3 : ℚ) → False) ∧
  ((4/3 : ℚ) < (3/4 : ℚ) → False) := by
sorry

end reciprocal_comparison_l3858_385865


namespace points_form_circle_l3858_385891

theorem points_form_circle :
  ∀ (x y : ℝ), (∃ t : ℝ, x = Real.cos t ∧ y = Real.sin t) →
  x^2 + y^2 = 1 :=
by sorry

end points_form_circle_l3858_385891


namespace altitude_angle_bisector_median_concurrent_l3858_385825

/-- Triangle ABC with sides a, b, c -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (a b c : ℝ)
  (side_a : dist B C = a)
  (side_b : dist C A = b)
  (side_c : dist A B = c)

/-- Altitude from A to BC -/
def altitude (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

/-- Angle bisector from B -/
def angle_bisector (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

/-- Median from C to AB -/
def median (t : Triangle) : ℝ × ℝ → ℝ × ℝ := sorry

/-- Three lines are concurrent -/
def concurrent (l₁ l₂ l₃ : ℝ × ℝ → ℝ × ℝ) : Prop := sorry

theorem altitude_angle_bisector_median_concurrent (t : Triangle) :
  concurrent (altitude t) (angle_bisector t) (median t) ↔
  t.a^2 * (t.a - t.c) = (t.b^2 - t.c^2) * (t.a + t.c) :=
sorry

end altitude_angle_bisector_median_concurrent_l3858_385825


namespace solution_set_when_a_is_one_minimum_a_for_always_greater_than_three_l3858_385805

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + x + a

-- Theorem 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 3} = {x : ℝ | x ≥ 1 ∨ x ≤ -2} := by sorry

-- Theorem 2
theorem minimum_a_for_always_greater_than_three :
  (∀ x : ℝ, f a x ≥ 3) ↔ a ≥ 13/4 := by sorry

end solution_set_when_a_is_one_minimum_a_for_always_greater_than_three_l3858_385805


namespace correct_calculation_l3858_385836

theorem correct_calculation (x : ℤ) (h : x + 35 = 77) : x - 35 = 7 := by
  sorry

end correct_calculation_l3858_385836


namespace inequality_system_solution_l3858_385882

theorem inequality_system_solution (x : ℝ) :
  (x + 5 < 4) ∧ ((3 * x + 1) / 2 ≥ 2 * x - 1) → x < -1 := by
  sorry

end inequality_system_solution_l3858_385882


namespace club_truncator_probability_l3858_385868

/-- The number of matches Club Truncator plays -/
def num_matches : ℕ := 8

/-- The probability of winning, losing, or tying a single match -/
def single_match_prob : ℚ := 1/3

/-- The probability of finishing with more wins than losses -/
def more_wins_prob : ℚ := 2741/6561

theorem club_truncator_probability :
  let total_outcomes := 3^num_matches
  let same_wins_losses := 1079
  (total_outcomes - same_wins_losses) / (2 * total_outcomes) = more_wins_prob :=
sorry

end club_truncator_probability_l3858_385868


namespace solution_Y_initial_weight_l3858_385847

/-- Represents the composition and transformation of a solution --/
structure Solution where
  initialWeight : ℝ
  liquidXPercentage : ℝ
  waterPercentage : ℝ
  evaporatedWater : ℝ
  addedSolutionWeight : ℝ
  newLiquidXPercentage : ℝ

/-- Theorem stating the initial weight of solution Y given the conditions --/
theorem solution_Y_initial_weight (s : Solution) 
  (h1 : s.liquidXPercentage = 0.30)
  (h2 : s.waterPercentage = 0.70)
  (h3 : s.liquidXPercentage + s.waterPercentage = 1)
  (h4 : s.evaporatedWater = 2)
  (h5 : s.addedSolutionWeight = 2)
  (h6 : s.newLiquidXPercentage = 0.36)
  (h7 : s.newLiquidXPercentage * s.initialWeight = 
        s.liquidXPercentage * s.initialWeight + 
        s.liquidXPercentage * s.addedSolutionWeight) :
  s.initialWeight = 10 := by
  sorry


end solution_Y_initial_weight_l3858_385847


namespace power_equality_implies_n_equals_four_l3858_385800

theorem power_equality_implies_n_equals_four (n : ℕ) : 4^8 = 16^n → n = 4 := by
  sorry

end power_equality_implies_n_equals_four_l3858_385800


namespace probability_red_or_white_l3858_385877

def total_marbles : ℕ := 50
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white :
  (red_marbles + white_marbles : ℚ) / total_marbles = 9 / 10 := by
  sorry

end probability_red_or_white_l3858_385877


namespace grape_juice_percentage_l3858_385811

/-- Calculates the percentage of grape juice in a mixture after adding pure grape juice -/
theorem grape_juice_percentage 
  (original_volume : ℝ) 
  (original_percentage : ℝ) 
  (added_volume : ℝ) : 
  original_volume = 40 →
  original_percentage = 0.1 →
  added_volume = 10 →
  (original_volume * original_percentage + added_volume) / (original_volume + added_volume) = 0.28 := by
sorry


end grape_juice_percentage_l3858_385811


namespace wednesday_spending_multiple_l3858_385850

def monday_spending : ℝ := 60
def tuesday_spending : ℝ := 4 * monday_spending
def total_spending : ℝ := 600

theorem wednesday_spending_multiple : 
  ∃ x : ℝ, 
    monday_spending + tuesday_spending + x * monday_spending = total_spending ∧ 
    x = 5 := by
  sorry

end wednesday_spending_multiple_l3858_385850


namespace article_purchase_price_l3858_385841

/-- The purchase price of an article given specific markup conditions -/
theorem article_purchase_price : 
  ∀ (markup overhead_percentage net_profit purchase_price : ℝ),
  markup = 40 →
  overhead_percentage = 0.15 →
  net_profit = 12 →
  markup = overhead_percentage * purchase_price + net_profit →
  purchase_price = 186.67 := by
sorry

end article_purchase_price_l3858_385841


namespace fourth_row_middle_cells_l3858_385807

/-- Represents a letter in the grid -/
inductive Letter : Type
| A | B | C | D | E | F

/-- Represents a position in the grid -/
structure Position :=
  (row : Fin 6)
  (col : Fin 6)

/-- Represents the 6x6 grid -/
def Grid := Position → Letter

/-- Checks if a 2x3 rectangle is valid (no repeats) -/
def validRectangle (g : Grid) (topLeft : Position) : Prop :=
  ∀ (i j : Fin 2) (k : Fin 3),
    g ⟨topLeft.row + i, topLeft.col + k⟩ ≠ g ⟨topLeft.row + j, topLeft.col + k⟩ ∨ i = j

/-- Checks if the entire grid is valid -/
def validGrid (g : Grid) : Prop :=
  (∀ r : Fin 6, ∀ i j : Fin 6, g ⟨r, i⟩ ≠ g ⟨r, j⟩ ∨ i = j) ∧  -- No repeats in rows
  (∀ c : Fin 6, ∀ i j : Fin 6, g ⟨i, c⟩ ≠ g ⟨j, c⟩ ∨ i = j) ∧  -- No repeats in columns
  (∀ r c : Fin 2, validRectangle g ⟨3*r, 3*c⟩)                 -- Valid 2x3 rectangles

/-- The main theorem -/
theorem fourth_row_middle_cells (g : Grid) (h : validGrid g) :
  g ⟨3, 1⟩ = Letter.E ∧
  g ⟨3, 2⟩ = Letter.D ∧
  g ⟨3, 3⟩ = Letter.C ∧
  g ⟨3, 4⟩ = Letter.F :=
by sorry

end fourth_row_middle_cells_l3858_385807


namespace unique_fraction_sum_l3858_385897

theorem unique_fraction_sum (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (n m : ℕ), n ≠ m ∧ 2/p = 1/n + 1/m ∧ n = (p + 1)/2 ∧ m = p * (p + 1)/2 :=
by sorry

end unique_fraction_sum_l3858_385897


namespace johns_remaining_money_l3858_385883

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Proves that John's remaining money after buying the ticket is 1725 dollars -/
theorem johns_remaining_money :
  let savings := base8_to_base10 5555
  let ticket_cost := 1200
  savings - ticket_cost = 1725 := by sorry

end johns_remaining_money_l3858_385883


namespace decagon_triangles_l3858_385852

theorem decagon_triangles : ∀ n : ℕ, n = 10 → (n.choose 3) = 120 := by sorry

end decagon_triangles_l3858_385852


namespace hyperbola_asymptote_perpendicular_line_l3858_385824

/-- Represents a hyperbola in the Cartesian plane -/
structure Hyperbola where
  a : ℝ
  equation : ℝ → ℝ → Prop
  asymptote : ℝ → ℝ → Prop

/-- Represents a line in the Cartesian plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def Perpendicular (l1 l2 : Line) : Prop :=
  ∃ m1 m2 : ℝ, (∀ x y, l1.equation x y ↔ y = m1 * x + (0 : ℝ)) ∧ 
              (∀ x y, l2.equation x y ↔ y = m2 * x + (0 : ℝ)) ∧ 
              m1 * m2 = -1

/-- The main theorem -/
theorem hyperbola_asymptote_perpendicular_line (h : Hyperbola) (l : Line) : 
  h.a > 0 ∧ 
  (∀ x y, h.equation x y ↔ x^2 / h.a^2 - y^2 = 1) ∧
  (∀ x y, l.equation x y ↔ 2*x - y + 1 = 0) ∧
  (∃ la : Line, h.asymptote = la.equation ∧ Perpendicular la l) →
  h.a = 2 := by
  sorry

end hyperbola_asymptote_perpendicular_line_l3858_385824


namespace shaded_area_of_overlapping_sectors_l3858_385880

/-- The area of the shaded region formed by two overlapping sectors of a circle -/
theorem shaded_area_of_overlapping_sectors (r : ℝ) (θ : ℝ) (h1 : r = 15) (h2 : θ = 45) :
  let sector_area := θ / 360 * π * r^2
  let triangle_area := Real.sqrt 3 / 4 * r^2
  2 * (sector_area - triangle_area) = 56.25 * π - 112.5 * Real.sqrt 3 := by
  sorry

end shaded_area_of_overlapping_sectors_l3858_385880


namespace village_population_l3858_385894

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.15) = 3553 → P = 4400 := by
  sorry

end village_population_l3858_385894


namespace orange_selling_gain_percentage_orange_selling_specific_gain_l3858_385826

/-- Calculates the gain percentage when changing selling rates of oranges -/
theorem orange_selling_gain_percentage 
  (initial_rate : ℝ) 
  (initial_loss_percentage : ℝ)
  (new_rate : ℝ) : ℝ :=
  let cost_price := 1 / (initial_rate * (1 - initial_loss_percentage / 100))
  let new_selling_price := 1 / new_rate
  let gain_percentage := (new_selling_price / cost_price - 1) * 100
  gain_percentage

/-- Proves that the specific change in orange selling rates results in a 44% gain -/
theorem orange_selling_specific_gain : 
  orange_selling_gain_percentage 12 10 7.5 = 44 := by
  sorry

end orange_selling_gain_percentage_orange_selling_specific_gain_l3858_385826


namespace inequality_solution_set_l3858_385840

theorem inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | a * x^2 - (a + 2) * x + 2 ≥ 0} = {x : ℝ | 2/a ≤ x ∧ x ≤ 1} := by
  sorry

end inequality_solution_set_l3858_385840


namespace trapezoid_garden_bases_l3858_385806

theorem trapezoid_garden_bases :
  let area : ℕ := 1350
  let altitude : ℕ := 45
  let valid_pair (b₁ b₂ : ℕ) : Prop :=
    area = (altitude * (b₁ + b₂)) / 2 ∧
    b₁ % 9 = 0 ∧
    b₂ % 9 = 0 ∧
    b₁ > 0 ∧
    b₂ > 0
  ∃! (pairs : Finset (ℕ × ℕ)), pairs.card = 3 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs ↔ valid_pair p.1 p.2 :=
by sorry

end trapezoid_garden_bases_l3858_385806


namespace stickers_needed_for_prizes_l3858_385843

def christine_stickers : ℕ := 2500
def robert_stickers : ℕ := 1750
def small_prize_requirement : ℕ := 4000
def medium_prize_requirement : ℕ := 7000
def large_prize_requirement : ℕ := 10000

def total_stickers : ℕ := christine_stickers + robert_stickers

theorem stickers_needed_for_prizes :
  (max 0 (small_prize_requirement - total_stickers) = 0) ∧
  (max 0 (medium_prize_requirement - total_stickers) = 2750) ∧
  (max 0 (large_prize_requirement - total_stickers) = 5750) := by
  sorry

end stickers_needed_for_prizes_l3858_385843


namespace smallest_difference_in_triangle_l3858_385835

theorem smallest_difference_in_triangle (PQ QR PR : ℕ) : 
  PQ + QR + PR = 2021 →
  PQ < QR →
  QR ≤ PR →
  PQ + QR > PR →
  PQ + PR > QR →
  QR + PR > PQ →
  ∃ (PQ' QR' PR' : ℕ), 
    PQ' + QR' + PR' = 2021 ∧
    PQ' < QR' ∧
    QR' ≤ PR' ∧
    PQ' + QR' > PR' ∧
    PQ' + PR' > QR' ∧
    QR' + PR' > PQ' ∧
    QR' - PQ' = 1 ∧
    ∀ (PQ'' QR'' PR'' : ℕ),
      PQ'' + QR'' + PR'' = 2021 →
      PQ'' < QR'' →
      QR'' ≤ PR'' →
      PQ'' + QR'' > PR'' →
      PQ'' + PR'' > QR'' →
      QR'' + PR'' > PQ'' →
      QR'' - PQ'' ≥ 1 :=
by sorry

end smallest_difference_in_triangle_l3858_385835


namespace maintenance_check_increase_l3858_385818

theorem maintenance_check_increase (original_days new_days : ℝ) 
  (h1 : original_days = 20)
  (h2 : new_days = 25) :
  ((new_days - original_days) / original_days) * 100 = 25 := by
  sorry

end maintenance_check_increase_l3858_385818


namespace g_at_negative_two_l3858_385889

/-- The function g is defined as g(x) = 2x^2 - 3x + 1 for all real x. -/
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

/-- Theorem: The value of g(-2) is 15. -/
theorem g_at_negative_two : g (-2) = 15 := by
  sorry

end g_at_negative_two_l3858_385889


namespace simplify_sqrt_difference_l3858_385834

theorem simplify_sqrt_difference : 
  (Real.sqrt 528 / Real.sqrt 64) - (Real.sqrt 242 / Real.sqrt 121) = 1.461 := by
  sorry

end simplify_sqrt_difference_l3858_385834


namespace existence_of_common_source_l3858_385823

/-- Represents the process of obtaining one number from another through digit manipulation -/
def Obtainable (m n : ℕ) : Prop := sorry

/-- Checks if a natural number contains the digit 5 in its decimal representation -/
def ContainsDigitFive (n : ℕ) : Prop := sorry

theorem existence_of_common_source (S : Finset ℕ) 
  (h1 : S.Nonempty) 
  (h2 : ∀ s ∈ S, ¬ContainsDigitFive s) : 
  ∃ N : ℕ, ∀ s ∈ S, Obtainable s N := by sorry

end existence_of_common_source_l3858_385823


namespace jims_journey_distance_l3858_385802

/-- The total distance of Jim's journey -/
def total_distance (driven : ℕ) (remaining : ℕ) : ℕ := driven + remaining

/-- Theorem stating the total distance of Jim's journey -/
theorem jims_journey_distance :
  total_distance 642 558 = 1200 := by
  sorry

end jims_journey_distance_l3858_385802


namespace data_analysis_l3858_385887

def data : List ℝ := [7, 11, 10, 11, 6, 14, 11, 10, 11, 9]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_analysis (d : List ℝ) (h : d = data) : 
  mode d = 11 ∧ 
  median d ≠ 10 ∧ 
  mean d = 10 ∧ 
  variance d = 4.6 := by sorry

end data_analysis_l3858_385887


namespace square_circle_ratio_l3858_385815

theorem square_circle_ratio (s r : ℝ) (h : s > 0 ∧ r > 0) :
  s^2 / (π * r^2) = 250 / 196 →
  ∃ (a b c : ℕ), (a : ℝ) * Real.sqrt b / c = s / r ∧ a = 5 ∧ b = 10 ∧ c = 14 ∧ a + b + c = 29 :=
by sorry

end square_circle_ratio_l3858_385815


namespace third_score_calculation_l3858_385879

/-- Given four scores where three are known and their average with an unknown fourth score is 76.6,
    prove that the unknown score must be 79.4. -/
theorem third_score_calculation (score1 score2 score4 : ℝ) (average : ℝ) :
  score1 = 65 →
  score2 = 67 →
  score4 = 95 →
  average = 76.6 →
  ∃ score3 : ℝ, score3 = 79.4 ∧ (score1 + score2 + score3 + score4) / 4 = average :=
by sorry

end third_score_calculation_l3858_385879


namespace distinct_prime_factors_of_450_l3858_385858

theorem distinct_prime_factors_of_450 : Nat.card (Nat.factors 450).toFinset = 3 := by
  sorry

end distinct_prime_factors_of_450_l3858_385858


namespace rectangular_parallelepiped_exists_l3858_385808

theorem rectangular_parallelepiped_exists : ∃ (a b c : ℕ+), 2 * (a * b + b * c + c * a) = 4 * (a + b + c) := by
  sorry

end rectangular_parallelepiped_exists_l3858_385808


namespace equidistant_is_circumcenter_l3858_385846

/-- Triangle represented by complex coordinates of its vertices -/
structure ComplexTriangle where
  z₁ : ℂ
  z₂ : ℂ
  z₃ : ℂ

/-- A point is equidistant from all vertices of the triangle -/
def isEquidistant (z : ℂ) (t : ComplexTriangle) : Prop :=
  Complex.abs (z - t.z₁) = Complex.abs (z - t.z₂) ∧
  Complex.abs (z - t.z₂) = Complex.abs (z - t.z₃)

/-- The circumcenter of a triangle -/
def isCircumcenter (z : ℂ) (t : ComplexTriangle) : Prop :=
  -- Definition of circumcenter (placeholder)
  True

theorem equidistant_is_circumcenter (t : ComplexTriangle) (z : ℂ) :
  isEquidistant z t → isCircumcenter z t := by
  sorry

end equidistant_is_circumcenter_l3858_385846


namespace probability_of_red_from_B_mutually_exclusive_events_l3858_385809

structure Bag where
  red : ℕ
  white : ℕ
  black : ℕ

def bagA : Bag := ⟨5, 2, 3⟩
def bagB : Bag := ⟨4, 3, 3⟩

def totalBalls (bag : Bag) : ℕ := bag.red + bag.white + bag.black

def P_A1 : ℚ := bagA.red / totalBalls bagA
def P_A2 : ℚ := bagA.white / totalBalls bagA
def P_A3 : ℚ := bagA.black / totalBalls bagA

def P_B_given_A1 : ℚ := (bagB.red + 1) / (totalBalls bagB + 1)
def P_B_given_A2 : ℚ := bagB.red / (totalBalls bagB + 1)
def P_B_given_A3 : ℚ := bagB.red / (totalBalls bagB + 1)

def P_B : ℚ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3

theorem probability_of_red_from_B : P_B = 9 / 22 := by sorry

theorem mutually_exclusive_events : P_A1 + P_A2 + P_A3 = 1 := by sorry

end probability_of_red_from_B_mutually_exclusive_events_l3858_385809


namespace impossible_arrangement_l3858_385899

theorem impossible_arrangement : ¬ ∃ (seq : Fin 3972 → Fin 1986), 
  (∀ k : Fin 1986, (∃! i j : Fin 3972, seq i = k ∧ seq j = k ∧ i ≠ j)) ∧
  (∀ k : Fin 1986, ∀ i j : Fin 3972, 
    seq i = k → seq j = k → i ≠ j → 
    (j.val > i.val → j.val - i.val = k.val + 1)) :=
by sorry

end impossible_arrangement_l3858_385899


namespace min_value_of_function_min_value_is_lower_bound_l3858_385890

theorem min_value_of_function (x : ℝ) (h : x > 1) : 
  2 * x + 2 / (x - 1) ≥ 6 := by
  sorry

theorem min_value_is_lower_bound (ε : ℝ) (hε : ε > 0) : 
  ∃ x : ℝ, x > 1 ∧ 2 * x + 2 / (x - 1) < 6 + ε := by
  sorry

end min_value_of_function_min_value_is_lower_bound_l3858_385890


namespace fraction_reducibility_l3858_385844

def is_reducible (a : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d ∣ (2 * a + 5) ∧ d ∣ (3 * a + 4)

theorem fraction_reducibility (a : ℕ) :
  is_reducible a ↔ ∃ k : ℕ, a = 7 * k + 1 :=
by sorry

end fraction_reducibility_l3858_385844


namespace percentage_increase_l3858_385848

theorem percentage_increase (original : ℝ) (new : ℝ) :
  original = 30 ∧ new = 40 →
  (new - original) / original * 100 = 100 / 3 :=
by sorry

end percentage_increase_l3858_385848


namespace acid_mixture_proof_l3858_385832

-- Define the volumes and concentrations
def volume_60_percent : ℝ := 4
def concentration_60_percent : ℝ := 0.60
def volume_75_percent : ℝ := 16
def concentration_75_percent : ℝ := 0.75
def total_volume : ℝ := 20
def final_concentration : ℝ := 0.72

-- Theorem statement
theorem acid_mixture_proof :
  (volume_60_percent * concentration_60_percent + 
   volume_75_percent * concentration_75_percent) / total_volume = final_concentration :=
by sorry

end acid_mixture_proof_l3858_385832


namespace regular_octagon_interior_angle_l3858_385872

/-- The measure of an interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_interior_angles : ℝ := (n - 2) * 180  -- sum of interior angles formula
  let interior_angle : ℝ := sum_interior_angles / n  -- each interior angle measure
  135

/-- Proof of the theorem -/
lemma prove_regular_octagon_interior_angle : 
  regular_octagon_interior_angle = 135 := by
  sorry

end regular_octagon_interior_angle_l3858_385872


namespace square_roots_theorem_l3858_385831

theorem square_roots_theorem (n : ℝ) (h : n > 0) :
  (∃ a : ℝ, (2*a + 1)^2 = n ∧ (a + 5)^2 = n) → 
  (∃ a : ℝ, 2*a + 1 + a + 5 = 0) :=
by sorry

end square_roots_theorem_l3858_385831


namespace fraction_equation_transformation_l3858_385863

/-- Given the fractional equation (x / (x - 1)) - (2 / x) = 1,
    prove that eliminating the denominators results in x^2 - 2(x-1) = x(x-1) -/
theorem fraction_equation_transformation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x / (x - 1)) - (2 / x) = 1 ↔ x^2 - 2*(x-1) = x*(x-1) :=
by sorry

end fraction_equation_transformation_l3858_385863


namespace trig_identities_l3858_385804

theorem trig_identities (α β γ : Real) (h : α + β + γ = Real.pi) :
  (Real.sin α)^2 + (Real.sin β)^2 - (Real.sin γ)^2 = 2 * Real.sin α * Real.sin β * Real.cos γ ∧
  (Real.cos α)^2 + (Real.cos β)^2 - (Real.cos γ)^2 = 1 - 2 * Real.sin α * Real.sin β * Real.cos γ := by
  sorry

end trig_identities_l3858_385804


namespace x_squared_gt_16_necessary_not_sufficient_for_x_gt_4_l3858_385839

theorem x_squared_gt_16_necessary_not_sufficient_for_x_gt_4 :
  (∃ x : ℝ, x^2 > 16 ∧ x ≤ 4) ∧
  (∀ x : ℝ, x > 4 → x^2 > 16) :=
by sorry

end x_squared_gt_16_necessary_not_sufficient_for_x_gt_4_l3858_385839


namespace allocation_schemes_l3858_385851

theorem allocation_schemes (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 9) :
  (Nat.choose (n + k - 1) (k - 1)) = 165 := by
  sorry

end allocation_schemes_l3858_385851


namespace election_majority_l3858_385875

/-- Calculates the majority in an election --/
theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 7520 → 
  winning_percentage = 60 / 100 → 
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 1504 := by
  sorry


end election_majority_l3858_385875


namespace apple_pairing_l3858_385830

theorem apple_pairing (weights : Fin 300 → ℝ) 
  (h_positive : ∀ i, weights i > 0)
  (h_ratio : ∀ i j, weights i ≤ 3 * weights j) :
  ∃ (pairs : Fin 150 → Fin 300 × Fin 300),
    (∀ i, (pairs i).1 ≠ (pairs i).2) ∧
    (∀ i, i ≠ j → (pairs i).1 ≠ (pairs j).1 ∧ (pairs i).1 ≠ (pairs j).2 ∧
                  (pairs i).2 ≠ (pairs j).1 ∧ (pairs i).2 ≠ (pairs j).2) ∧
    (∀ i j, weights (pairs i).1 + weights (pairs i).2 ≤ 
            2 * (weights (pairs j).1 + weights (pairs j).2)) :=
sorry

end apple_pairing_l3858_385830


namespace consecutive_negative_integers_sum_l3858_385878

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n + 1 < 0 ∧ n * (n + 1) = 2550 → n + (n + 1) = -101 := by
  sorry

end consecutive_negative_integers_sum_l3858_385878


namespace min_max_x_given_xy_eq_nx_plus_ny_l3858_385895

theorem min_max_x_given_xy_eq_nx_plus_ny (n x y : ℕ+) (h : x * y = n * x + n * y) :
  x ≥ n + 1 ∧ x ≤ n * (n + 1) :=
sorry

end min_max_x_given_xy_eq_nx_plus_ny_l3858_385895


namespace exists_number_with_digit_sum_property_l3858_385814

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with specific digit sum properties -/
theorem exists_number_with_digit_sum_property :
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n^2) = sum_of_digits (1000^2) := by
  sorry

end exists_number_with_digit_sum_property_l3858_385814


namespace square_ABCD_l3858_385893

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if a quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop :=
  let AB := (q.B.x - q.A.x, q.B.y - q.A.y)
  let BC := (q.C.x - q.B.x, q.C.y - q.B.y)
  let CD := (q.D.x - q.C.x, q.D.y - q.C.y)
  let DA := (q.A.x - q.D.x, q.A.y - q.D.y)
  -- All sides have equal length
  AB.1^2 + AB.2^2 = BC.1^2 + BC.2^2 ∧
  BC.1^2 + BC.2^2 = CD.1^2 + CD.2^2 ∧
  CD.1^2 + CD.2^2 = DA.1^2 + DA.2^2 ∧
  -- Adjacent sides are perpendicular
  AB.1 * BC.1 + AB.2 * BC.2 = 0 ∧
  BC.1 * CD.1 + BC.2 * CD.2 = 0 ∧
  CD.1 * DA.1 + CD.2 * DA.2 = 0 ∧
  DA.1 * AB.1 + DA.2 * AB.2 = 0

theorem square_ABCD :
  let q := Quadrilateral.mk
    (Point.mk (-1) 3)
    (Point.mk 1 (-2))
    (Point.mk 6 0)
    (Point.mk 4 5)
  is_square q := by
  sorry

end square_ABCD_l3858_385893


namespace range_of_a_l3858_385886

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

end range_of_a_l3858_385886


namespace max_product_of_digits_l3858_385871

theorem max_product_of_digits (A B : ℕ) : 
  (A ≤ 9) → 
  (B ≤ 9) → 
  (∃ (n : ℕ), A * 100000 + 2021 * 100 + B = 9 * n) →
  A * B ≤ 42 :=
sorry

end max_product_of_digits_l3858_385871


namespace correct_calculation_l3858_385842

theorem correct_calculation (x : ℚ) (h : 6 * x = 42) : 3 * x = 21 := by
  sorry

end correct_calculation_l3858_385842


namespace fixed_points_for_specific_values_two_fixed_points_condition_l3858_385855

/-- Definition of a fixed point for a function f -/
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

/-- The given function f(x) = ax² + (b + 1)x + b - 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

/-- Theorem: The function f has fixed points at 3 and -1 when a = 1 and b = -2 -/
theorem fixed_points_for_specific_values :
  is_fixed_point (f 1 (-2)) 3 ∧ is_fixed_point (f 1 (-2)) (-1) :=
sorry

/-- Theorem: The function f always has two fixed points for any real b if and only if 0 < a < 1 -/
theorem two_fixed_points_condition (a : ℝ) :
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) ↔
  (0 < a ∧ a < 1) :=
sorry

end fixed_points_for_specific_values_two_fixed_points_condition_l3858_385855


namespace positive_numbers_l3858_385856

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (pairwise_sum_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end positive_numbers_l3858_385856


namespace angle_between_vectors_l3858_385837

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define vectors a and b
variable (a b : V)

-- State the theorem
theorem angle_between_vectors 
  (h1 : ‖a‖ = Real.sqrt 3)
  (h2 : ‖b‖ = 1)
  (h3 : ‖a - 2 • b‖ = 1) :
  Real.arccos (inner a b / (‖a‖ * ‖b‖)) = π / 6 := by
  sorry

end angle_between_vectors_l3858_385837


namespace quadratic_equation_solution_l3858_385888

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 4*x} = {0, 4} := by sorry

end quadratic_equation_solution_l3858_385888


namespace log_expression_equality_l3858_385867

theorem log_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - (5 : ℝ) ^ (Real.log 3 / Real.log 5) = -3 := by
  sorry

end log_expression_equality_l3858_385867


namespace ericas_amount_l3858_385821

/-- The problem of calculating Erica's amount given the total and Sam's amount -/
theorem ericas_amount (total sam : ℚ) (h1 : total = 450.32) (h2 : sam = 325.67) :
  total - sam = 124.65 := by
  sorry

end ericas_amount_l3858_385821


namespace special_square_midpoint_sum_l3858_385810

/-- A square in the first quadrant with specific points on its sides -/
structure SpecialSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  in_first_quadrant : A.1 ≥ 0 ∧ A.2 ≥ 0 ∧ B.1 ≥ 0 ∧ B.2 ≥ 0 ∧ C.1 ≥ 0 ∧ C.2 ≥ 0 ∧ D.1 ≥ 0 ∧ D.2 ≥ 0
  is_square : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
              (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2 ∧
              (C.1 - D.1)^2 + (C.2 - D.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2
  point_on_AD : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (2, 0) = (t * A.1 + (1 - t) * D.1, t * A.2 + (1 - t) * D.2)
  point_on_BC : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (6, 0) = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)
  point_on_AB : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (10, 0) = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)
  point_on_CD : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (14, 0) = (t * C.1 + (1 - t) * D.1, t * C.2 + (1 - t) * D.2)

/-- The sum of coordinates of the midpoint of the special square is 10 -/
theorem special_square_midpoint_sum (sq : SpecialSquare) :
  (sq.A.1 + sq.C.1) / 2 + (sq.A.2 + sq.C.2) / 2 = 10 := by
  sorry

end special_square_midpoint_sum_l3858_385810


namespace angle_C_measure_side_c_length_l3858_385870

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition from the problem -/
def condition (t : Triangle) : Prop :=
  (t.a * Real.cos t.B + t.b * Real.cos t.A) / t.c = 2 * Real.cos t.C

theorem angle_C_measure (t : Triangle) (h : condition t) : t.C = π / 3 := by
  sorry

theorem side_c_length (t : Triangle) 
  (h1 : (1 / 2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3)
  (h2 : t.a + t.b = 6)
  (h3 : t.C = π / 3) : t.c = 2 * Real.sqrt 3 := by
  sorry

end angle_C_measure_side_c_length_l3858_385870


namespace trig_identity_l3858_385822

theorem trig_identity (x : ℝ) (h : Real.sin (π / 6 - x) = 1 / 2) :
  Real.sin (19 * π / 6 - x) + (Real.sin (-2 * π / 3 + x))^2 = 1 / 4 := by
  sorry

end trig_identity_l3858_385822


namespace range_of_m_l3858_385829

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, f m x < -m + 4) → m < 5/7 :=
by sorry

end range_of_m_l3858_385829


namespace simplify_fraction_calculate_logarithmic_expression_l3858_385857

-- Part 1
theorem simplify_fraction (a : ℝ) (ha : a > 0) :
  a^2 / (Real.sqrt a * 3 * a^2) = a^(5/6) := by sorry

-- Part 2
theorem calculate_logarithmic_expression :
  (2 * Real.log 2 + Real.log 3) / (1 + 1/2 * Real.log 0.36 + 1/3 * Real.log 8) = 1 := by sorry

end simplify_fraction_calculate_logarithmic_expression_l3858_385857


namespace water_tank_capacity_l3858_385833

theorem water_tank_capacity : ∀ (C : ℝ),
  (∃ (x : ℝ), x / C = 1 / 3 ∧ (x + 6) / C = 1 / 2) →
  C = 36 := by
sorry

end water_tank_capacity_l3858_385833


namespace chocolate_division_l3858_385881

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) :
  total_chocolate = 48/5 →
  num_piles = 4 →
  total_chocolate / num_piles = 12/5 := by
sorry

end chocolate_division_l3858_385881


namespace butter_problem_l3858_385853

theorem butter_problem (B : ℝ) : 
  (B / 2 + B / 5 + (B - B / 2 - B / 5) / 3 + 2 = B) → B = 10 :=
by sorry

end butter_problem_l3858_385853


namespace ratio_calculation_l3858_385849

theorem ratio_calculation (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) 
  (hw : w ≠ 0) : 
  x * z / (y * w) = 20 := by
  sorry

end ratio_calculation_l3858_385849


namespace parabola_transformation_l3858_385801

/-- A parabola is above a line if it opens upwards and doesn't intersect the line. -/
def parabola_above_line (a b c : ℝ) : Prop :=
  a > 0 ∧ (b - c)^2 < 4*a*c

theorem parabola_transformation (a b c : ℝ) 
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (above : parabola_above_line a b c) : 
  parabola_above_line c (-b) a :=
sorry

end parabola_transformation_l3858_385801


namespace remaining_marbles_l3858_385862

/-- Given Chris has 12 marbles and Ryan has 28 marbles, if they combine their marbles
    and each takes 1/4 of the total, the number of marbles remaining in the pile is 20. -/
theorem remaining_marbles (chris_marbles : ℕ) (ryan_marbles : ℕ) 
    (h1 : chris_marbles = 12) 
    (h2 : ryan_marbles = 28) : 
  let total_marbles := chris_marbles + ryan_marbles
  let taken_marbles := 2 * (total_marbles / 4)
  total_marbles - taken_marbles = 20 := by
  sorry

end remaining_marbles_l3858_385862


namespace jacket_price_reduction_l3858_385874

/-- Calculates the final price of a jacket after two successive price reductions -/
theorem jacket_price_reduction (initial_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  initial_price = 20 ∧ 
  first_reduction = 0.2 ∧ 
  second_reduction = 0.25 →
  initial_price * (1 - first_reduction) * (1 - second_reduction) = 12 :=
by sorry

end jacket_price_reduction_l3858_385874


namespace simplify_fraction_sum_l3858_385859

theorem simplify_fraction_sum (a b c d : ℕ) (h1 : a * d = b * c) (h2 : Nat.gcd a b = 1) :
  a + b = 11 → 75 * d = 200 * c :=
by
  sorry

end simplify_fraction_sum_l3858_385859


namespace distance_difference_l3858_385838

-- Define the dimensions
def street_width : ℕ := 25
def block_length : ℕ := 450
def block_width : ℕ := 350
def alley_width : ℕ := 25

-- Define Sarah's path
def sarah_long_side : ℕ := block_length + alley_width
def sarah_short_side : ℕ := block_width

-- Define Sam's path
def sam_long_side : ℕ := block_length + 2 * street_width
def sam_short_side : ℕ := block_width + 2 * street_width

-- Calculate total distances
def sarah_total : ℕ := 2 * sarah_long_side + 2 * sarah_short_side
def sam_total : ℕ := 2 * sam_long_side + 2 * sam_short_side

-- Theorem to prove
theorem distance_difference :
  sam_total - sarah_total = 150 := by
  sorry

end distance_difference_l3858_385838


namespace racecar_repair_cost_l3858_385860

/-- Proves that the original cost of fixing a racecar was $20,000 given specific conditions --/
theorem racecar_repair_cost 
  (discount_rate : Real) 
  (prize : Real) 
  (prize_keep_rate : Real) 
  (net_profit : Real) :
  discount_rate = 0.2 →
  prize = 70000 →
  prize_keep_rate = 0.9 →
  net_profit = 47000 →
  ∃ (original_cost : Real),
    original_cost = 20000 ∧
    prize * prize_keep_rate - original_cost * (1 - discount_rate) = net_profit :=
by
  sorry

end racecar_repair_cost_l3858_385860


namespace problem_statement_l3858_385845

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -8)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 12)
  (h3 : a * b * c = 1) :
  b / (a + b) + c / (b + c) + a / (c + a) = -8.5 := by
sorry

end problem_statement_l3858_385845


namespace tangent_line_problem_l3858_385866

/-- The problem statement -/
theorem tangent_line_problem (k : ℝ) (P : ℝ × ℝ) (A : ℝ × ℝ) :
  k > 0 →
  P.1 * k + P.2 + 4 = 0 →
  A.1^2 + A.2^2 - 2*A.2 = 0 →
  (∀ Q : ℝ × ℝ, Q.1^2 + Q.2^2 - 2*Q.2 = 0 → 
    (A.1 - P.1)^2 + (A.2 - P.2)^2 ≤ (Q.1 - P.1)^2 + (Q.2 - P.2)^2) →
  (A.1 - P.1)^2 + (A.2 - P.2)^2 = 4 →
  k = 2 := by
sorry

end tangent_line_problem_l3858_385866


namespace triangle_area_l3858_385803

theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  b * Real.cos C = 3 * a * Real.cos B - c * Real.cos B →
  a * c * Real.cos B = 2 →
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 2 := by
sorry

end triangle_area_l3858_385803


namespace wade_tips_theorem_l3858_385869

def tips_per_customer : ℕ := 2
def friday_customers : ℕ := 28
def sunday_customers : ℕ := 36

def saturday_customers : ℕ := 3 * friday_customers

def total_tips : ℕ := tips_per_customer * (friday_customers + saturday_customers + sunday_customers)

theorem wade_tips_theorem : total_tips = 296 := by
  sorry

end wade_tips_theorem_l3858_385869


namespace expand_expression_l3858_385816

theorem expand_expression (x : ℝ) : 5 * (9 * x^3 - 4 * x^2 + 3 * x - 7) = 45 * x^3 - 20 * x^2 + 15 * x - 35 := by
  sorry

end expand_expression_l3858_385816


namespace cory_fruit_arrangements_l3858_385885

/-- The number of ways to arrange fruits over a week -/
def arrangeWeekFruits (apples oranges : ℕ) : ℕ :=
  Nat.factorial (apples + oranges + 1) / (Nat.factorial apples * Nat.factorial oranges)

/-- The number of ways to arrange fruits over a week, excluding banana on first day -/
def arrangeWeekFruitsNoBananaFirst (apples oranges : ℕ) : ℕ :=
  (apples + oranges) * arrangeWeekFruits apples oranges

theorem cory_fruit_arrangements :
  arrangeWeekFruitsNoBananaFirst 4 2 = 90 := by
  sorry

end cory_fruit_arrangements_l3858_385885


namespace power_of_sixteen_five_fourths_l3858_385873

theorem power_of_sixteen_five_fourths : (16 : ℝ) ^ (5/4 : ℝ) = 32 := by sorry

end power_of_sixteen_five_fourths_l3858_385873


namespace wendys_laundry_l3858_385854

theorem wendys_laundry (machine_capacity : ℕ) (num_sweaters : ℕ) (num_loads : ℕ) :
  machine_capacity = 8 →
  num_sweaters = 33 →
  num_loads = 9 →
  num_loads * machine_capacity - num_sweaters = 39 :=
by
  sorry

end wendys_laundry_l3858_385854


namespace pocket_knife_value_l3858_385896

def is_fair_division (n : ℕ) (knife_value : ℕ) : Prop :=
  let total_revenue := n * n
  let elder_share := (total_revenue / 20) * 10
  let younger_share := ((total_revenue / 20) * 10) + (total_revenue % 20) + knife_value
  elder_share = younger_share

theorem pocket_knife_value :
  ∃ (n : ℕ), 
    n > 0 ∧ 
    (n * n) % 20 = 6 ∧ 
    is_fair_division n 2 :=
by sorry

end pocket_knife_value_l3858_385896


namespace students_taking_neither_music_nor_art_l3858_385828

theorem students_taking_neither_music_nor_art 
  (total_students : ℕ) 
  (music_students : ℕ) 
  (art_students : ℕ) 
  (both_students : ℕ) 
  (h1 : total_students = 500) 
  (h2 : music_students = 30) 
  (h3 : art_students = 10) 
  (h4 : both_students = 10) : 
  total_students - (music_students + art_students - both_students) = 460 := by
sorry

end students_taking_neither_music_nor_art_l3858_385828


namespace angle_OA_OC_l3858_385827

def angle_between_vectors (OA OB OC : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem angle_OA_OC (OA OB OC : ℝ × ℝ × ℝ) 
  (h1 : ‖OA‖ = 1)
  (h2 : ‖OB‖ = 2)
  (h3 : Real.cos (angle_between_vectors OA OB OC) = -1/2)
  (h4 : OC = (1/2 : ℝ) • OA + (1/4 : ℝ) • OB) :
  angle_between_vectors OA OC OC = π/3 :=
sorry

end angle_OA_OC_l3858_385827


namespace perfect_square_condition_l3858_385820

theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n^2 - 19*n + 91 = k^2) ↔ (n = 9 ∨ n = 10) :=
sorry

end perfect_square_condition_l3858_385820


namespace choir_competition_score_l3858_385812

/-- Calculates the final score of a choir competition team given their individual scores and weights -/
def final_score (song_content : ℝ) (singing_skills : ℝ) (spirit : ℝ) : ℝ :=
  0.3 * song_content + 0.5 * singing_skills + 0.2 * spirit

/-- Theorem stating that the final score for the given team is 93 -/
theorem choir_competition_score :
  final_score 90 94 95 = 93 := by
  sorry

#eval final_score 90 94 95

end choir_competition_score_l3858_385812


namespace line_point_value_l3858_385861

/-- Given a line with slope 2 passing through (3, 5) and (a, 7), prove a = 4 -/
theorem line_point_value (m : ℝ) (a : ℝ) : 
  m = 2 → -- The line has a slope of 2
  (5 : ℝ) - 5 = m * ((3 : ℝ) - 3) → -- The line passes through (3, 5)
  (7 : ℝ) - 5 = m * (a - 3) → -- The line passes through (a, 7)
  a = 4 := by sorry

end line_point_value_l3858_385861


namespace complex_equation_sum_l3858_385898

theorem complex_equation_sum (x y : ℝ) :
  (↑x + (↑y - 2) * I : ℂ) = 2 / (1 + I) →
  x + y = 2 := by
sorry

end complex_equation_sum_l3858_385898


namespace scientific_notation_proof_l3858_385817

theorem scientific_notation_proof : 
  (192000000 : ℝ) = 1.92 * (10 ^ 8) := by
  sorry

end scientific_notation_proof_l3858_385817


namespace middle_number_proof_l3858_385876

theorem middle_number_proof (a b c d e : ℕ) : 
  ({a, b, c, d, e} : Finset ℕ) = {7, 8, 9, 10, 11} →
  a + b + c = 26 →
  c + d + e = 30 →
  c = 11 := by
  sorry

end middle_number_proof_l3858_385876


namespace propositions_are_true_l3858_385819

-- Define the propositions
def similar_triangles_equal_perimeters : Prop := sorry
def similar_triangles_equal_angles : Prop := sorry
def sqrt_9_not_negative_3 : Prop := sorry
def diameter_bisects_chord : Prop := sorry
def diameter_bisects_arcs : Prop := sorry

-- Theorem to prove
theorem propositions_are_true :
  (similar_triangles_equal_perimeters ∨ similar_triangles_equal_angles) ∧
  sqrt_9_not_negative_3 ∧
  (diameter_bisects_chord ∧ diameter_bisects_arcs) :=
by
  sorry

end propositions_are_true_l3858_385819
