import Mathlib

namespace cards_in_new_deck_l1496_149624

/-- The number of cards in a new deck -/
def cards_per_deck : ℕ := 55

/-- The number of cards that can be torn at once -/
def cards_per_tear : ℕ := 30

/-- The number of times cards are torn per week -/
def tears_per_week : ℕ := 3

/-- The number of decks purchased -/
def decks_purchased : ℕ := 18

/-- The number of weeks the tearing can continue -/
def weeks_of_tearing : ℕ := 11

/-- Theorem stating the number of cards in a new deck -/
theorem cards_in_new_deck : 
  cards_per_deck * decks_purchased = cards_per_tear * tears_per_week * weeks_of_tearing :=
by sorry

end cards_in_new_deck_l1496_149624


namespace only_one_divides_l1496_149682

theorem only_one_divides (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end only_one_divides_l1496_149682


namespace greatest_sum_consecutive_integers_sum_30_satisfies_condition_greatest_sum_is_30_l1496_149695

theorem greatest_sum_consecutive_integers (n : ℤ) : 
  (n - 1) * n * (n + 1) < 1000 → (n - 1) + n + (n + 1) ≤ 30 := by
  sorry

theorem sum_30_satisfies_condition : 
  (9 : ℤ) * 10 * 11 < 1000 ∧ 9 + 10 + 11 = 30 := by
  sorry

theorem greatest_sum_is_30 : 
  ∃ (n : ℤ), (n - 1) * n * (n + 1) < 1000 ∧ 
             (n - 1) + n + (n + 1) = 30 ∧ 
             ∀ (m : ℤ), (m - 1) * m * (m + 1) < 1000 → 
                        (m - 1) + m + (m + 1) ≤ 30 := by
  sorry

end greatest_sum_consecutive_integers_sum_30_satisfies_condition_greatest_sum_is_30_l1496_149695


namespace rectangle_arrangement_l1496_149660

/-- Given 110 identical rectangular sheets where each sheet's length is 10 cm longer than its width,
    and when arranged as in Figure 1 they form a rectangle of length 2750 cm,
    prove that the length of the rectangle formed when arranged as in Figure 2 is 1650 cm. -/
theorem rectangle_arrangement (n : ℕ) (sheet_length sheet_width : ℝ) 
  (h1 : n = 110)
  (h2 : sheet_length = sheet_width + 10)
  (h3 : n * sheet_length = 2750) :
  n * sheet_width = 1650 := by
  sorry

#check rectangle_arrangement

end rectangle_arrangement_l1496_149660


namespace hexagonal_pyramid_base_neq_slant_l1496_149613

/-- A regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  /-- The edge length of the base -/
  baseEdge : ℝ
  /-- The slant height of the pyramid -/
  slantHeight : ℝ
  /-- The apex angle of each lateral face -/
  apexAngle : ℝ
  /-- Condition: baseEdge and slantHeight are positive -/
  baseEdge_pos : baseEdge > 0
  slantHeight_pos : slantHeight > 0
  /-- Condition: The apex angle is determined by the baseEdge and slantHeight -/
  apexAngle_eq : apexAngle = 2 * Real.arcsin (baseEdge / (2 * slantHeight))

/-- Theorem: It's impossible for a regular hexagonal pyramid to have its base edge length equal to its slant height -/
theorem hexagonal_pyramid_base_neq_slant (p : RegularHexagonalPyramid) : 
  p.baseEdge ≠ p.slantHeight := by
  sorry

end hexagonal_pyramid_base_neq_slant_l1496_149613


namespace subtraction_proof_l1496_149689

theorem subtraction_proof : 6236 - 797 = 5439 := by sorry

end subtraction_proof_l1496_149689


namespace chess_club_team_probability_l1496_149673

def total_members : ℕ := 20
def num_boys : ℕ := 12
def num_girls : ℕ := 8
def team_size : ℕ := 4

theorem chess_club_team_probability :
  let total_combinations := Nat.choose total_members team_size
  let valid_combinations := 
    Nat.choose num_boys 2 * Nat.choose num_girls 2 + 
    Nat.choose num_boys 3 * Nat.choose num_girls 1 + 
    Nat.choose num_boys 4 * Nat.choose num_girls 0
  (valid_combinations : ℚ) / total_combinations = 4103 / 4845 := by
  sorry

end chess_club_team_probability_l1496_149673


namespace arithmetic_sequence_problem_l1496_149634

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a)
    (h_a1 : a 1 = 1/3)
    (h_sum : a 2 + a 5 = 4)
    (h_an : ∃ n : ℕ, a n = 33) :
  ∃ n : ℕ, a n = 33 ∧ n = 50 := by
sorry

end arithmetic_sequence_problem_l1496_149634


namespace total_boxes_in_cases_l1496_149637

/-- The number of cases Jenny needs to deliver -/
def num_cases : ℕ := 3

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 8

/-- Theorem: The total number of boxes in the cases is 24 -/
theorem total_boxes_in_cases : num_cases * boxes_per_case = 24 := by
  sorry

end total_boxes_in_cases_l1496_149637


namespace tangent_lines_range_l1496_149663

/-- The range of k values for which two tangent lines can be drawn from (1, 2) to the circle x^2 + y^2 + kx + 2y + k^2 - 15 = 0 -/
theorem tangent_lines_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + k*x + 2*y + k^2 - 15 = 0 ∧ 
   ∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧ 
     (∀ (x' y' : ℝ), (y' - 2 = m₁ * (x' - 1) ∨ y' - 2 = m₂ * (x' - 1)) →
       (x'^2 + y'^2 + k*x' + 2*y' + k^2 - 15 = 0 → x' = 1 ∧ y' = 2))) ↔ 
  (k ∈ Set.Ioo (-8 * Real.sqrt 3 / 3) (-3) ∪ Set.Ioo 2 (8 * Real.sqrt 3 / 3)) :=
by sorry


end tangent_lines_range_l1496_149663


namespace subtract_from_zero_l1496_149627

theorem subtract_from_zero (x : ℚ) : 0 - x = -x := by sorry

end subtract_from_zero_l1496_149627


namespace train_distance_problem_l1496_149629

/-- The distance between two points P and Q, given the conditions of two trains traveling towards each other --/
theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 50) (h2 : v2 = 40) (h3 : d = 100) : 
  v1 * (d / (v1 - v2) + d / v2) = 900 := by
  sorry

end train_distance_problem_l1496_149629


namespace employee_salary_proof_l1496_149609

def total_salary : ℝ := 572
def m_salary_ratio : ℝ := 1.2

theorem employee_salary_proof (n_salary : ℝ) 
  (h1 : n_salary + m_salary_ratio * n_salary = total_salary) :
  n_salary = 260 := by
  sorry

end employee_salary_proof_l1496_149609


namespace union_of_A_and_B_l1496_149608

def A : Set ℤ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℤ := {-2, -1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1, 2} := by sorry

end union_of_A_and_B_l1496_149608


namespace base_number_proof_l1496_149687

theorem base_number_proof (x n b : ℝ) 
  (h1 : n = x^(1/4))
  (h2 : n^b = 16)
  (h3 : b = 16.000000000000004) :
  x = 2 := by
sorry

end base_number_proof_l1496_149687


namespace searchlight_probability_l1496_149645

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℝ := 2

/-- The time in seconds for which the man needs to stay in the dark -/
def dark_time : ℝ := 5

/-- The number of seconds in a minute -/
def seconds_per_minute : ℝ := 60

theorem searchlight_probability :
  let time_per_revolution := seconds_per_minute / revolutions_per_minute
  (dark_time / time_per_revolution : ℝ) = 1 / 6 := by sorry

end searchlight_probability_l1496_149645


namespace four_foldable_positions_l1496_149625

/-- Represents a position where an additional square can be attached --/
inductive Position
| Top
| TopRight
| Right
| BottomRight
| Bottom
| BottomLeft
| Left
| TopLeft
| CenterTop
| CenterRight
| CenterBottom
| CenterLeft

/-- Represents the cross-shaped polygon --/
structure CrossPolygon :=
  (squares : Fin 5 → Unit)

/-- Represents the resulting polygon after attaching an additional square --/
structure ResultingPolygon :=
  (base : CrossPolygon)
  (additional : Position)

/-- Predicate to check if a resulting polygon can be folded into a cube with one face missing --/
def can_fold_to_cube (p : ResultingPolygon) : Prop :=
  sorry

/-- The main theorem --/
theorem four_foldable_positions :
  ∃ (valid_positions : Finset Position),
    (valid_positions.card = 4) ∧
    (∀ p : Position, p ∈ valid_positions ↔ 
      can_fold_to_cube ⟨CrossPolygon.mk (λ _ => Unit.unit), p⟩) :=
  sorry

end four_foldable_positions_l1496_149625


namespace cookie_sale_charity_share_l1496_149644

/-- Calculates the amount each charity receives when John sells cookies and splits the profit. -/
theorem cookie_sale_charity_share :
  let dozen : ℕ := 6
  let cookies_per_dozen : ℕ := 12
  let total_cookies : ℕ := dozen * cookies_per_dozen
  let price_per_cookie : ℚ := 3/2
  let cost_per_cookie : ℚ := 1/4
  let total_revenue : ℚ := total_cookies * price_per_cookie
  let total_cost : ℚ := total_cookies * cost_per_cookie
  let profit : ℚ := total_revenue - total_cost
  let num_charities : ℕ := 2
  let charity_share : ℚ := profit / num_charities
  charity_share = 45
:= by sorry

end cookie_sale_charity_share_l1496_149644


namespace binomial_20_4_l1496_149630

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end binomial_20_4_l1496_149630


namespace initial_bananas_per_child_l1496_149672

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) :
  total_children = 740 →
  absent_children = 370 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ),
    initial_bananas * total_children = (initial_bananas + extra_bananas) * (total_children - absent_children) ∧
    initial_bananas = 2 :=
by
  sorry

end initial_bananas_per_child_l1496_149672


namespace color_change_probability_l1496_149621

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationDuration (cycle : TrafficLightCycle) (observationInterval : ℕ) : ℕ :=
  3 * observationInterval  -- 3 color changes per cycle

/-- Theorem: The probability of observing a color change is 12/85 -/
theorem color_change_probability (cycle : TrafficLightCycle) 
  (h1 : cycle.green = 45)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 35)
  (observationInterval : ℕ)
  (h4 : observationInterval = 4) :
  (changeObservationDuration cycle observationInterval : ℚ) / 
  (cycleDuration cycle : ℚ) = 12 / 85 := by
  sorry

end color_change_probability_l1496_149621


namespace bankers_discount_calculation_l1496_149654

/-- Banker's discount calculation -/
theorem bankers_discount_calculation 
  (present_worth : ℚ) 
  (true_discount : ℚ) 
  (h1 : present_worth = 400)
  (h2 : true_discount = 20) : 
  (true_discount * (present_worth + true_discount)) / present_worth = 21 :=
by
  sorry

#check bankers_discount_calculation

end bankers_discount_calculation_l1496_149654


namespace shaded_area_square_minus_semicircles_l1496_149622

/-- The area of a square with side length 14 cm minus the area of two semicircles 
    with diameters equal to the side length of the square is equal to 196 - 49π cm². -/
theorem shaded_area_square_minus_semicircles : 
  let side_length : ℝ := 14
  let square_area : ℝ := side_length ^ 2
  let semicircle_radius : ℝ := side_length / 2
  let semicircles_area : ℝ := π * semicircle_radius ^ 2
  square_area - semicircles_area = 196 - 49 * π := by sorry

end shaded_area_square_minus_semicircles_l1496_149622


namespace not_p_or_q_false_implies_p_and_q_false_l1496_149641

theorem not_p_or_q_false_implies_p_and_q_false (p q : Prop) :
  (¬(¬p ∨ q)) → ¬(p ∧ q) := by
  sorry

end not_p_or_q_false_implies_p_and_q_false_l1496_149641


namespace largest_last_digit_is_two_l1496_149668

/-- A function that checks if a two-digit number is divisible by 17 or 23 -/
def isDivisibleBy17Or23 (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n % 17 = 0 ∨ n % 23 = 0)

/-- A function that represents a valid digit string according to the problem conditions -/
def isValidDigitString (s : List Nat) : Prop :=
  s.length = 1001 ∧
  s.head? = some 2 ∧
  ∀ i, i < 1000 → isDivisibleBy17Or23 (s[i]! * 10 + s[i+1]!)

/-- The theorem stating that the largest possible last digit is 2 -/
theorem largest_last_digit_is_two (s : List Nat) (h : isValidDigitString s) :
  s[1000]! ≤ 2 :=
sorry

end largest_last_digit_is_two_l1496_149668


namespace magnitude_of_complex_fraction_l1496_149658

theorem magnitude_of_complex_fraction : Complex.abs (1 / (1 - 2 * Complex.I)) = Real.sqrt 5 / 5 := by
  sorry

end magnitude_of_complex_fraction_l1496_149658


namespace people_left_line_l1496_149628

theorem people_left_line (initial : ℕ) (joined : ℕ) (final : ℕ) (left : ℕ) : 
  initial = 9 → joined = 3 → final = 6 → initial - left + joined = final → left = 6 := by
  sorry

end people_left_line_l1496_149628


namespace abs_value_of_z_l1496_149662

/-- Given a complex number z = ((1+i)/(1-i))^2, prove that its absolute value |z| is equal to 1. -/
theorem abs_value_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.abs ((((1:ℂ) + i) / ((1:ℂ) - i))^2) = 1 := by sorry

end abs_value_of_z_l1496_149662


namespace x_range_l1496_149659

theorem x_range (x : ℝ) (h1 : 2 ≤ |x - 5| ∧ |x - 5| ≤ 10) (h2 : x > 0) :
  (0 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 15) := by
  sorry

end x_range_l1496_149659


namespace area_enclosed_is_nine_halves_l1496_149671

-- Define the constant term a
def a : ℝ := 3

-- Define the functions for the line and curve
def f (x : ℝ) : ℝ := a * x
def g (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem area_enclosed_is_nine_halves :
  ∫ x in (0)..(3), (f x - g x) = 9/2 := by
  sorry

end area_enclosed_is_nine_halves_l1496_149671


namespace total_cost_is_360_l1496_149618

def calculate_total_cost (sale_prices : List ℝ) (discounts : List ℝ) 
  (installation_fee : ℝ) (disposal_fee : ℝ) : ℝ :=
  let discounted_prices := List.zipWith (·-·) sale_prices discounts
  let with_installation := List.map (·+installation_fee) discounted_prices
  let total_per_tire := List.map (·+disposal_fee) with_installation
  List.sum total_per_tire

theorem total_cost_is_360 :
  let sale_prices := [75, 90, 120, 150]
  let discounts := [20, 30, 45, 60]
  let installation_fee := 15
  let disposal_fee := 5
  calculate_total_cost sale_prices discounts installation_fee disposal_fee = 360 := by
  sorry

end total_cost_is_360_l1496_149618


namespace coefficient_x_squared_in_expansion_coefficient_x_squared_is_three_l1496_149696

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 4).sum (λ k => (Nat.choose 3 k) * x^k) = 1 + 3*x + 3*x^2 + x^3 := by
  sorry

theorem coefficient_x_squared_is_three : 
  (Finset.range 4).sum (λ k => (Nat.choose 3 k) * (if k = 2 then 1 else 0)) = 3 := by
  sorry

end coefficient_x_squared_in_expansion_coefficient_x_squared_is_three_l1496_149696


namespace intersection_sum_l1496_149602

/-- Given two lines y = mx + 4 and y = 3x + b intersecting at (6, 10), prove b + m = -7 -/
theorem intersection_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + 4 ↔ y = 3 * x + b) → 
  (6 : ℝ) * m + 4 = 10 → 
  3 * 6 + b = 10 → 
  b + m = -7 := by sorry

end intersection_sum_l1496_149602


namespace volleyball_lineup_count_l1496_149681

/-- The number of players in the volleyball team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def num_starters : ℕ := 7

/-- The number of ways to choose starters with the given conditions -/
def valid_lineups : ℕ := Nat.choose total_players num_starters - Nat.choose (total_players - num_quadruplets) (num_starters - num_quadruplets)

theorem volleyball_lineup_count :
  valid_lineups = 11220 :=
sorry

end volleyball_lineup_count_l1496_149681


namespace arithmetic_segments_form_quadrilateral_l1496_149692

/-- Four segments in an arithmetic sequence with total length 3 can form a quadrilateral -/
theorem arithmetic_segments_form_quadrilateral :
  ∀ (a d : ℝ),
  a > 0 ∧ d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) = 3 →
  (a + (a + d) + (a + 2*d) > a + 3*d) ∧
  (a + (a + d) + (a + 3*d) > a + 2*d) ∧
  (a + (a + 2*d) + (a + 3*d) > a + d) ∧
  ((a + d) + (a + 2*d) + (a + 3*d) > a) :=
by
  sorry


end arithmetic_segments_form_quadrilateral_l1496_149692


namespace kozlov_inequality_l1496_149666

theorem kozlov_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : 
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 
  2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end kozlov_inequality_l1496_149666


namespace complex_equation_solution_l1496_149615

theorem complex_equation_solution (x y : ℝ) :
  (x + y - 3 : ℂ) + (x - 4 : ℂ) * I = 0 → x = 4 ∧ y = -1 := by
  sorry

end complex_equation_solution_l1496_149615


namespace owls_joined_l1496_149650

def initial_owls : ℕ := 3
def final_owls : ℕ := 5

theorem owls_joined : final_owls - initial_owls = 2 := by
  sorry

end owls_joined_l1496_149650


namespace fold_symmetry_l1496_149684

/-- A fold on a graph paper is represented by its axis of symmetry -/
structure Fold :=
  (axis : ℝ)

/-- A point on a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Determine if two points coincide after a fold -/
def coincide (p1 p2 : Point) (f : Fold) : Prop :=
  (p1.x + p2.x) / 2 = f.axis ∧ p1.y = p2.y

/-- Find the symmetric point of a given point with respect to a fold -/
def symmetric_point (p : Point) (f : Fold) : Point :=
  { x := 2 * f.axis - p.x, y := p.y }

theorem fold_symmetry (f : Fold) (p1 p2 p3 : Point) :
  coincide p1 p2 f →
  f.axis = 3 →
  p3 = { x := -4, y := 1 } →
  symmetric_point p3 f = { x := 10, y := 1 } := by
  sorry

#check fold_symmetry

end fold_symmetry_l1496_149684


namespace triangle_inequality_l1496_149661

theorem triangle_inequality (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_a_geq_b : a ≥ b)
  (h_a_geq_c : a ≥ c)
  (h_sum1 : a + b - c > 0)
  (h_sum2 : b + c - a > 0) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
by sorry

end triangle_inequality_l1496_149661


namespace monotonicity_for_a_2_non_monotonicity_condition_l1496_149686

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1/2 * x^2 - (1 + a) * x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := a / x + x - (1 + a)

theorem monotonicity_for_a_2 :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f 2 x₁ < f 2 x₂ ∧
  ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f 2 x₁ > f 2 x₂ ∧
  ∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f 2 x₁ < f 2 x₂ :=
sorry

theorem non_monotonicity_condition :
  ∀ a, (∃ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ f a x₁ < f a x₂ ∧
       ∃ y₁ y₂, 1 < y₁ ∧ y₁ < y₂ ∧ y₂ < 2 ∧ f a y₁ > f a y₂) ↔
  1 < a ∧ a < 2 :=
sorry

end monotonicity_for_a_2_non_monotonicity_condition_l1496_149686


namespace factorization_equality_l1496_149604

theorem factorization_equality (x y : ℝ) : 
  x^2 - 2*x*y + y^2 - 1 = (x - y + 1) * (x - y - 1) := by
  sorry

end factorization_equality_l1496_149604


namespace odd_integers_count_odd_integers_three_different_digits_count_l1496_149639

theorem odd_integers_count : ℕ := by
  -- Define the range of integers
  let lower_bound : ℕ := 2000
  let upper_bound : ℕ := 3000

  -- Define the set of possible odd units digits
  let odd_units : Finset ℕ := {1, 3, 5, 7, 9}

  -- Define the count of choices for each digit position
  let thousands_choices : ℕ := 1  -- Always 2
  let hundreds_choices : ℕ := 8   -- Excluding 2 and the chosen units digit
  let tens_choices : ℕ := 7       -- Excluding 2, hundreds digit, and units digit
  let units_choices : ℕ := Finset.card odd_units

  -- Calculate the total count
  let total_count : ℕ := thousands_choices * hundreds_choices * tens_choices * units_choices

  -- Prove that the count equals 280
  sorry

-- The theorem statement
theorem odd_integers_three_different_digits_count :
  (odd_integers_count : ℕ) = 280 := by sorry

end odd_integers_count_odd_integers_three_different_digits_count_l1496_149639


namespace find_number_l1496_149605

theorem find_number (x : ℝ) : x + 1.35 + 0.123 = 1.794 → x = 0.321 := by
  sorry

end find_number_l1496_149605


namespace dog_weight_is_ten_l1496_149674

/-- Represents the weights of a kitten, rabbit, and dog satisfying certain conditions -/
structure AnimalWeights where
  kitten : ℝ
  rabbit : ℝ
  dog : ℝ
  total_weight : kitten + rabbit + dog = 30
  kitten_rabbit_twice_dog : kitten + rabbit = 2 * dog
  kitten_dog_equals_rabbit : kitten + dog = rabbit

/-- The weight of the dog in the AnimalWeights structure is 10 pounds -/
theorem dog_weight_is_ten (w : AnimalWeights) : w.dog = 10 := by
  sorry

end dog_weight_is_ten_l1496_149674


namespace multiples_of_five_l1496_149675

/-- The largest number n such that there are 999 positive integers 
    between 5 and n (inclusive) that are multiples of 5 is 4995. -/
theorem multiples_of_five (n : ℕ) : 
  (∃ (k : ℕ), k = 999 ∧ 
    (∀ m : ℕ, 5 ≤ m ∧ m ≤ n ∧ m % 5 = 0 ↔ m ∈ Finset.range k)) →
  n = 4995 := by
sorry

end multiples_of_five_l1496_149675


namespace contractor_absent_days_l1496_149652

theorem contractor_absent_days 
  (total_days : ℕ) 
  (pay_per_day : ℚ) 
  (fine_per_day : ℚ) 
  (total_received : ℚ) 
  (h1 : total_days = 30)
  (h2 : pay_per_day = 25)
  (h3 : fine_per_day = 7.5)
  (h4 : total_received = 425) :
  ∃ (absent_days : ℕ), 
    absent_days = 10 ∧ 
    (total_days - absent_days) * pay_per_day - absent_days * fine_per_day = total_received :=
by sorry

end contractor_absent_days_l1496_149652


namespace parallelogram_condition_l1496_149655

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_a_gt_b : a > b

/-- Checks if a point lies on the unit circle -/
def onUnitCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 = 1

/-- Checks if a point lies on the given ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The main theorem stating the condition for the parallelogram property -/
theorem parallelogram_condition (e : Ellipse) :
  (∀ p : Point, onEllipse p e → 
    ∃ q r s : Point, 
      onEllipse q e ∧ onEllipse r e ∧ onEllipse s e ∧
      onUnitCircle q ∧ onUnitCircle s ∧
      -- Additional conditions for parallelogram property would be defined here
      True) → 
  1 / e.a^2 + 1 / e.b^2 = 1 :=
sorry

end parallelogram_condition_l1496_149655


namespace arithmetic_progression_condition_l1496_149669

theorem arithmetic_progression_condition (a b c : ℝ) : 
  (∃ d : ℝ, ∃ n k p : ℤ, b = a + d * (k - n) ∧ c = a + d * (p - n)) →
  (∃ A B : ℤ, (b - a) / (c - b) = A / B) :=
sorry

end arithmetic_progression_condition_l1496_149669


namespace circle_to_octagon_area_ratio_l1496_149642

/-- The ratio of the area of a circle inscribed in a regular octagon to the area of the octagon,
    where the circle's radius reaches the midpoint of each octagon side. -/
theorem circle_to_octagon_area_ratio : ∃ (a b : ℕ), 
  (a : ℝ).sqrt / b * π = π / (4 * (1 + Real.sqrt 2)) ∧ a * b = 16 := by
  sorry

end circle_to_octagon_area_ratio_l1496_149642


namespace not_perfect_square_for_prime_l1496_149640

theorem not_perfect_square_for_prime (p : ℕ) (h : Prime p) : ¬ ∃ t : ℤ, (7 * p + 3^p - 4 : ℤ) = t^2 := by
  sorry

end not_perfect_square_for_prime_l1496_149640


namespace correct_calculation_l1496_149699

theorem correct_calculation : 
  (2 * Real.sqrt 5 + 3 * Real.sqrt 5 = 5 * Real.sqrt 5) ∧ 
  (Real.sqrt 8 ≠ 2) ∧ 
  (Real.sqrt ((-3)^2) ≠ -3) ∧ 
  ((Real.sqrt 2 + 1)^2 ≠ 3) :=
by sorry

end correct_calculation_l1496_149699


namespace sequence_formula_l1496_149626

def sequence_a (n : ℕ) : ℕ := 2^n - 1

def sum_S : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2 * sum_S n + n + 1

theorem sequence_formula (n : ℕ) :
  n > 0 →
  sequence_a 1 = 1 ∧
  (∀ k, k > 0 → sum_S (k + 1) = 2 * sum_S k + k + 1) →
  sequence_a n = sum_S n - sum_S (n - 1) :=
sorry

end sequence_formula_l1496_149626


namespace expand_and_simplify_l1496_149665

theorem expand_and_simplify (x : ℝ) : (x + 2) * (x - 2) - x * (x + 1) = -x - 4 := by
  sorry

end expand_and_simplify_l1496_149665


namespace intersection_symmetry_l1496_149676

/-- Given a line y = kx that intersects the circle (x-1)^2 + y^2 = 1 at two points
    symmetric with respect to the line x - y + b = 0, prove that k = -1 and b = -1 -/
theorem intersection_symmetry (k b : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    -- The line intersects the circle at two points
    y₁ = k * x₁ ∧ (x₁ - 1)^2 + y₁^2 = 1 ∧
    y₂ = k * x₂ ∧ (x₂ - 1)^2 + y₂^2 = 1 ∧
    -- The points are distinct
    (x₁, y₁) ≠ (x₂, y₂) ∧
    -- The points are symmetric with respect to x - y + b = 0
    ∃ x₀ y₀ : ℝ, x₀ - y₀ + b = 0 ∧
    x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2) →
  k = -1 ∧ b = -1 := by
sorry

end intersection_symmetry_l1496_149676


namespace absolute_value_equation_simplification_l1496_149614

theorem absolute_value_equation_simplification (a b c : ℝ) :
  (∀ x : ℝ, |5 * x - 4| + a ≠ 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |4 * x₁ - 3| + b = 0 ∧ |4 * x₂ - 3| + b = 0) →
  (∃! x : ℝ, |3 * x - 2| + c = 0) →
  |a - c| + |c - b| - |a - b| = 0 := by
sorry

end absolute_value_equation_simplification_l1496_149614


namespace smallest_common_multiple_of_6_and_9_l1496_149670

theorem smallest_common_multiple_of_6_and_9 : 
  ∃ n : ℕ+, (∀ m : ℕ+, 6 ∣ m ∧ 9 ∣ m → n ≤ m) ∧ 6 ∣ n ∧ 9 ∣ n := by
  sorry

end smallest_common_multiple_of_6_and_9_l1496_149670


namespace nth_term_is_4021_l1496_149603

/-- An arithmetic sequence with given first three terms -/
structure ArithmeticSequence (x : ℝ) where
  first_term : ℝ := 3 * x - 4
  second_term : ℝ := 6 * x - 17
  third_term : ℝ := 4 * x + 5
  is_arithmetic : second_term - first_term = third_term - second_term

/-- The nth term of the arithmetic sequence -/
def nth_term (seq : ArithmeticSequence x) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * (seq.second_term - seq.first_term)

theorem nth_term_is_4021 (x : ℝ) (seq : ArithmeticSequence x) :
  ∃ n : ℕ, nth_term seq n = 4021 ∧ n = 502 := by
  sorry

end nth_term_is_4021_l1496_149603


namespace fox_can_eat_80_fox_cannot_eat_65_l1496_149607

/-- Represents a distribution of candies into three piles -/
structure CandyDistribution :=
  (pile1 pile2 pile3 : ℕ)
  (sum_eq_100 : pile1 + pile2 + pile3 = 100)

/-- Calculates the number of candies the fox eats given a distribution -/
def fox_candies (d : CandyDistribution) : ℕ :=
  if d.pile1 = d.pile2 ∨ d.pile1 = d.pile3 ∨ d.pile2 = d.pile3
  then max d.pile1 (max d.pile2 d.pile3)
  else d.pile1 + d.pile2 + d.pile3 - 2 * min d.pile1 (min d.pile2 d.pile3)

theorem fox_can_eat_80 : ∃ d : CandyDistribution, fox_candies d = 80 := by
  sorry

theorem fox_cannot_eat_65 : ¬ ∃ d : CandyDistribution, fox_candies d = 65 := by
  sorry

end fox_can_eat_80_fox_cannot_eat_65_l1496_149607


namespace infinite_diamond_2005_l1496_149610

/-- A number is diamond 2005 if it has the form ...ab999...99999cd... -/
def is_diamond_2005 (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ) (k m : ℕ), n = a * 10^(k+m+4) + b * 10^(k+m+3) + 999 * 10^m + c * 10 + d

/-- A sequence {a_n} is bounded by C*n if for all n, a_n < C*n -/
def is_bounded_by_linear (a : ℕ → ℕ) (C : ℝ) : Prop :=
  ∀ n, (a n : ℝ) < C * n

/-- A sequence {a_n} is increasing if for all n, a_n <= a_(n+1) -/
def is_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n, a n ≤ a (n + 1)

/-- Main theorem: An increasing sequence bounded by C*n contains infinitely many diamond 2005 numbers -/
theorem infinite_diamond_2005 (a : ℕ → ℕ) (C : ℝ) 
  (h_bound : is_bounded_by_linear a C) 
  (h_incr : is_increasing a) : 
  ∀ m : ℕ, ∃ n > m, is_diamond_2005 (a n) :=
sorry

end infinite_diamond_2005_l1496_149610


namespace water_pollution_scientific_notation_l1496_149693

/-- The amount of water polluted by a button-sized waste battery in liters -/
def water_pollution : ℕ := 600000

/-- Scientific notation representation of water_pollution -/
def scientific_notation : ℝ := 6 * (10 ^ 5)

theorem water_pollution_scientific_notation :
  (water_pollution : ℝ) = scientific_notation := by
  sorry

end water_pollution_scientific_notation_l1496_149693


namespace max_vouchers_for_680_yuan_l1496_149616

/-- Represents the shopping voucher system with a given initial cash amount -/
structure VoucherSystem where
  initial_cash : ℕ
  voucher_rate : ℚ

/-- Calculates the maximum total vouchers that can be received -/
def max_vouchers (system : VoucherSystem) : ℕ :=
  sorry

/-- The theorem stating the maximum vouchers for the given problem -/
theorem max_vouchers_for_680_yuan :
  let system : VoucherSystem := { initial_cash := 680, voucher_rate := 1/5 }
  max_vouchers system = 160 := by
  sorry

end max_vouchers_for_680_yuan_l1496_149616


namespace distance_calculation_l1496_149649

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 94

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time Maxwell walks before meeting Brad, in hours -/
def maxwell_time : ℝ := 10

/-- Time difference between Maxwell's start and Brad's start, in hours -/
def time_difference : ℝ := 1

theorem distance_calculation :
  distance_between_homes = 
    maxwell_speed * maxwell_time + 
    brad_speed * (maxwell_time - time_difference) :=
by sorry

end distance_calculation_l1496_149649


namespace no_prime_multiples_of_ten_in_range_l1496_149631

theorem no_prime_multiples_of_ten_in_range : 
  ¬ ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 10000 ∧ 10 ∣ n ∧ Nat.Prime n ∧ n > 10 := by
  sorry

end no_prime_multiples_of_ten_in_range_l1496_149631


namespace circles_intersect_l1496_149653

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 8*p.2 - 8 = 0}

/-- Circle C₂ with equation x² + y² - 4x - 5 = 0 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 5 = 0}

/-- The center of circle C₁ -/
def center_C₁ : ℝ × ℝ := (-1, -4)

/-- The radius of circle C₁ -/
def radius_C₁ : ℝ := 5

/-- The center of circle C₂ -/
def center_C₂ : ℝ × ℝ := (2, 0)

/-- The radius of circle C₂ -/
def radius_C₂ : ℝ := 3

/-- Theorem stating that circles C₁ and C₂ are intersecting -/
theorem circles_intersect : ∃ p : ℝ × ℝ, p ∈ C₁ ∩ C₂ := by sorry

end circles_intersect_l1496_149653


namespace number_of_ferries_divisible_by_four_l1496_149633

/-- Represents a ferry route between two points across a lake. -/
structure FerryRoute where
  /-- Time interval between ferry departures -/
  departureInterval : ℕ
  /-- Time taken to cross the lake -/
  crossingTime : ℕ
  /-- Number of ferries arriving during docking time -/
  arrivingFerries : ℕ

/-- Theorem stating that the number of ferries on a route with given conditions is divisible by 4 -/
theorem number_of_ferries_divisible_by_four (route : FerryRoute) 
  (h1 : route.crossingTime = route.arrivingFerries * route.departureInterval)
  (h2 : route.crossingTime > 0) : 
  ∃ (n : ℕ), (4 * route.crossingTime) / route.departureInterval = 4 * n := by
  sorry


end number_of_ferries_divisible_by_four_l1496_149633


namespace trig_identity_l1496_149679

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_identity_l1496_149679


namespace comic_books_bought_correct_comic_books_bought_l1496_149646

theorem comic_books_bought (initial : ℕ) (current : ℕ) : ℕ :=
  let sold := initial / 2
  let remaining := initial - sold
  let bought := current - remaining
  bought

theorem correct_comic_books_bought :
  comic_books_bought 22 17 = 6 := by
  sorry

end comic_books_bought_correct_comic_books_bought_l1496_149646


namespace acute_triangle_trig_ranges_l1496_149697

variable (B C : Real)

theorem acute_triangle_trig_ranges 
  (acute : 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (angle_sum : B + C = π/3) :
  let A : Real := π/3
  (((3 + Real.sqrt 3) / 2 < Real.sin A + Real.sin B + Real.sin C) ∧ 
   (Real.sin A + Real.sin B + Real.sin C ≤ (6 + Real.sqrt 3) / 2)) ∧
  ((0 < Real.sin A * Real.sin B * Real.sin C) ∧ 
   (Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8)) := by
  sorry

end acute_triangle_trig_ranges_l1496_149697


namespace tim_nickels_count_l1496_149690

/-- The number of nickels Tim had initially -/
def initial_nickels : ℕ := 9

/-- The number of nickels Tim received from his dad -/
def received_nickels : ℕ := 3

/-- The total number of nickels Tim has after receiving coins from his dad -/
def total_nickels : ℕ := initial_nickels + received_nickels

theorem tim_nickels_count : total_nickels = 12 := by
  sorry

end tim_nickels_count_l1496_149690


namespace rectangle_side_greater_than_twelve_l1496_149657

theorem rectangle_side_greater_than_twelve 
  (a b : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : a * b = 3 * (2 * a + 2 * b)) : 
  a > 12 ∨ b > 12 := by
sorry

end rectangle_side_greater_than_twelve_l1496_149657


namespace student_average_grade_l1496_149656

theorem student_average_grade 
  (courses_last_year : ℕ)
  (courses_year_before : ℕ)
  (avg_grade_year_before : ℚ)
  (avg_grade_two_years : ℚ)
  (h1 : courses_last_year = 6)
  (h2 : courses_year_before = 5)
  (h3 : avg_grade_year_before = 50)
  (h4 : avg_grade_two_years = 77)
  : ∃ (avg_grade_last_year : ℚ), avg_grade_last_year = 99.5 := by
  sorry

end student_average_grade_l1496_149656


namespace women_in_first_group_l1496_149623

/-- The number of women in the first group -/
def first_group : ℕ := 4

/-- The length of cloth colored by the first group in 2 days -/
def cloth_length_first_group : ℕ := 48

/-- The number of days taken by the first group to color the cloth -/
def days_first_group : ℕ := 2

/-- The number of women in the second group -/
def second_group : ℕ := 6

/-- The length of cloth colored by the second group in 1 day -/
def cloth_length_second_group : ℕ := 36

/-- The number of days taken by the second group to color the cloth -/
def days_second_group : ℕ := 1

theorem women_in_first_group : 
  first_group * cloth_length_second_group * days_first_group = 
  second_group * cloth_length_first_group * days_second_group :=
by sorry

end women_in_first_group_l1496_149623


namespace divisibility_property_l1496_149678

theorem divisibility_property (y : ℕ) (h : y ≠ 0) :
  (y - 1) ∣ (y^(y^2) - 2*y^(y+1) + 1) :=
by sorry

end divisibility_property_l1496_149678


namespace evaluate_expression_l1496_149651

theorem evaluate_expression : (0.5^4 / 0.05^3) = 500 := by sorry

end evaluate_expression_l1496_149651


namespace nested_arithmetic_expression_l1496_149680

theorem nested_arithmetic_expression : 1 - (2 - (3 - 4 - (5 - 6))) = -1 := by
  sorry

end nested_arithmetic_expression_l1496_149680


namespace M_is_range_of_f_l1496_149619

def M : Set ℝ := {y | ∃ x, y = x^2}

def f (x : ℝ) : ℝ := x^2

theorem M_is_range_of_f : M = Set.range f := by
  sorry

end M_is_range_of_f_l1496_149619


namespace square_roots_of_four_l1496_149688

theorem square_roots_of_four :
  {y : ℝ | y ^ 2 = 4} = {2, -2} := by sorry

end square_roots_of_four_l1496_149688


namespace office_employees_l1496_149667

theorem office_employees (total : ℝ) 
  (h1 : 0.65 * total = total * (1 - 0.35))  -- 65% of total are males
  (h2 : 0.25 * (0.65 * total) = (0.65 * total) * (1 - 0.75))  -- 25% of males are at least 50
  (h3 : 0.75 * (0.65 * total) = 3120)  -- number of males below 50
  : total = 6400 := by
sorry

end office_employees_l1496_149667


namespace unique_special_parallelogram_l1496_149611

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  B : ℤ × ℤ
  D : ℤ × ℤ
  C : ℚ × ℚ
  area_eq : abs (B.1 * D.2 + D.1 * C.2 + C.1 * 0 - (B.2 * D.1 + D.2 * C.1 + C.2 * 0)) / 2 = 2000000
  B_on_y_eq_x : B.2 = B.1
  D_on_y_eq_2x : D.2 = 2 * D.1
  C_on_y_eq_3x : C.2 = 3 * C.1
  first_quadrant : 0 < B.1 ∧ 0 < B.2 ∧ 0 < D.1 ∧ 0 < D.2 ∧ 0 < C.1 ∧ 0 < C.2
  parallelogram_condition : C.1 = B.1 + D.1 ∧ C.2 = B.2 + D.2

/-- There exists exactly one special parallelogram -/
theorem unique_special_parallelogram : ∃! p : SpecialParallelogram, True :=
  sorry

end unique_special_parallelogram_l1496_149611


namespace tree_cutting_and_planting_l1496_149683

theorem tree_cutting_and_planting (initial_trees : ℕ) : 
  (initial_trees : ℝ) - 0.2 * initial_trees + 5 * (0.2 * initial_trees) = 720 →
  initial_trees = 400 := by
sorry

end tree_cutting_and_planting_l1496_149683


namespace age_ratio_proof_l1496_149698

/-- Given that B's current age is 42 years and A is 12 years older than B,
    prove that the ratio of A's age in 10 years to B's age 10 years ago is 2:1 -/
theorem age_ratio_proof (B_current : ℕ) (A_current : ℕ) : 
  B_current = 42 →
  A_current = B_current + 12 →
  (A_current + 10) / (B_current - 10) = 2 := by
sorry

end age_ratio_proof_l1496_149698


namespace intersection_of_A_and_B_l1496_149664

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {x | Real.log (x + 1) > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l1496_149664


namespace shaded_area_between_circles_l1496_149638

theorem shaded_area_between_circles (r : ℝ) : 
  r > 0 → 
  (2 * r = 6) → 
  (π * (3 * r)^2 - π * r^2 = 72 * π) := by
sorry

end shaded_area_between_circles_l1496_149638


namespace total_hamburger_combinations_l1496_149600

/-- The number of condiments available for hamburgers. -/
def num_condiments : ℕ := 8

/-- The number of options for meat patties. -/
def num_patty_options : ℕ := 4

/-- Calculates the number of possible condiment combinations. -/
def condiment_combinations : ℕ := 2^num_condiments

/-- Theorem stating the total number of different hamburger combinations. -/
theorem total_hamburger_combinations : 
  num_patty_options * condiment_combinations = 1024 := by
  sorry

end total_hamburger_combinations_l1496_149600


namespace floor_sum_example_l1496_149601

theorem floor_sum_example : ⌊(24.7 : ℝ)⌋ + ⌊(-24.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l1496_149601


namespace first_number_value_l1496_149617

theorem first_number_value (a b c d : ℝ) : 
  (a + b + c) / 3 = 20 →
  (b + c + d) / 3 = 15 →
  d = 18 →
  a = 33 := by
sorry

end first_number_value_l1496_149617


namespace machine_value_depletion_rate_l1496_149647

theorem machine_value_depletion_rate 
  (present_value : ℝ) 
  (value_after_2_years : ℝ) 
  (depletion_rate : ℝ) : 
  present_value = 1100 → 
  value_after_2_years = 891 → 
  value_after_2_years = present_value * (1 - depletion_rate)^2 → 
  depletion_rate = 0.1 := by
sorry

end machine_value_depletion_rate_l1496_149647


namespace problem_solution_l1496_149694

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem problem_solution (m n : ℝ) 
  (h1 : 0 < m) (h2 : m < n) 
  (h3 : f m = f n) 
  (h4 : ∀ x ∈ Set.Icc (m^2) n, f x ≤ 2) 
  (h5 : ∃ x ∈ Set.Icc (m^2) n, f x = 2) : 
  n / m = Real.exp 2 := by
sorry

end problem_solution_l1496_149694


namespace calculate_expression_l1496_149643

theorem calculate_expression : 
  |-Real.sqrt 3| - (4 - Real.pi)^0 - 2 * Real.sin (60 * π / 180) + (1/5)⁻¹ = 4 := by
  sorry

end calculate_expression_l1496_149643


namespace inequality_proof_l1496_149677

theorem inequality_proof (a b : ℝ) (h : a ≠ b) :
  a^4 + 6*a^2*b^2 + b^4 > 4*a*b*(a^2 + b^2) := by
  sorry

end inequality_proof_l1496_149677


namespace yushu_donations_l1496_149636

/-- The number of matching combinations for backpacks and pencil cases -/
def matching_combinations (backpack_styles : ℕ) (pencil_case_styles : ℕ) : ℕ :=
  backpack_styles * pencil_case_styles

/-- Theorem: Given 2 backpack styles and 2 pencil case styles, there are 4 matching combinations -/
theorem yushu_donations : matching_combinations 2 2 = 4 := by
  sorry

end yushu_donations_l1496_149636


namespace oil_bill_ratio_change_l1496_149612

theorem oil_bill_ratio_change 
  (january_bill : ℝ) 
  (february_bill : ℝ) 
  (initial_ratio : ℚ) 
  (added_amount : ℝ) :
  january_bill = 59.99999999999997 →
  initial_ratio = 3 / 2 →
  february_bill / january_bill = initial_ratio →
  (february_bill + added_amount) / january_bill = 5 / 3 →
  added_amount = 10 := by
sorry

end oil_bill_ratio_change_l1496_149612


namespace triangle_angle_tangent_l1496_149648

theorem triangle_angle_tangent (A : Real) :
  (Real.sqrt 3 * Real.cos A + Real.sin A) / (Real.sqrt 3 * Real.sin A - Real.cos A) = Real.tan (-7 * π / 12) →
  Real.tan A = 1 := by
sorry

end triangle_angle_tangent_l1496_149648


namespace circle_area_increase_l1496_149632

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end circle_area_increase_l1496_149632


namespace three_speakers_from_different_companies_l1496_149606

/-- The number of companies -/
def num_companies : ℕ := 5

/-- The number of representatives for Company A -/
def company_a_reps : ℕ := 2

/-- The number of representatives for each of the other companies -/
def other_company_reps : ℕ := 1

/-- The number of speakers at the meeting -/
def num_speakers : ℕ := 3

/-- The number of ways to select 3 speakers from 3 different companies -/
def num_ways : ℕ := 16

theorem three_speakers_from_different_companies :
  let total_reps := company_a_reps + (num_companies - 1) * other_company_reps
  (Nat.choose total_reps num_speakers) = num_ways := by sorry

end three_speakers_from_different_companies_l1496_149606


namespace f_derivative_l1496_149635

/-- The function f(x) = (5x - 4)^3 -/
def f (x : ℝ) : ℝ := (5 * x - 4) ^ 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 15 * (5 * x - 4) ^ 2

theorem f_derivative :
  ∀ x : ℝ, deriv f x = f' x :=
by
  sorry

end f_derivative_l1496_149635


namespace simplify_expression_l1496_149685

theorem simplify_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = 3*(x + y)) : 
  x/y + y/x - 3/(x*y) = 1 := by sorry

end simplify_expression_l1496_149685


namespace second_part_sum_l1496_149620

/-- Calculates the interest on a principal amount for a given rate and time. -/
def interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Proves that given the conditions, the second part of the sum is 1672. -/
theorem second_part_sum (total : ℚ) (first_part : ℚ) (second_part : ℚ) 
  (h1 : total = 2717)
  (h2 : first_part + second_part = total)
  (h3 : interest first_part 3 8 = interest second_part 5 3) :
  second_part = 1672 := by
  sorry

end second_part_sum_l1496_149620


namespace A_intersect_Z_l1496_149691

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

theorem A_intersect_Z : A ∩ (Set.range (Int.cast : ℤ → ℝ)) = {-1, 0, 1} := by sorry

end A_intersect_Z_l1496_149691
