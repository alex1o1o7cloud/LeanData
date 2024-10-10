import Mathlib

namespace original_number_proof_l426_42666

theorem original_number_proof :
  ∃ (a x y q : ℕ), 
    7 * a = 10 * x + y ∧
    y ≤ 9 ∧
    9 * x = 80 + q ∧
    q ≤ 9 ∧
    (a = 13 ∨ a = 14) :=
by sorry

end original_number_proof_l426_42666


namespace m2_defective_percent_is_one_percent_l426_42667

-- Define the percentage of products from each machine
def m1_percent : ℝ := 0.4
def m2_percent : ℝ := 0.3
def m3_percent : ℝ := 1 - m1_percent - m2_percent

-- Define the percentage of defective products for m1 and m3
def m1_defective_percent : ℝ := 0.03
def m3_non_defective_percent : ℝ := 0.93

-- Define the total percentage of defective products
def total_defective_percent : ℝ := 0.036

-- State the theorem
theorem m2_defective_percent_is_one_percent :
  ∃ (m2_defective_percent : ℝ),
    m2_defective_percent = 0.01 ∧
    m1_percent * m1_defective_percent +
    m2_percent * m2_defective_percent +
    m3_percent * (1 - m3_non_defective_percent) =
    total_defective_percent :=
sorry

end m2_defective_percent_is_one_percent_l426_42667


namespace largest_prime_factor_of_sum_of_divisors_200_l426_42607

/-- The sum of divisors of a natural number n -/
def sumOfDivisors (n : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number n -/
def largestPrimeFactor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_200 :
  largestPrimeFactor (sumOfDivisors 200) = 31 := by sorry

end largest_prime_factor_of_sum_of_divisors_200_l426_42607


namespace brother_papaya_consumption_l426_42682

/-- The number of papayas Jake eats in one week -/
def jake_weekly : ℕ := 3

/-- The number of papayas Jake's father eats in one week -/
def father_weekly : ℕ := 4

/-- The total number of papayas needed for 4 weeks -/
def total_papayas : ℕ := 48

/-- The number of papayas Jake's brother eats in one week -/
def brother_weekly : ℕ := 5

theorem brother_papaya_consumption :
  4 * (jake_weekly + father_weekly + brother_weekly) = total_papayas := by
  sorry

end brother_papaya_consumption_l426_42682


namespace employed_males_percentage_l426_42649

theorem employed_males_percentage (total_population : ℝ) 
  (employed_percentage : ℝ) (employed_females_percentage : ℝ) :
  employed_percentage = 96 →
  employed_females_percentage = 75 →
  (employed_percentage / 100 * (100 - employed_females_percentage) / 100 * 100) = 24 := by
  sorry

end employed_males_percentage_l426_42649


namespace expected_additional_cases_l426_42640

/-- Proves the expected number of additional individuals with a condition in a sample -/
theorem expected_additional_cases
  (population_ratio : ℚ) -- Ratio of population with the condition
  (sample_size : ℕ) -- Size of the sample
  (known_cases : ℕ) -- Number of known cases in the sample
  (h1 : population_ratio = 1 / 4) -- Condition: 1/4 of population has the condition
  (h2 : sample_size = 300) -- Condition: Sample size is 300
  (h3 : known_cases = 20) -- Condition: 20 known cases in the sample
  : ℕ := by
  sorry

#check expected_additional_cases

end expected_additional_cases_l426_42640


namespace triangle_properties_l426_42604

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ) -- angles
  (a b c : ℝ) -- opposite sides

-- Define the conditions and theorems
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = Real.sqrt 7)
  (h2 : Real.sin t.A = Real.sqrt 3 * Real.sin t.C) :
  (t.B = π / 6 → Real.sin t.B = Real.sin t.C) ∧ 
  (t.B > π / 2 ∧ Real.cos (2 * t.B) = 1 / 2 → 
    t.a * Real.sin t.C = Real.sqrt 21 / 14) := by
  sorry

#check triangle_properties

end triangle_properties_l426_42604


namespace watermelon_price_in_ten_thousand_won_l426_42664

/-- The price of a watermelon in won -/
def watermelon_price : ℝ := 50000 - 2000

/-- Conversion factor from won to ten thousand won -/
def won_to_ten_thousand : ℝ := 10000

theorem watermelon_price_in_ten_thousand_won : 
  watermelon_price / won_to_ten_thousand = 4.8 := by sorry

end watermelon_price_in_ten_thousand_won_l426_42664


namespace quadratic_equation_properties_l426_42655

/-- Quadratic equation type -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  m : ℝ

/-- Checks if x is a root of the quadratic equation -/
def is_root (eq : QuadraticEquation) (x : ℝ) : Prop :=
  eq.a * x^2 + eq.b * x + eq.m = 0

/-- Defines the relationship between roots of a quadratic equation -/
def roots_relationship (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 + 5*x₁*x₂ - x₁^2*x₂^2 = 0

theorem quadratic_equation_properties (eq : QuadraticEquation) 
  (h_eq : eq.a = 2 ∧ eq.b = 4) :
  (is_root eq 1 → eq.m = -6 ∧ is_root eq (-3)) ∧
  (∃ x₁ x₂ : ℝ, is_root eq x₁ ∧ is_root eq x₂ ∧ roots_relationship x₁ x₂ → eq.m = -2) :=
sorry

end quadratic_equation_properties_l426_42655


namespace notebook_price_proof_l426_42648

/-- The cost of a notebook in cents, given the number of nickels used and the value of a nickel -/
def notebook_cost (num_nickels : ℕ) (nickel_value : ℕ) : ℕ :=
  num_nickels * nickel_value

/-- Converts cents to dollars -/
def cents_to_dollars (cents : ℕ) : ℚ :=
  (cents : ℚ) / 100

theorem notebook_price_proof :
  let num_nickels : ℕ := 26
  let nickel_value : ℕ := 5
  cents_to_dollars (notebook_cost num_nickels nickel_value) = 1.3 := by
  sorry

end notebook_price_proof_l426_42648


namespace sergey_ndfl_calculation_l426_42628

/-- Calculates the personal income tax (НДФЛ) for a Russian resident --/
def calculate_ndfl (monthly_income : ℕ) (bonus : ℕ) (car_sale : ℕ) (land_purchase : ℕ) : ℕ :=
  let annual_income := monthly_income * 12
  let total_income := annual_income + bonus + car_sale
  let total_deductions := car_sale + land_purchase
  let taxable_income := total_income - total_deductions
  let tax_rate := 13
  (taxable_income * tax_rate) / 100

/-- Theorem stating that the calculated НДФЛ for Sergey is 10400 rubles --/
theorem sergey_ndfl_calculation :
  calculate_ndfl 30000 20000 250000 300000 = 10400 := by
  sorry

end sergey_ndfl_calculation_l426_42628


namespace square_intersection_perimeter_ratio_l426_42692

/-- Given a square with side length 2b and a line y = -x/3 intersecting it,
    the ratio of the perimeter of one resulting quadrilateral to b is (14 + √13) / 3 -/
theorem square_intersection_perimeter_ratio (b : ℝ) (b_pos : b > 0) :
  let square_vertices := [(-b, -b), (b, -b), (-b, b), (b, b)]
  let line := fun x => -x / 3
  let intersection_points := [
    (b, line b),
    (-b, line (-b))
  ]
  let quadrilateral_vertices := [
    (-b, -b),
    (b, -b),
    (b, line b),
    (-b, line (-b))
  ]
  let perimeter := 
    (b - line b) + (b - line (-b)) + 2*b + 
    Real.sqrt ((2*b)^2 + (line b - line (-b))^2)
  perimeter / b = (14 + Real.sqrt 13) / 3 :=
by sorry

end square_intersection_perimeter_ratio_l426_42692


namespace probability_of_y_selection_l426_42658

theorem probability_of_y_selection (p_x p_both : ℝ) 
  (h_x : p_x = 1/5)
  (h_both : p_both = 0.13333333333333333)
  (h_independent : p_both = p_x * (p_both / p_x)) :
  p_both / p_x = 0.6666666666666667 :=
by sorry

end probability_of_y_selection_l426_42658


namespace opposite_and_reciprocal_expression_l426_42650

theorem opposite_and_reciprocal_expression (x y a b : ℝ) 
  (h1 : x = -y) 
  (h2 : a * b = 1) : 
  x + y - 3 / (a * b) = -3 := by sorry

end opposite_and_reciprocal_expression_l426_42650


namespace square_root_of_sixteen_l426_42617

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end square_root_of_sixteen_l426_42617


namespace triangle_side_length_l426_42676

theorem triangle_side_length (A B C : ℝ × ℝ) :
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let angle_BAC := Real.arccos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC))
  AC = 4 ∧ BC = 2 * Real.sqrt 7 ∧ angle_BAC = π / 3 → AB = 6 :=
by
  sorry

end triangle_side_length_l426_42676


namespace sum_of_max_and_min_equals_two_l426_42694

noncomputable def f (x : ℝ) : ℝ := ((x + 2)^2 + Real.sin x) / (x^2 + 4)

theorem sum_of_max_and_min_equals_two :
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
               (∀ x, m ≤ f x) ∧ (∃ x, f x = m) ∧
               (M + m = 2) := by
  sorry

end sum_of_max_and_min_equals_two_l426_42694


namespace exponential_function_value_l426_42660

/-- Given an exponential function f(x) = a^x where a > 0, a ≠ 1, and f(3) = 8, prove that f(1) = 2 -/
theorem exponential_function_value (a : ℝ) (f : ℝ → ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x, f x = a^x) 
  (h4 : f 3 = 8) : 
  f 1 = 2 := by
sorry

end exponential_function_value_l426_42660


namespace common_chord_of_circles_l426_42670

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 2*y - 40 = 0

-- Define the line
def common_chord (x y : ℝ) : Prop := 2*x + y - 5 = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → common_chord x y :=
by sorry

end common_chord_of_circles_l426_42670


namespace solve_bodyguard_problem_l426_42620

def bodyguard_problem (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (total_cost : ℕ) : Prop :=
  let daily_cost := num_bodyguards * hourly_rate * hours_per_day
  ∃ (days : ℕ), days * daily_cost = total_cost ∧ days = 7

theorem solve_bodyguard_problem :
  bodyguard_problem 2 20 8 2240 :=
sorry

end solve_bodyguard_problem_l426_42620


namespace chameleons_cannot_be_same_color_l426_42638

/-- Represents the color of a chameleon -/
inductive Color
  | Blue
  | White
  | Red

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  blue : Nat
  white : Nat
  red : Nat

/-- The initial state of chameleons -/
def initialState : ChameleonState :=
  { blue := 800, white := 220, red := 1003 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 2023

/-- Calculates the invariant Q for a given state -/
def calculateQ (state : ChameleonState) : Int :=
  state.blue - state.white

/-- Represents a meeting between two chameleons of different colors -/
def meetingTransformation (state : ChameleonState) (c1 c2 : Color) : ChameleonState :=
  match c1, c2 with
  | Color.Blue, Color.White => { state with blue := state.blue - 1, white := state.white - 1, red := state.red + 2 }
  | Color.Blue, Color.Red => { state with blue := state.blue - 1, white := state.white + 2, red := state.red - 1 }
  | Color.White, Color.Red => { state with blue := state.blue + 2, white := state.white - 1, red := state.red - 1 }
  | Color.White, Color.Blue => { state with blue := state.blue - 1, white := state.white - 1, red := state.red + 2 }
  | Color.Red, Color.Blue => { state with blue := state.blue - 1, white := state.white + 2, red := state.red - 1 }
  | Color.Red, Color.White => { state with blue := state.blue + 2, white := state.white - 1, red := state.red - 1 }
  | _, _ => state  -- No change if same color

/-- Theorem: It is impossible for all chameleons to become the same color -/
theorem chameleons_cannot_be_same_color :
  ∀ (finalState : ChameleonState),
  (finalState.blue + finalState.white + finalState.red = totalChameleons) →
  (finalState.blue = totalChameleons ∨ finalState.white = totalChameleons ∨ finalState.red = totalChameleons) →
  False := by
  sorry


end chameleons_cannot_be_same_color_l426_42638


namespace product_from_hcf_lcm_l426_42646

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 1485) :
  a * b = 17820 := by
  sorry

end product_from_hcf_lcm_l426_42646


namespace ellipse_m_value_l426_42631

/-- Definition of an ellipse with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (10 - m) + y^2 / (m - 2) = 1

/-- The major axis of the ellipse lies on the y-axis -/
def major_axis_on_y (m : ℝ) : Prop :=
  m - 2 > 10 - m

/-- The focal distance of the ellipse is 4 -/
def focal_distance_4 (m : ℝ) : Prop :=
  4^2 = 4 * (m - 2) - 4 * (10 - m)

/-- Theorem: For an ellipse with the given properties, m equals 8 -/
theorem ellipse_m_value (m : ℝ) 
  (h1 : is_ellipse m) 
  (h2 : major_axis_on_y m) 
  (h3 : focal_distance_4 m) : 
  m = 8 := by
  sorry

end ellipse_m_value_l426_42631


namespace area_after_transformation_l426_42632

/-- Given a 2x2 matrix and a planar region, this function returns the area of the transformed region --/
noncomputable def transformedArea (a b c d : ℝ) (originalArea : ℝ) : ℝ :=
  (a * d - b * c) * originalArea

/-- Theorem stating that applying the given matrix to a region of area 9 results in a region of area 126 --/
theorem area_after_transformation :
  let matrix := !![3, 2; -1, 4]
  let originalArea := 9
  transformedArea 3 2 (-1) 4 originalArea = 126 := by
  sorry

end area_after_transformation_l426_42632


namespace tetrahedron_special_points_l426_42602

-- Define the tetrahedron P-ABC
structure Tetrahedron :=
  (P A B C : EuclideanSpace ℝ (Fin 3))

-- Define the projection O of P onto the base plane ABC
def projection (t : Tetrahedron) : EuclideanSpace ℝ (Fin 3) := sorry

-- Define the property of equal angles between lateral edges and base plane
def equal_lateral_base_angles (t : Tetrahedron) : Prop := sorry

-- Define the property of mutually perpendicular lateral edges
def perpendicular_lateral_edges (t : Tetrahedron) : Prop := sorry

-- Define the property of equal angles between side faces and base plane
def equal_face_base_angles (t : Tetrahedron) : Prop := sorry

-- Define the circumcenter of a triangle
def is_circumcenter (O A B C : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Define the orthocenter of a triangle
def is_orthocenter (O A B C : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Define the incenter of a triangle
def is_incenter (O A B C : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Theorem statements
theorem tetrahedron_special_points (t : Tetrahedron) :
  (equal_lateral_base_angles t → is_circumcenter (projection t) t.A t.B t.C) ∧
  (perpendicular_lateral_edges t → is_orthocenter (projection t) t.A t.B t.C) ∧
  (equal_face_base_angles t → is_incenter (projection t) t.A t.B t.C) := by sorry

end tetrahedron_special_points_l426_42602


namespace carol_wins_probability_l426_42654

-- Define the probability of rolling an eight on an 8-sided die
def prob_eight : ℚ := 1 / 8

-- Define the probability of not rolling an eight
def prob_not_eight : ℚ := 1 - prob_eight

-- Define the probability of Carol winning in the first cycle
def prob_carol_first_cycle : ℚ := prob_not_eight * prob_not_eight * prob_eight

-- Define the probability of no one winning in a single cycle
def prob_no_winner_cycle : ℚ := prob_not_eight * prob_not_eight * prob_not_eight

-- State the theorem
theorem carol_wins_probability :
  (prob_carol_first_cycle / (1 - prob_no_winner_cycle)) = 49 / 169 := by
  sorry

end carol_wins_probability_l426_42654


namespace marble_arrangement_theorem_l426_42637

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def restricted_arrangements (n : ℕ) : ℕ := 
  Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 3

def valid_arrangements (n : ℕ) : ℕ := 
  total_arrangements n - restricted_arrangements n

theorem marble_arrangement_theorem : 
  valid_arrangements 5 = 48 := by sorry

end marble_arrangement_theorem_l426_42637


namespace harvest_earnings_problem_harvest_duration_l426_42625

/-- Represents the harvest earnings problem --/
theorem harvest_earnings_problem (total_earnings : ℕ) (initial_earnings : ℕ) 
  (weekly_increase : ℕ) (weekly_deduction : ℕ) (weeks : ℕ) : Prop :=
  total_earnings = 1216 ∧
  initial_earnings = 16 ∧
  weekly_increase = 8 ∧
  weekly_deduction = 12 ∧
  weeks = 17 →
  total_earnings = (weeks * (2 * initial_earnings + (weeks - 1) * weekly_increase)) / 2 - 
    weeks * weekly_deduction

/-- Proves that the harvest lasted 17 weeks --/
theorem harvest_duration : 
  ∃ (total_earnings initial_earnings weekly_increase weekly_deduction weeks : ℕ),
  harvest_earnings_problem total_earnings initial_earnings weekly_increase weekly_deduction weeks :=
by
  sorry

end harvest_earnings_problem_harvest_duration_l426_42625


namespace opposite_sign_implications_l426_42644

theorem opposite_sign_implications (a b : ℝ) 
  (h : (|a - 2| ≥ 0 ∧ (b + 1)^2 ≤ 0) ∨ (|a - 2| ≤ 0 ∧ (b + 1)^2 ≥ 0)) : 
  b^a = 1 ∧ a^3 + b^15 = 7 := by
  sorry

end opposite_sign_implications_l426_42644


namespace completing_square_equivalence_l426_42669

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end completing_square_equivalence_l426_42669


namespace sqrt_expression_simplification_l426_42623

theorem sqrt_expression_simplification :
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6 := by
  sorry

end sqrt_expression_simplification_l426_42623


namespace quadratic_function_passes_through_points_l426_42689

/-- A quadratic function f(x) = x^2 + 2x - 3 -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- The given points that the function should pass through -/
def points : List (ℝ × ℝ) := [(-2, -3), (-1, -4), (0, -3), (2, 5)]

theorem quadratic_function_passes_through_points :
  ∀ (p : ℝ × ℝ), p ∈ points → f p.1 = p.2 := by
  sorry

end quadratic_function_passes_through_points_l426_42689


namespace correct_average_marks_l426_42610

theorem correct_average_marks (n : ℕ) (incorrect_avg : ℝ) (wrong_mark correct_mark : ℝ) :
  n = 10 ∧ incorrect_avg = 100 ∧ wrong_mark = 60 ∧ correct_mark = 10 →
  (n * incorrect_avg - (wrong_mark - correct_mark)) / n = 95 := by
  sorry

end correct_average_marks_l426_42610


namespace smallest_n_congruence_l426_42600

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 567 [ZMOD 9]) → n ≥ 9 :=
by sorry

end smallest_n_congruence_l426_42600


namespace x_squared_minus_y_squared_l426_42612

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 3/8) 
  (h2 : x - y = 1/4) : 
  x^2 - y^2 = 3/32 := by sorry

end x_squared_minus_y_squared_l426_42612


namespace visitors_to_both_countries_l426_42685

theorem visitors_to_both_countries (total : ℕ) (iceland : ℕ) (norway : ℕ) (neither : ℕ) : 
  total = 90 → iceland = 55 → norway = 33 → neither = 53 → 
  total - neither = iceland + norway - (total - neither) := by
  sorry

end visitors_to_both_countries_l426_42685


namespace pizza_dough_scaling_l426_42635

/-- Calculates the required milk for a scaled pizza dough recipe -/
theorem pizza_dough_scaling (original_flour original_milk new_flour : ℚ) : 
  original_flour > 0 ∧ 
  original_milk > 0 ∧ 
  new_flour > 0 ∧
  original_flour = 400 ∧
  original_milk = 80 ∧
  new_flour = 1200 →
  (new_flour / original_flour) * original_milk = 240 := by
  sorry

end pizza_dough_scaling_l426_42635


namespace perfect_square_with_powers_of_three_l426_42679

/-- For which integers k (0 ≤ k ≤ 9) do there exist positive integers m and n
    so that 3^m + 3^n + k is a perfect square? -/
theorem perfect_square_with_powers_of_three (k : ℕ) : 
  (k ≤ 9) → 
  (∃ (m n : ℕ+), ∃ (s : ℕ), (3^m.val + 3^n.val + k = s^2)) ↔ 
  (k = 0 ∨ k = 3 ∨ k = 4 ∨ k = 6 ∨ k = 7) :=
by sorry

end perfect_square_with_powers_of_three_l426_42679


namespace correct_sets_count_l426_42662

/-- A set of weights is represented as a multiset of natural numbers -/
def WeightSet := Multiset ℕ

/-- A weight set is correct if it satisfies the given conditions -/
def is_correct_set (s : WeightSet) : Prop :=
  (s.sum = 200) ∧
  (∀ w : ℕ, w ≤ 200 → ∃! subset : Multiset ℕ, subset ⊆ s ∧ subset.sum = w)

/-- The number of correct weight sets -/
def num_correct_sets : ℕ := 3

theorem correct_sets_count :
  ∃ (sets : Finset WeightSet),
    sets.card = num_correct_sets ∧
    (∀ s : WeightSet, s ∈ sets ↔ is_correct_set s) :=
sorry

end correct_sets_count_l426_42662


namespace nba_conference_impossibility_l426_42699

/-- Represents the number of teams in the NBA -/
def total_teams : ℕ := 30

/-- Represents the number of games each team plays in a regular season -/
def games_per_team : ℕ := 82

/-- Represents a potential division of teams into conferences -/
structure Conference_Division where
  eastern : ℕ
  western : ℕ
  sum_teams : eastern + western = total_teams

/-- Represents the condition for inter-conference games -/
def valid_inter_conference_games (d : Conference_Division) : Prop :=
  ∃ (inter_games : ℕ), 
    inter_games * 2 = games_per_team ∧
    d.eastern * inter_games = d.western * inter_games

theorem nba_conference_impossibility : 
  ¬ ∃ (d : Conference_Division), valid_inter_conference_games d :=
sorry

end nba_conference_impossibility_l426_42699


namespace factorial_500_trailing_zeroes_l426_42615

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 500! has 124 trailing zeroes -/
theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end factorial_500_trailing_zeroes_l426_42615


namespace fraction_inequality_l426_42680

theorem fraction_inequality (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hm : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end fraction_inequality_l426_42680


namespace condition_one_condition_two_condition_three_condition_four_l426_42659

/-- Represents the number of male athletes -/
def num_male : ℕ := 6

/-- Represents the number of female athletes -/
def num_female : ℕ := 4

/-- Represents the total number of athletes -/
def total_athletes : ℕ := num_male + num_female

/-- Represents the size of the team to be selected -/
def team_size : ℕ := 5

/-- Represents the number of male athletes in the team for condition 1 -/
def male_in_team : ℕ := 3

/-- Represents the number of female athletes in the team for condition 1 -/
def female_in_team : ℕ := 2

/-- Theorem for the first condition -/
theorem condition_one : 
  (num_male.choose male_in_team) * (num_female.choose female_in_team) = 120 := by sorry

/-- Theorem for the second condition -/
theorem condition_two : 
  total_athletes.choose team_size - num_male.choose team_size = 246 := by sorry

/-- Theorem for the third condition -/
theorem condition_three : 
  total_athletes.choose team_size - (num_male - 1).choose team_size - (num_female - 1).choose team_size + (total_athletes - 2).choose team_size = 196 := by sorry

/-- Theorem for the fourth condition -/
theorem condition_four : 
  (total_athletes - 1).choose (team_size - 1) + ((total_athletes - 2).choose (team_size - 1) - (num_male - 1).choose (team_size - 1)) = 191 := by sorry

end condition_one_condition_two_condition_three_condition_four_l426_42659


namespace parabola_range_l426_42686

/-- A parabola with the equation y = ax² - 3x + 1 -/
structure Parabola where
  a : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The axis of symmetry for a parabola -/
def axisOfSymmetry (p : Parabola) : ℝ := sorry

/-- Predicate to check if a point is on the parabola -/
def isOnParabola (point : Point) (p : Parabola) : Prop :=
  point.y = p.a * point.x^2 - 3 * point.x + 1

/-- Predicate to check if the parabola intersects a line segment at two distinct points -/
def intersectsTwice (p : Parabola) (p1 p2 : Point) : Prop := sorry

theorem parabola_range (p : Parabola) (A B M N : Point)
    (hA : isOnParabola A p)
    (hB : isOnParabola B p)
    (hM : M.x = -1 ∧ M.y = -2)
    (hN : N.x = 3 ∧ N.y = 2)
    (hAB : ∀ x₀, |A.x - x₀| > |B.x - x₀| → A.y > B.y)
    (hIntersect : intersectsTwice p M N) :
    10/9 ≤ p.a ∧ p.a < 2 := by
  sorry

end parabola_range_l426_42686


namespace relationship_between_x_and_y_l426_42675

theorem relationship_between_x_and_y (x y : ℝ) 
  (h1 : x - y > x + 2) 
  (h2 : x + y + 3 < y - 1) : 
  x < -4 ∧ y < -2 := by
sorry

end relationship_between_x_and_y_l426_42675


namespace binary_101_eq_5_l426_42629

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 5 -/
def binary_5 : List Bool := [true, false, true]

/-- Theorem stating that the binary number 101 (represented as [true, false, true]) is equal to 5 in decimal -/
theorem binary_101_eq_5 : binary_to_decimal binary_5 = 5 := by sorry

end binary_101_eq_5_l426_42629


namespace tangent_points_parameter_l426_42693

/-- Given a circle and two tangent points, prove that the parameter 'a' has specific values -/
theorem tangent_points_parameter (a : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*a*x + 2*y - 1 = 0) →  -- Circle equation
  (x₁^2 + y₁^2 - 2*a*x₁ + 2*y₁ - 1 = 0) →  -- M is on the circle
  (x₂^2 + y₂^2 - 2*a*x₂ + 2*y₂ - 1 = 0) →  -- N is on the circle
  ((y₁ - (-1)) / (x₁ - a))^2 = (((-5) - a)^2 + (a - (-1))^2) / ((x₁ - (-5))^2 + (y₁ - a)^2) →  -- M is a tangent point
  ((y₂ - (-1)) / (x₂ - a))^2 = (((-5) - a)^2 + (a - (-1))^2) / ((x₂ - (-5))^2 + (y₂ - a)^2) →  -- N is a tangent point
  (y₂ - y₁) / (x₂ - x₁) + (x₁ + x₂ - 2) / (y₁ + y₂) = 0 →  -- Given condition
  a = 3 ∨ a = -2 := by
sorry

end tangent_points_parameter_l426_42693


namespace second_train_speed_l426_42673

/-- Proves that the speed of the second train is 120 kmph given the conditions of the problem -/
theorem second_train_speed 
  (first_train_departure : ℕ) -- Departure time of the first train in hours after midnight
  (second_train_departure : ℕ) -- Departure time of the second train in hours after midnight
  (first_train_speed : ℝ) -- Speed of the first train in kmph
  (meeting_distance : ℝ) -- Distance where the trains meet in km
  (h1 : first_train_departure = 9) -- First train leaves at 9 a.m.
  (h2 : second_train_departure = 15) -- Second train leaves at 3 p.m.
  (h3 : first_train_speed = 30) -- First train speed is 30 kmph
  (h4 : meeting_distance = 720) -- Trains meet 720 km away from Delhi
  : ∃ (second_train_speed : ℝ), second_train_speed = 120 := by
  sorry

end second_train_speed_l426_42673


namespace largest_reciprocal_l426_42624

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem largest_reciprocal :
  let a := (1 : ℚ) / 2
  let b := (3 : ℚ) / 7
  let c := (1 : ℚ) / 2  -- 0.5 as a rational number
  let d := 7
  let e := 2001
  (reciprocal b > reciprocal a) ∧
  (reciprocal b > reciprocal c) ∧
  (reciprocal b > reciprocal d) ∧
  (reciprocal b > reciprocal e) :=
by sorry

end largest_reciprocal_l426_42624


namespace right_triangle_acute_angle_l426_42641

theorem right_triangle_acute_angle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  c = 90 →           -- One angle is a right angle (90°)
  a = 55 →           -- One acute angle is 55°
  b = 35             -- The other acute angle is 35°
:= by sorry

end right_triangle_acute_angle_l426_42641


namespace phrase_repetition_l426_42688

/-- Represents a mapping from letters to words -/
def LetterWordMap (α : Type) := α → List α

/-- The result of applying the letter-to-word mapping n times -/
def applyMapping (n : ℕ) (map : LetterWordMap α) (initial : List α) : List α :=
  match n with
  | 0 => initial
  | n + 1 => applyMapping n map (initial.bind map)

theorem phrase_repetition 
  (α : Type) [Finite α] 
  (map : LetterWordMap α) 
  (initial : List α) 
  (h1 : initial.length ≥ 6) 
  (h2 : ∀ (a : α), (map a).length ≥ 1) :
  ∃ (i j : ℕ), 
    i ≠ j ∧ 
    i < (applyMapping 40 map initial).length ∧ 
    j < (applyMapping 40 map initial).length ∧
    (applyMapping 40 map initial).take 6 = 
      ((applyMapping 40 map initial).drop i).take 6 ∧
    (applyMapping 40 map initial).take 6 = 
      ((applyMapping 40 map initial).drop j).take 6 :=
by
  sorry


end phrase_repetition_l426_42688


namespace stating_assignment_methods_eq_36_l426_42601

/-- Represents the number of workshops --/
def num_workshops : ℕ := 3

/-- Represents the total number of employees --/
def total_employees : ℕ := 5

/-- Represents the number of employees that must be assigned together --/
def paired_employees : ℕ := 2

/-- Represents the number of remaining employees after considering the paired employees --/
def remaining_employees : ℕ := total_employees - paired_employees

/-- 
  Calculates the number of ways to assign employees to workshops
  given the constraints mentioned in the problem
--/
def assignment_methods : ℕ := 
  num_workshops * (remaining_employees.factorial + remaining_employees.choose 2 * (num_workshops - 1))

/-- 
  Theorem stating that the number of assignment methods
  satisfying the given conditions is 36
--/
theorem assignment_methods_eq_36 : assignment_methods = 36 := by
  sorry

end stating_assignment_methods_eq_36_l426_42601


namespace jane_vases_per_day_l426_42678

/-- The number of vases Jane can arrange per day given the total number of vases and the number of days -/
def vases_per_day (total_vases : ℕ) (days : ℕ) : ℚ :=
  (total_vases : ℚ) / (days : ℚ)

/-- Theorem stating that Jane can arrange 15.5 vases per day given the problem conditions -/
theorem jane_vases_per_day :
  vases_per_day 248 16 = 31/2 := by sorry

end jane_vases_per_day_l426_42678


namespace infinitely_many_solutions_implies_a_eq_neg_two_l426_42639

/-- A system of two linear equations in two variables -/
structure LinearSystem (a : ℝ) where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ
  h1 : ∀ x y, eq1 x y = 2*x + 2*y + 1
  h2 : ∀ x y, eq2 x y = 4*x + a^2*y - a

/-- The system has infinitely many solutions -/
def HasInfinitelySolutions (sys : LinearSystem a) : Prop :=
  ∃ x₀ y₀, ∀ t : ℝ, sys.eq1 (x₀ + t) (y₀ - t) = 0 ∧ sys.eq2 (x₀ + t) (y₀ - t) = 0

/-- When the system has infinitely many solutions, a = -2 -/
theorem infinitely_many_solutions_implies_a_eq_neg_two (a : ℝ) (sys : LinearSystem a) :
  HasInfinitelySolutions sys → a = -2 := by sorry

end infinitely_many_solutions_implies_a_eq_neg_two_l426_42639


namespace min_chestnuts_is_253_l426_42642

/-- Represents the process of a monkey dividing chestnuts -/
def monkey_divide (n : ℕ) : ℕ :=
  if n % 4 = 1 then (3 * (n - 1)) / 4 else 0

/-- Represents the process of all four monkeys dividing chestnuts -/
def all_monkeys_divide (n : ℕ) : ℕ :=
  monkey_divide (monkey_divide (monkey_divide (monkey_divide n)))

/-- Theorem stating that 253 is the minimum number of chestnuts that satisfies the problem conditions -/
theorem min_chestnuts_is_253 :
  (∀ m : ℕ, m < 253 → all_monkeys_divide m ≠ 0) ∧ all_monkeys_divide 253 = 0 :=
sorry

end min_chestnuts_is_253_l426_42642


namespace twins_age_today_twins_age_proof_l426_42695

/-- The age of twin brothers today, given that the product of their ages today is smaller by 11 from the product of their ages a year from today. -/
theorem twins_age_today : ℕ :=
  let age_today : ℕ → Prop := fun x => (x + 1) ^ 2 = x ^ 2 + 11
  5

theorem twins_age_proof (age : ℕ) : (age + 1) ^ 2 = age ^ 2 + 11 → age = 5 := by
  sorry

end twins_age_today_twins_age_proof_l426_42695


namespace four_students_three_events_sign_up_l426_42687

/-- The number of ways for students to sign up for events -/
def num_ways_to_sign_up (num_students : ℕ) (num_events : ℕ) : ℕ :=
  num_events ^ num_students

/-- Theorem: Four students choosing from three events results in 81 possible arrangements -/
theorem four_students_three_events_sign_up :
  num_ways_to_sign_up 4 3 = 81 := by
  sorry

end four_students_three_events_sign_up_l426_42687


namespace power_mod_nineteen_l426_42663

theorem power_mod_nineteen : 2^65537 % 19 = 2 := by
  sorry

end power_mod_nineteen_l426_42663


namespace area_of_valid_fold_points_l426_42671

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is a valid fold point -/
def isValidFoldPoint (t : Triangle) (p : Point) : Prop := sorry

/-- Calculate the area of a set of points -/
def areaOfSet (s : Set Point) : ℝ := sorry

/-- The set of all valid fold points in the triangle -/
def validFoldPoints (t : Triangle) : Set Point := sorry

/-- The main theorem -/
theorem area_of_valid_fold_points (t : Triangle) 
  (h1 : distance t.A t.B = 24)
  (h2 : distance t.B t.C = 48)
  (h3 : distance t.A t.C = 24 * Real.sqrt 2) :
  areaOfSet (validFoldPoints t) = 240 * Real.pi - 360 * Real.sqrt 3 := sorry

end area_of_valid_fold_points_l426_42671


namespace N_is_composite_l426_42603

/-- The number formed by k+1 ones and k zeros in between -/
def N (k : ℕ) : ℕ := 10^(k+1) + 1

/-- Theorem stating that N(k) is composite for k > 1 -/
theorem N_is_composite (k : ℕ) (h : k > 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N k = a * b :=
sorry

end N_is_composite_l426_42603


namespace nutty_professor_mixture_l426_42636

/-- The Nutty Professor's nut mixture problem -/
theorem nutty_professor_mixture
  (cashew_price : ℝ)
  (brazil_price : ℝ)
  (mixture_price : ℝ)
  (cashew_weight : ℝ)
  (h1 : cashew_price = 6.75)
  (h2 : brazil_price = 5.00)
  (h3 : mixture_price = 5.70)
  (h4 : cashew_weight = 20)
  : ∃ (brazil_weight : ℝ),
    cashew_weight * cashew_price + brazil_weight * brazil_price =
    (cashew_weight + brazil_weight) * mixture_price ∧
    cashew_weight + brazil_weight = 50 :=
by sorry

end nutty_professor_mixture_l426_42636


namespace prime_factorization_sum_l426_42691

theorem prime_factorization_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 945 → 2*w + 3*x + 5*y + 7*z = 21 := by
  sorry

end prime_factorization_sum_l426_42691


namespace rural_school_absence_percentage_l426_42645

/-- Rural School Z student absence problem -/
theorem rural_school_absence_percentage :
  let total_students : ℕ := 150
  let boys : ℕ := 90
  let girls : ℕ := 60
  let boys_absent_ratio : ℚ := 1 / 9
  let girls_absent_ratio : ℚ := 1 / 4
  let absent_students : ℚ := boys_absent_ratio * boys + girls_absent_ratio * girls
  absent_students / total_students = 1 / 6 := by
  sorry

end rural_school_absence_percentage_l426_42645


namespace f_decreasing_iff_a_range_l426_42684

/-- A function f(x) = x^2 + 2(a-1)x + 2 that is decreasing on (-∞, 4] -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*(a-1)

theorem f_decreasing_iff_a_range :
  ∀ a : ℝ, (∀ x ≤ 4, f_deriv a x ≤ 0) ↔ a < -3 := by sorry


end f_decreasing_iff_a_range_l426_42684


namespace z_in_first_quadrant_l426_42606

def i : ℂ := Complex.I

theorem z_in_first_quadrant : ∃ z : ℂ, 
  (1 + i) * z = 1 - 2 * i^3 ∧ 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end z_in_first_quadrant_l426_42606


namespace polygon_sides_l426_42611

/-- A polygon has n sides if its interior angles sum is 4 times its exterior angles sum -/
theorem polygon_sides (n : ℕ) : n = 10 :=
  by
  -- Define the sum of interior angles
  let interior_sum := (n - 2) * 180
  -- Define the sum of exterior angles
  let exterior_sum := 360
  -- State the condition that interior sum is 4 times exterior sum
  have h : interior_sum = 4 * exterior_sum := by sorry
  -- Prove that n = 10
  sorry


end polygon_sides_l426_42611


namespace max_power_under_500_l426_42618

theorem max_power_under_500 :
  ∃ (c d : ℕ), d > 1 ∧ c^d < 500 ∧
  (∀ (x y : ℕ), y > 1 → x^y < 500 → x^y ≤ c^d) ∧
  c + d = 24 :=
sorry

end max_power_under_500_l426_42618


namespace corrected_mean_l426_42661

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value correct_value : ℝ) :
  n = 50 →
  original_mean = 36 →
  incorrect_value = 23 →
  correct_value = 44 →
  let original_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := original_sum + difference
  let corrected_mean := corrected_sum / n
  corrected_mean = 36.42 := by
sorry

end corrected_mean_l426_42661


namespace no_valid_covering_l426_42672

/-- Represents a 3x3 grid with alternating colors -/
def AlternateColoredGrid : Type := Fin 3 → Fin 3 → Bool

/-- An L-shaped piece covers exactly 3 squares -/
def LPiece : Type := Fin 3 → Fin 2 → Bool

/-- A covering of the grid is a list of L-pieces and their positions -/
def Covering : Type := List (LPiece × Fin 3 × Fin 3)

/-- Checks if a given covering is valid for the 3x3 grid -/
def is_valid_covering (c : Covering) : Prop := sorry

/-- The coloring of the 3x3 grid where corners and center are one color (true) 
    and the rest are another color (false) -/
def standard_coloring : AlternateColoredGrid := 
  fun i j => (i = j) || (i + j = 2)

/-- Any L-piece covers an uneven number of squares of each color -/
axiom l_piece_uneven_coverage (l : LPiece) (i j : Fin 3) :
  (∃ (x : Fin 3) (y : Fin 3), 
    (x ≠ i ∨ y ≠ j) ∧ 
    standard_coloring x y ≠ standard_coloring i j ∧
    l (x - i) (y - j))

/-- Theorem: It's impossible to cover a 3x3 grid with L-shaped pieces -/
theorem no_valid_covering : ¬∃ (c : Covering), is_valid_covering c := by
  sorry

end no_valid_covering_l426_42672


namespace lowest_score_proof_l426_42643

def scores_sum (n : ℕ) (mean : ℝ) : ℝ := n * mean

theorem lowest_score_proof 
  (total_scores : ℕ) 
  (original_mean : ℝ) 
  (new_mean : ℝ) 
  (highest_score : ℝ) 
  (h1 : total_scores = 15)
  (h2 : original_mean = 85)
  (h3 : new_mean = 90)
  (h4 : highest_score = 105)
  (h5 : scores_sum total_scores original_mean = 
        scores_sum (total_scores - 2) new_mean + highest_score + 
        (scores_sum total_scores original_mean - scores_sum (total_scores - 2) new_mean - highest_score)) :
  scores_sum total_scores original_mean - scores_sum (total_scores - 2) new_mean - highest_score = 0 := by
sorry

end lowest_score_proof_l426_42643


namespace line_equation_correct_l426_42683

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfiesEquation (p : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * p.1 + eq.b * p.2 + eq.c = 0

/-- Check if a line equation represents a line with a given slope -/
def hasSlope (eq : LineEquation) (m : ℝ) : Prop :=
  eq.b ≠ 0 ∧ -eq.a / eq.b = m

theorem line_equation_correct (l : Line) (eq : LineEquation) :
    l.point = (2, 1) →
    l.slope = -2 →
    satisfiesEquation l.point eq →
    hasSlope eq l.slope →
    eq = { a := 2, b := 1, c := -5 } :=
  sorry

end line_equation_correct_l426_42683


namespace cos_minus_sin_value_l426_42630

theorem cos_minus_sin_value (θ : Real) 
  (h1 : θ ∈ Set.Ioo (3 * Real.pi / 4) Real.pi) 
  (h2 : Real.sin θ * Real.cos θ = -Real.sqrt 3 / 2) : 
  Real.cos θ - Real.sin θ = -Real.sqrt (1 + Real.sqrt 3) := by
  sorry

end cos_minus_sin_value_l426_42630


namespace complex_fraction_simplification_l426_42633

theorem complex_fraction_simplification :
  (2 * Complex.I) / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end complex_fraction_simplification_l426_42633


namespace max_area_triangle_is_isosceles_l426_42665

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Check if a point lies on a circle -/
def onCircle (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  sorry

/-- The theorem stating that the triangle of maximum area is isosceles -/
theorem max_area_triangle_is_isosceles
  (c : Circle) (p : Point) (h : ¬ onCircle c p) :
  ∃ (a b : Point),
    onCircle c a ∧ onCircle c b ∧
    ∀ (x y : Point),
      onCircle c x → onCircle c y →
      triangleArea p x y ≤ triangleArea p a b →
      (p.1 - a.1)^2 + (p.2 - a.2)^2 = (p.1 - b.1)^2 + (p.2 - b.2)^2 :=
sorry

end max_area_triangle_is_isosceles_l426_42665


namespace cube_max_volume_l426_42677

variable (a : ℝ) -- Sum of all edges
variable (x y z : ℝ) -- Dimensions of the parallelepiped

-- Define the volume function
def volume (x y z : ℝ) : ℝ := x * y * z

-- Define the constraint that the sum of edges is fixed
def sum_constraint (x y z : ℝ) : Prop := x + y + z = a

-- State the theorem
theorem cube_max_volume :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → sum_constraint a x y z →
  volume x y z ≤ volume (a/3) (a/3) (a/3) :=
sorry

end cube_max_volume_l426_42677


namespace ratio_of_sums_l426_42626

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  firstTerm : ℕ
  difference : ℕ
  length : ℕ

/-- Calculates the sum of an arithmetic progression -/
def sumArithmeticProgression (ap : ArithmeticProgression) : ℕ :=
  ap.length * (2 * ap.firstTerm + (ap.length - 1) * ap.difference) / 2

/-- Generates a list of arithmetic progressions for the first group -/
def firstGroup : List ArithmeticProgression :=
  List.range 15 |>.map (fun i => ⟨i + 1, 2 * (i + 1), 10⟩)

/-- Generates a list of arithmetic progressions for the second group -/
def secondGroup : List ArithmeticProgression :=
  List.range 15 |>.map (fun i => ⟨i + 1, 2 * i + 1, 10⟩)

/-- Calculates the sum of all elements in a group of arithmetic progressions -/
def sumGroup (group : List ArithmeticProgression) : ℕ :=
  group.map sumArithmeticProgression |>.sum

theorem ratio_of_sums : (sumGroup firstGroup : ℚ) / (sumGroup secondGroup) = 160 / 151 := by
  sorry

end ratio_of_sums_l426_42626


namespace sqrt_equation_solution_l426_42616

theorem sqrt_equation_solution :
  ∃! x : ℝ, x > 0 ∧ Real.sqrt x ≠ 1 ∧ Real.sqrt x + 1 = 1 / (Real.sqrt x - 1) ∧ x = 2 :=
by sorry

end sqrt_equation_solution_l426_42616


namespace complex_power_problem_l426_42609

theorem complex_power_problem (z : ℂ) (h : z = (1 + Complex.I)^2 / 2) : z^2023 = -Complex.I := by
  sorry

end complex_power_problem_l426_42609


namespace soap_box_dimension_proof_l426_42613

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

theorem soap_box_dimension_proof 
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h1 : carton.length = 25)
  (h2 : carton.width = 48)
  (h3 : carton.height = 60)
  (h4 : soap.width = 6)
  (h5 : soap.height = 5)
  (h6 : (300 : ℝ) * boxVolume soap = boxVolume carton) :
  soap.length = 8 := by
sorry

end soap_box_dimension_proof_l426_42613


namespace at_least_one_positive_discriminant_l426_42681

/-- Given three quadratic polynomials and a condition on their coefficients,
    prove that at least one polynomial has a positive discriminant. -/
theorem at_least_one_positive_discriminant
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ)
  (h : a₁ * a₂ * a₃ = b₁ * b₂ * b₃)
  (h' : b₁ * b₂ * b₃ > 1) :
  (4 * a₁^2 - 4 * b₁ > 0) ∨ (4 * a₂^2 - 4 * b₂ > 0) ∨ (4 * a₃^2 - 4 * b₃ > 0) :=
by sorry

end at_least_one_positive_discriminant_l426_42681


namespace total_notes_count_l426_42657

/-- Given a total amount of 400 rupees in equal numbers of one-rupee, five-rupee, and ten-rupee notes, 
    the total number of notes is 75. -/
theorem total_notes_count (total_amount : ℕ) (note_count : ℕ) : 
  total_amount = 400 →
  note_count * (1 + 5 + 10) = total_amount →
  3 * note_count = 75 := by
  sorry

#check total_notes_count

end total_notes_count_l426_42657


namespace cylinder_volume_l426_42674

/-- Given a cylinder and a cone with specific ratios of heights and base circumferences,
    and a known volume of the cone, prove the volume of the cylinder. -/
theorem cylinder_volume (h_cyl h_cone r_cyl r_cone : ℝ) (vol_cone : ℝ) : 
  h_cyl / h_cone = 4 / 5 →
  r_cyl / r_cone = 3 / 5 →
  vol_cone = 250 →
  (π * r_cyl^2 * h_cyl : ℝ) = 216 := by
sorry

end cylinder_volume_l426_42674


namespace sum_of_seven_consecutive_odds_mod_16_l426_42653

theorem sum_of_seven_consecutive_odds_mod_16 (n : ℕ) (h : n = 12001) :
  (List.sum (List.map (λ i => n + 2 * i) (List.range 7))) % 16 = 1 := by
  sorry

end sum_of_seven_consecutive_odds_mod_16_l426_42653


namespace average_weight_increase_l426_42619

/-- Given a group of 8 people, prove that replacing a person weighing 40 kg with a person weighing 60 kg increases the average weight by 2.5 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 40 + 60
  let new_average := new_total / 8
  new_average - initial_average = 2.5 := by
sorry

end average_weight_increase_l426_42619


namespace factorization_of_2m_squared_minus_2_l426_42621

theorem factorization_of_2m_squared_minus_2 (m : ℝ) : 2 * m^2 - 2 = 2 * (m + 1) * (m - 1) := by
  sorry

end factorization_of_2m_squared_minus_2_l426_42621


namespace skating_average_theorem_l426_42651

/-- Represents Gage's skating schedule --/
structure SkatingSchedule :=
  (days_1 : Nat) (time_1 : Nat)
  (days_2 : Nat) (time_2 : Nat)
  (days_3 : Nat) (time_3 : Nat)
  (days_4 : Nat) (time_4 : Nat)

/-- Calculates the total skating time for 9 days --/
def total_time_9_days (s : SkatingSchedule) : Nat :=
  s.days_1 * s.time_1 + s.days_2 * s.time_2 + s.days_3 * s.time_3 + s.days_4 * s.time_4

/-- Theorem: Skating 85 minutes on the 10th day results in a 90-minute average --/
theorem skating_average_theorem (s : SkatingSchedule) 
  (h1 : s.days_1 = 5 ∧ s.time_1 = 75)
  (h2 : s.days_2 = 3 ∧ s.time_2 = 90)
  (h3 : s.days_3 = 1 ∧ s.time_3 = 120)
  (h4 : s.days_4 = 1 ∧ s.time_4 = 50) :
  (total_time_9_days s + 85) / 10 = 90 := by
  sorry

#check skating_average_theorem

end skating_average_theorem_l426_42651


namespace adblock_interesting_ads_l426_42627

theorem adblock_interesting_ads 
  (total_ads : ℝ) 
  (unblocked_ratio : ℝ) 
  (uninteresting_unblocked_ratio : ℝ) 
  (h1 : unblocked_ratio = 0.2) 
  (h2 : uninteresting_unblocked_ratio = 0.16) : 
  (unblocked_ratio * total_ads - uninteresting_unblocked_ratio * total_ads) / (unblocked_ratio * total_ads) = 0.2 := by
sorry

end adblock_interesting_ads_l426_42627


namespace quadratic_polynomial_satisfies_conditions_l426_42605

-- Define the quadratic polynomial
def q (x : ℚ) : ℚ := (29 * x^2 - 44 * x + 135) / 15

-- State the theorem
theorem quadratic_polynomial_satisfies_conditions :
  q (-1) = 6 ∧ q 2 = 1 ∧ q 4 = 17 := by sorry

end quadratic_polynomial_satisfies_conditions_l426_42605


namespace subtracted_value_l426_42647

theorem subtracted_value (N V : ℝ) (h1 : N = 1152) (h2 : N / 6 - V = 3) : V = 189 := by
  sorry

end subtracted_value_l426_42647


namespace sum_from_difference_and_squares_l426_42668

theorem sum_from_difference_and_squares (x y : ℝ) 
  (h1 : x^2 - y^2 = 21) 
  (h2 : x - y = 3) : 
  x + y = 7 := by
sorry

end sum_from_difference_and_squares_l426_42668


namespace common_roots_product_l426_42697

-- Define the cubic equations
def cubic1 (C : ℝ) (x : ℝ) : ℝ := x^3 + C*x + 20
def cubic2 (D : ℝ) (x : ℝ) : ℝ := x^3 + D*x^2 + 80

-- Define the theorem
theorem common_roots_product (C D : ℝ) (u v : ℝ) :
  (∃ w, cubic1 C u = 0 ∧ cubic1 C v = 0 ∧ cubic1 C w = 0) →
  (∃ t, cubic2 D u = 0 ∧ cubic2 D v = 0 ∧ cubic2 D t = 0) →
  u * v = 10 * Real.rpow 4 (1/3) :=
by sorry

end common_roots_product_l426_42697


namespace problem_solution_l426_42690

theorem problem_solution (x y t : ℝ) 
  (h1 : 2^x = t) 
  (h2 : 7^y = t) 
  (h3 : 1/x + 1/y = 2) : 
  t = Real.sqrt 14 := by
sorry

end problem_solution_l426_42690


namespace sibling_ages_problem_l426_42622

/-- The current age of the eldest sibling given the conditions of the problem -/
def eldest_age : ℕ := 20

theorem sibling_ages_problem :
  let second_age := eldest_age - 5
  let youngest_age := second_age - 5
  let future_sum := (eldest_age + 10) + (second_age + 10) + (youngest_age + 10)
  future_sum = 75 ∧ eldest_age = 20 := by sorry

end sibling_ages_problem_l426_42622


namespace problem_statement_l426_42614

def C : Set ℕ := {x | ∃ s t : ℕ, x = 1999 * s + 2000 * t}

theorem problem_statement :
  (3994001 ∉ C) ∧
  (∀ n : ℕ, 0 ≤ n ∧ n ≤ 3994001 ∧ n ∉ C → (3994001 - n) ∈ C) := by
  sorry

end problem_statement_l426_42614


namespace walter_school_allocation_is_correct_l426_42698

/-- Calculates Walter's school expenses allocation based on his work schedule and earnings --/
def walter_school_allocation (
  fast_food_weekday_hours : ℕ)
  (fast_food_weekend_hours : ℕ)
  (fast_food_hourly_rate : ℚ)
  (fast_food_weekday_days : ℕ)
  (fast_food_weekend_days : ℕ)
  (convenience_store_hours : ℕ)
  (convenience_store_hourly_rate : ℚ)
  (school_allocation_fraction : ℚ) : ℚ :=
let fast_food_weekday_earnings := fast_food_weekday_hours * fast_food_weekday_days * fast_food_hourly_rate
let fast_food_weekend_earnings := fast_food_weekend_hours * fast_food_weekend_days * fast_food_hourly_rate
let convenience_store_earnings := convenience_store_hours * convenience_store_hourly_rate
let total_earnings := fast_food_weekday_earnings + fast_food_weekend_earnings + convenience_store_earnings
school_allocation_fraction * total_earnings

/-- Theorem stating that Walter's school expenses allocation is $146.25 --/
theorem walter_school_allocation_is_correct : 
  walter_school_allocation 4 6 5 5 2 5 7 (3/4) = 146.25 := by
  sorry

end walter_school_allocation_is_correct_l426_42698


namespace revenue_calculation_l426_42696

/-- Calculates the total revenue for a tax center given the prices and number of returns sold for each type of tax service. -/
def total_revenue (federal_price state_price quarterly_price : ℕ) 
                  (federal_sold state_sold quarterly_sold : ℕ) : ℕ :=
  federal_price * federal_sold + state_price * state_sold + quarterly_price * quarterly_sold

/-- Theorem stating that the total revenue for the day is 4400, given the specific prices and number of returns sold. -/
theorem revenue_calculation :
  total_revenue 50 30 80 60 20 10 = 4400 := by
  sorry

end revenue_calculation_l426_42696


namespace decagon_triangle_probability_l426_42608

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with at least one side being a side of the decagon -/
def favorable_triangles : ℕ := 60

/-- The probability of forming a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem decagon_triangle_probability :
  probability = 1 / 2 := by sorry

end decagon_triangle_probability_l426_42608


namespace pencil_distribution_l426_42634

/- Given conditions -/
def total_pencils : ℕ := 10
def num_friends : ℕ := 4

/- Function to calculate binomial coefficient -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/- Theorem statement -/
theorem pencil_distribution :
  binomial (total_pencils - num_friends + num_friends - 1) (num_friends - 1) = 84 :=
by sorry

end pencil_distribution_l426_42634


namespace isosceles_triangle_lengths_l426_42652

/-- An isosceles triangle with a median dividing the perimeter -/
structure IsoscelesTriangleWithMedian where
  leg : ℝ
  base : ℝ
  is_positive : 0 < leg ∧ 0 < base
  is_isosceles : leg > 0
  median_division : 2 * leg + base = 21
  perimeter_division : |2 * leg - base| = 9

/-- The legs of the triangle have length 10 and the base has length 1 -/
theorem isosceles_triangle_lengths (t : IsoscelesTriangleWithMedian) : 
  t.leg = 10 ∧ t.base = 1 := by sorry

end isosceles_triangle_lengths_l426_42652


namespace subset_implies_a_value_l426_42656

theorem subset_implies_a_value (A B : Set ℝ) (a : ℝ) :
  A = {-3} →
  B = {x : ℝ | a * x + 1 = 0} →
  B ⊆ A →
  a = 1/3 := by
  sorry

end subset_implies_a_value_l426_42656
