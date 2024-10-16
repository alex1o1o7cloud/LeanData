import Mathlib

namespace NUMINAMATH_CALUDE_permutations_of_111222_l1631_163145

/-- The number of permutations of a multiset with 6 elements, where 3 elements are of one type
    and 3 elements are of another type. -/
def permutations_of_multiset : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 3)

/-- The theorem states that the number of permutations of the multiset {1, 1, 1, 2, 2, 2}
    is equal to 20. -/
theorem permutations_of_111222 : permutations_of_multiset = 20 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_111222_l1631_163145


namespace NUMINAMATH_CALUDE_beavers_still_working_l1631_163189

theorem beavers_still_working (total : ℕ) (wood : ℕ) (dam : ℕ) (lodge : ℕ)
  (wood_break : ℕ) (dam_break : ℕ) (lodge_break : ℕ)
  (h1 : total = 12)
  (h2 : wood = 5)
  (h3 : dam = 4)
  (h4 : lodge = 3)
  (h5 : wood_break = 3)
  (h6 : dam_break = 2)
  (h7 : lodge_break = 1) :
  (wood - wood_break) + (dam - dam_break) + (lodge - lodge_break) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_beavers_still_working_l1631_163189


namespace NUMINAMATH_CALUDE_jericho_money_left_l1631_163175

/-- The amount of money Jericho has initially -/
def jerichos_money : ℚ := 30

/-- The amount Jericho owes Annika -/
def annika_debt : ℚ := 14

/-- The amount Jericho owes Manny -/
def manny_debt : ℚ := annika_debt / 2

/-- Theorem stating that Jericho will be left with $9 after paying his debts -/
theorem jericho_money_left : jerichos_money - (annika_debt + manny_debt) = 9 := by
  sorry

end NUMINAMATH_CALUDE_jericho_money_left_l1631_163175


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_50_l1631_163187

theorem consecutive_integers_sum_50 : 
  ∃ (x : ℕ), x > 0 ∧ x + (x + 1) + (x + 2) + (x + 3) = 50 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_50_l1631_163187


namespace NUMINAMATH_CALUDE_cosine_graph_transformation_l1631_163165

theorem cosine_graph_transformation (x : ℝ) :
  let f (x : ℝ) := 2 * Real.cos (x + π / 3)
  let g (x : ℝ) := 2 * Real.cos (2 * x + π / 6)
  let h (x : ℝ) := f (2 * x)
  h (x - π / 12) = g x :=
by sorry

end NUMINAMATH_CALUDE_cosine_graph_transformation_l1631_163165


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l1631_163125

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.6 * P) 
  (hP : P ≠ 0) : 
  M / N = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l1631_163125


namespace NUMINAMATH_CALUDE_certain_number_equation_l1631_163144

theorem certain_number_equation (x : ℝ) : 15 * x + 16 * x + 19 * x + 11 = 161 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l1631_163144


namespace NUMINAMATH_CALUDE_division_theorem_l1631_163127

theorem division_theorem (b : ℕ) (hb : b ≠ 0) :
  ∀ n : ℕ, ∃! (q r : ℕ), r < b ∧ n = q * b + r :=
sorry

end NUMINAMATH_CALUDE_division_theorem_l1631_163127


namespace NUMINAMATH_CALUDE_equation_solution_l1631_163111

theorem equation_solution : ∃ x : ℚ, x + 2/5 = 7/10 + 1/2 ∧ x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1631_163111


namespace NUMINAMATH_CALUDE_zodiac_pigeonhole_l1631_163151

/-- The number of Greek Zodiac signs -/
def greek_zodiac_count : ℕ := 12

/-- The number of Chinese Zodiac signs -/
def chinese_zodiac_count : ℕ := 12

/-- The minimum number of people required to ensure at least 3 people have the same Greek Zodiac sign -/
def min_people_same_greek_sign : ℕ := greek_zodiac_count * 2 + 1

/-- The minimum number of people required to ensure at least 2 people have the same combination of Greek and Chinese Zodiac signs -/
def min_people_same_combined_signs : ℕ := greek_zodiac_count * chinese_zodiac_count + 1

theorem zodiac_pigeonhole :
  (min_people_same_greek_sign = 25) ∧
  (min_people_same_combined_signs = 145) := by
  sorry

end NUMINAMATH_CALUDE_zodiac_pigeonhole_l1631_163151


namespace NUMINAMATH_CALUDE_stockholm_malmo_distance_l1631_163152

/-- The road distance between Stockholm and Malmo in kilometers -/
def road_distance (map_distance : ℝ) (scale : ℝ) (road_factor : ℝ) : ℝ :=
  map_distance * scale * road_factor

/-- Theorem: The road distance between Stockholm and Malmo is 1380 km -/
theorem stockholm_malmo_distance :
  road_distance 120 10 1.15 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_malmo_distance_l1631_163152


namespace NUMINAMATH_CALUDE_distance_to_left_focus_l1631_163167

/-- Given an ellipse and a hyperbola, prove that the distance from their intersection point
    in the first quadrant to the left focus of the ellipse is 4. -/
theorem distance_to_left_focus (x y : ℝ) : 
  x > 0 → y > 0 →  -- P is in the first quadrant
  x^2 / 9 + y^2 / 5 = 1 →  -- Ellipse equation
  x^2 - y^2 / 3 = 1 →  -- Hyperbola equation
  ∃ (f₁ : ℝ × ℝ), -- Left focus of the ellipse
    Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_left_focus_l1631_163167


namespace NUMINAMATH_CALUDE_ellipse_trace_l1631_163131

/-- Given a complex number z with |z| = 3, the locus of points (x, y) satisfying z + 2/z = x + yi forms an ellipse -/
theorem ellipse_trace (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), z + 2 / z = x + y * Complex.I ↔ x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_trace_l1631_163131


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1631_163100

theorem complex_equation_solution (z : ℂ) : (1 + 2*I)*z = 4 + 3*I → z = 2 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1631_163100


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l1631_163108

theorem mean_of_added_numbers (original_count : ℕ) (original_mean : ℚ) 
  (new_count : ℕ) (new_mean : ℚ) (added_count : ℕ) : 
  original_count = 8 →
  original_mean = 72 →
  new_count = 11 →
  new_mean = 85 →
  added_count = 3 →
  (new_count * new_mean - original_count * original_mean) / added_count = 119 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l1631_163108


namespace NUMINAMATH_CALUDE_domain_of_composition_l1631_163122

-- Define the function f with domain [1,5]
def f : Set ℝ := Set.Icc 1 5

-- State the theorem
theorem domain_of_composition (f : Set ℝ) (h : f = Set.Icc 1 5) :
  {x : ℝ | ∃ y ∈ f, y = 2*x - 1} = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_domain_of_composition_l1631_163122


namespace NUMINAMATH_CALUDE_imaginary_complex_implies_m_condition_l1631_163114

theorem imaginary_complex_implies_m_condition (m : ℝ) : 
  (Complex.I * (m^2 - 5*m - 6) ≠ 0) → (m ≠ -1 ∧ m ≠ 6) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_complex_implies_m_condition_l1631_163114


namespace NUMINAMATH_CALUDE_difference_of_squares_l1631_163119

theorem difference_of_squares (x y : ℝ) : x^2 - 25*y^2 = (x - 5*y) * (x + 5*y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1631_163119


namespace NUMINAMATH_CALUDE_total_employee_costs_february_l1631_163180

/-- Represents an employee in the car dealership -/
structure Employee where
  name : String
  hoursPerWeek : Nat
  hourlyRate : Nat
  weeksWorked : Nat
  overtime : Nat
  overtimeRate : Nat
  bonus : Int
  deduction : Nat

/-- Calculates the monthly earnings for an employee -/
def monthlyEarnings (e : Employee) : Int :=
  e.hoursPerWeek * e.hourlyRate * e.weeksWorked +
  e.overtime * e.overtimeRate +
  e.bonus -
  e.deduction

/-- Theorem stating the total employee costs for February -/
theorem total_employee_costs_february :
  let fiona : Employee := ⟨"Fiona", 40, 20, 3, 0, 0, 0, 0⟩
  let john : Employee := ⟨"John", 30, 22, 4, 10, 33, 0, 0⟩
  let jeremy : Employee := ⟨"Jeremy", 25, 18, 4, 0, 0, 200, 0⟩
  let katie : Employee := ⟨"Katie", 35, 21, 4, 0, 0, 0, 150⟩
  let matt : Employee := ⟨"Matt", 28, 19, 4, 0, 0, 0, 0⟩
  monthlyEarnings fiona + monthlyEarnings john + monthlyEarnings jeremy +
  monthlyEarnings katie + monthlyEarnings matt = 13278 := by
  sorry


end NUMINAMATH_CALUDE_total_employee_costs_february_l1631_163180


namespace NUMINAMATH_CALUDE_race_outcomes_count_l1631_163164

/-- Represents the number of participants in the race -/
def num_participants : ℕ := 6

/-- Represents the number of top positions we're considering -/
def num_top_positions : ℕ := 4

/-- Calculates the number of permutations of k items chosen from n items -/
def permutations (n k : ℕ) : ℕ := 
  if k > n then 0
  else (List.range n).foldr (λ i acc => (i + 1) * acc) 1

/-- Theorem stating the number of possible race outcomes -/
theorem race_outcomes_count : 
  (permutations (num_participants - 1) (num_top_positions - 1)) * num_participants - 
  (permutations (num_participants - 1) (num_top_positions - 1)) = 300 := by
  sorry


end NUMINAMATH_CALUDE_race_outcomes_count_l1631_163164


namespace NUMINAMATH_CALUDE_sum_of_ages_is_twelve_l1631_163139

/-- The sum of ages of four children born at one-year intervals -/
def sum_of_ages (youngest_age : ℝ) : ℝ :=
  youngest_age + (youngest_age + 1) + (youngest_age + 2) + (youngest_age + 3)

/-- Theorem: The sum of ages of four children, where the youngest is 1.5 years old
    and each subsequent child is 1 year older, is 12 years. -/
theorem sum_of_ages_is_twelve :
  sum_of_ages 1.5 = 12 := by sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_twelve_l1631_163139


namespace NUMINAMATH_CALUDE_expression_value_l1631_163177

theorem expression_value (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 1 / (a^2 + 1) + 1 / (b^2 + 1) = 2 / (a * b + 1)) :
  1 / (a^2 + 1) + 1 / (b^2 + 1) + 2 / (a * b + 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1631_163177


namespace NUMINAMATH_CALUDE_super_champion_tournament_24_teams_l1631_163120

/-- The number of games played in a tournament with a given number of teams --/
def tournament_games (n : ℕ) : ℕ :=
  n - 1

/-- The total number of games in a tournament with a "Super Champion" game --/
def super_champion_tournament (n : ℕ) : ℕ :=
  tournament_games n + 1

/-- Theorem: A tournament with 24 teams and a "Super Champion" game has 24 total games --/
theorem super_champion_tournament_24_teams :
  super_champion_tournament 24 = 24 := by
  sorry

end NUMINAMATH_CALUDE_super_champion_tournament_24_teams_l1631_163120


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l1631_163133

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Defines a line in 2D space using the equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : IntersectionPoint) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- The theorem stating that (2, 1) is the unique intersection point of y = 3 - x and y = 3x - 5 -/
theorem intersection_point_of_lines :
  ∃! p : IntersectionPoint, 
    (pointOnLine p ⟨-1, 3⟩) ∧ (pointOnLine p ⟨3, -5⟩) ∧ p.x = 2 ∧ p.y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l1631_163133


namespace NUMINAMATH_CALUDE_johns_father_age_difference_l1631_163150

/-- Given John's age and the sum of John and his father's ages, 
    prove the difference between John's father's age and twice John's age. -/
theorem johns_father_age_difference (john_age : ℕ) (sum_ages : ℕ) 
    (h1 : john_age = 15)
    (h2 : john_age + (2 * john_age + sum_ages - john_age) = sum_ages)
    (h3 : sum_ages = 77) : 
  (2 * john_age + sum_ages - john_age) - 2 * john_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_johns_father_age_difference_l1631_163150


namespace NUMINAMATH_CALUDE_square_area_ratio_l1631_163172

theorem square_area_ratio (s : ℝ) (h : s > 0) : 
  (s^2) / ((s * Real.sqrt 5)^2) = 1/5 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1631_163172


namespace NUMINAMATH_CALUDE_total_running_time_l1631_163121

def track_length : ℝ := 500
def num_laps : ℕ := 7
def first_section_length : ℝ := 200
def second_section_length : ℝ := 300
def first_section_speed : ℝ := 5
def second_section_speed : ℝ := 6

theorem total_running_time :
  (num_laps : ℝ) * (first_section_length / first_section_speed + second_section_length / second_section_speed) = 630 :=
by sorry

end NUMINAMATH_CALUDE_total_running_time_l1631_163121


namespace NUMINAMATH_CALUDE_solution_set_implies_k_empty_solution_set_implies_k_range_l1631_163193

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2*x + 3*k

-- Part 1
theorem solution_set_implies_k (k : ℝ) :
  (∀ x, f k x < 0 ↔ x < -3 ∨ x > -1) → k = -1/2 := by sorry

-- Part 2
theorem empty_solution_set_implies_k_range (k : ℝ) :
  (∀ x, ¬(f k x < 0)) → 0 < k ∧ k ≤ Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_k_empty_solution_set_implies_k_range_l1631_163193


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l1631_163146

theorem imaginary_part_of_complex_product : Complex.im ((4 - 8 * Complex.I) * Complex.I) = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l1631_163146


namespace NUMINAMATH_CALUDE_complex_number_location_l1631_163132

theorem complex_number_location :
  let z : ℂ := 1 / (3 + Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l1631_163132


namespace NUMINAMATH_CALUDE_log_sum_equality_l1631_163163

theorem log_sum_equality : Real.log 8 / Real.log 10 + 3 * (Real.log 5 / Real.log 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1631_163163


namespace NUMINAMATH_CALUDE_sqrt_sum_given_diff_l1631_163106

theorem sqrt_sum_given_diff (y : ℝ) : 
  Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4 → 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_given_diff_l1631_163106


namespace NUMINAMATH_CALUDE_constant_term_is_160_l1631_163170

/-- The constant term in the binomial expansion of (x + 2/x)^6 -/
def constant_term : ℕ :=
  (Nat.choose 6 3) * (2^3)

/-- Theorem: The constant term in the binomial expansion of (x + 2/x)^6 is 160 -/
theorem constant_term_is_160 : constant_term = 160 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_160_l1631_163170


namespace NUMINAMATH_CALUDE_max_value_expression_l1631_163183

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 6*x^2 + 1)) / x ≤ 2/3 ∧
  ∃ x₀ > 0, (x₀^2 + 3 - Real.sqrt (x₀^4 + 6*x₀^2 + 1)) / x₀ = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1631_163183


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1631_163178

def f (x : ℤ) : ℤ := x^3 - 4*x^2 - 14*x + 24

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, f x = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1631_163178


namespace NUMINAMATH_CALUDE_pythagorean_field_planting_l1631_163135

theorem pythagorean_field_planting (a b : ℝ) (h1 : a = 5) (h2 : b = 12) : 
  let c := Real.sqrt (a^2 + b^2)
  let x := (a * b) / c
  let triangle_area := (a * b) / 2
  let square_area := x^2
  let planted_area := triangle_area - square_area
  let shortest_distance := (2 * square_area) / c
  shortest_distance = 3 → planted_area / triangle_area = 792 / 845 := by
sorry


end NUMINAMATH_CALUDE_pythagorean_field_planting_l1631_163135


namespace NUMINAMATH_CALUDE_water_amount_depends_on_time_l1631_163134

/-- Represents the water amount in the reservoir -/
def water_amount (t : ℝ) : ℝ := 50 - 2 * t

/-- States that water_amount is a function of time -/
theorem water_amount_depends_on_time :
  ∃ (f : ℝ → ℝ), ∀ t, water_amount t = f t :=
sorry

end NUMINAMATH_CALUDE_water_amount_depends_on_time_l1631_163134


namespace NUMINAMATH_CALUDE_min_value_theorem_l1631_163160

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 2 / b = 1) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1 / x + 2 / y = 1 → 
  2 / (x - 1) + 1 / (y - 2) ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1631_163160


namespace NUMINAMATH_CALUDE_danny_found_seven_bottle_caps_l1631_163142

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found (initial_count : ℕ) (final_count : ℕ) : ℕ :=
  final_count - initial_count

/-- Theorem stating that Danny found 7 bottle caps at the park -/
theorem danny_found_seven_bottle_caps :
  bottle_caps_found 25 32 = 7 := by
  sorry

end NUMINAMATH_CALUDE_danny_found_seven_bottle_caps_l1631_163142


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_A_l1631_163118

theorem partial_fraction_decomposition_A (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ 1 →
    1 / (x^3 - 2*x^2 - 13*x + 10) = A / (x + 2) + B / (x - 1) + C / ((x - 1)^2)) →
  A = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_A_l1631_163118


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1631_163190

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > (1/2) → 2*x^2 + x - 1 > 0) ∧
  (∃ x, 2*x^2 + x - 1 > 0 ∧ x ≤ (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1631_163190


namespace NUMINAMATH_CALUDE_trig_expression_value_l1631_163169

theorem trig_expression_value (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l1631_163169


namespace NUMINAMATH_CALUDE_solve_system1_solve_system2_l1631_163129

-- First system of equations
theorem solve_system1 (x y : ℝ) : 
  2 * x + 3 * y = 7 ∧ x = -2 * y + 3 → x = 5 ∧ y = -1 := by sorry

-- Second system of equations
theorem solve_system2 (x y : ℝ) : 
  5 * x + y = 4 ∧ 2 * x - 3 * y = 5 → x = 1 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_solve_system1_solve_system2_l1631_163129


namespace NUMINAMATH_CALUDE_length_of_PQ_l1631_163110

/-- The problem setup -/
structure ProblemSetup where
  /-- Point R with coordinates (10, 15) -/
  R : ℝ × ℝ
  hR : R = (10, 15)
  
  /-- Line 1 with equation 7y = 24x -/
  line1 : ℝ → ℝ
  hline1 : ∀ x y, line1 y = 24 * x ∧ 7 * y = 24 * x
  
  /-- Line 2 with equation 15y = 4x -/
  line2 : ℝ → ℝ
  hline2 : ∀ x y, line2 y = 4/15 * x ∧ 15 * y = 4 * x
  
  /-- Point P on Line 1 -/
  P : ℝ × ℝ
  hP : line1 P.2 = 24 * P.1 ∧ 7 * P.2 = 24 * P.1
  
  /-- Point Q on Line 2 -/
  Q : ℝ × ℝ
  hQ : line2 Q.2 = 4/15 * Q.1 ∧ 15 * Q.2 = 4 * Q.1
  
  /-- R is the midpoint of PQ -/
  hMidpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

/-- The main theorem -/
theorem length_of_PQ (setup : ProblemSetup) : 
  Real.sqrt ((setup.P.1 - setup.Q.1)^2 + (setup.P.2 - setup.Q.2)^2) = 3460 / 83 := by
  sorry

end NUMINAMATH_CALUDE_length_of_PQ_l1631_163110


namespace NUMINAMATH_CALUDE_todds_time_is_correct_l1631_163197

/-- Todd's running time around the track -/
def todds_time : ℕ := 88

/-- Brian's running time around the track -/
def brians_time : ℕ := 96

/-- The difference in running time between Brian and Todd -/
def time_difference : ℕ := 8

/-- Theorem stating that Todd's time is correct given the conditions -/
theorem todds_time_is_correct : todds_time = brians_time - time_difference := by
  sorry

end NUMINAMATH_CALUDE_todds_time_is_correct_l1631_163197


namespace NUMINAMATH_CALUDE_work_completion_theorem_l1631_163116

def work_completion_time (rate_A rate_B rate_C : ℚ) (initial_days : ℕ) : ℚ :=
  let combined_rate_AB := rate_A + rate_B
  let work_done_AB := combined_rate_AB * initial_days
  let remaining_work := 1 - work_done_AB
  let combined_rate_AC := rate_A + rate_C
  initial_days + remaining_work / combined_rate_AC

theorem work_completion_theorem :
  let rate_A : ℚ := 1 / 30
  let rate_B : ℚ := 1 / 15
  let rate_C : ℚ := 1 / 20
  let initial_days : ℕ := 5
  work_completion_time rate_A rate_B rate_C initial_days = 11 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l1631_163116


namespace NUMINAMATH_CALUDE_decreasing_reciprocal_function_l1631_163149

theorem decreasing_reciprocal_function (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : 0 < x₂) (h3 : x₁ < x₂) :
  (1 : ℝ) / x₁ > (1 : ℝ) / x₂ := by
  sorry

end NUMINAMATH_CALUDE_decreasing_reciprocal_function_l1631_163149


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_at_midpoint_l1631_163199

theorem line_intersects_ellipse_at_midpoint (x y : ℝ) :
  let P : ℝ × ℝ := (1, 1)
  let ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1
  let line (x y : ℝ) : Prop := 4*x + 9*y = 13
  (∀ x y, line x y → (x, y) = P ∨ ellipse x y) ∧
  (∃ A B : ℝ × ℝ, A ≠ B ∧ line A.1 A.2 ∧ line B.1 B.2 ∧ 
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
    ((A.1 + B.1)/2, (A.2 + B.2)/2) = P) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_at_midpoint_l1631_163199


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1631_163105

/-- The parabola y = 2x^2 intersects the line y = x - 4 at exactly one point when
    shifted right by p units or down by q units, where p = q = 31/8 -/
theorem parabola_line_intersection (p q : ℝ) : 
  (∀ x y : ℝ, y = 2*(x - p)^2 ∧ y = x - 4 → (∃! z : ℝ, z = x)) ∧
  (∀ x y : ℝ, y = 2*x^2 - q ∧ y = x - 4 → (∃! z : ℝ, z = x)) →
  p = 31/8 ∧ q = 31/8 := by
sorry


end NUMINAMATH_CALUDE_parabola_line_intersection_l1631_163105


namespace NUMINAMATH_CALUDE_visitor_growth_rate_l1631_163168

theorem visitor_growth_rate (initial_visitors : ℝ) (final_visitors : ℝ) (x : ℝ) : 
  initial_visitors = 42 → 
  final_visitors = 133.91 → 
  initial_visitors * (1 + x)^2 = final_visitors :=
by sorry

end NUMINAMATH_CALUDE_visitor_growth_rate_l1631_163168


namespace NUMINAMATH_CALUDE_twin_brothers_age_l1631_163184

theorem twin_brothers_age :
  ∀ (x : ℕ), 
    (x * x + 9 = (x + 1) * (x + 1)) → 
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_twin_brothers_age_l1631_163184


namespace NUMINAMATH_CALUDE_box_volume_from_face_centers_l1631_163109

def rectangular_box_volume (a b c : ℝ) : ℝ := 8 * a * b * c

theorem box_volume_from_face_centers 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + b^2 = 4^2)
  (h2 : b^2 + c^2 = 5^2)
  (h3 : a^2 + c^2 = 6^2) :
  rectangular_box_volume a b c = 90 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_box_volume_from_face_centers_l1631_163109


namespace NUMINAMATH_CALUDE_prime_divides_sum_l1631_163155

theorem prime_divides_sum (a b c : ℕ+) (p : ℕ) 
  (h1 : a ^ 3 + 4 * b + c = a * b * c)
  (h2 : a ≥ c)
  (h3 : p = a ^ 2 + 2 * a + 2)
  (h4 : Nat.Prime p) :
  p ∣ (a + 2 * b + 2) := by
sorry

end NUMINAMATH_CALUDE_prime_divides_sum_l1631_163155


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_equations_l1631_163162

/-- A circle with center C on the line x - y + 1 = 0 passing through points (1, 1) and (2, -2) -/
structure CircleC where
  center : ℝ × ℝ
  center_on_line : center.1 - center.2 + 1 = 0
  passes_through_A : (center.1 - 1)^2 + (center.2 - 1)^2 = (center.1 - 2)^2 + (center.2 + 2)^2

/-- The standard equation of the circle and its tangent line -/
def circle_equation (c : CircleC) : Prop :=
  ∀ (x y : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = (c.center.1 - 1)^2 + (c.center.2 - 1)^2

def tangent_line_equation (c : CircleC) : Prop :=
  ∀ (x y : ℝ), 4*x + 3*y - 7 = 0 ↔ 
    ((x - 1) * (c.center.1 - 1) + (y - 1) * (c.center.2 - 1) = (c.center.1 - 1)^2 + (c.center.2 - 1)^2) ∧
    ((x, y) ≠ (1, 1))

/-- The main theorem stating that the circle equation and tangent line equation are correct -/
theorem circle_and_tangent_line_equations (c : CircleC) : 
  circle_equation c ∧ tangent_line_equation c :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_equations_l1631_163162


namespace NUMINAMATH_CALUDE_PQRS_equals_one_l1631_163171

theorem PQRS_equals_one :
  let P := Real.sqrt 2010 + Real.sqrt 2011
  let Q := -Real.sqrt 2010 - Real.sqrt 2011
  let R := Real.sqrt 2010 - Real.sqrt 2011
  let S := Real.sqrt 2011 - Real.sqrt 2010
  P * Q * R * S = 1 := by
sorry

end NUMINAMATH_CALUDE_PQRS_equals_one_l1631_163171


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1631_163182

theorem rationalize_denominator :
  (Real.sqrt 18 + Real.sqrt 2) / (Real.sqrt 3 + Real.sqrt 2) = 4 * (Real.sqrt 6 - 2) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1631_163182


namespace NUMINAMATH_CALUDE_f_two_range_l1631_163147

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the theorem
theorem f_two_range (a b c : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ 
   f a b c 1 = 0 ∧ f a b c x₁ = 0 ∧ f a b c x₂ = 0) →
  (∀ x : ℝ, f a b c (x + 1) = -f a b c (-x + 1)) →
  0 < f a b c 2 ∧ f a b c 2 < 1 :=
by sorry

end NUMINAMATH_CALUDE_f_two_range_l1631_163147


namespace NUMINAMATH_CALUDE_small_triangle_perimeter_is_ten_l1631_163191

/-- Represents a triangle with cuts parallel to its sides -/
structure CutTriangle where
  perimeter : ℝ
  trapezoid1_perimeter : ℝ
  trapezoid2_perimeter : ℝ
  trapezoid3_perimeter : ℝ

/-- The perimeter of the small triangle formed by the cuts -/
def small_triangle_perimeter (t : CutTriangle) : ℝ :=
  t.trapezoid1_perimeter + t.trapezoid2_perimeter + t.trapezoid3_perimeter - t.perimeter

/-- Theorem: The perimeter of the small triangle is 10 -/
theorem small_triangle_perimeter_is_ten (t : CutTriangle)
  (h1 : t.perimeter = 11)
  (h2 : t.trapezoid1_perimeter = 5)
  (h3 : t.trapezoid2_perimeter = 7)
  (h4 : t.trapezoid3_perimeter = 9) :
  small_triangle_perimeter t = 10 := by
  sorry

end NUMINAMATH_CALUDE_small_triangle_perimeter_is_ten_l1631_163191


namespace NUMINAMATH_CALUDE_sam_bought_one_lollipop_l1631_163143

/-- Calculates the number of lollipops Sam bought -/
def lollipops_bought (initial_dimes : ℕ) (initial_quarters : ℕ) (candy_bars : ℕ) 
  (dimes_per_candy : ℕ) (cents_per_lollipop : ℕ) (cents_left : ℕ) : ℕ :=
  let initial_cents := initial_dimes * 10 + initial_quarters * 25
  let candy_cost := candy_bars * dimes_per_candy * 10
  let cents_for_lollipops := initial_cents - candy_cost - cents_left
  cents_for_lollipops / cents_per_lollipop

theorem sam_bought_one_lollipop :
  lollipops_bought 19 6 4 3 25 195 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sam_bought_one_lollipop_l1631_163143


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1631_163157

/-- Given a circle C with equation x^2 + 12y + 57 = -y^2 - 10x, 
    prove that the sum of its center coordinates and radius is -9 -/
theorem circle_center_radius_sum (x y : ℝ) :
  (∃ (a b r : ℝ), 
    (∀ x y : ℝ, x^2 + 12*y + 57 = -y^2 - 10*x ↔ (x - a)^2 + (y - b)^2 = r^2) →
    a + b + r = -9) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1631_163157


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l1631_163181

theorem decimal_to_fraction_sum (x : ℚ) (n d : ℕ) (v : ℕ) : 
  x = 2.52 →
  x = n / d →
  (∀ k : ℕ, k > 1 → ¬(k ∣ n ∧ k ∣ d)) →
  n + v = 349 →
  v = 286 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l1631_163181


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l1631_163159

/-- The sum of divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number n -/
def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l1631_163159


namespace NUMINAMATH_CALUDE_checkerboard_probability_l1631_163173

/-- The size of one side of the square checkerboard -/
def board_size : ℕ := 10

/-- The total number of squares on the checkerboard -/
def total_squares : ℕ := board_size * board_size

/-- The number of squares on the perimeter of the checkerboard -/
def perimeter_squares : ℕ := 4 * board_size - 4

/-- The number of squares not on the perimeter of the checkerboard -/
def inner_squares : ℕ := total_squares - perimeter_squares

/-- The probability of choosing a square not on the perimeter -/
def inner_square_probability : ℚ := inner_squares / total_squares

theorem checkerboard_probability :
  inner_square_probability = 16 / 25 := by sorry

end NUMINAMATH_CALUDE_checkerboard_probability_l1631_163173


namespace NUMINAMATH_CALUDE_sum_remainder_mod_9_l1631_163112

theorem sum_remainder_mod_9 : (98134 + 98135 + 98136 + 98137 + 98138 + 98139) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_9_l1631_163112


namespace NUMINAMATH_CALUDE_sum_of_roots_l1631_163186

theorem sum_of_roots (y₁ y₂ k m : ℝ) (h1 : y₁ ≠ y₂) 
  (h2 : 5 * y₁^2 - k * y₁ = m) (h3 : 5 * y₂^2 - k * y₂ = m) : 
  y₁ + y₂ = k / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1631_163186


namespace NUMINAMATH_CALUDE_complement_of_intersection_equals_expected_l1631_163153

-- Define the sets M and N
def M : Set ℝ := {x | x ≥ 1/3}
def N : Set ℝ := {x | 0 < x ∧ x < 1/2}

-- Define the complement of the intersection
def complementOfIntersection : Set ℝ := {x | x < 1/3 ∨ x ≥ 1/2}

-- Theorem statement
theorem complement_of_intersection_equals_expected :
  complementOfIntersection = (Set.Iic (1/3 : ℝ)).diff {1/3} ∪ Set.Ici (1/2 : ℝ) := by
  sorry

#check complement_of_intersection_equals_expected

end NUMINAMATH_CALUDE_complement_of_intersection_equals_expected_l1631_163153


namespace NUMINAMATH_CALUDE_bisector_sum_squares_l1631_163128

/-- Given a triangle with side lengths a and b, angle C, and its angle bisector l and
    exterior angle bisector l', the sum of squares of these bisectors is equal to
    (64 R^2 S^2) / ((a^2 - b^2)^2), where R is the circumradius and S is the area of the triangle. -/
theorem bisector_sum_squares (a b l l' R S : ℝ) (ha : 0 < a) (hb : 0 < b) (hl : 0 < l) (hl' : 0 < l') (hR : 0 < R) (hS : 0 < S) :
  l'^2 + l^2 = (64 * R^2 * S^2) / ((a^2 - b^2)^2) := by
  sorry

end NUMINAMATH_CALUDE_bisector_sum_squares_l1631_163128


namespace NUMINAMATH_CALUDE_thabo_book_difference_l1631_163185

/-- Represents the number of books Thabo owns of each type -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- The properties of Thabo's book collection -/
def validCollection (books : BookCollection) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 280 ∧
  books.paperbackNonfiction > books.hardcoverNonfiction ∧
  books.paperbackFiction = 2 * books.paperbackNonfiction ∧
  books.hardcoverNonfiction = 55

theorem thabo_book_difference (books : BookCollection) 
  (h : validCollection books) : 
  books.paperbackNonfiction - books.hardcoverNonfiction = 20 := by
  sorry

end NUMINAMATH_CALUDE_thabo_book_difference_l1631_163185


namespace NUMINAMATH_CALUDE_stock_price_increase_l1631_163158

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (percent_increase : ℝ) (h1 : closing_price = 9) 
  (h2 : percent_increase = 12.5) :
  closing_price = opening_price * (1 + percent_increase / 100) →
  opening_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l1631_163158


namespace NUMINAMATH_CALUDE_buyer_ratio_l1631_163124

/-- Represents the number of buyers on a given day -/
structure BuyerCount where
  count : ℕ

/-- Represents the buyer counts for three consecutive days -/
structure ThreeDayBuyers where
  dayBeforeYesterday : BuyerCount
  yesterday : BuyerCount
  today : BuyerCount

/-- The conditions given in the problem -/
def storeConditions (buyers : ThreeDayBuyers) : Prop :=
  buyers.today.count = buyers.yesterday.count + 40 ∧
  buyers.dayBeforeYesterday.count + buyers.yesterday.count + buyers.today.count = 140 ∧
  buyers.dayBeforeYesterday.count = 50

/-- The theorem to prove -/
theorem buyer_ratio (buyers : ThreeDayBuyers) 
  (h : storeConditions buyers) : 
  buyers.yesterday.count * 2 = buyers.dayBeforeYesterday.count := by
  sorry


end NUMINAMATH_CALUDE_buyer_ratio_l1631_163124


namespace NUMINAMATH_CALUDE_k_eval_at_one_l1631_163115

-- Define the polynomials h and k
def h (p : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + 2*x + 15
def k (q r : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + q*x^2 + 150*x + r

-- State the theorem
theorem k_eval_at_one (p q r : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ h p x = 0 ∧ h p y = 0 ∧ h p z = 0) →  -- h has three distinct roots
  (∀ x : ℝ, h p x = 0 → k q r x = 0) →  -- each root of h is a root of k
  k q r 1 = -3322.25 := by
sorry

end NUMINAMATH_CALUDE_k_eval_at_one_l1631_163115


namespace NUMINAMATH_CALUDE_daniels_cats_l1631_163174

theorem daniels_cats (horses dogs turtles goats cats : ℕ) : 
  horses = 2 → 
  dogs = 5 → 
  turtles = 3 → 
  goats = 1 → 
  4 * (horses + dogs + cats + turtles + goats) = 72 → 
  cats = 7 := by
sorry

end NUMINAMATH_CALUDE_daniels_cats_l1631_163174


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1631_163140

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that in an arithmetic sequence {aₙ} where
    a₅ + a₆ = 16 and a₈ = 12, the third term a₃ equals 4. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 5 + a 6 = 16)
    (h_eighth : a 8 = 12) : 
  a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1631_163140


namespace NUMINAMATH_CALUDE_yazhong_point_problem1_yazhong_point_problem2_yazhong_point_problem3_1_yazhong_point_problem3_2_l1631_163113

/-- Definition of Yazhong point -/
def is_yazhong_point (a b m : ℝ) : Prop :=
  |m - a| = |m - b|

/-- Problem 1 -/
theorem yazhong_point_problem1 :
  is_yazhong_point (-5) 1 (-2) :=
sorry

/-- Problem 2 -/
theorem yazhong_point_problem2 :
  is_yazhong_point (-5/2) (13/2) 2 ∧ |(-5/2) - (13/2)| = 9 :=
sorry

/-- Problem 3 part 1 -/
theorem yazhong_point_problem3_1 :
  (∃ b : ℝ, -4 ≤ b ∧ b ≤ -2 ∧ is_yazhong_point (-6) b (-5)) ∧
  (∃ b : ℝ, -4 ≤ b ∧ b ≤ -2 ∧ is_yazhong_point (-6) b (-4)) ∧
  (∀ m : ℤ, (∃ b : ℝ, -4 ≤ b ∧ b ≤ -2 ∧ is_yazhong_point (-6) b (m : ℝ)) → m = -5 ∨ m = -4) :=
sorry

/-- Problem 3 part 2 -/
theorem yazhong_point_problem3_2 :
  (∀ n : ℤ, is_yazhong_point (-6) (6 : ℝ) 0 ∧ -4 + n ≤ 6 ∧ 6 ≤ -2 + n → n = 8 ∨ n = 9 ∨ n = 10) ∧
  (∀ n : ℤ, n = 8 ∨ n = 9 ∨ n = 10 → is_yazhong_point (-6) (6 : ℝ) 0 ∧ -4 + n ≤ 6 ∧ 6 ≤ -2 + n) :=
sorry

end NUMINAMATH_CALUDE_yazhong_point_problem1_yazhong_point_problem2_yazhong_point_problem3_1_yazhong_point_problem3_2_l1631_163113


namespace NUMINAMATH_CALUDE_average_screen_time_l1631_163101

/-- Calculates the average screen time per player in minutes given the screen times for 5 players in seconds -/
theorem average_screen_time (point_guard shooting_guard small_forward power_forward center : ℕ) 
  (h1 : point_guard = 130)
  (h2 : shooting_guard = 145)
  (h3 : small_forward = 85)
  (h4 : power_forward = 60)
  (h5 : center = 180) :
  (point_guard + shooting_guard + small_forward + power_forward + center) / (5 * 60) = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_screen_time_l1631_163101


namespace NUMINAMATH_CALUDE_triangular_prism_volume_l1631_163192

/-- The volume of a triangular prism with a right triangle base and specific conditions -/
theorem triangular_prism_volume (PQ PR h θ : ℝ) : 
  PQ = Real.sqrt 5 →
  PR = Real.sqrt 5 →
  Real.tan θ = h / Real.sqrt 5 →
  Real.sin θ = 3 / 5 →
  (1 / 2 * PQ * PR) * h = 15 * Real.sqrt 5 / 8 := by
  sorry

#check triangular_prism_volume

end NUMINAMATH_CALUDE_triangular_prism_volume_l1631_163192


namespace NUMINAMATH_CALUDE_initial_owls_count_l1631_163130

theorem initial_owls_count (initial_owls final_owls joined_owls : ℕ) : 
  initial_owls + joined_owls = final_owls →
  joined_owls = 2 →
  final_owls = 5 →
  initial_owls = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_owls_count_l1631_163130


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l1631_163161

def triangle_sides (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem similar_triangle_longest_side 
  (a b c : ℝ) 
  (h_triangle : triangle_sides a b c) 
  (h_sides : a = 5 ∧ b = 12 ∧ c = 13) 
  (k : ℝ) 
  (h_similar : k > 0)
  (h_perimeter : k * (a + b + c) = 150) :
  k * max a (max b c) = 65 := by
sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_l1631_163161


namespace NUMINAMATH_CALUDE_expression_simplification_l1631_163104

theorem expression_simplification (x : ℝ) (h : x = 1) :
  (2 * x) / (x + 2) - x / (x - 2) + (4 * x) / (x^2 - 4) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1631_163104


namespace NUMINAMATH_CALUDE_shaded_area_is_32_5_l1631_163166

/-- Represents a rectangular grid -/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Represents a right-angled triangle -/
structure RightTriangle where
  base : ℕ
  height : ℕ

/-- Calculates the area of a shaded region in a grid, excluding a right-angled triangle -/
def shadedArea (g : Grid) (t : RightTriangle) : ℚ :=
  (g.rows * g.cols : ℚ) - (t.base * t.height : ℚ) / 2

/-- Theorem stating that the shaded area in the given problem is 32.5 square units -/
theorem shaded_area_is_32_5 :
  let g : Grid := ⟨4, 13⟩
  let t : RightTriangle := ⟨13, 3⟩
  shadedArea g t = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_32_5_l1631_163166


namespace NUMINAMATH_CALUDE_arithmetic_progression_result_l1631_163123

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℕ → ℝ  -- The nth term of the progression
  S : ℕ → ℝ  -- The sum of the first n terms

/-- Theorem stating the result for the given arithmetic progression -/
theorem arithmetic_progression_result (ap : ArithmeticProgression) 
  (h1 : ap.a 1 + ap.a 3 = 5)
  (h2 : ap.S 4 = 20) :
  (ap.S 8 - 2 * ap.S 4) / (ap.S 6 - ap.S 4 - ap.S 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_result_l1631_163123


namespace NUMINAMATH_CALUDE_surprise_combinations_for_week_l1631_163138

/-- The number of combinations for surprise gift placement --/
def surprise_combinations (monday tuesday wednesday thursday friday : ℕ) : ℕ :=
  monday * tuesday * wednesday * thursday * friday

/-- Theorem stating the total number of combinations for the given week --/
theorem surprise_combinations_for_week :
  surprise_combinations 2 1 1 4 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_surprise_combinations_for_week_l1631_163138


namespace NUMINAMATH_CALUDE_regular_tetrahedron_properties_regular_tetrahedron_all_properties_l1631_163141

/-- Definition of a regular tetrahedron -/
structure RegularTetrahedron where
  /-- All edges of the tetrahedron are equal -/
  edges_equal : Bool
  /-- All faces of the tetrahedron are congruent equilateral triangles -/
  faces_congruent : Bool
  /-- The angle between any two edges at the same vertex is equal -/
  vertex_angles_equal : Bool
  /-- The dihedral angle between any two adjacent faces is equal -/
  dihedral_angles_equal : Bool

/-- Theorem: Properties of a regular tetrahedron -/
theorem regular_tetrahedron_properties (t : RegularTetrahedron) : 
  t.edges_equal ∧ 
  t.faces_congruent ∧ 
  t.vertex_angles_equal ∧ 
  t.dihedral_angles_equal := by
  sorry

/-- Corollary: All three properties mentioned in the problem are true for a regular tetrahedron -/
theorem regular_tetrahedron_all_properties (t : RegularTetrahedron) :
  (t.edges_equal ∧ t.vertex_angles_equal) ∧
  (t.faces_congruent ∧ t.dihedral_angles_equal) ∧
  (t.faces_congruent ∧ t.vertex_angles_equal) := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_properties_regular_tetrahedron_all_properties_l1631_163141


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1631_163154

theorem arithmetic_equality : 4 * 8 + 5 * 11 - 2 * 3 + 7 * 9 = 144 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1631_163154


namespace NUMINAMATH_CALUDE_min_value_function_l1631_163176

theorem min_value_function (x : ℝ) (h : x > 1) : 
  ∀ y : ℝ, y = 4 / (x - 1) + x → y ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_function_l1631_163176


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l1631_163107

/-- Proves that the 8th term of an arithmetic sequence with 26 terms, 
    first term 4, and last term 104, is equal to 32. -/
theorem arithmetic_sequence_8th_term 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 4) 
  (h2 : a 26 = 104) 
  (h3 : ∀ n : ℕ, 1 < n → n ≤ 26 → a n - a (n-1) = a 2 - a 1) :
  a 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l1631_163107


namespace NUMINAMATH_CALUDE_min_value_on_line_l1631_163196

theorem min_value_on_line (x y : ℝ) : 
  x + y = 4 → ∀ a b : ℝ, a + b = 4 → x^2 + y^2 ≤ a^2 + b^2 ∧ ∃ c d : ℝ, c + d = 4 ∧ c^2 + d^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_l1631_163196


namespace NUMINAMATH_CALUDE_student_ticket_price_l1631_163117

/-- The price of a senior citizen ticket -/
def senior_price : ℝ := sorry

/-- The price of a student ticket -/
def student_price : ℝ := sorry

/-- First day sales equation -/
axiom first_day_sales : 4 * senior_price + 3 * student_price = 79

/-- Second day sales equation -/
axiom second_day_sales : 12 * senior_price + 10 * student_price = 246

/-- Theorem stating that the student ticket price is 9 dollars -/
theorem student_ticket_price : student_price = 9 := by sorry

end NUMINAMATH_CALUDE_student_ticket_price_l1631_163117


namespace NUMINAMATH_CALUDE_expression_simplification_l1631_163136

theorem expression_simplification (x y : ℚ) (hx : x = 3) (hy : y = -1/2) :
  x * (x - 4 * y) + (2 * x + y) * (2 * x - y) - (2 * x - y)^2 = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1631_163136


namespace NUMINAMATH_CALUDE_contradiction_assumption_l1631_163148

theorem contradiction_assumption (a b : ℝ) : ¬(a > b) ↔ (a ≤ b) := by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l1631_163148


namespace NUMINAMATH_CALUDE_tailor_trim_problem_l1631_163102

theorem tailor_trim_problem (original_side : ℝ) (trimmed_opposite : ℝ) (remaining_area : ℝ) :
  original_side = 22 →
  trimmed_opposite = 6 →
  remaining_area = 120 →
  ∃ x : ℝ, (original_side - trimmed_opposite - trimmed_opposite) * (original_side - x) = remaining_area ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_tailor_trim_problem_l1631_163102


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1631_163194

theorem decimal_sum_to_fraction : 
  0.4 + 0.05 + 0.006 + 0.0007 + 0.00008 = 22839 / 50000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1631_163194


namespace NUMINAMATH_CALUDE_min_value_theorem_l1631_163103

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  1/x + 8/(1 - 2*x) ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1631_163103


namespace NUMINAMATH_CALUDE_smallest_n_for_coloring_property_l1631_163179

def is_valid_coloring (n : ℕ) (coloring : ℕ → Bool) : Prop :=
  ∀ x y z w, x ≤ n ∧ y ≤ n ∧ z ≤ n ∧ w ≤ n →
    coloring x = coloring y ∧ coloring y = coloring z ∧ coloring z = coloring w →
    x + y + z ≠ w

theorem smallest_n_for_coloring_property : 
  (∀ n < 11, ∃ coloring, is_valid_coloring n coloring) ∧
  (∀ coloring, ¬ is_valid_coloring 11 coloring) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_coloring_property_l1631_163179


namespace NUMINAMATH_CALUDE_sequence_a_equals_fibonacci_6n_l1631_163198

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def sequence_a : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (18 * sequence_a n + 8 * (Nat.sqrt (5 * (sequence_a n)^2 - 4))) / 2

theorem sequence_a_equals_fibonacci_6n :
  ∀ n : ℕ, sequence_a n = fibonacci (6 * n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_equals_fibonacci_6n_l1631_163198


namespace NUMINAMATH_CALUDE_special_function_property_l1631_163195

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧
  (∀ x : ℝ, f (x + 5) ≥ f x + 5) ∧
  (∀ x : ℝ, f (x + 1) ≤ f x + 1)

/-- The function g defined in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1 - x

/-- The main theorem -/
theorem special_function_property (f : ℝ → ℝ) (hf : special_function f) :
  g f 2009 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l1631_163195


namespace NUMINAMATH_CALUDE_max_gcd_of_sequence_l1631_163156

theorem max_gcd_of_sequence (n : ℕ) : 
  Nat.gcd ((8^n - 1) / 7) ((8^(n+1) - 1) / 7) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_sequence_l1631_163156


namespace NUMINAMATH_CALUDE_a_5_equals_13_l1631_163126

/-- A sequence defined by a_n = pn + q -/
def a (p q : ℝ) : ℕ+ → ℝ := fun n ↦ p * n.val + q

/-- Given a sequence a_n where a_1 = 5, a_8 = 19, and a_n = pn + q for all n ∈ ℕ+
    (where p and q are constants), prove that a_5 = 13 -/
theorem a_5_equals_13 (p q : ℝ) (h1 : a p q 1 = 5) (h8 : a p q 8 = 19) : a p q 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_a_5_equals_13_l1631_163126


namespace NUMINAMATH_CALUDE_inequality_for_increasing_function_l1631_163137

/-- An increasing function on the real line. -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

/-- Theorem: Given an increasing function f on ℝ and real numbers a and b
    such that a + b ≤ 0, the inequality f(a) + f(b) ≤ f(-a) + f(-b) holds. -/
theorem inequality_for_increasing_function
  (f : ℝ → ℝ) (hf : IncreasingFunction f) (a b : ℝ) (hab : a + b ≤ 0) :
  f a + f b ≤ f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_increasing_function_l1631_163137


namespace NUMINAMATH_CALUDE_total_players_is_fifty_l1631_163188

/-- The number of cricket players -/
def cricket_players : ℕ := 12

/-- The number of hockey players -/
def hockey_players : ℕ := 17

/-- The number of football players -/
def football_players : ℕ := 11

/-- The number of softball players -/
def softball_players : ℕ := 10

/-- The total number of players on the ground -/
def total_players : ℕ := cricket_players + hockey_players + football_players + softball_players

/-- Theorem stating that the total number of players is 50 -/
theorem total_players_is_fifty : total_players = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_players_is_fifty_l1631_163188
