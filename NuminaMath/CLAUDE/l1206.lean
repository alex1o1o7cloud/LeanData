import Mathlib

namespace NUMINAMATH_CALUDE_odd_implies_derivative_even_exists_increasing_not_increasing_deriv_l1206_120697

-- Define a real-valued function on R
variable (f : ℝ → ℝ)

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the property of being strictly increasing
def strictly_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Statement 1: If f is odd, then f' is even
theorem odd_implies_derivative_even (h : is_odd f) : is_even (deriv f) :=
sorry

-- Statement 2: There exists a strictly increasing function whose derivative is not strictly increasing
theorem exists_increasing_not_increasing_deriv : 
  ∃ f : ℝ → ℝ, strictly_increasing f ∧ ¬strictly_increasing (deriv f) :=
sorry

end NUMINAMATH_CALUDE_odd_implies_derivative_even_exists_increasing_not_increasing_deriv_l1206_120697


namespace NUMINAMATH_CALUDE_equal_connections_implies_square_l1206_120639

/-- Represents the coloring of vertices in a regular n-gon --/
structure VertexColoring (n : ℕ) where
  red : ℕ
  blue : ℕ
  sum_eq_n : red + blue = n

/-- Condition for equal number of same-colored and different-colored connections --/
def equal_connections (n : ℕ) (c : VertexColoring n) : Prop :=
  (c.red.choose 2) + (c.blue.choose 2) = c.red * c.blue

/-- Theorem stating that if equal_connections holds, then n is a perfect square --/
theorem equal_connections_implies_square (n : ℕ) (c : VertexColoring n) 
  (h : equal_connections n c) : ∃ k : ℕ, n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_equal_connections_implies_square_l1206_120639


namespace NUMINAMATH_CALUDE_shrinking_cities_proportion_l1206_120608

def western_cities : ℕ := 5
def middle_cities : ℕ := 13
def eastern_cities : ℕ := 18

def shrinking_western : ℕ := 5
def shrinking_middle : ℕ := 9
def shrinking_eastern : ℕ := 13

def proportion (shrinking : ℕ) (total : ℕ) : ℚ :=
  shrinking / total

theorem shrinking_cities_proportion :
  proportion shrinking_middle middle_cities < proportion shrinking_western western_cities ∧
  proportion shrinking_middle middle_cities < proportion shrinking_eastern eastern_cities :=
sorry

end NUMINAMATH_CALUDE_shrinking_cities_proportion_l1206_120608


namespace NUMINAMATH_CALUDE_elimination_theorem_l1206_120600

theorem elimination_theorem (x y a b c : ℝ) 
  (ha : a = x + y) 
  (hb : b = x^3 + y^3) 
  (hc : c = x^5 + y^5) : 
  5 * b * (a^3 + b) = a * (a^5 + 9 * c) := by
  sorry

end NUMINAMATH_CALUDE_elimination_theorem_l1206_120600


namespace NUMINAMATH_CALUDE_rectangle_area_l1206_120675

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the properties of the semicircle and rectangle
def is_semicircle (F E : ℝ × ℝ) : Prop := sorry

def is_inscribed_rectangle (A B C D : ℝ × ℝ) (F E : ℝ × ℝ) : Prop := sorry

def is_right_triangle (D F C : ℝ × ℝ) : Prop := sorry

-- Define the distances
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem rectangle_area (A B C D E F : ℝ × ℝ) :
  is_semicircle F E →
  is_inscribed_rectangle A B C D F E →
  is_right_triangle D F C →
  distance D A = 12 →
  distance F D = 7 →
  distance A E = 7 →
  distance D A * distance C D = 24 * Real.sqrt 30 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1206_120675


namespace NUMINAMATH_CALUDE_least_whole_number_subtraction_l1206_120610

theorem least_whole_number_subtraction (x : ℕ) : 
  x ≥ 3 ∧ 
  ∀ y : ℕ, y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21 ∧
  (6 - x : ℚ) / (7 - x) < 16 / 21 :=
sorry

end NUMINAMATH_CALUDE_least_whole_number_subtraction_l1206_120610


namespace NUMINAMATH_CALUDE_contractor_male_wage_l1206_120633

/-- Represents the daily wage structure and worker composition of a building contractor --/
structure ContractorData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the daily wage for male workers given the contractor's data --/
def male_wage (data : ContractorData) : ℕ :=
  let total_workers := data.male_workers + data.female_workers + data.child_workers
  let total_wage := total_workers * data.average_wage
  let female_total := data.female_workers * data.female_wage
  let child_total := data.child_workers * data.child_wage
  (total_wage - female_total - child_total) / data.male_workers

/-- Theorem stating that for the given contractor data, the male wage is 25 --/
theorem contractor_male_wage :
  male_wage {
    male_workers := 20,
    female_workers := 15,
    child_workers := 5,
    female_wage := 20,
    child_wage := 8,
    average_wage := 21
  } = 25 := by
  sorry


end NUMINAMATH_CALUDE_contractor_male_wage_l1206_120633


namespace NUMINAMATH_CALUDE_traditionalist_progressive_ratio_l1206_120687

/-- Represents a country with provinces, progressives, and traditionalists -/
structure Country where
  num_provinces : ℕ
  total_population : ℝ
  fraction_traditionalist : ℝ
  progressives : ℝ
  traditionalists_per_province : ℝ

/-- The theorem stating the ratio of traditionalists in one province to total progressives -/
theorem traditionalist_progressive_ratio (c : Country) 
  (h1 : c.num_provinces = 4)
  (h2 : c.fraction_traditionalist = 0.75)
  (h3 : c.total_population = c.progressives + c.num_provinces * c.traditionalists_per_province)
  (h4 : c.fraction_traditionalist * c.total_population = c.num_provinces * c.traditionalists_per_province) :
  c.traditionalists_per_province / c.progressives = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_traditionalist_progressive_ratio_l1206_120687


namespace NUMINAMATH_CALUDE_journey_solution_l1206_120664

def journey_problem (total_time : ℝ) (speed1 speed2 speed3 speed4 : ℝ) : Prop :=
  let distance := total_time * (speed1 + speed2 + speed3 + speed4) / 4
  total_time = (distance / 4) / speed1 + (distance / 4) / speed2 + (distance / 4) / speed3 + (distance / 4) / speed4 ∧
  distance = 960

theorem journey_solution :
  journey_problem 60 20 10 15 30 := by
  sorry

end NUMINAMATH_CALUDE_journey_solution_l1206_120664


namespace NUMINAMATH_CALUDE_solution_difference_l1206_120672

theorem solution_difference (x₀ y₀ : ℝ) : 
  (x₀^3 - 2023*x₀ = y₀^3 - 2023*y₀ + 2020) →
  (x₀^2 + x₀*y₀ + y₀^2 = 2022) →
  (x₀ - y₀ = -2020) := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1206_120672


namespace NUMINAMATH_CALUDE_allan_final_score_l1206_120650

/-- Calculates the final score on a test with the given parameters. -/
def final_score (total_questions : ℕ) (correct_answers : ℕ) (points_per_correct : ℚ) (points_per_incorrect : ℚ) : ℚ :=
  let incorrect_answers := total_questions - correct_answers
  (correct_answers : ℚ) * points_per_correct - (incorrect_answers : ℚ) * points_per_incorrect

/-- Theorem stating that Allan's final score is 100 given the test conditions. -/
theorem allan_final_score :
  let total_questions : ℕ := 120
  let correct_answers : ℕ := 104
  let points_per_correct : ℚ := 1
  let points_per_incorrect : ℚ := 1/4
  final_score total_questions correct_answers points_per_correct points_per_incorrect = 100 := by
  sorry

end NUMINAMATH_CALUDE_allan_final_score_l1206_120650


namespace NUMINAMATH_CALUDE_quadratic_trinomial_form_is_quadratic_trinomial_l1206_120615

-- Define variables
variable (x y : ℝ)

-- Define A, B, and C
def A : ℝ := x^2 * y + 2
def B : ℝ := 3 * x^2 * y + x
def C : ℝ := 4 * x^2 * y - x * y

-- Theorem statement
theorem quadratic_trinomial_form :
  A x y + B x y - C x y = 2 + x + x * y :=
by sorry

-- Theorem to classify the result as a quadratic trinomial
theorem is_quadratic_trinomial :
  ∃ (a b c : ℝ), A x y + B x y - C x y = a + b * x + c * x * y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_form_is_quadratic_trinomial_l1206_120615


namespace NUMINAMATH_CALUDE_angle_A_value_l1206_120620

noncomputable section

-- Define the triangle ABC
variable (A B C : Real)  -- Angles
variable (a b c : Real)  -- Side lengths

-- Define the conditions
axiom triangle : A + B + C = Real.pi  -- Sum of angles in a triangle
axiom side_a : a = Real.sqrt 3
axiom side_b : b = Real.sqrt 2
axiom angle_B : B = Real.pi / 4  -- 45° in radians

-- State the theorem
theorem angle_A_value : 
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_angle_A_value_l1206_120620


namespace NUMINAMATH_CALUDE_subset_condition_l1206_120667

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | -m < x ∧ x < m}

theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l1206_120667


namespace NUMINAMATH_CALUDE_intersection_singleton_l1206_120624

/-- The set A parameterized by a -/
def A (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = a * p.1 + 1}

/-- The set B -/
def B : Set (ℝ × ℝ) := {p | p.2 = |p.1|}

/-- The theorem stating the condition for A ∩ B to be a singleton -/
theorem intersection_singleton (a : ℝ) :
  (∃! p, p ∈ A a ∩ B) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_singleton_l1206_120624


namespace NUMINAMATH_CALUDE_extended_line_segment_l1206_120691

/-- Given a line segment AB extended to points P and Q, prove the expressions for P and Q -/
theorem extended_line_segment (A B P Q : ℝ × ℝ) : 
  (∃ (k₁ k₂ : ℝ), k₁ > 0 ∧ k₂ > 0 ∧ 
    7 * (P.1 - B.1) = 2 * (P.1 - A.1) ∧
    7 * (P.2 - B.2) = 2 * (P.2 - A.2) ∧
    5 * (Q.1 - B.1) = (Q.1 - A.1) ∧
    5 * (Q.2 - B.2) = (Q.2 - A.2)) →
  (P = (-2/5 : ℝ) • A + (7/5 : ℝ) • B ∧
   Q = (-1/4 : ℝ) • A + (5/4 : ℝ) • B) := by
sorry

end NUMINAMATH_CALUDE_extended_line_segment_l1206_120691


namespace NUMINAMATH_CALUDE_time_to_fill_leaking_tank_l1206_120660

/-- Time to fill a leaking tank -/
theorem time_to_fill_leaking_tank 
  (pump_fill_time : ℝ) 
  (leak_empty_time : ℝ) 
  (h1 : pump_fill_time = 6) 
  (h2 : leak_empty_time = 12) : 
  (pump_fill_time * leak_empty_time) / (leak_empty_time - pump_fill_time) = 12 := by
  sorry

#check time_to_fill_leaking_tank

end NUMINAMATH_CALUDE_time_to_fill_leaking_tank_l1206_120660


namespace NUMINAMATH_CALUDE_janet_siblings_difference_l1206_120617

/-- The number of siblings each person has -/
structure Siblings where
  masud : ℕ
  janet : ℕ
  carlos : ℕ
  stella : ℕ
  lila : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : Siblings) : Prop :=
  s.masud = 45 ∧
  s.janet = 4 * s.masud - 60 ∧
  s.carlos = s.stella + 20 ∧
  s.stella = (5 * s.carlos - 16) / 2 ∧
  s.lila = s.carlos + s.stella + (s.carlos + s.stella) / 3

/-- The theorem to be proved -/
theorem janet_siblings_difference (s : Siblings) 
  (h : problem_conditions s) : 
  s.janet = s.carlos + s.stella + s.lila - 286 := by
  sorry


end NUMINAMATH_CALUDE_janet_siblings_difference_l1206_120617


namespace NUMINAMATH_CALUDE_unique_number_with_remainders_and_quotient_condition_l1206_120625

theorem unique_number_with_remainders_and_quotient_condition :
  ∃! (n : ℕ),
    n > 0 ∧
    n % 7 = 2 ∧
    n % 8 = 4 ∧
    (n - 2) / 7 = (n - 4) / 8 + 7 ∧
    n = 380 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_remainders_and_quotient_condition_l1206_120625


namespace NUMINAMATH_CALUDE_octagon_diagonal_intersection_probability_l1206_120621

/-- The number of vertices in a regular octagon -/
def octagon_vertices : ℕ := 8

/-- The number of diagonals in a regular octagon -/
def octagon_diagonals : ℕ := octagon_vertices * (octagon_vertices - 3) / 2

/-- The number of ways to select two distinct diagonals from a regular octagon -/
def ways_to_select_two_diagonals : ℕ := 
  Nat.choose octagon_diagonals 2

/-- The number of ways to select four vertices from a regular octagon -/
def ways_to_select_four_vertices : ℕ := 
  Nat.choose octagon_vertices 4

/-- The probability that two randomly selected distinct diagonals 
    in a regular octagon intersect at a point strictly within the octagon -/
theorem octagon_diagonal_intersection_probability : 
  (ways_to_select_four_vertices : ℚ) / ways_to_select_two_diagonals = 7 / 19 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonal_intersection_probability_l1206_120621


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1206_120649

theorem sum_of_squares_of_roots : ∃ (a b c : ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -3 ∧ x ≠ -6 →
    (1 / x + 2 / (x + 3) + 3 / (x + 6) = 1) ↔ (x = a ∨ x = b ∨ x = c)) ∧
  a^2 + b^2 + c^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1206_120649


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1206_120658

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*x - 3
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1206_120658


namespace NUMINAMATH_CALUDE_kia_vehicles_count_l1206_120677

/-- The number of Kia vehicles on the lot -/
def num_kia (total vehicles : ℕ) (num_dodge num_hyundai : ℕ) : ℕ :=
  total - num_dodge - num_hyundai

/-- Theorem stating the number of Kia vehicles on the lot -/
theorem kia_vehicles_count :
  let total := 400
  let num_dodge := total / 2
  let num_hyundai := num_dodge / 2
  num_kia total num_dodge num_hyundai = 100 := by
sorry

end NUMINAMATH_CALUDE_kia_vehicles_count_l1206_120677


namespace NUMINAMATH_CALUDE_jogger_faster_speed_l1206_120657

/-- Represents the jogger's speed and distance scenario -/
def JoggerScenario (actual_distance : ℝ) (actual_speed : ℝ) (faster_distance : ℝ) (faster_speed : ℝ) : Prop :=
  (actual_distance / actual_speed) = (faster_distance / faster_speed)

/-- Theorem stating the jogger's faster speed given the conditions -/
theorem jogger_faster_speed :
  ∀ (actual_distance actual_speed faster_distance faster_speed : ℝ),
    actual_distance = 30 →
    actual_speed = 12 →
    faster_distance = actual_distance + 10 →
    JoggerScenario actual_distance actual_speed faster_distance faster_speed →
    faster_speed = 16 := by
  sorry


end NUMINAMATH_CALUDE_jogger_faster_speed_l1206_120657


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l1206_120684

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l1206_120684


namespace NUMINAMATH_CALUDE_range_of_m_for_inequality_l1206_120690

theorem range_of_m_for_inequality (m : ℝ) : 
  (∀ x : ℕ+, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) ↔ (3 * (x : ℝ) - 3 * m ≤ -2 * m)) → 
  (12 ≤ m ∧ m < 15) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_inequality_l1206_120690


namespace NUMINAMATH_CALUDE_min_clerks_problem_l1206_120629

theorem min_clerks_problem : ∃ n : ℕ, n > 0 ∧ (Nat.choose n 4 = 3 * Nat.choose n 3) ∧ ∀ m : ℕ, m > 0 ∧ m < n → Nat.choose m 4 ≠ 3 * Nat.choose m 3 := by
  sorry

end NUMINAMATH_CALUDE_min_clerks_problem_l1206_120629


namespace NUMINAMATH_CALUDE_time_per_toy_l1206_120694

/-- Given a worker who makes 50 toys in 150 hours, prove that the time taken to make one toy is 3 hours. -/
theorem time_per_toy (total_hours : ℝ) (total_toys : ℝ) (h1 : total_hours = 150) (h2 : total_toys = 50) :
  total_hours / total_toys = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_per_toy_l1206_120694


namespace NUMINAMATH_CALUDE_least_sum_p_q_l1206_120680

theorem least_sum_p_q : ∃ (p q : ℕ), 
  p > 1 ∧ q > 1 ∧ 
  17 * (p + 1) = 21 * (q + 1) ∧
  p + q = 38 ∧
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 21 * (q' + 1) → p' + q' ≥ 38 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_p_q_l1206_120680


namespace NUMINAMATH_CALUDE_carly_shipping_cost_l1206_120628

/-- Calculates the shipping cost given a flat fee, per-pound cost, and weight -/
def shipping_cost (flat_fee : ℝ) (per_pound_cost : ℝ) (weight : ℝ) : ℝ :=
  flat_fee + per_pound_cost * weight

/-- Theorem: The shipping cost for Carly's package is $9.00 -/
theorem carly_shipping_cost :
  shipping_cost 5 0.8 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_carly_shipping_cost_l1206_120628


namespace NUMINAMATH_CALUDE_greeting_card_profit_l1206_120681

/-- Represents the greeting card sale problem -/
theorem greeting_card_profit
  (purchase_price : ℚ)
  (total_sale : ℚ)
  (h_purchase : purchase_price = 21 / 100)
  (h_sale : total_sale = 1457 / 100)
  (h_price_limit : ∃ (selling_price : ℚ), 
    selling_price ≤ 2 * purchase_price ∧
    selling_price * (total_sale / selling_price) = total_sale)
  : ∃ (profit : ℚ), profit = 47 / 10 :=
sorry

end NUMINAMATH_CALUDE_greeting_card_profit_l1206_120681


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2145_l1206_120626

theorem smallest_prime_factor_of_2145 : Nat.minFac 2145 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2145_l1206_120626


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1206_120619

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 2) :
  (1 + (1 - x) / (x + 1)) / ((2 * x - 2) / (x^2 + 2 * x + 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1206_120619


namespace NUMINAMATH_CALUDE_emmalyn_earnings_l1206_120612

/-- The rate Emmalyn charges per meter for painting fences, in dollars -/
def rate : ℚ := 0.20

/-- The number of fences in the neighborhood -/
def num_fences : ℕ := 50

/-- The length of each fence in meters -/
def fence_length : ℕ := 500

/-- The total amount Emmalyn earned in dollars -/
def total_amount : ℚ := rate * (num_fences * fence_length)

theorem emmalyn_earnings :
  total_amount = 5000 := by sorry

end NUMINAMATH_CALUDE_emmalyn_earnings_l1206_120612


namespace NUMINAMATH_CALUDE_cost_of_ingredients_l1206_120627

-- Define the given values
def popcorn_sales_per_day : ℕ := 50
def cotton_candy_multiplier : ℕ := 3
def activity_duration : ℕ := 5
def rent : ℕ := 30
def final_earnings : ℕ := 895

-- Define the theorem
theorem cost_of_ingredients :
  let total_daily_sales := popcorn_sales_per_day + cotton_candy_multiplier * popcorn_sales_per_day
  let total_sales := total_daily_sales * activity_duration
  let earnings_after_rent := total_sales - rent
  earnings_after_rent - final_earnings = 75 := by
sorry

end NUMINAMATH_CALUDE_cost_of_ingredients_l1206_120627


namespace NUMINAMATH_CALUDE_a_greater_than_b_l1206_120601

theorem a_greater_than_b (A B : ℝ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : A * 4 = B * 5) : A > B := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l1206_120601


namespace NUMINAMATH_CALUDE_closest_integer_to_ratio_l1206_120648

/-- Given two positive real numbers a and b where a > b, and their arithmetic mean
    is equal to twice their geometric mean, prove that the integer closest to a/b is 14. -/
theorem closest_integer_to_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : (a + b) / 2 = 2 * Real.sqrt (a * b)) : 
    ∃ (n : ℤ), n = 14 ∧ ∀ (m : ℤ), |a / b - ↑n| ≤ |a / b - ↑m| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_ratio_l1206_120648


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1206_120641

theorem min_value_of_expression (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0) 
  (h_zeros : x₁^2 - 4*a*x₁ + a^2 = 0 ∧ x₂^2 - 4*a*x₂ + a^2 = 0) :
  x₁ + x₂ + a / (x₁ * x₂) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1206_120641


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1206_120669

theorem min_value_reciprocal_sum (a b : ℝ) (m n : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  m = (a - 2, 1) →
  n = (1, b + 1) →
  m.1 * n.1 + m.2 * n.2 = 0 →
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) →
  1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1206_120669


namespace NUMINAMATH_CALUDE_expression_simplification_appropriate_integer_value_l1206_120693

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ -2) :
  (x / (x - 2) - 4 / (x^2 - 2*x)) / ((x + 2) / x^2) = x :=
by sorry

theorem appropriate_integer_value :
  ∃ (x : ℤ), -2 ≤ x ∧ x < Real.sqrt 7 ∧ x ≠ 0 ∧ x ≠ 2 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_appropriate_integer_value_l1206_120693


namespace NUMINAMATH_CALUDE_max_value_constraint_l1206_120678

theorem max_value_constraint (p q r : ℝ) (h : 9 * p^2 + 4 * q^2 + 25 * r^2 = 4) :
  5 * p + 3 * q + 10 * r ≤ 10 * Real.sqrt 13 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1206_120678


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1206_120613

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 5 - 5 * Complex.I) * (2 * Real.sqrt 7 + 4 * Complex.I)) = 20 * Real.sqrt 77 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1206_120613


namespace NUMINAMATH_CALUDE_solve_motel_problem_l1206_120630

def motel_problem (higher_rate : ℕ) : Prop :=
  ∃ (num_higher_rate : ℕ) (num_lower_rate : ℕ),
    -- There are two types of room rates: $40 and a higher amount
    higher_rate > 40 ∧
    -- The actual total rent charged was $1000
    num_higher_rate * higher_rate + num_lower_rate * 40 = 1000 ∧
    -- If 10 rooms at the higher rate were rented for $40 instead, the total rent would be reduced by 20%
    (num_higher_rate - 10) * higher_rate + (num_lower_rate + 10) * 40 = 800

theorem solve_motel_problem : 
  ∃ (higher_rate : ℕ), motel_problem higher_rate ∧ higher_rate = 60 :=
sorry

end NUMINAMATH_CALUDE_solve_motel_problem_l1206_120630


namespace NUMINAMATH_CALUDE_zachary_crunches_pushups_difference_l1206_120618

/-- Zachary's push-ups -/
def zachary_pushups : ℕ := 46

/-- Zachary's crunches -/
def zachary_crunches : ℕ := 58

/-- David's push-ups in terms of Zachary's -/
def david_pushups : ℕ := zachary_pushups + 38

/-- David's crunches in terms of Zachary's -/
def david_crunches : ℕ := zachary_crunches - 62

/-- Theorem stating the difference between Zachary's crunches and push-ups -/
theorem zachary_crunches_pushups_difference :
  zachary_crunches - zachary_pushups = 12 := by sorry

end NUMINAMATH_CALUDE_zachary_crunches_pushups_difference_l1206_120618


namespace NUMINAMATH_CALUDE_expression_sign_negative_l1206_120692

theorem expression_sign_negative :
  0 < 1 ∧ 1 < Real.pi / 2 →
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.sin x < Real.sin y) →
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.cos y < Real.cos x) →
  (Real.cos (Real.cos 1) - Real.cos 1) * (Real.sin (Real.sin 1) - Real.sin 1) < 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_sign_negative_l1206_120692


namespace NUMINAMATH_CALUDE_sum_reciprocals_l1206_120696

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / ω^2) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l1206_120696


namespace NUMINAMATH_CALUDE_abc_equality_l1206_120682

theorem abc_equality (a b c : ℝ) (h : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/a) :
  a^2 * b^2 * c^2 = 1 ∨ (a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_abc_equality_l1206_120682


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1206_120642

theorem power_fraction_simplification : (8^15) / (16^7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1206_120642


namespace NUMINAMATH_CALUDE_max_food_per_guest_l1206_120665

theorem max_food_per_guest (total_food : ℝ) (min_guests : ℕ) 
  (h1 : total_food = 325) 
  (h2 : min_guests = 163) : 
  ∃ (max_food : ℝ), max_food ≤ 2 ∧ max_food > total_food / min_guests :=
by
  sorry

end NUMINAMATH_CALUDE_max_food_per_guest_l1206_120665


namespace NUMINAMATH_CALUDE_equation_transformation_l1206_120638

theorem equation_transformation (x y : ℝ) (hx : x ≠ 0) :
  y = x + 1/x →
  (x^4 + x^3 - 5*x^2 + x + 1 = 0) ↔ (x^2*(y^2 + y - 7) = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l1206_120638


namespace NUMINAMATH_CALUDE_tens_digit_of_five_pow_2023_l1206_120622

theorem tens_digit_of_five_pow_2023 : (5^2023 / 10) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_five_pow_2023_l1206_120622


namespace NUMINAMATH_CALUDE_car_speed_l1206_120689

/-- Theorem: Given a car travels 300 miles in 5 hours, its speed is 60 miles per hour. -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 300) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l1206_120689


namespace NUMINAMATH_CALUDE_number_divided_by_seven_l1206_120683

theorem number_divided_by_seven : ∃ x : ℚ, x / 7 = 5 / 14 ∧ x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_seven_l1206_120683


namespace NUMINAMATH_CALUDE_closest_to_standard_weight_l1206_120671

def quality_errors : List ℝ := [-0.02, 0.1, -0.23, -0.3, 0.2]

theorem closest_to_standard_weight :
  ∀ x ∈ quality_errors, |(-0.02)| ≤ |x| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_standard_weight_l1206_120671


namespace NUMINAMATH_CALUDE_weight_sum_proof_l1206_120695

/-- Given the weights of four people and their pairwise sums, 
    prove that the sum of the weights of the first and last person is 295 pounds. -/
theorem weight_sum_proof (a b c d : ℝ) 
  (h1 : a + b = 270)
  (h2 : b + c = 255)
  (h3 : c + d = 280)
  (h4 : a + b + c + d = 480) :
  a + d = 295 := by
  sorry

end NUMINAMATH_CALUDE_weight_sum_proof_l1206_120695


namespace NUMINAMATH_CALUDE_cricket_team_size_l1206_120636

/-- Represents the number of players on a cricket team -/
def total_players : ℕ := 55

/-- Represents the number of throwers on the team -/
def throwers : ℕ := 37

/-- Represents the number of right-handed players on the team -/
def right_handed : ℕ := 49

/-- Theorem stating the total number of players on the cricket team -/
theorem cricket_team_size :
  total_players = throwers + (right_handed - throwers) * 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_size_l1206_120636


namespace NUMINAMATH_CALUDE_triangle_third_vertex_l1206_120654

/-- Given a triangle with vertices at (0, 0), (10, 5), and (x, 0) where x < 0,
    if the area of the triangle is 50 square units, then x = -20. -/
theorem triangle_third_vertex (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * |x * 5| = 50 → x = -20 := by sorry

end NUMINAMATH_CALUDE_triangle_third_vertex_l1206_120654


namespace NUMINAMATH_CALUDE_monogram_cost_per_backpack_l1206_120668

/-- Proves the cost of monogramming each backpack --/
theorem monogram_cost_per_backpack 
  (num_backpacks : ℕ)
  (original_price : ℚ)
  (discount_percent : ℚ)
  (total_cost : ℚ)
  (h1 : num_backpacks = 5)
  (h2 : original_price = 20)
  (h3 : discount_percent = 20 / 100)
  (h4 : total_cost = 140) :
  (total_cost - num_backpacks * (original_price * (1 - discount_percent))) / num_backpacks = 12 := by
  sorry

#check monogram_cost_per_backpack

end NUMINAMATH_CALUDE_monogram_cost_per_backpack_l1206_120668


namespace NUMINAMATH_CALUDE_more_unrepresentable_ten_digit_numbers_l1206_120614

theorem more_unrepresentable_ten_digit_numbers :
  let total_ten_digit_numbers := 9 * (10 ^ 9)
  let five_digit_numbers := 9 * (10 ^ 4)
  let max_representable := five_digit_numbers * (five_digit_numbers + 1)
  max_representable < total_ten_digit_numbers / 2 := by
sorry

end NUMINAMATH_CALUDE_more_unrepresentable_ten_digit_numbers_l1206_120614


namespace NUMINAMATH_CALUDE_meter_to_skips_l1206_120605

theorem meter_to_skips 
  (a b c d e f g : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) (hg : g > 0)
  (hop_skip : a * 1 = b * 1)  -- a hops = b skips
  (jog_hop : c * 1 = d * 1)   -- c jogs = d hops
  (dash_jog : e * 1 = f * 1)  -- e dashes = f jogs
  (meter_dash : 1 = g * 1)    -- 1 meter = g dashes
  : 1 = (g * f * d * b) / (e * c * a) * 1 := by
  sorry

end NUMINAMATH_CALUDE_meter_to_skips_l1206_120605


namespace NUMINAMATH_CALUDE_father_sons_average_age_l1206_120688

/-- The average age of a father and his two sons -/
def average_age (father_age son1_age son2_age : ℕ) : ℚ :=
  (father_age + son1_age + son2_age) / 3

/-- Theorem stating the average age of the father and his two sons -/
theorem father_sons_average_age :
  ∀ (father_age son1_age son2_age : ℕ),
  father_age = 32 →
  son1_age - son2_age = 4 →
  (son1_age - 5 + son2_age - 5) / 2 = 15 →
  average_age father_age son1_age son2_age = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_father_sons_average_age_l1206_120688


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1206_120652

theorem arithmetic_calculation : 2^3 + 4 * 5 - 6 + 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1206_120652


namespace NUMINAMATH_CALUDE_soccer_substitution_ratio_l1206_120663

/-- Soccer team substitution ratio theorem -/
theorem soccer_substitution_ratio 
  (total_players : ℕ) 
  (starters : ℕ) 
  (first_half_subs : ℕ) 
  (non_players : ℕ) 
  (h1 : total_players = 24) 
  (h2 : starters = 11) 
  (h3 : first_half_subs = 2) 
  (h4 : non_players = 7) : 
  (total_players - non_players - (starters + first_half_subs)) / first_half_subs = 2 := by
sorry

end NUMINAMATH_CALUDE_soccer_substitution_ratio_l1206_120663


namespace NUMINAMATH_CALUDE_digit67_is_one_l1206_120604

/-- The sequence of digits formed by concatenating integers from 50 down to 1 -/
def integerSequence : List Nat := sorry

/-- The 67th digit in the sequence -/
def digit67 : Nat := sorry

/-- Theorem stating that the 67th digit in the sequence is 1 -/
theorem digit67_is_one : digit67 = 1 := by sorry

end NUMINAMATH_CALUDE_digit67_is_one_l1206_120604


namespace NUMINAMATH_CALUDE_power_of_product_l1206_120676

theorem power_of_product (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1206_120676


namespace NUMINAMATH_CALUDE_kwik_e_tax_center_problem_l1206_120643

/-- The Kwik-e-Tax Center problem -/
theorem kwik_e_tax_center_problem (federal_price state_price quarterly_price : ℕ) 
  (state_count quarterly_count total_revenue : ℕ) 
  (h1 : federal_price = 50)
  (h2 : state_price = 30)
  (h3 : quarterly_price = 80)
  (h4 : state_count = 20)
  (h5 : quarterly_count = 10)
  (h6 : total_revenue = 4400) :
  ∃ (federal_count : ℕ), 
    federal_count = 60 ∧ 
    federal_price * federal_count + state_price * state_count + quarterly_price * quarterly_count = total_revenue :=
by
  sorry


end NUMINAMATH_CALUDE_kwik_e_tax_center_problem_l1206_120643


namespace NUMINAMATH_CALUDE_fraction_equality_l1206_120609

theorem fraction_equality (b : ℕ+) : 
  (b : ℚ) / (b + 15 : ℚ) = 3/4 → b = 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1206_120609


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l1206_120651

theorem quadratic_form_h_value (a b c : ℝ) :
  (∃ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 7) →
  (∃ n k, ∀ x, 4 * (a * x^2 + b * x + c) = n * (x - 5)^2 + k) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l1206_120651


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l1206_120673

def cube_numbers (start : ℕ) : List ℕ := List.range 6 |>.map (· + start)

def opposite_faces_sum_equal (numbers : List ℕ) : Prop :=
  numbers.length = 6 ∧ 
  ∃ (sum : ℕ), 
    numbers[0]! + numbers[5]! = sum ∧
    numbers[1]! + numbers[4]! = sum ∧
    numbers[2]! + numbers[3]! = sum

theorem cube_sum_theorem :
  let numbers := cube_numbers 15
  opposite_faces_sum_equal numbers →
  numbers.sum = 105 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l1206_120673


namespace NUMINAMATH_CALUDE_amy_treasures_first_level_l1206_120646

def points_per_treasure : ℕ := 4
def treasures_second_level : ℕ := 2
def total_score : ℕ := 32

def treasures_first_level : ℕ := (total_score - points_per_treasure * treasures_second_level) / points_per_treasure

theorem amy_treasures_first_level : treasures_first_level = 6 := by
  sorry

end NUMINAMATH_CALUDE_amy_treasures_first_level_l1206_120646


namespace NUMINAMATH_CALUDE_bisection_method_next_interval_l1206_120611

def f (x : ℝ) := x^3 - 2*x - 5

theorem bisection_method_next_interval :
  let a := 2
  let b := 3
  let x₀ := (a + b) / 2
  f a * f x₀ < 0 → ∃ x ∈ Set.Icc a x₀, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_next_interval_l1206_120611


namespace NUMINAMATH_CALUDE_max_sum_of_squared_distances_l1206_120637

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def sum_of_squared_distances (a b c d : E) : ℝ :=
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2

theorem max_sum_of_squared_distances (a b c d : E) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) : 
  sum_of_squared_distances a b c d ≤ 16 ∧ 
  ∃ (a' b' c' d' : E), sum_of_squared_distances a' b' c' d' = 16 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squared_distances_l1206_120637


namespace NUMINAMATH_CALUDE_hawks_score_l1206_120623

theorem hawks_score (total_points : ℕ) (first_day_margin : ℕ) (second_day_margin : ℕ)
  (h_total : total_points = 130)
  (h_first_margin : first_day_margin = 10)
  (h_second_margin : second_day_margin = 20)
  (h_equal_total : ∃ (eagles_total hawks_total : ℕ),
    eagles_total + hawks_total = total_points ∧ eagles_total = hawks_total) :
  ∃ (hawks_score : ℕ), hawks_score = 65 ∧ hawks_score * 2 = total_points :=
sorry

end NUMINAMATH_CALUDE_hawks_score_l1206_120623


namespace NUMINAMATH_CALUDE_smallest_x_value_l1206_120632

theorem smallest_x_value (x : ℝ) : 
  x ≠ 9 → x ≠ -7 → (x^2 - 5*x - 84) / (x - 9) = 4 / (x + 7) → 
  ∃ (y : ℝ), y = -8 ∧ (y^2 - 5*y - 84) / (y - 9) = 4 / (y + 7) ∧ 
  ∀ (z : ℝ), z ≠ 9 → z ≠ -7 → (z^2 - 5*z - 84) / (z - 9) = 4 / (z + 7) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1206_120632


namespace NUMINAMATH_CALUDE_max_divisors_sympathetic_l1206_120685

/-- A number is sympathetic if for each of its divisors d, d+2 is prime. -/
def Sympathetic (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → Nat.Prime (d + 2)

/-- The maximum number of divisors a sympathetic number can have is 8. -/
theorem max_divisors_sympathetic :
  ∃ n : ℕ, Sympathetic n ∧ (∀ m : ℕ, Sympathetic m → Nat.card (Nat.divisors m) ≤ Nat.card (Nat.divisors n)) ∧
    Nat.card (Nat.divisors n) = 8 :=
sorry

end NUMINAMATH_CALUDE_max_divisors_sympathetic_l1206_120685


namespace NUMINAMATH_CALUDE_min_value_theorem_l1206_120670

theorem min_value_theorem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b + c) + 1 / (a + b + d) + 1 / (a + c + d) + 1 / (b + c + d)) ≥ 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1206_120670


namespace NUMINAMATH_CALUDE_solve_pq_system_l1206_120699

theorem solve_pq_system (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : 
  q = (9 + 3 * Real.sqrt 5) / 2 ∨ q = (9 - 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_pq_system_l1206_120699


namespace NUMINAMATH_CALUDE_blue_marble_probability_is_25_percent_l1206_120698

/-- Probability of picking a blue marble -/
def blue_marble_probability (total_marbles yellow_marbles : ℕ) : ℚ :=
  let green_marbles := yellow_marbles / 2
  let remaining_marbles := total_marbles - yellow_marbles - green_marbles
  let blue_marbles := remaining_marbles / 2
  blue_marbles / total_marbles

/-- Theorem stating that the probability of picking a blue marble is 25% -/
theorem blue_marble_probability_is_25_percent :
  blue_marble_probability 60 20 = 1/4 := by
  sorry

#eval (blue_marble_probability 60 20 * 100 : ℚ)

end NUMINAMATH_CALUDE_blue_marble_probability_is_25_percent_l1206_120698


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l1206_120603

theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 10) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_height := r
  let triangle_area := (1/2) * diameter * max_height
  triangle_area = 100 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l1206_120603


namespace NUMINAMATH_CALUDE_function_inequality_l1206_120647

theorem function_inequality (a b : ℝ) (h_a : a > 0) : 
  (∃ x : ℝ, x > 0 ∧ Real.log x - a * x - b ≥ 0) → a * b ≤ Real.exp (-2) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1206_120647


namespace NUMINAMATH_CALUDE_equation_equality_l1206_120635

theorem equation_equality (x : ℝ) : -x^3 + 7*x^2 + 2*x - 8 = -(x - 2)*(x - 4)*(x - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l1206_120635


namespace NUMINAMATH_CALUDE_flag_distance_not_nine_l1206_120661

theorem flag_distance_not_nine (track_length : ℝ) (num_flags : ℕ) : 
  track_length = 90 → 
  num_flags = 10 → 
  (track_length / (num_flags - 1) ≠ 9) :=
by sorry

end NUMINAMATH_CALUDE_flag_distance_not_nine_l1206_120661


namespace NUMINAMATH_CALUDE_committee_formation_l1206_120659

theorem committee_formation (n m k : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 4) :
  Nat.choose n m = 792 ∧ Nat.choose (n - k) m = 56 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_l1206_120659


namespace NUMINAMATH_CALUDE_function_sum_equals_two_l1206_120607

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x + Real.sqrt (4 * x^2 + 1)) + a

theorem function_sum_equals_two (a : ℝ) (h : f a 0 = 1) :
  f a (Real.log 2 / Real.log 10) + f a (Real.log (1/2) / Real.log 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_sum_equals_two_l1206_120607


namespace NUMINAMATH_CALUDE_max_value_of_S_fourth_power_l1206_120662

theorem max_value_of_S_fourth_power :
  let S (x : ℝ) := |Real.sqrt (x^2 + 4*x + 5) - Real.sqrt (x^2 + 2*x + 5)|
  ∀ x : ℝ, (S x)^4 ≤ 4 ∧ ∃ y : ℝ, (S y)^4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_S_fourth_power_l1206_120662


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1206_120653

-- Define the set A
def A : Set ℝ := {x | x^2 - x ≤ 0}

-- Define the function f
def f (x : ℝ) : ℝ := 2 - x

-- Define the set B as the range of f on A
def B : Set ℝ := f '' A

-- State the theorem
theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1206_120653


namespace NUMINAMATH_CALUDE_square_formation_theorem_l1206_120606

/-- Function to calculate the sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to calculate the minimum number of sticks to break -/
def min_sticks_to_break (n : ℕ) : ℕ :=
  let total_length := sum_to_n n
  if total_length % 4 = 0 then 0
  else 
    let target_length := ((total_length + 3) / 4) * 4
    (target_length - total_length + 1) / 2

theorem square_formation_theorem :
  (min_sticks_to_break 12 = 2) ∧ (min_sticks_to_break 15 = 0) :=
sorry

end NUMINAMATH_CALUDE_square_formation_theorem_l1206_120606


namespace NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l1206_120686

/-- Proves that the first interest rate is 5% given the problem conditions --/
theorem first_interest_rate_is_five_percent
  (total_amount : ℝ)
  (first_part : ℝ)
  (second_part : ℝ)
  (second_interest_rate : ℝ)
  (total_income : ℝ)
  (h1 : total_amount = 2500)
  (h2 : first_part = 1000)
  (h3 : second_part = total_amount - first_part)
  (h4 : second_interest_rate = 6)
  (h5 : total_income = 140)
  (h6 : total_income = (first_part * first_interest_rate / 100) + (second_part * second_interest_rate / 100)) :
  first_interest_rate = 5 := by
  sorry

#check first_interest_rate_is_five_percent

end NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l1206_120686


namespace NUMINAMATH_CALUDE_inequality_transformation_l1206_120656

theorem inequality_transformation (h : (1/4 : ℝ) > (1/8 : ℝ)) : (2 : ℝ) < (3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformation_l1206_120656


namespace NUMINAMATH_CALUDE_max_cars_divided_by_ten_limit_l1206_120645

/-- Represents the safety distance between cars in car lengths per 20 km/h -/
def safety_distance : ℝ := 1

/-- Represents the length of a car in meters -/
def car_length : ℝ := 4

/-- Calculates the maximum number of cars that can pass a counting device in one hour -/
noncomputable def max_cars_per_hour (m : ℝ) : ℝ :=
  (20000 * m) / (car_length * (m + 1))

/-- Theorem stating that the maximum number of cars passing in one hour divided by 10 approaches 500 -/
theorem max_cars_divided_by_ten_limit :
  ∀ ε > 0, ∃ M, ∀ m > M, |max_cars_per_hour m / 10 - 500| < ε :=
sorry

end NUMINAMATH_CALUDE_max_cars_divided_by_ten_limit_l1206_120645


namespace NUMINAMATH_CALUDE_minas_numbers_l1206_120640

theorem minas_numbers (x y : ℤ) (h1 : 3 * x + 4 * y = 135) (h2 : x = 15 ∨ y = 15) : x = 25 ∨ y = 25 :=
sorry

end NUMINAMATH_CALUDE_minas_numbers_l1206_120640


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l1206_120679

theorem consecutive_non_prime_powers (r : ℕ) (hr : r > 0) :
  ∃ x : ℕ, ∀ i ∈ Finset.range r, ¬ ∃ (p : ℕ) (n : ℕ), Prime p ∧ x + i + 1 = p ^ n :=
sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l1206_120679


namespace NUMINAMATH_CALUDE_organization_size_after_five_years_l1206_120616

def organization_size (n : ℕ) : ℕ :=
  match n with
  | 0 => 20
  | k + 1 => 4 * organization_size k - 21

theorem organization_size_after_five_years :
  organization_size 5 = 13343 := by
  sorry

end NUMINAMATH_CALUDE_organization_size_after_five_years_l1206_120616


namespace NUMINAMATH_CALUDE_gcd_459_357_l1206_120655

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1206_120655


namespace NUMINAMATH_CALUDE_no_common_points_range_two_common_points_product_l1206_120666

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := Real.log x
def g (a : ℝ) (x : ℝ) := a * x

-- Part I
theorem no_common_points_range (a : ℝ) :
  (∀ x > 0, f x ≠ g a x) → a > 1 / Real.exp 1 := by sorry

-- Part II
theorem two_common_points_product (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ → f x₁ = g a x₁ → f x₂ = g a x₂ → x₁ * x₂ > Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_no_common_points_range_two_common_points_product_l1206_120666


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1206_120634

theorem polynomial_remainder (x : ℝ) : 
  (x^5 + 2*x^2 + 3) % (x - 2) = 43 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1206_120634


namespace NUMINAMATH_CALUDE_exists_solution_infinitely_many_solutions_divisibility_property_no_extended_solution_l1206_120631

-- Define the condition (*) as a predicate
def condition_star (k a b c : ℕ) : Prop :=
  a^2 + k^2 = b^2 + (k+1)^2 ∧ b^2 + (k+1)^2 = c^2 + (k+2)^2

-- Part (a): Existence of a solution
theorem exists_solution : ∃ k a b c : ℕ, condition_star k a b c :=
sorry

-- Part (b): Infinitely many solutions
theorem infinitely_many_solutions : ∀ n : ℕ, ∃ k a b c : ℕ, k > n ∧ condition_star k a b c :=
sorry

-- Part (c): Divisibility property
theorem divisibility_property : ∀ k a b c : ℕ, condition_star k a b c → 144 ∣ (a * b * c) :=
sorry

-- Part (d): Non-existence of extended solution
def extended_condition (k a b c d : ℕ) : Prop :=
  a^2 + k^2 = b^2 + (k+1)^2 ∧ b^2 + (k+1)^2 = c^2 + (k+2)^2 ∧ c^2 + (k+2)^2 = d^2 + (k+3)^2

theorem no_extended_solution : ¬∃ k a b c d : ℕ, extended_condition k a b c d :=
sorry

end NUMINAMATH_CALUDE_exists_solution_infinitely_many_solutions_divisibility_property_no_extended_solution_l1206_120631


namespace NUMINAMATH_CALUDE_point_on_graph_and_sum_l1206_120644

/-- Given a function g such that g(3) = 10, prove that (1, 7.6) is on the graph of 5y = 4g(3x) - 2
    and the sum of its coordinates is 8.6 -/
theorem point_on_graph_and_sum (g : ℝ → ℝ) (h : g 3 = 10) :
  let f := fun x y => 5 * y = 4 * g (3 * x) - 2
  f 1 7.6 ∧ 1 + 7.6 = 8.6 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_and_sum_l1206_120644


namespace NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l1206_120674

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  N = !![2, 1; 7, -2] ∧
  N.mulVec ![2, 0] = ![4, 14] ∧
  N.mulVec ![-2, 10] = ![6, -34] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l1206_120674


namespace NUMINAMATH_CALUDE_exam_mistakes_l1206_120602

theorem exam_mistakes (bryan_score jen_score sammy_score total_points : ℕ) : 
  bryan_score = 20 →
  jen_score = bryan_score + 10 →
  sammy_score = jen_score - 2 →
  total_points = 35 →
  total_points - sammy_score = 7 :=
by sorry

end NUMINAMATH_CALUDE_exam_mistakes_l1206_120602
