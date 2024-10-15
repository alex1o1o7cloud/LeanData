import Mathlib

namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3367_336727

theorem fraction_equation_solution (n : ℚ) : 
  (1 : ℚ) / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4 → n = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3367_336727


namespace NUMINAMATH_CALUDE_weight_of_A_l3367_336778

/-- Prove that given the conditions, the weight of A is 79 kg -/
theorem weight_of_A (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 7 →
  (b + c + d + e) / 4 = 79 →
  a = 79 := by
sorry

end NUMINAMATH_CALUDE_weight_of_A_l3367_336778


namespace NUMINAMATH_CALUDE_part_I_part_II_l3367_336728

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (a b : ℝ) : Set ℝ := {x | x^2 - a*x + b < 0}

-- Theorem for part I
theorem part_I : ∀ a b : ℝ, A = B a b → a = 2 ∧ b = -3 := by sorry

-- Theorem for part II
theorem part_II : ∀ a : ℝ, (A ∩ B a 3) ⊇ B a 3 → -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l3367_336728


namespace NUMINAMATH_CALUDE_mary_savings_problem_l3367_336729

theorem mary_savings_problem (S : ℝ) (x : ℝ) (h1 : S > 0) (h2 : 0 ≤ x ∧ x ≤ 1) 
  (h3 : 12 * S * x = 7 * S * (1 - x)) : 
  1 - x = 12 / 19 := by sorry

end NUMINAMATH_CALUDE_mary_savings_problem_l3367_336729


namespace NUMINAMATH_CALUDE_no_nonzero_real_solution_l3367_336772

theorem no_nonzero_real_solution :
  ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 1/a + 1/b = 2/(a+b) := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_real_solution_l3367_336772


namespace NUMINAMATH_CALUDE_directrix_of_hyperbola_l3367_336763

/-- The directrix of the hyperbola xy = 1 -/
def directrix_equation (x y : ℝ) : Prop :=
  y = -x + Real.sqrt 2 ∨ y = -x - Real.sqrt 2

/-- The hyperbola equation xy = 1 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x * y = 1

/-- Theorem stating that the directrix of the hyperbola xy = 1 has the equation y = -x ± √2 -/
theorem directrix_of_hyperbola (x y : ℝ) :
  hyperbola_equation x y → directrix_equation x y :=
sorry

end NUMINAMATH_CALUDE_directrix_of_hyperbola_l3367_336763


namespace NUMINAMATH_CALUDE_shooting_probabilities_l3367_336720

/-- Represents the probabilities of hitting different rings in a shooting training session -/
structure ShootingProbabilities where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ
  ring7 : ℝ
  sum_to_one : ring10 + ring9 + ring8 + ring7 < 1
  non_negative : ring10 ≥ 0 ∧ ring9 ≥ 0 ∧ ring8 ≥ 0 ∧ ring7 ≥ 0

/-- The probability of hitting either the 10 or 9 ring -/
def prob_10_or_9 (p : ShootingProbabilities) : ℝ := p.ring10 + p.ring9

/-- The probability of scoring less than 7 rings -/
def prob_less_than_7 (p : ShootingProbabilities) : ℝ := 1 - (p.ring10 + p.ring9 + p.ring8 + p.ring7)

theorem shooting_probabilities (p : ShootingProbabilities) 
  (h1 : p.ring10 = 0.21) 
  (h2 : p.ring9 = 0.23) 
  (h3 : p.ring8 = 0.25) 
  (h4 : p.ring7 = 0.28) : 
  prob_10_or_9 p = 0.44 ∧ prob_less_than_7 p = 0.03 := by
  sorry

#eval prob_10_or_9 ⟨0.21, 0.23, 0.25, 0.28, by norm_num, by norm_num⟩
#eval prob_less_than_7 ⟨0.21, 0.23, 0.25, 0.28, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_shooting_probabilities_l3367_336720


namespace NUMINAMATH_CALUDE_unbounded_fraction_value_l3367_336700

theorem unbounded_fraction_value (M : ℝ) :
  ∃ (x y : ℝ), -3 ≤ x ∧ x ≤ 1 ∧ x ≠ 0 ∧ 1 ≤ y ∧ y ≤ 3 ∧ (x + y + 1) / x > M :=
by sorry

end NUMINAMATH_CALUDE_unbounded_fraction_value_l3367_336700


namespace NUMINAMATH_CALUDE_kaleb_second_half_score_l3367_336747

/-- 
Given that Kaleb scored 43 points in the first half of a trivia game and 66 points in total,
this theorem proves that he scored 23 points in the second half.
-/
theorem kaleb_second_half_score 
  (first_half_score : ℕ) 
  (total_score : ℕ) 
  (h1 : first_half_score = 43)
  (h2 : total_score = 66) :
  total_score - first_half_score = 23 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_second_half_score_l3367_336747


namespace NUMINAMATH_CALUDE_cards_eaten_ratio_l3367_336703

theorem cards_eaten_ratio (initial_cards new_cards remaining_cards : ℕ) :
  initial_cards = 84 →
  new_cards = 8 →
  remaining_cards = 46 →
  (initial_cards + new_cards - remaining_cards) * 2 = initial_cards + new_cards :=
by
  sorry

end NUMINAMATH_CALUDE_cards_eaten_ratio_l3367_336703


namespace NUMINAMATH_CALUDE_sqrt_three_minus_sin_squared_fifteen_l3367_336779

theorem sqrt_three_minus_sin_squared_fifteen (π : Real) :
  (Real.sqrt 3) / 2 - Real.sqrt 3 * (Real.sin (π / 12))^2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_sin_squared_fifteen_l3367_336779


namespace NUMINAMATH_CALUDE_remainder_3_pow_2017_mod_17_l3367_336790

theorem remainder_3_pow_2017_mod_17 : 3^2017 % 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2017_mod_17_l3367_336790


namespace NUMINAMATH_CALUDE_factorization_demonstrates_transformation_l3367_336768

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the method used to solve the equation -/
inductive SolvingMethod
  | Factorization

/-- Represents the mathematical idea demonstrated by the solving method -/
inductive MathematicalIdea
  | Transformation
  | Function
  | CombiningNumbersAndShapes
  | Axiomatic

/-- Solves a quadratic equation using the given method -/
def solveQuadratic (eq : QuadraticEquation) (method : SolvingMethod) : Set ℝ :=
  sorry

/-- Determines the mathematical idea demonstrated by the solving method -/
def demonstratedIdea (eq : QuadraticEquation) (method : SolvingMethod) : MathematicalIdea :=
  sorry

theorem factorization_demonstrates_transformation : 
  let eq : QuadraticEquation := { a := 3, b := -6, c := 0 }
  demonstratedIdea eq SolvingMethod.Factorization = MathematicalIdea.Transformation :=
by sorry

end NUMINAMATH_CALUDE_factorization_demonstrates_transformation_l3367_336768


namespace NUMINAMATH_CALUDE_system_solution_l3367_336735

theorem system_solution : ∃ (x y : ℚ), 
  (4 * x + 3 * y = 1) ∧ 
  (6 * x - 9 * y = -8) ∧ 
  (x = -5/18) ∧ 
  (y = 19/27) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3367_336735


namespace NUMINAMATH_CALUDE_monthly_income_problem_l3367_336759

/-- Given the average monthly incomes of three people, prove the monthly income of one person. -/
theorem monthly_income_problem (income_AB income_BC income_AC : ℚ) 
  (h1 : income_AB = 4050)
  (h2 : income_BC = 5250)
  (h3 : income_AC = 4200) :
  ∃ (A B C : ℚ), 
    (A + B) / 2 = income_AB ∧
    (B + C) / 2 = income_BC ∧
    (A + C) / 2 = income_AC ∧
    A = 3000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_problem_l3367_336759


namespace NUMINAMATH_CALUDE_complement_of_union_l3367_336785

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {4, 5}
def B : Set Nat := {3, 4}

theorem complement_of_union :
  (U \ (A ∪ B)) = {1, 2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3367_336785


namespace NUMINAMATH_CALUDE_common_tangents_exist_l3367_336733

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Theorem: Common tangents exist for two given circles -/
theorem common_tangents_exist (c1 c2 : Circle) 
  (h : c1.radius > c2.radius) : 
  ∃ (l : Line), isTangent l c1 ∧ isTangent l c2 := by
  sorry

/-- Function to construct common tangents -/
noncomputable def construct_common_tangents (c1 c2 : Circle) 
  (h : c1.radius > c2.radius) : 
  List Line := sorry

end NUMINAMATH_CALUDE_common_tangents_exist_l3367_336733


namespace NUMINAMATH_CALUDE_at_least_one_false_l3367_336762

theorem at_least_one_false (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_false_l3367_336762


namespace NUMINAMATH_CALUDE_electricity_cost_for_1200_watts_l3367_336750

/-- Calculates the total cost of electricity usage based on tiered pricing, late fees, and discounts --/
def calculate_electricity_cost (usage : ℕ) : ℚ :=
  let tier1_limit : ℕ := 300
  let tier2_limit : ℕ := 800
  let tier1_rate : ℚ := 4
  let tier2_rate : ℚ := 3.5
  let tier3_rate : ℚ := 3
  let late_fee_tier1 : ℚ := 150
  let late_fee_tier2 : ℚ := 200
  let late_fee_tier3 : ℚ := 250
  let discount_lower : ℕ := 900
  let discount_upper : ℕ := 1100

  let tier1_cost := min usage tier1_limit * tier1_rate
  let tier2_cost := max 0 (min usage tier2_limit - tier1_limit) * tier2_rate
  let tier3_cost := max 0 (usage - tier2_limit) * tier3_rate

  let total_electricity_cost := tier1_cost + tier2_cost + tier3_cost

  let late_fee := 
    if usage ≤ 600 then late_fee_tier1
    else if usage ≤ 1000 then late_fee_tier2
    else late_fee_tier3

  let total_cost := total_electricity_cost + late_fee

  -- No discount applied as usage is not in the 3rd highest quartile

  total_cost

theorem electricity_cost_for_1200_watts :
  calculate_electricity_cost 1200 = 4400 := by
  sorry

end NUMINAMATH_CALUDE_electricity_cost_for_1200_watts_l3367_336750


namespace NUMINAMATH_CALUDE_twenty_four_is_eighty_percent_of_thirty_l3367_336717

theorem twenty_four_is_eighty_percent_of_thirty : 
  ∃ x : ℝ, 24 = 0.8 * x ∧ x = 30 := by
sorry

end NUMINAMATH_CALUDE_twenty_four_is_eighty_percent_of_thirty_l3367_336717


namespace NUMINAMATH_CALUDE_proper_subset_condition_l3367_336734

def M : Set ℝ := {x : ℝ | 2 * x^2 - 3 * x - 2 = 0}

def N (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

theorem proper_subset_condition (a : ℝ) :
  N a ⊂ M → a = 0 ∨ a = -2 ∨ a = 1/2 := by sorry

end NUMINAMATH_CALUDE_proper_subset_condition_l3367_336734


namespace NUMINAMATH_CALUDE_rhombus_area_l3367_336797

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : 
  (1 / 2 : ℝ) * d1 * d2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3367_336797


namespace NUMINAMATH_CALUDE_binomial_15_choose_4_l3367_336719

theorem binomial_15_choose_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_choose_4_l3367_336719


namespace NUMINAMATH_CALUDE_evaluate_nested_square_roots_l3367_336725

theorem evaluate_nested_square_roots : 
  Real.sqrt (64 * Real.sqrt (32 * Real.sqrt (4^3))) = 64 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_nested_square_roots_l3367_336725


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_and_48_l3367_336741

theorem smallest_divisible_by_15_and_48 : ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 48 ∣ n → n ≥ 240 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_and_48_l3367_336741


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3367_336724

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*m*x + 1 ≥ 0) ↔ -1 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3367_336724


namespace NUMINAMATH_CALUDE_geometric_series_equation_l3367_336765

theorem geometric_series_equation (x : ℝ) : x = 9 →
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/x)^n := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_equation_l3367_336765


namespace NUMINAMATH_CALUDE_negative_less_than_positive_l3367_336781

theorem negative_less_than_positive : 
  (∀ x y : ℝ, x < 0 ∧ y > 0 → x < y) →
  -897 < 0.01 := by sorry

end NUMINAMATH_CALUDE_negative_less_than_positive_l3367_336781


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l3367_336746

theorem sphere_volume_increase (r : ℝ) (h : r > 0) :
  let V (radius : ℝ) := (4 / 3) * Real.pi * radius ^ 3
  V (2 * r) = 8 * V r :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l3367_336746


namespace NUMINAMATH_CALUDE_reporter_wrong_l3367_336737

/-- Represents a round-robin chess tournament --/
structure ChessTournament where
  num_players : ℕ
  wins : Fin num_players → ℕ
  draws : Fin num_players → ℕ
  losses : Fin num_players → ℕ

/-- The total number of games in a round-robin tournament --/
def total_games (t : ChessTournament) : ℕ :=
  t.num_players * (t.num_players - 1) / 2

/-- The total points scored in the tournament --/
def total_points (t : ChessTournament) : ℕ :=
  2 * total_games t

/-- Theorem stating that it's impossible for each player to have won as many games as they drew --/
theorem reporter_wrong (t : ChessTournament) (h1 : t.num_players = 20) 
    (h2 : ∀ i, t.wins i = t.draws i) : False := by
  sorry


end NUMINAMATH_CALUDE_reporter_wrong_l3367_336737


namespace NUMINAMATH_CALUDE_perfect_square_sum_l3367_336723

theorem perfect_square_sum (a b : ℕ) : 
  ∃ (n : ℕ), 3^a + 4^b = n^2 ↔ a = 2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l3367_336723


namespace NUMINAMATH_CALUDE_smallest_value_theorem_l3367_336770

theorem smallest_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 21 * a * b + 2 * b * c + 8 * c * a ≤ 12) :
  ∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → 
  21 * x * y + 2 * y * z + 8 * z * x ≤ 12 → 
  1 / a + 2 / b + 3 / c ≤ 1 / x + 2 / y + 3 / z :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_theorem_l3367_336770


namespace NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l3367_336795

theorem cos_pi_sixth_minus_alpha (α : ℝ) (h : Real.sin (α + π / 3) = 1 / 2) :
  Real.cos (π / 6 - α) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l3367_336795


namespace NUMINAMATH_CALUDE_expression_equals_sum_l3367_336713

theorem expression_equals_sum (a b c : ℚ) (ha : a = 7) (hb : b = 11) (hc : c = 13) :
  let numerator := a^3 * (1/b - 1/c) + b^3 * (1/c - 1/a) + c^3 * (1/a - 1/b)
  let denominator := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  numerator / denominator = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_sum_l3367_336713


namespace NUMINAMATH_CALUDE_square_of_odd_is_sum_of_consecutive_integers_l3367_336736

theorem square_of_odd_is_sum_of_consecutive_integers :
  ∀ n : ℕ, n > 1 → Odd n → ∃ j : ℕ, n^2 = j + (j + 1) := by sorry

end NUMINAMATH_CALUDE_square_of_odd_is_sum_of_consecutive_integers_l3367_336736


namespace NUMINAMATH_CALUDE_alphabet_composition_l3367_336706

/-- Represents an alphabet with letters containing dots and/or straight lines -/
structure Alphabet where
  total : ℕ
  only_line : ℕ
  only_dot : ℕ
  both : ℕ
  all_accounted : total = only_line + only_dot + both

/-- Theorem: In an alphabet of 40 letters, if 24 contain only a straight line
    and 5 contain only a dot, then 11 must contain both -/
theorem alphabet_composition (a : Alphabet)
  (h1 : a.total = 40)
  (h2 : a.only_line = 24)
  (h3 : a.only_dot = 5) :
  a.both = 11 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_composition_l3367_336706


namespace NUMINAMATH_CALUDE_coin_problem_l3367_336777

theorem coin_problem (total_coins : ℕ) (total_value : ℚ) 
  (h_total_coins : total_coins = 336)
  (h_total_value : total_value = 71)
  : ∃ (coins_20p coins_25p : ℕ),
    coins_20p + coins_25p = total_coins ∧
    (20 : ℚ)/100 * coins_20p + (25 : ℚ)/100 * coins_25p = total_value ∧
    coins_20p = 260 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3367_336777


namespace NUMINAMATH_CALUDE_kyles_profit_is_99_l3367_336714

/-- The amount of money Kyle will make by selling all his remaining baked goods -/
def kyles_profit (initial_cookies initial_brownies : ℕ) 
                 (kyle_eats_cookies kyle_eats_brownies : ℕ) 
                 (mom_eats_cookies mom_eats_brownies : ℕ) 
                 (cookie_price brownie_price : ℚ) : ℚ :=
  let remaining_cookies := initial_cookies - kyle_eats_cookies - mom_eats_cookies
  let remaining_brownies := initial_brownies - kyle_eats_brownies - mom_eats_brownies
  remaining_cookies * cookie_price + remaining_brownies * brownie_price

/-- Theorem stating that Kyle will make $99 by selling all his remaining baked goods -/
theorem kyles_profit_is_99 : 
  kyles_profit 60 32 2 2 1 2 1 (3/2) = 99 := by
  sorry

end NUMINAMATH_CALUDE_kyles_profit_is_99_l3367_336714


namespace NUMINAMATH_CALUDE_student_trip_cost_is_1925_l3367_336792

/-- Calculates the amount each student needs for a trip given fundraising conditions -/
def student_trip_cost (num_students : ℕ) 
                      (misc_expenses : ℕ) 
                      (day1_raised : ℕ) 
                      (day2_raised : ℕ) 
                      (day3_raised : ℕ) 
                      (additional_days : ℕ) 
                      (additional_per_student : ℕ) : ℕ :=
  let first_three_days := day1_raised + day2_raised + day3_raised
  let next_days_total := (first_three_days / 2) * additional_days
  let total_raised := first_three_days + next_days_total
  let total_needed := total_raised + misc_expenses + (num_students * additional_per_student)
  total_needed / num_students

/-- Theorem stating that given the specific conditions, each student needs $1925 for the trip -/
theorem student_trip_cost_is_1925 : 
  student_trip_cost 6 3000 600 900 400 4 475 = 1925 := by
  sorry

#eval student_trip_cost 6 3000 600 900 400 4 475

end NUMINAMATH_CALUDE_student_trip_cost_is_1925_l3367_336792


namespace NUMINAMATH_CALUDE_sum_of_integers_l3367_336721

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : 2 * (a - b + c) = 10)
  (eq2 : 2 * (b - c + d) = 12)
  (eq3 : 2 * (c - d + a) = 6)
  (eq4 : 2 * (d - a + b) = 4) :
  a + b + c + d = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3367_336721


namespace NUMINAMATH_CALUDE_enhanced_square_triangle_count_l3367_336716

/-- A square with diagonals, midpoint connections, and additional bisections -/
structure EnhancedSquare where
  /-- The original square -/
  square : Set (ℝ × ℝ)
  /-- The diagonals of the square -/
  diagonals : Set (Set (ℝ × ℝ))
  /-- The segments connecting midpoints of opposite sides -/
  midpoint_connections : Set (Set (ℝ × ℝ))
  /-- The additional bisections of midpoint connections -/
  bisections : Set (Set (ℝ × ℝ))

/-- A triangle in the enhanced square -/
structure Triangle where
  vertices : Fin 3 → (ℝ × ℝ)

/-- Count the number of triangles in the enhanced square -/
def countTriangles (es : EnhancedSquare) : ℕ :=
  sorry

/-- The main theorem: The number of triangles in the enhanced square is 28 -/
theorem enhanced_square_triangle_count (es : EnhancedSquare) : 
  countTriangles es = 28 := by
  sorry

end NUMINAMATH_CALUDE_enhanced_square_triangle_count_l3367_336716


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3367_336732

-- Define the cubic equation
def cubic_equation (x : ℝ) : Prop := x^3 - 9*x^2 + 10*x + 5 = 0

-- Define the property of distinct roots
def distinct_roots (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Theorem statement
theorem equilateral_triangle_side_length
  (a b c : ℝ)
  (ha : cubic_equation a)
  (hb : cubic_equation b)
  (hc : cubic_equation c)
  (hdistinct : distinct_roots a b c) :
  ∃ (side_length : ℝ), side_length = 2 * Real.sqrt 17 ∧
  side_length^2 = (a - b)^2 + (b - c)^2 + (c - a)^2 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3367_336732


namespace NUMINAMATH_CALUDE_box_face_ratio_l3367_336701

/-- Given a rectangular box with length l, width w, and height h -/
structure Box where
  l : ℝ
  w : ℝ
  h : ℝ

/-- Properties of the box -/
def BoxProperties (box : Box) : Prop :=
  box.l > 0 ∧ box.w > 0 ∧ box.h > 0 ∧
  box.l * box.w * box.h = 5184 ∧
  box.l * box.h = 288 ∧
  box.w * box.h = (1/2) * box.l * box.w

theorem box_face_ratio (box : Box) (hp : BoxProperties box) :
  (box.l * box.w) / (box.l * box.h) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_box_face_ratio_l3367_336701


namespace NUMINAMATH_CALUDE_ten_integer_segments_l3367_336791

/-- Right triangle DEF with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- The number of distinct integer lengths of line segments from E to DF -/
def num_integer_segments (t : RightTriangle) : ℕ :=
  sorry

/-- Our specific right triangle -/
def triangle : RightTriangle :=
  { de := 18, ef := 24 }

theorem ten_integer_segments : num_integer_segments triangle = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_integer_segments_l3367_336791


namespace NUMINAMATH_CALUDE_Q_equals_set_l3367_336712

def P : Set ℕ := {1, 2}

def Q : Set ℕ := {z | ∃ x y, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem Q_equals_set : Q = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_Q_equals_set_l3367_336712


namespace NUMINAMATH_CALUDE_students_answering_both_correctly_l3367_336722

theorem students_answering_both_correctly 
  (total_students : ℕ) 
  (answered_q1 : ℕ) 
  (answered_q2 : ℕ) 
  (not_taken : ℕ) 
  (h1 : total_students = 30) 
  (h2 : answered_q1 = 25) 
  (h3 : answered_q2 = 22) 
  (h4 : not_taken = 5) :
  answered_q1 + answered_q2 - (total_students - not_taken) = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_answering_both_correctly_l3367_336722


namespace NUMINAMATH_CALUDE_lee_cookies_l3367_336705

/-- Given that Lee can make 18 cookies with 2 cups of flour, 
    this function calculates how many cookies he can make with any number of cups of flour. -/
def cookies_from_flour (cups : ℚ) : ℚ :=
  (18 / 2) * cups

/-- Theorem stating that Lee can make 45 cookies with 5 cups of flour. -/
theorem lee_cookies : cookies_from_flour 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_l3367_336705


namespace NUMINAMATH_CALUDE_temperature_difference_l3367_336745

theorem temperature_difference (M L : ℝ) (N : ℝ) : 
  (M = L + N) →
  (abs ((M - 7) - (L + 5)) = 4) →
  (∃ N₁ N₂ : ℝ, (N = N₁ ∨ N = N₂) ∧ N₁ * N₂ = 128) :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_l3367_336745


namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l3367_336753

/-- Prove that the given parametric equations are equivalent to the Cartesian equation -/
theorem parametric_to_cartesian (t : ℝ) (x y : ℝ) (h1 : t ≠ 0) (h2 : x ≠ 1) 
  (h3 : x = 1 - 1/t) (h4 : y = 1 - t^2) : 
  y = x * (x - 2) / (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l3367_336753


namespace NUMINAMATH_CALUDE_probability_one_male_one_female_l3367_336787

/-- The probability of selecting exactly one male and one female student
    when randomly choosing two students from a group of four students
    (one male and three female) --/
theorem probability_one_male_one_female :
  let total_students : ℕ := 4
  let male_students : ℕ := 1
  let female_students : ℕ := 3
  let selected_students : ℕ := 2
  let ways_to_select_one_male_one_female : ℕ := male_students * female_students
  let total_ways_to_select : ℕ := Nat.choose total_students selected_students
  (ways_to_select_one_male_one_female : ℚ) / total_ways_to_select = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_male_one_female_l3367_336787


namespace NUMINAMATH_CALUDE_dan_found_no_money_l3367_336740

/-- The amount of money Dan spent on a snake toy -/
def snake_toy_cost : ℚ := 11.76

/-- The amount of money Dan spent on a cage -/
def cage_cost : ℚ := 14.54

/-- The total cost of Dan's purchases -/
def total_cost : ℚ := 26.3

/-- The amount of money Dan found on the ground -/
def money_found : ℚ := total_cost - (snake_toy_cost + cage_cost)

theorem dan_found_no_money : money_found = 0 := by sorry

end NUMINAMATH_CALUDE_dan_found_no_money_l3367_336740


namespace NUMINAMATH_CALUDE_addition_is_unique_solution_l3367_336766

-- Define the possible operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the operation
def applyOperation (op : Operation) (a b : Int) : Int :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- Theorem statement
theorem addition_is_unique_solution :
  ∃! op : Operation, applyOperation op 7 (-7) = 0 ∧ 
  (op = Operation.Add ∨ op = Operation.Sub ∨ op = Operation.Mul ∨ op = Operation.Div) :=
by sorry

end NUMINAMATH_CALUDE_addition_is_unique_solution_l3367_336766


namespace NUMINAMATH_CALUDE_division_by_reciprocal_five_divided_by_one_fifth_l3367_336739

theorem division_by_reciprocal (a : ℝ) (b : ℝ) (hb : b ≠ 0) :
  a / (1 / b) = a * b := by sorry

theorem five_divided_by_one_fifth :
  5 / (1 / 5) = 25 := by sorry

end NUMINAMATH_CALUDE_division_by_reciprocal_five_divided_by_one_fifth_l3367_336739


namespace NUMINAMATH_CALUDE_inequality_proof_l3367_336731

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (2*a+b)*(2*a-b)*(a-c) + (2*b+c)*(2*b-c)*(b-a) + (2*c+a)*(2*c-a)*(c-b) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3367_336731


namespace NUMINAMATH_CALUDE_jerry_age_l3367_336783

theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 24 → 
  mickey_age = 4 * jerry_age - 8 → 
  jerry_age = 8 := by
sorry

end NUMINAMATH_CALUDE_jerry_age_l3367_336783


namespace NUMINAMATH_CALUDE_vector_magnitude_l3367_336758

/-- Given two vectors a and b in ℝ², prove that |b| = √3 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (let angle := Real.pi / 3
   let a_x := 1
   let a_y := Real.sqrt 2
   a = (a_x, a_y) ∧ 
   Real.cos angle = a.1 * b.1 + a.2 * b.2 / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) ∧
   (a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 0)) →
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3367_336758


namespace NUMINAMATH_CALUDE_minimum_duty_days_l3367_336794

theorem minimum_duty_days (total_members : ℕ) (duty_size_1 duty_size_2 : ℕ) :
  total_members = 33 →
  duty_size_1 = 9 →
  duty_size_2 = 10 →
  ∃ (k n m : ℕ), 
    k + n = 7 ∧ 
    duty_size_1 * k + duty_size_2 * n = total_members * m ∧
    ∀ (k' n' m' : ℕ), 
      k' + n' < 7 → 
      duty_size_1 * k' + duty_size_2 * n' ≠ total_members * m' :=
by sorry

end NUMINAMATH_CALUDE_minimum_duty_days_l3367_336794


namespace NUMINAMATH_CALUDE_sum_of_digits_greater_than_4_l3367_336761

def digits_of_735 : List Nat := [7, 3, 5]

def is_valid_card (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

theorem sum_of_digits_greater_than_4 :
  (∀ d ∈ digits_of_735, is_valid_card d) →
  (List.sum (digits_of_735.filter (λ x => x > 4))) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_greater_than_4_l3367_336761


namespace NUMINAMATH_CALUDE_total_regular_games_count_l3367_336743

def num_teams : ℕ := 15
def top_teams : ℕ := 5
def mid_teams : ℕ := 5
def bottom_teams : ℕ := 5

def top_vs_top_games : ℕ := 12
def top_vs_others_games : ℕ := 8
def mid_vs_mid_games : ℕ := 10
def mid_vs_top_games : ℕ := 6
def bottom_vs_bottom_games : ℕ := 8

def combinations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem total_regular_games_count : 
  (combinations top_teams 2 * top_vs_top_games + 
   top_teams * (num_teams - top_teams) * top_vs_others_games +
   combinations mid_teams 2 * mid_vs_mid_games + 
   mid_teams * top_teams * mid_vs_top_games +
   combinations bottom_teams 2 * bottom_vs_bottom_games) = 850 := by
  sorry

end NUMINAMATH_CALUDE_total_regular_games_count_l3367_336743


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3367_336757

def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 + x - m = 0

theorem contrapositive_equivalence :
  (∀ m : ℝ, m ≤ 0 → has_real_roots m) ↔
  (∀ m : ℝ, ¬(has_real_roots m) → m > 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3367_336757


namespace NUMINAMATH_CALUDE_ursula_initial_money_l3367_336798

/-- Calculates the initial amount of money Ursula had given the purchase details --/
def initial_money (num_hotdogs : ℕ) (price_hotdog : ℚ) (num_salads : ℕ) (price_salad : ℚ) (change : ℚ) : ℚ :=
  num_hotdogs * price_hotdog + num_salads * price_salad + change

/-- Proves that Ursula's initial money was $20.00 given the purchase details --/
theorem ursula_initial_money :
  initial_money 5 (3/2) 3 (5/2) 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ursula_initial_money_l3367_336798


namespace NUMINAMATH_CALUDE_min_p_value_l3367_336788

theorem min_p_value (p q : ℕ+) 
  (h1 : (2008 : ℚ) / 2009 < p / q)
  (h2 : p / q < (2009 : ℚ) / 2010) : 
  (∀ p' q' : ℕ+, (2008 : ℚ) / 2009 < p' / q' → p' / q' < (2009 : ℚ) / 2010 → p ≤ p') → 
  p = 4017 := by
sorry

end NUMINAMATH_CALUDE_min_p_value_l3367_336788


namespace NUMINAMATH_CALUDE_solution_difference_l3367_336742

theorem solution_difference (r s : ℝ) : 
  ((6 * r - 18) / (r^2 + 4 * r - 21) = r + 3) →
  ((6 * s - 18) / (s^2 + 4 * s - 21) = s + 3) →
  r ≠ s →
  r > s →
  r - s = 12 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l3367_336742


namespace NUMINAMATH_CALUDE_mole_fractions_C4H8O2_l3367_336711

/-- Represents a chemical compound with counts of carbon, hydrogen, and oxygen atoms -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the total number of atoms in a compound -/
def totalAtoms (c : Compound) : ℕ := c.carbon + c.hydrogen + c.oxygen

/-- Calculates the mole fraction of an element in a compound -/
def moleFraction (elementCount : ℕ) (c : Compound) : ℚ :=
  elementCount / (totalAtoms c)

/-- The compound C4H8O2 -/
def C4H8O2 : Compound := ⟨4, 8, 2⟩

theorem mole_fractions_C4H8O2 :
  moleFraction C4H8O2.carbon C4H8O2 = 2/7 ∧
  moleFraction C4H8O2.hydrogen C4H8O2 = 4/7 ∧
  moleFraction C4H8O2.oxygen C4H8O2 = 1/7 := by
  sorry


end NUMINAMATH_CALUDE_mole_fractions_C4H8O2_l3367_336711


namespace NUMINAMATH_CALUDE_sine_ratio_equals_two_l3367_336751

/-- Triangle ABC with vertices A(-1, 0), C(1, 0), and B on the ellipse x²/4 + y²/3 = 1 -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A = (-1, 0)
  h_C : C = (1, 0)
  h_B : (B.1^2 / 4) + (B.2^2 / 3) = 1

/-- The sine of an angle in the triangle -/
noncomputable def sin_angle (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

/-- Theorem stating that (sin A + sin C) / sin B = 2 for the given triangle -/
theorem sine_ratio_equals_two (t : Triangle) :
  (sin_angle t 0 + sin_angle t 2) / sin_angle t 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_sine_ratio_equals_two_l3367_336751


namespace NUMINAMATH_CALUDE_only_C_nonlinear_l3367_336726

-- Define the structure for a system of two equations
structure SystemOfEquations where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ

-- Define the systems A, B, C, and D
def systemA : SystemOfEquations := ⟨λ x y => x - 2, λ x y => y - 3⟩
def systemB : SystemOfEquations := ⟨λ x y => x + y - 1, λ x y => x - y - 2⟩
def systemC : SystemOfEquations := ⟨λ x y => x + y - 5, λ x y => x * y - 1⟩
def systemD : SystemOfEquations := ⟨λ x y => y - x, λ x y => x - 2*y - 1⟩

-- Define what it means for an equation to be linear
def isLinear (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, f x y = a * x + b * y + c

-- Define what it means for a system to be linear
def isLinearSystem (s : SystemOfEquations) : Prop :=
  isLinear s.eq1 ∧ isLinear s.eq2

-- Theorem statement
theorem only_C_nonlinear :
  isLinearSystem systemA ∧
  isLinearSystem systemB ∧
  ¬isLinearSystem systemC ∧
  isLinearSystem systemD :=
sorry

end NUMINAMATH_CALUDE_only_C_nonlinear_l3367_336726


namespace NUMINAMATH_CALUDE_range_of_expression_l3367_336744

theorem range_of_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) :
  -(6 : ℝ) ≤ x + 2*y - 2*z ∧ x + 2*y - 2*z ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3367_336744


namespace NUMINAMATH_CALUDE_chord_intersection_probability_2010_l3367_336774

/-- Given a circle with n distinct points evenly placed around it,
    this function returns the probability that when four distinct points
    are randomly chosen, the chord formed by two of these points
    intersects the chord formed by the other two points. -/
def chord_intersection_probability (n : ℕ) : ℚ :=
  1 / 3

/-- Theorem stating that for a circle with 2010 distinct points,
    the probability of chord intersection is 1/3 -/
theorem chord_intersection_probability_2010 :
  chord_intersection_probability 2010 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersection_probability_2010_l3367_336774


namespace NUMINAMATH_CALUDE_distinct_triangles_in_grid_l3367_336708

/-- The number of points in each row or column of the grid -/
def grid_size : ℕ := 3

/-- The total number of points in the grid -/
def total_points : ℕ := grid_size * grid_size

/-- The number of collinear cases (rows + columns + diagonals) -/
def collinear_cases : ℕ := 2 * grid_size + 2

/-- Calculates the number of combinations of k items from n items -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of distinct triangles in a 3x3 grid -/
theorem distinct_triangles_in_grid :
  combinations total_points 3 - collinear_cases = 76 := by
  sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_grid_l3367_336708


namespace NUMINAMATH_CALUDE_family_total_weight_l3367_336780

/-- Represents the weights of a family consisting of a mother, daughter, and grandchild. -/
structure FamilyWeights where
  mother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- The total weight of the family members. -/
def FamilyWeights.total (fw : FamilyWeights) : ℝ :=
  fw.mother + fw.daughter + fw.grandchild

/-- The conditions given in the problem. -/
def satisfies_conditions (fw : FamilyWeights) : Prop :=
  fw.daughter + fw.grandchild = 60 ∧
  fw.grandchild = (1 / 5) * fw.mother ∧
  fw.daughter = 42

/-- Theorem stating that given the conditions, the total weight is 150 kg. -/
theorem family_total_weight (fw : FamilyWeights) 
  (h : satisfies_conditions fw) : fw.total = 150 := by
  sorry

end NUMINAMATH_CALUDE_family_total_weight_l3367_336780


namespace NUMINAMATH_CALUDE_fruit_sales_problem_l3367_336752

-- Define the variables and constants
def june_total_A : ℝ := 12000
def june_total_B : ℝ := 9000
def july_total_quantity : ℝ := 5000
def july_min_total : ℝ := 23400
def cost_A : ℝ := 2.7
def cost_B : ℝ := 3.5

-- Define the theorem
theorem fruit_sales_problem :
  ∃ (june_price_A : ℝ) (july_quantity_A : ℝ) (july_profit : ℝ),
    -- Conditions
    (june_total_A / june_price_A - june_total_B / (1.5 * june_price_A) = 1000) ∧
    (0.7 * june_price_A * july_quantity_A + 0.6 * 1.5 * june_price_A * (july_total_quantity - july_quantity_A) ≥ july_min_total) ∧
    -- Conclusions
    (june_price_A = 6) ∧
    (july_quantity_A = 3000) ∧
    (july_profit = (0.7 * june_price_A - cost_A) * july_quantity_A + 
                   (0.6 * 1.5 * june_price_A - cost_B) * (july_total_quantity - july_quantity_A)) ∧
    (july_profit = 8300) :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_sales_problem_l3367_336752


namespace NUMINAMATH_CALUDE_apples_left_in_basket_l3367_336715

/-- Given a basket of apples, calculate the number of apples left after Ricki and Samson remove some. -/
theorem apples_left_in_basket 
  (initial_apples : ℕ) 
  (ricki_removes : ℕ) 
  (h1 : initial_apples = 184) 
  (h2 : ricki_removes = 34) :
  initial_apples - (ricki_removes + 3 * ricki_removes) = 48 := by
  sorry

#check apples_left_in_basket

end NUMINAMATH_CALUDE_apples_left_in_basket_l3367_336715


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3367_336773

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 3a cos C = 2c cos A and tan A = 1/3, then angle B measures 135°. -/
theorem triangle_angle_measure (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- angles are positive
  A + B + C = π → -- sum of angles in a triangle
  3 * a * Real.cos C = 2 * c * Real.cos A →
  Real.tan A = 1 / 3 →
  B = π / 4 * 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3367_336773


namespace NUMINAMATH_CALUDE_circle_c_equation_l3367_336749

/-- A circle C with center in the first quadrant, satisfying specific conditions -/
structure CircleC where
  a : ℝ
  b : ℝ
  r : ℝ
  center_in_first_quadrant : a > 0 ∧ b > 0
  y_axis_chord : 2 * (r^2 - a^2).sqrt = 2
  x_axis_chord : 2 * (r^2 - b^2).sqrt = 4
  arc_length_ratio : (3 : ℝ) / 4 * 2 * Real.pi * r = 3 * ((1 : ℝ) / 4 * 2 * Real.pi * r)

/-- The equation of circle C is (x-√7)² + (y-2)² = 8 -/
theorem circle_c_equation (c : CircleC) : 
  c.a = Real.sqrt 7 ∧ c.b = 2 ∧ c.r = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_circle_c_equation_l3367_336749


namespace NUMINAMATH_CALUDE_max_k_value_l3367_336738

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * Real.log x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := a + Real.log x + 1

-- State the theorem
theorem max_k_value (a : ℝ) :
  (f' a (Real.exp (-1)) = 1) →
  (∃ k : ℤ, ∀ x > 1, f a x - k * x + k > 0) →
  (∀ k : ℤ, k > 3 → ∃ x > 1, f a x - k * x + k ≤ 0) :=
sorry

end

end NUMINAMATH_CALUDE_max_k_value_l3367_336738


namespace NUMINAMATH_CALUDE_angle_subtraction_l3367_336767

-- Define a structure for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle
def angle1 : Angle := ⟨20, 18⟩

-- Define the operation to subtract an Angle from 90 degrees
def subtractFrom90 (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- State the theorem
theorem angle_subtraction :
  subtractFrom90 angle1 = ⟨69, 42⟩ := by
  sorry


end NUMINAMATH_CALUDE_angle_subtraction_l3367_336767


namespace NUMINAMATH_CALUDE_twenty_paise_coins_count_l3367_336764

theorem twenty_paise_coins_count (total_coins : ℕ) (total_value : ℚ) :
  total_coins = 342 →
  total_value = 71 →
  ∃ (coins_20 coins_25 : ℕ),
    coins_20 + coins_25 = total_coins ∧
    (20 * coins_20 + 25 * coins_25 : ℚ) / 100 = total_value ∧
    coins_20 = 290 := by
  sorry

end NUMINAMATH_CALUDE_twenty_paise_coins_count_l3367_336764


namespace NUMINAMATH_CALUDE_blocks_lost_l3367_336799

/-- Given Carol's initial and final block counts, prove the number of blocks lost. -/
theorem blocks_lost (initial : ℕ) (final : ℕ) (h1 : initial = 42) (h2 : final = 17) :
  initial - final = 25 := by
  sorry

end NUMINAMATH_CALUDE_blocks_lost_l3367_336799


namespace NUMINAMATH_CALUDE_milk_bottles_remaining_l3367_336769

/-- Calculates the number of milk bottles remaining after purchases. -/
def remaining_bottles (initial : ℕ) (jason : ℕ) (harry_more : ℕ) : ℕ :=
  initial - (jason + (jason + harry_more))

/-- Theorem stating the number of remaining bottles in the given scenario. -/
theorem milk_bottles_remaining : remaining_bottles 35 5 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_milk_bottles_remaining_l3367_336769


namespace NUMINAMATH_CALUDE_team_division_probabilities_l3367_336755

def totalTeams : ℕ := 8
def weakTeams : ℕ := 3
def groupSize : ℕ := 4

/-- The probability that one of the groups has exactly 2 weak teams -/
def prob_exactly_two_weak : ℚ := 6/7

/-- The probability that group A has at least 2 weak teams -/
def prob_at_least_two_weak : ℚ := 1/2

theorem team_division_probabilities :
  (totalTeams = 8 ∧ weakTeams = 3 ∧ groupSize = 4) →
  (prob_exactly_two_weak = 6/7 ∧ prob_at_least_two_weak = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_team_division_probabilities_l3367_336755


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l3367_336793

theorem pirate_treasure_probability :
  let n : ℕ := 8
  let p_treasure : ℚ := 1/3
  let p_trap : ℚ := 1/6
  let p_empty : ℚ := 1/2
  let k : ℕ := 4
  p_treasure + p_trap + p_empty = 1 →
  (n.choose k : ℚ) * p_treasure^k * p_empty^(n-k) = 35/648 :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l3367_336793


namespace NUMINAMATH_CALUDE_chebyshev_polynomial_3_and_root_sum_l3367_336748

-- Define Chebyshev polynomials
def is_chebyshev_polynomial (P : ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ x, P (Real.cos x) = Real.cos (n * x)

-- Define P₃
def P₃ (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem chebyshev_polynomial_3_and_root_sum :
  ∃ (a b c d : ℝ),
    (is_chebyshev_polynomial (P₃ a b c d) 3) ∧
    (a = 4 ∧ b = 0 ∧ c = -3 ∧ d = 0) ∧
    (∃ (x₁ x₂ x₃ : ℝ),
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      x₁ ∈ Set.Ioo (-1 : ℝ) 1 ∧
      x₂ ∈ Set.Ioo (-1 : ℝ) 1 ∧
      x₃ ∈ Set.Ioo (-1 : ℝ) 1 ∧
      (4 * x₁^3 - 3 * x₁ = 1/2) ∧
      (4 * x₂^3 - 3 * x₂ = 1/2) ∧
      (4 * x₃^3 - 3 * x₃ = 1/2) ∧
      x₁ + x₂ + x₃ = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_chebyshev_polynomial_3_and_root_sum_l3367_336748


namespace NUMINAMATH_CALUDE_card_play_combinations_l3367_336771

/-- Represents the number of ways to play 5 cards (2 twos and 3 aces) -/
def ways_to_play_cards : ℕ :=
  Nat.factorial 5 + 
  Nat.factorial 2 + 
  Nat.factorial 4 + 
  (Nat.choose 3 2 * Nat.factorial 3) + 
  Nat.factorial 3 + 
  (Nat.choose 3 2 * Nat.factorial 4)

/-- Theorem stating that the number of ways to play the cards is 242 -/
theorem card_play_combinations : ways_to_play_cards = 242 := by
  sorry

end NUMINAMATH_CALUDE_card_play_combinations_l3367_336771


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l3367_336775

theorem rational_solutions_quadratic (m : ℕ+) : 
  (∃ x : ℚ, m * x^2 + 25 * x + m = 0) ↔ (m = 10 ∨ m = 12) :=
by sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l3367_336775


namespace NUMINAMATH_CALUDE_complex_fraction_squared_l3367_336707

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_squared : (2 * i / (1 + i)) ^ 2 = 2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_squared_l3367_336707


namespace NUMINAMATH_CALUDE_intersection_condition_l3367_336760

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≥ p.1^2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - a)^2 ≤ 1}

-- State the theorem
theorem intersection_condition (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3367_336760


namespace NUMINAMATH_CALUDE_binomial_divisibility_l3367_336796

theorem binomial_divisibility (k : ℕ) (hk : k ≥ 2) :
  (∃ m : ℕ, Nat.choose (2^k) 2 + Nat.choose (2^k) 3 = 2^(3*k) * m) ∧
  (∀ n : ℕ, Nat.choose (2^k) 2 + Nat.choose (2^k) 3 ≠ 2^(3*k + 1) * n) :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l3367_336796


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_l3367_336782

/-- Given two nonconstant geometric sequences with different common ratios,
    if 3(a₃ - b₃) = 4(a₂ - b₂), then the sum of their common ratios is 4/3 -/
theorem sum_of_common_ratios (k a₂ a₃ b₂ b₃ p r : ℝ) 
    (h1 : k ≠ 0)
    (h2 : p ≠ 1)
    (h3 : r ≠ 1)
    (h4 : p ≠ r)
    (h5 : a₂ = k * p)
    (h6 : a₃ = k * p^2)
    (h7 : b₂ = k * r)
    (h8 : b₃ = k * r^2)
    (h9 : 3 * (a₃ - b₃) = 4 * (a₂ - b₂)) :
  p + r = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_l3367_336782


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l3367_336710

theorem complex_expression_simplification :
  (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) - Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt (2/3)) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l3367_336710


namespace NUMINAMATH_CALUDE_mike_lego_bridge_l3367_336784

/-- Calculates the number of bricks of other types Mike needs for his LEGO bridge. -/
def other_bricks (type_a : ℕ) (total : ℕ) : ℕ :=
  total - (type_a + type_a / 2)

/-- Theorem stating that Mike will use 90 bricks of other types for his LEGO bridge. -/
theorem mike_lego_bridge :
  ∀ (type_a : ℕ) (total : ℕ),
    type_a ≥ 40 →
    total = 150 →
    other_bricks type_a total = 90 := by
  sorry

end NUMINAMATH_CALUDE_mike_lego_bridge_l3367_336784


namespace NUMINAMATH_CALUDE_triangle_properties_l3367_336754

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.a * Real.sin t.C = Real.sqrt 3 * t.c * Real.cos t.A ∧
  t.b = 2 ∧
  (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 3

-- State the theorem
theorem triangle_properties (t : Triangle) :
  satisfies_conditions t → t.A = π / 3 ∧ t.a = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3367_336754


namespace NUMINAMATH_CALUDE_problem_solution_l3367_336789

theorem problem_solution (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p) 
  (h_equation : p / (q - r) + q / (r - p) + r / (p - q) = 3) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3367_336789


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3367_336776

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem polygon_diagonals (m n : ℕ) : 
  m + n = 33 →
  diagonals m + diagonals n = 243 →
  max m n = 21 →
  diagonals (max m n) = 189 := by
sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3367_336776


namespace NUMINAMATH_CALUDE_average_age_decrease_l3367_336702

theorem average_age_decrease (N : ℕ) : 
  let original_avg : ℚ := 40
  let new_students : ℕ := 12
  let new_students_avg : ℚ := 34
  let total_original_age : ℚ := N * original_avg
  let total_new_age : ℚ := new_students * new_students_avg
  let new_total_students : ℕ := N + new_students
  let new_avg : ℚ := (total_original_age + total_new_age) / new_total_students
  original_avg - new_avg = 6 := by
sorry

end NUMINAMATH_CALUDE_average_age_decrease_l3367_336702


namespace NUMINAMATH_CALUDE_mark_apple_count_l3367_336709

/-- The number of apples Mark has chosen -/
def num_apples (total fruit_count banana_count orange_count : ℕ) : ℕ :=
  total - (banana_count + orange_count)

/-- Theorem stating that Mark has chosen 3 apples -/
theorem mark_apple_count :
  num_apples 12 4 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mark_apple_count_l3367_336709


namespace NUMINAMATH_CALUDE_magnitude_of_z_is_one_l3367_336786

/-- Given a complex number z defined as z = (1-i)/(1+i) + 2i, prove that its magnitude |z| is equal to 1 -/
theorem magnitude_of_z_is_one : 
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I) + 2 * Complex.I
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_is_one_l3367_336786


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l3367_336704

theorem quadratic_equation_result : ∀ y : ℝ, 6 * y^2 + 7 = 2 * y + 12 → (12 * y - 4)^2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l3367_336704


namespace NUMINAMATH_CALUDE_sum_of_integers_l3367_336756

theorem sum_of_integers (a b c d e : ℤ) 
  (eq1 : a - b + c - e = 7)
  (eq2 : b - c + d + e = 9)
  (eq3 : c - d + a - e = 5)
  (eq4 : d - a + b + e = 1) :
  a + b + c + d + e = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3367_336756


namespace NUMINAMATH_CALUDE_area_of_triangle_DBC_l3367_336718

/-- Given points A, B, C, D, and E in a coordinate plane, prove that the area of triangle DBC is 20 -/
theorem area_of_triangle_DBC (A B C D E : ℝ × ℝ) : 
  A = (0, 8) → 
  B = (0, 0) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  E = (B.1 + (C.1 - B.1) / 3, B.2 + (C.2 - B.2) / 3) → 
  (1 / 2) * (C.1 - B.1) * D.2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_DBC_l3367_336718


namespace NUMINAMATH_CALUDE_smallest_positive_angle_2002_l3367_336730

theorem smallest_positive_angle_2002 : 
  ∃ (θ : ℝ), θ > 0 ∧ θ < 360 ∧ ∀ (k : ℤ), -2002 = θ + 360 * k → θ = 158 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_2002_l3367_336730
