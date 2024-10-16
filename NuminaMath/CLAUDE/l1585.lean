import Mathlib

namespace NUMINAMATH_CALUDE_jackson_earnings_l1585_158594

def hourly_rate : ℝ := 5
def vacuuming_time : ℝ := 2
def vacuuming_repetitions : ℕ := 2
def dish_washing_time : ℝ := 0.5
def bathroom_cleaning_multiplier : ℕ := 3

def total_earnings : ℝ :=
  hourly_rate * (vacuuming_time * vacuuming_repetitions +
                 dish_washing_time +
                 bathroom_cleaning_multiplier * dish_washing_time)

theorem jackson_earnings :
  total_earnings = 30 := by
  sorry

end NUMINAMATH_CALUDE_jackson_earnings_l1585_158594


namespace NUMINAMATH_CALUDE_irrational_and_rational_numbers_l1585_158563

theorem irrational_and_rational_numbers : ∃ (x : ℝ), 
  (Irrational (-Real.sqrt 5)) ∧ 
  (¬ Irrational (Real.sqrt 4)) ∧ 
  (¬ Irrational (2 / 3)) ∧ 
  (¬ Irrational 0) := by
  sorry

end NUMINAMATH_CALUDE_irrational_and_rational_numbers_l1585_158563


namespace NUMINAMATH_CALUDE_fraction_reduction_l1585_158595

theorem fraction_reduction (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (4*x - 4*y) / (4*x * 4*y) = (1/4) * ((x - y) / (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l1585_158595


namespace NUMINAMATH_CALUDE_sum_of_decimals_l1585_158503

theorem sum_of_decimals : 5.623 + 4.76 = 10.383 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l1585_158503


namespace NUMINAMATH_CALUDE_power_function_through_point_l1585_158548

-- Define the power function
def f (x : ℝ) : ℝ := x^(1/3)

-- State the theorem
theorem power_function_through_point (h : f 27 = 3) : f 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1585_158548


namespace NUMINAMATH_CALUDE_cos_54_degrees_l1585_158587

theorem cos_54_degrees : Real.cos (54 * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_54_degrees_l1585_158587


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l1585_158524

/-- Three lines intersecting at a single point -/
structure ThreeIntersectingLines where
  k : ℚ
  intersect_point : ℝ × ℝ
  line1 : ∀ (x y : ℝ), x + k * y = 0 → (x, y) = intersect_point
  line2 : ∀ (x y : ℝ), 2 * x + 3 * y + 8 = 0 → (x, y) = intersect_point
  line3 : ∀ (x y : ℝ), x - y - 1 = 0 → (x, y) = intersect_point

/-- If three lines intersect at a single point, then k = -1/2 -/
theorem intersecting_lines_k_value (lines : ThreeIntersectingLines) : lines.k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l1585_158524


namespace NUMINAMATH_CALUDE_league_games_and_weeks_l1585_158562

/-- Represents a sports league --/
structure League where
  num_teams : ℕ
  games_per_week : ℕ

/-- Calculates the total number of games in a round-robin tournament --/
def total_games (league : League) : ℕ :=
  league.num_teams * (league.num_teams - 1) / 2

/-- Calculates the minimum number of weeks required to complete all games --/
def min_weeks (league : League) : ℕ :=
  (total_games league + league.games_per_week - 1) / league.games_per_week

/-- Theorem about the number of games and weeks in a specific league --/
theorem league_games_and_weeks :
  let league := League.mk 15 7
  total_games league = 105 ∧ min_weeks league = 15 := by
  sorry


end NUMINAMATH_CALUDE_league_games_and_weeks_l1585_158562


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l1585_158527

theorem square_difference_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 72)
  (diff_eq : x - y = 18) :
  x^2 - y^2 = 1296 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l1585_158527


namespace NUMINAMATH_CALUDE_larger_segment_approx_59_l1585_158549

/-- Triangle with sides 40, 50, and 110 units --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 40
  hb : b = 50
  hc : c = 110

/-- Altitude dropped on the longest side --/
def altitude (t : Triangle) : ℝ := sorry

/-- Larger segment cut off on the longest side --/
def larger_segment (t : Triangle) : ℝ := sorry

/-- Theorem stating that the larger segment is approximately 59 units --/
theorem larger_segment_approx_59 (t : Triangle) :
  |larger_segment t - 59| < 0.5 := by sorry

end NUMINAMATH_CALUDE_larger_segment_approx_59_l1585_158549


namespace NUMINAMATH_CALUDE_gcd_108_45_l1585_158547

theorem gcd_108_45 : Nat.gcd 108 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_108_45_l1585_158547


namespace NUMINAMATH_CALUDE_roberto_outfits_l1585_158512

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets : ℕ) : ℕ := trousers * shirts * jackets

/-- Theorem: Roberto can create 84 different outfits -/
theorem roberto_outfits :
  let trousers : ℕ := 4
  let shirts : ℕ := 7
  let jackets : ℕ := 3
  number_of_outfits trousers shirts jackets = 84 := by
  sorry

#eval number_of_outfits 4 7 3

end NUMINAMATH_CALUDE_roberto_outfits_l1585_158512


namespace NUMINAMATH_CALUDE_log_27_3_l1585_158540

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_27_3_l1585_158540


namespace NUMINAMATH_CALUDE_constant_term_product_l1585_158576

variables (p q r : ℝ[X])

theorem constant_term_product (hp : p.coeff 0 = 5) (hr : r.coeff 0 = -15) (h_prod : r = p * q) :
  q.coeff 0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_product_l1585_158576


namespace NUMINAMATH_CALUDE_binomial_expected_value_and_variance_l1585_158593

/-- A random variable following a binomial distribution with n trials and probability p -/
def binomial_distribution (n : ℕ) (p : ℝ) : Type := Unit

variable (ξ : binomial_distribution 200 0.01)

/-- The expected value of a binomial distribution -/
def expected_value (n : ℕ) (p : ℝ) : ℝ := n * p

/-- The variance of a binomial distribution -/
def variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_expected_value_and_variance :
  expected_value 200 0.01 = 2 ∧ variance 200 0.01 = 1.98 := by sorry

end NUMINAMATH_CALUDE_binomial_expected_value_and_variance_l1585_158593


namespace NUMINAMATH_CALUDE_median_trigonometric_values_max_condition_implies_range_l1585_158521

def median (a b c : ℝ) : ℝ := sorry

def max3 (a b c : ℝ) : ℝ := sorry

theorem median_trigonometric_values :
  median (Real.sin (30 * π / 180)) (Real.cos (45 * π / 180)) (Real.tan (60 * π / 180)) = Real.sqrt 2 / 2 := by sorry

theorem max_condition_implies_range (x : ℝ) :
  max3 5 (2*x - 3) (-10 - 3*x) = 5 → -5 ≤ x ∧ x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_median_trigonometric_values_max_condition_implies_range_l1585_158521


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_l1585_158525

theorem lawrence_county_kids_count :
  let kids_stayed_home : ℕ := 644997
  let kids_went_to_camp : ℕ := 893835
  let outside_kids_at_camp : ℕ := 78
  kids_stayed_home + kids_went_to_camp = 1538832 :=
by sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_count_l1585_158525


namespace NUMINAMATH_CALUDE_multiples_of_seven_l1585_158599

/-- The number of multiples of 7 between 200 and 500 -/
def c : ℕ := 
  (Nat.div 500 7 - Nat.div 200 7) + 1

theorem multiples_of_seven : c = 43 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_seven_l1585_158599


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l1585_158533

/-- Given a rhombus with one diagonal of length 120 meters and an area of 4800 square meters,
    prove that the length of the other diagonal is 80 meters. -/
theorem rhombus_diagonal (d₂ : ℝ) (area : ℝ) (h1 : d₂ = 120) (h2 : area = 4800) :
  ∃ d₁ : ℝ, d₁ = 80 ∧ area = (d₁ * d₂) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l1585_158533


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1585_158556

theorem solution_set_of_inequality (x : ℝ) :
  (x + 3) / (x - 1) > 0 ↔ x < -3 ∨ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1585_158556


namespace NUMINAMATH_CALUDE_unique_integer_term_l1585_158570

def is_integer_term (n : ℕ) : Prop :=
  ∃ k : ℤ, (n^2 + 1).factorial / ((n.factorial)^(n + 2)) = k

theorem unique_integer_term :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 100 ∧ is_integer_term n :=
sorry

end NUMINAMATH_CALUDE_unique_integer_term_l1585_158570


namespace NUMINAMATH_CALUDE_james_net_income_l1585_158545

def regular_price : ℝ := 20
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def maintenance_fee : ℝ := 35
def insurance_fee : ℝ := 15

def monday_hours : ℝ := 8
def wednesday_hours : ℝ := 8
def friday_hours : ℝ := 6
def sunday_hours : ℝ := 5

def total_hours : ℝ := monday_hours + wednesday_hours + friday_hours + sunday_hours
def rental_days : ℕ := 4

def discounted_rental : Bool := rental_days ≥ 3

theorem james_net_income :
  let total_rental_income := total_hours * regular_price
  let discounted_income := if discounted_rental then total_rental_income * (1 - discount_rate) else total_rental_income
  let income_with_tax := discounted_income * (1 + sales_tax_rate)
  let total_expenses := maintenance_fee + (insurance_fee * rental_days)
  let net_income := income_with_tax - total_expenses
  net_income = 415.30 := by sorry

end NUMINAMATH_CALUDE_james_net_income_l1585_158545


namespace NUMINAMATH_CALUDE_horner_v3_eq_7_9_l1585_158541

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 2x^4 + 3.5x^3 - 2.6x^2 + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- Coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-0.8, 1.7, -2.6, 3.5, 2, 5]

/-- Theorem: Horner's method for f(x) at x = 1 gives v₃ = 7.9 -/
theorem horner_v3_eq_7_9 : 
  (horner (coeffs.take 4) 1) = 7.9 := by sorry

end NUMINAMATH_CALUDE_horner_v3_eq_7_9_l1585_158541


namespace NUMINAMATH_CALUDE_sales_function_properties_l1585_158564

def f (x : ℝ) : ℝ := x^2 - 7*x + 14

theorem sales_function_properties :
  (∃ (a b : ℝ), a < b ∧ 
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y) ∧
    (∀ x y, b ≤ x ∧ x < y → f x ≤ f y)) ∧
  f 1 = 8 ∧
  f 3 = 2 := by sorry

end NUMINAMATH_CALUDE_sales_function_properties_l1585_158564


namespace NUMINAMATH_CALUDE_max_phoenix_number_l1585_158586

/-- Represents a four-digit number --/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h1 : 2 ≤ a
  h2 : a ≤ b
  h3 : b < c
  h4 : c ≤ d
  h5 : d ≤ 9

/-- Defines a Phoenix number --/
def isPhoenixNumber (n : FourDigitNumber) : Prop :=
  n.b - n.a = 2 * (n.d - n.c)

/-- Defines the G function for a four-digit number --/
def G (n : FourDigitNumber) : Rat :=
  (49 * n.a * n.c - 2 * n.a + 2 * n.d + 23 * n.b - 6) / 24

/-- Theorem stating the maximum Phoenix number --/
theorem max_phoenix_number :
    ∃ (M : FourDigitNumber),
      isPhoenixNumber M ∧
      (G M).isInt ∧
      (∀ (N : FourDigitNumber),
        isPhoenixNumber N →
        (G N).isInt →
        1000 * N.a + 100 * N.b + 10 * N.c + N.d ≤ 1000 * M.a + 100 * M.b + 10 * M.c + M.d) ∧
      1000 * M.a + 100 * M.b + 10 * M.c + M.d = 6699 := by
  sorry

end NUMINAMATH_CALUDE_max_phoenix_number_l1585_158586


namespace NUMINAMATH_CALUDE_rectangle_geometric_mean_l1585_158502

/-- Given a rectangle with side lengths a and b, where b is the geometric mean
    of a and the perimeter, prove that b = a + a√3 -/
theorem rectangle_geometric_mean (a b : ℝ) (h_pos : 0 < a) :
  b^2 = a * (2*a + 2*b) → b = a + a * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_rectangle_geometric_mean_l1585_158502


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1585_158543

theorem rectangle_circle_area_ratio (w : ℝ) (r : ℝ) (h1 : w > 0) (h2 : r > 0) :
  let l := 2 * w
  let rectangle_perimeter := 2 * l + 2 * w
  let circle_circumference := 2 * Real.pi * r
  rectangle_perimeter = circle_circumference →
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
    sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1585_158543


namespace NUMINAMATH_CALUDE_triple_angle_bracket_ten_l1585_158542

def divisor_sum (n : ℕ) : ℕ :=
  sorry

def angle_bracket (n : ℕ) : ℕ :=
  sorry

theorem triple_angle_bracket_ten : angle_bracket (angle_bracket (angle_bracket 10)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_triple_angle_bracket_ten_l1585_158542


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l1585_158523

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  startingNumber : ℕ

/-- Generates the list of selected student numbers -/
def generateSample (s : SystematicSampling) : List ℕ :=
  let interval := s.totalStudents / s.sampleSize
  List.range s.sampleSize |>.map (fun i => s.startingNumber + i * interval)

theorem systematic_sampling_result :
  ∀ (s : SystematicSampling),
    s.totalStudents = 50 →
    s.sampleSize = 5 →
    s.startingNumber = 3 →
    generateSample s = [3, 13, 23, 33, 43] :=
by
  sorry

#eval generateSample ⟨50, 5, 3⟩

end NUMINAMATH_CALUDE_systematic_sampling_result_l1585_158523


namespace NUMINAMATH_CALUDE_equation_solution_l1585_158596

theorem equation_solution : 
  ∃ x : ℝ, 0.3 * x + (0.4 * 0.5) = 0.26 ∧ x = 0.2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1585_158596


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l1585_158534

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l1585_158534


namespace NUMINAMATH_CALUDE_certain_number_proof_l1585_158511

theorem certain_number_proof : ∃ x : ℝ, x * 7 = (35 / 100) * 900 ∧ x = 45 := by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1585_158511


namespace NUMINAMATH_CALUDE_budget_calculation_l1585_158507

/-- The total budget for purchasing a TV, computer, and fridge -/
def total_budget (tv_cost computer_cost fridge_extra_cost : ℕ) : ℕ :=
  tv_cost + computer_cost + (computer_cost + fridge_extra_cost)

/-- Theorem stating that the total budget for the given costs is 1600 -/
theorem budget_calculation :
  total_budget 600 250 500 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_budget_calculation_l1585_158507


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1585_158558

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence condition
  (a 1 + a 2 = 16) →                        -- first given condition
  (a 3 + a 4 = 24) →                        -- second given condition
  (a 7 + a 8 = 54) :=                       -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1585_158558


namespace NUMINAMATH_CALUDE_largest_complete_graph_with_arithmetic_progression_edges_l1585_158597

/-- A function that assigns non-negative integers to edges of a complete graph -/
def EdgeAssignment (n : ℕ) := Fin n → Fin n → ℕ

/-- Predicate to check if three numbers form an arithmetic progression -/
def IsArithmeticProgression (a b c : ℕ) : Prop := 2 * b = a + c

/-- Predicate to check if all edges of a triangle form an arithmetic progression -/
def TriangleIsArithmeticProgression (f : EdgeAssignment n) (i j k : Fin n) : Prop :=
  IsArithmeticProgression (f i j) (f i k) (f j k) ∧
  IsArithmeticProgression (f i j) (f j k) (f i k) ∧
  IsArithmeticProgression (f i k) (f j k) (f i j)

/-- Predicate to check if the edge assignment is valid -/
def ValidAssignment (n : ℕ) (f : EdgeAssignment n) : Prop :=
  (∀ i j : Fin n, i ≠ j → f i j = f j i) ∧ 
  (∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → TriangleIsArithmeticProgression f i j k) ∧
  (∀ i j k l : Fin n, i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
    f i j ≠ f i k ∧ f i j ≠ f i l ∧ f i j ≠ f j k ∧ f i j ≠ f j l ∧ f i j ≠ f k l ∧
    f i k ≠ f i l ∧ f i k ≠ f j k ∧ f i k ≠ f j l ∧ f i k ≠ f k l ∧
    f i l ≠ f j k ∧ f i l ≠ f j l ∧ f i l ≠ f k l ∧
    f j k ≠ f j l ∧ f j k ≠ f k l ∧
    f j l ≠ f k l)

theorem largest_complete_graph_with_arithmetic_progression_edges :
  (∃ f : EdgeAssignment 4, ValidAssignment 4 f) ∧
  (∀ n : ℕ, n > 4 → ¬∃ f : EdgeAssignment n, ValidAssignment n f) :=
sorry

end NUMINAMATH_CALUDE_largest_complete_graph_with_arithmetic_progression_edges_l1585_158597


namespace NUMINAMATH_CALUDE_amys_candy_problem_l1585_158591

/-- Amy's candy problem -/
theorem amys_candy_problem (candy_given : ℕ) (difference : ℕ) : 
  candy_given = 6 → difference = 1 → candy_given - difference = 5 := by
  sorry

end NUMINAMATH_CALUDE_amys_candy_problem_l1585_158591


namespace NUMINAMATH_CALUDE_equation_solution_l1585_158566

theorem equation_solution : ∃! x : ℝ, (x - 12) / 3 = (3 * x + 9) / 8 ∧ x = -123 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1585_158566


namespace NUMINAMATH_CALUDE_ways_to_fifth_floor_l1585_158568

/-- Represents a building with a specified number of floors and staircases between each floor. -/
structure Building where
  floors : ℕ
  staircases : ℕ

/-- Calculates the number of different ways to go from the first floor to the top floor. -/
def waysToTopFloor (b : Building) : ℕ :=
  b.staircases ^ (b.floors - 1)

/-- Theorem stating that in a 5-floor building with 2 staircases between each pair of consecutive floors,
    the number of different ways to go from the first floor to the fifth floor is 2^4. -/
theorem ways_to_fifth_floor :
  let b : Building := { floors := 5, staircases := 2 }
  waysToTopFloor b = 2^4 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_fifth_floor_l1585_158568


namespace NUMINAMATH_CALUDE_equation_solution_l1585_158552

theorem equation_solution : ∃ (x : ℝ), x > 0 ∧ x = Real.sqrt (x - 1/x) + Real.sqrt (1 - 1/x) ∧ x = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1585_158552


namespace NUMINAMATH_CALUDE_sin_less_than_x_l1585_158592

theorem sin_less_than_x :
  (∀ x : ℝ, 0 < x → x < π / 2 → Real.sin x < x) ∧
  (∀ x : ℝ, x > 0 → Real.sin x < x) := by
  sorry

end NUMINAMATH_CALUDE_sin_less_than_x_l1585_158592


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1585_158538

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 15) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 19 :=
by sorry

theorem min_value_achieved (x : ℝ) (h : x > 4) :
  ∃ x₀ > 4, (x₀ + 15) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1585_158538


namespace NUMINAMATH_CALUDE_least_sum_exponents_for_896_l1585_158557

theorem least_sum_exponents_for_896 :
  ∃ (a b c : ℕ), 
    (a < b ∧ b < c) ∧ 
    (2^a + 2^b + 2^c = 896) ∧
    (∀ (x y z : ℕ), x < y ∧ y < z ∧ 2^x + 2^y + 2^z = 896 → a + b + c ≤ x + y + z) ∧
    (a + b + c = 24) := by
  sorry

end NUMINAMATH_CALUDE_least_sum_exponents_for_896_l1585_158557


namespace NUMINAMATH_CALUDE_jackson_vacuum_count_l1585_158551

def chore_pay_rate : ℝ := 5
def vacuum_time : ℝ := 2
def dish_washing_time : ℝ := 0.5
def total_earnings : ℝ := 30

def total_chore_time (vacuum_count : ℝ) : ℝ :=
  vacuum_count * vacuum_time + dish_washing_time + 3 * dish_washing_time

theorem jackson_vacuum_count :
  ∃ (vacuum_count : ℝ), 
    total_chore_time vacuum_count * chore_pay_rate = total_earnings ∧ 
    vacuum_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_jackson_vacuum_count_l1585_158551


namespace NUMINAMATH_CALUDE_min_value_inequality_min_value_achievable_l1585_158571

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 ≥ 5^(5/4) - 10*5^(1/4) + 5 :=
by sorry

theorem min_value_achievable : 
  ∃ (a b c d : ℝ), 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ 5 ∧
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 = 5^(5/4) - 10*5^(1/4) + 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_min_value_achievable_l1585_158571


namespace NUMINAMATH_CALUDE_veranda_area_is_148_l1585_158528

/-- Calculates the area of a veranda surrounding a rectangular room. -/
def verandaArea (roomLength roomWidth verandaWidth : ℝ) : ℝ :=
  let totalLength := roomLength + 2 * verandaWidth
  let totalWidth := roomWidth + 2 * verandaWidth
  let totalArea := totalLength * totalWidth
  let roomArea := roomLength * roomWidth
  totalArea - roomArea

/-- Proves that the area of the veranda is 148 m² given the specified dimensions. -/
theorem veranda_area_is_148 :
  verandaArea 21 12 2 = 148 := by
  sorry

end NUMINAMATH_CALUDE_veranda_area_is_148_l1585_158528


namespace NUMINAMATH_CALUDE_kangaroo_problem_l1585_158560

/-- Represents the number of days required to reach a target number of kangaroos -/
def daysToReach (initial : ℕ) (daily : ℕ) (target : ℕ) : ℕ :=
  if initial ≥ target then 0
  else ((target - initial) + (daily - 1)) / daily

theorem kangaroo_problem :
  let kameronKangaroos : ℕ := 100
  let bertInitial : ℕ := 20
  let bertDaily : ℕ := 2
  let christinaInitial : ℕ := 45
  let christinaDaily : ℕ := 3
  let davidInitial : ℕ := 10
  let davidDaily : ℕ := 5
  
  max (daysToReach bertInitial bertDaily kameronKangaroos)
      (max (daysToReach christinaInitial christinaDaily kameronKangaroos)
           (daysToReach davidInitial davidDaily kameronKangaroos)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_problem_l1585_158560


namespace NUMINAMATH_CALUDE_max_value_tangent_l1585_158539

theorem max_value_tangent (x₀ : ℝ) : 
  (∀ x : ℝ, 3 * Real.sin x - 4 * Real.cos x ≤ 3 * Real.sin x₀ - 4 * Real.cos x₀) → 
  Real.tan x₀ = -3/4 := by
sorry

end NUMINAMATH_CALUDE_max_value_tangent_l1585_158539


namespace NUMINAMATH_CALUDE_soccer_league_female_fraction_l1585_158565

theorem soccer_league_female_fraction :
  -- Last year's male participants
  ∀ (male_last_year : ℕ),
  male_last_year = 30 →
  -- Male increase rate
  ∀ (male_increase_rate : ℝ),
  male_increase_rate = 0.1 →
  -- Female increase rate
  ∀ (female_increase_rate : ℝ),
  female_increase_rate = 0.25 →
  -- Overall increase rate
  ∀ (total_increase_rate : ℝ),
  total_increase_rate = 0.1 →
  -- This year's female participants fraction
  ∃ (female_fraction : ℚ),
  female_fraction = 50 / 83 :=
by sorry

end NUMINAMATH_CALUDE_soccer_league_female_fraction_l1585_158565


namespace NUMINAMATH_CALUDE_angle_equivalence_l1585_158510

def angle_with_same_terminal_side (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 - 120

theorem angle_equivalence :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ angle_with_same_terminal_side θ ∧ θ = 240 := by
sorry

end NUMINAMATH_CALUDE_angle_equivalence_l1585_158510


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l1585_158505

theorem parabola_intersection_difference (a b c d : ℝ) : 
  (∀ x, 3 * x^2 - 6 * x + 6 = -2 * x^2 - 4 * x + 6 → x = a ∨ x = c) →
  (3 * a^2 - 6 * a + 6 = -2 * a^2 - 4 * a + 6) →
  (3 * c^2 - 6 * c + 6 = -2 * c^2 - 4 * c + 6) →
  c ≥ a →
  c - a = 2/5 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l1585_158505


namespace NUMINAMATH_CALUDE_angle_B_is_150_degrees_l1585_158578

/-- Given a triangle ABC where sin²B - sin²C - sin²A = √3 * sinA * sinC, prove that angle B is 150°. -/
theorem angle_B_is_150_degrees (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_condition : Real.sin B ^ 2 - Real.sin C ^ 2 - Real.sin A ^ 2 = Real.sqrt 3 * Real.sin A * Real.sin C) : 
  B = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_150_degrees_l1585_158578


namespace NUMINAMATH_CALUDE_n_value_l1585_158561

theorem n_value (n : ℝ) (h1 : n > 0) (h2 : Real.sqrt (4 * n^2) = 64) : n = 32 := by
  sorry

end NUMINAMATH_CALUDE_n_value_l1585_158561


namespace NUMINAMATH_CALUDE_rectangle_dimensions_theorem_l1585_158504

def rectangle_dimensions (w : ℝ) : Prop :=
  let l := w + 3
  let perimeter := 2 * (w + l)
  let area := w * l
  perimeter = 2 * area ∧ w > 0 ∧ l > 0 → w = 1 ∧ l = 4

theorem rectangle_dimensions_theorem :
  ∃ w : ℝ, rectangle_dimensions w := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_theorem_l1585_158504


namespace NUMINAMATH_CALUDE_inequality_proof_l1585_158575

theorem inequality_proof (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1)
  (h : a / (1 - x) + b / (1 - y) = 1) : 
  (a * y) ^ (1/3 : ℝ) + (b * x) ^ (1/3 : ℝ) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1585_158575


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l1585_158522

theorem perfect_square_polynomial (x : ℤ) : 
  ∃ (y : ℤ), x^4 + x^3 + x^2 + x + 1 = y^2 ↔ x = -1 ∨ x = 0 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l1585_158522


namespace NUMINAMATH_CALUDE_sum_of_ac_l1585_158532

theorem sum_of_ac (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 40) 
  (h2 : b + d = 8) : 
  a + c = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ac_l1585_158532


namespace NUMINAMATH_CALUDE_conference_games_count_l1585_158585

/-- The number of teams in the conference -/
def total_teams : ℕ := 16

/-- The number of divisions in the conference -/
def num_divisions : ℕ := 2

/-- The number of teams in each division -/
def teams_per_division : ℕ := 8

/-- The number of times each team plays others in its division -/
def intra_division_games : ℕ := 3

/-- The number of times each team plays teams in the other division -/
def inter_division_games : ℕ := 2

/-- Calculates the total number of games in a complete season for the conference -/
def total_games : ℕ :=
  let intra_division_total := num_divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_division_total := (total_teams * teams_per_division / 2) * inter_division_games
  intra_division_total + inter_division_total

theorem conference_games_count : total_games = 296 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_count_l1585_158585


namespace NUMINAMATH_CALUDE_rachel_chairs_l1585_158569

/-- The number of chairs Rachel bought -/
def num_chairs : ℕ := 7

/-- The number of tables Rachel bought -/
def num_tables : ℕ := 3

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_furniture : ℕ := 4

/-- The total time spent (in minutes) -/
def total_time : ℕ := 40

theorem rachel_chairs :
  num_chairs = (total_time - num_tables * time_per_furniture) / time_per_furniture :=
by sorry

end NUMINAMATH_CALUDE_rachel_chairs_l1585_158569


namespace NUMINAMATH_CALUDE_triangle_side_count_l1585_158589

theorem triangle_side_count (a b : ℕ) (ha : a = 8) (hb : b = 5) :
  ∃! n : ℕ, n = (Finset.range (a + b - 1) \ Finset.range (a - b + 1)).card :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_count_l1585_158589


namespace NUMINAMATH_CALUDE_total_study_hours_l1585_158531

/-- The number of weeks in the fall semester -/
def semester_weeks : ℕ := 15

/-- The number of study hours on weekdays -/
def weekday_hours : ℕ := 3

/-- The number of study hours on Saturday -/
def saturday_hours : ℕ := 4

/-- The number of study hours on Sunday -/
def sunday_hours : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- Theorem stating the total study hours during the semester -/
theorem total_study_hours :
  semester_weeks * (weekdays_per_week * weekday_hours + saturday_hours + sunday_hours) = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_study_hours_l1585_158531


namespace NUMINAMATH_CALUDE_kim_nail_polishes_l1585_158550

/-- Given information about nail polishes owned by Kim, Heidi, and Karen, prove that Kim has 12 nail polishes. -/
theorem kim_nail_polishes :
  ∀ (K : ℕ), -- Kim's nail polishes
  (K + 5) + (K - 4) = 25 → -- Heidi and Karen's total
  K = 12 := by
sorry

end NUMINAMATH_CALUDE_kim_nail_polishes_l1585_158550


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1585_158588

theorem max_sum_of_factors (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 144 →
  a + b + c ≤ 75 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1585_158588


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1585_158508

/-- The area of a square with perimeter 32 cm is 64 cm² -/
theorem square_area_from_perimeter (perimeter : ℝ) (side : ℝ) (area : ℝ) :
  perimeter = 32 →
  side = perimeter / 4 →
  area = side * side →
  area = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l1585_158508


namespace NUMINAMATH_CALUDE_ratio_AD_BC_l1585_158544

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def is_right_triangle (B C D : ℝ × ℝ) : Prop :=
  (C.1 - B.1) * (D.1 - B.1) + (C.2 - B.2) * (D.2 - B.2) = 0

def BC_twice_BD (B C D : ℝ × ℝ) : Prop :=
  dist B C = 2 * dist B D

-- Theorem statement
theorem ratio_AD_BC (A B C D : ℝ × ℝ) 
  (h1 : is_equilateral A B C)
  (h2 : is_right_triangle B C D)
  (h3 : BC_twice_BD B C D) :
  dist A D / dist B C = 3/2 :=
sorry

end NUMINAMATH_CALUDE_ratio_AD_BC_l1585_158544


namespace NUMINAMATH_CALUDE_mall_meal_pairs_l1585_158567

/-- The number of distinct pairs of meals for two people, given the number of options for each meal component. -/
def distinct_meal_pairs (num_entrees num_drinks num_desserts : ℕ) : ℕ :=
  let total_meals := num_entrees * num_drinks * num_desserts
  total_meals * (total_meals - 1)

/-- Theorem stating that the number of distinct meal pairs is 1260 given the specific options. -/
theorem mall_meal_pairs :
  distinct_meal_pairs 4 3 3 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_mall_meal_pairs_l1585_158567


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1585_158572

theorem quadratic_equation_solution :
  ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 11 * x - 20 = 0 :=
by
  use 4/3
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1585_158572


namespace NUMINAMATH_CALUDE_solution_values_l1585_158582

theorem solution_values (x : ℝ) (hx : x^2 + 4 * (x / (x - 2))^2 = 45) :
  let y := ((x - 2)^2 * (x + 3)) / (2*x - 3)
  y = 2 ∨ y = 16 :=
sorry

end NUMINAMATH_CALUDE_solution_values_l1585_158582


namespace NUMINAMATH_CALUDE_total_marbles_count_l1585_158555

def initial_marbles : ℝ := 87.0
def received_marbles : ℝ := 8.0

theorem total_marbles_count : 
  initial_marbles + received_marbles = 95.0 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_count_l1585_158555


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l1585_158546

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l1585_158546


namespace NUMINAMATH_CALUDE_problem_solution_l1585_158577

theorem problem_solution (a b c : ℚ) : 
  a + b + c = 150 ∧ 
  a + 10 = b - 5 ∧ 
  a + 10 = 7 * c → 
  b = 232 / 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1585_158577


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1585_158554

theorem diophantine_equation_solutions :
  ∀ (a b : ℕ), 2017^a = b^6 - 32*b + 1 ↔ (a = 0 ∧ b = 0) ∨ (a = 0 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1585_158554


namespace NUMINAMATH_CALUDE_scenario_is_simple_random_sampling_l1585_158530

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | ComplexRandom

/-- Represents a population of students -/
structure Population where
  size : ℕ
  is_first_year : Bool

/-- Represents a sample from a population -/
structure Sample where
  size : ℕ
  population : Population
  selection_method : SamplingMethod

/-- The sampling method used in the given scenario -/
def scenario_sampling : Sample where
  size := 20
  population := { size := 200, is_first_year := true }
  selection_method := SamplingMethod.SimpleRandom

/-- Theorem stating that the sampling method used in the scenario is simple random sampling -/
theorem scenario_is_simple_random_sampling :
  scenario_sampling.selection_method = SamplingMethod.SimpleRandom :=
by
  sorry


end NUMINAMATH_CALUDE_scenario_is_simple_random_sampling_l1585_158530


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1585_158573

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1585_158573


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_l1585_158517

theorem integral_sqrt_one_minus_x_squared (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f x = Real.sqrt (1 - x^2)) →
  (∫ x in Set.Icc (-1) 1, f x) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_l1585_158517


namespace NUMINAMATH_CALUDE_work_completion_time_work_completion_result_l1585_158529

/-- The time taken to complete a work when two people work together, given their individual completion times -/
theorem work_completion_time (ajay_time vijay_time : ℝ) (h1 : ajay_time > 0) (h2 : vijay_time > 0) :
  (ajay_time * vijay_time) / (ajay_time + vijay_time) = 
    (8 : ℝ) * 24 / ((8 : ℝ) + 24) :=
by sorry

/-- The result of the work completion time calculation is 6 days -/
theorem work_completion_result :
  (8 : ℝ) * 24 / ((8 : ℝ) + 24) = 6 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_work_completion_result_l1585_158529


namespace NUMINAMATH_CALUDE_integral_sin4_cos2_l1585_158536

theorem integral_sin4_cos2 (x : Real) :
  let f := fun (x : Real) => (1/16) * x - (1/64) * Real.sin (4*x) - (1/48) * Real.sin (2*x)^3
  (deriv f) x = Real.sin x^4 * Real.cos x^2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin4_cos2_l1585_158536


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l1585_158509

theorem quadratic_equation_proof :
  ∃ (x y : ℝ), x + y = 10 ∧ |x - y| = 6 ∧ x^2 - 10*x + 16 = 0 ∧ y^2 - 10*y + 16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l1585_158509


namespace NUMINAMATH_CALUDE_odd_function_sum_zero_l1585_158598

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_sum_zero (f : ℝ → ℝ) (h : OddFunction f) :
  f (-2012) + f (-2011) + f 0 + f 2011 + f 2012 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_zero_l1585_158598


namespace NUMINAMATH_CALUDE_bell_interval_problem_l1585_158584

theorem bell_interval_problem (x : ℕ+) : 
  Nat.lcm x (Nat.lcm 10 (Nat.lcm 14 18)) = 630 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_bell_interval_problem_l1585_158584


namespace NUMINAMATH_CALUDE_binary_110010_is_50_l1585_158537

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (fun acc (i, x) => acc + if x then 2^i else 0) 0

def binary_110010 : List Bool := [false, true, false, false, true, true]

theorem binary_110010_is_50 : binary_to_decimal binary_110010 = 50 := by
  sorry

end NUMINAMATH_CALUDE_binary_110010_is_50_l1585_158537


namespace NUMINAMATH_CALUDE_range_of_a_l1585_158519

/-- The solution set to the inequality a^2 - 4a + 3 < 0 -/
def P (a : ℝ) : Prop := a^2 - 4*a + 3 < 0

/-- The real number a for which (a-2)x^2 + 2(a-2)x - 4 < 0 holds for all real numbers x -/
def Q (a : ℝ) : Prop := ∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 < 0

/-- Given P ∨ Q is true, the range of values for the real number a is -2 < a < 3 -/
theorem range_of_a (a : ℝ) (h : P a ∨ Q a) : -2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1585_158519


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_l1585_158518

-- Define the geometric sequences and their properties
def geometric_sequence (k a₂ a₃ : ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 1 ∧ a₂ = k * p ∧ a₃ = k * p^2

-- Theorem statement
theorem sum_of_common_ratios 
  (k a₂ a₃ b₂ b₃ : ℝ) 
  (h₁ : geometric_sequence k a₂ a₃)
  (h₂ : geometric_sequence k b₂ b₃)
  (h₃ : ∃ p r : ℝ, p ≠ r ∧ 
    a₂ = k * p ∧ a₃ = k * p^2 ∧ 
    b₂ = k * r ∧ b₃ = k * r^2)
  (h₄ : a₃ - b₃ = 3 * (a₂ - b₂))
  : ∃ p r : ℝ, p + r = 3 ∧ 
    a₂ = k * p ∧ a₃ = k * p^2 ∧ 
    b₂ = k * r ∧ b₃ = k * r^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_l1585_158518


namespace NUMINAMATH_CALUDE_inequality_propositions_l1585_158500

theorem inequality_propositions :
  ∃ (correct : Finset (Fin 4)), correct.card = 2 ∧
  (∀ i, i ∈ correct ↔
    (i = 0 ∧ (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b)) ∨
    (i = 1 ∧ (∀ a b c d : ℝ, a > b → c > d → a + c > b + d)) ∨
    (i = 2 ∧ (∀ a b c d : ℝ, a > b → c > d → a * c > b * d)) ∨
    (i = 3 ∧ (∀ a b : ℝ, a > b → 1 / a > 1 / b))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_propositions_l1585_158500


namespace NUMINAMATH_CALUDE_area_between_circles_l1585_158513

/-- The area between two concentric circles, where the larger circle's radius is three times 
    the smaller circle's radius, and the smaller circle's diameter is 6 units, 
    is equal to 72π square units. -/
theorem area_between_circles (π : ℝ) : 
  let small_diameter : ℝ := 6
  let small_radius : ℝ := small_diameter / 2
  let large_radius : ℝ := 3 * small_radius
  let area_large : ℝ := π * large_radius ^ 2
  let area_small : ℝ := π * small_radius ^ 2
  area_large - area_small = 72 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_circles_l1585_158513


namespace NUMINAMATH_CALUDE_remainder_theorem_l1585_158580

-- Define the polynomial q(x)
def q (A B C x : ℝ) : ℝ := A * x^5 - B * x^3 + C * x - 2

-- Theorem statement
theorem remainder_theorem (A B C : ℝ) :
  (q A B C 2 = -6) → (q A B C (-2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1585_158580


namespace NUMINAMATH_CALUDE_sphere_wedge_volume_l1585_158583

/-- The volume of a wedge from a sphere -/
theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) : 
  circumference = 18 * Real.pi → num_wedges = 6 →
  (4 / 3 * Real.pi * (circumference / (2 * Real.pi))^3) / num_wedges = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_wedge_volume_l1585_158583


namespace NUMINAMATH_CALUDE_polynomial_leading_coefficient_l1585_158520

/-- A polynomial g satisfying g(x + 1) - g(x) = 12x + 2 for all x has leading coefficient 6 -/
theorem polynomial_leading_coefficient (g : ℝ → ℝ) :
  (∀ x, g (x + 1) - g x = 12 * x + 2) →
  ∃ c, ∀ x, g x = 6 * x^2 - 4 * x + c :=
sorry

end NUMINAMATH_CALUDE_polynomial_leading_coefficient_l1585_158520


namespace NUMINAMATH_CALUDE_min_segment_length_l1585_158501

/-- A cube with edge length 1 -/
structure Cube :=
  (edge_length : ℝ)
  (edge_length_eq : edge_length = 1)

/-- A point on the diagonal A₁D of the cube -/
structure PointM (cube : Cube) :=
  (coords : ℝ × ℝ × ℝ)

/-- A point on the edge CD₁ of the cube -/
structure PointN (cube : Cube) :=
  (coords : ℝ × ℝ × ℝ)

/-- The condition that MN is parallel to A₁ACC₁ -/
def is_parallel_to_diagonal_face (cube : Cube) (m : PointM cube) (n : PointN cube) : Prop :=
  sorry

/-- The length of segment MN -/
def segment_length (cube : Cube) (m : PointM cube) (n : PointN cube) : ℝ :=
  sorry

/-- The main theorem -/
theorem min_segment_length (cube : Cube) :
  ∃ (m : PointM cube) (n : PointN cube),
    is_parallel_to_diagonal_face cube m n ∧
    ∀ (m' : PointM cube) (n' : PointN cube),
      is_parallel_to_diagonal_face cube m' n' →
      segment_length cube m n ≤ segment_length cube m' n' ∧
      segment_length cube m n = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_segment_length_l1585_158501


namespace NUMINAMATH_CALUDE_work_ratio_l1585_158516

/-- The time (in days) it takes for worker A to complete the task alone -/
def time_A : ℝ := 6

/-- The time (in days) it takes for worker B to complete the task alone -/
def time_B : ℝ := 30

/-- The time (in days) it takes for workers A and B to complete the task together -/
def time_together : ℝ := 5

theorem work_ratio : 
  (1 / time_A + 1 / time_B = 1 / time_together) → 
  (time_A / time_B = 1 / 5) := by
  sorry

end NUMINAMATH_CALUDE_work_ratio_l1585_158516


namespace NUMINAMATH_CALUDE_tp_rolls_count_l1585_158526

/-- The time in seconds to clean up one egg -/
def egg_cleanup_time : ℕ := 15

/-- The time in minutes to clean up one roll of toilet paper -/
def tp_cleanup_time : ℕ := 30

/-- The total cleaning time in minutes -/
def total_cleanup_time : ℕ := 225

/-- The number of eggs to clean up -/
def num_eggs : ℕ := 60

/-- Theorem stating that the number of toilet paper rolls is 7 -/
theorem tp_rolls_count : ℕ := by
  sorry

end NUMINAMATH_CALUDE_tp_rolls_count_l1585_158526


namespace NUMINAMATH_CALUDE_expansion_and_reduction_l1585_158559

theorem expansion_and_reduction : 
  (234 * 205 = 47970) ∧ (86400 / 300 = 288) := by
  sorry

end NUMINAMATH_CALUDE_expansion_and_reduction_l1585_158559


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l1585_158506

theorem triangle_area_inequality (a b c T : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hT : T > 0) (triangle_area : T^2 = (a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c) / 16) :
  T^2 ≤ a * b * c * (a + b + c) / 16 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l1585_158506


namespace NUMINAMATH_CALUDE_marks_work_hours_l1585_158579

/-- Calculates the number of hours Mark works per day given his salary and expenses. -/
theorem marks_work_hours
  (old_hourly_wage : ℝ)
  (raise_percentage : ℝ)
  (days_per_week : ℕ)
  (old_bills : ℝ)
  (personal_trainer : ℝ)
  (leftover : ℝ)
  (h : old_hourly_wage = 40)
  (h1 : raise_percentage = 0.05)
  (h2 : days_per_week = 5)
  (h3 : old_bills = 600)
  (h4 : personal_trainer = 100)
  (h5 : leftover = 980) :
  ∃ (hours_per_day : ℝ),
    hours_per_day = 8 ∧
    (old_hourly_wage * (1 + raise_percentage) * hours_per_day * days_per_week) =
    (old_bills + personal_trainer + leftover) :=
by sorry

end NUMINAMATH_CALUDE_marks_work_hours_l1585_158579


namespace NUMINAMATH_CALUDE_parking_lot_problem_l1585_158581

theorem parking_lot_problem :
  ∀ (medium_cars small_cars : ℕ),
    medium_cars + small_cars = 36 →
    6 * medium_cars + 4 * small_cars = 176 →
    medium_cars = 16 ∧ small_cars = 20 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l1585_158581


namespace NUMINAMATH_CALUDE_pear_sales_l1585_158535

theorem pear_sales (morning_sales : ℝ) : 
  morning_sales > 0 →
  2 * morning_sales > 0 →
  3 * (2 * morning_sales) > 0 →
  morning_sales + 2 * morning_sales + 3 * (2 * morning_sales) = 510 →
  2 * morning_sales = 113.34 := by
sorry

end NUMINAMATH_CALUDE_pear_sales_l1585_158535


namespace NUMINAMATH_CALUDE_admin_teacher_ratio_l1585_158515

/-- The ratio of administrators to teachers at a graduation ceremony -/
theorem admin_teacher_ratio :
  let graduates : ℕ := 50
  let parents_per_graduate : ℕ := 2
  let teachers : ℕ := 20
  let total_chairs : ℕ := 180
  let grad_parent_chairs := graduates * (parents_per_graduate + 1)
  let admin_chairs := total_chairs - (grad_parent_chairs + teachers)
  (admin_chairs : ℚ) / teachers = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_admin_teacher_ratio_l1585_158515


namespace NUMINAMATH_CALUDE_valid_paths_count_l1585_158553

/-- Represents a face on the dodecahedron -/
inductive Face
| Top
| Bottom
| TopRing (n : Fin 5)
| BottomRing (n : Fin 5)

/-- Represents a valid path on the dodecahedron -/
def ValidPath : List Face → Prop :=
  sorry

/-- The specific face on the bottom ring that must be passed through -/
def SpecificBottomFace : Face :=
  Face.BottomRing 0

/-- A function that counts the number of valid paths -/
def CountValidPaths : ℕ :=
  sorry

/-- Theorem stating that the number of valid paths is 15 -/
theorem valid_paths_count :
  CountValidPaths = 15 :=
sorry

end NUMINAMATH_CALUDE_valid_paths_count_l1585_158553


namespace NUMINAMATH_CALUDE_defective_pens_l1585_158574

theorem defective_pens (total_pens : ℕ) (prob_non_defective : ℚ) : 
  total_pens = 12 →
  prob_non_defective = 7/33 →
  (∃ (defective : ℕ), 
    defective ≤ total_pens ∧ 
    (total_pens - defective : ℚ) / total_pens * ((total_pens - defective - 1) : ℚ) / (total_pens - 1) = prob_non_defective ∧
    defective = 4) := by
  sorry

end NUMINAMATH_CALUDE_defective_pens_l1585_158574


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l1585_158514

theorem complex_magnitude_theorem (b : ℝ) :
  (Complex.I * Complex.I.re = ((1 + b * Complex.I) * (2 + Complex.I)).re) →
  Complex.abs ((2 * b + 3 * Complex.I) / (1 + b * Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l1585_158514


namespace NUMINAMATH_CALUDE_factors_multiple_of_180_l1585_158590

/-- The number of natural-number factors of m that are multiples of 180 -/
def count_factors (m : ℕ) : ℕ :=
  sorry

theorem factors_multiple_of_180 :
  let m : ℕ := 2^12 * 3^15 * 5^9
  count_factors m = 1386 := by
  sorry

end NUMINAMATH_CALUDE_factors_multiple_of_180_l1585_158590
