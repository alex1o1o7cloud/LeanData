import Mathlib

namespace NUMINAMATH_CALUDE_sandys_water_goal_l2965_296597

/-- Sandy's water drinking goal problem -/
theorem sandys_water_goal (water_per_interval : ℕ) (hours_per_interval : ℕ) (total_hours : ℕ) : 
  water_per_interval = 500 →
  hours_per_interval = 2 →
  total_hours = 12 →
  (water_per_interval * (total_hours / hours_per_interval)) / 1000 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandys_water_goal_l2965_296597


namespace NUMINAMATH_CALUDE_circle_area_comparison_l2965_296541

theorem circle_area_comparison (r s : ℝ) (h : 2 * r = (3 + Real.sqrt 2) * s) :
  π * r^2 = ((11 + 6 * Real.sqrt 2) / 4) * (π * s^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_comparison_l2965_296541


namespace NUMINAMATH_CALUDE_diagonals_in_nonagon_l2965_296566

/-- The number of diagonals in a regular nine-sided polygon -/
theorem diagonals_in_nonagon : 
  (let n : ℕ := 9
   let total_connections := n.choose 2
   let num_sides := n
   total_connections - num_sides) = 27 := by
sorry

end NUMINAMATH_CALUDE_diagonals_in_nonagon_l2965_296566


namespace NUMINAMATH_CALUDE_stock_value_indeterminate_l2965_296553

theorem stock_value_indeterminate (yield : ℝ) (market_value : ℝ) 
  (h_yield : yield = 0.08) (h_market_value : market_value = 150) :
  ∀ original_value : ℝ, 
  (original_value > 0 ∧ yield * original_value = market_value) ∨
  (original_value > 0 ∧ yield * original_value ≠ market_value) :=
by sorry

end NUMINAMATH_CALUDE_stock_value_indeterminate_l2965_296553


namespace NUMINAMATH_CALUDE_scientist_news_sharing_l2965_296537

/-- Represents the state of scientists' knowledge before and after pairing -/
structure ScientistState where
  total : Nat
  initial_knowledgeable : Nat
  final_knowledgeable : Nat

/-- Probability of a specific final state given initial conditions -/
def probability (s : ScientistState) : Rat :=
  sorry

/-- Expected number of scientists knowing the news after pairing -/
def expected_final_knowledgeable (total : Nat) (initial_knowledgeable : Nat) : Rat :=
  sorry

/-- Main theorem about scientists and news sharing -/
theorem scientist_news_sharing :
  let s₁ : ScientistState := ⟨18, 10, 13⟩
  let s₂ : ScientistState := ⟨18, 10, 14⟩
  probability s₁ = 0 ∧
  probability s₂ = 1120 / 2431 ∧
  expected_final_knowledgeable 18 10 = 14^12 / 17 :=
by sorry

end NUMINAMATH_CALUDE_scientist_news_sharing_l2965_296537


namespace NUMINAMATH_CALUDE_square_root_problem_l2965_296547

theorem square_root_problem (a b c : ℝ) : 
  (a - 4)^(1/3) = 1 →
  (3 * a - b - 2)^(1/2) = 3 →
  c = ⌊Real.sqrt 13⌋ →
  (2 * a - 3 * b + c)^(1/2) = 1 ∨ (2 * a - 3 * b + c)^(1/2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l2965_296547


namespace NUMINAMATH_CALUDE_exists_ten_segments_no_triangle_l2965_296599

/-- A sequence of 10 positive real numbers in geometric progression -/
def geometricSequence : Fin 10 → ℝ
  | ⟨n, _⟩ => 2^n

/-- Predicate to check if three numbers can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- Theorem stating that there exists a set of 10 segments where no three segments can form a triangle -/
theorem exists_ten_segments_no_triangle :
  ∃ (s : Fin 10 → ℝ), ∀ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬(canFormTriangle (s i) (s j) (s k)) := by
  sorry

end NUMINAMATH_CALUDE_exists_ten_segments_no_triangle_l2965_296599


namespace NUMINAMATH_CALUDE_correct_calculation_l2965_296534

theorem correct_calculation (a b : ℝ) : 7 * a * b - 6 * a * b = a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2965_296534


namespace NUMINAMATH_CALUDE_first_month_sales_l2965_296533

def sales_second_month : ℤ := 8550
def sales_third_month : ℤ := 6855
def sales_fourth_month : ℤ := 3850
def sales_fifth_month : ℤ := 14045
def average_sale : ℤ := 7800
def num_months : ℤ := 5

theorem first_month_sales :
  (average_sale * num_months) - (sales_second_month + sales_third_month + sales_fourth_month + sales_fifth_month) = 8700 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sales_l2965_296533


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2965_296548

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2965_296548


namespace NUMINAMATH_CALUDE_fair_distribution_correctness_l2965_296581

/-- Represents the amount of bread each person has initially -/
structure BreadDistribution where
  personA : ℚ
  personB : ℚ

/-- Represents the fair distribution of currency -/
structure CurrencyDistribution where
  personA : ℚ
  personB : ℚ

/-- Calculates the fair distribution of currency based on initial bread distribution -/
def calculateFairDistribution (initial : BreadDistribution) (totalCurrency : ℚ) : CurrencyDistribution :=
  sorry

theorem fair_distribution_correctness 
  (initial : BreadDistribution)
  (h1 : initial.personA = 3)
  (h2 : initial.personB = 2)
  (totalCurrency : ℚ)
  (h3 : totalCurrency = 50) :
  let result := calculateFairDistribution initial totalCurrency
  result.personA = 40 ∧ result.personB = 10 := by
  sorry

end NUMINAMATH_CALUDE_fair_distribution_correctness_l2965_296581


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2965_296502

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (a - 1) * x^2

-- Define the property of f being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Main theorem
theorem solution_set_of_inequality (a : ℝ) :
  (is_odd_function (f a)) →
  {x : ℝ | f a (a * x) > f a (a - x)} = {x : ℝ | x > 1/2} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2965_296502


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l2965_296585

/-- Calculates the total money taken in on ticket sales given the prices and number of tickets sold. -/
def totalTicketSales (adultPrice childPrice : ℕ) (totalTickets adultTickets : ℕ) : ℕ :=
  adultPrice * adultTickets + childPrice * (totalTickets - adultTickets)

/-- Theorem stating that given the specific ticket prices and sales, the total money taken in is $206. -/
theorem theater_ticket_sales :
  totalTicketSales 8 5 34 12 = 206 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l2965_296585


namespace NUMINAMATH_CALUDE_wendy_recycling_points_l2965_296593

/-- Calculates the total points earned by Wendy for recycling cans and newspapers -/
def total_points (cans_recycled : ℕ) (newspapers_recycled : ℕ) : ℕ :=
  cans_recycled * 5 + newspapers_recycled * 10

/-- Proves that Wendy's total points earned is 75 given the problem conditions -/
theorem wendy_recycling_points :
  let cans_total : ℕ := 11
  let cans_recycled : ℕ := 9
  let newspapers_recycled : ℕ := 3
  total_points cans_recycled newspapers_recycled = 75 := by
  sorry

#eval total_points 9 3

end NUMINAMATH_CALUDE_wendy_recycling_points_l2965_296593


namespace NUMINAMATH_CALUDE_flower_bed_lilies_l2965_296596

/-- Given a flower bed with roses, tulips, and lilies, prove the number of lilies. -/
theorem flower_bed_lilies (roses tulips lilies : ℕ) : 
  roses = 57 → 
  tulips = 82 → 
  tulips = roses + lilies + 13 → 
  lilies = 12 := by
sorry


end NUMINAMATH_CALUDE_flower_bed_lilies_l2965_296596


namespace NUMINAMATH_CALUDE_x_value_l2965_296518

theorem x_value (h1 : 25 * x^2 - 9 = 7) (h2 : 8 * (x - 2)^3 = 27) : x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2965_296518


namespace NUMINAMATH_CALUDE_factorial_sum_of_powers_of_two_l2965_296554

theorem factorial_sum_of_powers_of_two (n : ℕ) :
  (∃ a b : ℕ, n.factorial = 2^a + 2^b) ↔ n = 3 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_of_powers_of_two_l2965_296554


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l2965_296551

def num_islands : ℕ := 8
def prob_treasure_no_traps : ℚ := 1/3
def prob_treasure_and_traps : ℚ := 1/6
def prob_traps_no_treasure : ℚ := 1/6
def prob_neither : ℚ := 1/3

def target_treasure_islands : ℕ := 4
def target_treasure_and_traps_islands : ℕ := 2

theorem pirate_treasure_probability :
  let prob_treasure := prob_treasure_no_traps + prob_treasure_and_traps
  let prob_non_treasure := prob_traps_no_treasure + prob_neither
  (Nat.choose num_islands target_treasure_islands) *
  (Nat.choose target_treasure_islands target_treasure_and_traps_islands) *
  (prob_treasure ^ target_treasure_islands) *
  (prob_treasure_and_traps ^ target_treasure_and_traps_islands) *
  (prob_treasure_no_traps ^ (target_treasure_islands - target_treasure_and_traps_islands)) *
  (prob_non_treasure ^ (num_islands - target_treasure_islands)) =
  105 / 104976 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l2965_296551


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2965_296584

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating that the intersection point of lines AB and CD is (-4/3, 35, 3/2) --/
theorem intersection_of_lines :
  let A : ℝ × ℝ × ℝ := (3, -5, 4)
  let B : ℝ × ℝ × ℝ := (13, -15, 9)
  let C : ℝ × ℝ × ℝ := (-6, 6, -12)
  let D : ℝ × ℝ × ℝ := (-4, -2, 8)
  intersection_point A B C D = (-4/3, 35, 3/2) := by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2965_296584


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_max_value_at_zero_l2965_296512

theorem max_value_of_sum_of_roots (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (x + 15) + Real.sqrt (17 - x) + Real.sqrt x ≤ Real.sqrt 15 + Real.sqrt 17 :=
by sorry

theorem max_value_at_zero :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 17 ∧
  Real.sqrt (x + 15) + Real.sqrt (17 - x) + Real.sqrt x = Real.sqrt 15 + Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_max_value_at_zero_l2965_296512


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2965_296501

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 9 → b = 9 → c = 4 →
  (a + b > c ∧ b + c > a ∧ a + c > b) →  -- Triangle inequality
  a + b + c = 22 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2965_296501


namespace NUMINAMATH_CALUDE_complex_inequality_l2965_296538

theorem complex_inequality (x y a b : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l2965_296538


namespace NUMINAMATH_CALUDE_clubsuit_difference_l2965_296520

/-- The clubsuit operation -/
def clubsuit (x y : ℝ) : ℝ := 4*x + 6*y

/-- Theorem stating that (5 ♣ 3) - (1 ♣ 4) = 10 -/
theorem clubsuit_difference : (clubsuit 5 3) - (clubsuit 1 4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_difference_l2965_296520


namespace NUMINAMATH_CALUDE_min_value_expression_l2965_296540

theorem min_value_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m : ℝ), (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → a^2 + b^2 + 4/a^2 + b/a ≥ m) ∧
             (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ c^2 + d^2 + 4/c^2 + d/c = m) ∧
             m = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2965_296540


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2965_296522

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem problem_solution :
  ∃ (k : ℕ), k < 17 ∧ (9857621 - k) % 17 = 0 ∧ ∀ (m : ℕ), m < k → (9857621 - m) % 17 ≠ 0 ∧ k = 8 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2965_296522


namespace NUMINAMATH_CALUDE_tim_change_l2965_296565

/-- The change Tim received after buying a candy bar -/
def change (initial_amount : ℕ) (candy_cost : ℕ) : ℕ :=
  initial_amount - candy_cost

/-- Theorem stating that Tim's change is 5 cents -/
theorem tim_change :
  change 50 45 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tim_change_l2965_296565


namespace NUMINAMATH_CALUDE_partition_has_all_distances_l2965_296583

-- Define a partition of a metric space into three sets
def Partition (X : Type*) [MetricSpace X] (M₁ M₂ M₃ : Set X) : Prop :=
  (M₁ ∪ M₂ ∪ M₃ = Set.univ) ∧ (M₁ ∩ M₂ = ∅) ∧ (M₁ ∩ M₃ = ∅) ∧ (M₂ ∩ M₃ = ∅)

-- Define the property that a set contains two points with any positive distance
def HasAllDistances (X : Type*) [MetricSpace X] (M : Set X) : Prop :=
  ∀ a : ℝ, a > 0 → ∃ x y : X, x ∈ M ∧ y ∈ M ∧ dist x y = a

-- State the theorem
theorem partition_has_all_distances (X : Type*) [MetricSpace X] (M₁ M₂ M₃ : Set X) 
  (h : Partition X M₁ M₂ M₃) : 
  HasAllDistances X M₁ ∨ HasAllDistances X M₂ ∨ HasAllDistances X M₃ := by
  sorry


end NUMINAMATH_CALUDE_partition_has_all_distances_l2965_296583


namespace NUMINAMATH_CALUDE_T_formula_l2965_296556

def T : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * T (n + 2) - 4 * (n + 3) * T (n + 1) + (4 * (n + 3) - 8) * T n

theorem T_formula (n : ℕ) : T n = n.factorial + 2^n := by
  sorry

end NUMINAMATH_CALUDE_T_formula_l2965_296556


namespace NUMINAMATH_CALUDE_max_visible_cubes_is_274_l2965_296592

/-- The dimension of the cube --/
def n : ℕ := 10

/-- The total number of unit cubes in the cube --/
def total_cubes : ℕ := n^3

/-- The number of unit cubes on one face of the cube --/
def face_cubes : ℕ := n^2

/-- The number of visible faces from a corner --/
def visible_faces : ℕ := 3

/-- The number of shared edges between visible faces --/
def shared_edges : ℕ := 3

/-- The length of each edge --/
def edge_length : ℕ := n

/-- The number of unit cubes along a shared edge, excluding the corner --/
def edge_cubes : ℕ := edge_length - 1

/-- The maximum number of visible unit cubes from a single point --/
def max_visible_cubes : ℕ := visible_faces * face_cubes - shared_edges * edge_cubes + 1

theorem max_visible_cubes_is_274 : max_visible_cubes = 274 := by
  sorry

end NUMINAMATH_CALUDE_max_visible_cubes_is_274_l2965_296592


namespace NUMINAMATH_CALUDE_biology_quiz_probability_l2965_296511

/-- The number of questions in the quiz --/
def total_questions : ℕ := 20

/-- The number of questions Jessica guesses randomly --/
def guessed_questions : ℕ := 5

/-- The number of answer choices for each question --/
def answer_choices : ℕ := 4

/-- The probability of getting a single question correct by random guessing --/
def prob_correct : ℚ := 1 / answer_choices

/-- The probability of getting at least two questions correct out of five randomly guessed questions --/
def prob_at_least_two_correct : ℚ := 47 / 128

theorem biology_quiz_probability :
  (1 : ℚ) - (Nat.choose guessed_questions 0 * (1 - prob_correct)^guessed_questions +
             Nat.choose guessed_questions 1 * (1 - prob_correct)^(guessed_questions - 1) * prob_correct) =
  prob_at_least_two_correct :=
sorry

end NUMINAMATH_CALUDE_biology_quiz_probability_l2965_296511


namespace NUMINAMATH_CALUDE_gcd_266_209_l2965_296586

theorem gcd_266_209 : Nat.gcd 266 209 = 19 := by
  sorry

end NUMINAMATH_CALUDE_gcd_266_209_l2965_296586


namespace NUMINAMATH_CALUDE_xy_value_l2965_296579

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2965_296579


namespace NUMINAMATH_CALUDE_max_area_AOB_l2965_296563

-- Define the circles E and F
def circle_E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 25
def circle_F (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 1

-- Define the curve C (locus of center of P)
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define a line l
def line_l (m n x y : ℝ) : Prop := x = m * y + n

-- Define points A and B on curve C and line l
def point_on_C_and_l (x y m n : ℝ) : Prop :=
  curve_C x y ∧ line_l m n x y

-- Define midpoint M of AB
def midpoint_M (xm ym xa ya xb yb : ℝ) : Prop :=
  xm = (xa + xb) / 2 ∧ ym = (ya + yb) / 2

-- Define |OM| = 1
def OM_unit_length (xm ym : ℝ) : Prop :=
  xm^2 + ym^2 = 1

-- Main theorem
theorem max_area_AOB :
  ∀ (xa ya xb yb xm ym m n : ℝ),
  point_on_C_and_l xa ya m n →
  point_on_C_and_l xb yb m n →
  midpoint_M xm ym xa ya xb yb →
  OM_unit_length xm ym →
  ∃ (S : ℝ), S ≤ 1 ∧
  (∀ (S' : ℝ), S' = abs ((xa * yb - xb * ya) / 2) → S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_max_area_AOB_l2965_296563


namespace NUMINAMATH_CALUDE_ellipse_equation_l2965_296578

/-- An ellipse with specific properties -/
structure Ellipse where
  -- Major axis is on the x-axis
  majorAxisOnX : Bool
  -- Length of the major axis
  majorAxisLength : ℝ
  -- Eccentricity
  eccentricity : ℝ
  -- Point on the ellipse
  pointOnEllipse : ℝ × ℝ

/-- The standard equation of an ellipse -/
def standardEquation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem stating the standard equation of the ellipse with given properties -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.majorAxisOnX = true)
  (h2 : e.majorAxisLength = 12)
  (h3 : e.eccentricity = 2/3)
  (h4 : e.pointOnEllipse = (-2, -4)) :
  standardEquation 36 20 = standardEquation 36 20 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2965_296578


namespace NUMINAMATH_CALUDE_term_free_of_x_l2965_296544

theorem term_free_of_x (m n k : ℕ) : 
  (∃ r : ℕ, r ≤ k ∧ m * k - (m + n) * r = 0) ↔ (m * k) % (m + n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_term_free_of_x_l2965_296544


namespace NUMINAMATH_CALUDE_median_siblings_is_two_l2965_296588

/-- Represents the number of students for each sibling count -/
def sibling_distribution : List (Nat × Nat) :=
  [(0, 2), (1, 3), (2, 2), (3, 1), (4, 2), (5, 1)]

/-- Calculates the total number of students -/
def total_students : Nat :=
  sibling_distribution.foldl (fun acc (_, count) => acc + count) 0

/-- Finds the median position -/
def median_position : Nat :=
  (total_students + 1) / 2

/-- Theorem: The median number of siblings in Mrs. Thompson's History class is 2 -/
theorem median_siblings_is_two :
  let cumulative_count := sibling_distribution.foldl
    (fun acc (siblings, count) => 
      match acc with
      | [] => [(siblings, count)]
      | (_, prev_count) :: _ => (siblings, prev_count + count) :: acc
    ) []
  cumulative_count.reverse.find? (fun (_, count) => count ≥ median_position)
    = some (2, 7) := by sorry

end NUMINAMATH_CALUDE_median_siblings_is_two_l2965_296588


namespace NUMINAMATH_CALUDE_problem_statement_l2965_296527

theorem problem_statement (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2965_296527


namespace NUMINAMATH_CALUDE_smallest_cube_divisor_l2965_296539

theorem smallest_cube_divisor (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  let m := a^3 * b^5 * c^7
  ∀ k : ℕ, k^3 ∣ m → (a * b * c^3)^3 ≤ k^3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_divisor_l2965_296539


namespace NUMINAMATH_CALUDE_quadratic_maximum_quadratic_maximum_achieved_l2965_296568

theorem quadratic_maximum (r : ℝ) : -3 * r^2 + 30 * r + 24 ≤ 99 :=
sorry

theorem quadratic_maximum_achieved : ∃ r : ℝ, -3 * r^2 + 30 * r + 24 = 99 :=
sorry

end NUMINAMATH_CALUDE_quadratic_maximum_quadratic_maximum_achieved_l2965_296568


namespace NUMINAMATH_CALUDE_M_not_finite_union_of_aps_l2965_296590

-- Define the set M
def M : Set ℕ := {n : ℕ | ∀ x y : ℕ, (1 : ℚ) / x + (1 : ℚ) / y ≠ 3 / n}

-- Define what it means for a set to be representable as a finite union of arithmetic progressions
def is_finite_union_of_aps (S : Set ℕ) : Prop :=
  ∃ (n : ℕ) (a d : Fin n → ℕ), S = ⋃ i, {k : ℕ | ∃ j : ℕ, k = a i + j * d i}

-- State the theorem
theorem M_not_finite_union_of_aps :
  (∀ n : ℕ, n ∉ M → ∀ m : ℕ, m * n ∉ M) →
  (∀ k : ℕ, k > 0 → (7 : ℕ) ^ k ∈ M) →
  ¬ is_finite_union_of_aps M :=
sorry

end NUMINAMATH_CALUDE_M_not_finite_union_of_aps_l2965_296590


namespace NUMINAMATH_CALUDE_saturday_to_monday_ratio_is_two_to_one_l2965_296513

/-- Represents Mona's weekly biking schedule -/
structure BikeSchedule where
  total_distance : ℕ
  monday_distance : ℕ
  wednesday_distance : ℕ
  saturday_distance : ℕ
  total_eq : total_distance = monday_distance + wednesday_distance + saturday_distance

/-- Calculates the ratio of Saturday's distance to Monday's distance -/
def saturday_to_monday_ratio (schedule : BikeSchedule) : ℚ :=
  schedule.saturday_distance / schedule.monday_distance

/-- Theorem stating that the ratio of Saturday's distance to Monday's distance is 2:1 -/
theorem saturday_to_monday_ratio_is_two_to_one (schedule : BikeSchedule)
  (h1 : schedule.total_distance = 30)
  (h2 : schedule.monday_distance = 6)
  (h3 : schedule.wednesday_distance = 12) :
  saturday_to_monday_ratio schedule = 2 := by
  sorry

#eval saturday_to_monday_ratio {
  total_distance := 30,
  monday_distance := 6,
  wednesday_distance := 12,
  saturday_distance := 12,
  total_eq := by rfl
}

end NUMINAMATH_CALUDE_saturday_to_monday_ratio_is_two_to_one_l2965_296513


namespace NUMINAMATH_CALUDE_min_pencils_in_box_l2965_296523

theorem min_pencils_in_box (total_pencils : ℕ) (num_boxes : ℕ) (max_capacity : ℕ)
  (h1 : total_pencils = 74)
  (h2 : num_boxes = 13)
  (h3 : max_capacity = 6) :
  ∃ (min_pencils : ℕ), 
    (∀ (box : ℕ), box ≤ num_boxes → min_pencils ≤ (total_pencils / num_boxes)) ∧
    (∃ (box : ℕ), box ≤ num_boxes ∧ (total_pencils / num_boxes) - min_pencils < 1) ∧
    min_pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_pencils_in_box_l2965_296523


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2965_296529

theorem fraction_multiplication : (2 : ℚ) / 15 * 5 / 8 = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2965_296529


namespace NUMINAMATH_CALUDE_max_value_of_2xy_l2965_296571

theorem max_value_of_2xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 → 2 * x * y ≤ 2 * a * b → 2 * x * y ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2xy_l2965_296571


namespace NUMINAMATH_CALUDE_factorization_condition_l2965_296582

-- Define the polynomial
def polynomial (x y m : ℤ) : ℤ := x^2 + 5*x*y + x + 2*m*y - 10

-- Define what it means for a polynomial to be factorizable into linear factors with integer coefficients
def is_factorizable (m : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ (x y : ℤ),
    polynomial x y m = (a*x + b*y + c) * (d*x + e*y + f)

-- State the theorem
theorem factorization_condition :
  ∀ m : ℤ, is_factorizable m ↔ m = 5 := by sorry

end NUMINAMATH_CALUDE_factorization_condition_l2965_296582


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2965_296516

/-- Given a train of length 240 m crossing a platform of equal length in 27 s,
    its speed is approximately 64 km/h. -/
theorem train_speed_calculation (train_length platform_length : ℝ)
  (crossing_time : ℝ) (h1 : train_length = 240)
  (h2 : platform_length = train_length) (h3 : crossing_time = 27) :
  ∃ (speed : ℝ), abs (speed - 64) < 0.5 ∧ speed = (train_length + platform_length) / crossing_time * 3.6 :=
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2965_296516


namespace NUMINAMATH_CALUDE_line_points_Q_value_l2965_296552

/-- Given a line x = 8y + 5 passing through points (m, n) and (m + Q, n + p), where p = 0.25,
    prove that Q = 2. -/
theorem line_points_Q_value (m n Q p : ℝ) : 
  p = 0.25 →
  m = 8 * n + 5 →
  m + Q = 8 * (n + p) + 5 →
  Q = 2 := by
sorry

end NUMINAMATH_CALUDE_line_points_Q_value_l2965_296552


namespace NUMINAMATH_CALUDE_ocean_area_scientific_notation_l2965_296560

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem ocean_area_scientific_notation :
  toScientificNotation 361000000 = ScientificNotation.mk 3.61 8 sorry := by
  sorry

end NUMINAMATH_CALUDE_ocean_area_scientific_notation_l2965_296560


namespace NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l2965_296536

-- Define a real-valued function on the real line
variable (f : ℝ → ℝ)

-- Assume f is differentiable
variable (hf : Differentiable ℝ f)

-- Define what it means for f to have an extremum at a point
def has_extremum_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- State the theorem
theorem derivative_zero_necessary_not_sufficient :
  (∀ x : ℝ, has_extremum_at f x → deriv f x = 0) ∧
  ¬(∀ x : ℝ, deriv f x = 0 → has_extremum_at f x) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l2965_296536


namespace NUMINAMATH_CALUDE_sin_15_cos_15_half_l2965_296532

theorem sin_15_cos_15_half : 2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_half_l2965_296532


namespace NUMINAMATH_CALUDE_intersection_distance_l2965_296575

theorem intersection_distance (m b k : ℝ) (h1 : b ≠ 0) (h2 : 1 = 2 * m + b) :
  let f := fun x => x^2 + 6 * x - 4
  let g := fun x => m * x + b
  let d := |f k - g k|
  (m = 4 ∧ b = -7) → d = 9 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l2965_296575


namespace NUMINAMATH_CALUDE_bus_seat_capacity_l2965_296574

theorem bus_seat_capacity (left_seats right_seats back_seat_capacity total_capacity : ℕ) 
  (h1 : left_seats = 15)
  (h2 : right_seats = left_seats - 3)
  (h3 : back_seat_capacity = 8)
  (h4 : total_capacity = 89) :
  ∃ (seat_capacity : ℕ), 
    seat_capacity * (left_seats + right_seats) + back_seat_capacity = total_capacity ∧ 
    seat_capacity = 3 := by
sorry

end NUMINAMATH_CALUDE_bus_seat_capacity_l2965_296574


namespace NUMINAMATH_CALUDE_laundry_theorem_l2965_296561

/-- Represents the laundry problem --/
structure LaundryProblem where
  machine_capacity : ℕ  -- in pounds
  shirts_per_pound : ℕ
  pants_pairs_per_pound : ℕ
  shirts_to_wash : ℕ
  loads : ℕ

/-- Calculates the number of pants pairs that can be washed --/
def pants_to_wash (p : LaundryProblem) : ℕ :=
  let total_capacity := p.machine_capacity * p.loads
  let shirt_weight := p.shirts_to_wash / p.shirts_per_pound
  let remaining_capacity := total_capacity - shirt_weight
  remaining_capacity * p.pants_pairs_per_pound

/-- States the theorem for the laundry problem --/
theorem laundry_theorem (p : LaundryProblem) 
  (h1 : p.machine_capacity = 5)
  (h2 : p.shirts_per_pound = 4)
  (h3 : p.pants_pairs_per_pound = 2)
  (h4 : p.shirts_to_wash = 20)
  (h5 : p.loads = 3) :
  pants_to_wash p = 20 := by
  sorry

#eval pants_to_wash { 
  machine_capacity := 5,
  shirts_per_pound := 4,
  pants_pairs_per_pound := 2,
  shirts_to_wash := 20,
  loads := 3
}

end NUMINAMATH_CALUDE_laundry_theorem_l2965_296561


namespace NUMINAMATH_CALUDE_quadrilateral_ae_length_l2965_296528

/-- Represents a convex quadrilateral ABCD with point E at the intersection of diagonals -/
structure ConvexQuadrilateral :=
  (A B C D E : ℝ × ℝ)

/-- Properties of the specific quadrilateral in the problem -/
def QuadrilateralProperties (quad : ConvexQuadrilateral) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist quad.A quad.B = 10 ∧
  dist quad.C quad.D = 15 ∧
  dist quad.A quad.C = 17 ∧
  (quad.E.1 - quad.A.1) * (quad.D.2 - quad.A.2) = (quad.E.2 - quad.A.2) * (quad.D.1 - quad.A.1) ∧
  (quad.E.1 - quad.B.1) * (quad.C.2 - quad.B.2) = (quad.E.2 - quad.B.2) * (quad.C.1 - quad.B.1)

theorem quadrilateral_ae_length 
  (quad : ConvexQuadrilateral) 
  (h : QuadrilateralProperties quad) : 
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist quad.A quad.E = 6.8 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_ae_length_l2965_296528


namespace NUMINAMATH_CALUDE_max_value_sum_reciprocals_l2965_296558

theorem max_value_sum_reciprocals (a b : ℝ) (h : a + b = 4) :
  (∃ x y : ℝ, x + y = 4 ∧ (1 / (x^2 + 1) + 1 / (y^2 + 1) ≤ 1 / (a^2 + 1) + 1 / (b^2 + 1))) ∧
  1 / (a^2 + 1) + 1 / (b^2 + 1) ≤ (Real.sqrt 5 + 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_reciprocals_l2965_296558


namespace NUMINAMATH_CALUDE_square_root_meaningful_implies_x_geq_5_l2965_296576

theorem square_root_meaningful_implies_x_geq_5 (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_meaningful_implies_x_geq_5_l2965_296576


namespace NUMINAMATH_CALUDE_unripe_oranges_eaten_l2965_296550

theorem unripe_oranges_eaten (total : ℕ) (uneaten : ℕ) : 
  total = 96 →
  uneaten = 78 →
  (1 : ℚ) / 8 = (total / 2 - uneaten) / (total / 2) := by
  sorry

end NUMINAMATH_CALUDE_unripe_oranges_eaten_l2965_296550


namespace NUMINAMATH_CALUDE_triangle_ratio_equation_l2965_296572

/-- In a triangle ABC, given the ratios of sides to heights, prove the equation. -/
theorem triangle_ratio_equation (a b c h_a h_b h_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0) 
  (h_triangle : h_a * b = h_b * a ∧ h_b * c = h_c * b ∧ h_c * a = h_a * c) 
  (x y z : ℝ) (h_x : x = a / h_a) (h_y : y = b / h_b) (h_z : z = c / h_c) : 
  x^2 + y^2 + z^2 - 2*x*y - 2*y*z - 2*z*x + 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_equation_l2965_296572


namespace NUMINAMATH_CALUDE_solve_jogging_problem_l2965_296535

def jogging_problem (daily_time : ℕ) (first_week_days : ℕ) (total_time : ℕ) : Prop :=
  let total_minutes : ℕ := total_time * 60
  let first_week_minutes : ℕ := first_week_days * daily_time
  let second_week_minutes : ℕ := total_minutes - first_week_minutes
  let second_week_days : ℕ := second_week_minutes / daily_time
  second_week_days = 5

theorem solve_jogging_problem :
  jogging_problem 30 3 4 := by sorry

end NUMINAMATH_CALUDE_solve_jogging_problem_l2965_296535


namespace NUMINAMATH_CALUDE_unique_subset_with_nonempty_intersection_l2965_296543

def A : Set ℕ := {1, 2, 3, 4, 5, 6}
def B : Set ℕ := {4, 5, 6, 7, 8}

theorem unique_subset_with_nonempty_intersection :
  ∃! S : Set ℕ, S ⊆ A ∧ S ∩ B ≠ ∅ ∧ S = {5, 6} := by sorry

end NUMINAMATH_CALUDE_unique_subset_with_nonempty_intersection_l2965_296543


namespace NUMINAMATH_CALUDE_archibald_apple_eating_l2965_296515

theorem archibald_apple_eating (apples_per_day_first_two_weeks : ℕ) 
  (apples_per_day_last_two_weeks : ℕ) (total_weeks : ℕ) (average_apples_per_week : ℕ) :
  apples_per_day_first_two_weeks = 1 →
  apples_per_day_last_two_weeks = 3 →
  total_weeks = 7 →
  average_apples_per_week = 10 →
  ∃ (weeks_same_as_first_two : ℕ),
    weeks_same_as_first_two = 2 ∧
    (2 * 7 * apples_per_day_first_two_weeks) + 
    (weeks_same_as_first_two * 7 * apples_per_day_first_two_weeks) + 
    (2 * 7 * apples_per_day_last_two_weeks) = 
    total_weeks * average_apples_per_week :=
by sorry

end NUMINAMATH_CALUDE_archibald_apple_eating_l2965_296515


namespace NUMINAMATH_CALUDE_erased_number_proof_l2965_296573

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  x ≤ n →
  (n * (n + 1) / 2 - x : ℚ) / (n - 1 : ℚ) = 35 + 7/17 →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l2965_296573


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l2965_296525

theorem restaurant_bill_proof (n : ℕ) (extra : ℚ) (total_bill : ℚ) : 
  n = 10 →
  extra = 3 →
  (n - 1) * ((total_bill / n) + extra) = total_bill →
  total_bill = 270 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l2965_296525


namespace NUMINAMATH_CALUDE_unique_solution_to_exponential_equation_l2965_296549

theorem unique_solution_to_exponential_equation :
  ∀ x y z : ℕ, 3^x + 4^y = 5^z ↔ x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_exponential_equation_l2965_296549


namespace NUMINAMATH_CALUDE_store_desktop_sales_l2965_296503

/-- Given a ratio of laptops to desktops and an expected number of laptop sales,
    calculate the expected number of desktop sales. -/
def expected_desktop_sales (laptop_ratio : ℕ) (desktop_ratio : ℕ) (expected_laptops : ℕ) : ℕ :=
  (expected_laptops * desktop_ratio) / laptop_ratio

/-- Proof that given the specific ratio and expected laptop sales,
    the expected desktop sales is 24. -/
theorem store_desktop_sales : expected_desktop_sales 5 3 40 = 24 := by
  sorry

#eval expected_desktop_sales 5 3 40

end NUMINAMATH_CALUDE_store_desktop_sales_l2965_296503


namespace NUMINAMATH_CALUDE_rectangle_area_l2965_296587

/-- Given a rectangle where the length is four times the width and the perimeter is 250 cm,
    prove that its area is 2500 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 250 → l * w = 2500 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2965_296587


namespace NUMINAMATH_CALUDE_square_side_length_l2965_296564

theorem square_side_length (d : ℝ) (s : ℝ) : d = 2 * Real.sqrt 2 → s * Real.sqrt 2 = d → s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2965_296564


namespace NUMINAMATH_CALUDE_boys_transferred_l2965_296542

theorem boys_transferred (initial_boys : ℕ) (initial_ratio_boys : ℕ) (initial_ratio_girls : ℕ)
  (final_ratio_boys : ℕ) (final_ratio_girls : ℕ) :
  initial_boys = 120 →
  initial_ratio_boys = 3 →
  initial_ratio_girls = 4 →
  final_ratio_boys = 4 →
  final_ratio_girls = 5 →
  ∃ (transferred_boys : ℕ),
    transferred_boys = 13 ∧
    ∃ (initial_girls : ℕ),
      initial_girls * initial_ratio_boys = initial_boys * initial_ratio_girls ∧
      (initial_boys - transferred_boys) * final_ratio_girls = 
      (initial_girls - 2 * transferred_boys) * final_ratio_boys :=
by sorry

end NUMINAMATH_CALUDE_boys_transferred_l2965_296542


namespace NUMINAMATH_CALUDE_sum_of_squares_bound_l2965_296506

theorem sum_of_squares_bound {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^4 + y^4 + z^4 = 1) : x^2 + y^2 + z^2 < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_bound_l2965_296506


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l2965_296555

-- Define the vertices of the tetrahedron
def A₁ : ℝ × ℝ × ℝ := (1, 5, -7)
def A₂ : ℝ × ℝ × ℝ := (-3, 6, 3)
def A₃ : ℝ × ℝ × ℝ := (-2, 7, 3)
def A₄ : ℝ × ℝ × ℝ := (-4, 8, -12)

-- Define a function to calculate the volume of a tetrahedron
def tetrahedronVolume (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the height of a tetrahedron
def tetrahedronHeight (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem stating the volume and height of the tetrahedron
theorem tetrahedron_properties :
  tetrahedronVolume A₁ A₂ A₃ A₄ = 17.5 ∧
  tetrahedronHeight A₁ A₂ A₃ A₄ = 7 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l2965_296555


namespace NUMINAMATH_CALUDE_set_intersection_complement_l2965_296530

theorem set_intersection_complement (U A B : Set ℤ) : 
  U = Set.univ ∧ A = {-1, 1, 2} ∧ B = {-1, 1} → A ∩ (U \ B) = {2} :=
by
  sorry

end NUMINAMATH_CALUDE_set_intersection_complement_l2965_296530


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_B_union_C_eq_B_iff_m_lt_4_l2965_296514

-- Define the sets A, B, and C
def A : Set ℝ := {x | (x - 7) / (x + 2) > 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 3*x + 28)}
def C (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem 1: Prove that (complement of A) ∩ B = [-2, 7)
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = Set.Icc (-2) 7 := by sorry

-- Theorem 2: Prove that B ∪ C = B if and only if m < 4
theorem B_union_C_eq_B_iff_m_lt_4 (m : ℝ) :
  B ∪ C m = B ↔ m < 4 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_B_union_C_eq_B_iff_m_lt_4_l2965_296514


namespace NUMINAMATH_CALUDE_george_total_blocks_l2965_296569

/-- The total number of blocks George has when combining large, small, and medium blocks. -/
def total_blocks (large_boxes small_boxes large_per_box small_per_box case_boxes medium_per_box : ℕ) : ℕ :=
  (large_boxes * large_per_box) + (small_boxes * small_per_box) + (case_boxes * medium_per_box)

/-- Theorem stating that George has 86 blocks in total. -/
theorem george_total_blocks :
  total_blocks 2 3 6 8 5 10 = 86 := by
  sorry

end NUMINAMATH_CALUDE_george_total_blocks_l2965_296569


namespace NUMINAMATH_CALUDE_magic_square_base_is_three_l2965_296519

/-- Represents a 3x3 magic square with elements in base b -/
def MagicSquare (b : ℕ) : Type :=
  Fin 3 → Fin 3 → ℕ

/-- The sum of a row, column, or diagonal in the magic square -/
def MagicSum (b : ℕ) (square : MagicSquare b) : ℕ :=
  square 0 0 + square 0 1 + square 0 2

/-- Predicate to check if a given square is magic -/
def IsMagicSquare (b : ℕ) (square : MagicSquare b) : Prop :=
  (∀ i : Fin 3, square i 0 + square i 1 + square i 2 = MagicSum b square) ∧
  (∀ j : Fin 3, square 0 j + square 1 j + square 2 j = MagicSum b square) ∧
  (square 0 0 + square 1 1 + square 2 2 = MagicSum b square) ∧
  (square 0 2 + square 1 1 + square 2 0 = MagicSum b square)

/-- The specific magic square given in the problem -/
def GivenSquare (b : ℕ) : MagicSquare b :=
  fun i j => match i, j with
  | 0, 0 => 5
  | 0, 1 => 11
  | 0, 2 => 15
  | 1, 0 => 4
  | 1, 1 => 11
  | 1, 2 => 12
  | 2, 0 => 14
  | 2, 1 => 2
  | 2, 2 => 3

theorem magic_square_base_is_three :
  ∃ (b : ℕ), b > 1 ∧ IsMagicSquare b (GivenSquare b) ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_magic_square_base_is_three_l2965_296519


namespace NUMINAMATH_CALUDE_angle_triple_supplement_l2965_296589

theorem angle_triple_supplement (x : ℝ) : x = 3 * (180 - x) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_l2965_296589


namespace NUMINAMATH_CALUDE_m_range_l2965_296567

theorem m_range (x m : ℝ) : 
  (∀ x, x^2 + 3*x - 4 < 0 → (x - m)^2 > 3*(x - m)) ∧ 
  (∃ x, (x - m)^2 > 3*(x - m) ∧ x^2 + 3*x - 4 ≥ 0) → 
  m ≥ 1 ∨ m ≤ -7 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2965_296567


namespace NUMINAMATH_CALUDE_pills_per_week_calculation_l2965_296580

/-- Calculates the number of pills taken in a week given the frequency of pill intake -/
def pills_per_week (hours_between_pills : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  (hours_per_day / hours_between_pills) * days_per_week

/-- Theorem stating that taking a pill every 6 hours results in 28 pills per week -/
theorem pills_per_week_calculation :
  pills_per_week 6 24 7 = 28 := by
  sorry

#eval pills_per_week 6 24 7

end NUMINAMATH_CALUDE_pills_per_week_calculation_l2965_296580


namespace NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l2965_296591

theorem cubic_root_reciprocal_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 26*a - 8 = 0 → 
  b^3 - 15*b^2 + 26*b - 8 = 0 → 
  c^3 - 15*c^2 + 26*c - 8 = 0 → 
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 109/16 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l2965_296591


namespace NUMINAMATH_CALUDE_log_inequality_l2965_296517

theorem log_inequality : (Real.log 2 / Real.log 3) < (Real.log 3 / Real.log 2) ∧ (Real.log 3 / Real.log 2) < (Real.log 5 / Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2965_296517


namespace NUMINAMATH_CALUDE_student_count_problem_l2965_296557

theorem student_count_problem (A B : ℕ) : 
  A = (5 : ℕ) * B / (7 : ℕ) →
  A + 3 = (4 : ℕ) * (B - 3) / (5 : ℕ) →
  A = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_student_count_problem_l2965_296557


namespace NUMINAMATH_CALUDE_rectangle_to_square_dissection_l2965_296594

theorem rectangle_to_square_dissection :
  ∃ (a b c d : ℝ),
    -- Rectangle dimensions
    16 * 9 = a * b + c * d ∧
    -- Two parts form a square
    12 * 12 = a * b + c * d ∧
    -- Dimensions are positive
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    -- One dimension of each part matches the square
    (a = 12 ∨ b = 12 ∨ c = 12 ∨ d = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_dissection_l2965_296594


namespace NUMINAMATH_CALUDE_tan_negative_three_pi_fourth_l2965_296570

theorem tan_negative_three_pi_fourth : Real.tan (-3 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_three_pi_fourth_l2965_296570


namespace NUMINAMATH_CALUDE_octagon_diagonals_l2965_296505

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diagonals :
  num_diagonals 8 = 20 := by sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l2965_296505


namespace NUMINAMATH_CALUDE_sticker_distribution_l2965_296531

theorem sticker_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (Nat.choose (n + k - 1) (k - 1)) = 1001 :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2965_296531


namespace NUMINAMATH_CALUDE_square_side_length_l2965_296510

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * Real.sqrt 2 = diagonal ∧ side = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2965_296510


namespace NUMINAMATH_CALUDE_distance_on_number_line_l2965_296508

theorem distance_on_number_line (a b : ℝ) (ha : a = 5) (hb : b = -3) :
  |a - b| = 8 := by sorry

end NUMINAMATH_CALUDE_distance_on_number_line_l2965_296508


namespace NUMINAMATH_CALUDE_lassis_from_mangoes_l2965_296595

/-- Given that 20 lassis can be made from 4 mangoes, prove that 80 lassis can be made from 16 mangoes. -/
theorem lassis_from_mangoes (make_lassis : ℕ → ℕ) 
  (h1 : make_lassis 4 = 20) 
  (h2 : ∀ x y : ℕ, make_lassis (x + y) = make_lassis x + make_lassis y) : 
  make_lassis 16 = 80 := by
  sorry

end NUMINAMATH_CALUDE_lassis_from_mangoes_l2965_296595


namespace NUMINAMATH_CALUDE_harry_bid_difference_l2965_296546

/-- Represents the auction process and calculates the difference between Harry's final bid and the third bidder's bid. -/
def auctionBidDifference (startingBid : ℕ) (harryFirstIncrement : ℕ) (harryFinalBid : ℕ) : ℕ :=
  let harryFirstBid := startingBid + harryFirstIncrement
  let secondBid := harryFirstBid * 2
  let thirdBid := secondBid + harryFirstIncrement * 3
  harryFinalBid - thirdBid

/-- Theorem stating that given the specific auction conditions, Harry's final bid exceeds the third bidder's bid by $2400. -/
theorem harry_bid_difference :
  auctionBidDifference 300 200 4000 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_harry_bid_difference_l2965_296546


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2965_296500

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2965_296500


namespace NUMINAMATH_CALUDE_fraction_simplification_l2965_296507

theorem fraction_simplification (c : ℝ) : (6 + 2 * c) / 7 + 3 = (27 + 2 * c) / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2965_296507


namespace NUMINAMATH_CALUDE_salary_spending_l2965_296509

theorem salary_spending (S : ℝ) (h1 : S > 0) : 
  let first_week := S / 4
  let unspent := S * 0.15
  let total_spent := S - unspent
  let last_three_weeks := total_spent - first_week
  last_three_weeks / (3 * S) = 0.2 := by sorry

end NUMINAMATH_CALUDE_salary_spending_l2965_296509


namespace NUMINAMATH_CALUDE_golf_score_difference_l2965_296545

/-- Given Richard's and Bruno's golf scores, prove the difference between their scores. -/
theorem golf_score_difference (richard_score bruno_score : ℕ) 
  (h1 : richard_score = 62) 
  (h2 : bruno_score = 48) : 
  richard_score - bruno_score = 14 := by
  sorry

end NUMINAMATH_CALUDE_golf_score_difference_l2965_296545


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2965_296598

theorem equal_roots_quadratic (p : ℝ) : 
  (∃! p, ∀ x : ℝ, x^2 - p*x + p^2 = 0 → (∃! x : ℝ, x^2 - p*x + p^2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2965_296598


namespace NUMINAMATH_CALUDE_division_problem_l2965_296521

theorem division_problem (x : ℕ+) (y : ℚ) (m : ℤ) 
  (h1 : (x : ℚ) = 11 * y + 4)
  (h2 : (2 * x : ℚ) = 8 * m * y + 3)
  (h3 : 13 * y - x = 1) :
  m = 3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2965_296521


namespace NUMINAMATH_CALUDE_abc_maximum_l2965_296577

theorem abc_maximum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : 2*a + 4*b + 8*c = 16) : a*b*c ≤ 64/27 := by
  sorry

end NUMINAMATH_CALUDE_abc_maximum_l2965_296577


namespace NUMINAMATH_CALUDE_periodic_sequence_prime_period_l2965_296559

/-- A sequence a is periodic with period m if a(m+n) = a(n) for all n -/
def isPeriodic (a : ℕ → ℂ) (m : ℕ) : Prop :=
  ∀ n, a (m + n) = a n

/-- m is the smallest positive period of sequence a -/
def isSmallestPeriod (a : ℕ → ℂ) (m : ℕ) : Prop :=
  isPeriodic a m ∧ ∀ k, 0 < k → k < m → ¬isPeriodic a k

/-- q is an m-th root of unity -/
def isRootOfUnity (q : ℂ) (m : ℕ) : Prop :=
  q ^ m = 1

theorem periodic_sequence_prime_period
  (q : ℂ) (m : ℕ) 
  (h1 : isSmallestPeriod (fun n => q^n) m)
  (h2 : m ≥ 2)
  (h3 : Nat.Prime m) :
  isRootOfUnity q m ∧ q ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_sequence_prime_period_l2965_296559


namespace NUMINAMATH_CALUDE_percent_relation_l2965_296524

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : c = 0.5 * b) :
  b = 0.5 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l2965_296524


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l2965_296562

theorem sin_cos_fourth_power_sum (α : ℝ) (h : Real.sin α - Real.cos α = 1/2) :
  Real.sin α ^ 4 + Real.cos α ^ 4 = 23/32 := by sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l2965_296562


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2965_296526

-- Define the equation
def equation (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 5

-- Define the roots of the equation
def roots (m : ℝ) : Set ℝ := {x | equation m x = 0}

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  base_positive : base > 0
  side_positive : side > 0
  isosceles : side ≥ base

-- State the theorem
theorem isosceles_triangle_perimeter : 
  ∃ (m : ℝ) (t : IsoscelesTriangle), 
    1 ∈ roots m ∧ 
    (∃ (x : ℝ), x ∈ roots m ∧ x ≠ 1) ∧
    {t.base, t.side} = roots m ∧
    t.base + 2 * t.side = 11 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2965_296526


namespace NUMINAMATH_CALUDE_octahedron_colorings_l2965_296504

/-- The symmetry group of a regular octahedron -/
structure OctahedronSymmetryGroup where
  order : ℕ
  h_order : order = 24

/-- The number of distinct vertex colorings of a regular octahedron -/
def vertex_colorings (G : OctahedronSymmetryGroup) (m : ℕ) : ℚ :=
  (m^6 + 3*m^4 + 12*m^3 + 8*m^2) / G.order

/-- The number of distinct face colorings of a regular octahedron -/
def face_colorings (G : OctahedronSymmetryGroup) (m : ℕ) : ℚ :=
  (m^8 + 17*m^6 + 6*m^2) / G.order

/-- Theorem stating the number of distinct colorings for vertices and faces -/
theorem octahedron_colorings (G : OctahedronSymmetryGroup) (m : ℕ) :
  (vertex_colorings G m = (m^6 + 3*m^4 + 12*m^3 + 8*m^2) / 24) ∧
  (face_colorings G m = (m^8 + 17*m^6 + 6*m^2) / 24) := by
  sorry

end NUMINAMATH_CALUDE_octahedron_colorings_l2965_296504
