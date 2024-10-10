import Mathlib

namespace triangle_side_length_l581_58192

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C,
    if a = √5, c = 2, and cos(A) = 2/3, then b = 3 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2/3 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  b = 3 :=
by sorry

end triangle_side_length_l581_58192


namespace semicircle_pattern_area_l581_58150

/-- Represents the pattern of alternating semicircles -/
structure SemicirclePattern where
  diameter : ℝ
  patternLength : ℝ

/-- Calculates the total shaded area of the semicircle pattern -/
def totalShadedArea (pattern : SemicirclePattern) : ℝ :=
  sorry

/-- Theorem stating that the total shaded area for the given pattern is 6.75π -/
theorem semicircle_pattern_area 
  (pattern : SemicirclePattern) 
  (h1 : pattern.diameter = 3)
  (h2 : pattern.patternLength = 10) : 
  totalShadedArea pattern = 6.75 * Real.pi := by
  sorry

end semicircle_pattern_area_l581_58150


namespace composite_shape_area_l581_58166

/-- The area of a rectangle -/
def rectangleArea (length width : ℕ) : ℕ := length * width

/-- The total area of the composite shape -/
def totalArea (a b c : ℕ × ℕ) : ℕ :=
  rectangleArea a.1 a.2 + rectangleArea b.1 b.2 + rectangleArea c.1 c.2

/-- Theorem stating that the total area of the given composite shape is 83 square units -/
theorem composite_shape_area :
  totalArea (8, 6) (5, 4) (3, 5) = 83 := by
  sorry

end composite_shape_area_l581_58166


namespace square_to_rectangle_ratio_l581_58174

/-- The number of rectangles formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_rectangles : ℕ := 2025

/-- The number of squares formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_squares : ℕ := 285

/-- The ratio of squares to rectangles on a 9x9 chessboard with 10 horizontal and 10 vertical lines -/
theorem square_to_rectangle_ratio : 
  (num_squares : ℚ) / num_rectangles = 19 / 135 := by sorry

end square_to_rectangle_ratio_l581_58174


namespace product_remainder_ten_l581_58170

theorem product_remainder_ten (a b c d : ℕ) (ha : a % 10 = 3) (hb : b % 10 = 7) (hc : c % 10 = 5) (hd : d % 10 = 3) :
  (a * b * c * d) % 10 = 5 := by
  sorry

end product_remainder_ten_l581_58170


namespace midpoint_distance_theorem_l581_58133

theorem midpoint_distance_theorem (t : ℝ) : 
  let P : ℝ × ℝ := (2 * t - 3, 2)
  let Q : ℝ × ℝ := (-2, 2 * t + 1)
  let M : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  (M.1 - P.1) ^ 2 + (M.2 - P.2) ^ 2 = t ^ 2 + 1 →
  t = 1 + Real.sqrt (3 / 2) ∨ t = 1 - Real.sqrt (3 / 2) :=
by sorry

end midpoint_distance_theorem_l581_58133


namespace sqrt_expression_equality_l581_58161

theorem sqrt_expression_equality : 
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) + (2 * Real.sqrt 2 - 1)^2 = 8 - 4 * Real.sqrt 2 := by
  sorry

end sqrt_expression_equality_l581_58161


namespace equation_solutions_l581_58193

def is_solution (x y : ℤ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 7

theorem equation_solutions :
  ∃! (s : Finset (ℤ × ℤ)), s.card = 5 ∧ ∀ (p : ℤ × ℤ), p ∈ s ↔ is_solution p.1 p.2 :=
sorry

end equation_solutions_l581_58193


namespace sunzi_wood_measurement_problem_l581_58178

theorem sunzi_wood_measurement_problem (x y : ℝ) :
  (x - y = 4.5 ∧ (1/2) * x + 1 = y) ↔
  (x - y = 4.5 ∧ ∃ (z : ℝ), z = x/2 ∧ z + 1 = y ∧ x - (z + 1) = 4.5) :=
by sorry

end sunzi_wood_measurement_problem_l581_58178


namespace student_sister_weight_l581_58144

/-- The combined weight of a student and his sister -/
theorem student_sister_weight (student_weight sister_weight : ℝ) : 
  student_weight = 71 →
  student_weight - 5 = 2 * sister_weight →
  student_weight + sister_weight = 104 := by
sorry

end student_sister_weight_l581_58144


namespace rectangle_area_l581_58117

/-- The area of the rectangle formed by the intersections of x^4 + y^4 = 100 and xy = 4 -/
theorem rectangle_area : ∃ (a b : ℝ), 
  (a^4 + b^4 = 100) ∧ 
  (a * b = 4) ∧ 
  (2 * (a^2 - b^2) = 4 * Real.sqrt 17) := by
  sorry

end rectangle_area_l581_58117


namespace smallest_sum_of_bases_l581_58198

theorem smallest_sum_of_bases : ∃ (a b : ℕ), 
  (a > 6 ∧ b > 6) ∧ 
  (6 * a + 2 = 2 * b + 6) ∧ 
  (∀ (a' b' : ℕ), (a' > 6 ∧ b' > 6) → (6 * a' + 2 = 2 * b' + 6) → a + b ≤ a' + b') ∧
  a + b = 26 := by
sorry

end smallest_sum_of_bases_l581_58198


namespace simplify_trig_expression_l581_58152

theorem simplify_trig_expression (x : ℝ) (h : 5 * π / 4 < x ∧ x < 3 * π / 2) :
  Real.sqrt (1 - 2 * Real.sin x * Real.cos x) = Real.cos x - Real.sin x := by
  sorry

end simplify_trig_expression_l581_58152


namespace trig_identities_l581_58163

theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α - Real.pi/4) = 1/3) ∧
  (Real.sin (2*α) / (Real.sin α^2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 80/37) := by
  sorry

end trig_identities_l581_58163


namespace melanie_brownies_given_out_l581_58106

def total_brownies : ℕ := 12 * 25

def bake_sale_brownies : ℕ := (7 * total_brownies) / 10

def remaining_after_bake_sale : ℕ := total_brownies - bake_sale_brownies

def container_brownies : ℕ := (2 * remaining_after_bake_sale) / 3

def remaining_after_container : ℕ := remaining_after_bake_sale - container_brownies

def charity_brownies : ℕ := (2 * remaining_after_container) / 5

def brownies_given_out : ℕ := remaining_after_container - charity_brownies

theorem melanie_brownies_given_out : brownies_given_out = 18 := by
  sorry

end melanie_brownies_given_out_l581_58106


namespace meaningful_expression_l581_58116

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 2)) ↔ x > 2 :=
by sorry

end meaningful_expression_l581_58116


namespace m_divided_by_8_l581_58127

theorem m_divided_by_8 (m : ℕ) (h : m = 16^1011) : m / 8 = 2^4041 := by
  sorry

end m_divided_by_8_l581_58127


namespace complex_number_system_l581_58108

theorem complex_number_system (a b c : ℂ) (h_real : a.im = 0) 
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 4)
  (h_prod : a * b * c = 4) : 
  a = 1 := by sorry

end complex_number_system_l581_58108


namespace chocolate_distribution_l581_58190

-- Define the total number of chocolate bars
def total_chocolate_bars : ℕ := 400

-- Define the number of small boxes
def num_small_boxes : ℕ := 16

-- Define the number of chocolate bars in each small box
def bars_per_small_box : ℕ := total_chocolate_bars / num_small_boxes

-- Theorem to prove
theorem chocolate_distribution :
  bars_per_small_box = 25 :=
sorry

end chocolate_distribution_l581_58190


namespace alternate_angle_measure_l581_58175

-- Define the angle measures as real numbers
def angle_A : ℝ := 0
def angle_B : ℝ := 0
def angle_C : ℝ := 0

-- State the theorem
theorem alternate_angle_measure :
  -- Conditions
  (angle_A = (1/4) * angle_B) →  -- ∠A is 1/4 of ∠B
  (angle_C = angle_A) →          -- ∠C and ∠A are alternate angles (due to parallel lines)
  (angle_B + angle_C = 180) →    -- ∠B and ∠C form a straight line
  -- Conclusion
  (angle_C = 36) := by
  sorry

end alternate_angle_measure_l581_58175


namespace unique_number_satisfying_equation_l581_58124

theorem unique_number_satisfying_equation : ∃! x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 := by
  sorry

end unique_number_satisfying_equation_l581_58124


namespace sufficient_not_necessary_l581_58134

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define when two lines are parallel
def parallel (a : ℝ) : Prop := ∀ x y z w : ℝ, l₁ a x y ∧ l₂ a z w → (a = 2 * (a + 1))

-- Statement to prove
theorem sufficient_not_necessary (a : ℝ) :
  (a = 1 → parallel a) ∧ ¬(parallel a → a = 1) :=
sorry

end sufficient_not_necessary_l581_58134


namespace room_height_proof_l581_58110

theorem room_height_proof (l b h : ℝ) : 
  l = 12 → b = 8 → (l^2 + b^2 + h^2 = 17^2) → h = 9 := by sorry

end room_height_proof_l581_58110


namespace jack_remaining_gift_card_value_jack_gift_card_return_l581_58164

/-- Calculates the remaining value of gift cards Jack can return after sending some to a scammer. -/
theorem jack_remaining_gift_card_value 
  (bb_count : ℕ) (bb_value : ℕ) (wm_count : ℕ) (wm_value : ℕ) 
  (bb_sent : ℕ) (wm_sent : ℕ) : ℕ :=
  let total_bb := bb_count * bb_value
  let total_wm := wm_count * wm_value
  let sent_bb := bb_sent * bb_value
  let sent_wm := wm_sent * wm_value
  let remaining_bb := total_bb - sent_bb
  let remaining_wm := total_wm - sent_wm
  remaining_bb + remaining_wm

/-- Proves that Jack can return gift cards worth $3900. -/
theorem jack_gift_card_return : 
  jack_remaining_gift_card_value 6 500 9 200 1 2 = 3900 := by
  sorry

end jack_remaining_gift_card_value_jack_gift_card_return_l581_58164


namespace product_of_exponents_l581_58102

theorem product_of_exponents (p r s : ℕ) : 
  4^p + 4^3 = 272 → 
  3^r + 54 = 135 → 
  7^2 + 6^s = 527 → 
  p * r * s = 64 :=
by
  sorry

end product_of_exponents_l581_58102


namespace non_intersecting_lines_parallel_or_skew_l581_58105

/-- Two lines in three-dimensional space -/
structure Line3D where
  -- We don't need to define the internal structure of a line
  -- for this problem, so we leave it abstract

/-- Predicate for two lines intersecting -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for two lines being parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for two lines being skew -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

theorem non_intersecting_lines_parallel_or_skew 
  (l1 l2 : Line3D) (h : ¬ intersect l1 l2) : 
  parallel l1 l2 ∨ skew l1 l2 :=
by
  sorry

end non_intersecting_lines_parallel_or_skew_l581_58105


namespace expression_simplification_l581_58140

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 - y^2) / (x*y) - (x^2*y - y^3) / (x^2*y - x*y^2) = (x^2 - x*y - 2*y^2) / (x*y) := by
  sorry

end expression_simplification_l581_58140


namespace sort_table_in_99_moves_l581_58151

/-- Represents a 10x10 table of distinct integers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Predicate to check if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  ∀ i j i' j', t i j = t i' j' → i = i' ∧ j = j'

/-- Predicate to check if the table is sorted in ascending order -/
def is_sorted (t : Table) : Prop :=
  (∀ i j j', j < j' → t i j < t i j') ∧
  (∀ i i' j, i < i' → t i j < t i' j)

/-- Represents a rectangular subset of the table -/
structure Rectangle where
  top_left : Fin 10 × Fin 10
  bottom_right : Fin 10 × Fin 10

/-- Represents a move (180° rotation of a rectangular subset) -/
def Move := Rectangle

/-- Applies a move to the table -/
def apply_move (t : Table) (m : Move) : Table :=
  sorry

/-- Theorem: It's always possible to sort the table in 99 or fewer moves -/
theorem sort_table_in_99_moves (t : Table) (h : all_distinct t) :
  ∃ (moves : List Move), moves.length ≤ 99 ∧ is_sorted (moves.foldl apply_move t) :=
  sorry

end sort_table_in_99_moves_l581_58151


namespace min_value_of_ab_l581_58182

theorem min_value_of_ab (a b : ℝ) (h : (1 / a) + (1 / b) = Real.sqrt (a * b)) : 
  2 ≤ a * b ∧ ∃ (x y : ℝ), (1 / x) + (1 / y) = Real.sqrt (x * y) ∧ x * y = 2 :=
sorry

end min_value_of_ab_l581_58182


namespace sum_of_digits_7_pow_1050_l581_58138

theorem sum_of_digits_7_pow_1050 : ∃ (a b : ℕ), 
  7^1050 % 100 = 10 * a + b ∧ a + b = 13 := by sorry

end sum_of_digits_7_pow_1050_l581_58138


namespace f_of_5_eq_2515_l581_58176

/-- The polynomial function f(x) -/
def f (x : ℝ) : ℝ := 3*x^5 - 15*x^4 + 27*x^3 - 20*x^2 - 72*x + 40

/-- Theorem: f(5) equals 2515 -/
theorem f_of_5_eq_2515 : f 5 = 2515 := by sorry

end f_of_5_eq_2515_l581_58176


namespace guthrie_market_souvenirs_cost_l581_58136

/-- The total cost of souvenirs distributed at Guthrie Market's Grand Opening -/
theorem guthrie_market_souvenirs_cost :
  let type1_cost : ℚ := 20 / 100  -- 20 cents in dollars
  let type2_cost : ℚ := 25 / 100  -- 25 cents in dollars
  let total_souvenirs : ℕ := 1000
  let type2_quantity : ℕ := 400
  let type1_quantity : ℕ := total_souvenirs - type2_quantity
  let total_cost : ℚ := type1_quantity * type1_cost + type2_quantity * type2_cost
  total_cost = 220 / 100  -- $220 in decimal form
:= by sorry

end guthrie_market_souvenirs_cost_l581_58136


namespace simplify_expression_l581_58120

theorem simplify_expression (x : ℝ) : (x + 15) + (100 * x + 15) = 101 * x + 30 := by
  sorry

end simplify_expression_l581_58120


namespace overlap_range_l581_58125

theorem overlap_range (total : ℕ) (math : ℕ) (chem : ℕ) (x : ℕ) 
  (h_total : total = 45)
  (h_math : math = 28)
  (h_chem : chem = 21)
  (h_overlap : x ≤ math ∧ x ≤ chem)
  (h_inclusion : math + chem - x ≤ total) :
  4 ≤ x ∧ x ≤ 21 := by
sorry

end overlap_range_l581_58125


namespace probability_of_not_losing_l581_58119

theorem probability_of_not_losing (prob_draw prob_win : ℚ) 
  (h1 : prob_draw = 1/2) 
  (h2 : prob_win = 1/3) : 
  prob_draw + prob_win = 5/6 := by
  sorry

end probability_of_not_losing_l581_58119


namespace inequality_proof_l581_58173

theorem inequality_proof (m n : ℝ) (h1 : m < n) (h2 : n < 0) : n / m + m / n > 2 := by
  sorry

end inequality_proof_l581_58173


namespace A_initial_investment_l581_58129

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := 27000

/-- Represents the investment of B in rupees -/
def B_investment : ℝ := 36000

/-- Represents the number of months in a year -/
def months_in_year : ℝ := 12

/-- Represents the number of months after which B joined -/
def B_join_time : ℝ := 7.5

/-- Represents the ratio of profit sharing between A and B -/
def profit_ratio : ℝ := 2

theorem A_initial_investment :
  A_investment * months_in_year = 
  profit_ratio * B_investment * (months_in_year - B_join_time) :=
by sorry

end A_initial_investment_l581_58129


namespace train_speed_l581_58167

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 240)
  (h2 : bridge_length = 750)
  (h3 : crossing_time = 80) :
  (train_length + bridge_length) / crossing_time = 12.375 := by
  sorry

end train_speed_l581_58167


namespace product_of_roots_cubic_equation_l581_58199

theorem product_of_roots_cubic_equation :
  let f : ℝ → ℝ := λ x => 3 * x^3 - x^2 - 20 * x + 27
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ * r₂ * r₃ = -9 := by
  sorry

end product_of_roots_cubic_equation_l581_58199


namespace smallest_n_for_cube_assembly_l581_58114

theorem smallest_n_for_cube_assembly (n : ℕ) : 
  (∀ m : ℕ, m < n → m^3 < (2*m)^3 - (2*m - 2)^3) ∧ 
  n^3 ≥ (2*n)^3 - (2*n - 2)^3 → 
  n = 23 := by
sorry

end smallest_n_for_cube_assembly_l581_58114


namespace sufficient_but_not_necessary_l581_58196

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 1 → 1/a < 1) ∧ ¬(1/a < 1 → a > 1) := by
  sorry

end sufficient_but_not_necessary_l581_58196


namespace no_solutions_to_sqrt_equation_l581_58183

theorem no_solutions_to_sqrt_equation :
  ∀ x : ℝ, x ≥ 4 →
  ¬∃ y : ℝ, y = Real.sqrt (x + 5 - 6 * Real.sqrt (x - 4)) + Real.sqrt (x + 18 - 8 * Real.sqrt (x - 4)) ∧ y = 2 :=
by sorry

end no_solutions_to_sqrt_equation_l581_58183


namespace binomial_product_theorem_l581_58100

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the smallest prime number greater than 10
def smallest_prime_gt_10 : ℕ := 11

-- Theorem statement
theorem binomial_product_theorem :
  binomial 18 6 * smallest_prime_gt_10 = 80080 := by
  sorry

end binomial_product_theorem_l581_58100


namespace tank_emptying_time_l581_58189

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initialFullness : ℚ
  pipeARatePerMinute : ℚ
  pipeBRatePerMinute : ℚ
  pipeCRatePerMinute : ℚ

/-- Calculates the time to empty or fill the tank given its properties -/
def timeToEmptyOrFill (tank : WaterTank) : ℚ :=
  tank.initialFullness / (tank.pipeARatePerMinute + tank.pipeBRatePerMinute + tank.pipeCRatePerMinute)

/-- Theorem stating the time to empty the specific tank configuration -/
theorem tank_emptying_time :
  let tank : WaterTank := {
    initialFullness := 7/11,
    pipeARatePerMinute := 1/15,
    pipeBRatePerMinute := -1/8,
    pipeCRatePerMinute := 1/20
  }
  timeToEmptyOrFill tank = 840/11 := by
  sorry

end tank_emptying_time_l581_58189


namespace power_sum_inequality_l581_58160

theorem power_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^4 + b^4 + c^4) / (a + b + c) ≥ a * b * c :=
by sorry

end power_sum_inequality_l581_58160


namespace ab_value_l581_58169

theorem ab_value (a b : ℝ) (h : (a - 2)^2 + Real.sqrt (b + 3) = 0) : a * b = -6 := by
  sorry

end ab_value_l581_58169


namespace tan_two_theta_value_l581_58121

theorem tan_two_theta_value (θ : Real) 
  (h : 2 * Real.sin (π / 2 + θ) + Real.sin (π + θ) = 0) : 
  Real.tan (2 * θ) = -4 / 3 := by
  sorry

end tan_two_theta_value_l581_58121


namespace sum_of_squares_l581_58157

theorem sum_of_squares (x y : ℝ) (h1 : x * (x + y) = 35) (h2 : y * (x + y) = 77) :
  (x + y)^2 = 112 := by
  sorry

end sum_of_squares_l581_58157


namespace smallest_abundant_not_multiple_of_five_l581_58147

def is_abundant (n : ℕ) : Prop :=
  n > 0 ∧ (Finset.sum (Finset.filter (λ x => x < n ∧ n % x = 0) (Finset.range n)) id > n)

def is_multiple_of_five (n : ℕ) : Prop :=
  n % 5 = 0

theorem smallest_abundant_not_multiple_of_five : 
  (∀ k : ℕ, k < 12 → ¬(is_abundant k ∧ ¬is_multiple_of_five k)) ∧ 
  (is_abundant 12 ∧ ¬is_multiple_of_five 12) := by
  sorry

end smallest_abundant_not_multiple_of_five_l581_58147


namespace table_color_change_l581_58115

/-- Represents the color of a cell in the table -/
inductive CellColor
| White
| Black
| Orange

/-- Represents a 3n × 3n table with the given coloring pattern -/
def Table (n : ℕ) := Fin (3*n) → Fin (3*n) → CellColor

/-- Predicate to check if a given 2×2 square can be chosen for color change -/
def CanChangeSquare (t : Table n) (i j : Fin (3*n-1)) : Prop := True

/-- Predicate to check if the table has all white cells turned to black and all black cells turned to white -/
def IsTargetState (t : Table n) : Prop := True

/-- Predicate to check if it's possible to reach the target state in a finite number of steps -/
def CanReachTargetState (n : ℕ) : Prop := 
  ∃ (t : Table n), IsTargetState t

theorem table_color_change (n : ℕ) : 
  CanReachTargetState n ↔ Even n :=
sorry

end table_color_change_l581_58115


namespace base9_726_to_base3_l581_58156

/-- Converts a base-9 digit to its two-digit base-3 representation -/
def base9ToBase3Digit (d : Nat) : Nat × Nat :=
  ((d / 3), (d % 3))

/-- Converts a base-9 number to its base-3 representation -/
def base9ToBase3 (n : Nat) : List Nat :=
  let digits := n.digits 9
  List.join (digits.map (fun d => let (a, b) := base9ToBase3Digit d; [a, b]))

theorem base9_726_to_base3 :
  base9ToBase3 726 = [2, 1, 0, 2, 2, 0] :=
sorry

end base9_726_to_base3_l581_58156


namespace purely_imaginary_complex_number_l581_58165

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 :=
by
  sorry

end purely_imaginary_complex_number_l581_58165


namespace sum_angles_regular_star_5_l581_58155

/-- A regular 5-pointed star inscribed in a circle -/
structure RegularStar5 where
  /-- The angle at each tip of the star -/
  tip_angle : ℝ
  /-- The number of points in the star -/
  num_points : ℕ
  /-- The number of points is 5 -/
  h_num_points : num_points = 5

/-- The sum of angles at the tips of a regular 5-pointed star is 540° -/
theorem sum_angles_regular_star_5 (star : RegularStar5) : 
  star.num_points * star.tip_angle = 540 := by
  sorry

end sum_angles_regular_star_5_l581_58155


namespace encyclopedia_pages_l581_58128

/-- The number of chapters in the encyclopedia -/
def num_chapters : ℕ := 7

/-- The number of pages in each chapter of the encyclopedia -/
def pages_per_chapter : ℕ := 566

/-- The total number of pages in the encyclopedia -/
def total_pages : ℕ := num_chapters * pages_per_chapter

/-- Theorem stating that the total number of pages in the encyclopedia is 3962 -/
theorem encyclopedia_pages : total_pages = 3962 := by
  sorry

end encyclopedia_pages_l581_58128


namespace speaker_is_tweedledee_l581_58130

-- Define the brothers
inductive Brother
| Tweedledum
| Tweedledee

-- Define the card suits
inductive Suit
| Black
| Red

-- Define the speaker
structure Speaker where
  identity : Brother
  card : Suit

-- Define the statement made by the speaker
def statement (s : Speaker) : Prop :=
  s.identity = Brother.Tweedledum → s.card ≠ Suit.Black

-- Theorem: The speaker must be Tweedledee
theorem speaker_is_tweedledee (s : Speaker) (h : statement s) : 
  s.identity = Brother.Tweedledee :=
sorry

end speaker_is_tweedledee_l581_58130


namespace range_of_a_l581_58137

def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x ≤ a^2 - a - 3

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (5 - 2*a)^x > (5 - 2*a)^y

theorem range_of_a : 
  (∀ a : ℝ, (p a ∨ q a)) ∧ (¬∃ a : ℝ, p a ∧ q a) → 
  {a : ℝ | a = 2 ∨ a ≥ 5/2} = {a : ℝ | ∃ x : ℝ, p x ∨ q x} :=
by sorry

end range_of_a_l581_58137


namespace bus_ride_difference_l581_58185

theorem bus_ride_difference (vince_ride : ℝ) (zachary_ride : ℝ)
  (h1 : vince_ride = 0.625)
  (h2 : zachary_ride = 0.5) :
  vince_ride - zachary_ride = 0.125 := by
sorry

end bus_ride_difference_l581_58185


namespace largest_valid_number_l581_58111

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0 → n % d = 0) ∧
  (n.digits 10).sum % 6 = 0

theorem largest_valid_number : 
  is_valid_number 936 ∧ ∀ m, is_valid_number m → m ≤ 936 :=
sorry

end largest_valid_number_l581_58111


namespace power_equation_l581_58159

theorem power_equation (a m n : ℝ) (h1 : a^m = 6) (h2 : a^n = 6) : a^(2*m - n) = 6 := by
  sorry

end power_equation_l581_58159


namespace probability_four_threes_eight_dice_l581_58186

def probability_four_threes (n m k : ℕ) : ℚ :=
  (n.choose k : ℚ) * (1 / m) ^ k * ((m - 1) / m) ^ (n - k)

theorem probability_four_threes_eight_dice :
  probability_four_threes 8 6 4 = 43750 / 1679616 := by
  sorry

end probability_four_threes_eight_dice_l581_58186


namespace train_crossing_time_l581_58109

/-- The time taken for a train to cross a man moving in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 210 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 28 :=
by sorry

end train_crossing_time_l581_58109


namespace max_value_abcd_l581_58168

theorem max_value_abcd (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (sum_eq_3 : a + b + c + d = 3) :
  3 * a^2 * b^3 * c * d^2 ≤ 177147 / 40353607 :=
sorry

end max_value_abcd_l581_58168


namespace speed_conversion_l581_58132

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- Given speed in meters per second -/
def speed_mps : ℝ := 12.7788

/-- Theorem stating the conversion of the given speed from m/s to km/h -/
theorem speed_conversion :
  speed_mps * mps_to_kmph = 45.96368 := by sorry

end speed_conversion_l581_58132


namespace max_pie_pieces_l581_58187

theorem max_pie_pieces : 
  (∃ (n : ℕ), n > 0 ∧ 
    ∃ (A B : ℕ), 
      10000 ≤ A ∧ A < 100000 ∧ 
      10000 ≤ B ∧ B < 100000 ∧ 
      A = B * n ∧ 
      (∀ (i j : Fin 5), i ≠ j → (A / 10^i.val % 10) ≠ (A / 10^j.val % 10)) ∧
    ∀ (m : ℕ), m > n → 
      ¬(∃ (C D : ℕ), 
        10000 ≤ C ∧ C < 100000 ∧ 
        10000 ≤ D ∧ D < 100000 ∧ 
        C = D * m ∧ 
        (∀ (i j : Fin 5), i ≠ j → (C / 10^i.val % 10) ≠ (C / 10^j.val % 10)))) ∧
  (∃ (A B : ℕ), 
    10000 ≤ A ∧ A < 100000 ∧ 
    10000 ≤ B ∧ B < 100000 ∧ 
    A = B * 7 ∧ 
    (∀ (i j : Fin 5), i ≠ j → (A / 10^i.val % 10) ≠ (A / 10^j.val % 10))) :=
by
  sorry

end max_pie_pieces_l581_58187


namespace polynomial_roots_and_product_l581_58180

/-- Given a polynomial p(x) = x³ + (3/2)(1-a)x² - 3ax + b where a and b are real numbers,
    and |p(x)| ≤ 1 for all x in [0, √3], prove that p(x) = 0 has three real roots
    and calculate a specific product of these roots. -/
theorem polynomial_roots_and_product (a b : ℝ) 
    (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 3 → 
      |x^3 + (3/2)*(1-a)*x^2 - 3*a*x + b| ≤ 1) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x : ℝ, x^3 + (3/2)*(1-a)*x^2 - 3*a*x + b = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    (x₁^2 - 2 - x₂) * (x₂^2 - 2 - x₃) * (x₃^2 - 2 - x₁) = -9 :=
by sorry

end polynomial_roots_and_product_l581_58180


namespace pizza_problem_l581_58118

/-- Calculates the total number of pizza pieces carried by children -/
def total_pizza_pieces (num_children : ℕ) (pizzas_per_child : ℕ) (pieces_per_pizza : ℕ) : ℕ :=
  num_children * pizzas_per_child * pieces_per_pizza

/-- Proves that 10 children buying 20 pizzas each, with 6 pieces per pizza, carry 1200 pieces total -/
theorem pizza_problem : total_pizza_pieces 10 20 6 = 1200 := by
  sorry

end pizza_problem_l581_58118


namespace train_passes_jogger_l581_58104

/-- Proves that a train passes a jogger in 40 seconds given specific conditions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 280 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 40 := by
  sorry

#check train_passes_jogger

end train_passes_jogger_l581_58104


namespace inverse_proportion_through_point_l581_58171

/-- The inverse proportion function passing through (2, -1) has k = -2 --/
theorem inverse_proportion_through_point (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, x ≠ 0 → k / x = -1 / 2) → k = -2 := by
  sorry

end inverse_proportion_through_point_l581_58171


namespace fence_sheets_count_l581_58145

/-- Represents the number of fence panels in the fence. -/
def num_panels : ℕ := 10

/-- Represents the number of metal beams in each fence panel. -/
def beams_per_panel : ℕ := 2

/-- Represents the number of metal rods in each sheet. -/
def rods_per_sheet : ℕ := 10

/-- Represents the number of metal rods in each beam. -/
def rods_per_beam : ℕ := 4

/-- Represents the total number of metal rods needed for the fence. -/
def total_rods : ℕ := 380

/-- Calculates the number of metal sheets in each fence panel. -/
def sheets_per_panel : ℕ :=
  let total_rods_per_panel := total_rods / num_panels
  let rods_for_beams := beams_per_panel * rods_per_beam
  (total_rods_per_panel - rods_for_beams) / rods_per_sheet

theorem fence_sheets_count : sheets_per_panel = 3 := by
  sorry

end fence_sheets_count_l581_58145


namespace problem_statement_l581_58139

theorem problem_statement (A B C : ℚ) 
  (h1 : 1 / A = -3)
  (h2 : 2 / B = 4)
  (h3 : 3 / C = 1 / 2) :
  6 * A - 8 * B + C = 0 := by
  sorry

end problem_statement_l581_58139


namespace solution_check_l581_58148

def is_solution (x : ℝ) : Prop :=
  4 * x + 5 = 8 * x - 3

theorem solution_check :
  is_solution 2 ∧ ¬is_solution 3 := by
  sorry

end solution_check_l581_58148


namespace quadratic_root_implies_u_l581_58162

theorem quadratic_root_implies_u (u : ℝ) : 
  (6 * ((-25 - Real.sqrt 469) / 12)^2 + 25 * ((-25 - Real.sqrt 469) / 12) + u = 0) → 
  u = 13/2 := by
  sorry

end quadratic_root_implies_u_l581_58162


namespace sqrt_equation_equals_difference_l581_58149

theorem sqrt_equation_equals_difference (a b : ℤ) : 
  Real.sqrt (16 - 12 * Real.cos (40 * π / 180)) = a + b * (1 / Real.cos (40 * π / 180)) →
  a = 4 ∧ b = -1 := by
  sorry

end sqrt_equation_equals_difference_l581_58149


namespace ellipse_eccentricity_l581_58122

/-- The eccentricity of an ellipse with a focus shared with the parabola y^2 = x -/
theorem ellipse_eccentricity (a : ℝ) : 
  let parabola := {(x, y) : ℝ × ℝ | y^2 = x}
  let ellipse := {(x, y) : ℝ × ℝ | (x^2 / a^2) + (y^2 / 3) = 1}
  let parabola_focus : ℝ × ℝ := (1/4, 0)
  (parabola_focus ∈ ellipse) →
  (∃ c b : ℝ, c^2 + b^2 = a^2 ∧ c = 1/4 ∧ b^2 = 3) →
  (c / a = 1/7) :=
by sorry

end ellipse_eccentricity_l581_58122


namespace min_tries_for_blue_and_yellow_is_thirteen_l581_58153

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  purple : Nat
  blue : Nat
  yellow : Nat

/-- The minimum number of tries required to guarantee obtaining one blue and one yellow ball -/
def minTriesForBlueAndYellow (counts : BallCounts) : Nat :=
  counts.purple + counts.blue + 1

theorem min_tries_for_blue_and_yellow_is_thirteen :
  let counts : BallCounts := { purple := 7, blue := 5, yellow := 11 }
  minTriesForBlueAndYellow counts = 13 := by sorry

end min_tries_for_blue_and_yellow_is_thirteen_l581_58153


namespace compound_molecular_weight_l581_58146

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.01

/-- The atomic weight of Bromine in g/mol -/
def bromine_weight : ℝ := 79.90

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of Bromine atoms in the compound -/
def bromine_count : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ :=
  hydrogen_count * hydrogen_weight +
  bromine_count * bromine_weight +
  oxygen_count * oxygen_weight

/-- Theorem stating that the molecular weight of the compound is 128.91 g/mol -/
theorem compound_molecular_weight : molecular_weight = 128.91 := by
  sorry

end compound_molecular_weight_l581_58146


namespace sum_to_k_perfect_cube_l581_58135

def sum_to_k (k : ℕ) : ℕ := k * (k + 1) / 2

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem sum_to_k_perfect_cube :
  ∀ k : ℕ, k > 0 → k < 200 →
    (is_perfect_cube (sum_to_k k) ↔ k = 1 ∨ k = 4) := by
  sorry

end sum_to_k_perfect_cube_l581_58135


namespace gcd_lcm_product_90_150_l581_58184

theorem gcd_lcm_product_90_150 : Nat.gcd 90 150 * Nat.lcm 90 150 = 13500 := by
  sorry

end gcd_lcm_product_90_150_l581_58184


namespace intersection_condition_l581_58103

-- Define the line and parabola
def line (k x : ℝ) : ℝ := k * x - 2 * k + 2
def parabola (a x : ℝ) : ℝ := a * x^2 - 2 * a * x - 3 * a

-- Define the condition for intersection
def hasCommonPoint (a : ℝ) : Prop :=
  ∀ k, ∃ x, line k x = parabola a x

-- State the theorem
theorem intersection_condition :
  ∀ a : ℝ, hasCommonPoint a ↔ (a ≤ -2/3 ∨ a > 0) :=
sorry

end intersection_condition_l581_58103


namespace ratio_expression_value_l581_58113

theorem ratio_expression_value (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by
  sorry

end ratio_expression_value_l581_58113


namespace simplify_radical_sum_l581_58107

theorem simplify_radical_sum : Real.sqrt 50 + Real.sqrt 18 = 8 * Real.sqrt 2 := by
  sorry

end simplify_radical_sum_l581_58107


namespace fraction_simplification_l581_58172

theorem fraction_simplification :
  let x : ℚ := 1/3
  1 / (1 / x^1 + 1 / x^2 + 1 / x^3) = 1 / 39 := by sorry

end fraction_simplification_l581_58172


namespace function_inequality_l581_58141

theorem function_inequality (f g : ℝ → ℝ) (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂) →
  (∀ x, f x = x + 4/x) →
  (∀ x, g x = 2^x + a) →
  a ≤ 1 := by
sorry

end function_inequality_l581_58141


namespace point_relationship_on_line_l581_58142

/-- Proves that for two points on a line with positive slope and non-negative y-intercept,
    if the x-coordinate of the first point is greater than the x-coordinate of the second point,
    then the y-coordinate of the first point is greater than the y-coordinate of the second point. -/
theorem point_relationship_on_line (k b x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : y₁ = k * x₁ + b)
  (h2 : y₂ = k * x₂ + b)
  (h3 : k > 0)
  (h4 : b ≥ 0)
  (h5 : x₁ > x₂) :
  y₁ > y₂ := by
  sorry

end point_relationship_on_line_l581_58142


namespace gcd_123456_789012_l581_58154

theorem gcd_123456_789012 : Nat.gcd 123456 789012 = 36 := by
  sorry

end gcd_123456_789012_l581_58154


namespace simplify_expression_l581_58179

theorem simplify_expression (a b : ℝ) :
  (15 * a + 45 * b) + (12 * a + 35 * b) - (7 * a + 30 * b) - (3 * a + 15 * b) = 17 * a + 35 * b :=
by sorry

end simplify_expression_l581_58179


namespace average_height_problem_l581_58123

/-- Given a class of girls with specific average heights, prove the average height of a subgroup -/
theorem average_height_problem (total_girls : ℕ) (subgroup_girls : ℕ) (remaining_girls : ℕ)
  (subgroup_avg_height : ℝ) (remaining_avg_height : ℝ) (total_avg_height : ℝ)
  (h1 : total_girls = subgroup_girls + remaining_girls)
  (h2 : total_girls = 40)
  (h3 : subgroup_girls = 30)
  (h4 : remaining_avg_height = 156)
  (h5 : total_avg_height = 159) :
  subgroup_avg_height = 160 := by
sorry


end average_height_problem_l581_58123


namespace complex_number_modulus_l581_58195

theorem complex_number_modulus (i : ℂ) (h : i * i = -1) :
  let z : ℂ := i * (1 - i)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_number_modulus_l581_58195


namespace z_percent_of_x_l581_58191

theorem z_percent_of_x (x y z : ℝ) 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.2 * x := by
sorry

end z_percent_of_x_l581_58191


namespace other_interest_rate_is_sixteen_percent_l581_58188

/-- Proves that given the investment conditions, the other interest rate is 16% -/
theorem other_interest_rate_is_sixteen_percent
  (investment_difference : ℝ)
  (higher_rate_investment : ℝ)
  (higher_rate : ℝ)
  (h1 : investment_difference = 1260)
  (h2 : higher_rate_investment = 2520)
  (h3 : higher_rate = 0.08)
  (h4 : higher_rate_investment = (higher_rate_investment - investment_difference) + investment_difference)
  (h5 : higher_rate_investment * higher_rate = (higher_rate_investment - investment_difference) * (16 / 100)) :
  ∃ (other_rate : ℝ), other_rate = 16 / 100 :=
by
  sorry

#check other_interest_rate_is_sixteen_percent

end other_interest_rate_is_sixteen_percent_l581_58188


namespace oranges_returned_l581_58126

def oranges_problem (initial_oranges : ℕ) (eaten_oranges : ℕ) (final_oranges : ℕ) : ℕ :=
  let remaining_after_eating := initial_oranges - eaten_oranges
  let stolen_oranges := remaining_after_eating / 2
  let remaining_after_theft := remaining_after_eating - stolen_oranges
  final_oranges - remaining_after_theft

theorem oranges_returned (initial_oranges eaten_oranges final_oranges : ℕ) 
  (h1 : initial_oranges = 60)
  (h2 : eaten_oranges = 10)
  (h3 : final_oranges = 30) : 
  oranges_problem initial_oranges eaten_oranges final_oranges = 5 := by
  sorry

#eval oranges_problem 60 10 30

end oranges_returned_l581_58126


namespace largest_integer_with_gcd_18_6_l581_58197

theorem largest_integer_with_gcd_18_6 :
  ∀ n : ℕ, n < 150 → n > 138 → Nat.gcd n 18 ≠ 6 :=
by sorry

end largest_integer_with_gcd_18_6_l581_58197


namespace georgia_has_24_students_l581_58181

/-- Represents the number of students Georgia has, given her muffin-making habits. -/
def georgia_students : ℕ :=
  let batches : ℕ := 36
  let muffins_per_batch : ℕ := 6
  let months : ℕ := 9
  let total_muffins : ℕ := batches * muffins_per_batch
  total_muffins / months

/-- Proves that Georgia has 24 students based on her muffin-making habits. -/
theorem georgia_has_24_students : georgia_students = 24 := by
  sorry

end georgia_has_24_students_l581_58181


namespace carolyns_silverware_percentage_l581_58101

/-- Represents the count of each type of silverware --/
structure SilverwareCount where
  knives : Int
  forks : Int
  spoons : Int
  teaspoons : Int

/-- Calculates the total count of silverware --/
def total_count (s : SilverwareCount) : Int :=
  s.knives + s.forks + s.spoons + s.teaspoons

/-- Represents a trade of silverware --/
structure Trade where
  give_knives : Int
  give_forks : Int
  give_spoons : Int
  give_teaspoons : Int
  receive_knives : Int
  receive_forks : Int
  receive_spoons : Int
  receive_teaspoons : Int

/-- Applies a trade to a silverware count --/
def apply_trade (s : SilverwareCount) (t : Trade) : SilverwareCount :=
  { knives := s.knives - t.give_knives + t.receive_knives,
    forks := s.forks - t.give_forks + t.receive_forks,
    spoons := s.spoons - t.give_spoons + t.receive_spoons,
    teaspoons := s.teaspoons - t.give_teaspoons + t.receive_teaspoons }

/-- Theorem representing Carolyn's silverware problem --/
theorem carolyns_silverware_percentage :
  let initial_count : SilverwareCount := { knives := 6, forks := 12, spoons := 18, teaspoons := 24 }
  let trade1 : Trade := { give_knives := 10, give_forks := 0, give_spoons := 0, give_teaspoons := 0,
                          receive_knives := 0, receive_forks := 0, receive_spoons := 0, receive_teaspoons := 6 }
  let trade2 : Trade := { give_knives := 0, give_forks := 8, give_spoons := 0, give_teaspoons := 0,
                          receive_knives := 0, receive_forks := 0, receive_spoons := 3, receive_teaspoons := 0 }
  let after_trades := apply_trade (apply_trade initial_count trade1) trade2
  let final_count := { after_trades with knives := after_trades.knives + 7 }
  (final_count.knives : Real) / (total_count final_count : Real) * 100 = 3 / 58 * 100 :=
by sorry

end carolyns_silverware_percentage_l581_58101


namespace chocolate_bar_cost_l581_58158

/-- Proves that the cost of each chocolate bar is $3 -/
theorem chocolate_bar_cost (initial_bars : ℕ) (unsold_bars : ℕ) (total_revenue : ℚ) : 
  initial_bars = 7 → unsold_bars = 4 → total_revenue = 9 → 
  (total_revenue / (initial_bars - unsold_bars : ℚ)) = 3 := by
  sorry

end chocolate_bar_cost_l581_58158


namespace find_y_l581_58112

theorem find_y : ∃ y : ℝ, (Real.sqrt (1 + Real.sqrt (4 * y - 5)) = Real.sqrt 8) ∧ y = 13.5 := by
  sorry

end find_y_l581_58112


namespace negation_equivalence_l581_58143

theorem negation_equivalence :
  (¬ ∃ (x y : ℝ), x^2 + y^2 - 1 ≤ 0) ↔ (∀ (x y : ℝ), x^2 + y^2 - 1 > 0) :=
by sorry

end negation_equivalence_l581_58143


namespace water_experiment_result_l581_58131

/-- Calculates the remaining water after an experiment and addition. -/
def remaining_water (initial : ℚ) (used : ℚ) (added : ℚ) : ℚ :=
  initial - used + added

/-- Proves that given the specific amounts in the problem, the remaining water is 13/6 gallons. -/
theorem water_experiment_result :
  remaining_water 3 (4/3) (1/2) = 13/6 := by
  sorry

end water_experiment_result_l581_58131


namespace range_of_m_l581_58177

theorem range_of_m (a b m : ℝ) 
  (ha : a > 0) 
  (hb : b > 1) 
  (hab : a + b = 2) 
  (h_ineq : ∀ m, (4/a) + (1/(b-1)) > m^2 + 8*m) :
  -9 < m ∧ m < 1 :=
sorry

end range_of_m_l581_58177


namespace remainder_of_five_n_mod_eleven_l581_58194

theorem remainder_of_five_n_mod_eleven (n : ℤ) (h : n % 11 = 1) : (5 * n) % 11 = 5 := by
  sorry

end remainder_of_five_n_mod_eleven_l581_58194
