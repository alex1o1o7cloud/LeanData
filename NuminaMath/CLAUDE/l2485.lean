import Mathlib

namespace NUMINAMATH_CALUDE_coin_problem_l2485_248502

/-- Represents the number and denomination of coins -/
structure CoinCount where
  twenties : Nat
  fifteens : Nat

/-- Calculates the total value of coins in kopecks -/
def totalValue (coins : CoinCount) : Nat :=
  20 * coins.twenties + 15 * coins.fifteens

/-- Represents the conditions of the problem -/
structure ProblemConditions where
  coins : CoinCount
  moreTwenties : coins.twenties > coins.fifteens
  fifthSpentWithTwoCoins : ∃ (a b : Nat), a + b = 2 ∧ 
    (a * 20 + b * 15 = totalValue coins / 5)
  halfRemainingSpentWithThreeCoins : ∃ (c d : Nat), c + d = 3 ∧ 
    (c * 20 + d * 15 = (4 * totalValue coins / 5) / 2)

/-- The theorem to be proved -/
theorem coin_problem (conditions : ProblemConditions) : 
  conditions.coins = CoinCount.mk 6 2 := by
  sorry


end NUMINAMATH_CALUDE_coin_problem_l2485_248502


namespace NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_sin_24_l2485_248507

theorem cos_96_cos_24_minus_sin_96_sin_24 : 
  Real.cos (96 * π / 180) * Real.cos (24 * π / 180) - 
  Real.sin (96 * π / 180) * Real.sin (24 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_96_cos_24_minus_sin_96_sin_24_l2485_248507


namespace NUMINAMATH_CALUDE_trace_bag_weight_l2485_248524

/-- Given:
  - Trace has 5 shopping bags
  - Trace's 5 bags weigh the same as Gordon's 2 bags
  - One of Gordon's bags weighs 3 pounds
  - The other of Gordon's bags weighs 7 pounds
  - All of Trace's bags weigh the same amount
Prove that one of Trace's bags weighs 2 pounds -/
theorem trace_bag_weight :
  ∀ (trace_bag_count : ℕ) 
    (gordon_bag1_weight gordon_bag2_weight : ℕ)
    (trace_total_weight : ℕ),
  trace_bag_count = 5 →
  gordon_bag1_weight = 3 →
  gordon_bag2_weight = 7 →
  trace_total_weight = gordon_bag1_weight + gordon_bag2_weight →
  ∃ (trace_single_bag_weight : ℕ),
    trace_single_bag_weight * trace_bag_count = trace_total_weight ∧
    trace_single_bag_weight = 2 :=
by sorry

end NUMINAMATH_CALUDE_trace_bag_weight_l2485_248524


namespace NUMINAMATH_CALUDE_min_value_sum_l2485_248561

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 1 / Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l2485_248561


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l2485_248523

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l2485_248523


namespace NUMINAMATH_CALUDE_original_price_calculation_l2485_248553

theorem original_price_calculation (discount_percentage : ℝ) (selling_price : ℝ) 
  (h1 : discount_percentage = 20)
  (h2 : selling_price = 14) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_percentage / 100) = selling_price ∧ 
    original_price = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2485_248553


namespace NUMINAMATH_CALUDE_stratified_sample_correct_l2485_248559

/-- Represents the count of households in each income category -/
structure Population :=
  (high : ℕ)
  (middle : ℕ)
  (low : ℕ)

/-- Represents the sample sizes for each income category -/
structure Sample :=
  (high : ℕ)
  (middle : ℕ)
  (low : ℕ)

/-- Calculates the total population size -/
def totalPopulation (p : Population) : ℕ :=
  p.high + p.middle + p.low

/-- Checks if the sample sizes are proportional to the population sizes -/
def isProportionalSample (p : Population) (s : Sample) (sampleSize : ℕ) : Prop :=
  s.high * (totalPopulation p) = sampleSize * p.high ∧
  s.middle * (totalPopulation p) = sampleSize * p.middle ∧
  s.low * (totalPopulation p) = sampleSize * p.low

/-- The main theorem stating that the given sample is proportional for the given population -/
theorem stratified_sample_correct 
  (pop : Population) 
  (sample : Sample) : 
  pop.high = 150 → 
  pop.middle = 360 → 
  pop.low = 90 → 
  sample.high = 25 → 
  sample.middle = 60 → 
  sample.low = 15 → 
  isProportionalSample pop sample 100 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_correct_l2485_248559


namespace NUMINAMATH_CALUDE_franks_mower_blades_expenditure_l2485_248511

theorem franks_mower_blades_expenditure 
  (total_earned : ℕ) 
  (games_affordable : ℕ) 
  (game_price : ℕ) 
  (h1 : total_earned = 19)
  (h2 : games_affordable = 4)
  (h3 : game_price = 2) :
  total_earned - games_affordable * game_price = 11 := by
sorry

end NUMINAMATH_CALUDE_franks_mower_blades_expenditure_l2485_248511


namespace NUMINAMATH_CALUDE_sqrt_36_times_sqrt_16_l2485_248514

theorem sqrt_36_times_sqrt_16 : Real.sqrt (36 * Real.sqrt 16) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_36_times_sqrt_16_l2485_248514


namespace NUMINAMATH_CALUDE_change_combinations_correct_l2485_248594

/-- The number of ways to change 15 dollars using nickels, dimes, and quarters -/
def change_combinations : ℕ :=
  (Finset.filter (fun (n, d, q) => n > 0 ∧ d > 0 ∧ q > 0)
    (Finset.filter (fun (n, d, q) => 5 * n + 10 * d + 25 * q = 1500)
      (Finset.product (Finset.range 301) (Finset.product (Finset.range 151) (Finset.range 61))))).card

/-- Theorem stating that change_combinations gives the correct number of ways to change 15 dollars -/
theorem change_combinations_correct :
  change_combinations = (Finset.filter (fun (n, d, q) => n > 0 ∧ d > 0 ∧ q > 0 ∧ 5 * n + 10 * d + 25 * q = 1500)
    (Finset.product (Finset.range 301) (Finset.product (Finset.range 151) (Finset.range 61)))).card :=
by sorry

#eval change_combinations

end NUMINAMATH_CALUDE_change_combinations_correct_l2485_248594


namespace NUMINAMATH_CALUDE_certain_value_calculation_l2485_248550

theorem certain_value_calculation (N : ℝ) : 
  (0.4 * N = 180) → ((1/4) * (1/3) * (2/5) * N = 15) := by
  sorry

end NUMINAMATH_CALUDE_certain_value_calculation_l2485_248550


namespace NUMINAMATH_CALUDE_prank_combinations_l2485_248596

theorem prank_combinations (monday tuesday wednesday thursday friday saturday sunday : ℕ) :
  monday = 1 →
  tuesday = 2 →
  wednesday = 6 →
  thursday = 5 →
  friday = 0 →
  saturday = 2 →
  sunday = 1 →
  monday * tuesday * wednesday * thursday * friday * saturday * sunday = 0 := by
sorry

end NUMINAMATH_CALUDE_prank_combinations_l2485_248596


namespace NUMINAMATH_CALUDE_jerome_speed_l2485_248520

/-- Proves that Jerome's speed is 4 MPH given the conditions of the problem -/
theorem jerome_speed (jerome_time nero_time nero_speed : ℝ) 
  (h1 : jerome_time = 6)
  (h2 : nero_time = 3)
  (h3 : nero_speed = 8) :
  jerome_time * (nero_speed * nero_time / jerome_time) = 4 := by
  sorry

#check jerome_speed

end NUMINAMATH_CALUDE_jerome_speed_l2485_248520


namespace NUMINAMATH_CALUDE_optimal_discount_savings_l2485_248579

def initial_order : ℝ := 15000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def apply_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

def discount_set1 : List ℝ := [0.25, 0.15, 0.10]
def discount_set2 : List ℝ := [0.30, 0.10, 0.05]

theorem optimal_discount_savings :
  apply_discounts initial_order discount_set2 - apply_discounts initial_order discount_set1 = 371.25 := by
  sorry

end NUMINAMATH_CALUDE_optimal_discount_savings_l2485_248579


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2485_248518

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 2 * (x - 175) * (x - 176) + 6

/-- The transformed quadratic function -/
def g (x : ℝ) : ℝ := f x - 6

/-- The roots of the transformed function -/
def root1 : ℝ := 175
def root2 : ℝ := 176

theorem quadratic_transformation :
  (g root1 = 0) ∧ 
  (g root2 = 0) ∧ 
  (root2 - root1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2485_248518


namespace NUMINAMATH_CALUDE_special_triangle_min_perimeter_l2485_248598

/-- Triangle ABC with integer side lengths and specific angle conditions -/
structure SpecialTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  angle_A_twice_B : angle_A = 2 * angle_B
  angle_C_obtuse : angle_C > Real.pi / 2
  angle_sum : angle_A + angle_B + angle_C = Real.pi

/-- The minimum perimeter of a SpecialTriangle is 77 -/
theorem special_triangle_min_perimeter (t : SpecialTriangle) : t.a + t.b + t.c ≥ 77 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_min_perimeter_l2485_248598


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l2485_248567

/-- Two triangles are similar if they have the same shape but not necessarily the same size. -/
def are_similar (t1 t2 : Triangle) : Prop := sorry

/-- An equilateral triangle has all sides equal and all angles equal to 60°. -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- An isosceles triangle with a 120° angle has two equal sides and two base angles of 30°. -/
def is_isosceles_120 (t : Triangle) : Prop := sorry

/-- Two triangles are congruent if they have the same shape and size. -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- A right triangle has one angle of 90°. -/
def is_right_triangle (t : Triangle) : Prop := sorry

theorem triangle_similarity_theorem :
  ∀ t1 t2 : Triangle,
  (is_equilateral t1 ∧ is_equilateral t2) → are_similar t1 t2 ∧
  (is_isosceles_120 t1 ∧ is_isosceles_120 t2) → are_similar t1 t2 ∧
  are_congruent t1 t2 → are_similar t1 t2 ∧
  ∃ t3 t4 : Triangle, is_right_triangle t3 ∧ is_right_triangle t4 ∧ ¬ are_similar t3 t4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l2485_248567


namespace NUMINAMATH_CALUDE_cube_root_8000_l2485_248584

theorem cube_root_8000 : ∃ (a b : ℕ+), (a : ℝ) * (b : ℝ)^(1/3 : ℝ) = 8000^(1/3 : ℝ) ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_8000_l2485_248584


namespace NUMINAMATH_CALUDE_point_b_coordinates_l2485_248581

/-- Given point A and vector a, if vector AB = 2a, then point B has specific coordinates -/
theorem point_b_coordinates (A B : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (1, -3) →
  a = (3, 4) →
  B - A = 2 • a →
  B = (7, 5) := by
  sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l2485_248581


namespace NUMINAMATH_CALUDE_total_cost_of_cows_l2485_248592

-- Define the number of cards in a standard deck
def standard_deck_size : ℕ := 52

-- Define the number of hearts per card
def hearts_per_card : ℕ := 4

-- Define the cost per cow
def cost_per_cow : ℕ := 200

-- Define the number of cows in Devonshire
def cows_in_devonshire : ℕ := 2 * (standard_deck_size * hearts_per_card)

-- State the theorem
theorem total_cost_of_cows : cows_in_devonshire * cost_per_cow = 83200 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_cows_l2485_248592


namespace NUMINAMATH_CALUDE_january_oil_bill_l2485_248546

/-- Proves that January's oil bill is $120 given the specified conditions --/
theorem january_oil_bill (feb_bill jan_bill : ℚ) : 
  (feb_bill / jan_bill = 3 / 2) → 
  ((feb_bill + 20) / jan_bill = 5 / 3) →
  jan_bill = 120 := by
  sorry

end NUMINAMATH_CALUDE_january_oil_bill_l2485_248546


namespace NUMINAMATH_CALUDE_widget_purchase_problem_l2485_248526

theorem widget_purchase_problem (C W : ℚ) 
  (h1 : 8 * (C - 1.25) = 16.67)
  (h2 : 16.67 / C = W) : 
  W = 5 := by
sorry

end NUMINAMATH_CALUDE_widget_purchase_problem_l2485_248526


namespace NUMINAMATH_CALUDE_incorrect_operation_l2485_248527

theorem incorrect_operation : 
  (5 - (-2) = 7) ∧ 
  (-9 / (-3) = 3) ∧ 
  (-4 * (-5) = 20) ∧ 
  (-5 + 3 ≠ 8) := by
sorry

end NUMINAMATH_CALUDE_incorrect_operation_l2485_248527


namespace NUMINAMATH_CALUDE_lyras_remaining_budget_l2485_248539

/-- Calculates the remaining budget after food purchases -/
def remaining_budget (weekly_budget : ℕ) (chicken_cost : ℕ) (beef_price_per_pound : ℕ) (beef_pounds : ℕ) : ℕ :=
  weekly_budget - (chicken_cost + beef_price_per_pound * beef_pounds)

/-- Proves that Lyra's remaining budget is $53 -/
theorem lyras_remaining_budget :
  remaining_budget 80 12 3 5 = 53 := by
  sorry

end NUMINAMATH_CALUDE_lyras_remaining_budget_l2485_248539


namespace NUMINAMATH_CALUDE_rakesh_distance_rakesh_walked_approx_distance_l2485_248568

/-- Proves that Rakesh walked approximately 28.29 kilometers given the conditions of the problem. -/
theorem rakesh_distance (hiro_distance : ℝ) : ℝ :=
  let rakesh_distance := 4 * hiro_distance - 10
  let sanjay_distance := 2 * hiro_distance + 3
  have total_distance : hiro_distance + rakesh_distance + sanjay_distance = 60 := by sorry
  have hiro_calc : hiro_distance = 67 / 7 := by sorry
  rakesh_distance

/-- The approximate distance Rakesh walked -/
def rakesh_approx_distance : ℝ := 28.29

theorem rakesh_walked_approx_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |rakesh_distance (67 / 7) - rakesh_approx_distance| < ε :=
by sorry

end NUMINAMATH_CALUDE_rakesh_distance_rakesh_walked_approx_distance_l2485_248568


namespace NUMINAMATH_CALUDE_xyz_equals_two_l2485_248537

theorem xyz_equals_two
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = (b + c) / (x - 3))
  (eq_b : b = (a + c) / (y - 3))
  (eq_c : c = (a + b) / (z - 3))
  (sum_xy_xz_yz : x * y + x * z + y * z = 7)
  (sum_x_y_z : x + y + z = 3) :
  x * y * z = 2 := by
sorry


end NUMINAMATH_CALUDE_xyz_equals_two_l2485_248537


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l2485_248516

theorem complex_modulus_equality (x y : ℝ) : 
  (Complex.I + 1) * x = Complex.I * y + 1 → Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l2485_248516


namespace NUMINAMATH_CALUDE_correct_decision_probability_l2485_248513

-- Define the probability of a consultant giving a correct opinion
def p_correct : ℝ := 0.8

-- Define the number of consultants
def n_consultants : ℕ := 3

-- Define the probability of making a correct decision
def p_correct_decision : ℝ :=
  (Nat.choose n_consultants 2) * p_correct^2 * (1 - p_correct) +
  (Nat.choose n_consultants 3) * p_correct^3

-- Theorem statement
theorem correct_decision_probability :
  p_correct_decision = 0.896 := by sorry

end NUMINAMATH_CALUDE_correct_decision_probability_l2485_248513


namespace NUMINAMATH_CALUDE_altitude_length_is_one_l2485_248572

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point lies on the parabola y = x^2 -/
def onParabola (p : Point) : Prop :=
  p.y = p.x^2

/-- Checks if a line segment is parallel to the x-axis -/
def parallelToXAxis (p1 p2 : Point) : Prop :=
  p1.y = p2.y

/-- Checks if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop :=
  let AC := (t.C.x - t.A.x, t.C.y - t.A.y)
  let BC := (t.C.x - t.B.x, t.C.y - t.B.y)
  AC.1 * BC.1 + AC.2 * BC.2 = 0

/-- Calculates the length of the altitude from C to AB -/
def altitudeLength (t : Triangle) : ℝ :=
  t.A.y - t.C.y

/-- The main theorem -/
theorem altitude_length_is_one (t : Triangle) :
  isRightTriangle t →
  onParabola t.A ∧ onParabola t.B ∧ onParabola t.C →
  parallelToXAxis t.A t.B →
  altitudeLength t = 1 := by
  sorry

end NUMINAMATH_CALUDE_altitude_length_is_one_l2485_248572


namespace NUMINAMATH_CALUDE_solution_set_f_range_g_a_gt_2_range_g_a_lt_2_range_g_a_eq_2_l2485_248512

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + x

-- Define the function g
def g (a x : ℝ) : ℝ := f x - |a*x - 1| - x

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f (x : ℝ) : 
  f x ≤ 5 ↔ x ∈ Set.Icc (-6) (4/3) :=
sorry

-- Theorem for the range of g(x) when a > 2
theorem range_g_a_gt_2 (a : ℝ) (h : a > 2) :
  Set.range (g a) = Set.Iic (2/a + 1) :=
sorry

-- Theorem for the range of g(x) when 0 < a < 2
theorem range_g_a_lt_2 (a : ℝ) (h1 : a > 0) (h2 : a < 2) :
  Set.range (g a) = Set.Ici (-a/2 - 1) :=
sorry

-- Theorem for the range of g(x) when a = 2
theorem range_g_a_eq_2 :
  Set.range (g 2) = Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_range_g_a_gt_2_range_g_a_lt_2_range_g_a_eq_2_l2485_248512


namespace NUMINAMATH_CALUDE_segment_equality_l2485_248525

/-- Given points A, B, C, D, E, F on a line with certain distance relationships,
    prove that CD = AB = EF. -/
theorem segment_equality 
  (A B C D E F : ℝ) -- Points on a line represented as real numbers
  (h1 : A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F) -- Points are ordered on the line
  (h2 : C - A = E - C) -- AC = CE
  (h3 : D - B = F - D) -- BD = DF
  (h4 : D - A = F - C) -- AD = CF
  : D - C = B - A ∧ D - C = F - E := by
  sorry

end NUMINAMATH_CALUDE_segment_equality_l2485_248525


namespace NUMINAMATH_CALUDE_care_package_weight_ratio_l2485_248530

/-- Represents the weight of the care package at different stages --/
structure CarePackage where
  initial_weight : ℝ
  after_brownies : ℝ
  before_gummies : ℝ
  final_weight : ℝ

/-- Theorem stating the ratio of final weight to weight before gummies is 2:1 --/
theorem care_package_weight_ratio (package : CarePackage) : 
  package.initial_weight = 2 →
  package.after_brownies = 3 * package.initial_weight →
  package.before_gummies = package.after_brownies + 2 →
  package.final_weight = 16 →
  package.final_weight / package.before_gummies = 2 := by
  sorry


end NUMINAMATH_CALUDE_care_package_weight_ratio_l2485_248530


namespace NUMINAMATH_CALUDE_range_of_a_l2485_248536

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 / Real.exp x - 2 * x

theorem range_of_a (a : ℝ) (h : f (a - 3) + f (2 * a^2) ≤ 0) :
  -3/2 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2485_248536


namespace NUMINAMATH_CALUDE_acute_angle_sine_equivalence_l2485_248529

theorem acute_angle_sine_equivalence (α β : Real) 
  (h_α_acute : 0 < α ∧ α < Real.pi / 2)
  (h_β_acute : 0 < β ∧ β < Real.pi / 2) :
  (α > 2 * β) ↔ (Real.sin (α - β) > Real.sin β) := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_sine_equivalence_l2485_248529


namespace NUMINAMATH_CALUDE_certain_number_problem_l2485_248551

theorem certain_number_problem (x : ℝ) : 0.7 * x = (4/5 * 25) + 8 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2485_248551


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2485_248545

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1: Solution set of f(x) ≥ 6 when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2: Range of a for which f(x) > -a for all x
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2485_248545


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l2485_248573

theorem triangle_side_ratio (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ x y, (x = a ∧ y = b) ∨ (x = a ∧ y = c) ∨ (x = b ∧ y = c) ∧
  ((Real.sqrt 5 - 1) / 2 ≤ x / y) ∧ (x / y ≤ (Real.sqrt 5 + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l2485_248573


namespace NUMINAMATH_CALUDE_binomial_1000_1000_l2485_248547

theorem binomial_1000_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1000_1000_l2485_248547


namespace NUMINAMATH_CALUDE_congruence_solution_l2485_248549

theorem congruence_solution (n : ℕ) : n ∈ Finset.range 47 ∧ 13 * n ≡ 9 [MOD 47] ↔ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2485_248549


namespace NUMINAMATH_CALUDE_cube_octahedron_surface_area_ratio_l2485_248543

/-- The ratio of the surface area of a cube to the surface area of an inscribed regular octahedron --/
theorem cube_octahedron_surface_area_ratio :
  ∀ (cube_side_length : ℝ) (octahedron_side_length : ℝ),
    cube_side_length = 2 →
    octahedron_side_length = Real.sqrt 2 →
    (6 * cube_side_length^2) / (2 * Real.sqrt 3 * octahedron_side_length^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_octahedron_surface_area_ratio_l2485_248543


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_l2485_248576

-- Define the problem parameters
def raw_material_price : ℝ := 30
def min_selling_price : ℝ := 30
def max_selling_price : ℝ := 60
def additional_cost : ℝ := 450

-- Define the sales volume function
def sales_volume (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - raw_material_price) * sales_volume x - additional_cost

-- State the theorem
theorem max_profit_at_max_price :
  ∀ x ∈ Set.Icc min_selling_price max_selling_price,
    profit x ≤ profit max_selling_price ∧
    profit max_selling_price = 1950 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_l2485_248576


namespace NUMINAMATH_CALUDE_min_value_theorem_l2485_248571

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / (x + 2)) + (1 / (y + 1)) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2485_248571


namespace NUMINAMATH_CALUDE_max_value_theorem_l2485_248564

theorem max_value_theorem (x y : ℝ) (h : x^2/4 + y^2 = 1) :
  ∃ (max_val : ℝ), max_val = (1 + Real.sqrt 2) / 2 ∧
  ∀ (z : ℝ), z = x*y/(x + 2*y - 2) → z ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2485_248564


namespace NUMINAMATH_CALUDE_craig_walk_distance_l2485_248566

/-- The distance Craig walked from school to David's house in miles -/
def school_to_david : ℝ := 0.2

/-- The total distance Craig walked in miles -/
def total_distance : ℝ := 0.9

/-- The distance Craig walked from David's house to his own house in miles -/
def david_to_craig : ℝ := total_distance - school_to_david

theorem craig_walk_distance : david_to_craig = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_craig_walk_distance_l2485_248566


namespace NUMINAMATH_CALUDE_initial_pens_count_l2485_248586

theorem initial_pens_count (P : ℕ) : P = 5 :=
  by
  have h1 : 2 * (P + 20) - 19 = 31 := by sorry
  sorry

end NUMINAMATH_CALUDE_initial_pens_count_l2485_248586


namespace NUMINAMATH_CALUDE_solution_ratio_l2485_248569

theorem solution_ratio (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_ratio_l2485_248569


namespace NUMINAMATH_CALUDE_derivative_at_two_l2485_248548

/-- Given a function f(x) = x^2 - x, prove that its derivative at x = 2 is 6 -/
theorem derivative_at_two (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - x) : 
  deriv f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_two_l2485_248548


namespace NUMINAMATH_CALUDE_bouncy_ball_difference_l2485_248599

-- Define the given quantities
def red_packs : ℕ := 12
def yellow_packs : ℕ := 9
def balls_per_red_pack : ℕ := 24
def balls_per_yellow_pack : ℕ := 20

-- Define the theorem
theorem bouncy_ball_difference :
  red_packs * balls_per_red_pack - yellow_packs * balls_per_yellow_pack = 108 := by
  sorry

end NUMINAMATH_CALUDE_bouncy_ball_difference_l2485_248599


namespace NUMINAMATH_CALUDE_perfume_usage_fraction_l2485_248595

/-- The fraction of perfume used in a cylindrical bottle -/
theorem perfume_usage_fraction 
  (r : ℝ) -- radius of the cylinder base
  (h : ℝ) -- height of the cylinder
  (v_remaining : ℝ) -- volume of remaining perfume in liters
  (hr : r = 7) -- given radius
  (hh : h = 10) -- given height
  (hv : v_remaining = 0.45) -- given remaining volume
  : (π * r^2 * h / 1000 - v_remaining) / (π * r^2 * h / 1000) = (49 * π - 45) / (49 * π) :=
by sorry

end NUMINAMATH_CALUDE_perfume_usage_fraction_l2485_248595


namespace NUMINAMATH_CALUDE_max_distinct_words_equiv_max_distinct_words_correct_l2485_248503

/-- The maximum number of distinct n-letter words that can be formed by observing a convex n-gon with n distinct labeled corners from outside the polygon. -/
def max_distinct_words (n : ℕ) : ℕ :=
  (1/12) * n * (n-1) * (n^2 - 5*n + 18)

/-- The maximum number of distinct n-letter words is equal to 2 * C(n,2) + 2 * C(n,4), where C(n,k) is the binomial coefficient. -/
theorem max_distinct_words_equiv (n : ℕ) :
  max_distinct_words n = 2 * (n.choose 2) + 2 * (n.choose 4) :=
sorry

/-- The maximum number of distinct n-letter words formula is correct for convex n-gons. -/
theorem max_distinct_words_correct (n : ℕ) (n_ge_3 : n ≥ 3) :
  max_distinct_words n = (1/12) * n * (n-1) * (n^2 - 5*n + 18) :=
sorry

end NUMINAMATH_CALUDE_max_distinct_words_equiv_max_distinct_words_correct_l2485_248503


namespace NUMINAMATH_CALUDE_floor_sum_inequality_l2485_248515

theorem floor_sum_inequality (x y : ℝ) :
  (⌊x⌋ : ℝ) + ⌊y⌋ ≤ ⌊x + y⌋ ∧ ⌊x + y⌋ ≤ (⌊x⌋ : ℝ) + ⌊y⌋ + 1 ∧
  ((⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋) ∨ (⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋ + 1)) ∧
  ¬((⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋) ∧ (⌊x + y⌋ = (⌊x⌋ : ℝ) + ⌊y⌋ + 1)) :=
by sorry


end NUMINAMATH_CALUDE_floor_sum_inequality_l2485_248515


namespace NUMINAMATH_CALUDE_place_value_ratio_l2485_248531

def number : ℚ := 86304.2957

theorem place_value_ratio :
  let thousands_place_value : ℚ := 1000
  let tenths_place_value : ℚ := 0.1
  (thousands_place_value / tenths_place_value : ℚ) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l2485_248531


namespace NUMINAMATH_CALUDE_second_set_cost_l2485_248510

/-- The cost of a set of footballs and soccer balls -/
def cost_of_set (football_price : ℝ) (soccer_price : ℝ) (num_footballs : ℕ) (num_soccers : ℕ) : ℝ :=
  football_price * (num_footballs : ℝ) + soccer_price * (num_soccers : ℝ)

/-- The theorem stating the cost of the second set of balls -/
theorem second_set_cost :
  ∀ (football_price : ℝ),
  cost_of_set football_price 50 3 1 = 155 →
  cost_of_set football_price 50 2 3 = 220 :=
by
  sorry

end NUMINAMATH_CALUDE_second_set_cost_l2485_248510


namespace NUMINAMATH_CALUDE_bookshop_unsold_percentage_l2485_248585

/-- The percentage of unsold books in a bookshop -/
def unsold_percentage (initial_stock : ℕ) (mon_sales tues_sales wed_sales thurs_sales fri_sales : ℕ) : ℚ :=
  (initial_stock - (mon_sales + tues_sales + wed_sales + thurs_sales + fri_sales)) / initial_stock * 100

/-- Theorem stating the percentage of unsold books for the given scenario -/
theorem bookshop_unsold_percentage :
  unsold_percentage 1300 75 50 64 78 135 = 69.15384615384615 := by
  sorry

end NUMINAMATH_CALUDE_bookshop_unsold_percentage_l2485_248585


namespace NUMINAMATH_CALUDE_z_relation_to_x_minus_2y_l2485_248570

theorem z_relation_to_x_minus_2y (x y z : ℝ) 
  (h1 : x > y) (h2 : y > 1) (h3 : z = (x + 3) - 2 * (y - 5)) :
  z = x - 2 * y + 13 := by
  sorry

end NUMINAMATH_CALUDE_z_relation_to_x_minus_2y_l2485_248570


namespace NUMINAMATH_CALUDE_juice_bar_group_size_l2485_248562

theorem juice_bar_group_size :
  let total_spent : ℕ := 94
  let mango_price : ℕ := 5
  let pineapple_price : ℕ := 6
  let pineapple_spent : ℕ := 54
  let mango_spent : ℕ := total_spent - pineapple_spent
  let mango_people : ℕ := mango_spent / mango_price
  let pineapple_people : ℕ := pineapple_spent / pineapple_price
  mango_people + pineapple_people = 17 :=
by sorry

end NUMINAMATH_CALUDE_juice_bar_group_size_l2485_248562


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2485_248560

theorem consecutive_integers_sum (m : ℤ) :
  let sequence := [m, m+1, m+2, m+3, m+4, m+5, m+6]
  (sequence.sum - (sequence.take 3).sum) = 4*m + 18 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2485_248560


namespace NUMINAMATH_CALUDE_nine_special_integers_l2485_248583

theorem nine_special_integers (m n : ℕ) (hm : m ≥ 16) (hn : n ≥ 24) :
  ∃ (a : Fin 9 → ℕ),
    (∀ k : Fin 9, a k = 2^(m + k.val) * 3^(n - k.val)) ∧
    (∀ k : Fin 9, 6 ∣ a k) ∧
    (∀ i j : Fin 9, i ≠ j → ¬(a i ∣ a j)) ∧
    (∀ i j : Fin 9, (a i)^3 ∣ (a j)^2) := by
  sorry

end NUMINAMATH_CALUDE_nine_special_integers_l2485_248583


namespace NUMINAMATH_CALUDE_exists_construction_with_1001_free_endpoints_l2485_248542

/-- Represents a point in the construction --/
structure Point :=
  (depth : ℕ)
  (branches : Fin 5)

/-- Represents the construction of line segments --/
def Construction := List Point

/-- Counts the number of free endpoints in a construction --/
def count_free_endpoints (c : Construction) : ℕ := sorry

/-- Theorem: There exists a construction with 1001 free endpoints --/
theorem exists_construction_with_1001_free_endpoints :
  ∃ (c : Construction), count_free_endpoints c = 1001 := by sorry

end NUMINAMATH_CALUDE_exists_construction_with_1001_free_endpoints_l2485_248542


namespace NUMINAMATH_CALUDE_lionel_graham_crackers_left_l2485_248590

/-- Represents the ingredients for making Oreo cheesecakes -/
structure Ingredients where
  graham_crackers : ℕ
  oreos : ℕ
  cream_cheese : ℕ

/-- Represents the recipe requirements for one Oreo cheesecake -/
structure Recipe where
  graham_crackers : ℕ
  oreos : ℕ
  cream_cheese : ℕ

/-- Calculates the maximum number of cheesecakes that can be made given the ingredients and recipe -/
def max_cheesecakes (ingredients : Ingredients) (recipe : Recipe) : ℕ :=
  min (ingredients.graham_crackers / recipe.graham_crackers)
      (min (ingredients.oreos / recipe.oreos)
           (ingredients.cream_cheese / recipe.cream_cheese))

/-- Calculates the number of Graham cracker boxes left over after making the maximum number of cheesecakes -/
def graham_crackers_left (ingredients : Ingredients) (recipe : Recipe) : ℕ :=
  ingredients.graham_crackers - (max_cheesecakes ingredients recipe * recipe.graham_crackers)

/-- Theorem stating that Lionel will have 4 boxes of Graham crackers left over -/
theorem lionel_graham_crackers_left :
  let ingredients := Ingredients.mk 14 15 36
  let recipe := Recipe.mk 2 3 4
  graham_crackers_left ingredients recipe = 4 := by
  sorry

end NUMINAMATH_CALUDE_lionel_graham_crackers_left_l2485_248590


namespace NUMINAMATH_CALUDE_cat_litter_cost_l2485_248589

/-- Proves the cost of a cat litter container given specific conditions --/
theorem cat_litter_cost 
  (container_size : ℕ) 
  (litter_box_capacity : ℕ) 
  (change_frequency : ℕ) 
  (total_cost : ℕ) 
  (total_days : ℕ) 
  (h1 : container_size = 45)
  (h2 : litter_box_capacity = 15)
  (h3 : change_frequency = 7)
  (h4 : total_cost = 210)
  (h5 : total_days = 210) :
  total_cost / (total_days / change_frequency * litter_box_capacity / container_size) = 21 := by
  sorry


end NUMINAMATH_CALUDE_cat_litter_cost_l2485_248589


namespace NUMINAMATH_CALUDE_one_pair_probability_l2485_248532

-- Define the total number of socks
def total_socks : ℕ := 12

-- Define the number of colors
def num_colors : ℕ := 4

-- Define the number of socks per color
def socks_per_color : ℕ := 3

-- Define the number of socks drawn
def socks_drawn : ℕ := 5

-- Define the probability of drawing exactly one pair of socks with the same color
def prob_one_pair : ℚ := 9/22

-- Theorem statement
theorem one_pair_probability :
  (total_socks = num_colors * socks_per_color) →
  (socks_drawn = 5) →
  (prob_one_pair = 9/22) := by
  sorry

end NUMINAMATH_CALUDE_one_pair_probability_l2485_248532


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_3007_l2485_248574

-- Define a function to get the last two digits of a number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define the sequence of last two digits of 13^n
def lastTwoDigitsOf13Pow (n : ℕ) : ℕ :=
  match n % 10 with
  | 0 => 49
  | 1 => 37
  | 2 => 81
  | 3 => 53
  | 4 => 89
  | 5 => 57
  | 6 => 41
  | 7 => 17
  | 8 => 21
  | 9 => 73
  | _ => 0  -- This case should never occur

-- Theorem statement
theorem tens_digit_of_13_pow_3007 :
  (lastTwoDigitsOf13Pow 3007) / 10 = 1 := by
  sorry


end NUMINAMATH_CALUDE_tens_digit_of_13_pow_3007_l2485_248574


namespace NUMINAMATH_CALUDE_remainder_8423_div_9_l2485_248558

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The digital root of a natural number (iterative sum of digits until a single digit is reached) -/
def digital_root (n : ℕ) : ℕ := sorry

theorem remainder_8423_div_9 : 8423 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_8423_div_9_l2485_248558


namespace NUMINAMATH_CALUDE_rafael_earnings_l2485_248534

def hours_monday : ℕ := 10
def hours_tuesday : ℕ := 8
def hours_left : ℕ := 20
def hourly_rate : ℕ := 20

theorem rafael_earnings : 
  (hours_monday + hours_tuesday + hours_left) * hourly_rate = 760 := by
  sorry

end NUMINAMATH_CALUDE_rafael_earnings_l2485_248534


namespace NUMINAMATH_CALUDE_quadratic_zero_condition_l2485_248554

/-- A quadratic function f(x) = x^2 - 2x + m has a zero in (-1, 0) if and only if -3 < m < 0 -/
theorem quadratic_zero_condition (m : ℝ) : 
  (∃ x : ℝ, -1 < x ∧ x < 0 ∧ x^2 - 2*x + m = 0) ↔ -3 < m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_zero_condition_l2485_248554


namespace NUMINAMATH_CALUDE_exam_average_l2485_248580

theorem exam_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) (h₁ : n₁ = 15) (h₂ : n₂ = 10)
  (h₃ : avg₁ = 75/100) (h₄ : avg_total = 83/100) (h₅ : n₁ + n₂ = 25) :
  let avg₂ := (((n₁ + n₂ : ℚ) * avg_total) - (n₁ * avg₁)) / n₂
  avg₂ = 95/100 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l2485_248580


namespace NUMINAMATH_CALUDE_m_arithmetic_pascal_triangle_structure_l2485_248544

/-- m-arithmetic Pascal triangle with s-th row zeros except extremes -/
structure MArithmeticPascalTriangle where
  m : ℕ
  s : ℕ
  Δ : ℕ → ℕ → ℕ
  P : ℕ → ℕ → ℕ

/-- Properties of the m-arithmetic Pascal triangle -/
class MArithmeticPascalTriangleProps (T : MArithmeticPascalTriangle) where
  zero_middle : ∀ (i k : ℕ), i % T.s = 0 → 0 < k → k < i → T.Δ i k = 0
  relation_a : ∀ (n k : ℕ), T.Δ n (k-1) + T.Δ n k = T.Δ (n+1) k
  relation_b : ∀ (n k : ℕ), T.Δ n k = T.P n k * T.Δ 0 0

/-- Theorem: The structure of the m-arithmetic Pascal triangle is correct -/
theorem m_arithmetic_pascal_triangle_structure 
  (T : MArithmeticPascalTriangle) 
  [MArithmeticPascalTriangleProps T] : 
  ∀ (n k : ℕ), T.Δ n k = T.P n k * T.Δ 0 0 := by
  sorry

end NUMINAMATH_CALUDE_m_arithmetic_pascal_triangle_structure_l2485_248544


namespace NUMINAMATH_CALUDE_honey_water_percentage_l2485_248557

/-- Given that 1.5 kg of flower-nectar yields 1 kg of honey and nectar contains 50% water,
    prove that the resulting honey contains 25% water. -/
theorem honey_water_percentage :
  ∀ (nectar_mass honey_mass water_percentage_nectar : ℝ),
    nectar_mass = 1.5 →
    honey_mass = 1 →
    water_percentage_nectar = 50 →
    (honey_mass - (nectar_mass * (1 - water_percentage_nectar / 100))) / honey_mass * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_honey_water_percentage_l2485_248557


namespace NUMINAMATH_CALUDE_luke_weed_eating_earnings_l2485_248528

/-- Proves that Luke made $18 weed eating given the conditions of the problem -/
theorem luke_weed_eating_earnings :
  ∀ (weed_eating_earnings : ℕ),
    9 + weed_eating_earnings = 3 * 9 →
    weed_eating_earnings = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_luke_weed_eating_earnings_l2485_248528


namespace NUMINAMATH_CALUDE_loss_per_metre_calculation_l2485_248578

/-- Calculates the loss per metre of cloth given the total metres sold, total selling price, and cost price per metre. -/
def loss_per_metre (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) : ℕ :=
  (total_metres * cost_price_per_metre - total_selling_price) / total_metres

/-- Theorem stating that given 200 metres of cloth sold for Rs. 18000 with a cost price of Rs. 95 per metre, the loss per metre is Rs. 5. -/
theorem loss_per_metre_calculation :
  loss_per_metre 200 18000 95 = 5 := by
  sorry

end NUMINAMATH_CALUDE_loss_per_metre_calculation_l2485_248578


namespace NUMINAMATH_CALUDE_least_integer_satisfying_condition_l2485_248505

/-- Given a positive integer n, returns the integer formed by removing its leftmost digit. -/
def removeLeftmostDigit (n : ℕ+) : ℕ :=
  sorry

/-- Checks if a positive integer satisfies the condition that removing its leftmost digit
    results in 1/29 of the original number. -/
def satisfiesCondition (n : ℕ+) : Prop :=
  removeLeftmostDigit n = n.val / 29

/-- Proves that 725 is the least positive integer that satisfies the given condition. -/
theorem least_integer_satisfying_condition :
  satisfiesCondition 725 ∧ ∀ m : ℕ+, m < 725 → ¬satisfiesCondition m :=
sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_condition_l2485_248505


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l2485_248538

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : (a^2 - b^2) + c^2 = 8)
  (h2 : a * b * c = 2) :
  a^4 + b^4 + c^4 = 70 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l2485_248538


namespace NUMINAMATH_CALUDE_kite_parabolas_sum_l2485_248588

/-- Given two parabolas that intersect the coordinate axes in four points forming a kite -/
structure KiteParabolas where
  /-- Coefficient of x^2 in the first parabola y = ax^2 - 3 -/
  a : ℝ
  /-- Coefficient of x^2 in the second parabola y = 5 - bx^2 -/
  b : ℝ
  /-- The four intersection points form a kite -/
  is_kite : Bool
  /-- The area of the kite formed by the intersection points -/
  kite_area : ℝ
  /-- The parabolas intersect the coordinate axes in exactly four points -/
  four_intersections : Bool

/-- Theorem stating that under the given conditions, a + b = 128/81 -/
theorem kite_parabolas_sum (k : KiteParabolas) 
  (h1 : k.is_kite = true) 
  (h2 : k.kite_area = 18) 
  (h3 : k.four_intersections = true) : 
  k.a + k.b = 128/81 := by
  sorry

end NUMINAMATH_CALUDE_kite_parabolas_sum_l2485_248588


namespace NUMINAMATH_CALUDE_mrs_heine_biscuits_l2485_248522

/-- Given a number of dogs and biscuits per dog, calculates the total number of biscuits needed -/
def total_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog

/-- Theorem: Mrs. Heine needs to buy 6 biscuits for her 2 dogs, given 3 biscuits per dog -/
theorem mrs_heine_biscuits : total_biscuits 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mrs_heine_biscuits_l2485_248522


namespace NUMINAMATH_CALUDE_bubble_sort_correct_l2485_248597

def bubble_sort (xs : List Int) : List Int :=
  let rec pass (ys : List Int) : List Int :=
    match ys with
    | [] => []
    | [x] => [x]
    | x :: y :: rest =>
      if x > y
      then y :: pass (x :: rest)
      else x :: pass (y :: rest)
  let rec sort (zs : List Int) (n : Nat) : List Int :=
    if n = 0 then zs
    else sort (pass zs) (n - 1)
  sort xs xs.length

theorem bubble_sort_correct (xs : List Int) :
  bubble_sort [8, 6, 3, 18, 21, 67, 54] = [3, 6, 8, 18, 21, 54, 67] := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_correct_l2485_248597


namespace NUMINAMATH_CALUDE_cost_per_bottle_l2485_248508

/-- Given that 3 bottles cost €1.50 and 4 bottles cost €2, prove that the cost per bottle is €0.50 -/
theorem cost_per_bottle (cost_three : ℝ) (cost_four : ℝ) 
  (h1 : cost_three = 1.5) 
  (h2 : cost_four = 2) : 
  cost_three / 3 = 0.5 ∧ cost_four / 4 = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_cost_per_bottle_l2485_248508


namespace NUMINAMATH_CALUDE_hollow_sphere_weight_double_radius_l2485_248587

/-- The weight of a hollow sphere given its radius -/
noncomputable def sphereWeight (r : ℝ) : ℝ :=
  4 * Real.pi * r^2

theorem hollow_sphere_weight_double_radius (r : ℝ) (h : r > 0) :
  sphereWeight r = 8 → sphereWeight (2 * r) = 32 := by
  sorry

end NUMINAMATH_CALUDE_hollow_sphere_weight_double_radius_l2485_248587


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2485_248521

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed_calculation (train_length bridge_length : Real) (time : Real) : 
  train_length = 120 → 
  bridge_length = 160 → 
  time = 25.2 → 
  ∃ (speed : Real), abs (speed - ((train_length + bridge_length) / time)) < 0.0001 ∧ 
                    abs (speed - 11.1111) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_calculation_l2485_248521


namespace NUMINAMATH_CALUDE_point_A_coordinates_l2485_248501

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation to the left -/
def translateLeft (p : Point) (dx : ℝ) : Point :=
  ⟨p.x - dx, p.y⟩

/-- Translation upwards -/
def translateUp (p : Point) (dy : ℝ) : Point :=
  ⟨p.x, p.y + dy⟩

theorem point_A_coordinates :
  ∃ (A : Point),
    ∃ (dx dy : ℝ),
      translateLeft A dx = Point.mk 1 2 ∧
      translateUp A dy = Point.mk 3 4 ∧
      A = Point.mk 3 2 := by
  sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l2485_248501


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt28_l2485_248582

theorem consecutive_integers_around_sqrt28 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 28) → (Real.sqrt 28 < b) → (a + b = 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt28_l2485_248582


namespace NUMINAMATH_CALUDE_range_of_a_l2485_248519

/-- Custom binary operation on real numbers -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of 'a' given the inequality condition -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2485_248519


namespace NUMINAMATH_CALUDE_cube_cutting_l2485_248593

theorem cube_cutting (n s : ℕ) : n > s → n^3 - s^3 = 152 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_l2485_248593


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2485_248533

def ellipse_equation (e : ℝ) (l : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) => x^2 / 32 + y^2 / 16 = 1

theorem ellipse_standard_equation (e l : ℝ) 
  (h1 : e = Real.sqrt 2 / 2) 
  (h2 : l = 8) : 
  ellipse_equation e l = fun (x, y) => x^2 / 32 + y^2 / 16 = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l2485_248533


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l2485_248506

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_a8 (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 4 + a 6 = 8)
  (h_a2 : a 2 = 3) :
  a 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l2485_248506


namespace NUMINAMATH_CALUDE_train_passengers_l2485_248541

theorem train_passengers (initial : ℕ) 
  (first_off first_on second_off second_on final : ℕ) : 
  first_off = 29 → 
  first_on = 17 → 
  second_off = 27 → 
  second_on = 35 → 
  final = 116 → 
  initial = 120 → 
  initial - first_off + first_on - second_off + second_on = final :=
by sorry

end NUMINAMATH_CALUDE_train_passengers_l2485_248541


namespace NUMINAMATH_CALUDE_largest_cube_surface_area_l2485_248535

-- Define the dimensions of the cuboid
def cuboid_width : ℝ := 12
def cuboid_length : ℝ := 16
def cuboid_height : ℝ := 14

-- Define the function to calculate the surface area of a cube
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length^2

-- Theorem statement
theorem largest_cube_surface_area :
  let max_side_length := min cuboid_width (min cuboid_length cuboid_height)
  cube_surface_area max_side_length = 864 := by
sorry

end NUMINAMATH_CALUDE_largest_cube_surface_area_l2485_248535


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l2485_248509

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 24 cm and height 16 cm is 384 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 24 16 = 384 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l2485_248509


namespace NUMINAMATH_CALUDE_shooting_range_problem_l2485_248556

theorem shooting_range_problem :
  ∀ n k : ℕ,
  (10 < n) →
  (n < 20) →
  (5 * k = 3 * (n - k)) →
  (n = 16 ∧ k = 6) :=
by sorry

end NUMINAMATH_CALUDE_shooting_range_problem_l2485_248556


namespace NUMINAMATH_CALUDE_larger_rhombus_side_length_l2485_248552

/-- Two similar rhombi sharing a diagonal -/
structure SimilarRhombi where
  small_area : ℝ
  large_area : ℝ
  shared_diagonal : ℝ
  similar : small_area > 0 ∧ large_area > 0

/-- The side length of a rhombus -/
def side_length (r : SimilarRhombi) : ℝ → ℝ := sorry

/-- Theorem: The side length of the larger rhombus is √15 -/
theorem larger_rhombus_side_length (r : SimilarRhombi) 
  (h1 : r.small_area = 1) 
  (h2 : r.large_area = 9) : 
  side_length r r.large_area = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_larger_rhombus_side_length_l2485_248552


namespace NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l2485_248500

theorem sin_50_plus_sqrt3_tan_10_equals_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_50_plus_sqrt3_tan_10_equals_1_l2485_248500


namespace NUMINAMATH_CALUDE_work_completion_time_l2485_248555

/-- The time it takes to complete a work given two workers with different rates and a delayed start for one worker. -/
theorem work_completion_time 
  (p_rate : ℝ) (q_rate : ℝ) (p_solo_days : ℝ) 
  (hp : p_rate = 1 / 80)
  (hq : q_rate = 1 / 48)
  (hp_solo : p_solo_days = 16) : 
  p_solo_days + (1 - p_rate * p_solo_days) / (p_rate + q_rate) = 40 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l2485_248555


namespace NUMINAMATH_CALUDE_journey_speed_theorem_l2485_248591

/-- Proves that given a round trip journey where the time taken to go up is twice the time taken to come down,
    the total journey time is 6 hours, and the average speed for the whole journey is 4 km/h,
    then the average speed while going up is 3 km/h. -/
theorem journey_speed_theorem (time_up : ℝ) (time_down : ℝ) (total_distance : ℝ) :
  time_up = 2 * time_down →
  time_up + time_down = 6 →
  total_distance / (time_up + time_down) = 4 →
  (total_distance / 2) / time_up = 3 :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_theorem_l2485_248591


namespace NUMINAMATH_CALUDE_oil_distribution_l2485_248563

theorem oil_distribution (total oil_A oil_B oil_C : ℕ) : 
  total = 3000 →
  oil_A = oil_B + 200 →
  oil_B = oil_C + 200 →
  total = oil_A + oil_B + oil_C →
  oil_B = 1000 := by
  sorry

end NUMINAMATH_CALUDE_oil_distribution_l2485_248563


namespace NUMINAMATH_CALUDE_certain_number_problem_l2485_248565

theorem certain_number_problem : 
  ∃ x : ℝ, (0.1 * x + 0.15 * 50 = 10.5) ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2485_248565


namespace NUMINAMATH_CALUDE_chess_competition_probabilities_l2485_248517

/-- Scoring system for the chess competition -/
structure ScoringSystem where
  win : Nat
  lose : Nat
  draw : Nat

/-- Probabilities for player A in a single game -/
structure PlayerProbabilities where
  win : Real
  lose : Real
  draw : Real

/-- Function to calculate the probability of player A scoring exactly 2 points in two games -/
def prob_A_scores_2 (s : ScoringSystem) (p : PlayerProbabilities) : Real :=
  sorry

/-- Function to calculate the probability of player B scoring at least 2 points in two games -/
def prob_B_scores_at_least_2 (s : ScoringSystem) (p : PlayerProbabilities) : Real :=
  sorry

theorem chess_competition_probabilities 
  (s : ScoringSystem) 
  (p : PlayerProbabilities) 
  (h1 : s.win = 2 ∧ s.lose = 0 ∧ s.draw = 1)
  (h2 : p.win = 0.5 ∧ p.lose = 0.3 ∧ p.draw = 0.2)
  (h3 : p.win + p.lose + p.draw = 1) :
  prob_A_scores_2 s p = 0.34 ∧ prob_B_scores_at_least_2 s p = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_chess_competition_probabilities_l2485_248517


namespace NUMINAMATH_CALUDE_f_is_perfect_square_l2485_248577

/-- The number of ordered pairs (a,b) of positive integers such that ab/(a+b) divides N -/
def f (N : ℕ+) : ℕ := sorry

/-- f(N) is always a perfect square -/
theorem f_is_perfect_square (N : ℕ+) : ∃ (k : ℕ), f N = k ^ 2 := by sorry

end NUMINAMATH_CALUDE_f_is_perfect_square_l2485_248577


namespace NUMINAMATH_CALUDE_line_relations_l2485_248540

-- Define the structure for a line
structure Line where
  slope : ℝ
  angle_of_inclination : ℝ

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem line_relations (l1 l2 : Line) (h_distinct : l1 ≠ l2) :
  (parallel l1 l2 → l1.slope = l2.slope) ∧
  (l1.slope = l2.slope → parallel l1 l2) ∧
  (parallel l1 l2 → l1.angle_of_inclination = l2.angle_of_inclination) ∧
  (l1.angle_of_inclination = l2.angle_of_inclination → parallel l1 l2) := by
  sorry

end NUMINAMATH_CALUDE_line_relations_l2485_248540


namespace NUMINAMATH_CALUDE_product_price_l2485_248575

/-- Given that m kilograms of a product costs 9 yuan, 
    prove that n kilograms of the same product costs (9n/m) yuan. -/
theorem product_price (m n : ℝ) (hm : m > 0) : 
  (9 : ℝ) / m * n = 9 * n / m := by sorry

end NUMINAMATH_CALUDE_product_price_l2485_248575


namespace NUMINAMATH_CALUDE_system_of_equations_l2485_248504

theorem system_of_equations (x y c d : ℝ) 
  (eq1 : 4 * x + 8 * y = c)
  (eq2 : 5 * x - 10 * y = d)
  (h_d_nonzero : d ≠ 0)
  (h_x_nonzero : x ≠ 0)
  (h_y_nonzero : y ≠ 0) :
  c / d = -4 / 5 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l2485_248504
