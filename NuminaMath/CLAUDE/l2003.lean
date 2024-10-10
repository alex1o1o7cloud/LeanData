import Mathlib

namespace cyclic_sum_inequality_l2003_200345

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≥ 
  Real.sqrt (3 / 2) * Real.sqrt (x + y + z) := by
  sorry

end cyclic_sum_inequality_l2003_200345


namespace bob_distance_from_start_l2003_200327

/-- Regular hexagon with side length 3 km -/
structure RegularHexagon where
  side_length : ℝ
  is_regular : side_length = 3

/-- Position after walking along the perimeter -/
def position_after_walk (h : RegularHexagon) (distance : ℝ) : ℝ × ℝ :=
  sorry

/-- Distance between two points -/
def distance_between_points (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem bob_distance_from_start (h : RegularHexagon) :
  let start_point := (0, 0)
  let end_point := position_after_walk h 7
  distance_between_points start_point end_point = 2 := by
  sorry

end bob_distance_from_start_l2003_200327


namespace grocer_banana_purchase_l2003_200370

/-- Calculates the number of pounds of bananas purchased by a grocer given the purchase price, selling price, and total profit. -/
theorem grocer_banana_purchase
  (purchase_price : ℚ)
  (purchase_quantity : ℚ)
  (selling_price : ℚ)
  (selling_quantity : ℚ)
  (total_profit : ℚ)
  (h1 : purchase_price / purchase_quantity = 0.50 / 3)
  (h2 : selling_price / selling_quantity = 1.00 / 4)
  (h3 : total_profit = 9.00) :
  ∃ (pounds : ℚ), pounds = 108 ∧ 
    pounds * (selling_price / selling_quantity - purchase_price / purchase_quantity) = total_profit :=
by sorry

end grocer_banana_purchase_l2003_200370


namespace complex_square_on_negative_y_axis_l2003_200375

theorem complex_square_on_negative_y_axis (a : ℝ) : 
  (∃ y : ℝ, y < 0 ∧ (a + Complex.I) ^ 2 = Complex.I * y) → a = -1 := by
  sorry

end complex_square_on_negative_y_axis_l2003_200375


namespace cow_profit_is_600_l2003_200301

/-- Calculates the profit from selling a cow given the purchase price, daily food cost,
    health care cost, number of days kept, and selling price. -/
def cowProfit (purchasePrice foodCostPerDay healthCareCost numDays sellingPrice : ℕ) : ℕ :=
  sellingPrice - (purchasePrice + foodCostPerDay * numDays + healthCareCost)

/-- Theorem stating that the profit from selling the cow under given conditions is $600. -/
theorem cow_profit_is_600 :
  cowProfit 600 20 500 40 2500 = 600 := by
  sorry

#eval cowProfit 600 20 500 40 2500

end cow_profit_is_600_l2003_200301


namespace intersection_distance_sum_l2003_200378

noncomputable section

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 1)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    A ≠ B ∧
    distance point_P A + distance point_P B = 8 * Real.sqrt 2 / 5 :=
sorry

end

end intersection_distance_sum_l2003_200378


namespace opposite_of_negative_two_l2003_200353

theorem opposite_of_negative_two : -((-2 : ℤ)) = 2 := by sorry

end opposite_of_negative_two_l2003_200353


namespace largest_multiple_six_negation_greater_than_neg_150_l2003_200340

theorem largest_multiple_six_negation_greater_than_neg_150 :
  (∀ n : ℤ, n % 6 = 0 ∧ -n > -150 → n ≤ 144) ∧
  144 % 6 = 0 ∧ -144 > -150 :=
by sorry

end largest_multiple_six_negation_greater_than_neg_150_l2003_200340


namespace inequality_proof_l2003_200313

theorem inequality_proof (x : ℝ) (hx : x > 0) : x^2 + 1/(4*x) ≥ 3/4 ∧ Real.sqrt 3 - 1 < 3/4 := by
  sorry

end inequality_proof_l2003_200313


namespace workshop_output_comparison_l2003_200315

/-- Represents the monthly increase factor for a workshop -/
structure WorkshopGrowth where
  fixed_amount : ℝ
  percentage : ℝ

/-- Theorem statement for workshop output comparison -/
theorem workshop_output_comparison 
  (growth_A growth_B : WorkshopGrowth)
  (h_initial_equal : growth_A.fixed_amount = growth_B.percentage) -- Initial outputs are equal
  (h_equal_after_7 : 1 + 6 * growth_A.fixed_amount = (1 + growth_B.percentage) ^ 6) -- Equal after 7 months
  : 1 + 3 * growth_A.fixed_amount > (1 + growth_B.percentage) ^ 3 := by
  sorry

end workshop_output_comparison_l2003_200315


namespace even_multiples_sum_difference_l2003_200392

theorem even_multiples_sum_difference : 
  let n : ℕ := 2025
  let even_sum : ℕ := n * (2 + 2 * n)
  let multiples_of_three_sum : ℕ := n * (3 + 3 * n)
  (even_sum : ℤ) - (multiples_of_three_sum : ℤ) = -2052155 := by
  sorry

end even_multiples_sum_difference_l2003_200392


namespace planes_parallel_from_skew_lines_l2003_200302

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (skew : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_from_skew_lines 
  (α β : Plane) (l m : Line) :
  skew l m →
  subset l α →
  parallel l β →
  subset m β →
  parallel m α →
  plane_parallel α β :=
sorry

end planes_parallel_from_skew_lines_l2003_200302


namespace geometric_sequence_problem_l2003_200303

/-- A geometric sequence with common ratio greater than 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : q > 1
  h_geometric : ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem to be proved -/
theorem geometric_sequence_problem (seq : GeometricSequence)
    (h1 : seq.a 3 * seq.a 7 = 72)
    (h2 : seq.a 2 + seq.a 8 = 27) :
  seq.a 12 = 96 := by
  sorry

end geometric_sequence_problem_l2003_200303


namespace inequality_solution_l2003_200308

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := (a * x) / (x - 1) < 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | x < 1 ∨ x > 3}

-- Theorem statement
theorem inequality_solution (a : ℝ) :
  (∀ x, x ∈ solution_set a ↔ inequality a x) → a = 2/3 := by
  sorry


end inequality_solution_l2003_200308


namespace concentric_circles_diameter_l2003_200398

theorem concentric_circles_diameter (r₁ r₂ : ℝ) : 
  r₁ > 0 → r₂ > r₁ → 
  π * r₁^2 = 4 * π → 
  π * r₂^2 - π * r₁^2 = 4 * π → 
  2 * r₂ = 4 * Real.sqrt 2 :=
by sorry

end concentric_circles_diameter_l2003_200398


namespace decimal_to_fraction_sum_l2003_200397

theorem decimal_to_fraction_sum (a b : ℕ+) :
  (a : ℚ) / b = 0.31375 ∧ 
  Int.gcd a.val b.val = 1 →
  a.val + b.val = 1051 := by
  sorry

end decimal_to_fraction_sum_l2003_200397


namespace parabola_hyperbola_disjunction_l2003_200320

-- Define the propositions
def p : Prop := ∀ y : ℝ, (∃ x : ℝ, x = 4 * y^2) → (∃ x : ℝ, x = 1)

def q : Prop := ∃ x y : ℝ, (x^2 / 4 - y^2 / 5 = -1) ∧ (x = 0 ∧ y = 3)

-- Theorem to prove
theorem parabola_hyperbola_disjunction : p ∨ q := by sorry

end parabola_hyperbola_disjunction_l2003_200320


namespace max_value_trig_expression_l2003_200362

theorem max_value_trig_expression :
  ∀ θ φ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → 0 ≤ φ ∧ φ ≤ π/2 →
  3 * Real.sin θ * Real.cos φ + 2 * (Real.sin φ)^2 ≤ 5 :=
by sorry

end max_value_trig_expression_l2003_200362


namespace ascetics_equal_distance_l2003_200372

theorem ascetics_equal_distance (h m : ℝ) (h_pos : h > 0) (m_pos : m > 0) :
  ∃ x : ℝ, x > 0 ∧ 
  (x + (((x + h)^2 + (m * h)^2).sqrt) = h + m * h) ∧
  x = (h * m) / (m + 2) := by
sorry

end ascetics_equal_distance_l2003_200372


namespace checkerboard_coverage_three_by_five_uncoverable_l2003_200309

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- A domino covers exactly two squares -/
def domino_size : ℕ := 2

/-- The total number of squares on a checkerboard -/
def total_squares (board : Checkerboard) : ℕ :=
  board.rows * board.cols

/-- A checkerboard can be covered by dominoes if its total squares is even -/
def can_be_covered_by_dominoes (board : Checkerboard) : Prop :=
  total_squares board % domino_size = 0

/-- Theorem: A checkerboard can be covered by dominoes iff its total squares is even -/
theorem checkerboard_coverage (board : Checkerboard) :
  can_be_covered_by_dominoes board ↔ Even (total_squares board) := by sorry

/-- The 3x5 checkerboard cannot be covered by dominoes -/
theorem three_by_five_uncoverable :
  ¬ can_be_covered_by_dominoes ⟨3, 5⟩ := by sorry

end checkerboard_coverage_three_by_five_uncoverable_l2003_200309


namespace intersection_implies_sum_of_translations_l2003_200326

/-- Given two functions f and g that intersect at points (1,7) and (9,1),
    prove that the sum of their x-axis translation parameters is 10 -/
theorem intersection_implies_sum_of_translations (a b c d : ℝ) :
  (∀ x, -2 * |x - a| + b = 2 * |x - c| + d ↔ (x = 1 ∧ -2 * |x - a| + b = 7) ∨ (x = 9 ∧ -2 * |x - a| + b = 1)) →
  a + c = 10 := by
sorry

end intersection_implies_sum_of_translations_l2003_200326


namespace janeth_round_balloon_bags_l2003_200351

/-- The number of balloons in each bag of round balloons -/
def round_balloons_per_bag : ℕ := 20

/-- The number of bags of long balloons bought -/
def long_balloon_bags : ℕ := 4

/-- The number of balloons in each bag of long balloons -/
def long_balloons_per_bag : ℕ := 30

/-- The number of round balloons that burst -/
def burst_balloons : ℕ := 5

/-- The total number of balloons left -/
def total_balloons_left : ℕ := 215

/-- Theorem stating that Janeth bought 5 bags of round balloons -/
theorem janeth_round_balloon_bags : ℕ := by
  -- The proof goes here
  sorry

end janeth_round_balloon_bags_l2003_200351


namespace quadratic_radical_for_all_real_l2003_200399

theorem quadratic_radical_for_all_real (a : ℝ) : ∃ (x : ℝ), x ^ 2 = a ^ 2 + 1 :=
sorry

end quadratic_radical_for_all_real_l2003_200399


namespace sum_of_digits_of_a_l2003_200382

def a : ℕ := (10^10) - 47

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_a : sum_of_digits a = 81 := by sorry

end sum_of_digits_of_a_l2003_200382


namespace dawn_at_6am_l2003_200343

/-- Represents the time of dawn in hours before noon -/
def dawn_time : ℝ := 6

/-- Represents the time (in hours after noon) when the first pedestrian arrives at B -/
def arrival_time_B : ℝ := 4

/-- Represents the time (in hours after noon) when the second pedestrian arrives at A -/
def arrival_time_A : ℝ := 9

/-- The theorem states that given the conditions of the problem, dawn occurred at 6 AM -/
theorem dawn_at_6am :
  dawn_time * arrival_time_B = arrival_time_A * dawn_time ∧
  dawn_time + arrival_time_B + dawn_time + arrival_time_A = 24 →
  dawn_time = 6 := by
  sorry

end dawn_at_6am_l2003_200343


namespace tangent_line_sum_l2003_200381

/-- Given a function f: ℝ → ℝ with a tangent line at x = 2 
    described by the equation 2x + y - 3 = 0,
    prove that f(2) + f'(2) = -3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : ∀ x y, y = f x → 2 * x + y - 3 = 0 → x = 2) :
  f 2 + deriv f 2 = -3 := by
  sorry

end tangent_line_sum_l2003_200381


namespace soap_brand_usage_l2003_200341

/-- The number of households using both brand R and brand B soap -/
def households_using_both : ℕ := 15

/-- The total number of households surveyed -/
def total_households : ℕ := 200

/-- The number of households using neither brand R nor brand B -/
def households_using_neither : ℕ := 80

/-- The number of households using only brand R -/
def households_using_only_R : ℕ := 60

/-- For every household using both brands, this many use only brand B -/
def ratio_B_to_both : ℕ := 3

theorem soap_brand_usage :
  households_using_both * (ratio_B_to_both + 1) + 
  households_using_neither + 
  households_using_only_R = 
  total_households := by sorry

end soap_brand_usage_l2003_200341


namespace order_of_values_l2003_200383

theorem order_of_values : 
  let a := Real.sin (60 * π / 180)
  let b := Real.sqrt (5 / 9)
  let c := π / 2014
  a > b ∧ b > c := by sorry

end order_of_values_l2003_200383


namespace unique_zero_implies_a_gt_one_l2003_200373

/-- A function f(x) = 2ax^2 - x - 1 has only one zero in the interval (0, 1) -/
def has_unique_zero_in_interval (a : ℝ) : Prop :=
  ∃! x : ℝ, 0 < x ∧ x < 1 ∧ 2 * a * x^2 - x - 1 = 0

/-- If f(x) = 2ax^2 - x - 1 has only one zero in the interval (0, 1), then a > 1 -/
theorem unique_zero_implies_a_gt_one :
  ∀ a : ℝ, has_unique_zero_in_interval a → a > 1 :=
by sorry

end unique_zero_implies_a_gt_one_l2003_200373


namespace playground_fence_posts_l2003_200318

/-- Calculates the minimum number of fence posts needed for a rectangular playground. -/
def min_fence_posts (length width post_spacing : ℕ) : ℕ :=
  let long_side := max length width
  let short_side := min length width
  let long_side_posts := long_side / post_spacing + 1
  let short_side_posts := short_side / post_spacing + 1
  long_side_posts + 2 * (short_side_posts - 1)

/-- Theorem stating the minimum number of fence posts needed for the given playground. -/
theorem playground_fence_posts :
  min_fence_posts 100 50 10 = 21 :=
by
  sorry

#eval min_fence_posts 100 50 10

end playground_fence_posts_l2003_200318


namespace intersection_complement_equality_l2003_200344

def U : Set Int := Set.univ

def A : Set Int := {-1, 1, 3, 5, 7, 9}

def B : Set Int := {-1, 5, 7}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {1, 3, 9} := by sorry

end intersection_complement_equality_l2003_200344


namespace raft_sticks_total_l2003_200307

/-- The number of sticks needed for Simon's raft -/
def simon_sticks : ℕ := 36

/-- The number of sticks needed for Gerry's raft -/
def gerry_sticks : ℕ := (2 * simon_sticks) / 3

/-- The number of sticks needed for Micky's raft -/
def micky_sticks : ℕ := simon_sticks + gerry_sticks + 9

/-- The total number of sticks needed for all three rafts -/
def total_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks

theorem raft_sticks_total : total_sticks = 129 := by
  sorry

end raft_sticks_total_l2003_200307


namespace tank_fill_time_with_leak_l2003_200305

def pump_rate : ℚ := 1 / 6
def leak_rate : ℚ := 1 / 12

theorem tank_fill_time_with_leak :
  let net_fill_rate := pump_rate - leak_rate
  (1 : ℚ) / net_fill_rate = 12 := by sorry

end tank_fill_time_with_leak_l2003_200305


namespace parallel_vectors_m_value_l2003_200393

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (1, 2) and b = (1-m, 2m-4), then m = 3/2 -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![1 - m, 2 * m - 4]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  m = 3/2 := by
  sorry

end parallel_vectors_m_value_l2003_200393


namespace cylinder_volume_change_l2003_200365

/-- Theorem: Tripling the radius and doubling the height of a cylinder increases its volume by a factor of 18. -/
theorem cylinder_volume_change (r h V : ℝ) (hV : V = π * r^2 * h) :
  π * (3*r)^2 * (2*h) = 18 * V := by sorry

end cylinder_volume_change_l2003_200365


namespace gumball_probability_l2003_200367

theorem gumball_probability (blue_prob : ℚ) : 
  blue_prob ^ 2 = 16 / 49 → (1 : ℚ) - blue_prob = 3 / 7 := by
  sorry

end gumball_probability_l2003_200367


namespace thomson_savings_l2003_200394

/-- Calculates the amount saved by Mrs. Thomson given her spending pattern -/
def amount_saved (X : ℝ) : ℝ :=
  let after_food := X * (1 - 0.375)
  let after_clothes := after_food * (1 - 0.22)
  let after_household := after_clothes * (1 - 0.15)
  let after_stocks := after_household * (1 - 0.30)
  let after_tuition := after_stocks * (1 - 0.40)
  after_tuition

theorem thomson_savings (X : ℝ) : amount_saved X = 0.1740375 * X := by
  sorry

end thomson_savings_l2003_200394


namespace cos_negative_nineteen_pi_sixths_l2003_200395

theorem cos_negative_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_negative_nineteen_pi_sixths_l2003_200395


namespace factor_expression_l2003_200377

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end factor_expression_l2003_200377


namespace f_lower_bound_f_condition_equivalent_l2003_200388

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x + 2*a + 3|

-- Theorem 1: For all real x and a, f(x) ≥ 2
theorem f_lower_bound (x a : ℝ) : f x a ≥ 2 := by sorry

-- Theorem 2: f(-3/2) < 3 is equivalent to -1 < a < 0
theorem f_condition_equivalent (a : ℝ) : 
  (f (-3/2) a < 3) ↔ (-1 < a ∧ a < 0) := by sorry

end f_lower_bound_f_condition_equivalent_l2003_200388


namespace inequality_proof_l2003_200331

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + x^2 + z^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end inequality_proof_l2003_200331


namespace lottery_theorem_l2003_200304

-- Define the lottery parameters
def total_numbers : ℕ := 90
def numbers_drawn : ℕ := 5
def numbers_played : ℕ := 7
def group_size : ℕ := 10

-- Define the ticket prices and payouts
def ticket_cost : ℕ := 60
def payout_three_match : ℕ := 7000
def payout_two_match : ℕ := 300

-- Define the probability of drawing 3 out of 7 specific numbers
def probability_three_match : ℚ := 119105 / 43949268

-- Define the profit per person
def profit_per_person : ℕ := 4434

-- Theorem statement
theorem lottery_theorem :
  (probability_three_match = 119105 / 43949268) ∧
  (profit_per_person = 4434) := by
  sorry


end lottery_theorem_l2003_200304


namespace complex_equation_sum_l2003_200368

theorem complex_equation_sum (a b : ℝ) : 
  (a - Complex.I = 2 + b * Complex.I) → (a + b = 1) := by
  sorry

end complex_equation_sum_l2003_200368


namespace binomial_18_choose_7_l2003_200360

theorem binomial_18_choose_7 : Nat.choose 18 7 = 31824 := by
  sorry

end binomial_18_choose_7_l2003_200360


namespace intersection_M_N_l2003_200354

-- Define the sets M and N
def M : Set ℝ := Set.univ
def N : Set ℝ := {x | 2 * x - x^2 > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by
  sorry

end intersection_M_N_l2003_200354


namespace solution_set_abs_x_times_one_minus_two_x_l2003_200371

theorem solution_set_abs_x_times_one_minus_two_x (x : ℝ) :
  (|x| * (1 - 2*x) > 0) ↔ (x < 0 ∨ (x > 0 ∧ x < 1/2)) := by sorry

end solution_set_abs_x_times_one_minus_two_x_l2003_200371


namespace polynomial_roots_imply_composite_sum_of_squares_l2003_200335

/-- A polynomial with integer coefficients -/
def IntPolynomial (p q : ℤ) : ℝ → ℝ := fun x ↦ x^2 + p*x + q + 1

/-- Definition of a composite number -/
def IsComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

theorem polynomial_roots_imply_composite_sum_of_squares (p q : ℤ) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ (IntPolynomial p q a = 0) ∧ (IntPolynomial p q b = 0)) →
  IsComposite (Int.natAbs (p^2 + q^2)) :=
sorry

end polynomial_roots_imply_composite_sum_of_squares_l2003_200335


namespace smallest_sum_of_reciprocals_l2003_200336

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ a + b ≤ x + y ∧ a + b = 64 :=
sorry

end smallest_sum_of_reciprocals_l2003_200336


namespace current_year_is_2021_l2003_200349

/-- The year Aziz's parents moved to America -/
def parents_move_year : ℕ := 1982

/-- Aziz's current age -/
def aziz_age : ℕ := 36

/-- Years Aziz's parents lived in America before his birth -/
def years_before_birth : ℕ := 3

/-- The current year -/
def current_year : ℕ := parents_move_year + aziz_age + years_before_birth

theorem current_year_is_2021 : current_year = 2021 := by
  sorry

end current_year_is_2021_l2003_200349


namespace cone_to_cylinder_volume_ratio_l2003_200321

/-- 
Given a cylinder and a cone with the same radius, where the cone's height is one-third of the cylinder's height,
prove that the ratio of the cone's volume to the cylinder's volume is 1/9.
-/
theorem cone_to_cylinder_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry

end cone_to_cylinder_volume_ratio_l2003_200321


namespace cube_displacement_l2003_200357

/-- The volume of water displaced by a cube in a cylindrical barrel -/
theorem cube_displacement (cube_side : ℝ) (barrel_radius barrel_height : ℝ) 
  (h_cube_side : cube_side = 6)
  (h_barrel_radius : barrel_radius = 5)
  (h_barrel_height : barrel_height = 12)
  (h_fully_submerged : cube_side ≤ barrel_height) :
  cube_side ^ 3 = 216 := by
  sorry

end cube_displacement_l2003_200357


namespace sample_correlation_coefficient_range_l2003_200350

/-- The sample correlation coefficient -/
def sample_correlation_coefficient (X Y : List ℝ) : ℝ := sorry

/-- Theorem: The sample correlation coefficient is in the closed interval [-1, 1] -/
theorem sample_correlation_coefficient_range 
  (X Y : List ℝ) : 
  ∃ r : ℝ, sample_correlation_coefficient X Y = r ∧ r ∈ Set.Icc (-1 : ℝ) 1 :=
sorry

end sample_correlation_coefficient_range_l2003_200350


namespace second_train_length_correct_l2003_200306

/-- Calculates the length of a train given the length of another train, their speeds, and the time they take to cross each other when moving in opposite directions. -/
def calculate_train_length (length_train1 : ℝ) (speed_train1 : ℝ) (speed_train2 : ℝ) (time_to_cross : ℝ) : ℝ :=
  let relative_speed := speed_train1 + speed_train2
  let total_distance := relative_speed * time_to_cross
  total_distance - length_train1

/-- Theorem stating that the calculated length of the second train is correct given the problem conditions. -/
theorem second_train_length_correct (length_train1 : ℝ) (speed_train1 : ℝ) (speed_train2 : ℝ) (time_to_cross : ℝ)
  (h1 : length_train1 = 110)
  (h2 : speed_train1 = 60 * 1000 / 3600)
  (h3 : speed_train2 = 40 * 1000 / 3600)
  (h4 : time_to_cross = 9.719222462203025) :
  let length_train2 := calculate_train_length length_train1 speed_train1 speed_train2 time_to_cross
  ∃ ε > 0, |length_train2 - 159.98| < ε :=
sorry

end second_train_length_correct_l2003_200306


namespace original_number_is_five_l2003_200348

theorem original_number_is_five : 
  ∃ x : ℚ, 3 * (2 * x + 9) = 57 ∧ x = 5 := by
  sorry

end original_number_is_five_l2003_200348


namespace max_q_minus_r_for_1057_l2003_200317

theorem max_q_minus_r_for_1057 :
  ∃ (q r : ℕ+), 1057 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 1057 = 23 * q' + r' → q' - r' ≤ q - r ∧ q - r = 23 := by
sorry

end max_q_minus_r_for_1057_l2003_200317


namespace system_solution_l2003_200369

theorem system_solution : ∃ (x y : ℝ), 
  (x = 4 + 2 * Real.sqrt 3 ∧ y = 12 + 6 * Real.sqrt 3) ∧
  (1 - 12 / (3 * x + y) = 2 / Real.sqrt x) ∧
  (1 + 12 / (3 * x + y) = 6 / Real.sqrt y) := by
  sorry

end system_solution_l2003_200369


namespace direct_proportion_through_point_l2003_200359

/-- A direct proportion function passing through (2, -1) -/
def f (x : ℝ) : ℝ := sorry

/-- The point (2, -1) lies on the graph of f -/
axiom point_on_graph : f 2 = -1

/-- f is a direct proportion function -/
axiom direct_proportion (x : ℝ) : ∃ k : ℝ, f x = k * x

theorem direct_proportion_through_point :
  ∀ x : ℝ, f x = -1/2 * x := by sorry

end direct_proportion_through_point_l2003_200359


namespace change_received_l2003_200352

def skirt_price : ℕ := 13
def skirt_count : ℕ := 2
def blouse_price : ℕ := 6
def blouse_count : ℕ := 3
def amount_paid : ℕ := 100

def total_cost : ℕ := skirt_price * skirt_count + blouse_price * blouse_count

theorem change_received : amount_paid - total_cost = 56 := by
  sorry

end change_received_l2003_200352


namespace worker_count_proof_l2003_200337

/-- The number of workers who raised money by equal contribution -/
def number_of_workers : ℕ := 1200

/-- The total contribution in rupees -/
def total_contribution : ℕ := 300000

/-- The increased total contribution if each worker contributed 50 rupees extra -/
def increased_contribution : ℕ := 360000

/-- The extra amount each worker would contribute in the increased scenario -/
def extra_contribution : ℕ := 50

theorem worker_count_proof :
  (number_of_workers * (total_contribution / number_of_workers) = total_contribution) ∧
  (number_of_workers * (total_contribution / number_of_workers + extra_contribution) = increased_contribution) :=
sorry

end worker_count_proof_l2003_200337


namespace complement_of_M_l2003_200346

-- Define the universal set U
def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}

-- Define the set M
def M : Set ℝ := {1}

-- Theorem statement
theorem complement_of_M (x : ℝ) : x ∈ (U \ M) ↔ 1 < x ∧ x ≤ 5 := by sorry

end complement_of_M_l2003_200346


namespace arkansas_game_profit_calculation_l2003_200332

/-- The amount of money made per t-shirt in dollars -/
def profit_per_shirt : ℕ := 98

/-- The total number of t-shirts sold during both games -/
def total_shirts_sold : ℕ := 163

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts_sold : ℕ := 89

/-- The money made from selling t-shirts during the Arkansas game -/
def arkansas_game_profit : ℕ := arkansas_shirts_sold * profit_per_shirt

theorem arkansas_game_profit_calculation :
  arkansas_game_profit = 8722 :=
sorry

end arkansas_game_profit_calculation_l2003_200332


namespace M_inequality_l2003_200329

/-- The number of h-subsets with property P_k(X) in a set X of size n -/
def M (n k h : ℕ) : ℕ := sorry

/-- Theorem stating the inequality for M(n,k,h) -/
theorem M_inequality (n k h : ℕ) :
  (n.choose h) / (k.choose h) ≤ M n k h ∧ M n k h ≤ (n - k + h).choose h :=
sorry

end M_inequality_l2003_200329


namespace function_composition_equality_l2003_200342

theorem function_composition_equality (m n p q c : ℝ) :
  let f := fun (x : ℝ) => m * x + n + c
  let g := fun (x : ℝ) => p * x + q + c
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p) - q * (1 - m) + c * (m - p) = 0 := by
  sorry

end function_composition_equality_l2003_200342


namespace line_difference_l2003_200391

theorem line_difference (line_length : ℝ) (h : line_length = 80) :
  (0.75 - 0.4) * line_length = 28 :=
by sorry

end line_difference_l2003_200391


namespace twice_a_plus_one_nonnegative_l2003_200300

theorem twice_a_plus_one_nonnegative (a : ℝ) : (2 * a + 1 ≥ 0) ↔ (∀ x : ℝ, x = 2 * a + 1 → x ≥ 0) := by sorry

end twice_a_plus_one_nonnegative_l2003_200300


namespace min_reciprocal_sum_l2003_200328

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : Real.log (x + y) = 0) :
  (1 / x + 1 / y) ≥ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Real.log (a + b) = 0 ∧ 1 / a + 1 / b = 4 := by
  sorry

end min_reciprocal_sum_l2003_200328


namespace octagon_area_theorem_l2003_200358

/-- The area of a regular octagon inscribed in a square with perimeter 144 cm,
    where each side of the square is trisected by the vertices of the octagon. -/
def inscribedOctagonArea : ℝ := 1008

/-- The perimeter of the square. -/
def squarePerimeter : ℝ := 144

/-- A side of the square is trisected by the vertices of the octagon. -/
def isTrisected (s : ℝ) : Prop := ∃ p : ℝ, s = 3 * p

theorem octagon_area_theorem (s : ℝ) (h1 : s * 4 = squarePerimeter) (h2 : isTrisected s) :
  inscribedOctagonArea = s^2 - 4 * (s/3)^2 :=
sorry

end octagon_area_theorem_l2003_200358


namespace det_A_eq_22_l2003_200333

def A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -5; -4, 6]

theorem det_A_eq_22 : A.det = 22 := by
  sorry

end det_A_eq_22_l2003_200333


namespace negation_of_implication_l2003_200376

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2 → x > 1) ↔ (x ≤ 2 → x ≤ 1) := by sorry

end negation_of_implication_l2003_200376


namespace f_even_and_increasing_l2003_200324

def f (x : ℝ) := x^2 + 1

theorem f_even_and_increasing : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
sorry

end f_even_and_increasing_l2003_200324


namespace mask_digit_correct_l2003_200363

/-- Represents the four masks in the problem -/
inductive Mask
| elephant
| mouse
| pig
| panda

/-- Associates each mask with a digit -/
def mask_digit : Mask → Nat
| Mask.elephant => 6
| Mask.mouse => 4
| Mask.pig => 8
| Mask.panda => 1

/-- The theorem to be proved -/
theorem mask_digit_correct :
  (mask_digit Mask.elephant) * (mask_digit Mask.elephant) = 36 ∧
  (mask_digit Mask.mouse) * (mask_digit Mask.mouse) = 16 ∧
  (mask_digit Mask.pig) * (mask_digit Mask.pig) = 64 ∧
  (mask_digit Mask.panda) * (mask_digit Mask.panda) = 1 ∧
  (∀ m1 m2 : Mask, m1 ≠ m2 → mask_digit m1 ≠ mask_digit m2) :=
by sorry

#check mask_digit_correct

end mask_digit_correct_l2003_200363


namespace work_completion_men_count_l2003_200310

/-- Given a work that can be completed by M men in 20 days, 
    or by (M - 4) men in 25 days, prove that M = 16. -/
theorem work_completion_men_count :
  ∀ (M : ℕ) (W : ℝ),
  (M : ℝ) * (W / 20) = (M - 4 : ℝ) * (W / 25) →
  M = 16 :=
by sorry

end work_completion_men_count_l2003_200310


namespace four_integer_solutions_l2003_200374

def satisfies_equation (a : ℤ) : Prop :=
  |2 * a + 7| + |2 * a - 1| = 8

theorem four_integer_solutions :
  ∃ (S : Finset ℤ), (∀ a ∈ S, satisfies_equation a) ∧ 
                    (∀ a : ℤ, satisfies_equation a → a ∈ S) ∧
                    Finset.card S = 4 :=
sorry

end four_integer_solutions_l2003_200374


namespace percentage_difference_l2003_200325

theorem percentage_difference : (0.8 * 40) - ((4 / 5) * 15) = 20 := by
  sorry

end percentage_difference_l2003_200325


namespace cell_phone_bill_is_45_l2003_200389

/-- Calculates the total cell phone bill based on given parameters --/
def calculate_bill (fixed_charge : ℚ) (daytime_rate : ℚ) (evening_rate : ℚ) 
                   (free_evening_minutes : ℕ) (daytime_minutes : ℕ) (evening_minutes : ℕ) : ℚ :=
  let daytime_cost := daytime_rate * daytime_minutes
  let chargeable_evening_minutes := max (evening_minutes - free_evening_minutes) 0
  let evening_cost := evening_rate * chargeable_evening_minutes
  fixed_charge + daytime_cost + evening_cost

/-- Theorem stating that the cell phone bill is $45 given the specified conditions --/
theorem cell_phone_bill_is_45 :
  calculate_bill 20 0.1 0.05 200 200 300 = 45 := by
  sorry


end cell_phone_bill_is_45_l2003_200389


namespace fifteen_tomorrow_l2003_200323

/-- Represents the fishing schedule in a coastal village -/
structure FishingSchedule where
  daily : ℕ           -- Number of people fishing daily
  everyOther : ℕ      -- Number of people fishing every other day
  everyThree : ℕ      -- Number of people fishing every three days
  yesterday : ℕ       -- Number of people who fished yesterday
  today : ℕ           -- Number of people fishing today

/-- Calculates the number of people who will fish tomorrow given a fishing schedule -/
def tomorrowsFishers (schedule : FishingSchedule) : ℕ :=
  sorry

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_tomorrow (schedule : FishingSchedule) 
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOther = 8)
  (h3 : schedule.everyThree = 3)
  (h4 : schedule.yesterday = 12)
  (h5 : schedule.today = 10) :
  tomorrowsFishers schedule = 15 := by
  sorry

end fifteen_tomorrow_l2003_200323


namespace no_fixed_point_implies_no_double_fixed_point_no_intersection_implies_no_double_intersection_l2003_200355

-- Part (a)
theorem no_fixed_point_implies_no_double_fixed_point
  (f : ℝ → ℝ) (hf : Continuous f) (h : ∀ x, f x ≠ x) :
  ∀ x, f (f x) ≠ x :=
sorry

-- Part (b)
theorem no_intersection_implies_no_double_intersection
  (f g : ℝ → ℝ) (hf : Continuous f) (hg : Continuous g)
  (h_comm : ∀ x, f (g x) = g (f x)) (h_neq : ∀ x, f x ≠ g x) :
  ∀ x, f (f x) ≠ g (g x) :=
sorry

end no_fixed_point_implies_no_double_fixed_point_no_intersection_implies_no_double_intersection_l2003_200355


namespace x_value_proof_l2003_200385

theorem x_value_proof (x : ℝ) (h : 3*x - 4*x + 7*x = 180) : x = 30 := by
  sorry

end x_value_proof_l2003_200385


namespace smallest_integer_above_sqrt3_plus_sqrt2_to_8th_l2003_200330

theorem smallest_integer_above_sqrt3_plus_sqrt2_to_8th (x : ℝ) : 
  x = (Real.sqrt 3 + Real.sqrt 2)^8 → 
  ∀ n : ℤ, (n : ℝ) > x → n ≥ 5360 ∧ 5360 > x :=
by sorry

end smallest_integer_above_sqrt3_plus_sqrt2_to_8th_l2003_200330


namespace angle_between_perpendicular_lines_to_dihedral_angle_l2003_200387

-- Define the dihedral angle
def dihedral_angle (α l β : Plane) : ℝ := sorry

-- Define perpendicularity between a line and a plane
def perpendicular (m : Line) (α : Plane) : Prop := sorry

-- Define the angle between two lines
def angle_between_lines (m n : Line) : ℝ := sorry

-- Define skew lines
def skew_lines (m n : Line) : Prop := sorry

theorem angle_between_perpendicular_lines_to_dihedral_angle 
  (α l β : Plane) (m n : Line) :
  dihedral_angle α l β = 60 ∧ 
  skew_lines m n ∧
  perpendicular m α ∧
  perpendicular n β →
  angle_between_lines m n = 60 := by sorry

end angle_between_perpendicular_lines_to_dihedral_angle_l2003_200387


namespace sum_of_squares_l2003_200316

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 50) : x^2 + y^2 = 44 := by
  sorry

end sum_of_squares_l2003_200316


namespace tripled_base_doubled_exponent_l2003_200334

theorem tripled_base_doubled_exponent 
  (a b x : ℝ) 
  (hb : b ≠ 0) 
  (hr : (3 * a) ^ (2 * b) = a ^ b * x ^ b) : 
  x = 9 * a := by sorry

end tripled_base_doubled_exponent_l2003_200334


namespace trailing_zeroes_500_factorial_l2003_200314

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial :
  trailingZeroes 500 = 124 := by
  sorry

end trailing_zeroes_500_factorial_l2003_200314


namespace cricket_team_age_difference_l2003_200312

/-- Represents a cricket team with its age-related properties -/
structure CricketTeam where
  total_members : ℕ
  team_average_age : ℝ
  wicket_keeper_age_difference : ℝ
  remaining_players_average_age : ℝ

/-- Theorem stating the difference between the team's average age and the remaining players' average age -/
theorem cricket_team_age_difference (team : CricketTeam)
  (h1 : team.total_members = 11)
  (h2 : team.team_average_age = 28)
  (h3 : team.wicket_keeper_age_difference = 3)
  (h4 : team.remaining_players_average_age = 25) :
  team.team_average_age - team.remaining_players_average_age = 3 := by
  sorry

end cricket_team_age_difference_l2003_200312


namespace friend_fruit_consumption_l2003_200366

/-- Given three friends who ate a total of 128 ounces of fruit, 
    where one friend ate 8 ounces and another ate 96 ounces,
    prove that the third friend ate 24 ounces. -/
theorem friend_fruit_consumption 
  (total : ℕ) 
  (friend1 : ℕ) 
  (friend2 : ℕ) 
  (h1 : total = 128)
  (h2 : friend1 = 8)
  (h3 : friend2 = 96) :
  total - friend1 - friend2 = 24 :=
by sorry

end friend_fruit_consumption_l2003_200366


namespace total_amount_is_152_l2003_200396

/-- Represents the share distribution among five individuals -/
structure ShareDistribution where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  t : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (sd : ShareDistribution) : Prop :=
  sd.q = 35 ∧
  sd.q / sd.p = 1.75 / 2 ∧
  sd.r / sd.p = 1.5 / 2 ∧
  sd.s / sd.p = 1.25 / 2 ∧
  sd.t / sd.p = 1.1 / 2

/-- The theorem stating that the total amount is $152 -/
theorem total_amount_is_152 (sd : ShareDistribution) 
  (h : satisfies_conditions sd) : 
  sd.p + sd.q + sd.r + sd.s + sd.t = 152 := by
  sorry


end total_amount_is_152_l2003_200396


namespace seashells_to_find_l2003_200322

def current_seashells : ℕ := 19
def target_seashells : ℕ := 25

theorem seashells_to_find : target_seashells - current_seashells = 6 := by
  sorry

end seashells_to_find_l2003_200322


namespace mirabel_candy_distribution_l2003_200379

theorem mirabel_candy_distribution :
  ∃ (k : ℕ), k = 2 ∧ 
  (∀ (j : ℕ), j < k → ¬∃ (n : ℕ), 10 ≤ n ∧ n < 20 ∧ (47 - j) % n = 0) ∧
  (∃ (n : ℕ), 10 ≤ n ∧ n < 20 ∧ (47 - k) % n = 0) :=
by sorry

end mirabel_candy_distribution_l2003_200379


namespace cos_equality_for_n_l2003_200339

theorem cos_equality_for_n (n : ℤ) : ∃ n, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (259 * π / 180) ∧ n = 101 := by
  sorry

end cos_equality_for_n_l2003_200339


namespace circle_square_area_difference_l2003_200356

/-- The difference between the area of a circle with diameter 10 inches
    and the area of a square with diagonal 10 inches is approximately 28.5 square inches. -/
theorem circle_square_area_difference :
  let square_diagonal : ℝ := 10
  let circle_diameter : ℝ := 10
  let square_area : ℝ := (square_diagonal ^ 2) / 2
  let circle_area : ℝ := π * ((circle_diameter / 2) ^ 2)
  ∃ ε > 0, ε < 0.1 ∧ |circle_area - square_area - 28.5| < ε :=
by
  sorry


end circle_square_area_difference_l2003_200356


namespace triangle_cosine_theorem_l2003_200311

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, prove that when 6S = a²sin A + b²sin B and (a+b)/c is maximized,
    cos C = 7/9 -/
theorem triangle_cosine_theorem (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h3 : A + B + C = π)
  (h4 : S = (1/2) * a * b * Real.sin C)
  (h5 : 6 * S = a^2 * Real.sin A + b^2 * Real.sin B)
  (h6 : ∀ (x y z : ℝ), (x + y) / z ≤ (a + b) / c) :
  Real.cos C = 7/9 :=
sorry

end triangle_cosine_theorem_l2003_200311


namespace min_value_of_reciprocal_sum_l2003_200380

theorem min_value_of_reciprocal_sum (t q a b : ℝ) : 
  (∀ x, x^2 - t*x + q = 0 ↔ x = a ∨ x = b) →
  a + b = a^2 + b^2 →
  a + b = a^3 + b^3 →
  a + b = a^4 + b^4 →
  ∃ (min : ℝ), min = 128 * Real.sqrt 3 / 45 ∧ 
    ∀ (t' q' a' b' : ℝ), 
      (∀ x, x^2 - t'*x + q' = 0 ↔ x = a' ∨ x = b') →
      a' + b' = a'^2 + b'^2 →
      a' + b' = a'^3 + b'^3 →
      a' + b' = a'^4 + b'^4 →
      1/a'^5 + 1/b'^5 ≥ min :=
sorry

end min_value_of_reciprocal_sum_l2003_200380


namespace geometric_series_r_value_l2003_200384

theorem geometric_series_r_value (a r : ℝ) (h1 : a ≠ 0) (h2 : |r| < 1) :
  (a / (1 - r) = 20) →
  (a * r / (1 - r^2) = 8) →
  r = 2/3 := by
sorry

end geometric_series_r_value_l2003_200384


namespace f_properties_l2003_200319

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 3

-- State the theorem
theorem f_properties :
  -- Monotonicity properties
  (∀ x y, x < y ∧ x < -1 → f x < f y) ∧
  (∀ x y, x < y ∧ 1 < x → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  -- Maximum and minimum values
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 59) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≥ f x ∧ f x = -49) :=
by sorry

end f_properties_l2003_200319


namespace flowmaster_pump_l2003_200361

/-- The FlowMaster pump problem -/
theorem flowmaster_pump (pump_rate : ℝ) (time : ℝ) (h1 : pump_rate = 600) (h2 : time = 0.5) :
  pump_rate * time = 300 := by
  sorry

end flowmaster_pump_l2003_200361


namespace vacuum_time_difference_l2003_200364

/-- Given vacuuming times, proves the difference between upstairs time and twice downstairs time -/
theorem vacuum_time_difference (total_time upstairs_time downstairs_time : ℕ) 
  (h1 : total_time = 38)
  (h2 : upstairs_time = 27)
  (h3 : total_time = upstairs_time + downstairs_time)
  (h4 : upstairs_time > 2 * downstairs_time) :
  upstairs_time - 2 * downstairs_time = 5 := by
  sorry


end vacuum_time_difference_l2003_200364


namespace blocks_left_in_second_tower_is_two_l2003_200386

/-- The number of blocks left standing in the second tower --/
def blocks_left_in_second_tower (first_stack_height : ℕ) 
                                (second_stack_diff : ℕ) 
                                (third_stack_diff : ℕ) 
                                (blocks_left_in_third : ℕ) 
                                (total_fallen : ℕ) : ℕ :=
  let second_stack_height := first_stack_height + second_stack_diff
  let third_stack_height := second_stack_height + third_stack_diff
  let total_blocks := first_stack_height + second_stack_height + third_stack_height
  let fallen_from_first := first_stack_height
  let fallen_from_third := third_stack_height - blocks_left_in_third
  let fallen_from_second := total_fallen - fallen_from_first - fallen_from_third
  second_stack_height - fallen_from_second

theorem blocks_left_in_second_tower_is_two :
  blocks_left_in_second_tower 7 5 7 3 33 = 2 := by
  sorry

end blocks_left_in_second_tower_is_two_l2003_200386


namespace theta_range_l2003_200347

theorem theta_range (θ : Real) : 
  (∀ x : Real, x ∈ Set.Icc 0 1 → x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) →
  π / 12 < θ ∧ θ < 5 * π / 12 :=
by sorry

end theta_range_l2003_200347


namespace blue_whale_tongue_weight_l2003_200390

-- Define the weight of one ton in pounds
def ton_in_pounds : ℕ := 2000

-- Define the weight of a blue whale's tongue in tons
def blue_whale_tongue_tons : ℕ := 3

-- Theorem: The weight of a blue whale's tongue in pounds
theorem blue_whale_tongue_weight :
  blue_whale_tongue_tons * ton_in_pounds = 6000 := by
  sorry

end blue_whale_tongue_weight_l2003_200390


namespace lists_with_high_number_l2003_200338

def total_balls : ℕ := 15
def draws : ℕ := 4
def threshold : ℕ := 10

theorem lists_with_high_number (total_balls draws threshold : ℕ) :
  total_balls = 15 ∧ draws = 4 ∧ threshold = 10 →
  (total_balls ^ draws) - (threshold ^ draws) = 40625 := by
  sorry

end lists_with_high_number_l2003_200338
