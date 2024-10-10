import Mathlib

namespace accounting_client_time_ratio_l3245_324512

/-- Given a total work time and time spent calling clients, 
    calculate the ratio of time spent doing accounting to time spent calling clients. -/
theorem accounting_client_time_ratio 
  (total_time : ℕ) 
  (client_time : ℕ) 
  (h1 : total_time = 560) 
  (h2 : client_time = 70) : 
  (total_time - client_time) / client_time = 7 := by
  sorry

end accounting_client_time_ratio_l3245_324512


namespace incorrect_inequality_l3245_324538

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(-2 * a > -2 * b) := by
  sorry

end incorrect_inequality_l3245_324538


namespace n_times_n_plus_one_is_even_l3245_324548

theorem n_times_n_plus_one_is_even (n : ℤ) : 2 ∣ n * (n + 1) := by
  sorry

end n_times_n_plus_one_is_even_l3245_324548


namespace mrs_randall_third_grade_years_l3245_324503

/-- Represents the number of years Mrs. Randall has been teaching -/
def total_teaching_years : ℕ := 26

/-- Represents the number of years Mrs. Randall taught second grade -/
def second_grade_years : ℕ := 8

/-- Represents the number of years Mrs. Randall has taught third grade -/
def third_grade_years : ℕ := total_teaching_years - second_grade_years

theorem mrs_randall_third_grade_years :
  third_grade_years = 18 :=
by sorry

end mrs_randall_third_grade_years_l3245_324503


namespace base_5_minus_base_7_digits_l3245_324569

-- Define the number of digits in a given base
def numDigits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

-- State the theorem
theorem base_5_minus_base_7_digits : 
  numDigits 2023 5 - numDigits 2023 7 = 1 := by
  sorry

end base_5_minus_base_7_digits_l3245_324569


namespace odd_function_sum_l3245_324565

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end odd_function_sum_l3245_324565


namespace triangle_third_side_minimum_l3245_324550

theorem triangle_third_side_minimum (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a - b = 5 ∨ b - a = 5) →
  Even (a + b + c) →
  c ≥ 7 :=
sorry

end triangle_third_side_minimum_l3245_324550


namespace lattice_point_proximity_probability_l3245_324559

theorem lattice_point_proximity_probability (d : ℝ) : 
  (d > 0) → 
  (π * d^2 = 1/3) → 
  (d = Real.sqrt (1 / (3 * π))) :=
by sorry

end lattice_point_proximity_probability_l3245_324559


namespace profit_percentage_l3245_324595

theorem profit_percentage (cost_price selling_price : ℝ) 
  (h : 58 * cost_price = 50 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 16 := by
  sorry

end profit_percentage_l3245_324595


namespace class_duration_theorem_l3245_324598

/-- Calculates the total duration of classes given the number of periods, period length, number of breaks, and break length. -/
def classDuration (numPeriods : ℕ) (periodLength : ℕ) (numBreaks : ℕ) (breakLength : ℕ) : ℕ :=
  numPeriods * periodLength + numBreaks * breakLength

/-- Proves that the total duration of classes with 5 periods of 40 minutes each and 4 breaks of 5 minutes each is 220 minutes. -/
theorem class_duration_theorem :
  classDuration 5 40 4 5 = 220 := by
  sorry

#eval classDuration 5 40 4 5

end class_duration_theorem_l3245_324598


namespace exists_function_satisfying_equation_l3245_324544

theorem exists_function_satisfying_equation : 
  ∃ f : ℤ → ℤ, ∀ a b : ℤ, f (a + b) - f (a * b) = f a * f b - 1 := by
  sorry

end exists_function_satisfying_equation_l3245_324544


namespace revenue_is_405_main_theorem_l3245_324505

/-- Represents the rental business scenario --/
structure RentalBusiness where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Calculates the total revenue for the rental business --/
def total_revenue (rb : RentalBusiness) : ℕ :=
  rb.canoe_cost * rb.canoe_count + rb.kayak_cost * rb.kayak_count

/-- Theorem stating that under the given conditions, the total revenue is $405 --/
theorem revenue_is_405 (rb : RentalBusiness) 
  (h1 : rb.canoe_cost = 15)
  (h2 : rb.kayak_cost = 18)
  (h3 : rb.canoe_count = (3 * rb.kayak_count) / 2)
  (h4 : rb.canoe_count = rb.kayak_count + 5) :
  total_revenue rb = 405 := by
  sorry

/-- Main theorem combining all conditions and proving the result --/
theorem main_theorem : ∃ (rb : RentalBusiness), 
  rb.canoe_cost = 15 ∧ 
  rb.kayak_cost = 18 ∧ 
  rb.canoe_count = (3 * rb.kayak_count) / 2 ∧
  rb.canoe_count = rb.kayak_count + 5 ∧
  total_revenue rb = 405 := by
  sorry

end revenue_is_405_main_theorem_l3245_324505


namespace opposite_of_half_l3245_324541

-- Define the concept of opposite
def opposite (x : ℝ) : ℝ := -x

-- Theorem statement
theorem opposite_of_half : opposite 0.5 = -0.5 := by
  sorry

end opposite_of_half_l3245_324541


namespace stretch_cosine_curve_l3245_324586

/-- Given a curve y = cos x and a stretch transformation x' = 2x and y' = 3y,
    prove that the new equation of the curve is y' = 3 cos (x' / 2) -/
theorem stretch_cosine_curve (x x' y y' : ℝ) :
  y = Real.cos x →
  x' = 2 * x →
  y' = 3 * y →
  y' = 3 * Real.cos (x' / 2) := by
  sorry

end stretch_cosine_curve_l3245_324586


namespace xy_squared_l3245_324589

theorem xy_squared (x y : ℝ) (h1 : 2 * x * (x + y) = 36) (h2 : 3 * y * (x + y) = 81) :
  (x + y)^2 = 117 / 5 := by
  sorry

end xy_squared_l3245_324589


namespace bertha_descendants_without_daughters_l3245_324580

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The given conditions of Bertha's family -/
def bertha_family : BerthaFamily :=
  { daughters := 8,
    granddaughters := 20,
    total_descendants := 28,
    daughters_with_children := 5 }

/-- Theorem stating the number of Bertha's descendants without daughters -/
theorem bertha_descendants_without_daughters :
  bertha_family.total_descendants - bertha_family.daughters_with_children = 23 := by
  sorry

#check bertha_descendants_without_daughters

end bertha_descendants_without_daughters_l3245_324580


namespace discount_rates_sum_l3245_324527

-- Define the regular prices
def fox_price : ℝ := 15
def pony_price : ℝ := 18

-- Define the number of pairs purchased
def fox_pairs : ℕ := 3
def pony_pairs : ℕ := 2

-- Define the total savings
def total_savings : ℝ := 8.55

-- Define the approximate discount rate for Pony jeans
def pony_discount_rate : ℝ := 0.15

-- Define the discount rates as variables
variable (fox_discount_rate : ℝ)

-- Theorem statement
theorem discount_rates_sum :
  fox_discount_rate + pony_discount_rate = 0.22 :=
by sorry

end discount_rates_sum_l3245_324527


namespace pyramid_inscribed_cube_volume_l3245_324504

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  height : ℝ

/-- A cube inscribed in the pyramid -/
structure InscribedCube where
  edge_length : ℝ

/-- The volume of a cube -/
def cube_volume (c : InscribedCube) : ℝ := c.edge_length ^ 3

theorem pyramid_inscribed_cube_volume 
  (p : Pyramid) 
  (c : InscribedCube) 
  (h_base : p.base_side = 2) 
  (h_height : p.height = Real.sqrt 6) 
  (h_cube_edge : c.edge_length = Real.sqrt 6 / 3) : 
  cube_volume c = 2 * Real.sqrt 6 / 9 := by
  sorry

end pyramid_inscribed_cube_volume_l3245_324504


namespace inscribed_triangle_existence_l3245_324546

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A triangle defined by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Check if a triangle is inscribed in a circle -/
def isInscribed (t : Triangle) (c : Circle) : Prop :=
  sorry

/-- Calculate an angle of a triangle -/
def angle (t : Triangle) (vertex : ℝ × ℝ) : ℝ :=
  sorry

/-- Calculate the length of a median in a triangle -/
def medianLength (t : Triangle) (vertex : ℝ × ℝ) : ℝ :=
  sorry

/-- The main theorem -/
theorem inscribed_triangle_existence (k : Circle) (α : ℝ) (s_b : ℝ) :
  ∃ n : Fin 3, ∃ triangles : Fin n → Triangle,
    (∀ i, isInscribed (triangles i) k) ∧
    (∀ i, ∃ v, angle (triangles i) v = α) ∧
    (∀ i, ∃ v, medianLength (triangles i) v = s_b) :=
  sorry

end inscribed_triangle_existence_l3245_324546


namespace arithmetic_sequence_a7_l3245_324575

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 + a 13 = 12 → a 7 = 6 := by
  sorry

end arithmetic_sequence_a7_l3245_324575


namespace sample_size_representation_l3245_324596

/-- Represents a population of students -/
structure Population where
  size : ℕ

/-- Represents a sample of students -/
structure Sample where
  size : ℕ
  population : Population
  h : size ≤ population.size

/-- Theorem: In a statistical analysis context, when 30 students are selected from a population of 500,
    the number 30 represents the sample size -/
theorem sample_size_representation (pop : Population) (s : Sample) :
  pop.size = 500 →
  s.size = 30 →
  s.population = pop →
  s.size = Sample.size s :=
by sorry

end sample_size_representation_l3245_324596


namespace alok_order_l3245_324500

/-- The number of chapatis ordered -/
def chapatis : ℕ := 16

/-- The number of rice plates ordered -/
def rice_plates : ℕ := 5

/-- The number of ice-cream cups ordered -/
def ice_cream_cups : ℕ := 6

/-- The cost of each chapati in rupees -/
def chapati_cost : ℕ := 6

/-- The cost of each rice plate in rupees -/
def rice_cost : ℕ := 45

/-- The cost of each mixed vegetable plate in rupees -/
def veg_cost : ℕ := 70

/-- The total amount paid by Alok in rupees -/
def total_paid : ℕ := 985

/-- The number of mixed vegetable plates ordered by Alok -/
def veg_plates : ℕ := (total_paid - (chapatis * chapati_cost + rice_plates * rice_cost)) / veg_cost

theorem alok_order : veg_plates = 9 := by sorry

end alok_order_l3245_324500


namespace area_ratio_abc_xyz_l3245_324564

-- Define points as pairs of real numbers
def Point := ℝ × ℝ

-- Define the given points
def A : Point := (2, 0)
def B : Point := (8, 12)
def C : Point := (14, 0)
def X : Point := (6, 0)
def Y : Point := (8, 4)
def Z : Point := (10, 0)

-- Function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem area_ratio_abc_xyz :
  (triangleArea X Y Z) / (triangleArea A B C) = 1 / 9 := by sorry

end area_ratio_abc_xyz_l3245_324564


namespace multiples_of_15_between_10_and_150_l3245_324513

theorem multiples_of_15_between_10_and_150 : 
  ∃ n : ℕ, n = (Finset.filter (λ x => 15 ∣ x ∧ x > 10 ∧ x < 150) (Finset.range 150)).card ∧ n = 10 := by
  sorry

end multiples_of_15_between_10_and_150_l3245_324513


namespace closest_integer_to_cube_root_l3245_324557

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 10 ∧ ∀ m : ℤ, |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| := by
  sorry

end closest_integer_to_cube_root_l3245_324557


namespace circle_radii_equation_l3245_324530

theorem circle_radii_equation (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (ea : a = 1 / a) (eb : b = 1 / b) (ec : c = 1 / c) (ed : d = 1 / d) :
  2 * (a^2 + b^2 + c^2 + d^2) = (a + b + c + d)^2 := by
  sorry

end circle_radii_equation_l3245_324530


namespace probability_three_tails_l3245_324535

def coin_flips : ℕ := 8
def p_tails : ℚ := 3/5
def p_heads : ℚ := 2/5
def num_tails : ℕ := 3

theorem probability_three_tails :
  (Nat.choose coin_flips num_tails : ℚ) * p_tails ^ num_tails * p_heads ^ (coin_flips - num_tails) = 48624/390625 := by
  sorry

end probability_three_tails_l3245_324535


namespace max_money_earned_is_zero_l3245_324591

/-- Represents the state of the three piles of stones -/
structure PileState where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ

/-- Represents a single move of a stone from one pile to another -/
inductive Move
  | oneToTwo
  | oneToThree
  | twoToOne
  | twoToThree
  | threeToOne
  | threeToTwo

/-- Applies a move to a pile state, returning the new state and the money earned -/
def applyMove (state : PileState) (move : Move) : PileState × ℤ := sorry

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to an initial state, returning the final state and total money earned -/
def applyMoves (initial : PileState) (moves : MoveSequence) : PileState × ℤ := sorry

/-- The main theorem: the maximum money that can be earned is 0 -/
theorem max_money_earned_is_zero (initial : PileState) :
  (∀ moves : MoveSequence, applyMoves initial moves = (initial, 0)) → 
  (∀ moves : MoveSequence, (applyMoves initial moves).2 ≤ 0) :=
sorry

end max_money_earned_is_zero_l3245_324591


namespace green_leaves_remaining_l3245_324509

theorem green_leaves_remaining (num_plants : ℕ) (initial_leaves : ℕ) (falling_fraction : ℚ) : 
  num_plants = 3 → 
  initial_leaves = 18 → 
  falling_fraction = 1/3 → 
  (num_plants * initial_leaves * (1 - falling_fraction) : ℚ) = 36 := by
sorry

end green_leaves_remaining_l3245_324509


namespace a_squared_lt_one_sufficient_not_necessary_for_a_lt_two_l3245_324529

theorem a_squared_lt_one_sufficient_not_necessary_for_a_lt_two :
  (∀ a : ℝ, a^2 < 1 → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 1) :=
by sorry

end a_squared_lt_one_sufficient_not_necessary_for_a_lt_two_l3245_324529


namespace correct_regression_sequence_l3245_324594

-- Define the steps of linear regression analysis
inductive RegressionStep
  | collectData
  | drawScatterPlot
  | calculateEquation
  | interpretEquation

-- Define a type for sequences of regression steps
def RegressionSequence := List RegressionStep

-- Define the correct sequence
def correctSequence : RegressionSequence :=
  [RegressionStep.collectData, RegressionStep.drawScatterPlot, 
   RegressionStep.calculateEquation, RegressionStep.interpretEquation]

-- Define a property for variables being linearly related
def linearlyRelated (x y : ℝ → ℝ) : Prop := sorry

-- Theorem stating that given linearly related variables, 
-- the correct sequence is as defined above
theorem correct_regression_sequence 
  (x y : ℝ → ℝ) 
  (h : linearlyRelated x y) : 
  correctSequence = 
    [RegressionStep.collectData, RegressionStep.drawScatterPlot, 
     RegressionStep.calculateEquation, RegressionStep.interpretEquation] :=
by
  sorry

end correct_regression_sequence_l3245_324594


namespace simplified_expression_implies_A_l3245_324570

/-- 
Given that (A - 3 / (a - 1)) * ((2 * a - 2) / (a + 2)) = 2 * a - 4,
prove that A = a + 1
-/
theorem simplified_expression_implies_A (a : ℝ) (A : ℝ) 
  (h : (A - 3 / (a - 1)) * ((2 * a - 2) / (a + 2)) = 2 * a - 4) :
  A = a + 1 := by
  sorry

end simplified_expression_implies_A_l3245_324570


namespace count_pairs_eq_45_l3245_324502

def count_pairs : Nat :=
  (Finset.range 6).sum fun m =>
    (Finset.range ((40 - (m + 1)^2) / 3 + 1)).card

theorem count_pairs_eq_45 : count_pairs = 45 := by
  sorry

end count_pairs_eq_45_l3245_324502


namespace parabola_properties_l3245_324525

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus
def focus : ℝ × ℝ := (0, 2)

-- Define the directrix
def directrix (x y : ℝ) : Prop := y = -2

-- Define a point on the parabola
def on_parabola (p : ℝ × ℝ) : Prop := parabola p.1 p.2

-- Define a point on the directrix
def on_directrix (p : ℝ × ℝ) : Prop := directrix p.1 p.2

-- Define the condition PF = FE
def PF_equals_FE (P E : ℝ × ℝ) : Prop :=
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = (E.1 - focus.1)^2 + (E.2 - focus.2)^2

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem parabola_properties :
  ∀ (P E : ℝ × ℝ),
  on_directrix P →
  on_parabola E →
  E.1 > 0 →
  E.2 > 0 →
  PF_equals_FE P E →
  (∃ (k : ℝ), k * P.1 - P.2 + 2 = 0 ∧ k = 1/Real.sqrt 3) ∧
  (∀ (D : ℝ × ℝ), on_parabola D →
    dot_product (D.1 - P.1, D.2 - P.2) (E.1 - P.1, E.2 - P.2) ≤ -64) ∧
  (∃ (P' : ℝ × ℝ), on_directrix P' ∧
    (P'.1 = 4 ∨ P'.1 = -4) ∧ P'.2 = -2 ∧
    (∀ (D E : ℝ × ℝ), on_parabola D → on_parabola E →
      dot_product (D.1 - P'.1, D.2 - P'.2) (E.1 - P'.1, E.2 - P'.2) = -64)) :=
sorry

end parabola_properties_l3245_324525


namespace painted_cube_problem_l3245_324558

theorem painted_cube_problem (n : ℕ) : n > 0 →
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 2 := by
  sorry

end painted_cube_problem_l3245_324558


namespace equation_is_hyperbola_l3245_324539

/-- A conic section type -/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section for a given quadratic equation -/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x^2 - 4y^2 - 2x + 8y - 8 = 0 represents a hyperbola -/
theorem equation_is_hyperbola :
  determineConicSection 1 (-4) 0 (-2) 8 (-8) = ConicSection.Hyperbola :=
sorry

end equation_is_hyperbola_l3245_324539


namespace necessary_not_sufficient_condition_l3245_324521

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}
def B : Set ℝ := {x | (x-3)*(2-x) ≥ 0}

-- State the theorem
theorem necessary_not_sufficient_condition (a : ℝ) (h1 : a > 0) 
  (h2 : B ⊂ A a) (h3 : A a ≠ B) : a ∈ Set.Ioo 1 2 := by
  sorry

end necessary_not_sufficient_condition_l3245_324521


namespace katie_cupcakes_made_l3245_324536

/-- The number of cupcakes Katie made after selling the first batch -/
def cupcakes_made_after (initial sold final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Theorem: Katie made 20 cupcakes after selling the first batch -/
theorem katie_cupcakes_made :
  cupcakes_made_after 26 20 26 = 20 := by
  sorry

end katie_cupcakes_made_l3245_324536


namespace prob_five_eight_sided_dice_l3245_324562

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The number of dice rolled -/
def k : ℕ := 5

/-- The probability of at least two dice showing the same number when rolling k fair n-sided dice -/
def prob_at_least_two_same (n k : ℕ) : ℚ :=
  1 - (n.factorial / (n - k).factorial : ℚ) / n^k

theorem prob_five_eight_sided_dice :
  prob_at_least_two_same n k = 3256 / 4096 :=
sorry

end prob_five_eight_sided_dice_l3245_324562


namespace amanda_earnings_l3245_324519

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℝ := 20

/-- Total hours worked on Monday -/
def monday_hours : ℝ := 5 * 1.5

/-- Total hours worked on Tuesday -/
def tuesday_hours : ℝ := 3

/-- Total hours worked on Thursday -/
def thursday_hours : ℝ := 2 * 2

/-- Total hours worked on Saturday -/
def saturday_hours : ℝ := 6

/-- Total hours worked in the week -/
def total_hours : ℝ := monday_hours + tuesday_hours + thursday_hours + saturday_hours

/-- Amanda's total earnings for the week -/
def total_earnings : ℝ := hourly_rate * total_hours

theorem amanda_earnings : total_earnings = 410 := by
  sorry

end amanda_earnings_l3245_324519


namespace longer_train_length_l3245_324518

-- Define the given values
def speed_train1 : Real := 60  -- km/hr
def speed_train2 : Real := 40  -- km/hr
def length_shorter : Real := 140  -- meters
def crossing_time : Real := 11.519078473722104  -- seconds

-- Define the theorem
theorem longer_train_length :
  ∃ (length_longer : Real),
    length_longer = 180 ∧
    length_shorter + length_longer =
      (speed_train1 + speed_train2) * 1000 / 3600 * crossing_time :=
by sorry

end longer_train_length_l3245_324518


namespace not_right_triangle_l3245_324593

/-- A predicate to check if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a)

/-- Theorem stating that (11, 40, 41) cannot form a right triangle --/
theorem not_right_triangle : ¬ is_right_triangle 11 40 41 := by
  sorry

#check not_right_triangle

end not_right_triangle_l3245_324593


namespace largest_share_is_12000_l3245_324553

/-- Represents the profit split ratio for four partners -/
structure ProfitSplit :=
  (a b c d : ℕ)

/-- Calculates the largest share given a total profit and a profit split ratio -/
def largest_share (total_profit : ℕ) (split : ProfitSplit) : ℕ :=
  let total_parts := split.a + split.b + split.c + split.d
  let largest_part := max split.a (max split.b (max split.c split.d))
  (total_profit / total_parts) * largest_part

/-- The theorem stating that the largest share is $12,000 -/
theorem largest_share_is_12000 :
  largest_share 30000 ⟨1, 4, 4, 6⟩ = 12000 := by
  sorry

#eval largest_share 30000 ⟨1, 4, 4, 6⟩

end largest_share_is_12000_l3245_324553


namespace regular_octagon_extended_sides_angle_l3245_324515

-- Define a regular octagon
structure RegularOctagon :=
  (vertices : Fin 8 → ℝ × ℝ)
  (is_regular : ∀ i j : Fin 8, dist (vertices i) (vertices ((i + 1) % 8)) = dist (vertices j) (vertices ((j + 1) % 8)))

-- Define the extension of sides CD and FG
def extend_sides (octagon : RegularOctagon) : ℝ × ℝ :=
  sorry

-- Define the angle at point Q
def angle_at_Q (octagon : RegularOctagon) : ℝ :=
  sorry

-- Theorem statement
theorem regular_octagon_extended_sides_angle (octagon : RegularOctagon) :
  angle_at_Q octagon = 180 :=
sorry

end regular_octagon_extended_sides_angle_l3245_324515


namespace smallest_positive_w_l3245_324567

theorem smallest_positive_w (y w : Real) (h1 : Real.sin y = 0) (h2 : Real.sin (y + w) = Real.sqrt 3 / 2) :
  ∃ (w_min : Real), w_min > 0 ∧ w_min = π / 3 ∧ ∀ (w' : Real), w' > 0 ∧ Real.sin y = 0 ∧ Real.sin (y + w') = Real.sqrt 3 / 2 → w' ≥ w_min :=
sorry

end smallest_positive_w_l3245_324567


namespace company_average_salary_l3245_324532

theorem company_average_salary
  (num_managers : ℕ)
  (num_associates : ℕ)
  (avg_salary_managers : ℚ)
  (avg_salary_associates : ℚ)
  (h1 : num_managers = 15)
  (h2 : num_associates = 75)
  (h3 : avg_salary_managers = 90000)
  (h4 : avg_salary_associates = 30000) :
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) / (num_managers + num_associates) = 40000 := by
sorry

end company_average_salary_l3245_324532


namespace laura_change_l3245_324545

/-- Calculates the change Laura should receive after her shopping trip -/
theorem laura_change : 
  let pants_cost : ℕ := 2 * 64
  let shirts_cost : ℕ := 4 * 42
  let shoes_cost : ℕ := 3 * 78
  let jackets_cost : ℕ := 2 * 103
  let watch_cost : ℕ := 215
  let jewelry_cost : ℕ := 2 * 120
  let total_cost : ℕ := pants_cost + shirts_cost + shoes_cost + jackets_cost + watch_cost + jewelry_cost
  let amount_given : ℕ := 800
  Int.ofNat amount_given - Int.ofNat total_cost = -391 := by
  sorry

end laura_change_l3245_324545


namespace min_lcm_ac_l3245_324573

theorem min_lcm_ac (a b c : ℕ+) (h1 : Nat.lcm a b = 18) (h2 : Nat.lcm b c = 28) :
  ∃ (a' c' : ℕ+), Nat.lcm a' c' = 126 ∧ 
    (∀ (x y : ℕ+), Nat.lcm x b = 18 → Nat.lcm b y = 28 → Nat.lcm a' c' ≤ Nat.lcm x y) :=
by sorry

end min_lcm_ac_l3245_324573


namespace quadratic_function_transformation_l3245_324568

-- Define the quadratic function f(x) = ax² + bx + c
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the function g(x) = cx² + 2bx + a
def g (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + 2 * b * x + a

theorem quadratic_function_transformation (a b c : ℝ) :
  (f a b c 0 = 1) ∧ 
  (f a b c 1 = -2) ∧ 
  (f a b c (-1) = 2) →
  (∀ x, g a b c x = x^2 - 4*x - 1) :=
by sorry

end quadratic_function_transformation_l3245_324568


namespace company_managers_count_l3245_324572

/-- Proves that the number of managers is 15 given the conditions in the problem --/
theorem company_managers_count :
  ∀ (num_managers : ℕ) (num_associates : ℕ) (avg_salary_managers : ℚ) (avg_salary_associates : ℚ) (avg_salary_company : ℚ),
  num_associates = 75 →
  avg_salary_managers = 90000 →
  avg_salary_associates = 30000 →
  avg_salary_company = 40000 →
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) / (num_managers + num_associates : ℚ) = avg_salary_company →
  num_managers = 15 :=
by
  sorry

end company_managers_count_l3245_324572


namespace charlie_missing_coins_l3245_324578

/-- Represents the fraction of coins Charlie has at different stages -/
structure CoinFraction where
  total : ℚ
  dropped : ℚ
  found : ℚ

/-- Calculates the fraction of coins still missing -/
def missing_fraction (cf : CoinFraction) : ℚ :=
  cf.total - (cf.total - cf.dropped + cf.found * cf.dropped)

/-- Theorem stating that the fraction of missing coins is 1/9 -/
theorem charlie_missing_coins :
  let cf : CoinFraction := { total := 1, dropped := 1/3, found := 2/3 }
  missing_fraction cf = 1/9 := by
  sorry

#check charlie_missing_coins

end charlie_missing_coins_l3245_324578


namespace sphere_volume_circumscribing_rectangular_prism_l3245_324582

theorem sphere_volume_circumscribing_rectangular_prism :
  let edge1 : ℝ := 1
  let edge2 : ℝ := Real.sqrt 10
  let edge3 : ℝ := 5
  let space_diagonal : ℝ := Real.sqrt (edge1^2 + edge2^2 + edge3^2)
  let sphere_radius : ℝ := space_diagonal / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = 36 * Real.pi := by
  sorry

end sphere_volume_circumscribing_rectangular_prism_l3245_324582


namespace complex_product_real_imag_parts_l3245_324590

theorem complex_product_real_imag_parts 
  (c d : ℂ) (x : ℝ) 
  (h1 : Complex.abs c = 3) 
  (h2 : Complex.abs d = 5) 
  (h3 : c * d = x + 6 * Complex.I) :
  x = 3 * Real.sqrt 21 :=
sorry

end complex_product_real_imag_parts_l3245_324590


namespace fixed_point_on_line_l3245_324551

theorem fixed_point_on_line (m : ℝ) : 
  (3*m + 4) * (-1) + (5 - 2*m) * 2 + 7*m - 6 = 0 := by
sorry

end fixed_point_on_line_l3245_324551


namespace max_value_expression_l3245_324577

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 12) :
  a * b + b * c + a * c + a * b * c ≤ 112 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 12 ∧ 
    a' * b' + b' * c' + a' * c' + a' * b' * c' = 112 :=
sorry

end max_value_expression_l3245_324577


namespace absolute_value_equation_solution_l3245_324587

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 2*y = 12 :=
by
  -- The unique solution is y = 4
  use 4
  sorry

end absolute_value_equation_solution_l3245_324587


namespace equality_except_two_l3245_324549

theorem equality_except_two (x : ℝ) : 
  x ≠ 2 → (x^2 - 4*x + 4) / (x - 2) = x - 2 := by
  sorry

end equality_except_two_l3245_324549


namespace geometric_sequence_problem_l3245_324566

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : is_geometric_sequence a)
  (h_prod1 : a 1 * a 2 * a 3 = 4)
  (h_prod2 : a 4 * a 5 * a 6 = 12)
  (h_prod3 : ∃ n : ℕ, a (n - 1) * a n * a (n + 1) = 324) :
  ∃ n : ℕ, a (n - 1) * a n * a (n + 1) = 324 ∧ n = 14 :=
by sorry

end geometric_sequence_problem_l3245_324566


namespace fourth_month_sales_l3245_324516

def sales_problem (sales1 sales2 sales3 sales5 sales6 : ℕ) (average : ℕ) : Prop :=
  let total := average * 6
  let known_sales := sales1 + sales2 + sales3 + sales5 + sales6
  total - known_sales = 7230

theorem fourth_month_sales :
  sales_problem 6735 6927 6855 6562 4691 6500 :=
by
  sorry

end fourth_month_sales_l3245_324516


namespace largest_k_for_real_roots_l3245_324552

theorem largest_k_for_real_roots (k : ℤ) : 
  (∃ x : ℝ, x * (k * x + 1) - x^2 + 3 = 0) → 
  k ≠ 1 → 
  k ≤ 0 :=
by sorry

end largest_k_for_real_roots_l3245_324552


namespace f_max_min_range_l3245_324531

/-- A function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*((a+2)*x+1)

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- Theorem stating the range of a for which f has both a maximum and minimum -/
theorem f_max_min_range (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), ∀ (x : ℝ), f a x₁ ≤ f a x ∧ f a x ≤ f a x₂) →
  a < -1 ∨ a > 2 :=
sorry

end f_max_min_range_l3245_324531


namespace ellipse_standard_equation_l3245_324510

/-- The standard equation of an ellipse given its focal distance and sum of distances from a point to foci -/
theorem ellipse_standard_equation (focal_distance sum_distances : ℝ) :
  focal_distance = 8 →
  sum_distances = 10 →
  (∃ x y : ℝ, x^2 / 25 + y^2 / 9 = 1) ∨ (∃ x y : ℝ, x^2 / 9 + y^2 / 25 = 1) :=
by sorry

end ellipse_standard_equation_l3245_324510


namespace stating_club_truncator_probability_l3245_324584

/-- The number of matches played by Club Truncator -/
def num_matches : ℕ := 8

/-- The probability of winning, losing, or tying a single match -/
def single_match_prob : ℚ := 1/3

/-- The probability of finishing with more wins than losses -/
def more_wins_prob : ℚ := 2741/6561

/-- 
Theorem stating that given 8 matches where the probability of winning, 
losing, or tying each match is 1/3, the probability of finishing with 
more wins than losses is 2741/6561.
-/
theorem club_truncator_probability : 
  (num_matches = 8) → 
  (single_match_prob = 1/3) → 
  (more_wins_prob = 2741/6561) :=
by sorry

end stating_club_truncator_probability_l3245_324584


namespace set_intersection_problem_l3245_324588

theorem set_intersection_problem :
  let A : Set ℤ := {0, 1, 2}
  let B : Set ℤ := {-2, -1, 0, 1}
  A ∩ B = {0, 1} := by
sorry

end set_intersection_problem_l3245_324588


namespace quadratic_roots_sum_minus_product_l3245_324524

theorem quadratic_roots_sum_minus_product (x₁ x₂ : ℝ) : 
  (x₁^2 - x₁ - 2022 = 0) → 
  (x₂^2 - x₂ - 2022 = 0) → 
  x₁ + x₂ - x₁ * x₂ = 2023 := by
sorry

end quadratic_roots_sum_minus_product_l3245_324524


namespace parallel_lines_m_value_l3245_324579

/-- Given two parallel lines, one passing through P(3,2m) and Q(m,2),
    and another passing through M(2,-1) and N(-3,4), prove that m = -1 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ,
  let P : ℝ × ℝ := (3, 2*m)
  let Q : ℝ × ℝ := (m, 2)
  let M : ℝ × ℝ := (2, -1)
  let N : ℝ × ℝ := (-3, 4)
  let slope_PQ := (Q.2 - P.2) / (Q.1 - P.1)
  let slope_MN := (N.2 - M.2) / (N.1 - M.1)
  slope_PQ = slope_MN →
  m = -1 :=
by sorry

end parallel_lines_m_value_l3245_324579


namespace beta_value_l3245_324501

theorem beta_value (α β : Real) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5)
  (h_sin_α_β : Real.sin (α - β) = -(Real.sqrt 10) / 10) : β = π / 4 := by
  sorry

end beta_value_l3245_324501


namespace geometric_series_ratio_l3245_324571

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a * r^4 / (1 - r)) = (1 / 64) * (a / (1 - r)) →
  r = 1/2 := by
sorry

end geometric_series_ratio_l3245_324571


namespace max_3m_plus_4n_l3245_324506

theorem max_3m_plus_4n (m n : ℕ) : 
  (∃ (evens : Finset ℕ) (odds : Finset ℕ), 
    evens.card = m ∧ 
    odds.card = n ∧
    (∀ x ∈ evens, Even x ∧ x > 0) ∧
    (∀ y ∈ odds, Odd y ∧ y > 0) ∧
    (evens.sum id + odds.sum id = 1987)) →
  3 * m + 4 * n ≤ 221 :=
by sorry

end max_3m_plus_4n_l3245_324506


namespace ab_value_l3245_324537

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 33) : a * b = 12 := by
  sorry

end ab_value_l3245_324537


namespace sin_40_tan_10_minus_sqrt3_l3245_324555

theorem sin_40_tan_10_minus_sqrt3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -8/3 := by
  sorry

end sin_40_tan_10_minus_sqrt3_l3245_324555


namespace supplementary_angles_difference_l3245_324522

theorem supplementary_angles_difference (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 3 →  -- ratio of angles is 5:3
  (∃ k : ℕ, a = 15 * k ∨ b = 15 * k) →  -- one angle is multiple of 15
  |a - b| = 45 := by
sorry

end supplementary_angles_difference_l3245_324522


namespace inequality_proof_l3245_324583

theorem inequality_proof (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hy₁ : x₁ * y₁ > z₁^2) (hy₂ : x₂ * y₂ > z₂^2) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) := by
sorry

end inequality_proof_l3245_324583


namespace triangle_inequality_l3245_324543

/-- Theorem: For any triangle with side lengths a, b, c and perimeter 2, 
    the inequality a^2 + b^2 + c^2 < 2(1 - abc) holds. -/
theorem triangle_inequality (a b c : ℝ) 
  (triangle_cond : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (perimeter_cond : a + b + c = 2) : 
  a^2 + b^2 + c^2 < 2*(1 - a*b*c) := by
  sorry

end triangle_inequality_l3245_324543


namespace janes_age_l3245_324599

/-- Jane's babysitting problem -/
theorem janes_age (start_age : ℕ) (years_since_stop : ℕ) (oldest_babysat_now : ℕ)
  (h1 : start_age = 18)
  (h2 : years_since_stop = 12)
  (h3 : oldest_babysat_now = 23) :
  ∃ (current_age : ℕ),
    current_age = 34 ∧
    current_age ≥ start_age + years_since_stop ∧
    2 * (oldest_babysat_now - years_since_stop) ≤ current_age - years_since_stop :=
by sorry

end janes_age_l3245_324599


namespace arithmetic_calculation_l3245_324508

theorem arithmetic_calculation : (-1 + 2) * 3 + 2^2 / (-4) = 2 := by
  sorry

end arithmetic_calculation_l3245_324508


namespace jefferson_bananas_l3245_324534

theorem jefferson_bananas (jefferson_bananas : ℕ) (walter_bananas : ℕ) : 
  walter_bananas = jefferson_bananas - (1/4 : ℚ) * jefferson_bananas →
  (jefferson_bananas + walter_bananas) / 2 = 49 →
  jefferson_bananas = 56 := by
sorry

end jefferson_bananas_l3245_324534


namespace amount_to_fifth_sixth_homes_l3245_324526

/-- The amount donated to the fifth and sixth nursing homes combined -/
def amount_fifth_sixth (total donation_1 donation_2 donation_3 donation_4 : ℕ) : ℕ :=
  total - (donation_1 + donation_2 + donation_3 + donation_4)

/-- Theorem stating the amount given to the fifth and sixth nursing homes -/
theorem amount_to_fifth_sixth_homes :
  amount_fifth_sixth 10000 2750 1945 1275 1890 = 2140 := by
  sorry

end amount_to_fifth_sixth_homes_l3245_324526


namespace problem_solution_l3245_324592

theorem problem_solution (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x + 8) = x^2 + b*x + 24) → 
  a + b = 14 := by
sorry

end problem_solution_l3245_324592


namespace hospital_nurses_count_l3245_324547

theorem hospital_nurses_count 
  (total_staff : ℕ) 
  (doctor_ratio : ℕ) 
  (nurse_ratio : ℕ) 
  (h1 : total_staff = 200)
  (h2 : doctor_ratio = 4)
  (h3 : nurse_ratio = 6) :
  (nurse_ratio * total_staff) / (doctor_ratio + nurse_ratio) = 120 := by
  sorry

end hospital_nurses_count_l3245_324547


namespace roses_in_vase_correct_l3245_324517

/-- Given a total number of roses and the number of roses left,
    calculate the number of roses put in a vase. -/
def roses_in_vase (total : ℕ) (left : ℕ) : ℕ :=
  total - left

theorem roses_in_vase_correct (total : ℕ) (left : ℕ) 
  (h : left ≤ total) : 
  roses_in_vase total left = total - left :=
by
  sorry

#eval roses_in_vase 29 12  -- Should evaluate to 17

end roses_in_vase_correct_l3245_324517


namespace uniform_random_transformation_l3245_324574

theorem uniform_random_transformation (b₁ : ℝ) (b : ℝ) :
  (∀ x ∈ Set.Icc 0 1, b₁ ∈ Set.Icc 0 1) →
  b = (b₁ - 2) * 3 →
  (∀ y ∈ Set.Icc (-6) (-3), b ∈ Set.Icc (-6) (-3)) :=
by sorry

end uniform_random_transformation_l3245_324574


namespace manny_marbles_after_sharing_l3245_324556

/-- Given a total number of marbles and a ratio, calculates the number of marbles for each part -/
def marbles_per_part (total : ℕ) (ratio_sum : ℕ) : ℕ := total / ratio_sum

/-- Calculates the initial number of marbles for a person given their ratio part and marbles per part -/
def initial_marbles (ratio_part : ℕ) (marbles_per_part : ℕ) : ℕ := ratio_part * marbles_per_part

/-- Calculates the final number of marbles after giving away some -/
def final_marbles (initial : ℕ) (given_away : ℕ) : ℕ := initial - given_away

theorem manny_marbles_after_sharing (total_marbles : ℕ) (mario_ratio : ℕ) (manny_ratio : ℕ) (shared_marbles : ℕ) :
  total_marbles = 36 →
  mario_ratio = 4 →
  manny_ratio = 5 →
  shared_marbles = 2 →
  final_marbles (initial_marbles manny_ratio (marbles_per_part total_marbles (mario_ratio + manny_ratio))) shared_marbles = 18 := by
  sorry

end manny_marbles_after_sharing_l3245_324556


namespace cyrus_pages_left_l3245_324528

/-- Represents the number of pages Cyrus writes on each day --/
def pages_written : Fin 4 → ℕ
| 0 => 25  -- Day 1
| 1 => 2 * 25  -- Day 2
| 2 => 2 * (2 * 25)  -- Day 3
| 3 => 10  -- Day 4

/-- The total number of pages Cyrus needs to write --/
def total_pages : ℕ := 500

/-- The number of pages Cyrus still needs to write --/
def pages_left : ℕ := total_pages - (pages_written 0 + pages_written 1 + pages_written 2 + pages_written 3)

theorem cyrus_pages_left : pages_left = 315 := by
  sorry

end cyrus_pages_left_l3245_324528


namespace triangle_max_area_l3245_324520

/-- Given a triangle ABC where AB = 9 and BC:AC = 3:4, 
    the maximum possible area of the triangle is 243 / (2√7) square units. -/
theorem triangle_max_area (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  AB = 9 ∧ BC / AC = 3 / 4 → 
  area ≤ 243 / (2 * Real.sqrt 7) := by
sorry


end triangle_max_area_l3245_324520


namespace geometric_sum_proof_l3245_324514

theorem geometric_sum_proof : 
  let a : ℚ := 3/2
  let r : ℚ := 3/2
  let n : ℕ := 15
  let sum : ℚ := (a * (1 - r^n)) / (1 - r)
  sum = 42948417/32768 := by sorry

end geometric_sum_proof_l3245_324514


namespace x_is_perfect_square_l3245_324585

theorem x_is_perfect_square (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ n : ℕ+, x = n^2 := by
sorry

end x_is_perfect_square_l3245_324585


namespace palindrome_count_l3245_324511

/-- Represents a time on a 12-hour digital clock --/
structure Time where
  hour : Nat
  minute : Nat
  hour_valid : 1 ≤ hour ∧ hour ≤ 12
  minute_valid : minute < 60

/-- Checks if a given time is a palindrome --/
def isPalindrome (t : Time) : Bool :=
  let digits := 
    if t.hour < 10 then
      [t.hour, t.minute / 10, t.minute % 10]
    else
      [t.hour / 10, t.hour % 10, t.minute / 10, t.minute % 10]
  digits = digits.reverse

/-- The set of all valid palindrome times on a 12-hour digital clock --/
def palindromeTimes : Finset Time :=
  sorry

theorem palindrome_count : palindromeTimes.card = 57 := by
  sorry

end palindrome_count_l3245_324511


namespace keanu_fish_problem_l3245_324540

theorem keanu_fish_problem :
  ∀ (dog_fish cat_fish : ℕ),
    cat_fish = dog_fish / 2 →
    dog_fish + cat_fish = 240 / 4 →
    dog_fish = 40 :=
by
  sorry

end keanu_fish_problem_l3245_324540


namespace kat_boxing_hours_l3245_324560

/-- Represents Kat's weekly training schedule -/
structure TrainingSchedule where
  strength_sessions : ℕ
  strength_hours_per_session : ℚ
  boxing_sessions : ℕ
  total_hours : ℚ

/-- Calculates the number of hours Kat trains at the boxing gym each time -/
def boxing_hours_per_session (schedule : TrainingSchedule) : ℚ :=
  (schedule.total_hours - schedule.strength_sessions * schedule.strength_hours_per_session) / schedule.boxing_sessions

/-- Theorem stating that Kat trains 1.5 hours at the boxing gym each time -/
theorem kat_boxing_hours (schedule : TrainingSchedule) 
  (h1 : schedule.strength_sessions = 3)
  (h2 : schedule.strength_hours_per_session = 1)
  (h3 : schedule.boxing_sessions = 4)
  (h4 : schedule.total_hours = 9) :
  boxing_hours_per_session schedule = 3/2 := by
  sorry

#eval boxing_hours_per_session { strength_sessions := 3, strength_hours_per_session := 1, boxing_sessions := 4, total_hours := 9 }

end kat_boxing_hours_l3245_324560


namespace collinear_probability_l3245_324554

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The number of dots to be chosen -/
def dotsChosen : ℕ := 4

/-- The total number of ways to choose 4 dots from a 5x5 grid -/
def totalWays : ℕ := Nat.choose (gridSize * gridSize) dotsChosen

/-- The number of ways to choose 4 collinear dots -/
def collinearWays : ℕ := 
  gridSize * Nat.choose gridSize dotsChosen + -- Horizontal lines
  gridSize * Nat.choose gridSize dotsChosen + -- Vertical lines
  2 * Nat.choose gridSize dotsChosen +        -- Main diagonals
  4                                           -- Adjacent diagonals

/-- The probability of choosing 4 collinear dots from a 5x5 grid -/
theorem collinear_probability : 
  (collinearWays : ℚ) / totalWays = 64 / 12650 := by sorry

end collinear_probability_l3245_324554


namespace lacson_sweet_potato_sales_l3245_324597

/-- The problem of Mrs. Lacson's sweet potato sales -/
theorem lacson_sweet_potato_sales 
  (total : ℕ)
  (sold_to_adams : ℕ)
  (unsold : ℕ)
  (h1 : total = 80)
  (h2 : sold_to_adams = 20)
  (h3 : unsold = 45) :
  total - sold_to_adams - unsold = 15 := by
  sorry

end lacson_sweet_potato_sales_l3245_324597


namespace choose_four_from_ten_l3245_324542

theorem choose_four_from_ten (n : ℕ) (k : ℕ) : n = 10 → k = 4 → Nat.choose n k = 210 := by
  sorry

end choose_four_from_ten_l3245_324542


namespace minimum_mass_for_upward_roll_l3245_324561

/-- Given a cylinder of mass M on rails inclined at angle α = 45°, 
    the minimum mass m of a weight attached to a string wound around the cylinder 
    for it to roll upward without slipping is M(√2 + 1) -/
theorem minimum_mass_for_upward_roll (M : ℝ) (α : ℝ) 
    (h_α : α = π / 4) : 
    ∃ m : ℝ, m = M * (Real.sqrt 2 + 1) ∧ 
    m * (1 - Real.sin α) = M * Real.sin α := by
  sorry

end minimum_mass_for_upward_roll_l3245_324561


namespace max_savings_is_90_l3245_324533

structure Airline where
  name : String
  originalPrice : ℕ
  discountPercentage : ℕ

def calculateDiscountedPrice (airline : Airline) : ℕ :=
  airline.originalPrice - (airline.originalPrice * airline.discountPercentage / 100)

def airlines : List Airline := [
  { name := "Delta", originalPrice := 850, discountPercentage := 20 },
  { name := "United", originalPrice := 1100, discountPercentage := 30 },
  { name := "American", originalPrice := 950, discountPercentage := 25 },
  { name := "Southwest", originalPrice := 900, discountPercentage := 15 },
  { name := "JetBlue", originalPrice := 1200, discountPercentage := 40 }
]

theorem max_savings_is_90 :
  let discountedPrices := airlines.map calculateDiscountedPrice
  let cheapestPrice := discountedPrices.minimum?
  let maxSavings := discountedPrices.map (fun price => price - cheapestPrice.getD 0)
  maxSavings.maximum? = some 90 := by
  sorry

end max_savings_is_90_l3245_324533


namespace line_plane_perp_sufficiency_not_necessity_l3245_324507

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicularity relation between a line and a plane
variable (line_perp_plane : Line → Plane → Prop)

-- Define the perpendicularity relation between two planes
variable (plane_perp_plane : Plane → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_perp_sufficiency_not_necessity
  (α β : Plane) (m : Line)
  (h_diff : α ≠ β)
  (h_m_in_α : line_in_plane m α) :
  (line_perp_plane m β → plane_perp_plane α β) ∧
  ¬(plane_perp_plane α β → line_perp_plane m β) :=
sorry

end line_plane_perp_sufficiency_not_necessity_l3245_324507


namespace frequency_of_a_l3245_324576

def sentence : String := "Happy Teachers'Day!"

theorem frequency_of_a (s : String) (h : s = sentence) : 
  (s.toList.filter (· = 'a')).length = 3 := by sorry

end frequency_of_a_l3245_324576


namespace circle_center_coordinates_l3245_324581

def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = Real.sqrt 2 * (Real.cos θ + Real.sin θ)

theorem circle_center_coordinates :
  ∃ (r θ : ℝ), polar_equation r θ ∧ r = 1 ∧ θ = π / 4 :=
sorry

end circle_center_coordinates_l3245_324581


namespace set_operations_l3245_324523

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {0, 1, 4}

-- Define set B
def B : Finset Nat := {0, 1, 3}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {0, 1}) ∧ (A ∪ B = {0, 1, 3, 4}) := by
  sorry

end set_operations_l3245_324523


namespace heartsuit_three_eight_l3245_324563

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 2 * y

-- State the theorem
theorem heartsuit_three_eight : heartsuit 3 8 = 28 := by
  sorry

end heartsuit_three_eight_l3245_324563
