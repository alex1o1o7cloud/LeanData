import Mathlib

namespace NUMINAMATH_CALUDE_larger_number_proof_l3130_313018

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 1360) 
  (h2 : y = 6 * x + 15) : 
  y = 1629 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3130_313018


namespace NUMINAMATH_CALUDE_pen_cost_ratio_l3130_313051

theorem pen_cost_ratio (blue_pens : ℕ) (red_pens : ℕ) (blue_cost : ℚ) (total_cost : ℚ) : 
  blue_pens = 10 →
  red_pens = 15 →
  blue_cost = 1/10 →
  total_cost = 4 →
  (total_cost - blue_pens * blue_cost) / red_pens / blue_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_ratio_l3130_313051


namespace NUMINAMATH_CALUDE_ice_cream_volume_l3130_313093

/-- The volume of ice cream on a cone -/
theorem ice_cream_volume (r h : ℝ) (hr : r = 3) (hh : h = 1) :
  (2/3) * π * r^3 + π * r^2 * h = 27 * π := by sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l3130_313093


namespace NUMINAMATH_CALUDE_square_of_nine_l3130_313068

theorem square_of_nine (x : ℝ) : x^2 = 9 → x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_square_of_nine_l3130_313068


namespace NUMINAMATH_CALUDE_solution_set_f_gt_3_range_of_a_l3130_313057

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 1|

-- Define the function g (although not used in the proof)
def g (x : ℝ) : ℝ := |2*x - 1| + 3

-- Theorem 1: The solution set of f(x) > 3 is (0, +∞)
theorem solution_set_f_gt_3 : 
  {x : ℝ | f x > 3} = {x : ℝ | x > 0} := by sorry

-- Theorem 2: If f(x) + 1 < 4^a - 5×2^a has a solution, then a < 0 or a > 2
theorem range_of_a (a : ℝ) : 
  (∃ x, f x + 1 < 4^a - 5*2^a) → (a < 0 ∨ a > 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_3_range_of_a_l3130_313057


namespace NUMINAMATH_CALUDE_field_of_miracles_l3130_313073

/-- The Field of Miracles problem -/
theorem field_of_miracles
  (a b : ℝ)
  (ha : a = 6)
  (hb : b = 2.5)
  (v_malvina : ℝ)
  (hv_malvina : v_malvina = 4)
  (v_buratino : ℝ)
  (hv_buratino : v_buratino = 6)
  (v_artemon : ℝ)
  (hv_artemon : v_artemon = 12) :
  let d := Real.sqrt (a^2 + b^2)
  let t := d / (v_malvina + v_buratino)
  v_artemon * t = 7.8 :=
by sorry

end NUMINAMATH_CALUDE_field_of_miracles_l3130_313073


namespace NUMINAMATH_CALUDE_x_in_terms_of_z_l3130_313075

theorem x_in_terms_of_z (x y z : ℝ) 
  (eq1 : 0.35 * (400 + y) = 0.20 * x)
  (eq2 : x = 2 * z^2)
  (eq3 : y = 3 * z - 5) :
  x = 2 * z^2 := by
sorry

end NUMINAMATH_CALUDE_x_in_terms_of_z_l3130_313075


namespace NUMINAMATH_CALUDE_planes_divide_space_l3130_313046

/-- The number of regions into which n planes can divide space -/
def R (n : ℕ) : ℚ := (n^3 + 5*n + 6) / 6

/-- Theorem stating that R(n) gives the correct number of regions for n planes -/
theorem planes_divide_space (n : ℕ) : 
  R n = (n^3 + 5*n + 6) / 6 := by sorry

end NUMINAMATH_CALUDE_planes_divide_space_l3130_313046


namespace NUMINAMATH_CALUDE_largest_valid_number_sum_of_digits_l3130_313080

def is_valid_remainder (r : ℕ) (m : ℕ) : Prop :=
  r > 1 ∧ r < m

def form_geometric_progression (r1 r2 r3 : ℕ) : Prop :=
  (r2 * r2 = r1 * r3) ∧ r1 ≠ r2

def satisfies_conditions (n : ℕ) : Prop :=
  ∃ (r1 r2 r3 : ℕ),
    is_valid_remainder r1 9 ∧
    is_valid_remainder r2 10 ∧
    is_valid_remainder r3 11 ∧
    form_geometric_progression r1 r2 r3 ∧
    n % 9 = r1 ∧
    n % 10 = r2 ∧
    n % 11 = r3

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_valid_number_sum_of_digits :
  ∃ (N : ℕ), N < 990 ∧ satisfies_conditions N ∧
  (∀ (m : ℕ), m < 990 → satisfies_conditions m → m ≤ N) ∧
  sum_of_digits N = 13 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_sum_of_digits_l3130_313080


namespace NUMINAMATH_CALUDE_coefficient_sum_is_five_sixths_l3130_313056

/-- A polynomial function from ℝ to ℝ -/
def PolynomialFunction := ℝ → ℝ

/-- The property that f(x) - f(x-2) = (2x-1)^2 for all x -/
def SatisfiesEquation (f : PolynomialFunction) : Prop :=
  ∀ x, f x - f (x - 2) = (2 * x - 1)^2

/-- The coefficient of x^2 in a polynomial function -/
def CoefficientOfXSquared (f : PolynomialFunction) : ℝ := sorry

/-- The coefficient of x in a polynomial function -/
def CoefficientOfX (f : PolynomialFunction) : ℝ := sorry

theorem coefficient_sum_is_five_sixths (f : PolynomialFunction) 
  (h : SatisfiesEquation f) : 
  CoefficientOfXSquared f + CoefficientOfX f = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_is_five_sixths_l3130_313056


namespace NUMINAMATH_CALUDE_courtyard_ratio_l3130_313023

/-- Given a courtyard with trees, stones, and birds, prove the ratio of trees to stones -/
theorem courtyard_ratio (stones birds : ℕ) (h1 : stones = 40) (h2 : birds = 400)
  (h3 : birds = 2 * (trees + stones)) : (trees : ℚ) / stones = 4 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_courtyard_ratio_l3130_313023


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l3130_313061

theorem common_number_in_overlapping_lists (list : List ℝ) : 
  list.length = 8 →
  (list.take 5).sum / 5 = 6 →
  (list.drop 3).sum / 5 = 9 →
  list.sum / 8 = 7.5 →
  ∃ x ∈ list.take 5 ∩ list.drop 3, x = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l3130_313061


namespace NUMINAMATH_CALUDE_sticker_distribution_l3130_313008

/-- The number of ways to distribute n identical objects into k identical containers -/
def distribute_objects (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 25 ways to distribute 10 identical stickers onto 5 identical sheets of paper -/
theorem sticker_distribution : distribute_objects 10 5 = 25 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3130_313008


namespace NUMINAMATH_CALUDE_expression_evaluation_l3130_313012

theorem expression_evaluation :
  let a : ℚ := -1/3
  let b : ℤ := -3
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3130_313012


namespace NUMINAMATH_CALUDE_job_completion_time_l3130_313002

/-- The time taken for two workers to complete a job together -/
def job_time (rate1 rate2 : ℚ) : ℚ := 1 / (rate1 + rate2)

theorem job_completion_time 
  (rate_A rate_B rate_C : ℚ) 
  (h1 : rate_A + rate_B = 1 / 6)  -- A and B can do the job in 6 days
  (h2 : rate_B + rate_C = 1 / 10) -- B and C can do the job in 10 days
  (h3 : rate_A + rate_B + rate_C = 1 / 5) -- A, B, and C can do the job in 5 days
  : job_time rate_A rate_C = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3130_313002


namespace NUMINAMATH_CALUDE_parallel_vectors_l3130_313044

theorem parallel_vectors (m n : ℝ × ℝ) : 
  m = (2, 8) → n = (-4, t) → m.1 * n.2 = m.2 * n.1 → t = -16 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l3130_313044


namespace NUMINAMATH_CALUDE_fractional_factorial_max_experiments_l3130_313039

/-- The number of experimental points -/
def n : ℕ := 20

/-- The maximum number of experiments needed -/
def max_experiments : ℕ := 6

/-- Theorem stating that for 20 experimental points, 
    the maximum number of experiments needed is 6 
    when using the fractional factorial design method -/
theorem fractional_factorial_max_experiments :
  n = 2^max_experiments - 1 := by sorry

end NUMINAMATH_CALUDE_fractional_factorial_max_experiments_l3130_313039


namespace NUMINAMATH_CALUDE_parallel_tangents_sum_l3130_313084

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1/a) * Real.log x + 1/x - x

theorem parallel_tangents_sum (a : ℝ) (h : a ≥ 3) :
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  (deriv (f a)) x₁ = (deriv (f a)) x₂ ∧
  x₁ + x₂ > 6/5 := by sorry

end NUMINAMATH_CALUDE_parallel_tangents_sum_l3130_313084


namespace NUMINAMATH_CALUDE_chip_credit_card_balance_l3130_313045

/-- Calculates the final balance on a credit card after two months with interest --/
def final_balance (initial_balance : ℝ) (interest_rate : ℝ) (additional_charge : ℝ) : ℝ :=
  let balance_after_first_month := initial_balance * (1 + interest_rate)
  let balance_before_second_interest := balance_after_first_month + additional_charge
  balance_before_second_interest * (1 + interest_rate)

/-- Theorem stating the final balance on Chip's credit card --/
theorem chip_credit_card_balance :
  final_balance 50 0.2 20 = 96 :=
by sorry

end NUMINAMATH_CALUDE_chip_credit_card_balance_l3130_313045


namespace NUMINAMATH_CALUDE_valid_plates_count_l3130_313072

/-- The number of digits available (0-9) -/
def num_digits : ℕ := 10

/-- The number of letters available (A-Z) -/
def num_letters : ℕ := 26

/-- A license plate is valid if it satisfies the given conditions -/
def is_valid_plate (plate : Fin 4 → Char) : Prop :=
  (plate 0).isDigit ∧
  (plate 1).isAlpha ∧
  (plate 2).isAlpha ∧
  (plate 3).isDigit ∧
  plate 0 = plate 3

/-- The number of valid license plates -/
def num_valid_plates : ℕ := num_digits * num_letters * num_letters

theorem valid_plates_count :
  num_valid_plates = 6760 :=
sorry

end NUMINAMATH_CALUDE_valid_plates_count_l3130_313072


namespace NUMINAMATH_CALUDE_simplify_expression_l3130_313037

theorem simplify_expression (y : ℝ) : 7 * y - 3 * y + 9 + 15 = 4 * y + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3130_313037


namespace NUMINAMATH_CALUDE_sum_of_distances_is_ten_l3130_313031

/-- Given a circle tangent to the sides of an angle at points A and B, with a point C on the circle,
    this structure represents the distances and conditions of the problem. -/
structure CircleTangentProblem where
  -- Distance from C to line AB
  h : ℝ
  -- Distance from C to the side of the angle passing through A
  h_A : ℝ
  -- Distance from C to the side of the angle passing through B
  h_B : ℝ
  -- Condition: h is equal to 4
  h_eq_four : h = 4
  -- Condition: One distance is four times the other
  one_distance_four_times_other : h_B = 4 * h_A

/-- The theorem states that under the given conditions, the sum of distances h_A and h_B is 10. -/
theorem sum_of_distances_is_ten (p : CircleTangentProblem) : p.h_A + p.h_B = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_is_ten_l3130_313031


namespace NUMINAMATH_CALUDE_milk_consumption_ratio_l3130_313020

/-- The ratio of Minyoung's milk consumption to Yuna's milk consumption -/
theorem milk_consumption_ratio (minyoung_milk yuna_milk : ℚ) 
  (h1 : minyoung_milk = 10)
  (h2 : yuna_milk = 2/3) :
  minyoung_milk / yuna_milk = 15 := by
sorry

end NUMINAMATH_CALUDE_milk_consumption_ratio_l3130_313020


namespace NUMINAMATH_CALUDE_happy_boys_count_l3130_313064

theorem happy_boys_count (total_children happy_children sad_children neutral_children
                          total_boys total_girls sad_girls neutral_boys : ℕ)
                         (h1 : total_children = 60)
                         (h2 : happy_children = 30)
                         (h3 : sad_children = 10)
                         (h4 : neutral_children = 20)
                         (h5 : total_boys = 17)
                         (h6 : total_girls = 43)
                         (h7 : sad_girls = 4)
                         (h8 : neutral_boys = 5)
                         (h9 : total_children = happy_children + sad_children + neutral_children)
                         (h10 : total_children = total_boys + total_girls) :
  total_boys - (sad_children - sad_girls) - neutral_boys = 6 := by
  sorry

end NUMINAMATH_CALUDE_happy_boys_count_l3130_313064


namespace NUMINAMATH_CALUDE_grapes_purchased_l3130_313016

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℕ := 9

/-- The price of mangoes per kg -/
def mango_price : ℕ := 45

/-- The total amount paid to the shopkeeper -/
def total_paid : ℕ := 965

/-- The amount of grapes purchased in kg -/
def grape_amount : ℕ := 8

theorem grapes_purchased : 
  grape_price * grape_amount + mango_price * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grapes_purchased_l3130_313016


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l3130_313028

/-- Calculate the percentage of loss given the cost price and selling price -/
def percentageLoss (costPrice sellingPrice : ℚ) : ℚ :=
  ((costPrice - sellingPrice) / costPrice) * 100

theorem loss_percentage_calculation (costPrice sellingPrice : ℚ) 
  (h1 : costPrice = 1750)
  (h2 : sellingPrice = 1610) :
  percentageLoss costPrice sellingPrice = 8 := by
  sorry

#eval percentageLoss 1750 1610

end NUMINAMATH_CALUDE_loss_percentage_calculation_l3130_313028


namespace NUMINAMATH_CALUDE_solution_range_l3130_313026

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 < 0) ↔ (a < -4 ∨ a > 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l3130_313026


namespace NUMINAMATH_CALUDE_norwich_carriages_l3130_313059

/-- The number of carriages in each town --/
structure Carriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flying_scotsman : ℕ

/-- The conditions of the carriage problem --/
def carriage_problem (c : Carriages) : Prop :=
  c.euston = c.norfolk + 20 ∧
  c.flying_scotsman = c.norwich + 20 ∧
  c.euston = 130 ∧
  c.euston + c.norfolk + c.norwich + c.flying_scotsman = 460

/-- The theorem stating that Norwich had 100 carriages --/
theorem norwich_carriages :
  ∃ c : Carriages, carriage_problem c ∧ c.norwich = 100 := by
  sorry

end NUMINAMATH_CALUDE_norwich_carriages_l3130_313059


namespace NUMINAMATH_CALUDE_soup_problem_solution_l3130_313099

/-- Represents the number of people a can of soup can feed -/
structure SoupCanCapacity where
  adults : Nat
  children : Nat

/-- Represents the problem scenario -/
structure SoupProblem where
  capacity : SoupCanCapacity
  totalCans : Nat
  childrenFed : Nat

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdultsFed (problem : SoupProblem) : Nat :=
  let cansForChildren := (problem.childrenFed + problem.capacity.children - 1) / problem.capacity.children
  let remainingCans := problem.totalCans - cansForChildren
  remainingCans * problem.capacity.adults

/-- Theorem stating the problem and its solution -/
theorem soup_problem_solution (problem : SoupProblem)
  (h1 : problem.capacity = ⟨4, 6⟩)
  (h2 : problem.totalCans = 7)
  (h3 : problem.childrenFed = 18) :
  remainingAdultsFed problem = 16 := by
  sorry

end NUMINAMATH_CALUDE_soup_problem_solution_l3130_313099


namespace NUMINAMATH_CALUDE_interval_bound_l3130_313077

theorem interval_bound (k m a b : ℝ) 
  (h : ∀ x ∈ Set.Icc a b, |x^2 - k*x - m| ≤ 1) : 
  b - a ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_interval_bound_l3130_313077


namespace NUMINAMATH_CALUDE_equation_solution_l3130_313062

theorem equation_solution :
  ∃! x : ℚ, (x + 2 ≠ 0) ∧ ((x^2 + 2*x + 3) / (x + 2) = x + 4) :=
by
  -- The unique solution is x = -5/4
  use -5/4
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3130_313062


namespace NUMINAMATH_CALUDE_rectangle_covers_ellipse_l3130_313087

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of an ellipse -/
structure Ellipse where
  major_axis : ℝ
  minor_axis : ℝ

/-- Checks if a rectangle can cover an ellipse -/
def can_cover (r : Rectangle) (e : Ellipse) : Prop :=
  r.length ≥ e.minor_axis ∧
  r.width ≥ e.minor_axis ∧
  r.length^2 + r.width^2 ≥ e.major_axis^2 + e.minor_axis^2

/-- The specific rectangle and ellipse from the problem -/
def problem_rectangle : Rectangle := ⟨140, 130⟩
def problem_ellipse : Ellipse := ⟨160, 100⟩

/-- Theorem stating that the problem_rectangle can cover the problem_ellipse -/
theorem rectangle_covers_ellipse : can_cover problem_rectangle problem_ellipse :=
  sorry

end NUMINAMATH_CALUDE_rectangle_covers_ellipse_l3130_313087


namespace NUMINAMATH_CALUDE_max_x_minus_y_l3130_313086

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l3130_313086


namespace NUMINAMATH_CALUDE_same_color_difference_l3130_313003

/-- The set of colors used for coloring integers. -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A function that colors integers with one of four colors. -/
def ColoringFunction := ℤ → Color

/-- Theorem stating the existence of two integers with the same color and specific difference. -/
theorem same_color_difference (f : ColoringFunction) (x y : ℤ) 
  (h_x_odd : Odd x) (h_y_odd : Odd y) (h_x_y_diff : |x| ≠ |y|) :
  ∃ a b : ℤ, f a = f b ∧ (b - a = x ∨ b - a = y ∨ b - a = x + y ∨ b - a = x - y) := by
  sorry

end NUMINAMATH_CALUDE_same_color_difference_l3130_313003


namespace NUMINAMATH_CALUDE_ab_inequality_relationship_l3130_313036

theorem ab_inequality_relationship (a b : ℝ) :
  (∀ a b, a < b ∧ b < 0 → a * b * (a - b) < 0) ∧
  (∃ a b, a * b * (a - b) < 0 ∧ ¬(a < b ∧ b < 0)) :=
by sorry

end NUMINAMATH_CALUDE_ab_inequality_relationship_l3130_313036


namespace NUMINAMATH_CALUDE_bart_firewood_calculation_l3130_313066

/-- The number of logs Bart burns per day -/
def logs_per_day : ℕ := 5

/-- The number of days Bart burns logs (Nov 1 through Feb 28) -/
def total_days : ℕ := 120

/-- The number of trees Bart needs to cut down -/
def trees_cut : ℕ := 8

/-- The number of pieces of firewood Bart gets from one tree -/
def firewood_per_tree : ℕ := total_days * logs_per_day / trees_cut

theorem bart_firewood_calculation :
  firewood_per_tree = 75 :=
sorry

end NUMINAMATH_CALUDE_bart_firewood_calculation_l3130_313066


namespace NUMINAMATH_CALUDE_total_pencils_count_l3130_313006

/-- The number of people in the group -/
def num_people : ℕ := 5

/-- The number of pencils each person has -/
def pencils_per_person : ℕ := 15

/-- The total number of pencils for the group -/
def total_pencils : ℕ := num_people * pencils_per_person

theorem total_pencils_count : total_pencils = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_count_l3130_313006


namespace NUMINAMATH_CALUDE_expected_balls_in_position_l3130_313007

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 5

/-- The probability that a specific ball is chosen for a swap -/
def prob_chosen : ℚ := 2 / 5

/-- The probability that a specific pair is chosen again -/
def prob_same_pair : ℚ := 1 / 5

/-- The probability that a specific ball is not involved in a swap -/
def prob_not_involved : ℚ := 3 / 5

/-- The number of independent transpositions -/
def num_transpositions : ℕ := 2

/-- 
Theorem: Given 5 balls arranged in a circle, with two independent random transpositions 
of adjacent balls, the expected number of balls in their original positions is 2.2.
-/
theorem expected_balls_in_position : 
  let prob_in_position := prob_chosen * prob_same_pair + prob_not_involved ^ num_transpositions
  num_balls * prob_in_position = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_expected_balls_in_position_l3130_313007


namespace NUMINAMATH_CALUDE_plane_relationship_l3130_313079

-- Define the plane and line types
variable (Point : Type) (Vector : Type)
variable (Plane : Type) (Line : Type)

-- Define the containment relation
variable (contains : Plane → Line → Prop)

-- Define the parallel relation for lines and planes
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection relation for planes
variable (intersect_planes : Plane → Plane → Prop)

-- Given conditions
variable (α β : Plane) (a b : Line)
variable (h1 : contains α a)
variable (h2 : contains β b)
variable (h3 : ¬ parallel_lines a b)

-- Theorem statement
theorem plane_relationship :
  parallel_planes α β ∨ intersect_planes α β :=
sorry

end NUMINAMATH_CALUDE_plane_relationship_l3130_313079


namespace NUMINAMATH_CALUDE_max_peak_consumption_for_savings_l3130_313005

/-- Proves that the maximum average monthly electricity consumption during peak hours
    that allows for at least 10% savings on the original electricity cost is ≤ 118 kWh --/
theorem max_peak_consumption_for_savings (
  original_price : ℝ) (peak_price : ℝ) (off_peak_price : ℝ) (total_consumption : ℝ) 
  (h1 : original_price = 0.52)
  (h2 : peak_price = 0.55)
  (h3 : off_peak_price = 0.35)
  (h4 : total_consumption = 200)
  (h5 : 0 < original_price ∧ 0 < peak_price ∧ 0 < off_peak_price)
  (h6 : total_consumption > 0) :
  let peak_consumption := 
    { x : ℝ | x ≥ 0 ∧ x ≤ total_consumption ∧ 
      (peak_price * x + off_peak_price * (total_consumption - x)) ≤ 
      0.9 * (original_price * total_consumption) }
  ∃ max_peak : ℝ, max_peak ∈ peak_consumption ∧ max_peak ≤ 118 ∧ 
    ∀ y ∈ peak_consumption, y ≤ max_peak := by
  sorry

#check max_peak_consumption_for_savings

end NUMINAMATH_CALUDE_max_peak_consumption_for_savings_l3130_313005


namespace NUMINAMATH_CALUDE_solution_to_equation_l3130_313082

theorem solution_to_equation :
  ∃! (x y : ℝ), x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ∧ x = 1/3 ∧ y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3130_313082


namespace NUMINAMATH_CALUDE_potato_price_is_one_l3130_313095

def initial_money : ℚ := 60
def celery_price : ℚ := 5
def cereal_price : ℚ := 12
def cereal_discount : ℚ := 0.5
def bread_price : ℚ := 8
def milk_price : ℚ := 10
def milk_discount : ℚ := 0.1
def num_potatoes : ℕ := 6
def money_left : ℚ := 26

def discounted_price (price : ℚ) (discount : ℚ) : ℚ :=
  price * (1 - discount)

theorem potato_price_is_one :
  let celery_cost := celery_price
  let cereal_cost := discounted_price cereal_price cereal_discount
  let bread_cost := bread_price
  let milk_cost := discounted_price milk_price milk_discount
  let total_cost := celery_cost + cereal_cost + bread_cost + milk_cost
  let potato_coffee_cost := initial_money - money_left
  let potato_cost := potato_coffee_cost - total_cost
  potato_cost / num_potatoes = 1 := by sorry

end NUMINAMATH_CALUDE_potato_price_is_one_l3130_313095


namespace NUMINAMATH_CALUDE_set_operation_example_l3130_313055

def set_operation (M N : Set ℕ) : Set ℕ :=
  {x | x ∈ M ∪ N ∧ x ∉ M ∩ N}

theorem set_operation_example :
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {2, 3, 4}
  set_operation M N = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_example_l3130_313055


namespace NUMINAMATH_CALUDE_time_after_1567_minutes_l3130_313013

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a day and time -/
structure DayTime where
  days : Nat
  time : Time
  deriving Repr

def addMinutes (start : Time) (minutes : Nat) : DayTime :=
  let totalMinutes := start.minutes + minutes
  let totalHours := start.hours + totalMinutes / 60
  let finalMinutes := totalMinutes % 60
  let finalHours := totalHours % 24
  let days := totalHours / 24
  { days := days
  , time := { hours := finalHours, minutes := finalMinutes } }

theorem time_after_1567_minutes :
  let start := Time.mk 17 0  -- 5:00 p.m.
  let result := addMinutes start 1567
  result = DayTime.mk 1 (Time.mk 19 7)  -- 7:07 p.m. next day
  := by sorry

end NUMINAMATH_CALUDE_time_after_1567_minutes_l3130_313013


namespace NUMINAMATH_CALUDE_parabola_with_focus_at_origin_five_l3130_313042

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  /-- The focus of the parabola -/
  focus : ℝ × ℝ
  /-- The vertex of the parabola -/
  vertex : ℝ × ℝ

/-- The equation of a parabola given its focus and vertex -/
def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  sorry

theorem parabola_with_focus_at_origin_five : 
  let p : Parabola := { focus := (0, 5), vertex := (0, 0) }
  ∀ x y : ℝ, parabola_equation p x y ↔ x^2 = 20*y :=
sorry

end NUMINAMATH_CALUDE_parabola_with_focus_at_origin_five_l3130_313042


namespace NUMINAMATH_CALUDE_divisibility_implication_l3130_313040

theorem divisibility_implication (k : ℕ) : 
  (∃ k, 7^17 + 17 * 3 - 1 = 9 * k) → 
  (∃ m, 7^18 + 18 * 3 - 1 = 9 * m) := by
sorry

end NUMINAMATH_CALUDE_divisibility_implication_l3130_313040


namespace NUMINAMATH_CALUDE_jane_drawing_paper_l3130_313041

/-- The number of old, brown sheets of drawing paper Jane has -/
def brown_sheets : ℕ := 28

/-- The number of old, yellow sheets of drawing paper Jane has -/
def yellow_sheets : ℕ := 27

/-- The total number of drawing paper sheets Jane has -/
def total_sheets : ℕ := brown_sheets + yellow_sheets

theorem jane_drawing_paper : total_sheets = 55 := by
  sorry

end NUMINAMATH_CALUDE_jane_drawing_paper_l3130_313041


namespace NUMINAMATH_CALUDE_integer_area_iff_specific_lengths_l3130_313083

/-- A right triangle with a circumscribed circle -/
structure RightTriangleWithCircle where
  AB : ℝ  -- Length of side AB
  BC : ℝ  -- Length of side BC (diameter of the circle)
  h : AB > 0
  d : BC > 0
  right_angle : AB * BC = AB^2  -- Condition for right angle and tangency

/-- The area of the triangle is an integer -/
def has_integer_area (t : RightTriangleWithCircle) : Prop :=
  ∃ n : ℕ, (1/2) * t.AB * t.BC = n

/-- The main theorem -/
theorem integer_area_iff_specific_lengths (t : RightTriangleWithCircle) :
  has_integer_area t ↔ t.AB ∈ ({4, 8, 12} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_integer_area_iff_specific_lengths_l3130_313083


namespace NUMINAMATH_CALUDE_line_slope_condition_l3130_313043

/-- Given a line passing through points (5, m) and (m, 8), prove that its slope is greater than 1
    if and only if m is in the open interval (5, 13/2). -/
theorem line_slope_condition (m : ℝ) :
  (8 - m) / (m - 5) > 1 ↔ 5 < m ∧ m < 13 / 2 := by
sorry

end NUMINAMATH_CALUDE_line_slope_condition_l3130_313043


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3130_313067

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 15*p^2 + 25*p - 10 = 0 →
  q^3 - 15*q^2 + 25*q - 10 = 0 →
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  p/(2/p + q*r) + q/(2/q + r*p) + r/(2/r + p*q) = 175/12 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3130_313067


namespace NUMINAMATH_CALUDE_parallel_vectors_solution_l3130_313011

def a (x : ℝ) : Fin 3 → ℝ := ![x, 4, 1]
def b (y : ℝ) : Fin 3 → ℝ := ![-2, y, -1]

theorem parallel_vectors_solution (x y : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ a x = k • b y) → x = 2 ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_solution_l3130_313011


namespace NUMINAMATH_CALUDE_sum_of_tens_for_hundred_to_ten_l3130_313096

theorem sum_of_tens_for_hundred_to_ten (n : ℕ) : (100 ^ 10) = n * 10 → n = 10 ^ 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tens_for_hundred_to_ten_l3130_313096


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3130_313098

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 1 < 0) ↔ (∃ x : ℝ, x^2 - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3130_313098


namespace NUMINAMATH_CALUDE_coefficient_x_squared_eq_40_l3130_313076

/-- The coefficient of x^2 in the expansion of (1+2x)^5 -/
def coefficient_x_squared : ℕ :=
  (Nat.choose 5 2) * 2^2

/-- Theorem stating that the coefficient of x^2 in (1+2x)^5 is 40 -/
theorem coefficient_x_squared_eq_40 : coefficient_x_squared = 40 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_eq_40_l3130_313076


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3130_313063

/-- The quadratic function we're minimizing -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The point where the function is minimized -/
def min_point : ℝ := 3

theorem quadratic_minimum :
  ∀ x : ℝ, f x ≥ f min_point :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3130_313063


namespace NUMINAMATH_CALUDE_book_distribution_l3130_313088

theorem book_distribution (total_books : ℕ) (girls boys non_binary : ℕ) 
  (h1 : total_books = 840)
  (h2 : girls = 20)
  (h3 : boys = 15)
  (h4 : non_binary = 5)
  (h5 : ∃ (x : ℕ), 
    girls * (2 * x) + boys * x + non_binary * x = total_books ∧ 
    x > 0) :
  ∃ (books_per_boy : ℕ),
    books_per_boy = 14 ∧
    girls * (2 * books_per_boy) + boys * books_per_boy + non_binary * books_per_boy = total_books :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_l3130_313088


namespace NUMINAMATH_CALUDE_football_game_attendance_difference_l3130_313014

theorem football_game_attendance_difference :
  let saturday : ℕ := 80
  let wednesday (monday : ℕ) : ℕ := monday + 50
  let friday (monday : ℕ) : ℕ := saturday + monday
  let total : ℕ := 390
  ∀ monday : ℕ,
    monday < saturday →
    saturday + monday + wednesday monday + friday monday = total →
    saturday - monday = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_football_game_attendance_difference_l3130_313014


namespace NUMINAMATH_CALUDE_assign_roles_specific_case_l3130_313089

/-- The number of ways to assign roles in a play. -/
def assignRoles (numMen numWomen numMaleRoles numFemaleRoles numEitherRoles : ℕ) : ℕ :=
  (numMen.choose numMaleRoles) *
  (numWomen.choose numFemaleRoles) *
  ((numMen + numWomen - numMaleRoles - numFemaleRoles).choose numEitherRoles)

/-- Theorem stating the number of ways to assign roles in the specific scenario. -/
theorem assign_roles_specific_case :
  assignRoles 7 8 3 3 3 = 35525760 :=
by sorry

end NUMINAMATH_CALUDE_assign_roles_specific_case_l3130_313089


namespace NUMINAMATH_CALUDE_selection_theorem_l3130_313050

/-- The number of volunteers --/
def n : ℕ := 5

/-- The number of days of service --/
def days : ℕ := 2

/-- The number of people selected each day --/
def selected_per_day : ℕ := 2

/-- The number of ways to select exactly one person to serve for both days --/
def selection_ways : ℕ := n * (n - 1) * (n - 2)

theorem selection_theorem : selection_ways = 60 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l3130_313050


namespace NUMINAMATH_CALUDE_intersection_points_count_l3130_313049

/-- A triangle with sides divided into p equal segments, where p is an odd prime -/
structure DividedTriangle where
  p : ℕ
  is_odd_prime : Nat.Prime p ∧ p % 2 = 1

/-- The number of intersection points in a divided triangle -/
def intersection_points (t : DividedTriangle) : ℕ := 3 * (t.p - 1)^2

/-- Theorem: The number of intersection points in a divided triangle is 3(p-1)^2 -/
theorem intersection_points_count (t : DividedTriangle) : 
  intersection_points t = 3 * (t.p - 1)^2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_points_count_l3130_313049


namespace NUMINAMATH_CALUDE_yogurt_combinations_l3130_313085

theorem yogurt_combinations (num_flavors : Nat) (num_toppings : Nat) (num_sizes : Nat) :
  num_flavors = 6 →
  num_toppings = 8 →
  num_sizes = 2 →
  num_flavors * (num_toppings.choose 2) * num_sizes = 336 := by
  sorry

#eval Nat.choose 8 2

end NUMINAMATH_CALUDE_yogurt_combinations_l3130_313085


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3130_313060

theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ (n : ℕ), a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 1/2 →                                -- First term is 1/2
  a 5 = 8 →                                  -- Fifth term is 8
  a 2 * a 3 * a 4 = 8 :=                     -- Product of middle terms is 8
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3130_313060


namespace NUMINAMATH_CALUDE_jerry_spent_difference_l3130_313069

/-- Jerry's initial amount of money in dollars -/
def initial_amount : ℕ := 18

/-- Jerry's remaining amount of money in dollars -/
def remaining_amount : ℕ := 12

/-- The amount Jerry spent on video games -/
def amount_spent : ℕ := initial_amount - remaining_amount

theorem jerry_spent_difference :
  amount_spent = initial_amount - remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_jerry_spent_difference_l3130_313069


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3130_313081

/-- Given an arithmetic sequence {aₙ} where Sₙ denotes the sum of its first n terms,
    if a₄ + a₆ + a₈ = 15, then S₁₁ = 55. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2) →          -- sum formula
  a 4 + a 6 + a 8 = 15 →                                -- given condition
  S 11 = 55 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3130_313081


namespace NUMINAMATH_CALUDE_equation_solutions_l3130_313092

theorem equation_solutions :
  let y₁ : ℝ := (3 + Real.sqrt 15) / 2
  let y₂ : ℝ := (3 - Real.sqrt 15) / 2
  (3 - y₁)^2 + y₁^2 = 12 ∧ (3 - y₂)^2 + y₂^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3130_313092


namespace NUMINAMATH_CALUDE_correct_representation_l3130_313074

/-- Represents "a number that is 3 more than twice x" -/
def number_3_more_than_twice_x (x : ℝ) : ℝ := 2 * x + 3

/-- The algebraic expression 2x + 3 correctly represents "a number that is 3 more than twice x" -/
theorem correct_representation (x : ℝ) :
  number_3_more_than_twice_x x = 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_representation_l3130_313074


namespace NUMINAMATH_CALUDE_algebraic_expansions_l3130_313047

theorem algebraic_expansions (x y : ℝ) :
  ((x + 2*y - 3) * (x - 2*y + 3) = x^2 - 4*y^2 + 12*y - 9) ∧
  ((2*x^3*y)^2 * (-2*x*y) + (-2*x^3*y)^3 / (2*x^2) = -12*x^7*y^3) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expansions_l3130_313047


namespace NUMINAMATH_CALUDE_solve_equation_l3130_313030

-- Define the functions
noncomputable def g : ℝ → ℝ := λ x => Real.log x
noncomputable def f : ℝ → ℝ := λ x => g (-x)

-- State the theorem
theorem solve_equation (m : ℝ) : f m = -1 → m = -1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3130_313030


namespace NUMINAMATH_CALUDE_average_score_is_94_l3130_313071

def june_score : ℝ := 97
def patty_score : ℝ := 85
def josh_score : ℝ := 100
def henry_score : ℝ := 94

def num_children : ℕ := 4

theorem average_score_is_94 :
  (june_score + patty_score + josh_score + henry_score) / num_children = 94 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_94_l3130_313071


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3130_313033

theorem algebraic_expression_equality (x y : ℝ) (h : x + 2*y = 3) : 2*x + 4*y - 7 = -1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3130_313033


namespace NUMINAMATH_CALUDE_factorization_equality_l3130_313065

theorem factorization_equality (x : ℝ) : -x^3 - 2*x^2 - x = -x*(x+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3130_313065


namespace NUMINAMATH_CALUDE_police_speed_l3130_313058

/-- Proves that the speed of a police officer chasing a thief is 40 km/hr given specific conditions --/
theorem police_speed (thief_speed : ℝ) (police_station_distance : ℝ) (police_delay : ℝ) (catch_time : ℝ) :
  thief_speed = 20 →
  police_station_distance = 60 →
  police_delay = 1 →
  catch_time = 4 →
  (police_station_distance + thief_speed * (police_delay + catch_time)) / catch_time = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_police_speed_l3130_313058


namespace NUMINAMATH_CALUDE_fruit_stand_problem_l3130_313038

/-- Represents the fruit stand problem --/
structure FruitStand where
  apple_price : ℝ
  banana_price : ℝ
  orange_price : ℝ
  apple_discount : ℝ
  min_fruit_qty : ℕ
  emmy_budget : ℝ
  gerry_budget : ℝ

/-- Calculates the maximum number of apples that can be bought --/
def max_apples (fs : FruitStand) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem fruit_stand_problem :
  let fs : FruitStand := {
    apple_price := 2,
    banana_price := 1,
    orange_price := 3,
    apple_discount := 0.2,
    min_fruit_qty := 5,
    emmy_budget := 200,
    gerry_budget := 100
  }
  max_apples fs = 160 :=
sorry

end NUMINAMATH_CALUDE_fruit_stand_problem_l3130_313038


namespace NUMINAMATH_CALUDE_hall_to_cube_edge_l3130_313000

theorem hall_to_cube_edge (floor_area : Real) (long_wall_area : Real) (short_wall_area : Real) 
  (h1 : floor_area = 20)
  (h2 : long_wall_area = 10)
  (h3 : short_wall_area = 8) :
  ∃ (cube_edge : Real), cube_edge^3 = floor_area * (long_wall_area * short_wall_area / floor_area) := by
  sorry

end NUMINAMATH_CALUDE_hall_to_cube_edge_l3130_313000


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_l3130_313032

theorem parametric_to_ordinary :
  ∀ θ : ℝ,
  let x : ℝ := 2 + Real.sin θ ^ 2
  let y : ℝ := -1 + Real.cos (2 * θ)
  2 * x + y - 4 = 0 ∧ x ∈ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_l3130_313032


namespace NUMINAMATH_CALUDE_job_completion_time_l3130_313078

/-- Given that person A can complete a job in 18 days and both A and B together can complete it in 10 days, 
    this theorem proves that person B can complete the job alone in 22.5 days. -/
theorem job_completion_time (a_time b_time combined_time : ℝ) 
    (ha : a_time = 18)
    (hc : combined_time = 10)
    (h_combined : 1 / a_time + 1 / b_time = 1 / combined_time) :
    b_time = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3130_313078


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l3130_313035

/-- Given an ellipse with equation x^2 + 9y^2 = 144, the distance between its foci is 16√2 -/
theorem ellipse_focal_distance : 
  ∀ (x y : ℝ), x^2 + 9*y^2 = 144 → 
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = (16 * Real.sqrt 2)^2 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_focal_distance_l3130_313035


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_is_achievable_l3130_313017

theorem min_value_of_function (x : ℝ) : x^2 + 6 / (x^2 + 1) ≥ 2 * Real.sqrt 6 - 1 := by
  sorry

theorem min_value_is_achievable : ∃ x : ℝ, x^2 + 6 / (x^2 + 1) = 2 * Real.sqrt 6 - 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_is_achievable_l3130_313017


namespace NUMINAMATH_CALUDE_min_a_for_nonnegative_f_l3130_313022

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (-3 * x^2 + a * x) - a / x

theorem min_a_for_nonnegative_f :
  ∀ a : ℝ, a > 0 →
  (∃ x₀ : ℝ, f a x₀ ≥ 0) →
  a ≥ 12 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_nonnegative_f_l3130_313022


namespace NUMINAMATH_CALUDE_solve_slurpee_problem_l3130_313070

def slurpee_problem (money_given : ℝ) (change_received : ℝ) (num_slurpees : ℕ) : Prop :=
  let total_spent := money_given - change_received
  let cost_per_slurpee := total_spent / num_slurpees
  cost_per_slurpee = 2

theorem solve_slurpee_problem :
  slurpee_problem 20 8 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_slurpee_problem_l3130_313070


namespace NUMINAMATH_CALUDE_boys_camp_total_l3130_313015

theorem boys_camp_total (total : ℕ) : 
  (total : ℝ) * 0.2 * 0.7 = 63 → total = 450 := by
  sorry

end NUMINAMATH_CALUDE_boys_camp_total_l3130_313015


namespace NUMINAMATH_CALUDE_rectangle_other_vertices_x_sum_l3130_313029

/-- Given a rectangle with two opposite vertices at (2, 23) and (8, -2),
    the sum of the x-coordinates of the other two vertices is 10. -/
theorem rectangle_other_vertices_x_sum :
  ∀ (A B : ℝ × ℝ),
  let v1 : ℝ × ℝ := (2, 23)
  let v2 : ℝ × ℝ := (8, -2)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  (A.1 + B.1) / 2 = midpoint.1 →
  A.1 + B.1 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_other_vertices_x_sum_l3130_313029


namespace NUMINAMATH_CALUDE_english_physical_novels_count_l3130_313010

/-- Represents Iesha's book collection -/
structure BookCollection where
  total : ℕ
  english : ℕ
  school : ℕ
  sports : ℕ
  novels : ℕ
  english_sports : ℕ
  english_school : ℕ
  english_novels : ℕ
  digital_novels : ℕ
  physical_novels : ℕ

/-- Theorem stating the number of English physical format novels in Iesha's collection -/
theorem english_physical_novels_count (c : BookCollection) : c.physical_novels = 135 :=
  by
  have h1 : c.total = 2000 := by sorry
  have h2 : c.english = c.total / 2 := by sorry
  have h3 : c.school = c.total * 30 / 100 := by sorry
  have h4 : c.sports = c.total * 25 / 100 := by sorry
  have h5 : c.novels = c.total - c.school - c.sports := by sorry
  have h6 : c.english_sports = c.english * 10 / 100 := by sorry
  have h7 : c.english_school = c.english * 45 / 100 := by sorry
  have h8 : c.english_novels = c.english - c.english_sports - c.english_school := by sorry
  have h9 : c.digital_novels = c.english_novels * 70 / 100 := by sorry
  have h10 : c.physical_novels = c.english_novels - c.digital_novels := by sorry
  sorry

end NUMINAMATH_CALUDE_english_physical_novels_count_l3130_313010


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l3130_313019

theorem no_solution_quadratic_inequality :
  ∀ x : ℝ, ¬(-x^2 + 2*x - 2 > 0) := by
sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l3130_313019


namespace NUMINAMATH_CALUDE_purple_cell_count_l3130_313001

/-- Represents the state of a cell on the board -/
inductive CellState
| Unpainted
| Blue
| Red
| Purple

/-- Represents a 2x2 square on the board -/
structure Square :=
  (topLeft : Nat × Nat)

/-- Represents the game board -/
def Board := Fin 2022 → Fin 2022 → CellState

/-- Represents a move in the game -/
structure Move :=
  (square : Square)
  (color : CellState)

/-- The game state -/
structure GameState :=
  (board : Board)
  (moves : List Move)

/-- Count the number of purple cells on the board -/
def countPurpleCells (board : Board) : Nat :=
  sorry

/-- Check if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  sorry

/-- Apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

theorem purple_cell_count (finalState : GameState) 
  (h1 : ∀ move ∈ finalState.moves, isValidMove (applyMove finalState move) move)
  (h2 : ∀ i j, finalState.board i j ≠ CellState.Unpainted) :
  countPurpleCells finalState.board = 2022 * 2020 ∨ 
  countPurpleCells finalState.board = 2020 * 2020 :=
sorry

end NUMINAMATH_CALUDE_purple_cell_count_l3130_313001


namespace NUMINAMATH_CALUDE_water_tank_capacity_l3130_313091

theorem water_tank_capacity : ∃ (C : ℝ), 
  (C > 0) ∧ (0.40 * C - 0.25 * C = 36) ∧ (C = 240) := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l3130_313091


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3130_313021

/-- The volume of a sphere with surface area 8π is equal to (8 * sqrt(2) * π) / 3 -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * π * r^2 = 8 * π → (4 / 3) * π * r^3 = (8 * Real.sqrt 2 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3130_313021


namespace NUMINAMATH_CALUDE_road_trip_driving_time_l3130_313094

/-- Calculates the total driving time for a road trip given the number of days and daily driving hours for two people. -/
def total_driving_time (days : ℕ) (person1_hours : ℕ) (person2_hours : ℕ) : ℕ :=
  days * (person1_hours + person2_hours)

/-- Theorem stating that for a 3-day road trip with given driving hours, the total driving time is 42 hours. -/
theorem road_trip_driving_time :
  total_driving_time 3 8 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_driving_time_l3130_313094


namespace NUMINAMATH_CALUDE_mushroom_collection_l3130_313054

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem mushroom_collection :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ sum_of_digits n = 14 ∧ n % 50 = 0 :=
by
  -- The proof would go here
  sorry

#eval sum_of_digits 950  -- Should output 14
#eval 950 % 50           -- Should output 0

end NUMINAMATH_CALUDE_mushroom_collection_l3130_313054


namespace NUMINAMATH_CALUDE_one_student_reviewed_l3130_313025

/-- Represents the students in the problem -/
inductive Student : Type
  | Zhang
  | Li
  | Wang
  | Zhao
  | Liu

/-- The statement made by each student about how many reviewed math -/
def statement (s : Student) : Nat :=
  match s with
  | Student.Zhang => 0
  | Student.Li => 1
  | Student.Wang => 2
  | Student.Zhao => 3
  | Student.Liu => 4

/-- Predicate to determine if a student reviewed math -/
def reviewed : Student → Prop := sorry

/-- The number of students who reviewed math -/
def num_reviewed : Nat := sorry

theorem one_student_reviewed :
  (∃ s : Student, reviewed s) ∧
  (∃ s : Student, ¬reviewed s) ∧
  (∀ s : Student, reviewed s ↔ statement s = num_reviewed) ∧
  (num_reviewed = 1) := by sorry

end NUMINAMATH_CALUDE_one_student_reviewed_l3130_313025


namespace NUMINAMATH_CALUDE_total_money_proof_l3130_313048

def sam_money : ℕ := 75

def billy_money (sam : ℕ) : ℕ := 2 * sam - 25

def total_money (sam : ℕ) : ℕ := sam + billy_money sam

theorem total_money_proof : total_money sam_money = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_money_proof_l3130_313048


namespace NUMINAMATH_CALUDE_second_greatest_number_l3130_313090

def digits : List Nat := [4, 3, 1, 7, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 10) % 10 = 3 ∧
  (∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ 3 ∧ b ≠ 3 ∧ n = 100 * a + 30 + b)

def is_second_greatest (n : Nat) : Prop :=
  is_valid_number n ∧
  (∃ (m : Nat), is_valid_number m ∧ m > n) ∧
  (∀ (k : Nat), is_valid_number k ∧ k ≠ n → k ≤ n ∨ k > n ∧ (∃ (m : Nat), is_valid_number m ∧ m > n ∧ m < k))

theorem second_greatest_number : 
  ∃ (n : Nat), is_second_greatest n ∧ n = 934 := by sorry

end NUMINAMATH_CALUDE_second_greatest_number_l3130_313090


namespace NUMINAMATH_CALUDE_systems_solution_l3130_313097

theorem systems_solution : ∃ (x y : ℝ), 
  (x - y = 1 ∧ 3*x + y = 11) ∧ 
  (3*x - 2*y = 5 ∧ 2*x + 3*y = 12) ∧
  (x = 3 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_systems_solution_l3130_313097


namespace NUMINAMATH_CALUDE_min_value_triangle_sides_l3130_313034

theorem min_value_triangle_sides (a b c : ℤ) : 
  a < b → b < c → a + b + c = 30 → 
  ∀ x y z : ℤ, x < y → y < z → x + y + z = 30 → 
  c^2 + 18*a + 18*b - 446 ≥ 17 :=
by sorry

end NUMINAMATH_CALUDE_min_value_triangle_sides_l3130_313034


namespace NUMINAMATH_CALUDE_distance_between_cities_l3130_313024

/-- The distance between two cities given the speeds of two cars and their time difference --/
theorem distance_between_cities (v1 v2 : ℝ) (t_diff : ℝ) (h1 : v1 = 60) (h2 : v2 = 70) (h3 : t_diff = 0.25) :
  ∃ d : ℝ, d = 105 ∧ d = v1 * (d / v1) ∧ d = v2 * (d / v2 - t_diff) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l3130_313024


namespace NUMINAMATH_CALUDE_remaining_hard_hats_l3130_313009

/-- Represents the number of hard hats in the truck -/
structure HardHats :=
  (pink : ℕ)
  (green : ℕ)
  (yellow : ℕ)

/-- Calculates the total number of hard hats -/
def totalHardHats (hats : HardHats) : ℕ :=
  hats.pink + hats.green + hats.yellow

/-- Represents the actions of Carl and John -/
def removeHardHats (initial : HardHats) : HardHats :=
  let afterCarl := HardHats.mk (initial.pink - 4) initial.green initial.yellow
  let johnPinkRemoval := 6
  HardHats.mk 
    (afterCarl.pink - johnPinkRemoval)
    (afterCarl.green - 2 * johnPinkRemoval)
    afterCarl.yellow

/-- The main theorem to prove -/
theorem remaining_hard_hats (initial : HardHats) 
  (h1 : initial.pink = 26) 
  (h2 : initial.green = 15) 
  (h3 : initial.yellow = 24) :
  totalHardHats (removeHardHats initial) = 43 := by
  sorry

end NUMINAMATH_CALUDE_remaining_hard_hats_l3130_313009


namespace NUMINAMATH_CALUDE_fruits_in_red_basket_l3130_313052

theorem fruits_in_red_basket :
  let blue_bananas : ℕ := 12
  let blue_apples : ℕ := 4
  let blue_total : ℕ := blue_bananas + blue_apples
  let red_total : ℕ := blue_total / 2
  red_total = 8 := by sorry

end NUMINAMATH_CALUDE_fruits_in_red_basket_l3130_313052


namespace NUMINAMATH_CALUDE_chocolate_game_student_count_l3130_313004

def is_valid_student_count (n : ℕ) : Prop :=
  n > 0 ∧ (120 - 1) % n = 0

theorem chocolate_game_student_count :
  {n : ℕ | is_valid_student_count n} = {7, 17} := by
  sorry

end NUMINAMATH_CALUDE_chocolate_game_student_count_l3130_313004


namespace NUMINAMATH_CALUDE_signup_ways_4_3_l3130_313027

/-- The number of ways 4 students can sign up for one of 3 interest groups -/
def signup_ways (num_students : ℕ) (num_groups : ℕ) : ℕ :=
  num_groups ^ num_students

/-- Theorem stating that the number of ways 4 students can sign up for one of 3 interest groups is 81 -/
theorem signup_ways_4_3 : signup_ways 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_signup_ways_4_3_l3130_313027


namespace NUMINAMATH_CALUDE_average_weight_of_children_l3130_313053

def ages : List ℝ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

def regression_equation (x : ℝ) : ℝ := 2 * x + 7

theorem average_weight_of_children :
  let avg_age := (ages.sum) / (ages.length : ℝ)
  regression_equation avg_age = 15 := by sorry

end NUMINAMATH_CALUDE_average_weight_of_children_l3130_313053
