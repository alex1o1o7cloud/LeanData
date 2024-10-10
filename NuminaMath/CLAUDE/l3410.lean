import Mathlib

namespace arithmetic_mean_problem_l3410_341078

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 27)
  (h3 : r - p = 34) : 
  (p + q) / 2 = 10 := by
sorry

end arithmetic_mean_problem_l3410_341078


namespace parallel_vectors_m_equals_one_l3410_341073

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_equals_one :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (m, m - 4)
  are_parallel a b → m = 1 := by
  sorry

end parallel_vectors_m_equals_one_l3410_341073


namespace coefficient_of_x4_in_expansion_l3410_341090

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def expansion_coefficient (n : ℕ) : ℕ :=
  binomial_coefficient n 4

theorem coefficient_of_x4_in_expansion : 
  expansion_coefficient 5 + expansion_coefficient 6 + expansion_coefficient 7 = 55 := by sorry

end coefficient_of_x4_in_expansion_l3410_341090


namespace ice_cream_cup_cost_l3410_341039

/-- Given Alok's order and payment, prove the cost of each ice-cream cup --/
theorem ice_cream_cup_cost
  (chapati_count : ℕ)
  (rice_count : ℕ)
  (vegetable_count : ℕ)
  (ice_cream_count : ℕ)
  (chapati_cost : ℕ)
  (rice_cost : ℕ)
  (vegetable_cost : ℕ)
  (total_paid : ℕ)
  (h1 : chapati_count = 16)
  (h2 : rice_count = 5)
  (h3 : vegetable_count = 7)
  (h4 : ice_cream_count = 6)
  (h5 : chapati_cost = 6)
  (h6 : rice_cost = 45)
  (h7 : vegetable_cost = 70)
  (h8 : total_paid = 1021) :
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 35 := by
  sorry

end ice_cream_cup_cost_l3410_341039


namespace bakery_rolls_combinations_l3410_341044

theorem bakery_rolls_combinations :
  let total_rolls : ℕ := 9
  let kinds_of_rolls : ℕ := 4
  let min_per_kind : ℕ := 1
  let remaining_rolls : ℕ := total_rolls - kinds_of_rolls * min_per_kind
  Nat.choose (kinds_of_rolls + remaining_rolls - 1) remaining_rolls = 56 := by
  sorry

end bakery_rolls_combinations_l3410_341044


namespace two_year_increase_l3410_341025

/-- 
Given an initial amount that increases by 1/8th of itself each year, 
this theorem proves that after two years, the amount will be as calculated.
-/
theorem two_year_increase (initial_amount : ℝ) : 
  initial_amount = 70400 → 
  (initial_amount * (9/8) * (9/8) : ℝ) = 89070 := by
  sorry

end two_year_increase_l3410_341025


namespace floor_inequality_l3410_341041

theorem floor_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ⌊5 * x⌋ + ⌊5 * y⌋ ≥ ⌊3 * x + y⌋ + ⌊3 * y + x⌋ := by
  sorry

#check floor_inequality

end floor_inequality_l3410_341041


namespace fraction_inequality_l3410_341081

def numerator (x : ℝ) : ℝ := 7 * x - 3

def denominator (x : ℝ) : ℝ := x^2 - x - 12

def valid_x (x : ℝ) : Prop := denominator x ≠ 0

def inequality_holds (x : ℝ) : Prop := numerator x ≥ denominator x

def solution_set : Set ℝ := {x | x ∈ Set.Icc (-1) 3 ∪ Set.Ioo 3 4 ∪ Set.Ico 4 9}

theorem fraction_inequality :
  {x : ℝ | inequality_holds x ∧ valid_x x} = solution_set := by sorry

end fraction_inequality_l3410_341081


namespace hyperbola_probability_l3410_341067

-- Define the set of possible values for m and n
def S : Set ℕ := {1, 2, 3}

-- Define the condition for (m, n) to be on the hyperbola
def on_hyperbola (m n : ℕ) : Prop := n = 6 / m

-- Define the probability space
def total_outcomes : ℕ := 6

-- Define the favorable outcomes
def favorable_outcomes : ℕ := 2

-- State the theorem
theorem hyperbola_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by sorry

end hyperbola_probability_l3410_341067


namespace cookie_bags_l3410_341072

theorem cookie_bags (cookies_per_bag : ℕ) (total_cookies : ℕ) (h1 : cookies_per_bag = 41) (h2 : total_cookies = 2173) :
  total_cookies / cookies_per_bag = 53 := by
  sorry

end cookie_bags_l3410_341072


namespace simplify_expression_l3410_341083

theorem simplify_expression : 18 * (7 / 12) * (1 / 6) + 1 / 4 = 2 := by
  sorry

end simplify_expression_l3410_341083


namespace marble_probability_l3410_341069

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) : 
  total = 120 → 
  p_white = 1/4 → 
  p_green = 1/3 → 
  ∃ (p_red_blue : ℚ), p_red_blue = 5/12 ∧ p_white + p_green + p_red_blue = 1 :=
by sorry

end marble_probability_l3410_341069


namespace exists_set_with_divisibility_property_l3410_341091

theorem exists_set_with_divisibility_property (n : ℕ) :
  ∃ (S : Finset ℕ), S.card = n ∧
    ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b →
      (max a b - min a b) ∣ max a b :=
sorry

end exists_set_with_divisibility_property_l3410_341091


namespace function_range_complement_l3410_341023

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 2

theorem function_range_complement :
  {k : ℝ | ∀ x, f x ≠ k} = Set.Iio (-3) :=
by sorry

end function_range_complement_l3410_341023


namespace empty_set_condition_l3410_341076

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | Real.sqrt (x - 3) = a * x + 1}

-- State the theorem
theorem empty_set_condition (a : ℝ) :
  IsEmpty (A a) ↔ a < -1/2 ∨ a > 1/6 := by sorry

end empty_set_condition_l3410_341076


namespace colored_square_theorem_l3410_341051

/-- Represents a coloring of a square grid -/
def Coloring (n : ℕ) := Fin (n^2 + 1) → Fin (n^2 + 1)

/-- Counts the number of distinct colors in a given row or column -/
def distinctColors (n : ℕ) (c : Coloring n) (isRow : Bool) (index : Fin (n^2 + 1)) : ℕ :=
  sorry

theorem colored_square_theorem (n : ℕ) (c : Coloring n) :
  ∃ (isRow : Bool) (index : Fin (n^2 + 1)), distinctColors n c isRow index ≥ n + 1 :=
sorry

end colored_square_theorem_l3410_341051


namespace circle_center_radius_sum_l3410_341082

/-- Given a circle D with equation x^2 - 8x + y^2 + 14y = -28,
    prove that the sum of its center coordinates and radius is -3 + √37 -/
theorem circle_center_radius_sum :
  let D : Set (ℝ × ℝ) := {p | (p.1^2 - 8*p.1 + p.2^2 + 14*p.2 = -28)}
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ D ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c + d + s = -3 + Real.sqrt 37 :=
by sorry

end circle_center_radius_sum_l3410_341082


namespace bouquet_cost_45_lilies_l3410_341004

/-- Represents the cost of a bouquet of lilies -/
def bouquet_cost (num_lilies : ℕ) : ℚ :=
  let base_price_per_lily : ℚ := 30 / 15
  let discount_threshold : ℕ := 30
  let discount_rate : ℚ := 1 / 10
  if num_lilies ≤ discount_threshold then
    num_lilies * base_price_per_lily
  else
    num_lilies * (base_price_per_lily * (1 - discount_rate))

theorem bouquet_cost_45_lilies :
  bouquet_cost 45 = 81 := by
  sorry

end bouquet_cost_45_lilies_l3410_341004


namespace system_solution_l3410_341022

theorem system_solution (x y z : ℚ) : 
  (x * y = 6 * (x + y) ∧ 
   x * z = 4 * (x + z) ∧ 
   y * z = 2 * (y + z)) ↔ 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = -24 ∧ y = 24/5 ∧ z = 24/7)) := by
sorry

end system_solution_l3410_341022


namespace sequence_2011th_term_l3410_341096

theorem sequence_2011th_term (a : ℕ → ℝ) 
  (h1 : a 1 = 0)
  (h2 : ∀ n : ℕ, a n + a (n + 1) = 2) : 
  a 2011 = 0 := by
sorry

end sequence_2011th_term_l3410_341096


namespace subtracted_value_l3410_341080

theorem subtracted_value (n : ℝ) (v : ℝ) (h1 : n = 1) (h2 : 3 * n - v = 2 * n) : v = 1 := by
  sorry

end subtracted_value_l3410_341080


namespace victoria_wheat_flour_packets_l3410_341098

/-- Calculates the number of wheat flour packets bought given the initial amount,
    costs of items, and remaining balance. -/
def wheat_flour_packets (initial_amount : ℕ) (rice_cost : ℕ) (rice_packets : ℕ) 
                        (soda_cost : ℕ) (wheat_flour_cost : ℕ) (remaining_balance : ℕ) : ℕ :=
  let total_spent := initial_amount - remaining_balance
  let rice_soda_cost := rice_cost * rice_packets + soda_cost
  let wheat_flour_total := total_spent - rice_soda_cost
  wheat_flour_total / wheat_flour_cost

/-- Theorem stating that Victoria bought 3 packets of wheat flour -/
theorem victoria_wheat_flour_packets : 
  wheat_flour_packets 500 20 2 150 25 235 = 3 := by
  sorry

end victoria_wheat_flour_packets_l3410_341098


namespace equation_transformation_l3410_341053

theorem equation_transformation (x y : ℝ) :
  (2 * x - 3 * y = 6) ↔ (y = (2 * x - 6) / 3) := by
  sorry

end equation_transformation_l3410_341053


namespace election_result_l3410_341046

/-- Represents the result of an election with three candidates. -/
structure ElectionResult where
  totalVotes : ℕ
  votesA : ℕ
  votesB : ℕ
  votesC : ℕ

/-- Theorem stating the correct election results given the conditions. -/
theorem election_result : ∃ (result : ElectionResult),
  result.totalVotes = 10000 ∧
  result.votesA = 3400 ∧
  result.votesB = 4800 ∧
  result.votesC = 2900 ∧
  result.votesA = (34 * result.totalVotes) / 100 ∧
  result.votesB = (48 * result.totalVotes) / 100 ∧
  result.votesB = result.votesA + 1400 ∧
  result.votesA = result.votesC + 500 ∧
  result.totalVotes = result.votesA + result.votesB + result.votesC :=
by
  sorry

#check election_result

end election_result_l3410_341046


namespace meat_for_35_tacos_l3410_341005

/-- The amount of meat (in pounds) needed to make a given number of tacos, 
    given that 4 pounds of meat make 10 tacos -/
def meat_needed (tacos : ℕ) : ℚ :=
  (4 : ℚ) * tacos / 10

theorem meat_for_35_tacos : meat_needed 35 = 14 := by
  sorry

end meat_for_35_tacos_l3410_341005


namespace impossibility_of_crossing_plan_l3410_341042

/-- Represents a group of friends -/
def FriendGroup := Finset (Fin 5)

/-- The set of all possible non-empty groups of friends -/
def AllGroups : Set FriendGroup :=
  {g : FriendGroup | g.Nonempty}

/-- A crossing plan is a function that assigns each group to a number of crossings -/
def CrossingPlan := FriendGroup → ℕ

/-- A valid crossing plan assigns exactly one crossing to each non-empty group -/
def IsValidPlan (plan : CrossingPlan) : Prop :=
  ∀ g : FriendGroup, g ∈ AllGroups → plan g = 1

theorem impossibility_of_crossing_plan :
  ¬∃ (plan : CrossingPlan), IsValidPlan plan :=
sorry

end impossibility_of_crossing_plan_l3410_341042


namespace right_angles_in_five_days_l3410_341087

/-- The number of times clock hands form a right angle in a 12-hour period -/
def right_angles_per_12hours : ℕ := 22

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days we're considering -/
def days : ℕ := 5

/-- Theorem: The number of times clock hands form a right angle in 5 days is 220 -/
theorem right_angles_in_five_days :
  (right_angles_per_12hours * 2 * days) = 220 := by sorry

end right_angles_in_five_days_l3410_341087


namespace max_profit_theorem_l3410_341048

/-- Represents the daily sales volume as a function of unit price -/
def sales_volume (x : ℝ) : ℝ := -100 * x + 5000

/-- Represents the daily profit as a function of unit price -/
def daily_profit (x : ℝ) : ℝ := (sales_volume x) * (x - 6)

/-- The theorem stating the maximum profit and the price at which it occurs -/
theorem max_profit_theorem :
  let x_min : ℝ := 6
  let x_max : ℝ := 32
  ∀ x ∈ Set.Icc x_min x_max,
    daily_profit x ≤ daily_profit 28 ∧
    daily_profit 28 = 48400 := by
  sorry

end max_profit_theorem_l3410_341048


namespace min_max_values_of_f_l3410_341028

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_of_f :
  ∃ (min_val max_val : ℝ),
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ min_val) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = min_val) ∧
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ max_val) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = max_val) ∧
    min_val = -3 * Real.pi / 2 ∧
    max_val = Real.pi / 2 + 2 :=
by sorry

end min_max_values_of_f_l3410_341028


namespace consecutive_points_segment_length_l3410_341059

/-- Given 5 consecutive points on a line, prove the length of the last segment -/
theorem consecutive_points_segment_length 
  (a b c d e : ℝ) -- Points represented as real numbers
  (h_consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Ensures points are consecutive
  (h_bc_cd : c - b = 2 * (d - c)) -- bc = 2 cd
  (h_ab : b - a = 5) -- ab = 5
  (h_ac : c - a = 11) -- ac = 11
  (h_ae : e - a = 22) -- ae = 22
  : e - d = 8 := by sorry

end consecutive_points_segment_length_l3410_341059


namespace base7_product_digit_sum_l3410_341008

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 9 --/
def decimalToBase9 (n : ℕ) : List ℕ := sorry

/-- Calculates the sum of digits in a list --/
def sumDigits (digits : List ℕ) : ℕ := sorry

/-- Theorem statement --/
theorem base7_product_digit_sum :
  let a := base7ToDecimal 34
  let b := base7ToDecimal 52
  let product := a * b
  let base9Product := decimalToBase9 product
  sumDigits base9Product = 10 := by sorry

end base7_product_digit_sum_l3410_341008


namespace complex_multiplication_sum_l3410_341054

theorem complex_multiplication_sum (a b : ℝ) (i : ℂ) : 
  i ^ 2 = -1 → 
  a + b * i = (1 + i) * (2 - i) → 
  a + b = 4 := by
  sorry

end complex_multiplication_sum_l3410_341054


namespace smallest_integer_gcd_lcm_problem_l3410_341052

theorem smallest_integer_gcd_lcm_problem (x : ℕ) (a b : ℕ) : 
  x > 0 →
  a > 0 →
  b > 0 →
  a = 72 →
  Nat.gcd a b = x + 6 →
  Nat.lcm a b = x * (x + 6) →
  ∃ m : ℕ, (∀ n : ℕ, n > 0 ∧ 
    Nat.gcd 72 n = x + 6 ∧ 
    Nat.lcm 72 n = x * (x + 6) → 
    m ≤ n) ∧ 
    m = 12 :=
by sorry

end smallest_integer_gcd_lcm_problem_l3410_341052


namespace problem_solution_l3410_341085

def problem (A B X : ℕ) : Prop :=
  A > 0 ∧ B > 0 ∧
  Nat.gcd A B = 20 ∧
  A = 300 ∧
  Nat.lcm A B = 20 * X * 15

theorem problem_solution :
  ∀ A B X, problem A B X → X = 1 := by
  sorry

end problem_solution_l3410_341085


namespace parallelogram_point_B_trajectory_l3410_341075

-- Define the parallelogram ABCD
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

-- Define the coordinates of points A and C
def A : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (2, -3)

-- Define the line on which D moves
def D_line (x y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the trajectory of point B
def B_trajectory (x y : ℝ) : Prop := 3 * x - y - 20 = 0 ∧ x ≠ 3

-- Theorem statement
theorem parallelogram_point_B_trajectory 
  (ABCD : Parallelogram) 
  (h1 : ABCD.A = A) 
  (h2 : ABCD.C = C) 
  (h3 : ∀ x y, ABCD.D = (x, y) → D_line x y) :
  ∀ x y, ABCD.B = (x, y) → B_trajectory x y :=
sorry

end parallelogram_point_B_trajectory_l3410_341075


namespace root_implies_a_value_l3410_341000

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x - 48

-- State the theorem
theorem root_implies_a_value (a b : ℚ) :
  f a b (-2 - 3 * Real.sqrt 3) = 0 → a = 44 / 23 := by
  sorry

end root_implies_a_value_l3410_341000


namespace amy_flash_drive_files_l3410_341089

/-- Calculates the number of remaining files on Amy's flash drive -/
def remainingFiles (musicFiles videoFiles deletedFiles : ℕ) : ℕ :=
  musicFiles + videoFiles - deletedFiles

/-- Theorem stating the number of remaining files on Amy's flash drive -/
theorem amy_flash_drive_files : remainingFiles 26 36 48 = 14 := by
  sorry

end amy_flash_drive_files_l3410_341089


namespace product_of_numbers_l3410_341092

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 106) :
  x * y = 32 := by
sorry

end product_of_numbers_l3410_341092


namespace floor_neg_sqrt_64_over_9_l3410_341060

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by sorry

end floor_neg_sqrt_64_over_9_l3410_341060


namespace geometric_sequence_nth_term_l3410_341015

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_nth_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 2 + a 4 = 5) :
  ∃ q : ℝ, ∀ n : ℕ, a n = 2^(4 - n) :=
sorry

end geometric_sequence_nth_term_l3410_341015


namespace quadratic_roots_imply_m_negative_l3410_341074

/-- If the equation 2x^2 + (m+1)x + m = 0 has one positive root and one negative root, then m < 0 -/
theorem quadratic_roots_imply_m_negative (m : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ 2 * x^2 + (m + 1) * x + m = 0 ∧ 2 * y^2 + (m + 1) * y + m = 0) →
  m < 0 :=
by sorry

end quadratic_roots_imply_m_negative_l3410_341074


namespace parabolas_intersection_l3410_341007

def parabola1 (x y : ℝ) : Prop := y = 3 * x^2 - 4 * x + 2
def parabola2 (x y : ℝ) : Prop := y = x^3 - 2 * x^2 + x + 2

def intersection_points : Set (ℝ × ℝ) :=
  {(0, 2),
   ((5 + Real.sqrt 5) / 2, 3 * ((5 + Real.sqrt 5) / 2)^2 - 4 * ((5 + Real.sqrt 5) / 2) + 2),
   ((5 - Real.sqrt 5) / 2, 3 * ((5 - Real.sqrt 5) / 2)^2 - 4 * ((5 - Real.sqrt 5) / 2) + 2)}

theorem parabolas_intersection :
  ∀ x y : ℝ, (parabola1 x y ∧ parabola2 x y) ↔ (x, y) ∈ intersection_points := by
  sorry

end parabolas_intersection_l3410_341007


namespace field_path_area_and_cost_l3410_341079

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

/-- Theorem: For a 60m x 55m field with a 2.5m wide path, the path area is 600 sq m
    and the construction cost at Rs. 2 per sq m is Rs. 1200 -/
theorem field_path_area_and_cost :
  let field_length : ℝ := 60
  let field_width : ℝ := 55
  let path_width : ℝ := 2.5
  let cost_per_unit : ℝ := 2
  (path_area field_length field_width path_width = 600) ∧
  (construction_cost (path_area field_length field_width path_width) cost_per_unit = 1200) :=
by sorry

end field_path_area_and_cost_l3410_341079


namespace complex_calculation_l3410_341014

theorem complex_calculation (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 2 + 4*I) :
  3*a - 4*b = 7 - 25*I :=
by sorry

end complex_calculation_l3410_341014


namespace annie_village_trick_or_treat_l3410_341009

/-- The number of blocks in Annie's village -/
def num_blocks : ℕ := 9

/-- The number of children on each block -/
def children_per_block : ℕ := 6

/-- The total number of children going trick or treating in Annie's village -/
def total_children : ℕ := num_blocks * children_per_block

theorem annie_village_trick_or_treat : total_children = 54 := by
  sorry

end annie_village_trick_or_treat_l3410_341009


namespace f_extrema_l3410_341050

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x + 1

theorem f_extrema : 
  (∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ f x) ∧
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ f x) := by
  sorry

end f_extrema_l3410_341050


namespace initial_average_marks_l3410_341055

theorem initial_average_marks (n : ℕ) (wrong_mark correct_mark : ℝ) (correct_avg : ℝ) :
  n = 10 →
  wrong_mark = 50 →
  correct_mark = 10 →
  correct_avg = 96 →
  (n * correct_avg * n - (wrong_mark - correct_mark)) / n = 92 :=
by
  sorry

end initial_average_marks_l3410_341055


namespace inequality_proof_l3410_341027

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  1 < (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) ∧ 
  (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) < 2 :=
by sorry

end inequality_proof_l3410_341027


namespace evaluate_expression_l3410_341063

/-- Given x = -1 and y = 2, prove that -2x²y-3(2xy-x²y)+4xy evaluates to 6 -/
theorem evaluate_expression (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  -2 * x^2 * y - 3 * (2 * x * y - x^2 * y) + 4 * x * y = 6 := by
  sorry

end evaluate_expression_l3410_341063


namespace speed_ratio_in_race_l3410_341012

/-- In a race, contestant A has a head start and wins. This theorem proves the ratio of their speeds. -/
theorem speed_ratio_in_race (total_distance : ℝ) (head_start : ℝ) (win_margin : ℝ)
  (h1 : total_distance = 500)
  (h2 : head_start = 300)
  (h3 : win_margin = 100)
  : (total_distance - head_start) / (total_distance - win_margin) = 1 / 2 := by
  sorry

end speed_ratio_in_race_l3410_341012


namespace sector_arc_length_l3410_341026

theorem sector_arc_length (central_angle : Real) (radius : Real) 
  (h1 : central_angle = 1/5)
  (h2 : radius = 5) : 
  central_angle * radius = 1 := by
  sorry

end sector_arc_length_l3410_341026


namespace ducks_to_chickens_ratio_l3410_341032

/-- Represents the farm animals --/
structure Farm :=
  (chickens : ℕ)
  (ducks : ℕ)
  (turkeys : ℕ)

/-- The conditions of Mr. Valentino's farm --/
def valentino_farm (f : Farm) : Prop :=
  f.chickens = 200 ∧
  f.turkeys = 3 * f.ducks ∧
  f.chickens + f.ducks + f.turkeys = 1800

/-- The theorem stating the ratio of ducks to chickens --/
theorem ducks_to_chickens_ratio (f : Farm) :
  valentino_farm f → (f.ducks : ℚ) / f.chickens = 2 := by
  sorry

#check ducks_to_chickens_ratio

end ducks_to_chickens_ratio_l3410_341032


namespace kelly_games_given_away_l3410_341095

/-- Given that Kelly initially had 50 Nintendo games and now has 35 games left,
    prove that she gave away 15 games. -/
theorem kelly_games_given_away :
  let initial_games : ℕ := 50
  let remaining_games : ℕ := 35
  let games_given_away := initial_games - remaining_games
  games_given_away = 15 :=
by sorry

end kelly_games_given_away_l3410_341095


namespace min_value_theorem_l3410_341038

theorem min_value_theorem (a : ℝ) (h : a > 2) :
  a + 1 / (a - 2) ≥ 4 ∧ (a + 1 / (a - 2) = 4 ↔ a = 3) := by sorry

end min_value_theorem_l3410_341038


namespace max_value_x_sqrt_3_minus_x_squared_l3410_341036

theorem max_value_x_sqrt_3_minus_x_squared (x : ℝ) (h1 : 0 < x) (h2 : x < Real.sqrt 3) :
  (∀ y, 0 < y ∧ y < Real.sqrt 3 → x * Real.sqrt (3 - x^2) ≥ y * Real.sqrt (3 - y^2)) →
  x * Real.sqrt (3 - x^2) = 3/2 :=
sorry

end max_value_x_sqrt_3_minus_x_squared_l3410_341036


namespace divisor_problem_l3410_341043

theorem divisor_problem : ∃ (d : ℕ), d > 0 ∧ (10154 - 14) % d = 0 ∧ d = 10140 :=
by
  sorry

end divisor_problem_l3410_341043


namespace probability_diamond_is_one_fourth_l3410_341034

/-- A special deck of cards -/
structure SpecialDeck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h1 : total_cards = num_ranks * num_suits)
  (h2 : cards_per_suit = num_ranks)

/-- The probability of drawing a diamond from the special deck -/
def probability_diamond (deck : SpecialDeck) : ℚ :=
  deck.cards_per_suit / deck.total_cards

/-- Theorem stating that the probability of drawing a diamond is 1/4 -/
theorem probability_diamond_is_one_fourth (deck : SpecialDeck) 
  (h3 : deck.num_suits = 4) : 
  probability_diamond deck = 1/4 := by
  sorry

#check probability_diamond_is_one_fourth

end probability_diamond_is_one_fourth_l3410_341034


namespace units_digit_of_n_l3410_341031

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 31^8 → 
  m % 10 = 7 → 
  n % 10 = 3 := by sorry

end units_digit_of_n_l3410_341031


namespace unique_solution_logarithmic_system_l3410_341066

theorem unique_solution_logarithmic_system :
  ∃! (x y : ℝ), 
    Real.log (x^2 + y^2) / Real.log 10 = 1 + Real.log 8 / Real.log 10 ∧
    Real.log (x + y) / Real.log 10 - Real.log (x - y) / Real.log 10 = Real.log 3 / Real.log 10 ∧
    x + y > 0 ∧
    x - y > 0 ∧
    x = 8 ∧
    y = 4 :=
by
  sorry

end unique_solution_logarithmic_system_l3410_341066


namespace bug_meeting_point_l3410_341020

/-- Triangle PQR with side lengths PQ=6, QR=8, and PR=9 -/
structure Triangle :=
  (PQ : ℝ) (QR : ℝ) (PR : ℝ)

/-- Two bugs crawling along the perimeter of the triangle -/
structure BugMeeting (t : Triangle) :=
  (S : ℝ)  -- Position of meeting point S on side QR

/-- Main theorem: QS = 5.5 when bugs meet -/
theorem bug_meeting_point (t : Triangle) (b : BugMeeting t) :
  t.PQ = 6 → t.QR = 8 → t.PR = 9 → b.S = 5.5 := by
  sorry

#check bug_meeting_point

end bug_meeting_point_l3410_341020


namespace staircase_steps_l3410_341003

def jumps (step_size : ℕ) (total_steps : ℕ) : ℕ :=
  (total_steps + step_size - 1) / step_size

theorem staircase_steps : ∃ (n : ℕ), n > 0 ∧ jumps 3 n - jumps 4 n = 10 ∧ n = 120 := by
  sorry

end staircase_steps_l3410_341003


namespace min_days_for_progress_ratio_l3410_341065

theorem min_days_for_progress_ratio : ∃ n : ℕ, n = 23 ∧ 
  (∀ x : ℕ, (1.2 : ℝ)^x / (0.8 : ℝ)^x ≥ 10000 → x ≥ n) ∧
  (1.2 : ℝ)^n / (0.8 : ℝ)^n ≥ 10000 :=
by sorry

end min_days_for_progress_ratio_l3410_341065


namespace fraction_equality_l3410_341018

theorem fraction_equality : (250 : ℚ) / ((20 + 15 * 3) - 10) = 250 / 55 := by sorry

end fraction_equality_l3410_341018


namespace sum_of_fractions_equals_one_l3410_341097

theorem sum_of_fractions_equals_one (x y z : ℝ) (h : x * y * z = 1) :
  (1 / (1 + x + x * y)) + (1 / (1 + y + y * z)) + (1 / (1 + z + z * x)) = 1 := by
  sorry

end sum_of_fractions_equals_one_l3410_341097


namespace abc_product_l3410_341056

theorem abc_product (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 29) (h5 : (1 : ℚ) / a + 1 / b + 1 / c + 399 / (a * b * c) = 1) :
  a * b * c = 992 := by
  sorry

end abc_product_l3410_341056


namespace sin_585_degrees_l3410_341011

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_585_degrees_l3410_341011


namespace inequality_system_solution_l3410_341094

def solution_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3}

def satisfies_inequalities (x : ℤ) : Prop :=
  2 * x ≥ 3 * (x - 1) ∧ 2 - x / 2 < 5

theorem inequality_system_solution :
  ∀ x : ℤ, x ∈ solution_set ↔ satisfies_inequalities x :=
by sorry

end inequality_system_solution_l3410_341094


namespace sum_172_83_base4_l3410_341045

/-- Converts a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Checks if a list of natural numbers represents a valid base 4 number -/
def isValidBase4 (l : List ℕ) : Prop :=
  ∀ d ∈ l, d < 4

theorem sum_172_83_base4 :
  toBase4 (172 + 83) = [3, 3, 3, 3, 3] ∧ isValidBase4 [3, 3, 3, 3, 3] := by
  sorry

end sum_172_83_base4_l3410_341045


namespace roots_sum_fraction_eq_neg_two_l3410_341068

theorem roots_sum_fraction_eq_neg_two (z₁ z₂ : ℂ) 
  (h₁ : z₁^2 + z₁ + 1 = 0) 
  (h₂ : z₂^2 + z₂ + 1 = 0) 
  (h₃ : z₁ ≠ z₂) : 
  z₂ / (z₁ + 1) + z₁ / (z₂ + 1) = -2 := by sorry

end roots_sum_fraction_eq_neg_two_l3410_341068


namespace simplify_sqrt_expression_l3410_341040

theorem simplify_sqrt_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (1 + ((x^4 - 1) / (2 * x^2))^2) = x^2 / 2 + 1 / (2 * x^2) := by
  sorry

end simplify_sqrt_expression_l3410_341040


namespace slope_angle_of_line_l3410_341062

theorem slope_angle_of_line (x y : ℝ) :
  let line_eq := x * Real.tan (π / 6) + y - 7 = 0
  let slope := -Real.tan (π / 6)
  let slope_angle := Real.arctan (-slope)
  slope_angle = 5 * π / 6 := by sorry

end slope_angle_of_line_l3410_341062


namespace parabola_directrix_l3410_341013

/-- A parabola with equation y^2 = 2px and focus on the line 2x + 3y - 4 = 0 has directrix x = -2 -/
theorem parabola_directrix (p : ℝ) : 
  ∃ (f : ℝ × ℝ), 
    (∀ (x y : ℝ), y^2 = 2*p*x ↔ ((x - f.1)^2 + (y - f.2)^2 = (x + f.1)^2)) ∧ 
    (2*f.1 + 3*f.2 - 4 = 0) → 
    (f.1 = 2 ∧ f.2 = 0 ∧ ∀ (x : ℝ), x = -2 ↔ x = f.1 - p) :=
by sorry

end parabola_directrix_l3410_341013


namespace stating_regular_ngon_diagonal_difference_l3410_341070

/-- 
Given a regular n-gon with n > 5, this function calculates the length of its longest diagonal.
-/
noncomputable def longest_diagonal (n : ℕ) (side_length : ℝ) : ℝ := sorry

/-- 
Given a regular n-gon with n > 5, this function calculates the length of its shortest diagonal.
-/
noncomputable def shortest_diagonal (n : ℕ) (side_length : ℝ) : ℝ := sorry

/-- 
Theorem stating that for a regular n-gon with n > 5, the difference between 
the longest diagonal and the shortest diagonal is equal to the side length 
if and only if n = 9.
-/
theorem regular_ngon_diagonal_difference (n : ℕ) (side_length : ℝ) : 
  n > 5 → 
  (longest_diagonal n side_length - shortest_diagonal n side_length = side_length ↔ n = 9) :=
by sorry

end stating_regular_ngon_diagonal_difference_l3410_341070


namespace melted_ice_cream_depth_l3410_341016

/-- Given a sphere of radius 3 inches and a cylinder of radius 12 inches with the same volume,
    prove that the height of the cylinder is 1/4 inch. -/
theorem melted_ice_cream_depth (sphere_radius : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  sphere_radius = 3 →
  cylinder_radius = 12 →
  (4 / 3) * Real.pi * sphere_radius ^ 3 = Real.pi * cylinder_radius ^ 2 * cylinder_height →
  cylinder_height = 1 / 4 := by
  sorry

end melted_ice_cream_depth_l3410_341016


namespace units_digit_47_power_47_l3410_341006

theorem units_digit_47_power_47 : (47^47) % 10 = 3 := by
  sorry

end units_digit_47_power_47_l3410_341006


namespace shortest_side_l3410_341099

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Theorem statement
theorem shortest_side (t : Triangle) (h : t.a^2 + t.b^2 > 5 * t.c^2) : 
  t.c < t.a ∧ t.c < t.b := by
  sorry

end shortest_side_l3410_341099


namespace inequality_proof_l3410_341084

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ 2*(a^3 + b^3 + c^3)/(a*b*c) + 3 := by
sorry

end inequality_proof_l3410_341084


namespace chickpea_flour_amount_l3410_341088

def rye_flour : ℕ := 5
def whole_wheat_bread_flour : ℕ := 10
def whole_wheat_pastry_flour : ℕ := 2
def total_flour : ℕ := 20

theorem chickpea_flour_amount :
  total_flour - (rye_flour + whole_wheat_bread_flour + whole_wheat_pastry_flour) = 3 := by
  sorry

end chickpea_flour_amount_l3410_341088


namespace housing_units_without_cable_or_vcr_l3410_341029

theorem housing_units_without_cable_or_vcr 
  (total : ℝ) 
  (cable : ℝ) 
  (vcr : ℝ) 
  (both : ℝ) 
  (h1 : cable = (1/5) * total) 
  (h2 : vcr = (1/10) * total) 
  (h3 : both = (1/3) * cable) :
  (total - (cable + vcr - both)) / total = 7/10 := by
sorry

end housing_units_without_cable_or_vcr_l3410_341029


namespace complex_plane_theorem_l3410_341019

def complex_plane_problem (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im
  Complex.abs z = Real.sqrt 2 ∧
  (z^2).im = 2 ∧
  x > 0 ∧ y > 0 →
  z = Complex.mk 1 1 ∧
  let A := z
  let B := z^2
  let C := z - z^2
  let cos_ABC := ((B.re - A.re) * (C.re - B.re) + (B.im - A.im) * (C.im - B.im)) /
                 (Complex.abs (B - A) * Complex.abs (C - B))
  cos_ABC = 3 * Real.sqrt 10 / 10

theorem complex_plane_theorem :
  ∃ z : ℂ, complex_plane_problem z :=
sorry

end complex_plane_theorem_l3410_341019


namespace vanya_climb_ratio_l3410_341077

-- Define the floors for Anya and Vanya
def anya_floor : ℕ := 2
def vanya_floor : ℕ := 6
def start_floor : ℕ := 1

-- Define the climbs for Anya and Vanya
def anya_climb : ℕ := anya_floor - start_floor
def vanya_climb : ℕ := vanya_floor - start_floor

-- Theorem statement
theorem vanya_climb_ratio :
  (vanya_climb : ℚ) / (anya_climb : ℚ) = 5 := by sorry

end vanya_climb_ratio_l3410_341077


namespace root_relationship_l3410_341093

theorem root_relationship (a b c x y : ℝ) (ha : a ≠ 0) :
  a * x^2 + b * x + c = 0 ∧ y^2 + b * y + a * c = 0 → x = y / a := by
  sorry

end root_relationship_l3410_341093


namespace pure_imaginary_complex_number_l3410_341064

theorem pure_imaginary_complex_number (x : ℝ) : 
  let z : ℂ := (x^2 - 1) + (x - 1) * I
  (∃ (y : ℝ), z = y * I) → x = -1 := by
sorry

end pure_imaginary_complex_number_l3410_341064


namespace intersection_point_l3410_341061

def A : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 - 1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1 + 3}

theorem intersection_point (m : ℝ × ℝ) (hA : m ∈ A) (hB : m ∈ B) : m = (4, 7) := by
  sorry

end intersection_point_l3410_341061


namespace smallest_integer_l3410_341002

theorem smallest_integer (x : ℕ) (m n : ℕ) (h1 : n = 36) (h2 : 0 < x) 
  (h3 : Nat.gcd m n = x + 3) (h4 : Nat.lcm m n = x * (x + 3)) :
  m ≥ 3 ∧ (∃ (x : ℕ), m = 3 ∧ 0 < x ∧ 
    Nat.gcd 3 36 = x + 3 ∧ Nat.lcm 3 36 = x * (x + 3)) :=
by sorry

end smallest_integer_l3410_341002


namespace lottery_probability_l3410_341086

def eligible_numbers : Finset ℕ := {1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100}

def valid_combinations : ℕ := 10

theorem lottery_probability :
  let total_combinations := Nat.choose (Finset.card eligible_numbers - 1) 5
  (valid_combinations : ℚ) / total_combinations = 10 / 3003 :=
by sorry

end lottery_probability_l3410_341086


namespace green_equals_purple_l3410_341058

/-- Proves that the number of green shoe pairs is equal to the number of purple shoe pairs -/
theorem green_equals_purple (total : ℕ) (blue : ℕ) (purple : ℕ)
  (h_total : total = 1250)
  (h_blue : blue = 540)
  (h_purple : purple = 355)
  (h_sum : total = blue + purple + (total - blue - purple)) :
  total - blue - purple = purple := by
  sorry

end green_equals_purple_l3410_341058


namespace pet_store_puppies_l3410_341057

theorem pet_store_puppies (sold : ℕ) (num_cages : ℕ) (puppies_per_cage : ℕ) : 
  sold = 21 → num_cages = 9 → puppies_per_cage = 9 → 
  sold + (num_cages * puppies_per_cage) = 102 := by
sorry

end pet_store_puppies_l3410_341057


namespace smallest_number_divisible_l3410_341001

theorem smallest_number_divisible (n : ℕ) : n = 92160 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 37 * 47 * 53 * k)) ∧ 
  (∃ k : ℕ, (n + 7) = 37 * 47 * 53 * k) := by
  sorry

end smallest_number_divisible_l3410_341001


namespace common_chord_of_circles_l3410_341047

/-- Given two circles C₁ and C₂, prove that their common chord lies on the line 3x - 4y + 6 = 0 -/
theorem common_chord_of_circles (x y : ℝ) :
  (x^2 + y^2 + 2*x - 6*y + 1 = 0) ∧ (x^2 + y^2 - 4*x + 2*y - 11 = 0) →
  (3*x - 4*y + 6 = 0) := by
  sorry

end common_chord_of_circles_l3410_341047


namespace x_between_one_third_and_two_thirds_l3410_341021

theorem x_between_one_third_and_two_thirds (x : ℝ) :
  (1 < 4 * x ∧ 4 * x < 3) ∧ (2 < 6 * x ∧ 6 * x < 4) → (1/3 < x ∧ x < 2/3) := by
  sorry

end x_between_one_third_and_two_thirds_l3410_341021


namespace find_tuesday_date_l3410_341049

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in a month -/
structure Date where
  day : ℕ
  month : ℕ
  dayOfWeek : DayOfWeek

/-- Given conditions of the problem -/
def problemConditions (tuesdayDate : Date) (thirdFridayDate : Date) : Prop :=
  tuesdayDate.dayOfWeek = DayOfWeek.Tuesday ∧
  thirdFridayDate.dayOfWeek = DayOfWeek.Friday ∧
  thirdFridayDate.day = 15 ∧
  thirdFridayDate.day + 3 = 18

/-- The theorem to prove -/
theorem find_tuesday_date (tuesdayDate : Date) (thirdFridayDate : Date) :
  problemConditions tuesdayDate thirdFridayDate →
  tuesdayDate.day = 29 ∧ tuesdayDate.month + 1 = thirdFridayDate.month :=
by sorry

end find_tuesday_date_l3410_341049


namespace base_seven_23456_equals_6068_l3410_341010

def base_seven_to_ten (n : List Nat) : Nat :=
  List.foldr (λ (digit : Nat) (acc : Nat) => 7 * acc + digit) 0 n

theorem base_seven_23456_equals_6068 :
  base_seven_to_ten [2, 3, 4, 5, 6] = 6068 := by
  sorry

end base_seven_23456_equals_6068_l3410_341010


namespace largest_number_with_123_l3410_341024

theorem largest_number_with_123 :
  let a := 321
  let b := 21^3
  let c := 3^21
  let d := 2^31
  (c > a) ∧ (c > b) ∧ (c > d) :=
by sorry

end largest_number_with_123_l3410_341024


namespace child_share_proof_l3410_341035

theorem child_share_proof (total_amount : ℕ) (ratio_a ratio_b ratio_c : ℕ) 
  (h1 : total_amount = 2700)
  (h2 : ratio_a = 2)
  (h3 : ratio_b = 3)
  (h4 : ratio_c = 4) : 
  (total_amount * ratio_b) / (ratio_a + ratio_b + ratio_c) = 900 := by
  sorry

end child_share_proof_l3410_341035


namespace craig_walk_distance_l3410_341017

/-- The distance Craig walked from school to David's house in miles -/
def school_to_david : ℝ := 0.2

/-- The total distance Craig walked in miles -/
def total_distance : ℝ := 0.9

/-- The distance Craig walked from David's house to his own house in miles -/
def david_to_craig : ℝ := total_distance - school_to_david

theorem craig_walk_distance : david_to_craig = 0.7 := by
  sorry

end craig_walk_distance_l3410_341017


namespace unique_value_at_two_l3410_341071

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y^2

/-- The theorem stating that f(2) = 5 for any function satisfying the functional equation -/
theorem unique_value_at_two (f : ℝ → ℝ) (h : FunctionalEquation f) : f 2 = 5 := by
  sorry

end unique_value_at_two_l3410_341071


namespace division_simplification_l3410_341037

theorem division_simplification (a : ℝ) (h : a ≠ 0) :
  (21 * a^3 - 7 * a) / (7 * a) = 3 * a^2 - 1 := by
  sorry

end division_simplification_l3410_341037


namespace floor_negative_seven_fourths_l3410_341030

theorem floor_negative_seven_fourths :
  ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end floor_negative_seven_fourths_l3410_341030


namespace min_value_f_l3410_341033

/-- Given positive real numbers x₁ and x₂, and a function f satisfying certain conditions,
    the value of f(x₁ + x₂) has a lower bound of 4/5. -/
theorem min_value_f (x₁ x₂ : ℝ) (f : ℝ → ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hf : ∀ x, 4^x = (1 + f x) / (1 - f x))
  (hsum : f x₁ + f x₂ = 1) :
  f (x₁ + x₂) ≥ 4/5 := by
  sorry

end min_value_f_l3410_341033
