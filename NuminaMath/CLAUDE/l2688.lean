import Mathlib

namespace original_price_from_discounted_l2688_268875

/-- 
Given a shirt sold at a discounted price with a known discount percentage, 
this theorem proves the original selling price.
-/
theorem original_price_from_discounted (discounted_price : ℝ) (discount_percent : ℝ) 
  (h1 : discounted_price = 560) 
  (h2 : discount_percent = 20) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_percent / 100) = discounted_price ∧ 
    original_price = 700 := by
  sorry

end original_price_from_discounted_l2688_268875


namespace tan_sum_alpha_beta_l2688_268867

theorem tan_sum_alpha_beta (α β : Real) 
  (h1 : Real.sin α + Real.sin β = 1/4)
  (h2 : Real.cos α + Real.cos β = 1/3) : 
  Real.tan (α + β) = 24/7 := by
  sorry

end tan_sum_alpha_beta_l2688_268867


namespace temperature_difference_l2688_268819

theorem temperature_difference (highest lowest : ℤ) (h1 : highest = 8) (h2 : lowest = -1) :
  highest - lowest = 9 := by
  sorry

end temperature_difference_l2688_268819


namespace petes_number_l2688_268872

theorem petes_number : ∃ x : ℝ, 5 * (3 * x - 5) = 200 ∧ x = 15 := by sorry

end petes_number_l2688_268872


namespace original_denominator_problem_l2688_268825

theorem original_denominator_problem (d : ℤ) : 
  (3 : ℚ) / d ≠ 0 →
  (9 : ℚ) / (d + 7) = (1 : ℚ) / 3 →
  d = 20 := by
sorry

end original_denominator_problem_l2688_268825


namespace polygon_edges_l2688_268879

theorem polygon_edges (n : ℕ) : n ≥ 3 → (
  (n - 2) * 180 = 4 * 360 + 180 ↔ n = 11
) := by sorry

end polygon_edges_l2688_268879


namespace triangle_exists_l2688_268850

/-- A triangle with given circumradius, centroid-circumcenter distance, and centroid-altitude distance --/
structure TriangleWithCircumcenter where
  r : ℝ  -- radius of circumscribed circle
  KS : ℝ  -- distance from circumcenter to centroid
  d : ℝ  -- distance from centroid to altitude

/-- Conditions for the existence of a triangle with given parameters --/
def triangle_existence_conditions (t : TriangleWithCircumcenter) : Prop :=
  t.d ≤ 2 * t.KS ∧ 
  t.r ≥ 3 * t.d / 2 ∧ 
  |Real.sqrt (4 * t.r^2 - 9 * t.d^2) - 3 * Real.sqrt (4 * t.KS^2 - t.d^2)| < 4 * t.r

/-- Theorem stating the existence of a triangle with given parameters --/
theorem triangle_exists (t : TriangleWithCircumcenter) : 
  triangle_existence_conditions t ↔ ∃ (triangle : Type), true :=
sorry

end triangle_exists_l2688_268850


namespace tan_double_angle_special_point_l2688_268840

theorem tan_double_angle_special_point (α : Real) :
  (∃ (x y : Real), x = 1 ∧ y = -2 ∧ x * Real.cos α = y * Real.sin α) →
  Real.tan (2 * α) = 4 / 3 :=
by sorry

end tan_double_angle_special_point_l2688_268840


namespace fraction_inequality_l2688_268811

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
sorry

end fraction_inequality_l2688_268811


namespace dot_product_theorem_l2688_268828

def vector_dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

def vector_perpendicular (v w : ℝ × ℝ) : Prop :=
  vector_dot_product v w = 0

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem dot_product_theorem (x y : ℝ) : 
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (1, y)
  let c : ℝ × ℝ := (3, -6)
  vector_perpendicular a c → vector_parallel b c →
  vector_dot_product (a.1 + b.1, a.2 + b.2) c = 15 := by
  sorry

end dot_product_theorem_l2688_268828


namespace g_difference_zero_l2688_268838

def sum_of_divisors (n : ℕ+) : ℕ := sorry

def f (n : ℕ+) : ℚ := (sum_of_divisors n : ℚ) / n

def g (n : ℕ+) : ℚ := f n + 1 / n

theorem g_difference_zero : g 512 - g 256 = 0 := by sorry

end g_difference_zero_l2688_268838


namespace simplify_fraction_l2688_268808

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (45 * b^3) = 2/3 := by
  sorry

end simplify_fraction_l2688_268808


namespace linear_pair_angle_ratio_l2688_268802

theorem linear_pair_angle_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Both angles are positive
  a + b = 180 ∧    -- Angles form a linear pair (sum to 180°)
  a = 5 * b →      -- Angles are in ratio 5:1
  b = 30 :=        -- The smaller angle is 30°
by sorry

end linear_pair_angle_ratio_l2688_268802


namespace sara_lunch_cost_l2688_268835

/-- The cost of Sara's lunch given the prices of a hotdog and a salad -/
def lunch_cost (hotdog_price salad_price : ℚ) : ℚ :=
  hotdog_price + salad_price

/-- Theorem stating that Sara's lunch cost is $10.46 -/
theorem sara_lunch_cost :
  lunch_cost 5.36 5.10 = 10.46 := by
  sorry

end sara_lunch_cost_l2688_268835


namespace equation_solution_l2688_268864

theorem equation_solution (x : ℝ) :
  x ≠ 2/3 →
  ((4*x + 3) / (3*x^2 + 4*x - 4) = 3*x / (3*x - 2)) ↔
  (x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3) :=
by sorry

end equation_solution_l2688_268864


namespace tens_digit_sum_factorials_l2688_268831

def factorial (n : ℕ) : ℕ := sorry

def tensDigit (n : ℕ) : ℕ := sorry

def sumFactorials (n : ℕ) : ℕ := sorry

theorem tens_digit_sum_factorials :
  tensDigit (sumFactorials 100) = 0 := by sorry

end tens_digit_sum_factorials_l2688_268831


namespace octopus_leg_solution_l2688_268897

-- Define the possible number of legs for an octopus
inductive LegCount : Type
  | six : LegCount
  | seven : LegCount
  | eight : LegCount

-- Define the colors of the octopuses
inductive OctopusColor : Type
  | blue : OctopusColor
  | green : OctopusColor
  | yellow : OctopusColor
  | red : OctopusColor

-- Define a function to determine if an octopus is truthful based on its leg count
def isTruthful (legs : LegCount) : Prop :=
  match legs with
  | LegCount.six => True
  | LegCount.seven => False
  | LegCount.eight => True

-- Define a function to convert LegCount to a natural number
def legCountToNat (legs : LegCount) : ℕ :=
  match legs with
  | LegCount.six => 6
  | LegCount.seven => 7
  | LegCount.eight => 8

-- Define the claims made by each octopus
def claim (color : OctopusColor) : ℕ :=
  match color with
  | OctopusColor.blue => 28
  | OctopusColor.green => 27
  | OctopusColor.yellow => 26
  | OctopusColor.red => 25

-- Define the theorem
theorem octopus_leg_solution :
  ∃ (legs : OctopusColor → LegCount),
    (legs OctopusColor.green = LegCount.six) ∧
    (legs OctopusColor.blue = LegCount.seven) ∧
    (legs OctopusColor.yellow = LegCount.seven) ∧
    (legs OctopusColor.red = LegCount.seven) ∧
    (∀ (c : OctopusColor), isTruthful (legs c) ↔ (claim c = legCountToNat (legs OctopusColor.blue) + legCountToNat (legs OctopusColor.green) + legCountToNat (legs OctopusColor.yellow) + legCountToNat (legs OctopusColor.red))) :=
  sorry

end octopus_leg_solution_l2688_268897


namespace trig_combination_l2688_268810

theorem trig_combination (x : ℝ) : 
  Real.cos (3 * x) + Real.cos (5 * x) + Real.tan (2 * x) = 
  2 * Real.cos (4 * x) * Real.cos x + Real.sin (2 * x) / Real.cos (2 * x) :=
by sorry

end trig_combination_l2688_268810


namespace sum_of_coefficients_bounds_l2688_268893

/-- A quadratic function with vertex in the first quadrant passing through (0,1) and (-1,0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  vertex_first_quadrant : -b / (2 * a) > 0
  passes_through_0_1 : c = 1
  passes_through_neg1_0 : a - b + c = 0

/-- The sum of coefficients of a quadratic function -/
def S (f : QuadraticFunction) : ℝ := f.a + f.b + f.c

/-- Theorem: The sum of coefficients S is between 0 and 2 -/
theorem sum_of_coefficients_bounds (f : QuadraticFunction) : 0 < S f ∧ S f < 2 := by
  sorry

end sum_of_coefficients_bounds_l2688_268893


namespace selling_price_fraction_l2688_268856

theorem selling_price_fraction (cost_price : ℝ) (original_selling_price : ℝ) : 
  original_selling_price = cost_price * (1 + 0.275) →
  ∃ (f : ℝ), f * original_selling_price = cost_price * (1 - 0.15) ∧ f = 17 / 25 :=
by
  sorry

end selling_price_fraction_l2688_268856


namespace line_formation_ways_l2688_268804

/-- The number of ways to form a line by selecting r people out of n -/
def permutations (n : ℕ) (r : ℕ) : ℕ := (n.factorial) / ((n - r).factorial)

/-- The total number of people -/
def total_people : ℕ := 7

/-- The number of people to select -/
def selected_people : ℕ := 5

theorem line_formation_ways :
  permutations total_people selected_people = 2520 := by
  sorry

end line_formation_ways_l2688_268804


namespace margarets_mean_score_l2688_268889

def scores : List ℝ := [88, 90, 94, 95, 96, 99]

theorem margarets_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ (cyprian_scores : List ℝ) (margaret_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        margaret_scores.length = 2 ∧ 
        cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        cyprian_scores.sum / cyprian_scores.length = 92) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 2 ∧ 
    margaret_scores.sum / margaret_scores.length = 97 := by
  sorry

end margarets_mean_score_l2688_268889


namespace loop_statement_efficiency_l2688_268805

/-- Enum representing different types of algorithm statements -/
inductive AlgorithmStatement
  | InputOutput
  | Assignment
  | Conditional
  | Loop

/-- Definition of a program's capability to handle large computational problems -/
def CanHandleLargeProblems (statements : List AlgorithmStatement) : Prop :=
  statements.length > 0

/-- Definition of the primary reason for efficient handling of large problems -/
def PrimaryReasonForEfficiency (statement : AlgorithmStatement) (statements : List AlgorithmStatement) : Prop :=
  CanHandleLargeProblems statements ∧ statement ∈ statements

theorem loop_statement_efficiency :
  ∀ (statements : List AlgorithmStatement),
    CanHandleLargeProblems statements →
    AlgorithmStatement.InputOutput ∈ statements →
    AlgorithmStatement.Assignment ∈ statements →
    AlgorithmStatement.Conditional ∈ statements →
    AlgorithmStatement.Loop ∈ statements →
    PrimaryReasonForEfficiency AlgorithmStatement.Loop statements :=
by
  sorry

#check loop_statement_efficiency

end loop_statement_efficiency_l2688_268805


namespace book_page_words_l2688_268853

theorem book_page_words (total_pages : ℕ) (max_words_per_page : ℕ) (total_words_mod : ℕ) :
  total_pages = 150 →
  max_words_per_page = 120 →
  total_words_mod = 221 →
  ∃ (words_per_page : ℕ),
    words_per_page ≤ max_words_per_page ∧
    Nat.Prime words_per_page ∧
    (total_pages * words_per_page) % total_words_mod = 220 ∧
    words_per_page = 67 :=
by sorry

end book_page_words_l2688_268853


namespace sqrt_5_irrational_l2688_268832

-- Define the set of numbers
def number_set : Set ℝ := {0.618, 22/7, Real.sqrt 5, -3}

-- Define irrationality
def is_irrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Theorem statement
theorem sqrt_5_irrational : ∃ (x : ℝ), x ∈ number_set ∧ is_irrational x :=
sorry

end sqrt_5_irrational_l2688_268832


namespace prob_sum_greater_than_four_is_five_sixths_l2688_268894

/-- The number of possible outcomes when tossing two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is 4 or less -/
def outcomes_sum_4_or_less : ℕ := 6

/-- The probability of getting a sum greater than four when tossing two dice -/
def prob_sum_greater_than_four : ℚ := 5 / 6

theorem prob_sum_greater_than_four_is_five_sixths :
  prob_sum_greater_than_four = 1 - (outcomes_sum_4_or_less : ℚ) / total_outcomes :=
by sorry

end prob_sum_greater_than_four_is_five_sixths_l2688_268894


namespace circle_radius_zero_l2688_268823

theorem circle_radius_zero (x y : ℝ) :
  x^2 - 4*x + y^2 - 6*y + 13 = 0 → (∃ r : ℝ, r = 0 ∧ (x - 2)^2 + (y - 3)^2 = r^2) :=
by sorry

end circle_radius_zero_l2688_268823


namespace function_properties_l2688_268890

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a^(2*x) - 2*a^x + 1

theorem function_properties (a : ℝ) (h_a : a > 1) :
  (∀ y : ℝ, y < 1 → ∃ x : ℝ, f a x = y) ∧
  (∀ x : ℝ, f a x < 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f a x ≥ -7) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ f a x = -7) →
  a = 2 :=
sorry

end function_properties_l2688_268890


namespace gcf_lcm_sum_36_56_l2688_268815

theorem gcf_lcm_sum_36_56 : Nat.gcd 36 56 + Nat.lcm 36 56 = 508 := by
  sorry

end gcf_lcm_sum_36_56_l2688_268815


namespace dye_mixture_amount_l2688_268869

/-- The total amount of mixture obtained by combining a fraction of water and a fraction of vinegar -/
def mixture_amount (water_total : ℚ) (vinegar_total : ℚ) (water_fraction : ℚ) (vinegar_fraction : ℚ) : ℚ :=
  water_fraction * water_total + vinegar_fraction * vinegar_total

/-- Theorem stating that the mixture amount for the given problem is 27 liters -/
theorem dye_mixture_amount :
  mixture_amount 20 18 (3/5) (5/6) = 27 := by
  sorry

end dye_mixture_amount_l2688_268869


namespace exam_average_l2688_268845

theorem exam_average (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 15 →
  n2 = 10 →
  avg1 = 70 / 100 →
  avg2 = 95 / 100 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 80 / 100 := by
  sorry

end exam_average_l2688_268845


namespace barbie_coconuts_l2688_268899

theorem barbie_coconuts (total_coconuts : ℕ) (trips : ℕ) (bruno_capacity : ℕ) 
  (h1 : total_coconuts = 144)
  (h2 : trips = 12)
  (h3 : bruno_capacity = 8) :
  ∃ barbie_capacity : ℕ, 
    barbie_capacity * trips + bruno_capacity * trips = total_coconuts ∧ 
    barbie_capacity = 4 := by
  sorry

end barbie_coconuts_l2688_268899


namespace sum_770_product_not_divisible_l2688_268863

theorem sum_770_product_not_divisible (a b : ℕ) : 
  a + b = 770 → ¬(770 ∣ (a * b)) := by
sorry

end sum_770_product_not_divisible_l2688_268863


namespace distinct_weights_theorem_l2688_268812

/-- The number of distinct weights that can be measured with four weights on a two-pan balance scale. -/
def distinct_weights : ℕ := 40

/-- The number of weights available. -/
def num_weights : ℕ := 4

/-- The number of possible placements for each weight (left pan, right pan, or not used). -/
def placement_options : ℕ := 3

/-- Represents the two-pan balance scale. -/
structure BalanceScale :=
  (left_pan : Finset ℕ)
  (right_pan : Finset ℕ)

/-- Calculates the total number of possible configurations. -/
def total_configurations : ℕ := placement_options ^ num_weights

/-- Theorem stating the number of distinct weights that can be measured. -/
theorem distinct_weights_theorem :
  distinct_weights = (total_configurations - 1) / 2 :=
sorry

end distinct_weights_theorem_l2688_268812


namespace complement_M_intersect_N_l2688_268848

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x : ℝ | x < -2} := by sorry

end complement_M_intersect_N_l2688_268848


namespace line_mb_value_l2688_268817

/-- A line passing through (-1, -3) and intersecting the y-axis at y = -1 has mb = 2 -/
theorem line_mb_value (m b : ℝ) : 
  (∀ x y, y = m * x + b → (x = -1 ∧ y = -3) ∨ (x = 0 ∧ y = -1)) → 
  m * b = 2 := by
  sorry

end line_mb_value_l2688_268817


namespace same_number_on_four_dice_l2688_268886

theorem same_number_on_four_dice : 
  let n : ℕ := 6  -- number of sides on each die
  let k : ℕ := 4  -- number of dice
  (1 : ℚ) / n^(k-1) = (1 : ℚ) / 216 :=
by sorry

end same_number_on_four_dice_l2688_268886


namespace average_theorem_l2688_268837

theorem average_theorem (x : ℝ) : 
  (x + 0.005) / 2 = 0.2025 → x = 0.400 := by
sorry

end average_theorem_l2688_268837


namespace motel_rent_problem_l2688_268862

/-- Represents the total rent charged by a motel on a given night -/
def TotalRent (r40 r60 : ℕ) : ℝ := 40 * r40 + 60 * r60

/-- The problem statement -/
theorem motel_rent_problem (r40 r60 : ℕ) :
  (∃ (total : ℝ), total = TotalRent r40 r60 ∧
    0.8 * total = TotalRent (r40 + 10) (r60 - 10)) →
  TotalRent r40 r60 = 1000 := by
  sorry

#check motel_rent_problem

end motel_rent_problem_l2688_268862


namespace square_ratio_theorem_l2688_268876

theorem square_ratio_theorem : ∃ (a b c : ℕ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (180 : ℝ) / 45 = (a * (b.sqrt : ℝ) / c : ℝ)^2 ∧ 
  a + b + c = 4 := by
sorry

end square_ratio_theorem_l2688_268876


namespace book_arrangement_theorem_l2688_268818

theorem book_arrangement_theorem :
  let total_books : ℕ := 10
  let spanish_books : ℕ := 4
  let french_books : ℕ := 3
  let german_books : ℕ := 3
  let number_of_units : ℕ := 2 + german_books

  spanish_books + french_books + german_books = total_books →
  (number_of_units.factorial * spanish_books.factorial * french_books.factorial : ℕ) = 17280 :=
by sorry

end book_arrangement_theorem_l2688_268818


namespace existence_of_m_and_k_l2688_268871

def f (p : ℕ × ℕ) : ℕ × ℕ :=
  let (a, b) := p
  if a < b then (2*a, b-a) else (a-b, 2*b)

def iter_f (k : ℕ) : (ℕ × ℕ) → (ℕ × ℕ) :=
  match k with
  | 0 => id
  | k+1 => f ∘ (iter_f k)

theorem existence_of_m_and_k (n : ℕ) (h : n > 1) :
  ∃ (m k : ℕ), m < n ∧ iter_f k (n, m) = (m, n) := by
  sorry

#check existence_of_m_and_k

end existence_of_m_and_k_l2688_268871


namespace complex_cube_problem_l2688_268834

theorem complex_cube_problem :
  ∀ (x y : ℕ+) (c : ℤ),
    (x : ℂ) + y * Complex.I ≠ 1 + 6 * Complex.I →
    ((x : ℂ) + y * Complex.I) ^ 3 ≠ -107 + c * Complex.I :=
by sorry

end complex_cube_problem_l2688_268834


namespace sum_of_partial_fractions_coefficients_l2688_268859

theorem sum_of_partial_fractions_coefficients (A B C D E : ℝ) :
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 ∧ x ≠ -6 →
    (x + 1) / ((x + 2) * (x + 3) * (x + 4) * (x + 5) * (x + 6)) =
    A / (x + 2) + B / (x + 3) + C / (x + 4) + D / (x + 5) + E / (x + 6)) →
  A + B + C + D + E = 0 := by
sorry

end sum_of_partial_fractions_coefficients_l2688_268859


namespace arithmetic_sequence_common_difference_l2688_268887

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum_odd : a 1 + a 3 + a 5 = 105)
  (h_sum_even : a 2 + a 4 + a 6 = 99) :
  ∃ d : ℝ, d = -2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end arithmetic_sequence_common_difference_l2688_268887


namespace intersection_A_complement_B_l2688_268885

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x < 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Bᶜ) = {x : ℝ | 2 ≤ x ∧ x < 5} := by sorry

end intersection_A_complement_B_l2688_268885


namespace intersecting_spheres_equal_volumes_l2688_268826

theorem intersecting_spheres_equal_volumes (r : ℝ) (d : ℝ) : 
  r = 1 → 
  0 < d ∧ d < 2 * r →
  (4 * π * r^3 / 3 - π * (r - d / 2)^2 * (2 * r + d / 2) / 3) * 2 = 4 * π * r^3 / 3 →
  d = 4 * Real.cos (4 * π / 9) :=
sorry

end intersecting_spheres_equal_volumes_l2688_268826


namespace teacher_engineer_ratio_l2688_268801

theorem teacher_engineer_ratio 
  (t : ℕ) -- number of teachers
  (e : ℕ) -- number of engineers
  (h_total : t + e > 0) -- ensure total group size is positive
  (h_avg : (40 * t + 55 * e) / (t + e) = 45) -- overall average age is 45
  : t = 2 * e := by
sorry

end teacher_engineer_ratio_l2688_268801


namespace monopoly_houses_theorem_l2688_268858

structure Player where
  name : String
  initialHouses : ℕ
  deriving Repr

def seanTransactions (houses : ℕ) : ℕ :=
  houses - 15 + 18

def karenTransactions (houses : ℕ) : ℕ :=
  0 + 10 + 8 + 15

def markTransactions (houses : ℕ) : ℕ :=
  houses + 12 - 25 - 15

def lucyTransactions (houses : ℕ) : ℕ :=
  houses - 8 + 6 - 20

def finalHouses (player : Player) : ℕ :=
  match player.name with
  | "Sean" => seanTransactions player.initialHouses
  | "Karen" => karenTransactions player.initialHouses
  | "Mark" => markTransactions player.initialHouses
  | "Lucy" => lucyTransactions player.initialHouses
  | _ => player.initialHouses

theorem monopoly_houses_theorem (sean karen mark lucy : Player)
  (h1 : sean.name = "Sean" ∧ sean.initialHouses = 45)
  (h2 : karen.name = "Karen" ∧ karen.initialHouses = 30)
  (h3 : mark.name = "Mark" ∧ mark.initialHouses = 55)
  (h4 : lucy.name = "Lucy" ∧ lucy.initialHouses = 35) :
  finalHouses sean = 48 ∧
  finalHouses karen = 33 ∧
  finalHouses mark = 27 ∧
  finalHouses lucy = 13 := by
  sorry

end monopoly_houses_theorem_l2688_268858


namespace sequence_a_properties_l2688_268873

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

axiom arithmetic_mean (n : ℕ) : sequence_a n = (sum_S n + 2) / 2

theorem sequence_a_properties :
  (sequence_a 1 = 2 ∧ sequence_a 2 = 4) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 2^n) := by sorry

end sequence_a_properties_l2688_268873


namespace coin_bill_combinations_l2688_268824

theorem coin_bill_combinations : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + 5 * p.2 = 207) (Finset.product (Finset.range 104) (Finset.range 42))).card :=
by
  sorry

end coin_bill_combinations_l2688_268824


namespace jordons_machine_l2688_268803

theorem jordons_machine (x : ℝ) : 2 * x + 3 = 27 → x = 12 := by
  sorry

end jordons_machine_l2688_268803


namespace unique_solutions_l2688_268878

def system_solution (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 1 ∧
  x^4 - 18*x^2*y^2 + 81*y^4 - 20*x^2 - 180*y^2 + 100 = 0

theorem unique_solutions :
  (∀ x y : ℝ, system_solution x y →
    ((x = -1/Real.sqrt 10 ∧ y = 3/Real.sqrt 10) ∨
     (x = -1/Real.sqrt 10 ∧ y = -3/Real.sqrt 10) ∨
     (x = 1/Real.sqrt 10 ∧ y = 3/Real.sqrt 10) ∨
     (x = 1/Real.sqrt 10 ∧ y = -3/Real.sqrt 10))) ∧
  (system_solution (-1/Real.sqrt 10) (3/Real.sqrt 10)) ∧
  (system_solution (-1/Real.sqrt 10) (-3/Real.sqrt 10)) ∧
  (system_solution (1/Real.sqrt 10) (3/Real.sqrt 10)) ∧
  (system_solution (1/Real.sqrt 10) (-3/Real.sqrt 10)) :=
by sorry

end unique_solutions_l2688_268878


namespace tangent_line_equation_inequality_l2688_268841

-- Define the function f(x) = x ln(x+1)
noncomputable def f (x : ℝ) : ℝ := x * Real.log (x + 1)

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  let slope := Real.log 2 + 1 / 2
  let y_intercept := -1 / 2
  (fun x => slope * x + y_intercept) 1 = f 1 ∧
  HasDerivAt f slope 1 :=
sorry

-- Theorem for the inequality
theorem inequality (x : ℝ) (h : x > -1) :
  f x + (1/2) * x^3 ≥ x^2 :=
sorry

end tangent_line_equation_inequality_l2688_268841


namespace gcd_plus_ten_l2688_268833

theorem gcd_plus_ten (a b : ℕ) (h : a = 8436 ∧ b = 156) :
  (Nat.gcd a b) + 10 = 22 := by
  sorry

end gcd_plus_ten_l2688_268833


namespace one_diagonal_implies_four_sides_l2688_268844

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ
  sides_pos : sides > 0

/-- A diagonal in a polygon is a line segment that connects two non-adjacent vertices. -/
def has_one_diagonal (p : Polygon) : Prop :=
  ∃ (v : ℕ), v < p.sides ∧ (p.sides - v - 2 = 1)

/-- Theorem: A polygon with exactly one diagonal that can be drawn from one vertex has 4 sides. -/
theorem one_diagonal_implies_four_sides (p : Polygon) (h : has_one_diagonal p) : p.sides = 4 := by
  sorry

end one_diagonal_implies_four_sides_l2688_268844


namespace find_number_l2688_268814

theorem find_number : ∃ x : ℝ, x = 50 ∧ (0.6 * x = 0.5 * 30 + 15) := by
  sorry

end find_number_l2688_268814


namespace rectangle_length_l2688_268860

/-- 
Given a rectangular garden with perimeter 950 meters and breadth 100 meters, 
this theorem proves that its length is 375 meters.
-/
theorem rectangle_length (perimeter breadth : ℝ) 
  (h_perimeter : perimeter = 950)
  (h_breadth : breadth = 100) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter :=
by
  sorry

#check rectangle_length

end rectangle_length_l2688_268860


namespace a_investment_l2688_268829

/-- Represents the investment scenario and proves A's investment amount -/
theorem a_investment (a_time b_time : ℕ) (b_investment total_profit a_share : ℚ) :
  a_time = 12 →
  b_time = 6 →
  b_investment = 200 →
  total_profit = 100 →
  a_share = 75 →
  ∃ (a_investment : ℚ),
    a_investment * a_time / (a_investment * a_time + b_investment * b_time) * total_profit = a_share ∧
    a_investment = 300 := by
  sorry


end a_investment_l2688_268829


namespace max_true_statements_l2688_268830

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^2 ∧ x^2 < 2),
    (x^2 > 2),
    (-2 < x ∧ x < 0),
    (0 < x ∧ x < 2),
    (0 < x - x^2 ∧ x - x^2 < 2)
  ]
  (∀ (s : Finset (Fin 5)), s.card > 3 → ¬(∀ i ∈ s, statements[i]))
  ∧
  (∃ (s : Finset (Fin 5)), s.card = 3 ∧ (∀ i ∈ s, statements[i])) :=
by sorry

#check max_true_statements

end max_true_statements_l2688_268830


namespace parallel_line_equation_l2688_268857

/-- A line in the Cartesian coordinate system -/
structure CartesianLine where
  slope : ℝ
  y_intercept : ℝ

/-- The equation of a line given its slope and y-intercept -/
def line_equation (l : CartesianLine) (x : ℝ) : ℝ :=
  l.slope * x + l.y_intercept

theorem parallel_line_equation 
  (l : CartesianLine) 
  (h1 : l.slope = -2) 
  (h2 : l.y_intercept = -3) : 
  ∀ x, line_equation l x = -2 * x - 3 :=
sorry

end parallel_line_equation_l2688_268857


namespace arithmetic_sequence_problem_l2688_268884

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating that a_15 = 24 for the given arithmetic sequence. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arithmetic : ArithmeticSequence a)
    (h_sum : a 3 + a 13 = 20)
    (h_a2 : a 2 = -2) : 
  a 15 = 24 := by
  sorry

end arithmetic_sequence_problem_l2688_268884


namespace max_teams_with_10_points_l2688_268877

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  total_teams : Nat
  points_per_win : Nat
  points_per_draw : Nat
  points_per_loss : Nat
  target_points : Nat

/-- The maximum number of teams that can achieve the target points -/
def max_teams_with_target_points (tournament : FootballTournament) : Nat :=
  sorry

/-- Theorem stating the maximum number of teams that can score exactly 10 points -/
theorem max_teams_with_10_points :
  let tournament := FootballTournament.mk 17 3 1 0 10
  max_teams_with_target_points tournament = 11 := by
  sorry

end max_teams_with_10_points_l2688_268877


namespace intersection_property_characterization_l2688_268813

/-- A function satisfying the property that the line through any two points
    on its graph intersects the y-axis at (0, pq) -/
def IntersectionProperty (f : ℝ → ℝ) : Prop :=
  ∀ p q : ℝ, p ≠ q →
    let m := (f q - f p) / (q - p)
    let b := f p - m * p
    b = p * q

/-- Theorem stating that functions satisfying the intersection property
    are of the form f(x) = x(c + x) for some constant c -/
theorem intersection_property_characterization (f : ℝ → ℝ) :
  IntersectionProperty f ↔ ∃ c : ℝ, ∀ x : ℝ, f x = x * (c + x) :=
sorry

end intersection_property_characterization_l2688_268813


namespace sqrt_2_times_sqrt_12_minus_2_between_2_and_3_l2688_268874

theorem sqrt_2_times_sqrt_12_minus_2_between_2_and_3 :
  2 < Real.sqrt 2 * Real.sqrt 12 - 2 ∧ Real.sqrt 2 * Real.sqrt 12 - 2 < 3 := by
  sorry

end sqrt_2_times_sqrt_12_minus_2_between_2_and_3_l2688_268874


namespace sqrt_product_quotient_sqrt_27_times_sqrt_32_div_sqrt_6_l2688_268888

theorem sqrt_product_quotient :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
  Real.sqrt (a * b) / Real.sqrt c = Real.sqrt a * Real.sqrt b / Real.sqrt c :=
by sorry

theorem sqrt_27_times_sqrt_32_div_sqrt_6 :
  Real.sqrt 27 * Real.sqrt 32 / Real.sqrt 6 = 12 :=
by sorry

end sqrt_product_quotient_sqrt_27_times_sqrt_32_div_sqrt_6_l2688_268888


namespace scientific_notation_equality_l2688_268852

theorem scientific_notation_equality : 21500000 = 2.15 * (10 ^ 7) := by
  sorry

end scientific_notation_equality_l2688_268852


namespace system_solution_l2688_268806

theorem system_solution (x y z : ℝ) : 
  (x + y + z = 6 ∧ x*y + y*z + z*x = 11 ∧ x*y*z = 6) ↔ 
  ((x = 1 ∧ y = 2 ∧ z = 3) ∨
   (x = 1 ∧ y = 3 ∧ z = 2) ∨
   (x = 2 ∧ y = 1 ∧ z = 3) ∨
   (x = 2 ∧ y = 3 ∧ z = 1) ∨
   (x = 3 ∧ y = 1 ∧ z = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 1)) :=
by sorry

end system_solution_l2688_268806


namespace choir_performance_theorem_l2688_268846

/-- Represents the number of singers joining in each verse of a choir performance --/
structure ChoirPerformance where
  total : ℕ
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- Calculates the number of singers joining in the fifth verse --/
def fifthVerseSingers (c : ChoirPerformance) : ℕ :=
  c.total - (c.first + c.second + c.third + c.fourth)

/-- Theorem stating the number of singers joining in the fifth verse --/
theorem choir_performance_theorem (c : ChoirPerformance) 
  (h_total : c.total = 60)
  (h_first : c.first = c.total / 2)
  (h_second : c.second = (c.total - c.first) / 3)
  (h_third : c.third = (c.total - c.first - c.second) / 4)
  (h_fourth : c.fourth = (c.total - c.first - c.second - c.third) / 5) :
  fifthVerseSingers c = 12 := by
  sorry

#eval fifthVerseSingers { total := 60, first := 30, second := 10, third := 5, fourth := 3, fifth := 12 }

end choir_performance_theorem_l2688_268846


namespace total_marbles_is_90_l2688_268865

/-- Represents the number of marbles of each color in the bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The ratio of red:blue:green marbles is 2:4:6 -/
def ratio_constraint (bag : MarbleBag) : Prop :=
  3 * bag.red = bag.blue ∧ 2 * bag.blue = bag.green

/-- There are 30 blue marbles -/
def blue_constraint (bag : MarbleBag) : Prop :=
  bag.blue = 30

/-- The total number of marbles in the bag -/
def total_marbles (bag : MarbleBag) : ℕ :=
  bag.red + bag.blue + bag.green

/-- Theorem stating that the total number of marbles is 90 -/
theorem total_marbles_is_90 (bag : MarbleBag) 
  (h_ratio : ratio_constraint bag) (h_blue : blue_constraint bag) : 
  total_marbles bag = 90 := by
  sorry

end total_marbles_is_90_l2688_268865


namespace max_product_constraint_l2688_268870

theorem max_product_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2*b = 1) :
  (∀ a' b' : ℝ, 0 < a' → 0 < b' → a' + 2*b' = 1 → a'*b' ≤ a*b) → a = 1/2 := by
  sorry

end max_product_constraint_l2688_268870


namespace water_rise_in_vessel_l2688_268800

/-- Represents the rise in water level when a cubical box is immersed in a rectangular vessel -/
theorem water_rise_in_vessel 
  (vessel_length : ℝ) 
  (vessel_breadth : ℝ) 
  (box_edge : ℝ) 
  (h : vessel_length = 60 ∧ vessel_breadth = 30 ∧ box_edge = 30) : 
  (box_edge ^ 3) / (vessel_length * vessel_breadth) = 15 := by
  sorry

#check water_rise_in_vessel

end water_rise_in_vessel_l2688_268800


namespace train_B_speed_train_B_speed_is_36_l2688_268880

-- Define the problem parameters
def train_A_length : ℝ := 125  -- meters
def train_B_length : ℝ := 150  -- meters
def train_A_speed : ℝ := 54    -- km/hr
def crossing_time : ℝ := 11    -- seconds

-- Define the theorem
theorem train_B_speed : ℝ :=
  let total_distance := train_A_length + train_B_length
  let relative_speed_mps := total_distance / crossing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  relative_speed_kmph - train_A_speed

-- Prove the theorem
theorem train_B_speed_is_36 : train_B_speed = 36 := by
  sorry

end train_B_speed_train_B_speed_is_36_l2688_268880


namespace b_share_is_180_l2688_268866

/-- Represents the rental arrangement for a pasture -/
structure PastureRental where
  total_rent : ℚ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  c_months : ℕ

/-- Calculates the share of rent for person b given a PastureRental arrangement -/
def calculate_b_share (rental : PastureRental) : ℚ :=
  let total_horse_months := rental.a_horses * rental.a_months +
                            rental.b_horses * rental.b_months +
                            rental.c_horses * rental.c_months
  let cost_per_horse_month := rental.total_rent / total_horse_months
  (rental.b_horses * rental.b_months : ℚ) * cost_per_horse_month

/-- Theorem stating that b's share of the rent is 180 for the given arrangement -/
theorem b_share_is_180 (rental : PastureRental)
  (h1 : rental.total_rent = 435)
  (h2 : rental.a_horses = 12) (h3 : rental.a_months = 8)
  (h4 : rental.b_horses = 16) (h5 : rental.b_months = 9)
  (h6 : rental.c_horses = 18) (h7 : rental.c_months = 6) :
  calculate_b_share rental = 180 := by
  sorry

end b_share_is_180_l2688_268866


namespace inscribed_half_area_rhombus_l2688_268861

/-- A centrally symmetric convex polygon -/
structure CentrallySymmetricConvexPolygon where
  -- Add necessary fields and properties
  area : ℝ
  centrally_symmetric : Bool
  convex : Bool

/-- A rhombus -/
structure Rhombus where
  -- Add necessary fields
  area : ℝ

/-- A rhombus is inscribed in a polygon -/
def is_inscribed (r : Rhombus) (p : CentrallySymmetricConvexPolygon) : Prop :=
  sorry

/-- Main theorem: For any centrally symmetric convex polygon, 
    there exists an inscribed rhombus with half the area of the polygon -/
theorem inscribed_half_area_rhombus (p : CentrallySymmetricConvexPolygon) :
  ∃ r : Rhombus, is_inscribed r p ∧ r.area = p.area / 2 :=
sorry

end inscribed_half_area_rhombus_l2688_268861


namespace solve_pencil_problem_l2688_268821

def pencil_problem (total_pencils : ℕ) (buy_price sell_price : ℚ) (desired_profit : ℚ) (pencils_to_sell : ℕ) : Prop :=
  let total_cost : ℚ := total_pencils * buy_price
  let revenue : ℚ := pencils_to_sell * sell_price
  let actual_profit : ℚ := revenue - total_cost
  actual_profit = desired_profit

theorem solve_pencil_problem :
  pencil_problem 2000 (15/100) (30/100) 180 1600 := by
  sorry

end solve_pencil_problem_l2688_268821


namespace find_a_minus_b_l2688_268881

-- Define the functions
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 7
def h (a b x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem find_a_minus_b (a b : ℝ) :
  (∀ x, h a b x = x - 9) → a - b = 7 := by
  sorry

end find_a_minus_b_l2688_268881


namespace geometric_sequence_sum_l2688_268816

/-- Given a geometric sequence {aₙ} with common ratio 2 and a₁ + a₃ = 5, prove that a₂ + a₄ = 10 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  a 1 + a 3 = 5 →               -- given condition
  a 2 + a 4 = 10 :=             -- conclusion to prove
by sorry

end geometric_sequence_sum_l2688_268816


namespace marker_distance_l2688_268896

theorem marker_distance (k : ℝ) (h1 : k > 0) 
  (h2 : ∀ n : ℕ+, Real.sqrt ((4:ℝ)^2 + (4*k)^2) = 31) : 
  Real.sqrt ((12:ℝ)^2 + (12*k)^2) = 93 := by sorry

end marker_distance_l2688_268896


namespace book_reading_increase_l2688_268868

theorem book_reading_increase (matt_last_year matt_this_year pete_last_year pete_this_year : ℕ) 
  (h1 : pete_last_year = 2 * matt_last_year)
  (h2 : pete_this_year = 2 * pete_last_year)
  (h3 : pete_last_year + pete_this_year = 300)
  (h4 : matt_this_year = 75) :
  (matt_this_year - matt_last_year) * 100 / matt_last_year = 50 := by
  sorry

end book_reading_increase_l2688_268868


namespace sandwich_cost_is_two_l2688_268822

/-- Calculates the cost per sandwich given the prices and discounts for ingredients -/
def cost_per_sandwich (bread_price : ℚ) (meat_price : ℚ) (cheese_price : ℚ) 
  (meat_discount : ℚ) (cheese_discount : ℚ) (num_sandwiches : ℕ) : ℚ :=
  let total_cost := bread_price + 2 * meat_price + 2 * cheese_price - meat_discount - cheese_discount
  total_cost / num_sandwiches

/-- Proves that the cost per sandwich is $2.00 given the specified conditions -/
theorem sandwich_cost_is_two :
  cost_per_sandwich 4 5 4 1 1 10 = 2 := by
  sorry

end sandwich_cost_is_two_l2688_268822


namespace complex_expression_odd_exponent_l2688_268895

theorem complex_expression_odd_exponent (n : ℕ) (h : Odd n) :
  (((1 + Complex.I) / (1 - Complex.I)) ^ (2 * n) + 
   ((1 - Complex.I) / (1 + Complex.I)) ^ (2 * n)) = -2 := by
  sorry

end complex_expression_odd_exponent_l2688_268895


namespace no_rational_squares_l2688_268820

def sequence_a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => sequence_a n + 2 / sequence_a n

theorem no_rational_squares :
  ∀ n : ℕ, ∀ r : ℚ, sequence_a n ≠ r^2 := by
  sorry

end no_rational_squares_l2688_268820


namespace inequality_proof_l2688_268827

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  Real.sqrt (a * b^2 + a^2 * b) + Real.sqrt ((1 - a) * (1 - b)^2 + (1 - a)^2 * (1 - b)) < Real.sqrt 2 := by
  sorry

end inequality_proof_l2688_268827


namespace quarters_sale_amount_l2688_268843

/-- The amount received for selling quarters at a percentage of their face value -/
def amount_received (num_quarters : ℕ) (face_value : ℚ) (percentage : ℕ) : ℚ :=
  (num_quarters : ℚ) * face_value * ((percentage : ℚ) / 100)

/-- Theorem stating that selling 8 quarters with face value $0.25 at 500% yields $10 -/
theorem quarters_sale_amount : 
  amount_received 8 (1/4) 500 = 10 := by sorry

end quarters_sale_amount_l2688_268843


namespace village_population_problem_l2688_268849

theorem village_population_problem (X : ℝ) : 
  (X > 0) →
  (0.9 * X * 0.75 + 0.9 * X * 0.25 * 0.15 = 5265) →
  X = 7425 := by
  sorry

end village_population_problem_l2688_268849


namespace richard_david_age_difference_l2688_268891

-- Define the ages of the three sons
def david_age : ℕ := 14
def scott_age : ℕ := david_age - 8
def richard_age : ℕ := scott_age * 2 + 8

-- Define the conditions
theorem richard_david_age_difference :
  richard_age - david_age = 6 :=
by
  -- Proof goes here
  sorry

#check richard_david_age_difference

end richard_david_age_difference_l2688_268891


namespace volume_ratio_specific_cone_l2688_268882

/-- Represents a right circular cone -/
structure Cone where
  base_diameter : ℝ
  height : ℝ

/-- Represents a plane intersecting the cone -/
structure IntersectingPlane where
  distance_from_apex : ℝ

/-- Calculates the volume ratio of the two parts resulting from intersecting a cone with a plane -/
def volume_ratio (cone : Cone) (plane : IntersectingPlane) : ℝ × ℝ :=
  sorry

/-- Theorem stating the volume ratio for the given cone and intersecting plane -/
theorem volume_ratio_specific_cone :
  let cone : Cone := { base_diameter := 26, height := 39 }
  let plane : IntersectingPlane := { distance_from_apex := 30 }
  volume_ratio cone plane = (0.4941, 0.5059) :=
sorry

end volume_ratio_specific_cone_l2688_268882


namespace prime_sum_of_powers_l2688_268842

theorem prime_sum_of_powers (n : ℕ) : 
  (∃ (a b c : ℤ), a + b + c = 0 ∧ Nat.Prime (Int.natAbs (a^n + b^n + c^n))) ↔ Even n := by
  sorry

end prime_sum_of_powers_l2688_268842


namespace equation_solution_l2688_268855

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 - Real.sqrt (16 + 8*x)) + Real.sqrt (5 - Real.sqrt (5 + x)) = 3 + Real.sqrt 5 :=
by
  use 4
  sorry

end equation_solution_l2688_268855


namespace fraction_simplification_l2688_268836

theorem fraction_simplification (x : ℝ) : (2*x + 3) / 4 + (4 - 2*x) / 3 = (-2*x + 25) / 12 := by
  sorry

end fraction_simplification_l2688_268836


namespace equation_roots_l2688_268847

theorem equation_roots :
  let f : ℝ → ℝ := λ x => (x^3 - 3*x^2 + x - 2)*(x^3 - x^2 - 4*x + 7) + 6*x^2 - 15*x + 18
  ∃ (a b c d e : ℝ), 
    (a = 1 ∧ b = 2 ∧ c = -2 ∧ d = 1 + Real.sqrt 2 ∧ e = 1 - Real.sqrt 2) ∧
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) :=
by sorry

end equation_roots_l2688_268847


namespace num_special_words_is_35280_l2688_268809

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of consonants in the English alphabet -/
def num_consonants : ℕ := 21

/-- The number of six-letter words that begin and end with the same vowel,
    alternate between vowels and consonants, and start with a vowel -/
def num_special_words : ℕ := num_vowels * num_consonants * (num_vowels - 1) * num_consonants * (num_vowels - 1)

/-- Theorem stating that the number of special words is 35280 -/
theorem num_special_words_is_35280 : num_special_words = 35280 := by sorry

end num_special_words_is_35280_l2688_268809


namespace min_distance_vectors_l2688_268854

def a (t : ℝ) : Fin 3 → ℝ := ![2, t, t]
def b (t : ℝ) : Fin 3 → ℝ := ![1 - t, 2 * t - 1, 0]

theorem min_distance_vectors (t : ℝ) :
  ∃ (min : ℝ), min = Real.sqrt 2 ∧ ∀ (s : ℝ), ‖b s - a s‖ ≥ min := by sorry

end min_distance_vectors_l2688_268854


namespace cube_has_eight_vertices_l2688_268807

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- The number of vertices in a cube -/
def num_vertices (c : Cube) : ℕ := 8

/-- Theorem: A cube has 8 vertices -/
theorem cube_has_eight_vertices (c : Cube) : num_vertices c = 8 := by
  sorry

end cube_has_eight_vertices_l2688_268807


namespace right_triangle_side_length_l2688_268892

theorem right_triangle_side_length 
  (west_distance : ℝ) 
  (total_distance : ℝ) 
  (h1 : west_distance = 10) 
  (h2 : total_distance = 14.142135623730951) : 
  ∃ (north_distance : ℝ), 
    north_distance^2 + west_distance^2 = total_distance^2 ∧ 
    north_distance = 10 :=
by sorry

end right_triangle_side_length_l2688_268892


namespace arithmetic_sequence_2023rd_term_l2688_268851

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2023rd_term 
  (p q : ℝ) 
  (h1 : arithmetic_sequence p (3*p - q - p) 1 = p)
  (h2 : arithmetic_sequence p (3*p - q - p) 2 = 3*p - q)
  (h3 : arithmetic_sequence p (3*p - q - p) 3 = 9)
  (h4 : arithmetic_sequence p (3*p - q - p) 4 = 3*p + q) :
  arithmetic_sequence p (3*p - q - p) 2023 = 18189 := by
sorry

end arithmetic_sequence_2023rd_term_l2688_268851


namespace quadratic_roots_sum_product_l2688_268898

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9 ∧ x * y = 20) →
  m + n = 87 := by
sorry

end quadratic_roots_sum_product_l2688_268898


namespace tangent_parallel_points_l2688_268883

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, (f x = y ∧ (3 * x^2 + 1 = 4)) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end tangent_parallel_points_l2688_268883


namespace repeating_decimal_sum_l2688_268839

theorem repeating_decimal_sum (x : ℚ) : 
  (∃ (n : ℕ), x = (457 : ℚ) / (10^n * 999)) → 
  (∃ (a b : ℕ), x = a / b ∧ a + b = 1456) :=
by sorry

end repeating_decimal_sum_l2688_268839
