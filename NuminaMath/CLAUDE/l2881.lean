import Mathlib

namespace reflection_line_sum_l2881_288189

/-- Given a line y = mx + b, if the reflection of point (1, 2) across this line is (7, 6), then m + b = 8.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    (x - 1)^2 + (y - 2)^2 = (7 - x)^2 + (6 - y)^2 ∧ 
    (x + 7) / 2 = (y + 6) / 2 / m + b ∧
    (y + 6) / 2 = m * (x + 7) / 2 + b) → 
  m + b = 8.5 := by sorry

end reflection_line_sum_l2881_288189


namespace bird_count_difference_l2881_288127

/-- Represents the count of birds on a single day -/
structure DailyCount where
  bluejays : ℕ
  cardinals : ℕ

/-- Calculates the difference between cardinals and blue jays for a single day -/
def dailyDifference (count : DailyCount) : ℤ :=
  count.cardinals - count.bluejays

/-- Theorem: The total difference between cardinals and blue jays over three days is 3 -/
theorem bird_count_difference (day1 day2 day3 : DailyCount)
  (h1 : day1 = { bluejays := 2, cardinals := 3 })
  (h2 : day2 = { bluejays := 3, cardinals := 3 })
  (h3 : day3 = { bluejays := 2, cardinals := 4 }) :
  dailyDifference day1 + dailyDifference day2 + dailyDifference day3 = 3 := by
  sorry

#eval dailyDifference { bluejays := 2, cardinals := 3 } +
      dailyDifference { bluejays := 3, cardinals := 3 } +
      dailyDifference { bluejays := 2, cardinals := 4 }

end bird_count_difference_l2881_288127


namespace sum_of_squares_problem_l2881_288145

theorem sum_of_squares_problem (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_sum_squares : a^2 + b^2 + c^2 = 48) 
  (h_sum_products : a*b + b*c + c*a = 26) : 
  a + b + c = 10 := by sorry

end sum_of_squares_problem_l2881_288145


namespace consultant_decision_probability_l2881_288120

theorem consultant_decision_probability :
  let p : ℝ := 0.8  -- probability of each consultant being correct
  let n : ℕ := 3    -- number of consultants
  let k : ℕ := 2    -- minimum number of correct opinions for a correct decision
  -- probability of making the correct decision
  (Finset.sum (Finset.range (n + 1 - k)) (λ i => 
    (n.choose (n - i)) * p^(n - i) * (1 - p)^i)) = 0.896 := by
  sorry

end consultant_decision_probability_l2881_288120


namespace necessary_but_not_sufficient_l2881_288140

theorem necessary_but_not_sufficient :
  (∃ x : ℝ, x > 1 ∧ ¬(Real.log (2^x) > 1)) ∧
  (∀ x : ℝ, Real.log (2^x) > 1 → x > 1) := by
  sorry

end necessary_but_not_sufficient_l2881_288140


namespace smallest_divisible_by_1_to_12_l2881_288146

def is_divisible_by_range (n : ℕ) (a b : ℕ) : Prop :=
  ∀ i : ℕ, a ≤ i → i ≤ b → n % i = 0

theorem smallest_divisible_by_1_to_12 :
  ∃ (n : ℕ), n > 0 ∧ is_divisible_by_range n 1 12 ∧
  ∀ (m : ℕ), m > 0 → is_divisible_by_range m 1 12 → n ≤ m :=
by
  use 27720
  sorry

end smallest_divisible_by_1_to_12_l2881_288146


namespace product_of_fractions_l2881_288194

theorem product_of_fractions : 
  (1/2 : ℚ) * (9/1 : ℚ) * (1/8 : ℚ) * (64/1 : ℚ) * (1/128 : ℚ) * (729/1 : ℚ) * (1/2187 : ℚ) * (19683/1 : ℚ) = 59049/32 := by
  sorry

end product_of_fractions_l2881_288194


namespace negation_of_implication_l2881_288155

theorem negation_of_implication (a b : ℝ) : 
  ¬(ab = 2 → a^2 + b^2 ≥ 4) ↔ (ab ≠ 2 → a^2 + b^2 < 4) := by
  sorry

end negation_of_implication_l2881_288155


namespace max_distance_A_B_l2881_288125

def set_A : Set ℂ := {z : ℂ | z^4 - 16 = 0}
def set_B : Set ℂ := {z : ℂ | z^3 - 12*z^2 + 36*z - 64 = 0}

theorem max_distance_A_B : 
  ∃ (a : ℂ) (b : ℂ), a ∈ set_A ∧ b ∈ set_B ∧ 
    Complex.abs (a - b) = 10 ∧
    ∀ (x : ℂ) (y : ℂ), x ∈ set_A → y ∈ set_B → Complex.abs (x - y) ≤ 10 :=
sorry

end max_distance_A_B_l2881_288125


namespace cannot_reach_2000_l2881_288161

theorem cannot_reach_2000 (a b : ℕ) : a * 12 + b * 17 ≠ 2000 := by
  sorry

end cannot_reach_2000_l2881_288161


namespace divisible_by_18_sqrt_between_30_and_30_5_l2881_288165

theorem divisible_by_18_sqrt_between_30_and_30_5 : 
  ∀ n : ℕ, 
    n > 0 ∧ 
    n % 18 = 0 ∧ 
    30 < Real.sqrt n ∧ 
    Real.sqrt n < 30.5 → 
    n = 900 ∨ n = 918 :=
by sorry

end divisible_by_18_sqrt_between_30_and_30_5_l2881_288165


namespace probability_no_empty_boxes_l2881_288198

/-- The number of distinct balls -/
def num_balls : ℕ := 3

/-- The number of distinct boxes -/
def num_boxes : ℕ := 3

/-- The probability of placing balls into boxes with no empty boxes -/
def prob_no_empty_boxes : ℚ := 2/9

/-- Theorem stating that the probability of placing 3 distinct balls into 3 distinct boxes
    with no empty boxes is 2/9 -/
theorem probability_no_empty_boxes :
  (num_balls = 3 ∧ num_boxes = 3) →
  prob_no_empty_boxes = 2/9 := by
  sorry

end probability_no_empty_boxes_l2881_288198


namespace quadratic_equivalence_l2881_288185

theorem quadratic_equivalence :
  ∀ x : ℝ, (x^2 - 6*x + 4 = 0) ↔ ((x - 3)^2 = 5) :=
by sorry

end quadratic_equivalence_l2881_288185


namespace salesman_profit_salesman_profit_is_442_l2881_288133

/-- Calculates the profit of a salesman selling backpacks --/
theorem salesman_profit (total_backpacks : ℕ) (total_cost : ℕ) 
  (first_sale_quantity : ℕ) (first_sale_price : ℕ)
  (second_sale_quantity : ℕ) (second_sale_price : ℕ)
  (remaining_price : ℕ) : ℕ :=
  let remaining_quantity := total_backpacks - first_sale_quantity - second_sale_quantity
  let total_revenue := 
    first_sale_quantity * first_sale_price +
    second_sale_quantity * second_sale_price +
    remaining_quantity * remaining_price
  total_revenue - total_cost

/-- The salesman's profit is $442 --/
theorem salesman_profit_is_442 : 
  salesman_profit 48 576 17 18 10 25 22 = 442 := by
  sorry

end salesman_profit_salesman_profit_is_442_l2881_288133


namespace ellipse_condition_l2881_288105

/-- The equation of the graph --/
def equation (x y k : ℝ) : Prop :=
  x^2 + 4*y^2 - 10*x + 56*y = k

/-- Definition of a non-degenerate ellipse --/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ 
    ∀ x y : ℝ, equation x y k ↔ (x - c)^2 / a + (y - d)^2 / b = e

/-- The main theorem --/
theorem ellipse_condition (k : ℝ) :
  is_non_degenerate_ellipse k ↔ k > -221 :=
sorry

end ellipse_condition_l2881_288105


namespace expression_value_l2881_288134

theorem expression_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end expression_value_l2881_288134


namespace circumcircle_radius_from_centroid_distance_l2881_288151

/-- Given a triangle ABC with sides a, b, c, where c = AB, prove that if
    (b - c) / (a + c) = (c - a) / (b + c), then the radius R of the circumcircle
    satisfies R² = d² + c²/3, where d is the distance from the circumcircle
    center to the centroid of the triangle. -/
theorem circumcircle_radius_from_centroid_distance (a b c d : ℝ) :
  (b - c) / (a + c) = (c - a) / (b + c) →
  ∃ (R : ℝ), R > 0 ∧ R^2 = d^2 + c^2 / 3 :=
sorry

end circumcircle_radius_from_centroid_distance_l2881_288151


namespace ratio_200_percent_l2881_288191

theorem ratio_200_percent (x : ℝ) : (6 : ℝ) / x = 2 → x = 3 :=
  sorry

end ratio_200_percent_l2881_288191


namespace village_population_l2881_288112

/-- Represents the vampire population dynamics in a village --/
structure VampireVillage where
  initialVampires : ℕ
  initialPopulation : ℕ
  vampiresPerNight : ℕ
  nightsPassed : ℕ
  finalVampires : ℕ

/-- Theorem stating the initial population of the village --/
theorem village_population (v : VampireVillage) 
  (h1 : v.initialVampires = 2)
  (h2 : v.vampiresPerNight = 5)
  (h3 : v.nightsPassed = 2)
  (h4 : v.finalVampires = 72) :
  v.initialPopulation = 72 := by
  sorry

end village_population_l2881_288112


namespace hexagon_areas_equal_l2881_288129

/-- Given a triangle T with sides of lengths r, g, and b, and area S,
    the area of both hexagons formed by extending the sides of T is equal to
    S * (4 + ((r^2 + g^2 + b^2)(r + g + b)) / (r * g * b)) -/
theorem hexagon_areas_equal (r g b S : ℝ) (hr : r > 0) (hg : g > 0) (hb : b > 0) (hS : S > 0) :
  let hexagon_area := S * (4 + ((r^2 + g^2 + b^2) * (r + g + b)) / (r * g * b))
  ∀ (area1 area2 : ℝ), area1 = hexagon_area ∧ area2 = hexagon_area → area1 = area2 := by
  sorry

end hexagon_areas_equal_l2881_288129


namespace fraction_simplification_l2881_288116

theorem fraction_simplification (x : ℝ) (h : x ≠ -2 ∧ x ≠ 2) :
  (x^2 - 4) / (x^2 - 4*x + 4) / ((x^2 + 4*x + 4) / (2*x - x^2)) = -x / (x + 2)^2 := by
  sorry

end fraction_simplification_l2881_288116


namespace subset_of_sqrt_two_in_sqrt_three_set_l2881_288157

theorem subset_of_sqrt_two_in_sqrt_three_set :
  {Real.sqrt 2} ⊆ {x : ℝ | x ≤ Real.sqrt 3} := by sorry

end subset_of_sqrt_two_in_sqrt_three_set_l2881_288157


namespace second_grade_years_l2881_288152

/-- Given information about Mrs. Randall's teaching career -/
def total_teaching_years : ℕ := 26
def third_grade_years : ℕ := 18

/-- Theorem stating the number of years Mrs. Randall taught second grade -/
theorem second_grade_years : total_teaching_years - third_grade_years = 8 := by
  sorry

end second_grade_years_l2881_288152


namespace flower_beds_count_l2881_288173

/-- Calculates the total number of flower beds in a garden with three sections. -/
def totalFlowerBeds (seeds1 seeds2 seeds3 : ℕ) (seedsPerBed1 seedsPerBed2 seedsPerBed3 : ℕ) : ℕ :=
  (seeds1 / seedsPerBed1) + (seeds2 / seedsPerBed2) + (seeds3 / seedsPerBed3)

/-- Proves that the total number of flower beds is 105 given the specific conditions. -/
theorem flower_beds_count :
  totalFlowerBeds 470 320 210 10 10 8 = 105 := by
  sorry

#eval totalFlowerBeds 470 320 210 10 10 8

end flower_beds_count_l2881_288173


namespace max_value_quadratic_expression_l2881_288199

/-- Given a system of equations, prove that the maximum value of a quadratic expression is 11 -/
theorem max_value_quadratic_expression (x y z : ℝ) 
  (eq1 : x - y + z - 1 = 0)
  (eq2 : x * y + 2 * z^2 - 6 * z + 1 = 0) :
  ∃ (max : ℝ), max = 11 ∧ ∀ (x' y' z' : ℝ), 
    x' - y' + z' - 1 = 0 → 
    x' * y' + 2 * z'^2 - 6 * z' + 1 = 0 → 
    (x' - 1)^2 + (y' + 1)^2 ≤ max :=
by sorry

end max_value_quadratic_expression_l2881_288199


namespace sum_of_coordinates_l2881_288109

/-- Given a point C with coordinates (3, k), its reflection D over the y-axis
    with y-coordinate increased by 4, prove that the sum of all coordinates
    of C and D is 2k + 4. -/
theorem sum_of_coordinates (k : ℝ) : 
  let C : ℝ × ℝ := (3, k)
  let D : ℝ × ℝ := (-3, k + 4)
  (C.1 + C.2 + D.1 + D.2) = 2 * k + 4 :=
by sorry

end sum_of_coordinates_l2881_288109


namespace snowman_volume_snowman_volume_calculation_l2881_288163

theorem snowman_volume (π : ℝ) : ℝ → ℝ → ℝ → ℝ :=
  fun r₁ r₂ r₃ =>
    let sphere_volume := fun r : ℝ => (4 / 3) * π * r^3
    sphere_volume r₁ + sphere_volume r₂ + sphere_volume r₃

theorem snowman_volume_calculation (π : ℝ) :
  snowman_volume π 4 5 6 = (1620 / 3) * π := by
  sorry

end snowman_volume_snowman_volume_calculation_l2881_288163


namespace files_sorted_in_one_and_half_hours_l2881_288114

/-- Represents the number of files sorted by a group of clerks under specific conditions. -/
def filesSortedInOneAndHalfHours (totalFiles : ℕ) (filesPerHourPerClerk : ℕ) (totalTime : ℚ) : ℕ :=
  let initialClerks := 22  -- Derived from the problem conditions
  let reassignedClerks := 3  -- Derived from the problem conditions
  initialClerks * filesPerHourPerClerk + (initialClerks - reassignedClerks) * (filesPerHourPerClerk / 2)

/-- Proves that under the given conditions, the number of files sorted in 1.5 hours is 945. -/
theorem files_sorted_in_one_and_half_hours :
  filesSortedInOneAndHalfHours 1775 30 (157/60) = 945 := by
  sorry

#eval filesSortedInOneAndHalfHours 1775 30 (157/60)

end files_sorted_in_one_and_half_hours_l2881_288114


namespace james_height_fraction_l2881_288117

/-- Proves that James was 2/3 as tall as his uncle before the growth spurt -/
theorem james_height_fraction (uncle_height : ℝ) (james_growth : ℝ) (height_difference : ℝ) :
  uncle_height = 72 →
  james_growth = 10 →
  height_difference = 14 →
  (uncle_height - (james_growth + height_difference)) / uncle_height = 2 / 3 := by
  sorry

end james_height_fraction_l2881_288117


namespace zero_not_identity_for_star_l2881_288101

-- Define the set S
def S : Set ℝ := {x : ℝ | x ≠ -1/3}

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * a * b + 1

-- Theorem statement
theorem zero_not_identity_for_star :
  ¬(∀ a ∈ S, (star 0 a = a ∧ star a 0 = a)) :=
sorry

end zero_not_identity_for_star_l2881_288101


namespace difference_of_squares_l2881_288132

theorem difference_of_squares (x y : ℝ) : x^2 - 4*y^2 = (x - 2*y) * (x + 2*y) := by
  sorry

end difference_of_squares_l2881_288132


namespace window_side_length_l2881_288123

/-- Represents the dimensions of a window pane -/
structure Pane where
  height : ℝ
  width : ℝ

/-- Represents the dimensions and properties of a window -/
structure Window where
  paneCount : ℕ
  rows : ℕ
  columns : ℕ
  pane : Pane
  borderWidth : ℝ

/-- The theorem stating that given the specified conditions, the window's side length is 27 inches -/
theorem window_side_length (w : Window) : 
  w.paneCount = 8 ∧ 
  w.rows = 2 ∧ 
  w.columns = 4 ∧ 
  w.pane.height = 3 * w.pane.width ∧
  w.borderWidth = 3 →
  (w.columns * w.pane.width + (w.columns + 1) * w.borderWidth : ℝ) = 27 :=
by sorry


end window_side_length_l2881_288123


namespace no_natural_square_diff_2014_l2881_288184

theorem no_natural_square_diff_2014 : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end no_natural_square_diff_2014_l2881_288184


namespace solve_for_a_l2881_288141

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 8*a^2

-- Define the theorem
theorem solve_for_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a > 0)
  (h₂ : ∀ x, f a x < 0 ↔ x₁ < x ∧ x < x₂)
  (h₃ : x₂ - x₁ = 15) :
  a = 5/2 := by
sorry

end solve_for_a_l2881_288141


namespace ninth_term_of_sequence_l2881_288162

/-- The nth term of a geometric sequence with first term a and common ratio r -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 9th term of the geometric sequence with first term 4 and common ratio 1 is 4 -/
theorem ninth_term_of_sequence : geometric_sequence 4 1 9 = 4 := by
  sorry

end ninth_term_of_sequence_l2881_288162


namespace visitors_in_scientific_notation_l2881_288147

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem visitors_in_scientific_notation :
  toScientificNotation 203000 = ScientificNotation.mk 2.03 5 sorry := by
  sorry

end visitors_in_scientific_notation_l2881_288147


namespace lemonade_glasses_l2881_288100

/-- The number of glasses of lemonade that can be made -/
def glasses_of_lemonade (total_lemons : ℕ) (lemons_per_glass : ℕ) : ℕ :=
  total_lemons / lemons_per_glass

/-- Theorem: Given 18 lemons and 2 lemons required per glass, 9 glasses of lemonade can be made -/
theorem lemonade_glasses : glasses_of_lemonade 18 2 = 9 := by
  sorry

end lemonade_glasses_l2881_288100


namespace patio_table_cost_l2881_288118

/-- The cost of the patio table given the total cost and chair costs -/
theorem patio_table_cost (total_cost : ℕ) (chair_cost : ℕ) (num_chairs : ℕ) :
  total_cost = 135 →
  chair_cost = 20 →
  num_chairs = 4 →
  total_cost - (num_chairs * chair_cost) = 55 :=
by sorry

end patio_table_cost_l2881_288118


namespace prove_a_value_l2881_288197

theorem prove_a_value (A B : Set ℤ) (a : ℤ) : 
  A = {0, 1} → 
  B = {-1, 0, a+3} → 
  A ⊆ B → 
  a = -2 := by sorry

end prove_a_value_l2881_288197


namespace sum_in_range_l2881_288160

theorem sum_in_range : 
  let sum := (17/4 : ℚ) + (11/4 : ℚ) + (57/8 : ℚ)
  14 < sum ∧ sum < 15 := by
  sorry

end sum_in_range_l2881_288160


namespace condition_relationship_l2881_288169

theorem condition_relationship (x : ℝ) : 
  (∀ x, x^2 - 2*x + 1 ≤ 0 → x > 0) ∧ 
  (∃ x, x > 0 ∧ x^2 - 2*x + 1 > 0) :=
by sorry

end condition_relationship_l2881_288169


namespace cone_volume_from_half_sector_l2881_288180

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  let base_radius : ℝ := r / 2
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let cone_volume : ℝ := (1/3) * π * base_radius^2 * cone_height
  cone_volume = 9 * π * Real.sqrt 3 := by
  sorry

end cone_volume_from_half_sector_l2881_288180


namespace f_decreasing_interval_l2881_288103

/-- The function f(x) = x^2(ax + b) where a and b are real numbers. -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 * (a * x + b)

/-- The derivative of f(x) -/
def f_prime (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem f_decreasing_interval (a b : ℝ) :
  (f_prime a b 2 = 0) →  -- f has an extremum at x = 2
  (f_prime a b 1 = -3) →  -- tangent line at (1, f(1)) is parallel to 3x + y = 0
  ∀ x, 0 < x → x < 2 → f_prime a b x < 0 :=
by sorry

end f_decreasing_interval_l2881_288103


namespace mission_duration_l2881_288164

theorem mission_duration (planned_duration : ℝ) : 
  (1.6 * planned_duration + 3 = 11) → planned_duration = 5 := by
  sorry

end mission_duration_l2881_288164


namespace arithmetic_geometric_inequality_l2881_288167

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

/-- A geometric sequence with positive terms -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q ∧ b n > 0

theorem arithmetic_geometric_inequality (a b : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) (h_geom : GeometricSequence b)
    (h_eq1 : a 1 = b 1) (h_eq2 : a 2 = b 2) (h_neq : a 1 ≠ a 2) :
    ∀ n : ℕ, n ≥ 3 → a n < b n := by
  sorry

end arithmetic_geometric_inequality_l2881_288167


namespace sqrt_seven_irrational_negative_one_third_rational_two_rational_decimal_rational_irrational_among_options_l2881_288107

theorem sqrt_seven_irrational :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 7 = p / q) :=
by
  sorry

theorem negative_one_third_rational :
  ∃ (p q : ℤ), q ≠ 0 ∧ (-1 : ℚ) / 3 = p / q :=
by
  sorry

theorem two_rational :
  ∃ (p q : ℤ), q ≠ 0 ∧ (2 : ℚ) = p / q :=
by
  sorry

theorem decimal_rational :
  ∃ (p q : ℤ), q ≠ 0 ∧ (0.0101 : ℚ) = p / q :=
by
  sorry

theorem irrational_among_options :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 7 = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-1 : ℚ) / 3 = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (2 : ℚ) = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (0.0101 : ℚ) = p / q) :=
by
  sorry

end sqrt_seven_irrational_negative_one_third_rational_two_rational_decimal_rational_irrational_among_options_l2881_288107


namespace slope_product_no_circle_through_A_l2881_288142

-- Define the ellipse
def E (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define a point P on the ellipse
def P (x₀ y₀ : ℝ) : Prop := E x₀ y₀ ∧ (x₀, y₀) ≠ A ∧ (x₀, y₀) ≠ B

-- Theorem: Product of slopes of PA and PB is -1/4
theorem slope_product (x₀ y₀ : ℝ) (h : P x₀ y₀) :
  (y₀ / (x₀ + 2)) * (y₀ / (x₀ - 2)) = -1/4 := by sorry

-- No circle with diameter MN passes through A
-- This part is more complex and would require additional definitions and theorems
-- We'll represent it as a proposition without proof
theorem no_circle_through_A (M N : ℝ × ℝ) (hM : E M.1 M.2) (hN : E N.1 N.2) :
  ¬∃ (center : ℝ × ℝ) (radius : ℝ), 
    (center.1 - M.1)^2 + (center.2 - M.2)^2 = radius^2 ∧
    (center.1 - N.1)^2 + (center.2 - N.2)^2 = radius^2 ∧
    (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2 := by sorry

end slope_product_no_circle_through_A_l2881_288142


namespace sum_of_fourth_powers_is_square_l2881_288166

theorem sum_of_fourth_powers_is_square (a b c : ℤ) (h : a + b + c = 0) :
  ∃ p : ℤ, 2 * a^4 + 2 * b^4 + 2 * c^4 = 4 * p^2 := by
sorry

end sum_of_fourth_powers_is_square_l2881_288166


namespace factor_theorem_l2881_288111

def Q (d : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + d*x + 20

theorem factor_theorem (d : ℝ) : (∀ x, (x - 4) ∣ Q d x) → d = -33 := by
  sorry

end factor_theorem_l2881_288111


namespace quadratic_decreasing_after_vertex_l2881_288183

def f (x : ℝ) : ℝ := -(x - 2)^2 - 7

theorem quadratic_decreasing_after_vertex :
  ∀ (x1 x2 : ℝ), x1 > 2 → x2 > x1 → f x2 < f x1 := by
  sorry

end quadratic_decreasing_after_vertex_l2881_288183


namespace pairwise_disjoint_sequences_l2881_288122

def largest_prime_power_divisor (n : ℕ) : ℕ := sorry

theorem pairwise_disjoint_sequences 
  (n : Fin 10000 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → n i ≠ n j) 
  (h_distinct_lpd : ∀ i j, i ≠ j → 
    largest_prime_power_divisor (n i) ≠ largest_prime_power_divisor (n j)) :
  ∃ a : Fin 10000 → ℤ, ∀ i j k l, i ≠ j → 
    a i + k * n i ≠ a j + l * n j :=
sorry

end pairwise_disjoint_sequences_l2881_288122


namespace ivan_dice_count_l2881_288113

theorem ivan_dice_count (x : ℕ) : 
  x + 2*x = 60 → x = 20 := by
  sorry

end ivan_dice_count_l2881_288113


namespace find_a_l2881_288182

theorem find_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 7) (h3 : c = 4) : a = 1 := by
  sorry

end find_a_l2881_288182


namespace travel_group_combinations_l2881_288115

def total_friends : ℕ := 12
def friends_to_choose : ℕ := 5
def previously_traveled_friends : ℕ := 6

theorem travel_group_combinations : 
  (total_friends.choose friends_to_choose) - 
  ((total_friends - previously_traveled_friends).choose friends_to_choose) = 786 := by
  sorry

end travel_group_combinations_l2881_288115


namespace quadratic_function_proof_l2881_288131

theorem quadratic_function_proof :
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 3
  ∀ a b c : ℝ, a ≠ 0 →
  (∀ x : ℝ, f x = a * x^2 + b * x + c) →
  f (-2) = 5 ∧ f (-1) = 0 ∧ f 0 = -3 ∧ f 1 = -4 ∧ f 2 = -3 :=
by sorry

end quadratic_function_proof_l2881_288131


namespace system_solution_l2881_288171

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3*x + Real.sqrt (3*x - y) + y = 6
def equation2 (x y : ℝ) : Prop := 9*x^2 + 3*x - y - y^2 = 36

-- Define the solution set
def solutions : Set (ℝ × ℝ) := {(2, -3), (6, -18)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
sorry

end system_solution_l2881_288171


namespace safe_mountain_climb_l2881_288186

theorem safe_mountain_climb : ∃ t : ℕ,
  t ≥ 0 ∧
  t % 26 ≠ 0 ∧ t % 26 ≠ 1 ∧
  t % 14 ≠ 0 ∧ t % 14 ≠ 1 ∧
  (t + 6) % 26 ≠ 0 ∧ (t + 6) % 26 ≠ 1 ∧
  (t + 6) % 14 ≠ 0 ∧ (t + 6) % 14 ≠ 1 ∧
  t + 24 < 26 * 14 := by
  sorry

end safe_mountain_climb_l2881_288186


namespace plaza_design_properties_l2881_288170

/-- Represents the plaza design and cost structure -/
structure PlazaDesign where
  sideLength : ℝ
  lightTileCost : ℝ
  darkTileCost : ℝ
  borderWidth : ℝ

/-- Calculates the total cost of materials for the plaza design -/
def totalCost (design : PlazaDesign) : ℝ :=
  sorry

/-- Calculates the side length of the central light square -/
def centralSquareSideLength (design : PlazaDesign) : ℝ :=
  sorry

/-- Theorem stating the properties of the plaza design -/
theorem plaza_design_properties (design : PlazaDesign) 
  (h1 : design.sideLength = 20)
  (h2 : design.lightTileCost = 100000)
  (h3 : design.darkTileCost = 300000)
  (h4 : design.borderWidth = 2)
  (h5 : totalCost design = 2 * (design.darkTileCost / 4)) :
  totalCost design = 150000 ∧ centralSquareSideLength design = 10.5 :=
sorry

end plaza_design_properties_l2881_288170


namespace function_equation_solution_l2881_288121

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) := by
sorry

end function_equation_solution_l2881_288121


namespace cos_alpha_value_l2881_288130

theorem cos_alpha_value (α : Real) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) :
  Real.cos α = 1 / 5 := by
  sorry

end cos_alpha_value_l2881_288130


namespace stock_investment_income_l2881_288137

theorem stock_investment_income 
  (investment : ℝ) 
  (stock_percentage : ℝ) 
  (stock_price : ℝ) 
  (face_value : ℝ) 
  (h1 : investment = 6800) 
  (h2 : stock_percentage = 0.20) 
  (h3 : stock_price = 136) 
  (h4 : face_value = 100) : 
  ∃ (annual_income : ℝ), 
    annual_income = 1000 ∧ 
    annual_income = (investment / stock_price) * (stock_percentage * face_value) :=
by
  sorry

end stock_investment_income_l2881_288137


namespace certain_number_equation_l2881_288181

theorem certain_number_equation (x : ℝ) : 5100 - (x / 20.4) = 5095 ↔ x = 102 := by
  sorry

end certain_number_equation_l2881_288181


namespace quadratic_inequality_range_l2881_288193

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end quadratic_inequality_range_l2881_288193


namespace ellipse_focus_k_value_l2881_288138

/-- Theorem: For an ellipse with equation x²/a² + y²/k = 1 and a focus at (0, √2), k = 2 -/
theorem ellipse_focus_k_value (a : ℝ) (k : ℝ) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / k = 1) →  -- Ellipse equation
  (0^2 / a^2 + (Real.sqrt 2)^2 / k = 1) →  -- Focus (0, √2) is on the ellipse
  k = 2 :=
by sorry

end ellipse_focus_k_value_l2881_288138


namespace light_path_in_cube_l2881_288128

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a light path in the cube -/
structure LightPath where
  start : Point3D
  reflection : Point3D
  length : ℝ

/-- Theorem stating the properties of the light path in the cube -/
theorem light_path_in_cube (cube : Cube) (path : LightPath) :
  cube.sideLength = 12 →
  path.start = Point3D.mk 0 0 0 →
  path.reflection = Point3D.mk 12 5 7 →
  ∃ (m n : ℕ), 
    path.length = m * Real.sqrt n ∧ 
    ¬ ∃ (p : ℕ), Prime p ∧ p^2 ∣ n ∧
    m + n = 230 := by
  sorry

end light_path_in_cube_l2881_288128


namespace f_strictly_increasing_and_odd_l2881_288176

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem f_strictly_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end f_strictly_increasing_and_odd_l2881_288176


namespace alex_hula_hoop_duration_l2881_288119

-- Define the hula hoop durations for each person
def nancy_duration : ℕ := 10

-- Casey's duration is 3 minutes less than Nancy's
def casey_duration : ℕ := nancy_duration - 3

-- Morgan's duration is three times Casey's duration
def morgan_duration : ℕ := casey_duration * 3

-- Alex's duration is the sum of Casey's and Morgan's durations minus 2 minutes
def alex_duration : ℕ := casey_duration + morgan_duration - 2

-- Theorem to prove Alex's hula hoop duration
theorem alex_hula_hoop_duration : alex_duration = 26 := by
  sorry

end alex_hula_hoop_duration_l2881_288119


namespace cubic_roots_divisibility_l2881_288106

theorem cubic_roots_divisibility (p a b c : ℤ) (hp : Prime p) 
  (ha : p ∣ a) (hb : p ∣ b) (hc : p ∣ c)
  (hroots : ∃ (r s : ℤ), r ≠ s ∧ r^3 + a*r^2 + b*r + c = 0 ∧ s^3 + a*s^2 + b*s + c = 0) :
  p^3 ∣ c := by
sorry

end cubic_roots_divisibility_l2881_288106


namespace empty_seats_calculation_l2881_288148

/-- Calculates the number of empty seats in a theater -/
def empty_seats (total_seats people_watching : ℕ) : ℕ :=
  total_seats - people_watching

/-- Theorem: The number of empty seats is the difference between total seats and people watching -/
theorem empty_seats_calculation (total_seats people_watching : ℕ) 
  (h1 : total_seats ≥ people_watching) :
  empty_seats total_seats people_watching = total_seats - people_watching :=
by sorry

end empty_seats_calculation_l2881_288148


namespace range_of_a_minus_b_l2881_288172

theorem range_of_a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 4) :
  ∀ x, x ∈ Set.Ioo (-3 : ℝ) 6 ↔ ∃ (a' b' : ℝ), 1 < a' ∧ a' < 4 ∧ -2 < b' ∧ b' < 4 ∧ x = a' - b' :=
by sorry

end range_of_a_minus_b_l2881_288172


namespace product_purely_imaginary_l2881_288188

theorem product_purely_imaginary (x : ℝ) :
  (∃ b : ℝ, (x + 2*Complex.I) * ((x + 1) + 2*Complex.I) * ((x + 2) + 2*Complex.I) * ((x + 3) + 2*Complex.I) = b * Complex.I) ↔
  x = -2 := by
sorry

end product_purely_imaginary_l2881_288188


namespace hexagon_minus_rhombus_area_l2881_288192

-- Define the regular hexagon
def regular_hexagon (area : ℝ) : Prop :=
  area > 0 ∧ ∃ (side : ℝ), area = (3 * Real.sqrt 3 / 2) * side^2

-- Define the rhombus inside the hexagon
def rhombus_in_hexagon (hexagon_area : ℝ) (rhombus_area : ℝ) : Prop :=
  ∃ (side : ℝ), 
    rhombus_area = 2 * (Real.sqrt 3 / 4) * (4 / 3 * 30 * Real.sqrt 3)

-- The theorem to be proved
theorem hexagon_minus_rhombus_area 
  (hexagon_area : ℝ) (rhombus_area : ℝ) (remaining_area : ℝ) :
  regular_hexagon hexagon_area →
  rhombus_in_hexagon hexagon_area rhombus_area →
  hexagon_area = 135 →
  remaining_area = hexagon_area - rhombus_area →
  remaining_area = 75 := by
sorry

end hexagon_minus_rhombus_area_l2881_288192


namespace base2_10101010_equals_base4_2212_l2881_288144

def base2_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List (Fin 4) :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

theorem base2_10101010_equals_base4_2212 :
  decimal_to_base4 (base2_to_decimal [true, false, true, false, true, false, true, false]) =
  [2, 2, 1, 2] := by sorry

end base2_10101010_equals_base4_2212_l2881_288144


namespace simplify_radical_fraction_l2881_288190

theorem simplify_radical_fraction :
  (3 * Real.sqrt 10) / (Real.sqrt 5 + 2) = 15 * Real.sqrt 2 - 6 * Real.sqrt 10 := by
  sorry

end simplify_radical_fraction_l2881_288190


namespace consecutive_negative_integers_sum_l2881_288179

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2210 → n + (n + 1) = -95 := by sorry

end consecutive_negative_integers_sum_l2881_288179


namespace carolyns_silverware_knives_percentage_l2881_288150

/-- The percentage of knives in Carolyn's silverware after a trade --/
theorem carolyns_silverware_knives_percentage 
  (initial_knives : ℕ) 
  (initial_forks : ℕ) 
  (initial_spoons_multiplier : ℕ) 
  (traded_knives : ℕ) 
  (traded_spoons : ℕ) 
  (h1 : initial_knives = 6)
  (h2 : initial_forks = 12)
  (h3 : initial_spoons_multiplier = 3)
  (h4 : traded_knives = 10)
  (h5 : traded_spoons = 6) :
  let initial_spoons := initial_knives * initial_spoons_multiplier
  let final_knives := initial_knives + traded_knives
  let final_spoons := initial_spoons - traded_spoons
  let total_silverware := final_knives + initial_forks + final_spoons
  (final_knives : ℚ) / (total_silverware : ℚ) = 2/5 := by
  sorry

end carolyns_silverware_knives_percentage_l2881_288150


namespace tangent_slope_point_A_l2881_288135

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 2*x + 3

-- Theorem statement
theorem tangent_slope_point_A :
  ∃ (x y : ℝ), 
    f_derivative x = 7 ∧ 
    f x = y ∧ 
    x = 2 ∧ 
    y = 10 := by sorry

end tangent_slope_point_A_l2881_288135


namespace fraction_sum_equality_l2881_288104

theorem fraction_sum_equality : 
  (1 : ℚ) / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 5 / 6 = -5 / 6 := by
  sorry

end fraction_sum_equality_l2881_288104


namespace min_translation_for_symmetry_l2881_288177

theorem min_translation_for_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x + Real.cos x) :
  ∃ φ : ℝ, φ > 0 ∧
    (∀ x, f (x - φ) = -f (-x + φ)) ∧
    (∀ ψ, ψ > 0 ∧ (∀ x, f (x - ψ) = -f (-x + ψ)) → φ ≤ ψ) ∧
    φ = Real.pi / 4 :=
by sorry

end min_translation_for_symmetry_l2881_288177


namespace repeating_digits_divisible_by_11_l2881_288178

/-- A function that generates a 9-digit number by repeating the first three digits three times -/
def repeatingDigits (a b c : ℕ) : ℕ :=
  100000000 * a + 10000000 * b + 1000000 * c +
  100000 * a + 10000 * b + 1000 * c +
  100 * a + 10 * b + c

/-- Theorem stating that any 9-digit number formed by repeating the first three digits three times is divisible by 11 -/
theorem repeating_digits_divisible_by_11 (a b c : ℕ) (h : 0 < a ∧ a < 10 ∧ b < 10 ∧ c < 10) :
  11 ∣ repeatingDigits a b c := by
  sorry


end repeating_digits_divisible_by_11_l2881_288178


namespace smallest_xy_value_smallest_xy_is_172_min_xy_value_l2881_288158

theorem smallest_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) :
  ∀ (a b : ℕ+), 7 * a + 4 * b = 200 → x * y ≤ a * b :=
by sorry

theorem smallest_xy_is_172 :
  ∃ (x y : ℕ+), 7 * x + 4 * y = 200 ∧ x * y = 172 :=
by sorry

theorem min_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) :
  x * y ≥ 172 :=
by sorry

end smallest_xy_value_smallest_xy_is_172_min_xy_value_l2881_288158


namespace unique_positive_integers_sum_l2881_288154

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 73) / 2 + 5 / 2)

theorem unique_positive_integers_sum (a b c : ℕ+) :
  x^80 = 3*x^78 + 18*x^74 + 15*x^72 - x^40 + (a : ℝ)*x^36 + (b : ℝ)*x^34 + (c : ℝ)*x^30 →
  a + b + c = 265 := by sorry

end unique_positive_integers_sum_l2881_288154


namespace equation_solution_l2881_288143

theorem equation_solution (x y : ℝ) :
  x^5 + y^5 = 33 ∧ x + y = 3 →
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) := by
sorry

end equation_solution_l2881_288143


namespace sibling_ages_theorem_l2881_288124

theorem sibling_ages_theorem :
  ∃ (a b c : ℕ+), 
    a * b * c = 72 ∧ 
    a + b + c = 13 ∧ 
    a > b ∧ b > c :=
by sorry

end sibling_ages_theorem_l2881_288124


namespace arcsin_neg_sqrt3_over_2_l2881_288139

theorem arcsin_neg_sqrt3_over_2 : Real.arcsin (-Real.sqrt 3 / 2) = -π / 3 := by
  sorry

end arcsin_neg_sqrt3_over_2_l2881_288139


namespace solutions_periodic_l2881_288195

/-- A system of differential equations with given initial conditions -/
structure DiffSystem where
  f : ℝ → ℝ  -- y = f(x)
  g : ℝ → ℝ  -- z = g(x)
  eqn1 : ∀ x, deriv f x = -(g x)^3
  eqn2 : ∀ x, deriv g x = (f x)^3
  init1 : f 0 = 1
  init2 : g 0 = 0
  unique : ∀ f' g', (∀ x, deriv f' x = -(g' x)^3) →
                    (∀ x, deriv g' x = (f' x)^3) →
                    f' 0 = 1 → g' 0 = 0 →
                    f' = f ∧ g' = g

/-- Definition of a periodic function -/
def Periodic (f : ℝ → ℝ) :=
  ∃ k : ℝ, k > 0 ∧ ∀ x, f (x + k) = f x

/-- The main theorem stating that solutions are periodic with the same period -/
theorem solutions_periodic (sys : DiffSystem) :
  ∃ k : ℝ, k > 0 ∧ Periodic sys.f ∧ Periodic sys.g ∧
  ∀ x, sys.f (x + k) = sys.f x ∧ sys.g (x + k) = sys.g x :=
sorry

end solutions_periodic_l2881_288195


namespace optimization_scheme_sales_l2881_288175

/-- Given a sequence of three terms forming an arithmetic progression with a sum of 2.46 million,
    prove that the middle term (second term) is equal to 0.82 million. -/
theorem optimization_scheme_sales (a₁ a₂ a₃ : ℝ) : 
  a₁ + a₂ + a₃ = 2.46 ∧ 
  a₂ - a₁ = a₃ - a₂ → 
  a₂ = 0.82 := by
sorry

end optimization_scheme_sales_l2881_288175


namespace beetle_probability_theorem_l2881_288174

/-- Represents the probability of a beetle touching a horizontal edge first -/
def beetle_horizontal_edge_probability (start_x start_y : ℕ) (grid_size : ℕ) : ℝ :=
  sorry

/-- The grid is 10x10 -/
def grid_size : ℕ := 10

/-- The beetle starts at (3, 4) -/
def start_x : ℕ := 3
def start_y : ℕ := 4

/-- Theorem stating the probability of the beetle touching a horizontal edge first -/
theorem beetle_probability_theorem :
  beetle_horizontal_edge_probability start_x start_y grid_size = 0.6 := by
  sorry

end beetle_probability_theorem_l2881_288174


namespace digit_difference_quotient_l2881_288126

/-- Given that 524 in base 7 equals 3cd in base 10, where c and d are single digits,
    prove that (c - d) / 5 = -0.8 -/
theorem digit_difference_quotient (c d : ℕ) : 
  c < 10 → d < 10 → (5 * 7^2 + 2 * 7 + 4 : ℕ) = 300 + 10 * c + d → 
  (c - d : ℚ) / 5 = -4/5 := by sorry

end digit_difference_quotient_l2881_288126


namespace triangle_max_area_l2881_288149

theorem triangle_max_area (a b c : ℝ) (h1 : a + b = 10) (h2 : c = 6) :
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  S ≤ 12 ∧ ∃ a b, a + b = 10 ∧ S = 12 := by
sorry

end triangle_max_area_l2881_288149


namespace prism_volume_l2881_288136

/-- The volume of a prism with an isosceles right triangular base and given dimensions -/
theorem prism_volume (leg : ℝ) (height : ℝ) (h_leg : leg = Real.sqrt 5) (h_height : height = 10) :
  (1 / 2) * leg * leg * height = 25 := by
  sorry

end prism_volume_l2881_288136


namespace greatest_gcd_4Tn_n_minus_1_l2881_288159

/-- The nth triangular number -/
def T (n : ℕ+) : ℕ := (n * (n + 1)) / 2

/-- The statement to be proved -/
theorem greatest_gcd_4Tn_n_minus_1 :
  ∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (4 * T n) (n - 1) ≤ 4 ∧
  Nat.gcd (4 * T k) (k - 1) = 4 :=
sorry

end greatest_gcd_4Tn_n_minus_1_l2881_288159


namespace biquadratic_equation_roots_l2881_288108

theorem biquadratic_equation_roots (x : ℝ) :
  x^4 - 8*x^2 + 4 = 0 ↔ x = Real.sqrt 3 - 1 ∨ x = Real.sqrt 3 + 1 ∨ x = -(Real.sqrt 3 - 1) ∨ x = -(Real.sqrt 3 + 1) :=
sorry

end biquadratic_equation_roots_l2881_288108


namespace binomial_variance_four_half_l2881_288156

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ :=
  ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: The variance of a binomial distribution B(4, 1/2) is 1 -/
theorem binomial_variance_four_half :
  ∀ ξ : BinomialDistribution, ξ.n = 4 ∧ ξ.p = 1/2 → variance ξ = 1 := by
  sorry

end binomial_variance_four_half_l2881_288156


namespace judge_court_cases_judge_court_cases_proof_l2881_288153

theorem judge_court_cases : ℕ → Prop :=
  fun total_cases =>
    let dismissed := 2
    let remaining := total_cases - dismissed
    let innocent := (2 * remaining) / 3
    let delayed := 1
    let guilty := 4
    remaining - innocent - delayed = guilty ∧ total_cases = 17

-- The proof
theorem judge_court_cases_proof : ∃ n : ℕ, judge_court_cases n := by
  sorry

end judge_court_cases_judge_court_cases_proof_l2881_288153


namespace negation_of_proposition_l2881_288196

theorem negation_of_proposition :
  ¬(∀ x y : ℤ, Even (x + y) → (Even x ∧ Even y)) ↔
  (∀ x y : ℤ, ¬Even (x + y) → ¬(Even x ∧ Even y)) :=
by sorry

end negation_of_proposition_l2881_288196


namespace teacher_assignment_theorem_l2881_288110

def number_of_teachers : ℕ := 4
def number_of_classes : ℕ := 3

-- Define a function that calculates the number of ways to assign teachers to classes
def ways_to_assign_teachers (teachers : ℕ) (classes : ℕ) : ℕ :=
  sorry -- The actual calculation goes here

theorem teacher_assignment_theorem :
  ways_to_assign_teachers number_of_teachers number_of_classes = 36 :=
by sorry

end teacher_assignment_theorem_l2881_288110


namespace tangent_and_max_chord_length_l2881_288168

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point M
def point_M (a : ℝ) : ℝ × ℝ := (1, a)

theorem tangent_and_max_chord_length :
  -- Part I: Point M is on the circle if and only if a = ±√3
  (∃ a : ℝ, circle_O (point_M a).1 (point_M a).2 ↔ a = Real.sqrt 3 ∨ a = -Real.sqrt 3) ∧
  -- Part II: Maximum value of |AC| + |BD| is 2√10
  (let a : ℝ := Real.sqrt 2
   ∀ A B C D : ℝ × ℝ,
   circle_O A.1 A.2 →
   circle_O B.1 B.2 →
   circle_O C.1 C.2 →
   circle_O D.1 D.2 →
   (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0 →  -- AC ⊥ BD
   (point_M a).1 = (A.1 + C.1) / 2 →  -- M is midpoint of AC
   (point_M a).1 = (B.1 + D.1) / 2 →  -- M is midpoint of BD
   (point_M a).2 = (A.2 + C.2) / 2 →  -- M is midpoint of AC
   (point_M a).2 = (B.2 + D.2) / 2 →  -- M is midpoint of BD
   Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) ≤ 2 * Real.sqrt 10) := by
sorry

end tangent_and_max_chord_length_l2881_288168


namespace car_price_calculation_l2881_288187

/-- Represents the price of a car given loan terms and payments. -/
def car_price (loan_years : ℕ) (interest_rate : ℚ) (down_payment : ℚ) (monthly_payment : ℚ) : ℚ :=
  down_payment + (loan_years * 12 : ℕ) * monthly_payment

/-- Theorem stating the price of the car under given conditions. -/
theorem car_price_calculation :
  car_price 5 (4/100) 5000 250 = 20000 := by
  sorry

end car_price_calculation_l2881_288187


namespace expression_value_l2881_288102

theorem expression_value (m : ℝ) (h : m^2 + m - 1 = 0) :
  2 / (m^2 + m) - (m + 2) / (m^2 + 2*m + 1) = 1 := by
  sorry

end expression_value_l2881_288102
