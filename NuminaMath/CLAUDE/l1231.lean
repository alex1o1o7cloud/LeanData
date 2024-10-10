import Mathlib

namespace min_sum_a1_a5_l1231_123172

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r > 0, a (n + 1) = r * a n ∧ a n > 0

-- State the theorem
theorem min_sum_a1_a5 (a : ℕ → ℝ) 
  (h_geom : is_positive_geometric_sequence a) 
  (h_prod : a 5 * a 4 * a 2 * a 1 = 16) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 4 → a 1 + a 5 ≥ x + y :=
sorry

end min_sum_a1_a5_l1231_123172


namespace spring_festival_gala_arrangements_l1231_123125

/-- The number of ways to insert 3 distinct objects into 11 spaces,
    such that no two inserted objects are adjacent. -/
def insert_non_adjacent (n m : ℕ) : ℕ :=
  Nat.descFactorial (n + 1) m

theorem spring_festival_gala_arrangements : insert_non_adjacent 10 3 = 990 := by
  sorry

end spring_festival_gala_arrangements_l1231_123125


namespace system_solutions_l1231_123133

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  (x + y - 2018 = (x - 2019) * y) ∧
  (x + z - 2014 = (x - 2019) * z) ∧
  (y + z + 2 = y * z)

-- State the theorem
theorem system_solutions :
  (∃ (x y z : ℝ), system x y z ∧ x = 2022 ∧ y = 2 ∧ z = 4) ∧
  (∃ (x y z : ℝ), system x y z ∧ x = 2017 ∧ y = 0 ∧ z = -2) ∧
  (∀ (x y z : ℝ), system x y z → (x = 2022 ∧ y = 2 ∧ z = 4) ∨ (x = 2017 ∧ y = 0 ∧ z = -2)) :=
by sorry


end system_solutions_l1231_123133


namespace negate_difference_l1231_123105

theorem negate_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end negate_difference_l1231_123105


namespace inequality_proof_l1231_123151

theorem inequality_proof (a b : ℝ) 
  (h : ∀ x : ℝ, Real.cos (a * Real.sin x) > Real.sin (b * Real.cos x)) : 
  a^2 + b^2 < (Real.pi^2) / 4 := by
sorry

end inequality_proof_l1231_123151


namespace value_of_a_l1231_123117

theorem value_of_a (A B : Set ℕ) (a : ℕ) 
  (hA : A = {1, 2})
  (hB : B = {2, a})
  (hUnion : A ∪ B = {1, 2, 4}) :
  a = 4 := by
sorry

end value_of_a_l1231_123117


namespace sarahs_bread_shop_profit_l1231_123107

/-- Sarah's bread shop profit calculation --/
theorem sarahs_bread_shop_profit :
  ∀ (total_loaves : ℕ) 
    (cost_per_loaf morning_price afternoon_price evening_price : ℚ)
    (morning_fraction afternoon_fraction : ℚ),
  total_loaves = 60 →
  cost_per_loaf = 1 →
  morning_price = 3 →
  afternoon_price = 3/2 →
  evening_price = 1 →
  morning_fraction = 1/3 →
  afternoon_fraction = 3/4 →
  let morning_sales := (total_loaves : ℚ) * morning_fraction * morning_price
  let remaining_after_morning := total_loaves - (total_loaves : ℚ) * morning_fraction
  let afternoon_sales := remaining_after_morning * afternoon_fraction * afternoon_price
  let remaining_after_afternoon := remaining_after_morning - remaining_after_morning * afternoon_fraction
  let evening_sales := remaining_after_afternoon * evening_price
  let total_revenue := morning_sales + afternoon_sales + evening_sales
  let total_cost := (total_loaves : ℚ) * cost_per_loaf
  let profit := total_revenue - total_cost
  profit = 55 := by
sorry


end sarahs_bread_shop_profit_l1231_123107


namespace product_sum_ratio_l1231_123162

theorem product_sum_ratio : (1 * 2 * 3 * 4 * 5 * 6) / (1 + 2 + 3 + 4 + 5 + 6) = 240 / 7 := by
  sorry

end product_sum_ratio_l1231_123162


namespace tangent_line_at_one_monotonicity_non_positive_monotonicity_positive_l1231_123161

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - Real.log x - 2

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * x - 1/x

theorem tangent_line_at_one (x : ℝ) :
  f 1 1 = -(3/2) ∧ f_deriv 1 1 = 0 :=
sorry

theorem monotonicity_non_positive (a : ℝ) (x : ℝ) (ha : a ≤ 0) (hx : x > 0) :
  f_deriv a x < 0 :=
sorry

theorem monotonicity_positive (a : ℝ) (x : ℝ) (ha : a > 0) (hx : x > 0) :
  (x < Real.sqrt a / a → f_deriv a x < 0) ∧
  (x > Real.sqrt a / a → f_deriv a x > 0) :=
sorry

end

end tangent_line_at_one_monotonicity_non_positive_monotonicity_positive_l1231_123161


namespace fraction_equality_l1231_123174

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (5 * a + 2 * b) / (2 * a - 5 * b) = 3) : 
  (2 * a + 5 * b) / (5 * a - 2 * b) = 39 / 83 := by
  sorry

end fraction_equality_l1231_123174


namespace dividend_calculation_l1231_123100

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 18 → quotient = 9 → remainder = 5 → 
  divisor * quotient + remainder = 167 := by
sorry

end dividend_calculation_l1231_123100


namespace polynomial_division_remainder_l1231_123163

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^4 + 16 * X^3 + 5 * X^2 - 36 * X + 58 = 
  (X^2 + 5 * X + 3) * q + (-28 * X + 55) := by
  sorry

end polynomial_division_remainder_l1231_123163


namespace abs_of_nonnegative_l1231_123180

theorem abs_of_nonnegative (x : ℝ) : x ≥ 0 → |x| = x := by
  sorry

end abs_of_nonnegative_l1231_123180


namespace modulus_of_complex_expression_l1231_123148

theorem modulus_of_complex_expression : 
  Complex.abs ((1 + Complex.I) / (1 - Complex.I) + Complex.I) = 2 := by sorry

end modulus_of_complex_expression_l1231_123148


namespace even_perfect_square_factors_l1231_123141

/-- The number of even perfect square factors of 2^6 * 7^10 * 3^2 -/
theorem even_perfect_square_factors : 
  (Finset.filter (fun a => a % 2 = 0 ∧ 2 ≤ a) (Finset.range 7)).card *
  (Finset.filter (fun b => b % 2 = 0) (Finset.range 11)).card *
  (Finset.filter (fun c => c % 2 = 0) (Finset.range 3)).card = 36 := by
  sorry

end even_perfect_square_factors_l1231_123141


namespace smallest_equivalent_angle_l1231_123175

theorem smallest_equivalent_angle (x : ℝ) (h : x = -11/4 * Real.pi) :
  ∃ (θ : ℝ) (k : ℤ),
    x = θ + 2 * ↑k * Real.pi ∧
    θ ∈ Set.Icc (-Real.pi) Real.pi ∧
    ∀ (φ : ℝ) (m : ℤ),
      x = φ + 2 * ↑m * Real.pi →
      φ ∈ Set.Icc (-Real.pi) Real.pi →
      |θ| ≤ |φ| ∧
    θ = -3/4 * Real.pi :=
sorry

end smallest_equivalent_angle_l1231_123175


namespace probability_at_least_two_correct_l1231_123144

theorem probability_at_least_two_correct (n : ℕ) (p : ℚ) : 
  n = 6 → p = 1/6 → 
  1 - (Nat.choose n 0 * p^0 * (1-p)^n + Nat.choose n 1 * p^1 * (1-p)^(n-1)) = 34369/58420 := by
  sorry

end probability_at_least_two_correct_l1231_123144


namespace min_abs_z_with_constraint_l1231_123111

theorem min_abs_z_with_constraint (z : ℂ) (h : Complex.abs (z - 2*Complex.I) + Complex.abs (z - 5) = 7) :
  Complex.abs z ≥ 20 * Real.sqrt 29 / 29 := by
  sorry

end min_abs_z_with_constraint_l1231_123111


namespace tilde_result_bounds_l1231_123193

def tilde (a b : ℚ) : ℚ := |a - b|

def consecutive_integers (n : ℕ) : List ℚ := List.range n

def perform_tilde (l : List ℚ) : ℚ :=
  l.foldl tilde (l.head!)

def max_tilde_result (n : ℕ) : ℚ :=
  if n % 4 == 1 then n - 1 else n

def min_tilde_result (n : ℕ) : ℚ :=
  if n % 4 == 2 || n % 4 == 3 then 1 else 0

theorem tilde_result_bounds (n : ℕ) (l : List ℚ) :
  l.length = n ∧ l.toFinset = (consecutive_integers n).toFinset →
  perform_tilde l ≤ max_tilde_result n ∧
  perform_tilde l ≥ min_tilde_result n :=
sorry

end tilde_result_bounds_l1231_123193


namespace income_ratio_l1231_123195

/-- Proves that the ratio of A's monthly income to B's monthly income is 2.5:1 -/
theorem income_ratio (c_monthly : ℕ) (a_annual : ℕ) : 
  c_monthly = 15000 →
  a_annual = 504000 →
  (a_annual / 12 : ℚ) / ((1 + 12/100) * c_monthly) = 5/2 := by
  sorry

end income_ratio_l1231_123195


namespace apple_pyramid_sum_l1231_123146

/-- Calculates the number of apples in a single layer of the pyramid --/
def layer_apples (base_width : ℕ) (base_length : ℕ) (layer : ℕ) : ℕ :=
  (base_width - layer + 1) * (base_length - layer + 1)

/-- Calculates the total number of apples in the pyramid --/
def total_apples (base_width : ℕ) (base_length : ℕ) : ℕ :=
  (List.range (min base_width base_length)).foldl (λ sum layer => sum + layer_apples base_width base_length layer) 0

theorem apple_pyramid_sum :
  total_apples 6 9 = 154 :=
by sorry

end apple_pyramid_sum_l1231_123146


namespace inequality_proof_l1231_123136

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end inequality_proof_l1231_123136


namespace fixed_point_theorem_l1231_123157

universe u

theorem fixed_point_theorem (S : Type u) [Nonempty S] (f : Set S → Set S) 
  (h : ∀ (X Y : Set S), X ⊆ Y → f X ⊆ f Y) :
  ∃ (A : Set S), f A = A := by
sorry

end fixed_point_theorem_l1231_123157


namespace equation_solution_l1231_123143

theorem equation_solution : 
  ∃ x : ℝ, ((0.02^2 + 0.52^2 + 0.035^2) / (0.002^2 + 0.052^2 + x^2) = 100) ∧ x = 0.0035 := by
  sorry

end equation_solution_l1231_123143


namespace apple_basket_theorem_l1231_123178

/-- Represents the capacity of an apple basket -/
structure Basket where
  capacity : ℕ

/-- Represents the current state of Jack's basket -/
structure JackBasket extends Basket where
  current : ℕ
  space_left : ℕ

/-- Theorem about apple baskets -/
theorem apple_basket_theorem (jack : JackBasket) (jill : Basket) : 
  jack.capacity = 12 →
  jack.space_left = 4 →
  jill.capacity = 2 * jack.capacity →
  (jill.capacity / (jack.capacity - jack.space_left) : ℕ) = 3 := by
  sorry

end apple_basket_theorem_l1231_123178


namespace solution_set_theorem_l1231_123170

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem solution_set_theorem (f : ℝ → ℝ) 
  (h_increasing : increasing_function f) 
  (h_f_0 : f 0 = -1) 
  (h_f_3 : f 3 = 1) :
  {x : ℝ | |f x| < 1} = Set.Ioo 0 3 := by
  sorry

end solution_set_theorem_l1231_123170


namespace intersection_equality_l1231_123194

def A : Set ℝ := {x | |x - 4| < 2 * x}
def B (a : ℝ) : Set ℝ := {x | x * (x - a) ≥ (a + 6) * (x - a)}

theorem intersection_equality (a : ℝ) : A ∩ B a = A ↔ a ≤ -14/3 := by
  sorry

end intersection_equality_l1231_123194


namespace probability_all_red_at_fourth_l1231_123101

/-- The number of white balls initially in the bag -/
def initial_white_balls : ℕ := 8

/-- The number of red balls initially in the bag -/
def initial_red_balls : ℕ := 2

/-- The total number of balls initially in the bag -/
def total_balls : ℕ := initial_white_balls + initial_red_balls

/-- The probability of drawing a specific sequence of balls -/
def sequence_probability (red_indices : List ℕ) : ℚ :=
  sorry

/-- The probability of drawing all red balls exactly at the 4th draw -/
def all_red_at_fourth_draw : ℚ :=
  sequence_probability [1, 4] + sequence_probability [2, 4] + sequence_probability [3, 4]

theorem probability_all_red_at_fourth : all_red_at_fourth_draw = 434/10000 := by
  sorry

end probability_all_red_at_fourth_l1231_123101


namespace quadratic_inequality_domain_l1231_123156

theorem quadratic_inequality_domain (a : ℝ) :
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) ↔ a ∈ Set.Ioo 1 5 ∪ Set.Ioc 5 5 :=
sorry

end quadratic_inequality_domain_l1231_123156


namespace parallel_lines_theorem_l1231_123150

/-- A line in a 3D space --/
structure Line3D where
  -- We don't need to define the internal structure of a line
  -- for this problem, so we leave it empty

/-- Two lines are parallel --/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines form equal angles with a third line --/
def equal_angles (l1 l2 l3 : Line3D) : Prop :=
  sorry

/-- A line is perpendicular to another line --/
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- Main theorem: Exactly two of the given propositions about parallel lines are false --/
theorem parallel_lines_theorem :
  ∃ (prop1 prop2 prop3 : Prop),
    prop1 = (∀ l1 l2 l3 : Line3D, equal_angles l1 l2 l3 → parallel l1 l2) ∧
    prop2 = (∀ l1 l2 l3 : Line3D, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2) ∧
    prop3 = (∀ l1 l2 l3 : Line3D, parallel l1 l3 → parallel l2 l3 → parallel l1 l2) ∧
    (¬prop1 ∧ ¬prop2 ∧ prop3) :=
  sorry

end parallel_lines_theorem_l1231_123150


namespace gaeun_taller_than_nana_l1231_123196

/-- Proves that Gaeun is taller than Nana by 0.5 centimeters -/
theorem gaeun_taller_than_nana :
  let nana_height_m : ℝ := 1.618
  let gaeun_height_cm : ℝ := 162.3
  let m_to_cm : ℝ := 100
  gaeun_height_cm - (nana_height_m * m_to_cm) = 0.5 := by sorry

end gaeun_taller_than_nana_l1231_123196


namespace right_triangle_hypotenuse_l1231_123123

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 12 → b = 5 → c^2 = a^2 + b^2 → c = 13 :=
by
  sorry

end right_triangle_hypotenuse_l1231_123123


namespace exists_special_multiple_l1231_123131

/-- A function that returns true if all digits of a natural number are in the set {0, 1, 8, 9} -/
def valid_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ ({0, 1, 8, 9} : Set ℕ)

/-- The main theorem stating the existence of a number with the required properties -/
theorem exists_special_multiple : ∃ n : ℕ, 
  2003 ∣ n ∧ n < 10^11 ∧ valid_digits n :=
sorry

end exists_special_multiple_l1231_123131


namespace sophia_reading_progress_l1231_123122

theorem sophia_reading_progress (total_pages : ℕ) (pages_read : ℕ) : 
  total_pages = 270 →
  pages_read = (total_pages - pages_read) + 90 →
  pages_read / total_pages = 2 / 3 := by
sorry

end sophia_reading_progress_l1231_123122


namespace inequality_holds_l1231_123181

/-- An equilateral triangle with height 1 -/
structure EquilateralTriangle :=
  (height : ℝ)
  (height_eq_one : height = 1)

/-- A point inside the equilateral triangle -/
structure PointInTriangle (t : EquilateralTriangle) :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (sum_eq_height : x + y + z = t.height)
  (all_positive : x > 0 ∧ y > 0 ∧ z > 0)

/-- The inequality holds for any point inside the equilateral triangle -/
theorem inequality_holds (t : EquilateralTriangle) (p : PointInTriangle t) :
  p.x^2 + p.y^2 + p.z^2 ≥ p.x^3 + p.y^3 + p.z^3 + 6*p.x*p.y*p.z :=
sorry

end inequality_holds_l1231_123181


namespace remainder_371073_div_6_l1231_123106

theorem remainder_371073_div_6 : 371073 % 6 = 3 := by
  sorry

end remainder_371073_div_6_l1231_123106


namespace birch_count_l1231_123177

/-- Represents the number of trees of each species in the forest --/
structure ForestComposition where
  oak : ℕ
  pine : ℕ
  spruce : ℕ
  birch : ℕ

/-- Theorem stating the number of birch trees in the forest --/
theorem birch_count (forest : ForestComposition) : forest.birch = 2160 :=
  by
  have total_trees : forest.oak + forest.pine + forest.spruce + forest.birch = 4000 := by sorry
  have spruce_percentage : forest.spruce = 4000 * 10 / 100 := by sorry
  have pine_percentage : forest.pine = 4000 * 13 / 100 := by sorry
  have oak_count : forest.oak = forest.spruce + forest.pine := by sorry
  sorry


end birch_count_l1231_123177


namespace T_coprime_and_sum_reciprocals_l1231_123191

def T : ℕ → ℕ
  | 0 => 2
  | n + 1 => T n^2 - T n + 1

theorem T_coprime_and_sum_reciprocals :
  (∀ m n, m ≠ n → Nat.gcd (T m) (T n) = 1) ∧
  (∑' i, (T i)⁻¹ : ℝ) = 1 := by
  sorry

end T_coprime_and_sum_reciprocals_l1231_123191


namespace min_sum_given_product_l1231_123187

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 3 → (∀ x y : ℝ, x > 0 → y > 0 → x * y = x + y + 3 → a + b ≤ x + y) → a + b = 6 :=
by sorry

end min_sum_given_product_l1231_123187


namespace stratified_sampling_sum_l1231_123108

theorem stratified_sampling_sum (total_population : ℕ) (sample_size : ℕ) 
  (type_a_count : ℕ) (type_b_count : ℕ) :
  total_population = 100 →
  sample_size = 20 →
  type_a_count = 10 →
  type_b_count = 20 →
  (type_a_count * sample_size / total_population + 
   type_b_count * sample_size / total_population : ℚ) = 6 := by
  sorry

#check stratified_sampling_sum

end stratified_sampling_sum_l1231_123108


namespace angle_D_is_60_l1231_123159

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Real)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  q.A + q.B + q.C + q.D = 360

-- Define the specific conditions of our quadrilateral
def special_quadrilateral (q : Quadrilateral) : Prop :=
  q.A + q.B = 180 ∧ q.C = 2 * q.D

-- Theorem statement
theorem angle_D_is_60 (q : Quadrilateral) 
  (h1 : is_valid_quadrilateral q) 
  (h2 : special_quadrilateral q) : 
  q.D = 60 := by
  sorry

end angle_D_is_60_l1231_123159


namespace jeremy_age_l1231_123199

/-- Represents the ages of three people -/
structure Ages where
  jeremy : ℕ
  sebastian : ℕ
  sophia : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.jeremy + 3) + (ages.sebastian + 3) + (ages.sophia + 3) = 150 ∧
  ages.sebastian = ages.jeremy + 4 ∧
  ages.sophia + 3 = 60

/-- The theorem stating Jeremy's current age -/
theorem jeremy_age (ages : Ages) (h : problem_conditions ages) : ages.jeremy = 40 := by
  sorry

end jeremy_age_l1231_123199


namespace remove_one_for_avg_eight_point_five_l1231_123167

theorem remove_one_for_avg_eight_point_five (n : Nat) (h : n = 15) :
  let list := List.range n
  let sum := n * (n + 1) / 2
  let removed := 1
  let remaining_sum := sum - removed
  let remaining_count := n - 1
  (remaining_sum : ℚ) / remaining_count = 17/2 := by
  sorry

end remove_one_for_avg_eight_point_five_l1231_123167


namespace ceiling_squared_fraction_l1231_123149

theorem ceiling_squared_fraction : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end ceiling_squared_fraction_l1231_123149


namespace exactly_three_solutions_l1231_123119

/-- The function that we're interested in -/
def f (m : ℕ+) : ℚ :=
  1260 / ((m : ℚ)^2 - 6)

/-- Predicate for f(m) being a positive integer -/
def is_positive_integer (m : ℕ+) : Prop :=
  ∃ (k : ℕ+), f m = k

/-- The main theorem -/
theorem exactly_three_solutions :
  ∃! (s : Finset ℕ+), s.card = 3 ∧ ∀ m : ℕ+, m ∈ s ↔ is_positive_integer m :=
sorry

end exactly_three_solutions_l1231_123119


namespace intersection_distance_l1231_123102

-- Define the circle centers and radius
structure CircleConfig where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  r : ℝ

-- Define the conditions
def validConfig (config : CircleConfig) : Prop :=
  1 < config.r ∧ config.r < 2 ∧
  dist config.A config.B = 2 ∧
  dist config.B config.C = 2 ∧
  dist config.A config.C = 2

-- Define the intersection points
def B' (config : CircleConfig) : ℝ × ℝ := sorry
def C' (config : CircleConfig) : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_distance (config : CircleConfig) 
  (h : validConfig config) :
  dist (B' config) (C' config) = 1 + Real.sqrt (3 * (config.r^2 - 1)) :=
sorry

end intersection_distance_l1231_123102


namespace cubic_function_property_l1231_123116

/-- Given a cubic function f(x) with certain properties, prove that f(1) has specific values -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => (1/3) * x^3 + a^2 * x^2 + a * x + b
  (f (-1) = -7/12 ∧ (λ x => x^2 + 2*a^2*x + a) (-1) = 0) → 
  (f 1 = 25/12 ∨ f 1 = 1/12) := by
sorry


end cubic_function_property_l1231_123116


namespace summit_conference_attendance_l1231_123173

/-- The number of diplomats who attended the summit conference -/
def D : ℕ := 120

/-- The number of diplomats who spoke French -/
def french_speakers : ℕ := 20

/-- The number of diplomats who did not speak Hindi -/
def non_hindi_speakers : ℕ := 32

/-- The proportion of diplomats who spoke neither French nor Hindi -/
def neither_french_nor_hindi : ℚ := 1/5

/-- The proportion of diplomats who spoke both French and Hindi -/
def both_french_and_hindi : ℚ := 1/10

theorem summit_conference_attendance :
  D = 120 ∧
  french_speakers = 20 ∧
  non_hindi_speakers = 32 ∧
  neither_french_nor_hindi = 1/5 ∧
  both_french_and_hindi = 1/10 ∧
  (D : ℚ) * neither_french_nor_hindi + (D : ℚ) * both_french_and_hindi + french_speakers = D :=
sorry

end summit_conference_attendance_l1231_123173


namespace square_divided_into_triangles_even_count_l1231_123154

theorem square_divided_into_triangles_even_count (a : ℕ) (h : a > 0) :
  let triangle_area : ℚ := 3 * 4 / 2
  let square_area : ℚ := a^2
  let num_triangles : ℚ := square_area / triangle_area
  (∃ k : ℕ, num_triangles = k ∧ k % 2 = 0) :=
sorry

end square_divided_into_triangles_even_count_l1231_123154


namespace expression_equals_zero_l1231_123183

theorem expression_equals_zero :
  (-1)^2023 - |1 - Real.sqrt 3| + Real.sqrt 6 * Real.sqrt (1/2) = 0 := by
  sorry

end expression_equals_zero_l1231_123183


namespace probability_of_quarter_l1231_123109

def quarter_value : ℚ := 25 / 100
def nickel_value : ℚ := 5 / 100
def penny_value : ℚ := 1 / 100

def total_quarter_value : ℚ := 10
def total_nickel_value : ℚ := 5
def total_penny_value : ℚ := 15

def num_quarters : ℕ := (total_quarter_value / quarter_value).num.toNat
def num_nickels : ℕ := (total_nickel_value / nickel_value).num.toNat
def num_pennies : ℕ := (total_penny_value / penny_value).num.toNat

def total_coins : ℕ := num_quarters + num_nickels + num_pennies

theorem probability_of_quarter : 
  (num_quarters : ℚ) / total_coins = 1 / 41 := by sorry

end probability_of_quarter_l1231_123109


namespace both_tea_probability_l1231_123120

-- Define the setup
def total_people : ℕ := 6
def tables : ℕ := 3
def people_per_table : ℕ := 2
def coffee_drinkers : ℕ := 3
def tea_drinkers : ℕ := 3

-- Define the probability function
def probability_both_tea : ℚ := 0.6

-- Theorem statement
theorem both_tea_probability :
  probability_both_tea = 0.6 :=
sorry

end both_tea_probability_l1231_123120


namespace unique_digit_property_l1231_123121

theorem unique_digit_property : ∃! x : ℕ, x < 10 ∧ ∀ a : ℕ, 10 * a + x = a + x + a * x := by
  sorry

end unique_digit_property_l1231_123121


namespace black_marble_probability_l1231_123137

/-- The probability of drawing a black marble from a bag -/
theorem black_marble_probability 
  (yellow : ℕ) 
  (blue : ℕ) 
  (green : ℕ) 
  (black : ℕ) 
  (h_yellow : yellow = 12) 
  (h_blue : blue = 10) 
  (h_green : green = 5) 
  (h_black : black = 1) : 
  (black : ℚ) / (yellow + blue + green + black : ℚ) = 1 / 28 := by
  sorry

end black_marble_probability_l1231_123137


namespace jewelry_store_profit_l1231_123164

/-- Calculates the gross profit for a pair of earrings -/
def earrings_gross_profit (purchase_price : ℚ) (markup_percentage : ℚ) (price_decrease_percentage : ℚ) : ℚ :=
  let initial_selling_price := purchase_price / (1 - markup_percentage)
  let price_decrease := initial_selling_price * price_decrease_percentage
  let final_selling_price := initial_selling_price - price_decrease
  final_selling_price - purchase_price

/-- Theorem stating the gross profit for the given scenario -/
theorem jewelry_store_profit :
  earrings_gross_profit 240 (25/100) (20/100) = 16 := by
  sorry

end jewelry_store_profit_l1231_123164


namespace perpendicular_planes_parallel_l1231_123142

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_parallel 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) 
  (h2 : perpendicular m β) : 
  parallel α β := by sorry

end perpendicular_planes_parallel_l1231_123142


namespace intersection_M_N_l1231_123103

-- Define set M
def M : Set ℝ := {x | x * (x - 3) < 0}

-- Define set N
def N : Set ℝ := {x | |x| < 2}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by
  sorry

end intersection_M_N_l1231_123103


namespace sum_fusion_2020_l1231_123185

/-- A number is a sum fusion number if it's equal to the square difference of two consecutive even numbers. -/
def IsSumFusionNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 2)^2 - (2*k)^2

/-- 2020 is a sum fusion number. -/
theorem sum_fusion_2020 : IsSumFusionNumber 2020 := by
  sorry

#check sum_fusion_2020

end sum_fusion_2020_l1231_123185


namespace enclosing_polygon_sides_l1231_123198

/-- Represents a regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the enclosing arrangement -/
structure EnclosingArrangement :=
  (central : RegularPolygon)
  (enclosing : RegularPolygon)
  (num_enclosing : ℕ)

/-- Checks if the arrangement is symmetrical and without gaps or overlaps -/
def is_valid_arrangement (arr : EnclosingArrangement) : Prop :=
  arr.central.sides = arr.num_enclosing ∧
  arr.num_enclosing * (180 / arr.enclosing.sides) = arr.central.sides * (180 - (arr.central.sides - 2) * 180 / arr.central.sides) / 2

theorem enclosing_polygon_sides
  (arr : EnclosingArrangement)
  (h_valid : is_valid_arrangement arr)
  (h_central_sides : arr.central.sides = 15) :
  arr.enclosing.sides = 15 :=
sorry

end enclosing_polygon_sides_l1231_123198


namespace min_difference_for_always_larger_l1231_123104

/-- Pratyya's daily number transformation -/
def pratyya_transform (n : ℤ) : ℤ := 2 * n - 2

/-- Payel's daily number transformation -/
def payel_transform (m : ℤ) : ℤ := 2 * m + 2

/-- The difference between Pratyya's and Payel's numbers after t days -/
def difference (n m : ℤ) (t : ℕ) : ℤ :=
  pratyya_transform (n + t) - payel_transform (m + t)

/-- The theorem stating the minimum difference for Pratyya's number to always be larger -/
theorem min_difference_for_always_larger (n m : ℤ) (h : n > m) :
  (∀ t : ℕ, difference n m t > 0) ↔ n - m ≥ 4 := by sorry

end min_difference_for_always_larger_l1231_123104


namespace triangle_ratio_l1231_123168

/-- Given an acute triangle ABC with a point D inside it, 
    if ∠ADB = ∠ACB + 90° and AC * BD = AD * BC, 
    then (AB * CD) / (AC * BD) = √2 -/
theorem triangle_ratio (A B C D : ℂ) : 
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧  -- A, B, C form a triangle
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ D = t*B + (1-t)*C) ∧  -- D is inside triangle ABC
  Complex.arg ((D - B) / (D - A)) = Complex.arg ((C - B) / (C - A)) + Real.pi / 2 ∧  -- ∠ADB = ∠ACB + 90°
  Complex.abs (C - A) * Complex.abs (D - B) = Complex.abs (D - A) * Complex.abs (C - B) →  -- AC * BD = AD * BC
  Complex.abs ((B - A) * (D - C)) / (Complex.abs (C - A) * Complex.abs (D - B)) = Real.sqrt 2 := by
sorry

end triangle_ratio_l1231_123168


namespace largest_quantity_l1231_123147

theorem largest_quantity (A B C : ℚ) : 
  A = 2006/2005 + 2006/2007 →
  B = 2006/2007 + 2008/2007 →
  C = 2007/2006 + 2007/2008 →
  A > B ∧ A > C := by
  sorry

end largest_quantity_l1231_123147


namespace cricket_team_throwers_l1231_123155

/-- Represents a cricket team with throwers and non-throwers -/
structure CricketTeam where
  total_players : ℕ
  throwers : ℕ
  right_handed : ℕ
  left_handed : ℕ

/-- Conditions for the cricket team problem -/
def valid_cricket_team (team : CricketTeam) : Prop :=
  team.total_players = 58 ∧
  team.throwers + team.right_handed + team.left_handed = team.total_players ∧
  team.throwers + team.right_handed = 51 ∧
  team.left_handed = (team.total_players - team.throwers) / 3

theorem cricket_team_throwers :
  ∀ team : CricketTeam, valid_cricket_team team → team.throwers = 37 := by
  sorry

end cricket_team_throwers_l1231_123155


namespace length_AB_given_P_Q_positions_AB_length_is_189_l1231_123176

/-- Represents a point on a line segment -/
structure PointOnSegment (A B : ℝ) where
  x : ℝ
  h1 : A ≤ x
  h2 : x ≤ B

/-- Theorem: Length of AB given P and Q positions -/
theorem length_AB_given_P_Q_positions
  (A B : ℝ)
  (P : PointOnSegment A B)
  (Q : PointOnSegment A B)
  (h_same_side : (P.x - (A + B) / 2) * (Q.x - (A + B) / 2) > 0)
  (h_P_ratio : P.x - A = 3 / 7 * (B - A))
  (h_Q_ratio : Q.x - A = 4 / 9 * (B - A))
  (h_PQ_distance : |Q.x - P.x| = 3)
  : B - A = 189 := by
  sorry

/-- Corollary: AB length is 189 -/
theorem AB_length_is_189 : ∃ A B : ℝ, B - A = 189 ∧ 
  ∃ (P Q : PointOnSegment A B), 
    (P.x - (A + B) / 2) * (Q.x - (A + B) / 2) > 0 ∧
    P.x - A = 3 / 7 * (B - A) ∧
    Q.x - A = 4 / 9 * (B - A) ∧
    |Q.x - P.x| = 3 := by
  sorry

end length_AB_given_P_Q_positions_AB_length_is_189_l1231_123176


namespace woman_birth_year_l1231_123152

/-- A woman born in the first half of the twentieth century was x years old in the year x^2. This theorem proves her birth year was 1892. -/
theorem woman_birth_year :
  ∃ (x : ℕ),
    (x^2 : ℕ) < 2000 ∧  -- Born in the first half of the 20th century
    (x^2 : ℕ) ≥ 1900 ∧  -- Born in the 20th century
    (x^2 - x : ℕ) = 1892  -- Birth year calculation
  := by sorry

#check woman_birth_year

end woman_birth_year_l1231_123152


namespace equations_represent_parabola_and_ellipse_l1231_123186

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equations mx + ny² = 0 and mx² + ny² = 1 -/
def Equations (m n : ℝ) : Prop :=
  m ≠ 0 ∧ n ≠ 0 ∧
  ∃ (p : Point), m * p.x + n * p.y^2 = 0 ∧ m * p.x^2 + n * p.y^2 = 1

/-- Represents a parabola opening to the right -/
def ParabolaOpeningRight (m n : ℝ) : Prop :=
  m < 0 ∧ n > 0 ∧
  ∀ (p : Point), m * p.x + n * p.y^2 = 0 → p.x = -n / m * p.y^2

/-- Represents an ellipse centered at the origin -/
def Ellipse (m n : ℝ) : Prop :=
  m ≠ 0 ∧ n ≠ 0 ∧
  ∀ (p : Point), m * p.x^2 + n * p.y^2 = 1

/-- Theorem stating that the equations represent a parabola opening right and an ellipse -/
theorem equations_represent_parabola_and_ellipse (m n : ℝ) :
  Equations m n → ParabolaOpeningRight m n ∧ Ellipse m n :=
by sorry

end equations_represent_parabola_and_ellipse_l1231_123186


namespace quadratic_discriminant_l1231_123192

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 2x² + (2 + 1/2)x + 1/2 -/
def a : ℚ := 2
def b : ℚ := 5/2
def c : ℚ := 1/2

theorem quadratic_discriminant :
  discriminant a b c = 9/4 := by
  sorry

end quadratic_discriminant_l1231_123192


namespace inverse_negation_implies_contrapositive_l1231_123160

-- Define propositions as boolean variables
variable (p q r : Prop)

-- Define the inverse relation
def is_inverse (a b : Prop) : Prop :=
  (a ↔ b) ∧ (¬a ↔ ¬b)

-- Define the negation relation
def is_negation (a b : Prop) : Prop :=
  a ↔ ¬b

-- Define the contrapositive relation
def is_contrapositive (a b : Prop) : Prop :=
  (a ↔ ¬b) ∧ (b ↔ ¬a)

-- State the theorem
theorem inverse_negation_implies_contrapositive
  (h1 : is_inverse p q)
  (h2 : is_negation q r) :
  is_contrapositive p r := by
sorry

end inverse_negation_implies_contrapositive_l1231_123160


namespace first_run_rate_l1231_123145

theorem first_run_rate (first_run_distance : ℝ) (second_run_distance : ℝ) 
  (second_run_rate : ℝ) (total_time : ℝ) :
  first_run_distance = 5 →
  second_run_distance = 4 →
  second_run_rate = 9.5 →
  total_time = 88 →
  first_run_distance * (total_time - second_run_distance * second_run_rate) / first_run_distance = 10 :=
by sorry

end first_run_rate_l1231_123145


namespace all_roots_nonzero_l1231_123140

theorem all_roots_nonzero :
  (∀ x : ℝ, 4 * x^2 - 6 = 34 → x ≠ 0) ∧
  (∀ x : ℝ, (3 * x - 1)^2 = (x + 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, (x^2 - 4 : ℝ) = (x + 3 : ℝ) → x ≠ 0) :=
by sorry

end all_roots_nonzero_l1231_123140


namespace four_letter_initials_count_l1231_123132

theorem four_letter_initials_count : 
  (Finset.range 10).card ^ 4 = 10000 := by sorry

end four_letter_initials_count_l1231_123132


namespace minimize_sum_distances_on_x_axis_l1231_123153

/-- The point that minimizes the sum of distances to two given points -/
def minimize_sum_distances (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem minimize_sum_distances_on_x_axis 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (h_A : A = (-1, 2)) 
  (h_B : B = (2, 1)) :
  minimize_sum_distances A B = (1, 0) :=
sorry

end minimize_sum_distances_on_x_axis_l1231_123153


namespace f_max_min_difference_l1231_123126

noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.sin x + Real.cos x) - (1/2) * Real.sin (2 * x)

theorem f_max_min_difference :
  (⨆ (x : ℝ), f x) - (⨅ (x : ℝ), f x) = Real.exp (Real.sqrt 2) - Real.exp (-Real.sqrt 2) :=
by sorry

end f_max_min_difference_l1231_123126


namespace miranda_goose_feathers_l1231_123128

/-- The number of feathers needed for one pillow -/
def feathers_per_pillow : ℕ := 2 * 300

/-- The number of pillows Miranda can stuff -/
def pillows_stuffed : ℕ := 6

/-- The number of feathers on Miranda's goose -/
def goose_feathers : ℕ := feathers_per_pillow * pillows_stuffed

theorem miranda_goose_feathers : goose_feathers = 3600 := by
  sorry

end miranda_goose_feathers_l1231_123128


namespace welders_who_left_l1231_123118

/-- Represents the problem of welders working on an order -/
structure WelderProblem where
  initial_welders : ℕ
  initial_days : ℕ
  remaining_days : ℕ
  welders_left : ℕ

/-- The specific problem instance -/
def problem : WelderProblem :=
  { initial_welders := 12
  , initial_days := 8
  , remaining_days := 28
  , welders_left := 3 }

/-- Theorem stating the number of welders who left for another project -/
theorem welders_who_left (p : WelderProblem) : 
  p.initial_welders - p.welders_left = 9 :=
by sorry

#check welders_who_left problem

end welders_who_left_l1231_123118


namespace product_41_reciprocal_squares_sum_l1231_123188

theorem product_41_reciprocal_squares_sum :
  ∀ a b : ℕ+,
  (a.val : ℕ) * (b.val : ℕ) = 41 →
  (1 : ℚ) / (a.val^2 : ℚ) + (1 : ℚ) / (b.val^2 : ℚ) = 1682 / 1681 :=
by sorry

end product_41_reciprocal_squares_sum_l1231_123188


namespace smallest_gcd_l1231_123135

theorem smallest_gcd (p q r : ℕ+) (h1 : Nat.gcd p q = 294) (h2 : Nat.gcd p r = 847) :
  ∃ (q' r' : ℕ+), Nat.gcd q' r' = 49 ∧ 
    ∀ (q'' r'' : ℕ+), Nat.gcd p q'' = 294 → Nat.gcd p r'' = 847 → 
      Nat.gcd q'' r'' ≥ 49 :=
by sorry

end smallest_gcd_l1231_123135


namespace polar_equivalence_l1231_123127

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Checks if two polar points are equivalent -/
def polar_equivalent (p1 p2 : PolarPoint) : Prop :=
  p1.r * (Real.cos p1.θ) = p2.r * (Real.cos p2.θ) ∧
  p1.r * (Real.sin p1.θ) = p2.r * (Real.sin p2.θ)

theorem polar_equivalence :
  let p1 : PolarPoint := ⟨6, 4*Real.pi/3⟩
  let p2 : PolarPoint := ⟨-6, Real.pi/3⟩
  polar_equivalent p1 p2 := by
  sorry

end polar_equivalence_l1231_123127


namespace unique_m_value_l1231_123130

-- Define the set A
def A (m : ℚ) : Set ℚ := {m + 2, 2 * m^2 + m}

-- Theorem statement
theorem unique_m_value : ∃! m : ℚ, 3 ∈ A m ∧ (∀ x ∈ A m, x = m + 2 ∨ x = 2 * m^2 + m) :=
by sorry

end unique_m_value_l1231_123130


namespace coordinates_wrt_origin_l1231_123184

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- The coordinate system -/
structure CoordinateSystem where
  origin : Point

/-- A point's coordinates with respect to a coordinate system -/
def coordinates (p : Point) (cs : CoordinateSystem) : Point :=
  ⟨p.x - cs.origin.x, p.y - cs.origin.y⟩

theorem coordinates_wrt_origin (A : Point) (cs : CoordinateSystem) :
  A.x = -1 ∧ A.y = 2 → coordinates A cs = A :=
by sorry

end coordinates_wrt_origin_l1231_123184


namespace circumcircle_equation_l1231_123165

/-- Given a triangle ABC with vertices A(0,4), B(0,0), and C(3,0),
    prove that (x-3/2)^2 + (y-2)^2 = 25/4 is the equation of its circumcircle. -/
theorem circumcircle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (0, 4)
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (3, 0)
  (x - 3/2)^2 + (y - 2)^2 = 25/4 ↔ 
    ∃ (center : ℝ × ℝ) (radius : ℝ), 
      (center.1 - A.1)^2 + (center.2 - A.2)^2 = radius^2 ∧
      (center.1 - B.1)^2 + (center.2 - B.2)^2 = radius^2 ∧
      (center.1 - C.1)^2 + (center.2 - C.2)^2 = radius^2 :=
by sorry


end circumcircle_equation_l1231_123165


namespace max_lateral_surface_area_l1231_123112

theorem max_lateral_surface_area (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 10 → 2 * π * x * y ≤ 50 * π :=
by sorry

end max_lateral_surface_area_l1231_123112


namespace solve_equation_l1231_123197

theorem solve_equation (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 := by
  sorry

end solve_equation_l1231_123197


namespace root_product_cubic_l1231_123189

theorem root_product_cubic (a b c : ℂ) : 
  (∀ x : ℂ, 3 * x^3 - 8 * x^2 + 5 * x - 9 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a * b * c = 3 := by
sorry

end root_product_cubic_l1231_123189


namespace walter_seal_time_l1231_123129

/-- Given Walter's zoo visit, prove he spent 13 minutes looking at seals. -/
theorem walter_seal_time : ∀ (S : ℕ), 
  S + 8 * S + 13 = 130 → S = 13 := by
  sorry

end walter_seal_time_l1231_123129


namespace problem_solution_l1231_123113

theorem problem_solution (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end problem_solution_l1231_123113


namespace work_completion_time_l1231_123114

-- Define the work rates
def work_rate_A : ℚ := 1 / 9
def work_rate_B : ℚ := 1 / 18
def work_rate_combined : ℚ := 1 / 6

-- Define the completion times
def time_A : ℚ := 9
def time_B : ℚ := 18
def time_combined : ℚ := 6

-- Theorem statement
theorem work_completion_time :
  (work_rate_A + work_rate_B = work_rate_combined) →
  (1 / work_rate_A = time_A) →
  (1 / work_rate_B = time_B) →
  (1 / work_rate_combined = time_combined) →
  time_B = 18 := by
  sorry


end work_completion_time_l1231_123114


namespace profit_margin_increase_l1231_123110

theorem profit_margin_increase (initial_margin : ℝ) (final_margin : ℝ) : 
  initial_margin = 0.25 →
  final_margin = 0.40 →
  let initial_price := 1 + initial_margin
  let final_price := 1 + final_margin
  (final_price / initial_price - 1) * 100 = 12 := by
sorry

end profit_margin_increase_l1231_123110


namespace total_cable_cost_neighborhood_cable_cost_l1231_123134

/-- The total cost of cable for a neighborhood with the given street configuration and cable requirements. -/
theorem total_cable_cost (ew_streets : ℕ) (ew_length : ℝ) (ns_streets : ℕ) (ns_length : ℝ) 
  (cable_per_mile : ℝ) (cost_per_mile : ℝ) : ℝ :=
  let total_street_length := ew_streets * ew_length + ns_streets * ns_length
  let total_cable_length := total_street_length * cable_per_mile
  total_cable_length * cost_per_mile

/-- The total cost of cable for the specific neighborhood described in the problem. -/
theorem neighborhood_cable_cost : total_cable_cost 18 2 10 4 5 2000 = 760000 := by
  sorry

end total_cable_cost_neighborhood_cable_cost_l1231_123134


namespace x_minus_y_value_l1231_123115

theorem x_minus_y_value (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 20) : x - y = 5/2 := by
  sorry

end x_minus_y_value_l1231_123115


namespace greatest_integer_b_for_quadratic_range_l1231_123124

theorem greatest_integer_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 15 ≠ -6) ↔ b ≤ 8 :=
sorry

end greatest_integer_b_for_quadratic_range_l1231_123124


namespace ravon_has_card_4_l1231_123182

structure Player where
  name : String
  score : Nat
  cards : Finset Nat

def card_set : Finset Nat := Finset.range 10

theorem ravon_has_card_4 (players : Finset Player)
  (h1 : players.card = 5)
  (h2 : ∀ p ∈ players, p.cards ⊆ card_set)
  (h3 : ∀ p ∈ players, p.cards.card = 2)
  (h4 : ∀ p ∈ players, p.score = (p.cards.sum id))
  (h5 : ∃ p ∈ players, p.name = "Ravon" ∧ p.score = 11)
  (h6 : ∃ p ∈ players, p.name = "Oscar" ∧ p.score = 4)
  (h7 : ∃ p ∈ players, p.name = "Aditi" ∧ p.score = 7)
  (h8 : ∃ p ∈ players, p.name = "Tyrone" ∧ p.score = 16)
  (h9 : ∃ p ∈ players, p.name = "Kim" ∧ p.score = 17)
  (h10 : ∀ c ∈ card_set, (players.filter (λ p => c ∈ p.cards)).card = 1) :
  ∃ p ∈ players, p.name = "Ravon" ∧ 4 ∈ p.cards :=
sorry

end ravon_has_card_4_l1231_123182


namespace max_large_chips_l1231_123190

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_large_chips :
  ∀ (small large prime : ℕ),
    small + large = 61 →
    small = large + prime →
    is_prime prime →
    large ≤ 29 :=
by sorry

end max_large_chips_l1231_123190


namespace cubic_root_sum_l1231_123139

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 13*r - 6 = 0 ∧ 
  s^3 - 15*s^2 + 13*s - 6 = 0 ∧ 
  t^3 - 15*t^2 + 13*t - 6 = 0 →
  r / (1/r + s*t) + s / (1/s + t*r) + t / (1/t + r*s) = 199/7 := by
sorry

end cubic_root_sum_l1231_123139


namespace impossible_cube_permutation_l1231_123169

/-- Represents a position in the 3x3x3 cube -/
structure Position :=
  (x y z : Fin 3)

/-- Represents a labeling of the 27 unit cubes -/
def Labeling := Fin 27 → Position

/-- Represents a move: swapping cube 27 with a neighbor -/
inductive Move
  | swap : Position → Move

/-- The parity of a position (even sum of coordinates is black, odd is white) -/
def Position.parity (p : Position) : Bool :=
  (p.x + p.y + p.z) % 2 = 0

/-- The final permutation required by the problem -/
def finalPermutation (n : Fin 27) : Fin 27 :=
  if n = 27 then 27 else 27 - n

/-- Theorem stating the impossibility of the required sequence of moves -/
theorem impossible_cube_permutation (initial : Labeling) :
  ¬ ∃ (moves : List Move), 
    (∀ n : Fin 27, 
      (initial n).parity = (initial (finalPermutation n)).parity) ∧
    (moves.length % 2 = 0) :=
  sorry

end impossible_cube_permutation_l1231_123169


namespace log_25_between_1_and_2_l1231_123166

theorem log_25_between_1_and_2 :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 25 / Real.log 10 ∧ Real.log 25 / Real.log 10 < b :=
sorry

end log_25_between_1_and_2_l1231_123166


namespace point_same_side_condition_l1231_123138

/-- A point on a line is on the same side as the origin with respect to another line -/
def same_side_as_origin (k b : ℝ) : Prop :=
  ∀ x : ℝ, (x - (k * x + b) + 2) * 2 > 0

/-- Theorem: If a point on y = kx + b is on the same side as the origin
    with respect to x - y + 2 = 0, then k = 1 and b < 2 -/
theorem point_same_side_condition (k b : ℝ) :
  same_side_as_origin k b → k = 1 ∧ b < 2 := by
  sorry

end point_same_side_condition_l1231_123138


namespace number_puzzle_l1231_123158

theorem number_puzzle (x : ℤ) : x - 13 = 31 → x + 11 = 55 := by
  sorry

end number_puzzle_l1231_123158


namespace fifth_subject_score_l1231_123179

/-- Given a student with 5 subject scores, prove that if 4 scores are known
    and the average of all 5 scores is 73, then the fifth score must be 85. -/
theorem fifth_subject_score
  (scores : Fin 5 → ℕ)
  (known_scores : scores 0 = 55 ∧ scores 1 = 67 ∧ scores 2 = 76 ∧ scores 3 = 82)
  (average : (scores 0 + scores 1 + scores 2 + scores 3 + scores 4) / 5 = 73) :
  scores 4 = 85 := by
sorry

end fifth_subject_score_l1231_123179


namespace min_value_of_expression_l1231_123171

theorem min_value_of_expression (x : ℝ) (h : x ≠ -7) :
  (2 * x^2 + 98) / (x + 7)^2 ≥ 1 := by
sorry

end min_value_of_expression_l1231_123171
