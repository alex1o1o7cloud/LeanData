import Mathlib

namespace smallest_four_digit_divisible_by_digits_l2869_286943

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

/-- Extracts the digits of a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- Checks if all elements in a list are distinct -/
def allDistinct (l : List ℕ) : Prop :=
  l.Nodup

theorem smallest_four_digit_divisible_by_digits :
  ∃ (n : ℕ),
    1000 ≤ n ∧ n < 10000 ∧
    (∀ d ∈ digits n, d ≠ 0 → isDivisibleBy n d) ∧
    allDistinct (digits n) ∧
    (∀ m, 1000 ≤ m ∧ m < n →
      ¬(∀ d ∈ digits m, d ≠ 0 → isDivisibleBy m d) ∨
      ¬(allDistinct (digits m))) ∧
    n = 1236 := by
  sorry

end smallest_four_digit_divisible_by_digits_l2869_286943


namespace gcd_of_80_180_450_l2869_286902

theorem gcd_of_80_180_450 : Nat.gcd 80 (Nat.gcd 180 450) = 10 := by sorry

end gcd_of_80_180_450_l2869_286902


namespace tank_filling_time_l2869_286982

/-- Represents the time taken to fill a tank using different pipe configurations -/
structure TankFilling where
  /-- Time taken by pipe B alone to fill the tank -/
  time_B : ℝ
  /-- Time taken by all three pipes together to fill the tank -/
  time_ABC : ℝ

/-- Proves that given the conditions, the time taken for all three pipes to fill the tank is 10 hours -/
theorem tank_filling_time (t : TankFilling) (h1 : t.time_B = 35) : t.time_ABC = 10 := by
  sorry

#check tank_filling_time

end tank_filling_time_l2869_286982


namespace probability_factor_less_than_5_l2869_286977

def factors_of_90 : Finset ℕ := sorry

theorem probability_factor_less_than_5 : 
  (Finset.filter (λ x => x < 5) factors_of_90).card / factors_of_90.card = 1 / 4 := by
  sorry

end probability_factor_less_than_5_l2869_286977


namespace tangent_secant_theorem_l2869_286976

theorem tangent_secant_theorem :
  ∃! (s : Finset ℕ), 
    (∀ t ∈ s, ∃ m n : ℕ, 
      t * t = m * n ∧ 
      m + n = 10 ∧ 
      m ≠ n ∧ 
      m > 0 ∧ 
      n > 0) ∧ 
    s.card = 2 := by
  sorry

end tangent_secant_theorem_l2869_286976


namespace expression_factorization_l2869_286936

theorem expression_factorization (a b c : ℝ) :
  ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = (a + b) * (a + c) * (b + c) :=
by sorry

end expression_factorization_l2869_286936


namespace geometric_sequence_sum_l2869_286926

theorem geometric_sequence_sum (a : ℕ → ℝ) (l : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 4 = 8 * a 1 →  -- given condition
  (a 1 + (a 2 + l) + a 3) * 2 = a 1 + (a 2 + l) * 2 + a 3 →  -- arithmetic sequence condition
  a 1 + a 2 + a 3 + a 4 + a 5 = 62 :=
by
  sorry

end geometric_sequence_sum_l2869_286926


namespace g_of_3_eq_10_l2869_286932

/-- The function g defined for all real numbers -/
def g (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that g(3) = 10 -/
theorem g_of_3_eq_10 : g 3 = 10 := by
  sorry

end g_of_3_eq_10_l2869_286932


namespace complement_of_A_range_of_m_l2869_286929

-- Define the sets A and B
def A : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}
def B (m : ℝ) : Set ℝ := {x | x ≥ m - 1}

-- Theorem for the complement of A
theorem complement_of_A : 
  (Set.univ \ A) = {x : ℝ | x < 3} := by sorry

-- Theorem for the range of m when A ⊆ B
theorem range_of_m (h : A ⊆ B m) : 
  m ≤ 4 := by sorry

end complement_of_A_range_of_m_l2869_286929


namespace carols_weight_l2869_286967

/-- Given two people's weights satisfying certain conditions, prove Carol's weight. -/
theorem carols_weight (alice_weight carol_weight : ℝ) 
  (h1 : alice_weight + carol_weight = 220)
  (h2 : alice_weight + 2 * carol_weight = 280) : 
  carol_weight = 60 := by
  sorry

end carols_weight_l2869_286967


namespace determinant_implies_fraction_value_l2869_286905

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem statement
theorem determinant_implies_fraction_value (θ : ℝ) :
  det (Real.sin θ) 2 (Real.cos θ) 1 = 0 →
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 3 :=
by sorry

end determinant_implies_fraction_value_l2869_286905


namespace complement_intersection_theorem_l2869_286903

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2, 3}

-- Define set N
def N : Set Nat := {2, 3, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {4} := by
  sorry

end complement_intersection_theorem_l2869_286903


namespace probability_at_least_two_same_l2869_286927

def num_dice : ℕ := 8
def sides_per_die : ℕ := 8

theorem probability_at_least_two_same (num_dice : ℕ) (sides_per_die : ℕ) :
  num_dice = 8 ∧ sides_per_die = 8 →
  (1 : ℚ) - (Nat.factorial num_dice : ℚ) / (sides_per_die ^ num_dice : ℚ) = 1291 / 1296 := by
  sorry

end probability_at_least_two_same_l2869_286927


namespace largest_divisible_n_l2869_286900

theorem largest_divisible_n : 
  ∀ n : ℕ, n > 5376 → ¬(((n : ℤ)^3 + 200) % (n - 8) = 0) :=
by sorry

end largest_divisible_n_l2869_286900


namespace min_workers_for_profit_l2869_286965

/-- Represents the company's daily operations and profit calculation -/
structure Company where
  maintenance_fee : ℕ := 600
  hourly_wage : ℕ := 20
  widgets_per_hour : ℕ := 6
  widget_price : ℚ := 7/2
  work_hours : ℕ := 8

/-- Calculates whether the company is profitable given a number of workers -/
def is_profitable (c : Company) (workers : ℕ) : Prop :=
  (c.widgets_per_hour * c.work_hours * c.widget_price : ℚ) * workers >
  (c.maintenance_fee : ℚ) + (c.hourly_wage * c.work_hours : ℚ) * workers

/-- Theorem stating the minimum number of workers needed for profitability -/
theorem min_workers_for_profit (c : Company) :
  ∀ n : ℕ, is_profitable c n ↔ n ≥ 76 :=
sorry

end min_workers_for_profit_l2869_286965


namespace power_of_seven_mod_hundred_l2869_286949

theorem power_of_seven_mod_hundred : 7^2010 % 100 = 49 := by
  sorry

end power_of_seven_mod_hundred_l2869_286949


namespace sales_and_profit_formula_optimal_price_reduction_no_solution_for_higher_profit_l2869_286998

-- Define constants
def initial_cost : ℝ := 80
def initial_price : ℝ := 120
def initial_sales : ℝ := 20
def sales_increase_rate : ℝ := 2

-- Define functions
def daily_sales_increase (x : ℝ) : ℝ := sales_increase_rate * x
def profit_per_piece (x : ℝ) : ℝ := initial_price - initial_cost - x

-- Theorem 1
theorem sales_and_profit_formula (x : ℝ) :
  (daily_sales_increase x = 2 * x) ∧ (profit_per_piece x = 40 - x) := by sorry

-- Theorem 2
theorem optimal_price_reduction :
  ∃ x : ℝ, (profit_per_piece x) * (initial_sales + daily_sales_increase x) = 1200 ∧ x = 20 := by sorry

-- Theorem 3
theorem no_solution_for_higher_profit :
  ¬∃ y : ℝ, (profit_per_piece y) * (initial_sales + daily_sales_increase y) = 1800 := by sorry

end sales_and_profit_formula_optimal_price_reduction_no_solution_for_higher_profit_l2869_286998


namespace arithmetic_progression_ratio_l2869_286994

theorem arithmetic_progression_ratio (a d : ℝ) : 
  (7 * a + (7 * (7 - 1) / 2) * d = 3 * a + (3 * (3 - 1) / 2) * d + 20) → 
  a / d = 1 / 2 := by
sorry

end arithmetic_progression_ratio_l2869_286994


namespace exists_special_triangle_l2869_286945

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The length of the median from vertex B to the midpoint of AC. -/
def median_length (t : Triangle) : ℝ := sorry

/-- The length of the angle bisector from vertex C. -/
def angle_bisector_length (t : Triangle) : ℝ := sorry

/-- The length of the altitude from vertex A to BC. -/
def altitude_length (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is scalene (no two sides are equal). -/
def is_scalene (t : Triangle) : Prop := sorry

/-- 
There exists a scalene triangle where the median from B, 
the angle bisector from C, and the altitude from A are all equal.
-/
theorem exists_special_triangle : 
  ∃ t : Triangle, 
    is_scalene t ∧ 
    median_length t = angle_bisector_length t ∧
    angle_bisector_length t = altitude_length t :=
sorry

end exists_special_triangle_l2869_286945


namespace class_A_student_count_l2869_286980

/-- Represents the number of students in class (A) -/
def class_A_students : ℕ := 50

/-- Represents the total number of groups in class (A) -/
def total_groups : ℕ := 8

/-- Represents the number of groups with 6 people -/
def groups_of_six : ℕ := 6

/-- Represents the number of groups with 7 people -/
def groups_of_seven : ℕ := 2

/-- Represents the number of people in each of the smaller groups -/
def people_in_smaller_groups : ℕ := 6

/-- Represents the number of people in each of the larger groups -/
def people_in_larger_groups : ℕ := 7

theorem class_A_student_count : 
  class_A_students = 
    groups_of_six * people_in_smaller_groups + 
    groups_of_seven * people_in_larger_groups ∧
  total_groups = groups_of_six + groups_of_seven := by
  sorry

end class_A_student_count_l2869_286980


namespace stripe_area_on_silo_l2869_286924

/-- The area of a stripe painted on a cylindrical silo -/
theorem stripe_area_on_silo (d h w r θ : ℝ) (hd : d = 40) (hh : h = 100) (hw : w = 4) (hr : r = 3) (hθ : θ = 30 * π / 180) :
  let circumference := π * d
  let stripe_length := r * circumference
  let effective_height := h / Real.cos θ
  let stripe_area := w * effective_height
  ⌊stripe_area⌋ = 462 := by sorry

end stripe_area_on_silo_l2869_286924


namespace prob_one_head_in_three_flips_l2869_286985

/-- The probability of getting exactly one head in three flips of a fair coin is 3/8 -/
theorem prob_one_head_in_three_flips :
  let p : ℝ := 1/2  -- probability of heads for a fair coin
  let n : ℕ := 3    -- number of flips
  let k : ℕ := 1    -- number of heads we want
  (n.choose k) * p^k * (1-p)^(n-k) = 3/8 := by
  sorry

end prob_one_head_in_three_flips_l2869_286985


namespace smallest_gcd_qr_l2869_286941

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) :
  ∃ (q' r' : ℕ+), Nat.gcd q' r' = 10 ∧ ∀ (q'' r'' : ℕ+), Nat.gcd q'' r'' ≥ 10 :=
by sorry

end smallest_gcd_qr_l2869_286941


namespace max_value_ab_l2869_286925

theorem max_value_ab (a b : ℝ) (h : ∀ x : ℝ, Real.exp x ≥ a * x + b) : 
  (∀ c d : ℝ, (∀ y : ℝ, Real.exp y ≥ c * y + d) → a * b ≥ c * d) → a * b = Real.exp 1 / 2 := by
  sorry

end max_value_ab_l2869_286925


namespace fraction_simplest_form_l2869_286907

/-- A fraction is in simplest form if its numerator and denominator have no common factors other than 1. -/
def IsSimplestForm (n d : ℤ) : Prop :=
  ∀ k : ℤ, k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1

/-- Given x and y are integers and x ≠ 0, prove that (x+y)/(2x) is in simplest form. -/
theorem fraction_simplest_form (x y : ℤ) (hx : x ≠ 0) : 
  IsSimplestForm (x + y) (2 * x) :=
sorry

end fraction_simplest_form_l2869_286907


namespace a_less_than_b_less_than_one_l2869_286913

theorem a_less_than_b_less_than_one
  (x a b : ℝ)
  (hx : x > 0)
  (ha : a > 0)
  (hb : b > 0)
  (h : a^x < b^x ∧ b^x < 1) :
  a < b ∧ b < 1 :=
by sorry

end a_less_than_b_less_than_one_l2869_286913


namespace total_ninja_stars_l2869_286915

/-- The number of ninja throwing stars each person has -/
structure NinjaStars where
  eric : ℕ
  chad : ℕ
  mark : ℕ
  jennifer : ℕ

/-- The initial distribution of ninja throwing stars -/
def initial_distribution : NinjaStars where
  eric := 4
  chad := 4 * 2
  mark := 0
  jennifer := 0

/-- The final distribution of ninja throwing stars after all transactions -/
def final_distribution : NinjaStars :=
  let chad_after_selling := initial_distribution.chad - 2
  let chad_final := chad_after_selling + 3
  let mark_jennifer_share := 10 / 2
  { eric := initial_distribution.eric,
    chad := chad_final,
    mark := mark_jennifer_share,
    jennifer := mark_jennifer_share }

/-- The theorem stating the total number of ninja throwing stars -/
theorem total_ninja_stars :
  final_distribution.eric +
  final_distribution.chad +
  final_distribution.mark +
  final_distribution.jennifer = 23 :=
by
  sorry

end total_ninja_stars_l2869_286915


namespace total_peaches_is_twelve_l2869_286923

/-- The number of baskets -/
def num_baskets : ℕ := 2

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 4

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 2

/-- The total number of peaches in all baskets -/
def total_peaches : ℕ := num_baskets * (red_peaches_per_basket + green_peaches_per_basket)

theorem total_peaches_is_twelve : total_peaches = 12 := by
  sorry

end total_peaches_is_twelve_l2869_286923


namespace smallest_triangle_side_l2869_286938

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def smallest_t : ℕ → Prop
| t => is_triangle 7.5 11 (t : ℝ) ∧ ∀ s : ℕ, s < t → ¬is_triangle 7.5 11 (s : ℝ)

theorem smallest_triangle_side : smallest_t 4 := by
  sorry

end smallest_triangle_side_l2869_286938


namespace rest_of_body_length_l2869_286984

theorem rest_of_body_length (total_height legs_ratio head_ratio : ℚ) : 
  total_height = 60 →
  legs_ratio = 1/3 →
  head_ratio = 1/4 →
  total_height - (legs_ratio * total_height + head_ratio * total_height) = 25 := by
sorry

end rest_of_body_length_l2869_286984


namespace course_selection_theorem_l2869_286950

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def total_courses : ℕ := physical_education_courses + art_courses

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def two_course_selections : ℕ := choose physical_education_courses 1 * choose art_courses 1

def three_course_selections : ℕ := 
  choose physical_education_courses 2 * choose art_courses 1 +
  choose physical_education_courses 1 * choose art_courses 2

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_theorem : total_selections = 64 := by
  sorry

end course_selection_theorem_l2869_286950


namespace consistent_grade_theorem_l2869_286954

/-- Represents the grades a student can receive -/
inductive Grade
| A
| B
| C
| D

/-- Represents the grade distribution for a test -/
structure GradeDistribution where
  a_count : Nat
  b_count : Nat
  c_count : Nat
  d_count : Nat

/-- The problem setup -/
structure TestConsistency where
  total_students : Nat
  num_tests : Nat
  grade_distribution : GradeDistribution

/-- Calculate the percentage of students with consistent grades -/
def consistent_grade_percentage (tc : TestConsistency) : Rat :=
  let consistent_count := tc.grade_distribution.a_count + tc.grade_distribution.b_count + 
                          tc.grade_distribution.c_count + tc.grade_distribution.d_count
  (consistent_count : Rat) / (tc.total_students : Rat) * 100

/-- The main theorem to prove -/
theorem consistent_grade_theorem (tc : TestConsistency) 
  (h1 : tc.total_students = 40)
  (h2 : tc.num_tests = 3)
  (h3 : tc.grade_distribution = { a_count := 3, b_count := 6, c_count := 7, d_count := 2 }) :
  consistent_grade_percentage tc = 45 := by
  sorry


end consistent_grade_theorem_l2869_286954


namespace corner_removed_cube_vertex_count_l2869_286983

/-- Represents a cube with a given side length. -/
structure Cube where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents the resulting solid after removing smaller cubes from the corners of a larger cube. -/
structure CornerRemovedCube where
  originalCube : Cube
  removedCubeSideLength : ℝ
  removedCubeSideLength_pos : removedCubeSideLength > 0
  removedCubeSideLength_lt : removedCubeSideLength < originalCube.sideLength

/-- Calculates the number of vertices in the resulting solid after removing smaller cubes from the corners of a larger cube. -/
def vertexCount (c : CornerRemovedCube) : ℕ :=
  8 * 5  -- Each corner of the original cube contributes 5 vertices

/-- Theorem stating that removing cubes of side length 2 from each corner of a cube with side length 5 results in a solid with 40 vertices. -/
theorem corner_removed_cube_vertex_count :
  ∀ (c : CornerRemovedCube),
  c.originalCube.sideLength = 5 →
  c.removedCubeSideLength = 2 →
  vertexCount c = 40 :=
by
  sorry


end corner_removed_cube_vertex_count_l2869_286983


namespace mass_of_substance_l2869_286971

-- Define the density of the substance
def density : ℝ := 500

-- Define the volume in cubic centimeters
def volume_cm : ℝ := 2

-- Define the conversion factor from cm³ to m³
def cm3_to_m3 : ℝ := 1e-6

-- Define the mass in kg
def mass : ℝ := density * (volume_cm * cm3_to_m3)

-- Theorem statement
theorem mass_of_substance :
  mass = 1e-3 := by sorry

end mass_of_substance_l2869_286971


namespace gcf_lcm_sum_l2869_286918

def numbers : List Nat := [9, 15, 27]

/-- The greatest common factor of 9, 15, and 27 -/
def A : Nat := numbers.foldl Nat.gcd 0

/-- The least common multiple of 9, 15, and 27 -/
def B : Nat := numbers.foldl Nat.lcm 1

/-- Theorem stating that the sum of the greatest common factor and 
    the least common multiple of 9, 15, and 27 is equal to 138 -/
theorem gcf_lcm_sum : A + B = 138 := by
  sorry

end gcf_lcm_sum_l2869_286918


namespace range_of_a_l2869_286914

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := (x - t) * abs x

-- State the theorem
theorem range_of_a (t : ℝ) (h_t : t ∈ Set.Ioo 0 2) :
  (∀ x ∈ Set.Icc (-1) 2, ∃ a : ℝ, f t x > x + a) →
  (∃ a : ℝ, ∀ x ∈ Set.Icc (-1) 2, f t x > x + a ∧ a ≤ -1/4) :=
sorry

end range_of_a_l2869_286914


namespace min_value_of_b_over_a_l2869_286920

theorem min_value_of_b_over_a (a b : ℝ) : 
  (∀ x > -1, Real.log (x + 1) - 1 ≤ a * x + b) → 
  (∃ c, c = 1 - Real.exp 1 ∧ ∀ a b, (∀ x > -1, Real.log (x + 1) - 1 ≤ a * x + b) → b / a ≥ c) :=
sorry

end min_value_of_b_over_a_l2869_286920


namespace long_division_problem_l2869_286970

theorem long_division_problem (dividend divisor quotient : ℕ) 
  (h1 : dividend = divisor * quotient)
  (h2 : dividend % divisor = 0)
  (h3 : (dividend / divisor) * 105 = 2015 * 10) :
  dividend = 20685 := by
  sorry

end long_division_problem_l2869_286970


namespace g_difference_l2869_286910

/-- The function g defined as g(x) = 6x^2 + 3x - 4 -/
def g (x : ℝ) : ℝ := 6 * x^2 + 3 * x - 4

/-- Theorem stating that g(x+h) - g(x) = h(12x + 6h + 3) for all real x and h -/
theorem g_difference (x h : ℝ) : g (x + h) - g x = h * (12 * x + 6 * h + 3) := by
  sorry

end g_difference_l2869_286910


namespace gcd_of_powers_of_two_minus_one_l2869_286942

theorem gcd_of_powers_of_two_minus_one :
  Nat.gcd (2^1015 - 1) (2^1020 - 1) = 1 := by sorry

end gcd_of_powers_of_two_minus_one_l2869_286942


namespace hailstone_conjecture_instance_l2869_286939

def hailstone_seq (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 1) = if a n % 2 = 0 then a n / 2 else 3 * a n + 1

theorem hailstone_conjecture_instance : ∃ a₁ : ℕ, a₁ < 50 ∧
  (∃ a : ℕ → ℕ, hailstone_seq a ∧ a 1 = a₁ ∧ a 10 = 1 ∧ ∀ i ∈ Finset.range 9, a (i + 1) ≠ 1) :=
sorry

end hailstone_conjecture_instance_l2869_286939


namespace max_value_quadratic_l2869_286968

theorem max_value_quadratic :
  (∃ (p : ℝ), -3 * p^2 + 24 * p + 5 = 53) ∧
  (∀ (p : ℝ), -3 * p^2 + 24 * p + 5 ≤ 53) := by
sorry

end max_value_quadratic_l2869_286968


namespace g_sum_negative_one_l2869_286960

noncomputable section

variable (f g : ℝ → ℝ)

axiom functional_equation : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y
axiom f_equality : f (-2) = f 1
axiom f_nonzero : f 1 ≠ 0

theorem g_sum_negative_one : g 1 + g (-1) = -1 := by
  sorry

end g_sum_negative_one_l2869_286960


namespace ellipse_parabola_intersection_range_l2869_286973

/-- The range of 'a' for which the ellipse x^2 + 4(y-a)^2 = 4 and 
    the parabola x^2 = 2y have common points -/
theorem ellipse_parabola_intersection_range :
  ∀ a : ℝ, 
  (∃ x y : ℝ, x^2 + 4*(y-a)^2 = 4 ∧ x^2 = 2*y) → 
  -1 ≤ a ∧ a ≤ 17/8 :=
by sorry

end ellipse_parabola_intersection_range_l2869_286973


namespace interval_for_72_students_8_sample_l2869_286962

/-- The interval of segmentation in systematic sampling -/
def interval_of_segmentation (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  total_students / sample_size

/-- Theorem: The interval of segmentation for 72 students and sample size 8 is 9 -/
theorem interval_for_72_students_8_sample : 
  interval_of_segmentation 72 8 = 9 := by
  sorry

end interval_for_72_students_8_sample_l2869_286962


namespace dairy_farm_husk_consumption_l2869_286978

/-- Represents the average number of days it takes for one cow to eat one bag of husk -/
def average_days_per_bag (num_cows : ℕ) (total_bags : ℕ) (total_days : ℕ) : ℚ :=
  (num_cows * total_days : ℚ) / total_bags

/-- Proves that given 30 cows consuming 50 bags of husk in 20 days, 
    the average number of days it takes for one cow to eat one bag of husk is 12 days -/
theorem dairy_farm_husk_consumption :
  average_days_per_bag 30 50 20 = 12 := by
  sorry

end dairy_farm_husk_consumption_l2869_286978


namespace polynomial_value_theorem_l2869_286931

/-- A fourth-degree polynomial with real coefficients -/
def fourth_degree_poly (g : ℝ → ℝ) : Prop :=
  ∃ a b c d e : ℝ, ∀ x, g x = a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem polynomial_value_theorem (g : ℝ → ℝ) 
  (h_poly : fourth_degree_poly g)
  (h_m1 : |g (-1)| = 15)
  (h_0 : |g 0| = 15)
  (h_2 : |g 2| = 15)
  (h_3 : |g 3| = 15)
  (h_4 : |g 4| = 15) :
  |g 1| = 11 := by
  sorry

end polynomial_value_theorem_l2869_286931


namespace fans_per_bleacher_set_l2869_286989

theorem fans_per_bleacher_set (total_fans : ℕ) (num_bleacher_sets : ℕ) 
  (h1 : total_fans = 2436) 
  (h2 : num_bleacher_sets = 3) 
  (h3 : total_fans % num_bleacher_sets = 0) : 
  total_fans / num_bleacher_sets = 812 := by
  sorry

end fans_per_bleacher_set_l2869_286989


namespace arithmetic_mean_problem_l2869_286948

theorem arithmetic_mean_problem (a b c d : ℝ) :
  (a + b + c + d + 130) / 5 = 90 →
  (a + b + c + d) / 4 = 80 :=
by
  sorry

end arithmetic_mean_problem_l2869_286948


namespace factorization_equality_l2869_286996

theorem factorization_equality (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := by
  sorry

end factorization_equality_l2869_286996


namespace distinguishable_triangles_l2869_286987

/-- Represents the number of available colors for small triangles -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles in the inner part of the large triangle -/
def inner_triangles : ℕ := 3

/-- Represents the number of small triangles in the outer part of the large triangle -/
def outer_triangles : ℕ := 3

/-- Represents the total number of small triangles in the large triangle -/
def total_triangles : ℕ := inner_triangles + outer_triangles

/-- Calculates the number of ways to color the inner triangle -/
def inner_colorings : ℕ := 
  num_colors + (num_colors * (num_colors - 1)) + (num_colors.choose inner_triangles * inner_triangles.factorial)

/-- Calculates the number of ways to color the outer triangle -/
def outer_colorings : ℕ := 
  num_colors + (num_colors * (num_colors - 1)) + (num_colors.choose outer_triangles * outer_triangles.factorial)

/-- Theorem stating the total number of distinguishable large triangles -/
theorem distinguishable_triangles : 
  inner_colorings * outer_colorings = 116096 := by sorry

end distinguishable_triangles_l2869_286987


namespace wedding_bouquets_l2869_286951

/-- Represents the number of flowers of each type --/
structure FlowerCount where
  roses : ℕ
  lilies : ℕ
  tulips : ℕ
  sunflowers : ℕ

/-- Represents the requirements for a single bouquet --/
def BouquetRequirement : FlowerCount :=
  { roses := 2, lilies := 1, tulips := 3, sunflowers := 1 }

/-- Calculates the number of complete bouquets that can be made --/
def completeBouquets (available : FlowerCount) : ℕ :=
  min (available.roses / BouquetRequirement.roses)
    (min (available.lilies / BouquetRequirement.lilies)
      (min (available.tulips / BouquetRequirement.tulips)
        (available.sunflowers / BouquetRequirement.sunflowers)))

theorem wedding_bouquets :
  let initial : FlowerCount := { roses := 48, lilies := 40, tulips := 76, sunflowers := 34 }
  let wilted : FlowerCount := { roses := 24, lilies := 10, tulips := 14, sunflowers := 7 }
  let remaining : FlowerCount := {
    roses := initial.roses - wilted.roses,
    lilies := initial.lilies - wilted.lilies,
    tulips := initial.tulips - wilted.tulips,
    sunflowers := initial.sunflowers - wilted.sunflowers
  }
  completeBouquets remaining = 12 := by sorry

end wedding_bouquets_l2869_286951


namespace min_value_expression_l2869_286940

theorem min_value_expression (x y : ℝ) (h1 : x * y + 3 * x = 3) (h2 : 0 < x) (h3 : x < 1/2) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x' y' : ℝ), x' * y' + 3 * x' = 3 → 0 < x' → x' < 1/2 → 3 / x' + 1 / (y' - 3) ≥ m :=
by sorry

end min_value_expression_l2869_286940


namespace hair_cut_first_day_l2869_286901

/-- Given that Elizabeth had her hair cut on two consecutive days, with a total of 0.88 inches
    cut off and 0.5 inches cut off on the second day, this theorem proves that 0.38 inches
    were cut off on the first day. -/
theorem hair_cut_first_day (total : ℝ) (second_day : ℝ) (h1 : total = 0.88) (h2 : second_day = 0.5) :
  total - second_day = 0.38 := by
  sorry

end hair_cut_first_day_l2869_286901


namespace tiger_deer_chase_theorem_l2869_286937

/-- Represents the chase between a tiger and a deer --/
structure TigerDeerChase where
  tiger_leaps_per_minute : ℕ
  deer_leaps_per_minute : ℕ
  tiger_meters_per_leap : ℕ
  deer_meters_per_leap : ℕ
  catch_distance : ℕ

/-- The number of leaps the tiger is initially behind the deer --/
def initial_leap_difference (chase : TigerDeerChase) : ℕ :=
  sorry

/-- Theorem stating the initial leap difference for the given chase scenario --/
theorem tiger_deer_chase_theorem (chase : TigerDeerChase) 
  (h1 : chase.tiger_leaps_per_minute = 5)
  (h2 : chase.deer_leaps_per_minute = 4)
  (h3 : chase.tiger_meters_per_leap = 8)
  (h4 : chase.deer_meters_per_leap = 5)
  (h5 : chase.catch_distance = 800) :
  initial_leap_difference chase = 40 := by
  sorry

end tiger_deer_chase_theorem_l2869_286937


namespace pittsburgh_schools_l2869_286958

theorem pittsburgh_schools (
  pittsburgh_stores : ℕ)
  (pittsburgh_hospitals : ℕ)
  (pittsburgh_police : ℕ)
  (new_city_total : ℕ) :
  pittsburgh_stores = 2000 →
  pittsburgh_hospitals = 500 →
  pittsburgh_police = 20 →
  new_city_total = 2175 →
  ∃ (pittsburgh_schools : ℕ),
    pittsburgh_schools = 200 ∧
    new_city_total = 
      pittsburgh_stores / 2 + 
      pittsburgh_hospitals * 2 + 
      (pittsburgh_schools - 50) + 
      (pittsburgh_police + 5) :=
by sorry

end pittsburgh_schools_l2869_286958


namespace quadratic_inequality_problem_l2869_286908

/-- Given that the solution set of x²-px-q<0 is {x | 2<x<3}, 
    prove the values of p and q and the solution set of qx²-px-1>0 -/
theorem quadratic_inequality_problem 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 - p*x - q < 0}) : 
  (p = 5 ∧ q = -6) ∧ 
  {x : ℝ | q*x^2 - p*x - 1 > 0} = Set.Ioo (-1/2) (-1/3) := by
  sorry


end quadratic_inequality_problem_l2869_286908


namespace inequality_holds_on_interval_largest_interval_l2869_286961

theorem inequality_holds_on_interval (x : ℝ) (h : x ∈ Set.Icc 0 4) :
  ∀ y : ℝ, y > 0 → (5 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y :=
sorry

theorem largest_interval :
  ∀ a : ℝ, a > 4 → ∃ y : ℝ, y > 0 ∧ (5 * (a * y^2 + a^2 * y + 4 * y^2 + 4 * a * y)) / (a + y) ≤ 3 * a^2 * y :=
sorry

end inequality_holds_on_interval_largest_interval_l2869_286961


namespace square_plus_one_geq_double_for_all_reals_l2869_286919

theorem square_plus_one_geq_double_for_all_reals :
  ∀ a : ℝ, a^2 + 1 ≥ 2*a := by sorry

end square_plus_one_geq_double_for_all_reals_l2869_286919


namespace max_friends_is_m_l2869_286999

/-- Represents a compartment of passengers -/
structure Compartment where
  passengers : Type
  friendship : passengers → passengers → Prop
  m : ℕ
  h_m : m ≥ 3
  h_symmetric : ∀ a b, friendship a b ↔ friendship b a
  h_irreflexive : ∀ a, ¬friendship a a
  h_unique_common_friend : ∀ (S : Finset passengers), S.card = m → 
    ∃! f, ∀ s ∈ S, friendship f s

/-- The maximum number of friends any passenger can have is m -/
theorem max_friends_is_m (C : Compartment) : 
  ∃ (max_friends : ℕ), max_friends = C.m ∧ 
    ∀ p : C.passengers, ∃ (friends : Finset C.passengers), 
      (∀ f ∈ friends, C.friendship p f) ∧ 
      friends.card ≤ max_friends :=
sorry

end max_friends_is_m_l2869_286999


namespace opposite_of_2022_l2869_286995

theorem opposite_of_2022 : -(2022 : ℝ) = -2022 := by sorry

end opposite_of_2022_l2869_286995


namespace isosceles_right_triangle_l2869_286991

theorem isosceles_right_triangle (a b c h_a h_b h_c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  h_a > 0 → h_b > 0 → h_c > 0 →
  a * h_a = 2 * area → b * h_b = 2 * area → c * h_c = 2 * area →
  a ≤ h_a → b ≤ h_b →
  a = b ∧ c = a * Real.sqrt 2 :=
sorry


end isosceles_right_triangle_l2869_286991


namespace integral_inequality_l2869_286934

theorem integral_inequality (a : ℝ) (ha : a > 1) :
  (1 / (a - 1)) * (1 - (Real.log a / (a - 1))) < 
  (a - Real.log a - 1) / (a * (Real.log a)^2) ∧
  (a - Real.log a - 1) / (a * (Real.log a)^2) < 
  (1 / Real.log a) * (1 - (Real.log (Real.log a + 1) / Real.log a)) := by
  sorry

end integral_inequality_l2869_286934


namespace hotel_room_charge_comparison_l2869_286935

theorem hotel_room_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.5 * R) 
  (h2 : P = G - 0.2 * G) : 
  R = 1.6 * G := by
  sorry

end hotel_room_charge_comparison_l2869_286935


namespace functional_equation_solution_l2869_286922

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) : 
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) → 
  ((∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = -x - 1)) := by
  sorry

end functional_equation_solution_l2869_286922


namespace geometric_sequence_sum_l2869_286990

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, a n > 0) →  -- positive sequence
  (∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25 →
  a 3 + a 5 = 5 := by
sorry

end geometric_sequence_sum_l2869_286990


namespace seashell_count_l2869_286997

theorem seashell_count (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) 
  (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end seashell_count_l2869_286997


namespace sqrt_difference_comparison_l2869_286955

theorem sqrt_difference_comparison (x : ℝ) (h : x ≥ 1) :
  Real.sqrt x - Real.sqrt (x - 1) > Real.sqrt (x + 1) - Real.sqrt x := by
  sorry

end sqrt_difference_comparison_l2869_286955


namespace parallelogram_diagonal_sum_l2869_286969

-- Define the parallelogram and its properties
structure Parallelogram :=
  (area : ℝ)
  (pq_length : ℝ)
  (rs_length : ℝ)

-- Define the diagonal representation
structure DiagonalRepresentation :=
  (m : ℕ)
  (n : ℕ)
  (p : ℕ)

-- Define the theorem
theorem parallelogram_diagonal_sum 
  (ABCD : Parallelogram) 
  (h_area : ABCD.area = 24)
  (h_pq : ABCD.pq_length = 8)
  (h_rs : ABCD.rs_length = 10)
  (d_rep : DiagonalRepresentation)
  (h_prime : ∀ (q : ℕ), Prime q → ¬(q^2 ∣ d_rep.p))
  (h_diagonal : ∃ (d : ℝ), d^2 = d_rep.m + d_rep.n * Real.sqrt d_rep.p) :
  d_rep.m + d_rep.n + d_rep.p = 50 := by
sorry

end parallelogram_diagonal_sum_l2869_286969


namespace smallest_bench_sections_l2869_286933

theorem smallest_bench_sections (N : ℕ) : 
  (∃ x : ℕ, x > 0 ∧ 8 * N = x ∧ 12 * N = x) ↔ N ≥ 3 :=
sorry

end smallest_bench_sections_l2869_286933


namespace sinusoidal_function_properties_l2869_286930

/-- Given a sinusoidal function with specific properties, prove its expression and range -/
theorem sinusoidal_function_properties (f : ℝ → ℝ) (A ω φ : ℝ)
  (h_def : ∀ x, f x = A * Real.sin (ω * x + φ))
  (h_A : A > 0)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π)
  (h_symmetry : (π / 2) = π / ω)  -- Distance between adjacent axes of symmetry
  (h_lowest : f (2 * π / 3) = -1/2)  -- One of the lowest points
  : (∀ x, f x = (1/2) * Real.sin (2 * x + π/6)) ∧
    (∀ x, x ∈ Set.Icc (-π/6) (π/3) → f x ∈ Set.Icc (-1/4) (1/2)) := by
  sorry

end sinusoidal_function_properties_l2869_286930


namespace total_sum_is_992_l2869_286993

/-- Represents the share of money for each person -/
structure Share where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The conditions of the money division problem -/
def money_division (s : Share) : Prop :=
  s.b = 0.75 * s.a ∧
  s.c = 0.60 * s.a ∧
  s.d = 0.45 * s.a ∧
  s.e = 0.30 * s.a ∧
  s.e = 96

/-- The theorem stating that the total sum of money is 992 -/
theorem total_sum_is_992 (s : Share) (h : money_division s) : 
  s.a + s.b + s.c + s.d + s.e = 992 := by
  sorry


end total_sum_is_992_l2869_286993


namespace smallest_mu_l2869_286916

theorem smallest_mu : 
  ∃ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≤ a*b + μ*b*c + c*d) ∧ 
  (∀ μ' : ℝ, (∀ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≤ a*b + μ'*b*c + c*d) → μ' ≥ μ) ∧
  μ = 3/2 :=
by sorry

end smallest_mu_l2869_286916


namespace kekai_sold_five_shirts_l2869_286986

/-- The number of shirts Kekai sold -/
def num_shirts : ℕ := sorry

/-- The number of pants Kekai sold -/
def num_pants : ℕ := 5

/-- The price of each shirt in dollars -/
def shirt_price : ℕ := 1

/-- The price of each pair of pants in dollars -/
def pants_price : ℕ := 3

/-- The amount of money Kekai has left after giving half to his parents -/
def money_left : ℕ := 10

/-- Theorem stating that Kekai sold 5 shirts -/
theorem kekai_sold_five_shirts :
  num_shirts = 5 :=
by
  sorry

end kekai_sold_five_shirts_l2869_286986


namespace worker_C_post_tax_income_l2869_286988

-- Define worker types
inductive Worker : Type
| A
| B
| C

-- Define survey types
inductive SurveyType : Type
| Basic
| Lifestyle
| Technology

-- Define payment rates for each worker and survey type
def baseRate (w : Worker) : ℚ :=
  match w with
  | Worker.A => 30
  | Worker.B => 25
  | Worker.C => 35

def rateMultiplier (w : Worker) (s : SurveyType) : ℚ :=
  match w, s with
  | Worker.A, SurveyType.Basic => 1
  | Worker.A, SurveyType.Lifestyle => 1.2
  | Worker.A, SurveyType.Technology => 1.5
  | Worker.B, SurveyType.Basic => 1
  | Worker.B, SurveyType.Lifestyle => 1.25
  | Worker.B, SurveyType.Technology => 1.45
  | Worker.C, SurveyType.Basic => 1
  | Worker.C, SurveyType.Lifestyle => 1.15
  | Worker.C, SurveyType.Technology => 1.6

-- Define commission rate for technology surveys
def commissionRate : ℚ := 0.05

-- Define tax rates for each worker
def taxRate (w : Worker) : ℚ :=
  match w with
  | Worker.A => 0.15
  | Worker.B => 0.18
  | Worker.C => 0.20

-- Define number of surveys completed by each worker
def surveysCompleted (w : Worker) (s : SurveyType) : ℕ :=
  match w, s with
  | Worker.A, SurveyType.Basic => 80
  | Worker.A, SurveyType.Lifestyle => 50
  | Worker.A, SurveyType.Technology => 35
  | Worker.B, SurveyType.Basic => 90
  | Worker.B, SurveyType.Lifestyle => 45
  | Worker.B, SurveyType.Technology => 40
  | Worker.C, SurveyType.Basic => 70
  | Worker.C, SurveyType.Lifestyle => 40
  | Worker.C, SurveyType.Technology => 60

-- Define health insurance deductions for each worker
def healthInsurance (w : Worker) : ℚ :=
  match w with
  | Worker.A => 200
  | Worker.B => 250
  | Worker.C => 300

-- Calculate earnings for a worker
def earnings (w : Worker) : ℚ :=
  let basicEarnings := (baseRate w) * (surveysCompleted w SurveyType.Basic)
  let lifestyleEarnings := (baseRate w) * (rateMultiplier w SurveyType.Lifestyle) * (surveysCompleted w SurveyType.Lifestyle)
  let techEarnings := (baseRate w) * (rateMultiplier w SurveyType.Technology) * (surveysCompleted w SurveyType.Technology)
  let techCommission := techEarnings * commissionRate
  basicEarnings + lifestyleEarnings + techEarnings + techCommission

-- Calculate post-tax income for a worker
def postTaxIncome (w : Worker) : ℚ :=
  let grossEarnings := earnings w
  let tax := grossEarnings * (taxRate w)
  grossEarnings - tax - (healthInsurance w)

-- Theorem to prove
theorem worker_C_post_tax_income :
  postTaxIncome Worker.C = 5770.40 :=
by sorry

end worker_C_post_tax_income_l2869_286988


namespace intersection_M_N_l2869_286904

def M : Set ℕ := {0, 2, 4, 8}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 4, 8} := by sorry

end intersection_M_N_l2869_286904


namespace power_function_classification_l2869_286911

/-- Definition of a power function -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ), ∀ (x : ℝ), f x = x ^ a

/-- The given functions -/
def f1 (x : ℝ) : ℝ := x^2 + 2
def f2 (x : ℝ) : ℝ := x^(1/2)
def f3 (x : ℝ) : ℝ := 2 * x^3
def f4 (x : ℝ) : ℝ := x^(3/4)
def f5 (x : ℝ) : ℝ := x^(1/3) + 1

/-- The theorem stating which functions are power functions -/
theorem power_function_classification :
  ¬(is_power_function f1) ∧
  (is_power_function f2) ∧
  ¬(is_power_function f3) ∧
  (is_power_function f4) ∧
  ¬(is_power_function f5) :=
sorry

end power_function_classification_l2869_286911


namespace f_properties_l2869_286921

open Real

noncomputable def f (x : ℝ) : ℝ := x / (exp x - 1)

theorem f_properties :
  (∀ x > 0, ∀ y > x, f y < f x) ∧
  (∀ a > 2, ∃ x₀ > 0, f x₀ < a / (exp x₀ + 1)) := by
  sorry

end f_properties_l2869_286921


namespace max_volume_sphere_in_cube_l2869_286953

theorem max_volume_sphere_in_cube (edge_length : Real) (h : edge_length = 1) :
  let sphere_volume := (4 / 3) * Real.pi * (edge_length / 2)^3
  sphere_volume = Real.pi / 6 := by
  sorry

end max_volume_sphere_in_cube_l2869_286953


namespace triangle_ABC_perimeter_l2869_286917

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) : Prop :=
  -- Condition for sides a and b
  (|a - 2| + (b - 5)^2 = 0) ∧
  -- Conditions for side c
  (c = 4) ∧
  (∀ x : ℤ, (x - 3 > 3*(x - 4) ∧ (4*x - 1) / 6 < x + 1) → x ≤ 4)

-- Theorem statement
theorem triangle_ABC_perimeter (a b c : ℝ) :
  triangle_ABC a b c → a + b + c = 11 :=
by
  sorry

end triangle_ABC_perimeter_l2869_286917


namespace equation_solution_l2869_286959

theorem equation_solution (y : ℝ) : 
  (y^3 - 3*y^2)/(y^2 + 2*y + 1) + 2*y = -1 ↔ y = 1/Real.sqrt 3 ∨ y = -1/Real.sqrt 3 :=
sorry

end equation_solution_l2869_286959


namespace smallest_number_of_cubes_for_given_box_l2869_286972

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem: The smallest number of identical cubes that can fill a box with dimensions 
    24 inches long, 40 inches wide, and 16 inches deep is 30 -/
theorem smallest_number_of_cubes_for_given_box :
  smallestNumberOfCubes { length := 24, width := 40, depth := 16 } = 30 := by
  sorry

end smallest_number_of_cubes_for_given_box_l2869_286972


namespace inclination_angle_range_l2869_286981

/-- Given a line with equation x*sin(α) - √3*y + 1 = 0, 
    the range of its inclination angle θ is [0, π/6] ∪ [5π/6, π) -/
theorem inclination_angle_range (α : Real) :
  let line := {(x, y) : Real × Real | x * Real.sin α - Real.sqrt 3 * y + 1 = 0}
  let θ := Real.arctan ((Real.sin α) / Real.sqrt 3)
  θ ∈ Set.union (Set.Icc 0 (π / 6)) (Set.Ico (5 * π / 6) π) := by
  sorry

end inclination_angle_range_l2869_286981


namespace lizette_minerva_stamp_difference_l2869_286966

theorem lizette_minerva_stamp_difference :
  let lizette_stamps : ℕ := 813
  let minerva_stamps : ℕ := 688
  lizette_stamps - minerva_stamps = 125 := by
sorry

end lizette_minerva_stamp_difference_l2869_286966


namespace jerry_shelf_problem_l2869_286956

/-- The number of action figures Jerry added to the shelf -/
def action_figures_added : ℕ := 7

/-- The initial number of action figures on the shelf -/
def initial_action_figures : ℕ := 5

/-- The number of books on the shelf (constant) -/
def books : ℕ := 9

/-- Theorem stating that the number of action figures added satisfies the problem conditions -/
theorem jerry_shelf_problem :
  initial_action_figures + action_figures_added = books + 3 :=
by sorry

end jerry_shelf_problem_l2869_286956


namespace min_value_a_l2869_286964

theorem min_value_a : 
  (∃ (a : ℝ), ∀ (x₁ x₂ x₃ x₄ : ℝ), 
    ∃ (k₁ k₂ k₃ k₄ : ℤ), 
      (x₂ - k₁ - (x₁ - k₂))^2 + 
      (x₃ - k₁ - (x₁ - k₃))^2 + 
      (x₄ - k₁ - (x₁ - k₄))^2 + 
      (x₃ - k₂ - (x₂ - k₃))^2 + 
      (x₄ - k₂ - (x₂ - k₄))^2 + 
      (x₄ - k₃ - (x₃ - k₄))^2 ≤ a) ∧ 
  (∀ (b : ℝ), (∀ (x₁ x₂ x₃ x₄ : ℝ), 
    ∃ (k₁ k₂ k₃ k₄ : ℤ), 
      (x₂ - k₁ - (x₁ - k₂))^2 + 
      (x₃ - k₁ - (x₁ - k₃))^2 + 
      (x₄ - k₁ - (x₁ - k₄))^2 + 
      (x₃ - k₂ - (x₂ - k₃))^2 + 
      (x₄ - k₂ - (x₂ - k₄))^2 + 
      (x₄ - k₃ - (x₃ - k₄))^2 ≤ b) → b ≥ 5/4) := by
  sorry

end min_value_a_l2869_286964


namespace equation_solution_l2869_286979

theorem equation_solution (y : ℝ) : (30 : ℝ) / 50 = Real.sqrt (y / 50) → y = 18 := by
  sorry

end equation_solution_l2869_286979


namespace triangle_abc_properties_l2869_286906

/-- Given a triangle ABC where BC = a and AC = b, and a and b are roots of x^2 - 2√3x + 2 = 0,
    prove that the measure of angle C is 2π/3 and the length of AB is √10 -/
theorem triangle_abc_properties (a b : ℝ) (A B C : ℝ) :
  a^2 - 2 * Real.sqrt 3 * a + 2 = 0 →
  b^2 - 2 * Real.sqrt 3 * b + 2 = 0 →
  2 * Real.cos (A + B) = 1 →
  C = 2 * π / 3 ∧ (a^2 + b^2 - 2 * a * b * Real.cos C) = 10 := by
  sorry


end triangle_abc_properties_l2869_286906


namespace expansion_without_x2_x3_terms_l2869_286944

theorem expansion_without_x2_x3_terms (m n : ℝ) : 
  (∀ x, (x^2 + m*x + 1) * (x^2 - 2*x + n) = x^4 + (m*n - 2)*x + n) → 
  m + n = 5 := by
sorry

end expansion_without_x2_x3_terms_l2869_286944


namespace fraction_sum_l2869_286952

theorem fraction_sum : (3 : ℚ) / 5 + 2 / 15 = 11 / 15 := by
  sorry

end fraction_sum_l2869_286952


namespace unique_solution_for_alpha_minus_one_l2869_286992

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the functional equation
def SatisfiesEquation (f : RealFunction) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y)) = f (x + y) + f x * f y + α * x * y

-- State the theorem
theorem unique_solution_for_alpha_minus_one (α : ℝ) (hα : α ≠ 0) :
  (∃ f : RealFunction, SatisfiesEquation f α) ↔ (α = -1 ∧ ∃ f : RealFunction, SatisfiesEquation f (-1) ∧ ∀ x, f x = x) :=
sorry

end unique_solution_for_alpha_minus_one_l2869_286992


namespace antique_shop_glass_price_l2869_286909

theorem antique_shop_glass_price :
  let num_dolls : ℕ := 3
  let num_clocks : ℕ := 2
  let num_glasses : ℕ := 5
  let doll_price : ℕ := 5
  let clock_price : ℕ := 15
  let total_cost : ℕ := 40
  let profit : ℕ := 25
  let total_revenue : ℕ := total_cost + profit
  let doll_revenue : ℕ := num_dolls * doll_price
  let clock_revenue : ℕ := num_clocks * clock_price
  let glass_revenue : ℕ := total_revenue - doll_revenue - clock_revenue
  glass_revenue / num_glasses = 4
  := by sorry

end antique_shop_glass_price_l2869_286909


namespace football_cost_l2869_286912

def total_cost : ℝ := 20.52
def marbles_cost : ℝ := 9.05
def baseball_cost : ℝ := 6.52

theorem football_cost :
  ∃ (football_cost : ℝ),
    football_cost = total_cost - marbles_cost - baseball_cost ∧
    football_cost = 5.45 := by
  sorry

end football_cost_l2869_286912


namespace multiply_mixed_number_l2869_286963

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end multiply_mixed_number_l2869_286963


namespace oliver_earnings_l2869_286975

def laundry_price : ℝ := 2
def day1_laundry : ℝ := 5
def day2_laundry : ℝ := day1_laundry + 5
def day3_laundry : ℝ := 2 * day2_laundry

def total_earnings : ℝ := laundry_price * (day1_laundry + day2_laundry + day3_laundry)

theorem oliver_earnings : total_earnings = 70 := by
  sorry

end oliver_earnings_l2869_286975


namespace line_relations_l2869_286946

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The first line l₁ -/
def l₁ (m : ℝ) : Line := { a := 1, b := m, c := 6 }

/-- The second line l₂ -/
def l₂ (m : ℝ) : Line := { a := m - 2, b := 3 * m, c := 18 }

/-- Two lines are parallel -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a ∧ l₁.a * l₂.c ≠ l₁.c * l₂.a

/-- Two lines are perpendicular -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

/-- Main theorem -/
theorem line_relations (m : ℝ) :
  (parallel (l₁ m) (l₂ m) ↔ m = 0) ∧
  (perpendicular (l₁ m) (l₂ m) ↔ m = -1 ∨ m = 2/3) := by
  sorry

end line_relations_l2869_286946


namespace student_distribution_proof_l2869_286974

def distribute_students (n : ℕ) (k : ℕ) (min_per_dorm : ℕ) : ℕ :=
  sorry

theorem student_distribution_proof :
  distribute_students 9 3 4 = 3570 :=
sorry

end student_distribution_proof_l2869_286974


namespace girls_without_notebooks_l2869_286928

theorem girls_without_notebooks (total_girls : ℕ) (total_boys : ℕ) 
  (notebooks_brought : ℕ) (boys_with_notebooks : ℕ) (girls_with_notebooks : ℕ) 
  (h1 : total_girls = 18)
  (h2 : total_boys = 20)
  (h3 : notebooks_brought = 30)
  (h4 : boys_with_notebooks = 17)
  (h5 : girls_with_notebooks = 11) :
  total_girls - girls_with_notebooks = 7 := by
sorry

end girls_without_notebooks_l2869_286928


namespace point_on_line_implies_fraction_value_l2869_286947

/-- If (m,7) lies on the graph of y = 3x + 1, then m²/(m-1) = 4 -/
theorem point_on_line_implies_fraction_value (m : ℝ) : 
  7 = 3 * m + 1 → m^2 / (m - 1) = 4 := by
  sorry

end point_on_line_implies_fraction_value_l2869_286947


namespace hyperbola_eccentricity_l2869_286957

/-- The eccentricity of a hyperbola passing through the focus of a parabola -/
theorem hyperbola_eccentricity (a : ℝ) (h_a : a > 0) :
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 = 1
  let parabola := fun (x y : ℝ) ↦ y^2 = 8 * x
  let focus : ℝ × ℝ := (2, 0)
  hyperbola focus.1 focus.2 →
  let c := Real.sqrt (a^2 + 1)
  c / a = Real.sqrt 5 / 2 := by
sorry

end hyperbola_eccentricity_l2869_286957
