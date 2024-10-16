import Mathlib

namespace NUMINAMATH_CALUDE_workmen_efficiency_ratio_l3551_355149

/-- Given two workmen with different efficiencies, prove their efficiency ratio -/
theorem workmen_efficiency_ratio 
  (combined_time : ℝ) 
  (b_alone_time : ℝ) 
  (ha : combined_time = 18) 
  (hb : b_alone_time = 54) : 
  (1 / combined_time - 1 / b_alone_time) / (1 / b_alone_time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_workmen_efficiency_ratio_l3551_355149


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l3551_355114

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, IsPerfectSquareTrinomial 4 m 121 → (m = 44 ∨ m = -44) :=
by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l3551_355114


namespace NUMINAMATH_CALUDE_pyramid_frustum_problem_l3551_355168

noncomputable def pyramid_frustum_theorem (AB BC height : ℝ) : Prop :=
  AB > 0 ∧ BC > 0 ∧ height > 0 →
  let ABCD := AB * BC
  let P_volume := (1/3) * ABCD * height
  let P'_volume := (1/8) * P_volume
  let F_height := height / 2
  let A'B' := AB / 2
  let B'C' := BC / 2
  let AC := Real.sqrt (AB^2 + BC^2)
  let A'C' := AC / 2
  let h := (73/8 : ℝ)
  let XT := h + F_height
  XT = 169/8

theorem pyramid_frustum_problem :
  pyramid_frustum_theorem 12 16 24 := by sorry

end NUMINAMATH_CALUDE_pyramid_frustum_problem_l3551_355168


namespace NUMINAMATH_CALUDE_inequality_proof_l3551_355108

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) :
  x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3551_355108


namespace NUMINAMATH_CALUDE_collinear_points_sum_l3551_355155

/-- Three points in 3D space are collinear if they all lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p3 = p1 + t • (p2 - p1) ∨ p2 = p1 + t • (p3 - p1) ∨ p1 = p2 + t • (p3 - p2)

/-- If the points (2,a,b), (a,3,b), and (a,b,4) are collinear, then 2a + b = 8. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → 2*a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l3551_355155


namespace NUMINAMATH_CALUDE_problem_solution_l3551_355124

theorem problem_solution : (2010^2 - 2010) / 2010^2 = 2009 / 2010 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3551_355124


namespace NUMINAMATH_CALUDE_quotient_reciprocal_sum_l3551_355183

theorem quotient_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hsum : x + y = 45) (hprod : x * y = 500) : 
  (x / y) + (y / x) = 41 / 20 := by
  sorry

end NUMINAMATH_CALUDE_quotient_reciprocal_sum_l3551_355183


namespace NUMINAMATH_CALUDE_parallelogram_area_l3551_355153

/-- Proves that the area of a parallelogram with base 7 and altitude twice the base is 98 square units --/
theorem parallelogram_area : ∀ (base altitude area : ℝ),
  base = 7 →
  altitude = 2 * base →
  area = base * altitude →
  area = 98 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3551_355153


namespace NUMINAMATH_CALUDE_two_x_eq_zero_is_linear_l3551_355148

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function representing the equation 2x = 0 -/
def f (x : ℝ) : ℝ := 2 * x

/-- Theorem stating that 2x = 0 is a linear equation -/
theorem two_x_eq_zero_is_linear : is_linear_equation f := by
  sorry

end NUMINAMATH_CALUDE_two_x_eq_zero_is_linear_l3551_355148


namespace NUMINAMATH_CALUDE_music_talent_sample_l3551_355137

/-- Represents the number of students selected in a stratified sampling -/
def stratified_sample (total_population : ℕ) (group_size : ℕ) (sample_size : ℕ) : ℕ :=
  (group_size * sample_size) / total_population

/-- Proves that in a stratified sampling of 40 students from a population of 100 students,
    where 40 students have music talent, the number of music-talented students selected is 16 -/
theorem music_talent_sample :
  stratified_sample 100 40 40 = 16 := by
  sorry

end NUMINAMATH_CALUDE_music_talent_sample_l3551_355137


namespace NUMINAMATH_CALUDE_jar_capacity_l3551_355161

/-- Proves that the capacity of each jar James needs to buy is 0.5 liters -/
theorem jar_capacity
  (num_hives : ℕ)
  (honey_per_hive : ℝ)
  (num_jars : ℕ)
  (h1 : num_hives = 5)
  (h2 : honey_per_hive = 20)
  (h3 : num_jars = 100)
  : (num_hives * honey_per_hive / 2) / num_jars = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_jar_capacity_l3551_355161


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3551_355178

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l3551_355178


namespace NUMINAMATH_CALUDE_trip_theorem_l3551_355122

/-- Represents the ticket prices and group sizes for a school trip -/
structure TripData where
  adultPrice : ℕ
  studentDiscount : ℚ
  groupDiscount : ℚ
  totalPeople : ℕ
  totalCost : ℕ

/-- Calculates the number of adults and students in the group -/
def calculateGroup (data : TripData) : ℕ × ℕ :=
  sorry

/-- Calculates the cost of tickets for different purchasing strategies -/
def calculateCosts (data : TripData) (adults : ℕ) (students : ℕ) : ℕ × ℕ × ℕ :=
  sorry

/-- Theorem stating the correct number of adults and students, and the most cost-effective purchasing strategy -/
theorem trip_theorem (data : TripData) 
  (h1 : data.adultPrice = 120)
  (h2 : data.studentDiscount = 1/2)
  (h3 : data.groupDiscount = 3/5)
  (h4 : data.totalPeople = 130)
  (h5 : data.totalCost = 9600) :
  let (adults, students) := calculateGroup data
  let (regularCost, allGroupCost, mixedCost) := calculateCosts data adults students
  adults = 30 ∧ 
  students = 100 ∧ 
  mixedCost < allGroupCost ∧
  mixedCost < regularCost :=
sorry

end NUMINAMATH_CALUDE_trip_theorem_l3551_355122


namespace NUMINAMATH_CALUDE_number_line_segment_sum_l3551_355107

theorem number_line_segment_sum : 
  ∀ (P V : ℝ) (Q R S T U : ℝ),
  P = 3 →
  V = 33 →
  Q - P = R - Q → R - Q = S - R → S - R = T - S → T - S = U - T → U - T = V - U →
  (S - P) + (V - T) = 25 := by
sorry

end NUMINAMATH_CALUDE_number_line_segment_sum_l3551_355107


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l3551_355138

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℚ, x - 2*y = 0 ∧ 3*x + 2*y = 8 ∧ x = 2 ∧ y = 1) ∧
  -- System 2
  (∃ x y : ℚ, 3*x - 5*y = 9 ∧ 2*x + 3*y = -6 ∧ x = -3/19 ∧ y = -36/19) := by
  sorry


end NUMINAMATH_CALUDE_system_of_equations_solutions_l3551_355138


namespace NUMINAMATH_CALUDE_cat_food_bags_l3551_355105

theorem cat_food_bags (cat_food_weight : ℕ) (dog_food_bags : ℕ) (weight_difference : ℕ) (ounces_per_pound : ℕ) (total_ounces : ℕ) : 
  cat_food_weight = 3 →
  dog_food_bags = 2 →
  weight_difference = 2 →
  ounces_per_pound = 16 →
  total_ounces = 256 →
  ∃ (x : ℕ), x * cat_food_weight * ounces_per_pound + 
    dog_food_bags * (cat_food_weight + weight_difference) * ounces_per_pound = total_ounces ∧ 
    x = 2 :=
by sorry

end NUMINAMATH_CALUDE_cat_food_bags_l3551_355105


namespace NUMINAMATH_CALUDE_duty_arrangement_for_three_leaders_l3551_355184

/-- The number of ways to arrange n leaders for duty over d days, 
    with each leader on duty for m days. -/
def dutyArrangements (n d m : ℕ) : ℕ := sorry

/-- The number of combinations of n items taken k at a time. -/
def nCk (n k : ℕ) : ℕ := sorry

theorem duty_arrangement_for_three_leaders :
  dutyArrangements 3 6 2 = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_duty_arrangement_for_three_leaders_l3551_355184


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l3551_355118

/-- The equation (x+y)^2 = x^2 + y^2 + 4 represents a hyperbola in the xy-plane. -/
theorem equation_represents_hyperbola :
  ∃ (f : ℝ → ℝ → Prop), (∀ x y : ℝ, f x y ↔ (x + y)^2 = x^2 + y^2 + 4) ∧
  (∃ a : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, f x y ↔ x * y = a) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l3551_355118


namespace NUMINAMATH_CALUDE_closest_integer_to_ten_minus_sqrt_thirteen_l3551_355129

theorem closest_integer_to_ten_minus_sqrt_thirteen :
  let sqrt_13 : ℝ := Real.sqrt 13
  ∀ n : ℤ, n ∈ ({4, 5, 7} : Set ℤ) →
    3 < sqrt_13 ∧ sqrt_13 < 4 →
    |10 - sqrt_13 - 6| < |10 - sqrt_13 - ↑n| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_ten_minus_sqrt_thirteen_l3551_355129


namespace NUMINAMATH_CALUDE_largest_number_l3551_355144

theorem largest_number (a b c d e : ℕ) :
  a = 30^20 ∧
  b = 10^30 ∧
  c = 30^10 + 20^20 ∧
  d = (30 + 10)^20 ∧
  e = (30 * 20)^10 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l3551_355144


namespace NUMINAMATH_CALUDE_cube_root_equality_l3551_355187

theorem cube_root_equality (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (m^4 * n^4)^(1/3) = (m * n)^(4/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_equality_l3551_355187


namespace NUMINAMATH_CALUDE_factor_expression_l3551_355132

theorem factor_expression (c : ℝ) : 189 * c^2 + 27 * c - 36 = 9 * (3 * c - 1) * (7 * c + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3551_355132


namespace NUMINAMATH_CALUDE_min_value_fractional_sum_l3551_355189

theorem min_value_fractional_sum (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^3 / (y - 2)) + (y^3 / (x - 2)) ≥ 96 ∧
  ((x^3 / (y - 2)) + (y^3 / (x - 2)) = 96 ↔ x = 4 ∧ y = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_fractional_sum_l3551_355189


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3551_355180

/-- Represents a repeating decimal where the digit repeats indefinitely after the decimal point. -/
def RepeatingDecimal (digit : ℕ) : ℚ :=
  (digit : ℚ) / 9

/-- The sum of 0.4444... and 0.7777... is equal to 11/9. -/
theorem repeating_decimal_sum :
  RepeatingDecimal 4 + RepeatingDecimal 7 = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3551_355180


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3551_355191

theorem lcm_of_ratio_and_hcf (a b : ℕ) (h_ratio : a * 3 = b * 2) (h_hcf : Nat.gcd a b = 6) : Nat.lcm a b = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3551_355191


namespace NUMINAMATH_CALUDE_triangle_theorem_l3551_355195

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  t.c * Real.sin t.B + 2 * Real.cos t.A = t.b * Real.sin t.C + 1

def condition2 (t : Triangle) : Prop :=
  Real.cos (2 * t.A) - 3 * Real.cos (t.B + t.C) - 1 = 0

def condition3 (t : Triangle) : Prop :=
  ∃ k : ℝ, k * Real.sqrt 3 * t.b = t.a * Real.sin t.B ∧ k * t.a = Real.cos t.A

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h : condition1 t ∨ condition2 t ∨ condition3 t) : 
  t.A = Real.pi / 3 ∧ 
  (t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3 / 2 → t.a ≥ Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3551_355195


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l3551_355154

theorem complex_magnitude_one (r : ℝ) (z : ℂ) (h1 : |r| < 3) (h2 : z + 1/z = r) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l3551_355154


namespace NUMINAMATH_CALUDE_sarah_flour_total_l3551_355158

/-- The total amount of flour Sarah has -/
def total_flour (rye whole_wheat chickpea pastry : ℕ) : ℕ :=
  rye + whole_wheat + chickpea + pastry

/-- Theorem: Sarah has 20 pounds of flour in total -/
theorem sarah_flour_total :
  total_flour 5 10 3 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sarah_flour_total_l3551_355158


namespace NUMINAMATH_CALUDE_exists_continuous_surjective_non_monotonic_l3551_355194

/-- A continuous function from ℝ to ℝ with full range that is not monotonic -/
theorem exists_continuous_surjective_non_monotonic :
  ∃ f : ℝ → ℝ, Continuous f ∧ Function.Surjective f ∧ ¬Monotone f := by
  sorry

end NUMINAMATH_CALUDE_exists_continuous_surjective_non_monotonic_l3551_355194


namespace NUMINAMATH_CALUDE_pig_count_l3551_355125

theorem pig_count (P H : ℕ) : 
  4 * P + 2 * H = 2 * (P + H) + 22 → P = 11 := by
sorry

end NUMINAMATH_CALUDE_pig_count_l3551_355125


namespace NUMINAMATH_CALUDE_algorithm_swaps_values_l3551_355111

-- Define the algorithm steps
def algorithm (x y : ℝ) : ℝ × ℝ :=
  let z := x
  let x' := y
  let y' := z
  (x', y')

-- Theorem statement
theorem algorithm_swaps_values (x y : ℝ) :
  algorithm x y = (y, x) := by sorry

end NUMINAMATH_CALUDE_algorithm_swaps_values_l3551_355111


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3551_355175

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3551_355175


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3551_355193

theorem trigonometric_simplification (α : ℝ) :
  (Real.sin (π - α) / Real.cos (π + α)) *
  (Real.cos (-α) * Real.cos (2*π - α)) /
  Real.sin (π/2 + α) = -Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3551_355193


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3551_355160

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3551_355160


namespace NUMINAMATH_CALUDE_simplify_expression_l3551_355113

theorem simplify_expression (x : ℝ) : 2*x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -4*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3551_355113


namespace NUMINAMATH_CALUDE_triangle_theorem_l3551_355119

theorem triangle_theorem (a b c A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = π) (h5 : 2 * c * Real.cos C + b * Real.cos A + a * Real.cos B = 0) :
  C = 2 * π / 3 ∧ 
  (c = 3 → A = π / 6 → (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3551_355119


namespace NUMINAMATH_CALUDE_parallel_planes_from_skew_lines_l3551_355162

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (planeParallel : Plane → Plane → Prop)

-- Define the skew relation between lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_from_skew_lines 
  (m n : Line) (α β : Plane) 
  (h_skew : skew m n) 
  (h_m_α : parallel m α) (h_n_α : parallel n α) 
  (h_m_β : parallel m β) (h_n_β : parallel n β) : 
  planeParallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_skew_lines_l3551_355162


namespace NUMINAMATH_CALUDE_boys_usual_time_to_school_l3551_355157

/-- 
Given a boy who reaches school 4 minutes early when walking at 9/8 of his usual rate,
prove that his usual time to reach the school is 36 minutes.
-/
theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) 
  (h1 : usual_rate > 0) 
  (h2 : usual_time > 0)
  (h3 : usual_rate * usual_time = (9/8 * usual_rate) * (usual_time - 4)) : 
  usual_time = 36 := by
  sorry

end NUMINAMATH_CALUDE_boys_usual_time_to_school_l3551_355157


namespace NUMINAMATH_CALUDE_terms_before_negative17_l3551_355182

/-- An arithmetic sequence with first term 103 and common difference -7 -/
def arithmeticSequence (n : ℕ) : ℤ := 103 - 7 * (n - 1)

/-- The position of -17 in the sequence -/
def positionOfNegative17 : ℕ := 18

theorem terms_before_negative17 :
  (∀ k < positionOfNegative17 - 1, arithmeticSequence k > -17) ∧
  arithmeticSequence positionOfNegative17 = -17 :=
sorry

end NUMINAMATH_CALUDE_terms_before_negative17_l3551_355182


namespace NUMINAMATH_CALUDE_book_cost_price_l3551_355102

/-- The cost price of a book satisfying certain profit conditions -/
def cost_price : ℝ := 2000

/-- The selling price of the book at 10% profit -/
def selling_price_10 : ℝ := cost_price * 1.1

/-- The selling price of the book at 15% profit -/
def selling_price_15 : ℝ := cost_price * 1.15

/-- Theorem stating the cost price of the book given the profit conditions -/
theorem book_cost_price : 
  (selling_price_10 + 100 = selling_price_15) → 
  cost_price = 2000 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l3551_355102


namespace NUMINAMATH_CALUDE_circle_area_theorem_l3551_355142

theorem circle_area_theorem (r : ℝ) (h : 3 * (1 / (2 * Real.pi * r)) = r) : 
  Real.pi * r^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l3551_355142


namespace NUMINAMATH_CALUDE_inequality_proof_l3551_355130

theorem inequality_proof (x y z : ℝ) (h : x + y + z = x*y + y*z + z*x) :
  x / Real.sqrt (x^4 + x^2 + 1) + y / Real.sqrt (y^4 + y^2 + 1) + z / Real.sqrt (z^4 + z^2 + 1) ≥ -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3551_355130


namespace NUMINAMATH_CALUDE_softball_players_count_l3551_355147

theorem softball_players_count (cricket hockey football total : ℕ) 
  (h1 : cricket = 22)
  (h2 : hockey = 15)
  (h3 : football = 21)
  (h4 : total = 77) :
  total - (cricket + hockey + football) = 19 := by
  sorry

end NUMINAMATH_CALUDE_softball_players_count_l3551_355147


namespace NUMINAMATH_CALUDE_one_third_of_one_fourth_l3551_355136

theorem one_third_of_one_fourth (n : ℝ) : (3 / 10 : ℝ) * n = 64.8 → (1 / 3 : ℝ) * (1 / 4 : ℝ) * n = 18 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_one_fourth_l3551_355136


namespace NUMINAMATH_CALUDE_sum_three_not_all_less_than_one_l3551_355134

theorem sum_three_not_all_less_than_one (a b c : ℝ) (h : a + b + c = 3) :
  ¬(a < 1 ∧ b < 1 ∧ c < 1) := by sorry

end NUMINAMATH_CALUDE_sum_three_not_all_less_than_one_l3551_355134


namespace NUMINAMATH_CALUDE_cubic_equation_complex_root_l3551_355139

theorem cubic_equation_complex_root (k : ℝ) : 
  (∃ z : ℂ, z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  k = 2 ∨ k = -2/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_complex_root_l3551_355139


namespace NUMINAMATH_CALUDE_stanley_tires_l3551_355128

/-- The number of tires Stanley bought -/
def num_tires : ℕ := 240 / 60

/-- The cost of each tire in dollars -/
def cost_per_tire : ℕ := 60

/-- The total amount Stanley spent in dollars -/
def total_spent : ℕ := 240

theorem stanley_tires :
  num_tires = 4 ∧ cost_per_tire * num_tires = total_spent :=
sorry

end NUMINAMATH_CALUDE_stanley_tires_l3551_355128


namespace NUMINAMATH_CALUDE_sum_of_ages_l3551_355116

/-- Represents the ages of two people P and Q -/
structure Ages where
  p : ℝ
  q : ℝ

/-- The condition that P's age is thrice Q's age when P was as old as Q is now -/
def age_relation (ages : Ages) : Prop :=
  ages.p = 3 * (ages.q - (ages.p - ages.q))

/-- Theorem stating the sum of P and Q's ages given the conditions -/
theorem sum_of_ages :
  ∀ (ages : Ages),
    ages.q = 37.5 →
    age_relation ages →
    ages.p + ages.q = 93.75 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_ages_l3551_355116


namespace NUMINAMATH_CALUDE_problem_statement_l3551_355103

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem problem_statement (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
    (h2 : f (-2) = 10) : 
  f 2 = -26 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3551_355103


namespace NUMINAMATH_CALUDE_middle_card_is_five_l3551_355104

/-- Represents a set of three cards with distinct positive integers. -/
structure CardSet where
  left : ℕ+
  middle : ℕ+
  right : ℕ+
  distinct : left ≠ middle ∧ middle ≠ right ∧ left ≠ right
  ascending : left < middle ∧ middle < right
  sum_15 : left + middle + right = 15

/-- Predicate for Ada's statement about the leftmost card -/
def ada_statement (cs : CardSet) : Prop :=
  ∃ cs' : CardSet, cs'.left = cs.left ∧ cs' ≠ cs

/-- Predicate for Bella's statement about the rightmost card -/
def bella_statement (cs : CardSet) : Prop :=
  ∃ cs' : CardSet, cs'.right = cs.right ∧ cs' ≠ cs

/-- The main theorem stating that the middle card must be 5 -/
theorem middle_card_is_five :
  ∀ cs : CardSet,
    ada_statement cs →
    bella_statement cs →
    cs.middle = 5 :=
sorry

end NUMINAMATH_CALUDE_middle_card_is_five_l3551_355104


namespace NUMINAMATH_CALUDE_unique_N_leads_to_five_l3551_355198

def machine_rule (N : ℕ) : ℕ :=
  if N % 2 = 1 then 2 * N + 2 else N / 2 + 1

def apply_rule_n_times (N : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => N
  | m + 1 => machine_rule (apply_rule_n_times N m)

theorem unique_N_leads_to_five : ∃! N : ℕ, N > 0 ∧ apply_rule_n_times N 6 = 5 ∧ N = 66 := by
  sorry

end NUMINAMATH_CALUDE_unique_N_leads_to_five_l3551_355198


namespace NUMINAMATH_CALUDE_exam_score_problem_l3551_355169

theorem exam_score_problem (total_questions : ℕ) (correct_marks : ℕ) (wrong_marks : ℕ) (total_score : ℤ) :
  total_questions = 100 →
  correct_marks = 5 →
  wrong_marks = 2 →
  total_score = 210 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    (correct_marks * correct_answers : ℤ) - (wrong_marks * (total_questions - correct_answers) : ℤ) = total_score ∧
    correct_answers = 58 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3551_355169


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l3551_355150

theorem students_taking_one_subject (total_students : ℕ) 
  (algebra_and_drafting : ℕ) (algebra_total : ℕ) (only_drafting : ℕ) 
  (neither_subject : ℕ) :
  algebra_and_drafting = 22 →
  algebra_total = 40 →
  only_drafting = 15 →
  neither_subject = 8 →
  total_students = algebra_total + only_drafting + neither_subject →
  (algebra_total - algebra_and_drafting) + only_drafting = 33 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l3551_355150


namespace NUMINAMATH_CALUDE_zoe_total_earnings_l3551_355188

/-- Represents Zoe's babysitting and pool cleaning earnings -/
structure ZoeEarnings where
  zachary_sessions : ℕ
  julie_sessions : ℕ
  chloe_sessions : ℕ
  zachary_earnings : ℕ
  pool_cleaning_earnings : ℕ

/-- Calculates Zoe's total earnings -/
def total_earnings (e : ZoeEarnings) : ℕ :=
  e.zachary_earnings + e.pool_cleaning_earnings

/-- Theorem stating that Zoe's total earnings are $3200 -/
theorem zoe_total_earnings (e : ZoeEarnings) 
  (h1 : e.julie_sessions = 3 * e.zachary_sessions)
  (h2 : e.zachary_sessions = e.chloe_sessions / 5)
  (h3 : e.zachary_earnings = 600)
  (h4 : e.pool_cleaning_earnings = 2600) : 
  total_earnings e = 3200 := by
  sorry


end NUMINAMATH_CALUDE_zoe_total_earnings_l3551_355188


namespace NUMINAMATH_CALUDE_smallest_Y_for_binary_multiple_of_15_l3551_355101

def is_binary_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_Y_for_binary_multiple_of_15 :
  (∃ U : ℕ, is_binary_number U ∧ U % 15 = 0 ∧ U = 15 * 74) ∧
  (∀ Y : ℕ, Y < 74 → ¬∃ U : ℕ, is_binary_number U ∧ U % 15 = 0 ∧ U = 15 * Y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_Y_for_binary_multiple_of_15_l3551_355101


namespace NUMINAMATH_CALUDE_expression_value_l3551_355151

theorem expression_value : ((2525 - 2424)^2 + 100) / 225 = 46 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3551_355151


namespace NUMINAMATH_CALUDE_triangle_shape_l3551_355185

theorem triangle_shape (a b c : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h3 : a^2 * c^2 + b^2 * c^2 = a^4 - b^4) : 
  a^2 = b^2 + c^2 := by sorry

end NUMINAMATH_CALUDE_triangle_shape_l3551_355185


namespace NUMINAMATH_CALUDE_monotonic_h_implies_a_leq_neg_one_l3551_355174

/-- Given functions f and g, prove that if h is monotonically increasing on [1,4],
    then a ≤ -1 -/
theorem monotonic_h_implies_a_leq_neg_one (a : ℝ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ Real.log x
  let g : ℝ → ℝ := λ x ↦ (1/2) * a * x^2 + 2*x
  let h : ℝ → ℝ := λ x ↦ f x - g x
  (∀ x ∈ Set.Icc 1 4, Monotone h) →
  a ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_monotonic_h_implies_a_leq_neg_one_l3551_355174


namespace NUMINAMATH_CALUDE_number_satisfying_condition_l3551_355115

theorem number_satisfying_condition : ∃ x : ℤ, (x - 29) / 13 = 15 ∧ x = 224 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l3551_355115


namespace NUMINAMATH_CALUDE_dog_food_cans_per_package_adam_dog_food_cans_l3551_355127

theorem dog_food_cans_per_package (cat_packages : Nat) (dog_packages : Nat) 
  (cat_cans_per_package : Nat) (extra_cat_cans : Nat) : Nat :=
  let total_cat_cans := cat_packages * cat_cans_per_package
  let dog_cans_per_package := (total_cat_cans - extra_cat_cans) / dog_packages
  dog_cans_per_package

/-- The number of cans in each package of dog food is 5. -/
theorem adam_dog_food_cans : dog_food_cans_per_package 9 7 10 55 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_cans_per_package_adam_dog_food_cans_l3551_355127


namespace NUMINAMATH_CALUDE_square_root_of_a_minus_b_l3551_355121

theorem square_root_of_a_minus_b (a b : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (a + 3)^2 = x ∧ (2*a - 6)^2 = x) →
  (b = -8) →
  Real.sqrt (a - b) = 3 := by sorry

end NUMINAMATH_CALUDE_square_root_of_a_minus_b_l3551_355121


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3551_355164

-- System 1
theorem system_one_solution (x : ℝ) : 
  (2 * x > 1 - x ∧ x + 2 < 4 * x - 1) ↔ x > 1 :=
sorry

-- System 2
theorem system_two_solution (x : ℝ) : 
  ((2 / 3) * x + 5 > 1 - x ∧ x - 1 ≤ (3 / 4) * x - (1 / 8)) ↔ 
  (-12 / 5 < x ∧ x ≤ 7 / 2) :=
sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3551_355164


namespace NUMINAMATH_CALUDE_willy_stuffed_animals_l3551_355141

def stuffed_animals_total (initial : ℕ) (mom_gift : ℕ) (dad_multiplier : ℕ) : ℕ :=
  let after_mom := initial + mom_gift
  let dad_gift := after_mom * dad_multiplier
  after_mom + dad_gift

theorem willy_stuffed_animals :
  stuffed_animals_total 10 2 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_willy_stuffed_animals_l3551_355141


namespace NUMINAMATH_CALUDE_basketball_height_data_field_survey_l3551_355126

def HeightData := List Nat

def isFieldSurveyMethod (data : HeightData) : Prop :=
  data.all (λ h => h ≥ 150 ∧ h ≤ 200) ∧ 
  data.length > 0 ∧
  data.length ≤ 20

def basketballTeamHeights : HeightData :=
  [167, 168, 167, 164, 168, 168, 163, 168, 167, 160]

theorem basketball_height_data_field_survey :
  isFieldSurveyMethod basketballTeamHeights := by
  sorry

end NUMINAMATH_CALUDE_basketball_height_data_field_survey_l3551_355126


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3551_355179

theorem arithmetic_sequence_length : 
  ∀ (a d last : ℕ), 
  a = 3 → d = 3 → last = 198 → 
  ∃ n : ℕ, n = 66 ∧ last = a + (n - 1) * d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3551_355179


namespace NUMINAMATH_CALUDE_monkey_banana_theorem_l3551_355159

/-- Represents the monkey's banana transportation problem -/
structure BananaProblem where
  total_bananas : ℕ
  distance : ℕ
  max_carry : ℕ
  eat_rate : ℕ

/-- Calculates the maximum number of bananas the monkey can bring home -/
def max_bananas_home (problem : BananaProblem) : ℕ :=
  sorry

/-- Theorem stating that for the given problem, the maximum number of bananas brought home is 25 -/
theorem monkey_banana_theorem (problem : BananaProblem) 
  (h1 : problem.total_bananas = 100)
  (h2 : problem.distance = 50)
  (h3 : problem.max_carry = 50)
  (h4 : problem.eat_rate = 1) :
  max_bananas_home problem = 25 := by
  sorry

end NUMINAMATH_CALUDE_monkey_banana_theorem_l3551_355159


namespace NUMINAMATH_CALUDE_smallest_item_is_a5_l3551_355135

def sequence_a (n : ℕ) : ℚ :=
  2 * n^2 - 21 * n + 40

theorem smallest_item_is_a5 :
  ∀ n : ℕ, n ≥ 1 → sequence_a 5 ≤ sequence_a n :=
sorry

end NUMINAMATH_CALUDE_smallest_item_is_a5_l3551_355135


namespace NUMINAMATH_CALUDE_hockey_league_games_l3551_355140

theorem hockey_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 16) (h2 : total_games = 1200) :
  ∃ x : ℕ, x * n * (n - 1) / 2 = total_games ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l3551_355140


namespace NUMINAMATH_CALUDE_sum_of_selected_flowerbeds_l3551_355133

/-- The number of seeds in each flowerbed -/
def seeds : Fin 9 → ℕ
  | 0 => 18  -- 1st flowerbed
  | 1 => 22  -- 2nd flowerbed
  | 2 => 30  -- 3rd flowerbed
  | 3 => 2 * seeds 0  -- 4th flowerbed
  | 4 => seeds 2  -- 5th flowerbed
  | 5 => seeds 1 / 2  -- 6th flowerbed
  | 6 => seeds 0  -- 7th flowerbed
  | 7 => seeds 3  -- 8th flowerbed
  | 8 => seeds 2 - 1  -- 9th flowerbed

theorem sum_of_selected_flowerbeds : seeds 0 + seeds 4 + seeds 8 = 77 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_selected_flowerbeds_l3551_355133


namespace NUMINAMATH_CALUDE_max_value_of_t_l3551_355165

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def t (m : ℝ) : ℝ := (2 * m + log m / m - m * log m) / 2

theorem max_value_of_t :
  ∃ (m : ℝ), m > 1 ∧ ∀ (x : ℝ), x > 1 → t x ≤ t m ∧ t m = (exp 2 + 1) / (2 * exp 1) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_t_l3551_355165


namespace NUMINAMATH_CALUDE_smallest_number_proof_l3551_355109

/-- The smallest natural number that is divisible by 55 and has exactly 117 distinct divisors -/
def smallest_number : ℕ := 12390400

/-- Count the number of distinct divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

theorem smallest_number_proof :
  smallest_number % 55 = 0 ∧
  count_divisors smallest_number = 117 ∧
  ∀ m : ℕ, m < smallest_number → (m % 55 = 0 ∧ count_divisors m = 117) → False :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l3551_355109


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l3551_355117

theorem unique_x_with_three_prime_divisors (x n : ℕ) : 
  x = 8^n - 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 31 ∧ q ≠ 31 ∧ x = 31 * p * q) →
  (∀ r : ℕ, Prime r ∧ r ∣ x → r = 31 ∨ r = p ∨ r = q) →
  x = 32767 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l3551_355117


namespace NUMINAMATH_CALUDE_even_quadratic_function_sum_l3551_355100

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_quadratic_function_sum (a b : ℝ) :
  let f := fun x => a * x^2 + b * x
  IsEven f ∧ (∀ x ∈ Set.Icc (a - 1) (2 * a), f x ∈ Set.range f) →
  a + b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_function_sum_l3551_355100


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l3551_355197

def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

theorem derivative_f_at_2 : 
  deriv f 2 = 15 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l3551_355197


namespace NUMINAMATH_CALUDE_paint_house_time_l3551_355196

/-- Represents the time taken to paint a house given the number of workers and their efficiency -/
def paintTime (workers : ℕ) (efficiency : ℚ) (time : ℚ) : Prop :=
  (workers : ℚ) * efficiency * time = 40

theorem paint_house_time :
  paintTime 5 (4/5) 8 → paintTime 4 (4/5) 10 := by sorry

end NUMINAMATH_CALUDE_paint_house_time_l3551_355196


namespace NUMINAMATH_CALUDE_second_worker_time_l3551_355176

-- Define the time it takes for the first worker to load the truck
def worker1_time : ℝ := 6

-- Define the time it takes for both workers to load the truck together
def combined_time : ℝ := 3.428571428571429

-- Define the time it takes for the second worker to load the truck
def worker2_time : ℝ := 8

-- Theorem statement
theorem second_worker_time : 
  (1 / worker1_time + 1 / worker2_time = 1 / combined_time) → 
  worker2_time = 8 := by
sorry

end NUMINAMATH_CALUDE_second_worker_time_l3551_355176


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l3551_355143

theorem tutor_schedule_lcm : Nat.lcm (Nat.lcm (Nat.lcm 4 5) 6) 8 = 120 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l3551_355143


namespace NUMINAMATH_CALUDE_inverse_inequality_l3551_355120

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l3551_355120


namespace NUMINAMATH_CALUDE_min_value_abs_sum_min_value_achievable_l3551_355163

theorem min_value_abs_sum (x : ℝ) : |x - 1| + |x - 4| ≥ 3 := by sorry

theorem min_value_achievable : ∃ x : ℝ, |x - 1| + |x - 4| = 3 := by sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_min_value_achievable_l3551_355163


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l3551_355170

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Theorem statement
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l3551_355170


namespace NUMINAMATH_CALUDE_sine_fraction_simplification_l3551_355167

theorem sine_fraction_simplification (b : Real) (h : b = 2 * Real.pi / 13) :
  (Real.sin (4 * b) * Real.sin (8 * b) * Real.sin (10 * b) * Real.sin (12 * b) * Real.sin (14 * b)) /
  (Real.sin b * Real.sin (2 * b) * Real.sin (4 * b) * Real.sin (6 * b) * Real.sin (10 * b)) =
  Real.sin (10 * Real.pi / 13) / Real.sin (4 * Real.pi / 13) := by
  sorry

end NUMINAMATH_CALUDE_sine_fraction_simplification_l3551_355167


namespace NUMINAMATH_CALUDE_automobile_distance_l3551_355145

/-- Proves the total distance traveled by an automobile given specific conditions -/
theorem automobile_distance (a r : ℝ) : 
  let first_half_distance : ℝ := a / 4
  let first_half_time : ℝ := 2 * r
  let first_half_speed : ℝ := first_half_distance / first_half_time
  let second_half_speed : ℝ := 2 * first_half_speed
  let second_half_time : ℝ := 2 * 60 -- 2 minutes in seconds
  let second_half_distance : ℝ := second_half_speed * second_half_time
  let total_distance_feet : ℝ := first_half_distance + second_half_distance
  let total_distance_yards : ℝ := total_distance_feet / 3
  total_distance_yards = 121 * a / 12 :=
by sorry

end NUMINAMATH_CALUDE_automobile_distance_l3551_355145


namespace NUMINAMATH_CALUDE_prob_D_is_one_fourth_l3551_355171

/-- A spinner with four regions -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)

/-- The properties of our specific spinner -/
def spinner : Spinner :=
  { probA := 1/4
  , probB := 1/3
  , probC := 1/6
  , probD := 1/4 }

/-- The sum of probabilities in a spinner must equal 1 -/
axiom probability_sum (s : Spinner) : s.probA + s.probB + s.probC + s.probD = 1

/-- Theorem: Given the probabilities of A, B, and C, the probability of D is 1/4 -/
theorem prob_D_is_one_fourth :
  spinner.probA = 1/4 → spinner.probB = 1/3 → spinner.probC = 1/6 →
  spinner.probD = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_D_is_one_fourth_l3551_355171


namespace NUMINAMATH_CALUDE_complex_fraction_problem_l3551_355186

theorem complex_fraction_problem (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 4) :
  (x^5 + y^5) / (x^5 - y^5) + (x^5 - y^5) / (x^5 + y^5) = 130 / 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_problem_l3551_355186


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l3551_355106

/-- Given a rectangle with dimensions 12 inches by 10 inches and an overlapping
    rectangle of 4 inches by 3 inches, if the shaded area is 130 square inches,
    then the perimeter of the non-shaded region is 7 1/3 inches. -/
theorem non_shaded_perimeter (shaded_area : ℝ) : 
  shaded_area = 130 → 
  (12 * 10 + 4 * 3 - shaded_area) / (12 - 4) * 2 + (12 - 4) * 2 = 22 / 3 :=
by sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l3551_355106


namespace NUMINAMATH_CALUDE_seedlings_per_packet_l3551_355177

theorem seedlings_per_packet (total_seedlings : ℕ) (num_packets : ℕ) 
  (h1 : total_seedlings = 420) (h2 : num_packets = 60) :
  total_seedlings / num_packets = 7 := by
  sorry

end NUMINAMATH_CALUDE_seedlings_per_packet_l3551_355177


namespace NUMINAMATH_CALUDE_yellow_paint_calculation_l3551_355190

/-- Given a ratio of red:yellow:blue paint and the amount of blue paint,
    calculate the amount of yellow paint required. -/
def yellow_paint_amount (red yellow blue : ℚ) (blue_amount : ℚ) : ℚ :=
  (yellow / blue) * blue_amount

/-- Prove that for the given ratio and blue paint amount, 
    the required yellow paint amount is 9 quarts. -/
theorem yellow_paint_calculation :
  let red : ℚ := 5
  let yellow : ℚ := 3
  let blue : ℚ := 7
  let blue_amount : ℚ := 21
  yellow_paint_amount red yellow blue blue_amount = 9 := by
  sorry

#eval yellow_paint_amount 5 3 7 21

end NUMINAMATH_CALUDE_yellow_paint_calculation_l3551_355190


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3551_355156

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_third : a 3 = 7)
  (h_sixth : a 6 = 16) :
  a 9 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3551_355156


namespace NUMINAMATH_CALUDE_pure_imaginary_square_root_l3551_355131

theorem pure_imaginary_square_root (a : ℝ) : 
  let z : ℂ := (a - Complex.I) ^ 2
  (∃ (b : ℝ), z = Complex.I * b) → (a = 1 ∨ a = -1) := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_root_l3551_355131


namespace NUMINAMATH_CALUDE_last_two_digits_2005_pow_base_3_representation_l3551_355173

-- Define the expression
def big_exp : ℕ := 2003^2004 + 3

-- Define the function to calculate the last two digits in base 3
def last_two_digits_base_3 (n : ℕ) : ℕ := n % 9

-- Theorem statement
theorem last_two_digits_2005_pow : last_two_digits_base_3 (2005^big_exp) = 4 := by
  sorry

-- Convert to base 3
theorem base_3_representation : (last_two_digits_base_3 (2005^big_exp)).digits 3 = [1, 1] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_2005_pow_base_3_representation_l3551_355173


namespace NUMINAMATH_CALUDE_correct_multiplication_result_l3551_355172

theorem correct_multiplication_result (a : ℕ) : 
  (153 * a ≠ 102325 ∧ 153 * a < 102357 ∧ 102357 - 153 * a < 153) → 
  153 * a = 102357 :=
by sorry

end NUMINAMATH_CALUDE_correct_multiplication_result_l3551_355172


namespace NUMINAMATH_CALUDE_hexagon_pentagon_angle_sum_l3551_355192

theorem hexagon_pentagon_angle_sum : 
  let hexagon_angle := 180 * (6 - 2) / 6
  let pentagon_angle := 180 * (5 - 2) / 5
  hexagon_angle + pentagon_angle = 228 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_pentagon_angle_sum_l3551_355192


namespace NUMINAMATH_CALUDE_ellipse_m_range_l3551_355146

/-- The equation of an ellipse in terms of m -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m + 1) = 1

/-- The range of m for which the equation represents an ellipse -/
def m_range (m : ℝ) : Prop :=
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1)

theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m → m_range m :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l3551_355146


namespace NUMINAMATH_CALUDE_number_thought_of_l3551_355181

theorem number_thought_of (x : ℝ) : (x / 5 + 8 = 61) → x = 265 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l3551_355181


namespace NUMINAMATH_CALUDE_average_problem_l3551_355112

theorem average_problem (x y : ℝ) :
  (7 + 9 + x + y + 17) / 5 = 10 →
  ((x + 3) + (x + 5) + (y + 2) + 8 + (y + 18)) / 5 = 14 :=
by sorry

end NUMINAMATH_CALUDE_average_problem_l3551_355112


namespace NUMINAMATH_CALUDE_abes_age_l3551_355152

theorem abes_age (present_age : ℕ) 
  (h : present_age + (present_age - 7) = 27) : 
  present_age = 17 := by
sorry

end NUMINAMATH_CALUDE_abes_age_l3551_355152


namespace NUMINAMATH_CALUDE_sum_of_evaluations_is_32_l3551_355110

/-- The expression to be evaluated -/
def expression : List ℕ := [1, 2, 3, 4, 5, 6]

/-- A sign assignment is a list of booleans, where true represents + and false represents - -/
def SignAssignment := List Bool

/-- Evaluate the expression given a sign assignment -/
def evaluate (signs : SignAssignment) : ℤ :=
  sorry

/-- Generate all possible sign assignments -/
def allSignAssignments : List SignAssignment :=
  sorry

/-- Calculate the sum of all evaluations -/
def sumOfEvaluations : ℤ :=
  sorry

/-- The main theorem: The sum of all evaluations is 32 -/
theorem sum_of_evaluations_is_32 : sumOfEvaluations = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_evaluations_is_32_l3551_355110


namespace NUMINAMATH_CALUDE_no_zeros_in_larger_interval_l3551_355166

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of having a unique zero in the given intervals
def has_unique_zero_in_intervals (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0 ∧ 
    0 < x ∧ x < 16 ∧
    0 < x ∧ x < 8 ∧
    0 < x ∧ x < 4 ∧
    0 < x ∧ x < 2

-- State the theorem
theorem no_zeros_in_larger_interval 
  (h : has_unique_zero_in_intervals f) : 
  ∀ x ∈ Set.Icc 2 16, f x ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_no_zeros_in_larger_interval_l3551_355166


namespace NUMINAMATH_CALUDE_power_of_power_l3551_355123

theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3551_355123


namespace NUMINAMATH_CALUDE_sarah_ate_36_candies_l3551_355199

/-- The number of candy pieces Sarah ate -/
def candyEaten (initialCandy : ℕ) (piles : ℕ) (piecesPerPile : ℕ) : ℕ :=
  initialCandy - (piles * piecesPerPile)

/-- Proof that Sarah ate 36 pieces of candy -/
theorem sarah_ate_36_candies :
  candyEaten 108 8 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sarah_ate_36_candies_l3551_355199
