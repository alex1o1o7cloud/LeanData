import Mathlib

namespace inequality_equivalence_l2651_265145

theorem inequality_equivalence (a b c : ℕ+) :
  (∀ x y z : ℝ, (x - y) ^ a.val * (x - z) ^ b.val * (y - z) ^ c.val ≥ 0) ↔ 
  (∃ m n p : ℕ, a.val = 2 * m ∧ b.val = 2 * n ∧ c.val = 2 * p) :=
by sorry

end inequality_equivalence_l2651_265145


namespace average_increase_l2651_265121

theorem average_increase (x₁ x₂ x₃ : ℝ) :
  (x₁ + x₂ + x₃) / 3 = 5 →
  ((x₁ + 2) + (x₂ + 2) + (x₃ + 2)) / 3 = 7 := by
sorry

end average_increase_l2651_265121


namespace negation_equivalence_l2651_265115

theorem negation_equivalence :
  (¬ ∃ x : ℤ, 7 ∣ x ∧ ¬ Odd x) ↔ (∀ x : ℤ, 7 ∣ x → Odd x) :=
by sorry

end negation_equivalence_l2651_265115


namespace nine_is_unique_digit_l2651_265195

/-- A function that returns true if a natural number ends with at least k repetitions of digit z -/
def endsWithKDigits (num : ℕ) (k : ℕ) (z : ℕ) : Prop :=
  ∃ m : ℕ, num = m * (10^k) + z * ((10^k - 1) / 9)

/-- The main theorem stating that 9 is the only digit satisfying the condition -/
theorem nine_is_unique_digit : 
  ∀ z : ℕ, z < 10 →
    (∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ endsWithKDigits (n^9) k z) ↔ z = 9 :=
sorry

end nine_is_unique_digit_l2651_265195


namespace shopkeeper_visits_l2651_265100

theorem shopkeeper_visits (initial_amount : ℚ) (spent_per_shop : ℚ) : initial_amount = 8.75 ∧ spent_per_shop = 10 →
  ∃ n : ℕ, n = 3 ∧ 2^n * initial_amount - spent_per_shop * (2^n - 1) = 0 := by
  sorry

end shopkeeper_visits_l2651_265100


namespace fraction_simplification_l2651_265158

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x - 1/y ≠ 0) :
  (y - 1/x) / (x - 1/y) = y/x :=
by sorry

end fraction_simplification_l2651_265158


namespace earth_surface_cultivation_l2651_265192

theorem earth_surface_cultivation (total : ℝ) (water_percentage : ℝ) (land_percentage : ℝ)
  (desert_ice_fraction : ℝ) (pasture_forest_mountain_fraction : ℝ) :
  water_percentage = 70 →
  land_percentage = 30 →
  water_percentage + land_percentage = 100 →
  desert_ice_fraction = 2/5 →
  pasture_forest_mountain_fraction = 1/3 →
  (land_percentage / 100 * total * (1 - desert_ice_fraction - pasture_forest_mountain_fraction)) / total * 100 = 8 :=
by sorry

end earth_surface_cultivation_l2651_265192


namespace sector_angle_l2651_265109

/-- A circular sector with arc length and area both equal to 4 has a central angle of 2 radians -/
theorem sector_angle (R : ℝ) (α : ℝ) (h1 : α * R = 4) (h2 : (1/2) * α * R^2 = 4) : α = 2 := by
  sorry

end sector_angle_l2651_265109


namespace parallelepiped_volume_l2651_265141

theorem parallelepiped_volume (j : ℝ) :
  j > 0 →
  (abs (3 * (j^2 - 9) - 2 * (4*j - 15) + 2 * (12 - 5*j)) = 36) →
  j = (9 + Real.sqrt 585) / 6 := by
sorry

end parallelepiped_volume_l2651_265141


namespace f_monotonicity_and_extrema_l2651_265106

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - 5

theorem f_monotonicity_and_extrema :
  (∀ x y, x < y ∧ x < 1 → f x < f y) ∧
  (∀ x y, 3 < x ∧ x < y → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y ∧ y < 3 → f x > f y) ∧
  (∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → f x < f 1) ∧
  (∃ δ > 0, ∀ x, 0 < |x - 3| ∧ |x - 3| < δ → f x > f 3) ∧
  f 1 = -1 ∧
  f 3 = -5 := by
sorry

end f_monotonicity_and_extrema_l2651_265106


namespace smallest_divisible_by_14_15_16_l2651_265168

theorem smallest_divisible_by_14_15_16 : ∃ n : ℕ+, 
  (∀ m : ℕ+, 14 ∣ m ∧ 15 ∣ m ∧ 16 ∣ m → n ≤ m) ∧
  14 ∣ n ∧ 15 ∣ n ∧ 16 ∣ n :=
by
  use 1680
  sorry

end smallest_divisible_by_14_15_16_l2651_265168


namespace knights_and_liars_l2651_265113

-- Define the inhabitants
inductive Inhabitant : Type
| A
| B
| C

-- Define the possible types of inhabitants
inductive InhabitantType : Type
| Knight
| Liar

-- Define a function to determine if an inhabitant is a knight or liar
def isKnight : Inhabitant → Bool
| Inhabitant.A => true  -- We assume A is a knight based on the solution
| Inhabitant.B => true  -- To be proved
| Inhabitant.C => false -- To be proved

-- Define what B and C claim about A's statement
def B_claim : Prop := isKnight Inhabitant.A = true
def C_claim : Prop := isKnight Inhabitant.A = false

-- The main theorem to prove
theorem knights_and_liars :
  (B_claim ∧ ¬C_claim) →
  (isKnight Inhabitant.B = true ∧ isKnight Inhabitant.C = false) := by
  sorry


end knights_and_liars_l2651_265113


namespace normal_distribution_symmetry_l2651_265170

/-- A random variable following a normal distribution with mean 3 and standard deviation σ -/
def X (σ : ℝ) : Type := Unit

/-- The probability that X is less than 2 -/
def prob_X_less_than_2 (σ : ℝ) : ℝ := 0.3

/-- The probability that X is between 2 and 4 -/
def prob_X_between_2_and_4 (σ : ℝ) : ℝ := 1 - 2 * prob_X_less_than_2 σ

theorem normal_distribution_symmetry (σ : ℝ) (h : σ > 0) :
  prob_X_between_2_and_4 σ = 0.4 := by
  sorry

end normal_distribution_symmetry_l2651_265170


namespace sin_80_minus_sin_20_over_cos_20_l2651_265175

theorem sin_80_minus_sin_20_over_cos_20 :
  (2 * Real.sin (80 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end sin_80_minus_sin_20_over_cos_20_l2651_265175


namespace abc_product_l2651_265174

theorem abc_product (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 30)
  (h_eq : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 300 / (a * b * c) = 1) :
  a * b * c = 768 := by
  sorry

end abc_product_l2651_265174


namespace sum_repeating_decimals_eq_l2651_265133

/-- The sum of the repeating decimals 0.141414... and 0.272727... -/
def sum_repeating_decimals : ℚ :=
  let a : ℚ := 14 / 99  -- 0.141414...
  let b : ℚ := 27 / 99  -- 0.272727...
  a + b

/-- Theorem: The sum of the repeating decimals 0.141414... and 0.272727... is 41/99 -/
theorem sum_repeating_decimals_eq :
  sum_repeating_decimals = 41 / 99 := by
  sorry

end sum_repeating_decimals_eq_l2651_265133


namespace isosceles_triangle_solution_l2651_265186

def isosceles_triangle_sides (perimeter : ℝ) (height_ratio : ℝ) : Prop :=
  let base := 130
  let leg := 169
  perimeter = base + 2 * leg ∧
  height_ratio = 10 / 13 ∧
  base * (13 : ℝ) = leg * (10 : ℝ)

theorem isosceles_triangle_solution :
  isosceles_triangle_sides 468 (10 / 13) :=
sorry

end isosceles_triangle_solution_l2651_265186


namespace xy_divided_by_three_l2651_265128

theorem xy_divided_by_three (x y : ℚ) 
  (eq1 : 2 * x + y = 6) 
  (eq2 : x + 2 * y = 5) : 
  (x + y) / 3 = 1.222222222222222 := by
sorry

end xy_divided_by_three_l2651_265128


namespace cube_sum_from_sum_and_product_l2651_265177

theorem cube_sum_from_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 14) : x^3 + y^3 = 580 := by
  sorry

end cube_sum_from_sum_and_product_l2651_265177


namespace number_count_proof_l2651_265101

theorem number_count_proof (total_avg : ℝ) (group1_avg : ℝ) (group2_avg : ℝ) (group3_avg : ℝ) :
  total_avg = 3.95 →
  group1_avg = 4.2 →
  group2_avg = 3.85 →
  group3_avg = 3.8000000000000007 →
  (2 * group1_avg + 2 * group2_avg + 2 * group3_avg) / total_avg = 6 :=
by
  sorry

end number_count_proof_l2651_265101


namespace one_volleyball_outside_range_l2651_265150

def volleyball_weights : List ℝ := [275, 263, 278, 270, 261, 277, 282, 269]
def standard_weight : ℝ := 270
def tolerance : ℝ := 10

theorem one_volleyball_outside_range : 
  (volleyball_weights.filter (λ w => w < standard_weight - tolerance ∨ 
                                     w > standard_weight + tolerance)).length = 1 :=
by sorry

end one_volleyball_outside_range_l2651_265150


namespace min_value_product_quotient_l2651_265110

theorem min_value_product_quotient (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) ≥ 343 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 + 5*a + 2) * (b^2 + 5*b + 2) * (c^2 + 5*c + 2) / (a*b*c) = 343 :=
by sorry

end min_value_product_quotient_l2651_265110


namespace probability_one_common_number_l2651_265114

/-- The number of numbers in the lottery -/
def totalNumbers : ℕ := 45

/-- The number of numbers each participant chooses -/
def chosenNumbers : ℕ := 6

/-- The probability of exactly one common number between two independently chosen combinations -/
def probabilityOneCommon : ℚ :=
  (chosenNumbers : ℚ) * (Nat.choose (totalNumbers - chosenNumbers) (chosenNumbers - 1) : ℚ) /
  (Nat.choose totalNumbers chosenNumbers : ℚ)

/-- Theorem stating the probability of exactly one common number -/
theorem probability_one_common_number :
  probabilityOneCommon = (6 : ℚ) * (Nat.choose 39 5 : ℚ) / (Nat.choose 45 6 : ℚ) :=
by sorry

end probability_one_common_number_l2651_265114


namespace decimal_point_problem_l2651_265138

theorem decimal_point_problem :
  ∃! (x : ℝ), x > 0 ∧ 100000 * x = 5 * (1 / x) ∧ x = Real.sqrt 2 / 200 := by
  sorry

end decimal_point_problem_l2651_265138


namespace isosceles_right_triangle_area_l2651_265197

/-- Represents a triangle XYZ -/
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Checks if a triangle is isosceles right -/
def isIsoscelesRight (t : Triangle) : Prop := sorry

/-- Calculates the length of a side given two points -/
def sideLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

theorem isosceles_right_triangle_area 
  (t : Triangle) 
  (h1 : isIsoscelesRight t) 
  (h2 : sideLength t.X t.Y > sideLength t.Y t.Z) 
  (h3 : sideLength t.X t.Y = 12.000000000000002) : 
  triangleArea t = 36.000000000000015 := by
  sorry

end isosceles_right_triangle_area_l2651_265197


namespace trapezoid_area_sum_l2651_265194

/-- Given a trapezoid with side lengths 5, 6, 8, and 9, the sum of all possible areas is 28√3 + 42√2. -/
theorem trapezoid_area_sum :
  ∀ (s₁ s₂ s₃ s₄ : ℝ),
  s₁ = 5 ∧ s₂ = 6 ∧ s₃ = 8 ∧ s₄ = 9 →
  ∃ (A₁ A₂ : ℝ),
  (A₁ = (s₁ + s₄) * Real.sqrt 3 ∧
   A₂ = (s₂ + s₃) * Real.sqrt 2) →
  A₁ + A₂ = 28 * Real.sqrt 3 + 42 * Real.sqrt 2 :=
by sorry

end trapezoid_area_sum_l2651_265194


namespace cube_root_of_product_l2651_265140

theorem cube_root_of_product (a b c : ℕ) : 
  (2^9 * 5^3 * 7^3 : ℝ)^(1/3) = 280 := by
  sorry

end cube_root_of_product_l2651_265140


namespace ball_distribution_l2651_265160

theorem ball_distribution (red_balls : ℕ) (white_balls : ℕ) (boxes : ℕ) : 
  red_balls = 17 → white_balls = 10 → boxes = 4 →
  (Nat.choose (white_balls + boxes - 1) (boxes - 1)) * 
  (Nat.choose (red_balls - 1) (boxes - 1)) = 5720 := by
sorry

end ball_distribution_l2651_265160


namespace subset_equality_l2651_265157

theorem subset_equality (h : ℕ) (X S : Set ℕ) : h ≥ 3 →
  X = {n : ℕ | n ≥ 2 * h} →
  S ⊆ X →
  S.Nonempty →
  (∀ a b : ℕ, a ≥ h → b ≥ h → (a + b) ∈ S → (a * b) ∈ S) →
  (∀ a b : ℕ, a ≥ h → b ≥ h → (a * b) ∈ S → (a + b) ∈ S) →
  S = X :=
by sorry

end subset_equality_l2651_265157


namespace perpendicular_lines_l2651_265190

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, x + 2*y + 3 = 0 ∧ 4*x - a*y + 5 = 0) →
  ((-(1:ℝ)/2) * (4/a) = -1) →
  a = 2 := by sorry

end perpendicular_lines_l2651_265190


namespace line_parallel_plane_not_all_lines_l2651_265130

/-- A plane in 3D space -/
structure Plane3D where
  -- Define plane properties here
  mk :: -- Constructor

/-- A line in 3D space -/
structure Line3D where
  -- Define line properties here
  mk :: -- Constructor

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines when a line is in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem line_parallel_plane_not_all_lines 
  (p : Plane3D) : 
  ∃ (l : Line3D), parallel_line_plane l p ∧ 
  ∃ (m : Line3D), line_in_plane m p ∧ ¬parallel_lines l m :=
sorry

end line_parallel_plane_not_all_lines_l2651_265130


namespace total_cost_is_100_l2651_265117

/-- Calculates the total cost in dollars for using whiteboards in all classes for one day -/
def whiteboard_cost (num_classes : ℕ) (boards_per_class : ℕ) (ink_per_board : ℕ) (cost_per_ml : ℚ) : ℚ :=
  num_classes * boards_per_class * ink_per_board * cost_per_ml

/-- Proves that the total cost for using whiteboards in all classes for one day is $100 -/
theorem total_cost_is_100 : 
  whiteboard_cost 5 2 20 (1/2) = 100 := by
  sorry

#eval whiteboard_cost 5 2 20 (1/2)

end total_cost_is_100_l2651_265117


namespace consecutive_numbers_divisibility_l2651_265143

theorem consecutive_numbers_divisibility (n : ℕ) :
  n ≥ 4 ∧
  n ∣ ((n - 3) * (n - 2) * (n - 1)) →
  n = 6 :=
by sorry

end consecutive_numbers_divisibility_l2651_265143


namespace arts_group_size_l2651_265154

/-- The number of days it takes one student to complete the project -/
def days_for_one_student : ℕ := 60

/-- The number of additional students who joined -/
def additional_students : ℕ := 15

/-- The total number of days worked -/
def total_days_worked : ℕ := 3

/-- The number of days worked with additional students -/
def days_with_additional : ℕ := 2

/-- The total amount of work to be done -/
def total_work : ℚ := 1

theorem arts_group_size :
  ∃ (x : ℕ),
    (x : ℚ) / days_for_one_student + 
    days_with_additional * ((x : ℚ) + additional_students) / days_for_one_student = total_work ∧
    x = 10 := by
  sorry

end arts_group_size_l2651_265154


namespace rotate_vector_90_degrees_l2651_265159

/-- Given points O and P in a 2D Cartesian coordinate system, and Q obtained by rotating OP counterclockwise by π/2, prove that Q has coordinates (-2, 1) -/
theorem rotate_vector_90_degrees (O P Q : ℝ × ℝ) : 
  O = (0, 0) → 
  P = (1, 2) → 
  (Q.1 - O.1, Q.2 - O.2) = (-(P.2 - O.2), P.1 - O.1) → 
  Q = (-2, 1) := by
  sorry

end rotate_vector_90_degrees_l2651_265159


namespace unique_solution_floor_equation_l2651_265112

-- Define the floor function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem unique_solution_floor_equation :
  ∃! x : ℝ, (floor (x - 1/2) : ℝ) = 3*x - 5 :=
sorry

end unique_solution_floor_equation_l2651_265112


namespace roots_equation_r_value_l2651_265132

theorem roots_equation_r_value (m p : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) → 
  (r = 16/3) := by
sorry

end roots_equation_r_value_l2651_265132


namespace dvd_cost_l2651_265134

/-- Given that two identical DVDs cost $36, prove that six of these DVDs cost $108. -/
theorem dvd_cost (two_dvd_cost : ℕ) (h : two_dvd_cost = 36) : 
  (6 * two_dvd_cost / 2 : ℚ) = 108 := by sorry

end dvd_cost_l2651_265134


namespace read_book_in_12_days_l2651_265137

/-- Represents the number of days it takes to read a book -/
def days_to_read_book (total_pages : ℕ) (weekday_pages : ℕ) (weekend_pages : ℕ) : ℕ :=
  let pages_per_week := 5 * weekday_pages + 2 * weekend_pages
  let full_weeks := total_pages / pages_per_week
  let remaining_pages := total_pages % pages_per_week
  let additional_days := 
    if remaining_pages ≤ 5 * weekday_pages
    then (remaining_pages + weekday_pages - 1) / weekday_pages
    else 5 + (remaining_pages - 5 * weekday_pages + weekend_pages - 1) / weekend_pages
  7 * full_weeks + additional_days

/-- Theorem stating that it takes 12 days to read the book under given conditions -/
theorem read_book_in_12_days : 
  days_to_read_book 285 23 35 = 12 := by
  sorry

end read_book_in_12_days_l2651_265137


namespace calculate_expression_l2651_265162

theorem calculate_expression : |1 - Real.sqrt 2| - Real.sqrt 8 + (Real.sqrt 2 - 1)^0 = -Real.sqrt 2 := by
  sorry

end calculate_expression_l2651_265162


namespace lions_and_majestic_l2651_265173

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Lion : U → Prop)
variable (Majestic : U → Prop)
variable (Bird : U → Prop)

-- State the given conditions
variable (h1 : ∀ x, Lion x → Majestic x)
variable (h2 : ∀ x, Bird x → ¬Lion x)

-- Theorem to prove
theorem lions_and_majestic :
  (∀ x, Lion x → ¬Bird x) ∧ (∃ x, Majestic x ∧ ¬Bird x) :=
sorry

end lions_and_majestic_l2651_265173


namespace twenty_percent_value_l2651_265131

theorem twenty_percent_value (x : ℝ) (h : 1.2 * x = 1200) : 0.2 * x = 200 := by
  sorry

end twenty_percent_value_l2651_265131


namespace tan_and_g_alpha_l2651_265191

open Real

theorem tan_and_g_alpha (α : ℝ) 
  (h1 : π / 2 < α) (h2 : α < π) 
  (h3 : tan α - (tan α)⁻¹ = -8/3) : 
  tan α = -3 ∧ 
  (sin (π + α) + 4 * cos (2*π + α)) / (sin (π/2 - α) - 4 * sin (-α)) = -7/11 := by
  sorry

end tan_and_g_alpha_l2651_265191


namespace f_properties_l2651_265189

noncomputable def f (x : ℝ) : ℝ := Real.cos (x - Real.pi / 3) - Real.sin (Real.pi / 2 - x)

theorem f_properties (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : f (α + Real.pi / 6) = 3 / 5) :
  (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S)) ∧
  f (2 * α) = (24 * Real.sqrt 3 - 7) / 50 :=
by sorry

end f_properties_l2651_265189


namespace perpendicular_bisector_equation_l2651_265155

/-- The perpendicular bisector of a line segment AB is the line that passes through 
    the midpoint of AB and is perpendicular to AB. This theorem proves that 
    y = -2x + 3 is the equation of the perpendicular bisector of the line segment 
    connecting points A(-1, 0) and B(3, 2). -/
theorem perpendicular_bisector_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (3, 2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let slope_AB : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let slope_perp : ℝ := -1 / slope_AB
  y = -2 * x + 3 ↔ 
    (x, y) ∈ {p : ℝ × ℝ | (p.1 - midpoint.1) * slope_AB = (midpoint.2 - p.2)} ∧
    (y - midpoint.2) = slope_perp * (x - midpoint.1) :=
sorry

end perpendicular_bisector_equation_l2651_265155


namespace line_through_point_with_equal_intercepts_l2651_265116

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

theorem line_through_point_with_equal_intercepts :
  ∃ (l₁ l₂ : Line),
    (passesThrough l₁ ⟨1, 2⟩ ∧ hasEqualIntercepts l₁) ∧
    (passesThrough l₂ ⟨1, 2⟩ ∧ hasEqualIntercepts l₂) ∧
    ((l₁.a = 2 ∧ l₁.b = -1 ∧ l₁.c = 0) ∨
     (l₂.a = 1 ∧ l₂.b = 1 ∧ l₂.c = -3)) :=
  sorry

end line_through_point_with_equal_intercepts_l2651_265116


namespace expand_and_simplify_l2651_265161

theorem expand_and_simplify : 
  (3 + Real.sqrt 10) * (Real.sqrt 2 - Real.sqrt 5) = -2 * Real.sqrt 2 - Real.sqrt 5 := by
  sorry

end expand_and_simplify_l2651_265161


namespace imaginary_part_of_z_l2651_265182

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z as given in the problem
def z : ℂ := (1 + i) * i

-- Theorem statement
theorem imaginary_part_of_z :
  Complex.im z = 1 := by sorry

end imaginary_part_of_z_l2651_265182


namespace new_average_salary_l2651_265156

theorem new_average_salary
  (initial_average : ℚ)
  (old_supervisor_salary : ℚ)
  (new_supervisor_salary : ℚ)
  (num_people : ℕ)
  (h1 : initial_average = 430)
  (h2 : old_supervisor_salary = 870)
  (h3 : new_supervisor_salary = 690)
  (h4 : num_people = 9)
  : (num_people - 1 : ℚ) * initial_average + new_supervisor_salary - old_supervisor_salary = 410 * num_people :=
by
  sorry

#eval (9 - 1 : ℚ) * 430 + 690 - 870
#eval 410 * 9

end new_average_salary_l2651_265156


namespace cube_coloring_count_l2651_265129

/-- The number of distinct orientations of a cube -/
def cubeOrientations : ℕ := 24

/-- The number of ways to permute 6 colors -/
def colorPermutations : ℕ := 720

/-- The number of distinct ways to paint a cube's faces with 6 different colors,
    where each color appears exactly once and rotations are considered identical -/
def distinctCubeColorings : ℕ := colorPermutations / cubeOrientations

theorem cube_coloring_count :
  distinctCubeColorings = 30 := by sorry

end cube_coloring_count_l2651_265129


namespace solution_satisfies_equation_l2651_265188

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the equation
def equation (y : ℝ) : Prop :=
  cubeRoot (30 * y + cubeRoot (30 * y + 26)) = 26

-- Theorem statement
theorem solution_satisfies_equation : equation 585 := by
  sorry

end solution_satisfies_equation_l2651_265188


namespace octal_to_decimal_23456_l2651_265179

/-- Converts a base-8 digit to its base-10 equivalent --/
def octal_to_decimal (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (8 ^ position)

/-- The base-10 equivalent of 23456 in base-8 --/
def base_10_equivalent : ℕ :=
  octal_to_decimal 6 0 +
  octal_to_decimal 5 1 +
  octal_to_decimal 4 2 +
  octal_to_decimal 3 3 +
  octal_to_decimal 2 4

/-- Theorem: The base-10 equivalent of 23456 in base-8 is 5934 --/
theorem octal_to_decimal_23456 : base_10_equivalent = 5934 := by
  sorry

end octal_to_decimal_23456_l2651_265179


namespace nth_equation_l2651_265187

theorem nth_equation (n : ℕ) : ((n + 1)^2 - n^2 - 1) / 2 = n := by
  sorry

end nth_equation_l2651_265187


namespace probability_is_three_fiftieths_l2651_265124

/-- Represents a 5x5x5 cube with painted faces -/
structure PaintedCube :=
  (size : ℕ)
  (total_cubes : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)

/-- Represents the count of different types of unit cubes -/
structure CubeCounts :=
  (two_blue : ℕ)
  (unpainted : ℕ)

/-- Calculates the probability of selecting specific cube types -/
def probability_two_blue_and_unpainted (cube : PaintedCube) (counts : CubeCounts) : ℚ :=
  let total_combinations := (cube.total_cubes.choose 2 : ℚ)
  let favorable_outcomes := (counts.two_blue * counts.unpainted : ℚ)
  favorable_outcomes / total_combinations

/-- The main theorem to be proved -/
theorem probability_is_three_fiftieths (cube : PaintedCube) (counts : CubeCounts) : 
  cube.size = 5 ∧ 
  cube.total_cubes = 125 ∧ 
  cube.blue_faces = 2 ∧ 
  cube.red_faces = 1 ∧
  counts.two_blue = 9 ∧
  counts.unpainted = 51 →
  probability_two_blue_and_unpainted cube counts = 3 / 50 :=
sorry

end probability_is_three_fiftieths_l2651_265124


namespace dot_product_sum_l2651_265184

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -2)

theorem dot_product_sum : 
  (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) = -1 := by sorry

end dot_product_sum_l2651_265184


namespace double_force_quadruple_power_l2651_265105

/-- Represents the scenario of tugboats pushing a barge -/
structure TugboatScenario where
  /-- Initial force applied by a single tugboat -/
  F : ℝ
  /-- Coefficient of water resistance -/
  k : ℝ
  /-- Initial speed of the barge -/
  v : ℝ
  /-- Water resistance is proportional to speed -/
  resistance_prop : F = k * v

/-- Theorem stating that doubling the force quadruples the power when water resistance is proportional to speed -/
theorem double_force_quadruple_power (scenario : TugboatScenario) :
  let v' := 2 * scenario.v  -- New speed after doubling force
  let P := scenario.F * scenario.v  -- Initial power
  let P' := (2 * scenario.F) * v'  -- New power after doubling force
  P' = 4 * P := by sorry

end double_force_quadruple_power_l2651_265105


namespace inequality_proof_l2651_265172

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  Real.sqrt (a^(1 - a) * b^(1 - b) * c^(1 - c)) ≤ 1/3 := by
  sorry

end inequality_proof_l2651_265172


namespace absolute_value_five_l2651_265151

theorem absolute_value_five (x : ℝ) : |x| = 5 → x = 5 ∨ x = -5 := by
  sorry

end absolute_value_five_l2651_265151


namespace pool_length_l2651_265171

theorem pool_length (width : ℝ) (depth : ℝ) (capacity : ℝ) (drain_rate : ℝ) (drain_time : ℝ) :
  width = 50 →
  depth = 10 →
  capacity = 0.8 →
  drain_rate = 60 →
  drain_time = 1000 →
  ∃ (length : ℝ), length = 150 ∧ capacity * width * length * depth = drain_rate * drain_time :=
by sorry

end pool_length_l2651_265171


namespace custom_op_problem_l2651_265149

/-- The custom operation @ defined as a @ b = a × (a + 1) × ... × (a + b - 1) -/
def custom_op (a b : ℕ) : ℕ := 
  (List.range b).foldl (fun acc i => acc * (a + i)) a

/-- Theorem stating that if x @ y @ 2 = 420, then y @ x = 20 -/
theorem custom_op_problem (x y : ℕ) : 
  custom_op x (custom_op y 2) = 420 → custom_op y x = 20 := by
  sorry

#check custom_op_problem

end custom_op_problem_l2651_265149


namespace count_solutions_eq_51_l2651_265111

/-- The number of distinct ordered pairs (a, b) of non-negative integers that satisfy a + b = 50 -/
def count_solutions : ℕ := 
  (Finset.range 51).card

/-- Theorem: There are 51 distinct ordered pairs (a, b) of non-negative integers that satisfy a + b = 50 -/
theorem count_solutions_eq_51 : count_solutions = 51 := by
  sorry

#eval count_solutions  -- This should output 51

end count_solutions_eq_51_l2651_265111


namespace cost_per_block_l2651_265108

/-- Proves that the cost per piece of concrete block is $2 -/
theorem cost_per_block (blocks_per_section : ℕ) (num_sections : ℕ) (total_cost : ℚ) :
  blocks_per_section = 30 →
  num_sections = 8 →
  total_cost = 480 →
  total_cost / (blocks_per_section * num_sections) = 2 := by
  sorry

end cost_per_block_l2651_265108


namespace right_triangle_30_deg_side_half_hypotenuse_l2651_265147

/-- Theorem: In a right-angled triangle with one angle of 30°, 
    the length of the side opposite to the 30° angle is equal to 
    half the length of the hypotenuse. -/
theorem right_triangle_30_deg_side_half_hypotenuse 
  (A B C : ℝ × ℝ) -- Three points representing the vertices of the triangle
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) -- Right angle condition
  (angle_30_deg : ∃ i j k, i^2 + j^2 = k^2 ∧ i / k = 1 / 2) -- 30° angle condition
  : ∃ side hypotenuse, side = hypotenuse / 2 := by
  sorry

end right_triangle_30_deg_side_half_hypotenuse_l2651_265147


namespace inequality_proof_l2651_265139

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 := by
  sorry

end inequality_proof_l2651_265139


namespace arithmetic_sequence_sum_l2651_265120

/-- An arithmetic sequence {a_n} with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 3 + a 4 + a 5 = 12 →
  a 6 = 2 →
  a 2 + a 8 = 6 := by
  sorry

end arithmetic_sequence_sum_l2651_265120


namespace f_satisfies_conditions_l2651_265136

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + |x| + 1

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
  (∀ x : ℝ, f x = f (-x)) := by
  sorry

end f_satisfies_conditions_l2651_265136


namespace garden_width_is_eleven_l2651_265122

/-- Represents a rectangular garden with specific dimensions. -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_width_relation : length = width + 2
  perimeter_formula : perimeter = 2 * (length + width)

/-- Theorem: The width of a rectangular garden with perimeter 48m and length 2m more than width is 11m. -/
theorem garden_width_is_eleven (garden : RectangularGarden) 
    (h_perimeter : garden.perimeter = 48) : garden.width = 11 := by
  sorry

#check garden_width_is_eleven

end garden_width_is_eleven_l2651_265122


namespace lcm_problem_l2651_265152

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 6) (h2 : a * b = 432) :
  Nat.lcm a b = 72 := by
  sorry

end lcm_problem_l2651_265152


namespace goose_eggs_count_l2651_265164

theorem goose_eggs_count (total_eggs : ℕ) : 
  (total_eggs : ℚ) * (1/2) * (3/4) * (2/5) = 120 →
  total_eggs = 400 := by
  sorry

end goose_eggs_count_l2651_265164


namespace intersection_point_of_f_and_inverse_l2651_265196

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 4*x^2 + 13*x + 20

-- Theorem statement
theorem intersection_point_of_f_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-2, -2) :=
by
  sorry


end intersection_point_of_f_and_inverse_l2651_265196


namespace georgie_guacamole_servings_l2651_265102

/-- The number of avocados needed for one serving of guacamole -/
def avocados_per_serving : ℕ := 3

/-- The number of avocados Georgie initially has -/
def initial_avocados : ℕ := 5

/-- The number of avocados Georgie's sister buys -/
def sister_bought_avocados : ℕ := 4

/-- The total number of avocados Georgie has -/
def total_avocados : ℕ := initial_avocados + sister_bought_avocados

/-- The number of servings of guacamole Georgie can make -/
def servings_of_guacamole : ℕ := total_avocados / avocados_per_serving

theorem georgie_guacamole_servings : servings_of_guacamole = 3 := by
  sorry

end georgie_guacamole_servings_l2651_265102


namespace right_distance_is_73_l2651_265126

/-- Represents a square table with a centered round plate -/
structure TableWithPlate where
  /-- Length of the square table's side -/
  table_side : ℝ
  /-- Diameter of the round plate -/
  plate_diameter : ℝ
  /-- Distance from plate edge to left table edge -/
  left_distance : ℝ
  /-- Distance from plate edge to top table edge -/
  top_distance : ℝ
  /-- Distance from plate edge to bottom table edge -/
  bottom_distance : ℝ
  /-- The plate is centered on the table -/
  centered : left_distance + plate_diameter + (table_side - left_distance - plate_diameter) = top_distance + plate_diameter + bottom_distance

/-- Theorem stating the distance from plate edge to right table edge -/
theorem right_distance_is_73 (t : TableWithPlate) (h1 : t.left_distance = 10) (h2 : t.top_distance = 63) (h3 : t.bottom_distance = 20) : 
  t.table_side - t.left_distance - t.plate_diameter = 73 := by
  sorry

end right_distance_is_73_l2651_265126


namespace cubic_root_sum_l2651_265123

/-- Given r is the positive real solution to x³ - x² + ¼x - 1 = 0,
    prove that the infinite sum r³ + 2r⁶ + 3r⁹ + 4r¹² + ... equals 16r -/
theorem cubic_root_sum (r : ℝ) (hr : r > 0) (hroot : r^3 - r^2 + (1/4)*r - 1 = 0) :
  (∑' n, (n : ℝ) * r^(3*n)) = 16*r := by
  sorry

end cubic_root_sum_l2651_265123


namespace probability_intersection_bounds_l2651_265193

theorem probability_intersection_bounds (A B : Set Ω) (P : Set Ω → ℝ) 
  (hA : P A = 3/4) (hB : P B = 2/3) :
  5/12 ≤ P (A ∩ B) ∧ P (A ∩ B) ≤ 2/3 := by
sorry

end probability_intersection_bounds_l2651_265193


namespace prime_divides_all_f_l2651_265166

def f (n x : ℕ) : ℕ := Nat.choose n x

theorem prime_divides_all_f (p : ℕ) (hp : Prime p) (n : ℕ) (hn : n > 1) :
  (∀ x : ℕ, 1 ≤ x ∧ x < n → p ∣ f n x) ↔ ∃ m : ℕ, n = p ^ m :=
sorry

end prime_divides_all_f_l2651_265166


namespace polynomial_root_sum_l2651_265180

theorem polynomial_root_sum (b c d e : ℝ) : 
  (∀ x : ℝ, 2*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ↔ x = 4 ∨ x = -3 ∨ x = 5 ∨ x = ((-b-c-d)/2)) →
  (b + c + d) / 2 = 151 := by
sorry

end polynomial_root_sum_l2651_265180


namespace colored_paper_count_l2651_265144

theorem colored_paper_count (used left : ℕ) (h1 : used = 9) (h2 : left = 12) :
  used + left = 21 := by
  sorry

end colored_paper_count_l2651_265144


namespace arrangement_count_is_70_l2651_265181

/-- The number of ways to arrange 6 indistinguishable objects of type A
    and 4 indistinguishable objects of type B in a row of 10 positions,
    where type A objects must occupy the first and last positions. -/
def arrangement_count : ℕ := sorry

/-- Theorem stating that the number of arrangements is 70 -/
theorem arrangement_count_is_70 : arrangement_count = 70 := by sorry

end arrangement_count_is_70_l2651_265181


namespace rectangular_prism_diagonal_l2651_265165

/-- The diagonal length of a rectangular prism with given surface area and total edge length -/
theorem rectangular_prism_diagonal
  (x y z : ℝ)  -- lengths of sides
  (h1 : 2*x*y + 2*x*z + 2*y*z = 22)  -- surface area condition
  (h2 : 4*x + 4*y + 4*z = 24)  -- total edge length condition
  : ∃ d : ℝ, d^2 = x^2 + y^2 + z^2 ∧ d^2 = 14 :=
by sorry

end rectangular_prism_diagonal_l2651_265165


namespace fraction_equals_244_375_l2651_265153

/-- The fraction in the original problem -/
def original_fraction : ℚ :=
  (12^4+400)*(24^4+400)*(36^4+400)*(48^4+400)*(60^4+400) /
  ((6^4+400)*(18^4+400)*(30^4+400)*(42^4+400)*(54^4+400))

/-- The theorem stating that the original fraction equals 244.375 -/
theorem fraction_equals_244_375 : original_fraction = 244.375 := by
  sorry

end fraction_equals_244_375_l2651_265153


namespace water_formed_ethanol_combustion_l2651_265142

/-- Represents a chemical equation with reactants and products -/
structure ChemicalEquation :=
  (reactants : List (String × ℚ))
  (products : List (String × ℚ))

/-- Represents the available moles of reactants -/
structure AvailableReactants :=
  (ethanol : ℚ)
  (oxygen : ℚ)

/-- The balanced chemical equation for ethanol combustion -/
def ethanolCombustion : ChemicalEquation :=
  { reactants := [("C2H5OH", 1), ("O2", 3)],
    products := [("CO2", 2), ("H2O", 3)] }

/-- Calculates the amount of H2O formed in the ethanol combustion reaction -/
def waterFormed (available : AvailableReactants) (equation : ChemicalEquation) : ℚ :=
  sorry

/-- Theorem stating that 2 moles of H2O are formed when 2 moles of ethanol react with 2 moles of oxygen -/
theorem water_formed_ethanol_combustion :
  waterFormed { ethanol := 2, oxygen := 2 } ethanolCombustion = 2 := by
  sorry

end water_formed_ethanol_combustion_l2651_265142


namespace equation_solution_l2651_265119

theorem equation_solution : 
  ∃ x : ℝ, (10 : ℝ)^(2*x) * (100 : ℝ)^(3*x) = (1000 : ℝ)^7 ∧ x = 21/8 := by
  sorry

end equation_solution_l2651_265119


namespace factors_of_x4_plus_16_l2651_265125

theorem factors_of_x4_plus_16 (x : ℝ) : 
  (x^4 + 16 : ℝ) = (x^2 - 4*x + 4) * (x^2 + 4*x + 4) := by
  sorry

end factors_of_x4_plus_16_l2651_265125


namespace line_intersects_circle_through_center_l2651_265185

open Real

/-- Proves that a line intersects a circle through its center -/
theorem line_intersects_circle_through_center (α : ℝ) :
  let line := fun (x y : ℝ) => x * cos α - y * sin α = 1
  let circle := fun (x y : ℝ) => (x - cos α)^2 + (y + sin α)^2 = 4
  let center := (cos α, -sin α)
  line center.1 center.2 ∧ circle center.1 center.2 := by sorry

end line_intersects_circle_through_center_l2651_265185


namespace abie_initial_bags_l2651_265148

/-- The number of bags of chips Abie initially had -/
def initial_bags : ℕ := sorry

/-- The number of bags Abie gave away -/
def bags_given_away : ℕ := 4

/-- The number of bags Abie bought -/
def bags_bought : ℕ := 6

/-- The final number of bags Abie has -/
def final_bags : ℕ := 22

/-- Theorem stating that Abie initially had 20 bags of chips -/
theorem abie_initial_bags : 
  initial_bags = 20 ∧ 
  initial_bags - bags_given_away + bags_bought = final_bags :=
sorry

end abie_initial_bags_l2651_265148


namespace quadratic_inequality_solution_set_l2651_265127

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < -2) :
  {x : ℝ | a * x^2 + (a - 2) * x - 2 ≥ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2/a} := by
  sorry

end quadratic_inequality_solution_set_l2651_265127


namespace subtraction_and_divisibility_implies_sum_l2651_265199

theorem subtraction_and_divisibility_implies_sum (a b : Nat) : 
  (741 - (300 + 10*a + 4) = 400 + 10*b + 7) → 
  ((400 + 10*b + 7) % 11 = 0) → 
  (a + b = 3) := by
sorry

end subtraction_and_divisibility_implies_sum_l2651_265199


namespace total_potatoes_l2651_265104

-- Define the number of people sharing the potatoes
def num_people : Nat := 3

-- Define the number of potatoes each person received
def potatoes_per_person : Nat := 8

-- Theorem to prove the total number of potatoes
theorem total_potatoes : num_people * potatoes_per_person = 24 := by
  sorry

end total_potatoes_l2651_265104


namespace sum_of_two_positive_integers_greater_than_one_l2651_265103

theorem sum_of_two_positive_integers_greater_than_one (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b > 1 := by sorry

end sum_of_two_positive_integers_greater_than_one_l2651_265103


namespace sixth_score_for_target_mean_l2651_265118

def emily_scores : List ℕ := [88, 90, 85, 92, 97]

def target_mean : ℚ := 91

theorem sixth_score_for_target_mean :
  let all_scores := emily_scores ++ [94]
  (all_scores.sum : ℚ) / all_scores.length = target_mean := by sorry

end sixth_score_for_target_mean_l2651_265118


namespace explanatory_variable_is_fertilizer_amount_l2651_265163

/-- A study on crop yield prediction -/
structure CropStudy where
  fertilizer_amount : ℝ
  crop_yield : ℝ

/-- The explanatory variable in a regression analysis -/
inductive ExplanatoryVariable
  | CropYield
  | FertilizerAmount
  | Experimenter
  | OtherFactors

/-- The study aims to predict crop yield based on fertilizer amount -/
def study_aim (s : CropStudy) : Prop :=
  ∃ f : ℝ → ℝ, s.crop_yield = f s.fertilizer_amount

/-- The correct explanatory variable for the given study -/
def correct_explanatory_variable : ExplanatoryVariable :=
  ExplanatoryVariable.FertilizerAmount

/-- Theorem: The explanatory variable in the crop yield study is the fertilizer amount -/
theorem explanatory_variable_is_fertilizer_amount 
  (s : CropStudy) (aim : study_aim s) :
  correct_explanatory_variable = ExplanatoryVariable.FertilizerAmount :=
sorry

end explanatory_variable_is_fertilizer_amount_l2651_265163


namespace inequality_proof_l2651_265167

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / Real.sqrt y + y / Real.sqrt x ≥ Real.sqrt x + Real.sqrt y := by
  sorry

end inequality_proof_l2651_265167


namespace new_encoding_of_old_message_l2651_265169

/-- Represents the old encoding system --/
def OldEncoding : Type := String

/-- Represents the new encoding system --/
def NewEncoding : Type := String

/-- Decodes a message from the old encoding system --/
def decode (msg : OldEncoding) : String :=
  sorry

/-- Encodes a message using the new encoding system --/
def encode (msg : String) : NewEncoding :=
  sorry

/-- The new encoding rules --/
def newEncodingRules : List (Char × String) :=
  [('A', "21"), ('B', "122"), ('C', "1")]

/-- The theorem to be proved --/
theorem new_encoding_of_old_message :
  let oldMsg : OldEncoding := "011011010011"
  let decodedMsg := decode oldMsg
  encode decodedMsg = "211221121" :=
by sorry

end new_encoding_of_old_message_l2651_265169


namespace jimmy_earnings_theorem_l2651_265135

/-- Calculates Jimmy's total earnings from selling all his action figures --/
def jimmy_total_earnings : ℕ := by
  -- Define the number of each type of action figure
  let num_type_a : ℕ := 5
  let num_type_b : ℕ := 4
  let num_type_c : ℕ := 3

  -- Define the original value of each type of action figure
  let value_type_a : ℕ := 20
  let value_type_b : ℕ := 30
  let value_type_c : ℕ := 40

  -- Define the discount for each type of action figure
  let discount_type_a : ℕ := 7
  let discount_type_b : ℕ := 10
  let discount_type_c : ℕ := 12

  -- Calculate the selling price for each type of action figure
  let sell_price_a := value_type_a - discount_type_a
  let sell_price_b := value_type_b - discount_type_b
  let sell_price_c := value_type_c - discount_type_c

  -- Calculate the total earnings
  let total := num_type_a * sell_price_a + num_type_b * sell_price_b + num_type_c * sell_price_c

  exact total

/-- Theorem stating that Jimmy's total earnings is 229 --/
theorem jimmy_earnings_theorem : jimmy_total_earnings = 229 := by
  sorry

end jimmy_earnings_theorem_l2651_265135


namespace abs_negative_eleven_l2651_265146

theorem abs_negative_eleven : abs (-11 : ℤ) = 11 := by
  sorry

end abs_negative_eleven_l2651_265146


namespace sugar_concentration_of_second_solution_l2651_265198

/-- Given two solutions A and B, where:
    - A is 10% sugar by weight
    - B has an unknown sugar concentration
    - 3/4 of A is mixed with 1/4 of B
    - The resulting mixture is 16% sugar by weight
    This theorem proves that B must be 34% sugar by weight -/
theorem sugar_concentration_of_second_solution
  (W : ℝ) -- Total weight of the original solution
  (h_W_pos : W > 0) -- Assumption that W is positive
  : let A := 0.10 -- Sugar concentration of solution A (10%)
    let final_concentration := 0.16 -- Sugar concentration of final mixture (16%)
    let B := (4 * final_concentration - 3 * A) -- Sugar concentration of solution B
    B = 0.34 -- B is 34% sugar by weight
  := by sorry

end sugar_concentration_of_second_solution_l2651_265198


namespace total_cement_is_54_4_l2651_265183

/-- Amount of cement used for Lexi's street in tons -/
def lexis_cement : ℝ := 10

/-- Amount of cement used for Tess's street in tons -/
def tess_cement : ℝ := lexis_cement * 1.2

/-- Amount of cement used for Ben's street in tons -/
def bens_cement : ℝ := tess_cement * 0.9

/-- Amount of cement used for Olivia's street in tons -/
def olivias_cement : ℝ := bens_cement * 2

/-- Total amount of cement used for all four streets in tons -/
def total_cement : ℝ := lexis_cement + tess_cement + bens_cement + olivias_cement

theorem total_cement_is_54_4 : total_cement = 54.4 := by
  sorry

end total_cement_is_54_4_l2651_265183


namespace binary_sum_proof_l2651_265176

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ ⟨i, bit⟩ acc => acc + if bit then 2^i else 0) 0

theorem binary_sum_proof :
  let b1 := [true, true, false, true]  -- 1101₂
  let b2 := [true, false, true]        -- 101₂
  let b3 := [true, true, true]         -- 111₂
  let b4 := [true, false, false, false, true]  -- 10001₂
  let result := [false, true, false, true, false, true]  -- 101010₂
  binary_to_decimal b1 + binary_to_decimal b2 + binary_to_decimal b3 + binary_to_decimal b4 = binary_to_decimal result :=
by sorry

end binary_sum_proof_l2651_265176


namespace average_temperature_proof_l2651_265107

def daily_temperatures : List ℝ := [40, 50, 65, 36, 82, 72, 26]
def days_in_week : ℕ := 7

theorem average_temperature_proof :
  (daily_temperatures.sum / days_in_week : ℝ) = 53 := by
  sorry

end average_temperature_proof_l2651_265107


namespace arccos_equation_solution_l2651_265178

theorem arccos_equation_solution :
  ∃! x : ℝ, 
    x = 1 / (2 * Real.sqrt (19 - 8 * Real.sqrt 2)) ∧
    Real.arccos (4 * x) - Real.arccos (2 * x) = π / 4 ∧
    0 ≤ 4 * x ∧ 4 * x ≤ 1 ∧
    0 ≤ 2 * x ∧ 2 * x ≤ 1 :=
by sorry

end arccos_equation_solution_l2651_265178
