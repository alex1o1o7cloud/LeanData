import Mathlib

namespace NUMINAMATH_CALUDE_third_term_is_five_l2982_298289

-- Define the sequence a_n
def a (n : ℕ) : ℕ := sorry

-- Define the sum function S_n
def S (n : ℕ) : ℕ := n^2

-- State the theorem
theorem third_term_is_five :
  a 3 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_third_term_is_five_l2982_298289


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l2982_298223

theorem opposite_of_fraction (n : ℕ) (hn : n ≠ 0) :
  ∃ x : ℚ, (1 : ℚ) / n + x = 0 → x = -(1 : ℚ) / n := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l2982_298223


namespace NUMINAMATH_CALUDE_kelly_initial_apples_l2982_298219

theorem kelly_initial_apples (initial : ℕ) (to_pick : ℕ) (total : ℕ) 
  (h1 : to_pick = 49)
  (h2 : total = 105)
  (h3 : initial + to_pick = total) : 
  initial = 56 := by
  sorry

end NUMINAMATH_CALUDE_kelly_initial_apples_l2982_298219


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l2982_298296

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 4 / 5 →  -- angles are in ratio 4:5
  a = 80 :=  -- smaller angle is 80°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l2982_298296


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_for_abs_x_leq_one_l2982_298248

theorem sufficient_but_not_necessary_condition_for_abs_x_leq_one :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x| ≤ 1) ∧
  ¬(∀ x : ℝ, |x| ≤ 1 → 0 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_for_abs_x_leq_one_l2982_298248


namespace NUMINAMATH_CALUDE_number_difference_proof_l2982_298286

theorem number_difference_proof (x : ℚ) : x - (3/5) * x = 58 → x = 145 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l2982_298286


namespace NUMINAMATH_CALUDE_sphere_radius_from_hemisphere_volume_l2982_298234

/-- Given a sphere whose hemisphere has a volume of 36π cm³, prove that the radius of the sphere is 3 cm. -/
theorem sphere_radius_from_hemisphere_volume :
  ∀ r : ℝ, (2 / 3 * π * r^3 = 36 * π) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hemisphere_volume_l2982_298234


namespace NUMINAMATH_CALUDE_distance_run_in_two_hours_l2982_298207

/-- Given a person's running capabilities, calculate the distance they can run in 2 hours -/
theorem distance_run_in_two_hours 
  (distance : ℝ) -- The unknown distance the person can run in 2 hours
  (time_for_distance : ℝ) -- Time taken to run the unknown distance
  (track_length : ℝ) -- Length of the track
  (time_for_track : ℝ) -- Time taken to run the track
  (h1 : time_for_distance = 2) -- The person can run the unknown distance in 2 hours
  (h2 : track_length = 10000) -- The track is 10000 meters long
  (h3 : time_for_track = 10) -- It takes 10 hours to run the track
  (h4 : distance / time_for_distance = track_length / time_for_track) -- The speed is constant
  : distance = 2000 := by
  sorry

end NUMINAMATH_CALUDE_distance_run_in_two_hours_l2982_298207


namespace NUMINAMATH_CALUDE_pies_per_row_l2982_298259

theorem pies_per_row (total_pies : ℕ) (num_rows : ℕ) (h1 : total_pies = 30) (h2 : num_rows = 6) :
  total_pies / num_rows = 5 := by
sorry

end NUMINAMATH_CALUDE_pies_per_row_l2982_298259


namespace NUMINAMATH_CALUDE_opposite_pairs_l2982_298267

-- Define the pairs of numbers
def pair_A : ℚ × ℚ := (-5, 1/5)
def pair_B : ℤ × ℤ := (8, 8)
def pair_C : ℤ × ℤ := (-3, 3)
def pair_D : ℚ × ℚ := (7/2, 7/2)

-- Define what it means for two numbers to be opposite
def are_opposite (a b : ℚ) : Prop := a = -b

-- Theorem stating that pair C contains opposite numbers, while others do not
theorem opposite_pairs :
  (¬ are_opposite pair_A.1 pair_A.2) ∧
  (¬ are_opposite pair_B.1 pair_B.2) ∧
  (are_opposite pair_C.1 pair_C.2) ∧
  (¬ are_opposite pair_D.1 pair_D.2) :=
sorry

end NUMINAMATH_CALUDE_opposite_pairs_l2982_298267


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2982_298231

theorem inequality_equivalence (x y : ℝ) : y - x < Real.sqrt (x^2) ↔ y < 0 ∨ y < 2*x := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2982_298231


namespace NUMINAMATH_CALUDE_books_per_shelf_l2982_298261

theorem books_per_shelf 
  (mystery_shelves : ℕ) 
  (picture_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : mystery_shelves = 6) 
  (h2 : picture_shelves = 2) 
  (h3 : total_books = 72) : 
  total_books / (mystery_shelves + picture_shelves) = 9 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l2982_298261


namespace NUMINAMATH_CALUDE_system_solution_l2982_298204

theorem system_solution :
  ∃! (x y : ℝ), (x + 3 * y = 7) ∧ (x + 4 * y = 8) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_l2982_298204


namespace NUMINAMATH_CALUDE_peach_count_correct_l2982_298285

/-- The number of baskets -/
def total_baskets : ℕ := 150

/-- The number of peaches in each odd-numbered basket -/
def peaches_odd : ℕ := 14

/-- The number of peaches in each even-numbered basket -/
def peaches_even : ℕ := 12

/-- The total number of peaches -/
def total_peaches : ℕ := 1950

theorem peach_count_correct : 
  (total_baskets / 2) * peaches_odd + (total_baskets / 2) * peaches_even = total_peaches := by
  sorry

end NUMINAMATH_CALUDE_peach_count_correct_l2982_298285


namespace NUMINAMATH_CALUDE_pencil_count_l2982_298272

/-- Given a shop with pencils, pens, and exercise books in a ratio of 10 : 2 : 3,
    and 36 exercise books in total, prove that there are 120 pencils. -/
theorem pencil_count (ratio_pencils : ℕ) (ratio_pens : ℕ) (ratio_books : ℕ) 
    (total_books : ℕ) (h1 : ratio_pencils = 10) (h2 : ratio_pens = 2) 
    (h3 : ratio_books = 3) (h4 : total_books = 36) : 
    ratio_pencils * (total_books / ratio_books) = 120 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2982_298272


namespace NUMINAMATH_CALUDE_expression_factorization_l2982_298213

theorem expression_factorization (x : ℝ) : 
  (20 * x^3 + 100 * x - 10) - (-5 * x^3 + 5 * x - 10) = 5 * x * (5 * x^2 + 19) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l2982_298213


namespace NUMINAMATH_CALUDE_square_equality_l2982_298297

theorem square_equality (a b : ℝ) : a = b → a^2 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_l2982_298297


namespace NUMINAMATH_CALUDE_least_k_for_error_bound_l2982_298283

-- Define the sequence u_k
def u : ℕ → ℚ
  | 0 => 1/3
  | k+1 => 2.5 * u k - 3 * (u k)^2

-- Define the limit L
noncomputable def L : ℚ := 2/5

-- Define the error bound
def error_bound : ℚ := 1 / 2^500

-- Theorem statement
theorem least_k_for_error_bound :
  ∃ k : ℕ, (∀ j : ℕ, j < k → |u j - L| > error_bound) ∧
           |u k - L| ≤ error_bound ∧
           k = 5 := by sorry

end NUMINAMATH_CALUDE_least_k_for_error_bound_l2982_298283


namespace NUMINAMATH_CALUDE_son_age_l2982_298255

/-- Represents the ages of a father and son -/
structure Ages where
  father : ℕ
  son : ℕ

/-- The conditions of the age problem -/
def AgeConditions (ages : Ages) : Prop :=
  (ages.father + ages.son = 75) ∧
  (∃ (x : ℕ), ages.father = 8 * (ages.son - x) ∧ ages.father - x = ages.son)

/-- The theorem stating that under the given conditions, the son's age is 27 -/
theorem son_age (ages : Ages) (h : AgeConditions ages) : ages.son = 27 := by
  sorry

end NUMINAMATH_CALUDE_son_age_l2982_298255


namespace NUMINAMATH_CALUDE_saturn_diameter_times_ten_l2982_298212

/-- The diameter of Saturn in kilometers -/
def saturn_diameter : ℝ := 1.2 * 10^5

/-- Theorem stating the correct multiplication of Saturn's diameter by 10 -/
theorem saturn_diameter_times_ten :
  saturn_diameter * 10 = 1.2 * 10^6 := by
  sorry

end NUMINAMATH_CALUDE_saturn_diameter_times_ten_l2982_298212


namespace NUMINAMATH_CALUDE_function_equality_l2982_298222

theorem function_equality (x : ℝ) (h : x ≠ 0) : x^0 = x/x := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2982_298222


namespace NUMINAMATH_CALUDE_sales_tax_difference_l2982_298270

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) : 
  price = 50 → tax_rate1 = 0.075 → tax_rate2 = 0.07 →
  price * tax_rate1 - price * tax_rate2 = 0.25 := by sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l2982_298270


namespace NUMINAMATH_CALUDE_max_cubes_in_box_l2982_298278

theorem max_cubes_in_box (box_length box_width box_height cube_volume : ℕ) 
  (h1 : box_length = 8)
  (h2 : box_width = 9)
  (h3 : box_height = 12)
  (h4 : cube_volume = 27) :
  (box_length * box_width * box_height) / cube_volume = 32 := by
  sorry

#check max_cubes_in_box

end NUMINAMATH_CALUDE_max_cubes_in_box_l2982_298278


namespace NUMINAMATH_CALUDE_lake_pleasant_activities_l2982_298273

theorem lake_pleasant_activities (total_kids : ℕ) (tubing_fraction : ℚ) (rafting_fraction : ℚ) (kayaking_fraction : ℚ)
  (h_total : total_kids = 40)
  (h_tubing : tubing_fraction = 1/4)
  (h_rafting : rafting_fraction = 1/2)
  (h_kayaking : kayaking_fraction = 1/3) :
  ⌊(total_kids : ℚ) * tubing_fraction * rafting_fraction * kayaking_fraction⌋ = 1 := by
sorry

end NUMINAMATH_CALUDE_lake_pleasant_activities_l2982_298273


namespace NUMINAMATH_CALUDE_complex_power_simplification_l2982_298229

theorem complex_power_simplification :
  ((1 + 2 * Complex.I) / (1 - 2 * Complex.I)) ^ 1012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l2982_298229


namespace NUMINAMATH_CALUDE_advertising_cost_proof_l2982_298246

/-- Proves that the advertising cost is $1000 given the problem conditions -/
theorem advertising_cost_proof 
  (total_customers : ℕ) 
  (purchase_rate : ℚ) 
  (item_cost : ℕ) 
  (profit : ℕ) :
  total_customers = 100 →
  purchase_rate = 4/5 →
  item_cost = 25 →
  profit = 1000 →
  (total_customers : ℚ) * purchase_rate * item_cost - profit = 1000 :=
by sorry

end NUMINAMATH_CALUDE_advertising_cost_proof_l2982_298246


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l2982_298260

theorem purely_imaginary_condition (a : ℝ) :
  a = -1 ↔ (∃ b : ℝ, Complex.mk (a^2 - 1) (a - 1) = Complex.I * b) := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l2982_298260


namespace NUMINAMATH_CALUDE_rhombus_area_l2982_298217

/-- The area of a rhombus with diagonals of 14 cm and 20 cm is 140 square centimeters. -/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 140 := by
sorry

end NUMINAMATH_CALUDE_rhombus_area_l2982_298217


namespace NUMINAMATH_CALUDE_min_operations_for_2006_l2982_298206

/-- The minimal number of operations needed to calculate x^2006 -/
def min_operations : ℕ := 17

/-- A function that represents the number of operations needed to calculate x^n given x -/
noncomputable def operations (n : ℕ) : ℕ := sorry

/-- The theorem stating that the minimal number of operations to calculate x^2006 is 17 -/
theorem min_operations_for_2006 : operations 2006 = min_operations := by sorry

end NUMINAMATH_CALUDE_min_operations_for_2006_l2982_298206


namespace NUMINAMATH_CALUDE_range_of_a_l2982_298249

def A : Set ℝ := {x | x^2 - 5*x + 4 > 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + (a+2) = 0}

theorem range_of_a (a : ℝ) : 
  (A ∩ B a).Nonempty → a ∈ {x | x < -1 ∨ x > 18/7} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2982_298249


namespace NUMINAMATH_CALUDE_expr_D_not_complete_square_expr_A_is_complete_square_expr_B_is_complete_square_expr_C_is_complete_square_l2982_298205

-- Define the expressions
def expr_A (x : ℝ) := x^2 - 2*x + 1
def expr_B (x : ℝ) := 1 - 2*x + x^2
def expr_C (a b : ℝ) := a^2 + b^2 - 2*a*b
def expr_D (x : ℝ) := 4*x^2 + 4*x - 1

-- Define what it means for an expression to be factored as a complete square
def is_complete_square (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * (x - b)^2

-- Theorem stating that expr_D cannot be factored as a complete square
theorem expr_D_not_complete_square :
  ¬ is_complete_square expr_D :=
sorry

-- Theorems stating that the other expressions can be factored as complete squares
theorem expr_A_is_complete_square :
  is_complete_square expr_A :=
sorry

theorem expr_B_is_complete_square :
  is_complete_square expr_B :=
sorry

theorem expr_C_is_complete_square :
  ∃ (f : ℝ → ℝ → ℝ), ∀ a b, expr_C a b = f a b ∧ is_complete_square (f a) :=
sorry

end NUMINAMATH_CALUDE_expr_D_not_complete_square_expr_A_is_complete_square_expr_B_is_complete_square_expr_C_is_complete_square_l2982_298205


namespace NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_fifteen_l2982_298276

/-- The constant term in the expansion of (x - 1/x^2)^6 -/
theorem constant_term_expansion : ℕ :=
  let n : ℕ := 6
  let k : ℕ := 2
  Nat.choose n k

/-- The constant term in the expansion of (x - 1/x^2)^6 is 15 -/
theorem constant_term_is_fifteen : constant_term_expansion = 15 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_fifteen_l2982_298276


namespace NUMINAMATH_CALUDE_elective_courses_schemes_l2982_298232

theorem elective_courses_schemes (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 10 → k = 3 → m = 3 →
  (Nat.choose (n - m) k + m * Nat.choose (n - m) (k - 1) = 98) :=
by sorry

end NUMINAMATH_CALUDE_elective_courses_schemes_l2982_298232


namespace NUMINAMATH_CALUDE_eleven_subtractions_to_zero_l2982_298256

def digit_sum (n : ℕ) : ℕ := sorry

def subtract_digit_sum (n : ℕ) : ℕ := n - digit_sum n

def repeat_subtract_digit_sum (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => repeat_subtract_digit_sum (subtract_digit_sum n) k

theorem eleven_subtractions_to_zero (n : ℕ) (h : 100 ≤ n ∧ n ≤ 109) :
  repeat_subtract_digit_sum n 11 = 0 := by sorry

end NUMINAMATH_CALUDE_eleven_subtractions_to_zero_l2982_298256


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2982_298268

-- Define the function f(x) = x^3 - 3x + 2
def f (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Define the closed interval [0, 3]
def interval : Set ℝ := Set.Icc 0 3

-- Theorem statement
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), max = 20 ∧ min = 0 ∧
  (∀ x ∈ interval, f x ≤ max) ∧
  (∃ x ∈ interval, f x = max) ∧
  (∀ x ∈ interval, min ≤ f x) ∧
  (∃ x ∈ interval, f x = min) := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2982_298268


namespace NUMINAMATH_CALUDE_max_sum_with_gcf_six_l2982_298238

theorem max_sum_with_gcf_six (a b : ℕ) : 
  10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 →  -- a and b are two-digit positive integers
  Nat.gcd a b = 6 →                    -- greatest common factor of a and b is 6
  a + b ≤ 186 ∧                        -- upper bound
  ∃ (a' b' : ℕ), 10 ≤ a' ∧ a' ≤ 99 ∧ 10 ≤ b' ∧ b' ≤ 99 ∧ 
    Nat.gcd a' b' = 6 ∧ a' + b' = 186  -- existence of a pair that achieves the maximum
  := by sorry

end NUMINAMATH_CALUDE_max_sum_with_gcf_six_l2982_298238


namespace NUMINAMATH_CALUDE_equation_transformation_l2982_298241

theorem equation_transformation (x y : ℝ) (h : 2*x - 3*y + 6 = 0) : 
  6*x - 9*y + 6 = -12 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l2982_298241


namespace NUMINAMATH_CALUDE_vessel_width_calculation_l2982_298247

/-- Proves that given a cube with edge length 15 cm immersed in a rectangular vessel 
    with base length 20 cm, if the water level rises by 11.25 cm, 
    then the width of the vessel's base is 15 cm. -/
theorem vessel_width_calculation (cube_edge : ℝ) (vessel_length : ℝ) (water_rise : ℝ) :
  cube_edge = 15 →
  vessel_length = 20 →
  water_rise = 11.25 →
  (cube_edge ^ 3) = (vessel_length * (cube_edge ^ 3 / (vessel_length * water_rise))) * water_rise →
  cube_edge ^ 3 / (vessel_length * water_rise) = 15 := by
  sorry

#check vessel_width_calculation

end NUMINAMATH_CALUDE_vessel_width_calculation_l2982_298247


namespace NUMINAMATH_CALUDE_common_factor_implies_a_values_l2982_298228

theorem common_factor_implies_a_values (a : ℝ) :
  (∃ (p : ℝ) (A B : ℝ → ℝ), p ≠ 0 ∧
    (∀ x, x^3 - x - a = A x * (x + p)) ∧
    (∀ x, x^2 + x - a = B x * (x + p))) →
  (a = 0 ∨ a = 10 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_common_factor_implies_a_values_l2982_298228


namespace NUMINAMATH_CALUDE_max_value_2x_plus_y_l2982_298236

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2*y ≤ 3) (h2 : x ≥ 0) (h3 : y ≥ 0) :
  (∀ x' y' : ℝ, x' + 2*y' ≤ 3 → x' ≥ 0 → y' ≥ 0 → 2*x' + y' ≤ 2*x + y) →
  2*x + y = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_plus_y_l2982_298236


namespace NUMINAMATH_CALUDE_intersection_points_concyclic_l2982_298290

/-- A circle in which quadrilateral ABCD is inscribed -/
structure CircumCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- A convex quadrilateral ABCD inscribed in a circle -/
structure InscribedQuadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  circle : CircumCircle

/-- Circles drawn with each side of ABCD as a chord -/
structure SideCircles where
  AB : CircumCircle
  BC : CircumCircle
  CD : CircumCircle
  DA : CircumCircle

/-- Intersection points of circles drawn over adjacent sides -/
structure IntersectionPoints where
  A1 : ℝ × ℝ
  B1 : ℝ × ℝ
  C1 : ℝ × ℝ
  D1 : ℝ × ℝ

/-- Function to check if four points are concyclic -/
def areConcyclic (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

theorem intersection_points_concyclic 
  (quad : InscribedQuadrilateral) 
  (sides : SideCircles) 
  (points : IntersectionPoints) : 
  areConcyclic points.A1 points.B1 points.C1 points.D1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_concyclic_l2982_298290


namespace NUMINAMATH_CALUDE_other_divisor_proof_l2982_298202

theorem other_divisor_proof (x : ℕ) : x = 5 ↔ 
  x ≠ 11 ∧ 
  x > 0 ∧
  (386 % x = 1 ∧ 386 % 11 = 1) ∧
  ∀ y : ℕ, y < x → y ≠ 11 → y > 0 → (386 % y = 1 ∧ 386 % 11 = 1) → False :=
by sorry

end NUMINAMATH_CALUDE_other_divisor_proof_l2982_298202


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l2982_298254

theorem rectangle_width_decrease (L W : ℝ) (h1 : L > 0) (h2 : W > 0) : 
  let new_length := 1.5 * L
  let new_width := W * (L / new_length)
  let percent_decrease := (W - new_width) / W * 100
  percent_decrease = 100/3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l2982_298254


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l2982_298250

theorem power_tower_mod_500 : 2^(2^(2^2)) ≡ 36 [ZMOD 500] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l2982_298250


namespace NUMINAMATH_CALUDE_min_liking_both_mozart_and_bach_l2982_298214

theorem min_liking_both_mozart_and_bach
  (total : ℕ)
  (like_mozart : ℕ)
  (like_bach : ℕ)
  (h_total : total = 200)
  (h_mozart : like_mozart = 160)
  (h_bach : like_bach = 150) :
  like_mozart + like_bach - total ≥ 110 :=
by sorry

end NUMINAMATH_CALUDE_min_liking_both_mozart_and_bach_l2982_298214


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2982_298299

/-- Given vectors a and b, where b and b-a are collinear, prove |a+b| = 3√5/2 -/
theorem vector_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![1, 2]
  (∃ (k : ℝ), b = k • (b - a)) →
  ‖a + b‖ = 3 * Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2982_298299


namespace NUMINAMATH_CALUDE_cyclist_speed_north_cyclist_speed_north_proof_l2982_298264

/-- The speed of the cyclist going north, given two cyclists starting from the same place
    in opposite directions, with one going south at 25 km/h, and they take 1.4285714285714286 hours
    to be 50 km apart. -/
theorem cyclist_speed_north : ℝ → Prop :=
  fun v : ℝ =>
    let south_speed : ℝ := 25
    let time : ℝ := 1.4285714285714286
    let distance : ℝ := 50
    v > 0 ∧ distance = (v + south_speed) * time → v = 10

/-- Proof of the cyclist_speed_north theorem -/
theorem cyclist_speed_north_proof : cyclist_speed_north 10 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_north_cyclist_speed_north_proof_l2982_298264


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2982_298230

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (λ acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert ternary to decimal
def ternary_to_decimal (ternary : List ℕ) : ℕ :=
  ternary.enum.foldl (λ acc (i, d) => acc + d * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary := [false, true, false, true]  -- 1010 in binary
  let ternary := [2, 0, 1]  -- 102 in ternary
  (binary_to_decimal binary) * (ternary_to_decimal ternary) = 110 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l2982_298230


namespace NUMINAMATH_CALUDE_product_of_complex_in_polar_form_specific_complex_product_l2982_298203

/-- 
Given two complex numbers in polar form, prove that their product 
is equal to the product of their magnitudes and the sum of their angles.
-/
theorem product_of_complex_in_polar_form 
  (z₁ : ℂ) (z₂ : ℂ) (r₁ r₂ θ₁ θ₂ : ℝ) :
  z₁ = r₁ * Complex.exp (θ₁ * Complex.I) →
  z₂ = r₂ * Complex.exp (θ₂ * Complex.I) →
  r₁ > 0 →
  r₂ > 0 →
  z₁ * z₂ = (r₁ * r₂) * Complex.exp ((θ₁ + θ₂) * Complex.I) :=
by sorry

/-- 
Prove that the product of 5cis(25°) and 4cis(48°) is equal to 20cis(73°).
-/
theorem specific_complex_product :
  let z₁ : ℂ := 5 * Complex.exp (25 * π / 180 * Complex.I)
  let z₂ : ℂ := 4 * Complex.exp (48 * π / 180 * Complex.I)
  z₁ * z₂ = 20 * Complex.exp (73 * π / 180 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_product_of_complex_in_polar_form_specific_complex_product_l2982_298203


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l2982_298251

theorem rectangle_area_diagonal_relation :
  ∀ (length width diagonal : ℝ),
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 5 / 2 →
  length ^ 2 + width ^ 2 = diagonal ^ 2 →
  diagonal = 13 →
  ∃ (k : ℝ), length * width = k * diagonal ^ 2 ∧ k = 10 / 29 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l2982_298251


namespace NUMINAMATH_CALUDE_triangle_inequality_squares_l2982_298257

theorem triangle_inequality_squares (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + a * c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_squares_l2982_298257


namespace NUMINAMATH_CALUDE_prob_select_dime_l2982_298243

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The total value of quarters in the container in dollars -/
def total_quarters_value : ℚ := 12.50

/-- The total value of nickels in the container in dollars -/
def total_nickels_value : ℚ := 15.00

/-- The total value of dimes in the container in dollars -/
def total_dimes_value : ℚ := 5.00

/-- The probability of randomly selecting a dime from the container -/
theorem prob_select_dime : 
  (total_dimes_value / dime_value) / 
  ((total_quarters_value / quarter_value) + 
   (total_nickels_value / nickel_value) + 
   (total_dimes_value / dime_value)) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_prob_select_dime_l2982_298243


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2982_298263

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_chord : 4 * a + 2 * b = 2) : 
  (1 / a + 2 / b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4 * a₀ + 2 * b₀ = 2 ∧ 1 / a₀ + 2 / b₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2982_298263


namespace NUMINAMATH_CALUDE_bob_finishes_24_minutes_after_alice_l2982_298282

/-- Represents the race scenario -/
structure RaceScenario where
  distance : ℕ  -- Race distance in miles
  alice_speed : ℕ  -- Alice's speed in minutes per mile
  bob_speed : ℕ  -- Bob's speed in minutes per mile

/-- Calculates the time difference between Alice and Bob finishing the race -/
def finish_time_difference (race : RaceScenario) : ℕ :=
  race.distance * race.bob_speed - race.distance * race.alice_speed

/-- Theorem stating that in the given race scenario, Bob finishes 24 minutes after Alice -/
theorem bob_finishes_24_minutes_after_alice :
  let race := RaceScenario.mk 12 7 9
  finish_time_difference race = 24 := by
  sorry

end NUMINAMATH_CALUDE_bob_finishes_24_minutes_after_alice_l2982_298282


namespace NUMINAMATH_CALUDE_travel_equations_correct_l2982_298253

/-- Represents the travel scenario with bike riding and walking -/
structure TravelScenario where
  total_time : ℝ
  total_distance : ℝ
  bike_speed : ℝ
  walk_speed : ℝ
  bike_time : ℝ
  walk_time : ℝ

/-- The given travel scenario matches the system of equations -/
def scenario_matches_equations (s : TravelScenario) : Prop :=
  s.total_time = 1.5 ∧
  s.total_distance = 20 ∧
  s.bike_speed = 15 ∧
  s.walk_speed = 5 ∧
  s.bike_time + s.walk_time = s.total_time ∧
  s.bike_speed * s.bike_time + s.walk_speed * s.walk_time = s.total_distance

/-- The system of equations correctly represents the travel scenario -/
theorem travel_equations_correct (s : TravelScenario) :
  scenario_matches_equations s →
  s.bike_time + s.walk_time = 1.5 ∧
  15 * s.bike_time + 5 * s.walk_time = 20 :=
by sorry

end NUMINAMATH_CALUDE_travel_equations_correct_l2982_298253


namespace NUMINAMATH_CALUDE_connected_graphs_lower_bound_l2982_298281

/-- The number of connected labeled graphs on n vertices -/
def g (n : ℕ) : ℕ := sorry

/-- The total number of labeled graphs on n vertices -/
def total_graphs (n : ℕ) : ℕ := 2^(n * (n - 1) / 2)

/-- Theorem: The number of connected labeled graphs on n vertices is at least half of the total number of labeled graphs on n vertices -/
theorem connected_graphs_lower_bound (n : ℕ) : g n ≥ total_graphs n / 2 := by sorry

end NUMINAMATH_CALUDE_connected_graphs_lower_bound_l2982_298281


namespace NUMINAMATH_CALUDE_no_inscribed_triangle_with_sine_roots_l2982_298221

theorem no_inscribed_triangle_with_sine_roots :
  ¬ ∃ (a b c : ℝ) (A B C : ℝ),
    0 < a ∧ 0 < b ∧ 0 < c ∧
    0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
    A + B + C = π ∧
    a = 2 * Real.sin (A / 2) ∧
    b = 2 * Real.sin (B / 2) ∧
    c = 2 * Real.sin (C / 2) ∧
    ∃ (p : ℝ),
      (Real.sin A)^3 - 2 * a * (Real.sin A)^2 + b * c * Real.sin A = p ∧
      (Real.sin B)^3 - 2 * a * (Real.sin B)^2 + b * c * Real.sin B = p ∧
      (Real.sin C)^3 - 2 * a * (Real.sin C)^2 + b * c * Real.sin C = p :=
by sorry

end NUMINAMATH_CALUDE_no_inscribed_triangle_with_sine_roots_l2982_298221


namespace NUMINAMATH_CALUDE_relay_race_distance_l2982_298240

theorem relay_race_distance (siwon_fraction dawon_fraction : ℚ) 
  (combined_distance : ℝ) (total_distance : ℝ) : 
  siwon_fraction = 3 / 10 →
  dawon_fraction = 4 / 10 →
  combined_distance = 140 →
  (siwon_fraction + dawon_fraction : ℝ) * total_distance = combined_distance →
  total_distance = 200 :=
by sorry

end NUMINAMATH_CALUDE_relay_race_distance_l2982_298240


namespace NUMINAMATH_CALUDE_solve_equation_l2982_298287

theorem solve_equation (k l x : ℝ) : 
  (2 : ℝ) / 3 = k / 54 ∧ 
  (2 : ℝ) / 3 = (k + l) / 90 ∧ 
  (2 : ℝ) / 3 = (x - l) / 150 → 
  x = 106 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2982_298287


namespace NUMINAMATH_CALUDE_triangle_angle_value_l2982_298237

theorem triangle_angle_value (A B C : Real) : 
  -- A, B, and C are internal angles of a triangle
  A + B + C = π → 
  0 < A → 0 < B → 0 < C →
  -- Given equation
  Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin C ^ 2 + Real.sin A * Real.sin B →
  -- Conclusion
  C = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l2982_298237


namespace NUMINAMATH_CALUDE_forgot_homework_percentage_l2982_298291

/-- Represents the percentage of students who forgot their homework in group B -/
def percentage_forgot_B : ℝ := 15

theorem forgot_homework_percentage :
  let total_students : ℕ := 100
  let group_A_students : ℕ := 20
  let group_B_students : ℕ := 80
  let percentage_forgot_A : ℝ := 20
  let percentage_forgot_total : ℝ := 16
  percentage_forgot_B = ((percentage_forgot_total * total_students) - 
                         (percentage_forgot_A * group_A_students)) / group_B_students * 100 :=
by sorry

end NUMINAMATH_CALUDE_forgot_homework_percentage_l2982_298291


namespace NUMINAMATH_CALUDE_weight_of_B_l2982_298266

def weight_problem (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧
  (A + B) / 2 = 40 ∧
  (B + C) / 2 = 41 ∧
  ∃ (x : ℝ), A = 2*x ∧ B = 3*x ∧ C = 5*x ∧
  A + B + C = 144

theorem weight_of_B (A B C : ℝ) (h : weight_problem A B C) : B = 43.2 :=
sorry

end NUMINAMATH_CALUDE_weight_of_B_l2982_298266


namespace NUMINAMATH_CALUDE_cost_of_paints_paint_cost_is_five_l2982_298215

theorem cost_of_paints (classes : ℕ) (folders_per_class : ℕ) (pencils_per_class : ℕ) 
  (pencils_per_eraser : ℕ) (folder_cost : ℕ) (pencil_cost : ℕ) (eraser_cost : ℕ) 
  (total_spent : ℕ) : ℕ :=
  let total_folders := classes * folders_per_class
  let total_pencils := classes * pencils_per_class
  let total_erasers := total_pencils / pencils_per_eraser
  let folder_expense := total_folders * folder_cost
  let pencil_expense := total_pencils * pencil_cost
  let eraser_expense := total_erasers * eraser_cost
  let total_expense := folder_expense + pencil_expense + eraser_expense
  total_spent - total_expense

theorem paint_cost_is_five :
  cost_of_paints 6 1 3 6 6 2 1 80 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_paints_paint_cost_is_five_l2982_298215


namespace NUMINAMATH_CALUDE_remainder_2457633_div_25_l2982_298269

theorem remainder_2457633_div_25 : 2457633 % 25 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2457633_div_25_l2982_298269


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l2982_298275

theorem muffin_banana_price_ratio : 
  ∀ (muffin_price banana_price : ℝ),
  (5 * muffin_price + 4 * banana_price > 0) →
  (3 * (5 * muffin_price + 4 * banana_price) = 3 * muffin_price + 20 * banana_price) →
  (muffin_price / banana_price = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l2982_298275


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l2982_298218

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the exponentiation operation
def pow (base : ℕ) (exp : ℕ) : ℕ := base ^ exp

-- Theorem statement
theorem units_digit_sum_of_powers : 
  unitsDigit (pow 3 2014 + pow 4 2015 + pow 5 2016) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l2982_298218


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2982_298288

theorem min_value_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 9) :
  2/a + 2/b + 2/c ≥ 2 ∧ 
  (2/a + 2/b + 2/c = 2 ↔ a = 3 ∧ b = 3 ∧ c = 3) := by
  sorry

#check min_value_reciprocal_sum

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2982_298288


namespace NUMINAMATH_CALUDE_miran_has_fewest_paper_l2982_298208

def miran_paper : ℕ := 6
def junga_paper : ℕ := 13
def minsu_paper : ℕ := 10

theorem miran_has_fewest_paper :
  miran_paper ≤ junga_paper ∧ miran_paper ≤ minsu_paper :=
sorry

end NUMINAMATH_CALUDE_miran_has_fewest_paper_l2982_298208


namespace NUMINAMATH_CALUDE_triangle_inequality_l2982_298279

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  can_form_triangle 2 6 6 ∧
  ¬can_form_triangle 2 6 2 ∧
  ¬can_form_triangle 2 6 4 ∧
  ¬can_form_triangle 2 6 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2982_298279


namespace NUMINAMATH_CALUDE_books_vs_figures_difference_l2982_298262

theorem books_vs_figures_difference :
  ∀ (initial_figures initial_books added_figures : ℕ),
    initial_figures = 2 →
    initial_books = 10 →
    added_figures = 4 →
    initial_books - (initial_figures + added_figures) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_books_vs_figures_difference_l2982_298262


namespace NUMINAMATH_CALUDE_car_price_calculation_l2982_298210

/-- Represents the price of a car given loan terms and payments -/
def carPrice (loanYears : ℕ) (downPayment : ℚ) (monthlyPayment : ℚ) : ℚ :=
  downPayment + (loanYears * 12 : ℕ) * monthlyPayment

/-- Theorem stating that given the specific loan terms, the car price is $20,000 -/
theorem car_price_calculation :
  let loanYears : ℕ := 5
  let downPayment : ℚ := 5000
  let monthlyPayment : ℚ := 250
  carPrice loanYears downPayment monthlyPayment = 20000 := by
  sorry

#eval carPrice 5 5000 250

end NUMINAMATH_CALUDE_car_price_calculation_l2982_298210


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2982_298201

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, |2*x - 3| < 1 → x*(x - 3) < 0) ∧
  (∃ x : ℝ, x*(x - 3) < 0 ∧ ¬(|2*x - 3| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2982_298201


namespace NUMINAMATH_CALUDE_inequality_proof_l2982_298274

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y / z) + (y^2 * z / x) + (z^2 * x / y) ≥ x^2 + y^2 + z^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2982_298274


namespace NUMINAMATH_CALUDE_trip_distance_calculation_l2982_298211

theorem trip_distance_calculation (total_distance : ℝ) (speed1 speed2 avg_speed : ℝ) 
  (h1 : total_distance = 70)
  (h2 : speed1 = 48)
  (h3 : speed2 = 24)
  (h4 : avg_speed = 32) :
  ∃ (first_part : ℝ),
    first_part = 35 ∧
    first_part / speed1 + (total_distance - first_part) / speed2 = total_distance / avg_speed :=
by sorry

end NUMINAMATH_CALUDE_trip_distance_calculation_l2982_298211


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2982_298200

theorem min_value_of_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2*y = 1) :
  ∃ (m : ℝ), m = 3/4 ∧ ∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*y'^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2982_298200


namespace NUMINAMATH_CALUDE_common_chord_length_l2982_298245

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 2*y - 40 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Theorem statement
theorem common_chord_length :
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l2982_298245


namespace NUMINAMATH_CALUDE_expression_evaluation_l2982_298252

theorem expression_evaluation : (1/3)⁻¹ - 2 * Real.cos (30 * π / 180) - |2 - Real.sqrt 3| - (4 - Real.pi)^0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2982_298252


namespace NUMINAMATH_CALUDE_population_exceeds_target_in_2075_l2982_298225

/-- The initial population of Nisos in the year 2000 -/
def initial_population : ℕ := 500

/-- The year when the population count starts -/
def start_year : ℕ := 2000

/-- The number of years it takes for the population to triple -/
def tripling_period : ℕ := 25

/-- The target population we want to exceed -/
def target_population : ℕ := 9000

/-- Calculate the population after a given number of tripling periods -/
def population_after (periods : ℕ) : ℕ :=
  initial_population * (3 ^ periods)

/-- Calculate the year after a given number of tripling periods -/
def year_after (periods : ℕ) : ℕ :=
  start_year + tripling_period * periods

/-- The theorem to be proved -/
theorem population_exceeds_target_in_2075 :
  ∃ n : ℕ, year_after n = 2075 ∧ 
    population_after n > target_population ∧
    population_after (n - 1) ≤ target_population :=
by
  sorry


end NUMINAMATH_CALUDE_population_exceeds_target_in_2075_l2982_298225


namespace NUMINAMATH_CALUDE_meal_cost_l2982_298265

theorem meal_cost (total_bill : ℝ) (tip_percentage : ℝ) (payment : ℝ) (change : ℝ) :
  total_bill = 2.5 →
  tip_percentage = 0.2 →
  payment = 20 →
  change = 5 →
  ∃ (meal_cost : ℝ), meal_cost = 12.5 ∧ meal_cost + tip_percentage * meal_cost = payment - change :=
by sorry

end NUMINAMATH_CALUDE_meal_cost_l2982_298265


namespace NUMINAMATH_CALUDE_tetra_edge_is_2sqrt3_l2982_298216

/-- Configuration of five mutually tangent spheres with a circumscribed tetrahedron -/
structure SphereTetConfig where
  /-- Radius of each sphere -/
  r : ℝ
  /-- Centers of the four bottom spheres -/
  bottom_centers : Fin 4 → ℝ × ℝ × ℝ
  /-- Center of the top sphere -/
  top_center : ℝ × ℝ × ℝ
  /-- Vertices of the tetrahedron -/
  tetra_vertices : Fin 4 → ℝ × ℝ × ℝ

/-- The spheres are mutually tangent and properly configured -/
def is_valid_config (cfg : SphereTetConfig) : Prop :=
  cfg.r = 2 ∧
  ∀ i j, i ≠ j → dist (cfg.bottom_centers i) (cfg.bottom_centers j) = 4 ∧
  ∀ i, dist (cfg.bottom_centers i) cfg.top_center = 4 ∧
  cfg.top_center.2 = 2 ∧
  cfg.tetra_vertices 0 = cfg.top_center ∧
  ∀ i : Fin 3, cfg.tetra_vertices (i + 1) = cfg.bottom_centers i

/-- The edge length of the tetrahedron -/
def tetra_edge_length (cfg : SphereTetConfig) : ℝ :=
  dist (cfg.tetra_vertices 0) (cfg.tetra_vertices 1)

/-- Main theorem: The edge length of the tetrahedron is 2√3 -/
theorem tetra_edge_is_2sqrt3 (cfg : SphereTetConfig) (h : is_valid_config cfg) :
  tetra_edge_length cfg = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tetra_edge_is_2sqrt3_l2982_298216


namespace NUMINAMATH_CALUDE_additional_houses_built_l2982_298227

/-- Proves the number of additional houses built between the first half of the year and October -/
theorem additional_houses_built
  (total_houses : ℕ)
  (first_half_fraction : ℚ)
  (remaining_houses : ℕ)
  (h1 : total_houses = 2000)
  (h2 : first_half_fraction = 3/5)
  (h3 : remaining_houses = 500) :
  (total_houses - remaining_houses) - (first_half_fraction * total_houses) = 300 := by
  sorry

end NUMINAMATH_CALUDE_additional_houses_built_l2982_298227


namespace NUMINAMATH_CALUDE_central_angle_approx_longitude_diff_l2982_298293

/-- Represents a point on Earth's surface --/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Calculates the central angle between two points on Earth's surface,
    assuming Earth is a perfect sphere --/
def centralAngle (p1 p2 : EarthPoint) : Real :=
  sorry

theorem central_angle_approx_longitude_diff
  (L M : EarthPoint)
  (h1 : L.latitude = 0)
  (h2 : L.longitude = 45)
  (h3 : M.latitude = 23.5)
  (h4 : M.longitude = -90)
  (h5 : abs M.latitude < 30) :
  abs (centralAngle L M - 135) < 5 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_approx_longitude_diff_l2982_298293


namespace NUMINAMATH_CALUDE_cubic_polynomials_constant_term_l2982_298220

/-- Given two cubic polynomials p(x) and q(x) with specific root relationships,
    prove that there are only two possible values for the constant term d of p(x). -/
theorem cubic_polynomials_constant_term (c d : ℝ) : 
  (∃ (r s : ℝ), (r^3 + c*r + d = 0 ∧ s^3 + c*s + d = 0) ∧
   ((r+5)^3 + c*(r+5) + (d+210) = 0 ∧ (s-4)^3 + c*(s-4) + (d+210) = 0)) →
  (d = 240 ∨ d = 420) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_constant_term_l2982_298220


namespace NUMINAMATH_CALUDE_quadratic_one_root_l2982_298226

theorem quadratic_one_root (m : ℝ) (h : m > 0) :
  (∃! x : ℝ, x^2 + 4*m*x + m = 0) ↔ m = 1/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l2982_298226


namespace NUMINAMATH_CALUDE_louises_initial_toys_l2982_298284

/-- Proves that Louise initially had 28 toys in her cart -/
theorem louises_initial_toys (initial_toy_cost : ℕ) (teddy_bear_count : ℕ) (teddy_bear_cost : ℕ) (total_cost : ℕ) :
  initial_toy_cost = 10 →
  teddy_bear_count = 20 →
  teddy_bear_cost = 15 →
  total_cost = 580 →
  ∃ (initial_toy_count : ℕ), initial_toy_count * initial_toy_cost + teddy_bear_count * teddy_bear_cost = total_cost ∧ initial_toy_count = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_louises_initial_toys_l2982_298284


namespace NUMINAMATH_CALUDE_rational_inequality_l2982_298233

theorem rational_inequality (a b c d : ℚ) 
  (h : a^3 - 2005 = b^3 + 2027 ∧ 
       b^3 + 2027 = c^3 - 2822 ∧ 
       c^3 - 2822 = d^3 + 2820) : 
  c > a ∧ a > b ∧ b > d := by
sorry

end NUMINAMATH_CALUDE_rational_inequality_l2982_298233


namespace NUMINAMATH_CALUDE_four_fours_exist_l2982_298239

/-- A datatype representing arithmetic expressions using only the digit 4 --/
inductive Expr4
  | four : Expr4
  | add : Expr4 → Expr4 → Expr4
  | sub : Expr4 → Expr4 → Expr4
  | mul : Expr4 → Expr4 → Expr4
  | div : Expr4 → Expr4 → Expr4

/-- Evaluate an Expr4 to a rational number --/
def eval : Expr4 → ℚ
  | Expr4.four => 4
  | Expr4.add e1 e2 => eval e1 + eval e2
  | Expr4.sub e1 e2 => eval e1 - eval e2
  | Expr4.mul e1 e2 => eval e1 * eval e2
  | Expr4.div e1 e2 => eval e1 / eval e2

/-- Count the number of 4's used in an Expr4 --/
def count_fours : Expr4 → ℕ
  | Expr4.four => 1
  | Expr4.add e1 e2 => count_fours e1 + count_fours e2
  | Expr4.sub e1 e2 => count_fours e1 + count_fours e2
  | Expr4.mul e1 e2 => count_fours e1 + count_fours e2
  | Expr4.div e1 e2 => count_fours e1 + count_fours e2

/-- Theorem stating that expressions for 2, 3, 4, 5, and 6 exist using four 4's --/
theorem four_fours_exist : 
  ∃ (e2 e3 e4 e5 e6 : Expr4), 
    (count_fours e2 = 4 ∧ eval e2 = 2) ∧
    (count_fours e3 = 4 ∧ eval e3 = 3) ∧
    (count_fours e4 = 4 ∧ eval e4 = 4) ∧
    (count_fours e5 = 4 ∧ eval e5 = 5) ∧
    (count_fours e6 = 4 ∧ eval e6 = 6) := by
  sorry

end NUMINAMATH_CALUDE_four_fours_exist_l2982_298239


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2982_298244

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), 
    x^6 + x^3 + x^3*y + y = 147^157 ∧
    x^3 + x^3*y + y^2 + y + z^9 = 157^147 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2982_298244


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2982_298242

/-- The interval of segmentation for systematic sampling -/
def interval_of_segmentation (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The interval of segmentation for a population of 1200 and sample size of 40 is 30 -/
theorem systematic_sampling_interval :
  interval_of_segmentation 1200 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2982_298242


namespace NUMINAMATH_CALUDE_congruence_problem_l2982_298235

theorem congruence_problem (y : ℤ) 
  (h1 : (4 + y) % (4^3) = 3^2 % (4^3))
  (h2 : (6 + y) % (6^3) = 4^2 % (6^3))
  (h3 : (8 + y) % (8^3) = 6^2 % (8^3)) :
  y % 168 = 4 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l2982_298235


namespace NUMINAMATH_CALUDE_adults_in_sleeper_class_l2982_298298

def total_passengers : ℕ := 320
def adult_percentage : ℚ := 75 / 100
def sleeper_adult_percentage : ℚ := 15 / 100

theorem adults_in_sleeper_class : 
  ⌊(total_passengers : ℚ) * adult_percentage * sleeper_adult_percentage⌋ = 36 := by
  sorry

end NUMINAMATH_CALUDE_adults_in_sleeper_class_l2982_298298


namespace NUMINAMATH_CALUDE_mirasol_initial_balance_l2982_298292

/-- Mirasol's initial account balance -/
def initial_balance : ℕ := sorry

/-- Amount spent on coffee beans -/
def coffee_cost : ℕ := 10

/-- Amount spent on tumbler -/
def tumbler_cost : ℕ := 30

/-- Amount left in account -/
def remaining_balance : ℕ := 10

/-- Theorem: Mirasol's initial account balance was $50 -/
theorem mirasol_initial_balance :
  initial_balance = coffee_cost + tumbler_cost + remaining_balance :=
by sorry

end NUMINAMATH_CALUDE_mirasol_initial_balance_l2982_298292


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2982_298271

theorem quadratic_one_solution (q : ℝ) (hq : q ≠ 0) :
  (∃! x, q * x^2 - 18 * x + 8 = 0) ↔ q = 81/8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2982_298271


namespace NUMINAMATH_CALUDE_min_jugs_proof_l2982_298209

/-- The capacity of each jug in ounces -/
def jug_capacity : ℕ := 16

/-- The capacity of the container to be filled in ounces -/
def container_capacity : ℕ := 200

/-- The minimum number of jugs needed to fill or exceed the container capacity -/
def min_jugs : ℕ := 13

theorem min_jugs_proof :
  (∀ n : ℕ, n < min_jugs → n * jug_capacity < container_capacity) ∧
  min_jugs * jug_capacity ≥ container_capacity :=
sorry

end NUMINAMATH_CALUDE_min_jugs_proof_l2982_298209


namespace NUMINAMATH_CALUDE_books_difference_l2982_298277

def summer_reading (june july august : ℕ) : Prop :=
  june = 8 ∧ july = 2 * june ∧ june + july + august = 37

theorem books_difference (june july august : ℕ) 
  (h : summer_reading june july august) : july - august = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_difference_l2982_298277


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l2982_298224

theorem cycle_price_calculation (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 1125)
  (h2 : gain_percentage = 25) : 
  ∃ original_price : ℝ, 
    original_price * (1 + gain_percentage / 100) = selling_price ∧ 
    original_price = 900 := by
sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l2982_298224


namespace NUMINAMATH_CALUDE_valid_numbers_l2982_298280

def is_valid_number (n : ℕ) : Prop :=
  100000 > n ∧ n ≥ 10000 ∧  -- five-digit number
  n % 72 = 0 ∧  -- divisible by 72
  (n.digits 10).count 1 = 3  -- exactly three digits are 1

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {41112, 14112, 11016, 11160} := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_l2982_298280


namespace NUMINAMATH_CALUDE_buoy_distance_is_24_l2982_298295

/-- The distance between two consecutive buoys in the ocean -/
def buoy_distance (d1 d2 : ℝ) : ℝ := d2 - d1

/-- Theorem: The distance between two consecutive buoys is 24 meters -/
theorem buoy_distance_is_24 :
  let d1 := 72 -- distance of first buoy from beach
  let d2 := 96 -- distance of second buoy from beach
  buoy_distance d1 d2 = 24 := by sorry

end NUMINAMATH_CALUDE_buoy_distance_is_24_l2982_298295


namespace NUMINAMATH_CALUDE_geometric_sequence_302nd_term_l2982_298258

/-- Given a geometric sequence with first term 8 and second term -16, 
    the 302nd term is -2^304 -/
theorem geometric_sequence_302nd_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n + 2) = a (n + 1) * (a (n + 1) / a n)) →  -- geometric sequence condition
    a 1 = 8 →                                           -- first term
    a 2 = -16 →                                         -- second term
    a 302 = -2^304 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_302nd_term_l2982_298258


namespace NUMINAMATH_CALUDE_square_sum_product_l2982_298294

theorem square_sum_product (x : ℝ) :
  (Real.sqrt (8 + x) + Real.sqrt (27 - x) = 9) →
  (8 + x) * (27 - x) = 529 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l2982_298294
