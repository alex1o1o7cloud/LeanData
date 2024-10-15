import Mathlib

namespace NUMINAMATH_CALUDE_short_bushes_count_l2923_292366

/-- The number of short bushes initially in the park -/
def initial_short_bushes : ℕ := 37

/-- The number of short bushes planted by workers -/
def planted_short_bushes : ℕ := 20

/-- The total number of short bushes after planting -/
def total_short_bushes : ℕ := 57

/-- Theorem stating that the initial number of short bushes plus the planted ones equals the total -/
theorem short_bushes_count : 
  initial_short_bushes + planted_short_bushes = total_short_bushes := by
  sorry

end NUMINAMATH_CALUDE_short_bushes_count_l2923_292366


namespace NUMINAMATH_CALUDE_fraction_inequality_counterexample_l2923_292378

theorem fraction_inequality_counterexample :
  ∃ (a b c d A B C D : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
    a/b > A/B ∧ 
    c/d > C/D ∧ 
    (a+c)/(b+d) ≤ (A+C)/(B+D) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_counterexample_l2923_292378


namespace NUMINAMATH_CALUDE_palmer_photos_l2923_292301

theorem palmer_photos (initial_photos : ℕ) (first_week : ℕ) (final_total : ℕ) :
  initial_photos = 100 →
  first_week = 50 →
  final_total = 380 →
  final_total - initial_photos - first_week - 2 * first_week = 130 := by
sorry

end NUMINAMATH_CALUDE_palmer_photos_l2923_292301


namespace NUMINAMATH_CALUDE_winter_migration_l2923_292363

/-- The number of bird families living near the mountain -/
def mountain_families : ℕ := 18

/-- The number of bird families that flew to Africa -/
def africa_families : ℕ := 38

/-- The number of bird families that flew to Asia -/
def asia_families : ℕ := 80

/-- The total number of bird families that flew away for the winter -/
def total_migrated_families : ℕ := africa_families + asia_families

theorem winter_migration :
  total_migrated_families = 118 :=
by sorry

end NUMINAMATH_CALUDE_winter_migration_l2923_292363


namespace NUMINAMATH_CALUDE_circle_on_parabola_through_focus_l2923_292311

/-- A circle with center on the parabola y² = 8x and tangent to x + 2 = 0 passes through (2, 0) -/
theorem circle_on_parabola_through_focus (x y : ℝ) :
  y^2 = 8*x →  -- center (x, y) is on the parabola
  (x + 2)^2 + y^2 = (x + 4)^2 →  -- circle is tangent to x + 2 = 0
  (2 - x)^2 + y^2 = (x + 4)^2 :=  -- circle passes through (2, 0)
by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_through_focus_l2923_292311


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l2923_292330

/-- Converts a two-digit number in base b to base 10 -/
def to_base_10 (digit : Nat) (base : Nat) : Nat :=
  base * digit + digit

/-- Checks if a digit is valid in the given base -/
def is_valid_digit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_base_representation :
  ∃ (C D : Nat),
    is_valid_digit C 6 ∧
    is_valid_digit D 8 ∧
    to_base_10 C 6 = to_base_10 D 8 ∧
    to_base_10 C 6 = 63 ∧
    (∀ (C' D' : Nat),
      is_valid_digit C' 6 →
      is_valid_digit D' 8 →
      to_base_10 C' 6 = to_base_10 D' 8 →
      to_base_10 C' 6 ≥ 63) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l2923_292330


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l2923_292394

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l2923_292394


namespace NUMINAMATH_CALUDE_brass_composition_ratio_l2923_292357

theorem brass_composition_ratio (total_mass zinc_mass : ℝ) 
  (h_total : total_mass = 100)
  (h_zinc : zinc_mass = 35) :
  (total_mass - zinc_mass) / zinc_mass = 13 / 7 := by
  sorry

end NUMINAMATH_CALUDE_brass_composition_ratio_l2923_292357


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2923_292396

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 24 → volume = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2923_292396


namespace NUMINAMATH_CALUDE_total_peaches_l2923_292345

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 19

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 4

/-- The total number of baskets -/
def number_of_baskets : ℕ := 15

/-- Theorem: The total number of peaches in all baskets is 345 -/
theorem total_peaches :
  (red_peaches_per_basket + green_peaches_per_basket) * number_of_baskets = 345 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l2923_292345


namespace NUMINAMATH_CALUDE_square_diff_divided_by_24_l2923_292304

theorem square_diff_divided_by_24 : (145^2 - 121^2) / 24 = 266 := by sorry

end NUMINAMATH_CALUDE_square_diff_divided_by_24_l2923_292304


namespace NUMINAMATH_CALUDE_unique_integer_solution_l2923_292350

theorem unique_integer_solution :
  ∃! (x y z : ℤ), x^2 + y^2 + z^2 + 3 < x*y + 3*y + 2*z ∧ x = 1 ∧ y = 2 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l2923_292350


namespace NUMINAMATH_CALUDE_glove_probability_l2923_292334

/-- The probability of picking one left-handed glove and one right-handed glove -/
theorem glove_probability (left_gloves right_gloves : ℕ) 
  (h1 : left_gloves = 12) 
  (h2 : right_gloves = 10) : 
  (left_gloves * right_gloves : ℚ) / (Nat.choose (left_gloves + right_gloves) 2) = 120 / 231 := by
  sorry

end NUMINAMATH_CALUDE_glove_probability_l2923_292334


namespace NUMINAMATH_CALUDE_smaller_square_area_l2923_292399

theorem smaller_square_area (larger_square_area : ℝ) 
  (h1 : larger_square_area = 144) 
  (h2 : ∀ (side : ℝ), side * side = larger_square_area → 
        ∃ (smaller_side : ℝ), smaller_side = side / 2) : 
  ∃ (smaller_area : ℝ), smaller_area = 72 := by
sorry

end NUMINAMATH_CALUDE_smaller_square_area_l2923_292399


namespace NUMINAMATH_CALUDE_museum_tour_time_l2923_292347

theorem museum_tour_time (total_students : ℕ) (num_groups : ℕ) (time_per_student : ℕ) 
  (h1 : total_students = 18)
  (h2 : num_groups = 3)
  (h3 : time_per_student = 4)
  (h4 : total_students % num_groups = 0) : -- Ensuring equal groups
  (total_students / num_groups) * time_per_student = 24 := by
  sorry

end NUMINAMATH_CALUDE_museum_tour_time_l2923_292347


namespace NUMINAMATH_CALUDE_min_square_sum_on_line_l2923_292317

/-- The minimum value of x^2 + y^2 for points on the line x + y - 4 = 0 is 8 -/
theorem min_square_sum_on_line :
  ∃ (min : ℝ), min = 8 ∧
  ∀ (x y : ℝ), x + y - 4 = 0 →
  x^2 + y^2 ≥ min ∧
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ - 4 = 0 ∧ x₀^2 + y₀^2 = min :=
by sorry

end NUMINAMATH_CALUDE_min_square_sum_on_line_l2923_292317


namespace NUMINAMATH_CALUDE_paint_calculation_l2923_292367

theorem paint_calculation (total_paint : ℚ) : 
  (2 / 3 : ℚ) * total_paint + (1 / 5 : ℚ) * ((1 / 3 : ℚ) * total_paint) = 264 → 
  total_paint = 360 := by
sorry

end NUMINAMATH_CALUDE_paint_calculation_l2923_292367


namespace NUMINAMATH_CALUDE_max_third_side_of_triangle_l2923_292327

theorem max_third_side_of_triangle (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), c = 17 ∧ 
  (∀ (x : ℕ), (a + b > x ∧ x > b - a ∧ x > a - b) → x ≤ c) :=
sorry

end NUMINAMATH_CALUDE_max_third_side_of_triangle_l2923_292327


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l2923_292338

theorem middle_part_of_proportional_division (total : ℝ) (p1 p2 p3 : ℝ) :
  total = 120 →
  p1 > 0 →
  p2 > 0 →
  p3 > 0 →
  p1 / 2 = p2 / (2/3) →
  p1 / 2 = p3 / (2/9) →
  p1 + p2 + p3 = total →
  p2 = 27.6 := by
  sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l2923_292338


namespace NUMINAMATH_CALUDE_multiplication_subtraction_difference_l2923_292339

/-- Given that x = 5, prove that the number n satisfying 3x = (16 - x) + n is equal to 4 -/
theorem multiplication_subtraction_difference (x : ℝ) (n : ℝ) : 
  x = 5 → 3 * x = (16 - x) + n → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_difference_l2923_292339


namespace NUMINAMATH_CALUDE_xyz_inequality_l2923_292336

theorem xyz_inequality (x y z : ℝ) 
  (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z)
  (h_sum : x + y + z = 1) : 
  0 ≤ x*y + y*z + z*x - 2*x*y*z ∧ x*y + y*z + z*x - 2*x*y*z ≤ 7/27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2923_292336


namespace NUMINAMATH_CALUDE_height_equals_median_implies_angle_leq_60_height_equals_median_and_bisector_implies_equilateral_l2923_292321

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Checks if a triangle is acute -/
def Triangle.isAcute (t : Triangle) : Prop := sorry

/-- Returns the length of the height from vertex A to side BC -/
def Triangle.heightAH (t : Triangle) : ℝ := sorry

/-- Returns the length of the median from vertex B to side AC -/
def Triangle.medianBM (t : Triangle) : ℝ := sorry

/-- Returns the length of the angle bisector from vertex C -/
def Triangle.angleBisectorCD (t : Triangle) : ℝ := sorry

/-- Returns the measure of angle ABC in degrees -/
def Triangle.angleABC (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def Triangle.isEquilateral (t : Triangle) : Prop := sorry

/-- Theorem: If the largest height AH is equal to the median BM in an acute triangle,
    then angle ABC is not greater than 60 degrees -/
theorem height_equals_median_implies_angle_leq_60 (t : Triangle) 
  (h1 : t.isAcute) 
  (h2 : t.heightAH = t.medianBM) : 
  t.angleABC ≤ 60 := by sorry

/-- Theorem: If the height AH is equal to both the median BM and the angle bisector CD 
    in an acute triangle, then the triangle is equilateral -/
theorem height_equals_median_and_bisector_implies_equilateral (t : Triangle) 
  (h1 : t.isAcute) 
  (h2 : t.heightAH = t.medianBM) 
  (h3 : t.heightAH = t.angleBisectorCD) : 
  t.isEquilateral := by sorry

end NUMINAMATH_CALUDE_height_equals_median_implies_angle_leq_60_height_equals_median_and_bisector_implies_equilateral_l2923_292321


namespace NUMINAMATH_CALUDE_find_N_l2923_292328

theorem find_N : ∃ N : ℕ, 
  (981 + 983 + 985 + 987 + 989 + 991 + 993 = 7000 - N) ∧ (N = 91) := by
  sorry

end NUMINAMATH_CALUDE_find_N_l2923_292328


namespace NUMINAMATH_CALUDE_megan_folders_l2923_292341

/-- The number of folders Megan ended up with -/
def num_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : ℕ :=
  (initial_files - deleted_files) / files_per_folder

/-- Proof that Megan ended up with 9 folders -/
theorem megan_folders : num_folders 93 21 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_megan_folders_l2923_292341


namespace NUMINAMATH_CALUDE_susan_missed_pay_l2923_292331

/-- Calculates the pay Susan will miss during her three-week vacation --/
def missed_pay (regular_rate : ℚ) (overtime_rate : ℚ) (sunday_rate : ℚ) 
               (regular_hours : ℕ) (overtime_hours : ℕ) (sunday_hours : ℕ)
               (sunday_count : List ℕ) (vacation_days : ℕ) (workweek_days : ℕ) : ℚ :=
  let weekly_pay := regular_rate * regular_hours + overtime_rate * overtime_hours
  let sunday_pay := sunday_rate * sunday_hours * (sunday_count.sum)
  let total_pay := weekly_pay * 3 + sunday_pay
  let paid_vacation_pay := regular_rate * regular_hours * (vacation_days / workweek_days)
  total_pay - paid_vacation_pay

/-- The main theorem stating that Susan will miss $2160 during her vacation --/
theorem susan_missed_pay : 
  missed_pay 15 20 25 40 8 8 [1, 2, 0] 6 5 = 2160 := by
  sorry

end NUMINAMATH_CALUDE_susan_missed_pay_l2923_292331


namespace NUMINAMATH_CALUDE_museum_revenue_calculation_l2923_292326

/-- Revenue calculation for The Metropolitan Museum of Art --/
theorem museum_revenue_calculation 
  (total_visitors : ℕ) 
  (nyc_resident_ratio : ℚ)
  (college_student_ratio : ℚ)
  (college_ticket_price : ℕ) :
  total_visitors = 200 →
  nyc_resident_ratio = 1/2 →
  college_student_ratio = 3/10 →
  college_ticket_price = 4 →
  (total_visitors : ℚ) * nyc_resident_ratio * college_student_ratio * college_ticket_price = 120 := by
  sorry

#check museum_revenue_calculation

end NUMINAMATH_CALUDE_museum_revenue_calculation_l2923_292326


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2923_292308

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → e = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2923_292308


namespace NUMINAMATH_CALUDE_system_solution_l2923_292379

variables (a b x y : ℝ)

theorem system_solution (h1 : x / (a - 2*b) - y / (a + 2*b) = (6*a*b) / (a^2 - 4*b^2))
                        (h2 : (x + y) / (a + 2*b) + (x - y) / (a - 2*b) = (2*(a^2 - a*b + 2*b^2)) / (a^2 - 4*b^2))
                        (h3 : a ≠ 2*b)
                        (h4 : a ≠ -2*b)
                        (h5 : a^2 ≠ 4*b^2) :
  x = a + b ∧ y = a - b :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l2923_292379


namespace NUMINAMATH_CALUDE_dwarf_truth_count_l2923_292335

/-- Represents the number of dwarfs who always tell the truth -/
def truthful_dwarfs : ℕ := sorry

/-- Represents the number of dwarfs who always lie -/
def lying_dwarfs : ℕ := sorry

/-- The total number of dwarfs -/
def total_dwarfs : ℕ := 10

/-- The number of times hands were raised for vanilla ice cream -/
def vanilla_hands : ℕ := 10

/-- The number of times hands were raised for chocolate ice cream -/
def chocolate_hands : ℕ := 5

/-- The number of times hands were raised for fruit ice cream -/
def fruit_hands : ℕ := 1

/-- The total number of times hands were raised -/
def total_hands_raised : ℕ := vanilla_hands + chocolate_hands + fruit_hands

theorem dwarf_truth_count :
  truthful_dwarfs + lying_dwarfs = total_dwarfs ∧
  truthful_dwarfs + 2 * lying_dwarfs = total_hands_raised ∧
  truthful_dwarfs = 4 := by sorry

end NUMINAMATH_CALUDE_dwarf_truth_count_l2923_292335


namespace NUMINAMATH_CALUDE_total_veranda_area_l2923_292324

/-- Calculates the total area of verandas in a multi-story building. -/
theorem total_veranda_area (floors : ℕ) (room_length room_width veranda_width : ℝ) :
  floors = 4 →
  room_length = 21 →
  room_width = 12 →
  veranda_width = 2 →
  (floors * ((room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width)) = 592 :=
by sorry

end NUMINAMATH_CALUDE_total_veranda_area_l2923_292324


namespace NUMINAMATH_CALUDE_log_sum_simplification_l2923_292395

theorem log_sum_simplification :
  1 / (Real.log 3 / Real.log 18 + 1) +
  1 / (Real.log 2 / Real.log 12 + 1) +
  1 / (Real.log 7 / Real.log 8 + 1) =
  13 / 12 := by sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l2923_292395


namespace NUMINAMATH_CALUDE_prime_equation_solution_l2923_292323

theorem prime_equation_solution (p : ℕ) (hp : Prime p) :
  (∃ (n : ℤ) (k m : ℕ+), (m * k^2 + 2) * p - (m^2 + 2 * k^2) = n^2 * (m * p + 2)) →
  p = 3 ∨ p % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l2923_292323


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2923_292322

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (2*m - 1) * 2 - (m + 3) * 3 - (m - 11) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2923_292322


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2923_292320

theorem sum_with_radical_conjugate :
  ∃ (x : ℝ), x^2 = 2023 ∧ (15 - x) + (15 + x) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2923_292320


namespace NUMINAMATH_CALUDE_car_fuel_usage_l2923_292364

/-- Proves that a car traveling at 40 miles per hour for 5 hours, with a fuel efficiency
    of 1 gallon per 40 miles and starting with a full 12-gallon tank, uses 5/12 of its fuel. -/
theorem car_fuel_usage (speed : ℝ) (time : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) :
  speed = 40 →
  time = 5 →
  fuel_efficiency = 40 →
  tank_capacity = 12 →
  (speed * time / fuel_efficiency) / tank_capacity = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_car_fuel_usage_l2923_292364


namespace NUMINAMATH_CALUDE_student_age_ratio_l2923_292370

/-- Represents the number of students in different age groups -/
structure SchoolPopulation where
  total : ℕ
  below_eight : ℕ
  eight_years : ℕ
  above_eight : ℕ

/-- Theorem stating the ratio of students above 8 years to 8 years old -/
theorem student_age_ratio (school : SchoolPopulation) 
  (h1 : school.total = 80)
  (h2 : school.below_eight = school.total / 4)
  (h3 : school.eight_years = 36)
  (h4 : school.above_eight = school.total - school.below_eight - school.eight_years) :
  (school.above_eight : ℚ) / school.eight_years = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_age_ratio_l2923_292370


namespace NUMINAMATH_CALUDE_binomial_150_150_equals_1_l2923_292359

theorem binomial_150_150_equals_1 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_equals_1_l2923_292359


namespace NUMINAMATH_CALUDE_fraction_reducibility_l2923_292356

theorem fraction_reducibility (n : ℕ) :
  (∃ k : ℕ, k > 1 ∧ (n^2 + 1).gcd (n + 1) = k) ↔ n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_reducibility_l2923_292356


namespace NUMINAMATH_CALUDE_expression_equality_l2923_292354

theorem expression_equality : 1 - (1 / (1 + Real.sqrt 2)) + (1 / (1 - Real.sqrt 2)) = 1 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2923_292354


namespace NUMINAMATH_CALUDE_relationship_abxy_l2923_292383

theorem relationship_abxy (a b x y : ℚ) 
  (eq1 : x + y = a + b) 
  (ineq1 : y - x < a - b) 
  (ineq2 : b > a) : 
  y < a ∧ a < b ∧ b < x :=
sorry

end NUMINAMATH_CALUDE_relationship_abxy_l2923_292383


namespace NUMINAMATH_CALUDE_bleacher_sets_l2923_292392

theorem bleacher_sets (total_fans : ℕ) (fans_per_set : ℕ) (h1 : total_fans = 2436) (h2 : fans_per_set = 812) :
  total_fans / fans_per_set = 3 :=
by sorry

end NUMINAMATH_CALUDE_bleacher_sets_l2923_292392


namespace NUMINAMATH_CALUDE_smallest_angle_tan_equation_l2923_292377

open Real

theorem smallest_angle_tan_equation (x : ℝ) : 
  (0 < x) ∧ 
  (x < 2 * π) ∧
  (tan (6 * x) = (cos (2 * x) - sin (2 * x)) / (cos (2 * x) + sin (2 * x))) →
  x = 5.625 * (π / 180) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_tan_equation_l2923_292377


namespace NUMINAMATH_CALUDE_star_equality_implies_x_equals_four_l2923_292337

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b - d)

-- Theorem statement
theorem star_equality_implies_x_equals_four :
  ∀ x y : ℤ, star 5 5 2 1 = star x y 1 4 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_implies_x_equals_four_l2923_292337


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_31_l2923_292375

/-- Represents the pricing model of a caterer -/
structure CatererPricing where
  basic_fee : ℕ
  per_person : ℕ
  additional_fee : ℕ

/-- The first caterer's pricing model -/
def caterer1 : CatererPricing := {
  basic_fee := 150,
  per_person := 20,
  additional_fee := 0
}

/-- The second caterer's pricing model -/
def caterer2 : CatererPricing := {
  basic_fee := 250,
  per_person := 15,
  additional_fee := 50
}

/-- Calculate the total cost for a caterer given the number of people -/
def total_cost (c : CatererPricing) (people : ℕ) : ℕ :=
  c.basic_fee + c.per_person * people + c.additional_fee

/-- Theorem stating that 31 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_31 :
  (∀ n : ℕ, n < 31 → total_cost caterer1 n ≤ total_cost caterer2 n) ∧
  (total_cost caterer1 31 > total_cost caterer2 31) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_31_l2923_292375


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2923_292329

theorem inequality_system_solution_set :
  {x : ℝ | x + 2 ≤ 3 ∧ 1 + x > -2} = {x : ℝ | -3 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2923_292329


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2923_292307

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = 5) : z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2923_292307


namespace NUMINAMATH_CALUDE_penny_species_count_l2923_292388

/-- The number of shark species Penny identified -/
def shark_species : ℕ := 35

/-- The number of eel species Penny identified -/
def eel_species : ℕ := 15

/-- The number of whale species Penny identified -/
def whale_species : ℕ := 5

/-- The total number of species Penny identified -/
def total_species : ℕ := shark_species + eel_species + whale_species

/-- Theorem stating that the total number of species Penny identified is 55 -/
theorem penny_species_count : total_species = 55 := by
  sorry

end NUMINAMATH_CALUDE_penny_species_count_l2923_292388


namespace NUMINAMATH_CALUDE_parabola_equation_after_coordinate_shift_l2923_292332

/-- Represents a parabola in a 2D coordinate system -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ
  eq : ℝ → ℝ := λ x => a * (x - h)^2 + k

/-- Represents a 2D coordinate system -/
structure CoordinateSystem where
  origin : ℝ × ℝ

/-- Translates a point from one coordinate system to another -/
def translate (p : ℝ × ℝ) (old_sys new_sys : CoordinateSystem) : ℝ × ℝ :=
  (p.1 - (new_sys.origin.1 - old_sys.origin.1), 
   p.2 - (new_sys.origin.2 - old_sys.origin.2))

theorem parabola_equation_after_coordinate_shift 
  (p : Parabola) 
  (old_sys new_sys : CoordinateSystem) :
  p.a = 3 ∧ 
  p.h = 0 ∧ 
  p.k = 0 ∧
  new_sys.origin = (-1, -1) →
  ∀ x y : ℝ, 
    (translate (x, y) new_sys old_sys).2 = p.eq (translate (x, y) new_sys old_sys).1 ↔
    y = 3 * (x + 1)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_after_coordinate_shift_l2923_292332


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l2923_292390

/-- Given that 3/4 of 12 bananas are worth 9 oranges, 
    prove that 1/3 of 9 bananas are worth 3 oranges -/
theorem banana_orange_equivalence 
  (h : (3/4 : ℚ) * 12 * (banana_value : ℚ) = 9 * (orange_value : ℚ)) :
  (1/3 : ℚ) * 9 * banana_value = 3 * orange_value :=
by sorry


end NUMINAMATH_CALUDE_banana_orange_equivalence_l2923_292390


namespace NUMINAMATH_CALUDE_triangle_sine_identity_l2923_292351

theorem triangle_sine_identity (A B C : Real) (h : A + B + C = Real.pi) :
  Real.sin (2 * A) + Real.sin (2 * B) + Real.sin (2 * C) = 4 * Real.sin A * Real.sin B * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_identity_l2923_292351


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l2923_292316

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l2923_292316


namespace NUMINAMATH_CALUDE_mixed_yellow_ratio_is_quarter_l2923_292398

/-- Represents a bag of jelly beans -/
structure JellyBeanBag where
  total : ℕ
  yellow_ratio : ℚ

/-- Calculates the total number of yellow jelly beans in a bag -/
def yellow_count (bag : JellyBeanBag) : ℚ :=
  bag.total * bag.yellow_ratio

/-- Calculates the ratio of yellow jelly beans to total jelly beans when multiple bags are mixed -/
def mixed_yellow_ratio (bags : List JellyBeanBag) : ℚ :=
  let total_yellow := bags.map yellow_count |>.sum
  let total_beans := bags.map (·.total) |>.sum
  total_yellow / total_beans

theorem mixed_yellow_ratio_is_quarter (bags : List JellyBeanBag) :
  bags = [
    ⟨24, 2/5⟩,
    ⟨30, 3/10⟩,
    ⟨32, 1/4⟩,
    ⟨34, 1/10⟩
  ] →
  mixed_yellow_ratio bags = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_mixed_yellow_ratio_is_quarter_l2923_292398


namespace NUMINAMATH_CALUDE_max_x_minus_y_l2923_292391

theorem max_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) :
  ∃ (m : ℝ), m = 1 / (2 * Real.sqrt 3) ∧ ∀ (a b : ℝ), 3 * (a^2 + b^2) = a + b → (a - b) ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l2923_292391


namespace NUMINAMATH_CALUDE_alcohol_dilution_l2923_292346

/-- Proves that adding 3 litres of water to 18 litres of a 20% alcohol mixture 
    results in a new mixture with 17.14285714285715% alcohol. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) 
    (water_added : ℝ) (final_percentage : ℝ) : 
    initial_volume = 18 →
    initial_percentage = 0.20 →
    water_added = 3 →
    final_percentage = 0.1714285714285715 →
    (initial_volume * initial_percentage) / (initial_volume + water_added) = final_percentage := by
  sorry


end NUMINAMATH_CALUDE_alcohol_dilution_l2923_292346


namespace NUMINAMATH_CALUDE_inequality_proof_l2923_292386

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2923_292386


namespace NUMINAMATH_CALUDE_expression_simplification_l2923_292372

theorem expression_simplification (b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  let a : ℝ := 0.04
  1.24 * (Real.sqrt ((a * b * c + 4) / a + 4 * Real.sqrt (b * c / a))) / (Real.sqrt (a * b * c) + 2) = 6.2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2923_292372


namespace NUMINAMATH_CALUDE_five_digit_numbers_count_l2923_292389

/-- The number of odd digits available -/
def num_odd_digits : ℕ := 5

/-- The number of even digits available -/
def num_even_digits : ℕ := 5

/-- The total number of digits in the formed numbers -/
def total_digits : ℕ := 5

/-- Function to calculate the number of ways to form five-digit numbers -/
def count_five_digit_numbers : ℕ :=
  let case1 := Nat.choose num_odd_digits 2 * Nat.choose (num_even_digits - 1) 3 * Nat.factorial total_digits
  let case2 := Nat.choose num_odd_digits 2 * Nat.choose (num_even_digits - 1) 2 * Nat.choose 4 1 * Nat.factorial 4
  case1 + case2

/-- Theorem stating that the number of unique five-digit numbers is 10,560 -/
theorem five_digit_numbers_count : count_five_digit_numbers = 10560 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_numbers_count_l2923_292389


namespace NUMINAMATH_CALUDE_makeup_problem_solution_l2923_292355

/-- Represents the makeup problem with given parameters -/
structure MakeupProblem where
  people_per_tube : ℕ
  total_people : ℕ
  num_tubs : ℕ

/-- Calculates the number of tubes per tub for a given makeup problem -/
def tubes_per_tub (p : MakeupProblem) : ℕ :=
  (p.total_people / p.people_per_tube) / p.num_tubs

/-- Theorem stating that for the given problem, the number of tubes per tub is 2 -/
theorem makeup_problem_solution :
  let p : MakeupProblem := ⟨3, 36, 6⟩
  tubes_per_tub p = 2 := by
  sorry

end NUMINAMATH_CALUDE_makeup_problem_solution_l2923_292355


namespace NUMINAMATH_CALUDE_apple_tree_yield_l2923_292397

theorem apple_tree_yield (total : ℕ) : 
  (total / 5 : ℚ) +             -- First day
  (2 * (total / 5) : ℚ) +       -- Second day
  (total / 5 + 20 : ℚ) +        -- Third day
  20 = total →                  -- Remaining apples
  total = 200 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_yield_l2923_292397


namespace NUMINAMATH_CALUDE_gathering_attendance_l2923_292382

theorem gathering_attendance (W S B : ℕ) (hW : W = 26) (hS : S = 22) (hB : B = 17) :
  W + S - B = 31 := by sorry

end NUMINAMATH_CALUDE_gathering_attendance_l2923_292382


namespace NUMINAMATH_CALUDE_sqrt_two_cos_sin_equality_l2923_292384

theorem sqrt_two_cos_sin_equality (x : ℝ) :
  Real.sqrt 2 * (Real.cos (2 * x))^4 - Real.sqrt 2 * (Real.sin (2 * x))^4 = Real.cos (2 * x) + Real.sin (2 * x) →
  ∃ k : ℤ, x = Real.pi * (4 * k - 1) / 8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_two_cos_sin_equality_l2923_292384


namespace NUMINAMATH_CALUDE_quadratic_equations_integer_roots_l2923_292303

theorem quadratic_equations_integer_roots :
  ∃ (a b c : ℕ),
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ∧
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 + b * x - c = 0 ∧ a * y^2 + b * y - c = 0) ∧
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) ∧
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 - b * x - c = 0 ∧ a * y^2 - b * y - c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_integer_roots_l2923_292303


namespace NUMINAMATH_CALUDE_max_n_is_14_l2923_292343

/-- A function that divides a list of integers into two groups -/
def divide_into_groups (n : ℕ) : (List ℕ) × (List ℕ) := sorry

/-- Predicate to check if a list contains no pair of numbers that sum to a perfect square -/
def no_square_sum (l : List ℕ) : Prop := sorry

/-- Predicate to check if two lists have no common elements -/
def no_common_elements (l1 l2 : List ℕ) : Prop := sorry

/-- The main theorem stating that 14 is the maximum value of n satisfying the conditions -/
theorem max_n_is_14 : 
  ∀ n : ℕ, n > 14 → 
  ¬∃ (g1 g2 : List ℕ), 
    (divide_into_groups n = (g1, g2)) ∧ 
    (no_square_sum g1) ∧ 
    (no_square_sum g2) ∧ 
    (no_common_elements g1 g2) ∧ 
    (g1.length + g2.length = n) ∧
    (∀ i : ℕ, i ∈ g1 ∨ i ∈ g2 ↔ 1 ≤ i ∧ i ≤ n) :=
sorry

end NUMINAMATH_CALUDE_max_n_is_14_l2923_292343


namespace NUMINAMATH_CALUDE_twelfth_term_of_sequence_l2923_292387

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

-- State the theorem
theorem twelfth_term_of_sequence (a₁ d : ℚ) (h₁ : a₁ = 1/4) :
  arithmetic_sequence a₁ d 12 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_sequence_l2923_292387


namespace NUMINAMATH_CALUDE_fraction_equality_l2923_292361

theorem fraction_equality : (5 * 6 + 3) / 9 = 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2923_292361


namespace NUMINAMATH_CALUDE_polynomial_degree_product_l2923_292306

-- Define the polynomials
def p (x : ℝ) := 5*x^3 - 4*x + 7
def q (x : ℝ) := 2*x^2 + 9

-- State the theorem
theorem polynomial_degree_product : 
  Polynomial.degree ((Polynomial.monomial 0 1 + Polynomial.X)^10 * (Polynomial.monomial 0 1 + Polynomial.X)^5) = 40 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_degree_product_l2923_292306


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l2923_292385

-- Define the function f(x) = x³ - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- State the theorem
theorem tangent_line_triangle_area :
  let tangent_slope : ℝ := f' 0
  let tangent_y_intercept : ℝ := 1
  let tangent_x_intercept : ℝ := 1
  (1 / 2) * tangent_x_intercept * tangent_y_intercept = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l2923_292385


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l2923_292305

theorem largest_x_sqrt_3x_eq_5x : 
  (∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x) →
  (∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y → y ≤ 3/25) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l2923_292305


namespace NUMINAMATH_CALUDE_total_players_l2923_292310

theorem total_players (cricket : ℕ) (hockey : ℕ) (football : ℕ) (softball : ℕ)
  (h1 : cricket = 16)
  (h2 : hockey = 12)
  (h3 : football = 18)
  (h4 : softball = 13) :
  cricket + hockey + football + softball = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l2923_292310


namespace NUMINAMATH_CALUDE_five_fourths_of_x_over_three_l2923_292313

theorem five_fourths_of_x_over_three (x : ℝ) : (5 / 4) * (x / 3) = 5 * x / 12 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_x_over_three_l2923_292313


namespace NUMINAMATH_CALUDE_magnitude_equality_not_implies_vector_equality_l2923_292342

-- Define vectors a and b in a real vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (a b : V)

-- State the theorem
theorem magnitude_equality_not_implies_vector_equality :
  ∃ a b : V, (‖a‖ = 3 * ‖b‖) ∧ (a ≠ 3 • b) ∧ (a ≠ -3 • b) := by
  sorry

end NUMINAMATH_CALUDE_magnitude_equality_not_implies_vector_equality_l2923_292342


namespace NUMINAMATH_CALUDE_subtraction_absolute_value_l2923_292333

theorem subtraction_absolute_value (x y : ℝ) : 
  |8 - 3| - |x - y| = 3 → |x - y| = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_absolute_value_l2923_292333


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l2923_292353

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l2923_292353


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l2923_292325

theorem fraction_sum_difference : 1/2 + 3/4 - 5/8 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l2923_292325


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2923_292371

/-- Given a hyperbola with the following properties:
    - The distance from the vertex to its asymptote is 2
    - The distance from the focus to the asymptote is 6
    Then the eccentricity of the hyperbola is 3 -/
theorem hyperbola_eccentricity (vertex_to_asymptote : ℝ) (focus_to_asymptote : ℝ) 
  (h1 : vertex_to_asymptote = 2)
  (h2 : focus_to_asymptote = 6) :
  let e := focus_to_asymptote / vertex_to_asymptote
  e = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2923_292371


namespace NUMINAMATH_CALUDE_property_price_reduction_l2923_292373

theorem property_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 5000)
  (h2 : final_price = 4050)
  (h3 : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price) :
  ∃ x : ℝ, x = 0.1 ∧ initial_price * (1 - x)^2 = final_price :=
by sorry

end NUMINAMATH_CALUDE_property_price_reduction_l2923_292373


namespace NUMINAMATH_CALUDE_intersection_M_N_l2923_292319

-- Define the sets M and N
def M : Set ℝ := {x | x ≤ 4}
def N : Set ℝ := {x | x > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2923_292319


namespace NUMINAMATH_CALUDE_min_value_inequality_l2923_292376

theorem min_value_inequality (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) : 
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2923_292376


namespace NUMINAMATH_CALUDE_sms_authenticity_l2923_292393

/-- Represents an SMS message -/
structure SMS where
  content : String
  sender : String

/-- Represents a bank card -/
structure BankCard where
  number : String
  bank : String
  officialPhoneNumber : String

/-- Represents a bank's SMS characteristics -/
structure BankSMSCharacteristics where
  shortNumber : String
  messageFormat : String → Bool

/-- Determines if an SMS is genuine based on comparison and bank confirmation -/
def isGenuineSMS (message : SMS) (card : BankCard) (prevMessages : List SMS) 
                 (bankCharacteristics : BankSMSCharacteristics) : Prop :=
  (∃ prev ∈ prevMessages, message.sender = prev.sender ∧ 
                          bankCharacteristics.messageFormat message.content) ∧
  (∃ confirmation : Bool, confirmation = true)

/-- Main theorem: An SMS is genuine iff it matches previous messages and is confirmed by the bank -/
theorem sms_authenticity 
  (message : SMS) 
  (card : BankCard) 
  (prevMessages : List SMS) 
  (bankCharacteristics : BankSMSCharacteristics) :
  isGenuineSMS message card prevMessages bankCharacteristics ↔ 
  (∃ prev ∈ prevMessages, message.sender = prev.sender ∧ 
                          bankCharacteristics.messageFormat message.content) ∧
  (∃ confirmation : Bool, confirmation = true) :=
by sorry


end NUMINAMATH_CALUDE_sms_authenticity_l2923_292393


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2923_292315

theorem product_of_three_numbers (a b c : ℚ) : 
  a + b + c = 30 →
  a = 3 * (b + c) →
  b = 6 * c →
  a * b * c = 675 / 28 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2923_292315


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2923_292340

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem first_term_of_arithmetic_sequence :
  ∃ a₁ : ℤ, arithmetic_sequence a₁ 2 15 = -10 ∧ a₁ = -38 := by sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2923_292340


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2923_292300

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  A = π / 3 →
  Real.cos C = 1 / 3 →
  c = 4 * Real.sqrt 2 →
  (1 / 2) * a * c * Real.sin B = 4 * Real.sqrt 3 + 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_proof_l2923_292300


namespace NUMINAMATH_CALUDE_ax5_plus_by5_exists_l2923_292369

theorem ax5_plus_by5_exists (a b x y : ℝ) 
  (h1 : a*x + b*y = 4)
  (h2 : a*x^2 + b*y^2 = 10)
  (h3 : a*x^3 + b*y^3 = 28)
  (h4 : a*x^4 + b*y^4 = 82) :
  ∃ s5 : ℝ, a*x^5 + b*y^5 = s5 :=
by
  sorry

end NUMINAMATH_CALUDE_ax5_plus_by5_exists_l2923_292369


namespace NUMINAMATH_CALUDE_exists_parallelepiped_with_square_coverage_l2923_292309

/-- A rectangular parallelepiped with integer dimensions -/
structure Parallelepiped where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- A square with integer side length -/
structure Square where
  side : ℕ+

/-- Represents the coverage of a parallelepiped by three squares -/
structure Coverage where
  parallelepiped : Parallelepiped
  squares : Fin 3 → Square
  covers_without_gaps : Bool
  each_pair_shares_edge : Bool

/-- Theorem stating the existence of a parallelepiped covered by three squares with shared edges -/
theorem exists_parallelepiped_with_square_coverage : 
  ∃ (c : Coverage), c.covers_without_gaps ∧ c.each_pair_shares_edge := by
  sorry

end NUMINAMATH_CALUDE_exists_parallelepiped_with_square_coverage_l2923_292309


namespace NUMINAMATH_CALUDE_division_remainder_l2923_292348

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 686 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 2 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l2923_292348


namespace NUMINAMATH_CALUDE_age_equation_solution_l2923_292312

theorem age_equation_solution (A : ℝ) (N : ℝ) (h1 : A = 64) :
  (1 / 2) * ((A + 8) * N - N * (A - 8)) = A ↔ N = 8 := by
  sorry

end NUMINAMATH_CALUDE_age_equation_solution_l2923_292312


namespace NUMINAMATH_CALUDE_squirrel_stockpiling_days_l2923_292302

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts each busy squirrel stockpiles per day -/
def busy_squirrel_nuts_per_day : ℕ := 30

/-- The number of sleepy squirrels -/
def sleepy_squirrels : ℕ := 1

/-- The number of nuts the sleepy squirrel stockpiles per day -/
def sleepy_squirrel_nuts_per_day : ℕ := 20

/-- The total number of nuts found in Mason's car -/
def total_nuts : ℕ := 3200

/-- The number of days squirrels have been stockpiling nuts -/
def stockpiling_days : ℕ := 40

theorem squirrel_stockpiling_days :
  stockpiling_days * (busy_squirrels * busy_squirrel_nuts_per_day + sleepy_squirrels * sleepy_squirrel_nuts_per_day) = total_nuts :=
by sorry

end NUMINAMATH_CALUDE_squirrel_stockpiling_days_l2923_292302


namespace NUMINAMATH_CALUDE_not_regressive_a_regressive_increasing_is_arithmetic_l2923_292344

-- Definition of a regressive sequence
def IsRegressive (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, x n + x (n + 2) - x (n + 1) = x m

-- Part 1
def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 3 * a n

theorem not_regressive_a : ¬ IsRegressive a := by sorry

-- Part 2
theorem regressive_increasing_is_arithmetic (b : ℕ → ℝ) 
  (h_regressive : IsRegressive b) (h_increasing : ∀ n : ℕ, b n < b (n + 1)) :
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) - b n = d := by sorry

end NUMINAMATH_CALUDE_not_regressive_a_regressive_increasing_is_arithmetic_l2923_292344


namespace NUMINAMATH_CALUDE_two_digit_number_proof_l2923_292362

theorem two_digit_number_proof : 
  ∀ (n : ℕ), 
  (n ≥ 10 ∧ n < 100) →  -- n is a two-digit number
  (n % 10 + n / 10 = 9) →  -- sum of digits is 9
  (10 * (n % 10) + n / 10 = n - 9) →  -- swapping digits results in n - 9
  n = 54 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_proof_l2923_292362


namespace NUMINAMATH_CALUDE_odometer_problem_l2923_292380

theorem odometer_problem (a b c : ℕ) (ha : a ≥ 1) (hsum : a + b + c ≤ 10) 
  (hdiv : ∃ t : ℕ, (100 * a + 10 * c) - (100 * a + 10 * b + c) = 60 * t) :
  a^2 + b^2 + c^2 = 26 := by
sorry

end NUMINAMATH_CALUDE_odometer_problem_l2923_292380


namespace NUMINAMATH_CALUDE_sum_bounds_l2923_292381

theorem sum_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b + 1/a + 1/b = 5) : 1 ≤ a + b ∧ a + b ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_bounds_l2923_292381


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2923_292358

/-- The curve function f(x) = (x-5)^2 * (x+3) -/
def f (x : ℝ) : ℝ := (x - 5)^2 * (x + 3)

/-- The area of the triangle bounded by the axes and the curve y = f(x) -/
def triangle_area : ℝ := 300

theorem triangle_area_proof : 
  triangle_area = 300 := by sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l2923_292358


namespace NUMINAMATH_CALUDE_roses_in_vase_l2923_292352

theorem roses_in_vase (initial_roses : ℕ) : initial_roses + 8 = 18 → initial_roses = 10 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l2923_292352


namespace NUMINAMATH_CALUDE_original_number_proof_l2923_292368

theorem original_number_proof : 
  ∃ x : ℝ, (x * 1.4 = 700) ∧ (x = 500) := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2923_292368


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l2923_292374

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- The sum of faces, edges, and vertices of a rectangular prism is 26. -/
theorem rectangular_prism_sum (rp : RectangularPrism) :
  rp.faces + rp.edges + rp.vertices = 26 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l2923_292374


namespace NUMINAMATH_CALUDE_triangle_constructible_l2923_292318

/-- Given a side length, angle bisector length, and altitude length of a triangle,
    prove that the triangle can be constructed uniquely if and only if
    the angle bisector length is greater than the altitude length. -/
theorem triangle_constructible (a f_a m_a : ℝ) (h_pos : a > 0 ∧ f_a > 0 ∧ m_a > 0) :
  ∃! (b c : ℝ), (b > 0 ∧ c > 0) ∧
    (∃ (α β γ : ℝ), 
      α > 0 ∧ β > 0 ∧ γ > 0 ∧
      α + β + γ = π ∧
      a^2 = b^2 + c^2 - 2*b*c*Real.cos α ∧
      f_a^2 = (b*c / (b + c))^2 + (a/2)^2 ∧
      m_a = a * Real.sin β / 2) ↔
  f_a > m_a :=
sorry

end NUMINAMATH_CALUDE_triangle_constructible_l2923_292318


namespace NUMINAMATH_CALUDE_coin_value_difference_l2923_292360

def total_coins : ℕ := 5050

def penny_value : ℕ := 1
def dime_value : ℕ := 10

def total_value (num_pennies : ℕ) : ℕ :=
  num_pennies * penny_value + (total_coins - num_pennies) * dime_value

theorem coin_value_difference :
  ∃ (max_value min_value : ℕ),
    (∀ (num_pennies : ℕ), 1 ≤ num_pennies ∧ num_pennies ≤ total_coins - 1 →
      min_value ≤ total_value num_pennies ∧ total_value num_pennies ≤ max_value) ∧
    max_value - min_value = 45432 :=
sorry

end NUMINAMATH_CALUDE_coin_value_difference_l2923_292360


namespace NUMINAMATH_CALUDE_function_increment_l2923_292349

theorem function_increment (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 19) ≤ f x + 19) 
  (h2 : ∀ x, f (x + 94) ≥ f x + 94) : 
  ∀ x, f (x + 1) = f x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_increment_l2923_292349


namespace NUMINAMATH_CALUDE_pirate_treasure_l2923_292365

theorem pirate_treasure (x : ℕ) : x > 0 → (
  let paul_coins := x
  let pete_coins := x^2
  paul_coins + pete_coins = 12
) ↔ (
  -- Pete's coins follow the pattern 1, 3, 5, ..., (2x-1)
  pete_coins = x^2 ∧
  -- Paul receives x coins in total
  paul_coins = x ∧
  -- Pete has exactly three times as many coins as Paul
  pete_coins = 3 * paul_coins ∧
  -- All coins are distributed (implied by the other conditions)
  True
) := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_l2923_292365


namespace NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l2923_292314

theorem proposition_false_iff_a_in_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l2923_292314
