import Mathlib

namespace NUMINAMATH_CALUDE_bird_families_flew_away_l2857_285713

/-- The number of bird families that flew away is equal to the difference between
    the total number of bird families and the number of bird families left. -/
theorem bird_families_flew_away (total : ℕ) (left : ℕ) (flew_away : ℕ) 
    (h1 : total = 67) (h2 : left = 35) (h3 : flew_away = total - left) : 
    flew_away = 32 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_flew_away_l2857_285713


namespace NUMINAMATH_CALUDE_square_field_area_l2857_285791

/-- The area of a square field with a diagonal of 26 meters is 338.0625 square meters. -/
theorem square_field_area (d : ℝ) (h : d = 26) : 
  let s := d / Real.sqrt 2
  s^2 = 338.0625 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l2857_285791


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_seven_l2857_285735

theorem no_solution_iff_k_eq_seven :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_seven_l2857_285735


namespace NUMINAMATH_CALUDE_pizza_combinations_l2857_285767

theorem pizza_combinations (n k : ℕ) (h1 : n = 8) (h2 : k = 5) :
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l2857_285767


namespace NUMINAMATH_CALUDE_performance_stability_comparison_l2857_285753

/-- Represents the variance of a student's scores -/
structure StudentVariance where
  value : ℝ
  positive : value > 0

/-- Defines when one performance is more stable than another based on variance -/
def more_stable (a b : StudentVariance) : Prop :=
  a.value > b.value

theorem performance_stability_comparison
  (S_A : StudentVariance)
  (S_B : StudentVariance)
  (h_A : S_A.value = 0.2)
  (h_B : S_B.value = 0.09) :
  more_stable S_A S_B = false :=
by sorry

end NUMINAMATH_CALUDE_performance_stability_comparison_l2857_285753


namespace NUMINAMATH_CALUDE_play_role_assignments_l2857_285752

def number_of_assignments (men women : ℕ) (specific_male_roles specific_female_roles either_gender_roles : ℕ) : ℕ :=
  men * women * (Nat.choose (men + women - 2) either_gender_roles)

theorem play_role_assignments :
  number_of_assignments 6 7 1 1 4 = 13860 := by sorry

end NUMINAMATH_CALUDE_play_role_assignments_l2857_285752


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_280_450_l2857_285790

theorem lcm_gcf_ratio_280_450 : Nat.lcm 280 450 / Nat.gcd 280 450 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_280_450_l2857_285790


namespace NUMINAMATH_CALUDE_simplify_expression_l2857_285784

theorem simplify_expression :
  let x : ℝ := Real.sqrt 2
  let y : ℝ := Real.sqrt 3
  (x + 1) ^ (y - 1) / (x - 1) ^ (y + 1) = 3 - 2 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2857_285784


namespace NUMINAMATH_CALUDE_inscribed_trapezoid_median_l2857_285793

-- Define the trapezoid and its properties
structure InscribedTrapezoid where
  radius : ℝ
  baseAngle : ℝ
  leg : ℝ

-- Define the median (midsegment) of the trapezoid
def median (t : InscribedTrapezoid) : ℝ :=
  sorry

-- Theorem statement
theorem inscribed_trapezoid_median
  (t : InscribedTrapezoid)
  (h1 : t.radius = 13)
  (h2 : t.baseAngle = 30 * π / 180)  -- Convert degrees to radians
  (h3 : t.leg = 10) :
  median t = 12 :=
sorry

end NUMINAMATH_CALUDE_inscribed_trapezoid_median_l2857_285793


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l2857_285732

/-- A circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- The sum of y-coordinates of intersection points between a circle and the y-axis -/
def sumYIntersections (c : Circle) : ℝ :=
  2 * c.b

/-- Theorem: For a circle with center (-3, -4) and radius 7, 
    the sum of y-coordinates of its intersection points with the y-axis is -8 -/
theorem circle_y_axis_intersection_sum :
  ∃ (c : Circle), c.a = -3 ∧ c.b = -4 ∧ c.r = 7 ∧ sumYIntersections c = -8 := by
  sorry


end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l2857_285732


namespace NUMINAMATH_CALUDE_angle_between_hexagon_and_square_diagonal_l2857_285709

/-- A configuration with a square inside a regular hexagon sharing a common vertex. -/
structure SquareInHexagon where
  /-- The measure of an interior angle of the regular hexagon -/
  hexagon_angle : ℝ
  /-- The measure of an interior angle of the square -/
  square_angle : ℝ
  /-- The hexagon is regular -/
  hexagon_regular : hexagon_angle = 120
  /-- The square has right angles -/
  square_right : square_angle = 90

/-- The theorem stating that the angle between the hexagon side and square diagonal is 75° -/
theorem angle_between_hexagon_and_square_diagonal (config : SquareInHexagon) :
  config.hexagon_angle - (config.square_angle / 2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_hexagon_and_square_diagonal_l2857_285709


namespace NUMINAMATH_CALUDE_specific_triangle_area_l2857_285701

/-- The area of the triangle formed by the intersection of three lines -/
def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ :=
  -- Define the area calculation here
  sorry

/-- Theorem: The area of the specific triangle is 256/33 -/
theorem specific_triangle_area :
  triangleArea
    (fun x => (2/3) * x + 4)  -- y = (2/3)x + 4
    (fun x => -3 * x + 9)     -- y = -3x + 9
    (fun x => 2)              -- y = 2
  = 256/33 := by
  sorry

end NUMINAMATH_CALUDE_specific_triangle_area_l2857_285701


namespace NUMINAMATH_CALUDE_average_score_calculation_l2857_285795

/-- Calculates the average score of all students given the proportion of male students,
    the average score of male students, and the average score of female students. -/
def average_score (male_proportion : ℝ) (male_avg : ℝ) (female_avg : ℝ) : ℝ :=
  male_proportion * male_avg + (1 - male_proportion) * female_avg

/-- Theorem stating that when 40% of students are male, with male average score 75
    and female average score 80, the overall average score is 78. -/
theorem average_score_calculation :
  average_score 0.4 75 80 = 78 := by
  sorry

end NUMINAMATH_CALUDE_average_score_calculation_l2857_285795


namespace NUMINAMATH_CALUDE_complex_expression_equals_one_l2857_285768

theorem complex_expression_equals_one : 
  (((4.5 * (1 + 2/3) - 6.75) * (2/3)) / 
   ((3 + 1/3) * 0.3 + (5 + 1/3) * (1/8)) / (2 + 2/3)) + 
  ((1 + 4/11) * 0.22 / 0.3 - 0.96) / 
   ((0.2 - 3/40) * 1.6) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_one_l2857_285768


namespace NUMINAMATH_CALUDE_discount_percentages_l2857_285711

/-- Merchant's markup percentage -/
def markup : ℚ := 75 / 100

/-- Profit percentage for 65 items -/
def profit65 : ℚ := 575 / 1000

/-- Profit percentage for 30 items -/
def profit30 : ℚ := 525 / 1000

/-- Profit percentage for 5 items -/
def profit5 : ℚ := 48 / 100

/-- Calculate discount percentage given profit percentage -/
def calcDiscount (profit : ℚ) : ℚ :=
  (markup - profit) / (1 + markup) * 100

/-- Round to nearest integer -/
def roundToInt (q : ℚ) : ℤ :=
  (q + 1/2).floor

/-- Theorem stating the discount percentages -/
theorem discount_percentages :
  let x := roundToInt (calcDiscount profit5)
  let y := roundToInt (calcDiscount profit30)
  let z := roundToInt (calcDiscount profit65)
  x = 15 ∧ y = 13 ∧ z = 10 ∧
  (5 ≤ x ∧ x ≤ 25) ∧ (5 ≤ y ∧ y ≤ 25) ∧ (5 ≤ z ∧ z ≤ 25) :=
by sorry


end NUMINAMATH_CALUDE_discount_percentages_l2857_285711


namespace NUMINAMATH_CALUDE_archer_probability_l2857_285715

theorem archer_probability (p10 p9 p8 : ℝ) (h1 : p10 = 0.2) (h2 : p9 = 0.3) (h3 : p8 = 0.3) :
  1 - p10 - p9 - p8 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_archer_probability_l2857_285715


namespace NUMINAMATH_CALUDE_min_sum_given_product_l2857_285748

theorem min_sum_given_product (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 8) :
  ∃ (min : ℝ), min = 6 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x * y * z = 8 → x + y + z ≥ min :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l2857_285748


namespace NUMINAMATH_CALUDE_total_mail_delivered_l2857_285747

-- Define the number of junk mail pieces
def junk_mail : ℕ := 6

-- Define the number of magazines
def magazines : ℕ := 5

-- Theorem to prove
theorem total_mail_delivered : junk_mail + magazines = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_mail_delivered_l2857_285747


namespace NUMINAMATH_CALUDE_money_sharing_l2857_285763

theorem money_sharing (jane_share : ℕ) (total : ℕ) : 
  jane_share = 30 →
  (2 : ℕ) * total = jane_share * (2 + 3 + 8) →
  total = 195 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l2857_285763


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2857_285728

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 4 ↔ (x : ℚ) / 4 + 3 / 5 < 7 / 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2857_285728


namespace NUMINAMATH_CALUDE_chocolate_division_l2857_285740

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_for_shaina : ℕ) : 
  total_chocolate = 60 / 7 →
  num_piles = 5 →
  piles_for_shaina = 2 →
  piles_for_shaina * (total_chocolate / num_piles) = 24 / 7 := by
sorry

end NUMINAMATH_CALUDE_chocolate_division_l2857_285740


namespace NUMINAMATH_CALUDE_problem_solution_l2857_285772

/-- Given R = gS² - 6, S = 3, R = 15, and g = 7/3, prove that when S = 5, R = 157/3 -/
theorem problem_solution (g : ℚ) (S R : ℚ) 
  (h1 : R = g * S^2 - 6)
  (h2 : S = 3)
  (h3 : R = 15)
  (h4 : g = 7/3) :
  R = 157/3 ∧ S = 5 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2857_285772


namespace NUMINAMATH_CALUDE_output_for_eight_l2857_285737

-- Define the function f
def f (n : ℕ+) : ℚ := n / (n^2 + 1)

-- State the theorem
theorem output_for_eight : f 8 = 8 / 65 := by sorry

end NUMINAMATH_CALUDE_output_for_eight_l2857_285737


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_achievable_l2857_285743

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  1 / x + 1 / y ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_reciprocal_sum_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_achievable_l2857_285743


namespace NUMINAMATH_CALUDE_lawn_mowing_difference_l2857_285702

theorem lawn_mowing_difference (spring_mowings summer_mowings : ℕ) 
  (h1 : spring_mowings = 8) 
  (h2 : summer_mowings = 5) : 
  spring_mowings - summer_mowings = 3 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_difference_l2857_285702


namespace NUMINAMATH_CALUDE_jake_watching_show_l2857_285756

theorem jake_watching_show (total_show_length : ℝ) (friday_watch_time : ℝ)
  (monday_fraction : ℝ) (tuesday_watch_time : ℝ) (thursday_fraction : ℝ) :
  total_show_length = 52 →
  friday_watch_time = 19 →
  monday_fraction = 1/2 →
  tuesday_watch_time = 4 →
  thursday_fraction = 1/2 →
  ∃ (wednesday_fraction : ℝ),
    wednesday_fraction = 1/4 ∧
    total_show_length = 
      (monday_fraction * 24 + tuesday_watch_time + wednesday_fraction * 24 +
       thursday_fraction * (monday_fraction * 24 + tuesday_watch_time + wednesday_fraction * 24)) +
      friday_watch_time :=
by
  sorry

#check jake_watching_show

end NUMINAMATH_CALUDE_jake_watching_show_l2857_285756


namespace NUMINAMATH_CALUDE_triangle_area_at_most_half_parallelogram_l2857_285745

-- Define a parallelogram
structure Parallelogram where
  area : ℝ
  area_pos : area > 0

-- Define a triangle inscribed in the parallelogram
structure InscribedTriangle (p : Parallelogram) where
  area : ℝ
  area_pos : area > 0
  inscribed : True  -- This represents that the triangle is inscribed in the parallelogram

-- Theorem statement
theorem triangle_area_at_most_half_parallelogram (p : Parallelogram) (t : InscribedTriangle p) :
  t.area ≤ p.area / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_at_most_half_parallelogram_l2857_285745


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_750_l2857_285794

theorem least_integer_greater_than_sqrt_750 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 750 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 750 → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_750_l2857_285794


namespace NUMINAMATH_CALUDE_right_vertex_intersection_l2857_285750

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1

-- Define the line
def line (x y a : ℝ) : Prop := y = x - a

-- State the theorem
theorem right_vertex_intersection (a : ℝ) :
  ellipse 3 0 ∧ line 3 0 a → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_right_vertex_intersection_l2857_285750


namespace NUMINAMATH_CALUDE_translation_theorem_l2857_285726

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translate_right (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_theorem :
  ∀ (A : Point),
    translate_right A 2 = Point.mk 3 2 →
    A = Point.mk 1 2 := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l2857_285726


namespace NUMINAMATH_CALUDE_deriv_zero_necessary_not_sufficient_l2857_285754

-- Define a differentiable function f from ℝ to ℝ
variable (f : ℝ → ℝ) (hf : Differentiable ℝ f)

-- Define what it means for a point to be an extremum
def IsExtremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem deriv_zero_necessary_not_sufficient :
  (∀ x₀, IsExtremum f x₀ → deriv f x₀ = 0) ∧
  ¬(∀ x₀, deriv f x₀ = 0 → IsExtremum f x₀) :=
sorry

end NUMINAMATH_CALUDE_deriv_zero_necessary_not_sufficient_l2857_285754


namespace NUMINAMATH_CALUDE_linear_function_proof_l2857_285707

def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

theorem linear_function_proof (k b : ℝ) 
  (h1 : linear_function k b 3 = 5)
  (h2 : linear_function k b (-4) = -9) :
  (k = 2 ∧ b = -1) ∧ linear_function k b (-1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_proof_l2857_285707


namespace NUMINAMATH_CALUDE_quadratic_roots_l2857_285755

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ x₁^2 - 3 = 0 ∧ x₂^2 - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2857_285755


namespace NUMINAMATH_CALUDE_combined_tax_rate_l2857_285777

theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.30) 
  (h2 : mindy_rate = 0.20) 
  (h3 : income_ratio = 3) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.225 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l2857_285777


namespace NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_18447_l2857_285700

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_greatest_prime_divisor_18447 :
  sum_of_digits (greatest_prime_divisor 18447) = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_18447_l2857_285700


namespace NUMINAMATH_CALUDE_class_size_l2857_285736

theorem class_size (total : ℕ) (brown_eyes : ℕ) (brown_eyes_black_hair : ℕ) : 
  (3 * brown_eyes = 2 * total) →
  (2 * brown_eyes_black_hair = brown_eyes) →
  (brown_eyes_black_hair = 6) →
  total = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l2857_285736


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_sides_l2857_285779

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℚ := n * (n - 3) / 2

/-- Theorem: A regular polygon whose number of diagonals is three times its number of sides has 9 sides -/
theorem regular_polygon_diagonals_sides : ∃ n : ℕ, n > 2 ∧ num_diagonals n = 3 * n ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_sides_l2857_285779


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_and_constant_term_l2857_285782

theorem binomial_coefficient_sum_and_constant_term 
  (x : ℝ) (a : ℝ) (n : ℕ) :
  (1 + a)^n = 32 →
  (∃ (r : ℕ), (n.choose r) * a^r = 80 ∧ 10 - 5*r = 0) →
  a = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_and_constant_term_l2857_285782


namespace NUMINAMATH_CALUDE_right_angled_triangle_l2857_285724

theorem right_angled_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  2 * c * (Real.sin (A / 2))^2 = c - b →
  C = π / 2 := by
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l2857_285724


namespace NUMINAMATH_CALUDE_solution_set_g_range_of_a_l2857_285788

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + |2*x + 5|
def g (x : ℝ) : ℝ := |x - 1| - |2*x|

-- Theorem for part I
theorem solution_set_g (x : ℝ) : g x > -4 ↔ -5 < x ∧ x < -3 := by sorry

-- Theorem for part II
theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, f a x₁ = g x₂) → -6 ≤ a ∧ a ≤ -4 := by sorry

end NUMINAMATH_CALUDE_solution_set_g_range_of_a_l2857_285788


namespace NUMINAMATH_CALUDE_three_zeros_implies_a_equals_four_l2857_285729

-- Define the function f
def f (x a : ℝ) : ℝ := |x^2 - 4*x| - a

-- State the theorem
theorem three_zeros_implies_a_equals_four :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0 ∧
    (∀ x : ℝ, f x a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_implies_a_equals_four_l2857_285729


namespace NUMINAMATH_CALUDE_percentage_problem_l2857_285718

theorem percentage_problem (x : ℝ) : (350 / 100) * x = 140 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2857_285718


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2857_285714

theorem consecutive_integers_square_sum (e f g h : ℤ) : 
  (e + 1 = f) → (f + 1 = g) → (g + 1 = h) →
  (e < f) → (f < g) → (g < h) →
  (e^2 + h^2 = 3405) →
  (f^2 * g^2 = 2689600) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2857_285714


namespace NUMINAMATH_CALUDE_power_neg_square_cube_l2857_285716

theorem power_neg_square_cube (b : ℝ) : ((-b)^2)^3 = b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_neg_square_cube_l2857_285716


namespace NUMINAMATH_CALUDE_building_cost_theorem_l2857_285719

/-- Calculates the total cost of all units in a building -/
def total_cost (total_units : ℕ) (cost_1bed : ℕ) (cost_2bed : ℕ) (num_2bed : ℕ) : ℕ :=
  let num_1bed := total_units - num_2bed
  num_1bed * cost_1bed + num_2bed * cost_2bed

/-- Theorem stating the total cost of all units in the given building configuration -/
theorem building_cost_theorem : 
  total_cost 12 360 450 7 = 4950 := by
  sorry

#eval total_cost 12 360 450 7

end NUMINAMATH_CALUDE_building_cost_theorem_l2857_285719


namespace NUMINAMATH_CALUDE_quadratic_points_order_l2857_285710

/-- The quadratic function f(x) = x² - 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- Theorem: Given points A(-1, y₁), B(1, y₂), C(4, y₃) on the graph of f(x) = x² - 6x + c,
    prove that y₁ > y₂ > y₃ -/
theorem quadratic_points_order (c y₁ y₂ y₃ : ℝ) 
  (h₁ : f c (-1) = y₁)
  (h₂ : f c 1 = y₂)
  (h₃ : f c 4 = y₃) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_order_l2857_285710


namespace NUMINAMATH_CALUDE_a_divisibility_l2857_285773

/-- Sequence a_n defined recursively -/
def a (k : ℤ) : ℕ → ℤ
  | 0 => 0
  | 1 => k
  | (n + 2) => k^2 * a k (n + 1) - a k n

/-- Theorem stating that a_{n+1} * a_n + 1 divides a_{n+1}^2 + a_n^2 for all n -/
theorem a_divisibility (k : ℤ) (n : ℕ) :
  ∃ m : ℤ, (a k (n + 1))^2 + (a k n)^2 = ((a k (n + 1)) * (a k n) + 1) * m := by
  sorry

end NUMINAMATH_CALUDE_a_divisibility_l2857_285773


namespace NUMINAMATH_CALUDE_symmetric_line_y_axis_neg_2x_minus_3_l2857_285734

/-- Given a line with equation y = mx + b, this function returns the equation
    of the line symmetric to it with respect to the y-axis -/
def symmetricLineYAxis (m : ℝ) (b : ℝ) : ℝ → ℝ := fun x ↦ -m * x + b

theorem symmetric_line_y_axis_neg_2x_minus_3 :
  symmetricLineYAxis (-2) (-3) = fun x ↦ 2 * x - 3 := by sorry

end NUMINAMATH_CALUDE_symmetric_line_y_axis_neg_2x_minus_3_l2857_285734


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2857_285770

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2016) + 2016
  f 2016 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2857_285770


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2857_285717

theorem least_subtraction_for_divisibility (n m k : ℕ) (h : n - k ≡ 0 [MOD m]) : 
  ∀ j < k, ¬(n - j ≡ 0 [MOD m]) → k = n % m :=
sorry

-- The specific problem instance
def original_number : ℕ := 1852745
def divisor : ℕ := 251
def subtrahend : ℕ := 130

theorem problem_solution :
  (original_number - subtrahend) % divisor = 0 ∧
  ∀ j < subtrahend, (original_number - j) % divisor ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2857_285717


namespace NUMINAMATH_CALUDE_range_of_a_l2857_285758

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a-1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2857_285758


namespace NUMINAMATH_CALUDE_cards_distribution_l2857_285786

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2857_285786


namespace NUMINAMATH_CALUDE_four_item_match_probability_correct_match_probability_theorem_l2857_285727

/-- The probability of correctly matching n distinct items to n distinct positions when guessing randomly. -/
def correct_match_probability (n : ℕ) : ℚ :=
  1 / n.factorial

/-- Theorem: For 4 items, the probability of a correct random match is 1/24. -/
theorem four_item_match_probability :
  correct_match_probability 4 = 1 / 24 := by
  sorry

/-- Theorem: The probability of correctly matching n distinct items to n distinct positions
    when guessing randomly is 1/n!. -/
theorem correct_match_probability_theorem (n : ℕ) :
  correct_match_probability n = 1 / n.factorial := by
  sorry

end NUMINAMATH_CALUDE_four_item_match_probability_correct_match_probability_theorem_l2857_285727


namespace NUMINAMATH_CALUDE_right_triangle_area_l2857_285776

/-- The area of a right triangle with hypotenuse 12 inches and one angle 30° is 18√3 square inches -/
theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 12 →  -- hypotenuse is 12 inches
  θ = 30 * π / 180 →  -- one angle is 30°
  area = h * h * Real.sin θ * Real.cos θ / 2 →  -- area formula for right triangle
  area = 18 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2857_285776


namespace NUMINAMATH_CALUDE_floor_sum_example_l2857_285721

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l2857_285721


namespace NUMINAMATH_CALUDE_p_shape_point_count_l2857_285781

/-- Calculates the number of distinct points on a "P" shape derived from a square --/
def count_points_on_p_shape (side_length : ℕ) (point_interval : ℕ) : ℕ :=
  let points_per_side := side_length / point_interval + 1
  let total_points := points_per_side * 3
  total_points - 2

theorem p_shape_point_count :
  count_points_on_p_shape 10 1 = 31 := by
  sorry

#eval count_points_on_p_shape 10 1

end NUMINAMATH_CALUDE_p_shape_point_count_l2857_285781


namespace NUMINAMATH_CALUDE_boxes_given_away_l2857_285787

def total_cupcakes : ℕ := 53
def cupcakes_left_at_home : ℕ := 2
def cupcakes_per_box : ℕ := 3

theorem boxes_given_away : 
  (total_cupcakes - cupcakes_left_at_home) / cupcakes_per_box = 17 := by
  sorry

end NUMINAMATH_CALUDE_boxes_given_away_l2857_285787


namespace NUMINAMATH_CALUDE_ellipse_equation_l2857_285708

/-- Given a parabola C₁ and an ellipse C₂ with the following properties:
    1. C₁: x² = 4y
    2. C₂: y²/a² + x²/b² = 1, where a > b > 0
    3. The focus F of C₁ is also a focus of C₂
    4. The common chord length of C₁ and C₂ is 2√6
    5. a² = b² + 1
    Prove that the equation of C₂ is y²/9 + x²/8 = 1 -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h_ab_relation : a^2 = b^2 + 1) 
  (h_common_chord : ∃ x y : ℝ, x^2 = 4*y ∧ y^2/a^2 + x^2/b^2 = 1 ∧ x^2 + y^2 = 36) :
  a^2 = 9 ∧ b^2 = 8 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ellipse_equation_l2857_285708


namespace NUMINAMATH_CALUDE_family_ages_correct_l2857_285764

-- Define the family members' ages as natural numbers
def son_age : Nat := 7
def daughter_age : Nat := 12
def man_age : Nat := 27
def wife_age : Nat := 22
def father_age : Nat := 59

-- State the theorem
theorem family_ages_correct :
  -- Man is 20 years older than son
  man_age = son_age + 20 ∧
  -- Man is 15 years older than daughter
  man_age = daughter_age + 15 ∧
  -- In two years, man's age will be twice son's age
  man_age + 2 = 2 * (son_age + 2) ∧
  -- In two years, man's age will be three times daughter's age
  man_age + 2 = 3 * (daughter_age + 2) ∧
  -- Wife is 5 years younger than man
  wife_age = man_age - 5 ∧
  -- In 6 years, wife will be twice as old as daughter
  wife_age + 6 = 2 * (daughter_age + 6) ∧
  -- Father is 32 years older than man
  father_age = man_age + 32 := by
  sorry


end NUMINAMATH_CALUDE_family_ages_correct_l2857_285764


namespace NUMINAMATH_CALUDE_leila_bought_two_armchairs_l2857_285731

/-- Represents the living room set purchase --/
structure LivingRoomSet where
  sofaCost : ℕ
  armchairCost : ℕ
  coffeeTableCost : ℕ
  totalCost : ℕ

/-- Calculates the number of armchairs in the living room set --/
def numberOfArmchairs (set : LivingRoomSet) : ℕ :=
  (set.totalCost - set.sofaCost - set.coffeeTableCost) / set.armchairCost

/-- Theorem stating that Leila bought 2 armchairs --/
theorem leila_bought_two_armchairs (set : LivingRoomSet)
    (h1 : set.sofaCost = 1250)
    (h2 : set.armchairCost = 425)
    (h3 : set.coffeeTableCost = 330)
    (h4 : set.totalCost = 2430) :
    numberOfArmchairs set = 2 := by
  sorry

#eval numberOfArmchairs {
  sofaCost := 1250,
  armchairCost := 425,
  coffeeTableCost := 330,
  totalCost := 2430
}

end NUMINAMATH_CALUDE_leila_bought_two_armchairs_l2857_285731


namespace NUMINAMATH_CALUDE_max_value_of_x_l2857_285749

theorem max_value_of_x (x y z : ℝ) (sum_eq : x + y + z = 7) (prod_eq : x*y + x*z + y*z = 12) :
  x ≤ 1 ∧ ∃ (a b : ℝ), a + b + 1 = 7 ∧ a*b + a*1 + b*1 = 12 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_l2857_285749


namespace NUMINAMATH_CALUDE_relationship_abc_l2857_285744

theorem relationship_abc : 3^(1/10) > (1/2)^(1/10) ∧ (1/2)^(1/10) > (-1/2)^3 := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2857_285744


namespace NUMINAMATH_CALUDE_percentage_increase_l2857_285769

theorem percentage_increase (x y z : ℝ) : 
  x = 1.25 * y →
  x + y + z = 925 →
  z = 250 →
  (y - z) / z = 0.2 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2857_285769


namespace NUMINAMATH_CALUDE_count_three_digit_divisible_by_nine_l2857_285792

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem count_three_digit_divisible_by_nine :
  let min_num : ℕ := 108
  let max_num : ℕ := 999
  let common_diff : ℕ := 9
  (∀ n, is_three_digit n ∧ n % 9 = 0 → min_num ≤ n ∧ n ≤ max_num) →
  (∀ n, min_num ≤ n ∧ n ≤ max_num ∧ n % 9 = 0 → is_three_digit n) →
  (∀ n m, min_num ≤ n ∧ n < m ∧ m ≤ max_num ∧ n % 9 = 0 ∧ m % 9 = 0 → m - n = common_diff) →
  (Finset.filter (λ n => n % 9 = 0) (Finset.range (max_num - min_num + 1))).card + 1 = 100 :=
by sorry

end NUMINAMATH_CALUDE_count_three_digit_divisible_by_nine_l2857_285792


namespace NUMINAMATH_CALUDE_peach_difference_l2857_285778

def red_peaches : ℕ := 19
def yellow_peaches : ℕ := 11

theorem peach_difference : red_peaches - yellow_peaches = 8 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l2857_285778


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l2857_285774

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (2*m + 1)*x + m^2 + m

-- Define the condition for the roots
def root_condition (a b : ℝ) : Prop := (2*a + b) * (a + 2*b) = 20

-- Theorem statement
theorem quadratic_roots_theorem (m : ℝ) :
  (∃ a b : ℝ, a ≠ b ∧ quadratic m a = 0 ∧ quadratic m b = 0) ∧
  (∀ a b : ℝ, quadratic m a = 0 → quadratic m b = 0 → root_condition a b → (m = -2 ∨ m = 1)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l2857_285774


namespace NUMINAMATH_CALUDE_best_calorie_deal_l2857_285765

-- Define the food options
structure FoodOption where
  name : String
  quantity : Nat
  price : Nat
  caloriesPerItem : Nat

-- Define the function to calculate calories per dollar
def caloriesPerDollar (option : FoodOption) : Rat :=
  (option.quantity * option.caloriesPerItem : Rat) / option.price

-- Define the food options
def burritos : FoodOption := ⟨"Burritos", 10, 6, 120⟩
def burgers : FoodOption := ⟨"Burgers", 5, 8, 400⟩
def pizza : FoodOption := ⟨"Pizza", 8, 10, 300⟩
def donuts : FoodOption := ⟨"Donuts", 15, 12, 250⟩

-- Define the list of food options
def foodOptions : List FoodOption := [burritos, burgers, pizza, donuts]

-- Theorem statement
theorem best_calorie_deal :
  (caloriesPerDollar donuts = 312.5) ∧
  (∀ option ∈ foodOptions, caloriesPerDollar option ≤ caloriesPerDollar donuts) ∧
  (caloriesPerDollar donuts - caloriesPerDollar burgers = 62.5) :=
sorry

end NUMINAMATH_CALUDE_best_calorie_deal_l2857_285765


namespace NUMINAMATH_CALUDE_perimeter_of_specific_triangle_l2857_285703

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of DP, where P is the tangency point on DE -/
  dp : ℝ
  /-- The length of PE, where P is the tangency point on DE -/
  pe : ℝ
  /-- The length of the tangent from vertex F to the circle -/
  ft : ℝ

/-- The perimeter of the triangle -/
def perimeter (t : TriangleWithInscribedCircle) : ℝ :=
  2 * (t.dp + t.pe + t.ft)

theorem perimeter_of_specific_triangle :
  let t : TriangleWithInscribedCircle := {
    r := 13,
    dp := 17,
    pe := 31,
    ft := 20
  }
  perimeter t = 136 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_triangle_l2857_285703


namespace NUMINAMATH_CALUDE_range_of_abc_squared_l2857_285706

theorem range_of_abc_squared (a b c : ℝ) 
  (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) 
  (h4 : -2 < c) (h5 : c < -1) : 
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_abc_squared_l2857_285706


namespace NUMINAMATH_CALUDE_white_daisies_count_l2857_285766

theorem white_daisies_count :
  ∀ (white pink red : ℕ),
    pink = 9 * white →
    red = 4 * pink - 3 →
    white + pink + red = 273 →
    white = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_white_daisies_count_l2857_285766


namespace NUMINAMATH_CALUDE_largest_integer_under_sqrt_constraint_l2857_285796

theorem largest_integer_under_sqrt_constraint : 
  ∀ x : ℤ, (Real.sqrt (x^2 : ℝ) < 15) → x ≤ 14 ∧ ∃ y : ℤ, y > x ∧ ¬(Real.sqrt (y^2 : ℝ) < 15) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_under_sqrt_constraint_l2857_285796


namespace NUMINAMATH_CALUDE_stratified_sampling_seniors_l2857_285798

theorem stratified_sampling_seniors (total_students : ℕ) (seniors : ℕ) (sample_size : ℕ) 
  (h_total : total_students = 900)
  (h_seniors : seniors = 400)
  (h_sample : sample_size = 45) :
  (seniors * sample_size) / total_students = 20 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_seniors_l2857_285798


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l2857_285759

-- Define the equations
def equation1 (x : ℝ) : Prop := (x - 1)^2 - 1 = 15
def equation2 (x : ℝ) : Prop := (1/3) * (x + 3)^3 - 9 = 0

-- Theorem for equation 1
theorem equation1_solutions : 
  (∃ x : ℝ, equation1 x) ↔ (equation1 5 ∧ equation1 (-3)) :=
sorry

-- Theorem for equation 2
theorem equation2_solution : 
  (∃ x : ℝ, equation2 x) ↔ equation2 0 :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l2857_285759


namespace NUMINAMATH_CALUDE_triangle_tangent_ratio_l2857_285738

theorem triangle_tangent_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  a * Real.sin C = c * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  a * Real.cos B - b * Real.cos A = (3/5) * c →
  Real.tan A / Real.tan B = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_ratio_l2857_285738


namespace NUMINAMATH_CALUDE_existence_of_index_l2857_285712

theorem existence_of_index (n : ℕ) (x : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_x : ∀ i, i ∈ Finset.range (n + 1) → 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i ∈ Finset.range n, x 1 * (1 - x (i + 1)) ≥ (1 / 4) * x 1 * (1 - x n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_index_l2857_285712


namespace NUMINAMATH_CALUDE_problem_statement_l2857_285733

theorem problem_statement (x y : ℝ) (h1 : x + y = 7) (h2 : x * y = 10) : 3 * x^2 + 3 * y^2 = 87 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2857_285733


namespace NUMINAMATH_CALUDE_both_products_not_qualified_l2857_285746

-- Define the qualification rates for Factory A and Factory B
def qualification_rate_A : ℝ := 0.9
def qualification_rate_B : ℝ := 0.8

-- Define the probability that both products are not qualified
def both_not_qualified : ℝ := (1 - qualification_rate_A) * (1 - qualification_rate_B)

-- Theorem statement
theorem both_products_not_qualified :
  both_not_qualified = 0.02 :=
sorry

end NUMINAMATH_CALUDE_both_products_not_qualified_l2857_285746


namespace NUMINAMATH_CALUDE_square_roots_theorem_l2857_285780

theorem square_roots_theorem (x a : ℝ) (hx : x > 0) 
  (h1 : (a + 1) ^ 2 = x) (h2 : (2 * a - 7) ^ 2 = x) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l2857_285780


namespace NUMINAMATH_CALUDE_walking_path_area_l2857_285785

/-- The area of a circular walking path -/
theorem walking_path_area (outer_radius inner_radius : ℝ) 
  (h_outer : outer_radius = 26)
  (h_inner : inner_radius = 16) : 
  π * (outer_radius^2 - inner_radius^2) = 420 * π := by
  sorry

#check walking_path_area

end NUMINAMATH_CALUDE_walking_path_area_l2857_285785


namespace NUMINAMATH_CALUDE_balls_in_boxes_l2857_285730

def num_balls : ℕ := 6
def num_boxes : ℕ := 3

theorem balls_in_boxes :
  (num_boxes ^ num_balls : ℕ) = 729 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l2857_285730


namespace NUMINAMATH_CALUDE_boat_current_speed_l2857_285723

theorem boat_current_speed 
  (boat_speed : ℝ) 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (h1 : boat_speed = 16)
  (h2 : upstream_time = 20 / 60)
  (h3 : downstream_time = 15 / 60) :
  ∃ (current_speed : ℝ),
    (boat_speed - current_speed) * upstream_time = 
    (boat_speed + current_speed) * downstream_time ∧ 
    current_speed = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_current_speed_l2857_285723


namespace NUMINAMATH_CALUDE_complement_A_implies_m_eq_4_l2857_285783

def S : Finset ℕ := {1, 2, 3, 4}

def A (m : ℕ) : Finset ℕ := S.filter (λ x => x^2 - 5*x + m = 0)

theorem complement_A_implies_m_eq_4 :
  (S \ A m) = {2, 3} → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_complement_A_implies_m_eq_4_l2857_285783


namespace NUMINAMATH_CALUDE_inequality_range_l2857_285751

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 ≤ 0) ↔ -2 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l2857_285751


namespace NUMINAMATH_CALUDE_money_difference_l2857_285761

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- Mrs. Hilt's coin counts -/
def mrs_hilt_coins : Fin 3 → ℕ
| 0 => 2  -- pennies
| 1 => 2  -- nickels
| 2 => 2  -- dimes
| _ => 0

/-- Jacob's coin counts -/
def jacob_coins : Fin 3 → ℕ
| 0 => 4  -- pennies
| 1 => 1  -- nickel
| 2 => 1  -- dime
| _ => 0

/-- The value of a coin type in dollars -/
def coin_value : Fin 3 → ℚ
| 0 => penny_value
| 1 => nickel_value
| 2 => dime_value
| _ => 0

/-- Calculate the total value of coins -/
def total_value (coins : Fin 3 → ℕ) : ℚ :=
  (coins 0 : ℚ) * penny_value + (coins 1 : ℚ) * nickel_value + (coins 2 : ℚ) * dime_value

theorem money_difference :
  total_value mrs_hilt_coins - total_value jacob_coins = 13 / 100 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l2857_285761


namespace NUMINAMATH_CALUDE_largest_integer_problem_l2857_285705

theorem largest_integer_problem (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d →  -- four different integers
  (a + b + c + d) / 4 = 74 →  -- average is 74
  a ≥ 29 →  -- smallest integer is at least 29
  d ≤ 206 :=  -- largest integer is at most 206
by sorry

end NUMINAMATH_CALUDE_largest_integer_problem_l2857_285705


namespace NUMINAMATH_CALUDE_cheapest_caterer_l2857_285799

def first_caterer_cost (people : ℕ) : ℚ := 120 + 18 * people
def second_caterer_cost (people : ℕ) : ℚ := 250 + 15 * people

theorem cheapest_caterer (people : ℕ) :
  (people ≥ 44 → second_caterer_cost people ≤ first_caterer_cost people) ∧
  (people < 44 → second_caterer_cost people > first_caterer_cost people) :=
sorry

end NUMINAMATH_CALUDE_cheapest_caterer_l2857_285799


namespace NUMINAMATH_CALUDE_unique_solution_l2857_285797

def is_divisible (x y : ℕ) : Prop := ∃ k : ℕ, x = y * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def condition1 (a b : ℕ) : Prop := is_divisible (a^2 + 4*a + 3) b

def condition2 (a b : ℕ) : Prop := a^2 + a*b - 6*b^2 - 2*a - 16*b - 8 = 0

def condition3 (a b : ℕ) : Prop := is_divisible (a + 2*b + 1) 4

def condition4 (a b : ℕ) : Prop := is_prime (a + 6*b + 1)

def exactly_three_true (a b : ℕ) : Prop :=
  (condition1 a b ∧ condition2 a b ∧ condition3 a b ∧ ¬condition4 a b) ∨
  (condition1 a b ∧ condition2 a b ∧ ¬condition3 a b ∧ condition4 a b) ∨
  (condition1 a b ∧ ¬condition2 a b ∧ condition3 a b ∧ condition4 a b) ∨
  (¬condition1 a b ∧ condition2 a b ∧ condition3 a b ∧ condition4 a b)

theorem unique_solution :
  ∀ a b : ℕ, exactly_three_true a b ↔ (a = 6 ∧ b = 1) ∨ (a = 18 ∧ b = 7) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2857_285797


namespace NUMINAMATH_CALUDE_window_side_length_is_five_l2857_285742

/-- Represents the dimensions of a glass pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the window configuration -/
structure Window where
  pane : Pane
  rows : ℕ
  columns : ℕ
  border_width : ℝ

/-- Calculates the side length of a square window -/
def window_side_length (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- Theorem stating that the window's side length is 5 inches -/
theorem window_side_length_is_five (w : Window) 
  (h_square : window_side_length w = w.rows * w.pane.height + (w.rows + 1) * w.border_width)
  (h_rows : w.rows = 2)
  (h_columns : w.columns = 3)
  (h_border : w.border_width = 1) :
  window_side_length w = 5 := by
  sorry

#check window_side_length_is_five

end NUMINAMATH_CALUDE_window_side_length_is_five_l2857_285742


namespace NUMINAMATH_CALUDE_triangle_side_angle_relation_l2857_285757

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the theorem
theorem triangle_side_angle_relation (t : Triangle) 
  (h1 : t.α > 0 ∧ t.β > 0 ∧ t.γ > 0)
  (h2 : t.α + t.β + t.γ = Real.pi)
  (h3 : 3 * t.α + 2 * t.β = Real.pi) :
  t.a^2 + t.b * t.c - t.c^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_angle_relation_l2857_285757


namespace NUMINAMATH_CALUDE_cube_surface_area_l2857_285722

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 20) :
  6 * edge_length^2 = 2400 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2857_285722


namespace NUMINAMATH_CALUDE_max_profit_is_180_l2857_285741

/-- Represents a neighborhood with its characteristics --/
structure Neighborhood where
  homes : ℕ
  boxesPerHome : ℕ
  pricePerBox : ℚ
  transportCost : ℚ

/-- Calculates the profit for a given neighborhood --/
def profit (n : Neighborhood) : ℚ :=
  n.homes * n.boxesPerHome * n.pricePerBox - n.transportCost

/-- Checks if the neighborhood is within the stock limit --/
def withinStockLimit (n : Neighborhood) (stockLimit : ℕ) : Prop :=
  n.homes * n.boxesPerHome ≤ stockLimit

/-- The main theorem stating the maximum profit --/
theorem max_profit_is_180 (stockLimit : ℕ) (A B C D : Neighborhood)
  (hStock : stockLimit = 50)
  (hA : A = { homes := 12, boxesPerHome := 3, pricePerBox := 3, transportCost := 10 })
  (hB : B = { homes := 8, boxesPerHome := 6, pricePerBox := 4, transportCost := 15 })
  (hC : C = { homes := 15, boxesPerHome := 2, pricePerBox := 5/2, transportCost := 5 })
  (hD : D = { homes := 5, boxesPerHome := 8, pricePerBox := 5, transportCost := 20 })
  (hAStock : withinStockLimit A stockLimit)
  (hBStock : withinStockLimit B stockLimit)
  (hCStock : withinStockLimit C stockLimit)
  (hDStock : withinStockLimit D stockLimit) :
  (max (profit A) (max (profit B) (max (profit C) (profit D)))) = 180 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_180_l2857_285741


namespace NUMINAMATH_CALUDE_gel_pen_price_ratio_l2857_285704

/-- Represents the price ratio of gel pens to ballpoint pens -/
def price_ratio (x y : ℕ) (b g : ℝ) : Prop :=
  let total := x * b + y * g
  (x + y) * g = 4 * total ∧ (x + y) * b = (1 / 2) * total ∧ g = 8 * b

/-- Theorem stating that under the given conditions, a gel pen costs 8 times as much as a ballpoint pen -/
theorem gel_pen_price_ratio {x y : ℕ} {b g : ℝ} (h : price_ratio x y b g) :
  g = 8 * b := by
  sorry

end NUMINAMATH_CALUDE_gel_pen_price_ratio_l2857_285704


namespace NUMINAMATH_CALUDE_arcade_tickets_l2857_285739

theorem arcade_tickets (dave_spent : ℕ) (dave_left : ℕ) (alex_spent : ℕ) (alex_left : ℕ) :
  dave_spent = 43 →
  dave_left = 55 →
  alex_spent = 65 →
  alex_left = 42 →
  dave_spent + dave_left + alex_spent + alex_left = 205 := by
  sorry

end NUMINAMATH_CALUDE_arcade_tickets_l2857_285739


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2857_285789

/-- An even function on ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem sufficient_not_necessary
  (f : ℝ → ℝ) (hf : EvenFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ - f x₂ = 0) ∧
  (∃ x₁ x₂ : ℝ, f x₁ - f x₂ = 0 ∧ x₁ + x₂ ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2857_285789


namespace NUMINAMATH_CALUDE_total_potatoes_eq_sum_l2857_285720

/-- The number of potatoes mother bought -/
def total_potatoes : ℕ := sorry

/-- The number of potatoes used for salads -/
def salad_potatoes : ℕ := 15

/-- The number of potatoes used for mashed potatoes -/
def mashed_potatoes : ℕ := 24

/-- The number of leftover potatoes -/
def leftover_potatoes : ℕ := 13

/-- Theorem stating that the total number of potatoes is equal to the sum of
    potatoes used for salads, mashed potatoes, and leftover potatoes -/
theorem total_potatoes_eq_sum :
  total_potatoes = salad_potatoes + mashed_potatoes + leftover_potatoes := by sorry

end NUMINAMATH_CALUDE_total_potatoes_eq_sum_l2857_285720


namespace NUMINAMATH_CALUDE_parabola_through_origin_l2857_285760

/-- A parabola passing through the origin can be represented by the equation y = ax^2 + bx,
    where a and b are real numbers and at least one of them is non-zero. -/
theorem parabola_through_origin :
  ∀ (f : ℝ → ℝ), (∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0) →
  (f 0 = 0) →
  (∀ x, ∃ y, f x = y ∧ y = a * x^2 + b * x) →
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (∀ x, f x = a * x^2 + b * x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_through_origin_l2857_285760


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l2857_285762

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State the theorem
theorem parallel_planes_condition 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β ∧ α ≠ γ ∧ β ≠ γ)
  (h_parallel : parallel m n)
  (h_perp1 : perpendicular n α)
  (h_perp2 : perpendicular m β) :
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l2857_285762


namespace NUMINAMATH_CALUDE_vector_perpendicular_and_parallel_l2857_285775

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

theorem vector_perpendicular_and_parallel (k : ℝ) :
  (∀ i : Fin 2, (k * a i + b i) * (a i - 3 * b i) = 0) → k = 19 ∧
  (∃ t : ℝ, ∀ i : Fin 2, k * a i + b i = t * (a i - 3 * b i)) → k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_and_parallel_l2857_285775


namespace NUMINAMATH_CALUDE_ratio_fraction_equality_l2857_285771

theorem ratio_fraction_equality (a b c : ℝ) (h : a ≠ 0) :
  (a : ℝ) / 2 = b / 3 ∧ a / 2 = c / 4 →
  (a - b + c) / b = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_fraction_equality_l2857_285771


namespace NUMINAMATH_CALUDE_group_size_problem_l2857_285725

theorem group_size_problem (total_cents : ℕ) (h1 : total_cents = 64736) : ∃ n : ℕ, n * n = total_cents ∧ n = 254 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l2857_285725
