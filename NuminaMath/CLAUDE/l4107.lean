import Mathlib

namespace NUMINAMATH_CALUDE_relay_race_first_leg_time_l4107_410775

/-- Represents a relay race with two runners -/
structure RelayRace where
  y_time : ℝ  -- Time taken by runner y for the first leg
  z_time : ℝ  -- Time taken by runner z for the second leg

/-- Theorem: In a relay race where the second runner takes 26 seconds and the average time per leg is 42 seconds, the first runner takes 58 seconds. -/
theorem relay_race_first_leg_time (race : RelayRace) 
  (h1 : race.z_time = 26)
  (h2 : (race.y_time + race.z_time) / 2 = 42) : 
  race.y_time = 58 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_first_leg_time_l4107_410775


namespace NUMINAMATH_CALUDE_complex_coordinate_l4107_410790

theorem complex_coordinate (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = (2 + 4*i) / i → (z.re = 4 ∧ z.im = -2) :=
by sorry

end NUMINAMATH_CALUDE_complex_coordinate_l4107_410790


namespace NUMINAMATH_CALUDE_distance_AB_is_130_l4107_410764

-- Define the speeds of the three people
def speed_A : ℝ := 3
def speed_B : ℝ := 2
def speed_C : ℝ := 1

-- Define the initial distance traveled by A
def initial_distance_A : ℝ := 50

-- Define the distance between C and D
def distance_CD : ℝ := 12

-- Theorem statement
theorem distance_AB_is_130 :
  let total_distance := 4 * (speed_A + speed_B + speed_C) * distance_CD + initial_distance_A
  total_distance = 130 := by sorry

end NUMINAMATH_CALUDE_distance_AB_is_130_l4107_410764


namespace NUMINAMATH_CALUDE_f_2015_equals_2_l4107_410700

/-- A function satisfying the given conditions -/
def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 6) + f x = 0) ∧
  (∀ x : ℝ, f (x - 1) = f (3 - x)) ∧
  (f 1 = -2)

/-- Theorem stating that any function satisfying the conditions has f(2015) = 2 -/
theorem f_2015_equals_2 (f : ℝ → ℝ) (hf : f_conditions f) : f 2015 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2015_equals_2_l4107_410700


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_sup_ratio_l4107_410731

/-- A quadrilateral inscribed in a unit circle with two parallel sides -/
structure InscribedQuadrilateral where
  /-- The difference between the lengths of the parallel sides -/
  d : ℝ
  /-- The distance from the intersection of the diagonals to the center of the circle -/
  h : ℝ
  /-- The difference d is positive -/
  d_pos : d > 0
  /-- The quadrilateral is inscribed in a unit circle -/
  h_bound : h ≤ 1

/-- The supremum of d/h for inscribed quadrilaterals is 2 -/
theorem inscribed_quadrilateral_sup_ratio :
  ∀ ε > 0, ∃ q : InscribedQuadrilateral, q.d / q.h > 2 - ε ∧ ∀ q' : InscribedQuadrilateral, q'.d / q'.h ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_sup_ratio_l4107_410731


namespace NUMINAMATH_CALUDE_jenny_cat_expense_first_year_l4107_410728

/-- Jenny's cat expenses for the first year -/
def jenny_cat_expense : ℕ → ℕ → ℕ → ℕ → ℕ := fun adoption_fee vet_cost monthly_food_cost toy_cost =>
  let shared_cost := adoption_fee + vet_cost + (monthly_food_cost * 12)
  (shared_cost / 2) + toy_cost

/-- Theorem: Jenny's cat expense for the first year is $625 -/
theorem jenny_cat_expense_first_year :
  jenny_cat_expense 50 500 25 200 = 625 := by
  sorry

end NUMINAMATH_CALUDE_jenny_cat_expense_first_year_l4107_410728


namespace NUMINAMATH_CALUDE_allens_mother_age_l4107_410736

theorem allens_mother_age (allen_age mother_age : ℕ) : 
  allen_age = mother_age - 25 →
  allen_age + mother_age + 6 = 41 →
  mother_age = 30 := by
sorry

end NUMINAMATH_CALUDE_allens_mother_age_l4107_410736


namespace NUMINAMATH_CALUDE_fraction_calculation_and_comparison_l4107_410703

theorem fraction_calculation_and_comparison : 
  let x := (1/6 - 1/7) / (1/3 - 1/5)
  x = 5/28 ∧ 
  x ≠ 1/4 ∧ 
  x ≠ 1/3 ∧ 
  x ≠ 1/2 ∧ 
  x ≠ 2/5 ∧ 
  x ≠ 3/5 ∧
  x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_and_comparison_l4107_410703


namespace NUMINAMATH_CALUDE_C_is_largest_l4107_410788

-- Define A, B, and C
def A : ℚ := 2010/2009 + 2010/2011
def B : ℚ := (2010/2011) * (2012/2011)
def C : ℚ := 2011/2010 + 2011/2012 + 1/10000

-- Theorem statement
theorem C_is_largest : C > A ∧ C > B := by
  sorry

end NUMINAMATH_CALUDE_C_is_largest_l4107_410788


namespace NUMINAMATH_CALUDE_inequality_solution_l4107_410746

theorem inequality_solution (a : ℝ) : 
  (∃ x : ℝ, 2 * x - (1/3) * a ≤ 0 ∧ x ≤ 2) → a = 12 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4107_410746


namespace NUMINAMATH_CALUDE_smallest_integer_above_root_sum_sixth_power_l4107_410724

theorem smallest_integer_above_root_sum_sixth_power :
  ∃ n : ℕ, n = 3323 ∧ (∀ m : ℕ, m < n → (m : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^6) ∧
  n > (Real.sqrt 5 + Real.sqrt 3)^6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_above_root_sum_sixth_power_l4107_410724


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l4107_410747

theorem factorization_of_quadratic (a : ℝ) : a^2 + 5*a = a*(a+5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l4107_410747


namespace NUMINAMATH_CALUDE_max_product_of_sum_one_l4107_410765

theorem max_product_of_sum_one (a b : ℝ) : a + b = 1 → ab ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_sum_one_l4107_410765


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l4107_410785

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; -1, 2]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, -3]

theorem matrix_sum_theorem :
  A + B = !![3, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_matrix_sum_theorem_l4107_410785


namespace NUMINAMATH_CALUDE_histogram_classes_l4107_410744

def max_value : ℝ := 169
def min_value : ℝ := 143
def class_interval : ℝ := 3

theorem histogram_classes : 
  ∃ (n : ℕ), n = ⌈(max_value - min_value) / class_interval⌉ ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_histogram_classes_l4107_410744


namespace NUMINAMATH_CALUDE_relative_errors_equal_l4107_410733

theorem relative_errors_equal (length1 length2 error1 error2 : ℝ) 
  (h1 : length1 = 25)
  (h2 : length2 = 150)
  (h3 : error1 = 0.05)
  (h4 : error2 = 0.3) : 
  (error1 / length1) = (error2 / length2) := by
  sorry

end NUMINAMATH_CALUDE_relative_errors_equal_l4107_410733


namespace NUMINAMATH_CALUDE_total_dolls_l4107_410741

def doll_problem (vera_dolls : ℕ) (sophie_dolls : ℕ) (aida_dolls : ℕ) : Prop :=
  vera_dolls = 20 ∧
  sophie_dolls = 2 * vera_dolls ∧
  aida_dolls = 2 * sophie_dolls ∧
  aida_dolls + sophie_dolls + vera_dolls = 140

theorem total_dolls :
  ∃ (vera_dolls sophie_dolls aida_dolls : ℕ),
    doll_problem vera_dolls sophie_dolls aida_dolls :=
by
  sorry

end NUMINAMATH_CALUDE_total_dolls_l4107_410741


namespace NUMINAMATH_CALUDE_family_reunion_count_l4107_410755

/-- The number of people at a family reunion -/
def family_reunion_attendance (male_adults female_adults children : ℕ) : ℕ :=
  male_adults + female_adults + children

/-- Theorem stating the total number of people at the family reunion -/
theorem family_reunion_count :
  ∃ (male_adults female_adults children : ℕ),
    male_adults = 100 ∧
    female_adults = male_adults + 50 ∧
    children = 2 * (male_adults + female_adults) ∧
    family_reunion_attendance male_adults female_adults children = 750 :=
by
  sorry


end NUMINAMATH_CALUDE_family_reunion_count_l4107_410755


namespace NUMINAMATH_CALUDE_bus_problem_l4107_410754

theorem bus_problem (initial_children on_bus off_bus final_children : ℕ) :
  initial_children = 22 →
  on_bus = 40 →
  final_children = 2 →
  initial_children + on_bus - off_bus = final_children →
  off_bus = 60 :=
by sorry

end NUMINAMATH_CALUDE_bus_problem_l4107_410754


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l4107_410718

theorem exponential_equation_solution :
  ∃ y : ℝ, (20 : ℝ)^y * 200^(3*y) = 8000^7 ∧ y = 3 :=
by sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l4107_410718


namespace NUMINAMATH_CALUDE_binary_to_decimal_octal_conversion_l4107_410705

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_to_decimal_octal_conversion :
  (binary_to_decimal binary_101101 = 45) ∧
  (decimal_to_octal 45 = [5, 5]) := by
sorry

end NUMINAMATH_CALUDE_binary_to_decimal_octal_conversion_l4107_410705


namespace NUMINAMATH_CALUDE_problem1_l4107_410794

theorem problem1 (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 + y^2 = 30) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l4107_410794


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l4107_410784

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- The property of the trapezoid that the segment joining the midpoints of the diagonals
    is half the difference of the bases -/
def trapezoid_property (t : Trapezoid) : Prop :=
  t.midpoint_segment = (t.longer_base - t.shorter_base) / 2

theorem trapezoid_shorter_base (t : Trapezoid) 
  (h1 : t.longer_base = 113)
  (h2 : t.midpoint_segment = 5)
  (h3 : trapezoid_property t) :
  t.shorter_base = 103 := by
  sorry

#check trapezoid_shorter_base

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l4107_410784


namespace NUMINAMATH_CALUDE_inverse_A_cubed_l4107_410795

theorem inverse_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) : 
  A⁻¹ = !![3, 7; -2, -4] → (A^3)⁻¹ = !![11, 17; 2, 6] := by
  sorry

end NUMINAMATH_CALUDE_inverse_A_cubed_l4107_410795


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4107_410712

theorem inequality_equivalence (x : ℝ) : 
  (x - 2) / (x - 4) ≤ 3 ↔ 4 < x ∧ x ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4107_410712


namespace NUMINAMATH_CALUDE_equation_solution_l4107_410749

theorem equation_solution :
  ∃ x : ℝ, (10 : ℝ)^x * 500^x = 1000000^3 ∧ x = 18 / 3.699 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4107_410749


namespace NUMINAMATH_CALUDE_parallel_tangents_condition_l4107_410701

-- Define the two curves
def curve1 (x : ℝ) : ℝ := x^2 - 1
def curve2 (x : ℝ) : ℝ := 1 - x^3

-- Define the derivatives of the curves
def curve1_derivative (x : ℝ) : ℝ := 2 * x
def curve2_derivative (x : ℝ) : ℝ := -3 * x^2

-- Theorem statement
theorem parallel_tangents_condition (x₀ : ℝ) :
  curve1_derivative x₀ = curve2_derivative x₀ ↔ x₀ = 0 ∨ x₀ = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_condition_l4107_410701


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l4107_410717

theorem vector_subtraction_and_scalar_multiplication :
  (⟨3, -8⟩ : ℝ × ℝ) - 3 • (⟨-2, 6⟩ : ℝ × ℝ) = (⟨9, -26⟩ : ℝ × ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l4107_410717


namespace NUMINAMATH_CALUDE_rhombus_with_60_degree_angles_l4107_410750

/-- A configuration of four points in the plane -/
structure QuadConfig where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ
  A₄ : ℝ × ℝ

/-- The sum of the smallest angles in the four triangles formed by the points -/
def sumSmallestAngles (q : QuadConfig) : ℝ := sorry

/-- Predicate to check if four points form a rhombus -/
def isRhombus (q : QuadConfig) : Prop := sorry

/-- Predicate to check if all angles in a quadrilateral are at least 60° -/
def allAnglesAtLeast60 (q : QuadConfig) : Prop := sorry

/-- The main theorem -/
theorem rhombus_with_60_degree_angles 
  (q : QuadConfig) 
  (h : sumSmallestAngles q = π) : 
  isRhombus q ∧ allAnglesAtLeast60 q := by
  sorry

end NUMINAMATH_CALUDE_rhombus_with_60_degree_angles_l4107_410750


namespace NUMINAMATH_CALUDE_excluded_numbers_sum_l4107_410739

theorem excluded_numbers_sum (numbers : Finset ℕ) (sum_all : ℕ) (sum_six : ℕ) :
  Finset.card numbers = 8 →
  sum_all = Finset.sum numbers id →
  sum_all / 8 = 34 →
  ∃ (excluded : Finset ℕ), Finset.card excluded = 2 ∧
    Finset.card (numbers \ excluded) = 6 ∧
    sum_six = Finset.sum (numbers \ excluded) id ∧
    sum_six / 6 = 29 →
  sum_all - sum_six = 98 :=
by sorry

end NUMINAMATH_CALUDE_excluded_numbers_sum_l4107_410739


namespace NUMINAMATH_CALUDE_correct_regression_l4107_410751

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if two variables are positively correlated -/
def positively_correlated (x y : ℝ → ℝ) : Prop := sorry

/-- Calculates the sample mean of a variable -/
def sample_mean (x : ℝ → ℝ) : ℝ := sorry

/-- Checks if a linear regression equation is valid for given data -/
def is_valid_regression (reg : LinearRegression) (x y : ℝ → ℝ) : Prop :=
  positively_correlated x y ∧
  sample_mean x = 3 ∧
  sample_mean y = 3.5 ∧
  reg.slope > 0 ∧
  reg.slope * (sample_mean x) + reg.intercept = sample_mean y

theorem correct_regression :
  is_valid_regression ⟨0.4, 2.3⟩ (λ _ => sorry) (λ _ => sorry) := by sorry

end NUMINAMATH_CALUDE_correct_regression_l4107_410751


namespace NUMINAMATH_CALUDE_equation_solution_l4107_410782

theorem equation_solution :
  let f (x : ℚ) := 1 - (3 + 2*x) / 4 = (x + 3) / 6
  ∃ (x : ℚ), f x ∧ x = -3/8 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4107_410782


namespace NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l4107_410727

theorem hemisphere_cylinder_surface_area
  (base_area : ℝ)
  (cylinder_height : ℝ)
  (h_base_area : base_area = 144 * Real.pi)
  (h_cylinder_height : cylinder_height = 5) :
  let radius := Real.sqrt (base_area / Real.pi)
  let hemisphere_area := 2 * Real.pi * radius ^ 2
  let cylinder_area := 2 * Real.pi * radius * cylinder_height
  hemisphere_area + cylinder_area = 408 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l4107_410727


namespace NUMINAMATH_CALUDE_min_draw_count_correct_l4107_410758

/-- Represents the number of balls of a specific color in the bag -/
structure ColorCount where
  red : Nat
  blue : Nat
  yellow : Nat
  other : Nat

/-- The minimum number of balls to draw to ensure at least 10 of one color -/
def minDrawCount : Nat := 38

/-- Theorem stating the minimum number of balls to draw -/
theorem min_draw_count_correct (total : Nat) (colors : ColorCount) 
  (h_total : total = 70)
  (h_red : colors.red = 20)
  (h_blue : colors.blue = 20)
  (h_yellow : colors.yellow = 20)
  (h_other : colors.other = total - colors.red - colors.blue - colors.yellow)
  : minDrawCount = 38 ∧ 
    ∀ n : Nat, n < minDrawCount → 
    ∃ draw : ColorCount, 
      draw.red < 10 ∧ 
      draw.blue < 10 ∧ 
      draw.yellow < 10 ∧ 
      draw.other < 10 ∧
      draw.red + draw.blue + draw.yellow + draw.other = n :=
by sorry

end NUMINAMATH_CALUDE_min_draw_count_correct_l4107_410758


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l4107_410774

theorem geometric_sequence_first_term 
  (a₅ a₆ : ℚ)
  (h₁ : a₅ = 48)
  (h₂ : a₆ = 64)
  : ∃ (a : ℚ), a₅ = a * (a₆ / a₅)^4 ∧ a = 243 / 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l4107_410774


namespace NUMINAMATH_CALUDE_exists_coverable_prism_l4107_410704

/-- Represents a regular triangular prism -/
structure RegularTriangularPrism where
  base_side : ℝ
  lateral_edge : ℝ
  lateral_edge_eq : lateral_edge = Real.sqrt 3 * base_side

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ

/-- Predicate to check if a prism can be covered by equilateral triangles -/
def can_cover_with_equilateral_triangles (p : RegularTriangularPrism) (t : EquilateralTriangle) : Prop :=
  p.base_side = t.side ∧
  p.lateral_edge = Real.sqrt 3 * t.side

/-- Theorem stating that there exists a regular triangular prism that can be covered by equilateral triangles -/
theorem exists_coverable_prism : 
  ∃ (p : RegularTriangularPrism) (t : EquilateralTriangle), 
    can_cover_with_equilateral_triangles p t := by
  sorry

end NUMINAMATH_CALUDE_exists_coverable_prism_l4107_410704


namespace NUMINAMATH_CALUDE_distribute_negative_two_l4107_410753

theorem distribute_negative_two (x : ℝ) : -2 * (x + 1) = -2 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_two_l4107_410753


namespace NUMINAMATH_CALUDE_max_product_sum_300_l4107_410757

theorem max_product_sum_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 ∧ ∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l4107_410757


namespace NUMINAMATH_CALUDE_max_area_at_45_degrees_l4107_410742

/-- A screen in a room corner --/
structure Screen where
  length : ℝ
  angle : ℝ

/-- Configuration of two screens in a room corner --/
structure CornerScreens where
  screen1 : Screen
  screen2 : Screen

/-- The area enclosed by two screens in a room corner --/
noncomputable def enclosedArea (cs : CornerScreens) : ℝ := sorry

/-- Theorem: The area enclosed by two equal-length screens in a right-angled corner
    is maximized when each screen forms a 45° angle with its adjacent wall --/
theorem max_area_at_45_degrees (l : ℝ) (h : l > 0) :
  ∃ (cs : CornerScreens),
    cs.screen1.length = l ∧
    cs.screen2.length = l ∧
    cs.screen1.angle = π/4 ∧
    cs.screen2.angle = π/4 ∧
    ∀ (other : CornerScreens),
      other.screen1.length = l →
      other.screen2.length = l →
      enclosedArea other ≤ enclosedArea cs :=
sorry

end NUMINAMATH_CALUDE_max_area_at_45_degrees_l4107_410742


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l4107_410714

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y ∧
   x^2 + 12*x + k = 0 ∧ 
   y^2 + 12*y + k = 0 ∧
   x / y = 3) → k = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l4107_410714


namespace NUMINAMATH_CALUDE_smallest_root_of_unity_order_l4107_410778

theorem smallest_root_of_unity_order (z : ℂ) : 
  (∃ (n : ℕ), n > 0 ∧ (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^n = 1) ∧ 
   (∀ m : ℕ, m > 0 → (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^m = 1) → m ≥ n)) → 
  (∃ (n : ℕ), n = 18 ∧ n > 0 ∧ (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^n = 1) ∧ 
   (∀ m : ℕ, m > 0 → (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^m = 1) → m ≥ n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_root_of_unity_order_l4107_410778


namespace NUMINAMATH_CALUDE_ellipse_condition_l4107_410797

/-- The equation represents an ellipse with foci on the x-axis -/
def is_ellipse_on_x_axis (n : ℝ) : Prop :=
  2 - n > 0 ∧ n + 1 > 0 ∧ 2 - n > n + 1

/-- The condition -1 < n < 2 is sufficient but not necessary for the equation to represent an ellipse with foci on the x-axis -/
theorem ellipse_condition (n : ℝ) :
  ((-1 < n ∧ n < 2) → is_ellipse_on_x_axis n) ∧
  ¬(is_ellipse_on_x_axis n → (-1 < n ∧ n < 2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l4107_410797


namespace NUMINAMATH_CALUDE_sum_of_mod2_and_mod3_l4107_410781

theorem sum_of_mod2_and_mod3 (a b : ℤ) : 
  a % 4 = 2 → b % 4 = 3 → (a + b) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_mod2_and_mod3_l4107_410781


namespace NUMINAMATH_CALUDE_emerie_quarters_l4107_410709

/-- Represents the number of coins of a specific type --/
structure CoinCount where
  dimes : Nat
  nickels : Nat
  quarters : Nat

/-- The total number of coins --/
def totalCoins (c : CoinCount) : Nat :=
  c.dimes + c.nickels + c.quarters

/-- Emerie's coin count --/
def emerie : CoinCount :=
  { dimes := 7, nickels := 5, quarters := 0 }

/-- Zain's coin count --/
def zain (e : CoinCount) : CoinCount :=
  { dimes := e.dimes + 10, nickels := e.nickels + 10, quarters := e.quarters + 10 }

theorem emerie_quarters : 
  totalCoins (zain emerie) = 48 → emerie.quarters = 6 := by
  sorry

end NUMINAMATH_CALUDE_emerie_quarters_l4107_410709


namespace NUMINAMATH_CALUDE_problem_solution_l4107_410702

theorem problem_solution (x y : ℝ) (h : |x - 3| + Real.sqrt (x - y + 1) = 0) :
  Real.sqrt (x^2 * y + x * y^2 + 1/4 * y^3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4107_410702


namespace NUMINAMATH_CALUDE_triangle_side_length_l4107_410748

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 4 → b = 2 → Real.cos A = 1/4 → c^2 = a^2 + b^2 - 2*a*b*(Real.cos A) → c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4107_410748


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4107_410710

theorem complex_equation_solution (a : ℂ) :
  (1 + a * Complex.I) / (2 + Complex.I) = 1 + 2 * Complex.I → a = 5 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4107_410710


namespace NUMINAMATH_CALUDE_eel_cost_l4107_410729

theorem eel_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : E = 180 := by
  sorry

end NUMINAMATH_CALUDE_eel_cost_l4107_410729


namespace NUMINAMATH_CALUDE_calculate_female_students_l4107_410780

/-- Given a population of students and a sample, calculate the number of female students in the population -/
theorem calculate_female_students 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (male_in_sample : ℕ) 
  (h1 : total_population = 2000) 
  (h2 : sample_size = 200) 
  (h3 : male_in_sample = 103) :
  (total_population - (male_in_sample * (total_population / sample_size))) = 970 := by
  sorry

#check calculate_female_students

end NUMINAMATH_CALUDE_calculate_female_students_l4107_410780


namespace NUMINAMATH_CALUDE_basic_computer_price_l4107_410767

theorem basic_computer_price 
  (total_price : ℝ) 
  (price_difference : ℝ) 
  (printer_ratio : ℝ) :
  total_price = 2500 →
  price_difference = 500 →
  printer_ratio = 1/6 →
  ∃ (basic_price printer_price : ℝ),
    basic_price + printer_price = total_price ∧
    printer_price = printer_ratio * (basic_price + price_difference + printer_price) ∧
    basic_price = 2000 :=
by sorry

end NUMINAMATH_CALUDE_basic_computer_price_l4107_410767


namespace NUMINAMATH_CALUDE_simplify_expression_l4107_410706

theorem simplify_expression : (12 ^ 0.6) * (12 ^ 0.4) * (8 ^ 0.2) * (8 ^ 0.8) = 96 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4107_410706


namespace NUMINAMATH_CALUDE_lakota_used_cd_count_l4107_410783

/-- The price of a new CD in dollars -/
def new_cd_price : ℝ := 17.99

/-- The price of a used CD in dollars -/
def used_cd_price : ℝ := 9.99

/-- The number of new CDs Lakota bought -/
def lakota_new_cds : ℕ := 6

/-- The total amount Lakota spent in dollars -/
def lakota_total : ℝ := 127.92

/-- The number of new CDs Mackenzie bought -/
def mackenzie_new_cds : ℕ := 3

/-- The number of used CDs Mackenzie bought -/
def mackenzie_used_cds : ℕ := 8

/-- The total amount Mackenzie spent in dollars -/
def mackenzie_total : ℝ := 133.89

/-- The number of used CDs Lakota bought -/
def lakota_used_cds : ℕ := 2

theorem lakota_used_cd_count : 
  lakota_new_cds * new_cd_price + lakota_used_cds * used_cd_price = lakota_total ∧
  mackenzie_new_cds * new_cd_price + mackenzie_used_cds * used_cd_price = mackenzie_total :=
by sorry

end NUMINAMATH_CALUDE_lakota_used_cd_count_l4107_410783


namespace NUMINAMATH_CALUDE_volume_of_specific_pyramid_l4107_410734

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ)
  (shorter_diagonal : ℝ)

/-- Represents a pyramid -/
structure Pyramid :=
  (base : Quadrilateral)
  (lateral_face_angle : ℝ)

/-- Calculate the volume of a pyramid -/
def pyramid_volume (p : Pyramid) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem volume_of_specific_pyramid :
  let base := Quadrilateral.mk 5 5 10 10 (4 * Real.sqrt 5)
  let pyr := Pyramid.mk base (π / 4)  -- 45° in radians
  pyramid_volume pyr = 500 / 9 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_pyramid_l4107_410734


namespace NUMINAMATH_CALUDE_combined_value_l4107_410716

def sum_even (a b : ℕ) : ℕ := 
  (b - a + 2) / 2 * (a + b) / 2

def sum_odd (a b : ℕ) : ℕ := 
  ((b - a) / 2 + 1) * (a + b) / 2

def i : ℕ := sum_even 2 500
def k : ℕ := sum_even 8 200
def j : ℕ := sum_odd 5 133

theorem combined_value : 2 * i - k + 3 * j = 128867 := by sorry

end NUMINAMATH_CALUDE_combined_value_l4107_410716


namespace NUMINAMATH_CALUDE_coin_distribution_l4107_410719

theorem coin_distribution (x y k : ℕ) (hxy : x + y = 81) (hne : x ≠ y) 
  (hsq : x^2 - y^2 = k * (x - y)) : k = 81 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_l4107_410719


namespace NUMINAMATH_CALUDE_seating_arrangements_l4107_410715

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def totalArrangements : ℕ := factorial 10

def restrictedArrangements : ℕ := factorial 7 * factorial 4

theorem seating_arrangements :
  totalArrangements - restrictedArrangements = 3507840 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l4107_410715


namespace NUMINAMATH_CALUDE_vector_sum_zero_l4107_410770

variable {V : Type*} [AddCommGroup V]

def vector (A B : V) : V := B - A

theorem vector_sum_zero (M B O A C D : V) : 
  (vector M B + vector B O + vector O M = 0) ∧
  (vector O B + vector O C + vector B O + vector C O = 0) ∧
  (vector A B - vector A C + vector B D - vector C D = 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_zero_l4107_410770


namespace NUMINAMATH_CALUDE_joan_gave_25_marbles_l4107_410745

/-- The number of yellow marbles Joan gave Sam -/
def marbles_from_joan (initial_yellow : ℝ) (final_yellow : ℕ) : ℝ :=
  final_yellow - initial_yellow

theorem joan_gave_25_marbles :
  let initial_yellow : ℝ := 86.0
  let final_yellow : ℕ := 111
  marbles_from_joan initial_yellow final_yellow = 25 := by
  sorry

end NUMINAMATH_CALUDE_joan_gave_25_marbles_l4107_410745


namespace NUMINAMATH_CALUDE_inequality_solution_equation_solution_l4107_410752

-- Part 1: System of inequalities
def inequality_system (x : ℝ) : Prop :=
  x + 2 > 1 ∧ 2*x < x + 3

theorem inequality_solution :
  ∀ x : ℝ, inequality_system x ↔ -1 < x ∧ x < 3 :=
sorry

-- Part 2: System of linear equations
def equation_system (x y : ℝ) : Prop :=
  3*x + 2*y = 12 ∧ 2*x - y = 1

theorem equation_solution :
  ∀ x y : ℝ, equation_system x y ↔ x = 2 ∧ y = 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_equation_solution_l4107_410752


namespace NUMINAMATH_CALUDE_sum_of_roots_l4107_410763

theorem sum_of_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, (3*r₁ + 4)*(r₁ - 5) + (3*r₁ + 4)*(r₁ - 7) = 0 ∧ 
                (3*r₂ + 4)*(r₂ - 5) + (3*r₂ + 4)*(r₂ - 7) = 0 ∧ 
                r₁ ≠ r₂) → 
  (∃ r₁ r₂ : ℝ, (3*r₁ + 4)*(r₁ - 5) + (3*r₁ + 4)*(r₁ - 7) = 0 ∧ 
                (3*r₂ + 4)*(r₂ - 5) + (3*r₂ + 4)*(r₂ - 7) = 0 ∧ 
                r₁ + r₂ = 14/3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l4107_410763


namespace NUMINAMATH_CALUDE_cos_inequality_l4107_410707

theorem cos_inequality : 
  2 * π > 15 * π / 8 ∧ 
  15 * π / 8 > 14 * π / 9 ∧ 
  14 * π / 9 > 3 * π / 2 → 
  Real.cos (-15 * π / 8) > Real.cos (14 * π / 9) := by
sorry

end NUMINAMATH_CALUDE_cos_inequality_l4107_410707


namespace NUMINAMATH_CALUDE_square_difference_601_597_l4107_410786

theorem square_difference_601_597 : (601 : ℤ)^2 - (597 : ℤ)^2 = 4792 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_601_597_l4107_410786


namespace NUMINAMATH_CALUDE_marble_exchange_problem_l4107_410730

/-- Represents the marble exchange problem with Woong, Youngsoo, and Hyogeun --/
theorem marble_exchange_problem (W Y H : ℕ) : 
  (W + 2 = 20) →  -- Woong's final marbles
  (Y - 5 = 20) →  -- Youngsoo's final marbles
  (H + 3 = 20) →  -- Hyogeun's final marbles
  W = 18 :=        -- Woong's initial marbles
by sorry

end NUMINAMATH_CALUDE_marble_exchange_problem_l4107_410730


namespace NUMINAMATH_CALUDE_flag_puzzle_l4107_410756

theorem flag_puzzle (x : ℝ) : 
  (8 * 5 : ℝ) + (10 * 7 : ℝ) + (x * 5 : ℝ) = (15 * 9 : ℝ) → x = 5 := by
sorry

end NUMINAMATH_CALUDE_flag_puzzle_l4107_410756


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_l4107_410743

def smallest_one_digit_primes : List Nat := [2, 3]
def smallest_two_digit_prime : Nat := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 := by
  sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_l4107_410743


namespace NUMINAMATH_CALUDE_final_S_value_l4107_410766

def i (n : ℕ) : ℕ := 2 * n + 1

def S (n : ℕ) : ℕ := 2 * i n + 3

theorem final_S_value :
  ∃ n : ℕ, i n ≥ 8 ∧ i (n - 1) < 8 ∧ S (n - 1) = 21 :=
sorry

end NUMINAMATH_CALUDE_final_S_value_l4107_410766


namespace NUMINAMATH_CALUDE_fraction_meaningful_l4107_410759

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l4107_410759


namespace NUMINAMATH_CALUDE_min_distance_curve_to_line_l4107_410796

theorem min_distance_curve_to_line :
  let C₁ := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1 ∧ p.2 ≠ 0}
  let C₂ := {p : ℝ × ℝ | p.1 + p.2 - 8 = 0}
  (∀ p ∈ C₁, ∃ q ∈ C₂, ∀ r ∈ C₂, dist p q ≤ dist p r) →
  (∃ p ∈ C₁, ∃ q ∈ C₂, dist p q = 3 * Real.sqrt 2) →
  ∀ p ∈ C₁, ∀ q ∈ C₂, dist p q ≥ 3 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_min_distance_curve_to_line_l4107_410796


namespace NUMINAMATH_CALUDE_abby_singles_percentage_l4107_410738

/-- Represents the statistics of a softball player's hits -/
structure HitStatistics where
  total_hits : ℕ
  home_runs : ℕ
  triples : ℕ
  doubles : ℕ

/-- Calculates the percentage of singles given hit statistics -/
def percentage_singles (stats : HitStatistics) : ℚ :=
  let singles := stats.total_hits - (stats.home_runs + stats.triples + stats.doubles)
  (singles : ℚ) / (stats.total_hits : ℚ) * 100

/-- Abby's hit statistics -/
def abby_stats : HitStatistics :=
  { total_hits := 45
  , home_runs := 2
  , triples := 3
  , doubles := 7
  }

/-- Theorem stating that the percentage of Abby's singles is 73.33% -/
theorem abby_singles_percentage :
  percentage_singles abby_stats = 73 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_abby_singles_percentage_l4107_410738


namespace NUMINAMATH_CALUDE_floor_factorial_ratio_l4107_410737

open BigOperators

def factorial (n : ℕ) : ℕ := ∏ i in Finset.range n, i + 1

theorem floor_factorial_ratio : 
  ⌊(factorial 2007 + factorial 2004 : ℚ) / (factorial 2006 + factorial 2005)⌋ = 2006 := by
  sorry

end NUMINAMATH_CALUDE_floor_factorial_ratio_l4107_410737


namespace NUMINAMATH_CALUDE_scientific_notation_of_320000_l4107_410740

theorem scientific_notation_of_320000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 320000 = a * (10 : ℝ) ^ n ∧ a = 3.2 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_320000_l4107_410740


namespace NUMINAMATH_CALUDE_certain_number_proof_l4107_410735

theorem certain_number_proof (x y : ℤ) 
  (eq1 : 4 * x + y = 34) 
  (eq2 : y^2 = 4) : 
  2 * x - y = 14 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l4107_410735


namespace NUMINAMATH_CALUDE_sum_of_xyz_l4107_410777

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) : 
  x + y + z = 11 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l4107_410777


namespace NUMINAMATH_CALUDE_no_solution_sqrt_equation_l4107_410787

theorem no_solution_sqrt_equation :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → Real.sqrt (x + 1) + Real.sqrt (3 - x) < 17 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_sqrt_equation_l4107_410787


namespace NUMINAMATH_CALUDE_axis_of_symmetry_implies_r_equals_s_l4107_410789

/-- Represents a rational function of the form (px + q) / (rx + s) -/
structure RationalFunction (α : Type) [Field α] where
  p : α
  q : α
  r : α
  s : α
  p_nonzero : p ≠ 0
  q_nonzero : q ≠ 0
  r_nonzero : r ≠ 0
  s_nonzero : s ≠ 0

/-- Defines the property of y = -x being an axis of symmetry for a given rational function -/
def isAxisOfSymmetry {α : Type} [Field α] (f : RationalFunction α) : Prop :=
  ∀ (x y : α), y = (f.p * x + f.q) / (f.r * x + f.s) → (-x) = (f.p * (-y) + f.q) / (f.r * (-y) + f.s)

/-- Theorem stating that if y = -x is an axis of symmetry for the rational function,
    then r - s = 0 -/
theorem axis_of_symmetry_implies_r_equals_s {α : Type} [Field α] (f : RationalFunction α) :
  isAxisOfSymmetry f → f.r = f.s :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_implies_r_equals_s_l4107_410789


namespace NUMINAMATH_CALUDE_werewolf_victims_l4107_410772

/-- Given a village with a certain population, a vampire's weekly victim count, 
    and a time period, calculate the werewolf's weekly victim count. -/
theorem werewolf_victims (village_population : ℕ) (vampire_victims_per_week : ℕ) (weeks : ℕ) 
  (h1 : village_population = 72)
  (h2 : vampire_victims_per_week = 3)
  (h3 : weeks = 9) :
  ∃ (werewolf_victims_per_week : ℕ), 
    werewolf_victims_per_week * weeks + vampire_victims_per_week * weeks = village_population ∧ 
    werewolf_victims_per_week = 5 :=
by sorry

end NUMINAMATH_CALUDE_werewolf_victims_l4107_410772


namespace NUMINAMATH_CALUDE_inequality_proof_l4107_410773

theorem inequality_proof (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  1 / Real.sqrt (1 + x^2) + 1 / Real.sqrt (1 + y^2) ≤ 2 / Real.sqrt (1 + x*y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4107_410773


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l4107_410732

theorem polynomial_coefficient_sum (a₄ a₃ a₂ a₁ a₀ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ - a₃ + a₂ - a₁ = 15 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l4107_410732


namespace NUMINAMATH_CALUDE_inverse_function_point_correspondence_l4107_410762

theorem inverse_function_point_correspondence
  (f : ℝ → ℝ) (h_inv : Function.Bijective f) :
  (Function.invFun f) 1 = 2 → f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_point_correspondence_l4107_410762


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4107_410761

theorem quadratic_inequality_solution_set (a : ℝ) : 
  (∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) → 
  (∀ x, ax^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4107_410761


namespace NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l4107_410723

theorem rectangular_solid_edge_sum (a r : ℝ) : 
  a > 0 ∧ r > 0 →
  (a / r) * a * (a * r) = 512 →
  2 * ((a^2 / r) + (a^2 * r) + a^2) = 320 →
  4 * (a / r + a + a * r) = 56 + 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l4107_410723


namespace NUMINAMATH_CALUDE_book_arrangement_count_l4107_410768

/-- The number of ways to arrange math and history books on a shelf -/
def arrange_books (num_math_books num_history_books : ℕ) : ℕ :=
  let end_arrangements := num_math_books * (num_math_books - 1)
  let remaining_math_arrangements := 2  -- factorial of (num_math_books - 2)
  let history_distributions := (Nat.choose num_history_books 2) * 
                               (Nat.choose (num_history_books - 2) 2) *
                               2  -- Last 2 is automatic
  let history_permutations := (2 * 2 * 2)  -- 2! for each of the 3 slots
  end_arrangements * remaining_math_arrangements * history_distributions * history_permutations

/-- Theorem stating the number of ways to arrange the books -/
theorem book_arrangement_count :
  arrange_books 4 6 = 17280 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l4107_410768


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4107_410769

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 3 → ∃ x : ℝ, x^2 + a*x + 1 < 0) ∧
  (∃ a, (∃ x : ℝ, x^2 + a*x + 1 < 0) ∧ ¬(a > 3)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4107_410769


namespace NUMINAMATH_CALUDE_complex_division_1_complex_division_2_l4107_410791

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem for the first calculation
theorem complex_division_1 : (1 - i) * (1 + 2*i) / (1 + i) = 2 - i := by sorry

-- Theorem for the second calculation
theorem complex_division_2 : ((1 + 2*i)^2 + 3*(1 - i)) / (2 + i) = 3 - 6/5 * i := by sorry

end NUMINAMATH_CALUDE_complex_division_1_complex_division_2_l4107_410791


namespace NUMINAMATH_CALUDE_a_can_be_any_real_l4107_410771

theorem a_can_be_any_real : ∀ (a b c d e : ℝ), 
  bd ≠ 0 → e ≠ 0 → (a / b + e < -(c / d)) → 
  (∃ (a_pos a_neg a_zero : ℝ), 
    (a_pos > 0 ∧ a_pos / b + e < -(c / d)) ∧
    (a_neg < 0 ∧ a_neg / b + e < -(c / d)) ∧
    (a_zero = 0 ∧ a_zero / b + e < -(c / d))) :=
by sorry

end NUMINAMATH_CALUDE_a_can_be_any_real_l4107_410771


namespace NUMINAMATH_CALUDE_complex_equation_sum_l4107_410726

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a + 2 * i) / i = b + i) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l4107_410726


namespace NUMINAMATH_CALUDE_adult_ticket_price_l4107_410798

/-- Proves that the price of an adult ticket is $32 given the specified conditions -/
theorem adult_ticket_price
  (num_adults : ℕ)
  (num_children : ℕ)
  (total_amount : ℕ)
  (h_adults : num_adults = 400)
  (h_children : num_children = 200)
  (h_total : total_amount = 16000)
  (h_price_ratio : ∃ (child_price : ℕ), 
    total_amount = num_adults * (2 * child_price) + num_children * child_price) :
  ∃ (adult_price : ℕ), adult_price = 32 ∧
    total_amount = num_adults * adult_price + num_children * (adult_price / 2) :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l4107_410798


namespace NUMINAMATH_CALUDE_child_share_calculation_l4107_410776

theorem child_share_calculation (total_amount : ℚ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_amount = 4500 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 4 →
  (ratio_b : ℚ) / (ratio_a + ratio_b + ratio_c : ℚ) * total_amount = 1500 := by
sorry

end NUMINAMATH_CALUDE_child_share_calculation_l4107_410776


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l4107_410720

/-- The total surface area of a cube with holes -/
def total_surface_area (cube_edge : ℝ) (hole_side : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_side^2
  let new_exposed_area := 6 * 4 * hole_side^2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem: The total surface area of a cube with edge length 3 and square holes of side 1 is 72 -/
theorem cube_with_holes_surface_area :
  total_surface_area 3 1 = 72 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l4107_410720


namespace NUMINAMATH_CALUDE_savings_calculation_l4107_410792

theorem savings_calculation (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : 
  income = 19000 → 
  income_ratio = 5 → 
  expenditure_ratio = 4 → 
  income - (income * expenditure_ratio / income_ratio) = 3800 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l4107_410792


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l4107_410722

-- Define the function f(x) = x^3 + 4x - 3
def f (x : ℝ) := x^3 + 4*x - 3

-- State the theorem
theorem f_has_root_in_interval :
  ∃ c ∈ Set.Icc 0 1, f c = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_f_has_root_in_interval_l4107_410722


namespace NUMINAMATH_CALUDE_fraction_comparison_l4107_410760

theorem fraction_comparison (x : ℝ) : 
  x > 3/4 → x ≠ 3 → (9 - 3*x ≠ 0) → (5*x + 3 > 9 - 3*x) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l4107_410760


namespace NUMINAMATH_CALUDE_base7_divisible_by_13_l4107_410713

/-- Converts a base-7 number of the form 3dd6₇ to base 10 --/
def base7ToBase10 (d : Nat) : Nat :=
  3 * 7^3 + d * 7^2 + d * 7 + 6

/-- Checks if a number is divisible by 13 --/
def isDivisibleBy13 (n : Nat) : Prop :=
  n % 13 = 0

/-- A base-7 digit is between 0 and 6 inclusive --/
def isBase7Digit (d : Nat) : Prop :=
  d ≤ 6

theorem base7_divisible_by_13 :
  ∃ (d : Nat), isBase7Digit d ∧ isDivisibleBy13 (base7ToBase10 d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_base7_divisible_by_13_l4107_410713


namespace NUMINAMATH_CALUDE_cube_equation_solution_l4107_410779

theorem cube_equation_solution :
  ∃! x : ℝ, (8 - x)^3 = x^3 ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l4107_410779


namespace NUMINAMATH_CALUDE_circle_square_intersection_probability_l4107_410711

/-- The probability that a circle of radius 1 centered at a random point
    inside a square of side length 4 intersects the square exactly twice. -/
theorem circle_square_intersection_probability :
  let square_side : ℝ := 4
  let circle_radius : ℝ := 1
  let favorable_area : ℝ := π + 8
  let total_area : ℝ := square_side ^ 2
  (favorable_area / total_area : ℝ) = (π + 8) / 16 := by
sorry

end NUMINAMATH_CALUDE_circle_square_intersection_probability_l4107_410711


namespace NUMINAMATH_CALUDE_cubic_polynomial_theorem_l4107_410725

/-- Given a, b, c are roots of x³ + 4x² + 6x + 8 = 0 -/
def cubic_roots (a b c : ℝ) : Prop :=
  a^3 + 4*a^2 + 6*a + 8 = 0 ∧
  b^3 + 4*b^2 + 6*b + 8 = 0 ∧
  c^3 + 4*c^2 + 6*c + 8 = 0

/-- Q is a cubic polynomial satisfying the given conditions -/
def Q_conditions (Q : ℝ → ℝ) (a b c : ℝ) : Prop :=
  (∃ p q r s : ℝ, ∀ x, Q x = p*x^3 + q*x^2 + r*x + s) ∧
  Q a = b + c ∧
  Q b = a + c ∧
  Q c = a + b ∧
  Q (a + b + c) = -20

theorem cubic_polynomial_theorem (a b c : ℝ) (Q : ℝ → ℝ) 
  (h1 : cubic_roots a b c) (h2 : Q_conditions Q a b c) :
  ∀ x, Q x = 5/4*x^3 + 4*x^2 + 17/4*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_theorem_l4107_410725


namespace NUMINAMATH_CALUDE_total_rooms_in_hotel_l4107_410708

/-- Represents a wing of the hotel -/
structure Wing where
  floors : ℕ
  hallsPerFloor : ℕ
  singleRoomsPerHall : ℕ
  doubleRoomsPerHall : ℕ
  suitesPerHall : ℕ

/-- Calculates the total number of rooms in a wing -/
def totalRoomsInWing (w : Wing) : ℕ :=
  w.floors * w.hallsPerFloor * (w.singleRoomsPerHall + w.doubleRoomsPerHall + w.suitesPerHall)

/-- The first wing of the hotel -/
def wing1 : Wing :=
  { floors := 9
    hallsPerFloor := 6
    singleRoomsPerHall := 20
    doubleRoomsPerHall := 8
    suitesPerHall := 4 }

/-- The second wing of the hotel -/
def wing2 : Wing :=
  { floors := 7
    hallsPerFloor := 9
    singleRoomsPerHall := 25
    doubleRoomsPerHall := 10
    suitesPerHall := 5 }

/-- The third wing of the hotel -/
def wing3 : Wing :=
  { floors := 12
    hallsPerFloor := 4
    singleRoomsPerHall := 30
    doubleRoomsPerHall := 15
    suitesPerHall := 5 }

/-- Theorem stating the total number of rooms in the hotel -/
theorem total_rooms_in_hotel :
  totalRoomsInWing wing1 + totalRoomsInWing wing2 + totalRoomsInWing wing3 = 6648 := by
  sorry

end NUMINAMATH_CALUDE_total_rooms_in_hotel_l4107_410708


namespace NUMINAMATH_CALUDE_rod_triangle_theorem_l4107_410721

/-- A triple of natural numbers representing the side lengths of a triangle --/
structure TriangleSides where
  a : ℕ
  b : ℕ
  c : ℕ
  a_le_b : a ≤ b
  b_le_c : b ≤ c

/-- Checks if a natural number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- Checks if a TriangleSides forms an isosceles triangle --/
def isIsosceles (t : TriangleSides) : Prop :=
  t.a = t.b ∨ t.b = t.c

/-- The main theorem --/
theorem rod_triangle_theorem :
  ∃! (sol : Finset TriangleSides),
    (∀ t ∈ sol, 
      t.a + t.b + t.c = 25 ∧ 
      isPrime t.a ∧ isPrime t.b ∧ isPrime t.c) ∧
    sol.card = 2 ∧
    (∀ t ∈ sol, isIsosceles t) := by sorry

end NUMINAMATH_CALUDE_rod_triangle_theorem_l4107_410721


namespace NUMINAMATH_CALUDE_winnie_lollipop_distribution_l4107_410799

/-- Winnie's lollipop distribution problem -/
theorem winnie_lollipop_distribution 
  (total_lollipops : ℕ) 
  (num_friends : ℕ) 
  (h1 : total_lollipops = 72 + 89 + 23 + 316) 
  (h2 : num_friends = 14) : 
  total_lollipops % num_friends = 10 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipop_distribution_l4107_410799


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l4107_410793

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct arrangements of the letters in "BANANA" is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l4107_410793
