import Mathlib

namespace NUMINAMATH_CALUDE_maria_water_bottles_l2606_260674

/-- Calculates the final number of water bottles Maria has after a series of actions. -/
def final_bottle_count (initial : ℕ) (drunk : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - drunk - given_away + bought

/-- Theorem stating that Maria ends up with 71 bottles given the initial conditions and actions. -/
theorem maria_water_bottles : final_bottle_count 23 12 5 65 = 71 := by
  sorry

end NUMINAMATH_CALUDE_maria_water_bottles_l2606_260674


namespace NUMINAMATH_CALUDE_polynomial_factor_l2606_260640

theorem polynomial_factor (a b c : ℤ) (x : ℚ) : 
  a = 1 ∧ b = -1 ∧ c = -8 →
  ∃ k : ℚ, (x^2 - 2*x - 1) * (2*a*x + k) = 2*a*x^3 + b*x^2 + c*x - 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factor_l2606_260640


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l2606_260690

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line with slope m and y-intercept -3
def line (m x : ℝ) : ℝ := m * x - 3

-- Define the set of valid slopes
def valid_slopes : Set ℝ := {m : ℝ | m ≤ -Real.sqrt (4/55) ∨ m ≥ Real.sqrt (4/55)}

-- Theorem statement
theorem line_ellipse_intersection_slopes :
  ∀ m : ℝ, (∃ x : ℝ, ellipse x (line m x)) ↔ m ∈ valid_slopes :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l2606_260690


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l2606_260642

/-- Represents the systematic sampling problem -/
def SystematicSampling (total_population : ℕ) (sample_size : ℕ) (first_number : ℕ) (threshold : ℕ) : Prop :=
  let interval := total_population / sample_size
  let selected_numbers := List.range sample_size |>.map (fun i => first_number + i * interval)
  (selected_numbers.filter (fun n => n > threshold)).length = 8

/-- Theorem stating the result of the systematic sampling problem -/
theorem systematic_sampling_result : 
  SystematicSampling 960 32 9 750 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l2606_260642


namespace NUMINAMATH_CALUDE_unique_integer_congruence_l2606_260625

theorem unique_integer_congruence :
  ∃! n : ℤ, 6 ≤ n ∧ n ≤ 12 ∧ n ≡ 10403 [ZMOD 7] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_congruence_l2606_260625


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l2606_260648

theorem divisibility_by_twelve (n : Nat) : n ≤ 9 → (512 * 10 + n) % 12 = 0 ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l2606_260648


namespace NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l2606_260623

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 143 → ∀ a b : ℕ, a^2 - b^2 = 143 → x^2 + y^2 ≤ a^2 + b^2 → x^2 + y^2 = 145 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l2606_260623


namespace NUMINAMATH_CALUDE_unique_prime_polynomial_l2606_260633

theorem unique_prime_polynomial : ∃! (n : ℕ), n > 0 ∧ Nat.Prime (n^3 - 9*n^2 + 27*n - 28) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_polynomial_l2606_260633


namespace NUMINAMATH_CALUDE_sum_of_radii_tangent_circles_l2606_260691

/-- The sum of radii of a circle tangent to x and y axes and externally tangent to another circle -/
theorem sum_of_radii_tangent_circles : ∀ r : ℝ,
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  ∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ (r₁ + r₂ = 14) := by
sorry

end NUMINAMATH_CALUDE_sum_of_radii_tangent_circles_l2606_260691


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2606_260693

/-- The sum of the infinite series ∑(n=1 to ∞) 3^n / (1 + 3^n + 3^(n+1) + 3^(2n+1)) equals 1/4. -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (3:ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))) = (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2606_260693


namespace NUMINAMATH_CALUDE_percentage_calculation_l2606_260626

theorem percentage_calculation (x : ℝ) (p : ℝ) (h1 : x = 60) (h2 : x = (p / 100) * x + 52.8) : p = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2606_260626


namespace NUMINAMATH_CALUDE_fish_length_difference_l2606_260680

theorem fish_length_difference :
  let fish1_length : ℝ := 0.3
  let fish2_length : ℝ := 0.2
  fish1_length - fish2_length = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_fish_length_difference_l2606_260680


namespace NUMINAMATH_CALUDE_angle_B_value_max_sum_sides_l2606_260646

-- Define the triangle
variable (A B C a b c : ℝ)

-- Define the conditions
variable (triangle_abc : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variable (side_angle_relation : b * Real.cos C = (2 * a - c) * Real.cos B)

-- First theorem: B = π/3
theorem angle_B_value : B = π / 3 := by sorry

-- Second theorem: Maximum value of a + c when b = √3
theorem max_sum_sides (h_b : b = Real.sqrt 3) :
  ∃ (max : ℝ), max = 2 * Real.sqrt 3 ∧ 
  ∀ (a c : ℝ), a + c ≤ max := by sorry

end NUMINAMATH_CALUDE_angle_B_value_max_sum_sides_l2606_260646


namespace NUMINAMATH_CALUDE_linear_function_composition_l2606_260660

/-- A linear function from ℝ to ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x + 9) →
  (∀ x, f x = 2 * x + 3) ∨ (∀ x, f x = -2 * x - 9) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_composition_l2606_260660


namespace NUMINAMATH_CALUDE_cosine_rational_values_l2606_260684

theorem cosine_rational_values (α : ℚ) (h : ∃ (q : ℚ), q = Real.cos (α * Real.pi)) :
  Real.cos (α * Real.pi) = 0 ∨ 
  Real.cos (α * Real.pi) = (1/2 : ℝ) ∨ 
  Real.cos (α * Real.pi) = -(1/2 : ℝ) ∨ 
  Real.cos (α * Real.pi) = 1 ∨ 
  Real.cos (α * Real.pi) = -1 := by
sorry

end NUMINAMATH_CALUDE_cosine_rational_values_l2606_260684


namespace NUMINAMATH_CALUDE_p_sequence_constant_difference_l2606_260606

/-- A P-sequence is a geometric sequence {a_n} where (a_1 + 1, a_2 + 2, a_3 + 3) also forms a geometric sequence -/
def is_p_sequence (a : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∀ n, a_n (n + 1) = a_n n * a_n 1) ∧
  (∃ r, (a_n 1 + 1) * r = a_n 2 + 2 ∧ (a_n 2 + 2) * r = a_n 3 + 3)

theorem p_sequence_constant_difference (a : ℝ) (h1 : 1/2 < a) (h2 : a < 1) :
  let a_n : ℕ → ℝ := λ n => a^(2*n - 1)
  let x_n : ℕ → ℝ := λ n => a_n n - 1 / (a_n n)
  is_p_sequence a a_n →
  ∀ n ≥ 2, x_n n^2 - x_n (n-1) * x_n (n+1) = 5 := by
sorry

end NUMINAMATH_CALUDE_p_sequence_constant_difference_l2606_260606


namespace NUMINAMATH_CALUDE_triplet_convergence_l2606_260687

/-- Given a triplet of numbers, compute the absolute differences -/
def absDiff (t : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a, b, c) := t
  (|a - b|, |b - c|, |c - a|)

/-- Generate the sequence of triplets -/
def tripletSeq (x y z : ℝ) : ℕ → ℝ × ℝ × ℝ
  | 0 => (x, y, z)
  | n + 1 => absDiff (tripletSeq x y z n)

theorem triplet_convergence (y z : ℝ) :
  (∃ n : ℕ, tripletSeq 1 y z n = (1, y, z)) → y = 1 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_triplet_convergence_l2606_260687


namespace NUMINAMATH_CALUDE_average_percent_change_population_l2606_260682

-- Define the initial and final population
def initial_population : ℕ := 175000
def final_population : ℕ := 297500

-- Define the time period in years
def years : ℕ := 10

-- Define the theorem
theorem average_percent_change_population (initial_pop : ℕ) (final_pop : ℕ) (time : ℕ) :
  initial_pop = initial_population →
  final_pop = final_population →
  time = years →
  (((final_pop - initial_pop : ℝ) / initial_pop) * 100) / time = 7 :=
by sorry

end NUMINAMATH_CALUDE_average_percent_change_population_l2606_260682


namespace NUMINAMATH_CALUDE_perimeter_of_AMN_l2606_260628

-- Define the triangle ABC
structure Triangle :=
  (AB BC CA : ℝ)

-- Define the properties of triangle AMN
structure TriangleAMN (ABC : Triangle) :=
  (M : ℝ) -- Distance BM
  (N : ℝ) -- Distance CN
  (parallel_to_BC : True) -- MN is parallel to BC

-- Theorem statement
theorem perimeter_of_AMN (ABC : Triangle) (AMN : TriangleAMN ABC) :
  ABC.AB = 26 ∧ ABC.BC = 17 ∧ ABC.CA = 19 →
  (ABC.AB - AMN.M) + (ABC.CA - AMN.N) + 
    ((AMN.M / ABC.AB) * ABC.BC) = 45 :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_AMN_l2606_260628


namespace NUMINAMATH_CALUDE_teacup_lid_arrangement_l2606_260661

def teacups : ℕ := 6
def lids : ℕ := 6
def matching_lids : ℕ := 2

theorem teacup_lid_arrangement :
  (teacups.choose matching_lids) * 
  ((teacups - matching_lids - 1) * (lids - matching_lids - 1)) = 135 := by
sorry

end NUMINAMATH_CALUDE_teacup_lid_arrangement_l2606_260661


namespace NUMINAMATH_CALUDE_F_fraction_difference_l2606_260611

def F : ℚ := 925 / 999

theorem F_fraction_difference : ∃ (a b : ℕ), 
  F = a / b ∧ 
  (∀ (c d : ℕ), F = c / d → b ≤ d) ∧
  b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_F_fraction_difference_l2606_260611


namespace NUMINAMATH_CALUDE_l_shape_area_l2606_260694

/-- The area of an "L" shape formed by removing a smaller rectangle from a larger rectangle -/
theorem l_shape_area (large_width large_height small_width small_height : ℕ) :
  large_width = 10 →
  large_height = 7 →
  small_width = 4 →
  small_height = 3 →
  (large_width * large_height) - (small_width * small_height) = 58 := by
  sorry

#check l_shape_area

end NUMINAMATH_CALUDE_l_shape_area_l2606_260694


namespace NUMINAMATH_CALUDE_eggs_sold_count_l2606_260692

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 30

/-- The initial number of trays to be collected -/
def initial_trays : ℕ := 10

/-- The number of trays dropped -/
def dropped_trays : ℕ := 2

/-- The number of additional trays added after the accident -/
def additional_trays : ℕ := 7

/-- Theorem stating the total number of eggs sold -/
theorem eggs_sold_count : 
  (initial_trays - dropped_trays + additional_trays) * eggs_per_tray = 450 := by
sorry

end NUMINAMATH_CALUDE_eggs_sold_count_l2606_260692


namespace NUMINAMATH_CALUDE_legitimate_paths_count_l2606_260609

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Defines the grid dimensions -/
def gridWidth : Nat := 12
def gridHeight : Nat := 4

/-- Defines the start and end points -/
def pointA : GridPoint := ⟨0, 0⟩
def pointB : GridPoint := ⟨gridWidth - 1, gridHeight - 1⟩

/-- Checks if a path is legitimate based on the column restrictions -/
def isLegitimate (path : List GridPoint) : Bool :=
  path.all fun p =>
    (p.x ≠ 2 || p.y = 0 || p.y = 1 || p.y = gridHeight - 1) &&
    (p.x ≠ 4 || p.y = 0 || p.y = gridHeight - 2 || p.y = gridHeight - 1)

/-- Counts the number of legitimate paths from A to B -/
def countLegitimatePaths : Nat :=
  sorry -- The actual implementation would go here

/-- The main theorem to prove -/
theorem legitimate_paths_count :
  countLegitimatePaths = 1289 := by
  sorry

end NUMINAMATH_CALUDE_legitimate_paths_count_l2606_260609


namespace NUMINAMATH_CALUDE_complex_point_location_l2606_260619

theorem complex_point_location (z : ℂ) (h : z = Complex.I * 2) : 
  z.re = 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_point_location_l2606_260619


namespace NUMINAMATH_CALUDE_sum_squares_range_l2606_260676

/-- Given a positive constant k and a sequence of positive real numbers x_i whose sum equals k,
    the sum of x_i^2 can take any value in the open interval (0, k^2). -/
theorem sum_squares_range (k : ℝ) (x : ℕ → ℝ) (h_k_pos : k > 0) (h_x_pos : ∀ n, x n > 0)
    (h_sum_x : ∑' n, x n = k) :
  ∀ y, 0 < y ∧ y < k^2 → ∃ x : ℕ → ℝ,
    (∀ n, x n > 0) ∧ (∑' n, x n = k) ∧ (∑' n, (x n)^2 = y) :=
by sorry

end NUMINAMATH_CALUDE_sum_squares_range_l2606_260676


namespace NUMINAMATH_CALUDE_project_hours_l2606_260675

theorem project_hours (kate mark pat : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : pat * 3 = mark)
  (h3 : mark = kate + 80) :
  kate + mark + pat = 144 := by
sorry

end NUMINAMATH_CALUDE_project_hours_l2606_260675


namespace NUMINAMATH_CALUDE_floor_length_is_twelve_l2606_260656

/-- Represents a rectangular floor with a rug -/
structure FloorWithRug where
  length : ℝ
  width : ℝ
  strip_width : ℝ
  rug_area : ℝ

/-- Theorem: Given the conditions, the floor length is 12 meters -/
theorem floor_length_is_twelve (floor : FloorWithRug) 
  (h1 : floor.width = 10)
  (h2 : floor.strip_width = 3)
  (h3 : floor.rug_area = 24)
  (h4 : floor.rug_area = (floor.length - 2 * floor.strip_width) * (floor.width - 2 * floor.strip_width)) :
  floor.length = 12 := by
  sorry

end NUMINAMATH_CALUDE_floor_length_is_twelve_l2606_260656


namespace NUMINAMATH_CALUDE_g_value_at_four_l2606_260685

/-- The cubic polynomial f(x) = x^3 - 2x + 5 -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 5

/-- g is a cubic polynomial such that g(0) = 1 and its roots are the squares of the roots of f -/
def g : ℝ → ℝ :=
  sorry

theorem g_value_at_four :
  g 4 = -9/25 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_four_l2606_260685


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l2606_260653

theorem ratio_a_to_c (a b c d : ℝ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 8) :
  a / c = 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l2606_260653


namespace NUMINAMATH_CALUDE_rectangle_vertical_length_l2606_260655

/-- Given a rectangle with perimeter 50 cm and horizontal length 13 cm, prove its vertical length is 12 cm -/
theorem rectangle_vertical_length (perimeter : ℝ) (horizontal_length : ℝ) (vertical_length : ℝ) : 
  perimeter = 50 ∧ horizontal_length = 13 → 
  perimeter = 2 * (horizontal_length + vertical_length) →
  vertical_length = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_vertical_length_l2606_260655


namespace NUMINAMATH_CALUDE_vertex_segments_different_colors_l2606_260671

structure ColoredTriangle where
  n : ℕ  -- number of marked points inside the triangle
  k₁ : ℕ  -- number of segments of first color connected to vertices
  k₂ : ℕ  -- number of segments of second color connected to vertices
  k₃ : ℕ  -- number of segments of third color connected to vertices
  sum_k : k₁ + k₂ + k₃ = 3
  valid_k : 0 ≤ k₁ ∧ k₁ ≤ 3 ∧ 0 ≤ k₂ ∧ k₂ ≤ 3 ∧ 0 ≤ k₃ ∧ k₃ ≤ 3
  even_sum : Even (n + k₁) ∧ Even (n + k₂) ∧ Even (n + k₃)

theorem vertex_segments_different_colors (t : ColoredTriangle) : t.k₁ = 1 ∧ t.k₂ = 1 ∧ t.k₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_vertex_segments_different_colors_l2606_260671


namespace NUMINAMATH_CALUDE_problem_statement_l2606_260624

theorem problem_statement (p q r s : ℝ) (ω : ℂ) 
  (hp : p ≠ -1) (hq : q ≠ -1) (hr : r ≠ -1) (hs : s ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1/(p + ω) + 1/(q + ω) + 1/(r + ω) + 1/(s + ω) = 3/ω^2) :
  1/(p + 1) + 1/(q + 1) + 1/(r + 1) + 1/(s + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2606_260624


namespace NUMINAMATH_CALUDE_num_assignment_plans_l2606_260610

/-- The number of male doctors -/
def num_male_doctors : ℕ := 6

/-- The number of female doctors -/
def num_female_doctors : ℕ := 4

/-- The number of male doctors to be selected -/
def selected_male_doctors : ℕ := 3

/-- The number of female doctors to be selected -/
def selected_female_doctors : ℕ := 2

/-- The number of regions -/
def num_regions : ℕ := 5

/-- Function to calculate the number of assignment plans -/
def calculate_assignment_plans : ℕ := sorry

/-- Theorem stating the number of different assignment plans -/
theorem num_assignment_plans : 
  calculate_assignment_plans = 12960 := by sorry

end NUMINAMATH_CALUDE_num_assignment_plans_l2606_260610


namespace NUMINAMATH_CALUDE_sum_ratio_equals_3_l2606_260600

def sum_multiples_of_3 (n : ℕ) : ℕ := 
  3 * (n * (n + 1) / 2)

def sum_integers (m : ℕ) : ℕ := 
  m * (m + 1) / 2

theorem sum_ratio_equals_3 : 
  (sum_multiples_of_3 200) / (sum_integers 200) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_ratio_equals_3_l2606_260600


namespace NUMINAMATH_CALUDE_exists_counterexample_l2606_260666

-- Define a structure for the set S with the binary operation *
structure BinarySystem where
  S : Type u
  op : S → S → S
  at_least_two_elements : ∃ (a b : S), a ≠ b
  property : ∀ (a b : S), op a (op b a) = b

-- State the theorem
theorem exists_counterexample (B : BinarySystem) :
  ∃ (a b : B.S), B.op (B.op a b) a ≠ a := by sorry

end NUMINAMATH_CALUDE_exists_counterexample_l2606_260666


namespace NUMINAMATH_CALUDE_expenditure_increase_l2606_260670

theorem expenditure_increase (income : ℝ) (expenditure : ℝ) (savings : ℝ) 
  (new_income : ℝ) (new_expenditure : ℝ) (new_savings : ℝ) :
  expenditure = 0.75 * income →
  savings = income - expenditure →
  new_income = 1.2 * income →
  new_savings = 1.5 * savings →
  new_savings = new_income - new_expenditure →
  new_expenditure = 1.1 * expenditure :=
by sorry

end NUMINAMATH_CALUDE_expenditure_increase_l2606_260670


namespace NUMINAMATH_CALUDE_mitten_knitting_time_l2606_260627

/-- Represents the time (in hours) to knit each item -/
structure KnittingTimes where
  hat : ℝ
  scarf : ℝ
  sweater : ℝ
  sock : ℝ
  mitten : ℝ

/-- Represents the number of each item in a set -/
structure SetComposition where
  hats : ℕ
  scarves : ℕ
  sweaters : ℕ
  mittens : ℕ
  socks : ℕ

def numGrandchildren : ℕ := 3

def knittingTimes : KnittingTimes := {
  hat := 2,
  scarf := 3,
  sweater := 6,
  sock := 1.5,
  mitten := 0  -- We'll solve for this
}

def setComposition : SetComposition := {
  hats := 1,
  scarves := 1,
  sweaters := 1,
  mittens := 2,
  socks := 2
}

def totalTime : ℝ := 48

theorem mitten_knitting_time :
  ∃ (mittenTime : ℝ),
    mittenTime > 0 ∧
    (let kt := { knittingTimes with mitten := mittenTime };
     (kt.hat * setComposition.hats +
      kt.scarf * setComposition.scarves +
      kt.sweater * setComposition.sweaters +
      kt.mitten * setComposition.mittens +
      kt.sock * setComposition.socks) * numGrandchildren = totalTime) ∧
    mittenTime = 1 := by sorry

end NUMINAMATH_CALUDE_mitten_knitting_time_l2606_260627


namespace NUMINAMATH_CALUDE_solve_for_a_l2606_260650

-- Define the equation for all a, b, and c
def equation (a b c : ℝ) : Prop :=
  a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)

-- Define the theorem
theorem solve_for_a :
  ∀ a : ℝ, (∀ b c : ℝ, equation a b c) → a * 15 * 2 = 4 → a = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_solve_for_a_l2606_260650


namespace NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l2606_260644

/-- Represents a point in 2D Cartesian coordinates -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_of_P_across_x_axis :
  let P : Point2D := { x := 2, y := -3 }
  reflectAcrossXAxis P = { x := 2, y := 3 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l2606_260644


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_B_l2606_260654

def B : Set ℕ := {n : ℕ | ∀ p : ℕ, Nat.Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 7}

theorem sum_of_reciprocals_B : ∑' (n : B), (1 : ℚ) / n = 7 / 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_B_l2606_260654


namespace NUMINAMATH_CALUDE_not_divisible_by_100_l2606_260637

theorem not_divisible_by_100 : ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by sorry

end NUMINAMATH_CALUDE_not_divisible_by_100_l2606_260637


namespace NUMINAMATH_CALUDE_bowling_tournament_sequences_l2606_260614

/-- Represents a tournament with a fixed number of players and rounds. -/
structure Tournament :=
  (num_players : ℕ)
  (num_rounds : ℕ)
  (outcomes_per_match : ℕ)

/-- Calculates the number of possible award distribution sequences for a given tournament. -/
def award_sequences (t : Tournament) : ℕ :=
  t.outcomes_per_match ^ t.num_rounds

/-- Theorem stating that a tournament with 5 players, 4 rounds, and 2 outcomes per match has 16 possible award sequences. -/
theorem bowling_tournament_sequences :
  ∃ t : Tournament, t.num_players = 5 ∧ t.num_rounds = 4 ∧ t.outcomes_per_match = 2 ∧ award_sequences t = 16 :=
sorry

end NUMINAMATH_CALUDE_bowling_tournament_sequences_l2606_260614


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2606_260621

theorem negation_of_universal_proposition (m : ℝ) :
  (¬ ∀ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ (∃ x : ℝ, x^2 + 2*x + m > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2606_260621


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2606_260698

theorem fixed_point_on_line (m : ℝ) : 
  (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2606_260698


namespace NUMINAMATH_CALUDE_f_2017_eq_cos_l2606_260652

open Real

/-- Recursive definition of the function sequence f_n -/
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => sin
  | n + 1 => deriv (f n)

/-- The 2017th function in the sequence equals cosine -/
theorem f_2017_eq_cos : f 2017 = cos := by
  sorry

end NUMINAMATH_CALUDE_f_2017_eq_cos_l2606_260652


namespace NUMINAMATH_CALUDE_not_always_true_parallel_intersection_l2606_260672

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- State the theorem
theorem not_always_true_parallel_intersection
  (h_diff_planes : α ≠ β)
  (h_diff_lines : m ≠ n)
  (h_m_parallel_α : parallel_line_plane m α)
  (h_intersect : intersect α β n) :
  ¬ (parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_not_always_true_parallel_intersection_l2606_260672


namespace NUMINAMATH_CALUDE_gerald_furniture_problem_l2606_260632

/-- Represents the problem of determining the maximum number of chairs Gerald can make --/
theorem gerald_furniture_problem 
  (x t c b : ℕ) 
  (r1 r2 r3 : ℕ) 
  (h_x : x = 2250)
  (h_t : t = 18)
  (h_c : c = 12)
  (h_b : b = 30)
  (h_ratio : r1 = 2 ∧ r2 = 3 ∧ r3 = 1) :
  ∃ (chairs : ℕ), 
    chairs ≤ (x / (t * r1 / r2 + c + b * r3 / r2)) ∧ 
    chairs = 66 := by
  sorry


end NUMINAMATH_CALUDE_gerald_furniture_problem_l2606_260632


namespace NUMINAMATH_CALUDE_map_scale_l2606_260630

/-- If 15 cm on a map represents 90 km, then 20 cm represents 120 km -/
theorem map_scale (map_length : ℝ) (actual_distance : ℝ) : 
  (15 * map_length = 90 * actual_distance) → 
  (20 * map_length = 120 * actual_distance) := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l2606_260630


namespace NUMINAMATH_CALUDE_parabola_translation_l2606_260629

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * dx + p.b
    c := p.a * dx^2 - p.b * dx + p.c + dy }

/-- The original parabola y = x² -/
def original : Parabola :=
  { a := 1, b := 0, c := 0 }

/-- The resulting parabola after translation -/
def translated : Parabola :=
  translate original 2 1

theorem parabola_translation :
  translated = { a := 1, b := -4, c := 5 } :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_l2606_260629


namespace NUMINAMATH_CALUDE_function_domain_condition_l2606_260688

/-- A function f(x) = (kx + 5) / (kx^2 + 4kx + 3) is defined for all real x if and only if 0 ≤ k < 3/4 -/
theorem function_domain_condition (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (k * x + 5) / (k * x^2 + 4 * k * x + 3)) ↔ 
  (0 ≤ k ∧ k < 3/4) :=
by sorry

end NUMINAMATH_CALUDE_function_domain_condition_l2606_260688


namespace NUMINAMATH_CALUDE_road_trip_equation_correct_l2606_260673

/-- Represents a road trip with a stop -/
structure RoadTrip where
  totalDistance : ℝ
  totalTime : ℝ
  stopDuration : ℝ
  speedBeforeStop : ℝ
  speedAfterStop : ℝ

/-- The equation for the road trip is correct -/
def correctEquation (trip : RoadTrip) (t : ℝ) : Prop :=
  trip.speedBeforeStop * t + trip.speedAfterStop * (trip.totalTime - trip.stopDuration / 60 - t) = trip.totalDistance

theorem road_trip_equation_correct (trip : RoadTrip) (t : ℝ) :
  trip.totalDistance = 300 ∧
  trip.totalTime = 4 ∧
  trip.stopDuration = 30 ∧
  trip.speedBeforeStop = 70 ∧
  trip.speedAfterStop = 90 →
  correctEquation trip t ↔ 70 * t + 90 * (3.5 - t) = 300 := by
  sorry


end NUMINAMATH_CALUDE_road_trip_equation_correct_l2606_260673


namespace NUMINAMATH_CALUDE_max_dot_product_l2606_260663

-- Define the hyperbola
def hyperbola (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 16) - (P.2^2 / 9) = 1

-- Define point A in terms of P and t
def point_A (P : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * P.1, t * P.2)

-- Define the dot product condition
def dot_product_condition (P : ℝ × ℝ) (t : ℝ) : Prop :=
  (point_A P t).1 * P.1 + (point_A P t).2 * P.2 = 64

-- Define point B
def B : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem max_dot_product 
  (P : ℝ × ℝ) 
  (t : ℝ) 
  (h1 : hyperbola P) 
  (h2 : dot_product_condition P t) :
  ∃ (M : ℝ), M = 24/5 ∧ ∀ (A : ℝ × ℝ), A = point_A P t → |B.1 * A.1 + B.2 * A.2| ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_l2606_260663


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2606_260631

theorem necessary_but_not_sufficient_condition (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x^2 - 3*a*x + 2*a^2 ≤ 0 → 1/x < 1) ∧
  (∃ x : ℝ, 1/x < 1 ∧ x^2 - 3*a*x + 2*a^2 > 0) →
  a > 1 := by
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2606_260631


namespace NUMINAMATH_CALUDE_max_squares_covered_2inch_card_l2606_260618

/-- Represents a square card -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard -/
structure Checkerboard where
  square_size : ℝ

/-- Calculates the maximum number of squares a card can cover on a checkerboard -/
def max_squares_covered (card : Card) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating the maximum number of squares covered by a 2-inch card on a 1-inch checkerboard -/
theorem max_squares_covered_2inch_card (card : Card) (board : Checkerboard) :
  card.side_length = 2 → board.square_size = 1 → max_squares_covered card board = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_squares_covered_2inch_card_l2606_260618


namespace NUMINAMATH_CALUDE_impossibleTransformation_l2606_260605

/-- Represents a pile of stones -/
structure Pile :=
  (count : ℕ)

/-- Represents the state of all piles -/
structure PileState :=
  (piles : List Pile)

/-- Allowed operations on piles -/
inductive Operation
  | Combine : Pile → Pile → Operation
  | Split : Pile → Operation

/-- Applies an operation to a pile state -/
def applyOperation (state : PileState) (op : Operation) : PileState :=
  sorry

/-- Checks if a pile state is the desired final state -/
def isFinalState (state : PileState) : Prop :=
  state.piles.length = 105 ∧ state.piles.all (fun p => p.count = 1)

/-- The main theorem -/
theorem impossibleTransformation :
  ∀ (operations : List Operation),
    let initialState : PileState := ⟨[⟨51⟩, ⟨49⟩, ⟨5⟩]⟩
    let finalState := operations.foldl applyOperation initialState
    ¬(isFinalState finalState) := by
  sorry

end NUMINAMATH_CALUDE_impossibleTransformation_l2606_260605


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l2606_260617

/-- Proves that a 45% increase in breadth and 88.5% increase in area results in a 30% increase in length for a rectangle -/
theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h_positive : L > 0 ∧ B > 0) :
  B' = 1.45 * B ∧ L' * B' = 1.885 * (L * B) → L' = 1.3 * L := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l2606_260617


namespace NUMINAMATH_CALUDE_probability_ratio_l2606_260669

def total_slips : ℕ := 50
def distinct_numbers : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def p : ℚ := distinct_numbers / Nat.choose total_slips drawn_slips

def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

theorem probability_ratio :
  q / p = 450 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l2606_260669


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l2606_260607

/-- Number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l2606_260607


namespace NUMINAMATH_CALUDE_orchestra_members_count_l2606_260697

theorem orchestra_members_count : ∃! x : ℕ, 
  150 < x ∧ x < 250 ∧ 
  x % 4 = 2 ∧ 
  x % 5 = 3 ∧ 
  x % 8 = 4 ∧ 
  x % 9 = 5 ∧ 
  x = 58 :=
by sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l2606_260697


namespace NUMINAMATH_CALUDE_curve_C_properties_l2606_260699

noncomputable section

/-- Curve C in parametric form -/
def curve_C (φ : ℝ) : ℝ × ℝ := (3 * Real.cos φ, 3 + 3 * Real.sin φ)

/-- Polar equation of a curve -/
structure PolarEquation where
  f : ℝ → ℝ

/-- Line with slope angle and passing through a point -/
structure Line where
  slope_angle : ℝ
  point : ℝ × ℝ

/-- Intersection points of a line and a curve -/
structure Intersection where
  M : ℝ × ℝ
  N : ℝ × ℝ

/-- Main theorem statement -/
theorem curve_C_properties :
  ∃ (polar_eq : PolarEquation) (l : Line) (int : Intersection),
    (∀ θ : ℝ, polar_eq.f θ = 6 * Real.sin θ) ∧
    l.slope_angle = 135 * π / 180 ∧
    l.point = (1, 2) ∧
    (let (xM, yM) := int.M
     let (xN, yN) := int.N
     1 / Real.sqrt ((xM - 1)^2 + (yM - 2)^2) +
     1 / Real.sqrt ((xN - 1)^2 + (yN - 2)^2) = 6 / 7) := by
  sorry

end

end NUMINAMATH_CALUDE_curve_C_properties_l2606_260699


namespace NUMINAMATH_CALUDE_triangle_angle_and_vector_properties_l2606_260678

theorem triangle_angle_and_vector_properties 
  (A B C : ℝ) 
  (h_triangle : A + B + C = Real.pi)
  (m : ℝ × ℝ)
  (h_m : m = (Real.tan A + Real.tan B, Real.sqrt 3))
  (n : ℝ × ℝ)
  (h_n : n = (1, 1 - Real.tan A * Real.tan B))
  (h_perp : m.1 * n.1 + m.2 * n.2 = 0)
  (a : ℝ × ℝ)
  (h_a : a = (Real.sqrt 2 * Real.cos ((A + B) / 2), Real.sin ((A - B) / 2)))
  (h_norm_a : a.1^2 + a.2^2 = 3/2) : 
  C = Real.pi / 3 ∧ Real.tan A * Real.tan B = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_and_vector_properties_l2606_260678


namespace NUMINAMATH_CALUDE_roger_lawn_mowing_l2606_260638

/-- The number of lawns Roger had to mow -/
def total_lawns : ℕ := 14

/-- The amount Roger earns per lawn -/
def earnings_per_lawn : ℕ := 9

/-- The number of lawns Roger forgot to mow -/
def forgotten_lawns : ℕ := 8

/-- The total amount Roger actually earned -/
def actual_earnings : ℕ := 54

/-- Theorem stating that the total number of lawns Roger had to mow is 14 -/
theorem roger_lawn_mowing :
  total_lawns = (actual_earnings / earnings_per_lawn) + forgotten_lawns :=
by sorry

end NUMINAMATH_CALUDE_roger_lawn_mowing_l2606_260638


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2606_260643

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (x_ge : x ≥ -2/3)
  (y_ge : y ≥ -1)
  (z_ge : z ≥ -2) :
  ∃ (max : ℝ), max = Real.sqrt 57 ∧ 
    ∀ a b c : ℝ, a + b + c = 2 → a ≥ -2/3 → b ≥ -1 → c ≥ -2 →
      Real.sqrt (3*a + 2) + Real.sqrt (3*b + 4) + Real.sqrt (3*c + 7) ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2606_260643


namespace NUMINAMATH_CALUDE_inequality_range_l2606_260649

theorem inequality_range :
  {a : ℝ | ∀ x : ℝ, a * (4 - Real.sin x)^4 - 3 + (Real.cos x)^2 + a > 0} = {a : ℝ | a > 3/82} := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2606_260649


namespace NUMINAMATH_CALUDE_units_digit_of_17_power_28_l2606_260667

theorem units_digit_of_17_power_28 :
  ∃ n : ℕ, 17^28 ≡ 1 [ZMOD 10] ∧ 17 ≡ 7 [ZMOD 10] :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_power_28_l2606_260667


namespace NUMINAMATH_CALUDE_geometric_number_difference_l2606_260620

/-- A function that checks if a 3-digit number is geometric --/
def is_geometric (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  ∃ (a b c : ℕ) (r : ℚ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    r > 0 ∧
    b = (a : ℚ) * r ∧
    c = (b : ℚ) * r

/-- A function that checks if a number starts with an even digit --/
def starts_with_even (n : ℕ) : Prop :=
  ∃ (a : ℕ), n = 100 * a + n % 100 ∧ Even a

/-- The theorem to be proved --/
theorem geometric_number_difference :
  ∃ (max min : ℕ),
    is_geometric max ∧
    is_geometric min ∧
    starts_with_even max ∧
    starts_with_even min ∧
    (∀ n, is_geometric n ∧ starts_with_even n → n ≤ max) ∧
    (∀ n, is_geometric n ∧ starts_with_even n → n ≥ min) ∧
    max - min = 403 :=
sorry

end NUMINAMATH_CALUDE_geometric_number_difference_l2606_260620


namespace NUMINAMATH_CALUDE_parabola_translation_l2606_260616

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the vertical translation
def vertical_translation (y : ℝ) : ℝ := y + 3

-- Define the horizontal translation
def horizontal_translation (x : ℝ) : ℝ := x - 1

-- State the theorem
theorem parabola_translation :
  ∀ x : ℝ, vertical_translation (original_parabola (horizontal_translation x)) = (x - 1)^2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2606_260616


namespace NUMINAMATH_CALUDE_total_shaded_area_is_75_over_4_l2606_260659

/-- Represents a truncated square-based pyramid -/
structure TruncatedPyramid where
  base_side : ℝ
  top_side : ℝ
  height : ℝ

/-- Calculate the total shaded area of the truncated pyramid -/
def total_shaded_area (p : TruncatedPyramid) : ℝ :=
  sorry

/-- The main theorem stating that the total shaded area is 75/4 -/
theorem total_shaded_area_is_75_over_4 :
  ∀ (p : TruncatedPyramid),
  p.base_side = 7 ∧ p.top_side = 1 ∧ p.height = 4 →
  total_shaded_area p = 75 / 4 := by
  sorry

end NUMINAMATH_CALUDE_total_shaded_area_is_75_over_4_l2606_260659


namespace NUMINAMATH_CALUDE_inequality_proof_l2606_260686

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2606_260686


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l2606_260645

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -6*x^2 + 36*x - 30

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l2606_260645


namespace NUMINAMATH_CALUDE_worker_savings_proof_l2606_260612

/-- Represents the fraction of take-home pay saved each month -/
def savings_fraction : ℚ := 2 / 5

theorem worker_savings_proof (monthly_pay : ℝ) (h1 : monthly_pay > 0) :
  let yearly_savings := 12 * savings_fraction * monthly_pay
  let monthly_not_saved := (1 - savings_fraction) * monthly_pay
  yearly_savings = 8 * monthly_not_saved :=
by sorry

end NUMINAMATH_CALUDE_worker_savings_proof_l2606_260612


namespace NUMINAMATH_CALUDE_age_problem_l2606_260668

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l2606_260668


namespace NUMINAMATH_CALUDE_car_price_proof_l2606_260647

-- Define the original cost price
def original_price : ℝ := 52325.58

-- Define the first sale price (14% loss)
def first_sale_price : ℝ := original_price * 0.86

-- Define the second sale price (20% gain from first sale)
def second_sale_price : ℝ := 54000

-- Theorem statement
theorem car_price_proof :
  (first_sale_price * 1.2 = second_sale_price) ∧
  (original_price > 0) ∧
  (first_sale_price > 0) ∧
  (second_sale_price > 0) :=
sorry

end NUMINAMATH_CALUDE_car_price_proof_l2606_260647


namespace NUMINAMATH_CALUDE_smoking_hospitalization_percentage_l2606_260602

theorem smoking_hospitalization_percentage 
  (total_students : ℕ) 
  (smoking_percentage : ℚ) 
  (non_hospitalized : ℕ) 
  (h1 : total_students = 300)
  (h2 : smoking_percentage = 2/5)
  (h3 : non_hospitalized = 36) :
  (total_students * smoking_percentage - non_hospitalized) / (total_students * smoking_percentage) = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_smoking_hospitalization_percentage_l2606_260602


namespace NUMINAMATH_CALUDE_no_consecutive_squares_l2606_260604

/-- The number of positive integer divisors of n -/
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The sequence defined by a_(n+1) = a_n + τ(n) -/
def a : ℕ → ℕ
  | 0 => 1  -- Arbitrary starting value
  | n + 1 => a n + tau n

/-- Two consecutive terms of the sequence cannot both be perfect squares -/
theorem no_consecutive_squares (n : ℕ) : ¬(∃ k m : ℕ, a n = k^2 ∧ a (n + 1) = m^2) := by
  sorry


end NUMINAMATH_CALUDE_no_consecutive_squares_l2606_260604


namespace NUMINAMATH_CALUDE_bugs_meeting_time_l2606_260664

/-- The time for two bugs to meet again at the starting point on two tangent circles -/
theorem bugs_meeting_time (r1 r2 v1 v2 : ℝ) (hr1 : r1 = 8) (hr2 : r2 = 4) 
  (hv1 : v1 = 3 * Real.pi) (hv2 : v2 = 4 * Real.pi) : 
  ∃ t : ℝ, t = 48 ∧ 
  (∃ n1 n2 : ℕ, t * v1 = n1 * (2 * Real.pi * r1) ∧ 
               t * v2 = n2 * (2 * Real.pi * r2)) := by
  sorry

#check bugs_meeting_time

end NUMINAMATH_CALUDE_bugs_meeting_time_l2606_260664


namespace NUMINAMATH_CALUDE_second_equation_result_l2606_260683

theorem second_equation_result (x y : ℤ) 
  (eq1 : 3 * x + y = 40) 
  (eq2 : 3 * y^2 = 48) : 
  2 * x - y = 20 := by
sorry

end NUMINAMATH_CALUDE_second_equation_result_l2606_260683


namespace NUMINAMATH_CALUDE_carbon_atoms_in_compound_l2606_260636

-- Define atomic weights
def atomic_weight_C : ℝ := 12
def atomic_weight_H : ℝ := 1
def atomic_weight_O : ℝ := 16

-- Define the compound properties
def hydrogen_atoms : ℕ := 6
def oxygen_atoms : ℕ := 2
def molecular_weight : ℝ := 122

-- Theorem to prove
theorem carbon_atoms_in_compound :
  ∃ (carbon_atoms : ℕ),
    (carbon_atoms : ℝ) * atomic_weight_C +
    (hydrogen_atoms : ℝ) * atomic_weight_H +
    (oxygen_atoms : ℝ) * atomic_weight_O =
    molecular_weight ∧
    carbon_atoms = 7 := by
  sorry

end NUMINAMATH_CALUDE_carbon_atoms_in_compound_l2606_260636


namespace NUMINAMATH_CALUDE_cube_difference_divisibility_l2606_260695

theorem cube_difference_divisibility (a b : ℤ) :
  24 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3) ↔ 3 ∣ (a - b) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_divisibility_l2606_260695


namespace NUMINAMATH_CALUDE_arcsin_sqrt3_div2_l2606_260615

theorem arcsin_sqrt3_div2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sqrt3_div2_l2606_260615


namespace NUMINAMATH_CALUDE_circle_symmetry_l2606_260601

-- Define the original circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y = 2

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y x' y' : ℝ),
    circle_C x y →
    line_l ((x + x') / 2) ((y + y') / 2) →
    symmetric_circle x' y' :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2606_260601


namespace NUMINAMATH_CALUDE_sector_max_area_angle_l2606_260677

/-- Given a sector with circumference 4 cm, the central angle that maximizes the area is π radians. -/
theorem sector_max_area_angle (r : ℝ) (θ : ℝ) :
  r * θ + 2 * r = 4 →  -- Circumference condition
  (∀ r' θ', r' * θ' + 2 * r' = 4 → 
    (1/2) * r^2 * θ ≥ (1/2) * r'^2 * θ') →  -- Area maximization condition
  θ = π :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_angle_l2606_260677


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2606_260634

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence a_n, if a_3 + a_8 = 22 and a_6 = 7, then a_5 = 15 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 22) 
  (h_a6 : a 6 = 7) : 
  a 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2606_260634


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l2606_260622

def boat_speed : ℝ := 24
def downstream_distance : ℝ := 56
def downstream_time : ℝ := 2

theorem stream_speed_calculation (v : ℝ) : 
  downstream_distance = (boat_speed + v) * downstream_time →
  v = 4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l2606_260622


namespace NUMINAMATH_CALUDE_inlet_pipe_rate_l2606_260603

/-- Given a tank with the following properties:
  * Capacity of 6048 liters
  * Empties in 7 hours due to a leak
  * Empties in 12 hours when both the leak and an inlet pipe are open
  Prove that the rate at which the inlet pipe fills water is 360 liters per hour -/
theorem inlet_pipe_rate (tank_capacity : ℝ) (leak_empty_time : ℝ) (both_empty_time : ℝ)
  (h1 : tank_capacity = 6048)
  (h2 : leak_empty_time = 7)
  (h3 : both_empty_time = 12) :
  let leak_rate := tank_capacity / leak_empty_time
  let net_empty_rate := tank_capacity / both_empty_time
  leak_rate - (leak_rate - net_empty_rate) = 360 := by
sorry


end NUMINAMATH_CALUDE_inlet_pipe_rate_l2606_260603


namespace NUMINAMATH_CALUDE_non_juniors_playing_instruments_l2606_260613

theorem non_juniors_playing_instruments (total_students : ℕ) 
  (junior_play_percent : ℚ) (non_junior_not_play_percent : ℚ) 
  (total_not_play_percent : ℚ) :
  total_students = 600 →
  junior_play_percent = 30 / 100 →
  non_junior_not_play_percent = 35 / 100 →
  total_not_play_percent = 40 / 100 →
  ∃ (non_juniors_playing : ℕ), non_juniors_playing = 334 :=
by sorry

end NUMINAMATH_CALUDE_non_juniors_playing_instruments_l2606_260613


namespace NUMINAMATH_CALUDE_two_distinct_cool_triples_for_odd_x_l2606_260641

/-- A cool type triple (x, y, z) consists of positive integers with y ≥ 2 
    and satisfies the equation x^2 - 3y^2 = z^2 - 3 -/
def CoolTriple (x y z : ℕ) : Prop :=
  x > 0 ∧ y ≥ 2 ∧ z > 0 ∧ x^2 - 3*y^2 = z^2 - 3

/-- For every odd x ≥ 5, there exist at least two distinct cool triples -/
theorem two_distinct_cool_triples_for_odd_x (x : ℕ) (h_odd : Odd x) (h_ge_5 : x ≥ 5) :
  ∃ y1 z1 y2 z2 : ℕ, 
    CoolTriple x y1 z1 ∧ 
    CoolTriple x y2 z2 ∧ 
    (y1 ≠ y2 ∨ z1 ≠ z2) :=
by
  sorry

end NUMINAMATH_CALUDE_two_distinct_cool_triples_for_odd_x_l2606_260641


namespace NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l2606_260696

theorem square_difference_formula_inapplicable :
  ∀ (a b : ℝ), ¬∃ (x y : ℝ), (-a + b) * (-b + a) = x^2 - y^2 := by
  sorry

#check square_difference_formula_inapplicable

end NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l2606_260696


namespace NUMINAMATH_CALUDE_job_completion_time_l2606_260608

theorem job_completion_time (r_A r_B r_C : ℝ) 
  (h_AB : r_A + r_B = 1 / 3)
  (h_BC : r_B + r_C = 1 / 6)
  (h_CA : r_C + r_A = 1 / 4) :
  1 / r_C = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2606_260608


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l2606_260658

theorem jason_pokemon_cards (initial_cards new_cards : ℕ) 
  (h1 : initial_cards = 676)
  (h2 : new_cards = 224) :
  initial_cards + new_cards = 900 := by
  sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l2606_260658


namespace NUMINAMATH_CALUDE_sum_of_squares_219_l2606_260679

theorem sum_of_squares_219 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^2 + b^2 + c^2 = 219 →
  (a : ℕ) + b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_219_l2606_260679


namespace NUMINAMATH_CALUDE_tournament_outcomes_l2606_260657

/-- Represents a bowler in the tournament -/
inductive Bowler
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents a match between two bowlers -/
structure Match where
  player1 : Bowler
  player2 : Bowler

/-- Represents the tournament structure -/
structure Tournament where
  initialRound : List Match
  subsequentRounds : List Match

/-- Represents the outcome of the tournament -/
structure Outcome where
  prizeOrder : List Bowler

/-- The number of possible outcomes for the tournament -/
def numberOfOutcomes (t : Tournament) : Nat :=
  2^5

/-- Theorem stating that the number of possible outcomes is 32 -/
theorem tournament_outcomes (t : Tournament) :
  numberOfOutcomes t = 32 := by
  sorry

end NUMINAMATH_CALUDE_tournament_outcomes_l2606_260657


namespace NUMINAMATH_CALUDE_james_travel_distance_l2606_260689

/-- The distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: James' travel distance -/
theorem james_travel_distance :
  let speed : ℝ := 80.0
  let time : ℝ := 16.0
  distance speed time = 1280.0 := by
  sorry

end NUMINAMATH_CALUDE_james_travel_distance_l2606_260689


namespace NUMINAMATH_CALUDE_exam_time_allocation_l2606_260662

theorem exam_time_allocation (total_questions : ℕ) (exam_duration_hours : ℕ) 
  (type_a_problems : ℕ) (h1 : total_questions = 200) (h2 : exam_duration_hours = 3) 
  (h3 : type_a_problems = 25) :
  let exam_duration_minutes : ℕ := exam_duration_hours * 60
  let type_b_problems : ℕ := total_questions - type_a_problems
  let x : ℚ := (exam_duration_minutes : ℚ) / (type_a_problems * 2 + type_b_problems)
  (2 * x * type_a_problems : ℚ) = 40 := by
  sorry

#check exam_time_allocation

end NUMINAMATH_CALUDE_exam_time_allocation_l2606_260662


namespace NUMINAMATH_CALUDE_triangle_altitude_on_square_diagonal_l2606_260665

theorem triangle_altitude_on_square_diagonal (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let diagonal := s * Real.sqrt 2
  let triangle_area := (1/2) * diagonal * altitude
  ∃ altitude : ℝ, 
    (square_area = triangle_area) ∧ 
    (altitude = s * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_on_square_diagonal_l2606_260665


namespace NUMINAMATH_CALUDE_connie_marbles_l2606_260639

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℝ := 183.0

/-- The number of marbles Connie has left -/
def marbles_left : ℝ := 593.0

/-- The initial number of marbles Connie had -/
def initial_marbles : ℝ := marbles_given + marbles_left

theorem connie_marbles : initial_marbles = 776.0 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l2606_260639


namespace NUMINAMATH_CALUDE_prob_two_s_is_one_tenth_l2606_260651

/-- The set of tiles containing letters G, A, U, S, and S -/
def tiles : Finset Char := {'G', 'A', 'U', 'S', 'S'}

/-- The number of S tiles in the set -/
def num_s_tiles : Nat := (tiles.filter (· = 'S')).card

/-- The probability of selecting two S tiles when choosing 2 tiles at random -/
def prob_two_s : ℚ := (num_s_tiles.choose 2 : ℚ) / (tiles.card.choose 2)

/-- Theorem stating that the probability of selecting two S tiles is 1/10 -/
theorem prob_two_s_is_one_tenth : prob_two_s = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_prob_two_s_is_one_tenth_l2606_260651


namespace NUMINAMATH_CALUDE_first_term_to_common_difference_ratio_l2606_260681

/-- An arithmetic progression where the sum of the first 14 terms is three times the sum of the first 7 terms -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_condition : (14 * a + 91 * d) = 3 * (7 * a + 21 * d)

/-- The ratio of the first term to the common difference is 4:1 -/
theorem first_term_to_common_difference_ratio 
  (ap : ArithmeticProgression) : ap.a / ap.d = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_term_to_common_difference_ratio_l2606_260681


namespace NUMINAMATH_CALUDE_hannahs_appliance_cost_l2606_260635

/-- The total cost of a washing machine and dryer after applying a discount -/
def total_cost_after_discount (washing_machine_cost : ℝ) (dryer_cost_difference : ℝ) (discount_rate : ℝ) : ℝ :=
  let dryer_cost := washing_machine_cost - dryer_cost_difference
  let total_cost := washing_machine_cost + dryer_cost
  let discount := total_cost * discount_rate
  total_cost - discount

/-- Theorem stating the total cost after discount for Hannah's purchase -/
theorem hannahs_appliance_cost :
  total_cost_after_discount 100 30 0.1 = 153 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_appliance_cost_l2606_260635
