import Mathlib

namespace NUMINAMATH_CALUDE_xy_value_l2789_278978

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2789_278978


namespace NUMINAMATH_CALUDE_fifth_term_geometric_l2789_278920

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n-1)

/-- The fifth term of a geometric sequence with first term 5 and common ratio 3y is 405y^4 -/
theorem fifth_term_geometric (y : ℝ) :
  geometric_term 5 (3*y) 5 = 405 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_geometric_l2789_278920


namespace NUMINAMATH_CALUDE_bob_cannot_win_and_prevent_alice_l2789_278951

def game_number : Set ℕ := {19, 20}
def start_number : Set ℕ := {9, 10}

theorem bob_cannot_win_and_prevent_alice (s : ℕ) (a : ℕ) :
  s ∈ start_number →
  a ∈ game_number →
  (∀ n : ℤ, s + 39 * n ≠ 2019) ∧
  (s = 9 → ∀ n : ℤ, s + 39 * n + a ≠ 2019) :=
by sorry

end NUMINAMATH_CALUDE_bob_cannot_win_and_prevent_alice_l2789_278951


namespace NUMINAMATH_CALUDE_pasta_bins_l2789_278955

theorem pasta_bins (soup_bins vegetables_bins total_bins : ℝ)
  (h1 : soup_bins = 0.125)
  (h2 : vegetables_bins = 0.125)
  (h3 : total_bins = 0.75) :
  total_bins - (soup_bins + vegetables_bins) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_pasta_bins_l2789_278955


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2789_278949

theorem polynomial_factorization :
  ∀ x : ℝ, (x^2 + 2*x + 1) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 7*x + 1) * (x^2 + 3*x + 7) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2789_278949


namespace NUMINAMATH_CALUDE_quadratic_sum_abc_l2789_278935

theorem quadratic_sum_abc : ∃ (a b c : ℝ), 
  (∀ x, 15 * x^2 + 75 * x + 375 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 298.75) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_abc_l2789_278935


namespace NUMINAMATH_CALUDE_original_trees_eq_sum_l2789_278911

/-- The number of trees Haley originally grew in her backyard -/
def original_trees : ℕ := 20

/-- The number of trees left after the typhoon -/
def trees_left : ℕ := 4

/-- The number of trees that died in the typhoon -/
def trees_died : ℕ := 16

/-- Theorem stating that the original number of trees equals the sum of trees left and trees that died -/
theorem original_trees_eq_sum : original_trees = trees_left + trees_died := by
  sorry

end NUMINAMATH_CALUDE_original_trees_eq_sum_l2789_278911


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2789_278947

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i : ℂ) / (3 + i) = (1 : ℂ) / 10 + (3 : ℂ) / 10 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2789_278947


namespace NUMINAMATH_CALUDE_C_closest_to_one_l2789_278921

def A : ℝ := 0.959595
def B : ℝ := 1.05555
def C : ℝ := 0.960960
def D : ℝ := 1.040040
def E : ℝ := 0.955555

theorem C_closest_to_one :
  |1 - C| < |1 - A| ∧
  |1 - C| < |1 - B| ∧
  |1 - C| < |1 - D| ∧
  |1 - C| < |1 - E| := by
sorry

end NUMINAMATH_CALUDE_C_closest_to_one_l2789_278921


namespace NUMINAMATH_CALUDE_figurine_cost_calculation_l2789_278907

def brand_a_price : ℝ := 65
def brand_b_price : ℝ := 75
def num_brand_a : ℕ := 3
def num_brand_b : ℕ := 2
def num_figurines : ℕ := 8
def figurine_total_cost : ℝ := brand_b_price + 40

theorem figurine_cost_calculation :
  (figurine_total_cost / num_figurines : ℝ) = 14.375 := by sorry

end NUMINAMATH_CALUDE_figurine_cost_calculation_l2789_278907


namespace NUMINAMATH_CALUDE_least_number_of_grapes_l2789_278967

theorem least_number_of_grapes : ∃ n : ℕ, n > 0 ∧ 
  n % 19 = 1 ∧ n % 23 = 1 ∧ n % 29 = 1 ∧ 
  ∀ m : ℕ, m > 0 → m % 19 = 1 → m % 23 = 1 → m % 29 = 1 → n ≤ m :=
by
  use 12209
  sorry

end NUMINAMATH_CALUDE_least_number_of_grapes_l2789_278967


namespace NUMINAMATH_CALUDE_cubic_function_property_l2789_278966

/-- Given a cubic function y = ax³ + bx² + cx + d, if (2, y₁) and (-2, y₂) lie on its graph
    and y₁ - y₂ = 12, then c = 3 - 4a. -/
theorem cubic_function_property (a b c d y₁ y₂ : ℝ) :
  y₁ = 8*a + 4*b + 2*c + d →
  y₂ = -8*a + 4*b - 2*c + d →
  y₁ - y₂ = 12 →
  c = 3 - 4*a :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2789_278966


namespace NUMINAMATH_CALUDE_equation_solution_l2789_278946

theorem equation_solution : 
  ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 ∧ x = -30 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2789_278946


namespace NUMINAMATH_CALUDE_multiplier_value_l2789_278988

theorem multiplier_value (n : ℝ) (x : ℝ) (h1 : n = 1) (h2 : 3 * n - 1 = x * n) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_value_l2789_278988


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_l2789_278909

theorem x_positive_sufficient_not_necessary :
  (∀ x : ℝ, x > 0 → |x - 1| - |x| ≤ 1) ∧
  (∃ x : ℝ, x ≤ 0 ∧ |x - 1| - |x| ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_l2789_278909


namespace NUMINAMATH_CALUDE_M_eq_real_l2789_278969

/-- The set of complex numbers Z satisfying (Z-1)^2 = |Z-1|^2 -/
def M : Set ℂ := {Z | (Z - 1)^2 = Complex.abs (Z - 1)^2}

/-- Theorem stating that M is equal to the set of real numbers -/
theorem M_eq_real : M = {Z : ℂ | Z.im = 0} := by sorry

end NUMINAMATH_CALUDE_M_eq_real_l2789_278969


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2789_278929

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2789_278929


namespace NUMINAMATH_CALUDE_min_stamps_for_60_cents_l2789_278998

theorem min_stamps_for_60_cents : ∃ (s t : ℕ), 
  5 * s + 6 * t = 60 ∧ 
  s + t = 11 ∧
  ∀ (s' t' : ℕ), 5 * s' + 6 * t' = 60 → s + t ≤ s' + t' := by
  sorry

end NUMINAMATH_CALUDE_min_stamps_for_60_cents_l2789_278998


namespace NUMINAMATH_CALUDE_hueys_pizza_size_proof_l2789_278977

theorem hueys_pizza_size_proof (small_side : ℝ) (small_cost : ℝ) (large_cost : ℝ) 
  (individual_budget : ℝ) (extra_area : ℝ) :
  small_side = 12 →
  small_cost = 10 →
  large_cost = 20 →
  individual_budget = 30 →
  extra_area = 36 →
  ∃ (large_side : ℝ),
    large_side = 10 * Real.sqrt 3 ∧
    3 * (large_side ^ 2) = 2 * (3 * small_side ^ 2) + extra_area :=
by
  sorry

end NUMINAMATH_CALUDE_hueys_pizza_size_proof_l2789_278977


namespace NUMINAMATH_CALUDE_amount_less_than_five_times_number_l2789_278915

theorem amount_less_than_five_times_number (N : ℕ) (A : ℕ) : 
  N = 52 → A < 5 * N → A = 232 → A = A 
:= by sorry

end NUMINAMATH_CALUDE_amount_less_than_five_times_number_l2789_278915


namespace NUMINAMATH_CALUDE_orange_distribution_l2789_278956

theorem orange_distribution (total_oranges : ℕ) (bad_oranges : ℕ) (difference : ℕ) :
  total_oranges = 108 →
  bad_oranges = 36 →
  difference = 3 →
  (total_oranges : ℚ) / (total_oranges / difference - bad_oranges / difference) - 
  ((total_oranges - bad_oranges) : ℚ) / (total_oranges / difference - bad_oranges / difference) = difference →
  total_oranges / difference - bad_oranges / difference = 12 :=
by sorry

end NUMINAMATH_CALUDE_orange_distribution_l2789_278956


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2789_278950

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence where a_4 = 3, a_2 * a_6 = 9 -/
theorem geometric_sequence_product (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) (h_a4 : a 4 = 3) : 
  a 2 * a 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2789_278950


namespace NUMINAMATH_CALUDE_diagonal_passes_810_cubes_l2789_278904

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem: The number of unit cubes an internal diagonal passes through
    in a 160 × 330 × 380 rectangular solid is 810 -/
theorem diagonal_passes_810_cubes :
  cubes_passed_by_diagonal 160 330 380 = 810 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_810_cubes_l2789_278904


namespace NUMINAMATH_CALUDE_max_triangle_area_ellipse_circle_intersection_l2789_278954

/-- Given an ellipse E and a line x = t intersecting it, this theorem proves
    the maximum area of triangle ABC formed by the intersection of a circle
    with the y-axis, where the circle's diameter is the chord of the ellipse. -/
theorem max_triangle_area_ellipse_circle_intersection
  (a : ℝ) (t : ℝ) 
  (ha : a > Real.sqrt 3) 
  (ht : t > 0) 
  (he : Real.sqrt (a^2 - 3) / a = 1/2) :
  let E := {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / 3 = 1}
  let M := (t, Real.sqrt ((1 - t^2 / a^2) * 3))
  let N := (t, -Real.sqrt ((1 - t^2 / a^2) * 3))
  let C := {p : ℝ × ℝ | (p.1 - t)^2 + p.2^2 = ((M.2 - N.2) / 2)^2}
  let A := (0, Real.sqrt ((M.2 - N.2)^2 / 4 - t^2))
  let B := (0, -Real.sqrt ((M.2 - N.2)^2 / 4 - t^2))
  ∃ (tmax : ℝ), tmax > 0 ∧ 
    (∀ t' > 0, t' * Real.sqrt (12 - 7 * t'^2) / 2 ≤ tmax * Real.sqrt (12 - 7 * tmax^2) / 2) ∧
    tmax * Real.sqrt (12 - 7 * tmax^2) / 2 = 3 * Real.sqrt 7 / 7 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_ellipse_circle_intersection_l2789_278954


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2789_278953

theorem solution_set_inequality (x : ℝ) : 
  (3 - 2*x) * (x + 1) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2789_278953


namespace NUMINAMATH_CALUDE_equation_solution_l2789_278942

theorem equation_solution : ∃ c : ℚ, (c - 23) / 2 = (2 * c + 5) / 7 ∧ c = 57 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2789_278942


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2789_278968

/-- The line (m-1)x + (2m-1)y = m-5 passes through the point (9, -4) for any real m -/
theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2789_278968


namespace NUMINAMATH_CALUDE_sock_pair_count_l2789_278970

/-- The number of ways to choose a pair of socks of different colors -/
def differentColorPairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem: There are 93 ways to choose a pair of socks of different colors
    from a drawer containing 5 white, 5 brown, 4 blue, and 2 red socks -/
theorem sock_pair_count :
  differentColorPairs 5 5 4 2 = 93 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l2789_278970


namespace NUMINAMATH_CALUDE_dataset_groups_l2789_278938

/-- Calculate the number of groups for a dataset given its maximum value, minimum value, and class interval. -/
def number_of_groups (max_value min_value class_interval : ℕ) : ℕ :=
  (max_value - min_value) / class_interval + 1

/-- Theorem: For a dataset with maximum value 140, minimum value 50, and class interval 10, 
    the number of groups is 10. -/
theorem dataset_groups :
  number_of_groups 140 50 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dataset_groups_l2789_278938


namespace NUMINAMATH_CALUDE_lattice_points_sum_l2789_278940

/-- Number of lattice points in a plane region -/
noncomputable def N (D : Set (ℝ × ℝ)) : ℕ := sorry

/-- Region A -/
def A : Set (ℝ × ℝ) := {(x, y) | y = x^2 ∧ x ≤ 0 ∧ x ≥ -10 ∧ y ≤ 1}

/-- Region B -/
def B : Set (ℝ × ℝ) := {(x, y) | y = x^2 ∧ x ≥ 0 ∧ x ≤ 1 ∧ y ≤ 100}

/-- Theorem: The sum of lattice points in the union and intersection of A and B is 1010 -/
theorem lattice_points_sum : N (A ∪ B) + N (A ∩ B) = 1010 := by sorry

end NUMINAMATH_CALUDE_lattice_points_sum_l2789_278940


namespace NUMINAMATH_CALUDE_angle_C_is_105_degrees_l2789_278912

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem angle_C_is_105_degrees (t : Triangle) 
  (h1 : t.a = 3)
  (h2 : t.b = 3 * Real.sqrt 2)
  (h3 : t.B = π / 4) : -- 45° in radians
  t.C = 7 * π / 12 := -- 105° in radians
by sorry

end NUMINAMATH_CALUDE_angle_C_is_105_degrees_l2789_278912


namespace NUMINAMATH_CALUDE_product_digit_sum_l2789_278980

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

theorem product_digit_sum :
  let product := number1 * number2
  let thousands_digit := (product / 1000) % 10
  let units_digit := product % 10
  thousands_digit + units_digit = 5 := by
sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2789_278980


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2789_278997

/-- A rhombus with diagonals of 6 and 8 units has a perimeter of 20 units. -/
theorem rhombus_perimeter (d₁ d₂ : ℝ) (h₁ : d₁ = 6) (h₂ : d₂ = 8) :
  let side := Real.sqrt ((d₁/2)^2 + (d₂/2)^2)
  4 * side = 20 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2789_278997


namespace NUMINAMATH_CALUDE_domain_of_f_l2789_278979

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iic 2 :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l2789_278979


namespace NUMINAMATH_CALUDE_inequality_solution_l2789_278931

theorem inequality_solution (x : ℝ) : 3 - 2 / (3 * x + 4) < 5 ↔ x < -4/3 ∨ x > -5/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2789_278931


namespace NUMINAMATH_CALUDE_factorization_equality_l2789_278932

theorem factorization_equality (x : ℝ) : 
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 = 
  (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + 2) * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2789_278932


namespace NUMINAMATH_CALUDE_population_growth_l2789_278963

theorem population_growth (x : ℝ) : 
  (((1 + x / 100) * 4) - 1) * 100 = 1100 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_l2789_278963


namespace NUMINAMATH_CALUDE_marions_score_l2789_278957

theorem marions_score (total_items : ℕ) (ellas_incorrect : ℕ) (marions_additional : ℕ) : 
  total_items = 40 →
  ellas_incorrect = 4 →
  marions_additional = 6 →
  (total_items - ellas_incorrect) / 2 + marions_additional = 24 :=
by sorry

end NUMINAMATH_CALUDE_marions_score_l2789_278957


namespace NUMINAMATH_CALUDE_initial_books_l2789_278919

theorem initial_books (x : ℚ) : 
  (1/2 * x + 3 = 23) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_l2789_278919


namespace NUMINAMATH_CALUDE_blanket_thickness_proof_l2789_278936

-- Define the initial thickness of the blanket
def initial_thickness : ℝ := 3

-- Define a function that calculates the thickness after n foldings
def thickness_after_foldings (n : ℕ) : ℝ :=
  initial_thickness * (2 ^ n)

-- Theorem statement
theorem blanket_thickness_proof :
  thickness_after_foldings 4 = 48 :=
by
  sorry


end NUMINAMATH_CALUDE_blanket_thickness_proof_l2789_278936


namespace NUMINAMATH_CALUDE_quadratic_properties_l2789_278916

def f (x : ℝ) := -2 * x^2 + 4 * x + 3

theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x > f y) ∧
  (∀ x : ℝ, f (x + 1) = f (1 - x)) ∧
  (f 1 = 5) ∧
  (∀ x : ℝ, x > 1 → ∀ y : ℝ, y > x → f y < f x) ∧
  (∀ x : ℝ, x < 1 → ∀ y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2789_278916


namespace NUMINAMATH_CALUDE_dans_final_limes_l2789_278993

def initial_limes : ℕ := 9
def sara_gift : ℕ := 4
def juice_used : ℕ := 5
def neighbor_gift : ℕ := 3

theorem dans_final_limes : 
  initial_limes + sara_gift - juice_used - neighbor_gift = 5 := by
  sorry

end NUMINAMATH_CALUDE_dans_final_limes_l2789_278993


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l2789_278972

def ben_lap_time : ℕ := 5
def clara_lap_time : ℕ := 9
def david_lap_time : ℕ := 8

theorem earliest_meeting_time :
  let meeting_time := Nat.lcm (Nat.lcm ben_lap_time clara_lap_time) david_lap_time
  meeting_time = 360 := by sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l2789_278972


namespace NUMINAMATH_CALUDE_alternating_arithmetic_series_sum_l2789_278908

def arithmetic_series (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => a₁ - i * d)

def alternating_sign (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => if i % 2 == 0 then 1 else -1)

def series_sum (series : List ℤ) : ℤ :=
  series.sum

theorem alternating_arithmetic_series_sum :
  let a₁ : ℤ := 2005
  let d : ℤ := 10
  let n : ℕ := 200
  let series := List.zip (arithmetic_series a₁ d n) (alternating_sign n) |>.map (fun (x, y) => x * y)
  series_sum series = 1000 := by
  sorry

end NUMINAMATH_CALUDE_alternating_arithmetic_series_sum_l2789_278908


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2789_278923

theorem two_numbers_problem (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : 
  a * b = 875 ∧ a^2 + b^2 = 1850 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2789_278923


namespace NUMINAMATH_CALUDE_gcf_of_360_and_150_l2789_278982

theorem gcf_of_360_and_150 : Nat.gcd 360 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_360_and_150_l2789_278982


namespace NUMINAMATH_CALUDE_b_work_days_l2789_278983

/-- Represents the number of days it takes for a person to complete the work alone -/
structure WorkDays where
  days : ℕ

/-- Represents the rate at which a person completes the work per day -/
def workRate (w : WorkDays) : ℚ :=
  1 / w.days

theorem b_work_days (total_payment : ℕ) (a_work : WorkDays) (abc_work : WorkDays) (c_share : ℕ) :
  total_payment = 1200 →
  a_work.days = 6 →
  abc_work.days = 3 →
  c_share = 150 →
  ∃ b_work : WorkDays,
    b_work.days = 24 ∧
    workRate a_work + workRate b_work + (c_share : ℚ) / total_payment = workRate abc_work :=
by sorry

end NUMINAMATH_CALUDE_b_work_days_l2789_278983


namespace NUMINAMATH_CALUDE_train_length_l2789_278960

/-- Calculates the length of a train given its speed, the speed of a bus moving in the opposite direction, and the time it takes for the train to pass the bus. -/
theorem train_length (train_speed : ℝ) (bus_speed : ℝ) (passing_time : ℝ) :
  train_speed = 90 →
  bus_speed = 60 →
  passing_time = 5.279577633789296 →
  let relative_speed := (train_speed + bus_speed) * (5 / 18)
  let train_length := relative_speed * passing_time
  train_length = 41.663147 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2789_278960


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2789_278928

theorem inequality_equivalence :
  ∀ a : ℝ, a > 0 →
  ((∀ t₁ t₂ t₃ t₄ : ℝ, t₁ > 0 → t₂ > 0 → t₃ > 0 → t₄ > 0 → 
    t₁ * t₂ * t₃ * t₄ = a^4 →
    (1 / Real.sqrt (1 + t₁)) + (1 / Real.sqrt (1 + t₂)) + 
    (1 / Real.sqrt (1 + t₃)) + (1 / Real.sqrt (1 + t₄)) ≤ 
    4 / Real.sqrt (1 + a))
  ↔ 
  (0 < a ∧ a ≤ 7/9)) := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2789_278928


namespace NUMINAMATH_CALUDE_product_and_difference_imply_sum_l2789_278974

theorem product_and_difference_imply_sum (x y : ℕ+) : 
  x * y = 24 → x - y = 5 → x + y = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_and_difference_imply_sum_l2789_278974


namespace NUMINAMATH_CALUDE_files_deleted_l2789_278901

theorem files_deleted (initial_music : ℕ) (initial_video : ℕ) (remaining : ℕ) : 
  initial_music = 26 → initial_video = 36 → remaining = 14 →
  initial_music + initial_video - remaining = 48 := by
  sorry

end NUMINAMATH_CALUDE_files_deleted_l2789_278901


namespace NUMINAMATH_CALUDE_fireworks_display_total_l2789_278900

/-- The number of fireworks used in a New Year's Eve display -/
def fireworks_display (fireworks_per_number : ℕ) (fireworks_per_letter : ℕ) 
  (year_digits : ℕ) (phrase_letters : ℕ) (additional_boxes : ℕ) (fireworks_per_box : ℕ) : ℕ :=
  (fireworks_per_number * year_digits) + 
  (fireworks_per_letter * phrase_letters) + 
  (additional_boxes * fireworks_per_box)

/-- Theorem stating the total number of fireworks used in the display -/
theorem fireworks_display_total : 
  fireworks_display 6 5 4 12 50 8 = 484 := by
  sorry

end NUMINAMATH_CALUDE_fireworks_display_total_l2789_278900


namespace NUMINAMATH_CALUDE_supermarket_spending_l2789_278930

theorem supermarket_spending (total : ℝ) : 
  (1/2 : ℝ) * total + (1/3 : ℝ) * total + (1/10 : ℝ) * total + 10 = total → 
  total = 150 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2789_278930


namespace NUMINAMATH_CALUDE_remainder_problem_l2789_278959

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 34 = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2789_278959


namespace NUMINAMATH_CALUDE_california_texas_plate_difference_l2789_278914

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of possible California license plates -/
def california_plates : ℕ := num_letters^5 * num_digits^2

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_digits^2 * num_letters^4

/-- The difference in the number of possible license plates between California and Texas -/
def plate_difference : ℕ := california_plates - texas_plates

theorem california_texas_plate_difference :
  plate_difference = 1142440000 := by
  sorry

end NUMINAMATH_CALUDE_california_texas_plate_difference_l2789_278914


namespace NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_317_l2789_278990

theorem multiplicative_inverse_203_mod_317 :
  ∃ x : ℕ, x < 317 ∧ (203 * x) % 317 = 1 :=
by
  use 46
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_317_l2789_278990


namespace NUMINAMATH_CALUDE_salary_change_l2789_278905

theorem salary_change (S : ℝ) : 
  S * (1 + 0.25) * (1 - 0.15) * (1 + 0.10) * (1 - 0.20) = S * 0.935 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_l2789_278905


namespace NUMINAMATH_CALUDE_symmetry_about_point_period_four_l2789_278948

-- Define the function f
variable (f : ℝ → ℝ)

-- Statement ②
theorem symmetry_about_point (h : ∀ x, f (x + 1) + f (1 - x) = 0) :
  ∀ x, f (2 - x) = -f x :=
sorry

-- Statement ④
theorem period_four (h : ∀ x, f (1 + x) + f (x - 1) = 0) :
  ∀ x, f (x + 4) = f x :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_point_period_four_l2789_278948


namespace NUMINAMATH_CALUDE_closest_to_sqrt_diff_l2789_278992

def options : List ℝ := [0.18, 0.19, 0.20, 0.21, 0.22]

theorem closest_to_sqrt_diff (x : ℝ) (hx : x ∈ options) :
  x = 0.21 ↔ ∀ y ∈ options, |Real.sqrt 68 - Real.sqrt 64 - x| ≤ |Real.sqrt 68 - Real.sqrt 64 - y| :=
sorry

end NUMINAMATH_CALUDE_closest_to_sqrt_diff_l2789_278992


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2789_278944

/-- Given a rhombus with diagonals of 14 inches and 48 inches, its perimeter is 100 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 48) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2789_278944


namespace NUMINAMATH_CALUDE_base_9_conversion_l2789_278975

/-- Converts a list of digits in base 9 to its decimal (base 10) representation -/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The problem statement -/
theorem base_9_conversion :
  base9ToDecimal [1, 3, 3, 2] = 1729 := by
  sorry

end NUMINAMATH_CALUDE_base_9_conversion_l2789_278975


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l2789_278925

theorem reciprocal_of_sum : (1 / (1/3 + 1/4) : ℚ) = 12/7 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l2789_278925


namespace NUMINAMATH_CALUDE_infinitely_many_triples_l2789_278962

theorem infinitely_many_triples :
  ∀ n : ℕ, ∃ (a b p : ℕ),
    Prime p ∧
    0 < a ∧ a ≤ b ∧ b < p ∧
    (p^5 ∣ (a + b)^p - a^p - b^p) ∧
    p > n :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_triples_l2789_278962


namespace NUMINAMATH_CALUDE_not_all_odd_have_all_five_multiple_l2789_278985

theorem not_all_odd_have_all_five_multiple : ∃ n : ℕ, Odd n ∧ ∀ k : ℕ, ∃ d : ℕ, d ≠ 5 ∧ d ∈ (k * n).digits 10 := by
  sorry

end NUMINAMATH_CALUDE_not_all_odd_have_all_five_multiple_l2789_278985


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2789_278995

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (49 - b) + c / (81 - c) = 8) :
  6 / (36 - a) + 7 / (49 - b) + 9 / (81 - c) = 66 / 36 + 77 / 49 + 99 / 81 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2789_278995


namespace NUMINAMATH_CALUDE_sum_of_angles_in_figure_l2789_278999

-- Define the angles
def angle_A : ℝ := 34
def angle_B : ℝ := 80
def angle_C : ℝ := 24

-- Define x and y as real numbers (measures of angles)
variable (x y : ℝ)

-- Define the theorem
theorem sum_of_angles_in_figure (h1 : 0 ≤ x) (h2 : 0 ≤ y) : x + y = 132 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_angles_in_figure_l2789_278999


namespace NUMINAMATH_CALUDE_units_digit_of_M_M_10_l2789_278994

-- Define the sequence M_n
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | n + 2 => M (n + 1) + M n

-- Function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_M_M_10 : unitsDigit (M (M 10)) = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_M_M_10_l2789_278994


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2789_278933

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = (a^2 - 3*a + 2) + Complex.I * (a - 1)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2789_278933


namespace NUMINAMATH_CALUDE_problem_solution_l2789_278941

def even_squared_sum : ℕ := (2^2) + (4^2) + (6^2) + (8^2) + (10^2)

def prime_count : ℕ := 4

def odd_product : ℕ := 1 * 3 * 5 * 7 * 9

theorem problem_solution :
  let x := even_squared_sum
  let y := prime_count
  let z := odd_product
  x - y + z = 1161 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2789_278941


namespace NUMINAMATH_CALUDE_katie_homework_problem_l2789_278934

/-- The number of math problems Katie finished on the bus ride home. -/
def finished_problems : ℕ := 5

/-- The number of math problems Katie had left to do. -/
def remaining_problems : ℕ := 4

/-- The total number of math problems Katie had for homework. -/
def total_problems : ℕ := finished_problems + remaining_problems

theorem katie_homework_problem :
  total_problems = 9 := by sorry

end NUMINAMATH_CALUDE_katie_homework_problem_l2789_278934


namespace NUMINAMATH_CALUDE_jovanas_shells_l2789_278910

/-- The total amount of shells Jovana has after her friends add to her collection -/
def total_shells (initial : ℕ) (friend1 : ℕ) (friend2 : ℕ) : ℕ :=
  initial + friend1 + friend2

/-- Theorem stating that Jovana's total shells equal 37 pounds -/
theorem jovanas_shells :
  total_shells 5 15 17 = 37 := by
  sorry

end NUMINAMATH_CALUDE_jovanas_shells_l2789_278910


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l2789_278913

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -8 * x^2 + 4 * x - 3 < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l2789_278913


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l2789_278952

theorem add_preserves_inequality (a b : ℝ) (h : a < b) : 3 + a < 3 + b := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l2789_278952


namespace NUMINAMATH_CALUDE_gcd_lcm_product_36_210_l2789_278991

theorem gcd_lcm_product_36_210 : Nat.gcd 36 210 * Nat.lcm 36 210 = 7560 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_36_210_l2789_278991


namespace NUMINAMATH_CALUDE_hyperbola_t_squared_l2789_278903

/-- A hyperbola centered at the origin, opening horizontally -/
structure Hyperbola where
  /-- The equation of the hyperbola: x²/a² - y²/b² = 1 -/
  equation : ℝ → ℝ → Prop

/-- The hyperbola passes through the given points -/
def passes_through (h : Hyperbola) (x y : ℝ) : Prop :=
  h.equation x y

theorem hyperbola_t_squared (h : Hyperbola) :
  passes_through h 2 3 →
  passes_through h 3 0 →
  passes_through h t 5 →
  t^2 = 1854/81 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_t_squared_l2789_278903


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2789_278917

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (3*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a₆ + a₅ + a₄ + a₃ + a₂ + a₁ + a = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2789_278917


namespace NUMINAMATH_CALUDE_four_Y_three_equals_negative_twentythree_l2789_278965

-- Define the Y operation
def Y (a b : ℤ) : ℤ := a^2 - 2 * a * b * 2 + b^2

-- Theorem statement
theorem four_Y_three_equals_negative_twentythree :
  Y 4 3 = -23 := by
  sorry

end NUMINAMATH_CALUDE_four_Y_three_equals_negative_twentythree_l2789_278965


namespace NUMINAMATH_CALUDE_exists_rational_triangle_l2789_278981

/-- A triangle with integer sides, height, and median, all less than 100 -/
structure RationalTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  height : ℕ
  median : ℕ
  a_lt_100 : a < 100
  b_lt_100 : b < 100
  c_lt_100 : c < 100
  height_lt_100 : height < 100
  median_lt_100 : median < 100
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  not_right_triangle : a^2 + b^2 ≠ c^2 ∧ b^2 + c^2 ≠ a^2 ∧ c^2 + a^2 ≠ b^2

/-- There exists a triangle with integer sides, height, and median, all less than 100, that is not a right triangle -/
theorem exists_rational_triangle : ∃ t : RationalTriangle, True := by
  sorry

end NUMINAMATH_CALUDE_exists_rational_triangle_l2789_278981


namespace NUMINAMATH_CALUDE_find_number_l2789_278973

theorem find_number : ∃ x : ℝ, (5 * x) / (180 / 3) + 70 = 71 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_find_number_l2789_278973


namespace NUMINAMATH_CALUDE_unique_stamp_arrangements_l2789_278924

/-- Represents the number of stamps of each denomination -/
def stamp_counts : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Represents the value of each stamp denomination -/
def stamp_values : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- A type to represent a stamp arrangement -/
structure StampArrangement where
  stamps : List Nat
  sum_to_ten : (stamps.sum = 10)

/-- Function to count unique arrangements -/
def count_unique_arrangements (stamps : List Nat) (values : List Nat) : Nat :=
  sorry

/-- The main theorem stating that there are 88 unique arrangements -/
theorem unique_stamp_arrangements :
  count_unique_arrangements stamp_counts stamp_values = 88 := by
  sorry

end NUMINAMATH_CALUDE_unique_stamp_arrangements_l2789_278924


namespace NUMINAMATH_CALUDE_product_simplification_l2789_278906

theorem product_simplification (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l2789_278906


namespace NUMINAMATH_CALUDE_parallelogram_rotation_volume_ratio_l2789_278996

/-- Given a parallelogram with adjacent sides a and b, the ratio of the volume of the cylinder
    formed by rotating the parallelogram around side a to the volume of the cylinder formed by
    rotating the parallelogram around side b is equal to a/b. -/
theorem parallelogram_rotation_volume_ratio
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (π * (a/2)^2 * b) / (π * (b/2)^2 * a) = a / b :=
sorry

end NUMINAMATH_CALUDE_parallelogram_rotation_volume_ratio_l2789_278996


namespace NUMINAMATH_CALUDE_difference_number_and_fraction_difference_150_and_its_three_fifths_l2789_278939

theorem difference_number_and_fraction (n : ℚ) : n - (3 / 5) * n = (2 / 5) * n := by sorry

theorem difference_150_and_its_three_fifths : 150 - (3 / 5) * 150 = 60 := by sorry

end NUMINAMATH_CALUDE_difference_number_and_fraction_difference_150_and_its_three_fifths_l2789_278939


namespace NUMINAMATH_CALUDE_power_simplification_l2789_278989

theorem power_simplification :
  (8^5 / 8^2) * 2^10 - 2^2 = 2^19 - 4 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l2789_278989


namespace NUMINAMATH_CALUDE_inequality_proof_l2789_278926

theorem inequality_proof (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y^2 + z^2 = 1) :
  1 ≤ (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ∧
  (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2789_278926


namespace NUMINAMATH_CALUDE_not_iff_right_angle_and_equation_l2789_278987

/-- Definition of a triangle with sides a, b, c and altitude m from vertex C -/
structure Triangle :=
  (a b c m : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (positive_altitude : 0 < m)

/-- The equation in question -/
def satisfies_equation (t : Triangle) : Prop :=
  1 / t.m^2 = 1 / t.a^2 + 1 / t.b^2

/-- Theorem stating that the original statement is not true in general -/
theorem not_iff_right_angle_and_equation :
  ∃ (t : Triangle), satisfies_equation t ∧ ¬(t.a^2 + t.b^2 = t.c^2) :=
sorry

end NUMINAMATH_CALUDE_not_iff_right_angle_and_equation_l2789_278987


namespace NUMINAMATH_CALUDE_value_of_y_l2789_278927

theorem value_of_y (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 21) : y = 84 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2789_278927


namespace NUMINAMATH_CALUDE_opposite_values_theorem_l2789_278945

theorem opposite_values_theorem (a b : ℝ) 
  (h : |a - 2| + (b + 1)^2 = 0) : 
  b^a = 1 ∧ a^3 + b^15 = 7 := by sorry

end NUMINAMATH_CALUDE_opposite_values_theorem_l2789_278945


namespace NUMINAMATH_CALUDE_integer_sum_problem_l2789_278986

theorem integer_sum_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l2789_278986


namespace NUMINAMATH_CALUDE_barbara_wins_iff_odd_sum_l2789_278918

/-- Newspaper cutting game -/
def newspaper_game_winner (a b d : ℝ) : Prop :=
  let x := ⌊a / d⌋
  let y := ⌊b / d⌋
  Odd (x + y)

/-- Barbara wins the newspaper cutting game if and only if the sum of the floor divisions is odd -/
theorem barbara_wins_iff_odd_sum (a b d : ℝ) (h : d > 0) :
  newspaper_game_winner a b d ↔ Barbara_wins :=
sorry

end NUMINAMATH_CALUDE_barbara_wins_iff_odd_sum_l2789_278918


namespace NUMINAMATH_CALUDE_job_completion_time_l2789_278961

/-- The time taken for a, b, and c to finish a job together, given the conditions. -/
theorem job_completion_time (a b c : ℝ) : 
  (a + b = 1 / 15) →  -- a and b finish the job in 15 days
  (c = 1 / 7.5) →     -- c alone finishes the job in 7.5 days
  (1 / (a + b + c) = 5) :=  -- a, b, and c together finish the job in 5 days
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l2789_278961


namespace NUMINAMATH_CALUDE_determine_c_absolute_value_l2789_278902

/-- The polynomial g(x) = ax^4 + bx^3 + cx^2 + bx + a -/
def g (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

/-- The theorem statement -/
theorem determine_c_absolute_value (a b c : ℤ) : 
  g a b c (3 + Complex.I) = 0 ∧ 
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 111 := by
  sorry

end NUMINAMATH_CALUDE_determine_c_absolute_value_l2789_278902


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2789_278937

theorem arithmetic_equality : 5 * 7 + 6 * 12 + 7 * 4 + 2 * 9 = 153 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2789_278937


namespace NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l2789_278984

theorem integer_solutions_cubic_equation :
  ∀ x y : ℤ, x^3 - y^3 = 2*x*y + 8 ↔ (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l2789_278984


namespace NUMINAMATH_CALUDE_largest_non_composite_sum_l2789_278971

def is_composite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

theorem largest_non_composite_sum : 
  (∀ n : ℕ, n > 11 → ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b) ∧
  ¬(∃ a b : ℕ, is_composite a ∧ is_composite b ∧ 11 = a + b) :=
sorry

end NUMINAMATH_CALUDE_largest_non_composite_sum_l2789_278971


namespace NUMINAMATH_CALUDE_car_lot_power_windows_l2789_278943

theorem car_lot_power_windows 
  (total : ℕ) 
  (air_bags : ℕ) 
  (both : ℕ) 
  (neither : ℕ) 
  (h1 : total = 65)
  (h2 : air_bags = 45)
  (h3 : both = 12)
  (h4 : neither = 2) :
  ∃ power_windows : ℕ, power_windows = 30 ∧ 
    total = air_bags + power_windows - both + neither :=
by sorry

end NUMINAMATH_CALUDE_car_lot_power_windows_l2789_278943


namespace NUMINAMATH_CALUDE_range_of_z_l2789_278976

theorem range_of_z (x y z : ℝ) 
  (hx : -1 ≤ x ∧ x ≤ 2) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : z = 2*x - y) : 
  -3 ≤ z ∧ z ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_z_l2789_278976


namespace NUMINAMATH_CALUDE_average_marks_math_biology_l2789_278922

theorem average_marks_math_biology 
  (P C M B : ℕ) -- Marks in Physics, Chemistry, Mathematics, and Biology
  (h : P + C + M + B = P + C + 200) -- Total marks condition
  : (M + B) / 2 = 100 := by
sorry

end NUMINAMATH_CALUDE_average_marks_math_biology_l2789_278922


namespace NUMINAMATH_CALUDE_cone_volume_increase_l2789_278958

/-- The volume of a cone increases by 612.8% when its height is increased by 120% and its radius is increased by 80% -/
theorem cone_volume_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  let v := (1/3) * Real.pi * r^2 * h
  let r_new := 1.8 * r
  let h_new := 2.2 * h
  let v_new := (1/3) * Real.pi * r_new^2 * h_new
  (v_new - v) / v * 100 = 612.8 := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_increase_l2789_278958


namespace NUMINAMATH_CALUDE_sandwich_meal_combinations_l2789_278964

theorem sandwich_meal_combinations : 
  ∃! n : ℕ, n = (Finset.filter 
    (λ (pair : ℕ × ℕ) => 5 * pair.1 + 7 * pair.2 = 90) 
    (Finset.product (Finset.range 19) (Finset.range 13))).card := by
  sorry

end NUMINAMATH_CALUDE_sandwich_meal_combinations_l2789_278964
