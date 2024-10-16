import Mathlib

namespace NUMINAMATH_CALUDE_sphere_box_height_l2918_291893

/-- A rectangular box with a large sphere and eight smaller spheres -/
structure SphereBox where
  length : ℝ
  width : ℝ
  height : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ
  small_sphere_count : ℕ

/-- Conditions for the sphere arrangement in the box -/
def valid_sphere_arrangement (box : SphereBox) : Prop :=
  box.length = 6 ∧
  box.width = 6 ∧
  box.large_sphere_radius = 3 ∧
  box.small_sphere_radius = 1 ∧
  box.small_sphere_count = 8 ∧
  ∀ (small_sphere : Fin box.small_sphere_count),
    (∃ (side1 side2 side3 : ℝ), side1 + side2 + side3 = box.length + box.width + box.height) ∧
    (box.large_sphere_radius + box.small_sphere_radius = 
     (box.length / 2)^2 + (box.width / 2)^2 + (box.height / 2 - box.small_sphere_radius)^2)

/-- Theorem stating that the height of the box is 8 -/
theorem sphere_box_height (box : SphereBox) 
  (h : valid_sphere_arrangement box) : box.height = 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_box_height_l2918_291893


namespace NUMINAMATH_CALUDE_value_of_T_l2918_291816

theorem value_of_T : ∃ T : ℚ, (1/2 : ℚ) * (1/7 : ℚ) * T = (1/3 : ℚ) * (1/5 : ℚ) * 90 ∧ T = 84 := by
  sorry

end NUMINAMATH_CALUDE_value_of_T_l2918_291816


namespace NUMINAMATH_CALUDE_sock_order_ratio_l2918_291807

def sock_order_problem (black_socks green_socks : ℕ) (price_green : ℝ) : Prop :=
  let price_black := 3 * price_green
  let original_cost := black_socks * price_black + green_socks * price_green
  let interchanged_cost := green_socks * price_black + black_socks * price_green
  black_socks = 5 ∧
  interchanged_cost = 1.8 * original_cost ∧
  (black_socks : ℝ) / green_socks = 3 / 11

theorem sock_order_ratio :
  ∃ (green_socks : ℕ) (price_green : ℝ),
    sock_order_problem 5 green_socks price_green :=
by sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l2918_291807


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2918_291894

theorem condition_necessary_not_sufficient :
  (∀ x y : ℝ, x = 1 ∧ y = 2 → x + y = 3) ∧
  (∃ x y : ℝ, x + y = 3 ∧ (x ≠ 1 ∨ y ≠ 2)) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2918_291894


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2918_291822

theorem quadratic_roots_sum_and_product (x₁ x₂ : ℝ) :
  x₁^2 - 3*x₁ + 1 = 0 → x₂^2 - 3*x₂ + 1 = 0 → x₁ + x₂ + x₁*x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2918_291822


namespace NUMINAMATH_CALUDE_intersection_when_a_4_range_of_a_for_sufficient_condition_l2918_291844

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x ∈ Set.Icc 2 3, y = -2^x}
def B (a : ℝ) : Set ℝ := {x | x^2 + 3*x - a^2 - 3*a > 0}

-- Part 1: Intersection when a = 4
theorem intersection_when_a_4 :
  A ∩ B 4 = {x | -8 < x ∧ x < -7} := by sorry

-- Part 2: Range of a for sufficient but not necessary condition
theorem range_of_a_for_sufficient_condition :
  (∀ x, x ∈ A → x ∈ B a) ∧ (∃ x, x ∈ B a ∧ x ∉ A) ↔ -4 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_4_range_of_a_for_sufficient_condition_l2918_291844


namespace NUMINAMATH_CALUDE_min_value_of_f_l2918_291817

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (x^2 + 2)) + Real.sqrt (x^2 + 2)

theorem min_value_of_f :
  ∃ (min_val : ℝ), (∀ x, f x ≥ min_val) ∧ (min_val = (3 * Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2918_291817


namespace NUMINAMATH_CALUDE_parallel_vectors_l2918_291802

/-- Given two 2D vectors a and b, find the value of k such that 
    (2a + b) is parallel to (1/2a + kb) -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (1, 2)) :
  ∃ k : ℝ, k = (1 : ℝ) / 4 ∧ 
  ∃ c : ℝ, c ≠ 0 ∧ c • (2 • a + b) = (1 / 2 : ℝ) • a + k • b :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2918_291802


namespace NUMINAMATH_CALUDE_digits_of_2_to_70_l2918_291897

theorem digits_of_2_to_70 : ∃ n : ℕ, n = 22 ∧ (2^70 : ℕ) < 10^n ∧ 10^(n-1) ≤ 2^70 :=
  sorry

end NUMINAMATH_CALUDE_digits_of_2_to_70_l2918_291897


namespace NUMINAMATH_CALUDE_work_time_problem_l2918_291875

/-- The time taken to complete a work when multiple workers work together -/
def combined_work_time (work_rates : List ℚ) : ℚ :=
  1 / (work_rates.sum)

/-- The problem of finding the combined work time for A, B, and C -/
theorem work_time_problem :
  let a_rate : ℚ := 1 / 12
  let b_rate : ℚ := 1 / 24
  let c_rate : ℚ := 1 / 18
  combined_work_time [a_rate, b_rate, c_rate] = 72 / 13 := by
  sorry

#eval combined_work_time [1/12, 1/24, 1/18]

end NUMINAMATH_CALUDE_work_time_problem_l2918_291875


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l2918_291854

/-- A quadratic function with specific properties -/
def q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, q a b c x = q a b c (15 - x)) →
  q a b c 0 = -3 →
  q a b c 15 = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l2918_291854


namespace NUMINAMATH_CALUDE_decimal_value_changes_when_removing_zeros_l2918_291853

theorem decimal_value_changes_when_removing_zeros : 7.0800 ≠ 7.8 := by sorry

end NUMINAMATH_CALUDE_decimal_value_changes_when_removing_zeros_l2918_291853


namespace NUMINAMATH_CALUDE_not_perfect_square_l2918_291811

theorem not_perfect_square (k : ℕ+) : ¬ ∃ (n : ℕ), (16 * k + 8 : ℕ) = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2918_291811


namespace NUMINAMATH_CALUDE_magician_earnings_l2918_291874

/-- Represents the sales and conditions of the magician's card deck business --/
structure MagicianSales where
  initialPrice : ℝ
  initialStock : ℕ
  finalStock : ℕ
  promotionPrice : ℝ
  initialExchangeRate : ℝ
  changedExchangeRate : ℝ
  foreignCustomersBulk : ℕ
  foreignCustomersSingle : ℕ
  domesticCustomers : ℕ

/-- Calculates the total earnings of the magician in dollars --/
def calculateEarnings (sales : MagicianSales) : ℝ :=
  sorry

/-- Theorem stating that the magician's earnings equal 11 dollars --/
theorem magician_earnings (sales : MagicianSales) 
  (h1 : sales.initialPrice = 2)
  (h2 : sales.initialStock = 5)
  (h3 : sales.finalStock = 3)
  (h4 : sales.promotionPrice = 3)
  (h5 : sales.initialExchangeRate = 1)
  (h6 : sales.changedExchangeRate = 1.5)
  (h7 : sales.foreignCustomersBulk = 2)
  (h8 : sales.foreignCustomersSingle = 1)
  (h9 : sales.domesticCustomers = 2) :
  calculateEarnings sales = 11 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_l2918_291874


namespace NUMINAMATH_CALUDE_jerry_feathers_l2918_291892

theorem jerry_feathers (x : ℕ) : 
  let hawk_feathers : ℕ := 6
  let eagle_feathers : ℕ := x * hawk_feathers
  let total_feathers : ℕ := hawk_feathers + eagle_feathers
  let remaining_after_gift : ℕ := total_feathers - 10
  let sold_feathers : ℕ := remaining_after_gift / 2
  let final_feathers : ℕ := remaining_after_gift - sold_feathers
  (final_feathers = 49) → (x = 17) :=
by sorry

end NUMINAMATH_CALUDE_jerry_feathers_l2918_291892


namespace NUMINAMATH_CALUDE_riddle_count_l2918_291843

theorem riddle_count (josh ivory taso : ℕ) : 
  josh = 8 → 
  ivory = josh + 4 → 
  taso = 2 * ivory → 
  taso = 24 := by sorry

end NUMINAMATH_CALUDE_riddle_count_l2918_291843


namespace NUMINAMATH_CALUDE_correct_cases_delivered_l2918_291841

/-- The number of tins in each case -/
def tins_per_case : ℕ := 24

/-- The percentage of undamaged tins -/
def undamaged_percentage : ℚ := 95/100

/-- The number of undamaged tins left -/
def undamaged_tins : ℕ := 342

/-- The number of cases delivered -/
def cases_delivered : ℕ := 15

theorem correct_cases_delivered :
  cases_delivered * tins_per_case * undamaged_percentage = undamaged_tins := by
  sorry

end NUMINAMATH_CALUDE_correct_cases_delivered_l2918_291841


namespace NUMINAMATH_CALUDE_expected_pine_saplings_l2918_291898

/-- Represents the number of pine saplings in a stratified sample. -/
def pine_saplings_in_sample (total_saplings : ℕ) (pine_saplings : ℕ) (sample_size : ℕ) : ℚ :=
  (pine_saplings : ℚ) / (total_saplings : ℚ) * (sample_size : ℚ)

/-- Theorem stating the expected number of pine saplings in the sample. -/
theorem expected_pine_saplings :
  pine_saplings_in_sample 30000 4000 150 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expected_pine_saplings_l2918_291898


namespace NUMINAMATH_CALUDE_ratio_of_powers_compute_power_ratio_l2918_291879

theorem ratio_of_powers (a b : ℕ) (n : ℕ) (h : b ≠ 0) :
  (a ^ n) / (b ^ n) = (a / b) ^ n :=
sorry

theorem compute_power_ratio :
  (90000 ^ 5) / (30000 ^ 5) = 243 :=
sorry

end NUMINAMATH_CALUDE_ratio_of_powers_compute_power_ratio_l2918_291879


namespace NUMINAMATH_CALUDE_smallest_perimeter_consecutive_odd_triangle_l2918_291884

/-- Represents three consecutive odd integers -/
structure ConsecutiveOddIntegers where
  a : ℕ
  h_odd : Odd a
  h_consecutive : (a + 2, a + 4) = (a.succ.succ, a.succ.succ.succ.succ)

/-- Checks if three numbers form a valid triangle -/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- The smallest possible perimeter of a triangle with consecutive odd integer side lengths -/
theorem smallest_perimeter_consecutive_odd_triangle :
  ∃ (t : ConsecutiveOddIntegers),
    (is_valid_triangle t.a (t.a + 2) (t.a + 4)) ∧
    (∀ (s : ConsecutiveOddIntegers),
      is_valid_triangle s.a (s.a + 2) (s.a + 4) →
      t.a + (t.a + 2) + (t.a + 4) ≤ s.a + (s.a + 2) + (s.a + 4)) ∧
    t.a + (t.a + 2) + (t.a + 4) = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_consecutive_odd_triangle_l2918_291884


namespace NUMINAMATH_CALUDE_line_through_points_l2918_291846

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def lies_on_line (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

theorem line_through_points : 
  let p1 : Point := ⟨8, 16⟩
  let p2 : Point := ⟨2, -2⟩
  let p3 : Point := ⟨5, 7⟩
  let p4 : Point := ⟨4, 4⟩
  let p5 : Point := ⟨10, 22⟩
  let p6 : Point := ⟨-2, -12⟩
  let p7 : Point := ⟨1, -5⟩
  lies_on_line p3 p1 p2 ∧
  lies_on_line p4 p1 p2 ∧
  lies_on_line p5 p1 p2 ∧
  lies_on_line p7 p1 p2 ∧
  ¬ lies_on_line p6 p1 p2 :=
by
  sorry


end NUMINAMATH_CALUDE_line_through_points_l2918_291846


namespace NUMINAMATH_CALUDE_male_population_in_village_l2918_291862

theorem male_population_in_village (total_population : ℕ) 
  (h1 : total_population = 800) 
  (num_groups : ℕ) 
  (h2 : num_groups = 4) 
  (h3 : total_population % num_groups = 0) 
  (h4 : ∃ (male_group : ℕ), male_group ≤ num_groups ∧ 
    male_group * (total_population / num_groups) = total_population / num_groups) :
  total_population / num_groups = 200 :=
by sorry

end NUMINAMATH_CALUDE_male_population_in_village_l2918_291862


namespace NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l2918_291847

/-- A linear function f(x) = -x - 2 -/
def f (x : ℝ) : ℝ := -x - 2

/-- The first quadrant of the coordinate plane -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem linear_function_not_in_first_quadrant :
  ∀ x : ℝ, ¬(first_quadrant x (f x)) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l2918_291847


namespace NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l2918_291805

theorem divisibility_implies_multiple_of_three (a b : ℤ) : 
  (9 ∣ a^2 + a*b + b^2) → (3 ∣ a) ∧ (3 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l2918_291805


namespace NUMINAMATH_CALUDE_fraction_calculation_l2918_291861

theorem fraction_calculation : 
  let f1 := 531 / 135
  let f2 := 579 / 357
  let f3 := 753 / 975
  let f4 := 135 / 531
  (f1 + f2 + f3) * (f2 + f3 + f4) - (f1 + f2 + f3 + f4) * (f2 + f3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2918_291861


namespace NUMINAMATH_CALUDE_dvd_book_capacity_l2918_291829

theorem dvd_book_capacity (total_capacity : ℕ) (current_dvds : ℕ) (h1 : total_capacity = 126) (h2 : current_dvds = 81) :
  total_capacity - current_dvds = 45 := by
  sorry

end NUMINAMATH_CALUDE_dvd_book_capacity_l2918_291829


namespace NUMINAMATH_CALUDE_constant_value_l2918_291830

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define points
def P : ℝ × ℝ := (-2, 0)
def Q : ℝ × ℝ := (-2, -1)

-- Define the line l
def line_l (n : ℝ) (x y : ℝ) : Prop := x = n * (y + 1) - 2

-- Define the intersection points A and B
def A (n : ℝ) : ℝ × ℝ := sorry
def B (n : ℝ) : ℝ × ℝ := sorry

-- Define points C and D
def C (n : ℝ) : ℝ × ℝ := sorry
def D (n : ℝ) : ℝ × ℝ := sorry

-- Define the distances |QC| and |QD|
def QC (n : ℝ) : ℝ := sorry
def QD (n : ℝ) : ℝ := sorry

-- The main theorem
theorem constant_value (n : ℝ) :
  ellipse (A n).1 (A n).2 ∧ 
  ellipse (B n).1 (B n).2 ∧ 
  (A n).2 < 0 ∧ 
  (B n).2 < 0 ∧
  line_l n (A n).1 (A n).2 ∧
  line_l n (B n).1 (B n).2 →
  QC n + QD n - QC n * QD n = 0 :=
sorry

end

end NUMINAMATH_CALUDE_constant_value_l2918_291830


namespace NUMINAMATH_CALUDE_f_2015_value_l2918_291835

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) + f x = 0) ∧
  (∀ x, f (x + 1) = f (1 - x)) ∧
  (f 1 = 5)

theorem f_2015_value (f : ℝ → ℝ) (h : f_properties f) : f 2015 = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_2015_value_l2918_291835


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2918_291869

theorem complex_fraction_evaluation : 
  1 / ( 3 + 1 / ( 3 + 1 / ( 3 - 1 / 3 ) ) ) = 27/89 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2918_291869


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l2918_291877

/-- The range of m for which exactly two distinct points on the circle x^2 + (y-1)^2 = 4 
    are at a distance of 1 from the line √3x + y + m = 0 -/
theorem circle_line_intersection_range (m : ℝ) : 
  (∃! (p q : ℝ × ℝ), p ≠ q ∧ 
    (p.1^2 + (p.2 - 1)^2 = 4) ∧ 
    (q.1^2 + (q.2 - 1)^2 = 4) ∧
    ((Real.sqrt 3 * p.1 + p.2 + m)^2 / (3 + 1) = 1) ∧
    ((Real.sqrt 3 * q.1 + q.2 + m)^2 / (3 + 1) = 1)) ↔ 
  (-7 < m ∧ m < -3) ∨ (1 < m ∧ m < 5) :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l2918_291877


namespace NUMINAMATH_CALUDE_train_crossing_time_l2918_291891

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 240 →
  train_speed_kmh = 216 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2918_291891


namespace NUMINAMATH_CALUDE_percent_of_x_is_y_l2918_291837

theorem percent_of_x_is_y (x y : ℝ) (h : 0.7 * (x - y) = 0.3 * (x + y)) : y = 0.4 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_is_y_l2918_291837


namespace NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l2918_291899

theorem least_sum_of_equal_multiples (x y z : ℕ+) (h : (2 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (8 : ℕ) * z.val) :
  x.val + y.val + z.val ≥ 33 ∧ ∃ (a b c : ℕ+), (2 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (8 : ℕ) * c.val ∧ a.val + b.val + c.val = 33 :=
by
  sorry

#check least_sum_of_equal_multiples

end NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l2918_291899


namespace NUMINAMATH_CALUDE_nail_color_percentage_difference_l2918_291821

theorem nail_color_percentage_difference (total nails : ℕ) (purple blue : ℕ) :
  total = 20 →
  purple = 6 →
  blue = 8 →
  let striped := total - purple - blue
  let blue_percentage := (blue : ℚ) / (total : ℚ) * 100
  let striped_percentage := (striped : ℚ) / (total : ℚ) * 100
  blue_percentage - striped_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_nail_color_percentage_difference_l2918_291821


namespace NUMINAMATH_CALUDE_simplify_sqrt_neg_seven_squared_l2918_291809

theorem simplify_sqrt_neg_seven_squared : Real.sqrt ((-7)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_neg_seven_squared_l2918_291809


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2918_291896

/-- The value of m for a hyperbola with equation (y^2/16) - (x^2/9) = 1 and asymptotes y = ±mx -/
theorem hyperbola_asymptote_slope (m : ℝ) : m > 0 →
  (∀ x y : ℝ, y^2/16 - x^2/9 = 1 → (y = m*x ∨ y = -m*x)) → m = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2918_291896


namespace NUMINAMATH_CALUDE_min_values_theorem_l2918_291832

theorem min_values_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = 2 * m * n) :
  (m + n ≥ 2) ∧ (Real.sqrt (m * n) ≥ 1) ∧ (n^2 / m + m^2 / n ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_min_values_theorem_l2918_291832


namespace NUMINAMATH_CALUDE_square_mod_32_l2918_291849

theorem square_mod_32 (n : ℕ) (h : n % 8 = 6) : n^2 % 32 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_mod_32_l2918_291849


namespace NUMINAMATH_CALUDE_ice_cream_cost_is_two_l2918_291814

/-- The cost of a single topping in dollars -/
def topping_cost : ℚ := 1/2

/-- The number of toppings on the sundae -/
def num_toppings : ℕ := 10

/-- The total cost of the sundae in dollars -/
def sundae_cost : ℚ := 7

/-- The cost of the ice cream in dollars -/
def ice_cream_cost : ℚ := sundae_cost - num_toppings * topping_cost

theorem ice_cream_cost_is_two :
  ice_cream_cost = 2 :=
sorry

end NUMINAMATH_CALUDE_ice_cream_cost_is_two_l2918_291814


namespace NUMINAMATH_CALUDE_roots_equation_sum_l2918_291886

theorem roots_equation_sum (a b : ℝ) : 
  a^2 - 6*a + 8 = 0 → b^2 - 6*b + 8 = 0 → a^4 + b^4 + a^3*b + a*b^3 = 432 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_sum_l2918_291886


namespace NUMINAMATH_CALUDE_probability_even_product_excluding_13_l2918_291845

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def valid_integer (n : ℕ) : Prop :=
  4 ≤ n ∧ n ≤ 20 ∧ n ≠ 13

def count_valid_integers : ℕ := 16

def count_even_valid_integers : ℕ := 9

def count_odd_valid_integers : ℕ := 7

def total_combinations : ℕ := count_valid_integers.choose 2

def even_product_combinations : ℕ := 
  count_even_valid_integers.choose 2 + count_even_valid_integers * count_odd_valid_integers

theorem probability_even_product_excluding_13 :
  (even_product_combinations : ℚ) / total_combinations = 33 / 40 := by sorry

end NUMINAMATH_CALUDE_probability_even_product_excluding_13_l2918_291845


namespace NUMINAMATH_CALUDE_triangle_formation_l2918_291855

/-- Triangle inequality theorem check for three sides -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Check if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 5 10 13 ∧
  ¬can_form_triangle 1 2 3 ∧
  ¬can_form_triangle 4 5 10 ∧
  ∀ a : ℝ, a > 0 → ¬can_form_triangle (2*a) (3*a) (6*a) :=
by sorry


end NUMINAMATH_CALUDE_triangle_formation_l2918_291855


namespace NUMINAMATH_CALUDE_exists_infinite_subset_with_constant_gcd_l2918_291813

-- Define the set of natural numbers that are products of at most 1990 primes
def ProductOfLimitedPrimes (n : ℕ) : Prop :=
  ∃ (primes : Finset ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ primes.card ≤ 1990 ∧ n = primes.prod id

-- Define the property of A
def InfiniteSetOfLimitedPrimeProducts (A : Set ℕ) : Prop :=
  Set.Infinite A ∧ ∀ a ∈ A, ProductOfLimitedPrimes a

-- The main theorem
theorem exists_infinite_subset_with_constant_gcd
  (A : Set ℕ) (hA : InfiniteSetOfLimitedPrimeProducts A) :
  ∃ (B : Set ℕ) (k : ℕ), Set.Infinite B ∧ B ⊆ A ∧
    ∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → Nat.gcd x y = k :=
sorry

end NUMINAMATH_CALUDE_exists_infinite_subset_with_constant_gcd_l2918_291813


namespace NUMINAMATH_CALUDE_no_real_solution_log_equation_l2918_291824

theorem no_real_solution_log_equation :
  ¬∃ (x : ℝ), (Real.log (x + 4) + Real.log (x - 2) = Real.log (x^2 - 6*x + 8)) ∧ 
  (x + 4 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 6*x + 8 > 0) :=
sorry

end NUMINAMATH_CALUDE_no_real_solution_log_equation_l2918_291824


namespace NUMINAMATH_CALUDE_trinomial_binomial_product_l2918_291895

theorem trinomial_binomial_product : 
  ∀ x : ℝ, (2 * x^2 + 3 * x + 1) * (x - 4) = 2 * x^3 - 5 * x^2 - 11 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_binomial_product_l2918_291895


namespace NUMINAMATH_CALUDE_simplify_fraction_l2918_291825

theorem simplify_fraction (x y : ℝ) (hxy : x ≠ y) (hxy_neg : x ≠ -y) (hx : x ≠ 0) :
  (1 / (x - y) - 1 / (x + y)) / (x * y / (x^2 - y^2)) = 2 / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2918_291825


namespace NUMINAMATH_CALUDE_even_sum_probability_l2918_291852

def wheel1_sections : ℕ := 6
def wheel1_even_sections : ℕ := 2
def wheel1_odd_sections : ℕ := 4

def wheel2_sections : ℕ := 4
def wheel2_even_sections : ℕ := 1
def wheel2_odd_sections : ℕ := 3

theorem even_sum_probability :
  let p_even_sum := (wheel1_even_sections / wheel1_sections) * (wheel2_even_sections / wheel2_sections) +
                    (wheel1_odd_sections / wheel1_sections) * (wheel2_odd_sections / wheel2_sections)
  p_even_sum = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_probability_l2918_291852


namespace NUMINAMATH_CALUDE_sqrt_simplification_l2918_291810

theorem sqrt_simplification :
  (Real.sqrt 24 - Real.sqrt 2) - (Real.sqrt 8 + Real.sqrt 6) = Real.sqrt 6 - 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l2918_291810


namespace NUMINAMATH_CALUDE_minimum_cologne_drops_l2918_291836

theorem minimum_cologne_drops (f : ℕ) (n : ℕ) : 
  f > 0 →  -- number of boys is positive
  n > 0 →  -- number of drops is positive
  (∀ g : ℕ, g ≤ 4 → (3 * n : ℝ) ≥ (f * (n / 2 + 15) : ℝ)) →  -- no girl receives more than 3 bottles worth
  (f * ((n / 2 : ℝ) - 15) > (3 * n : ℝ)) →  -- mother receives more than any girl
  n ≥ 53 :=
by sorry

end NUMINAMATH_CALUDE_minimum_cologne_drops_l2918_291836


namespace NUMINAMATH_CALUDE_impossible_average_weight_problem_l2918_291851

theorem impossible_average_weight_problem :
  ¬ ∃ (n : ℕ), n > 0 ∧ (n * 55 + 50) / (n + 1) = 50 := by
  sorry

end NUMINAMATH_CALUDE_impossible_average_weight_problem_l2918_291851


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2918_291812

/-- The perimeter of a triangle with vertices A(3,7), B(-5,2), and C(3,2) is √89 + 13. -/
theorem triangle_perimeter : 
  let A : ℝ × ℝ := (3, 7)
  let B : ℝ × ℝ := (-5, 2)
  let C : ℝ × ℝ := (3, 2)
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B + d B C + d C A = Real.sqrt 89 + 13 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2918_291812


namespace NUMINAMATH_CALUDE_simplify_expression_l2918_291887

theorem simplify_expression : (3 / 4 : ℚ) * 60 - (8 / 5 : ℚ) * 60 + 63 = 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2918_291887


namespace NUMINAMATH_CALUDE_polynomial_value_l2918_291885

theorem polynomial_value (a b : ℝ) : 
  (a * 2^3 + b * 2 + 3 = 5) → 
  (a * (-2)^2 - 1/2 * b * (-2) - 3 = -2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_l2918_291885


namespace NUMINAMATH_CALUDE_digits_of_power_product_l2918_291859

theorem digits_of_power_product : 
  (Nat.log 10 (2^15 * 5^10) + 1 : ℕ) = 12 := by sorry

end NUMINAMATH_CALUDE_digits_of_power_product_l2918_291859


namespace NUMINAMATH_CALUDE_sin_2theta_plus_pi_4_l2918_291876

theorem sin_2theta_plus_pi_4 (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ + Real.pi / 4) = Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_plus_pi_4_l2918_291876


namespace NUMINAMATH_CALUDE_initial_ratio_proof_l2918_291866

/-- Represents the ratio of two quantities -/
structure Ratio :=
  (numerator : ℚ)
  (denominator : ℚ)

/-- Represents the contents of a bucket with two liquids -/
structure Bucket :=
  (liquidA : ℚ)
  (liquidB : ℚ)

def replace_mixture (b : Bucket) (amount : ℚ) : Bucket :=
  { liquidA := b.liquidA,
    liquidB := b.liquidB + amount }

def ratio (b : Bucket) : Ratio :=
  { numerator := b.liquidA,
    denominator := b.liquidB }

theorem initial_ratio_proof (initial : Bucket) 
  (h1 : initial.liquidA = 21)
  (h2 : ratio (replace_mixture initial 9) = Ratio.mk 7 9) :
  ratio initial = Ratio.mk 7 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_proof_l2918_291866


namespace NUMINAMATH_CALUDE_racing_track_width_l2918_291820

theorem racing_track_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) : 
  r₁ - r₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_racing_track_width_l2918_291820


namespace NUMINAMATH_CALUDE_equality_conditions_l2918_291833

theorem equality_conditions (a b c : ℝ) : 
  ((a + (a * b * c) / (a - b * c + b)) / (b + (a * b * c) / (a - a * c + b)) = 
   (a - (a * b) / (a + 2 * b)) / (b - (a * b) / (2 * a + b)) ∧
   (a - (a * b) / (a + 2 * b)) / (b - (a * b) / (2 * a + b)) = 
   ((2 * a * b) / (a - b) + a) / ((2 * a * b) / (a - b) - b) ∧
   ((2 * a * b) / (a - b) + a) / ((2 * a * b) / (a - b) - b) = a / b) ↔
  (a = 0 ∧ b ≠ 0 ∧ c ≠ 1) :=
by sorry


end NUMINAMATH_CALUDE_equality_conditions_l2918_291833


namespace NUMINAMATH_CALUDE_paint_cost_rectangular_floor_l2918_291868

/-- The cost to paint a rectangular floor given its length and the ratio of length to breadth -/
theorem paint_cost_rectangular_floor 
  (length : ℝ) 
  (length_to_breadth_ratio : ℝ) 
  (paint_rate : ℝ) 
  (h1 : length = 15.491933384829668)
  (h2 : length_to_breadth_ratio = 3)
  (h3 : paint_rate = 3) : 
  ⌊length * (length / length_to_breadth_ratio) * paint_rate⌋ = 240 := by
sorry

end NUMINAMATH_CALUDE_paint_cost_rectangular_floor_l2918_291868


namespace NUMINAMATH_CALUDE_koh_nh4i_reaction_l2918_291823

/-- Represents a chemical reaction with reactants and products -/
structure ChemicalReaction where
  reactants : List String
  products : List String
  ratio : Nat

/-- Represents the state of a chemical system -/
structure ChemicalSystem where
  compounds : List String
  moles : List ℚ

/-- Calculates the moles of products formed and remaining reactants -/
def reactComplete (reaction : ChemicalReaction) (initial : ChemicalSystem) : ChemicalSystem :=
  sorry

theorem koh_nh4i_reaction 
  (reaction : ChemicalReaction)
  (initial : ChemicalSystem)
  (h_reaction : reaction = 
    { reactants := ["KOH", "NH4I"]
    , products := ["KI", "NH3", "H2O"]
    , ratio := 1 })
  (h_initial : initial = 
    { compounds := ["KOH", "NH4I"]
    , moles := [3, 3] })
  : 
  let final := reactComplete reaction initial
  (final.compounds = ["KI", "NH3", "H2O", "KOH", "NH4I"] ∧
   final.moles = [3, 3, 3, 0, 0]) :=
by sorry

end NUMINAMATH_CALUDE_koh_nh4i_reaction_l2918_291823


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2918_291804

-- Define the radius of the cylinder
def cylinder_radius : ℝ := 2

-- Define the ratio between major and minor axes
def axis_ratio : ℝ := 1.25

-- Theorem statement
theorem ellipse_major_axis_length :
  let minor_axis : ℝ := 2 * cylinder_radius
  let major_axis : ℝ := minor_axis * axis_ratio
  major_axis = 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2918_291804


namespace NUMINAMATH_CALUDE_ohara_triple_49_64_l2918_291838

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ (Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x)

/-- Theorem: If (49, 64, x) is an O'Hara triple, then x = 15 -/
theorem ohara_triple_49_64 (x : ℕ) :
  is_ohara_triple 49 64 x → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_64_l2918_291838


namespace NUMINAMATH_CALUDE_range_of_a_l2918_291842

def A (a : ℝ) := {x : ℝ | 1 ≤ x ∧ x ≤ a}
def B (a : ℝ) := {y : ℝ | ∃ x ∈ A a, y = 5 * x - 6}
def C (a : ℝ) := {m : ℝ | ∃ x ∈ A a, m = x^2}

theorem range_of_a (a : ℝ) :
  (B a ∩ C a = C a) ↔ (2 ≤ a ∧ a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2918_291842


namespace NUMINAMATH_CALUDE_worker_speed_comparison_l2918_291858

/-- Given two workers A and B, this theorem proves that A is 3 times faster than B
    under the specified conditions. -/
theorem worker_speed_comparison 
  (work_rate_A : ℝ) 
  (work_rate_B : ℝ) 
  (total_work : ℝ) 
  (h1 : work_rate_A + work_rate_B = total_work / 24)
  (h2 : work_rate_A = total_work / 32) :
  work_rate_A = 3 * work_rate_B :=
sorry

end NUMINAMATH_CALUDE_worker_speed_comparison_l2918_291858


namespace NUMINAMATH_CALUDE_matrix_sum_proof_l2918_291890

theorem matrix_sum_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; -3, 7]
  A + B = !![-2, 5; -1, 12] := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_proof_l2918_291890


namespace NUMINAMATH_CALUDE_decreasing_power_function_l2918_291860

theorem decreasing_power_function (m : ℝ) : 
  (m^2 - 2*m - 2 = 1) ∧ (-4*m - 2 < 0) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_power_function_l2918_291860


namespace NUMINAMATH_CALUDE_exists_good_permutation_iff_power_of_two_l2918_291806

/-- A permutation is "good" if for any i < j < k, n doesn't divide (aᵢ + aₖ - 2aⱼ) -/
def is_good_permutation (n : ℕ) (a : Fin n → ℕ) : Prop :=
  ∀ i j k : Fin n, i < j → j < k → ¬(n ∣ a i + a k - 2 * a j)

/-- A natural number n ≥ 3 has a good permutation if and only if it's a power of 2 -/
theorem exists_good_permutation_iff_power_of_two (n : ℕ) (h : n ≥ 3) :
  (∃ (a : Fin n → ℕ), Function.Bijective a ∧ is_good_permutation n a) ↔ ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_exists_good_permutation_iff_power_of_two_l2918_291806


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l2918_291827

/-- The number of sides in a convex polygon where the sum of all angles except two is 3420 degrees. -/
def polygon_sides : ℕ := 22

/-- The sum of interior angles of a polygon with n sides. -/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- The sum of all angles except two in the polygon. -/
def given_angle_sum : ℝ := 3420

theorem convex_polygon_sides :
  ∃ (missing_angles : ℝ), 
    missing_angles ≥ 0 ∧ 
    missing_angles < 360 ∧
    interior_angle_sum polygon_sides = given_angle_sum + missing_angles := by
  sorry

#check convex_polygon_sides

end NUMINAMATH_CALUDE_convex_polygon_sides_l2918_291827


namespace NUMINAMATH_CALUDE_solution_systems_l2918_291864

-- System a
def system_a (x y : ℝ) : Prop :=
  x + y + x*y = 5 ∧ x*y*(x + y) = 6

-- System b
def system_b (x y : ℝ) : Prop :=
  x^3 + y^3 + 2*x*y = 4 ∧ x^2 - x*y + y^2 = 1

theorem solution_systems :
  (∃ x y : ℝ, system_a x y ∧ ((x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2))) ∧
  (∃ x y : ℝ, system_b x y ∧ x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_systems_l2918_291864


namespace NUMINAMATH_CALUDE_exists_non_grid_aligned_right_triangle_l2918_291857

/-- A triangle represented by its three vertices -/
structure Triangle where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ

/-- Check if a triangle is right-angled -/
def is_right_angled (t : Triangle) : Prop :=
  let ab := (t.b.1 - t.a.1, t.b.2 - t.a.2)
  let ac := (t.c.1 - t.a.1, t.c.2 - t.a.2)
  ab.1 * ac.1 + ab.2 * ac.2 = 0

/-- Check if a line segment is aligned with the grid -/
def is_grid_aligned (p1 p2 : ℤ × ℤ) : Prop :=
  p1.1 = p2.1 ∨ p1.2 = p2.2 ∨ (p2.2 - p1.2) * (p2.1 - p1.1) = 0

/-- The main theorem -/
theorem exists_non_grid_aligned_right_triangle :
  ∃ (t : Triangle),
    is_right_angled t ∧
    ¬is_grid_aligned t.a t.b ∧
    ¬is_grid_aligned t.b t.c ∧
    ¬is_grid_aligned t.c t.a :=
  sorry

end NUMINAMATH_CALUDE_exists_non_grid_aligned_right_triangle_l2918_291857


namespace NUMINAMATH_CALUDE_jessica_remaining_seashells_l2918_291831

/-- The number of seashells Jessica initially found -/
def initial_seashells : ℕ := 8

/-- The number of seashells Jessica gave to Joan -/
def given_seashells : ℕ := 6

/-- The number of seashells Jessica is left with -/
def remaining_seashells : ℕ := initial_seashells - given_seashells

theorem jessica_remaining_seashells : remaining_seashells = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_remaining_seashells_l2918_291831


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l2918_291865

/-- The number of sides in a regular pentadecagon -/
def n : ℕ := 15

/-- The total number of triangles that can be formed using any three vertices of a regular pentadecagon -/
def total_triangles : ℕ := n.choose 3

/-- The number of triangles formed by three consecutive vertices in a regular pentadecagon -/
def consecutive_triangles : ℕ := n

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon, 
    where no triangle is formed by three consecutive vertices -/
def valid_triangles : ℕ := total_triangles - consecutive_triangles

theorem pentadecagon_triangles : valid_triangles = 440 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l2918_291865


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2918_291880

theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  (1 + 1 / (x + 1)) / ((x^2 + 4*x + 4) / (x + 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2918_291880


namespace NUMINAMATH_CALUDE_sunzi_carriage_problem_l2918_291834

/-- 
Given a number of carriages and people satisfying the conditions from 
"The Mathematical Classic of Sunzi", prove that the number of carriages 
satisfies the equation 3(x-2) = 2x + 9.
-/
theorem sunzi_carriage_problem (x : ℕ) (people : ℕ) :
  (3 * (x - 2) = people) →  -- Three people per carriage, two empty
  (2 * x + 9 = people) →    -- Two people per carriage, nine walking
  3 * (x - 2) = 2 * x + 9 := by
sorry

end NUMINAMATH_CALUDE_sunzi_carriage_problem_l2918_291834


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2918_291815

theorem inequality_equivalence (x : ℝ) : 
  3/16 + |x - 17/64| < 7/32 ↔ 15/64 < x ∧ x < 19/64 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2918_291815


namespace NUMINAMATH_CALUDE_mildred_oranges_l2918_291872

def oranges_problem (initial_oranges father_multiplier sister_takes mother_multiplier : ℕ) : ℕ :=
  let after_father := initial_oranges + father_multiplier * initial_oranges
  let after_sister := after_father - sister_takes
  mother_multiplier * after_sister

theorem mildred_oranges :
  oranges_problem 215 3 174 2 = 1372 := by sorry

end NUMINAMATH_CALUDE_mildred_oranges_l2918_291872


namespace NUMINAMATH_CALUDE_sales_solution_l2918_291882

def sales_problem (month1 month3 month4 month5 month6 average : ℕ) : Prop :=
  let total_sales := average * 6
  let known_sales := month1 + month3 + month4 + month5 + month6
  let month2 := total_sales - known_sales
  month2 = 11860

theorem sales_solution :
  sales_problem 5420 6350 6500 6200 8270 6400 := by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l2918_291882


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equivalence_l2918_291828

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def is_arithmetic_seq (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

theorem arithmetic_sequence_equivalence
  (a b c : ℕ → ℝ)
  (h1 : ∀ n : ℕ, b n = a n - a (n + 2))
  (h2 : ∀ n : ℕ, c n = a n + 2 * a (n + 1) + 3 * a (n + 2)) :
  is_arithmetic_seq a ↔ is_arithmetic_seq c ∧ (∀ n : ℕ, b n ≤ b (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equivalence_l2918_291828


namespace NUMINAMATH_CALUDE_cube_inequality_l2918_291850

theorem cube_inequality (a b : ℝ) : a < b → a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l2918_291850


namespace NUMINAMATH_CALUDE_initial_staples_count_l2918_291878

/-- The number of staples used per report -/
def staples_per_report : ℕ := 1

/-- The number of reports in a dozen -/
def reports_per_dozen : ℕ := 12

/-- The number of dozens of reports Stacie staples -/
def dozens_of_reports : ℕ := 3

/-- The number of staples remaining after stapling -/
def remaining_staples : ℕ := 14

/-- Theorem: The initial number of staples in the stapler is 50 -/
theorem initial_staples_count : 
  dozens_of_reports * reports_per_dozen * staples_per_report + remaining_staples = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_staples_count_l2918_291878


namespace NUMINAMATH_CALUDE_comparison_theorems_l2918_291839

theorem comparison_theorems :
  (∀ a b : ℝ, a - b = 4 → a > b) ∧
  (∀ a b : ℝ, a - b = -2 → a < b) ∧
  (∀ x : ℝ, x > 0 → -x + 5 > -2*x + 4) ∧
  (∀ x y : ℝ, 
    (y > x → 5*x + 13*y + 2 > 6*x + 12*y + 2) ∧
    (y = x → 5*x + 13*y + 2 = 6*x + 12*y + 2) ∧
    (y < x → 5*x + 13*y + 2 < 6*x + 12*y + 2)) :=
by sorry

end NUMINAMATH_CALUDE_comparison_theorems_l2918_291839


namespace NUMINAMATH_CALUDE_photography_preference_l2918_291818

-- Define the number of students who dislike photography
variable (x : ℕ)

-- Define the total number of students in the class
def total : ℕ := 9 * x

-- Define the number of students who like photography
def like : ℕ := 5 * x

-- Define the number of students who are neutral towards photography
def neutral : ℕ := x + 12

-- Theorem statement
theorem photography_preference (x : ℕ) :
  like x = (total x / 2) + 3 := by
  sorry

end NUMINAMATH_CALUDE_photography_preference_l2918_291818


namespace NUMINAMATH_CALUDE_renovation_cost_effectiveness_l2918_291883

def turnkey_cost : ℕ := 50000
def material_cost : ℕ := 20000
def husband_wage : ℕ := 2000
def wife_wage : ℕ := 1500

def max_renovation_days : ℕ := 8

theorem renovation_cost_effectiveness :
  max_renovation_days = 
    (turnkey_cost - material_cost) / (husband_wage + wife_wage) :=
by sorry

end NUMINAMATH_CALUDE_renovation_cost_effectiveness_l2918_291883


namespace NUMINAMATH_CALUDE_gorilla_exhibit_percentage_is_80_l2918_291881

-- Define the given parameters
def visitors_per_hour : ℕ := 50
def open_hours : ℕ := 8
def gorilla_exhibit_visitors : ℕ := 320

-- Define the total number of visitors
def total_visitors : ℕ := visitors_per_hour * open_hours

-- Define the percentage of visitors going to the gorilla exhibit
def gorilla_exhibit_percentage : ℚ := (gorilla_exhibit_visitors : ℚ) / (total_visitors : ℚ) * 100

-- Theorem statement
theorem gorilla_exhibit_percentage_is_80 : 
  gorilla_exhibit_percentage = 80 := by sorry

end NUMINAMATH_CALUDE_gorilla_exhibit_percentage_is_80_l2918_291881


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l2918_291863

/-- Given a line segment with midpoint (3, 1) and one endpoint (7, -3), prove that the other endpoint is (-1, 5) -/
theorem other_endpoint_of_line_segment (x₂ y₂ : ℚ) : 
  (3 = (7 + x₂) / 2) ∧ (1 = (-3 + y₂) / 2) → (x₂ = -1 ∧ y₂ = 5) := by
sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l2918_291863


namespace NUMINAMATH_CALUDE_khali_snow_volume_l2918_291889

/-- The volume of snow on a rectangular sidewalk -/
def snow_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of snow on Khali's sidewalk is 20 cubic feet -/
theorem khali_snow_volume :
  snow_volume 20 2 (1/2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_khali_snow_volume_l2918_291889


namespace NUMINAMATH_CALUDE_number_of_valid_arrangements_l2918_291888

/-- Represents the number of students -/
def n : ℕ := 5

/-- Represents the number of ways to arrange n students without restrictions -/
def total_arrangements (n : ℕ) : ℕ := n.factorial

/-- Represents the number of ways to arrange n-1 students (excluding the one at the end) -/
def interior_arrangements (n : ℕ) : ℕ := (n-1).factorial

/-- Represents the number of choices for the student who must stand at either end -/
def end_choices : ℕ := 2

/-- Represents the number of ways to arrange n students with one at either end -/
def arrangements_with_end_restriction (n : ℕ) : ℕ :=
  end_choices * interior_arrangements n

/-- Represents the number of ways to arrange n students where two specific students are together -/
def arrangements_with_pair_together (n : ℕ) : ℕ :=
  end_choices * (n-2) * 2 * 2

/-- The main theorem stating the number of valid arrangements -/
theorem number_of_valid_arrangements :
  arrangements_with_end_restriction n - arrangements_with_pair_together n = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_arrangements_l2918_291888


namespace NUMINAMATH_CALUDE_split_bill_proof_l2918_291803

def num_friends : ℕ := 5
def num_hamburgers : ℕ := 5
def price_hamburger : ℚ := 3
def num_fries : ℕ := 4
def price_fries : ℚ := 1.20
def num_soda : ℕ := 5
def price_soda : ℚ := 0.50
def price_spaghetti : ℚ := 2.70

theorem split_bill_proof :
  let total_bill := num_hamburgers * price_hamburger +
                    num_fries * price_fries +
                    num_soda * price_soda +
                    price_spaghetti
  (total_bill / num_friends : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_split_bill_proof_l2918_291803


namespace NUMINAMATH_CALUDE_no_valid_n_l2918_291871

theorem no_valid_n : ¬∃ (n : ℕ), 
  n > 0 ∧ 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
sorry

end NUMINAMATH_CALUDE_no_valid_n_l2918_291871


namespace NUMINAMATH_CALUDE_longest_side_range_l2918_291826

-- Define an obtuse triangle
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_obtuse : ∃ angle, angle > π/2 ∧ angle < π

-- Theorem statement
theorem longest_side_range (triangle : ObtuseTriangle) 
  (ha : triangle.a = 1) 
  (hb : triangle.b = 2) : 
  (Real.sqrt 5 < triangle.c ∧ triangle.c < 3) ∨ triangle.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_range_l2918_291826


namespace NUMINAMATH_CALUDE_total_students_correct_l2918_291856

/-- Represents the total number of high school students -/
def total_students : ℕ := 1800

/-- Represents the sample size -/
def sample_size : ℕ := 45

/-- Represents the number of second-year students -/
def second_year_students : ℕ := 600

/-- Represents the number of second-year students selected in the sample -/
def selected_second_year : ℕ := 15

/-- Theorem stating that the total number of students is correct given the sampling information -/
theorem total_students_correct :
  (total_students : ℚ) / sample_size = (second_year_students : ℚ) / selected_second_year :=
sorry

end NUMINAMATH_CALUDE_total_students_correct_l2918_291856


namespace NUMINAMATH_CALUDE_sqrt_two_squared_inverse_l2918_291800

theorem sqrt_two_squared_inverse : ((-Real.sqrt 2)^2)⁻¹ = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_inverse_l2918_291800


namespace NUMINAMATH_CALUDE_corn_syrup_amount_l2918_291873

/-- Represents the ratio of ingredients in a drink formulation -/
structure Ratio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- Represents a drink formulation -/
structure Formulation :=
  (ratio : Ratio)
  (water_amount : ℚ)

def standard_ratio : Ratio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

def sport_ratio (r : Ratio) : Ratio :=
  { flavoring := r.flavoring,
    corn_syrup := r.corn_syrup / 3,
    water := r.water * 2 }

def sport_formulation : Formulation :=
  { ratio := sport_ratio standard_ratio,
    water_amount := 120 }

theorem corn_syrup_amount :
  (sport_formulation.ratio.corn_syrup / sport_formulation.ratio.water) *
    sport_formulation.water_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_corn_syrup_amount_l2918_291873


namespace NUMINAMATH_CALUDE_cake_serving_capacity_l2918_291840

-- Define the original cake properties
def original_radius : ℝ := 20
def original_people_served : ℕ := 4

-- Define the new cake radius
def new_radius : ℝ := 50

-- Theorem statement
theorem cake_serving_capacity :
  ∃ (new_people_served : ℕ), 
    new_people_served = 25 ∧
    (new_radius^2 / original_radius^2) * original_people_served = new_people_served :=
by
  sorry

end NUMINAMATH_CALUDE_cake_serving_capacity_l2918_291840


namespace NUMINAMATH_CALUDE_student_sister_weight_ratio_l2918_291867

theorem student_sister_weight_ratio : 
  ∀ (student_weight sister_weight : ℝ),
    student_weight = 90 →
    student_weight + sister_weight = 132 →
    (student_weight - 6) / sister_weight = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_student_sister_weight_ratio_l2918_291867


namespace NUMINAMATH_CALUDE_triple_composition_even_l2918_291848

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem triple_composition_even (f : ℝ → ℝ) (h : IsEven f) :
  ∀ x, f (f (f (-x))) = f (f (f x)) := by sorry

end NUMINAMATH_CALUDE_triple_composition_even_l2918_291848


namespace NUMINAMATH_CALUDE_advertisement_length_main_theorem_l2918_291819

/-- Proves that the advertisement length is 20 minutes given the movie theater conditions -/
theorem advertisement_length : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (movie_length : ℕ) (replays : ℕ) (operating_time : ℕ) (ad_length : ℕ) =>
    movie_length = 90 ∧ 
    replays = 6 ∧ 
    operating_time = 660 ∧
    movie_length * replays + ad_length * replays = operating_time →
    ad_length = 20

/-- The main theorem stating the advertisement length -/
theorem main_theorem : advertisement_length 90 6 660 20 := by
  sorry

end NUMINAMATH_CALUDE_advertisement_length_main_theorem_l2918_291819


namespace NUMINAMATH_CALUDE_equation_roots_and_m_values_l2918_291870

-- Define the equation
def equation (x m : ℝ) : Prop := (x + m)^2 - 4 = 0

-- Theorem statement
theorem equation_roots_and_m_values :
  -- For all real m
  ∀ m : ℝ,
  -- There exist two distinct real roots
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ m ∧ equation x₂ m ∧
  -- If the roots p and q satisfy pq = p + q
  (∀ p q : ℝ, equation p m → equation q m → p * q = p + q →
  -- Then m equals one of these two values
  (m = Real.sqrt 5 - 1 ∨ m = -Real.sqrt 5 - 1)) :=
sorry

end NUMINAMATH_CALUDE_equation_roots_and_m_values_l2918_291870


namespace NUMINAMATH_CALUDE_fraction_equality_l2918_291808

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 2) 
  (h2 : s / u = 11 / 7) : 
  (5 * p * s - 3 * q * u) / (7 * q * u - 2 * p * s) = -233 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2918_291808


namespace NUMINAMATH_CALUDE_revenue_loss_l2918_291801

/-- Represents the types of tickets sold in the theater. -/
inductive TicketType
  | GeneralRegular
  | GeneralVIP
  | ChildRegular
  | ChildVIP
  | SeniorRegular
  | SeniorVIP
  | VeteranRegular
  | VeteranVIP

/-- Calculates the revenue for a given ticket type. -/
def ticketRevenue (t : TicketType) : ℚ :=
  match t with
  | .GeneralRegular => 10
  | .GeneralVIP => 15
  | .ChildRegular => 6
  | .ChildVIP => 11
  | .SeniorRegular => 8
  | .SeniorVIP => 13
  | .VeteranRegular => 8
  | .VeteranVIP => 13

/-- Represents the theater's seating and pricing structure. -/
structure Theater where
  regularSeats : ℕ
  vipSeats : ℕ
  regularPrice : ℚ
  vipSurcharge : ℚ

/-- Calculates the potential revenue if all seats were sold at full price. -/
def potentialRevenue (t : Theater) : ℚ :=
  t.regularSeats * t.regularPrice + t.vipSeats * (t.regularPrice + t.vipSurcharge)

/-- Represents the actual sales for the night. -/
structure ActualSales where
  generalRegular : ℕ
  generalVIP : ℕ
  childRegular : ℕ
  childVIP : ℕ
  seniorRegular : ℕ
  seniorVIP : ℕ
  veteranRegular : ℕ
  veteranVIP : ℕ

/-- Calculates the actual revenue from the given sales. -/
def actualRevenue (s : ActualSales) : ℚ :=
  s.generalRegular * ticketRevenue .GeneralRegular +
  s.generalVIP * ticketRevenue .GeneralVIP +
  s.childRegular * ticketRevenue .ChildRegular +
  s.childVIP * ticketRevenue .ChildVIP +
  s.seniorRegular * ticketRevenue .SeniorRegular +
  s.seniorVIP * ticketRevenue .SeniorVIP +
  s.veteranRegular * ticketRevenue .VeteranRegular +
  s.veteranVIP * ticketRevenue .VeteranVIP

theorem revenue_loss (t : Theater) (s : ActualSales) :
    t.regularSeats = 40 ∧
    t.vipSeats = 10 ∧
    t.regularPrice = 10 ∧
    t.vipSurcharge = 5 ∧
    s.generalRegular = 12 ∧
    s.generalVIP = 6 ∧
    s.childRegular = 3 ∧
    s.childVIP = 1 ∧
    s.seniorRegular = 4 ∧
    s.seniorVIP = 2 ∧
    s.veteranRegular = 2 ∧
    s.veteranVIP = 1 →
    potentialRevenue t - actualRevenue s = 224 := by
  sorry

#eval potentialRevenue { regularSeats := 40, vipSeats := 10, regularPrice := 10, vipSurcharge := 5 }
#eval actualRevenue { generalRegular := 12, generalVIP := 6, childRegular := 3, childVIP := 1,
                      seniorRegular := 4, seniorVIP := 2, veteranRegular := 2, veteranVIP := 1 }

end NUMINAMATH_CALUDE_revenue_loss_l2918_291801
